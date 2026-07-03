//! Sub-stepping temporal Gauss-Seidel solver primitives with impulse
//! warm-starting and connected-component islands.
//!
//! This module provides trait-abstracted building blocks that layer on
//! top of any concrete body/contact representation. The design is
//! independent of the existing PGS-style solver in [`crate::solver`]
//! so that both can coexist during a gradual migration.
//!
//! The three techniques implemented here are standard textbook
//! rigid-body dynamics:
//!
//! * **Impulse warm-starting** — the applied impulse from the previous
//!   frame is stored per stable contact ID and re-applied as the
//!   initial guess of the next frame's velocity iteration, which
//!   collapses convergence for stationary pile-ups.
//! * **Sub-stepping TGS** — a physics step is split into `N` equal
//!   sub-steps. Each sub-step runs a small number of velocity
//!   iterations followed by a positional relaxation pass. Splitting
//!   the frame keeps high-mass-ratio stacks stable without paying for
//!   `N × velocity_iters` global iterations.
//! * **Connected-component islands** — bodies coupled through
//!   contacts or joints are grouped into disjoint islands with a
//!   union-find pass. Independent islands can be solved without any
//!   cross-communication, which unlocks deterministic per-island
//!   parallelism (dispatched by consumers of this module, e.g. via
//!   `rayon`).
//!
//! The trait shape (`BodyLike`, `ContactLike`, `JointLike`) is
//! deliberately narrow so that adapters can be written for the crate's
//! existing [`RigidBody`](crate::solver::RigidBody) /
//! [`ContactConstraint`](crate::solver::ContactConstraint) types
//! without leaking their internals into this module.

use crate::math::Fix128;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trait shape
// ---------------------------------------------------------------------------

/// Minimum body surface the sub-stepping solver needs. Adapters over
/// concrete rigid-body types implement this on the outside.
pub trait BodyLike {
    /// Stable identity used for warm-start indexing and Kalman-style
    /// caches. Must be stable across a full physics tick.
    fn stable_id(&self) -> u64;
    /// `true` for bodies that participate in velocity/position updates.
    /// Static or kinematic bodies return `false` and are skipped by the
    /// island builder as sinks (they connect neighbours but do not
    /// spread).
    fn is_dynamic(&self) -> bool;
}

/// A pairwise contact between two bodies.
pub trait ContactLike {
    /// Index into the body slice for one side of the contact.
    fn body_a(&self) -> usize;
    /// Index into the body slice for the other side.
    fn body_b(&self) -> usize;
    /// Stable identity of the contact point. Used to look up the
    /// previously applied impulse for warm-starting; must survive
    /// re-detection across frames when the same feature pair is in
    /// contact (features can be encoded via hashing of feature IDs).
    fn stable_id(&self) -> u64;
}

/// A bilateral joint (distance, revolute, prismatic …) between two
/// bodies. Only the coupling for island detection is required here.
pub trait JointLike {
    fn body_a(&self) -> usize;
    fn body_b(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Impulse cache (warm-start memory)
// ---------------------------------------------------------------------------

/// Impulse applied on a contact during a frame, broken down into the
/// contact-normal component and the two tangential (friction)
/// components. All values are in the contact frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CachedImpulse {
    /// Normal impulse magnitude (non-negative in a healthy solve).
    pub normal: Fix128,
    /// First tangential impulse magnitude.
    pub tangent1: Fix128,
    /// Second tangential impulse magnitude.
    pub tangent2: Fix128,
}

/// Per-contact impulse memory carried across frames.
///
/// The cache is keyed by [`ContactLike::stable_id`]. Entries that are
/// not touched during a physics step are dropped by [`Self::sweep`] so
/// that stale IDs do not grow the map indefinitely.
#[derive(Debug, Default, Clone)]
pub struct ImpulseCache {
    entries: HashMap<u64, CachedImpulse>,
    // Bit set of IDs touched during the current tick.
    live: HashMap<u64, ()>,
}

impl ImpulseCache {
    /// Creates an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retrieves the previously applied impulse for `contact_id`, or
    /// zero if the contact is new. The read also marks the entry as
    /// alive for the current tick so that [`Self::sweep`] preserves it.
    #[must_use]
    pub fn take(&mut self, contact_id: u64) -> CachedImpulse {
        self.live.insert(contact_id, ());
        self.entries.get(&contact_id).copied().unwrap_or_default()
    }

    /// Retrieves without marking as alive (peek). Useful for
    /// diagnostics; regular solvers should call [`Self::take`].
    #[must_use]
    pub fn peek(&self, contact_id: u64) -> CachedImpulse {
        self.entries.get(&contact_id).copied().unwrap_or_default()
    }

    /// Stores the applied impulse for `contact_id`. The entry is kept
    /// only as long as some subsequent [`Self::take`] call touches it
    /// before the next [`Self::sweep`]; setter calls do not implicitly
    /// mark the entry as alive so that stale contacts are pruned by
    /// the very next tick's sweep.
    pub fn set(&mut self, contact_id: u64, imp: CachedImpulse) {
        self.entries.insert(contact_id, imp);
    }

    /// Removes any entries that were not touched via [`Self::take`] or
    /// [`Self::set`] during the current tick. Call this once per
    /// physics step after all contacts have been visited.
    pub fn sweep(&mut self) {
        self.entries.retain(|k, _| self.live.contains_key(k));
        self.live.clear();
    }

    /// Number of impulses currently cached.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no impulses are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drops all cached impulses.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.live.clear();
    }
}

// ---------------------------------------------------------------------------
// Union-find (path compression + rank union)
// ---------------------------------------------------------------------------

/// Disjoint-set data structure with path compression and union by
/// rank. Used to group bodies into islands.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    /// Creates a forest of `n` singleton sets.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Returns the representative of the set containing `i`. Amortised
    /// α(n) via path compression.
    pub fn find(&mut self, i: usize) -> usize {
        let mut root = i;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression: point every visited node directly at root.
        let mut cur = i;
        while self.parent[cur] != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Unions the sets containing `i` and `j`. Returns `true` when a
    /// merge actually happened.
    pub fn union(&mut self, i: usize, j: usize) -> bool {
        let ri = self.find(i);
        let rj = self.find(j);
        if ri == rj {
            return false;
        }
        // Union by rank: hang the shorter tree under the taller one.
        if self.rank[ri] < self.rank[rj] {
            self.parent[ri] = rj;
        } else if self.rank[ri] > self.rank[rj] {
            self.parent[rj] = ri;
        } else {
            self.parent[rj] = ri;
            self.rank[ri] += 1;
        }
        true
    }

    /// Number of elements in the forest.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// `true` when the forest is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Island detection
// ---------------------------------------------------------------------------

/// A connected component of bodies plus the contacts and joints that
/// bind them. The three vectors are sorted ascending by index so that
/// downstream dispatch (e.g. `rayon::par_iter`) walks the same order
/// on every run, preserving determinism.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Island {
    pub bodies: Vec<usize>,
    pub contacts: Vec<usize>,
    pub joints: Vec<usize>,
}

/// Groups bodies into islands using contacts and joints as edges.
/// Static bodies (`is_dynamic == false`) act as separators: they are
/// listed in every island that touches them but do not fuse islands
/// together.
///
/// # Panics
/// Panics when any contact or joint references a body index outside of
/// `bodies`.
#[must_use]
pub fn build_islands<B: BodyLike, C: ContactLike, J: JointLike>(
    bodies: &[B],
    contacts: &[C],
    joints: &[J],
) -> Vec<Island> {
    let n = bodies.len();
    let mut uf = UnionFind::new(n);

    // Union dynamic pairs. Static bodies stay in their own set so they
    // do not glue moving stacks together across the world.
    for c in contacts {
        let (a, b) = (c.body_a(), c.body_b());
        assert!(a < n && b < n, "contact references body out of bounds");
        if bodies[a].is_dynamic() && bodies[b].is_dynamic() {
            uf.union(a, b);
        }
    }
    for j in joints {
        let (a, b) = (j.body_a(), j.body_b());
        assert!(a < n && b < n, "joint references body out of bounds");
        if bodies[a].is_dynamic() && bodies[b].is_dynamic() {
            uf.union(a, b);
        }
    }

    // Bucket bodies by representative. Static bodies are added to
    // every island that references them below.
    let mut buckets: HashMap<usize, Island> = HashMap::new();
    for (i, body) in bodies.iter().enumerate() {
        if body.is_dynamic() {
            let root = uf.find(i);
            buckets.entry(root).or_default().bodies.push(i);
        }
    }

    // Attach contacts and joints, and copy static neighbours into
    // whichever island touches them so that the constraint has both
    // sides available during solve.
    for (idx, c) in contacts.iter().enumerate() {
        let (a, b) = (c.body_a(), c.body_b());
        let root = if bodies[a].is_dynamic() {
            uf.find(a)
        } else if bodies[b].is_dynamic() {
            uf.find(b)
        } else {
            continue; // static-static contact: nothing to solve.
        };
        let island = buckets.entry(root).or_default();
        island.contacts.push(idx);
        if !bodies[a].is_dynamic() && !island.bodies.contains(&a) {
            island.bodies.push(a);
        }
        if !bodies[b].is_dynamic() && !island.bodies.contains(&b) {
            island.bodies.push(b);
        }
    }
    for (idx, j) in joints.iter().enumerate() {
        let (a, b) = (j.body_a(), j.body_b());
        let root = if bodies[a].is_dynamic() {
            uf.find(a)
        } else if bodies[b].is_dynamic() {
            uf.find(b)
        } else {
            continue;
        };
        let island = buckets.entry(root).or_default();
        island.joints.push(idx);
        if !bodies[a].is_dynamic() && !island.bodies.contains(&a) {
            island.bodies.push(a);
        }
        if !bodies[b].is_dynamic() && !island.bodies.contains(&b) {
            island.bodies.push(b);
        }
    }

    // Sort each vector so consumers walk in the same order on every
    // machine (determinism), then collect islands in a canonical order.
    let mut islands: Vec<Island> = buckets.into_values().collect();
    for island in &mut islands {
        island.bodies.sort_unstable();
        island.contacts.sort_unstable();
        island.joints.sort_unstable();
    }
    islands.sort_by_key(|i| i.bodies.first().copied().unwrap_or(usize::MAX));
    islands
}

// ---------------------------------------------------------------------------
// Sub-stepping TGS configuration + driver
// ---------------------------------------------------------------------------

/// Sub-stepping TGS parameters. The defaults match the "small
/// substeps, few iterations" recipe recommended by recent rigid-body
/// literature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TgsConfig {
    /// Number of equal-sized sub-steps per physics tick.
    pub substeps: u32,
    /// Velocity iterations per sub-step.
    pub velocity_iters: u32,
    /// Positional relaxation iterations per sub-step.
    pub position_iters: u32,
    /// When `true`, the impulse cache is consulted at the start of
    /// each velocity phase to seed the constraint impulses.
    pub warmstart: bool,
}

impl Default for TgsConfig {
    fn default() -> Self {
        Self {
            substeps: 4,
            velocity_iters: 4,
            position_iters: 2,
            warmstart: true,
        }
    }
}

/// Callbacks that drive one velocity or position iteration inside the
/// sub-stepping loop. Consumers wire these up to their concrete solver
/// so that the traversal (substeps, warm-start, sweep) stays here
/// while the per-constraint math stays outside.
pub trait TgsHooks {
    /// Called once at the beginning of every sub-step. Typical
    /// implementations apply gravity and integrate velocities forward
    /// by `sub_dt`.
    fn begin_substep(&mut self, sub_dt: Fix128);

    /// Called once per velocity iteration inside a sub-step. The
    /// implementation walks the island's contacts and joints and
    /// applies impulses.
    fn velocity_iteration(&mut self, sub_dt: Fix128);

    /// Called once per position iteration inside a sub-step. The
    /// implementation walks the island's contacts and joints and
    /// applies positional correction.
    fn position_iteration(&mut self, sub_dt: Fix128);

    /// Called once at the end of every sub-step. Typical
    /// implementations integrate positions using the just-solved
    /// velocities.
    fn end_substep(&mut self, sub_dt: Fix128);
}

/// Runs the sub-stepping loop for a single island. Splits the frame
/// `dt` into `cfg.substeps` equal sub-steps and, for each sub-step,
/// runs `velocity_iters` velocity iterations followed by
/// `position_iters` positional iterations. Warm-starting decisions are
/// left to the hook implementation (the module publishes the
/// [`ImpulseCache`] type that hooks are expected to consult when
/// `cfg.warmstart` is `true`).
///
/// # Panics
/// Panics when `cfg.substeps == 0`.
pub fn tgs_step<H: TgsHooks>(hooks: &mut H, cfg: &TgsConfig, dt: Fix128) {
    assert!(cfg.substeps > 0, "TgsConfig::substeps must be positive");
    let inv = Fix128::from_f32(1.0 / cfg.substeps as f32);
    let sub_dt = dt * inv;
    for _ in 0..cfg.substeps {
        hooks.begin_substep(sub_dt);
        for _ in 0..cfg.velocity_iters {
            hooks.velocity_iteration(sub_dt);
        }
        for _ in 0..cfg.position_iters {
            hooks.position_iteration(sub_dt);
        }
        hooks.end_substep(sub_dt);
    }
}

// ---------------------------------------------------------------------------
// Adapter to the existing solver types
// ---------------------------------------------------------------------------
//
// The wrappers below let this module operate on the crate's concrete
// [`RigidBody`], [`ContactConstraint`] and [`DistanceConstraint`]
// without a hard coupling to their internal layout. Stable IDs are
// externally injected so that callers control warm-start persistence
// (typically a `HashMap<BodyHandle, u64>` maintained by the world).

use crate::solver::{BodyType, ContactConstraint, DistanceConstraint, RigidBody};

/// Borrowed view of a [`RigidBody`] paired with an externally-provided
/// stable identifier. Frame-to-frame persistence of the ID is the
/// caller's responsibility.
pub struct BodyRef<'a> {
    pub body: &'a RigidBody,
    pub id: u64,
}

impl BodyLike for BodyRef<'_> {
    fn stable_id(&self) -> u64 {
        self.id
    }
    fn is_dynamic(&self) -> bool {
        matches!(self.body.body_type, BodyType::Dynamic)
    }
}

/// Borrowed view of a [`ContactConstraint`] plus its stable ID.
pub struct ContactRef<'a> {
    pub contact: &'a ContactConstraint,
    pub id: u64,
}

impl ContactLike for ContactRef<'_> {
    fn body_a(&self) -> usize {
        self.contact.body_a
    }
    fn body_b(&self) -> usize {
        self.contact.body_b
    }
    fn stable_id(&self) -> u64 {
        self.id
    }
}

/// Borrowed view of a [`DistanceConstraint`] as a bilateral joint.
pub struct DistanceRef<'a> {
    pub joint: &'a DistanceConstraint,
}

impl JointLike for DistanceRef<'_> {
    fn body_a(&self) -> usize {
        self.joint.body_a
    }
    fn body_b(&self) -> usize {
        self.joint.body_b
    }
}

// ---------------------------------------------------------------------------
// Per-island dispatch (rayon, feature-gated)
// ---------------------------------------------------------------------------

/// Serially dispatches `f` over each island in canonical order.
///
/// Available even without the `parallel` feature so that consumers can
/// share a single call-site regardless of build configuration.
pub fn dispatch_islands<F>(islands: &[Island], mut f: F)
where
    F: FnMut(&Island),
{
    for island in islands {
        f(island);
    }
}

/// Dispatches `f` over each island in parallel via `rayon`. Islands
/// are independent by construction (see [`build_islands`]), so no
/// inter-island synchronisation is required. Determinism of the
/// aggregate result is preserved as long as `f` mutates only state
/// belonging to bodies inside the island passed to it.
///
/// Available only with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn par_dispatch_islands<F>(islands: &[Island], f: F)
where
    F: Fn(&Island) + Send + Sync,
{
    use rayon::prelude::*;
    islands.par_iter().for_each(f);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Mock BodyLike / ContactLike / JointLike --------------------------

    struct MockBody {
        id: u64,
        dynamic: bool,
    }
    impl BodyLike for MockBody {
        fn stable_id(&self) -> u64 {
            self.id
        }
        fn is_dynamic(&self) -> bool {
            self.dynamic
        }
    }

    struct MockContact {
        id: u64,
        a: usize,
        b: usize,
    }
    impl ContactLike for MockContact {
        fn body_a(&self) -> usize {
            self.a
        }
        fn body_b(&self) -> usize {
            self.b
        }
        fn stable_id(&self) -> u64 {
            self.id
        }
    }

    struct MockJoint {
        a: usize,
        b: usize,
    }
    impl JointLike for MockJoint {
        fn body_a(&self) -> usize {
            self.a
        }
        fn body_b(&self) -> usize {
            self.b
        }
    }

    // --- ImpulseCache -----------------------------------------------------

    #[test]
    fn impulse_cache_returns_zero_for_new_contact() {
        let mut cache = ImpulseCache::new();
        let got = cache.take(42);
        assert_eq!(got, CachedImpulse::default());
    }

    #[test]
    fn impulse_cache_round_trips_stored_impulse() {
        let mut cache = ImpulseCache::new();
        let imp = CachedImpulse {
            normal: Fix128::from_f32(1.5),
            tangent1: Fix128::from_f32(-0.25),
            tangent2: Fix128::from_f32(0.0),
        };
        cache.set(7, imp);
        assert_eq!(cache.take(7), imp);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn impulse_cache_sweep_drops_untouched_entries() {
        let mut cache = ImpulseCache::new();
        cache.set(1, CachedImpulse::default());
        cache.set(2, CachedImpulse::default());
        // Touch only #1 this tick.
        let _ = cache.take(1);
        cache.sweep();
        assert_eq!(cache.len(), 1);
        assert!(cache.entries_contains(1));
        assert!(!cache.entries_contains(2));
    }

    // Small test-only helper so we do not expose the internal map.
    impl ImpulseCache {
        fn entries_contains(&self, id: u64) -> bool {
            self.entries.contains_key(&id)
        }
    }

    // --- UnionFind --------------------------------------------------------

    #[test]
    fn union_find_singletons_are_distinct() {
        let mut uf = UnionFind::new(5);
        for i in 0..5 {
            assert_eq!(uf.find(i), i);
        }
    }

    #[test]
    fn union_find_merges_and_deduplicates() {
        let mut uf = UnionFind::new(4);
        assert!(uf.union(0, 1));
        assert!(uf.union(1, 2));
        assert!(!uf.union(0, 2), "already in same set");
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
    }

    // --- Island detection -------------------------------------------------

    #[test]
    fn islands_empty_world_returns_no_islands() {
        let bodies: [MockBody; 0] = [];
        let contacts: [MockContact; 0] = [];
        let joints: [MockJoint; 0] = [];
        assert!(build_islands(&bodies, &contacts, &joints).is_empty());
    }

    #[test]
    fn islands_two_disjoint_pairs_form_two_groups() {
        let bodies = [
            MockBody {
                id: 0,
                dynamic: true,
            },
            MockBody {
                id: 1,
                dynamic: true,
            },
            MockBody {
                id: 2,
                dynamic: true,
            },
            MockBody {
                id: 3,
                dynamic: true,
            },
        ];
        let contacts = [
            MockContact {
                id: 100,
                a: 0,
                b: 1,
            },
            MockContact {
                id: 101,
                a: 2,
                b: 3,
            },
        ];
        let islands = build_islands(&bodies, &contacts, &[] as &[MockJoint]);
        assert_eq!(islands.len(), 2);
        assert_eq!(islands[0].bodies, vec![0, 1]);
        assert_eq!(islands[1].bodies, vec![2, 3]);
    }

    #[test]
    fn islands_joint_bridges_two_pairs() {
        let bodies = [
            MockBody {
                id: 0,
                dynamic: true,
            },
            MockBody {
                id: 1,
                dynamic: true,
            },
            MockBody {
                id: 2,
                dynamic: true,
            },
        ];
        let contacts = [MockContact {
            id: 100,
            a: 0,
            b: 1,
        }];
        let joints = [MockJoint { a: 1, b: 2 }];
        let islands = build_islands(&bodies, &contacts, &joints);
        assert_eq!(islands.len(), 1);
        assert_eq!(islands[0].bodies, vec![0, 1, 2]);
        assert_eq!(islands[0].contacts, vec![0]);
        assert_eq!(islands[0].joints, vec![0]);
    }

    #[test]
    fn islands_static_body_does_not_fuse_two_stacks() {
        // ground (static) touches two independent dynamic stacks
        let bodies = [
            MockBody {
                id: 0,
                dynamic: false,
            }, // static ground
            MockBody {
                id: 1,
                dynamic: true,
            },
            MockBody {
                id: 2,
                dynamic: true,
            },
            MockBody {
                id: 3,
                dynamic: true,
            },
            MockBody {
                id: 4,
                dynamic: true,
            },
        ];
        let contacts = [
            MockContact {
                id: 100,
                a: 0,
                b: 1,
            }, // stack A resting on ground
            MockContact {
                id: 101,
                a: 1,
                b: 2,
            }, // stack A internal
            MockContact {
                id: 200,
                a: 0,
                b: 3,
            }, // stack B resting on ground
            MockContact {
                id: 201,
                a: 3,
                b: 4,
            }, // stack B internal
        ];
        let islands = build_islands(&bodies, &contacts, &[] as &[MockJoint]);
        assert_eq!(islands.len(), 2, "static ground must not fuse islands");
        // Each island should contain the static ground as a member (so
        // that the solver has access to both sides of the contact).
        assert!(islands[0].bodies.contains(&0));
        assert!(islands[1].bodies.contains(&0));
    }

    #[test]
    fn islands_are_deterministic_across_runs() {
        let bodies = [
            MockBody {
                id: 0,
                dynamic: true,
            },
            MockBody {
                id: 1,
                dynamic: true,
            },
            MockBody {
                id: 2,
                dynamic: true,
            },
        ];
        let contacts = [
            MockContact {
                id: 100,
                a: 2,
                b: 0,
            },
            MockContact {
                id: 101,
                a: 1,
                b: 2,
            },
        ];
        let a = build_islands(&bodies, &contacts, &[] as &[MockJoint]);
        let b = build_islands(&bodies, &contacts, &[] as &[MockJoint]);
        assert_eq!(a, b);
    }

    // --- TgsConfig / driver ----------------------------------------------

    #[test]
    fn tgs_config_defaults_are_reasonable() {
        let cfg = TgsConfig::default();
        assert!(cfg.substeps >= 1);
        assert!(cfg.velocity_iters >= 1);
        assert!(cfg.position_iters >= 1);
        assert!(cfg.warmstart);
    }

    struct HookCounter {
        begin: u32,
        vel: u32,
        pos: u32,
        end: u32,
    }
    impl TgsHooks for HookCounter {
        fn begin_substep(&mut self, _sub_dt: Fix128) {
            self.begin += 1;
        }
        fn velocity_iteration(&mut self, _sub_dt: Fix128) {
            self.vel += 1;
        }
        fn position_iteration(&mut self, _sub_dt: Fix128) {
            self.pos += 1;
        }
        fn end_substep(&mut self, _sub_dt: Fix128) {
            self.end += 1;
        }
    }

    // --- Adapter --------------------------------------------------------

    #[test]
    fn body_ref_reports_dynamic_state() {
        let mut body = RigidBody::default();
        body.body_type = BodyType::Dynamic;
        let dyn_ref = BodyRef {
            body: &body,
            id: 42,
        };
        assert!(dyn_ref.is_dynamic());
        assert_eq!(dyn_ref.stable_id(), 42);

        let mut sbody = RigidBody::default();
        sbody.body_type = BodyType::Static;
        let stat_ref = BodyRef {
            body: &sbody,
            id: 7,
        };
        assert!(!stat_ref.is_dynamic());

        let mut kbody = RigidBody::default();
        kbody.body_type = BodyType::Kinematic;
        let kin_ref = BodyRef {
            body: &kbody,
            id: 3,
        };
        // kinematic bodies are treated as separators, same as static
        assert!(!kin_ref.is_dynamic());
    }

    // --- Dispatch -------------------------------------------------------

    #[test]
    fn dispatch_islands_visits_every_island_once() {
        let islands = vec![
            Island {
                bodies: vec![0, 1],
                contacts: vec![0],
                joints: vec![],
            },
            Island {
                bodies: vec![2, 3],
                contacts: vec![1],
                joints: vec![],
            },
        ];
        let mut visited = Vec::new();
        dispatch_islands(&islands, |isl| {
            visited.push(isl.bodies[0]);
        });
        assert_eq!(visited, vec![0, 2]);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn par_dispatch_islands_visits_every_island() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let islands: Vec<Island> = (0..16)
            .map(|i| Island {
                bodies: vec![i, i + 100],
                contacts: vec![],
                joints: vec![],
            })
            .collect();
        let counter = AtomicUsize::new(0);
        par_dispatch_islands(&islands, |_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn tgs_step_visits_hooks_the_expected_number_of_times() {
        let cfg = TgsConfig {
            substeps: 3,
            velocity_iters: 2,
            position_iters: 1,
            warmstart: false,
        };
        let mut h = HookCounter {
            begin: 0,
            vel: 0,
            pos: 0,
            end: 0,
        };
        tgs_step(&mut h, &cfg, Fix128::from_f32(1.0 / 60.0));
        assert_eq!(h.begin, 3);
        assert_eq!(h.vel, 6); // 3 * 2
        assert_eq!(h.pos, 3); // 3 * 1
        assert_eq!(h.end, 3);
    }
}
