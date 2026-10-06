//! Per-island scoped solve helpers for [`Pgs6DofOrientedHooks`].
//!
//! Phase E of the sub-stepping TGS solver stack.
//!
//! This module provides the per-island scoping layer for the oriented,
//! full 6-DOF variant defined in
//! [`crate::solver_tgs_hooks_6dof_oriented`].
//!
//! The oriented hook applies gravity, integrates linear + angular
//! velocity and updates the body orientation quaternion inside
//! `end_substep`; calling it once per island against the raw world
//! slice would therefore accumulate gravity multiple times. This layer
//! extracts the isolated body + contact subset for a single island,
//! remaps contact body indices into the local `[0..N)` range, runs the
//! standard oriented hook + [`tgs_step`] against the subset, and writes
//! the updated body states back to the world slice.
//!
//! Determinism: because the islands built by
//! [`crate::solver_tgs::build_islands`] never share a dynamic body,
//! the parallel dispatch is bit-perfect identical to the serial
//! variant thanks to Fix128 arithmetic and canonical island ordering.
//!
//! # Visibility
//!
//! `pub(crate)` since v0.14.0-preview.8 — see [`crate::solver_tgs`] for the
//! Option-C rationale.

#![allow(dead_code)]
#![allow(rustdoc::broken_intra_doc_links)]

use crate::math::Fix128;
use crate::solver_tgs::{dispatch_islands, tgs_step, ImpulseCache, Island, TgsConfig};
use crate::solver_tgs_hooks_6dof_oriented::{
    Body6DofOrientedState, ContactOriented, JointOriented, Pgs6DofOrientedConfig,
    Pgs6DofOrientedHooks,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Single-island solve
// ---------------------------------------------------------------------------

/// Run [`tgs_step`] with a [`Pgs6DofOrientedHooks`] that only sees the
/// bodies and contacts belonging to `island`.
///
/// The world body/contact slices are read to build a local copy that
/// mirrors only what the island refers to; body indices inside the
/// island's contacts are remapped to the `[0..N)` range of the local
/// body buffer. After the sub-step the updated body states are written
/// back to `world_bodies`, and the contact accumulators are written
/// back to `world_contacts` while restoring the original world body
/// indices.
///
/// # Panics
/// Panics if a contact in `island.contacts` refers to a body that is
/// not listed in `island.bodies`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_oriented_island_isolated(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    world_joints: &mut [JointOriented],
    island: &Island,
    cache: &mut ImpulseCache,
    cfg: Pgs6DofOrientedConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    // 1. Local body buffer + reverse-map into world indices.
    let world_indices: Vec<usize> = island.bodies.clone();
    let mut local_bodies: Vec<Body6DofOrientedState> =
        world_indices.iter().map(|&i| world_bodies[i]).collect();
    let world_to_local: HashMap<usize, usize> = world_indices
        .iter()
        .enumerate()
        .map(|(local, &world)| (world, local))
        .collect();

    // 2. Local contact buffer with body indices remapped into
    //    [0..local_bodies.len()).
    let mut local_contacts: Vec<ContactOriented> = island
        .contacts
        .iter()
        .map(|&ci| {
            let mut c = world_contacts[ci];
            c.body_a = *world_to_local
                .get(&c.body_a)
                .expect("contact body_a not in island");
            c.body_b = *world_to_local
                .get(&c.body_b)
                .expect("contact body_b not in island");
            c
        })
        .collect();

    // 2.5. Local joint buffer, remapped the same way as contacts above.
    let mut local_joints: Vec<JointOriented> = island
        .joints
        .iter()
        .map(|&ji| {
            let mut j = world_joints[ji];
            j.body_a = *world_to_local
                .get(&j.body_a)
                .expect("joint body_a not in island");
            j.body_b = *world_to_local
                .get(&j.body_b)
                .expect("joint body_b not in island");
            j
        })
        .collect();

    // 3. Standard oriented hook + tgs_step against the local buffers.
    {
        let mut hooks = Pgs6DofOrientedHooks::new(
            &mut local_bodies,
            &mut local_contacts,
            &mut local_joints,
            cache,
            cfg,
        );
        tgs_step(&mut hooks, tgs_cfg, dt);
    }

    // 4. Write body updates back to the world slice at the correct
    //    world indices (dynamic and static alike; the hook already
    //    skipped movement for the static ones).
    for (local, &world) in world_indices.iter().enumerate() {
        world_bodies[world] = local_bodies[local];
    }

    // 5. Write contact accumulators back to the world slice, but
    //    restore the original world body indices so downstream code
    //    keeps pointing at the world bodies.
    for (local_i, &world_i) in island.contacts.iter().enumerate() {
        let orig = world_contacts[world_i];
        let updated = local_contacts[local_i];
        world_contacts[world_i] = ContactOriented {
            body_a: orig.body_a,
            body_b: orig.body_b,
            ..updated
        };
    }

    // 5.5. Write joint accumulators back the same way.
    for (local_i, &world_i) in island.joints.iter().enumerate() {
        let orig = world_joints[world_i];
        let updated = local_joints[local_i];
        world_joints[world_i] = JointOriented {
            body_a: orig.body_a,
            body_b: orig.body_b,
            ..updated
        };
    }
}

// ---------------------------------------------------------------------------
// Bulk solve — serial and parallel
// ---------------------------------------------------------------------------

/// Serially solves every island using [`solve_oriented_island_isolated`],
/// sharing a single [`ImpulseCache`] across islands. Because the
/// islands are disjoint, this is equivalent to calling
/// [`solve_oriented_island_isolated`] once per island in canonical
/// order — dispatched via [`dispatch_islands`] rather than a
/// hand-written loop so the traversal is shared with any other consumer
/// of the `solver_tgs` island-dispatch surface.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_oriented_islands_serial(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    world_joints: &mut [JointOriented],
    islands: &[Island],
    cache: &mut ImpulseCache,
    cfg: Pgs6DofOrientedConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    dispatch_islands(islands, |island| {
        solve_oriented_island_isolated(
            world_bodies,
            world_contacts,
            world_joints,
            island,
            cache,
            cfg,
            tgs_cfg,
            dt,
        );
    });
}

/// Solves every island in parallel via [`par_dispatch_islands`].
///
/// Each island receives its own dedicated [`ImpulseCache`] so that
/// there is no shared state between threads. Callers that want a
/// unified warm-start pool can merge the per-island caches after the
/// call. `caches.len()` must equal `islands.len()`.
///
/// Determinism: the per-island solve is a pure function of the island's
/// inputs, so the aggregated result is byte-identical to the serial
/// variant [`solve_oriented_islands_serial`] modulo the shared-vs-split
/// cache split (which is a caller-visible policy choice).
///
/// # Panics
/// Panics when `caches.len() != islands.len()` (enforced by
/// [`par_dispatch_islands`] itself).
/// Per-island result of the parallel solve phase:
/// `(world body indices, solved local bodies, contact write-back list)`.
#[cfg(feature = "parallel")]
type IslandUpdate = (
    Vec<usize>,
    Vec<Body6DofOrientedState>,
    Vec<(usize, ContactOriented)>,
    Vec<(usize, JointOriented)>,
);

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_oriented_islands_parallel(
    world_bodies: &mut [Body6DofOrientedState],
    world_contacts: &mut [ContactOriented],
    world_joints: &mut [JointOriented],
    islands: &[Island],
    caches: &mut [ImpulseCache],
    cfg: Pgs6DofOrientedConfig,
    tgs_cfg: &TgsConfig,
    dt: Fix128,
) {
    use crate::solver_tgs::par_dispatch_islands;

    // 1. Parallel-solve into local buffers via `par_dispatch_islands`,
    //    pairing each island with its own persisted `ImpulseCache` (the
    //    `aux` slot). Each call produces (world_indices,
    //    updated_local_bodies, contact_writeback[], joint_writeback[]).
    let updates: Vec<IslandUpdate> = par_dispatch_islands(islands, caches, |island, cache| {
        let world_indices: Vec<usize> = island.bodies.clone();
        let mut local_bodies: Vec<Body6DofOrientedState> =
            world_indices.iter().map(|&i| world_bodies[i]).collect();
        let world_to_local: HashMap<usize, usize> = world_indices
            .iter()
            .enumerate()
            .map(|(local, &world)| (world, local))
            .collect();
        let mut local_contacts: Vec<ContactOriented> = island
            .contacts
            .iter()
            .map(|&ci| {
                let mut c = world_contacts[ci];
                c.body_a = *world_to_local
                    .get(&c.body_a)
                    .expect("contact body_a not in island");
                c.body_b = *world_to_local
                    .get(&c.body_b)
                    .expect("contact body_b not in island");
                c
            })
            .collect();
        let mut local_joints: Vec<JointOriented> = island
            .joints
            .iter()
            .map(|&ji| {
                let mut j = world_joints[ji];
                j.body_a = *world_to_local
                    .get(&j.body_a)
                    .expect("joint body_a not in island");
                j.body_b = *world_to_local
                    .get(&j.body_b)
                    .expect("joint body_b not in island");
                j
            })
            .collect();
        {
            let mut hooks = Pgs6DofOrientedHooks::new(
                &mut local_bodies,
                &mut local_contacts,
                &mut local_joints,
                cache,
                cfg,
            );
            tgs_step(&mut hooks, tgs_cfg, dt);
        }
        let contact_writeback: Vec<(usize, ContactOriented)> = island
            .contacts
            .iter()
            .zip(local_contacts)
            .map(|(&world_i, updated)| (world_i, updated))
            .collect();
        let joint_writeback: Vec<(usize, JointOriented)> = island
            .joints
            .iter()
            .zip(local_joints)
            .map(|(&world_i, updated)| (world_i, updated))
            .collect();
        (
            world_indices,
            local_bodies,
            contact_writeback,
            joint_writeback,
        )
    });

    // 2. Sequential write-back stage (canonical island order preserved by
    //    `par_dispatch_islands`'s `par_iter().zip().map().collect()` over
    //    indexed slices). Because the islands are disjoint on dynamic
    //    bodies, this stage only ever writes into disjoint world indices.
    for (world_indices, local_bodies, contact_writeback, joint_writeback) in updates {
        for (local, &world) in world_indices.iter().enumerate() {
            world_bodies[world] = local_bodies[local];
        }
        for (world_i, updated) in contact_writeback {
            let orig = world_contacts[world_i];
            world_contacts[world_i] = ContactOriented {
                body_a: orig.body_a,
                body_b: orig.body_b,
                ..updated
            };
        }
        for (world_i, updated) in joint_writeback {
            let orig = world_joints[world_i];
            world_joints[world_i] = JointOriented {
                body_a: orig.body_a,
                body_b: orig.body_b,
                ..updated
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::solver_tgs::{build_islands, tgs_step, ImpulseCache, JointLike, TgsConfig};

    struct NoJoint;
    impl JointLike for NoJoint {
        fn body_a(&self) -> usize {
            0
        }
        fn body_b(&self) -> usize {
            0
        }
    }
    const NO_JOINTS: [NoJoint; 0] = [];

    fn make_body(px: i64, py: i64, is_dynamic: bool, stable_id: u64) -> Body6DofOrientedState {
        Body6DofOrientedState {
            position: [Fix128::from_int(px), Fix128::from_int(py), Fix128::ZERO],
            orientation: QuatFix::IDENTITY,
            linear_velocity: [Fix128::ZERO; 3],
            angular_velocity: [Fix128::ZERO; 3],
            inv_mass: if is_dynamic {
                Fix128::from_int(1)
            } else {
                Fix128::ZERO
            },
            inv_inertia_local: if is_dynamic {
                [Fix128::from_int(1); 3]
            } else {
                [Fix128::ZERO; 3]
            },
            is_dynamic,
            stable_id,
        }
    }

    fn make_contact(a: usize, b: usize, stable_id: u64) -> ContactOriented {
        ContactOriented {
            body_a: a,
            body_b: b,
            stable_id,
            normal: [Fix128::ZERO, Fix128::from_int(1), Fix128::ZERO],
            tangent1: [Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO],
            tangent2: [Fix128::ZERO, Fix128::ZERO, Fix128::from_int(1)],
            r_a: [Fix128::ZERO; 3],
            r_b: [Fix128::ZERO; 3],
            penetration: Fix128::from_ratio(1, 100),
            friction: Fix128::from_ratio(1, 2),
            restitution: Fix128::ZERO,
            accum_normal: Fix128::ZERO,
            accum_tangent1: Fix128::ZERO,
            accum_tangent2: Fix128::ZERO,
        }
    }

    /// `solve_oriented_island_isolated` on a single-island scene must
    /// produce byte-identical results to the direct whole-world
    /// `Pgs6DofOrientedHooks + tgs_step` invocation, since there is no
    /// cross-island interference in the single-island case.
    #[test]
    fn scoped_single_island_bit_perfect_with_direct_hook() {
        let mut bodies_scoped = vec![make_body(0, 0, true, 1), make_body(0, 1, true, 2)];
        let mut contacts_scoped = vec![make_contact(0, 1, 100)];
        let mut bodies_direct = bodies_scoped.clone();
        let mut contacts_direct = contacts_scoped.clone();

        let cfg = Pgs6DofOrientedConfig::default();
        let tgs_cfg = TgsConfig::default();
        let dt = Fix128::from_ratio(1, 60);

        // Direct whole-world solve.
        {
            let mut cache = ImpulseCache::default();
            let mut no_joints: Vec<JointOriented> = Vec::new();
            let mut hooks = Pgs6DofOrientedHooks::new(
                &mut bodies_direct,
                &mut contacts_direct,
                &mut no_joints,
                &mut cache,
                cfg,
            );
            tgs_step(&mut hooks, &tgs_cfg, dt);
        }

        // Scoped single-island solve.
        {
            let islands = build_islands(&bodies_scoped, &contacts_scoped, &NO_JOINTS)
                .expect("valid island inputs");
            assert_eq!(islands.len(), 1, "expected exactly one island");
            let mut cache = ImpulseCache::default();
            let mut no_joints: Vec<JointOriented> = Vec::new();
            solve_oriented_island_isolated(
                &mut bodies_scoped,
                &mut contacts_scoped,
                &mut no_joints,
                &islands[0],
                &mut cache,
                cfg,
                &tgs_cfg,
                dt,
            );
        }

        for (a, b) in bodies_direct.iter().zip(bodies_scoped.iter()) {
            assert_eq!(a.position[0].hi, b.position[0].hi);
            assert_eq!(a.position[1].hi, b.position[1].hi);
            assert_eq!(a.position[2].hi, b.position[2].hi);
            assert_eq!(a.linear_velocity[1].hi, b.linear_velocity[1].hi);
        }
    }

    /// Parallel dispatch (`solve_oriented_islands_parallel`) must
    /// produce byte-identical results to the serial variant thanks to
    /// Fix128 arithmetic and canonical island ordering.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial_bit_perfect() {
        // Two disjoint dynamic-only islands (no static separator).
        let mut bodies_serial = vec![
            make_body(0, 0, true, 1),
            make_body(0, 1, true, 2),
            make_body(10, 0, true, 3),
            make_body(10, 1, true, 4),
        ];
        let mut contacts_serial = vec![make_contact(0, 1, 100), make_contact(2, 3, 200)];
        let mut bodies_parallel = bodies_serial.clone();
        let mut contacts_parallel = contacts_serial.clone();

        let islands = build_islands(&bodies_serial, &contacts_serial, &NO_JOINTS)
            .expect("valid island inputs");
        assert_eq!(islands.len(), 2, "expected two disjoint islands");

        let cfg = Pgs6DofOrientedConfig::default();
        let tgs_cfg = TgsConfig::default();
        let dt = Fix128::from_ratio(1, 60);

        let mut cache_serial = ImpulseCache::default();
        let mut no_joints_serial: Vec<JointOriented> = Vec::new();
        solve_oriented_islands_serial(
            &mut bodies_serial,
            &mut contacts_serial,
            &mut no_joints_serial,
            &islands,
            &mut cache_serial,
            cfg,
            &tgs_cfg,
            dt,
        );

        let mut caches_parallel: Vec<ImpulseCache> = (0..islands.len())
            .map(|_| ImpulseCache::default())
            .collect();
        let mut no_joints_parallel: Vec<JointOriented> = Vec::new();
        solve_oriented_islands_parallel(
            &mut bodies_parallel,
            &mut contacts_parallel,
            &mut no_joints_parallel,
            &islands,
            &mut caches_parallel,
            cfg,
            &tgs_cfg,
            dt,
        );

        for (bs, bp) in bodies_serial.iter().zip(bodies_parallel.iter()) {
            assert_eq!(bs.position[0].hi, bp.position[0].hi);
            assert_eq!(bs.position[1].hi, bp.position[1].hi);
            assert_eq!(bs.position[2].hi, bp.position[2].hi);
            assert_eq!(bs.linear_velocity[1].hi, bp.linear_velocity[1].hi);
        }
    }
}

/// Closed-form checks of [`Pgs6DofOrientedHooks`] driven by [`tgs_step`]
/// (or a single hook phase) on point-mass contacts (`r_a = r_b = 0`, so
/// the angular terms vanish and every result is a 1-D impulse formula).
#[cfg(test)]
mod closed_form {
    use crate::math::Fix128;
    use crate::solver_tgs::{tgs_step, CachedImpulse, ImpulseCache, TgsConfig, TgsHooks};
    use crate::solver_tgs_hooks_6dof_oriented::{
        Body6DofOrientedState, ContactOriented, Pgs6DofOrientedConfig, Pgs6DofOrientedHooks,
    };

    const Z3: [Fix128; 3] = [Fix128::ZERO; 3];

    fn dynamic(id: u64, inv_mass: Fix128, y: Fix128) -> Body6DofOrientedState {
        Body6DofOrientedState {
            position: [Fix128::ZERO, y, Fix128::ZERO],
            inv_mass,
            inv_inertia_local: [Fix128::ONE; 3],
            is_dynamic: true,
            stable_id: id,
            ..Default::default()
        }
    }

    fn fixed(id: u64) -> Body6DofOrientedState {
        Body6DofOrientedState {
            is_dynamic: false,
            stable_id: id,
            ..Default::default()
        }
    }

    /// Contact with normal `+y` (from `a` into `b`), tangents `+x` / `+z`.
    fn contact(a: usize, b: usize, id: u64, mu: Fix128, e: Fix128, pen: Fix128) -> ContactOriented {
        ContactOriented {
            body_a: a,
            body_b: b,
            stable_id: id,
            normal: [Fix128::ZERO, Fix128::ONE, Fix128::ZERO],
            tangent1: [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            tangent2: [Fix128::ZERO, Fix128::ZERO, Fix128::ONE],
            r_a: Z3,
            r_b: Z3,
            penetration: pen,
            friction: mu,
            restitution: e,
            accum_normal: Fix128::ZERO,
            accum_tangent1: Fix128::ZERO,
            accum_tangent2: Fix128::ZERO,
        }
    }

    fn no_gravity_cold() -> Pgs6DofOrientedConfig {
        Pgs6DofOrientedConfig {
            gravity: Z3,
            warmstart: false,
            ..Pgs6DofOrientedConfig::default()
        }
    }

    fn one_substep(velocity_iters: u32, position_iters: u32, warmstart: bool) -> TgsConfig {
        TgsConfig {
            substeps: 1,
            velocity_iters,
            position_iters,
            warmstart,
        }
    }

    fn dt() -> Fix128 {
        Fix128::from_ratio(1, 60)
    }

    /// Two point masses `m_a = 1`, `m_b = 2` closing along the normal
    /// (`v_a = +1`, `v_b = -2`). Newton's impact law for a frictionless
    /// central impact (e.g. Brach, *Mechanical Impact Dynamics*, ch. 2):
    /// `e = 0` gives the common velocity `(m_a v_a + m_b v_b) / (m_a + m_b)
    /// = -1`; `e = 1` gives `v_a' = ((m_a - m_b) v_a + 2 m_b v_b) / (m_a + m_b)
    /// = -3` and `v_b' = ((m_b - m_a) v_b + 2 m_a v_a) / (m_a + m_b) = 0`.
    #[test]
    fn two_body_normal_impact_matches_newton_restitution_law() {
        for (e, want_a, want_b) in [(0, -1.0, -1.0), (1, -3.0, 0.0)] {
            let mut bodies = [
                dynamic(1, Fix128::ONE, Fix128::from_int(-1)),
                dynamic(2, Fix128::from_ratio(1, 2), Fix128::ONE),
            ];
            bodies[0].linear_velocity = [Fix128::ZERO, Fix128::ONE, Fix128::ZERO];
            bodies[1].linear_velocity = [Fix128::ZERO, Fix128::from_int(-2), Fix128::ZERO];
            let mut contacts = [contact(
                0,
                1,
                10,
                Fix128::ZERO,
                Fix128::from_int(e),
                Fix128::ZERO,
            )];
            let mut cache = ImpulseCache::new();
            let mut hooks = Pgs6DofOrientedHooks::new(
                &mut bodies,
                &mut contacts,
                &mut [],
                &mut cache,
                no_gravity_cold(),
            );
            tgs_step(&mut hooks, &one_substep(8, 0, false), dt());
            let va = bodies[0].linear_velocity[1].to_f64();
            let vb = bodies[1].linear_velocity[1].to_f64();
            assert!(
                (va - want_a).abs() < 1e-9 && (vb - want_b).abs() < 1e-9,
                "e = {e}: v_a = {va} (want {want_a}), v_b = {vb} (want {want_b}), tol 1e-9"
            );
            // momentum m_a v_a + m_b v_b = -3 is conserved by the impulse pair
            let p = va + 2.0 * vb;
            assert!((p + 3.0).abs() < 1e-9, "e = {e}: momentum {p}, want -3");
        }
    }

    /// Coulomb friction on a body (`m = 1`) sliding at `v_x = 1` over a
    /// fixed body with a held normal impulse `λ_n`: the tangential impulse
    /// is clamped to `μ λ_n`, so `v_x' = max(0, v_x - μ λ_n / m)`.
    /// `μ = 1/2, λ_n = 1` hits the cone limit (`v_x' = 1/2`);
    /// `μ = 1, λ_n = 2` is inside the cone (`v_x' = 0`).
    #[test]
    fn friction_impulse_is_clamped_to_the_coulomb_cone() {
        for (mu, lambda_n, want) in [((1, 2), 1, 0.5), ((1, 1), 2, 0.0)] {
            let mut bodies = [fixed(1), dynamic(2, Fix128::ONE, Fix128::ZERO)];
            bodies[1].linear_velocity = [Fix128::ONE, Fix128::ZERO, Fix128::ZERO];
            let mut contacts = [contact(
                0,
                1,
                20,
                Fix128::from_ratio(mu.0, mu.1),
                Fix128::ZERO,
                Fix128::ZERO,
            )];
            contacts[0].accum_normal = Fix128::from_int(lambda_n);
            let mut cache = ImpulseCache::new();
            let mut hooks = Pgs6DofOrientedHooks::new(
                &mut bodies,
                &mut contacts,
                &mut [],
                &mut cache,
                no_gravity_cold(),
            );
            // velocity iterations only: `begin_substep` would reset the held
            // normal impulse; the normal row sees `v_n = 0` and leaves it.
            for _ in 0..8 {
                hooks.velocity_iteration(dt());
            }
            let vx = bodies[1].linear_velocity[0].to_f64();
            // the cone clamp rescales through f32 (`scale_f`), hence 1e-6
            assert!(
                (vx - want).abs() < 1e-6,
                "mu = {}/{}, lambda_n = {lambda_n}: v_x = {vx}, want {want} (tol 1e-6)",
                mu.0,
                mu.1
            );
            assert_eq!(contacts[0].accum_normal, Fix128::from_int(lambda_n));
        }
    }

    /// Warm start applies the cached impulse `J = (t1, n, t2)` once at the
    /// start of the sub-step, so with no iterations `Δv = J / m`
    /// (`m = 1`): `(1/2, 2, -1/2)` exactly.
    #[test]
    fn warm_start_applies_the_cached_impulse_on_all_three_axes() {
        let mut cache = ImpulseCache::new();
        cache.set(
            30,
            CachedImpulse {
                normal: Fix128::from_int(2),
                tangent1: Fix128::from_ratio(1, 2),
                tangent2: Fix128::from_ratio(-1, 2),
            },
        );
        let mut bodies = [fixed(1), dynamic(2, Fix128::ONE, Fix128::ZERO)];
        let mut contacts = [contact(0, 1, 30, Fix128::ONE, Fix128::ZERO, Fix128::ZERO)];
        let cfg = Pgs6DofOrientedConfig {
            gravity: Z3,
            warmstart: true,
            ..Pgs6DofOrientedConfig::default()
        };
        let mut hooks =
            Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut [], &mut cache, cfg);
        tgs_step(&mut hooks, &one_substep(0, 0, true), dt());
        assert_eq!(
            bodies[1].linear_velocity,
            [
                Fix128::from_ratio(1, 2),
                Fix128::from_int(2),
                Fix128::from_ratio(-1, 2)
            ]
        );
        // the fixed body is never moved
        assert_eq!(bodies[0].linear_velocity, Z3);
    }

    /// Baumgarte position correction: penetration at or below `slop` is
    /// left alone; above it each position pass moves the dynamic body
    /// (fixed partner, so the effective mass is `m`) by
    /// `β (p - slop)` along the normal, the penetration being held fixed
    /// within the tick. `n` passes give `n β (p - slop)`.
    #[test]
    fn position_correction_moves_by_baumgarte_times_excess_per_pass() {
        let cfg = no_gravity_cold();
        let beta = cfg.baumgarte.to_f64();
        let slop = cfg.slop.to_f64();
        for (pen, passes) in [
            (Fix128::from_ratio(1, 1000), 4u32),
            (Fix128::from_ratio(1, 20), 4),
        ] {
            let mut bodies = [fixed(1), dynamic(2, Fix128::ONE, Fix128::ZERO)];
            let mut contacts = [contact(0, 1, 40, Fix128::ZERO, Fix128::ZERO, pen)];
            let mut cache = ImpulseCache::new();
            let mut hooks =
                Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut [], &mut cache, cfg);
            tgs_step(&mut hooks, &one_substep(0, passes, false), dt());
            let y = bodies[1].position[1].to_f64();
            let want = f64::from(passes) * beta * (pen.to_f64() - slop).max(0.0);
            assert!(
                (y - want).abs() < 1e-12,
                "p = {}: y = {y}, want {want} (tol 1e-12)",
                pen.to_f64()
            );
        }
    }

    /// Free fall with every default (`TgsConfig` 4/4/2, gravity `-9.81`):
    /// symplectic Euler over `s` sub-steps of `h = dt / s` gives per frame
    /// `y += v dt + g dt² (s + 1) / (2 s)`, `v += g dt`, so after one
    /// second `v = g`.
    #[test]
    fn default_free_fall_matches_the_symplectic_closed_form() {
        let tgs = TgsConfig::default();
        assert_eq!(
            (tgs.substeps, tgs.velocity_iters, tgs.position_iters),
            (4, 4, 2)
        );
        assert!(tgs.warmstart);
        let cfg = Pgs6DofOrientedConfig::default();
        let g = cfg.gravity[1].to_f64();
        assert!((g + 9.81).abs() < 1e-6, "default gravity {g}");
        assert!(cfg.gravity[0].is_zero() && cfg.gravity[2].is_zero());

        let mut bodies = [dynamic(1, Fix128::ONE, Fix128::from_int(100))];
        let mut contacts: [ContactOriented; 0] = [];
        let mut cache = ImpulseCache::new();
        let frames = 60;
        for _ in 0..frames {
            let mut hooks =
                Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut [], &mut cache, cfg);
            tgs_step(&mut hooks, &tgs, dt());
        }
        let s = f64::from(tgs.substeps);
        let dtf = 1.0 / 60.0;
        let (mut y, mut v) = (100.0, 0.0);
        for _ in 0..frames {
            y += v * dtf + g * dtf * dtf * (s + 1.0) / (2.0 * s);
            v += g * dtf;
        }
        let got_y = bodies[0].position[1].to_f64();
        let got_v = bodies[0].linear_velocity[1].to_f64();
        assert!((got_y - y).abs() < 1e-6, "y = {got_y}, closed form {y}");
        assert!((got_v - v).abs() < 1e-6, "v = {got_v}, closed form {v}");
        assert!(bodies[0].position[0].is_zero() && bodies[0].position[2].is_zero());
    }

    /// A body resting 2 mm into a fixed floor (below the default 5 mm
    /// slop) under default gravity for two seconds stays within 5 cm of
    /// the floor, never above it, nearly at rest, and the normal impulse
    /// carries its weight.
    #[test]
    fn default_resting_contact_is_stable() {
        let cfg = Pgs6DofOrientedConfig::default();
        assert!((cfg.slop.to_f64() - 0.005).abs() < 1e-6);
        assert!((cfg.baumgarte.to_f64() - 0.2).abs() < 1e-6);
        assert!(cfg.warmstart);
        let tgs = TgsConfig::default();
        let mut bodies = [
            fixed(0),
            dynamic(1, Fix128::ONE, Fix128::from_ratio(-2, 1000)),
        ];
        let mut contacts = [contact(
            0,
            1,
            50,
            Fix128::from_ratio(1, 2),
            Fix128::ZERO,
            Fix128::from_ratio(2, 1000),
        )];
        let mut cache = ImpulseCache::new();
        for _ in 0..120 {
            let mut hooks =
                Pgs6DofOrientedHooks::new(&mut bodies, &mut contacts, &mut [], &mut cache, cfg);
            tgs_step(&mut hooks, &tgs, dt());
        }
        let y = bodies[1].position[1].to_f64();
        let vy = bodies[1].linear_velocity[1].to_f64();
        assert!(y <= 0.0 && y > -0.05, "resting body at y = {y}");
        assert!(vy.abs() < 0.2, "resting body still moving: vy = {vy}");
        assert!(contacts[0].accum_normal > Fix128::ZERO);
    }
}
