//! Continuous collision in the world step: sweeping fast bodies so that they
//! cannot pass through thin geometry between two substeps.
//!
//! Off by default ([`WorldCcdConfig::new`]); [`PhysicsWorld::set_continuous_collision`]
//! turns it on. While it is off nothing in this module runs and the step is
//! the step without it, operation for operation.
//!
//! # Method
//!
//! In every XPBD substep ([`PhysicsWorld::step`] and `step_parallel`), right
//! after the positions are predicted and before the discrete contacts are
//! detected:
//!
//! 1. **Which bodies.** A dynamic awake or a kinematic, non-sensor body with
//!    a collision radius `r` is swept when its displacement in the substep `d` is longer
//!    than `motion_threshold · r` (strictly). With the default threshold `1`,
//!    a body that moves at most its radius per substep is left to the discrete
//!    detection, which then always sees it on the near side of a surface of
//!    zero thickness before its centre crosses it.
//! 2. **Sweep.** Every body is put back at its pose at the start of the
//!    substep, and each swept body's sphere of radius `r` is cast along `d`
//!    with [`PhysicsWorld::cast_sphere`] against the other bodies (their exact
//!    shapes) and the static colliders. SDF colliders are not swept (they keep
//!    their discrete push-out). The bodies are then put back at their predicted
//!    poses. The cast sees what its filter sees: bodies on the layers of the
//!    swept body's mask, no sensors. Before the cast, every target the swept
//!    body overlaps at the start of the substep is collected
//!    ([`PhysicsWorld::overlap_sphere`]) and every one it does not move
//!    further into is hidden at once, however many there are (a sphere
//!    rolling on a floor, or wedged among many bodies, still sees the wall
//!    ahead). Moving further into a target it overlaps is a hit at `t = 0`
//!    along the separating normal of the discrete contact. The cast returns
//!    only the nearest target, so a target that cannot answer the sweep (a
//!    body the pair filter does not let it collide with) is hidden and the
//!    cast repeated until it answers or finds nothing; each repetition hides
//!    one more target, so there is no count limit to run out of.
//! 3. **Moving obstacles.** A hit on a body that moved in the substep is
//!    re-timed by the relative motion of the two bounding spheres
//!    ([`crate::ccd::sphere_sphere_toi`]), exact for two plain spheres and
//!    early (never late) for a shaped body. When the bounding spheres do not
//!    meet under the relative motion, the target is hidden as above.
//! 4. **Response.** For the first hit at fraction `t` of the substep with
//!    normal `n` (from the obstacle toward the swept body), both bodies are
//!    placed at their poses at `t`, after which the pair moves as one along
//!    `n` (their mass-weighted displacement along `n`) and each keeps its own
//!    tangential displacement for the rest of the substep. A hit on a body
//!    also adds a contact of depth `0` between the two, so the velocity pass
//!    of the substep applies the pair's restitution to the approach velocity
//!    the bodies had before the substep's solve. A static or kinematic
//!    obstacle is not moved. A hit on a static collider has no body to carry
//!    a contact: the body stops at the surface along `n` and slides along it.
//!
//! A **kinematic** body is swept for the dynamic bodies it would pass
//! through and only those (static geometry and other kinematic bodies are
//! hidden). It is not placed: it follows its target, and step 4 with its
//! infinite mass carries the dynamic body along `n` to touch it and gives it
//! `(1 + e)` times the approach speed, as a wall moving into a free body.
//!
//! Step 4 makes a single head-on impact of two spheres come out as the
//! textbook collision: the momentum along `n` is kept and the separation
//! speed is `e` times the approach speed (`tests/analytic_world_step_ccd.rs`).
//!
//! # Why this method
//!
//! Stopping the body at the time of impact and handing the contact to the
//! existing velocity pass reuses the restitution law the discrete contacts
//! already use, so a body is not treated differently because it was fast.
//! Moving the pair as one along `n` after the impact keeps the momentum of
//! the substep: stopping both bodies at `t` would scale their velocities by
//! `t` and lose momentum when the masses or speeds differ. A speculative
//! contact alone (a contact with a negative depth solved by the position pass)
//! was not used, because the position pass only resolves positive depths and
//! changing it would change the step while the setting is off.
//!
//! # Determinism
//!
//! The sweep runs on one thread in body-index order, reads no clock and no
//! thread count, and computes every hit before it moves any body, so the
//! result depends only on the world. It is the same with `--features parallel`.
//!
//! # Not covered
//!
//! - Rotation during the substep (`COV-RIGID-075`): the sweep is a translating
//!   sphere, the bounding sphere for a shaped body.
//! - The TGS backend ([`crate::solver::SolverBackend::Tgs`]) does not sweep.
//! - A target the swept body overlaps at the start and moves out of or along
//!   is left to the discrete contacts.
//! - The displacement is measured with
//!   [`Vec3Fix::checked_length_scaled`](crate::math::Vec3Fix::checked_length_scaled);
//!   a body whose displacement in one substep is `2⁶³` or more is not swept.
//! - Only the first hit of a substep is answered; a second obstacle reached
//!   by the slide after it is left to the discrete detection of later
//!   substeps.
//! - When several swept bodies hit the same movable body in one substep, that
//!   body is placed for each hit in body-index order and the last placement
//!   stands.

use super::{BodyType, PhysicsWorld};
use crate::collider::Contact;
use crate::math::{Fix128, Vec3Fix};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// The setting for continuous collision in the world step (see the module
/// documentation of `world_ccd` for the method).
///
/// Built with [`Self::new`] (off) or [`Self::on`] and adjusted with the
/// `with_*` methods; the fields cannot be written from outside the crate.
///
/// # Examples
///
/// ```
/// use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, WorldCcdConfig};
///
/// let mut world = PhysicsWorld::new(PhysicsConfig::default());
/// assert!(!world.continuous_collision().is_enabled());
///
/// world.set_continuous_collision(WorldCcdConfig::on().with_motion_threshold(Fix128::from_ratio(1, 2)));
/// assert!(world.continuous_collision().is_enabled());
/// assert_eq!(world.continuous_collision().motion_threshold(), Fix128::from_ratio(1, 2));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct WorldCcdConfig {
    enabled: bool,
    motion_threshold: Fix128,
}

impl WorldCcdConfig {
    /// Off, with the default threshold `1` (a body is swept when it moves more
    /// than its collision radius in one substep).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            enabled: false,
            motion_threshold: Fix128::ONE,
        }
    }

    /// On, with the default threshold `1`.
    #[must_use]
    pub const fn on() -> Self {
        Self::new().with_enabled(true)
    }

    /// The same setting, switched on or off.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// The same setting with another threshold, in collision radii: a body is
    /// swept when its displacement in one substep is longer than
    /// `threshold × radius`. A threshold `≤ 0` sweeps every moving body.
    #[must_use]
    pub const fn with_motion_threshold(mut self, threshold: Fix128) -> Self {
        self.motion_threshold = threshold;
        self
    }

    /// Whether the step sweeps fast bodies.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The threshold, in collision radii.
    #[must_use]
    pub const fn motion_threshold(&self) -> Fix128 {
        self.motion_threshold
    }
}

impl Default for WorldCcdConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// The first hit of one swept body in one substep.
#[derive(Clone, Copy, Debug)]
pub(super) struct CcdHit {
    /// The swept body.
    body: usize,
    /// The body hit, `None` for a static collider.
    other: Option<usize>,
    /// Fraction of the substep at the hit, in `[0, 1]`.
    t: Fix128,
    /// Unit normal at the hit, from the obstacle toward the swept body.
    normal: Vec3Fix,
    /// Contact point on the obstacle.
    point: Vec3Fix,
}

impl PhysicsWorld {
    /// Set continuous collision for the world step. Off by default.
    pub fn set_continuous_collision(&mut self, config: WorldCcdConfig) {
        self.ccd = config;
    }

    /// The continuous collision setting of the world step.
    #[must_use]
    pub fn continuous_collision(&self) -> WorldCcdConfig {
        self.ccd
    }

    /// Whether the body is moved by the substep and may be placed by a hit.
    fn ccd_movable(&self, i: usize) -> bool {
        self.bodies[i].body_type == BodyType::Dynamic && !self.park.is_parked(i)
    }

    /// The displacement of body `i` in this substep (zero for a body the
    /// substep does not move).
    fn ccd_displacement(&self, i: usize) -> Vec3Fix {
        let b = &self.bodies[i];
        match b.body_type {
            BodyType::Static => Vec3Fix::ZERO,
            BodyType::Kinematic | BodyType::Dynamic if self.park.is_parked(i) => Vec3Fix::ZERO,
            _ => b.position - b.prev_position,
        }
    }

    /// Exchange the predicted and the start-of-substep pose of every body the
    /// substep moved (twice is the identity).
    fn ccd_swap_poses(&mut self) {
        for i in 0..self.bodies.len() {
            let b = &self.bodies[i];
            if b.body_type == BodyType::Static || self.park.is_parked(i) {
                continue;
            }
            let b = &mut self.bodies[i];
            core::mem::swap(&mut b.position, &mut b.prev_position);
            core::mem::swap(&mut b.rotation, &mut b.prev_rotation);
        }
    }

    /// Steps 1–4 of the module documentation up to the placement: find the
    /// first hit of every swept body and place the bodies. Returns the hits,
    /// whose contacts [`Self::ccd_add_contacts`] adds after the discrete
    /// detection. Does nothing while the setting is off.
    pub(super) fn ccd_sweep(&mut self) -> Vec<CcdHit> {
        let mut hits = Vec::new();
        if !self.ccd.enabled {
            return hits;
        }
        let n = self.bodies.len();
        let mut swept = Vec::new();
        for i in 0..n {
            let b = &self.bodies[i];
            let mover = match b.body_type {
                BodyType::Dynamic => !self.islands.is_sleeping(i),
                BodyType::Kinematic => true,
                BodyType::Static => false,
            };
            if !mover || b.is_sensor || self.park.is_parked(i) {
                continue;
            }
            let Some(radius) = self.body_collision_radii.get(i).and_then(|r| *r) else {
                continue;
            };
            let d = b.position - b.prev_position;
            let Some(travel) = d.checked_length_scaled() else {
                continue;
            };
            if travel.is_zero() {
                continue;
            }
            let Some(limit) = self.ccd.motion_threshold.checked_mul(radius) else {
                continue;
            };
            if travel > limit {
                swept.push((i, radius, d, travel));
            }
        }
        if swept.is_empty() {
            return hits;
        }

        self.ccd_swap_poses();
        for &(i, radius, d, travel) in &swept {
            let cast = self.ccd_first_hit(i, radius, d, travel);
            let pair = self.ccd_first_moving_hit(i, radius, d);
            let first = match (cast, pair) {
                (Some(c), Some(p)) => Some(if p.t < c.t { p } else { c }),
                (c, p) => c.or(p),
            };
            if let Some(hit) = first {
                hits.push(hit);
            }
        }
        self.ccd_swap_poses();

        // Every hit is known, and every displacement read, before any body
        // moves.
        let displacement: Vec<Vec3Fix> = (0..n).map(|i| self.ccd_displacement(i)).collect();
        for hit in &hits {
            self.ccd_place(hit, &displacement);
        }
        hits
    }

    /// The first hit of body `i` (at its start pose, the world at the start of
    /// the substep), moving by `d` of length `travel`.
    ///
    /// First every target the body overlaps at the start is collected
    /// ([`PhysicsWorld::overlap_sphere`] with the cast's filter) and every one
    /// that cannot answer the sweep is hidden at once: a target it does not
    /// move further into (`d·n ≥ 0` along the separating normal) and one the
    /// pair filter rejects. A target it does move into is left visible, so
    /// the cast reports it at `t = 0`. The cast then returns only the nearest
    /// target, so a target found unable to answer is hidden and the cast
    /// repeated: a body the pair filter rejects, a moving body whose bounding
    /// sphere the relative motion misses, and, for a kinematic body, everything
    /// but the dynamic bodies it can push. Every repetition hides one more
    /// target and a hidden target is not returned again, so the loop ends after
    /// at most as many casts as there are targets; there is no count limit
    /// that could end it without an answer. Hidden targets are put back, in
    /// reverse order, before returning.
    fn ccd_first_hit(
        &mut self,
        i: usize,
        radius: Fix128,
        d: Vec3Fix,
        travel: Fix128,
    ) -> Option<CcdHit> {
        let own = self.body_filter(i);
        let kinematic = self.bodies[i].body_type == BodyType::Kinematic;
        let filter = crate::shape_raycast::RayFilter::new()
            .with_layer_mask(own.mask)
            .with_sdf(false)
            .with_static(!kinematic)
            .excluding_body(i);
        let start = self.bodies[i].position;
        let mut hidden = Hidden::default();
        for target in self.overlap_sphere(start, radius, &filter) {
            let other = match target {
                crate::shape_raycast::RayTarget::Body(j) => Some(j),
                _ => None,
            };
            // A target without a separating normal is left to the cast (it
            // either sweeps it or reports a start overlap, judged below).
            let Some(n) = self.ccd_start_normal(i, radius, target) else {
                continue;
            };
            if d.dot(n) >= Fix128::ZERO || !self.ccd_accepts(i, other, kinematic) {
                self.ccd_hide(target, &mut hidden);
            }
        }
        let mut found = None;
        while let Some(hit) = self.cast_sphere(start, radius, d, travel, &filter) {
            match self.ccd_judge(i, radius, d, travel, kinematic, &hit) {
                Judged::Hit(h) => {
                    found = Some(h);
                    break;
                }
                Judged::Hide => {
                    if !self.ccd_hide(hit.target, &mut hidden) {
                        break;
                    }
                }
            }
        }
        for (j, layer) in hidden.bodies.into_iter().rev() {
            self.body_filters[j].layer = layer;
        }
        for (k, c) in hidden.statics.into_iter().rev() {
            self.static_colliders[k] = c;
        }
        found
    }

    /// Hide `target` from the casts of [`Self::ccd_first_hit`] (a body by
    /// clearing its layer, a static collider by an empty mesh), recording what
    /// to put back. `false` for a target that cannot be hidden (an SDF
    /// collider, which the cast's filter excludes anyway).
    fn ccd_hide(&mut self, target: crate::shape_raycast::RayTarget, hidden: &mut Hidden) -> bool {
        match target {
            crate::shape_raycast::RayTarget::Body(j) => {
                hidden.bodies.push((j, self.body_filters[j].layer));
                self.body_filters[j].layer = 0;
                true
            }
            crate::shape_raycast::RayTarget::StaticCollider(k) => {
                let empty = crate::static_collider::StaticCollider::TriMesh(
                    crate::trimesh::TriMesh::from_indexed(&[], &[]),
                );
                hidden
                    .statics
                    .push((k, core::mem::replace(&mut self.static_colliders[k], empty)));
                true
            }
            crate::shape_raycast::RayTarget::Sdf(_) => false,
        }
    }

    /// What one cast hit means for the sweep of body `i` (see
    /// [`Self::ccd_first_hit`]).
    fn ccd_judge(
        &self,
        i: usize,
        radius: Fix128,
        d: Vec3Fix,
        travel: Fix128,
        kinematic: bool,
        hit: &crate::world_shape_query::WorldShapeHit,
    ) -> Judged {
        let start = self.bodies[i].position;
        // Overlapping at the start: the query reports the body's own centre.
        // Moving further into the target is a hit at `t = 0` along the
        // separating normal; anything else is left to the discrete contacts.
        if hit.t.is_zero() && hit.point == start {
            return match self.ccd_start_normal(i, radius, hit.target) {
                Some(n) if d.dot(n) < Fix128::ZERO && self.ccd_accepts(i, hit.body, kinematic) => {
                    Judged::Hit(CcdHit {
                        body: i,
                        other: hit.body,
                        t: Fix128::ZERO,
                        normal: n,
                        point: start - n * radius,
                    })
                }
                _ => Judged::Hide,
            };
        }
        if !self.ccd_accepts(i, hit.body, kinematic) {
            return Judged::Hide;
        }
        let Some(j) = hit.body else {
            return Judged::Hit(CcdHit {
                body: i,
                other: None,
                t: fraction(hit.t, travel),
                normal: hit.normal,
                point: hit.point,
            });
        };
        // The swapped body holds its predicted position in `prev_position`.
        let dj = match self.bodies[j].body_type {
            BodyType::Static => Vec3Fix::ZERO,
            _ if self.park.is_parked(j) => Vec3Fix::ZERO,
            _ => self.bodies[j].prev_position - self.bodies[j].position,
        };
        if dj == Vec3Fix::ZERO {
            return Judged::Hit(CcdHit {
                body: i,
                other: Some(j),
                t: fraction(hit.t, travel),
                normal: hit.normal,
                point: hit.point,
            });
        }
        let Some(rj) = self.body_collision_radii.get(j).and_then(|r| *r) else {
            return Judged::Hide;
        };
        // Obstacle as sphere A, the swept body as sphere B: the normal points
        // from the obstacle toward the swept body.
        match crate::ccd::sphere_sphere_toi(self.bodies[j].position, rj, dj, start, radius, d) {
            Some(toi) => Judged::Hit(CcdHit {
                body: i,
                other: Some(j),
                t: toi.t,
                normal: toi.normal,
                point: toi.point,
            }),
            None => Judged::Hide,
        }
    }

    /// Whether a hit of body `i` on `other` (`None`: a static collider) is
    /// answered: the pair filter lets them collide, and a kinematic body only
    /// answers dynamic bodies it can move.
    fn ccd_accepts(&self, i: usize, other: Option<usize>, kinematic: bool) -> bool {
        match other {
            None => !kinematic,
            Some(j) => {
                crate::filter::CollisionFilter::can_collide(
                    &self.body_filter(i),
                    &self.body_filter(j),
                ) && (!kinematic || self.ccd_movable(j))
            }
        }
    }

    /// The separating normal (toward body `i`'s sphere at its start) of a
    /// target `i` overlaps at the start of the substep, as the discrete
    /// contacts compute it; `None` when there is none to take.
    fn ccd_start_normal(
        &self,
        i: usize,
        radius: Fix128,
        target: crate::shape_raycast::RayTarget,
    ) -> Option<Vec3Fix> {
        let start = self.bodies[i].position;
        match target {
            crate::shape_raycast::RayTarget::Body(j) => {
                let pose = (self.bodies[j].position, self.bodies[j].rotation);
                match self.body_colliders.get(j).and_then(Option::as_ref) {
                    Some(c) => crate::body_collider::contact_with_sphere(
                        c,
                        pose,
                        crate::collider::Sphere::new(start, radius),
                        false,
                    )
                    .map(|c| c.normal),
                    None => (start - pose.0).try_normalize_scaled(),
                }
            }
            crate::shape_raycast::RayTarget::StaticCollider(k) => self
                .static_colliders
                .get(k)
                .and_then(|c| c.collide_sphere(start, radius))
                .map(|c| c.normal),
            crate::shape_raycast::RayTarget::Sdf(_) => None,
        }
    }

    /// The first hit of body `i` (at its start pose, radius `radius`, moving by
    /// `d`) on another body that moves in this substep, by the relative motion
    /// of the bounding spheres; ties go to the lower body index. The cast of
    /// [`Self::ccd_first_hit`] sees such a body only where it starts, which
    /// misses two bodies that close on each other within the substep.
    fn ccd_first_moving_hit(&self, i: usize, radius: Fix128, d: Vec3Fix) -> Option<CcdHit> {
        let own = self.body_filter(i);
        let start = self.bodies[i].position;
        let mut best: Option<CcdHit> = None;
        for j in 0..self.bodies.len() {
            if j == i || self.bodies[j].is_sensor {
                continue;
            }
            if self.bodies[i].body_type == BodyType::Kinematic && !self.ccd_movable(j) {
                continue;
            }
            let Some(rj) = self.body_collision_radii.get(j).and_then(|r| *r) else {
                continue;
            };
            // The swapped body holds its predicted position in `prev_position`.
            let dj = match self.bodies[j].body_type {
                BodyType::Static => continue,
                _ if self.park.is_parked(j) => continue,
                _ => self.bodies[j].prev_position - self.bodies[j].position,
            };
            if dj == Vec3Fix::ZERO {
                continue;
            }
            let other = self.body_filter(j);
            if !crate::filter::CollisionFilter::can_collide(&own, &other) {
                continue;
            }
            let pj = self.bodies[j].position;
            // Overlapping at the start: left to the discrete contacts.
            let gap = start - pj;
            let reach = radius + rj;
            if gap
                .checked_length_squared()
                .zip(reach.checked_mul(reach))
                .is_some_and(|(g2, r2)| g2 <= r2)
            {
                continue;
            }
            let Some(toi) = crate::ccd::sphere_sphere_toi(pj, rj, dj, start, radius, d) else {
                continue;
            };
            if best.is_none_or(|b| toi.t < b.t) {
                best = Some(CcdHit {
                    body: i,
                    other: Some(j),
                    t: toi.t,
                    normal: toi.normal,
                    point: toi.point,
                });
            }
        }
        best
    }

    /// Place the bodies of one hit (step 4 of the module documentation).
    fn ccd_place(&mut self, hit: &CcdHit, displacement: &[Vec3Fix]) {
        let n = hit.normal;
        let rest = Fix128::ONE - hit.t;
        let di = displacement[hit.body];
        let (dj, wj) = match hit.other {
            Some(j) => (
                displacement[j],
                if self.ccd_movable(j) {
                    self.bodies[j].inv_mass
                } else {
                    Fix128::ZERO
                },
            ),
            None => (Vec3Fix::ZERO, Fix128::ZERO),
        };
        let wi = self.bodies[hit.body].inv_mass;
        let w = wi + wj;
        // Displacement of the pair along `n` after the hit: the mass-weighted
        // mean (a static or kinematic obstacle carries the pair with it).
        let along = if w.is_zero() {
            Fix128::ZERO
        } else {
            (di.dot(n) * wj + dj.dot(n) * wi) / w
        };
        let place = |p0: Vec3Fix, d: Vec3Fix| -> Vec3Fix {
            let tangent = d - n * d.dot(n);
            p0 + d * hit.t + (tangent + n * along) * rest
        };
        // A kinematic body follows its target: it is not placed.
        if self.ccd_movable(hit.body) {
            let pi = place(self.bodies[hit.body].prev_position, di);
            self.bodies[hit.body].position = pi;
        }
        if let Some(j) = hit.other {
            if self.ccd_movable(j) {
                let pj = place(self.bodies[j].prev_position, dj);
                self.bodies[j].position = pj;
            }
        }
    }

    /// Add the depth-0 contacts of the hits on bodies, after the substep's
    /// discrete detection (which clears the contacts), once per pair.
    pub(super) fn ccd_add_contacts(&mut self, hits: &[CcdHit]) {
        for (k, hit) in hits.iter().enumerate() {
            let Some(j) = hit.other else {
                continue;
            };
            let i = hit.body;
            // The other body of a pair hit from both sides adds it once.
            if hits[..k].iter().any(|h| h.body == j && h.other == Some(i)) {
                continue;
            }
            let radius = self
                .body_collision_radii
                .get(i)
                .and_then(|r| *r)
                .unwrap_or(Fix128::ZERO);
            let contact = Contact {
                depth: Fix128::ZERO,
                normal: hit.normal,
                point_a: self.bodies[i].position - hit.normal * radius,
                point_b: hit.point,
            };
            let rel_vel = (self.bodies[i].velocity - self.bodies[j].velocity).dot(hit.normal);
            self.events
                .report_contact(i, j, hit.normal, hit.point, Fix128::ZERO, rel_vel);
            for x in [i, j] {
                if !self.islands.is_sleeping(x) {
                    continue;
                }
                if self.park.is_parked(x) {
                    self.islands.wake_body(x);
                    self.park.unpark(x, &mut self.stage_work);
                } else {
                    self.islands.wake_island(x);
                }
            }
            self.add_contact_with_material(i, j, contact);
        }
    }
}

/// The targets one sweep has hidden, in the order they were hidden.
#[derive(Default)]
struct Hidden {
    /// Body index and its collision layer before it was hidden.
    bodies: Vec<(usize, u32)>,
    /// Static collider index and the collider itself.
    statics: Vec<(usize, crate::static_collider::StaticCollider)>,
}

/// The verdict on one cast hit.
enum Judged {
    /// The sweep's answer.
    Hit(CcdHit),
    /// Hide the target and cast again.
    Hide,
}

/// `hit / travel` as a fraction of the substep, `1` at or past the end.
fn fraction(hit: Fix128, travel: Fix128) -> Fix128 {
    if hit >= travel {
        Fix128::ONE
    } else {
        hit / travel
    }
}
