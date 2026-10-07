//! Move-and-slide of a [`CharacterController`] against the geometry a
//! [`PhysicsWorld`] collides with.
//!
//! [`CharacterController::move_and_slide`] takes a slice of bodies and SDF
//! colliders and sees each static body as a sphere of the character's radius; it
//! does not see the world's static colliders (planes, height fields, triangle
//! meshes) or the bodies' true shapes. [`PhysicsWorld::move_character`] sweeps the
//! controller's capsule with [`PhysicsWorld::cast_capsule`], so it meets the same
//! geometry the world collides with: body shapes, compound children, static
//! colliders and (`std`) SDF colliders.
//!
//! # The capsule
//!
//! Centre [`CharacterController::position`], total height
//! [`CharacterConfig::height`] (hemispheres included) and radius
//! [`CharacterConfig::radius`]: the segment runs from
//! `position − (height/2 − radius)·Y` to `position + (height/2 − radius)·Y`
//! (a sphere when `height ≤ 2·radius`). `+Y` is up, as in
//! [`CharacterController::move_and_slide`].
//!
//! # One move
//!
//! The displacement plus [`CharacterController::platform_velocity`] (as in
//! [`CharacterController::move_and_slide`]) is swept in up to
//! [`CharacterConfig::max_slides`] steps:
//!
//! 1. Sweep the capsule along the rest of the displacement (`len + skin`). No hit:
//!    move the whole rest and stop.
//! 2. Hit at distance `t` with normal `n`: move along the direction until the
//!    capsule is [`CharacterConfig::skin_width`] from the surface along `n`,
//!    `t − skin / (−direction·n)` (at least 0). The capsule only moves along the
//!    swept path, which the cast found clear, so a move never makes it overlap.
//! 3. Project the rest (`len` minus the distance moved, along the direction)
//!    onto the contact plane:
//!    `rest − n (rest · n)`. A surface steeper than
//!    [`CharacterConfig::max_slope_angle`] whose projection would climb (`y > 0`)
//!    is treated as a vertical wall: the rest is projected onto its horizontal
//!    normal instead, so a steep slope cannot be walked up.
//!
//! The rest left after the last step is dropped. In a corner the projections run
//! from one wall into the other; each such step moves 0 (the capsule already is
//! a skin width from the wall it runs into), so the capsule stays put.
//!
//! # Steps
//!
//! When the controller was grounded before the move, its step height is positive,
//! and the sweep met a surface steeper than the slope limit, the move is tried
//! again raised: up by the step height (less if a ceiling is nearer), the
//! horizontal part of the displacement swept from there, then down by the rise
//! plus two skin widths, stopping a skin width above the surface. The raised result is taken when it lands on a walkable
//! surface and gets more than a skin width further along the horizontal direction
//! than the plain sweep.
//!
//! # Ground
//!
//! After the move a downward sweep of [`CharacterConfig::ground_probe_distance`]
//! plus a skin width looks for ground: grounded when it hits a surface whose
//! normal makes at most [`CharacterConfig::max_slope_angle`] with `+Y`. The body
//! it belongs to (if any) becomes [`CharacterController::ground_body_index`] and
//! its velocity [`CharacterController::platform_velocity`].
//!
//! # Starting overlap
//!
//! Before the move, a capsule that starts overlapping colliders is pushed out:
//!
//! 1. Find every collider the capsule overlaps and how far: the distance `δ` from
//!    the capsule's segment to the collider (the queries of
//!    [`PhysicsWorld::overlap_sphere`], on the segment) is below the radius, the
//!    penetration is `r − δ` and the direction the collider's normal at the
//!    nearest point.
//! 2. Push the capsule along that normal by the penetration plus
//!    [`CharacterConfig::skin_width`], the deepest overlap first (ties by target:
//!    bodies, then static colliders, then SDF colliders, each by index), so it
//!    ends a skin width clear of that surface, as a move leaves it.
//! 3. Repeat until nothing overlaps, at most [`CharacterConfig::max_slides`]
//!    pushes.
//!
//! The move of § One move then starts from the freed position. A plane is
//! two-sided: a capsule below it is pushed down.
//!
//! The capsule is **not** freed, and keeps its position, when its segment itself
//! meets a solid (an overlap deeper than the radius: there is no distance and so
//! no push-out direction), when overlaps remain after
//! [`CharacterConfig::max_slides`] pushes, or when the pushes would add up to more
//! than [`CharacterConfig::radius`] `+` [`CharacterConfig::height`] (so a capsule
//! deep inside a large body is not flung out of it). It then starts the move where
//! it is, and [`PhysicsWorld::cast_capsule`] reports a cast that starts overlapping
//! at `t = 0` with normal `−direction`: there is no surface normal to slide along,
//! the move goes 0 along the direction and its projection onto the plane normal
//! to the direction is 0, so the controller does not move
//! ([`MoveResult::position`] is the old position); such a contact is not ground,
//! and no step is tried from inside (the upward sweep of § Steps also starts
//! overlapping).
//!
//! # Configuration
//!
//! A [`CharacterConfig`] with a negative [`CharacterConfig::radius`],
//! [`CharacterConfig::skin_width`] or [`CharacterConfig::height`] describes no
//! capsule: the move is refused, the controller is left unchanged and the
//! returned [`MoveResult`] is its current state. [`CharacterConfig::max_slides`]
//! of 0 is taken as 1 (one sweep, no slide). A height below `2·radius` is a
//! sphere of the radius.
//!
//! # Excluding the character's own body
//!
//! [`PhysicsWorld::move_character_with_filter`] takes a [`RayFilter`]: pass
//! [`RayFilter::excluding_body`] when the character also has a body in the world.
//!
//! Author: Moroya Sakamoto

use crate::character::{CharacterConfig, CharacterController, MoveResult};
use crate::math::{Fix128, Vec3Fix};
use crate::shape_raycast::RayFilter;
use crate::solver::PhysicsWorld;
use crate::world_shape_query::{Penetration, WorldShapeHit};

/// The controller's capsule and the constants a move uses.
#[derive(Clone, Copy)]
struct Sweep<'a> {
    world: &'a PhysicsWorld,
    filter: &'a RayFilter,
    /// Half the segment length, `height/2 − radius` (at least 0).
    half: Fix128,
    radius: Fix128,
    skin: Fix128,
    /// `cos(max_slope_angle)`: a normal with `n·Y` at least this is walkable.
    walkable: Fix128,
    max_slides: usize,
}

/// Result of [`Sweep::slide`].
#[derive(Clone, Copy)]
struct Slid {
    position: Vec3Fix,
    /// A surface steeper than the slope limit was met.
    steep: bool,
}

impl Sweep<'_> {
    /// The capsule centred at `center` swept along `direction` for `max_t`.
    fn cast(self, center: Vec3Fix, direction: Vec3Fix, max_t: Fix128) -> Option<WorldShapeHit> {
        let off = Vec3Fix::new(Fix128::ZERO, self.half, Fix128::ZERO);
        self.world.cast_capsule(
            center - off,
            center + off,
            self.radius,
            direction,
            max_t,
            self.filter,
        )
    }

    /// How far to move along the unit `dir` toward a contact at distance `t` with
    /// normal `n` to stop a skin width from the surface along `n`:
    /// `t − skin / (−dir·n)`, at least 0 (`t − skin` when `n` does not face `dir`).
    fn short_of(self, t: Fix128, dir: Vec3Fix, n: Vec3Fix) -> Fix128 {
        let facing = -dir.dot(n);
        let back = if facing > Fix128::ZERO {
            self.skin / facing
        } else {
            self.skin
        };
        if t > back {
            t - back
        } else {
            Fix128::ZERO
        }
    }

    fn is_walkable(self, n: Vec3Fix) -> bool {
        n.y >= self.walkable
    }

    /// Move-and-slide of `rest` from `position` (module doc, steps 1-3).
    fn slide(self, mut position: Vec3Fix, mut rest: Vec3Fix) -> Slid {
        let mut steep = false;
        for _ in 0..self.max_slides {
            let Some(dir) = rest.try_normalize() else {
                break;
            };
            let len = rest.length();
            let Some(hit) = self.cast(position, rest, len + self.skin) else {
                position = position + rest;
                break;
            };
            let n = hit.normal;
            let travel = self.short_of(hit.t, dir, n);
            position = position + dir * travel;
            let left = dir * (len - travel);
            let walkable = self.is_walkable(n);
            steep |= !walkable;
            let mut next = project(left, n);
            if !walkable && n.y > Fix128::ZERO && next.y > Fix128::ZERO {
                if let Some(wall) = Vec3Fix::new(n.x, Fix128::ZERO, n.z).try_normalize() {
                    next = project(left, wall);
                }
            }
            rest = next;
        }
        Slid { position, steep }
    }

    /// The raised move of the module doc § Steps, if it is taken.
    fn step(
        self,
        start: Vec3Fix,
        total: Vec3Fix,
        step_height: Fix128,
        plain: Vec3Fix,
    ) -> Option<Vec3Fix> {
        let horizontal = Vec3Fix::new(total.x, Fix128::ZERO, total.z);
        let along = horizontal.try_normalize()?;
        let rise = match self.cast(start, Vec3Fix::UNIT_Y, step_height + self.skin) {
            Some(hit) if hit.t > self.skin => hit.t - self.skin,
            Some(_) => return None,
            None => step_height,
        };
        let raised = start + Vec3Fix::new(Fix128::ZERO, rise, Fix128::ZERO);
        let across = self.slide(raised, horizontal).position;
        let down = -Vec3Fix::UNIT_Y;
        let hit = self.cast(across, down, rise + self.skin + self.skin)?;
        if !self.is_walkable(hit.normal) {
            return None;
        }
        let landed = across + down * self.short_of(hit.t, down, hit.normal);
        let progress = |p: Vec3Fix| (p - start).dot(along);
        (progress(landed) > progress(plain) + self.skin).then_some(landed)
    }

    /// The capsule at `position` pushed out of the colliders it starts overlapping
    /// (module doc § Starting overlap): `None` when it cannot be freed.
    fn depenetrate(self, position: Vec3Fix, max_push: Fix128) -> Option<Vec3Fix> {
        let off = Vec3Fix::new(Fix128::ZERO, self.half, Fix128::ZERO);
        let mut p = position;
        let mut pushed = Fix128::ZERO;
        for _ in 0..=self.max_slides {
            let overlaps =
                self.world
                    .capsule_penetrations(p - off, p + off, self.radius, self.filter)?;
            // Deepest first, ties by target (the list is sorted by target and only
            // a strictly deeper one replaces the first).
            let Some(deepest) =
                overlaps
                    .iter()
                    .fold(None, |best: Option<&Penetration>, o| match best {
                        Some(b) if b.depth >= o.depth => Some(b),
                        _ => Some(o),
                    })
            else {
                return Some(p);
            };
            let step = deepest.depth + self.skin;
            pushed = pushed + step;
            if pushed > max_push {
                return None;
            }
            p = p + deepest.normal * step;
        }
        None
    }

    /// The walkable ground under a capsule at `position` (module doc § Ground).
    fn ground(self, position: Vec3Fix, probe: Fix128) -> Option<WorldShapeHit> {
        let down = -Vec3Fix::UNIT_Y;
        let hit = self.cast(position, down, probe + self.skin)?;
        (!starts_inside(&hit, down) && self.is_walkable(hit.normal)).then_some(hit)
    }
}

/// `v` without its component along the unit `n`.
fn project(v: Vec3Fix, n: Vec3Fix) -> Vec3Fix {
    v - n * v.dot(n)
}

/// A cast that started overlapping: `t = 0`, normal `−direction` (the
/// [`crate::world_shape_query`] convention). `direction` is the vector passed to
/// the cast, normalized here the same way the cast normalizes it.
fn starts_inside(hit: &WorldShapeHit, direction: Vec3Fix) -> bool {
    hit.t.is_zero() && direction.try_normalize().is_some_and(|d| hit.normal == -d)
}

impl PhysicsWorld {
    /// Move `ctrl` by `displacement`, sliding along everything the world collides
    /// with (see [`crate::world_character`]), with the default [`RayFilter`].
    ///
    /// Updates the controller's position, grounded state, ground body and platform
    /// velocity as [`CharacterController::move_and_slide`] does, and returns them.
    pub fn move_character(
        &self,
        ctrl: &mut CharacterController,
        displacement: Vec3Fix,
    ) -> MoveResult {
        self.move_character_with_filter(ctrl, displacement, &RayFilter::default())
    }

    /// [`PhysicsWorld::move_character`] seeing only what `filter` sees (for
    /// instance [`RayFilter::excluding_body`] for the character's own body).
    pub fn move_character_with_filter(
        &self,
        ctrl: &mut CharacterController,
        displacement: Vec3Fix,
        filter: &RayFilter,
    ) -> MoveResult {
        let config: CharacterConfig = ctrl.config;
        // A negative radius, skin width or height describes no capsule: refused
        // (module doc § Configuration).
        if config.radius.is_negative()
            || config.skin_width.is_negative()
            || config.height.is_negative()
        {
            return MoveResult {
                position: ctrl.position,
                grounded: ctrl.grounded,
                velocity: ctrl.velocity,
                platform_velocity: ctrl.platform_velocity,
            };
        }
        let half = config.height.half() - config.radius;
        let sweep = Sweep {
            world: self,
            filter,
            half: if half.is_negative() {
                Fix128::ZERO
            } else {
                half
            },
            radius: config.radius,
            skin: config.skin_width,
            walkable: config.max_slope_angle.cos(),
            max_slides: config.max_slides.max(1),
        };
        // A capsule that cannot be freed keeps its position, and the sweep from
        // inside then blocks the move (module doc § Starting overlap).
        let start = sweep
            .depenetrate(ctrl.position, config.radius + config.height)
            .unwrap_or(ctrl.position);
        let total = displacement + ctrl.platform_velocity;
        let plain = sweep.slide(start, total);
        let mut position = plain.position;
        if ctrl.grounded && plain.steep && config.step_height > Fix128::ZERO {
            if let Some(stepped) = sweep.step(start, total, config.step_height, position) {
                position = stepped;
            }
        }

        let ground = sweep.ground(position, config.ground_probe_distance);
        let ground_body = ground.and_then(|h| h.body);
        let platform_velocity = ground_body
            .and_then(|i| self.bodies.get(i))
            .map_or(Vec3Fix::ZERO, |b| b.velocity);

        ctrl.position = position;
        ctrl.grounded = ground.is_some();
        ctrl.ground_body_index = ground_body;
        ctrl.platform_velocity = platform_velocity;
        ctrl.velocity = displacement;

        MoveResult {
            position,
            grounded: ctrl.grounded,
            velocity: displacement,
            platform_velocity,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Closed-form unit tests of the move: every expected position is written
    //! from the geometry by hand (the closed form is in a comment next to each
    //! assertion); none calls the code under test.
    //!
    //! Default config: radius `r = 0.3`, height `h = 1.8` (segment half `0.6`),
    //! skin `s = 0.01`, step height `0.3`, slope limit `0.785` rad
    //! (`cos = 0.7074`), 4 slides. A capsule standing a skin width above the
    //! floor `y = 0` has its centre at `h/2 + s = 0.91`.

    use super::*;
    use crate::plane_collider::PlaneCollider;
    use crate::shape::Shape;
    use crate::solver::{PhysicsConfig, RigidBody};
    use crate::static_collider::StaticCollider;

    const TOL: f64 = 1e-9;
    const ITER: f64 = 1e-7;

    fn fx(v: f64) -> Fix128 {
        Fix128::from_f64(v)
    }

    fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
        Vec3Fix::new(fx(x), fx(y), fx(z))
    }

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(PhysicsConfig::default())
    }

    fn plane(w: &mut PhysicsWorld, normal: Vec3Fix, offset: f64) -> usize {
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            normal,
            fx(offset),
        )))
    }

    #[track_caller]
    fn assert_pos(got: Vec3Fix, want: [f64; 3], tol: f64) {
        let g = [got.x.to_f64(), got.y.to_f64(), got.z.to_f64()];
        for k in 0..3 {
            assert!(
                (g[k] - want[k]).abs() < tol,
                "position {g:?} but the closed form is {want:?}"
            );
        }
    }

    #[test]
    fn a_negative_radius_skin_or_height_refuses_the_move() {
        let w = world();
        for k in 0..3 {
            let mut config = CharacterConfig::default();
            match k {
                0 => config.radius = fx(-0.3),
                1 => config.skin_width = fx(-0.01),
                _ => config.height = fx(-1.8),
            }
            let mut ctrl = CharacterController::new(v3(1.0, 2.0, 3.0), config);
            ctrl.velocity = v3(4.0, 0.0, 0.0);
            ctrl.grounded = true;
            ctrl.platform_velocity = v3(0.5, 0.0, 0.0);
            let r = w.move_character(&mut ctrl, v3(5.0, 0.0, 0.0));
            // oracle: refused, the result is the controller's current state.
            assert_eq!(r.position, v3(1.0, 2.0, 3.0));
            assert!(r.grounded);
            assert_eq!(r.velocity, v3(4.0, 0.0, 0.0));
            assert_eq!(r.platform_velocity, v3(0.5, 0.0, 0.0));
            assert_eq!(ctrl.position, v3(1.0, 2.0, 3.0));
            assert_eq!(ctrl.velocity, v3(4.0, 0.0, 0.0));
        }
    }

    #[test]
    fn a_free_move_takes_the_whole_displacement() {
        let w = world();
        let mut ctrl = CharacterController::new_default(v3(1.0, 5.0, -2.0));
        let r = w.move_character(&mut ctrl, v3(3.0, -1.0, 2.0));
        // oracle: nothing in the way: start + displacement, nothing under it.
        assert_pos(r.position, [4.0, 4.0, 0.0], TOL);
        assert!(!r.grounded);
        assert_eq!(r.velocity, v3(3.0, -1.0, 2.0));
        assert_eq!(r.platform_velocity, Vec3Fix::ZERO);
        assert_eq!(ctrl.position, r.position);
        assert_eq!(ctrl.ground_body_index, None);
    }

    #[test]
    fn walking_down_into_the_floor_keeps_the_tangential_part() {
        let mut w = world();
        plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        let mut ctrl = CharacterController::new_default(v3(0.0, 0.91, 0.0));
        let r = w.move_character(&mut ctrl, v3(1.0, -0.5, 2.0));
        // oracle: starting a skin width above the plane, D − n (D·n) with
        // n = +Y: (1, 0, 2) added, the height stays 0.91; the probe finds the
        // floor 0.01 below (a static collider: no ground body).
        assert_pos(r.position, [1.0, 0.91, 2.0], TOL);
        assert!(r.grounded);
        assert!(ctrl.grounded);
        assert_eq!(ctrl.ground_body_index, None);
        assert_eq!(r.platform_velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn a_wall_removes_the_normal_component() {
        let mut w = world();
        // The wall x = 2 (normal −X, offset −2).
        plane(&mut w, -Vec3Fix::UNIT_X, -2.0);
        let mut ctrl = CharacterController::new_default(Vec3Fix::ZERO);
        let r = w.move_character(&mut ctrl, v3(3.0, 0.0, 1.0));
        // oracle: the capsule stops a skin width from the wall, x = 2 − r − s =
        // 1.69; the tangential part of the whole displacement is kept, z = 1.
        assert_pos(r.position, [1.69, 0.0, 1.0], TOL);
        assert!(!r.grounded);

        // max_slides 0 is one sweep without a slide: the move along the unit
        // direction (3, 0, 1)/√10 stops at x = 1.69, z = 1.69/3.
        let config = CharacterConfig {
            max_slides: 0,
            ..CharacterConfig::default()
        };
        let mut ctrl = CharacterController::new(Vec3Fix::ZERO, config);
        let r = w.move_character(&mut ctrl, v3(3.0, 0.0, 1.0));
        assert_pos(r.position, [1.69, 0.0, 1.69 / 3.0], TOL);
    }

    #[test]
    fn a_steep_slope_is_a_wall_that_is_not_climbed() {
        let mut w = world();
        // The plane through the origin with normal n = (−√3/2, 1/2, 0), 60° from
        // horizontal: steeper than the slope limit (n·Y = 0.5 < 0.7074).
        let s3 = 3f64.sqrt();
        plane(&mut w, v3(-s3, 1.0, 0.0), 0.0);
        let mut ctrl = CharacterController::new_default(v3(-5.0, 0.0, 0.0));
        let r = w.move_character(&mut ctrl, v3(10.0, 0.0, 0.0));
        // oracle: the lower end (x, −0.6) has n·a = −√3/2·x − 0.3; the capsule is
        // a skin width from the plane when that is r + s = 0.31, x = −1.22/√3.
        // The rest projected on the plane would climb (y > 0), so it is projected
        // on the horizontal normal −X instead: 0. No walkable ground.
        assert_pos(r.position, [-1.22 / s3, 0.0, 0.0], TOL);
        assert!(!r.grounded);
    }

    #[test]
    fn a_low_step_is_climbed_when_grounded() {
        let mut w = world();
        // A box step: half extents (1, 0.1, 5) at (3, 0.1, 0), top y = 0.2,
        // front face x = 2 (below the 0.3 step height).
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: v3(1.0, 0.1, 5.0),
                },
                Fix128::ONE,
                v3(3.0, 0.1, 0.0),
            )
            .expect("valid shape");
        let mut ctrl = CharacterController::new_default(v3(0.0, 0.91, 0.0));
        ctrl.grounded = true;
        let r = w.move_character(&mut ctrl, v3(2.5, 0.0, 0.0));
        // oracle: raised by the step height, swept to x = 2.5 over the step,
        // lowered to a skin width above its top: y = 0.2 + 0.91.
        assert_pos(r.position, [2.5, 1.11, 0.0], ITER);
        assert!(r.grounded);
        assert_eq!(ctrl.ground_body_index, Some(b));

        // Not grounded before the move: no step. The lower hemisphere (centre
        // y = 0.31) meets the step's edge (2, 0.2), 0.11 below it, when its
        // centre is √(0.3² − 0.11²) = √0.0779 short of x = 2, normal
        // n = (−√0.0779, 0.11)/0.3; it backs off s/(−d·n) = 0.003/√0.0779, and
        // the rest, which would climb the steep normal, is cancelled.
        let mut ctrl = CharacterController::new_default(v3(0.0, 0.91, 0.0));
        let r = w.move_character(&mut ctrl, v3(2.5, 0.0, 0.0));
        let q = 0.0779f64.sqrt();
        assert_pos(r.position, [2.0 - q - 0.003 / q, 0.91, 0.0], ITER);
    }

    #[test]
    fn a_moving_platform_becomes_ground_and_carries_the_next_move() {
        let mut w = world();
        // A box platform: half extents (5, 0.5, 5) at the origin, top y = 0.5,
        // moving at (2, 0, 0).
        let b = w
            .add_shaped_body(
                &Shape::Box {
                    half_extents: v3(5.0, 0.5, 5.0),
                },
                Fix128::ONE,
                Vec3Fix::ZERO,
            )
            .expect("valid shape");
        w.bodies[b].velocity = v3(2.0, 0.0, 0.0);
        let mut ctrl = CharacterController::new_default(v3(0.0, 1.41, 0.0));
        let r = w.move_character(&mut ctrl, Vec3Fix::ZERO);
        // oracle: standing a skin width above the top (0.5 + 0.91): no move,
        // the platform is the ground and its velocity is inherited.
        assert_pos(r.position, [0.0, 1.41, 0.0], TOL);
        assert!(r.grounded);
        assert_eq!(ctrl.ground_body_index, Some(b));
        assert_eq!(r.platform_velocity, v3(2.0, 0.0, 0.0));
        // oracle: the next move adds the platform velocity: x = 0 + 2.
        let r = w.move_character(&mut ctrl, Vec3Fix::ZERO);
        assert_pos(r.position, [2.0, 1.41, 0.0], ITER);
        assert_eq!(r.velocity, Vec3Fix::ZERO);
    }

    #[test]
    fn a_capsule_sunk_into_the_floor_is_pushed_out_first() {
        let mut w = world();
        plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        let mut ctrl = CharacterController::new_default(v3(0.0, 0.8, 0.0));
        let r = w.move_character(&mut ctrl, Vec3Fix::ZERO);
        // oracle: the segment's lowest point y = 0.2 is 0.1 deeper than r: pushed
        // up by 0.1 + s, centre 0.91.
        assert_pos(r.position, [0.0, 0.91, 0.0], TOL);
        assert!(r.grounded);
    }

    #[test]
    fn a_capsule_whose_segment_crosses_a_solid_stays_put() {
        let mut w = world();
        plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        // The segment runs from y = −0.1 to 1.1 and crosses the plane.
        let mut ctrl = CharacterController::new_default(v3(0.0, 0.5, 0.0));
        let r = w.move_character(&mut ctrl, v3(1.0, 0.0, 0.0));
        // oracle: not freed; the cast starts overlapping (t = 0, normal
        // −direction): the move is 0 and its projection is 0; not ground.
        assert_pos(r.position, [0.0, 0.5, 0.0], TOL);
        assert!(!r.grounded);
    }

    #[test]
    fn a_capsule_between_walls_closer_than_its_diameter_is_not_freed() {
        let mut w = world();
        // Walls x = −0.25 and x = 0.25: a capsule of radius 0.3 at x = 0
        // overlaps both by 0.05, and every push out of one goes into the other.
        plane(&mut w, Vec3Fix::UNIT_X, -0.25);
        plane(&mut w, Vec3Fix::UNIT_X, 0.25);
        for max_slides in [4, 100] {
            // 4 slides: overlaps remain after the pushes; 100 slides: the pushes
            // (0.06, then 0.12 each) add up past r + h = 2.1.
            let config = CharacterConfig {
                max_slides,
                ..CharacterConfig::default()
            };
            let mut ctrl = CharacterController::new(Vec3Fix::ZERO, config);
            let r = w.move_character(&mut ctrl, v3(0.0, 0.0, 1.0));
            // oracle: not freed, the cast from inside blocks the move.
            assert_pos(r.position, [0.0, 0.0, 0.0], TOL);
        }
    }

    #[test]
    fn a_height_below_the_diameter_is_a_sphere() {
        let mut w = world();
        plane(&mut w, Vec3Fix::UNIT_Y, 0.0);
        let config = CharacterConfig {
            radius: fx(0.5),
            height: fx(0.4),
            ..CharacterConfig::default()
        };
        let mut ctrl = CharacterController::new(v3(0.0, 2.0, 0.0), config);
        let r = w.move_character(&mut ctrl, v3(0.0, -5.0, 0.0));
        // oracle: a sphere of radius 0.5 falls onto y = 0 and stops a skin width
        // above it: centre y = 0.5 + 0.01.
        assert_pos(r.position, [0.0, 0.51, 0.0], TOL);
        assert!(r.grounded);
    }

    #[test]
    fn the_filter_excludes_the_characters_own_body() {
        let mut w = world();
        // The character's own body: a sphere of radius 0.5 at its centre.
        let own = w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), fx(0.5));
        let mut ctrl = CharacterController::new_default(Vec3Fix::ZERO);
        let r = w.move_character(&mut ctrl, v3(1.0, 0.0, 0.0));
        // oracle: the segment crosses its own body: not freed, no move.
        assert_pos(r.position, [0.0, 0.0, 0.0], TOL);
        let r = w.move_character_with_filter(
            &mut ctrl,
            v3(1.0, 0.0, 0.0),
            &RayFilter::default().excluding_body(own),
        );
        // oracle: with its body excluded nothing is in the way: x = 1.
        assert_pos(r.position, [1.0, 0.0, 0.0], TOL);
    }

    #[test]
    fn short_of_and_project_closed_forms() {
        let w = world();
        let filter = RayFilter::default();
        let sweep = Sweep {
            world: &w,
            filter: &filter,
            half: fx(0.6),
            radius: fx(0.3),
            skin: fx(0.01),
            walkable: fx(0.5),
            max_slides: 4,
        };
        let d = Vec3Fix::UNIT_X;
        // oracle: head-on (−d·n = 1): t − s = 0.99.
        assert_pos(
            Vec3Fix::new(
                sweep.short_of(Fix128::ONE, d, -d),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            [0.99, 0.0, 0.0],
            TOL,
        );
        // oracle: at 60° (−d·n = 1/2): t − s/(1/2) = 1 − 0.02.
        let n = v3(-0.5, 0.75f64.sqrt(), 0.0);
        assert_pos(
            Vec3Fix::new(
                sweep.short_of(Fix128::ONE, d, n),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            [0.98, 0.0, 0.0],
            TOL,
        );
        // oracle: a normal not facing d backs off s; never below 0.
        assert_pos(
            Vec3Fix::new(
                sweep.short_of(Fix128::ONE, d, d),
                Fix128::ZERO,
                Fix128::ZERO,
            ),
            [0.99, 0.0, 0.0],
            TOL,
        );
        assert_eq!(sweep.short_of(fx(0.005), d, -d), Fix128::ZERO);
        // oracle: walkable when n·Y ≥ 0.5.
        assert!(sweep.is_walkable(Vec3Fix::UNIT_Y));
        assert!(!sweep.is_walkable(v3(0.9, 0.4359, 0.0)));
        // oracle: v − n (v·n): (3, 4, 5) without its Y part.
        assert_eq!(
            project(v3(3.0, 4.0, 5.0), Vec3Fix::UNIT_Y),
            v3(3.0, 0.0, 5.0)
        );
    }
}
