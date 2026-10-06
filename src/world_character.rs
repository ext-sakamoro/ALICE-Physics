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
//! # Starting inside a collider
//!
//! [`PhysicsWorld::cast_capsule`] reports a cast that starts overlapping a
//! collider at `t = 0` with normal `−direction`: there is no surface normal to
//! slide along. The move then goes 0 along the direction and its projection onto
//! the plane normal to the direction is 0, so the controller does not move; such
//! a contact is not ground, and no step is tried from inside (the upward sweep
//! of § Steps also starts overlapping).
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
use crate::world_shape_query::WorldShapeHit;

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
            max_slides: config.max_slides,
        };
        let start = ctrl.position;
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
