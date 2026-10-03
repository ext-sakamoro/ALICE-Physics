//! SDF-swept kinematic character controller.
//!
//! Alternative to [`crate::character`] (trimesh / capsule move-and-slide)
//! that treats the world as a signed distance field. The character is a
//! vertical capsule (`radius`, `height`); its position advances by a
//! sequence of penetration-resolving surface projections instead of
//! swept-shape collision against a triangle soup.
//!
//! The controller is deterministic in the sense that it uses `f32`
//! throughout (matching [`crate::sdf_collider::SdfField`]), so it can
//! be driven by ALICE-SDF's compiled fields for level geometry with a
//! stable evaluation cost.
//!
//! # Move-and-slide algorithm
//!
//! 1. Advance the tentative position by the full desired displacement.
//! 2. Sample the SDF at the character's centre. If the sample is
//!    smaller than `radius`, the character is penetrating.
//! 3. Push the character out along the surface normal by
//!    `radius − sample`, rounding up to
//!    `radius − sample + skin_width` so the next query starts safely
//!    outside the field, and capping the step at `max_push`.
//! 4. Repeat steps 2-3 until either the sample is `>= radius` or the
//!    iteration budget is exhausted.
//! 5. Return the corrected position (regardless of whether all
//!    iterations converged; a caller wanting hard guarantees can
//!    inspect [`MoveOutcome::converged`]).
//!
//! # Fields that are not exact distance fields
//!
//! Step 3 treats the sample as a true distance. For a field with
//! `|∇f| = L > 1` — gyroid walls and many other implicit surfaces are in
//! that class — the sample **overstates** the penetration by up to `L`,
//! so the full push can clear the free gap and land inside the next
//! sheet of geometry, i.e. the character tunnels through a wall.
//!
//! ⚠️ [`MoveOutcome::converged`] does **not** rule this out. The hop
//! length depends on the penetration depth, so the loop keeps jumping
//! until one hop happens to land in a gap and then reports success — in
//! a pocket the character never legally reached. On the periodic-sheet
//! field of `tests/analytic_sdf_character_up_axis.rs` it converges on
//! iteration 5, two and a half periods away, having crossed three solid
//! sheets.
//!
//! [`SdfCharacter::max_push`] caps one iteration's step for exactly that
//! case: with the cap below the free gap's width the character walks out
//! into the adjacent pocket instead of jumping past it. It defaults to
//! [`f32::INFINITY`], so an exact distance field keeps the original
//! single-step arithmetic bit for bit.
//!
//! Two more consequences of an inexact field, both opt-in for the same
//! bit-compatibility reason:
//!
//! - **The normal can face the centre of the world.** Geometry buried in
//!   the ground reports that at the seam, and pushing along it drives the
//!   character underground. [`SdfCharacter::min_up_alignment`] rejects
//!   those normals and pushes along [`SdfCharacter::up`] instead, at the
//!   same magnitude. Default [`f32::NEG_INFINITY`] never substitutes.
//! - **A push can land deeper than it started.** The run therefore
//!   records the least-penetrating sample it saw in
//!   [`MoveOutcome::best_position`] / [`MoveOutcome::best_distance`],
//!   including the position produced by the final push. A caller that
//!   must never return a worse position than it was handed uses those
//!   when [`MoveOutcome::converged`] is `false`. This is reporting only —
//!   [`MoveOutcome::position`] keeps its old value.
//!
//! # Ground detection
//!
//! [`SdfCharacter::ground_contact`] probes a short distance along
//! `−up` and reports the sampled distance together with the surface
//! normal and its alignment to `up`;
//! [`SdfCharacter::is_grounded`] thresholds that alignment against
//! `ground_up_threshold`.
//!
//! [`SdfCharacter::up`] defaults to `+Y`, which is what a level with a
//! single gravity direction wants. A sphere world's up axis is
//! **radial** (`p / |p|`) and differs per position, so those consumers
//! set `up` every frame — with a fixed `+Y` probe a character standing
//! on the equator of a planet reports "not grounded" while standing on
//! the ground.
//!
//! # One frame
//!
//! [`SdfCharacter::velocity`] plus [`SdfCharacter::apply_gravity`] and
//! [`SdfCharacter::step`] cover a whole frame, so a ballistic fall and
//! its landing live here rather than in each consumer:
//!
//! ```
//! # use alice_physics::sdf_character::SdfCharacter;
//! # use alice_physics::sdf_collider::ClosureSdf;
//! # let field = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
//! let mut ch = SdfCharacter::new([0.0, 4.0, 0.0], 0.35, 1.8);
//! let dt = 1.0 / 60.0;
//! for _ in 0..240 {
//!     ch.apply_gravity([0.0, -9.81, 0.0], dt);
//!     // `control` is the frame's own locomotion, already scaled by dt.
//!     ch.step(&field, dt, [0.0, 0.0, 0.0]);
//! }
//! // Resting on the surface, with the fall absorbed by the contact.
//! assert!((ch.position[1] - (0.35 + ch.skin_width)).abs() < 1.0e-3);
//! assert!(ch.velocity[1].abs() < 1.0e-6);
//! ```
//!
//! [`SdfCharacter::step`] removes only the velocity component pointing
//! **into** what was hit, so the character slides along a wall rather
//! than sticking to it, and a jump that brushes the floor keeps its
//! upward speed.

use crate::character_state::CharacterStateContext;
use crate::sdf_collider::SdfField;

/// What the ground probe found below the character.
///
/// Returned by [`SdfCharacter::ground_contact`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundContact {
    /// SDF value at the probe point. Negative when the probe is inside
    /// the surface, which is the common case for a character resting on
    /// the ground.
    pub distance: f32,
    /// Unit surface normal at the probe point, as the field reports it.
    pub normal: [f32; 3],
    /// `normal · up` with `up` normalized — the quantity
    /// [`SdfCharacter::is_grounded`] compares against
    /// `ground_up_threshold`. `1.0` is a surface square to the up axis.
    pub up_alignment: f32,
}

/// Result of an SDF move-and-slide call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoveOutcome {
    /// Final character centre after penetration resolution.
    pub position: [f32; 3],
    /// True when the last sampled distance was already `>= radius` (no
    /// residual penetration).
    pub converged: bool,
    /// Number of penetration-resolution iterations that ran (0 when
    /// the initial displacement was already collision-free).
    pub iterations: usize,
    /// Position of the least-penetrating sample the run saw, including
    /// the starting one.
    ///
    /// Equal to [`Self::position`] whenever the run converged, and
    /// whenever the samples improved monotonically. They differ on
    /// fields that are not exact distance fields, where a push can land
    /// **deeper** than where it started — see the module docs. A caller
    /// that must never hand back a worse position than it was given uses
    /// this field when [`Self::converged`] is `false`.
    pub best_position: [f32; 3],
    /// The sampled distance at [`Self::best_position`].
    pub best_distance: f32,
}

impl MoveOutcome {
    /// The position a caller should adopt: [`Self::position`] when the
    /// resolution converged, [`Self::best_position`] otherwise.
    ///
    /// Reading [`Self::position`] unconditionally hands back a point that
    /// can be **deeper than the one the caller passed in** on a field that
    /// is not an exact distance field — see the module docs. This is the
    /// safe read, in one call, so consumers do not each rewrite the same
    /// conditional.
    #[must_use]
    pub const fn resolved_position(&self) -> [f32; 3] {
        if self.converged {
            self.position
        } else {
            self.best_position
        }
    }
}

/// Vertical-capsule character driven by SDF-swept move-and-slide.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfCharacter {
    /// Centre of the character.
    pub position: [f32; 3],
    /// Capsule radius (m).
    pub radius: f32,
    /// Total capsule height (m). Currently unused by the MVP move
    /// algorithm but preserved for downstream systems that need the
    /// character AABB.
    pub height: f32,
    /// Maximum penetration-resolution iterations. Typical value: 8.
    pub max_iterations: usize,
    /// Small offset added on top of the pushed-out distance so the
    /// next SDF query is safely outside the surface. Typical: 1e-4 m.
    pub skin_width: f32,
    /// SDF distance below the character considered "grounded". Typical
    /// value: 0.05 m.
    pub ground_probe_epsilon: f32,
    /// Minimum component of the ground normal along [`Self::up`] for the
    /// character to be considered standing on that surface. Typical:
    /// 0.7 (≈ 45° slope).
    pub ground_up_threshold: f32,
    /// Up axis the ground probe walks against, in world space. Need not
    /// be normalized (it is normalized on use); a zero or non-finite
    /// axis falls back to `+Y`.
    ///
    /// Defaults to `+Y`. A sphere world sets this to the radial
    /// direction `p / |p|` each frame.
    pub up: [f32; 3],
    /// Upper bound on one penetration-resolution step (m).
    ///
    /// Defaults to [`f32::INFINITY`] — no clamping, which keeps the
    /// arithmetic of an exact distance field unchanged. Set it for
    /// fields whose gradient magnitude exceeds 1 (gyroid walls and
    /// similar), where an unclamped push can jump over the free gap into
    /// the next sheet of geometry; see the module docs.
    pub max_push: f32,
    /// Minimum `normal · up` for the field's normal to be used as the
    /// push direction. Below it the push runs along [`Self::up`] instead,
    /// at the same magnitude.
    ///
    /// Defaults to [`f32::NEG_INFINITY`] — the normal is always taken, so
    /// the original arithmetic is untouched (and [`Self::up`] is not even
    /// normalized, so the default path costs nothing).
    ///
    /// Geometry buried in the ground produces normals that face the
    /// centre of the world at the seam, and pushing along one drives the
    /// character underground rather than out of the surface. `-0.2` keeps
    /// sideways pushes (a wall has `normal · up = 0`) while rejecting
    /// those.
    pub min_up_alignment: f32,
    /// Kinematic velocity (m/s, world space), integrated by
    /// [`Self::apply_gravity`] and consumed by [`Self::step`].
    ///
    /// Defaults to zero. [`crate::character::CharacterController`] has
    /// carried a velocity since 1.0; this is the SDF controller's
    /// equivalent, so a ballistic fall and its landing are expressed by
    /// the controller rather than by each consumer.
    pub velocity: [f32; 3],
}

impl Default for SdfCharacter {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            radius: 0.35,
            height: 1.8,
            max_iterations: 8,
            skin_width: 1.0e-4,
            ground_probe_epsilon: 5.0e-2,
            ground_up_threshold: 0.7,
            up: [0.0, 1.0, 0.0],
            max_push: f32::INFINITY,
            min_up_alignment: f32::NEG_INFINITY,
            velocity: [0.0, 0.0, 0.0],
        }
    }
}

impl SdfCharacter {
    /// Constructor with sensible defaults.
    #[must_use]
    pub fn new(position: [f32; 3], radius: f32, height: f32) -> Self {
        Self {
            position,
            radius,
            height,
            ..Self::default()
        }
    }

    /// Move the character by `displacement` and resolve any resulting
    /// SDF penetration.
    ///
    /// Returns the outcome without mutating `self`. Callers who want
    /// the character to remember the new position should assign
    /// `outcome.position` back to `self.position`.
    #[must_use]
    pub fn move_and_slide<F: SdfField + ?Sized>(
        &self,
        field: &F,
        displacement: [f32; 3],
    ) -> MoveOutcome {
        let mut pos = [
            self.position[0] + displacement[0],
            self.position[1] + displacement[1],
            self.position[2] + displacement[2],
        ];
        let mut iterations = 0;
        let mut converged = false;
        // `min_up_alignment` is `NEG_INFINITY` by default; skipping the
        // normalization entirely in that case keeps the default path free
        // of the `sqrt` as well as of the substitution.
        let guard_up = if self.min_up_alignment.is_finite() {
            Some(self.up_unit())
        } else {
            None
        };
        let mut best_position = pos;
        let mut best_distance = f32::NEG_INFINITY;
        while iterations < self.max_iterations {
            let d = field.distance(pos[0], pos[1], pos[2]);
            if d > best_distance {
                best_distance = d;
                best_position = pos;
            }
            if d >= self.radius {
                converged = true;
                break;
            }
            let (mut nx, mut ny, mut nz) = field.normal(pos[0], pos[1], pos[2]);
            if let Some(up) = guard_up {
                // A normal facing the centre of the world would push the
                // character further in; run along `up` at the same
                // magnitude instead.
                if nx * up[0] + ny * up[1] + nz * up[2] < self.min_up_alignment {
                    [nx, ny, nz] = up;
                }
            }
            // `max_push` is `INFINITY` by default, and `min` with
            // `INFINITY` returns the left operand unchanged, so an exact
            // distance field keeps the original arithmetic bit for bit.
            let push = (self.radius - d + self.skin_width).min(self.max_push);
            pos[0] += nx * push;
            pos[1] += ny * push;
            pos[2] += nz * push;
            iterations += 1;
        }
        if !converged {
            // The loop only samples at its top, so the position produced
            // by the last push has not been measured yet. Without this
            // the budget-exhausted case would compare `best` against one
            // sample too few — and that final push is exactly the one
            // that can have made things worse.
            let d = field.distance(pos[0], pos[1], pos[2]);
            if d > best_distance {
                best_distance = d;
                best_position = pos;
            }
        }
        MoveOutcome {
            position: pos,
            converged,
            iterations,
            best_position,
            best_distance,
        }
    }

    /// Integrate an acceleration into [`Self::velocity`] (`v += a·dt`).
    ///
    /// Unconditional: gating it on whether the character is standing is
    /// the caller's decision (a hovering or flying mode overrides the
    /// radial velocity outright rather than accumulating).
    pub fn apply_gravity(&mut self, gravity: [f32; 3], dt: f32) {
        self.velocity[0] += gravity[0] * dt;
        self.velocity[1] += gravity[1] * dt;
        self.velocity[2] += gravity[2] * dt;
    }

    /// Advance one frame: integrate the velocity, resolve the resulting
    /// penetration, adopt the resulting position, and remove the part of
    /// the velocity that points into whatever was hit.
    ///
    /// `control` is the caller's own displacement for this frame (tangent
    /// locomotion, a teleport nudge, …) and is added to `velocity · dt`.
    ///
    /// The position adopted is [`MoveOutcome::position`] when the
    /// resolution converged and [`MoveOutcome::best_position`] otherwise,
    /// so a frame can never end deeper than it started on a field that is
    /// not an exact distance field.
    ///
    /// # Contact response
    ///
    /// The resolution's net correction gives the contact direction. Only
    /// the velocity component pointing **into** the surface is removed,
    /// which is the inelastic kinematic law: landing stops the fall and
    /// leaves the tangential motion, so the character slides along a wall
    /// instead of sticking to it. Velocity already pointing away from the
    /// surface is untouched, so brushing the floor mid-jump does not eat
    /// the jump.
    pub fn step<F: SdfField + ?Sized>(
        &mut self,
        field: &F,
        dt: f32,
        control: [f32; 3],
    ) -> MoveOutcome {
        let displacement = [
            self.velocity[0] * dt + control[0],
            self.velocity[1] * dt + control[1],
            self.velocity[2] * dt + control[2],
        ];
        let unresolved = [
            self.position[0] + displacement[0],
            self.position[1] + displacement[1],
            self.position[2] + displacement[2],
        ];
        let outcome = self.move_and_slide(field, displacement);
        let adopted = outcome.resolved_position();
        self.position = adopted;

        let correction = [
            adopted[0] - unresolved[0],
            adopted[1] - unresolved[1],
            adopted[2] - unresolved[2],
        ];
        let len = crate::det_math::sqrt(
            correction[0] * correction[0]
                + correction[1] * correction[1]
                + correction[2] * correction[2],
        );
        // No correction means no contact, so the velocity is untouched.
        // The threshold is `skin_width` because a converged resolution
        // always overshoots by at least that much.
        if len > self.skin_width {
            let n = [
                correction[0] / len,
                correction[1] / len,
                correction[2] / len,
            ];
            let into = self.velocity[0] * n[0] + self.velocity[1] * n[1] + self.velocity[2] * n[2];
            if into < 0.0 {
                self.velocity[0] -= n[0] * into;
                self.velocity[1] -= n[1] * into;
                self.velocity[2] -= n[2] * into;
            }
        }
        outcome
    }

    /// [`Self::up`] normalized, falling back to `+Y` for a zero or
    /// non-finite axis so the probe can never produce `NaN` coordinates.
    ///
    /// `+Y` (the default) normalizes to itself exactly: the length is
    /// `1.0` and dividing by `1.0` is exact in IEEE-754.
    ///
    /// The length goes through [`crate::det_math::sqrt`] rather than
    /// `f32::sqrt` — this module is not `std`-gated, and the platform
    /// libm is not cross-platform bit-exact.
    #[must_use]
    fn up_unit(&self) -> [f32; 3] {
        let [x, y, z] = self.up;
        let len = crate::det_math::sqrt(x * x + y * y + z * z);
        if len > 0.0 && len.is_finite() {
            [x / len, y / len, z / len]
        } else {
            [0.0, 1.0, 0.0]
        }
    }

    /// Probe `radius + ground_probe_epsilon` along `−up` and report what
    /// the field says there.
    ///
    /// `None` when the probe point is farther than
    /// `ground_probe_epsilon` from any surface (the character is in the
    /// air). The alignment test itself is left to the caller, or to
    /// [`Self::is_grounded`].
    #[must_use]
    pub fn ground_contact<F: SdfField + ?Sized>(&self, field: &F) -> Option<GroundContact> {
        let up = self.up_unit();
        let reach = self.radius + self.ground_probe_epsilon;
        let probe = [
            self.position[0] - up[0] * reach,
            self.position[1] - up[1] * reach,
            self.position[2] - up[2] * reach,
        ];
        let distance = field.distance(probe[0], probe[1], probe[2]);
        if distance > self.ground_probe_epsilon {
            return None;
        }
        let (nx, ny, nz) = field.normal(probe[0], probe[1], probe[2]);
        Some(GroundContact {
            distance,
            normal: [nx, ny, nz],
            up_alignment: nx * up[0] + ny * up[1] + nz * up[2],
        })
    }

    /// Build the per-tick sensor readings for [`crate::character_state::transition`]
    /// from the ground probe.
    ///
    /// `is_grounded` is "the probe found a surface" (any contact from
    /// [`Self::ground_contact`], not the [`Self::ground_up_threshold`]
    /// test), and `slope_radians` is the angle between the contact normal
    /// and [`Self::up`], `acos(up_alignment)`. The state machine, not
    /// [`Self::is_grounded`], then decides between standing and sliding with
    /// `max_walkable_slope`, so a steep contact yields `Sliding` instead of
    /// being reported as airborne. With no contact the slope is `0` and
    /// `is_grounded` is `false`.
    ///
    /// `in_water`, `crouch_requested` and `jump_pressed` are the caller's own
    /// inputs and are copied through.
    #[must_use]
    pub fn locomotion_context<F: SdfField + ?Sized>(
        &self,
        field: &F,
        in_water: bool,
        crouch_requested: bool,
        jump_pressed: bool,
        max_walkable_slope: f32,
    ) -> CharacterStateContext {
        let (is_grounded, slope_radians) = match self.ground_contact(field) {
            Some(c) => (true, crate::det_math::acos(c.up_alignment.clamp(-1.0, 1.0))),
            None => (false, 0.0),
        };
        CharacterStateContext {
            is_grounded,
            slope_radians,
            in_water,
            crouch_requested,
            jump_pressed,
            max_walkable_slope,
        }
    }

    /// `true` when the character stands on a surface whose normal points
    /// predominantly along [`Self::up`].
    #[must_use]
    pub fn is_grounded<F: SdfField + ?Sized>(&self, field: &F) -> bool {
        self.ground_contact(field)
            .is_some_and(|c| c.up_alignment >= self.ground_up_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    fn ground_plane() -> ClosureSdf {
        // Infinite plane at y = 0, positive above.
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
                (x / len, y / len, z / len)
            },
        )
    }

    #[test]
    fn move_in_free_space_leaves_character_unchanged() {
        let field = ground_plane();
        let character = SdfCharacter::new([0.0, 2.0, 0.0], 0.35, 1.8);
        let outcome = character.move_and_slide(&field, [1.0, 0.0, 0.0]);
        assert!(outcome.converged);
        assert_eq!(outcome.iterations, 0);
        assert!((outcome.position[0] - 1.0).abs() < 1.0e-6);
        assert!((outcome.position[1] - 2.0).abs() < 1.0e-6);
    }

    #[test]
    fn character_pushed_out_of_plane() {
        let field = ground_plane();
        // Character radius 0.35 sitting with centre at y = 0.1 →
        // penetrating the ground.
        let character = SdfCharacter::new([0.0, 0.1, 0.0], 0.35, 1.8);
        let outcome = character.move_and_slide(&field, [0.0, 0.0, 0.0]);
        assert!(outcome.converged);
        // Final centre should be at y ≥ radius.
        assert!(
            outcome.position[1] >= character.radius,
            "y = {} < radius {}",
            outcome.position[1],
            character.radius
        );
    }

    #[test]
    fn character_pushed_off_sphere_surface() {
        let field = unit_sphere();
        // Character penetrating a unit sphere at the +x pole.
        let character = SdfCharacter::new([0.9, 0.0, 0.0], 0.35, 1.8);
        let outcome = character.move_and_slide(&field, [0.0, 0.0, 0.0]);
        // After resolution the centre should sit sphere radius +
        // character radius outward.
        let dist = (outcome.position[0] * outcome.position[0]
            + outcome.position[1] * outcome.position[1]
            + outcome.position[2] * outcome.position[2])
            .sqrt();
        assert!(
            dist >= 1.0 + character.radius - 1.0e-3,
            "expected distance ≥ 1.35, got {dist}"
        );
    }

    #[test]
    fn is_grounded_true_on_plane() {
        let field = ground_plane();
        let character = SdfCharacter::new([0.0, 0.35, 0.0], 0.35, 1.8);
        assert!(character.is_grounded(&field));
    }

    #[test]
    fn is_grounded_false_in_air() {
        let field = ground_plane();
        let character = SdfCharacter::new([0.0, 5.0, 0.0], 0.35, 1.8);
        assert!(!character.is_grounded(&field));
    }

    #[test]
    fn is_grounded_false_on_steep_slope() {
        // A "slope" field: surface at y = x (45°), gradient tilted.
        let slope = ClosureSdf::new(
            |x, y, _z| (y - x) / 2.0_f32.sqrt(),
            |_x, _y, _z| (-1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt(), 0.0),
        );
        // ny = 1/√2 ≈ 0.707, at the default ground_up_threshold = 0.7 the
        // slope should be borderline; make the threshold stricter.
        let mut character = SdfCharacter::new([0.5, 0.85, 0.0], 0.35, 1.8);
        character.ground_up_threshold = 0.9;
        assert!(!character.is_grounded(&slope));
    }

    #[test]
    fn default_config_is_reasonable() {
        let c = SdfCharacter::default();
        assert!(c.radius > 0.0);
        assert!(c.height > c.radius);
        assert!(c.max_iterations > 0);
    }

    #[test]
    fn zero_iterations_when_no_penetration_after_move() {
        let field = ground_plane();
        let character = SdfCharacter::new([0.0, 5.0, 0.0], 0.35, 1.8);
        let outcome = character.move_and_slide(&field, [1.0, 0.0, 0.0]);
        assert_eq!(outcome.iterations, 0);
        assert!(outcome.converged);
    }
}
