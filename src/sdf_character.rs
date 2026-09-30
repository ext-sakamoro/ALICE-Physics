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
        while iterations < self.max_iterations {
            let d = field.distance(pos[0], pos[1], pos[2]);
            if d >= self.radius {
                converged = true;
                break;
            }
            let (nx, ny, nz) = field.normal(pos[0], pos[1], pos[2]);
            // `max_push` is `INFINITY` by default, and `min` with
            // `INFINITY` returns the left operand unchanged, so an exact
            // distance field keeps the original arithmetic bit for bit.
            let push = (self.radius - d + self.skin_width).min(self.max_push);
            pos[0] += nx * push;
            pos[1] += ny * push;
            pos[2] += nz * push;
            iterations += 1;
        }
        MoveOutcome {
            position: pos,
            converged,
            iterations,
        }
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
