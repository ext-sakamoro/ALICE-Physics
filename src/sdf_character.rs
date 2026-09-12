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
//!    outside the field.
//! 4. Repeat steps 2-3 until either the sample is `>= radius` or the
//!    iteration budget is exhausted.
//! 5. Return the corrected position (regardless of whether all
//!    iterations converged; a caller wanting hard guarantees can
//!    inspect [`MoveOutcome::converged`]).
//!
//! # Ground detection
//!
//! [`SdfCharacter::is_grounded`] returns `true` when the SDF value a
//! short probe distance below the character is closer than
//! `ground_probe_epsilon` and the surface normal at that probe points
//! roughly upward (positive Y component ≥ `ground_up_threshold`).

use crate::sdf_collider::SdfField;

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
    /// Minimum vertical component of the ground normal for the
    /// character to be considered standing on that surface. Typical:
    /// 0.7 (≈ 45° slope).
    pub ground_up_threshold: f32,
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
            let push = self.radius - d + self.skin_width;
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

    /// `true` when the character stands on a surface with normal
    /// pointing predominantly upward.
    #[must_use]
    pub fn is_grounded<F: SdfField + ?Sized>(&self, field: &F) -> bool {
        let probe_y = self.position[1] - self.radius - self.ground_probe_epsilon;
        let d = field.distance(self.position[0], probe_y, self.position[2]);
        if d > self.ground_probe_epsilon {
            return false;
        }
        let (_nx, ny, _nz) = field.normal(self.position[0], probe_y, self.position[2]);
        ny >= self.ground_up_threshold
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
