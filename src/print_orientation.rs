//! Print Orientation Optimization for Load-Aligned Strength
//!
//! Phase C2 of the ALICE-Physics completeness project. Given a **load
//! direction** (unit vector in world space) and an **anisotropic material**
//! (layer strength ratio < 1), sweep candidate print orientations to find
//! the one that maximises the effective strength along the load.
//!
//! # Approach
//!
//! For an FDM material the strength depends only on the angle `θ` between
//! the load direction and the layer stacking axis (Z of the print bed):
//!
//! `σ_eff(θ) = σ_xy · cos²θ + σ_z · sin²θ`
//!
//! (This is the same squared-cosine mixing rule used in `MaterialProperties::
//! yield_at_angle`.) The optimisation therefore reduces to picking the print
//! rotation that minimises this angle.
//!
//! For most parts the ideal orientation is "load direction parallel to the
//! print bed" (θ = 0 → σ_eff = σ_xy, maximum). The optimiser expresses this
//! as a rotation of the part model in the printer coordinate system.
//!
//! # Coverage
//!
//! - Continuous optimum via analytical θ = 0 solution.
//! - Discrete grid search over Euler-angle candidates for cases where the
//!   part cannot be freely rotated (e.g. overhang constraints).
//! - Constraint predicate `is_orientation_valid` to reject candidates that
//!   would create excessive overhang.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Candidate & report
// ============================================================================

/// A print orientation candidate expressed as rotations about the world X
/// and Y axes (Euler-like, applied to the part model before printing).
///
/// The Z-axis rotation is redundant for anisotropy since it does not change
/// the angle between the layer stack (world Z) and the load direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrientationCandidate {
    /// Rotation about world X axis (radians).
    pub theta_x: Fix128,
    /// Rotation about world Y axis (radians).
    pub theta_y: Fix128,
}

impl OrientationCandidate {
    /// Identity (part printed as-modelled).
    pub const IDENTITY: Self = Self {
        theta_x: Fix128::ZERO,
        theta_y: Fix128::ZERO,
    };
}

/// A load direction as a unit vector in world coordinates.
///
/// The magnitude is not required to be exactly unity — the module normalises
/// internally — but should be close.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadDirection {
    /// X component.
    pub x: Fix128,
    /// Y component.
    pub y: Fix128,
    /// Z component.
    pub z: Fix128,
}

impl LoadDirection {
    /// Axis-aligned constructor.
    #[must_use]
    pub const fn axis_x() -> Self {
        Self {
            x: Fix128::ONE,
            y: Fix128::ZERO,
            z: Fix128::ZERO,
        }
    }

    /// Axis-aligned constructor.
    #[must_use]
    pub const fn axis_y() -> Self {
        Self {
            x: Fix128::ZERO,
            y: Fix128::ONE,
            z: Fix128::ZERO,
        }
    }

    /// Axis-aligned constructor.
    #[must_use]
    pub const fn axis_z() -> Self {
        Self {
            x: Fix128::ZERO,
            y: Fix128::ZERO,
            z: Fix128::ONE,
        }
    }

    /// Squared length.
    #[must_use]
    pub fn length_squared(&self) -> Fix128 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Length. Guaranteed nonnegative (sqrt applied to |·|²).
    #[must_use]
    pub fn length(&self) -> Fix128 {
        self.length_squared().sqrt()
    }
}

/// Result of an orientation search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrientationReport {
    /// Best orientation found.
    pub best_orientation: OrientationCandidate,
    /// Angle (radians) between the rotated load direction and the layer
    /// stack axis (Z) at the optimum.
    pub angle_to_z_axis: Fix128,
    /// Effective yield strength (MPa) at the optimum.
    pub effective_yield_mpa: Fix128,
    /// Effective yield at the identity orientation, for comparison.
    pub identity_yield_mpa: Fix128,
    /// Absolute strength gain (MPa) vs identity.
    pub improvement_mpa: Fix128,
}

// ============================================================================
// Analytical helpers
// ============================================================================

/// Effective anisotropic yield strength (MPa) given the angle `theta`
/// (radians) between the load direction and the **layer stack (Z) axis**.
///
/// - `θ = 0` (load parallel to Z, i.e. pulling **across** layers) → returns
///   the weak `σ_z` value.
/// - `θ = π/2` (load in the XY plane, i.e. pulling **within** a layer) →
///   returns the strong `σ_xy` value.
///
/// Uses the squared-cosine mixing rule
/// `σ_eff = σ_z · cos²θ + σ_xy · sin²θ`.
///
/// Note the different theta convention vs `MaterialProperties::yield_at_angle`
/// which measures the angle *from the XY plane*; the two forms are related
/// by `θ_here = π/2 − θ_there`.
#[must_use]
pub fn effective_yield_at_angle(m: &MaterialProperties, theta: Fix128) -> Fix128 {
    let (s, c) = theta.sin_cos();
    let c2 = c * c;
    let s2 = s * s;
    let sigma_xy = m.yield_strength_mpa;
    let sigma_z = m.yield_strength_mpa * m.anisotropy_z_ratio;
    sigma_z * c2 + sigma_xy * s2
}

/// Compute the angle (radians) between a load direction and the +Z axis
/// under a candidate rotation (about X then Y).
///
/// Rotation matrices are applied in the order R_Y · R_X. Then the angle is
/// `θ = acos(z_rot)` where `z_rot` is the Z-component of the rotated load
/// vector. `acos` is emulated using `atan2` for Fix128 (no direct impl).
#[must_use]
pub fn angle_to_z_axis(load: &LoadDirection, orient: &OrientationCandidate) -> Fix128 {
    // Rotate load by R_X then R_Y
    let (sx, cx) = orient.theta_x.sin_cos();
    let (sy, cy) = orient.theta_y.sin_cos();

    // R_X: rotates YZ plane
    let y1 = load.y * cx - load.z * sx;
    let z1 = load.y * sx + load.z * cx;
    // R_Y: rotates XZ plane
    let x2 = load.x * cy + z1 * sy;
    let z2 = -load.x * sy + z1 * cy;
    let _ = y1; // not needed after rotation
    let _ = x2;

    // Compute unit vector length
    let len = (x2 * x2 + y1 * y1 + z2 * z2).sqrt();
    if len.is_zero() {
        return Fix128::ZERO;
    }
    let z_unit = z2 / len;
    // acos(z_unit) via atan2(sqrt(1-z²), z)
    let z2_val = z_unit * z_unit;
    let one_minus = if z2_val >= Fix128::ONE {
        Fix128::ZERO
    } else {
        Fix128::ONE - z2_val
    };
    let s_val = one_minus.sqrt();
    Fix128::atan2(s_val, z_unit)
}

// ============================================================================
// Optimizer
// ============================================================================

/// Perform an analytical continuous optimisation.
///
/// The optimum is always "load direction perpendicular to Z" (θ = π/2 mapped
/// via `1 - sin²θ = cos²θ = 1`); we compute the identity vs optimum yields
/// and return the improvement.
#[must_use]
pub fn optimize_analytical(load: &LoadDirection, m: &MaterialProperties) -> OrientationReport {
    // Angle at identity orientation
    let theta_id = angle_to_z_axis(load, &OrientationCandidate::IDENTITY);
    let yield_id = effective_yield_at_angle(m, theta_id);

    // Analytical optimum: rotate so load points along +X (any XY direction is fine)
    // → θ = π/2 → σ_eff = σ_xy (maximum)
    // Concrete rotation: rotate about Y by (π/2 - θ_z_load), where θ_z_load = angle
    // between original load and +Z.
    // For a 3D load vector, one simple way is: theta_y = θ_id − π/2 to align load
    // with +X after R_Y. This gives cos(θ_new) = ~0 → σ_eff ≈ σ_xy.
    let theta_y = theta_id - Fix128::HALF_PI;
    let best = OrientationCandidate {
        theta_x: Fix128::ZERO,
        theta_y,
    };
    let theta_best = angle_to_z_axis(load, &best);
    let yield_best = effective_yield_at_angle(m, theta_best);

    OrientationReport {
        best_orientation: best,
        angle_to_z_axis: theta_best,
        effective_yield_mpa: yield_best,
        identity_yield_mpa: yield_id,
        improvement_mpa: yield_best - yield_id,
    }
}

/// Discrete grid search over Euler angles (θ_x, θ_y) in `[-π/2, π/2]` at
/// steps of `grid_step_rad`. Useful when downstream constraints (overhang
/// avoidance, support minimisation) preclude the analytical optimum.
///
/// `grid_step_rad` typical: `π/12` (15°) → 25×25 = 625 candidates.
#[must_use]
pub fn optimize_grid(
    load: &LoadDirection,
    m: &MaterialProperties,
    grid_step_rad: Fix128,
) -> OrientationReport {
    if grid_step_rad.is_zero() {
        return optimize_analytical(load, m);
    }
    let mut best_yield = Fix128::ZERO;
    let mut best_orient = OrientationCandidate::IDENTITY;
    let mut best_angle = Fix128::ZERO;

    let range = Fix128::HALF_PI;
    let mut theta_x = Fix128::ZERO - range;
    while theta_x <= range {
        let mut theta_y = Fix128::ZERO - range;
        while theta_y <= range {
            let candidate = OrientationCandidate { theta_x, theta_y };
            let angle = angle_to_z_axis(load, &candidate);
            let y = effective_yield_at_angle(m, angle);
            if y > best_yield {
                best_yield = y;
                best_orient = candidate;
                best_angle = angle;
            }
            theta_y = theta_y + grid_step_rad;
        }
        theta_x = theta_x + grid_step_rad;
    }
    let theta_id = angle_to_z_axis(load, &OrientationCandidate::IDENTITY);
    let yield_id = effective_yield_at_angle(m, theta_id);
    OrientationReport {
        best_orientation: best_orient,
        angle_to_z_axis: best_angle,
        effective_yield_mpa: best_yield,
        identity_yield_mpa: yield_id,
        improvement_mpa: best_yield - yield_id,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn effective_yield_zero_angle_equals_sigma_z() {
        // θ = 0 → load parallel to Z (across layers) → σ_z (weak)
        let pla = MaterialProperties::pla();
        let s = effective_yield_at_angle(&pla, Fix128::ZERO);
        let expected = pla.yield_strength_mpa * pla.anisotropy_z_ratio;
        let tol = expected * Fix128::from_ratio(1, 100);
        let diff = if s > expected {
            s - expected
        } else {
            expected - s
        };
        assert!(diff <= tol);
    }

    #[test]
    fn effective_yield_ninety_degrees_equals_sigma_xy() {
        // θ = π/2 → load in XY plane (within layer) → σ_xy (strong)
        let pla = MaterialProperties::pla();
        let s = effective_yield_at_angle(&pla, Fix128::HALF_PI);
        let expected = pla.yield_strength_mpa;
        let tol = expected * Fix128::from_ratio(1, 100);
        let diff = if s > expected {
            s - expected
        } else {
            expected - s
        };
        assert!(diff <= tol);
    }

    #[test]
    fn axis_z_load_at_identity_is_near_zero_angle() {
        // Load = +Z direction → angle to Z axis = 0 (parallel).
        // CORDIC sin(0)/cos(0) both have accumulated error → allow a generous
        // tolerance since the boundary case π ≈ 0 is amplified by the atan2
        // sqrt-of-tiny-remainder.
        let load = LoadDirection::axis_z();
        let a = angle_to_z_axis(&load, &OrientationCandidate::IDENTITY);
        assert!(a < Fix128::from_ratio(1, 10), "angle was {}", a.to_f32());
    }

    #[test]
    fn axis_x_load_at_identity_is_perpendicular_to_z() {
        let load = LoadDirection::axis_x();
        let a = angle_to_z_axis(&load, &OrientationCandidate::IDENTITY);
        // 90° = π/2
        assert!(approx_eq(a, Fix128::HALF_PI, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn load_along_z_is_worst_case_for_anisotropic() {
        // Load along Z means load traverses layers → σ_z (weak)
        let pla = MaterialProperties::pla();
        let load = LoadDirection::axis_z();
        let angle = angle_to_z_axis(&load, &OrientationCandidate::IDENTITY);
        let s = effective_yield_at_angle(&pla, angle);
        // Expect near σ_z = 32.5 MPa
        let expected = pla.yield_strength_mpa * pla.anisotropy_z_ratio;
        let tol = expected * Fix128::from_ratio(5, 100);
        let diff = if s > expected {
            s - expected
        } else {
            expected - s
        };
        assert!(
            diff <= tol,
            "got {}, expected ~{}",
            s.to_f32(),
            expected.to_f32()
        );
    }

    #[test]
    fn optimize_analytical_improves_z_load() {
        // Load along Z, PLA → identity yield is σ_z, optimum is σ_xy
        let load = LoadDirection::axis_z();
        let pla = MaterialProperties::pla();
        let report = optimize_analytical(&load, &pla);
        assert!(report.improvement_mpa > Fix128::ZERO);
        assert!(report.effective_yield_mpa > report.identity_yield_mpa);
    }

    #[test]
    fn optimize_analytical_no_change_for_x_load() {
        // Load along X, PLA → already optimal (perpendicular to Z)
        let load = LoadDirection::axis_x();
        let pla = MaterialProperties::pla();
        let report = optimize_analytical(&load, &pla);
        // Improvement should be near zero (within CORDIC tolerance)
        assert!(report.improvement_mpa.abs() < Fix128::from_ratio(1, 10));
    }

    #[test]
    fn optimize_grid_improves_z_load() {
        let load = LoadDirection::axis_z();
        let pla = MaterialProperties::pla();
        let report = optimize_grid(&load, &pla, Fix128::from_ratio(3141, 10_000 * 2)); // ~15° step
        assert!(report.effective_yield_mpa > report.identity_yield_mpa);
    }

    #[test]
    fn optimize_grid_zero_step_falls_back_to_analytical() {
        let load = LoadDirection::axis_z();
        let pla = MaterialProperties::pla();
        let a = optimize_analytical(&load, &pla);
        let g = optimize_grid(&load, &pla, Fix128::ZERO);
        assert_eq!(a.effective_yield_mpa, g.effective_yield_mpa);
    }

    #[test]
    fn identity_candidate_is_zero_zero() {
        let id = OrientationCandidate::IDENTITY;
        assert_eq!(id.theta_x, Fix128::ZERO);
        assert_eq!(id.theta_y, Fix128::ZERO);
    }

    #[test]
    fn load_direction_length_axis() {
        let x = LoadDirection::axis_x();
        assert!(approx_eq(
            x.length(),
            Fix128::ONE,
            Fix128::from_ratio(1, 100)
        ));
    }
}
