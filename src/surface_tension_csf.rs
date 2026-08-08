//! Continuum Surface Force (CSF) for Surface Tension
//!
//! Phase G2 of the ALICE-Physics completeness project. Implements the
//! **Brackbill/Kothe/Zemach 1992** continuum surface force reformulation
//! of surface tension, suitable for level-set / VOF grid-based fluid
//! solvers.
//!
//! # Model
//!
//! Interfacial surface tension `σ` becomes a body force per unit volume
//! applied only near the interface:
//!
//! `f_st = σ · κ · n̂ · δ_smoothed(φ)`
//!
//! - `σ` (N/m): surface tension coefficient (water 0.072, PLA melt 0.030,
//!   mercury 0.485).
//! - `κ` (1/m): interface curvature `κ = ∇·(∇φ / |∇φ|)`.
//! - `n̂`: unit normal `∇φ / |∇φ|`.
//! - `δ_smoothed(φ)`: smeared Dirac delta centred on the interface
//!   (`φ = 0`). Common choice: `δ = 1 − |φ/ε|` for `|φ| < ε`, 0 otherwise.
//!
//! # References
//!
//! - Brackbill, Kothe & Zemach, "A continuum method for modeling surface
//!   tension", J. Comp. Phys. 100, 1992.
//! - Sethian, *Level Set Methods and Fast Marching Methods* 2nd ed. Ch. 6.
//! - de Gennes, Brochard-Wyart, Quéré, *Capillarity and Wetting Phenomena*
//!   (Springer 2004).

use crate::math::{Fix128, Vec3Fix};
use crate::multiphase::{curvature_at, Grid3d};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Surface tension coefficients (presets)
// ============================================================================

/// Water / air at 20 °C (N/m).
pub const SIGMA_WATER_AIR: Fix128 = Fix128 {
    hi: 0,
    lo: 0x126E_978D_4FDF_3B64, // ≈ 0.072
};
/// Molten PLA / air at 210 °C (N/m).
pub const SIGMA_PLA_AIR: Fix128 = Fix128 {
    hi: 0,
    lo: 0x07AE_147A_E147_AE14, // ≈ 0.030
};
/// Mercury / air at 20 °C (N/m).
pub const SIGMA_MERCURY_AIR: Fix128 = Fix128 {
    hi: 0,
    lo: 0x7C28_F5C2_8F5C_28F6, // ≈ 0.485
};
/// Molten steel / argon at 1600 °C (N/m).
pub const SIGMA_STEEL_ARGON: Fix128 = Fix128 { hi: 1, lo: 0 }; // ≈ 1.6 (dominant surface tension)

// ============================================================================
// Interface normal and delta
// ============================================================================

/// Unit normal at cell `(i, j, k)` computed from central differences on
/// the level set field. Returns `Vec3Fix::default()` if the gradient
/// magnitude is negligibly small.
#[must_use]
pub fn interface_normal(field: &Grid3d, i: usize, j: usize, k: usize) -> Vec3Fix {
    if i == 0 || j == 0 || k == 0 || i + 1 >= field.nx || j + 1 >= field.ny || k + 1 >= field.nz {
        return Vec3Fix::default();
    }
    let two_dx = field.dx + field.dx;
    if two_dx.is_zero() {
        return Vec3Fix::default();
    }
    let dpdx = (field.get(i + 1, j, k) - field.get(i - 1, j, k)) / two_dx;
    let dpdy = (field.get(i, j + 1, k) - field.get(i, j - 1, k)) / two_dx;
    let dpdz = (field.get(i, j, k + 1) - field.get(i, j, k - 1)) / two_dx;
    let mag = (dpdx * dpdx + dpdy * dpdy + dpdz * dpdz).sqrt();
    if mag.is_zero() {
        return Vec3Fix::default();
    }
    Vec3Fix::new(dpdx / mag, dpdy / mag, dpdz / mag)
}

/// Smeared Dirac delta centred at `φ = 0`:
///
/// `δ(φ) = (1/ε) · (1 − |φ|/ε)` for `|φ| < ε`, else 0.
///
/// `epsilon` is the smearing half-width, typically `1.5·dx` in cell units.
#[must_use]
pub fn smeared_delta(phi: Fix128, epsilon: Fix128) -> Fix128 {
    if epsilon.is_zero() {
        return Fix128::ZERO;
    }
    let abs_phi = phi.abs();
    if abs_phi >= epsilon {
        return Fix128::ZERO;
    }
    (Fix128::ONE - abs_phi / epsilon) / epsilon
}

// ============================================================================
// CSF body force
// ============================================================================

/// Surface-tension body force per unit volume at cell `(i, j, k)`:
///
/// `f = σ · κ · n̂ · δ`
///
/// Returns `Vec3Fix::default()` outside the smeared interface band.
#[must_use]
pub fn csf_body_force(
    field: &Grid3d,
    i: usize,
    j: usize,
    k: usize,
    sigma_n_per_m: Fix128,
    epsilon: Fix128,
) -> Vec3Fix {
    let phi = field.get(i, j, k);
    let delta = smeared_delta(phi, epsilon);
    if delta.is_zero() {
        return Vec3Fix::default();
    }
    let kappa = curvature_at(field, i, j, k);
    let normal = interface_normal(field, i, j, k);
    let scale = sigma_n_per_m * kappa * delta;
    Vec3Fix::new(normal.x * scale, normal.y * scale, normal.z * scale)
}

/// Convenience wrapper: compute the CSF force field on every cell of a
/// grid, returning three separate scalar grids `(fx, fy, fz)` sized
/// `nx·ny·nz`.
#[must_use]
pub fn compute_csf_field(
    field: &Grid3d,
    sigma_n_per_m: Fix128,
    epsilon: Fix128,
) -> (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>) {
    let n = field.total();
    let mut fx = vec![Fix128::ZERO; n];
    let mut fy = vec![Fix128::ZERO; n];
    let mut fz = vec![Fix128::ZERO; n];
    for k in 0..field.nz {
        for j in 0..field.ny {
            for i in 0..field.nx {
                let f = csf_body_force(field, i, j, k, sigma_n_per_m, epsilon);
                let idx = i + field.nx * (j + field.ny * k);
                fx[idx] = f.x;
                fy[idx] = f.y;
                fz[idx] = f.z;
            }
        }
    }
    (fx, fy, fz)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multiphase::initialize_level_set_sphere;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn sigma_water_correct() {
        assert!(approx_eq(
            SIGMA_WATER_AIR,
            Fix128::from_ratio(72, 1000),
            Fix128::from_ratio(1, 1000)
        ));
    }

    #[test]
    fn sigma_ordering_by_material() {
        // Water < mercury < steel; PLA melt < water
        assert!(SIGMA_PLA_AIR < SIGMA_WATER_AIR);
        assert!(SIGMA_WATER_AIR < SIGMA_MERCURY_AIR);
        assert!(SIGMA_MERCURY_AIR < SIGMA_STEEL_ARGON);
    }

    #[test]
    fn smeared_delta_zero_outside_band() {
        // φ = 2, ε = 1 → outside → δ = 0
        assert_eq!(
            smeared_delta(Fix128::from_int(2), Fix128::ONE),
            Fix128::ZERO
        );
    }

    #[test]
    fn smeared_delta_peak_at_zero() {
        // φ = 0, ε = 1 → δ = 1
        let d = smeared_delta(Fix128::ZERO, Fix128::ONE);
        assert_eq!(d, Fix128::ONE);
    }

    #[test]
    fn smeared_delta_linear_falloff() {
        // φ = 0.5, ε = 1 → δ = 0.5
        let d = smeared_delta(Fix128::from_ratio(5, 10), Fix128::ONE);
        assert_eq!(d, Fix128::from_ratio(5, 10));
    }

    #[test]
    fn smeared_delta_symmetric() {
        let d1 = smeared_delta(Fix128::from_ratio(3, 10), Fix128::ONE);
        let d2 = smeared_delta(Fix128::from_ratio(-3, 10), Fix128::ONE);
        assert_eq!(d1, d2);
    }

    #[test]
    fn smeared_delta_zero_epsilon_zero() {
        let d = smeared_delta(Fix128::ZERO, Fix128::ZERO);
        assert_eq!(d, Fix128::ZERO);
    }

    #[test]
    fn interface_normal_sphere_points_outward() {
        let mut g = Grid3d::new(11, 11, 11, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(3),
        );
        // Take a cell offset from centre — normal should point outward (+X here)
        let n = interface_normal(&g, 8, 5, 5);
        assert!(n.x > Fix128::ZERO);
        // Y, Z components near zero
        assert!(n.y.abs() < Fix128::from_ratio(2, 10));
        assert!(n.z.abs() < Fix128::from_ratio(2, 10));
    }

    #[test]
    fn interface_normal_out_of_range_default() {
        let g = Grid3d::new(3, 3, 3, Fix128::ONE, Fix128::ZERO);
        let n = interface_normal(&g, 0, 0, 0);
        assert_eq!(n, Vec3Fix::default());
    }

    #[test]
    fn csf_zero_outside_band() {
        let mut g = Grid3d::new(11, 11, 11, Fix128::ONE, Fix128::from_int(100));
        // φ = 100 everywhere → far outside interface → force zero
        let f = csf_body_force(&g, 5, 5, 5, SIGMA_WATER_AIR, Fix128::from_ratio(15, 10));
        let _ = &mut g;
        assert_eq!(f, Vec3Fix::default());
    }

    #[test]
    fn csf_nonzero_at_interface() {
        let mut g = Grid3d::new(11, 11, 11, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(5),
            Fix128::from_int(3),
        );
        // Cell (8,5,5) has φ ≈ 0 → within smeared band ε=1.5
        let f = csf_body_force(&g, 8, 5, 5, SIGMA_WATER_AIR, Fix128::from_ratio(15, 10));
        // Should have some non-zero X component (outward normal)
        assert!(f.x.abs() > Fix128::ZERO);
    }

    #[test]
    fn compute_csf_field_returns_three_grids() {
        let mut g = Grid3d::new(6, 6, 6, Fix128::ONE, Fix128::ZERO);
        initialize_level_set_sphere(
            &mut g,
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(2),
        );
        let (fx, fy, fz) = compute_csf_field(&g, SIGMA_WATER_AIR, Fix128::ONE);
        assert_eq!(fx.len(), 216);
        assert_eq!(fy.len(), 216);
        assert_eq!(fz.len(), 216);
        // At least one cell should have non-zero force
        assert!(fx.iter().any(|f| !f.is_zero()));
    }
}
