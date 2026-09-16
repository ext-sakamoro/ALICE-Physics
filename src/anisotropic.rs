//! Anisotropic Material Behaviour for FDM & Fiber-Reinforced Composites
//!
//! Phase B1 of the ALICE-Physics completeness project. Elevates the isotropic
//! `MaterialProperties` (see [`crate::filament_db`]) to a full **orthotropic**
//! description that captures the direction-dependence of 3D printed parts:
//! `E_L` (along-fiber, XY), `E_T` (cross-fiber in-plane), `E_Z` (through-layer).
//!
//! # When you need this
//!
//! - **FDM printed parts**: layer stacking creates strength reduction along Z.
//!   PLA / PETG / ABS lose 30–35 % strength across layers; CF-Nylon can lose
//!   50 % because fibres align in XY only.
//! - **Fibre-reinforced composites** (CFRP, GFRP): the engineering fibres
//!   dominate stiffness in one direction.
//! - **Sheet metals with rolling direction texture** (mild anisotropy).
//!
//! # What's provided
//!
//! - `OrthotropicElasticity` — 9-constant orthotropic stiffness (E_L, E_T,
//!   E_Z, ν_LT, ν_LZ, ν_TZ, G_LT, G_LZ, G_TZ).
//! - `AnisotropicStrength` — direction-dependent yield / tensile / compressive
//!   / shear strength.
//! - Effective Young's modulus at an arbitrary load angle using the classical
//!   transformation formula (Vinson & Sierakowski, *Behaviour of Structures*).
//! - Failure criteria: **Maximum Stress**, **Hill anisotropic yield**, and
//!   **Tsai-Wu quadratic interaction**.
//! - Convenience constructor `from_fdm_material()` that lifts an isotropic
//!   `MaterialProperties` to orthotropic by applying `anisotropy_z_ratio`.
//!
//! # Unit convention
//!
//! Same as `beam_stress`: mm / N / MPa. Poisson ratios are dimensionless
//! and typically 0.30–0.40.
//!
//! # References
//!
//! - Jones, "Mechanics of Composite Materials" 2nd ed. Chapters 2 & 4.
//! - Tsai & Wu, "A general theory of strength for anisotropic materials",
//!   J. Composite Materials 5(1), 1971.
//! - Hill, "A theory of the yielding and plastic flow of anisotropic
//!   metals", Proc. Roy. Soc. A 193, 1948.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// OrthotropicElasticity
// ============================================================================

/// Nine independent engineering constants defining an orthotropic linear
/// elastic material.
///
/// The three principal axes are:
/// - **L (longitudinal)**: in-plane fibre / print direction (XY parallel).
/// - **T (transverse)**: in-plane perpendicular to L (XY orthogonal).
/// - **Z (through-thickness)**: normal to the layer stack (build direction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrthotropicElasticity {
    /// Young's modulus along L (MPa).
    pub e_l_mpa: Fix128,
    /// Young's modulus along T (MPa).
    pub e_t_mpa: Fix128,
    /// Young's modulus along Z (MPa).
    pub e_z_mpa: Fix128,
    /// Major Poisson ratio (L → T contraction).
    pub nu_lt: Fix128,
    /// Poisson ratio (L → Z contraction).
    pub nu_lz: Fix128,
    /// Poisson ratio (T → Z contraction).
    pub nu_tz: Fix128,
    /// In-plane shear modulus L–T (MPa).
    pub g_lt_mpa: Fix128,
    /// Interlaminar shear modulus L–Z (MPa).
    pub g_lz_mpa: Fix128,
    /// Interlaminar shear modulus T–Z (MPa).
    pub g_tz_mpa: Fix128,
}

impl OrthotropicElasticity {
    /// Construct from an isotropic base material by applying the material's
    /// anisotropy ratio to the Z-direction quantities.
    ///
    /// Approximations used:
    /// - `E_L = E_T = E_iso` (in-plane isotropy: typical FDM cross-hatched raster)
    /// - `E_Z = E_iso · anisotropy_z_ratio`
    /// - `G_LT = E / (2·(1+ν))` (isotropic shear formula)
    /// - `G_LZ = G_TZ = G_LT · anisotropy_z_ratio` (shear across layers weakens)
    /// - `ν_LT = 0.35` (default polymer Poisson ratio; overrideable)
    /// - `ν_LZ = ν_TZ = 0.30`
    #[must_use]
    pub fn from_fdm_material(m: &MaterialProperties) -> Self {
        let e_iso = m.youngs_modulus_gpa * Fix128::from_int(1000); // GPa → MPa
        let e_z = e_iso * m.anisotropy_z_ratio;
        let nu_lt = Fix128::from_ratio(35, 100);
        let nu_lz = Fix128::from_ratio(3, 10);
        // G = E / (2·(1+ν))
        let two_plus_2nu = Fix128::from_int(2) + nu_lt.double();
        let g_lt = e_iso / two_plus_2nu;
        let g_lz = g_lt * m.anisotropy_z_ratio;
        Self {
            e_l_mpa: e_iso,
            e_t_mpa: e_iso,
            e_z_mpa: e_z,
            nu_lt,
            nu_lz,
            nu_tz: nu_lz,
            g_lt_mpa: g_lt,
            g_lz_mpa: g_lz,
            g_tz_mpa: g_lz,
        }
    }

    /// Effective Young's modulus (MPa) for a load applied at angle `theta`
    /// (radians) from the L axis, staying in the L-T plane.
    ///
    /// Classical composites formula (Jones eq. 2.85):
    /// `1 / E(θ) = c⁴ / E_L + s⁴ / E_T + c²s² · (1/G_LT − 2ν_LT/E_L)`
    /// where `c = cos(θ)`, `s = sin(θ)`.
    ///
    /// At `θ = 0` returns `e_l_mpa`; at `π/2` returns `e_t_mpa`.
    #[must_use]
    pub fn e_at_angle_lt(&self, theta: Fix128) -> Fix128 {
        let (s, c) = theta.sin_cos();
        let c2 = c * c;
        let s2 = s * s;
        let c4 = c2 * c2;
        let s4 = s2 * s2;

        // Guard against zero moduli
        if self.e_l_mpa.is_zero() || self.e_t_mpa.is_zero() || self.g_lt_mpa.is_zero() {
            return Fix128::ZERO;
        }

        // Compute 1/E as sum of terms
        let inv_e_l = Fix128::ONE / self.e_l_mpa;
        let inv_e_t = Fix128::ONE / self.e_t_mpa;
        let inv_g_lt = Fix128::ONE / self.g_lt_mpa;
        // 2·ν_LT / E_L
        let term_nu = self.nu_lt.double() * inv_e_l;
        // Combined: c²s² · (1/G − 2ν/E)
        let mixed = c2 * s2 * (inv_g_lt - term_nu);

        let inv_e = c4 * inv_e_l + s4 * inv_e_t + mixed;
        if inv_e.is_zero() {
            return Fix128::ZERO;
        }
        Fix128::ONE / inv_e
    }

    /// Effective Young's modulus in the L-Z plane for load at angle `theta`
    /// from the L axis. Uses the same transformation with (E_L, E_Z, G_LZ, ν_LZ).
    #[must_use]
    pub fn e_at_angle_lz(&self, theta: Fix128) -> Fix128 {
        let (s, c) = theta.sin_cos();
        let c2 = c * c;
        let s2 = s * s;
        let c4 = c2 * c2;
        let s4 = s2 * s2;
        if self.e_l_mpa.is_zero() || self.e_z_mpa.is_zero() || self.g_lz_mpa.is_zero() {
            return Fix128::ZERO;
        }
        let inv_e_l = Fix128::ONE / self.e_l_mpa;
        let inv_e_z = Fix128::ONE / self.e_z_mpa;
        let inv_g_lz = Fix128::ONE / self.g_lz_mpa;
        let term_nu = self.nu_lz.double() * inv_e_l;
        let mixed = c2 * s2 * (inv_g_lz - term_nu);
        let inv_e = c4 * inv_e_l + s4 * inv_e_z + mixed;
        if inv_e.is_zero() {
            return Fix128::ZERO;
        }
        Fix128::ONE / inv_e
    }
}

// ============================================================================
// AnisotropicStrength
// ============================================================================

/// Direction-dependent strength envelope for an orthotropic material.
///
/// Stores tensile (X_t), compressive (X_c) and shear (S) strengths for each
/// principal axis. For symmetric materials (metals) `X_t = X_c`; polymers and
/// composites typically show `X_c > X_t` (better in compression than tension).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnisotropicStrength {
    /// Tensile strength along L (MPa).
    pub x_l_tension_mpa: Fix128,
    /// Compressive strength along L (MPa, positive value).
    pub x_l_compression_mpa: Fix128,
    /// Tensile strength along T (MPa).
    pub x_t_tension_mpa: Fix128,
    /// Compressive strength along T (MPa).
    pub x_t_compression_mpa: Fix128,
    /// Tensile strength along Z (MPa).
    pub x_z_tension_mpa: Fix128,
    /// Compressive strength along Z (MPa).
    pub x_z_compression_mpa: Fix128,
    /// In-plane shear strength L-T (MPa).
    pub s_lt_mpa: Fix128,
    /// Interlaminar shear strength L-Z (MPa).
    pub s_lz_mpa: Fix128,
    /// Interlaminar shear strength T-Z (MPa).
    pub s_tz_mpa: Fix128,
}

impl AnisotropicStrength {
    /// Construct from an isotropic FDM material.
    ///
    /// Applies:
    /// - Tension = compression = `yield_strength_mpa` (symmetric assumption)
    /// - Z-direction tension = tension × anisotropy_z_ratio
    /// - Shear = tension × 0.6 (typical polymer ratio, `S ≈ 0.6·σ_y`)
    /// - Interlaminar shear = in-plane shear × anisotropy_z_ratio
    #[must_use]
    pub fn from_fdm_material(m: &MaterialProperties) -> Self {
        let x_iso = m.yield_strength_mpa;
        let x_z = x_iso * m.anisotropy_z_ratio;
        let s_lt = x_iso * Fix128::from_ratio(6, 10);
        let s_lz = s_lt * m.anisotropy_z_ratio;
        Self {
            x_l_tension_mpa: x_iso,
            x_l_compression_mpa: x_iso,
            x_t_tension_mpa: x_iso,
            x_t_compression_mpa: x_iso,
            x_z_tension_mpa: x_z,
            x_z_compression_mpa: x_z,
            s_lt_mpa: s_lt,
            s_lz_mpa: s_lz,
            s_tz_mpa: s_lz,
        }
    }
}

// ============================================================================
// Stress state & failure criteria
// ============================================================================

/// Full 3D stress state expressed in the material's principal axes.
///
/// Sign convention: positive = tension, negative = compression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrthotropicStress {
    /// Normal stress along L (MPa).
    pub sigma_l: Fix128,
    /// Normal stress along T (MPa).
    pub sigma_t: Fix128,
    /// Normal stress along Z (MPa).
    pub sigma_z: Fix128,
    /// Shear stress L-T (MPa).
    pub tau_lt: Fix128,
    /// Shear stress L-Z (MPa).
    pub tau_lz: Fix128,
    /// Shear stress T-Z (MPa).
    pub tau_tz: Fix128,
}

impl OrthotropicStress {
    /// Construct with only in-plane axial stresses (useful for beam bending).
    #[must_use]
    pub const fn axial(sigma_l: Fix128, sigma_t: Fix128, sigma_z: Fix128) -> Self {
        Self {
            sigma_l,
            sigma_t,
            sigma_z,
            tau_lt: Fix128::ZERO,
            tau_lz: Fix128::ZERO,
            tau_tz: Fix128::ZERO,
        }
    }
}

/// Failure criterion selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureCriterion {
    /// Maximum-stress: each stress component must stay under its allowable.
    /// Simple to interpret but ignores component interactions.
    MaximumStress,
    /// Hill's anisotropic yield criterion (extension of von Mises).
    /// Isotropic strength (tension = compression); good for metals.
    Hill,
    /// Tsai-Wu quadratic interaction — most general, allows tension ≠ compression.
    /// Recommended for polymer composites and FDM parts.
    TsaiWu,
}

/// Failure analysis output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailureReport {
    /// Failure index: `< 1` safe, `= 1` incipient failure, `> 1` failed.
    ///
    /// Meaning depends on criterion; for Tsai-Wu this is the LHS of the
    /// quadratic inequality.
    pub failure_index: Fix128,
    /// Reserve factor = `1 / max(failure_index, ε)` — how much the loading
    /// could scale before failure.
    pub reserve_factor: Fix128,
    /// True iff `failure_index < 1` (no failure predicted).
    pub is_safe: bool,
}

/// Evaluate a failure criterion for a given stress state.
#[must_use]
pub fn evaluate_failure(
    stress: &OrthotropicStress,
    strength: &AnisotropicStrength,
    criterion: FailureCriterion,
) -> FailureReport {
    let idx = match criterion {
        FailureCriterion::MaximumStress => max_stress_index(stress, strength),
        FailureCriterion::Hill => hill_index(stress, strength),
        FailureCriterion::TsaiWu => tsai_wu_index(stress, strength),
    };
    let eps = Fix128::from_ratio(1, 1_000_000);
    let denom = if idx > eps { idx } else { eps };
    FailureReport {
        failure_index: idx,
        reserve_factor: Fix128::ONE / denom,
        is_safe: idx < Fix128::ONE,
    }
}

/// Maximum-stress failure index (largest ratio across all components).
#[must_use]
fn max_stress_index(stress: &OrthotropicStress, s: &AnisotropicStrength) -> Fix128 {
    let mut worst = Fix128::ZERO;

    let check = |worst: &mut Fix128, val: Fix128, allow: Fix128| {
        if allow.is_zero() {
            return;
        }
        let r = val.abs() / allow;
        if r > *worst {
            *worst = r;
        }
    };

    // Normal — pick tension or compression allowable based on sign
    let al = if stress.sigma_l >= Fix128::ZERO {
        s.x_l_tension_mpa
    } else {
        s.x_l_compression_mpa
    };
    let at = if stress.sigma_t >= Fix128::ZERO {
        s.x_t_tension_mpa
    } else {
        s.x_t_compression_mpa
    };
    let az = if stress.sigma_z >= Fix128::ZERO {
        s.x_z_tension_mpa
    } else {
        s.x_z_compression_mpa
    };
    check(&mut worst, stress.sigma_l, al);
    check(&mut worst, stress.sigma_t, at);
    check(&mut worst, stress.sigma_z, az);
    check(&mut worst, stress.tau_lt, s.s_lt_mpa);
    check(&mut worst, stress.tau_lz, s.s_lz_mpa);
    check(&mut worst, stress.tau_tz, s.s_tz_mpa);
    worst
}

/// Hill's anisotropic yield criterion (1948). Uses only tensile strength
/// values (isotropic tension/compression).
///
/// f = F·(σ_T − σ_Z)² + G·(σ_Z − σ_L)² + H·(σ_L − σ_T)²
///     + 2L·τ_TZ² + 2M·τ_LZ² + 2N·τ_LT²
///
/// where F, G, H, L, M, N derive from the tensile strengths:
///   F = ½·(1/X_T² + 1/X_Z² − 1/X_L²)
///   G = ½·(1/X_Z² + 1/X_L² − 1/X_T²)
///   H = ½·(1/X_L² + 1/X_T² − 1/X_Z²)
///   L = 1/(2·S_TZ²), M = 1/(2·S_LZ²), N = 1/(2·S_LT²)
#[must_use]
fn hill_index(stress: &OrthotropicStress, s: &AnisotropicStrength) -> Fix128 {
    // Precompute 1 / X² for each strength
    let inv_sq = |x: Fix128| {
        if x.is_zero() {
            Fix128::ZERO
        } else {
            let inv = Fix128::ONE / x;
            inv * inv
        }
    };
    let x_l2 = inv_sq(s.x_l_tension_mpa);
    let x_t2 = inv_sq(s.x_t_tension_mpa);
    let x_z2 = inv_sq(s.x_z_tension_mpa);
    let s_lt2 = inv_sq(s.s_lt_mpa);
    let s_lz2 = inv_sq(s.s_lz_mpa);
    let s_tz2 = inv_sq(s.s_tz_mpa);

    // F, G, H can be negative for very anisotropic materials but Hill still
    // works algebraically. Just compute directly.
    let half = Fix128::from_ratio(1, 2);
    let f_c = half * (x_t2 + x_z2 - x_l2);
    let g_c = half * (x_z2 + x_l2 - x_t2);
    let h_c = half * (x_l2 + x_t2 - x_z2);

    let d_tz = stress.sigma_t - stress.sigma_z;
    let d_zl = stress.sigma_z - stress.sigma_l;
    let d_lt = stress.sigma_l - stress.sigma_t;

    f_c * d_tz * d_tz
        + g_c * d_zl * d_zl
        + h_c * d_lt * d_lt
        + s_tz2 * stress.tau_tz * stress.tau_tz
        + s_lz2 * stress.tau_lz * stress.tau_lz
        + s_lt2 * stress.tau_lt * stress.tau_lt
}

/// Tsai-Wu quadratic interaction (1971). Handles tension ≠ compression
/// asymmetry and the six independent shear allowables.
///
/// F_i · σ_i + F_ij · σ_i · σ_j ≤ 1
///
/// where F_i (linear) captures tension/compression asymmetry:
///   F_L = 1/X_Lt − 1/X_Lc
///   F_T = 1/X_Tt − 1/X_Tc
///   F_Z = 1/X_Zt − 1/X_Zc
/// and F_ii (quadratic) uses geometric mean:
///   F_LL = 1/(X_Lt · X_Lc), F_TT, F_ZZ, F_SS = 1/S²
///
/// Interaction terms F_LT, F_LZ, F_TZ are commonly set to
/// `-½ · √(F_ii · F_jj)` per Hoffman (Tsai-Hoffman variant); we use that.
#[must_use]
fn tsai_wu_index(stress: &OrthotropicStress, s: &AnisotropicStrength) -> Fix128 {
    let inv = |x: Fix128| {
        if x.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / x
        }
    };

    let f_l = inv(s.x_l_tension_mpa) - inv(s.x_l_compression_mpa);
    let f_t = inv(s.x_t_tension_mpa) - inv(s.x_t_compression_mpa);
    let f_z = inv(s.x_z_tension_mpa) - inv(s.x_z_compression_mpa);

    let f_ll = inv(s.x_l_tension_mpa * s.x_l_compression_mpa);
    let f_tt = inv(s.x_t_tension_mpa * s.x_t_compression_mpa);
    let f_zz = inv(s.x_z_tension_mpa * s.x_z_compression_mpa);
    let f_ss_lt = inv(s.s_lt_mpa * s.s_lt_mpa);
    let f_ss_lz = inv(s.s_lz_mpa * s.s_lz_mpa);
    let f_ss_tz = inv(s.s_tz_mpa * s.s_tz_mpa);

    // Hoffman interaction terms: F_ij = -0.5 · sqrt(F_ii · F_jj)
    let neg_half = Fix128::from_ratio(-1, 2);
    let f_lt = neg_half * (f_ll * f_tt).sqrt();
    let f_lz = neg_half * (f_ll * f_zz).sqrt();
    let f_tz = neg_half * (f_tt * f_zz).sqrt();

    let sl = stress.sigma_l;
    let st = stress.sigma_t;
    let sz = stress.sigma_z;

    f_l * sl
        + f_t * st
        + f_z * sz
        + f_ll * sl * sl
        + f_tt * st * st
        + f_zz * sz * sz
        + f_ss_lt * stress.tau_lt * stress.tau_lt
        + f_ss_lz * stress.tau_lz * stress.tau_lz
        + f_ss_tz * stress.tau_tz * stress.tau_tz
        + f_lt.double() * sl * st
        + f_lz.double() * sl * sz
        + f_tz.double() * st * sz
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
    fn ortho_from_pla_has_reduced_z_stiffness() {
        let pla = MaterialProperties::pla();
        let ortho = OrthotropicElasticity::from_fdm_material(&pla);
        assert!(ortho.e_z_mpa < ortho.e_l_mpa);
        assert_eq!(ortho.e_l_mpa, ortho.e_t_mpa); // in-plane isotropic
                                                  // E_Z = 3500 MPa · 0.65 = 2275 MPa
        let expected_ez = Fix128::from_int(3500) * Fix128::from_ratio(65, 100);
        assert!(approx_eq(ortho.e_z_mpa, expected_ez, Fix128::ONE));
    }

    #[test]
    fn ortho_cf_nylon_more_z_anisotropic_than_pla() {
        let cfn = OrthotropicElasticity::from_fdm_material(&MaterialProperties::cf_nylon());
        let pla = OrthotropicElasticity::from_fdm_material(&MaterialProperties::pla());
        // CF-Nylon anisotropy 0.50 vs PLA 0.65 → CF-Nylon E_Z / E_L smaller
        let ratio_cfn = cfn.e_z_mpa / cfn.e_l_mpa;
        let ratio_pla = pla.e_z_mpa / pla.e_l_mpa;
        assert!(ratio_cfn < ratio_pla);
    }

    #[test]
    fn e_at_angle_zero_equals_e_l() {
        let ortho = OrthotropicElasticity::from_fdm_material(&MaterialProperties::pla());
        let e_at_0 = ortho.e_at_angle_lt(Fix128::ZERO);
        let diff = if e_at_0 > ortho.e_l_mpa {
            e_at_0 - ortho.e_l_mpa
        } else {
            ortho.e_l_mpa - e_at_0
        };
        // CORDIC + division precision — allow 0.01% tolerance
        let tol = ortho.e_l_mpa * Fix128::from_ratio(1, 10_000);
        assert!(diff <= tol);
    }

    #[test]
    fn e_at_angle_lz_pla_z_below_l() {
        let ortho = OrthotropicElasticity::from_fdm_material(&MaterialProperties::pla());
        let e_at_90 = ortho.e_at_angle_lz(Fix128::HALF_PI);
        // Should be near E_Z (2275 MPa)
        let diff = if e_at_90 > ortho.e_z_mpa {
            e_at_90 - ortho.e_z_mpa
        } else {
            ortho.e_z_mpa - e_at_90
        };
        let tol = ortho.e_z_mpa * Fix128::from_ratio(1, 100);
        assert!(diff <= tol);
    }

    #[test]
    fn strength_from_pla_reduces_z() {
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        assert!(s.x_z_tension_mpa < s.x_l_tension_mpa);
        // X_Z_t = 50 · 0.65 = 32.5
        assert!(approx_eq(
            s.x_z_tension_mpa,
            Fix128::from_ratio(325, 10),
            Fix128::from_ratio(1, 10)
        ));
    }

    #[test]
    fn max_stress_safe_below_allowable() {
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        // sigma_L = 10 MPa, allowable = 50 MPa → ratio 0.2
        let stress = OrthotropicStress::axial(
            Fix128::from_int(10),
            Fix128::from_int(5),
            Fix128::from_int(3),
        );
        let r = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
        assert!(r.is_safe);
        assert!(r.failure_index < Fix128::ONE);
        // reserve factor = 1 / 0.2 = 5
        assert!(approx_eq(
            r.reserve_factor,
            Fix128::from_int(5),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn max_stress_fails_above_allowable() {
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        // sigma_L = 80 MPa > 50 MPa yield
        let stress = OrthotropicStress::axial(Fix128::from_int(80), Fix128::ZERO, Fix128::ZERO);
        let r = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
        assert!(!r.is_safe);
        assert!(r.failure_index > Fix128::ONE);
    }

    #[test]
    fn max_stress_uses_z_allowable_for_z_stress() {
        // PLA X_Z = 32.5 MPa. sigma_Z = 40 MPa should fail.
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        let stress = OrthotropicStress::axial(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(40));
        let r = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
        assert!(!r.is_safe);
    }

    #[test]
    fn hill_reduces_to_von_mises_for_isotropic() {
        // Isotropic: X_L = X_T = X_Z = 100 MPa. Uniaxial σ_L = 100 → failure.
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(100),
            x_l_compression_mpa: Fix128::from_int(100),
            x_t_tension_mpa: Fix128::from_int(100),
            x_t_compression_mpa: Fix128::from_int(100),
            x_z_tension_mpa: Fix128::from_int(100),
            x_z_compression_mpa: Fix128::from_int(100),
            s_lt_mpa: Fix128::from_int(60),
            s_lz_mpa: Fix128::from_int(60),
            s_tz_mpa: Fix128::from_int(60),
        };
        let stress = OrthotropicStress::axial(Fix128::from_int(100), Fix128::ZERO, Fix128::ZERO);
        let r = evaluate_failure(&stress, &s, FailureCriterion::Hill);
        // For isotropic with pure uniaxial at X, index should be ~1.0
        assert!(approx_eq(
            r.failure_index,
            Fix128::ONE,
            Fix128::from_ratio(5, 100)
        ));
    }

    #[test]
    fn tsai_wu_captures_tension_compression_asymmetry() {
        // Suppose PLA has X_L_t = 50, X_L_c = 80 (higher in compression)
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(50),
            x_l_compression_mpa: Fix128::from_int(80),
            x_t_tension_mpa: Fix128::from_int(50),
            x_t_compression_mpa: Fix128::from_int(80),
            x_z_tension_mpa: Fix128::from_int(30),
            x_z_compression_mpa: Fix128::from_int(50),
            s_lt_mpa: Fix128::from_int(30),
            s_lz_mpa: Fix128::from_int(20),
            s_tz_mpa: Fix128::from_int(20),
        };
        // 40 MPa tensile → LHS < 1 (safe)
        let stress_t = OrthotropicStress::axial(Fix128::from_int(40), Fix128::ZERO, Fix128::ZERO);
        // 40 MPa compressive → also safe (higher allowable) → smaller LHS
        let stress_c = OrthotropicStress::axial(Fix128::from_int(-40), Fix128::ZERO, Fix128::ZERO);
        let r_t = evaluate_failure(&stress_t, &s, FailureCriterion::TsaiWu);
        let r_c = evaluate_failure(&stress_c, &s, FailureCriterion::TsaiWu);
        // Tensile should have larger failure index than compressive
        assert!(r_t.failure_index > r_c.failure_index);
        assert!(r_t.is_safe);
        assert!(r_c.is_safe);
    }

    #[test]
    fn axial_constructor_zeros_shear() {
        let s = OrthotropicStress::axial(
            Fix128::from_int(10),
            Fix128::from_int(20),
            Fix128::from_int(30),
        );
        assert_eq!(s.tau_lt, Fix128::ZERO);
        assert_eq!(s.tau_lz, Fix128::ZERO);
        assert_eq!(s.tau_tz, Fix128::ZERO);
    }

    #[test]
    fn shear_stress_flags_failure() {
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        // In-plane shear allowable = 30 MPa; apply 50 MPa
        let stress = OrthotropicStress {
            sigma_l: Fix128::ZERO,
            sigma_t: Fix128::ZERO,
            sigma_z: Fix128::ZERO,
            tau_lt: Fix128::from_int(50),
            tau_lz: Fix128::ZERO,
            tau_tz: Fix128::ZERO,
        };
        let r = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
        assert!(!r.is_safe);
    }

    #[test]
    fn zero_stress_safe_all_criteria() {
        let s = AnisotropicStrength::from_fdm_material(&MaterialProperties::pla());
        let zero = OrthotropicStress::axial(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        for c in [
            FailureCriterion::MaximumStress,
            FailureCriterion::Hill,
            FailureCriterion::TsaiWu,
        ] {
            let r = evaluate_failure(&zero, &s, c);
            assert!(r.is_safe, "criterion {:?} should pass at zero stress", c);
            assert_eq!(r.failure_index, Fix128::ZERO);
        }
    }

    // ------------------------------------------------------------------
    // Mutation-killing closed-form tests (cargo-mutants 2026-09-16)
    // ------------------------------------------------------------------

    /// 1e-9 absolute tolerance: CORDIC sin/cos and Fix128 division are exact
    /// to ~1e-19, so any operator mutant moves the result by orders more.
    fn tight() -> Fix128 {
        Fix128::from_ratio(1, 1_000_000_000)
    }

    /// Distinct-modulus test material so that every term of the Jones
    /// transformation formula has a unique magnitude:
    /// E_L = 200, E_T = 100, E_Z = 400, ν_LT = 1/4, ν_LZ = 1/8, G_LT = 50,
    /// G_LZ = 25 (MPa).
    fn distinct_ortho() -> OrthotropicElasticity {
        OrthotropicElasticity {
            e_l_mpa: Fix128::from_int(200),
            e_t_mpa: Fix128::from_int(100),
            e_z_mpa: Fix128::from_int(400),
            nu_lt: Fix128::from_ratio(1, 4),
            nu_lz: Fix128::from_ratio(1, 8),
            nu_tz: Fix128::from_ratio(3, 10),
            g_lt_mpa: Fix128::from_int(50),
            g_lz_mpa: Fix128::from_int(25),
            g_tz_mpa: Fix128::from_int(20),
        }
    }

    /// Kills `from_fdm_material` (elasticity) lines 97–99: G_LT = E/(2(1+ν))
    /// with E = 2 GPa = 2000 MPa, ν = 0.35 → G_LT = 2000/2.7 = 20000/27,
    /// G_LZ = G_TZ = G_LT·0.5 = 10000/27.
    #[test]
    fn ortho_from_fdm_shear_moduli_closed_form() {
        let m = MaterialProperties {
            youngs_modulus_gpa: Fix128::from_int(2),
            anisotropy_z_ratio: Fix128::from_ratio(1, 2),
            ..MaterialProperties::pla()
        };
        let o = OrthotropicElasticity::from_fdm_material(&m);
        assert_eq!(o.e_l_mpa, Fix128::from_int(2000));
        assert_eq!(o.e_z_mpa, Fix128::from_int(1000));
        assert!(
            approx_eq(o.g_lt_mpa, Fix128::from_ratio(20000, 27), tight()),
            "G_LT = {} expected 740.740…",
            o.g_lt_mpa.to_f64()
        );
        assert!(
            approx_eq(o.g_lz_mpa, Fix128::from_ratio(10000, 27), tight()),
            "G_LZ = {} expected 370.370…",
            o.g_lz_mpa.to_f64()
        );
        assert_eq!(o.g_tz_mpa, o.g_lz_mpa);
        assert_eq!(o.nu_lt, Fix128::from_ratio(35, 100));
        assert_eq!(o.nu_lz, Fix128::from_ratio(3, 10));
        assert_eq!(o.nu_tz, o.nu_lz);
    }

    /// Kills `e_at_angle_lt` lines 124–127, 136–143 (all `*`/`/`/`+`/`-`
    /// operator mutants). Jones eq. 2.85 at θ = 45° (c² = s² = ½, c⁴ = s⁴ = ¼):
    /// 1/E = ¼·(1/200) + ¼·(1/100) + ¼·(1/50 − 2·¼/200)
    ///     = 0.00125 + 0.0025 + 0.004375 = 0.008125 → E = 1600/13 ≈ 123.077.
    #[test]
    fn e_at_angle_lt_45deg_jones_closed_form() {
        let o = distinct_ortho();
        let e45 = o.e_at_angle_lt(Fix128::HALF_PI.half());
        assert!(
            approx_eq(e45, Fix128::from_ratio(1600, 13), tight()),
            "E(45°) = {} expected 123.0769…",
            e45.to_f64()
        );
        // θ = 90° → E_T exactly (E_L ≠ E_T here, unlike the PLA fixture)
        let e90 = o.e_at_angle_lt(Fix128::HALF_PI);
        assert!(
            approx_eq(e90, Fix128::from_int(100), tight()),
            "E(90°) = {} expected 100",
            e90.to_f64()
        );
        // θ = 0 → E_L exactly
        let e0 = o.e_at_angle_lt(Fix128::ZERO);
        assert!(approx_eq(e0, Fix128::from_int(200), tight()));
    }

    /// Kills `e_at_angle_lz` lines 155–158, 162–167. Same formula with
    /// (E_L, E_Z, G_LZ, ν_LZ) = (200, 400, 25, 1/8) at 45°:
    /// 1/E = ¼·(1/200) + ¼·(1/400) + ¼·(1/25 − 2·⅛/200)
    ///     = 0.00125 + 0.000625 + ¼·(0.04 − 0.00125) = 0.01156250 → E = 3200/37.
    #[test]
    fn e_at_angle_lz_45deg_jones_closed_form() {
        let o = distinct_ortho();
        let e45 = o.e_at_angle_lz(Fix128::HALF_PI.half());
        assert!(
            approx_eq(e45, Fix128::from_ratio(3200, 37), tight()),
            "E_LZ(45°) = {} expected 86.486…",
            e45.to_f64()
        );
        let e90 = o.e_at_angle_lz(Fix128::HALF_PI);
        assert!(
            approx_eq(e90, Fix128::from_int(400), tight()),
            "E_LZ(90°) = {} expected 400",
            e90.to_f64()
        );
        let e0 = o.e_at_angle_lz(Fix128::ZERO);
        assert!(approx_eq(e0, Fix128::from_int(200), tight()));
    }

    /// Kills the `||` → `&&` guard mutants at lines 130 and 159: with exactly
    /// one modulus zero the function must still return 0 (Fix128 `x / 0 = 0`,
    /// so the mutated guard would fall through and produce a finite modulus).
    #[test]
    fn e_at_angle_zero_modulus_guard_each_operand() {
        let base = distinct_ortho();
        let theta = Fix128::HALF_PI.half();
        let cases_lt = [
            OrthotropicElasticity {
                e_l_mpa: Fix128::ZERO,
                ..base
            },
            OrthotropicElasticity {
                e_t_mpa: Fix128::ZERO,
                ..base
            },
            OrthotropicElasticity {
                g_lt_mpa: Fix128::ZERO,
                ..base
            },
        ];
        for (i, o) in cases_lt.iter().enumerate() {
            assert_eq!(o.e_at_angle_lt(theta), Fix128::ZERO, "LT case {i}");
        }
        let cases_lz = [
            OrthotropicElasticity {
                e_l_mpa: Fix128::ZERO,
                ..base
            },
            OrthotropicElasticity {
                e_z_mpa: Fix128::ZERO,
                ..base
            },
            OrthotropicElasticity {
                g_lz_mpa: Fix128::ZERO,
                ..base
            },
        ];
        for (i, o) in cases_lz.iter().enumerate() {
            assert_eq!(o.e_at_angle_lz(theta), Fix128::ZERO, "LZ case {i}");
        }
        // Sanity: the unmodified material is non-zero at the same angle.
        assert!(base.e_at_angle_lt(theta) > Fix128::ZERO);
        assert!(base.e_at_angle_lz(theta) > Fix128::ZERO);
    }

    /// Kills `AnisotropicStrength::from_fdm_material` line 219:
    /// σ_y = 40, ratio = ½ → S_LT = 24, S_LZ = S_TZ = 12, X_Z = 20 (all dyadic).
    #[test]
    fn strength_from_fdm_closed_form() {
        let m = MaterialProperties {
            yield_strength_mpa: Fix128::from_int(40),
            anisotropy_z_ratio: Fix128::from_ratio(1, 2),
            ..MaterialProperties::pla()
        };
        let s = AnisotropicStrength::from_fdm_material(&m);
        assert_eq!(s.x_l_tension_mpa, Fix128::from_int(40));
        assert_eq!(s.x_l_compression_mpa, Fix128::from_int(40));
        assert_eq!(s.x_t_tension_mpa, Fix128::from_int(40));
        assert_eq!(s.x_t_compression_mpa, Fix128::from_int(40));
        assert_eq!(s.x_z_tension_mpa, Fix128::from_int(20));
        assert_eq!(s.x_z_compression_mpa, Fix128::from_int(20));
        assert!(approx_eq(s.s_lt_mpa, Fix128::from_int(24), tight()));
        assert!(
            approx_eq(s.s_lz_mpa, Fix128::from_int(12), tight()),
            "S_LZ = {} expected 12",
            s.s_lz_mpa.to_f64()
        );
        assert_eq!(s.s_tz_mpa, s.s_lz_mpa);
    }

    /// Asymmetric strength envelope: tension ≠ compression on every axis so
    /// the sign-dependent allowable selection is observable.
    fn asym_strength() -> AnisotropicStrength {
        AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(40),
            x_l_compression_mpa: Fix128::from_int(80),
            x_t_tension_mpa: Fix128::from_int(20),
            x_t_compression_mpa: Fix128::from_int(60),
            x_z_tension_mpa: Fix128::from_int(10),
            x_z_compression_mpa: Fix128::from_int(50),
            s_lt_mpa: Fix128::from_int(16),
            s_lz_mpa: Fix128::from_int(8),
            s_tz_mpa: Fix128::from_int(4),
        }
    }

    /// Kills `evaluate_failure` line 318 (`<` → `<=`): at failure_index
    /// exactly 1 the report is *not* safe (doc: "True iff failure_index < 1").
    /// One ulp below 1 is safe.
    #[test]
    fn evaluate_failure_index_exactly_one_is_not_safe() {
        let s = asym_strength();
        let at_limit = OrthotropicStress::axial(Fix128::from_int(40), Fix128::ZERO, Fix128::ZERO);
        let r = evaluate_failure(&at_limit, &s, FailureCriterion::MaximumStress);
        assert_eq!(r.failure_index, Fix128::ONE);
        assert!(!r.is_safe, "index == 1 is incipient failure, not safe");
        assert_eq!(r.reserve_factor, Fix128::ONE);

        let one_ulp = Fix128::from_raw(0, 1);
        let just_below =
            OrthotropicStress::axial(Fix128::from_int(40) - one_ulp, Fix128::ZERO, Fix128::ZERO);
        let r2 = evaluate_failure(&just_below, &s, FailureCriterion::MaximumStress);
        assert!(r2.failure_index < Fix128::ONE);
        assert!(r2.is_safe);
    }

    /// Reserve factor = 1/index for index > ε; with index = ¼ → 4 exactly.
    #[test]
    fn evaluate_failure_reserve_factor_exact() {
        let s = asym_strength();
        let stress = OrthotropicStress::axial(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO);
        let r = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
        assert_eq!(r.failure_index, Fix128::from_ratio(1, 4));
        assert_eq!(r.reserve_factor, Fix128::from_int(4));
        assert!(r.is_safe);
    }

    /// Kills `max_stress_index` lines 338, 343, 348 (`>=` → `<`): a positive
    /// normal stress must be divided by the *tensile* allowable and a negative
    /// one by the *compressive* allowable, on each axis independently.
    #[test]
    fn max_stress_selects_tension_vs_compression_allowable_per_axis() {
        let s = asym_strength();
        // L: +20 / 40 = ½ ; −20 / 80 = ¼
        let l_pos = OrthotropicStress::axial(Fix128::from_int(20), Fix128::ZERO, Fix128::ZERO);
        let l_neg = OrthotropicStress::axial(Fix128::from_int(-20), Fix128::ZERO, Fix128::ZERO);
        assert_eq!(max_stress_index(&l_pos, &s), Fix128::from_ratio(1, 2));
        assert_eq!(max_stress_index(&l_neg, &s), Fix128::from_ratio(1, 4));
        // T: +10 / 20 = ½ ; −15 / 60 = ¼
        let t_pos = OrthotropicStress::axial(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO);
        let t_neg = OrthotropicStress::axial(Fix128::ZERO, Fix128::from_int(-15), Fix128::ZERO);
        assert_eq!(max_stress_index(&t_pos, &s), Fix128::from_ratio(1, 2));
        assert_eq!(max_stress_index(&t_neg, &s), Fix128::from_ratio(1, 4));
        // Z: +5 / 10 = ½ ; −25 / 50 = ½ (compressive allowable 5× larger)
        let z_pos = OrthotropicStress::axial(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(5));
        let z_neg = OrthotropicStress::axial(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(-25));
        assert_eq!(max_stress_index(&z_pos, &s), Fix128::from_ratio(1, 2));
        assert_eq!(max_stress_index(&z_neg, &s), Fix128::from_ratio(1, 2));
        // Shear components use their own allowables: 4/16, 2/8, 3/4 → worst ¾
        let shear = OrthotropicStress {
            sigma_l: Fix128::ZERO,
            sigma_t: Fix128::ZERO,
            sigma_z: Fix128::ZERO,
            tau_lt: Fix128::from_int(4),
            tau_lz: Fix128::from_int(-2),
            tau_tz: Fix128::from_int(3),
        };
        assert_eq!(max_stress_index(&shear, &s), Fix128::from_ratio(3, 4));
    }

    /// Kills `hill_index` lines 394–407 (every operator mutant). All inputs
    /// are powers of two so the arithmetic is exact in Fix128:
    ///
    /// X_L = 2, X_T = 4, X_Z = 8 → 1/X² = ¼, 1/16, 1/64
    /// F = ½(1/16 + 1/64 − ¼) = −11/128, G = ½(1/64 + ¼ − 1/16) = 13/128,
    /// H = ½(¼ + 1/16 − 1/64) = 19/128
    /// S_LT = 2, S_LZ = 4, S_TZ = 8 → 1/S² = ¼, 1/16, 1/64
    /// σ = (3, 5, 11) → (σ_T−σ_Z)² = 36, (σ_Z−σ_L)² = 64, (σ_L−σ_T)² = 4
    /// τ_LT = 3, τ_LZ = 3, τ_TZ = 2
    ///
    /// f = (−11·36 + 13·64 + 19·4)/128 + 4/64 + 9/16 + 9/4
    ///   = 4 + 1/16 + 9/16 + 9/4 = 55/8.
    #[test]
    fn hill_index_dyadic_closed_form() {
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(2),
            x_l_compression_mpa: Fix128::from_int(2),
            x_t_tension_mpa: Fix128::from_int(4),
            x_t_compression_mpa: Fix128::from_int(4),
            x_z_tension_mpa: Fix128::from_int(8),
            x_z_compression_mpa: Fix128::from_int(8),
            s_lt_mpa: Fix128::from_int(2),
            s_lz_mpa: Fix128::from_int(4),
            s_tz_mpa: Fix128::from_int(8),
        };
        let stress = OrthotropicStress {
            sigma_l: Fix128::from_int(3),
            sigma_t: Fix128::from_int(5),
            sigma_z: Fix128::from_int(11),
            tau_lt: Fix128::from_int(3),
            tau_lz: Fix128::from_int(3),
            tau_tz: Fix128::from_int(2),
        };
        let f = hill_index(&stress, &s);
        assert_eq!(f, Fix128::from_ratio(55, 8), "hill = {}", f.to_f64());
        let r = evaluate_failure(&stress, &s, FailureCriterion::Hill);
        assert_eq!(r.failure_index, f);
        assert!(!r.is_safe);
    }

    /// Hill is pressure-insensitive: hydrostatic normal stress with no shear
    /// gives exactly 0 (all three differences vanish). Guards the `-` → `+`
    /// mutants at lines 398–400 with a second, independent fixture.
    #[test]
    fn hill_index_hydrostatic_is_zero() {
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(2),
            x_l_compression_mpa: Fix128::from_int(2),
            x_t_tension_mpa: Fix128::from_int(4),
            x_t_compression_mpa: Fix128::from_int(4),
            x_z_tension_mpa: Fix128::from_int(8),
            x_z_compression_mpa: Fix128::from_int(8),
            s_lt_mpa: Fix128::from_int(2),
            s_lz_mpa: Fix128::from_int(4),
            s_tz_mpa: Fix128::from_int(8),
        };
        let hydro = OrthotropicStress::axial(
            Fix128::from_int(7),
            Fix128::from_int(7),
            Fix128::from_int(7),
        );
        assert_eq!(hill_index(&hydro, &s), Fix128::ZERO);
    }

    /// Kills `tsai_wu_index` lines 435–466 (every operator mutant and the
    /// `-½` sign deletion at line 446). All strengths are powers of two so
    /// products, reciprocals and the Hoffman square roots are exact dyadics:
    ///
    /// X_Lt = 2, X_Lc = 8 → F_L = ½ − ⅛ = 3/8,  F_LL = 1/16
    /// X_Tt = 4, X_Tc = 16 → F_T = ¼ − 1/16 = 3/16, F_TT = 1/64
    /// X_Zt = 8, X_Zc = 32 → F_Z = ⅛ − 1/32 = 3/32, F_ZZ = 1/256
    /// S_LT = 4, S_LZ = 8, S_TZ = 16 → F_SS = 1/16, 1/64, 1/256
    /// F_LT = −½·√(1/1024) = −1/64, F_LZ = −½·√(1/4096) = −1/128,
    /// F_TZ = −½·√(1/16384) = −1/256
    /// σ = (3, 5, 7), τ_LT = 3, τ_LZ = 2, τ_TZ = 5
    ///
    /// f = 9/8 + 15/16 + 21/32 + 9/16 + 25/64 + 49/256 + 9/16 + 1/16 + 25/256
    ///     − 15/32 − 21/64 − 35/128 = 900/256 = 225/64.
    #[test]
    fn tsai_wu_index_dyadic_closed_form() {
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(2),
            x_l_compression_mpa: Fix128::from_int(8),
            x_t_tension_mpa: Fix128::from_int(4),
            x_t_compression_mpa: Fix128::from_int(16),
            x_z_tension_mpa: Fix128::from_int(8),
            x_z_compression_mpa: Fix128::from_int(32),
            s_lt_mpa: Fix128::from_int(4),
            s_lz_mpa: Fix128::from_int(8),
            s_tz_mpa: Fix128::from_int(16),
        };
        let stress = OrthotropicStress {
            sigma_l: Fix128::from_int(3),
            sigma_t: Fix128::from_int(5),
            sigma_z: Fix128::from_int(7),
            tau_lt: Fix128::from_int(3),
            tau_lz: Fix128::from_int(2),
            tau_tz: Fix128::from_int(5),
        };
        let f = tsai_wu_index(&stress, &s);
        assert_eq!(f, Fix128::from_ratio(225, 64), "tsai-wu = {}", f.to_f64());
        let r = evaluate_failure(&stress, &s, FailureCriterion::TsaiWu);
        assert_eq!(r.failure_index, f);
    }

    /// Tsai-Wu at the uniaxial strengths gives exactly 1 on both sides:
    /// σ_L = +X_Lt → F_L·X_Lt + F_LL·X_Lt² = (1/X_Lt − 1/X_Lc)·X_Lt + X_Lt/X_Lc = 1,
    /// σ_L = −X_Lc → −(1/X_Lt − 1/X_Lc)·X_Lc + X_Lc/X_Lt = 1.
    /// Kills the linear-term sign mutants (435–437) independently of the
    /// quadratic terms, and the same on the T and Z axes.
    #[test]
    fn tsai_wu_unit_at_uniaxial_strengths() {
        let s = AnisotropicStrength {
            x_l_tension_mpa: Fix128::from_int(2),
            x_l_compression_mpa: Fix128::from_int(8),
            x_t_tension_mpa: Fix128::from_int(4),
            x_t_compression_mpa: Fix128::from_int(16),
            x_z_tension_mpa: Fix128::from_int(8),
            x_z_compression_mpa: Fix128::from_int(32),
            s_lt_mpa: Fix128::from_int(4),
            s_lz_mpa: Fix128::from_int(8),
            s_tz_mpa: Fix128::from_int(16),
        };
        let z = Fix128::ZERO;
        let cases = [
            OrthotropicStress::axial(Fix128::from_int(2), z, z),
            OrthotropicStress::axial(Fix128::from_int(-8), z, z),
            OrthotropicStress::axial(z, Fix128::from_int(4), z),
            OrthotropicStress::axial(z, Fix128::from_int(-16), z),
            OrthotropicStress::axial(z, z, Fix128::from_int(8)),
            OrthotropicStress::axial(z, z, Fix128::from_int(-32)),
        ];
        for (i, st) in cases.iter().enumerate() {
            let f = tsai_wu_index(st, &s);
            assert_eq!(f, Fix128::ONE, "case {i}: tsai-wu = {}", f.to_f64());
        }
        // Pure shear at each shear strength is also exactly 1 (F_SS·S² = 1).
        let shear_cases = [
            (Fix128::from_int(4), z, z),
            (z, Fix128::from_int(8), z),
            (z, z, Fix128::from_int(16)),
        ];
        for (i, (lt, lz, tz)) in shear_cases.iter().enumerate() {
            let st = OrthotropicStress {
                sigma_l: z,
                sigma_t: z,
                sigma_z: z,
                tau_lt: *lt,
                tau_lz: *lz,
                tau_tz: *tz,
            };
            let f = tsai_wu_index(&st, &s);
            assert_eq!(f, Fix128::ONE, "shear case {i}: tsai-wu = {}", f.to_f64());
        }
    }
}
