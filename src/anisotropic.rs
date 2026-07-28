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
}
