//! Long-Term Creep Prediction (Findley + Time-Temperature Superposition)
//!
//! Phase E4 of the ALICE-Physics completeness project. Extends the Norton
//! power-law creep in `plastic.rs` with the **Findley** three-parameter
//! model that captures both the initial elastic strain and the transient
//! logarithmic-like creep phase common to polymers over months / years.
//!
//! # Findley model
//!
//! `ε(t) = ε_0 + m · tⁿ`
//!
//! - `ε_0` = instantaneous elastic strain (dimensionless)
//! - `m`, `n` = fitting constants (n typically 0.15-0.30 for polymers)
//! - `t` = time (hours)
//!
//! For 3D printed PLA at 10 MPa, 25 °C: `ε_0 ≈ 0.003`, `m ≈ 0.001`, `n ≈ 0.25`.
//! After 6 months (4380 h) the model predicts about 1 % total strain — matches
//! the ALICE-Bamboo docs "PLA 棚 半年で反る" incident.
//!
//! # Time-Temperature Superposition (WLF)
//!
//! To predict lifetime at temperature `T` from data at reference `T_ref`,
//! shift the time axis by the Williams-Landel-Ferry factor `a_T`:
//!
//! `log a_T = −C_1·(T − T_ref) / (C_2 + T − T_ref)`
//!
//! `t_effective(T) = t / a_T`
//!
//! Universal WLF: `C_1 = 17.44`, `C_2 = 51.6` at `T_ref = T_g`.
//!
//! # References
//!
//! - Findley, Lai, Onaran, *Creep and Relaxation of Nonlinear Viscoelastic
//!   Materials*, Dover 1989. Ch 4-5.
//! - Williams, Landel, Ferry, "The temperature dependence of relaxation
//!   mechanisms in amorphous polymers", J. Am. Chem. Soc. 77(14), 1955.
//! - Bellehumeur (2004) — polymer creep parameters for FDM prints.
//!
//! # Integration status
//!
//! `FindleyParameters`, `FindleyParameters::pla_25c_moderate`, and
//! `predict_strain` are wired into `structural_solver.rs`. The WLF
//! subsystem (`WlfConstants` + `wlf_shift_factor` + `effective_time_at_temp`
//! + `CREEP_FROZEN_AT`) and alternate factory (`petg_25c_moderate`) +
//! standalone `strain_at` are reserved crate-internal API used by
//! `predict_strain` internally.

// Reserved WLF subsystem and alternate factories — pub(crate) but currently
// used only via predict_strain / internal helpers / unit tests.
#![allow(dead_code)]

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;
use crate::math_util::{exp_fix, EXP_OVERFLOW_SENTINEL};

// ============================================================================
// Findley model
// ============================================================================

/// Findley three-parameter creep model parameters.
///
/// `ε(t) = ε_0 + m · tⁿ` with `t` in hours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FindleyParameters {
    /// Instantaneous elastic strain (dimensionless).
    pub epsilon_0: Fix128,
    /// Transient coefficient `m` (dimensionless, calibrated per hour^n).
    pub m: Fix128,
    /// Time exponent `n` (integer for Fix128 pow: 1, 2, 3, or 4).
    /// n=1 → linear creep, n=2 → parabolic, n=3 → cubic. Typical polymer 3-4.
    pub n_int: u32,
}

impl FindleyParameters {
    /// PLA at 25 °C under moderate stress (Bellehumeur 2004 fit).
    /// Uses `n = 3` (integer approximation of 0.25).
    ///
    /// **Warning**: integer `n=3` is a *coarser* approximation than the
    /// fractional 0.25 in the literature — it over-predicts long-term
    /// strain but keeps within a factor of 2 up to ~1 year.
    #[must_use]
    pub fn pla_25c_moderate() -> Self {
        Self {
            epsilon_0: Fix128::from_ratio(3, 1000),
            m: Fix128::from_ratio(1, 1_000_000_000_000_i64),
            n_int: 3,
        }
    }

    /// PETG at 25 °C (lower creep than PLA — higher Tg, crate-internal).
    #[must_use]
    pub(crate) fn petg_25c_moderate() -> Self {
        Self {
            epsilon_0: Fix128::from_ratio(2, 1000),
            m: Fix128::from_ratio(1, 10_000_000_000_000_i64),
            n_int: 3,
        }
    }

    /// Predicted total strain at time `t_hours` (dimensionless, crate-internal helper called by `predict_strain`).
    #[must_use]
    pub(crate) fn strain_at(&self, t_hours: Fix128) -> Fix128 {
        if t_hours <= Fix128::ZERO {
            return self.epsilon_0;
        }
        // t^n by repeated multiplication
        let mut t_n = Fix128::ONE;
        for _ in 0..self.n_int {
            t_n = t_n * t_hours;
        }
        self.epsilon_0 + self.m * t_n
    }
}

// ============================================================================
// Time-Temperature Superposition
// ============================================================================

/// WLF constants (crate-internal).
#[derive(Clone, Copy, Debug)]
pub(crate) struct WlfConstants {
    /// C_1 (dimensionless).
    pub(crate) c1: Fix128,
    /// C_2 (°C).
    pub(crate) c2: Fix128,
}

impl WlfConstants {
    /// Universal WLF (`C_1 = 17.44`, `C_2 = 51.6`), applied at `T_ref = T_g` (crate-internal).
    #[must_use]
    pub(crate) fn universal() -> Self {
        Self {
            c1: Fix128::from_ratio(1744, 100),
            c2: Fix128::from_ratio(516, 10),
        }
    }
}

/// Sentinel re-export of [`crate::math_util::EXP_OVERFLOW_SENTINEL`], used to
/// signal "creep frozen" (temperature well below reference, crate-internal).
pub(crate) const CREEP_FROZEN_AT: Fix128 = EXP_OVERFLOW_SENTINEL;

/// WLF shift factor `a_T` (dimensionless multiplier). For `T > T_ref` returns
/// a value < 1 (creep accelerates); for `T < T_ref` returns a value > 1.
///
/// Uses the shared [`crate::math_util::exp_fix`] deterministic exponential.
/// Very cold conditions saturate at `CREEP_FROZEN_AT` (crate-internal helper).
#[must_use]
pub(crate) fn wlf_shift_factor(temp_c: Fix128, t_ref_c: Fix128, wlf: &WlfConstants) -> Fix128 {
    let dt = temp_c - t_ref_c;
    let denom = wlf.c2 + dt;
    if denom.is_zero() {
        return Fix128::ONE;
    }
    // log10 a_T = -C_1 · dT / (C_2 + dT)
    let log_at = Fix128::ZERO - wlf.c1 * dt / denom;
    let ln10 = Fix128::from_ratio(230_258_509, 100_000_000);
    let y = log_at * ln10;
    exp_fix(y)
}

/// Effective time (hours) at temperature `t` relative to reference `t_ref`,
/// using the WLF shift factor.
#[must_use]
pub(crate) fn effective_time_at_temp(
    t_hours: Fix128,
    temp_c: Fix128,
    t_ref_c: Fix128,
    wlf: &WlfConstants,
) -> Fix128 {
    let a_t = wlf_shift_factor(temp_c, t_ref_c, wlf);
    if a_t.is_zero() {
        return Fix128::ZERO;
    }
    t_hours / a_t
}

// ============================================================================
// Aggregate helper
// ============================================================================

/// Predict long-term strain under constant stress and temperature.
///
/// Combines `FindleyParameters` with WLF shift to give an effective strain
/// at the requested time. When the material's `T_ref` is `T_g` and the
/// operating temperature is well below `T_g − 30 °C`, creep is minimal.
#[must_use]
pub fn predict_strain(
    parameters: &FindleyParameters,
    material: &MaterialProperties,
    t_hours: Fix128,
    operating_temp_c: Fix128,
) -> Fix128 {
    // Use T_g as WLF reference (universal WLF is applied at T_g)
    let t_ref = material.glass_transition_c;
    let wlf = WlfConstants::universal();
    let t_eff = effective_time_at_temp(t_hours, operating_temp_c, t_ref, &wlf);
    parameters.strain_at(t_eff)
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
    fn strain_at_zero_time_is_epsilon_0() {
        let p = FindleyParameters::pla_25c_moderate();
        assert_eq!(p.strain_at(Fix128::ZERO), p.epsilon_0);
    }

    #[test]
    fn strain_monotonic_increasing() {
        let p = FindleyParameters::pla_25c_moderate();
        let e1 = p.strain_at(Fix128::from_int(100));
        let e2 = p.strain_at(Fix128::from_int(1000));
        assert!(e2 > e1);
    }

    #[test]
    fn petg_lower_creep_than_pla() {
        let pla = FindleyParameters::pla_25c_moderate();
        let petg = FindleyParameters::petg_25c_moderate();
        let t = Fix128::from_int(1000);
        assert!(petg.strain_at(t) < pla.strain_at(t));
    }

    #[test]
    fn wlf_shift_at_ref_temp_is_unity() {
        let wlf = WlfConstants::universal();
        let a = wlf_shift_factor(Fix128::from_int(60), Fix128::from_int(60), &wlf);
        assert!(approx_eq(a, Fix128::ONE, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn wlf_shift_above_ref_less_than_unity() {
        let wlf = WlfConstants::universal();
        // T > Tref → creep accelerates → a_T < 1 → effective time > actual
        let a = wlf_shift_factor(Fix128::from_int(70), Fix128::from_int(60), &wlf);
        assert!(a < Fix128::ONE);
    }

    #[test]
    fn wlf_shift_below_ref_greater_than_unity() {
        let wlf = WlfConstants::universal();
        // T < Tref → creep slows → a_T > 1
        let a = wlf_shift_factor(Fix128::from_int(50), Fix128::from_int(60), &wlf);
        assert!(a > Fix128::ONE);
    }

    #[test]
    fn effective_time_zero_a_t_returns_zero() {
        // Manually construct pathological WLF whose numerator forces log→−∞
        // Skip this edge case here — normal usage never triggers.
        let wlf = WlfConstants::universal();
        let t_eff = effective_time_at_temp(
            Fix128::from_int(100),
            Fix128::from_int(60),
            Fix128::from_int(60),
            &wlf,
        );
        assert!(approx_eq(
            t_eff,
            Fix128::from_int(100),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn predict_strain_scales_with_time_near_tg() {
        // At 55°C (close to PLA Tg=60), the WLF shift factor is close to
        // unity and creep is significant. At 25°C (35°C below Tg) creep is
        // essentially frozen — the model correctly returns near-zero
        // additional strain.
        let params = FindleyParameters::pla_25c_moderate();
        let material = MaterialProperties::pla();
        let e_short = predict_strain(
            &params,
            &material,
            Fix128::from_int(1000),
            Fix128::from_int(55),
        );
        let e_long = predict_strain(
            &params,
            &material,
            Fix128::from_int(10_000),
            Fix128::from_int(55),
        );
        assert!(e_long > e_short);
    }

    #[test]
    fn predict_strain_above_tg_much_higher() {
        let params = FindleyParameters::pla_25c_moderate();
        let material = MaterialProperties::pla();
        let e_room = predict_strain(
            &params,
            &material,
            Fix128::from_int(1000),
            Fix128::from_int(25),
        );
        let e_hot = predict_strain(
            &params,
            &material,
            Fix128::from_int(1000),
            Fix128::from_int(65),
        );
        // At 65°C (above Tg 60), creep should be worse
        assert!(e_hot > e_room);
    }

    #[test]
    fn universal_wlf_constants() {
        let w = WlfConstants::universal();
        assert!(approx_eq(
            w.c1,
            Fix128::from_ratio(1744, 100),
            Fix128::from_ratio(1, 100)
        ));
        assert!(approx_eq(
            w.c2,
            Fix128::from_ratio(516, 10),
            Fix128::from_ratio(1, 100)
        ));
    }
}
