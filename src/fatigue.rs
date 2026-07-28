//! Fatigue Life Prediction (S-N Curve + Miner's Rule)
//!
//! Phase D1 of the ALICE-Physics completeness project. Predicts the number of
//! stress cycles a part will survive under repeated loading, using the
//! classical **Woehler S-N curve** in Basquin power-law form together with
//! **Miner's linear damage accumulation rule** for variable-amplitude spectra.
//!
//! # Model
//!
//! Basquin's power law relates cycle count `N` to the alternating stress `S`:
//!
//! `N · Sᵐ = C`
//!
//! Equivalently, given a reference point `(N_ref, S_ref)`:
//!
//! `N(S) = N_ref · (S_ref / S)ᵐ`
//!
//! `m` (dimensionless integer exponent) is the fatigue slope:
//! - Polymers (PLA / ABS / PETG): m ≈ 5
//! - Aluminum alloys: m ≈ 6
//! - Steels: m ≈ 10
//!
//! Below the **endurance stress** `S_e` a metal survives an unlimited number
//! of cycles (typical steel behaviour). Polymers do **not** have a true
//! endurance limit — for these `S_e` is defined as the stress that survives
//! `10⁶` cycles, with continued log-linear degradation past that point. This
//! module currently applies the "true endurance" cutoff to all materials for
//! simplicity; refine at Phase E4 if creep-fatigue interaction matters.
//!
//! Miner's rule:
//!
//! `D = Σ nᵢ / N(Sᵢ)`, failure at `D ≥ 1`.
//!
//! # References
//!
//! - Basquin, "The exponential law of endurance tests", Proc. ASTM 10, 1910.
//! - Miner, "Cumulative damage in fatigue", J. Applied Mech. 12, 1945.
//! - Suresh, *Fatigue of Materials* 2nd ed. Ch. 5.
//! - ASME BPVC Section VIII, Division 2 (fatigue design curves).

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// SnCurve
// ============================================================================

/// Woehler S-N curve for a material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnCurve {
    /// Ultimate tensile strength (MPa). Cycle count = 1 approximate.
    pub ultimate_tensile_mpa: Fix128,
    /// Endurance / fatigue-limit stress (MPa). Below this, life is infinite.
    pub endurance_stress_mpa: Fix128,
    /// Cycle count at which the endurance stress is defined (typically 1e6
    /// for polymers, 1e7 for steels, but treated as the practical "infinite
    /// life" reference).
    pub endurance_cycles: u64,
    /// Basquin exponent m (integer). Larger m ⇒ steeper curve ⇒ more sensitive
    /// to small stress increases. Polymers 5, aluminum 6, steel 10.
    pub fatigue_exponent_m: u32,
}

impl SnCurve {
    /// Construct a reasonable S-N curve from an isotropic FDM material.
    ///
    /// Defaults: `S_e = 0.3 · UTS` (polymer rule of thumb), `N_e = 1e6`,
    /// `m = 5`.
    #[must_use]
    pub fn from_fdm_material(m: &MaterialProperties) -> Self {
        let uts = m.tensile_strength_mpa;
        let se = uts * Fix128::from_ratio(3, 10);
        Self {
            ultimate_tensile_mpa: uts,
            endurance_stress_mpa: se,
            endurance_cycles: 1_000_000,
            fatigue_exponent_m: 5,
        }
    }

    /// Standard steel preset (SUS304 grade): S_e = 0.5·UTS, m = 10.
    #[must_use]
    pub fn steel_sus304() -> Self {
        Self {
            ultimate_tensile_mpa: Fix128::from_int(505),
            endurance_stress_mpa: Fix128::from_int(240),
            endurance_cycles: 10_000_000,
            fatigue_exponent_m: 10,
        }
    }

    /// Aluminum A5052 preset: S_e = 0.4·UTS, m = 6.
    #[must_use]
    pub fn aluminum_a5052() -> Self {
        Self {
            ultimate_tensile_mpa: Fix128::from_int(230),
            endurance_stress_mpa: Fix128::from_int(92),
            endurance_cycles: 5_000_000,
            fatigue_exponent_m: 6,
        }
    }
}

/// Sentinel indicating infinite fatigue life (stress at or below endurance).
pub const INFINITE_LIFE: u64 = u64::MAX;

// ============================================================================
// Life predictions
// ============================================================================

/// Cycles to failure at a given alternating stress amplitude.
///
/// Returns `INFINITE_LIFE` when `stress_mpa ≤ endurance_stress_mpa`.
///
/// Uses Basquin: `N = N_e · (S_e / S)^m`. When `S > UTS` the model is
/// physically invalid; the function still returns a value (very small
/// cycle count) but the caller should treat this as "static failure".
#[must_use]
pub fn cycles_to_failure(curve: &SnCurve, stress_mpa: Fix128) -> u64 {
    if stress_mpa <= curve.endurance_stress_mpa {
        return INFINITE_LIFE;
    }
    if stress_mpa <= Fix128::ZERO {
        return INFINITE_LIFE;
    }

    // ratio = S_e / S  (dimensionless, < 1 since S > S_e)
    let ratio = curve.endurance_stress_mpa / stress_mpa;

    // ratio^m by repeated multiplication (m integer)
    let mut r_m = Fix128::ONE;
    for _ in 0..curve.fatigue_exponent_m {
        r_m = r_m * ratio;
    }

    // n = N_e · ratio^m
    let n_e = Fix128::from_int(curve.endurance_cycles as i64);
    let n_fp = n_e * r_m;

    // Convert Fix128 → u64 (round toward zero)
    let hi = n_fp.hi;
    if hi <= 0 {
        // Very-short life — one cycle minimum for physical meaning.
        1
    } else {
        hi as u64
    }
}

/// Alternating stress that would produce failure at exactly `cycles`.
///
/// The inverse of `cycles_to_failure`. For `cycles ≥ endurance_cycles`
/// returns `endurance_stress_mpa`; for `cycles == 0` returns UTS.
#[must_use]
pub fn stress_at_cycles(curve: &SnCurve, cycles: u64) -> Fix128 {
    if cycles == 0 {
        return curve.ultimate_tensile_mpa;
    }
    if cycles >= curve.endurance_cycles {
        return curve.endurance_stress_mpa;
    }
    // S = S_e · (N_e / N)^(1/m)
    // Since we cannot take fractional root cheaply, use a small Newton
    // iteration on f(x) = x^m − (N_e / N).
    let target = Fix128::from_int(curve.endurance_cycles as i64) / Fix128::from_int(cycles as i64);

    // Initial guess: geometric mean of 1 and target
    let mut x = target.sqrt();
    // Newton: x_{k+1} = x_k − (x^m − target) / (m · x^(m-1))
    for _ in 0..32 {
        let mut xm = Fix128::ONE;
        for _ in 0..curve.fatigue_exponent_m {
            xm = xm * x;
        }
        // xm_minus_1 = x^(m-1) = xm / x
        let mut xm1 = Fix128::ONE;
        for _ in 0..curve.fatigue_exponent_m - 1 {
            xm1 = xm1 * x;
        }
        let f = xm - target;
        let df = Fix128::from_int(curve.fatigue_exponent_m as i64) * xm1;
        if df.is_zero() {
            break;
        }
        x = x - f / df;
    }
    curve.endurance_stress_mpa * x
}

// ============================================================================
// Miner's rule
// ============================================================================

/// One entry in a stress spectrum: `(alternating stress, applied cycles)`.
pub type SpectrumEntry = (Fix128, u64);

/// Miner's rule cumulative damage `D = Σ nᵢ / Nᵢ`.
///
/// A part is expected to fail when `D ≥ 1`. Any spectrum entry at or below
/// the endurance limit contributes zero damage (infinite `Nᵢ`).
#[must_use]
pub fn miner_damage(spectrum: &[SpectrumEntry], curve: &SnCurve) -> Fix128 {
    let mut d = Fix128::ZERO;
    for &(stress, n) in spectrum {
        let n_fail = cycles_to_failure(curve, stress);
        if n_fail == INFINITE_LIFE {
            continue;
        }
        let ratio = Fix128::from_int(n as i64) / Fix128::from_int(n_fail as i64);
        d = d + ratio;
    }
    d
}

/// Cumulative damage report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FatigueReport {
    /// Total damage `D`.
    pub damage: Fix128,
    /// True iff `damage < 1` (part is expected to survive the spectrum).
    pub is_safe: bool,
    /// Safety factor `1 / D` — the multiplier by which the entire spectrum
    /// could be repeated before failure. Reported as a large sentinel for
    /// zero damage.
    pub safety_factor: Fix128,
}

/// Convenience wrapper that produces a full report.
#[must_use]
pub fn analyze_spectrum(spectrum: &[SpectrumEntry], curve: &SnCurve) -> FatigueReport {
    let damage = miner_damage(spectrum, curve);
    let sf = if damage.is_zero() {
        Fix128::from_int(i64::MAX >> 8)
    } else {
        Fix128::ONE / damage
    };
    FatigueReport {
        damage,
        is_safe: damage < Fix128::ONE,
        safety_factor: sf,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_material_uses_polymer_defaults() {
        let curve = SnCurve::from_fdm_material(&MaterialProperties::pla());
        assert_eq!(curve.fatigue_exponent_m, 5);
        assert_eq!(curve.endurance_cycles, 1_000_000);
        // S_e = 0.3 × 60 (PLA UTS) = 18 MPa (within Fix128 ULP)
        let diff = if curve.endurance_stress_mpa > Fix128::from_int(18) {
            curve.endurance_stress_mpa - Fix128::from_int(18)
        } else {
            Fix128::from_int(18) - curve.endurance_stress_mpa
        };
        assert!(diff < Fix128::from_ratio(1, 1000));
    }

    #[test]
    fn stress_below_endurance_is_infinite_life() {
        let curve = SnCurve::from_fdm_material(&MaterialProperties::pla());
        let n = cycles_to_failure(&curve, Fix128::from_int(10));
        assert_eq!(n, INFINITE_LIFE);
    }

    #[test]
    fn stress_at_endurance_is_infinite_life() {
        let curve = SnCurve::from_fdm_material(&MaterialProperties::pla());
        let n = cycles_to_failure(&curve, curve.endurance_stress_mpa);
        assert_eq!(n, INFINITE_LIFE);
    }

    #[test]
    fn higher_stress_shorter_life() {
        let curve = SnCurve::steel_sus304();
        let n_low = cycles_to_failure(&curve, Fix128::from_int(300));
        let n_high = cycles_to_failure(&curve, Fix128::from_int(400));
        assert!(n_high < n_low);
    }

    #[test]
    fn stress_at_endurance_cycles_returns_endurance() {
        let curve = SnCurve::steel_sus304();
        let s = stress_at_cycles(&curve, curve.endurance_cycles);
        assert_eq!(s, curve.endurance_stress_mpa);
    }

    #[test]
    fn stress_at_zero_cycles_returns_uts() {
        let curve = SnCurve::steel_sus304();
        let s = stress_at_cycles(&curve, 0);
        assert_eq!(s, curve.ultimate_tensile_mpa);
    }

    #[test]
    fn stress_at_cycles_monotonic_decreasing() {
        let curve = SnCurve::aluminum_a5052();
        let s_short = stress_at_cycles(&curve, 100);
        let s_long = stress_at_cycles(&curve, 100_000);
        assert!(s_short > s_long);
    }

    #[test]
    fn miner_damage_empty_spectrum_is_zero() {
        let curve = SnCurve::from_fdm_material(&MaterialProperties::pla());
        let d = miner_damage(&[], &curve);
        assert_eq!(d, Fix128::ZERO);
    }

    #[test]
    fn miner_damage_below_endurance_is_zero() {
        let curve = SnCurve::steel_sus304();
        let d = miner_damage(&[(Fix128::from_int(100), 1_000_000)], &curve);
        // Below 240 MPa → no damage
        assert_eq!(d, Fix128::ZERO);
    }

    #[test]
    fn miner_damage_at_full_life_is_one() {
        let curve = SnCurve::steel_sus304();
        let stress = Fix128::from_int(400);
        let n_fail = cycles_to_failure(&curve, stress);
        let d = miner_damage(&[(stress, n_fail)], &curve);
        // Full life consumed → D ≈ 1
        let diff = if d > Fix128::ONE {
            d - Fix128::ONE
        } else {
            Fix128::ONE - d
        };
        assert!(diff < Fix128::from_ratio(1, 100), "D was {}", d.to_f32());
    }

    #[test]
    fn miner_damage_additive_over_multiple_stresses() {
        let curve = SnCurve::steel_sus304();
        let entry_a = (Fix128::from_int(300), 100u64);
        let entry_b = (Fix128::from_int(350), 50u64);
        let d_ab = miner_damage(&[entry_a, entry_b], &curve);
        let d_a = miner_damage(&[entry_a], &curve);
        let d_b = miner_damage(&[entry_b], &curve);
        assert_eq!(d_ab, d_a + d_b);
    }

    #[test]
    fn analyze_spectrum_safe_report() {
        let curve = SnCurve::steel_sus304();
        // 100 cycles at 300 MPa — tiny damage
        let report = analyze_spectrum(&[(Fix128::from_int(300), 100)], &curve);
        assert!(report.is_safe);
        assert!(report.damage > Fix128::ZERO);
        assert!(report.safety_factor > Fix128::from_int(1000));
    }

    #[test]
    fn analyze_spectrum_unsafe_when_overloaded() {
        let curve = SnCurve::steel_sus304();
        // Consume triple the life at endurance-doubling stress
        let stress = Fix128::from_int(480);
        let n_fail = cycles_to_failure(&curve, stress);
        let report = analyze_spectrum(&[(stress, n_fail * 3)], &curve);
        assert!(!report.is_safe);
        assert!(report.damage > Fix128::ONE);
    }

    #[test]
    fn steel_has_higher_endurance_than_polymer() {
        let curve_steel = SnCurve::steel_sus304();
        let curve_pla = SnCurve::from_fdm_material(&MaterialProperties::pla());
        assert!(curve_steel.endurance_stress_mpa > curve_pla.endurance_stress_mpa);
    }
}
