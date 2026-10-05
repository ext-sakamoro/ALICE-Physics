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
//!
//! # Integration status
//!
//! `SnCurve`, `SpectrumEntry`, and `miner_damage` are wired into
//! `structural_solver.rs`. `StructuralSolver::new` selects the curve with
//! [`SnCurve::for_material`]: the metal presets ([`SnCurve::steel_sus304`] /
//! [`SnCurve::aluminum_a5052`]) for the crate's `MaterialProperties::sus304()`
//! / `a5052()` presets (sheet-metal category and the preset name), and
//! [`SnCurve::from_fdm_material`] for every other material.
//! `StructuralSolver::with_sn_curve` overrides the selection. The Basquin
//! inverse `stress_at_cycles` is reached through
//! `StructuralSolver::fatigue_strength_mpa` and the `analyze_spectrum`
//! wrapper through `StructuralSolver::fatigue_spectrum_report`, both
//! evaluated on the solver's selected curve. `INFINITE_LIFE` and
//! `cycles_to_failure` are crate-internal and reached through
//! `miner_damage`.

use crate::filament_db::{MaterialCategory, MaterialProperties};
use crate::math::Fix128;

/// Name carried by `MaterialProperties::sus304()`.
const SUS304_PRESET_NAME: &str = "SUS304";
/// Name carried by `MaterialProperties::a5052()`.
const A5052_PRESET_NAME: &str = "A5052";

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

    /// Curve for `material`: [`Self::steel_sus304`] for the crate's SUS304
    /// preset and [`Self::aluminum_a5052`] for its A5052 preset (both
    /// identified by the sheet-metal category and the preset name, as
    /// `MaterialProperties::sus304()` / `a5052()` carry them), and
    /// [`Self::from_fdm_material`] for every other material, sheet metals
    /// with another name included.
    #[must_use]
    pub fn for_material(material: &MaterialProperties) -> Self {
        if material.category == MaterialCategory::SheetMetal {
            if material.name == SUS304_PRESET_NAME {
                return Self::steel_sus304();
            }
            if material.name == A5052_PRESET_NAME {
                return Self::aluminum_a5052();
            }
        }
        Self::from_fdm_material(material)
    }

    /// Austenitic stainless steel preset (SUS304 grade): `UTS = 505 MPa`,
    /// `S_e = 240 MPa` (`0.475 · UTS`), `N_e = 10⁷`, `m = 10`.
    #[must_use]
    pub fn steel_sus304() -> Self {
        Self {
            ultimate_tensile_mpa: Fix128::from_int(505),
            endurance_stress_mpa: Fix128::from_int(240),
            endurance_cycles: 10_000_000,
            fatigue_exponent_m: 10,
        }
    }

    /// Aluminum A5052 preset: `UTS = 230 MPa`, `S_e = 92 MPa` (`0.4 · UTS`),
    /// `N_e = 5·10⁶`, `m = 6`.
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

/// Sentinel indicating infinite fatigue life (stress at or below endurance, crate-internal).
pub(crate) const INFINITE_LIFE: u64 = u64::MAX;

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
pub(crate) fn cycles_to_failure(curve: &SnCurve, stress_mpa: Fix128) -> u64 {
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

/// Smallest cycle count the Basquin inverse answers for.
///
/// Basquin's law is a high-cycle fit (about `10³` to `10⁷` cycles); below
/// it the formula returns stresses above the ultimate strength that are
/// numerically exact and physically meaningless, so the query is refused
/// rather than clamped (a clamp would hand back a plausible number and hide
/// that the law does not apply).
///
/// Provisional: the low-cycle boundary depends on the material and the
/// stress ratio, so this single constant is a placeholder for a per-curve
/// field of `SnCurve` (Backlog `fatigue-low-cycle-bound-per-material`).
pub(crate) const BASQUIN_LOW_CYCLE_BOUND: u64 = 1_000;

/// Why the Basquin inverse (`StructuralSolver::fatigue_strength_mpa`) could
/// not answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FatigueRangeError {
    /// `cycles == 0` has no stress on the S–N curve.
    ZeroCycles,
    /// Below the low-cycle bound (currently 1 000 cycles for every curve),
    /// outside the law's range.
    BelowLowCycleBound {
        /// The cycle count asked for.
        cycles: u64,
        /// The bound it fell under.
        bound: u64,
    },
    /// The curve's exponent `m` is zero: `(N_e/N)^(1/m)` is undefined.
    NonPositiveExponent,
}

/// Alternating stress that would produce failure at exactly `cycles`.
///
/// The inverse of the Basquin life `N = N_e · (S_e / S)^m`, i.e.
/// `S = S_e · (N_e / N)^(1/m)`, evaluated as the fixed-point power
/// `Fix128::powf_pos` (24 fractional bits of the exponent) polished by two
/// Newton steps on `x^m = N_e/N` taken from that seed. The seed sits within
/// `ln(N_e/N) · 2⁻²⁴` of the root, so `x^m` never leaves the range of
/// `Fix128` (the pre-2026-10-03 version seeded with `√(N_e/N)` and wrapped
/// `x^m` for every `N_e/N > 2^(126/m)`, returning e.g. 760 490 MPa for
/// SUS304 at `N = 1`); the two polishes are a fixed count, not a loop to a
/// tolerance, so there is no drift at the rounding floor.
///
/// For `cycles ≥ endurance_cycles` returns `endurance_stress_mpa` exactly.
///
/// # Errors
///
/// [`FatigueRangeError::ZeroCycles`] for `cycles == 0`,
/// [`FatigueRangeError::BelowLowCycleBound`] under
/// [`BASQUIN_LOW_CYCLE_BOUND`], [`FatigueRangeError::NonPositiveExponent`]
/// for a curve with `fatigue_exponent_m == 0`.
pub(crate) fn stress_at_cycles(curve: &SnCurve, cycles: u64) -> Result<Fix128, FatigueRangeError> {
    if cycles == 0 {
        return Err(FatigueRangeError::ZeroCycles);
    }
    if curve.fatigue_exponent_m == 0 {
        return Err(FatigueRangeError::NonPositiveExponent);
    }
    if cycles < BASQUIN_LOW_CYCLE_BOUND {
        return Err(FatigueRangeError::BelowLowCycleBound {
            cycles,
            bound: BASQUIN_LOW_CYCLE_BOUND,
        });
    }
    if cycles >= curve.endurance_cycles {
        return Ok(curve.endurance_stress_mpa);
    }
    let target = Fix128::from_int(curve.endurance_cycles as i64) / Fix128::from_int(cycles as i64);
    let m = Fix128::from_int(i64::from(curve.fatigue_exponent_m));
    let mut x = target.powf_pos(Fix128::ONE / m);
    for _ in 0..2 {
        let mut xm1 = Fix128::ONE;
        for _ in 1..curve.fatigue_exponent_m {
            xm1 = xm1 * x;
        }
        let xm = xm1 * x;
        let df = m * xm1;
        if df.is_zero() {
            break;
        }
        x = x - (xm - target) / df;
    }
    Ok(curve.endurance_stress_mpa * x)
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

/// Cumulative damage report for one stress spectrum, returned by
/// `StructuralSolver::fatigue_spectrum_report`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FatigueReport {
    /// Total damage `D`.
    pub damage: Fix128,
    /// True iff `damage < 1` (part is expected to survive the spectrum).
    pub is_safe: bool,
    /// Safety factor `1 / D` — the multiplier by which the entire spectrum
    /// could be repeated before failure. Reported as the sentinel
    /// `Fix128::from_int(i64::MAX >> 8)` for zero damage.
    pub safety_factor: Fix128,
}

/// Miner's rule over the whole spectrum, packaged as a [`FatigueReport`]
/// (`damage` is exactly [`miner_damage`], crate-internal).
#[must_use]
pub(crate) fn analyze_spectrum(spectrum: &[SpectrumEntry], curve: &SnCurve) -> FatigueReport {
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
        let s = stress_at_cycles(&curve, curve.endurance_cycles).unwrap();
        assert_eq!(s, curve.endurance_stress_mpa);
    }

    /// Zero cycles has no stress on the curve: refused, not UTS (the
    /// pre-2026-10-03 contract returned UTS, a number outside the law).
    #[test]
    fn stress_at_zero_cycles_is_refused() {
        let curve = SnCurve::steel_sus304();
        assert_eq!(
            stress_at_cycles(&curve, 0),
            Err(FatigueRangeError::ZeroCycles)
        );
    }

    /// Oracle: `S` decreases with `N` over the whole high-cycle range, and
    /// the round trip `cycles_to_failure(stress_at_cycles(N))` lands within
    /// 0.1 % of `N` (the forward map truncates to an integer).
    #[test]
    fn stress_at_cycles_monotonic_decreasing_and_round_trips() {
        let curve = SnCurve::aluminum_a5052();
        let mut previous = Fix128::from_int(1_000_000);
        for n in [
            1_000u64, 2_000, 5_000, 20_000, 100_000, 1_000_000, 4_000_000,
        ] {
            let s = stress_at_cycles(&curve, n).unwrap();
            assert!(s < previous, "N = {n}: {s:?} not below {previous:?}");
            previous = s;
            let back = cycles_to_failure(&curve, s);
            assert!(
                back.abs_diff(n) * 1000 <= n,
                "N = {n}: round trip gave {back}"
            );
        }
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

    /// Oracle: Basquin `N = N_e · (S_e / S)^m` with the SUS304 preset at
    /// `S = 300 MPa`: `10⁷ · 0.8¹⁰ = 10⁷ · 0.1073741824 = 1 073 741.824`,
    /// truncated toward zero to `1 073 741` cycles. Applying exactly that
    /// many cycles gives Miner `D = 1` (unsafe, safety factor 1); one cycle
    /// fewer is still safe.
    #[test]
    fn sus304_preset_basquin_life_at_300_mpa_is_1073741_cycles() {
        let curve = SnCurve::steel_sus304();
        let s = Fix128::from_int(300);
        let full = analyze_spectrum(&[(s, 1_073_741)], &curve);
        assert_eq!(full.damage, Fix128::ONE, "D = {}", full.damage.to_f64());
        assert!(!full.is_safe);
        assert_eq!(full.safety_factor, Fix128::ONE);
        let almost = analyze_spectrum(&[(s, 1_073_740)], &curve);
        assert!(almost.is_safe);
        assert!(almost.damage < Fix128::ONE);
    }

    /// Oracle: Miner's rule by hand for the A5052 preset
    /// (`S_e = 92`, `N_e = 5·10⁶`, `m = 6`).
    /// `N(120) = ⌊5·10⁶ · (23/30)⁶⌋ = ⌊5·10⁶ · 148 035 889 / 729 000 000⌋
    ///  = 1 015 335`, `N(150) = ⌊5·10⁶ · 46⁶ / 75⁶⌋ = 266 164`.
    /// Applying `N(120)/5 = 203 067` and `N(150)/4 = 66 541` cycles gives
    /// `D = 0.2 + 0.25 = 0.45` and a safety factor of `1/0.45 = 2.2̅`.
    #[test]
    fn a5052_preset_miner_sum_matches_hand_calculation() {
        let n120: u128 = 5_000_000 * 148_035_889 / 729_000_000;
        let n150: u128 = 5_000_000 * 9_474_296_896 / 177_978_515_625;
        assert_eq!((n120, n150), (1_015_335, 266_164), "hand calculation");
        let curve = SnCurve::aluminum_a5052();
        let report = analyze_spectrum(
            &[
                (Fix128::from_int(120), 203_067),
                (Fix128::from_int(150), 66_541),
            ],
            &curve,
        );
        let d = report.damage.to_f64();
        assert!((d - 0.45).abs() < 1e-12, "D = {d}");
        assert!(report.is_safe);
        let sf = report.safety_factor.to_f64();
        assert!((sf - 1.0 / 0.45).abs() < 1e-12, "SF = {sf}");
    }

    /// Degenerate spectra documented on `analyze_spectrum` /
    /// `cycles_to_failure`: an empty spectrum and an entry at the endurance
    /// stress (infinite life, even with `u64::MAX` cycles) both report zero
    /// damage, `is_safe`, and the zero-damage safety-factor sentinel
    /// `Fix128::from_int(i64::MAX >> 8)`.
    #[test]
    fn analyze_spectrum_empty_and_below_endurance_report_zero_damage_sentinel() {
        let curve = SnCurve::steel_sus304();
        let sentinel = Fix128::from_int(i64::MAX >> 8);
        for spectrum in [&[][..], &[(Fix128::from_int(240), u64::MAX)][..]] {
            let report = analyze_spectrum(spectrum, &curve);
            assert_eq!(report.damage, Fix128::ZERO);
            assert!(report.is_safe);
            assert_eq!(report.safety_factor, sentinel);
        }
    }

    /// Oracle: the same Basquin inverse at small cycle counts, where the
    /// Newton seed `x₀ = √(N_e/N)` makes `x₀^m = (N_e/N)^(m/2)` exceed the
    /// Fix128 integer range (`2⁶³ ≈ 9.2e18`): SUS304 (`m = 10`) wraps for
    /// `N < 1e7 / 6191 ≈ 1615`, A5052 (`m = 6`) for `N < 5e6 / 2.1e6 ≈ 2.4`.
    /// Hand values: `240 · (10⁷)^0.1 = 240 · 5.0118723 = 1202.849 MPa`,
    /// `92 · (5·10⁶)^(1/6) = 92 · 13.076605 = 1203.048 MPa`. Measured
    /// 2026-10-03: `760 490.6` and `206 835.0` MPa (wrapped Newton), and
    /// SUS304 at `N = 5000` is still `1.3e-4 MPa` off (not converged).
    /// Oracle: the cycle counts at which the pre-2026-10-03 Newton seed
    /// wrapped (`SUS304 N ≤ 1500`, `A5052 N ≤ 2`) now give the closed form
    /// `S_e · (N_e/N)^(1/m)` to 1e-6 MPa where the law applies, and the
    /// documented refusal below the low-cycle bound.
    #[test]
    fn stress_at_cycles_small_cycle_counts_match_basquin() {
        let steel = SnCurve::steel_sus304();
        for n in [1_000u64, 1_500, 2_000, 5_000] {
            let got = stress_at_cycles(&steel, n).unwrap().to_f64();
            let want = 240.0 * crate::det_math::powf64(1e7 / n as f64, 0.1);
            assert!(
                (got - want).abs() < 1e-6,
                "SUS304 N = {n}: got {got}, want {want}"
            );
        }
        assert_eq!(
            stress_at_cycles(&steel, 1),
            Err(FatigueRangeError::BelowLowCycleBound {
                cycles: 1,
                bound: 1_000
            })
        );
        let alu = SnCurve::aluminum_a5052();
        assert_eq!(
            stress_at_cycles(&alu, 2),
            Err(FatigueRangeError::BelowLowCycleBound {
                cycles: 2,
                bound: 1_000
            })
        );
        let got = stress_at_cycles(&alu, 1_000).unwrap().to_f64();
        let want = 92.0 * crate::det_math::powf64(5e3, 1.0 / 6.0);
        assert!(
            (got - want).abs() < 1e-6,
            "A5052 N = 1000: got {got}, want {want}"
        );
    }

    /// Oracle: `S = S_e · (N_e / N)^(1/m)` by hand.
    /// SUS304 at `N = 10⁵`: `240 · 100^(0.1) = 240 · 1.5848932 = 380.3744 MPa`.
    /// A5052 at `N = 5·10⁵`: `92 · 10^(1/6) = 92 · 1.4677993 = 135.0375 MPa`.
    /// The Newton iteration stops at `|Δx| < 1e-6`, so the result is good
    /// to far better than the `1e-6 MPa` asked for here.
    #[test]
    fn stress_at_cycles_inverts_basquin_for_sus304_and_a5052() {
        let steel = stress_at_cycles(&SnCurve::steel_sus304(), 100_000)
            .unwrap()
            .to_f64();
        let want_steel = 240.0 * crate::det_math::powf64(100.0, 0.1);
        assert!(
            (steel - want_steel).abs() < 1e-6,
            "got {steel}, want {want_steel}"
        );
        let alu = stress_at_cycles(&SnCurve::aluminum_a5052(), 500_000)
            .unwrap()
            .to_f64();
        let want_alu = 92.0 * crate::det_math::powf64(10.0, 1.0 / 6.0);
        assert!((alu - want_alu).abs() < 1e-6, "got {alu}, want {want_alu}");
    }

    /// Degenerate cycle counts documented on `stress_at_cycles`: `0` is
    /// refused, `N_e` and anything above (up to `u64::MAX`, which never
    /// reaches the `as i64` conversion) return the endurance stress exactly,
    /// and a curve with `m = 0` is refused.
    #[test]
    fn stress_at_cycles_degenerate_cycle_counts() {
        let curve = SnCurve::aluminum_a5052();
        assert_eq!(
            stress_at_cycles(&curve, 0),
            Err(FatigueRangeError::ZeroCycles)
        );
        assert_eq!(
            stress_at_cycles(&curve, curve.endurance_cycles),
            Ok(curve.endurance_stress_mpa)
        );
        assert_eq!(
            stress_at_cycles(&curve, u64::MAX),
            Ok(curve.endurance_stress_mpa)
        );
        let mut flat = SnCurve::aluminum_a5052();
        flat.fatigue_exponent_m = 0;
        assert_eq!(
            stress_at_cycles(&flat, 10_000),
            Err(FatigueRangeError::NonPositiveExponent)
        );
    }

    #[test]
    fn steel_has_higher_endurance_than_polymer() {
        let curve_steel = SnCurve::steel_sus304();
        let curve_pla = SnCurve::from_fdm_material(&MaterialProperties::pla());
        assert!(curve_steel.endurance_stress_mpa > curve_pla.endurance_stress_mpa);
    }
}
