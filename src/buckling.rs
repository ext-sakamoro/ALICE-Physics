//! Column and Shell Buckling Analysis
//!
//! Phase B3 of the ALICE-Physics completeness project. Extends the basic
//! Euler buckling formula already provided in `beam_stress.rs` with the
//! **slenderness-ratio family** of column-stability criteria:
//!
//! - Radius of gyration `r = √(I / A)`.
//! - Slenderness ratio `λ = K·L / r`.
//! - **Euler regime** (`λ > λ_transition`): `σ_cr = π²·E / λ²`.
//! - **Johnson parabolic regime** (`λ < λ_transition`): captures short
//!   ("stocky") columns where Euler massively over-predicts.
//! - **Transition slenderness** `λ_t = π · √(2E / σ_y)`.
//! - **Local plate buckling** for thin-walled hollow sections (Timoshenko).
//! - **Snap-through** critical load for shallow arches / clip-in features.
//!
//! # Why these matter for 3D printing
//!
//! SKADIS pegs, snap-fit tabs, and any thin extrusion behave as slender
//! columns under bench load. Pure Euler over-predicts the safe load by 10-100×
//! when the column is stocky; the Johnson correction gives realistic values.
//! Local plate buckling matters for hollow-square supports where the walls
//! bulge before the whole column bends.
//!
//! # References
//!
//! - Timoshenko & Gere, *Theory of Elastic Stability* 2nd ed. Chapters 1
//!   (columns), 8 (plate buckling), 11 (shells).
//! - Johnson, "Column and Strut Formulas", ASCE Trans. 42 (1899) —
//!   parabolic formula for short columns.
//! - Bažant & Cedolin, *Stability of Structures* (snap-through).
//!
//! # Integration status
//!
//! `analyze_column`, `ColumnBucklingReport`, and `BucklingRegime` are
//! wired into `structural_solver.rs`. The radius-of-gyration / slenderness
//! helpers are crate-internal and reached through `analyze_column`.
//! `plate_buckling_mpa` (local wall buckling of a hollow section) and
//! `snap_through_load_n` (shallow arch / clip) are reached through
//! `StructuralSolver::plate_buckling_mpa` / `StructuralSolver::snap_through_load_n`,
//! which take Young's modulus from the solver's material (and, for
//! snap-through, the bar area from its cross-section). The life loop
//! (`StructuralSolver::step`) does not trip on either: the plate width,
//! edge factor and Poisson ratio are caller inputs that neither
//! `CrossSection` nor `MaterialProperties` carries.

use crate::beam_stress::{ColumnEndCondition, CrossSection};
use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Radius of gyration & slenderness
// ============================================================================

/// Radius of gyration `r = √(I / A)` (mm).
///
/// Governs how far the cross-section extends from the neutral axis on
/// average — the natural "size" for buckling calculations.
#[must_use]
pub(crate) fn radius_of_gyration_mm(section: &CrossSection) -> Fix128 {
    let a = section.area_mm2();
    let i = section.second_moment_of_area_mm4();
    if a.is_zero() {
        return Fix128::ZERO;
    }
    (i / a).sqrt()
}

/// Slenderness ratio `λ = K·L / r`. Dimensionless.
///
/// - `λ < 50`: very stocky, buckling not a factor.
/// - `50 < λ < 100`: intermediate — use Johnson formula.
/// - `λ > 100`: slender — use Euler formula.
#[must_use]
pub(crate) fn slenderness_ratio(
    section: &CrossSection,
    length_mm: Fix128,
    end_condition: ColumnEndCondition,
) -> Fix128 {
    let r = radius_of_gyration_mm(section);
    if r.is_zero() {
        return Fix128::ZERO;
    }
    end_condition.k_factor() * length_mm / r
}

/// Transition slenderness `λ_t = π · √(2·E / σ_y)`.
///
/// Below this the Johnson parabolic formula applies; above, Euler.
/// The junction is such that both formulas give the same σ_cr = σ_y / 2.
#[must_use]
pub(crate) fn transition_slenderness(e_mpa: Fix128, sigma_y_mpa: Fix128) -> Fix128 {
    if sigma_y_mpa.is_zero() {
        return Fix128::ZERO;
    }
    let two_e = e_mpa.double();
    let ratio = two_e / sigma_y_mpa;
    Fix128::PI * ratio.sqrt()
}

// ============================================================================
// Critical stress (Euler + Johnson)
// ============================================================================

/// Regime selected by the slenderness ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucklingRegime {
    /// Short / stocky — Johnson parabolic formula applies.
    Johnson,
    /// Slender — Euler formula applies.
    Euler,
    /// Below the compressive yield — no buckling; pure yielding governs.
    Yielding,
}

/// Critical column-buckling stress (MPa) using the appropriate regime.
///
/// - Euler:  `σ_cr = π²·E / λ²`
/// - Johnson: `σ_cr = σ_y · (1 − (σ_y / (4·π²·E)) · λ²)`
///
/// Returns 0 if `slenderness` is zero (degenerate input).
#[must_use]
pub(crate) fn critical_stress_mpa(
    slenderness: Fix128,
    e_mpa: Fix128,
    sigma_y_mpa: Fix128,
) -> (Fix128, BucklingRegime) {
    if slenderness.is_zero() {
        return (Fix128::ZERO, BucklingRegime::Yielding);
    }
    let lambda_t = transition_slenderness(e_mpa, sigma_y_mpa);
    if slenderness >= lambda_t {
        // Euler regime
        let l2 = slenderness * slenderness;
        let sigma = Fix128::PI * Fix128::PI * e_mpa / l2;
        (sigma, BucklingRegime::Euler)
    } else if !sigma_y_mpa.is_zero() {
        // Johnson parabolic regime: σ_cr = σ_y · (1 − (σ_y·λ²) / (4·π²·E))
        //                          = σ_y − (σ_y² · λ²) / (4·π²·E)
        let l2 = slenderness * slenderness;
        let pi2_e_4 = Fix128::PI * Fix128::PI * e_mpa * Fix128::from_int(4);
        if pi2_e_4.is_zero() {
            return (sigma_y_mpa, BucklingRegime::Yielding);
        }
        let reduction = sigma_y_mpa * sigma_y_mpa * l2 / pi2_e_4;
        let sigma = if reduction < sigma_y_mpa {
            sigma_y_mpa - reduction
        } else {
            Fix128::ZERO
        };
        (sigma, BucklingRegime::Johnson)
    } else {
        (Fix128::ZERO, BucklingRegime::Yielding)
    }
}

/// Full column buckling analysis: geometry + material + length → critical
/// load and stress with the appropriate regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnBucklingReport {
    /// Radius of gyration (mm).
    pub radius_of_gyration_mm: Fix128,
    /// Slenderness ratio λ = KL/r.
    pub slenderness: Fix128,
    /// Transition slenderness λ_t.
    pub transition_slenderness: Fix128,
    /// Critical stress σ_cr (MPa).
    pub critical_stress_mpa: Fix128,
    /// Critical load `P_cr = σ_cr · A` (N).
    pub critical_load_n: Fix128,
    /// Regime used.
    pub regime: BucklingRegime,
}

/// Full analysis given cross-section, length, end condition and material.
#[must_use]
pub fn analyze_column(
    section: &CrossSection,
    length_mm: Fix128,
    end_condition: ColumnEndCondition,
    material: &MaterialProperties,
) -> ColumnBucklingReport {
    let e_mpa = material.youngs_modulus_gpa * Fix128::from_int(1000);
    let sigma_y_mpa = material.yield_strength_mpa;
    let r = radius_of_gyration_mm(section);
    let lambda = slenderness_ratio(section, length_mm, end_condition);
    let lambda_t = transition_slenderness(e_mpa, sigma_y_mpa);
    let (sigma_cr, regime) = critical_stress_mpa(lambda, e_mpa, sigma_y_mpa);
    let p_cr = sigma_cr * section.area_mm2();
    ColumnBucklingReport {
        radius_of_gyration_mm: r,
        slenderness: lambda,
        transition_slenderness: lambda_t,
        critical_stress_mpa: sigma_cr,
        critical_load_n: p_cr,
        regime,
    }
}

// ============================================================================
// Local plate buckling
// ============================================================================

/// Critical plate buckling stress for a thin flange in a hollow section
/// (Timoshenko eq. 8.8): `σ_cr = k · π²·E / (12·(1−ν²)) · (t / b)²`
///
/// - `b`: plate width (short dimension).
/// - `t`: plate thickness.
/// - `k`: geometric factor (simply supported all edges = 4, one edge free = 0.425).
///
/// Returns 0 when `width_mm` is zero or when `poisson` is ±1 (the
/// `1 − ν²` denominator vanishes); both are degenerate inputs, not plates.
#[must_use]
pub(crate) fn plate_buckling_mpa(
    e_mpa: Fix128,
    poisson: Fix128,
    thickness_mm: Fix128,
    width_mm: Fix128,
    k: Fix128,
) -> Fix128 {
    if width_mm.is_zero() {
        return Fix128::ZERO;
    }
    let nu2 = poisson * poisson;
    let one_minus_nu2 = Fix128::ONE - nu2;
    if one_minus_nu2.is_zero() {
        return Fix128::ZERO;
    }
    let ratio = thickness_mm / width_mm;
    let ratio2 = ratio * ratio;
    let numer = k * Fix128::PI * Fix128::PI * e_mpa * ratio2;
    numer / (Fix128::from_int(12) * one_minus_nu2)
}

// ============================================================================
// Snap-through buckling
// ============================================================================

/// Limit (snap-through) load of a shallow two-bar truss or clip feature of
/// rise `h`, span `L`, cross-section area `A` and Young's modulus `E`.
///
/// The two bars of length `L₀ = √(a² + h²)`, `a = L/2`, carry the apex at
/// height `y`; with the shallow strain `ε ≈ (y² − h²)/(2a²)` and the bar
/// force `N = E A ε` the vertical balance at the apex is
/// `P(y) = E A (h² − y²) y / a³`, stationary at `y = h/√3`, so
///
/// `P_max = (2 / (3√3)) · E A · (h/a)³ = (16 / (3√3)) · E A · (h/L)³ ≈ 3.079 · E A · (h/L)³`
///
/// (the classical von Mises truss, Bažant & Cedolin, two-bar truss). The cube
/// is structural: it comes from the geometric nonlinearity of the strain.
/// ⚠️ Before 2026-10-03 this function returned `3.72 · E A · (h/L)²`, a form
/// whose source could not be reconstructed (a square law with a constant
/// coefficient arises for arches with bending stiffness, not for an `E A`
/// truss); the unit test of that version pinned the wrong law as a golden
/// value. The shallow approximation is good to `O((h/a)²)`; the exact
/// `P(y) = 2 E A (L₀/L(y) − 1) · y/L(y)` is what the unit tests compare
/// against.
///
/// A flat truss (`rise_mm == 0`) has no snap-through and returns `Ok(0)`.
///
/// # Errors
///
/// [`SnapThroughError::NonPositiveSpan`] for `span_mm ≤ 0` (the ratio
/// `h/L` is undefined; before 2026-10-03 a zero span returned a silent
/// `0 N`, which a caller cannot tell from a flat truss),
/// [`SnapThroughError::NonPositiveStiffness`] for `e_mpa ≤ 0` or
/// `section_area_mm2 ≤ 0` (no bar to buckle),
/// [`SnapThroughError::NegativeRise`] for `rise_mm < 0` (the apex would be
/// below the supports, so the load is a pull-through, not this formula).
pub(crate) fn snap_through_load_n(
    e_mpa: Fix128,
    section_area_mm2: Fix128,
    rise_mm: Fix128,
    span_mm: Fix128,
) -> Result<Fix128, SnapThroughError> {
    if span_mm <= Fix128::ZERO {
        return Err(SnapThroughError::NonPositiveSpan);
    }
    if e_mpa <= Fix128::ZERO || section_area_mm2 <= Fix128::ZERO {
        return Err(SnapThroughError::NonPositiveStiffness);
    }
    if rise_mm < Fix128::ZERO {
        return Err(SnapThroughError::NegativeRise);
    }
    // (16 / (3√3)) · E · A · (h/L)³
    let ratio = rise_mm / span_mm;
    let ratio3 = ratio * ratio * ratio;
    let coefficient = Fix128::from_int(16) / (Fix128::from_int(3) * Fix128::from_int(3).sqrt());
    Ok(coefficient * e_mpa * section_area_mm2 * ratio3)
}

/// Why the snap-through load (`StructuralSolver::snap_through_load_n`)
/// could not be computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SnapThroughError {
    /// `span_mm ≤ 0`: the rise ratio `h/L` is undefined.
    NonPositiveSpan,
    /// `e_mpa ≤ 0` or `section_area_mm2 ≤ 0`: there is no bar.
    NonPositiveStiffness,
    /// `rise_mm < 0`: the apex is below the supports.
    NegativeRise,
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
    fn radius_of_gyration_rectangle() {
        // 10x20 rectangle: I = 6666.67, A = 200, r = √33.33 ≈ 5.774
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let r = radius_of_gyration_mm(&s);
        let expected = Fix128::from_ratio(5774, 1000);
        assert!(
            approx_eq(r, expected, Fix128::from_ratio(1, 100)),
            "got {}, expected ~5.774",
            r.to_f32()
        );
    }

    #[test]
    fn radius_of_gyration_circle_is_d_over_4() {
        // For solid circle: r = √(I/A) = √(π·d⁴/64 / (π·d²/4)) = d/4
        let s = CrossSection::Circular {
            diameter_mm: Fix128::from_int(8),
        };
        let r = radius_of_gyration_mm(&s);
        assert!(approx_eq(
            r,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn slenderness_pin_pin_baseline() {
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        // L = 500mm, r ≈ 5.77 → λ ≈ 86.6
        let lambda = slenderness_ratio(&s, Fix128::from_int(500), ColumnEndCondition::PinPin);
        assert!(
            approx_eq(lambda, Fix128::from_ratio(866, 10), Fix128::from_int(1)),
            "got {}, expected ~86.6",
            lambda.to_f32()
        );
    }

    #[test]
    fn slenderness_cantilever_double_pinpin() {
        let s = CrossSection::Circular {
            diameter_mm: Fix128::from_int(10),
        };
        let l = Fix128::from_int(300);
        let lambda_pin = slenderness_ratio(&s, l, ColumnEndCondition::PinPin);
        let lambda_cant = slenderness_ratio(&s, l, ColumnEndCondition::Cantilever);
        // K=2 → cantilever slenderness = 2× PinPin
        let ratio = lambda_cant / lambda_pin;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn transition_slenderness_pla() {
        // PLA: E=3500, σ_y=50 → λ_t = π√(7000/50) = π√140 ≈ 37.19
        let lambda_t = transition_slenderness(Fix128::from_int(3500), Fix128::from_int(50));
        let expected = Fix128::from_ratio(3719, 100);
        assert!(
            approx_eq(lambda_t, expected, Fix128::ONE),
            "got {}, expected ~37.2",
            lambda_t.to_f32()
        );
    }

    #[test]
    fn euler_regime_slender_column() {
        // Very slender: λ = 200, well above transition
        let e = Fix128::from_int(3500);
        let sy = Fix128::from_int(50);
        let (sigma, regime) = critical_stress_mpa(Fix128::from_int(200), e, sy);
        assert_eq!(regime, BucklingRegime::Euler);
        // σ_cr = π²·E/λ² = 9.87·3500/40000 ≈ 0.863 MPa
        assert!(
            approx_eq(
                sigma,
                Fix128::from_ratio(863, 1000),
                Fix128::from_ratio(1, 10)
            ),
            "got {}",
            sigma.to_f32()
        );
    }

    #[test]
    fn johnson_regime_stocky_column() {
        // λ = 20 (very stocky) with PLA transition ~37
        let e = Fix128::from_int(3500);
        let sy = Fix128::from_int(50);
        let (sigma, regime) = critical_stress_mpa(Fix128::from_int(20), e, sy);
        assert_eq!(regime, BucklingRegime::Johnson);
        // Johnson: σ_cr = 50 · (1 - 50·400/(4π²·3500)) = 50·(1 - 20000/138175)
        //        = 50·(1 - 0.1447) = 50·0.855 = 42.7 MPa
        assert!(
            approx_eq(sigma, Fix128::from_ratio(427, 10), Fix128::ONE),
            "got {}, expected ~42.7",
            sigma.to_f32()
        );
    }

    #[test]
    fn at_transition_regimes_agree() {
        let e = Fix128::from_int(3500);
        let sy = Fix128::from_int(50);
        let lambda_t = transition_slenderness(e, sy);
        // Just below λ_t (Johnson)
        let (s_below, r_below) = critical_stress_mpa(lambda_t - Fix128::from_ratio(1, 10), e, sy);
        // Just above λ_t (Euler)
        let (s_above, r_above) = critical_stress_mpa(lambda_t + Fix128::from_ratio(1, 10), e, sy);
        assert_eq!(r_below, BucklingRegime::Johnson);
        assert_eq!(r_above, BucklingRegime::Euler);
        // Both should give ≈ σ_y / 2 = 25 MPa
        let diff = if s_below > s_above {
            s_below - s_above
        } else {
            s_above - s_below
        };
        // Small mismatch due to derivative discontinuity at transition
        assert!(diff < Fix128::from_int(2), "diff = {}", diff.to_f32());
    }

    #[test]
    fn analyze_column_full_report_pla() {
        // 10x10 PLA column 500mm long, pin-pin
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(10),
        };
        let report = analyze_column(
            &s,
            Fix128::from_int(500),
            ColumnEndCondition::PinPin,
            &MaterialProperties::pla(),
        );
        // λ = 500 / r where r ≈ 2.89 → λ ≈ 173 → Euler regime
        assert!(report.slenderness > Fix128::from_int(150));
        assert_eq!(report.regime, BucklingRegime::Euler);
        assert!(report.critical_load_n > Fix128::ZERO);
    }

    #[test]
    fn analyze_column_short_pla_is_johnson() {
        // Very short PLA column: 20mm, 10x10 cross-section
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(10),
        };
        let report = analyze_column(
            &s,
            Fix128::from_int(20),
            ColumnEndCondition::PinPin,
            &MaterialProperties::pla(),
        );
        // λ ≈ 20/2.89 ≈ 6.9 → well below transition → Johnson
        assert_eq!(report.regime, BucklingRegime::Johnson);
    }

    #[test]
    fn plate_buckling_scales_with_thickness_squared() {
        let e = Fix128::from_int(3500);
        let nu = Fix128::from_ratio(35, 100);
        let s1 = plate_buckling_mpa(
            e,
            nu,
            Fix128::from_int(1),
            Fix128::from_int(20),
            Fix128::from_int(4),
        );
        let s2 = plate_buckling_mpa(
            e,
            nu,
            Fix128::from_int(2),
            Fix128::from_int(20),
            Fix128::from_int(4),
        );
        // Doubling t → 4× σ_cr
        let ratio = s2 / s1;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn plate_buckling_scales_with_width_inverse_squared() {
        let e = Fix128::from_int(3500);
        let nu = Fix128::from_ratio(35, 100);
        let s_narrow = plate_buckling_mpa(
            e,
            nu,
            Fix128::from_int(1),
            Fix128::from_int(10),
            Fix128::from_int(4),
        );
        let s_wide = plate_buckling_mpa(
            e,
            nu,
            Fix128::from_int(1),
            Fix128::from_int(20),
            Fix128::from_int(4),
        );
        // Doubling width → σ_cr / 4
        let ratio = s_narrow / s_wide;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    /// Oracle: Timoshenko & Gere eq. 8.8 evaluated by hand in f64,
    /// `σ_cr = k·π²·E / (12·(1 − ν²)) · (t/b)²` with `k = 4`, `E = 3500 MPa`,
    /// `ν = 0.35`, `t = 1 mm`, `b = 20 mm`:
    /// `4·π²·3500 / (12·0.8775) · 0.0025 = 32.8050 MPa`.
    #[test]
    fn plate_buckling_matches_timoshenko_closed_form() {
        let got = plate_buckling_mpa(
            Fix128::from_int(3500),
            Fix128::from_ratio(35, 100),
            Fix128::from_int(1),
            Fix128::from_int(20),
            Fix128::from_int(4),
        )
        .to_f64();
        let pi = core::f64::consts::PI;
        let want =
            4.0 * pi * pi * 3500.0 / (12.0 * (1.0 - 0.35 * 0.35)) * (1.0 / 20.0) * (1.0 / 20.0);
        assert!(((got - want) / want).abs() < 1e-9, "got {got}, want {want}");
    }

    /// Degenerate inputs documented on `plate_buckling_mpa`: zero width and
    /// `ν = ±1` return 0 (early return, not a division by zero); zero
    /// thickness and zero modulus give 0 through the formula itself.
    #[test]
    fn plate_buckling_degenerate_inputs_return_zero() {
        let e = Fix128::from_int(3500);
        let nu = Fix128::from_ratio(35, 100);
        let one = Fix128::ONE;
        let k = Fix128::from_int(4);
        assert_eq!(
            plate_buckling_mpa(e, nu, one, Fix128::ZERO, k),
            Fix128::ZERO
        );
        assert_eq!(
            plate_buckling_mpa(e, one, one, Fix128::from_int(20), k),
            Fix128::ZERO
        );
        assert_eq!(
            plate_buckling_mpa(e, Fix128::NEG_ONE, one, Fix128::from_int(20), k),
            Fix128::ZERO
        );
        assert_eq!(
            plate_buckling_mpa(e, nu, Fix128::ZERO, Fix128::from_int(20), k),
            Fix128::ZERO
        );
        assert_eq!(
            plate_buckling_mpa(Fix128::ZERO, nu, one, Fix128::from_int(20), k),
            Fix128::ZERO
        );
    }

    /// Degenerate inputs for `snap_through_load_n`: a flat truss (zero rise)
    /// is `Ok(0)` (no snap-through exists), zero or negative area / modulus
    /// are `NonPositiveStiffness`, a negative rise is `NegativeRise`, and a
    /// zero or negative span is `NonPositiveSpan` rather than the silent
    /// `0 N` of the pre-2026-10-03 version (which `Fix128`'s zero division
    /// would also have produced without any guard).
    #[test]
    fn snap_through_degenerate_inputs() {
        let e = Fix128::from_int(3500);
        let a = Fix128::from_int(10);
        let h = Fix128::from_int(2);
        let l = Fix128::from_int(20);
        let neg = Fix128::from_int(-1);
        assert_eq!(snap_through_load_n(e, a, Fix128::ZERO, l), Ok(Fix128::ZERO));
        assert_eq!(
            snap_through_load_n(e, Fix128::ZERO, h, l),
            Err(SnapThroughError::NonPositiveStiffness)
        );
        assert_eq!(
            snap_through_load_n(Fix128::ZERO, a, h, l),
            Err(SnapThroughError::NonPositiveStiffness)
        );
        assert_eq!(
            snap_through_load_n(e, neg, h, l),
            Err(SnapThroughError::NonPositiveStiffness)
        );
        assert_eq!(
            snap_through_load_n(e, a, neg, l),
            Err(SnapThroughError::NegativeRise)
        );
        assert_eq!(
            snap_through_load_n(e, a, h, Fix128::ZERO),
            Err(SnapThroughError::NonPositiveSpan)
        );
        assert_eq!(
            snap_through_load_n(e, a, h, neg),
            Err(SnapThroughError::NonPositiveSpan)
        );
    }

    /// Oracle: `P_max = (16/(3√3)) · E A · (h/L)³` by hand.
    /// E = 3500 MPa, A = 10 mm², h = 2, L = 20 (h/L = 0.1):
    /// `3.0792014356780038 · 3500 · 10 · 0.001 = 107.772050 N`.
    /// The earlier golden (1302 N) pinned the square law this function used
    /// to return; it is kept here as the value that must NOT come back.
    #[test]
    fn snap_through_shallow_arch() {
        let p = snap_through_load_n(
            Fix128::from_int(3500),
            Fix128::from_int(10),
            Fix128::from_int(2),
            Fix128::from_int(20),
        )
        .unwrap();
        let want = 3.079_201_435_678_003_8_f64 * 3500.0 * 10.0 * 0.001;
        assert!(
            (p.to_f64() - want).abs() < 1e-9 * want,
            "got {} N, closed form {want} N",
            p.to_f64()
        );
        assert!(
            (p.to_f64() - 1302.0).abs() > 1000.0,
            "the square law came back"
        );
    }

    /// Exact two-bar (von Mises) truss load at apex height `y`, in the test's
    /// own arithmetic: `P(y) = 2 E A (L₀/L(y) − 1) · y / L(y)` with
    /// `L(y) = √(a² + y²)`, `L₀ = L(h)`, `a = L/2` (magnitude of the
    /// compressive load).
    fn exact_truss_load(e: f64, area: f64, h: f64, span: f64, y: f64) -> f64 {
        let a = span / 2.0;
        let l0 = crate::det_math::sqrt64(a * a + h * h);
        let ly = crate::det_math::sqrt64(a * a + y * y);
        2.0 * e * area * (l0 / ly - 1.0) * (y / ly)
    }

    /// Oracle: the shallow formula is the `h/L → 0` limit of the exact truss.
    /// The exact limit load is found by scanning `y ∈ (0, h)`; the shallow
    /// approximation differs by `O((h/a)²)`, so the ratio must sit within
    /// `1 ± 8 (h/L)²` and approach 1 as `h/L` shrinks (three rises). The
    /// stationary point of the shallow law, `y = h/√3`, is where the exact
    /// scan peaks up to the same order of shift.
    #[test]
    fn snap_through_is_the_shallow_limit_of_the_exact_truss() {
        let (e, area, span) = (3500.0_f64, 10.0_f64, 20.0_f64);
        let mut previous_gap = f64::INFINITY;
        for ratio in [0.1_f64, 0.05, 0.02] {
            let h = ratio * span;
            let steps = 2000;
            let mut best = (0.0_f64, 0.0_f64);
            for i in 1..steps {
                let y = h * f64::from(i) / f64::from(steps);
                let p = exact_truss_load(e, area, h, span, y);
                if p > best.0 {
                    best = (p, y);
                }
            }
            let shallow = snap_through_load_n(
                Fix128::from_f64(e),
                Fix128::from_f64(area),
                Fix128::from_f64(h),
                Fix128::from_f64(span),
            )
            .unwrap()
            .to_f64();
            let gap = (shallow / best.0 - 1.0).abs();
            assert!(gap <= 8.0 * ratio * ratio, "h/L = {ratio}: ratio gap {gap}");
            assert!(
                gap < previous_gap,
                "h/L = {ratio}: the gap must shrink with the rise"
            );
            previous_gap = gap;
            // The exact peak sits at `h/√3` up to the same `O((h/a)²)`
            // relative shift (h/a = 2 h/L), plus one scan step.
            let y_star = h / crate::det_math::sqrt64(3.0);
            let shift = 2.0 * (2.0 * ratio) * (2.0 * ratio) * h + 2.0 * h / f64::from(steps);
            assert!(
                (best.1 - y_star).abs() <= shift,
                "h/L = {ratio}: peak at {} vs h/√3 = {y_star}",
                best.1
            );
        }
    }
}
