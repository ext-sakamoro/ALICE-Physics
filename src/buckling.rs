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
pub fn radius_of_gyration_mm(section: &CrossSection) -> Fix128 {
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
pub fn slenderness_ratio(
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
pub fn transition_slenderness(e_mpa: Fix128, sigma_y_mpa: Fix128) -> Fix128 {
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
pub fn critical_stress_mpa(
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
#[must_use]
pub fn plate_buckling_mpa(
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

/// Critical downward load for snap-through of a shallow (low-rise) arch or
/// clip feature (Bažant & Cedolin §5).
///
/// Approximate formula for a two-hinged arch of rise `h`, half-span `L/2`,
/// cross-section area `A`, Young's modulus `E`:
///
/// `P_snap ≈ 3.72 · E · A · (h / L)²`
///
/// Coefficient is empirically derived assuming h/L ≤ 0.2. For deeper arches
/// the Roark full solution should be used.
#[must_use]
pub fn snap_through_load_n(
    e_mpa: Fix128,
    section_area_mm2: Fix128,
    rise_mm: Fix128,
    span_mm: Fix128,
) -> Fix128 {
    if span_mm.is_zero() {
        return Fix128::ZERO;
    }
    // 3.72 · E · A · (h/L)²
    let ratio = rise_mm / span_mm;
    let ratio2 = ratio * ratio;
    // 3.72 ≈ 372 / 100
    Fix128::from_ratio(372, 100) * e_mpa * section_area_mm2 * ratio2
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

    #[test]
    fn snap_through_zero_span_is_zero() {
        let p = snap_through_load_n(
            Fix128::from_int(3500),
            Fix128::from_int(10),
            Fix128::from_int(2),
            Fix128::ZERO,
        );
        assert_eq!(p, Fix128::ZERO);
    }

    #[test]
    fn snap_through_shallow_arch() {
        // E=3500, A=10 mm², rise=2, span=20 → h/L=0.1
        // P = 3.72 · 3500 · 10 · 0.01 = 1302 N
        let p = snap_through_load_n(
            Fix128::from_int(3500),
            Fix128::from_int(10),
            Fix128::from_int(2),
            Fix128::from_int(20),
        );
        assert!(
            approx_eq(p, Fix128::from_int(1302), Fix128::from_int(5)),
            "got {} N",
            p.to_f32()
        );
    }
}
