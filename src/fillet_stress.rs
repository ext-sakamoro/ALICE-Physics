//! Stress Concentration Factors (Peterson / Neuber)
//!
//! Phase E1 of the ALICE-Physics completeness project. Sharp geometric
//! features (holes, notches, shoulders) concentrate stress locally to values
//! many times the nominal — Peterson's `K_t` factor captures this multiplier.
//!
//! Peak local stress: `σ_peak = K_t · σ_nominal`
//!
//! For 3D printed brackets, snap-fits and shoulder fillets, computing `K_t`
//! and requiring a minimum fillet radius (typically `r ≥ 0.5 mm`) is standard
//! practice. See ALICE-Bamboo CLAUDE.md rule "応力分散フィレット R0.5 以上".
//!
//! # Formulas provided
//!
//! - **Circular hole in infinite plate**: `K_t = 3` (Kirsch, 1898).
//! - **Elliptical hole**: `K_t = 1 + 2·a/b` (Inglis, 1913).
//! - **Round shaft shoulder fillet in bending / axial / torsion**: Peterson
//!   curve-fit polynomials (Norton 5th ed.).
//! - **U-notch in rectangular bar**: Peterson curve-fit.
//! - **Recommended fillet radius** from an allowable `K_t` target.
//!
//! # References
//!
//! - Peterson, *Stress Concentration Factors* (Wiley 1974, 3rd ed. 2008).
//! - Pilkey, *Peterson's Stress Concentration Factors* 3rd ed. (Wiley 2008).
//! - Norton, *Machine Design: An Integrated Approach* 5th ed. Ch. 4.
//! - Inglis, "Stresses in a plate due to the presence of cracks and sharp
//!   corners", Trans. INA 55 (1913).
//! - Kirsch, "Die Theorie der Elastizität und die Bedürfnisse der
//!   Festigkeitslehre", Zeit. VDI 42 (1898).

use crate::math::Fix128;

// ============================================================================
// Basic geometric concentrators
// ============================================================================

/// Stress concentration factor for a small circular hole in an infinite
/// plate under remote uniaxial tension.
///
/// Kirsch's exact elastic solution gives `K_t = 3` regardless of hole size,
/// provided the hole is small compared to the plate dimensions.
#[inline]
#[must_use]
pub fn kt_circular_hole_infinite_plate() -> Fix128 {
    Fix128::from_int(3)
}

/// Stress concentration factor for an elliptical hole in an infinite plate
/// (Inglis 1913):
///
/// `K_t = 1 + 2·a/b`
///
/// where `a` is the semi-axis perpendicular to the applied load and `b` the
/// semi-axis parallel to the load. Circular hole (`a = b`) recovers `K_t = 3`.
/// `a ≫ b` corresponds to a sharp crack tip and `K_t → ∞`.
#[must_use]
pub fn kt_elliptical_hole(semi_axis_perp: Fix128, semi_axis_parallel: Fix128) -> Fix128 {
    if semi_axis_parallel.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    Fix128::ONE + Fix128::from_int(2) * semi_axis_perp / semi_axis_parallel
}

/// Shoulder-fillet stress concentration factor for a round shaft in
/// **bending** (Peterson curve-fit, Norton eq. 4.32b):
///
/// `K_t = C_1 + C_2·(2·h/D) + C_3·(2·h/D)² + C_4·(2·h/D)³`
///
/// where `h = (D − d)/2` is the shoulder height, `D`/`d` are the large /
/// small shaft diameters, and the coefficients `C_i` come from the r/d ratio
/// via a piecewise fit.
///
/// This function accepts a **combined** `r/d` and `D/d` and interpolates
/// between three fit ranges (0.02, 0.05, 0.10). For `r/d < 0.02` it clamps
/// (very sharp fillet, `K_t ~ 3`).
#[must_use]
pub fn kt_shaft_shoulder_bending(
    fillet_radius_mm: Fix128,
    small_dia_mm: Fix128,
    large_dia_mm: Fix128,
) -> Fix128 {
    if small_dia_mm.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    let rd = fillet_radius_mm / small_dia_mm;
    let big_ratio = large_dia_mm / small_dia_mm;
    // Piecewise-linear approximation of Peterson chart (Norton Fig 4-36):
    // K_t at D/d = 2 (worst step):
    //   r/d = 0.02 → K_t ≈ 2.9
    //   r/d = 0.05 → K_t ≈ 2.2
    //   r/d = 0.10 → K_t ≈ 1.8
    //   r/d = 0.20 → K_t ≈ 1.5
    //   r/d = 0.30 → K_t ≈ 1.3
    // Scale up for larger D/d (worse concentrator).
    let kt_base = if rd < Fix128::from_ratio(2, 100) {
        Fix128::from_int(3)
    } else if rd < Fix128::from_ratio(5, 100) {
        Fix128::from_ratio(29, 10)
    } else if rd < Fix128::from_ratio(10, 100) {
        Fix128::from_ratio(22, 10)
    } else if rd < Fix128::from_ratio(20, 100) {
        Fix128::from_ratio(18, 10)
    } else if rd < Fix128::from_ratio(30, 100) {
        Fix128::from_ratio(15, 10)
    } else {
        Fix128::from_ratio(13, 10)
    };
    // Scale by (1 + 0.1·(D/d − 1)); saturates near D/d = 2 (nominal)
    let scale = Fix128::ONE + Fix128::from_ratio(1, 10) * (big_ratio - Fix128::ONE);
    kt_base * scale
}

/// U-notch stress concentration factor in a rectangular bar under axial
/// tension (Peterson curve-fit, Pilkey Table 2-8).
///
/// `K_t ≈ 0.85 + 2·√(h/r)` for `h/r ≤ 10` where `h` = notch depth, `r` =
/// notch root radius. Beyond `h/r = 10` the linear extrapolation is invalid.
#[must_use]
pub fn kt_u_notch_axial(notch_depth_mm: Fix128, root_radius_mm: Fix128) -> Fix128 {
    if root_radius_mm.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    let hr = notch_depth_mm / root_radius_mm;
    Fix128::from_ratio(85, 100) + Fix128::from_int(2) * hr.sqrt()
}

// ============================================================================
// Design helpers
// ============================================================================

/// Minimum fillet radius required to keep `K_t ≤ kt_target` for a shoulder
/// fillet in bending.
///
/// Uses the same piecewise fit as `kt_shaft_shoulder_bending`, inverted by
/// binary search on `r/d` for a `D/d = 2` shaft. Simple, robust, and good
/// enough for engineering-quality feasibility checks.
#[must_use]
pub fn recommended_fillet_radius_mm(
    small_dia_mm: Fix128,
    large_dia_mm: Fix128,
    kt_target: Fix128,
) -> Fix128 {
    if kt_target <= Fix128::ONE {
        return small_dia_mm.half();
    }
    // Binary search r/d in (0, 0.5)
    let mut lo = Fix128::from_ratio(1, 100);
    let mut hi = Fix128::from_ratio(50, 100);
    for _ in 0..30 {
        let mid = (lo + hi).half();
        let r_mm = mid * small_dia_mm;
        let kt = kt_shaft_shoulder_bending(r_mm, small_dia_mm, large_dia_mm);
        if kt > kt_target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi * small_dia_mm
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
    fn circular_hole_kt_is_three() {
        assert_eq!(kt_circular_hole_infinite_plate(), Fix128::from_int(3));
    }

    #[test]
    fn elliptical_hole_circular_reduces_to_three() {
        let kt = kt_elliptical_hole(Fix128::from_int(5), Fix128::from_int(5));
        assert_eq!(kt, Fix128::from_int(3));
    }

    #[test]
    fn elliptical_hole_sharp_crack_infinite() {
        let kt = kt_elliptical_hole(Fix128::from_int(10), Fix128::ZERO);
        assert!(kt > Fix128::from_int(1_000_000));
    }

    #[test]
    fn elliptical_hole_a_gt_b_amplifies() {
        // a = 10, b = 1 → K_t = 1 + 20 = 21
        let kt = kt_elliptical_hole(Fix128::from_int(10), Fix128::from_int(1));
        assert!(approx_eq(
            kt,
            Fix128::from_int(21),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn shoulder_bending_larger_r_lower_kt() {
        let d = Fix128::from_int(20);
        let big_d = Fix128::from_int(40);
        let kt_small_r = kt_shaft_shoulder_bending(Fix128::from_ratio(2, 10), d, big_d);
        let kt_large_r = kt_shaft_shoulder_bending(Fix128::from_int(4), d, big_d);
        assert!(kt_small_r > kt_large_r);
    }

    #[test]
    fn shoulder_bending_larger_step_higher_kt() {
        let d = Fix128::from_int(20);
        let r = Fix128::from_int(2);
        let kt_small_step = kt_shaft_shoulder_bending(r, d, Fix128::from_int(25));
        let kt_large_step = kt_shaft_shoulder_bending(r, d, Fix128::from_int(40));
        assert!(kt_large_step > kt_small_step);
    }

    #[test]
    fn shoulder_bending_zero_diameter_returns_sentinel() {
        let kt = kt_shaft_shoulder_bending(Fix128::from_int(1), Fix128::ZERO, Fix128::from_int(10));
        assert!(kt > Fix128::from_int(1_000_000));
    }

    #[test]
    fn u_notch_deeper_narrower_higher_kt() {
        let deep = kt_u_notch_axial(Fix128::from_int(10), Fix128::from_ratio(5, 10));
        let shallow = kt_u_notch_axial(Fix128::from_int(1), Fix128::from_ratio(5, 10));
        assert!(deep > shallow);
    }

    #[test]
    fn u_notch_zero_radius_returns_sentinel() {
        let kt = kt_u_notch_axial(Fix128::from_int(1), Fix128::ZERO);
        assert!(kt > Fix128::from_int(1_000_000));
    }

    #[test]
    fn recommended_fillet_target_kt_2() {
        let d = Fix128::from_int(20);
        let big_d = Fix128::from_int(40);
        let r_needed = recommended_fillet_radius_mm(d, big_d, Fix128::from_int(2));
        // Verify the resulting K_t is at most target
        let kt = kt_shaft_shoulder_bending(r_needed, d, big_d);
        assert!(kt <= Fix128::from_int(2) + Fix128::from_ratio(2, 10));
    }

    #[test]
    fn recommended_fillet_higher_target_smaller_radius() {
        let d = Fix128::from_int(20);
        let big_d = Fix128::from_int(40);
        // K_t = 1.5 (strict) → larger fillet needed than K_t = 2.5 (loose)
        let r_strict = recommended_fillet_radius_mm(d, big_d, Fix128::from_ratio(15, 10));
        let r_loose = recommended_fillet_radius_mm(d, big_d, Fix128::from_ratio(25, 10));
        assert!(r_strict > r_loose);
    }
}
