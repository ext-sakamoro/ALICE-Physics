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
//! practice.
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
/// **bending**, from the Peterson curve fit for stepped round bars in
/// bending (Pilkey, *Formulas for Stress, Strain, and Structural Matrices*
/// 2nd ed.; Peterson's chart for the shoulder fillet). With the step height
/// `h = (D − d)/2`, `x = h/r` and `t = 2h/D`:
///
/// `K_t = C1 + C2·t + C3·t² + C4·t³`, each `Ci = ai + bi·√x + ci·x`, with
///
/// | `0.1 ≤ x ≤ 2` | a | b | c |
/// |---|---|---|---|
/// | C1 | 0.947 | 1.206 | −0.131 |
/// | C2 | 0.022 | −3.405 | 0.915 |
/// | C3 | 0.869 | 1.777 | −0.555 |
/// | C4 | −0.810 | 0.422 | −0.260 |
///
/// | `2 ≤ x ≤ 20` | a | b | c |
/// |---|---|---|---|
/// | C1 | 1.232 | 0.832 | −0.008 |
/// | C2 | −3.813 | 0.968 | −0.260 |
/// | C3 | 7.423 | −4.868 | 0.869 |
/// | C4 | −3.839 | 3.070 | −0.600 |
///
/// At `D/d = 2` this gives `K_t = 2.27 / 1.80 / 1.48 / 1.34` at
/// `r/d = 0.05 / 0.10 / 0.20 / 0.30`. Two adjustments keep the result
/// continuous and non-increasing in `r`, and only ever raise the fit: the two
/// coefficient sets differ by up to 0.03 at `x = 2`, and that step is ramped
/// in over `√x ∈ [√1.5, √2]`; and the fit dips with `x` for large steps
/// (`D/d ≥ 3`), so `K_t(x)` is the running maximum of the fit over `[0.1, x]`.
/// Below `x = 0.1` (a large radius for the step) `K_t` falls linearly to 1 at
/// `x = 0`, so `D = d` (no shoulder) gives exactly 1 and `D < d` is treated
/// as no shoulder. A sharp corner (`r ≤ 0`) gives the saturating sentinel.
#[must_use]
pub fn kt_shaft_shoulder_bending(
    fillet_radius_mm: Fix128,
    small_dia_mm: Fix128,
    large_dia_mm: Fix128,
) -> Fix128 {
    if small_dia_mm.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    if large_dia_mm <= small_dia_mm {
        return Fix128::ONE;
    }
    if fillet_radius_mm <= Fix128::ZERO {
        return Fix128::from_int(i64::MAX >> 8);
    }
    let h = (large_dia_mm - small_dia_mm).half();
    let x = h / fillet_radius_mm;
    let t = (large_dia_mm - small_dia_mm) / large_dia_mm;
    let x_lo = Fix128::from_ratio(1, 10);
    let x_hi = Fix128::from_int(20);
    let kt = if x < x_lo {
        Fix128::ONE + (shoulder_envelope(x_lo.sqrt(), t) - Fix128::ONE) * x / x_lo
    } else if x <= x_hi {
        shoulder_envelope(x.sqrt(), t)
    } else {
        // LIMITATION(COV-STRUCT-092): Beyond `h/r = 20` (a fillet sharper than the fit covers) the value is extrapolated
        // as `1 + (K_t(20) − 1)·√((h/r)/20)`, the square-root growth of a deep notch (Neuber), not a chart value.
        Fix128::ONE + (shoulder_envelope(x_hi.sqrt(), t) - Fix128::ONE) * (x / x_hi).sqrt()
    };
    kt.max(Fix128::ONE)
}

/// One piece of the shoulder fit as a quadratic in `s = √(h/r)`:
/// `p + q·s + r·s²`, valid for `s` in `[lo, hi]`.
#[derive(Clone, Copy)]
struct ShoulderPiece {
    lo: Fix128,
    hi: Fix128,
    p: Fix128,
    q: Fix128,
    r: Fix128,
}

impl ShoulderPiece {
    fn at(&self, s: Fix128) -> Fix128 {
        self.p + (self.q + self.r * s) * s
    }

    /// Largest value on `[lo, min(hi, s)]` (the endpoints, or the vertex of a
    /// downward parabola inside the interval).
    fn max_up_to(&self, s: Fix128) -> Fix128 {
        let b = if s < self.hi { s } else { self.hi };
        let mut m = self.at(self.lo).max(self.at(b));
        if self.r < Fix128::ZERO {
            let v = -self.q / (self.r + self.r);
            if v > self.lo && v < b {
                m = m.max(self.at(v));
            }
        }
        m
    }
}

/// `C1 + C2·t + C3·t² + C4·t³` with `Ci = ai + bi·√x + ci·x`, collected as a
/// quadratic in `√x`: the coefficient rows are `[a, b, c]` in thousandths.
fn shoulder_set(rows: [[i64; 3]; 4], t: Fix128) -> (Fix128, Fix128, Fix128) {
    let k = |v: i64| Fix128::from_ratio(v, 1000);
    let mut tp = Fix128::ONE;
    let (mut p, mut q, mut r) = (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    for row in rows {
        p = p + k(row[0]) * tp;
        q = q + k(row[1]) * tp;
        r = r + k(row[2]) * tp;
        tp = tp * t;
    }
    (p, q, r)
}

/// Running maximum over `[√0.1, s]` of the fit with the step between the two
/// coefficient sets ramped in over `√x ∈ [√1.5, √2]` (`s ≤ √20`).
fn shoulder_envelope(s: Fix128, t: Fix128) -> Fix128 {
    const LOW: [[i64; 3]; 4] = [
        [947, 1206, -131],
        [22, -3405, 915],
        [869, 1777, -555],
        [-810, 422, -260],
    ];
    const HIGH: [[i64; 3]; 4] = [
        [1232, 832, -8],
        [-3813, 968, -260],
        [7423, -4868, 869],
        [-3839, 3070, -600],
    ];
    let (p1, q1, r1) = shoulder_set(LOW, t);
    let (p2, q2, r2) = shoulder_set(HIGH, t);
    let s0 = Fix128::from_ratio(1, 10).sqrt();
    let sa = Fix128::from_ratio(3, 2).sqrt();
    let sb = Fix128::from_int(2).sqrt();
    let s_end = Fix128::from_int(20).sqrt();
    let low = ShoulderPiece {
        lo: s0,
        hi: sa,
        p: p1,
        q: q1,
        r: r1,
    };
    let high = ShoulderPiece {
        lo: sb,
        hi: s_end,
        p: p2,
        q: q2,
        r: r2,
    };
    // the step at x = 2, added linearly in s over [sa, sb]: still a quadratic
    let jump = high.at(sb) - low.at(sb);
    let slope = jump / (sb - sa);
    let ramp = ShoulderPiece {
        lo: sa,
        hi: sb,
        p: p1 - slope * sa,
        q: q1 + slope,
        r: r1,
    };
    let mut m = Fix128::ONE;
    for piece in [low, ramp, high] {
        if s >= piece.lo {
            m = m.max(piece.max_up_to(s));
        }
    }
    m
}

/// U-notch stress concentration factor in a rectangular bar under axial
/// tension (Peterson curve-fit, Pilkey Table 2-8).
///
/// `K_t ≈ 0.85 + 2·√(h/r)` for `h/r ≤ 10` where `h` = notch depth, `r` =
// LIMITATION(COV-STRUCT-094): Beyond `h/r = 10` the linear extrapolation is invalid.
/// notch root radius. Beyond `h/r = 10` the linear extrapolation is invalid.
/// The fit falls below 1 for a very shallow notch (`h/r < 0.0056`), where no
/// concentration remains; the result is floored at `K_t = 1`.
#[must_use]
pub fn kt_u_notch_axial(notch_depth_mm: Fix128, root_radius_mm: Fix128) -> Fix128 {
    if root_radius_mm.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    let hr = notch_depth_mm / root_radius_mm;
    let kt = Fix128::from_ratio(85, 100) + Fix128::from_int(2) * hr.sqrt();
    if kt < Fix128::ONE {
        Fix128::ONE
    } else {
        kt
    }
}

// ============================================================================
// Design helpers
// ============================================================================

/// Minimum fillet radius required to keep `K_t ≤ kt_target` for a shoulder
/// fillet in bending.
///
/// Inverts `kt_shaft_shoulder_bending` (decreasing in `r`) by binary search
/// on `r/d` in `[0, 0.5]`. A target that even a sharp corner (`r = 0`)
/// satisfies gives 0.
#[must_use]
pub fn recommended_fillet_radius_mm(
    small_dia_mm: Fix128,
    large_dia_mm: Fix128,
    kt_target: Fix128,
) -> Fix128 {
    if kt_target <= Fix128::ONE {
        return small_dia_mm.half();
    }
    if kt_shaft_shoulder_bending(Fix128::ZERO, small_dia_mm, large_dia_mm) <= kt_target {
        return Fix128::ZERO;
    }
    // Binary search r/d in [0, 0.5]
    let mut lo = Fix128::ZERO;
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

    #[test]
    fn shoulder_piece_max_includes_an_interior_vertex() {
        // 2 + 4s - s^2 peaks at s = 2 (value 6); on [0, 3] the endpoints give 2 and 5
        let piece = ShoulderPiece {
            lo: Fix128::ZERO,
            hi: Fix128::from_int(3),
            p: Fix128::from_int(2),
            q: Fix128::from_int(4),
            r: -Fix128::ONE,
        };
        assert_eq!(piece.max_up_to(Fix128::from_int(3)), Fix128::from_int(6));
        // before the vertex only the endpoints count
        assert_eq!(piece.max_up_to(Fix128::ONE), Fix128::from_int(5));
        // with the fit's coefficients no vertex falls inside a piece for any
        // step (t in [0, 1)), so the branch only guards the general case
    }

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
