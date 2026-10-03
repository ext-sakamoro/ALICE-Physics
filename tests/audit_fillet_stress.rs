//! Audit S1-5 oracles for `alice_physics::fillet_stress`.
//! Expected values are derived from the documented formulas / the module's own
//! stated reference table (Peterson chart, D/d = 2), not from the implementation.
#![allow(clippy::disallowed_methods)]

use alice_physics::fillet_stress::*;
use alice_physics::math::Fix128;

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Doc comment table: "K_t at D/d = 2 (worst step)": r/d=0.02 -> 2.9, 0.05 -> 2.2,
/// 0.10 -> 1.8, 0.20 -> 1.5, 0.30 -> 1.3. At D/d = 2 the function must return these.
#[test]
#[ignore = "known defect: AUD-A-S1W5-001: kt_shaft_shoulder_bending at D/d=2 returns 1.1x the module's own D/d=2 table (r/d=0.10: 1.98 vs 1.8)"]
fn shoulder_bending_at_d_over_d_two_matches_documented_table() {
    let d = Fix128::from_int(20);
    let big = Fix128::from_int(40);
    // (r/d numerator over 1000, table value)
    for (rd_milli, want) in [(20, 2.9), (50, 2.2), (100, 1.8), (200, 1.5), (300, 1.3)] {
        let r = fx(rd_milli, 1000) * d;
        let kt = kt_shaft_shoulder_bending(r, d, big).to_f64();
        assert!(
            (kt - want).abs() < 0.02,
            "r/d={} K_t={} want {}",
            rd_milli as f64 / 1000.0,
            kt,
            want
        );
    }
}

/// No shoulder (D = d) means no stress concentration: K_t = 1.
#[test]
#[ignore = "known defect: AUD-A-S1W5-002: kt_shaft_shoulder_bending(D=d) returns the fillet table value (1.3..3.0), not 1"]
fn shoulder_bending_without_step_is_unity() {
    let d = Fix128::from_int(20);
    for r in [fx(1, 1), fx(2, 1), fx(5, 1)] {
        let kt = kt_shaft_shoulder_bending(r, d, d).to_f64();
        assert!(
            (kt - 1.0).abs() < 0.05,
            "K_t(D=d, r={}) = {}",
            r.to_f64(),
            kt
        );
    }
}

/// Doc: "interpolates between fit ranges". An interpolation is continuous in r/d.
#[test]
#[ignore = "known defect: AUD-A-S1W5-003: kt_shaft_shoulder_bending is a piecewise-CONSTANT step table (jump 2.9->2.2 at r/d=0.05, -24%), doc says Peterson polynomial + interpolation"]
fn shoulder_bending_is_continuous_in_fillet_radius() {
    let d = Fix128::from_int(1000);
    let big = Fix128::from_int(2000);
    for b_milli in [20i64, 50, 100, 200, 300] {
        let below = fx(b_milli * 1000 - 1, 1_000_000) * d;
        let above = fx(b_milli * 1000 + 1, 1_000_000) * d;
        let k0 = kt_shaft_shoulder_bending(below, d, big).to_f64();
        let k1 = kt_shaft_shoulder_bending(above, d, big).to_f64();
        assert!(
            (k0 - k1).abs() / k0 < 0.02,
            "jump at r/d={}: {} -> {}",
            b_milli as f64 / 1000.0,
            k0,
            k1
        );
    }
}

/// Table values are exact rationals times scale (1 + (D/d-1)/10); pin the present
/// piecewise values at D/d = 3 (scale 1.2) so a changed constant is caught.
#[test]
fn shoulder_bending_current_table_at_d_over_d_three() {
    let d = Fix128::from_int(10);
    let big = Fix128::from_int(30);
    // r/d -> base: <.02:3.0 [.02,.05):2.9 [.05,.10):2.2 [.10,.20):1.8 [.20,.30):1.5 >=.30:1.3
    let cases = [
        (1, 3.0),
        (3, 2.9),
        (7, 2.2),
        (15, 1.8),
        (25, 1.5),
        (40, 1.3),
    ];
    for (rd_hundredth, base) in cases {
        let r = fx(rd_hundredth, 100) * d;
        let kt = kt_shaft_shoulder_bending(r, d, big).to_f64();
        assert!(
            (kt - base * 1.2).abs() < 1e-9,
            "r/d={} kt={} want {}",
            rd_hundredth,
            kt,
            base * 1.2
        );
    }
    // exact breakpoint belongs to the upper bin (strict <)
    let at = kt_shaft_shoulder_bending(fx(5, 100) * d, d, big).to_f64();
    assert!((at - 2.2 * 1.2).abs() < 1e-9);
}

/// K_t >= 1 always: stress concentration can't relieve stress.
#[test]
#[ignore = "known defect: AUD-A-S1W5-004: kt_u_notch_axial returns 0.913 at h/r=1e-3 and 0.85 at h=0 (<1); formula has no K_t >= 1 floor"]
fn u_notch_kt_never_below_one() {
    let kt = kt_u_notch_axial(fx(1, 1000), Fix128::ONE).to_f64();
    assert!(kt >= 1.0, "K_t = {}", kt);
    let kt0 = kt_u_notch_axial(Fix128::ZERO, Fix128::ONE).to_f64();
    assert!(kt0 >= 1.0, "K_t(h=0) = {}", kt0);
}

#[test]
fn u_notch_formula_exact_values() {
    // 0.85 + 2*sqrt(h/r); h/r = 1, 4, 9 -> 2.85, 4.85, 6.85
    for (h, want) in [(1, 2.85), (4, 4.85), (9, 6.85)] {
        let kt = kt_u_notch_axial(Fix128::from_int(h), Fix128::ONE).to_f64();
        assert!((kt - want).abs() < 1e-9, "h/r={} kt={}", h, kt);
    }
    // scale-invariance: depends on h/r only
    let a = kt_u_notch_axial(Fix128::from_int(3), Fix128::from_int(2));
    let b = kt_u_notch_axial(Fix128::from_int(300), Fix128::from_int(200));
    assert!((a.to_f64() - b.to_f64()).abs() < 1e-9);
}

#[test]
fn inglis_exact_and_monotonic() {
    // K_t = 1 + 2a/b
    for (a, b, want) in [
        (1, 1, 3.0),
        (4, 2, 5.0),
        (1, 4, 1.5),
        (10, 1, 21.0),
        (0, 3, 1.0),
    ] {
        let kt = kt_elliptical_hole(Fix128::from_int(a), Fix128::from_int(b)).to_f64();
        assert!((kt - want).abs() < 1e-9, "a={} b={} kt={}", a, b, kt);
    }
    // a -> 0 (hole parallel to load, slit) gives K_t -> 1 from above
    let thin = kt_elliptical_hole(fx(1, 1000), Fix128::ONE).to_f64();
    assert!(thin > 1.0 && thin < 1.01);
}

/// recommended_fillet_radius_mm: result satisfies K_t <= target, and it is the
/// MINIMUM radius (a slightly smaller radius violates the target).
#[test]
fn recommended_fillet_is_minimal_and_satisfying() {
    let d = Fix128::from_int(20);
    let big = Fix128::from_int(40);
    for tgt in [fx(15, 10), fx(2, 1), fx(25, 10), fx(3, 1)] {
        let r = recommended_fillet_radius_mm(d, big, tgt);
        let kt = kt_shaft_shoulder_bending(r, d, big);
        assert!(
            kt <= tgt,
            "satisfy: target {} kt {}",
            tgt.to_f64(),
            kt.to_f64()
        );
        let smaller = r - fx(1, 1000);
        let kt_s = kt_shaft_shoulder_bending(smaller, d, big);
        assert!(
            kt_s > tgt,
            "minimality: target {} r {} kt(r-1e-3)={}",
            tgt.to_f64(),
            r.to_f64(),
            kt_s.to_f64()
        );
    }
}

/// An unreachable target (below the fit's floor) must not be reported as satisfied.
#[test]
#[ignore = "known defect: AUD-A-S1W5-005: recommended_fillet_radius_mm returns d/2 (K_t=1.43 > target 1.2) for an unreachable target with no signal; existing test only comments on it"]
fn recommended_fillet_unreachable_target_not_silently_returned() {
    let d = Fix128::from_int(20);
    let big = Fix128::from_int(40);
    let tgt = fx(12, 10);
    let r = recommended_fillet_radius_mm(d, big, tgt);
    let kt = kt_shaft_shoulder_bending(r, d, big);
    assert!(
        kt <= tgt,
        "returned r={} gives K_t={} > target {}",
        r.to_f64(),
        kt.to_f64(),
        tgt.to_f64()
    );
}

/// Loose target (>= max of the table): minimum radius is 0 (anything satisfies);
/// the bisection lower bound r/d = 0.01 is not a minimum.
#[test]
#[ignore = "known defect: AUD-A-S1W5-006: recommended_fillet_radius_mm returns ~0.01*d for a target every radius satisfies (bisection lo=0.01 not 0)"]
fn recommended_fillet_trivial_target_gives_zero_radius() {
    let d = Fix128::from_int(20);
    let big = Fix128::from_int(40);
    let r = recommended_fillet_radius_mm(d, big, Fix128::from_int(5));
    assert!(r.to_f64() < 1e-6, "r = {}", r.to_f64());
}

#[test]
fn recommended_fillet_uses_actual_large_diameter() {
    // doc says "for a D/d = 2 shaft" but the argument is honoured: bigger step -> bigger r
    let d = Fix128::from_int(20);
    let r_small_step = recommended_fillet_radius_mm(d, Fix128::from_int(24), fx(25, 10));
    let r_big_step = recommended_fillet_radius_mm(d, Fix128::from_int(60), fx(25, 10));
    assert!(r_big_step > r_small_step);
}

/// Bin edges of the step table (strict `<` upper bound: an edge value belongs to the
/// upper bin) and the bins just either side, at D/d = 1 + 10 (scale = 2.0 exactly).
#[test]
fn shoulder_bending_bin_edges_pinned() {
    let d = Fix128::ONE;
    let big = Fix128::from_int(11); // scale = 1 + 0.1*10 = 2
    let eps = fx(1, 1000);
    // (edge numerator/100, base below edge, base at/above edge)
    let edges = [
        (2, 3.0, 2.9),
        (5, 2.9, 2.2),
        (10, 2.2, 1.8),
        (20, 1.8, 1.5),
        (30, 1.5, 1.3),
    ];
    for (e, below, above) in edges {
        let edge = fx(e, 100);
        let at = kt_shaft_shoulder_bending(edge, d, big).to_f64();
        let lo = kt_shaft_shoulder_bending(edge - eps, d, big).to_f64();
        let hi = kt_shaft_shoulder_bending(edge + eps, d, big).to_f64();
        assert!((at - above * 2.0).abs() < 1e-9, "edge {e}/100 at: {at}");
        assert!((lo - below * 2.0).abs() < 1e-9, "edge {e}/100 below: {lo}");
        assert!((hi - above * 2.0).abs() < 1e-9, "edge {e}/100 above: {hi}");
    }
}

/// All three guard branches return one and the same finite sentinel.
#[test]
fn zero_denominator_sentinels_are_consistent() {
    let s1 = kt_elliptical_hole(Fix128::from_int(2), Fix128::ZERO);
    let s2 = kt_shaft_shoulder_bending(Fix128::ONE, Fix128::ZERO, Fix128::from_int(2));
    let s3 = kt_u_notch_axial(Fix128::ONE, Fix128::ZERO);
    let want = Fix128::from_int(i64::MAX >> 8);
    assert_eq!(s1, want);
    assert_eq!(s2, want);
    assert_eq!(s3, want);
}
