//! Audit S1-5 oracles for `alice_physics::fillet_stress`.
//! Expected values are derived from the documented formulas, not from the
//! implementation: the shoulder fillet is checked against `pilkey`, an f64
//! transcription of the published Peterson curve fit (Pilkey, Formulas for
//! Stress, Strain, and Structural Matrices 2nd ed.), written here separately.
#![allow(clippy::disallowed_methods)]

use alice_physics::fillet_stress::*;
use alice_physics::math::Fix128;

fn fx(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// The published fit for a stepped round bar in bending: `x = h/r`,
/// `t = 2h/D`, `K_t = C1 + C2 t + C3 t^2 + C4 t^3` (valid for 0.1 <= x <= 20).
fn pilkey(x: f64, t: f64) -> f64 {
    let s = x.sqrt();
    let c = if x <= 2.0 {
        [
            0.947 + 1.206 * s - 0.131 * x,
            0.022 - 3.405 * s + 0.915 * x,
            0.869 + 1.777 * s - 0.555 * x,
            -0.810 + 0.422 * s - 0.260 * x,
        ]
    } else {
        [
            1.232 + 0.832 * s - 0.008 * x,
            -3.813 + 0.968 * s - 0.260 * x,
            7.423 - 4.868 * s + 0.869 * x,
            -3.839 + 3.070 * s - 0.600 * x,
        ]
    };
    c[0] + c[1] * t + c[2] * t * t + c[3] * t * t * t
}

/// `K_t` at `D/d` and `r/d` (d = 1000 mm).
fn kt_at(dd: f64, rd: f64) -> f64 {
    let d = Fix128::from_int(1000);
    kt_shaft_shoulder_bending(
        Fix128::from_f64(rd * 1000.0),
        d,
        Fix128::from_f64(dd * 1000.0),
    )
    .to_f64()
}

/// `(x, t) = (h/r, 2h/D)` for `D/d` and `r/d`.
fn x_t(dd: f64, rd: f64) -> (f64, f64) {
    let h = (dd - 1.0) / 2.0;
    (h / rd, 2.0 * h / dd)
}

/// AUD-A-S1W5-001: the function follows the published fit, not a table of its
/// own. Where the fit needs no adjustment (D/d <= 2, x = h/r in [0.1, 1.5] or
/// [2, 20], see the doc) it equals `pilkey` up to Fix128 rounding; this
/// includes the doc's D/d = 2 values 2.27 / 1.80 / 1.48 at r/d = 0.05 / 0.10 /
/// 0.20. For D/d = 1.1, r/d = 0.05 it is about 1.81 (the earlier linear
/// scaling of the D/d = 2 curve gave 1.12, below the chart)
#[test]
fn shoulder_bending_follows_the_published_fit() {
    let mut checked = 0;
    for dd in [1.02, 1.05, 1.1, 1.2, 1.5, 2.0] {
        for rd in [
            0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0,
        ] {
            let (x, t) = x_t(dd, rd);
            if !((0.1..=1.5).contains(&x) || (2.0..=20.0).contains(&x)) {
                continue;
            }
            let got = kt_at(dd, rd);
            let want = pilkey(x, t);
            assert!(
                (got - want).abs() < 1e-9,
                "D/d={dd} r/d={rd}: {got} vs {want}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 30, "compared {checked}");
    for (rd, want) in [(0.05, 2.27), (0.10, 1.80), (0.20, 1.48)] {
        assert!(
            (kt_at(2.0, rd) - want).abs() < 0.005,
            "doc value at r/d={rd}"
        );
    }
    assert!((kt_at(1.1, 0.05) - 1.814).abs() < 1e-3);
}

/// Where the fit is adjusted the result is never below it (the adjustments
/// only raise it): the ramp over x in [1.5, 2] and the running maximum that
/// removes the dip of the fit for large steps
#[test]
fn shoulder_bending_is_never_below_the_fit() {
    for dd in [1.05, 1.2, 2.0, 3.0, 6.0, 10.0] {
        let h = (dd - 1.0) / 2.0;
        for i in 0..=400 {
            let x = 0.1 + 19.9 * f64::from(i) / 400.0;
            let (_, t) = x_t(dd, h / x);
            let got = kt_at(dd, h / x);
            let fit = pilkey(x, t);
            assert!(got >= fit - 1e-9, "D/d={dd} x={x}: {got} < {fit}");
        }
    }
}

/// Beyond x = 20 the doc's extrapolation `1 + (K_t(20) - 1) sqrt(x / 20)`;
/// below x = 0.1 linear to 1 at x = 0
#[test]
fn shoulder_bending_outside_the_fit_range() {
    for dd in [1.1, 2.0] {
        let h = (dd - 1.0) / 2.0;
        let k20 = kt_at(dd, h / 20.0);
        for x in [25.0, 50.0, 200.0] {
            let want = 1.0 + (k20 - 1.0) * (x / 20.0_f64).sqrt();
            assert!((kt_at(dd, h / x) - want).abs() < 1e-6, "D/d={dd} x={x}");
        }
        let k01 = kt_at(dd, h / 0.1);
        for x in [0.05, 0.01] {
            let want = 1.0 + (k01 - 1.0) * x / 0.1;
            assert!((kt_at(dd, h / x) - want).abs() < 1e-6, "D/d={dd} x={x}");
        }
    }
}

/// No shoulder (D = d) means no stress concentration: K_t = 1.
#[test]
fn shoulder_bending_without_step_is_unity() {
    // AUD-A-S1W5-002
    let d = Fix128::from_int(20);
    for r in [fx(1, 1), fx(2, 1), fx(5, 1)] {
        assert_eq!(kt_shaft_shoulder_bending(r, d, d), Fix128::ONE);
        // a smaller "large" diameter is no shoulder either
        assert_eq!(
            kt_shaft_shoulder_bending(r, d, d - Fix128::ONE),
            Fix128::ONE
        );
    }
}

/// AUD-A-S1W5-003: continuous and non-increasing in r for every step, across
/// the seams (x = 0.1, the ramp over [1.5, 2], x = 20) and the large-step dip
/// of the fit: on a fine grid in x no step of K_t exceeds the slope bound
#[test]
fn shoulder_bending_is_continuous_and_monotone_in_the_radius() {
    for dd in [1.01, 1.1, 1.5, 2.0, 3.0, 6.0] {
        let h = (dd - 1.0) / 2.0;
        let mut prev: Option<f64> = None;
        for i in 0..=6000 {
            let x = 0.02 + 39.98 * f64::from(i) / 6000.0; // r falls as x rises
            let k = kt_at(dd, h / x);
            if let Some(p) = prev {
                assert!(k >= p - 1e-12, "D/d={dd} x={x}: K_t fell {p} -> {k}");
            }
            prev = Some(k);
        }
        // no step at the seams: 1e-6 either side differs by about the slope
        // times 2e-6 (the unramped step between the two sets was up to 0.03)
        for x0 in [0.1, 1.5, 2.0, 20.0] {
            let lo = kt_at(dd, h / (x0 - 1e-6));
            let hi = kt_at(dd, h / (x0 + 1e-6));
            assert!((hi - lo).abs() < 1e-4, "D/d={dd} seam x={x0}: {lo} -> {hi}");
        }
    }
}

/// K_t >= 1 always: stress concentration can't relieve stress.
#[test]
fn u_notch_kt_never_below_one() {
    // AUD-A-S1W5-004
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
    // 4.5 needs r/d below 0.01 (K_t(r/d = 0.01) is about 4.07 at D/d = 2)
    for tgt in [fx(15, 10), fx(2, 1), fx(25, 10), fx(3, 1), fx(45, 10)] {
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
#[ignore = "known defect: AUD-A-S1W5-005: recommended_fillet_radius_mm returns d/2 (K_t=1.2297 > target 1.2 at D/d = 2) for an unreachable target with no signal; existing test only comments on it"]
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

/// AUD-A-S1W5-006: a target that the sharp corner (r = 0) already meets gives
/// 0, not the bisection's lower end. With a shoulder the sharp corner is the
/// saturating sentinel, so this is the shaft without a shoulder (K_t = 1 for
/// every r) and a target at the sentinel itself
#[test]
fn recommended_fillet_trivial_target_gives_zero_radius() {
    let d = Fix128::from_int(20);
    // exactly 0: the bisection alone would stop at 0.5 d / 2^30 (about 9e-9 mm here)
    let r = recommended_fillet_radius_mm(d, d, fx(3, 2));
    assert_eq!(r, Fix128::ZERO, "r = {}", r.to_f64());
    let sentinel = Fix128::from_int(i64::MAX >> 8);
    let r = recommended_fillet_radius_mm(d, Fix128::from_int(40), sentinel);
    assert_eq!(r, Fix128::ZERO, "r = {}", r.to_f64());
}

#[test]
fn recommended_fillet_uses_actual_large_diameter() {
    // doc says "for a D/d = 2 shaft" but the argument is honoured: bigger step -> bigger r
    let d = Fix128::from_int(20);
    let r_small_step = recommended_fillet_radius_mm(d, Fix128::from_int(24), fx(25, 10));
    let r_big_step = recommended_fillet_radius_mm(d, Fix128::from_int(60), fx(25, 10));
    assert!(r_big_step > r_small_step);
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
