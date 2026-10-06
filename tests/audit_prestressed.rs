//! Audit oracle for `prestressed`: independent derivations, not copies of the
//! formulas in `src/prestressed.rs`.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::prestressed::{
    bolt_load_fraction, bolt_peak_tension, cable_pretension_n, preload_from_torque,
    recommended_preload_n, separation_load_n, tensioned_cable_stiffness_n_per_mm,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn rel(a: Fix128, b: f64) -> f64 {
    (a.to_f64() - b).abs() / b.abs()
}

/// Bolt + member free body: bolt F_b = F_i + C P, member compression F_m = F_i - (1-C) P.
/// Equilibrium of the joint interface: F_b - F_m = P (external load). Check for several P.
#[test]
fn load_sharing_equilibrium_bolt_minus_member_equals_external() {
    let fi = fx(7500.0);
    let c = bolt_load_fraction(fx(2.0e5), fx(6.0e5)); // 0.25
    assert!(rel(c, 0.25) < 1e-12);
    for p in [0.0, 1000.0, 5000.0, 9000.0] {
        let fb = bolt_peak_tension(fi, c, fx(p)).to_f64();
        let fm = 7500.0 - (1.0 - 0.25) * p;
        assert!((fb - fm - p).abs() < 1e-6, "p={p}");
    }
}

/// At the separation load the member compression is exactly zero, i.e. F_i = (1-C) P_sep.
#[test]
fn separation_load_zeroes_member_compression() {
    for (ki, kb, km) in [
        (5000.0, 1.0e5, 3.0e5),
        (12000.0, 2.0e5, 2.0e5),
        (800.0, 1.0, 9.0),
    ] {
        let c = bolt_load_fraction(fx(kb), fx(km));
        let psep = separation_load_n(fx(ki), c).to_f64();
        let cf = kb / (kb + km);
        assert!((ki - (1.0 - cf) * psep).abs() < 1e-3 * ki, "ki={ki}");
    }
}

/// Motosh: T = K F d. Round trip: torque applied back from F reproduces T.
#[test]
fn preload_round_trip_torque() {
    for (t, k, d) in [(25.0, 0.2, 10.0), (3.0, 0.1, 4.0), (120.0, 0.15, 16.0)] {
        let f = preload_from_torque(fx(t), fx(k), fx(d)).to_f64();
        let t_back = k * f * (d / 1000.0);
        assert!((t_back - t).abs() / t < 1e-6, "t={t} back={t_back}");
    }
}

/// Parabolic cable vs the exact catenary: horizontal tension H = w a with s = a (cosh(L/2a) - 1).
/// The doc claims validity for s << L: relative error is ~ (8/3)(s/L)^2 and must shrink as s/L shrinks.
#[test]
fn cable_pretension_approaches_catenary_as_sag_shrinks() {
    let exact = |w: f64, l: f64, s: f64| {
        // solve s = a (cosh(l/(2a)) - 1) for a by bisection (decreasing in a)
        let (mut lo, mut hi) = (l / 1e4, l * 1e4);
        for _ in 0..200 {
            let mid = (lo + hi) / 2.0;
            let sag = mid * (((l / (2.0 * mid)).exp() + (-(l / (2.0 * mid))).exp()) / 2.0 - 1.0);
            if sag > s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        w * (lo + hi) / 2.0
    };
    let mut prev = f64::MAX;
    for ratio in [0.1, 0.05, 0.02, 0.01] {
        let (w, l) = (0.5, 4000.0);
        let s = l * ratio;
        let t = cable_pretension_n(fx(w), fx(l), fx(s)).to_f64();
        let e = (t - exact(w, l, s)).abs() / exact(w, l, s);
        assert!(e < 3.0 * ratio * ratio, "ratio {ratio}: err {e}");
        assert!(e < prev);
        prev = e;
    }
}

/// UDL equivalence: stiffness against distributed load w L is  k = w L / s  (mid-span sag s),
/// and tension T = w L^2 / 8 s  =>  k = 8 T / L.
#[test]
fn cable_stiffness_times_sag_equals_total_distributed_load() {
    for (w, l, s) in [(0.2, 3000.0, 30.0), (1.5, 800.0, 12.0)] {
        let t = cable_pretension_n(fx(w), fx(l), fx(s));
        let k = tensioned_cable_stiffness_n_per_mm(t, fx(l)).to_f64();
        assert!(((k * s) - w * l).abs() / (w * l) < 1e-9);
    }
}

/// Doc: stiffness of a taut string against a mid-span POINT load is 4T/L, i.e. half of 8T/L.
/// The function documents that it is the UDL stiffness; check the factor-2 relation is stated by value.
#[test]
fn cable_stiffness_is_twice_the_midspan_point_load_stiffness() {
    let (t, l) = (900.0, 150.0);
    let k = tensioned_cable_stiffness_n_per_mm(fx(t), fx(l)).to_f64();
    let k_point = 4.0 * t / l;
    assert!((k - 2.0 * k_point).abs() < 1e-9);
}

#[test]
fn recommended_preload_is_product_of_three_terms() {
    // 0.9 fraction (permanent), M12 proof 830 MPa, A_t 84.3 mm^2
    let f = recommended_preload_n(fx(830.0), fx(84.3), fx(0.9)).to_f64();
    assert!(rel(Fix128::from_f64(f), 830.0 * 84.3 * 0.9) < 1e-9);
    // linear in every argument
    let f2 = recommended_preload_n(fx(830.0), fx(84.3), fx(0.45)).to_f64();
    assert!((f / f2 - 2.0).abs() < 1e-9);
}

/// bolt_load_fraction claims values in [0,1] (for non-negative stiffnesses).
#[test]
fn bolt_load_fraction_stays_in_unit_interval() {
    for kb in [0.0, 1e-3, 1.0, 1e6] {
        for km in [0.0, 1e-3, 1.0, 1e6] {
            let c = bolt_load_fraction(fx(kb), fx(km)).to_f64();
            assert!((0.0..=1.0).contains(&c), "kb={kb} km={km} c={c}");
        }
    }
}

/// Out-of-domain C > 1 (bolt softer than the formula's assumption never produces it, but the
/// separation formula is unguarded): the result must not be a negative "load".
#[test]
fn separation_load_for_c_above_one_is_not_negative() {
    // AUD-A-S1W6-001: C >= 1 never separates, the same sentinel as C = 1
    let never = separation_load_n(fx(1000.0), Fix128::ONE);
    for c in [1.2, 2.0, 1e6] {
        let p = separation_load_n(fx(1000.0), fx(c));
        assert_eq!(p, never, "P_sep = {} for C = {c}", p.to_f64());
    }
    assert_eq!(never, Fix128::from_int(i64::MAX >> 8));
}
