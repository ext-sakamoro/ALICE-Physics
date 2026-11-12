//! `Fix128::sin_cos` near zero.
//!
//! oracle (the closed forms, f64): for `|x|` from `1e-15` to `1e-3` rad, both
//! signs, `sin x` and `cos x` agree with the f64 functions at the same
//! argument. Below `2⁻¹⁰` rad the Taylor series is used, whose first omitted
//! terms (`x⁷/5040`, `x⁸/40320`) are far below the `2⁻⁶⁴` resolution, so the
//! error is the f64 reference's own (`2⁻⁵²` relative) plus a few `2⁻⁶⁴`. At and
//! above it the 48 CORDIC iterations stop at an angle of `2⁻⁴⁷`, so the error
//! is within `2⁻⁴⁶` absolute. At the switch the two forms agree to that same
//! bound (no step larger than the CORDIC's own error).
//!
//! What it catches: the CORDIC alone near zero (its absolute error of about
//! `1e-14` is a relative error of 7e-5 at `1e-11` rad and 6% at `1e-13`), and a
//! series cut a term short.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 sin and cos of the same argument"
)]

use alice_physics_core::math::Fix128;

const SWITCH: f64 = 1.0 / 1024.0;

/// `v` as f64 through its magnitude (`to_f64` of a small negative value adds a
/// small fraction to −1 and keeps only about `1e-16` of it).
fn f64_of(v: Fix128) -> f64 {
    if v.is_negative() {
        -(-v).to_f64()
    } else {
        v.to_f64()
    }
}

fn bound(x: f64, reference: f64) -> f64 {
    if x.abs() < SWITCH {
        reference.abs() * f64::EPSILON + 8.0 * p2(-64)
    } else {
        p2(-46)
    }
}

/// `2^e`, exact (built from the bits; the platform `powi` is not allowed).
fn p2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// `10^(12 k / 240)` as repeated products of `10^(12 / 240)` (a fixed
/// sequence, the same on every platform; the sweep needs no exact decades).
fn ratio_pow(k: u32) -> f64 {
    let mut m = 1.0;
    for _ in 0..k {
        m *= 1.122_018_454_301_963_3;
    }
    m
}

#[test]
fn sin_and_cos_near_zero_match_the_closed_forms() {
    let mut worst = (0.0f64, 0.0f64);
    let mut compared = 0;
    for k in 0..=240 {
        let magnitude = 1e-15 * ratio_pow(k);
        for sign in [1.0, -1.0] {
            let fx = Fix128::from_f64(sign * magnitude);
            let x = f64_of(fx);
            let (s, c) = fx.sin_cos();
            let (es, ec) = ((f64_of(s) - x.sin()).abs(), (f64_of(c) - x.cos()).abs());
            assert!(
                es <= bound(x, x.sin()),
                "sin({x:e}) = {:e}, off by {es:e}",
                f64_of(s)
            );
            assert!(
                ec <= bound(x, x.cos()),
                "cos({x:e}) = {:.17}, off by {ec:e}",
                f64_of(c)
            );
            if x.abs() < SWITCH {
                worst.0 = worst.0.max(es / x.abs());
            }
            worst.1 = worst.1.max(ec);
            compared += 1;
        }
    }
    assert_eq!(compared, 2 * 241);
    eprintln!(
        "largest relative sin error below the switch {:e}, largest cos error {:e}",
        worst.0, worst.1
    );
}

#[test]
fn the_two_forms_meet_at_the_switch() {
    let switch = Fix128::from_ratio(1, 1024);
    let ulp = Fix128::from_raw(0, 1);
    let (below_s, below_c) = (switch - ulp).sin_cos();
    let (at_s, at_c) = switch.sin_cos();
    // across one 2⁻⁶⁴ step the functions move by about 2⁻⁶⁴ (sin) and 0 (cos)
    let step_s = f64_of(at_s - below_s).abs();
    let step_c = f64_of(at_c - below_c).abs();
    assert!(step_s <= p2(-46), "sin steps by {step_s:e} at the switch");
    assert!(step_c <= p2(-46), "cos steps by {step_c:e} at the switch");
}
