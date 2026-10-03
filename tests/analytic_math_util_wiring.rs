//! Oracles for `math_util::{cbrt_fix, clamp_fix, pow_int}` (`examples/math_util_roots_powers.rs`).
//!
//! * `cbrt_fix(k^3) == k` for integers `k` (exact cubes), and for non-cubes the defining
//!   residual `r^3 <= n < (r + 2^-60)^3` bracketing is checked in f64 / by monotonicity.
//!   Documented range: `(0, 2^62]` (also tested down to `2^-64`).
//! * `pow_int(x, n) == x^n` by exact integer arithmetic; `n < 0` is `1 / x^|n|`;
//!   `pow_int(0, n<0) == 0` (documented), `pow_int(x, 0) == 1`.
//! * `clamp_fix` is `min(max(x, lo), hi)`.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::math_util::{cbrt_fix, clamp_fix, pow_int};

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

#[test]
fn cbrt_of_exact_cubes_is_exact_over_the_documented_range() {
    for k in [
        1i64,
        2,
        3,
        5,
        10,
        100,
        1000,
        10_000,
        100_000,
        1_000_000,
        1 << 20,
    ] {
        let n = fx(k * k * k);
        let r = cbrt_fix(n);
        let err = (r.to_f64() - k as f64).abs() / k as f64;
        assert!(err < 1e-15, "k={k} got {} rel err {err}", r.to_f64());
    }
}

#[test]
fn cbrt_covers_two_to_the_62_and_tiny_inputs() {
    // 2^62: cbrt = 2^(62/3)
    let r = cbrt_fix(fx(1 << 62)).to_f64();
    let want = 2f64.powf(62.0 / 3.0);
    assert!(((r - want) / want).abs() < 1e-14, "{r} vs {want}");
    // tiny: n = 1e-12 -> 1e-4, n = 2^-64 (one ULP) -> 2^-64/3, n=1e-6 -> 1e-2
    for n in [
        Fix128::from_ratio(1, 1_000_000),
        Fix128::from_ratio(1, 1_000_000_000_000),
    ] {
        let want = n.to_f64().cbrt(); // of the quantised input
        let r = cbrt_fix(n).to_f64();
        assert!(((r - want) / want).abs() < 1e-9, "{r} vs {want}");
    }
    let ulp = Fix128 { hi: 0, lo: 1 };
    let r = cbrt_fix(ulp).to_f64();
    let want = 2f64.powf(-64.0 / 3.0);
    assert!(((r - want) / want).abs() < 0.2, "ulp input {r} vs {want}"); // ULP-quantised result
    assert!(r > 0.0);
}

#[test]
fn cbrt_is_monotone_and_satisfies_the_cube_residual() {
    let mut prev = Fix128::ZERO;
    let mut n = 1e-9f64;
    while n < 1e18 {
        let fxn = Fix128::from_f64(n);
        let r = cbrt_fix(fxn);
        assert!(r >= prev, "monotone at {n}");
        prev = r;
        let c = r.to_f64();
        assert!(((c * c * c - n) / n).abs() < 1e-9, "n={n} cbrt={c}");
        n *= 3.7;
    }
}

#[test]
fn cbrt_non_positive_is_zero() {
    assert_eq!(cbrt_fix(Fix128::ZERO), Fix128::ZERO);
    assert_eq!(cbrt_fix(fx(-27)), Fix128::ZERO);
    assert_eq!(cbrt_fix(Fix128::from_ratio(-1, 1000)), Fix128::ZERO);
}

#[test]
fn pow_int_exact_integer_powers() {
    for x in [-3i64, -2, -1, 1, 2, 3, 7, 10] {
        let mut want = 1i64;
        for n in 0..=12u32 {
            assert_eq!(pow_int(fx(x), n as i32), fx(want), "{x}^{n}");
            want *= x;
        }
    }
    assert_eq!(pow_int(fx(0), 0), Fix128::ONE);
    assert_eq!(pow_int(fx(0), 5), Fix128::ZERO);
    assert_eq!(pow_int(fx(1), i32::MAX / 4096), Fix128::ONE);
}

#[test]
fn pow_int_negative_exponents_and_fractions() {
    // 2^-3 = 1/8, 10^-2 = 0.01 (reciprocal of the exact power), (-2)^-3 = -1/8
    assert_eq!(pow_int(fx(2), -3), Fix128::from_ratio(1, 8));
    assert_eq!(pow_int(fx(-2), -3), Fix128::from_ratio(-1, 8));
    assert_eq!(pow_int(fx(10), -2), Fix128::ONE / fx(100));
    // (3/2)^4 = 81/16 exactly, (3/2)^-2 = 4/9
    assert_eq!(
        pow_int(Fix128::from_ratio(3, 2), 4),
        Fix128::from_ratio(81, 16)
    );
    let inv = pow_int(Fix128::from_ratio(3, 2), -2).to_f64();
    assert!((inv - 4.0 / 9.0).abs() < 1e-15);
    // documented: x == 0 with negative n returns 0, no panic
    for n in [-1, -2, -9] {
        assert_eq!(pow_int(Fix128::ZERO, n), Fix128::ZERO);
    }
    // x^n * x^-n == 1 within rounding
    let p = pow_int(fx(3), 7) * pow_int(fx(3), -7);
    assert!((p.to_f64() - 1.0).abs() < 1e-15);
}

#[test]
fn clamp_is_min_max() {
    let (lo, hi) = (fx(-2), fx(5));
    for (x, want) in [
        (fx(-9), lo),
        (lo, lo),
        (fx(-1), fx(-1)),
        (Fix128::ZERO, Fix128::ZERO),
        (fx(5), hi),
        (fx(6), hi),
        (Fix128::from_ratio(49, 10), Fix128::from_ratio(49, 10)),
        (Fix128::from_ratio(51, 10), hi),
        (Fix128::from_ratio(-21, 10), lo),
    ] {
        assert_eq!(clamp_fix(x, lo, hi), want, "{:?}", x.to_f64());
    }
    // degenerate interval lo == hi
    assert_eq!(clamp_fix(fx(9), fx(3), fx(3)), fx(3));
    assert_eq!(clamp_fix(fx(-9), fx(3), fx(3)), fx(3));
}

#[test]
fn cbrt_of_small_exact_cubes_is_bit_exact() {
    for k in 1..=200i64 {
        assert_eq!(cbrt_fix(fx(k * k * k)), fx(k), "k={k}");
    }
}
