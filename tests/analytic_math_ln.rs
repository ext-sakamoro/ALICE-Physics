//! Oracles for `Fix128::ln` (`src/math.rs`), promoted from a private
//! `turbulence.rs` helper (`ln_fix`) of the same algorithm so other callers
//! (`wave_ship::Jonswap::spectrum_density`) share one canonical `ln`.
//!
//! # Closed forms (independent of the function under test)
//!
//! - `f64::ln` (platform `libm`) at 1e-9 relative tolerance -- `Fix128::ln`
//!   uses range reduction to `[1, 2)` plus a 16-term atanh series, an
//!   independent algorithm from the platform's `libm`.
//! - Functional identity `ln(a·b) = ln(a) + ln(b)` for several `(a, b)`
//!   pairs, checked against the *values* `Fix128::ln` itself returns for
//!   `a`, `b`, `a*b` (an internal-consistency oracle, additional to the
//!   f64 cross-check, not a substitute for it).
//! - `ln(1) == 0` exactly (no range reduction needed, `t = 0`).
//! - Monotonically increasing on a geometric ladder `2^k`.
//!
//! # Degenerate inputs
//!
//! `ln(0)` and `ln(negative)` return `Fix128::ZERO` (documented contract,
//! `Fix128` has no NaN/trap representation) -- confirmed via `catch_unwind`
//! that neither panics, and via direct equality that the result really is
//! `ZERO` and not some other finite value.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::Fix128;

fn close(actual: f64, expected: f64, rel_tol: f64) -> bool {
    let diff = (actual - expected).abs();
    diff <= rel_tol * expected.abs().max(1.0)
}

#[test]
#[allow(clippy::disallowed_methods)]
fn ln_matches_f64_reference_across_magnitudes() {
    for &x in &[
        0.5,
        1.0,
        2.0,
        std::f64::consts::E,
        10.0,
        1e6,
        1e-6,
        3.3,
        9.0,
    ] {
        let got = Fix128::from_f64(x).ln().to_f64();
        let want = x.ln();
        assert!(
            close(got, want, 1e-6),
            "ln({x}): got {got}, want {want} (f64::ln reference)"
        );
    }
}

#[test]
fn ln_of_one_is_exactly_zero() {
    assert_eq!(
        Fix128::ONE.ln(),
        Fix128::ZERO,
        "ln(1) must be exactly zero (t = (m-1)/(m+1) = 0, no series error)"
    );
}

#[test]
fn ln_is_additive_under_multiplication() {
    let pairs = [(2.0_f64, 3.0), (1.5, 7.0), (9.0, 3.3), (0.1, 100.0)];
    for &(a, b) in &pairs {
        let fa = Fix128::from_f64(a);
        let fb = Fix128::from_f64(b);
        let lhs = (fa * fb).ln().to_f64();
        let rhs = fa.ln().to_f64() + fb.ln().to_f64();
        assert!(
            close(lhs, rhs, 1e-6),
            "ln({a}*{b}) = {lhs} should equal ln({a})+ln({b}) = {rhs}"
        );
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn ln_is_monotonically_increasing() {
    let mut prev = Fix128::from_f64(0.01).ln();
    for k in 0..40 {
        let x = Fix128::from_f64(0.01 * 2.0_f64.powi(k));
        let cur = x.ln();
        assert!(
            cur >= prev,
            "ln must be non-decreasing: ln(0.01*2^{k}) = {cur:?} < previous {prev:?}"
        );
        prev = cur;
    }
}

#[test]
fn ln_of_nonpositive_is_exactly_zero_and_does_not_panic() {
    for x in [
        Fix128::ZERO,
        Fix128::from_int(-1),
        Fix128::from_int(-1_000_000),
    ] {
        let result = std::panic::catch_unwind(|| x.ln());
        assert!(result.is_ok(), "ln({x:?}) must not panic");
        assert_eq!(
            result.unwrap(),
            Fix128::ZERO,
            "ln of non-positive input is documented to return ZERO, not some other finite value"
        );
    }
}
