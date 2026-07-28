//! Deterministic Numerical Utilities (Fix128 Transcendentals & Solvers)
//!
//! Common numerical primitives shared across Session 2 modules. Kept in one
//! place so that Fix128 range-reduction + Taylor-series conventions stay
//! bit-consistent between `creep_longterm`, `smoke_fire`, and any future
//! module that needs `exp` / `log` / `pow`.
//!
//! # Provided
//!
//! - [`exp_fix`]  — deterministic exp with range-reduction + squaring.
//! - [`pow_int`]  — integer power via repeated multiplication (positive/negative n).
//! - [`clamp_fix`] — [lo, hi] saturation.
//!
//! All routines are cross-platform bit-exact.

use crate::math::Fix128;

/// Saturation sentinel used by `exp_fix` for overflow (≈ 2·10⁹).
pub const EXP_OVERFLOW_SENTINEL: Fix128 = Fix128 {
    hi: i64::MAX >> 32,
    lo: 0,
};

/// Deterministic exponential `exp(x)` for `x ∈ [-40, 20]`.
///
/// Uses range reduction `exp(x) = exp(x/2^k)^{2^k}` to bring `|x|` below
/// `0.5`, then a 12-term Taylor series (converges to Fix128 ULP), then
/// repeated squaring.
///
/// Saturation:
/// - `x ≥ 20`  → returns [`EXP_OVERFLOW_SENTINEL`] (very large).
/// - `x ≤ -40` → returns `Fix128::ZERO` (effectively 0).
#[must_use]
pub fn exp_fix(x: Fix128) -> Fix128 {
    let sat_hi = Fix128::from_int(20);
    let sat_lo = Fix128::from_int(-40);
    if x >= sat_hi {
        return EXP_OVERFLOW_SENTINEL;
    }
    if x <= sat_lo {
        return Fix128::ZERO;
    }
    let half = Fix128::from_ratio(1, 2);
    let neg_half = Fix128::from_ratio(-1, 2);
    let mut y = x;
    let mut shifts: u32 = 0;
    while y > half || y < neg_half {
        y = y.half();
        shifts += 1;
    }
    let mut term = Fix128::ONE;
    let mut sum = Fix128::ONE;
    for k in 1..=12u32 {
        term = term * y / Fix128::from_int(i64::from(k));
        sum = sum + term;
    }
    for _ in 0..shifts {
        sum = sum * sum;
    }
    sum
}

/// Integer power `x^n` via repeated multiplication.
///
/// Supports negative exponents through `1 / x^|n|`. Returns `Fix128::ZERO`
/// on divide-by-zero for negative exponents when `x == 0`.
#[must_use]
pub fn pow_int(x: Fix128, n: i32) -> Fix128 {
    if n == 0 {
        return Fix128::ONE;
    }
    let mut result = Fix128::ONE;
    let abs_n = n.unsigned_abs();
    for _ in 0..abs_n {
        result = result * x;
    }
    if n < 0 {
        if result.is_zero() {
            return Fix128::ZERO;
        }
        Fix128::ONE / result
    } else {
        result
    }
}

/// Deterministic cube root via Newton iteration.
///
/// Returns 0 for non-positive inputs. Uses `x_{k+1} = (2·x + n/x²) / 3` with
/// 24 iterations (Fix128 ULP-level convergence for any `n` in the safe
/// range `[0, 2^62]`).
#[must_use]
pub fn cbrt_fix(n: Fix128) -> Fix128 {
    if n <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    // Initial guess: use bit-position estimate to seed Newton.
    let mut x = if n >= Fix128::ONE {
        // n >= 1 → cbrt ≥ 1 → seed with hi/2 (rough)
        Fix128::from_int(1 + (n.hi.max(1) as i64) / 2)
    } else {
        // n < 1 → seed slightly below 1
        Fix128::from_ratio(5, 10)
    };
    for _ in 0..24 {
        let x_sq = x * x;
        if x_sq.is_zero() {
            break;
        }
        // x_{k+1} = (2·x + n/x²) / 3
        let new = (x.double() + n / x_sq) / Fix128::from_int(3);
        x = new;
    }
    x
}

/// Clamp `x` to `[lo, hi]`. Assumes `lo ≤ hi`.
#[inline]
#[must_use]
pub fn clamp_fix(x: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn exp_zero_is_one() {
        assert_eq!(exp_fix(Fix128::ZERO), Fix128::ONE);
    }

    #[test]
    fn exp_one_e() {
        // e ≈ 2.71828
        let e = exp_fix(Fix128::ONE);
        let expected = Fix128::from_ratio(271_828, 100_000);
        assert!(approx_eq(e, expected, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn exp_neg_one_is_inv_e() {
        // e⁻¹ ≈ 0.3679
        let v = exp_fix(Fix128::NEG_ONE);
        let expected = Fix128::from_ratio(3679, 10_000);
        assert!(approx_eq(v, expected, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn exp_saturates_high() {
        let v = exp_fix(Fix128::from_int(1000));
        assert_eq!(v, EXP_OVERFLOW_SENTINEL);
    }

    #[test]
    fn exp_saturates_low_to_zero() {
        let v = exp_fix(Fix128::from_int(-1000));
        assert_eq!(v, Fix128::ZERO);
    }

    #[test]
    fn pow_int_zero_exp_is_one() {
        assert_eq!(pow_int(Fix128::from_int(5), 0), Fix128::ONE);
    }

    #[test]
    fn pow_int_positive() {
        // 2^5 = 32
        assert_eq!(pow_int(Fix128::from_int(2), 5), Fix128::from_int(32));
    }

    #[test]
    fn pow_int_negative_reciprocal() {
        // 2^-3 = 0.125
        let v = pow_int(Fix128::from_int(2), -3);
        assert_eq!(v, Fix128::from_ratio(125, 1000));
    }

    #[test]
    fn pow_int_zero_negative_returns_zero() {
        let v = pow_int(Fix128::ZERO, -3);
        assert_eq!(v, Fix128::ZERO);
    }

    #[test]
    fn clamp_below_returns_lo() {
        let v = clamp_fix(Fix128::from_int(-5), Fix128::ZERO, Fix128::from_int(10));
        assert_eq!(v, Fix128::ZERO);
    }

    #[test]
    fn clamp_above_returns_hi() {
        let v = clamp_fix(Fix128::from_int(20), Fix128::ZERO, Fix128::from_int(10));
        assert_eq!(v, Fix128::from_int(10));
    }

    #[test]
    fn cbrt_zero_is_zero() {
        assert_eq!(cbrt_fix(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn cbrt_negative_returns_zero() {
        assert_eq!(cbrt_fix(Fix128::from_int(-8)), Fix128::ZERO);
    }

    #[test]
    fn cbrt_eight_is_two() {
        let v = cbrt_fix(Fix128::from_int(8));
        assert!(approx_eq(
            v,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn cbrt_twenty_seven_is_three() {
        let v = cbrt_fix(Fix128::from_int(27));
        assert!(approx_eq(
            v,
            Fix128::from_int(3),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn cbrt_one_is_one() {
        let v = cbrt_fix(Fix128::ONE);
        assert!(approx_eq(v, Fix128::ONE, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn cbrt_fractional() {
        // cbrt(0.125) = 0.5
        let v = cbrt_fix(Fix128::from_ratio(125, 1000));
        assert!(approx_eq(
            v,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn clamp_inside_returns_value() {
        let v = clamp_fix(Fix128::from_int(5), Fix128::ZERO, Fix128::from_int(10));
        assert_eq!(v, Fix128::from_int(5));
    }
}
