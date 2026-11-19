//! High-precision integer kernels behind the `Fix128` transcendental functions.
//!
//! Every kernel works on `i128` fixed-point values with 120 or 124 fraction bits
//! (60 more than `Fix128`'s 64), uses only integer shifts, additions,
//! multiplications and divisions, and rounds once to the nearest `Fix128` at the
//! end. The extra bits absorb the truncation of the intermediate steps, so the
//! rounded result is within about one `Fix128` ulp of the exact value. The
//! constants below are the stated real numbers rounded (or floored where noted)
//! to the stated number of fraction bits, computed with 400-bit arithmetic.

use crate::math::Fix128;

/// atan(2^-i) at Q124 (floor), i = 0..68 (the residual angle after the last
/// step is below 2^-67)
const ATAN_Q124: [i128; 68] = [
    0x0C90FDAA22168C234C4C6628B80DC1CD,
    0x076B19C1586ED3DA2B7F222F65E1D468,
    0x03EB6EBF25901BAC55B71E7BD7DE885F,
    0x01FD5BA9AAC2F6DC65912F313E7D111D,
    0x00FFAADDB967EF4E36CB2792DC0E2E0D,
    0x007FF556EEA5D892A13BCEBBB6ED4631,
    0x003FFEAAB776E5356EF9E31590057DD8,
    0x001FFFD555BBBA972D00C46A3F77CC15,
    0x000FFFFAAAADDDDB94BB12AFB6B6D4F7,
    0x0007FFFF55556EEEEA5CA6ADEAB02251,
    0x0003FFFFEAAAAB77776E52E5A019FBCE,
    0x0001FFFFFD55555BBBBBA97297625624,
    0x0000FFFFFFAAAAAADDDDDDB94B94D5BD,
    0x00007FFFFFF5555556EEEEEEA5CA5CB4,
    0x00003FFFFFFEAAAAAAB7777776E52E52,
    0x00001FFFFFFFD5555555BBBBBBBA9729,
    0x00000FFFFFFFFAAAAAAAADDDDDDDDB94,
    0x000007FFFFFFFF555555556EEEEEEEEA,
    0x000003FFFFFFFFEAAAAAAAAB77777777,
    0x000001FFFFFFFFFD555555555BBBBBBB,
    0x000000FFFFFFFFFFAAAAAAAAAADDDDDD,
    0x0000007FFFFFFFFFF55555555556EEEE,
    0x0000003FFFFFFFFFFEAAAAAAAAAAB777,
    0x0000001FFFFFFFFFFFD55555555555BB,
    0x0000000FFFFFFFFFFFFAAAAAAAAAAAAD,
    0x00000007FFFFFFFFFFFF555555555555,
    0x00000003FFFFFFFFFFFFEAAAAAAAAAAA,
    0x00000001FFFFFFFFFFFFFD5555555555,
    0x00000000FFFFFFFFFFFFFFAAAAAAAAAA,
    0x000000007FFFFFFFFFFFFFF555555555,
    0x000000003FFFFFFFFFFFFFFEAAAAAAAA,
    0x000000001FFFFFFFFFFFFFFFD5555555,
    0x000000000FFFFFFFFFFFFFFFFAAAAAAA,
    0x0000000007FFFFFFFFFFFFFFFF555555,
    0x0000000003FFFFFFFFFFFFFFFFEAAAAA,
    0x0000000001FFFFFFFFFFFFFFFFFD5555,
    0x0000000000FFFFFFFFFFFFFFFFFFAAAA,
    0x00000000007FFFFFFFFFFFFFFFFFF555,
    0x00000000003FFFFFFFFFFFFFFFFFFEAA,
    0x00000000001FFFFFFFFFFFFFFFFFFFD5,
    0x00000000000FFFFFFFFFFFFFFFFFFFFA,
    0x000000000007FFFFFFFFFFFFFFFFFFFF,
    0x000000000003FFFFFFFFFFFFFFFFFFFF,
    0x000000000001FFFFFFFFFFFFFFFFFFFF,
    0x000000000000FFFFFFFFFFFFFFFFFFFF,
    0x0000000000007FFFFFFFFFFFFFFFFFFF,
    0x0000000000003FFFFFFFFFFFFFFFFFFF,
    0x0000000000001FFFFFFFFFFFFFFFFFFF,
    0x0000000000000FFFFFFFFFFFFFFFFFFF,
    0x00000000000007FFFFFFFFFFFFFFFFFF,
    0x00000000000003FFFFFFFFFFFFFFFFFF,
    0x00000000000001FFFFFFFFFFFFFFFFFF,
    0x00000000000000FFFFFFFFFFFFFFFFFF,
    0x000000000000007FFFFFFFFFFFFFFFFF,
    0x000000000000003FFFFFFFFFFFFFFFFF,
    0x000000000000001FFFFFFFFFFFFFFFFF,
    0x000000000000000FFFFFFFFFFFFFFFFF,
    0x0000000000000007FFFFFFFFFFFFFFFF,
    0x0000000000000003FFFFFFFFFFFFFFFF,
    0x0000000000000001FFFFFFFFFFFFFFFF,
    0x0000000000000000FFFFFFFFFFFFFFFF,
    0x00000000000000007FFFFFFFFFFFFFFF,
    0x00000000000000003FFFFFFFFFFFFFFF,
    0x00000000000000001FFFFFFFFFFFFFFF,
    0x00000000000000000FFFFFFFFFFFFFFF,
    0x000000000000000007FFFFFFFFFFFFFF,
    0x000000000000000003FFFFFFFFFFFFFF,
    0x000000000000000001FFFFFFFFFFFFFF,
];
/// prod cos(atan(2^-i)), i = 0..68, at Q124 (nearest)
const CORDIC_K_Q124: i128 = 0x09B74EDA8435E5A67F5F9092BD7FD40F;
/// pi/2 at Q124 (nearest)
const HALF_PI_Q124: i128 = 0x1921FB54442D18469898CC51701B839A;
/// pi at Q124 (nearest)
const PI_Q124: i128 = 0x3243F6A8885A308D313198A2E0370734;
/// ln 2 at Q120 (nearest)
const LN2_Q120: i128 = 0x00B17217F7D1CF79ABC9E3B39803F2F7;
/// pi/2 at 188 fraction bits (nearest), little-endian u64 limbs
const HALF_PI_F188: [u64; 4] = [0x252049C1114CF98F, 0x9898CC51701B839A, 0x1921FB54442D1846, 0x0000000000000000];
/// 2/pi at Q64 (nearest), for the quadrant estimate
const TWO_OVER_PI_RAW: u64 = 0xA2F9836E4E44152A;

/// `1.0` at Q124.
const ONE_Q124: i128 = 1 << 124;
/// `√2` at Q124 (floor), the upper end of the reduced mantissa range of `ln`.
const SQRT2_Q124: i128 = 0x16A09E667F3BCC908B2FB1366EA957D3;

/// Unsigned 128 x 128 -> 256-bit product as (high, low).
fn widening_mul_u128(a: u128, b: u128) -> (u128, u128) {
    let (a1, a0) = ((a >> 64) as u64, a as u64);
    let (b1, b0) = ((b >> 64) as u64, b as u64);
    let p00 = u128::from(a0) * u128::from(b0);
    let p01 = u128::from(a0) * u128::from(b1);
    let p10 = u128::from(a1) * u128::from(b0);
    let p11 = u128::from(a1) * u128::from(b1);
    let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
    let lo = (p00 & u128::from(u64::MAX)) | (mid << 64);
    let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
    (hi, lo)
}

/// `a · b / 2^f`, rounded toward zero, for signed fixed-point values whose true
/// quotient fits `i128` (`f < 128`).
fn mul_shift(a: i128, b: i128, f: u32) -> i128 {
    let negative = (a < 0) != (b < 0);
    let (hi, lo) = widening_mul_u128(a.unsigned_abs(), b.unsigned_abs());
    let magnitude = (hi << (128 - f)) | (lo >> f);
    if negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    }
}

/// `(a << 124) / b` for `0 ≤ a < b`, `0 < b < 2^126`: the 124 fraction bits of
/// `a / b`, by restoring division (truncated).
fn div_q124_unit(a: i128, b: i128) -> i128 {
    let (mut rem, b) = (a as u128, b as u128);
    let mut q: u128 = 0;
    for _ in 0..124 {
        rem <<= 1;
        q <<= 1;
        if rem >= b {
            rem -= b;
            q |= 1;
        }
    }
    q as i128
}

/// Round a value with `f` fraction bits to the nearest `Fix128` (ties away from
/// negative infinity, i.e. `+½` then floor); `f ≥ 64`.
fn round_to_fix128(v: i128, f: u32) -> Fix128 {
    let shift = f - 64;
    let r = if shift == 0 {
        v
    } else {
        (v >> shift) + ((v >> (shift - 1)) & 1)
    };
    Fix128::from_raw((r >> 64) as i64, r as u64)
}

/// The raw two's-complement value of a `Fix128`.
fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

/// `Fix128` from its raw value.
fn from_i128(v: i128) -> Fix128 {
    Fix128::from_raw((v >> 64) as i64, v as u64)
}

// ---------------------------------------------------------------------------
// exp / ln
// ---------------------------------------------------------------------------

/// `e^a` for an argument with 120 fraction bits, as the nearest `Fix128`.
/// Saturates at the representable maximum when the result reaches `2^63`, and
/// returns `ZERO` when it is below half an ulp.
///
/// `a = k·ln2 + r` with `|r| ≤ ½ ln2`, `e^r` from its Taylor series at Q124
/// (Horner form, 28 terms: the first omitted term is below `0.35^28/28! < 2^-150`),
/// then `2^k` by a shift.
pub(crate) fn exp_q120(a: i128) -> Fix128 {
    let k = (a + LN2_Q120 / 2).div_euclid(LN2_Q120);
    if k >= 64 {
        return Fix128::from_raw(i64::MAX, u64::MAX);
    }
    if k < -70 {
        return Fix128::ZERO;
    }
    let r = (a - k * LN2_Q120) << 4;
    let mut s = ONE_Q124;
    for n in (1..=28i128).rev() {
        s = ONE_Q124 + mul_shift(r, s, 124) / n;
    }
    // s ∈ [0.70, 1.42] at Q124; the result's raw value is s · 2^(k − 60).
    let shift = 60 - k;
    if shift <= 0 {
        let m = (s as u128) << (-shift) as u32;
        if m > i128::MAX as u128 {
            return Fix128::from_raw(i64::MAX, u64::MAX);
        }
        return from_i128(m as i128);
    }
    if shift >= 127 {
        return Fix128::ZERO;
    }
    let shift = shift as u32;
    from_i128((s >> shift) + ((s >> (shift - 1)) & 1))
}

/// `ln x` for `x > 0` at Q120.
///
/// `x = 2^k · m` with `m ∈ [√2/2, √2)`, `ln m = 2·atanh t` with
/// `t = (m − 1)/(m + 1)`, `|t| ≤ 0.172`, series to the 27th odd power (first
/// omitted term `< 0.172^55/55 < 2^-145`).
pub(crate) fn ln_q120(x: Fix128) -> i128 {
    let v = raw(x) as u128;
    let p = 127 - v.leading_zeros() as i32;
    let mut k = i128::from(p - 64);
    let mut m = if p <= 124 {
        (v << (124 - p) as u32) as i128
    } else {
        (v >> (p - 124) as u32) as i128
    };
    if m > SQRT2_Q124 {
        m >>= 1;
        k += 1;
    }
    let num = m - ONE_Q124;
    let den = m + ONE_Q124;
    let t = if num < 0 {
        -div_q124_unit(-num, den)
    } else {
        div_q124_unit(num, den)
    };
    let t2 = mul_shift(t, t, 124);
    let mut term = t;
    let mut sum = t;
    for n in 1..=26i128 {
        term = mul_shift(term, t2, 124);
        sum += term / (2 * n + 1);
    }
    k * LN2_Q120 + ((sum << 1) >> 4)
}

/// `ln x` rounded to the nearest `Fix128`; `x ≤ 0` returns `ZERO`.
pub(crate) fn ln(x: Fix128) -> Fix128 {
    if x <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    round_to_fix128(ln_q120(x), 120)
}

/// `e^x` rounded to the nearest `Fix128`, saturating from `2^63`.
pub(crate) fn exp(x: Fix128) -> Fix128 {
    // e^x < 2^-65 (half an ulp, rounds to 0) for x < −65·ln 2 ≈ −45.05
    if x.hi < -46 {
        return Fix128::ZERO;
    }
    if x.hi >= 44 {
        return Fix128::from_raw(i64::MAX, u64::MAX);
    }
    exp_q120(raw(x) << 56)
}

/// `x^y` for `x > 0`, `y ≥ 0`, saturating at the representable maximum
/// instead of wrapping. `x ≤ 0` or `y < 0` returns `ZERO`.
///
/// An integer `y` is tried first by exponentiation by squaring: when every
/// product is exact (no bit dropped, no overflow) that is the exact power and is
/// returned as is, so `2^62`, `3^39`, `1.5^40` and the like stay exact. Otherwise
/// (a fractional `y`, or a product that rounds) the result is `e^(y · ln x)`
/// rounded once, within about one ulp.
pub(crate) fn powf_pos(x: Fix128, y: Fix128) -> Fix128 {
    if x <= Fix128::ZERO || y.is_negative() {
        return Fix128::ZERO;
    }
    if y.lo == 0 {
        if let Some(r) = exact_int_pow(x, y.hi as u64) {
            return r;
        }
    }
    exp_mul_ln(x, y)
}

/// `x^n` by exponentiation by squaring when every product is exact and
/// representable; `None` as soon as one is not.
fn exact_int_pow(x: Fix128, mut n: u64) -> Option<Fix128> {
    let mut r = Fix128::ONE;
    let mut base = x;
    while n > 0 {
        if n & 1 != 0 {
            r = exact_mul(r, base)?;
        }
        n >>= 1;
        if n > 0 {
            base = exact_mul(base, base)?;
        }
    }
    Some(r)
}

/// `a · b` for non-negative `a`, `b` when the product is exact (its low 64 bits
/// below the `Fix128` resolution are zero) and below `2^63`.
fn exact_mul(a: Fix128, b: Fix128) -> Option<Fix128> {
    let (hi, lo) = widening_mul_u128(raw(a) as u128, raw(b) as u128);
    // the product has 128 fraction bits: drop 64, keep 128, the rest must be 0
    if lo as u64 != 0 || hi >> 63 != 0 {
        return None;
    }
    Some(from_i128(((hi << 64) | (lo >> 64)) as i128))
}

/// `e^(e · ln x)` for `x > 0` and an exponent of either sign, rounded once to
/// the nearest `Fix128` (the product `e · ln x` is kept at Q120, not rounded to
/// `Fix128` first). Saturates like [`exp_q120`] (a product beyond `±64` is
/// past both ends of the range).
pub(crate) fn exp_mul_ln(x: Fix128, e: Fix128) -> Fix128 {
    let l = ln_q120(x);
    let (hi, lo) = widening_mul_u128(raw(e).unsigned_abs(), l.unsigned_abs());
    let negative = (raw(e) < 0) != (l < 0);
    // the Q120 product is (hi·2^128 + lo) / 2^64; it is below 64 · 2^120 = 2^126
    // exactly when hi < 2^62. Beyond ±64 the result is past both ends of the
    // range (e^64 > 2^63, e^-64 < 2^-65).
    if hi >> 62 != 0 {
        return if negative {
            Fix128::ZERO
        } else {
            Fix128::from_raw(i64::MAX, u64::MAX)
        };
    }
    let magnitude = ((hi << 64) | (lo >> 64)) as i128;
    exp_q120(if negative { -magnitude } else { magnitude })
}

// ---------------------------------------------------------------------------
// sin / cos / atan2 (CORDIC at Q124)
// ---------------------------------------------------------------------------

/// CORDIC rotation of `(K, 0)` by `r` (`|r| ≤ 1.7`, Q124): `(sin r, cos r)` at Q124.
fn sin_cos_q124(r: i128) -> (i128, i128) {
    let (mut x, mut y, mut z) = (CORDIC_K_Q124, 0i128, r);
    for (i, &angle) in ATAN_Q124.iter().enumerate() {
        let (xs, ys) = (x >> i, y >> i);
        if z >= 0 {
            x -= ys;
            y += xs;
            z -= angle;
        } else {
            x += ys;
            y -= xs;
            z += angle;
        }
    }
    (y, x)
}

/// `θ = q·π/2 + r` with `|r| ≤ π/4 + 2^-60` at Q124 and `q = round(θ / (π/2))`
/// mod 4. The product `q·π/2` uses π/2 to 188 fraction bits, so `r` is exact to
/// about `2^-125` for every representable `θ` (`|q| < 2^63`).
fn reduce_half_pi(theta: Fix128) -> (i128, u32) {
    let t = raw(theta);
    let quarter_pi_raw: i128 = (HALF_PI_Q124 >> 60) >> 1;
    if t.unsigned_abs() <= quarter_pi_raw as u128 {
        return (t << 60, 0);
    }
    let est = theta * Fix128::from_raw(0, TWO_OVER_PI_RAW) + Fix128::ONE.half();
    let k = est.floor().hi;
    // A = t · 2^124 and B = k · (π/2 · 2^188), both mod 2^256 as four u64 limbs
    let t_ext: [u64; 4] = {
        let s = if t < 0 { u64::MAX } else { 0 };
        [t as u64, (t >> 64) as u64, s, s]
    };
    // t · 2^124 = (t · 2^60) · 2^64: shift the 256-bit value left by 124
    let a = shl256(t_ext, 124);
    let b = {
        let mag = mul256_u64(HALF_PI_F188, k.unsigned_abs());
        if k < 0 {
            neg256(mag)
        } else {
            mag
        }
    };
    let d = sub256(a, b);
    // d is r at 188 fraction bits, |r| < 1: drop 64 bits for Q124
    let r = ((d[2] as u128) << 64 | d[1] as u128) as i128;
    (r, (k & 3) as u32)
}

fn shl256(v: [u64; 4], s: u32) -> [u64; 4] {
    let (limbs, bits) = ((s / 64) as usize, s % 64);
    let mut out = [0u64; 4];
    for i in (limbs..4).rev() {
        let src = i - limbs;
        let mut w = v[src] << bits;
        if bits != 0 && src > 0 {
            w |= v[src - 1] >> (64 - bits);
        }
        out[i] = w;
    }
    out
}

fn mul256_u64(v: [u64; 4], m: u64) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry: u128 = 0;
    for i in 0..4 {
        let p = u128::from(v[i]) * u128::from(m) + carry;
        out[i] = p as u64;
        carry = p >> 64;
    }
    out
}

fn neg256(v: [u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut carry = 1u64;
    for i in 0..4 {
        let (s, c) = (!v[i]).overflowing_add(carry);
        out[i] = s;
        carry = u64::from(c);
    }
    out
}

fn sub256(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        out[i] = d2;
        borrow = u64::from(b1 || b2);
    }
    out
}

/// `(sin θ, cos θ)`, each the nearest `Fix128` to within about one ulp, for
/// every representable `θ`.
pub(crate) fn sin_cos(theta: Fix128) -> (Fix128, Fix128) {
    let (r, q) = reduce_half_pi(theta);
    let (s, c) = sin_cos_q124(r);
    let (s, c) = (round_to_fix128(s, 124), round_to_fix128(c, 124));
    match q {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

/// `atan2(y, x)` in `(−π, π]`, the nearest `Fix128` to within about one ulp.
/// `atan2(0, 0) = 0`; `x = 0` returns `±π/2` ([`Fix128::HALF_PI`]).
pub(crate) fn atan2(y: Fix128, x: Fix128) -> Fix128 {
    if x.is_zero() {
        return if y.is_zero() {
            Fix128::ZERO
        } else if y.is_negative() {
            -Fix128::HALF_PI
        } else {
            Fix128::HALF_PI
        };
    }
    let (mut xr, mut yr) = (raw(x), raw(y));
    let mut offset = 0i128;
    if xr < 0 {
        // rotate by π: atan2(y, x) = atan2(−y, −x) ± π
        offset = if yr < 0 { -PI_Q124 } else { PI_Q124 };
        xr = xr.wrapping_neg();
        yr = yr.wrapping_neg();
    }
    // scale both so the larger magnitude has its top bit at 2^123 (angle unchanged)
    let m = xr.unsigned_abs().max(yr.unsigned_abs());
    let p = 127 - m.leading_zeros() as i32;
    let (mut xv, mut yv) = if p <= 123 {
        let s = (123 - p) as u32;
        // the magnitudes fit: |v| · 2^s < 2^124
        ((xr.unsigned_abs() << s) as i128, shift_signed_left(yr, s))
    } else {
        let s = (p - 123) as u32;
        ((xr.unsigned_abs() >> s) as i128, yr >> s)
    };
    let mut z = 0i128;
    for (i, &angle) in ATAN_Q124.iter().enumerate() {
        let (xs, ys) = (xv >> i, yv >> i);
        if yv >= 0 {
            xv += ys;
            yv -= xs;
            z += angle;
        } else {
            xv -= ys;
            yv += xs;
            z -= angle;
        }
    }
    round_to_fix128(z + offset, 124)
}

fn shift_signed_left(v: i128, s: u32) -> i128 {
    if v < 0 {
        -((v.unsigned_abs() << s) as i128)
    } else {
        (v.unsigned_abs() << s) as i128
    }
}

// ---------------------------------------------------------------------------
// cbrt
// ---------------------------------------------------------------------------

/// `⌊∛x⌋` to the `Fix128` resolution (the largest `r` with `r³ ≤ x`), by binary
/// digit recurrence on the 256-bit integer `raw(x) · 2^128`; `x ≤ 0` returns
/// `ZERO`.
pub(crate) fn cbrt_floor(x: Fix128) -> Fix128 {
    if x <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    let v = raw(x) as u128;
    // N = v · 2^128; bitlen(N) = bitlen(v) + 128; result < 2^ceil(bitlen/3)
    let bits = 256 - v.leading_zeros() as i32;
    let top = (bits - 1) / 3;
    let mut r: u128 = 0;
    for b in (0..=top).rev() {
        let c = r | (1u128 << b);
        let (c2_hi, c2_lo) = widening_mul_u128(c, c);
        // c³ = c² · c compared with v · 2^128 (high 256 bits: v, low 128: 0)
        let (lo_hi, lo_lo) = widening_mul_u128(c2_lo, c);
        let (hi_hi, hi_lo) = widening_mul_u128(c2_hi, c);
        // c³ = hi·2^128 + lo, with hi = (hi_hi, hi_lo) and lo = (lo_hi, lo_lo)
        let (mid, carry) = hi_lo.overflowing_add(lo_hi);
        let top_word = hi_hi + u128::from(carry);
        // c³ = top_word·2^256 + mid·2^128 + lo_lo
        let fits = top_word == 0 && (mid < v || (mid == v && lo_lo == 0));
        if fits {
            r = c;
        }
    }
    from_i128(r as i128)
}

// The accuracy oracle (every function against the 400-bit reference table) also runs as part of
// the lib tests, so the coverage the lib tests measure includes these kernels.
#[cfg(test)]
#[path = "../tests/analytic_fix128_transcendentals.rs"]
mod analytic_fix128_transcendentals_oracle;
