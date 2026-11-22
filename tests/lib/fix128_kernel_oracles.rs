//! Lib tests of the private helpers of `src/fix128_kernels.rs`, each against an
//! exact integer: the sign rule of `mul_shift`, the word edges of the 256-bit
//! product, the two input ranges of `ln_q120` (above `2^61` and the reduced
//! mantissa above `√2`), and the range reduction of `exp_q120`.
//!
//! The 400-bit reference table (`tests/analytic_fix128_transcendentals.rs`)
//! checks the public functions at 64 fraction bits; these check what that table
//! cannot reach: helper calls no public function makes (a negative second
//! operand of `mul_shift`), inputs the table has no row for (`ln` from `2^61`),
//! and errors far below one `Fix128` ulp (`ln_q120` at 120 fraction bits).
//!
//! Included from `src/fix128_kernels.rs` as a `#[cfg(test)]` module, so they run
//! with `cargo test --lib`, the test set the mutation run uses.

use super::*;

const Q124: i128 = 1 << 124;

/// `ln 2 · 2^128`, nearest (computed with 150-digit decimal arithmetic:
/// `round(Decimal(2).ln() * 2**128)`).
const LN2_Q128: u128 = 0xB172_17F7_D1CF_79AB_C9E3_B398_03F2_F6AF;

/// `k · ln 2` at 120 fraction bits, nearest, from the 128-bit constant (its
/// error, below half a unit at Q128, stays below one unit at Q120 for `k < 128`).
/// `k · LN2_Q128` needs up to 135 bits: with `LN2_Q128 = h·2^64 + l`, the
/// `k·h·2^64` part is a multiple of `2^8`, so the shift splits exactly into
/// `k·h·2^56 + ⌊(k·l + 2^7) / 2^8⌋`, both terms in 128 bits for `k < 128`.
fn k_ln2_q120(k: u128) -> i128 {
    assert!(k < 128);
    let (h, l) = (LN2_Q128 >> 64, LN2_Q128 & u128::from(u64::MAX));
    (((k * h) << 56) + ((k * l + (1 << 7)) >> 8)) as i128
}

#[test]
fn mul_shift_takes_the_sign_of_each_operand() {
    // 3 · (±1) and (±3) · 1 at Q124: exact, the magnitude 3 and the product sign
    for (a, b, want) in [
        (3 * Q124, Q124, 3 * Q124),
        (3 * Q124, -Q124, -3 * Q124),
        (-3 * Q124, Q124, -3 * Q124),
        (-3 * Q124, -Q124, 3 * Q124),
    ] {
        assert_eq!(mul_shift(a, b, 124), want, "mul_shift({a}, {b})");
    }
}

#[test]
fn a_zero_operand_gives_zero_whatever_the_other_sign() {
    // i128 has one zero: the sign chosen for a zero magnitude cannot show
    for (a, b) in [
        (0, 5 * Q124),
        (0, -5 * Q124),
        (5 * Q124, 0),
        (-5 * Q124, 0),
        (0, 0),
    ] {
        assert_eq!(mul_shift(a, b, 124), 0, "mul_shift({a}, {b})");
    }
}

#[test]
fn the_256_bit_product_is_exact_at_the_word_edges() {
    // (2^128 − 1)^2 = 2^256 − 2^129 + 1: high word 2^128 − 2, low word 1
    assert_eq!(widening_mul_u128(u128::MAX, u128::MAX), (u128::MAX - 1, 1));
    // 2^64 · 2^64 = 2^128: the carry leaves the low word empty
    assert_eq!(widening_mul_u128(1 << 64, 1 << 64), (1, 0));
    // (2^64 − 1)^2 = 2^128 − 2^65 + 1 fits the low word
    let m = u128::from(u64::MAX);
    assert_eq!(widening_mul_u128(m, m), (0, m * m));
    // the low word's high half comes from the middle sum, its low half from p00
    // only: (2^64 + 1)(2^64 + 1) = 2^128 + 2^65 + 1
    assert_eq!(
        widening_mul_u128((1 << 64) + 1, (1 << 64) + 1),
        (1, (1 << 65) + 1)
    );
}

#[test]
fn mul_shift_joins_the_high_and_low_words_exactly() {
    // x = A·2^62, y = B·2^62: x·y / 2^124 = A·B exactly, with A·B ≥ 2^64 so both
    // words of the 256-bit product carry bits of the result
    let a: i128 = 0x8000_0000_0000_0001;
    let b: i128 = 0x8000_0000_0000_0003;
    let want = a * b; // 2^126 + 2^65 + 2^64 + 3 < 2^127
    assert_eq!(mul_shift(a << 62, b << 62, 124), want);
    assert_eq!(mul_shift(-(a << 62), b << 62, 124), -want);
}

#[test]
fn the_raw_value_round_trips_through_both_words() {
    for v in [
        0,
        1,
        -1,
        i128::MIN,
        i128::MAX,
        -(1 << 64),
        (1 << 64) - 1,
        1 << 64,
    ] {
        assert_eq!(raw(from_i128(v)), v, "raw(from_i128({v}))");
    }
}

#[test]
fn ln_from_two_to_the_61_is_k_ln2() {
    // a raw value of 2^125 or 2^126 (x = 2^61, 2^62) takes the right-shift branch
    // of the mantissa; ln x = k ln 2 exactly. The error is k times that of the
    // Q120 constant (0.33 unit, so 20 at k = 62) plus the series: bound 32
    for k in [61u32, 62] {
        let x = from_i128(1 << (64 + k));
        let got = ln_q120(x);
        let want = k_ln2_q120(u128::from(k));
        assert!(got.abs_diff(want) <= 32, "ln 2^{k}: got {got}, want {want}");
    }
}

#[test]
fn ln_with_a_reduced_mantissa_above_sqrt2_keeps_full_precision() {
    // m ∈ (√2, 2) is halved to [√2/2, 1) so |t| ≤ 0.172; references with
    // 150-digit decimal arithmetic: round(Decimal(x).ln() * 2**120)
    for (x_raw, want) in [
        // 1.875 = 15/8
        (
            15_i128 << 61,
            835_564_228_497_045_147_879_125_376_200_391_627_i128,
        ),
        // 1.96875 = 63/32
        (
            63_i128 << 59,
            900_417_480_629_996_263_900_716_620_729_603_080_i128,
        ),
    ] {
        let got = ln_q120(from_i128(x_raw));
        assert!(
            got.abs_diff(want) <= 4,
            "ln_q120({x_raw}): got {got}, want {want}"
        );
    }
}

#[test]
fn the_exp_range_reduction_leaves_at_most_half_ln2() {
    // a = (j + f)·ln2 for every k the kernel uses and fractions up to the
    // next multiple: the reduced r = a − k·ln2 lies in [−½ ln2, ½ ln2]
    let half = LN2_Q120 / 2;
    let mut checked = 0;
    for j in -93_i128..=64 {
        for f in [
            0,
            1,
            LN2_Q120 / 4,
            half - 1,
            half,
            half + 1,
            LN2_Q120 * 99 / 100,
            LN2_Q120 - 1,
        ] {
            let a = j * LN2_Q120 + f;
            let k = nearest_ln2_multiple(a);
            let r = a - k * LN2_Q120;
            assert!(
                -half <= r && r <= LN2_Q120 - half,
                "a = {a}: k = {k}, r = {r}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 158 * 8);
}

#[test]
fn exp_is_zero_for_every_k_from_minus_92_to_minus_66() {
    // the result's raw value is s·2^(k − 60) with s < 2^125: below half a unit
    // from k = −66 down; −92 is the smallest k an argument the callers pass can
    // have (|a| < 2^126 from pow_exp_ln, a ≥ −46·2^120 from exp)
    for k in -92_i128..=-66 {
        for f in [-(LN2_Q120 / 2), 0, LN2_Q120 / 2 - 1] {
            assert_eq!(exp_q120(k * LN2_Q120 + f), Fix128::ZERO, "k = {k}, f = {f}");
        }
    }
}

#[test]
fn exp_below_saturation_is_an_even_multiple_of_its_shift() {
    // k = 60..=63, r < 0 (below 2^63; r = 0 at k = 63 is 2^63 itself and
    // saturates): the raw result is s << (k − 60) with s < 2^124, so it is below
    // i128::MAX and its low k − 60 bits are zero (i128::MAX is odd, so the
    // saturation test `m > i128::MAX` never meets equality)
    for k in 60_i128..=63 {
        for f in [-(LN2_Q120 / 2), -(LN2_Q120 / 4), -(LN2_Q120 / 64)] {
            let v = raw(exp_q120(k * LN2_Q120 + f));
            assert!(v > 0 && v < i128::MAX, "k = {k}, f = {f}: {v}");
            assert_eq!(v & ((1 << (k - 60)) - 1), 0, "k = {k}, f = {f}: {v}");
        }
    }
}
