//! `Fix128::div_int(n)` returns the same bits as `self / Fix128::from_int(n)`.
//!
//! oracle: two references that do not call `div_int`. (1) The `/` operator, the long division
//! that `tests/fix128_bit_identity.rs` keeps equal to its earlier implementations. (2) The
//! closed form of the claim on the raw value `v` (`v / 2⁶⁴` is the number): with
//! `q = floor(|v| / |n|)` on unsigned 128-bit integers, the result is `q` when `v` and `n` have
//! the same sign and the wrapping negation of `q` otherwise. That is
//! `floor(|v| · 2⁶⁴ / (|n| · 2⁶⁴)) = floor(|v| / |n|)`, the identity in the doc of `div_int`.

use alice_physics_core::math::Fix128;

fn raw(v: i128) -> Fix128 {
    Fix128 {
        hi: (v >> 64) as i64,
        lo: v as u64,
    }
}

fn bits(v: Fix128) -> (i64, u64) {
    (v.hi, v.lo)
}

/// The closed form: magnitude rounded down, sign of the two signs, `MIN` wraps as `/` does.
fn closed_form(v: i128, n: i64) -> Fix128 {
    if n == 0 {
        return Fix128::ZERO;
    }
    let q = v.unsigned_abs() / u128::from(n.unsigned_abs());
    let neg = (v < 0) != (n < 0);
    raw(if neg {
        (q as i128).wrapping_neg()
    } else {
        q as i128
    })
}

fn check(v: i128, n: i64) {
    let x = raw(v);
    let got = bits(x.div_int(n));
    assert_eq!(
        got,
        bits(x / Fix128::from_int(n)),
        "div_int differs from `/` at raw {v:#x}, n {n}"
    );
    assert_eq!(
        got,
        bits(closed_form(v, n)),
        "div_int differs from the closed form at raw {v:#x}, n {n}"
    );
}

const DIVISORS: [i64; 17] = [
    1,
    2,
    3,
    6,
    7,
    8,
    10,
    1000,
    -1,
    -2,
    -3,
    -6,
    i64::MAX,
    i64::MIN,
    i64::MIN + 1,
    1 << 32,
    (1 << 32) + 1,
];

#[test]
fn boundary_values_match_the_long_division() {
    let one = 1i128 << 64;
    let values = [
        0,
        1,
        -1,
        2,
        -2,
        5,
        -5,
        6,
        -6,
        7,
        -7,
        one,
        -one,
        one + 1,
        -one - 1,
        6 * one,
        -6 * one,
        i128::MAX,
        i128::MIN,
        i128::MIN + 1,
        i128::MAX - 1,
    ];
    let mut compared = 0;
    for &v in &values {
        for &n in DIVISORS.iter().chain(&[0]) {
            check(v, n);
            compared += 1;
        }
    }
    assert_eq!(compared, values.len() * (DIVISORS.len() + 1));
}

/// The cases the rounding rule decides, written out: the remainder is dropped from the
/// magnitude, so a negative quotient rounds toward zero, and `MIN` keeps the wrap of `/`.
#[test]
fn rounding_and_wrap_are_those_of_the_division() {
    // -7 ulp / 6 = -1 ulp (toward zero; the floor of the number would be -2 ulp)
    assert_eq!(bits(raw(-7).div_int(6)), bits(raw(-1)));
    assert_eq!(bits(raw(7).div_int(-6)), bits(raw(-1)));
    assert_eq!(bits(raw(-7).div_int(-6)), bits(raw(1)));
    // below one divisor: the magnitude rounds to zero on both sides
    assert_eq!(bits(raw(-5).div_int(6)), bits(Fix128::ZERO));
    assert_eq!(bits(raw(5).div_int(6)), bits(Fix128::ZERO));
    // one exact: 6 / 6 = 1, -6 / 6 = -1
    assert_eq!(bits(Fix128::from_int(6).div_int(6)), bits(Fix128::ONE));
    assert_eq!(
        bits(Fix128::from_int(-6).div_int(6)),
        bits(Fix128::from_int(-1))
    );
    // MIN: |MIN| = 2¹²⁷ and the negation wraps back, as for `/`
    let min = raw(i128::MIN);
    assert_eq!(bits(min.div_int(1)), bits(min));
    assert_eq!(bits(min.div_int(-1)), bits(min));
    assert_eq!(bits(min.div_int(2)), bits(raw(i128::MIN / 2)));
    // a zero divisor returns zero, as `/` does
    assert_eq!(bits(Fix128::ONE.div_int(0)), bits(Fix128::ZERO));
}

/// A fixed sequence (splitmix64), so every run and every platform compares the same inputs.
struct Seq(u64);

impl Seq {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[test]
fn twenty_thousand_samples_of_every_magnitude_match() {
    let mut seq = Seq(0x0064_6976_5f69_6e74);
    let mut compared = 0;
    for _ in 0..20_000 {
        let wide = ((u128::from(seq.next()) << 64) | u128::from(seq.next())) as i128;
        // every magnitude: shift the value down by 0..127 bits
        let v = wide >> (seq.next() % 128);
        let n = match seq.next() % 4 {
            0 => DIVISORS[(seq.next() % DIVISORS.len() as u64) as usize],
            1 => (seq.next() % 64) as i64 - 32,
            2 => seq.next() as i64 >> (seq.next() % 64),
            _ => seq.next() as i64,
        };
        check(v, n);
        compared += 1;
    }
    assert_eq!(compared, 20_000);
}
