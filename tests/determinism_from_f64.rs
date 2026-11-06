//! `Fix128::from_f64` saturates outside the range, the same in every build profile, and keeps the bits
//! of every value inside it.
//!
//! oracle: outside the range, the range itself: `Fix128` holds [-2^63, 2^63), so the nearest value to
//! anything below is -2^63 and to anything from 2^63 up is 2^63 - 2^-64. Inside it, `from_f64_ref` is
//! the implementation before saturation, word for word, compared on every magnitude of `f64`; the
//! determinism goldens hold the same bits downstream. NaN stays 0.

use alice_physics_core::math::Fix128;

const LOWEST: Fix128 = Fix128 {
    hi: i64::MIN,
    lo: 0,
};
const HIGHEST: Fix128 = Fix128 {
    hi: i64::MAX,
    lo: u64::MAX,
};
/// 2^63, the first `f64` out of range upward.
const TOP: f64 = 9_223_372_036_854_775_808.0;

fn from_f64_ref(f: f64) -> Fix128 {
    let hi = f as i64;
    let frac = f - (hi as f64);
    let abs_frac = frac.abs();
    let lo = (abs_frac * (1u128 << 64) as f64) as u64;
    if f < 0.0 && lo != 0 {
        Fix128 {
            hi: hi - 1,
            lo: (!lo).wrapping_add(1),
        }
    } else {
        Fix128 { hi, lo }
    }
}

/// The next `f64` towards positive infinity.
fn up(f: f64) -> f64 {
    let bits = f.to_bits();
    f64::from_bits(if f > 0.0 { bits + 1 } else { bits - 1 })
}

/// The next `f64` towards negative infinity.
fn down(f: f64) -> f64 {
    let bits = f.to_bits();
    f64::from_bits(if f > 0.0 { bits - 1 } else { bits + 1 })
}

#[test]
fn above_the_range_is_the_highest_value() {
    for f in [TOP, up(TOP), 1e19, 1e30, f64::MAX, f64::INFINITY] {
        assert_eq!(Fix128::from_f64(f), HIGHEST, "{f:e}");
    }
    assert_eq!(Fix128::from_f32(f32::INFINITY), HIGHEST);
    assert_eq!(Fix128::from_f32(f32::MAX), HIGHEST);
}

#[test]
fn below_the_range_is_the_lowest_value() {
    for f in [down(-TOP), -1e19, -1e30, f64::MIN, f64::NEG_INFINITY] {
        assert_eq!(Fix128::from_f64(f), LOWEST, "{f:e}");
    }
    assert_eq!(Fix128::from_f32(f32::NEG_INFINITY), LOWEST);
    assert_eq!(Fix128::from_f32(f32::MIN), LOWEST);
}

#[test]
fn the_ends_of_the_range_are_exact() {
    assert_eq!(Fix128::from_f64(-TOP), LOWEST);
    assert_eq!(
        Fix128::from_f64(up(-TOP)),
        Fix128 {
            hi: i64::MIN + 1024,
            lo: 0
        }
    );
    assert_eq!(
        Fix128::from_f64(down(TOP)),
        Fix128 {
            hi: i64::MAX - 1023,
            lo: 0
        }
    );
}

#[test]
fn nan_is_zero() {
    for f in [f64::NAN, -f64::NAN] {
        assert_eq!(Fix128::from_f64(f), Fix128::ZERO);
    }
    assert_eq!(Fix128::from_f32(f32::NAN), Fix128::ZERO);
}

#[test]
fn inside_the_range_the_bits_are_the_previous_ones() {
    let mut compared = 0u32;
    let mut same = |f: f64| {
        assert!(f.abs() < TOP || f == -TOP, "{f:e} is not inside");
        assert_eq!(Fix128::from_f64(f), from_f64_ref(f), "{f:e}");
        compared += 1;
    };
    for f in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ] {
        same(f);
    }
    same(down(TOP));
    same(up(-TOP));
    same(-TOP);
    // every exponent inside the range, both signs, with a varying mantissa
    let mut state = 0x4636_3431_u64;
    for exponent in (1023 - 80)..(1023 + 63) {
        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let bits = (exponent << 52) | (state >> 12);
            same(f64::from_bits(bits));
            same(-f64::from_bits(bits));
        }
    }
    assert_eq!(compared, 11 + 143 * 512 * 2);
}
