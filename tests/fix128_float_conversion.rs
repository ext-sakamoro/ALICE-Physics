//! Float → `Fix128` conversion at the edges of its range
//!
//! `Fix128` holds `[-2^63, 2^63 - 2^-64]` (`hi: i64`, `lo: u64` the fraction). A finite
//! float outside that range saturates to the nearest end with its sign; inside it the
//! conversion is the truncation toward zero of the 64-bit fraction (unchanged). The
//! expected values are written from the representation, not from the code under test.

use alice_physics_core::math::{Fix128, FloatToFixError, Vec3Fix};

/// `-2^63`, the smallest `Fix128`
const LOWEST: Fix128 = Fix128::from_raw(i64::MIN, 0);
/// `2^63 - 2^-64`, the largest `Fix128`
const HIGHEST: Fix128 = Fix128::from_raw(i64::MAX, u64::MAX);

/// Below `-2^63` the value saturates to `-2^63`: before, `hi = i64::MIN` then `hi - 1`
/// overflowed (a panic in debug, a wrap to a large positive value in release)
#[test]
fn a_finite_value_below_the_range_saturates_to_the_lowest() {
    let two63 = 9_223_372_036_854_775_808.0f64;
    for f in [-1e19, -two63 * 2.0, -f64::MAX, -(two63 + 2048.0)] {
        assert_eq!(Fix128::from_f64(f), LOWEST, "{f:e}");
        assert_eq!(Fix128::from_f32(f as f32), LOWEST, "{f:e} as f32");
    }
    // the end itself is exact
    assert_eq!(Fix128::from_f64(-two63), LOWEST);
}

/// At and above `2^63` the value saturates to the largest `Fix128`
#[test]
fn a_finite_value_at_or_above_the_range_saturates_to_the_highest() {
    let two63 = 9_223_372_036_854_775_808.0f64;
    for f in [two63, 1e19, f64::MAX] {
        assert_eq!(Fix128::from_f64(f), HIGHEST, "{f:e}");
        assert_eq!(Fix128::from_f32(f as f32), HIGHEST, "{f:e} as f32");
    }
}

/// Inside the range nothing changes: the integer part and the 64-bit fraction
#[test]
fn values_inside_the_range_keep_their_bits() {
    let cases = [
        (0.0, Fix128::from_raw(0, 0)),
        (1.5, Fix128::from_raw(1, 1 << 63)),
        (-1.5, Fix128::from_raw(-2, 1 << 63)),
        (-0.25, Fix128::from_raw(-1, 3 << 62)),
        // the largest f64 below 2^63: 2^63 - 1024, an integer
        (
            9_223_372_036_854_774_784.0,
            Fix128::from_raw(9_223_372_036_854_774_784, 0),
        ),
        (
            -9_223_372_036_854_774_784.0,
            Fix128::from_raw(-9_223_372_036_854_774_784, 0),
        ),
    ];
    for (f, want) in cases {
        assert_eq!(Fix128::from_f64(f), want, "{f:e}");
    }
    assert_eq!(
        Vec3Fix::from_f32(-1.5, 0.25, -1e30),
        Vec3Fix::new(
            Fix128::from_raw(-2, 1 << 63),
            Fix128::from_raw(0, 1 << 62),
            LOWEST
        )
    );
}

/// The named ends are the representation's ends
#[test]
fn min_and_max_are_the_ends_of_the_range() {
    assert_eq!(Fix128::MIN, LOWEST);
    assert_eq!(Fix128::MAX, HIGHEST);
}

/// `try_from_*` refuses what has no `Fix128` and agrees with `from_*` elsewhere
#[test]
fn try_from_refuses_non_finite_and_out_of_range_values() {
    let two63 = 9_223_372_036_854_775_808.0f64;
    for f in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0xfff8_0000_0000_0001),
    ] {
        assert_eq!(
            Fix128::try_from_f64(f),
            Err(FloatToFixError::NotFinite),
            "{f:e}"
        );
    }
    for f in [two63, -1e19, f64::MAX, -(two63 + 2048.0)] {
        assert_eq!(
            Fix128::try_from_f64(f),
            Err(FloatToFixError::OutOfRange),
            "{f:e}"
        );
    }
    for f in [0.0, -0.0, 1.5, -0.25, -two63, 9_223_372_036_854_774_784.0] {
        assert_eq!(Fix128::try_from_f64(f), Ok(Fix128::from_f64(f)), "{f:e}");
    }
    assert_eq!(
        Fix128::try_from_f32(f32::NAN),
        Err(FloatToFixError::NotFinite)
    );
    assert_eq!(Fix128::try_from_f32(1e30), Err(FloatToFixError::OutOfRange));
    assert_eq!(
        Fix128::try_from_f32(-1.5),
        Ok(Fix128::from_raw(-2, 1 << 63))
    );
    assert_eq!(
        Vec3Fix::try_from_f32(1.0, f32::INFINITY, f32::NAN),
        Err(FloatToFixError::NotFinite)
    );
    assert_eq!(
        Vec3Fix::try_from_f32(1.0, -1e30, 0.0),
        Err(FloatToFixError::OutOfRange)
    );
    assert_eq!(
        Vec3Fix::try_from_f32(-1.5, 0.25, 0.0),
        Ok(Vec3Fix::from_f32(-1.5, 0.25, 0.0))
    );
}

/// The stated values for non-finite input: `∓∞` saturate with their sign, NaN gives zero
#[test]
fn from_f64_of_non_finite_values_in_release() {
    assert_eq!(
        Fix128::from_f64(std::hint::black_box(f64::NEG_INFINITY)),
        LOWEST
    );
    assert_eq!(
        Fix128::from_f64(std::hint::black_box(f64::INFINITY)),
        HIGHEST
    );
    assert_eq!(
        Fix128::from_f64(std::hint::black_box(f64::NAN)),
        Fix128::ZERO
    );
}
