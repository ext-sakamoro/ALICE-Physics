//! Lib tests of `src/math.rs`: `sin` / `cos` over the whole `Fix128` range.
//!
//! Included from `src/math.rs` as a `#[cfg(test)]` module, so they run with
//! `cargo test --lib`, the test set the mutation run uses.

use super::*;

/// `(hi, lo, sin, cos)` of `Fix128 { hi, lo }`.
/// oracle: the angle reduced modulo 2π and expanded as a Taylor series in
/// 100-digit decimal arithmetic (Python `decimal`, π to 100 digits), rounded
/// to f64. Independent of the CORDIC and of `alice-det-math`.
const LARGE_ANGLES: &[(i64, u64, f64, f64)] = &[
    (
        9223372036854775807,
        0x0000000000000000,
        0.5303352662202238,
        0.8477880073480187,
    ),
    (
        -9223372036854775808,
        0x0000000000000000,
        -0.9999303766734422,
        0.011800076512800236,
    ),
    (
        9223372036854775807,
        0xffffffffffffffff,
        0.9999303766734422,
        0.011800076512800236,
    ),
    (
        -9223372036854775808,
        0x0000000000000001,
        -0.9999303766734422,
        0.011800076512800236,
    ),
    (
        4611686018427387904,
        0x0000000000000000,
        -0.7029224436192089,
        -0.7112665029764864,
    ),
    (
        -4611686018427387904,
        0x0000000000003039,
        0.7029224436192084,
        -0.7112665029764869,
    ),
    (
        1000000000000000,
        0x0000000000000000,
        0.8582727931702359,
        -0.5131937377869703,
    ),
    (
        -1000000000000000,
        0x8000000000000000,
        -0.9992434207779639,
        -0.03889197901821149,
    ),
    (
        123456789012345,
        0x000000003ade68b1,
        -0.598657294204856,
        0.801005270953519,
    ),
    (
        1099511627776,
        0x1000000000000000,
        -0.46200094768064515,
        -0.8868794305553522,
    ),
    (
        -1125899906842624,
        0x0000000000000005,
        -0.49639651520894085,
        0.8680959046605506,
    ),
    (
        1000003,
        0x0000000000000000,
        0.4786854087960669,
        -0.8779864915850029,
    ),
    (
        7,
        0x0000000000000000,
        0.6569865987187891,
        0.7539022543433046,
    ),
    (
        -7,
        0x0000000000000000,
        -0.6569865987187891,
        0.7539022543433046,
    ),
];

/// `sin` and `cos` of angles up to the ends of the `Fix128` range reduce to
/// the right angle: the reduction neither wraps around nor loses the turns'
/// error (at `hi = i64::MAX` an angle is about 1.5 · 10^18 turns).
#[test]
fn sin_cos_hold_over_the_whole_range() {
    for &(hi, lo, sin, cos) in LARGE_ANGLES {
        let x = Fix128 { hi, lo };
        let (s, c) = x.sin_cos();
        assert!(
            (s.to_f64() - sin).abs() < 1e-12,
            "sin({hi}, {lo:#x}) = {} vs {sin}",
            s.to_f64()
        );
        assert!(
            (c.to_f64() - cos).abs() < 1e-12,
            "cos({hi}, {lo:#x}) = {} vs {cos}",
            c.to_f64()
        );
        assert_eq!((x.sin(), x.cos()), (s, c));
    }
}
