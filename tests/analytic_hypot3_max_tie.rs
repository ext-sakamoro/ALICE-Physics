//! The fact the `replace > with >= in hypot3` entries of
//! `scripts/mutants-equivalent.txt` rest on: `hypot3` keeps the running largest
//! magnitude `m` and replaces it when a later component is larger; with `>=` it
//! also replaces it when the later component is equal. That changes nothing only
//! if two `Fix128` values that compare equal are the same value in every bit,
//! measured here over values built by different routes (integer, ratio,
//! negation and `abs`, sums, products, raw words).

use alice_physics_core::Fix128;

/// Pairs that compare equal are the same 128 bits, and the ordering agrees.
#[test]
fn fix128_values_that_compare_equal_are_equal_in_every_bit() {
    let pairs = [
        (Fix128::from_int(3), Fix128::from_ratio(6, 2)),
        (Fix128::from_int(-7).abs(), Fix128::from_int(7)),
        (
            Fix128::from_ratio(1, 4) + Fix128::from_ratio(1, 4),
            Fix128::from_ratio(1, 2),
        ),
        (Fix128::from_int(4) * Fix128::from_ratio(1, 4), Fix128::ONE),
        (
            Fix128::from_raw(0, 1 << 24),
            Fix128::from_raw(0, 1 << 23) + Fix128::from_raw(0, 1 << 23),
        ),
        (-Fix128::from_int(2), Fix128::from_int(-2)),
        (Fix128::ZERO.abs(), -Fix128::ZERO),
    ];
    let mut compared = 0;
    for (a, b) in pairs {
        assert!(
            a == b,
            "{} and {} are meant to be equal",
            a.to_f64(),
            b.to_f64()
        );
        assert_eq!(
            (a.hi, a.lo),
            (b.hi, b.lo),
            "equal values with different words"
        );
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Equal);
        assert!(
            a >= b && !(a > b),
            "`>=` and `>` differ exactly on equality"
        );
        compared += 1;
    }
    assert_eq!(compared, pairs.len());
}
