//! The CI profile `ci-test` (Cargo.toml) builds the whole suite at opt-level 2 and keeps
//! debug assertions and overflow checks on: the suite's oracles count on both (a
//! `debug_assert!` inside the engine, an arithmetic overflow that must panic rather than
//! wrap). Run under that profile with a count gate
//! (`scripts/run_tests.sh test_profile_checks=2 -- cargo test --profile ci-test --test
//! test_profile_checks`, ci.yml and scripts/preflight.sh):
//!
//! - debug assertions off: both tests are compiled out, 0 executed, the gate is red
//! - overflow checks off: the overflow test wraps and fails
//!
//! A release build (`cargo test --release`) compiles neither test, as intended.

#![cfg(debug_assertions)]

use std::hint::black_box;

#[test]
fn debug_assertions_are_on() {
    let fired = std::panic::catch_unwind(|| debug_assert!(black_box(false), "fires")).is_err();
    assert!(
        fired,
        "a failing debug_assert! did not panic: debug assertions are off"
    );
}

#[test]
fn an_integer_overflow_panics_rather_than_wraps() {
    let sum = std::panic::catch_unwind(|| black_box(u8::MAX) + black_box(1u8));
    assert!(
        sum.is_err(),
        "u8::MAX + 1 gave {:?}: overflow checks are off",
        sum.ok()
    );
}
