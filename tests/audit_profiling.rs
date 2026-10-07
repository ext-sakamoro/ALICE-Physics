//! Audit oracles for `profiling`: claims not covered by
//! `analytic_profiling_wiring.rs`.
//!
//! Closed forms (integer bookkeeping done in the test, not by the code under test):
//!
//! ```text
//! begin_frame   :  frame_count += 1, stats = default, entries untouched
//! frame_count   =  number of begin_frame calls since new()/reset()
//! record(a)+record(b) never decreases total_ticks
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::profiling::{
    PhysicsProfiler, ProfileEntry, StepStats, TickCounter, STAGE_BROADPHASE, STAGE_SOLVER,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// `begin_frame` counts frames exactly: after N calls `frame_count = N`.
#[test]
fn begin_frame_counts_calls_exactly() {
    let mut p = PhysicsProfiler::new();
    assert_eq!(p.frame_count, 0);
    for n in 1..=5u64 {
        p.begin_frame();
        assert_eq!(p.frame_count, n);
    }
    // enabled = false does not stop the frame counter (pinned: the counter is
    // not a stage record).
    p.enabled = false;
    p.begin_frame();
    assert_eq!(p.frame_count, 6);
}

/// `begin_frame` documents "reset per-frame stats": every `StepStats` field is
/// zero afterwards, and the per-stage rows are not touched.
#[test]
fn begin_frame_zeroes_all_nine_stats_and_keeps_rows() {
    let mut p = PhysicsProfiler::new();
    p.stats = StepStats {
        broadphase_pairs: 1,
        narrowphase_tests: 2,
        active_contacts: 3,
        active_manifolds: 4,
        active_bodies: 5,
        static_bodies: 6,
        solver_iterations: 7,
        ccd_checks: 8,
        joint_count: 9,
    };
    p.record(STAGE_SOLVER, 40);
    p.record(STAGE_SOLVER, 60);
    p.begin_frame();
    let s = p.stats;
    assert_eq!(
        [
            s.broadphase_pairs,
            s.narrowphase_tests,
            s.active_contacts,
            s.active_manifolds,
            s.active_bodies,
            s.static_bodies,
            s.solver_iterations,
            s.ccd_checks,
            s.joint_count
        ],
        [0; 9]
    );
    let e = p.get(STAGE_SOLVER).expect("in range");
    assert_eq!(
        (e.total_ticks, e.call_count, e.last_ticks, e.peak_ticks),
        (100, 2, 60, 60)
    );
}

/// Semantic pin (not a defect claim): a stage that is not recorded in the new
/// frame still reports the previous frame's value from `last_ticks`.
#[test]
fn last_ticks_survives_begin_frame_pinned() {
    let mut p = PhysicsProfiler::new();
    p.begin_frame();
    p.record(STAGE_BROADPHASE, 77);
    p.begin_frame();
    assert_eq!(p.last_ticks(STAGE_BROADPHASE), 77);
}

/// `reset` clears the per-frame stats as well as the rows (every field).
#[test]
fn reset_clears_stats_fields() {
    let mut p = PhysicsProfiler::new();
    p.stats.joint_count = 9;
    p.stats.ccd_checks = 8;
    p.begin_frame();
    p.stats.solver_iterations = 7;
    p.reset();
    assert_eq!(
        p.stats.joint_count + p.stats.ccd_checks + p.stats.solver_iterations,
        0
    );
    assert_eq!(p.frame_count, 0);
}

/// `peak_ticks` is the max, and a tie / smaller value later does not lower it.
#[test]
fn peak_is_monotone_non_decreasing() {
    let mut e = ProfileEntry::new("x");
    let mut prev = 0;
    for t in [5u64, 3, 9, 9, 1, 0, 8] {
        e.record(t);
        assert!(e.peak_ticks >= prev);
        prev = e.peak_ticks;
    }
    assert_eq!(e.peak_ticks, 9);
}

/// `TickCounter::advance(0)` is the identity and `now` equals the sum.
#[test]
fn counter_advance_zero_is_identity() {
    let mut c = TickCounter::new();
    c.advance(10);
    c.advance(0);
    assert_eq!(c.now(), 10);
    assert_eq!(c.elapsed(c.now()), 0);
}

/// AUD-A-S4W1-001 (known defect): `ProfileEntry::record` accumulates with a
/// plain `+=`, so tick totals that overflow `u64` panic in debug and wrap
/// silently in release (`total_ticks` goes DOWN, `average_ticks` then lies),
/// while the sibling `TickCounter` documents explicit wrapping.
/// Expected (closed form): a total of u64::MAX + u64::MAX must never be
/// smaller than either term (saturate) and must not panic.
#[test]
// AUD-A-S4W1-001
fn entry_total_never_decreases_on_overflow() {
    let mut e = ProfileEntry::new("x");
    e.record(u64::MAX);
    let r = catch_unwind(AssertUnwindSafe(|| e.record(u64::MAX)));
    assert!(r.is_ok(), "record panicked on overflow");
    assert!(
        e.total_ticks == u64::MAX,
        "total_ticks went down: {}",
        e.total_ticks
    );
}

/// The call count saturates too: from `u64::MAX - 1` two more calls stop at
/// `u64::MAX` (no panic, no wrap to 0).
#[test]
fn call_count_saturates() {
    let mut e = ProfileEntry::new("x");
    e.call_count = u64::MAX - 1;
    e.record(1);
    e.record(1);
    assert_eq!(e.call_count, u64::MAX);
}
