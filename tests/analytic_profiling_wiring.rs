//! Oracles for `profiling`: `PhysicsProfiler` / `ProfileEntry` / `TickCounter`
//! as the production consumer `examples/profiling_stages.rs` uses them.
//!
//! Nothing here is physics; every expected value is integer arithmetic done
//! in the test on its own list of recorded ticks, never by calling the code
//! under test twice. The closed forms:
//!
//! ```text
//! average_ticks  =  total div count        (remainder dropped, 0 when count = 0)
//! last_ticks     =  the most recent record  (order matters, not the max)
//! peak_ticks     =  max over records
//! elapsed(s)     =  (now − s) mod 2^64      (wrapping_sub)
//! summary()      =  [(name_i, last_i, average_i, peak_i)] for i in 0..7, in index order
//! reset()        =  state of new()           (idempotent)
//! ```
//!
//! # Degenerate input (what the code does today, pinned so a change is loud)
//!
//! - no record: `average_ticks` and `last_ticks` are `0`, not `None`;
//! - stage index `≥ 7`: `record` is ignored, the getters return `0`, `get`
//!   returns `None`; nothing panics;
//! - `enabled = false`: `record` is ignored;
//! - tick overflow (`u64::MAX` recorded twice): `ProfileEntry::record` uses
//!   plain `+=`, so with debug assertions it panics (`attempt to add with
//!   overflow`) and in release it wraps silently to `u64::MAX − 1`. Neither is
//!   documented; the test pins the behaviour of the profile it runs under and
//!   the finding is reported upstream rather than hidden;
//! - `reset` twice is the same as once.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::profiling::{
    PhysicsProfiler, ProfileEntry, StepStats, TickCounter, STAGE_BROADPHASE, STAGE_CCD,
    STAGE_CONTACT_CACHE, STAGE_INTEGRATION, STAGE_NARROWPHASE, STAGE_SOLVER, STAGE_TOTAL_STEP,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// The seven stages in index order with their names, as the module documents
/// them (index = position in this table).
const STAGE_TABLE: [(usize, &str); 7] = [
    (STAGE_BROADPHASE, "broadphase"),
    (STAGE_NARROWPHASE, "narrowphase"),
    (STAGE_SOLVER, "solver"),
    (STAGE_CCD, "ccd"),
    (STAGE_INTEGRATION, "integration"),
    (STAGE_CONTACT_CACHE, "contact_cache"),
    (STAGE_TOTAL_STEP, "total_step"),
];

/// Independent bookkeeping of what a stage has seen: `(total, count, last, peak)`.
fn fold(ticks: &[u64]) -> (u64, u64, u64, u64) {
    let total: u64 = ticks.iter().sum();
    let count = ticks.len() as u64;
    let last = ticks.last().copied().unwrap_or(0);
    let peak = ticks.iter().copied().max().unwrap_or(0);
    (total, count, last, peak)
}

/// Truncating integer mean, `0` for an empty list.
fn trunc_mean(ticks: &[u64]) -> u64 {
    let (total, count, _, _) = fold(ticks);
    total.checked_div(count).unwrap_or(0)
}

/// `(frame_count, enabled, stats as a tuple, summary)` — every observable
/// field of a profiler, so two profilers can be compared without `PartialEq`.
#[allow(clippy::type_complexity)]
fn observe(
    p: &PhysicsProfiler,
) -> (
    u64,
    bool,
    (u32, u32, u32, u32, u32, u32, u32, u32, u32),
    Vec<(&'static str, u64, u64, u64)>,
) {
    let s: StepStats = p.stats;
    (
        p.frame_count,
        p.enabled,
        (
            s.broadphase_pairs,
            s.narrowphase_tests,
            s.active_contacts,
            s.active_manifolds,
            s.active_bodies,
            s.static_bodies,
            s.solver_iterations,
            s.ccd_checks,
            s.joint_count,
        ),
        p.summary(),
    )
}

// ---------------------------------------------------------------------------
// Stage constants
// ---------------------------------------------------------------------------

/// The seven stage indices are `0..7` without gaps or repeats, and `new()`
/// names them in that order. Both directions: 7 distinct inputs ↔ 7 distinct
/// summary rows.
#[test]
fn stage_indices_are_a_permutation_of_0_to_6_in_table_order() {
    let mut seen = [false; 7];
    for (i, (stage, _)) in STAGE_TABLE.iter().enumerate() {
        assert_eq!(*stage, i, "stage constant {i} out of order");
        assert!(!seen[*stage], "stage {stage} repeated");
        seen[*stage] = true;
    }
    assert!(seen.iter().all(|s| *s));

    let p = PhysicsProfiler::new();
    let names: Vec<&str> = p.summary().iter().map(|r| r.0).collect();
    let expect: Vec<&str> = STAGE_TABLE.iter().map(|r| r.1).collect();
    assert_eq!(names, expect);
}

// ---------------------------------------------------------------------------
// record / average_ticks / last_ticks on ProfileEntry
// ---------------------------------------------------------------------------

/// `7 div 2 = 3`: truncation, not round-half-up (which would be 4).
#[test]
fn entry_average_truncates_7_over_2_to_3() {
    let mut e = ProfileEntry::new("x");
    e.record(3);
    e.record(4);
    assert_eq!(e.total_ticks, 7);
    assert_eq!(e.call_count, 2);
    assert_eq!(
        e.average_ticks(),
        3,
        "truncation gives 3; rounding would give 4"
    );
    assert_ne!(e.average_ticks(), 4);
}

/// For every prefix of a mixed list the entry matches the fold, and the
/// truncation invariant `avg · count ≤ total < (avg + 1) · count` holds.
#[test]
fn entry_matches_independent_fold_on_every_prefix() {
    let ticks = [100u64, 200, 150, 1, 0, 999, 7, 7, 7, 13];
    let mut e = ProfileEntry::new("x");
    for n in 1..=ticks.len() {
        e.record(ticks[n - 1]);
        let (total, count, last, peak) = fold(&ticks[..n]);
        assert_eq!(e.total_ticks, total, "prefix {n} total");
        assert_eq!(e.call_count, count, "prefix {n} count");
        assert_eq!(e.last_ticks, last, "prefix {n} last");
        assert_eq!(e.peak_ticks, peak, "prefix {n} peak");
        let avg = e.average_ticks();
        assert_eq!(avg, trunc_mean(&ticks[..n]), "prefix {n} average");
        assert!(
            avg * count <= total && total < (avg + 1) * count,
            "prefix {n} truncation bound"
        );
    }
}

/// `last_ticks` is the most recent value even when it is smaller than an
/// earlier one (distinguishes it from `peak_ticks`).
#[test]
fn entry_last_is_most_recent_not_max() {
    let mut e = ProfileEntry::new("x");
    e.record(50);
    e.record(5);
    assert_eq!(e.last_ticks, 5);
    assert_eq!(e.peak_ticks, 50);
}

/// No record: average is `0` (documented: `checked_div` by `0` → `0`), last is `0`.
#[test]
fn entry_average_of_nothing_is_zero() {
    let e = ProfileEntry::new("x");
    assert_eq!(e.call_count, 0);
    assert_eq!(e.average_ticks(), 0);
    assert_eq!(e.last_ticks, 0);
}

/// `reset` returns the four counters to `new()`, and a second `reset`
/// changes nothing.
#[test]
fn entry_reset_is_new_and_idempotent() {
    let mut e = ProfileEntry::new("x");
    e.record(9);
    e.record(1);
    e.reset();
    let f = ProfileEntry::new("x");
    let snap = |e: &ProfileEntry| {
        (
            e.name,
            e.total_ticks,
            e.call_count,
            e.last_ticks,
            e.peak_ticks,
        )
    };
    assert_eq!(snap(&e), snap(&f));
    e.reset();
    assert_eq!(snap(&e), snap(&f));
}

/// Tick overflow: `u64::MAX` recorded twice. With debug assertions the `+=`
/// panics; in release it wraps to `u64::MAX − 1`. Pinned per profile so that
/// a change to saturating / checked arithmetic shows up as red here.
#[test]
fn entry_record_overflow_panics_in_debug_and_wraps_in_release() {
    let mut e = ProfileEntry::new("x");
    e.record(u64::MAX);
    let r = catch_unwind(AssertUnwindSafe(|| e.record(u64::MAX)));
    if cfg!(debug_assertions) {
        assert!(
            r.is_err(),
            "debug profile: `total_ticks += ticks` must panic on overflow"
        );
    } else {
        assert!(r.is_ok(), "release profile: `+=` wraps silently");
        assert_eq!(e.total_ticks, u64::MAX.wrapping_add(u64::MAX));
        assert_eq!(e.total_ticks, u64::MAX - 1);
        assert_eq!(e.call_count, 2);
    }
}

// ---------------------------------------------------------------------------
// PhysicsProfiler: record / last_ticks / average_ticks / summary / reset
// ---------------------------------------------------------------------------

/// Each stage is an independent accumulator: recording into stage `i`
/// changes only row `i` of the summary. Both directions: 7 distinct ticks in
/// → 7 distinct averages out, and the untouched rows stay at `0`.
#[test]
fn profiler_stages_are_independent_accumulators() {
    let per_stage: [&[u64]; 7] = [
        &[100, 200, 150],
        &[3, 3, 1],
        &[5, 9, 6],
        &[],
        &[40],
        &[8, 2, 2],
        &[156, 254, 199],
    ];
    let mut p = PhysicsProfiler::new();
    for (stage, ticks) in per_stage.iter().enumerate() {
        for t in *ticks {
            p.record(stage, *t);
        }
        // Stages after this one must still be untouched.
        for later in stage + 1..7 {
            assert_eq!(p.last_ticks(later), 0, "stage {stage} leaked into {later}");
            assert_eq!(
                p.average_ticks(later),
                0,
                "stage {stage} leaked into {later}"
            );
        }
    }
    for (stage, ticks) in per_stage.iter().enumerate() {
        let (_, _, last, peak) = fold(ticks);
        assert_eq!(p.last_ticks(stage), last, "stage {stage} last");
        assert_eq!(
            p.average_ticks(stage),
            trunc_mean(ticks),
            "stage {stage} average"
        );
        let e = p.get(stage).expect("stage in range");
        assert_eq!(e.peak_ticks, peak, "stage {stage} peak");
        assert_eq!(e.call_count, ticks.len() as u64, "stage {stage} count");
    }
    // Truncation where it matters: 7 div 3 = 2 and 20 div 3 = 6.
    assert_eq!(p.average_ticks(STAGE_NARROWPHASE), 2);
    assert_eq!(p.average_ticks(STAGE_SOLVER), 6);
}

/// `summary()` has every stage name exactly once, in index order, with
/// `(last, average, peak)` matching the fold.
#[test]
fn profiler_summary_lists_each_stage_once_with_its_average() {
    let per_stage: [&[u64]; 7] = [
        &[10, 11],
        &[1],
        &[7, 7, 7, 1],
        &[0, 0],
        &[],
        &[2, 1],
        &[31, 29],
    ];
    let mut p = PhysicsProfiler::new();
    for (stage, ticks) in per_stage.iter().enumerate() {
        for t in *ticks {
            p.record(stage, *t);
        }
    }
    let summary = p.summary();
    assert_eq!(summary.len(), 7);
    for (i, (name, last, avg, peak)) in summary.iter().enumerate() {
        let occurrences = summary.iter().filter(|r| r.0 == *name).count();
        assert_eq!(occurrences, 1, "{name} appears {occurrences} times");
        assert_eq!(*name, STAGE_TABLE[i].1, "row {i} name");
        let (_, _, exp_last, exp_peak) = fold(per_stage[i]);
        assert_eq!(*last, exp_last, "row {i} last");
        assert_eq!(*avg, trunc_mean(per_stage[i]), "row {i} average");
        assert_eq!(*peak, exp_peak, "row {i} peak");
    }
}

/// No record anywhere: every getter is `0`, `summary` is seven zero rows.
#[test]
fn profiler_average_and_last_of_nothing_are_zero() {
    let p = PhysicsProfiler::new();
    for stage in 0..7 {
        assert_eq!(p.average_ticks(stage), 0);
        assert_eq!(p.last_ticks(stage), 0);
    }
    for (i, row) in p.summary().iter().enumerate() {
        assert_eq!(*row, (STAGE_TABLE[i].1, 0, 0, 0));
    }
}

/// Stage index out of range (`7`, `usize::MAX`): `record` is ignored (no
/// panic, no row grows), getters return `0`, `get` returns `None`.
#[test]
fn profiler_out_of_range_stage_is_ignored() {
    let mut p = PhysicsProfiler::new();
    let before = observe(&p);
    for stage in [7usize, 8, usize::MAX] {
        let r = catch_unwind(AssertUnwindSafe(|| p.record(stage, 42)));
        assert!(r.is_ok(), "record({stage}) must not panic");
        assert_eq!(p.last_ticks(stage), 0);
        assert_eq!(p.average_ticks(stage), 0);
        assert!(p.get(stage).is_none());
    }
    assert_eq!(
        observe(&p),
        before,
        "out-of-range record must not change any row"
    );
    assert_eq!(p.summary().len(), 7, "no row may be appended");
}

/// `enabled = false` drops records; re-enabling resumes.
#[test]
fn profiler_disabled_drops_records() {
    let mut p = PhysicsProfiler::new();
    p.enabled = false;
    p.record(STAGE_SOLVER, 100);
    assert_eq!(p.last_ticks(STAGE_SOLVER), 0);
    assert_eq!(p.get(STAGE_SOLVER).expect("in range").call_count, 0);
    p.enabled = true;
    p.record(STAGE_SOLVER, 100);
    assert_eq!(p.last_ticks(STAGE_SOLVER), 100);
    assert_eq!(p.average_ticks(STAGE_SOLVER), 100);
}

/// `reset()` returns every observable field (frame_count, stats, all seven
/// rows) to `new()`, keeps `enabled`, and a second reset changes nothing.
#[test]
fn profiler_reset_is_new_and_idempotent() {
    let mut p = PhysicsProfiler::new();
    p.begin_frame();
    p.begin_frame();
    p.stats.active_bodies = 5;
    p.stats.joint_count = 2;
    for stage in 0..7 {
        p.record(stage, 10 + stage as u64);
    }
    assert_ne!(
        observe(&p),
        observe(&PhysicsProfiler::new()),
        "setup must be observable"
    );
    p.reset();
    assert_eq!(observe(&p), observe(&PhysicsProfiler::new()));
    p.reset();
    assert_eq!(observe(&p), observe(&PhysicsProfiler::new()));
}

// ---------------------------------------------------------------------------
// TickCounter: start / elapsed
// ---------------------------------------------------------------------------

/// `elapsed(start) = now − start` exactly, for any sequence of advances;
/// `start()` does not move the counter.
#[test]
fn counter_start_elapsed_is_the_sum_of_advances() {
    let mut c = TickCounter::new();
    let s0 = c.start();
    assert_eq!(s0, 0);
    assert_eq!(c.start(), s0, "start() must not advance");
    let advances = [100u64, 0, 7, 1 << 40, 3];
    let mut sum = 0u64;
    for a in advances {
        c.advance(a);
        sum += a;
        assert_eq!(c.elapsed(s0), sum);
        assert_eq!(c.now(), sum);
    }
}

/// A bracket that straddles the 2⁶⁴ wrap still reads the advance
/// (`wrapping_sub`): start at `MAX − 1`, advance 3 → elapsed 3, now 1.
#[test]
fn counter_elapsed_wraps_modulo_2_pow_64() {
    let mut c = TickCounter::new();
    c.advance(u64::MAX - 1);
    let s = c.start();
    assert_eq!(s, u64::MAX - 1);
    let r = catch_unwind(AssertUnwindSafe(|| {
        c.advance(3);
        (c.now(), c.elapsed(s))
    }));
    let (now, elapsed) = r.expect("advance / elapsed are wrapping, must not panic");
    assert_eq!(now, 1);
    assert_eq!(elapsed, 3);
}

/// The production wiring end to end: bracket a stage with
/// `start` / `elapsed`, feed `record`, read back through `last_ticks` /
/// `average_ticks` / `summary`. The brackets straddle the wrap so a
/// non-wrapping `elapsed` and a dropped `record` are both visible.
#[test]
fn counter_feeds_profiler_through_the_stage_constants() {
    let mut c = TickCounter::new();
    c.advance(u64::MAX - 5);
    let mut p = PhysicsProfiler::new();
    let frames: [[u64; 7]; 2] = [[4, 2, 9, 0, 1, 3, 19], [6, 1, 3, 0, 1, 1, 12]];
    for frame in &frames {
        p.begin_frame();
        for (stage, _) in STAGE_TABLE {
            let s = c.start();
            c.advance(frame[stage]);
            p.record(stage, c.elapsed(s));
        }
    }
    assert_eq!(p.frame_count, 2);
    for (stage, name) in STAGE_TABLE {
        let ticks = [frames[0][stage], frames[1][stage]];
        assert_eq!(p.last_ticks(stage), ticks[1], "{name} last");
        assert_eq!(p.average_ticks(stage), trunc_mean(&ticks), "{name} average");
    }
    // 9 + 3 = 12, 12 div 2 = 6; 19 + 12 = 31, 31 div 2 = 15 (truncated).
    assert_eq!(p.average_ticks(STAGE_SOLVER), 6);
    assert_eq!(p.average_ticks(STAGE_TOTAL_STEP), 15);
    let total_advance: u64 = frames.iter().flatten().sum();
    assert_eq!(c.now(), (u64::MAX - 5).wrapping_add(total_advance));
}
