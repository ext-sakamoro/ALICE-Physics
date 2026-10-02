//! Per-stage tick profiling with the deterministic counter
//!
//! `PhysicsWorld::step` does not feed a profiler (there is no hook in the
//! solver), so the production reading of `profiling` is a caller that brackets
//! its own stages with `TickCounter::start` / `elapsed` and records the ticks
//! into a `PhysicsProfiler` under the `STAGE_*` indices. This example runs
//! three frames with hand-fed tick counts, prints each stage next to the
//! closed form it must satisfy, and refuses to exit 0 if either differs.
//!
//! Closed forms (all on `u64`, no floating point):
//!
//! - `elapsed(start)` is `now − start` modulo 2⁶⁴ (`wrapping_sub`), so a
//!   bracket that straddles the counter wrap still reads the advance;
//! - `average_ticks` is `total div count` with the remainder dropped
//!   (truncation; `7 div 2 = 3`, never 4), and `0` before any record;
//! - `last_ticks` is the value of the most recent `record` only;
//! - `summary()` lists the seven stages once each, in index order, as
//!   `(name, last, average, peak)`;
//! - `reset()` returns every counter to the `new()` state, and a second
//!   `reset()` changes nothing.
//!
//! ```bash
//! cargo run --example profiling_stages --features std
//! ```

use alice_physics::profiling::{
    PhysicsProfiler, ProfileEntry, TickCounter, STAGE_BROADPHASE, STAGE_CCD, STAGE_CONTACT_CACHE,
    STAGE_INTEGRATION, STAGE_NARROWPHASE, STAGE_SOLVER, STAGE_TOTAL_STEP,
};

/// Stage index, name and the ticks each of the three frames spends in it.
///
/// The totals are chosen so that `total div 3` has a non-zero remainder for
/// the narrowphase and the solver (`7 div 3 = 2`, `20 div 3 = 6`), which is
/// where truncation and rounding give different answers.
const STAGES: [(usize, &str, [u64; 3]); 7] = [
    (STAGE_BROADPHASE, "broadphase", [100, 200, 150]),
    (STAGE_NARROWPHASE, "narrowphase", [3, 3, 1]),
    (STAGE_SOLVER, "solver", [5, 9, 6]),
    (STAGE_CCD, "ccd", [0, 0, 0]),
    (STAGE_INTEGRATION, "integration", [40, 40, 40]),
    (STAGE_CONTACT_CACHE, "contact_cache", [8, 2, 2]),
    (STAGE_TOTAL_STEP, "total_step", [156, 254, 199]),
];

/// Closed form of `average_ticks`: the truncating integer mean.
const fn closed_form_average(ticks: &[u64]) -> u64 {
    let mut total = 0u64;
    let mut i = 0;
    while i < ticks.len() {
        total += ticks[i];
        i += 1;
    }
    if ticks.is_empty() {
        0
    } else {
        total / ticks.len() as u64
    }
}

fn main() {
    let mut profiler = PhysicsProfiler::new();
    // Start the counter two ticks below the wrap so the first frame's
    // brackets cross 2⁶⁴ − 1 → 0 and `elapsed` has to use modular arithmetic.
    let mut counter = TickCounter::new();
    counter.advance(u64::MAX - 1);

    for frame in 0..3 {
        profiler.begin_frame();
        for (stage, name, ticks) in &STAGES {
            let start = counter.start();
            counter.advance(ticks[frame]);
            let elapsed = counter.elapsed(start);
            assert_eq!(
                elapsed, ticks[frame],
                "[profiling] frame {frame} {name}: elapsed {elapsed} != advance {}",
                ticks[frame]
            );
            profiler.record(*stage, elapsed);
            println!(
                "[profiling] frame {frame} stage {stage} {name:<14} elapsed {elapsed:>4} (counter now {})",
                counter.now()
            );
        }
    }

    println!(
        "[profiling] after 3 frames (frame_count {}):",
        profiler.frame_count
    );
    assert_eq!(
        profiler.frame_count, 3,
        "[profiling] begin_frame must count the frames"
    );
    println!(
        "[profiling] {:<14} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "stage", "last", "avg", "peak", "avg*", "last*", "peak*"
    );
    for (stage, name, ticks) in &STAGES {
        let expect_avg = closed_form_average(ticks);
        let expect_last = ticks[2];
        let expect_peak = ticks[0].max(ticks[1]).max(ticks[2]);
        let got_last = profiler.last_ticks(*stage);
        let got_avg = profiler.average_ticks(*stage);
        let got_peak = profiler
            .get(*stage)
            .map_or(0, |e: &ProfileEntry| e.peak_ticks);
        println!(
            "[profiling] {name:<14} {got_last:>6} {got_avg:>6} {got_peak:>6} {expect_avg:>6} {expect_last:>6} {expect_peak:>6}"
        );
        assert_eq!(
            got_avg, expect_avg,
            "[profiling] {name}: average != truncating mean"
        );
        assert_eq!(
            got_last, expect_last,
            "[profiling] {name}: last != last record"
        );
        assert_eq!(
            got_peak, expect_peak,
            "[profiling] {name}: peak != max of records"
        );
    }

    let summary = profiler.summary();
    println!("[profiling] summary rows {}", summary.len());
    for (i, (name, last, avg, peak)) in summary.iter().enumerate() {
        println!("[profiling] summary[{i}] {name:<14} last {last:>4} avg {avg:>4} peak {peak:>4}");
        let (stage, expect_name, ticks) = &STAGES[i];
        assert_eq!(
            *stage, i,
            "[profiling] summary row {i} is not stage {stage}"
        );
        assert_eq!(name, expect_name, "[profiling] summary row {i} name");
        assert_eq!(
            *avg,
            closed_form_average(ticks),
            "[profiling] summary row {i} avg"
        );
    }
    assert_eq!(
        summary.len(),
        STAGES.len(),
        "[profiling] summary must list each stage once"
    );

    // A standalone entry: `ProfileEntry` is the same arithmetic without the
    // stage table, used when a caller times a single custom stage.
    let mut custom = ProfileEntry::new("custom");
    custom.record(3);
    custom.record(4);
    println!(
        "[profiling] custom entry total {} count {} avg {} (closed form 7 div 2 = 3)",
        custom.total_ticks,
        custom.call_count,
        custom.average_ticks()
    );
    assert_eq!(
        (custom.total_ticks, custom.call_count),
        (7, 2),
        "[profiling] custom: two records must sum to 7"
    );
    assert_eq!(
        custom.average_ticks(),
        3,
        "[profiling] custom: 7 div 2 must truncate to 3"
    );
    custom.reset();
    assert_eq!(
        (
            custom.total_ticks,
            custom.call_count,
            custom.last_ticks,
            custom.peak_ticks
        ),
        (0, 0, 0, 0),
        "[profiling] custom: reset must clear every counter"
    );

    profiler.reset();
    let fresh = PhysicsProfiler::new();
    println!(
        "[profiling] after reset: frame_count {} summary == new(): {}",
        profiler.frame_count,
        profiler.summary() == fresh.summary()
    );
    assert_eq!(
        profiler.summary(),
        fresh.summary(),
        "[profiling] reset must match new()"
    );
    assert_eq!(
        profiler.frame_count, 0,
        "[profiling] reset must zero frame_count"
    );
    println!("[profiling] ok");
}
