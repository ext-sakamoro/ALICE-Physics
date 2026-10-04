//! Audit oracles for `alice_physics::db_bridge::PhysicsMetricsSink`.
//!
//! Expected values are constructed from closed-form step formulas that are
//! never read back from the sink under test.

#![cfg(all(feature = "std", feature = "replay"))]

use alice_physics::db_bridge::PhysicsMetricsSink;

fn e(step: i64) -> f32 {
    0.25 * step as f32 + 1.0
}
fn b(step: i64) -> f32 {
    50.0 - 2.0 * step as f32
}
fn c(step: i64) -> f32 {
    ((step * 7) % 11) as f32 * 0.5
}

/// `open` creates missing parent directories and one sub-database per
/// metric under the given directory.
#[test]
fn open_creates_nested_directory_with_three_metric_databases() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("a").join("b").join("metrics");
    assert!(!nested.exists());
    let sink = PhysicsMetricsSink::open(&nested).unwrap();
    sink.record_step(0, 1.0, 2.0, 3.0).unwrap();
    sink.flush().unwrap();
    assert!(nested.is_dir());
    for name in ["energy", "bodies", "contacts"] {
        assert!(nested.join(name).exists(), "missing sub-database {name}");
    }
}

/// Data flushed by one sink is readable by a second sink opened on the same
/// directory (persistence for replay / debugging).
#[test]
fn flushed_metrics_survive_reopen() {
    let dir = tempfile::tempdir().unwrap();
    {
        let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
        for s in 0..8i64 {
            sink.record_step(s, e(s), b(s), c(s)).unwrap();
        }
        sink.flush().unwrap();
    }
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    let want_e: Vec<(i64, f32)> = (0..8).map(|s| (s, e(s))).collect();
    let want_b: Vec<(i64, f32)> = (0..8).map(|s| (s, b(s))).collect();
    let want_c: Vec<(i64, f32)> = (0..8).map(|s| (s, c(s))).collect();
    assert_eq!(sink.query_energy(0, 7).unwrap(), want_e);
    assert_eq!(sink.query_bodies(0, 7).unwrap(), want_b);
    assert_eq!(sink.query_contacts(0, 7).unwrap(), want_c);
}

/// Each argument of `record_step` lands in its own series, in argument
/// order (kinetic energy, body count, contact count), with distinct values.
#[test]
fn record_step_argument_order_maps_to_series() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    for s in 0..4i64 {
        sink.record_step(s, 1000.0 + s as f32, 2000.0 + s as f32, 3000.0 + s as f32)
            .unwrap();
    }
    sink.flush().unwrap();
    assert_eq!(sink.query_energy(2, 2).unwrap(), vec![(2, 1002.0)]);
    assert_eq!(sink.query_bodies(2, 2).unwrap(), vec![(2, 2002.0)]);
    assert_eq!(sink.query_contacts(2, 2).unwrap(), vec![(2, 3002.0)]);
}

/// A series starting at a non-zero step reads back at its real step indices.
#[test]
fn series_starting_at_nonzero_step_keeps_step_indices() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    for s in 100..106i64 {
        sink.record_step(s, e(s), b(s), c(s)).unwrap();
    }
    sink.flush().unwrap();
    let want: Vec<(i64, f32)> = (100..106).map(|s| (s, e(s))).collect();
    assert_eq!(sink.query_energy(0, 1000).unwrap(), want);
    let want_b: Vec<(i64, f32)> = (101..=103).map(|s| (s, b(s))).collect();
    assert_eq!(sink.query_bodies(101, 103).unwrap(), want_b);
}

/// Non-finite metric values are stored losslessly (a NaN energy is a real
/// diagnostic signal and must not be replaced by a number).
#[test]
fn non_finite_energy_round_trips_bitwise() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    let vals = [1.0_f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0];
    for (i, v) in vals.iter().enumerate() {
        sink.record_energy(i as i64, *v).unwrap();
    }
    sink.flush().unwrap();
    let got = sink.query_energy(0, 4).unwrap();
    assert_eq!(got.len(), vals.len());
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(got[i].0, i as i64);
        assert_eq!(got[i].1.to_bits(), v.to_bits(), "step {i}");
    }
}

/// Negative step indices are valid i64 timestamps and round-trip.
#[test]
fn negative_steps_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    for s in -3..3i64 {
        sink.record_step(s, e(s), b(s), c(s)).unwrap();
    }
    sink.flush().unwrap();
    let want: Vec<(i64, f32)> = (-3..3).map(|s| (s, c(s))).collect();
    assert_eq!(sink.query_contacts(-3, 2).unwrap(), want);
}

/// A single-record series reads back (segment with one point).
#[test]
fn single_record_series_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    sink.record_step(5, 9.5, 4.0, 1.0).unwrap();
    sink.flush().unwrap();
    assert_eq!(sink.query_energy(5, 5).unwrap(), vec![(5, 9.5)]);
    assert_eq!(sink.query_bodies(0, 10).unwrap(), vec![(5, 4.0)]);
    assert_eq!(sink.query_contacts(5, 6).unwrap(), vec![(5, 1.0)]);
}

/// Flush is idempotent and a second flush with no new data changes nothing.
#[test]
fn double_flush_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    for s in 0..5i64 {
        sink.record_step(s, e(s), b(s), c(s)).unwrap();
    }
    sink.flush().unwrap();
    sink.flush().unwrap();
    let want: Vec<(i64, f32)> = (0..5).map(|s| (s, e(s))).collect();
    assert_eq!(sink.query_energy(0, 4).unwrap(), want);
}

/// Uniformly strided steps (every 10th step) read back at their real step
/// indices.
#[test]
fn uniformly_strided_steps_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    for k in 0..6i64 {
        sink.record_step(k * 10, e(k), b(k), c(k)).unwrap();
    }
    sink.flush().unwrap();
    let want: Vec<(i64, f32)> = (0..6).map(|k| (k * 10, e(k))).collect();
    assert_eq!(sink.query_energy(0, 50).unwrap(), want);
}

/// A series with a gap returns exactly the recorded `(step, value)` pairs
/// and nothing for the unrecorded steps.
#[test]
#[ignore = "known defect: AUD-A-S5W1-001: non-contiguous steps are re-spaced uniformly by the storage layer; a query for an unrecorded step is non-empty and recorded steps come back at wrong indices, with no error; root: external alice-db 0.2.0-beta.3 (Segment::query_range rebuilds timestamps at a uniform step from start_time / end_time / point_count)"]
fn gapped_series_returns_exactly_the_recorded_pairs() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    let steps: Vec<i64> = (0..5).chain(10..15).collect();
    for &s in &steps {
        sink.record_step(s, e(s), b(s), c(s)).unwrap();
    }
    sink.flush().unwrap();
    let want: Vec<(i64, f32)> = steps.iter().map(|&s| (s, e(s))).collect();
    assert_eq!(sink.query_energy(0, 14).unwrap(), want);
    assert!(sink.query_energy(7, 7).unwrap().is_empty());
}

/// A smooth trend with small pseudo-random jitter reads back bit-exactly.
/// A fitted polynomial alone would reproduce it only approximately, so this
/// pins the lossless setting.
#[test]
fn smooth_trend_with_jitter_round_trips_bit_exact() {
    let dir = tempfile::tempdir().unwrap();
    let sink = PhysicsMetricsSink::open(dir.path()).unwrap();
    let mut state: u32 = 0x1234_5678;
    let mut vals = Vec::new();
    for s in 0..64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let jitter = ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 0.002;
        vals.push(100.0 + 0.5 * s as f32 + 0.01 * (s * s) as f32 + jitter);
    }
    for (i, v) in vals.iter().enumerate() {
        sink.record_step(i as i64, *v, *v * 2.0, *v + 1.0).unwrap();
    }
    sink.flush().unwrap();
    let got = sink.query_energy(0, 63).unwrap();
    assert_eq!(got.len(), 64);
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(got[i].1.to_bits(), v.to_bits(), "energy step {i}");
    }
    let got_b = sink.query_bodies(0, 63).unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert_eq!(
            got_b[i].1.to_bits(),
            (*v * 2.0).to_bits(),
            "bodies step {i}"
        );
    }
}
