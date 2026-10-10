//! Round-trip and degenerate-input oracles for the wiring of
//! `alice_physics::db_bridge::PhysicsMetricsSink`
//! (`examples/db_bridge_roundtrip.rs`): `record_step`, `record_energy`,
//! `query_energy`, `query_bodies`, `query_contacts`.
//!
//! `PhysicsMetricsSink` is a persistence bridge, not a closed-form formula:
//! each of its three `AliceDB` instances (`energy_db` / `bodies_db` /
//! `contacts_db`) is opened with `FitConfig { lossless: true, .. }`
//! (`src/db_bridge.rs`'s own module doc: "metric series must read back
//! exactly"), so the oracle throughout this file is **bit-exact equality**
//! between what was passed to `record_step`/`record_energy` and what comes
//! back from `query_energy`/`query_bodies`/`query_contacts` -- never a
//! value re-derived by calling the sink a second time.
//!
//! There is no `aggregate`/`downsample` call anywhere in `src/db_bridge.rs`
//! -- `query_energy`/`query_bodies`/`query_contacts` are each a direct
//! `self.<metric>_db.scan(start, end)`, i.e. a raw range query that hands
//! back every recorded `(step, value)` pair unmodified. There is therefore
//! no sum/average semantics to independently verify here; the tests below
//! instead cover the round trip itself, channel isolation (`record_energy`
//! must not touch `bodies_db`/`contacts_db`), and the boundary behavior of
//! the underlying `AliceDB::scan` range query (confirmed directly against
//! `alice-db` 0.2.0-beta.3's `StorageEngine::query_range` /
//! `DataSegment::query_range`, which clamp `[start, end]` to
//! `[self.start_time, self.end_time]` and return an empty `Vec` whenever
//! the clamped `query_start > query_end` -- covering both a plain reversed
//! range and a range entirely outside what was ever recorded).
//!
//! Unlike `alice_physics::replay` (`ReplayRecorder`/`ReplayPlayer`), there
//! is no per-body `id` parameter here -- each metric is a single scalar
//! time series -- so "querying a non-existent id" below means querying a
//! step that was never written to that particular series (in particular,
//! a step recorded only via `record_energy`, which `bodies_db`/
//! `contacts_db` never see).

#![cfg(all(feature = "std", feature = "replay"))]

use alice_physics::db_bridge::PhysicsMetricsSink;

/// Independent closed-form energy formula (never read back from a query).
fn energy_at(step: i64) -> f32 {
    3.0 + 0.5 * step as f32
}

/// Independent closed-form body-count formula.
fn bodies_at(step: i64) -> f32 {
    20.0 - step as f32
}

/// Independent closed-form contact-count formula.
fn contacts_at(step: i64) -> f32 {
    (step % 3) as f32 + 0.5
}

fn open_sink(dir: &tempfile::TempDir, name: &str) -> PhysicsMetricsSink {
    PhysicsMetricsSink::open(dir.path().join(name)).expect("PhysicsMetricsSink::open")
}

// ---------------------------------------------------------------------------
// Round-trip correctness
// ---------------------------------------------------------------------------

#[test]
fn record_step_round_trips_exactly_through_all_three_series() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "roundtrip");

    for step in 0..12i64 {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    sink.flush().unwrap();

    let want_energy: Vec<(i64, f32)> = (0..12).map(|s| (s, energy_at(s))).collect();
    let want_bodies: Vec<(i64, f32)> = (0..12).map(|s| (s, bodies_at(s))).collect();
    let want_contacts: Vec<(i64, f32)> = (0..12).map(|s| (s, contacts_at(s))).collect();

    assert_eq!(sink.query_energy(0, 11).unwrap(), want_energy);
    assert_eq!(sink.query_bodies(0, 11).unwrap(), want_bodies);
    assert_eq!(sink.query_contacts(0, 11).unwrap(), want_contacts);
}

#[test]
fn record_energy_round_trips_exactly_and_touches_only_energy_db() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "energy_only");

    sink.record_energy(0, 42.25).unwrap();
    sink.record_energy(1, -17.5).unwrap();
    sink.flush().unwrap();

    assert_eq!(
        sink.query_energy(0, 1).unwrap(),
        vec![(0, 42.25), (1, -17.5)]
    );
    assert!(
        sink.query_bodies(0, 1).unwrap().is_empty(),
        "record_energy must never write to bodies_db"
    );
    assert!(
        sink.query_contacts(0, 1).unwrap().is_empty(),
        "record_energy must never write to contacts_db"
    );
}

#[test]
fn partial_range_query_is_inclusive_on_both_ends() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "partial");

    for step in 0..10i64 {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    sink.flush().unwrap();

    // [3, 7] must include both endpoints: steps 3,4,5,6,7 -> 5 entries.
    let got = sink.query_energy(3, 7).unwrap();
    let want: Vec<(i64, f32)> = (3..=7).map(|s| (s, energy_at(s))).collect();
    assert_eq!(got.len(), 5);
    assert_eq!(got, want);

    // single-point range (start == end) returns exactly that one row.
    assert_eq!(sink.query_bodies(5, 5).unwrap(), vec![(5, bodies_at(5))]);
}

// ---------------------------------------------------------------------------
// Empty-query behavior
// ---------------------------------------------------------------------------

#[test]
fn querying_before_any_record_call_returns_empty_for_all_three_series() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "nothing_recorded");

    assert!(sink.query_energy(0, 1000).unwrap().is_empty());
    assert!(sink.query_bodies(0, 1000).unwrap().is_empty());
    assert!(sink.query_contacts(0, 1000).unwrap().is_empty());
}

#[test]
fn a_write_is_visible_before_flush() {
    // the sink writes exact series records, which reads see from the moment
    // they are written (the time-series store the sink used up to 2.2.0 only
    // read persisted segments, and this test pinned that)
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "unflushed");

    sink.record_step(0, 1.0, 2.0, 3.0).unwrap();
    assert_eq!(sink.query_energy(0, 0).unwrap(), vec![(0, 1.0)]);
    assert_eq!(sink.query_bodies(0, 0).unwrap(), vec![(0, 2.0)]);
    assert_eq!(sink.query_contacts(0, 0).unwrap(), vec![(0, 3.0)]);

    sink.flush().unwrap();
    assert_eq!(sink.query_energy(0, 0).unwrap(), vec![(0, 1.0)]);
}

// ---------------------------------------------------------------------------
// Boundary cases
// ---------------------------------------------------------------------------

#[test]
fn querying_a_step_recorded_only_in_energy_db_is_absent_from_the_other_two() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "mixed");

    for step in 0..5i64 {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    // step 5 exists only in energy_db.
    sink.record_energy(5, 123.0).unwrap();
    sink.flush().unwrap();

    assert_eq!(sink.query_energy(5, 5).unwrap(), vec![(5, 123.0)]);
    assert!(
        sink.query_bodies(5, 5).unwrap().is_empty(),
        "step 5 was never passed to record_step, so bodies_db has no row for it"
    );
    assert!(
        sink.query_contacts(5, 5).unwrap().is_empty(),
        "step 5 was never passed to record_step, so contacts_db has no row for it"
    );

    // the energy-only step must not leak into a wider bodies_db/contacts_db
    // range query either.
    assert_eq!(sink.query_bodies(0, 5).unwrap().len(), 5);
    assert_eq!(sink.query_contacts(0, 5).unwrap().len(), 5);
}

#[test]
fn reversed_range_start_after_end_is_empty_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "reversed");

    for step in 0..5i64 {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    sink.flush().unwrap();

    // sanity: the forward range is non-empty.
    assert_eq!(sink.query_energy(0, 4).unwrap().len(), 5);

    assert!(sink.query_energy(4, 0).unwrap().is_empty());
    assert!(sink.query_bodies(3, 1).unwrap().is_empty());
    assert!(sink.query_contacts(4, 2).unwrap().is_empty());
}

#[test]
fn range_entirely_outside_recorded_steps_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "out_of_range");

    for step in 10..15i64 {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    sink.flush().unwrap();

    // before the recorded window
    assert!(sink.query_energy(-100, -1).unwrap().is_empty());
    assert!(sink.query_bodies(0, 9).unwrap().is_empty());
    // after the recorded window
    assert!(sink.query_contacts(15, 1000).unwrap().is_empty());
    // sanity: the recorded window itself is not empty
    assert_eq!(sink.query_energy(10, 14).unwrap().len(), 5);
}

/// A series with a gap (steps `0..5`, then `10..15`) reads back exactly the
/// recorded steps: a step in the gap has no value, and a scan over the whole
/// span returns the ten recorded steps with their values bit for bit. (Up to
/// 2.2.0 the sink used a fitted time-series store that assumed evenly spaced
/// steps, and this test pinned the wrong answers it gave for a gap.)
#[test]
fn a_series_with_a_gap_reads_back_exactly_the_recorded_steps() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "gap");

    let recorded: Vec<i64> = (0..5i64).chain(10..15i64).collect();
    for &step in &recorded {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .unwrap();
    }
    sink.flush().unwrap();

    assert!(
        sink.query_energy(7, 7).unwrap().is_empty(),
        "step 7 was never recorded"
    );
    assert!(
        sink.query_energy(5, 9).unwrap().is_empty(),
        "nothing in the gap"
    );

    let bits = |v: Vec<(i64, f32)>| {
        v.into_iter()
            .map(|(s, x)| (s, x.to_bits()))
            .collect::<Vec<_>>()
    };
    let want = |f: fn(i64) -> f32| {
        recorded
            .iter()
            .map(|&s| (s, f(s).to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(sink.query_energy(0, 14).unwrap()), want(energy_at));
    assert_eq!(bits(sink.query_bodies(0, 14).unwrap()), want(bodies_at));
    assert_eq!(bits(sink.query_contacts(0, 14).unwrap()), want(contacts_at));
}

// ---------------------------------------------------------------------------
// Lossless round trip of non-trivial f32 bit patterns
// ---------------------------------------------------------------------------

#[test]
fn extreme_and_fractional_f32_values_round_trip_bit_exact() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "extreme");

    let energy_values = [f32::MAX, f32::MIN, f32::MIN_POSITIVE, -0.0, 0.1_f32];
    for (step, &value) in energy_values.iter().enumerate() {
        sink.record_energy(step as i64, value).unwrap();
    }
    sink.flush().unwrap();

    let got = sink
        .query_energy(0, (energy_values.len() - 1) as i64)
        .unwrap();
    assert_eq!(got.len(), energy_values.len());
    for (step, &want) in energy_values.iter().enumerate() {
        let (got_step, got_value) = got[step];
        assert_eq!(got_step, step as i64);
        assert_eq!(
            got_value.to_bits(),
            want.to_bits(),
            "step {step}: lossless storage must preserve the exact bit pattern of {want} \
             (including the sign of zero), not just an approximately-equal value"
        );
    }
}

// ---------------------------------------------------------------------------
// Sparse steps: every key reads back its own value (point and range)
// ---------------------------------------------------------------------------

/// Steps far apart (a long run, a sparse sampling schedule) must each read
/// back the value written at that step, bit for bit, from a point query and
/// from one range query over all of them. The tests above only write
/// contiguous steps `0..n`, where this holds.
#[test]
fn sparse_steps_read_back_their_own_values_bit_exact() {
    let dir = tempfile::tempdir().unwrap();
    let sink = open_sink(&dir, "sparse");

    let steps: [i64; 6] = [0, 1, 1_000, 1_000_003, 1 << 40, (1 << 53) - 1];
    let value = |i: usize| 0.1_f32 + 1.7 * i as f32;
    for (i, &step) in steps.iter().enumerate() {
        sink.record_energy(step, value(i)).unwrap();
    }
    sink.flush().unwrap();

    for (i, &step) in steps.iter().enumerate() {
        let got = sink.query_energy(step, step).unwrap();
        assert_eq!(got.len(), 1, "step {step}: one row, got {got:?}");
        assert_eq!(got[0].0, step);
        assert_eq!(
            got[0].1.to_bits(),
            value(i).to_bits(),
            "step {step}: wrote {}, read {}",
            value(i),
            got[0].1
        );
    }
    let all = sink.query_energy(0, steps[steps.len() - 1]).unwrap();
    let want: Vec<(i64, u32)> = steps
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, value(i).to_bits()))
        .collect();
    let got: Vec<(i64, u32)> = all.iter().map(|&(s, v)| (s, v.to_bits())).collect();
    assert_eq!(
        got, want,
        "one range over every sparse step returns exactly those rows"
    );
}
