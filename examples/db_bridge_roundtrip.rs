//! Production entry point for `alice_physics::db_bridge::PhysicsMetricsSink`'s
//! query API (`std,replay` feature -- `wiring_guard`: `src/db_bridge.rs` had
//! zero production callers for `query_bodies`, `query_contacts`,
//! `query_energy`, `record_energy` and `record_step` -- only
//! `#[cfg(test)]` unit tests in the same file called in, which the guard
//! does not count).
//!
//! `PhysicsMetricsSink` is a round-trip persistence API, not a closed-form
//! formula, so the oracle here is "what you put in via `record_step` /
//! `record_energy` is exactly what `query_energy` / `query_bodies` /
//! `query_contacts` give back" -- each of the three metrics lives in its
//! own `AliceDB` instance opened with `FitConfig::lossless = true`
//! (`src/db_bridge.rs`'s own module doc: "metric series must read back
//! exactly"), so every value below is checked for **bit-exact** equality
//! against a hand-written expected tuple, never a value re-derived by
//! calling the sink again.
//!
//! The three per-step values are built from independent closed-form
//! formulas (`energy_at`, `bodies_at`, `contacts_at`) evaluated here, in
//! this file, as plain `f32` arithmetic -- never by reading back a
//! previous query result -- so a mutation that silently swapped which
//! underlying `AliceDB` a `record_*`/`query_*` call touches would be
//! observable.
//!
//! ```bash
//! cargo run --example db_bridge_roundtrip --features std,replay
//! ```

use alice_physics::db_bridge::PhysicsMetricsSink;

const STEPS: i64 = 8; // steps 0..=7 go through record_step

/// Kinetic energy hand-assigned at `step` (not an integer multiple of 1.0,
/// to exercise the lossless round trip on a non-trivial `f32` bit pattern).
fn energy_at(step: i64) -> f32 {
    10.0 + 0.25 * step as f32
}

/// Active body count hand-assigned at `step`.
fn bodies_at(step: i64) -> f32 {
    50.0 - 1.5 * step as f32
}

/// Contact count hand-assigned at `step`.
fn contacts_at(step: i64) -> f32 {
    (step % 4) as f32 * 3.0
}

/// The energy-only step recorded via `record_energy` (never touches
/// `bodies_db` / `contacts_db`).
const ENERGY_ONLY_STEP: i64 = STEPS; // = 8
const ENERGY_ONLY_VALUE: f32 = 99.5;

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");

    // --- boundary: querying before anything has been recorded ---
    {
        let sink = PhysicsMetricsSink::open(dir.path().join("boundary")).expect("open");
        let energy = sink.query_energy(0, 100).expect("query_energy");
        let bodies = sink.query_bodies(0, 100).expect("query_bodies");
        let contacts = sink.query_contacts(0, 100).expect("query_contacts");
        println!(
            "[db_bridge] before any record_step/record_energy: energy={energy:?} bodies={bodies:?} contacts={contacts:?}"
        );
        assert!(energy.is_empty(), "no data has been recorded yet");
        assert!(bodies.is_empty());
        assert!(contacts.is_empty());
        println!("[db_bridge] ok: query_* before any record returns empty, no error");
    }

    // --- record_step for STEPS steps, then one record_energy-only step ---
    let sink = PhysicsMetricsSink::open(dir.path().join("main")).expect("open");
    for step in 0..STEPS {
        sink.record_step(step, energy_at(step), bodies_at(step), contacts_at(step))
            .expect("record_step");
        println!(
            "[db_bridge] record_step({step}): energy={} bodies={} contacts={}",
            energy_at(step),
            bodies_at(step),
            contacts_at(step)
        );
    }
    sink.record_energy(ENERGY_ONLY_STEP, ENERGY_ONLY_VALUE)
        .expect("record_energy");
    println!(
        "[db_bridge] record_energy({ENERGY_ONLY_STEP}): energy={ENERGY_ONLY_VALUE} (bodies_db / contacts_db untouched)"
    );
    sink.flush().expect("flush");

    // --- full-range round trip: energy_db has STEPS+1 rows, bodies_db/contacts_db have STEPS ---
    let energy_full = sink
        .query_energy(0, ENERGY_ONLY_STEP)
        .expect("query_energy full range");
    let mut want_energy_full: Vec<(i64, f32)> = (0..STEPS).map(|s| (s, energy_at(s))).collect();
    want_energy_full.push((ENERGY_ONLY_STEP, ENERGY_ONLY_VALUE));
    println!("[db_bridge] query_energy(0, {ENERGY_ONLY_STEP}) -> {energy_full:?}");
    assert_eq!(
        energy_full, want_energy_full,
        "energy_db must round-trip every record_step value plus the record_energy-only value"
    );
    println!("[db_bridge] ok: query_energy full-range round trip matches exactly");

    let bodies_full = sink
        .query_bodies(0, ENERGY_ONLY_STEP)
        .expect("query_bodies full range");
    let want_bodies_full: Vec<(i64, f32)> = (0..STEPS).map(|s| (s, bodies_at(s))).collect();
    println!("[db_bridge] query_bodies(0, {ENERGY_ONLY_STEP}) -> {bodies_full:?}");
    assert_eq!(
        bodies_full, want_bodies_full,
        "bodies_db must round-trip exactly the record_step values, and must NOT gain a row at \
         the record_energy-only step"
    );
    println!("[db_bridge] ok: query_bodies full-range round trip matches exactly (no phantom row)");

    let contacts_full = sink
        .query_contacts(0, ENERGY_ONLY_STEP)
        .expect("query_contacts full range");
    let want_contacts_full: Vec<(i64, f32)> = (0..STEPS).map(|s| (s, contacts_at(s))).collect();
    println!("[db_bridge] query_contacts(0, {ENERGY_ONLY_STEP}) -> {contacts_full:?}");
    assert_eq!(
        contacts_full, want_contacts_full,
        "contacts_db must round-trip exactly the record_step values, and must NOT gain a row at \
         the record_energy-only step"
    );
    println!(
        "[db_bridge] ok: query_contacts full-range round trip matches exactly (no phantom row)"
    );

    // --- partial-range round trip (inclusive start, inclusive end) ---
    let energy_partial = sink.query_energy(2, 5).expect("query_energy partial");
    let want_energy_partial: Vec<(i64, f32)> = (2..=5).map(|s| (s, energy_at(s))).collect();
    println!("[db_bridge] query_energy(2, 5) -> {energy_partial:?}");
    assert_eq!(energy_partial, want_energy_partial);
    println!("[db_bridge] ok: query_energy partial range [2, 5] is inclusive on both ends");

    // --- boundary: the record_energy-only step has no row in bodies_db / contacts_db ---
    let bodies_at_energy_only_step = sink
        .query_bodies(ENERGY_ONLY_STEP, ENERGY_ONLY_STEP)
        .expect("query_bodies at energy-only step");
    let contacts_at_energy_only_step = sink
        .query_contacts(ENERGY_ONLY_STEP, ENERGY_ONLY_STEP)
        .expect("query_contacts at energy-only step");
    println!(
        "[db_bridge] query_bodies/query_contacts({ENERGY_ONLY_STEP}, {ENERGY_ONLY_STEP}) -> {bodies_at_energy_only_step:?} / {contacts_at_energy_only_step:?}"
    );
    assert!(bodies_at_energy_only_step.is_empty());
    assert!(contacts_at_energy_only_step.is_empty());
    println!("[db_bridge] ok: record_energy writes only energy_db, confirmed by direct query");

    // --- boundary: a range entirely before any recorded step ---
    let before_any = sink
        .query_energy(-100, -1)
        .expect("query_energy before any step");
    println!("[db_bridge] query_energy(-100, -1) -> {before_any:?}");
    assert!(before_any.is_empty(), "no step has a negative index");
    println!("[db_bridge] ok: query_energy range entirely before recorded data is empty");

    sink.flush().expect("final flush");
    println!("[db_bridge] all 5 wiring targets exercised: record_step, record_energy, query_energy, query_bodies, query_contacts");
}
