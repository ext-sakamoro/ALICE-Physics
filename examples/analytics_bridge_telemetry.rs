//! Production entry point for `alice_physics::analytics_bridge::PhysicsTelemetry`
//! (the `analytics` feature's `ALICE-Analytics` profiling bridge).
//!
//! A hand-built 200-step run is fed through every recorder
//! (`record_step_time` / `record_contacts` / `record_energy_drift` /
//! `record_collision_pair`) and every reader is printed next to the closed
//! form it is held to:
//!
//! * `step_time_p50` / `step_time_p99`, `contacts_p50` / `contacts_p99`,
//!   `energy_drift_p99`: nearest-rank order statistics
//!   (`rank = ceil(q * n)`, clamped to `[1, n]`) of this file's own sorted
//!   copy of the fed sequence, read back within the sketch's documented
//!   relative-error contract `[v * (1 - ALPHA), v * (1 + ALPHA)]`.
//! * `total_steps`: exactly the number of `record_step_time` calls (200),
//!   unaffected by the other three recorders.
//! * `unique_collision_pairs`: `record_collision_pair` inserts the caller's
//!   `pair_hash` into a `HyperLogLog12` verbatim — it does **not** canonicalize
//!   `(a, b)` vs `(b, a)` itself (see the module doc of `record_collision_pair`).
//!   This run demonstrates both sides: 37 synthetic pairs recorded through a
//!   canonical `min << 32 | max` hash (and each repeated once, which must not
//!   grow the count), one real body pair recorded canonically from both call
//!   orders (1 distinct observation), and one real body pair recorded with
//!   the *raw*, non-canonicalized `a << 32 | b` form from both call orders
//!   (2 distinct observations — this is the order-sensitivity the doc warns
//!   about). Expected unique count: `37 + 1 + 2 = 40`.
//!
//! ```bash
//! cargo run --example analytics_bridge_telemetry --features std,analytics
//! ```

use alice_physics::analytics_bridge::PhysicsTelemetry;

const ALPHA: f64 = PhysicsTelemetry::ALPHA;

/// `rank = ceil(q * n)` clamped to `[1, n]`, nearest-rank order statistic of
/// `sorted` (already sorted ascending). This is the sketch's own documented
/// rank law (see `PhysicsTelemetry::ALPHA` and the sibling integration test);
/// it is computed here from the fed sequence, never from the sketch.
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    assert!(n > 0, "nearest_rank of an empty sample set is undefined");
    let rank = (q * n as f64).ceil().max(1.0) as usize;
    sorted[rank.min(n) - 1]
}

/// Asserts `actual` lies in the documented `DDSketch256` contract window
/// `[v * (1 - ALPHA), v * (1 + ALPHA)]` around the true order statistic `v`.
fn assert_within_contract(actual: f64, v: f64, what: &str) {
    let lo = v * (1.0 - ALPHA);
    let hi = v * (1.0 + ALPHA);
    assert!(
        actual >= lo && actual <= hi,
        "{what}: {actual} outside [{lo}, {hi}] (v={v}, ALPHA={ALPHA})"
    );
    println!("[analytics_bridge]   {what:<16} = {actual:>10.3}  (closed form {v}, window [{lo:.3}, {hi:.3}])");
}

fn main() {
    println!("[analytics_bridge] PhysicsTelemetry::new() ALPHA={ALPHA}");
    let mut tel = PhysicsTelemetry::new();

    // --- step_time / contacts: 196 steps "nominal" + 4 steps "jank spike" ---
    // --- energy_drift: 197 steps "stable" + 3 steps "divergence spike" ---
    const N: usize = 200;
    let mut step_times = Vec::with_capacity(N);
    let mut contacts = Vec::with_capacity(N);
    let mut drifts = Vec::with_capacity(N);
    for step in 0..N {
        let st = if step < 196 { 1_000.0 } else { 50_000.0 };
        let ct = if step < 196 { 8.0 } else { 120.0 };
        // Fed as negative; the module stores |drift|.
        let dr = if step < 197 { -0.0015 } else { -0.35 };
        tel.record_step_time(st);
        tel.record_contacts(ct);
        tel.record_energy_drift(dr);
        step_times.push(st);
        contacts.push(ct);
        drifts.push(dr.abs());
    }
    step_times.sort_by(f64::total_cmp);
    contacts.sort_by(f64::total_cmp);
    drifts.sort_by(f64::total_cmp);

    assert_eq!(
        tel.total_steps(),
        N as u64,
        "total_steps counts exactly the record_step_time calls"
    );
    println!(
        "[analytics_bridge] total_steps = {} (== {N})",
        tel.total_steps()
    );

    assert_within_contract(
        tel.step_time_p50(),
        nearest_rank(&step_times, 0.5),
        "step_time_p50",
    );
    assert_within_contract(
        tel.step_time_p99(),
        nearest_rank(&step_times, 0.99),
        "step_time_p99",
    );
    assert_within_contract(
        tel.contacts_p50(),
        nearest_rank(&contacts, 0.5),
        "contacts_p50",
    );
    assert_within_contract(
        tel.contacts_p99(),
        nearest_rank(&contacts, 0.99),
        "contacts_p99",
    );
    assert_within_contract(
        tel.energy_drift_p99(),
        nearest_rank(&drifts, 0.99),
        "energy_drift_p99",
    );

    // --- collision pairs: canonical vs. raw (order-sensitivity) ---
    // 37 synthetic pairs, canonical form, each repeated once (idempotent-ish
    // on the HLL's bucket maxima, must not raise the unique count).
    for i in 0u64..37 {
        let (a, b) = (i, i + 1);
        let canonical = a.min(b) << 32 | a.max(b);
        tel.record_collision_pair(canonical);
        tel.record_collision_pair(canonical);
    }
    // One real body pair, canonical hash computed the same way from both
    // call orders -> the two calls collapse to one observation.
    let (body_a, body_b) = (12u64, 1_000u64);
    tel.record_collision_pair(body_a.min(body_b) << 32 | body_a.max(body_b));
    tel.record_collision_pair(body_b.min(body_a) << 32 | body_b.max(body_a));
    // One real body pair, *raw* (a << 32 | b) hash with no normalization ->
    // the two call orders are different u64s, so they are two observations.
    let (body_c, body_d) = (777u64, 2_001u64);
    tel.record_collision_pair(body_c << 32 | body_d);
    tel.record_collision_pair(body_d << 32 | body_c);

    let pairs = tel.unique_collision_pairs();
    println!(
        "[analytics_bridge] unique_collision_pairs = {pairs:.3} (expected ~= 40, see module doc)"
    );
    assert!(
        (38.5..=41.5).contains(&pairs),
        "unique_collision_pairs {pairs} outside the HyperLogLog12 window around 40"
    );

    println!("[analytics_bridge] done");
}
