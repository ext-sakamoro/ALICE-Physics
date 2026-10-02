//! Oracles for the wiring of `analytics_bridge`
//! (`examples/analytics_bridge_telemetry.rs`): `PhysicsTelemetry::{new,
//! record_step_time, record_contacts, record_energy_drift,
//! record_collision_pair, step_time_p50, step_time_p99, contacts_p50,
//! contacts_p99, energy_drift_p99, unique_collision_pairs, total_steps}`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * **Percentiles** (`DDSketch256`, `alice-analytics` 0.1.1,
//!   `src/sketch.rs`): `quantile(q)` picks the `rank = max(1, ceil(q *
//!   count))`-th order statistic (nearest-rank, 1-indexed) and returns a
//!   value within `v * (1 - ALPHA) ..= v * (1 + ALPHA)` of it (`ALPHA =
//!   PhysicsTelemetry::ALPHA = 0.05`; the bucket representative `2 *
//!   gamma^i / (gamma + 1)` sits at exactly that relative distance from both
//!   bucket edges — the published `DDSketch` guarantee). This test sorts its
//!   own fed sequence and computes the rank independently of the sketch;
//!   it never derives the expected value by calling `quantile` itself.
//!   `record_energy_drift` stores `drift.abs()` (checked with a mixed-sign
//!   sequence).
//! * **`total_steps`**: a plain counter incremented only by
//!   `record_step_time`; `record_contacts` / `record_energy_drift` /
//!   `record_collision_pair` must not touch it.
//! * **`unique_collision_pairs`** (`HyperLogLog12`, `p = 12`, `m = 4096`):
//!   `record_collision_pair(pair_hash)` calls `insert_hash(pair_hash)`
//!   *verbatim* — no FNV mixing, no canonicalization of `(a, b)` vs.
//!   `(b, a)`. `insert_hash` sets register `j = hash & 4095` to
//!   `max(current, rho)` where `rho` is `1 +` the count of leading zero bits
//!   of `hash >> 12` (or `64 - 12 + 1` when that shift is `0`). Cardinality
//!   is the HyperLogLog++ estimator with the documented small-range switch
//!   to linear counting (`m * ln(m / zeros)`) when the raw estimate is
//!   `<= 2.5 * m` and at least one register is still `0` — exactly the
//!   regime every scene below sits in. This test rebuilds the 4096-register
//!   array independently (`ref_registers`) from the literal `pair_hash`
//!   values it feeds the module, the same way
//!   `tests/analytic_sketch_wiring.rs` holds `alice_physics::sketch`'s own
//!   `HyperLogLog` to its reference registers, and evaluates the published
//!   formula with `alice_physics::det_math::ln64` (never `f64::ln`, and
//!   never the module's own `unique_collision_pairs()` output).
//!
//! # Degenerate input (each result is pinned, panics are measured)
//!
//! A freshly-constructed `PhysicsTelemetry` reads back `total_steps() == 0`,
//! every percentile `== 0.0` (the sketch's documented empty-count early
//! return) and `unique_collision_pairs() == 0.0` (`m * ln(m / m) = 0`
//! exactly). A single recording makes `p50 == p99` (both ranks round to the
//! same, only, sample). An all-identical sequence collapses every quantile
//! to that value within `ALPHA`. Extreme magnitudes: `record_step_time`
//! with `f64::INFINITY` — `value.ln()` is `+inf`, `* inv_ln_gamma` is `+inf`,
//! `.ceil()` is `+inf`, and the crate's `f64_i32` cast saturates that to
//! `i32::MAX`; adding the sketch's bucket `offset` (`64` for the 256-bin
//! sketch) then overflows `i32` — measured: panics in a debug build, wraps
//! in release (`catch_unwind`, matching the precedent in
//! `tests/analytic_sketch_wiring.rs::ddsketch_degenerate_inputs`). A finite
//! but astronomically large magnitude (`1e300`) does not panic: its bucket
//! index is clamped to the sketch's edge bin
//! (`idx.min(BINS - 1)`), so it is retained (not dropped) and `quantile`
//! returns a finite edge-bin value, not `max` — this is the documented
//! 1.2.0 fix referenced in `PhysicsTelemetry::new`'s doc comment (pre-1.2.0
//! `alpha = 0.01` values above the bucket range were silently uncounted and
//! every quantile degraded to the running maximum; post-fix they are
//! retained in the edge bin). `NaN` compares false to both `> 0.0` and
//! `< 0.0`, so `record_step_time(f64::NAN)` lands in the zero bucket, not a
//! panic.
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "std", feature = "analytics"))]

use alice_physics::analytics_bridge::PhysicsTelemetry;
use alice_physics::det_math::ln64;
use std::panic::{catch_unwind, AssertUnwindSafe};

const ALPHA: f64 = PhysicsTelemetry::ALPHA;

// ---------------------------------------------------------------------------
// Reference implementations (never call the crate)
// ---------------------------------------------------------------------------

/// Nearest-rank order statistic of an *already sorted* slice:
/// `rank = max(1, ceil(q * n))`, 1-indexed, clamped to `n`.
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    assert!(n > 0, "nearest_rank of an empty sample set is undefined");
    let rank = ((q * n as f64).ceil().max(1.0) as usize).min(n);
    sorted[rank - 1]
}

/// `DDSketch256`'s documented relative-error contract around the true order
/// statistic `v`: `[v * (1 - ALPHA), v * (1 + ALPHA)]`.
fn assert_within_contract(actual: f64, v: f64, what: &str) {
    let lo = v * (1.0 - ALPHA);
    let hi = v * (1.0 + ALPHA);
    assert!(
        actual >= lo && actual <= hi,
        "{what} = {actual}, expected in [{lo}, {hi}] (v={v}, ALPHA={ALPHA})"
    );
}

/// Reference `HyperLogLog12` (`p = 12`, `m = 4096`) register array, built
/// from the *literal* `pair_hash` values fed to `record_collision_pair` —
/// `insert_hash` performs no mixing, so this is exact, not approximate.
fn ref_registers(hashes: &[u64]) -> [u8; 4096] {
    const P: u32 = 12;
    const M: u64 = 4096;
    let mut regs = [0u8; 4096];
    for &h in hashes {
        let idx = (h & (M - 1)) as usize;
        let w = h >> P;
        let rho: u8 = if w == 0 {
            (64 - P + 1) as u8
        } else {
            (w.leading_zeros() - P + 1) as u8
        };
        if rho > regs[idx] {
            regs[idx] = rho;
        }
    }
    regs
}

/// Published HyperLogLog++ estimator with the documented small-range switch
/// to linear counting. Uses `alice_physics::det_math::ln64`, never `f64::ln`.
fn ref_cardinality(regs: &[u8; 4096]) -> f64 {
    let m = 4096.0f64;
    let alpha_bias = 0.7213 / (1.0 + 1.079 / m);
    let sum: f64 = regs.iter().map(|&r| 1.0 / (1u64 << r) as f64).sum();
    let zeros = regs.iter().filter(|&&r| r == 0).count();
    let raw = alpha_bias * m * m / sum;
    if raw <= 2.5 * m && zeros > 0 {
        m * ln64(m / zeros as f64)
    } else {
        raw
    }
}

// ---------------------------------------------------------------------------
// Main scenario (shared shape with the example; independently built here)
// ---------------------------------------------------------------------------

/// 196 nominal + 4 jank-spike steps (`step_time`/`contacts`), 197 stable + 3
/// divergence-spike steps (`energy_drift`, fed negative to exercise `abs`).
#[test]
fn telemetry_scenario_matches_closed_form_percentiles_and_total_steps() {
    let mut tel = PhysicsTelemetry::new();
    const N: usize = 200;
    let mut step_times = Vec::with_capacity(N);
    let mut contacts = Vec::with_capacity(N);
    let mut drifts = Vec::with_capacity(N);
    for step in 0..N {
        let st = if step < 196 { 1_000.0 } else { 50_000.0 };
        let ct = if step < 196 { 8.0 } else { 120.0 };
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

    assert_eq!(tel.total_steps(), N as u64);

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

    // A scene with no teeth would pass if p50/p99 were swapped; assert the
    // ordering explicitly too.
    assert!(tel.step_time_p99() > tel.step_time_p50() * 1.5);
    assert!(tel.contacts_p99() > tel.contacts_p50() * 1.5);

    // record_contacts / record_energy_drift / record_collision_pair must not
    // move total_steps.
    tel.record_contacts(999.0);
    tel.record_energy_drift(-999.0);
    tel.record_collision_pair(0xDEAD_BEEF);
    assert_eq!(
        tel.total_steps(),
        N as u64,
        "only record_step_time counts steps"
    );
}

/// `record_collision_pair` is order-sensitive on the *raw* hash it is given:
/// the module does not canonicalize `(a, b)` vs `(b, a)` itself (per its own
/// doc comment, the caller is responsible for e.g. `min(a,b)<<32|max(a,b)`).
/// 37 synthetic canonical pairs (each repeated once) + 1 real pair recorded
/// canonically from both call orders (1 observation) + 1 real pair recorded
/// *raw* from both call orders (2 observations, order-sensitive) = 40.
#[test]
fn record_collision_pair_is_order_sensitive_on_the_raw_hash() {
    let mut tel = PhysicsTelemetry::new();
    let mut hashes = Vec::new();

    for i in 0u64..37 {
        let (a, b) = (i, i + 1);
        let canonical = a.min(b) << 32 | a.max(b);
        tel.record_collision_pair(canonical);
        tel.record_collision_pair(canonical); // repeat: must not add a 38th
        hashes.push(canonical);
        hashes.push(canonical);
    }
    let (body_a, body_b) = (12u64, 1_000u64);
    let canon_ab = body_a.min(body_b) << 32 | body_a.max(body_b);
    tel.record_collision_pair(canon_ab); // (a, b) canonical
    tel.record_collision_pair(canon_ab); // (b, a) canonical -> same hash
    hashes.push(canon_ab);
    hashes.push(canon_ab);

    let (body_c, body_d) = (777u64, 2_001u64);
    let raw_cd = body_c << 32 | body_d; // (c, d) raw
    let raw_dc = body_d << 32 | body_c; // (d, c) raw -> DIFFERENT hash
    assert_ne!(
        raw_cd, raw_dc,
        "the scene must actually exercise order-sensitivity"
    );
    tel.record_collision_pair(raw_cd);
    tel.record_collision_pair(raw_dc);
    hashes.push(raw_cd);
    hashes.push(raw_dc);

    let expected_regs = ref_registers(&hashes);
    let expected = ref_cardinality(&expected_regs);
    let touched = expected_regs.iter().filter(|&&r| r != 0).count();
    assert_eq!(
        touched, 40,
        "expected exactly 40 distinct registers (37 + 1 + 2)"
    );

    let actual = tel.unique_collision_pairs();
    assert!(
        (actual - expected).abs() <= 1e-6 * expected,
        "unique_collision_pairs {actual} vs reference formula {expected} (touched={touched})"
    );
    // The documented order-sensitivity, stated as a direct consequence: had
    // record_collision_pair canonicalized (c, d) internally, raw_cd and
    // raw_dc would collapse to one observation and the total would be 39,
    // not 40 (within the sketch's error, i.e. clearly < 40 - 1 sigma-ish
    // margin here since this is an exact-formula regime).
    let without_order_sensitivity_regs = ref_registers(&hashes[..hashes.len() - 1]); // drop raw_dc
    let without = ref_cardinality(&without_order_sensitivity_regs);
    assert!(
        expected > without,
        "dropping the second (order-sensitive) observation must lower the reference cardinality: {expected} vs {without}"
    );
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

#[test]
fn zero_recordings_read_as_the_documented_empty_defaults() {
    let tel = PhysicsTelemetry::new();
    assert_eq!(tel.total_steps(), 0);
    assert_eq!(
        tel.step_time_p50(),
        0.0,
        "empty DDSketch256: documented early return"
    );
    assert_eq!(tel.step_time_p99(), 0.0);
    assert_eq!(tel.contacts_p50(), 0.0);
    assert_eq!(tel.contacts_p99(), 0.0);
    assert_eq!(tel.energy_drift_p99(), 0.0);
    assert_eq!(
        tel.unique_collision_pairs(),
        0.0,
        "empty HyperLogLog12: m * ln(m / m) = 0 exactly"
    );

    let tel_default = PhysicsTelemetry::default();
    assert_eq!(tel_default.total_steps(), 0);
    assert_eq!(tel_default.unique_collision_pairs(), 0.0);
}

#[test]
fn single_recording_gives_p50_equal_p99() {
    let mut tel = PhysicsTelemetry::new();
    tel.record_step_time(42.0);
    tel.record_contacts(7.0);
    tel.record_energy_drift(-3.5);
    tel.record_collision_pair(0xA11CE);

    assert_eq!(tel.total_steps(), 1);
    assert_eq!(
        tel.step_time_p50(),
        tel.step_time_p99(),
        "n=1: rank 1 for both q=0.5 and q=0.99"
    );
    assert_eq!(tel.contacts_p50(), tel.contacts_p99());
    assert_within_contract(tel.step_time_p50(), 42.0, "step_time_p50(n=1)");
    assert_within_contract(tel.contacts_p50(), 7.0, "contacts_p50(n=1)");
    assert_within_contract(
        tel.energy_drift_p99(),
        3.5,
        "energy_drift_p99(n=1, abs of -3.5)",
    );

    let expected = ref_cardinality(&ref_registers(&[0xA11CE]));
    assert!((tel.unique_collision_pairs() - expected).abs() <= 1e-6 * expected.max(1.0));
}

#[test]
fn all_identical_values_collapse_every_percentile_to_that_value() {
    let mut tel = PhysicsTelemetry::new();
    for _ in 0..500 {
        tel.record_step_time(2_000.0);
        tel.record_contacts(3.0);
        tel.record_energy_drift(0.25);
    }
    assert_eq!(tel.total_steps(), 500);
    assert_within_contract(tel.step_time_p50(), 2_000.0, "flat step_time_p50");
    assert_within_contract(tel.step_time_p99(), 2_000.0, "flat step_time_p99");
    assert_within_contract(tel.contacts_p50(), 3.0, "flat contacts_p50");
    assert_within_contract(tel.contacts_p99(), 3.0, "flat contacts_p99");
    assert_within_contract(tel.energy_drift_p99(), 0.25, "flat energy_drift_p99");
}

/// Extreme magnitudes: `+-inf` overflows the bucket-index cast and panics in
/// a debug build; a finite astronomical magnitude (`1e300`) does not panic
/// and is retained (clamped) in the edge bin, not silently dropped.
#[test]
fn extreme_magnitudes_panic_or_saturate_as_measured() {
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut tel = PhysicsTelemetry::new();
            tel.record_step_time(v);
            tel.total_steps()
        }));
        if cfg!(debug_assertions) {
            assert!(r.is_err(), "record_step_time({v}) panics in debug builds");
        } else {
            assert_eq!(
                r.expect("release wraps"),
                1,
                "release build still counts the step"
            );
        }
    }

    // A finite but astronomically large magnitude does not panic, and the
    // post-1.2.0 fix retains it (clamped to the edge bin) instead of
    // dropping it from the histogram.
    let mut huge = PhysicsTelemetry::new();
    huge.record_step_time(1e300);
    assert_eq!(huge.total_steps(), 1);
    let p = huge.step_time_p99();
    assert!(
        p.is_finite() && p > 0.0,
        "huge magnitude retained, finite: {p}"
    );

    // A finite but astronomically small positive magnitude: clamped to the
    // opposite edge bin, also finite and retained.
    let mut tiny = PhysicsTelemetry::new();
    tiny.record_step_time(1e-300);
    assert_eq!(tiny.total_steps(), 1);
    let p = tiny.step_time_p99();
    assert!(
        p.is_finite() && p > 0.0,
        "tiny magnitude retained, finite: {p}"
    );

    // NaN compares false to both `> 0.0` and `< 0.0`, so it lands in the
    // zero bucket rather than panicking.
    let mut nan = PhysicsTelemetry::new();
    let r = catch_unwind(AssertUnwindSafe(|| {
        nan.record_step_time(f64::NAN);
        nan.total_steps()
    }));
    assert!(r.is_ok(), "NaN does not panic: it lands in the zero bucket");
    assert_eq!(nan.total_steps(), 1);
    assert_eq!(
        nan.step_time_p99(),
        0.0,
        "NaN recording: zero bucket, rank 1 of 1"
    );
}
