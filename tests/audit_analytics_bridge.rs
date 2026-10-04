//! Audit oracles for `alice_physics::analytics_bridge::PhysicsTelemetry`.
//!
//! Percentile expectations are nearest-rank values of known sample sets
//! with the documented relative accuracy `alpha`; cardinality expectations
//! are the true number of distinct inputs with the 1.6 % standard error of
//! a 4096-register HyperLogLog (checked with a 4-sigma margin).

#![cfg(all(feature = "std", feature = "analytics"))]
#![allow(clippy::disallowed_methods)]

use alice_physics::analytics_bridge::PhysicsTelemetry;

/// Relative accuracy documented for the sketches (5 %), written as a
/// literal so the oracle does not move with the constant under test.
const DOC_ALPHA: f64 = 0.05;

fn within(actual: f64, v: f64) -> bool {
    actual >= v * (1.0 - DOC_ALPHA) && actual <= v * (1.0 + DOC_ALPHA)
}

/// The documented relative accuracy is 5 %.
#[test]
fn alpha_is_five_percent() {
    assert_eq!(PhysicsTelemetry::ALPHA, DOC_ALPHA);
}

/// Every sketch (step time, contacts, drift) covers magnitudes up to the
/// documented 2.1e8 range, not only small values: 49 samples at 1000, 50 at
/// 2000 and one outlier at 1e6 give p99 = rank 99 = 2000.
#[test]
fn all_three_sketches_cover_large_magnitudes() {
    let mut t = PhysicsTelemetry::new();
    for _ in 0..49 {
        t.record_step_time(1000.0);
        t.record_contacts(1000.0);
        t.record_energy_drift(1000.0);
    }
    for _ in 0..50 {
        t.record_step_time(2000.0);
        t.record_contacts(2000.0);
        t.record_energy_drift(2000.0);
    }
    t.record_step_time(1.0e6);
    t.record_contacts(1.0e6);
    t.record_energy_drift(1.0e6);
    assert!(within(t.step_time_p99(), 2000.0), "{}", t.step_time_p99());
    assert!(within(t.contacts_p99(), 2000.0), "{}", t.contacts_p99());
    assert!(
        within(t.energy_drift_p99(), 2000.0),
        "{}",
        t.energy_drift_p99()
    );
}

/// Median of a three-level sample (40 x 1, 20 x 10, 40 x 100): nearest rank
/// 50 is 10 for contacts, and p99 is 100.
#[test]
fn contacts_median_is_rank_fifty_of_one_hundred() {
    let mut t = PhysicsTelemetry::new();
    for _ in 0..40 {
        t.record_contacts(1.0);
    }
    for _ in 0..20 {
        t.record_contacts(10.0);
    }
    for _ in 0..40 {
        t.record_contacts(100.0);
    }
    assert!(within(t.contacts_p50(), 10.0), "p50 {}", t.contacts_p50());
    assert!(within(t.contacts_p99(), 100.0), "p99 {}", t.contacts_p99());
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact documented range: representable from about 1.6e-3 to 2.1e8.
/// Samples just inside both ends keep the relative accuracy.
#[test]
fn documented_representable_range_keeps_relative_accuracy_at_both_ends() {
    for v in [2.0e-3, 1.0e-2, 1.0, 1.0e3, 1.0e6, 1.9e8] {
        let mut t = PhysicsTelemetry::new();
        t.record_step_time(v);
        t.record_contacts(v);
        t.record_energy_drift(v);
        assert!(
            within(t.step_time_p50(), v),
            "step {v}: {}",
            t.step_time_p50()
        );
        assert!(
            within(t.contacts_p99(), v),
            "contacts {v}: {}",
            t.contacts_p99()
        );
        assert!(
            within(t.energy_drift_p99(), v),
            "drift {v}: {}",
            t.energy_drift_p99()
        );
    }
}

/// Nearest-rank percentiles over a decade ladder 1e-2 .. 1e8 (11 samples):
/// p50 is rank 6 = 1e3, p99 is rank 11 = 1e8.
#[test]
fn decade_ladder_percentiles() {
    let mut t = PhysicsTelemetry::new();
    for k in -2..=8 {
        t.record_step_time(10f64.powi(k));
    }
    assert!(
        within(t.step_time_p50(), 1.0e3),
        "p50 {}",
        t.step_time_p50()
    );
    assert!(
        within(t.step_time_p99(), 1.0e8),
        "p99 {}",
        t.step_time_p99()
    );
    assert_eq!(t.total_steps(), 11);
}

/// Drift is recorded as a magnitude: +d and -d are the same observation.
#[test]
fn energy_drift_sign_is_ignored() {
    let mut pos = PhysicsTelemetry::new();
    let mut neg = PhysicsTelemetry::new();
    for i in 1..=50 {
        let d = 0.01 * i as f64;
        pos.record_energy_drift(d);
        neg.record_energy_drift(-d);
    }
    assert_eq!(pos.energy_drift_p99(), neg.energy_drift_p99());
    // rank 50 of 50 = 0.5
    assert!(within(pos.energy_drift_p99(), 0.5));
}

/// A zero drift stays zero (a perfectly conserving step is not reported as
/// a positive drift).
#[test]
fn zero_drift_reports_zero() {
    let mut t = PhysicsTelemetry::new();
    for _ in 0..100 {
        t.record_energy_drift(0.0);
    }
    assert_eq!(t.energy_drift_p99(), 0.0);
}

/// p50 never exceeds p99 for an arbitrary stream.
#[test]
fn p50_never_exceeds_p99() {
    let mut t = PhysicsTelemetry::new();
    for i in 0..500u64 {
        let v = 1.0 + (splitmix(i) % 100_000) as f64 * 0.37;
        t.record_step_time(v);
        t.record_contacts(v * 0.01);
    }
    assert!(t.step_time_p50() <= t.step_time_p99());
    assert!(t.contacts_p50() <= t.contacts_p99());
}

/// An out-of-range outlier does not disturb the percentiles of the
/// in-range bulk: 99 samples of 100 and one of 1e12 give p50 = p99 = 100
/// (nearest rank 99 of 100).
#[test]
fn out_of_range_outlier_does_not_move_in_range_percentiles() {
    let mut t = PhysicsTelemetry::new();
    for _ in 0..99 {
        t.record_step_time(100.0);
    }
    t.record_step_time(1.0e12);
    assert!(
        within(t.step_time_p50(), 100.0),
        "p50 {}",
        t.step_time_p50()
    );
    assert!(
        within(t.step_time_p99(), 100.0),
        "p99 {}",
        t.step_time_p99()
    );
}

/// Cardinality of well-mixed hashes: 10 000 distinct values, each recorded
/// twice, estimate within 4 sigma (6.4 %) of the true count.
#[test]
fn unique_pairs_for_well_mixed_hashes_is_accurate() {
    let mut t = PhysicsTelemetry::new();
    for i in 0..10_000u64 {
        let h = splitmix(i);
        t.record_collision_pair(h);
        t.record_collision_pair(h);
    }
    let est = t.unique_collision_pairs();
    assert!((est - 10_000.0).abs() < 640.0, "estimate {est}");
}

/// The documented packing `min(a,b) << 32 | max(a,b)` of real body ids
/// gives a usable estimate of the number of distinct colliding pairs:
/// 100 bodies, all 4950 pairs.
#[test]
fn documented_pair_packing_gives_usable_cardinality() {
    let mut t = PhysicsTelemetry::new();
    for a in 0..100u64 {
        for b in (a + 1)..100u64 {
            t.record_collision_pair((a << 32) | b);
        }
    }
    let est = t.unique_collision_pairs();
    assert!((est - 4950.0).abs() < 0.1 * 4950.0, "estimate {est}");
}

/// Re-recording the same pair does not change the estimate (idempotent).
#[test]
fn recording_the_same_pair_is_idempotent() {
    let mut t = PhysicsTelemetry::new();
    for _ in 0..1000 {
        t.record_collision_pair(splitmix(42));
    }
    let est = t.unique_collision_pairs();
    assert!((est - 1.0).abs() < 1.0e-3, "estimate {est}");
}
