//! Oracles for the metric pipeline (`pipeline::RingBuffer`, `MetricEvent`,
//! `MetricSlot`, `MetricPipeline`, `MetricRegistry`).
//!
//! Every expected value is derived here from the definitions, never from
//! the code under test:
//!
//! ```text
//! ring buffer of QUEUE_SIZE = N     capacity C = N − 1 (one cell stays empty)
//! k pushes                           len = min(k, C), dropped = max(k − C, 0),
//!                                    is_full ⇔ len == C
//! one flush                          total_events = len before the flush,
//!                                    total_events + dropped_events = k
//! histogram, alpha = 1/2             gamma = (1 + α)/(1 − α) = 3
//!   sample v > 0                     bucket b(v) = ceil(log₃ v)
//!   quantile(q), n samples           rank r = ceil(q·n); walk buckets in
//!                                    ascending order, report 3^(b − 1) of the
//!                                    bucket where the cumulative count ≥ r
//! unique, m = 1024 registers         k items in k distinct registers →
//!                                    cardinality = m · ln(m / (m − k))
//!                                    (linear counting, raw estimate ≤ 2.5 m)
//! ```
//!
//! The logarithm in the unique oracle is the test's own series
//! `ln(1/(1 − x)) = Σ xⁿ/n`, so the expected value does not go through the
//! crate's `ln64`.
//!
//! # Degenerate input
//!
//! `QUEUE_SIZE = 1` is a legal queue of capacity 0: every submit is dropped
//! and `is_full` is true while empty. `SLOTS = 0` and `QUEUE_SIZE = 0`
//! currently panic with a divide by zero (measured, reported as a finding;
//! whether that becomes a compile-time refusal is a design ruling).
//! `alpha` is not validated by `MetricPipeline::new`: `alpha = 0`
//! overflows in the bucket index (debug) and `alpha ≥ 1` or `alpha < 0`
//! silently break the quantile contract; the test pins that the contract is
//! broken rather than hiding it. Item hashes above 2^53 collapse in
//! `MetricEvent::unique` because the hash travels through `f64`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::float_cmp)]

use alice_physics::pipeline::{
    MetricEntry, MetricEvent, MetricPipeline, MetricRegistry, MetricSlot, MetricType, RingBuffer,
};
use alice_physics::sketch::FnvHasher;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// `ln(m / (m − k))` for the HLL linear-counting oracle, as the series
/// `Σ_{n≥1} (k/m)^n / n`; with `k/m ≤ 5/1024` sixteen terms are far below
/// 1e-30 of the true value.
fn ln_m_over_m_minus_k(m: f64, k: f64) -> f64 {
    let x = k / m;
    let mut term = 1.0;
    let mut sum = 0.0;
    for n in 1..=16 {
        term *= x;
        sum += term / f64::from(n);
    }
    sum
}

/// `3^e` for small integer `e ≥ 0`, by repeated multiplication.
fn pow3(e: u32) -> f64 {
    (0..e).fold(1.0, |acc, _| acc * 3.0)
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1.0)
}

/// Fresh pipeline with `alpha = 1/2` (gamma = 3) so buckets are hand-readable.
fn pipeline<const S: usize, const Q: usize>() -> MetricPipeline<S, Q> {
    MetricPipeline::<S, Q>::new(0.5)
}

// ---------------------------------------------------------------------------
// Ring buffer: capacity / is_full / dropped
// ---------------------------------------------------------------------------

#[test]
fn ring_buffer_capacity_is_full_and_dropped_are_closed_form() {
    fn check<const N: usize>() {
        let c = N - 1;
        let mut rb = RingBuffer::<u32, N>::new();
        assert_eq!(rb.capacity(), c, "N = {N}");
        assert!(!rb.is_full() || c == 0, "N = {N}: fresh buffer is not full");
        for k in 1..=(2 * N) {
            let accepted = rb.push(k as u32);
            assert_eq!(accepted, k <= c, "N = {N}, push #{k}");
            assert_eq!(rb.len(), k.min(c), "N = {N}, push #{k}");
            assert_eq!(
                rb.dropped(),
                k.saturating_sub(c) as u64,
                "N = {N}, push #{k}"
            );
            assert_eq!(rb.is_full(), rb.len() == c, "N = {N}, push #{k}");
        }
        // Draining one makes room for exactly one more: is_full flips
        // false then back to true, dropped does not move on a successful push.
        let dropped = rb.dropped();
        if c > 0 {
            assert_eq!(rb.pop(), Some(1));
            assert!(!rb.is_full());
            assert!(rb.push(99));
            assert!(rb.is_full());
            assert_eq!(rb.len(), c, "N = {N}: length across the wrap");
            assert_eq!(rb.dropped(), dropped);
        }
        // Lifetime counter: clear() resets it, pop() does not.
        rb.clear();
        assert_eq!(rb.dropped(), 0);
        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
    }
    check::<2>();
    check::<4>();
    check::<8>();
    check::<13>();
}

// ---------------------------------------------------------------------------
// Pipeline queue: submit / queue_len / dropped_events / total_events
// ---------------------------------------------------------------------------

#[test]
fn pipeline_queue_counts_conserve_submitted_events() {
    const Q: usize = 8;
    let c = Q - 1;
    for n in [0usize, 1, 3, 7, 8, 12, 40] {
        let mut p = pipeline::<16, Q>();
        let h = 5u64;
        let mut accepted = 0usize;
        for i in 0..n {
            if p.submit(MetricEvent::counter(h, 1.0).with_timestamp(i as u64)) {
                accepted += 1;
            }
            assert_eq!(p.queue_len(), (i + 1).min(c), "n = {n}, submit #{i}");
            assert_eq!(
                p.dropped_events(),
                (i + 1).saturating_sub(c) as u64,
                "n = {n}, submit #{i}"
            );
            assert_eq!(p.total_events(), 0, "nothing is processed before a flush");
        }
        assert_eq!(accepted, n.min(c), "n = {n}");
        p.flush();
        assert_eq!(p.queue_len(), 0, "n = {n}: flush drains the queue");
        assert_eq!(p.total_events(), n.min(c) as u64, "n = {n}");
        assert_eq!(p.dropped_events(), n.saturating_sub(c) as u64, "n = {n}");
        assert_eq!(
            p.total_events() + p.dropped_events(),
            n as u64,
            "n = {n}: every submitted event is either processed or dropped"
        );
        if n > 0 {
            let slot = p.get_slot(h).expect("flush creates the slot");
            assert_eq!(slot.event_count, n.min(c) as u64);
            assert_eq!(slot.counter, n.min(c) as f64);
            // Timestamps are the first `min(n, c)` ones: the queue is FIFO
            // and the drops are the trailing submits.
            assert_eq!(slot.last_update, (n.min(c) - 1) as u64);
        } else {
            assert!(p.get_slot(h).is_none());
            assert_eq!(p.iter_slots().count(), 0);
        }
        // A second round after the flush accumulates: dropped is lifetime,
        // and the queue length is right across the ring's wrap-around.
        for _ in 0..n {
            p.submit(MetricEvent::counter(h, 1.0));
        }
        assert_eq!(p.queue_len(), n.min(c), "n = {n}: length across the wrap");
        p.flush();
        assert_eq!(p.total_events(), 2 * n.min(c) as u64);
        assert_eq!(p.dropped_events(), 2 * n.saturating_sub(c) as u64);
    }
}

// ---------------------------------------------------------------------------
// MetricEvent constructors: counter / gauge / histogram / unique / with_timestamp
// ---------------------------------------------------------------------------

#[test]
fn with_timestamp_round_trips_and_differs_from_the_default() {
    let h = 11u64;
    for ts in [1u64, 42, 1_700_000_000_123, u64::MAX] {
        for base in [
            MetricEvent::counter(h, 1.5),
            MetricEvent::gauge(h, -2.0),
            MetricEvent::histogram(h, 9.0),
            MetricEvent::unique(h, 77),
        ] {
            assert_eq!(base.timestamp, 0, "constructors start at 0");
            let e = base.with_timestamp(ts);
            assert_eq!(e.timestamp, ts);
            assert_ne!(e.timestamp, base.timestamp);
            assert_eq!(e.name_hash, base.name_hash);
            assert_eq!(e.metric_type, base.metric_type);
            assert_eq!(e.value, base.value);
            assert_eq!(e.with_timestamp(3).timestamp, 3, "last stamp wins");
        }
    }
    // Through the pipeline: last_update is the max stamp seen, not the last.
    let mut p = pipeline::<4, 8>();
    p.submit(MetricEvent::gauge(h, 1.0).with_timestamp(50));
    p.submit(MetricEvent::gauge(h, 2.0).with_timestamp(20));
    p.flush();
    let s = p.get_slot(h).unwrap();
    assert_eq!(s.last_update, 50);
    assert_eq!(s.gauge, 2.0, "gauge is the last value regardless of stamp");
}

#[test]
fn counter_and_gauge_fold_exactly() {
    let (hc, hg) = (2u64, 3u64);
    let mut p = pipeline::<8, 64>();
    // Σ of dyadic deltas, including negatives, is exact in f64.
    let deltas = [1.0, 0.5, -0.25, 4.0, -1.0, 0.125];
    let expected: f64 = 1.0 + 0.5 - 0.25 + 4.0 - 1.0 + 0.125;
    for d in deltas {
        p.submit(MetricEvent::counter(hc, d));
    }
    let gauges = [3.0, -7.5, 0.0, 12.25];
    for g in gauges {
        p.submit(MetricEvent::gauge(hg, g));
    }
    p.flush();
    let c = p.get_slot(hc).unwrap();
    assert_eq!(c.counter, expected);
    assert_eq!(c.event_count, deltas.len() as u64);
    assert_eq!(c.gauge, 0.0, "counter events leave the gauge alone");
    let g = p.get_slot(hg).unwrap();
    assert_eq!(g.gauge, 12.25, "gauge is the last value, not a sum");
    assert_eq!(g.counter, 0.0, "gauge events leave the counter alone");
    assert_eq!(g.event_count, gauges.len() as u64);
    assert_eq!(p.total_events(), (deltas.len() + gauges.len()) as u64);
}

#[test]
fn histogram_buckets_by_hand_with_gamma_three() {
    // alpha = 1/2 → gamma = 3. bucket(v) = ceil(log₃ v):
    //   2 → 1 (1 < 2 ≤ 3)   4, 5, 8 → 2 (3 < v ≤ 9)   10 → 3 (9 < 10 ≤ 27)
    //   100 → 5 (81 < 100 ≤ 243)
    // bucket counts: {1: 1, 2: 3, 3: 1, 5: 1}, cumulative: 1, 4, 5, 6.
    // quantile(q) with n = 6: rank r = ceil(6 q), reported value 3^(bucket − 1).
    let h = 7u64;
    let sample = [2.0, 4.0, 5.0, 8.0, 10.0, 100.0];
    let mut p = pipeline::<8, 16>();
    for v in sample {
        assert!(p.submit(MetricEvent::histogram(h, v)));
    }
    p.flush();
    let s = p.get_slot(h).unwrap();
    assert_eq!(s.ddsketch.count(), 6);
    assert_eq!(s.ddsketch.min(), 2.0);
    assert_eq!(s.ddsketch.max(), 100.0);
    assert_eq!(s.ddsketch.sum(), 129.0);
    assert_eq!(s.ddsketch.mean(), 21.5);
    assert_eq!(s.event_count, 6);
    assert_eq!(s.counter, 0.0, "histogram events leave the counter alone");
    let table = [
        (0.1, 1, 1u32), // r = 1 → bucket 1 → 3^0
        (0.3, 2, 2),    // r = 2 → bucket 2 → 3^1
        (0.5, 3, 2),    // r = 3 → bucket 2
        (0.6, 4, 2),    // r = 4 → bucket 2 (cumulative 4 ≥ 4)
        (0.7, 5, 3),    // r = 5 → bucket 3 → 3^2
        (0.95, 6, 5),   // r = 6 → bucket 5 → 3^4
        (1.0, 6, 5),
    ];
    for (q, rank, bucket) in table {
        assert_eq!(
            (q * 6.0_f64).ceil() as u64,
            rank,
            "rank bookkeeping for q = {q}"
        );
        let expected = pow3(bucket - 1);
        let got = s.ddsketch.quantile(q);
        assert!(
            close(got, expected, 1e-9),
            "q = {q}: got {got}, expected 3^{} = {expected}",
            bucket - 1
        );
    }
    // The bucket is a property of the value alone: a second sample set with
    // the same buckets and multiplicities reports identical quantiles.
    // (Values stay off exact powers of 3: `ln64(9) / ln64(3)` rounds to
    // 2.0000000000000004 and `ceil` moves 9 into bucket 3.)
    //   2.5 → 1;  3.5, 7, 6 → 2;  20 → 3;  200 → 5
    let mut p2 = pipeline::<8, 16>();
    for v in [2.5, 3.5, 7.0, 6.0, 20.0, 200.0] {
        p2.submit(MetricEvent::histogram(h, v));
    }
    p2.flush();
    let s2 = p2.get_slot(h).unwrap();
    for (q, _, _) in table {
        assert_eq!(s.ddsketch.quantile(q), s2.ddsketch.quantile(q), "q = {q}");
    }
    assert_ne!(s2.ddsketch.sum(), s.ddsketch.sum());

    // Negative and zero samples: {−2, 0, 4}. |−2| → bucket 1 on the negative
    // side (reported −3^0 = −1), zero has its own count, 4 → bucket 2 (3).
    let mut p3 = pipeline::<8, 16>();
    for v in [-2.0, 0.0, 4.0] {
        p3.submit(MetricEvent::histogram(h, v));
    }
    p3.flush();
    let s3 = p3.get_slot(h).unwrap();
    assert!(
        close(s3.ddsketch.quantile(0.3), -1.0, 1e-9),
        "r = 1 → the negative sample"
    );
    assert_eq!(s3.ddsketch.quantile(0.5), 0.0, "r = 2 → the zero");
    assert!(
        close(s3.ddsketch.quantile(0.9), 3.0, 1e-9),
        "r = 3 → bucket 2"
    );
    assert_eq!(s3.ddsketch.min(), -2.0);
    assert_eq!(s3.ddsketch.max(), 4.0);
}

#[test]
fn unique_counts_a_multiset_by_its_set_with_linear_counting() {
    // Item hashes 1..=k, mixed with splitmix64 before insert, land in k
    // distinct registers (checked: the k = 1..=5 estimates below equal the
    // k-register closed form), so the linear-counting branch sees exactly k
    // non-zero registers.
    let h = 9u64;
    let m = 1024.0;
    for k in 1..=5u64 {
        let mut set = pipeline::<8, 64>();
        let mut multiset = pipeline::<8, 64>();
        let mut events = 0u64;
        for item in 1..=k {
            set.submit(MetricEvent::unique(h, item));
            // item inserted `item` times: multiplicities 1, 2, …, k
            for _ in 0..item {
                multiset.submit(MetricEvent::unique(h, item));
                events += 1;
            }
        }
        set.flush();
        multiset.flush();
        let expected = m * ln_m_over_m_minus_k(m, k as f64);
        let got_set = set.get_slot(h).unwrap().hll.cardinality();
        let got_multi = multiset.get_slot(h).unwrap().hll.cardinality();
        assert!(
            close(got_set, expected, 1e-9),
            "k = {k}: {got_set} vs {expected}"
        );
        assert_eq!(
            got_multi, got_set,
            "k = {k}: cardinality depends only on the set"
        );
        assert_eq!(multiset.get_slot(h).unwrap().event_count, events);
        assert_eq!(events, k * (k + 1) / 2);
        assert_eq!(
            multiset.get_slot(h).unwrap().counter,
            0.0,
            "unique events leave the counter alone"
        );
    }
    // Distinct inputs give distinct outputs: k = 1..=5 are strictly increasing.
    let mut prev = 0.0;
    for k in 1..=5u64 {
        let v = m * ln_m_over_m_minus_k(m, k as f64);
        assert!(v > prev);
        prev = v;
    }
}

// ---------------------------------------------------------------------------
// Slot registry: get_slot / get_slot_mut / iter_slots
// ---------------------------------------------------------------------------

#[test]
fn pipeline_slots_get_iter_and_mut_are_consistent() {
    const S: usize = 16;
    // Buckets are hash % 16: 3 → 3, 5 → 5, 9 → 9, 21 → 5 (collides with 5).
    let (a, b, c, b_alias) = (3u64, 5u64, 9u64, 21u64);
    assert_eq!((b_alias as usize) % S, (b as usize) % S);
    let mut p = pipeline::<S, 32>();
    assert_eq!(p.iter_slots().count(), 0);
    assert!(p.get_slot(a).is_none());
    assert!(p.get_slot_mut(a).is_none());

    // Submit in an order that is not bucket order.
    p.submit(MetricEvent::counter(c, 1.0));
    p.submit(MetricEvent::counter(a, 2.0));
    p.submit(MetricEvent::counter(b, 4.0));
    p.submit(MetricEvent::counter(b_alias, 8.0));
    assert_eq!(
        p.iter_slots().count(),
        0,
        "nothing is active before a flush"
    );
    p.flush();

    // iter_slots walks the slot array, i.e. ascending bucket index.
    let order: Vec<u64> = p.iter_slots().map(|s| s.name_hash).collect();
    assert_eq!(order, vec![a, b, c]);
    assert_eq!(p.total_events(), 4);
    // The colliding hash is folded into the existing slot (documented
    // approximation) and is not retrievable by its own hash.
    assert!(p.get_slot(b_alias).is_none());
    assert!(p.get_slot_mut(b_alias).is_none());
    let sb = p.get_slot(b).unwrap();
    assert_eq!(sb.event_count, 2);
    assert_eq!(sb.counter, 12.0);
    // get_slot and iter_slots hand out the same slot.
    for s in p.iter_slots() {
        let g = p.get_slot(s.name_hash).unwrap();
        assert!(std::ptr::eq(s, g));
        assert_eq!(s.counter, g.counter);
    }
    // Σ counters over the iterator = 1 + 2 + 4 + 8.
    let total: f64 = p.iter_slots().map(|s| s.counter).sum();
    assert_eq!(total, 15.0);

    // get_slot_mut edits are the slot that get_slot / iter_slots see.
    {
        let sa = p.get_slot_mut(a).unwrap();
        assert_eq!(sa.name_hash, a);
        sa.process(&MetricEvent::counter(a, 0.5));
        sa.gauge = -1.0;
    }
    let sa = p.get_slot(a).unwrap();
    assert_eq!(sa.counter, 2.5);
    assert_eq!(sa.gauge, -1.0);
    assert_eq!(sa.event_count, 2);
    assert_eq!(p.iter_slots().map(|s| s.counter).sum::<f64>(), 15.5);
    assert_eq!(
        p.total_events(),
        4,
        "direct slot edits bypass the pipeline counter"
    );
    assert_eq!(p.iter_slots().count(), 3, "edits do not add slots");
}

// ---------------------------------------------------------------------------
// Named registry: lookup / lookup_by_hash / name_str
// ---------------------------------------------------------------------------

#[test]
fn registry_lookup_by_name_and_by_hash_agree_and_iter_is_registration_order() {
    let names = ["http.requests", "", "db.latency_ms", "user.ids", "x"];
    let kinds = [
        MetricType::Counter,
        MetricType::Gauge,
        MetricType::Histogram,
        MetricType::Unique,
        MetricType::Counter,
    ];
    let mut reg = MetricRegistry::<8>::new();
    assert_eq!(reg.count(), 0);
    assert_eq!(reg.iter().count(), 0);
    let mut hashes = Vec::new();
    for (name, kind) in names.iter().zip(kinds) {
        let h = reg.register(name, kind).expect("room");
        // The registry hash is the FNV hash of the full name (an input, not
        // the item under test).
        assert_eq!(h, FnvHasher::hash_bytes(name.as_bytes()));
        hashes.push(h);
    }
    assert_eq!(reg.count(), names.len());
    for (i, name) in names.iter().enumerate() {
        let by_name = reg.lookup(name).expect("by name");
        let by_hash = reg.lookup_by_hash(hashes[i]).expect("by hash");
        assert!(
            std::ptr::eq(by_name, by_hash),
            "{name:?}: both lookups return the same entry"
        );
        assert_eq!(by_name.name_str(), *name);
        assert_eq!(by_name.name_len, name.len());
        assert_eq!(by_name.hash, hashes[i]);
        assert_eq!(by_name.metric_type, kinds[i]);
    }
    // Iteration is registration order, and the i-th entry is the i-th name.
    let seen: Vec<&str> = reg.iter().map(MetricEntry::name_str).collect();
    assert_eq!(seen, names);
    let seen_hashes: Vec<u64> = reg.iter().map(|e| e.hash).collect();
    assert_eq!(seen_hashes, hashes);

    // Unregistered name / hash → None; the empty name is a real key.
    assert!(reg.lookup("http.request").is_none());
    assert!(reg.lookup("HTTP.REQUESTS").is_none());
    assert!(reg.lookup_by_hash(0).is_none());
    assert!(reg.lookup_by_hash(hashes[0] ^ 1).is_none());
    assert_eq!(reg.lookup("").unwrap().name_str(), "");
    assert_ne!(reg.lookup("").unwrap().hash, 0);

    // Registering the same name twice gives two entries with the same hash;
    // lookup returns the first (registration order).
    let dup = reg.register("x", MetricType::Gauge).unwrap();
    assert_eq!(dup, hashes[4]);
    assert_eq!(reg.count(), 6);
    assert_eq!(reg.lookup("x").unwrap().metric_type, MetricType::Counter);
    // Full registry refuses.
    assert!(reg.register("y", MetricType::Gauge).is_some());
    assert!(reg.register("z", MetricType::Gauge).is_some());
    assert_eq!(reg.count(), 8);
    assert!(reg.register("overflow", MetricType::Gauge).is_none());
    assert_eq!(reg.count(), 8);
    assert!(reg.lookup("overflow").is_none());

    // name_str truncates at 64 bytes; the hash is over the full name.
    let long = "n".repeat(100);
    let hl = reg_fresh_hash(&long);
    assert_eq!(hl, FnvHasher::hash_bytes(long.as_bytes()));
}

fn reg_fresh_hash(name: &str) -> u64 {
    let mut r = MetricRegistry::<1>::new();
    let h = r.register(name, MetricType::Gauge).unwrap();
    let e = r.lookup_by_hash(h).unwrap();
    assert_eq!(e.name_str().len(), 64);
    assert_eq!(e.name_str(), &name[..64]);
    assert!(r.lookup(name).is_some(), "lookup uses the full name");
    h
}

// ---------------------------------------------------------------------------
// reset (MetricSlot::reset and MetricPipeline::reset)
// ---------------------------------------------------------------------------

#[test]
fn reset_returns_observables_to_the_fresh_state() {
    let fresh_pipeline = format!("{:?}", pipeline::<4, 8>());
    let mut p = pipeline::<4, 8>();
    // reset on an empty pipeline is the identity, bit for bit (Debug covers
    // every field including the ring positions).
    p.reset();
    assert_eq!(format!("{p:?}"), fresh_pipeline);

    let (h1, h2) = (1u64, 2u64);
    p.submit(MetricEvent::counter(h1, 3.0).with_timestamp(10));
    p.submit(MetricEvent::gauge(h1, 4.0).with_timestamp(20));
    p.submit(MetricEvent::histogram(h2, 5.0).with_timestamp(30));
    p.submit(MetricEvent::unique(h2, 6).with_timestamp(40));
    p.flush();
    for _ in 0..9 {
        p.submit(MetricEvent::counter(h1, 1.0));
    }
    assert_eq!(p.queue_len(), 7);
    assert_eq!(p.dropped_events(), 2);
    assert_eq!(p.total_events(), 4);

    // MetricSlot::reset through get_slot_mut: the slot equals a fresh one
    // with the same hash and alpha, bit for bit.
    {
        let s = p.get_slot_mut(h2).unwrap();
        assert_eq!(s.event_count, 2);
        s.reset();
    }
    assert_eq!(
        format!("{:?}", p.get_slot(h2).unwrap()),
        format!("{:?}", MetricSlot::new(h2, 0.5))
    );
    // The other slot is untouched by a per-slot reset.
    assert_eq!(p.get_slot(h1).unwrap().event_count, 2);
    assert_eq!(p.get_slot(h1).unwrap().last_update, 20);

    // MetricPipeline::reset: every observable is the fresh value, and every
    // slot is bit-identical to a fresh slot; the slots stay registered.
    p.reset();
    assert_eq!(p.queue_len(), 0);
    assert_eq!(p.dropped_events(), 0);
    assert_eq!(p.total_events(), 0);
    assert_eq!(
        p.iter_slots().count(),
        2,
        "reset keeps the pre-allocated slots"
    );
    for s in p.iter_slots() {
        assert_eq!(
            format!("{s:?}"),
            format!("{:?}", MetricSlot::new(s.name_hash, 0.5))
        );
    }
    assert_eq!(p.get_slot(h1).unwrap().counter, 0.0);
    assert_eq!(p.get_slot(h1).unwrap().gauge, 0.0);
    assert_eq!(p.get_slot(h1).unwrap().last_update, 0);
    // A reset pipeline accepts and counts again from zero.
    for _ in 0..10 {
        p.submit(MetricEvent::counter(h1, 1.0));
    }
    p.flush();
    assert_eq!(p.total_events(), 7);
    assert_eq!(p.dropped_events(), 3);
    assert_eq!(p.get_slot(h1).unwrap().counter, 7.0);
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

#[test]
fn capacity_zero_queue_drops_every_submit() {
    // QUEUE_SIZE = 1 → capacity 0: the ring is full and empty at once.
    let mut rb = RingBuffer::<u32, 1>::new();
    assert_eq!(rb.capacity(), 0);
    assert!(rb.is_full());
    assert!(rb.is_empty());
    assert!(!rb.push(1));
    assert_eq!(rb.dropped(), 1);
    assert_eq!(rb.len(), 0);
    assert_eq!(rb.pop(), None);

    let mut p = pipeline::<4, 1>();
    for i in 1..=5u64 {
        assert!(!p.submit(MetricEvent::counter(1, 1.0)));
        assert_eq!(p.dropped_events(), i);
        assert_eq!(p.queue_len(), 0);
    }
    p.flush();
    assert_eq!(p.total_events(), 0);
    assert!(p.get_slot(1).is_none());
    assert_eq!(p.iter_slots().count(), 0);
    // reset on a pipeline that never accepted anything.
    p.reset();
    assert_eq!(p.dropped_events(), 0);
    assert_eq!(format!("{p:?}"), format!("{:?}", pipeline::<4, 1>()));
}

/// Open issue `pipeline-degenerate-config-panics` (1/2): measured current
/// behaviour, pinned so the fix has a red to turn green. Zero-sized const
/// generics reach `% 0` (`src/pipeline.rs` `RingBuffer::push` /
/// `is_full`, `MetricPipeline::process_event` / `get_slot`). Whether this
/// becomes a compile-time refusal is a design ruling; this test does not
/// decide it, it only reports what happens today.
#[test]
fn zero_sized_const_generics_currently_panic_with_divide_by_zero() {
    fn panics<F: FnOnce()>(f: F) -> bool {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }
    // QUEUE_SIZE = 0: construction is fine, the first push / is_full is not.
    assert!(!panics(|| {
        let _ = pipeline::<4, 0>();
    }));
    assert!(panics(|| {
        let mut p = pipeline::<4, 0>();
        p.submit(MetricEvent::counter(1, 1.0));
    }));
    assert!(panics(|| {
        let rb = RingBuffer::<u32, 0>::new();
        let _ = rb.is_full();
    }));
    // SLOTS = 0: submit is fine (it only queues), flush and get_slot are not.
    assert!(!panics(|| {
        let mut p = pipeline::<0, 8>();
        assert!(p.submit(MetricEvent::counter(1, 1.0)));
        assert_eq!(p.queue_len(), 1);
    }));
    assert!(panics(|| {
        let mut p = pipeline::<0, 8>();
        p.submit(MetricEvent::counter(1, 1.0));
        p.flush();
    }));
    assert!(panics(|| {
        let p = pipeline::<0, 8>();
        let _ = p.get_slot(1);
    }));
    assert!(!panics(|| {
        let p = pipeline::<0, 8>();
        assert_eq!(p.iter_slots().count(), 0);
    }));
}

/// `MetricPipeline::new(alpha)` passes `alpha` through unvalidated.
/// `alpha = 0` gives `gamma = 1` and `ln_gamma = 0`, so every bucket key is
/// `ln(v) / 0` = ±∞ or NaN. The DDSketch's key arithmetic saturates, so the
/// samples are counted without a panic (before the window followed the data,
/// `|v| > 1` overflowed the `i32` bucket index and panicked in a debug
/// build); the answers carry no meaning, which is the unvalidated-α
/// characterization below.
#[test]
fn alpha_zero_counts_samples_without_panicking() {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut p = MetricPipeline::<2, 4>::new(0.0);
        p.submit(MetricEvent::histogram(1, 2.0));
        p.submit(MetricEvent::histogram(1, 1.0));
        p.submit(MetricEvent::histogram(1, 0.5));
        p.flush();
        p.get_slot(1).unwrap().ddsketch.count()
    }));
    assert_eq!(res.ok(), Some(3));
}

/// `alpha ≥ 1`, `alpha < 0`, `NaN` and `inf` are accepted by
/// `MetricPipeline::new` and silently break the DDSketch contract
/// `q finite, q > 0, |q − v| ≤ alpha·v` for the single sample `v = 2`
/// (`alpha = 1`: every quantile is 0; `alpha = −0.5`: 3.0; `alpha > 1` /
/// `NaN`: NaN). The control row with a valid `alpha` keeps the predicate
/// honest. Reported with `pipeline-degenerate-config-panics`; not fixed
/// here because refusing an `alpha` changes the constructor's signature.
#[test]
fn degenerate_alpha_is_accepted_and_breaks_the_quantile_contract() {
    fn within_contract(q: f64, v: f64, alpha: f64) -> bool {
        q.is_finite() && q > 0.0 && (q - v).abs() <= alpha * v
    }
    let v = 2.0;
    // Control: alpha = 1/2 → bucket 1 → reported 1.0, |1 − 2| = 1 ≤ 0.5·2.
    let mut ok = pipeline::<2, 4>();
    ok.submit(MetricEvent::histogram(1, v));
    ok.flush();
    assert!(within_contract(
        ok.get_slot(1).unwrap().ddsketch.quantile(0.5),
        v,
        0.5
    ));

    for alpha in [1.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let res = catch_unwind(AssertUnwindSafe(|| {
            let mut p = MetricPipeline::<2, 4>::new(alpha);
            p.submit(MetricEvent::histogram(1, v));
            p.flush();
            let s = p.get_slot(1).unwrap();
            // Count / min / max / sum never depend on alpha.
            assert_eq!(s.ddsketch.count(), 1);
            assert_eq!(s.ddsketch.min(), v);
            assert_eq!(s.ddsketch.max(), v);
            s.ddsketch.quantile(0.5)
        }));
        let q = res.unwrap_or_else(|_| panic!("alpha = {alpha}: unexpected panic"));
        assert!(
            !within_contract(q, v, alpha),
            "alpha = {alpha}: quantile {q} unexpectedly satisfies the contract"
        );
    }
}

/// `MetricEvent::unique` carries the item hash as `f64`; above 2^53 two
/// distinct hashes round to the same value and land in the same register.
/// Pinned as a finding (fixing it changes the event layout).
#[test]
fn unique_item_hashes_above_2_pow_53_collapse_through_f64() {
    let h = 4u64;
    let m = 1024.0;
    let one_item = m * ln_m_over_m_minus_k(m, 1.0);
    let two_items = m * ln_m_over_m_minus_k(m, 2.0);
    // Exactly representable: 2^52 and 2^52 + 1 stay distinct and, after
    // splitmix64 mixing, land in two distinct registers.
    let mut fine = pipeline::<4, 8>();
    fine.submit(MetricEvent::unique(h, 1 << 52));
    fine.submit(MetricEvent::unique(h, (1 << 52) + 1));
    fine.flush();
    assert!(close(
        fine.get_slot(h).unwrap().hll.cardinality(),
        two_items,
        1e-9
    ));
    // Not representable: 2^53 + 1 rounds to 2^53.
    let mut lost = pipeline::<4, 8>();
    lost.submit(MetricEvent::unique(h, 1 << 53));
    lost.submit(MetricEvent::unique(h, (1 << 53) + 1));
    lost.flush();
    assert!(
        close(lost.get_slot(h).unwrap().hll.cardinality(), one_item, 1e-9),
        "2^53 + 1 collapsed onto 2^53"
    );
    // u64::MAX and u64::MAX − 1 both saturate back to u64::MAX.
    let mut top = pipeline::<4, 8>();
    top.submit(MetricEvent::unique(h, u64::MAX));
    top.submit(MetricEvent::unique(h, u64::MAX - 1));
    top.flush();
    assert!(close(
        top.get_slot(h).unwrap().hll.cardinality(),
        one_item,
        1e-9
    ));
    assert_eq!(top.get_slot(h).unwrap().event_count, 2);
}

#[test]
fn counts_at_extremes_do_not_panic_and_follow_ieee_and_u64_rules() {
    let h = 6u64;
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut p = pipeline::<4, 8>();
        // Counter overflow in f64 is +inf by IEEE 754, not a panic.
        p.submit(MetricEvent::counter(h, f64::MAX));
        p.submit(MetricEvent::counter(h, f64::MAX));
        p.submit(MetricEvent::gauge(h, f64::MIN).with_timestamp(u64::MAX));
        p.submit(MetricEvent::histogram(h, f64::MAX));
        p.submit(MetricEvent::histogram(h, f64::MIN_POSITIVE));
        p.flush();
        let s = p.get_slot(h).unwrap();
        assert_eq!(s.counter, f64::INFINITY);
        assert_eq!(s.gauge, f64::MIN);
        assert_eq!(s.last_update, u64::MAX);
        assert_eq!(s.event_count, 5);
        assert_eq!(s.ddsketch.count(), 2);
        assert_eq!(s.ddsketch.max(), f64::MAX);
        assert_eq!(s.ddsketch.min(), f64::MIN_POSITIVE);
        // A later, smaller stamp does not move last_update back from u64::MAX.
        p.submit(MetricEvent::gauge(h, 1.0).with_timestamp(1));
        p.flush();
        assert_eq!(p.get_slot(h).unwrap().last_update, u64::MAX);
        assert_eq!(p.total_events(), 6);
    }));
    assert!(res.is_ok(), "extreme values must not panic");
}
