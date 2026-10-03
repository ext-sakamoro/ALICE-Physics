//! Audit oracles for `pipeline`.
//!
//! Reference models (independent of the crate):
//!
//! * `RingBuffer<_, N>` behaves as a FIFO queue of capacity `N - 1` (one slot
//!   is kept empty): checked against `VecDeque` on a long pseudo-random
//!   push/pop sequence;
//! * counters add, gauges keep the last written value, `last_update` is the
//!   maximum timestamp seen, `event_count` counts every processed event;
//! * the quantile estimate of a `DDSketch` with accuracy `alpha` is within a
//!   relative `alpha` of the exact value.
#![allow(clippy::float_cmp, clippy::needless_range_loop)]

use alice_physics::pipeline::{
    MetricEntry, MetricEvent, MetricPipeline, MetricRegistry, MetricSlot, MetricSnapshot,
    MetricType, RingBuffer,
};
use alice_physics::sketch::{FnvHasher, Mergeable};
use std::collections::VecDeque;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

fn hash(s: &str) -> u64 {
    FnvHasher::hash_bytes(s.as_bytes())
}

// ---------------------------------------------------------------- RingBuffer

#[test]
fn ring_buffer_matches_a_fifo_model_of_capacity_n_minus_one() {
    const N: usize = 5;
    let mut rb = RingBuffer::<u32, N>::new();
    let mut model: VecDeque<u32> = VecDeque::new();
    let mut dropped = 0u64;
    let mut s = 1u64;
    let mut next = 0u32;
    for step in 0..20_000 {
        if lcg(&mut s) % 3 != 0 {
            let ok = rb.push(next);
            if model.len() < N - 1 {
                assert!(ok, "step {step}: push must succeed below capacity");
                model.push_back(next);
            } else {
                assert!(!ok, "step {step}: push must fail at capacity");
                dropped += 1;
            }
            next += 1;
        } else {
            assert_eq!(rb.pop(), model.pop_front(), "step {step}");
        }
        assert_eq!(rb.len(), model.len(), "step {step}");
        assert_eq!(rb.is_empty(), model.is_empty());
        assert_eq!(rb.is_full(), model.len() == N - 1);
        assert_eq!(rb.dropped(), dropped);
    }
    assert_eq!(rb.capacity(), N - 1);
}

#[test]
fn ring_buffer_clear_empties_resets_dropped_and_stays_usable() {
    let mut rb = RingBuffer::<u8, 3>::new();
    assert!(rb.push(1));
    assert!(rb.push(2));
    assert!(!rb.push(3));
    assert_eq!(rb.dropped(), 1);
    rb.clear();
    assert!(rb.is_empty());
    assert_eq!(rb.len(), 0);
    assert_eq!(rb.dropped(), 0);
    assert_eq!(rb.pop(), None);
    assert!(rb.push(9));
    assert_eq!(rb.pop(), Some(9));
}

#[test]
fn ring_buffer_of_one_slot_has_zero_capacity() {
    let mut rb = RingBuffer::<u8, 1>::new();
    assert_eq!(rb.capacity(), 0);
    assert!(rb.is_full() && rb.is_empty());
    assert!(!rb.push(1));
    assert_eq!(rb.dropped(), 1);
    assert_eq!(rb.pop(), None);
}

// --------------------------------------------------------------- MetricEvent

#[test]
fn event_constructors_set_type_value_and_zero_timestamp() {
    let c = MetricEvent::counter(7, 2.5);
    assert_eq!(
        (c.name_hash, c.metric_type, c.value, c.timestamp),
        (7, MetricType::Counter, 2.5, 0)
    );
    let g = MetricEvent::gauge(8, -1.0);
    assert_eq!(
        (g.name_hash, g.metric_type, g.value, g.timestamp),
        (8, MetricType::Gauge, -1.0, 0)
    );
    let h = MetricEvent::histogram(9, 42.5);
    assert_eq!(
        (h.name_hash, h.metric_type, h.value, h.timestamp),
        (9, MetricType::Histogram, 42.5, 0)
    );
    let u = MetricEvent::unique(10, 12345);
    assert_eq!(
        (u.name_hash, u.metric_type, u.value, u.timestamp),
        (10, MetricType::Unique, 12345.0, 0)
    );
    assert_eq!(c.with_timestamp(99).timestamp, 99);
    assert_eq!(MetricEvent::default().metric_type, MetricType::Counter);
}

// --------------------------------------------------------------- MetricSlot

#[test]
fn slot_counts_events_sums_counters_keeps_last_gauge_and_max_timestamp() {
    let mut slot = MetricSlot::new(1, 0.05);
    slot.process(&MetricEvent::counter(1, 2.0).with_timestamp(10));
    slot.process(&MetricEvent::counter(1, 3.5).with_timestamp(5));
    slot.process(&MetricEvent::gauge(1, 7.0).with_timestamp(8));
    slot.process(&MetricEvent::gauge(1, -2.0).with_timestamp(9));
    slot.process(&MetricEvent::histogram(1, 10.0));
    slot.process(&MetricEvent::unique(1, 77));
    assert_eq!(slot.counter, 5.5);
    assert_eq!(slot.gauge, -2.0);
    assert_eq!(slot.event_count, 6);
    assert_eq!(slot.last_update, 10, "max timestamp, not the last one seen");
    assert_eq!(slot.ddsketch.count(), 1);
    assert_eq!(slot.name_hash, 1);
}

#[test]
fn slot_reset_zeroes_every_aggregate_but_keeps_the_name() {
    let mut slot = MetricSlot::new(3, 0.05);
    slot.process(&MetricEvent::counter(3, 2.0).with_timestamp(10));
    slot.process(&MetricEvent::gauge(3, 7.0));
    slot.process(&MetricEvent::histogram(3, 10.0));
    slot.process(&MetricEvent::unique(3, 5));
    slot.reset();
    assert_eq!(
        (slot.counter, slot.gauge, slot.event_count, slot.last_update),
        (0.0, 0.0, 0, 0)
    );
    assert_eq!(slot.ddsketch.count(), 0);
    assert_eq!(slot.hll.cardinality(), 0.0);
    assert_eq!(slot.name_hash, 3);
}

#[test]
fn slot_merge_adds_counters_and_event_counts_and_merges_histograms() {
    let mut a = MetricSlot::new(1, 0.05);
    let mut b = MetricSlot::new(1, 0.05);
    a.process(&MetricEvent::counter(1, 2.0));
    b.process(&MetricEvent::counter(1, 5.0));
    for v in [1.0, 2.0, 3.0] {
        a.process(&MetricEvent::histogram(1, v));
    }
    for v in [4.0, 5.0] {
        b.process(&MetricEvent::histogram(1, v));
    }
    a.merge(&b);
    assert_eq!(a.counter, 7.0);
    assert_eq!(a.event_count, 7);
    assert_eq!(a.ddsketch.count(), 5);
    assert_eq!(a.ddsketch.max(), 5.0);
}

#[test]
fn slot_merge_takes_the_newer_gauge() {
    let mut a = MetricSlot::new(1, 0.05);
    let mut b = MetricSlot::new(1, 0.05);
    a.process(&MetricEvent::gauge(1, 1.0).with_timestamp(10));
    b.process(&MetricEvent::gauge(1, 2.0).with_timestamp(20));
    a.merge(&b);
    assert_eq!(a.gauge, 2.0);
    assert_eq!(a.last_update, 20);
    let mut c = MetricSlot::new(1, 0.05);
    let mut d = MetricSlot::new(1, 0.05);
    c.process(&MetricEvent::gauge(1, 1.0).with_timestamp(30));
    d.process(&MetricEvent::gauge(1, 2.0).with_timestamp(20));
    c.merge(&d);
    assert_eq!(c.gauge, 1.0);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-015: MetricSlot::merge overwrites the gauge with the other slot's value whenever that slot has a newer timestamp, even if it never wrote a gauge (self gauge 100 at t=10, other only a counter at t=20: merged gauge 0.0)"]
fn slot_merge_keeps_the_gauge_when_the_other_slot_never_wrote_one() {
    let mut a = MetricSlot::new(1, 0.05);
    let mut b = MetricSlot::new(1, 0.05);
    a.process(&MetricEvent::gauge(1, 100.0).with_timestamp(10));
    b.process(&MetricEvent::counter(1, 1.0).with_timestamp(20));
    a.merge(&b);
    assert_eq!(a.gauge, 100.0);
}

// ------------------------------------------------------------ MetricPipeline

#[test]
fn pipeline_queue_accepts_n_minus_one_events_then_drops() {
    let mut p = MetricPipeline::<8, 4>::new(0.05);
    for i in 0..3 {
        assert!(p.submit(MetricEvent::counter(1, f64::from(i))), "event {i}");
    }
    assert!(!p.submit(MetricEvent::counter(1, 9.0)));
    assert!(!p.submit(MetricEvent::counter(1, 9.0)));
    assert_eq!(p.queue_len(), 3);
    assert_eq!(p.dropped_events(), 2);
    assert_eq!(p.total_events(), 0, "nothing processed before flush");
    p.flush();
    assert_eq!(p.queue_len(), 0);
    assert_eq!(p.total_events(), 3);
    assert_eq!(p.get_slot(1).unwrap().counter, 0.0 + 1.0 + 2.0);
}

#[test]
fn pipeline_routes_each_metric_to_its_own_slot_and_get_slot_is_exact() {
    let mut p = MetricPipeline::<64, 64>::new(0.05);
    let (a, b) = (hash("a"), hash("b"));
    assert_ne!(
        a as usize % 64,
        b as usize % 64,
        "test premise: no slot collision"
    );
    p.submit(MetricEvent::counter(a, 1.0));
    p.submit(MetricEvent::counter(b, 10.0));
    p.submit(MetricEvent::counter(a, 1.0));
    p.flush();
    assert_eq!(p.get_slot(a).unwrap().counter, 2.0);
    assert_eq!(p.get_slot(b).unwrap().counter, 10.0);
    assert!(p.get_slot(hash("never")).is_none());
    assert_eq!(p.iter_slots().count(), 2);
    assert_eq!(p.total_events(), 3);
    p.get_slot_mut(a).unwrap().counter = 100.0;
    assert_eq!(p.get_slot(a).unwrap().counter, 100.0);
    assert!(p.get_slot_mut(hash("never")).is_none());
}

#[test]
fn pipeline_reset_clears_slots_queue_and_totals() {
    let mut p = MetricPipeline::<8, 4>::new(0.05);
    p.submit(MetricEvent::counter(1, 2.0));
    p.flush();
    for _ in 0..5 {
        p.submit(MetricEvent::counter(1, 1.0));
    }
    assert!(p.dropped_events() > 0 && p.queue_len() > 0);
    p.reset();
    assert_eq!(
        (p.queue_len(), p.dropped_events(), p.total_events()),
        (0, 0, 0)
    );
    assert_eq!(p.get_slot(1).map_or(0.0, |s| s.counter), 0.0);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-016: two metrics whose hashes agree modulo SLOTS share one slot: the second metric's events are folded into the first (counter 2.0 instead of 1.0) and get_slot(second) returns None although its event was consumed (hash 1 and 1 + 8, SLOTS = 8)"]
fn pipeline_colliding_metric_names_do_not_corrupt_each_other() {
    let mut p = MetricPipeline::<8, 16>::new(0.05);
    let (a, b) = (1u64, 1u64 + 8);
    p.submit(MetricEvent::counter(a, 1.0));
    p.submit(MetricEvent::counter(b, 1.0));
    p.flush();
    assert_eq!(p.get_slot(a).map(|s| s.counter), Some(1.0));
    assert_eq!(p.get_slot(b).map(|s| s.counter), Some(1.0));
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-017: a Unique event stores the item hash as f64 and casts back to u64, which zeroes the low 11 bits of a 64-bit hash; HyperLogLog picks its register from the low 10 bits, so every item lands in register 0 (4000 distinct FNV hashes: direct insert estimates 4055, through the pipeline 54)"]
fn pipeline_unique_events_count_distinct_64_bit_hashes() {
    let n = 4000u64;
    let mut direct = alice_physics::sketch::HyperLogLog10::new();
    let mut p = MetricPipeline::<8, 16>::new(0.05);
    let name = hash("users");
    for i in 0..n {
        let item = FnvHasher::hash_u64(i);
        direct.insert_hash(item);
        assert!(p.submit(MetricEvent::unique(name, item)));
        p.flush();
    }
    let want = direct.cardinality();
    assert!(
        (want - n as f64).abs() < 0.1 * n as f64,
        "premise: direct estimate {want}"
    );
    let got = p.get_slot(name).unwrap().hll.cardinality();
    assert!(
        (got - want).abs() < 0.1 * want,
        "pipeline estimate {got} vs direct {want}"
    );
}

// ---------------------------------------------------------------- registry

#[test]
fn registry_register_lookup_and_capacity() {
    let mut r = MetricRegistry::<3>::new();
    assert_eq!(r.count(), 0);
    let h1 = r.register("http.requests", MetricType::Counter).unwrap();
    let h2 = r.register("queue.depth", MetricType::Gauge).unwrap();
    assert_eq!(h1, hash("http.requests"));
    assert_eq!(h2, hash("queue.depth"));
    assert_eq!(r.count(), 2);
    let e = r.lookup("queue.depth").unwrap();
    assert_eq!(
        (e.hash, e.metric_type, e.name_str()),
        (h2, MetricType::Gauge, "queue.depth")
    );
    assert_eq!(r.lookup_by_hash(h1).unwrap().name_str(), "http.requests");
    assert!(r.lookup("absent").is_none());
    assert!(r.lookup_by_hash(0).is_none());
    assert!(r.register("c", MetricType::Unique).is_some());
    assert!(r.register("d", MetricType::Unique).is_none(), "full");
    assert_eq!(r.count(), 3);
    assert_eq!(r.iter().count(), 3);
    let names: Vec<&str> = r.iter().map(MetricEntry::name_str).collect();
    assert_eq!(names, ["http.requests", "queue.depth", "c"]);
}

#[test]
fn entry_truncates_long_names_to_64_bytes_but_hashes_the_full_name() {
    let long = "x".repeat(100);
    let e = MetricEntry::new(&long, MetricType::Counter);
    assert_eq!(e.name_len, 64);
    assert_eq!(e.name_str(), "x".repeat(64));
    assert_eq!(e.hash, hash(&long));
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-018: MetricEntry::new truncates the name at byte 64 even inside a multi-byte character, and name_str() then returns the empty string (63 'a' + 'e-acute': name_str() == \"\" instead of 63 'a')"]
fn entry_name_truncated_inside_a_multibyte_character_is_not_lost() {
    let name = format!("{}\u{e9}", "a".repeat(63));
    let e = MetricEntry::new(&name, MetricType::Counter);
    assert_eq!(e.name_str(), "a".repeat(63));
}

#[test]
fn entry_short_name_round_trips() {
    let e = MetricEntry::new("lat\u{e9}ncy", MetricType::Histogram);
    assert_eq!(e.name_str(), "lat\u{e9}ncy");
    assert_eq!(e.metric_type, MetricType::Histogram);
}

// ---------------------------------------------------------------- snapshot

#[test]
fn snapshot_reports_counter_gauge_counts_and_quantiles_within_alpha() {
    let alpha = 0.05;
    let mut slot = MetricSlot::new(5, alpha);
    slot.process(&MetricEvent::counter(5, 3.0));
    slot.process(&MetricEvent::gauge(5, 4.0));
    for i in 1..=1000 {
        slot.process(&MetricEvent::histogram(5, f64::from(i)));
    }
    let snap = MetricSnapshot::from(&slot);
    assert_eq!(snap.name_hash, 5);
    assert_eq!(snap.counter, 3.0);
    assert_eq!(snap.gauge, 4.0);
    assert_eq!(snap.event_count, 1002);
    assert_eq!(snap.min, 1.0);
    assert_eq!(snap.max, 1000.0);
    assert!((snap.mean - 500.5).abs() < 1e-9);
    for (q, exact) in [(snap.p50, 500.0), (snap.p95, 950.0), (snap.p99, 990.0)] {
        assert!(
            (q - exact).abs() / exact <= 2.0 * alpha / (1.0 + alpha) + 1e-9,
            "{q} vs {exact}"
        );
    }
    assert!(snap.p50 <= snap.p95 && snap.p95 <= snap.p99);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-019: MetricPipeline::new documents alpha = 0.01 as a normal choice, but a 256-bin sketch at that accuracy only holds values up to about 47; larger observations are counted but stored in no bin, so quantiles fall back to the maximum (values 1..=1000: p50 reports 1000 instead of 500)"]
fn pipeline_histogram_with_alpha_one_percent_keeps_quantiles_for_values_above_fifty() {
    let mut p = MetricPipeline::<8, 2048>::new(0.01);
    let h = hash("latency");
    for i in 1..=1000 {
        assert!(p.submit(MetricEvent::histogram(h, f64::from(i))));
        p.flush();
    }
    let snap = MetricSnapshot::from(p.get_slot(h).unwrap());
    let bound = 2.0 * 0.01 / 1.01;
    assert!(
        (snap.p50 - 500.0).abs() / 500.0 <= bound + 1e-9,
        "p50 {}",
        snap.p50
    );
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-020: a snapshot of a slot without histogram data reports min = +inf and max = -inf while mean and the quantiles report 0.0"]
fn snapshot_of_a_slot_without_histogram_data_has_finite_extrema() {
    let mut slot = MetricSlot::new(1, 0.05);
    slot.process(&MetricEvent::counter(1, 1.0));
    let snap = MetricSnapshot::from(&slot);
    assert!(
        snap.min.is_finite() && snap.max.is_finite(),
        "min {} max {}",
        snap.min,
        snap.max
    );
}

#[test]
fn slot_merge_unions_the_unique_sets() {
    // Distinct small item hashes: slot a sees 0..600, slot b sees 400..1000.
    let mut a = MetricSlot::new(1, 0.05);
    let mut b = MetricSlot::new(1, 0.05);
    let mut want = alice_physics::sketch::HyperLogLog10::new();
    for i in 0..600u64 {
        a.process(&MetricEvent::unique(1, i));
    }
    for i in 400..1000u64 {
        b.process(&MetricEvent::unique(1, i));
    }
    for i in 0..1000u64 {
        want.insert_hash(i);
    }
    let alone = a.hll.cardinality();
    a.merge(&b);
    assert_eq!(a.hll.cardinality(), want.cardinality());
    assert!(a.hll.cardinality() > alone);
}

#[test]
fn snapshot_cardinality_comes_from_the_unique_set() {
    let mut slot = MetricSlot::new(1, 0.05);
    slot.process(&MetricEvent::counter(1, 7.0));
    let mut want = alice_physics::sketch::HyperLogLog10::new();
    for i in 0..300u64 {
        slot.process(&MetricEvent::unique(1, i));
        want.insert_hash(i);
    }
    let snap = MetricSnapshot::from(&slot);
    assert_eq!(snap.cardinality, want.cardinality());
    assert!(snap.cardinality > 100.0, "{}", snap.cardinality);
    assert_ne!(snap.cardinality, snap.counter);
}

#[test]
fn snapshot_quantiles_follow_nearest_rank_on_a_three_cluster_sample() {
    // 100 values: 90 x 10, 6 x 100, 4 x 1000. Nearest-rank quantiles: p50 = 10, p95 = 100, p99 = 1000.
    let alpha = 0.05;
    let mut slot = MetricSlot::new(1, alpha);
    for (v, n) in [(10.0, 90), (100.0, 6), (1000.0, 4)] {
        for _ in 0..n {
            slot.process(&MetricEvent::histogram(1, v));
        }
    }
    let snap = MetricSnapshot::from(&slot);
    let bound = 2.0 * alpha / (1.0 + alpha) + 1e-9;
    for (got, want) in [(snap.p50, 10.0), (snap.p95, 100.0), (snap.p99, 1000.0)] {
        assert!((got - want).abs() / want <= bound, "{got} vs {want}");
    }
}

#[test]
fn slot_merge_keeps_the_receivers_gauge_when_timestamps_are_equal() {
    // No rule is documented for a tie; the current behaviour (strictly newer wins) is pinned.
    let mut a = MetricSlot::new(1, 0.05);
    let mut b = MetricSlot::new(1, 0.05);
    a.process(&MetricEvent::gauge(1, 1.0));
    b.process(&MetricEvent::gauge(1, 2.0));
    a.merge(&b);
    assert_eq!(a.gauge, 1.0);
}
