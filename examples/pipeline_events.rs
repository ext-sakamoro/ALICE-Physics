//! The metric pipeline end to end: staging ring buffer, event queue, named
//! registry, the four event kinds and the reset paths.
//!
//! Every count printed here is known before the code runs: `N` events into
//! a queue of capacity `C = QUEUE_SIZE − 1` leaves `min(N, C)` queued and
//! `max(N − C, 0)` dropped, a flush processes exactly the queued ones, and
//! `total_events + dropped_events == N` afterwards. The histogram uses
//! `alpha = 0.5` (`gamma = 3`) so the bucket of every sample is a power of
//! three that can be read off by hand, and the unique counter is fed a
//! multiset whose cardinality is the size of its underlying set.
//!
//! ```bash
//! cargo run --example pipeline_events --features std
//! ```

use alice_physics::pipeline::{
    MetricEvent, MetricPipeline, MetricRegistry, MetricSnapshot, MetricType, RingBuffer,
};

// 64 slots: the four registry hashes below land in buckets 33 / 35 / 63 / 19.
// (With 16 slots `queue.depth` and `user.ids` both map to bucket 3 and the
// pipeline folds the second into the first — the documented approximation.)
const SLOTS: usize = 64;
const QUEUE: usize = 8;

fn main() {
    // ---- 1. staging ring buffer: capacity, is_full, dropped ---------------
    let mut staging = RingBuffer::<MetricEvent, QUEUE>::new();
    let capacity = staging.capacity();
    println!("[pipeline] ring buffer QUEUE_SIZE = {QUEUE}: capacity = {capacity} (closed form QUEUE_SIZE - 1 = {})", QUEUE - 1);
    let n = 10usize;
    let mut full_at = None;
    for i in 0..n {
        staging.push(MetricEvent::counter(1, 1.0).with_timestamp(i as u64));
        if full_at.is_none() && staging.is_full() {
            full_at = Some(i + 1);
        }
    }
    println!(
        "[pipeline] pushed {n}: is_full first true after {:?} pushes (expected {capacity}), len = {} (expected {}), dropped = {} (expected {})",
        full_at,
        staging.len(),
        n.min(capacity),
        staging.dropped(),
        n.saturating_sub(capacity)
    );
    let drained = std::iter::from_fn(|| staging.pop()).count();
    println!(
        "[pipeline] drained {drained} events (expected {capacity}), is_full now {}",
        staging.is_full()
    );

    // ---- 2. named registry: lookup by name and by hash agree --------------
    let mut registry = MetricRegistry::<8>::new();
    let names = [
        ("http.requests", MetricType::Counter),
        ("queue.depth", MetricType::Gauge),
        ("http.latency_ms", MetricType::Histogram),
        ("user.ids", MetricType::Unique),
    ];
    let mut hashes = Vec::new();
    for (name, kind) in names {
        hashes.push(registry.register(name, kind).expect("registry has room"));
    }
    for (name, _) in names {
        let by_name = registry.lookup(name).expect("registered");
        let by_hash = registry.lookup_by_hash(by_name.hash).expect("registered");
        println!(
            "[pipeline] registry: lookup({name:?}).name_str = {:?}, lookup_by_hash(0x{:016x}).name_str = {:?}, same entry = {}",
            by_name.name_str(),
            by_name.hash,
            by_hash.name_str(),
            std::ptr::eq(by_name, by_hash)
        );
    }
    let order: Vec<&str> = registry.iter().map(|e| e.name_str()).collect();
    println!("[pipeline] registry iter order = {order:?} (registration order)");
    println!(
        "[pipeline] registry lookup of an unregistered name is None: {}, of hash 0 is None: {}",
        registry.lookup("not.registered").is_none(),
        registry.lookup_by_hash(0).is_none()
    );
    let (h_req, h_depth, h_lat, h_ids) = (hashes[0], hashes[1], hashes[2], hashes[3]);
    let mut buckets: Vec<usize> = hashes.iter().map(|h| (*h as usize) % SLOTS).collect();
    println!("[pipeline] slot buckets (hash % {SLOTS}) = {buckets:?}");
    buckets.sort_unstable();
    buckets.dedup();
    assert_eq!(
        buckets.len(),
        hashes.len(),
        "the four metrics must occupy distinct slots for the counts below to be per metric"
    );

    // ---- 3. queue accounting: submit N into capacity C -------------------
    let mut p = MetricPipeline::<SLOTS, QUEUE>::new(0.5);
    let n = 10u64;
    for i in 0..n {
        p.submit(MetricEvent::counter(h_req, 1.0).with_timestamp(1_000 + i));
    }
    println!(
        "[pipeline] submit {n} into capacity {capacity}: queue_len = {} (expected {}), dropped_events = {} (expected {})",
        p.queue_len(),
        (n as usize).min(capacity),
        p.dropped_events(),
        n.saturating_sub(capacity as u64)
    );
    p.flush();
    println!(
        "[pipeline] after flush: queue_len = {}, total_events = {} (expected {}), total + dropped = {} (expected {n})",
        p.queue_len(),
        p.total_events(),
        (n as usize).min(capacity),
        p.total_events() + p.dropped_events()
    );

    // ---- 4. with_timestamp: the stamp is what the slot records ----------
    let stamped = MetricEvent::gauge(h_depth, 3.0).with_timestamp(1_700_000_000_000);
    println!(
        "[pipeline] with_timestamp: default ts = {}, stamped ts = {} (differs: {})",
        MetricEvent::gauge(h_depth, 3.0).timestamp,
        stamped.timestamp,
        stamped.timestamp != MetricEvent::gauge(h_depth, 3.0).timestamp
    );
    p.submit(stamped);
    p.submit(MetricEvent::gauge(h_depth, 7.0).with_timestamp(1_699_999_999_999));
    p.flush();
    let depth = p.get_slot(h_depth).expect("gauge slot");
    println!(
        "[pipeline] gauge slot: gauge = {} (last value wins: 7), last_update = {} (max of the stamps)",
        depth.gauge, depth.last_update
    );

    // ---- 5. histogram with gamma = 3: buckets by hand --------------------
    // alpha = 0.5 → gamma = 3; sample v lands in bucket ceil(log3 v) and
    // quantile() reports that bucket's lower bound 3^(ceil(log3 v) − 1).
    //   2 → bucket 1 → 1;  4, 5, 8 → bucket 2 → 3;  10 → bucket 3 → 9;  100 → bucket 5 → 81
    let sample = [2.0, 4.0, 5.0, 8.0, 10.0, 100.0];
    for v in sample {
        p.submit(MetricEvent::histogram(h_lat, v));
    }
    p.flush();
    let lat = p.get_slot(h_lat).expect("histogram slot");
    let snap = MetricSnapshot::from(lat);
    println!(
        "[pipeline] histogram {sample:?}: count = {} (6), min = {} (2), max = {} (100), mean = {} (21.5)",
        lat.ddsketch.count(),
        snap.min,
        snap.max,
        snap.mean
    );
    println!(
        "[pipeline] histogram quantiles: p50 = {:.6} (rank 3 → bucket 2 → 3), p95 = {:.6} (rank 6 → bucket 5 → 81), p99 = {:.6} (81)",
        snap.p50, snap.p95, snap.p99
    );

    // ---- 6. unique: a multiset counts as its set --------------------------
    // Items 1..=5 inserted 1, 2, 3, 4, 5 times (15 events, 5 distinct),
    // flushed per item so no batch exceeds the queue capacity of 7.
    for item in 1..=5u64 {
        for _ in 0..item {
            assert!(p.submit(MetricEvent::unique(h_ids, item)));
        }
        p.flush();
    }
    let ids = p.get_slot(h_ids).expect("unique slot");
    println!(
        "[pipeline] unique: 15 events over 5 distinct items → event_count = {} (15), cardinality = {:.4} (linear counting 1024·ln(1024/1019) = 5.0122)",
        ids.event_count,
        ids.hll.cardinality()
    );

    // ---- 7. slot registry: get_slot / get_slot_mut / iter_slots ----------
    let active: Vec<u64> = p.iter_slots().map(|s| s.name_hash).collect();
    println!(
        "[pipeline] iter_slots yields {} slots (4 metrics registered), in bucket order (hash % {SLOTS}): {:?}",
        active.len(),
        active.iter().map(|h| (*h as usize) % SLOTS).collect::<Vec<_>>()
    );
    {
        let req = p.get_slot_mut(h_req).expect("counter slot");
        println!(
            "[pipeline] counter slot before per-slot reset: counter = {} (7 accepted of 10)",
            req.counter
        );
        req.reset();
    }
    let req = p.get_slot(h_req).expect("counter slot still registered");
    println!(
        "[pipeline] after MetricSlot::reset: counter = {} (0), event_count = {} (0), slot still listed = {}",
        req.counter,
        req.event_count,
        p.iter_slots().any(|s| s.name_hash == h_req)
    );
    println!(
        "[pipeline] get_slot of an unregistered hash is None: {}",
        p.get_slot(0xDEAD_BEEF).is_none()
    );

    // ---- 8. pipeline reset: observables back to fresh --------------------
    p.submit(MetricEvent::counter(h_req, 1.0));
    p.reset();
    println!(
        "[pipeline] after MetricPipeline::reset: queue_len = {}, dropped_events = {}, total_events = {}, active slots = {} (slots stay registered, contents zeroed)",
        p.queue_len(),
        p.dropped_events(),
        p.total_events(),
        p.iter_slots().count()
    );

    // ---- 9. capacity 0: QUEUE_SIZE = 1 drops everything -----------------
    let mut tiny = MetricPipeline::<4, 1>::new(0.5);
    let accepted = tiny.submit(MetricEvent::counter(h_req, 1.0));
    println!(
        "[pipeline] QUEUE_SIZE = 1 (capacity 0): submit accepted = {accepted} (false), dropped_events = {} (1), queue_len = {} (0)",
        tiny.dropped_events(),
        tiny.queue_len()
    );
}
