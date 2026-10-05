//! Streaming Window Counts Example
//!
//! Production entry point for the occupancy accessors of the streaming
//! analytics types: `StreamingMedian::count` and `MadDetector::count`
//! (`src/anomaly.rs`), `RingBuffer::is_empty` and `MetricRegistry::count`
//! (`src/pipeline.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - a window of `W = 100` counts `min(n, W)` samples after `n` pushes; NaN is
//!   rejected and does not count; `clear` returns it to zero
//! - the median of the integers `a..=b` is `(a + b) / 2`, and after `n > W`
//!   pushes of `1..=n` the window holds `n − W + 1..=n`
//! - the MAD of `1..=2j+1` is the median of `|i − (j + 1)|`, which for `j = 4`
//!   is 2 (the sorted deviations are `0 1 1 2 2 3 3 4 4`)
//! - a ring buffer of `N` slots holds `N − 1` items, is empty exactly when every
//!   pushed item has been popped, and pops in push order across the wrap
//! - a registry of `N` slots counts each accepted registration and refuses the
//!   `N + 1`-th
//!
//! Run with: `cargo run --example streaming_window_counts`

use alice_physics::anomaly::{MadDetector, StreamingMedian};
use alice_physics::pipeline::{MetricRegistry, MetricType, RingBuffer};

const WINDOW: usize = 100;

fn streaming_median() {
    let mut m = StreamingMedian::new();
    assert_eq!(m.count(), 0, "a new window is empty");
    for n in 1..=150usize {
        m.push(n as f64);
        assert_eq!(m.count(), n.min(WINDOW), "count after {n} pushes");
        if n == 9 || n == 100 || n == 150 {
            let lo = n.saturating_sub(WINDOW) + 1;
            let want = (lo + n) as f64 / 2.0;
            assert_eq!(m.median(), want, "median of {lo}..={n}");
            println!(
                "StreamingMedian after {n:>3}: count {:>3}, median {}",
                m.count(),
                m.median()
            );
        }
    }
    m.push(f64::NAN);
    assert_eq!(m.count(), WINDOW, "NaN is rejected");
    m.clear();
    assert_eq!(m.count(), 0, "clear empties the window");
    m.push(f64::NAN);
    assert_eq!(
        m.count(),
        0,
        "NaN into an empty window still does not count"
    );
}

fn mad_detector() {
    let mut d = MadDetector::new(3.0);
    assert_eq!(d.count(), 0, "a new detector has no observations");
    for i in 1..=9 {
        d.observe(f64::from(i));
    }
    d.observe(f64::NAN);
    assert_eq!(d.count(), 9, "nine observations, NaN rejected");
    assert_eq!(d.median(), 5.0, "median of 1..=9");
    assert_eq!(d.mad(), 2.0, "MAD of 1..=9");
    for i in 10..=250 {
        d.observe(f64::from(i));
    }
    assert_eq!(d.count(), WINDOW, "the count saturates at the window");
    println!(
        "MadDetector: count {} after 250 observations, window median {}",
        d.count(),
        d.median()
    );
    assert_eq!(d.median(), (151.0 + 250.0) / 2.0, "median of 151..=250");
    d.clear();
    assert_eq!(d.count(), 0, "clear forgets every observation");
}

fn ring_buffer() {
    let mut rb: RingBuffer<u32, 4> = RingBuffer::new();
    assert!(rb.is_empty(), "a new buffer is empty");
    for v in 1..=3 {
        assert!(rb.push(v), "slot for item {v}");
        assert!(!rb.is_empty(), "not empty after pushing {v}");
    }
    assert!(!rb.push(4), "four slots hold three items");
    assert_eq!(rb.pop(), Some(1));
    assert_eq!(rb.pop(), Some(2));
    assert!(!rb.is_empty(), "item 3 is still queued");
    // These two pushes wrap the write position past the end of the storage.
    assert!(rb.push(5));
    assert!(rb.push(6));
    let mut drained = Vec::new();
    while !rb.is_empty() {
        drained.push(rb.pop().expect("is_empty said there is an item"));
    }
    assert_eq!(drained, [3, 5, 6], "first in, first out across the wrap");
    assert_eq!(rb.pop(), None, "is_empty agrees with pop");
    println!("RingBuffer<_, 4>: drained {drained:?} in push order, then empty");
}

fn metric_registry() {
    let mut reg: MetricRegistry<3> = MetricRegistry::new();
    assert_eq!(reg.count(), 0, "a new registry is empty");
    let names = [
        ("steps", MetricType::Counter),
        ("bodies", MetricType::Gauge),
        ("step_ms", MetricType::Histogram),
    ];
    for (i, (name, kind)) in names.into_iter().enumerate() {
        assert!(reg.register(name, kind).is_some(), "slot for {name}");
        assert_eq!(reg.count(), i + 1, "count after registering {name}");
    }
    assert!(
        reg.register("pairs", MetricType::Unique).is_none(),
        "three slots refuse a fourth metric"
    );
    assert_eq!(reg.count(), 3, "a refused registration does not count");
    assert_eq!(reg.iter().count(), reg.count(), "count matches the entries");
    println!("MetricRegistry<3>: {} metrics, fourth refused", reg.count());
}

fn main() {
    streaming_median();
    mad_detector();
    ring_buffer();
    metric_registry();
    println!("all closed forms hold");
}
