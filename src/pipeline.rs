//! Lock-Free Metric Aggregation Pipeline
//!
//! Zero-allocation metric collection into pre-allocated sketches.
//! Designed for high-throughput telemetry without GC pauses.
//!
//! # Examples
//!
//! ```
//! use alice_physics::pipeline::{MetricPipeline, MetricEvent};
//! use alice_physics::sketch::FnvHasher;
//!
//! // Create pipeline with 64 slots, 1024-event queue
//! let mut pipeline = MetricPipeline::<64, 1024>::new(0.05);
//!
//! // Submit metrics using name hashes
//! let latency_hash = FnvHasher::hash_bytes(b"latency");
//! pipeline.submit(MetricEvent::histogram(latency_hash, 42.5));
//! pipeline.submit(MetricEvent::histogram(latency_hash, 38.1));
//! pipeline.submit(MetricEvent::counter(latency_hash, 1.0));
//!
//! // Process pending events
//! pipeline.flush();
//! assert!(pipeline.total_events() >= 3);
//! ```

use crate::sketch::{DDSketch256, HyperLogLog10, Mergeable};

// ============================================================================
// Ring Buffer for Hot Metrics
// ============================================================================

/// Lock-free single-producer single-consumer ring buffer
///
/// Used for pushing hot metrics from producer to consumer without locks.
#[derive(Debug)]
pub struct RingBuffer<T: Copy + Default, const N: usize> {
    /// Buffer storage
    buffer: [T; N],
    /// Write position (only modified by producer)
    write_pos: usize,
    /// Read position (only modified by consumer)
    read_pos: usize,
    /// Count of dropped items due to full buffer
    dropped: u64,
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    /// Create a new empty ring buffer
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: [T::default(); N],
            write_pos: 0,
            read_pos: 0,
            dropped: 0,
        }
    }

    /// Push an item (returns false if buffer is full)
    #[inline]
    pub fn push(&mut self, item: T) -> bool {
        let next_write = (self.write_pos + 1) % N;
        if next_write == self.read_pos {
            // Buffer full
            self.dropped += 1;
            return false;
        }
        self.buffer[self.write_pos] = item;
        self.write_pos = next_write;
        true
    }

    /// Pop an item (returns None if buffer is empty)
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.read_pos == self.write_pos {
            // Buffer empty
            return None;
        }
        let item = self.buffer[self.read_pos];
        self.read_pos = (self.read_pos + 1) % N;
        Some(item)
    }

    /// Check if buffer is empty
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.read_pos == self.write_pos
    }

    /// Check if buffer is full
    #[inline]
    pub const fn is_full(&self) -> bool {
        (self.write_pos + 1) % N == self.read_pos
    }

    /// Get current length
    #[inline]
    pub const fn len(&self) -> usize {
        if self.write_pos >= self.read_pos {
            self.write_pos - self.read_pos
        } else {
            N - self.read_pos + self.write_pos
        }
    }

    /// Get capacity
    #[inline]
    pub const fn capacity(&self) -> usize {
        N - 1 // One slot is always empty to distinguish full from empty
    }

    /// Get count of dropped items
    #[inline]
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        while self.pop().is_some() {}
        self.dropped = 0;
    }
}

impl<T: Copy + Default, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Metric Event Types
// ============================================================================

/// Type of metric event
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MetricType {
    /// Counter increment
    #[default]
    Counter,
    /// Gauge value (current state)
    Gauge,
    /// Histogram observation (latency, size, etc.)
    Histogram,
    /// Unique item (for cardinality)
    Unique,
}

/// A metric event to be processed
#[derive(Clone, Copy, Debug, Default)]
pub struct MetricEvent {
    /// Metric name hash
    pub name_hash: u64,
    /// Metric type
    pub metric_type: MetricType,
    /// Value (interpretation depends on type)
    pub value: f64,
    /// Timestamp (optional, epoch millis)
    pub timestamp: u64,
}

impl MetricEvent {
    /// Create a counter increment event
    #[inline]
    #[must_use]
    pub const fn counter(name_hash: u64, delta: f64) -> Self {
        Self {
            name_hash,
            metric_type: MetricType::Counter,
            value: delta,
            timestamp: 0,
        }
    }

    /// Create a gauge event
    #[inline]
    #[must_use]
    pub const fn gauge(name_hash: u64, value: f64) -> Self {
        Self {
            name_hash,
            metric_type: MetricType::Gauge,
            value,
            timestamp: 0,
        }
    }

    /// Create a histogram observation
    #[inline]
    #[must_use]
    pub const fn histogram(name_hash: u64, value: f64) -> Self {
        Self {
            name_hash,
            metric_type: MetricType::Histogram,
            value,
            timestamp: 0,
        }
    }

    /// Create a unique item event
    #[inline]
    #[must_use]
    pub const fn unique(name_hash: u64, item_hash: u64) -> Self {
        Self {
            name_hash,
            metric_type: MetricType::Unique,
            value: item_hash as f64,
            timestamp: 0,
        }
    }

    /// Set timestamp
    #[inline]
    #[must_use]
    pub const fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }
}

// ============================================================================
// Metric Slot - Pre-allocated storage for a single metric
// ============================================================================

/// Pre-allocated storage for a metric's aggregated data
#[derive(Clone, Debug)]
pub struct MetricSlot {
    /// Metric name hash
    pub name_hash: u64,
    /// Counter value (for Counter type)
    pub counter: f64,
    /// Gauge value (for Gauge type)
    pub gauge: f64,
    /// `HyperLogLog` for unique counting (P=10, 1KB)
    pub hll: HyperLogLog10,
    /// `DDSketch` for histogram/quantiles (256 bins, use alpha >= 0.05)
    pub ddsketch: DDSketch256,
    /// Event count
    pub event_count: u64,
    /// Last update timestamp
    pub last_update: u64,
}

impl MetricSlot {
    /// Create a new metric slot
    ///
    /// Note: With 256 bins, use alpha >= 0.05 for best results
    #[must_use]
    pub fn new(name_hash: u64, alpha: f64) -> Self {
        Self {
            name_hash,
            counter: 0.0,
            gauge: 0.0,
            hll: HyperLogLog10::new(),
            ddsketch: DDSketch256::new(alpha),
            event_count: 0,
            last_update: 0,
        }
    }

    /// Process a metric event
    pub fn process(&mut self, event: &MetricEvent) {
        self.event_count += 1;
        if event.timestamp > self.last_update {
            self.last_update = event.timestamp;
        }

        match event.metric_type {
            MetricType::Counter => {
                self.counter += event.value;
            }
            MetricType::Gauge => {
                self.gauge = event.value;
            }
            MetricType::Histogram => {
                self.ddsketch.insert(event.value);
            }
            MetricType::Unique => {
                self.hll.insert_hash(event.value as u64);
            }
        }
    }

    /// Reset the slot
    pub fn reset(&mut self) {
        self.counter = 0.0;
        self.gauge = 0.0;
        self.hll.clear();
        self.ddsketch.clear();
        self.event_count = 0;
        self.last_update = 0;
    }
}

impl Mergeable for MetricSlot {
    fn merge(&mut self, other: &Self) {
        self.counter += other.counter;
        // For gauge, take the latest
        if other.last_update > self.last_update {
            self.gauge = other.gauge;
            self.last_update = other.last_update;
        }
        self.hll.merge(&other.hll);
        self.ddsketch.merge(&other.ddsketch);
        self.event_count += other.event_count;
    }
}

// ============================================================================
// Metric Pipeline - Main aggregation pipeline
// ============================================================================

/// High-performance metric aggregation pipeline
///
/// Features:
/// - Pre-allocated metric slots (zero runtime allocation)
/// - Ring buffer for async metric submission
/// - Mergeable sketches for distributed aggregation
///
/// # Example
/// ```
/// use alice_physics::pipeline::{MetricPipeline, MetricEvent};
///
/// // Create pipeline with 64 metric slots
/// let mut pipeline = MetricPipeline::<64, 1024>::new(0.05);
///
/// // Submit metrics
/// pipeline.submit(MetricEvent::counter(hash("requests"), 1.0));
/// pipeline.submit(MetricEvent::histogram(hash("latency"), 42.5));
///
/// // Process pending events
/// pipeline.flush();
///
/// fn hash(s: &str) -> u64 {
///     use alice_physics::sketch::FnvHasher;
///     FnvHasher::hash_bytes(s.as_bytes())
/// }
/// ```
#[derive(Debug)]
pub struct MetricPipeline<const SLOTS: usize, const QUEUE_SIZE: usize> {
    /// Pre-allocated metric slots
    slots: [Option<MetricSlot>; SLOTS],
    /// Event queue (ring buffer)
    queue: RingBuffer<MetricEvent, QUEUE_SIZE>,
    /// `DDSketch` alpha parameter
    alpha: f64,
    /// Total events processed
    total_events: u64,
}

impl<const SLOTS: usize, const QUEUE_SIZE: usize> MetricPipeline<SLOTS, QUEUE_SIZE> {
    /// Create a new metric pipeline
    ///
    /// # Arguments
    /// * `alpha` - Relative error for `DDSketch` (e.g., 0.01 for 1%)
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        // Use array initialization with Default
        const NONE_SLOT: Option<MetricSlot> = None;
        Self {
            slots: [NONE_SLOT; SLOTS],
            queue: RingBuffer::new(),
            alpha,
            total_events: 0,
        }
    }

    /// Submit a metric event to the queue
    ///
    /// Returns false if queue is full (event dropped)
    #[inline]
    pub fn submit(&mut self, event: MetricEvent) -> bool {
        self.queue.push(event)
    }

    /// Process all pending events in the queue
    pub fn flush(&mut self) {
        while let Some(event) = self.queue.pop() {
            self.process_event(event);
        }
    }

    /// Process a single event directly (bypasses queue)
    fn process_event(&mut self, event: MetricEvent) {
        self.total_events += 1;

        // Find or create slot for this metric
        let slot_idx = (event.name_hash as usize) % SLOTS;

        // Check if slot exists and matches
        if let Some(ref mut slot) = self.slots[slot_idx] {
            if slot.name_hash == event.name_hash {
                slot.process(&event);
                return;
            }
            // Hash collision - process into existing slot anyway (approximation)
            slot.process(&event);
        } else {
            // Create new slot
            let mut slot = MetricSlot::new(event.name_hash, self.alpha);
            slot.process(&event);
            self.slots[slot_idx] = Some(slot);
        }
    }

    /// Get a metric slot by name hash
    #[must_use]
    pub fn get_slot(&self, name_hash: u64) -> Option<&MetricSlot> {
        let slot_idx = (name_hash as usize) % SLOTS;
        self.slots[slot_idx]
            .as_ref()
            .filter(|s| s.name_hash == name_hash)
    }

    /// Get mutable reference to a metric slot
    pub fn get_slot_mut(&mut self, name_hash: u64) -> Option<&mut MetricSlot> {
        let slot_idx = (name_hash as usize) % SLOTS;
        self.slots[slot_idx]
            .as_mut()
            .filter(|s| s.name_hash == name_hash)
    }

    /// Iterate over all active slots
    pub fn iter_slots(&self) -> impl Iterator<Item = &MetricSlot> {
        self.slots.iter().filter_map(|s| s.as_ref())
    }

    /// Get total events processed
    #[inline]
    #[must_use]
    pub const fn total_events(&self) -> u64 {
        self.total_events
    }

    /// Get count of dropped events
    #[inline]
    #[must_use]
    pub const fn dropped_events(&self) -> u64 {
        self.queue.dropped()
    }

    /// Get queue length
    #[inline]
    #[must_use]
    pub const fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Reset all slots
    pub fn reset(&mut self) {
        for slot in self.slots.iter_mut().flatten() {
            slot.reset();
        }
        self.queue.clear();
        self.total_events = 0;
    }
}

// ============================================================================
// Metric Registry - Named metric management
// ============================================================================

/// Entry in the metric registry
#[derive(Clone, Debug)]
pub struct MetricEntry {
    /// Metric name (for display)
    pub name: [u8; 64],
    /// Name length
    pub name_len: usize,
    /// Name hash
    pub hash: u64,
    /// Metric type
    pub metric_type: MetricType,
}

impl MetricEntry {
    /// Create a new metric entry
    #[must_use]
    pub fn new(name: &str, metric_type: MetricType) -> Self {
        use crate::sketch::FnvHasher;

        let mut name_buf = [0u8; 64];
        let len = name.len().min(64);
        name_buf[..len].copy_from_slice(&name.as_bytes()[..len]);

        Self {
            name: name_buf,
            name_len: len,
            hash: FnvHasher::hash_bytes(name.as_bytes()),
            metric_type,
        }
    }

    /// Get name as string slice
    #[must_use]
    pub fn name_str(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("")
    }
}

/// Registry of named metrics
#[derive(Clone, Debug)]
pub struct MetricRegistry<const N: usize> {
    /// Registered metrics
    entries: [Option<MetricEntry>; N],
    /// Count of registered metrics
    count: usize,
}

impl<const N: usize> MetricRegistry<N> {
    /// Create a new empty registry
    #[must_use]
    pub const fn new() -> Self {
        const NONE_ENTRY: Option<MetricEntry> = None;
        Self {
            entries: [NONE_ENTRY; N],
            count: 0,
        }
    }

    /// Register a new metric
    ///
    /// Returns the metric hash, or None if registry is full
    pub fn register(&mut self, name: &str, metric_type: MetricType) -> Option<u64> {
        if self.count >= N {
            return None;
        }

        let entry = MetricEntry::new(name, metric_type);
        let hash = entry.hash;

        // Find empty slot
        for slot in &mut self.entries {
            if slot.is_none() {
                *slot = Some(entry);
                self.count += 1;
                return Some(hash);
            }
        }

        None
    }

    /// Look up a metric by name
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&MetricEntry> {
        use crate::sketch::FnvHasher;
        let hash = FnvHasher::hash_bytes(name.as_bytes());
        self.lookup_by_hash(hash)
    }

    /// Look up a metric by hash
    #[must_use]
    pub fn lookup_by_hash(&self, hash: u64) -> Option<&MetricEntry> {
        self.entries
            .iter()
            .filter_map(|e| e.as_ref())
            .find(|e| e.hash == hash)
    }

    /// Iterate over registered metrics
    pub fn iter(&self) -> impl Iterator<Item = &MetricEntry> {
        self.entries.iter().filter_map(|e| e.as_ref())
    }

    /// Get count of registered metrics
    #[inline]
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }
}

impl<const N: usize> Default for MetricRegistry<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Snapshot - Point-in-time metric export
// ============================================================================

/// Snapshot of a single metric's state
#[derive(Clone, Debug)]
pub struct MetricSnapshot {
    /// Metric name hash
    pub name_hash: u64,
    /// Counter value
    pub counter: f64,
    /// Gauge value
    pub gauge: f64,
    /// Cardinality estimate (from HLL)
    pub cardinality: f64,
    /// P50 latency
    pub p50: f64,
    /// P95 latency
    pub p95: f64,
    /// P99 latency
    pub p99: f64,
    /// Mean value
    pub mean: f64,
    /// Min value
    pub min: f64,
    /// Max value
    pub max: f64,
    /// Event count
    pub event_count: u64,
}

impl From<&MetricSlot> for MetricSnapshot {
    fn from(slot: &MetricSlot) -> Self {
        Self {
            name_hash: slot.name_hash,
            counter: slot.counter,
            gauge: slot.gauge,
            cardinality: slot.hll.cardinality(),
            p50: slot.ddsketch.quantile(0.50),
            p95: slot.ddsketch.quantile(0.95),
            p99: slot.ddsketch.quantile(0.99),
            mean: slot.ddsketch.mean(),
            min: slot.ddsketch.min(),
            max: slot.ddsketch.max(),
            event_count: slot.event_count,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::sketch::FnvHasher;

    fn hash(s: &str) -> u64 {
        FnvHasher::hash_bytes(s.as_bytes())
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut rb = RingBuffer::<u32, 4>::new();

        assert!(rb.is_empty());
        assert!(!rb.is_full());
        assert_eq!(rb.capacity(), 3);

        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert!(!rb.push(4)); // Full

        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let mut rb = RingBuffer::<u32, 4>::new();

        rb.push(1);
        rb.push(2);
        assert_eq!(rb.pop(), Some(1));

        rb.push(3);
        rb.push(4);
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_metric_event() {
        let counter = MetricEvent::counter(hash("requests"), 1.0);
        assert_eq!(counter.metric_type, MetricType::Counter);
        assert_eq!(counter.value, 1.0);

        let hist = MetricEvent::histogram(hash("latency"), 42.5);
        assert_eq!(hist.metric_type, MetricType::Histogram);
        assert_eq!(hist.value, 42.5);
    }

    #[test]
    fn test_metric_slot() {
        let mut slot = MetricSlot::new(hash("test"), 0.05);

        slot.process(&MetricEvent::counter(hash("test"), 5.0));
        slot.process(&MetricEvent::counter(hash("test"), 3.0));
        assert_eq!(slot.counter, 8.0);

        slot.process(&MetricEvent::gauge(hash("test"), 100.0));
        assert_eq!(slot.gauge, 100.0);

        slot.process(&MetricEvent::histogram(hash("test"), 10.0));
        slot.process(&MetricEvent::histogram(hash("test"), 20.0));
        assert_eq!(slot.ddsketch.count(), 2);
    }

    #[test]
    fn test_metric_pipeline() {
        let mut pipeline = MetricPipeline::<64, 1024>::new(0.05);

        let req_hash = hash("http.requests");
        let lat_hash = hash("http.latency");

        // Submit events
        for _ in 0..100 {
            pipeline.submit(MetricEvent::counter(req_hash, 1.0));
        }
        for lat in [10.0, 20.0, 30.0, 40.0, 50.0] {
            pipeline.submit(MetricEvent::histogram(lat_hash, lat));
        }

        // Flush
        pipeline.flush();

        // Check results
        let req_slot = pipeline.get_slot(req_hash).unwrap();
        assert_eq!(req_slot.counter, 100.0);

        let lat_slot = pipeline.get_slot(lat_hash).unwrap();
        assert_eq!(lat_slot.ddsketch.count(), 5);
    }

    #[test]
    fn test_metric_registry() {
        let mut registry = MetricRegistry::<16>::new();

        let h1 = registry
            .register("http.requests", MetricType::Counter)
            .unwrap();
        let h2 = registry
            .register("http.latency", MetricType::Histogram)
            .unwrap();

        assert_ne!(h1, h2);
        assert_eq!(registry.count(), 2);

        let entry = registry.lookup("http.requests").unwrap();
        assert_eq!(entry.metric_type, MetricType::Counter);
    }

    #[test]
    fn test_metric_snapshot() {
        let mut slot = MetricSlot::new(hash("test"), 0.05);

        slot.process(&MetricEvent::counter(hash("test"), 100.0));
        for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
            slot.process(&MetricEvent::histogram(hash("test"), v));
        }

        let snapshot = MetricSnapshot::from(&slot);
        assert_eq!(snapshot.counter, 100.0);
        assert_eq!(snapshot.event_count, 11);
        assert!((snapshot.mean - 55.0).abs() < 1.0);
    }

    #[test]
    fn metric_event_with_timestamp_sets_only_the_timestamp() {
        let base = MetricEvent::gauge(hash("g"), 2.5);
        assert_eq!(base.timestamp, 0, "constructors start at ts = 0");

        let stamped = base.with_timestamp(1_700_000_000_123);
        assert_eq!(stamped.timestamp, 1_700_000_000_123);
        // Every other field is carried through unchanged.
        assert_eq!(stamped.name_hash, base.name_hash);
        assert_eq!(stamped.metric_type, MetricType::Gauge);
        assert!((stamped.value - 2.5).abs() < 1e-12);

        // Chaining overwrites: the last timestamp wins.
        let twice = stamped.with_timestamp(7);
        assert_eq!(twice.timestamp, 7);
        assert_eq!(
            MetricEvent::counter(1, 1.0)
                .with_timestamp(u64::MAX)
                .timestamp,
            u64::MAX
        );

        // The timestamp is what a slot records as last_update.
        let mut slot = MetricSlot::new(hash("g"), 0.05);
        slot.process(&MetricEvent::gauge(hash("g"), 1.0).with_timestamp(50));
        slot.process(&MetricEvent::gauge(hash("g"), 2.0).with_timestamp(20));
        assert_eq!(
            slot.last_update, 50,
            "last_update is the max timestamp seen"
        );
    }

    #[test]
    fn pipeline_queue_len_and_dropped_events_count_exactly() {
        // QUEUE_SIZE = 8 → ring capacity is 7 (one slot kept empty).
        let mut p = MetricPipeline::<4, 8>::new(0.05);
        assert_eq!(p.queue_len(), 0);
        assert_eq!(p.dropped_events(), 0);

        let h = hash("q");
        let mut accepted = 0u64;
        let mut rejected = 0u64;
        for i in 0..12 {
            if p.submit(MetricEvent::counter(h, 1.0).with_timestamp(i)) {
                accepted += 1;
            } else {
                rejected += 1;
            }
            // queue_len tracks accepted-but-unflushed events, capped at capacity
            assert_eq!(p.queue_len(), accepted as usize, "after submit #{i}");
            assert_eq!(p.dropped_events(), rejected, "after submit #{i}");
        }
        assert_eq!(accepted, 7, "capacity = QUEUE_SIZE - 1");
        assert_eq!(rejected, 5, "12 pushes over capacity 7 → 5 dropped");
        assert_eq!(p.queue_len(), 7);
        assert_eq!(p.dropped_events(), 5);

        // flush drains the queue; dropped is a lifetime counter and persists.
        p.flush();
        assert_eq!(p.queue_len(), 0);
        assert_eq!(p.dropped_events(), 5);
        assert_eq!(p.total_events(), 7);
        let slot = p.get_slot(h).expect("slot created by flush");
        assert_eq!(slot.event_count, 7);
        assert!((slot.counter - 7.0).abs() < 1e-12);

        // After flushing, the queue accepts again and drops accumulate.
        for _ in 0..9 {
            p.submit(MetricEvent::counter(h, 1.0));
        }
        assert_eq!(p.queue_len(), 7);
        assert_eq!(p.dropped_events(), 7);

        // reset clears both the queue and the dropped counter.
        p.reset();
        assert_eq!(p.queue_len(), 0);
        assert_eq!(p.dropped_events(), 0);
        assert_eq!(p.total_events(), 0);
    }

    #[test]
    fn pipeline_get_slot_mut_edits_are_visible_and_hash_checked() {
        let mut p = MetricPipeline::<16, 32>::new(0.05);
        let h = hash("mutable");
        assert!(p.get_slot_mut(h).is_none(), "no slot before any event");

        p.submit(MetricEvent::counter(h, 4.0));
        p.flush();

        {
            let slot = p.get_slot_mut(h).expect("slot exists after flush");
            assert_eq!(slot.name_hash, h);
            assert!((slot.counter - 4.0).abs() < 1e-12);
            // Mutate through the &mut: process another event and set the gauge.
            slot.process(&MetricEvent::counter(h, 6.0));
            slot.gauge = 99.0;
        }
        let slot = p.get_slot(h).expect("slot still present");
        assert!(
            (slot.counter - 10.0).abs() < 1e-12,
            "mutation via get_slot_mut persisted"
        );
        assert!((slot.gauge - 99.0).abs() < 1e-12);
        assert_eq!(slot.event_count, 2);

        // A hash that maps to the same bucket but is not the slot's hash is
        // rejected by the name_hash filter (never returns the wrong metric).
        let other = h.wrapping_add(16); // same index in a 16-slot pipeline
        assert_eq!((other as usize) % 16, (h as usize) % 16);
        assert!(p.get_slot_mut(other).is_none());
        assert!(p.get_slot(other).is_none());
    }

    #[test]
    fn pipeline_iter_slots_yields_exactly_the_active_slots() {
        let mut p = MetricPipeline::<64, 128>::new(0.05);
        assert_eq!(p.iter_slots().count(), 0);

        // Pick names whose hashes land in distinct buckets (mod 64) so each
        // one creates its own slot; the assertion below guards that choice.
        let names = ["a", "b", "c", "d", "e"];
        let mut hashes = Vec::new();
        for n in names {
            hashes.push(hash(n));
        }
        let mut buckets: Vec<usize> = hashes.iter().map(|&h| (h as usize) % 64).collect();
        buckets.sort_unstable();
        buckets.dedup();
        assert_eq!(
            buckets.len(),
            names.len(),
            "test names must not collide mod 64"
        );

        for (i, &h) in hashes.iter().enumerate() {
            p.submit(MetricEvent::counter(h, (i + 1) as f64));
        }
        // Nothing is active until flushed.
        assert_eq!(p.iter_slots().count(), 0);
        p.flush();

        let mut seen: Vec<u64> = p.iter_slots().map(|s| s.name_hash).collect();
        assert_eq!(seen.len(), names.len());
        seen.sort_unstable();
        let mut expected = hashes.clone();
        expected.sort_unstable();
        assert_eq!(seen, expected);

        // Sum of counters over the iterator equals Σ (i+1) = 15.
        let total: f64 = p.iter_slots().map(|s| s.counter).sum();
        assert!((total - 15.0).abs() < 1e-12);

        // Re-submitting an existing metric does not add a slot.
        p.submit(MetricEvent::counter(hashes[0], 1.0));
        p.flush();
        assert_eq!(p.iter_slots().count(), names.len());
    }

    #[test]
    fn metric_entry_name_str_roundtrips_and_truncates_at_64() {
        let e = MetricEntry::new("http.requests_total", MetricType::Counter);
        assert_eq!(e.name_str(), "http.requests_total");
        assert_eq!(e.name_len, "http.requests_total".len());
        assert_eq!(e.hash, hash("http.requests_total"));

        let empty = MetricEntry::new("", MetricType::Gauge);
        assert_eq!(empty.name_str(), "");
        assert_eq!(empty.name_len, 0);

        // Exactly 64 ASCII bytes fit; the 65th and beyond are dropped.
        let long = "x".repeat(70);
        let truncated = MetricEntry::new(&long, MetricType::Histogram);
        assert_eq!(truncated.name_len, 64);
        assert_eq!(truncated.name_str().len(), 64);
        assert_eq!(truncated.name_str(), &long[..64]);
        // The hash is still over the full, untruncated name.
        assert_eq!(truncated.hash, hash(&long));

        // Registry lookups hand back an entry whose name_str is the key.
        let mut reg = MetricRegistry::<4>::new();
        assert!(reg.register("db.latency", MetricType::Histogram).is_some());
        let found = reg.lookup("db.latency").expect("registered");
        assert_eq!(found.name_str(), "db.latency");
    }
}
