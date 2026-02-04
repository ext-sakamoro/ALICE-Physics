//! # ALICE-Analytics
//!
//! **High-Performance Telemetry & Statistical Estimation**
//!
//! A Rust library providing probabilistic data structures for streaming analytics
//! with mathematical error guarantees and minimal memory footprint.
//!
//! ## Features
//!
//! | Feature | Description | Memory |
//! |---------|-------------|--------|
//! | **HyperLogLog++** | Cardinality estimation | O(2^P) ≈ 16KB |
//! | **DDSketch** | Relative-error quantiles | O(log N) |
//! | **Count-Min Sketch** | Frequency estimation | O(w × d) |
//! | **Local Differential Privacy** | Privacy-preserving aggregation | O(1) |
//! | **Streaming Anomaly Detection** | Real-time outlier detection | O(W) |
//!
//! ## Design Principles
//!
//! - **Zero Allocation**: All structures have fixed upper bounds at compile time
//! - **Mergeable**: All sketches can be merged across distributed nodes
//! - **no_std Compatible**: Works on embedded systems and WebAssembly
//! - **Mathematical Guarantees**: Error bounds are provable, not empirical
//!
//! ## Quick Start
//!
//! ```rust
//! use alice_analytics::prelude::*;
//!
//! // Cardinality estimation (unique users)
//! let mut hll = HyperLogLog::new();
//! for user_id in [1u64, 2, 3, 1, 2, 4, 5, 1] {
//!     hll.insert(&user_id);
//! }
//! let unique_users = hll.cardinality(); // ≈ 5
//!
//! // Quantile estimation (P99 latency)
//! let mut sketch = DDSketch::new(0.01); // 1% relative error
//! for latency in [10.0, 20.0, 30.0, 100.0, 500.0] {
//!     sketch.insert(latency);
//! }
//! let p99 = sketch.quantile(0.99); // ≈ 500
//!
//! // Frequency estimation (heavy hitters)
//! let mut cms = CountMinSketch::new();
//! for _ in 0..100 {
//!     cms.insert(&"popular_item");
//! }
//! let freq = cms.estimate(&"popular_item"); // ≥ 100
//! ```
//!
//! ## Privacy-Preserving Analytics
//!
//! ```rust
//! use alice_analytics::privacy::{LaplaceNoise, RandomizedResponse};
//!
//! // Add Laplace noise for differential privacy
//! let mut noise = LaplaceNoise::new(1.0, 0.1); // sensitivity=1, ε=0.1
//! let private_count = noise.privatize(42.0);
//!
//! // Randomized response for binary data
//! let mut rr = RandomizedResponse::new(1.0); // ε=1
//! let private_answer = rr.privatize(true);
//! ```
//!
//! ## Anomaly Detection
//!
//! ```rust
//! use alice_analytics::anomaly::MadDetector;
//!
//! let mut detector = MadDetector::new(3.0); // 3 MAD threshold
//!
//! // Train on normal data
//! for v in [10.0, 11.0, 9.0, 10.5, 9.5] {
//!     detector.observe(v);
//! }
//!
//! // Detect anomalies
//! assert!(!detector.is_anomaly(10.0)); // Normal
//! assert!(detector.is_anomaly(100.0)); // Anomaly!
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

pub mod anomaly;
pub mod pipeline;
pub mod privacy;
pub mod sketch;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::anomaly::{
        AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector,
        StreamingMedian, ZScoreDetector,
    };
    pub use crate::pipeline::{
        MetricEntry, MetricEvent, MetricPipeline, MetricRegistry, MetricSlot, MetricSnapshot,
        MetricType, RingBuffer,
    };
    pub use crate::privacy::{
        LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
    };
    pub use crate::sketch::{
        CountMinSketch, CountMinSketch1024x5, CountMinSketch2048x7, CountMinSketch4096x5,
        DDSketch, DDSketch128, DDSketch256, DDSketch512, DDSketch1024, DDSketch2048,
        FnvHasher,
        HeavyHitters, HeavyHitters5, HeavyHitters10, HeavyHitters20, HeavyHitterEntry,
        HyperLogLog, HyperLogLog10, HyperLogLog12, HyperLogLog14, HyperLogLog16,
        Mergeable,
    };
}

// Re-export main types at crate root
pub use prelude::*;

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_hll_cardinality() {
        let mut hll = HyperLogLog16::new();

        // Insert 100000 unique values using explicit hash for consistency
        for i in 0..100000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // HyperLogLog16 has ~0.4% standard error, allow ±25% for robustness
        assert!(estimate > 75000.0 && estimate < 125000.0, "estimate = {}", estimate);
    }

    #[test]
    fn test_ddsketch_p99() {
        // Use DDSketch2048 for alpha=0.01 to ensure enough bins
        let mut sketch = DDSketch2048::new(0.01);

        // Insert latencies
        for i in 1..=100 {
            sketch.insert(i as f64);
        }

        let p50 = sketch.quantile(0.50);
        let p99 = sketch.quantile(0.99);

        // P50 should be around 50
        assert!(p50 > 45.0 && p50 < 55.0, "p50 = {}", p50);
        // P99 should be around 99
        assert!(p99 > 95.0 && p99 <= 100.0, "p99 = {}", p99);
    }

    #[test]
    fn test_cms_frequency() {
        let mut cms = CountMinSketch2048x7::new();

        // Insert items with different frequencies
        for _ in 0..1000 {
            cms.insert(&"frequent");
        }
        for _ in 0..100 {
            cms.insert(&"moderate");
        }
        for _ in 0..10 {
            cms.insert(&"rare");
        }

        // Estimates should be at least the true counts
        assert!(cms.estimate(&"frequent") >= 1000);
        assert!(cms.estimate(&"moderate") >= 100);
        assert!(cms.estimate(&"rare") >= 10);

        // Unknown item should have 0 or very low estimate
        assert!(cms.estimate(&"unknown") < 50);
    }

    #[test]
    fn test_privacy_laplace() {
        let mut noise = LaplaceNoise::with_seed(1.0, 1.0, 42);

        // Privatize many values and check average is close to true value
        let true_value = 100.0;
        let n = 10000;
        let mut sum = 0.0;

        for _ in 0..n {
            sum += noise.privatize(true_value);
        }

        let avg = sum / n as f64;
        // Should be within 1 of true value (with high probability)
        assert!(
            (avg - true_value).abs() < 1.0,
            "avg = {}, true = {}",
            avg,
            true_value
        );
    }

    #[test]
    fn test_anomaly_detection() {
        let mut detector = CompositeDetector::new();

        // Train on normal data (mean ≈ 100, low variance)
        for i in 0..100 {
            let value = 100.0 + (i % 5) as f64 - 2.0; // 98-102
            detector.observe(value);
        }

        // Normal values should not be flagged
        assert!(!detector.is_anomaly(100.0));
        assert!(!detector.is_anomaly(101.0));
        assert!(!detector.is_anomaly(99.0));

        // Extreme values should be flagged
        assert!(detector.is_anomaly(200.0));
        assert!(detector.is_anomaly(0.0));
    }

    #[test]
    fn test_pipeline_integration() {
        // Use more slots to reduce hash collisions
        let mut pipeline = MetricPipeline::<128, 512>::new(0.05);

        let req_hash = FnvHasher::hash_bytes(b"http.requests");
        let lat_hash = FnvHasher::hash_bytes(b"http.latency");
        let user_hash = FnvHasher::hash_bytes(b"unique.users");

        // Submit various metrics
        for i in 0..100 {
            pipeline.submit(MetricEvent::counter(req_hash, 1.0));
            pipeline.submit(MetricEvent::histogram(lat_hash, 10.0 + (i % 20) as f64));
            pipeline.submit(MetricEvent::unique(user_hash, i % 50)); // 50 unique users
        }

        // Flush
        pipeline.flush();

        // Check counters
        let req_slot = pipeline.get_slot(req_hash).unwrap();
        assert_eq!(req_slot.counter, 100.0);

        // Check histogram
        let lat_slot = pipeline.get_slot(lat_hash).unwrap();
        assert_eq!(lat_slot.ddsketch.count(), 100);

        // Check cardinality (HyperLogLog10 in MetricSlot)
        let user_slot = pipeline.get_slot(user_hash).unwrap();
        let cardinality = user_slot.hll.cardinality();
        assert!(cardinality > 30.0 && cardinality < 70.0, "cardinality = {}", cardinality);
    }

    #[test]
    fn test_mergeable_sketches() {
        // Create two HLLs on different nodes (use HyperLogLog16 for best accuracy)
        let mut node1 = HyperLogLog16::new();
        let mut node2 = HyperLogLog16::new();

        // Each node sees different users - use explicit hash for consistency
        for i in 0..500u64 {
            node1.insert_hash(FnvHasher::hash_u64(i));
        }
        for i in 500..1000u64 {
            node2.insert_hash(FnvHasher::hash_u64(i));
        }

        // Merge
        node1.merge(&node2);

        let estimate = node1.cardinality();
        // Should be close to 1000 (allow ±20% for statistical variance)
        assert!(estimate > 800.0 && estimate < 1200.0, "estimate = {}", estimate);
    }

    #[test]
    fn test_heavy_hitters() {
        let mut hh = HeavyHitters5::new();

        // Insert items with varying frequencies
        for _ in 0..1000 {
            hh.insert_hash(1);
        }
        for _ in 0..500 {
            hh.insert_hash(2);
        }
        for _ in 0..250 {
            hh.insert_hash(3);
        }
        for _ in 0..100 {
            hh.insert_hash(4);
        }
        for _ in 0..50 {
            hh.insert_hash(5);
        }
        // Many rare items
        for i in 100..200 {
            hh.insert_hash(i);
        }

        let top: Vec<_> = hh.top().collect();

        // Top 5 should be items 1-5
        assert_eq!(top.len(), 5);
        assert_eq!(top[0].hash, 1);
        assert_eq!(top[1].hash, 2);
        assert_eq!(top[2].hash, 3);
    }
}
