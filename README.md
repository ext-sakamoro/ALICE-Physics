# ALICE-Analytics

**High-Performance Telemetry & Statistical Estimation** - v0.1.0

> "Truth is not found in the raw data, but in the aggregate."

A Rust library providing probabilistic data structures for streaming analytics with mathematical error guarantees and minimal memory footprint.

## Features

| Feature | Description | Memory | Error Guarantee |
|---------|-------------|--------|-----------------|
| **HyperLogLog++** | Cardinality (unique count) estimation | O(2^P) ≈ 16KB | 1.04/√m |
| **DDSketch** | Relative-error quantile estimation | O(log N) | ±α% relative |
| **Count-Min Sketch** | Frequency estimation | O(w×d) | ε·N with prob 1-δ |
| **Local Differential Privacy** | Privacy-preserving aggregation | O(1) | ε-DP |
| **Streaming Anomaly Detection** | Real-time outlier detection | O(W) | Robust statistics |

## Design Philosophy

ALICE-Analytics is **not** a log collection system. Logs are discarded. Only **statistical sketches** are stored and transmitted, solving both privacy concerns and storage costs simultaneously.

### Core Principles

- **Zero Allocation**: All structures have fixed upper bounds at compile time
- **Mergeable**: All sketches can be merged across distributed nodes
- **no_std Compatible**: Works on embedded systems and WebAssembly
- **Mathematical Guarantees**: Error bounds are provable, not empirical

### Performance Optimizations

- **LUT-Based HyperLogLog**: Precomputed `2^{-k}` lookup table eliminates expensive `powi()` calls
- **Avalanche Hash Mixer**: MurmurHash3-style bit mixing for uniform hash distribution
- **Sorted Rolling Window**: O(N) sliding median via binary search + memmove
- **Optional SIMD**: AVX2 zero-register counting for HyperLogLog (feature flag)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ALICE-Analytics v0.1.0                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │     sketch      │  │    privacy      │  │    anomaly      │     │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤     │
│  │ HyperLogLog++   │  │ LaplaceNoise    │  │ MadDetector     │     │
│  │ DDSketch        │  │ RandomResponse  │  │ EwmaDetector    │     │
│  │ CountMinSketch  │  │ RAPPOR          │  │ ZScoreDetector  │     │
│  │ HeavyHitters    │  │ PrivacyBudget   │  │ CompositeDetect │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                      pipeline                                  │  │
│  ├───────────────────────────────────────────────────────────────┤  │
│  │  RingBuffer → MetricEvent → MetricSlot → MetricSnapshot       │  │
│  │  Lock-free SPSC queue     Pre-allocated    Exportable state   │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Cardinality Estimation (Unique Users)

```rust
use alice_analytics::prelude::*;

// HyperLogLog14: 16K registers, ~0.8% error
let mut hll = HyperLogLog14::new();

for user_id in [1u64, 2, 3, 1, 2, 4, 5, 1] {
    hll.insert(&user_id);
}

let unique_users = hll.cardinality();
println!("Unique users: ~{:.0}", unique_users); // ≈ 5
```

**Available sizes:**
- `HyperLogLog10` - 1KB, ~3.2% error
- `HyperLogLog12` - 4KB, ~1.6% error
- `HyperLogLog14` - 16KB, ~0.8% error (recommended)
- `HyperLogLog16` - 64KB, ~0.4% error

### Quantile Estimation (P99 Latency)

```rust
use alice_analytics::prelude::*;

// DDSketch256: 256 bins, 1% relative error
let mut sketch = DDSketch256::new(0.01);

for latency in [10.0, 20.0, 30.0, 100.0, 500.0] {
    sketch.insert(latency);
}

println!("P50: {:.2}ms", sketch.quantile(0.50));
println!("P99: {:.2}ms", sketch.quantile(0.99)); // ≈ 500
```

**Available sizes:**
- `DDSketch128` - 128 bins (use with α ≥ 0.05)
- `DDSketch256` - 256 bins (use with α ≥ 0.02)
- `DDSketch512` - 512 bins (use with α ≥ 0.01)
- `DDSketch1024` - 1024 bins
- `DDSketch2048` - 2048 bins (use with α = 0.01)

### Frequency Estimation (Heavy Hitters / DDoS Detection)

```rust
use alice_analytics::prelude::*;

// Width=2048, Depth=7
let mut cms = CountMinSketch2048x7::new();

for _ in 0..1000 {
    cms.insert(&"popular_item");
}
cms.insert(&"rare_item");

let freq = cms.estimate(&"popular_item"); // ≥ 1000
```

**Available sizes:**
- `CountMinSketch1024x5` - 1024×5 (40KB)
- `CountMinSketch2048x7` - 2048×7 (112KB, recommended)
- `CountMinSketch4096x5` - 4096×5 (160KB)

### Top-K Heavy Hitters

```rust
use alice_analytics::prelude::*;

let mut hh = HeavyHitters5::new(); // Track top 5

for _ in 0..1000 { hh.insert_hash(1); }
for _ in 0..500 { hh.insert_hash(2); }
for _ in 0..100 { hh.insert_hash(3); }

for entry in hh.top() {
    println!("Hash {}: {} occurrences", entry.hash, entry.count);
}
```

## Privacy-Preserving Analytics

### Local Differential Privacy with Laplace Mechanism

```rust
use alice_analytics::privacy::{LaplaceNoise, PrivacyBudget, PrivateAggregator};

// Client-side: add noise before sending (ε=0.1)
let mut noise = LaplaceNoise::new(1.0, 0.1); // sensitivity=1, ε=0.1
let true_value = 42.0;
let noisy_value = noise.privatize(true_value);

// Server-side: aggregate noisy values
let mut aggregator = PrivateAggregator::new(noise.scale());
aggregator.add(noisy_value);
// ... add more reports ...

let estimated_mean = aggregator.estimate_mean();
```

### Randomized Response for Binary Data

```rust
use alice_analytics::privacy::RandomizedResponse;

// ε = 1.0 differential privacy
let mut rr = RandomizedResponse::new(1.0);

// Client-side: randomize before sending
let true_answer = true;
let noisy_answer = rr.privatize(true_answer);

// Server-side: estimate true proportion
let true_proportion = RandomizedResponse::estimate_proportion(rr.p_true(), n, k);
```

### RAPPOR for Categorical Data

```rust
use alice_analytics::privacy::Rappor;

let mut rappor = Rappor::new(0.5, 0.75); // f=0.5, q=0.75

// Encode and privatize a category
let category_hash = 42u64;
let encoded = rappor.privatize(category_hash);
```

## Streaming Anomaly Detection

### MAD-Based Detection (Robust to Outliers)

```rust
use alice_analytics::anomaly::MadDetector;

let mut detector = MadDetector::new(3.0); // 3 MAD threshold

// Train on normal data
for v in [10.0, 11.0, 9.0, 10.5, 9.5] {
    detector.observe(v);
}

// Detect anomalies
assert!(!detector.is_anomaly(10.0)); // Normal
assert!(detector.is_anomaly(100.0)); // Anomaly!
```

### EWMA-Based Detection (Trend-Aware)

```rust
use alice_analytics::anomaly::EwmaDetector;

// α=0.1 (slow adaptation), threshold=3σ
let mut detector = EwmaDetector::new(0.1, 3.0);

for value in stream {
    if detector.is_anomaly(value) {
        println!("Anomaly: {} (expected: {:.2})", value, detector.ewma());
    }
    detector.observe(value);
}
```

### Composite Detection (Ensemble)

```rust
use alice_analytics::anomaly::CompositeDetector;

let mut detector = CompositeDetector::new();

// Train and detect
for value in data {
    detector.observe(value);
}

if detector.is_anomaly(suspicious_value) {
    let score = detector.anomaly_score(suspicious_value);
    println!("Anomaly score: {:.2}", score);
}
```

## Metric Pipeline

### High-Throughput Metric Collection

```rust
use alice_analytics::prelude::*;

// Create pipeline: 128 slots, queue size 512, α=0.05
let mut pipeline = MetricPipeline::<128, 512>::new(0.05);

// Hash metric names once
let req_hash = FnvHasher::hash_bytes(b"http.requests");
let lat_hash = FnvHasher::hash_bytes(b"http.latency");
let user_hash = FnvHasher::hash_bytes(b"unique.users");

// Submit metrics (lock-free)
pipeline.submit(MetricEvent::counter(req_hash, 1.0));
pipeline.submit(MetricEvent::histogram(lat_hash, 42.5));
pipeline.submit(MetricEvent::unique(user_hash, 12345));

// Periodic flush
pipeline.flush();

// Access aggregated data
if let Some(slot) = pipeline.get_slot(req_hash) {
    println!("Requests: {}", slot.counter);
}
```

### Distributed Aggregation

```rust
use alice_analytics::prelude::*;

// Node 1
let mut node1_hll = HyperLogLog14::new();
for i in 0..500u64 { node1_hll.insert(&i); }

// Node 2
let mut node2_hll = HyperLogLog14::new();
for i in 500..1000u64 { node2_hll.insert(&i); }

// Coordinator: merge sketches (Mergeable trait)
node1_hll.merge(&node2_hll);
let total_unique = node1_hll.cardinality(); // ≈ 1000
```

## Modules

### `sketch` - Probabilistic Data Structures

| Type | Description | Memory | Error |
|------|-------------|--------|-------|
| `HyperLogLog10/12/14/16` | Cardinality estimation | 1-64KB | 0.4-3.2% |
| `DDSketch128-2048` | Quantile estimation | 1-16KB | ±α% relative |
| `CountMinSketch*` | Frequency estimation | 40-160KB | ε·N w.p. 1-δ |
| `HeavyHitters5/10/20` | Top-K tracking | CMS + K×16 | CMS bounds |
| `FnvHasher` | Deterministic hashing + avalanche mixer | 8 bytes | - |

### `privacy` - Local Differential Privacy

| Type | Description | Guarantee |
|------|-------------|-----------|
| `LaplaceNoise` | Continuous value privacy | ε-DP |
| `RandomizedResponse` | Binary value privacy | ε-DP |
| `Rappor` | Categorical value privacy | (ε,δ)-DP |
| `PrivacyBudget` | Composition tracking | Sequential |
| `PrivateAggregator` | Noisy value aggregation | - |
| `XorShift64` | Fast deterministic PRNG | - |

### `anomaly` - Streaming Anomaly Detection

| Type | Description | Algorithm |
|------|-------------|-----------|
| `MadDetector` | Median Absolute Deviation | Sorted Rolling Window O(N) |
| `EwmaDetector` | Exponentially Weighted MA | Online O(1) |
| `ZScoreDetector` | Standard deviation based | Online O(1) |
| `CompositeDetector` | Ensemble of above | Configurable |
| `StreamingMedian` | Exact median in window | Binary search + memmove |

### `pipeline` - Metric Aggregation

| Type | Description |
|------|-------------|
| `MetricPipeline<SLOTS, QUEUE>` | Main aggregation pipeline |
| `MetricSlot` | Pre-allocated metric storage (HLL10 + DDSketch256) |
| `MetricEvent` | Metric submission event |
| `MetricSnapshot` | Point-in-time export |
| `MetricRegistry<N>` | Named metric management |
| `RingBuffer<T, N>` | Lock-free SPSC queue |

## Performance Characteristics

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| HLL insert | O(1) | O(2^P) | Single register update |
| HLL cardinality | O(m) | - | **LUT-optimized** (no powi) |
| HLL merge | O(m) | - | Register-wise max |
| DDSketch insert | O(1) | O(BINS) | ln() + array access |
| DDSketch quantile | O(BINS) | - | Linear scan |
| CMS insert | O(D) | O(W×D) | D hash computations |
| CMS estimate | O(D) | - | D lookups + min |
| MAD observe | O(N) | O(W) | **Sorted Rolling Window** |
| MAD is_anomaly | O(1) | - | Direct median access |
| StreamingMedian push | O(N) | O(W) | Binary search + memmove |
| StreamingMedian median | O(1) | - | Direct array access |

## Building

```bash
# Standard build
cargo build --release

# no_std build (for embedded/WASM)
cargo build --release --no-default-features

# Run tests
cargo test

# With SIMD optimizations (AVX2 zero counting)
cargo build --release --features simd
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `simd` | No | AVX2-accelerated HyperLogLog zero counting |

## Mathematical Background

### HyperLogLog++

Uses the harmonic mean of 2^(-register) values with bias correction:
```
E = α_m · m² / Σ 2^(-M[i])
```
Optimized with precomputed LUT: `POW2_NEG_LUT[k] = 2^{-k}` for k ∈ [0, 64].

### DDSketch

Maps values to buckets using γ = (1+α)/(1-α):
```
bucket(v) = ⌈log_γ(v)⌉ = ⌈ln(v) / ln(γ)⌉
```
Guarantees relative error ±α for any quantile.

### Count-Min Sketch

Uses d hash functions mapping to width w:
```
estimate(x) = min_{i∈[d]} table[i][h_i(x)]
```
Error bound: E[estimate(x)] ≤ count(x) + ε·N with probability ≥ 1-δ

### MAD (Median Absolute Deviation)

```
MAD = median(|X_i - median(X)|)
Anomaly if: |x - median| > k · MAD · 1.4826
```
Computed using Sorted Rolling Window for O(1) median access.

### FnvHasher with Avalanche Mixer

```
// FNV-1a core
for byte in data:
    state ^= byte
    state *= FNV_PRIME

// MurmurHash3 finalizer (avalanche)
h ^= h >> 33
h *= 0xff51afd7ed558ccd
h ^= h >> 33
h *= 0xc4ceb9fe1a85ec53
h ^= h >> 33
```

## Comparison with Traditional Approaches

| Aspect | ALICE-Analytics | Traditional Logging |
|--------|----------------|---------------------|
| Storage | O(1) per metric | O(N) raw logs |
| Privacy | Built-in LDP | Requires anonymization |
| Query Latency | O(1) | O(N) aggregation |
| Distributed | Trivial merge | Complex MapReduce |
| Accuracy | Mathematical bounds | Exact (overkill) |

## Use Cases

- **Real-time Analytics**: Dashboard metrics without storing raw events
- **Privacy-Preserving Surveys**: Collect sensitive data with plausible deniability
- **DDoS Detection**: Identify heavy hitters in network traffic
- **Latency Monitoring**: Track P99 without storing all request times
- **Unique User Counting**: Estimate DAU/MAU with 16KB memory
- **Anomaly Alerting**: Detect production issues in real-time

## License

AGPL-3.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

- HyperLogLog: Flajolet et al., "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm"
- DDSketch: Masson et al., "DDSketch: A Fast and Fully-Mergeable Quantile Sketch"
- Count-Min Sketch: Cormode & Muthukrishnan, "An Improved Data Stream Summary"
- RAPPOR: Erlingsson et al., "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response"
- MAD: Leys et al., "Detecting outliers: Do not use standard deviation around the mean"
- MurmurHash3: Austin Appleby, avalanche finalizer for hash distribution
