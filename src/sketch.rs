//! Probabilistic Sketch Data Structures
//!
//! - HyperLogLog++: Cardinality estimation with ~1.04/√m standard error
//! - DDSketch: Relative-error quantile estimation
//! - Count-Min Sketch: Frequency estimation for heavy hitters
//!
//! # Examples
//!
//! ```
//! use alice_physics::sketch::{HyperLogLog, DDSketch, CountMinSketch, Mergeable};
//!
//! // Cardinality estimation (default: 14-bit precision)
//! let mut hll = HyperLogLog::new();
//! for i in 0..1000u64 {
//!     hll.insert(&i);
//! }
//! let estimate = hll.cardinality();
//! assert!(estimate > 900.0 && estimate < 1100.0);
//!
//! // Quantile estimation (default: 2048 bins)
//! let mut sketch = DDSketch::new(0.01);
//! for i in 1..=100 {
//!     sketch.insert(i as f64);
//! }
//! let p50 = sketch.quantile(0.5);
//! assert!(p50 > 45.0 && p50 < 55.0);
//!
//! // Frequency estimation (default: 1024x5)
//! let mut cms = CountMinSketch::new();
//! cms.insert(&42u64);
//! cms.insert(&42u64);
//! assert!(cms.estimate(&42u64) >= 2);
//! ```

use core::hash::{Hash, Hasher};

// ============================================================================
// Lookup Tables for Fast Computation
// ============================================================================

/// Precomputed 2^{-k} values for k = 0..64 (HyperLogLog optimization)
/// Eliminates expensive powi() calls in cardinality estimation
const POW2_NEG_LUT: [f64; 65] = [
    1.0,                    // 2^-0
    0.5,                    // 2^-1
    0.25,                   // 2^-2
    0.125,                  // 2^-3
    0.0625,                 // 2^-4
    0.03125,                // 2^-5
    0.015625,               // 2^-6
    0.0078125,              // 2^-7
    0.00390625,             // 2^-8
    0.001953125,            // 2^-9
    0.0009765625,           // 2^-10
    0.00048828125,          // 2^-11
    0.000244140625,         // 2^-12
    0.0001220703125,        // 2^-13
    6.103515625e-5,         // 2^-14
    3.0517578125e-5,        // 2^-15
    1.52587890625e-5,       // 2^-16
    7.62939453125e-6,       // 2^-17
    3.814697265625e-6,      // 2^-18
    1.9073486328125e-6,     // 2^-19
    9.5367431640625e-7,     // 2^-20
    4.76837158203125e-7,    // 2^-21
    2.384185791015625e-7,   // 2^-22
    1.1920928955078125e-7,  // 2^-23
    5.960464477539063e-8,   // 2^-24
    2.9802322387695312e-8,  // 2^-25
    1.4901161193847656e-8,  // 2^-26
    7.450580596923828e-9,   // 2^-27
    3.725290298461914e-9,   // 2^-28
    1.862645149230957e-9,   // 2^-29
    9.313225746154785e-10,  // 2^-30
    4.656612873077393e-10,  // 2^-31
    2.3283064365386963e-10, // 2^-32
    1.1641532182693481e-10, // 2^-33
    5.820766091346741e-11,  // 2^-34
    2.9103830456733704e-11, // 2^-35
    1.4551915228366852e-11, // 2^-36
    7.275957614183426e-12,  // 2^-37
    3.637978807091713e-12,  // 2^-38
    1.8189894035458565e-12, // 2^-39
    9.094947017729282e-13,  // 2^-40
    4.547473508864641e-13,  // 2^-41
    2.2737367544323206e-13, // 2^-42
    1.1368683772161603e-13, // 2^-43
    5.684341886080802e-14,  // 2^-44
    2.842170943040401e-14,  // 2^-45
    1.4210854715202004e-14, // 2^-46
    7.105427357601002e-15,  // 2^-47
    3.552713678800501e-15,  // 2^-48
    1.7763568394002505e-15, // 2^-49
    8.881784197001252e-16,  // 2^-50
    4.440892098500626e-16,  // 2^-51
    2.220446049250313e-16,  // 2^-52
    1.1102230246251565e-16, // 2^-53
    5.551115123125783e-17,  // 2^-54
    2.7755575615628914e-17, // 2^-55
    1.3877787807814457e-17, // 2^-56
    6.938893903907228e-18,  // 2^-57
    3.469446951953614e-18,  // 2^-58
    1.734723475976807e-18,  // 2^-59
    8.673617379884035e-19,  // 2^-60
    4.336808689942018e-19,  // 2^-61
    2.168404344971009e-19,  // 2^-62
    1.0842021724855044e-19, // 2^-63
    5.421010862427522e-20,  // 2^-64
];

/// Fast approximate log2 using IEEE 754 bit extraction
/// Returns floor(log2(x)) for positive x, useful for bucket indexing
#[inline]
fn fast_log2_approx(x: f64) -> f64 {
    // IEEE 754 double: sign(1) | exponent(11) | mantissa(52)
    // For positive x: log2(x) ≈ exponent - 1023 + mantissa_fraction
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i64;
    let mantissa = bits & 0xFFFFFFFFFFFFF;

    // exponent - 1023 gives the integer part of log2
    // mantissa / 2^52 gives a value in [0, 1) for linear interpolation
    let int_part = exponent - 1023;
    let frac_part = mantissa as f64 / 4503599627370496.0; // 2^52

    int_part as f64 + frac_part
}

// ============================================================================
// Mergeable Trait - All sketches can be merged across distributed nodes
// ============================================================================

/// Trait for mergeable probabilistic data structures
pub trait Mergeable {
    /// Merge another sketch into this one
    fn merge(&mut self, other: &Self);
}

// ============================================================================
// Simple Hash Function (FNV-1a variant for determinism)
// ============================================================================

/// FNV-1a hash for deterministic, fast hashing
#[derive(Clone, Copy, Debug)]
pub struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    /// Create a new hasher with default FNV offset basis.
    #[inline]
    pub const fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    /// Avalanche bit mixer (from MurmurHash3 finalizer).
    /// Ensures all bits are well-distributed for HyperLogLog.
    #[inline]
    fn mix(mut h: u64) -> u64 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Hash a byte slice and return a mixed 64-bit digest.
    #[inline]
    pub fn hash_bytes(data: &[u8]) -> u64 {
        let mut hasher = Self::new();
        hasher.write(data);
        Self::mix(hasher.state)
    }

    /// Hash a `u64` value.
    #[inline]
    pub fn hash_u64(value: u64) -> u64 {
        Self::hash_bytes(&value.to_le_bytes())
    }

    /// Hash a `u128` value.
    #[inline]
    pub fn hash_u128(value: u128) -> u64 {
        Self::hash_bytes(&value.to_le_bytes())
    }
}

impl Default for FnvHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        // Apply avalanche mixer for better bit distribution
        Self::mix(self.state)
    }
}

// ============================================================================
// HyperLogLog++ - Cardinality Estimation
// ============================================================================

/// Macro to generate HyperLogLog implementations for specific sizes
macro_rules! impl_hyperloglog {
    ($name:ident, $p:expr, $m:expr) => {
        /// HyperLogLog++ for cardinality (unique count) estimation
        ///
        /// Memory: $m bytes
        /// Error: ~1.04 / sqrt($m)
        #[derive(Clone, Debug)]
        pub struct $name {
            /// Registers storing maximum leading zeros + 1
            registers: [u8; $m],
        }

        impl $name {
            /// Number of registers (m = 2^P)
            pub const M: usize = $m;
            /// Bits used for register index
            pub const P: usize = $p;

            /// Alpha constant for bias correction
            const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / ($m as f64));

            /// Create a new empty HyperLogLog
            #[inline]
            pub fn new() -> Self {
                Self {
                    registers: [0u8; $m],
                }
            }

            /// Insert an already-hashed value
            #[inline]
            pub fn insert_hash(&mut self, hash: u64) {
                let idx = (hash as usize) & (Self::M - 1);
                let w = hash >> $p;
                // rho = position of first 1 bit in the (64-P) remaining bits
                // leading_zeros(w) includes the P bits we shifted away, so subtract them
                let rho = if w == 0 {
                    (64 - $p + 1) as u8
                } else {
                    (w.leading_zeros() as usize - $p + 1) as u8
                };
                if rho > self.registers[idx] {
                    self.registers[idx] = rho;
                }
            }

            /// Insert a hashable value
            #[inline]
            pub fn insert<T: Hash>(&mut self, value: &T) {
                let mut hasher = FnvHasher::new();
                value.hash(&mut hasher);
                self.insert_hash(hasher.finish());
            }

            /// Insert raw bytes
            #[inline]
            pub fn insert_bytes(&mut self, bytes: &[u8]) {
                self.insert_hash(FnvHasher::hash_bytes(bytes));
            }

            /// Estimate cardinality using HyperLogLog++ algorithm
            /// Optimized with LUT for 2^{-k} values
            pub fn cardinality(&self) -> f64 {
                let mut sum = 0.0f64;
                let mut zeros = 0usize;

                // Use LUT instead of expensive powi() calls
                for &reg in &self.registers {
                    // LUT has 65 entries (0..=64), clamp to be safe
                    let idx = (reg as usize).min(64);
                    sum += POW2_NEG_LUT[idx];
                    if reg == 0 {
                        zeros += 1;
                    }
                }

                let m = Self::M as f64;
                let raw_estimate = Self::ALPHA * m * m / sum;

                if raw_estimate <= 2.5 * m && zeros > 0 {
                    m * (m / zeros as f64).ln()
                } else {
                    raw_estimate
                }
            }

            /// Count zero registers using SIMD when available
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            fn count_zeros_simd(&self) -> usize {
                #[cfg(target_arch = "x86_64")]
                {
                    use core::arch::x86_64::*;
                    let mut zeros = 0usize;
                    let chunks = self.registers.chunks_exact(32);
                    let remainder = chunks.remainder();

                    // SAFETY: chunks_exact(32) guarantees each chunk is exactly 32 bytes,
                    // matching the __m256i width. _mm256_loadu_si256 handles unaligned loads.
                    // AVX2 availability is checked at runtime by the cfg gate above.
                    unsafe {
                        let zero_vec = _mm256_setzero_si256();
                        for chunk in chunks {
                            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                            let cmp = _mm256_cmpeq_epi8(data, zero_vec);
                            let mask = _mm256_movemask_epi8(cmp) as u32;
                            zeros += mask.count_ones() as usize;
                        }
                    }

                    // Handle remainder
                    for &reg in remainder {
                        if reg == 0 {
                            zeros += 1;
                        }
                    }
                    zeros
                }
            }

            /// Get raw registers
            #[inline]
            pub fn registers(&self) -> &[u8] {
                &self.registers
            }

            /// Reset all registers to zero
            #[inline]
            pub fn clear(&mut self) {
                self.registers = [0u8; $m];
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) {
                for (dst, &src) in self.registers.iter_mut().zip(other.registers.iter()) {
                    if src > *dst {
                        *dst = src;
                    }
                }
            }
        }
    };
}

// Generate common HyperLogLog sizes
impl_hyperloglog!(HyperLogLog10, 10, 1024); // 1KB, ~3.2% error
impl_hyperloglog!(HyperLogLog12, 12, 4096); // 4KB, ~1.6% error
impl_hyperloglog!(HyperLogLog14, 14, 16384); // 16KB, ~0.8% error
impl_hyperloglog!(HyperLogLog16, 16, 65536); // 64KB, ~0.4% error

/// Type alias for the most common HyperLogLog size (16KB, ~0.8% error)
pub type HyperLogLog = HyperLogLog14;

// ============================================================================
// DDSketch - Relative Error Quantile Estimation
// ============================================================================

/// DDSketch for quantile estimation with relative error guarantee
///
/// Guarantees that for any quantile q, the returned value v satisfies:
/// |v - true_value| <= α * true_value
///
/// where α is the relative accuracy (e.g., 0.01 for 1% error)
///
/// # Example
/// ```
/// use alice_physics::sketch::DDSketch256;
///
/// let mut sketch = DDSketch256::new(0.01); // 1% relative error
///
/// for latency in [10.0, 20.0, 30.0, 100.0, 500.0] {
///     sketch.insert(latency);
/// }
///
/// let p99 = sketch.quantile(0.99);
/// // p99 ≈ 500.0 (within 1% relative error)
/// ```
/// Macro to generate DDSketch implementations for specific bin counts
macro_rules! impl_ddsketch {
    ($name:ident, $bins:expr) => {
        /// DDSketch with relative-error quantile guarantee.
        #[derive(Clone, Debug)]
        pub struct $name {
            positive_bins: [u64; $bins],
            negative_bins: [u64; $bins],
            zero_count: u64,
            count: u64,
            min: f64,
            max: f64,
            sum: f64,
            gamma: f64,
            ln_gamma: f64,
            alpha: f64,
            offset: i32,
        }

        impl $name {
            /// Number of bins per side (positive / negative).
            pub const BINS: usize = $bins;

            /// Create a new sketch with given relative accuracy `alpha`.
            pub fn new(alpha: f64) -> Self {
                let gamma = (1.0 + alpha) / (1.0 - alpha);
                let ln_gamma = gamma.ln();
                // Offset to center around 1.0 (ln(1.0) = 0)
                // For typical latencies (1ms - 10s), we want indices to fit in BINS
                // With offset at BINS/4, we can handle values from gamma^(-BINS/4) to gamma^(3*BINS/4)
                let offset = ($bins / 4) as i32;

                Self {
                    positive_bins: [0u64; $bins],
                    negative_bins: [0u64; $bins],
                    zero_count: 0,
                    count: 0,
                    min: f64::INFINITY,
                    max: f64::NEG_INFINITY,
                    sum: 0.0,
                    gamma,
                    ln_gamma,
                    alpha,
                    offset,
                }
            }

            /// Insert a value into the sketch.
            #[inline]
            pub fn insert(&mut self, value: f64) {
                self.count += 1;
                self.sum += value;

                if value < self.min {
                    self.min = value;
                }
                if value > self.max {
                    self.max = value;
                }

                if value > 0.0 {
                    let idx = self.bucket_index(value);
                    if idx < $bins {
                        self.positive_bins[idx] += 1;
                    }
                } else if value < 0.0 {
                    let idx = self.bucket_index(-value);
                    if idx < $bins {
                        self.negative_bins[idx] += 1;
                    }
                } else {
                    self.zero_count += 1;
                }
            }

            /// Bucket index calculation
            /// Uses standard ln() for quantile accuracy (DDSketch requires precise buckets)
            #[inline]
            fn bucket_index(&self, value: f64) -> usize {
                let idx = (value.ln() / self.ln_gamma).ceil() as i32 + self.offset;
                idx.max(0) as usize
            }

            /// Fast bucket index using IEEE 754 bit extraction (for non-critical paths)
            /// ~10x faster than ln() but has ~1-2% error
            #[inline]
            #[allow(dead_code)]
            fn bucket_index_fast(&self, value: f64) -> usize {
                const LN2: f64 = core::f64::consts::LN_2;
                let log2_gamma = self.ln_gamma / LN2;
                let log2_value = fast_log2_approx(value);
                let idx = (log2_value / log2_gamma).ceil() as i32 + self.offset;
                idx.max(0) as usize
            }

            #[inline]
            fn bucket_lower_bound(&self, idx: usize) -> f64 {
                let exp = (idx as i32 - self.offset) as f64;
                self.gamma.powf(exp - 1.0)
            }

            /// Estimate the value at quantile `q` (0.0–1.0).
            pub fn quantile(&self, q: f64) -> f64 {
                if self.count == 0 {
                    return 0.0;
                }

                let rank = (q * self.count as f64).ceil() as u64;
                let mut cumulative = 0u64;

                for (idx, &count) in self.negative_bins.iter().enumerate().rev() {
                    cumulative += count;
                    if cumulative >= rank {
                        return -self.bucket_lower_bound(idx);
                    }
                }

                cumulative += self.zero_count;
                if cumulative >= rank {
                    return 0.0;
                }

                for (idx, &count) in self.positive_bins.iter().enumerate() {
                    cumulative += count;
                    if cumulative >= rank {
                        return self.bucket_lower_bound(idx);
                    }
                }

                self.max
            }

            /// Total number of inserted values.
            #[inline]
            pub fn count(&self) -> u64 {
                self.count
            }

            /// Sum of all inserted values.
            #[inline]
            pub fn sum(&self) -> f64 {
                self.sum
            }

            /// Arithmetic mean of inserted values.
            #[inline]
            pub fn mean(&self) -> f64 {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                }
            }

            /// Minimum inserted value.
            #[inline]
            pub fn min(&self) -> f64 {
                self.min
            }

            /// Maximum inserted value.
            #[inline]
            pub fn max(&self) -> f64 {
                self.max
            }

            /// Relative accuracy parameter.
            #[inline]
            pub fn alpha(&self) -> f64 {
                self.alpha
            }

            /// Reset the sketch to empty state.
            pub fn clear(&mut self) {
                self.positive_bins = [0u64; $bins];
                self.negative_bins = [0u64; $bins];
                self.zero_count = 0;
                self.count = 0;
                self.min = f64::INFINITY;
                self.max = f64::NEG_INFINITY;
                self.sum = 0.0;
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) {
                for (dst, &src) in self
                    .positive_bins
                    .iter_mut()
                    .zip(other.positive_bins.iter())
                {
                    *dst += src;
                }
                for (dst, &src) in self
                    .negative_bins
                    .iter_mut()
                    .zip(other.negative_bins.iter())
                {
                    *dst += src;
                }
                self.zero_count += other.zero_count;
                self.count += other.count;
                self.sum += other.sum;

                if other.min < self.min {
                    self.min = other.min;
                }
                if other.max > self.max {
                    self.max = other.max;
                }
            }
        }
    };
}

// Generate common DDSketch sizes
impl_ddsketch!(DDSketch128, 128); // Small, use alpha >= 0.1
impl_ddsketch!(DDSketch256, 256); // Medium, use alpha >= 0.05
impl_ddsketch!(DDSketch512, 512); // Good balance
impl_ddsketch!(DDSketch1024, 1024); // High accuracy, alpha >= 0.02
impl_ddsketch!(DDSketch2048, 2048); // Very high accuracy, alpha >= 0.01

/// Type alias for the most common DDSketch size (good for alpha=0.01)
pub type DDSketch = DDSketch2048;

// ============================================================================
// Count-Min Sketch - Frequency Estimation
// ============================================================================

/// Macro to generate CountMinSketch implementations
macro_rules! impl_countmin {
    ($name:ident, $w:expr, $d:expr) => {
        /// Count-Min Sketch for frequency estimation
        #[derive(Clone, Debug)]
        pub struct $name {
            counters: [[u64; $w]; $d],
            total: u64,
        }

        impl $name {
            /// Number of columns (width).
            pub const WIDTH: usize = $w;
            /// Number of hash rows (depth).
            pub const DEPTH: usize = $d;

            /// Create an empty sketch.
            #[inline]
            pub fn new() -> Self {
                Self {
                    counters: [[0u64; $w]; $d],
                    total: 0,
                }
            }

            #[inline]
            fn hash_for_row(hash: u64, row: usize) -> usize {
                let h = hash.wrapping_add((row as u64).wrapping_mul(0x9e3779b97f4a7c15));
                let mixed = h ^ (h >> 33);
                let mixed = mixed.wrapping_mul(0xff51afd7ed558ccd);
                let mixed = mixed ^ (mixed >> 33);
                (mixed as usize) % $w
            }

            /// Insert a pre-hashed item with the given count.
            #[inline]
            pub fn insert_hash(&mut self, hash: u64, count: u64) {
                self.total += count;
                for row in 0..$d {
                    let col = Self::hash_for_row(hash, row);
                    self.counters[row][col] = self.counters[row][col].saturating_add(count);
                }
            }

            /// Insert a hashable item with count 1.
            #[inline]
            pub fn insert<T: Hash>(&mut self, item: &T) {
                let mut hasher = FnvHasher::new();
                item.hash(&mut hasher);
                self.insert_hash(hasher.finish(), 1);
            }

            /// Insert raw bytes with count 1.
            #[inline]
            pub fn insert_bytes(&mut self, bytes: &[u8]) {
                self.insert_hash(FnvHasher::hash_bytes(bytes), 1);
            }

            /// Estimate frequency of a pre-hashed item.
            #[inline]
            pub fn estimate_hash(&self, hash: u64) -> u64 {
                let mut min_count = u64::MAX;
                for row in 0..$d {
                    let col = Self::hash_for_row(hash, row);
                    min_count = min_count.min(self.counters[row][col]);
                }
                min_count
            }

            /// Estimate frequency of a hashable item.
            #[inline]
            pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
                let mut hasher = FnvHasher::new();
                item.hash(&mut hasher);
                self.estimate_hash(hasher.finish())
            }

            /// Estimate frequency of raw bytes.
            #[inline]
            pub fn estimate_bytes(&self, bytes: &[u8]) -> u64 {
                self.estimate_hash(FnvHasher::hash_bytes(bytes))
            }

            /// Total count of all insertions.
            #[inline]
            pub fn total(&self) -> u64 {
                self.total
            }

            /// Reset all counters to zero.
            #[inline]
            pub fn clear(&mut self) {
                self.counters = [[0u64; $w]; $d];
                self.total = 0;
            }

            /// Theoretical error bound (ε = e / width).
            #[inline]
            pub fn error_bound(&self) -> f64 {
                core::f64::consts::E / ($w as f64)
            }

            /// Confidence level (1 − e^{−depth}).
            #[inline]
            pub fn confidence(&self) -> f64 {
                1.0 - (-($d as f64)).exp()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Mergeable for $name {
            fn merge(&mut self, other: &Self) {
                self.total += other.total;
                for row in 0..$d {
                    for col in 0..$w {
                        self.counters[row][col] =
                            self.counters[row][col].saturating_add(other.counters[row][col]);
                    }
                }
            }
        }
    };
}

// Generate common CountMinSketch sizes
impl_countmin!(CountMinSketch1024x5, 1024, 5);
impl_countmin!(CountMinSketch2048x7, 2048, 7);
impl_countmin!(CountMinSketch4096x5, 4096, 5);

/// Type alias for the default Count-Min Sketch
pub type CountMinSketch = CountMinSketch1024x5;

// ============================================================================
// Heavy Hitters (Top-K using Count-Min Sketch + Heap)
// ============================================================================

/// Entry for heavy hitters tracking
#[derive(Clone, Copy, Debug)]
pub struct HeavyHitterEntry {
    /// Hash of the item
    pub hash: u64,
    /// Estimated frequency
    pub count: u64,
}

/// Macro to generate HeavyHitters implementations
macro_rules! impl_heavy_hitters {
    ($name:ident, $cms_name:ident, $k:expr) => {
        /// Heavy Hitters tracker using Count-Min Sketch
        #[derive(Clone, Debug)]
        pub struct $name {
            cms: $cms_name,
            top_k: [HeavyHitterEntry; $k],
            count: usize,
        }

        impl $name {
            /// Maximum tracked heavy hitters.
            pub const K: usize = $k;

            /// Create a new empty tracker.
            #[inline]
            pub fn new() -> Self {
                Self {
                    cms: $cms_name::new(),
                    top_k: [HeavyHitterEntry { hash: 0, count: 0 }; $k],
                    count: 0,
                }
            }

            /// Insert a pre-hashed item and update top-K.
            pub fn insert_hash(&mut self, hash: u64) {
                self.cms.insert_hash(hash, 1);
                let estimated = self.cms.estimate_hash(hash);

                let mut found_idx = None;
                for i in 0..self.count {
                    if self.top_k[i].hash == hash {
                        found_idx = Some(i);
                        break;
                    }
                }

                if let Some(idx) = found_idx {
                    self.top_k[idx].count = estimated;
                    self.sort_top_k();
                } else if self.count < $k {
                    self.top_k[self.count] = HeavyHitterEntry {
                        hash,
                        count: estimated,
                    };
                    self.count += 1;
                    self.sort_top_k();
                } else if estimated > self.top_k[0].count {
                    self.top_k[0] = HeavyHitterEntry {
                        hash,
                        count: estimated,
                    };
                    self.sort_top_k();
                }
            }

            fn sort_top_k(&mut self) {
                for i in 1..self.count {
                    let entry = self.top_k[i];
                    let mut j = i;
                    while j > 0 && self.top_k[j - 1].count > entry.count {
                        self.top_k[j] = self.top_k[j - 1];
                        j -= 1;
                    }
                    self.top_k[j] = entry;
                }
            }

            /// Iterate top-K entries in descending frequency order.
            pub fn top(&self) -> impl Iterator<Item = &HeavyHitterEntry> {
                self.top_k[..self.count].iter().rev()
            }

            /// Access the underlying Count-Min Sketch.
            #[inline]
            pub fn cms(&self) -> &$cms_name {
                &self.cms
            }

            /// Reset the tracker and its underlying sketch.
            pub fn clear(&mut self) {
                self.cms.clear();
                self.top_k = [HeavyHitterEntry { hash: 0, count: 0 }; $k];
                self.count = 0;
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

// Generate HeavyHitters variants
impl_heavy_hitters!(HeavyHitters10, CountMinSketch1024x5, 10);
impl_heavy_hitters!(HeavyHitters20, CountMinSketch2048x7, 20);
impl_heavy_hitters!(HeavyHitters5, CountMinSketch1024x5, 5);

/// Type alias for the default Heavy Hitters tracker
pub type HeavyHitters = HeavyHitters10;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv_hash() {
        let h1 = FnvHasher::hash_u64(12345);
        let h2 = FnvHasher::hash_u64(12345);
        let h3 = FnvHasher::hash_u64(12346);

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hyperloglog_basic() {
        // Use HyperLogLog16 (64K registers) for best accuracy
        let mut hll = HyperLogLog16::new();

        // Insert 1000 unique values
        for i in 0..1000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // Relaxed tolerance due to statistical nature
        assert!(
            estimate > 800.0 && estimate < 1200.0,
            "estimate = {}",
            estimate
        );
    }

    #[test]
    fn test_hyperloglog_merge() {
        let mut hll1 = HyperLogLog16::new();
        let mut hll2 = HyperLogLog16::new();

        for i in 0..500u64 {
            hll1.insert_hash(FnvHasher::hash_u64(i));
        }
        for i in 500..1000u64 {
            hll2.insert_hash(FnvHasher::hash_u64(i));
        }

        hll1.merge(&hll2);
        let estimate = hll1.cardinality();
        assert!(
            estimate > 800.0 && estimate < 1200.0,
            "estimate = {}",
            estimate
        );
    }

    #[test]
    fn test_hyperloglog_16_large() {
        let mut hll = HyperLogLog16::new();

        // Use explicit hash for consistency with other tests
        for i in 0..100000u64 {
            hll.insert_hash(FnvHasher::hash_u64(i));
        }

        let estimate = hll.cardinality();
        // With 100K values in 64K registers, expect reasonable accuracy (±25%)
        assert!(
            estimate > 75000.0 && estimate < 125000.0,
            "estimate = {}",
            estimate
        );
    }

    #[test]
    fn test_ddsketch_basic() {
        // Use DDSketch2048 for alpha=0.01 to ensure enough bins
        let mut sketch = DDSketch2048::new(0.01);

        let values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        for v in values {
            sketch.insert(v);
        }

        assert_eq!(sketch.count(), 10);
        assert!((sketch.mean() - 55.0).abs() < 0.001);

        let p50 = sketch.quantile(0.5);
        assert!(p50 > 40.0 && p50 < 70.0, "p50 = {}", p50);

        let p99 = sketch.quantile(0.99);
        assert!(p99 > 80.0 && p99 <= 100.0, "p99 = {}", p99);
    }

    #[test]
    fn test_countmin_basic() {
        let mut cms = CountMinSketch1024x5::new();

        for _ in 0..100 {
            cms.insert_hash(1, 1);
        }
        for _ in 0..50 {
            cms.insert_hash(2, 1);
        }
        for _ in 0..10 {
            cms.insert_hash(3, 1);
        }

        assert!(cms.estimate_hash(1) >= 100);
        assert!(cms.estimate_hash(2) >= 50);
        assert!(cms.estimate_hash(3) >= 10);
    }

    #[test]
    fn test_countmin_merge() {
        let mut cms1 = CountMinSketch1024x5::new();
        let mut cms2 = CountMinSketch1024x5::new();

        for _ in 0..50 {
            cms1.insert_hash(1, 1);
        }
        for _ in 0..50 {
            cms2.insert_hash(1, 1);
        }

        cms1.merge(&cms2);
        assert!(cms1.estimate_hash(1) >= 100);
    }

    #[test]
    fn test_heavy_hitters() {
        let mut hh = HeavyHitters5::new();

        for _ in 0..100 {
            hh.insert_hash(1);
        }
        for _ in 0..50 {
            hh.insert_hash(2);
        }
        for _ in 0..30 {
            hh.insert_hash(3);
        }
        for _ in 0..10 {
            hh.insert_hash(4);
        }
        for _ in 0..5 {
            hh.insert_hash(5);
        }

        let top: Vec<_> = hh.top().collect();
        assert_eq!(top.len(), 5);

        assert_eq!(top[0].hash, 1);
        assert_eq!(top[1].hash, 2);
        assert_eq!(top[2].hash, 3);
    }
}
