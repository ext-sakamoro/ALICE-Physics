//! Probabilistic Sketch Data Structures
//!
//! - HyperLogLog++: Cardinality estimation with ~1.04/√m standard error
//! - `DDSketch`: Relative-error quantile estimation
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

/// Precomputed 2^{-k} values for k = 0..64 (`HyperLogLog` optimization)
/// Eliminates expensive `powi()` calls in cardinality estimation
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

/// splitmix64 output for the input `x`: `x + 0x9E37_79B9_7F4A_7C15`, then the
/// splitmix64 finaliser (xor-shift / multiply, Stafford variant 13).
///
/// A bijection on `u64` that spreads structured keys (small integers, packed
/// id pairs, values whose low bits are zero) over all 64 bits. `HyperLogLog`
/// takes its register index from the low `P` bits of an already-mixed hash,
/// so raw structured keys must pass through this before `insert_hash`
/// (AUD-A-S5W1-006, AUD-A-S5W3-017).
#[inline]
#[must_use]
pub(crate) const fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FnvHasher {
    state: u64,
}

impl FnvHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    /// Create a new hasher with default FNV offset basis.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    /// Avalanche bit mixer (from `MurmurHash3` finalizer).
    /// Ensures all bits are well-distributed for `HyperLogLog`.
    #[inline]
    const fn mix(mut h: u64) -> u64 {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }

    /// Hash a byte slice and return a mixed 64-bit digest.
    #[inline]
    #[must_use]
    pub fn hash_bytes(data: &[u8]) -> u64 {
        let mut hasher = Self::new();
        hasher.write(data);
        Self::mix(hasher.state)
    }

    /// Hash a `u64` value.
    #[inline]
    #[must_use]
    pub fn hash_u64(value: u64) -> u64 {
        Self::hash_bytes(&value.to_le_bytes())
    }

    /// Hash a `u128` value.
    #[inline]
    #[must_use]
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

/// `HyperLogLog++` for cardinality (unique count) estimation, generic over the
/// register count `M` (a power of two, `M = 2^P`).
///
/// Memory: `M` bytes. Error: ~1.04 / sqrt(`M`). The named sizes
/// [`HyperLogLog10`] / [`HyperLogLog12`] / [`HyperLogLog14`] /
/// [`HyperLogLog16`] and the default [`HyperLogLog`] are aliases of this type.
///
/// ```
/// use alice_physics::sketch::{HyperLogLog12, HyperLogLogN};
///
/// let mut hll: HyperLogLogN<4096> = HyperLogLog12::new();
/// hll.insert(&7u64);
/// assert_eq!(HyperLogLogN::<4096>::P, 12);
/// ```
///
/// A register count that is not a power of two does not compile:
///
/// ```compile_fail,E0080
/// let _ = alice_physics::sketch::HyperLogLogN::<1000>::new();
/// ```
#[derive(Clone, Debug)]
pub struct HyperLogLogN<const M: usize> {
    /// Registers storing maximum leading zeros + 1
    registers: [u8; M],
}

impl<const M: usize> HyperLogLogN<M> {
    /// Number of registers (m = 2^P)
    pub const M: usize = M;
    /// Bits used for register index
    pub const P: usize = {
        assert!(
            M.is_power_of_two(),
            "HyperLogLogN<M>: M must be a power of two"
        );
        M.trailing_zeros() as usize
    };

    /// Alpha constant for bias correction
    const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / (M as f64));

    /// Create a new empty `HyperLogLog`
    #[inline]
    pub const fn new() -> Self {
        // Evaluating `P` runs the power-of-two check at compile time for every
        // instantiation, since every value of this type starts here.
        let _p: usize = Self::P;
        Self {
            registers: [0u8; M],
        }
    }

    /// Insert an already-hashed value
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
        let idx = (hash as usize) & (Self::M - 1);
        let w = hash >> Self::P;
        // rho = position of first 1 bit in the (64-P) remaining bits
        // leading_zeros(w) includes the P bits we shifted away, so subtract them
        let rho = if w == 0 {
            (64 - Self::P + 1) as u8
        } else {
            (w.leading_zeros() as usize - Self::P + 1) as u8
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

    /// Estimate cardinality using `HyperLogLog++` algorithm
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
            m * crate::det_math::ln64(m / zeros as f64)
        } else {
            raw_estimate
        }
    }

    /// Get raw registers
    #[inline]
    pub const fn registers(&self) -> &[u8] {
        &self.registers
    }

    /// Reset all registers to zero
    #[inline]
    pub fn clear(&mut self) {
        self.registers = [0u8; M];
    }
}

impl<const M: usize> Default for HyperLogLogN<M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const M: usize> Mergeable for HyperLogLogN<M> {
    fn merge(&mut self, other: &Self) {
        for (dst, &src) in self.registers.iter_mut().zip(other.registers.iter()) {
            if src > *dst {
                *dst = src;
            }
        }
    }
}

/// `HyperLogLog` with 2^10 registers (1KB, ~3.2% error)
pub type HyperLogLog10 = HyperLogLogN<1024>;
/// `HyperLogLog` with 2^12 registers (4KB, ~1.6% error)
pub type HyperLogLog12 = HyperLogLogN<4096>;
/// `HyperLogLog` with 2^14 registers (16KB, ~0.8% error)
pub type HyperLogLog14 = HyperLogLogN<16384>;
/// `HyperLogLog` with 2^16 registers (64KB, ~0.4% error)
pub type HyperLogLog16 = HyperLogLogN<65536>;

/// Type alias for the most common `HyperLogLog` size (16KB, ~0.8% error)
pub type HyperLogLog = HyperLogLog14;

// ============================================================================
// DDSketch - Relative Error Quantile Estimation
// ============================================================================

/// `DDSketch` for quantile estimation with relative error guarantee, generic
/// over the number of bins per side `BINS`.
///
/// Guarantees that for any quantile q, the returned value v satisfies:
/// |v - `true_value`| <= α * `true_value`
///
/// where α is the relative accuracy (e.g., 0.01 for 1% error), chosen at
/// construction and independent of `BINS`.
///
/// Each side holds `BINS` consecutive buckets `(γ^(k−1), γ^k]`. The window of
/// keys `k` follows the data: it starts centred a quarter of the way below
/// 1.0 and shifts when a value falls outside it, so the guarantee holds for
/// any data whose magnitudes span fewer than `BINS` buckets (a ratio of
/// `γ^BINS` between the largest and the smallest non-zero magnitude, about
/// 6·10¹⁷ for `DDSketch2048` at α = 0.01). Data spanning more buckets
/// collapses its smallest magnitudes into the lowest bucket (the collapsing
/// scheme of Masson et al. 2019), which keeps the guarantee for the quantiles
/// whose magnitudes still have their own bucket. The named sizes [`DDSketch128`] …
/// [`DDSketch2048`] and the default [`DDSketch`] are aliases of this type.
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
#[derive(Clone, Debug)]
pub struct DDSketchN<const BINS: usize> {
    positive_bins: [u64; BINS],
    negative_bins: [u64; BINS],
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

impl<const BINS: usize> DDSketchN<BINS> {
    /// Number of bins per side (positive / negative).
    pub const BINS: usize = BINS;

    /// Create a new sketch with given relative accuracy `alpha`.
    pub fn new(alpha: f64) -> Self {
        let gamma = (1.0 + alpha) / (1.0 - alpha);
        let ln_gamma = crate::det_math::ln64(gamma);
        // Offset to center around 1.0 (ln(1.0) = 0)
        // For typical latencies (1ms - 10s), we want indices to fit in BINS
        // With offset at BINS/4, we can handle values from gamma^(-BINS/4) to gamma^(3*BINS/4)
        let offset = (BINS / 4) as i32;

        Self {
            positive_bins: [0u64; BINS],
            negative_bins: [0u64; BINS],
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
    ///
    /// NaN and ±infinity are rejected outright (not counted, not
    /// binned, `sum` / `min` / `max` untouched): NaN would be
    /// counted as a zero value (neither `> 0.0` nor `< 0.0`) and
    /// poison `sum`/`mean` permanently, and ±infinity would make
    /// `bucket_index`'s `ceil() as i32 + self.offset` overflow
    /// (panic in debug builds, silently wrap in release).
    #[inline]
    pub fn insert(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        self.sum += value;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        if value > 0.0 {
            self.add_at_key(false, self.bucket_key(value), 1);
        } else if value < 0.0 {
            self.add_at_key(true, self.bucket_key(-value), 1);
        } else {
            self.zero_count += 1;
        }
    }

    /// Bucket key `k = ⌈ln(v) / ln γ⌉` of a positive magnitude, so that
    /// `v ∈ (γ^(k−1), γ^k]`.
    /// Uses standard `ln()` for quantile accuracy (`DDSketch` requires precise buckets)
    #[inline]
    fn bucket_key(&self, magnitude: f64) -> i64 {
        // `as i64` saturates; every finite magnitude at a usable α is far inside
        (crate::det_math::ln64(magnitude) / self.ln_gamma).ceil() as i64
    }

    /// Add `n` to the bucket of `key` on one side, shifting the window first
    /// when the key lies outside it.
    fn add_at_key(&mut self, negative: bool, key: i64, n: u64) {
        let mut idx = key.saturating_add(i64::from(self.offset));
        // below a window whose top bucket is in use: the window cannot move
        // down without dropping that bucket, so the key collapses into bucket
        // 0 exactly as `make_room` would decide, without its O(BINS) pass
        if idx < 0 && (self.positive_bins[BINS - 1] != 0 || self.negative_bins[BINS - 1] != 0) {
            idx = 0;
        } else if idx < 0 || idx >= BINS as i64 {
            self.make_room(key);
            // after a collapse the key may still lie below the window
            idx = key
                .saturating_add(i64::from(self.offset))
                .clamp(0, BINS as i64 - 1);
        }
        let bins = if negative {
            &mut self.negative_bins
        } else {
            &mut self.positive_bins
        };
        bins[idx as usize] = bins[idx as usize].saturating_add(n);
    }

    /// Lowest and highest occupied index over both sides, if any.
    fn occupied(&self) -> Option<(usize, usize)> {
        let used = |i: usize| self.positive_bins[i] != 0 || self.negative_bins[i] != 0;
        let lo = (0..BINS).find(|&i| used(i))?;
        let hi = (0..BINS).rev().find(|&i| used(i))?;
        Some((lo, hi))
    }

    /// Move the window so that `key` gets a bucket: centre the occupied keys
    /// and `key` when they span fewer than `BINS` buckets, otherwise keep the
    /// highest key at the top and collapse everything below the window into
    /// bucket 0.
    fn make_room(&mut self, key: i64) {
        let bins = BINS as i64;
        let (lo_key, hi_key) = match self.occupied() {
            Some((lo, hi)) => {
                let off = i64::from(self.offset);
                (
                    (lo as i64).saturating_sub(off).min(key),
                    (hi as i64).saturating_sub(off).max(key),
                )
            }
            None => (key, key),
        };
        let span = hi_key.saturating_sub(lo_key);
        let new_offset = if span < bins {
            ((bins - 1 - span) / 2).saturating_sub(lo_key)
        } else {
            (bins - 1).saturating_sub(hi_key)
        };
        let shift = new_offset.saturating_sub(i64::from(self.offset));
        for side in [&mut self.positive_bins, &mut self.negative_bins] {
            let old = *side;
            *side = [0u64; BINS];
            for (i, &c) in old.iter().enumerate() {
                if c != 0 {
                    let j = (i as i64).saturating_add(shift).clamp(0, bins - 1) as usize;
                    side[j] = side[j].saturating_add(c);
                }
            }
        }
        // the window only ever holds keys an i32 offset can address
        self.offset = new_offset.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    }

    #[inline]
    fn bucket_lower_bound(&self, idx: usize) -> f64 {
        let exp = (idx as i64 - i64::from(self.offset)) as f64;
        crate::det_math::powf64(self.gamma, exp - 1.0)
    }

    /// Estimate the value at quantile `q` (0.0–1.0).
    ///
    /// Returns the matched bucket's lower edge `γ^(i−1)`, not the
    /// paper's mid-point estimator `2γ^i/(γ+1)`. The relative-error
    /// guarantee on the mid-point is `α`; on the edge it is
    /// `2α/(1+α)` (worst case, measured: ~1.98% at `α = 0.01`), not
    /// `α` itself — a caller reading this doc as an `α`-accurate
    /// quantile estimator (open issue
    /// `sketch-quantile-edge-vs-midpoint`) was getting up to
    /// 2× the documented error.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // rank 1 is the smallest value: q = 0 (and any q whose rank rounds to 0,
        // including negative q and NaN) answers with the smallest bucket
        let rank = ((q * self.count as f64).ceil() as u64).max(1);
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
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all inserted values.
    #[inline]
    pub const fn sum(&self) -> f64 {
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
    pub const fn min(&self) -> f64 {
        self.min
    }

    /// Maximum inserted value.
    #[inline]
    pub const fn max(&self) -> f64 {
        self.max
    }

    /// Relative accuracy parameter.
    #[inline]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Reset the sketch to empty state.
    pub fn clear(&mut self) {
        self.positive_bins = [0u64; BINS];
        self.negative_bins = [0u64; BINS];
        self.zero_count = 0;
        self.count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.sum = 0.0;
        self.offset = (BINS / 4) as i32;
    }
}

impl<const BINS: usize> Mergeable for DDSketchN<BINS> {
    /// Adds `other`'s buckets by key, so sketches whose windows have moved
    /// apart merge correctly. Both sketches must use the same α.
    fn merge(&mut self, other: &Self) {
        let other_offset = i64::from(other.offset);
        for i in 0..BINS {
            let key = i as i64 - other_offset;
            if other.positive_bins[i] != 0 {
                self.add_at_key(false, key, other.positive_bins[i]);
            }
            if other.negative_bins[i] != 0 {
                self.add_at_key(true, key, other.negative_bins[i]);
            }
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

/// `DDSketch` with 128 bins per side. Small, use alpha >= 0.1
pub type DDSketch128 = DDSketchN<128>;
/// `DDSketch` with 256 bins per side. Medium, use alpha >= 0.05
pub type DDSketch256 = DDSketchN<256>;
/// `DDSketch` with 512 bins per side. Good balance
pub type DDSketch512 = DDSketchN<512>;
/// `DDSketch` with 1024 bins per side. High accuracy, alpha >= 0.02
pub type DDSketch1024 = DDSketchN<1024>;
/// `DDSketch` with 2048 bins per side. Very high accuracy, alpha >= 0.01
pub type DDSketch2048 = DDSketchN<2048>;

/// Type alias for the most common `DDSketch` size (good for alpha=0.01)
pub type DDSketch = DDSketch2048;

// ============================================================================
// Count-Min Sketch - Frequency Estimation
// ============================================================================

/// Count-Min Sketch for frequency estimation, generic over the width `W`
/// (columns) and depth `D` (hash rows).
///
/// The named sizes [`CountMinSketch1024x5`] / [`CountMinSketch2048x7`] /
/// [`CountMinSketch4096x5`] and the default [`CountMinSketch`] are aliases of
/// this type.
#[derive(Clone, Debug)]
pub struct CountMinSketchN<const W: usize, const D: usize> {
    counters: [[u64; W]; D],
    total: u64,
}

impl<const W: usize, const D: usize> CountMinSketchN<W, D> {
    /// Number of columns (width).
    pub const WIDTH: usize = W;
    /// Number of hash rows (depth).
    pub const DEPTH: usize = D;

    /// Create an empty sketch.
    #[inline]
    pub const fn new() -> Self {
        Self {
            counters: [[0u64; W]; D],
            total: 0,
        }
    }

    #[inline]
    const fn hash_for_row(hash: u64, row: usize) -> usize {
        let h = hash.wrapping_add((row as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let mixed = h ^ (h >> 33);
        let mixed = mixed.wrapping_mul(0xff51afd7ed558ccd);
        let mixed = mixed ^ (mixed >> 33);
        (mixed as usize) % W
    }

    /// Insert a pre-hashed item with the given count.
    #[inline]
    pub fn insert_hash(&mut self, hash: u64, count: u64) {
        // saturates like the counters
        self.total = self.total.saturating_add(count);
        for row in 0..D {
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
        for row in 0..D {
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
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Reset all counters to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.counters = [[0u64; W]; D];
        self.total = 0;
    }

    /// Theoretical error bound (ε = e / width).
    #[inline]
    pub fn error_bound(&self) -> f64 {
        core::f64::consts::E / (W as f64)
    }

    /// Confidence level (1 − e^{−depth}).
    #[inline]
    pub fn confidence(&self) -> f64 {
        1.0 - crate::det_math::exp64(-(D as f64))
    }
}

impl<const W: usize, const D: usize> Default for CountMinSketchN<W, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const W: usize, const D: usize> Mergeable for CountMinSketchN<W, D> {
    fn merge(&mut self, other: &Self) {
        self.total = self.total.saturating_add(other.total);
        for row in 0..D {
            for col in 0..W {
                self.counters[row][col] =
                    self.counters[row][col].saturating_add(other.counters[row][col]);
            }
        }
    }
}

/// Count-Min Sketch with 1024 columns and 5 rows
pub type CountMinSketch1024x5 = CountMinSketchN<1024, 5>;
/// Count-Min Sketch with 2048 columns and 7 rows
pub type CountMinSketch2048x7 = CountMinSketchN<2048, 7>;
/// Count-Min Sketch with 4096 columns and 5 rows
pub type CountMinSketch4096x5 = CountMinSketchN<4096, 5>;

/// Type alias for the default Count-Min Sketch
pub type CountMinSketch = CountMinSketch1024x5;

// ============================================================================
// Heavy Hitters (Top-K using Count-Min Sketch + Heap)
// ============================================================================

/// Entry for heavy hitters tracking
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeavyHitterEntry {
    /// Hash of the item
    pub hash: u64,
    /// Estimated frequency
    pub count: u64,
}

/// Heavy Hitters tracker using Count-Min Sketch, generic over the number of
/// tracked entries `K` and the underlying sketch's width `W` and depth `D`.
///
/// The named sizes [`HeavyHitters5`] / [`HeavyHitters10`] (on a
/// [`CountMinSketch1024x5`]) / [`HeavyHitters20`] (on a
/// [`CountMinSketch2048x7`]) and the default [`HeavyHitters`] are aliases of
/// this type.
#[derive(Clone, Debug)]
pub struct HeavyHittersN<const K: usize, const W: usize, const D: usize> {
    cms: CountMinSketchN<W, D>,
    top_k: [HeavyHitterEntry; K],
    count: usize,
}

impl<const K: usize, const W: usize, const D: usize> HeavyHittersN<K, W, D> {
    /// Maximum tracked heavy hitters.
    pub const K: usize = K;

    /// Create a new empty tracker.
    #[inline]
    pub const fn new() -> Self {
        Self {
            cms: CountMinSketchN::new(),
            top_k: [HeavyHitterEntry { hash: 0, count: 0 }; K],
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
        } else if self.count < K {
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
    pub const fn cms(&self) -> &CountMinSketchN<W, D> {
        &self.cms
    }

    /// Reset the tracker and its underlying sketch.
    pub fn clear(&mut self) {
        self.cms.clear();
        self.top_k = [HeavyHitterEntry { hash: 0, count: 0 }; K];
        self.count = 0;
    }
}

impl<const K: usize, const W: usize, const D: usize> Default for HeavyHittersN<K, W, D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Heavy Hitters tracking the top 10 on a [`CountMinSketch1024x5`]
pub type HeavyHitters10 = HeavyHittersN<10, 1024, 5>;
/// Heavy Hitters tracking the top 20 on a [`CountMinSketch2048x7`]
pub type HeavyHitters20 = HeavyHittersN<20, 2048, 7>;
/// Heavy Hitters tracking the top 5 on a [`CountMinSketch1024x5`]
pub type HeavyHitters5 = HeavyHittersN<5, 1024, 5>;

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
            "estimate = {estimate}"
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
            "estimate = {estimate}"
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
            "estimate = {estimate}"
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
        assert!(p50 > 40.0 && p50 < 70.0, "p50 = {p50}");

        let p99 = sketch.quantile(0.99);
        assert!(p99 > 80.0 && p99 <= 100.0, "p99 = {p99}");
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

    #[test]
    fn countmin_error_bound_is_e_over_width() {
        let e = core::f64::consts::E;
        let cases: [(f64, usize); 3] = [
            (CountMinSketch1024x5::new().error_bound(), 1024),
            (CountMinSketch2048x7::new().error_bound(), 2048),
            (CountMinSketch4096x5::new().error_bound(), 4096),
        ];
        for (bound, width) in cases {
            let expected = e / width as f64;
            assert!(
                (bound - expected).abs() < 1e-15,
                "width={width}: bound={bound} expected={expected}"
            );
        }
        // The bound is a property of the width, not of the contents.
        let mut cms = CountMinSketch1024x5::new();
        let empty_bound = cms.error_bound();
        for i in 0..500u64 {
            cms.insert_hash(i, 3);
        }
        assert!((cms.error_bound() - empty_bound).abs() < 1e-15);
        // Wider sketch → tighter bound.
        assert!(
            CountMinSketch4096x5::new().error_bound() < CountMinSketch1024x5::new().error_bound()
        );
    }

    #[test]
    fn countmin_insert_bytes_and_estimate_bytes_roundtrip() {
        let mut cms = CountMinSketch2048x7::new();
        assert_eq!(cms.estimate_bytes(b"never-inserted"), 0);

        for _ in 0..7 {
            cms.insert_bytes(b"alpha");
        }
        for _ in 0..3 {
            cms.insert_bytes(b"beta");
        }
        assert_eq!(cms.total(), 10);

        // Count-Min never under-estimates; with 2048 columns × 7 rows and
        // only two distinct keys the min over rows is exact.
        assert_eq!(cms.estimate_bytes(b"alpha"), 7);
        assert_eq!(cms.estimate_bytes(b"beta"), 3);
        assert_eq!(cms.estimate_bytes(b"gamma"), 0);

        // insert_bytes is exactly insert_hash(FnvHasher::hash_bytes(b), 1):
        // the byte path and the pre-hashed path must agree bit-for-bit.
        let h = FnvHasher::hash_bytes(b"alpha");
        assert_eq!(cms.estimate_hash(h), cms.estimate_bytes(b"alpha"));
        let mut via_hash = CountMinSketch2048x7::new();
        for _ in 0..7 {
            via_hash.insert_hash(h, 1);
        }
        assert_eq!(via_hash.estimate_bytes(b"alpha"), 7);
        assert_eq!(via_hash.estimate_bytes(b"beta"), 0);

        // Merging two sketches built through insert_bytes adds counts.
        cms.merge(&via_hash);
        assert_eq!(cms.estimate_bytes(b"alpha"), 14);
        assert_eq!(cms.estimate_bytes(b"beta"), 3);
        assert_eq!(cms.total(), 17);
    }

    #[test]
    fn hyperloglog_insert_bytes_is_idempotent_and_matches_hash_path() {
        let mut hll = HyperLogLog10::new();
        assert!(hll.cardinality().abs() < 1e-12, "empty HLL → 0");

        // Same bytes inserted many times count once.
        for _ in 0..1000 {
            hll.insert_bytes(b"duplicate");
        }
        let one = hll.cardinality();
        assert!(one > 0.5 && one < 1.5, "cardinality={one}");
        let set = hll.registers().iter().filter(|&&r| r != 0).count();
        assert_eq!(set, 1, "exactly one register touched");

        // insert_bytes(b) == insert_hash(FnvHasher::hash_bytes(b)): the two
        // paths must produce identical register arrays.
        let mut via_bytes = HyperLogLog10::new();
        let mut via_hash = HyperLogLog10::new();
        for i in 0..200u32 {
            let key = i.to_le_bytes();
            via_bytes.insert_bytes(&key);
            via_hash.insert_hash(FnvHasher::hash_bytes(&key));
        }
        assert_eq!(via_bytes.registers(), via_hash.registers());
        let est = via_bytes.cardinality();
        // 1024 registers → ~3.2 % typical error; allow 25 % on n = 200
        assert!(est > 150.0 && est < 250.0, "estimate={est}");
    }
}
