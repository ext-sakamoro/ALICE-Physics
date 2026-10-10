//! Local Differential Privacy (LDP) Mechanisms
//!
//! Privacy-preserving data collection where noise is added at the source.
//! Individual data points are deniable, but aggregate statistics emerge.
//!
//! # Which types to use
//!
//! The private mechanisms are those of `alice-crypto` (re-exported here):
//! keyed ChaCha20 ([`SecureRng`]), discrete Laplace noise sampled with integer
//! arithmetic in constant time ([`dp_count`], [`dp_int`], [`dp_sum`],
//! [`DpNoise`]), [`randomized_response`] and [`bernoulli_ratio`], plus
//! [`KeyedRappor`] built on them. [`PrivacyBudget`] and [`PrivateAggregator`]
//! draw no randomness and stay as they are.
//!
//! # Security
//!
//! [`XorShift64`], [`LaplaceNoise`], [`RandomizedResponse`] and [`Rappor`] are
//! **not private** and are deprecated for that reason (removal is a later,
//! breaking release): `XorShift64` hands out its internal state, so one
//! observed draw determines every later one, and its `from_entropy` seed is
//! derived from the clock; `LaplaceNoise::sample` uses the floating-point
//! inverse transform, whose low bits leak the uniform draw (Mironov 2012), so
//! ε does not hold even with a perfect random source.
//!
//! # Examples
//!
//! ```
//! use alice_physics::privacy::{dp_int, KeyedRappor, PrivacyBudget, SecureRng};
//!
//! // integer value with sensitivity 1, ε = 1, from a 32-byte secret key
//! let mut rng = SecureRng::from_key([7u8; 32]);
//! let noisy: i64 = dp_int(100, 1, 1.0, &mut rng).unwrap();
//! assert!((noisy - 100).abs() < 100);
//!
//! // RAPPOR for categorical data, probabilities as exact fractions
//! let mut rappor = KeyedRappor::with_key((1, 2), (3, 4), (1, 4), [9u8; 32]).unwrap();
//! let report = rappor.privatize(12345);
//! assert_eq!(report.len(), 64);
//!
//! // Privacy budget tracking
//! let mut budget = PrivacyBudget::new(10.0);
//! assert!(budget.try_spend(1.0));
//! assert_eq!(budget.remaining(), 9.0);
//! ```

// the deprecated types refer to each other; the deprecation is for callers
#![allow(deprecated)]

pub use alice_crypto::dp::{
    bernoulli_ratio, dp_count, dp_int, dp_sum, randomized_response, DpError, DpNoise, EntropyError,
    SecureRng, RR_MAX_EPSILON_WHOLE,
};

// ============================================================================
// Random Number Generation (ChaCha20-based for determinism)
// ============================================================================

#[deprecated(
    since = "2.3.0",
    note = "not differentially private (predictable xorshift64 noise, clock seed, floating-point inverse transform); use privacy::{dp_int, dp_sum, DpNoise, randomized_response, KeyedRappor}"
)]
/// Simple xorshift64 PRNG for fast random numbers
///
// LIMITATION(COV-ENGINE-101): Not cryptographically secure, but fast and sufficient for noise injection.
/// Not cryptographically secure, but fast and sufficient for noise injection.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new PRNG with given seed
    ///
    /// The seed is scrambled with one splitmix64 step before it becomes the
    /// xorshift state, so nearby seeds (1, 2, 3, ...) start from unrelated
    /// states and their first draws are not correlated or tiny
    /// (AUD-A-S4W3-030). Without this, a small seed is a state with only a few
    /// low bits set and the first outputs of every small seed are below 1e-6.
    ///
    /// The step is the splitmix64 output function: add the golden gamma
    /// `0x9E37_79B9_7F4A_7C15`, then the finaliser
    /// `z ^= z >> 30; z *= 0xBF58_476D_1CE4_E5B9; z ^= z >> 27;
    /// z *= 0x94D0_49BB_1331_11EB; z ^= z >> 31` (wrapping). Source: G. L.
    /// Steele Jr., D. Lea, C. H. Flood, "Fast splittable pseudorandom number
    /// generators", OOPSLA 2014, with the constants of S. Vigna's reference
    /// implementation `splitmix64.c` (<https://prng.di.unimi.it/splitmix64.c>).
    ///
    /// xorshift cannot leave the all-zero state. splitmix64 is a bijection on
    /// `u64`, so exactly one seed scrambles to zero; that seed is mapped to
    /// the fixed non-zero state `0x853C_49E6_748F_EA9B`, so the state is never
    /// zero for any seed.
    #[inline]
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        let z = crate::sketch::splitmix64(seed);
        Self {
            state: if z == 0 { 0x853c49e6748fea9b } else { z },
        }
    }

    /// Create from system entropy (uses address as seed if no std)
    ///
    /// The seed is the wall-clock time hashed with the standard library's
    /// randomly keyed hasher (its keys come from the operating system), so it
    /// cannot be recovered by trying the clock values around the call
    /// (AUD-A-S4W3-029).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn from_entropy() -> Self {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c49e6748fea9b);
        let mut h = RandomState::new().build_hasher();
        h.write_u64(nanos);
        Self::new(h.finish())
    }

    #[cfg(not(feature = "std"))]
    pub fn from_entropy() -> Self {
        // Use a fixed seed in no_std environments
        Self::new(0x853c49e6748fea9b)
    }

    /// Generate next u64
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate uniform f64 in [0, 1)
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate uniform f64 in [low, high)
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[inline]
    pub fn next_f64_range(&mut self, low: f64, high: f64) -> f64 {
        (high - low).mul_add(self.next_f64(), low)
    }

    /// Generate a random boolean with given probability of true
    #[inline]
    pub fn next_bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

impl Default for XorShift64 {
    fn default() -> Self {
        Self::from_entropy()
    }
}

// ============================================================================
// Laplace Distribution
// ============================================================================

/// Generate Laplace-distributed random variable
///
/// Laplace(μ, b) has PDF: f(x) = (1/2b) * exp(-|x-μ|/b)
///
/// Used for ε-differential privacy with sensitivity Δf:
/// noise scale b = Δf / ε
#[deprecated(
    since = "2.3.0",
    note = "not differentially private (predictable xorshift64 noise, clock seed, floating-point inverse transform); use privacy::{dp_int, dp_sum, DpNoise, randomized_response, KeyedRappor}"
)]
// ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
#[derive(Clone, Debug)]
pub struct LaplaceNoise {
    /// Scale parameter b
    scale: f64,
    /// PRNG
    rng: XorShift64,
}

/// `sensitivity / epsilon`, refusing an `epsilon` that makes the mechanism
/// meaningless.
fn laplace_scale(sensitivity: f64, epsilon: f64) -> f64 {
    assert!(
        epsilon.is_finite() && epsilon > 0.0,
        "Laplace epsilon must be finite and positive, got {epsilon}"
    );
    sensitivity / epsilon
}

impl LaplaceNoise {
    /// Create a new Laplace noise generator
    ///
    /// # Arguments
    /// * `sensitivity` - Maximum change in output for one input change (Δf)
    /// * `epsilon` - Privacy parameter ε (smaller = more privacy)
    ///
    /// # Panics
    ///
    /// Panics unless `epsilon` is finite and positive (`ε = 0` would need
    /// infinite noise: every sample was ±inf).
    #[must_use]
    pub fn new(sensitivity: f64, epsilon: f64) -> Self {
        let scale = laplace_scale(sensitivity, epsilon);
        Self {
            scale,
            rng: XorShift64::from_entropy(),
        }
    }

    /// Create with explicit seed
    ///
    /// # Panics
    ///
    /// As [`Self::new`]: unless `epsilon` is finite and positive.
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[must_use]
    pub fn with_seed(sensitivity: f64, epsilon: f64, seed: u64) -> Self {
        let scale = laplace_scale(sensitivity, epsilon);
        Self {
            scale,
            rng: XorShift64::new(seed),
        }
    }

    /// Generate Laplace noise
    #[inline]
    pub fn sample(&mut self) -> f64 {
        // Inverse transform sampling: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
        // U is drawn from [0, 1); U = 0 would give ln(0) = -inf, so that one value
        // is moved half a step in (2^-54), every other draw is unchanged
        let raw = self.rng.next_f64();
        let raw = if raw == 0.0 {
            0.5 / (1u64 << 53) as f64
        } else {
            raw
        };
        let u = raw - 0.5;
        let sign = if u < 0.0 { -1.0 } else { 1.0 };
        -sign * self.scale * crate::det_math::ln64(2.0f64.mul_add(-u.abs(), 1.0))
    }

    /// Add noise to a value
    #[inline]
    pub fn privatize(&mut self, value: f64) -> f64 {
        value + self.sample()
    }

    /// Add noise to an integer value (rounds result)
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[inline]
    pub fn privatize_int(&mut self, value: i64) -> i64 {
        (value as f64 + self.sample()).round() as i64
    }

    /// Get the scale parameter
    #[inline]
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }
}

// ============================================================================
// Randomized Response (RAPPOR-style)
// ============================================================================

/// Randomized Response for binary data
///
/// Classic technique: to report a sensitive bit b:
/// - With probability p, report b truthfully
/// - With probability 1-p, report a random bit
///
/// This provides ε-differential privacy where ε = ln((p + 0.5(1-p)) / (0.5(1-p)))
#[deprecated(
    since = "2.3.0",
    note = "not differentially private (predictable xorshift64 noise, clock seed, floating-point inverse transform); use privacy::{dp_int, dp_sum, DpNoise, randomized_response, KeyedRappor}"
)]
// ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
#[derive(Clone, Debug)]
pub struct RandomizedResponse {
    /// Probability of truthful response
    p_true: f64,
    /// PRNG
    rng: XorShift64,
}

impl RandomizedResponse {
    /// Create from privacy parameter epsilon
    ///
    /// Higher epsilon = more accuracy, less privacy
    #[must_use]
    pub fn new(epsilon: f64) -> Self {
        // Truthful with probability p, otherwise a fair coin: the report
        // probabilities are (1 + p) / 2 and (1 - p) / 2, so ε = ln((1 + p) / (1 - p))
        // and p = (e^ε - 1) / (e^ε + 1), written as 1 - 2 / (e^ε + 1) so a large
        // ε gives 1 rather than ∞/∞ (AUD-A-S4W3-020). A negative ε clamps to 0.
        let exp_eps = crate::det_math::exp64(epsilon);
        let p_true = (1.0 - 2.0 / (exp_eps + 1.0)).clamp(0.0, 1.0);
        Self {
            p_true,
            rng: XorShift64::from_entropy(),
        }
    }

    /// Create with explicit probability and seed
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[must_use]
    pub fn with_probability(p_true: f64, seed: u64) -> Self {
        Self {
            // any p in [0, 1] is a valid mechanism (p < 1/2 is more private,
            // not invalid), so only the range is enforced (AUD-A-S4W3-022)
            p_true: p_true.clamp(0.0, 1.0),
            rng: XorShift64::new(seed),
        }
    }

    /// Privatize a boolean value
    #[inline]
    pub fn privatize(&mut self, value: bool) -> bool {
        if self.rng.next_bool(self.p_true) {
            // Report truthfully
            value
        } else {
            // Report random
            self.rng.next_bool(0.5)
        }
    }

    /// Privatize a bit (0 or 1)
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[inline]
    pub fn privatize_bit(&mut self, bit: u8) -> u8 {
        u8::from(self.privatize(bit != 0))
    }

    /// Get the probability of truthful response
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[inline]
    #[must_use]
    pub const fn p_true(&self) -> f64 {
        self.p_true
    }

    /// Estimate true proportion from noisy counts
    ///
    /// Given N total responses with K positive responses,
    /// estimate the true proportion of positive values.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < p_true ≤ 1`: with `p_true = 0` every report is a coin
    /// flip and carries no information (the estimate was NaN).
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[must_use]
    pub fn estimate_proportion(p_true: f64, n: u64, k: u64) -> f64 {
        assert!(
            p_true > 0.0 && p_true <= 1.0,
            "randomized response p must be in (0, 1], got {p_true}"
        );
        if n == 0 {
            return 0.0;
        }
        let observed_rate = k as f64 / n as f64;
        // Debiasing: true_rate = (observed_rate - 0.5*(1-p)) / (p - 0.5*(1-p))
        // Simplifies to: true_rate = (observed_rate - 0.5 + 0.5*p) / (p - 0.5 + 0.5*p)
        //              = (2*observed_rate - 1 + p) / (2p - 1 + p)
        //              = (2*observed_rate - 1 + p) / (3p - 1)
        // Wait, let me recalculate...
        // P(report=1) = p*true_rate + (1-p)*0.5
        // observed_rate = p*true_rate + 0.5 - 0.5*p
        // true_rate = (observed_rate - 0.5 + 0.5*p) / p
        let true_rate = 0.5f64.mul_add(p_true, observed_rate - 0.5) / p_true;
        true_rate.clamp(0.0, 1.0)
    }
}

// ============================================================================
// RAPPOR (Randomized Aggregatable Privacy-Preserving Ordinal Response)
// ============================================================================

/// Default RAPPOR bit size
pub const RAPPOR_BITS: usize = 64;

/// RAPPOR for categorical data with multiple bits
///
/// Encodes categorical values into a Bloom filter, then applies
/// randomized response to each bit.
#[deprecated(
    since = "2.3.0",
    note = "not differentially private (predictable xorshift64 noise, clock seed, floating-point inverse transform); use privacy::{dp_int, dp_sum, DpNoise, randomized_response, KeyedRappor}"
)]
// ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
#[derive(Clone, Debug)]
pub struct Rappor {
    /// Permanent randomized response (for longitudinal studies)
    f: f64,
    /// Instantaneous randomized response parameters
    p: f64,
    q: f64,
    /// PRNG
    rng: XorShift64,
}

impl Rappor {
    /// Number of bits in the Bloom filter
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    pub const BITS: usize = RAPPOR_BITS;

    /// Create a new RAPPOR encoder
    ///
    /// # Arguments
    /// * `f` - Probability of flipping a bit in permanent response (0.0 to 0.5)
    /// * `p` - Probability of setting a 1 bit to 1 in instantaneous response
    /// * `q` - Probability of setting a 0 bit to 1 in instantaneous response
    #[must_use]
    pub fn new(f: f64, p: f64, q: f64) -> Self {
        Self {
            f: f.clamp(0.0, 0.5),
            p: p.clamp(0.0, 1.0),
            q: q.clamp(0.0, 1.0),
            rng: XorShift64::from_entropy(),
        }
    }

    /// Create with typical parameters for ε-differential privacy
    ///
    /// Uses f=0.5, p=0.75, q=0.25 for approximately ε=2 privacy
    // ALLOW-UNWIRED: wiring debt deprecated-privacy (deprecated, not private: kept for existing callers until the breaking release removes it), oracle tests/audit_privacy.rs
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(0.5, 0.75, 0.25)
    }

    /// Encode a value into a Bloom filter (simple hash-based)
    #[allow(clippy::unused_self)]
    fn encode_bloom(&self, value: u64) -> [u8; RAPPOR_BITS] {
        bloom_bits(value)
    }

    /// Apply permanent randomized response
    fn permanent_response(&mut self, bloom: &[u8; RAPPOR_BITS]) -> [u8; RAPPOR_BITS] {
        let mut result = [0u8; RAPPOR_BITS];
        for i in 0..RAPPOR_BITS {
            if self.rng.next_bool(self.f) {
                // Flip with probability f
                result[i] = u8::from(self.rng.next_bool(0.5));
            } else {
                // Keep original
                result[i] = bloom[i];
            }
        }
        result
    }

    /// Apply instantaneous randomized response
    fn instantaneous_response(&mut self, permanent: &[u8; RAPPOR_BITS]) -> [u8; RAPPOR_BITS] {
        let mut result = [0u8; RAPPOR_BITS];
        for i in 0..RAPPOR_BITS {
            if permanent[i] == 1 {
                result[i] = u8::from(self.rng.next_bool(self.p));
            } else {
                result[i] = u8::from(self.rng.next_bool(self.q));
            }
        }
        result
    }

    /// Encode and privatize a value
    ///
    /// Returns the privatized bit array that can be sent to the aggregator.
    pub fn privatize(&mut self, value: u64) -> [u8; RAPPOR_BITS] {
        let bloom = self.encode_bloom(value);
        let permanent = self.permanent_response(&bloom);
        self.instantaneous_response(&permanent)
    }

    /// Get privacy parameters
    #[must_use]
    pub const fn params(&self) -> (f64, f64, f64) {
        (self.f, self.p, self.q)
    }
}

// ============================================================================
// Keyed RAPPOR
// ============================================================================

/// RAPPOR with a keyed, unpredictable noise source and exact probabilities
///
/// Each probability is a fraction `(num, den)`: `f` for the permanent
/// response (at most 1/2), `p` and `q` for reporting a 1 when the permanent
/// bit is 1 or 0. Every bit draws the same four Bernoulli trials from the
/// keyed ChaCha20 stream ([`bernoulli_ratio`]) and the result is chosen by
/// arithmetic, so the work and the keystream use do not depend on the value
/// being reported. The Bloom encoding is the one [`Rappor`] uses.
pub struct KeyedRappor {
    f: (u64, u64),
    p: (u64, u64),
    q: (u64, u64),
    rng: SecureRng,
}

impl core::fmt::Debug for KeyedRappor {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fmt.debug_struct("KeyedRappor")
            .field("f", &self.f)
            .field("p", &self.p)
            .field("q", &self.q)
            .finish_non_exhaustive()
    }
}

impl KeyedRappor {
    /// A RAPPOR encoder whose noise comes from the 32-byte secret `key`
    ///
    /// # Errors
    ///
    /// [`DpError::InvalidProbability`] when a fraction has `den == 0` or
    /// `num > den`, or when `f` exceeds 1/2.
    pub fn with_key(
        f: (u64, u64),
        p: (u64, u64),
        q: (u64, u64),
        key: [u8; 32],
    ) -> Result<Self, DpError> {
        for (num, den) in [f, p, q] {
            if den == 0 || num > den {
                return Err(DpError::InvalidProbability);
            }
        }
        if u128::from(f.0) * 2 > u128::from(f.1) {
            return Err(DpError::InvalidProbability);
        }
        Ok(Self {
            f,
            p,
            q,
            rng: SecureRng::from_key(key),
        })
    }

    /// The probabilities `(f, p, q)` as given
    #[must_use]
    pub const fn params(&self) -> ((u64, u64), (u64, u64), (u64, u64)) {
        (self.f, self.p, self.q)
    }

    /// Encode and privatize a value: Bloom encoding, permanent response
    /// (each bit replaced by a fair coin with probability `f`), instantaneous
    /// response (a 1 reported with probability `p` for a 1 and `q` for a 0)
    pub fn privatize(&mut self, value: u64) -> [u8; RAPPOR_BITS] {
        let bloom = bloom_bits(value);
        let mut out = [0u8; RAPPOR_BITS];
        for (o, &b) in out.iter_mut().zip(bloom.iter()) {
            let flip = self.coin(self.f);
            let fair = self.coin((1, 2));
            let one = self.coin(self.p);
            let zero = self.coin(self.q);
            let permanent = (flip & fair) | ((1 - flip) & b);
            *o = (permanent & one) | ((1 - permanent) & zero);
        }
        out
    }

    fn coin(&mut self, (num, den): (u64, u64)) -> u8 {
        // the fractions were validated in `with_key`
        u8::from(bernoulli_ratio(num, den, &mut self.rng).unwrap_or(false))
    }
}

/// The Bloom encoding of `value` used by [`Rappor`] and [`KeyedRappor`]:
/// three FNV hashes, one bit each
fn bloom_bits(value: u64) -> [u8; RAPPOR_BITS] {
    use crate::sketch::FnvHasher;
    let mut bloom = [0u8; RAPPOR_BITS];
    for i in 0..3u128 {
        let h = FnvHasher::hash_u128(u128::from(value) | (i << 64));
        bloom[(h as usize) % RAPPOR_BITS] = 1;
    }
    bloom
}

// ============================================================================
// Privacy Budget Tracker
// ============================================================================

/// Track cumulative privacy budget (composition theorem)
///
/// Under sequential composition, ε values add up.
/// Under parallel composition over disjoint data, use max.
#[derive(Clone, Debug)]
pub struct PrivacyBudget {
    /// Total epsilon spent
    total_epsilon: f64,
    /// Maximum allowed epsilon
    max_epsilon: f64,
    /// Number of queries made
    query_count: u64,
}

impl PrivacyBudget {
    /// Create a new privacy budget tracker
    #[must_use]
    pub const fn new(max_epsilon: f64) -> Self {
        Self {
            total_epsilon: 0.0,
            max_epsilon,
            query_count: 0,
        }
    }

    /// Try to spend epsilon from budget
    ///
    /// Returns true if budget allows, false if would exceed.
    pub fn try_spend(&mut self, epsilon: f64) -> bool {
        // a negative (or NaN) spend would give budget back (AUD-A-S4W3-021)
        if epsilon.is_nan() || epsilon < 0.0 {
            return false;
        }
        if self.total_epsilon + epsilon <= self.max_epsilon {
            self.total_epsilon += epsilon;
            self.query_count += 1;
            true
        } else {
            false
        }
    }

    /// Get remaining budget
    #[inline]
    #[must_use]
    pub fn remaining(&self) -> f64 {
        (self.max_epsilon - self.total_epsilon).max(0.0)
    }

    /// Get total spent
    #[inline]
    #[must_use]
    pub const fn spent(&self) -> f64 {
        self.total_epsilon
    }

    /// Get query count
    #[inline]
    #[must_use]
    pub const fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Check if budget is exhausted
    #[inline]
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.total_epsilon >= self.max_epsilon
    }

    /// Reset the budget
    pub fn reset(&mut self) {
        self.total_epsilon = 0.0;
        self.query_count = 0;
    }
}

// ============================================================================
// Differentially Private Aggregator
// ============================================================================

/// Aggregates noisy reports and estimates true statistics
#[derive(Clone, Debug)]
pub struct PrivateAggregator {
    /// Sum of noisy values
    noisy_sum: f64,
    /// Count of reports
    count: u64,
    /// Laplace noise scale used
    noise_scale: f64,
}

impl PrivateAggregator {
    /// Create a new aggregator
    #[must_use]
    pub const fn new(noise_scale: f64) -> Self {
        Self {
            noisy_sum: 0.0,
            count: 0,
            noise_scale,
        }
    }

    /// Add a noisy report
    #[inline]
    pub fn add(&mut self, noisy_value: f64) {
        self.noisy_sum += noisy_value;
        self.count += 1;
    }

    /// Estimate the true mean
    ///
    /// As count increases, noise averages out to zero.
    #[must_use]
    pub fn estimate_mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.noisy_sum / self.count as f64
        }
    }

    /// Estimate the true sum
    #[must_use]
    pub const fn estimate_sum(&self) -> f64 {
        self.noisy_sum
    }

    /// Get the standard error of the mean estimate
    ///
    /// SE = scale * sqrt(2) / sqrt(n)
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            self.noise_scale * core::f64::consts::SQRT_2 / (self.count as f64).sqrt()
        }
    }

    /// Get the count
    #[inline]
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Reset the aggregator
    pub fn reset(&mut self) {
        self.noisy_sum = 0.0;
        self.count = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod keyed_rappor_tests {
    use super::{bloom_bits, DpError, KeyedRappor, RAPPOR_BITS};

    /// With `f = 1/2, p = 3/4, q = 1/4` a Bloom bit of 1 is reported as 1 with
    /// probability `(1 − f)·p + f·(p + q)/2 = 5/8` and a 0 with `3/8` (closed
    /// form from the RAPPOR definition); 4000 reports of one value, each
    /// frequency within 6 standard errors
    #[test]
    fn report_frequencies_match_the_closed_form() {
        let mut r = KeyedRappor::with_key((1, 2), (3, 4), (1, 4), [3u8; 32]).unwrap();
        let bloom = bloom_bits(777);
        let n = 4000u32;
        let mut ones = [0u32; RAPPOR_BITS];
        for _ in 0..n {
            for (c, &bit) in ones.iter_mut().zip(r.privatize(777).iter()) {
                *c += u32::from(bit);
            }
        }
        let (mut set_ones, mut set_n, mut clear_ones, mut clear_n) = (0u32, 0u32, 0u32, 0u32);
        for i in 0..RAPPOR_BITS {
            if bloom[i] == 1 {
                set_ones += ones[i];
                set_n += n;
            } else {
                clear_ones += ones[i];
                clear_n += n;
            }
        }
        for (k, m, p) in [
            (set_ones, set_n, 5.0 / 8.0),
            (clear_ones, clear_n, 3.0 / 8.0),
        ] {
            let mean = p * f64::from(m);
            let sd = (f64::from(m) * p * (1.0 - p)).sqrt();
            assert!(
                (f64::from(k) - mean).abs() <= 6.0 * sd,
                "{k} of {m}, expected {mean}"
            );
        }
    }

    #[test]
    fn f_zero_p_one_q_zero_reports_the_bloom_filter() {
        let mut r = KeyedRappor::with_key((0, 1), (1, 1), (0, 1), [5u8; 32]).unwrap();
        for v in [0u64, 1, 12345, u64::MAX] {
            assert_eq!(r.privatize(v), bloom_bits(v));
        }
    }

    #[test]
    fn the_same_key_replays_and_another_key_differs() {
        let run = |key| {
            let mut r = KeyedRappor::with_key((1, 2), (3, 4), (1, 4), key).unwrap();
            (0..16).map(|v| r.privatize(v)).collect::<Vec<_>>()
        };
        assert_eq!(run([1u8; 32]), run([1u8; 32]));
        assert_ne!(run([1u8; 32]), run([2u8; 32]));
    }

    #[test]
    fn impossible_fractions_and_f_over_one_half_are_refused() {
        let k = [0u8; 32];
        for (f, p, q) in [
            ((1, 0), (1, 2), (1, 2)),
            ((1, 2), (3, 2), (1, 2)),
            ((1, 2), (1, 2), (1, 0)),
            ((3, 5), (1, 2), (1, 2)),
        ] {
            assert_eq!(
                KeyedRappor::with_key(f, p, q, k).map(|r| r.params()),
                Err(DpError::InvalidProbability),
                "{f:?} {p:?} {q:?}"
            );
        }
        assert!(KeyedRappor::with_key((1, 2), (0, 1), (1, 1), k).is_ok());
        assert!(format!(
            "{:?}",
            KeyedRappor::with_key((1, 2), (1, 2), (1, 2), k).unwrap()
        )
        .starts_with("KeyedRappor"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift() {
        let mut rng = XorShift64::new(12345);
        let v1 = rng.next_u64();
        let v2 = rng.next_u64();
        assert_ne!(v1, v2);

        // Test reproducibility
        let mut rng2 = XorShift64::new(12345);
        assert_eq!(v1, rng2.next_u64());
    }

    #[test]
    fn test_laplace_noise() {
        let mut noise = LaplaceNoise::with_seed(1.0, 1.0, 42);

        // Generate many samples and check mean is close to 0
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            sum += noise.sample();
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1, "mean = {mean}");
    }

    #[test]
    fn test_randomized_response() {
        let mut rr = RandomizedResponse::with_probability(0.75, 42);

        // With high p_true, most responses should match truth
        let mut correct = 0;
        let n = 1000;
        for i in 0..n {
            let truth = i % 2 == 0;
            let response = rr.privatize(truth);
            if response == truth {
                correct += 1;
            }
        }

        // Should be correct more than 50% of the time
        assert!(correct > n / 2, "correct = {correct}");
    }

    #[test]
    fn test_rr_proportion_estimation() {
        // Simulate: true rate = 0.3, n = 10000
        let p_true = 0.8;
        let true_rate = 0.3;
        let n = 10000u64;

        let mut rr = RandomizedResponse::with_probability(p_true, 42);

        let mut positive_reports = 0u64;
        for i in 0..n {
            // Simulate true value with 30% positive rate
            let truth = (i as f64 / n as f64) < true_rate;
            if rr.privatize(truth) {
                positive_reports += 1;
            }
        }

        let estimated = RandomizedResponse::estimate_proportion(p_true, n, positive_reports);
        // Should be within 0.1 of true rate
        assert!(
            (estimated - true_rate).abs() < 0.1,
            "estimated = {estimated}, true = {true_rate}"
        );
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(1.0);

        assert!(budget.try_spend(0.3));
        assert!(budget.try_spend(0.3));
        assert!(budget.try_spend(0.3));
        assert!(!budget.try_spend(0.3)); // Would exceed

        assert_eq!(budget.query_count(), 3);
        assert!((budget.spent() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_private_aggregator() {
        let noise_scale = 1.0;
        let mut aggregator = PrivateAggregator::new(noise_scale);
        let mut noise = LaplaceNoise::with_seed(1.0, 1.0, 42);

        // True values are all 100
        let true_value = 100.0;
        let n = 10000;

        for _ in 0..n {
            let noisy = noise.privatize(true_value);
            aggregator.add(noisy);
        }

        let estimated_mean = aggregator.estimate_mean();
        // Should be close to 100
        assert!(
            (estimated_mean - true_value).abs() < 1.0,
            "estimated = {estimated_mean}"
        );
    }

    #[test]
    fn test_rappor() {
        let mut rappor = Rappor::default_params();

        let encoded1 = rappor.privatize(12345);
        let encoded2 = rappor.privatize(12345);

        // Different instances should produce different outputs (randomized)
        // but with same structure (64 bits)
        assert_eq!(encoded1.len(), 64);
        assert_eq!(encoded2.len(), 64);
    }

    #[test]
    fn aggregator_estimate_sum_is_exact_sum_of_inputs() {
        let mut agg = PrivateAggregator::new(0.5);
        assert!(agg.estimate_sum().abs() < 1e-12, "empty sum must be 0");

        // Dyadic values → the running f64 sum is exact, so the contract
        // Σ inputs can be checked to ~1 ulp.
        let inputs = [1.5, -2.25, 4.0, 0.125, 100.0];
        let mut expected = 0.0;
        for &v in &inputs {
            agg.add(v);
            expected += v;
        }
        assert_eq!(agg.count(), inputs.len() as u64);
        assert!(
            (agg.estimate_sum() - expected).abs() < 1e-12,
            "sum={} expected={expected}",
            agg.estimate_sum()
        );
        assert!((agg.estimate_sum() - 103.375).abs() < 1e-12);
        // mean = sum / n
        assert!((agg.estimate_mean() - 103.375 / 5.0).abs() < 1e-12);

        agg.reset();
        assert_eq!(agg.count(), 0);
        assert!(agg.estimate_sum().abs() < 1e-12);
    }

    #[test]
    fn aggregator_standard_error_matches_closed_form() {
        // SE = scale · √2 / √n
        let scale = 0.75;
        let mut agg = PrivateAggregator::new(scale);
        assert!(agg.standard_error().is_infinite(), "n = 0 → SE must be +∞");

        for n in 1..=16u64 {
            agg.add(0.0);
            let expected = scale * core::f64::consts::SQRT_2 / (n as f64).sqrt();
            let se = agg.standard_error();
            assert!(
                (se - expected).abs() < 1e-15,
                "n={n}: se={se} expected={expected}"
            );
        }
        // n = 2 → SE = scale exactly (√2/√2 = 1); n = 8 → SE = scale / 2
        let mut two = PrivateAggregator::new(scale);
        two.add(1.0);
        two.add(1.0);
        assert!((two.standard_error() - scale).abs() < 1e-15);
        let mut eight = PrivateAggregator::new(scale);
        for _ in 0..8 {
            eight.add(1.0);
        }
        assert!((eight.standard_error() - scale / 2.0).abs() < 1e-15);
        // SE is independent of the values added, only of n and scale.
        let mut shifted = PrivateAggregator::new(scale);
        for v in [10.0, -30.0] {
            shifted.add(v);
        }
        assert!((shifted.standard_error() - two.standard_error()).abs() < 1e-15);
    }

    #[test]
    fn budget_is_exhausted_only_when_spent_reaches_max() {
        let mut b = PrivacyBudget::new(1.0);
        assert!(!b.is_exhausted());
        assert!((b.remaining() - 1.0).abs() < 1e-12);

        // Spend in dyadic steps so the running total is exact.
        assert!(b.try_spend(0.5));
        assert!(!b.is_exhausted());
        assert!(b.try_spend(0.25));
        assert!(!b.is_exhausted());
        assert!((b.remaining() - 0.25).abs() < 1e-12);

        // Reaching exactly max_epsilon exhausts the budget (>=).
        assert!(b.try_spend(0.25));
        assert!(b.is_exhausted());
        assert!(b.remaining().abs() < 1e-12);
        assert_eq!(b.query_count(), 3);

        // Once exhausted, any positive spend is refused and state is frozen.
        assert!(!b.try_spend(1e-9));
        assert!(b.is_exhausted());
        assert_eq!(b.query_count(), 3);
        assert!((b.spent() - 1.0).abs() < 1e-12);

        // A refused spend does not exhaust the budget.
        let mut c = PrivacyBudget::new(1.0);
        assert!(!c.try_spend(1.5));
        assert!(!c.is_exhausted());
        assert_eq!(c.query_count(), 0);

        // Zero-capacity budget is exhausted from the start.
        let z = PrivacyBudget::new(0.0);
        assert!(z.is_exhausted());

        // reset clears exhaustion.
        b.reset();
        assert!(!b.is_exhausted());
        assert_eq!(b.query_count(), 0);
    }
}
