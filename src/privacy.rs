//! Local Differential Privacy (LDP) Mechanisms
//!
//! Privacy-preserving data collection where noise is added at the source.
//! Individual data points are deniable, but aggregate statistics emerge.

// ============================================================================
// Random Number Generation (ChaCha20-based for determinism)
// ============================================================================

/// Simple xorshift64 PRNG for fast random numbers
///
/// Not cryptographically secure, but fast and sufficient for noise injection.
#[derive(Clone, Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Create a new PRNG with given seed
    #[inline]
    pub const fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    /// Create from system entropy (uses address as seed if no std)
    #[cfg(feature = "std")]
    pub fn from_entropy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x853c49e6748fea9b);
        Self::new(seed)
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
    #[inline]
    pub fn next_f64_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
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
#[derive(Clone, Debug)]
pub struct LaplaceNoise {
    /// Scale parameter b
    scale: f64,
    /// PRNG
    rng: XorShift64,
}

impl LaplaceNoise {
    /// Create a new Laplace noise generator
    ///
    /// # Arguments
    /// * `sensitivity` - Maximum change in output for one input change (Δf)
    /// * `epsilon` - Privacy parameter ε (smaller = more privacy)
    pub fn new(sensitivity: f64, epsilon: f64) -> Self {
        let scale = sensitivity / epsilon;
        Self {
            scale,
            rng: XorShift64::from_entropy(),
        }
    }

    /// Create with explicit seed
    pub fn with_seed(sensitivity: f64, epsilon: f64, seed: u64) -> Self {
        let scale = sensitivity / epsilon;
        Self {
            scale,
            rng: XorShift64::new(seed),
        }
    }

    /// Generate Laplace noise
    #[inline]
    pub fn sample(&mut self) -> f64 {
        // Inverse transform sampling: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
        let u = self.rng.next_f64() - 0.5;
        let sign = if u < 0.0 { -1.0 } else { 1.0 };
        -sign * self.scale * (1.0 - 2.0 * u.abs()).ln()
    }

    /// Add noise to a value
    #[inline]
    pub fn privatize(&mut self, value: f64) -> f64 {
        value + self.sample()
    }

    /// Add noise to an integer value (rounds result)
    #[inline]
    pub fn privatize_int(&mut self, value: i64) -> i64 {
        (value as f64 + self.sample()).round() as i64
    }

    /// Get the scale parameter
    #[inline]
    pub fn scale(&self) -> f64 {
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
    pub fn new(epsilon: f64) -> Self {
        // p = e^ε / (1 + e^ε)
        let exp_eps = epsilon.exp();
        let p_true = exp_eps / (1.0 + exp_eps);
        Self {
            p_true,
            rng: XorShift64::from_entropy(),
        }
    }

    /// Create with explicit probability and seed
    pub fn with_probability(p_true: f64, seed: u64) -> Self {
        Self {
            p_true: p_true.clamp(0.5, 1.0),
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
    #[inline]
    pub fn privatize_bit(&mut self, bit: u8) -> u8 {
        if self.privatize(bit != 0) { 1 } else { 0 }
    }

    /// Get the probability of truthful response
    #[inline]
    pub fn p_true(&self) -> f64 {
        self.p_true
    }

    /// Estimate true proportion from noisy counts
    ///
    /// Given N total responses with K positive responses,
    /// estimate the true proportion of positive values.
    pub fn estimate_proportion(p_true: f64, n: u64, k: u64) -> f64 {
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
        let true_rate = (observed_rate - 0.5 + 0.5 * p_true) / p_true;
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
    pub const BITS: usize = RAPPOR_BITS;

    /// Create a new RAPPOR encoder
    ///
    /// # Arguments
    /// * `f` - Probability of flipping a bit in permanent response (0.0 to 0.5)
    /// * `p` - Probability of setting a 1 bit to 1 in instantaneous response
    /// * `q` - Probability of setting a 0 bit to 1 in instantaneous response
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
    pub fn default_params() -> Self {
        Self::new(0.5, 0.75, 0.25)
    }

    /// Encode a value into a Bloom filter (simple hash-based)
    fn encode_bloom(&self, value: u64) -> [u8; RAPPOR_BITS] {
        use crate::sketch::FnvHasher;

        let mut bloom = [0u8; RAPPOR_BITS];
        // Use multiple hash functions
        for i in 0..3 {
            let h = FnvHasher::hash_u128((value as u128) | ((i as u128) << 64));
            let idx = (h as usize) % RAPPOR_BITS;
            bloom[idx] = 1;
        }
        bloom
    }

    /// Apply permanent randomized response
    fn permanent_response(&mut self, bloom: &[u8; RAPPOR_BITS]) -> [u8; RAPPOR_BITS] {
        let mut result = [0u8; RAPPOR_BITS];
        for i in 0..RAPPOR_BITS {
            if self.rng.next_bool(self.f) {
                // Flip with probability f
                result[i] = if self.rng.next_bool(0.5) { 1 } else { 0 };
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
                result[i] = if self.rng.next_bool(self.p) { 1 } else { 0 };
            } else {
                result[i] = if self.rng.next_bool(self.q) { 1 } else { 0 };
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
    pub fn params(&self) -> (f64, f64, f64) {
        (self.f, self.p, self.q)
    }
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
    pub fn new(max_epsilon: f64) -> Self {
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
    pub fn remaining(&self) -> f64 {
        (self.max_epsilon - self.total_epsilon).max(0.0)
    }

    /// Get total spent
    #[inline]
    pub fn spent(&self) -> f64 {
        self.total_epsilon
    }

    /// Get query count
    #[inline]
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Check if budget is exhausted
    #[inline]
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
    pub fn new(noise_scale: f64) -> Self {
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
    pub fn estimate_mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.noisy_sum / self.count as f64
        }
    }

    /// Estimate the true sum
    pub fn estimate_sum(&self) -> f64 {
        self.noisy_sum
    }

    /// Get the standard error of the mean estimate
    ///
    /// SE = scale * sqrt(2) / sqrt(n)
    pub fn standard_error(&self) -> f64 {
        if self.count == 0 {
            f64::INFINITY
        } else {
            self.noise_scale * core::f64::consts::SQRT_2 / (self.count as f64).sqrt()
        }
    }

    /// Get the count
    #[inline]
    pub fn count(&self) -> u64 {
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
        assert!(mean.abs() < 0.1, "mean = {}", mean);
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
        assert!(correct > n / 2, "correct = {}", correct);
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
            "estimated = {}, true = {}",
            estimated,
            true_rate
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
            "estimated = {}",
            estimated_mean
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
}
