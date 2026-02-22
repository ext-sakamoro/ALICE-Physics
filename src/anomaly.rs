//! Streaming Anomaly Detection
//!
//! Real-time outlier detection using robust statistics.
//! Designed for telemetry streams where batch processing is not feasible.
//!
//! # Examples
//!
//! ```
//! use alice_physics::anomaly::{MadDetector, EwmaDetector};
//!
//! // MAD-based outlier detection
//! let mut mad = MadDetector::new(3.0);
//! for &v in &[1.0, 2.0, 1.5, 1.8, 2.1, 1.9] {
//!     mad.observe(v);
//! }
//! assert!(mad.is_anomaly(100.0)); // 100.0 is an outlier
//!
//! // EWMA-based detector
//! let mut ewma = EwmaDetector::new(0.1, 3.0);
//! for _ in 0..50 {
//!     ewma.observe(10.0); // establish baseline
//! }
//! assert!(ewma.is_anomaly(1000.0)); // spike detected
//! ```

// ============================================================================
// Streaming Median (for MAD calculation)
// ============================================================================

/// Default window size for streaming algorithms
pub const DEFAULT_WINDOW: usize = 100;

/// Streaming median using Sorted Rolling Window algorithm
///
/// Maintains a sorted array that is incrementally updated with O(N) memmove
/// operations instead of O(N log N) sorting on each access.
///
/// - push(): O(N) - binary search + memmove
/// - median(): O(1) - direct array access
#[derive(Clone, Debug)]
pub struct StreamingMedian {
    /// Circular buffer of recent values (insertion order)
    buffer: [f64; DEFAULT_WINDOW],
    /// Sorted array maintained incrementally
    sorted: [f64; DEFAULT_WINDOW],
    /// Current write position in circular buffer
    pos: usize,
    /// Number of values seen (saturates at window size)
    count: usize,
}

impl StreamingMedian {
    /// Window size
    pub const WINDOW: usize = DEFAULT_WINDOW;

    /// Create a new streaming median estimator
    pub fn new() -> Self {
        Self {
            buffer: [0.0; DEFAULT_WINDOW],
            sorted: [0.0; DEFAULT_WINDOW],
            pos: 0,
            count: 0,
        }
    }

    /// Binary search for insertion point in sorted array
    #[inline]
    fn binary_search_insert(&self, value: f64, len: usize) -> usize {
        let mut left = 0;
        let mut right = len;
        while left < right {
            let mid = left + (right - left) / 2;
            if self.sorted[mid] < value {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        left
    }

    /// Binary search for removal point (finds exact match)
    #[inline]
    fn binary_search_remove(&self, value: f64, len: usize) -> Option<usize> {
        let mut left = 0;
        let mut right = len;
        while left < right {
            let mid = left + (right - left) / 2;
            if self.sorted[mid] < value {
                left = mid + 1;
            } else if self.sorted[mid] > value {
                right = mid;
            } else {
                return Some(mid);
            }
        }
        // Handle floating point equality - search nearby
        if left < len && (self.sorted[left] - value).abs() < f64::EPSILON {
            return Some(left);
        }
        if left > 0 && (self.sorted[left - 1] - value).abs() < f64::EPSILON {
            return Some(left - 1);
        }
        None
    }

    /// Add a new value with O(N) sorted array maintenance
    #[inline]
    pub fn push(&mut self, value: f64) {
        if self.count < DEFAULT_WINDOW {
            // Buffer not full: just insert into sorted array
            let insert_pos = self.binary_search_insert(value, self.count);

            // Shift elements right to make room (memmove)
            if insert_pos < self.count {
                self.sorted.copy_within(insert_pos..self.count, insert_pos + 1);
            }
            self.sorted[insert_pos] = value;

            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % DEFAULT_WINDOW;
            self.count += 1;
        } else {
            // Buffer full: remove old value, insert new value
            let old_value = self.buffer[self.pos];

            // Remove old value from sorted array
            if let Some(remove_pos) = self.binary_search_remove(old_value, self.count) {
                // Shift elements left (memmove)
                if remove_pos < self.count - 1 {
                    self.sorted.copy_within(remove_pos + 1..self.count, remove_pos);
                }
            }

            // Insert new value into sorted array
            let insert_pos = self.binary_search_insert(value, self.count - 1);

            // Shift elements right to make room (memmove)
            if insert_pos < self.count - 1 {
                self.sorted.copy_within(insert_pos..self.count - 1, insert_pos + 1);
            }
            self.sorted[insert_pos] = value;

            // Update circular buffer
            self.buffer[self.pos] = value;
            self.pos = (self.pos + 1) % DEFAULT_WINDOW;
        }
    }

    /// Get the current median estimate - O(1)
    #[inline]
    pub fn median(&mut self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let mid = self.count / 2;
        if self.count % 2 == 0 {
            (self.sorted[mid - 1] + self.sorted[mid]) / 2.0
        } else {
            self.sorted[mid]
        }
    }

    /// Get count of values
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count >= DEFAULT_WINDOW
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer = [0.0; DEFAULT_WINDOW];
        self.sorted = [0.0; DEFAULT_WINDOW];
        self.pos = 0;
        self.count = 0;
    }
}

impl Default for StreamingMedian {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// MAD (Median Absolute Deviation) Detector
// ============================================================================

/// MAD-based anomaly detector
///
/// Uses Median Absolute Deviation for robust outlier detection.
/// MAD = median(|X_i - median(X)|)
///
/// An observation is anomalous if:
/// |x - median| > k * MAD * 1.4826
///
/// where 1.4826 is a scale factor for normal distributions,
/// and k is typically 3 (for 3-sigma equivalent).
///
/// # Example
/// ```
/// use alice_physics::anomaly::MadDetector;
///
/// let mut detector = MadDetector::new(3.0);
///
/// // Train on normal data
/// for v in [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 10.8] {
///     detector.observe(v);
/// }
///
/// // Check for anomalies
/// assert!(!detector.is_anomaly(10.0)); // Normal
/// assert!(detector.is_anomaly(100.0)); // Anomaly!
/// ```
#[derive(Clone, Debug)]
pub struct MadDetector {
    /// Streaming median for values
    values_median: StreamingMedian,
    /// Streaming median for absolute deviations
    deviations_median: StreamingMedian,
    /// Recent values for deviation calculation
    recent_values: [f64; DEFAULT_WINDOW],
    /// Position in recent values
    pos: usize,
    /// Count of observations
    count: usize,
    /// Threshold multiplier (k)
    threshold_k: f64,
    /// Cached median value
    cached_median: f64,
    /// Cached MAD value
    cached_mad: f64,
    /// Whether cache is valid
    cache_valid: bool,
}

impl MadDetector {
    /// Scale factor for normal distribution (1/Phi^-1(0.75))
    const MAD_SCALE: f64 = 1.4826;

    /// Create a new MAD detector
    ///
    /// # Arguments
    /// * `threshold_k` - Number of MAD units for anomaly threshold (typically 3.0)
    pub fn new(threshold_k: f64) -> Self {
        Self {
            values_median: StreamingMedian::new(),
            deviations_median: StreamingMedian::new(),
            recent_values: [0.0; DEFAULT_WINDOW],
            pos: 0,
            count: 0,
            threshold_k,
            cached_median: 0.0,
            cached_mad: 0.0,
            cache_valid: false,
        }
    }

    /// Observe a new value (updates statistics)
    pub fn observe(&mut self, value: f64) {
        // Store in recent values
        self.recent_values[self.pos] = value;
        self.pos = (self.pos + 1) % DEFAULT_WINDOW;
        if self.count < DEFAULT_WINDOW {
            self.count += 1;
        }

        // Update values median
        self.values_median.push(value);

        // Invalidate cache
        self.cache_valid = false;
    }

    /// Update cached statistics
    fn update_cache(&mut self) {
        if self.cache_valid {
            return;
        }

        // Get current median
        self.cached_median = self.values_median.median();

        // Calculate absolute deviations and their median
        self.deviations_median.clear();
        for i in 0..self.count {
            let dev = (self.recent_values[i] - self.cached_median).abs();
            self.deviations_median.push(dev);
        }
        self.cached_mad = self.deviations_median.median();

        self.cache_valid = true;
    }

    /// Check if a value is an anomaly
    pub fn is_anomaly(&mut self, value: f64) -> bool {
        if self.count < 3 {
            // Not enough data to determine anomalies
            return false;
        }

        self.update_cache();

        // Avoid division by zero
        if self.cached_mad < 1e-10 {
            // All values are the same, any different value is anomalous
            return (value - self.cached_median).abs() > 1e-10;
        }

        let deviation = (value - self.cached_median).abs();
        let threshold = self.threshold_k * self.cached_mad * Self::MAD_SCALE;

        deviation > threshold
    }

    /// Get the anomaly score (higher = more anomalous)
    ///
    /// Returns the number of MAD units from the median.
    pub fn anomaly_score(&mut self, value: f64) -> f64 {
        if self.count < 3 {
            return 0.0;
        }

        self.update_cache();

        if self.cached_mad < 1e-10 {
            return if (value - self.cached_median).abs() > 1e-10 {
                f64::INFINITY
            } else {
                0.0
            };
        }

        (value - self.cached_median).abs() / (self.cached_mad * Self::MAD_SCALE)
    }

    /// Get the current median
    pub fn median(&mut self) -> f64 {
        self.update_cache();
        self.cached_median
    }

    /// Get the current MAD
    pub fn mad(&mut self) -> f64 {
        self.update_cache();
        self.cached_mad
    }

    /// Get the threshold multiplier
    #[inline]
    pub fn threshold_k(&self) -> f64 {
        self.threshold_k
    }

    /// Set the threshold multiplier
    #[inline]
    pub fn set_threshold_k(&mut self, k: f64) {
        self.threshold_k = k;
    }

    /// Get count of observations
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Clear all observations
    pub fn clear(&mut self) {
        self.values_median.clear();
        self.deviations_median.clear();
        self.recent_values = [0.0; DEFAULT_WINDOW];
        self.pos = 0;
        self.count = 0;
        self.cache_valid = false;
    }
}

// ============================================================================
// EWMA (Exponentially Weighted Moving Average) Detector
// ============================================================================

/// EWMA-based anomaly detector
///
/// Uses exponentially weighted moving average and standard deviation
/// for change point detection and anomaly flagging.
///
/// # Example
/// ```
/// use alice_physics::anomaly::EwmaDetector;
///
/// let mut detector = EwmaDetector::new(0.1, 3.0);
///
/// // Feed normal values
/// for v in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1] {
///     detector.observe(v);
/// }
///
/// // Check for anomalies
/// assert!(!detector.is_anomaly(10.0));
/// assert!(detector.is_anomaly(50.0));
/// ```
#[derive(Clone, Debug)]
pub struct EwmaDetector {
    /// Smoothing factor (0 < alpha <= 1)
    alpha: f64,
    /// Current EWMA
    ewma: f64,
    /// EWMA of squared deviations (for variance)
    ewma_var: f64,
    /// Threshold multiplier (number of std devs)
    threshold_k: f64,
    /// Whether initialized
    initialized: bool,
    /// Count of observations
    count: u64,
}

impl EwmaDetector {
    /// Create a new EWMA detector
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0.0-1.0, higher = more reactive)
    /// * `threshold_k` - Number of standard deviations for anomaly threshold
    pub fn new(alpha: f64, threshold_k: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 1.0),
            ewma: 0.0,
            ewma_var: 0.0,
            threshold_k,
            initialized: false,
            count: 0,
        }
    }

    /// Observe a new value and update statistics
    pub fn observe(&mut self, value: f64) {
        self.count += 1;

        if !self.initialized {
            self.ewma = value;
            self.ewma_var = 0.0;
            self.initialized = true;
            return;
        }

        // Update EWMA
        let deviation = value - self.ewma;
        self.ewma += self.alpha * deviation;

        // Update variance EWMA
        self.ewma_var = (1.0 - self.alpha) * (self.ewma_var + self.alpha * deviation * deviation);
    }

    /// Check if a value is an anomaly (without updating)
    pub fn is_anomaly(&self, value: f64) -> bool {
        if !self.initialized || self.count < 3 {
            return false;
        }

        let std_dev = self.ewma_var.sqrt();
        if std_dev < 1e-10 {
            return (value - self.ewma).abs() > 1e-10;
        }

        let deviation = (value - self.ewma).abs();
        deviation > self.threshold_k * std_dev
    }

    /// Get anomaly score (number of standard deviations)
    pub fn anomaly_score(&self, value: f64) -> f64 {
        if !self.initialized {
            return 0.0;
        }

        let std_dev = self.ewma_var.sqrt();
        if std_dev < 1e-10 {
            return if (value - self.ewma).abs() > 1e-10 {
                f64::INFINITY
            } else {
                0.0
            };
        }

        (value - self.ewma).abs() / std_dev
    }

    /// Get current EWMA
    #[inline]
    pub fn ewma(&self) -> f64 {
        self.ewma
    }

    /// Get current standard deviation estimate
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.ewma_var.sqrt()
    }

    /// Get smoothing factor
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Set smoothing factor
    #[inline]
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.001, 1.0);
    }

    /// Get threshold multiplier
    #[inline]
    pub fn threshold_k(&self) -> f64 {
        self.threshold_k
    }

    /// Set threshold multiplier
    #[inline]
    pub fn set_threshold_k(&mut self, k: f64) {
        self.threshold_k = k;
    }

    /// Get count of observations
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.ewma = 0.0;
        self.ewma_var = 0.0;
        self.initialized = false;
        self.count = 0;
    }
}

// ============================================================================
// Z-Score Detector (Simple threshold-based)
// ============================================================================

/// Simple Z-score based anomaly detector
///
/// Maintains running mean and variance, flags values that exceed
/// a threshold number of standard deviations.
#[derive(Clone, Debug)]
pub struct ZScoreDetector {
    /// Running mean
    mean: f64,
    /// Running variance (using Welford's algorithm)
    m2: f64,
    /// Count of observations
    count: u64,
    /// Threshold (number of standard deviations)
    threshold_k: f64,
}

impl ZScoreDetector {
    /// Create a new Z-score detector
    pub fn new(threshold_k: f64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            threshold_k,
        }
    }

    /// Observe a new value (updates running statistics)
    pub fn observe(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the variance
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the standard deviation
    #[inline]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Check if a value is an anomaly
    pub fn is_anomaly(&self, value: f64) -> bool {
        if self.count < 3 {
            return false;
        }

        let std_dev = self.std_dev();
        if std_dev < 1e-10 {
            return (value - self.mean).abs() > 1e-10;
        }

        let z_score = (value - self.mean).abs() / std_dev;
        z_score > self.threshold_k
    }

    /// Get the Z-score for a value
    pub fn z_score(&self, value: f64) -> f64 {
        let std_dev = self.std_dev();
        if std_dev < 1e-10 {
            return if (value - self.mean).abs() > 1e-10 {
                f64::INFINITY
            } else {
                0.0
            };
        }
        (value - self.mean) / std_dev
    }

    /// Get current mean
    #[inline]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get count
    #[inline]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.m2 = 0.0;
        self.count = 0;
    }
}

// ============================================================================
// Anomaly Event
// ============================================================================

/// Represents a detected anomaly
#[derive(Clone, Debug)]
pub struct AnomalyEvent {
    /// The anomalous value
    pub value: f64,
    /// Anomaly score (e.g., number of MAD/std devs)
    pub score: f64,
    /// Expected value (median or mean)
    pub expected: f64,
    /// Timestamp (if applicable)
    pub timestamp: u64,
    /// Metric identifier
    pub metric_id: u64,
}

/// Callback for anomaly detection
pub trait AnomalyCallback {
    /// Called when an anomaly is detected
    fn on_anomaly(&mut self, event: AnomalyEvent);
}

// ============================================================================
// Composite Anomaly Detector
// ============================================================================

/// Composite detector that combines multiple detection methods
///
/// An observation is flagged as anomalous if ANY detector flags it.
#[derive(Clone, Debug)]
pub struct CompositeDetector {
    /// MAD-based detector
    pub mad: MadDetector,
    /// EWMA-based detector
    pub ewma: EwmaDetector,
    /// Z-score detector
    pub zscore: ZScoreDetector,
    /// Whether to require all detectors to agree
    pub require_consensus: bool,
}

impl CompositeDetector {
    /// Create a new composite detector with default settings
    pub fn new() -> Self {
        Self {
            mad: MadDetector::new(3.0),
            ewma: EwmaDetector::new(0.1, 3.0),
            zscore: ZScoreDetector::new(3.0),
            require_consensus: false,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(mad_k: f64, ewma_alpha: f64, ewma_k: f64, zscore_k: f64) -> Self {
        Self {
            mad: MadDetector::new(mad_k),
            ewma: EwmaDetector::new(ewma_alpha, ewma_k),
            zscore: ZScoreDetector::new(zscore_k),
            require_consensus: false,
        }
    }

    /// Observe a new value (updates all detectors)
    pub fn observe(&mut self, value: f64) {
        self.mad.observe(value);
        self.ewma.observe(value);
        self.zscore.observe(value);
    }

    /// Check if a value is an anomaly
    pub fn is_anomaly(&mut self, value: f64) -> bool {
        let mad_anomaly = self.mad.is_anomaly(value);
        let ewma_anomaly = self.ewma.is_anomaly(value);
        let zscore_anomaly = self.zscore.is_anomaly(value);

        if self.require_consensus {
            // All must agree
            mad_anomaly && ewma_anomaly && zscore_anomaly
        } else {
            // Any detector flags it
            mad_anomaly || ewma_anomaly || zscore_anomaly
        }
    }

    /// Get combined anomaly score (max of all scores)
    pub fn anomaly_score(&mut self, value: f64) -> f64 {
        let mad_score = self.mad.anomaly_score(value);
        let ewma_score = self.ewma.anomaly_score(value);
        let zscore_score = self.zscore.z_score(value).abs();

        mad_score.max(ewma_score).max(zscore_score)
    }

    /// Get count of observations
    #[inline]
    pub fn count(&self) -> u64 {
        self.zscore.count()
    }

    /// Reset all detectors
    pub fn reset(&mut self) {
        self.mad.clear();
        self.ewma.reset();
        self.zscore.reset();
    }
}

impl Default for CompositeDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_median() {
        let mut sm = StreamingMedian::new();

        for v in [5.0, 2.0, 8.0, 1.0, 9.0] {
            sm.push(v);
        }

        let median = sm.median();
        assert_eq!(median, 5.0); // Sorted: 1, 2, 5, 8, 9 -> median = 5
    }

    #[test]
    fn test_streaming_median_even() {
        let mut sm = StreamingMedian::new();

        for v in [1.0, 2.0, 3.0, 4.0] {
            sm.push(v);
        }

        let median = sm.median();
        assert_eq!(median, 2.5); // Sorted: 1, 2, 3, 4 -> median = (2+3)/2 = 2.5
    }

    #[test]
    fn test_mad_detector_normal() {
        let mut detector = MadDetector::new(3.0);

        // Add normal data centered around 10
        for v in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.4] {
            detector.observe(v);
        }

        // Normal values should not be anomalies
        assert!(!detector.is_anomaly(10.0));
        assert!(!detector.is_anomaly(10.5));
        assert!(!detector.is_anomaly(9.5));

        // Extreme values should be anomalies
        assert!(detector.is_anomaly(100.0));
        assert!(detector.is_anomaly(-50.0));
    }

    #[test]
    fn test_ewma_detector() {
        let mut detector = EwmaDetector::new(0.1, 3.0);

        // Train on values with some variance
        for i in 0..100 {
            let value = 10.0 + (i % 3) as f64 - 1.0; // 9, 10, 11
            detector.observe(value);
        }

        // Should be close to 10
        assert!((detector.ewma() - 10.0).abs() < 1.0);

        // Normal values (within trained range)
        assert!(!detector.is_anomaly(10.0));
        assert!(!detector.is_anomaly(11.0));

        // Anomaly (far outside trained range)
        assert!(detector.is_anomaly(50.0));
    }

    #[test]
    fn test_zscore_detector() {
        let mut detector = ZScoreDetector::new(3.0);

        // Add normal data
        for v in [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3] {
            detector.observe(v);
        }

        assert!((detector.mean() - 10.03).abs() < 0.01);

        // Normal values
        assert!(!detector.is_anomaly(10.0));

        // Anomaly
        assert!(detector.is_anomaly(100.0));
    }

    #[test]
    fn test_composite_detector() {
        let mut detector = CompositeDetector::new();

        // Train on normal data
        for v in [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.4] {
            detector.observe(v);
        }

        // Normal
        assert!(!detector.is_anomaly(10.0));

        // Anomaly
        assert!(detector.is_anomaly(100.0));

        let score = detector.anomaly_score(100.0);
        assert!(score > 3.0);
    }

    #[test]
    fn test_mad_detector_constant() {
        let mut detector = MadDetector::new(3.0);

        // All same values
        for _ in 0..10 {
            detector.observe(5.0);
        }

        // Same value is not anomaly
        assert!(!detector.is_anomaly(5.0));

        // Different value is anomaly (MAD = 0)
        assert!(detector.is_anomaly(5.1));
    }
}
