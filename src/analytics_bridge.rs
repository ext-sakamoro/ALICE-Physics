//! ALICE-Analytics bridge: Physics simulation metrics
//!
//! Feeds per-step simulation metrics (step time, collision count,
//! energy drift) into ALICE-Analytics sketches for profiling and
//! anomaly detection.

use alice_analytics::sketch::{DDSketch256, HyperLogLog12};

/// Physics telemetry collector backed by ALICE-Analytics sketches.
pub struct PhysicsTelemetry {
    /// Step execution time in microseconds (quantile estimation).
    step_time: DDSketch256,
    /// Collision / contact count per step.
    contacts: DDSketch256,
    /// Energy drift per step (|E_n - E_{n-1}|).
    energy_drift: DDSketch256,
    /// Unique body pair collisions (cardinality estimation).
    collision_pairs: HyperLogLog12,
    /// Total simulation steps.
    total_steps: u64,
}

impl PhysicsTelemetry {
    /// DDSketch relative accuracy (`alice-analytics` recommends `α ≥ 0.05`
    /// for the 256-bin sketch so the bin range covers realistic magnitudes).
    pub const ALPHA: f64 = 0.05;

    /// Create a new physics telemetry collector.
    ///
    /// The sketches use `α = 0.05` (5 % relative error). `DDSketch256` has
    /// 256 log-spaced bins with an offset of 64, so its representable range
    /// is `[γ⁻⁶⁴, γ¹⁹¹]` with `γ = (1 + α)/(1 − α)`: for `α = 0.05` that is
    /// `≈ [1.6e-3, 2.1e8]`, which covers step times in µs, contact counts
    /// and energy drift. Before 1.2.0 `α = 0.01` was used (`γ = 1.0202`,
    /// range `≈ [0.28, 45.6]`): every step time above 45 µs fell outside the
    /// histogram, so `p50` and `p99` both degraded to the running maximum
    /// (990 × 1000 µs + 10 × 50 000 µs reported p50 = 50 000).
    pub fn new() -> Self {
        Self {
            step_time: DDSketch256::new(Self::ALPHA),
            contacts: DDSketch256::new(Self::ALPHA),
            energy_drift: DDSketch256::new(Self::ALPHA),
            collision_pairs: HyperLogLog12::new(),
            total_steps: 0,
        }
    }

    /// Record a simulation step's timing.
    pub fn record_step_time(&mut self, time_us: f64) {
        self.step_time.insert(time_us);
        self.total_steps += 1;
    }

    /// Record contact count for a step.
    pub fn record_contacts(&mut self, count: f64) {
        self.contacts.insert(count);
    }

    /// Record energy drift between consecutive steps.
    pub fn record_energy_drift(&mut self, drift: f64) {
        self.energy_drift.insert(drift.abs());
    }

    /// Record a collision pair observation (for unique pair counting).
    ///
    /// `pair_hash` should be a deterministic hash of the two body IDs
    /// (e.g. `min(a,b) << 32 | max(a,b)`).
    pub fn record_collision_pair(&mut self, pair_hash: u64) {
        self.collision_pairs.insert_hash(pair_hash);
    }

    /// Estimated p50 step time.
    pub fn step_time_p50(&self) -> f64 {
        self.step_time.quantile(0.5)
    }

    /// Estimated p99 step time.
    pub fn step_time_p99(&self) -> f64 {
        self.step_time.quantile(0.99)
    }

    /// Estimated p50 contact count.
    pub fn contacts_p50(&self) -> f64 {
        self.contacts.quantile(0.5)
    }

    /// Estimated p99 contact count.
    pub fn contacts_p99(&self) -> f64 {
        self.contacts.quantile(0.99)
    }

    /// Estimated p99 energy drift (should stay near zero for stable sims).
    pub fn energy_drift_p99(&self) -> f64 {
        self.energy_drift.quantile(0.99)
    }

    /// Estimated unique collision pairs observed.
    pub fn unique_collision_pairs(&self) -> f64 {
        self.collision_pairs.cardinality()
    }

    /// Total simulation steps recorded.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }
}

impl Default for PhysicsTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_telemetry() {
        let mut tel = PhysicsTelemetry::new();

        for step in 0u64..200 {
            tel.record_step_time(100.0 + (step % 20) as f64);
            tel.record_contacts((step % 10) as f64);
            tel.record_energy_drift(0.001 * step as f64);

            let pair = (step % 5) << 32 | ((step + 1) % 5);
            tel.record_collision_pair(pair);
        }

        assert_eq!(tel.total_steps(), 200);
        assert!(tel.step_time_p50() > 0.0);
        assert!(tel.contacts_p50() >= 0.0);
        assert!(tel.energy_drift_p99() >= 0.0);
        assert!(tel.unique_collision_pairs() >= 1.0);
    }

    /// p99 estimators: for a known distribution the 99th percentile is bracketed
    /// by the sample values (DDSketch relative error), and it is monotone —
    /// feeding a larger tail raises it, feeding only small values keeps it low.
    #[test]
    fn step_time_and_contacts_p99_track_the_upper_tail() {
        let mut tel = PhysicsTelemetry::new();
        // 990 steps at 1.0 ms and 10 at 50 ms: p99 sits at the boundary of the
        // 50 ms tail (between 1 and 50, and >= the p50 of 1 ms)
        for _ in 0..990 {
            tel.record_step_time(1000.0);
            tel.record_contacts(10.0);
        }
        for _ in 0..10 {
            tel.record_step_time(50_000.0);
            tel.record_contacts(400.0);
        }
        let st99 = tel.step_time_p99();
        let ct99 = tel.contacts_p99();
        assert!(
            st99 >= tel.step_time_p50(),
            "p99 {st99} below p50 {}",
            tel.step_time_p50()
        );
        assert!(
            (1000.0 * 0.95..=50_000.0 * 1.05).contains(&st99), // DDSketch relative error
            "step p99 {st99} outside [1 ms, 50 ms]"
        );
        assert!(
            ct99 >= tel.contacts_p50() * 0.95,
            "contacts p99 {ct99} below p50"
        );
        assert!(
            (10.0 * 0.95..=400.0 * 1.05).contains(&ct99),
            "contacts p99 {ct99} outside [10, 400]"
        );
        // a uniform stream: p99 ≈ the (only) value within sketch error
        let mut flat = PhysicsTelemetry::new();
        for _ in 0..500 {
            flat.record_step_time(2000.0);
            flat.record_contacts(3.0);
        }
        // DDSketch returns the bucket's lower bound: within [v/γ, v]
        assert_bucket_of(flat.step_time_p99(), 2000.0, "flat step p99");
        assert!(
            flat.contacts_p99() <= 3.0 && flat.contacts_p99() >= 3.0 / GAMMA,
            "{}",
            flat.contacts_p99()
        );
        // ordering: a tail that occupies rank ≥ 99 % (15 of 1000 samples)
        // raises p99 to the tail value; 10 of 1000 (above) leaves it at the
        // base value — rank arithmetic, not the sketch's max fallback
        let mut heavy = PhysicsTelemetry::new();
        for _ in 0..985 {
            heavy.record_step_time(1000.0);
            heavy.record_contacts(10.0);
        }
        for _ in 0..15 {
            heavy.record_step_time(50_000.0);
            heavy.record_contacts(400.0);
        }
        assert_bucket_of(heavy.step_time_p99(), 50_000.0, "heavy step p99");
        assert_bucket_of(heavy.contacts_p99(), 400.0, "heavy contacts p99");
        assert!(heavy.step_time_p99() > tel.step_time_p99());
        assert!(heavy.contacts_p99() > tel.contacts_p99());
    }

    /// `DDSketch256::new(α)` has `γ = (1 + α) / (1 − α)` and returns the lower
    /// bound `γ^(k−1)` of the bucket `(γ^(k−1), γ^k]` holding the value of
    /// rank `⌈q·n⌉`, so for a sample `v` of that rank the estimate lies in
    /// `[v/γ, v]` (`≈ [0.905 v, v]` for `α = 0.05`). With 20 samples, p50 is
    /// rank 10 and p99 is rank 20 (the maximum).
    const GAMMA: f64 = (1.0 + PhysicsTelemetry::ALPHA) / (1.0 - PhysicsTelemetry::ALPHA);

    fn assert_bucket_of(actual: f64, v: f64, what: &str) {
        assert!(
            actual <= v && actual >= v / GAMMA,
            "{what} = {actual}, expected in [{}, {v}]",
            v / GAMMA
        );
    }

    /// Realistic step times in µs (the documented unit): 990 × 1000 µs plus
    /// 10 × 50 000 µs. p50 is rank 500 → 1000 µs, p99 is rank 990 → 1000 µs
    /// (only the last 10 samples are the 50 ms outliers). With the pre-1.2.0
    /// `α = 0.01` sketch every sample was above the histogram range and both
    /// quantiles reported 50 000.
    #[test]
    fn quantiles_stay_in_range_for_microsecond_step_times() {
        let mut tel = PhysicsTelemetry::new();
        for _ in 0..990 {
            tel.record_step_time(1000.0);
        }
        for _ in 0..10 {
            tel.record_step_time(50_000.0);
        }
        assert_bucket_of(tel.step_time_p50(), 1000.0, "p50");
        assert_bucket_of(tel.step_time_p99(), 1000.0, "p99");
        // the tail is still visible one rank higher: rank 991+ is 50 000
        let mut tail = PhysicsTelemetry::new();
        for _ in 0..90 {
            tail.record_step_time(1000.0);
        }
        for _ in 0..10 {
            tail.record_step_time(50_000.0);
        }
        assert_bucket_of(tail.step_time_p99(), 50_000.0, "p99 with 10 % outliers");
    }

    /// Known sample sets, exact rank arithmetic:
    ///
    /// ```text
    /// step_time    10 × 10 µs, 10 × 40 µs → p50 = rank 10 → 10,  p99 = rank 20 → 40
    /// contacts     15 × 3,     5 × 30     → p50 = rank 10 → 3,   p99 = rank 20 → 30
    /// energy_drift 19 × (−0.5), 1 × (−20) → p99 = rank 20 → |−20| = 20 (abs)
    /// pairs        40 distinct hashes, each recorded twice → 40 (HLL linear
    ///              counting 4096·ln(4096/4056) = 40.20; 80 if not deduplicated)
    /// total_steps  20 (one per record_step_time, not per other record_*)
    /// ```
    ///
    /// All sample values sit inside the sketch's bucketed range
    /// `[γ^−64, γ^191] ≈ [0.278, 45.6]` — see the bug note in the report:
    /// values above that are dropped from the histogram and `quantile`
    /// degrades to `max`.
    #[test]
    fn quantiles_and_cardinality_match_known_sample_sets() {
        let mut tel = PhysicsTelemetry::new();
        for _ in 0..10 {
            tel.record_step_time(10.0);
        }
        for _ in 0..10 {
            tel.record_step_time(40.0);
        }
        for _ in 0..15 {
            tel.record_contacts(3.0);
        }
        for _ in 0..5 {
            tel.record_contacts(30.0);
        }
        for _ in 0..19 {
            tel.record_energy_drift(-0.5);
        }
        tel.record_energy_drift(-20.0);
        for i in 0..40u64 {
            let pair = (i << 32) | (i + 1); // low 12 bits distinct → 40 registers
            tel.record_collision_pair(pair);
            tel.record_collision_pair(pair);
        }

        assert_eq!(tel.total_steps(), 20);
        assert_bucket_of(tel.step_time_p50(), 10.0, "step_time_p50");
        assert_bucket_of(tel.step_time_p99(), 40.0, "step_time_p99");
        assert_bucket_of(tel.contacts_p50(), 3.0, "contacts_p50");
        assert_bucket_of(tel.contacts_p99(), 30.0, "contacts_p99");
        assert_bucket_of(tel.energy_drift_p99(), 20.0, "energy_drift_p99");
        let pairs = tel.unique_collision_pairs();
        assert!(
            (39.5..=41.0).contains(&pairs),
            "unique_collision_pairs = {pairs}, expected ≈ 40.20"
        );
    }

    /// Each `record_*` method touches exactly one metric: after a single
    /// call the other sketches are still empty (`quantile` of an empty
    /// DDSketch is 0, cardinality of an empty HLL is `4096·ln(4096/4096) =
    /// 0`) and `total_steps` only advances on `record_step_time`.
    #[test]
    fn each_record_method_changes_only_its_own_metric() {
        // record_step_time → step_time sketch + total_steps
        let mut t = PhysicsTelemetry::new();
        t.record_step_time(10.0);
        assert_eq!(t.total_steps(), 1);
        assert_bucket_of(t.step_time_p50(), 10.0, "step_time_p50");
        assert_bucket_of(t.step_time_p99(), 10.0, "step_time_p99");
        assert_eq!(t.contacts_p50(), 0.0);
        assert_eq!(t.contacts_p99(), 0.0);
        assert_eq!(t.energy_drift_p99(), 0.0);
        assert_eq!(t.unique_collision_pairs(), 0.0);

        // record_contacts → contacts sketch only
        let mut t = PhysicsTelemetry::new();
        t.record_contacts(3.0);
        assert_eq!(t.total_steps(), 0);
        assert_eq!(t.step_time_p50(), 0.0);
        assert_eq!(t.step_time_p99(), 0.0);
        assert_bucket_of(t.contacts_p50(), 3.0, "contacts_p50");
        assert_bucket_of(t.contacts_p99(), 3.0, "contacts_p99");
        assert_eq!(t.energy_drift_p99(), 0.0);
        assert_eq!(t.unique_collision_pairs(), 0.0);

        // record_energy_drift → energy sketch only, magnitude of the drift
        let mut t = PhysicsTelemetry::new();
        t.record_energy_drift(-0.5);
        assert_eq!(t.total_steps(), 0);
        assert_eq!(t.step_time_p50(), 0.0);
        assert_eq!(t.contacts_p50(), 0.0);
        assert_bucket_of(t.energy_drift_p99(), 0.5, "energy_drift_p99 (|−0.5|)");
        assert_eq!(t.unique_collision_pairs(), 0.0);

        // record_collision_pair → HLL only; one register → 4096·ln(4096/4095)
        let mut t = PhysicsTelemetry::new();
        t.record_collision_pair((1u64 << 32) | 2);
        assert_eq!(t.total_steps(), 0);
        assert_eq!(t.step_time_p50(), 0.0);
        assert_eq!(t.contacts_p50(), 0.0);
        assert_eq!(t.energy_drift_p99(), 0.0);
        let one = t.unique_collision_pairs();
        assert!(
            (one - 1.000_122).abs() < 1.0e-3,
            "unique_collision_pairs = {one}, expected 4096·ln(4096/4095) = 1.000122"
        );
    }

    /// `total_steps` is a plain counter: 7 then 5 more steps give 12, never
    /// 0 (`*=`) or a wrap (`-=`), regardless of the other record calls.
    #[test]
    fn total_steps_counts_each_step_once() {
        let mut t = PhysicsTelemetry::default();
        for _ in 0..7 {
            t.record_step_time(2.0);
        }
        assert_eq!(t.total_steps(), 7);
        t.record_contacts(4.0);
        t.record_energy_drift(1.5);
        t.record_collision_pair(99);
        assert_eq!(t.total_steps(), 7);
        for _ in 0..5 {
            t.record_step_time(3.0);
        }
        assert_eq!(t.total_steps(), 12);
    }
}
