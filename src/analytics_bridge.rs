//! ALICE-Analytics bridge: Physics simulation metrics
//!
//! Feeds per-step simulation metrics (step time, collision count,
//! energy drift) into ALICE-Analytics sketches for profiling and
//! anomaly detection.

use alice_analytics::prelude::*;

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
    /// Create a new physics telemetry collector.
    pub fn new() -> Self {
        Self {
            step_time: DDSketch256::new(0.01),
            contacts: DDSketch256::new(0.01),
            energy_drift: DDSketch256::new(0.01),
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
}
