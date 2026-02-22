//! Physics Profiling API
//!
//! Lightweight performance counters and timers for physics pipeline stages.
//! Uses deterministic "tick" counting rather than wall-clock time for reproducibility.
//!
//! # Profiled Stages
//!
//! - Broadphase (AABB overlap tests)
//! - Narrowphase (GJK/EPA collision detection)
//! - Solver (constraint solving iterations)
//! - CCD (continuous collision detection)
//! - Integration (velocity/position updates)

#[cfg(not(feature = "std"))]
use alloc::string::String;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A single profiling timer entry
#[derive(Clone, Debug)]
pub struct ProfileEntry {
    /// Stage name
    pub name: &'static str,
    /// Total accumulated ticks
    pub total_ticks: u64,
    /// Number of invocations
    pub call_count: u64,
    /// Last frame's ticks
    pub last_ticks: u64,
    /// Peak ticks (single frame)
    pub peak_ticks: u64,
}

impl ProfileEntry {
    /// Create a new profile entry
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            total_ticks: 0,
            call_count: 0,
            last_ticks: 0,
            peak_ticks: 0,
        }
    }

    /// Average ticks per call
    #[inline]
    pub fn average_ticks(&self) -> u64 {
        if self.call_count == 0 {
            0
        } else {
            self.total_ticks / self.call_count
        }
    }

    /// Record a measurement
    pub fn record(&mut self, ticks: u64) {
        self.total_ticks += ticks;
        self.call_count += 1;
        self.last_ticks = ticks;
        if ticks > self.peak_ticks {
            self.peak_ticks = ticks;
        }
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        self.total_ticks = 0;
        self.call_count = 0;
        self.last_ticks = 0;
        self.peak_ticks = 0;
    }
}

/// Physics step statistics (per-frame counters)
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    /// Number of broadphase pairs tested
    pub broadphase_pairs: u32,
    /// Number of narrowphase tests performed
    pub narrowphase_tests: u32,
    /// Number of active contacts
    pub active_contacts: u32,
    /// Number of contact manifolds
    pub active_manifolds: u32,
    /// Number of active bodies (non-sleeping)
    pub active_bodies: u32,
    /// Number of static bodies
    pub static_bodies: u32,
    /// Number of solver iterations performed
    pub solver_iterations: u32,
    /// Number of CCD checks
    pub ccd_checks: u32,
    /// Number of joint constraints
    pub joint_count: u32,
}

/// Physics profiler: collects timing and statistics
pub struct PhysicsProfiler {
    /// Stage timers
    entries: Vec<ProfileEntry>,
    /// Current frame stats
    pub stats: StepStats,
    /// Frame counter
    pub frame_count: u64,
    /// Whether profiling is enabled
    pub enabled: bool,
}

/// Stage indices for fast lookup
pub const STAGE_BROADPHASE: usize = 0;
/// Narrowphase stage index
pub const STAGE_NARROWPHASE: usize = 1;
/// Solver stage index
pub const STAGE_SOLVER: usize = 2;
/// CCD stage index
pub const STAGE_CCD: usize = 3;
/// Integration stage index
pub const STAGE_INTEGRATION: usize = 4;
/// Contact cache stage index
pub const STAGE_CONTACT_CACHE: usize = 5;
/// Total step stage index
pub const STAGE_TOTAL_STEP: usize = 6;

impl PhysicsProfiler {
    /// Create a new profiler with default stages
    pub fn new() -> Self {
        let entries = vec![
            ProfileEntry::new("broadphase"),
            ProfileEntry::new("narrowphase"),
            ProfileEntry::new("solver"),
            ProfileEntry::new("ccd"),
            ProfileEntry::new("integration"),
            ProfileEntry::new("contact_cache"),
            ProfileEntry::new("total_step"),
        ];

        Self {
            entries,
            stats: StepStats::default(),
            frame_count: 0,
            enabled: true,
        }
    }

    /// Record ticks for a stage
    #[inline]
    pub fn record(&mut self, stage: usize, ticks: u64) {
        if self.enabled && stage < self.entries.len() {
            self.entries[stage].record(ticks);
        }
    }

    /// Begin a new frame (reset per-frame stats)
    pub fn begin_frame(&mut self) {
        self.stats = StepStats::default();
        self.frame_count += 1;
    }

    /// Get a profile entry by stage index
    pub fn get(&self, stage: usize) -> Option<&ProfileEntry> {
        self.entries.get(stage)
    }

    /// Get last frame's ticks for a stage
    pub fn last_ticks(&self, stage: usize) -> u64 {
        self.entries.get(stage).map_or(0, |e| e.last_ticks)
    }

    /// Get average ticks for a stage
    pub fn average_ticks(&self, stage: usize) -> u64 {
        self.entries
            .get(stage)
            .map_or(0, ProfileEntry::average_ticks)
    }

    /// Reset all profiling data
    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.reset();
        }
        self.stats = StepStats::default();
        self.frame_count = 0;
    }

    /// Get a summary of all stages
    pub fn summary(&self) -> Vec<(&'static str, u64, u64, u64)> {
        self.entries
            .iter()
            .map(|e| (e.name, e.last_ticks, e.average_ticks(), e.peak_ticks))
            .collect()
    }
}

impl Default for PhysicsProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Deterministic tick counter (monotonic, platform-independent)
///
/// Uses a simple incrementing counter. For real-time profiling,
/// wrap with platform-specific high-resolution timer.
pub struct TickCounter {
    current: u64,
}

impl TickCounter {
    /// Create a new tick counter
    pub fn new() -> Self {
        Self { current: 0 }
    }

    /// Start timing
    #[inline]
    pub fn start(&self) -> u64 {
        self.current
    }

    /// Stop timing and return elapsed ticks
    #[inline]
    pub fn elapsed(&self, start: u64) -> u64 {
        self.current.wrapping_sub(start)
    }

    /// Advance the counter by N ticks
    #[inline]
    pub fn advance(&mut self, ticks: u64) {
        self.current = self.current.wrapping_add(ticks);
    }

    /// Current tick value
    #[inline]
    pub fn now(&self) -> u64 {
        self.current
    }
}

impl Default for TickCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_entry() {
        let mut entry = ProfileEntry::new("test");
        entry.record(100);
        entry.record(200);
        entry.record(150);

        assert_eq!(entry.call_count, 3);
        assert_eq!(entry.total_ticks, 450);
        assert_eq!(entry.average_ticks(), 150);
        assert_eq!(entry.peak_ticks, 200);
        assert_eq!(entry.last_ticks, 150);
    }

    #[test]
    fn test_profiler() {
        let mut profiler = PhysicsProfiler::new();
        profiler.begin_frame();

        profiler.record(STAGE_BROADPHASE, 100);
        profiler.record(STAGE_NARROWPHASE, 200);
        profiler.record(STAGE_SOLVER, 500);

        assert_eq!(profiler.last_ticks(STAGE_BROADPHASE), 100);
        assert_eq!(profiler.last_ticks(STAGE_SOLVER), 500);
    }

    #[test]
    fn test_profiler_summary() {
        let mut profiler = PhysicsProfiler::new();
        profiler.record(STAGE_BROADPHASE, 42);

        let summary = profiler.summary();
        assert!(summary.len() >= 7);
        assert_eq!(summary[STAGE_BROADPHASE].1, 42); // last_ticks
    }

    #[test]
    fn test_tick_counter() {
        let mut counter = TickCounter::new();
        let start = counter.start();
        counter.advance(100);
        let elapsed = counter.elapsed(start);
        assert_eq!(elapsed, 100);
    }

    #[test]
    fn test_step_stats() {
        let stats = StepStats::default();
        assert_eq!(stats.broadphase_pairs, 0);
        assert_eq!(stats.active_contacts, 0);
    }

    #[test]
    fn test_profiler_reset() {
        let mut profiler = PhysicsProfiler::new();
        profiler.record(STAGE_BROADPHASE, 100);
        profiler.reset();
        assert_eq!(profiler.last_ticks(STAGE_BROADPHASE), 0);
        assert_eq!(profiler.frame_count, 0);
    }
}
