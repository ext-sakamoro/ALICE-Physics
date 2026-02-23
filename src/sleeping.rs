//! Sleeping and Island Management
//!
//! Puts low-energy bodies to sleep and groups connected bodies into islands.
//!
//! # Sleeping
//!
//! Bodies whose velocities remain below a threshold for a sustained period
//! are put to sleep, skipping integration and constraint solving.
//!
//! # Islands
//!
//! Connected components of bodies (via joints/contacts) are grouped into islands.
//! If all bodies in an island are sleeping, the entire island sleeps.

use crate::math::Fix128;
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Sleep state for a body
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SleepState {
    /// Body is fully active
    Awake,
    /// Body is sleeping (skipped in simulation)
    Sleeping,
}

/// Per-body sleep tracking data
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepData {
    /// Current sleep state
    pub state: SleepState,
    /// Number of consecutive frames below velocity threshold
    pub idle_frames: u32,
}

impl SleepData {
    /// Create new awake sleep data
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: SleepState::Awake,
            idle_frames: 0,
        }
    }

    /// Check if this body is currently sleeping
    #[inline]
    #[must_use]
    pub fn is_sleeping(&self) -> bool {
        self.state == SleepState::Sleeping
    }

    /// Wake up this body
    #[inline]
    pub fn wake(&mut self) {
        self.state = SleepState::Awake;
        self.idle_frames = 0;
    }
}

impl Default for SleepData {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the sleeping system
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SleepConfig {
    /// Linear velocity threshold below which a body is considered idle
    pub linear_threshold: Fix128,
    /// Angular velocity threshold below which a body is considered idle
    pub angular_threshold: Fix128,
    /// Number of consecutive idle frames before putting to sleep
    pub frames_to_sleep: u32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            linear_threshold: Fix128::from_ratio(1, 100), // 0.01 m/s
            angular_threshold: Fix128::from_ratio(1, 100), // 0.01 rad/s
            frames_to_sleep: 60,                          // 1 second at 60fps
        }
    }
}

/// Island: a group of connected bodies
#[derive(Clone, Debug)]
pub struct Island {
    /// Body indices in this island
    pub bodies: Vec<usize>,
    /// Whether all bodies in the island are sleeping
    pub all_sleeping: bool,
}

/// Island manager using Union-Find (disjoint set)
pub struct IslandManager {
    /// Parent array for union-find
    parent: Vec<usize>,
    /// Rank for union by rank
    rank: Vec<u32>,
    /// Sleep data per body
    pub sleep_data: Vec<SleepData>,
    /// Sleep configuration
    pub config: SleepConfig,
}

impl IslandManager {
    /// Create a new island manager for `num_bodies` bodies
    #[must_use]
    pub fn new(num_bodies: usize, config: SleepConfig) -> Self {
        let mut parent = Vec::with_capacity(num_bodies);
        let rank = vec![0u32; num_bodies];
        let sleep_data = vec![SleepData::new(); num_bodies];
        for i in 0..num_bodies {
            parent.push(i);
        }
        Self {
            parent,
            rank,
            sleep_data,
            config,
        }
    }

    /// Resize to accommodate more bodies
    pub fn resize(&mut self, num_bodies: usize) {
        while self.parent.len() < num_bodies {
            let idx = self.parent.len();
            self.parent.push(idx);
            self.rank.push(0);
            self.sleep_data.push(SleepData::new());
        }
    }

    /// Reset all unions (call before rebuilding islands)
    pub fn reset_unions(&mut self) {
        for i in 0..self.parent.len() {
            self.parent[i] = i;
            self.rank[i] = 0;
        }
    }

    /// Find root with path compression
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // Path halving
            x = self.parent[x];
        }
        x
    }

    /// Union two bodies into the same island
    pub fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank
        match self.rank[ra].cmp(&self.rank[rb]) {
            core::cmp::Ordering::Less => self.parent[ra] = rb,
            core::cmp::Ordering::Greater => self.parent[rb] = ra,
            core::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }

    /// Build islands from current union-find state
    pub fn build_islands(&mut self) -> Vec<Island> {
        let n = self.parent.len();
        let mut island_map: Vec<Option<usize>> = vec![None; n];
        let mut islands: Vec<Island> = Vec::new();

        for i in 0..n {
            let root = self.find(i);
            let island_idx = if let Some(idx) = island_map[root] {
                idx
            } else {
                let idx = islands.len();
                island_map[root] = Some(idx);
                islands.push(Island {
                    bodies: Vec::new(),
                    all_sleeping: true,
                });
                idx
            };
            islands[island_idx].bodies.push(i);
            if !self.sleep_data[i].is_sleeping() {
                islands[island_idx].all_sleeping = false;
            }
        }

        islands
    }

    /// Update sleep states based on current velocities
    pub fn update_sleep(&mut self, bodies: &[RigidBody]) {
        for (i, body) in bodies.iter().enumerate() {
            if i >= self.sleep_data.len() {
                break;
            }

            // Static bodies are always "sleeping"
            if body.is_static() {
                self.sleep_data[i].state = SleepState::Sleeping;
                continue;
            }

            let linear_speed = body.velocity.length();
            let angular_speed = body.angular_velocity.length();

            let is_idle = linear_speed < self.config.linear_threshold
                && angular_speed < self.config.angular_threshold;

            if is_idle {
                self.sleep_data[i].idle_frames += 1;
                if self.sleep_data[i].idle_frames >= self.config.frames_to_sleep {
                    self.sleep_data[i].state = SleepState::Sleeping;
                }
            } else {
                self.sleep_data[i].state = SleepState::Awake;
                self.sleep_data[i].idle_frames = 0;
            }
        }
    }

    /// Wake up a body and all bodies in its island
    pub fn wake_island(&mut self, body_index: usize) {
        let root = self.find(body_index);
        for i in 0..self.parent.len() {
            if self.find(i) == root {
                self.sleep_data[i].wake();
            }
        }
    }

    /// Wake up a single body
    #[inline]
    pub fn wake_body(&mut self, body_index: usize) {
        if body_index < self.sleep_data.len() {
            self.sleep_data[body_index].wake();
        }
    }

    /// Check if a body is sleeping
    #[inline]
    pub fn is_sleeping(&self, body_index: usize) -> bool {
        self.sleep_data
            .get(body_index)
            .is_some_and(SleepData::is_sleeping)
    }

    /// Get the number of sleeping bodies
    #[must_use]
    pub fn sleeping_count(&self) -> usize {
        self.sleep_data.iter().filter(|d| d.is_sleeping()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3Fix;

    #[test]
    fn test_union_find() {
        let config = SleepConfig::default();
        let mut mgr = IslandManager::new(5, config);

        mgr.union(0, 1);
        mgr.union(2, 3);

        assert_eq!(mgr.find(0), mgr.find(1));
        assert_ne!(mgr.find(0), mgr.find(2));
        assert_eq!(mgr.find(2), mgr.find(3));
    }

    #[test]
    fn test_build_islands() {
        let config = SleepConfig::default();
        let mut mgr = IslandManager::new(5, config);

        mgr.union(0, 1);
        mgr.union(2, 3);
        // Body 4 is alone

        let islands = mgr.build_islands();
        assert_eq!(islands.len(), 3, "Should have 3 islands");

        let total_bodies: usize = islands.iter().map(|i| i.bodies.len()).sum();
        assert_eq!(total_bodies, 5);
    }

    #[test]
    fn test_sleep_after_threshold() {
        let config = SleepConfig {
            frames_to_sleep: 3,
            ..Default::default()
        };
        let mut mgr = IslandManager::new(1, config);

        let still_body = RigidBody::new_static(Vec3Fix::ZERO);
        let bodies = [still_body];

        // Should become sleeping after 3 frames
        for _ in 0..3 {
            mgr.update_sleep(&bodies);
        }

        assert!(mgr.is_sleeping(0));
    }

    #[test]
    fn test_wake_on_motion() {
        let config = SleepConfig {
            linear_threshold: Fix128::from_ratio(1, 100),
            frames_to_sleep: 2,
            ..Default::default()
        };
        let mut mgr = IslandManager::new(1, config);

        // Put to sleep
        let still_body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        mgr.update_sleep(&[still_body]);
        mgr.update_sleep(&[still_body]);
        assert!(mgr.is_sleeping(0));

        // Moving body should wake up
        let mut moving_body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
        moving_body.velocity = Vec3Fix::from_int(10, 0, 0);
        mgr.update_sleep(&[moving_body]);
        assert!(!mgr.is_sleeping(0));
    }

    #[test]
    fn test_wake_island() {
        let config = SleepConfig::default();
        let mut mgr = IslandManager::new(3, config);

        // Put all to sleep
        for d in &mut mgr.sleep_data {
            d.state = SleepState::Sleeping;
        }

        mgr.union(0, 1);

        // Wake body 0 => should wake body 1 too (same island)
        mgr.wake_island(0);

        assert!(!mgr.is_sleeping(0));
        assert!(!mgr.is_sleeping(1));
        assert!(mgr.is_sleeping(2)); // Different island, still sleeping
    }

    #[test]
    fn test_resize() {
        let config = SleepConfig::default();
        let mut mgr = IslandManager::new(2, config);
        mgr.resize(5);

        assert_eq!(mgr.parent.len(), 5);
        assert_eq!(mgr.sleep_data.len(), 5);
    }

    #[test]
    fn test_sleeping_count() {
        let config = SleepConfig::default();
        let mut mgr = IslandManager::new(5, config);

        mgr.sleep_data[0].state = SleepState::Sleeping;
        mgr.sleep_data[2].state = SleepState::Sleeping;
        mgr.sleep_data[4].state = SleepState::Sleeping;

        assert_eq!(mgr.sleeping_count(), 3);
    }
}
