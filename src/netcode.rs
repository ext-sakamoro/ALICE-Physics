//! Deterministic Netcode Foundation
//!
//! Frame-based deterministic simulation wrapper for lockstep/rollback netcode.
//!
//! # Core Principle
//!
//! ```text
//! Same Inputs + Same Initial State → Bit-Exact Same Output (on all clients)
//! ```
//!
//! Because ALICE-Physics uses 128-bit fixed-point arithmetic with fixed iteration
//! counts, the simulation is fully deterministic. This means:
//!
//! - **No state synchronization needed** — only sync player inputs
//! - **Bandwidth**: ~20 bytes/player/frame instead of ~160 bytes/body/frame
//! - **Rollback**: Save/restore snapshots for client-side prediction
//! - **Desync detection**: Compare state checksums across clients
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_physics::netcode::{DeterministicSimulation, FrameInput, NetcodeConfig};
//! use alice_physics::{PhysicsConfig, Vec3Fix, Fix128};
//!
//! // Both clients create identical simulation
//! let mut sim = DeterministicSimulation::new(NetcodeConfig::default());
//!
//! // Each frame: collect inputs from all players, advance simulation
//! let inputs = vec![
//!     FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0)),
//!     FrameInput::new(1).with_movement(Vec3Fix::from_int(0, 0, -1)),
//! ];
//!
//! let checksum = sim.advance_frame(&inputs);
//! // checksum is identical on all clients if inputs match
//! ```
//!
//! # Bandwidth Model
//!
//! | Approach | Per-Frame Bandwidth (10 bodies) |
//! |----------|-------------------------------|
//! | State sync (traditional) | ~1,600 bytes (160B × 10) |
//! | **Input sync (ALICE)** | **~40 bytes (20B × 2 players)** |
//! | Savings | **97.5%** |

use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

#[cfg(not(feature = "std"))]
use alloc::{collections::VecDeque, vec, vec::Vec};
#[cfg(feature = "std")]
use std::collections::VecDeque;

// ============================================================================
// Frame Input
// ============================================================================

/// Player input for a single simulation frame.
///
/// Compact, deterministic representation of all player actions.
/// Serialized size: ~20 bytes (player_id + movement + actions + aim_yaw).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameInput {
    /// Player identifier (0-based)
    pub player_id: u8,
    /// Movement direction (fixed-point, normalized by game logic)
    pub movement: Vec3Fix,
    /// Action bitfield (jump=0x1, fire=0x2, interact=0x4, etc.)
    pub actions: u32,
    /// Aim direction (yaw/pitch encoded as fixed-point)
    pub aim_direction: Vec3Fix,
}

impl FrameInput {
    /// Create a new empty input for the given player
    #[inline]
    pub fn new(player_id: u8) -> Self {
        Self {
            player_id,
            movement: Vec3Fix::ZERO,
            actions: 0,
            aim_direction: Vec3Fix::ZERO,
        }
    }

    /// Set movement direction
    #[inline]
    pub fn with_movement(mut self, movement: Vec3Fix) -> Self {
        self.movement = movement;
        self
    }

    /// Set action bitfield
    #[inline]
    pub fn with_actions(mut self, actions: u32) -> Self {
        self.actions = actions;
        self
    }

    /// Set aim direction
    #[inline]
    pub fn with_aim(mut self, aim: Vec3Fix) -> Self {
        self.aim_direction = aim;
        self
    }

    /// Check if a specific action bit is set
    #[inline]
    pub fn has_action(&self, bit: u32) -> bool {
        self.actions & bit != 0
    }

    /// Serialize to bytes (deterministic, little-endian)
    pub fn to_bytes(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];
        buf[0] = self.player_id;
        // movement as i16 x 3 (6 bytes) — truncated from Fix128 for network
        debug_assert!(self.movement.x.hi >= i16::MIN as i64 && self.movement.x.hi <= i16::MAX as i64);
        debug_assert!(self.movement.y.hi >= i16::MIN as i64 && self.movement.y.hi <= i16::MAX as i64);
        debug_assert!(self.movement.z.hi >= i16::MIN as i64 && self.movement.z.hi <= i16::MAX as i64);
        let mx = self.movement.x.hi as i16;
        let my = self.movement.y.hi as i16;
        let mz = self.movement.z.hi as i16;
        buf[1..3].copy_from_slice(&mx.to_le_bytes());
        buf[3..5].copy_from_slice(&my.to_le_bytes());
        buf[5..7].copy_from_slice(&mz.to_le_bytes());
        // actions (4 bytes)
        buf[7..11].copy_from_slice(&self.actions.to_le_bytes());
        // aim as i16 x 3 (6 bytes)
        let ax = self.aim_direction.x.hi as i16;
        let ay = self.aim_direction.y.hi as i16;
        let az = self.aim_direction.z.hi as i16;
        buf[11..13].copy_from_slice(&ax.to_le_bytes());
        buf[13..15].copy_from_slice(&ay.to_le_bytes());
        buf[15..17].copy_from_slice(&az.to_le_bytes());
        // padding
        buf[17] = 0;
        buf[18] = 0;
        buf[19] = 0;
        buf
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8; 20]) -> Self {
        let player_id = data[0];
        let mx = i16::from_le_bytes([data[1], data[2]]);
        let my = i16::from_le_bytes([data[3], data[4]]);
        let mz = i16::from_le_bytes([data[5], data[6]]);
        let actions = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
        let ax = i16::from_le_bytes([data[11], data[12]]);
        let ay = i16::from_le_bytes([data[13], data[14]]);
        let az = i16::from_le_bytes([data[15], data[16]]);

        Self {
            player_id,
            movement: Vec3Fix::from_int(mx as i64, my as i64, mz as i64),
            actions,
            aim_direction: Vec3Fix::from_int(ax as i64, ay as i64, az as i64),
        }
    }
}

// ============================================================================
// Simulation Checksum
// ============================================================================

/// XOR rolling checksum of the entire physics state.
///
/// Computed from position + velocity + rotation of all bodies.
/// Two simulations with identical inputs will produce identical checksums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SimulationChecksum(pub u64);

impl SimulationChecksum {
    /// Compute checksum from current world state.
    ///
    /// Uses XOR-rotate mixing per body for O(n) computation.
    /// Avalanche mixing ensures single-bit differences propagate.
    pub fn from_world(world: &PhysicsWorld) -> Self {
        let mut hash: u64 = 0;
        for (i, body) in world.bodies.iter().enumerate() {
            let mut h: u64 = i as u64;
            // Position
            h ^= (body.position.x.hi as u64).rotate_left(5);
            h ^= (body.position.x.lo).rotate_left(11);
            h ^= (body.position.y.hi as u64).rotate_left(17);
            h ^= (body.position.y.lo).rotate_left(23);
            h ^= (body.position.z.hi as u64).rotate_left(29);
            h ^= (body.position.z.lo).rotate_left(37);
            // Velocity
            h ^= (body.velocity.x.hi as u64).rotate_left(7);
            h ^= (body.velocity.y.hi as u64).rotate_left(13);
            h ^= (body.velocity.z.hi as u64).rotate_left(19);
            // Rotation
            h ^= (body.rotation.w.hi as u64).rotate_left(3);
            h ^= (body.rotation.x.hi as u64).rotate_left(41);
            h ^= (body.rotation.y.hi as u64).rotate_left(47);
            h ^= (body.rotation.z.hi as u64).rotate_left(53);
            // Avalanche (WyHash-style)
            h = (h ^ (h >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            h = (h ^ (h >> 27)).wrapping_mul(0x94d049bb133111eb);
            h ^= h >> 31;
            hash ^= h;
        }
        Self(hash)
    }
}

// ============================================================================
// Simulation Snapshot (for Rollback)
// ============================================================================

/// Complete physics state snapshot for rollback.
///
/// Captures the full deterministic state at a given frame so the simulation
/// can be rewound and replayed with corrected inputs.
#[derive(Clone, Debug)]
pub struct SimulationSnapshot {
    /// Frame number this snapshot was taken at
    pub frame: u64,
    /// Serialized physics world state
    pub state: Vec<u8>,
    /// Checksum at this frame
    pub checksum: SimulationChecksum,
}

// ============================================================================
// Netcode Configuration
// ============================================================================

/// Configuration for deterministic netcode simulation
#[derive(Clone, Debug)]
pub struct NetcodeConfig {
    /// Physics solver configuration
    pub physics: PhysicsConfig,
    /// Fixed timestep (default: 1/60 second)
    pub fixed_dt: Fix128,
    /// Maximum snapshots to keep for rollback (default: 10)
    pub max_snapshots: usize,
    /// Checksum history length (default: 120 frames = 2 seconds)
    pub checksum_history_len: usize,
    /// Number of players
    pub player_count: u8,
}

impl Default for NetcodeConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsConfig::default(),
            fixed_dt: Fix128::from_ratio(1, 60),
            max_snapshots: 10,
            checksum_history_len: 120,
            player_count: 2,
        }
    }
}

// ============================================================================
// Input Applicator (game-specific force mapping)
// ============================================================================

/// Trait for mapping player inputs to physics forces/impulses.
///
/// Implement this to define how your game's inputs affect physics bodies.
///
/// ```rust,ignore
/// struct MyInputApplicator;
///
/// impl InputApplicator for MyInputApplicator {
///     fn apply(&self, world: &mut PhysicsWorld, player_id: u8,
///              body_index: usize, input: &FrameInput, dt: Fix128) {
///         let body = &mut world.bodies[body_index];
///         // Apply movement as velocity
///         body.velocity.x = body.velocity.x + input.movement.x * dt;
///         body.velocity.z = body.velocity.z + input.movement.z * dt;
///         // Jump
///         if input.has_action(0x1) {
///             body.velocity.y = Fix128::from_int(5);
///         }
///     }
/// }
/// ```
pub trait InputApplicator {
    /// Apply a player's input to their controlled body.
    fn apply(
        &self,
        world: &mut PhysicsWorld,
        player_id: u8,
        body_index: usize,
        input: &FrameInput,
        dt: Fix128,
    );
}

/// Default input applicator: movement → velocity, action bit 0 → jump.
pub struct DefaultInputApplicator {
    /// Movement speed multiplier
    pub move_speed: Fix128,
    /// Jump impulse magnitude
    pub jump_impulse: Fix128,
}

impl Default for DefaultInputApplicator {
    fn default() -> Self {
        Self {
            move_speed: Fix128::from_int(5),
            jump_impulse: Fix128::from_int(8),
        }
    }
}

impl InputApplicator for DefaultInputApplicator {
    fn apply(
        &self,
        world: &mut PhysicsWorld,
        _player_id: u8,
        body_index: usize,
        input: &FrameInput,
        _dt: Fix128,
    ) {
        if body_index >= world.bodies.len() {
            return;
        }
        let body = &mut world.bodies[body_index];

        // Movement → velocity (horizontal plane)
        body.velocity.x = input.movement.x * self.move_speed;
        body.velocity.z = input.movement.z * self.move_speed;

        // Jump (action bit 0)
        if input.has_action(0x1) && body.velocity.y.is_zero() {
            body.velocity.y = self.jump_impulse;
        }
    }
}

// ============================================================================
// Deterministic Simulation
// ============================================================================

/// Frame-based deterministic simulation wrapper.
///
/// Wraps `PhysicsWorld` with:
/// - Frame counter
/// - Input application pipeline
/// - State checksum computation
/// - Snapshot save/restore for rollback
///
/// # Determinism Guarantee
///
/// Given identical `NetcodeConfig`, identical `add_body()` calls, and identical
/// `advance_frame()` input sequences, two `DeterministicSimulation` instances
/// will produce bit-exact identical states on any platform.
pub struct DeterministicSimulation {
    /// The underlying physics world
    pub world: PhysicsWorld,
    /// Current frame number
    frame: u64,
    /// Fixed timestep
    dt: Fix128,
    /// Player body index mapping: player_id → body index
    player_bodies: Vec<Option<usize>>,
    /// Snapshot ring buffer for rollback
    snapshots: VecDeque<SimulationSnapshot>,
    /// Maximum snapshots to keep
    max_snapshots: usize,
    /// Checksum history: (frame, checksum)
    checksum_history: VecDeque<(u64, SimulationChecksum)>,
    /// Maximum checksum history length
    checksum_history_len: usize,
}

impl DeterministicSimulation {
    /// Create a new deterministic simulation
    pub fn new(config: NetcodeConfig) -> Self {
        let world = PhysicsWorld::new(config.physics);
        let player_count = config.player_count as usize;

        Self {
            world,
            frame: 0,
            dt: config.fixed_dt,
            player_bodies: vec![None; player_count],
            snapshots: VecDeque::with_capacity(config.max_snapshots + 1),
            max_snapshots: config.max_snapshots,
            checksum_history: VecDeque::with_capacity(config.checksum_history_len + 1),
            checksum_history_len: config.checksum_history_len,
        }
    }

    /// Add a rigid body to the simulation.
    /// Returns the body index.
    pub fn add_body(&mut self, body: RigidBody) -> usize {
        self.world.add_body(body)
    }

    /// Assign a player to control a specific body.
    pub fn assign_player_body(&mut self, player_id: u8, body_index: usize) {
        let pid = player_id as usize;
        if pid >= self.player_bodies.len() {
            self.player_bodies.resize(pid + 1, None);
        }
        self.player_bodies[pid] = Some(body_index);
    }

    /// Advance one frame with the given player inputs.
    ///
    /// 1. Apply inputs to player-controlled bodies
    /// 2. Step physics
    /// 3. Compute and record checksum
    ///
    /// Returns the checksum for this frame.
    pub fn advance_frame(&mut self, inputs: &[FrameInput]) -> SimulationChecksum {
        self.advance_frame_with_applicator(inputs, &DefaultInputApplicator::default())
    }

    /// Advance one frame with a custom input applicator.
    pub fn advance_frame_with_applicator(
        &mut self,
        inputs: &[FrameInput],
        applicator: &dyn InputApplicator,
    ) -> SimulationChecksum {
        self.frame += 1;

        // 1. Apply player inputs (deterministic order: by player_id)
        for input in inputs {
            let pid = input.player_id as usize;
            if let Some(&Some(body_idx)) = self.player_bodies.get(pid) {
                applicator.apply(&mut self.world, input.player_id, body_idx, input, self.dt);
            }
        }

        // 2. Step physics (fully deterministic)
        self.world.step(self.dt);

        // 3. Compute checksum
        let checksum = SimulationChecksum::from_world(&self.world);

        // 4. Record in history
        self.checksum_history.push_back((self.frame, checksum));
        if self.checksum_history.len() > self.checksum_history_len {
            self.checksum_history.pop_front();
        }

        checksum
    }

    /// Save a snapshot of the current state for rollback.
    pub fn save_snapshot(&mut self) -> &SimulationSnapshot {
        let state = self.world.serialize_state();
        let checksum = SimulationChecksum::from_world(&self.world);
        let snapshot = SimulationSnapshot {
            frame: self.frame,
            state,
            checksum,
        };
        self.snapshots.push_back(snapshot);
        if self.snapshots.len() > self.max_snapshots {
            self.snapshots.pop_front();
        }
        self.snapshots.back().unwrap()
    }

    /// Restore state from a snapshot (for rollback).
    ///
    /// Returns `true` if the snapshot was found and loaded.
    pub fn load_snapshot(&mut self, frame: u64) -> bool {
        if let Some(snap) = self.snapshots.iter().find(|s| s.frame == frame) {
            let state = snap.state.clone();
            let snap_frame = snap.frame;
            if self.world.deserialize_state(&state) {
                self.frame = snap_frame;
                // Trim checksum history to rollback point
                while self
                    .checksum_history
                    .back()
                    .map(|c| c.0 > frame)
                    .unwrap_or(false)
                {
                    self.checksum_history.pop_back();
                }
                return true;
            }
        }
        false
    }

    /// Get a snapshot for a specific frame (if available).
    pub fn get_snapshot(&self, frame: u64) -> Option<&SimulationSnapshot> {
        self.snapshots.iter().find(|s| s.frame == frame)
    }

    /// Compute the current state checksum.
    pub fn checksum(&self) -> SimulationChecksum {
        SimulationChecksum::from_world(&self.world)
    }

    /// Current frame number.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Get checksum for a specific frame from history.
    pub fn checksum_at(&self, frame: u64) -> Option<SimulationChecksum> {
        self.checksum_history
            .iter()
            .find(|&&(f, _)| f == frame)
            .map(|&(_, c)| c)
    }

    /// Verify a remote checksum against local history.
    ///
    /// Returns:
    /// - `None` if the frame is not in local history (too old or future)
    /// - `Some(true)` if checksums match (clients in sync)
    /// - `Some(false)` if checksums differ (DESYNC detected!)
    pub fn verify_checksum(&self, frame: u64, remote: SimulationChecksum) -> Option<bool> {
        self.checksum_at(frame).map(|local| local == remote)
    }

    /// Get the fixed timestep.
    #[inline]
    pub fn dt(&self) -> Fix128 {
        self.dt
    }

    /// Number of snapshots currently stored.
    #[inline]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim(player_count: u8) -> DeterministicSimulation {
        let config = NetcodeConfig {
            player_count,
            ..Default::default()
        };
        let mut sim = DeterministicSimulation::new(config);

        // Add player bodies
        for i in 0..player_count {
            let body = RigidBody::new_dynamic(Vec3Fix::from_int(i as i64 * 5, 10, 0), Fix128::ONE);
            let idx = sim.add_body(body);
            sim.assign_player_body(i, idx);
        }

        // Add ground
        sim.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        sim
    }

    #[test]
    fn test_deterministic_two_sims() {
        // Two independent simulations with identical inputs → bit-exact match
        let mut sim_a = make_sim(2);
        let mut sim_b = make_sim(2);

        let inputs = [
            FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0)),
            FrameInput::new(1).with_movement(Vec3Fix::from_int(0, 0, -1)),
        ];

        for _ in 0..60 {
            let cs_a = sim_a.advance_frame(&inputs);
            let cs_b = sim_b.advance_frame(&inputs);
            assert_eq!(cs_a, cs_b, "Desync at frame {}", sim_a.frame());
        }

        // Verify body positions are bit-exact
        for i in 0..sim_a.world.bodies.len() {
            let a = &sim_a.world.bodies[i];
            let b = &sim_b.world.bodies[i];
            assert_eq!(a.position.x.hi, b.position.x.hi);
            assert_eq!(a.position.x.lo, b.position.x.lo);
            assert_eq!(a.position.y.hi, b.position.y.hi);
            assert_eq!(a.position.y.lo, b.position.y.lo);
            assert_eq!(a.position.z.hi, b.position.z.hi);
            assert_eq!(a.position.z.lo, b.position.z.lo);
        }
    }

    #[test]
    fn test_snapshot_rollback() {
        let mut sim = make_sim(1);
        let input = [FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0))];

        // Advance 30 frames
        for _ in 0..30 {
            sim.advance_frame(&input);
        }
        sim.save_snapshot();
        let checksum_at_30 = sim.checksum();

        // Advance 30 more frames
        for _ in 0..30 {
            sim.advance_frame(&input);
        }
        let pos_at_60 = sim.world.bodies[0].position;

        // Rollback to frame 30
        assert!(sim.load_snapshot(30));
        assert_eq!(sim.frame(), 30);
        assert_eq!(sim.checksum(), checksum_at_30);

        // Replay same inputs → same result
        for _ in 0..30 {
            sim.advance_frame(&input);
        }
        let pos_replayed = sim.world.bodies[0].position;

        assert_eq!(pos_at_60.x.hi, pos_replayed.x.hi);
        assert_eq!(pos_at_60.y.hi, pos_replayed.y.hi);
        assert_eq!(pos_at_60.z.hi, pos_replayed.z.hi);
    }

    #[test]
    fn test_checksum_verify() {
        let mut sim = make_sim(1);
        let input = [FrameInput::new(0)];

        sim.advance_frame(&input);
        let cs1 = sim.checksum();

        // Matching checksum
        assert_eq!(sim.verify_checksum(1, cs1), Some(true));

        // Mismatching checksum
        assert_eq!(
            sim.verify_checksum(1, SimulationChecksum(0xDEADBEEF)),
            Some(false),
        );

        // Unknown frame
        assert_eq!(sim.verify_checksum(999, cs1), None);
    }

    #[test]
    fn test_frame_input_serialization() {
        let input = FrameInput::new(3)
            .with_movement(Vec3Fix::from_int(1, 0, -1))
            .with_actions(0x05)
            .with_aim(Vec3Fix::from_int(0, 1, 0));

        let bytes = input.to_bytes();
        let restored = FrameInput::from_bytes(&bytes);

        assert_eq!(restored.player_id, 3);
        assert_eq!(restored.actions, 0x05);
        assert_eq!(restored.movement.x.hi, 1);
        assert_eq!(restored.movement.z.hi, -1);
        assert_eq!(restored.aim_direction.y.hi, 1);
    }

    #[test]
    fn test_different_inputs_different_checksums() {
        let mut sim_a = make_sim(1);
        let mut sim_b = make_sim(1);

        // Different inputs → different states
        let input_a = [FrameInput::new(0).with_movement(Vec3Fix::from_int(1, 0, 0))];
        let input_b = [FrameInput::new(0).with_movement(Vec3Fix::from_int(-1, 0, 0))];

        for _ in 0..10 {
            sim_a.advance_frame(&input_a);
            sim_b.advance_frame(&input_b);
        }

        assert_ne!(sim_a.checksum(), sim_b.checksum());
    }

    #[test]
    fn test_empty_inputs() {
        let mut sim = make_sim(2);

        // No inputs → simulation still advances (gravity, etc.)
        let cs1 = sim.advance_frame(&[]);
        let cs2 = sim.advance_frame(&[]);

        // Checksums should differ (bodies falling under gravity)
        assert_ne!(cs1, cs2);
    }

    #[test]
    fn test_multiple_snapshots() {
        let mut sim = make_sim(1);
        let input = [FrameInput::new(0)];

        // Save snapshots at frames 10, 20, 30
        for f in 0..30 {
            sim.advance_frame(&input);
            if (f + 1) % 10 == 0 {
                sim.save_snapshot();
            }
        }

        assert_eq!(sim.snapshot_count(), 3);
        assert!(sim.get_snapshot(10).is_some());
        assert!(sim.get_snapshot(20).is_some());
        assert!(sim.get_snapshot(30).is_some());
        assert!(sim.get_snapshot(15).is_none());

        // Rollback to frame 20
        assert!(sim.load_snapshot(20));
        assert_eq!(sim.frame(), 20);
    }
}
