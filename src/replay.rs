//! Replay Recording/Playback via ALICE-DB
//!
//! Streams rigid body positions to ALICE-DB as time-series data.
//! ALICE-DB's model-based compression (polynomial, Fourier) fits physics
//! trajectories naturally — constant velocity becomes a linear model,
//! projectile arcs become quadratics.
//!
//! # Key Encoding
//!
//! Single DB instance with multiplexed channels:
//! ```text
//! timestamp = channel * MAX_FRAMES + frame
//! channel   = body_id * 6 + component
//! component = 0:pos_x, 1:pos_y, 2:pos_z, 3:vel_x, 4:vel_y, 5:vel_z
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Vec3Fix, Fix128};
//! use alice_physics::replay::{ReplayRecorder, ReplayPlayer};
//!
//! let config = PhysicsConfig::default();
//! let mut world = PhysicsWorld::new(config);
//! let body_id = world.add_body(RigidBody::new_dynamic(
//!     Vec3Fix::from_int(0, 10, 0), Fix128::ONE,
//! ));
//!
//! // Record
//! let mut recorder = ReplayRecorder::new("./replay_data", 1).unwrap();
//! let dt = Fix128::from_ratio(1, 60);
//! for _ in 0..60 {
//!     world.step(dt);
//!     recorder.record_frame(&world).unwrap();
//! }
//! recorder.close().unwrap();
//!
//! // Playback
//! let player = ReplayPlayer::open("./replay_data", 1).unwrap();
//! let pos = player.get_position(30, 0).unwrap(); // frame 30, body 0
//! ```

use crate::solver::PhysicsWorld;
use alice_db::AliceDB;
use std::io;
use std::path::Path;

/// Maximum frames per channel (~46 hours at 60fps)
const MAX_FRAMES: i64 = 10_000_000;

/// Components per body: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
const COMPONENTS_PER_BODY: i64 = 6;

/// Records rigid body positions and velocities to ALICE-DB each frame.
///
/// Physics trajectories compress extremely well:
/// - Constant velocity → linear model (2 coefficients)
/// - Projectile arc → quadratic model (3 coefficients)
/// - Oscillation → Fourier model
///
/// Uses a reusable internal buffer to avoid heap allocation per frame.
pub struct ReplayRecorder {
    db: AliceDB,
    frame: u64,
    body_count: usize,
    /// Pre-computed channel base offsets: `body_id * 6 * MAX_FRAMES`
    channel_bases: Vec<i64>,
    /// Reusable batch buffer — allocated once, cleared each frame
    batch_buf: Vec<(i64, f32)>,
}

impl ReplayRecorder {
    /// Create a new replay recorder.
    ///
    /// # Arguments
    /// * `path` - Directory for ALICE-DB storage
    /// * `body_count` - Number of bodies to record per frame
    pub fn new<P: AsRef<Path>>(path: P, body_count: usize) -> io::Result<Self> {
        let db = AliceDB::open(path)?;
        // Pre-compute channel base offsets to avoid per-frame multiplication
        let channel_bases: Vec<i64> = (0..body_count)
            .map(|i| i as i64 * COMPONENTS_PER_BODY * MAX_FRAMES)
            .collect();
        let batch_buf = Vec::with_capacity(body_count * COMPONENTS_PER_BODY as usize);
        Ok(Self { db, frame: 0, body_count, channel_bases, batch_buf })
    }

    /// Record all body positions and velocities for the current frame.
    ///
    /// Zero heap allocation after the first call (reuses internal buffer).
    #[inline]
    pub fn record_frame(&mut self, world: &PhysicsWorld) -> io::Result<()> {
        let frame = self.frame as i64;
        let count = self.body_count.min(world.bodies.len());
        self.batch_buf.clear();

        for i in 0..count {
            let body = &world.bodies[i];
            let (px, py, pz) = body.position.to_f32();
            let (vx, vy, vz) = body.velocity.to_f32();
            let base = self.channel_bases[i];

            self.batch_buf.push((base                        + frame, px));
            self.batch_buf.push((base +     MAX_FRAMES       + frame, py));
            self.batch_buf.push((base + 2 * MAX_FRAMES       + frame, pz));
            self.batch_buf.push((base + 3 * MAX_FRAMES       + frame, vx));
            self.batch_buf.push((base + 4 * MAX_FRAMES       + frame, vy));
            self.batch_buf.push((base + 5 * MAX_FRAMES       + frame, vz));
        }

        self.db.put_batch(&self.batch_buf)?;
        self.frame += 1;
        Ok(())
    }

    /// Record only positions (smaller footprint, no velocity).
    ///
    /// Zero heap allocation after the first call.
    #[inline]
    pub fn record_positions(&mut self, world: &PhysicsWorld) -> io::Result<()> {
        let frame = self.frame as i64;
        let count = self.body_count.min(world.bodies.len());
        self.batch_buf.clear();

        for i in 0..count {
            let body = &world.bodies[i];
            let (px, py, pz) = body.position.to_f32();
            let base = self.channel_bases[i];

            self.batch_buf.push((base                        + frame, px));
            self.batch_buf.push((base +     MAX_FRAMES       + frame, py));
            self.batch_buf.push((base + 2 * MAX_FRAMES       + frame, pz));
        }

        self.db.put_batch(&self.batch_buf)?;
        self.frame += 1;
        Ok(())
    }

    /// Number of frames recorded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame
    }

    /// Flush buffered data to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.db.flush()
    }

    /// Close the recorder, flushing all data.
    pub fn close(self) -> io::Result<()> {
        self.db.flush()?;
        self.db.close()
    }
}

/// Plays back recorded physics data from ALICE-DB.
///
/// Positions are reconstructed from ALICE-DB's fitted models,
/// meaning O(1) point queries (compute polynomial/Fourier, no disk seek).
pub struct ReplayPlayer {
    db: AliceDB,
    body_count: usize,
}

impl ReplayPlayer {
    /// Open a replay for playback.
    pub fn open<P: AsRef<Path>>(path: P, body_count: usize) -> io::Result<Self> {
        let db = AliceDB::open(path)?;
        Ok(Self { db, body_count })
    }

    /// Get position of a body at a specific frame.
    ///
    /// Returns `None` if the frame/body wasn't recorded.
    #[inline]
    pub fn get_position(&self, frame: u64, body_id: usize) -> io::Result<Option<(f32, f32, f32)>> {
        let f = frame as i64;
        let base = body_id as i64 * COMPONENTS_PER_BODY * MAX_FRAMES;

        let x = self.db.get(base                  + f)?;
        let y = self.db.get(base +     MAX_FRAMES + f)?;
        let z = self.db.get(base + 2 * MAX_FRAMES + f)?;

        match (x, y, z) {
            (Some(x), Some(y), Some(z)) => Ok(Some((x, y, z))),
            _ => Ok(None),
        }
    }

    /// Get velocity of a body at a specific frame.
    #[inline]
    pub fn get_velocity(&self, frame: u64, body_id: usize) -> io::Result<Option<(f32, f32, f32)>> {
        let f = frame as i64;
        let base = body_id as i64 * COMPONENTS_PER_BODY * MAX_FRAMES;

        let vx = self.db.get(base + 3 * MAX_FRAMES + f)?;
        let vy = self.db.get(base + 4 * MAX_FRAMES + f)?;
        let vz = self.db.get(base + 5 * MAX_FRAMES + f)?;

        match (vx, vy, vz) {
            (Some(vx), Some(vy), Some(vz)) => Ok(Some((vx, vy, vz))),
            _ => Ok(None),
        }
    }

    /// Scan a range of frames for a body's position (returns frame-relative timestamps).
    pub fn scan_positions(
        &self,
        body_id: usize,
        start_frame: u64,
        end_frame: u64,
    ) -> io::Result<Vec<(u64, f32, f32, f32)>> {
        let base = body_id as i64 * COMPONENTS_PER_BODY * MAX_FRAMES;
        let s = start_frame as i64;
        let e = end_frame as i64;

        let xs = self.db.scan(base                  + s, base                  + e)?;
        let ys = self.db.scan(base +     MAX_FRAMES + s, base +     MAX_FRAMES + e)?;
        let zs = self.db.scan(base + 2 * MAX_FRAMES + s, base + 2 * MAX_FRAMES + e)?;

        let mut result = Vec::with_capacity(xs.len());
        for ((xt, xv), ((_, yv), (_, zv))) in xs.into_iter().zip(ys.into_iter().zip(zs.into_iter())) {
            let frame = (xt - base) as u64;
            result.push((frame, xv, yv, zv));
        }

        Ok(result)
    }

    /// Number of bodies in this replay.
    pub fn body_count(&self) -> usize {
        self.body_count
    }

    /// Close the player.
    pub fn close(self) -> io::Result<()> {
        self.db.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, Fix128};

    #[test]
    fn test_replay_record_and_playback() {
        let dir = tempfile::tempdir().unwrap();
        let replay_path = dir.path().join("replay");

        // Create world and simulate
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 100, 0),
            Fix128::ONE,
        ));

        // Record 10 frames
        let mut recorder = ReplayRecorder::new(&replay_path, 1).unwrap();
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..10 {
            world.step(dt);
            recorder.record_frame(&world).unwrap();
        }
        assert_eq!(recorder.frame_count(), 10);
        recorder.close().unwrap();

        // Playback
        let player = ReplayPlayer::open(&replay_path, 1).unwrap();
        let pos = player.get_position(0, 0).unwrap();
        assert!(pos.is_some(), "Frame 0 should have data");

        let (_, y, _) = pos.unwrap();
        assert!(y < 100.0, "Body should have fallen from 100");

        player.close().unwrap();
    }

    #[test]
    fn test_replay_positions_only() {
        let dir = tempfile::tempdir().unwrap();
        let replay_path = dir.path().join("replay_pos");

        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(5, 50, 0),
            Fix128::ONE,
        ));

        let mut recorder = ReplayRecorder::new(&replay_path, 1).unwrap();
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..5 {
            world.step(dt);
            recorder.record_positions(&world).unwrap();
        }
        recorder.close().unwrap();

        let player = ReplayPlayer::open(&replay_path, 1).unwrap();

        // Position should be available
        assert!(player.get_position(0, 0).unwrap().is_some());

        // Velocity was not recorded
        assert!(player.get_velocity(0, 0).unwrap().is_none());

        player.close().unwrap();
    }
}
