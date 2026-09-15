//! Replay Recording/Playback via ALICE-DB
//!
//! Streams rigid body positions to ALICE-DB as time-series data.
//! ALICE-DB's model-based compression (polynomial, Fourier) fits physics
//! trajectories naturally — constant velocity becomes a linear model,
//! projectile arcs become quadratics.
//!
//! # Key Encoding
//!
//! Single DB instance, channels **interleaved per frame** so the key sequence
//! is dense and uniformly spaced — the property ALICE-DB's segment models
//! (including the `RawLzma` fallback) assume when they map a timestamp back
//! to a sample index:
//! ```text
//! timestamp = frame * channels + body_id * components + component
//! channels  = body_count * components
//! components = 6 (record_frame: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
//!            | 3 (record_positions: pos_x, pos_y, pos_z)
//! ```
//! The recorder writes the layout (`body_count`, `components`) to
//! `<path>/replay_layout` so the player can decode without being told.
//! Before 1.2.0 the layout was `channel * MAX_FRAMES + frame` (one block of
//! 10⁷ keys per channel); inside one flushed segment those keys are sparse
//! and irregular, so every read landed on the wrong sample
//! (`scan_positions_matches_get_position_per_frame_and_body`).
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

/// Components per body when velocities are recorded: pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
const FULL_COMPONENTS: usize = 6;
/// Components per body for position-only recordings
const POSITION_COMPONENTS: usize = 3;
/// Layout manifest written next to the ALICE-DB files
const LAYOUT_FILE: &str = "replay_layout";

/// A replay of a deterministic engine must read back the bits it wrote.
/// ALICE-DB fits procedural models (polynomial / Fourier) to a series and,
/// by default, keeps the model when its relative error is below a threshold —
/// a *lossy* reconstruction. `FitConfig::lossless` stores the per-sample
/// residuals so the model + residual is exact (requires alice-db ≥
/// 0.2.0-beta.2, where the mmap read path applies them).
fn open_lossless(path: &Path) -> io::Result<AliceDB> {
    let config = alice_db::StorageConfig {
        data_dir: path.to_path_buf(),
        fit_config: alice_db::FitConfig {
            lossless: true,
            ..alice_db::FitConfig::default()
        },
        ..alice_db::StorageConfig::default()
    };
    AliceDB::with_config(config)
}

fn layout_path(db_path: &Path) -> std::path::PathBuf {
    db_path.join(LAYOUT_FILE)
}

fn write_layout(db_path: &Path, body_count: usize, components: usize) -> io::Result<()> {
    std::fs::create_dir_all(db_path)?;
    std::fs::write(layout_path(db_path), format!("{body_count} {components}\n"))
}

fn read_layout(db_path: &Path) -> io::Result<Option<(usize, usize)>> {
    match std::fs::read_to_string(layout_path(db_path)) {
        Ok(text) => {
            let mut it = text.split_whitespace();
            let parse = |s: Option<&str>| s.and_then(|v| v.parse::<usize>().ok());
            match (parse(it.next()), parse(it.next())) {
                (Some(b), Some(c)) if c == FULL_COMPONENTS || c == POSITION_COMPONENTS => {
                    Ok(Some((b, c)))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "malformed replay_layout",
                )),
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

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
    /// Components per body (6 after `record_frame`, 3 after
    /// `record_positions`); fixed by the first recorded frame.
    components: Option<usize>,
    db_path: std::path::PathBuf,
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
        let db_path = path.as_ref().to_path_buf();
        let db = open_lossless(&db_path)?;
        let batch_buf = Vec::with_capacity(body_count * FULL_COMPONENTS);
        Ok(Self {
            db,
            frame: 0,
            body_count,
            components: None,
            db_path,
            batch_buf,
        })
    }

    /// Fix the layout on the first recorded frame (and reject mixing
    /// `record_frame` with `record_positions` in one recording, which would
    /// break the dense key sequence).
    fn set_components(&mut self, components: usize) -> io::Result<()> {
        match self.components {
            None => {
                write_layout(&self.db_path, self.body_count, components)?;
                self.components = Some(components);
                Ok(())
            }
            Some(c) if c == components => Ok(()),
            Some(_) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a replay records either frames (6 components) or positions (3); do not mix",
            )),
        }
    }

    #[inline]
    fn key(&self, frame: i64, body: usize, component: usize, components: usize) -> i64 {
        frame * (self.body_count * components) as i64 + (body * components + component) as i64
    }

    /// Record all body positions and velocities for the current frame.
    ///
    /// Zero heap allocation after the first call (reuses internal buffer).
    #[inline]
    pub fn record_frame(&mut self, world: &PhysicsWorld) -> io::Result<()> {
        self.set_components(FULL_COMPONENTS)?;
        let frame = self.frame as i64;
        self.batch_buf.clear();

        // every body of the layout gets its 6 keys (missing bodies as 0.0) so
        // the key sequence stays dense
        for i in 0..self.body_count {
            let (px, py, pz, vx, vy, vz) = match world.bodies.get(i) {
                Some(body) => {
                    let (px, py, pz) = body.position.to_f32();
                    let (vx, vy, vz) = body.velocity.to_f32();
                    (px, py, pz, vx, vy, vz)
                }
                None => (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            };
            for (c, v) in [px, py, pz, vx, vy, vz].into_iter().enumerate() {
                self.batch_buf
                    .push((self.key(frame, i, c, FULL_COMPONENTS), v));
            }
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
        self.set_components(POSITION_COMPONENTS)?;
        let frame = self.frame as i64;
        self.batch_buf.clear();

        for i in 0..self.body_count {
            let (px, py, pz) = world
                .bodies
                .get(i)
                .map_or((0.0, 0.0, 0.0), |b| b.position.to_f32());
            for (c, v) in [px, py, pz].into_iter().enumerate() {
                self.batch_buf
                    .push((self.key(frame, i, c, POSITION_COMPONENTS), v));
            }
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
    /// Components per body from `replay_layout` (6 or 3; 6 when the file is absent)
    components: usize,
}

impl ReplayPlayer {
    /// Open a replay for playback.
    pub fn open<P: AsRef<Path>>(path: P, body_count: usize) -> io::Result<Self> {
        let layout = read_layout(path.as_ref())?;
        let db = open_lossless(path.as_ref())?;
        let (body_count, components) = match layout {
            Some((b, c)) => (b, c),
            None => (body_count, FULL_COMPONENTS),
        };
        Ok(Self {
            db,
            body_count,
            components,
        })
    }

    #[inline]
    fn key(&self, frame: u64, body: usize, component: usize) -> i64 {
        frame as i64 * (self.body_count * self.components) as i64
            + (body * self.components + component) as i64
    }

    /// Get position of a body at a specific frame.
    ///
    /// Returns `None` if the frame/body wasn't recorded.
    #[inline]
    pub fn get_position(&self, frame: u64, body_id: usize) -> io::Result<Option<(f32, f32, f32)>> {
        if body_id >= self.body_count {
            return Ok(None);
        }
        let x = self.db.get(self.key(frame, body_id, 0))?;
        let y = self.db.get(self.key(frame, body_id, 1))?;
        let z = self.db.get(self.key(frame, body_id, 2))?;

        match (x, y, z) {
            (Some(x), Some(y), Some(z)) => Ok(Some((x, y, z))),
            _ => Ok(None),
        }
    }

    /// Get velocity of a body at a specific frame.
    #[inline]
    pub fn get_velocity(&self, frame: u64, body_id: usize) -> io::Result<Option<(f32, f32, f32)>> {
        if body_id >= self.body_count || self.components < FULL_COMPONENTS {
            return Ok(None);
        }
        let vx = self.db.get(self.key(frame, body_id, 3))?;
        let vy = self.db.get(self.key(frame, body_id, 4))?;
        let vz = self.db.get(self.key(frame, body_id, 5))?;

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
        if body_id >= self.body_count || end_frame < start_frame {
            return Ok(Vec::new());
        }
        // one dense range over the interleaved keys, then pick this body's
        // three position channels frame by frame
        let channels = (self.body_count * self.components) as i64;
        let rows = self.db.scan(
            self.key(start_frame, 0, 0),
            self.key(end_frame, self.body_count - 1, self.components - 1),
        )?;
        let mut result: Vec<(u64, f32, f32, f32)> = Vec::new();
        let mut current: Option<(u64, [Option<f32>; 3])> = None;
        for (key, value) in rows {
            let frame = (key / channels) as u64;
            let ch = (key % channels) as usize;
            if ch / self.components != body_id {
                continue;
            }
            let comp = ch % self.components;
            if comp > 2 {
                continue;
            }
            match &mut current {
                Some((f, acc)) if *f == frame => acc[comp] = Some(value),
                _ => {
                    if let Some((f, [Some(x), Some(y), Some(z)])) = current.take() {
                        result.push((f, x, y, z));
                    }
                    let mut acc = [None; 3];
                    acc[comp] = Some(value);
                    current = Some((frame, acc));
                }
            }
        }
        if let Some((f, [Some(x), Some(y), Some(z)])) = current {
            result.push((f, x, y, z));
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
    use crate::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

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

    /// `scan_positions` returns the recorded frames of one body in order, with
    /// frame-relative indices, and agrees with `get_position` frame by frame;
    /// it does not leak other bodies' rows.
    #[test]
    #[allow(clippy::disallowed_methods)] // f32::powi builds the closed-form reference only
    fn scan_positions_matches_get_position_per_frame_and_body() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scan.replay");
        let mut world = PhysicsWorld::new(PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            ..PhysicsConfig::default()
        });
        let mut a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        a.velocity = Vec3Fix::from_int(6, 0, 0); // 0.1 m per frame at 60 Hz
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
        b.velocity = Vec3Fix::from_int(0, 0, -6);
        world.add_body(a);
        world.add_body(b);
        let mut recorder = ReplayRecorder::new(&path, 2).unwrap();
        for _ in 0..8 {
            world.step(Fix128::from_ratio(1, 60));
            recorder.record_frame(&world).unwrap();
        }
        recorder.close().unwrap();

        let player = ReplayPlayer::open(&path, 2).unwrap();
        let rows = player.scan_positions(0, 2, 5).unwrap();
        assert_eq!(rows.len(), 4, "inclusive frame range 2..=5: {rows:?}");
        for (i, (frame, x, y, z)) in rows.iter().enumerate() {
            assert_eq!(*frame, 2 + i as u64, "frame-relative index");
            let (gx, gy, gz) = player.get_position(*frame, 0).unwrap().expect("recorded");
            assert!((x - gx).abs() < 1e-6 && (y - gy).abs() < 1e-6 && (z - gz).abs() < 1e-6);
            // body 0 moves along +x: 0.1 m in frame 0, then × 0.99 per frame (default
            // frame damping) → x_n = 0.1 · Σ_{k=0}^{n} 0.99^k
            let want_x: f32 = (0..=*frame).map(|k| 0.1 * 0.99f32.powi(k as i32)).sum();
            assert!(
                (x - want_x).abs() < 1e-4,
                "frame {frame}: x = {x}, want {want_x}"
            );
            assert!(
                y.abs() < 1e-6 && z.abs() < 1e-6,
                "body 0 does not move in y/z"
            );
        }
        // body 1: y stays 10, z decreases — proves the scan is per body, not interleaved
        let rows_b = player.scan_positions(1, 0, 7).unwrap();
        assert_eq!(rows_b.len(), 8);
        for (frame, x, y, z) in rows_b {
            assert!(
                x.abs() < 1e-6 && (y - 10.0).abs() < 1e-6,
                "body 1 frame {frame}: ({x}, {y}, {z})"
            );
            let want_z: f32 = -(0..=frame)
                .map(|k| 0.1 * 0.99f32.powi(k as i32))
                .sum::<f32>();
            assert!(
                (z - want_z).abs() < 1e-4,
                "body 1 frame {frame}: z = {z}, want {want_z}"
            );
        }
        // out of range: empty
        assert!(player.scan_positions(0, 20, 30).unwrap().is_empty());
        player.close().unwrap();
    }
}
