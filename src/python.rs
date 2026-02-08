//! Python Bindings for ALICE-Physics (PyO3 + NumPy Zero-Copy)
//!
//! # Optimization Layers
//!
//! | Layer | Technique | Effect |
//! |-------|-----------|--------|
//! | L1 | GIL Release (`py.allow_threads`) | Parallel physics stepping |
//! | L2 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for bulk data |
//! | L3 | Batch API (positions/velocities) | FFI amortization |
//! | L4 | `#[repr(C)]` FrameInput (20 bytes) | Direct buffer cast |

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray, PyArrayMethods};

use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsWorld, PhysicsConfig, RigidBody};
use crate::netcode::{
    DeterministicSimulation, FrameInput, SimulationChecksum,
    NetcodeConfig, DefaultInputApplicator,
};

// ============================================================================
// PyPhysicsWorld — Core physics world
// ============================================================================

/// Deterministic 128-bit fixed-point physics world.
///
/// All operations are bit-exact across platforms (no floating-point).
#[pyclass(name = "PhysicsWorld")]
pub struct PyPhysicsWorld {
    inner: PhysicsWorld,
}

#[pymethods]
impl PyPhysicsWorld {
    /// Create a new physics world with default configuration.
    #[new]
    fn new() -> Self {
        Self {
            inner: PhysicsWorld::new(PhysicsConfig::default()),
        }
    }

    /// Add a dynamic rigid body at position (x, y, z) with given mass.
    ///
    /// Returns the body index.
    fn add_dynamic_body(&mut self, x: f64, y: f64, z: f64, mass: f64) -> usize {
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_f32(x as f32, y as f32, z as f32),
            Fix128::from_f64(mass),
        );
        self.inner.add_body(body)
    }

    /// Add a static (immovable) body at position (x, y, z).
    ///
    /// Returns the body index.
    fn add_static_body(&mut self, x: f64, y: f64, z: f64) -> usize {
        let body = RigidBody::new_static(
            Vec3Fix::from_f32(x as f32, y as f32, z as f32),
        );
        self.inner.add_body(body)
    }

    /// Step the simulation by dt seconds.
    ///
    /// GIL is released during the physics computation for full parallelism.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        let dt_fix = Fix128::from_f64(dt);
        py.allow_threads(|| {
            self.inner.step(dt_fix);
        });
    }

    /// Step the simulation N times with fixed dt (batch stepping).
    ///
    /// GIL released for the entire batch — ideal for training loops.
    fn step_n(&mut self, py: Python<'_>, dt: f64, steps: usize) {
        let dt_fix = Fix128::from_f64(dt);
        py.allow_threads(|| {
            for _ in 0..steps {
                self.inner.step(dt_fix);
            }
        });
    }

    /// Get all body positions as a NumPy (N, 3) float64 array.
    ///
    /// Zero-copy via direct PyArray allocation.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 3);
        for body in &self.inner.bodies {
            let (x, y, z) = body.position.to_f32();
            data.push(x as f64);
            data.push(y as f64);
            data.push(z as f64);
        }
        data.into_pyarray_bound(py).reshape([n, 3]).unwrap()
    }

    /// Get all body velocities as a NumPy (N, 3) float64 array.
    fn velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 3);
        for body in &self.inner.bodies {
            let (x, y, z) = body.velocity.to_f32();
            data.push(x as f64);
            data.push(y as f64);
            data.push(z as f64);
        }
        data.into_pyarray_bound(py).reshape([n, 3]).unwrap()
    }

    /// Get a single body's position as (x, y, z) tuple.
    fn get_position(&self, body_id: usize) -> PyResult<(f64, f64, f64)> {
        if body_id >= self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("body_id out of range"));
        }
        let (x, y, z) = self.inner.bodies[body_id].position.to_f32();
        Ok((x as f64, y as f64, z as f64))
    }

    /// Set a body's velocity.
    fn set_velocity(&mut self, body_id: usize, vx: f64, vy: f64, vz: f64) -> PyResult<()> {
        if body_id >= self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("body_id out of range"));
        }
        self.inner.bodies[body_id].velocity = Vec3Fix::from_f32(vx as f32, vy as f32, vz as f32);
        Ok(())
    }

    /// Serialize the entire world state to bytes (for rollback/save).
    fn serialize_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let data = py.allow_threads(|| self.inner.serialize_state());
        data.into_pyarray_bound(py)
    }

    /// Restore world state from bytes.
    fn deserialize_state(&mut self, py: Python<'_>, data: Vec<u8>) -> bool {
        py.allow_threads(|| self.inner.deserialize_state(&data))
    }

    /// Number of bodies in the world.
    fn body_count(&self) -> usize {
        self.inner.bodies.len()
    }

    /// Add multiple dynamic bodies from a NumPy (N, 4) array.
    ///
    /// Columns: x, y, z, mass. Returns list of body indices.
    fn add_bodies_batch(&mut self, data: PyReadonlyArray2<f64>) -> PyResult<Vec<usize>> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (N, 4) array with columns [x, y, z, mass]"
            ));
        }

        let n = shape[0];
        let mut indices = Vec::with_capacity(n);

        for i in 0..n {
            let x = array[[i, 0]];
            let y = array[[i, 1]];
            let z = array[[i, 2]];
            let mass = array[[i, 3]];

            let body = RigidBody::new_dynamic(
                Vec3Fix::from_f32(x as f32, y as f32, z as f32),
                Fix128::from_f64(mass),
            );
            let idx = self.inner.add_body(body);
            indices.push(idx);
        }

        Ok(indices)
    }

    /// Set velocities for all bodies from a NumPy (N, 3) array.
    ///
    /// GIL released during the update.
    fn set_velocities_batch(&mut self, py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (N, 3) array with columns [vx, vy, vz]"
            ));
        }

        let n = shape[0];
        if n != self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Array size {} does not match body count {}", n, self.inner.bodies.len())
            ));
        }

        // Collect velocities before GIL release
        let velocities: Vec<Vec3Fix> = (0..n)
            .map(|i| {
                let vx = array[[i, 0]];
                let vy = array[[i, 1]];
                let vz = array[[i, 2]];
                Vec3Fix::from_f32(vx as f32, vy as f32, vz as f32)
            })
            .collect();

        py.allow_threads(|| {
            for (i, vel) in velocities.into_iter().enumerate() {
                self.inner.bodies[i].velocity = vel;
            }
        });

        Ok(())
    }

    /// Apply impulses to specified bodies. data is (M, 4) where columns are: body_id, ix, iy, iz
    fn apply_impulses_batch(&mut self, py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (M, 4) array with columns [body_id, ix, iy, iz]"
            ));
        }

        let m = shape[0];

        // Collect impulses before GIL release
        let impulses: Vec<(usize, Vec3Fix)> = (0..m)
            .map(|i| {
                let body_id = array[[i, 0]] as usize;
                let ix = array[[i, 1]];
                let iy = array[[i, 2]];
                let iz = array[[i, 3]];
                (body_id, Vec3Fix::from_f32(ix as f32, iy as f32, iz as f32))
            })
            .collect();

        // Validate body IDs
        let body_count = self.inner.bodies.len();
        for (body_id, _) in &impulses {
            if *body_id >= body_count {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    format!("body_id {} out of range (max: {})", body_id, body_count - 1)
                ));
            }
        }

        py.allow_threads(|| {
            for (body_id, impulse) in impulses {
                // Apply impulse: v += impulse / mass
                let inv_mass = self.inner.bodies[body_id].inv_mass;
                if !inv_mass.is_zero() {
                    self.inner.bodies[body_id].velocity =
                        self.inner.bodies[body_id].velocity + impulse * inv_mass;
                }
            }
        });

        Ok(())
    }

    /// Get all body states as NumPy (N, 10) array: [px,py,pz, vx,vy,vz, qx,qy,qz,qw]
    fn states<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 10);

        for body in &self.inner.bodies {
            let (px, py, pz) = body.position.to_f32();
            let (vx, vy, vz) = body.velocity.to_f32();
            let qx = body.rotation.x.to_f32();
            let qy = body.rotation.y.to_f32();
            let qz = body.rotation.z.to_f32();
            let qw = body.rotation.w.to_f32();

            data.push(px as f64);
            data.push(py as f64);
            data.push(pz as f64);
            data.push(vx as f64);
            data.push(vy as f64);
            data.push(vz as f64);
            data.push(qx as f64);
            data.push(qy as f64);
            data.push(qz as f64);
            data.push(qw as f64);
        }

        data.into_pyarray_bound(py).reshape([n, 10]).unwrap()
    }

    fn __repr__(&self) -> String {
        format!("<PhysicsWorld bodies={}>", self.inner.bodies.len())
    }
}

// ============================================================================
// PyDeterministicSimulation — Netcode wrapper
// ============================================================================

/// Deterministic simulation for multiplayer netcode.
///
/// Wraps PhysicsWorld with frame-based stepping, checksum verification,
/// and snapshot save/restore for rollback.
///
/// Only sync player inputs (~20 bytes/player/frame) instead of
/// full state (~160 bytes/body/frame) for 97.5% bandwidth savings.
#[pyclass(name = "DeterministicSimulation")]
pub struct PyDeterministicSimulation {
    inner: DeterministicSimulation,
}

#[pymethods]
impl PyDeterministicSimulation {
    /// Create a new deterministic simulation.
    ///
    /// Args:
    ///     player_count: Number of players (default: 2)
    ///     fps: Simulation tick rate (default: 60)
    ///     max_snapshots: Rollback buffer size (default: 10)
    #[new]
    #[pyo3(signature = (player_count=2, fps=60, max_snapshots=10))]
    fn new(player_count: u8, fps: i64, max_snapshots: usize) -> Self {
        let config = NetcodeConfig {
            physics: PhysicsConfig::default(),
            fixed_dt: Fix128::from_ratio(1, fps),
            max_snapshots,
            checksum_history_len: (fps as usize) * 2,
            player_count,
        };
        Self {
            inner: DeterministicSimulation::new(config),
        }
    }

    /// Add a dynamic body. Returns body index.
    fn add_body(&mut self, x: f64, y: f64, z: f64, mass: f64) -> usize {
        let body = RigidBody::new_dynamic(
            Vec3Fix::from_f32(x as f32, y as f32, z as f32),
            Fix128::from_f64(mass),
        );
        self.inner.add_body(body)
    }

    /// Assign a player to control a body.
    fn assign_player(&mut self, player_id: u8, body_index: usize) {
        self.inner.assign_player_body(player_id, body_index);
    }

    /// Advance one frame with player inputs.
    ///
    /// Args:
    ///     inputs: List of (player_id, move_x, move_y, move_z, actions) tuples
    ///
    /// Returns:
    ///     Checksum (u64) for desync detection
    fn advance_frame(
        &mut self,
        py: Python<'_>,
        inputs: Vec<(u8, f64, f64, f64, u32)>,
    ) -> u64 {
        let frame_inputs: Vec<FrameInput> = inputs.iter().map(|&(pid, mx, my, mz, act)| {
            FrameInput::new(pid)
                .with_movement(Vec3Fix::from_f32(mx as f32, my as f32, mz as f32))
                .with_actions(act)
        }).collect();

        let checksum = py.allow_threads(|| {
            self.inner.advance_frame_with_applicator(
                &frame_inputs,
                &DefaultInputApplicator::default(),
            )
        });
        checksum.0
    }

    /// Save a snapshot for rollback. Returns the frame number.
    fn save_snapshot(&mut self) -> u64 {
        let snap = self.inner.save_snapshot();
        snap.frame
    }

    /// Load a snapshot (rollback to given frame). Returns success.
    fn load_snapshot(&mut self, frame: u64) -> bool {
        self.inner.load_snapshot(frame)
    }

    /// Verify a remote checksum. Returns None (frame not found), True (match), False (desync).
    fn verify_checksum(&self, frame: u64, remote_checksum: u64) -> Option<bool> {
        self.inner.verify_checksum(frame, SimulationChecksum(remote_checksum))
    }

    /// Get current frame number.
    fn frame(&self) -> u64 {
        self.inner.frame()
    }

    /// Get current checksum.
    fn checksum(&self) -> u64 {
        self.inner.checksum().0
    }

    /// Get all body positions as NumPy (N, 3) float64 array.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.world.bodies.len();
        let mut data = Vec::with_capacity(n * 3);
        for body in &self.inner.world.bodies {
            let (x, y, z) = body.position.to_f32();
            data.push(x as f64);
            data.push(y as f64);
            data.push(z as f64);
        }
        data.into_pyarray_bound(py).reshape([n, 3]).unwrap()
    }

    fn __repr__(&self) -> String {
        format!(
            "<DeterministicSimulation frame={} bodies={}>",
            self.inner.frame(),
            self.inner.world.bodies.len(),
        )
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Compute simulation checksum from serialized state bytes.
///
/// Useful for verifying state consistency without a full PhysicsWorld.
#[pyfunction]
fn compute_checksum(py: Python<'_>, state_bytes: Vec<u8>) -> u64 {
    py.allow_threads(|| {
        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        if world.deserialize_state(&state_bytes) {
            SimulationChecksum::from_world(&world).0
        } else {
            0
        }
    })
}

/// Encode a FrameInput to 20 bytes.
///
/// Args:
///     player_id, move_x, move_y, move_z, actions, aim_x, aim_y, aim_z
///
/// Returns:
///     bytes (20 bytes)
#[pyfunction]
#[pyo3(signature = (player_id, move_x=0.0, move_y=0.0, move_z=0.0, actions=0, aim_x=0.0, aim_y=0.0, aim_z=0.0))]
fn encode_frame_input(
    player_id: u8,
    move_x: f64, move_y: f64, move_z: f64,
    actions: u32,
    aim_x: f64, aim_y: f64, aim_z: f64,
) -> Vec<u8> {
    let input = FrameInput::new(player_id)
        .with_movement(Vec3Fix::from_f32(move_x as f32, move_y as f32, move_z as f32))
        .with_actions(actions)
        .with_aim(Vec3Fix::from_f32(aim_x as f32, aim_y as f32, aim_z as f32));
    input.to_bytes().to_vec()
}

/// Decode a FrameInput from 20 bytes.
///
/// Returns: (player_id, move_x, move_y, move_z, actions, aim_x, aim_y, aim_z)
#[pyfunction]
fn decode_frame_input(data: Vec<u8>) -> PyResult<(u8, f64, f64, f64, u32, f64, f64, f64)> {
    if data.len() < 20 {
        return Err(pyo3::exceptions::PyValueError::new_err("Need 20 bytes"));
    }
    let mut buf = [0u8; 20];
    buf.copy_from_slice(&data[..20]);
    let input = FrameInput::from_bytes(&buf);
    let (mx, my, mz) = input.movement.to_f32();
    let (ax, ay, az) = input.aim_direction.to_f32();
    Ok((
        input.player_id,
        mx as f64, my as f64, mz as f64,
        input.actions,
        ax as f64, ay as f64, az as f64,
    ))
}

/// Module version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// Module registration
// ============================================================================

#[pymodule]
fn alice_physics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhysicsWorld>()?;
    m.add_class::<PyDeterministicSimulation>()?;
    m.add_function(wrap_pyfunction!(compute_checksum, m)?)?;
    m.add_function(wrap_pyfunction!(encode_frame_input, m)?)?;
    m.add_function(wrap_pyfunction!(decode_frame_input, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_input_encode_decode() {
        let input = FrameInput::new(0)
            .with_movement(Vec3Fix::from_int(1, 0, -1))
            .with_actions(0x3);
        let bytes = input.to_bytes();
        let decoded = FrameInput::from_bytes(&bytes);
        assert_eq!(decoded.player_id, 0);
        assert_eq!(decoded.actions, 0x3);
    }
}
