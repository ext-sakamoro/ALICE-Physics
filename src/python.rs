// PyO3 macro expansion generates identity conversions for PyResult wrappers
#![allow(clippy::useless_conversion)]
//! Python Bindings for ALICE-Physics (PyO3 + NumPy Zero-Copy)
//!
//! # Optimization Layers
//!
//! | Layer | Technique | Effect |
//! |-------|-----------|--------|
//! | L1 | GIL Release (`py.detach`) | Parallel physics stepping |
//! | L2 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for bulk data |
//! | L3 | Batch API (positions/velocities) | FFI amortization |
//! | L4 | `#[repr(C)]` FrameInput (20 bytes) | Direct buffer cast |

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::binding_api;
use crate::joint::{BallJoint, FixedJoint, HingeJoint, Joint, SliderJoint, SpringJoint};
use crate::math::{Fix128, Vec3Fix};
use crate::netcode::{
    DefaultInputApplicator, DeterministicSimulation, FrameInput, NetcodeConfig, SimulationChecksum,
};
use crate::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

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

#[allow(clippy::useless_conversion)]
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
        let body = RigidBody::new_static(Vec3Fix::from_f32(x as f32, y as f32, z as f32));
        self.inner.add_body(body)
    }

    /// Step the simulation by dt seconds.
    ///
    /// GIL is released during the physics computation for full parallelism.
    fn step(&mut self, py: Python<'_>, dt: f64) {
        let dt_fix = Fix128::from_f64(dt);
        py.detach(|| {
            self.inner.step(dt_fix);
        });
    }

    /// Step the simulation N times with fixed dt (batch stepping).
    ///
    /// GIL released for the entire batch — ideal for training loops.
    fn step_n(&mut self, py: Python<'_>, dt: f64, steps: usize) {
        let dt_fix = Fix128::from_f64(dt);
        py.detach(|| {
            self.inner.step_n(steps, dt_fix);
        });
    }

    /// Get all body positions as a NumPy (N, 3) float64 array.
    ///
    /// Pre-allocated flat buffer with direct indexing, then zero-copy
    /// ownership transfer via `into_pyarray`.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.bodies.len();
        let mut data = vec![0.0f64; n * 3];
        for (i, body) in self.inner.bodies.iter().enumerate() {
            let (x, y, z) = body.position.to_f32();
            let base = i * 3;
            data[base] = x as f64;
            data[base + 1] = y as f64;
            data[base + 2] = z as f64;
        }
        data.into_pyarray(py)
            .reshape([n, 3])
            .expect("buffer length is n*3")
    }

    /// Get all body velocities as a NumPy (N, 3) float64 array.
    fn velocities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n = self.inner.bodies.len();
        let mut data = vec![0.0f64; n * 3];
        for (i, body) in self.inner.bodies.iter().enumerate() {
            let (x, y, z) = body.velocity.to_f32();
            let base = i * 3;
            data[base] = x as f64;
            data[base + 1] = y as f64;
            data[base + 2] = z as f64;
        }
        data.into_pyarray(py)
            .reshape([n, 3])
            .expect("buffer length is n*3")
    }

    /// Get a single body's position as (x, y, z) tuple.
    fn get_position(&self, body_id: usize) -> PyResult<(f64, f64, f64)> {
        if body_id >= self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "body_id out of range",
            ));
        }
        let (x, y, z) = self.inner.bodies[body_id].position.to_f32();
        Ok((x as f64, y as f64, z as f64))
    }

    /// Set a body's velocity.
    fn set_velocity(&mut self, body_id: usize, vx: f64, vy: f64, vz: f64) -> PyResult<()> {
        if body_id >= self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "body_id out of range",
            ));
        }
        self.inner.bodies[body_id].velocity = Vec3Fix::from_f32(vx as f32, vy as f32, vz as f32);
        Ok(())
    }

    /// Set positions for all bodies from a NumPy (N, 3) array.
    ///
    /// GIL released during the update.
    fn set_positions_batch(&mut self, py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (N, 3) array with columns [x, y, z]",
            ));
        }

        let n = shape[0];
        if n != self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Array size {} does not match body count {}",
                n,
                self.inner.bodies.len()
            )));
        }

        // Collect positions before GIL release
        let positions: Vec<Vec3Fix> = (0..n)
            .map(|i| {
                Vec3Fix::from_f32(
                    array[[i, 0]] as f32,
                    array[[i, 1]] as f32,
                    array[[i, 2]] as f32,
                )
            })
            .collect();

        py.detach(|| {
            for (i, pos) in positions.into_iter().enumerate() {
                self.inner.bodies[i].position = pos;
            }
        });

        Ok(())
    }

    /// Serialize the entire world state to bytes (for rollback/save).
    fn serialize_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let data = py.detach(|| self.inner.serialize_state());
        data.into_pyarray(py)
    }

    /// Restore world state from bytes.
    fn deserialize_state(&mut self, py: Python<'_>, data: Vec<u8>) -> bool {
        py.detach(|| self.inner.deserialize_state(&data))
    }

    /// Number of bodies in the world.
    fn body_count(&self) -> usize {
        self.inner.bodies.len()
    }

    /// Add multiple dynamic bodies from a NumPy (N, 4) array.
    ///
    /// Columns: x, y, z, mass. Returns list of body indices.
    /// GIL released during body creation and world insertion.
    fn add_bodies_batch(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Vec<usize>> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (N, 4) array with columns [x, y, z, mass]",
            ));
        }

        let n = shape[0];

        // Gather parameters from NumPy while GIL is held
        let params: Vec<(f32, f32, f32, f64)> = (0..n)
            .map(|i| {
                (
                    array[[i, 0]] as f32,
                    array[[i, 1]] as f32,
                    array[[i, 2]] as f32,
                    array[[i, 3]],
                )
            })
            .collect();

        // Release GIL for body creation and world insertion
        let indices = py.detach(|| {
            let mut out = Vec::with_capacity(n);
            for &(x, y, z, mass) in &params {
                let body =
                    RigidBody::new_dynamic(Vec3Fix::from_f32(x, y, z), Fix128::from_f64(mass));
                out.push(self.inner.add_body(body));
            }
            out
        });

        Ok(indices)
    }

    /// Set velocities for all bodies from a NumPy (N, 3) array.
    ///
    /// GIL released during the update.
    fn set_velocities_batch(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (N, 3) array with columns [vx, vy, vz]",
            ));
        }

        let n = shape[0];
        if n != self.inner.bodies.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Array size {} does not match body count {}",
                n,
                self.inner.bodies.len()
            )));
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

        py.detach(|| {
            for (i, vel) in velocities.into_iter().enumerate() {
                self.inner.bodies[i].velocity = vel;
            }
        });

        Ok(())
    }

    /// Apply impulses to specified bodies. data is (M, 4) where columns are: body_id, ix, iy, iz
    fn apply_impulses_batch(
        &mut self,
        py: Python<'_>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let array = data.as_array();
        let shape = array.shape();

        if shape.len() != 2 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Expected (M, 4) array with columns [body_id, ix, iy, iz]",
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
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "body_id {} out of range (max: {})",
                    body_id,
                    body_count - 1
                )));
            }
        }

        py.detach(|| {
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
        let mut data = vec![0.0f64; n * 10];

        for (i, body) in self.inner.bodies.iter().enumerate() {
            let (px, py_val, pz) = body.position.to_f32();
            let (vx, vy, vz) = body.velocity.to_f32();
            let base = i * 10;
            data[base] = px as f64;
            data[base + 1] = py_val as f64;
            data[base + 2] = pz as f64;
            data[base + 3] = vx as f64;
            data[base + 4] = vy as f64;
            data[base + 5] = vz as f64;
            data[base + 6] = body.rotation.x.to_f32() as f64;
            data[base + 7] = body.rotation.y.to_f32() as f64;
            data[base + 8] = body.rotation.z.to_f32() as f64;
            data[base + 9] = body.rotation.w.to_f32() as f64;
        }

        data.into_pyarray(py)
            .reshape([n, 10])
            .expect("buffer length is n*10")
    }

    fn __repr__(&self) -> String {
        format!("<PhysicsWorld bodies={}>", self.inner.bodies.len())
    }

    // ------------------------------------------------------------------
    // Collision radius, shapes, static colliders, joints. A refused
    // argument raises `ValueError` (see `binding_api` for the checks), an
    // unknown body / index raises `IndexError`.
    // ------------------------------------------------------------------

    /// Set a body's collision sphere radius (finite and positive).
    fn set_collision_radius(&mut self, body_id: usize, radius: f64) -> PyResult<()> {
        self.check_body(body_id)?;
        ok_or_value(
            binding_api::set_collision_radius(&mut self.inner, body_id, radius),
            "radius must be finite and positive",
        )
    }

    /// Drop a body's own collision radius (it falls back to the world default).
    fn clear_collision_radius(&mut self, body_id: usize) -> PyResult<()> {
        self.check_body(body_id)?;
        binding_api::clear_collision_radius(&mut self.inner, body_id);
        Ok(())
    }

    /// Add a dynamic body with a collision shape and return its index.
    ///
    /// `kind`: 0 box (half extents a, b, c), 1 cylinder (radius a, half
    /// height b), 2 cone (radius a, half height b), 3 ellipsoid (radii a, b,
    /// c), 4 wedge (width a, height b, depth c), 5 torus (major a, minor b).
    /// Mass and inertia come from `density`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kind, a, b, c, density, x, y, z))]
    fn add_shaped_body(
        &mut self,
        kind: u32,
        a: f64,
        b: f64,
        c: f64,
        density: f64,
        x: f64,
        y: f64,
        z: f64,
    ) -> PyResult<usize> {
        let shape = some_or_value(binding_api::shape(kind, a, b, c), "invalid shape")?;
        let p = some_or_value(binding_api::vec3(x, y, z), "position must be finite")?;
        some_or_value(
            binding_api::add_shaped_body(&mut self.inner, shape, density, p),
            "density must be finite and positive and the shape must have a mass",
        )
    }

    /// Give an existing body a collision shape (its mass is unchanged).
    fn set_body_shape(
        &mut self,
        body_id: usize,
        kind: u32,
        a: f64,
        b: f64,
        c: f64,
    ) -> PyResult<()> {
        self.check_body(body_id)?;
        let shape = some_or_value(binding_api::shape(kind, a, b, c), "invalid shape")?;
        ok_or_value(
            binding_api::set_body_shape(&mut self.inner, body_id, shape),
            "invalid shape",
        )
    }

    /// Add the static plane `normal · p = offset` and return its index.
    fn add_static_plane(&mut self, nx: f64, ny: f64, nz: f64, offset: f64) -> PyResult<usize> {
        let n = some_or_value(binding_api::vec3(nx, ny, nz), "normal must be finite")?;
        some_or_value(
            binding_api::add_static_plane(&mut self.inner, n, offset),
            "normal must be non-zero and offset finite",
        )
    }

    /// Add a static height field from a NumPy (depth, width) array of heights
    /// (`x` along columns) spaced `spacing` apart from the min corner `origin`.
    fn add_static_heightfield(
        &mut self,
        heights: PyReadonlyArray2<f64>,
        spacing: f64,
        origin: (f64, f64, f64),
    ) -> PyResult<usize> {
        let a = heights.as_array();
        let (depth, width) = a.dim();
        let flat: Vec<f64> = a.iter().copied().collect();
        let o = some_or_value(
            binding_api::vec3(origin.0, origin.1, origin.2),
            "origin must be finite",
        )?;
        let (w, d) = match (u32::try_from(width), u32::try_from(depth)) {
            (Ok(w), Ok(d)) => (w, d),
            _ => return Err(PyValueError::new_err("height field too large")),
        };
        some_or_value(
            binding_api::add_static_heightfield(&mut self.inner, &flat, w, d, spacing, o),
            "height field needs at least 2 × 2 finite heights and a positive spacing",
        )
    }

    /// Add a static triangle mesh from a NumPy (N, 3) vertex array and a flat
    /// list of vertex indices (three per triangle).
    fn add_static_trimesh(
        &mut self,
        vertices: PyReadonlyArray2<f64>,
        indices: Vec<u32>,
    ) -> PyResult<usize> {
        let a = vertices.as_array();
        if a.dim().1 != 3 {
            return Err(PyValueError::new_err("vertices must have shape (N, 3)"));
        }
        let flat: Vec<f64> = a.iter().copied().collect();
        some_or_value(
            binding_api::add_static_trimesh(&mut self.inner, &flat, &indices),
            "mesh needs finite vertices and a non-empty index list of whole triangles within range",
        )
    }

    /// Remove static collider `index` (later colliders shift down by one).
    fn remove_static_collider(&mut self, index: usize) -> PyResult<()> {
        if binding_api::remove_static_collider(&mut self.inner, index) {
            Ok(())
        } else {
            Err(PyIndexError::new_err("static collider index out of range"))
        }
    }

    /// Number of static colliders.
    fn static_collider_count(&self) -> usize {
        self.inner.static_collider_count()
    }

    /// Add a ball-and-socket joint (anchors in body-local coordinates) and
    /// return its index.
    fn add_ball_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Arg,
        anchor_b: Vec3Arg,
    ) -> PyResult<usize> {
        let j = Joint::Ball(BallJoint::new(body_a, body_b, v3(anchor_a)?, v3(anchor_b)?));
        self.add_joint_checked(j)
    }

    /// Add a hinge joint (anchors and axes in body-local coordinates; axes
    /// non-zero) and return its index.
    fn add_hinge_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Arg,
        anchor_b: Vec3Arg,
        axis_a: Vec3Arg,
        axis_b: Vec3Arg,
    ) -> PyResult<usize> {
        let j = Joint::Hinge(HingeJoint::new(
            body_a,
            body_b,
            v3(anchor_a)?,
            v3(anchor_b)?,
            unit(axis_a)?,
            unit(axis_b)?,
        ));
        self.add_joint_checked(j)
    }

    /// Add a fixed joint; `relative_rotation` is `(x, y, z, w)` (normalised,
    /// non-zero). Returns its index.
    fn add_fixed_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Arg,
        anchor_b: Vec3Arg,
        relative_rotation: (f64, f64, f64, f64),
    ) -> PyResult<usize> {
        let (x, y, z, w) = relative_rotation;
        let q = some_or_value(
            binding_api::unit_quat(x, y, z, w),
            "rotation must be finite and non-zero",
        )?;
        let j = Joint::Fixed(FixedJoint::new(
            body_a,
            body_b,
            v3(anchor_a)?,
            v3(anchor_b)?,
            q,
        ));
        self.add_joint_checked(j)
    }

    /// Add a slider joint along `axis` (body A local, non-zero) and return its
    /// index.
    fn add_slider_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        axis: Vec3Arg,
        anchor_a: Vec3Arg,
        anchor_b: Vec3Arg,
    ) -> PyResult<usize> {
        let j = Joint::Slider(SliderJoint::new(
            body_a,
            body_b,
            unit(axis)?,
            v3(anchor_a)?,
            v3(anchor_b)?,
        ));
        self.add_joint_checked(j)
    }

    /// Add a spring (rest length and damping not negative, stiffness
    /// positive) and return its index.
    #[allow(clippy::too_many_arguments)]
    fn add_spring_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchor_a: Vec3Arg,
        anchor_b: Vec3Arg,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    ) -> PyResult<usize> {
        let rest = some_or_value(
            binding_api::non_negative(rest_length),
            "rest_length must be finite and not negative",
        )?;
        let k = some_or_value(
            binding_api::positive(stiffness),
            "stiffness must be finite and positive",
        )?;
        let c = some_or_value(
            binding_api::non_negative(damping),
            "damping must be finite and not negative",
        )?;
        let j = Joint::Spring(SpringJoint::new(
            body_a,
            body_b,
            v3(anchor_a)?,
            v3(anchor_b)?,
            rest,
            k,
            c,
        ));
        self.add_joint_checked(j)
    }

    /// Remove joint `index` (the last joint moves into `index`).
    fn remove_joint(&mut self, index: usize) -> PyResult<()> {
        if binding_api::remove_joint(&mut self.inner, index) {
            Ok(())
        } else {
            Err(PyIndexError::new_err("joint index out of range"))
        }
    }

    /// Number of joints.
    fn joint_count(&self) -> usize {
        self.inner.joint_count()
    }
}

/// A Python `(x, y, z)` tuple argument.
type Vec3Arg = (f64, f64, f64);

fn v3(v: Vec3Arg) -> PyResult<Vec3Fix> {
    some_or_value(binding_api::vec3(v.0, v.1, v.2), "vector must be finite")
}

fn unit(v: Vec3Arg) -> PyResult<Vec3Fix> {
    some_or_value(v3(v)?.try_normalize(), "axis must be non-zero")
}

fn some_or_value<T>(v: Option<T>, msg: &str) -> PyResult<T> {
    v.ok_or_else(|| PyValueError::new_err(msg.to_string()))
}

fn ok_or_value(ok: bool, msg: &str) -> PyResult<()> {
    if ok {
        Ok(())
    } else {
        Err(PyValueError::new_err(msg.to_string()))
    }
}

impl PyPhysicsWorld {
    fn check_body(&self, body_id: usize) -> PyResult<()> {
        if body_id < self.inner.bodies.len() {
            Ok(())
        } else {
            Err(PyIndexError::new_err("body_id out of range"))
        }
    }

    /// Add a joint, telling an unknown body (`IndexError`) from a joint
    /// between a body and itself (`ValueError`).
    fn add_joint_checked(&mut self, joint: Joint) -> PyResult<usize> {
        let (a, b) = joint.bodies();
        self.check_body(a)?;
        self.check_body(b)?;
        some_or_value(
            binding_api::add_joint(&mut self.inner, joint),
            "a joint needs two different bodies",
        )
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

#[allow(clippy::useless_conversion)]
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
    fn advance_frame(&mut self, py: Python<'_>, inputs: Vec<(u8, f64, f64, f64, u32)>) -> u64 {
        let frame_inputs: Vec<FrameInput> = inputs
            .iter()
            .map(|&(pid, mx, my, mz, act)| {
                FrameInput::new(pid)
                    .with_movement(Vec3Fix::from_f32(mx as f32, my as f32, mz as f32))
                    .with_actions(act)
            })
            .collect();

        let checksum = py.detach(|| {
            self.inner
                .advance_frame_with_applicator(&frame_inputs, &DefaultInputApplicator::default())
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
        self.inner
            .verify_checksum(frame, SimulationChecksum(remote_checksum))
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
        let mut data = vec![0.0f64; n * 3];
        for (i, body) in self.inner.world.bodies.iter().enumerate() {
            let (x, y, z) = body.position.to_f32();
            let base = i * 3;
            data[base] = x as f64;
            data[base + 1] = y as f64;
            data[base + 2] = z as f64;
        }
        data.into_pyarray(py)
            .reshape([n, 3])
            .expect("buffer length is n*3")
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
    py.detach(|| {
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
#[allow(clippy::too_many_arguments)]
fn encode_frame_input(
    player_id: u8,
    move_x: f64,
    move_y: f64,
    move_z: f64,
    actions: u32,
    aim_x: f64,
    aim_y: f64,
    aim_z: f64,
) -> Vec<u8> {
    let input = FrameInput::new(player_id)
        .with_movement(Vec3Fix::from_f32(
            move_x as f32,
            move_y as f32,
            move_z as f32,
        ))
        .with_actions(actions)
        .with_aim(Vec3Fix::from_f32(aim_x as f32, aim_y as f32, aim_z as f32));
    input.to_bytes().to_vec()
}

/// Decode a FrameInput from 20 bytes.
///
/// Returns: (player_id, move_x, move_y, move_z, actions, aim_x, aim_y, aim_z)
#[pyfunction]
#[allow(clippy::type_complexity)]
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
        mx as f64,
        my as f64,
        mz as f64,
        input.actions,
        ax as f64,
        ay as f64,
        az as f64,
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
