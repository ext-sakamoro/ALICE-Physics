//! WebAssembly Bindings for ALICE-Physics
//!
//! Browser-side deterministic physics via wasm-bindgen.
//!
//! # Ecosystem Position
//!
//! | Layer    | Technology        | Module      |
//! |----------|-------------------|-------------|
//! | Edge     | no_std + C FFI    | `ffi.rs`    |
//! | Desktop  | Unity/UE5 C FFI   | `ffi.rs`    |
//! | Server   | Native Rust       | `solver.rs` |
//! | Python   | PyO3 + NumPy      | `python.rs` |
//! | **Browser** | **wasm-bindgen** | **`wasm.rs`** |
//!
//! # Features
//!
//! - Full PhysicsWorld API (create, step, add/remove bodies)
//! - Batch position/velocity getters (Float64Array zero-copy)
//! - State serialization for rollback netcode
//! - Raycast queries (closest-hit, all-hits, any-hit)
//!
//! Author: Moroya Sakamoto

use wasm_bindgen::prelude::*;

use crate::math::{Fix128, Vec3Fix, QuatFix};
use crate::solver::{PhysicsWorld, PhysicsConfig, RigidBody, BodyType};
use crate::raycast::{Ray, ray_sphere};
use crate::collider::Sphere;

// ============================================================================
// WasmPhysicsWorld
// ============================================================================

/// Deterministic 128-bit fixed-point physics world for browsers.
///
/// All operations are bit-exact across platforms.
/// Use Float64Array batch APIs for best performance.
#[wasm_bindgen]
pub struct WasmPhysicsWorld {
    inner: PhysicsWorld,
}

#[wasm_bindgen]
impl WasmPhysicsWorld {
    /// Create a new physics world with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: PhysicsWorld::new(PhysicsConfig::default()),
        }
    }

    /// Create with custom gravity and substeps.
    #[wasm_bindgen(js_name = "withConfig")]
    pub fn with_config(gravity_x: f64, gravity_y: f64, gravity_z: f64, substeps: u32, iterations: u32) -> Self {
        let config = PhysicsConfig {
            substeps: substeps as usize,
            iterations: iterations as usize,
            gravity: Vec3Fix::new(
                Fix128::from_f64(gravity_x),
                Fix128::from_f64(gravity_y),
                Fix128::from_f64(gravity_z),
            ),
            damping: Fix128::from_ratio(99, 100),
        };
        Self {
            inner: PhysicsWorld::new(config),
        }
    }

    // ========================================================================
    // Body Management
    // ========================================================================

    /// Add a dynamic body at (x, y, z) with given mass. Returns body index.
    #[wasm_bindgen(js_name = "addDynamicBody")]
    pub fn add_dynamic_body(&mut self, x: f64, y: f64, z: f64, mass: f64) -> usize {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z)),
            Fix128::from_f64(mass),
        );
        self.inner.add_body(body)
    }

    /// Add a static (immovable) body at (x, y, z). Returns body index.
    #[wasm_bindgen(js_name = "addStaticBody")]
    pub fn add_static_body(&mut self, x: f64, y: f64, z: f64) -> usize {
        let body = RigidBody::new_static(
            Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z)),
        );
        self.inner.add_body(body)
    }

    /// Add a kinematic body at (x, y, z). Returns body index.
    #[wasm_bindgen(js_name = "addKinematicBody")]
    pub fn add_kinematic_body(&mut self, x: f64, y: f64, z: f64) -> usize {
        let body = RigidBody::new_kinematic(
            Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z)),
        );
        self.inner.add_body(body)
    }

    /// Number of bodies in the world.
    #[wasm_bindgen(js_name = "bodyCount")]
    pub fn body_count(&self) -> usize {
        self.inner.bodies.len()
    }

    // ========================================================================
    // Simulation
    // ========================================================================

    /// Step the simulation by dt seconds.
    pub fn step(&mut self, dt: f64) {
        self.inner.step(Fix128::from_f64(dt));
    }

    /// Step the simulation N times with fixed dt (batch stepping).
    #[wasm_bindgen(js_name = "stepN")]
    pub fn step_n(&mut self, dt: f64, steps: u32) {
        let dt_fix = Fix128::from_f64(dt);
        for _ in 0..steps {
            self.inner.step(dt_fix);
        }
    }

    // ========================================================================
    // Batch Getters (Float64Array for JS interop)
    // ========================================================================

    /// Get all body positions as flat Float64Array [x,y,z, x,y,z, ...].
    ///
    /// Returns a `Float64Array` of length `bodyCount * 3`.
    #[wasm_bindgen(js_name = "getPositions")]
    pub fn get_positions(&self) -> Vec<f64> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 3);
        for body in &self.inner.bodies {
            data.push(body.position.x.to_f64());
            data.push(body.position.y.to_f64());
            data.push(body.position.z.to_f64());
        }
        data
    }

    /// Get all body velocities as flat Float64Array [vx,vy,vz, ...].
    #[wasm_bindgen(js_name = "getVelocities")]
    pub fn get_velocities(&self) -> Vec<f64> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 3);
        for body in &self.inner.bodies {
            data.push(body.velocity.x.to_f64());
            data.push(body.velocity.y.to_f64());
            data.push(body.velocity.z.to_f64());
        }
        data
    }

    /// Get all body rotations as flat Float64Array [qx,qy,qz,qw, ...].
    #[wasm_bindgen(js_name = "getRotations")]
    pub fn get_rotations(&self) -> Vec<f64> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 4);
        for body in &self.inner.bodies {
            data.push(body.rotation.x.to_f64());
            data.push(body.rotation.y.to_f64());
            data.push(body.rotation.z.to_f64());
            data.push(body.rotation.w.to_f64());
        }
        data
    }

    /// Get full state as flat Float64Array [px,py,pz, vx,vy,vz, qx,qy,qz,qw, ...].
    ///
    /// 10 values per body.
    #[wasm_bindgen(js_name = "getStates")]
    pub fn get_states(&self) -> Vec<f64> {
        let n = self.inner.bodies.len();
        let mut data = Vec::with_capacity(n * 10);
        for body in &self.inner.bodies {
            data.push(body.position.x.to_f64());
            data.push(body.position.y.to_f64());
            data.push(body.position.z.to_f64());
            data.push(body.velocity.x.to_f64());
            data.push(body.velocity.y.to_f64());
            data.push(body.velocity.z.to_f64());
            data.push(body.rotation.x.to_f64());
            data.push(body.rotation.y.to_f64());
            data.push(body.rotation.z.to_f64());
            data.push(body.rotation.w.to_f64());
        }
        data
    }

    /// Get body type for a body (0=Dynamic, 1=Static, 2=Kinematic).
    #[wasm_bindgen(js_name = "getBodyType")]
    pub fn get_body_type(&self, body_id: usize) -> u8 {
        self.inner.bodies.get(body_id).map_or(255, |b| b.body_type as u8)
    }

    // ========================================================================
    // Per-Body Setters
    // ========================================================================

    /// Set a body's position.
    #[wasm_bindgen(js_name = "setPosition")]
    pub fn set_position(&mut self, body_id: usize, x: f64, y: f64, z: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.position = Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z));
        }
    }

    /// Set a body's velocity.
    #[wasm_bindgen(js_name = "setVelocity")]
    pub fn set_velocity(&mut self, body_id: usize, vx: f64, vy: f64, vz: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.velocity = Vec3Fix::new(Fix128::from_f64(vx), Fix128::from_f64(vy), Fix128::from_f64(vz));
        }
    }

    /// Apply impulse at center of mass.
    #[wasm_bindgen(js_name = "applyImpulse")]
    pub fn apply_impulse(&mut self, body_id: usize, ix: f64, iy: f64, iz: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.apply_impulse(Vec3Fix::new(Fix128::from_f64(ix), Fix128::from_f64(iy), Fix128::from_f64(iz)));
        }
    }

    /// Set a body's restitution (bounciness, 0.0-1.0).
    #[wasm_bindgen(js_name = "setRestitution")]
    pub fn set_restitution(&mut self, body_id: usize, restitution: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.restitution = Fix128::from_f64(restitution);
        }
    }

    /// Set a body's friction coefficient.
    #[wasm_bindgen(js_name = "setFriction")]
    pub fn set_friction(&mut self, body_id: usize, friction: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.friction = Fix128::from_f64(friction);
        }
    }

    /// Set a body's gravity scale.
    #[wasm_bindgen(js_name = "setGravityScale")]
    pub fn set_gravity_scale(&mut self, body_id: usize, scale: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.gravity_scale = Fix128::from_f64(scale);
        }
    }

    /// Set kinematic target position and rotation.
    #[wasm_bindgen(js_name = "setKinematicTarget")]
    pub fn set_kinematic_target(&mut self, body_id: usize, x: f64, y: f64, z: f64, qx: f64, qy: f64, qz: f64, qw: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.set_kinematic_target(
                Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z)),
                QuatFix {
                    x: Fix128::from_f64(qx),
                    y: Fix128::from_f64(qy),
                    z: Fix128::from_f64(qz),
                    w: Fix128::from_f64(qw),
                },
            );
        }
    }

    // ========================================================================
    // Batch Setters
    // ========================================================================

    /// Set velocities for all bodies from flat array [vx,vy,vz, ...].
    #[wasm_bindgen(js_name = "setVelocitiesBatch")]
    pub fn set_velocities_batch(&mut self, data: &[f64]) {
        let n = self.inner.bodies.len();
        if data.len() < n * 3 {
            return;
        }
        for (i, body) in self.inner.bodies.iter_mut().enumerate() {
            body.velocity = Vec3Fix::new(
                Fix128::from_f64(data[i * 3]),
                Fix128::from_f64(data[i * 3 + 1]),
                Fix128::from_f64(data[i * 3 + 2]),
            );
        }
    }

    /// Apply impulses in batch. Data is flat [body_id, ix, iy, iz, ...].
    #[wasm_bindgen(js_name = "applyImpulsesBatch")]
    pub fn apply_impulses_batch(&mut self, data: &[f64]) {
        let n_bodies = self.inner.bodies.len();
        for chunk in data.chunks_exact(4) {
            let body_id = chunk[0] as usize;
            if body_id >= n_bodies {
                continue;
            }
            let impulse = Vec3Fix::new(
                Fix128::from_f64(chunk[1]),
                Fix128::from_f64(chunk[2]),
                Fix128::from_f64(chunk[3]),
            );
            self.inner.bodies[body_id].apply_impulse(impulse);
        }
    }

    // ========================================================================
    // World Config
    // ========================================================================

    /// Set world gravity.
    #[wasm_bindgen(js_name = "setGravity")]
    pub fn set_gravity(&mut self, x: f64, y: f64, z: f64) {
        self.inner.config.gravity = Vec3Fix::new(
            Fix128::from_f64(x), Fix128::from_f64(y), Fix128::from_f64(z),
        );
    }

    /// Set number of substeps per frame.
    #[wasm_bindgen(js_name = "setSubsteps")]
    pub fn set_substeps(&mut self, substeps: u32) {
        self.inner.config.substeps = substeps as usize;
    }

    // ========================================================================
    // State Serialization (Rollback Netcode)
    // ========================================================================

    /// Serialize world state to bytes (for rollback/save).
    #[wasm_bindgen(js_name = "serializeState")]
    pub fn serialize_state(&self) -> Vec<u8> {
        self.inner.serialize_state()
    }

    /// Restore world state from bytes.
    #[wasm_bindgen(js_name = "deserializeState")]
    pub fn deserialize_state(&mut self, data: &[u8]) -> bool {
        self.inner.deserialize_state(data)
    }

    // ========================================================================
    // Raycast
    // ========================================================================

    /// Cast a ray from (ox,oy,oz) in direction (dx,dy,dz).
    ///
    /// Tests against all bodies as spheres of `body_radius`.
    /// Returns `[t, hit_x, hit_y, hit_z, normal_x, normal_y, normal_z, body_index]`
    /// or empty array if no hit.
    #[wasm_bindgen(js_name = "raycast")]
    pub fn raycast(
        &self,
        ox: f64, oy: f64, oz: f64,
        dx: f64, dy: f64, dz: f64,
        max_distance: f64,
        body_radius: f64,
    ) -> Vec<f64> {
        let origin = Vec3Fix::new(Fix128::from_f64(ox), Fix128::from_f64(oy), Fix128::from_f64(oz));
        let direction = Vec3Fix::new(Fix128::from_f64(dx), Fix128::from_f64(dy), Fix128::from_f64(dz));
        let dir_len = direction.length();
        if dir_len.is_zero() {
            return Vec::new();
        }
        let dir_norm = direction / dir_len;
        let ray = Ray::new(origin, dir_norm);
        let max_t = Fix128::from_f64(max_distance);
        let br = Fix128::from_f64(body_radius);

        let mut best_t = max_t;
        let mut result = Vec::new();

        for (i, body) in self.inner.bodies.iter().enumerate() {
            let expanded = Sphere::new(body.position, br);
            if let Some(hit) = ray_sphere(&ray, &expanded, best_t) {
                best_t = hit.t;
                result = vec![
                    hit.t.to_f64(),
                    hit.point.x.to_f64(), hit.point.y.to_f64(), hit.point.z.to_f64(),
                    hit.normal.x.to_f64(), hit.normal.y.to_f64(), hit.normal.z.to_f64(),
                    i as f64,
                ];
            }
        }

        result
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the library version string.
#[wasm_bindgen(js_name = "alicePhysicsVersion")]
pub fn alice_physics_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
