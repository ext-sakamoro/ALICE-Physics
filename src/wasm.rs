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

use crate::binding_api;
use crate::collider::Sphere;
use crate::joint::{BallJoint, FixedJoint, HingeJoint, Joint, SliderJoint, SpringJoint};
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::raycast::{ray_sphere, Ray};
use crate::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

#[cfg(test)]
use crate::solver::BodyType;

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
    /// The hands on it, from the first hold
    hands: Option<crate::holds::Holds>,
}

#[wasm_bindgen]
impl WasmPhysicsWorld {
    /// Create a new physics world with default configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: PhysicsWorld::new(PhysicsConfig::default()),
            hands: None,
        }
    }

    /// Create with custom gravity and substeps.
    #[wasm_bindgen(js_name = "withConfig")]
    pub fn with_config(
        gravity_x: f64,
        gravity_y: f64,
        gravity_z: f64,
        substeps: u32,
        iterations: u32,
    ) -> Self {
        let config = PhysicsConfig {
            substeps: substeps as usize,
            iterations: iterations as usize,
            gravity: Vec3Fix::new(
                Fix128::from_f64(gravity_x),
                Fix128::from_f64(gravity_y),
                Fix128::from_f64(gravity_z),
            ),
            // the per-second rate equal to the retention 0.99 per step this
            // binding used before the rate, at its 60 steps per second
            damping_rate: -Fix128::from_ratio(99, 100).ln() * Fix128::from_int(60),
            ..Default::default()
        };
        Self {
            inner: PhysicsWorld::new(config),
            hands: None,
        }
    }

    // ========================================================================
    // Body Management
    // ========================================================================

    /// Add a dynamic body at (x, y, z) with given mass. Returns body index.
    #[wasm_bindgen(js_name = "addDynamicBody")]
    pub fn add_dynamic_body(&mut self, x: f64, y: f64, z: f64, mass: f64) -> usize {
        let body = RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_f64(x),
                Fix128::from_f64(y),
                Fix128::from_f64(z),
            ),
            Fix128::from_f64(mass),
        );
        self.inner.add_body(body)
    }

    /// Add a static (immovable) body at (x, y, z). Returns body index.
    #[wasm_bindgen(js_name = "addStaticBody")]
    pub fn add_static_body(&mut self, x: f64, y: f64, z: f64) -> usize {
        let body = RigidBody::new_static(Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        ));
        self.inner.add_body(body)
    }

    /// Add a kinematic body at (x, y, z). Returns body index.
    #[wasm_bindgen(js_name = "addKinematicBody")]
    pub fn add_kinematic_body(&mut self, x: f64, y: f64, z: f64) -> usize {
        let body = RigidBody::new_kinematic(Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        ));
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
        self.inner.step_n(steps as usize, Fix128::from_f64(dt));
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
        self.inner
            .bodies
            .get(body_id)
            .map_or(255, |b| b.body_type as u8)
    }

    // ========================================================================
    // Per-Body Setters
    // ========================================================================

    /// Set a body's position.
    #[wasm_bindgen(js_name = "setPosition")]
    pub fn set_position(&mut self, body_id: usize, x: f64, y: f64, z: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.position = Vec3Fix::new(
                Fix128::from_f64(x),
                Fix128::from_f64(y),
                Fix128::from_f64(z),
            );
        }
    }

    /// Set a body's velocity.
    #[wasm_bindgen(js_name = "setVelocity")]
    pub fn set_velocity(&mut self, body_id: usize, vx: f64, vy: f64, vz: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.velocity = Vec3Fix::new(
                Fix128::from_f64(vx),
                Fix128::from_f64(vy),
                Fix128::from_f64(vz),
            );
        }
    }

    /// Set a body's orientation to the quaternion `(qx, qy, qz, qw)`
    /// (normalised in `Fix128`). `false` for an unknown body, a non-finite
    /// component or the zero quaternion.
    #[wasm_bindgen(js_name = "setRotation")]
    pub fn set_rotation(&mut self, body_id: usize, qx: f64, qy: f64, qz: f64, qw: f64) -> bool {
        binding_api::set_rotation(&mut self.inner, body_id, qx, qy, qz, qw)
    }

    /// Set a body's angular velocity (world frame, radians per second).
    /// `false` for an unknown body or a non-finite component.
    #[wasm_bindgen(js_name = "setAngularVelocity")]
    pub fn set_angular_velocity(&mut self, body_id: usize, wx: f64, wy: f64, wz: f64) -> bool {
        binding_api::set_angular_velocity(&mut self.inner, body_id, wx, wy, wz)
    }

    /// Apply impulse at center of mass.
    #[wasm_bindgen(js_name = "applyImpulse")]
    pub fn apply_impulse(&mut self, body_id: usize, ix: f64, iy: f64, iz: f64) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.apply_impulse(Vec3Fix::new(
                Fix128::from_f64(ix),
                Fix128::from_f64(iy),
                Fix128::from_f64(iz),
            ));
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
    pub fn set_kinematic_target(
        &mut self,
        body_id: usize,
        x: f64,
        y: f64,
        z: f64,
        qx: f64,
        qy: f64,
        qz: f64,
        qw: f64,
    ) {
        if let Some(body) = self.inner.bodies.get_mut(body_id) {
            body.set_kinematic_target(
                Vec3Fix::new(
                    Fix128::from_f64(x),
                    Fix128::from_f64(y),
                    Fix128::from_f64(z),
                ),
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
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
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

    /// The whole world as bytes ([`PhysicsWorld::snapshot_world`]): bodies,
    /// joints, constraints, static colliders, materials and the rest, under the
    /// header's semantics and law identifiers. `serializeState` keeps body
    /// motion only; this is the one to rewind a scene with joints.
    #[wasm_bindgen(js_name = "snapshotWorld")]
    pub fn snapshot_world(&self) -> Vec<u8> {
        self.inner.snapshot_world()
    }

    /// Restore the whole world from [`Self::snapshot_world`]'s bytes. A refused
    /// blob throws an `Error` whose `name` is the [`WorldSnapshotError`] variant
    /// (`"SemanticsMismatch"` for a blob another build wrote, `"ChecksumMismatch"`,
    /// `"Truncated"`, ...) and leaves the world unchanged.
    ///
    /// [`WorldSnapshotError`]: crate::solver::WorldSnapshotError
    #[wasm_bindgen(js_name = "restoreWorld")]
    pub fn restore_world(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.inner.restore_world(data).map_err(|e| {
            let err = js_sys::Error::new(&e.to_string());
            err.set_name(e.kind());
            err.into()
        })
    }

    /// A blow from a point ([`PhysicsWorld::apply_radial_impulse`]), a negative strength pulling in. How many bodies
    /// it reached, `undefined` for a value that cannot be used.
    #[wasm_bindgen(js_name = "applyRadialImpulse")]
    pub fn apply_radial_impulse(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        radius: f64,
        strength: f64,
    ) -> Option<u32> {
        let center = binding_api::vec3(x, y, z)?;
        binding_api::radial_impulse(&mut self.inner, center, radius, strength).map(|n| n as u32)
    }

    fn hands(&mut self) -> crate::holds::Holds {
        if let Some(h) = self.hands {
            return h;
        }
        let h = crate::holds::Holds::new(&mut self.inner);
        self.hands = Some(h);
        h
    }

    /// Hold a body at a place, reaching it at `omega` radians per second ([`crate::holds::Holds::hold`]). The first
    /// hold adds one static body to the world. `false` when it cannot be held.
    #[wasm_bindgen(js_name = "holdBody")]
    pub fn hold_body(&mut self, body: u32, x: f64, y: f64, z: f64, omega: f64) -> bool {
        let (Some(target), Some(omega)) =
            (binding_api::vec3(x, y, z), binding_api::positive(omega))
        else {
            return false;
        };
        let hands = self.hands();
        hands
            .hold(&mut self.inner, body as usize, target, omega)
            .is_ok()
    }

    /// Move the place a held body is going to. `false` when it is not held.
    #[wasm_bindgen(js_name = "moveHold")]
    pub fn move_hold(&mut self, body: u32, x: f64, y: f64, z: f64) -> bool {
        let Some(target) = binding_api::vec3(x, y, z) else {
            return false;
        };
        let hands = self.hands();
        hands
            .move_to(&mut self.inner, body as usize, target)
            .is_ok()
    }

    /// Let go of a body: it keeps its velocity and its weight. `false` when it is not held.
    #[wasm_bindgen(js_name = "releaseHold")]
    pub fn release_hold(&mut self, body: u32) -> bool {
        let hands = self.hands();
        hands.release(&mut self.inner, body as usize).is_ok()
    }

    /// Hold each body at the place at its position in `places` (x, y, z each): all, or `false` and none.
    #[wasm_bindgen(js_name = "formBodies")]
    pub fn form_bodies(&mut self, bodies: &[u32], places: &[f64], omega: f64) -> bool {
        let Some(omega) = binding_api::positive(omega) else {
            return false;
        };
        if places.len() != bodies.len() * 3 {
            return false;
        }
        let mut targets = Vec::with_capacity(bodies.len());
        for p in places.chunks_exact(3) {
            let Some(t) = binding_api::vec3(p[0], p[1], p[2]) else {
                return false;
            };
            targets.push(t);
        }
        let list: Vec<usize> = bodies.iter().map(|b| *b as usize).collect();
        let hands = self.hands();
        hands.form(&mut self.inner, &list, &targets, omega).is_ok()
    }

    /// Let go of everything held. How many.
    #[wasm_bindgen(js_name = "releaseHolds")]
    pub fn release_holds(&mut self) -> u32 {
        let hands = self.hands();
        hands.release_all(&mut self.inner) as u32
    }

    /// SHA-256 of the world snapshot ([`PhysicsWorld::state_digest`], 32 bytes):
    /// equal for two worlds exactly when their state and rules are equal.
    #[wasm_bindgen(js_name = "stateDigest")]
    pub fn state_digest(&self) -> Vec<u8> {
        self.inner.state_digest().to_vec()
    }

    /// The world's law identifier under this build's semantics
    /// ([`PhysicsWorld::law_id`] with [`crate::PHYSICS_SEMANTICS_ID`], 32 bytes).
    #[wasm_bindgen(js_name = "lawId")]
    pub fn law_id(&self) -> Vec<u8> {
        self.inner.law_id(&crate::PHYSICS_SEMANTICS_ID).to_vec()
    }

    // ========================================================================
    // Raycast
    // ========================================================================

    /// Cast a ray from (ox,oy,oz) in direction (dx,dy,dz).
    ///
    // LIMITATION(COV-ENGINE-070): Tests against all bodies as spheres of `body_radius`.
    /// Tests against all bodies as spheres of `body_radius`.
    /// Returns `[t, hit_x, hit_y, hit_z, normal_x, normal_y, normal_z, body_index]`
    /// or empty array if no hit.
    #[wasm_bindgen(js_name = "raycast")]
    pub fn raycast(
        &self,
        ox: f64,
        oy: f64,
        oz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
        max_distance: f64,
        body_radius: f64,
    ) -> Vec<f64> {
        let origin = Vec3Fix::new(
            Fix128::from_f64(ox),
            Fix128::from_f64(oy),
            Fix128::from_f64(oz),
        );
        let direction = Vec3Fix::new(
            Fix128::from_f64(dx),
            Fix128::from_f64(dy),
            Fix128::from_f64(dz),
        );
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
                    hit.point.x.to_f64(),
                    hit.point.y.to_f64(),
                    hit.point.z.to_f64(),
                    hit.normal.x.to_f64(),
                    hit.normal.y.to_f64(),
                    hit.normal.z.to_f64(),
                    i as f64,
                ];
            }
        }

        result
    }

    // ------------------------------------------------------------------
    // World queries against the collided geometry (bodies, their shapes,
    // static colliders, SDF colliders) and body observation. Vectors are
    // `[x, y, z]`; `excludeBody` (or `undefined`) is a body the query
    // ignores. A hit is `[t, px, py, pz, nx, ny, nz, kind, index, body]`
    // (kind 0 body, 1 static collider, 2 SDF collider; body -1 for none),
    // every value the Rust result through `Fix128::to_f64`. No hit (also a
    // zero direction, `maxT <= 0` or a negative radius, as the Rust API) and
    // a refused argument (non-finite, not 3 values, `excludeBody` not a body)
    // give an empty array.
    // ------------------------------------------------------------------

    /// The nearest hit of a ray with the world's geometry
    /// (`PhysicsWorld::cast_ray`).
    #[wasm_bindgen(js_name = "castRay")]
    pub fn cast_ray(
        &self,
        origin: &[f64],
        direction: &[f64],
        max_t: f64,
        exclude_body: Option<u32>,
    ) -> Vec<f64> {
        let w = &self.inner;
        let (Some(o), Some(d), Some(m), Some(f)) = (
            binding_api::vec3_slice(origin),
            binding_api::vec3_slice(direction),
            binding_api::finite(max_t),
            binding_api::query_filter(w, exclude_body.map(|b| b as usize)),
        ) else {
            return Vec::new();
        };
        hit_array(binding_api::cast_ray(w, o, d, m, &f))
    }

    /// The nearest collider a sphere of `radius` touches when its centre
    /// moves along `direction` for at most `max_t`
    /// (`PhysicsWorld::cast_sphere`).
    #[wasm_bindgen(js_name = "castSphere")]
    pub fn cast_sphere(
        &self,
        center: &[f64],
        radius: f64,
        direction: &[f64],
        max_t: f64,
        exclude_body: Option<u32>,
    ) -> Vec<f64> {
        let w = &self.inner;
        let (Some(c), Some(r), Some(d), Some(m), Some(f)) = (
            binding_api::vec3_slice(center),
            binding_api::finite(radius),
            binding_api::vec3_slice(direction),
            binding_api::finite(max_t),
            binding_api::query_filter(w, exclude_body.map(|b| b as usize)),
        ) else {
            return Vec::new();
        };
        hit_array(binding_api::cast_sphere(w, c, r, d, m, &f))
    }

    /// The nearest collider a capsule (segment `a`-`b` grown by `radius`)
    /// touches when it moves along `direction` for at most `max_t`
    /// (`PhysicsWorld::cast_capsule`).
    #[wasm_bindgen(js_name = "castCapsule")]
    pub fn cast_capsule(
        &self,
        a: &[f64],
        b: &[f64],
        radius: f64,
        direction: &[f64],
        max_t: f64,
        exclude_body: Option<u32>,
    ) -> Vec<f64> {
        let w = &self.inner;
        let (Some(a), Some(b), Some(r), Some(d), Some(m), Some(f)) = (
            binding_api::vec3_slice(a),
            binding_api::vec3_slice(b),
            binding_api::finite(radius),
            binding_api::vec3_slice(direction),
            binding_api::finite(max_t),
            binding_api::query_filter(w, exclude_body.map(|b| b as usize)),
        ) else {
            return Vec::new();
        };
        hit_array(binding_api::cast_capsule(w, a, b, r, d, m, &f))
    }

    /// Every collider a sphere of `radius` about `center` overlaps
    /// (`PhysicsWorld::overlap_sphere`), as flat `[kind, index, ...]` sorted
    /// by kind then index.
    #[wasm_bindgen(js_name = "overlapSphere")]
    pub fn overlap_sphere(
        &self,
        center: &[f64],
        radius: f64,
        exclude_body: Option<u32>,
    ) -> Vec<f64> {
        let w = &self.inner;
        let (Some(c), Some(r), Some(f)) = (
            binding_api::vec3_slice(center),
            binding_api::finite(radius),
            binding_api::query_filter(w, exclude_body.map(|b| b as usize)),
        ) else {
            return Vec::new();
        };
        binding_api::overlap_sphere(w, c, r, &f)
            .into_iter()
            .flat_map(|(kind, index)| [f64::from(kind), index as f64])
            .collect()
    }

    /// Observe one body (`PhysicsWorld::observe_body`):
    /// `[px, py, pz, vx, vy, vz, qx, qy, qz, qw, wx, wy, wz, sleeping,
    /// inContact]` (flags 0 / 1), or an empty array for an unknown body.
    #[wasm_bindgen(js_name = "observeBody")]
    pub fn observe_body(&self, body_id: usize) -> Vec<f64> {
        let Some(o) = binding_api::observe_body(&self.inner, body_id) else {
            return Vec::new();
        };
        let mut out = Vec::with_capacity(15);
        out.extend_from_slice(&o.position);
        out.extend_from_slice(&o.velocity);
        out.extend_from_slice(&o.rotation);
        out.extend_from_slice(&o.angular_velocity);
        out.push(f64::from(u8::from(o.sleeping)));
        out.push(f64::from(u8::from(o.in_contact)));
        out
    }

    /// The contacts of the last step, 10 numbers each: `[bodyA, bodyB, kind, speed, px,
    /// py, pz, nx, ny, nz]`. `kind` is 0 begin, 1 persist, 2 end, 3 a static collider, for
    /// which `bodyB` is `-(collider index) - 1`. `speed` is the speed of approach in m/s.
    #[wasm_bindgen(js_name = "getContactEvents")]
    pub fn get_contact_events(&self) -> Vec<f64> {
        binding_api::contact_events(&self.inner)
    }

    // ------------------------------------------------------------------
    // Collision radius, shapes, static colliders, joints. Indices come back
    // as `number`, or `undefined` when an argument is refused (see
    // `binding_api` for the checks); setters return `false` instead.
    // ------------------------------------------------------------------

    /// Set a body's collision sphere radius (finite and positive).
    #[wasm_bindgen(js_name = "setCollisionRadius")]
    pub fn set_collision_radius(&mut self, body_id: usize, radius: f64) -> bool {
        binding_api::set_collision_radius(&mut self.inner, body_id, radius)
    }

    /// Lock world axes of a body: `bits` 1, 2, 4 lock the x, y, z position and 8, 16, 32 the
    /// x, y, z rotation (0 unlocks). A planar game locks z and the x and y rotations (52).
    /// `false` for an unknown body or a bit above 32.
    #[wasm_bindgen(js_name = "setBodyAxisLock")]
    pub fn set_body_axis_lock(&mut self, body_id: usize, bits: u32) -> bool {
        binding_api::set_body_axis_lock(&mut self.inner, body_id, bits)
    }

    /// The axis-lock bits of a body (see `setBodyAxisLock`), `undefined` for an unknown body.
    #[wasm_bindgen(js_name = "bodyAxisLock")]
    pub fn body_axis_lock(&self, body_id: usize) -> Option<u32> {
        binding_api::body_axis_lock(&self.inner, body_id)
    }

    /// Register a contact material with `friction` and `restitution` (finite, not negative) and
    /// return its id (`undefined` for a bad value). A body given the id with `setBodyMaterial`
    /// meets others with these values instead of its own.
    #[wasm_bindgen(js_name = "registerMaterial")]
    pub fn register_material(&mut self, friction: f64, restitution: f64) -> Option<u32> {
        binding_api::register_material(&mut self.inner, friction, restitution)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Set a body's material id (0 to 65535); `false` for an unknown body or an id out of range.
    /// Magnets (`addMagnet`) pull only the bodies of the material they name.
    #[wasm_bindgen(js_name = "setBodyMaterial")]
    pub fn set_body_material(&mut self, body_id: usize, material: u32) -> bool {
        binding_api::set_body_material(&mut self.inner, body_id, material)
    }

    /// Add a permanent magnet: an ideal dipole at (x, y, z) with moment (mx, my, mz) in A·m² and
    /// its own `radius`, pulling the bodies of `material`, a soft ferromagnet of effective
    /// susceptibility `chi_eff`, saturation magnetisation `saturation` (A/m) and `density`
    /// (kg/m³). Returns the field index, or `undefined` for a bad value or a material another
    /// magnet names with different properties.
    #[wasm_bindgen(js_name = "addMagnet")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export cannot take a struct from JS cheaply; position, moment, radius and the material's properties are flat numbers"
    )]
    pub fn add_magnet(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        mx: f64,
        my: f64,
        mz: f64,
        radius: f64,
        material: u32,
        chi_eff: f64,
        saturation: f64,
        density: f64,
    ) -> Option<u32> {
        let m = binding_api::MagnetMaterial {
            material,
            chi_eff,
            saturation,
            density,
        };
        binding_api::add_magnet(
            &mut self.inner,
            binding_api::vec3(x, y, z)?,
            binding_api::vec3(mx, my, mz)?,
            radius,
            &m,
        )
        .and_then(|i| u32::try_from(i).ok())
    }

    /// Add a wind zone: inside the box (`min_*` to `max_*`) a body feels the drag of a wind along
    /// `dir_*` (normalised) of `base_speed` plus `gust_amplitude`·sin(2π·`gust_hz`·t) m/s, with
    /// `air_density` (kg/m³), `drag_coefficient` and reference `area` (m²). The zone is part of the
    /// world: `snapshotWorld` keeps its clock, and `restoreWorld` needs the same zones added in the
    /// same order. Returns its index, or `undefined` for a bad value.
    #[cfg(feature = "std")]
    #[wasm_bindgen(js_name = "addWindZone")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export cannot take a struct from JS cheaply; the box, the direction and the wind's values are flat numbers"
    )]
    pub fn add_wind_zone(
        &mut self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
        dir_x: f64,
        dir_y: f64,
        dir_z: f64,
        base_speed: f64,
        gust_amplitude: f64,
        gust_hz: f64,
        air_density: f64,
        drag_coefficient: f64,
        area: f64,
    ) -> Option<u32> {
        let args = binding_api::WindZoneArgs {
            min: binding_api::vec3(min_x, min_y, min_z)?,
            max: binding_api::vec3(max_x, max_y, max_z)?,
            direction: binding_api::vec3(dir_x, dir_y, dir_z)?,
            base_speed,
            gust_amplitude,
            gust_hz,
            air_density,
            drag_coefficient,
            area,
        };
        binding_api::add_wind_zone(&mut self.inner, &args).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a box of fluid (`min_*` to `max_*`, surface at `max_y`): a body touching it feels the
    /// lift of the volume it displaces, from its collision radius, `density` (kg/m³) and `gravity`
    /// (m/s², upward lift), and the drag `drag_linear`·v + `drag_quadratic`·v·|v|. The zone is part
    /// of the world: `snapshotWorld` keeps it, and `restoreWorld` needs the same zones added in the
    /// same order. Returns its index, or `undefined` for a bad value.
    #[cfg(feature = "std")]
    #[wasm_bindgen(js_name = "addBuoyancyZone")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export cannot take a struct from JS cheaply; the box and the fluid's values are flat numbers"
    )]
    pub fn add_buoyancy_zone(
        &mut self,
        min_x: f64,
        min_y: f64,
        min_z: f64,
        max_x: f64,
        max_y: f64,
        max_z: f64,
        density: f64,
        gravity: f64,
        drag_linear: f64,
        drag_quadratic: f64,
    ) -> Option<u32> {
        let args = binding_api::BuoyancyZoneArgs {
            min: binding_api::vec3(min_x, min_y, min_z)?,
            max: binding_api::vec3(max_x, max_y, max_z)?,
            density,
            gravity,
            drag_linear,
            drag_quadratic,
        };
        binding_api::add_buoyancy_zone(&mut self.inner, &args).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a rope from `start` to `end` (`[x, y, z]`) of `segments` segments and total `mass`,
    /// falling under the world's gravity, keeping `damping` of its velocity per frame (1 keeps it
    /// all). `start_body` and `end_body` tie an end to a body at `start_local` / `end_local` in the
    /// body's frame: the rope then pulls that body and is pulled by it. The rope is part of the
    /// world: `snapshotWorld` keeps it, and `restoreWorld` needs the same ropes added in the same
    /// order. Returns its index, or `undefined` for a bad value.
    #[cfg(feature = "std")]
    #[wasm_bindgen(js_name = "addRope")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export cannot take a struct from JS cheaply; the ends and their ties are what a rope is"
    )]
    pub fn add_rope(
        &mut self,
        start: &[f64],
        end: &[f64],
        segments: u32,
        mass: f64,
        damping: f64,
        start_body: Option<u32>,
        start_local: &[f64],
        end_body: Option<u32>,
        end_local: &[f64],
    ) -> Option<u32> {
        let tie = |body: Option<u32>, local: &[f64]| match body {
            None => Some(None),
            Some(b) => Some(Some(binding_api::RopeTie {
                body: b as usize,
                local: binding_api::vec3_slice(local)?,
            })),
        };
        let args = binding_api::RopeArgs {
            start: binding_api::vec3_slice(start)?,
            end: binding_api::vec3_slice(end)?,
            segments,
            mass,
            damping,
            start_tie: tie(start_body, start_local)?,
            end_tie: tie(end_body, end_local)?,
        };
        binding_api::add_rope(&mut self.inner, &args).and_then(|i| u32::try_from(i).ok())
    }

    /// The particles of the rope at participant `index` as `[x, y, z, ...]`, or `undefined` when
    /// that participant is not a rope.
    #[cfg(feature = "std")]
    #[wasm_bindgen(js_name = "ropePositions")]
    pub fn rope_positions(&self, index: u32) -> Option<Vec<f64>> {
        let particles = binding_api::rope_particles(&self.inner, index as usize)?;
        Some(
            particles
                .iter()
                .flat_map(|(p, _)| [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()])
                .collect(),
        )
    }

    /// Drop a body's own collision radius (it falls back to the world default).
    #[wasm_bindgen(js_name = "clearCollisionRadius")]
    pub fn clear_collision_radius(&mut self, body_id: usize) -> bool {
        binding_api::clear_collision_radius(&mut self.inner, body_id)
    }

    /// Add a dynamic body with a collision shape (`kind` and sizes as in the
    /// C ABI's `AlicePhysicsShape`: 0 box, 1 cylinder, 2 cone, 3 ellipsoid,
    /// 4 wedge, 5 torus); mass and inertia come from `density`.
    #[wasm_bindgen(js_name = "addShapedBody")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export cannot take a struct from JS cheaply; kind, three sizes, density and xyz position are flat numbers"
    )]
    pub fn add_shaped_body(
        &mut self,
        kind: u32,
        a: f64,
        b: f64,
        c: f64,
        density: f64,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<u32> {
        let shape = binding_api::shape(kind, a, b, c)?;
        let p = binding_api::vec3(x, y, z)?;
        binding_api::add_shaped_body(&mut self.inner, shape, density, p)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Give an existing body a collision shape (its mass is unchanged).
    #[wasm_bindgen(js_name = "setBodyShape")]
    pub fn set_body_shape(&mut self, body_id: usize, kind: u32, a: f64, b: f64, c: f64) -> bool {
        match binding_api::shape(kind, a, b, c) {
            Some(s) => binding_api::set_body_shape(&mut self.inner, body_id, s),
            None => false,
        }
    }

    /// Add the static plane `normal · p = offset`.
    #[wasm_bindgen(js_name = "addStaticPlane")]
    pub fn add_static_plane(&mut self, nx: f64, ny: f64, nz: f64, offset: f64) -> Option<u32> {
        let n = binding_api::vec3(nx, ny, nz)?;
        binding_api::add_static_plane(&mut self.inner, n, offset)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Add a static height field (`width × depth` heights, row-major, `x`
    /// fastest) from the min corner `(ox, oy, oz)`.
    #[wasm_bindgen(js_name = "addStaticHeightField")]
    #[allow(
        clippy::too_many_arguments,
        reason = "wasm-bindgen export takes heights, grid size, spacing and origin xyz as flat JS numbers instead of a struct"
    )]
    pub fn add_static_heightfield(
        &mut self,
        heights: &[f64],
        width: u32,
        depth: u32,
        spacing: f64,
        ox: f64,
        oy: f64,
        oz: f64,
    ) -> Option<u32> {
        let o = binding_api::vec3(ox, oy, oz)?;
        binding_api::add_static_heightfield(&mut self.inner, heights, width, depth, spacing, o)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Add a static triangle mesh (`vertices` as `x, y, z` triples, three
    /// indices per triangle).
    #[wasm_bindgen(js_name = "addStaticTriMesh")]
    pub fn add_static_trimesh(&mut self, vertices: &[f64], indices: &[u32]) -> Option<u32> {
        binding_api::add_static_trimesh(&mut self.inner, vertices, indices)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Remove static collider `index` (later colliders shift down by one).
    #[wasm_bindgen(js_name = "removeStaticCollider")]
    pub fn remove_static_collider(&mut self, index: usize) -> bool {
        binding_api::remove_static_collider(&mut self.inner, index)
    }

    /// Number of static colliders.
    #[wasm_bindgen(js_name = "staticColliderCount")]
    pub fn static_collider_count(&self) -> usize {
        self.inner.static_collider_count()
    }

    /// Add a ball-and-socket joint. `anchors` = `[ax, ay, az, bx, by, bz]`
    /// (body-local).
    #[wasm_bindgen(js_name = "addBallJoint")]
    pub fn add_ball_joint(&mut self, body_a: usize, body_b: usize, anchors: &[f64]) -> Option<u32> {
        let (aa, ab) = binding_api::two_vec3(anchors)?;
        let j = Joint::Ball(BallJoint::new(body_a, body_b, aa, ab));
        binding_api::add_joint(&mut self.inner, j).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a hinge joint. `anchors` = `[ax, ay, az, bx, by, bz]`, `axes` =
    /// `[axis_a…, axis_b…]` (body-local, non-zero).
    #[wasm_bindgen(js_name = "addHingeJoint")]
    pub fn add_hinge_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchors: &[f64],
        axes: &[f64],
    ) -> Option<u32> {
        let (aa, ab) = binding_api::two_vec3(anchors)?;
        let (xa, xb) = binding_api::two_vec3(axes)?;
        let j = Joint::Hinge(HingeJoint::new(
            body_a,
            body_b,
            aa,
            ab,
            xa.try_normalize()?,
            xb.try_normalize()?,
        ));
        binding_api::add_joint(&mut self.inner, j).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a fixed joint. `anchors` = `[ax, ay, az, bx, by, bz]`,
    /// `relative_rotation` = `[x, y, z, w]` (normalised, non-zero).
    #[wasm_bindgen(js_name = "addFixedJoint")]
    pub fn add_fixed_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchors: &[f64],
        relative_rotation: &[f64],
    ) -> Option<u32> {
        let (aa, ab) = binding_api::two_vec3(anchors)?;
        let [x, y, z, w] = <[f64; 4]>::try_from(relative_rotation).ok()?;
        let j = Joint::Fixed(FixedJoint::new(
            body_a,
            body_b,
            aa,
            ab,
            binding_api::unit_quat(x, y, z, w)?,
        ));
        binding_api::add_joint(&mut self.inner, j).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a slider joint along `axis` = `[x, y, z]` (body A local, non-zero).
    /// `anchors` = `[ax, ay, az, bx, by, bz]`.
    #[wasm_bindgen(js_name = "addSliderJoint")]
    pub fn add_slider_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        axis: &[f64],
        anchors: &[f64],
    ) -> Option<u32> {
        let [x, y, z] = <[f64; 3]>::try_from(axis).ok()?;
        let (aa, ab) = binding_api::two_vec3(anchors)?;
        let j = Joint::Slider(SliderJoint::new(
            body_a,
            body_b,
            binding_api::vec3(x, y, z)?.try_normalize()?,
            aa,
            ab,
        ));
        binding_api::add_joint(&mut self.inner, j).and_then(|i| u32::try_from(i).ok())
    }

    /// Add a spring. `anchors` = `[ax, ay, az, bx, by, bz]`; rest length and
    /// damping not negative, stiffness positive.
    #[wasm_bindgen(js_name = "addSpringJoint")]
    pub fn add_spring_joint(
        &mut self,
        body_a: usize,
        body_b: usize,
        anchors: &[f64],
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    ) -> Option<u32> {
        let (aa, ab) = binding_api::two_vec3(anchors)?;
        let j = Joint::Spring(SpringJoint::new(
            body_a,
            body_b,
            aa,
            ab,
            binding_api::non_negative(rest_length)?,
            binding_api::positive(stiffness)?,
            binding_api::non_negative(damping)?,
        ));
        binding_api::add_joint(&mut self.inner, j).and_then(|i| u32::try_from(i).ok())
    }

    /// Remove joint `index` (the last joint moves into `index`).
    #[wasm_bindgen(js_name = "removeJoint")]
    pub fn remove_joint(&mut self, index: usize) -> bool {
        binding_api::remove_joint(&mut self.inner, index)
    }

    /// Number of joints.
    #[wasm_bindgen(js_name = "jointCount")]
    pub fn joint_count(&self) -> usize {
        self.inner.joint_count()
    }

    // ---- cloths / soft letters ([`crate::cloth::Cloth::from_outline`]) ----

    /// Add a cloth cut from a planar outline (a soft letter). `contours` is the flat list of plane
    /// coordinates `[x0, y0, x1, y1, …]` of every contour in turn and `contour_lens`
    /// the point count of each; the plane is `origin + u·x + v·y` (`origin`, `u`, `v`
    /// as 9 numbers in `plane`). `tether_k` / `tether_c` are the rest tether's
    /// stiffness and damping (both 0: no tether). Returns the cloth index, or -1 when
    /// the description is refused (empty outline, spacing or mass not positive, no
    /// triangle kept).
    #[wasm_bindgen(js_name = "addClothOutline")]
    #[allow(
        clippy::too_many_arguments,
        reason = "the outline, lattice and cloth configuration arrive as plain numbers from JS"
    )]
    pub fn add_cloth_outline(
        &mut self,
        plane: &[f64],
        contours: &[f64],
        contour_lens: &[u32],
        spacing: f64,
        mass_per_particle: f64,
        tether_k: f64,
        tether_c: f64,
        iterations: u32,
        substeps: u32,
        damping: f64,
        stretch_compliance: f64,
        bend_compliance: f64,
    ) -> i32 {
        use crate::cloth::{Cloth, ClothConfig, OutlineDesc};
        if plane.len() != 9 || contours.len() % 2 != 0 {
            return -1;
        }
        let total: usize = contour_lens.iter().map(|&n| n as usize).sum();
        if total * 2 != contours.len() {
            return -1;
        }
        let v3 = |k: usize| {
            Vec3Fix::new(
                Fix128::from_f64(plane[k]),
                Fix128::from_f64(plane[k + 1]),
                Fix128::from_f64(plane[k + 2]),
            )
        };
        let mut polys = Vec::with_capacity(contour_lens.len());
        let mut at = 0usize;
        for &n in contour_lens {
            let n = n as usize;
            let mut poly = Vec::with_capacity(n);
            for k in 0..n {
                poly.push([
                    Fix128::from_f64(contours[(at + k) * 2]),
                    Fix128::from_f64(contours[(at + k) * 2 + 1]),
                ]);
            }
            at += n;
            polys.push(poly);
        }
        let tether = (tether_k != 0.0 || tether_c != 0.0)
            .then(|| (Fix128::from_f64(tether_k), Fix128::from_f64(tether_c)));
        let desc = OutlineDesc {
            origin: v3(0),
            u: v3(3),
            v: v3(6),
            contours: polys,
            spacing: Fix128::from_f64(spacing),
            mass_per_particle: Fix128::from_f64(mass_per_particle),
            config: ClothConfig {
                iterations: iterations as usize,
                substeps: substeps.max(1) as usize,
                gravity: Vec3Fix::ZERO,
                damping: Fix128::from_f64(damping),
                stretch_compliance: Fix128::from_f64(stretch_compliance),
                bend_compliance: Fix128::from_f64(bend_compliance),
                ..ClothConfig::default()
            },
            tether,
        };
        match Cloth::from_outline(&desc) {
            Ok(cloth) => i32::try_from(self.inner.add_cloth(cloth)).unwrap_or(-1),
            Err(_) => -1,
        }
    }

    /// Number of cloths.
    #[wasm_bindgen(js_name = "clothCount")]
    pub fn cloth_count(&self) -> usize {
        self.inner.cloths.len()
    }

    /// Number of particles of cloth `body` (0 when there is no such cloth).
    #[wasm_bindgen(js_name = "clothParticleCount")]
    pub fn cloth_particle_count(&self, body: usize) -> usize {
        self.inner
            .cloths
            .get(body)
            .map_or(0, crate::cloth::Cloth::particle_count)
    }

    /// Particle positions of cloth `body` as `[x, y, z, …]` (empty when none).
    #[wasm_bindgen(js_name = "clothPositions")]
    pub fn cloth_positions(&self, body: usize) -> Vec<f64> {
        self.inner.cloths.get(body).map_or_else(Vec::new, |c| {
            c.positions
                .iter()
                .flat_map(|p| [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()])
                .collect()
        })
    }

    /// Rest positions of cloth `body` as `[x, y, z, …]`.
    #[wasm_bindgen(js_name = "clothRestPositions")]
    pub fn cloth_rest_positions(&self, body: usize) -> Vec<f64> {
        self.inner.cloths.get(body).map_or_else(Vec::new, |c| {
            c.rest_positions()
                .iter()
                .flat_map(|p| [p.x.to_f64(), p.y.to_f64(), p.z.to_f64()])
                .collect()
        })
    }

    /// Triangles of cloth `body` as `[i0, i1, i2, …]`.
    #[wasm_bindgen(js_name = "clothTriangles")]
    pub fn cloth_triangles(&self, body: usize) -> Vec<u32> {
        self.inner.cloths.get(body).map_or_else(Vec::new, |c| {
            c.triangles
                .iter()
                .flat_map(|t| t.iter().map(|&k| k as u32))
                .collect()
        })
    }

    /// State digest of cloth `body` ([`crate::cloth::Cloth::state_digest`]: SHA-256 over every
    /// bit a step reads or writes), 32 bytes; empty when there is no such cloth.
    #[wasm_bindgen(js_name = "clothStateDigest")]
    pub fn cloth_state_digest(&self, body: usize) -> Vec<u8> {
        self.inner
            .cloths
            .get(body)
            .map_or_else(Vec::new, |c| c.state_digest().to_vec())
    }

    /// The bytes [`crate::cloth::Cloth::state_bytes`] hashes for cloth `body` (its section of
    /// the world snapshot); empty when there is no such cloth.
    #[wasm_bindgen(js_name = "clothStateBytes")]
    pub fn cloth_state_bytes(&self, body: usize) -> Vec<u8> {
        self.inner
            .cloths
            .get(body)
            .map_or_else(Vec::new, crate::cloth::Cloth::state_bytes)
    }

    /// Whether cloth `body` is asleep at its exact fixed point.
    #[wasm_bindgen(js_name = "clothAsleep")]
    pub fn cloth_asleep(&self, body: usize) -> bool {
        self.inner
            .cloths
            .get(body)
            .is_some_and(crate::cloth::Cloth::is_asleep)
    }

    /// Grab cloth `body` at `(x, y, z)` within `reach`: the grab id (a `number`, the id
    /// `clothMoveGrab` / `clothRelease` take), or `undefined` when no free particle is in reach.
    #[wasm_bindgen(js_name = "clothGrab")]
    pub fn cloth_grab(&mut self, body: usize, x: f64, y: f64, z: f64, reach: f64) -> Option<u32> {
        let p = Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        );
        self.inner
            .cloth_grab(body, p, Fix128::from_f64(reach))
            .map(|g| g.0)
    }

    /// Move grab `id`'s pointer; `false` when no grab has that id.
    #[wasm_bindgen(js_name = "clothMoveGrab")]
    pub fn cloth_move_grab(&mut self, id: u32, x: f64, y: f64, z: f64) -> bool {
        let p = Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        );
        self.inner.cloth_move_grab(crate::cloth::GrabId(id), p)
    }

    /// Release grab `id` with the pointer's velocity over the last step.
    #[wasm_bindgen(js_name = "clothRelease")]
    pub fn cloth_release(&mut self, id: u32) -> bool {
        self.inner.cloth_release(crate::cloth::GrabId(id))
    }

    /// Release grab `id` giving the particle exactly `(vx, vy, vz)`.
    #[wasm_bindgen(js_name = "clothReleaseWithVelocity")]
    pub fn cloth_release_with_velocity(&mut self, id: u32, vx: f64, vy: f64, vz: f64) -> bool {
        let v = Vec3Fix::new(
            Fix128::from_f64(vx),
            Fix128::from_f64(vy),
            Fix128::from_f64(vz),
        );
        self.inner
            .cloth_release_with_velocity(crate::cloth::GrabId(id), v)
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

/// The semantics identifier of this build ([`crate::PHYSICS_SEMANTICS_ID`], 32 bytes): two builds
/// with the same value compute every pinned scenario to the same bits, so a caller can tell whether
/// a replay recorded with another build still reproduces (the package's content hash changes on any
/// rebuild, this only when results can change).
#[wasm_bindgen(js_name = "semanticsId")]
#[must_use]
pub fn semantics_id() -> Vec<u8> {
    crate::PHYSICS_SEMANTICS_ID.to_vec()
}

// ============================================================================
// Tests
// ============================================================================

/// A query hit as `[t, px, py, pz, nx, ny, nz, kind, index, body]` (body -1
/// for none), or an empty array.
fn hit_array(hit: Option<binding_api::QueryHit>) -> Vec<f64> {
    let Some(h) = hit else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(10);
    out.push(h.t);
    out.extend_from_slice(&h.point);
    out.extend_from_slice(&h.normal);
    out.push(f64::from(h.target.0));
    out.push(h.target.1 as f64);
    out.push(h.body.map_or(-1.0, |b| b as f64));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantics_id_is_the_physics_semantics_id() {
        assert_eq!(semantics_id(), crate::PHYSICS_SEMANTICS_ID.to_vec());
        assert_eq!(semantics_id().len(), 32);
    }

    /// A world with a magnet at the origin and a steel-mass ball of material 1 above it, no
    /// gravity; returns (world, ball)
    fn magnet_world() -> (WasmPhysicsWorld, usize) {
        let mut w = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let ball = w.add_dynamic_body(0.0, 0.06, 0.0, 0.00411);
        assert!(w.set_collision_radius(ball, 0.005));
        assert!(w.set_body_material(ball, 1));
        assert_eq!(
            w.add_magnet(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.01, 1, 3.0, 1.7e6, 7850.0),
            Some(0)
        );
        (w, ball)
    }

    /// A ball falling onto a resting ball and a plane: each contact in the flat form, with
    /// the speed of the fall.
    #[test]
    fn contact_events_report_bodies_and_static_colliders() {
        let mut w = WasmPhysicsWorld::with_config(0.0, -10.0, 0.0, 4, 8);
        let ball = w.add_dynamic_body(0.0, 2.5, 0.0, 1.0);
        assert!(w.set_collision_radius(ball, 0.5));
        assert_eq!(w.add_static_plane(0.0, 1.0, 0.0, 0.0), Some(0));
        let mut hit = Vec::new();
        for _ in 0..600 {
            w.step(1.0 / 240.0);
            hit = w.get_contact_events();
            if !hit.is_empty() {
                break;
            }
        }
        assert_eq!(hit.len(), 10);
        assert_eq!(&hit[..3], &[0.0, -1.0, 3.0]);
        // a fall of 2 m at 10 m/s2 with this binding's damping rate k: y = g/k t - g/k2 (1 -
        // exp(-k t)), v = g/k (1 - exp(-k t)); bisection on t
        let k = -(0.99_f64.ln()) * 60.0;
        let fallen = |t: f64| 10.0 / k * t - 10.0 / (k * k) * (1.0 - (-k * t).exp());
        let (mut low, mut high) = (0.0_f64, 2.0_f64);
        for _ in 0..60 {
            let mid = 0.5 * (low + high);
            if fallen(mid) < 2.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        let want = 10.0 / k * (1.0 - (-k * low).exp());
        assert!(
            (hit[3] - want).abs() < 0.03 * want,
            "{} against {want}",
            hit[3]
        );
        assert_eq!(hit[8], 1.0);

        let mut w = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let a = w.add_dynamic_body(0.0, 0.0, 0.0, 1.0);
        let b = w.add_dynamic_body(1.5, 0.0, 0.0, 1.0);
        assert!(w.set_collision_radius(a, 1.0));
        assert!(w.set_collision_radius(b, 1.0));
        w.set_velocity(a, 1.0, 0.0, 0.0);
        w.set_velocity(b, -1.0, 0.0, 0.0);
        w.step(1.0 / 60.0);
        let hit = w.get_contact_events();
        assert_eq!(&hit[..3], &[0.0, 1.0, 0.0]);
        assert!((hit[3] - 2.0).abs() < 0.1, "{}", hit[3]);
    }

    /// The wasm call builds the fluid the Rust API builds: every argument has its own value, a
    /// body sinks through it, and 240 steps later both worlds are bit-equal.
    #[test]
    fn buoyancy_zone_call_equals_the_rust_api() {
        let v = |x: f64, y: f64, z: f64| {
            Vec3Fix::new(
                Fix128::from_f64(x),
                Fix128::from_f64(y),
                Fix128::from_f64(z),
            )
        };
        let mut a = WasmPhysicsWorld::with_config(0.0, -1.5, 0.0, 4, 8);
        let ball = a.add_dynamic_body(0.5, 1.0, -0.25, 2.5);
        assert_eq!(
            a.add_buoyancy_zone(-3.0, -4.0, -5.0, 6.0, 7.0, 8.0, 1.25, 1.5, 0.75, 0.3),
            Some(0)
        );
        let mut b = WasmPhysicsWorld::with_config(0.0, -1.5, 0.0, 4, 8);
        b.add_dynamic_body(0.5, 1.0, -0.25, 2.5);
        let zone = crate::buoyancy_zone::BuoyancyZone {
            shape: crate::buoyancy_zone::ZoneShape::Aabb {
                min: v(-3.0, -4.0, -5.0),
                max: v(6.0, 7.0, 8.0),
            },
            density_kg_m3: Fix128::from_f64(1.25),
            gravity: Fix128::from_f64(1.5),
            drag_linear: Fix128::from_f64(0.75),
            drag_quadratic: Fix128::from_f64(0.3),
        };
        b.inner
            .add_participant(Box::new(
                crate::buoyancy_zone::BuoyancyZoneParticipant::new(zone),
            ))
            .expect("register");
        for _ in 0..240 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        assert_eq!(a.inner.bodies[ball].position, b.inner.bodies[ball].position);
        assert_eq!(a.inner.bodies[ball].velocity, b.inner.bodies[ball].velocity);
        assert_eq!(a.snapshot_world(), b.snapshot_world());
    }

    /// Every argument the binding checks is refused alone, and a refusal registers nothing.
    #[test]
    fn a_bad_buoyancy_zone_is_refused() {
        let good = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let call = |w: &mut WasmPhysicsWorld, p: &[f64; 10]| {
            w.add_buoyancy_zone(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
        };
        let mut w = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let bad: &[(usize, f64)] = &[
            (0, f64::NAN),
            (0, 2.0), // inside out on x
            (1, 2.0), // inside out on y
            (2, 2.0), // inside out on z
            (5, f64::INFINITY),
            (6, 0.0),
            (6, 1e-300),
            (7, 0.0),
            (7, 1e-300),
            (8, -1.0),
            (9, -1.0),
        ];
        for &(i, value) in bad {
            let mut p = good;
            p[i] = value;
            assert_eq!(call(&mut w, &p), None, "argument {i} = {value}");
        }
        // past the range of Fix128, on every argument and on either side
        for i in 0..good.len() {
            for value in [1e19, 1e30, f64::MAX, -1e19, -1e30, f64::MIN] {
                let mut p = good;
                p[i] = value;
                assert_eq!(call(&mut w, &p), None, "argument {i} = {value}");
            }
        }
        assert_eq!(w.inner.participant_kinds().len(), 0);
        assert_eq!(call(&mut w, &good), Some(0));
    }

    /// The wasm call builds the zone the Rust API builds: every argument has its own value, the
    /// wind blows off the axes with a gust running, and 240 steps later both worlds are bit-equal.
    #[test]
    fn wind_zone_call_equals_the_rust_api() {
        let v = |x: f64, y: f64, z: f64| {
            Vec3Fix::new(
                Fix128::from_f64(x),
                Fix128::from_f64(y),
                Fix128::from_f64(z),
            )
        };
        let mut a = WasmPhysicsWorld::with_config(0.0, -1.5, 0.0, 4, 8);
        let ball = a.add_dynamic_body(0.5, 1.0, -0.25, 2.5);
        assert_eq!(
            a.add_wind_zone(
                -3.0, -4.0, -5.0, 6.0, 7.0, 8.0, 1.0, 2.0, -2.0, 4.0, 1.5, 0.75, 1.25, 0.8, 0.6
            ),
            Some(0)
        );
        let mut b = WasmPhysicsWorld::with_config(0.0, -1.5, 0.0, 4, 8);
        b.add_dynamic_body(0.5, 1.0, -0.25, 2.5);
        let zone = crate::wind_zone::WindZone {
            shape: crate::buoyancy_zone::ZoneShape::Aabb {
                min: v(-3.0, -4.0, -5.0),
                max: v(6.0, 7.0, 8.0),
            },
            air_density_kg_m3: Fix128::from_f64(1.25),
            direction: v(1.0, 2.0, -2.0).try_normalize().expect("direction"),
            base_speed_m_s: Fix128::from_f64(4.0),
            turbulence_amplitude: Fix128::from_f64(1.5),
            gust_frequency_hz: Fix128::from_f64(0.75),
            drag_coefficient: Fix128::from_f64(0.8),
            reference_area_m2: Fix128::from_f64(0.6),
        };
        b.inner
            .add_participant(Box::new(crate::wind_zone::WindZoneParticipant::new(zone)))
            .expect("register");
        for _ in 0..240 {
            a.step(1.0 / 60.0);
            b.step(1.0 / 60.0);
        }
        assert_eq!(a.inner.bodies[ball].position, b.inner.bodies[ball].position);
        assert_eq!(a.inner.bodies[ball].velocity, b.inner.bodies[ball].velocity);
        assert_eq!(a.snapshot_world(), b.snapshot_world());
    }

    /// Every argument the binding checks is refused alone, and a refusal registers nothing.
    /// oracle: the wasm call builds the rope the Rust API builds, bit for bit after 60 frames,
    /// and past the range of Fix128 every float argument is refused.
    #[test]
    fn rope_call_equals_the_rust_api_and_refuses_bad_values() {
        let mut a = WasmPhysicsWorld::with_config(0.0, -10.0, 0.0, 4, 8);
        let ball = a.add_dynamic_body(1.0, -1.0, 0.0, 2.0);
        let index = a.add_rope(
            &[0.0, 0.0, 0.0],
            &[1.0, -1.0, 0.0],
            4,
            0.5,
            0.99,
            None,
            &[],
            Some(ball as u32),
            &[0.0, 0.0, 0.0],
        );
        assert_eq!(index, Some(0));
        let fx = Fix128::from_f64;
        let v3 = |x, y, z| Vec3Fix::new(fx(x), fx(y), fx(z));
        let mut b = PhysicsWorld::new(a.inner.config);
        let body = b.add_body(RigidBody::new_dynamic(v3(1.0, -1.0, 0.0), fx(2.0)));
        let length = v3(1.0, -1.0, 0.0).length();
        let mut rope =
            crate::rope::Rope::new(Vec3Fix::ZERO, v3(1.0, -1.0, 0.0), 4, fx(0.5) / length);
        rope.config.damping = fx(0.99);
        let anchors = vec![crate::rope_participant::RopeAnchor {
            particle: 4,
            body,
            local_anchor: Vec3Fix::ZERO,
        }];
        let resident = crate::rope_participant::RopeParticipant::new(rope, anchors).unwrap();
        b.add_participant(Box::new(resident)).unwrap();
        for _ in 0..60 {
            a.step(1.0 / 60.0);
            b.step(fx(1.0 / 60.0));
        }
        assert_eq!(a.inner.snapshot_world(), b.snapshot_world());
        assert_eq!(a.rope_positions(0).unwrap().len(), 15);
        assert_eq!(a.rope_positions(1), None);

        let o = [0.0, 0.0, 0.0];
        let x = [1.0, 0.0, 0.0];
        let mut w = WasmPhysicsWorld::with_config(0.0, -10.0, 0.0, 4, 8);
        for value in [1e19, 1e30, f64::MAX, -1e19, -1e30, f64::MIN, f64::NAN] {
            let high = [value, 0.0, 0.0];
            let mut add = |s: &[f64], e: &[f64], m: f64, d: f64| {
                w.add_rope(s, e, 2, m, d, None, &[], None, &[])
            };
            assert_eq!(add(&high, &x, 1.0, 1.0), None, "start {value}");
            assert_eq!(add(&o, &high, 1.0, 1.0), None, "end {value}");
            assert_eq!(add(&o, &x, value, 1.0), None, "mass {value}");
            assert_eq!(add(&o, &x, 1.0, value), None, "damping {value}");
        }
        assert_eq!(w.add_rope(&o, &o, 2, 1.0, 1.0, None, &[], None, &[]), None);
        assert_eq!(w.add_rope(&o, &x, 0, 1.0, 1.0, None, &[], None, &[]), None);
        assert_eq!(
            w.add_rope(&o, &x, 1025, 1.0, 1.0, None, &[], None, &[]),
            None
        );
        assert_eq!(
            w.add_rope(&o, &x, 2, 1.0, 1.0, Some(9), &o, None, &[]),
            None
        );
        assert_eq!(
            w.add_rope(&o, &x, 2, 1.0, 1.0, None, &[], None, &[]),
            Some(0)
        );
    }

    #[test]
    fn a_bad_wind_zone_is_refused() {
        let good = [
            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0,
        ];
        let call = |w: &mut WasmPhysicsWorld, p: &[f64; 15]| {
            w.add_wind_zone(
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12],
                p[13], p[14],
            )
        };
        let mut w = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let bad: &[(usize, f64)] = &[
            (0, f64::NAN),
            (0, 2.0), // inside out on x
            (1, 2.0), // inside out on y
            (2, 2.0), // inside out on z
            (5, f64::INFINITY),
            (6, f64::NAN),
            (9, -1.0),
            (10, -1.0),
            (11, -1.0),
            (12, 0.0),
            (12, 1e-300),
            (13, 0.0),
            (14, 0.0),
        ];
        for &(i, value) in bad {
            let mut p = good;
            p[i] = value;
            assert_eq!(call(&mut w, &p), None, "argument {i} = {value}");
        }
        // past the range of Fix128, on every argument and on either side
        for i in 0..good.len() {
            for value in [1e19, 1e30, f64::MAX, -1e19, -1e30, f64::MIN] {
                let mut p = good;
                p[i] = value;
                assert_eq!(call(&mut w, &p), None, "argument {i} = {value}");
            }
        }
        let mut zero = good;
        zero[6] = 0.0;
        assert_eq!(call(&mut w, &zero), None, "zero direction");
        assert_eq!(call(&mut w, &good), Some(0), "nothing registered before it");
    }

    /// The wasm calls build the world the Rust API builds: a registered material with friction and
    /// restitution of different values, a ball of it rolling into a resting ball, and a magnet
    /// with a different value for every argument; 120 steps later both are bit-equal (a swapped
    /// argument would differ, the hit makes the material's values matter).
    #[test]
    fn material_and_magnet_calls_equal_the_rust_api() {
        let scene = |w: &mut WasmPhysicsWorld| {
            let a = w
                .add_shaped_body(3, 0.2, 0.2, 0.2, 7800.0, 0.0, 0.0, 0.0)
                .unwrap() as usize;
            let b = w
                .add_shaped_body(3, 0.2, 0.2, 0.2, 7800.0, 0.9, 0.05, 0.0)
                .unwrap() as usize;
            w.set_velocity(a, 1.5, 0.0, 0.0);
            (a, b)
        };
        let mut c = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let (a, _) = scene(&mut c);
        let m = c.register_material(0.7, 0.4).unwrap();
        assert!(c.set_body_material(a, m));
        assert_eq!(
            c.add_magnet(1.6, 0.3, 0.1, 0.2, 900.0, 0.3, 0.15, m, 2.5, 1.2e6, 7700.0),
            Some(0)
        );

        let mut r = WasmPhysicsWorld::with_config(0.0, 0.0, 0.0, 4, 8);
        let (ra, _) = scene(&mut r);
        let fx = Fix128::from_f64;
        let v = |x: f64, y: f64, z: f64| Vec3Fix::new(fx(x), fx(y), fx(z));
        let rm = r
            .inner
            .material_table
            .register(crate::material::PhysicsMaterial::new(0, fx(0.7), fx(0.4)));
        r.inner.set_body_material(ra, rm);
        r.inner
            .add_force_field(crate::force::ForceFieldInstance::new(
                crate::force::ForceField::InducedDipole {
                    position: v(1.6, 0.3, 0.1),
                    moment: v(0.2, 900.0, 0.3),
                    radius: fx(0.15),
                    material: rm,
                    response: crate::electromagnetic::MagneticResponse {
                        chi_eff: fx(2.5),
                        saturation_magnetization: fx(1.2e6),
                    },
                    density: fx(7700.0),
                },
            ));
        for _ in 0..120 {
            c.step(1.0 / 120.0);
            r.step(1.0 / 120.0);
        }
        assert_eq!(c.get_positions(), r.get_positions());
        assert_eq!(c.get_velocities(), r.get_velocities());
        // the hit happened: the resting ball moves
        assert!(c.get_velocities()[3] > 0.0, "{:?}", c.get_velocities());
    }

    /// A magnet survives the snapshot: a run saved at step 30, run on, restored and run again
    /// ends bit for bit where it did; the ball is pulled toward the magnet. Bad values give `None`.
    #[test]
    fn magnet_pulls_and_survives_save_and_restore() {
        let (mut a, ball) = magnet_world();
        for _ in 0..30 {
            a.step(1.0 / 120.0);
        }
        let saved = a.serialize_state();
        for _ in 0..30 {
            a.step(1.0 / 120.0);
        }
        let straight = a.get_positions();
        assert!(a.deserialize_state(&saved));
        for _ in 0..30 {
            a.step(1.0 / 120.0);
        }
        assert_eq!(a.get_positions(), straight);
        assert!(a.get_positions()[ball * 3 + 1] < 0.06);
        assert_eq!(
            a.add_magnet(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1, 3.0, 1.7e6, 7850.0),
            None
        );
        assert_eq!(
            a.add_magnet(
                f64::NAN,
                0.0,
                0.0,
                0.0,
                0.5,
                0.0,
                0.01,
                1,
                3.0,
                1.7e6,
                7850.0
            ),
            None
        );
        assert_eq!(
            a.add_magnet(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.01, 1, 2.0, 1.7e6, 7850.0),
            None
        );
        assert!(!a.set_body_material(ball, 70_000));
        assert!(!a.set_body_material(99, 1));
    }

    #[test]
    fn test_world_creation() {
        let world = WasmPhysicsWorld::new();
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_world_with_config() {
        let world = WasmPhysicsWorld::with_config(0.0, -9.81, 0.0, 4, 8);
        assert_eq!(world.body_count(), 0);
    }

    #[test]
    fn test_add_bodies() {
        let mut world = WasmPhysicsWorld::new();
        let d = world.add_dynamic_body(0.0, 10.0, 0.0, 1.0);
        let s = world.add_static_body(0.0, 0.0, 0.0);
        let k = world.add_kinematic_body(5.0, 0.0, 0.0);
        assert_eq!(d, 0);
        assert_eq!(s, 1);
        assert_eq!(k, 2);
        assert_eq!(world.body_count(), 3);
    }

    #[test]
    fn test_body_type() {
        let mut world = WasmPhysicsWorld::new();
        world.add_dynamic_body(0.0, 0.0, 0.0, 1.0);
        world.add_static_body(0.0, 0.0, 0.0);
        world.add_kinematic_body(0.0, 0.0, 0.0);
        assert_eq!(world.get_body_type(0), BodyType::Dynamic as u8);
        assert_eq!(world.get_body_type(1), BodyType::Static as u8);
        assert_eq!(world.get_body_type(2), BodyType::Kinematic as u8);
        assert_eq!(world.get_body_type(999), 255);
    }

    #[test]
    fn test_step_gravity() {
        let mut world = WasmPhysicsWorld::new();
        world.add_dynamic_body(0.0, 10.0, 0.0, 1.0);

        for _ in 0..60 {
            world.step(1.0 / 60.0);
        }

        let positions = world.get_positions();
        assert!(positions[1] < 10.0, "Body should fall, y={}", positions[1]);
    }

    #[test]
    fn test_batch_getters() {
        let mut world = WasmPhysicsWorld::new();
        world.add_static_body(1.0, 2.0, 3.0);
        world.add_static_body(4.0, 5.0, 6.0);

        let pos = world.get_positions();
        assert_eq!(pos.len(), 6);
        assert!((pos[0] - 1.0).abs() < 1e-10);
        assert!((pos[4] - 5.0).abs() < 1e-10);

        let vel = world.get_velocities();
        assert_eq!(vel.len(), 6);

        let rot = world.get_rotations();
        assert_eq!(rot.len(), 8); // 2 bodies * 4 quat components

        let states = world.get_states();
        assert_eq!(states.len(), 20); // 2 bodies * 10 values
    }

    #[test]
    fn test_set_position_velocity() {
        let mut world = WasmPhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);
        world.add_dynamic_body(0.0, 0.0, 0.0, 1.0);

        world.set_position(0, 5.0, 6.0, 7.0);
        let pos = world.get_positions();
        assert!((pos[0] - 5.0).abs() < 1e-10);
        assert!((pos[1] - 6.0).abs() < 1e-10);

        world.set_velocity(0, 1.0, 0.0, 0.0);
        let vel = world.get_velocities();
        assert!((vel[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_serialization() {
        let mut world = WasmPhysicsWorld::new();
        world.add_dynamic_body(0.0, 10.0, 0.0, 1.0);

        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }

        let state = world.serialize_state();
        assert!(!state.is_empty());

        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let pos_after = world.get_positions();

        world.deserialize_state(&state);
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let pos_restored = world.get_positions();

        assert!(
            (pos_after[1] - pos_restored[1]).abs() < 1e-10,
            "State restore should be deterministic"
        );
    }

    #[test]
    fn test_impulse_batch() {
        let mut world = WasmPhysicsWorld::new();
        world.set_gravity(0.0, 0.0, 0.0);
        world.add_dynamic_body(0.0, 0.0, 0.0, 1.0);

        world.apply_impulses_batch(&[0.0, 10.0, 0.0, 0.0]);
        world.step(1.0 / 60.0);

        let pos = world.get_positions();
        assert!(pos[0] > 0.0, "Impulse should move body, x={}", pos[0]);
    }

    #[test]
    fn test_raycast_miss() {
        let mut world = WasmPhysicsWorld::new();
        world.add_static_body(100.0, 0.0, 0.0);

        // Ray pointing away from body
        let result = world.raycast(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 50.0, 1.0);
        assert!(result.is_empty(), "Should miss body at x=100");
    }

    /// oracle: the WebAssembly calls do what the Rust API calls with the
    /// same values do — the worlds stay bit-identical through 30 steps.
    #[test]
    fn binding_calls_match_the_rust_api_bit_for_bit() {
        use crate::plane_collider::PlaneCollider;
        use crate::shape::Shape;
        use crate::static_collider::StaticCollider;
        let f = Fix128::from_f64;
        let mut w = WasmPhysicsWorld::new();
        let mut r = PhysicsWorld::new(PhysicsConfig::default());
        w.add_static_body(0.0, 0.0, 0.0);
        w.add_dynamic_body(2.0, 0.0, 0.0, 1.0);
        r.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        r.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(2, 0, 0),
            Fix128::ONE,
        ));

        assert!(w.set_collision_radius(1, 0.5));
        r.set_body_collision_radius(1, f(0.5));
        assert_eq!(
            w.add_shaped_body(0, 0.5, 0.5, 0.5, 2.0, 5.0, 3.0, 0.0),
            Some(2)
        );
        let cube = Shape::Box {
            half_extents: Vec3Fix::new(f(0.5), f(0.5), f(0.5)),
        };
        r.add_shaped_body(&cube, f(2.0), Vec3Fix::from_int(5, 3, 0))
            .unwrap();
        assert_eq!(w.add_static_plane(0.0, 2.0, 0.0, -1.0), Some(0));
        r.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            -Fix128::ONE,
        )));
        assert_eq!(
            w.add_ball_joint(0, 1, &[0.0, 0.0, 0.0, -2.0, 0.0, 0.0]),
            Some(0)
        );
        r.add_joint(Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-2, 0, 0),
        )));
        assert_eq!(w.add_spring_joint(1, 2, &[0.0; 6], 3.0, 50.0, 0.5), Some(1));
        r.add_joint(Joint::Spring(SpringJoint::new(
            1,
            2,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            f(3.0),
            f(50.0),
            f(0.5),
        )));
        assert_eq!((w.joint_count(), w.static_collider_count()), (2, 1));

        for _ in 0..30 {
            w.step(1.0 / 60.0);
            r.step(Fix128::from_ratio(1, 60));
        }
        for (a, b) in w.inner.bodies.iter().zip(&r.bodies) {
            assert_eq!(
                (a.position, a.velocity, a.rotation),
                (b.position, b.velocity, b.rotation)
            );
        }
        assert_eq!(w.inner.joints, r.joints);
    }

    /// Refused arguments return `None` / `false` and leave the world as it was.
    #[test]
    fn binding_calls_refuse_bad_arguments_without_touching_the_world() {
        let mut w = WasmPhysicsWorld::new();
        w.add_static_body(0.0, 0.0, 0.0);
        w.add_dynamic_body(2.0, 0.0, 0.0, 1.0);
        assert!(!w.set_collision_radius(1, 0.0));
        assert!(!w.set_collision_radius(9, 1.0));
        assert!(!w.clear_collision_radius(9));
        assert_eq!(
            w.add_shaped_body(9, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            None
        );
        assert_eq!(
            w.add_shaped_body(0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0),
            None
        );
        assert!(
            !w.set_body_shape(1, 5, 1.0, 2.0, 0.0),
            "torus minor >= major"
        );
        assert_eq!(w.add_static_plane(0.0, 0.0, 0.0, 0.0), None);
        assert_eq!(
            w.add_static_heightfield(&[0.0; 3], 2, 2, 1.0, 0.0, 0.0, 0.0),
            None
        );
        assert_eq!(w.add_static_trimesh(&[0.0; 9], &[0, 1, 3]), None);
        assert!(!w.remove_static_collider(0));
        assert_eq!(w.add_ball_joint(1, 1, &[0.0; 6]), None, "self joint");
        assert_eq!(
            w.add_ball_joint(0, 1, &[0.0; 5]),
            None,
            "five anchor values"
        );
        assert_eq!(
            w.add_hinge_joint(0, 1, &[0.0; 6], &[0.0; 6]),
            None,
            "zero axes"
        );
        assert_eq!(
            w.add_fixed_joint(0, 1, &[0.0; 6], &[0.0; 4]),
            None,
            "zero rotation"
        );
        assert_eq!(
            w.add_slider_joint(0, 1, &[0.0, 0.0, 0.0], &[0.0; 6]),
            None,
            "zero axis"
        );
        assert_eq!(
            w.add_spring_joint(0, 1, &[0.0; 6], 1.0, 0.0, 0.0),
            None,
            "stiffness 0"
        );
        assert!(!w.remove_joint(0));
        assert_eq!(
            (w.body_count(), w.joint_count(), w.static_collider_count()),
            (2, 0, 0)
        );
    }

    // ------------------------------------------------------------------
    // World queries and body observation
    // ------------------------------------------------------------------

    /// A static body of collision radius 1 at (0, 0, 10), a dynamic body of
    /// radius 1 at (0, 0, 20) and the static plane y = -2, through the
    /// binding and through the Rust API.
    fn query_twins() -> (WasmPhysicsWorld, PhysicsWorld) {
        use crate::plane_collider::PlaneCollider;
        use crate::static_collider::StaticCollider;
        let f = Fix128::from_f64;
        let mut w = WasmPhysicsWorld::new();
        assert_eq!(w.add_static_body(0.0, 0.0, 10.0), 0);
        assert_eq!(w.add_dynamic_body(0.0, 0.0, 20.0, 1.0), 1);
        assert!(w.set_collision_radius(0, 1.0));
        assert!(w.set_collision_radius(1, 1.0));
        assert_eq!(w.add_static_plane(0.0, 1.0, 0.0, -2.0), Some(0));
        let mut r = PhysicsWorld::new(PhysicsConfig::default());
        r.add_body(RigidBody::new_static(Vec3Fix::new(f(0.0), f(0.0), f(10.0))));
        r.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(f(0.0), f(0.0), f(20.0)),
            Fix128::ONE,
        ));
        r.set_body_collision_radius(0, Fix128::ONE);
        r.set_body_collision_radius(1, Fix128::ONE);
        r.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            Vec3Fix::UNIT_Y,
            f(-2.0),
        )));
        (w, r)
    }

    /// `[t, point, normal, kind, index, body]` of a Rust hit, every Fix128
    /// through `to_f64` (the binding must equal it exactly).
    fn expected(t: Fix128, p: Vec3Fix, n: Vec3Fix, kind: f64, index: f64, body: f64) -> Vec<f64> {
        vec![
            t.to_f64(),
            p.x.to_f64(),
            p.y.to_f64(),
            p.z.to_f64(),
            n.x.to_f64(),
            n.y.to_f64(),
            n.z.to_f64(),
            kind,
            index,
            body,
        ]
    }

    fn near(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-12, "{a:?} vs {b:?}");
        }
    }

    /// oracle: castRay / castSphere / castCapsule / overlapSphere return the
    /// Rust API's answer on the same scene, and it is the closed form (ray to
    /// a sphere of radius 1 at distance 10: t = 9; a radius 0.5 sphere or
    /// capsule: t = 8.5; the plane y = -2 straight down: t = 2).
    // covers: COV-ENGINE-070
    #[test]
    fn query_calls_match_the_rust_api_and_the_closed_form() {
        use crate::shape_raycast::{RayFilter, RayTarget};
        let f = Fix128::from_f64;
        let vf = |x, y, z| Vec3Fix::new(f(x), f(y), f(z));
        let (w, r) = query_twins();
        let flt = RayFilter::default();
        let o = [0.0, 0.0, 0.0];
        let pz = [0.0, 0.0, 1.0];

        let got = w.cast_ray(&o, &pz, 100.0, None);
        let h = r
            .cast_ray(Vec3Fix::ZERO, Vec3Fix::UNIT_Z, f(100.0), &flt)
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 0.0, 0.0, 0.0));
        near(&got, &[9.0, 0.0, 0.0, 9.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]);

        // off-axis ray from (0.25, 0.5, 0): lateral d^2 = 0.3125, so
        // t = 10 - sqrt(1 - d^2) and the normal is hit - centre
        let got = w.cast_ray(&[0.25, 0.5, 0.0], &pz, 100.0, None);
        let h = r
            .cast_ray(vf(0.25, 0.5, 0.0), Vec3Fix::UNIT_Z, f(100.0), &flt)
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 0.0, 0.0, 0.0));
        let tc = 10.0 - 0.6875f64.sqrt();
        near(
            &got,
            &[tc, 0.25, 0.5, tc, 0.25, 0.5, tc - 10.0, 0.0, 0.0, 0.0],
        );

        let got = w.cast_ray(&o, &pz, 100.0, Some(0));
        let h = r
            .cast_ray(
                Vec3Fix::ZERO,
                Vec3Fix::UNIT_Z,
                f(100.0),
                &flt.excluding_body(0),
            )
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 0.0, 1.0, 1.0));
        near(&got[..1], &[19.0]);

        let got = w.cast_ray(&o, &[0.0, -3.0, 0.0], 10.0, None);
        let h = r
            .cast_ray(Vec3Fix::ZERO, vf(0.0, -3.0, 0.0), f(10.0), &flt)
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 1.0, 0.0, -1.0));
        near(&got, &[2.0, 0.0, -2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0]);

        assert!(w.cast_ray(&o, &[1.0, 0.0, 0.0], 100.0, None).is_empty());

        let got = w.cast_sphere(&o, 0.5, &pz, 100.0, None);
        let h = r
            .cast_sphere(Vec3Fix::ZERO, f(0.5), Vec3Fix::UNIT_Z, f(100.0), &flt)
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 0.0, 0.0, 0.0));
        near(&got, &[8.5, 0.0, 0.0, 9.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]);

        let got = w.cast_capsule(&[-1.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 0.5, &pz, 100.0, None);
        let h = r
            .cast_capsule(
                vf(-1.0, 0.0, 0.0),
                vf(1.0, 0.0, 0.0),
                f(0.5),
                Vec3Fix::UNIT_Z,
                f(100.0),
                &flt,
            )
            .unwrap();
        assert_eq!(got, expected(h.t, h.point, h.normal, 0.0, 0.0, 0.0));
        near(&got, &[8.5, 0.0, 0.0, 9.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]);

        // centre (0, -1.2, 10), radius 1: body 0 (distance 1.2 < 2) and the
        // plane (distance 0.8 < 1)
        assert_eq!(
            r.overlap_sphere(vf(0.0, -1.2, 10.0), Fix128::ONE, &flt),
            vec![RayTarget::Body(0), RayTarget::StaticCollider(0)]
        );
        assert_eq!(
            w.overlap_sphere(&[0.0, -1.2, 10.0], 1.0, None),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            w.overlap_sphere(&[0.0, -1.2, 10.0], 1.0, Some(0)),
            vec![1.0, 0.0]
        );
    }

    /// oracle: observeBody is the Rust observation through `to_f64`, before
    /// and after a step.
    #[test]
    fn observe_body_matches_the_rust_api() {
        let (mut w, mut r) = query_twins();
        w.set_velocity(1, 1.0, 2.0, 3.0);
        r.bodies[1].velocity = Vec3Fix::new(
            Fix128::from_f64(1.0),
            Fix128::from_f64(2.0),
            Fix128::from_f64(3.0),
        );
        assert_eq!(
            w.observe_body(1),
            vec![0.0, 0.0, 20.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        w.step(1.0 / 60.0);
        r.step(Fix128::from_f64(1.0 / 60.0));
        for i in 0..2 {
            let o = r.observe_body(i).unwrap();
            let b = |x: bool| if x { 1.0 } else { 0.0 };
            assert_eq!(
                w.observe_body(i),
                vec![
                    o.position.x.to_f64(),
                    o.position.y.to_f64(),
                    o.position.z.to_f64(),
                    o.velocity.x.to_f64(),
                    o.velocity.y.to_f64(),
                    o.velocity.z.to_f64(),
                    o.rotation.x.to_f64(),
                    o.rotation.y.to_f64(),
                    o.rotation.z.to_f64(),
                    o.rotation.w.to_f64(),
                    o.angular_velocity.x.to_f64(),
                    o.angular_velocity.y.to_f64(),
                    o.angular_velocity.z.to_f64(),
                    b(o.sleeping),
                    b(o.in_contact),
                ]
            );
        }
    }

    /// oracle: degenerate queries give an empty array (zero direction,
    /// negative radius, max_t <= 0 as the Rust API; non-finite values, a
    /// vector that is not 3 values, an exclude index that is not a body, an
    /// unknown body to observe refused) and an empty world answers nothing.
    #[test]
    fn query_calls_refuse_degenerate_arguments() {
        let (w, _) = query_twins();
        let o = [0.0, 0.0, 0.0];
        let pz = [0.0, 0.0, 1.0];
        let nan = f64::NAN;
        assert!(w.cast_ray(&o, &o, 100.0, None).is_empty(), "zero dir");
        assert!(w.cast_ray(&o, &pz, 0.0, None).is_empty(), "max_t 0");
        assert!(w.cast_ray(&o, &pz, nan, None).is_empty(), "max_t nan");
        assert!(w.cast_ray(&[nan, 0.0, 0.0], &pz, 100.0, None).is_empty());
        assert!(
            w.cast_ray(&[0.0, 0.0], &pz, 100.0, None).is_empty(),
            "2 values"
        );
        assert!(w.cast_ray(&o, &pz, 100.0, Some(2)).is_empty(), "exclude 2");
        assert!(
            w.cast_sphere(&o, -0.5, &pz, 100.0, None).is_empty(),
            "r < 0"
        );
        assert!(w.cast_sphere(&o, nan, &pz, 100.0, None).is_empty());
        assert!(
            w.cast_sphere(&o, 0.5, &o, 100.0, None).is_empty(),
            "zero dir"
        );
        let (a, b) = ([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(w.cast_capsule(&a, &b, -0.5, &pz, 100.0, None).is_empty());
        assert!(w.cast_capsule(&a, &b, 0.5, &o, 100.0, None).is_empty());
        assert!(w.cast_capsule(&a, &b, 0.5, &pz, 100.0, Some(9)).is_empty());
        let c = [0.0, -1.2, 10.0];
        assert!(w.overlap_sphere(&c, -1.0, None).is_empty(), "r < 0");
        assert!(w.overlap_sphere(&c, nan, None).is_empty());
        assert!(w.overlap_sphere(&c, 1.0, Some(5)).is_empty());
        assert!(w.observe_body(2).is_empty());

        let empty = WasmPhysicsWorld::new();
        assert!(empty.cast_ray(&o, &pz, 100.0, None).is_empty());
        assert!(empty.overlap_sphere(&o, 1.0, None).is_empty());
        assert!(empty.observe_body(0).is_empty());
    }
}

/// What was done to a world, as bytes the host writes ([`crate::input_log::InputLog`]).
#[wasm_bindgen]
#[derive(Default)]
pub struct WasmInputLog {
    inner: crate::input_log::InputLog<Vec<u8>>,
}

#[wasm_bindgen]
impl WasmInputLog {
    /// An empty log.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// What happened at `step`. `false` for a step before the last.
    pub fn record(&mut self, step: f64, input: &[u8]) -> bool {
        if !(step.is_finite()
            && step >= 0.0
            && step.fract() == 0.0
            && step <= 9_007_199_254_740_991.0)
        {
            return false;
        }
        self.inner.record(step as u64, input.to_vec()).is_ok()
    }

    /// How many entries.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> u32 {
        self.inner.entries().len() as u32
    }

    /// The step of entry `i`.
    #[wasm_bindgen(js_name = "stepAt")]
    pub fn step_at(&self, i: u32) -> Option<f64> {
        self.inner.entries().get(i as usize).map(|e| e.0 as f64)
    }

    /// The bytes of entry `i`.
    #[wasm_bindgen(js_name = "inputAt")]
    pub fn input_at(&self, i: u32) -> Option<Vec<u8>> {
        self.inner.entries().get(i as usize).map(|e| e.1.clone())
    }

    /// The log as a code, with the first bytes of the digest it reaches (32 bytes). `undefined` for another length.
    #[wasm_bindgen(js_name = "toCode")]
    pub fn to_code(&self, digest: &[u8]) -> Option<String> {
        let digest: &[u8; 32] = digest.try_into().ok()?;
        Some(self.inner.to_code(digest))
    }

    /// A code back into a log. `undefined` when it is not a code.
    #[wasm_bindgen(js_name = "fromCode")]
    pub fn from_code(code: &str) -> Option<WasmInputLog> {
        let (_, inner) = crate::input_log::InputLog::from_code(code).ok()?;
        Some(Self { inner })
    }

    /// The digest's first four bytes a code says it reaches. `undefined` when it is not a code.
    #[wasm_bindgen(js_name = "codeHead")]
    pub fn code_head(code: &str) -> Option<Vec<u8>> {
        let (head, _) = crate::input_log::InputLog::<Vec<u8>>::from_code(code).ok()?;
        Some(head.to_vec())
    }
}
