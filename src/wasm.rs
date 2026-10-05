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
            damping: Fix128::from_ratio(99, 100),
            ..Default::default()
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
    // Collision radius, shapes, static colliders, joints. Indices come back
    // as `number`, or `undefined` when an argument is refused (see
    // `binding_api` for the checks); setters return `false` instead.
    // ------------------------------------------------------------------

    /// Set a body's collision sphere radius (finite and positive).
    #[wasm_bindgen(js_name = "setCollisionRadius")]
    pub fn set_collision_radius(&mut self, body_id: usize, radius: f64) -> bool {
        binding_api::set_collision_radius(&mut self.inner, body_id, radius)
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get the library version string.
#[wasm_bindgen(js_name = "alicePhysicsVersion")]
pub fn alice_physics_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
}
