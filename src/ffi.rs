//! C Foreign Function Interface for ALICE-Physics
//!
//! Provides a C-compatible API for Unity, Unreal Engine, and other
//! game engines to use the deterministic physics simulation.
//!
//! # Safety
//!
//! All functions that take raw pointers require valid, non-null pointers.
//! The caller is responsible for proper lifecycle management (create/destroy pairs).
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::{PhysicsWorld, RigidBody, SolverConfig};

// ============================================================================
// C-compatible types
// ============================================================================

/// C-compatible 3D vector (f64 for FFI boundary, converted to Fix128 internally)
#[repr(C)]
pub struct AliceVec3 {
    /// X component
    pub x: f64,
    /// Y component
    pub y: f64,
    /// Z component
    pub z: f64,
}

/// C-compatible quaternion
#[repr(C)]
pub struct AliceQuat {
    /// X component (imaginary i)
    pub x: f64,
    /// Y component (imaginary j)
    pub y: f64,
    /// Z component (imaginary k)
    pub z: f64,
    /// W component (scalar/real part)
    pub w: f64,
}

/// C-compatible physics config
#[repr(C)]
pub struct AlicePhysicsConfig {
    /// Number of substeps per step
    pub substeps: u32,
    /// Number of solver iterations per substep
    pub iterations: u32,
    /// Gravity X component (m/s^2)
    pub gravity_x: f64,
    /// Gravity Y component (m/s^2)
    pub gravity_y: f64,
    /// Gravity Z component (m/s^2)
    pub gravity_z: f64,
    /// Velocity damping factor (0..1)
    pub damping: f64,
}

/// C-compatible body info (read-only snapshot)
#[repr(C)]
pub struct AliceBodyInfo {
    /// Body position in world space
    pub position: AliceVec3,
    /// Linear velocity (m/s)
    pub velocity: AliceVec3,
    /// Angular velocity (rad/s)
    pub angular_velocity: AliceVec3,
    /// Orientation quaternion
    pub rotation: AliceQuat,
    /// Inverse mass (0 for static bodies)
    pub inv_mass: f64,
    /// 1 if static body, 0 otherwise
    pub is_static: u8,
    /// 1 if sensor/trigger body, 0 otherwise
    pub is_sensor: u8,
}

// ============================================================================
// Helper conversions
// ============================================================================

impl AliceVec3 {
    fn to_vec3fix(&self) -> Vec3Fix {
        Vec3Fix::new(
            Fix128::from_f64(self.x),
            Fix128::from_f64(self.y),
            Fix128::from_f64(self.z),
        )
    }

    fn from_vec3fix(v: Vec3Fix) -> Self {
        Self {
            x: v.x.to_f64(),
            y: v.y.to_f64(),
            z: v.z.to_f64(),
        }
    }
}

impl AliceQuat {
    fn from_quatfix(q: QuatFix) -> Self {
        Self {
            x: q.x.to_f64(),
            y: q.y.to_f64(),
            z: q.z.to_f64(),
            w: q.w.to_f64(),
        }
    }
}

// ============================================================================
// World lifecycle
// ============================================================================

/// Create a new physics world with default config.
/// Returns an opaque pointer. Must be freed with `alice_physics_world_destroy`.
#[no_mangle]
pub extern "C" fn alice_physics_world_create() -> *mut PhysicsWorld {
    let world = PhysicsWorld::new(SolverConfig::default());
    Box::into_raw(Box::new(world))
}

/// Create a physics world with custom config.
#[no_mangle]
pub extern "C" fn alice_physics_world_create_with_config(
    config: AlicePhysicsConfig,
) -> *mut PhysicsWorld {
    let solver_config = SolverConfig {
        substeps: config.substeps as usize,
        iterations: config.iterations as usize,
        gravity: Vec3Fix::new(
            Fix128::from_f64(config.gravity_x),
            Fix128::from_f64(config.gravity_y),
            Fix128::from_f64(config.gravity_z),
        ),
        damping: Fix128::from_f64(config.damping),
    };
    let world = PhysicsWorld::new(solver_config);
    Box::into_raw(Box::new(world))
}

/// Destroy a physics world.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_destroy(world: *mut PhysicsWorld) {
    if !world.is_null() {
        drop(Box::from_raw(world));
    }
}

/// Step the simulation by dt seconds (as f64, converted to Fix128).
///
/// Returns 1 on success, 0 on failure (null pointer or internal panic).
///
/// # Safety
/// `world` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_step(world: *mut PhysicsWorld, dt: f64) -> u8 {
    let w = match world.as_mut() {
        Some(w) => w as *mut PhysicsWorld,
        None => return 0,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (*w).step(Fix128::from_f64(dt));
    }));
    result.is_ok() as u8
}

/// Step the simulation N times with fixed dt (batch stepping).
///
/// Amortizes FFI overhead — ideal for training loops and rollback re-simulation.
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `world` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_step_n(world: *mut PhysicsWorld, dt: f64, steps: u32) -> u8 {
    let w = match world.as_mut() {
        Some(w) => w as *mut PhysicsWorld,
        None => return 0,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dt_fix = Fix128::from_f64(dt);
        for _ in 0..steps {
            (*w).step(dt_fix);
        }
    }));
    result.is_ok() as u8
}

/// Get the number of bodies.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`, or null.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_body_count(world: *const PhysicsWorld) -> u32 {
    match world.as_ref() {
        Some(w) => w.bodies.len() as u32,
        None => 0,
    }
}

/// Get all body positions as a flat [x,y,z, x,y,z, ...] f64 array (zero-copy write).
///
/// Caller provides a buffer of `body_count * 3` f64 values.
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `out` must point to at least `body_count * 3` f64 values.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_get_positions_batch(
    world: *const PhysicsWorld,
    out: *mut f64,
    out_capacity: u32,
) -> u8 {
    let w = match world.as_ref() {
        Some(w) => w,
        None => return 0,
    };
    if out.is_null() {
        return 0;
    }
    let n = w.bodies.len();
    if (out_capacity as usize) < n * 3 {
        return 0;
    }
    let buf = std::slice::from_raw_parts_mut(out, n * 3);
    for (i, body) in w.bodies.iter().enumerate() {
        buf[i * 3] = body.position.x.to_f64();
        buf[i * 3 + 1] = body.position.y.to_f64();
        buf[i * 3 + 2] = body.position.z.to_f64();
    }
    1
}

/// Get all body velocities as a flat [vx,vy,vz, ...] f64 array (zero-copy write).
///
/// # Safety
/// `out` must point to at least `body_count * 3` f64 values.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_get_velocities_batch(
    world: *const PhysicsWorld,
    out: *mut f64,
    out_capacity: u32,
) -> u8 {
    let w = match world.as_ref() {
        Some(w) => w,
        None => return 0,
    };
    if out.is_null() {
        return 0;
    }
    let n = w.bodies.len();
    if (out_capacity as usize) < n * 3 {
        return 0;
    }
    let buf = std::slice::from_raw_parts_mut(out, n * 3);
    for (i, body) in w.bodies.iter().enumerate() {
        buf[i * 3] = body.velocity.x.to_f64();
        buf[i * 3 + 1] = body.velocity.y.to_f64();
        buf[i * 3 + 2] = body.velocity.z.to_f64();
    }
    1
}

/// Set all body velocities from a flat [vx,vy,vz, ...] f64 array (batch update).
///
/// # Safety
/// `data` must point to at least `body_count * 3` f64 values.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_set_velocities_batch(
    world: *mut PhysicsWorld,
    data: *const f64,
    count: u32,
) -> u8 {
    let w = match world.as_mut() {
        Some(w) => w,
        None => return 0,
    };
    if data.is_null() {
        return 0;
    }
    let n = w.bodies.len();
    if (count as usize) < n * 3 {
        return 0;
    }
    let buf = std::slice::from_raw_parts(data, n * 3);
    for (i, body) in w.bodies.iter_mut().enumerate() {
        body.velocity = Vec3Fix::new(
            Fix128::from_f64(buf[i * 3]),
            Fix128::from_f64(buf[i * 3 + 1]),
            Fix128::from_f64(buf[i * 3 + 2]),
        );
    }
    1
}

/// Apply impulses to multiple bodies in batch.
///
/// `data` is a flat array of [body_id_as_f64, ix, iy, iz, ...] with `count/4` impulses.
///
/// # Safety
/// `data` must point to at least `count` f64 values, with `count` divisible by 4.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_apply_impulses_batch(
    world: *mut PhysicsWorld,
    data: *const f64,
    count: u32,
) -> u8 {
    let w = match world.as_mut() {
        Some(w) => w,
        None => return 0,
    };
    if data.is_null() || !count.is_multiple_of(4) {
        return 0;
    }
    let buf = std::slice::from_raw_parts(data, count as usize);
    let n_bodies = w.bodies.len();

    for chunk in buf.chunks_exact(4) {
        let body_id = chunk[0] as usize;
        if body_id >= n_bodies {
            continue;
        }
        let impulse = Vec3Fix::new(
            Fix128::from_f64(chunk[1]),
            Fix128::from_f64(chunk[2]),
            Fix128::from_f64(chunk[3]),
        );
        w.bodies[body_id].apply_impulse(impulse);
    }
    1
}

// ============================================================================
// Body management
// ============================================================================

/// Add a dynamic body. Returns body index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_add_dynamic(
    world: *mut PhysicsWorld,
    position: AliceVec3,
    mass: f64,
) -> u32 {
    match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_dynamic(position.to_vec3fix(), Fix128::from_f64(mass));
            w.add_body(body) as u32
        }
        None => u32::MAX,
    }
}

/// Add a static body. Returns body index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_add_static(
    world: *mut PhysicsWorld,
    position: AliceVec3,
) -> u32 {
    match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_static(position.to_vec3fix());
            w.add_body(body) as u32
        }
        None => u32::MAX,
    }
}

/// Add a sensor (trigger) body. Returns body index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_add_sensor(
    world: *mut PhysicsWorld,
    position: AliceVec3,
) -> u32 {
    match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_sensor(position.to_vec3fix());
            w.add_body(body) as u32
        }
        None => u32::MAX,
    }
}

/// Get body info (read-only snapshot).
///
/// # Safety
/// `world` must be a valid pointer. `out` must point to a valid `AliceBodyInfo`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_get_info(
    world: *const PhysicsWorld,
    body_id: u32,
    out: *mut AliceBodyInfo,
) -> u8 {
    let (w, o) = match (world.as_ref(), out.as_mut()) {
        (Some(w), Some(o)) => (w, o),
        _ => return 0,
    };
    match w.bodies.get(body_id as usize) {
        Some(b) => {
            o.position = AliceVec3::from_vec3fix(b.position);
            o.velocity = AliceVec3::from_vec3fix(b.velocity);
            o.angular_velocity = AliceVec3::from_vec3fix(b.angular_velocity);
            o.rotation = AliceQuat::from_quatfix(b.rotation);
            o.inv_mass = b.inv_mass.to_f64();
            o.is_static = b.is_static() as u8;
            o.is_sensor = b.is_sensor as u8;
            1
        }
        None => 0,
    }
}

/// Get body position.
///
/// # Safety
/// `world` must be a valid pointer. `out` must point to a valid `AliceVec3`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_get_position(
    world: *const PhysicsWorld,
    body_id: u32,
    out: *mut AliceVec3,
) -> u8 {
    let (w, o) = match (world.as_ref(), out.as_mut()) {
        (Some(w), Some(o)) => (w, o),
        _ => return 0,
    };
    match w.bodies.get(body_id as usize) {
        Some(b) => {
            *o = AliceVec3::from_vec3fix(b.position);
            1
        }
        None => 0,
    }
}

/// Set body position.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_position(
    world: *mut PhysicsWorld,
    body_id: u32,
    position: AliceVec3,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.position = position.to_vec3fix();
                1
            }
            None => 0,
        },
        None => 0,
    }
}

/// Get body velocity.
///
/// # Safety
/// `world` must be a valid pointer. `out` must point to a valid `AliceVec3`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_get_velocity(
    world: *const PhysicsWorld,
    body_id: u32,
    out: *mut AliceVec3,
) -> u8 {
    let (w, o) = match (world.as_ref(), out.as_mut()) {
        (Some(w), Some(o)) => (w, o),
        _ => return 0,
    };
    match w.bodies.get(body_id as usize) {
        Some(b) => {
            *o = AliceVec3::from_vec3fix(b.velocity);
            1
        }
        None => 0,
    }
}

/// Set body velocity.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_velocity(
    world: *mut PhysicsWorld,
    body_id: u32,
    velocity: AliceVec3,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.velocity = velocity.to_vec3fix();
                1
            }
            None => 0,
        },
        None => 0,
    }
}

/// Get body rotation as quaternion.
///
/// # Safety
/// `world` must be a valid pointer. `out` must point to a valid `AliceQuat`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_get_rotation(
    world: *const PhysicsWorld,
    body_id: u32,
    out: *mut AliceQuat,
) -> u8 {
    let (w, o) = match (world.as_ref(), out.as_mut()) {
        (Some(w), Some(o)) => (w, o),
        _ => return 0,
    };
    match w.bodies.get(body_id as usize) {
        Some(b) => {
            *o = AliceQuat::from_quatfix(b.rotation);
            1
        }
        None => 0,
    }
}

/// Set body restitution (bounciness, 0.0-1.0).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_restitution(
    world: *mut PhysicsWorld,
    body_id: u32,
    restitution: f64,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.restitution = Fix128::from_f64(restitution);
                1
            }
            None => 0,
        },
        None => 0,
    }
}

/// Set body friction coefficient.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_friction(
    world: *mut PhysicsWorld,
    body_id: u32,
    friction: f64,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.friction = Fix128::from_f64(friction);
                1
            }
            None => 0,
        },
        None => 0,
    }
}

/// Apply impulse at center of mass.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_apply_impulse(
    world: *mut PhysicsWorld,
    body_id: u32,
    impulse: AliceVec3,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.apply_impulse(impulse.to_vec3fix());
                1
            }
            None => 0,
        },
        None => 0,
    }
}

/// Apply impulse at a world-space point.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_apply_impulse_at(
    world: *mut PhysicsWorld,
    body_id: u32,
    impulse: AliceVec3,
    point: AliceVec3,
) -> u8 {
    match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.apply_impulse_at(impulse.to_vec3fix(), point.to_vec3fix());
                1
            }
            None => 0,
        },
        None => 0,
    }
}

// ============================================================================
// Config
// ============================================================================

/// Get default physics config.
#[no_mangle]
pub extern "C" fn alice_physics_config_default() -> AlicePhysicsConfig {
    let cfg = SolverConfig::default();
    AlicePhysicsConfig {
        substeps: cfg.substeps as u32,
        iterations: cfg.iterations as u32,
        gravity_x: cfg.gravity.x.to_f64(),
        gravity_y: cfg.gravity.y.to_f64(),
        gravity_z: cfg.gravity.z.to_f64(),
        damping: cfg.damping.to_f64(),
    }
}

/// Set gravity on an existing world.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_set_gravity(
    world: *mut PhysicsWorld,
    x: f64,
    y: f64,
    z: f64,
) {
    if let Some(w) = world.as_mut() {
        w.config.gravity = Vec3Fix::new(
            Fix128::from_f64(x),
            Fix128::from_f64(y),
            Fix128::from_f64(z),
        );
    }
}

/// Set substeps on an existing world.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_set_substeps(world: *mut PhysicsWorld, substeps: u32) {
    if let Some(w) = world.as_mut() {
        w.config.substeps = substeps as usize;
    }
}

// ============================================================================
// State serialization (for rollback netcode)
// ============================================================================

/// Serialize world state. Caller must free with `alice_physics_state_free`.
/// Returns data pointer and writes length to `out_len`.
///
/// # Safety
/// `world` must be a valid pointer. `out_len` must point to a valid `u32`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_state_serialize(
    world: *const PhysicsWorld,
    out_len: *mut u32,
) -> *mut u8 {
    let w = match world.as_ref() {
        Some(w) => w,
        None => {
            if let Some(len) = out_len.as_mut() {
                *len = 0;
            }
            return std::ptr::null_mut();
        }
    };
    let state = w.serialize_state();
    let len = state.len();
    if let Some(out) = out_len.as_mut() {
        *out = len as u32;
    }
    let boxed = state.into_boxed_slice();
    Box::into_raw(boxed) as *mut u8
}

/// Deserialize world state (restores from serialized snapshot).
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `world` must be a valid pointer. `data` must point to `len` valid bytes.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_state_deserialize(
    world: *mut PhysicsWorld,
    data: *const u8,
    len: u32,
) -> u8 {
    let w = match world.as_mut() {
        Some(w) => w,
        None => return 0,
    };
    if data.is_null() || len == 0 {
        return 0;
    }
    let slice = std::slice::from_raw_parts(data, len as usize);
    w.deserialize_state(slice) as u8
}

/// Free a serialized state buffer.
///
/// # Safety
/// `data` must be a pointer returned by `alice_physics_state_serialize` with matching `len`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_state_free(data: *mut u8, len: u32) {
    if !data.is_null() && len > 0 {
        let slice = std::slice::from_raw_parts_mut(data, len as usize);
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

// ============================================================================
// Version
// ============================================================================

/// Get library version string. Returns a static null-terminated string.
#[no_mangle]
pub extern "C" fn alice_physics_version() -> *const std::os::raw::c_char {
    c"0.3.0".as_ptr()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_conversion_roundtrip() {
        let original = Vec3Fix::new(
            Fix128::from_f64(1.5),
            Fix128::from_f64(-2.25),
            Fix128::from_f64(3.75),
        );
        let c = AliceVec3::from_vec3fix(original);
        let back = c.to_vec3fix();
        assert_eq!(original.x.hi, back.x.hi);
        assert_eq!(original.y.hi, back.y.hi);
        assert_eq!(original.z.hi, back.z.hi);
    }

    #[test]
    fn test_world_create_destroy() {
        unsafe {
            let world = alice_physics_world_create();
            assert!(!world.is_null());
            assert_eq!(alice_physics_world_body_count(world), 0);
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_world_destroy_null() {
        unsafe {
            alice_physics_world_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_world_create_with_config() {
        unsafe {
            let config = alice_physics_config_default();
            let world = alice_physics_world_create_with_config(config);
            assert!(!world.is_null());
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_body_add_and_get() {
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 { x: 1.0, y: 2.0, z: 3.0 };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);
            assert_eq!(id, 0);
            assert_eq!(alice_physics_world_body_count(world), 1);

            let mut out = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            let ok = alice_physics_body_get_position(world, id, &mut out);
            assert_eq!(ok, 1);
            assert!((out.x - 1.0).abs() < 1e-10);
            assert!((out.y - 2.0).abs() < 1e-10);
            assert!((out.z - 3.0).abs() < 1e-10);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_body_get_invalid_id() {
        unsafe {
            let world = alice_physics_world_create();
            let mut out = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            let ok = alice_physics_body_get_position(world, 999, &mut out);
            assert_eq!(ok, 0);
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_null_world_safety() {
        unsafe {
            let null: *mut PhysicsWorld = std::ptr::null_mut();
            assert_eq!(alice_physics_world_body_count(null), 0);
            assert_eq!(alice_physics_world_step(null, 0.016), 0);
            assert_eq!(alice_physics_world_step_n(null, 0.016, 10), 0);

            let pos = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            assert_eq!(alice_physics_body_add_dynamic(null, pos, 1.0), u32::MAX);
        }
    }

    #[test]
    fn test_step_and_gravity() {
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 { x: 0.0, y: 10.0, z: 0.0 };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);

            for _ in 0..60 {
                alice_physics_world_step(world, 1.0 / 60.0);
            }

            let mut out = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            alice_physics_body_get_position(world, id, &mut out);
            assert!(out.y < 10.0, "Body should fall under gravity, y={}", out.y);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_batch_positions() {
        unsafe {
            let world = alice_physics_world_create();
            let p1 = AliceVec3 { x: 1.0, y: 2.0, z: 3.0 };
            let p2 = AliceVec3 { x: 4.0, y: 5.0, z: 6.0 };
            alice_physics_body_add_static(world, p1);
            alice_physics_body_add_static(world, p2);

            let mut buf = [0.0f64; 6];
            let ok = alice_physics_world_get_positions_batch(world, buf.as_mut_ptr(), 6);
            assert_eq!(ok, 1);
            assert!((buf[0] - 1.0).abs() < 1e-10);
            assert!((buf[1] - 2.0).abs() < 1e-10);
            assert!((buf[2] - 3.0).abs() < 1e-10);
            assert!((buf[3] - 4.0).abs() < 1e-10);
            assert!((buf[4] - 5.0).abs() < 1e-10);
            assert!((buf[5] - 6.0).abs() < 1e-10);

            // Insufficient capacity
            let fail = alice_physics_world_get_positions_batch(world, buf.as_mut_ptr(), 3);
            assert_eq!(fail, 0);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_state_serialization_ffi() {
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 { x: 0.0, y: 10.0, z: 0.0 };
            alice_physics_body_add_dynamic(world, pos, 1.0);

            // Serialize
            let mut len: u32 = 0;
            let data = alice_physics_state_serialize(world, &mut len);
            assert!(!data.is_null());
            assert!(len > 0);

            // Deserialize into same world
            let ok = alice_physics_state_deserialize(world, data, len);
            assert_eq!(ok, 1);

            // Free
            alice_physics_state_free(data, len);
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_impulse_application() {
        unsafe {
            let world = alice_physics_world_create();
            alice_physics_world_set_gravity(world, 0.0, 0.0, 0.0);
            let pos = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);

            let impulse = AliceVec3 { x: 10.0, y: 0.0, z: 0.0 };
            alice_physics_body_apply_impulse(world, id, impulse);
            alice_physics_world_step(world, 1.0 / 60.0);

            let mut out = AliceVec3 { x: 0.0, y: 0.0, z: 0.0 };
            alice_physics_body_get_position(world, id, &mut out);
            assert!(out.x > 0.0, "Impulse should move body right, x={}", out.x);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_body_info() {
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 { x: 1.0, y: 2.0, z: 3.0 };
            let id = alice_physics_body_add_sensor(world, pos);

            let mut info = std::mem::zeroed::<AliceBodyInfo>();
            let ok = alice_physics_body_get_info(world, id, &mut info);
            assert_eq!(ok, 1);
            assert_eq!(info.is_sensor, 1);
            assert!((info.position.x - 1.0).abs() < 1e-10);

            alice_physics_world_destroy(world);
        }
    }
}
