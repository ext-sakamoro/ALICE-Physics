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

use crate::binding_api;
use crate::joint::{BallJoint, FixedJoint, HingeJoint, Joint, SliderJoint, SpringJoint};
use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::{PhysicsWorld, RigidBody, SolverConfig};

// ============================================================================
// Panic isolation (1.2.0)
// ============================================================================

/// Message of the most recent panic caught at the FFI boundary on this
/// thread; read with [`alice_physics_last_error`], cleared with
/// [`alice_physics_clear_last_error`].
///
/// A panic that reaches an `extern "C"` frame aborts the process on Rust
/// 1.81+ and takes the host (Unity, Unreal, a Python interpreter) down with
/// it. Every exported function therefore runs its body through
/// [`ffi_guard`]: a panic is caught inside the function, its message stored
/// here, and the function returns its documented sentinel (`0`, `u32::MAX`,
/// null). Before 1.2.0 only `alice_physics_world_step` / `_step_n` were
/// guarded (31 exported functions, 29 raw).
mod guard {
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    thread_local! {
        static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    /// Record an error message for `alice_physics_last_error`.
    pub fn set_last_error(msg: impl Into<String>) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg.into()));
    }

    /// Take the most recent error message (leaves the slot empty).
    pub fn take_last_error() -> Option<String> {
        LAST_ERROR.with(|slot| slot.borrow_mut().take())
    }

    /// Clear the most recent error message.
    pub fn clear_last_error() {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
    }

    /// Run `body`, converting a panic into `default` plus a recorded message.
    ///
    /// The closure is treated as unwind-safe: every FFI body only touches its
    /// arguments and the world behind the caller's pointer, and a
    /// `PhysicsWorld` that panicked mid-step is left in whatever state the
    /// panic found it in — the host is told (return sentinel + message) and
    /// should destroy the world rather than keep stepping it.
    #[inline]
    pub fn ffi_guard<T>(default: T, body: impl FnOnce() -> T) -> T {
        match catch_unwind(AssertUnwindSafe(body)) {
            Ok(v) => v,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with non-string payload".to_string());
                set_last_error(format!("alice-physics FFI panic: {msg}"));
                default
            }
        }
    }
}

use guard::ffi_guard;

/// Most recent panic message caught at the FFI boundary on this thread, as a
/// heap-allocated C string, or null when there is none. Free it with
/// [`alice_physics_string_free`]. Reading takes the message (a second call
/// returns null until the next error).
#[no_mangle]
pub extern "C" fn alice_physics_last_error() -> *mut std::os::raw::c_char {
    ffi_guard(std::ptr::null_mut(), || match guard::take_last_error() {
        Some(msg) => std::ffi::CString::new(msg.replace('\0', " "))
            .map_or(std::ptr::null_mut(), std::ffi::CString::into_raw),
        None => std::ptr::null_mut(),
    })
}

/// Discard the most recent FFI panic message on this thread.
#[no_mangle]
pub extern "C" fn alice_physics_clear_last_error() {
    ffi_guard((), guard::clear_last_error);
}

/// Free a string returned by [`alice_physics_last_error`].
///
/// # Safety
/// `s` must be null or a pointer returned by `alice_physics_last_error` that
/// has not been freed yet.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_string_free(s: *mut std::os::raw::c_char) {
    ffi_guard((), || {
        if !s.is_null() {
            drop(std::ffi::CString::from_raw(s));
        }
    });
}

// ============================================================================
// C-compatible types
// ============================================================================

/// C-compatible 3D vector (f64 for FFI boundary, converted to Fix128 internally)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
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
    fn to_vec3fix(self) -> Vec3Fix {
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
    ffi_guard(std::ptr::null_mut(), || {
        let world = PhysicsWorld::new(SolverConfig::default());
        Box::into_raw(Box::new(world))
    })
}

/// Create a physics world with custom config.
#[no_mangle]
pub extern "C" fn alice_physics_world_create_with_config(
    config: AlicePhysicsConfig,
) -> *mut PhysicsWorld {
    ffi_guard(std::ptr::null_mut(), || {
        let solver_config = SolverConfig {
            substeps: config.substeps as usize,
            iterations: config.iterations as usize,
            gravity: Vec3Fix::new(
                Fix128::from_f64(config.gravity_x),
                Fix128::from_f64(config.gravity_y),
                Fix128::from_f64(config.gravity_z),
            ),
            damping: Fix128::from_f64(config.damping),
            ..Default::default()
        };
        let world = PhysicsWorld::new(solver_config);
        Box::into_raw(Box::new(world))
    })
}

/// Destroy a physics world.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_destroy(world: *mut PhysicsWorld) {
    ffi_guard((), || {
        if !world.is_null() {
            drop(Box::from_raw(world));
        }
    })
}

/// Step the simulation by dt seconds (as f64, converted to Fix128).
///
/// Returns 1 on success, 0 on failure (null pointer or internal panic).
///
/// # Safety
/// `world` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_step(world: *mut PhysicsWorld, dt: f64) -> u8 {
    ffi_guard(0, || {
        let w = match world.as_mut() {
            Some(w) => w as *mut PhysicsWorld,
            None => return 0,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (*w).step(Fix128::from_f64(dt));
        }));
        result.is_ok() as u8
    })
}

/// Step the simulation N times with fixed dt (batch stepping).
///
/// Amortizes FFI overhead — ideal for training loops and rollback re-simulation.
/// Returns 1 on success, 0 on failure.
///
/// # Safety
/// `world` must be a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_step_n(
    world: *mut PhysicsWorld,
    dt: f64,
    steps: u32,
) -> u8 {
    ffi_guard(0, || {
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
    })
}

/// Get the number of bodies.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`, or null.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_body_count(world: *const PhysicsWorld) -> u32 {
    ffi_guard(0, || match world.as_ref() {
        Some(w) => w.bodies.len() as u32,
        None => 0,
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || {
        let w = match world.as_mut() {
            Some(w) => w,
            None => return 0,
        };
        if data.is_null() || count % 4 != 0 {
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
    })
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
    ffi_guard(u32::MAX, || match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_dynamic(position.to_vec3fix(), Fix128::from_f64(mass));
            w.add_body(body) as u32
        }
        None => u32::MAX,
    })
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
    ffi_guard(u32::MAX, || match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_static(position.to_vec3fix());
            w.add_body(body) as u32
        }
        None => u32::MAX,
    })
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
    ffi_guard(u32::MAX, || match world.as_mut() {
        Some(w) => {
            let body = RigidBody::new_sensor(position.to_vec3fix());
            w.add_body(body) as u32
        }
        None => u32::MAX,
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || {
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
    })
}

/// Raw Fix128 pair (hi:i64, lo:u64) exposed across the C ABI so
/// Unity/UE5 hosts can pin down determinism to the exact bit-pattern
/// used by the Rust solver (Phase F 11.3 byte-for-byte contract).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliceFix128Raw {
    /// High 64 bits (integer part).
    pub hi: i64,
    /// Low 64 bits (fractional part).
    pub lo: u64,
}

/// Raw Fix128 3-tuple exposed for byte-for-byte determinism contracts.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliceVec3Fix128Raw {
    /// X component (raw Fix128 hi/lo pair).
    pub x: AliceFix128Raw,
    /// Y component (raw Fix128 hi/lo pair).
    pub y: AliceFix128Raw,
    /// Z component (raw Fix128 hi/lo pair).
    pub z: AliceFix128Raw,
}

/// Get body position as the raw Fix128 hi/lo pair (Phase F 11.3
/// byte-for-byte determinism contract). Unity/UE5 hosts assert on
/// the exact `(hi, lo)` pair to guarantee that every solver step
/// produces identical state on every platform.
///
/// Returns 1 on success, 0 on invalid pointer or body index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
/// `out` must be a valid pointer to writable `AliceVec3Fix128Raw`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_get_position_fix128_raw(
    world: *const PhysicsWorld,
    body_id: u32,
    out: *mut AliceVec3Fix128Raw,
) -> u8 {
    ffi_guard(0, || {
        let (w, o) = match (world.as_ref(), out.as_mut()) {
            (Some(w), Some(o)) => (w, o),
            _ => return 0,
        };
        match w.bodies.get(body_id as usize) {
            Some(b) => {
                *o = AliceVec3Fix128Raw {
                    x: AliceFix128Raw {
                        hi: b.position.x.hi,
                        lo: b.position.x.lo,
                    },
                    y: AliceFix128Raw {
                        hi: b.position.y.hi,
                        lo: b.position.y.lo,
                    },
                    z: AliceFix128Raw {
                        hi: b.position.z.hi,
                        lo: b.position.z.lo,
                    },
                };
                1
            }
            None => 0,
        }
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.position = position.to_vec3fix();
                1
            }
            None => 0,
        },
        None => 0,
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.velocity = velocity.to_vec3fix();
                1
            }
            None => 0,
        },
        None => 0,
    })
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
    ffi_guard(0, || {
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
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.restitution = Fix128::from_f64(restitution);
                1
            }
            None => 0,
        },
        None => 0,
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.friction = Fix128::from_f64(friction);
                1
            }
            None => 0,
        },
        None => 0,
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.apply_impulse(impulse.to_vec3fix());
                1
            }
            None => 0,
        },
        None => 0,
    })
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
    ffi_guard(0, || match world.as_mut() {
        Some(w) => match w.bodies.get_mut(body_id as usize) {
            Some(b) => {
                b.apply_impulse_at(impulse.to_vec3fix(), point.to_vec3fix());
                1
            }
            None => 0,
        },
        None => 0,
    })
}

// ============================================================================
// Config
// ============================================================================

/// Get default physics config.
#[no_mangle]
pub extern "C" fn alice_physics_config_default() -> AlicePhysicsConfig {
    // panic 時は全 0 (substeps 0 は solver 側で無効 config として扱われる)
    ffi_guard(
        AlicePhysicsConfig {
            substeps: 0,
            iterations: 0,
            gravity_x: 0.0,
            gravity_y: 0.0,
            gravity_z: 0.0,
            damping: 0.0,
        },
        || {
            let cfg = SolverConfig::default();
            AlicePhysicsConfig {
                substeps: cfg.substeps as u32,
                iterations: cfg.iterations as u32,
                gravity_x: cfg.gravity.x.to_f64(),
                gravity_y: cfg.gravity.y.to_f64(),
                gravity_z: cfg.gravity.z.to_f64(),
                damping: cfg.damping.to_f64(),
            }
        },
    )
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
    ffi_guard((), || {
        if let Some(w) = world.as_mut() {
            w.config.gravity = Vec3Fix::new(
                Fix128::from_f64(x),
                Fix128::from_f64(y),
                Fix128::from_f64(z),
            );
        }
    })
}

/// Set substeps on an existing world.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_world_set_substeps(world: *mut PhysicsWorld, substeps: u32) {
    ffi_guard((), || {
        if let Some(w) = world.as_mut() {
            w.config.substeps = substeps as usize;
        }
    })
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
    ffi_guard(std::ptr::null_mut(), || {
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
    })
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
    ffi_guard(0, || {
        let w = match world.as_mut() {
            Some(w) => w,
            None => return 0,
        };
        if data.is_null() || len == 0 {
            return 0;
        }
        let slice = std::slice::from_raw_parts(data, len as usize);
        w.deserialize_state(slice) as u8
    })
}

/// Free a serialized state buffer.
///
/// # Safety
/// `data` must be a pointer returned by `alice_physics_state_serialize` with matching `len`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_state_free(data: *mut u8, len: u32) {
    ffi_guard((), || {
        if !data.is_null() && len > 0 {
            let slice = std::slice::from_raw_parts_mut(data, len as usize);
            drop(Box::from_raw(slice as *mut [u8]));
        }
    })
}

// ============================================================================
// Version
// ============================================================================

/// Get library version string. Returns a static null-terminated string.
///
/// Always equals `CARGO_PKG_VERSION` of the built crate (v1.0.1 fix: the
/// string was a hardcoded `"0.6.0"` before).
#[no_mangle]
pub extern "C" fn alice_physics_version() -> *const std::os::raw::c_char {
    ffi_guard(std::ptr::null(), || {
        // `concat!` + `env!` keeps this a `&'static str` with an explicit NUL;
        // no C-string literal so the `ffi` feature stays within the 1.70 MSRV.
        const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
        VERSION.as_ptr().cast()
    })
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Collision radius, shapes, static colliders, joints
// ============================================================================

/// A collision shape for [`alice_physics_body_add_shaped`] /
/// [`alice_physics_body_set_shape`]. `kind` picks the shape and `a`, `b`, `c`
/// its sizes (unused sizes are ignored):
///
/// | kind | shape     | a            | b           | c          |
/// |------|-----------|--------------|-------------|------------|
/// | 0    | box       | half x       | half y      | half z     |
/// | 1    | cylinder  | radius       | half height |            |
/// | 2    | cone      | radius       | half height |            |
/// | 3    | ellipsoid | radius x     | radius y    | radius z   |
/// | 4    | wedge     | width        | height      | depth      |
/// | 5    | torus     | major radius | minor radius|            |
///
/// Every used size must be finite and positive; a torus needs
/// `minor < major`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlicePhysicsShape {
    /// Shape kind (0 box … 5 torus, see the table above).
    pub kind: u32,
    /// First size.
    pub a: f64,
    /// Second size.
    pub b: f64,
    /// Third size.
    pub c: f64,
}

impl AlicePhysicsShape {
    fn to_shape(self) -> Option<crate::shape::Shape> {
        binding_api::shape(self.kind, self.a, self.b, self.c)
    }
}

fn vec3_arg(v: &AliceVec3) -> Option<Vec3Fix> {
    binding_api::vec3(v.x, v.y, v.z)
}

fn index_result(r: Option<usize>) -> u32 {
    r.and_then(|i| u32::try_from(i).ok()).unwrap_or(u32::MAX)
}

/// Set a body's collision sphere radius. Returns 1 on success, 0 for a null
/// world, an unknown body or a radius that is not finite and positive.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_collision_radius(
    world: *mut PhysicsWorld,
    body_id: u32,
    radius: f64,
) -> u8 {
    ffi_guard(0, || match world.as_mut() {
        Some(w) => u8::from(binding_api::set_collision_radius(
            w,
            body_id as usize,
            radius,
        )),
        None => 0,
    })
}

/// Drop a body's own collision radius; it falls back to the world default.
/// Returns 1 on success, 0 for a null world or an unknown body.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_clear_collision_radius(
    world: *mut PhysicsWorld,
    body_id: u32,
) -> u8 {
    ffi_guard(0, || match world.as_mut() {
        Some(w) => u8::from(binding_api::clear_collision_radius(w, body_id as usize)),
        None => 0,
    })
}

/// Add a dynamic body with a collision shape; its mass and inertia come from
/// `density`. Returns the body index, or `u32::MAX` for a null world, an
/// invalid shape, a density that is not finite and positive, or a
/// non-finite position.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_add_shaped(
    world: *mut PhysicsWorld,
    shape: AlicePhysicsShape,
    density: f64,
    position: AliceVec3,
) -> u32 {
    ffi_guard(u32::MAX, || {
        let Some(w) = world.as_mut() else {
            return u32::MAX;
        };
        let (Some(s), Some(p)) = (shape.to_shape(), vec3_arg(&position)) else {
            return u32::MAX;
        };
        index_result(binding_api::add_shaped_body(w, s, density, p))
    })
}

/// Give an existing body a collision shape (its mass is unchanged). Returns 1
/// on success, 0 for a null world, an unknown body or an invalid shape.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_body_set_shape(
    world: *mut PhysicsWorld,
    body_id: u32,
    shape: AlicePhysicsShape,
) -> u8 {
    ffi_guard(0, || {
        let (Some(w), Some(s)) = (world.as_mut(), shape.to_shape()) else {
            return 0;
        };
        u8::from(binding_api::set_body_shape(w, body_id as usize, s))
    })
}

/// Add the static plane `normal · p = offset` (the normal is normalised).
/// Returns the static collider index, or `u32::MAX` for a null world, a zero
/// or non-finite normal, or a non-finite offset.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_static_add_plane(
    world: *mut PhysicsWorld,
    normal: AliceVec3,
    offset: f64,
) -> u32 {
    ffi_guard(u32::MAX, || {
        let (Some(w), Some(n)) = (world.as_mut(), vec3_arg(&normal)) else {
            return u32::MAX;
        };
        index_result(binding_api::add_static_plane(w, n, offset))
    })
}

/// Add a static height field: `width × depth` heights (row-major, `x`
/// fastest, both at least 2) spaced `spacing` apart from the min corner
/// `origin`. Returns the static collider index, or `u32::MAX` for a null
/// pointer, a size below 2, a non-positive spacing or a non-finite value.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
/// `heights` must point to `width * depth` readable `f64`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_static_add_heightfield(
    world: *mut PhysicsWorld,
    heights: *const f64,
    width: u32,
    depth: u32,
    spacing: f64,
    origin: AliceVec3,
) -> u32 {
    ffi_guard(u32::MAX, || {
        let (Some(w), Some(o)) = (world.as_mut(), vec3_arg(&origin)) else {
            return u32::MAX;
        };
        let Some(count) = (width as usize).checked_mul(depth as usize) else {
            return u32::MAX;
        };
        if heights.is_null() || count == 0 {
            return u32::MAX;
        }
        let hs = std::slice::from_raw_parts(heights, count);
        index_result(binding_api::add_static_heightfield(
            w, hs, width, depth, spacing, o,
        ))
    })
}

/// Add a static triangle mesh: `vertices` holds `vertex_count` `x, y, z`
/// triples, `indices` holds `index_count` vertex indices (three per
/// triangle). Returns the static collider index, or `u32::MAX` for a null
/// pointer, an empty mesh, an index count that is not a multiple of 3, an
/// index past the last vertex or a non-finite coordinate.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
/// `vertices` must point to `3 * vertex_count` readable `f64` and `indices`
/// to `index_count` readable `u32`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_static_add_trimesh(
    world: *mut PhysicsWorld,
    vertices: *const f64,
    vertex_count: u32,
    indices: *const u32,
    index_count: u32,
) -> u32 {
    ffi_guard(u32::MAX, || {
        let Some(w) = world.as_mut() else {
            return u32::MAX;
        };
        if vertices.is_null() || indices.is_null() || vertex_count == 0 || index_count == 0 {
            return u32::MAX;
        }
        let Some(coords) = (vertex_count as usize).checked_mul(3) else {
            return u32::MAX;
        };
        let vs = std::slice::from_raw_parts(vertices, coords);
        let is = std::slice::from_raw_parts(indices, index_count as usize);
        index_result(binding_api::add_static_trimesh(w, vs, is))
    })
}

/// Remove static collider `index`; later colliders shift down by one.
/// Returns 1 on success, 0 for a null world or an unknown index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_static_remove(world: *mut PhysicsWorld, index: u32) -> u8 {
    ffi_guard(0, || match world.as_mut() {
        Some(w) => u8::from(binding_api::remove_static_collider(w, index as usize)),
        None => 0,
    })
}

/// Number of static colliders (0 for a null world).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_static_count(world: *const PhysicsWorld) -> u32 {
    ffi_guard(0, || match world.as_ref() {
        Some(w) => u32::try_from(w.static_collider_count()).unwrap_or(u32::MAX),
        None => 0,
    })
}

unsafe fn add_joint_ffi(world: *mut PhysicsWorld, make: impl FnOnce() -> Option<Joint>) -> u32 {
    ffi_guard(u32::MAX, || {
        let Some(w) = world.as_mut() else {
            return u32::MAX;
        };
        match make() {
            Some(j) => index_result(binding_api::add_joint(w, j)),
            None => u32::MAX,
        }
    })
}

/// Add a ball-and-socket joint: `anchor_a` on body A meets `anchor_b` on body
/// B (both in body-local coordinates). Returns the joint index, or `u32::MAX`
/// for a null world, an unknown body, `body_a == body_b` or a non-finite
/// anchor.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_add_ball(
    world: *mut PhysicsWorld,
    body_a: u32,
    body_b: u32,
    anchor_a: AliceVec3,
    anchor_b: AliceVec3,
) -> u32 {
    add_joint_ffi(world, || {
        Some(Joint::Ball(BallJoint::new(
            body_a as usize,
            body_b as usize,
            vec3_arg(&anchor_a)?,
            vec3_arg(&anchor_b)?,
        )))
    })
}

/// Add a hinge joint: the anchors meet and `axis_a` (body A local) stays
/// aligned with `axis_b` (body B local). Returns the joint index or
/// `u32::MAX` (see [`alice_physics_joint_add_ball`]; the axes must also be
/// non-zero).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_add_hinge(
    world: *mut PhysicsWorld,
    body_a: u32,
    body_b: u32,
    anchor_a: AliceVec3,
    anchor_b: AliceVec3,
    axis_a: AliceVec3,
    axis_b: AliceVec3,
) -> u32 {
    add_joint_ffi(world, || {
        Some(Joint::Hinge(HingeJoint::new(
            body_a as usize,
            body_b as usize,
            vec3_arg(&anchor_a)?,
            vec3_arg(&anchor_b)?,
            vec3_arg(&axis_a)?.try_normalize()?,
            vec3_arg(&axis_b)?.try_normalize()?,
        )))
    })
}

/// Add a fixed (weld) joint: the anchors meet and body B keeps the rotation
/// `relative_rotation` relative to body A (normalised; must be non-zero).
/// Returns the joint index or `u32::MAX` (see
/// [`alice_physics_joint_add_ball`]).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_add_fixed(
    world: *mut PhysicsWorld,
    body_a: u32,
    body_b: u32,
    anchor_a: AliceVec3,
    anchor_b: AliceVec3,
    relative_rotation: AliceQuat,
) -> u32 {
    add_joint_ffi(world, || {
        let q = &relative_rotation;
        Some(Joint::Fixed(FixedJoint::new(
            body_a as usize,
            body_b as usize,
            vec3_arg(&anchor_a)?,
            vec3_arg(&anchor_b)?,
            binding_api::unit_quat(q.x, q.y, q.z, q.w)?,
        )))
    })
}

/// Add a slider (prismatic) joint along `axis` (body A local, normalised;
/// must be non-zero). Returns the joint index or `u32::MAX` (see
/// [`alice_physics_joint_add_ball`]).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_add_slider(
    world: *mut PhysicsWorld,
    body_a: u32,
    body_b: u32,
    axis: AliceVec3,
    anchor_a: AliceVec3,
    anchor_b: AliceVec3,
) -> u32 {
    add_joint_ffi(world, || {
        Some(Joint::Slider(SliderJoint::new(
            body_a as usize,
            body_b as usize,
            vec3_arg(&axis)?.try_normalize()?,
            vec3_arg(&anchor_a)?,
            vec3_arg(&anchor_b)?,
        )))
    })
}

/// Add a spring between the anchors with rest length `rest_length`,
/// `stiffness` and `damping` (rest length and damping finite and not
/// negative, stiffness finite and positive). Returns the joint index or
/// `u32::MAX` (see [`alice_physics_joint_add_ball`]).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_add_spring(
    world: *mut PhysicsWorld,
    body_a: u32,
    body_b: u32,
    anchor_a: AliceVec3,
    anchor_b: AliceVec3,
    rest_length: f64,
    stiffness: f64,
    damping: f64,
) -> u32 {
    add_joint_ffi(world, || {
        Some(Joint::Spring(SpringJoint::new(
            body_a as usize,
            body_b as usize,
            vec3_arg(&anchor_a)?,
            vec3_arg(&anchor_b)?,
            binding_api::non_negative(rest_length)?,
            binding_api::positive(stiffness)?,
            binding_api::non_negative(damping)?,
        )))
    })
}

/// Remove joint `index`. The last joint moves into `index` (its index
/// changes). Returns 1 on success, 0 for a null world or an unknown index.
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_remove(world: *mut PhysicsWorld, index: u32) -> u8 {
    ffi_guard(0, || match world.as_mut() {
        Some(w) => u8::from(binding_api::remove_joint(w, index as usize)),
        None => 0,
    })
}

/// Number of joints (0 for a null world).
///
/// # Safety
/// `world` must be a valid pointer from `alice_physics_world_create*`.
#[no_mangle]
pub unsafe extern "C" fn alice_physics_joint_count(world: *const PhysicsWorld) -> u32 {
    ffi_guard(0, || match world.as_ref() {
        Some(w) => u32::try_from(w.joint_count()).unwrap_or(u32::MAX),
        None => 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_string_matches_cargo_pkg_version() {
        // SAFETY: `alice_physics_version` returns a NUL-terminated static
        // string owned by the binary.
        let s = unsafe { std::ffi::CStr::from_ptr(alice_physics_version()) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

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
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            assert!(!world.is_null());
            assert_eq!(alice_physics_world_body_count(world), 0);
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_world_destroy_null() {
        // SAFETY: Null pointer is explicitly handled by the C API.
        unsafe {
            alice_physics_world_destroy(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_world_create_with_config() {
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let config = alice_physics_config_default();
            let world = alice_physics_world_create_with_config(config);
            assert!(!world.is_null());
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_body_add_and_get() {
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);
            assert_eq!(id, 0);
            assert_eq!(alice_physics_world_body_count(world), 1);

            let mut out = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
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
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let mut out = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            let ok = alice_physics_body_get_position(world, 999, &mut out);
            assert_eq!(ok, 0);
            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_null_world_safety() {
        // SAFETY: Null pointers are explicitly handled by the C API.
        unsafe {
            let null: *mut PhysicsWorld = std::ptr::null_mut();
            assert_eq!(alice_physics_world_body_count(null), 0);
            assert_eq!(alice_physics_world_step(null, 0.016), 0);
            assert_eq!(alice_physics_world_step_n(null, 0.016, 10), 0);

            let pos = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            assert_eq!(alice_physics_body_add_dynamic(null, pos, 1.0), u32::MAX);
        }
    }

    #[test]
    fn test_step_and_gravity() {
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);

            for _ in 0..60 {
                alice_physics_world_step(world, 1.0 / 60.0);
            }

            let mut out = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            alice_physics_body_get_position(world, id, &mut out);
            assert!(out.y < 10.0, "Body should fall under gravity, y={}", out.y);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_batch_positions() {
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let p1 = AliceVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            let p2 = AliceVec3 {
                x: 4.0,
                y: 5.0,
                z: 6.0,
            };
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
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            };
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
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            alice_physics_world_set_gravity(world, 0.0, 0.0, 0.0);
            let pos = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            let id = alice_physics_body_add_dynamic(world, pos, 1.0);

            let impulse = AliceVec3 {
                x: 10.0,
                y: 0.0,
                z: 0.0,
            };
            alice_physics_body_apply_impulse(world, id, impulse);
            alice_physics_world_step(world, 1.0 / 60.0);

            let mut out = AliceVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            alice_physics_body_get_position(world, id, &mut out);
            assert!(out.x > 0.0, "Impulse should move body right, x={}", out.x);

            alice_physics_world_destroy(world);
        }
    }

    #[test]
    fn test_body_info() {
        // SAFETY: All pointers are created/destroyed within this test scope.
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            };
            let id = alice_physics_body_add_sensor(world, pos);

            let mut info = std::mem::zeroed::<AliceBodyInfo>();
            let ok = alice_physics_body_get_info(world, id, &mut info);
            assert_eq!(ok, 1);
            assert_eq!(info.is_sensor, 1);
            assert!((info.position.x - 1.0).abs() < 1e-10);

            alice_physics_world_destroy(world);
        }
    }

    /// FFI contract test (Phase F 11.3): verifies deterministic
    /// gravity fall through the C ABI so Unity/UE5 host integrations
    /// can be checked against a Rust-side reference.
    ///
    /// The Rust side runs the reference invocation and asserts on
    /// bracketing tolerances that describe the current behaviour;
    /// Unity/UE5 binding tests must reproduce the same
    /// `(x, y, z)` final position to within these brackets. Once
    /// the Fix128 output is pinned in a follow-up, the tolerances
    /// will be replaced with byte-for-byte assertions.
    #[test]
    fn ffi_contract_gravity_fall_deterministic() {
        // SAFETY: All FFI pointers are created and destroyed within
        // this test scope; no aliasing occurs because no other test
        // touches the returned world pointer.
        unsafe {
            let world = alice_physics_world_create();
            let pos = AliceVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            };
            let mass = 1.0_f64;
            let id = alice_physics_body_add_dynamic(world, pos, mass);

            // 60 steps at 60 Hz = 1 s under default gravity.
            for _ in 0..60 {
                let _ = alice_physics_world_step(world, 1.0 / 60.0);
            }

            let mut info = std::mem::zeroed::<AliceBodyInfo>();
            let ok = alice_physics_body_get_info(world, id, &mut info);
            assert_eq!(ok, 1);

            // Reference contract:
            // - Body must fall (y strictly less than starting height).
            // - No horizontal drift under vertical gravity.
            // - Final y is bounded below by a conservative lower limit
            //   (well within the physically plausible range for 1 s of
            //   uniform gravity ≈ -9.81 m/s² over one-second flight).
            assert!(
                info.position.y < 10.0,
                "body must fall under gravity, got y={}",
                info.position.y
            );
            assert!(
                info.position.y > -20.0,
                "final y out of expected range, got y={}",
                info.position.y
            );
            assert!(
                info.position.x.abs() < 1e-9,
                "no horizontal drift expected, got x={}",
                info.position.x
            );
            assert!(
                info.position.z.abs() < 1e-9,
                "no horizontal drift expected, got z={}",
                info.position.z
            );

            // Phase F 11.3 body: byte-for-byte determinism gate.
            // Read the Fix128 hi/lo raw pair via the dedicated FFI
            // getter and verify that a second identical simulation
            // produces the exact same hi/lo pair — this is the runtime
            // contract that Unity/UE5 host bindings must satisfy.
            let mut raw = std::mem::zeroed::<AliceVec3Fix128Raw>();
            let ok_raw = alice_physics_body_get_position_fix128_raw(world, id, &mut raw);
            assert_eq!(ok_raw, 1, "raw fix128 getter must succeed");

            let world2 = alice_physics_world_create();
            let pos2 = AliceVec3 {
                x: 0.0,
                y: 10.0,
                z: 0.0,
            };
            let id2 = alice_physics_body_add_dynamic(world2, pos2, mass);
            for _ in 0..60 {
                let _ = alice_physics_world_step(world2, 1.0 / 60.0);
            }
            let mut raw2 = std::mem::zeroed::<AliceVec3Fix128Raw>();
            let ok_raw2 = alice_physics_body_get_position_fix128_raw(world2, id2, &mut raw2);
            assert_eq!(
                ok_raw2, 1,
                "raw fix128 getter must succeed on the replay world"
            );
            alice_physics_world_destroy(world2);

            assert_eq!(raw.x.hi, raw2.x.hi, "byte-for-byte determinism (x.hi)");
            assert_eq!(raw.x.lo, raw2.x.lo, "byte-for-byte determinism (x.lo)");
            assert_eq!(raw.y.hi, raw2.y.hi, "byte-for-byte determinism (y.hi)");
            assert_eq!(raw.y.lo, raw2.y.lo, "byte-for-byte determinism (y.lo)");
            assert_eq!(raw.z.hi, raw2.z.hi, "byte-for-byte determinism (z.hi)");
            assert_eq!(raw.z.lo, raw2.z.lo, "byte-for-byte determinism (z.lo)");

            // Golden pin: the exact raw pair produced by this scenario
            // (dynamic body at y=10, mass 1, 60 steps of 1/60 s under default
            // gravity). Unity / UE5 host bindings replay the same scenario and
            // must read back these six values; any solver-side change that
            // moves them is a determinism break and must bump the golden
            // together with a CHANGELOG entry.
            // v1.0.1 pin: (8, 6_631_112_686_738_674_129) = y 8.36 — the
            // per-substep damping bug (terminal velocity ≈ 2 m/s).
            // 1.2.0 pin: y = 10 − 4.1406 = 5.8594, the discrete closed form
            // of frame damping with 8 substeps (`tests/analytic_physics.rs`,
            // `default_config_free_fall_reaches_analytic_within_frame_damping`).
            const GOLDEN: [(i64, u64); 3] = [(0, 0), (5, 15_853_912_786_096_156_128), (0, 0)];
            let y = raw.y.hi as f64 + raw.y.lo as f64 / (1u128 << 64) as f64;
            assert!((y - 5.859_4).abs() < 1e-3, "FFI free fall y = {y}");
            assert_eq!(
                [
                    (raw.x.hi, raw.x.lo),
                    (raw.y.hi, raw.y.lo),
                    (raw.z.hi, raw.z.lo)
                ],
                GOLDEN,
                "FFI free-fall golden raw Fix128 pair drifted"
            );

            alice_physics_world_destroy(world);
        }
    }

    // ---- panic isolation (1.2.0) ----------------------------------------

    #[test]
    fn ffi_guard_turns_panic_into_sentinel_and_message() {
        guard::clear_last_error();
        let v = ffi_guard(u32::MAX, || {
            if true {
                panic!("boom {}", 42);
            }
            7
        });
        assert_eq!(v, u32::MAX);
        let msg = guard::take_last_error().expect("message recorded");
        assert!(msg.contains("boom 42"), "{msg}");
        assert!(guard::take_last_error().is_none(), "take clears the slot");
        assert_eq!(ffi_guard(0u8, || 1u8), 1);
        assert!(guard::take_last_error().is_none());
    }

    #[test]
    fn last_error_ffi_round_trips_through_c_string() {
        guard::clear_last_error();
        assert!(alice_physics_last_error().is_null());
        guard::set_last_error("alice-physics FFI panic: index 7 out of range");
        let p = alice_physics_last_error();
        assert!(!p.is_null());
        let text = unsafe { std::ffi::CStr::from_ptr(p) }
            .to_str()
            .expect("utf-8")
            .to_owned();
        assert_eq!(text, "alice-physics FFI panic: index 7 out of range");
        unsafe { alice_physics_string_free(p) };
        // take semantics: gone after one read
        assert!(alice_physics_last_error().is_null());
        guard::set_last_error("x");
        alice_physics_clear_last_error();
        assert!(alice_physics_last_error().is_null());
        // null is a no-op for the free
        unsafe { alice_physics_string_free(std::ptr::null_mut()) };
    }

    #[test]
    fn hostile_inputs_return_sentinels_without_unwinding() {
        // null world everywhere → sentinel, never a panic reaching the caller
        let null = std::ptr::null_mut::<PhysicsWorld>();
        unsafe {
            assert_eq!(alice_physics_world_step(null, 1.0 / 60.0), 0);
            assert_eq!(alice_physics_world_step_n(null, 1.0 / 60.0, 5), 0);
            assert_eq!(alice_physics_world_body_count(null), 0);
            assert_eq!(
                alice_physics_body_add_static(
                    null,
                    AliceVec3 {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0
                    }
                ),
                u32::MAX
            );
            alice_physics_world_set_gravity(null, 0.0, -1.0, 0.0);
            alice_physics_world_set_substeps(null, 3);
            alice_physics_world_destroy(null);
        }
        // zero substeps then step: the world must survive (dt / 0 → ZERO → no substep)
        let world = alice_physics_world_create();
        unsafe {
            alice_physics_world_set_substeps(world, 0);
            alice_physics_body_add_dynamic(
                world,
                AliceVec3 {
                    x: 0.0,
                    y: 5.0,
                    z: 0.0,
                },
                1.0,
            );
            assert_eq!(alice_physics_world_step(world, 1.0 / 60.0), 1);
            // NaN / infinite dt is clamped by Fix128::from_f64, not a panic
            assert_eq!(alice_physics_world_step(world, f64::NAN), 1);
            assert_eq!(alice_physics_world_step(world, f64::INFINITY), 1);
            // garbage state
            let junk = [0xFFu8; 13];
            assert_eq!(alice_physics_state_deserialize(world, junk.as_ptr(), 13), 0);
            alice_physics_world_destroy(world);
        }
    }

    // ---- collision radius, shapes, static colliders, joints ----------

    fn v(x: f64, y: f64, z: f64) -> AliceVec3 {
        AliceVec3 { x, y, z }
    }

    fn fx(x: f64) -> Fix128 {
        Fix128::from_f64(x)
    }

    fn vf(x: f64, y: f64, z: f64) -> Vec3Fix {
        Vec3Fix::new(fx(x), fx(y), fx(z))
    }

    /// Two bodies in a fresh world, and the same world built through the
    /// Rust API, for bit-for-bit comparison.
    unsafe fn twin_worlds() -> (*mut PhysicsWorld, PhysicsWorld) {
        let world = alice_physics_world_create();
        alice_physics_body_add_static(world, v(0.0, 0.0, 0.0));
        alice_physics_body_add_dynamic(world, v(2.0, 0.0, 0.0), 1.0);
        let mut rust = PhysicsWorld::new(SolverConfig::default());
        rust.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        rust.add_body(RigidBody::new_dynamic(vf(2.0, 0.0, 0.0), Fix128::ONE));
        (world, rust)
    }

    fn same_state(a: &PhysicsWorld, b: &PhysicsWorld) {
        assert_eq!(a.bodies.len(), b.bodies.len());
        for (x, y) in a.bodies.iter().zip(&b.bodies) {
            assert_eq!(
                (
                    x.position,
                    x.velocity,
                    x.rotation,
                    x.inv_mass,
                    x.inv_inertia
                ),
                (
                    y.position,
                    y.velocity,
                    y.rotation,
                    y.inv_mass,
                    y.inv_inertia
                )
            );
        }
        assert_eq!(a.joints, b.joints);
        assert_eq!(a.static_collider_count(), b.static_collider_count());
    }

    /// oracle: each binding call does what the Rust API call with the same
    /// values does — the worlds stay bit-identical through 30 steps.
    #[test]
    fn binding_calls_match_the_rust_api_bit_for_bit() {
        unsafe {
            let (world, mut rust) = twin_worlds();
            let w = &mut *world;

            assert_eq!(alice_physics_body_set_collision_radius(world, 1, 0.5), 1);
            rust.set_body_collision_radius(1, fx(0.5));

            let cube = AlicePhysicsShape {
                kind: 0,
                a: 0.5,
                b: 0.5,
                c: 0.5,
            };
            let id = alice_physics_body_add_shaped(world, cube, 2.0, v(5.0, 3.0, 0.0));
            assert_eq!(id, 2);
            let rust_cube = crate::shape::Shape::Box {
                half_extents: vf(0.5, 0.5, 0.5),
            };
            rust.add_shaped_body(&rust_cube, fx(2.0), vf(5.0, 3.0, 0.0))
                .unwrap();

            let cyl = AlicePhysicsShape {
                kind: 1,
                a: 0.25,
                b: 1.0,
                c: 0.0,
            };
            assert_eq!(alice_physics_body_set_shape(world, 1, cyl), 1);
            let rust_cyl = crate::shape::Shape::Cylinder {
                radius: fx(0.25),
                half_height: Fix128::ONE,
            };
            assert!(rust.set_body_shape(1, &rust_cyl));

            assert_eq!(
                alice_physics_static_add_plane(world, v(0.0, 2.0, 0.0), -1.0),
                0
            );
            rust.add_static_collider(crate::static_collider::StaticCollider::Plane(
                crate::plane_collider::PlaneCollider::new(Vec3Fix::UNIT_Y, -Fix128::ONE),
            ));

            assert_eq!(
                alice_physics_joint_add_ball(world, 0, 1, v(0.0, 0.0, 0.0), v(-2.0, 0.0, 0.0)),
                0
            );
            rust.add_joint(Joint::Ball(BallJoint::new(
                0,
                1,
                Vec3Fix::ZERO,
                vf(-2.0, 0.0, 0.0),
            )));
            assert_eq!(
                alice_physics_joint_add_spring(
                    world,
                    1,
                    2,
                    v(0.0, 0.0, 0.0),
                    v(0.0, 0.0, 0.0),
                    3.0,
                    50.0,
                    0.5
                ),
                1
            );
            rust.add_joint(Joint::Spring(SpringJoint::new(
                1,
                2,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                fx(3.0),
                fx(50.0),
                fx(0.5),
            )));
            assert_eq!(alice_physics_joint_count(world), 2);
            assert_eq!(alice_physics_static_count(world), 1);

            let dt = Fix128::from_ratio(1, 60);
            for _ in 0..30 {
                assert_eq!(alice_physics_world_step(world, 1.0 / 60.0), 1);
                rust.step(dt);
            }
            same_state(w, &rust);
            alice_physics_world_destroy(world);
        }
    }

    /// oracle: hinge / fixed / slider joints and the mesh colliders reach
    /// the world with the values given (axes normalised, rotation
    /// normalised), and removal shifts or swaps as documented.
    #[test]
    fn joint_and_static_constructors_store_what_was_passed() {
        unsafe {
            let (world, _) = twin_worlds();
            alice_physics_body_add_dynamic(world, v(0.0, 5.0, 0.0), 1.0);
            let h = alice_physics_joint_add_hinge(
                world,
                0,
                1,
                v(0.0, 0.0, 0.0),
                v(1.0, 0.0, 0.0),
                v(0.0, 0.0, 3.0),
                v(0.0, 0.0, 1.0),
            );
            let f = alice_physics_joint_add_fixed(
                world,
                0,
                2,
                v(0.0, 0.0, 0.0),
                v(0.0, 0.0, 0.0),
                AliceQuat {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 2.0,
                },
            );
            let s = alice_physics_joint_add_slider(
                world,
                1,
                2,
                v(2.0, 0.0, 0.0),
                v(0.0, 0.0, 0.0),
                v(0.0, 0.0, 0.0),
            );
            assert_eq!((h, f, s), (0, 1, 2));
            let w = &*world;
            match w.joints[0] {
                Joint::Hinge(j) => {
                    assert_eq!((j.body_a, j.body_b), (0, 1));
                    assert_eq!(j.local_axis_a, Vec3Fix::UNIT_Z);
                    assert_eq!(j.local_anchor_b, vf(1.0, 0.0, 0.0));
                }
                other => panic!("expected a hinge, got {other:?}"),
            }
            match w.joints[1] {
                Joint::Fixed(j) => assert_eq!(j.relative_rotation, QuatFix::IDENTITY),
                other => panic!("expected a fixed joint, got {other:?}"),
            }
            match w.joints[2] {
                Joint::Slider(j) => assert_eq!(j.local_axis, Vec3Fix::UNIT_X),
                other => panic!("expected a slider, got {other:?}"),
            }
            assert_eq!(alice_physics_joint_remove(world, 0), 1);
            assert_eq!(alice_physics_joint_count(world), 2);
            let w = &*world;
            assert!(
                matches!(w.joints[0], Joint::Slider(_)),
                "the last joint moves into the hole"
            );

            let heights = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
            assert_eq!(
                alice_physics_static_add_heightfield(
                    world,
                    heights.as_ptr(),
                    3,
                    2,
                    1.0,
                    v(0.0, 0.0, 0.0)
                ),
                0
            );
            let verts = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            let idx = [0u32, 1, 2];
            assert_eq!(
                alice_physics_static_add_trimesh(world, verts.as_ptr(), 3, idx.as_ptr(), 3),
                1
            );
            assert_eq!(alice_physics_static_count(world), 2);
            assert_eq!(alice_physics_static_remove(world, 0), 1);
            assert_eq!(alice_physics_static_count(world), 1);
            alice_physics_world_destroy(world);
        }
    }

    /// Every refused argument returns the sentinel and leaves the world as
    /// it was; nothing panics (a panic would have set the error slot).
    #[test]
    fn binding_calls_refuse_bad_arguments_without_touching_the_world() {
        unsafe {
            let (world, _) = twin_worlds();
            alice_physics_clear_last_error();
            let nan = f64::NAN;
            let null: *mut PhysicsWorld = std::ptr::null_mut();
            let unit = AlicePhysicsShape {
                kind: 0,
                a: 1.0,
                b: 1.0,
                c: 1.0,
            };
            let q1 = AliceQuat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            };
            let o = v(0.0, 0.0, 0.0);

            for r in [0.0, -1.0, nan, f64::INFINITY] {
                assert_eq!(
                    alice_physics_body_set_collision_radius(world, 1, r),
                    0,
                    "radius {r}"
                );
            }
            assert_eq!(alice_physics_body_set_collision_radius(world, 9, 1.0), 0);
            assert_eq!(alice_physics_body_set_collision_radius(null, 0, 1.0), 0);
            assert_eq!(alice_physics_body_clear_collision_radius(world, 9), 0);

            let bad_shapes = [
                AlicePhysicsShape { kind: 6, ..unit },
                AlicePhysicsShape { a: 0.0, ..unit },
                AlicePhysicsShape { c: nan, ..unit },
                AlicePhysicsShape {
                    kind: 5,
                    a: 1.0,
                    b: 1.0,
                    c: 0.0,
                },
            ];
            for s in bad_shapes {
                assert_eq!(alice_physics_body_add_shaped(world, s, 1.0, o), u32::MAX);
                assert_eq!(alice_physics_body_set_shape(world, 1, s), 0);
            }
            assert_eq!(alice_physics_body_add_shaped(world, unit, 0.0, o), u32::MAX);
            assert_eq!(
                alice_physics_body_add_shaped(world, unit, 1.0, v(nan, 0.0, 0.0)),
                u32::MAX
            );
            assert_eq!(alice_physics_body_set_shape(world, 9, unit), 0);

            assert_eq!(alice_physics_static_add_plane(world, o, 0.0), u32::MAX);
            assert_eq!(
                alice_physics_static_add_plane(world, v(0.0, 1.0, 0.0), nan),
                u32::MAX
            );
            let hs = [0.0; 4];
            assert_eq!(
                alice_physics_static_add_heightfield(world, hs.as_ptr(), 1, 4, 1.0, o),
                u32::MAX
            );
            assert_eq!(
                alice_physics_static_add_heightfield(world, hs.as_ptr(), 2, 2, 0.0, o),
                u32::MAX
            );
            assert_eq!(
                alice_physics_static_add_heightfield(world, std::ptr::null(), 2, 2, 1.0, o),
                u32::MAX
            );
            let vs = [0.0; 9];
            assert_eq!(
                alice_physics_static_add_trimesh(world, vs.as_ptr(), 3, [0u32, 1, 3].as_ptr(), 3),
                u32::MAX
            );
            assert_eq!(
                alice_physics_static_add_trimesh(world, vs.as_ptr(), 3, [0u32, 1].as_ptr(), 2),
                u32::MAX
            );
            assert_eq!(
                alice_physics_static_add_trimesh(world, vs.as_ptr(), 0, [0u32, 1, 2].as_ptr(), 3),
                u32::MAX
            );
            assert_eq!(alice_physics_static_remove(world, 0), 0);

            assert_eq!(
                alice_physics_joint_add_ball(world, 1, 1, o, o),
                u32::MAX,
                "self joint"
            );
            assert_eq!(
                alice_physics_joint_add_ball(world, 0, 9, o, o),
                u32::MAX,
                "unknown body"
            );
            assert_eq!(
                alice_physics_joint_add_ball(world, 0, 1, v(nan, 0.0, 0.0), o),
                u32::MAX
            );
            assert_eq!(
                alice_physics_joint_add_hinge(world, 0, 1, o, o, o, v(0.0, 0.0, 1.0)),
                u32::MAX,
                "zero axis"
            );
            assert_eq!(
                alice_physics_joint_add_fixed(world, 0, 1, o, o, AliceQuat { w: 0.0, ..q1 }),
                u32::MAX,
                "zero rotation"
            );
            assert_eq!(
                alice_physics_joint_add_slider(world, 0, 1, o, o, o),
                u32::MAX,
                "zero axis"
            );
            assert_eq!(
                alice_physics_joint_add_spring(world, 0, 1, o, o, 1.0, 0.0, 0.0),
                u32::MAX,
                "stiffness 0"
            );
            assert_eq!(
                alice_physics_joint_add_spring(world, 0, 1, o, o, -1.0, 1.0, 0.0),
                u32::MAX,
                "rest < 0"
            );
            assert_eq!(alice_physics_joint_add_ball(null, 0, 1, o, o), u32::MAX);
            assert_eq!(alice_physics_joint_remove(world, 0), 0);
            assert_eq!(alice_physics_joint_count(null), 0);
            assert_eq!(alice_physics_static_count(null), 0);

            let w = &*world;
            assert_eq!(w.bodies.len(), 2);
            assert!(w.joints.is_empty());
            assert_eq!(w.static_collider_count(), 0);
            assert!(
                alice_physics_last_error().is_null(),
                "a refusal must not come from a caught panic"
            );
            alice_physics_world_destroy(world);
        }
    }
}
