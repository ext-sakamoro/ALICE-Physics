//! Vehicle Physics
//!
//! Wheel + suspension + engine model for ground vehicles.
//!
//! The ground is the horizontal plane `y = VehicleConfig::ground_height`;
//! height fields, triangle meshes and SDF surfaces are not read. All wheel
// LIMITATION(COV-MBD-051): forces are summed and applied at the centre of mass as one impulse, so the wheels do not produce yaw or load transfer by themselves.
//! forces are summed and applied at the centre of mass as one impulse, so
//! the wheels do not produce yaw or load transfer by themselves. For
//! stopping distances, cornering, wheel spin, ABS, tyre models, road
//! surfaces or weather, use [`crate::vehicle_dynamics`].
//!
//! # Components
//!
//! - **Wheel**: vertical contact test against the ground plane; the spin
//!   follows the forward speed (`v / r`) and `lateral_slip` /
//!   `longitudinal_slip` are not computed (they stay 0)
//! - **Suspension**: Spring-damper for each wheel
//! - **Engine**: constant torque `max_torque * throttle` times the selected
//!   gear ratio, split over the driven wheels (no rpm-dependent torque curve;
//!   `EngineConfig::max_rpm`, `engine_brake` and `num_gears` are not read by
//!   [`Vehicle::update`])
//! - **Steering**: a lateral force per steerable grounded wheel,
//!   `|v| * 1000 * (right sin d + forward (cos d - 1))` with `d = steering *
//!   max_steer_angle`; it is not Ackermann geometry (no inner / outer wheel
//!   angle difference)
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, Port, PortId, StateError,
    SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Vehicle Configuration
// ============================================================================

/// Wheel configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelConfig {
    /// Suspension attachment point (local to chassis)
    pub local_position: Vec3Fix,
    /// Suspension rest length
    pub suspension_rest: Fix128,
    /// Suspension spring stiffness
    pub spring_stiffness: Fix128,
    /// Suspension damping coefficient
    pub damping: Fix128,
    /// Wheel radius
    pub radius: Fix128,
    /// Maximum steering angle (radians, 0 for rear wheels)
    pub max_steer_angle: Fix128,
    /// Whether this wheel is driven (receives engine torque)
    pub driven: bool,
    /// Whether this wheel has brakes
    pub has_brake: bool,
    /// Progressive spring rate coefficient (0 = linear, >0 = stiffness increases with compression)
    pub progressive_rate: Fix128,
    /// Bump damping coefficient (compression direction).
    /// 0 = use `damping` for both directions (symmetric).
    pub bump_damping: Fix128,
    /// Rebound damping coefficient (extension direction).
    /// 0 = use `damping` for both directions (symmetric).
    pub rebound_damping: Fix128,
    /// Bump stop stiffness (N/m). Very stiff spring activated when compression exceeds `bump_stop_threshold`.
    /// 0 = no bump stop.
    pub bump_stop_stiffness: Fix128,
    /// Bump stop activation threshold (0..1, fraction of full compression).
    /// Default 0.9 = bump stop activates at 90% compression.
    pub bump_stop_threshold: Fix128,
}

impl Default for WheelConfig {
    fn default() -> Self {
        Self {
            local_position: Vec3Fix::ZERO,
            suspension_rest: Fix128::from_ratio(3, 10),
            spring_stiffness: Fix128::from_int(50000),
            damping: Fix128::from_int(4500),
            radius: Fix128::from_ratio(3, 10),
            max_steer_angle: Fix128::from_ratio(5, 10), // ~28.6 degrees
            driven: false,
            has_brake: true,
            progressive_rate: Fix128::ZERO,
            bump_damping: Fix128::ZERO,
            rebound_damping: Fix128::ZERO,
            bump_stop_stiffness: Fix128::ZERO,
            bump_stop_threshold: Fix128::from_ratio(9, 10), // 0.9
        }
    }
}

/// Engine configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngineConfig {
    /// Maximum engine torque (Nm)
    pub max_torque: Fix128,
    /// Maximum engine RPM (informational: not read by `Vehicle::update`)
    pub max_rpm: Fix128,
    /// Idle RPM
    pub idle_rpm: Fix128,
    /// Engine braking coefficient (informational: not read by `Vehicle::update`)
    pub engine_brake: Fix128,
    /// Number of gears (informational: the gear table is `VehicleConfig::gear_ratios`)
    pub num_gears: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_torque: Fix128::from_int(300),
            max_rpm: Fix128::from_int(7000),
            idle_rpm: Fix128::from_int(800),
            engine_brake: Fix128::from_ratio(5, 10),
            num_gears: 5,
        }
    }
}

/// Vehicle configuration
#[derive(Clone, Debug)]
pub struct VehicleConfig {
    /// Wheel configurations
    pub wheels: Vec<WheelConfig>,
    /// Engine configuration
    pub engine: EngineConfig,
    /// Gear ratios (including final drive)
    pub gear_ratios: Vec<Fix128>,
    /// Brake force (N)
    pub brake_force: Fix128,
    /// Aerodynamic drag coefficient
    pub aero_drag: Fix128,
    /// Downforce coefficient
    pub downforce: Fix128,
    /// Ground plane height (Y coordinate)
    pub ground_height: Fix128,
    /// Anti-roll bar stiffness (N/m)。左右ホイール圧縮差に比例するスタビライザー力。
    /// 0 = 無効。
    pub anti_roll_stiffness: Fix128,
}

impl Default for VehicleConfig {
    fn default() -> Self {
        Self {
            wheels: vec![
                WheelConfig {
                    local_position: Vec3Fix::from_f32(-0.8, -0.2, 1.2),
                    driven: false,
                    max_steer_angle: Fix128::from_ratio(5, 10),
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(0.8, -0.2, 1.2),
                    driven: false,
                    max_steer_angle: Fix128::from_ratio(5, 10),
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(-0.8, -0.2, -1.2),
                    driven: true,
                    ..Default::default()
                },
                WheelConfig {
                    local_position: Vec3Fix::from_f32(0.8, -0.2, -1.2),
                    driven: true,
                    ..Default::default()
                },
            ],
            engine: EngineConfig::default(),
            gear_ratios: vec![
                Fix128::from_ratio(35, 10), // 1st
                Fix128::from_ratio(22, 10), // 2nd
                Fix128::from_ratio(15, 10), // 3rd
                Fix128::from_ratio(11, 10), // 4th
                Fix128::from_ratio(8, 10),  // 5th
            ],
            brake_force: Fix128::from_int(10000),
            aero_drag: Fix128::from_ratio(4, 10),
            downforce: Fix128::from_ratio(1, 10),
            ground_height: Fix128::ZERO,
            anti_roll_stiffness: Fix128::from_int(5000),
        }
    }
}

// ============================================================================
// Wheel State
// ============================================================================

/// Runtime wheel state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelState {
    /// Current suspension compression (0 = fully extended, 1 = bottomed out)
    pub compression: Fix128,
    /// Suspension force applied this frame
    pub suspension_force: Fix128,
    /// Whether wheel is touching ground
    pub grounded: bool,
    /// Ground contact point (world space)
    pub contact_point: Vec3Fix,
    /// Ground normal at contact
    pub ground_normal: Vec3Fix,
    /// Current wheel rotation angle (radians)
    pub spin_angle: Fix128,
    /// Wheel angular velocity (rad/s)
    pub spin_speed: Fix128,
    /// Lateral slip ratio
    pub lateral_slip: Fix128,
    /// Longitudinal slip ratio
    pub longitudinal_slip: Fix128,
}

impl Default for WheelState {
    fn default() -> Self {
        Self {
            compression: Fix128::ZERO,
            suspension_force: Fix128::ZERO,
            grounded: false,
            contact_point: Vec3Fix::ZERO,
            ground_normal: Vec3Fix::UNIT_Y,
            spin_angle: Fix128::ZERO,
            spin_speed: Fix128::ZERO,
            lateral_slip: Fix128::ZERO,
            longitudinal_slip: Fix128::ZERO,
        }
    }
}

// ============================================================================
// Vehicle
// ============================================================================

/// Vehicle physics simulation
pub struct Vehicle {
    /// Vehicle configuration
    pub config: VehicleConfig,
    /// Per-wheel runtime state
    pub wheel_states: Vec<WheelState>,
    /// Current gear (0 = first gear)
    pub current_gear: usize,
    /// Engine speed indicator: `|km/h| * 100/36 * gear ratio`, floored at
    /// `idle_rpm`. Proportional to ground speed, not a physical rpm (no wheel
    /// radius or final drive), and it does not feed back into the torque.
    pub engine_rpm: Fix128,
    /// Throttle input (0..1)
    pub throttle: Fix128,
    /// Brake input (0..1)
    pub brake: Fix128,
    /// Steering input (-1..1)
    pub steering: Fix128,
    /// Forward speed (km/h)
    pub speed_kmh: Fix128,
}

impl Vehicle {
    /// Create a new vehicle
    #[must_use]
    pub fn new(config: VehicleConfig) -> Self {
        let n = config.wheels.len();
        Self {
            wheel_states: vec![WheelState::default(); n],
            current_gear: 0,
            engine_rpm: config.engine.idle_rpm,
            throttle: Fix128::ZERO,
            brake: Fix128::ZERO,
            steering: Fix128::ZERO,
            speed_kmh: Fix128::ZERO,
            config,
        }
    }

    /// Create default 4-wheel vehicle
    #[must_use]
    pub fn new_default() -> Self {
        Self::new(VehicleConfig::default())
    }

    /// Update vehicle physics: raycast the wheels, evaluate suspension /
    /// drive / brake / steering / aero forces and apply them to `chassis` as
    /// one impulse `F·dt`. Call once per frame **before** `PhysicsWorld::step`.
    ///
    /// # Frame-level coupling
    ///
    /// The impulse lands at the start of the frame while the world integrates
    /// gravity per substep, so at rest the chassis' *position* is stationary
    /// (ride height `r + rest·(1 − mg/k)`, `tests/default_configs.rs`) but its
    /// frame-end vertical velocity samples the saw-tooth at ≈ `−g·dt/2`
    /// (−0.08 m/s at 60 Hz) rather than 0. Read positions, not velocities, for
    /// "is the car settled" logic. Substep-level coupling needs a force
    /// accumulator on `RigidBody`, which is a layout change (`#[repr(C)]`,
    /// FFI header) and therefore a 2.0 item.
    pub fn update(&mut self, chassis: &mut RigidBody, dt: Fix128) {
        if chassis.is_static() {
            return;
        }
        let force = self.chassis_force(chassis, dt);
        chassis.apply_impulse(force * dt);
    }

    /// The force the wheels, the engine, the brakes and the air put on
    /// `chassis` over a step of `dt`, with the vehicle state advanced as
    /// [`Self::update`] advances it (wheel contact, suspension, spin, speed,
    /// engine rpm). The chassis is only read. [`Self::update`] is this force
    /// applied as one impulse `F·dt`; a world participant can stage it in
    /// every substep instead.
    ///
    /// A static chassis gets no force and the state is left unchanged.
    #[must_use]
    pub fn chassis_force(&mut self, chassis: &RigidBody, dt: Fix128) -> Vec3Fix {
        if chassis.is_static() {
            return Vec3Fix::ZERO;
        }

        let forward = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Z);
        let right = chassis.rotation.rotate_vec(Vec3Fix::UNIT_X);
        let up = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Y);

        // Compute forward speed
        self.speed_kmh = chassis.velocity.dot(forward) * Fix128::from_ratio(36, 10);

        // Engine torque
        let gear_ratio = if self.current_gear < self.config.gear_ratios.len() {
            self.config.gear_ratios[self.current_gear]
        } else {
            Fix128::ONE
        };
        let engine_torque = self.config.engine.max_torque * self.throttle;
        let wheel_torque = engine_torque * gear_ratio;

        // Process each wheel
        let num_driven = self.config.wheels.iter().filter(|w| w.driven).count() as i64;
        let torque_per_wheel = if num_driven > 0 {
            wheel_torque / Fix128::from_int(num_driven)
        } else {
            Fix128::ZERO
        };

        let mut total_force = Vec3Fix::ZERO;

        for i in 0..self.config.wheels.len() {
            let wheel_cfg = self.config.wheels[i];

            // World-space wheel position
            let wheel_world =
                chassis.position + chassis.rotation.rotate_vec(wheel_cfg.local_position);

            // Raycast down for ground contact
            let ray_start = wheel_world;
            let max_dist = wheel_cfg.suspension_rest + wheel_cfg.radius;

            // Ground plane check (configurable height)
            let ground_y = self.config.ground_height;
            let dist_to_ground = ray_start.y - ground_y;

            // A wheel centre at or below the ground (`dist_to_ground <= 0`) is the
            // deepest compression, not the absence of contact: the suspension
            // keeps its full force (compression clamps at 1) instead of letting
            // a bottomed-out chassis fall through.
            if dist_to_ground < max_dist {
                self.wheel_states[i].grounded = true;
                self.wheel_states[i].ground_normal = Vec3Fix::UNIT_Y;
                self.wheel_states[i].contact_point =
                    Vec3Fix::new(wheel_world.x, ground_y, wheel_world.z);

                // Suspension
                let compression =
                    Fix128::ONE - (dist_to_ground - wheel_cfg.radius) / wheel_cfg.suspension_rest;
                let compression = if compression < Fix128::ZERO {
                    Fix128::ZERO
                } else if compression > Fix128::ONE {
                    Fix128::ONE
                } else {
                    compression
                };

                // Progressive spring: stiffness increases with compression
                let effective_stiffness = wheel_cfg.spring_stiffness
                    * (Fix128::ONE + wheel_cfg.progressive_rate * compression);
                let mut spring_force = effective_stiffness * compression;

                // Bump stop: very stiff spring at high compression
                if !wheel_cfg.bump_stop_stiffness.is_zero()
                    && compression > wheel_cfg.bump_stop_threshold
                {
                    let overshoot = compression - wheel_cfg.bump_stop_threshold;
                    spring_force = spring_force + wheel_cfg.bump_stop_stiffness * overshoot;
                }

                // Asymmetric damping: bump (compression) vs rebound (extension)
                let vel_y = chassis.velocity.dot(up);
                let damp_coeff = if vel_y < Fix128::ZERO {
                    // Moving downward = compressing suspension = bump
                    if wheel_cfg.bump_damping.is_zero() {
                        wheel_cfg.damping
                    } else {
                        wheel_cfg.bump_damping
                    }
                } else {
                    // Moving upward = extending suspension = rebound
                    if wheel_cfg.rebound_damping.is_zero() {
                        wheel_cfg.damping
                    } else {
                        wheel_cfg.rebound_damping
                    }
                };
                let damp_force = damp_coeff * vel_y;
                let susp_force = spring_force - damp_force;

                self.wheel_states[i].compression = compression;
                self.wheel_states[i].suspension_force = susp_force;

                total_force = total_force + up * susp_force;

                // Driving force
                if wheel_cfg.driven {
                    let drive_force = forward * (torque_per_wheel / wheel_cfg.radius);
                    total_force = total_force + drive_force;
                }

                // Braking
                if wheel_cfg.has_brake && self.brake > Fix128::ZERO {
                    let speed = chassis.velocity.dot(forward);
                    if speed.abs() > Fix128::from_ratio(1, 10) {
                        let brake_dir = if speed > Fix128::ZERO {
                            -forward
                        } else {
                            forward
                        };
                        let brake_mag = self.config.brake_force * self.brake;
                        total_force = total_force + brake_dir * brake_mag;
                    }
                }

                // Lateral friction (resist sliding)
                let lateral_vel = chassis.velocity.dot(right);
                let lateral_friction = -right * (lateral_vel * Fix128::from_int(5000));
                total_force = total_force + lateral_friction;

                // Steering
                if wheel_cfg.max_steer_angle > Fix128::ZERO {
                    let steer_angle = self.steering * wheel_cfg.max_steer_angle;
                    let (sin_a, cos_a) = steer_angle.sin_cos();
                    let steer_force = (right * sin_a + forward * cos_a) - forward;
                    let speed_factor = chassis.velocity.dot(forward).abs();
                    total_force =
                        total_force + steer_force * (speed_factor * Fix128::from_int(1000));
                }
            } else {
                self.wheel_states[i].grounded = false;
                self.wheel_states[i].compression = Fix128::ZERO;
            }

            // Update wheel spin
            if self.wheel_states[i].grounded {
                let ground_speed = chassis.velocity.dot(forward);
                self.wheel_states[i].spin_speed = ground_speed / wheel_cfg.radius;
            }
            self.wheel_states[i].spin_angle =
                self.wheel_states[i].spin_angle + self.wheel_states[i].spin_speed * dt;
        }

        // Anti-roll bar: 左右ホイール圧縮差に比例する復元力。
        // ホイールをペアで処理（0-1がフロント、2-3がリア等）。
        if !self.config.anti_roll_stiffness.is_zero() && self.config.wheels.len() >= 2 {
            let num_pairs = self.config.wheels.len() / 2;
            for pair in 0..num_pairs {
                let left = pair * 2;
                let right = pair * 2 + 1;
                let diff =
                    self.wheel_states[left].compression - self.wheel_states[right].compression;
                let anti_roll_force = self.config.anti_roll_stiffness * diff;
                // 左を押し下げ、右を押し上げ（差が正→左が沈んでいる）
                if self.wheel_states[left].grounded {
                    total_force = total_force - up * anti_roll_force;
                }
                if self.wheel_states[right].grounded {
                    total_force = total_force + up * anti_roll_force;
                }
            }
        }

        // Aerodynamic drag
        let speed_sq = chassis.velocity.length_squared();
        if !speed_sq.is_zero() {
            let speed = speed_sq.sqrt();
            let drag_dir = chassis.velocity / speed;
            total_force = total_force - drag_dir * (self.config.aero_drag * speed_sq);
            // Downforce
            total_force = total_force - up * (self.config.downforce * speed_sq);
        }

        // Update engine RPM
        let wheel_rpm = self.speed_kmh.abs() * Fix128::from_ratio(100, 36) * gear_ratio;
        self.engine_rpm = wheel_rpm.abs();
        if self.engine_rpm < self.config.engine.idle_rpm {
            self.engine_rpm = self.config.engine.idle_rpm;
        }
        total_force
    }

    /// Shift gear up
    pub fn shift_up(&mut self) {
        if self.current_gear + 1 < self.config.gear_ratios.len() {
            self.current_gear += 1;
        }
    }

    /// Shift gear down
    pub fn shift_down(&mut self) {
        if self.current_gear > 0 {
            self.current_gear -= 1;
        }
    }

    /// Number of grounded wheels
    #[must_use]
    pub fn grounded_wheels(&self) -> usize {
        self.wheel_states.iter().filter(|w| w.grounded).count()
    }
}

impl core::fmt::Debug for Vehicle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vehicle")
            .field("config", &self.config)
            .field(
                "wheel_states",
                &format_args!("[{} items]", self.wheel_states.len()),
            )
            .field("current_gear", &self.current_gear)
            .field("engine_rpm", &self.engine_rpm)
            .field("throttle", &self.throttle)
            .field("brake", &self.brake)
            .field("steering", &self.steering)
            .field("speed_kmh", &self.speed_kmh)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// World participant
// ============================================================================

/// Snapshot tag of [`VehicleParticipant`]: the ASCII code `VHCL`, big endian.
pub const VEHICLE_PARTICIPANT_KIND: ParticipantKind =
    ParticipantKind::new(u32::from_be_bytes(*b"VHCL"));

/// Number of values in the action field a [`VehicleParticipant`] reads
/// ([`crate::world_participant::FieldLayout::Channels`] with this count).
pub const VEHICLE_INPUTS: usize = 3;
/// Index of the throttle (`0..=1`) in the action field.
pub const VEHICLE_INPUT_THROTTLE: usize = 0;
/// Index of the brake (`0..=1`) in the action field.
pub const VEHICLE_INPUT_BRAKE: usize = 1;
/// Index of the steering (`−1..=1`) in the action field.
pub const VEHICLE_INPUT_STEERING: usize = 2;

/// Observation channel: forward speed in km/h.
pub const VEHICLE_OBS_SPEED_KMH: u32 = 0;
/// Observation channel: the engine speed indicator ([`Vehicle::engine_rpm`]).
pub const VEHICLE_OBS_ENGINE_RPM: u32 = 1;
/// Observation channel: the current gear (0 = first).
pub const VEHICLE_OBS_GEAR: u32 = 2;
/// Observation channel: the throttle in force.
pub const VEHICLE_OBS_THROTTLE: u32 = 3;
/// Observation channel: the brake in force.
pub const VEHICLE_OBS_BRAKE: u32 = 4;
/// Observation channel: the steering in force.
pub const VEHICLE_OBS_STEERING: u32 = 5;
/// Observation channel: how many wheels touch the ground.
pub const VEHICLE_OBS_GROUNDED: u32 = 6;

/// Payload layout version written by [`VehicleParticipant`].
const VEHICLE_STATE_VERSION: u32 = 1;

/// Bytes of the payload before the wheels: `version: u32`, `digest: u64`,
/// `gear: u64`, then rpm, throttle, brake, steering, speed (`Fix128`).
const VEHICLE_STATE_HEAD: usize = 4 + 8 + 8 + 5 * 16;

/// Bytes per wheel: compression, suspension force, spin angle, spin speed,
/// lateral and longitudinal slip, contact point and ground normal (12
/// `Fix128`), then `grounded: u8`.
const VEHICLE_STATE_WHEEL: usize = 12 * 16 + 1;

/// A [`Vehicle`] as a participant of a [`crate::PhysicsWorld`]: it drives
/// the body `chassis` with the force of [`Vehicle::chassis_force`] in every
/// substep, so the suspension, drive, brakes and drag are integrated with
/// the substep width like gravity (the frame-head impulse of
/// [`Vehicle::update`] leaves the chassis velocity sampled near
/// `−g·dt/2` at rest; staged per substep it rests).
///
/// # Inputs as world state
///
/// With [`Self::with_inputs`] the throttle, brake and steering are read in
/// every substep from a world field declared
/// [`crate::world_participant::FieldLayout::Channels`] `{ count: VEHICLE_INPUTS }` (indices
/// [`VEHICLE_INPUT_THROTTLE`], [`VEHICLE_INPUT_BRAKE`],
/// [`VEHICLE_INPUT_STEERING`]), read committed. The caller writes the field
/// between steps with [`crate::PhysicsWorld::set_field`]: one write holds for
/// every substep of the next frame (and the frames after, until the next
/// write). Because the inputs are world state, a snapshot carries them, a
/// fork continues with the inputs in force, and a drive is replayed bit for
/// bit from the initial snapshot and the list of `(frame, inputs)` writes.
/// Without inputs the vehicle keeps the throttle, brake and steering it was
/// built with.
///
/// # Faults
///
/// A throttle or brake outside `0..=1` or a steering outside `−1..=1` is
/// [`ParticipantFault::OutOfRange`]; an input field of another length or a
/// chassis index past the world's bodies is
/// [`ParticipantFault::InvalidState`]. In both cases nothing is staged and
/// the participant is unchanged. The vehicle's own arithmetic is not
/// range-checked; the force reaching the body is
/// (`WorldFault::ForceOutOfRange`). A static chassis gets no force and the
/// vehicle state is left unchanged.
///
/// # Snapshot payload
///
/// Little endian: `version: u32` (1), `digest: u64`, the gear (`u64`), the
/// engine rpm, throttle, brake, steering and speed (`Fix128` each, `hi: i64`,
/// `lo: u64`), then per wheel the compression, suspension force, spin angle,
/// spin speed, lateral slip, longitudinal slip, contact point (3) and
/// ground normal (3) as `Fix128` and `grounded` as one byte. The digest is
/// FNV-1a 64 over the whole configuration, the chassis index and the input
/// field; [`Participant::check_state`] refuses a payload of another length
/// ([`StateError::Length`]), version or digest ([`StateError::InvalidValue`]).
pub struct VehicleParticipant {
    vehicle: Vehicle,
    chassis: usize,
    inputs: Option<PortId>,
    ports: Vec<Port>,
    digest: u64,
}

impl VehicleParticipant {
    /// The participant driving body `chassis` with `vehicle`, its inputs
    /// those the vehicle holds.
    #[must_use]
    pub fn new(vehicle: Vehicle, chassis: usize) -> Self {
        let digest = vehicle_digest(&vehicle.config, chassis, None);
        Self {
            vehicle,
            chassis,
            inputs: None,
            ports: Vec::new(),
            digest,
        }
    }

    /// Read the throttle, brake and steering from the world field `field`
    /// (declared [`crate::world_participant::FieldLayout::Channels`] `{ count: VEHICLE_INPUTS }`
    /// before this participant is registered).
    #[must_use]
    pub fn with_inputs(mut self, field: PortId) -> Self {
        self.inputs = Some(field);
        self.ports = vec![Port::reads_committed(field)];
        self.digest = vehicle_digest(&self.vehicle.config, self.chassis, Some(field));
        self
    }

    /// The vehicle and its state.
    #[must_use]
    pub fn vehicle(&self) -> &Vehicle {
        &self.vehicle
    }

    /// The throttle, brake and steering for this substep (checked), or the
    /// fault.
    fn inputs(&self, ctx: &SubstepCtx<'_>) -> Result<Option<[Fix128; 3]>, ParticipantFault> {
        let Some(id) = self.inputs else {
            return Ok(None);
        };
        let values = ctx.field(id).map_err(|_| ParticipantFault::InvalidState)?;
        if values.len() != VEHICLE_INPUTS {
            return Err(ParticipantFault::InvalidState);
        }
        let (zero, one) = (Fix128::ZERO, Fix128::ONE);
        let throttle = values[VEHICLE_INPUT_THROTTLE];
        let brake = values[VEHICLE_INPUT_BRAKE];
        let steering = values[VEHICLE_INPUT_STEERING];
        if !(zero..=one).contains(&throttle)
            || !(zero..=one).contains(&brake)
            || !(-one..=one).contains(&steering)
        {
            return Err(ParticipantFault::OutOfRange);
        }
        Ok(Some([throttle, brake, steering]))
    }
}

fn vehicle_digest(config: &VehicleConfig, chassis: usize, inputs: Option<PortId>) -> u64 {
    let mut h = Fnv::new();
    h.word(config.wheels.len() as u64);
    for w in &config.wheels {
        h.vec(w.local_position);
        for f in [
            w.suspension_rest,
            w.spring_stiffness,
            w.damping,
            w.radius,
            w.max_steer_angle,
            w.progressive_rate,
            w.bump_damping,
            w.rebound_damping,
            w.bump_stop_stiffness,
            w.bump_stop_threshold,
        ] {
            h.fix(f);
        }
        h.word(u64::from(w.driven) | (u64::from(w.has_brake) << 1));
    }
    let e = &config.engine;
    for f in [e.max_torque, e.max_rpm, e.idle_rpm, e.engine_brake] {
        h.fix(f);
    }
    h.word(e.num_gears as u64);
    h.word(config.gear_ratios.len() as u64);
    for &g in &config.gear_ratios {
        h.fix(g);
    }
    for f in [
        config.brake_force,
        config.aero_drag,
        config.downforce,
        config.ground_height,
        config.anti_roll_stiffness,
    ] {
        h.fix(f);
    }
    h.word(chassis as u64);
    h.word(inputs.map_or(u64::MAX, |p| u64::from(p.get())));
    h.0
}

/// FNV-1a 64.
struct Fnv(u64);

impl Fnv {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn word(&mut self, w: u64) {
        self.bytes(&w.to_le_bytes());
    }
    fn fix(&mut self, f: Fix128) {
        self.bytes(&f.hi.to_le_bytes());
        self.bytes(&f.lo.to_le_bytes());
    }
    fn vec(&mut self, v: Vec3Fix) {
        self.fix(v.x);
        self.fix(v.y);
        self.fix(v.z);
    }
}

fn put_fix(out: &mut Vec<u8>, f: Fix128) {
    out.extend_from_slice(&f.hi.to_le_bytes());
    out.extend_from_slice(&f.lo.to_le_bytes());
}

fn get_fix(bytes: &[u8], at: &mut usize) -> Fix128 {
    let mut hi = [0u8; 8];
    let mut lo = [0u8; 8];
    hi.copy_from_slice(&bytes[*at..*at + 8]);
    lo.copy_from_slice(&bytes[*at + 8..*at + 16]);
    *at += 16;
    Fix128 {
        hi: i64::from_le_bytes(hi),
        lo: u64::from_le_bytes(lo),
    }
}

fn get_u64(bytes: &[u8], at: &mut usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[*at..*at + 8]);
    *at += 8;
    u64::from_le_bytes(b)
}

impl Participant for VehicleParticipant {
    fn kind(&self) -> ParticipantKind {
        VEHICLE_PARTICIPANT_KIND
    }

    fn ports(&self) -> &[Port] {
        &self.ports
    }

    fn substep(&mut self, ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        let inputs = self.inputs(ctx)?;
        let force = {
            let Some(body) = ctx.bodies().get(self.chassis) else {
                return Err(ParticipantFault::InvalidState);
            };
            if body.is_static() {
                return Ok(());
            }
            if let Some([throttle, brake, steering]) = inputs {
                self.vehicle.throttle = throttle;
                self.vehicle.brake = brake;
                self.vehicle.steering = steering;
            }
            self.vehicle.chassis_force(body, h)
        };
        ctx.add_force(self.chassis, force)
            .map_err(|_| ParticipantFault::InvalidState)
    }

    fn observe(&self, out: &mut ObservationSink) {
        let v = &self.vehicle;
        let grounded = v.wheel_states.iter().filter(|w| w.grounded).count();
        out.push(VEHICLE_OBS_SPEED_KMH, v.speed_kmh);
        out.push(VEHICLE_OBS_ENGINE_RPM, v.engine_rpm);
        out.push(VEHICLE_OBS_GEAR, Fix128::from_int(v.current_gear as i64));
        out.push(VEHICLE_OBS_THROTTLE, v.throttle);
        out.push(VEHICLE_OBS_BRAKE, v.brake);
        out.push(VEHICLE_OBS_STEERING, v.steering);
        out.push(VEHICLE_OBS_GROUNDED, Fix128::from_int(grounded as i64));
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let v = &self.vehicle;
        out.extend_from_slice(&VEHICLE_STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&self.digest.to_le_bytes());
        out.extend_from_slice(&(v.current_gear as u64).to_le_bytes());
        for f in [v.engine_rpm, v.throttle, v.brake, v.steering, v.speed_kmh] {
            put_fix(out, f);
        }
        for w in &v.wheel_states {
            for f in [
                w.compression,
                w.suspension_force,
                w.spin_angle,
                w.spin_speed,
                w.lateral_slip,
                w.longitudinal_slip,
                w.contact_point.x,
                w.contact_point.y,
                w.contact_point.z,
                w.ground_normal.x,
                w.ground_normal.y,
                w.ground_normal.z,
            ] {
                put_fix(out, f);
            }
            out.extend_from_slice(&[u8::from(w.grounded)]);
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let expected = VEHICLE_STATE_HEAD + VEHICLE_STATE_WHEEL * self.vehicle.wheel_states.len();
        if bytes.len() != expected {
            return Err(StateError::Length {
                expected,
                found: bytes.len(),
            });
        }
        let mut version = [0u8; 4];
        version.copy_from_slice(&bytes[0..4]);
        let mut at = 4;
        let digest = get_u64(bytes, &mut at);
        let gear = get_u64(bytes, &mut at);
        if u32::from_le_bytes(version) != VEHICLE_STATE_VERSION
            || digest != self.digest
            || usize::try_from(gear).is_err()
        {
            return Err(StateError::InvalidValue);
        }
        let mut at = 4 + 8 + 8 + 5 * 16;
        for _ in 0..self.vehicle.wheel_states.len() {
            at += 12 * 16;
            if bytes[at] > 1 {
                return Err(StateError::InvalidValue);
            }
            at += 1;
        }
        Ok(())
    }

    fn read_state(&mut self, bytes: &[u8]) {
        let v = &mut self.vehicle;
        let mut at = 4 + 8;
        v.current_gear = usize::try_from(get_u64(bytes, &mut at)).unwrap_or_default();
        v.engine_rpm = get_fix(bytes, &mut at);
        v.throttle = get_fix(bytes, &mut at);
        v.brake = get_fix(bytes, &mut at);
        v.steering = get_fix(bytes, &mut at);
        v.speed_kmh = get_fix(bytes, &mut at);
        for w in &mut v.wheel_states {
            w.compression = get_fix(bytes, &mut at);
            w.suspension_force = get_fix(bytes, &mut at);
            w.spin_angle = get_fix(bytes, &mut at);
            w.spin_speed = get_fix(bytes, &mut at);
            w.lateral_slip = get_fix(bytes, &mut at);
            w.longitudinal_slip = get_fix(bytes, &mut at);
            let cx = get_fix(bytes, &mut at);
            let cy = get_fix(bytes, &mut at);
            let cz = get_fix(bytes, &mut at);
            w.contact_point = Vec3Fix::new(cx, cy, cz);
            let nx = get_fix(bytes, &mut at);
            let ny = get_fix(bytes, &mut at);
            let nz = get_fix(bytes, &mut at);
            w.ground_normal = Vec3Fix::new(nx, ny, nz);
            w.grounded = bytes[at] == 1;
            at += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vehicle_creation() {
        let vehicle = Vehicle::new_default();
        assert_eq!(vehicle.config.wheels.len(), 4);
        assert_eq!(vehicle.wheel_states.len(), 4);
    }

    #[test]
    fn test_vehicle_suspension() {
        let mut vehicle = Vehicle::new_default();
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // Some wheels should be grounded
        assert!(vehicle.grounded_wheels() > 0, "Wheels should touch ground");
    }

    #[test]
    fn test_vehicle_acceleration() {
        let mut vehicle = Vehicle::new_default();
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        vehicle.throttle = Fix128::ONE; // Full throttle

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            vehicle.update(&mut chassis, dt);
        }

        // Vehicle should have gained forward velocity
        let speed = vehicle.speed_kmh.to_f32();
        assert!(
            speed.abs() > 0.01,
            "Vehicle should accelerate, speed={speed}"
        );
    }

    #[test]
    fn test_gear_shifting() {
        let mut vehicle = Vehicle::new_default();
        assert_eq!(vehicle.current_gear, 0);

        vehicle.shift_up();
        assert_eq!(vehicle.current_gear, 1);

        vehicle.shift_up();
        vehicle.shift_up();
        vehicle.shift_up();
        vehicle.shift_up(); // Should cap at max gear
        assert_eq!(vehicle.current_gear, 4);

        vehicle.shift_down();
        assert_eq!(vehicle.current_gear, 3);
    }

    #[test]
    fn test_progressive_spring() {
        // progressive_rate > 0 → 圧縮が深いほどばねが硬くなる
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.progressive_rate = Fix128::ONE; // stiffness doubles at full compression
        }
        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // 圧縮が0以上のホイールがあれば、プログレッシブで suspension_force が増加しているはず
        let grounded: Vec<_> = vehicle.wheel_states.iter().filter(|w| w.grounded).collect();
        assert!(!grounded.is_empty(), "Should have grounded wheels");
        for ws in &grounded {
            assert!(
                ws.suspension_force > Fix128::ZERO,
                "Suspension force should be positive"
            );
        }
    }

    #[test]
    fn test_anti_roll_bar() {
        let config = VehicleConfig::default();
        assert!(
            !config.anti_roll_stiffness.is_zero(),
            "Default anti-roll stiffness should be non-zero"
        );

        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle.update(&mut chassis, dt);

        // 左右圧縮差が小さければ anti-roll bar は微小な影響のみ
        // 基本的なシミュレーションが動作していることを確認
        assert!(vehicle.grounded_wheels() > 0);
    }

    #[test]
    fn test_asymmetric_damping_defaults() {
        let w = WheelConfig::default();
        // デフォルトではバンプ/リバウンド個別設定はゼロ（対称ダンピング使用）
        assert!(w.bump_damping.is_zero());
        assert!(w.rebound_damping.is_zero());
    }

    #[test]
    fn test_asymmetric_damping_simulation() {
        // バンプ減衰 > リバウンド減衰 の Rally 向け設定
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_damping = Fix128::from_int(6000); // 強いバンプ減衰
            w.rebound_damping = Fix128::from_int(3000); // 弱いリバウンド減衰
        }
        let mut vehicle = Vehicle::new(config);
        let mut chassis = RigidBody::new(Vec3Fix::from_f32(0.0, 0.5, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..30 {
            vehicle.update(&mut chassis, dt);
        }

        assert!(vehicle.grounded_wheels() > 0, "Should have grounded wheels");
        // サスペンション力が正（支持している）ことを確認
        for ws in &vehicle.wheel_states {
            if ws.grounded {
                assert!(
                    ws.suspension_force > Fix128::ZERO,
                    "Suspension force should be positive with asymmetric damping"
                );
            }
        }
    }

    #[test]
    fn test_bump_stop_defaults() {
        let w = WheelConfig::default();
        assert!(w.bump_stop_stiffness.is_zero()); // デフォルトはバンプストップ無効
        let expected = Fix128::from_ratio(9, 10);
        assert_eq!(w.bump_stop_threshold, expected); // 閾値 0.9
    }

    #[test]
    fn test_bump_stop_activation() {
        // バンプストップ有効: 高圧縮時に追加スプリング力
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_stop_stiffness = Fix128::from_int(100_000); // 非常に硬い
            w.bump_stop_threshold = Fix128::from_ratio(5, 10); // 50% 圧縮で発動
        }

        let mut vehicle_with_stop = Vehicle::new(config.clone());
        let mut vehicle_without_stop = Vehicle::new(VehicleConfig::default());

        // 低い位置から開始（高い圧縮を生じさせる）
        let mut chassis_with =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.35, 0.0), Fix128::from_int(1500));
        let mut chassis_without =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.35, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle_with_stop.update(&mut chassis_with, dt);
        vehicle_without_stop.update(&mut chassis_without, dt);

        // バンプストップありの方がサスペンション力が大きい
        let force_with: Fix128 = vehicle_with_stop
            .wheel_states
            .iter()
            .filter(|w| w.grounded)
            .map(|w| w.suspension_force)
            .fold(Fix128::ZERO, |a, b| a + b);
        let force_without: Fix128 = vehicle_without_stop
            .wheel_states
            .iter()
            .filter(|w| w.grounded)
            .map(|w| w.suspension_force)
            .fold(Fix128::ZERO, |a, b| a + b);

        assert!(
            force_with >= force_without,
            "Bump stop should increase suspension force: with={}, without={}",
            force_with.to_f32(),
            force_without.to_f32()
        );
    }

    #[test]
    fn test_bump_stop_no_effect_low_compression() {
        // 圧縮が閾値以下ならバンプストップは効かない
        let mut config = VehicleConfig::default();
        for w in &mut config.wheels {
            w.bump_stop_stiffness = Fix128::from_int(100_000);
            w.bump_stop_threshold = Fix128::from_ratio(95, 100); // 95%
        }

        let mut vehicle_with = Vehicle::new(config);
        let mut vehicle_without = Vehicle::new(VehicleConfig::default());

        // 高い位置 = 低い圧縮
        let mut chassis_with =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.55, 0.0), Fix128::from_int(1500));
        let mut chassis_without =
            RigidBody::new(Vec3Fix::from_f32(0.0, 0.55, 0.0), Fix128::from_int(1500));

        let dt = Fix128::from_ratio(1, 60);
        vehicle_with.update(&mut chassis_with, dt);
        vehicle_without.update(&mut chassis_without, dt);

        // 低圧縮では両者の力は同じ
        for (ws_with, ws_without) in vehicle_with
            .wheel_states
            .iter()
            .zip(vehicle_without.wheel_states.iter())
        {
            if ws_with.grounded && ws_without.grounded {
                let diff = (ws_with.suspension_force - ws_without.suspension_force)
                    .to_f32()
                    .abs();
                assert!(
                    diff < 0.01,
                    "Low compression: bump stop should have no effect, diff={diff}"
                );
            }
        }
    }
}

#[cfg(test)]
mod participant_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn resident() -> VehicleParticipant {
        VehicleParticipant::new(Vehicle::new(VehicleConfig::default()), 0)
    }

    /// The kind is `VHCL`; the digest separates the configuration, the
    /// chassis index and the input field.
    #[test]
    fn kind_and_digest() {
        let a = resident();
        assert_eq!(a.kind().get(), 0x5648_434c);
        let mut config = VehicleConfig::default();
        config.aero_drag = config.aero_drag + Fix128::ONE;
        assert_ne!(
            a.digest,
            VehicleParticipant::new(Vehicle::new(config), 0).digest
        );
        assert_ne!(
            a.digest,
            VehicleParticipant::new(Vehicle::new(VehicleConfig::default()), 1).digest
        );
        let with = resident().with_inputs(PortId::new(7));
        assert_ne!(a.digest, with.digest);
        assert_eq!(with.ports(), &[Port::reads_committed(PortId::new(7))]);
        assert!(a.ports().is_empty());
    }

    /// The payload reads back what was written, and is refused with the
    /// stated error for another length, version, digest, gear or grounded
    /// byte.
    #[test]
    fn payload_round_trip_and_refusals() {
        let mut a = resident();
        a.vehicle.current_gear = 2;
        a.vehicle.throttle = Fix128::from_ratio(1, 2);
        a.vehicle.wheel_states[1].grounded = true;
        a.vehicle.wheel_states[3].spin_angle = Fix128::from_int(7);
        let mut bytes = Vec::new();
        a.write_state(&mut bytes);
        let expected = VEHICLE_STATE_HEAD + 4 * VEHICLE_STATE_WHEEL;
        assert_eq!(bytes.len(), expected);
        let mut b = resident();
        assert_eq!(b.check_state(&bytes), Ok(()));
        b.read_state(&bytes);
        let mut again = Vec::new();
        b.write_state(&mut again);
        assert_eq!(again, bytes);
        assert_eq!(b.vehicle.wheel_states, a.vehicle.wheel_states);
        assert_eq!(
            b.check_state(&bytes[1..]),
            Err(StateError::Length {
                expected,
                found: expected - 1
            })
        );
        let mut bad = bytes.clone();
        bad[0] = 9;
        assert_eq!(b.check_state(&bad), Err(StateError::InvalidValue));
        let mut bad = bytes.clone();
        bad[4] ^= 1;
        assert_eq!(b.check_state(&bad), Err(StateError::InvalidValue));
        let mut bad = bytes.clone();
        bad[VEHICLE_STATE_HEAD + 12 * 16] = 2;
        assert_eq!(b.check_state(&bad), Err(StateError::InvalidValue));
    }

    use crate::world_participant::{
        FieldBoard, FieldLayout, FieldMode, FieldStage, ForceAccumulator,
    };

    const INPUTS: PortId = PortId::new(5);

    /// A chassis resting with every default wheel on the ground.
    fn chassis() -> RigidBody {
        let w = VehicleConfig::default().wheels[0];
        let y = w.suspension_rest + w.radius - w.local_position.y - Fix128::from_ratio(1, 10);
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO),
            Fix128::from_int(1200),
        )
    }

    fn board(values: &[Fix128]) -> FieldBoard {
        let mut b = FieldBoard::new();
        b.declare(
            INPUTS,
            FieldLayout::Channels {
                count: values.len(),
            },
            FieldMode::Replace,
        )
        .expect("declare");
        b.set(INPUTS, values).expect("set");
        b
    }

    /// Run one substep of `p` on `bodies`, with the input board if given;
    /// returns the result and the staged force on body 0.
    fn run(
        p: &mut VehicleParticipant,
        bodies: &[RigidBody],
        inputs: Option<&FieldBoard>,
    ) -> (Result<(), ParticipantFault>, Vec3Fix) {
        let h = Fix128::from_ratio(1, 240);
        let mut acc = ForceAccumulator::new(bodies.len());
        let mut stage = FieldStage::new();
        let ports = p.ports.clone();
        let result = {
            let ctx = SubstepCtx::new(bodies, &mut acc, 0, 4, h).expect("ctx");
            let mut ctx = match inputs {
                Some(b) => ctx.with_fields(b, &mut stage, &ports),
                None => ctx,
            };
            p.substep(&mut ctx, h)
        };
        let force = acc.force(0).unwrap_or(Vec3Fix::ZERO);
        (result, force)
    }

    /// Without inputs the staged force is `chassis_force` of the vehicle as
    /// built; the state advances as `chassis_force` advances it.
    #[test]
    fn substep_stages_the_chassis_force() {
        let bodies = [chassis()];
        let mut p = resident();
        let mut twin = Vehicle::new(VehicleConfig::default());
        let want = twin.chassis_force(&bodies[0], Fix128::from_ratio(1, 240));
        let (r, f) = run(&mut p, &bodies, None);
        assert_eq!(r, Ok(()));
        assert_eq!(f, want);
        assert_ne!(f, Vec3Fix::ZERO);
        assert_eq!(p.vehicle.wheel_states, twin.wheel_states);
    }

    /// Inputs in range are applied before the force; out of range, wrong
    /// length or a missing field are faults that stage nothing and leave the
    /// participant unchanged.
    #[test]
    fn inputs_are_checked_and_applied() {
        let bodies = [chassis()];
        let half = Fix128::from_ratio(1, 2);
        let mut p = resident().with_inputs(INPUTS);
        let ok = board(&[half, Fix128::ZERO, -half]);
        let (r, f) = run(&mut p, &bodies, Some(&ok));
        assert_eq!(r, Ok(()));
        assert_ne!(f, Vec3Fix::ZERO);
        assert_eq!((p.vehicle.throttle, p.vehicle.steering), (half, -half));

        for (values, fault) in [
            (
                vec![Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO],
                ParticipantFault::OutOfRange,
            ),
            (
                vec![Fix128::ZERO, -half, Fix128::ZERO],
                ParticipantFault::OutOfRange,
            ),
            (
                vec![Fix128::ZERO, Fix128::ZERO, Fix128::from_int(-2)],
                ParticipantFault::OutOfRange,
            ),
            (
                vec![Fix128::ZERO, Fix128::ZERO],
                ParticipantFault::InvalidState,
            ),
        ] {
            let mut q = resident().with_inputs(INPUTS);
            let mut before = Vec::new();
            q.write_state(&mut before);
            let (r, f) = run(&mut q, &bodies, Some(&board(&values)));
            assert_eq!(r, Err(fault), "{values:?}");
            assert_eq!(f, Vec3Fix::ZERO);
            let mut after = Vec::new();
            q.write_state(&mut after);
            assert_eq!(after, before);
        }
        let mut q = resident().with_inputs(INPUTS);
        assert_eq!(
            run(&mut q, &bodies, None).0,
            Err(ParticipantFault::InvalidState)
        );
    }

    /// A chassis index past the bodies is `InvalidState`; a static chassis
    /// is left alone (no force, no state change).
    #[test]
    fn chassis_index_and_static_chassis() {
        let mut p = VehicleParticipant::new(Vehicle::new(VehicleConfig::default()), 3);
        assert_eq!(
            run(&mut p, &[chassis()], None).0,
            Err(ParticipantFault::InvalidState)
        );
        let static_body = RigidBody::new_static(chassis().position);
        let mut p = resident();
        let (r, f) = run(&mut p, &[static_body], None);
        assert_eq!((r, f), (Ok(()), Vec3Fix::ZERO));
        assert_eq!(
            p.vehicle.wheel_states,
            Vehicle::new(VehicleConfig::default()).wheel_states
        );

        // with inputs, the static chassis is still left alone: the inputs are
        // not recorded in the vehicle state (payload and observation unchanged)
        let half = Fix128::from_ratio(1, 2);
        let mut p = resident().with_inputs(INPUTS);
        let mut before = Vec::new();
        p.write_state(&mut before);
        let mut seen_before = ObservationSink::new();
        p.observe(&mut seen_before);
        let (r, f) = run(&mut p, &[static_body], Some(&board(&[half, half, half])));
        assert_eq!((r, f), (Ok(()), Vec3Fix::ZERO));
        let mut after = Vec::new();
        p.write_state(&mut after);
        assert_eq!(after, before);
        let mut seen_after = ObservationSink::new();
        p.observe(&mut seen_after);
        assert_eq!(seen_after.values(), seen_before.values());
    }

    /// The observation reports speed, rpm, gear, inputs and grounded wheels
    /// in channel order.
    #[test]
    fn observation_channels() {
        let mut p = resident();
        p.vehicle.current_gear = 1;
        p.vehicle.throttle = Fix128::ONE;
        p.vehicle.wheel_states[0].grounded = true;
        p.vehicle.wheel_states[2].grounded = true;
        let mut sink = ObservationSink::new();
        p.observe(&mut sink);
        let v = sink.values();
        assert_eq!(v.len(), 7);
        assert_eq!(
            v[VEHICLE_OBS_GEAR as usize],
            (VEHICLE_OBS_GEAR, Fix128::ONE)
        );
        assert_eq!(
            v[VEHICLE_OBS_THROTTLE as usize],
            (VEHICLE_OBS_THROTTLE, Fix128::ONE)
        );
        assert_eq!(
            v[VEHICLE_OBS_GROUNDED as usize],
            (VEHICLE_OBS_GROUNDED, Fix128::from_int(2))
        );
        assert_eq!(v[VEHICLE_OBS_ENGINE_RPM as usize].1, p.vehicle.engine_rpm);
        assert_eq!(v[VEHICLE_OBS_SPEED_KMH as usize].1, Fix128::ZERO);
        assert_eq!(v[VEHICLE_OBS_BRAKE as usize].1, Fix128::ZERO);
        assert_eq!(v[VEHICLE_OBS_STEERING as usize].1, Fix128::ZERO);
    }

    /// A gear that does not fit `usize` is refused.
    #[test]
    fn gear_out_of_usize_is_refused() {
        let a = resident();
        let mut bytes = Vec::new();
        a.write_state(&mut bytes);
        bytes[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        if usize::try_from(u64::MAX).is_err() {
            assert_eq!(a.check_state(&bytes), Err(StateError::InvalidValue));
        } else {
            assert_eq!(a.check_state(&bytes), Ok(()));
        }
    }
}
