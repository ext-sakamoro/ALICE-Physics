//! Vehicle dynamics: per-wheel contact forces, wheel spin, brake torque,
//! ABS, tyre models, road surfaces and weather, powertrain.
//!
//! [`crate::vehicle::Vehicle`] is kept unchanged for existing callers. This
//! module is the model to use when stopping distance, cornering or load
//! transfer must follow the physics:
//!
//! - every wheel's suspension and tyre force is applied at its own contact
//!   point (`RigidBody::apply_impulse_at`), so steering yields yaw and
//!   braking / cornering shift load between axles and sides
//! - each wheel carries a spin state `ω` driven by drive torque, brake torque
//!   and the tyre's longitudinal force; brakes can lock a wheel, ABS keeps the
//!   slip ratio near its target
//! - tyre forces come from [`tire::TireModel`] (brush or Magic Formula) on the
//!   grip of [`surface::RoadCondition`] (material × weather × hydroplaning)
//! - the road is any [`surface::RoadSurface`] (plane, slope, height field,
//!   triangle mesh, SDF)
//!
//! Call [`DynamicVehicle::update`] once per frame before `PhysicsWorld::step`,
//! like the legacy model.

pub mod powertrain;
pub mod surface;
pub mod tire;

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;
use crate::vehicle::VehicleConfig;
use crate::wind_zone::WindZone;
use powertrain::Powertrain;
use surface::{RoadCondition, RoadSurface};
use tire::TireModel;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Anti-lock braking parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbsConfig {
    /// Slip ratio magnitude ABS regulates towards (e.g. 0.12).
    pub target_slip: Fix128,
    /// Below this forward speed (m/s) ABS is inactive and wheels may lock.
    pub min_speed: Fix128,
}

/// Brake system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrakeSystem {
    /// Maximum brake torque per front wheel (Nm) at full pedal.
    pub max_torque_front: Fix128,
    /// Maximum brake torque per rear wheel (Nm) at full pedal.
    pub max_torque_rear: Fix128,
    /// Maximum handbrake torque per rear wheel (Nm).
    pub handbrake_torque: Fix128,
    /// ABS, or `None` for none.
    pub abs: Option<AbsConfig>,
}

/// Aerodynamics of the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AeroConfig {
    /// Air density (kg/m³).
    pub air_density: Fix128,
    /// Drag area `C_d A` (m²).
    pub drag_area: Fix128,
    /// Lift area `C_l A` (m²); negative values are downforce.
    pub lift_area: Fix128,
}

/// Configuration of a [`DynamicVehicle`].
///
/// Wheel geometry, suspension and anti-roll bar come from `base`
/// (same meaning as in the legacy model: spring force =
/// `spring_stiffness · compression ratio`). A wheel is "front" when its
/// local `z` is positive.
#[derive(Clone, Debug)]
pub struct DynamicVehicleConfig {
    /// Wheel layout, suspension, anti-roll bar (legacy drive / brake / aero
    /// fields of `base` are not used by this model).
    pub base: VehicleConfig,
    /// Spin inertia of one wheel about its axle (kg m²).
    pub wheel_inertia: Fix128,
    /// Tyre model shared by all wheels.
    pub tire: TireModel,
    /// Tyre inflation pressure (kPa), used for hydroplaning.
    pub tyre_pressure_kpa: Fix128,
    /// Brakes.
    pub brakes: BrakeSystem,
    /// Engine, gearbox, differential (driven wheels are `base.wheels[i].driven`).
    pub powertrain: Powertrain,
    /// Body aerodynamics.
    pub aero: AeroConfig,
    /// Ackermann steering: inner / outer front wheel angles from the wheelbase
    /// and track; `false` steers every steerable wheel by the same angle.
    pub ackermann: bool,
    /// Slip-ratio / slip-angle velocity floor `v_floor` (m/s).
    pub slip_velocity_floor: Fix128,
}

/// Per-wheel runtime state of a [`DynamicVehicle`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WheelDynamicsState {
    /// Contact with the road this frame.
    pub grounded: bool,
    /// World-space contact point.
    pub contact_point: Vec3Fix,
    /// Road normal at the contact.
    pub contact_normal: Vec3Fix,
    /// Suspension compression ratio (0 extended .. 1 bottomed out).
    pub compression: Fix128,
    /// Normal load `F_z` (N), suspension + anti-roll, clamped at 0.
    pub normal_load: Fix128,
    /// Steering angle of this wheel (rad).
    pub steer_angle: Fix128,
    /// Spin `ω` (rad/s), positive rolling forward.
    pub omega: Fix128,
    /// Accumulated rotation angle (rad).
    pub spin_angle: Fix128,
    /// Slip ratio `κ` (see [`tire`] conventions).
    pub slip_ratio: Fix128,
    /// Lateral slip `tan α` (see [`tire`] conventions).
    pub slip_tan_alpha: Fix128,
    /// Tyre force along the wheel heading (N).
    pub longitudinal_force: Fix128,
    /// Tyre force along the wheel's lateral axis (N).
    pub lateral_force: Fix128,
    /// Brake torque applied this frame (Nm, magnitude).
    pub brake_torque: Fix128,
    /// ABS released / modulated this wheel this frame.
    pub abs_active: bool,
}

/// Inputs from the driver (or a controller) for one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverInput {
    /// Throttle `[0, 1]`.
    pub throttle: Fix128,
    /// Brake pedal `[0, 1]`.
    pub brake: Fix128,
    /// Handbrake `[0, 1]`.
    pub handbrake: Fix128,
    /// Steering `[-1, 1]`, positive steers right (towards local `+x`).
    pub steering: Fix128,
}

/// Environment of one frame: road condition, optional wind, time.
#[derive(Clone, Copy, Debug)]
pub struct Environment<'a> {
    /// Road material, weather, rolling resistance.
    pub condition: &'a RoadCondition,
    /// Wind acting on the body, if any (`WindZone::force_on`).
    pub wind: Option<&'a WindZone>,
    /// Simulation time (s) for gusts.
    pub time: Fix128,
    /// Gravity of the world (`world.config.gravity`).
    pub gravity: Vec3Fix,
}

/// Vehicle with per-wheel dynamics.
#[derive(Clone, Debug)]
pub struct DynamicVehicle {
    /// Configuration.
    pub config: DynamicVehicleConfig,
    /// Per-wheel state, same order as `config.base.wheels`.
    pub wheels: Vec<WheelDynamicsState>,
    /// Current inputs.
    pub input: DriverInput,
    /// Engine speed of the last update (rpm).
    pub engine_rpm: Fix128,
}

impl DynamicVehicleConfig {
    /// Passenger car on the legacy default wheel layout: brush tyres,
    /// 220 kPa, front/rear brakes 2500 / 1500 Nm, no ABS, powertrain from the
    /// legacy engine config, Ackermann steering.
    #[must_use]
    pub fn passenger_car() -> Self {
        todo!("STUB: DynamicVehicleConfig::passenger_car")
    }
}

impl DynamicVehicle {
    /// New vehicle at rest (all wheel spins 0).
    #[must_use]
    pub fn new(config: DynamicVehicleConfig) -> Self {
        let _ = config;
        todo!("STUB: DynamicVehicle::new")
    }

    /// Advance the wheels by `dt` and apply every wheel's suspension + tyre
    /// impulse at its contact point, plus rolling resistance, aerodynamics
    /// and wind, to `chassis`.
    pub fn update(
        &mut self,
        chassis: &mut RigidBody,
        road: &dyn RoadSurface,
        env: &Environment<'_>,
        dt: Fix128,
    ) {
        let _ = (chassis, road, env, dt);
        todo!("STUB: DynamicVehicle::update")
    }

    /// Forward speed of `chassis` along its local `+z` (m/s).
    #[must_use]
    pub fn forward_speed(chassis: &RigidBody) -> Fix128 {
        let _ = chassis;
        todo!("STUB: DynamicVehicle::forward_speed")
    }

    /// Number of grounded wheels.
    #[must_use]
    pub fn grounded_wheels(&self) -> usize {
        todo!("STUB: DynamicVehicle::grounded_wheels")
    }
}
