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

/// Environment of one frame: road condition, optional wind, gravity, time.
#[derive(Clone, Copy, Debug)]
pub struct Environment<'a> {
    /// Road material, weather, rolling resistance.
    pub condition: &'a RoadCondition,
    /// Wind acting on the body, if any (`WindZone::force_on`).
    pub wind: Option<&'a WindZone>,
    /// Gravity the world will integrate during the coming step (pass
    /// `world.config.gravity`; scaled by `chassis.gravity_scale` here).
    ///
    /// The vehicle applies its impulses at the start of the frame and the
    /// world adds gravity afterwards, so static friction must already absorb
    /// the in-plane part of `g · dt` that arrives within the frame; without it
    /// a car held on a slope creeps by about `g sinθ dt / 2` per frame.
    pub gravity: Vec3Fix,
    /// Simulation time (s) for gusts.
    pub time: Fix128,
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
    /// Passenger car on the legacy default wheel layout
    /// ([`VehicleConfig::default`]: track 1.6 m, wheelbase 2.4 m, rear-wheel
    /// drive): brush tyres ([`tire::BrushTire::passenger_car`]), 220 kPa,
    /// front / rear brakes 2500 / 1500 Nm per wheel, handbrake 1500 Nm per
    /// rear wheel, no ABS, powertrain from the legacy engine config and gear
    /// table ([`Powertrain::from_engine_config`]), Ackermann steering,
    /// wheel inertia 1.2 kg m², air density 1.225 kg/m³, `C_d A` 0.66 m²,
    /// `C_l A` 0, slip velocity floor 0.5 m/s.
    #[must_use]
    pub fn passenger_car() -> Self {
        let base = VehicleConfig::default();
        let powertrain = Powertrain::from_engine_config(&base.engine, &base.gear_ratios);
        Self {
            base,
            wheel_inertia: Fix128::from_ratio(12, 10),
            tire: TireModel::Brush(tire::BrushTire::passenger_car()),
            tyre_pressure_kpa: Fix128::from_int(220),
            brakes: BrakeSystem {
                max_torque_front: Fix128::from_int(2500),
                max_torque_rear: Fix128::from_int(1500),
                handbrake_torque: Fix128::from_int(1500),
                abs: None,
            },
            powertrain,
            aero: AeroConfig {
                air_density: Fix128::from_ratio(1225, 1000),
                drag_area: Fix128::from_ratio(66, 100),
                lift_area: Fix128::ZERO,
            },
            ackermann: true,
            slip_velocity_floor: Fix128::from_ratio(1, 2),
        }
    }
}

/// Per-wheel scratch of one `update`.
#[derive(Clone, Copy, Debug, Default)]
struct WheelScratch {
    /// Contact present and the wheel frame is well defined.
    loaded: bool,
    /// Contact point relative to the chassis centre of mass.
    arm: Vec3Fix,
    /// Wheel heading projected onto the contact plane (unit).
    x_dir: Vec3Fix,
    /// `normal × x_dir`.
    y_dir: Vec3Fix,
    /// Contact-point velocity along `x_dir` / `y_dir` (start of frame).
    v_x: Fix128,
    v_y: Fix128,
    /// `max(|v_x|, v_floor)`.
    denom: Fix128,
    /// Road grip at this contact (`Some` exactly when `loaded`).
    grip: Option<crate::anisotropic_friction::AnisotropicFriction>,
    /// Longitudinal tyre force at the start-of-frame spin.
    fx_start: Fix128,
    /// Slope `∂F_x/∂κ` used by the implicit spin update (0 = explicit).
    slope: Fix128,
    /// Pedal + handbrake torque before ABS (Nm).
    brake: Fix128,
    /// Drive torque on this wheel (open differential share, Nm).
    drive: Fix128,
}

/// `x` clamped to `[lo, hi]`.
fn clamp(x: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Coulomb brake: reduce `|omega|` by `delta ≥ 0` without crossing zero.
fn coulomb(omega: Fix128, delta: Fix128) -> Fix128 {
    if omega > Fix128::ZERO {
        let w = omega - delta;
        if w < Fix128::ZERO {
            Fix128::ZERO
        } else {
            w
        }
    } else if omega < Fix128::ZERO {
        let w = omega + delta;
        if w > Fix128::ZERO {
            Fix128::ZERO
        } else {
            w
        }
    } else {
        Fix128::ZERO
    }
}

/// Effective mass of `body` at arm `r` along unit `d`:
/// `1 / (m⁻¹ + (r×d)·I⁻¹(r×d))`, or 0 when the body cannot move.
fn effective_mass(body: &RigidBody, r: Vec3Fix, d: Vec3Fix) -> Fix128 {
    let rd = r.cross(d);
    let inv = body.inv_mass + rd.dot(body.world_inv_inertia_apply(rd));
    if inv <= Fix128::ZERO {
        Fix128::ZERO
    } else {
        Fix128::ONE / inv
    }
}

/// Clamp a friction impulse `j` (along some direction) so that it never
/// reverses the slip velocity `slip` it opposes: friction never pushes in the
/// direction of the slip, and at most `m_eff · |slip|` against it.
fn clamp_friction(j: Fix128, slip: Fix128, m_eff: Fix128) -> Fix128 {
    if j.is_zero() || slip.is_zero() {
        return Fix128::ZERO;
    }
    if (j > Fix128::ZERO) == (slip > Fix128::ZERO) {
        return Fix128::ZERO;
    }
    let limit = m_eff * slip.abs();
    if j > limit {
        limit
    } else if j < -limit {
        -limit
    } else {
        j
    }
}

/// Contact-point velocity of `body` at arm `r`.
fn point_velocity(body: &RigidBody, r: Vec3Fix) -> Vec3Fix {
    body.velocity + body.angular_velocity.cross(r)
}

impl DynamicVehicle {
    /// New vehicle at rest (all wheel spins 0, engine at idle).
    #[must_use]
    pub fn new(config: DynamicVehicleConfig) -> Self {
        let n = config.base.wheels.len();
        let engine_rpm = config.powertrain.idle_rpm;
        let mut wheels = Vec::with_capacity(n);
        wheels.resize(n, WheelDynamicsState::default());
        Self {
            config,
            wheels,
            input: DriverInput::default(),
            engine_rpm,
        }
    }

    /// Steering angle of every wheel for steering input `s ∈ [-1, 1]`.
    ///
    /// The commanded angle `δ = s · max_steer_angle` is the equivalent
    /// bicycle-model angle. With `ackermann`, the kinematic turn radius about
    /// the rear-axle centre is `R = L / tan δ` and a symmetric front axle gets
    /// `tan δ_in = L / (R − t/2)`, `tan δ_out = L / (R + t/2)` (`L` = front
    /// minus rear axle `z`, `t` = front-wheel `x` spacing, inner = the side
    /// the car turns towards); `δ = 0` gives 0 on every wheel.
    ///
    /// For general layouts the same rule is applied per wheel, as follows.
    /// The reference angle of a steerable wheel is `δ₀ = s · max_steer_angle`.
    /// With `ackermann`, a front wheel (`z > 0`) at wheelbase
    /// `L = z − z_rear` (`z_rear` = mean `z` of the wheels with `z ≤ 0`) and
    /// lateral offset `d = x − x_c` (`x_c` = mean `x` of all wheels) gets
    /// `tan δ = L tan δ₀ / (L − d tan δ₀)`, i.e. `tan δ = L / (R − d)` with the
    /// turn radius `R = L / tan δ₀` (the inner wheel of a symmetric axle at
    /// `d = ±t/2` gets `atan(L / (R − t/2))`, the outer `atan(L / (R + t/2))`).
    /// A wheel without a rear axle behind it, or for which `L − d tan δ₀ ≤ 0`
    /// (turn centre inside the track), keeps `δ₀`.
    fn steer_angles(&self, steering: Fix128) -> Vec<Fix128> {
        let wheels = &self.config.base.wheels;
        let n = wheels.len();
        let mut out = Vec::with_capacity(n);
        let mut x_sum = Fix128::ZERO;
        let mut rear_sum = Fix128::ZERO;
        let mut rear_n: i64 = 0;
        for w in wheels {
            x_sum = x_sum + w.local_position.x;
            if w.local_position.z <= Fix128::ZERO {
                rear_sum = rear_sum + w.local_position.z;
                rear_n += 1;
            }
        }
        let x_c = if n == 0 {
            Fix128::ZERO
        } else {
            x_sum / Fix128::from_int(n as i64)
        };
        let z_rear = if rear_n == 0 {
            None
        } else {
            Some(rear_sum / Fix128::from_int(rear_n))
        };
        for w in wheels {
            if w.max_steer_angle <= Fix128::ZERO {
                out.push(Fix128::ZERO);
                continue;
            }
            let delta0 = steering * w.max_steer_angle;
            let mut delta = delta0;
            if self.config.ackermann && w.local_position.z > Fix128::ZERO && !delta0.is_zero() {
                if let Some(zr) = z_rear {
                    let l = w.local_position.z - zr;
                    let d = w.local_position.x - x_c;
                    let (s, c) = delta0.sin_cos();
                    if l > Fix128::ZERO && c > Fix128::ZERO {
                        let t0 = s / c;
                        let den = l - d * t0;
                        if den > Fix128::ZERO {
                            delta = (l * t0 / den).atan();
                        }
                    }
                }
            }
            out.push(delta);
        }
        out
    }

    /// Advance the wheels by `dt` and apply every wheel's suspension + tyre
    /// impulse at its contact point, plus rolling resistance, aerodynamics
    /// and wind, to `chassis`.
    ///
    /// # Model
    ///
    /// 1. **Contact**: each attachment `p + R·local_position` probes the road
    ///    along `−up` up to `suspension_rest + radius`; compression ratio
    ///    `c = 1 − (distance − radius)/rest` clamped to `[0, 1]`.
    /// 2. **Suspension** (legacy meaning): `k (1 + progressive·c) c` plus the
    ///    bump stop, minus damping on the attachment-point velocity
    ///    `(v + ω × r)·up` (bump coefficient while compressing, rebound while
    ///    extending, `damping` when either is 0).
    ///
    ///    The suspension is coupled explicitly, one impulse per frame. For a
    ///    wheel carrying the mass share `m_share` (≈ `m / wheel count` in
    ///    heave) with local stiffness `k_eff = ∂F/∂x` (spring, progressive
    ///    term and bump stop, divided by `rest`) and damping `c`, the
    ///    impulse-then-integrate update is stable only while
    ///    `(ω_n dt)² + 2 c dt / m_share < 4` with `ω_n = √(k_eff / m_share)`
    ///    (undamped: `ω_n dt < 2`; with `ζ = c / (2 m_share ω_n)`:
    ///    `(ω_n dt)² + 4 ζ ω_n dt < 4`). A light chassis on stiff springs at a
    ///    coarse `dt` violates it and oscillates with growing amplitude; this
    ///    is the documented limit of the frame-level coupling, not clamped.
    /// 3. **Anti-roll** on the legacy pairs (0,1), (2,3): with
    ///    `diff = c_left − c_right` the more compressed wheel gains
    ///    `k_ar · diff` of load and the other loses it (a grounded wheel only).
    ///    The legacy model pushes the compressed side *down*; that sign is
    ///    harmless at the centre of mass but would be a negative roll
    ///    stiffness when applied at the contact points, so it is not copied.
    ///    `F_z = max(0, suspension + anti-roll)`.
    /// 4. **Wheel frame**: heading `forward cos δ + right sin δ` projected onto
    ///    the contact plane = `x`, `y = n × x`; slip per [`tire`] conventions
    ///    with `v_floor = slip_velocity_floor`.
    /// 5. **Spin** `I ω̇ = T_drive − T_brake − F_x r`, see below.
    /// 6. **Impulses**: `F_z up dt` at every contact point
    ///    (`apply_impulse_at`), then, wheel by wheel on the updated chassis
    ///    velocity, the tyre impulse `(F_x x + F_y y) dt` at the contact point
    ///    clamped per axis by the effective mass
    ///    `m_d = 1/(m⁻¹ + (r×d)·I⁻¹(r×d))` so that it never reverses the slip
    ///    it opposes (`v_x − ω r` along `x`, `v_y` along `y`). The slip used by
    ///    the clamp is the predicted end-of-frame one, i.e. it includes the
    ///    in-plane gravity `(g·d) dt` that the world adds during the step, so
    ///    static friction already holds against it. Then rolling resistance
    ///    `C_rr F_z` against `v_x` on wheels with `ω ≠ 0`, clamped the same way
    ///    on `v_x`. The stored `longitudinal_force` / `lateral_force` are the
    ///    applied (clamped) values.
    /// 7. **Aerodynamics** at the centre of mass: `v_rel = v − w` (`w` is the
    ///    wind of `env.wind` when its shape contains the chassis position),
    ///    drag `−½ ρ C_dA |v_rel| v_rel`, lift `½ ρ C_lA |v_rel|²` along `up`.
    ///
    /// # Wheel spin
    ///
    /// Semi-implicit in the tyre force: with `F_x(ω') ≈ F_x(ω) + C r/v̄ (ω' − ω)`
    /// (`C = longitudinal_slope(F_z, μ_x,static)`, `v̄ = max(|v_x|, v_floor)`,
    /// `v_x` frozen over the frame) the free spin is
    /// `ω_f = ω + dt (T_drive − Σ F_x(ω) r) / I_eff`,
    /// `I_eff = Σ I + dt Σ C r² / v̄`. The slope is used only while the tyre is
    /// in its linear range (`|C κ| ≤ μ_x,static F_z`); a saturated tyre is
    /// integrated explicitly (slope 0). A massless wheel (`wheel_inertia ≤ 0`)
    /// always uses the slope, i.e. it jumps to its quasi-static balance.
    /// Brakes are Coulomb: `ω' = ω_f` reduced in magnitude by
    /// `dt T_brake / I_eff` without crossing zero, so a large enough torque
    /// locks the wheel (`ω' = 0`) within one frame and never spins it
    /// backwards. The tyre force applied to the chassis is evaluated at `ω'`.
    /// A spin group whose `I_eff` is 0 (massless wheel off the ground) keeps
    /// its spin.
    ///
    /// Driven wheels receive `powertrain.axle_torque(throttle, mean driven ω)`:
    /// split equally with [`powertrain::Differential::Open`], or with
    /// [`powertrain::Differential::Locked`] the driven wheels form one spin
    /// group (inertias, torques, slopes and brake torques summed, one shared
    /// `ω`). Rear wheels (`z ≤ 0`) take the handbrake; pedal and handbrake
    /// torques act only on wheels with `has_brake`.
    ///
    /// # ABS
    ///
    /// When `abs` is set and `forward_speed > min_speed`, a braked spin group
    /// whose braked spin `ω'` gives some member `κ < −target_slip` has its
    /// brake torque lowered (deterministically, in one step) to the value
    /// that puts the group exactly at the slip limit on the linearised wheel
    /// model: `ω_t = max_i v_x,i (1 − target_slip) / r_i`,
    /// `T = clamp((ω_f − ω_t) I_eff / dt, 0, T_pedal)`; the members' stored
    /// `brake_torque` is scaled by `T / T_pedal` and `abs_active` is set.
    ///
    /// # Degenerate inputs
    ///
    /// - `dt ≤ 0` or a static chassis (`inv_mass = 0`): nothing changes (no
    ///   impulse, wheel state and `engine_rpm` untouched)
    /// - no wheels: only aerodynamics is applied; `engine_rpm = idle_rpm`
    /// - wheels off the ground: `grounded = false`, zero load and tyre force;
    ///   the spin still follows drive and brake torque
    /// - `self.wheels` of the wrong length is resized to the wheel count
    pub fn update(
        &mut self,
        chassis: &mut RigidBody,
        road: &dyn RoadSurface,
        env: &Environment<'_>,
        dt: Fix128,
    ) {
        if dt <= Fix128::ZERO || chassis.is_static() {
            return;
        }
        let n = self.config.base.wheels.len();
        if self.wheels.len() != n {
            self.wheels.resize(n, WheelDynamicsState::default());
        }

        let forward = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Z);
        let right = chassis.rotation.rotate_vec(Vec3Fix::UNIT_X);
        let up = chassis.rotation.rotate_vec(Vec3Fix::UNIT_Y);
        let gravity = env.gravity * chassis.gravity_scale;

        let throttle = clamp(self.input.throttle, Fix128::ZERO, Fix128::ONE);
        let pedal = clamp(self.input.brake, Fix128::ZERO, Fix128::ONE);
        let handbrake = clamp(self.input.handbrake, Fix128::ZERO, Fix128::ONE);
        let steering = clamp(self.input.steering, Fix128::NEG_ONE, Fix128::ONE);
        let steer = self.steer_angles(steering);

        // --- 1-3: contact, suspension, anti-roll -------------------------
        let mut susp = Vec::with_capacity(n);
        for i in 0..n {
            let wc = self.config.base.wheels[i];
            let attach = chassis.position + chassis.rotation.rotate_vec(wc.local_position);
            let st = &mut self.wheels[i];
            st.steer_angle = steer[i];
            st.abs_active = false;
            st.brake_torque = Fix128::ZERO;
            st.longitudinal_force = Fix128::ZERO;
            st.lateral_force = Fix128::ZERO;
            st.slip_ratio = Fix128::ZERO;
            st.slip_tan_alpha = Fix128::ZERO;
            match road.probe(attach, -up, wc.suspension_rest + wc.radius) {
                Some(hit) => {
                    let c = clamp(
                        Fix128::ONE - (hit.distance - wc.radius) / wc.suspension_rest,
                        Fix128::ZERO,
                        Fix128::ONE,
                    );
                    let mut spring =
                        wc.spring_stiffness * (Fix128::ONE + wc.progressive_rate * c) * c;
                    if !wc.bump_stop_stiffness.is_zero() && c > wc.bump_stop_threshold {
                        spring = spring + wc.bump_stop_stiffness * (c - wc.bump_stop_threshold);
                    }
                    let v_up = point_velocity(chassis, attach - chassis.position).dot(up);
                    let coeff = if v_up < Fix128::ZERO {
                        if wc.bump_damping.is_zero() {
                            wc.damping
                        } else {
                            wc.bump_damping
                        }
                    } else if wc.rebound_damping.is_zero() {
                        wc.damping
                    } else {
                        wc.rebound_damping
                    };
                    st.grounded = true;
                    st.contact_point = hit.point;
                    st.contact_normal = hit.normal;
                    st.compression = c;
                    susp.push(spring - coeff * v_up);
                }
                None => {
                    st.grounded = false;
                    st.compression = Fix128::ZERO;
                    susp.push(Fix128::ZERO);
                }
            }
        }
        let k_ar = self.config.base.anti_roll_stiffness;
        if !k_ar.is_zero() {
            for pair in 0..n / 2 {
                let (l, r) = (2 * pair, 2 * pair + 1);
                let f = k_ar * (self.wheels[l].compression - self.wheels[r].compression);
                if self.wheels[l].grounded {
                    susp[l] = susp[l] + f;
                }
                if self.wheels[r].grounded {
                    susp[r] = susp[r] - f;
                }
            }
        }
        for i in 0..n {
            let st = &mut self.wheels[i];
            st.normal_load = if st.grounded && susp[i] > Fix128::ZERO {
                susp[i]
            } else {
                Fix128::ZERO
            };
        }

        // --- 4: wheel frames, grip, start-of-frame tyre force -------------
        let floor = self.config.slip_velocity_floor;
        let massless = self.config.wheel_inertia <= Fix128::ZERO;
        let mut sc = Vec::with_capacity(n);
        for i in 0..n {
            let wc = self.config.base.wheels[i];
            let st = self.wheels[i];
            let mut s = WheelScratch::default();
            let is_rear = wc.local_position.z <= Fix128::ZERO;
            if wc.has_brake {
                let per = if is_rear {
                    self.config.brakes.max_torque_rear
                } else {
                    self.config.brakes.max_torque_front
                };
                s.brake = pedal * per;
                if is_rear {
                    s.brake = s.brake + handbrake * self.config.brakes.handbrake_torque;
                }
            }
            if st.grounded {
                let (sn, cs) = st.steer_angle.sin_cos();
                let heading = forward * cs + right * sn;
                let nrm = st.contact_normal;
                if let Some(x_dir) = (heading - nrm * heading.dot(nrm)).try_normalize() {
                    let arm = st.contact_point - chassis.position;
                    let vc = point_velocity(chassis, arm);
                    s.loaded = true;
                    s.arm = arm;
                    s.x_dir = x_dir;
                    s.y_dir = nrm.cross(x_dir);
                    s.v_x = vc.dot(s.x_dir);
                    s.v_y = vc.dot(s.y_dir);
                    s.denom = if s.v_x.abs() > floor {
                        s.v_x.abs()
                    } else {
                        floor
                    };
                    let speed = (s.v_x * s.v_x + s.v_y * s.v_y).sqrt();
                    let grip = env.condition.grip(speed, self.config.tyre_pressure_kpa);
                    s.grip = Some(grip);
                    if st.normal_load > Fix128::ZERO {
                        let kappa = (st.omega * wc.radius - s.v_x) / s.denom;
                        let f = self.config.tire.force(&tire::TireInput {
                            slip_ratio: kappa,
                            slip_tan_alpha: s.v_y / s.denom,
                            normal_load: st.normal_load,
                            grip,
                        });
                        s.fx_start = f.longitudinal;
                        let mu = grip.longitudinal_static;
                        let c = self.config.tire.longitudinal_slope(st.normal_load, mu);
                        if massless || (c * kappa).abs() <= mu * st.normal_load {
                            s.slope = c;
                        }
                    }
                }
            }
            sc.push(s);
        }

        // --- 5: drive torque and spin groups -----------------------------
        let driven: Vec<usize> = (0..n).filter(|&i| self.config.base.wheels[i].driven).collect();
        if !driven.is_empty() {
            let mut sum = Fix128::ZERO;
            for &i in &driven {
                sum = sum + self.wheels[i].omega;
            }
            let mean = sum / Fix128::from_int(driven.len() as i64);
            let axle = self.config.powertrain.axle_torque(throttle, mean);
            let locked = self.config.powertrain.differential == powertrain::Differential::Locked;
            let share = if locked {
                axle
            } else {
                axle / Fix128::from_int(driven.len() as i64)
            };
            if locked {
                // whole axle torque goes to the group; carried on its first member
                sc[driven[0]].drive = share;
            } else {
                for &i in &driven {
                    sc[i].drive = share;
                }
            }
        }
        let locked_group = !driven.is_empty()
            && self.config.powertrain.differential == powertrain::Differential::Locked;
        let mut groups: Vec<Vec<usize>> = Vec::new();
        if locked_group {
            groups.push(driven.clone());
        }
        for i in 0..n {
            if !(locked_group && self.config.base.wheels[i].driven) {
                groups.push([i].to_vec());
            }
        }
        let fwd_speed = chassis.velocity.dot(forward);
        for g in &groups {
            self.solve_spin_group(g, &sc, dt, fwd_speed);
        }

        // --- tyre force at the updated spin ---------------------------------
        let mut fx = Vec::with_capacity(n);
        let mut fy = Vec::with_capacity(n);
        for i in 0..n {
            let wc = self.config.base.wheels[i];
            let s = sc[i];
            let st = &mut self.wheels[i];
            st.spin_angle = st.spin_angle + st.omega * dt;
            let Some(grip) = s.grip else {
                fx.push(Fix128::ZERO);
                fy.push(Fix128::ZERO);
                continue;
            };
            let kappa = (st.omega * wc.radius - s.v_x) / s.denom;
            let tan_a = s.v_y / s.denom;
            st.slip_ratio = kappa;
            st.slip_tan_alpha = tan_a;
            if st.normal_load > Fix128::ZERO {
                let f = self.config.tire.force(&tire::TireInput {
                    slip_ratio: kappa,
                    slip_tan_alpha: tan_a,
                    normal_load: st.normal_load,
                    grip,
                });
                fx.push(f.longitudinal);
                fy.push(f.lateral);
            } else {
                fx.push(Fix128::ZERO);
                fy.push(Fix128::ZERO);
            }
        }

        // --- 6: impulses at the contact points --------------------------------
        for i in 0..n {
            let st = self.wheels[i];
            if st.grounded && st.normal_load > Fix128::ZERO {
                chassis.apply_impulse_at(up * (st.normal_load * dt), st.contact_point);
            }
        }
        let c_rr = env.condition.rolling_resistance;
        for i in 0..n {
            let s = sc[i];
            if !s.loaded {
                continue;
            }
            let radius = self.config.base.wheels[i].radius;
            let point = self.wheels[i].contact_point;
            let omega = self.wheels[i].omega;
            let g_x = gravity.dot(s.x_dir) * dt;
            let g_y = gravity.dot(s.y_dir) * dt;

            let vc = point_velocity(chassis, s.arm);
            let mx = effective_mass(chassis, s.arm, s.x_dir);
            let jx = clamp_friction(fx[i] * dt, vc.dot(s.x_dir) + g_x - omega * radius, mx);
            if !jx.is_zero() {
                chassis.apply_impulse_at(s.x_dir * jx, point);
            }
            let vc = point_velocity(chassis, s.arm);
            let my = effective_mass(chassis, s.arm, s.y_dir);
            let jy = clamp_friction(fy[i] * dt, vc.dot(s.y_dir) + g_y, my);
            if !jy.is_zero() {
                chassis.apply_impulse_at(s.y_dir * jy, point);
            }
            self.wheels[i].longitudinal_force = jx / dt;
            self.wheels[i].lateral_force = jy / dt;

            let load = self.wheels[i].normal_load;
            if !omega.is_zero() && !c_rr.is_zero() && load > Fix128::ZERO {
                let vc = point_velocity(chassis, s.arm);
                let vx = vc.dot(s.x_dir);
                let mag = c_rr * load * dt;
                let want = if vx > Fix128::ZERO {
                    -mag
                } else if vx < Fix128::ZERO {
                    mag
                } else {
                    Fix128::ZERO
                };
                let j = clamp_friction(want, vx, mx);
                if !j.is_zero() {
                    chassis.apply_impulse_at(s.x_dir * j, point);
                }
            }
        }

        // --- 7: aerodynamics and wind -------------------------------------------
        let wind = match env.wind {
            Some(w) if w.shape.contains(chassis.position) => w.instantaneous_wind_vector(env.time),
            _ => Vec3Fix::ZERO,
        };
        let v_rel = chassis.velocity - wind;
        let speed_sq = v_rel.length_squared();
        if !speed_sq.is_zero() {
            let half_rho = self.config.aero.air_density.half();
            let speed = speed_sq.sqrt();
            let drag = v_rel * (-(half_rho * self.config.aero.drag_area * speed));
            let lift = up * (half_rho * self.config.aero.lift_area * speed_sq);
            chassis.apply_impulse((drag + lift) * dt);
        }

        // --- engine speed -----------------------------------------------------------
        self.engine_rpm = if driven.is_empty() {
            self.config.powertrain.idle_rpm
        } else {
            let mut sum = Fix128::ZERO;
            for &i in &driven {
                sum = sum + self.wheels[i].omega;
            }
            self.config
                .powertrain
                .engine_rpm(sum / Fix128::from_int(driven.len() as i64))
        };
    }

    /// Spin update of one group of wheels sharing `ω` (see [`Self::update`]).
    fn solve_spin_group(&mut self, members: &[usize], sc: &[WheelScratch], dt: Fix128, fwd: Fix128) {
        if members.is_empty() {
            return;
        }
        let inertia = if self.config.wheel_inertia > Fix128::ZERO {
            self.config.wheel_inertia
        } else {
            Fix128::ZERO
        };
        let mut i_eff = Fix128::ZERO;
        let mut torque = Fix128::ZERO;
        let mut brake = Fix128::ZERO;
        let mut omega_sum = Fix128::ZERO;
        for &i in members {
            let r = self.config.base.wheels[i].radius;
            let s = sc[i];
            i_eff = i_eff + inertia;
            if s.loaded {
                i_eff = i_eff + dt * s.slope * r * r / s.denom;
            }
            torque = torque + s.drive - s.fx_start * r;
            brake = brake + s.brake;
            omega_sum = omega_sum + self.wheels[i].omega;
        }
        let omega = omega_sum / Fix128::from_int(members.len() as i64);
        if i_eff.is_zero() {
            for &i in members {
                self.wheels[i].omega = omega;
                self.wheels[i].brake_torque = sc[i].brake;
            }
            return;
        }
        let omega_free = omega + dt * torque / i_eff;
        let mut applied = brake;
        let mut new_omega = coulomb(omega_free, dt * brake / i_eff);
        let mut abs_on = false;
        if let Some(abs) = self.config.brakes.abs {
            if brake > Fix128::ZERO && fwd > abs.min_speed {
                let mut target: Option<Fix128> = None;
                for &i in members {
                    let s = sc[i];
                    let r = self.config.base.wheels[i].radius;
                    if s.loaded && s.v_x > Fix128::ZERO && r > Fix128::ZERO {
                        let w = s.v_x * (Fix128::ONE - abs.target_slip) / r;
                        target = Some(match target {
                            Some(t) if t >= w => t,
                            _ => w,
                        });
                    }
                }
                if let Some(w_t) = target {
                    if new_omega < w_t {
                        applied = clamp((omega_free - w_t) * i_eff / dt, Fix128::ZERO, brake);
                        new_omega = coulomb(omega_free, dt * applied / i_eff);
                        abs_on = true;
                    }
                }
            }
        }
        for &i in members {
            self.wheels[i].omega = new_omega;
            self.wheels[i].abs_active = abs_on;
            self.wheels[i].brake_torque = if brake.is_zero() {
                Fix128::ZERO
            } else {
                sc[i].brake * applied / brake
            };
        }
    }

    /// Forward speed of `chassis` along its local `+z` (m/s).
    #[must_use]
    pub fn forward_speed(chassis: &RigidBody) -> Fix128 {
        chassis
            .velocity
            .dot(chassis.rotation.rotate_vec(Vec3Fix::UNIT_Z))
    }

    /// Number of grounded wheels.
    #[must_use]
    pub fn grounded_wheels(&self) -> usize {
        self.wheels.iter().filter(|w| w.grounded).count()
    }
}
