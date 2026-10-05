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
pub mod scenario;
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
    /// Normal load `F_z` (N): road-normal component of the strut force
    /// (suspension + anti-roll), clamped at 0.
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
    /// Static-friction anchor: world-space road point a locked wheel's
    /// contact is held to, `None` while rolling or sliding (see
    /// [`DynamicVehicle::update`], "Static friction").
    pub anchor: Option<Vec3Fix>,
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
    /// table ([`Powertrain::from_engine_config`]), Ackermann steering on the
    /// front axle (the legacy default gives the rear wheels `max_steer_angle`
    /// 0.5 as well, which would steer them in phase; it is set to 0 here),
    /// wheel inertia 1.2 kg m², air density 1.225 kg/m³, `C_d A` 0.66 m²,
    /// `C_l A` 0, slip velocity floor 0.5 m/s.
    #[must_use]
    pub fn passenger_car() -> Self {
        let mut base = VehicleConfig::default();
        rear_wheels_unsteered(&mut base);
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

/// Set `max_steer_angle = 0` on every rear wheel (`z ≤ 0`).
fn rear_wheels_unsteered(base: &mut VehicleConfig) {
    for w in &mut base.wheels {
        if w.local_position.z <= Fix128::ZERO {
            w.max_steer_angle = Fix128::ZERO;
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

/// Tyre force for slip `(κ, tan α)` at contact-point forward velocity `v_x`.
/// The tyre models are written for forward travel (the brush theoretical slip
/// `σ = κ / (1 + κ)` treats `κ = +1` as driving), so for `v_x < 0` the wheel
/// frame is mirrored (`x, y → −x, −y`): the model is evaluated at
/// `(−κ, −tan α)` and the force negated. A wheel locked while sliding
/// backwards thus sees `κ' = −1`, full sliding.
fn tyre_force(
    model: &TireModel,
    kappa: Fix128,
    tan_a: Fix128,
    v_x: Fix128,
    load: Fix128,
    grip: crate::anisotropic_friction::AnisotropicFriction,
) -> tire::TireForce {
    let mirror = v_x < Fix128::ZERO;
    let (k, t) = if mirror {
        (-kappa, -tan_a)
    } else {
        (kappa, tan_a)
    };
    let f = model.force(&tire::TireInput {
        slip_ratio: k,
        slip_tan_alpha: t,
        normal_load: load,
        grip,
    });
    if mirror {
        tire::TireForce {
            longitudinal: -f.longitudinal,
            lateral: -f.lateral,
        }
    } else {
        f
    }
}

/// One Newton step of the implicit friction impulse `λ = F(s(λ)) dt` for a
/// tyre with slope `c = −∂F/∂(slip velocity) ≥ 0` acting on effective mass
/// `m`: `Δλ = residual / (1 + c dt / m)`. With `m = 0` (immovable contact)
/// the step is 0.
fn newton_step(residual: Fix128, c: Fix128, dt: Fix128, m: Fix128) -> Fix128 {
    if m <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    residual / (Fix128::ONE + c * dt / m)
}

/// Small-slip lateral slope `−∂F_y/∂tan α` (N per unit `tan α`):
/// brush `C_α`, Magic Formula `B_y C_y μ_y,s F_z`; clamped at 0.
fn tyre_lateral_slope(
    model: &TireModel,
    load: Fix128,
    grip: crate::anisotropic_friction::AnisotropicFriction,
) -> Fix128 {
    let c = match model {
        TireModel::Brush(b) => b.cornering_stiffness,
        TireModel::MagicFormula(m) => m.b_y * m.c_y * grip.transverse_static * load,
    };
    if c > Fix128::ZERO {
        c
    } else {
        Fix128::ZERO
    }
}

/// Maximum Gauss-Seidel sweeps of the friction solve; the solve stops
/// earlier once no impulse changes by more than [`friction_tolerance`].
///
/// Symmetric scenes must come out symmetric: four locked wheels sliding on
/// an isotropic road with the front wheels steered must give zero yaw, and
/// the two driven wheels of a mirror-symmetric car launching under an open
/// differential must carry equal force (`tests/analytic_vehicle_dynamics.rs`).
/// Gauss-Seidel (wheel index order) solves one side first, so the
/// unconverged residual shows up as asymmetry. Measured: max yaw rate
/// 1.2e-5 rad/s at 8 sweeps, 1.3e-9 at 16, 2.4e-13 at 24; launch left /
/// right force difference 1.3e-5 N at 32 sweeps, below 1e-6 N at 64 (the
/// saturated launch converges slowest: the ellipse projection makes the
/// sweep non-smooth). Alternating the order between sweeps does not speed
/// this up (1.2e-4 N at 16) and lets a light wheel of a car held on a slope
/// take its static limit first and break away (1.7 mm drift in one frame),
/// so the order stays fixed.
const FRICTION_SWEEPS: usize = 64;

/// Convergence threshold of the friction solve: largest impulse change of a
/// sweep, `1e-12` N s (a force change of `6e-11` N at 60 Hz).
fn friction_tolerance() -> Fix128 {
    Fix128::from_ratio(1, 1_000_000_000_000)
}

/// A spin group re-solved inside the friction solve.
#[derive(Clone, Debug)]
struct SpinGroup {
    members: Vec<usize>,
    /// Mean spin at the start of the frame.
    omega0: Fix128,
    /// Drive torque on the group (Nm).
    torque: Fix128,
    /// Summed wheel inertia.
    inertia: Fix128,
    /// Brake torque available (pedal + handbrake, Nm).
    brake: Fix128,
    /// ABS regulates this group this frame.
    abs: bool,
}

impl SpinGroup {
    /// Spin for the longitudinal impulses `lam_x` of its members:
    /// `ω_f = ω₀ + dt T / I − Σ r λ_x / I`, then the Coulomb brake
    /// (`|ω|` reduced by `dt T_b / I`, not across 0). With ABS the brake
    /// stops at `ω_t = max v_x (1 − target) / r` over the members (current
    /// contact velocity). Returns `(ω, applied brake torque, spin free, ABS bound)`;
    /// the spin is not free when the brake holds it at 0 or ABS holds it at
    /// `ω_t`. The last flag says whether ABS bound.
    fn spin(
        &self,
        lam_x: &[Fix128],
        veh: &DynamicVehicle,
        chassis: &RigidBody,
        sc: &[WheelScratch],
        dt: Fix128,
        target_slip: Option<Fix128>,
    ) -> (Fix128, Fix128, bool, bool) {
        let mut react = Fix128::ZERO;
        for &j in &self.members {
            react = react + veh.config.base.wheels[j].radius * lam_x[j];
        }
        let free = self.omega0 + (dt * self.torque - react) / self.inertia;
        let full = coulomb(free, dt * self.brake / self.inertia);
        if self.abs {
            if let Some(target) = target_slip {
                let mut w_t: Option<Fix128> = None;
                for &j in &self.members {
                    let s = sc[j];
                    let r = veh.config.base.wheels[j].radius;
                    if s.grip.is_none() || r <= Fix128::ZERO {
                        continue;
                    }
                    let v_x = point_velocity(chassis, s.arm).dot(s.x_dir);
                    if v_x > Fix128::ZERO {
                        let w = v_x * (Fix128::ONE - target) / r;
                        w_t = Some(match w_t {
                            Some(t) if t >= w => t,
                            _ => w,
                        });
                    }
                }
                if let Some(w_t) = w_t {
                    if full < w_t {
                        let w = if free < w_t { free } else { w_t };
                        let applied =
                            clamp((free - w) * self.inertia / dt, Fix128::ZERO, self.brake);
                        return (w, applied, false, true);
                    }
                }
            }
        }
        let held = full.is_zero() && !self.brake.is_zero();
        let applied = if self.brake.is_zero() {
            Fix128::ZERO
        } else {
            clamp(
                (free - full).abs() * self.inertia / dt,
                Fix128::ZERO,
                self.brake,
            )
        };
        (full, applied, !held, false)
    }
}

/// Radial projection of `(a, b)` onto the ellipse with semi-axes `(ca, cb)`
/// when it lies outside; a zero semi-axis zeroes that component.
fn project_ellipse(a: Fix128, b: Fix128, ca: Fix128, cb: Fix128) -> (Fix128, Fix128) {
    let a = if ca <= Fix128::ZERO { Fix128::ZERO } else { a };
    let b = if cb <= Fix128::ZERO { Fix128::ZERO } else { b };
    let qa = if a.is_zero() { Fix128::ZERO } else { a / ca };
    let qb = if b.is_zero() { Fix128::ZERO } else { b / cb };
    let q = (qa * qa + qb * qb).sqrt();
    if q <= Fix128::ONE {
        (a, b)
    } else {
        (a / q, b / q)
    }
}

/// `(a / ca)² + (b / cb)² ≤ 1`; a zero semi-axis admits only a zero component.
fn within_ellipse(a: Fix128, b: Fix128, ca: Fix128, cb: Fix128) -> bool {
    let part = |v: Fix128, c: Fix128| -> Option<Fix128> {
        if v.is_zero() {
            Some(Fix128::ZERO)
        } else if c <= Fix128::ZERO {
            None
        } else {
            let q = v / c;
            Some(q * q)
        }
    };
    match (part(a, ca), part(b, cb)) {
        (Some(x), Some(y)) => x + y <= Fix128::ONE,
        _ => false,
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
    // LIMITATION(COV-MBD-015): The suspension is coupled explicitly, one impulse per frame.
    ///    The suspension is coupled explicitly, one impulse per frame. For a
    ///    wheel carrying the mass share `m_share` (≈ `m / wheel count` in
    ///    heave, `I / (N d²)` for a pitch or roll mode of inertia `I` with
    ///    `N` wheels at lever `d`) with local stiffness `k_eff = ∂F/∂x` (spring, progressive
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
    ///    The strut force `S` (suspension + anti-roll) acts along `up`; a
    ///    ray contact can only push along the road normal `n`, so the contact
    ///    carries `F_z = max(0, S (up·n))` and the in-plane remainder is left
    ///    to tyre friction.
    /// 4. **Wheel frame**: heading `forward cos δ + right sin δ` projected onto
    ///    the contact plane = `x`, `y = n × x`; slip per [`tire`] conventions
    ///    with `v_floor = slip_velocity_floor`.
    /// 5. **Spin** `I ω̇ = T_drive − T_brake − F_x r`, see below.
    /// 6. **Impulses**: `F_z n dt` at every contact point
    ///    (`apply_impulse_at`), then the friction impulses of all loaded
    ///    wheels from one solve ("Friction solve"), then rolling resistance
    ///    `C_rr F_z` against `v_x` on rolling free wheels, clamped by the
    ///    effective mass `m_x = 1/(m⁻¹ + (r×x)·I⁻¹(r×x))` so that it never
    // LIMITATION(COV-MBD-014): it does not act on the spin
    ///    reverses `v_x` (it does not act on the spin). The stored
    ///    `longitudinal_force` / `lateral_force` are the applied friction
    ///    impulses divided by `dt`.
    // LIMITATION(COV-MBD-044): **Aerodynamics** at the centre of mass
    /// 7. **Aerodynamics** at the centre of mass: `v_rel = v − w` (`w` is the
    ///    wind of `env.wind` when its shape contains the chassis position),
    ///    drag `−½ ρ C_dA |v_rel| v_rel`, lift `½ ρ C_lA |v_rel|²` along `up`.
    ///
    /// # Wheel spin prediction
    ///
    /// Before the friction solve each spin group is advanced on a frozen
    /// contact velocity, semi-implicit in the tyre force: with
    /// `F_x(ω') ≈ F_x(ω) + C r/v̄ (ω' − ω)` (`C = longitudinal_slope(F_z,
    /// μ_x,static)`, `v̄ = max(|v_x|, v_floor)`) the free spin is
    /// `ω_f = ω + dt (T_drive − Σ F_x(ω) r) / I_eff`, `I_eff = Σ I + dt Σ C r² / v̄`.
    /// `C` is the small-slip slope while the tyre is in its linear range
    /// (`|C κ| ≤ μ_x,static F_z`); a saturated tyre uses the secant slope
    /// `F_x(κ)/κ` instead (0 when not positive). An explicit update of a
    /// saturated tyre is not used: its spin change `dt F_x r / I` (≈ 10 rad/s
    /// per frame at 60 Hz for a passenger wheel) flips the slip sign every
    /// frame and the wheel never returns to the linear range. Brakes are
    /// Coulomb: `|ω_f|` is reduced by `dt T_brake / I_eff` without crossing
    /// zero. This prediction decides which wheels are locked (`ω' = 0`,
    /// static-friction candidates) and is the final spin of a massless wheel
    /// (`wheel_inertia ≤ 0`, one Newton step towards its quasi-static balance)
    /// and of a wheel without a friction solve (airborne, unloaded, held).
    /// Every other spin group is re-solved inside the friction solve.
    ///
    /// Driven wheels receive `powertrain.axle_torque(throttle, mean driven ω)`:
    /// split equally with [`powertrain::Differential::Open`], or with
    /// [`powertrain::Differential::Locked`] the driven wheels form one spin
    /// group (inertias, torques and brake torques summed, one shared `ω`).
    /// Rear wheels (`z ≤ 0`) take the handbrake; pedal and handbrake torques
    /// act only on wheels with `has_brake`.
    ///
    /// # Friction solve
    ///
    /// Projected Gauss-Seidel over the loaded wheels (`FRICTION_SWEEPS`
    /// sweeps, wheel index order), both tangential axes and the wheel spin
    /// together, every quantity from the current chassis velocity:
    ///
    /// - **spin** of a re-solved group from its members' accumulated
    ///   longitudinal impulses `λ_x`: `ω_f = ω₀ + (dt T_drive − Σ r λ_x) / Σ I`,
    ///   then the Coulomb brake `|ω_f|` reduced by `dt T_brake / Σ I` without
    ///   crossing 0; with ABS (below) the brake stops at the slip limit
    /// - **kinetic wheel** (not held): `κ = (ω r − v_x)/v̄`, `tan α = v_y/v̄`
    ///   from the current contact velocity, tyre force `F` of the model, one
    ///   Newton step per axis on `λ = F dt`:
    ///   `Δλ_x = (F_x dt − λ_x) / (1 + c_x dt (1/m_x + r²/I) / v̄)` (the
    ///   `r²/I` term only while the spin is free, i.e. not held at 0 by the
    ///   brake nor pinned by ABS; `c_x` = small-slip or secant slope as
    ///   above), `Δλ_y = (F_y dt − λ_y) / (1 + c_y dt / (v̄ m_y))`
    ///   (`c_y` = brush `C_α`, Magic Formula `B_y C_y μ_y,s F_z`); the
    ///   accumulated `λ` is projected radially onto the static ellipse
    ///   `(μ_x,s F_z dt, μ_y,s F_z dt)`
    /// - **held wheel** (static candidate): `λ` is driven to the value that
    ///   brings the contact to the anchor, `λ_d += m_d (e·d / dt − v_c·d)`,
    ///   projected onto the same static ellipse and `|λ_x| ≤ T_brake dt / r`.
    ///   `e` is the mean anchor offset of all held wheels (one rigid
    ///   translation): per-wheel offsets differ by the pitch / roll the
    ///   suspension allows, and holding every contact to its own anchor
    ///   over-constrains the tangential motion so that the sweeps fight each
    ///   other (measured: 1.7 mm drift in one frame on a slope)
    ///
    /// # Static friction
    ///
    /// A locked loaded wheel (`ω' = 0` from the prediction, brake applied) is
    /// a static-friction candidate when it was held in the previous frame, or
    /// when its contact slides slower than the static budget can stop in this
    /// frame: `(m_x v_x / (μ_x,s F_z dt))² + (m_y v_y / (μ_y,s F_z dt))² ≤ 1`.
    /// A faster locked wheel is sliding and stays on the tyre model (so a
    /// locked car decelerates at `μ_k g`, not `μ_s g`). The candidate set is
    /// found by a fixed point within the frame: solve; every candidate whose
    /// hold impulse was clipped by its limit breaks away, moves to the kinetic
    /// set, the chassis is restored and the solve repeats; it stops when no
    /// candidate breaks away (at most one pass per wheel plus one; a tie
    /// therefore goes to kinetic). Surviving candidates keep / create their
    /// anchor (`anchor = Some`), the others get `None`. Holding the position
    /// rather than the velocity is what keeps a parked car in place: the
    /// gravity the world integrates after the impulse moves the contact by
    /// `O(g dt²)` within a frame and the next hold returns it, so the offset
    /// stays bounded (a velocity-only clamp creeps by `≈ g sin θ dt² / 2` per
    /// frame). A slope with `tan θ > μ_s` breaks away and then slides at
    /// `g (sin θ − μ_k cos θ)`.
    ///
    /// # ABS
    ///
    /// With `abs` set and `forward_speed > min_speed`, the brake of a
    /// re-solved spin group stops lowering the spin at
    /// `ω_t = max_i v_x,i (1 − target_slip) / r_i` (current contact velocity,
    /// re-evaluated every sweep); below it the brake is released. The stored
    /// `brake_torque` is the torque that realises the result,
    /// `clamp((ω_f − ω) Σ I / dt, 0, T_pedal)` scaled per wheel, and
    /// `abs_active` is set when the limit bound.
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

        let throttle = clamp(self.input.throttle, Fix128::ZERO, Fix128::ONE);
        let pedal = clamp(self.input.brake, Fix128::ZERO, Fix128::ONE);
        let handbrake = clamp(self.input.handbrake, Fix128::ZERO, Fix128::ONE);
        let steering = clamp(self.input.steering, Fix128::NEG_ONE, Fix128::ONE);
        let steer = self.steer_angles(steering);

        // --- 1-3: contact, suspension, anti-roll -------------------------
        let mut susp = Vec::with_capacity(n);
        for (i, &steer_i) in steer.iter().enumerate() {
            let wc = self.config.base.wheels[i];
            let attach = chassis.position + chassis.rotation.rotate_vec(wc.local_position);
            let st = &mut self.wheels[i];
            st.steer_angle = steer_i;
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
        for (st, &f) in self.wheels.iter_mut().zip(&susp) {
            // the strut force acts along `up`; the contact transmits its
            // component along the road normal
            let f = f * up.dot(st.contact_normal);
            st.normal_load = if st.grounded && f > Fix128::ZERO {
                f
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
                        let f = tyre_force(
                            &self.config.tire,
                            kappa,
                            s.v_y / s.denom,
                            s.v_x,
                            st.normal_load,
                            grip,
                        );
                        s.fx_start = f.longitudinal;
                        let mu = grip.longitudinal_static;
                        let c = self.config.tire.longitudinal_slope(st.normal_load, mu);
                        if massless || (c * kappa).abs() <= mu * st.normal_load {
                            s.slope = c;
                        } else if !kappa.is_zero() {
                            // saturated: secant slope F_x(κ)/κ (≥ 0, ≤ C)
                            let secant = f.longitudinal / kappa;
                            if secant > Fix128::ZERO {
                                s.slope = secant;
                            }
                        }
                    }
                }
            }
            sc.push(s);
        }

        // --- 5: drive torque and spin groups -----------------------------
        let driven: Vec<usize> = (0..n)
            .filter(|&i| self.config.base.wheels[i].driven)
            .collect();
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
        let omega0: Vec<Fix128> = self.wheels.iter().map(|w| w.omega).collect();
        for g in &groups {
            self.solve_spin_group(g, &sc, dt);
        }

        // --- 6: impulses at the contact points --------------------------------
        for i in 0..n {
            let st = self.wheels[i];
            if st.grounded && st.normal_load > Fix128::ZERO {
                chassis
                    .apply_impulse_at(st.contact_normal * (st.normal_load * dt), st.contact_point);
            }
        }

        // friction participants: wheels with grip and load. A locked wheel
        // (`ω' = 0` from the spin update, brake applied) is a static-friction
        // candidate when it was held last frame, or when its contact slides
        // slower than the static budget can stop within this frame:
        // `(m_x v_x / (μ_x,s F_z dt))² + (m_y v_y / (μ_y,s F_z dt))² ≤ 1`.
        // A faster locked wheel is sliding and stays on the tyre model.
        let mut free: Vec<usize> = Vec::with_capacity(n);
        let mut hold: Vec<usize> = Vec::with_capacity(n);
        for (i, &s) in sc.iter().enumerate() {
            let load = self.wheels[i].normal_load;
            let Some(grip) = s.grip.filter(|_| load > Fix128::ZERO) else {
                self.wheels[i].anchor = None;
                continue;
            };
            let locked = self.wheels[i].omega.is_zero() && !self.wheels[i].brake_torque.is_zero();
            let candidate = locked
                && (self.wheels[i].anchor.is_some() || {
                    let vc = point_velocity(chassis, s.arm);
                    within_ellipse(
                        effective_mass(chassis, s.arm, s.x_dir) * vc.dot(s.x_dir),
                        effective_mass(chassis, s.arm, s.y_dir) * vc.dot(s.y_dir),
                        grip.longitudinal_static * load * dt,
                        grip.transverse_static * load * dt,
                    )
                });
            if candidate {
                hold.push(i);
            } else {
                self.wheels[i].anchor = None;
                free.push(i);
            }
        }

        // fixed point on the static / kinetic split: solve, move every
        // candidate whose hold impulse hit the static limit to the kinetic
        // set, restore the chassis and solve again, until no candidate
        // breaks away (at most `n + 1` passes; the last pass has no
        // candidates left, i.e. ties go to kinetic)
        let before = *chassis;
        let spin_groups = loop {
            let (groups_out, broken) =
                self.friction_pgs(&groups, (&free, &hold), &sc, &omega0, chassis, dt);
            if broken.is_empty() {
                break groups_out;
            }
            *chassis = before;
            hold.retain(|i| !broken.contains(i));
            for i in broken {
                self.wheels[i].anchor = None;
                free.push(i);
            }
            free.sort_unstable();
        };
        for &i in &hold {
            let point = self.wheels[i].contact_point;
            self.wheels[i].anchor = Some(self.wheels[i].anchor.unwrap_or(point));
            self.wheels[i].slip_ratio = Fix128::ZERO;
            self.wheels[i].slip_tan_alpha = Fix128::ZERO;
        }
        for &(ref g, omega) in &spin_groups {
            for &j in g {
                self.wheels[j].omega = omega;
            }
        }
        for st in &mut self.wheels {
            st.spin_angle = st.spin_angle + st.omega * dt;
        }

        // rolling resistance on rolling free wheels
        let c_rr = env.condition.rolling_resistance;
        for &i in &free {
            let s = sc[i];
            let load = self.wheels[i].normal_load;
            if self.wheels[i].omega.is_zero() || c_rr.is_zero() || load <= Fix128::ZERO {
                continue;
            }
            let vx = point_velocity(chassis, s.arm).dot(s.x_dir);
            let mag = c_rr * load * dt;
            let want = if vx > Fix128::ZERO {
                -mag
            } else if vx < Fix128::ZERO {
                mag
            } else {
                Fix128::ZERO
            };
            let mx = effective_mass(chassis, s.arm, s.x_dir);
            let j = clamp_friction(want, vx, mx);
            if !j.is_zero() {
                chassis.apply_impulse_at(s.x_dir * j, self.wheels[i].contact_point);
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

    /// Friction of the loaded wheels, `split = (kinetic, held)`: projected
    /// Gauss-Seidel over both axes, coupled to the wheel spin (see
    /// [`Self::update`], "Friction solve"). Returns the final spin of every
    /// spin group that it re-solved and the held wheels that broke away.
    fn friction_pgs(
        &mut self,
        groups: &[Vec<usize>],
        split: (&[usize], &[usize]),
        sc: &[WheelScratch],
        omega0: &[Fix128],
        chassis: &mut RigidBody,
        dt: Fix128,
    ) -> (Vec<(Vec<usize>, Fix128)>, Vec<usize>) {
        let (free, hold) = split;
        let n = self.wheels.len();
        let floor = self.config.slip_velocity_floor;
        let inertia = self.config.wheel_inertia;
        let massless = inertia <= Fix128::ZERO;
        let mut is_free = Vec::with_capacity(n);
        is_free.resize(n, false);
        for &i in free {
            is_free[i] = true;
        }
        // spin groups re-solved here: every loaded member free, wheel has mass
        let mut group_of = Vec::with_capacity(n);
        group_of.resize(n, usize::MAX);
        let mut solved: Vec<SpinGroup> = Vec::new();
        // ABS is decided inside the solve (it follows the contact velocity
        // of every sweep): eligible while the car moves forward faster than
        // `min_speed`; it binds only when the braked spin would fall below
        // the slip limit
        let fwd = DynamicVehicle::forward_speed(chassis);
        let target_slip = self
            .config
            .brakes
            .abs
            .filter(|a| fwd > a.min_speed)
            .map(|a| a.target_slip);
        let abs_eligible = target_slip.is_some();
        if !massless {
            for g in groups {
                let any_free = g.iter().any(|&j| is_free[j]);
                let all_ok = g.iter().all(|&j| is_free[j] || sc[j].grip.is_none());
                if !(any_free && all_ok) {
                    continue;
                }
                let mut sg = SpinGroup {
                    members: g.clone(),
                    omega0: Fix128::ZERO,
                    torque: Fix128::ZERO,
                    inertia: Fix128::ZERO,
                    brake: Fix128::ZERO,
                    abs: abs_eligible,
                };
                for &j in g {
                    sg.omega0 = sg.omega0 + omega0[j];
                    sg.torque = sg.torque + sc[j].drive;
                    sg.inertia = sg.inertia + inertia;
                    sg.brake = sg.brake + sc[j].brake;
                    group_of[j] = solved.len();
                }
                sg.omega0 = sg.omega0 / Fix128::from_int(g.len() as i64);
                solved.push(sg);
            }
        }

        let mut lam_x = Vec::with_capacity(n);
        lam_x.resize(n, Fix128::ZERO);
        let mut lam_y = lam_x.clone();
        let mut clipped = Vec::with_capacity(n);
        clipped.resize(n, false);
        // position error of the held contacts as one rigid translation (the
        // mean anchor offset): per-wheel offsets differ by the pitch / roll
        // the suspension allows, and holding each contact to its own anchor
        // over-constrains the tangential motion so the solve fights itself
        let mut e_sum = Vec3Fix::ZERO;
        for &i in hold {
            let point = self.wheels[i].contact_point;
            e_sum = e_sum + (self.wheels[i].anchor.unwrap_or(point) - point);
        }
        let e_mean = if hold.is_empty() {
            Vec3Fix::ZERO
        } else {
            e_sum / Fix128::from_int(hold.len() as i64)
        };
        for _ in 0..FRICTION_SWEEPS {
            let mut largest = Fix128::ZERO;
            for &i in hold {
                let s = sc[i];
                let Some(grip) = s.grip else { continue };
                let load = self.wheels[i].normal_load;
                let r = self.config.base.wheels[i].radius;
                let point = self.wheels[i].contact_point;
                let e = e_mean;
                let vc = point_velocity(chassis, s.arm);
                let mx = effective_mass(chassis, s.arm, s.x_dir);
                let my = effective_mass(chassis, s.arm, s.y_dir);
                let want_x = lam_x[i] + (e.dot(s.x_dir) / dt - vc.dot(s.x_dir)) * mx;
                let want_y = lam_y[i] + (e.dot(s.y_dir) / dt - vc.dot(s.y_dir)) * my;
                let (mut nx, ny) = project_ellipse(
                    want_x,
                    want_y,
                    grip.longitudinal_static * load * dt,
                    grip.transverse_static * load * dt,
                );
                let brake_cap = if r > Fix128::ZERO {
                    self.wheels[i].brake_torque * dt / r
                } else {
                    Fix128::ZERO
                };
                nx = clamp(nx, -brake_cap, brake_cap);
                clipped[i] = nx != want_x || ny != want_y;
                let (ddx, ddy) = (nx - lam_x[i], ny - lam_y[i]);
                largest = largest.max(ddx.abs()).max(ddy.abs());
                lam_x[i] = nx;
                lam_y[i] = ny;
                if !(ddx.is_zero() && ddy.is_zero()) {
                    chassis.apply_impulse_at(s.x_dir * ddx + s.y_dir * ddy, point);
                }
            }
            for &i in free {
                let s = sc[i];
                let Some(grip) = s.grip else { continue };
                let load = self.wheels[i].normal_load;
                let r = self.config.base.wheels[i].radius;
                let point = self.wheels[i].contact_point;
                let vc = point_velocity(chassis, s.arm);
                let v_x = vc.dot(s.x_dir);
                let v_y = vc.dot(s.y_dir);
                let den = if v_x.abs() > floor { v_x.abs() } else { floor };
                let (omega, spin_free, group_inertia) = match group_of[i] {
                    usize::MAX => (self.wheels[i].omega, false, Fix128::ZERO),
                    gi => {
                        let (w, _, free_spin, _) =
                            solved[gi].spin(&lam_x, self, chassis, sc, dt, target_slip);
                        (w, free_spin, solved[gi].inertia)
                    }
                };
                let kappa = (omega * r - v_x) / den;
                let tan_a = v_y / den;
                let f = tyre_force(&self.config.tire, kappa, tan_a, v_x, load, grip);
                self.wheels[i].slip_ratio = kappa;
                self.wheels[i].slip_tan_alpha = tan_a;

                let mx = effective_mass(chassis, s.arm, s.x_dir);
                let my = effective_mass(chassis, s.arm, s.y_dir);
                let mu = grip.longitudinal_static;
                let c = self.config.tire.longitudinal_slope(load, mu);
                let c_x = if (c * kappa).abs() <= mu * load {
                    c
                } else if !kappa.is_zero() && f.longitudinal / kappa > Fix128::ZERO {
                    f.longitudinal / kappa
                } else {
                    Fix128::ZERO
                };
                let c_y = tyre_lateral_slope(&self.config.tire, load, grip);
                // 1/m along x: chassis plus, while the spin is free, the wheel
                let mut inv_x = if mx > Fix128::ZERO {
                    Fix128::ONE / mx
                } else {
                    Fix128::ZERO
                };
                if spin_free && group_inertia > Fix128::ZERO {
                    inv_x = inv_x + r * r / group_inertia;
                }
                let dx = if inv_x > Fix128::ZERO {
                    (f.longitudinal * dt - lam_x[i]) / (Fix128::ONE + c_x * dt * inv_x / den)
                } else {
                    Fix128::ZERO
                };
                let dy = newton_step(f.lateral * dt - lam_y[i], c_y / den, dt, my);
                // project the accumulated impulse onto the static ellipse
                let (nx, ny) = project_ellipse(
                    lam_x[i] + dx,
                    lam_y[i] + dy,
                    grip.longitudinal_static * load * dt,
                    grip.transverse_static * load * dt,
                );
                let (ddx, ddy) = (nx - lam_x[i], ny - lam_y[i]);
                largest = largest.max(ddx.abs()).max(ddy.abs());
                lam_x[i] = nx;
                lam_y[i] = ny;
                if !(ddx.is_zero() && ddy.is_zero()) {
                    chassis.apply_impulse_at(s.x_dir * ddx + s.y_dir * ddy, point);
                }
            }
            if largest <= friction_tolerance() {
                break;
            }
        }
        for &i in free.iter().chain(hold) {
            self.wheels[i].longitudinal_force = lam_x[i] / dt;
            self.wheels[i].lateral_force = lam_y[i] / dt;
        }
        let broken: Vec<usize> = hold.iter().copied().filter(|&i| clipped[i]).collect();
        let mut out = Vec::with_capacity(solved.len());
        for sg in &solved {
            let (w, applied, _, abs_bound) = sg.spin(&lam_x, self, chassis, sc, dt, target_slip);
            for &j in &sg.members {
                self.wheels[j].abs_active = abs_bound;
                self.wheels[j].brake_torque = if sg.brake.is_zero() {
                    Fix128::ZERO
                } else {
                    sc[j].brake * applied / sg.brake
                };
            }
            out.push((sg.members.clone(), w));
        }
        (out, broken)
    }

    /// Spin update of one group of wheels sharing `ω` (see [`Self::update`]).
    fn solve_spin_group(&mut self, members: &[usize], sc: &[WheelScratch], dt: Fix128) {
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
        // ABS is decided in the friction solve, which follows the contact
        // velocity; the prediction brakes with the full pedal torque
        let new_omega = coulomb(omega_free, dt * brake / i_eff);
        for &i in members {
            self.wheels[i].omega = new_omega;
            self.wheels[i].abs_active = false;
            self.wheels[i].brake_torque = sc[i].brake;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anisotropic_friction::AnisotropicFriction;
    use crate::math::QuatFix;
    use surface::{GroundHit, Weather};
    use tire::BrushTire;

    fn fx(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    fn dt60() -> Fix128 {
        fx(1, 60)
    }

    fn tol(a: Fix128, b: Fix128, eps: Fix128) -> bool {
        (a - b).abs() <= eps
    }

    fn vtol(a: Vec3Fix, b: Vec3Fix, eps: Fix128) -> bool {
        tol(a.x, b.x, eps) && tol(a.y, b.y, eps) && tol(a.z, b.z, eps)
    }

    /// Road that is never hit.
    struct NoRoad;
    impl RoadSurface for NoRoad {
        fn probe(&self, _: Vec3Fix, _: Vec3Fix, _: Fix128) -> Option<GroundHit> {
            None
        }
    }

    /// Infinite plane (test-local so these unit tests do not depend on `surface`).
    struct TestPlane {
        point: Vec3Fix,
        normal: Vec3Fix,
    }
    impl RoadSurface for TestPlane {
        fn probe(&self, origin: Vec3Fix, dir: Vec3Fix, max_dist: Fix128) -> Option<GroundHit> {
            let h = (origin - self.point).dot(self.normal);
            if h <= Fix128::ZERO {
                return Some(GroundHit {
                    distance: Fix128::ZERO,
                    point: origin - self.normal * h,
                    normal: self.normal,
                });
            }
            let den = dir.dot(self.normal);
            if den >= Fix128::ZERO {
                return None;
            }
            let t = -(h / den);
            if t > max_dist {
                return None;
            }
            Some(GroundHit {
                distance: t,
                point: origin + dir * t,
                normal: self.normal,
            })
        }
    }

    fn flat() -> TestPlane {
        TestPlane {
            point: Vec3Fix::ZERO,
            normal: Vec3Fix::UNIT_Y,
        }
    }

    fn dry(rolling: Fix128) -> RoadCondition {
        RoadCondition {
            material: AnisotropicFriction::tyre_asphalt(),
            weather: Weather::Dry,
            rolling_resistance: rolling,
        }
    }

    fn env(cond: &RoadCondition) -> Environment<'_> {
        Environment {
            condition: cond,
            wind: None,
            time: Fix128::ZERO,
        }
    }

    /// Passenger layout with explicit tyre numbers and no aero, so these tests
    /// do not rely on the presets of other files.
    fn config() -> DynamicVehicleConfig {
        let mut base = VehicleConfig::default();
        rear_wheels_unsteered(&mut base);
        let powertrain = Powertrain::from_engine_config(&base.engine, &base.gear_ratios);
        DynamicVehicleConfig {
            base,
            wheel_inertia: fx(12, 10),
            tire: TireModel::Brush(BrushTire {
                longitudinal_stiffness: Fix128::from_int(80_000),
                cornering_stiffness: Fix128::from_int(60_000),
            }),
            tyre_pressure_kpa: Fix128::from_int(220),
            brakes: BrakeSystem {
                max_torque_front: Fix128::from_int(2500),
                max_torque_rear: Fix128::from_int(1500),
                handbrake_torque: Fix128::from_int(1500),
                abs: None,
            },
            powertrain,
            aero: AeroConfig {
                air_density: fx(1225, 1000),
                drag_area: Fix128::ZERO,
                lift_area: Fix128::ZERO,
            },
            ackermann: true,
            slip_velocity_floor: fx(1, 2),
        }
    }

    /// Chassis of 1000 kg near its static ride height on `flat()`
    /// (`c = mg/4k = 0.05`, centre at `0.2 + r + rest (1 − c)`), inertia of a
    /// 1.8 × 1.4 × 4.4 m box (`RigidBody::new`'s unit-sphere default, 400 kg m²
    /// in pitch, puts the pitch damping past the explicit-coupling limit).
    fn chassis() -> RigidBody {
        let mut b = RigidBody::new(
            Vec3Fix::new(Fix128::ZERO, fx(785, 1000), Fix128::ZERO),
            Fix128::from_int(1000),
        );
        // I = m/12 (a² + b²): x (pitch) h,l / y (yaw) w,l / z (roll) w,h
        b.inv_inertia = Vec3Fix::new(
            Fix128::from_int(12) / Fix128::from_int(1000 * (196 + 1936) / 100),
            Fix128::from_int(12) / Fix128::from_int(1000 * (324 + 1936) / 100),
            Fix128::from_int(12) / Fix128::from_int(1000 * (324 + 196) / 100),
        );
        b
    }

    // ---- oracle 1: tyre force acts at the contact point --------------------

    /// One wheel sliding sideways: the change of angular velocity equals
    /// `I⁻¹ (r × F) dt` with `F` the sum of the applied suspension and tyre
    /// forces and `r` the contact arm; the linear change equals `F dt / m`.
    #[test]
    fn contact_force_gives_arm_torque() {
        let mut cfg = config();
        cfg.base.wheels.truncate(1);
        cfg.base.anti_roll_stiffness = Fix128::ZERO;
        let mut v = DynamicVehicle::new(cfg);
        let mut body = RigidBody::new(
            Vec3Fix::new(Fix128::ZERO, fx(65, 100), Fix128::ZERO),
            Fix128::from_int(1000),
        );
        body.velocity = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let cond = dry(Fix128::ZERO);
        let before = body;
        let dt = dt60();
        v.update(&mut body, &flat(), &env(&cond), dt);
        let w = v.wheels[0];
        assert!(w.grounded);
        assert!(
            w.lateral_force < -Fix128::from_int(100),
            "a sideways slide must give a lateral force against it: {:?}",
            w.lateral_force
        );
        let (s, c) = w.steer_angle.sin_cos();
        let heading = Vec3Fix::UNIT_Z * c + Vec3Fix::UNIT_X * s;
        let x_dir = (heading - w.contact_normal * heading.dot(w.contact_normal)).normalize();
        let y_dir = w.contact_normal.cross(x_dir);
        let f = Vec3Fix::UNIT_Y * w.normal_load
            + x_dir * w.longitudinal_force
            + y_dir * w.lateral_force;
        let arm = w.contact_point - before.position;
        let torque = arm.cross(f * dt);
        let expect_w = Vec3Fix::new(
            torque.x * before.inv_inertia.x,
            torque.y * before.inv_inertia.y,
            torque.z * before.inv_inertia.z,
        );
        let eps = fx(1, 1_000_000_000);
        assert!(
            vtol(body.angular_velocity, expect_w, eps),
            "Δω {:?} vs I⁻¹(r×F)dt {:?}",
            body.angular_velocity,
            expect_w
        );
        assert!(
            expect_w.y.abs() > fx(1, 100),
            "yaw must be excited: {expect_w:?}"
        );
        let expect_v = before.velocity + f * dt * before.inv_mass;
        assert!(vtol(body.velocity, expect_v, eps));
    }

    // ---- oracle 2: Coulomb brake --------------------------------------------

    #[test]
    fn brake_reduces_spin_without_reversal() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(36);
        let mut v = DynamicVehicle::new(cfg);
        v.wheels[0].omega = Fix128::from_int(10);
        v.wheels[1].omega = Fix128::from_int(-10);
        v.input.brake = Fix128::ONE;
        let mut body = chassis();
        let cond = dry(Fix128::ZERO);
        let dt = dt60();
        v.update(&mut body, &NoRoad, &env(&cond), dt);
        let i = fx(12, 10);
        let step = dt * Fix128::from_int(36) / i;
        assert_eq!(v.wheels[0].omega, Fix128::from_int(10) - step);
        assert_eq!(v.wheels[1].omega, Fix128::from_int(-10) + step);
        assert_eq!(v.wheels[0].brake_torque, Fix128::from_int(36));
    }

    #[test]
    fn large_brake_locks_in_one_frame_and_holds() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        let mut v = DynamicVehicle::new(cfg);
        v.wheels[0].omega = Fix128::from_int(60);
        v.wheels[1].omega = Fix128::from_int(-60);
        v.input.brake = Fix128::ONE;
        let mut body = chassis();
        let cond = dry(Fix128::ZERO);
        for _ in 0..3 {
            v.update(&mut body, &NoRoad, &env(&cond), dt60());
            assert_eq!(v.wheels[0].omega, Fix128::ZERO);
            assert_eq!(v.wheels[1].omega, Fix128::ZERO);
        }
    }

    /// On the road at 20 m/s a front wheel braked with a huge torque is locked
    /// in the same frame (the tyre force is evaluated at the locked spin, i.e.
    /// full sliding, κ = −1).
    #[test]
    fn grounded_wheel_locks_and_slides() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        cfg.brakes.max_torque_rear = Fix128::ZERO;
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(20));
        for w in &mut v.wheels {
            w.omega = Fix128::from_int(20) / fx(3, 10);
        }
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        assert_eq!(v.wheels[0].omega, Fix128::ZERO);
        assert!(tol(
            v.wheels[0].slip_ratio,
            Fix128::NEG_ONE,
            fx(1, 1_000_000)
        ));
        assert!(v.wheels[0].longitudinal_force < Fix128::ZERO);
        assert!(!v.wheels[0].abs_active);
    }

    /// Locked wheel sliding backwards: the tyre sees full sliding (κ' = −1 in
    /// the mirrored frame), so the force is `+μ_k F_z` (against the motion).
    #[test]
    fn locked_wheel_sliding_backwards_is_full_sliding() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        cfg.brakes.max_torque_rear = Fix128::from_int(100_000);
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(-20));
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        let mu_k = AnisotropicFriction::tyre_asphalt().longitudinal_kinetic;
        for (i, w) in v.wheels.iter().enumerate() {
            assert_eq!(w.omega, Fix128::ZERO);
            assert_eq!(w.slip_ratio, Fix128::ONE, "wheel {i}: contract κ");
            let want = mu_k * w.normal_load;
            assert!(
                tol(w.longitudinal_force, want, fx(1, 1_000_000)),
                "wheel {i}: {:?} vs μ_k F_z {want:?}",
                w.longitudinal_force
            );
        }
    }

    // ---- anti-roll -----------------------------------------------------------

    /// Rolled chassis (roll φ), one axle pair: the more compressed wheel
    /// gains `k_ar (c_l − c_r)` of strut force, the other loses it (spring
    /// `k c`, no velocity so no damping); the load is the road-normal
    /// component, `F_z = strut · cos φ`.
    #[test]
    fn anti_roll_loads_the_compressed_side() {
        let mut cfg = config();
        cfg.base.wheels.truncate(2);
        let k_ar = cfg.base.anti_roll_stiffness;
        assert!(k_ar > Fix128::ZERO);
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        body.position = Vec3Fix::new(Fix128::ZERO, fx(70, 100), Fix128::ZERO);
        body.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(2, 100));
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        let (l, r) = (v.wheels[0], v.wheels[1]);
        assert!(l.compression > r.compression, "left side is lowered");
        assert!(r.compression > Fix128::ZERO && l.compression < Fix128::ONE);
        let k = v.config.base.wheels[0].spring_stiffness;
        let f = k_ar * (l.compression - r.compression);
        // the contact carries the road-normal component of the strut force
        let cos = body
            .rotation
            .rotate_vec(Vec3Fix::UNIT_Y)
            .dot(Vec3Fix::UNIT_Y);
        assert!(cos < Fix128::ONE);
        assert_eq!(l.normal_load, (k * l.compression + f) * cos);
        assert_eq!(r.normal_load, (k * r.compression - f) * cos);
    }

    // ---- ABS ------------------------------------------------------------------

    #[test]
    fn abs_holds_target_slip() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        cfg.brakes.max_torque_rear = Fix128::from_int(100_000);
        let target = fx(12, 100);
        cfg.brakes.abs = Some(AbsConfig {
            target_slip: target,
            min_speed: Fix128::ONE,
        });
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(20));
        for w in &mut v.wheels {
            w.omega = Fix128::from_int(20) / fx(3, 10);
        }
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        for (i, w) in v.wheels.iter().enumerate() {
            assert!(w.abs_active, "wheel {i}");
            assert!(w.omega > Fix128::ZERO, "wheel {i} locked under ABS");
            assert!(
                tol(w.slip_ratio, -target, fx(1, 1_000_000)),
                "wheel {i}: κ {:?}",
                w.slip_ratio
            );
            assert!(w.brake_torque < Fix128::from_int(100_000));
            assert!(w.brake_torque > Fix128::ZERO);
        }
    }

    #[test]
    fn abs_inactive_below_min_speed() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        cfg.brakes.abs = Some(AbsConfig {
            target_slip: fx(12, 100),
            min_speed: Fix128::from_int(30),
        });
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(20));
        for w in &mut v.wheels {
            w.omega = Fix128::from_int(20) / fx(3, 10);
        }
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        assert!(!v.wheels[0].abs_active);
        assert_eq!(v.wheels[0].omega, Fix128::ZERO);
    }

    // ---- oracle 3: Ackermann ----------------------------------------------------

    #[test]
    fn ackermann_matches_closed_form() {
        let v = DynamicVehicle::new(config());
        let w = &v.config.base.wheels;
        let l = w[0].local_position.z - w[2].local_position.z;
        let half_t = (w[1].local_position.x - w[0].local_position.x).half();
        let eps = fx(1, 1_000_000_000_000);
        for s in [fx(1, 2), fx(-1, 2), fx(1, 10), Fix128::ONE] {
            let d0 = s * w[0].max_steer_angle;
            let (sn, cs) = d0.sin_cos();
            let r = l / (sn / cs);
            let a = v.steer_angles(s);
            // right wheel (+x) is inner for a right turn (s > 0, R > 0)
            let right = (l / (r - half_t)).atan();
            let left = (l / (r + half_t)).atan();
            assert!(tol(a[1], right, eps), "s={s:?}: {:?} vs {right:?}", a[1]);
            assert!(tol(a[0], left, eps), "s={s:?}: {:?} vs {left:?}", a[0]);
            assert!(a[if s > Fix128::ZERO { 1 } else { 0 }].abs() > d0.abs());
            assert_eq!(a[2], Fix128::ZERO);
            assert_eq!(a[3], Fix128::ZERO);
        }
        assert!(v.steer_angles(Fix128::ZERO).iter().all(|a| a.is_zero()));
    }

    #[test]
    fn parallel_steer_without_ackermann() {
        let mut cfg = config();
        cfg.ackermann = false;
        let v = DynamicVehicle::new(cfg);
        let a = v.steer_angles(fx(1, 2));
        let d0 = fx(1, 2) * v.config.base.wheels[0].max_steer_angle;
        assert_eq!(a[0], d0);
        assert_eq!(a[1], d0);
    }

    // ---- oracle 4: effective-mass clamp ---------------------------------------

    /// A slowly rolling car with locked wheels on flat ground: friction stops
    /// it but never pushes it backwards.
    #[test]
    fn friction_never_reverses_contact_velocity() {
        let mut cfg = config();
        cfg.brakes.max_torque_rear = Fix128::from_int(100_000);
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        let mut v = DynamicVehicle::new(cfg);
        let cond = dry(Fix128::ZERO);
        for v0 in [fx(1, 100), fx(1, 10), Fix128::ONE] {
            let mut body = chassis();
            body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, v0);
            v.input.brake = Fix128::ONE;
            v.update(&mut body, &flat(), &env(&cond), dt60());
            let after = DynamicVehicle::forward_speed(&body);
            assert!(
                after >= -fx(1, 1_000_000_000),
                "v0={v0:?} reversed to {after:?}"
            );
            assert!(after <= v0);
        }
    }

    /// Rolling at 2 m/s with a small side slip (free wheels, no anchors): the
    /// stiff lateral tyre force at low speed would overshoot
    /// (`4 C_α tan α dt` exceeds `m v_y`); the effective-mass clamp keeps the
    /// side velocity from reversing.
    #[test]
    fn lateral_friction_does_not_overshoot_at_low_speed() {
        let mut v = DynamicVehicle::new(config());
        let cond = dry(Fix128::ZERO);
        let mut body = chassis();
        let v_side = fx(5, 100);
        body.velocity = Vec3Fix::new(v_side, Fix128::ZERO, Fix128::from_int(2));
        for w in &mut v.wheels {
            w.omega = Fix128::from_int(2) / fx(3, 10);
        }
        v.update(&mut body, &flat(), &env(&cond), dt60());
        assert!(v.wheels.iter().all(|w| w.anchor.is_none()));
        assert!(
            body.velocity.x >= Fix128::ZERO,
            "side velocity reversed: {:?}",
            body.velocity.x
        );
        assert!(body.velocity.x < v_side);
    }

    /// Implicit lateral friction: rolling at 0.3 m/s (slip floor regime)
    /// with side velocity `v_y = 1 mm/s` on a chassis with no rotational
    /// freedom (`m_y = m` at every contact). The solve must land on the
    /// backward-Euler fixed point `m (v' − v) = Σ F_y(tan α(v')) dt`, about
    /// `v' ≈ v / (1 + 4 C_α dt / (v̄ m)) = v / 9`; a per-wheel clamp instead
    /// lets the first wheel cancel `v_y` alone (`v' = 0`).
    #[test]
    fn lateral_friction_lands_on_the_implicit_fixed_point() {
        let mut v = DynamicVehicle::new(config());
        let cond = dry(Fix128::ZERO);
        let mut body = chassis();
        body.inv_inertia = Vec3Fix::ZERO;
        let vy0 = fx(1, 1000);
        let vz = fx(3, 10);
        body.velocity = Vec3Fix::new(vy0, Fix128::ZERO, vz);
        for w in &mut v.wheels {
            w.omega = vz / fx(3, 10);
        }
        let dt = dt60();
        v.update(&mut body, &flat(), &env(&cond), dt);
        let vy1 = body.velocity.x;
        let m = Fix128::from_int(1000);
        let applied: Fix128 = v
            .wheels
            .iter()
            .fold(Fix128::ZERO, |a, w| a + w.lateral_force);
        assert!(tol(m * (vy1 - vy0) / dt, applied, fx(1, 1_000_000)));
        let den = v.config.slip_velocity_floor;
        let mut want = Fix128::ZERO;
        for w in &v.wheels {
            let f = tyre_force(
                &v.config.tire,
                w.slip_ratio,
                vy1 / den,
                vz,
                w.normal_load,
                AnisotropicFriction::tyre_asphalt(),
            );
            want = want + f.lateral;
        }
        assert!(
            (applied - want).abs() <= want.abs() * fx(1, 100),
            "applied {:?} vs F_y(v') {:?}",
            applied.to_f64(),
            want.to_f64()
        );
        assert!(vy1 > vy0 / Fix128::from_int(12) && vy1 < vy0 / Fix128::from_int(7));
    }

    /// Pitched chassis on a frictionless road: the contacts push only along
    /// the road normal, so the impulse is `Σ F_z n dt` (no horizontal part
    /// although the strut axis `up` is tilted), and `F_z = strut · (up·n)`.
    #[test]
    fn suspension_pushes_along_the_road_normal() {
        let mut v = DynamicVehicle::new(config());
        let zero = AnisotropicFriction {
            longitudinal_static: Fix128::ZERO,
            longitudinal_kinetic: Fix128::ZERO,
            transverse_static: Fix128::ZERO,
            transverse_kinetic: Fix128::ZERO,
            slip_threshold_m_s: fx(5, 100),
        };
        let cond = RoadCondition {
            material: zero,
            weather: Weather::Dry,
            rolling_resistance: Fix128::ZERO,
        };
        let mut body = chassis();
        body.inv_inertia = Vec3Fix::ZERO;
        body.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, fx(5, 100));
        let dt = dt60();
        v.update(&mut body, &flat(), &env(&cond), dt);
        let total: Fix128 = v.wheels.iter().fold(Fix128::ZERO, |a, w| a + w.normal_load);
        assert!(total > Fix128::from_int(1000));
        assert_eq!(body.velocity.x, Fix128::ZERO);
        assert_eq!(body.velocity.z, Fix128::ZERO);
        assert!(tol(
            body.velocity.y,
            total * dt / Fix128::from_int(1000),
            fx(1, 1_000_000_000)
        ));
    }

    /// Heavy wheels spinning at `ω r = 1 cm/s` under a car at rest: the
    /// low-speed tyre force (`C_κ κ` with `v̄ = v_floor`) is far larger than
    /// what it takes to bring the contact to `ω r`; the effective-mass clamp
    /// stops the chassis there instead of overshooting.
    #[test]
    fn longitudinal_friction_does_not_overshoot_the_wheel_speed() {
        let mut cfg = config();
        cfg.wheel_inertia = Fix128::from_int(100_000);
        let mut v = DynamicVehicle::new(cfg);
        let cond = dry(Fix128::ZERO);
        let mut body = chassis();
        body.inv_inertia = Vec3Fix::ZERO;
        let rw = fx(1, 100);
        for w in &mut v.wheels {
            w.omega = rw / fx(3, 10);
        }
        v.update(&mut body, &flat(), &env(&cond), dt60());
        let surface = v.wheels[0].omega * fx(3, 10);
        assert!(body.velocity.z > Fix128::ZERO);
        assert!(
            body.velocity.z <= surface + fx(1, 1_000_000_000),
            "v {:?} beyond ω r {:?}",
            body.velocity.z.to_f64(),
            surface.to_f64()
        );
    }

    /// Locked wheels sliding diagonally (45° to the heading, fast enough to
    /// stay kinetic): combined slip on an isotropic road gives
    /// `|F| = μ_k F_z` per wheel, not `μ_k F_z` per axis.
    #[test]
    fn combined_slip_stays_on_the_friction_circle() {
        let mut cfg = config();
        cfg.brakes.max_torque_front = Fix128::from_int(100_000);
        cfg.brakes.max_torque_rear = Fix128::from_int(100_000);
        let mut v = DynamicVehicle::new(cfg);
        let iso = AnisotropicFriction {
            longitudinal_static: Fix128::ONE,
            longitudinal_kinetic: fx(8, 10),
            transverse_static: Fix128::ONE,
            transverse_kinetic: fx(8, 10),
            slip_threshold_m_s: fx(5, 100),
        };
        let cond = RoadCondition {
            material: iso,
            weather: Weather::Dry,
            rolling_resistance: Fix128::ZERO,
        };
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_int(10));
        v.input.brake = Fix128::ONE;
        v.update(&mut body, &flat(), &env(&cond), dt60());
        for (i, w) in v.wheels.iter().enumerate() {
            assert!(w.anchor.is_none(), "wheel {i} is sliding");
            let f = (w.longitudinal_force * w.longitudinal_force
                + w.lateral_force * w.lateral_force)
                .sqrt();
            let want = fx(8, 10) * w.normal_load;
            assert!(
                (f - want).abs() <= want * fx(1, 1000),
                "wheel {i}: |F| {:?} vs μ_k F_z {:?}",
                f.to_f64(),
                want.to_f64()
            );
        }
    }

    /// Rolling resistance far larger than what the slow roll can absorb
    /// (`C_rr = 5`, 1 mm/s, one wheel): it slows the car but never pushes it
    /// backwards.
    #[test]
    fn rolling_resistance_never_reverses_the_roll() {
        // one wheel: with several wheels a reversal by one is undone by the
        // next (each opposes the current velocity) and would hide here
        let mut cfg = config();
        cfg.base.wheels.truncate(1);
        let mut v = DynamicVehicle::new(cfg);
        let cond = dry(Fix128::from_int(5));
        let mut body = chassis();
        let v0 = fx(1, 1000);
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, v0);
        for w in &mut v.wheels {
            w.omega = v0 / fx(3, 10);
        }
        v.update(&mut body, &flat(), &env(&cond), dt60());
        let w = v.wheels[0];
        assert!(!w.omega.is_zero());
        // the clamp guards the contact point (an off-centre single wheel
        // also turns the chassis)
        let vx = point_velocity(&body, w.contact_point - body.position).dot(Vec3Fix::UNIT_Z);
        assert!(vx >= -fx(1, 1_000_000_000), "{:?}", vx.to_f64());
        assert!(vx < v0);
    }

    /// A car at rest with the brake on gets anchored wheels and no
    /// horizontal motion from friction.
    #[test]
    fn car_at_rest_is_held_on_flat() {
        let mut v = DynamicVehicle::new(config());
        let cond = dry(Fix128::ZERO);
        let mut body = chassis();
        v.input.brake = Fix128::ONE;
        v.update(&mut body, &flat(), &env(&cond), dt60());
        for w in &v.wheels {
            assert_eq!(w.omega, Fix128::ZERO);
            assert_eq!(w.anchor, Some(w.contact_point));
        }
        assert!(body.velocity.x.abs() <= fx(1, 1_000_000_000_000));
        assert!(body.velocity.z.abs() <= fx(1, 1_000_000_000_000));
    }

    // ---- oracle 5: degenerate inputs -----------------------------------------

    #[test]
    fn zero_dt_changes_nothing() {
        let mut v = DynamicVehicle::new(config());
        v.wheels[0].omega = Fix128::from_int(5);
        v.input.throttle = Fix128::ONE;
        let snapshot = v.wheels.clone();
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        let before = body;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), -dt60());
        assert_eq!(body, before);
        assert_eq!(v.wheels, snapshot);
    }

    #[test]
    fn static_chassis_changes_nothing() {
        let mut v = DynamicVehicle::new(config());
        v.input.throttle = Fix128::ONE;
        let snapshot = v.wheels.clone();
        let mut body =
            RigidBody::new_static(Vec3Fix::new(Fix128::ZERO, fx(785, 1000), Fix128::ZERO));
        let before = body;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        assert_eq!(body, before);
        assert_eq!(v.wheels, snapshot);
    }

    /// No wheels: only aerodynamics, `Δv = −½ρ C_dA |v| v dt / m`, rpm at idle.
    #[test]
    fn no_wheels_applies_only_aero() {
        let mut cfg = config();
        cfg.base.wheels.clear();
        cfg.aero.drag_area = fx(66, 100);
        let mut v = DynamicVehicle::new(cfg);
        assert_eq!(v.grounded_wheels(), 0);
        let mut body = chassis();
        let vel = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(30));
        body.velocity = vel;
        let cond = dry(Fix128::ZERO);
        let dt = dt60();
        v.update(&mut body, &flat(), &env(&cond), dt);
        let k = fx(1225, 1000).half() * fx(66, 100) * Fix128::from_int(30);
        let expect = vel + vel * (-k) * dt * body.inv_mass;
        assert!(vtol(body.velocity, expect, fx(1, 1_000_000_000)));
        assert_eq!(body.angular_velocity, Vec3Fix::ZERO);
        assert_eq!(v.engine_rpm, v.config.powertrain.idle_rpm);
    }

    /// Wind inside its zone: drag acts on `v − w`; outside, on `v`.
    #[test]
    fn wind_sets_relative_air_speed() {
        use crate::buoyancy_zone::ZoneShape;
        let mut cfg = config();
        cfg.base.wheels.clear();
        cfg.aero.drag_area = Fix128::ONE;
        cfg.aero.lift_area = -Fix128::ONE;
        let zone = WindZone::light_breeze(ZoneShape::Sphere {
            centre: Vec3Fix::ZERO,
            radius: Fix128::from_int(10),
        });
        let cond = dry(Fix128::ZERO);
        let mut e = env(&cond);
        e.wind = Some(&zone);
        let mut v = DynamicVehicle::new(cfg);
        let mut body = chassis();
        let dt = dt60();
        v.update(&mut body, &NoRoad, &e, dt);
        // car at rest, wind 3 m/s along +x (t = 0): v_rel = (−3, 0, 0)
        let w = zone.instantaneous_wind_vector(Fix128::ZERO);
        let v_rel = -w;
        let sp = v_rel.length();
        let half_rho = fx(1225, 1000).half();
        let f = v_rel * (-(half_rho * sp)) + Vec3Fix::UNIT_Y * (-(half_rho * sp * sp));
        assert!(vtol(
            body.velocity,
            f * dt * body.inv_mass,
            fx(1, 1_000_000_000)
        ));
        assert!(body.velocity.x > Fix128::ZERO, "wind must push along +x");
        assert!(
            body.velocity.y < Fix128::ZERO,
            "negative lift area is downforce"
        );

        let mut far = chassis();
        far.position = Vec3Fix::new(Fix128::from_int(100), Fix128::ZERO, Fix128::ZERO);
        v.update(&mut far, &NoRoad, &e, dt);
        assert_eq!(far.velocity, Vec3Fix::ZERO);
    }

    /// `wheel_inertia = 0`: an airborne wheel keeps its spin; on the road a
    /// massless wheel takes one Newton step towards its quasi-static balance
    /// (free rolling with no torque: `ω r = v_x`): from `κ = −0.01` the spin
    /// error shrinks at least tenfold in one frame.
    #[test]
    fn massless_wheel() {
        let mut cfg = config();
        cfg.wheel_inertia = Fix128::ZERO;
        let mut v = DynamicVehicle::new(cfg);
        v.wheels[0].omega = Fix128::from_int(7);
        v.input.brake = Fix128::ONE;
        let mut body = chassis();
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &NoRoad, &env(&cond), dt60());
        assert_eq!(v.wheels[0].omega, Fix128::from_int(7));

        v.input.brake = Fix128::ZERO;
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(10));
        // start at κ = −0.01 (linear range), front wheel is undriven
        v.wheels[0].omega = Fix128::from_int(10) * fx(99, 100) / fx(3, 10);
        v.update(&mut body, &flat(), &env(&cond), dt60());
        let target = Fix128::from_int(10) / fx(3, 10);
        let err0 = (Fix128::from_int(10) * fx(99, 100) / fx(3, 10) - target).abs();
        assert!(
            (v.wheels[0].omega - target).abs() * Fix128::from_int(10) <= err0,
            "{:?} vs {target:?}",
            v.wheels[0].omega
        );
    }

    /// All wheels airborne: no contact, no impulse (aero 0), spin follows
    /// drive / brake only.
    #[test]
    fn airborne_applies_nothing() {
        let mut v = DynamicVehicle::new(config());
        v.input.throttle = Fix128::ONE;
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::from_int(3));
        let before = body;
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &NoRoad, &env(&cond), dt60());
        assert_eq!(body, before);
        assert_eq!(v.grounded_wheels(), 0);
        assert!(v
            .wheels
            .iter()
            .all(|w| w.normal_load.is_zero() && w.longitudinal_force.is_zero()));
        // rear wheels are driven: positive drive torque spins them up
        assert!(v.wheels[2].omega > Fix128::ZERO);
        assert_eq!(v.wheels[2].omega, v.wheels[3].omega);
        assert_eq!(v.wheels[0].omega, Fix128::ZERO);
    }

    #[test]
    fn locked_differential_shares_spin() {
        let mut cfg = config();
        cfg.powertrain.differential = powertrain::Differential::Locked;
        let mut v = DynamicVehicle::new(cfg);
        v.wheels[2].omega = Fix128::from_int(10);
        v.wheels[3].omega = Fix128::from_int(20);
        v.input.throttle = Fix128::ONE;
        let mut body = chassis();
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &NoRoad, &env(&cond), dt60());
        assert_eq!(v.wheels[2].omega, v.wheels[3].omega);
        let axle = v
            .config
            .powertrain
            .axle_torque(Fix128::ONE, Fix128::from_int(15));
        let expect = Fix128::from_int(15) + dt60() * axle / (fx(12, 10) + fx(12, 10));
        assert_eq!(v.wheels[2].omega, expect);
    }

    #[test]
    fn open_differential_splits_torque() {
        let mut v = DynamicVehicle::new(config());
        v.input.throttle = Fix128::ONE;
        let mut body = chassis();
        let cond = dry(Fix128::ZERO);
        v.update(&mut body, &NoRoad, &env(&cond), dt60());
        let axle = v.config.powertrain.axle_torque(Fix128::ONE, Fix128::ZERO);
        let expect = dt60() * (axle / Fix128::from_int(2)) / fx(12, 10);
        assert_eq!(v.wheels[2].omega, expect);
        assert_eq!(v.engine_rpm, v.config.powertrain.engine_rpm(expect));
    }

    // ---- static hold on a slope (frame-level gravity coupling) ---------------

    /// Run a braked car on a plane inclined by `atan(grade)` in a
    /// `PhysicsWorld` and return the displacement of the chassis along the
    /// slope over `frames` frames after a settling phase.
    fn slope_drift(grade: Fix128, frames: usize) -> Fix128 {
        slope_run(grade, frames, fx(99, 100), false).0
    }

    /// `(drift along the car's forward axis over `frames`, forward velocity at
    /// the end)` after a 120-frame settle, world damping `damping`; the car
    /// faces up the slope, or down it with `facing_down`.
    fn slope_run(
        grade: Fix128,
        frames: usize,
        damping: Fix128,
        facing_down: bool,
    ) -> (Fix128, Fix128) {
        use crate::sleeping::SleepConfig;
        use crate::solver::{PhysicsWorld, SolverConfig};
        let theta = grade.atan();
        // plane tilted about +x so that the car's +z points up (or down) the slope
        let rot =
            QuatFix::from_axis_angle(Vec3Fix::UNIT_X, if facing_down { theta } else { -theta });
        let normal = rot.rotate_vec(Vec3Fix::UNIT_Y);
        let road = TestPlane {
            point: Vec3Fix::ZERO,
            normal,
        };
        let mut world = PhysicsWorld::new(SolverConfig {
            damping,
            ..SolverConfig::default()
        });
        world.set_sleep_config(SleepConfig {
            frames_to_sleep: u32::MAX,
            ..SleepConfig::default()
        });
        let mut body = chassis();
        body.position = normal * fx(785, 1000);
        body.rotation = rot;
        body.prev_position = body.position;
        body.prev_rotation = rot;
        let idx = world.add_body(body);
        let mut v = DynamicVehicle::new(config());
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        let e = Environment {
            condition: &cond,
            wind: None,
            time: Fix128::ZERO,
        };
        let up_slope = rot.rotate_vec(Vec3Fix::UNIT_Z); // car forward
        let dt = dt60();
        for _ in 0..120 {
            v.update(&mut world.bodies[idx], &road, &e, dt);
            world.step(dt);
        }
        let start = world.bodies[idx].position;
        for _ in 0..frames {
            v.update(&mut world.bodies[idx], &road, &e, dt);
            world.step(dt);
        }
        assert_eq!(v.grounded_wheels(), 4);
        (
            (world.bodies[idx].position - start).dot(up_slope),
            world.bodies[idx].velocity.dot(up_slope),
        )
    }

    /// World-driven braking from 5 m/s on flat ground: the car stops, every
    /// wheel ends up anchored, and the next 600 frames move it by less than
    /// 1 µm.
    #[test]
    fn braked_car_stops_and_stays_on_flat() {
        use crate::sleeping::SleepConfig;
        use crate::solver::{PhysicsWorld, SolverConfig};
        let mut world = PhysicsWorld::new(SolverConfig::default());
        world.set_sleep_config(SleepConfig {
            frames_to_sleep: u32::MAX,
            ..SleepConfig::default()
        });
        let mut body = chassis();
        body.velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_int(5));
        let idx = world.add_body(body);
        let mut v = DynamicVehicle::new(config());
        for w in &mut v.wheels {
            w.omega = Fix128::from_int(5) / fx(3, 10);
        }
        v.input.brake = Fix128::ONE;
        let cond = dry(Fix128::ZERO);
        let road = flat();
        let dt = dt60();
        for _ in 0..240 {
            v.update(&mut world.bodies[idx], &road, &env(&cond), dt);
            world.step(dt);
        }
        assert!(v.wheels.iter().all(|w| w.anchor.is_some()));
        let start = world.bodies[idx].position;
        for _ in 0..600 {
            v.update(&mut world.bodies[idx], &road, &env(&cond), dt);
            world.step(dt);
        }
        let d = world.bodies[idx].position - start;
        let horiz = (d.x * d.x + d.z * d.z).sqrt();
        assert!(horiz <= fx(1, 1_000_000), "flat drift {horiz:?}");
    }

    /// `tan θ = 1.5 > μ_s = 1.1`: a locked car breaks away and accelerates at
    /// `g (sin θ − μ_k cos θ)` (μ_k = 0.9) down the slope, within 2 %.
    #[test]
    fn locked_car_slides_on_steep_slope() {
        let grade = fx(3, 2);
        let frames = 120;
        // facing down the slope so the locked wheel slides forward (κ = −1,
        // the brush contract's exact full-sliding point)
        let (_, v1) = slope_run(grade, 0, Fix128::ONE, true);
        let (_, v2) = slope_run(grade, frames, Fix128::ONE, true);
        let a = (v2 - v1) / (dt60() * Fix128::from_int(frames as i64));
        let theta = grade.atan();
        let (sn, cs) = theta.sin_cos();
        let expect = Fix128::from_int(10) * (sn - fx(9, 10) * cs);
        assert!(
            (a - expect).abs() <= expect.abs() * fx(2, 100),
            "a {:?} vs {:?}",
            a.to_f64(),
            expect.to_f64()
        );
    }

    #[test]
    fn braked_car_holds_on_slope() {
        // tan θ = 0.3 < μ (1.1 dry asphalt)
        let drift = slope_drift(fx(3, 10), 600);
        let drift_undamped = slope_run(fx(3, 10), 600, Fix128::ONE, false).0;
        assert!(drift_undamped.abs() <= fx(1, 1000), "{drift_undamped:?}");
        assert!(
            drift.abs() <= fx(1, 1000),
            "drift along the slope over 600 frames: {drift:?} m"
        );
    }
}
