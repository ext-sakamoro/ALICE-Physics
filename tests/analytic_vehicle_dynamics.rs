//! Closed-form oracles for `vehicle_dynamics::DynamicVehicle`.
//!
//! Every test drives the production entry point: a `PhysicsWorld`, the
//! chassis added with `add_body`, and per frame
//! `DynamicVehicle::update(&mut world.bodies[car], &road, &env, dt)` followed
//! by `world.step(dt)`. Expected values are textbook closed forms evaluated in
//! `f64`; no implementation function is called to build an expectation
//! (the only exception is `RoadCondition::weather_factor`, whose table is a
//! model definition documented on the implementation, see
//! `stopping_distance_*`).
//!
//! Common setup (`config`), starting from `DynamicVehicleConfig::passenger_car()`:
//! - wheel layout: track `t = 1.6`, attachment `y = -0.2`, front axle at
//!   `z = +a`, rear axle at `z = -b` (default `a = b = 1.2`, `L = 2.4`),
//!   radius `r = 0.3`, rest `0.3`, `k = 50000` (legacy wheel defaults)
//! - brush tyre `C_κ = 80 kN`, `C_α = 60 kN` per wheel (pinned explicitly)
//! - wheel spin inertia `I_w = 1 kg m²`, tyre pressure 220 kPa
//! - brakes 2500 / 1500 Nm, no ABS, no handbrake
//! - aerodynamics off (`drag_area = lift_area = 0`)
//! - powertrain pinned: flat 300 Nm curve, idle 800, limit 7000 rpm,
//!   gears 3.5 / 2.2 / 1.5 / 1.1 / 0.8, final drive 1, open differential,
//!   engine braking 0
//! - Ackermann off unless the test says otherwise
//! - world damping set to 1 (`PhysicsConfig::default()` retains 0.99 of the
//!   velocity per frame, which is a 0.6 /s exponential drag at 60 Hz and would
//!   swamp every closed form below)
//!
//! Chassis: mass `M = 1000 kg`, CG at the body origin, inertia
//! `I = M · (1.5, 1.8, 0.5)` (pitch, yaw, roll). Gravity is read from
//! `world.config.gravity`. Road grip is an isotropic material with
//! `μ_s = 1.0`, `μ_k = 0.8` unless stated.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle_dynamics::powertrain::{Differential, Powertrain, TorqueCurve};
use alice_physics::vehicle_dynamics::surface::{
    FlatGround, InclinedPlane, RoadCondition, RoadSurface, Weather,
};
use alice_physics::vehicle_dynamics::tire::{BrushTire, TireModel};
use alice_physics::vehicle_dynamics::{
    AbsConfig, AeroConfig, BrakeSystem, DynamicVehicle, DynamicVehicleConfig, Environment,
};
use alice_physics::wind_zone::WindZone;

const M: f64 = 1000.0;
const DT: f64 = 1.0 / 60.0;
const R_WHEEL: f64 = 0.3;
const REST: f64 = 0.3;
const K_SPRING: f64 = 50000.0;
const DAMPER: f64 = 4500.0;
const HALF_TRACK: f64 = 0.8;
const ATTACH_Y: f64 = -0.2;
const C_KAPPA: f64 = 80000.0;
const C_ALPHA: f64 = 60000.0;
const I_WHEEL: f64 = 1.0;
const MU_S: f64 = 1.0;
const MU_K: f64 = 0.8;
const TYRE_KPA: f64 = 220.0;
const ENGINE_TORQUE: f64 = 300.0;
const GEARS: [f64; 5] = [3.5, 2.2, 1.5, 1.1, 0.8];

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn f(x: Fix128) -> f64 {
    x.to_f64()
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Isotropic material with the given peak / sliding coefficients.
fn material(mu_s: f64, mu_k: f64) -> AnisotropicFriction {
    AnisotropicFriction {
        longitudinal_static: fx(mu_s),
        longitudinal_kinetic: fx(mu_k),
        transverse_static: fx(mu_s),
        transverse_kinetic: fx(mu_k),
        slip_threshold_m_s: fx(0.05),
    }
}

fn condition(weather: Weather, c_rr: f64) -> RoadCondition {
    RoadCondition {
        material: material(MU_S, MU_K),
        weather,
        rolling_resistance: fx(c_rr),
    }
}

fn powertrain(max_rpm: f64) -> Powertrain {
    Powertrain {
        curve: TorqueCurve {
            points: vec![
                (fx(800.0), fx(ENGINE_TORQUE)),
                (fx(max_rpm), fx(ENGINE_TORQUE)),
            ],
        },
        idle_rpm: fx(800.0),
        max_rpm: fx(max_rpm),
        engine_brake_per_rpm: Fix128::ZERO,
        gear_ratios: GEARS.iter().map(|&g| fx(g)).collect(),
        final_drive: Fix128::ONE,
        current_gear: 0,
        differential: Differential::Open,
    }
}

/// Common configuration (see module doc), front axle at `+a`, rear at `-b`.
fn config(a: f64, b: f64) -> DynamicVehicleConfig {
    let mut c = DynamicVehicleConfig::passenger_car();
    assert_eq!(c.base.wheels.len(), 4, "passenger_car has four wheels");
    let pos = [
        (-HALF_TRACK, a),
        (HALF_TRACK, a),
        (-HALF_TRACK, -b),
        (HALF_TRACK, -b),
    ];
    for (w, (x, z)) in c.base.wheels.iter_mut().zip(pos) {
        w.local_position = v3(x, ATTACH_Y, z);
        w.radius = fx(R_WHEEL);
        w.suspension_rest = fx(REST);
        w.spring_stiffness = fx(K_SPRING);
        w.damping = fx(DAMPER);
        w.progressive_rate = Fix128::ZERO;
        w.bump_damping = Fix128::ZERO;
        w.rebound_damping = Fix128::ZERO;
        w.bump_stop_stiffness = Fix128::ZERO;
        w.max_steer_angle = if z > 0.0 { fx(0.5) } else { Fix128::ZERO };
        w.driven = z < 0.0;
        w.has_brake = true;
    }
    c.wheel_inertia = fx(I_WHEEL);
    c.tire = TireModel::Brush(BrushTire {
        longitudinal_stiffness: fx(C_KAPPA),
        cornering_stiffness: fx(C_ALPHA),
    });
    c.tyre_pressure_kpa = fx(TYRE_KPA);
    c.brakes = BrakeSystem {
        max_torque_front: fx(2500.0),
        max_torque_rear: fx(1500.0),
        handbrake_torque: Fix128::ZERO,
        abs: None,
    };
    c.aero = AeroConfig {
        air_density: fx(1.225),
        drag_area: Fix128::ZERO,
        lift_area: Fix128::ZERO,
    };
    c.powertrain = powertrain(7000.0);
    c.ackermann = false;
    c
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.config.damping = Fix128::ONE;
    w
}

/// `g` as a positive number from the world's gravity vector.
fn g_of(w: &PhysicsWorld) -> f64 {
    f(w.config.gravity.length())
}

/// Static ride height of the CG over flat ground: wheels hang `0.2` below the
/// CG, spring force `k · compression` with `compression = m g / (n k)` per
/// wheel (legacy suspension contract).
fn ride_height(m: f64, g: f64, n_wheels: f64) -> f64 {
    -ATTACH_Y + R_WHEEL + REST * (1.0 - m * g / (n_wheels * K_SPRING))
}

struct Sim<R: RoadSurface> {
    world: PhysicsWorld,
    car: usize,
    veh: DynamicVehicle,
    road: R,
    cond: RoadCondition,
    wind: Option<WindZone>,
    t: Fix128,
}

impl<R: RoadSurface> Sim<R> {
    fn new(
        mut world: PhysicsWorld,
        cfg: DynamicVehicleConfig,
        road: R,
        cond: RoadCondition,
        pos: Vec3Fix,
        rot: QuatFix,
        mass: f64,
    ) -> Self {
        let mut body = RigidBody::new_dynamic(pos, fx(mass));
        body.rotation = rot;
        body.prev_rotation = rot;
        body.inv_inertia = v3(1.0 / (1.5 * mass), 1.0 / (1.8 * mass), 1.0 / (0.5 * mass));
        let car = world.add_body(body);
        Self {
            world,
            car,
            veh: DynamicVehicle::new(cfg),
            road,
            cond,
            wind: None,
            t: Fix128::ZERO,
        }
    }

    fn frame(&mut self) {
        let env = Environment {
            condition: &self.cond,
            wind: self.wind.as_ref(),
            time: self.t,
            gravity: self.world.config.gravity,
        };
        self.veh
            .update(&mut self.world.bodies[self.car], &self.road, &env, dt());
        self.world.step(dt());
        self.t = self.t + dt();
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.frame();
        }
    }

    fn body(&self) -> &RigidBody {
        &self.world.bodies[self.car]
    }

    fn g(&self) -> f64 {
        g_of(&self.world)
    }

    /// Give the chassis forward speed `v` along its heading and spin every
    /// wheel at the rolling speed `v / r` (so no slip transient).
    fn set_speed(&mut self, v: f64) {
        let fwd = self.body().rotation.rotate_vec(Vec3Fix::UNIT_Z);
        self.world.bodies[self.car].velocity = fwd * fx(v);
        for w in &mut self.veh.wheels {
            w.omega = fx(v / R_WHEEL);
        }
    }

    /// Nose-down pitch angle (rad): `asin(-forward.y)`.
    fn nose_down(&self) -> f64 {
        let fwd = self.body().rotation.rotate_vec(Vec3Fix::UNIT_Z);
        (-f(fwd.y)).asin()
    }
}

/// Car on flat ground at its closed-form ride height, at rest.
fn flat_sim(cfg: DynamicVehicleConfig, cond: RoadCondition) -> Sim<FlatGround> {
    let w = world();
    let g = g_of(&w);
    let y0 = ride_height(M, g, 4.0);
    Sim::new(
        w,
        cfg,
        FlatGround {
            height: Fix128::ZERO,
        },
        cond,
        v3(0.0, y0, 0.0),
        QuatFix::IDENTITY,
        M,
    )
}

// ---------------------------------------------------------------------------
// 1. Locked-wheel stopping distance `v0² / (2 μ_k g)`
// ---------------------------------------------------------------------------

const V0_STOP: f64 = 20.0;
const LOCK_TORQUE: f64 = 10000.0;

struct Stop {
    distance: f64,
    expected: f64,
    tol: f64,
    sim: Sim<FlatGround>,
}

/// All four wheels locked from `v0 = 20 m/s` (brake torque 10 kNm per wheel,
/// no ABS, aero 0, `C_rr = 0`), effective sliding coefficient `μ = μ_k · f_w`
/// with `f_w = weather_factor()`.
///
/// Closed form (Coulomb sliding of a rigid body, `Σ F_z = M g`):
/// `s = v0² / (2 μ g)`.
///
/// Tolerance `v0 dt + v0 dt + v_floor² / (μ g)`:
/// - semi-implicit frame integration: `Σ v_k dt` differs from `∫ v dt` by at
///   most one frame of travel `v0 dt`
/// - lock onset: the wheel needs `t_lock = I_w ω0 / (T_b − μ_s F_z r)` to stop
///   (asserted `< dt` below), during which the force may be below sliding:
///   at most `v0 dt` of extra travel
/// - below `v_floor` the slip ratio is `−v / v_floor` (tyre contract) and the
///   force can leave the full-sliding value: at most `v_floor² / (2 μ g)`
///   doubled for margin
fn locked_stop(weather: Weather) -> Stop {
    let mut cfg = config(1.2, 1.2);
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let cond = condition(weather, 0.0);
    let factor = f(cond.weather_factor());
    let mu = MU_K * factor;
    let mut sim = flat_sim(cfg, cond);
    let g = sim.g();
    // Lock-onset precondition (upper bound F_z ≤ M g, μ ≤ μ_s).
    let t_lock = I_WHEEL * (V0_STOP / R_WHEEL) / (LOCK_TORQUE - MU_S * M * g * R_WHEEL);
    assert!(t_lock < DT, "lock onset {t_lock} s must fit in one frame");
    let v_floor = f(sim.veh.config.slip_velocity_floor);

    sim.frames(120);
    sim.set_speed(V0_STOP);
    let z0 = f(sim.body().position.z);
    sim.veh.input.brake = Fix128::ONE;
    let expected = V0_STOP * V0_STOP / (2.0 * mu * g);
    let t_stop = V0_STOP / (mu * g);
    sim.frames((1.5 * t_stop / DT) as usize + 120);
    let distance = f(sim.body().position.z) - z0;
    let tol = 2.0 * V0_STOP * DT + v_floor * v_floor / (mu * g);
    Stop {
        distance,
        expected,
        tol,
        sim,
    }
}

/// Dry: `f_w = 1` exactly (contract "dry = 1"), `s = 20² / (2 · 0.8 · 10) = 25 m`.
#[test]
fn stopping_distance_dry_locked_wheels() {
    let cond = condition(Weather::Dry, 0.0);
    assert_eq!(
        cond.weather_factor(),
        Fix128::ONE,
        "dry weather factor is 1"
    );
    let s = locked_stop(Weather::Dry);
    assert!(
        (s.distance - s.expected).abs() <= s.tol,
        "dry: s = {} m, closed form {} ± {}",
        s.distance,
        s.expected,
        s.tol
    );
}

/// Stopping-distance ratio when only the weather changes:
/// `s_w / s_dry = μ_dry / μ_w = 1 / f_w`, tolerance propagated from both runs
/// `(1/f_w) (tol_w / s_w + tol_dry / s_dry)`.
fn ratio_check(w: &Stop, dry: &Stop, factor: f64, name: &str) {
    let ratio = w.distance / dry.distance;
    let want = 1.0 / factor;
    let tol = want * (w.tol / w.expected + dry.tol / dry.expected);
    assert!(
        (ratio - want).abs() <= tol,
        "{name}: s ratio {ratio}, closed form 1/f_w = {want} ± {tol}"
    );
}

/// Wet (1 mm film, `v0 = 20 m/s` below the hydroplaning onset
/// `6.35 √220 km/h = 26.16 m/s`, so no hydroplaning loss):
/// `s = v0² / (2 μ_k f_wet g)`; `0 < f_wet < 1`; ratio to dry `1 / f_wet`.
#[test]
fn stopping_distance_wet_and_ratio_to_dry() {
    let weather = Weather::Wet {
        water_depth_mm: Fix128::ONE,
    };
    let f_wet = f(condition(weather, 0.0).weather_factor());
    assert!(f_wet > 0.0 && f_wet < 1.0, "wet factor {f_wet} in (0, 1)");
    let wet = locked_stop(weather);
    assert!(
        (wet.distance - wet.expected).abs() <= wet.tol,
        "wet: s = {} m, closed form {} ± {}",
        wet.distance,
        wet.expected,
        wet.tol
    );
    let dry = locked_stop(Weather::Dry);
    ratio_check(&wet, &dry, f_wet, "wet/dry");
}

/// Ice: `s = v0² / (2 μ_k f_ice g)`; physical ordering `f_ice < f_wet`;
/// ratio to dry `1 / f_ice`.
#[test]
fn stopping_distance_ice_and_ratio_to_dry() {
    let f_ice = f(condition(Weather::Ice, 0.0).weather_factor());
    let f_wet = f(condition(
        Weather::Wet {
            water_depth_mm: Fix128::ONE,
        },
        0.0,
    )
    .weather_factor());
    assert!(
        f_ice > 0.0 && f_ice < f_wet,
        "ice factor {f_ice} in (0, f_wet = {f_wet})"
    );
    let ice = locked_stop(Weather::Ice);
    assert!(
        (ice.distance - ice.expected).abs() <= ice.tol,
        "ice: s = {} m, closed form {} ± {}",
        ice.distance,
        ice.expected,
        ice.tol
    );
    let dry = locked_stop(Weather::Dry);
    ratio_check(&ice, &dry, f_ice, "ice/dry");
}

// ---------------------------------------------------------------------------
// 2. Exact rest after stopping
// ---------------------------------------------------------------------------

/// After the dry locked stop, wait 5 s, then for 60 frames the horizontal
/// velocity is `≤ 1e-9 m/s` and the CG moves `≤ 1e-9 m`.
///
/// Bound: on flat ground with locked wheels nothing drives the car (Coulomb
/// statics: zero tangential force needed, zero motion). The only residual is
/// the suspension transient of the stop, damped with
/// `ζ ω_n = c / (2 m_w) = 4500 / (2 · 250) = 9 /s` per corner; after 5 s it
/// is `e^{-45} ≈ 3e-20` of its initial amplitude (< 1 m/s), below Fix128
/// resolution. `1e-9` leaves ten orders of magnitude for rounding.
#[test]
fn locked_car_comes_to_exact_rest() {
    let mut s = locked_stop(Weather::Dry).sim;
    s.frames(300);
    let p0 = s.body().position;
    for k in 0..60 {
        s.frame();
        let v = s.body().velocity;
        assert!(
            f(v.x).abs() <= 1e-9 && f(v.z).abs() <= 1e-9,
            "frame {k}: horizontal velocity ({}, {}) after stop",
            f(v.x),
            f(v.z)
        );
    }
    let d = s.body().position - p0;
    assert!(
        f(d.x).abs() <= 1e-9 && f(d.z).abs() <= 1e-9,
        "drift after stop ({}, {}) m",
        f(d.x),
        f(d.z)
    );
}

// ---------------------------------------------------------------------------
// 3. ABS
// ---------------------------------------------------------------------------

const ABS_MIN_SPEED: f64 = 3.0;

fn abs_config() -> DynamicVehicleConfig {
    let mut cfg = config(1.2, 1.2);
    cfg.brakes.abs = Some(AbsConfig {
        target_slip: fx(0.12),
        min_speed: fx(ABS_MIN_SPEED),
    });
    cfg
}

/// Full pedal with ABS (brakes 2500 / 1500 Nm) from 20 m/s on dry road.
///
/// Precondition: without ABS these brakes lock every wheel — front
/// `2500 > μ_s F_zf r` with `F_zf ≤ M g / 2 + M μ_s g h / (2L)`, rear
/// `1500 > μ_s (M g / 4) r` (computed below).
///
/// (a) While the forward speed exceeds `min_speed` no wheel is locked
///     (`ω > 0` at every frame end).
/// (b) Stopping distance `≥ v0² / (2 μ_s g) − v0 dt`: no tyre can exceed the
///     peak coefficient, minus one frame of discretisation.
#[test]
fn abs_keeps_wheels_rolling_and_respects_peak_grip_bound() {
    let mut sim = flat_sim(abs_config(), condition(Weather::Dry, 0.0));
    let g = sim.g();
    let h = ride_height(M, g, 4.0);
    let fzf_max = M * g / 2.0 + M * MU_S * g * h / (2.0 * 2.4);
    assert!(
        2500.0 > MU_S * fzf_max * R_WHEEL,
        "front brakes lock without ABS"
    );
    assert!(
        1500.0 > MU_S * (M * g / 4.0) * R_WHEEL,
        "rear brakes lock without ABS"
    );

    sim.frames(120);
    sim.set_speed(V0_STOP);
    let z0 = f(sim.body().position.z);
    sim.veh.input.brake = Fix128::ONE;
    let t_max = V0_STOP / (0.3 * MU_K * g); // generous upper bound on the stop
    let mut stopped = false;
    for k in 0..(t_max / DT) as usize {
        sim.frame();
        let v = f(sim.body().velocity.z);
        if v > ABS_MIN_SPEED {
            for (i, w) in sim.veh.wheels.iter().enumerate() {
                assert!(
                    w.omega > Fix128::ZERO,
                    "frame {k}, v = {v}: wheel {i} locked under ABS"
                );
            }
        }
        if v <= 0.0 {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "car stops within {t_max} s");
    sim.frames(120);
    let s = f(sim.body().position.z) - z0;
    let lower = V0_STOP * V0_STOP / (2.0 * MU_S * g) - V0_STOP * DT;
    assert!(
        s >= lower,
        "ABS stopping distance {s} below the peak-grip bound {lower}"
    );
}

/// (c) Steering while braking: brakes applied first (wheels at steady
/// braking), then steering 0.5 (front wheels at 0.25 rad) for 1 s.
///
/// - No ABS, brakes 10 kNm (all wheels locked): an isotropic sliding contact
///   (`μ_x = μ_y` here) pushes against the sliding velocity whatever the wheel
///   heading (Coulomb), every contact slides with the CG velocity and the
///   layout is left/right symmetric, so the yaw moment is zero:
///   `max |ω_y| ≤ 1e-6 rad/s` (rounding only).
/// - ABS: the rolling front tyres keep lateral grip; detection threshold
///   `max |ω_y| ≥ 0.1 rad/s`, 6 % of the kinematic `v tan δ / L ≈ 1.6 rad/s`
///   at 15 m/s.
#[test]
fn braking_with_steering_yaws_only_with_abs() {
    fn run(cfg: DynamicVehicleConfig) -> f64 {
        let mut sim = flat_sim(cfg, condition(Weather::Dry, 0.0));
        sim.frames(120);
        sim.set_speed(15.0);
        sim.veh.input.brake = Fix128::ONE;
        sim.frames(10);
        sim.veh.input.steering = fx(0.5);
        let mut max_yaw: f64 = 0.0;
        for _ in 0..60 {
            sim.frame();
            max_yaw = max_yaw.max(f(sim.body().angular_velocity.y).abs());
        }
        max_yaw
    }
    let mut locked = config(1.2, 1.2);
    locked.brakes.max_torque_front = fx(LOCK_TORQUE);
    locked.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let yaw_locked = run(locked);
    let yaw_abs = run(abs_config());
    assert!(yaw_locked <= 1e-6, "locked wheels yaw {yaw_locked} rad/s");
    assert!(
        yaw_abs >= 0.1,
        "ABS braking with steering yaw {yaw_abs} rad/s"
    );
}

// ---------------------------------------------------------------------------
// 4. Low-speed turning radius
// ---------------------------------------------------------------------------

/// Circumradius of three points in the ground plane `(x, z)`.
fn circumradius(p: [(f64, f64); 3]) -> f64 {
    let d = |a: (f64, f64), b: (f64, f64)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
    let (a, b, c) = (d(p[0], p[1]), d(p[1], p[2]), d(p[2], p[0]));
    let area2 =
        ((p[1].0 - p[0].0) * (p[2].1 - p[0].1) - (p[1].1 - p[0].1) * (p[2].0 - p[0].0)).abs();
    a * b * c / (2.0 * area2)
}

/// Coast at 2 m/s with steering `s` (no resistances), wait 3 s, then take
/// the CG positions at `t, t + 3 s, t + 6 s` and return their circumradius.
fn turning_radius(ackermann: bool, steering: f64) -> f64 {
    let mut cfg = config(1.2, 1.2);
    cfg.ackermann = ackermann;
    let mut sim = flat_sim(cfg, condition(Weather::Dry, 0.0));
    sim.frames(120);
    sim.set_speed(2.0);
    sim.veh.input.steering = fx(steering);
    sim.frames(180);
    let mut pts = [(0.0, 0.0); 3];
    for p in &mut pts {
        let b = sim.body().position;
        *p = (f(b.x), f(b.z));
        sim.frames(180);
    }
    circumradius(pts)
}

/// Parallel steering (`ackermann = false`), `δ = 0.2 · 0.5 = 0.1 rad`.
///
/// Closed form: rear-axle kinematic radius `R = L / tan δ = 23.92 m`; the CG
/// sits `b = 1.2` ahead of the rear axle, so it runs on
/// `R_cg = √(R² + b²) = 23.950 m`.
///
/// Tolerance 1 %: parallel steering misaligns the front wheels by
/// `±(δ_in − δ_out)/2`, which cancels to first order (equal `C_α`) and leaves
/// `O(δ (t / 2R)²) ≈ 0.1 %`; with `a = b` and equal stiffness the understeer
/// gradient is 0, so lateral slip does not change the radius
/// (`a_y = v² / R = 0.17 m/s²`); deceleration from the steered tyre's lateral
/// force (`< 0.01 m/s²`) changes speed, not the radius.
#[test]
fn low_speed_turning_radius_matches_wheelbase_over_tan_delta() {
    let (l, b, delta) = (2.4_f64, 1.2_f64, 0.1_f64);
    let r_cg = ((l / delta.tan()).powi(2) + b * b).sqrt();
    let r = turning_radius(false, 0.2);
    assert!(
        (r - r_cg).abs() <= 0.01 * r_cg,
        "turning radius {r} m, closed form {r_cg} m"
    );
}

/// Ackermann on, large angle `δ = 0.8 · 0.5 = 0.4 rad` (the commanded angle
/// is the equivalent bicycle angle, inner / outer
/// `tan δ_in/out = L / (R ∓ t/2)`).
///
/// Closed form: `R = L / tan δ = 5.6765 m`, `R_cg = √(R² + b²) = 5.8020 m`.
/// Tolerance 0.5 %: zero kinematic slip by construction; at `a_y = 0.7 m/s²`
/// the front lateral force is tilted by `δ`, mismatching front / rear slip by
/// `α (1/cos δ − 1) ≈ 3e-4 rad` (0.06 % of `δ`).
///
/// Parallel steering at the same `δ` balances its opposite front slips at
/// `δ = (δ_in*(R) + δ_out*(R)) / 2`, i.e. `R_cg = 5.8943 m` (+1.6 %), so the
/// Ackermann error must be the smaller one.
#[test]
fn low_speed_turning_radius_with_ackermann_beats_parallel_steering() {
    let (l, b, delta) = (2.4_f64, 1.2_f64, 0.4_f64);
    let r_cg = ((l / delta.tan()).powi(2) + b * b).sqrt();
    let r_ack = turning_radius(true, 0.8);
    let r_par = turning_radius(false, 0.8);
    assert!(
        (r_ack - r_cg).abs() <= 0.005 * r_cg,
        "Ackermann radius {r_ack} m, closed form {r_cg} m"
    );
    assert!(
        (r_ack - r_cg).abs() < (r_par - r_cg).abs(),
        "Ackermann error {} must be below parallel error {}",
        r_ack - r_cg,
        r_par - r_cg
    );
}

// ---------------------------------------------------------------------------
// 5. Linear two-wheel (bicycle) model
// ---------------------------------------------------------------------------

/// Front axle `a = 1.0`, rear `b = 1.4` from the CG (`L = 2.4`), parallel
/// steering `δ = 0.02 · 0.5 = 0.01 rad`, coasting at 15 m/s.
///
/// Closed form (Gillespie, *Fundamentals of Vehicle Dynamics*, ch. 6; axle
/// stiffness `C_f = C_r = 2 C_α = 120 kN`):
/// `r = v δ / (L + K v²)`, `K = M (b C_r − a C_f) / (L C_f C_r)`
/// `= 1000 · 0.4 / (2.4 · 120000) = 1.3889e-3 s²/m` (understeer, `b > a`).
/// Sign: steering right (`+x`) gives `ω_y > 0` (rotation about `+y` carries
/// `+z` towards `+x`).
///
/// Tolerance 1 %:
/// - brush non-linearity: both axles run at `F_y / F_z = a_y / g`, so the
///   brush secant stiffness drops by the same `a_y / (3 μ_s g) ≈ 2.8 %` at
///   both, scaling only the `K v²` part (11.5 % of the denominator): 0.32 %
/// - `tan α` vs `α` at `α ≈ 0.004`: 5e-6
/// - quasi-steady speed: deceleration `< 0.01 m/s²` over the 0.5 s window,
///   evaluated with the window-mean speed
/// - frame discretisation of the heading rotation `r dt ≈ 1e-3`
#[test]
fn linear_bicycle_model_steady_yaw_rate() {
    let (a, b, l) = (1.0_f64, 1.4_f64, 2.4_f64);
    let mut sim = flat_sim(config(a, b), condition(Weather::Dry, 0.0));
    let (cf, cr) = (2.0 * C_ALPHA, 2.0 * C_ALPHA);
    let k_us = M * (b * cr - a * cf) / (l * cf * cr);
    sim.frames(120);
    sim.set_speed(15.0);
    sim.veh.input.steering = fx(0.02);
    let delta = 0.01_f64;
    sim.frames(120);
    let (mut r_sum, mut v_sum) = (0.0, 0.0);
    for _ in 0..30 {
        sim.frame();
        r_sum += f(sim.body().angular_velocity.y);
        let v = sim.body().velocity;
        v_sum += (f(v.x).powi(2) + f(v.z).powi(2)).sqrt();
    }
    let (r, v) = (r_sum / 30.0, v_sum / 30.0);
    let want = v * delta / (l + k_us * v * v);
    assert!(
        (r - want).abs() <= 0.01 * want,
        "yaw rate {r} rad/s at {v} m/s, linear model {want} (K = {k_us})"
    );
}

// ---------------------------------------------------------------------------
// 6. Load transfer under braking + 11b. pitch from contact-point forces
// ---------------------------------------------------------------------------

struct Braking {
    a: f64,
    h: f64,
    pitch: f64,
    pitch_static: f64,
    front: f64,
    front_static: f64,
    g: f64,
}

/// Steady braking: pedal 0.2 (front 500 Nm, rear 300 Nm per wheel, not
/// locked), from 30 m/s, wheel inertia lowered to 0.05 kg m² so the wheels'
/// spin-down angular momentum `Σ I_w a / r` (≤ 1 Nm) does not enter the
/// pitch balance. Measured over 30 frames after 1 s: deceleration
/// `a = Δv / Δt`, CG height `h` (ground at 0), front-axle `Σ normal_load`,
/// nose-down pitch.
fn steady_braking() -> Braking {
    let mut cfg = config(1.2, 1.2);
    cfg.wheel_inertia = fx(0.05);
    let mut sim = flat_sim(cfg, condition(Weather::Dry, 0.0));
    let g = sim.g();
    sim.frames(180);
    let front_static = f(sim.veh.wheels[0].normal_load + sim.veh.wheels[1].normal_load);
    let pitch_static = sim.nose_down();
    sim.set_speed(30.0);
    sim.veh.input.brake = fx(0.2);
    sim.frames(60);
    let v_start = f(sim.body().velocity.z);
    let (mut h, mut pitch, mut front) = (0.0, 0.0, 0.0);
    for _ in 0..30 {
        sim.frame();
        h += f(sim.body().position.y);
        pitch += sim.nose_down();
        front += f(sim.veh.wheels[0].normal_load + sim.veh.wheels[1].normal_load);
        assert!(
            sim.veh.wheels.iter().all(|w| w.omega > Fix128::ZERO),
            "pedal 0.2 must not lock a wheel"
        );
    }
    let a = (v_start - f(sim.body().velocity.z)) / (30.0 * DT);
    Braking {
        a,
        h: h / 30.0,
        pitch: pitch / 30.0,
        pitch_static,
        front: front / 30.0,
        front_static,
        g,
    }
}

/// Closed form (moment balance about the CG, contact forces at ground level):
/// front axle `F_zf = M g b / L + M a h / L` (static: `M g b / L`).
///
/// Tolerance `M g (h + 0.5) |φ| / L + 5 N`: pitching by `φ` moves the CG
/// over the contacts by `h φ` and the contacts under the attachments by
/// `(rest + r) φ ≤ 0.6 φ`; the suspension transient after 1 s is
/// `e^{-9}` of its amplitude (`< 5 N`). The damper's frame-sampling bias is
/// equal at all four corners and cancels in the axle split.
#[test]
fn braking_load_transfer_to_front_axle() {
    let br = steady_braking();
    let (l, b) = (2.4, 1.2);
    let static_want = M * br.g * b / l;
    assert!(
        (br.front_static - static_want).abs() <= 5.0,
        "static front axle {} N, closed form {static_want}",
        br.front_static
    );
    let want = M * br.g * b / l + M * br.a * br.h / l;
    let tol = M * br.g * (br.h + 0.5) * br.pitch.abs() / l + 5.0;
    assert!(br.a > 3.0, "steady deceleration {} m/s²", br.a);
    assert!(
        (br.front - want).abs() <= tol,
        "front axle {} N under a = {} m/s², closed form {want} ± {tol}",
        br.front,
        br.a
    );
}

// ---------------------------------------------------------------------------
// 7. Slope with locked wheels
// ---------------------------------------------------------------------------

/// Car on an `InclinedPlane` rising towards `+z` at angle `θ`, nose uphill,
/// all wheels locked (10 kNm), starting at rest at the closed-form ride
/// height along the normal. Returns the sim and the uphill unit tangent.
fn slope_sim(theta: f64) -> (Sim<InclinedPlane>, Vec3Fix) {
    let mut cfg = config(1.2, 1.2);
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let w = world();
    let g = g_of(&w);
    let normal = v3(0.0, theta.cos(), -theta.sin());
    let tangent = v3(0.0, theta.sin(), theta.cos());
    let d0 = ride_height(M, g * theta.cos(), 4.0);
    let mut sim = Sim::new(
        w,
        cfg,
        InclinedPlane {
            point: Vec3Fix::ZERO,
            normal,
        },
        condition(Weather::Dry, 0.0),
        normal * fx(d0),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_X, fx(-theta)),
        M,
    );
    sim.veh.input.brake = Fix128::ONE;
    (sim, tangent)
}

/// Drift bound for a held car on a slope: static friction anchors each
/// contact, and within one frame the uncorrected downhill acceleration
/// `g sin θ` can move the CG by at most `g sin θ dt²` before the anchor
/// correction of the next frame (`0.447 · 10 / 3600 = 1.24e-3 m` at
/// `tan θ = 0.5`). To be checked against the measured anchor residual at
/// integration.
fn slope_hold_drift_bound(g: f64, theta: f64) -> f64 {
    g * theta.sin() * DT * DT
}

/// `tan θ = 0.5 < μ_s = 1.0`: Coulomb statics holds the car
/// (`M g sin θ ≤ μ_s M g cos θ`).
///
/// Oracle, over 600 frames after 120 frames of settling:
/// - bounded: `|drift along the slope| ≤ g sin θ dt²` at every frame
///   (`slope_hold_drift_bound`)
/// - not accumulating: the largest `|drift|` of frames 301..600 does not
///   exceed that of frames 1..300 by more than `1e-9 m` (a creeping car,
///   e.g. a slip-based tyre that needs motion to make force, grows its drift
///   linearly and fails here even when each frame is small)
///
/// The suspension transient of settling decays as `e^{-9 t}` (see
/// `locked_car_comes_to_exact_rest`), `e^{-18}` of the initial offset after
/// 2 s, below `1e-9 m`.
#[test]
fn slope_below_static_friction_holds_the_car() {
    let theta = 0.5_f64.atan();
    let (mut sim, tangent) = slope_sim(theta);
    let bound = slope_hold_drift_bound(sim.g(), theta);
    sim.frames(120);
    let p0 = sim.body().position;
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for k in 0..600 {
        sim.frame();
        let drift = f((sim.body().position - p0).dot(tangent));
        assert!(
            drift.abs() <= bound,
            "frame {k}: drift {drift} m beyond {bound}"
        );
        if k < 300 {
            first = first.max(drift.abs());
        } else {
            second = second.max(drift.abs());
        }
    }
    assert!(
        second <= first + 1e-9,
        "drift accumulates: max |drift| {first} m (frames 1..300) then {second} m (301..600)"
    );
}

/// `tan θ = 1.2 > μ_s`: the locked car slides downhill with
/// `a = g (sin θ − μ_k cos θ) = 10 (0.76822 − 0.8 · 0.64018) = 2.5608 m/s²`.
///
/// Measured over 60 frames once the sliding speed exceeds
/// `max(1, 10 v_floor)` (full sliding, `|κ| = 1`). Tolerance 0.5 %: constant
/// force so no integration error in `Δv / Δt`; residual suspension transient
/// `< e^{-9}`.
#[test]
fn slope_above_static_friction_slides_with_kinetic_deceleration() {
    let theta = 1.2_f64.atan();
    let (mut sim, tangent) = slope_sim(theta);
    let g = sim.g();
    let v_floor = f(sim.veh.config.slip_velocity_floor);
    let v_gate = 1.0_f64.max(10.0 * v_floor);
    let downhill = |s: &Sim<InclinedPlane>| -f(s.body().velocity.dot(tangent));
    let mut k = 0;
    while downhill(&sim) < v_gate {
        sim.frame();
        k += 1;
        assert!(k < 600, "car did not start sliding within 10 s");
    }
    let v1 = downhill(&sim);
    sim.frames(60);
    let a = (downhill(&sim) - v1) / (60.0 * DT);
    let want = g * (theta.sin() - MU_K * theta.cos());
    assert!(
        (a - want).abs() <= 0.005 * want,
        "sliding acceleration {a} m/s², closed form {want}"
    );
}

// ---------------------------------------------------------------------------
// 8. Rolling-resistance coast-down
// ---------------------------------------------------------------------------

/// Throttle 0, engine braking 0, aero 0, `C_rr = 0.012`, from 20 m/s with the
/// wheels already rolling.
///
/// Closed form: the rolling-resistance force `C_rr Σ F_z = C_rr M g` also
/// spins down the four wheels, so the decelerating mass is
/// `M + 4 I_w / r²`: `a = C_rr M g / (M + 4 I_w / r²)
/// = 0.012 · 1000 · 10 / 1044.44 = 0.11489 m/s²`, constant (speed falls
/// linearly).
///
/// Measured over two consecutive 2 s windows; each must match `a` and they
/// must match each other. Tolerance 1 %: tyre slip needed for the spin-down
/// torque `κ = I_w a / (r² C_κ) ≈ 2e-5`; constant force otherwise.
#[test]
fn rolling_resistance_coast_down_is_linear() {
    let c_rr = 0.012;
    let mut sim = flat_sim(config(1.2, 1.2), condition(Weather::Dry, c_rr));
    let g = sim.g();
    sim.frames(120);
    sim.set_speed(20.0);
    sim.frames(30);
    let v0 = f(sim.body().velocity.z);
    sim.frames(120);
    let v1 = f(sim.body().velocity.z);
    sim.frames(120);
    let v2 = f(sim.body().velocity.z);
    let want = c_rr * M * g / (M + 4.0 * I_WHEEL / (R_WHEEL * R_WHEEL));
    let (a1, a2) = ((v0 - v1) / 2.0, (v1 - v2) / 2.0);
    for (i, a) in [a1, a2].into_iter().enumerate() {
        assert!(
            (a - want).abs() <= 0.01 * want,
            "window {i}: deceleration {a} m/s², closed form {want}"
        );
    }
    assert!(
        (a1 - a2).abs() <= 0.01 * want,
        "non-linear coast-down: {a1} vs {a2}"
    );
}

// ---------------------------------------------------------------------------
// 9. Top speed per gear
// ---------------------------------------------------------------------------

/// Full throttle, no resistances (aero 0, `C_rr = 0`), rev limit 3000 rpm,
/// final drive 1. The speed converges to
/// `v_n = max_rpm · 2π / 60 · r / ratio_n`: gear 1 `26.928 m/s`,
/// gear 2 `42.840 m/s`.
///
/// Tolerance `κ_n v_n + a_n dt`: the last full-torque frame leaves the wheel
/// ahead of the road by the drive slip `κ_n = F_n / C_κ` with
/// `F_n = T ratio_n / (2 r)` per driven wheel (gear 1: 1750 N, `κ = 2.2 %`),
/// and the bang-bang limiter overshoots by at most one frame of acceleration
/// `a_n = 2 F_n / (M + 4 I_w / r²)`. Also during gear 1 the speed never
/// exceeds `v_1 (1 + κ_1) + a_1 dt`.
#[test]
fn gear_top_speed_converges_to_rev_limit() {
    let max_rpm = 3000.0;
    let mut cfg = config(1.2, 1.2);
    cfg.powertrain = powertrain(max_rpm);
    let mut sim = flat_sim(cfg, condition(Weather::Dry, 0.0));
    sim.frames(120);
    sim.veh.input.throttle = Fix128::ONE;
    let m_eff = M + 4.0 * I_WHEEL / (R_WHEEL * R_WHEEL);
    for (gear, frames) in [(0usize, 720usize), (1, 900)] {
        let ratio = GEARS[gear];
        let v_lim = max_rpm * 2.0 * std::f64::consts::PI / 60.0 * R_WHEEL / ratio;
        let f_wheel = ENGINE_TORQUE * ratio / (2.0 * R_WHEEL);
        let kappa = f_wheel / C_KAPPA;
        let a_n = 2.0 * f_wheel / m_eff;
        let tol = kappa * v_lim + a_n * DT;
        assert_eq!(sim.veh.config.powertrain.current_gear, gear);
        for _ in 0..frames {
            sim.frame();
            let v = f(sim.body().velocity.z);
            assert!(
                v <= v_lim + tol,
                "gear {}: speed {v} above limit {v_lim} + {tol}",
                gear + 1
            );
        }
        let mut v_mean = 0.0;
        for _ in 0..60 {
            sim.frame();
            v_mean += f(sim.body().velocity.z) / 60.0;
        }
        assert!(
            (v_mean - v_lim).abs() <= tol,
            "gear {}: top speed {v_mean} m/s, closed form {v_lim} ± {tol}",
            gear + 1
        );
        sim.veh.config.powertrain.shift_up();
    }
}

// ---------------------------------------------------------------------------
// 10. Hydroplaning
// ---------------------------------------------------------------------------

/// Locked-wheel deceleration at speed `v` on a wet road with a 10 mm water
/// film, measured over 15 frames after 3 frames of lock onset.
fn wet_locked_deceleration(v: f64) -> (f64, f64) {
    let mut cfg = config(1.2, 1.2);
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let mut sim = flat_sim(
        cfg,
        condition(
            Weather::Wet {
                water_depth_mm: fx(10.0),
            },
            0.0,
        ),
    );
    sim.frames(120);
    sim.set_speed(v);
    sim.veh.input.brake = Fix128::ONE;
    sim.frames(3);
    let v1 = f(sim.body().velocity.z);
    sim.frames(15);
    let v2 = f(sim.body().velocity.z);
    ((v1 - v2) / (15.0 * DT), sim.g())
}

/// Horne onset `V = 6.35 √p km/h` at 220 kPa: `26.16 m/s`.
///
/// - at `0.8 V` (stays below `V` over the window): no hydroplaning loss,
///   deceleration `μ_k f_wet g` ± 2 % (pitch transient of lock onset moves
///   load between axles, the sum stays `M g`; window-averaged)
/// - at `1.2 V` (stays above `V`): deceleration `≤ 0.96 μ_k f_wet g`, i.e.
///   smaller by more than twice the measurement tolerance
#[test]
fn hydroplaning_reduces_deceleration_only_above_onset_speed() {
    let v_h = 6.35 * TYRE_KPA.sqrt() / 3.6;
    let f_wet = f(condition(
        Weather::Wet {
            water_depth_mm: fx(10.0),
        },
        0.0,
    )
    .weather_factor());
    let (below, g) = wet_locked_deceleration(0.8 * v_h);
    let want = MU_K * f_wet * g;
    assert!(
        (below - want).abs() <= 0.02 * want,
        "below onset: deceleration {below} m/s², wet sliding {want}"
    );
    let (above, _) = wet_locked_deceleration(1.2 * v_h);
    assert!(
        above <= 0.96 * want,
        "above onset ({} m/s): deceleration {above} m/s² not reduced from {want}",
        1.2 * v_h
    );
}

// ---------------------------------------------------------------------------
// 11. Forces at the contact points (wiring)
// ---------------------------------------------------------------------------

/// Steering right at 10 m/s (`δ = 0.05 rad`, `a = b` so `K = 0`) yields a
/// positive yaw rate of at least half the kinematic `v tan δ / L = 0.2085`;
/// a model that applies the tyre forces at the CG yields none.
#[test]
fn steering_produces_yaw_through_contact_points() {
    let mut sim = flat_sim(config(1.2, 1.2), condition(Weather::Dry, 0.0));
    sim.frames(120);
    sim.set_speed(10.0);
    sim.veh.input.steering = fx(0.1);
    sim.frames(60);
    let r = f(sim.body().angular_velocity.y);
    let kin = 10.0 * 0.05_f64.tan() / 2.4;
    assert!(r >= 0.5 * kin, "yaw rate {r} rad/s, kinematic {kin}");
}

/// Braking pitches the nose down by
/// `φ = M a h / (L² k_w)` with corner rate `k_w = k / rest = 166667 N/m`
/// (front axle gains `M a h / L`, each front spring compresses
/// `M a h / (2 L k_w)`, each rear one extends as much, over the wheelbase).
/// Measured relative to the static pitch. Tolerance 5 %: arm changes
/// `O(φ (h + 0.6) / a) < 0.3 %`, damper bias cancels front / rear.
#[test]
fn braking_pitches_nose_down_through_contact_points() {
    let br = steady_braking();
    let k_w = K_SPRING / REST;
    let l = 2.4_f64;
    let want = M * br.a * br.h / (l * l * k_w);
    let got = br.pitch - br.pitch_static;
    assert!(
        (got - want).abs() <= 0.05 * want,
        "braking pitch {got} rad, closed form {want}"
    );
}

// ---------------------------------------------------------------------------
// 12. Cross wind
// ---------------------------------------------------------------------------

const WIND: f64 = 10.0;
const CD: f64 = 1.0;
const AREA: f64 = 2.0;
const RHO: f64 = 1.225;

fn cross_wind() -> WindZone {
    WindZone {
        shape: ZoneShape::Aabb {
            min: v3(-1.0e4, -1.0e4, -1.0e4),
            max: v3(1.0e4, 1.0e4, 1.0e4),
        },
        air_density_kg_m3: fx(RHO),
        direction: v3(1.0, 0.0, 0.0),
        base_speed_m_s: fx(WIND),
        turbulence_amplitude: Fix128::ZERO,
        gust_frequency_hz: Fix128::ZERO,
        drag_coefficient: fx(CD),
        reference_area_m2: fx(AREA),
    }
}

/// Car in the air (no tyre force) moving at `v` along `+z` in a 10 m/s wind
/// along `+x`: one update changes the velocity by `F dt / M` with
/// `F = ½ ρ C_d A |v_rel| v_rel`, `v_rel = (10, 0, −v)` (`WindZone` contract).
/// Checked at `v = 0, 10, 20`: lateral component `∝ |v_rel| · 10`, the
/// component along the motion `∝ −|v_rel| v`. Tolerance `1e-9` relative
/// (Fix128 rounding).
#[test]
fn cross_wind_force_follows_relative_velocity_squared() {
    for v in [0.0, 10.0, 20.0] {
        let w = world();
        let mut sim = Sim::new(
            w,
            config(1.2, 1.2),
            FlatGround {
                height: Fix128::ZERO,
            },
            condition(Weather::Dry, 0.0),
            v3(0.0, 50.0, 0.0),
            QuatFix::IDENTITY,
            M,
        );
        sim.wind = Some(cross_wind());
        sim.set_speed(v);
        let before = sim.body().velocity;
        let env = Environment {
            condition: &sim.cond,
            wind: sim.wind.as_ref(),
            time: Fix128::ZERO,
            gravity: sim.world.config.gravity,
        };
        sim.veh
            .update(&mut sim.world.bodies[sim.car], &sim.road, &env, dt());
        let dv = sim.body().velocity - before;
        sim.world.step(dt());
        assert_eq!(sim.veh.grounded_wheels(), 0, "car is airborne");
        let rel = (WIND * WIND + v * v).sqrt();
        let k = 0.5 * RHO * CD * AREA * rel * DT / M;
        let (want_x, want_z) = (k * WIND, -k * v);
        let tol = 1e-9 * k * rel + 1e-15;
        assert!(
            (f(dv.x) - want_x).abs() <= tol && (f(dv.z) - want_z).abs() <= tol,
            "v = {v}: Δv = ({}, {}), closed form ({want_x}, {want_z})",
            f(dv.x),
            f(dv.z)
        );
        assert!(f(dv.y).abs() <= 1e-15, "v = {v}: wind has no vertical part");
    }
}

/// On the road at 15 m/s with the same cross wind, after 2 s the tyres carry
/// the wind's lateral force: `Σ lateral_force = −F_x` with
/// `F_x = ½ ρ C_d A |v_rel| (10 − v_x)` evaluated at the measured velocity
/// (unsteered wheels: wheel `y` axis = world `+x`). Tolerance 3 %: `a = b`
/// and equal `C_α` make the CG force a pure side-slip load (no yaw moment);
/// the drag part decelerates the car by `≈ 0.33 m/s²`, so the lateral
/// balance is quasi-steady (side-slip time constant `M v / (4 C_α) = 0.06 s`).
#[test]
fn cross_wind_on_the_road_is_carried_by_lateral_tyre_force() {
    let mut sim = flat_sim(config(1.2, 1.2), condition(Weather::Dry, 0.0));
    sim.wind = Some(cross_wind());
    sim.frames(120);
    sim.set_speed(15.0);
    sim.frames(120);
    let v = sim.body().velocity;
    let rel_x = WIND - f(v.x);
    let rel = (rel_x * rel_x + f(v.y).powi(2) + f(v.z).powi(2)).sqrt();
    let f_x = 0.5 * RHO * CD * AREA * rel * rel_x;
    let lateral: f64 = sim.veh.wheels.iter().map(|w| f(w.lateral_force)).sum();
    assert!(
        (lateral + f_x).abs() <= 0.03 * f_x,
        "Σ lateral tyre force {lateral} N, wind side force {f_x} N"
    );
    assert!(f(v.x) > 0.0, "cross wind pushes the car towards +x");
}
