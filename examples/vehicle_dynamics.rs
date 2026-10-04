//! Per-wheel vehicle dynamics: braking on dry, wet and icy roads, ABS, and a
//! car parked on a slope.
//!
//! Starts from `DynamicVehicleConfig::passenger_car()` (legacy wheel layout,
//! brush tyres, powertrain built from the legacy engine table) on
//! `RoadCondition::dry_asphalt()`, and calls `DynamicVehicle::update` once per
//! frame before `PhysicsWorld::step`.
//!
//! Closed forms checked:
//!
//! - all four wheels locked from `v0`: Coulomb sliding stops the car after
//!   `s = v0² / (2 μ_k g)`, `μ_k` the road's sliding coefficient times the
//!   weather factor. Tolerance `2 v0 dt + v_floor² / (μ_k g)` (one frame of
//!   travel for the frame integration, one for the lock onset, and the slip
//!   floor below `v_floor`; same derivation as
//!   `tests/analytic_vehicle_dynamics.rs`)
//! - with ABS: no wheel locks above the ABS cut-off speed, and no tyre can
//!   exceed the peak coefficient, so `s ≥ v0² / (2 μ_s g) − v0 dt`
//! - locked wheels on a slope with `tan θ < μ_s`: the car stays put; the
//!   drift along the slope never exceeds `g sin θ dt²` and does not grow
//! - the same locked-wheel stop on a height field, a triangle mesh and an SDF
//!   road (each a horizontal surface): same closed form and tolerance
//! - full throttle against the rev limit `n_max`: the car settles at
//!   `v = n_max · 2π / 60 / R · r` for the total ratio `R` of the gear, and
//!   shifting up raises that speed
//!
//! Aerodynamic drag and rolling resistance are switched off and the world's
//! per-frame velocity damping is set to 1 so that only tyre friction slows
//! the car (the closed forms above have no other resistance).
//!
//! ```bash
//! cargo run --release --example vehicle_dynamics --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::trimesh::{TriMesh, Triangle};
use alice_physics::vehicle::EngineConfig;
use alice_physics::vehicle_dynamics::powertrain::Powertrain;
use alice_physics::vehicle_dynamics::surface::{
    FlatGround, HeightFieldRoad, InclinedPlane, RoadCondition, RoadSurface, SdfRoad, TriMeshRoad,
    Weather,
};
use alice_physics::vehicle_dynamics::tire::{MagicFormulaTire, TireInput, TireModel};
use alice_physics::vehicle_dynamics::{
    AbsConfig, DynamicVehicle, DynamicVehicleConfig, Environment,
};

const MASS: f64 = 1000.0;
const DT: f64 = 1.0 / 60.0;
const V0: f64 = 20.0;
const LOCK_TORQUE: f64 = 10_000.0;

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Passenger car with aerodynamics off (see module doc).
fn car_config() -> DynamicVehicleConfig {
    let mut cfg = DynamicVehicleConfig::passenger_car();
    cfg.aero.drag_area = Fix128::ZERO;
    cfg.aero.lift_area = Fix128::ZERO;
    cfg
}

/// Dry asphalt in the given weather, rolling resistance off.
fn road(weather: Weather) -> RoadCondition {
    RoadCondition {
        weather,
        rolling_resistance: Fix128::ZERO,
        ..RoadCondition::dry_asphalt()
    }
}

struct Sim<R: RoadSurface> {
    world: PhysicsWorld,
    car: usize,
    vehicle: DynamicVehicle,
    road: R,
    condition: RoadCondition,
    t: Fix128,
}

impl<R: RoadSurface> Sim<R> {
    fn new(
        cfg: DynamicVehicleConfig,
        road: R,
        condition: RoadCondition,
        pos: Vec3Fix,
        rot: QuatFix,
    ) -> Self {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        world.config.damping = Fix128::ONE;
        let mut body = RigidBody::new_dynamic(pos, fx(MASS));
        body.rotation = rot;
        body.prev_rotation = rot;
        // Box-like chassis inertia: pitch 1.5 M, yaw 1.8 M, roll 0.5 M (kg m²).
        body.inv_inertia = Vec3Fix::new(
            fx(1.0 / (1.5 * MASS)),
            fx(1.0 / (1.8 * MASS)),
            fx(1.0 / (0.5 * MASS)),
        );
        let car = world.add_body(body);
        Self {
            world,
            car,
            vehicle: DynamicVehicle::new(cfg),
            road,
            condition,
            t: Fix128::ZERO,
        }
    }

    fn frame(&mut self) {
        let env = Environment {
            condition: &self.condition,
            wind: None,
            time: self.t,
        };
        self.vehicle
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
        self.world.config.gravity.length().to_f64()
    }

    fn forward_speed(&self) -> f64 {
        DynamicVehicle::forward_speed(self.body()).to_f64()
    }

    /// Chassis at forward speed `v`, every wheel rolling at `v / r`.
    fn set_speed(&mut self, v: f64) {
        let fwd = self.body().rotation.rotate_vec(Vec3Fix::UNIT_Z);
        self.world.bodies[self.car].velocity = fwd * fx(v);
        for (w, cfg) in self
            .vehicle
            .wheels
            .iter_mut()
            .zip(&self.vehicle.config.base.wheels)
        {
            w.omega = fx(v) / cfg.radius;
        }
    }
}

/// Static ride height of the CG above the road along its normal, with gravity
/// component `g_n` along the normal: wheels hang 0.2 m below the CG, spring
/// force `k · compression` with `compression = M g_n / (4 k)` per wheel.
fn ride_height(cfg: &DynamicVehicleConfig, g_n: f64) -> f64 {
    let w = &cfg.base.wheels[0];
    let (r, rest, k) = (
        w.radius.to_f64(),
        w.suspension_rest.to_f64(),
        w.spring_stiffness.to_f64(),
    );
    -w.local_position.y.to_f64() + r + rest * (1.0 - MASS * g_n / (4.0 * k))
}

/// Car at rest on a horizontal road whose surface is at `ground_y`, at its
/// closed-form ride height, after 2 s of settling.
fn level_sim<R: RoadSurface>(
    cfg: DynamicVehicleConfig,
    road: R,
    ground_y: f64,
    condition: RoadCondition,
) -> Sim<R> {
    let g = PhysicsConfig::default().gravity.length().to_f64();
    let y0 = ground_y + ride_height(&cfg, g);
    let mut sim = Sim::new(
        cfg,
        road,
        condition,
        Vec3Fix::new(Fix128::ZERO, fx(y0), Fix128::ZERO),
        QuatFix::IDENTITY,
    );
    sim.frames(120);
    sim
}

fn flat_sim(cfg: DynamicVehicleConfig, condition: RoadCondition) -> Sim<FlatGround> {
    level_sim(
        cfg,
        FlatGround {
            height: Fix128::ZERO,
        },
        0.0,
        condition,
    )
}

/// All four wheels locked from `V0`; returns the stopping distance.
fn locked_stop(name: &str, weather: Weather) -> f64 {
    locked_stop_on(
        name,
        FlatGround {
            height: Fix128::ZERO,
        },
        0.0,
        weather,
    )
}

/// All four wheels locked from `V0` on a horizontal `road` whose surface is at
/// `ground_y`; returns the stopping distance (asserted against the closed form).
fn locked_stop_on<R: RoadSurface>(
    name: &str,
    road_surface: R,
    ground_y: f64,
    weather: Weather,
) -> f64 {
    let mut cfg = car_config();
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    cfg.brakes.abs = None;
    let condition = road(weather);
    let mu_k =
        condition.material.longitudinal_kinetic.to_f64() * condition.weather_factor().to_f64();
    let mu_s =
        condition.material.longitudinal_static.to_f64() * condition.weather_factor().to_f64();
    let r = cfg.base.wheels[0].radius.to_f64();
    let i_w = cfg.wheel_inertia.to_f64();
    let v_floor = cfg.slip_velocity_floor.to_f64();
    let mut sim = level_sim(cfg, road_surface, ground_y, condition);
    assert_eq!(
        sim.vehicle.grounded_wheels(),
        4,
        "{name}: settled on the road"
    );
    let g = sim.g();
    // Lock-onset precondition of the tolerance: the wheel stops within one frame.
    let t_lock = i_w * (V0 / r) / (LOCK_TORQUE - mu_s * MASS * g * r);
    assert!(t_lock < DT, "lock onset {t_lock} s must fit in one frame");

    sim.set_speed(V0);
    let z0 = sim.body().position.z.to_f64();
    sim.vehicle.input.brake = Fix128::ONE;
    let t_stop = V0 / (mu_k * g);
    sim.frames((1.5 * t_stop / DT) as usize + 120);
    let s = sim.body().position.z.to_f64() - z0;
    let want = V0 * V0 / (2.0 * mu_k * g);
    let tol = 2.0 * V0 * DT + v_floor * v_floor / (mu_k * g);
    println!(
        "[vehicle_dynamics] locked {name:<11} mu_k {mu_k:.3}: s = {s:8.3} m \
         (closed form {want:8.3} ± {tol:.3}), final speed {:.1e} m/s",
        sim.forward_speed()
    );
    assert!(
        (s - want).abs() <= tol,
        "{name}: stopping distance {s} m, closed form {want} ± {tol}"
    );
    s
}

/// Full pedal from `V0` on dry asphalt with the default brakes (2500 / 1500 Nm
/// per wheel); returns (distance, whether a wheel locked above `min_speed`).
fn braked_stop(abs: Option<AbsConfig>) -> (f64, bool) {
    let mut cfg = car_config();
    cfg.brakes.abs = abs;
    let mut sim = flat_sim(cfg, road(Weather::Dry));
    let min_speed = 3.0;
    sim.set_speed(V0);
    let z0 = sim.body().position.z.to_f64();
    sim.vehicle.input.brake = Fix128::ONE;
    let mut locked = false;
    let mut stopped = false;
    for _ in 0..(10.0 / DT) as usize {
        sim.frame();
        let v = sim.forward_speed();
        if v > min_speed && sim.vehicle.wheels.iter().any(|w| w.omega <= Fix128::ZERO) {
            locked = true;
        }
        if v <= 0.0 {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "car stops within 10 s");
    sim.frames(120);
    (sim.body().position.z.to_f64() - z0, locked)
}

/// Locked car parked on a slope `tan θ = 0.5`: the drift along the slope is
/// bounded by `g sin θ dt²` and does not grow between the two halves of 10 s.
fn parked_on_slope() {
    let theta = 0.5_f64.atan();
    let mut cfg = car_config();
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let condition = road(Weather::Dry);
    let mu_s = condition.material.longitudinal_static.to_f64();
    assert!(theta.tan() < mu_s, "slope below static friction");
    let g0 = PhysicsConfig::default().gravity.length().to_f64();
    let d0 = ride_height(&cfg, g0 * theta.cos());
    let normal = Vec3Fix::new(Fix128::ZERO, fx(theta.cos()), fx(-theta.sin()));
    let tangent = Vec3Fix::new(Fix128::ZERO, fx(theta.sin()), fx(theta.cos()));
    let mut sim = Sim::new(
        cfg,
        InclinedPlane {
            point: Vec3Fix::ZERO,
            normal,
        },
        condition,
        normal * fx(d0),
        QuatFix::from_axis_angle(Vec3Fix::UNIT_X, fx(-theta)),
    );
    sim.vehicle.input.brake = Fix128::ONE;
    sim.frames(120);
    let bound = sim.g() * theta.sin() * DT * DT;
    let p0 = sim.body().position;
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for k in 0..600 {
        sim.frame();
        let drift = (sim.body().position - p0).dot(tangent).to_f64();
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
    println!(
        "[vehicle_dynamics] parked on tan θ = 0.5: max drift {first:.2e} m (0-5 s), \
         {second:.2e} m (5-10 s), bound g sin θ dt² = {bound:.2e} m, {} wheels grounded",
        sim.vehicle.grounded_wheels()
    );
    assert_eq!(sim.vehicle.grounded_wheels(), 4);
    assert!(second <= first + 1e-9, "drift accumulates");
}

/// Longitudinal force of the two passenger-car tyre models at a few slip
/// ratios (braking) on dry asphalt, 2500 N load.
fn tyre_curves() {
    let condition = road(Weather::Dry);
    let grip = condition.grip(Fix128::ZERO, fx(220.0));
    let fz = 2500.0;
    let brush = car_config().tire;
    let mf = TireModel::MagicFormula(MagicFormulaTire::passenger_car());
    let peak = grip.longitudinal_static.to_f64() * fz;
    for kappa in [-0.02, -0.05, -0.12, -0.3, -1.0] {
        let input = TireInput {
            slip_ratio: fx(kappa),
            slip_tan_alpha: Fix128::ZERO,
            normal_load: fx(fz),
            grip,
        };
        let fb = brush.force(&input).longitudinal.to_f64();
        let fm = mf.force(&input).longitudinal.to_f64();
        println!(
            "[vehicle_dynamics] tyre kappa {kappa:5.2}: brush F_x {fb:8.1} N, \
             Magic Formula F_x {fm:8.1} N (peak mu_s F_z = {peak:.1} N)"
        );
        assert!(fb <= 0.0 && fm <= 0.0, "braking slip gives a braking force");
        assert!(
            fb.abs() <= peak + 1e-6 && fm.abs() <= peak + 1e-6,
            "tyre force within the peak coefficient"
        );
    }
    // Brush contract: a locked wheel slides at exactly μ_k F_z.
    let locked = brush.force(&TireInput {
        slip_ratio: Fix128::NEG_ONE,
        slip_tan_alpha: Fix128::ZERO,
        normal_load: fx(fz),
        grip,
    });
    assert_eq!(locked.longitudinal, -(grip.longitudinal_kinetic * fx(fz)));
}

/// Locked-wheel stop on the three geometry-backed roads (height field,
/// triangle mesh, SDF), each a horizontal surface covering the stopping path:
/// the same closed form `v0² / (2 μ_k g)` as on `FlatGround` must hold.
fn stops_on_road_geometries() {
    // Height field 9 × 41 points, 1 m spacing, x ∈ [−4, 4], z ∈ [−5, 35], all
    // heights 0.5. The road follows `sample_height`, which is read here.
    let field = HeightField::flat(
        9,
        41,
        Fix128::ONE,
        Vec3Fix::new(fx(-4.0), Fix128::ZERO, fx(-5.0)),
        fx(0.5),
    );
    let field_y = field.sample_height(Fix128::ZERO, Fix128::ZERO).to_f64();
    locked_stop_on(
        "heightfield",
        HeightFieldRoad { field: &field },
        field_y,
        Weather::Dry,
    );

    // Two triangles covering x ∈ [−5, 5], z ∈ [−5, 40] at y = 0, wound
    // counter-clockwise seen from above (outward normal up).
    let y = Fix128::ZERO;
    let a = Vec3Fix::new(fx(-5.0), y, fx(-5.0));
    let b = Vec3Fix::new(fx(-5.0), y, fx(40.0));
    let c = Vec3Fix::new(fx(5.0), y, fx(40.0));
    let d = Vec3Fix::new(fx(5.0), y, fx(-5.0));
    let mesh = TriMesh::from_triangles(vec![Triangle::new(a, b, c), Triangle::new(a, c, d)]);
    locked_stop_on("trimesh", TriMeshRoad { mesh: &mesh }, 0.0, Weather::Dry);

    // SDF of the half-space below y = 0: distance y, normal +y.
    let plane = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    locked_stop_on(
        "sdf",
        SdfRoad {
            field: &plane,
            tolerance: Fix128::from_ratio(1, 10_000),
            max_steps: 64,
        },
        0.0,
        Weather::Dry,
    );
}

/// Rev-limited speed per gear. With the powertrain built from an engine whose
/// limit is `n_max` rpm, full throttle drives the wheels until the crank reaches
/// `n_max`, i.e. wheel spin `ω = n_max · 2π / 60 / R` (`R` the total ratio), and
/// with no resistance the car runs at `v = ω r`. Shifting up lowers `R`, so the
/// limit speed rises by the ratio of the gears.
///
/// Tolerance 2 %: the drive stops at the limit but the wheel can overshoot it
/// by one frame of spin-up, and the chassis lags the wheel by the small
/// rolling slip that the remaining drive force needs.
fn gears_raise_the_rev_limited_speed() {
    let mut cfg = car_config();
    let engine = EngineConfig {
        max_rpm: fx(2000.0),
        ..cfg.base.engine
    };
    cfg.powertrain = Powertrain::from_engine_config(&engine, &cfg.base.gear_ratios);
    let r = cfg.base.wheels[0].radius.to_f64();
    let n_max = engine.max_rpm.to_f64();
    let mut sim = flat_sim(cfg, road(Weather::Dry));
    sim.vehicle.input.throttle = Fix128::ONE;
    let mut last = 0.0;
    for gear in 0..2 {
        let pt = &sim.vehicle.config.powertrain;
        assert_eq!(pt.current_gear, gear);
        let ratio = pt.total_ratio().to_f64();
        let want = n_max * 2.0 * std::f64::consts::PI / 60.0 / ratio * r;
        sim.frames(900);
        let v = sim.forward_speed();
        println!(
            "[vehicle_dynamics] gear {} ratio {ratio}: speed {v:.3} m/s at the rev limit \
             (closed form {want:.3})",
            gear + 1
        );
        assert!(
            (v - want).abs() <= 0.02 * want,
            "gear {}: {v} m/s, closed form {want}",
            gear + 1
        );
        assert!(v > last, "a higher gear runs faster at the rev limit");
        last = v;
        sim.vehicle.config.powertrain.shift_up();
    }
    let pt = &mut sim.vehicle.config.powertrain;
    let top = pt.gear_ratios.len() - 1;
    for _ in 0..10 {
        pt.shift_up();
    }
    assert_eq!(pt.current_gear, top, "shift_up stops at the top gear");
    for _ in 0..10 {
        pt.shift_down();
    }
    assert_eq!(pt.current_gear, 0, "shift_down stops at first gear");
}

fn main() {
    tyre_curves();

    let dry = locked_stop("dry", Weather::Dry);
    let wet = locked_stop(
        "wet",
        Weather::Wet {
            water_depth_mm: Fix128::ONE,
        },
    );
    let ice = locked_stop("ice", Weather::Ice);
    stops_on_road_geometries();
    assert!(dry < wet && wet < ice, "less grip, longer stop");

    let abs = AbsConfig {
        target_slip: fx(0.12),
        min_speed: fx(3.0),
    };
    let (s_plain, locked_plain) = braked_stop(None);
    let (s_abs, locked_abs) = braked_stop(Some(abs));
    let g = PhysicsConfig::default().gravity.length().to_f64();
    let mu_s = road(Weather::Dry).material.longitudinal_static.to_f64();
    let lower = V0 * V0 / (2.0 * mu_s * g) - V0 * DT;
    println!(
        "[vehicle_dynamics] full pedal, no ABS: s = {s_plain:.3} m, wheel locked: {locked_plain}"
    );
    println!(
        "[vehicle_dynamics] full pedal, ABS:    s = {s_abs:.3} m, wheel locked: {locked_abs} \
         (peak-grip bound s ≥ {lower:.3} m)"
    );
    assert!(locked_plain, "these brakes lock a wheel without ABS");
    assert!(
        !locked_abs,
        "ABS keeps every wheel rolling above its cut-off"
    );
    assert!(s_abs >= lower, "no tyre exceeds the peak coefficient");

    parked_on_slope();
    gears_raise_the_rev_limited_speed();
}
