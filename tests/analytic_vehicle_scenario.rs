//! Oracles for `vehicle_dynamics::scenario`: following metrics (TTC, time
//! headway), the stopping-distance meter, the multi-vehicle runner and the
//! lossless input replay.
//!
//! Every test goes through the production entry point
//! (`Scenario::add_vehicle` / `step` / `step_with` / `run_tracks`,
//! `Recording::to_bytes` / `from_bytes` / `restore` / `replay`). Expected
//! values are closed forms in `f64` or bit comparisons against an
//! independent run; no scenario function is used to build an expectation.
//!
//! Vehicle setup is the one of `tests/analytic_vehicle_dynamics.rs`
//! (brush tyre `C_κ = 80 kN`, `C_α = 60 kN`, `I_w = 1 kg m²`, aero off,
//! flat 300 Nm powertrain, world damping 1, `M = 1000 kg`,
//! `μ_s = 1.0`, `μ_k = 0.8`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle_dynamics::powertrain::{Differential, Powertrain, TorqueCurve};
use alice_physics::vehicle_dynamics::scenario::{
    longitudinal_state, time_headway, time_to_collision, up_from_gravity, Recording, ReplayError,
    Scenario, StoppingDistanceMeter, RECORDING_MAGIC, RECORDING_VERSION,
};
use alice_physics::vehicle_dynamics::surface::{FlatGround, RoadCondition, Weather};
use alice_physics::vehicle_dynamics::tire::{BrushTire, TireModel};
use alice_physics::vehicle_dynamics::{
    AeroConfig, BrakeSystem, DriverInput, DynamicVehicle, DynamicVehicleConfig,
};
use alice_physics::wind_zone::WindZone;

const M: f64 = 1000.0;
const DT: f64 = 1.0 / 60.0;
const R_WHEEL: f64 = 0.3;
const REST: f64 = 0.3;
const K_SPRING: f64 = 50000.0;
const ATTACH_Y: f64 = -0.2;
const MU_S: f64 = 1.0;
const MU_K: f64 = 0.8;
const LOCK_TORQUE: f64 = 10000.0;
const CAR_LENGTH: f64 = 4.0;

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

fn condition() -> RoadCondition {
    RoadCondition {
        material: AnisotropicFriction {
            longitudinal_static: fx(MU_S),
            longitudinal_kinetic: fx(MU_K),
            transverse_static: fx(MU_S),
            transverse_kinetic: fx(MU_K),
            slip_threshold_m_s: fx(0.05),
        },
        weather: Weather::Dry,
        rolling_resistance: Fix128::ZERO,
    }
}

fn config() -> DynamicVehicleConfig {
    let mut c = DynamicVehicleConfig::passenger_car();
    let pos = [(-0.8, 1.2), (0.8, 1.2), (-0.8, -1.2), (0.8, -1.2)];
    for (w, (x, z)) in c.base.wheels.iter_mut().zip(pos) {
        w.local_position = v3(x, ATTACH_Y, z);
        w.radius = fx(R_WHEEL);
        w.suspension_rest = fx(REST);
        w.spring_stiffness = fx(K_SPRING);
        w.damping = fx(4500.0);
        w.progressive_rate = Fix128::ZERO;
        w.bump_damping = Fix128::ZERO;
        w.rebound_damping = Fix128::ZERO;
        w.bump_stop_stiffness = Fix128::ZERO;
        w.max_steer_angle = if z > 0.0 { fx(0.5) } else { Fix128::ZERO };
        w.driven = z < 0.0;
        w.has_brake = true;
    }
    c.wheel_inertia = Fix128::ONE;
    c.tire = TireModel::Brush(BrushTire {
        longitudinal_stiffness: fx(80000.0),
        cornering_stiffness: fx(60000.0),
    });
    c.tyre_pressure_kpa = fx(220.0);
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
    c.powertrain = Powertrain {
        curve: TorqueCurve {
            points: vec![(fx(800.0), fx(300.0)), (fx(7000.0), fx(300.0))],
        },
        idle_rpm: fx(800.0),
        max_rpm: fx(7000.0),
        engine_brake_per_rpm: Fix128::ZERO,
        gear_ratios: [3.5, 2.2, 1.5, 1.1, 0.8].iter().map(|&g| fx(g)).collect(),
        final_drive: Fix128::ONE,
        current_gear: 0,
        differential: Differential::Open,
    };
    c.ackermann = false;
    c
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.config.damping = Fix128::ONE;
    w
}

fn g_of(w: &PhysicsWorld) -> f64 {
    f(w.config.gravity.length())
}

fn ride_height(g: f64) -> f64 {
    -ATTACH_Y + R_WHEEL + REST * (1.0 - M * g / (4.0 * K_SPRING))
}

fn chassis(x: f64, z: f64, g: f64) -> RigidBody {
    let mut body = RigidBody::new_dynamic(v3(x, ride_height(g), z), fx(M));
    body.rotation = QuatFix::IDENTITY;
    body.prev_rotation = QuatFix::IDENTITY;
    body.inv_inertia = v3(1.0 / (1.5 * M), 1.0 / (1.8 * M), 1.0 / (0.5 * M));
    body
}

fn empty_scenario() -> Scenario<FlatGround> {
    Scenario::new(
        world(),
        FlatGround {
            height: Fix128::ZERO,
        },
        condition(),
        dt(),
    )
}

/// Scenario with cars at the given `(x, z)` positions, in order.
fn scenario(cars: &[(f64, f64)], cfg: &DynamicVehicleConfig) -> Scenario<FlatGround> {
    let mut sc = empty_scenario();
    let g = g_of(&sc.world);
    for &(x, z) in cars {
        sc.add_vehicle(DynamicVehicle::new(cfg.clone()), chassis(x, z, g));
    }
    sc
}

/// Give car `i` forward speed `v` (+z) with wheels rolling at `v / r`.
fn set_speed(sc: &mut Scenario<FlatGround>, i: usize, v: f64) {
    let body = sc.vehicles[i].body;
    sc.world.bodies[body].velocity = v3(0.0, 0.0, v);
    for w in &mut sc.vehicles[i].vehicle.wheels {
        w.omega = fx(v / R_WHEEL);
    }
}

fn input(throttle: f64, brake: f64, steering: f64) -> DriverInput {
    DriverInput {
        throttle: fx(throttle),
        brake: fx(brake),
        handbrake: Fix128::ZERO,
        steering: fx(steering),
    }
}

/// Full dynamic state of one car: chassis kinematics + every wheel state +
/// engine speed. Compared with `==` (bit equality of every `Fix128`).
#[derive(Debug, PartialEq, Eq)]
struct CarState {
    position: Vec3Fix,
    velocity: Vec3Fix,
    rotation: QuatFix,
    angular_velocity: Vec3Fix,
    wheels: Vec<alice_physics::vehicle_dynamics::WheelDynamicsState>,
    engine_rpm: Fix128,
    gear: usize,
}

fn car_state(sc: &Scenario<FlatGround>, i: usize) -> CarState {
    let v = &sc.vehicles[i];
    let b = &sc.world.bodies[v.body];
    CarState {
        position: b.position,
        velocity: b.velocity,
        rotation: b.rotation,
        angular_velocity: b.angular_velocity,
        wheels: v.vehicle.wheels.clone(),
        engine_rpm: v.vehicle.engine_rpm,
        gear: v.vehicle.config.powertrain.current_gear,
    }
}

fn all_states(sc: &Scenario<FlatGround>) -> Vec<CarState> {
    (0..sc.vehicles.len()).map(|i| car_state(sc, i)).collect()
}

// ---------------------------------------------------------------------------
// 1. TTC and time headway
// ---------------------------------------------------------------------------

/// Pure metric: gap 30 m, rear 20 m/s, front 10 m/s ⇒ TTC `30 / (20 − 10) = 3 s`,
/// headway `30 / 20 = 1.5 s`; not closing (equal speeds, front faster) ⇒ `None`;
/// rear at rest ⇒ no headway.
#[test]
fn ttc_and_headway_closed_form_on_values() {
    let ttc = time_to_collision(fx(30.0), fx(20.0), fx(10.0)).expect("closing");
    assert!((f(ttc) - 3.0).abs() <= 1e-15, "TTC {}", f(ttc));
    let thw = time_headway(fx(30.0), fx(20.0)).expect("moving");
    assert!((f(thw) - 1.5).abs() <= 1e-15, "headway {}", f(thw));
    assert_eq!(time_to_collision(fx(30.0), fx(10.0), fx(10.0)), None);
    assert_eq!(time_to_collision(fx(30.0), fx(10.0), fx(12.0)), None);
    assert_eq!(time_headway(fx(30.0), Fix128::ZERO), None);
    assert_eq!(time_headway(fx(30.0), fx(-1.0)), None);
    // Overlapping bodies (gap < 0) that still close: contact is now.
    assert_eq!(
        time_to_collision(fx(-0.5), fx(20.0), fx(10.0)),
        Some(Fix128::ZERO)
    );
    assert_eq!(time_headway(fx(-0.5), fx(20.0)), Some(Fix128::ZERO));
}

/// Two cars in one lane on the runner, coasting (all inputs released, no
/// rolling resistance, aero off): rear at 20 m/s, front at 10 m/s, 30 m
/// bumper gap (`Δz = 30 + 4`, car length 4). For every frame of 1 s the
/// metric equals the closed form `gap / (v_rear − v_front)` evaluated in
/// `f64` from the chassis state (relative 1e-12), and over the 1 s the TTC
/// drops by `1 s` (constant speeds; tolerance 0.01 s covers the
/// suspension/tyre transient, whose speed change is checked below 0.05 m/s).
/// A front car pulling away gives `None` on every frame.
#[test]
fn ttc_on_the_runner_matches_closed_form() {
    let cfg = config();
    let mut sc = scenario(&[(0.0, 0.0), (0.0, 30.0 + CAR_LENGTH)], &cfg);
    sc.run_tracks(&[], 120);
    set_speed(&mut sc, 0, 20.0);
    set_speed(&mut sc, 1, 10.0);
    let dir = Vec3Fix::UNIT_Z;
    let extent = fx(CAR_LENGTH);
    let rear = sc.vehicles[0].body;
    let front = sc.vehicles[1].body;
    let ttc0 = {
        let (gap, vr, vf) =
            longitudinal_state(&sc.world.bodies[rear], &sc.world.bodies[front], dir, extent);
        f(time_to_collision(gap, vr, vf).expect("closing"))
    };
    assert!((ttc0 - 3.0).abs() <= 1e-9, "initial TTC {ttc0}");
    let mut last = ttc0;
    for k in 0..60 {
        sc.step(&[]);
        let br = &sc.world.bodies[rear];
        let bf = &sc.world.bodies[front];
        let (gap, vr, vf) = longitudinal_state(br, bf, dir, extent);
        let want_gap = f(bf.position.z) - f(br.position.z) - CAR_LENGTH;
        let want = want_gap / (f(br.velocity.z) - f(bf.velocity.z));
        let got = f(time_to_collision(gap, vr, vf).expect("closing"));
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "frame {k}: TTC {got}, closed form {want}"
        );
        assert!((f(br.velocity.z) - 20.0).abs() < 0.05, "rear speed drift");
        assert!((f(bf.velocity.z) - 10.0).abs() < 0.05, "front speed drift");
        last = got;
    }
    assert!(
        ((ttc0 - last) - 1.0).abs() <= 0.01,
        "TTC fell by {} over 1 s",
        ttc0 - last
    );
    // Swap the speeds: the front car pulls away ⇒ no TTC on any frame.
    set_speed(&mut sc, 0, 10.0);
    set_speed(&mut sc, 1, 20.0);
    for k in 0..30 {
        sc.step(&[]);
        let (gap, vr, vf) =
            longitudinal_state(&sc.world.bodies[rear], &sc.world.bodies[front], dir, extent);
        assert_eq!(time_to_collision(gap, vr, vf), None, "frame {k}");
        let thw = f(time_headway(gap, vr).expect("rear moves"));
        let want = f(gap) / f(vr);
        assert!((thw - want).abs() <= 1e-12 * want, "frame {k}: headway");
    }
}

// ---------------------------------------------------------------------------
// 2. Stopping-distance meter
// ---------------------------------------------------------------------------

/// Locked stop from 20 m/s (brakes 10 kNm per wheel, no ABS), measured by the
/// meter from the brake-on frame with the exact criterion (`rest_speed = 0`:
/// the forward speed reaches 0 or reverses): `s = v0² / (2 μ_k g)` with the
/// tolerance of `stopping_distance_dry_locked_wheels`
/// (`2 v0 dt + v_floor² / (μ g)`, same derivation).
///
/// Stop time `v0 / (μ g)` within `2 dt + v_floor / (μ g)` (frame
/// quantisation of the onset and of the latch, and the low-speed slip regime
/// below `v_floor` where the tyre force may leave the sliding value).
///
/// After the latch nothing drives the car: 5 s later its horizontal speed is
/// `≤ 1e-6 m/s`, the latched distance has not changed, and the chassis
/// origin is within the pitch recoil of the latched position: the brake dive
/// `θ = M μ g h / k_θ` (`h` = CG height over the road, pitch stiffness
/// `k_θ = 4 k a²` of the four springs at `±a = ±1.2`) relaxes and moves the
/// origin by at most `h θ` horizontally (1.7 cm here).
#[test]
fn stopping_distance_meter_matches_locked_closed_form() {
    let mut cfg = config();
    cfg.brakes.max_torque_front = fx(LOCK_TORQUE);
    cfg.brakes.max_torque_rear = fx(LOCK_TORQUE);
    let mut sc = scenario(&[(0.0, 0.0)], &cfg);
    let g = g_of(&sc.world);
    let v0 = 20.0;
    sc.run_tracks(&[], 120);
    set_speed(&mut sc, 0, v0);
    let up = up_from_gravity(sc.world.config.gravity).expect("gravity");
    let body = sc.vehicles[0].body;
    let mut meter = StoppingDistanceMeter::new(&sc.world.bodies[body], up, Fix128::ZERO);
    assert!(!meter.is_stopped());
    let brake = [input(0.0, 1.0, 0.0)];
    let mut latched = None;
    for k in 0..400 {
        sc.step(&brake);
        if meter.sample(&sc.world.bodies[body]) && latched.is_none() {
            latched = Some((k + 1, f(sc.world.bodies[body].position.z)));
        }
    }
    let (latched_at, z_latch) = latched.expect("the car stops");
    assert_eq!(meter.frames_to_stop(), Some(latched_at));
    let mu = MU_K;
    let v_floor = f(cfg.slip_velocity_floor);
    let expected = v0 * v0 / (2.0 * mu * g);
    let tol = 2.0 * v0 * DT + v_floor * v_floor / (mu * g);
    let s = f(meter.distance().expect("stopped"));
    assert!(
        (s - expected).abs() <= tol,
        "meter: s = {s} m, closed form {expected} ± {tol}"
    );
    let t_stop = latched_at as f64 * DT;
    let t_want = v0 / (mu * g);
    let t_tol = 2.0 * DT + v_floor / (mu * g);
    assert!(
        (t_stop - t_want).abs() <= t_tol,
        "stop after {t_stop} s, closed form {t_want} ± {t_tol}"
    );
    sc.run_tracks(&[&brake], 300);
    meter.sample(&sc.world.bodies[body]);
    assert_eq!(f(meter.distance().unwrap()), s, "latched distance changed");
    let b = &sc.world.bodies[body];
    assert!(f(b.velocity.x).abs() <= 1e-6 && f(b.velocity.z).abs() <= 1e-6);
    let h = ride_height(g);
    let theta = M * mu * g * h / (4.0 * K_SPRING * 1.2 * 1.2);
    let recoil = f(b.position.z) - z_latch;
    assert!(
        recoil.abs() <= h * theta,
        "moved {recoil} m after the latch, recoil bound {}",
        h * theta
    );
}

/// The meter measures the horizontal path, not the vertical motion: a body
/// moved straight up 1 m and then 3 m along x and 4 m along z reads 5 m only
/// after the horizontal move; with `rest_speed = 0` it latches exactly when
/// the horizontal speed is 0.
#[test]
fn stopping_distance_meter_is_horizontal_path_length() {
    let g = 9.81;
    let mut b = chassis(0.0, 0.0, g);
    b.velocity = v3(0.0, 0.0, 1.0);
    let up = Vec3Fix::UNIT_Y;
    let mut meter = StoppingDistanceMeter::new(&b, up, Fix128::ZERO);
    b.position = b.position + v3(0.0, 1.0, 0.0);
    assert!(!meter.sample(&b), "still moving horizontally");
    assert_eq!(meter.distance(), None);
    assert_eq!(f(meter.travelled()), 0.0, "vertical motion is not travel");
    b.position = b.position + v3(3.0, 0.0, 4.0);
    b.velocity = v3(0.0, -2.0, 0.0);
    assert!(meter.sample(&b), "horizontal speed 0 ⇒ stopped");
    assert!((f(meter.distance().unwrap()) - 5.0).abs() <= 1e-15);
    assert_eq!(meter.frames_to_stop(), Some(2));
}

// ---------------------------------------------------------------------------
// 3. Multi-vehicle runner
// ---------------------------------------------------------------------------

fn track_a() -> Vec<DriverInput> {
    (0..300)
        .map(|k| {
            if k < 120 {
                input(0.8, 0.0, 0.3)
            } else {
                input(0.0, 0.6, -0.2)
            }
        })
        .collect()
}

fn track_b() -> Vec<DriverInput> {
    (0..300)
        .map(|k| {
            if k < 60 {
                input(0.0, 0.0, 0.0)
            } else {
                input(0.0, 1.0, 0.5)
            }
        })
        .collect()
}

/// Two cars 50 m apart (no contact in the world, roads probe per wheel) run
/// together; each also runs alone in its own world. Every frame of 300, both
/// orders of insertion (A then B, B then A) reproduce the single-car runs
/// bit for bit.
#[test]
fn separated_cars_together_equal_each_alone_bitwise() {
    let cfg = config();
    let (pa, pb) = ((0.0, 0.0), (50.0, 0.0));
    let (ta, tb) = (track_a(), track_b());
    let mut alone_a = scenario(&[pa], &cfg);
    let mut alone_b = scenario(&[pb], &cfg);
    let mut ab = scenario(&[pa, pb], &cfg);
    let mut ba = scenario(&[pb, pa], &cfg);
    set_speed(&mut alone_b, 0, 15.0);
    set_speed(&mut ab, 1, 15.0);
    set_speed(&mut ba, 0, 15.0);
    for k in 0..300 {
        alone_a.step(&[ta[k]]);
        alone_b.step(&[tb[k]]);
        ab.step(&[ta[k], tb[k]]);
        ba.step(&[tb[k], ta[k]]);
        let (a, b) = (car_state(&alone_a, 0), car_state(&alone_b, 0));
        assert_eq!(car_state(&ab, 0), a, "frame {k}: A in (A, B)");
        assert_eq!(car_state(&ab, 1), b, "frame {k}: B in (A, B)");
        assert_eq!(car_state(&ba, 1), a, "frame {k}: A in (B, A)");
        assert_eq!(car_state(&ba, 0), b, "frame {k}: B in (B, A)");
    }
    // The tracks really moved the cars (not a trivially equal rest state).
    assert!(f(car_state(&alone_a, 0).position.z) > 0.5);
    assert!(f(car_state(&alone_b, 0).position.z) > 5.0);
    assert!(
        f(car_state(&alone_a, 0).rotation.y).abs() > 1e-3,
        "A turned"
    );
}

/// `run_tracks` and `step_with` drive the same frames as `step`: a track run
/// equals frame-by-frame `step`, and a closed-loop `step_with` controller
/// that reproduces the track's inputs equals it too.
#[test]
fn run_tracks_and_step_with_equal_step() {
    let cfg = config();
    let (ta, tb) = (track_a(), track_b());
    let mut by_step = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    let mut by_tracks = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    let mut by_fn = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    for k in 0..300 {
        by_step.step(&[ta[k], tb[k]]);
    }
    by_tracks.run_tracks(&[&ta, &tb], 300);
    by_fn.step_with(300, |frame, i, _veh, _body| {
        let k = frame as usize;
        if i == 0 {
            ta[k]
        } else {
            tb[k]
        }
    });
    assert_eq!(by_tracks.frame, 300);
    assert_eq!(all_states(&by_tracks), all_states(&by_step));
    assert_eq!(all_states(&by_fn), all_states(&by_step));
}

/// The runner passes its wind and its `time` to every vehicle. A car hanging
/// in the air (no tyre contact, aero `C_dA = 0.66`, `ρ = 1.225`) in the
/// storm preset `w(t) = 20 + 5 sin(2π · 1.5 · t)` along `+x` gains in one
/// frame `Δv_x = ½ ρ C_dA w² dt / M` (drag on the relative air speed, car at
/// rest): `w = 20` at `t = 0`, `w = 25` at `t = 1/6 s` (gust peak).
/// Relative tolerance 1e-9 (Fix128 `sin` and products).
#[test]
fn wind_and_time_reach_every_vehicle() {
    let mut cfg = config();
    cfg.aero.drag_area = fx(0.66);
    for (t, w) in [(Fix128::ZERO, 20.0), (Fix128::from_ratio(1, 6), 25.0)] {
        let mut sc = empty_scenario();
        sc.wind = Some(WindZone::storm(ZoneShape::Aabb {
            min: v3(-1.0e4, -1.0e4, -1.0e4),
            max: v3(1.0e4, 1.0e4, 1.0e4),
        }));
        sc.time = t;
        let g = g_of(&sc.world);
        for x in [0.0, 30.0] {
            let mut body = chassis(x, 0.0, g);
            body.position.y = fx(50.0);
            sc.add_vehicle(DynamicVehicle::new(cfg.clone()), body);
        }
        sc.step(&[]);
        let want = 0.5 * 1.225 * 0.66 * w * w * DT / M;
        for i in 0..2 {
            let got = f(sc.chassis(i).unwrap().velocity.x);
            assert!(
                (got - want).abs() <= 1e-9 * want,
                "t = {}, car {i}: Δv_x {got}, closed form {want}",
                f(t)
            );
            assert_eq!(sc.vehicles[i].vehicle.grounded_wheels(), 0);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Lossless replay
// ---------------------------------------------------------------------------

/// Record 300 frames of two cars (closed-loop inputs that depend on the state)
/// after a 120-frame settle that is not recorded; serialize, deserialize,
/// restore into a freshly built scenario (no settle) and replay frame by
/// frame: every car state is bit-equal to the original on every frame.
/// A gusting wind (force depends on `time`) and a car in second gear make
/// the recorded time and gear part of the state that must come back.
#[test]
fn replay_round_trip_is_bit_exact() {
    let mut cfg = config();
    cfg.aero.drag_area = fx(0.66);
    let build = || {
        let mut sc = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
        sc.wind = Some(WindZone::storm(ZoneShape::Aabb {
            min: v3(-1.0e4, -1.0e4, -1.0e4),
            max: v3(1.0e4, 1.0e4, 1.0e4),
        }));
        sc
    };
    let mut sc = build();
    sc.run_tracks(&[], 120);
    set_speed(&mut sc, 0, 18.0);
    set_speed(&mut sc, 1, 8.0);
    sc.vehicles[1].vehicle.config.powertrain.shift_up();
    // One frame so that the recorded engine speed is off idle (car 0).
    sc.run_tracks(&[], 1);
    assert!(sc.vehicles[0].vehicle.engine_rpm > fx(1000.0));
    let initial = all_states(&sc);
    sc.start_recording();
    let mut trajectory = Vec::new();
    for _ in 0..300 {
        sc.step_with(1, |frame, i, veh, body| {
            let v = DynamicVehicle::forward_speed(body);
            let steer = if frame % 50 < 25 { 0.2 } else { -0.15 };
            if i == 0 {
                // Rear car: brake harder when faster than 12 m/s.
                if v > fx(12.0) {
                    input(0.0, 0.7, steer)
                } else {
                    input(0.5, 0.0, steer)
                }
            } else if veh.engine_rpm > fx(2000.0) {
                input(0.2, 0.0, 0.0)
            } else {
                input(0.9, 0.0, 0.1)
            }
        });
        trajectory.push(all_states(&sc));
    }
    let rec = sc.stop_recording().expect("recording");
    assert!(!sc.is_recording());
    assert_eq!(rec.frame_count(), 300);
    assert_eq!(rec.vehicle_count(), 2);
    let bytes = rec.to_bytes();
    assert_eq!(&bytes[0..4], &RECORDING_MAGIC);
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), RECORDING_VERSION);
    let back = Recording::from_bytes(&bytes).expect("decode");
    assert_eq!(back, rec, "decode(encode(r)) == r");
    assert_eq!(back.to_bytes(), bytes, "encode is canonical");

    let mut fresh = build();
    back.restore(&mut fresh).expect("restore");
    assert_eq!(fresh.frame, 121);
    assert_eq!(
        all_states(&fresh),
        initial,
        "restore = state at record start"
    );
    for (k, want) in trajectory.iter().enumerate() {
        fresh.step(back.inputs(k).expect("frame in range"));
        assert_eq!(&all_states(&fresh), want, "frame {k}");
    }
    assert_eq!(fresh.world.serialize_state(), sc.world.serialize_state());

    // `replay` = restore + every frame.
    let mut fresh2 = build();
    back.replay(&mut fresh2).expect("replay");
    assert_eq!(all_states(&fresh2), all_states(&sc));
    assert_eq!(fresh2.frame, sc.frame);
    assert_eq!(fresh2.time, sc.time);

    // The motion was not trivial.
    let s0 = &trajectory[0][0];
    let s1 = &trajectory[299][0];
    assert!(f(s1.position.z - s0.position.z) > 10.0);
}

/// Detection power: flipping one bit of one input on one frame, or replaying
/// without restoring the recorded initial state, ends in a different state.
#[test]
fn replay_detects_one_bit_input_change_and_missing_restore() {
    let cfg = config();
    let mut sc = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
    sc.run_tracks(&[], 60);
    set_speed(&mut sc, 0, 12.0);
    sc.start_recording();
    sc.run_tracks(&[&track_a(), &track_b()], 300);
    let rec = sc.stop_recording().unwrap();
    let want = all_states(&sc);

    let mut edited = rec.clone();
    let inp = edited.input_mut(100, 0).expect("frame 100, car 0");
    inp.steering.lo ^= 1 << 40;
    let mut a = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
    edited.replay(&mut a).unwrap();
    assert_ne!(
        all_states(&a),
        want,
        "one input bit changed, same end state"
    );

    // Same bit flipped in the byte stream: decode → replay also differs.
    let mut bytes = rec.to_bytes();
    let mut b = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
    Recording::from_bytes(&bytes)
        .unwrap()
        .replay(&mut b)
        .unwrap();
    assert_eq!(all_states(&b), want, "unedited bytes replay exactly");
    let last = bytes.len() - 1;
    bytes[last - 3] ^= 0x01; // bit 32 of the last frame's last steering input
    let mut c = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
    Recording::from_bytes(&bytes)
        .unwrap()
        .replay(&mut c)
        .unwrap();
    assert_ne!(
        car_state(&c, 1).wheels,
        want[1].wheels,
        "edited last input changed nothing"
    );

    // Inputs alone, without the recorded initial state, do not reproduce.
    let mut d = scenario(&[(0.0, 0.0), (0.0, 40.0)], &cfg);
    for k in 0..rec.frame_count() {
        d.step(rec.inputs(k).unwrap());
    }
    assert_ne!(all_states(&d), want);
}

// ---------------------------------------------------------------------------
// 5. Degenerate inputs
// ---------------------------------------------------------------------------

/// No vehicles: `step` advances world, frame and time; a recording with zero
/// vehicles round-trips and replays; `inputs(k)` is an empty slice.
#[test]
fn zero_vehicles() {
    let mut sc = empty_scenario();
    sc.start_recording();
    sc.step(&[input(1.0, 0.0, 0.0)]);
    sc.run_tracks(&[], 4);
    assert_eq!(sc.frame, 5);
    assert_eq!(sc.time, dt() * Fix128::from_int(5));
    let rec = sc.stop_recording().unwrap();
    assert_eq!(rec.vehicle_count(), 0);
    assert_eq!(rec.frame_count(), 5);
    assert_eq!(rec.inputs(4), Some(&[][..]));
    assert_eq!(rec.inputs(5), None);
    let back = Recording::from_bytes(&rec.to_bytes()).unwrap();
    assert_eq!(back, rec);
    let mut fresh = empty_scenario();
    back.replay(&mut fresh).unwrap();
    assert_eq!(fresh.frame, 5);
    assert_eq!(fresh.time, sc.time);
}

/// Short input lists: a track shorter than the frame count, a missing track
/// and a `step` slice shorter than the vehicle count all mean
/// `DriverInput::default()` (pedals released, wheel straight) for the missing
/// entries, bit-equal to passing the defaults explicitly; the recording
/// stores the defaults that were applied. Extra entries are ignored.
#[test]
fn short_tracks_mean_released_inputs() {
    let cfg = config();
    let short = vec![input(1.0, 0.0, 0.4); 10];
    let mut padded = short.clone();
    padded.resize(30, DriverInput::default());
    let rest = vec![DriverInput::default(); 30];

    let mut a = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    a.start_recording();
    a.run_tracks(&[&short], 30);
    let mut b = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    b.run_tracks(&[&padded, &rest], 30);
    assert_eq!(all_states(&a), all_states(&b));
    let rec = a.stop_recording().unwrap();
    assert_eq!(rec.inputs(9).unwrap()[0], short[9]);
    assert_eq!(rec.inputs(10).unwrap(), &[DriverInput::default(); 2][..]);

    let mut c = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    let mut d = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    for _ in 0..30 {
        c.step(&[input(0.3, 0.0, 0.1)]);
        d.step(&[
            input(0.3, 0.0, 0.1),
            DriverInput::default(),
            input(1.0, 1.0, 1.0),
        ]);
    }
    let mut e = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    for _ in 0..30 {
        e.step(&[input(0.3, 0.0, 0.1), DriverInput::default()]);
    }
    assert_eq!(all_states(&c), all_states(&e));
    assert_eq!(all_states(&d), all_states(&e), "extra entries ignored");

    // A vehicle that had an input and is then left out of the slice is
    // released, not held at its previous input.
    let mut held = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    let mut released = scenario(&[(0.0, 0.0), (50.0, 0.0)], &cfg);
    held.step(&[input(0.3, 0.0, 0.1), input(1.0, 0.0, 0.5)]);
    released.step(&[input(0.3, 0.0, 0.1), input(1.0, 0.0, 0.5)]);
    for _ in 0..30 {
        held.step(&[input(0.3, 0.0, 0.1)]);
        released.step(&[input(0.3, 0.0, 0.1), DriverInput::default()]);
    }
    assert_eq!(held.vehicles[1].vehicle.input, DriverInput::default());
    assert_eq!(all_states(&held), all_states(&released));
}

/// `dt ≤ 0`: `step` changes nothing (no frame, no time, nothing recorded).
#[test]
fn non_positive_dt_is_a_no_op() {
    let cfg = config();
    let mut sc = scenario(&[(0.0, 0.0)], &cfg);
    sc.dt = Fix128::ZERO;
    sc.start_recording();
    let before = all_states(&sc);
    sc.step(&[input(1.0, 0.0, 0.0)]);
    assert_eq!(sc.frame, 0);
    assert_eq!(sc.time, Fix128::ZERO);
    assert_eq!(all_states(&sc), before);
    assert_eq!(sc.stop_recording().unwrap().frame_count(), 0);
}

fn small_recording() -> Recording {
    let cfg = config();
    let mut sc = scenario(&[(0.0, 0.0)], &cfg);
    sc.start_recording();
    sc.run_tracks(&[&track_a()], 3);
    sc.stop_recording().unwrap()
}

/// Decoding rejects, with the documented error and never a panic:
/// a different magic (`BadMagic`), another version
/// (`UnsupportedVersion(v)`), every strict prefix (`Truncated`), trailing
/// bytes (`TrailingBytes`), a flag byte that is neither 0 nor 1
/// (`InvalidFlag`), and frame counts too large for the bytes (`Truncated`)
/// or for `usize` arithmetic (`Oversized`).
#[test]
fn decode_rejects_malformed_bytes() {
    let bytes = small_recording().to_bytes();
    let mut m = bytes.clone();
    m[0] ^= 0xFF;
    assert_eq!(Recording::from_bytes(&m), Err(ReplayError::BadMagic));
    let mut v = bytes.clone();
    let other = RECORDING_VERSION + 1;
    v[4..6].copy_from_slice(&other.to_le_bytes());
    assert_eq!(
        Recording::from_bytes(&v),
        Err(ReplayError::UnsupportedVersion(other))
    );
    for len in 0..bytes.len() {
        assert_eq!(
            Recording::from_bytes(&bytes[..len]),
            Err(ReplayError::Truncated),
            "prefix of {len} bytes"
        );
    }
    // A frame count whose input table cannot be in the remaining bytes is
    // `Truncated` before anything is allocated: `2^57 + 1` frames × 64 bytes
    // fits `usize` on 64-bit targets but exceeds `isize::MAX`, so allocating
    // first would abort with a capacity overflow. The count sits right
    // before the 3 recorded frames of 1 vehicle (3 × 64 bytes).
    let frames_at = bytes.len() - 3 * 64 - 8;
    assert_eq!(
        u64::from_le_bytes(bytes[frames_at..frames_at + 8].try_into().unwrap()),
        3
    );
    let mut huge = bytes.clone();
    huge[frames_at..frames_at + 8].copy_from_slice(&((1u64 << 57) + 1).to_le_bytes());
    let want = if cfg!(target_pointer_width = "64") {
        ReplayError::Truncated
    } else {
        ReplayError::Oversized
    };
    assert_eq!(Recording::from_bytes(&huge), Err(want));
    // A count whose byte size overflows `usize` is `Oversized`.
    huge[frames_at..frames_at + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(Recording::from_bytes(&huge), Err(ReplayError::Oversized));
    let mut t = bytes.clone();
    t.push(0);
    assert_eq!(Recording::from_bytes(&t), Err(ReplayError::TrailingBytes));
    // The first wheel's `grounded` flag: header 8 + dt 16 + time 16 +
    // start frame 8 + world-blob length 4 + blob + vehicle count 4 + body 8 +
    // engine rpm 16 + gear 8 + input 64 + wheel count 4.
    let blob_len = u32::from_le_bytes(bytes[48..52].try_into().unwrap()) as usize;
    let flag = 52 + blob_len + 4 + 8 + 16 + 8 + 64 + 4;
    assert!(bytes[flag] <= 1, "offset {flag} is a flag byte");
    let mut fl = bytes.clone();
    fl[flag] = 2;
    assert_eq!(Recording::from_bytes(&fl), Err(ReplayError::InvalidFlag));
}

/// Restoring into a scenario of another shape fails with the documented
/// error and leaves the scenario untouched: different vehicle count,
/// different chassis body index, different wheel count, a world the
/// recorded world blob does not fit.
#[test]
fn restore_rejects_mismatched_scenarios() {
    let cfg = config();
    let rec = small_recording();
    let mut two = scenario(&[(0.0, 0.0), (5.0, 0.0)], &cfg);
    let before = all_states(&two);
    assert_eq!(
        rec.restore(&mut two),
        Err(ReplayError::VehicleCountMismatch {
            recorded: 1,
            scenario: 2
        })
    );
    assert_eq!(all_states(&two), before);

    let mut shifted = empty_scenario();
    let g = g_of(&shifted.world);
    shifted.world.add_body(chassis(9.0, 9.0, g));
    shifted.add_vehicle(DynamicVehicle::new(cfg.clone()), chassis(0.0, 0.0, g));
    assert_eq!(
        rec.restore(&mut shifted),
        Err(ReplayError::BodyMismatch {
            vehicle: 0,
            recorded: 0,
            scenario: 1
        })
    );

    let mut three_wheels = cfg.clone();
    three_wheels.base.wheels.truncate(3);
    let mut sc3 = scenario(&[(0.0, 0.0)], &three_wheels);
    assert_eq!(
        rec.restore(&mut sc3),
        Err(ReplayError::WheelCountMismatch {
            vehicle: 0,
            recorded: 4,
            scenario: 3
        })
    );

    let mut extra_body = scenario(&[(0.0, 0.0)], &cfg);
    let g = g_of(&extra_body.world);
    extra_body.world.add_body(chassis(9.0, 9.0, g));
    let before = extra_body.world.serialize_state();
    assert_eq!(
        rec.restore(&mut extra_body),
        Err(ReplayError::WorldStateRejected)
    );
    assert_eq!(extra_body.world.serialize_state(), before);
}
