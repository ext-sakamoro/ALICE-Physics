//! Degenerate-input oracles for `vehicle_dynamics::DynamicVehicle`.
//!
//! Each test states the correct outcome (state unchanged / free fall /
//! clamp to the boundary input / closed-form value) and asserts it; none
//! settles for "does not panic" (`Fix128` arithmetic wraps, so a no-panic
//! check would pass on overflow).
//!
//! Setup as in `analytic_vehicle_dynamics.rs`: `passenger_car()` with the
//! legacy wheel layout (`a = b = 1.2`, track 1.6, `k = 50000`,
//! `c = 4500`, rest 0.3, radius 0.3), wheel inertia 1 kg m², aero 0,
//! pinned flat 300 Nm powertrain (gears from 3.5, final drive 1, engine
//! braking 0), parallel steering; world from `PhysicsConfig::default()` with
//! damping 1; gravity read from `world.config.gravity`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic_friction::AnisotropicFriction;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::vehicle_dynamics::powertrain::{Differential, Powertrain, TorqueCurve};
use alice_physics::vehicle_dynamics::surface::{FlatGround, RoadCondition, Weather};
use alice_physics::vehicle_dynamics::{
    AeroConfig, DriverInput, DynamicVehicle, DynamicVehicleConfig, Environment,
};

const M: f64 = 1000.0;
const DT: f64 = 1.0 / 60.0;
const R_WHEEL: f64 = 0.3;
const REST: f64 = 0.3;
const K_SPRING: f64 = 50000.0;
const DAMPER: f64 = 4500.0;
const I_WHEEL: f64 = 1.0;
const ENGINE_TORQUE: f64 = 300.0;
const FIRST_GEAR: f64 = 3.5;

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

fn config() -> DynamicVehicleConfig {
    let mut c = DynamicVehicleConfig::passenger_car();
    assert_eq!(c.base.wheels.len(), 4, "passenger_car has four wheels");
    let pos = [(-0.8, 1.2), (0.8, 1.2), (-0.8, -1.2), (0.8, -1.2)];
    for (w, (x, z)) in c.base.wheels.iter_mut().zip(pos) {
        w.local_position = v3(x, -0.2, z);
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
    c.aero = AeroConfig {
        air_density: fx(1.225),
        drag_area: Fix128::ZERO,
        lift_area: Fix128::ZERO,
    };
    c.powertrain = Powertrain {
        curve: TorqueCurve {
            points: vec![
                (fx(800.0), fx(ENGINE_TORQUE)),
                (fx(7000.0), fx(ENGINE_TORQUE)),
            ],
        },
        idle_rpm: fx(800.0),
        max_rpm: fx(7000.0),
        engine_brake_per_rpm: Fix128::ZERO,
        gear_ratios: vec![fx(FIRST_GEAR), fx(2.2), fx(1.5), fx(1.1), fx(0.8)],
        final_drive: Fix128::ONE,
        current_gear: 0,
        differential: Differential::Open,
    };
    c.ackermann = false;
    c
}

fn condition() -> RoadCondition {
    RoadCondition {
        material: AnisotropicFriction {
            longitudinal_static: fx(1.0),
            longitudinal_kinetic: fx(0.8),
            transverse_static: fx(1.0),
            transverse_kinetic: fx(0.8),
            slip_threshold_m_s: fx(0.05),
        },
        weather: Weather::Dry,
        rolling_resistance: Fix128::ZERO,
    }
}

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.config.damping = Fix128::ONE;
    w
}

fn ground() -> FlatGround {
    FlatGround {
        height: Fix128::ZERO,
    }
}

/// Static ride height: `0.2 + r + rest (1 − m g / (4 k))`.
fn ride_height(m: f64, g: f64) -> f64 {
    0.2 + R_WHEEL + REST * (1.0 - m * g / (4.0 * K_SPRING))
}

struct Sim {
    world: PhysicsWorld,
    car: usize,
    veh: DynamicVehicle,
    road: FlatGround,
    cond: RoadCondition,
}

impl Sim {
    fn new(cfg: DynamicVehicleConfig, body: RigidBody) -> Self {
        let mut world = world();
        let car = world.add_body(body);
        Self {
            world,
            car,
            veh: DynamicVehicle::new(cfg),
            road: ground(),
            cond: condition(),
        }
    }

    fn update(&mut self, step: Fix128) {
        let env = Environment {
            condition: &self.cond,
            wind: None,
            time: Fix128::ZERO,
        };
        self.veh
            .update(&mut self.world.bodies[self.car], &self.road, &env, step);
    }

    fn frame(&mut self) {
        self.update(dt());
        self.world.step(dt());
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
        f(self.world.config.gravity.length())
    }
}

fn dynamic_body(m: f64, y: f64) -> RigidBody {
    let mut b = RigidBody::new_dynamic(v3(0.0, y, 0.0), fx(m));
    b.inv_inertia = v3(1.0 / (1.5 * m), 1.0 / (1.8 * m), 1.0 / (0.5 * m));
    b
}

/// Car on the road at its ride height, settled 2 s, then rolling at 10 m/s
/// along `+z` with the wheels at the rolling speed.
fn rolling_sim() -> Sim {
    let w = world();
    let g = f(w.config.gravity.length());
    let mut sim = Sim::new(config(), dynamic_body(M, ride_height(M, g)));
    sim.frames(120);
    sim.world.bodies[sim.car].velocity = v3(0.0, 0.0, 10.0);
    for w in &mut sim.veh.wheels {
        w.omega = fx(10.0 / R_WHEEL);
    }
    sim
}

/// `dt = 0`: every impulse is `F · 0`, every spin change `ω̇ · 0`, so the
/// chassis (position, velocity, angular velocity, rotation) and every wheel's
/// `omega` / `spin_angle` are bit-identical after the update, with throttle,
/// brake and steering all active.
#[test]
fn zero_dt_changes_nothing() {
    let mut sim = rolling_sim();
    sim.veh.input = DriverInput {
        throttle: Fix128::ONE,
        brake: fx(0.5),
        handbrake: Fix128::ZERO,
        steering: fx(0.3),
    };
    sim.frames(5);
    let before = *sim.body();
    let wheels: Vec<(Fix128, Fix128)> = sim
        .veh
        .wheels
        .iter()
        .map(|w| (w.omega, w.spin_angle))
        .collect();
    sim.update(Fix128::ZERO);
    let after = *sim.body();
    assert_eq!(after.position, before.position);
    assert_eq!(after.velocity, before.velocity);
    assert_eq!(after.angular_velocity, before.angular_velocity);
    assert_eq!(after.rotation, before.rotation);
    for (i, (w, (omega, angle))) in sim.veh.wheels.iter().zip(wheels).enumerate() {
        assert_eq!(w.omega, omega, "wheel {i} spin changed with dt = 0");
        assert_eq!(w.spin_angle, angle, "wheel {i} angle changed with dt = 0");
    }
}

/// Free-fall closed form for the world step with damping 1:
/// `v_y(N) = −g N dt`, horizontal velocity constant.
///
/// The update itself must leave the velocity bit-identical (no contact, no
/// force). Across `world.step` the velocity is rebuilt from positions each
/// substep, `v = (x − x_prev) / h` with `h = dt / S`, so Fix128 rounding
/// (fraction `ε = 2⁻⁶⁴`) enters as: `h` and `1/h` are rounded (relative
/// `ε / h` each) and `x` is rounded (absolute `ε`, i.e. `ε / h` in velocity).
/// Over `n = 60 S` substeps:
/// `|Δv| ≤ n ε (2 |v| + 2) / h` (2.7e-13 m/s at `|v| = 10`, `S = 8`; measured
/// 2e-14 horizontally). The vertical check adds the same bound to the
/// `f64` evaluation of `g N dt`.
fn assert_free_fall(sim: &mut Sim, label: &str) {
    let g = sim.g();
    let v0 = sim.body().velocity;
    for k in 1..=60 {
        let before = sim.body().velocity;
        sim.update(dt());
        assert_eq!(
            sim.body().velocity,
            before,
            "{label}: frame {k}: update changed the velocity of a car with no contact"
        );
        sim.world.step(dt());
    }
    let substeps = sim.world.config.substeps as f64;
    let h = DT / substeps;
    let eps = 2.0_f64.powi(-64);
    let n = 60.0 * substeps;
    let rounding = |v: f64| n * eps * (2.0 * v.abs() + 2.0) / h;
    let want = f(v0.y) - g * 60.0 * DT;
    let got = f(sim.body().velocity.y);
    let tol_y = rounding(want) + 1e-12;
    assert!(
        (got - want).abs() <= tol_y,
        "{label}: v_y {got}, free fall {want} ± {tol_y}"
    );
    for (axis, got, want) in [
        ("x", sim.body().velocity.x, v0.x),
        ("z", sim.body().velocity.z, v0.z),
    ] {
        let (got, want) = (f(got), f(want));
        let tol = rounding(want);
        assert!(
            (got - want).abs() <= tol,
            "{label}: v_{axis} {got} changed from {want} beyond the step rounding {tol}"
        );
    }
}

/// No wheels: no contact, no force (aero 0, no wind) whatever the inputs, so
/// `grounded_wheels() == 0`, each update leaves the velocity unchanged and the
/// chassis falls freely.
#[test]
fn zero_wheel_config_falls_freely() {
    let mut cfg = config();
    cfg.base.wheels.clear();
    let w = world();
    let g = f(w.config.gravity.length());
    let mut sim = Sim::new(cfg, dynamic_body(M, ride_height(M, g)));
    assert!(sim.veh.wheels.is_empty(), "wheel state follows the config");
    sim.veh.input = DriverInput {
        throttle: Fix128::ONE,
        brake: Fix128::ONE,
        handbrake: Fix128::ONE,
        steering: Fix128::ONE,
    };
    assert_free_fall(&mut sim, "no wheels");
    assert_eq!(sim.veh.grounded_wheels(), 0);
}

/// All wheels airborne (CG 50 m up, probe reach 0.6 m): no wheel grounded,
/// no tyre force, free fall of the chassis.
#[test]
fn airborne_car_falls_freely() {
    let mut sim = Sim::new(config(), dynamic_body(M, 50.0));
    sim.world.bodies[sim.car].velocity = v3(0.0, 0.0, 10.0);
    assert_free_fall(&mut sim, "airborne");
    assert_eq!(sim.veh.grounded_wheels(), 0);
    assert!(sim.veh.wheels.iter().all(|w| !w.grounded));
}

/// Airborne, full throttle from rest in first gear: each driven wheel gets
/// half the axle torque (open differential) and no tyre reaction, so after
/// one frame `ω = (T · 3.5 / 2) dt / I_w = 525 / 60 = 8.75 rad/s`; undriven
/// wheels stay at 0. Tolerance `1e-9` (Fix128 rounding).
#[test]
fn airborne_driven_wheels_spin_up_freely() {
    let mut sim = Sim::new(config(), dynamic_body(M, 50.0));
    sim.veh.input.throttle = Fix128::ONE;
    sim.frame();
    let want = ENGINE_TORQUE * FIRST_GEAR / 2.0 * DT / I_WHEEL;
    for (i, w) in sim.veh.wheels.iter().enumerate() {
        let driven = sim.veh.config.base.wheels[i].driven;
        let got = f(w.omega);
        if driven {
            assert!(
                (got - want).abs() <= 1e-9,
                "driven wheel {i}: ω {got}, closed form {want}"
            );
        } else {
            assert_eq!(w.omega, Fix128::ZERO, "undriven wheel {i} spun");
        }
    }
}

/// A static chassis is immovable: ten frames with every input active leave
/// it bit-identical.
#[test]
fn static_chassis_is_left_untouched() {
    let w = world();
    let g = f(w.config.gravity.length());
    let body = RigidBody::new_static(v3(0.0, ride_height(M, g), 0.0));
    let mut sim = Sim::new(config(), body);
    sim.veh.input = DriverInput {
        throttle: Fix128::ONE,
        brake: fx(0.5),
        handbrake: Fix128::ZERO,
        steering: Fix128::ONE,
    };
    sim.frames(10);
    assert_eq!(*sim.body(), body, "static chassis moved");
}

/// Static ride height for chassis mass `m` after 10 s at rest, mean over the
/// last 60 frames, against `0.2 + r + rest (1 − m g / (4 k))`.
///
/// Tolerance `g dt² + c g dt / (2 k_w)` with `k_w = k / rest`: the chassis
/// receives the suspension once per frame and gravity per substep, so its
/// position moves by up to `g dt²` within a frame, and the damper reads the
/// frame-end vertical velocity (up to `g dt / 2`), shifting the spring's
/// share by `c g dt / 2` per corner (2.25 mm here).
fn assert_ride_height(m: f64) {
    let w = world();
    let g = f(w.config.gravity.length());
    let y_ref = ride_height(m, g);
    let mut sim = Sim::new(config(), dynamic_body(m, y_ref));
    sim.frames(540);
    let mut y = 0.0;
    for _ in 0..60 {
        sim.frame();
        y += f(sim.body().position.y) / 60.0;
    }
    let k_w = K_SPRING / REST;
    let tol = g * DT * DT + DAMPER * g * DT / (2.0 * k_w);
    assert_eq!(
        sim.veh.grounded_wheels(),
        4,
        "m = {m}: all wheels on the road"
    );
    assert!(
        (y - y_ref).abs() <= tol,
        "m = {m}: ride height {y}, closed form {y_ref} ± {tol}"
    );
}

/// Heavy chassis, 19 t: static compression `m g / (4 k) = 0.95` (close to
/// bottoming, no bump stop), `ω_n dt = √(4 k_w / m) dt ≈ 0.10`.
#[test]
fn heavy_chassis_settles_at_closed_form_ride_height() {
    assert_ride_height(19000.0);
}

/// Light chassis at the stiff end of frame-coupled suspension:
/// `ω_n dt = √(4 k_w / m) dt = 0.5` gives `m = 4 k_w dt² / 0.25 = 740.7 kg`
/// (compression 0.037).
#[test]
fn light_chassis_settles_at_closed_form_ride_height() {
    let m = 4.0 * (K_SPRING / REST) * DT * DT / 0.25;
    assert_ride_height(m);
}

/// Run a rolling car 30 frames with `input` and return the final chassis and
/// wheel states.
fn run_with(
    input: DriverInput,
) -> (
    RigidBody,
    Vec<alice_physics::vehicle_dynamics::WheelDynamicsState>,
) {
    let mut sim = rolling_sim();
    sim.veh.input = input;
    sim.frames(30);
    (*sim.body(), sim.veh.wheels.clone())
}

/// Inputs outside their range behave exactly like the nearest bound:
/// throttle / brake / handbrake clamp to `[0, 1]`, steering to `[-1, 1]`.
/// Expected value: bit-identical chassis and wheel states to the run with the
/// boundary input.
#[test]
fn out_of_range_inputs_clamp_to_the_boundary() {
    let base = DriverInput::default();
    let cases = [
        (
            "throttle 3",
            DriverInput {
                throttle: fx(3.0),
                ..base
            },
            DriverInput {
                throttle: Fix128::ONE,
                ..base
            },
        ),
        (
            "throttle -1",
            DriverInput {
                throttle: fx(-1.0),
                ..base
            },
            DriverInput {
                throttle: Fix128::ZERO,
                ..base
            },
        ),
        (
            "brake 2",
            DriverInput {
                brake: fx(2.0),
                ..base
            },
            DriverInput {
                brake: Fix128::ONE,
                ..base
            },
        ),
        (
            "brake -1",
            DriverInput {
                brake: fx(-1.0),
                ..base
            },
            DriverInput {
                brake: Fix128::ZERO,
                ..base
            },
        ),
        (
            "handbrake 2",
            DriverInput {
                handbrake: fx(2.0),
                ..base
            },
            DriverInput {
                handbrake: Fix128::ONE,
                ..base
            },
        ),
        (
            "steering 5",
            DriverInput {
                steering: fx(5.0),
                ..base
            },
            DriverInput {
                steering: Fix128::ONE,
                ..base
            },
        ),
        (
            "steering -5",
            DriverInput {
                steering: fx(-5.0),
                ..base
            },
            DriverInput {
                steering: -Fix128::ONE,
                ..base
            },
        ),
    ];
    for (label, out, bound) in cases {
        let (b_out, w_out) = run_with(out);
        let (b_bound, w_bound) = run_with(bound);
        assert_eq!(
            b_out, b_bound,
            "{label}: chassis differs from the boundary input"
        );
        assert_eq!(
            w_out, w_bound,
            "{label}: wheel states differ from the boundary input"
        );
    }
}

/// Steering 5 (parallel steering): every steerable wheel sits at
/// `|steer_angle| = max_steer_angle = 0.5 rad`, rear wheels at 0.
#[test]
fn out_of_range_steering_stops_at_max_steer_angle() {
    let mut sim = rolling_sim();
    sim.veh.input.steering = fx(5.0);
    sim.frame();
    for (i, w) in sim.veh.wheels.iter().enumerate() {
        let max = sim.veh.config.base.wheels[i].max_steer_angle;
        assert_eq!(
            w.steer_angle.abs(),
            max,
            "wheel {i} steer angle {}",
            f(w.steer_angle)
        );
    }
}
