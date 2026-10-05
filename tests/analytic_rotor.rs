//! Oracles for `rotor::Rotor` and the hover momentum-theory functions.
//!
//! # Where the expected values come from
//!
//! - Static propeller / rotor coefficients: `T = C_T ρ n² D⁴`,
//!   `Q = C_Q ρ n² D⁵` (n in rev/s; McCormick, *Aerodynamics, Aeronautics and
//!   Flight Mechanics*, ch. 6).
//! - Momentum theory in hover: `v_i = √(T / (2 ρ A))`, `P = T v_i`
//!   (Leishman, *Principles of Helicopter Aerodynamics*, §2.3).
//! - Reaction torque on the body: `−s Q â` (Newton's third law on the motor).
//! - Free rotation of a sphere-inertia body under constant torque about a
//!   principal axis: `ω(t) = τ t / I`.
//!
//! Expected values are evaluated with `f64` here and compared through
//! `to_f64()`. Tolerance `1e-9` relative for the closed forms (a handful of
//! `Fix128` multiplications plus one `sqrt`, exact floor at 2⁻⁶⁴).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `powi` / `sqrt` compute closed-form references outside the crate,
// not simulation state (same convention as the other analytic tests).
#![allow(clippy::disallowed_methods)]

use std::f64::consts::PI;

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rotor::{
    hover_induced_velocity_m_s, ideal_hover_power_w, Rotor, RotorError, RotorParams, RotorSpin,
};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const TOL: f64 = 1e-9;
const RHO: f64 = 1.225;
const D: f64 = 0.3;
const CT: f64 = 0.1;
const CQ: f64 = 0.005;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = if expected.abs() < 1e-12 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    };
    assert!(
        err <= tol,
        "{label}: {actual} vs {expected} (err {err} > {tol})"
    );
}

fn params(spin: RotorSpin) -> RotorParams {
    RotorParams {
        diameter_m: fx(D),
        thrust_coefficient: fx(CT),
        torque_coefficient: fx(CQ),
        thrust_axis_local: Vec3Fix::UNIT_Y,
        hub_position_local: Vec3Fix::ZERO,
        spin,
    }
}

fn rotor(spin: RotorSpin) -> Rotor {
    Rotor::new(params(spin)).expect("valid rotor")
}

#[test]
fn thrust_and_torque_follow_the_coefficient_laws() {
    let r = rotor(RotorSpin::RightHanded);
    for n in [10.0_f64, 50.0, 123.4] {
        let t = r.thrust_n(fx(RHO), fx(n)).unwrap().to_f64();
        let q = r.shaft_torque_nm(fx(RHO), fx(n)).unwrap().to_f64();
        rel_close("T", t, CT * RHO * n * n * D.powi(4), TOL);
        rel_close("Q", q, CQ * RHO * n * n * D.powi(5), TOL);
    }
    // n × 2 → T × 4 and Q × 4.
    let t1 = r.thrust_n(fx(RHO), fx(40.0)).unwrap().to_f64();
    let t2 = r.thrust_n(fx(RHO), fx(80.0)).unwrap().to_f64();
    let q1 = r.shaft_torque_nm(fx(RHO), fx(40.0)).unwrap().to_f64();
    let q2 = r.shaft_torque_nm(fx(RHO), fx(80.0)).unwrap().to_f64();
    rel_close("T(2n)/T(n)", t2 / t1, 4.0, TOL);
    rel_close("Q(2n)/Q(n)", q2 / q1, 4.0, TOL);
    // Density scales linearly (thin air at altitude gives less thrust).
    let thin = r.thrust_n(fx(RHO / 2.0), fx(40.0)).unwrap().to_f64();
    rel_close("T(ρ/2)", thin, t1 / 2.0, TOL);
}

#[test]
fn disk_area_induced_velocity_and_ideal_power() {
    // oracle: A = π D²/4; v_i = √(T/(2ρA)); P = T v_i.
    let r = rotor(RotorSpin::RightHanded);
    let a = PI * D * D / 4.0;
    rel_close("A", r.disk_area_m2().to_f64(), a, TOL);
    let t = 19.6;
    let vi = (t / (2.0 * RHO * a)).sqrt();
    rel_close(
        "v_i",
        hover_induced_velocity_m_s(fx(t), fx(RHO), fx(a))
            .unwrap()
            .to_f64(),
        vi,
        TOL,
    );
    rel_close(
        "P",
        ideal_hover_power_w(fx(t), fx(RHO), fx(a)).unwrap().to_f64(),
        t * vi,
        TOL,
    );
    // P ∝ T^(3/2): doubling thrust multiplies power by 2√2.
    let p1 = ideal_hover_power_w(fx(t), fx(RHO), fx(a)).unwrap().to_f64();
    let p2 = ideal_hover_power_w(fx(2.0 * t), fx(RHO), fx(a))
        .unwrap()
        .to_f64();
    rel_close("P(2T)/P(T)", p2 / p1, 2.0 * 2.0_f64.sqrt(), TOL);
}

#[test]
fn speed_for_thrust_inverts_the_thrust_law() {
    // oracle: n = √(T / (C_T ρ D⁴)).
    let r = rotor(RotorSpin::LeftHanded);
    let t = 20.0;
    let n = (t / (CT * RHO * D.powi(4))).sqrt();
    rel_close(
        "n",
        r.speed_for_thrust(fx(t), fx(RHO)).unwrap().to_f64(),
        n,
        TOL,
    );
}

#[test]
fn load_directions_and_reaction_torque_sign() {
    // Thrust along the body-rotated axis; reaction torque −s Q â; thrust
    // moment r × T â for a hub off the COM.
    let n = 60.0_f64;
    let t = CT * RHO * n * n * D.powi(4);
    let q = CQ * RHO * n * n * D.powi(5);
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);

    let rh = rotor(RotorSpin::RightHanded)
        .load(&body, fx(RHO), fx(n))
        .unwrap();
    rel_close("F_y", rh.force.y.to_f64(), t, TOL);
    rel_close("τ_y (RH)", rh.reaction_torque.y.to_f64(), -q, TOL);
    let lh = rotor(RotorSpin::LeftHanded)
        .load(&body, fx(RHO), fx(n))
        .unwrap();
    rel_close("τ_y (LH)", lh.reaction_torque.y.to_f64(), q, TOL);
    assert_eq!(rh.force, lh.force);

    // Hub at (0.2, 0, 0), axis +y: r × F = (0, 0, 0.2 T).
    let mut p = params(RotorSpin::RightHanded);
    p.hub_position_local = v3(0.2, 0.0, 0.0);
    let off = Rotor::new(p).unwrap().load(&body, fx(RHO), fx(n)).unwrap();
    rel_close("τ_z", off.torque.z.to_f64(), 0.2 * t, TOL);
    rel_close("τ_y", off.torque.y.to_f64(), -q, TOL);
    assert_eq!(off.application_point, v3(0.2, 0.0, 0.0));

    // `apply` carries both moments: Δω = I⁻¹ (r × T ŷ + τ_reaction) dt
    // (sphere inertia I = 0.4 m, identity rotation).
    let mut spun = body;
    let dt = 0.01;
    Rotor::new(p)
        .unwrap()
        .apply(&mut spun, fx(RHO), fx(n), fx(dt))
        .unwrap();
    rel_close(
        "Δω_z (r × F)",
        spun.angular_velocity.z.to_f64(),
        0.2 * t * dt / 0.4,
        1e-9,
    );
    rel_close(
        "Δω_y (reaction)",
        spun.angular_velocity.y.to_f64(),
        -q * dt / 0.4,
        1e-9,
    );
    rel_close("Δv_y", spun.velocity.y.to_f64(), t * dt, 1e-9);

    // Body rolled 90° about +z: the axis +y becomes −x.
    let mut rolled = body;
    rolled.set_rotation(QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI));
    let l = rotor(RotorSpin::RightHanded)
        .load(&rolled, fx(RHO), fx(n))
        .unwrap();
    rel_close("F_x rolled", l.force.x.to_f64(), -t, 1e-9);
    assert!(l.force.y.to_f64().abs() < 1e-9 * t);
    rel_close("τ_x rolled", l.reaction_torque.x.to_f64(), q, 1e-9);
}

fn config_substeps() -> usize {
    SolverConfig::default().substeps
}

fn hover_world() -> (PhysicsWorld, usize, f64) {
    let config = SolverConfig {
        damping: Fix128::ONE,
        ..SolverConfig::default()
    };
    let g = -config.gravity.y.to_f64();
    let mut world = PhysicsWorld::new(config);
    let idx = world.add_body(RigidBody::new(v3(0.0, 50.0, 0.0), fx(1.5)));
    (world, idx, g)
}

/// Hover run shared by the physical-expectation and characterization tests:
/// `frames` frames of `Rotor::apply` + `PhysicsWorld::step` at the hover
/// speed `n = √(m g / (C_T ρ D⁴))`. Returns (final body, g, Q).
fn run_hover(frames: u32) -> (RigidBody, f64, f64) {
    let (mut world, idx, g) = hover_world();
    let mass = 1.5;
    let n = (mass * g / (CT * RHO * D.powi(4))).sqrt();
    let q = CQ * RHO * n * n * D.powi(5);
    let r = rotor(RotorSpin::RightHanded);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..frames {
        let b = world.get_body_mut(idx).unwrap();
        r.apply(b, fx(RHO), fx(n), dt).unwrap();
        world.step(dt);
    }
    (*world.get_body(idx).unwrap(), g, q)
}

const HOVER_MASS: f64 = 1.5;

#[test]
#[ignore = "src gap: frame 先頭の外力 impulse と substep 重力の splitting (world 側に substep 内で外力を掛ける経路が無い)"]
fn world_hover_holds_altitude() {
    // Physical expectation: T = m g holds the altitude. Tolerance at the
    // rounding level: n carries the f64 → Fix128 rounding of √(…) (~1e-15
    // relative), a residual acceleration ~1e-14 g, ~1e-12 m over 10 s.
    // Measured today: +0.7292 m over 10 s (the splitting below), red.
    let (b, _, _) = run_hover(600);
    let drift = (b.position.y.to_f64() - 50.0).abs();
    assert!(drift < 1e-6, "altitude drift {drift} m");
}

#[test]
#[ignore = "src gap: 回転の積分で角速度が substep ごとに ω ← (2/h) sin(ωh/2) と目減りする (率 ω³h/24、h は substep 幅)"]
fn world_reaction_torque_spin_rate_matches_free_rotation() {
    // Physical expectation: the reaction torque −Q ŷ spins a sphere-inertia
    // body (I = 0.4 m) at ω_y(t) = −Q t / I. Measured today after 10 s:
    // −3.7386 vs −3.75 rad/s (0.30 % low); with 1 substep the loss grows
    // to 24 % at 15 rad/s, reproduced by the per-substep map ω ← (2/h) sin(ωh/2),
    // i.e. dω/dt = τ/I − ω³h/24.
    let (b, _, q) = run_hover(600);
    rel_close(
        "ω_y",
        b.angular_velocity.y.to_f64(),
        -q * 10.0 / (0.4 * HOVER_MASS),
        1e-9,
    );
}

#[test]
fn world_hover_characterization_of_the_current_discretization() {
    // Characterization of the current world discretization, NOT the physical
    // expectation (that is `world_hover_holds_altitude`, ignored above):
    //
    // - T = m g: the vertical velocity at every frame boundary stays 0 (the
    //   thrust impulse T dt cancels the gravity impulse g dt exactly).
    // - The thrust impulse lands at the start of the frame while gravity is
    //   spread over the s substeps (the same split as PhysicsWorld force
    //   fields), so within a frame v_y falls from g dt to 0 and the body
    //   rises by g dt² (s − 1)/(2 s) per frame
    //   (Σ_{i=1..s} (g dt − i g dt/s) dt/s). Linear in time, not quadratic.
    let frames = 600;
    let (b, g, _) = run_hover(frames);
    assert!(
        b.velocity.y.to_f64().abs() < 1e-9,
        "v_y {}",
        b.velocity.y.to_f64()
    );
    let s = config_substeps() as f64;
    let dt_f = 1.0 / 60.0;
    let split = f64::from(frames) * g * dt_f * dt_f * (s - 1.0) / (2.0 * s);
    let rise = b.position.y.to_f64() - 50.0;
    assert!(
        (rise - split).abs() < 1e-6,
        "altitude change {rise} m vs splitting {split} m"
    );
}

#[test]
fn world_reaction_torque_spins_the_body_backwards() {
    // Sign and first-frame magnitude of the reaction torque through the
    // world: after one frame ω_y = −Q dt / I (the ω³h/24 loss is ~1e-10
    // relative at this speed, far below 1e-8). Over 10 s the body keeps
    // spinning left-handed about +y only (the thrust axis stays vertical).
    let (b1, _, q) = run_hover(1);
    rel_close(
        "ω_y after 1 frame",
        b1.angular_velocity.y.to_f64(),
        -q / 60.0 / (0.4 * HOVER_MASS),
        1e-8,
    );
    let (b, _, _) = run_hover(600);
    assert!(b.angular_velocity.y < Fix128::ZERO);
    assert!(b.angular_velocity.x.to_f64().abs() < 1e-12);
    assert!(b.angular_velocity.z.to_f64().abs() < 1e-12);
}

#[test]
fn world_rotor_below_hover_speed_sinks_at_the_net_acceleration() {
    // Control for the hover test: 95 % of the hover speed gives
    // T = 0.9025 m g, so v_y(t) = −(g − T/m) t exactly (constant force,
    // the per-frame velocity update is exact).
    let (mut world, idx, g) = hover_world();
    let mass = 1.5;
    let n = 0.95 * (mass * g / (CT * RHO * D.powi(4))).sqrt();
    let t = CT * RHO * n * n * D.powi(4);
    let r = rotor(RotorSpin::RightHanded);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        let b = world.get_body_mut(idx).unwrap();
        r.apply(b, fx(RHO), fx(n), dt).unwrap();
        world.step(dt);
    }
    let b = world.get_body(idx).unwrap();
    rel_close("v_y", b.velocity.y.to_f64(), -(g - t / mass) * 2.0, 1e-9);
}

#[test]
fn degenerate_inputs() {
    let r = rotor(RotorSpin::RightHanded);
    // n = 0 and ρ = 0: zero thrust and torque.
    assert_eq!(r.thrust_n(fx(RHO), Fix128::ZERO), Ok(Fix128::ZERO));
    assert_eq!(r.shaft_torque_nm(fx(RHO), Fix128::ZERO), Ok(Fix128::ZERO));
    assert_eq!(r.thrust_n(Fix128::ZERO, fx(50.0)), Ok(Fix128::ZERO));
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let idle = r.load(&body, fx(RHO), Fix128::ZERO).unwrap();
    assert_eq!(idle.force, Vec3Fix::ZERO);
    assert_eq!(idle.torque, Vec3Fix::ZERO);
    // Negative n / ρ are errors and `apply` leaves the body unchanged.
    assert_eq!(
        r.thrust_n(fx(RHO), fx(-1.0)),
        Err(RotorError::NegativeRotationSpeed)
    );
    assert_eq!(
        r.shaft_torque_nm(fx(-1.0), fx(1.0)),
        Err(RotorError::NegativeAirDensity)
    );
    let mut b = body;
    assert_eq!(
        r.apply(&mut b, fx(RHO), fx(-5.0), fx(0.01)),
        Err(RotorError::NegativeRotationSpeed)
    );
    assert_eq!(b, body);
    // speed_for_thrust
    assert_eq!(r.speed_for_thrust(Fix128::ZERO, fx(RHO)), Ok(Fix128::ZERO));
    assert_eq!(
        r.speed_for_thrust(fx(-1.0), fx(RHO)),
        Err(RotorError::NegativeThrust)
    );
    assert_eq!(
        r.speed_for_thrust(fx(1.0), fx(-1.0)),
        Err(RotorError::NegativeAirDensity)
    );
    assert_eq!(r.speed_for_thrust(fx(1.0), Fix128::ZERO), Ok(Fix128::ZERO));
    // Momentum theory
    let a = fx(0.07);
    assert_eq!(
        hover_induced_velocity_m_s(Fix128::ZERO, fx(RHO), a),
        Ok(Fix128::ZERO)
    );
    assert_eq!(
        ideal_hover_power_w(Fix128::ZERO, fx(RHO), a),
        Ok(Fix128::ZERO)
    );
    assert_eq!(
        hover_induced_velocity_m_s(fx(-1.0), fx(RHO), a),
        Err(RotorError::NegativeThrust)
    );
    assert_eq!(
        hover_induced_velocity_m_s(fx(1.0), fx(-1.0), a),
        Err(RotorError::NegativeAirDensity)
    );
    assert_eq!(
        hover_induced_velocity_m_s(fx(1.0), fx(RHO), Fix128::ZERO),
        Err(RotorError::NonPositiveDiskArea)
    );
    assert_eq!(
        ideal_hover_power_w(fx(1.0), fx(RHO), fx(-0.1)),
        Err(RotorError::NonPositiveDiskArea)
    );
    assert_eq!(
        hover_induced_velocity_m_s(fx(1.0), Fix128::ZERO, a),
        Ok(Fix128::ZERO)
    );
    // Constructor
    let base = params(RotorSpin::RightHanded);
    for (p, e) in [
        (
            RotorParams {
                diameter_m: Fix128::ZERO,
                ..base
            },
            RotorError::NonPositiveDiameter,
        ),
        (
            RotorParams {
                diameter_m: fx(-0.1),
                ..base
            },
            RotorError::NonPositiveDiameter,
        ),
        (
            RotorParams {
                thrust_coefficient: fx(-0.1),
                ..base
            },
            RotorError::NegativeThrustCoefficient,
        ),
        (
            RotorParams {
                torque_coefficient: fx(-0.1),
                ..base
            },
            RotorError::NegativeTorqueCoefficient,
        ),
        (
            RotorParams {
                thrust_axis_local: Vec3Fix::ZERO,
                ..base
            },
            RotorError::DegenerateAxis,
        ),
    ] {
        assert_eq!(Rotor::new(p), Err(e));
    }
    // A non-unit axis is normalised: thrust magnitude is T, not T·|axis|.
    let long = Rotor::new(RotorParams {
        thrust_axis_local: v3(0.0, 5.0, 0.0),
        ..base
    })
    .unwrap();
    let l = long.load(&body, fx(RHO), fx(30.0)).unwrap();
    rel_close(
        "|F| with |axis| = 5",
        l.force.y.to_f64(),
        CT * RHO * 900.0 * D.powi(4),
        TOL,
    );
    assert_eq!(long.params().thrust_axis_local, v3(0.0, 5.0, 0.0));
}
