//! Oracles for damping as a rate per second.
//!
//! # What is measured
//!
//! A body in free flight (no gravity, nothing to touch) with a damping rate `c`
//! (the world's plus its own) keeps `v(t) = v₀·e^{−c·t}`: its velocity is
//! multiplied by `exp(−c·dt)` once per `step(dt)`, so after one second the loss is
//! the same at any step size. Before, damping was a retention per step, and one
//! second at 120 Hz with the old default 0.99 left `0.99¹²⁰ ≈ 0.299` of the speed
//! against `0.99⁶⁰ ≈ 0.547` at 60 Hz.
//!
//! The tolerance comes from the error of `Fix128::exp` (relative ≲ 1e-6 per call,
//! `math::tests::exp_matches_f64_reference`): `n` steps multiply `n` factors, each
//! within a relative 1e-6 of `e^{−c·dt}`, so the product is within a relative
//! `n·1e-6` (to first order) of `e^{−c·t}`, plus the 2⁻⁶⁴ rounding of each product.
//!
//! - **World rate** at `dt = 1/60, 1/120, 1/240`, linear and angular velocity.
//! - **Body rates** add to the world's, separately for linear and angular.
//! - **No damping** (rate 0, the default) leaves the velocity bit for bit.
//! - **Replay**: the same scene stepped twice ends in the same bits.
//!
//! Author: Moroya Sakamoto

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 e^(-c t) is the closed form the Fix128 product of per-step factors is checked against"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Free flight for one second at `1 / hz`: (linear speed, angular speed) at the end.
fn after_one_second(
    hz: u32,
    world_rate: f64,
    body: impl FnOnce(RigidBody) -> RigidBody,
) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: fx(world_rate),
        ..SolverConfig::default()
    });
    let mut b = body(RigidBody::new_dynamic(v3(0.0, 100.0, 0.0), Fix128::ONE));
    b.velocity = v3(1.0, 0.0, 0.0);
    b.angular_velocity = v3(0.0, 2.0, 0.0);
    let id = w.add_body(b);
    let dt = Fix128::ONE / Fix128::from_int(i64::from(hz));
    for _ in 0..hz {
        w.step(dt);
    }
    let b = w.get_body(id).expect("body");
    (
        b.velocity.length().to_f64(),
        b.angular_velocity.length().to_f64(),
    )
}

/// The relative tolerance after `n` steps (see the module doc).
fn tolerance(n: u32) -> f64 {
    f64::from(n) * 1e-6 + 1e-12
}

#[test]
fn a_world_rate_loses_the_same_share_per_second_at_every_step_size() {
    let c = 0.7;
    let want = (-c * 1.0_f64).exp();
    for hz in [60, 120, 240] {
        let (v, w) = after_one_second(hz, c, |b| b);
        let tol = tolerance(hz);
        assert!(
            (v / want - 1.0).abs() <= tol && (w / (2.0 * want) - 1.0).abs() <= tol,
            "{hz} Hz, rate {c}/s: after 1 s speed {v} (e^-c = {want}), spin {w} (2·e^-c = {}), relative tolerance {tol:.1e}",
            2.0 * want
        );
    }
}

#[test]
fn body_rates_add_to_the_world_rate() {
    let (world, lin, ang) = (0.3, 0.5, 1.1);
    for hz in [60, 240] {
        let (v, w) = after_one_second(hz, world, |b| {
            b.with_linear_damping_rate(fx(lin))
                .with_angular_damping_rate(fx(ang))
        });
        let (want_v, want_w) = ((-(world + lin)).exp(), 2.0 * (-(world + ang)).exp());
        let tol = tolerance(hz);
        assert!(
            (v / want_v - 1.0).abs() <= tol && (w / want_w - 1.0).abs() <= tol,
            "{hz} Hz: speed {v} want {want_v}, spin {w} want {want_w} (tolerance {tol:.1e})"
        );
    }
}

#[test]
fn no_damping_leaves_the_velocity_exactly() {
    let (v, w) = after_one_second(120, 0.0, |b| b);
    assert_eq!(
        (v, w),
        (1.0, 2.0),
        "rate 0 (the default) must not round the velocity"
    );
    assert!(
        SolverConfig::default().damping_rate.is_zero(),
        "the default world rate is 0"
    );
    let b = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    assert!(b.linear_damping_rate.is_zero() && b.angular_damping_rate.is_zero());
}

#[test]
fn a_damped_flight_replays_to_the_same_bits() {
    let run = || {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: v3(0.0, -10.0, 0.0),
            damping_rate: fx(0.4),
            ..SolverConfig::default()
        });
        let mut b = RigidBody::new_dynamic(v3(0.0, 100.0, 0.0), Fix128::ONE)
            .with_linear_damping_rate(fx(0.2));
        b.velocity = v3(3.0, 1.0, -2.0);
        let id = w.add_body(b);
        for _ in 0..240 {
            w.step(fx(1.0 / 120.0));
        }
        let b = w.get_body(id).expect("body");
        format!("{:?} {:?}", b.position, b.velocity)
    };
    assert_eq!(run(), run());
}
