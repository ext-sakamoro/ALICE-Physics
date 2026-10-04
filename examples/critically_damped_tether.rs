//! Critically Damped Tether Example
//!
//! Pulls bodies back to a target point with the critically damped law
//! `a = ω²·(target − x) − 2ω·v`, solved inside the world instead of by hand, using
//! existing pieces only:
//!
//! - a static (or kinematic) anchor body at the target,
//! - a [`DistanceConstraint`] of rest length 0 and compliance `1 / (m ω²)` (an isotropic
//!   linear spring of stiffness `m ω²`, solved implicitly by XPBD),
//! - the body's `linear_damping` set to the discrete critical value
//!   `d = 1 + 2a² − 2a·sqrt(1 + a²)`, `a = ω·dt`.
//!
//! With `substeps = 1`, `iterations = 1` and `SolverConfig::damping = 1`, one step maps
//! `(x, v)` to `x' = (x + dt·v) / (1 + a²)`, `v' = d·(x' − x) / dt`, whose two eigenvalues
//! coincide at `λ = 1 − a / sqrt(1 + a²) ∈ (0, 1)`: the discrete solution has the same
//! `(A + B·n)·λⁿ` form as the closed form `x(t) = (x0 + (v0 + ω x0) t)·e^(−ωt)`, never
//! overshoots from rest, and stays within `0.42·ω·dt·|x0|` of it. Each body carries its
//! own `ω` (the damping is per body), so tethers of different stiffness share one world.
//!
//! The default sleep rule (`|v| < 0.01` for 60 frames) freezes a body once the decay is
//! slow enough; this example switches it off to compare the whole trajectory. A body that
//! fell asleep has to be woken (`wake_body`) when its target moves.
//!
//! ```bash
//! cargo run --example critically_damped_tether --features std
//! ```

use alice_physics::det_math::exp64;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sleeping::SleepConfig;
use alice_physics::solver::{DistanceConstraint, PhysicsWorld, RigidBody, SolverConfig};

/// Discrete critical damping factor for one step of length `dt` and angular frequency `omega`.
fn critical_retention(omega: Fix128, dt: Fix128) -> Fix128 {
    let a = omega * dt;
    let a2 = a * a;
    Fix128::ONE + a2 + a2 - (a + a) * (Fix128::ONE + a2).sqrt()
}

/// Add a body of mass `mass` at `start`, tethered to a static anchor at `target`.
fn add_tether(
    world: &mut PhysicsWorld,
    target: Vec3Fix,
    start: Vec3Fix,
    mass: Fix128,
    omega: Fix128,
    dt: Fix128,
) -> usize {
    let anchor = world.add_body(RigidBody::new_static(target));
    let body = world.add_body(
        RigidBody::new_dynamic(start, mass).with_linear_damping(critical_retention(omega, dt)),
    );
    let stiffness = mass * omega * omega;
    world.add_distance_constraint(
        DistanceConstraint::new(anchor, body, Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ZERO)
            .with_compliance(Fix128::ONE / stiffness),
    );
    body
}

fn main() {
    let dt = Fix128::from_ratio(1, 60);
    let mut world = PhysicsWorld::new(SolverConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..SolverConfig::default()
    });
    world.set_sleep_config(SleepConfig {
        frames_to_sleep: u32::MAX,
        ..SleepConfig::default()
    });
    let target = Vec3Fix::ZERO;
    // two tethers with different stiffness in the same world
    let slow = add_tether(
        &mut world,
        target,
        Vec3Fix::from_int(1, 0, 0),
        Fix128::ONE,
        Fix128::from_int(4),
        dt,
    );
    let fast = add_tether(
        &mut world,
        target,
        Vec3Fix::from_int(0, 1, 0),
        Fix128::from_int(2),
        Fix128::from_int(12),
        dt,
    );

    let mut worst = [0.0_f64; 2];
    for n in 1..=180 {
        world.step(dt);
        let t = n as f64 / 60.0;
        for (k, (body, omega, axis)) in [(slow, 4.0, 0), (fast, 12.0, 1)].into_iter().enumerate() {
            let (x, y, _) = world.bodies[body].position.to_f32();
            let got = f64::from(if axis == 0 { x } else { y });
            let exact = (1.0 + omega * t) * exp64(-omega * t);
            worst[k] = worst[k].max((got - exact).abs());
        }
    }
    println!(
        "max |x - closed form|: omega 4 -> {:.5}, omega 12 -> {:.5}",
        worst[0], worst[1]
    );
    println!(
        "bounds 0.42*omega*dt:   omega 4 -> {:.5}, omega 12 -> {:.5}",
        0.42 * 4.0 / 60.0,
        0.42 * 12.0 / 60.0
    );
}
