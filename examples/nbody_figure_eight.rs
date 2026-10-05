//! Mutual gravity from `nbody`: the Chenciner–Montgomery figure-eight orbit
//! of three equal masses integrated with the velocity-Verlet leapfrog, and a
//! circular pair of bodies run inside `PhysicsWorld` with
//! `DirectSum::step_world`.
//!
//! The figure-eight initial conditions and period are Simó's published
//! values (G = m = 1). The pair has `G = 1`, masses `1/2`, separation 1, so
//! its period is `2π`.
//!
//! ```bash
//! cargo run --release --example nbody_figure_eight
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::nbody::{kinetic_energy, total_momentum, DirectSum, VelocityVerlet};
use alice_physics::{PhysicsConfig, PhysicsWorld, RigidBody};

fn v(x: f64, y: f64) -> Vec3Fix {
    Vec3Fix::new(Fix128::from_f64(x), Fix128::from_f64(y), Fix128::ZERO)
}

fn main() {
    // 1. figure eight
    let gravity = DirectSum::new(Fix128::ONE, Fix128::ZERO).unwrap();
    let (x1, y1) = (0.970_004_36, -0.243_087_53);
    let (vx3, vy3) = (-0.932_407_37, -0.864_731_46);
    let start = vec![v(x1, y1), v(-x1, -y1), Vec3Fix::ZERO];
    let mut pos = start.clone();
    let mut vel = vec![
        v(-vx3 / 2.0, -vy3 / 2.0),
        v(-vx3 / 2.0, -vy3 / 2.0),
        v(vx3, vy3),
    ];
    let mass = vec![Fix128::ONE; 3];
    let energy = |pos: &[Vec3Fix], vel: &[Vec3Fix]| {
        kinetic_energy(vel, &mass).unwrap() + gravity.potential_energy(pos, &mass).unwrap()
    };
    let e0 = energy(&pos, &vel);
    let steps = 20_000;
    let dt = Fix128::from_f64(6.325_913_98) / Fix128::from_int(steps);
    let mut integrator = VelocityVerlet::new();
    for _ in 0..steps {
        integrator
            .step(&gravity, &mut pos, &mut vel, &mass, dt)
            .unwrap();
    }
    let mut acc = Vec::new();
    gravity.accelerations(&pos, &mass, &mut acc).unwrap();
    println!(
        "figure eight (G = {}, ε = {}): after one period",
        gravity.gravitational_constant(),
        gravity.softening()
    );
    for k in 0..3 {
        println!(
            "  body {k}: |x(T) − x(0)| = {:.3e}, |a| = {:.4}",
            (pos[k] - start[k]).length().to_f64(),
            acc[k].length().to_f64()
        );
    }
    let e1 = energy(&pos, &vel);
    println!(
        "  energy {:.10} -> {:.10}, momentum |p| = {:.3e}",
        e0.to_f64(),
        e1.to_f64(),
        total_momentum(&vel, &mass).unwrap().length().to_f64()
    );

    // 2. a circular pair in PhysicsWorld (no uniform gravity, no damping)
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    let half = Fix128::from_ratio(1, 2);
    for sign in [1_i64, -1] {
        let s = Fix128::from_int(sign);
        let mut body =
            RigidBody::new_dynamic(Vec3Fix::new(half * s, Fix128::ZERO, Fix128::ZERO), half);
        body.velocity = Vec3Fix::new(Fix128::ZERO, half * s, Fix128::ZERO);
        world.add_body(body);
    }
    let frames = 1000;
    let dt = Fix128::TWO_PI / Fix128::from_int(frames);
    for _ in 0..frames {
        gravity.step_world(&mut world, dt);
    }
    let sep = world.bodies[0].position - world.bodies[1].position;
    let h = std::f64::consts::TAU / frames as f64;
    println!(
        "world pair after one period: separation ({:.6}, {:.6}), closure error {:.3e} (leapfrog 2πh²/3 = {:.3e})",
        sep.x.to_f64(),
        sep.y.to_f64(),
        (sep - Vec3Fix::UNIT_X).length().to_f64(),
        std::f64::consts::TAU * h * h / 3.0
    );

    // the first-order alternative: one kick at the start of the frame
    let mut bodies = vec![
        RigidBody::new_dynamic(Vec3Fix::new(half, Fix128::ZERO, Fix128::ZERO), half),
        RigidBody::new_dynamic(Vec3Fix::new(-half, Fix128::ZERO, Fix128::ZERO), half),
    ];
    gravity.kick_bodies(&mut bodies, dt);
    println!(
        "one frame-head kick of dt: v = {:.6e}",
        bodies[0].velocity.x.to_f64()
    );
}
