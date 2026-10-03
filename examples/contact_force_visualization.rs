//! Contact forces for a debug view
//!
//! Two balls stacked on a static one. Every contact the solver holds carries a normal
//! force: the lower contact the weight of both balls, the upper contact the weight of
//! the top ball. `contact_forces` turns the solver's applied separation into those
//! forces, and `contact_arrows`, `contact_friction_arrows` and `contact_friction_cones`
//! are what a renderer draws from them (friction scaled by each contact's own
//! coefficient).
//!
//! ```bash
//! cargo run --example contact_force_visualization --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn main() {
    let g = 9.81;
    let (m1, m2) = (2.0, 3.0);
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -g, 0.0),
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    let radius = Fix128::from_f64(0.5);
    world.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), radius);
    world.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), Fix128::from_f64(m1)),
        radius,
    );
    world.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 2.0, 0.0), Fix128::from_f64(m2)),
        radius,
    );

    // Settle for 40 frames: a body at rest goes to sleep after about a minute, and a
    // sleeping body has no contacts to draw.
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..40 {
        world.step(dt);
    }

    let mut forces = world.contact_forces(dt);
    forces.sort_by_key(|(point, _, _)| point.y);
    for (point, normal, force) in &forces {
        println!(
            "contact at y = {:.3}: force {:.3} N along ({:.1}, {:.1}, {:.1})",
            point.y.to_f64(),
            force.to_f64(),
            normal.x.to_f64(),
            normal.y.to_f64(),
            normal.z.to_f64()
        );
    }
    println!(
        "expected: {:.3} N (both balls) and {:.3} N (the top ball)",
        (m1 + m2) * g,
        m2 * g
    );

    println!(
        "drawn: {} normal arrows, {} friction arrows, {} friction cones",
        world.contact_arrows(dt).len(),
        world.contact_friction_arrows(dt).len(),
        world.contact_friction_cones(dt).len()
    );
    for cone in world.contact_friction_cones(dt) {
        println!(
            "friction cone: half-angle {:.4} rad (atan of the friction coefficient), height {:.3} N",
            cone.half_angle.to_f64(),
            cone.height.to_f64()
        );
    }
}
