//! Basic 3D Physics Example
//!
//! Demonstrates creating a physics world, adding rigid bodies,
//! and stepping the simulation.
//!
//! ```bash
//! cargo run --example basic_physics --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    // Configure the physics world
    let config = PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(99, 100),
    };
    let mut world = PhysicsWorld::new(config);

    // Add a static floor at y=0
    let floor = RigidBody::new_static(Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(-1),
        Fix128::ZERO,
    ));
    world.add_body(floor);

    // Add a dynamic sphere at y=10
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(10), Fix128::ZERO),
        Fix128::ONE, // 1 kg
    );
    world.add_body(ball);

    // Add a second ball with builder pattern
    let ball2 = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(2), Fix128::from_int(15), Fix128::ZERO),
        Fix128::from_int(2), // 2 kg
    )
    .with_restitution(Fix128::from_ratio(8, 10))
    .with_linear_damping(Fix128::from_ratio(95, 100));
    world.add_body(ball2);

    println!("ALICE-Physics Basic Example");
    println!("===========================");
    println!("Bodies: {}", world.body_count());
    println!();

    // Simulate 2 seconds at 60 FPS
    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..120 {
        world.step(dt);

        if frame % 20 == 0 {
            let pos1 = world.bodies[1].position;
            let pos2 = world.bodies[2].position;
            println!(
                "Frame {:3}: ball1 y={:.4}  ball2 y={:.4}",
                frame,
                pos1.y.to_f64(),
                pos2.y.to_f64(),
            );
        }
    }

    println!();
    println!("Simulation complete (120 frames, 2 seconds).");
    println!("Debug: {:?}", world);
}
