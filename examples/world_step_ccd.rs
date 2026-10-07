//! Continuous collision in the world step: a fast sphere and a thin plate.
//!
//! A plate 1/32 m thick (faces at x = 5 ∓ 1/64) and a sphere of radius 1/4
//! fired at 640 m/s along +x. One step of 1/64 s with one substep moves it
//! 10 m, from x = 0 to x = 10: without the setting the discrete detection at
//! the substep's end sees nothing and the sphere is past the plate. With
//! `PhysicsWorld::set_continuous_collision(WorldCcdConfig::on())` the step
//! sweeps it, stops it on the near face at x = 5 − 1/64 − 1/4 = 4.734375 and
//! the contact's restitution (`e = 1/2`) sends it back at −320 m/s.
//!
//! ```bash
//! cargo run --release --example world_step_ccd
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::material::PhysicsMaterial;
use alice_physics::shape::Shape;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldCcdConfig};

fn run(ccd: WorldCcdConfig) -> (f64, f64) {
    let config = PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.set_continuous_collision(ccd);
    assert_eq!(world.continuous_collision(), ccd);

    let plate = world.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)));
    world.set_body_shape(
        plate,
        &Shape::Box {
            half_extents: Vec3Fix::new(
                Fix128::from_ratio(1, 64),
                Fix128::from_int(2),
                Fix128::from_int(2),
            ),
        },
    );
    let mut bullet = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    bullet.velocity = Vec3Fix::from_int(640, 0, 0);
    let bullet = world.add_body_with_radius(bullet, Fix128::from_ratio(1, 4));

    let bouncy = world.material_table.register(PhysicsMaterial::new(
        1,
        Fix128::ZERO,
        Fix128::from_ratio(1, 2),
    ));
    world.set_body_material(plate, bouncy);
    world.set_body_material(bullet, bouncy);

    world.step(Fix128::from_ratio(1, 64));
    let b = &world.bodies[bullet];
    (b.position.x.to_f64(), b.velocity.x.to_f64())
}

fn main() {
    let (x, v) = run(WorldCcdConfig::new());
    println!("[world_ccd] off: x = {x}, v = {v} (passed through the plate)");
    assert!(x > 5.0 + 1.0 / 64.0 + 0.25);

    let (x, v) = run(WorldCcdConfig::on());
    println!("[world_ccd] on:  x = {x}, v = {v} (closed form 4.734375, -320)");
    assert!((x - 4.734_375).abs() < 1e-9);
    assert!((v + 320.0).abs() < 1e-6);
}
