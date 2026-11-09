//! Body axis locks: a ball kept in the x-y plane, a slider on a vertical rail
//! and a turntable that only turns about z.
//!
//! The ball is thrown with a velocity out of the plane and a spin about every
//! axis; locked to the plane (z position, x and y rotations) it moves and turns
//! as a ball thrown in the plane. The printed z and the tilt of each body stay
//! exactly where they started.
//!
//! Run with `cargo run --example axis_lock_planar`.

use alice_physics_core::{AxisLock, Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn main() {
    let i = Fix128::from_int;
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-9.81), Fix128::ZERO),
        ..PhysicsConfig::default()
    });

    let mut ball = RigidBody::new_dynamic(Vec3Fix::new(i(0), i(10), i(2)), Fix128::ONE);
    ball.velocity = Vec3Fix::new(i(3), i(4), i(5));
    ball.angular_velocity = Vec3Fix::new(i(1), i(1), i(2));
    let ball = world.add_body(ball);
    let planar = AxisLock::LINEAR_Z | AxisLock::ANGULAR_X | AxisLock::ANGULAR_Y;
    world.set_body_axis_lock(ball, planar);

    let mut slider = RigidBody::new_dynamic(Vec3Fix::new(i(5), i(10), i(0)), Fix128::ONE);
    slider.velocity = Vec3Fix::new(i(2), i(0), i(-1));
    let slider = world.add_body(slider);
    world.set_body_axis_lock(
        slider,
        AxisLock::LINEAR_X | AxisLock::LINEAR_Z | AxisLock::ANGULAR,
    );

    let mut table = RigidBody::new_dynamic(Vec3Fix::new(i(-5), i(0), i(0)), Fix128::ONE);
    table.angular_velocity = Vec3Fix::new(i(1), i(1), i(1));
    let table = world.add_body(table);
    let turntable = AxisLock::LINEAR | AxisLock::ANGULAR_X | AxisLock::ANGULAR_Y;
    world.set_body_axis_lock(table, turntable);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }
    let b = &world.bodies[ball];
    println!(
        "ball: x {:.4}, y {:.4}, z {} (locked at 2), tilt ({}, {})",
        b.position.x.to_f64(),
        b.position.y.to_f64(),
        b.position.z.to_f64(),
        b.rotation.x.to_f64(),
        b.rotation.y.to_f64()
    );
    let s = &world.bodies[slider];
    println!(
        "slider: x {} z {} (locked at 5 and 0), y {:.4} (fell)",
        s.position.x.to_f64(),
        s.position.z.to_f64(),
        s.position.y.to_f64()
    );
    let t = &world.bodies[table];
    let lock = world.body_axis_lock(table).unwrap_or_default();
    println!(
        "turntable: turns about z: {}, holds its height: {}, spin {:.4} rad/s, tilt ({}, {}), position {:?}",
        !lock.contains(AxisLock::ANGULAR_Z),
        lock.contains(AxisLock::LINEAR_Y),
        t.angular_velocity.z.to_f64(),
        t.rotation.x.to_f64(),
        t.rotation.y.to_f64(),
        (
            t.position.x.to_f64(),
            t.position.y.to_f64(),
            t.position.z.to_f64()
        )
    );
}
