#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Fix128, Vec3Fix};

#[derive(Debug, Arbitrary)]
struct CollisionInput {
    /// Two bodies' positions (close together to force collision)
    x1: i8,
    y1: i8,
    x2: i8,
    y2: i8,
    /// Steps to run
    steps: u8,
}

// Fuzz collision detection by placing bodies close together.
// Must never panic even with overlapping bodies.
fuzz_target!(|input: CollisionInput| {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    // Place two bodies at potentially overlapping positions
    let body_a = RigidBody::new_dynamic(
        Vec3Fix::from_int(input.x1 as i64, input.y1 as i64, 0),
        Fix128::ONE,
    );
    let body_b = RigidBody::new_dynamic(
        Vec3Fix::from_int(input.x2 as i64, input.y2 as i64, 0),
        Fix128::ONE,
    );
    world.add_body(body_a);
    world.add_body(body_b);

    let dt = Fix128::from_ratio(1, 60);
    let steps = (input.steps as usize).min(64);
    for _ in 0..steps {
        world.step(dt);
    }
});
