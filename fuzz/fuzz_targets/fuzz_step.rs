#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use alice_physics::{PhysicsWorld, PhysicsConfig, RigidBody, Fix128, Vec3Fix};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Number of bodies to add (capped)
    body_count: u8,
    /// Position components (i16 to keep values reasonable)
    positions: Vec<(i16, i16, i16)>,
    /// Mass numerator (> 0)
    masses: Vec<u16>,
    /// Number of simulation steps (capped)
    step_count: u8,
}

// Fuzz the physics world: add random bodies and step.
// Must never panic regardless of input.
fuzz_target!(|input: FuzzInput| {
    let config = PhysicsConfig::default();
    let mut world = PhysicsWorld::new(config);

    let body_count = (input.body_count as usize).min(16);
    let dt = Fix128::from_ratio(1, 60);

    for i in 0..body_count {
        let (px, py, pz) = input.positions.get(i).copied().unwrap_or((0, 0, 0));
        let mass_raw = input.masses.get(i).copied().unwrap_or(1);
        // Ensure mass > 0
        let mass = if mass_raw == 0 { 1u16 } else { mass_raw };

        let body = RigidBody::new_dynamic(
            Vec3Fix::from_int(px as i64, py as i64, pz as i64),
            Fix128::from_ratio(mass as i64, 1),
        );
        world.add_body(body);
    }

    let steps = (input.step_count as usize).min(32);
    for _ in 0..steps {
        world.step(dt);
    }
});
