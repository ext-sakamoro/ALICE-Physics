//! World Auditor Observation Example
//!
//! Demonstrates the typed observation API (`PhysicsWorld::observe_body` /
//! `observe_bodies`) and the `reset_world()` contract: a Law / goal predicate
//! reads `BodyObservation` instead of parsing the
//! `serialize_state` blob or reading fields directly.
//!
//! ```bash
//! cargo run --example world_auditor_observation --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    // Gravity zero so the two bodies stay in contact frame over frame
    // (this is a demo of the observation API, not of gravity/contact
    // resolution — `tests/wm07_rollback_event_parity.rs` uses the same
    // "start already overlapping" idiom).
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Two overlapping spheres so a contact event exists to observe.
    let radius = Fix128::from_int(2);
    world.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), radius);
    world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        radius,
    );

    let dt = Fix128::from_ratio(1, 60);
    world.step(dt);

    println!("World Auditor Observation Example");
    println!("==================================");
    for obs in world.observe_bodies() {
        println!(
            "body {}: y={:.4} sleeping={} in_contact={}",
            obs.body_index,
            obs.position.y.to_f64(),
            obs.sleeping,
            obs.in_contact,
        );
    }

    // reset_world(): same contract as a goal/retreat predicate "start over"
    // action — every field (bodies, joints, overflow flag, islands) goes
    // back to PhysicsWorld::new's defaults.
    world.reset_world();
    println!();
    println!(
        "after reset_world(): bodies={} overflow_detected={}",
        world.body_count(),
        world.overflow_detected(),
    );
}
