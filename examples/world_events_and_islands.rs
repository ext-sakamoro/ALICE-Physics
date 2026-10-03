//! World-level event queue and island partition:
//! `EventCollector::has_events` and `IslandManager::build_islands`.
//!
//! Both are reached through the public fields of `PhysicsWorld`
//! (`world.events`, `world.islands`), which `step()` drives every frame.
//!
//! ```bash
//! cargo run --release --example world_events_and_islands --features std
//! ```
//!
//! Expected output (hand derivation, see `tests/analytic_event_sleeping_wiring.rs`):
//! * two overlapping collision spheres: frame 1 has a `Begin` contact event, so
//!   `has_events() == true`; after draining it is `false`
//! * bodies {0,1} and {2,3} joined by ball joints, body 4 alone and static:
//!   three islands `[0,1]`, `[2,3]`, `[4]`, and only `[4]` is all-sleeping

use alice_physics::event::ContactEventType;
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let dt = Fix128::from_ratio(1, 60);

    // Event queue: two unit spheres whose centres are 1.5 apart overlap by 0.5.
    let mut world = PhysicsWorld::new(cfg);
    let r = Fix128::ONE;
    world.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE),
        r,
    );
    world.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        r,
    );
    println!("before step: has_events = {}", world.events.has_events());
    world.step(dt);
    println!("after step : has_events = {}", world.events.has_events());
    for e in world.events.contact_events() {
        println!("  contact ({}, {}) {:?}", e.body_a, e.body_b, e.event_type);
        assert_eq!(e.event_type, ContactEventType::Begin);
    }
    let drained = world.drain_contact_events();
    println!(
        "drained {} contact event(s); has_events = {}",
        drained.len(),
        world.events.has_events()
    );

    // Islands: joined pairs plus one static loner.
    let mut world = PhysicsWorld::new(cfg);
    for i in 0..4 {
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(i, 0, 0),
            Fix128::ONE,
        ));
    }
    world.add_body(RigidBody::new_static(Vec3Fix::from_int(10, 0, 0)));
    for (a, b) in [(0usize, 1usize), (2, 3)] {
        world.add_joint(Joint::Ball(BallJoint::new(
            a,
            b,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::ZERO,
        )));
    }
    world.step(dt);
    for (k, island) in world.islands.build_islands().iter().enumerate() {
        println!(
            "island {k}: bodies {:?} all_sleeping {}",
            island.bodies, island.all_sleeping
        );
    }
}
