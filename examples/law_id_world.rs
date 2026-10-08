//! Content identifiers of the stepping law.
//!
//! `PHYSICS_SEMANTICS_ID` names the arithmetic and step implementation of this
//! build; `PhysicsWorld::law_id` names the rules a world is configured with,
//! hashed together with it. The identifier depends on the rules, not on the
//! state: stepping the world or adding bodies leaves it unchanged, while
//! changing a rule (here the solver backend) gives a different one.
//!
//! Run with `cargo run --example law_id_world`.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::semantics::PHYSICS_SEMANTICS_ID;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() {
    println!("semantics id: {}", hex(&PHYSICS_SEMANTICS_ID));

    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let before = world.law_id(&PHYSICS_SEMANTICS_ID);
    println!("law id (XPBD): {}", hex(&before));

    // the state changes, the law does not
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    for _ in 0..60 {
        world.step(Fix128::from_ratio(1, 60));
    }
    let after = world.law_id(&PHYSICS_SEMANTICS_ID);
    assert_eq!(before, after, "stepping must not change the law id");
    println!("after one second of stepping: unchanged");

    // a different rule is a different law
    let tgs = PhysicsWorld::new(PhysicsConfig {
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    });
    let other = tgs.law_id(&PHYSICS_SEMANTICS_ID);
    assert_ne!(before, other, "the backend is part of the law");
    println!("law id (TGS):  {}", hex(&other));
}
