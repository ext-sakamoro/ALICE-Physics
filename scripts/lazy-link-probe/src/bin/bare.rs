//! A world with a body that is stepped and digested, and nothing else: no continuous collision,
//! no cloth. scripts/lazy_link_check.py reads which functions it links.

use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};
use std::hint::black_box;

fn main() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let at = Vec3Fix::new(Fix128::ZERO, black_box(Fix128::from_int(3)), Fix128::ZERO);
    w.add_body(RigidBody::new(at, Fix128::ONE));
    for _ in 0..black_box(4) {
        w.step(Fix128::from_ratio(1, 60));
    }
    black_box(w.state_digest());
}
