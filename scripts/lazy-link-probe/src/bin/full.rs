//! The same world with continuous collision turned on and a cloth added: the positive control of
//! scripts/lazy_link_check.py (the functions the bare world must not link are linked here).

use alice_physics_core::cloth::Cloth;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldCcdConfig};
use std::hint::black_box;

fn main() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let at = Vec3Fix::new(Fix128::ZERO, black_box(Fix128::from_int(3)), Fix128::ZERO);
    w.add_body(RigidBody::new(at, Fix128::ONE));
    w.set_continuous_collision(WorldCcdConfig::on());
    w.add_cloth(Cloth::new_grid(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE, 3, 3, Fix128::ONE));
    for _ in 0..black_box(4) {
        w.step(Fix128::from_ratio(1, 60));
    }
    black_box(w.state_digest());
}
