//! Oracle: `RigidBody::add_force` changes the velocity at once, by
//! `force · inv_mass · dt`, and nothing is accumulated for a later step.
//!
//! The doc once said "accumulated, applied during next step"; the closed form
//! here is what the code does and what the doc now says.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

#[test]
fn add_force_changes_the_velocity_at_once_and_leaves_nothing_for_the_step() {
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2));
    body.add_force(Vec3Fix::from_int(6, 0, 0), Fix128::from_ratio(1, 2));
    // v = F · (1/m) · dt = 6 · 0.5 · 0.5 = 1.5, now
    assert_eq!(body.velocity, Vec3Fix::new(Fix128::from_ratio(3, 2), Fix128::ZERO, Fix128::ZERO));
    // a step with no gravity keeps that velocity: no second application
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let i = w.add_body(body);
    w.step(Fix128::from_ratio(1, 60));
    assert_eq!(w.bodies[i].velocity.x, Fix128::from_ratio(3, 2), "the force was not applied again");
    // a static body takes no force
    let mut fixed = RigidBody::new_static(Vec3Fix::ZERO);
    fixed.add_force(Vec3Fix::from_int(6, 0, 0), Fix128::ONE);
    assert_eq!(fixed.velocity, Vec3Fix::ZERO);
}
