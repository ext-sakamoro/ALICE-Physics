//! A joint whose body index is past the last body.
//!
//! `PhysicsWorld::add_joint` refuses it (assert); `joints` is a public field,
//! so it can still be pushed directly. The documented behaviour is a panic in
//! the next step's joint solve (index out of bounds), before anything else
//! reads the joint: in particular before the island rebuild that follows a
//! joint break, which therefore needs no range check of its own.

use alice_physics::{BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn world() -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-8), Fix128::ZERO),
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(config);
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)));
    w
}

#[test]
#[should_panic(expected = "Joint body indices (0, 7) out of bounds (body count = 2)")]
fn add_joint_refuses_an_index_past_the_last_body() {
    let mut w = world();
    w.add_joint(Joint::Ball(BallJoint::new(
        0,
        7,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 7")]
fn a_dangling_joint_pushed_directly_panics_in_the_next_step() {
    let mut w = world();
    w.joints.push(Joint::Ball(BallJoint::new(
        0,
        7,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    w.step(Fix128::from_ratio(1, 64));
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 7")]
fn a_dangling_joint_panics_before_a_joint_break_rebuilds_the_islands() {
    // a breakable joint carrying 16 N with a 15 N limit breaks in this step;
    // the dangling joint behind it panics in the joint solve first
    let mut w = world();
    w.add_joint(Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::from_int(15)),
    ));
    w.joints.push(Joint::Ball(BallJoint::new(
        0,
        7,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    w.step(Fix128::from_ratio(1, 64));
}
