//! A joint whose body index is past the last body.
//!
//! `PhysicsWorld::add_joint` refuses it (assert); `joints` is a public field,
//! so it can still be pushed directly. The documented behaviour: a joint
//! between two different bodies panics in the next step's joint solve (index
//! out of bounds); a joint from a body to itself is skipped by the solve
//! (`joint::solve_one` returns before indexing) and by the island rebuild that
//! follows a joint break, so the world steps as if it were not there.

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

fn dangling(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO))
}

/// A joint holding the 2 kg body under `g = 8` (16 N) with a 15 N limit: it
/// breaks in the first step.
fn breaking() -> Joint {
    Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::from_int(15)),
    )
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn states(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

#[test]
#[should_panic(expected = "Joint body indices (0, 7) out of bounds (body count = 2)")]
fn add_joint_refuses_an_index_past_the_last_body() {
    let mut w = world();
    w.add_joint(dangling(0, 7));
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 7")]
fn a_dangling_joint_between_two_bodies_panics_in_the_next_step() {
    let mut w = world();
    w.joints.push(dangling(0, 7));
    w.step(dt());
}

#[test]
#[should_panic(expected = "index out of bounds: the len is 2 but the index is 7")]
fn a_dangling_joint_between_two_bodies_panics_before_a_joint_break() {
    let mut w = world();
    w.add_joint(breaking());
    w.joints.push(dangling(0, 7));
    w.step(dt());
}

#[test]
fn a_dangling_self_joint_alone_steps_as_if_absent() {
    let mut with = world();
    with.joints.push(dangling(7, 7));
    let mut without = world();
    for _ in 0..4 {
        with.step(dt());
        without.step(dt());
    }
    assert_eq!(states(&with), states(&without));
}

/// The island rebuild after a break walks every remaining joint; the self-joint
/// past the last body must not be unioned (it would index the island table out
/// of bounds). Indices 7 (far past) and 2 (exactly the body count, the edge of
/// the `<` comparison) both.
#[test]
fn a_dangling_self_joint_beside_a_breaking_joint_steps_as_if_absent() {
    for index in [7, 2] {
        let mut with = world();
        with.add_joint(breaking());
        with.joints.push(dangling(index, index));
        let mut without = world();
        without.add_joint(breaking());
        for _ in 0..4 {
            with.step(dt());
            without.step(dt());
        }
        assert!(
            without.joints.is_empty(),
            "the 15 N joint breaks under 16 N"
        );
        assert_eq!(
            with.joints.len(),
            1,
            "only the self-joint ({index}, {index}) is left"
        );
        assert_eq!(
            states(&with),
            states(&without),
            "self-joint ({index}, {index})"
        );
    }
}
