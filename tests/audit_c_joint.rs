//! Audit oracles for `joint` (self-referential joint): a `BallJoint(a, a)` must
//! leave the body exactly as an unconstrained body (closed-form free fall,
//! bit-equal to the same world without the joint, finite state).
//!
//! Expected values: semi-implicit Euler free fall from rest with `n` substeps of
//! `h`, `y = y0 - g h^2 n (n + 1) / 2` (exact in Fix128 for `h = 1/512`), and the
//! same world stepped without any joint.

#![cfg(feature = "std")]

use alice_physics::joint::{solve_joints, BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

/// `dt = 1/64`, 8 substeps: `h = 1/512` (exact).
fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// Default gravity and substeps, no frame damping so velocity carries exactly.
fn undamped() -> PhysicsConfig {
    let c = PhysicsConfig {
        damping: Fix128::ONE,
        ..PhysicsConfig::default()
    };
    assert_eq!(c.substeps, 8, "closed forms below assume 8 substeps");
    assert_eq!(c.gravity, Vec3Fix::from_int(0, -10, 0));
    c
}

fn world(with_self_joint: Option<(Vec3Fix, Vec3Fix)>) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(undamped());
    let a = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 5, 0),
        Fix128::ONE,
    ));
    if let Some((la, lb)) = with_self_joint {
        w.add_joint(Joint::Ball(BallJoint::new(a, a, la, lb)));
    }
    (w, a)
}

/// `y0 - 10 h^2 n (n + 1) / 2` with `h = 1/512`.
fn free_fall_y(y0: i64, substeps_total: i64) -> Fix128 {
    Fix128::from_int(y0)
        - Fix128::from_ratio(10 * substeps_total * (substeps_total + 1) / 2, 512 * 512)
}

fn finite(v: Vec3Fix) -> bool {
    [v.x, v.y, v.z].iter().all(|c| c.to_f64().is_finite())
}

/// The existing self-referential case (both local anchors at the centre): after
/// 60 frames the body is exactly where a free body falls and bit-equal to the world
/// without the joint, rotation untouched.
#[test]
fn self_referential_ball_joint_with_coincident_anchors_is_a_free_body() {
    let frames = 60_i64;
    let (mut with, a) = world(Some((Vec3Fix::ZERO, Vec3Fix::ZERO)));
    let (mut without, b) = world(None);
    for _ in 0..frames {
        with.step(dt());
        without.step(dt());
    }
    let p = with.bodies[a].position;
    assert!(finite(p) && finite(with.bodies[a].velocity));
    assert_eq!(p.x, Fix128::ZERO);
    assert_eq!(p.z, Fix128::ZERO);
    assert_eq!(p.y, free_fall_y(5, frames * 8), "y after {frames} frames");
    assert_eq!(with.bodies[a].position, without.bodies[b].position);
    assert_eq!(with.bodies[a].velocity, without.bodies[b].velocity);
    assert_eq!(with.bodies[a].rotation, QuatFix::IDENTITY);
    assert_eq!(with.bodies[a].angular_velocity, Vec3Fix::ZERO);
}

/// A body cannot be displaced relative to itself: with two different local anchors
/// on the same body the constraint is unsatisfiable by any motion, and the only
/// motion-free answer is to leave the body as a free body.
#[test]
#[ignore = "known defect: AUD-A-S34-010: `solve_ball_joint` with `BallJoint(a, a)` and distinct local anchors (1,0,0) / (-1,0,0) applies the correction to the same body as A and B and injects velocity: x = 36 after 1 frame and 28920 after 30 frames (dt 1/64, 8 substeps) instead of 0 for the free body"]
fn self_referential_ball_joint_with_distinct_anchors_is_a_free_body() {
    let frames = 30_i64;
    let la = Vec3Fix::from_int(1, 0, 0);
    let lb = Vec3Fix::from_int(-1, 0, 0);
    let (mut with, a) = world(Some((la, lb)));
    let (mut without, b) = world(None);
    for _ in 0..frames {
        with.step(dt());
        without.step(dt());
    }
    let p = with.bodies[a].position;
    assert!(
        finite(p) && finite(with.bodies[a].velocity),
        "non-finite state {p:?}"
    );
    assert_eq!(
        with.bodies[a].position, without.bodies[b].position,
        "position differs from the unconstrained body"
    );
    assert_eq!(with.bodies[a].rotation, without.bodies[b].rotation);
    assert_eq!(p.y, free_fall_y(5, frames * 8));
}

/// Direct solver call: one `solve_joints` pass with `BallJoint(0, 0)` and coincident
/// anchors leaves position and rotation bit-unchanged.
#[test]
fn solve_joints_with_a_self_ball_joint_does_not_move_the_body() {
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(3, -2, 7), Fix128::from_int(2));
    b.rotation = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_ratio(3, 5),
        Fix128::ZERO,
        Fix128::from_ratio(4, 5),
    );
    let before = b;
    let mut bodies = vec![b];
    let j = Joint::Ball(BallJoint::new(
        0,
        0,
        Vec3Fix::from_int(1, 2, 3),
        Vec3Fix::from_int(1, 2, 3),
    ));
    for _ in 0..5 {
        solve_joints(&[j], &mut bodies, dt());
    }
    assert_eq!(bodies[0].position, before.position);
    assert_eq!(bodies[0].rotation, before.rotation);
}
