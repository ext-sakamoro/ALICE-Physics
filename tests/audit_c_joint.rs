//! Audit oracles for `joint` (self-referential joint `BallJoint(a, a)`).
//!
//! A world refuses a self-joint: `add_joint` panics, and one pushed into
//! `joints` directly makes `try_step` return `StepError::Joint` with the body
//! bit-unchanged (and `step` panic). The free function `solve_joints` takes
//! any joints and skips a self-joint, leaving the body bit-unchanged.
//!
//! The unconstrained reference is semi-implicit Euler free fall from rest with
//! `n` substeps of `h`, `y = y0 - g h^2 n (n + 1) / 2` (exact in Fix128 for
//! `h = 1/512`).

#![cfg(feature = "std")]

use alice_physics::joint::{solve_joints, BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::world_participant::StepError;

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
        // pushed directly: `add_joint` refuses a self-joint
        w.joints.push(Joint::Ball(BallJoint::new(a, a, la, lb)));
    }
    (w, a)
}

/// `y0 - 10 h^2 n (n + 1) / 2` with `h = 1/512`.
fn free_fall_y(y0: i64, substeps_total: i64) -> Fix128 {
    Fix128::from_int(y0)
        - Fix128::from_ratio(10 * substeps_total * (substeps_total + 1) / 2, 512 * 512)
}

#[test]
#[should_panic(expected = "Joint joins body 0 to itself")]
fn add_joint_refuses_a_self_referential_joint() {
    let (mut w, a) = world(None);
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        a,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
}

/// A self-joint pushed into `joints` directly, with coincident or distinct
/// anchors: every `try_step` is refused with `StepError::Joint` and the body is
/// bit-unchanged, while the same world without the joint falls freely.
#[test]
// AUD-A-S34-010
fn a_self_referential_joint_refuses_the_step_and_leaves_the_body_unchanged() {
    for (la, lb) in [
        (Vec3Fix::ZERO, Vec3Fix::ZERO),
        (Vec3Fix::from_int(1, 0, 0), Vec3Fix::from_int(-1, 0, 0)),
    ] {
        let frames = 30_i64;
        let (mut with, a) = world(Some((la, lb)));
        let (mut without, b) = world(None);
        let start = (
            with.bodies[a].position,
            with.bodies[a].velocity,
            with.bodies[a].rotation,
        );
        for _ in 0..frames {
            assert_eq!(
                with.try_step(dt()),
                Err(StepError::Joint {
                    index: 0,
                    bodies: (a, a),
                    body_count: 1
                })
            );
            without.step(dt());
        }
        assert_eq!(
            (
                with.bodies[a].position,
                with.bodies[a].velocity,
                with.bodies[a].rotation
            ),
            start,
            "a refused step moved the body"
        );
        assert_eq!(without.bodies[b].position.y, free_fall_y(5, frames * 8));
    }
}

#[test]
#[should_panic(expected = "step refused: joint 0 joins body 0 to itself (body count = 1)")]
fn step_panics_on_a_self_referential_joint() {
    let (mut w, _) = world(Some((Vec3Fix::ZERO, Vec3Fix::ZERO)));
    w.step(dt());
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

/// Every joint kind between a body and itself leaves the body as it was (none
/// of the seven constrains anything a motion could satisfy), and a breakable
/// one does not break (it carries no force).
#[allow(deprecated)] // pins the separation-based solve_joints_breakable
#[test]
fn self_joints_of_every_kind_are_skipped_and_never_break() {
    #[allow(deprecated)] // the separation-based variant is pinned below
    use alice_physics::joint::solve_joints_breakable;
    use alice_physics::joint::{
        solve_joints, BallJoint, ConeTwistJoint, D6Joint, FixedJoint, HingeJoint, Joint,
        SliderJoint, SpringJoint,
    };
    use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
    use alice_physics::solver::RigidBody;
    let la = Vec3Fix::from_int(1, 0, 0);
    let lb = Vec3Fix::from_int(-1, 0, 0);
    let joints = [
        Joint::Ball(BallJoint::new(0, 0, la, lb)),
        Joint::Hinge(HingeJoint::new(
            0,
            0,
            la,
            lb,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Y,
        )),
        Joint::Fixed(FixedJoint::new(
            0,
            0,
            la,
            lb,
            QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::ONE),
        )),
        Joint::Slider(SliderJoint::new(0, 0, Vec3Fix::UNIT_X, la, lb)),
        Joint::Spring(SpringJoint::new(
            0,
            0,
            la,
            lb,
            Fix128::ONE,
            Fix128::from_int(10),
            Fix128::ONE,
        )),
        Joint::D6(D6Joint::new(0, 0, la, lb)),
        Joint::ConeTwist(ConeTwistJoint::new(
            0,
            0,
            la,
            lb,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Y,
        )),
    ];
    let mut start = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);
    start.velocity = Vec3Fix::from_int(1, -1, 2);
    start.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 3));
    let dt = Fix128::from_ratio(1, 60);
    for (k, j) in joints.iter().enumerate() {
        let mut bodies = vec![start];
        solve_joints(core::slice::from_ref(j), &mut bodies, dt);
        assert_eq!(bodies[0], start, "joint kind {k} moved its own body");
    }
    let tiny = Fix128::from_ratio(1, 1000);
    let breakable = [
        Joint::Ball(BallJoint::new(0, 0, la, lb).with_break_force(tiny)),
        Joint::Hinge(
            HingeJoint::new(0, 0, la, lb, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Y).with_break_force(tiny),
        ),
        Joint::Fixed(FixedJoint::new(0, 0, la, lb, QuatFix::IDENTITY).with_break_force(tiny)),
    ];
    let mut bodies = vec![start];
    let broken = solve_joints_breakable(&breakable, &mut bodies, dt);
    assert!(broken.is_empty(), "self joints broke: {broken:?}");
    assert_eq!(bodies[0], start);
}
