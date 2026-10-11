//! The reaction force `|Σ λ · n| / h²` that
//! `joint::solve_joints_with_reaction_forces` reports, for every joint type
//! (Macklin, Müller, Chentanez 2016, eq. 10 and 18).
//!
//! Closed form shared by the rigid joints: a static anchor `A` at the origin
//! and a body `B` of mass `m = 2` that one gravity substep moved
//! `g h² = 8 / 64² = 1/512` below it. The positional row closes that gap with
//! `λ = (1/512) / (1/m) = 1/256`, a reaction of `λ / h² = 16 N = m g`. Every
//! quantity is dyadic, so the force is exactly 16. Only the positional rows
//! that close the gap act: the axes and frames are aligned, so the angular
//! rows of hinge / fixed / cone-twist / D6 see no error.
//!
//! The spring has its own closed form (XPBD spring row): with rest length 0,
//! stiffness `k`, no damping and the body at rest, the row is
//! `λ = k s h² / (1 + k h² w)`, a reaction of `k s / (1 + k h² w)`.
//!
//! Only the magnitude is reported. Where a joint adds to the impulse on one
//! line only and from zero, flipping that line's sign negates the impulse
//! exactly and leaves the magnitude unchanged in every bit
//! (tests/analytic_impulse_sign_exact.rs); those sign mutants are equivalent.
//! The slider adds a limit row after its perpendicular row, and the two are
//! orthogonal only up to fixed-point rounding, so its sign flips are not
//! equivalent: tests/analytic_slider_reaction_limit.rs pins them. The
//! magnitude below is what the rows' scaling must reproduce.

use alice_physics::joint::{
    solve_joints_with_reaction_forces, BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint,
    HingeJoint, Joint, SliderJoint, SpringJoint,
};
use alice_physics::{Fix128, QuatFix, RigidBody, Vec3Fix};

fn h() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// The static anchor and the 2 kg body one gravity substep below it
fn hung_bodies() -> Vec<RigidBody> {
    vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(
            Vec3Fix::new(Fix128::ZERO, -Fix128::from_ratio(1, 512), Fix128::ZERO),
            Fix128::from_int(2),
        ),
    ]
}

fn reaction(joint: Joint) -> Fix128 {
    let mut bodies = hung_bodies();
    let forces = solve_joints_with_reaction_forces(&[joint], &mut bodies, h());
    assert_eq!(forces.len(), 1);
    forces[0]
}

fn z() -> Vec3Fix {
    Vec3Fix::ZERO
}

#[test]
fn ball_joint_transmits_m_g() {
    let j = Joint::Ball(BallJoint::new(0, 1, z(), z()));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn hinge_joint_transmits_m_g() {
    let j = Joint::Hinge(HingeJoint::new(
        0,
        1,
        z(),
        z(),
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    ));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn fixed_joint_transmits_m_g() {
    let j = Joint::Fixed(FixedJoint::new(0, 1, z(), z(), QuatFix::IDENTITY));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn slider_joint_transmits_m_g_across_its_axis() {
    // the axis is x; the gap is along y, so the perpendicular row closes it
    let j = Joint::Slider(SliderJoint::new(0, 1, Vec3Fix::UNIT_X, z(), z()));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn cone_twist_joint_transmits_m_g() {
    let j = Joint::ConeTwist(ConeTwistJoint::new(
        0,
        1,
        z(),
        z(),
        Vec3Fix::UNIT_Y,
        Vec3Fix::UNIT_Y,
    ));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn d6_joint_with_locked_linear_axes_transmits_m_g() {
    let j = Joint::D6(D6Joint::new(0, 1, z(), z()).with_linear_motion(
        D6Motion::Locked,
        D6Motion::Locked,
        D6Motion::Locked,
    ));
    assert_eq!(reaction(j), Fix128::from_int(16));
}

#[test]
fn spring_joint_transmits_its_xpbd_row_force() {
    // k = 64, s = 1/512, w = 1/2, h = 1/64: k s = 1/8, k h² w = 1/128, so the
    // reaction is (1/8) / (129/128) = 16/129 N
    let j = Joint::Spring(SpringJoint::new(
        0,
        1,
        z(),
        z(),
        Fix128::ZERO,
        Fix128::from_int(64),
        Fix128::ZERO,
    ));
    let got = reaction(j);
    let want = Fix128::from_ratio(16, 129);
    let err = (got - want).abs();
    assert!(
        err < Fix128::from_ratio(1, 1 << 40),
        "spring reaction {got:?}, closed form 16/129 = {want:?}"
    );
}
