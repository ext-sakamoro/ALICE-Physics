//! The reaction force of a slider joint with an active translation limit and an
//! off-axis error, through `joint::solve_joints_with_reaction_forces`.
//!
//! Closed form: with both bodies dynamic, zero anchors and zero compliance, the
//! perpendicular row moves the bodies by `p = perp_dist / w` along the
//! perpendicular direction and the limit row by `c = error / w` along the axis
//! (`w = 1/m_a + 1/m_b`). The two directions are orthogonal in exact arithmetic,
//! so the reported force is `sqrt(perp_dist² + error²) / (w h²)`; the test
//! checks it to 1e-12 N.
//!
//! The axis `(1, 2, 3) / |(1, 2, 3)|` is not dyadic, so in fixed point the
//! perpendicular direction and the axis are orthogonal only up to rounding and
//! `|p + c|` differs from `|p − c|` in the low bits. The full bits of the force
//! are therefore pinned as well: flipping the sign of the perpendicular row or
//! of either limit correction changes them (joint.rs `solve_slider_joint`).
//! The bits were read from this implementation; the closed form above is the
//! independent check of the value.

use alice_physics::joint::{solve_joints_with_reaction_forces, Joint, SliderJoint};
use alice_physics::{Fix128, RigidBody, Vec3Fix};

fn h() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn axis() -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_int(3),
    )
    .normalize()
}

/// Orthogonal to (1, 2, 3) in exact arithmetic, length sqrt(1e-5)
fn perp() -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_ratio(3, 1000),
        Fix128::ZERO,
        -Fix128::from_ratio(1, 1000),
    )
}

fn limits() -> (Fix128, Fix128) {
    (Fix128::from_ratio(2, 100), Fix128::from_ratio(5, 100))
}

/// (reported force, closed form) for body B placed `along` the axis plus `perp()`
fn case(mass_a: i64, mass_b: i64, along: Fix128) -> (Fix128, f64) {
    let (min, max) = limits();
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(mass_a)),
        RigidBody::new(axis() * along + perp(), Fix128::from_int(mass_b)),
    ];
    let joint = Joint::Slider(
        SliderJoint::new(0, 1, axis(), Vec3Fix::ZERO, Vec3Fix::ZERO).with_limits(min, max),
    );
    let force = solve_joints_with_reaction_forces(&[joint], &mut bodies, h())[0];

    let w = 1.0 / mass_a as f64 + 1.0 / mass_b as f64;
    let perp_dist = 1e-5f64.sqrt();
    let a = along.to_f64();
    let error = if a < min.to_f64() {
        min.to_f64() - a
    } else {
        a - max.to_f64()
    };
    let h2 = (1.0f64 / 64.0) * (1.0f64 / 64.0);
    (
        force,
        (perp_dist * perp_dist + error * error).sqrt() / (w * h2),
    )
}

fn check(mass_a: i64, mass_b: i64, along: Fix128, bits: (i64, u64)) {
    let (force, closed_form) = case(mass_a, mass_b, along);
    assert!(
        (force.to_f64() - closed_form).abs() < 1e-12,
        "masses ({mass_a}, {mass_b}), along {}: {} vs closed form {closed_form}",
        along.to_f64(),
        force.to_f64()
    );
    assert_eq!(
        (force.hi, force.lo),
        bits,
        "masses ({mass_a}, {mass_b}), along {}",
        along.to_f64()
    );
}

#[test]
fn below_the_minimum_equal_masses() {
    check(
        1,
        1,
        Fix128::from_ratio(1, 100),
        (21, 0x7ac7_6836_31ce_5000),
    );
}

#[test]
fn above_the_maximum_equal_masses() {
    check(
        1,
        1,
        Fix128::from_ratio(7, 100),
        (41, 0x7805_dba4_d078_b000),
    );
}

#[test]
fn below_the_minimum_unequal_masses() {
    check(
        2,
        5,
        Fix128::from_ratio(1, 100),
        (61, 0x5ecc_052d_20c0_e000),
    );
}

#[test]
fn above_the_maximum_unequal_masses() {
    check(
        2,
        5,
        Fix128::from_ratio(7, 100),
        (118, 0x7b7e_738d_c168_9000),
    );
}
