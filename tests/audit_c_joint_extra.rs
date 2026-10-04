//! Audit oracles for `joint_extra` dispatch and kinematic couplings: values after
//! `solve_extra_joints` for pulley / gear / rack-and-pinion / weld / mouse.
//!
//! Expected values: the linearised XPBD projection of each documented constraint
//! (`C = lin - ratio * ang` for rack-and-pinion, `C = ang_a + ratio * ang_b` for the
//! gear, `len_a + ratio * len_b = rest` for the pulley), with
//! `lambda = C / (w_1 + ratio^2 w_2)`; and, for the dispatch, each joint solved alone
//! on its own bodies.

#![allow(clippy::disallowed_methods)]

use alice_physics::joint_extra::{
    solve_extra_joints, ExtraJoint, GearJoint, MouseJoint, PulleyJoint, RackAndPinionJoint,
    WeldJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// Dynamic body with unit mass and inverse inertia `(0, 0, w)` so that the
/// effective angular inverse mass `|inv_inertia|` is exactly `w`.
fn spinner(pos: Vec3Fix, w: f64) -> RigidBody {
    let mut b = RigidBody::new(pos, Fix128::ONE);
    b.inv_inertia = v3(0.0, 0.0, w);
    b
}

fn rz(theta: f64) -> QuatFix {
    QuatFix::new(
        Fix128::ZERO,
        Fix128::ZERO,
        fx((theta / 2.0).sin()),
        fx((theta / 2.0).cos()),
    )
}

/// Signed rotation angle about z of a pure-z quaternion.
fn angle_z(q: QuatFix) -> f64 {
    2.0 * q.z.to_f64().atan2(q.w.to_f64())
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 16)
}

/// Bodies for the mixed dispatch: weld (0, 1), mouse (2), rack-and-pinion (3, 4),
/// gear (5, 6), pulley (7, 8).
fn mixed_bodies() -> Vec<RigidBody> {
    let mut rack = RigidBody::new(v3(0.5, 0.0, 0.0), Fix128::ONE);
    rack.prev_position = Vec3Fix::ZERO;
    vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE),
        rack,
        RigidBody::new_static(Vec3Fix::from_int(0, -3, 0)),
        spinner(Vec3Fix::from_int(10, 0, 0), 1.0),
        spinner(Vec3Fix::from_int(12, 0, 0), 1.0),
        RigidBody::new(Vec3Fix::from_int(-1, 5, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(1, 5, 0), Fix128::ONE),
    ]
}

fn mixed_joints() -> Vec<ExtraJoint> {
    vec![
        ExtraJoint::Weld(WeldJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )),
        ExtraJoint::Mouse(MouseJoint::new(
            2,
            Vec3Fix::from_int(0, 10, 0),
            Fix128::from_int(100),
            Fix128::from_int(50),
            Fix128::from_int(5),
        )),
        ExtraJoint::RackAndPinion(RackAndPinionJoint::new(
            3,
            4,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Z,
            Fix128::from_int(2),
        )),
        ExtraJoint::Gear(GearJoint::new(5, 6, 0, 1, Fix128::from_int(2))),
        ExtraJoint::Pulley(PulleyJoint::new(
            7,
            8,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-1, 10, 0),
            Vec3Fix::from_int(1, 10, 0),
            Fix128::ONE,
        )),
    ]
}

/// One `solve_extra_joints` call over all five kinds equals solving each joint alone
/// (no cross-talk, every arm dispatched), and the kinds with a closed form hit it:
/// rigid weld to a static body closes the gap exactly (`x = 0`), the rack moved 0.5
/// against a static pinion is pulled back to `x = 0`, the resting gear and the pulley
/// at its rest length do not move.
#[test]
fn mixed_dispatch_equals_each_joint_alone_and_meets_the_closed_forms() {
    let joints = mixed_joints();
    let mut mixed = mixed_bodies();
    solve_extra_joints(&mut mixed, &joints, dt());

    let mut alone = mixed_bodies();
    for j in &joints {
        let mut b = mixed_bodies();
        solve_extra_joints(&mut b, core::slice::from_ref(j), dt());
        let touched: &[usize] = match j {
            ExtraJoint::Weld(_) => &[0, 1],
            ExtraJoint::Mouse(_) => &[2],
            ExtraJoint::RackAndPinion(_) => &[3, 4],
            ExtraJoint::Gear(_) => &[5, 6],
            ExtraJoint::Pulley(_) => &[7, 8],
        };
        for &i in touched {
            alone[i] = b[i];
        }
    }
    for i in 0..mixed.len() {
        assert_eq!(mixed[i].position, alone[i].position, "body {i} position");
        assert_eq!(mixed[i].rotation, alone[i].rotation, "body {i} rotation");
    }

    let start = mixed_bodies();
    // normalising the 5-unit gap leaves a few ulp (2^-64 each)
    for c in [
        mixed[1].position.x,
        mixed[1].position.y,
        mixed[1].position.z,
    ] {
        assert!(c.to_f64().abs() < 1e-15, "weld closes the gap: {c:?}");
    }
    assert_eq!(mixed[3].position, Vec3Fix::ZERO, "rack pulled back");
    assert_eq!(mixed[4].rotation, QuatFix::IDENTITY, "static pinion");
    assert_eq!(mixed[5].rotation, QuatFix::IDENTITY, "resting gear a");
    assert_eq!(mixed[6].rotation, QuatFix::IDENTITY, "resting gear b");
    assert_eq!(mixed[7].position, start[7].position, "pulley at rest");
    assert_eq!(mixed[8].position, start[8].position, "pulley at rest");
    // the mouse moves its body toward the target along +y only
    assert_eq!(mixed[2].position.x, Fix128::ZERO);
    assert_eq!(mixed[2].position.z, Fix128::ZERO);
    assert!(mixed[2].position.y > start[2].position.y);
}

/// Both bodies dynamic, `w_lin = w_ang = 1`, rack moved `C = 0.5`, pinion still:
/// `lambda = C / (1 + r^2)`, rack moves `-lambda`, pinion turns `+r lambda` about the
/// pinion axis. ratio 1: x = 0.25, theta = 0.25; ratio 2: x = 0.4, theta = 0.2. The
/// angle tolerance covers the unnormalised `(axis * half, 1)` quaternion step
/// (`2 atan(theta / 2)` vs `theta`, 0.5% at 0.25).
#[test]
fn rack_and_pinion_splits_the_correction_by_the_xpbd_projection() {
    for (ratio, want_x, want_theta) in [(1.0, 0.25, 0.25), (2.0, 0.4, 0.2)] {
        let mut rack = RigidBody::new(v3(0.5, 0.0, 0.0), Fix128::ONE);
        rack.prev_position = Vec3Fix::ZERO;
        let pinion = spinner(Vec3Fix::from_int(0, -3, 0), 1.0);
        let mut bodies = vec![rack, pinion];
        let j = RackAndPinionJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, fx(ratio));
        solve_extra_joints(&mut bodies, &[ExtraJoint::RackAndPinion(j)], dt());
        let x = bodies[0].position.x.to_f64();
        let theta = angle_z(bodies[1].rotation);
        assert!(
            (x - want_x).abs() < 1e-15,
            "ratio {ratio}: rack x {x}, want {want_x}"
        );
        assert_eq!(bodies[0].position.y, Fix128::ZERO);
        assert!(
            (theta - want_theta).abs() < 1e-2 * want_theta,
            "ratio {ratio}: pinion angle {theta}, want {want_theta}"
        );
        assert_eq!(bodies[1].position, Vec3Fix::from_int(0, -3, 0));
    }
}

/// Static rack at rest, pinion turned by +0.02 this step: `lin = 0` so the pinion is
/// turned back to `theta = 0` (to third order in the angle: residual ~1e-6).
#[test]
fn rack_and_pinion_static_rack_turns_the_pinion_back() {
    let rack = RigidBody::new_static(Vec3Fix::ZERO);
    let mut pinion = spinner(Vec3Fix::from_int(0, -3, 0), 1.0);
    pinion.rotation = rz(0.02);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z, fx(2.0));
    solve_extra_joints(&mut bodies, &[ExtraJoint::RackAndPinion(j)], dt());
    let theta = angle_z(bodies[1].rotation);
    assert!(theta.abs() < 1e-5, "pinion angle {theta}");
    assert_eq!(bodies[0].position, Vec3Fix::ZERO);
}

/// Gear ratio 2, `w_a = w_b = 1`, gear a turned `0.1` this step, b still:
/// `C = 0.1`, `lambda = C / (1 + 4) = 0.02`, `d_theta_a = -0.02`, `d_theta_b = -0.04`,
/// so after one solve `theta_a + 2 theta_b = 0` (to second order).
#[test]
#[ignore = "known defect: AUD-A-S34-011: same root as AUD-A-S3W1-015 for ratio 2: `solve_gear` rotates gear a by +lambda w_a (the sign that grows C) and the read angle is unsigned; measured theta_a 0.1200 (want 0.08) and theta_b -0.0400 after one solve, residual 0.0400 instead of 0"]
fn gear_ratio_two_closes_the_signed_residual() {
    let mut a = spinner(Vec3Fix::ZERO, 1.0);
    a.rotation = rz(0.1);
    let b = spinner(Vec3Fix::from_int(3, 0, 0), 1.0);
    let mut bodies = vec![a, b];
    let j = GearJoint::new(0, 1, 0, 1, fx(2.0));
    solve_extra_joints(&mut bodies, &[ExtraJoint::Gear(j)], dt());
    let ta = angle_z(bodies[0].rotation);
    let tb = angle_z(bodies[1].rotation);
    let c = ta + 2.0 * tb;
    assert!(c.abs() < 1e-3, "theta_a {ta}, theta_b {tb}, residual {c}");
    assert!((ta - 0.08).abs() < 1e-3, "theta_a {ta}, want 0.08");
    assert!((tb + 0.04).abs() < 1e-3, "theta_b {tb}, want -0.04");
}

/// Pulley ratio 2, rest `len_a + 2 len_b = 5 + 2 * 5 = 15`; body a is pulled down
/// by 1 (total 16). Repeated solves must return the total to the rest value
/// (rope length conservation).
#[test]
#[ignore = "known defect: AUD-A-S34-012: same root as AUD-A-S3W1-011 for ratio 2: `solve_pulley` uses `error = Fix128::ZERO` and stores no rest length, so the bodies never move; measured total length 16 after 50 solves, want the rest value 15"]
fn pulley_ratio_two_conserves_the_rope_length() {
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::from_int(-1, 5, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(1, 5, 0), Fix128::ONE),
    ];
    let j = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 10, 0),
        Vec3Fix::from_int(1, 10, 0),
        fx(2.0),
    );
    assert_eq!(j.total_length(&bodies), Fix128::from_int(15));
    bodies[0].position = Vec3Fix::from_int(-1, 4, 0);
    for _ in 0..50 {
        solve_extra_joints(&mut bodies, &[ExtraJoint::Pulley(j)], dt());
    }
    let t = j.total_length(&bodies).to_f64();
    assert!((t - 15.0).abs() < 1e-6, "total length {t}, want 15");
}
