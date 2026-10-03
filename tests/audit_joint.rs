//! Audit oracle for `joint`: XPBD position / angle corrections of the seven joint types
//! checked against hand closed forms (generalised inverse mass, exact single-solve rigid limits).
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::joint::{
    solve_joints, solve_joints_breakable, BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint,
    HingeJoint, Joint, JointType, SliderJoint, SpringJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn dt() -> Fix128 {
    fx(1.0 / 60.0)
}
fn st(p: Vec3Fix) -> RigidBody {
    RigidBody::new_static(p)
}
fn dynb(p: Vec3Fix, mass: f64) -> RigidBody {
    RigidBody::new(p, fx(mass))
}
fn with_inertia(mut b: RigidBody, i: f64) -> RigidBody {
    b.inv_inertia = v3(i, i, i);
    b
}
fn rot(axis: Vec3Fix, ang: f64) -> QuatFix {
    QuatFix::from_axis_angle(axis, fx(ang))
}
fn p(b: &RigidBody) -> (f64, f64, f64) {
    (
        b.position.x.to_f64(),
        b.position.y.to_f64(),
        b.position.z.to_f64(),
    )
}
fn near(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b}");
}
fn len(v: Vec3Fix) -> f64 {
    v.length().to_f64()
}
/// angle between the world images of a local axis before / after
fn axis_angle(q0: QuatFix, q1: QuatFix, local: Vec3Fix) -> f64 {
    let a = q0.rotate_vec(local);
    let b = q1.rotate_vec(local);
    let c = a.dot(b).to_f64().clamp(-1.0, 1.0);
    let s = len(a.cross(b));
    s.atan2(c)
}
/// signed twist of q about unit axis (swing-twist, angle in (-pi, pi])
fn twist(q: QuatFix, axis: Vec3Fix) -> f64 {
    let (qx, qy, qz, qw) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64());
    let (ax, ay, az) = (axis.x.to_f64(), axis.y.to_f64(), axis.z.to_f64());
    let s = qx * ax + qy * ay + qz * az;
    let n = (s * s + qw * qw).sqrt();
    let (s, w) = if qw < 0.0 {
        (-s / n, -qw / n)
    } else {
        (s / n, qw / n)
    };
    2.0 * s.atan2(w)
}

// ---------------- accessors ----------------

#[test]
fn accessors_for_all_seven_variants() {
    let o = Vec3Fix::ZERO;
    let ux = Vec3Fix::UNIT_X;
    let f = Some(fx(7.0));
    let cases: Vec<(Joint, JointType)> = vec![
        (
            Joint::Ball(BallJoint::new(1, 2, o, o).with_break_force(fx(7.0))),
            JointType::Ball,
        ),
        (
            Joint::Hinge(HingeJoint::new(1, 2, o, o, ux, ux).with_break_force(fx(7.0))),
            JointType::Hinge,
        ),
        (
            Joint::Fixed(FixedJoint::new(1, 2, o, o, QuatFix::IDENTITY).with_break_force(fx(7.0))),
            JointType::Fixed,
        ),
        (
            Joint::Slider(SliderJoint::new(1, 2, ux, o, o).with_break_force(fx(7.0))),
            JointType::Slider,
        ),
        (
            Joint::Spring(
                SpringJoint::new(1, 2, o, o, fx(1.0), fx(2.0), fx(0.0)).with_break_force(fx(7.0)),
            ),
            JointType::Spring,
        ),
        (
            Joint::D6(D6Joint::new(1, 2, o, o).with_break_force(fx(7.0))),
            JointType::D6,
        ),
        (
            Joint::ConeTwist(ConeTwistJoint::new(1, 2, o, o, ux, ux).with_break_force(fx(7.0))),
            JointType::ConeTwist,
        ),
    ];
    for (j, t) in cases {
        assert_eq!(j.bodies(), (1, 2));
        assert_eq!(j.joint_type(), t);
        assert_eq!(j.break_force(), f);
    }
    assert_eq!(Joint::Ball(BallJoint::new(0, 1, o, o)).break_force(), None);
    assert_eq!(Joint::D6(D6Joint::new(0, 1, o, o)).break_force(), None);
}

#[test]
fn builders_set_only_their_field() {
    let o = Vec3Fix::ZERO;
    let b = BallJoint::new(0, 1, o, o).with_compliance(fx(0.5));
    assert_eq!(b.compliance, fx(0.5));
    assert_eq!(b.break_force, None);
    let h = HingeJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
        .with_limits(fx(-0.5), fx(0.75))
        .with_compliance(fx(0.1));
    assert_eq!((h.angle_min, h.angle_max), (Some(fx(-0.5)), Some(fx(0.75))));
    assert_eq!(h.compliance, fx(0.1));
    assert_eq!(h.angular_compliance, Fix128::ZERO);
    let s = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, o, o).with_limits(fx(-1.0), fx(2.0));
    assert_eq!((s.limit_min, s.limit_max), (Some(fx(-1.0)), Some(fx(2.0))));
    let c = ConeTwistJoint::new(0, 1, o, o, Vec3Fix::UNIT_X, Vec3Fix::UNIT_X);
    assert_eq!((c.cone_limit, c.twist_limit), (Fix128::HALF_PI, Fix128::PI));
    let c = c.with_limits(fx(0.3), fx(0.2));
    assert_eq!((c.cone_limit, c.twist_limit), (fx(0.3), fx(0.2)));
    let d = D6Joint::new(0, 1, o, o)
        .with_linear_motion(D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
        .with_angular_motion(D6Motion::Free, D6Motion::Locked, D6Motion::Limited)
        .with_linear_limits(v3(-1.0, -2.0, -3.0), v3(1.0, 2.0, 3.0))
        .with_angular_limits(v3(-0.1, -0.2, -0.3), v3(0.1, 0.2, 0.3));
    assert_eq!(
        (d.linear_x, d.linear_y, d.linear_z),
        (D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
    );
    assert_eq!(
        (d.angular_x, d.angular_y, d.angular_z),
        (D6Motion::Free, D6Motion::Locked, D6Motion::Limited)
    );
    assert_eq!(d.linear_limit_min, v3(-1.0, -2.0, -3.0));
    assert_eq!(d.angular_limit_max, v3(0.1, 0.2, 0.3));
    // defaults: everything free, limits +-1 / +-pi
    let d0 = D6Joint::new(0, 1, o, o);
    assert_eq!(
        (d0.linear_x, d0.angular_z),
        (D6Motion::Free, D6Motion::Free)
    );
    assert_eq!(d0.linear_limit_max, v3(1.0, 1.0, 1.0));
    assert_eq!(
        d0.angular_limit_min,
        Vec3Fix::new(-Fix128::PI, -Fix128::PI, -Fix128::PI)
    );
}

// ---------------- ball ----------------

#[test]
fn ball_static_dynamic_pulls_the_anchor_onto_the_other_anchor() {
    let mut b = vec![st(v3(1.0, 2.0, 3.0)), dynb(v3(4.0, 6.0, 3.0), 2.0)];
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))],
        &mut b,
        dt(),
    );
    let (x, y, z) = p(&b[1]);
    near(x, 1.0, 1e-12, "x");
    near(y, 2.0, 1e-12, "y");
    near(z, 3.0, 1e-12, "z");
    assert_eq!(b[0].position, v3(1.0, 2.0, 3.0));
}

/// Two dynamic bodies with masses 1 and 3: the separation d closes in the inverse-mass ratio 3:1.
#[test]
fn ball_dynamic_pair_splits_the_correction_by_inverse_mass() {
    let mut b = vec![dynb(v3(0.0, 0.0, 0.0), 1.0), dynb(v3(4.0, 0.0, 0.0), 3.0)];
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))],
        &mut b,
        dt(),
    );
    let (xa, _, _) = p(&b[0]);
    let (xb, _, _) = p(&b[1]);
    near(xa, 3.0, 1e-12, "light body moves 3/4 of the gap");
    near(xb, 3.0, 1e-12, "heavy body ends at the same point");
    // centre of mass is unchanged: 1*0 + 3*4 = 12 = (1+3)*3
    near(1.0 * xa + 3.0 * xb, 12.0, 1e-11, "momentum-like invariant");
}

/// Compliance: the closure fraction is w / (w + alpha / dt^2).
#[test]
fn ball_compliance_leaves_the_documented_fraction_of_the_error() {
    let d = 1.0 / 60.0;
    let alpha = 4.0 * d * d; // alpha/dt^2 = 4 ; static + unit mass: w = 1
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 0.0, 0.0), 1.0)];
    solve_joints(
        &[Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(fx(alpha)),
        )],
        &mut b,
        fx(d),
    );
    // correction = 2 * 1/(1+4) = 0.4
    near(p(&b[1]).0, 2.0 - 0.4, 1e-9, "compliant ball");
    // a larger compliance closes less
    let mut c = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 0.0, 0.0), 1.0)];
    solve_joints(
        &[Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(fx(9.0 * d * d)),
        )],
        &mut c,
        fx(d),
    );
    near(p(&c[1]).0, 2.0 - 0.2, 1e-9, "alpha/dt^2 = 9");
}

#[test]
fn ball_no_ops() {
    // coincident anchors
    let mut b = vec![dynb(v3(1.0, 1.0, 1.0), 1.0), dynb(v3(1.0, 1.0, 1.0), 1.0)];
    let before = (b[0].position, b[1].position);
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))],
        &mut b,
        dt(),
    );
    assert_eq!((b[0].position, b[1].position), before);
    // two static bodies
    let mut s = vec![st(v3(0.0, 0.0, 0.0)), st(v3(5.0, 0.0, 0.0))];
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        ))],
        &mut s,
        dt(),
    );
    assert_eq!(
        (s[0].position, s[1].position),
        (v3(0.0, 0.0, 0.0), v3(5.0, 0.0, 0.0))
    );
    // no joints, no change
    solve_joints(&[], &mut s, dt());
}

#[test]
fn ball_anchor_uses_the_body_rotation() {
    // A rotated +90 deg about z, local anchor (1,0,0) -> world (0,1,0) relative to A
    let mut a = st(v3(10.0, 0.0, 0.0));
    a.rotation = rot(Vec3Fix::UNIT_Z, std::f64::consts::FRAC_PI_2);
    let mut b = vec![a, dynb(v3(0.0, 0.0, 0.0), 1.0)];
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            v3(1.0, 0.0, 0.0),
            Vec3Fix::ZERO,
        ))],
        &mut b,
        dt(),
    );
    let (x, y, _) = p(&b[1]);
    near(x, 10.0, 1e-9, "x");
    near(y, 1.0, 1e-9, "y");
}

/// An anchor offset from the centre of mass has lever arm r: the XPBD generalised inverse mass is
/// w = 1/m + (r x n)^T I^-1 (r x n) and the correction splits between translation and rotation.
/// B: m = 1, I^-1 = 2, anchor r = (-0.5,0,0), error d along +y: w = 1 + 0.25 * 2 = 1.5, so the
/// centre of mass moves d / 1.5 and the body rotates.
#[test]
#[ignore = "known defect: AUD-A-S1W6-006: ball/hinge/fixed/slider/cone-twist/D6 positional corrections use w = inv_m_a + inv_m_b with no lever arm and apply translation only: an anchor offset 0.5 from the COM moves the COM by the full 0.01 (expected 0.01/1.5 = 0.00667) and the body does not rotate"]
fn ball_offset_anchor_splits_the_correction_between_translation_and_rotation() {
    let d = 0.01;
    let mut b = vec![
        st(Vec3Fix::ZERO),
        with_inertia(dynb(v3(0.5, -d, 0.0), 1.0), 2.0),
    ];
    let q0 = b[1].rotation;
    solve_joints(
        &[Joint::Ball(BallJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            v3(-0.5, 0.0, 0.0),
        ))],
        &mut b,
        dt(),
    );
    let dy = b[1].position.y.to_f64() - (-d);
    near(dy, d / 1.5, 1e-9, "COM correction");
    assert!(
        axis_angle(q0, b[1].rotation, Vec3Fix::UNIT_X) > 1e-6,
        "no rotation produced"
    );
}

// ---------------- hinge ----------------

fn hinge(limits: Option<(f64, f64)>) -> Joint {
    let mut h = HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    );
    if let Some((lo, hi)) = limits {
        h = h.with_limits(fx(lo), fx(hi));
    }
    Joint::Hinge(h)
}

#[test]
fn hinge_realigns_a_tilted_axis_in_one_rigid_solve() {
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = rot(Vec3Fix::UNIT_X, 0.3);
    solve_joints(&[hinge(None)], &mut b, dt());
    let a = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
    near(a.x.to_f64(), 0.0, 1e-9, "ax");
    near(a.y.to_f64(), 0.0, 1e-9, "ay");
    near(a.z.to_f64(), 1.0, 1e-9, "az");
}

/// Both bodies dynamic, I^-1 = 1 (A) and 3 (B): corrections are in the ratio w_a : w_b = 1 : 3 and
/// together remove the full 0.4 rad of misalignment.
#[test]
fn hinge_splits_the_angular_correction_by_inverse_inertia() {
    let mut b = vec![
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 1.0),
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 3.0),
    ];
    b[1].rotation = rot(Vec3Fix::UNIT_X, 0.4);
    let q = (b[0].rotation, b[1].rotation);
    solve_joints(&[hinge(None)], &mut b, dt());
    let ta = axis_angle(q.0, b[0].rotation, Vec3Fix::UNIT_Z);
    let tb = axis_angle(q.1, b[1].rotation, Vec3Fix::UNIT_Z);
    near(ta, 0.4 * 1.0 / 4.0, 1e-9, "A share");
    near(tb, 0.4 * 3.0 / 4.0, 1e-9, "B share");
    // relative misalignment is gone
    let aa = b[0].rotation.rotate_vec(Vec3Fix::UNIT_Z);
    let ab = b[1].rotation.rotate_vec(Vec3Fix::UNIT_Z);
    near(len(aa.cross(ab)), 0.0, 1e-9, "residual");
}

#[test]
fn hinge_limits_clamp_to_the_exact_limit_on_both_sides() {
    let limits = Some((-0.5, 0.5));
    for (start, want) in [
        (0.9, 0.5),
        (-0.9, -0.5),
        (0.2, 0.2),
        (-0.5, -0.5),
        (0.5, 0.5),
    ] {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
        b[1].rotation = rot(Vec3Fix::UNIT_Z, start);
        solve_joints(&[hinge(limits)], &mut b, dt());
        near(
            twist(b[1].rotation, Vec3Fix::UNIT_Z),
            want,
            1e-9,
            &format!("start {start}"),
        );
    }
}

#[test]
fn hinge_limits_with_dynamic_a_share_by_inverse_inertia() {
    // A has I^-1 = 1, B has I^-1 = 3: the 0.4 rad violation (0.9 vs 0.5) is removed 1:3
    let mut b = vec![
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 1.0),
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 3.0),
    ];
    b[1].rotation = rot(Vec3Fix::UNIT_Z, 0.9);
    solve_joints(&[hinge(Some((-0.5, 0.5)))], &mut b, dt());
    near(
        twist(b[0].rotation, Vec3Fix::UNIT_Z),
        0.1,
        1e-9,
        "A turns the same way as B's excess, taking 1/4",
    );
    near(twist(b[1].rotation, Vec3Fix::UNIT_Z), 0.9 - 0.3, 1e-9, "B");
}

#[test]
fn hinge_position_part_is_the_ball_constraint() {
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(1.0, 2.0, 0.0), 1.0)];
    solve_joints(&[hinge(None)], &mut b, dt());
    let (x, y, _) = p(&b[1]);
    near(x, 0.0, 1e-12, "x");
    near(y, 0.0, 1e-12, "y");
}

/// Field doc: "Minimum angle (radians, None = no limit)" - each side independently.
#[test]
#[ignore = "known defect: AUD-A-S1W6-007: hinge limits are applied only when BOTH angle_min and angle_max are Some; a lone angle_min (or angle_max) is silently ignored although each field documents `None = no limit` independently (same for SliderJoint.limit_min / limit_max)"]
fn hinge_with_only_a_minimum_still_enforces_it() {
    let mut h = HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    );
    h.angle_min = Some(fx(-0.5));
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = rot(Vec3Fix::UNIT_Z, -0.9);
    solve_joints(&[Joint::Hinge(h)], &mut b, dt());
    near(
        twist(b[1].rotation, Vec3Fix::UNIT_Z),
        -0.5,
        1e-9,
        "angle_min only",
    );
}

// ---------------- fixed ----------------

#[test]
fn fixed_locks_position_and_relative_rotation() {
    let rel = rot(Vec3Fix::UNIT_Y, 0.6);
    let anchor_a = v3(1.0, 0.0, 0.0);
    let j = Joint::Fixed(FixedJoint::new(0, 1, anchor_a, Vec3Fix::ZERO, rel));
    let mut b = vec![st(v3(0.0, 0.0, 0.0)), dynb(v3(3.0, 1.0, -2.0), 1.0)];
    b[1].rotation = rot(v3(1.0, 1.0, 0.0).normalize(), 0.8);
    solve_joints(&[j], &mut b, dt());
    let (x, y, z) = p(&b[1]);
    near(x, 1.0, 1e-9, "x");
    near(y, 0.0, 1e-9, "y");
    near(z, 0.0, 1e-9, "z");
    // rotation: q_b == q_a * rel (A identity)
    let r = b[1].rotation;
    let w = rel;
    let dotq = r.x.to_f64() * w.x.to_f64()
        + r.y.to_f64() * w.y.to_f64()
        + r.z.to_f64() * w.z.to_f64()
        + r.w.to_f64() * w.w.to_f64();
    near(
        dotq.abs(),
        1.0,
        1e-9,
        "q_b equals relative_rotation up to sign",
    );
}

#[test]
fn fixed_satisfied_state_is_a_fixed_point() {
    let rel = rot(Vec3Fix::UNIT_Z, 0.4);
    let j = Joint::Fixed(FixedJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, rel));
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = rel;
    solve_joints(&[j], &mut b, dt());
    assert_eq!(b[1].position, Vec3Fix::ZERO);
    near(
        twist(b[1].rotation, Vec3Fix::UNIT_Z),
        0.4,
        1e-9,
        "unchanged",
    );
}

#[test]
fn fixed_rotation_lock_splits_by_inverse_inertia() {
    let j = Joint::Fixed(FixedJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    let mut b = vec![
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 1.0),
        with_inertia(dynb(Vec3Fix::ZERO, 1.0), 3.0),
    ];
    b[1].rotation = rot(Vec3Fix::UNIT_Z, 0.8);
    solve_joints(&[j], &mut b, dt());
    let ta = twist(b[0].rotation, Vec3Fix::UNIT_Z);
    let tb = twist(b[1].rotation, Vec3Fix::UNIT_Z);
    near(ta, 0.2, 1e-9, "A takes 1/4");
    near(tb, 0.8 - 0.6, 1e-9, "B removes 3/4");
}

// ---------------- slider ----------------

fn slider(limits: Option<(f64, f64)>) -> Joint {
    let mut s = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO);
    if let Some((lo, hi)) = limits {
        s = s.with_limits(fx(lo), fx(hi));
    }
    Joint::Slider(s)
}

#[test]
fn slider_removes_only_the_perpendicular_error() {
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(3.0, 4.0, -2.0), 1.0)];
    solve_joints(&[slider(None)], &mut b, dt());
    let (x, y, z) = p(&b[1]);
    near(x, 3.0, 1e-12, "along the axis is free");
    near(y, 0.0, 1e-12, "y");
    near(z, 0.0, 1e-12, "z");
}

#[test]
fn slider_axis_follows_body_a_rotation() {
    let mut a = st(Vec3Fix::ZERO);
    a.rotation = rot(Vec3Fix::UNIT_Z, std::f64::consts::FRAC_PI_2); // axis X -> world Y
    let mut b = vec![a, dynb(v3(2.0, 3.0, 0.0), 1.0)];
    solve_joints(&[slider(None)], &mut b, dt());
    let (x, y, _) = p(&b[1]);
    near(x, 0.0, 1e-9, "x");
    near(y, 3.0, 1e-9, "y free along the rotated axis");
}

#[test]
fn slider_limits_clamp_both_ends_and_split_by_inverse_mass() {
    for (start, want) in [(5.0, 2.0), (-4.0, -1.0), (1.5, 1.5)] {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(start, 0.0, 0.0), 1.0)];
        solve_joints(&[slider(Some((-1.0, 2.0)))], &mut b, dt());
        near(p(&b[1]).0, want, 1e-9, &format!("start {start}"));
    }
    // dynamic pair 1 : 3 masses, overshoot 3 beyond the max (5 vs 2): A moves 3/4*3 forward
    let mut b = vec![dynb(Vec3Fix::ZERO, 1.0), dynb(v3(5.0, 0.0, 0.0), 3.0)];
    solve_joints(&[slider(Some((-1.0, 2.0)))], &mut b, dt());
    near(p(&b[0]).0, 2.25, 1e-9, "A (light) moves +2.25");
    near(p(&b[1]).0, 5.0 - 0.75, 1e-9, "B (heavy) moves -0.75");
}

#[test]
#[ignore = "known defect: AUD-A-S1W6-007: lone SliderJoint.limit_max (limit_min None) is silently ignored, see hinge_with_only_a_minimum_still_enforces_it"]
fn slider_with_only_a_maximum_still_enforces_it() {
    let mut s = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO);
    s.limit_max = Some(fx(2.0));
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(5.0, 0.0, 0.0), 1.0)];
    solve_joints(&[Joint::Slider(s)], &mut b, dt());
    near(p(&b[1]).0, 2.0, 1e-9, "limit_max only");
}

// ---------------- spring ----------------

fn spring(rest: f64, k: f64, c: f64) -> Joint {
    Joint::Spring(SpringJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        fx(rest),
        fx(k),
        fx(c),
    ))
}

#[test]
fn spring_displacement_is_force_times_dt_times_inverse_mass_toward_rest() {
    // stretched by 1 (distance 3, rest 2), k = 12, mass 2: B moves -k x dt / m toward A
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(3.0, 0.0, 0.0), 2.0)];
    solve_joints(&[spring(2.0, 12.0, 0.0)], &mut b, dt());
    near(p(&b[1]).0, 3.0 - 12.0 * 1.0 / 60.0 / 2.0, 1e-9, "stretched");
    // compressed (distance 1, rest 2): pushed away
    let mut c = vec![st(Vec3Fix::ZERO), dynb(v3(1.0, 0.0, 0.0), 2.0)];
    solve_joints(&[spring(2.0, 12.0, 0.0)], &mut c, dt());
    near(
        p(&c[1]).0,
        1.0 + 12.0 * 1.0 / 60.0 / 2.0,
        1e-9,
        "compressed",
    );
    // at rest length nothing happens
    let mut d = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 0.0, 0.0), 2.0)];
    solve_joints(&[spring(2.0, 12.0, 0.0)], &mut d, dt());
    near(p(&d[1]).0, 2.0, 1e-12, "at rest");
}

#[test]
fn spring_damping_opposes_the_relative_velocity_along_the_line() {
    // at rest length, B moving away at 3 m/s with c = 2: damping force = c v = 6 pulls B back
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 0.0, 0.0), 1.0)];
    b[1].velocity = v3(3.0, 0.0, 0.0);
    solve_joints(&[spring(2.0, 5.0, 2.0)], &mut b, dt());
    near(p(&b[1]).0, 2.0 - 6.0 / 60.0, 1e-9, "damping");
    // dynamic pair: split by inverse mass 1 : 1/3, force = k x
    let mut c = vec![dynb(Vec3Fix::ZERO, 1.0), dynb(v3(4.0, 0.0, 0.0), 3.0)];
    solve_joints(&[spring(2.0, 6.0, 0.0)], &mut c, dt());
    near(
        p(&c[0]).0,
        6.0 * 2.0 / 60.0,
        1e-9,
        "A moves toward B by F dt / m_a",
    );
    near(
        p(&c[1]).0,
        4.0 - 6.0 * 2.0 / 60.0 / 3.0,
        1e-9,
        "B moves toward A by F dt / m_b",
    );
}

/// A force-law spring changes a position by F dt^2 / m (acceleration integrated twice). The
/// correction here scales with dt, so the effective stiffness depends on the time step.
#[test]
#[ignore = "known defect: AUD-A-S1W6-008: solve_spring_joint adds `F*dt*inv_mass` to the POSITION (a velocity-sized quantity): the correction scales with dt, not dt^2 (halving dt halves it instead of quartering it), so the spring's effective stiffness is k/dt in force units"]
fn spring_correction_scales_with_dt_squared() {
    let run = |dt: f64| {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(3.0, 0.0, 0.0), 1.0)];
        solve_joints(&[spring(2.0, 12.0, 0.0)], &mut b, fx(dt));
        3.0 - b[1].position.x.to_f64()
    };
    let ratio = run(1.0 / 60.0) / run(1.0 / 120.0);
    near(ratio, 4.0, 1e-6, "dx(dt)/dx(dt/2)");
}

// ---------------- compute_force / breakable ----------------

#[test]
fn compute_force_closed_forms_per_variant() {
    let o = Vec3Fix::ZERO;
    let bodies = vec![st(o), dynb(v3(3.0, 4.0, 0.0), 1.0)];
    near(
        Joint::Ball(BallJoint::new(0, 1, o, o))
            .compute_force(&bodies)
            .to_f64(),
        5.0,
        1e-9,
        "ball",
    );
    near(
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            o,
            o,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        ))
        .compute_force(&bodies)
        .to_f64(),
        5.0,
        1e-9,
        "hinge",
    );
    near(
        Joint::Fixed(FixedJoint::new(0, 1, o, o, QuatFix::IDENTITY))
            .compute_force(&bodies)
            .to_f64(),
        5.0,
        1e-9,
        "fixed",
    );
    near(
        Joint::D6(D6Joint::new(0, 1, o, o))
            .compute_force(&bodies)
            .to_f64(),
        5.0,
        1e-9,
        "d6",
    );
    near(
        Joint::ConeTwist(ConeTwistJoint::new(
            0,
            1,
            o,
            o,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_X,
        ))
        .compute_force(&bodies)
        .to_f64(),
        5.0,
        1e-9,
        "cone",
    );
    // slider: only the perpendicular component (axis X): 4
    near(
        Joint::Slider(SliderJoint::new(0, 1, Vec3Fix::UNIT_X, o, o))
            .compute_force(&bodies)
            .to_f64(),
        4.0,
        1e-9,
        "slider",
    );
    // spring: |k (d - rest)| = |2 * (5 - 3)| = 4 and also for compression |2 * (5 - 8)| = 6
    near(
        Joint::Spring(SpringJoint::new(0, 1, o, o, fx(3.0), fx(2.0), fx(0.0)))
            .compute_force(&bodies)
            .to_f64(),
        4.0,
        1e-9,
        "spring stretched",
    );
    near(
        Joint::Spring(SpringJoint::new(0, 1, o, o, fx(8.0), fx(2.0), fx(0.0)))
            .compute_force(&bodies)
            .to_f64(),
        6.0,
        1e-9,
        "spring compressed",
    );
}

/// XPBD: the constraint force of a compliant joint with position error C is C / alpha, so a softer
/// joint (larger alpha) carries less force at the same separation. compute_force ignores compliance.
#[test]
#[ignore = "known defect: AUD-A-S1W6-009: Joint::compute_force returns the anchor SEPARATION (metres) for ball/hinge/fixed/slider/D6/cone-twist and only the spring returns a force (N); `break_force` is compared against both, and compliance never enters: a ball joint 0.02 m apart reports 0.02 for compliance 0.01 and for 0.04"]
fn compute_force_is_a_force_not_a_separation() {
    let o = Vec3Fix::ZERO;
    let bodies = vec![st(o), dynb(v3(0.02, 0.0, 0.0), 1.0)];
    let soft = Joint::Ball(BallJoint::new(0, 1, o, o).with_compliance(fx(0.04)))
        .compute_force(&bodies)
        .to_f64();
    let stiff = Joint::Ball(BallJoint::new(0, 1, o, o).with_compliance(fx(0.01)))
        .compute_force(&bodies)
        .to_f64();
    near(stiff, 0.02 / 0.01, 1e-9, "C / alpha");
    near(soft, 0.02 / 0.04, 1e-9, "C / alpha");
}

#[test]
fn breakable_returns_descending_indices_skips_them_and_uses_strict_greater() {
    let o = Vec3Fix::ZERO;
    // joint 0: sep 1 vs break 0.5 -> breaks; joint 1: sep 2 vs 5 -> holds; joint 2: sep 3 vs 3 (equal) -> holds;
    // joint 3: sep 4 vs 3.9 -> breaks; joint 4: unbreakable
    let mut bodies = vec![
        st(o),
        dynb(v3(1.0, 0.0, 0.0), 1.0),
        dynb(v3(2.0, 0.0, 0.0), 1.0),
        dynb(v3(3.0, 0.0, 0.0), 1.0),
        dynb(v3(4.0, 0.0, 0.0), 1.0),
        dynb(v3(5.0, 0.0, 0.0), 1.0),
    ];
    let joints = [
        Joint::Ball(BallJoint::new(0, 1, o, o).with_break_force(fx(0.5))),
        Joint::Ball(BallJoint::new(0, 2, o, o).with_break_force(fx(5.0))),
        Joint::Ball(BallJoint::new(0, 3, o, o).with_break_force(fx(3.0))),
        Joint::Ball(BallJoint::new(0, 4, o, o).with_break_force(fx(3.9))),
        Joint::Ball(BallJoint::new(0, 5, o, o)),
    ];
    let broken = solve_joints_breakable(&joints, &mut bodies, dt());
    assert_eq!(broken, vec![3, 0]);
    // broken joints were not solved
    assert_eq!(bodies[1].position, v3(1.0, 0.0, 0.0));
    assert_eq!(bodies[4].position, v3(4.0, 0.0, 0.0));
    // surviving joints were solved
    near(p(&bodies[2]).0, 0.0, 1e-12, "joint 1 solved");
    near(
        p(&bodies[3]).0,
        0.0,
        1e-12,
        "joint 2 solved (equal is not broken)",
    );
    near(p(&bodies[5]).0, 0.0, 1e-12, "unbreakable solved");
    // nothing to report when nothing is breakable
    let mut b2 = vec![st(o), dynb(v3(1.0, 0.0, 0.0), 1.0)];
    assert!(
        solve_joints_breakable(&[Joint::Ball(BallJoint::new(0, 1, o, o))], &mut b2, dt())
            .is_empty()
    );
}

// ---------------- D6 ----------------

#[test]
fn d6_default_is_free_and_does_nothing() {
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(5.0, -3.0, 2.0), 1.0)];
    b[1].rotation = rot(Vec3Fix::UNIT_Y, 1.0);
    let before = (b[1].position, b[1].rotation);
    solve_joints(
        &[Joint::D6(D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO))],
        &mut b,
        dt(),
    );
    assert_eq!((b[1].position, b[1].rotation), before);
}

#[test]
fn d6_locked_axes_close_exactly_those_axes() {
    let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_linear_motion(
        D6Motion::Locked,
        D6Motion::Free,
        D6Motion::Locked,
    );
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 3.0, -4.0), 1.0)];
    solve_joints(&[Joint::D6(j)], &mut b, dt());
    let (x, y, z) = p(&b[1]);
    near(x, 0.0, 1e-12, "x locked");
    near(y, 3.0, 1e-12, "y free");
    near(z, 0.0, 1e-12, "z locked");
}

#[test]
fn d6_limited_axis_clamps_to_the_limit() {
    let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
        .with_linear_motion(D6Motion::Limited, D6Motion::Free, D6Motion::Free)
        .with_linear_limits(v3(-1.0, 0.0, 0.0), v3(2.0, 0.0, 0.0));
    for (start, want) in [(5.0, 2.0), (-3.0, -1.0), (1.0, 1.0)] {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(start, 0.0, 0.0), 1.0)];
        solve_joints(&[Joint::D6(j)], &mut b, dt());
        near(p(&b[1]).0, want, 1e-12, &format!("start {start}"));
    }
}

#[test]
fn d6_linear_axes_follow_frame_a() {
    let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_linear_motion(
        D6Motion::Locked,
        D6Motion::Free,
        D6Motion::Free,
    );
    j.local_frame_a = rot(Vec3Fix::UNIT_Z, std::f64::consts::FRAC_PI_2); // frame x -> world y
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 3.0, 0.0), 1.0)];
    solve_joints(&[Joint::D6(j)], &mut b, dt());
    let (x, y, _) = p(&b[1]);
    near(x, 2.0, 1e-9, "world x free");
    near(y, 0.0, 1e-9, "world y (frame x) locked");
}

#[test]
fn d6_locked_and_limited_angles() {
    // locked about z: a 0.5 rad twist is removed
    let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_angular_motion(
        D6Motion::Free,
        D6Motion::Free,
        D6Motion::Locked,
    );
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = rot(Vec3Fix::UNIT_Z, 0.5);
    solve_joints(&[Joint::D6(j)], &mut b, dt());
    near(twist(b[1].rotation, Vec3Fix::UNIT_Z), 0.0, 1e-9, "locked");
    // limited about z to [-0.2, 0.3]
    let j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)
        .with_angular_motion(D6Motion::Free, D6Motion::Free, D6Motion::Limited)
        .with_angular_limits(v3(0.0, 0.0, -0.2), v3(0.0, 0.0, 0.3));
    for (start, want) in [(0.8, 0.3), (-0.7, -0.2), (0.1, 0.1)] {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
        b[1].rotation = rot(Vec3Fix::UNIT_Z, start);
        solve_joints(&[Joint::D6(j)], &mut b, dt());
        near(
            twist(b[1].rotation, Vec3Fix::UNIT_Z),
            want,
            1e-9,
            &format!("start {start}"),
        );
    }
}

/// Field doc: "Reference frame for body B". The joint frames are what zero error is measured from;
/// local_frame_b is stored but never read by the solver.
#[test]
#[ignore = "known defect: AUD-A-S1W6-010 / AUD-B-S1W6-001: D6Joint.local_frame_b is never read (grep: only its definition and default): with all angles locked and B at the pose where frame_b maps onto frame_a (q_b = frame_b^-1) the solver still rotates B back to identity"]
fn d6_local_frame_b_defines_the_zero_error_pose() {
    let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_angular_motion(
        D6Motion::Locked,
        D6Motion::Locked,
        D6Motion::Locked,
    );
    let fb = rot(Vec3Fix::UNIT_Z, std::f64::consts::FRAC_PI_2);
    j.local_frame_b = fb;
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = fb.conjugate(); // frame_b rotated into the world equals frame_a (identity)
    solve_joints(&[Joint::D6(j)], &mut b, dt());
    near(
        twist(b[1].rotation, Vec3Fix::UNIT_Z),
        -std::f64::consts::FRAC_PI_2,
        1e-9,
        "B must not move",
    );
}

// ---------------- cone-twist ----------------

fn cone(cone_limit: f64, twist_limit: f64) -> Joint {
    Joint::ConeTwist(
        ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_X,
        )
        .with_limits(fx(cone_limit), fx(twist_limit)),
    )
}

#[test]
fn cone_limit_clamps_the_swing_to_exactly_the_limit() {
    // B swings 0.8 rad about z, cone limit 0.5 -> swing becomes 0.5
    let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    b[1].rotation = rot(Vec3Fix::UNIT_Z, 0.8);
    solve_joints(&[cone(0.5, 3.0)], &mut b, dt());
    let swing = axis_angle(QuatFix::IDENTITY, b[1].rotation, Vec3Fix::UNIT_X);
    near(swing, 0.5, 1e-9, "swing");
    // within the cone: unchanged
    let mut c = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
    c[1].rotation = rot(Vec3Fix::UNIT_Z, 0.3);
    solve_joints(&[cone(0.5, 3.0)], &mut c, dt());
    near(
        axis_angle(QuatFix::IDENTITY, c[1].rotation, Vec3Fix::UNIT_X),
        0.3,
        1e-9,
        "inside",
    );
}

#[test]
fn twist_limit_clamps_both_signs() {
    for (start, want) in [(0.9, 0.4), (-0.9, -0.4), (0.2, 0.2)] {
        let mut b = vec![st(Vec3Fix::ZERO), dynb(Vec3Fix::ZERO, 1.0)];
        b[1].rotation = rot(Vec3Fix::UNIT_X, start);
        solve_joints(&[cone(1.5, 0.4)], &mut b, dt());
        near(
            twist(b[1].rotation, Vec3Fix::UNIT_X),
            want,
            1e-9,
            &format!("start {start}"),
        );
    }
}

#[test]
fn cone_twist_position_part_closes_the_anchor_gap() {
    let mut b = vec![st(Vec3Fix::ZERO), dynb(v3(2.0, 2.0, 1.0), 1.0)];
    solve_joints(&[cone(1.5, 3.0)], &mut b, dt());
    assert!(p(&b[1]).0.abs() + p(&b[1]).1.abs() + p(&b[1]).2.abs() < 1e-12);
}

#[test]
fn solvers_are_deterministic_and_order_of_joints_is_sequential() {
    let o = Vec3Fix::ZERO;
    let joints = [
        Joint::Ball(BallJoint::new(0, 1, o, o)),
        Joint::Ball(BallJoint::new(1, 2, o, o)),
    ];
    let mk = || {
        vec![
            st(o),
            dynb(v3(1.0, 0.0, 0.0), 1.0),
            dynb(v3(3.0, 0.0, 0.0), 1.0),
        ]
    };
    let (mut a, mut b) = (mk(), mk());
    solve_joints(&joints, &mut a, dt());
    solve_joints(&joints, &mut b, dt());
    assert_eq!(a[1].position, b[1].position);
    assert_eq!(a[2].position, b[2].position);
    // Gauss-Seidel order: joint 0 pulls body 1 onto the static origin (x = 0), then joint 1 closes the
    // 3 m gap between the two equal masses at the midpoint 1.5
    near(p(&a[1]).0, 1.5, 1e-12, "body 1 (moved again by joint 1)");
    near(p(&a[2]).0, 1.5, 1e-12, "body 2");
}
