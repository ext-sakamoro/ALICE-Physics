//! Audit oracles for `motor` (`PdController`, `PdController3D`, `JointMotor`,
//! `apply_motors`).
//!
//! Expected values are hand-derived PD laws, momentum conservation and an
//! independent rotation-matrix extraction of the rotation error (so the 3-D
//! controller is not checked against its own quaternion product).

#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::motor::{apply_motors, JointMotor, MotorMode, PdController, PdController3D};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

// ---------------------------------------------------------------------------
// PdController (scalar)
// ---------------------------------------------------------------------------

#[test]
fn new_controller_is_off_with_zero_targets() {
    let pd = PdController::new(fx(1.0), fx(2.0), fx(3.0));
    assert_eq!(pd.mode, MotorMode::Off);
    assert_eq!(pd.target_position, Fix128::ZERO);
    assert_eq!(pd.target_velocity, Fix128::ZERO);
    assert_eq!((pd.kp, pd.kd, pd.max_force), (fx(1.0), fx(2.0), fx(3.0)));
}

#[test]
fn default_gains_are_100_10_1000() {
    let pd = PdController::default();
    assert_eq!(pd.kp, fx(100.0));
    assert_eq!(pd.kd, fx(10.0));
    assert_eq!(pd.max_force, fx(1000.0));
    assert_eq!(pd.mode, MotorMode::Off);
}

#[test]
fn setters_switch_mode_and_record_target() {
    let mut pd = PdController::default();
    pd.set_position_target(fx(2.5));
    assert_eq!(
        (pd.mode, pd.target_position),
        (MotorMode::Position, fx(2.5))
    );
    pd.set_velocity_target(fx(-1.5));
    assert_eq!(
        (pd.mode, pd.target_velocity),
        (MotorMode::Velocity, fx(-1.5))
    );
    // the position target survives a switch of mode
    assert_eq!(pd.target_position, fx(2.5));
    pd.disable();
    assert_eq!(pd.mode, MotorMode::Off);
    assert_eq!(pd.compute(fx(100.0), fx(100.0)), Fix128::ZERO);
}

#[test]
fn position_mode_is_kp_error_plus_kd_velocity_error() {
    // u = kp (x* - x) + kd (v* - v), v* = 0 unless set
    let mut pd = PdController::new(fx(10.0), fx(3.0), fx(1000.0));
    pd.set_position_target(fx(5.0));
    // x = 2, v = 4 : 10*3 + 3*(0-4) = 18
    assert_eq!(pd.compute(fx(2.0), fx(4.0)), fx(18.0));
    // x = 7, v = -2 : 10*(-2) + 3*(0+2) = -14
    assert_eq!(pd.compute(fx(7.0), fx(-2.0)), fx(-14.0));
    // with a velocity target stored
    pd.target_velocity = fx(1.0);
    // x = 5, v = 0 : 0 + 3*(1-0) = 3
    assert_eq!(pd.compute(fx(5.0), fx(0.0)), fx(3.0));
}

#[test]
fn velocity_mode_is_kp_velocity_error_and_ignores_kd_and_position() {
    let mut pd = PdController::new(fx(10.0), fx(99.0), fx(1000.0));
    pd.set_velocity_target(fx(5.0));
    pd.target_position = fx(1234.0);
    assert_eq!(pd.compute(fx(-77.0), fx(2.0)), fx(30.0));
    assert_eq!(pd.compute(fx(0.0), fx(9.0)), fx(-40.0));
}

#[test]
fn output_is_clamped_symmetrically_and_exact_at_the_bound() {
    let mut pd = PdController::new(fx(100.0), Fix128::ZERO, fx(10.0));
    pd.set_position_target(fx(1.0));
    assert_eq!(pd.compute(fx(0.0), fx(0.0)), fx(10.0)); // 100 -> 10
    assert_eq!(pd.compute(fx(2.0), fx(0.0)), fx(-10.0)); // -100 -> -10
                                                         // exactly at the bound (8 * 1.25 = 10) is not altered
    pd.kp = fx(8.0);
    assert_eq!(pd.compute(fx(-0.25), fx(0.0)), fx(10.0));
    pd.kp = fx(100.0);
    // just inside
    let inside = pd.compute(fx(0.95), fx(0.0)).to_f64();
    assert!((inside - 5.0).abs() < 1e-9);
    // velocity mode clamps too
    pd.set_velocity_target(fx(-1.0));
    assert_eq!(pd.compute(fx(0.0), fx(0.0)), fx(-10.0));
}

#[test]
fn position_mode_damps_toward_the_stored_velocity_target() {
    // `target_velocity` is documented "(for Velocity mode)" but Position mode
    // also consumes it: a stale velocity target left by `set_velocity_target`
    // survives `set_position_target`.
    let mut pd = PdController::new(fx(10.0), fx(2.0), fx(1000.0));
    pd.set_velocity_target(fx(3.0));
    pd.set_position_target(fx(0.0));
    // error 0, damping 2*(3 - 0) = 6 (not 0)
    assert_eq!(pd.compute(fx(0.0), fx(0.0)), fx(6.0));
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-001: negative max_force makes clamp(v, -max, +max) an empty interval and compute returns +|max_force| even at zero error (observed +10 for max_force = -10, kp error = 0)"]
fn negative_max_force_must_not_produce_force_at_zero_error() {
    let mut pd = PdController::new(fx(5.0), fx(1.0), fx(-10.0));
    pd.set_position_target(fx(3.0));
    // at the target and at rest the PD law is exactly zero
    assert_eq!(pd.compute(fx(3.0), fx(0.0)), Fix128::ZERO);
}

// ---------------------------------------------------------------------------
// PdController3D
// ---------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

fn quat_axis_angle(axis: [f64; 3], ang: f64) -> QuatFix {
    QuatFix::from_axis_angle(v3(axis[0], axis[1], axis[2]), fx(ang))
}

/// Rotation matrix of the unit-axis rotation (Rodrigues), independent of QuatFix.
fn rot_matrix(axis: [f64; 3], ang: f64) -> M3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = (ang.sin(), ang.cos());
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

fn mul(a: M3, b: M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

fn transpose(a: M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[j][i];
        }
    }
    r
}

/// Shortest rotation vector (angle in [0, pi], axis) of a rotation matrix.
fn rotation_vector(r: M3) -> ([f64; 3], f64) {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let theta = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    let s = 2.0 * theta.sin();
    let axis = [
        (r[2][1] - r[1][2]) / s,
        (r[0][2] - r[2][0]) / s,
        (r[1][0] - r[0][1]) / s,
    ];
    (axis, theta)
}

fn pd3(kp: [f64; 3], kd: [f64; 3], max: f64) -> PdController3D {
    PdController3D::new(v3(kp[0], kp[1], kp[2]), v3(kd[0], kd[1], kd[2]), fx(max))
}

fn pos_mode(kp: [f64; 3], kd: [f64; 3], max: f64, target: QuatFix) -> PdController3D {
    let mut c = pd3(kp, kd, max);
    c.set_rotation_target(target);
    c
}

#[test]
fn controller3d_new_defaults() {
    let c = pd3([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 7.0);
    assert_eq!(c.mode, MotorMode::Off);
    assert_eq!(c.target_rotation, QuatFix::IDENTITY);
    assert_eq!(c.target_angular_velocity, Vec3Fix::ZERO);
    assert_eq!(c.max_torque, fx(7.0));
    assert_eq!(
        c.compute_torque(quat_axis_angle([1.0, 0.0, 0.0], 1.0), v3(3.0, 4.0, 5.0)),
        Vec3Fix::ZERO
    );
}

#[test]
fn set_rotation_target_switches_to_position_mode() {
    let mut c = pd3([1.0; 3], [0.0; 3], 100.0);
    let q = quat_axis_angle([0.0, 0.0, 1.0], 0.5);
    c.set_rotation_target(q);
    assert_eq!(c.mode, MotorMode::Position);
    assert_eq!(c.target_rotation, q);
}

#[test]
fn small_rotation_error_gives_per_axis_gain_times_angle_axis() {
    // error rotation (world frame) = R_target * R_current^T, small angle
    let kp = [3.0, 5.0, 7.0];
    let cur_axis = [1.0, -2.0, 0.5];
    let cur_ang = 0.9;
    let err_axis: [f64; 3] = [0.3, 0.5, -0.8];
    let n =
        (err_axis[0] * err_axis[0] + err_axis[1] * err_axis[1] + err_axis[2] * err_axis[2]).sqrt();
    let err_u = [err_axis[0] / n, err_axis[1] / n, err_axis[2] / n];
    let theta = 0.02;
    let r_e = rot_matrix(err_u, theta);
    let r_c = rot_matrix(cur_axis, cur_ang);
    let r_t = mul(r_e, r_c);
    // build the target quaternion from the composed matrix via two axis-angle
    // rotations (error then current), so no matrix -> quaternion routine is needed
    let q_c = quat_axis_angle(cur_axis, cur_ang);
    let q_e = quat_axis_angle(err_u, theta);
    let q_t = q_e.mul(q_c);
    let _ = r_t;
    let c = pos_mode(kp, [0.0; 3], 1e6, q_t);
    let t = arr(c.compute_torque(q_c, Vec3Fix::ZERO));
    for k in 0..3 {
        let want = kp[k] * theta * err_u[k];
        assert!(
            (t[k] - want).abs() < 1e-4 * want.abs().max(1e-3),
            "axis {k}: {} vs {}",
            t[k],
            want
        );
    }
}

#[test]
fn rotation_error_is_expressed_in_the_world_frame() {
    // current = 90 deg about x, target = current rotated a further 0.02 about
    // WORLD z. The torque must point along world z (not along a body axis).
    let q_c = quat_axis_angle([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2);
    let q_e = quat_axis_angle([0.0, 0.0, 1.0], 0.02);
    let q_t = q_e.mul(q_c);
    let c = pos_mode([10.0; 3], [0.0; 3], 1e6, q_t);
    let t = arr(c.compute_torque(q_c, Vec3Fix::ZERO));
    assert!(t[0].abs() < 1e-6 && t[1].abs() < 1e-6, "{t:?}");
    assert!((t[2] - 10.0 * 0.02).abs() < 1e-3, "{t:?}");
}

#[test]
fn rotation_error_magnitude_is_two_sine_of_half_angle_of_the_shortest_arc() {
    // moderate angle (1.0 rad): the vector part of the error quaternion
    // times two is 2 sin(theta/2) about the (matrix-derived) axis
    let q_c = quat_axis_angle([0.0, 1.0, 1.0], 0.4);
    let q_e = quat_axis_angle([1.0, 0.0, 0.0], 1.0);
    let q_t = q_e.mul(q_c);
    let r_e = mul(
        mul(
            rot_matrix([1.0, 0.0, 0.0], 1.0),
            rot_matrix([0.0, 1.0, 1.0], 0.4),
        ),
        transpose(rot_matrix([0.0, 1.0, 1.0], 0.4)),
    );
    let (axis, theta) = rotation_vector(r_e);
    let c = pos_mode([1.0; 3], [0.0; 3], 1e6, q_t);
    let t = arr(c.compute_torque(q_c, Vec3Fix::ZERO));
    let m = 2.0 * (theta / 2.0).sin();
    for k in 0..3 {
        assert!(
            (t[k] - m * axis[k]).abs() < 1e-8,
            "axis {k}: {} vs {}",
            t[k],
            m * axis[k]
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-002: the in-code comment calls the position error \"axis-angle\" but the controller uses 2*sin(theta/2)*axis; at 90 degrees the magnitude is 1.4142 (not 1.5708), at 170 degrees 1.992 (not 2.967), so kp does not mean torque per radian outside small angles"]
fn rotation_error_magnitude_equals_the_angle_for_a_quarter_turn() {
    let q_t = quat_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
    let c = pos_mode([1.0; 3], [0.0; 3], 1e6, q_t);
    let t = arr(c.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO));
    assert!((t[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "{t:?}");
}

#[test]
fn torque_is_invariant_under_target_quaternion_sign() {
    let q = quat_axis_angle([0.0, 0.0, 1.0], -0.17453292519943295); // -10 deg, w > 0
    let neg = QuatFix::new(-q.x, -q.y, -q.z, -q.w); // same rotation
    let a = pos_mode([1.0; 3], [0.0; 3], 1e6, q).compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO);
    let b = pos_mode([1.0; 3], [0.0; 3], 1e6, neg).compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO);
    let (a, b) = (arr(a), arr(b));
    for k in 0..3 {
        assert!((a[k] - b[k]).abs() < 1e-9, "axis {k}: {} vs {}", a[k], b[k]);
    }
}

#[test]
fn torque_is_invariant_under_current_quaternion_sign() {
    // same statement for the current orientation: the product target * conj(current)
    // changes sign with current, so the error is taken from its w >= 0 representative
    let tgt = quat_axis_angle([0.0, 0.0, 1.0], 0.1);
    let cur = quat_axis_angle([0.0, 0.0, 1.0], 0.0);
    let neg_cur = QuatFix::new(-cur.x, -cur.y, -cur.z, -cur.w);
    let a = arr(pos_mode([1.0; 3], [0.0; 3], 1e6, tgt).compute_torque(cur, Vec3Fix::ZERO));
    let b = arr(pos_mode([1.0; 3], [0.0; 3], 1e6, tgt).compute_torque(neg_cur, Vec3Fix::ZERO));
    for k in 0..3 {
        assert!((a[k] - b[k]).abs() < 1e-9, "axis {k}: {} vs {}", a[k], b[k]);
    }
}

#[test]
fn position_mode_damping_is_kd_times_angular_velocity_error() {
    let kd = [2.0, 3.0, 4.0];
    let mut c = pos_mode([100.0; 3], kd, 1e6, QuatFix::IDENTITY);
    c.target_angular_velocity = v3(1.0, 0.0, -1.0);
    // zero rotation error; w = (0.5, 2, 3) : kd*(w* - w) = (2*0.5, 3*-2, 4*-4)
    let t = arr(c.compute_torque(QuatFix::IDENTITY, v3(0.5, 2.0, 3.0)));
    assert!((t[0] - 1.0).abs() < 1e-12);
    assert!((t[1] + 6.0).abs() < 1e-12);
    assert!((t[2] + 16.0).abs() < 1e-12);
}

#[test]
fn velocity_mode_is_kp_times_velocity_error_and_ignores_kd_and_rotation() {
    let mut c = pd3([2.0, 3.0, 4.0], [50.0, 50.0, 50.0], 1e6);
    c.mode = MotorMode::Velocity;
    c.target_angular_velocity = v3(1.0, 1.0, 1.0);
    c.target_rotation = quat_axis_angle([1.0, 0.0, 0.0], 2.0);
    let t = arr(c.compute_torque(quat_axis_angle([0.0, 1.0, 0.0], 1.0), v3(0.0, 2.0, 3.0)));
    assert!((t[0] - 2.0).abs() < 1e-12);
    assert!((t[1] + 3.0).abs() < 1e-12);
    assert!((t[2] + 8.0).abs() < 1e-12);
}

#[test]
fn magnitude_is_clamped_preserving_direction_and_not_altered_at_the_bound() {
    let mut c = pd3([1.0; 3], [0.0; 3], 4.0);
    c.mode = MotorMode::Velocity;
    c.target_angular_velocity = v3(3.0, 4.0, 0.0); // |torque| = 5
    let t = arr(c.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO));
    assert!((t[0] - 2.4).abs() < 1e-9 && (t[1] - 3.2).abs() < 1e-9 && t[2].abs() < 1e-12);
    // exactly at the bound: unchanged
    c.max_torque = fx(5.0);
    let t = c.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO);
    assert_eq!(t, v3(3.0, 4.0, 0.0));
    // position mode clamps the same way
    let mut p = pos_mode([0.0; 3], [3.0, 4.0, 0.0], 1.0, QuatFix::IDENTITY);
    p.target_angular_velocity = v3(1.0, 1.0, 0.0);
    let t = arr(p.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO)); // (3,4,0) -> |.|=5 -> 1
    assert!((t[0] - 0.6).abs() < 1e-9 && (t[1] - 0.8).abs() < 1e-9);
}

#[test]
fn zero_max_torque_gives_zero_vector_not_nan() {
    let mut c = pd3([1.0; 3], [0.0; 3], 0.0);
    c.mode = MotorMode::Velocity;
    c.target_angular_velocity = v3(1.0, 2.0, 3.0);
    let t = arr(c.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO));
    for v in t {
        assert!(v.abs() < 1e-12, "{t:?}");
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-004: negative max_torque is not rejected; `mag > max` is always true and torque * (max / mag) reverses the vector (torque (3,4,0) with max_torque = -5 returns (-3,-4,0))"]
fn negative_max_torque_must_not_reverse_the_torque() {
    let mut c = pd3([1.0; 3], [0.0; 3], -5.0);
    c.mode = MotorMode::Velocity;
    c.target_angular_velocity = v3(3.0, 4.0, 0.0);
    let t = arr(c.compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO));
    // a torque must never point against the commanded error
    assert!(t[0] * 3.0 + t[1] * 4.0 >= 0.0, "{t:?}");
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-005: a non-unit target or current quaternion is not normalised; scaling the target quaternion by 2 doubles the commanded torque (0.0999 -> 0.1999 at kp 1, angle 0.1 rad)"]
fn non_unit_target_quaternion_must_not_scale_the_torque() {
    let q = quat_axis_angle([0.0, 0.0, 1.0], 0.1);
    let big = QuatFix::new(q.x * fx(2.0), q.y * fx(2.0), q.z * fx(2.0), q.w * fx(2.0));
    let a =
        arr(pos_mode([1.0; 3], [0.0; 3], 1e6, q).compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO));
    let b = arr(
        pos_mode([1.0; 3], [0.0; 3], 1e6, big).compute_torque(QuatFix::IDENTITY, Vec3Fix::ZERO)
    );
    for k in 0..3 {
        assert!((a[k] - b[k]).abs() < 1e-9, "axis {k}: {} vs {}", a[k], b[k]);
    }
}

// ---------------------------------------------------------------------------
// JointMotor / apply_motors
// ---------------------------------------------------------------------------

fn ball(a: usize, b: usize) -> Joint {
    Joint::Ball(BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO))
}

fn pair(pa: [f64; 3], pb: [f64; 3], ia: f64, ib: f64) -> Vec<RigidBody> {
    let mut a = RigidBody::new_dynamic(v3(pa[0], pa[1], pa[2]), Fix128::ONE);
    let mut b = RigidBody::new_dynamic(v3(pb[0], pb[1], pb[2]), Fix128::ONE);
    a.inv_mass = fx(ia);
    b.inv_mass = fx(ib);
    vec![a, b]
}

#[test]
fn joint_motor_new_stores_index_and_controller() {
    let pd = PdController::default();
    let m = JointMotor::new(3, pd);
    assert_eq!(m.joint_index, 3);
    assert_eq!(m.controller, pd);
}

#[test]
fn position_mode_damping_uses_relative_velocity_along_the_axis() {
    // B at +4x moving +2 along the axis; kp 10, kd 3, target separation 6
    // force = 10*(6-4) + 3*(0-2) = 14 ; dt 1/4 -> impulse 3.5 -> B 2 + 3.5
    let mut bodies = pair([0.0; 3], [4.0, 0.0, 0.0], 0.0, 1.0);
    bodies[1].velocity = v3(2.0, 0.0, 0.0);
    let mut pd = PdController::new(fx(10.0), fx(3.0), fx(1000.0));
    pd.set_position_target(fx(6.0));
    apply_motors(
        &[JointMotor::new(0, pd)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.25),
    );
    assert_eq!(bodies[1].velocity, v3(5.5, 0.0, 0.0));
}

#[test]
fn velocity_mode_sign_follows_b_minus_a_relative_velocity() {
    // A moves +1 along the axis, B at rest: relative velocity along +x is -1.
    // kp 10, target 0 -> vel_error +1 -> force 10 -> impulse 2.5, A -= 2.5, B += 2.5
    let mut bodies = pair([0.0; 3], [1.0, 0.0, 0.0], 1.0, 1.0);
    bodies[0].velocity = v3(1.0, 0.0, 0.0);
    let mut pd = PdController::new(fx(10.0), Fix128::ZERO, fx(1000.0));
    pd.set_velocity_target(fx(0.0));
    apply_motors(
        &[JointMotor::new(0, pd)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.25),
    );
    assert_eq!(bodies[0].velocity, v3(-1.5, 0.0, 0.0));
    assert_eq!(bodies[1].velocity, v3(2.5, 0.0, 0.0));
}

#[test]
fn oblique_axis_impulse_follows_the_unit_separation_vector() {
    // B at (3, 4, 0): separation 5, axis (0.6, 0.8, 0). target 7, kp 5 -> force 10,
    // dt 1/2 -> impulse 5 -> B velocity (3, 4, 0)
    let mut bodies = pair([0.0; 3], [3.0, 4.0, 0.0], 0.0, 1.0);
    let mut pd = PdController::new(fx(5.0), Fix128::ZERO, fx(1000.0));
    pd.set_position_target(fx(7.0));
    apply_motors(
        &[JointMotor::new(0, pd)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.5),
    );
    let v = arr(bodies[1].velocity);
    assert!(
        (v[0] - 3.0).abs() < 1e-9 && (v[1] - 4.0).abs() < 1e-9 && v[2].abs() < 1e-12,
        "{v:?}"
    );
}

#[test]
fn equal_and_opposite_impulse_conserves_momentum_for_unequal_masses() {
    let mut a = RigidBody::new_dynamic(Vec3Fix::ZERO, fx(2.0));
    let mut b = RigidBody::new_dynamic(v3(0.0, 0.0, 5.0), fx(3.0));
    a.velocity = v3(0.0, 0.0, 0.0);
    b.velocity = v3(0.0, 0.0, 0.0);
    let mut bodies = vec![a, b];
    let mut pd = PdController::new(fx(7.0), Fix128::ZERO, fx(1000.0));
    pd.set_position_target(fx(9.0));
    apply_motors(
        &[JointMotor::new(0, pd)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.5),
    );
    let pa = 2.0 * bodies[0].velocity.z.to_f64();
    let pb = 3.0 * bodies[1].velocity.z.to_f64();
    assert!((pa + pb).abs() < 1e-9, "momentum {pa} + {pb}");
    // impulse = force*dt = 7*4*0.5 = 14 : B gets +14/3, A gets -14/2
    assert!((pb - 14.0).abs() < 1e-9 && (pa + 14.0).abs() < 1e-9);
}

#[test]
fn velocity_change_is_linear_in_dt() {
    let mk = |dt: f64| {
        let mut bodies = pair([0.0; 3], [4.0, 0.0, 0.0], 0.0, 1.0);
        let mut pd = PdController::new(fx(10.0), Fix128::ZERO, fx(1000.0));
        pd.set_position_target(fx(6.0));
        apply_motors(
            &[JointMotor::new(0, pd)],
            &[ball(0, 1)],
            &mut bodies,
            fx(dt),
        );
        bodies[1].velocity.x.to_f64()
    };
    assert_eq!(mk(0.0), 0.0);
    assert!((mk(0.5) - 2.0 * mk(0.25)).abs() < 1e-12);
    assert!((mk(0.25) - 5.0).abs() < 1e-12);
}

#[test]
fn two_static_bodies_are_untouched_and_positions_never_change() {
    let mut bodies = pair([0.0; 3], [4.0, 0.0, 0.0], 0.0, 0.0);
    let mut pd = PdController::new(fx(10.0), Fix128::ZERO, fx(1000.0));
    pd.set_position_target(fx(6.0));
    apply_motors(
        &[JointMotor::new(0, pd)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.25),
    );
    assert_eq!(bodies[0].velocity, Vec3Fix::ZERO);
    assert_eq!(bodies[1].velocity, Vec3Fix::ZERO);
    assert_eq!(bodies[1].position, v3(4.0, 0.0, 0.0));
}

#[test]
fn two_motors_apply_in_order_each_seeing_the_updated_velocity() {
    let mut bodies = pair([0.0; 3], [1.0, 0.0, 0.0], 0.0, 1.0);
    let mut pd = PdController::new(fx(4.0), Fix128::ZERO, fx(1000.0));
    pd.set_velocity_target(fx(2.0));
    let m = JointMotor::new(0, pd);
    // dt 1/4: first: err 2 -> force 8 -> +2 -> v 2 ; second: err 0 -> nothing
    apply_motors(&[m, m], &[ball(0, 1)], &mut bodies, fx(0.25));
    assert_eq!(bodies[1].velocity, v3(2.0, 0.0, 0.0));
}

#[test]
fn off_motor_is_skipped_even_if_listed_first() {
    let mut bodies = pair([0.0; 3], [4.0, 0.0, 0.0], 0.0, 1.0);
    let mut off = PdController::new(fx(10.0), Fix128::ZERO, fx(1000.0));
    off.set_position_target(fx(6.0));
    off.disable();
    let mut on = off;
    on.set_position_target(fx(6.0));
    apply_motors(
        &[JointMotor::new(0, off), JointMotor::new(0, on)],
        &[ball(0, 1)],
        &mut bodies,
        fx(0.25),
    );
    assert_eq!(bodies[1].velocity, v3(5.0, 0.0, 0.0));
}
