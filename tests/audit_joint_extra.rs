//! Audit S3-1 oracles for `joint_extra` (pulley / gear / weld / rack-and-pinion / mouse).
//!
//! Existing oracles (`analytic_joint_extra_wiring`) cover the builders, the
//! break comparisons, `total_length` for identity rotations and "dispatches
//! without panicking".  They pin the *current* mouse formula
//! (`k d dt inv_mass`) and never check that `solve_pulley`, `solve_gear` or
//! `solve_rack_and_pinion` satisfy their documented constraint.  Expected
//! values here are closed forms (mass-weighted XPBD projection, constraint
//! residual of the documented equation), not the module's own formulas.

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

fn body(pos: Vec3Fix, mass: f64) -> RigidBody {
    let mut b = RigidBody::new(pos, fx(mass));
    b.inv_inertia = v3(1.0, 1.0, 1.0);
    b
}

fn stat(pos: Vec3Fix) -> RigidBody {
    RigidBody::new_static(pos)
}

/// Rotation about z by `theta`, as an exact-enough unit quaternion (f64 sin/cos).
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

fn solve(bodies: &mut [RigidBody], j: ExtraJoint, dt: f64) {
    solve_extra_joints(bodies, &[j], fx(dt));
}

// ---------------------------------------------------------------------------
// PulleyJoint
// ---------------------------------------------------------------------------

/// `total_length` with rotated bodies and local anchors:
/// `|R a + p - g_a| + ratio |R b + p - g_b|`.
#[test]
fn pulley_total_length_uses_the_rotated_local_anchors() {
    let mut a = body(v3(0.0, 0.0, 0.0), 1.0);
    let mut b = body(v3(4.0, 0.0, 0.0), 1.0);
    a.rotation = rz(std::f64::consts::FRAC_PI_2); // (1,0,0) -> (0,1,0)
    b.rotation = rz(std::f64::consts::PI); // (1,0,0) -> (-1,0,0)
    let j = PulleyJoint::new(
        0,
        1,
        v3(1.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        v3(0.0, 4.0, 0.0), // world_a = (0,1,0): len_a = 3
        v3(3.0, 0.0, 4.0), // world_b = (3,0,0): len_b = 4
        fx(2.0),
    );
    let t = j.total_length(&[a, b]).to_f64();
    assert!((t - (3.0 + 2.0 * 4.0)).abs() < 1e-9, "total {t}");
}

/// The documented constraint `len_a + ratio * len_b = total_length` must hold after
/// solving: a rope pulled out of its rest configuration is drawn back.
#[test]
#[ignore = "known defect: AUD-A-S3W1-011: solve_pulley hard-codes `error = Fix128::ZERO` (no rest length is stored), so solve_extra_joints never moves a pulley body; total_length stays 11 instead of returning to the rest value 10"]
fn pulley_solver_restores_the_total_rope_length() {
    let ground_a = v3(-1.0, 10.0, 0.0);
    let ground_b = v3(1.0, 10.0, 0.0);
    let mut bodies = vec![body(v3(-1.0, 5.0, 0.0), 1.0), body(v3(1.0, 5.0, 0.0), 1.0)];
    let j = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        ground_a,
        ground_b,
        fx(1.0),
    );
    let rest = j.total_length(&bodies).to_f64(); // 10
    bodies[0].position = v3(-1.0, 4.0, 0.0); // pulled down by 1: total 11
    for _ in 0..50 {
        solve(&mut bodies, ExtraJoint::Pulley(j), 1.0 / 16.0);
    }
    let t = j.total_length(&bodies).to_f64();
    assert!((t - rest).abs() < 1e-6, "total length {t}, rest {rest}");
}

// ---------------------------------------------------------------------------
// WeldJoint
// ---------------------------------------------------------------------------

/// Position projection splits by inverse mass: a (m=1) and b (m=3) one unit apart meet
/// at `x = w_a d / (w_a + w_b) = 0.75` from a.
#[test]
fn weld_position_correction_is_split_by_inverse_mass() {
    let mut bodies = vec![body(v3(0.0, 0.0, 0.0), 1.0), body(v3(1.0, 0.0, 0.0), 3.0)];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
    assert!((bodies[0].position.x.to_f64() - 0.75).abs() < 1e-12);
    assert!((bodies[1].position.x.to_f64() - 0.75).abs() < 1e-12);
}

/// XPBD compliance: `lambda = d / (w_a + w_b + alpha / dt^2)`; each body moves `w lambda`.
#[test]
fn weld_compliance_softens_the_position_correction() {
    let mut bodies = vec![body(v3(0.0, 0.0, 0.0), 1.0), body(v3(1.0, 0.0, 0.0), 1.0)];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_compliance(fx(0.0625));
    solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
    // alpha/dt^2 = 1.0 ; lambda = 1/(1+1+1)
    assert!((bodies[0].position.x.to_f64() - 1.0 / 3.0).abs() < 1e-12);
    assert!((bodies[1].position.x.to_f64() - 2.0 / 3.0).abs() < 1e-12);
}

/// Local anchors are rotated with their bodies: with matching orientations the
/// translation alone closes the anchor gap.
#[test]
fn weld_uses_rotated_local_anchors() {
    let q = rz(std::f64::consts::FRAC_PI_2);
    let mut a = stat(v3(0.0, 0.0, 0.0));
    let mut b = body(v3(5.0, 0.0, 0.0), 1.0);
    a.rotation = q;
    b.rotation = q;
    let mut bodies = vec![a, b];
    // both anchors (1,0,0) local -> world offset (0,1,0): gap = (5,0,0)
    let j = WeldJoint::new(
        0,
        1,
        v3(1.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
    assert!(
        bodies[1].position.x.to_f64().abs() < 1e-9,
        "{}",
        bodies[1].position.x.to_f64()
    );
    assert!(bodies[1].position.y.to_f64().abs() < 1e-9);
}

/// `compute_torque` is `|xyz|` of the error quaternion = `sin(phi/2)` for an error of
/// `phi`; it is also invariant under the double cover (`q` and `-q` are one rotation).
#[test]
fn weld_compute_torque_is_sin_half_angle_and_double_cover_invariant() {
    let phi = 0.8;
    let mut a = stat(Vec3Fix::ZERO);
    let mut b = body(Vec3Fix::ZERO, 1.0);
    a.rotation = QuatFix::IDENTITY;
    b.rotation = rz(phi);
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    let t = j.compute_torque(&[a, b]).to_f64();
    assert!((t - (phi / 2.0).sin()).abs() < 1e-12);
    let mut nb = b;
    nb.rotation = QuatFix::new(-b.rotation.x, -b.rotation.y, -b.rotation.z, -b.rotation.w);
    let t2 = j.compute_torque(&[a, nb]).to_f64();
    assert!(
        (t - t2).abs() < 1e-12,
        "q and -q must read the same: {t} vs {t2}"
    );
}

/// A weld must reduce the angular error whichever sign of the quaternion represents
/// the orientation (`q` and `-q` are the same rotation).
#[test]
#[ignore = "known defect: AUD-A-S3W1-012: solve_weld takes error_vec = rot_error.xyz without making w >= 0; for the negated quaternion the correction axis flips and the angular error grows (0.0998 -> 0.379 after 3 solves) instead of shrinking"]
fn weld_angular_correction_is_independent_of_quaternion_sign() {
    for neg in [false, true] {
        let mut b = body(Vec3Fix::ZERO, 1.0);
        let q = rz(0.2);
        b.rotation = if neg {
            QuatFix::new(-q.x, -q.y, -q.z, -q.w)
        } else {
            q
        };
        let mut bodies = vec![stat(Vec3Fix::ZERO), b];
        let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let before = j.compute_torque(&bodies).to_f64();
        for _ in 0..3 {
            solve(&mut bodies, ExtraJoint::Weld(j), 1.0 / 16.0);
        }
        let after = j.compute_torque(&bodies).to_f64();
        assert!(after < before, "neg = {neg}: {before} -> {after}");
    }
}

/// A rigid (compliance 0) weld between a static body and one dynamic body with
/// inverse inertia `i` about the error axis is one XPBD projection: one solve closes the
/// angular error (to first order in the angle).
#[test]
#[ignore = "known defect: AUD-A-S3W1-013: solve_weld's angular effective inverse mass is |inv_inertia| (vector norm, sqrt(3) for the isotropic case) and the correction is split equally between the bodies instead of by inverse inertia; one solve leaves 0.0085 of 0.02 rad (closes 1/sqrt(3))"]
fn rigid_weld_closes_a_small_angular_error_in_one_projection() {
    let phi = 0.02;
    let mut b = body(Vec3Fix::ZERO, 1.0);
    b.rotation = rz(phi);
    let mut bodies = vec![stat(Vec3Fix::ZERO), b];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    solve(&mut bodies, ExtraJoint::Weld(j), 1.0 / 16.0);
    let left = angle_z(bodies[1].rotation).abs();
    assert!(left < 1e-3 * phi + 1e-9, "remaining angle {left} of {phi}");
}

/// With both bodies rotating, the correction is distributed by inverse inertia
/// (`w_a : w_b`), the same rule as the position projection.
#[test]
#[ignore = "known defect: AUD-A-S3W1-013: apply_angular_correction rotates both bodies by the same angle regardless of inv_inertia (1:1 measured for w_a:w_b = 1:3)"]
fn weld_angular_correction_is_split_by_inverse_inertia() {
    let mut a = body(Vec3Fix::ZERO, 1.0);
    a.inv_inertia = v3(1.0, 1.0, 1.0);
    let mut b = body(Vec3Fix::ZERO, 1.0);
    b.inv_inertia = v3(3.0, 3.0, 3.0);
    b.rotation = rz(0.02);
    let mut bodies = vec![a, b];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    solve(&mut bodies, ExtraJoint::Weld(j), 1.0 / 16.0);
    let da = angle_z(bodies[0].rotation).abs();
    let db = (angle_z(bodies[1].rotation) - 0.02).abs();
    assert!(da > 0.0 && db > 0.0);
    let ratio = db / da; // want w_b / w_a = 3
    assert!((ratio - 3.0).abs() < 0.05, "b/a correction ratio {ratio}");
}

// ---------------------------------------------------------------------------
// RackAndPinionJoint
// ---------------------------------------------------------------------------

/// Pinion static, rack moved 0.5 along its axis this step: the whole correction lands
/// on the rack (constraint `lin = ratio * ang` with `ang = 0`).
#[test]
fn rack_and_pinion_static_pinion_pulls_the_rack_back() {
    let mut rack = body(v3(0.5, 0.0, 0.0), 1.0);
    rack.prev_position = v3(0.0, 0.0, 0.0);
    let pinion = stat(Vec3Fix::ZERO);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    assert!(
        bodies[0].position.x.to_f64().abs() < 1e-12,
        "{}",
        bodies[0].position.x.to_f64()
    );
}

/// A consistent step (`lin = ratio * ang`) is left alone to within the small-angle
/// approximation `angle ~ 2 |xyz| = 2 sin(theta/2)` (0.17% low at 0.2 rad).
#[test]
fn rack_and_pinion_consistent_motion_is_untouched() {
    let mut rack = body(v3(0.4, 0.0, 0.0), 1.0);
    rack.prev_position = Vec3Fix::ZERO;
    let mut pinion = body(Vec3Fix::ZERO, 1.0);
    pinion.rotation = rz(0.2);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    assert!(
        (bodies[0].position.x.to_f64() - 0.4).abs() < 2e-3,
        "{}",
        bodies[0].position.x.to_f64()
    );
    assert!((angle_z(bodies[1].rotation) - 0.2).abs() < 2e-3);
}

/// The documented constraint `linear = ratio * angular` is signed: a pinion turning
/// the other way must pair with a rack moving the other way.
#[test]
#[ignore = "known defect: AUD-A-S3W1-014: extract_angle_around_axis returns |projection| * 2 (unsigned), so a pinion rotated by -0.2 reads +0.2; with rack lin = -0.4 and ratio 2 the solver sees error -0.8 instead of 0 and moves both bodies (measured lin -0.299 vs ratio*ang -1.092)"]
fn rack_and_pinion_constraint_is_signed() {
    let mut rack = body(v3(-0.4, 0.0, 0.0), 1.0);
    rack.prev_position = Vec3Fix::ZERO;
    let mut pinion = body(Vec3Fix::ZERO, 1.0);
    pinion.rotation = rz(-0.2);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    let lin = bodies[0].position.x.to_f64();
    let ang = angle_z(bodies[1].rotation);
    assert!(
        (lin - 2.0 * ang).abs() < 1e-3,
        "lin {lin}, ratio*ang {}",
        2.0 * ang
    );
    // and the consistent step is not disturbed
    assert!((lin + 0.4).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// GearJoint
// ---------------------------------------------------------------------------

/// `angle_a + ratio * angle_b = constant`: a solve must reduce the signed
/// residual `angle_a + ratio * angle_b` (per-step rotation about z) to well below its start (half).
#[test]
#[ignore = "known defect: AUD-A-S3W1-015: solve_gear rotates a by +lambda w_a and b by -lambda ratio w_b, so angle_a + ratio*angle_b changes by lambda (w_a - ratio^2 w_b) instead of -lambda (w_a + ratio^2 w_b); measured residual 0.1 -> 0.1 (ratio 1, equal w), the angle read is unsigned (2|xyz|), and joint_a / joint_b hinge indices and axes are never used (z axis fixed)"]
fn gear_solver_reduces_the_signed_angle_residual() {
    for (ang_a, ang_b) in [(0.1, 0.0), (0.0, 0.1), (-0.1, 0.0), (0.1, 0.05)] {
        let mut a = body(Vec3Fix::ZERO, 1.0);
        let mut b = body(Vec3Fix::ZERO, 1.0);
        a.rotation = rz(ang_a);
        b.rotation = rz(ang_b);
        let mut bodies = vec![a, b];
        let j = GearJoint::new(0, 1, 0, 1, fx(1.0));
        let before = (ang_a + 1.0 * ang_b).abs();
        solve(&mut bodies, ExtraJoint::Gear(j), 0.25);
        let after = (angle_z(bodies[0].rotation) + angle_z(bodies[1].rotation)).abs();
        assert!(
            after < 0.5 * before,
            "({ang_a}, {ang_b}): {before} -> {after}"
        );
    }
}

/// No relative motion in either gear: nothing to correct (green).
#[test]
fn gear_with_no_motion_is_inert() {
    let a = body(Vec3Fix::ZERO, 1.0);
    let b = body(Vec3Fix::ZERO, 1.0);
    let mut bodies = vec![a, b];
    let j = GearJoint::new(0, 1, 0, 1, fx(2.0));
    solve(&mut bodies, ExtraJoint::Gear(j), 0.25);
    assert_eq!(bodies[0].rotation, QuatFix::IDENTITY);
    assert_eq!(bodies[1].rotation, QuatFix::IDENTITY);
}

// ---------------------------------------------------------------------------
// MouseJoint
// ---------------------------------------------------------------------------

/// Spring-damper as a position-based step: the force `F = k d` acts for `dt` on mass
/// `m`, so the position change is `F dt^2 / m` (the existing oracle expects `F dt / m`, which is
/// a velocity change in length units).
#[test]
#[ignore = "known defect: AUD-A-S3W1-016: solve_mouse adds impulse * inv_mass = F dt w to the *position* (a velocity-dimension quantity); expected F dt^2 w: k=10, d=1, dt=1/8, m=2 moves 0.3125 instead of 0.01953125"]
fn mouse_spring_moves_the_body_by_f_dt_squared_over_m() {
    let mut bodies = vec![body(Vec3Fix::ZERO, 2.0)];
    let j = MouseJoint::new(0, v3(1.0, 0.0, 0.0), fx(1000.0), fx(10.0), Fix128::ZERO);
    solve(&mut bodies, ExtraJoint::Mouse(j), 1.0 / 16.0);
    let dt = 1.0 / 16.0;
    let want = 10.0 * 1.0 * dt * dt / 2.0;
    let got = bodies[0].position.x.to_f64();
    assert!((got - want).abs() < 1e-12, "moved {got}, want {want}");
}

/// Dimensional consistency: halving `dt` must scale a position-based step by 1/4
/// (`dt^2`), not 1/2.
#[test]
#[ignore = "known defect: AUD-A-S3W1-016: displacement scales with dt (not dt^2); halving dt halves the move"]
fn mouse_step_scales_with_dt_squared() {
    let mk = |dt: f64| {
        let mut bodies = vec![body(Vec3Fix::ZERO, 1.0)];
        let j = MouseJoint::new(0, v3(1.0, 0.0, 0.0), fx(1000.0), fx(10.0), Fix128::ZERO);
        solve(&mut bodies, ExtraJoint::Mouse(j), dt);
        bodies[0].position.x.to_f64()
    };
    let r = mk(0.125) / mk(0.0625);
    assert!((r - 4.0).abs() < 1e-9, "ratio {r}");
}

/// `max_force` caps the magnitude symmetrically, a static body is not moved, and the step
/// direction is toward the target (direction and clamp behaviour, independent of units).
#[test]
fn mouse_clamps_symmetrically_and_ignores_static_bodies() {
    // very stiff spring: clamped. Compare against a second joint with a huge cap scaled
    // to the same cap value: both give the same step
    let capped = {
        let mut bodies = vec![body(Vec3Fix::ZERO, 1.0)];
        let j = MouseJoint::new(0, v3(0.0, 3.0, 4.0), fx(2.0), fx(1000.0), Fix128::ZERO);
        solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
        arr(bodies[0].position)
    };
    let direct = {
        let mut bodies = vec![body(Vec3Fix::ZERO, 1.0)];
        // F = k d = 2 exactly when k = 2/5 (d = 5): same force as the cap
        let j = MouseJoint::new(0, v3(0.0, 3.0, 4.0), fx(1000.0), fx(0.4), Fix128::ZERO);
        solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
        arr(bodies[0].position)
    };
    for k in 0..3 {
        assert!(
            (capped[k] - direct[k]).abs() < 1e-9,
            "axis {k}: {capped:?} vs {direct:?}"
        );
    }
    // direction (0, 0.6, 0.8)
    assert!(capped[0].abs() < 1e-12 && (capped[1] / capped[2] - 0.75).abs() < 1e-9);
    // net negative force (damping beyond the spring) is clamped to -max_force
    let neg = {
        let mut b = body(Vec3Fix::ZERO, 1.0);
        b.velocity = v3(10.0, 0.0, 0.0);
        let mut bodies = vec![b];
        let j = MouseJoint::new(0, v3(1.0, 0.0, 0.0), fx(2.0), fx(1.0), fx(100.0));
        solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
        bodies[0].position.x.to_f64()
    };
    let pos = {
        let mut bodies = vec![body(Vec3Fix::ZERO, 1.0)];
        let j = MouseJoint::new(0, v3(1.0, 0.0, 0.0), fx(2.0), fx(1000.0), Fix128::ZERO);
        solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
        bodies[0].position.x.to_f64()
    };
    assert!((neg + pos).abs() < 1e-9, "symmetric clamp: {neg} vs {pos}");
    // static body
    let mut bodies = vec![stat(Vec3Fix::ZERO)];
    let j = MouseJoint::new(0, v3(1.0, 0.0, 0.0), fx(2.0), fx(10.0), Fix128::ZERO);
    solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
    assert_eq!(bodies[0].position, Vec3Fix::ZERO);
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// A weld whose `local_rotation` equals the actual relative rotation has no angular
/// error: the solve leaves both orientations alone.
#[test]
fn weld_with_matching_local_rotation_is_inert() {
    let mut a = body(Vec3Fix::ZERO, 1.0);
    let mut b = body(Vec3Fix::ZERO, 1.0);
    a.rotation = QuatFix::IDENTITY;
    b.rotation = rz(0.3);
    let mut bodies = vec![a, b];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, rz(0.3));
    assert!(j.compute_torque(&bodies).to_f64().abs() < 1e-12);
    solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
    assert!((angle_z(bodies[0].rotation)).abs() < 1e-12);
    assert!((angle_z(bodies[1].rotation) - 0.3).abs() < 1e-12);
}

/// Compliance softens the angular correction too: more compliance leaves more error.
#[test]
fn weld_compliance_softens_the_angular_correction() {
    let left = |alpha: f64| {
        let mut b = body(Vec3Fix::ZERO, 1.0);
        b.rotation = rz(0.02);
        let mut bodies = vec![stat(Vec3Fix::ZERO), b];
        let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
            .with_compliance(fx(alpha));
        solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
        angle_z(bodies[1].rotation).abs()
    };
    let (rigid, soft) = (left(0.0), left(0.5));
    assert!(rigid < soft && soft < 0.02, "rigid {rigid}, soft {soft}");
}

/// With both bodies free, the two corrections oppose each other so the relative angle
/// shrinks (a identity, b offset by +0.02).
#[test]
fn weld_with_two_free_bodies_reduces_the_relative_angle() {
    let a = body(Vec3Fix::ZERO, 1.0);
    let mut b = body(Vec3Fix::ZERO, 1.0);
    b.rotation = rz(0.02);
    let mut bodies = vec![a, b];
    let j = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    solve(&mut bodies, ExtraJoint::Weld(j), 0.25);
    let rel = (angle_z(bodies[1].rotation) - angle_z(bodies[0].rotation)).abs();
    assert!(rel < 0.02 * 0.9, "relative angle {rel}");
}

/// The rack displacement is signed along its axis: a rack that moved -0.5 with the pinion
/// static is pulled back to where it started.
#[test]
fn rack_and_pinion_negative_rack_motion_is_pulled_back() {
    let mut rack = body(v3(-0.5, 0.0, 0.0), 1.0);
    rack.prev_position = Vec3Fix::ZERO;
    let mut bodies = vec![rack, stat(Vec3Fix::ZERO)];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    assert!(
        bodies[0].position.x.to_f64().abs() < 1e-12,
        "{}",
        bodies[0].position.x.to_f64()
    );
}

/// Only rotation about the pinion axis counts: spin about z with the pinion axis along x
/// is zero angular displacement, so a rack at rest is not disturbed.
#[test]
fn rack_and_pinion_ignores_rotation_about_other_axes() {
    let rack = body(Vec3Fix::ZERO, 1.0);
    let mut pinion = body(Vec3Fix::ZERO, 1.0);
    pinion.rotation = rz(0.2);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(1.0, 0.0, 0.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    assert_eq!(bodies[0].position, Vec3Fix::ZERO);
}

/// With an inverse-inertia about the pinion axis `i = 1`, the rigid projection splits the
/// error `lin - ratio * ang` by `w_lin + ratio^2 w_axis`: rack moved 0.4, ratio 2 ->
/// lambda = 0.4 / (1 + 4) and the rack ends at 0.32.
#[test]
#[ignore = "known defect: AUD-A-S3W1-013: rack-and-pinion (and gear / weld) use |inv_inertia| (sqrt(3) for isotropic i = 1) instead of the inverse inertia about the axis, rack ends at 0.3495 instead of 0.32"]
fn rack_and_pinion_uses_the_axis_inverse_inertia() {
    let mut rack = body(v3(0.4, 0.0, 0.0), 1.0);
    rack.prev_position = Vec3Fix::ZERO;
    let pinion = body(Vec3Fix::ZERO, 1.0);
    let mut bodies = vec![rack, pinion];
    let j = RackAndPinionJoint::new(0, 1, v3(1.0, 0.0, 0.0), v3(0.0, 0.0, 1.0), fx(2.0));
    solve(&mut bodies, ExtraJoint::RackAndPinion(j), 0.25);
    assert!(
        (bodies[0].position.x.to_f64() - 0.32).abs() < 1e-9,
        "{}",
        bodies[0].position.x.to_f64()
    );
}

/// Mouse clamp: a net force between `max` and `2 max` is clamped to `max`, and the step is
/// inversely proportional to mass.
#[test]
fn mouse_clamp_and_mass_scaling() {
    let run = |mass: f64, k: f64, max: f64| {
        let mut bodies = vec![body(Vec3Fix::ZERO, mass)];
        let j = MouseJoint::new(0, v3(5.0, 0.0, 0.0), fx(max), fx(k), Fix128::ZERO);
        solve(&mut bodies, ExtraJoint::Mouse(j), 0.25);
        bodies[0].position.x.to_f64()
    };
    // F = 0.6 * 5 = 3 > max 2: clamps to the same step as F = 2 (k = 0.4)
    assert!((run(1.0, 0.6, 2.0) - run(1.0, 0.4, 100.0)).abs() < 1e-12);
    // double mass, half the step
    assert!((run(2.0, 0.4, 100.0) * 2.0 - run(1.0, 0.4, 100.0)).abs() < 1e-12);
}
