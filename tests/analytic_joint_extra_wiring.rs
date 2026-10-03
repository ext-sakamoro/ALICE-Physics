//! Oracles for `alice_physics::joint_extra`'s `ExtraJoint`, `MouseJoint::set_target`,
//! `solve_extra_joints`, `PulleyJoint::total_length`, and `WeldJoint::with_break_torque`
//! (and, for the name-collision backstory documented below, `WeldJoint::with_break_force`).
//! Production entry point: `examples/joint_extra_wiring.rs`.
//!
//! # Closed forms
//!
//! * **`PulleyJoint::total_length`**: with both anchors at each body's local
//!   origin and identity rotations, `world_a == body_a.position`, so
//!   `len_a` / `len_b` are plain Euclidean distances to the ground anchors.
//!   `Fix128::sqrt` is a documented exact `floor(sqrt(x))`, so a perfect-square
//!   radicand returns the exact integer root, and `total_length = len_a +
//!   ratio * len_b` is then a single exact multiply-add.
//! * **`MouseJoint::set_target` / `solve_extra_joints` (Mouse variant)**: from
//!   rest (`velocity = 0`), one XPBD step moves the body by `direction *
//!   (stiffness * distance * dt) * inv_mass`, clamped to `max_force`.
//! * **`WeldJoint::with_break_force`**: `compute_force` is a plain vector
//!   `.length()` (no division), so an axis-aligned separation is exact. The
//!   break test is strict (`separation > break_force`). When the weld holds
//!   and body A is static (`inv_mass_a = 0`), the whole correction lands on
//!   B: `correction * inv_mass_b == distance` exactly, landing B on A's
//!   anchor in one solve call -- provided the separation is a power of two
//!   so `normalize_with_length`'s `1/length` division is itself exact.
//! * **`WeldJoint::with_break_torque`**: with `local_rotation` and
//!   `body_a.rotation` both `QuatFix::IDENTITY`, `compute_torque` reduces to
//!   the magnitude of `body_b.rotation`'s `(x, y, z)` part. A rotation of
//!   `pi/3` about a unit axis gives `sin(pi/6) = 1/2` exactly in real
//!   arithmetic; `sin_cos` is a 48-iteration CORDIC routine (documented
//!   `~2^-48` error per call), so assertions use a `2^-40` tolerance.
//!
//! # Degenerate input
//!
//! * `PulleyJoint::total_length` with `ratio = 0`: the second rope drops out
//!   of the sum entirely (`total_length == len_a`).
//! * `PulleyJoint::total_length` with a zero-length rope (an anchor already
//!   coincident with its ground anchor): that rope's term is exactly zero,
//!   not NaN or a panic (`Vec3Fix::length()` of a zero vector is `ZERO`).
//! * `MouseJoint::set_target` to the body's own current position: zero
//!   distance triggers `solve_mouse`'s early return, so `solve_extra_joints`
//!   applies no correction at all (not an infinitesimal one).
//! * `WeldJoint::with_break_force(Fix128::ZERO)`: a joint already satisfied
//!   (`separation == 0`) does **not** break (the comparison is strict `>`),
//!   but any nonzero separation breaks immediately regardless of magnitude.
//! * `WeldJoint::with_break_torque(Fix128::ZERO)`: identical shape -- a
//!   rotation already at `IDENTITY` (`torque == 0`) does not break, but any
//!   nonzero relative rotation does.
//! * `WeldJoint::with_break_force` at the exact boundary (`separation ==
//!   break_force`): must NOT break (`>`, not `>=`); the solver still runs
//!   the positional correction as if unbroken.
//! * Broken weld joints: [`solve_extra_joints`] is documented to skip (not
//!   remove) broken welds -- a joint still reports [`WeldJoint::is_broken`]
//!   `true` after a solve call, and the slice length is unchanged (it is a
//!   `&[ExtraJoint]`, which cannot shrink through this API in any case, but
//!   this pins that no element is mutated into a different variant either).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::joint_extra::{
    solve_extra_joints, ExtraJoint, MouseJoint, PulleyJoint, WeldJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// `2^-40`: 256x the documented `~2^-48` per-call CORDIC error, for the one
/// `sin_cos` call inside `QuatFix::from_axis_angle`.
fn cordic_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24) // 2^24 / 2^64 = 2^-40
}

fn near(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    (a - b).abs() < tol
}

/// Static A at the origin and a dynamic B with unit mass and unit inverse
/// inertia, both at the origin.
fn pair() -> Vec<RigidBody> {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    vec![RigidBody::new_static(Vec3Fix::ZERO), b]
}

// ===========================================================================
// A. ExtraJoint enum
// ===========================================================================

#[test]
fn extra_joint_variants_round_trip_through_a_match() {
    let o = Vec3Fix::ZERO;
    let cases = [
        (
            ExtraJoint::Pulley(PulleyJoint::new(0, 1, o, o, o, o, Fix128::ONE)),
            "Pulley",
        ),
        (
            ExtraJoint::Gear(alice_physics::joint_extra::GearJoint::new(
                0,
                1,
                0,
                1,
                Fix128::ONE,
            )),
            "Gear",
        ),
        (
            ExtraJoint::Weld(WeldJoint::new(0, 1, o, o, QuatFix::IDENTITY)),
            "Weld",
        ),
        (
            ExtraJoint::RackAndPinion(alice_physics::joint_extra::RackAndPinionJoint::new(
                0,
                1,
                Vec3Fix::UNIT_X,
                Vec3Fix::UNIT_Z,
                Fix128::ONE,
            )),
            "RackAndPinion",
        ),
        (
            ExtraJoint::Mouse(MouseJoint::new(
                0,
                o,
                Fix128::ONE,
                Fix128::ONE,
                Fix128::ZERO,
            )),
            "Mouse",
        ),
    ];
    for (joint, expected) in cases {
        let kind = match joint {
            ExtraJoint::Pulley(_) => "Pulley",
            ExtraJoint::Gear(_) => "Gear",
            ExtraJoint::Weld(_) => "Weld",
            ExtraJoint::RackAndPinion(_) => "RackAndPinion",
            ExtraJoint::Mouse(_) => "Mouse",
        };
        assert_eq!(kind, expected);
    }
}

// ===========================================================================
// B. PulleyJoint::total_length
// ===========================================================================

#[test]
fn total_length_exact_pythagorean_closed_form() {
    let bodies = [
        RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(26, 8, 0), Fix128::ONE),
    ];
    let pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(20, 0, 0),
        q(3, 2),
    );
    // len_a = |(3,4,0)| = 5, len_b = |(6,8,0)| = 10, total = 5 + 1.5*10 = 20
    assert_eq!(pulley.total_length(&bodies), Fix128::from_int(20));
}

#[test]
fn total_length_ratio_zero_drops_rope_b() {
    let bodies = [
        RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(26, 8, 0), Fix128::ONE),
    ];
    let pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(20, 0, 0),
        Fix128::ZERO,
    );
    assert_eq!(pulley.total_length(&bodies), Fix128::from_int(5));
}

#[test]
fn total_length_zero_length_rope_a_contributes_zero() {
    let bodies = [
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE), // coincides with ground_anchor_a
        RigidBody::new(Vec3Fix::from_int(26, 8, 0), Fix128::ONE),
    ];
    let pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO, // == bodies[0].position -> len_a = 0
        Vec3Fix::from_int(20, 0, 0),
        q(3, 2),
    );
    assert_eq!(pulley.total_length(&bodies), Fix128::from_int(15));
}

#[test]
fn total_length_both_ropes_zero_length_is_zero() {
    let bodies = [
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
    ];
    let pulley = PulleyJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        q(3, 2),
    );
    assert_eq!(pulley.total_length(&bodies), Fix128::ZERO);
}

// ===========================================================================
// C. MouseJoint::set_target + solve_extra_joints
// ===========================================================================

#[test]
fn set_target_then_solve_moves_body_by_exact_spring_impulse() {
    let dt = q(1, 4);
    let mut mj = MouseJoint::new(
        1,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        Fix128::ONE,
        Fix128::ZERO,
    );
    mj.set_target(Vec3Fix::from_int(8, 0, 0));
    assert_eq!(mj.target_position, Vec3Fix::from_int(8, 0, 0));

    let joints = [ExtraJoint::Mouse(mj)];
    let mut bodies = pair();
    solve_extra_joints(&mut bodies, &joints, dt);
    // distance=8, direction=(1,0,0), spring_force=stiffness*distance=8,
    // damping_force=0 (v=0), clamped_force=8 (< max_force=100),
    // impulse=(8*dt,0,0)=(2,0,0), position += impulse*inv_mass(1).
    assert_eq!(bodies[1].position, Vec3Fix::from_int(2, 0, 0));
}

#[test]
fn set_target_overwrites_the_constructor_value() {
    let mut mj = MouseJoint::new(
        1,
        Vec3Fix::from_int(99, 99, 99),
        Fix128::ONE,
        Fix128::ONE,
        Fix128::ZERO,
    );
    mj.set_target(Vec3Fix::from_int(1, 2, 3));
    assert_eq!(mj.target_position, Vec3Fix::from_int(1, 2, 3));
}

#[test]
fn set_target_to_current_position_zero_distance_no_correction() {
    let dt = q(1, 4);
    let mut mj = MouseJoint::new(
        0,
        Vec3Fix::from_int(5, 0, 0),
        Fix128::from_int(100),
        Fix128::ONE,
        Fix128::ZERO,
    );
    mj.set_target(Vec3Fix::ZERO); // == the body's own position below
    let joints = [ExtraJoint::Mouse(mj)];
    let mut bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];
    solve_extra_joints(&mut bodies, &joints, dt);
    assert_eq!(
        bodies[0].position,
        Vec3Fix::ZERO,
        "zero distance must trigger solve_mouse's early return, not an infinitesimal correction"
    );
}

#[test]
fn solve_extra_joints_clamps_to_max_force() {
    let dt = q(1, 4);
    // Far target, tiny max_force: the raw spring force would be huge, but the
    // applied impulse must be bounded by max_force*dt*inv_mass exactly.
    // max_force = 1/128 is dyadic (a power-of-two denominator), so the clamp
    // product `max_force * dt` is an exact bit shift with no rounding, unlike
    // a non-dyadic fraction such as 1/100 (whose Fix128 encoding is already
    // an approximation, so multiplying it by dt would not reproduce an
    // independently-computed `from_ratio` reference bit-for-bit). The target
    // distance (512) is also a power of two so `normalize_with_length`'s
    // `1/length` division -- and therefore `direction` -- is exact too
    // (a non-power-of-two distance such as 1000 would make `direction`
    // approximate, since `1000 * (1/1000)` does not round-trip to exactly
    // `1` in `Fix128`, which otherwise swamps this test's exactness by
    // roughly 1 ULP).
    let mut mj = MouseJoint::new(
        1,
        Vec3Fix::ZERO,
        q(1, 128), // max_force
        Fix128::from_int(1000),
        Fix128::ZERO,
    );
    mj.set_target(Vec3Fix::from_int(512, 0, 0));
    let joints = [ExtraJoint::Mouse(mj)];
    let mut bodies = pair();
    solve_extra_joints(&mut bodies, &joints, dt);
    // clamped_force = max_force = 1/128, impulse = (1/128 * 1/4, 0, 0) = (1/512, 0, 0)
    assert_eq!(
        bodies[1].position,
        Vec3Fix::new(q(1, 512), Fix128::ZERO, Fix128::ZERO)
    );
}

// ===========================================================================
// D. WeldJoint::with_break_force (name-collision backstory item)
// ===========================================================================

#[test]
fn with_break_force_sets_the_option_some() {
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::from_int(5));
    assert_eq!(wj.break_force, Some(Fix128::from_int(5)));
    assert_eq!(
        wj.break_torque, None,
        "with_break_force must not touch break_torque"
    );
}

#[test]
fn with_break_force_strict_boundary_does_not_break_at_equality() {
    let bodies_at = |sep: i64| {
        let mut b = pair();
        b[1].position = Vec3Fix::from_int(sep, 0, 0);
        b
    };
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::from_int(4));
    assert!(
        !wj.is_broken(&bodies_at(4)),
        "separation == break_force must not break (strict >)"
    );
    assert!(
        wj.is_broken(&bodies_at(5)),
        "separation > break_force must break"
    );
    assert!(
        !wj.is_broken(&bodies_at(3)),
        "separation < break_force must not break"
    );
}

#[test]
fn with_break_force_zero_threshold_boundary() {
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::ZERO);
    let touching = pair(); // separation == 0
    assert!(
        !wj.is_broken(&touching),
        "an already-satisfied joint (separation 0) must not break even with threshold 0"
    );
    let mut separated = pair();
    separated[1].position = Vec3Fix::from_int(4, 0, 0);
    assert!(
        wj.is_broken(&separated),
        "any nonzero separation must break a threshold-0 weld"
    );
}

#[test]
fn with_break_force_broken_weld_skipped_by_solve_but_not_removed() {
    let dt = q(1, 4);
    let mut bodies = pair();
    bodies[1].position = Vec3Fix::from_int(4, 0, 0);
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::from_int(3)); // 4 > 3 -> broken
    assert!(wj.is_broken(&bodies));

    let joints = [ExtraJoint::Weld(wj)];
    let before = bodies[1].position;
    solve_extra_joints(&mut bodies, &joints, dt);
    assert_eq!(
        bodies[1].position, before,
        "a broken weld must not move body B"
    );
    assert_eq!(
        joints.len(),
        1,
        "solve_extra_joints must not remove the broken joint from the slice"
    );
    assert!(
        matches!(joints[0], ExtraJoint::Weld(_)),
        "the broken joint must remain a Weld variant, not be mutated into something else"
    );
    // Re-querying is_broken on the same (unmoved) bodies must still report
    // broken: solve_extra_joints does not clear or cache the broken state.
    assert!(wj.is_broken(&bodies));
}

#[test]
fn with_break_force_holding_weld_lands_exactly_on_anchor() {
    let dt = q(1, 4);
    let mut bodies = pair();
    bodies[1].position = Vec3Fix::from_int(4, 0, 0); // power of two -> exact 1/length
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_force(Fix128::from_int(5)); // 4 < 5 -> holds
    assert!(!wj.is_broken(&bodies));

    let joints = [ExtraJoint::Weld(wj)];
    solve_extra_joints(&mut bodies, &joints, dt);
    // A static (inv_mass=0): w_sum = inv_mass_b = 1, inv_w_sum = 1,
    // lambda = distance*1 = 4, correction = (4,0,0), B moves by
    // -correction*inv_mass_b = -(4,0,0) -> lands exactly on (0,0,0).
    assert_eq!(bodies[1].position, Vec3Fix::ZERO);
}

// ===========================================================================
// E. WeldJoint::with_break_torque
// ===========================================================================

#[test]
fn with_break_torque_sets_the_option_some() {
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(Fix128::from_int(7));
    assert_eq!(wj.break_torque, Some(Fix128::from_int(7)));
    assert_eq!(
        wj.break_force, None,
        "with_break_torque must not touch break_force"
    );
}

#[test]
fn compute_torque_sin_pi_over_6_closed_form() {
    let mut bodies = pair();
    let angle = Fix128::PI / Fix128::from_int(3); // theta = pi/3 about +Z
    bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY);
    let torque = wj.compute_torque(&bodies);
    assert!(
        near(torque, q(1, 2), cordic_tol()),
        "compute_torque should equal sin(pi/6)=1/2 within CORDIC tolerance, got {}",
        torque.to_f64()
    );
}

#[test]
fn with_break_torque_breaks_above_and_holds_below_sin_pi_over_6() {
    let angle = Fix128::PI / Fix128::from_int(3);
    let bodies_rotated = {
        let mut b = pair();
        b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
        b
    };
    let holds = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(q(3, 4)); // 3/4 > ~0.5
    assert!(!holds.is_broken(&bodies_rotated));
    let breaks = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(q(1, 4)); // 1/4 < ~0.5
    assert!(breaks.is_broken(&bodies_rotated));
}

#[test]
fn with_break_torque_zero_threshold_boundary() {
    let identity_bodies = pair(); // rotation stays IDENTITY -> torque = 0
    let wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(Fix128::ZERO);
    assert!(
        !wj.is_broken(&identity_bodies),
        "identity relative rotation (torque 0) must not break even with threshold 0"
    );

    let angle = Fix128::PI / Fix128::from_int(3);
    let mut rotated_bodies = pair();
    rotated_bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
    assert!(
        wj.is_broken(&rotated_bodies),
        "any nonzero relative rotation must break a threshold-0 weld"
    );
}

#[test]
fn solve_extra_joints_skips_angular_correction_when_broken_applies_when_holding() {
    let dt = q(1, 4);
    let angle = Fix128::PI / Fix128::from_int(3);

    // Breaks: threshold below the ~0.5 torque -> rotation untouched.
    let mut broken_bodies = pair();
    broken_bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
    let breaking_wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(q(1, 4));
    let before = broken_bodies[1].rotation;
    solve_extra_joints(&mut broken_bodies, &[ExtraJoint::Weld(breaking_wj)], dt);
    assert_eq!(
        broken_bodies[1].rotation, before,
        "broken weld must not apply angular correction"
    );

    // Holds: threshold above the ~0.5 torque -> rotation must be corrected.
    let mut holding_bodies = pair();
    holding_bodies[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, angle);
    let holding_wj = WeldJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_break_torque(q(3, 4));
    let before = holding_bodies[1].rotation;
    solve_extra_joints(&mut holding_bodies, &[ExtraJoint::Weld(holding_wj)], dt);
    assert_ne!(
        holding_bodies[1].rotation, before,
        "a holding weld must apply the angular correction toward local_rotation"
    );
}

// ===========================================================================
// F. Mixed-variant solve (ExtraJoint dispatch sanity)
// ===========================================================================

#[test]
fn solve_extra_joints_dispatches_every_variant_without_panicking() {
    let dt = q(1, 4);
    let o = Vec3Fix::ZERO;
    let mut bodies = vec![
        RigidBody::new_static(o),
        RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(0, 0, 1), Fix128::ONE),
        RigidBody::new(o, Fix128::ONE),
    ];
    let joints = [
        ExtraJoint::Pulley(PulleyJoint::new(
            1,
            2,
            o,
            o,
            Vec3Fix::from_int(5, 0, 0),
            Vec3Fix::from_int(0, 5, 0),
            Fix128::ONE,
        )),
        ExtraJoint::Gear(alice_physics::joint_extra::GearJoint::new(
            1,
            2,
            0,
            1,
            Fix128::ONE,
        )),
        ExtraJoint::Weld(WeldJoint::new(0, 3, o, o, QuatFix::IDENTITY)),
        ExtraJoint::RackAndPinion(alice_physics::joint_extra::RackAndPinionJoint::new(
            1,
            2,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Z,
            Fix128::ONE,
        )),
        ExtraJoint::Mouse(MouseJoint::new(
            4,
            Vec3Fix::from_int(3, 3, 3),
            Fix128::from_int(10),
            Fix128::ONE,
            Fix128::ZERO,
        )),
    ];
    for _ in 0..5 {
        solve_extra_joints(&mut bodies, &joints, dt);
    }
    // No closed-form assertion here -- this test's job is dispatch coverage
    // (every ExtraJoint variant goes through one solve_extra_joints call
    // without panicking); the per-variant closed forms are pinned above.
    for b in &bodies {
        assert!(!b.position.x.to_f64().is_nan());
    }
}
