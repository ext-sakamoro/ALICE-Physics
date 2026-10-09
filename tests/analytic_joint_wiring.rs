//! Oracles for the `joint` breakable-joint path (`JointType` / `joint_type`
//! / `break_force` / `solve_joints_breakable` / `with_break_force` on every
//! joint type) and the limit / motion builders (`with_limits`,
//! `with_linear_limits`, `with_angular_limits`, `with_linear_motion`,
//! `with_angular_motion`). Production entry point: `examples/joint_limits_and_breaking.rs`.
//!
//! # Closed forms
//!
//! * **Breakable ramp**: a rigid ball joint (compliance 0) between a static
//!   body A and a dynamic body B (`inv_mass = 1/m`) with B placed at the
//!   *predicted* separation `d_k = k·F0·dt²/m` for a hypothetical constant
//!   pull `F_k = k·F0` applied from rest over one step of size `dt`: solving
//!   the one-constraint scene computes `λ = d_k·m` and the constraint force
//!   `λ/dt² = k·F0` exactly (`dt = 1/4`, `m = 2`, `F0 = 1` are all dyadic, so
//!   every quantity is bit-exact in `Fix128`). `solve_joints_breakable`
//!   compares the **pre-solve separation** `d_k`, not `λ/dt²`, against
//!   `break_force`, so a force threshold `F_b` corresponds to a configured
//!   `break_force = F_b·dt²/m`, and the break is strict (`>`): the scene
//!   holds through `d_k == break_force` and breaks on the first `d_k >
//!   break_force`.
//! * **Linear / angular limit landing**: with A static and B carrying
//!   `inv_mass = w_b` (and unit inverse inertia for the angular cases), one
//!   XPBD solve moves B by exactly the signed overshoot past the violated
//!   bound (`correction = error · (1/w_sum) · inv_mass_b` collapses to
//!   `error` when `w_sum == inv_mass_b`), so B lands exactly on the bound.
//!   `D6Motion::Locked` is the `error = raw projection` case (bound 0),
//!   `Free` never computes an error, `Limited` is the general case.
//! * **Cone / twist limit landing**: `cone_angle = atan2(|a×z|, a·z)` with
//!   `a` the rotated twist axis, and `twist_angle = 2·atan2(q_z, q_w)` on
//!   the `w ≥ 0` cover — both independent of `joint::compute_twist_angle` /
//!   the solver's own cone-angle computation, so a landing test that reads
//!   these back pins the solver's angle convention, not just its magnitude.
//!   `sin_cos` / `atan2` are 48-iteration CORDIC (documented ≈ 2⁻⁴⁸ per
//!   call), so two chained CORDIC calls (one inside the solver, one in this
//!   oracle) justify a tolerance of `2⁻⁴⁴` (16 ULP of the per-call error),
//!   not bit-exact equality, for the angular landing cases.
//!
//! # Degenerate input
//!
//! * `break_force = Fix128::ZERO`: the comparison is strict `separation >
//!   break_force`, so a joint already satisfied (`separation == 0`) does
//!   **not** break. The smallest separation `compute_force` can actually
//!   *report* as nonzero is `2⁻³²`, not `Fix128`'s smallest representable
//!   value `2⁻⁶⁴`: squaring `2⁻⁶⁴` underflows a single lane to exactly `0`
//!   (`length_squared` loses it before `sqrt` ever runs), while `2⁻³²`
//!   squares to exactly `2⁻⁶⁴` and its `sqrt` round-trips back to `2⁻³²` bit
//!   for bit — that value **does** break a `break_force = 0` joint.
//! * `break_force` negative: `separation` from [`Joint::compute_force`] is a
//!   vector length, so it can never be negative; a separation of exactly
//!   `0` still satisfies `0 > break_force` when `break_force < 0`, so a
//!   negative threshold breaks the joint unconditionally, even when the
//!   anchors already coincide.
//! * `with_limits(lo, hi)` / `with_linear_limits` / `with_angular_limits`
//!   with `lo > hi`: these builders now panic at construction (see their
//!   `# Panics`), so this interval cannot reach the solver at all — there
//!   is no "inverted interval" case left to pin here.
//! * `lo == hi`: both branches converge on the same point, so every input
//!   outside that single value lands exactly on it, from either side.
//! * Extreme separations whose square overflows a single `Fix128` lane: the
//!   crate's own `checked_mul` doc (WM-01 / doctrine B-12) establishes that
//!   plain (wrapping) `Mul` on two values whose product does not fit folds
//!   the result **to exactly zero** whenever the combined magnitude shift is
//!   a multiple of 2⁶⁴ (this file re-derives the same fact independently
//!   from the documented 128×128→128 formula, without calling
//!   [`alice_physics::math::Fix128::mul`]). `Joint::compute_force` and
//!   `solve_joints_breakable` use plain `Mul`, not `checked_mul`, so a joint
//!   whose bodies are placed at such a separation reports `compute_force ==
//!   0` and **never breaks**, regardless of `break_force` — this file pins
//!   that as the implemented behavior (not a panic) and flags it as a fact
//!   for the caller in the report, since it is outside this module's scope
//!   to fix.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::joint::{
    solve_joints, solve_joints_breakable, BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint,
    HingeJoint, Joint, JointType, SliderJoint, SpringJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn near(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    (a - b).abs() < tol
}

/// 2⁻⁴⁴ (16 ULP of a single 48-iteration CORDIC call), for angular landing
/// assertions that chain a CORDIC solve with a CORDIC readback.
fn cordic_tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 20) // 2^20 / 2^64 = 2^-44
}

/// Static A at the origin and a dynamic B with the given inverse mass and
/// (for angular cases) unit inverse inertia, both at the origin.
fn pair(inv_mass_b: Fix128) -> Vec<RigidBody> {
    let mut b = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    b.inv_mass = inv_mass_b;
    b.inv_inertia = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    vec![RigidBody::new_static(Vec3Fix::ZERO), b]
}

/// Angle between the rotated +z axis and +z, independent of the solver's
/// own cone-angle computation: `atan2(|a×z|, a·z)`.
fn cone_angle_from_z(rot: QuatFix) -> Fix128 {
    let a = rot.rotate_vec(Vec3Fix::UNIT_Z);
    Fix128::atan2(a.cross(Vec3Fix::UNIT_Z).length(), a.dot(Vec3Fix::UNIT_Z))
}

const DT: Fix128 = Fix128 { hi: 0, lo: 1 << 62 }; // 1/4

// ===========================================================================
// A. JointType / joint_type / break_force on every joint type
// ===========================================================================

#[test]
fn every_joint_type_reports_its_kind_and_break_force() {
    let o = Vec3Fix::ZERO;
    let threshold = q(5, 32);
    let cases: [(Joint, JointType); 7] = [
        (
            Joint::Ball(BallJoint::new(0, 1, o, o).with_break_force(threshold)),
            JointType::Ball,
        ),
        (
            Joint::Hinge(
                HingeJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
                    .with_limits(-q(1, 2), q(1, 2))
                    .with_break_force(threshold),
            ),
            JointType::Hinge,
        ),
        (
            Joint::Fixed(
                FixedJoint::new(0, 1, o, o, QuatFix::IDENTITY).with_break_force(threshold),
            ),
            JointType::Fixed,
        ),
        (
            Joint::Slider(
                SliderJoint::new(0, 1, Vec3Fix::UNIT_X, o, o)
                    .with_limits(-q(1, 2), Fix128::ONE)
                    .with_break_force(threshold),
            ),
            JointType::Slider,
        ),
        (
            Joint::Spring(
                SpringJoint::new(0, 1, o, o, Fix128::ONE, Fix128::from_int(4), Fix128::ZERO)
                    .with_break_force(threshold),
            ),
            JointType::Spring,
        ),
        (
            Joint::D6(
                D6Joint::new(0, 1, o, o)
                    .with_linear_motion(D6Motion::Locked, D6Motion::Limited, D6Motion::Free)
                    .with_linear_limits(
                        Vec3Fix::new(-q(1, 2), -q(1, 2), -q(1, 2)),
                        Vec3Fix::new(q(1, 2), Fix128::ONE, q(1, 2)),
                    )
                    .with_angular_motion(D6Motion::Free, D6Motion::Free, D6Motion::Limited)
                    .with_angular_limits(
                        Vec3Fix::new(-q(1, 2), -q(1, 2), -q(1, 2)),
                        Vec3Fix::new(q(1, 2), q(1, 2), q(1, 2)),
                    )
                    .with_break_force(threshold),
            ),
            JointType::D6,
        ),
        (
            Joint::ConeTwist(
                ConeTwistJoint::new(0, 1, o, o, Vec3Fix::UNIT_Z, Vec3Fix::UNIT_Z)
                    .with_limits(q(1, 2), q(1, 4))
                    .with_break_force(threshold),
            ),
            JointType::ConeTwist,
        ),
    ];
    for (joint, expected_kind) in cases {
        assert_eq!(joint.joint_type(), expected_kind, "{expected_kind:?}");
        assert_eq!(
            joint.break_force(),
            Some(threshold),
            "{expected_kind:?} did not carry its break_force"
        );
    }
}

/// `with_angular_motion`'s first parameter (`x`) is `D6Motion::Free` in
/// every other test in this crate that exercises the builder (both the
/// existing `src/joint.rs` unit test and `every_joint_type_reports_its_kind_and_break_force`
/// above pass `Free` for `x`), which happens to equal the field's own
/// default — a mutant that hard-codes `angular_x = D6Motion::Free` (ignoring
/// the parameter) is invisible to every one of those scenes. This test
/// gives `x` a non-default value so the parameter is actually load-bearing.
#[test]
fn with_angular_motion_x_axis_parameter_is_not_ignored() {
    let o = Vec3Fix::ZERO;
    let j = D6Joint::new(0, 1, o, o).with_angular_motion(
        D6Motion::Locked,
        D6Motion::Free,
        D6Motion::Free,
    );
    assert_eq!(
        j.angular_x,
        D6Motion::Locked,
        "x must take the value passed in, not the default"
    );
}

#[test]
fn joints_without_with_break_force_report_none() {
    let o = Vec3Fix::ZERO;
    assert_eq!(Joint::Ball(BallJoint::new(0, 1, o, o)).break_force(), None);
    assert_eq!(
        Joint::D6(D6Joint::new(0, 1, o, o)).break_force(),
        None,
        "D6Joint::new must default break_force to None"
    );
}

// ===========================================================================
// B. Breakable ramp: exact bit closed form
// ===========================================================================

#[test]
fn solve_joints_breakable_ramp_breaks_exactly_at_k_star() {
    let m = Fix128::from_int(2);
    let f0 = Fix128::ONE;
    let fb_force = Fix128::from_int(5);
    let configured = fb_force * DT * DT / m; // 5/32, dyadic, exact
    assert_eq!(configured, q(5, 32));

    let ramp = [Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(configured),
    )];
    let mut broke_at: Option<i64> = None;
    for k in 1..=8i64 {
        let d_k = Fix128::from_int(k) * f0 * DT * DT / m;
        let mut bodies = pair(Fix128::ONE / m);
        bodies[1].position = Vec3Fix::new(Fix128::ZERO, -d_k, Fix128::ZERO);
        let lambda = d_k * m;
        let force = lambda / (DT * DT);
        assert_eq!(
            force,
            Fix128::from_int(k),
            "λ/dt² must equal F_k exactly at k={k}"
        );
        let broken = solve_joints_breakable(&ramp, &mut bodies, DT);
        if k <= 5 {
            assert!(
                broken.is_empty(),
                "k={k} <= 5 must hold (equality does not break)"
            );
            assert_eq!(
                bodies[1].position,
                Vec3Fix::ZERO,
                "held joint must solve back onto the anchor exactly, k={k}"
            );
        } else {
            assert_eq!(broken, vec![0], "k={k} > 5 must break");
            assert_eq!(
                bodies[1].position,
                Vec3Fix::new(Fix128::ZERO, -d_k, Fix128::ZERO),
                "broken joint must not be solved, k={k}"
            );
            if broke_at.is_none() {
                broke_at = Some(k);
            }
        }
    }
    assert_eq!(
        broke_at,
        Some(6),
        "first break must be at k* = floor(F_b/F0) + 1 = 6"
    );
}

#[test]
fn solve_joints_non_breakable_variant_never_removes_joints() {
    // Same scene as the k=8 break case, but through `solve_joints` (not the
    // `_breakable` variant): the joint has no way to report "broken", and it
    // keeps solving (pulling B back) regardless of separation.
    let configured = q(5, 32);
    let m = Fix128::from_int(2);
    let d_8 = Fix128::from_int(8) * Fix128::ONE * DT * DT / m;
    let joint = [Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(configured),
    )];
    let mut bodies = pair(Fix128::ONE / m);
    bodies[1].position = Vec3Fix::new(Fix128::ZERO, -d_8, Fix128::ZERO);
    solve_joints(&joint, &mut bodies, DT);
    assert_eq!(
        bodies[1].position,
        Vec3Fix::ZERO,
        "solve_joints must solve the constraint even past break_force (it has no break logic)"
    );
}

// ===========================================================================
// C.1 break_force degenerate values: zero and negative
// ===========================================================================

#[test]
fn break_force_zero_is_a_strict_boundary() {
    let joint_zero = [Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::ZERO),
    )];
    // separation exactly 0 (anchors coincide): 0 > 0 is false -> does not break
    let mut satisfied = pair(Fix128::ONE);
    let broken = solve_joints_breakable(&joint_zero, &mut satisfied, DT);
    assert!(
        broken.is_empty(),
        "separation 0 must not exceed break_force 0 (strict >)"
    );

    // A separation whose *square* is still representable: `2^-64` itself
    // squares to `2^-128`, which underflows a single `Fix128` lane to exactly
    // zero (so `compute_force` would wrongly read 0, not "the smallest
    // nonzero input"); `2^-32` squares to exactly `2^-64` (the smallest
    // nonzero `Fix128` value) and its `sqrt` round-trips to `2^-32` bit for
    // bit, so it is the smallest separation `compute_force` can report.
    let smallest_representable = Fix128::from_raw(0, 1 << 32); // 2^-32
    let mut tiny = pair(Fix128::ONE);
    tiny[1].position = Vec3Fix::new(smallest_representable, Fix128::ZERO, Fix128::ZERO);
    assert_eq!(joint_zero[0].compute_force(&tiny), smallest_representable);
    let broken = solve_joints_breakable(&joint_zero, &mut tiny, DT);
    assert_eq!(
        broken,
        vec![0],
        "any positive representable separation must exceed break_force 0"
    );
}

#[test]
fn negative_break_force_breaks_unconditionally_even_at_zero_separation() {
    let joint_neg = [Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(-Fix128::ONE),
    )];
    let mut satisfied = pair(Fix128::ONE); // anchors coincide, separation == 0
    let broken = solve_joints_breakable(&joint_neg, &mut satisfied, DT);
    assert_eq!(
        broken,
        vec![0],
        "compute_force is a length (>= 0); `0 > -1` is true, so a negative \
         break_force always breaks, even with the joint already satisfied"
    );
}

// ===========================================================================
// C.2 degenerate limit intervals
//
// An inverted interval (min > max) can no longer reach these solver tests:
// `SliderJoint::with_limits` / `HingeJoint::with_limits` now panic at
// construction when `min > max` (see their rustdoc `# Panics`), so the
// prior pair of tests here documenting that the solver's `min` branch wins
// an inverted interval ("not swapped") has no construction left to reach —
// removed, not relaxed, along with the interval they pinned.
// ===========================================================================

#[test]
fn slider_limit_hi_equals_lo_is_a_single_point_from_either_side() {
    let point = q(3, 4);
    let joint = [Joint::Slider(
        SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO)
            .with_limits(point, point),
    )];
    for travel in [Fix128::from_int(10), Fix128::from_int(-10)] {
        let mut b = pair(q(1, 2));
        b[1].position = Vec3Fix::new(travel, Fix128::ZERO, Fix128::ZERO);
        solve_joints(&joint, &mut b, DT);
        assert_eq!(
            b[1].position.x, point,
            "travel={travel:?} must land on the single point"
        );
    }
}

#[test]
fn d6_linear_limited_axis_with_lo_equal_to_hi() {
    let point = Vec3Fix::new(q(1, 4), Fix128::ZERO, Fix128::ZERO);
    let mut j = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO);
    j.linear_x = D6Motion::Limited;
    j.linear_limit_min = point;
    j.linear_limit_max = point;
    for travel in [Fix128::from_int(5), Fix128::from_int(-5)] {
        let mut b = pair(q(1, 2));
        b[1].position = Vec3Fix::new(travel, Fix128::ZERO, Fix128::ZERO);
        solve_joints(&[Joint::D6(j)], &mut b, DT);
        assert_eq!(b[1].position.x, point.x, "travel={travel:?}");
    }
}

// ===========================================================================
// C.2b inverted limit intervals are refused at construction
//
// `ConeTwistJoint::with_limits` is not covered by the first four tests:
// `cone_limit` and `twist_limit` are each a single symmetric magnitude, not
// a min/max pair, so there is no interval to invert — it has its own pair
// of tests further down (negative values, not an inverted interval).
// ===========================================================================

#[test]
#[should_panic(expected = "must not exceed max")]
fn hinge_with_limits_rejects_an_inverted_interval() {
    let _ = HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    )
    .with_limits(q(1, 2), -q(1, 2));
}

#[test]
#[should_panic(expected = "must not exceed max")]
fn slider_with_limits_rejects_an_inverted_interval() {
    let _ = SliderJoint::new(0, 1, Vec3Fix::UNIT_X, Vec3Fix::ZERO, Vec3Fix::ZERO)
        .with_limits(Fix128::ONE, -Fix128::ONE);
}

#[test]
#[should_panic(expected = "must not exceed max")]
fn d6_with_linear_limits_rejects_an_inverted_interval_on_any_axis() {
    // Only the z axis is inverted (x and y are a valid, equal interval);
    // one bad axis among three must still be refused.
    let _ = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_linear_limits(
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, -Fix128::ONE),
    );
}

#[test]
#[should_panic(expected = "must not exceed max")]
fn d6_with_angular_limits_rejects_an_inverted_interval_on_any_axis() {
    let _ = D6Joint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_angular_limits(
        Vec3Fix::new(Fix128::ZERO, q(1, 2), Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, -q(1, 2), Fix128::ZERO),
    );
}

#[test]
#[should_panic(expected = "cone must not be negative")]
fn cone_twist_with_limits_rejects_a_negative_cone() {
    let _ = ConeTwistJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    )
    .with_limits(-q(1, 2), q(1, 4));
}

#[test]
#[should_panic(expected = "twist must not be negative")]
fn cone_twist_with_limits_rejects_a_negative_twist() {
    let _ = ConeTwistJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Z,
        Vec3Fix::UNIT_Z,
    )
    .with_limits(q(1, 4), -q(1, 2));
}

#[test]
fn cone_twist_zero_limits_always_correct_toward_the_axis() {
    // cone_limit = 0: any nonzero cone angle exceeds it and is corrected.
    // twist_limit = 0: likewise for any nonzero twist.
    let joint = [Joint::ConeTwist(
        ConeTwistJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )
        .with_limits(Fix128::ZERO, Fix128::ZERO),
    )];
    let mut b = pair(Fix128::ONE);
    b[1].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_X, q(1, 2));
    solve_joints(&joint, &mut b, DT);
    let cone = cone_angle_from_z(b[1].rotation);
    assert!(
        near(cone, Fix128::ZERO, cordic_tol()),
        "cone_limit=0 must correct all the way back to the axis, got {cone:?}"
    );
}

// ===========================================================================
// C.3 extreme separations: compute_force / solve_joints_breakable wrap, not panic
// ===========================================================================

/// Re-derives, independently of [`alice_physics::math::Fix128::mul`], whether
/// squaring a pure-integer `Fix128` value `2^shift` (fractional part zero)
/// wraps to a value whose `hi` (and hence its square root) is exactly zero.
/// Mirrors the documented 128×128→128 middle-bits formula (see the crate's
/// `Fix128::checked_mul` doc, WM-01 / doctrine B-12) using `i128`/`u128`
/// arithmetic only: `hh = (2^shift)² = 2^(2·shift)`; the `hi` output is the
/// low 64 bits of `hh` (as the formula's `hh as i64`), so the result's
/// integer part is zero exactly when `2^(2·shift) mod 2^64 == 0`, i.e. when
/// `2*shift >= 64`.
fn wraps_to_zero_when_squared(shift: u32) -> bool {
    let hh: u128 = 1u128 << (2 * shift as u128).min(127);
    (hh % (1u128 << 64)) == 0
}

#[test]
fn extreme_separation_squares_wrap_per_the_documented_formula_not_panic() {
    for shift in [32u32, 40, 50, 62] {
        assert!(
            wraps_to_zero_when_squared(shift),
            "independent check: shift={shift} should wrap to zero per 2*shift>=64"
        );
        let far = Fix128::from_raw(1i64 << shift, 0);
        let joint = [Joint::Ball(
            BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::ONE),
        )];
        let mut bodies = pair(Fix128::ONE);
        bodies[1].position = Vec3Fix::new(far, Fix128::ZERO, Fix128::ZERO);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let force = joint[0].compute_force(&bodies);
            let broken = solve_joints_breakable(&joint, &mut bodies, DT);
            (force, broken)
        }));
        let (force, broken) = result.unwrap_or_else(|e| {
            panic!("shift={shift} must not panic (Fix128 wraps, never traps): {e:?}")
        });
        assert_eq!(
            force,
            Fix128::ZERO,
            "shift={shift}: squared-overflow must wrap compute_force to exactly 0 \
             (matches the documented Fix128 Mul wraparound, WM-01/B-12)"
        );
        assert!(
            broken.is_empty(),
            "shift={shift}: a wrapped force of 0 can never exceed break_force=1, \
             so solve_joints_breakable reports the joint as unbroken despite the \
             true separation being 2^{shift} — a real-world caller relying on \
             break_force to cap forces at astronomical separations will not see \
             the joint break"
        );
    }
}

#[test]
fn moderate_separation_below_the_wrap_threshold_computes_the_true_length() {
    // shift=30: 2*shift=60 < 64, no wrap — compute_force must equal 2^30 exactly.
    let shift = 30u32;
    assert!(!wraps_to_zero_when_squared(shift));
    let far = Fix128::from_raw(1i64 << shift, 0);
    let joint = [Joint::Ball(
        BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_break_force(Fix128::ONE),
    )];
    let mut bodies = pair(Fix128::ONE);
    bodies[1].position = Vec3Fix::new(far, Fix128::ZERO, Fix128::ZERO);
    let force = joint[0].compute_force(&bodies);
    assert_eq!(
        force, far,
        "below the wrap threshold, compute_force is exact"
    );
    let broken = solve_joints_breakable(&joint, &mut bodies, DT);
    assert_eq!(
        broken,
        vec![0],
        "a true, un-wrapped separation of 2^30 must break (threshold 1)"
    );
}
