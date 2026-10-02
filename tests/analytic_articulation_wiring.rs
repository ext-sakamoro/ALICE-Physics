//! Oracles for the `articulation` module's API surface that
//! `tests/analytic_multibody_dynamics.rs` does not already cover
//! (`add_link` / `link_count` / `dof_count` / `body_indices` / `joints` /
//! `forward_kinematics` / `apply_motors` / `set_motor` / `build_ragdoll`),
//! plus degenerate input for every item including `solve` and
//! `solve_with_mass_splitting`. Production entry point:
//! `examples/articulated_body_chain.rs`.
//!
//! `FeatherstoneSolver::solve`'s *physics* is pinned exhaustively by
//! `tests/analytic_multibody_dynamics.rs` (14 closed-form oracles); nothing
//! in this file re-derives that. This file's job is the plumbing around it.
//!
//! # Closed forms
//!
//! * **`forward_kinematics`**: `bodies[child].position = parent.position +
//!   parent.rotation.rotate_vec(local_offset)`, read directly from
//!   `ArticulatedBody::fk_recursive`'s own documented formula. The rotation
//!   used here, `QuatFix::new(0, 0, 1, 0)`, is a 180-degree turn about `z`
//!   built from raw components (no `sin_cos` CORDIC call), so the rotation
//!   itself is exact: `q*v*q⁻¹` with `q=(x=0,y=0,z=1,w=0)`,
//!   `v=(0,2,0,0)` gives `q*v=(-2,0,0,0)` and `(q*v)*q⁻¹=(0,-2,0,0)` — i.e.
//!   `(x,y,0)` maps to `(-x,-y,0)`, independently of
//!   [`alice_physics::math::QuatFix::rotate_vec`] (this file performs the
//!   quaternion multiplication itself).
//! * **`apply_motors`' wraparound**: [`alice_physics::math::Fix128::Mul`] is
//!   documented (WM-01 / doctrine B-12, `src/math.rs`) as computing the
//!   128x128->256 product and keeping only the middle 128 bits with plain
//!   (wrapping) arithmetic. For two integer operands `a`, `b` with `a*b` an
//!   exact multiple of `2^64`, the `hh = a_hi*b_hi` cross term is itself a
//!   multiple of `2^64`, so its low 64 bits — which become the wrapped
//!   product's `hi` field — are exactly zero, and the lo/mixed terms are
//!   zero because both operands have `lo == 0`. With `kp = 2^40` and
//!   `error = ±2^40` (so `kp*error = ±2^80`, a multiple of `2^64` since
//!   `80 = 64 + 16`), `PdController::compute`'s `kp*error` therefore wraps
//!   to *exactly* `Fix128::ZERO` before the clamp ever runs, independent of
//!   `max_force`'s sign or magnitude.
//!
//! # Degenerate input
//!
//! * **Zero links**: not constructible through the public API —
//!   [`ArticulatedBody::new`] always seeds exactly one root link, so the
//!   most degenerate chain reachable is a lone root with no joints at all.
//! * **`add_link` with an out-of-range `parent_link`**: unvalidated; it
//!   panics via plain slice indexing (`self.links[parent_link]`) when the
//!   new link is registered as that parent's child. Fail-fast, not a
//!   silent corruption.
//! * **`set_motor` with an out-of-range `link_index`**: explicitly guarded
//!   (`if link_index < self.links.len()`) and silently ignored.
//! * **A joint whose `body_a`/`body_b` do not match the link hierarchy's
//!   own `body_index`/parent `body_index`** (a "disconnected" joint
//!   reference): `add_link` does not cross-check them, and
//!   `subspace_for` indexes `bodies[body_a]` / `bodies[body_b]` directly,
//!   so a joint naming out-of-range bodies panics inside `solve`.
//! * **`forward_kinematics` / `solve` with `ArticulatedBody::root` mutated
//!   out of range** (both `root` and the rest of `ArticulatedBody`'s
//!   fields are `pub`, so this is reachable through the public API without
//!   any unsafe code): `solve` explicitly guards `artic.root >= n` and
//!   returns without touching `bodies` or panicking; `forward_kinematics`
//!   has no such guard and panics via `self.links[link_idx]` indexing.
//!   Same field, two different documented behaviors — this file pins both.
//! * **`solve_with_mass_splitting` with no dynamic body** (`has_dynamic ==
//!   false`, every link's body has `inv_mass <= 0`): the ratio check is
//!   short-circuited by `has_dynamic`, so splitting never activates
//!   regardless of `mass_ratio_split_threshold`, and the call forwards to
//!   a single plain `solve` at the full `dt` — exercising a branch the
//!   `src/articulation.rs` unit tests never reach, since all three of their
//!   scenes have at least one positive-mass link.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::articulation::{ArticulatedBody, FeatherstoneSolver};
use alice_physics::joint::{BallJoint, FixedJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::motor::PdController;
use alice_physics::solver::RigidBody;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn gravity() -> Vec3Fix {
    Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO)
}

// ---------------------------------------------------------------------------
// Zero links is not constructible; the lone root is the degenerate floor.
// ---------------------------------------------------------------------------

#[test]
fn lone_root_is_the_most_degenerate_chain_constructible() {
    let artic = ArticulatedBody::new(5, false);
    assert_eq!(artic.link_count(), 1, "new() always seeds exactly the root");
    assert_eq!(artic.dof_count(), 0, "a lone root has no joint");
    assert_eq!(artic.body_indices(), vec![5]);
    assert!(artic.joints().is_empty());
}

// ---------------------------------------------------------------------------
// add_link
// ---------------------------------------------------------------------------

#[test]
fn add_link_builds_a_three_link_chain_with_correct_bookkeeping() {
    let mut artic = ArticulatedBody::new(10, false);
    let link1 = artic.add_link(
        0,
        11,
        Joint::Ball(BallJoint::new(10, 11, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 1, 0),
    );
    let link2 = artic.add_link(
        link1,
        12,
        Joint::Hinge(HingeJoint::new(
            11,
            12,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_X,
        )),
        Vec3Fix::from_int(0, 1, 0),
    );

    assert_eq!(artic.link_count(), 3);
    assert_eq!(link1, 1, "first add_link returns index 1 (after the root)");
    assert_eq!(link2, 2);
    assert_eq!(artic.body_indices(), vec![10, 11, 12]);
    assert_eq!(artic.links[link1].parent, 0);
    assert_eq!(artic.links[link2].parent, link1);
}

#[test]
fn add_link_panics_on_out_of_range_parent() {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut artic = ArticulatedBody::new(0, false);
        artic.add_link(
            99,
            1,
            Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
            Vec3Fix::ZERO,
        );
    }));
    assert!(
        result.is_err(),
        "add_link does not validate parent_link; an out-of-range parent must panic, not silently corrupt the tree"
    );
}

// ---------------------------------------------------------------------------
// dof_count: counts jointed LINKS, not summed joint DOF.
// ---------------------------------------------------------------------------

#[test]
fn dof_count_counts_jointed_links_not_summed_joint_dof() {
    let mut artic = ArticulatedBody::new(0, false);
    // A weld (Fixed) joint has zero true kinematic freedom...
    let weld = artic.add_link(
        0,
        1,
        Joint::Fixed(FixedJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )),
        Vec3Fix::from_int(0, 1, 0),
    );
    // ...and a ball joint has three.
    artic.add_link(
        weld,
        2,
        Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 1, 0),
    );

    // dof_count's own doc says "approximate: each non-root link's joint" —
    // it is link.joint.is_some().count(), so the weld (0 true DOF) and the
    // ball (3 true DOF) both contribute exactly 1, for a total of 2, not 3.
    assert_eq!(
        artic.dof_count(),
        2,
        "dof_count must count jointed links (2), not the weld's 0 + the ball's 3 = 3 true DOF"
    );
}

// ---------------------------------------------------------------------------
// forward_kinematics
// ---------------------------------------------------------------------------

#[test]
fn forward_kinematics_propagates_parent_rotation_into_child_offset() {
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
        RigidBody::new(Vec3Fix::ZERO, Fix128::ONE),
    ];
    // q = (x=0, y=0, z=1, w=0): a 180-degree rotation about z, built from
    // raw components, so no CORDIC sin_cos is involved anywhere in this
    // scene. See the module doc for the by-hand q*v*q^-1 derivation.
    bodies[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

    let mut artic = ArticulatedBody::new(0, true);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );
    artic.add_link(
        link1,
        2,
        Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );

    artic.forward_kinematics(&mut bodies);

    assert_eq!(
        bodies[1].position,
        Vec3Fix::from_int(0, -2, 0),
        "link 1 = root.position + rotate_180_about_z((0,2,0)) = (0,-2,0)"
    );
    assert_eq!(
        bodies[2].position,
        Vec3Fix::ZERO,
        "link 2's own parent (link 1) kept IDENTITY rotation, so its offset \
         is added unrotated: (0,-2,0) + (0,2,0) = (0,0,0)"
    );
}

#[test]
fn forward_kinematics_panics_when_root_is_mutated_out_of_range() {
    let mut bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::ONE)];
    let mut artic = ArticulatedBody::new(0, false);
    artic.root = 99; // reachable through the public `pub root: usize` field.

    let result = catch_unwind(AssertUnwindSafe(|| {
        artic.forward_kinematics(&mut bodies);
    }));
    assert!(
        result.is_err(),
        "forward_kinematics has no root-range guard (unlike solve), so an \
         out-of-range root must panic via self.links[link_idx] indexing"
    );
}

// ---------------------------------------------------------------------------
// apply_motors / set_motor
// ---------------------------------------------------------------------------

#[test]
fn apply_motors_drives_a_linked_body_toward_its_target() {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(3, 0, 0),
    );
    let mut motor = PdController::new(Fix128::from_int(4), Fix128::ZERO, Fix128::from_int(1000));
    motor.set_position_target(Fix128::from_int(7));
    artic.set_motor(link1, motor);

    // error = 7 - 3 = 4, force = kp*error = 16, dt = 1/2 -> impulse = 8.
    artic.apply_motors(&mut bodies, Fix128::from_ratio(1, 2));

    assert_eq!(
        bodies[0].velocity,
        Vec3Fix::ZERO,
        "static body is never pushed"
    );
    assert_eq!(bodies[1].velocity, Vec3Fix::from_int(8, 0, 0));
}

#[test]
fn set_motor_out_of_range_link_index_is_a_silent_no_op() {
    let mut artic = ArticulatedBody::new(0, false);
    artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::ZERO,
    );
    let before = artic.links.iter().filter(|l| l.motor.is_some()).count();

    artic.set_motor(99, PdController::default());

    assert_eq!(
        artic.link_count(),
        2,
        "out-of-range set_motor must not resize the chain"
    );
    let after = artic.links.iter().filter(|l| l.motor.is_some()).count();
    assert_eq!(
        before, after,
        "out-of-range set_motor must not attach a motor anywhere"
    );
}

#[test]
fn apply_motors_with_overflowing_gain_and_target_wraps_force_to_exactly_zero() {
    // kp = 2^40, target = 0, current separation = 2^40 (bit-exact: isqrt of
    // a perfect square along a single axis), kd = 0 so there is no
    // velocity term. error = 0 - 2^40 = -2^40, so kp*error = -2^80, which
    // wraps to exactly Fix128::ZERO per this file's module doc.
    let two_pow_40 = 1i64 << 40;
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_dynamic(Vec3Fix::from_int(two_pow_40, 0, 0), Fix128::ONE),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(two_pow_40, 0, 0),
    );
    let mut motor = PdController::new(
        Fix128::from_int(two_pow_40),
        Fix128::ZERO,
        Fix128::from_int(two_pow_40), // max_force, irrelevant: the product
                                      // wraps to 0 before the clamp ever sees it.
    );
    motor.set_position_target(Fix128::ZERO);
    artic.set_motor(link1, motor);

    let result = catch_unwind(AssertUnwindSafe(|| {
        artic.apply_motors(&mut bodies, Fix128::ONE);
    }));
    assert!(
        result.is_ok(),
        "Fix128 multiplication is wrapping, never panics"
    );
    assert_eq!(
        bodies[1].velocity,
        Vec3Fix::ZERO,
        "kp*error wraps to exactly 0 (not merely 'doesn't panic' — this is \
         the Fix128::ZERO value the wraparound actually produces), so \
         apply_motors' force.is_zero() early-continue guard fires and \
         no impulse is ever applied, despite an enormous commanded error"
    );
}

// ---------------------------------------------------------------------------
// FeatherstoneSolver::solve / solve_with_mass_splitting — degenerate input
// (the physics itself is covered by tests/analytic_multibody_dynamics.rs)
// ---------------------------------------------------------------------------

#[test]
fn solve_treats_an_out_of_range_root_as_a_documented_no_op() {
    let mut bodies = vec![RigidBody::new(Vec3Fix::from_int(0, 5, 0), Fix128::ONE)];
    let before = bodies[0].position;
    let mut artic = ArticulatedBody::new(0, false);
    artic.root = 99; // reachable through the public field, same as above.

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut solver = FeatherstoneSolver::new();
        solver.solve(&artic, &mut bodies, gravity(), Fix128::from_ratio(1, 60));
    }));
    assert!(
        result.is_ok(),
        "solve explicitly guards `artic.root >= n` and returns early"
    );
    assert_eq!(
        bodies[0].position, before,
        "the early return must leave bodies untouched, not merely avoid panicking"
    );
}

#[test]
fn solve_panics_when_a_joint_names_bodies_outside_the_chain() {
    // The joint's own body_a/body_b (999, 1000) do not match either this
    // link's body_index (1) or its parent's (0) — a "disconnected" joint
    // reference that add_link never validates.
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(999, 1000, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(1, 0, 0),
    );

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut solver = FeatherstoneSolver::new();
        solver.solve(&artic, &mut bodies, gravity(), Fix128::from_ratio(1, 60));
    }));
    assert!(
        result.is_err(),
        "subspace_for indexes bodies[body_a]/bodies[body_b] directly from \
         the joint, so out-of-range or mismatched body indices must panic"
    );
}

#[test]
fn solve_with_mass_splitting_is_inert_when_no_link_is_dynamic() {
    // Every link's body has inv_mass == 0 (both static), so has_dynamic is
    // false and the ratio check never runs, regardless of how low the
    // threshold is set. None of the three src unit tests reach this branch
    // (all of them give every link a positive mass).
    let mut bodies_split = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_static(Vec3Fix::from_int(0, 2, 0)),
    ];
    let mut bodies_plain = bodies_split.clone();
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );

    let dt = Fix128::from_ratio(1, 60);
    FeatherstoneSolver::new().solve_with_mass_splitting(
        &artic,
        &mut bodies_split,
        gravity(),
        dt,
        Fix128::from_ratio(1, 1_000_000), // threshold near 0: would trigger
                                          // splitting for any real ratio.
    );
    FeatherstoneSolver::new().solve(&artic, &mut bodies_plain, gravity(), dt);

    assert_eq!(
        bodies_split[0].position, bodies_plain[0].position,
        "has_dynamic == false must forward to a single plain solve at full dt, bit-exact"
    );
    assert_eq!(bodies_split[1].position, bodies_plain[1].position);
    assert_eq!(
        bodies_split[1].position,
        Vec3Fix::from_int(0, 2, 0),
        "both links are static, so nothing should move at all"
    );
}

/// A single free (unjointed) root body under gravity never acquires
/// rotation (see the derivation below), so its acceleration is exactly
/// `gravity` at every re-solve, which makes the two-half-step vs
/// one-full-step position difference an exact closed form rather than an
/// approximation.
///
/// # Why bias is exactly zero for a non-rotating free body
///
/// `forward_velocity_pass` sets `vel = {ang: angular_velocity, lin:
/// velocity - angular_velocity x position}`; with `angular_velocity == 0`
/// this is `{ang: 0, lin: velocity}`. The backward pass's bias is
/// `vel.cross_force(inertia.mul_vec(vel))`. Spelling out
/// `SpatialMat::mul_vec` and `SpatialVec::cross_force` with `vel.ang == 0`:
/// `inertia.mul_vec(vel) = {ang: mass*(position x velocity), lin:
/// mass*velocity}`, and `cross_force` of that against `vel` collapses to
/// `{ang: velocity x (mass*velocity), lin: 0} = {ang: 0, lin: 0}` because
/// `velocity x velocity` is identically zero — independent of position, so
/// the world-origin moment arm never enters. Since `alpha` (the angular part
/// of `ã`) is therefore always exactly zero, `angular_velocity` never
/// leaves zero across the whole run, which is what keeps the bias zero at
/// every subsequent re-solve too, not just the first.
///
/// # Closed form
///
/// With `bias == 0` always, `ã = -(I^A)^-1 * 0 = 0`, so `a_cm = gravity`
/// exactly, every step. Semi-implicit Euler from rest:
/// - one full step of `dt`:  `v1 = g*dt`,  `x1 = v1*dt = g*dt^2`
/// - two half steps of `dt/2`: `v1' = g*dt/2`, `x1' = x0 + v1'*dt/2 =
///   g*dt^2/4`; `v2' = v1' + g*dt/2 = g*dt` (same final velocity);
///   `x2' = x1' + v2'*dt/2 = g*dt^2/4 + g*dt^2/2 = (3/4)*g*dt^2`.
///
/// So `x2' - x1 = -(1/4)*g*dt^2` exactly, while `v2' == v1` exactly — a
/// mutation that deletes the actual half-step recursion (always forwarding
/// to a single `solve(dt)`) is caught by the position difference, and one
/// that always splits regardless of the ratio check is caught by the
/// below-threshold case matching plain `solve` bit-exact.
#[test]
fn solve_with_mass_splitting_two_half_steps_differ_from_one_full_step_by_exactly_one_quarter_g_dt2()
{
    let dt = Fix128::from_ratio(1, 4);
    let scene = || {
        (
            ArticulatedBody::new(0, false),
            vec![RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2))],
        )
    };

    let (artic_full, mut bodies_full) = scene();
    FeatherstoneSolver::new().solve(&artic_full, &mut bodies_full, gravity(), dt);

    let (artic_split, mut bodies_split) = scene();
    // A single body always has min_inv_mass == max_inv_mass, so the ratio
    // is exactly 1: threshold <= 1 activates splitting, threshold > 1 does
    // not. This is the simplest constructible non-default (threshold != 0)
    // scene that exercises both branches with full control.
    FeatherstoneSolver::new().solve_with_mass_splitting(
        &artic_split,
        &mut bodies_split,
        gravity(),
        dt,
        Fix128::ONE,
    );

    let (artic_inert, mut bodies_inert) = scene();
    FeatherstoneSolver::new().solve_with_mass_splitting(
        &artic_inert,
        &mut bodies_inert,
        gravity(),
        dt,
        Fix128::from_int(2),
    );

    let expected_diff = Fix128::from_ratio(-1, 4) * gravity().y * dt * dt;
    let actual_diff = bodies_split[0].position.y - bodies_full[0].position.y;
    assert_eq!(
        actual_diff,
        expected_diff,
        "two half-steps must land exactly (1/4)*g*dt^2 away from one full \
         step; got diff {} expected {}",
        actual_diff.to_f64(),
        expected_diff.to_f64(),
    );
    assert_eq!(
        bodies_split[0].velocity, bodies_full[0].velocity,
        "constant acceleration means both schemes reach the same final velocity"
    );
    assert_eq!(
        bodies_inert[0].position, bodies_full[0].position,
        "threshold 2 > ratio 1 must forward to a single plain solve, bit-exact"
    );
}

// ---------------------------------------------------------------------------
// build_ragdoll: pin the hierarchy against its own documented contract
// ("Pelvis -> Spine -> Chest -> Head / L/R Upper Arm -> Lower Arm / L/R
// Upper Leg -> Lower Leg"), not against the implementation's internals.
// ---------------------------------------------------------------------------

#[test]
fn build_ragdoll_matches_its_documented_hierarchy() {
    let (artic, bodies) = alice_physics::articulation::build_ragdoll(Vec3Fix::ZERO, 0);

    assert_eq!(bodies.len(), 12);
    assert_eq!(artic.link_count(), 12);
    assert_eq!(artic.dof_count(), 11, "every non-root link carries a joint");

    // Link indices, in creation order: 0 pelvis(root), 1 spine, 2 chest,
    // 3 head, 4 l_upper_arm, 5 l_lower_arm, 6 r_upper_arm, 7 r_lower_arm,
    // 8 l_upper_leg, 9 l_lower_leg, 10 r_upper_leg, 11 r_lower_leg.
    assert_eq!(artic.links[1].parent, 0, "spine attaches to the pelvis");
    assert_eq!(artic.links[2].parent, 1, "chest attaches to the spine");
    assert_eq!(artic.links[3].parent, 2, "head attaches to the chest");
    assert_eq!(
        artic.links[4].parent, 2,
        "left upper arm attaches to the chest"
    );
    assert_eq!(
        artic.links[5].parent, 4,
        "left lower arm attaches to the left upper arm"
    );
    assert_eq!(
        artic.links[6].parent, 2,
        "right upper arm attaches to the chest"
    );
    assert_eq!(
        artic.links[7].parent, 6,
        "right lower arm attaches to the right upper arm"
    );
    assert_eq!(
        artic.links[8].parent, 0,
        "left upper leg attaches to the pelvis"
    );
    assert_eq!(
        artic.links[9].parent, 8,
        "left lower leg attaches to the left upper leg"
    );
    assert_eq!(
        artic.links[10].parent, 0,
        "right upper leg attaches to the pelvis"
    );
    assert_eq!(
        artic.links[11].parent, 10,
        "right lower leg attaches to the right upper leg"
    );
}

#[test]
fn build_ragdoll_body_start_index_offsets_every_body() {
    let start = 1000;
    let (artic, bodies) = alice_physics::articulation::build_ragdoll(Vec3Fix::ZERO, start);

    assert_eq!(bodies.len(), 12);
    let indices = artic.body_indices();
    assert_eq!(
        indices,
        (start..start + 12).collect::<Vec<_>>(),
        "a mutation that ignores body_start_index would pass with start = 0; \
         non-default start pins that every body index is actually offset"
    );
}
