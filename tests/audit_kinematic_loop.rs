//! Audit S3W3 oracles for `src/kinematic_loop.rs`.
//!
//! Closed forms: for anchors `a = p_a + R_a * l_a`, `b = p_b + R_b * l_b` the
//! residual is `a - b`; one Baumgarte pass with compliance `c` moves the
//! bodies by `delta = residual / (1 + c)` split by normalised inverse mass
//! (`w_a = inv_a / (inv_a + inv_b)`), so the residual after one pass is
//! `residual * (1 - 1/(1+c))` exactly. Expected values are derived by hand
//! below, never by calling `residual` / `apply`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::kinematic_loop::{four_bar_linkage, LoopClosureConstraint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

const TOL: f64 = 1e-9;

fn close(label: &str, got: Fix128, want: f64) {
    let g = got.to_f64();
    assert!((g - want).abs() < TOL, "{label}: got {g} want {want}");
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig::default())
}

/// a at (0,0,0) mass 1, anchor (1,2,0); b at (5,0,1) mass 3, anchor (0,-1,0).
fn two_bodies() -> (PhysicsWorld, LoopClosureConstraint) {
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new(
        Vec3Fix::from_int(5, 0, 1),
        Fix128::from_int(3),
    ));
    let c = LoopClosureConstraint {
        body_a: a,
        body_b: b,
        local_anchor_a: Vec3Fix::from_int(1, 2, 0),
        local_anchor_b: Vec3Fix::from_int(0, -1, 0),
        compliance: Fix128::ZERO,
    };
    (w, c)
}

#[test]
fn residual_with_anchors_is_anchor_a_minus_anchor_b() {
    // (0,0,0)+(1,2,0) - ((5,0,1)+(0,-1,0)) = (-4, 3, -1)
    let (w, c) = two_bodies();
    let res = c.residual(&w);
    close("x", res.x, -4.0);
    close("y", res.y, 3.0);
    close("z", res.z, -1.0);
}

#[test]
fn apply_splits_correction_by_normalised_inverse_mass() {
    // inv_a = 1, inv_b = 1/3 -> w_a = 3/4, w_b = 1/4. delta = residual.
    // a += (4,-3,1)*3/4 ; b += (-4,3,-1)/4
    let (mut w, c) = two_bodies();
    c.apply(&mut w);
    let pa = w.bodies[c.body_a].position;
    let pb = w.bodies[c.body_b].position;
    close("a.x", pa.x, 3.0);
    close("a.y", pa.y, -2.25);
    close("a.z", pa.z, 0.75);
    close("b.x", pb.x, 4.0);
    close("b.y", pb.y, 0.75);
    close("b.z", pb.z, 0.75);
    let res = c.residual(&w);
    close("res.x", res.x, 0.0);
    close("res.y", res.y, 0.0);
    close("res.z", res.z, 0.0);
}

#[test]
fn compliance_leaves_exactly_c_over_one_plus_c_of_the_residual() {
    // c = 3: scale = 1/4; residual after one pass = 3/4 of (-4,3,-1)
    let (mut w, mut c) = two_bodies();
    c.compliance = Fix128::from_int(3);
    c.apply(&mut w);
    let res = c.residual(&w);
    close("x", res.x, -3.0);
    close("y", res.y, 2.25);
    close("z", res.z, -0.75);
    // c = 1/2: leaves (1/2)/(3/2) = 1/3
    let (mut w, mut c) = two_bodies();
    c.compliance = r(1, 2);
    c.apply(&mut w);
    let res = c.residual(&w);
    close("x", res.x, -4.0 / 3.0);
}

#[test]
fn static_body_is_held_and_dynamic_body_takes_the_whole_correction() {
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO));
    let b = w.add_body(RigidBody::new(
        Vec3Fix::from_int(4, 0, 0),
        Fix128::from_int(2),
    ));
    let c = LoopClosureConstraint::centre_to_centre(a, b);
    c.apply(&mut w);
    assert_eq!(w.bodies[a].position, Vec3Fix::ZERO);
    close("b.x", w.bodies[b].position.x, 0.0);
    // and mirrored: static b, dynamic a
    let mut w = world();
    let a = w.add_body(RigidBody::new(
        Vec3Fix::from_int(4, 0, 0),
        Fix128::from_int(2),
    ));
    let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO));
    let c = LoopClosureConstraint::centre_to_centre(a, b);
    c.apply(&mut w);
    assert_eq!(w.bodies[b].position, Vec3Fix::ZERO);
    close("a.x", w.bodies[a].position.x, 0.0);
}

#[test]
fn apply_changes_only_positions_of_the_two_bodies() {
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new(Vec3Fix::from_int(4, 0, 0), Fix128::ONE));
    let other = w.add_body(RigidBody::new(Vec3Fix::from_int(9, 9, 9), Fix128::ONE));
    w.bodies[a].velocity = Vec3Fix::from_int(1, 2, 3);
    w.bodies[b].angular_velocity = Vec3Fix::from_int(3, 2, 1);
    let (va, wb, po) = (
        w.bodies[a].velocity,
        w.bodies[b].angular_velocity,
        w.bodies[other],
    );
    LoopClosureConstraint::centre_to_centre(a, b).apply(&mut w);
    assert_eq!(w.bodies[a].velocity, va);
    assert_eq!(w.bodies[b].angular_velocity, wb);
    assert_eq!(w.bodies[other].position, po.position);
    assert_eq!(w.bodies[other].velocity, po.velocity);
}

#[test]
// AUD-A-S3W3-002
fn residual_rotates_the_local_anchor_into_world_space() {
    // `local_anchor_a` is documented as "Anchor point in body A's local
    // frame". Body A rotated +90 deg about z: local (1,0,0) is world (0,1,0).
    // Body B at (0,1,0) with identity rotation and zero anchor -> residual 0.
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new(Vec3Fix::from_int(0, 1, 0), Fix128::ONE));
    let half_pi = Fix128::PI / Fix128::from_int(2);
    w.bodies[a].rotation = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), half_pi);
    let c = LoopClosureConstraint {
        body_a: a,
        body_b: b,
        local_anchor_a: Vec3Fix::from_int(1, 0, 0),
        local_anchor_b: Vec3Fix::ZERO,
        compliance: Fix128::ZERO,
    };
    let res = c.residual(&w);
    assert!(
        res.length().to_f64() < 1e-9,
        "anchor not rotated into world frame: residual = ({}, {}, {})",
        res.x.to_f64(),
        res.y.to_f64(),
        res.z.to_f64()
    );
}

#[test]
fn four_bar_translates_with_ground_position_and_masses_follow_arguments() {
    let g = Vec3Fix::from_int(3, -2, 5);
    let mut w = world();
    let l = four_bar_linkage(
        &mut w,
        g,
        Fix128::ONE,
        Fix128::from_int(2),
        r(3, 2),
        r(5, 2),
        Fix128::from_int(4),
    );
    assert_eq!(w.bodies.len(), 4);
    // ground and rocker are static, crank and coupler carry body_mass
    assert!(w.bodies[l.ground].inv_mass.is_zero());
    assert!(w.bodies[l.rocker].inv_mass.is_zero());
    close("crank inv_mass", w.bodies[l.crank].inv_mass, 0.25);
    close("coupler inv_mass", w.bodies[l.coupler].inv_mass, 0.25);
    // O2 = g, A = g + (r2,0,0), O4 = g + (r1,0,0)
    assert_eq!(w.bodies[l.ground].position, g);
    close("A.x", w.bodies[l.crank].position.x, 4.0);
    close("A.y", w.bodies[l.crank].position.y, -2.0);
    close("A.z", w.bodies[l.crank].position.z, 5.0);
    close("O4.x", w.bodies[l.rocker].position.x, 5.5);
    close("O4.y", w.bodies[l.rocker].position.y, -2.0);
    // planar: every pin keeps the ground z
    close("B.z", w.bodies[l.coupler].position.z, 5.0);
}

#[test]
fn four_bar_joints_index_the_three_link_constraints_with_their_lengths() {
    let mut w = world();
    // pre-existing constraint so that `joints` is not trivially 0,1,2
    let a0 = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b0 = w.add_body(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE));
    w.add_distance_constraint(alice_physics::solver::DistanceConstraint::new(
        a0,
        b0,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::ONE,
    ));
    let l = four_bar_linkage(
        &mut w,
        Vec3Fix::ZERO,
        Fix128::ONE,
        Fix128::from_int(2),
        r(3, 2),
        r(5, 2),
        Fix128::ONE,
    );
    assert_eq!(l.joints, [1, 2, 3]);
    let dc = &w.distance_constraints;
    assert_eq!(
        (dc[l.joints[0]].body_a, dc[l.joints[0]].body_b),
        (l.ground, l.crank)
    );
    assert_eq!(
        (dc[l.joints[1]].body_a, dc[l.joints[1]].body_b),
        (l.crank, l.coupler)
    );
    assert_eq!(
        (dc[l.joints[2]].body_a, dc[l.joints[2]].body_b),
        (l.coupler, l.rocker)
    );
    close("crank len", dc[l.joints[0]].target_distance, 1.0);
    close("coupler len", dc[l.joints[1]].target_distance, 2.0);
    close("rocker len", dc[l.joints[2]].target_distance, 1.5);
    for j in l.joints {
        assert!(dc[j].compliance.is_zero(), "links are rigid");
    }
}

#[test]
fn four_bar_closure_recloses_a_freed_rocker_pin() {
    // Doc: "`closure` re-closes it if the caller frees the body".
    let g = Vec3Fix::from_int(1, 1, 0);
    let mut w = world();
    let l = four_bar_linkage(
        &mut w,
        g,
        Fix128::ONE,
        Fix128::from_int(2),
        r(3, 2),
        r(5, 2),
        Fix128::ONE,
    );
    close("closed at build", l.closure.residual(&w).x, 0.0);
    // free the rocker and knock it off the ground pin
    w.bodies[l.rocker].inv_mass = Fix128::ONE;
    w.bodies[l.rocker].position = w.bodies[l.rocker].position + Vec3Fix::from_int(0, 1, 2);
    let before = l.closure.residual(&w);
    close("knocked y", before.y, 1.0);
    close("knocked z", before.z, 2.0);
    l.closure.apply(&mut w);
    let after = l.closure.residual(&w);
    close("x", after.x, 0.0);
    close("y", after.y, 0.0);
    close("z", after.z, 0.0);
    // ground body must not have moved
    assert_eq!(w.bodies[l.ground].position, g);
}

/// AUD-A-S3W3-002 (apply): A rotated +90 deg about z with local anchor (1,0,0)
/// has its anchor at world (0,1,0); B at (0,3,0) with no anchor; equal masses,
/// rigid: one apply moves each by half the gap (A to (0,1,0), B to (0,2,0)) and
/// the anchors coincide.
#[test]
fn apply_closes_the_gap_between_rotated_anchors() {
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let b = w.add_body(RigidBody::new(Vec3Fix::from_int(0, 3, 0), Fix128::ONE));
    let half_pi = Fix128::PI / Fix128::from_int(2);
    w.bodies[a].rotation = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), half_pi);
    let c = LoopClosureConstraint {
        body_a: a,
        body_b: b,
        local_anchor_a: Vec3Fix::from_int(1, 0, 0),
        local_anchor_b: Vec3Fix::ZERO,
        compliance: Fix128::ZERO,
    };
    c.apply(&mut w);
    let pa = w.bodies[a].position;
    let pb = w.bodies[b].position;
    assert!(
        (pa.y.to_f64() - 1.0).abs() < 1e-9 && pa.x.to_f64().abs() < 1e-9,
        "{pa:?}"
    );
    assert!(
        (pb.y.to_f64() - 2.0).abs() < 1e-9 && pb.x.to_f64().abs() < 1e-9,
        "{pb:?}"
    );
    assert!(c.residual(&w).length().to_f64() < 1e-9);
}

/// Both anchors rotate: B turned -90 deg about z with local anchor (0,1,0) has
/// its anchor at world (1,0,0) from its position; A at (1,0,0) with no anchor
/// coincides with it when B sits at the origin.
#[test]
fn residual_rotates_anchor_b_too() {
    let mut w = world();
    let a = w.add_body(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE));
    let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let half_pi = Fix128::PI / Fix128::from_int(2);
    w.bodies[b].rotation = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), -half_pi);
    let c = LoopClosureConstraint {
        body_a: a,
        body_b: b,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::from_int(0, 1, 0),
        compliance: Fix128::ZERO,
    };
    assert!(
        c.residual(&w).length().to_f64() < 1e-9,
        "{:?}",
        c.residual(&w)
    );
}
