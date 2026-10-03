//! Oracles for `ik_physics_bridge`: one stabilisation pass moves a body by
//! `residual * clamp(weight, 0, 1) / (1 + max(compliance, 0))`, never moves an
//! immovable body, and (driven through `RagdollBuilder`) steers a bone to a target.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::ik_physics_bridge::{IkTarget, IkTargetSet};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::ragdoll::{Bone, RagdollBuilder, RagdollProportions};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn world_with(x: i64) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let b = w.add_body(RigidBody::new(Vec3Fix::from_int(x, 0, 0), Fix128::ONE));
    (w, b)
}

fn pos_x(w: &PhysicsWorld, b: usize) -> f64 {
    w.bodies[b].position.x.to_f64()
}

fn target(b: usize, x: i64, weight: Fix128) -> IkTarget {
    IkTarget {
        body: b,
        position: Vec3Fix::from_int(x, 0, 0),
        weight,
    }
}

#[test]
fn constructors_set_the_documented_weights() {
    let p = Vec3Fix::from_int(1, 2, 3);
    let s = IkTarget::snap(4, p);
    assert_eq!((s.body, s.position, s.weight), (4, p, Fix128::ONE));
    let b = IkTarget::blended(5, p);
    assert_eq!((b.body, b.position), (5, p));
    assert_eq!(b.weight, Fix128::from_ratio(1, 2));
}

#[test]
fn one_pass_moves_by_weight_times_residual() {
    // body at x = 2, target x = 10: residual 8.
    for (num, den, want) in [
        (0, 1, 2.0),
        (1, 4, 4.0),
        (1, 2, 6.0),
        (3, 4, 8.0),
        (1, 1, 10.0),
    ] {
        let (mut w, b) = world_with(2);
        let mut s = IkTargetSet::new();
        s.push(target(b, 10, Fix128::from_ratio(num, den)));
        s.apply(&mut w);
        assert!(
            (pos_x(&w, b) - want).abs() < 1e-9,
            "weight {num}/{den}: {}",
            pos_x(&w, b)
        );
    }
}

#[test]
fn compliance_scales_the_correction_by_one_over_one_plus_c() {
    for (c, want) in [(0, 10.0), (1, 6.0), (3, 4.0), (7, 3.0)] {
        // residual 8 * 1/(1+c) added to x = 2
        let (mut w, b) = world_with(2);
        let mut s = IkTargetSet::new();
        s.compliance = Fix128::from_int(c);
        s.push(IkTarget::snap(b, Vec3Fix::from_int(10, 0, 0)));
        s.apply(&mut w);
        assert!(
            (pos_x(&w, b) - want).abs() < 1e-9,
            "compliance {c}: {}",
            pos_x(&w, b)
        );
    }
}

#[test]
fn out_of_range_weight_and_negative_compliance_are_clamped() {
    // weight 2 must not overshoot, weight -1 must not move away.
    for (wt, want) in [(2, 10.0), (-1, 2.0)] {
        let (mut w, b) = world_with(2);
        let mut s = IkTargetSet::new();
        s.push(target(b, 10, Fix128::from_int(wt)));
        s.apply(&mut w);
        assert!(
            (pos_x(&w, b) - want).abs() < 1e-9,
            "weight {wt}: {}",
            pos_x(&w, b)
        );
    }
    // compliance -1 would divide by zero, -2 would reverse the correction.
    for c in [-1, -2, -100] {
        let (mut w, b) = world_with(2);
        let mut s = IkTargetSet::new();
        s.compliance = Fix128::from_int(c);
        s.push(IkTarget::snap(b, Vec3Fix::from_int(10, 0, 0)));
        s.apply(&mut w);
        assert!(
            (pos_x(&w, b) - 10.0).abs() < 1e-9,
            "compliance {c}: {}",
            pos_x(&w, b)
        );
    }
}

#[test]
fn immovable_missing_and_empty_are_untouched() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let fixed = w.add_body(RigidBody::new_static(Vec3Fix::from_int(1, 0, 0)));
    let free = w.add_body(RigidBody::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE));
    let mut s = IkTargetSet::new();
    s.push(IkTarget::snap(fixed, Vec3Fix::from_int(9, 0, 0)))
        .push(IkTarget::snap(99, Vec3Fix::from_int(9, 0, 0)))
        .push(IkTarget::snap(free, Vec3Fix::from_int(9, 0, 0)));
    assert_eq!(s.targets.len(), 3);
    s.apply(&mut w);
    assert_eq!(pos_x(&w, fixed), 1.0);
    assert_eq!(pos_x(&w, free), 9.0);
    // Empty set: nothing moves.
    IkTargetSet::new().apply(&mut w);
    assert_eq!(pos_x(&w, free), 9.0);
    // Heavy and light bodies move identically (no mass weighting, as documented).
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let light = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
    let heavy = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(1000)));
    let mut s = IkTargetSet::new();
    s.push(IkTarget::blended(light, Vec3Fix::from_int(4, 0, 0)))
        .push(IkTarget::blended(heavy, Vec3Fix::from_int(4, 0, 0)));
    s.apply(&mut w);
    assert_eq!(pos_x(&w, light), pos_x(&w, heavy));
    assert!((pos_x(&w, light) - 2.0).abs() < 1e-9);
}

#[test]
fn repeated_passes_converge_geometrically() {
    // weight 1/2 per pass: the residual halves each time -> after n passes 8 * 2^-n remains.
    let (mut w, b) = world_with(2);
    let mut s = IkTargetSet::new();
    s.push(IkTarget::blended(b, Vec3Fix::from_int(10, 0, 0)));
    for n in 1..=6 {
        s.apply(&mut w);
        let want = 10.0 - 8.0 * 0.5_f64.powi(n);
        assert!(
            (pos_x(&w, b) - want).abs() < 1e-9,
            "pass {n}: {}",
            pos_x(&w, b)
        );
    }
}

#[test]
fn a_ragdoll_hand_is_steered_to_its_target() {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let rag = RagdollBuilder::build(
        &mut w,
        RagdollProportions::human_male(),
        Vec3Fix::from_int(0, 2, 0),
    );
    let hand = rag.body(Bone::RightHand);
    let goal = Vec3Fix::new(
        Fix128::from_int(1),
        Fix128::from_int(2),
        Fix128::from_int(1),
    );
    let mut s = IkTargetSet::new();
    s.push(IkTarget::snap(hand, goal));
    s.apply(&mut w);
    assert_eq!(w.bodies[hand].position, goal);
    // Only the targeted bone moved.
    let pelvis = w.bodies[rag.body(Bone::Pelvis)].position;
    assert_eq!(pelvis, Vec3Fix::from_int(0, 2, 0));
}

#[test]
fn targets_apply_in_insertion_order_and_the_first_out_of_range_index_is_ignored() {
    // Snap to x = 10, then half-way back to x = 0: 10 -> 5. Reversed order would end at 10.
    let (mut w, b) = world_with(2);
    let mut s = IkTargetSet::new();
    s.push(IkTarget::snap(b, Vec3Fix::from_int(10, 0, 0)))
        .push(IkTarget::blended(b, Vec3Fix::ZERO));
    s.apply(&mut w);
    assert!((pos_x(&w, b) - 5.0).abs() < 1e-9, "{}", pos_x(&w, b));
    // body index == bodies.len() is the first invalid one: ignored, not an out-of-bounds panic.
    let (mut w, b) = world_with(2);
    let mut s = IkTargetSet::new();
    s.push(IkTarget::snap(w.bodies.len(), Vec3Fix::from_int(10, 0, 0)));
    s.apply(&mut w);
    assert_eq!(pos_x(&w, b), 2.0);
}
