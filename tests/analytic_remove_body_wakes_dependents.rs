//! `PhysicsWorld::remove_body` must wake the sleeping bodies that may lose
//! their support or their joint partner, and only those.
//!
//! Oracles (external to the solver):
//! - a body resting on the removed one is no longer supported, so it must
//!   fall under gravity and come to rest on what is below: its height
//!   decreases step by step (never rising by more than the contact
//!   solver's correction) until it lands at the floor height given by the
//!   geometry (floor top + its own half height);
//! - a body joint-connected to the removed one belongs to its island and
//!   must be woken with it;
//! - a sleeping body touching neither the removed body nor its island
//!   stays asleep (see also `tests/analytic_remove_body_keeps_sleep.rs`).
//!
//! The contact partners of a sleeping body are not held by any contact
//! list at removal time (sleeping pairs are not re-detected and the contact
//! cache is not kept for them), which is why these scenes remove a support
//! only after everything has fallen asleep.

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::{SleepConfig, SleepState};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn sleepy_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 10),
        angular_threshold: Fix128::from_ratio(1, 10),
        frames_to_sleep: 3,
    });
    // static sphere (r = 10) whose top is at y = 0
    w.add_body_with_radius(
        RigidBody::new_static(v3(0.0, -10.0, 0.0)),
        Fix128::from_f64(10.0),
    );
    w
}

fn asleep(w: &PhysicsWorld, i: usize) -> bool {
    w.islands.sleep_data[i].state == SleepState::Sleeping
}

fn step_until_asleep(w: &mut PhysicsWorld, bodies: &[usize]) {
    for _ in 0..600 {
        if bodies.iter().all(|&i| asleep(w, i)) {
            break;
        }
        w.step(dt());
    }
    assert!(
        bodies.iter().all(|&i| asleep(w, i)),
        "premise: the stack must fall asleep; sleep_data = {:?}",
        w.islands.sleep_data
    );
    // stay asleep a while so the scene is a settled rest state
    for _ in 0..30 {
        w.step(dt());
    }
    assert!(bodies.iter().all(|&i| asleep(w, i)));
}

/// After the support is removed, `top` (now at index `top_after`) falls and
/// lands: until it first reaches the floor its height never rises (beyond
/// `slack`, the contact correction), after that any rebound stays below
/// `rebound` (well under the drop height), and it ends within `tol` of
/// `rest_y`, asleep again, having dropped by about the support's height.
fn assert_falls_and_lands(w: &mut PhysicsWorld, top_after: usize, rest_y: f64, drop: f64) {
    let slack = 1e-3;
    let tol = 1e-2;
    let rebound = 0.1;
    let y0 = w.bodies[top_after].position.y.to_f64();
    assert!(
        !asleep(w, top_after),
        "the body resting on the removed support must be woken by remove_body"
    );
    let mut prev = y0;
    let mut landed = false;
    for k in 0..240 {
        w.step(dt());
        let y = w.bodies[top_after].position.y.to_f64();
        if landed {
            assert!(
                y < rest_y + rebound,
                "step {k}: rebound to {y} above {rest_y}"
            );
        } else {
            assert!(
                y <= prev + slack,
                "step {k}: y rose from {prev} to {y} before landing"
            );
            landed = y < rest_y + tol;
        }
        prev = y;
    }
    assert!(landed, "never reached the floor: y = {prev}");
    assert!(
        (y0 - prev - drop).abs() < 2.0 * tol,
        "drops by the support's height {drop}: from {y0} to {prev}"
    );
    assert!(
        (prev - rest_y).abs() < tol,
        "lands on the floor at y = {rest_y}, got {prev}"
    );
    assert!(asleep(w, top_after), "comes to rest and falls asleep again");
}

/// floor (0), sup (1), top (2), all asleep; remove sup ⇒ top moves to 1.
#[test]
fn sphere_on_removed_sphere_falls_to_the_floor() {
    let mut w = sleepy_world();
    let r = Fix128::from_ratio(1, 2);
    let sup = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE), r);
    let top = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 1.5, 0.0), Fix128::ONE), r);
    step_until_asleep(&mut w, &[sup, top]);
    w.remove_body(sup).expect("in range");
    assert_falls_and_lands(&mut w, sup, 0.5, 1.0);
}

/// Same with a resting body that is not the last one: removing the support
/// at index 1 moves the last body (`top`) into slot 1.
#[test]
fn sphere_on_removed_sphere_falls_when_not_last() {
    let mut w = sleepy_world();
    let r = Fix128::from_ratio(1, 2);
    let sup = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE), r);
    let other = w.add_body_with_radius(RigidBody::new_dynamic(v3(3.0, 0.5, 0.0), Fix128::ONE), r);
    let top = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 1.5, 0.0), Fix128::ONE), r);
    step_until_asleep(&mut w, &[sup, other, top]);
    w.remove_body(sup).expect("in range");
    // `top` moved into the removed slot; `other` touches neither and sleeps on
    assert!(asleep(&w, other), "an unrelated resting body stays asleep");
    assert_falls_and_lands(&mut w, sup, 0.5, 1.0);
    assert!(asleep(&w, other));
}

/// Box (half extent 0.5) resting on a box: removing the lower one drops the
/// upper one to the floor.
#[test]
fn box_on_removed_box_falls_to_the_floor() {
    let mut w = sleepy_world();
    let cube = Shape::Box {
        half_extents: v3(0.5, 0.5, 0.5),
    };
    let sup = w
        .add_shaped_body(&cube, Fix128::ONE, v3(0.0, 0.5, 0.0))
        .expect("valid box");
    let top = w
        .add_shaped_body(&cube, Fix128::ONE, v3(0.0, 1.5, 0.0))
        .expect("valid box");
    step_until_asleep(&mut w, &[sup, top]);
    let sup_y = w.bodies[sup].position.y.to_f64();
    w.remove_body(sup).expect("in range");
    assert_falls_and_lands(&mut w, sup, sup_y, 1.0);
}

/// A body joint-connected to the removed one is in its island and is woken
/// with it, even though it touches nothing.
#[test]
fn joint_partner_of_removed_body_is_woken() {
    let mut w = sleepy_world();
    let r = Fix128::from_ratio(1, 2);
    let a = w.add_body_with_radius(RigidBody::new_dynamic(v3(-3.0, 0.5, 0.0), Fix128::ONE), r);
    // `a` and `b` rest apart, joined by a ball joint whose anchors meet
    // above the centre; `c` rests between them and touches neither
    let b = w.add_body_with_radius(RigidBody::new_dynamic(v3(3.0, 0.5, 0.0), Fix128::ONE), r);
    let c = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE), r);
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        v3(3.0, 0.0, 0.0),
        v3(-3.0, 0.0, 0.0),
    )));
    step_until_asleep(&mut w, &[a, b, c]);
    // remove `a`: `c` (last) moves into slot `a`; `b` keeps its index
    w.remove_body(a).expect("in range");
    assert!(
        !asleep(&w, b),
        "the joint partner of the removed body must be woken"
    );
    assert!(
        asleep(&w, a),
        "the body moved into the removed slot is unrelated and stays asleep"
    );
}
