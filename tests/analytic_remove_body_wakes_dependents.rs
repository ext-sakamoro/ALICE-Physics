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

// ---- immovable supports, tolerance, SDF colliders, transfer -------------

/// Removing the support leaves a body in free fall: woken, and its height
/// strictly decreasing step by step, by more than 0.5 over 30 steps (free
/// fall under the default gravity of 10 m/s² covers about 1.2 m in 0.5 s,
/// less the velocity retention of the default damping).
fn assert_free_falls(w: &mut PhysicsWorld, i: usize, what: &str) {
    assert!(!asleep(w, i), "{what}: the unsupported body must be woken");
    let y0 = w.bodies[i].position.y.to_f64();
    let mut prev = y0;
    for k in 0..30 {
        w.step(dt());
        let y = w.bodies[i].position.y.to_f64();
        assert!(
            y < prev,
            "{what}: step {k}: y {y} did not fall below {prev}"
        );
        prev = y;
    }
    assert!(y0 - prev > 0.5, "{what}: fell only from {y0} to {prev}");
}

/// A ball resting on a static sphere floor sits one ulp (`2^-64`) clear of
/// the floor's box; removing the floor must still wake it.
#[test]
fn ball_on_removed_static_sphere_floor_falls() {
    for (num, den) in [(1, 4), (1, 2), (13, 10)] {
        let mut w = sleepy_world(); // index 0: static sphere r = 10, top at y = 0
        let r = Fix128::from_ratio(num, den);
        let ball = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, r, Fix128::ZERO), Fix128::ONE),
            r,
        );
        step_until_asleep(&mut w, &[ball]);
        assert!(
            w.bodies[ball].position.y - r > Fix128::ZERO,
            "premise: the ball rests clear of the floor box"
        );
        w.remove_body(0).expect("in range");
        // the ball (last) moved into slot 0
        assert_free_falls(&mut w, 0, &format!("ball r = {num}/{den}"));
    }
}

/// The same with a static box floor (a box collider), whose top the ball
/// rests on with no gap.
#[test]
fn ball_on_removed_static_box_floor_falls() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 10),
        angular_threshold: Fix128::from_ratio(1, 10),
        frames_to_sleep: 3,
    });
    let floor = w.add_body_with_radius(
        RigidBody::new_static(v3(0.0, -1.0, 0.0)),
        Fix128::from_int(3),
    );
    assert!(w.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(2.0, 1.0, 2.0),
        }
    ));
    let r = Fix128::from_ratio(1, 2);
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE), r);
    step_until_asleep(&mut w, &[ball]);
    w.remove_body(floor).expect("in range");
    assert_free_falls(&mut w, 0, "ball on box floor");
}

/// A static body touching the removed one is not woken (static bodies are
/// asleep by definition and nothing integrates them).
#[test]
fn static_floor_under_removed_body_is_not_woken() {
    let mut w = sleepy_world();
    let r = Fix128::from_ratio(1, 2);
    let ball = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE), r);
    let other = w.add_body_with_radius(RigidBody::new_dynamic(v3(3.0, 0.5, 0.0), Fix128::ONE), r);
    step_until_asleep(&mut w, &[0, ball, other]);
    w.remove_body(ball).expect("in range");
    assert!(asleep(&w, 0), "the static floor stays asleep");
    assert!(
        asleep(&w, 1),
        "the unrelated ball (moved into slot 1) stays asleep"
    );
}

/// Separation thresholds, without gravity so the bodies stay where placed.
/// The gap is the exact distance between the two broad-phase boxes, in
/// `Fix128` ulps (`2^-64`): the removed ball's box is grown by `2^-56`, so a
/// ball up to that far away is woken (a resting body is left one ulp clear
/// of an immovable support, plus rounding) and one farther away is not.
#[test]
fn box_gap_thresholds_of_removed_body() {
    let ulps = |lo: u64| Fix128::from_raw(0, lo);
    let cases = [
        (Fix128::ZERO, true, "touching"),
        (ulps(1), true, "1 ulp (rest on an immovable support)"),
        (ulps(2), true, "2 ulps"),
        (ulps(1 << 7), true, "2^-57"),
        (ulps(1 << 8), true, "2^-56, the bound itself"),
        (ulps(1 << 24), false, "2^-40"),
        (Fix128::from_f64(1e-9), false, "1e-9"),
    ];
    for (gap, woken, what) in cases {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let r = Fix128::from_ratio(1, 2);
        let removed =
            w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE), r);
        w.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(Fix128::ONE + gap, Fix128::ZERO, Fix128::ZERO),
                Fix128::ONE,
            ),
            r,
        );
        w.islands.resize(2);
        for sd in &mut w.islands.sleep_data {
            sd.state = SleepState::Sleeping;
        }
        w.remove_body(removed).expect("in range");
        assert_eq!(
            !asleep(&w, 0),
            woken,
            "gap {what}: near ball (moved into slot 0)"
        );
    }
}

/// Bodies pushed onto the public `bodies` field have no per-body entries;
/// removing one, or moving one by removing another, must not panic, and the
/// sleeping bodies stay asleep.
#[test]
fn removing_around_bodies_pushed_onto_the_public_field() {
    let mut w = sleepy_world();
    let r = Fix128::from_ratio(1, 2);
    let a = w.add_body_with_radius(RigidBody::new_dynamic(v3(-3.0, 0.5, 0.0), Fix128::ONE), r);
    step_until_asleep(&mut w, &[a]);
    w.bodies
        .push(RigidBody::new_dynamic(v3(0.0, 50.0, 0.0), Fix128::ONE));
    w.bodies
        .push(RigidBody::new_dynamic(v3(0.0, 60.0, 0.0), Fix128::ONE));
    // the pushed tail body itself
    assert!(w.remove_body(3).is_some());
    assert_eq!(w.bodies.len(), 3);
    assert!(asleep(&w, a), "the resting ball stays asleep");
    // a pushed body moved into the slot of a removed body without entries
    w.bodies
        .push(RigidBody::new_dynamic(v3(0.0, 70.0, 0.0), Fix128::ONE));
    assert!(w.remove_body(2).is_some());
    assert_eq!(w.bodies.len(), 3);
    assert_eq!(w.bodies[2].position, v3(0.0, 70.0, 0.0));
    assert!(asleep(&w, a));
    assert_eq!(w.islands.sleep_data.len(), 3);
    w.step(dt());
}

#[cfg(feature = "std")]
mod sdf {
    use super::*;
    use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

    fn plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    /// floor-less world: a static support at y = 1 carrying an attached
    /// plane field (surface at y = 1), a radius-less body resting on it.
    fn sdf_scene() -> (PhysicsWorld, usize, usize) {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        w.set_sleep_config(SleepConfig {
            linear_threshold: Fix128::from_ratio(1, 10),
            angular_threshold: Fix128::from_ratio(1, 10),
            frames_to_sleep: 3,
        });
        let support = w.add_body(RigidBody::new_static(v3(0.0, 1.0, 0.0)));
        w.add_sdf_collider(SdfCollider::new_dynamic(Box::new(plane()), support));
        let body = w.add_body(RigidBody::new_dynamic(v3(0.0, 1.5, 0.0), Fix128::ONE));
        step_until_asleep(&mut w, &[body]);
        (w, support, body)
    }

    #[test]
    fn body_on_removed_sdf_support_falls() {
        let (mut w, support, _body) = sdf_scene();
        assert!(
            (w.bodies[1].position.y.to_f64() - 1.5).abs() < 1e-2,
            "premise: rests on the field"
        );
        w.remove_body(support).expect("in range");
        assert!(
            w.sdf_colliders.is_empty(),
            "the attached field went with its body"
        );
        assert_free_falls(&mut w, 0, "body on SDF support");
    }

    /// A zero-gravity world: a static support at y = 1 carrying an attached
    /// plane field (surface at y = 1), a sleeping radius-less body at
    /// `gap` above touching it (sphere of `sdf_collision_radius` 0.5), and a
    /// sleeping static body touching the field. Returns the world after the
    /// support is removed.
    fn sdf_gap_scene(gap: f64) -> PhysicsWorld {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let support = w.add_body(RigidBody::new_static(v3(0.0, 1.0, 0.0)));
        w.add_sdf_collider(SdfCollider::new_dynamic(Box::new(plane()), support));
        w.add_body(RigidBody::new_dynamic(v3(0.0, 1.5 + gap, 0.0), Fix128::ONE)); // 1
        w.add_body(RigidBody::new_static(v3(3.0, 1.5, 0.0))); // 2, moves into 0
        w.islands.resize(3);
        for sd in &mut w.islands.sleep_data {
            sd.state = SleepState::Sleeping;
        }
        w.remove_body(support).expect("in range");
        w
    }

    /// The SDF wake reaches `2^-12` (about 2.4e-4) beyond touching, measured
    /// in `f32` the way the SDF contact path measures it.
    #[test]
    fn sdf_gap_thresholds_of_removed_field() {
        for (gap, woken) in [
            (0.0, true),
            (1e-5, true),
            (1e-3, false),
            (0.1, false),
            (2.0, false),
        ] {
            let w = sdf_gap_scene(gap);
            assert_eq!(!asleep(&w, 1), woken, "body {gap} above the field");
        }
    }

    /// A static body touching the removed body's field is not woken.
    #[test]
    fn static_body_at_removed_field_is_not_woken() {
        let w = sdf_gap_scene(0.0);
        assert!(
            w.bodies[0].is_static(),
            "premise: the static body moved into slot 0"
        );
        assert!(asleep(&w, 0), "the static body stays asleep");
        assert!(!asleep(&w, 1), "premise: the dynamic body there is woken");
    }

    /// Removing a body drops its attached colliders in order and points the
    /// collider of the moved last body at its new index; static colliders
    /// and colliders of other bodies are untouched.
    #[test]
    fn attached_colliders_are_dropped_and_remapped() {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        for i in 0..4 {
            w.add_body(RigidBody::new_static(v3(f64::from(i) * 10.0, 0.0, 0.0)));
        }
        let sdf = |b: usize| SdfCollider::new_dynamic(Box::new(plane()), b);
        w.add_sdf_collider(sdf(1)); // 0
        w.add_sdf_collider(SdfCollider::new_static(
            Box::new(plane()),
            Vec3Fix::ZERO,
            alice_physics::math::QuatFix::IDENTITY,
        )); // 1
        w.add_sdf_collider(sdf(3)); // 2
        w.add_sdf_collider(sdf(1)); // 3
        w.add_sdf_collider(sdf(2)); // 4
        w.remove_body(1).expect("in range");
        let idx: Vec<usize> = w.sdf_colliders.iter().map(|c| c.body_index).collect();
        assert_eq!(
            idx,
            vec![alice_physics::sdf_collider::SDF_STATIC, 1, 2],
            "body 1's two colliders dropped, body 3's collider follows it to 1"
        );
        assert_eq!(w.sdf_colliders[1].position, w.bodies[1].position);
    }

    /// Moving a body to another world moves its attached field with it.
    #[test]
    fn transfer_moves_the_attached_collider() {
        use alice_physics::MultiWorld;
        let mut mw = MultiWorld::new();
        let a = mw.add_world(SolverConfig::default());
        let b = mw.add_world(SolverConfig::default());
        let keep = mw.worlds[a].add_body(RigidBody::new_static(v3(5.0, 0.0, 0.0)));
        let moved = mw.worlds[a].add_body(RigidBody::new_static(v3(0.0, 1.0, 0.0)));
        mw.worlds[a].add_sdf_collider(SdfCollider::new_dynamic(Box::new(plane()), moved));
        mw.worlds[a].add_sdf_collider(SdfCollider::new_dynamic(Box::new(plane()), keep));
        let new_id = mw
            .transfer_body(a, moved, b, v3(7.0, 2.0, 0.0))
            .expect("transferred");
        assert_eq!(mw.worlds[a].sdf_colliders.len(), 1);
        assert_eq!(mw.worlds[a].sdf_colliders[0].body_index, keep);
        assert_eq!(mw.worlds[b].sdf_colliders.len(), 1);
        let c = &mw.worlds[b].sdf_colliders[0];
        assert_eq!(c.body_index, new_id);
        assert_eq!(
            c.position,
            v3(7.0, 2.0, 0.0),
            "placed at the body's new pose"
        );
    }
}
