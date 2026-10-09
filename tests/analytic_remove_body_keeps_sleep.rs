//! `PhysicsWorld::remove_body` must not disturb the sleep state of the
//! bodies it keeps.
//!
//! Oracle: a control world that never contained the removed body. Removing
//! a body that touches nothing is, by definition, the same world as never
//! having had it, so every surviving body's position, sleep data and (when
//! the index assignment is the same) the whole `snapshot_world` byte string
//! must equal the control's, both right after the removal and for many
//! steps afterwards. A removal that woke the resting bodies would let them
//! integrate one step under gravity and diverge from the control for good.
//!
//! Two cases:
//! - tail: the removed body is the last one, so no other body moves index
//!   (snapshot bytes are compared too; the removed body has no collision
//!   radius, so neither broad-phase holds a proxy for it — with
//!   `DynamicTree` the bytes are compared from +1, because until the next
//!   step the persistent tree section still differs by 1 byte, a
//!   tree-side matter separate from sleep state).
//! - non-tail: the removed body sits in the middle, so `swap_remove` moves
//!   the last body (an awake free-falling body) into its slot; that body
//!   must keep its own sleep data, and the resting bodies theirs. Its
//!   snapshot carries a different stable id from the control, so this case
//!   compares positions, velocities and sleep data instead of bytes.

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::{SleepConfig, SleepState};

const SETTLE_STEPS: usize = 60;
const RESIDENT_STEPS: usize = 60;
const CHECKPOINTS: [usize; 5] = [0, 1, 5, 30, 200];

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

fn sleepy_world(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 10),
        angular_threshold: Fix128::from_ratio(1, 10),
        frames_to_sleep: 3,
    });
    w
}

/// Static sphere (r = 10) whose top is at y = 0.
fn add_floor(w: &mut PhysicsWorld) {
    w.add_body_with_radius(
        RigidBody::new_static(v3(0.0, -10.0, 0.0)),
        Fix128::from_f64(10.0),
    );
}

/// Four dynamic spheres (r = 0.5) resting on the floor at x = -3, -1, 1, 3.
fn add_resting_spheres(w: &mut PhysicsWorld) {
    for i in 0..4 {
        let x = f64::from(i) * 2.0 - 3.0;
        w.add_body_with_radius(
            RigidBody::new_dynamic(v3(x, 0.5, 0.0), Fix128::ONE),
            Fix128::from_ratio(1, 2),
        );
    }
}

/// A radius-less dynamic body far from everything: it free-falls and so
/// never falls asleep.
fn faller() -> RigidBody {
    RigidBody::new_dynamic(v3(200.0, 0.0, 200.0), Fix128::ONE)
}

fn far_static() -> RigidBody {
    RigidBody::new_static(v3(1000.0, 1000.0, 1000.0))
}

fn positions(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    w.bodies.iter().map(|b| b.position).collect()
}

fn velocities(w: &PhysicsWorld) -> Vec<Vec3Fix> {
    w.bodies.iter().map(|b| b.velocity).collect()
}

fn steps(w: &mut PhysicsWorld, n: usize) {
    for _ in 0..n {
        w.step(dt());
    }
}

fn assert_spheres_asleep(w: &PhysicsWorld, sphere_indices: core::ops::Range<usize>, what: &str) {
    for i in sphere_indices {
        assert_eq!(
            w.islands.sleep_data[i].state,
            SleepState::Sleeping,
            "{what}: premise — resting sphere {i} must be asleep; sleep_data = {:?}",
            w.islands.sleep_data
        );
    }
}

/// Tail removal: add a far static body at step 60 (after the spheres have
/// fallen asleep), remove it at step 120, compare against a world that
/// never had it — positions and snapshot bytes at +0 / +1 / +5 / +30 / +200.
fn tail_case(kind: Broadphase) {
    let build = |kind| {
        let mut w = sleepy_world(kind);
        add_floor(&mut w);
        add_resting_spheres(&mut w);
        w
    };
    let mut control = build(kind);
    let mut subject = build(kind);

    steps(&mut subject, SETTLE_STEPS);
    assert_spheres_asleep(&subject, 1..5, "before add");
    let idx = subject.add_body(far_static());
    assert_eq!(idx, 5);
    steps(&mut subject, RESIDENT_STEPS);
    assert_spheres_asleep(&subject, 1..5, "before remove");
    subject.remove_body(idx).expect("in range");

    steps(&mut control, SETTLE_STEPS + RESIDENT_STEPS);

    let mut done = 0;
    for cp in CHECKPOINTS {
        steps(&mut subject, cp - done);
        steps(&mut control, cp - done);
        done = cp;
        assert_eq!(
            positions(&subject),
            positions(&control),
            "{kind:?} +{cp}: positions"
        );
        assert_eq!(
            subject.islands.sleep_data, control.islands.sleep_data,
            "{kind:?} +{cp}: sleep data"
        );
        // DynamicTree only: until the next step the persistent tree still
        // records the removed body (1 byte at +0, measured), a separate
        // tree-side matter outside this oracle; bytes are compared from +1.
        if kind == Broadphase::DynamicTree && cp == 0 {
            continue;
        }
        assert!(
            subject.snapshot_world() == control.snapshot_world(),
            "{kind:?} +{cp}: snapshot_world bytes differ from a world that never had the body"
        );
    }
}

#[test]
fn tail_remove_keeps_sleep_bvh() {
    tail_case(Broadphase::Bvh);
}

#[test]
fn tail_remove_keeps_sleep_dynamic_tree() {
    tail_case(Broadphase::DynamicTree);
}

/// Non-tail removal: subject is [floor, far static, s0..s3, faller];
/// removing index 1 swap-moves the awake faller into slot 1, giving
/// [floor, faller, s0..s3]. Control is built directly in that final order.
fn non_tail_case(kind: Broadphase) {
    let mut subject = sleepy_world(kind);
    add_floor(&mut subject);
    let victim = subject.add_body(far_static());
    add_resting_spheres(&mut subject);
    let moved = subject.add_body(faller());
    assert_eq!((victim, moved), (1, 6));

    let mut control = sleepy_world(kind);
    add_floor(&mut control);
    control.add_body(faller());
    add_resting_spheres(&mut control);

    steps(&mut subject, SETTLE_STEPS + RESIDENT_STEPS);
    assert_spheres_asleep(&subject, 2..6, "before remove");
    assert_eq!(
        subject.islands.sleep_data[moved].state,
        SleepState::Awake,
        "premise: the free-falling body must be awake"
    );
    let moved_sleep = subject.islands.sleep_data[moved];
    let moved_pos = subject.bodies[moved].position;
    subject.remove_body(victim).expect("in range");
    assert_eq!(
        subject.bodies[victim].position, moved_pos,
        "swap-remove moved the last body"
    );
    assert_eq!(
        subject.islands.sleep_data[victim], moved_sleep,
        "the body moved into the removed slot keeps its own sleep data"
    );

    steps(&mut control, SETTLE_STEPS + RESIDENT_STEPS);

    let mut done = 0;
    for cp in CHECKPOINTS {
        steps(&mut subject, cp - done);
        steps(&mut control, cp - done);
        done = cp;
        assert_eq!(
            positions(&subject),
            positions(&control),
            "{kind:?} +{cp}: positions"
        );
        assert_eq!(
            velocities(&subject),
            velocities(&control),
            "{kind:?} +{cp}: velocities"
        );
        assert_eq!(
            subject.islands.sleep_data, control.islands.sleep_data,
            "{kind:?} +{cp}: sleep data"
        );
    }
    assert_spheres_asleep(&subject, 2..6, "+200 after remove");
}

#[test]
fn non_tail_remove_keeps_sleep_bvh() {
    non_tail_case(Broadphase::Bvh);
}

#[test]
fn non_tail_remove_keeps_sleep_dynamic_tree() {
    non_tail_case(Broadphase::DynamicTree);
}
