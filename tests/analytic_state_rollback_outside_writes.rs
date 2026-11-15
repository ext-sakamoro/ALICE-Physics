//! The rollback state saved between a write from outside the step and the next step.
//!
//! A host that rolls back writes its inputs between steps (a raw write of a position or a
//! velocity, a joint added, removed or changed, a body added or removed) and may save the state
//! right after them. The step after such a write treats it differently from the dynamics' own
//! motion: a moved body's contact points count as new and its overlap is pushed out by position,
//! a joint the dynamics did not make has its error removed by position. So the state has to carry
//! which bodies and joints were written since the last substep, and every other piece of step
//! memory the next step reads; whatever it does not carry has to come out the same when the
//! loaded world rebuilds it.
//!
//! # Oracle
//!
//! Exact, not derived from physics: the run that saves right after the write, takes another
//! branch (a velocity kick and 5 steps), loads the state and steps 30 times has to end in the
//! same bits as the run that stepped 30 times right after the write: the whole rollback state
//! (motion, sleep, step memory) and the joints, under both solvers, every broad phase and with
//! the sleep skip on and off. A control with no write checks the harness.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::joint::{BallJoint, Joint};
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{
    Broadphase, PhysicsWorld, RigidBody, SolverBackend, SolverConfig,
};

const WARM: usize = 120;
const BRANCH: usize = 5;
const REPLAY: usize = 30;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];
const KINDS: [Broadphase; 3] = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];

/// A floor (a static box body, index 0), two balls touching each other at rest on it (1, 2), a
/// cube rolling on it (3) and a ball hanging from the cube by a rigid ball joint (4).
fn scene(backend: SolverBackend, kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -10.0, 0.0),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    let floor = w.add_body(RigidBody::new_static(v3(0.0, -0.5, 0.0)));
    w.set_body_shape(
        floor,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 20.0),
        },
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(-3.0, 0.25, 0.0), Fix128::ONE),
        fx(0.25),
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(-2.5, 0.25, 0.0), Fix128::ONE),
        fx(0.25),
    );
    let mut c = RigidBody::new_dynamic(v3(1.0, 0.25, 0.0), Fix128::ONE);
    c.velocity = v3(1.0, 0.0, 0.0);
    let cube = w.add_body(c);
    w.set_body_shape(
        cube,
        &Shape::Box {
            half_extents: v3(0.25, 0.25, 0.25),
        },
    );
    let bob = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(1.0, 0.25, 1.0), Fix128::ONE),
        fx(0.2),
    );
    w.add_joint(Joint::Ball(BallJoint::new(
        cube,
        bob,
        v3(0.0, 0.0, 0.5),
        v3(0.0, 0.0, -0.5),
    )));
    w
}

/// The writes a host makes between steps.
#[derive(Clone, Copy, Debug)]
enum Write {
    None,
    Position,
    Velocity,
    AddJoint,
    RemoveJoint,
    ChangeJoint,
    AddBody,
    RemoveBody,
}

const WRITES: [Write; 8] = [
    Write::None,
    Write::Position,
    Write::Velocity,
    Write::AddJoint,
    Write::RemoveJoint,
    Write::ChangeJoint,
    Write::AddBody,
    Write::RemoveBody,
];

fn write(w: &mut PhysicsWorld, what: Write) {
    match what {
        Write::None => {}
        // the resting ball woken and pushed 0.05 into the floor (a write to a sleeping body
        // that is not woken moves nothing the step reads)
        Write::Position => {
            w.wake_body(1);
            let p = w.bodies[1].position;
            w.bodies[1].position = Vec3Fix::new(p.x, p.y - fx(0.05), p.z);
        }
        Write::Velocity => {
            w.wake_body(2);
            w.bodies[2].velocity = v3(-2.0, 1.0, 0.5);
        }
        // the two resting balls pinned 0.1 apart from where they touch
        Write::AddJoint => {
            w.add_joint(Joint::Ball(BallJoint::new(
                1,
                2,
                v3(0.3, 0.0, 0.0),
                v3(-0.3, 0.0, 0.0),
            )));
        }
        Write::RemoveJoint => {
            assert!(w.remove_joint(0).is_some());
        }
        Write::ChangeJoint => match &mut w.joints[0] {
            Joint::Ball(j) => j.local_anchor_b = v3(0.0, 0.1, -0.5),
            other => panic!("not the scene's joint: {other:?}"),
        },
        // a ball dropped into the resting pair
        Write::AddBody => {
            w.add_body_with_radius(
                RigidBody::new_dynamic(v3(-2.75, 0.4, 0.0), Fix128::ONE),
                fx(0.2),
            );
        }
        Write::RemoveBody => {
            assert!(w.remove_body(2).is_some());
        }
    }
}

/// The other branch: the resting ball woken and thrown up.
fn kick(w: &mut PhysicsWorld) {
    w.wake_body(1);
    w.bodies[1].velocity = v3(0.0, 3.0, 0.0);
}

type End = (Vec<u8>, Vec<Joint>);

fn end(w: &PhysicsWorld) -> End {
    (w.serialize_state(), w.joints.clone())
}

/// `(straight, replayed)`: the write, then `REPLAY` steps; and the write, a save, another branch
/// (body 1 woken and thrown up, `BRANCH` steps), the load and `REPLAY` steps.
fn run(backend: SolverBackend, kind: Broadphase, skip: bool, what: Write) -> (End, End) {
    let one = |restore: bool| {
        let mut w = scene(backend, kind, skip);
        for _ in 0..WARM {
            w.step(dt());
        }
        write(&mut w, what);
        let blob = w.serialize_state();
        if restore {
            kick(&mut w);
            for _ in 0..BRANCH {
                w.step(dt());
            }
            assert!(
                w.deserialize_state(&blob),
                "{backend:?} {kind:?} skip {skip} {what:?}: refused"
            );
        }
        for _ in 0..REPLAY {
            w.step(dt());
        }
        end(&w)
    };
    (one(false), one(true))
}

#[test]
fn a_state_saved_right_after_any_outside_write_replays_bit_for_bit() {
    let mut split = Vec::new();
    let mut compared = 0;
    for backend in BACKENDS {
        for kind in KINDS {
            for skip in [false, true] {
                for what in WRITES {
                    let (straight, replayed) = run(backend, kind, skip, what);
                    compared += 1;
                    if straight != replayed {
                        split.push(format!("{backend:?} {kind:?} skip {skip} {what:?}"));
                    }
                }
            }
        }
    }
    assert_eq!(compared, 2 * 3 * 2 * WRITES.len());
    assert!(
        split.is_empty(),
        "{} of {compared} left the straight run: {split:?}",
        split.len()
    );
}

/// The branch the replays go through does differ from the straight run (the load has
/// something to undo), and the scene's resting pair is asleep when the sleep skip is on (so the
/// park and its tree are in play).
#[test]
fn the_branch_differs_and_the_resting_pair_sleeps() {
    for backend in BACKENDS {
        for kind in KINDS {
            let mut a = scene(backend, kind, true);
            for _ in 0..WARM {
                a.step(dt());
            }
            assert!(
                a.is_sleeping(1) && a.is_sleeping(2),
                "{backend:?} {kind:?}: awake"
            );
            let mut b = scene(backend, kind, true);
            for _ in 0..WARM {
                b.step(dt());
            }
            kick(&mut b);
            for _ in 0..BRANCH {
                a.step(dt());
                b.step(dt());
            }
            assert_ne!(
                end(&a),
                end(&b),
                "{backend:?} {kind:?}: the branch changed nothing"
            );
        }
    }
}

fn pose_bytes(i: u64, p: Vec3Fix, q: QuatFix) -> Vec<u8> {
    let mut b = i.to_le_bytes().to_vec();
    for f in [p.x, p.y, p.z, q.x, q.y, q.z, q.w] {
        b.extend_from_slice(&f.hi.to_le_bytes());
        b.extend_from_slice(&f.lo.to_le_bytes());
    }
    b
}

/// The written bodies and joints a state names have to fit the world it is loaded into, as the
/// solver keeps them: ascending, and indices of its bodies / joints. Saved right after bodies 1
/// and 2 are moved, the scene's joint is changed (joint 0) and a joint is added (joint 1), the
/// state holds, after the poses of the last substep's start, the count 2, body 1 and 2 each with
/// the pose the dynamics left it at, the count 2 and joints 0 and 1; found by those bytes, each is broken in turn and the load is refused with
/// the world unchanged.
#[test]
fn a_state_naming_writes_that_do_not_fit_the_world_is_refused_unchanged() {
    let mut w = scene(SolverBackend::Xpbd, Broadphase::Bvh, false);
    for _ in 0..WARM {
        w.step(dt());
    }
    let left: Vec<(Vec3Fix, QuatFix)> = (1..=2)
        .map(|i| (w.bodies[i].position, w.bodies[i].rotation))
        .collect();
    for i in 1..=2 {
        w.wake_body(i);
        let p = w.bodies[i].position;
        w.bodies[i].position = Vec3Fix::new(p.x, p.y + fx(0.05), p.z);
    }
    write(&mut w, Write::ChangeJoint);
    write(&mut w, Write::AddJoint);
    let blob = w.serialize_state();
    let (b1, b2) = (
        pose_bytes(1, left[0].0, left[0].1),
        pose_bytes(2, left[1].0, left[1].1),
    );
    let mut section = 2u32.to_le_bytes().to_vec();
    section.extend_from_slice(&b1);
    section.extend_from_slice(&b2);
    section.extend_from_slice(&2u32.to_le_bytes());
    section.extend_from_slice(&0u64.to_le_bytes());
    section.extend_from_slice(&1u64.to_le_bytes());
    let at: Vec<usize> = (0..=blob.len() - section.len())
        .filter(|&k| blob[k..k + section.len()] == section[..])
        .collect();
    assert_eq!(
        at.len(),
        1,
        "the written bodies and joints are not in the state once"
    );
    let at = at[0];
    let first = at + 4;
    let second = first + b1.len();
    let joint = second + b2.len() + 4;
    let index =
        |blob: &mut Vec<u8>, k: usize, v: u64| blob[k..k + 8].copy_from_slice(&v.to_le_bytes());
    let mut past_bodies = blob.clone();
    index(&mut past_bodies, second, w.bodies.len() as u64);
    let mut out_of_order = blob.clone();
    out_of_order[first..first + b1.len()].copy_from_slice(&b2);
    out_of_order[second..second + b2.len()].copy_from_slice(&b1);
    let mut repeated = blob.clone();
    repeated[second..second + b2.len()].copy_from_slice(&b1);
    let mut past_joints = blob.clone();
    index(&mut past_joints, joint + 8, w.joints.len() as u64);
    let mut joints_out_of_order = blob.clone();
    index(&mut joints_out_of_order, joint, 1);
    index(&mut joints_out_of_order, joint + 8, 0);
    let mut joint_repeated = blob.clone();
    index(&mut joint_repeated, joint + 8, 0);
    let before = w.serialize_state();
    for (what, bad) in [
        ("a body past the bodies", past_bodies),
        ("bodies out of order", out_of_order),
        ("a body named twice", repeated),
        ("a joint past the joints", past_joints),
        ("joints out of order", joints_out_of_order),
        ("a joint named twice", joint_repeated),
    ] {
        assert!(!w.deserialize_state(&bad), "{what}: accepted");
        assert_eq!(w.serialize_state(), before, "{what}: the world changed");
    }
    assert!(w.deserialize_state(&blob), "the state itself was refused");
}

/// The scene with one more ball (index 5) rolling on the floor away from the rest.
fn scene_with_a_roller(backend: SolverBackend, kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = scene(backend, kind, skip);
    let mut r = RigidBody::new_dynamic(v3(5.0, 0.25, 3.0), Fix128::ONE);
    r.velocity = v3(0.0, 0.0, 1.0);
    w.add_body_with_radius(r, fx(0.25));
    w
}

/// A branch that removes a body (or a joint) and adds an equal one back at the same index leaves
/// the body count, every body's mass and the joints the same, so the state loads; the replay has
/// to come out as if the branch never happened, under both solvers, every broad phase and with
/// the sleep skip on and off. The TGS warm start names a contact or a joint by the stable ids of
/// its bodies (and its ordinal among the entries of the same pair): a body added back has a new
/// id, so the state carries the ids. A joint has no id of its own.
#[test]
fn a_branch_that_removes_a_body_or_a_joint_and_adds_an_equal_one_back_replays_bit_for_bit() {
    let mut split = Vec::new();
    let mut compared = 0;
    for joint in [false, true] {
        for backend in BACKENDS {
            for kind in KINDS {
                for skip in [false, true] {
                    let one = |restore: bool| {
                        let mut w = scene_with_a_roller(backend, kind, skip);
                        for _ in 0..WARM {
                            w.step(dt());
                        }
                        let blob = w.serialize_state();
                        if restore {
                            if joint {
                                let gone = w.remove_joint(0).expect("the scene's joint");
                                for _ in 0..BRANCH {
                                    w.step(dt());
                                }
                                assert_eq!(w.add_joint(gone), 0);
                            } else {
                                let gone = w.remove_body(5).expect("the roller");
                                for _ in 0..BRANCH {
                                    w.step(dt());
                                }
                                let back = w.add_body_with_radius_keeping_inertia(gone, fx(0.25));
                                assert_eq!(back, 5);
                            }
                            w.step(dt());
                            assert!(
                                w.deserialize_state(&blob),
                                "{backend:?} {kind:?} skip {skip} joint {joint}: refused"
                            );
                        }
                        for _ in 0..REPLAY {
                            w.step(dt());
                        }
                        end(&w)
                    };
                    compared += 1;
                    if one(false) != one(true) {
                        split.push(format!("{backend:?} {kind:?} skip {skip} joint {joint}"));
                    }
                }
            }
        }
    }
    assert_eq!(compared, 2 * 2 * 3 * 2);
    assert!(
        split.is_empty(),
        "{} of {compared} left the straight run: {split:?}",
        split.len()
    );
}

/// A body removed between steps leaves its TGS warm-start entries in the cache until the next
/// step sweeps them, so the next id has to be carried as it is: an id derived from the ids held
/// (the largest plus one) would hand the removed body's id to a body added right after the load,
/// and that body would pick up the removed body's impulses where the straight run gives it a
/// fresh id. Saved right after the removal, the replay that adds an equal body back has to match
/// the straight run, warm-start hits and misses included, under both solvers, every broad phase
/// and with the sleep skip on and off. The carried counter decides which warm-start entry a new
/// body takes, and the hit count is where that shows: with the next id derived from the ids held,
/// the body added after the load picks up the removed body's entry (a rolling ball: 8 hits and
/// no miss where the straight run has 7 and 1) while the motion stayed bit-equal in every scene
/// measured under TGS — the same ball added back, a ball five times heavier, a box on its four
/// corners, a stack of three boxes with one substep and one iteration, and a chain of three balls
/// with the new ball jointed to its end. A single contact with no restitution reaches the same end
/// velocity from any warm start; the box and the stack got no stale hit (their corner features
/// differ); the chain did (72 hits and 6 misses against 71 and 7) and still moved the same. A
/// test on the motion alone would pass that mutant, so the counts are compared as well as the
/// state bytes.
#[test]
fn a_state_saved_between_a_removal_and_the_next_step_hands_out_the_same_next_id() {
    let mut split = Vec::new();
    let mut compared = 0;
    for backend in BACKENDS {
        for kind in KINDS {
            for skip in [false, true] {
                let one = |restore: bool| {
                    let mut w = scene_with_a_roller(backend, kind, skip);
                    for _ in 0..WARM {
                        w.step(dt());
                    }
                    let gone = w.remove_body(5).expect("the roller");
                    let blob = w.serialize_state();
                    if restore {
                        kick(&mut w);
                        for _ in 0..BRANCH {
                            w.step(dt());
                        }
                        assert!(
                            w.deserialize_state(&blob),
                            "{backend:?} {kind:?} skip {skip}: refused"
                        );
                    }
                    // the hit and miss counters are this world's diagnostics, not state
                    w.reset_tgs_cache_stats();
                    assert_eq!(w.add_body_with_radius_keeping_inertia(gone, fx(0.25)), 5);
                    for _ in 0..REPLAY {
                        w.step(dt());
                    }
                    let stats = w.tgs_cache_stats();
                    (end(&w), stats.hits, stats.misses)
                };
                compared += 1;
                if one(false) != one(true) {
                    split.push(format!("{backend:?} {kind:?} skip {skip}"));
                }
            }
        }
    }
    assert_eq!(compared, 2 * 3 * 2);
    assert!(
        split.is_empty(),
        "{} of {compared} left the straight run: {split:?}",
        split.len()
    );
}
