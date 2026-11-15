//! A world snapshot taken after the bodies were churned restores to a world that steps on as
//! the original does.
//!
//! The TGS warm start names a contact or a joint by the stable ids of its bodies
//! (`PhysicsWorld::step_tgs`). Removing a body that is not the last one moves the last body into
//! its index (`swap_remove`) with its id, and a body added afterwards gets a new id, so after
//! such churn the ids are no longer `0..n`. A snapshot has to carry them: a world restored from
//! it that named its bodies `0..n` again would look up the carried warm-start entries under other
//! keys, miss them, and leave the straight run on its first step.
//!
//! Scene: a floor body, two balls resting against each other, a rolling cube and a ball hanging
//! from it by a ball joint. Churn: the first ball is removed (the hanging ball takes its index)
//! and an equal ball is added back, then the world steps until the warm-start cache holds the
//! new keys. Compared: the snapshot bytes and the warm-start hits and misses after one step and
//! after 30, straight against restored, under both solvers and every broad phase, through both
//! entry points (`from_world_snapshot` and `restore_world` into a world built the same way).
//! Author: Moroya Sakamoto

use alice_physics_core::joint::{BallJoint, Joint};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{
    Broadphase, PhysicsWorld, RigidBody, SolverBackend, SolverConfig,
};

const WARM: usize = 60;
const SETTLE: usize = 20;
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

fn scene(backend: SolverBackend, kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -10.0, 0.0),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
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

/// The scene warmed up, its first ball removed (the hanging ball, the last body, takes index 1
/// with its id) and an equal ball added back (index 4, a new id), then stepped until the
/// warm-start cache is keyed by the churned ids.
fn churned(backend: SolverBackend, kind: Broadphase) -> PhysicsWorld {
    let mut w = scene(backend, kind);
    for _ in 0..WARM {
        w.step(dt());
    }
    let gone = w.remove_body(1).expect("the first ball");
    assert_eq!(w.add_body_with_radius_keeping_inertia(gone, fx(0.25)), 4);
    for _ in 0..SETTLE {
        w.step(dt());
    }
    w
}

/// After one step and after `REPLAY`: the snapshot bytes and the warm-start hits and misses
/// counted from the snapshot on.
fn run(mut w: PhysicsWorld) -> Vec<(Vec<u8>, u64, u64)> {
    w.reset_tgs_cache_stats();
    let mut out = Vec::new();
    for n in 1..=REPLAY {
        w.step(dt());
        if n == 1 || n == REPLAY {
            let s = w.tgs_cache_stats();
            out.push((w.snapshot_world(), s.hits, s.misses));
        }
    }
    out
}

#[test]
fn a_snapshot_after_the_bodies_were_churned_steps_on_as_the_original() {
    let mut split = Vec::new();
    let mut compared = 0;
    for backend in BACKENDS {
        for kind in KINDS {
            let blob = churned(backend, kind).snapshot_world();
            let straight = run(churned(backend, kind));
            let from = PhysicsWorld::from_world_snapshot(&blob).expect("the snapshot restores");
            let mut into = scene(backend, kind);
            into.restore_world(&blob)
                .expect("the snapshot restores into the scene");
            for (entry, restored) in [("from_world_snapshot", from), ("restore_world", into)] {
                compared += 1;
                let replay = run(restored);
                for (at, (s, r)) in ["first step", "step 30"]
                    .iter()
                    .zip(straight.iter().zip(&replay))
                {
                    if s != r {
                        split.push(format!(
                            "{backend:?} {kind:?} {entry} {at}: hits/misses {}/{} vs {}/{}{}",
                            s.1,
                            s.2,
                            r.1,
                            r.2,
                            if s.0 == r.0 { "" } else { ", bytes differ" }
                        ));
                    }
                }
            }
        }
    }
    assert_eq!(compared, 2 * 3 * 2);
    assert!(
        split.is_empty(),
        "{} replays left the straight run: {split:#?}",
        split.len()
    );
}
