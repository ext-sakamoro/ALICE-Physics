//! Oracle: the overflow flag on a pair whose radius sum leaves the `Fix128`
//! range follows one rule on every path.
//!
//! # The rule (`PhysicsWorld::pair_radius_sum_overflows`)
//!
//! The flag is raised for a pair iff the pair is one the simulation solves
//! (neither body static, their layers collide) and their broad-phase boxes meet
//! (closed intervals), the test the exact-overlap broad-phase makes. The rule is
//! evaluated only when the sum does not fit, so a world in range never reaches
//! it. The narrow phase of every broad-phase kind, the insertion into the tree
//! of parked bodies (sleep skip) and a restore that lowered the flag (the
//! parked bodies are re-parked on the next step) all go through it, so the
//! flag does not depend on which path found the pair.
//!
//! Scenes: two bodies of collision radius `2⁶²` (a sum of `2⁶³`, past the range)
//! at `x = 0` and `x = 2⁴⁰`, in every role (awake, static pair, layers that do
//! not collide, parked), and two unit ellipsoids carrying that radius whose tight
//! boxes meet or not by their gap, under the three kinds with the sleep skip on
//! and off.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::sleeping::SleepState;
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

const KINDS: [Broadphase; 3] = [Broadphase::Bvh, Broadphase::DynamicTree, Broadphase::Hybrid];
const R: Fix128 = Fix128 { hi: 1 << 62, lo: 0 };

fn x_at(x: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, 0, 0)
}

fn world(kind: Broadphase, skip: bool) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    w.set_sleep_skip(skip);
    w
}

fn sleep(w: &mut PhysicsWorld, i: usize) {
    w.islands.sleep_data[i].state = SleepState::Sleeping;
    w.islands.sleep_data[i].idle_frames = 100;
}

fn step(w: &mut PhysicsWorld, n: usize) {
    for _ in 0..n {
        w.step(Fix128::from_ratio(1, 60));
    }
}

/// Every (kind, skip) combination.
fn combos() -> Vec<(Broadphase, bool)> {
    KINDS
        .iter()
        .flat_map(|&k| [(k, false), (k, true)])
        .collect()
}

/// `build` makes the pair in a fresh world; the flag after `steps` steps must be
/// `want` under every combination, and the states must agree across kinds.
fn every_path_gives(want: bool, steps: usize, what: &str, build: impl Fn(&mut PhysicsWorld)) {
    let mut states: Vec<(String, bool, Vec<u8>)> = Vec::new();
    for (kind, skip) in combos() {
        let mut w = world(kind, skip);
        build(&mut w);
        step(&mut w, steps);
        states.push((
            format!("{kind:?} skip={skip}"),
            w.overflow_detected(),
            w.serialize_state(),
        ));
    }
    for (tag, flag, _) in &states {
        assert_eq!(*flag, want, "{what}: {tag}");
    }
    let first = &states[0].2;
    for (tag, _, state) in &states[1..] {
        assert_eq!(
            state, first,
            "{what}: {tag} steps to a different state than {}",
            states[0].0
        );
    }
}

#[test]
fn an_awake_pair_the_simulation_solves_raises_the_flag_on_every_path() {
    every_path_gives(true, 1, "awake pair", |w| {
        w.add_body_with_radius(RigidBody::new_dynamic(x_at(0), Fix128::ONE), R);
        w.add_body_with_radius(RigidBody::new_dynamic(x_at(1 << 40), Fix128::ONE), R);
    });
}

#[test]
fn a_static_pair_never_raises_the_flag() {
    every_path_gives(false, 1, "static pair", |w| {
        w.add_body_with_radius(RigidBody::new_static(x_at(0)), R);
        w.add_body_with_radius(RigidBody::new_static(x_at(1 << 40)), R);
    });
}

#[test]
fn a_static_and_a_dynamic_body_raise_the_flag_on_every_path() {
    every_path_gives(true, 3, "static + parked", |w| {
        w.add_body_with_radius(RigidBody::new_static(x_at(0)), R);
        let b = w.add_body_with_radius(RigidBody::new_dynamic(x_at(1 << 40), Fix128::ONE), R);
        sleep(w, b);
    });
}

#[test]
fn a_pair_whose_layers_do_not_collide_never_raises_the_flag() {
    every_path_gives(false, 1, "filtered pair", |w| {
        let a = w.add_body_with_radius(RigidBody::new_dynamic(x_at(0), Fix128::ONE), R);
        let b = w.add_body_with_radius(RigidBody::new_dynamic(x_at(1 << 40), Fix128::ONE), R);
        w.set_body_filter(a, CollisionFilter::new(1, 1));
        w.set_body_filter(b, CollisionFilter::new(2, 2));
    });
}

#[test]
fn a_parked_pair_raises_the_flag_with_the_skip_on_as_without() {
    every_path_gives(true, 3, "parked pair", |w| {
        w.add_body_with_radius(RigidBody::new_dynamic(x_at(0), Fix128::ONE), R);
        w.add_body_with_radius(RigidBody::new_dynamic(x_at(1 << 40), Fix128::ONE), R);
        sleep(w, 0);
        sleep(w, 1);
    });
}

/// Two unit ellipsoids carrying the radius `2⁶²`: their broad-phase boxes are the
/// colliders' tight boxes, so the pair raises the flag exactly when those boxes
/// meet (gap ≤ 0, closed intervals), under every kind, parked or awake.
#[test]
fn a_collider_pair_raises_the_flag_exactly_when_its_tight_boxes_meet() {
    for gap_q in [-1i64, 0, 1, 3, 5, 12] {
        for parked in [false, true] {
            let gap = Fix128::from_ratio(gap_q, 4);
            every_path_gives(
                gap_q <= 0,
                3,
                &format!("ellipsoids gap={gap_q}/4 parked={parked}"),
                |w| {
                    let a = w.add_body_with_radius(RigidBody::new_dynamic(x_at(0), Fix128::ONE), R);
                    let b = w.add_body_with_radius(
                        RigidBody::new_dynamic(
                            Vec3Fix::new(Fix128::from_int(2) + gap, Fix128::ZERO, Fix128::ZERO),
                            Fix128::ONE,
                        ),
                        R,
                    );
                    for i in [a, b] {
                        assert!(w.set_body_shape(
                            i,
                            &Shape::Ellipsoid {
                                radii: Vec3Fix::from_int(1, 1, 1)
                            }
                        ));
                        w.set_body_collision_radius(i, R);
                        if parked {
                            sleep(w, i);
                        }
                    }
                },
            );
        }
    }
}

/// A snapshot taken below the flag, loaded into a world that raised it: the flag
/// follows the snapshot (down), and the next step raises it again on every path,
/// with the skip on as without (the parked bodies are re-parked after a load).
#[test]
fn a_load_below_the_flag_raises_it_again_on_the_next_step() {
    for (kind, skip) in combos() {
        let build = |w: &mut PhysicsWorld| {
            w.add_body_with_radius(RigidBody::new_dynamic(x_at(0), Fix128::ONE), R);
            w.add_body_with_radius(RigidBody::new_dynamic(x_at(1 << 40), Fix128::ONE), R);
            sleep(w, 0);
            sleep(w, 1);
        };
        let mut below = world(kind, skip);
        build(&mut below);
        assert!(!below.overflow_detected());
        let snapshot = below.serialize_state();
        let mut w = world(kind, skip);
        build(&mut w);
        step(&mut w, 3);
        assert!(
            w.overflow_detected(),
            "{kind:?} skip={skip}: raised before the load"
        );
        assert!(w.deserialize_state(&snapshot), "{kind:?} skip={skip}: load");
        assert!(
            !w.overflow_detected(),
            "{kind:?} skip={skip}: the flag follows the snapshot"
        );
        step(&mut w, 1);
        assert!(
            w.overflow_detected(),
            "{kind:?} skip={skip}: raised again after one step"
        );
    }
}
