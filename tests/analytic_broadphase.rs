//! Oracles for the selectable broad-phase of a `PhysicsWorld`.
//!
//! # What is measured
//!
//! `Broadphase::Bvh` (rebuilt every step) and `Broadphase::DynamicTree` (a
//! persistent tree of fattened proxies) both hand **sorted candidate pairs** to the
//! same exact narrow-phase, so two worlds built alike must step to *bit-identical*
//! states whichever they use. That is the main oracle: a dense, deterministic scene
//! of spheres and shaped bodies is stepped under both and every field of every body
//! compared, also across `add_body`, `remove_body` (which renumbers bodies) and a
//! change of collision radius, and across a switch of broad-phase mid-run.
//!
//! The tree's own bookkeeping has closed forms: a fresh proxy's box is the tight
//! sphere box fattened by exactly the margin 0.5, a body that moves less than that
//! keeps the same proxy box and one that moves more is re-inserted around its new
//! box, and the proxy count is the number of bodies with a collision radius.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// A tiny deterministic generator (a 64-bit LCG, high bits), no `rand`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f64) / f64::from(1u32 << 31)
    }

    fn between(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// A dense scene: `n` bodies in a 6×6×6 box, mostly spheres of radius 0.5, every
/// fifth a shaped box, with random velocities and a little gravity.
fn scene(kind: Broadphase, n: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -1.0, 0.0),
        substeps: 2,
        ..SolverConfig::default()
    });
    w.set_broadphase(kind);
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    for i in 0..n {
        let p = v3(
            rng.between(-3.0, 3.0),
            rng.between(-3.0, 3.0),
            rng.between(-3.0, 3.0),
        );
        let idx = if i % 5 == 4 {
            w.add_shaped_body(
                &Shape::Box {
                    half_extents: v3(0.4, 0.3, 0.5),
                },
                Fix128::from_int(1000),
                p,
            )
            .expect("a valid solid")
        } else {
            w.add_body_with_radius(RigidBody::new(p, Fix128::ONE), fx(0.5))
        };
        w.get_body_mut(idx).expect("body").velocity = v3(
            rng.between(-2.0, 2.0),
            rng.between(-2.0, 2.0),
            rng.between(-2.0, 2.0),
        );
    }
    w
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn assert_same(a: &PhysicsWorld, b: &PhysicsWorld, what: &str) {
    assert_eq!(a.bodies.len(), b.bodies.len(), "{what}: body count");
    for (i, (x, y)) in a.bodies.iter().zip(&b.bodies).enumerate() {
        assert_eq!(x.position, y.position, "{what}: body {i} position");
        assert_eq!(x.velocity, y.velocity, "{what}: body {i} velocity");
        assert_eq!(x.rotation, y.rotation, "{what}: body {i} rotation");
        assert_eq!(
            x.angular_velocity, y.angular_velocity,
            "{what}: body {i} angular velocity"
        );
    }
}

/// The two broad-phases step a dense scene to bit-identical states.
#[test]
fn the_two_broadphases_step_a_dense_scene_identically() {
    let mut bvh = scene(Broadphase::Bvh, 60);
    let mut tree = scene(Broadphase::DynamicTree, 60);
    assert_same(&bvh, &tree, "start");
    let mut touched = false;
    for step in 0..60 {
        bvh.step(dt());
        tree.step(dt());
        assert_same(&bvh, &tree, &format!("step {step}"));
        touched |= !bvh.contact_constraints.is_empty();
    }
    assert!(touched, "the scene is dense enough that bodies collide");
}

/// Adding bodies, removing a body (the last one takes its index), and changing a
/// collision radius mid-run keep the two worlds identical.
#[test]
fn the_two_broadphases_agree_across_structural_changes() {
    let mut bvh = scene(Broadphase::Bvh, 40);
    let mut tree = scene(Broadphase::DynamicTree, 40);
    for w in [&mut bvh, &mut tree] {
        for _ in 0..10 {
            w.step(dt());
        }
    }
    assert_same(&bvh, &tree, "after 10 steps");
    for (w, name) in [(&mut bvh, "bvh"), (&mut tree, "tree")] {
        let _ = name;
        // A new sphere in the middle of the crowd.
        w.add_body_with_radius(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE), fx(0.7));
        // Remove a body from the middle: the last one is renumbered into its slot.
        w.remove_body(7).expect("body 7 exists");
        // A body loses its collision radius, another changes it.
        w.clear_body_collision_radius(3);
        w.set_body_collision_radius(11, fx(0.9));
    }
    for step in 0..25 {
        bvh.step(dt());
        tree.step(dt());
        assert_same(&bvh, &tree, &format!("after the changes, step {step}"));
    }
}

/// Switching the broad-phase mid-run changes nothing.
#[test]
fn switching_the_broadphase_mid_run_changes_nothing() {
    let mut steady = scene(Broadphase::Bvh, 40);
    let mut switching = scene(Broadphase::Bvh, 40);
    for step in 0..30 {
        if step == 10 {
            switching.set_broadphase(Broadphase::DynamicTree);
        }
        if step == 20 {
            switching.set_broadphase(Broadphase::Bvh);
        }
        steady.step(dt());
        switching.step(dt());
        assert_same(&steady, &switching, &format!("step {step}"));
    }
}

/// The default broad-phase is the BVH, and it keeps no tree.
#[test]
fn the_default_is_the_bvh_and_keeps_no_tree() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    assert_eq!(w.broadphase(), Broadphase::Bvh);
    for i in 0..4 {
        w.add_body_with_radius(
            RigidBody::new(v3(f64::from(i) * 3.0, 0.0, 0.0), Fix128::ONE),
            fx(0.5),
        );
    }
    w.step(dt());
    assert_eq!(w.broadphase_stats().proxies, 0);
    assert_eq!(w.broadphase_stats().height, 0);
    assert!(w.broadphase_proxy_aabb(0).is_none());
}

/// The tree holds one proxy per body with a radius; a body without one is absent,
/// and a body that loses its radius leaves the tree.
#[test]
fn the_tree_holds_one_proxy_per_body_with_a_radius() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.set_broadphase(Broadphase::DynamicTree);
    for i in 0..6 {
        w.add_body_with_radius(
            RigidBody::new(v3(f64::from(i) * 4.0, 0.0, 0.0), Fix128::ONE),
            fx(0.5),
        );
    }
    let bare = w.add_body(RigidBody::new(v3(0.0, 9.0, 0.0), Fix128::ONE));
    w.step(dt());
    let stats = w.broadphase_stats();
    assert_eq!(stats.proxies, 6, "six bodies have a radius, one does not");
    assert!(
        stats.height >= 2 && stats.height <= 4,
        "six leaves make a tree 2 to 4 levels tall: {}",
        stats.height
    );
    assert!(w.broadphase_proxy_aabb(bare).is_none());
    w.clear_body_collision_radius(2);
    w.step(dt());
    assert_eq!(w.broadphase_stats().proxies, 5, "body 2 left the tree");
    assert!(w.broadphase_proxy_aabb(2).is_none());
    w.set_broadphase(Broadphase::Bvh);
    assert_eq!(w.broadphase_stats().proxies, 0, "switching drops the tree");
}

/// A fresh proxy's box is the sphere's box fattened by exactly 0.5; a body that
/// moves less keeps that box, a body that moves more gets a new one around its new
/// position.
#[test]
fn a_proxy_is_fattened_by_the_margin_and_follows_a_body_that_leaves_it() {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    w.set_broadphase(Broadphase::DynamicTree);
    let a = w.add_body_with_radius(RigidBody::new(v3(0.0, 0.0, 0.0), Fix128::ONE), fx(0.5));
    // A second body far away so the pair search has something to do.
    w.add_body_with_radius(RigidBody::new(v3(20.0, 0.0, 0.0), Fix128::ONE), fx(0.5));
    w.step(dt());
    let first = w.broadphase_proxy_aabb(a).expect("a proxy");
    let close = |got: Vec3Fix, want: [f64; 3]| {
        let g = [got.x.to_f64(), got.y.to_f64(), got.z.to_f64()];
        (0..3).all(|k| (g[k] - want[k]).abs() < 1e-12)
    };
    // radius 0.5 + margin 0.5 = 1.0 each way.
    assert!(close(first.min, [-1.0, -1.0, -1.0]), "fat min");
    assert!(close(first.max, [1.0, 1.0, 1.0]), "fat max");

    // Move 0.3 in x: the tight box [-0.2, 0.8] is inside the fat box: unchanged.
    w.get_body_mut(a).expect("body").position = v3(0.3, 0.0, 0.0);
    w.step(dt());
    assert_eq!(w.broadphase_proxy_aabb(a), Some(first), "inside: kept");

    // Move to x = 2: the tight box [1.5, 2.5] is outside: a new fat box [1, 3].
    w.get_body_mut(a).expect("body").position = v3(2.0, 0.0, 0.0);
    w.step(dt());
    let moved = w.broadphase_proxy_aabb(a).expect("a proxy");
    assert!(close(moved.min, [1.0, -1.0, -1.0]), "re-inserted min");
    assert!(close(moved.max, [3.0, 1.0, 1.0]), "re-inserted max");
}

/// Restoring a saved state under the tree broad-phase: the tree is rebuilt from the
/// restored bodies, and the run continues exactly as one that never stopped.
#[test]
fn a_restored_state_continues_identically_under_the_tree() {
    let mut reference = scene(Broadphase::Bvh, 40);
    let mut tree = scene(Broadphase::DynamicTree, 40);
    for _ in 0..15 {
        reference.step(dt());
        tree.step(dt());
    }
    let saved = tree.serialize_state();
    // Wander off, then restore: the proxies of the wandered state are stale.
    for _ in 0..5 {
        tree.step(dt());
    }
    assert!(tree.deserialize_state(&saved), "the state loads");
    assert_same(&reference, &tree, "restored");
    for step in 0..20 {
        reference.step(dt());
        tree.step(dt());
        assert_same(&reference, &tree, &format!("after restore, step {step}"));
    }
}

/// Removing bodies shrinks the tree to match: the proxies of the indices that no
/// longer exist are dropped (and only those), and the run stays identical to the
/// BVH's.
#[test]
fn removing_bodies_drops_exactly_their_proxies() {
    let mut bvh = scene(Broadphase::Bvh, 30);
    let mut tree = scene(Broadphase::DynamicTree, 30);
    for w in [&mut bvh, &mut tree] {
        w.step(dt());
    }
    assert_eq!(tree.broadphase_stats().proxies, 30);
    for w in [&mut bvh, &mut tree] {
        w.remove_body(5).expect("body 5");
        w.remove_body(10).expect("body 10");
        w.remove_body(0).expect("body 0");
    }
    for step in 0..10 {
        bvh.step(dt());
        tree.step(dt());
        assert_same(&bvh, &tree, &format!("after removals, step {step}"));
    }
    assert_eq!(tree.broadphase_stats().proxies, 27, "27 bodies remain");
    // And growing again adds exactly the new bodies' proxies.
    tree.add_body_with_radius(RigidBody::new(v3(9.0, 9.0, 9.0), Fix128::ONE), fx(0.5));
    tree.step(dt());
    assert_eq!(tree.broadphase_stats().proxies, 28);
}
