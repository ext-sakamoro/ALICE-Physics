//! `snapshot_world` writes the persistent `Broadphase::DynamicTree` tree as
//! its leaf set (per body index: whether the body has a proxy, and the
//! stored fattened box), and restore rebuilds the tree from those leaves.
//!
//! oracle: a world that never held the removed body (the "clean" world),
//! stepped live from the same start. Adding a body with a collision radius,
//! stepping once and removing it leaves a freed node in the live tree; its
//! state (bodies, leaves, pairs) is the clean world's, so its snapshot bytes
//! and every later step must be the clean world's too. Body positions and
//! velocities are compared field by field, independently of the encoder.

use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::{Fix128, SleepConfig, Vec3Fix};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// A static floor (no radius) and four dynamic bodies of radius 1/2 close
/// enough that their boxes overlap, so the tree reports pairs.
fn probe_scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(Broadphase::DynamicTree);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
    w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)));
    let half = Fix128::from_ratio(1, 2);
    for (x, z) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(x, 2, z), Fix128::ONE),
            half,
        );
    }
    w
}

/// `probe_scene` after a body of radius 1/4 far away was added, stepped once
/// (so it entered the tree) and removed again: the dirty world.
fn dirty() -> PhysicsWorld {
    let mut w = probe_scene();
    let extra = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(50, 50, 50), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.step(dt());
    assert_eq!(
        w.broadphase_stats().proxies,
        5,
        "control: the body entered the tree"
    );
    w.remove_body(extra);
    w
}

/// `probe_scene` stepped once without the extra body: the clean world.
fn clean() -> PhysicsWorld {
    let mut w = probe_scene();
    w.step(dt());
    w
}

/// Position, velocity, rotation and angular velocity of every body.
fn bodies(w: &PhysicsWorld) -> String {
    let v: Vec<_> = w
        .bodies
        .iter()
        .map(|b| {
            format!(
                "{:?} {:?} {:?} {:?}",
                b.position, b.velocity, b.rotation, b.angular_velocity
            )
        })
        .collect();
    format!("{v:?}")
}

fn assert_same(a: &PhysicsWorld, b: &PhysicsWorld, ctx: &str) {
    assert_eq!(bodies(a), bodies(b), "bodies: {ctx}");
    assert_eq!(
        a.snapshot_world(),
        b.snapshot_world(),
        "snapshot bytes: {ctx}"
    );
}

/// A second body with a radius, added to both worlds midway.
fn add_late_body(w: &mut PhysicsWorld) {
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
        Fix128::from_ratio(1, 2),
    );
}

/// The dirty and the clean world write the same bytes 1, 5, 30 and 200
/// steps after the removal, and their bodies agree throughout.
#[test]
fn dirty_and_clean_snapshots_match_after_removal() {
    let (mut d, mut c) = (dirty(), clean());
    assert!(
        d.broadphase_stats().proxies == 4 || d.broadphase_stats().proxies == 5,
        "control: the proxy goes on the next step"
    );
    let mut stepped = 0;
    for at in [1, 5, 30, 200] {
        while stepped < at {
            d.step(dt());
            c.step(dt());
            stepped += 1;
        }
        assert_eq!(d.broadphase_stats().proxies, 4, "+{at}");
        assert_same(&d, &c, &format!("+{at}"));
    }
}

/// snapshot → restore → snapshot gives the same bytes, for the live dirty
/// and clean worlds and again for the restored worlds.
#[test]
fn snapshot_restore_snapshot_is_a_fixed_point() {
    for (name, mut w) in [("dirty", dirty()), ("clean", clean())] {
        w.step(dt());
        let blob = w.snapshot_world();
        let r = PhysicsWorld::from_world_snapshot(&blob).expect("restore");
        assert_eq!(r.snapshot_world(), blob, "{name}: restored");
        let rr = PhysicsWorld::from_world_snapshot(&r.snapshot_world()).expect("again");
        assert_eq!(rr.snapshot_world(), blob, "{name}: restored twice");
        assert_eq!(bodies(&rr), bodies(&w), "{name}");
        // public views of the tree: the proxy count and every body's stored
        // box survive the rebuild (the height may not: the layout is rebuilt)
        assert_eq!(rr.broadphase_stats().proxies, w.broadphase_stats().proxies);
        for i in 0..w.bodies.len() + 1 {
            assert_eq!(
                rr.broadphase_proxy_aabb(i),
                w.broadphase_proxy_aabb(i),
                "{name}: body {i}"
            );
        }
        assert!(w.broadphase_proxy_aabb(1).is_some(), "control: {name}");
    }
}

/// From the dirty and from the clean blob, a restored world stepped 50 times
/// is the live clean world stepped 50 times, bodies and bytes.
#[test]
fn restore_then_50_steps_matches_the_live_clean_world() {
    let (mut d, mut c) = (dirty(), clean());
    d.step(dt());
    c.step(dt());
    let blobs = [d.snapshot_world(), c.snapshot_world()];
    assert_eq!(blobs[0], blobs[1]);
    let mut restored: Vec<PhysicsWorld> = blobs
        .iter()
        .map(|b| PhysicsWorld::from_world_snapshot(b).expect("restore"))
        .collect();
    c.step_n(50, dt());
    for (i, r) in restored.iter_mut().enumerate() {
        r.step_n(50, dt());
        assert_same(r, &c, ["from dirty", "from clean"][i]);
    }
}

/// The live dirty world and a world restored from its blob, stepped 200
/// more with a radius body added to both after 100, stay identical at every
/// step, and match the live clean world doing the same.
#[test]
fn restored_dirty_world_continues_like_the_live_one() {
    let (mut d, mut c) = (dirty(), clean());
    let mut r = PhysicsWorld::from_world_snapshot(&d.snapshot_world()).expect("restore");
    for i in 0..200 {
        if i == 100 {
            add_late_body(&mut d);
            add_late_body(&mut r);
            add_late_body(&mut c);
        }
        d.step(dt());
        r.step(dt());
        c.step(dt());
        assert_same(&d, &r, &format!("live vs restored, step {i}"));
        assert_same(&d, &c, &format!("dirty vs clean, step {i}"));
    }
    assert_eq!(d.broadphase_stats().proxies, 5);
}

/// A version 3 blob of a dirty `DynamicTree` world (written node by node, a
/// freed node and the free list included, by the version 3 writer) restores
/// to the clean world: same bytes, same bodies, same later steps.
#[test]
fn version_3_blob_with_a_freed_node_restores_to_the_clean_world() {
    let v3: &[u8] = include_bytes!("fixtures/world_snapshot_v3_dynamic_tree_freed.bin");
    assert_eq!(&v3[4..6], &3u16.to_le_bytes());
    // the scene the fixture was written from: `base_scene(DynamicTree)` of
    // `world_snapshot_history_independence.rs`, a body of radius 1/4 added,
    // one step, removed, 30 steps; the clean world skips the extra body
    let mut c = PhysicsWorld::new(SolverConfig::default());
    c.set_broadphase(Broadphase::DynamicTree);
    c.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
    let half = Fix128::from_ratio(1, 2);
    c.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)));
    let a = c.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 1, 0), Fix128::ONE),
        half,
    );
    let b = c.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(-1, 1, 0), Fix128::ONE),
        half,
    );
    c.add_joint(alice_physics::joint::Joint::Ball(
        alice_physics::joint::BallJoint::new(a, b, Vec3Fix::ZERO, Vec3Fix::ZERO),
    ));
    c.step_n(31, dt());

    let mut r = PhysicsWorld::from_world_snapshot(v3).expect("v3");
    assert_same(&r, &c, "restored");
    for i in 0..50 {
        r.step(dt());
        c.step(dt());
        assert_same(&r, &c, &format!("step {i}"));
    }
}

// ── `broadphase_stats().height` ───────────────────────────────────────────
//
// oracle: the world restored from the live world's snapshot. Its tree is
// rebuilt from the leaf set, so it holds the tree the height is defined on;
// the incremental live tree has a different shape whenever leaves were
// inserted out of body index order, removed or re-inserted after moving.

/// After every step, the live world reports the same stats as the world
/// restored from its snapshot, and the height is at least the height of a
/// balanced tree of that many leaves.
fn heights_agree(
    w: &mut PhysicsWorld,
    steps: usize,
    mut each: impl FnMut(&mut PhysicsWorld, usize),
) {
    let mut with_tree = 0;
    for i in 0..steps {
        each(w, i);
        w.step(dt());
        let r = PhysicsWorld::from_world_snapshot(&w.snapshot_world()).expect("restore");
        let (a, b) = (w.broadphase_stats(), r.broadphase_stats());
        assert_eq!(a, b, "step {i}");
        if a.proxies >= 2 {
            with_tree += 1;
            // a tree of n leaves is at least ceil(log2 n) tall
            let floor = usize::BITS - (a.proxies - 1).leading_zeros();
            assert!(a.height as u32 >= floor, "step {i}: {a:?}");
        }
    }
    assert!(
        with_tree >= steps - 1,
        "control: {with_tree} of {steps} steps hold a tree"
    );
}

/// Bodies gaining a radius out of index order (the inserts land in a
/// shuffled order) and bodies removed from the middle: the live tree's height
/// differs from the rebuilt one's on 23 of these 200 steps.
#[test]
fn height_is_the_same_after_restore_with_shuffled_inserts_and_removes() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(Broadphase::DynamicTree);
    heights_agree(&mut w, 200, |w, i| {
        let x = (i as i64 * 37) % 200;
        w.add_body_with_radius(
            RigidBody::new_static(Vec3Fix::from_int(3 * x, 0, 0)),
            Fix128::from_ratio(1, 2),
        );
        if i % 7 == 3 {
            let n = w.bodies.len();
            w.remove_body(n / 2);
        }
    });
}

/// 1,000 bodies moving at different velocities without gravity: their
/// proxies are re-inserted as they leave their fattened boxes, and the live
/// tree grows taller than the rebuilt one.
#[test]
fn height_is_the_same_after_restore_in_a_moving_scene() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(Broadphase::DynamicTree);
    w.config.gravity = Vec3Fix::ZERO;
    for i in 0..1000i64 {
        let (x, y, z) = (i % 10, (i / 10) % 10, i / 100);
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(4 * x, 4 * y, 4 * z), Fix128::ONE);
        b.velocity = Vec3Fix::new(
            Fix128::from_ratio((i * 7919) % 97 + 30, 30),
            Fix128::from_ratio((i * 104_729) % 89 - 44, 40),
            Fix128::ZERO,
        );
        w.add_body_with_radius(b, Fix128::from_ratio(1, 2));
    }
    heights_agree(&mut w, 40, |_, _| {});
}

/// No proxy, no height: before any step, and with `Bvh`.
#[test]
fn height_is_zero_without_proxies() {
    let mut w = probe_scene();
    assert_eq!(w.broadphase_stats().height, 0);
    w.set_broadphase(Broadphase::Bvh);
    w.step(dt());
    assert_eq!(w.broadphase_stats().height, 0);
    assert_eq!(w.broadphase_stats().proxies, 0);
}
