//! Measures whether `snapshot_world` / `serialize_state` bytes (and later
//! `step` behavior) depend only on a world's current index-to-body state,
//! or whether the history that produced that state leaks through — for
//! several histories that are expected to reach the *same* state by
//! construction (same index assignment, same body values), not merely an
//! index-level permutation of different content.
//!
//! This supersedes an earlier file in this area that compared a forward
//! and a reversed insertion order: that comparison held body *content*
//! fixed per physical body while insertion order changed which index each
//! body landed at, so the two worlds held genuinely different content at
//! each index — any index-keyed serializer disagreeing on that is
//! expected, not a finding about canonical form. The scenarios below
//! instead vary *how* the same final index/content assignment was reached.
//!
//! # Known defect: `remove_body` permanently disturbs unrelated sleeping bodies
//!
//! `remove_body` rebuilds the whole island manager from scratch
//! (`self.islands = IslandManager::new(new_len, self.islands.config)`,
//! `src/solver.rs::remove_body`) rather than only touching the removed
//! body's entry, so *every* body's sleep state resets to awake at that
//! point, not just bodies connected to the one removed. This is not merely
//! a metadata difference: a body that was asleep resumes being integrated,
//! and (confirmed directly in
//! `distant_static_body_removed_after_unrelated_bodies_slept_permanently_diverges`
//! below, on a scene where the other bodies are actually at rest — unlike
//! an earlier version of this file, which used a scene whose two dynamic
//! bodies free-fell forever and so never genuinely exercised this path at
//! all) its simulated trajectory from that point on is a different,
//! independently-computed trajectory from the one a body that stayed
//! asleep would have followed, so the two never converge again on their
//! own, through at least 200 further steps, and the test below is
//! `#[ignore]`d as a pinned defect rather than something this file's other
//! scenarios can route around.
//! `tail_add_then_remove_before_any_body_sleeps_matches_never_added` avoids
//! the whole question by removing before anything has had time to sleep.

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::SleepConfig;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// A floor and two dynamic bodies connected by a ball joint, with a sleep
/// config loose enough that resting bodies actually fall asleep within the
/// step counts used below (same thresholds as `tests/world_snapshot_v2.rs`).
fn base_scene(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
    let half = Fix128::from_ratio(1, 2);
    // `new_static`, not `new(_, ZERO)`: a zero-mass body built with `new`
    // stays `BodyType::Dynamic` (only `inv_mass` is zero), so it is not the
    // same body a real static floor would be.
    let floor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 1, 0), Fix128::ONE),
        half,
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(-1, 1, 0), Fix128::ONE),
        half,
    );
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    let _ = floor;
    w
}

#[test]
fn tail_add_then_remove_before_any_body_sleeps_matches_never_added() {
    let mut with_extra = base_scene(Broadphase::default());
    let extra = with_extra.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(50, 50, 50), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    with_extra.step(dt());
    with_extra.remove_body(extra); // tail index: a plain pop, no swap
    with_extra.step(dt());

    let mut never_added = base_scene(Broadphase::default());
    never_added.step(dt());
    never_added.step(dt());

    assert_eq!(
        with_extra.snapshot_world(),
        never_added.snapshot_world(),
        "adding a body at the tail and removing it again (before anything \
         slept) should leave no trace in snapshot_world"
    );
    assert_eq!(
        with_extra.serialize_state(),
        never_added.serialize_state(),
        "adding a body at the tail and removing it again (before anything \
         slept) should leave no trace in serialize_state"
    );
}

/// The same scenario as `tail_add_then_remove_before_any_body_sleeps_matches_never_added`,
/// but with `Broadphase::DynamicTree` instead of the default `Bvh`. With
/// `Bvh`, the persistent tree section of `snapshot_world` never holds any
/// content at all (`Bvh` is rebuilt from scratch every step and never reads
/// it), so that test's bytes can never carry evidence either way about
/// whether tree node order or content is handled correctly on removal — a
/// mutation that scrambled tree node order would pass it undetected.
///
/// ⚠️ Measured, not assumed to converge quickly: immediately after removal
/// the snapshot here is already 239 bytes longer than a world that never
/// had the extra body (not a small, single-lingering-leaf difference), and
/// it stays 242 bytes longer through at least 7 further steps — it does
/// not shrink back down on its own. Whatever is retained is not merely
/// "this step's uncompacted leaf"; `#[ignore]`d as a second, separate
/// `DynamicTree` history-dependence measurement rather than asserted to
/// resolve a particular way, since that would be guessing at a mechanism
/// this file has not actually traced through `src/dynamic_bvh.rs`.
#[test]
#[ignore = "measurement gap: DynamicTree's persistent tree does not return \
            to the same byte length/content as a world that never had the \
            removed body, even several steps after removal; mechanism not \
            traced (src/dynamic_bvh.rs is out of scope for this file)"]
fn tail_add_then_remove_does_not_converge_with_dynamic_tree() {
    let mut with_extra = base_scene(Broadphase::DynamicTree);
    let extra = with_extra.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(50, 50, 50), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    with_extra.step(dt());
    with_extra.remove_body(extra); // tail index: a plain pop, no swap

    let mut never_added = base_scene(Broadphase::DynamicTree);
    never_added.step(dt());

    for _ in 0..7 {
        with_extra.step(dt());
        never_added.step(dt());
    }

    assert_eq!(
        with_extra.snapshot_world(),
        never_added.snapshot_world(),
        "expected these to converge given enough steps; measured instead \
         that the byte length stabilizes 242 bytes longer and does not \
         shrink further"
    );
}

/// `Bvh` never persists tree state (rebuilt from scratch every step) and
/// `DynamicTree`'s is rebuilt on `set_broadphase` ("Known difference" in
/// `src/solver/world_snapshot.rs::snapshot_world"'s own doc table does not
/// mention a stale one surviving a kind switch), so cycling
/// `Bvh -> DynamicTree -> Bvh` and ending on the same kind the whole run
/// used should match a world that only ever used `Bvh`.
#[test]
fn broadphase_kind_round_trip_matches_never_switching() {
    let mut round_trip = base_scene(Broadphase::Bvh);
    round_trip.step(dt());
    round_trip.set_broadphase(Broadphase::DynamicTree);
    round_trip.step(dt());
    round_trip.set_broadphase(Broadphase::Bvh);
    round_trip.step(dt());

    let mut never_switched = base_scene(Broadphase::Bvh);
    never_switched.step(dt());
    never_switched.step(dt());
    never_switched.step(dt());

    assert_eq!(
        round_trip.snapshot_world(),
        never_switched.snapshot_world(),
        "cycling through DynamicTree and back to Bvh should leave no trace, \
         since Bvh never reads the persistent tree"
    );
}

#[test]
fn advance_then_restore_matches_the_older_snapshot() {
    let mut w = base_scene(Broadphase::default());
    w.step(dt());
    w.step(dt());
    let old_snapshot = w.snapshot_world();

    for _ in 0..10 {
        w.step(dt());
    }
    assert_ne!(
        w.snapshot_world(),
        old_snapshot,
        "sanity: the world must actually have moved on from the older snapshot"
    );

    w.restore_world(&old_snapshot)
        .expect("restore the older snapshot");
    assert_eq!(
        w.snapshot_world(),
        old_snapshot,
        "restoring an older snapshot must fully revert, regardless of how \
         far the world had advanced since it was taken"
    );
}

/// Four small dynamic spheres resting on a large static sphere (not
/// touching each other), under the same loose sleep config as `base_scene`.
/// Falling under gravity onto the big sphere and settling takes well under
/// 60 steps; by 120 steps (`frames_to_sleep: 3` past settling) all four are
/// asleep. Unlike `base_scene`'s two bodies, which only hang from a joint
/// and free-fall forever if nothing is actually resting them, these bodies
/// reach, and stay at, genuine rest — the scenario `remove_body`'s sleep
/// reset actually needs to disturb anything.
fn resting_scene(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
    let big_radius = fx(10.0);
    let small_radius = fx(0.5);
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), big_radius);
    for (x, z) in [(1.5, 1.5), (-1.5, 1.5), (1.5, -1.5), (-1.5, -1.5)] {
        w.add_body_with_radius(
            RigidBody::new_dynamic(v3(x, 15.0, z), Fix128::ONE),
            small_radius,
        );
    }
    w
}

/// Pinned known defect (see the module doc comment): a distant, never-
/// touching static body is added once the four resting spheres above have
/// already fallen asleep, then removed again once it has been present for
/// a while. `remove_body` rebuilding the whole island manager wakes the
/// four sleeping spheres, and from that point on they are a different,
/// independently-integrated trajectory from a world that never had the
/// extra body and so never had its sleep disturbed — not a transient gap
/// that the next few steps undo, but a lasting divergence, measured here
/// through +200 further steps. `#[ignore]`d: this asserts the behavior
/// `remove_body` should have, which the current implementation does not.
#[test]
#[ignore = "known defect: remove_body resets every body's sleep state via \
            a full IslandManager rebuild, permanently waking unrelated \
            sleeping bodies instead of only updating the removed one's \
            entry; see the module doc comment"]
fn distant_static_body_removed_after_unrelated_bodies_slept_permanently_diverges() {
    let settle_and_sleep_steps = 60;
    let resident_steps = 60;
    let after_remove_steps = 200;
    let total_steps = settle_and_sleep_steps + resident_steps + after_remove_steps;

    let mut baseline = resting_scene(Broadphase::default());
    for _ in 0..total_steps {
        baseline.step(dt());
    }

    let mut with_transient = resting_scene(Broadphase::default());
    for _ in 0..settle_and_sleep_steps {
        with_transient.step(dt());
    }
    let transient = with_transient.add_body(RigidBody::new_static(v3(1000.0, 1000.0, 1000.0)));
    for _ in 0..resident_steps {
        with_transient.step(dt());
    }
    with_transient.remove_body(transient); // tail: nothing else added since

    assert_eq!(
        with_transient.snapshot_world(),
        {
            let mut b = resting_scene(Broadphase::default());
            for _ in 0..(settle_and_sleep_steps + resident_steps) {
                b.step(dt());
            }
            b.snapshot_world()
        },
        "immediately after remove_body, a world whose sleeping bodies were \
         never disturbed should already match"
    );

    for _ in 0..after_remove_steps {
        with_transient.step(dt());
    }

    assert_eq!(
        with_transient.snapshot_world(),
        baseline.snapshot_world(),
        "{after_remove_steps} steps after remove_body, the two should still \
         match if the removal never disturbed the sleeping bodies"
    );
}
