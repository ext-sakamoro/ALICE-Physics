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
//! # Known difference: sleep state resets across `remove_body`
//!
//! `remove_body` rebuilds the whole island manager from scratch
//! (`self.islands = IslandManager::new(new_len, self.islands.config)`,
//! `src/solver.rs::remove_body`) rather than only touching the removed
//! body's entry, so *every* body's sleep state resets to awake at that
//! point, not just bodies connected to the one removed. A world that had
//! any sleeping body therefore disagrees with a same-state-reached-a-
//! different-way world immediately after a `remove_body` call, and
//! reconverges once asleep bodies fall back asleep over the next few
//! `step`s. `tail_add_then_remove_before_any_body_sleeps_matches_never_added`
//! avoids this by removing before anything has had time to sleep;
//! `distant_static_body_add_and_remove_converges_after_settling` measures
//! the gap directly instead of assuming it.

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::SleepConfig;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
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

/// The known sleep-state gap (see the module doc comment), measured
/// directly rather than assumed: adding a distant static body, letting the
/// scene settle, then removing it again disagrees with a world that never
/// had it right at the step in which `remove_body` resets every body's
/// sleep state, and converges again over the following steps as bodies
/// that were asleep settle back asleep. Run twice, adding the transient
/// body at two different points (step 3 and step 20), to measure that it
/// is the removal itself that causes the gap, not how long the body had
/// been present.
#[test]
fn distant_static_body_add_and_remove_converges_after_settling() {
    for add_at_step in [3usize, 20usize] {
        let settle_steps = 15; // long enough, with frames_to_sleep: 3, for resting bodies to sleep
        let steps_so_far = add_at_step + settle_steps;
        let after_steps = 5;
        let total_steps = steps_so_far + after_steps;

        // A separate world, stepped only to `steps_so_far`, exclusively for
        // the "immediately after remove_body" comparison below — the main
        // `baseline` below is kept at exactly `total_steps` for the final
        // comparison, never stepped past it.
        let mut baseline_at_removal_point = base_scene(Broadphase::default());
        for _ in 0..steps_so_far {
            baseline_at_removal_point.step(dt());
        }

        let mut baseline = base_scene(Broadphase::default());
        for _ in 0..total_steps {
            baseline.step(dt());
        }

        let mut with_transient = base_scene(Broadphase::default());
        for _ in 0..add_at_step {
            with_transient.step(dt());
        }
        let transient =
            with_transient.add_body(RigidBody::new_static(Vec3Fix::from_int(1000, 1000, 1000)));
        for _ in 0..settle_steps {
            with_transient.step(dt());
        }
        with_transient.remove_body(transient); // tail: nothing else added since

        assert_ne!(
            with_transient.snapshot_world(),
            baseline_at_removal_point.snapshot_world(),
            "add_at_step={add_at_step}: immediately after remove_body, sleep \
             state has just been reset for every body, so this is expected \
             to disagree with the baseline for exactly this one step"
        );

        for _ in 0..after_steps {
            with_transient.step(dt());
        }

        assert_eq!(
            with_transient.snapshot_world(),
            baseline.snapshot_world(),
            "add_at_step={add_at_step}: a few steps after remove_body, bodies \
             that were asleep before should be asleep again, converging \
             back to the baseline"
        );
    }
}
