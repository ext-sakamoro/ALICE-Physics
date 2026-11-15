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

use alice_physics_core::joint::{BallJoint, Joint};
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::{SleepConfig, SleepState};

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
        time_to_sleep: Fix128::from_ratio(3, 60),
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

    // The snapshot carries the bodies' stable ids when they are not `0..n` with `n` next
    // (flag bit 3 of the contact section): the tail add and removal used one id, so the ids
    // are 0, 1, 2 with 4 next and that section is the only trace. Laid out from the end: the
    // checksum (8), the section's length (8), the id count (8), the ids (8 each), the next id
    // (8), and before them the flags byte.
    let (a, b) = (with_extra.snapshot_world(), never_added.snapshot_world());
    let word = |s: &[u8], at: usize| u64::from_le_bytes(s[at..at + 8].try_into().unwrap());
    let ids = 8 + 3 * 8 + 8;
    let end = a.len() - 16;
    let start = end - ids;
    assert_eq!(a.len(), b.len() + ids, "the id section's bytes");
    assert_ne!(a[start - 1] & 8, 0, "flag bit 3 set");
    assert_eq!(
        (
            word(&a, start),
            word(&a, start + 8),
            word(&a, start + 16),
            word(&a, start + 24)
        ),
        (3, 0, 1, 2),
        "three ids, 0..3"
    );
    assert_eq!(
        word(&a, start + 32),
        4,
        "the next id after the tail add and removal"
    );
    let mut stripped = [&a[..start], &a[end..a.len() - 8]].concat();
    stripped[start - 1] &= !8;
    let section = word(&stripped, start) - ids as u64;
    stripped[start..start + 8].copy_from_slice(&section.to_le_bytes());
    let payload = word(&stripped, 8) - ids as u64;
    stripped[8..16].copy_from_slice(&payload.to_le_bytes());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in &stripped {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0001_0000_01b3);
    }
    stripped.extend_from_slice(&hash.to_le_bytes());
    assert_eq!(
        stripped, b,
        "adding a body at the tail and removing it again (before anything \
         slept) should leave no trace in snapshot_world but the id section"
    );
    // The rollback state carries the next stable id as it is (a body removed between steps
    // leaves warm-start entries keyed by its id until the next step, so the id is never handed
    // out again): the tail add and removal used one id, and that word is the only trace.
    let (a, b) = (with_extra.serialize_state(), never_added.serialize_state());
    assert_eq!(a.len(), b.len(), "serialize_state lengths");
    // The next-id word follows the header (12 bytes), the motion (208 a body), the sleep
    // records (17), the overflow flag (1), the population fingerprint (8), the elastic islands'
    // record (8, no island), the cloth count and the next grab id (4 each), the previous poses
    // (112 a body) and the bodies' stable ids (8 a body); this scene has no spring and no cloth.
    let n = 3;
    let at = 12 + n * 208 + n * 17 + 1 + 8 + 8 + 4 + 4 + n * 112 + n * 8;
    let word = |s: &[u8]| u64::from_le_bytes(s[at..at + 8].try_into().unwrap());
    assert_eq!(
        (word(&a), word(&b)),
        (4, 3),
        "next stable id after the tail add and removal vs never added"
    );
    let masked = |s: &[u8]| [&s[..at], &s[at + 8..]].concat();
    assert_eq!(
        masked(&a),
        masked(&b),
        "serialize_state should differ in the next stable id word only"
    );

    // The id word changes no motion: the same body added to both worlds moves the same.
    for w in [&mut with_extra, &mut never_added] {
        w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE),
            Fix128::from_ratio(1, 4),
        );
        for _ in 0..30 {
            w.step(dt());
        }
    }
    let motion = |w: &PhysicsWorld| -> Vec<_> {
        w.bodies
            .iter()
            .map(|b| (b.position, b.rotation, b.velocity, b.angular_velocity))
            .collect()
    };
    assert_eq!(
        motion(&with_extra),
        motion(&never_added),
        "a body added after the tail add and removal should move as in a world that never had it"
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
/// it stays 242 bytes longer through at least 30 further steps — it does
/// not shrink back down on its own, even though the *bodies'* own positions
/// do agree (this is purely a serialized-tree-content difference).
///
/// Confirmed mechanism (read directly, not inferred):
/// `w_tree` (`src/solver/world_snapshot.rs`) writes `nodes.len()` and every
/// node in the backing array — including freed ones, which `free_node`
/// (`src/dynamic_bvh.rs`) never removes, only resets and pushes onto
/// `free_list` — plus `free_list` itself, unconditionally. Nothing shrinks
/// the array on removal. This is behaviorally inert, not a correctness bug
/// in query results: `find_pairs` (`src/dynamic_bvh.rs`) sorts and dedups
/// its output before returning it, so the tree's internal layout never
/// reaches a caller either way; the only place it shows up is these bytes.
///
/// ⚠️ This test does not yet guard against a future tree-node-order
/// mutation: the current implementation already writes every node
/// (including freed ones) and the free list as-is, so a mutation that
/// merely reordered that existing content would not be "newly caught" by
/// anything here — there is no canonical order yet to diverge from. It
/// would become a meaningful guard once a future fix canonicalizes the
/// tree section and this test is un-ignored.
///
/// `#[ignore]`d as a second, separate `DynamicTree` history-dependence
/// defect, distinct from the sleep-state one above (confirmed distinct:
/// with a radius-less extra body that never enters the tree at all, the
/// equivalent scenario differs by only 1 byte and converges after one
/// step).
#[test]
#[ignore = "known defect: DynamicTree's remove_body frees the persistent \
            tree node without shrinking its backing array (confirmed: \
            src/solver/world_snapshot.rs's w_tree writes every node \
            including freed ones, plus the free list, unconditionally; \
            src/dynamic_bvh.rs's free_node never removes a node, only \
            resets and frees it), so the snapshot stays permanently \
            longer than a world that never had the removed body, even \
            though body positions agree; not yet a guard against a tree- \
            node-order mutation until a fix canonicalizes this section"]
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

    for _ in 0..30 {
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
/// 60 steps; by 120 steps (`time_to_sleep: Fix128::from_ratio(3, 60)` past settling) all four are
/// asleep. Unlike `base_scene`'s two bodies, which only hang from a joint
/// and free-fall forever if nothing is actually resting them, these bodies
/// reach, and stay at, genuine rest — the scenario `remove_body`'s sleep
/// reset actually needs to disturb anything.
fn resting_scene(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 10),
        angular_threshold: Fix128::from_ratio(1, 10),
        time_to_sleep: Fix128::from_ratio(3, 60),
    });
    let small_radius = fx(0.5);
    // A static flat box whose top face is at y=0 (a static body, as the
    // removal tests need); the four dynamic spheres rest on it along the x
    // axis, far enough apart not to touch each other. (On a big static
    // sphere they sit off its apex and roll off, so they never sleep.)
    let floor = w.add_body(RigidBody::new_static(v3(0.0, -1.0, 0.0)));
    assert!(w.set_body_shape(
        floor,
        &alice_physics_core::shape::Shape::Box {
            half_extents: v3(10.0, 1.0, 10.0),
        },
    ));
    for i in 0..4 {
        let x = f64::from(i) * 2.0 - 3.0; // -3, -1, 1, 3
        w.add_body_with_radius(
            RigidBody::new_dynamic(v3(x, 0.5, 0.0), Fix128::ONE),
            small_radius,
        );
    }
    w
}

/// Whether every body in `w` (including static ones, which start already
/// below the sleep threshold) is currently `SleepState::Sleeping` — the
/// premise `distant_static_body_removed_after_unrelated_bodies_slept_permanently_diverges`
/// depends on, checked explicitly rather than assumed (an earlier version
/// of `resting_scene` placed the four spheres asymmetrically near the top
/// of the big sphere, which never stops sliding down its curve and so
/// never reaches the sleep threshold at all — this measures the premise
/// instead of repeating that mistake silently).
fn all_bodies_sleeping(w: &PhysicsWorld) -> bool {
    w.islands
        .sleep_data
        .iter()
        .all(|sd| sd.state == SleepState::Sleeping)
}

/// The premise `distant_static_body_removed_after_unrelated_bodies_slept_permanently_diverges`
/// depends on, checked by a test that actually runs under a plain
/// `cargo test` — not only inside that `#[ignore]`d test, which does not
/// run by default and so would not catch a regression here on its own
/// (measured directly: setting `frames_to_sleep` to an unreachably large
/// value in `resting_scene` still leaves a default `cargo test` run green,
/// 3 passed / 2 ignored, with no indication the scene stopped sleeping at
/// all). An earlier version of `resting_scene` placed the four spheres
/// asymmetrically near the top of the big sphere, which never stops
/// sliding down its curve and so never reaches the sleep threshold either
/// — this is the test that would have caught that.
#[test]
fn resting_scene_reaches_sleep_by_step_120() {
    let mut w = resting_scene(Broadphase::default());
    for _ in 0..120 {
        w.step(dt());
    }
    assert!(
        all_bodies_sleeping(&w),
        "every body (the static floor trivially, and all four resting \
         spheres) should be asleep by step 120; sleep_data = {:?}",
        w.islands.sleep_data
    );
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
    // No collision radius: this is a probe for the sleep-reset defect only,
    // not for the separate DynamicTree persistent-tree-leaf measurement in
    // `tail_add_then_remove_does_not_converge_with_dynamic_tree` — giving it
    // a radius here would additionally perturb the broadphase tree and
    // conflate the two.
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
