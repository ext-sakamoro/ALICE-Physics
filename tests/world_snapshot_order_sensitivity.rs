//! Measures whether whole-world snapshot bytes depend on the order bodies
//! and joints were added to the world, as opposed to depending only on the
//! resulting state (COV-ENGINE-012).
//!
//! Motivation: a content hash of world state is only meaningful as "the same
//! state" if equal states always serialize to the same bytes regardless of
//! how they were built. This file measures that property for the two
//! serializers in scope, `PhysicsWorld::snapshot_world`
//! (src/solver/world_snapshot.rs) and `PhysicsWorld::serialize_state`
//! (src/solver.rs, the lighter rollback-netcode format); it does not change
//! either serializer.
//!
//! Both walk `self.bodies` / `self.joints` in `Vec` order, which is push
//! order (the index `PhysicsWorld::add_body` / `add_joint` hands back), not
//! a sort over any stable per-body id. `snapshot_world` additionally writes
//! the persistent broadphase tree "node by node (pair order follows the
//! tree layout)" per its own doc comment, and a dynamic AABB tree's layout
//! is itself a function of insertion order even when the final set of leaf
//! AABBs is identical. No `HashMap` sits on either path (checked by reading
//! both functions): the in-scope collections are `Vec`s in push order plus a
//! `tgs_impulse_cache` that is written sorted by id, so no accidental
//! hash-iteration-order source exists here; the order sensitivity measured
//! below comes from `Vec` push order and broadphase tree shape, not from an
//! unordered map.
//!
//! A `PhysicsWorld::Participant` concept exists (src/world_participant.rs,
//! registration order is part of its documented contract) but as of this
//! writing it is consumed only by `src/pipeline.rs`, never by
//! `PhysicsWorld` itself (`grep -c Participant src/solver.rs` is 0) — it has
//! no bearing on `snapshot_world` / `serialize_state` today, so this file
//! does not exercise it.

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

/// Three bodies: a static floor and two dynamic spheres, added to a fresh
/// world in the order given by `order` (a permutation of `[0, 1, 2]` naming
/// which of the three bodies to add first/second/third), plus a ball joint
/// between the two dynamic spheres. Returns the resulting world along with
/// the indices the joint actually used, so callers can see the indices
/// moved with the permutation.
fn build_world(order: [usize; 3]) -> (PhysicsWorld, [usize; 3]) {
    let bodies = [
        RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::ZERO), // static floor
        RigidBody::new(Vec3Fix::from_int(5, 10, 0), Fix128::ONE), // dynamic sphere A
        RigidBody::new(Vec3Fix::from_int(-3, 20, 7), Fix128::from_int(2)), // dynamic sphere B
    ];

    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(Broadphase::DynamicTree);

    let mut index_of = [0usize; 3];
    for &slot in &order {
        index_of[slot] = w.add_body(bodies[slot]);
    }
    w.add_joint(Joint::Ball(BallJoint::new(
        index_of[1],
        index_of[2],
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));

    (w, index_of)
}

/// Positive control: two worlds built via the *same* insertion order are
/// the same construction path twice, so this must hold regardless of
/// anything measured below — if it failed, `snapshot_world` would be
/// nondeterministic even for a single fixed path (allocator addresses,
/// uninitialized padding, etc.), which would make any order-sensitivity
/// finding below meaningless.
#[test]
fn same_order_twice_is_byte_identical() {
    let (w1, _) = build_world([0, 1, 2]);
    let (w2, _) = build_world([0, 1, 2]);
    assert_eq!(
        w1.snapshot_world(),
        w2.snapshot_world(),
        "snapshot_world must be deterministic for one fixed construction path"
    );
    assert_eq!(
        w1.serialize_state(),
        w2.serialize_state(),
        "serialize_state must be deterministic for one fixed construction path"
    );
}

/// The measurement this file exists for: building the same three bodies and
/// the same joint via a different insertion order. `#[ignore]`d because the
/// bytes currently differ (see the module doc comment for why) — this pins
/// the gap as a known, measured fact rather than asserting a property that
/// does not hold yet.
#[test]
#[ignore = "measurement gap: snapshot_world bytes are not canonical under reordering (body Vec push order + broadphase tree shape both vary with insertion order); see module doc comment"]
fn reordered_insertion_is_byte_identical_snapshot_world() {
    let (forward, _) = build_world([0, 1, 2]);
    let (reversed, _) = build_world([2, 1, 0]);
    assert_eq!(
        forward.snapshot_world(),
        reversed.snapshot_world(),
        "snapshot_world bytes should not depend on body/joint insertion order"
    );
}

/// Same measurement for the lighter rollback-netcode serializer
/// (`serialize_state`), which carries only body transforms/sleep state (no
/// joints), so only `Vec` push order of bodies is in play here.
#[test]
#[ignore = "measurement gap: serialize_state bytes are not canonical under reordering (body Vec push order varies with insertion order); see module doc comment"]
fn reordered_insertion_is_byte_identical_serialize_state() {
    let (forward, _) = build_world([0, 1, 2]);
    let (reversed, _) = build_world([2, 1, 0]);
    assert_eq!(
        forward.serialize_state(),
        reversed.serialize_state(),
        "serialize_state bytes should not depend on body insertion order"
    );
}

/// Confirms the two `#[ignore]`d assertions above actually have teeth
/// (would catch a real mismatch) rather than being vacuously true for some
/// unrelated reason (e.g. a bug in this test file that made both sides
/// identical no matter what). Run without `#[ignore]`: demonstrates, without
/// touching src, that the forward- and reversed-order snapshots are in fact
/// unequal today, which is the same comparison the two tests above make —
/// this one just asserts the finding instead of pinning it.
#[test]
fn reordered_insertion_is_measurably_different_today() {
    let (forward, forward_idx) = build_world([0, 1, 2]);
    let (reversed, reversed_idx) = build_world([2, 1, 0]);

    assert_ne!(
        forward_idx, reversed_idx,
        "the two construction paths must actually use different indices for \
         the joint's bodies, or this test is not exercising reordering at all"
    );
    assert_ne!(
        forward.snapshot_world(),
        reversed.snapshot_world(),
        "expected the two construction paths to currently disagree (that is \
         the measurement this file reports); if this now passes, the \
         #[ignore]d tests above should be un-ignored instead"
    );
    assert_ne!(
        forward.serialize_state(),
        reversed.serialize_state(),
        "expected the two construction paths to currently disagree (that is \
         the measurement this file reports); if this now passes, the \
         #[ignore]d tests above should be un-ignored instead"
    );
}

/// Sanity check that byte comparison in this file is not vacuous: flipping
/// one byte of an otherwise-identical snapshot must be caught. This is the
/// closest thing to a mutation test available here without touching
/// `src/`, which is out of scope for this measurement (the comparisons
/// above are plain `Vec<u8>` equality, which cannot be made to silently
/// ignore a one-byte difference by anything in this test file — this just
/// demonstrates that directly).
#[test]
fn a_single_flipped_byte_is_detected() {
    let (w, _) = build_world([0, 1, 2]);
    let original = w.snapshot_world();
    let mut corrupted = original.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0x01;
    assert_ne!(
        original, corrupted,
        "a one-byte difference must not compare equal"
    );
}
