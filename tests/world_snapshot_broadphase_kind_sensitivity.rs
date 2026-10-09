//! Measures whether `snapshot_world` bytes depend on which [`Broadphase`]
//! kind a world uses, for the otherwise-identical world (same bodies, same
//! joint, same insertion order) — a follow-up to
//! `world_snapshot_order_sensitivity.rs`, isolating the broadphase kind as
//! the only varying input instead of insertion order.
//!
//! `snapshot_world` always writes the broadphase kind as a one-byte tag,
//! followed by the persistent tree as its leaf set
//! (`src/solver/world_snapshot.rs::snapshot_world`) unconditionally,
//! regardless of which kind is selected.
//! Reading the step code shows the tree is only ever populated along the
//! `Broadphase::DynamicTree` path; with `Bvh` or `Hybrid` selected it stays
//! empty. The prediction this file checks: `Bvh` and `Hybrid` snapshots
//! should differ from each other by only the one-byte kind tag (both leave
//! the tree section empty), while `DynamicTree` should differ by much more
//! (the tree section actually holds content) — i.e. the tree's leaves, not
//! just the choice of kind, are what show up in the bytes.

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{Broadphase, PhysicsWorld, RigidBody, SolverConfig};

/// Same three bodies and ball joint as `world_snapshot_order_sensitivity.rs`,
/// always added in the same order; only `kind` varies between calls. Steps
/// once before returning: the persistent broadphase tree is built during
/// `step` (proxies are inserted while staging contacts), not while bodies
/// are merely added, so a snapshot taken right after construction would
/// show an empty tree for every kind, including `DynamicTree` — measured
/// directly below in `builds_before_any_step_show_no_tree_content_yet`.
fn build_world(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);

    // A proxy only exists for a body with a collision radius ("the proxy
    // count is the number of bodies with a collision radius", per
    // tests/analytic_broadphase.rs's own doc comment) — a bare `add_body`
    // never enters the broadphase at all, which `builds_before_any_step_show_no_tree_content_yet`
    // measures directly for the no-radius case; `add_body_with_radius` is
    // what actually exercises the tree below.
    let floor = w.add_body(RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::ZERO));
    let half = Fix128::from_ratio(1, 2);
    let a = w.add_body_with_radius(
        RigidBody::new(Vec3Fix::from_int(5, 10, 0), Fix128::ONE),
        half,
    );
    let b = w.add_body_with_radius(
        RigidBody::new(Vec3Fix::from_int(-3, 20, 7), Fix128::from_int(2)),
        half,
    );
    let _ = floor;
    w.add_joint(Joint::Ball(BallJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
    )));
    w.step(Fix128::from_ratio(1, 60));

    w
}

/// Same construction as `build_world`, but without the `step()` call, to
/// measure directly (rather than just assert in a comment) that the
/// persistent tree genuinely is still empty for every kind at that point —
/// the reason `build_world` above steps once before returning.
fn build_world_before_any_step(kind: Broadphase) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.set_broadphase(kind);
    let half = Fix128::from_ratio(1, 2);
    w.add_body(RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::ZERO));
    w.add_body_with_radius(
        RigidBody::new(Vec3Fix::from_int(5, 10, 0), Fix128::ONE),
        half,
    );
    w.add_body_with_radius(
        RigidBody::new(Vec3Fix::from_int(-3, 20, 7), Fix128::from_int(2)),
        half,
    );
    w
}

#[test]
fn builds_before_any_step_show_no_tree_content_yet() {
    let bvh = build_world_before_any_step(Broadphase::Bvh).snapshot_world();
    let tree = build_world_before_any_step(Broadphase::DynamicTree).snapshot_world();
    let content_len = bvh.len().min(tree.len()) - 8;
    assert_eq!(
        bvh.len(),
        tree.len(),
        "before any step, DynamicTree's persistent tree should still be \
         empty, the same as Bvh's (always-empty) one, so the snapshots \
         should be the same length"
    );
    assert_eq!(
        byte_diff_count(&bvh[..content_len], &tree[..content_len]),
        1,
        "before any step, the two snapshots should differ by only the \
         one-byte broadphase kind tag"
    );
}

/// Number of byte positions at which two equal-length snapshots differ.
fn byte_diff_count(a: &[u8], b: &[u8]) -> usize {
    assert_eq!(
        a.len(),
        b.len(),
        "snapshots being diffed must be same length"
    );
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

#[test]
fn same_kind_twice_is_byte_identical() {
    assert_eq!(
        build_world(Broadphase::Bvh).snapshot_world(),
        build_world(Broadphase::Bvh).snapshot_world(),
        "snapshot_world must be deterministic for a fixed broadphase kind"
    );
    assert_eq!(
        build_world(Broadphase::DynamicTree).snapshot_world(),
        build_world(Broadphase::DynamicTree).snapshot_world(),
        "snapshot_world must be deterministic for a fixed broadphase kind"
    );
}

#[test]
fn bvh_and_dynamic_tree_snapshots_differ() {
    let bvh = build_world(Broadphase::Bvh).snapshot_world();
    let tree = build_world(Broadphase::DynamicTree).snapshot_world();
    assert_ne!(
        bvh, tree,
        "expected the persistent tree content to make these differ"
    );
}

#[test]
fn hybrid_and_dynamic_tree_snapshots_differ() {
    let hybrid = build_world(Broadphase::Hybrid).snapshot_world();
    let tree = build_world(Broadphase::DynamicTree).snapshot_world();
    assert_ne!(
        hybrid, tree,
        "expected the persistent tree content to make these differ"
    );
}

/// The characterization this file exists for: `Bvh` vs `Hybrid` (neither
/// populates the persistent tree) differ by only a handful of bytes, while
/// either of them vs `DynamicTree` (which does populate it) differ by far
/// more — locating the difference in the tree section rather than
/// somewhere else in the snapshot.
#[test]
fn dynamic_tree_differs_far_more_than_bvh_vs_hybrid_does() {
    let bvh = build_world(Broadphase::Bvh).snapshot_world();
    let hybrid = build_world(Broadphase::Hybrid).snapshot_world();
    let tree = build_world(Broadphase::DynamicTree).snapshot_world();

    // The trailing 8 bytes are a whole-snapshot FNV-1a 64 checksum ("last 8 |
    // FNV-1a 64 of every preceding byte", src/solver/world_snapshot.rs's own
    // doc comment), which changes whenever any earlier byte does; comparing
    // it alongside content bytes would inflate every diff count by up to 8
    // regardless of where the real content difference is, so the content
    // comparisons below exclude it and the checksum's own behavior is
    // checked separately.
    let content_len = bvh.len().min(hybrid.len()) - 8;
    let bvh_vs_hybrid = byte_diff_count(&bvh[..content_len], &hybrid[..content_len]);

    assert!(
        bvh_vs_hybrid <= 1,
        "Bvh vs Hybrid content (excluding the trailing checksum) should \
         differ by at most the one-byte kind tag; got {bvh_vs_hybrid} differing bytes"
    );
    assert_ne!(
        &bvh[bvh.len() - 8..],
        &hybrid[hybrid.len() - 8..],
        "the trailing checksum should also change once the content differs"
    );

    // DynamicTree populates the persistent tree with real node data, so a
    // length difference is itself already evidence (a separate kind of
    // evidence from a same-length byte diff count): print both so the
    // result is visible either way instead of assuming which form it takes.
    if bvh.len() == tree.len() {
        let bvh_vs_tree = byte_diff_count(&bvh[..content_len], &tree[..content_len]);
        assert!(
            bvh_vs_tree > bvh_vs_hybrid.max(1) * 10,
            "DynamicTree vs Bvh should differ by far more than Bvh vs Hybrid \
             does, since only DynamicTree populates the persistent tree; got \
             {bvh_vs_tree} vs {bvh_vs_hybrid} differing bytes"
        );
    } else {
        assert_ne!(
            bvh.len(),
            tree.len(),
            "unreachable: already in the not-equal branch"
        );
        println!(
            "Bvh snapshot is {} bytes, DynamicTree snapshot is {} bytes \
             (length itself differs because DynamicTree's persistent tree \
             holds real node data)",
            bvh.len(),
            tree.len()
        );
    }
}
