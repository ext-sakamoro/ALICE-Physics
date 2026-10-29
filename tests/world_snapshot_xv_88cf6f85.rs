//! Snapshots of `88cf6f85…`, the stepping semantics identifier of the build before the secondary
//! chain was pinned (main `b672ff2e`, `alice-det-math` 0.5.2 at 2d01c63f), are refused by this
//! build, and the header facts of the two fixtures that build wrote stay frozen here.
//!
//! The fixtures: `tests/fixtures/world_snapshot_xv_88cf6f85_a.bin` is `scene()` stepped 30 times
//! by `dt()` and written by `snapshot_world` with the build at `b672ff2e`, and `…_b.bin` the same
//! world stepped 30 more times by that build. The writer printed semantics `88cf6f85…` and
//! `scene().law_id(..)` `bce0de45…` (with its world damping field holding the retention 0.99 per
//! step it held); `a` is 3289 bytes (sha256 `c4192689…`), `b` 3128 bytes (sha256 `93226904…`).
//!
//! # History
//!
//! This file was a cross-version oracle while `88cf6f85…` was listed in
//! `PREVIOUS_PHYSICS_SEMANTICS_IDS`. Pinning the secondary chain had added one entry to
//! `PHYSICS_SEMANTICS_PINS` and changed no other, and two checks showed that every path the
//! earlier identifier pinned stepped to the same bits:
//!
//! * the fold of the pins without the `"secondary chain"` entry was `88cf6f85…` itself (the test
//!   `the_pins_without_the_secondary_chain_fold_to_the_earlier_identifier`);
//! * `a`, restored and stepped 30 times, gave the world `b` restores to, byte for byte.
//!
//! The stepping then changed: shaped bodies collide with static colliders by their shape, an
//! overlap with a static surface at the start of a substep is pushed out without becoming
//! velocity, and damping became a rate per second (0 by default, where the earlier default was a
//! retention 0.99 per step). Several pins moved, so the fold of today's pins without the secondary
//! chain is no longer `88cf6f85…` (that test was removed: it folded today's pins, so it could only
//! fail), and this build no longer reproduces that build's steps. `88cf6f85…` was taken out of
//! `PREVIOUS_PHYSICS_SEMANTICS_IDS`, whose rule lists an earlier identifier only when a
//! cross-version oracle shows a snapshot of that release, restored and stepped here, equal to the
//! release's own steps.
//!
//! What this file checks now: the fixtures are still what the writer wrote (sizes, sha256, the
//! header identifiers, and the `law_id` recomputed from the writer's scene), and every way of
//! restoring either of them is refused as a semantics mismatch naming `88cf6f85…` and this
//! build's identifier, the target world unchanged.
//!
//! oracle: the header identifiers are compared with what the writing build printed, not with
//! anything this build derives from the blobs.

use alice_physics_core::solver::DistanceConstraint;
use alice_physics_core::{
    Fix128, LegacyPolicy, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldSnapshotError,
    PHYSICS_SEMANTICS_ID, PREVIOUS_PHYSICS_SEMANTICS_IDS,
};
use sha2::{Digest, Sha256};

const A: &[u8] = include_bytes!("fixtures/world_snapshot_xv_88cf6f85_a.bin");
const B: &[u8] = include_bytes!("fixtures/world_snapshot_xv_88cf6f85_b.bin");
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
/// What the build at `b672ff2e` printed
const WRITER_SEMANTICS: &str = "88cf6f85f1943db5d348d22275e4962602082837a0213c4ca3219d105e1fe7a6";
const WRITER_LAW_ID: &str = "bce0de453c17e5f66d330c11c503bf7e6a552822fd2e7c4488fb0c23bc56753c";
const A_SHA256: &str = "c41926894f7b04d565243ee5a0857fe509ef23aac741fe04272252de77a3558d";
const B_SHA256: &str = "9322690468d96cb9ce7b849f8c901fe3efd6cc37e4923c4e92b6fd9eeb54d4eb";

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -7, 0),
        ..PhysicsConfig::default()
    });
    // two spheres that meet, so the contact solve runs
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 4, 0), Fix128::ONE),
        Fix128::ONE,
    );
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 1, 0), Fix128::from_int(2));
    b.velocity = Vec3Fix::from_int(0, 3, 0);
    w.add_body_with_radius(b, Fix128::ONE);
    // a pendulum pair held by a distance constraint
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(-4, 6, 1),
        Fix128::ONE,
    ));
    let mut c = RigidBody::new_dynamic(Vec3Fix::from_int(-2, 6, 1), Fix128::ONE);
    c.velocity = Vec3Fix::from_int(0, 0, 2);
    w.add_body(c);
    w.add_distance_constraint(DistanceConstraint {
        body_a: 2,
        body_b: 3,
        local_anchor_a: Vec3Fix::ZERO,
        local_anchor_b: Vec3Fix::ZERO,
        target_distance: Fix128::from_int(2),
        compliance: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    });
    w
}

/// `scene()` as the writing build held it: its world damping field was a velocity retention per
/// step, 0.99 by default, where this build's field is a rate per second (0 by default). `law_id`
/// encodes the field's value, so the writer's `law_id` is reproduced only from the value it held;
/// the field is set to that raw value to recompute it, not used as a rate.
fn writer_scene() -> PhysicsWorld {
    let mut w = scene();
    w.config.damping_rate = Fix128::from_ratio(99, 100);
    w
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn writer_id() -> [u8; 32] {
    A[SEMANTICS_AT..SEMANTICS_AT + 32]
        .try_into()
        .expect("32 bytes")
}

#[test]
fn the_fixtures_are_what_the_writer_wrote() {
    assert_eq!(A.len(), 3289);
    assert_eq!(B.len(), 3128);
    assert_eq!(hex(&Sha256::digest(A)), A_SHA256);
    assert_eq!(hex(&Sha256::digest(B)), B_SHA256);
    for blob in [A, B] {
        assert_eq!(
            hex(&blob[SEMANTICS_AT..SEMANTICS_AT + 32]),
            WRITER_SEMANTICS
        );
    }
    assert_eq!(hex(&A[LAW_AT..LAW_AT + 32]), WRITER_LAW_ID);
    let writer = writer_id();
    assert_ne!(writer, PHYSICS_SEMANTICS_ID);
    // the header law_id is the scene's rule under the writer's semantics, with the world damping
    // field holding the value the writer held; this build's own `scene()` holds rate 0 there, so
    // it gives another id
    assert_eq!(writer_scene().law_id(&writer), A[LAW_AT..LAW_AT + 32]);
    assert_ne!(scene().law_id(&writer), A[LAW_AT..LAW_AT + 32]);
}

/// This build does not list `88cf6f85…` as an earlier semantics it reproduces
#[test]
fn the_earlier_identifier_is_not_listed() {
    assert!(!PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&writer_id()));
    assert!(!PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&PHYSICS_SEMANTICS_ID));
}

/// Every way of restoring `blob` (the plain restore, the checked one with no expected id, the
/// target's own `law_id` and the header's own, under both legacy policies, and the legacy damping
/// reader with the writer's step) refuses it as a semantics mismatch naming the writer's
/// identifier and this build's, and leaves the target's snapshot bytes unchanged
fn refused_every_way(blob: &[u8]) -> usize {
    let want = || WorldSnapshotError::SemanticsMismatch {
        stored: writer_id(),
        expected: PHYSICS_SEMANTICS_ID,
    };
    let mut compared = 0;
    let mut target = scene();
    let before = target.snapshot_world();
    assert_eq!(target.restore_world(blob), Err(want()));
    assert_eq!(target.snapshot_world(), before);
    compared += 1;
    let own = scene().law_id(&PHYSICS_SEMANTICS_ID);
    let header: [u8; 32] = blob[LAW_AT..LAW_AT + 32].try_into().expect("32 bytes");
    for expected in [None, Some(own), Some(header)] {
        for policy in [LegacyPolicy::AcceptUnpinned, LegacyPolicy::RequirePinned] {
            let mut target = scene();
            let before = target.snapshot_world();
            assert_eq!(
                target.restore_world_checked(blob, expected.as_ref(), policy),
                Err(want()),
                "expected {expected:?}, policy {policy:?}"
            );
            assert_eq!(target.snapshot_world(), before);
            compared += 1;
        }
    }
    let mut target = scene();
    let before = target.snapshot_world();
    assert_eq!(target.restore_world_legacy_damping(blob, dt()), Err(want()));
    assert_eq!(target.snapshot_world(), before);
    compared += 1;
    compared
}

#[test]
fn the_first_fixture_is_refused_every_way() {
    assert_eq!(refused_every_way(A), 8);
}

#[test]
fn the_second_fixture_is_refused_every_way() {
    assert_eq!(refused_every_way(B), 8);
}
