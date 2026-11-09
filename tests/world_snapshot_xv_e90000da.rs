//! Snapshots of `e90000da…`, the stepping semantics identifier of the build before alice-det-math
//! was folded per function (main `a58b37cd`, `alice-det-math` 0.5.2 at 2d01c63f), are refused by
//! this build, and the header facts of the two fixtures that build wrote stay frozen here.
//!
//! The fixtures were written outside this tree by a crate depending on `alice-physics-core` at
//! `a58b37cd9dc874e3653b985e4c4bb89e421a5736` (path dependency, `default-features = false,
//! features = ["std"]`), `cargo run --release -- <dir>`: `scene()` (below, the same source)
//! stepped 40 times by `dt()` and written by `snapshot_world` is
//! `tests/fixtures/world_snapshot_xv_e90000da_a.bin`; the same world stepped 40 more times is
//! `…_b.bin`. The writer printed semantics `e90000da…`, `scene().law_id(..)` `806977a8…`,
//! `a` 5096 bytes (sha256 `c302d779…`), `b` 6021 bytes (sha256 `0f64c9a2…`).
//!
//! # History
//!
//! This file was a cross-version oracle while `e90000da…` was listed in
//! `PREVIOUS_PHYSICS_SEMANTICS_IDS`: folding alice-det-math per function moved the identifier
//! (to `c82742de…`) and no step, and `a` restored and stepped 40 times gave `b` byte for byte.
//! The stepping then changed: the sleep rule became elapsed time (a world stepped at a step other
//! than `1/60 s` falls asleep at another step), sleep became per island, and a sleeper a step
//! moves now wakes its island. This build no longer reproduces that build's steps, so
//! `e90000da…` was taken out of the list; its snapshots are format 6, whose sleep record counts
//! steps.
//!
//! What this file checks now: the fixtures are still what the writer wrote (sizes, sha256, the
//! header identifiers, and the `law_id` recomputed from the writer's rules as that build encoded
//! them), and every way of restoring either of them is refused as a semantics mismatch naming
//! `e90000da…` and this build's identifier, the target world unchanged.
//!
//! oracle: the header identifiers are compared with what the writing build printed, not with
//! anything this build derives from the blobs.

use alice_physics_core::solver::DistanceConstraint;
use alice_physics_core::{
    Fix128, LegacyPolicy, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldSnapshotError,
    PHYSICS_SEMANTICS_ID, PREVIOUS_PHYSICS_SEMANTICS_IDS,
};
use sha2::{Digest, Sha256};

#[path = "support/law_rows.rs"]
mod law_rows;
use law_rows::{spec, Laws};

/// The rules of `scene()` folded as the writing build encoded them: gravity (0, −7, 0), the
/// damping rate 3/10, and the sleep rule as the count of 60 idle steps.
fn writer_laws() -> Laws {
    let mut laws = Laws::defaults();
    laws.config.gravity = Vec3Fix::from_int(0, -7, 0);
    laws.config.damping_rate = Fix128::from_ratio(3, 10);
    laws.legacy_sleep_steps = Some(60);
    laws
}

const A: &[u8] = include_bytes!("fixtures/world_snapshot_xv_e90000da_a.bin");
const B: &[u8] = include_bytes!("fixtures/world_snapshot_xv_e90000da_b.bin");
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
/// What the build at `a58b37cd` printed
const WRITER_SEMANTICS: &str = "e90000da9e7073b2c82c1851e78a9c08d1640298bb3dc4e11ea91fb0b8ad01d6";
const WRITER_LAW_ID: &str = "806977a8f9d09eb96ac4d6d713fc8fc924f2c78618a736d048a8103134c5b8a2";
const A_SHA256: &str = "c302d7792717beaa9f19628ed20f7fc95743e8b772377f40177a3394ccc49d27";
const B_SHA256: &str = "0f64c9a2149c521bcb042f5bc6ece686097b7b0f8f69189571bdf7f45167b3e9";

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -7, 0),
        damping_rate: Fix128::from_ratio(3, 10),
        ..PhysicsConfig::default()
    });
    // a static sphere as a floor, with a dynamic sphere falling onto it
    w.add_body_with_radius(
        RigidBody::new_static(Vec3Fix::from_int(0, -4, 0)),
        Fix128::from_int(3),
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 1, 0), Fix128::ONE),
        Fix128::ONE,
    );
    // two spheres that meet mid-air
    let mut a = RigidBody::new_dynamic(Vec3Fix::from_int(3, 4, 0), Fix128::ONE);
    a.velocity = Vec3Fix::from_int(-1, 0, 0);
    w.add_body_with_radius(a, Fix128::ONE);
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(1, 4, 0), Fix128::from_int(2));
    b.velocity = Vec3Fix::from_int(1, 2, 0);
    w.add_body_with_radius(b, Fix128::ONE);
    // a chain of three bodies held by two distance constraints
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(-4, 6, 1),
        Fix128::ONE,
    ));
    let mut c = RigidBody::new_dynamic(Vec3Fix::from_int(-2, 6, 1), Fix128::ONE);
    c.velocity = Vec3Fix::from_int(0, 0, 2);
    w.add_body(c);
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 6, 1),
        Fix128::from_int(3),
    ));
    for (i, j) in [(4usize, 5usize), (5, 6)] {
        w.add_distance_constraint(DistanceConstraint {
            body_a: i,
            body_b: j,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            target_distance: Fix128::from_int(2),
            compliance: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        });
    }
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
    assert_eq!(A.len(), 5096);
    assert_eq!(B.len(), 6021);
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
    // the header law_id is the scene's rule under the writer's semantics, folded as the writer
    // encoded it (the sleep rule as 60 idle steps); this build folds the sleep rule as a time,
    // so its own `law_id` under the writer's semantics is another one
    assert_eq!(spec(&writer_laws(), &writer), A[LAW_AT..LAW_AT + 32]);
    assert_ne!(scene().law_id(&writer), A[LAW_AT..LAW_AT + 32]);
}

/// This build does not list `e90000da…` as an earlier semantics it reproduces
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
