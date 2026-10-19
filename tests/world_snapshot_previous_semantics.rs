//! A version 4 world snapshot written under an earlier stepping semantics
//! identifier that this build lists in `PREVIOUS_PHYSICS_SEMANTICS_IDS`
//! restores, and one under any other identifier is refused.
//!
//! `tests/fixtures/world_snapshot_v4_semantics_cd9aa3b8.bin` is `scene()`
//! stepped 30 times by `dt()` and written by `snapshot_world`, once with the
//! published `alice-physics` 3.0.1 and once with this crate at the commit
//! before the `alice-det-math` 0.5.1 change (`alice-det-math` 0.5.0 in the
//! lock): the two writers gave the same 2136 bytes. Its header carries
//! semantics identifier `cd9aa3b8…` and `law_id` `d7096844…`, both printed
//! by the writing build.
//!
//! `tests/fixtures/world_snapshot_v4_semantics_79ca5297.bin` is the same scene
//! and steps written by the published `alice-physics` 3.1.0 (a crate that
//! depends on `alice-physics = "=3.1.0"` from crates.io and calls
//! `snapshot_world`): 2136 bytes, sha256 `43eeb4be…`, semantics identifier
//! `79ca5297…` (the 3.1.0 release value, the first entry of
//! `PREVIOUS_PHYSICS_SEMANTICS_IDS`) and `law_id` `bb5c15c7…`, both printed by
//! that build. It is the test that depends on that list entry: taking
//! `79ca5297…` out of the list makes it fail.
//!
//! oracle: the restored state is compared with `scene()` stepped 30 times
//! by this build (the stepping bits did not change with the identifier: no
//! determinism golden moved), and the identifiers with the values the
//! writing build printed, not with anything this build derives from the
//! blob.

use alice_physics_core::{
    Fix128, LawCheck, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldSnapshotError,
    PHYSICS_SEMANTICS_ID, PREVIOUS_PHYSICS_SEMANTICS_IDS,
};

const BLOB: &[u8] = include_bytes!("fixtures/world_snapshot_v4_semantics_cd9aa3b8.bin");
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
const CHECKSUM: usize = 8;

/// The semantics identifier of `alice-physics` 3.0.x, as the writer printed it.
const BLOB_3_1_0: &[u8] = include_bytes!("fixtures/world_snapshot_v4_semantics_79ca5297.bin");
/// The semantics identifier and `scene().law_id(..)` the published 3.1.0 printed.
const WRITER_3_1_0_SEMANTICS: &str =
    "79ca52976f404dfb453908970ebe45f584970ff9344a56a06aae3fc9c3c38108";
const WRITER_3_1_0_LAW_ID: &str =
    "bb5c15c7521b9d6ad268a5b261090a43032e11e7f75740eb5e504a90da2a7976";
const WRITER_SEMANTICS: &str = "cd9aa3b89257fc8b3e5d6714362ce4e52d4ff142baf30de2f80324e44bdd3469";
/// `scene().law_id(&WRITER_SEMANTICS)`, as the writer printed it.
const WRITER_LAW_ID: &str = "d7096844b0294b098af7f0ad5c1efa5793ceeb808cb86e2ab3aafb85b20008e8";

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -7, 0),
        ..PhysicsConfig::default()
    });
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 20, 0),
        Fix128::ONE,
    ));
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 12, -1),
        Fix128::from_int(3),
    ));
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(-3, 6, 2), Fix128::ONE);
    b.velocity = Vec3Fix::from_int(1, 4, -2);
    w.add_body(b);
    w
}

fn stepped() -> PhysicsWorld {
    let mut w = scene();
    for _ in 0..30 {
        w.step(dt());
    }
    w
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    h
}

/// `BLOB` with 32 header bytes at `at` replaced and the checksum recomputed,
/// so the identifier check is the one that decides.
fn with_id(at: usize, id: &[u8; 32]) -> Vec<u8> {
    let mut b = BLOB.to_vec();
    b[at..at + 32].copy_from_slice(id);
    let n = b.len() - CHECKSUM;
    let h = fnv1a(&b[..n]);
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

fn same_bodies(a: &PhysicsWorld, b: &PhysicsWorld) {
    assert_eq!(a.bodies.len(), b.bodies.len());
    for (i, (x, y)) in a.bodies.iter().zip(&b.bodies).enumerate() {
        assert_eq!(x.position, y.position, "body {i} position");
        assert_eq!(x.velocity, y.velocity, "body {i} velocity");
        assert_eq!(x.rotation, y.rotation, "body {i} rotation");
        assert_eq!(
            x.angular_velocity, y.angular_velocity,
            "body {i} angular velocity"
        );
    }
}

#[test]
fn the_fixture_header_is_what_the_writer_printed() {
    assert_eq!(&BLOB[4..6], &4u16.to_le_bytes(), "a version 4 blob");
    assert_eq!(
        hex(&BLOB[SEMANTICS_AT..SEMANTICS_AT + 32]),
        WRITER_SEMANTICS
    );
    assert_eq!(hex(&BLOB[LAW_AT..LAW_AT + 32]), WRITER_LAW_ID);
    // this build's identifier is another one, so the blob is from an
    // earlier semantics, not this one
    assert_ne!(hex(&PHYSICS_SEMANTICS_ID), WRITER_SEMANTICS);
    // and the header law_id is the scene's rule under the writer's semantics
    assert_eq!(
        hex(&scene().law_id(&unhex(WRITER_SEMANTICS))),
        WRITER_LAW_ID
    );
}

/// The 3.0.x blob restores into the scene and holds the state this build
/// reaches by stepping the scene itself.
#[test]
fn a_blob_of_the_previous_semantics_restores() {
    let mut target = scene();
    target.restore_world(BLOB).expect("a 3.0.x blob restores");
    same_bodies(&target, &stepped());
    // and steps on as the original does
    let mut original = stepped();
    for _ in 0..10 {
        target.step(dt());
        original.step(dt());
    }
    same_bodies(&target, &original);
}

#[test]
fn the_released_3_1_0_fixture_header_is_what_the_writer_printed() {
    assert_eq!(BLOB_3_1_0.len(), 2136);
    assert_eq!(&BLOB_3_1_0[4..6], &4u16.to_le_bytes(), "a version 4 blob");
    assert_eq!(
        hex(&BLOB_3_1_0[SEMANTICS_AT..SEMANTICS_AT + 32]),
        WRITER_3_1_0_SEMANTICS
    );
    assert_eq!(hex(&BLOB_3_1_0[LAW_AT..LAW_AT + 32]), WRITER_3_1_0_LAW_ID);
    assert_ne!(hex(&PHYSICS_SEMANTICS_ID), WRITER_3_1_0_SEMANTICS);
    assert_eq!(
        hex(&scene().law_id(&unhex(WRITER_3_1_0_SEMANTICS))),
        WRITER_3_1_0_LAW_ID
    );
}

/// The blob the published 3.1.0 wrote restores (its identifier is the first
/// entry of `PREVIOUS_PHYSICS_SEMANTICS_IDS`), holds the state this build
/// reaches by stepping the scene, and the target's own `law_id` verifies it.
#[test]
fn a_blob_of_the_released_3_1_0_semantics_restores() {
    let mut target = scene();
    target
        .restore_world(BLOB_3_1_0)
        .expect("a blob written by the published 3.1.0 restores");
    same_bodies(&target, &stepped());

    let mut checked = scene();
    let expected = checked.law_id(&PHYSICS_SEMANTICS_ID);
    assert_eq!(
        checked.restore_world_checked(
            BLOB_3_1_0,
            Some(&expected),
            alice_physics_core::LegacyPolicy::AcceptUnpinned
        ),
        Ok(LawCheck::VerifiedUnderPrevious)
    );
    same_bodies(&checked, &stepped());
}

/// An identifier that is neither this build's nor a listed earlier one is
/// still refused, and the target is unchanged.
#[test]
fn a_semantics_id_outside_the_list_is_refused() {
    let foreign = [0x5a; 32];
    let blob = with_id(SEMANTICS_AT, &foreign);
    let mut target = scene();
    let before = target.snapshot_world();
    assert_eq!(
        target.restore_world(&blob),
        Err(WorldSnapshotError::SemanticsMismatch {
            stored: foreign,
            expected: PHYSICS_SEMANTICS_ID,
        })
    );
    assert_eq!(target.snapshot_world(), before);
}

/// The list holds the 3.0.x identifier and never this build's own.
#[test]
fn the_list_holds_the_writer_semantics_and_not_the_current_one() {
    assert!(PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&unhex(WRITER_SEMANTICS)));
    assert!(!PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&PHYSICS_SEMANTICS_ID));
}

/// The documented use (the target's own `law_id` under this build's
/// semantics, taken before the restore) verifies the 3.0.x blob.
#[test]
fn the_target_own_law_id_verifies_a_previous_blob() {
    let mut target = scene();
    let expected = target.law_id(&PHYSICS_SEMANTICS_ID);
    assert_eq!(
        target.restore_world_checked(
            BLOB,
            Some(&expected),
            alice_physics_core::LegacyPolicy::AcceptUnpinned
        ),
        Ok(LawCheck::VerifiedUnderPrevious)
    );
    // the earlier identifier the blob was written under
    assert_eq!(
        PhysicsWorld::world_snapshot_ids(BLOB),
        Ok(Some((unhex(WRITER_SEMANTICS), unhex(WRITER_LAW_ID))))
    );
    same_bodies(&target, &stepped());
}

#[test]
fn without_an_expected_id_only_the_semantics_is_reported() {
    let mut target = scene();
    assert_eq!(
        target.restore_world_checked(BLOB, None, alice_physics_core::LegacyPolicy::AcceptUnpinned),
        Ok(LawCheck::SemanticsPrevious)
    );
}

/// `expected` that is not the target's own current `law_id` is refused,
/// even the header's own value (which would compare the blob with itself).
#[test]
fn an_expected_id_other_than_the_target_own_is_refused() {
    for wrong in [[0x11; 32], unhex(WRITER_LAW_ID)] {
        let mut target = scene();
        let before = target.snapshot_world();
        assert_eq!(
            target.restore_world_checked(
                BLOB,
                Some(&wrong),
                alice_physics_core::LegacyPolicy::AcceptUnpinned
            ),
            Err(WorldSnapshotError::LawIdMismatch {
                stored: unhex(WRITER_LAW_ID),
                expected: wrong,
            })
        );
        assert_eq!(target.snapshot_world(), before);
    }
}

/// A target running another rule (another gravity) passes its own id, and
/// the header is not that target's id under the earlier semantics.
#[test]
fn a_target_of_another_rule_is_refused() {
    let mut target = scene();
    target.config.gravity = Vec3Fix::from_int(0, -3, 0);
    let expected = target.law_id(&PHYSICS_SEMANTICS_ID);
    let before = target.snapshot_world();
    assert_eq!(
        target.restore_world_checked(
            BLOB,
            Some(&expected),
            alice_physics_core::LegacyPolicy::AcceptUnpinned
        ),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: unhex(WRITER_LAW_ID),
            expected: target.law_id(&unhex(WRITER_SEMANTICS)),
        })
    );
    assert_eq!(target.snapshot_world(), before);
}

/// A header `law_id` that is not the scene's under the earlier semantics
/// (here: the scene's under this build's semantics) is refused.
#[test]
fn a_header_law_id_not_under_the_earlier_semantics_is_refused() {
    let current = scene().law_id(&PHYSICS_SEMANTICS_ID);
    let blob = with_id(LAW_AT, &current);
    let mut target = scene();
    let before = target.snapshot_world();
    assert_eq!(
        target.restore_world_checked(
            &blob,
            Some(&current),
            alice_physics_core::LegacyPolicy::AcceptUnpinned
        ),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: current,
            expected: unhex(WRITER_LAW_ID),
        })
    );
    assert_eq!(target.snapshot_world(), before);
}

/// `world_snapshot_ids` reads the header without judging it: the 3.0.x blob,
/// a blob of this build, and a blob of a foreign semantics (which a restore
/// refuses) all give their identifiers; a blob older than version 4 gives `None`; a blob
/// that fails a structural check gives that check's error.
#[test]
fn the_header_identifiers_are_read_without_judging_them() {
    let current = stepped();
    let own = current.snapshot_world();
    assert_eq!(
        PhysicsWorld::world_snapshot_ids(&own),
        Ok(Some((
            PHYSICS_SEMANTICS_ID,
            current.law_id(&PHYSICS_SEMANTICS_ID)
        )))
    );
    let foreign = with_id(SEMANTICS_AT, &[0x5a; 32]);
    assert_eq!(
        PhysicsWorld::world_snapshot_ids(&foreign),
        Ok(Some(([0x5a; 32], unhex(WRITER_LAW_ID))))
    );
    let older: &[u8] = include_bytes!("fixtures/world_snapshot_v2_stacked.bin");
    assert_eq!(PhysicsWorld::world_snapshot_ids(older), Ok(None));
    let mut corrupt = BLOB.to_vec();
    corrupt[100] ^= 1;
    assert!(matches!(
        PhysicsWorld::world_snapshot_ids(&corrupt),
        Err(WorldSnapshotError::ChecksumMismatch { .. })
    ));
    assert_eq!(
        PhysicsWorld::world_snapshot_ids(&BLOB[..10]),
        Err(WorldSnapshotError::Truncated)
    );
}
