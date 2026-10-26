//! A version 4 world snapshot written under an earlier stepping semantics
//! identifier restores only if this build lists it in
//! `PREVIOUS_PHYSICS_SEMANTICS_IDS`; every other identifier is refused.
//!
//! The list is empty since `alice-det-math` 0.5.2 at 39507d1b: its clamp
//! moved Law output bits, so this build is not shown to reproduce data
//! stamped with an identifier of an earlier generation, and the two 3.x
//! fixtures below are refused naming both identifiers, the target unchanged.
//! An earlier identifier comes back only with a cross-version oracle (a
//! snapshot written by that release, restored and stepped here, bit-equal to
//! that release's own steps).
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
//! that build.
//!
//! oracle: the identifiers are compared with the values the writing build
//! printed, not with anything this build derives from the blob, and a
//! refused restore leaves the target's snapshot bytes unchanged.

use alice_physics_core::{
    Fix128, LegacyPolicy, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldSnapshotError,
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

/// Every way of restoring `blob` (the plain restore, and the checked one with no expected id,
/// the target's own `law_id` and the header's own, under both legacy policies) refuses it as a
/// semantics mismatch naming `stored` and this build's identifier, and leaves the target unchanged
fn refused_every_way(blob: &[u8], stored: &str) -> usize {
    let want = || WorldSnapshotError::SemanticsMismatch {
        stored: unhex(stored),
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
    compared
}

/// The blob of the 3.0.x semantics (`cd9aa3b8…`) is refused: its generation's clamp differs from
/// this build's and no cross-version oracle covers it
#[test]
fn a_blob_of_the_3_0_semantics_is_refused() {
    assert_eq!(refused_every_way(BLOB, WRITER_SEMANTICS), 7);
}

/// The blob the published 3.1.0 wrote (`79ca5297…`) is refused for the same reason
#[test]
fn a_blob_of_the_released_3_1_0_semantics_is_refused() {
    assert_eq!(refused_every_way(BLOB_3_1_0, WRITER_3_1_0_SEMANTICS), 7);
}

/// An identifier that is neither this build's nor a listed earlier one is
/// refused, and the target is unchanged.
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

/// The list holds neither 3.x identifier nor this build's own.
#[test]
fn the_list_holds_no_3_x_identifier_and_not_the_current_one() {
    for id in [WRITER_SEMANTICS, WRITER_3_1_0_SEMANTICS] {
        assert!(!PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&unhex(id)), "{id}");
    }
    assert!(!PREVIOUS_PHYSICS_SEMANTICS_IDS.contains(&PHYSICS_SEMANTICS_ID));
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
