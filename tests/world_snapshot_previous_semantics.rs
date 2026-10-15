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
//! oracle: the restored state is compared with `scene()` stepped 30 times
//! by this build (the stepping bits did not change with the identifier: no
//! determinism golden moved), and the identifiers with the values the
//! writing build printed, not with anything this build derives from the
//! blob.

use alice_physics::{
    Fix128, LawCheck, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix, WorldSnapshotError,
    PHYSICS_SEMANTICS_ID, PREVIOUS_PHYSICS_SEMANTICS_IDS,
};

const BLOB: &[u8] = include_bytes!("fixtures/world_snapshot_v4_semantics_cd9aa3b8.bin");
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
const CHECKSUM: usize = 8;

/// The semantics identifier of `alice-physics` 3.0.x, as the writer printed it.
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
        target.restore_world_checked(BLOB, Some(&expected)),
        Ok(LawCheck::VerifiedUnderPrevious {
            semantics_id: unhex(WRITER_SEMANTICS)
        })
    );
    same_bodies(&target, &stepped());
}

#[test]
fn without_an_expected_id_only_the_semantics_is_reported() {
    let mut target = scene();
    assert_eq!(
        target.restore_world_checked(BLOB, None),
        Ok(LawCheck::SemanticsPrevious {
            semantics_id: unhex(WRITER_SEMANTICS)
        })
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
            target.restore_world_checked(BLOB, Some(&wrong)),
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
        target.restore_world_checked(BLOB, Some(&expected)),
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
        target.restore_world_checked(&blob, Some(&current)),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: current,
            expected: unhex(WRITER_LAW_ID),
        })
    );
    assert_eq!(target.snapshot_world(), before);
}
