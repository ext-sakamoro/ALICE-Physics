//! ALICE-Law G7 / G8: a participant that runs its own law under its own arithmetic declares
//! that arithmetic's semantics identifier (`Participant::semantics_id`), the world's `law_id`
//! carries it, and a snapshot payload computed under another identifier is refused as a
//! semantics mismatch, told apart from a corrupt payload.
//!
//! oracle: the law_id encoding table of `PhysicsWorld::law_id` (rows 1 to 16 unchanged, row 17
//! `participant-semantics` written only when a participant declares an identifier), so
//!
//! - two worlds that differ only in one participant's declared identifier (or in declaring one
//!   at all) have different law_id;
//! - a world whose participants declare none keeps the law_id it had before the row existed
//!   (pinned below from main 88f76d3b, where the row did not exist);
//! - a payload carrying identifier A restored into a world whose participant declares B is
//!   refused by `restore_world` with `ParticipantState { error: SemanticsMismatch { stored: A,
//!   expected: B } }`, and by `restore_world_checked` with the target's own law_id as
//!   `LawIdMismatch`.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "std")]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody, WorldSnapshotError};
use alice_physics_core::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use alice_physics_core::{LegacyPolicy, PHYSICS_SEMANTICS_ID};

const KIND_TAGGED: ParticipantKind = ParticipantKind::new(0x5441_4747);

const ID_A: [u8; 32] = [0xa1; 32];
const ID_B: [u8; 32] = [0xb2; 32];

/// A participant that declares `id` (or none) and whose whole state is that identifier: its
/// payload is the 32 bytes it ran under, and it refuses a payload from another identifier.
struct Tagged {
    id: Option<[u8; 32]>,
}

impl Participant for Tagged {
    fn kind(&self) -> ParticipantKind {
        KIND_TAGGED
    }
    fn semantics_id(&self) -> Option<[u8; 32]> {
        self.id
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.id.unwrap_or([0; 32]));
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        let stored: [u8; 32] = bytes.try_into().map_err(|_| StateError::Length {
            expected: 32,
            found: bytes.len(),
        })?;
        let expected = self.id.unwrap_or([0; 32]);
        if stored == expected {
            Ok(())
        } else {
            Err(StateError::SemanticsMismatch { stored, expected })
        }
    }
    fn read_state(&mut self, _: &[u8]) {}
}

fn world(ids: &[Option<[u8; 32]>]) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    for &id in ids {
        w.add_participant(Box::new(Tagged { id }))
            .expect("register");
    }
    w
}

fn law_id(w: &PhysicsWorld) -> [u8; 32] {
    w.law_id(&PHYSICS_SEMANTICS_ID)
}

fn hex(b: &[u8; 32]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// O1: the declared identifier is part of the law: changing it, or declaring one at all,
/// changes the law_id; so does moving it to another participant
#[test]
fn a_participant_semantics_id_changes_the_law_id() {
    let none = law_id(&world(&[None, None]));
    let a = law_id(&world(&[Some(ID_A), None]));
    let b = law_id(&world(&[Some(ID_B), None]));
    let a_second = law_id(&world(&[None, Some(ID_A)]));
    let ids = [none, a, b, a_second];
    for i in 0..ids.len() {
        for j in i + 1..ids.len() {
            assert_ne!(
                ids[i],
                ids[j],
                "law_id {i} and {j} collide: {}",
                hex(&ids[i])
            );
        }
    }
    // the same declaration gives the same law_id
    assert_eq!(a, law_id(&world(&[Some(ID_A), None])));
}

/// O2: a world whose participants declare no identifier keeps the law_id of main 88f76d3b
/// (before the row existed), so the existing goldens and the version 4 snapshot headers stay
#[test]
fn participants_without_an_identifier_keep_the_law_id() {
    assert_eq!(
        hex(&law_id(&world(&[None, None]))),
        "9d2b8645ed9796ec38fcf7d54eb86eae0c3abc22dcb113078fa79ab313299b6e",
        "pinned from main 88f76d3b"
    );
}

/// O3: a payload computed under identifier A is refused in a world whose participant declares
/// B: as a semantics mismatch naming both (not as a corrupt payload) by `restore_world`, and by
/// the law_id in the header when the caller checks it with the target's own law_id
#[test]
fn a_payload_under_another_identifier_is_a_semantics_mismatch() {
    let blob = world(&[Some(ID_A)]).snapshot_world();

    let mut target = world(&[Some(ID_B)]);
    let restored = target.restore_world(&blob);
    // the message names the stored identifier first and the participant's own second, in hex
    let message = restored
        .as_ref()
        .err()
        .map(ToString::to_string)
        .unwrap_or_default();
    assert!(
        message.contains(&format!(
            "written under semantics {} and the participant runs under {}",
            hex(&ID_A),
            hex(&ID_B)
        )),
        "{message}"
    );
    match restored {
        Err(WorldSnapshotError::ParticipantState { index, error }) => {
            assert_eq!(index, 0);
            assert_eq!(
                error,
                StateError::SemanticsMismatch {
                    stored: ID_A,
                    expected: ID_B
                }
            );
        }
        other => panic!("restore_world gave {other:?}"),
    }

    let mut target = world(&[Some(ID_B)]);
    let own = law_id(&target);
    match target.restore_world_checked(&blob, Some(&own), LegacyPolicy::RequirePinned) {
        Err(WorldSnapshotError::LawIdMismatch { stored, expected }) => {
            assert_eq!(stored, law_id(&world(&[Some(ID_A)])));
            assert_eq!(expected, own);
        }
        other => panic!("restore_world_checked gave {other:?}"),
    }

    // the same identifier restores
    let mut same = world(&[Some(ID_A)]);
    let own = law_id(&same);
    assert!(same
        .restore_world_checked(&blob, Some(&own), LegacyPolicy::RequirePinned)
        .is_ok());
}
