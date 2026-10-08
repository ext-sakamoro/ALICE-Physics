//! Version 4 world snapshot: the header carries the writer's stepping
//! semantics identifier and the world's rule identifier, through the public
//! API only.
//!
//! oracle: the identifiers themselves, computed independently of the blob
//! (`PHYSICS_SEMANTICS_ID` and `PhysicsWorld::law_id` of the written world),
//! the documented byte offsets, and the original world for round trips.
//! Every rejection is checked with the checksum recomputed, so the check
//! under test is the one that fires, and with the target world compared
//! before and after.

use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::joint::{BallJoint, Joint};
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, StepRule,
    SubstepCtx,
};
use alice_physics::{
    Fix128, LawCheck, PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend, Vec3Fix,
    WorldCcdConfig, WorldSnapshotError, PHYSICS_SEMANTICS_ID,
};

const MAGIC_END: usize = 4;
const VERSION_AT: usize = 4;
const RESERVED_AT: usize = 6;
const LEN_AT: usize = 8;
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
const PAYLOAD_AT: usize = 80;
const CHECKSUM: usize = 8;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 5, 0), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(2, 5, 1), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    w.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        a,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, 0),
    )));
    w.step_n(3, dt());
    w
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in bytes {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    h
}

/// Recompute the trailing FNV-1a 64 so only the targeted check can fire.
fn reseal(mut b: Vec<u8>) -> Vec<u8> {
    let n = b.len() - CHECKSUM;
    let h = fnv1a(&b[..n]);
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

fn payload_len(b: &[u8]) -> usize {
    u64::from_le_bytes(b[LEN_AT..LEN_AT + 8].try_into().expect("8 bytes")) as usize
}

/// The version 3 form of a version 4 blob: the same payload behind the
/// 16-byte header (no identifiers).
fn as_version_3(v4: &[u8]) -> Vec<u8> {
    let mut out = v4[..SEMANTICS_AT].to_vec();
    out[VERSION_AT..VERSION_AT + 2].copy_from_slice(&3u16.to_le_bytes());
    out.extend_from_slice(&v4[PAYLOAD_AT..v4.len() - CHECKSUM]);
    out.extend_from_slice(&[0; CHECKSUM]);
    reseal(out)
}

fn id(b: &[u8], at: usize) -> [u8; 32] {
    b[at..at + 32].try_into().expect("32 bytes")
}

/// `restore_world_checked` on `t`, asserting `t` is unchanged on error.
fn checked(
    t: &mut PhysicsWorld,
    data: &[u8],
    expected: Option<&[u8; 32]>,
) -> Result<LawCheck, WorldSnapshotError> {
    let before = t.snapshot_world();
    let r = t.restore_world_checked(data, expected);
    if r.is_err() {
        assert_eq!(
            t.snapshot_world(),
            before,
            "rejected restore touched the target"
        );
    }
    r
}

#[test]
fn the_writer_emits_version_4() {
    assert_eq!(PhysicsWorld::WORLD_SNAPSHOT_VERSION, 4);
    let b = scene().snapshot_world();
    assert_eq!(&b[..MAGIC_END], b"APWS");
    assert_eq!(&b[VERSION_AT..VERSION_AT + 2], &4u16.to_le_bytes());
}

/// The header holds `PHYSICS_SEMANTICS_ID` at `[16..48)` and the world's
/// `law_id` under it at `[48..80)`, the reserved bytes stay 0, and the
/// declared payload length counts what lies between the 80-byte header and
/// the checksum. Two worlds that differ only in rule (backend, gravity)
/// carry their own, different `law_id`.
#[test]
fn header_holds_the_identifiers_at_their_offsets() {
    let mut worlds = vec![scene()];
    for cfg in [
        PhysicsConfig {
            solver_backend: SolverBackend::Tgs,
            ..PhysicsConfig::default()
        },
        PhysicsConfig {
            gravity: Vec3Fix::from_int(0, -3, 1),
            ..PhysicsConfig::default()
        },
    ] {
        let mut w = PhysicsWorld::new(cfg);
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        worlds.push(w);
    }
    let mut law_ids = Vec::new();
    for w in &worlds {
        let b = w.snapshot_world();
        assert_eq!(&b[RESERVED_AT..RESERVED_AT + 2], &[0, 0]);
        assert_eq!(id(&b, SEMANTICS_AT), PHYSICS_SEMANTICS_ID);
        let law = w.law_id(&PHYSICS_SEMANTICS_ID);
        assert_eq!(id(&b, LAW_AT), law);
        assert_ne!(
            law,
            w.law_id(&[0; 32]),
            "the law id depends on the semantics"
        );
        assert_eq!(payload_len(&b), b.len() - PAYLOAD_AT - CHECKSUM);
        law_ids.push(law);
    }
    assert_ne!(law_ids[0], law_ids[1]);
    assert_ne!(law_ids[1], law_ids[2]);
    assert_ne!(law_ids[0], law_ids[2]);
}

/// A version 4 blob restores to a world that writes the same bytes, and
/// both step on bit for bit.
#[test]
fn version_4_round_trips_bytes_and_steps() {
    let mut a = scene();
    let blob = a.snapshot_world();
    let mut b = PhysicsWorld::from_world_snapshot(&blob).expect("v4 blob");
    assert_eq!(b.snapshot_world(), blob);
    let mut c = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(
        c.restore_world_checked(&blob, None),
        Ok(LawCheck::SemanticsVerified)
    );
    for _ in 0..30 {
        a.step(dt());
        b.step(dt());
        c.step(dt());
    }
    assert_eq!(a.snapshot_world(), b.snapshot_world());
    assert_eq!(a.snapshot_world(), c.snapshot_world());
}

/// The three results of `restore_world_checked` on a version 4 blob.
#[test]
fn checked_restore_reports_what_it_verified() {
    let a = scene();
    let blob = a.snapshot_world();
    let law = a.law_id(&PHYSICS_SEMANTICS_ID);

    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(checked(&mut t, &blob, Some(&law)), Ok(LawCheck::Verified));
    assert_eq!(t.snapshot_world(), blob);

    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(
        checked(&mut t, &blob, None),
        Ok(LawCheck::SemanticsVerified)
    );
    assert_eq!(t.snapshot_world(), blob);

    let mut other = law;
    other[31] ^= 0x01;
    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(
        checked(&mut t, &blob, Some(&other)),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: law,
            expected: other
        })
    );
}

/// The expected identifier is compared with the header as written: a header
/// `law_id` changed (checksum recomputed) is refused against the world's
/// true identifier, although the payload is that world's, and the plain
/// restore, which does not compare it, still accepts the blob.
#[test]
fn the_header_law_id_is_what_is_compared() {
    let a = scene();
    let law = a.law_id(&PHYSICS_SEMANTICS_ID);
    let mut x = a.snapshot_world();
    x[LAW_AT] ^= 0x80;
    let x = reseal(x);
    let tampered = id(&x, LAW_AT);

    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(
        checked(&mut t, &x, Some(&law)),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: tampered,
            expected: law
        })
    );
    assert_eq!(checked(&mut t, &x, Some(&tampered)), Ok(LawCheck::Verified));
    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(t.restore_world(&x), Ok(()));
}

/// Any one byte of the header `semantics_id` changed, with the checksum
/// recomputed, is refused by every reader with `SemanticsMismatch`, and the
/// target is untouched.
#[test]
fn a_foreign_semantics_id_is_refused() {
    let a = scene();
    let blob = a.snapshot_world();
    let law = a.law_id(&PHYSICS_SEMANTICS_ID);
    for i in SEMANTICS_AT..LAW_AT {
        let mut x = blob.clone();
        x[i] ^= 0x01;
        let x = reseal(x);
        let want = WorldSnapshotError::SemanticsMismatch {
            stored: id(&x, SEMANTICS_AT),
            expected: PHYSICS_SEMANTICS_ID,
        };
        let mut t = scene();
        t.step(dt());
        assert_eq!(checked(&mut t, &x, Some(&law)), Err(want), "byte {i}");
        assert_eq!(checked(&mut t, &x, None), Err(want), "byte {i}");
        let before = t.snapshot_world();
        assert_eq!(t.restore_world(&x), Err(want), "byte {i}");
        assert_eq!(t.snapshot_world(), before);
        assert_eq!(
            PhysicsWorld::from_world_snapshot(&x).err(),
            Some(want),
            "byte {i}"
        );
    }
}

/// Version 1, 2 and 3 blobs are still read and reported as `Unpinned`,
/// whether or not an identifier is expected (it cannot be compared).
#[test]
fn older_versions_are_accepted_as_unpinned() {
    let v1: &[u8] = include_bytes!("fixtures/world_snapshot_v1_stacked.bin");
    let v2: &[u8] = include_bytes!("fixtures/world_snapshot_v2_stacked.bin");
    let a = scene();
    let v3 = as_version_3(&a.snapshot_world());
    let some = a.law_id(&PHYSICS_SEMANTICS_ID);
    for (version, blob) in [(1u16, v1), (2, v2), (3, &v3[..])] {
        assert_eq!(&blob[VERSION_AT..VERSION_AT + 2], &version.to_le_bytes());
        for expected in [None, Some(&some), Some(&[0xAB; 32])] {
            let mut t = PhysicsWorld::new(PhysicsConfig::default());
            assert_eq!(
                checked(&mut t, blob, expected),
                Ok(LawCheck::Unpinned),
                "version {version}, {expected:?}"
            );
        }
        let mut t = PhysicsWorld::new(PhysicsConfig::default());
        assert_eq!(t.restore_world(blob), Ok(()), "version {version}");
    }
    // the version 3 form holds the same world as the version 4 blob
    let b = PhysicsWorld::from_world_snapshot(&v3).expect("v3 blob");
    assert_eq!(b.snapshot_world(), a.snapshot_world());
}

/// Every prefix of a version 4 blob, including those that end inside the
/// 64 identifier bytes, is refused as truncated.
#[test]
fn every_prefix_of_a_version_4_blob_is_truncated() {
    let b = scene().snapshot_world();
    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    for len in 0..b.len() {
        assert_eq!(
            checked(&mut t, &b[..len], None),
            Err(WorldSnapshotError::Truncated),
            "prefix of {len} bytes"
        );
    }
}

/// A blob of the version after this build's is refused: a reader only
/// accepts versions up to its own `WORLD_SNAPSHOT_VERSION`.
#[test]
fn the_next_version_is_unsupported() {
    let mut x = scene().snapshot_world();
    let next = PhysicsWorld::WORLD_SNAPSHOT_VERSION + 1;
    x[VERSION_AT..VERSION_AT + 2].copy_from_slice(&next.to_le_bytes());
    let x = reseal(x);
    let mut t = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(
        checked(&mut t, &x, None),
        Err(WorldSnapshotError::UnsupportedVersion {
            found: next,
            supported: PhysicsWorld::WORLD_SNAPSHOT_VERSION
        })
    );
}

// ── header law id versus the restored world's law id ─────────────────────

const KIND_TAG: ParticipantKind = ParticipantKind::new(0x5441_4721);

/// A participant with no state whose step rule is chosen at construction.
struct Tag(StepRule);

impl Participant for Tag {
    fn kind(&self) -> ParticipantKind {
        KIND_TAG
    }
    fn step_rule(&self) -> StepRule {
        self.0
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, _: &mut Vec<u8>) {}
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.is_empty() {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, _: &[u8]) {}
}

/// A world whose rule is not the default in every part `law_id` reads from
/// the blob: material table and pair override, force fields, continuous
/// collision, contact cache settings, and one participant.
fn rich_world(rule: StepRule) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        solver_backend: SolverBackend::Tgs,
        substeps: 3,
        ..PhysicsConfig::default()
    });
    let metal = w.material_table.register_metal();
    let rubber = w.material_table.register_rubber();
    w.material_table.set_pair_override(
        rubber,
        metal,
        Fix128::from_ratio(3, 10),
        Fix128::from_ratio(1, 10),
    );
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE),
        Fix128::from_ratio(1, 2),
    );
    w.set_body_material(a, metal);
    w.add_force_field(ForceFieldInstance::new(ForceField::Drag {
        coefficient: Fix128::from_ratio(1, 10),
    }));
    w.add_force_field(
        ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::from_int(1, 0, 0),
            strength: Fix128::from_int(2),
        })
        .with_affected_bodies(vec![a]),
    );
    w.set_continuous_collision(
        WorldCcdConfig::on().with_motion_threshold(Fix128::from_ratio(1, 8)),
    );
    w.add_participant(Box::new(Tag(rule))).expect("register");
    w.step_n(2, dt());
    w
}

/// Measured: restoring into a world that holds the same participant
/// (same kind, same step rule) gives a world whose `law_id` equals the
/// header's, with every other part `law_id` reads taken from the blob. A
/// target whose participant has the same kind but another step rule is
/// accepted (restore checks kinds only), and the restored world's `law_id`
/// then differs from the header's: the header records the writer's rule,
/// not the restored world's.
#[test]
fn header_law_id_versus_the_restored_world() {
    let a = rich_world(StepRule::FollowSubstep);
    let blob = a.snapshot_world();
    let header = id(&blob, LAW_AT);
    assert_eq!(header, a.law_id(&PHYSICS_SEMANTICS_ID));
    assert_ne!(
        header,
        scene().law_id(&PHYSICS_SEMANTICS_ID),
        "the rich world's rule differs from the default scene's"
    );

    // same participant: equal
    let mut same = PhysicsWorld::new(PhysicsConfig::default());
    same.add_participant(Box::new(Tag(StepRule::FollowSubstep)))
        .expect("register");
    assert_eq!(
        same.restore_world_checked(&blob, Some(&header)),
        Ok(LawCheck::Verified)
    );
    assert_eq!(same.law_id(&PHYSICS_SEMANTICS_ID), header);

    // same kind, other step rule: accepted, and the identifiers differ
    let mut other = PhysicsWorld::new(PhysicsConfig::default());
    other
        .add_participant(Box::new(Tag(StepRule::Subcycle)))
        .expect("register");
    assert_eq!(
        other.restore_world_checked(&blob, Some(&header)),
        Ok(LawCheck::Verified)
    );
    assert_ne!(other.law_id(&PHYSICS_SEMANTICS_ID), header);
}

/// The intended use: the target passes its own `law_id`, taken before the
/// restore. A target whose participant has the same kind as the blob's but
/// another step rule refuses the blob with `LawIdMismatch` and keeps its
/// bytes, although the plain restore would accept it; a target with the same
/// participant passes its own id and gets `Verified`.
#[test]
fn the_target_own_law_id_refuses_another_step_rule() {
    let blob = rich_world(StepRule::FollowSubstep).snapshot_world();
    let header = id(&blob, LAW_AT);

    let mut other = PhysicsWorld::new(PhysicsConfig::default());
    other
        .add_participant(Box::new(Tag(StepRule::Subcycle)))
        .expect("register");
    let expected = other.law_id(&PHYSICS_SEMANTICS_ID);
    assert_ne!(expected, header);
    let before = other.snapshot_world();
    assert_eq!(
        other.restore_world_checked(&blob, Some(&expected)),
        Err(WorldSnapshotError::LawIdMismatch {
            stored: header,
            expected
        })
    );
    assert_eq!(other.snapshot_world(), before);
    assert_eq!(other.law_id(&PHYSICS_SEMANTICS_ID), expected);

    // control: the same participant, and the rest of the rule as the blob's
    let mut same = rich_world(StepRule::FollowSubstep);
    same.step(dt());
    let expected = same.law_id(&PHYSICS_SEMANTICS_ID);
    assert_eq!(expected, header);
    assert_eq!(
        same.restore_world_checked(&blob, Some(&expected)),
        Ok(LawCheck::Verified)
    );
    assert_eq!(same.snapshot_world(), blob);
}
