//! Whole-world snapshot (`snapshot_world` / `restore_world` /
//! `from_world_snapshot`) and native `step_n`, through the public API only.
//!
//! oracle: the original world itself. A restored world must produce the same
//! bodies, joints, sleep state and blob as the original after every later
//! step; the per-field comparison that does not go through the encoder lives
//! in `src/solver/world_snapshot/tests.rs`.

use alice_physics::force::{ForceField, ForceFieldInstance};
use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::{
    Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SleepConfig, Vec3Fix, WorldSnapshotError,
};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn scene() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 100),
        angular_threshold: Fix128::from_ratio(1, 100),
        frames_to_sleep: 3,
    });
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)));
    let a = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 5, 0), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(Vec3Fix::from_int(2, 5, 1), Fix128::ONE),
        Fix128::from_ratio(1, 4),
    );
    // rests on nothing but has zero gravity scale: falls asleep
    let mut idle = RigidBody::new_dynamic(Vec3Fix::from_int(9, 0, 0), Fix128::ONE);
    idle.gravity_scale = Fix128::ZERO;
    w.add_body(idle);
    w.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        a,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, 0),
    )));
    let up = Vec3Fix::from_int(0, 1, 0);
    w.add_joint(Joint::Hinge(HingeJoint::new(
        a,
        b,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-1, 0, -1),
        up,
        up,
    )));
    w.add_force_field(
        ForceFieldInstance::new(ForceField::Directional {
            direction: Vec3Fix::from_int(0, 0, 1),
            strength: Fix128::from_int(2),
        })
        .with_affected_bodies(vec![b]),
    );
    w
}

/// Everything observable through the public API that later steps depend on.
fn observe(w: &PhysicsWorld) -> String {
    format!(
        "{:?} {:?} {:?} {:?} {}",
        w.bodies,
        w.joints,
        w.islands.sleep_data,
        w.config,
        w.overflow_detected()
    )
}

#[test]
fn restored_world_matches_the_original_after_every_later_step() {
    for (k, m) in [(0, 0), (0, 4), (3, 3), (12, 20)] {
        let mut a = scene();
        a.step_n(k, dt());
        let blob = a.snapshot_world();
        let mut b = PhysicsWorld::from_world_snapshot(&blob).expect("restore");
        assert_eq!(observe(&a), observe(&b), "k={k} at restore");
        assert_eq!(b.snapshot_world(), blob, "k={k} re-snapshot");
        for i in 0..m {
            a.step(dt());
            b.step(dt());
            assert_eq!(observe(&a), observe(&b), "k={k} step {i}");
        }
        assert_eq!(a.snapshot_world(), b.snapshot_world(), "k={k} m={m}");
    }
}

/// Control: the scene really has a sleeping body when the snapshot is taken
/// at k = 12, so the round trip above covers sleep state.
#[test]
fn scene_has_a_sleeping_body_at_k12() {
    let mut a = scene();
    a.step_n(12, dt());
    assert!(a.is_sleeping(3));
    assert!(!a.is_sleeping(1));
}

/// A branch restored from the snapshot does not share state with the
/// original: stepping one leaves the other where it was.
#[test]
fn branches_are_independent() {
    let mut a = scene();
    a.step_n(5, dt());
    let blob = a.snapshot_world();
    let mut b = PhysicsWorld::from_world_snapshot(&blob).unwrap();
    b.step_n(10, dt());
    assert_eq!(a.snapshot_world(), blob);
    assert_ne!(b.snapshot_world(), blob);
}

#[test]
fn restore_world_replaces_a_different_population() {
    let mut src = scene();
    src.step_n(4, dt());
    let blob = src.snapshot_world();
    let mut dst = PhysicsWorld::new(PhysicsConfig::default());
    for i in 0..9 {
        dst.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(i, 0, 0),
            Fix128::ONE,
        ));
    }
    dst.restore_world(&blob).unwrap();
    assert_eq!(observe(&dst), observe(&src));
    src.step_n(6, dt());
    dst.step_n(6, dt());
    assert_eq!(dst.snapshot_world(), src.snapshot_world());
}

#[test]
fn step_n_is_step_repeated_and_zero_is_a_no_op() {
    for n in [0usize, 1, 5, 17] {
        let mut a = scene();
        let mut b = scene();
        a.step_n(n, dt());
        for _ in 0..n {
            b.step(dt());
        }
        assert_eq!(a.snapshot_world(), b.snapshot_world(), "n={n}");
    }
    let mut a = scene();
    let before = a.snapshot_world();
    a.step_n(0, dt());
    assert_eq!(a.snapshot_world(), before);
    a.step_n(3, -dt());
    assert_eq!(
        a.snapshot_world(),
        before,
        "non-positive dt is a no-op like step"
    );
}

#[test]
fn empty_world_round_trips() {
    let a = PhysicsWorld::new(PhysicsConfig::default());
    let blob = a.snapshot_world();
    let mut b = PhysicsWorld::from_world_snapshot(&blob).unwrap();
    assert!(b.bodies.is_empty());
    assert_eq!(b.snapshot_world(), blob);
    b.step_n(3, dt());
    assert_eq!(b.snapshot_world().len(), blob.len());
}

/// A non-default config is part of the snapshot, not taken from the target.
#[test]
fn config_comes_from_the_snapshot() {
    let cfg = PhysicsConfig {
        substeps: 3,
        gravity: Vec3Fix::from_int(0, -3, 1),
        ..PhysicsConfig::default()
    };
    let mut a = PhysicsWorld::new(cfg);
    a.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let b = PhysicsWorld::from_world_snapshot(&a.snapshot_world()).unwrap();
    assert_eq!(b.config, cfg);
}

// ── degenerate input: each has one expected error, and the target is untouched ──

fn blob() -> Vec<u8> {
    let mut a = scene();
    a.step_n(2, dt());
    a.snapshot_world()
}

/// Recompute the trailing FNV-1a 64 so only the targeted check can fire.
fn reseal(mut b: Vec<u8>) -> Vec<u8> {
    let n = b.len() - 8;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in &b[..n] {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

fn restore_err(data: &[u8]) -> WorldSnapshotError {
    let mut t = scene();
    let before = t.snapshot_world();
    let e = t.restore_world(data).unwrap_err();
    assert_eq!(
        t.snapshot_world(),
        before,
        "rejected restore must not touch the target"
    );
    e
}

#[test]
fn every_truncation_is_rejected_as_truncated() {
    let b = blob();
    for len in 0..b.len() {
        assert_eq!(
            restore_err(&b[..len]),
            WorldSnapshotError::Truncated,
            "prefix of {len} bytes"
        );
    }
}

#[test]
fn header_errors() {
    let b = blob();
    let mut x = b.clone();
    x[0] = b'X';
    assert_eq!(restore_err(&x), WorldSnapshotError::BadMagic);

    // the old rollback blob is not a world snapshot
    assert_eq!(
        restore_err(&scene().serialize_state()),
        WorldSnapshotError::BadMagic
    );

    let mut x = b.clone();
    x[4..6].copy_from_slice(&4u16.to_le_bytes());
    assert_eq!(
        restore_err(&reseal(x)),
        WorldSnapshotError::UnsupportedVersion {
            found: 4,
            supported: PhysicsWorld::WORLD_SNAPSHOT_VERSION
        }
    );

    let mut x = b.clone();
    x[6] = 1;
    assert_eq!(restore_err(&reseal(x)), WorldSnapshotError::ReservedNotZero);

    let mut x = b.clone();
    x.push(0);
    assert_eq!(
        restore_err(&x),
        WorldSnapshotError::TrailingBytes { extra: 1 }
    );
}

/// Flipping any one payload or checksum byte is caught by the checksum.
#[test]
fn every_single_byte_flip_after_the_header_is_a_checksum_mismatch() {
    let b = blob();
    for i in 16..b.len() {
        let mut x = b.clone();
        x[i] ^= 0x01;
        assert!(
            matches!(restore_err(&x), WorldSnapshotError::ChecksumMismatch { .. }),
            "byte {i}"
        );
    }
}

/// A joint that names a body the snapshot does not hold.
#[test]
fn joint_referring_past_the_bodies_is_rejected() {
    let mut a = scene();
    a.bodies.truncate(2); // joint 1 joins bodies 1 and 2
    let e = restore_err(&a.snapshot_world());
    assert_eq!(
        e,
        WorldSnapshotError::DanglingIndex {
            section: "joints",
            index: 2,
            len: 2
        }
    );
}

/// A resealed payload with an unknown enum tag is an invalid value, not a
/// panic: the solver backend byte is the last byte of the config section.
#[test]
fn unknown_tag_is_an_invalid_value() {
    let b = blob();
    // header 16 + substeps 8 + iterations 8 + gravity 48 + damping 16 + warm start 16
    let backend = 16 + 8 + 8 + 48 + 16 + 16;
    assert_eq!(b[backend], 0, "Xpbd");
    let mut x = b.clone();
    x[backend] = 9;
    assert_eq!(
        restore_err(&reseal(x)),
        WorldSnapshotError::InvalidValue { section: "config" }
    );
}

/// A body count larger than the rest of the blob can hold is truncation,
/// rejected before anything is allocated for it.
#[test]
fn huge_count_is_truncated_not_an_allocation() {
    let b = blob();
    let bodies = 16 + 8 + 8 + 48 + 16 + 16 + 1;
    let mut x = b.clone();
    x[bodies..bodies + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(restore_err(&reseal(x)), WorldSnapshotError::Truncated);
}

#[test]
fn errors_display() {
    let e = WorldSnapshotError::DanglingIndex {
        section: "joints",
        index: 2,
        len: 2,
    };
    assert_eq!(
        e.to_string(),
        "world snapshot joints refers to index 2 of 2"
    );
}
