//! The reader of this build takes the payload of a real snapshot of main before format 8: a
//! version 7 blob (a body with an axis lock) and a version 6 blob (no lock), both written by main
//! `492ed114` (semantics `c82742de…`).
//!
//! The fixtures were written outside this tree by a crate depending on `alice-physics-core` at
//! `492ed114018ea151f9d9d58964a1b5f406db1d7a` (path dependency, `default-features = false,
//! features = ["std"]`), `cargo run --release -- <dir>`: `scene(lock)` below (a kinematic body
//! at the origin, a falling body of mass 1 at (0, 5, 0), a body of mass 2 at (2, 7, 1) moving
//! (1, 0, −1), locked along `LINEAR_Z` in the locked scene) stepped 30 times by 1/60 s and
//! written by `snapshot_world`. The writer printed
//!
//! - locked: version 7, 2147 bytes, sha256 `cecd4718…`, law_id `478fc75b…`
//! - unlocked: version 6, 2136 bytes, sha256 `41ff3154…`, law_id `00c3ebd9…`
//!
//! and the position, velocity and lock of bodies 1 and 2 (the constants below, Fix128 as
//! `(hi, lo)`).
//!
//! A restore checks the header's semantics identifier before it reads the payload, so a blob of
//! another stepping semantics is refused before its payload is read (that refusal is pinned in
//! `world_snapshot_xv_*.rs`). To reach the payload reader the tests here put this build's
//! identifier in the header and reseal the checksum, and read the step-count sleep rule with the
//! writer's step (`restore_world_legacy_damping`, 1/60 s). The restored bodies must be what the
//! writer printed, and the lock read from the version 7 section must be the writer's.
//!
//! oracle: the body state and the lock are compared with what the writing build printed.

use alice_physics_core::axis_lock::AxisLock;
use alice_physics_core::{
    Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix, WorldSnapshotError, PHYSICS_SEMANTICS_ID,
};
use sha2::{Digest, Sha256};

const LOCKED: &[u8] = include_bytes!("fixtures/world_snapshot_main_v7_locked.bin");
const UNLOCKED: &[u8] = include_bytes!("fixtures/world_snapshot_main_v6_unlocked.bin");
const VERSION_AT: usize = 4;
const SEMANTICS_AT: usize = 16;
const LAW_AT: usize = 48;
const CHECKSUM: usize = 8;
/// What the build at `492ed114` printed
const WRITER_SEMANTICS: &str = "c82742de000629b51a5fa1717bb68620a683460617a89dc009c2374e7e9a470f";
const LOCKED_SHA256: &str = "cecd47180b88755248f1dc634d3cfeb8d553996d5c97c0582b1d440545636252";
const UNLOCKED_SHA256: &str = "41ff3154d3167f1d12fd4ca95dbc31789a8d1c5db706467a2f459cd8737709ee";
const LOCKED_LAW_ID: &str = "478fc75b2ca6be30cad067dcaf25b48c387dcd2efd2c3d1e3e2a189ae147d58d";
const UNLOCKED_LAW_ID: &str = "00c3ebd997a6b94884aef54907773130e1c7859eb198cd287a299a3222f70642";

fn fix(hi: i64, lo: u64) -> Fix128 {
    Fix128 { hi, lo }
}

/// Body 1 in both scenes: position, velocity
fn body1() -> (Vec3Fix, Vec3Fix) {
    (
        Vec3Fix::new(fix(0, 0), fix(3, 13_738_981_263_231_593_651), fix(0, 0)),
        Vec3Fix::new(fix(0, 0), fix(-5, 1280), fix(0, 0)),
    )
}

/// Body 2: position, velocity, lock; the locked scene keeps z = 1 and v_z = 0
fn body2(locked: bool) -> (Vec3Fix, Vec3Fix, AxisLock) {
    let x = fix(2, 9_223_372_036_854_775_680);
    let y = fix(5, 13_738_981_263_231_593_651);
    if locked {
        (
            Vec3Fix::new(x, y, fix(1, 0)),
            Vec3Fix::new(fix(1, 0), fix(-5, 1280), fix(0, 0)),
            AxisLock::LINEAR_Z,
        )
    } else {
        (
            Vec3Fix::new(x, y, fix(0, 9_223_372_036_854_775_936)),
            Vec3Fix::new(fix(1, 0), fix(-5, 1280), fix(-1, 0)),
            AxisLock::NONE,
        )
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in bytes {
        h ^= u64::from(x);
        h = h.wrapping_mul(0x0000_0001_0000_01b3);
    }
    h
}

/// The blob with this build's semantics identifier (and, when given, another version) in its
/// header and the checksum recomputed
fn restamped(blob: &[u8], version: Option<u16>) -> Vec<u8> {
    let mut b = blob.to_vec();
    b[SEMANTICS_AT..SEMANTICS_AT + 32].copy_from_slice(&PHYSICS_SEMANTICS_ID);
    if let Some(v) = version {
        b[VERSION_AT..VERSION_AT + 2].copy_from_slice(&v.to_le_bytes());
    }
    let n = b.len() - CHECKSUM;
    let h = fnv1a(&b[..n]);
    b[n..].copy_from_slice(&h.to_le_bytes());
    b
}

fn read(blob: &[u8]) -> Result<PhysicsWorld, WorldSnapshotError> {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.restore_world_legacy_damping(blob, Fix128::from_ratio(1, 60))
        .map(|_| w)
}

#[test]
fn the_fixtures_are_what_the_writer_wrote() {
    for (blob, len, version, sha, law) in [
        (LOCKED, 2147, 7u16, LOCKED_SHA256, LOCKED_LAW_ID),
        (UNLOCKED, 2136, 6, UNLOCKED_SHA256, UNLOCKED_LAW_ID),
    ] {
        assert_eq!(blob.len(), len);
        assert_eq!(hex(&Sha256::digest(blob)), sha);
        assert_eq!(
            u16::from_le_bytes([blob[VERSION_AT], blob[VERSION_AT + 1]]),
            version
        );
        assert_eq!(
            hex(&blob[SEMANTICS_AT..SEMANTICS_AT + 32]),
            WRITER_SEMANTICS
        );
        assert_eq!(hex(&blob[LAW_AT..LAW_AT + 32]), law);
    }
    assert_ne!(hex(&PHYSICS_SEMANTICS_ID), WRITER_SEMANTICS);
}

#[test]
fn the_version_7_payload_restores_with_its_axis_lock() {
    let w = read(&restamped(LOCKED, None)).expect("the version 7 payload reads");
    assert_eq!(w.bodies.len(), 3);
    let (p1, v1) = body1();
    assert_eq!((w.bodies[1].position, w.bodies[1].velocity), (p1, v1));
    let (p2, v2, lock) = body2(true);
    assert_eq!((w.bodies[2].position, w.bodies[2].velocity), (p2, v2));
    assert_eq!(w.body_axis_lock(1), Some(AxisLock::NONE));
    assert_eq!(w.body_axis_lock(2), Some(lock));
}

#[test]
fn the_version_6_payload_restores_without_locks() {
    let w = read(&restamped(UNLOCKED, None)).expect("the version 6 payload reads");
    assert_eq!(w.bodies.len(), 3);
    let (p1, v1) = body1();
    assert_eq!((w.bodies[1].position, w.bodies[1].velocity), (p1, v1));
    let (p2, v2, lock) = body2(false);
    assert_eq!((w.bodies[2].position, w.bodies[2].velocity), (p2, v2));
    for i in 0..3 {
        assert_eq!(w.body_axis_lock(i), Some(lock), "body {i}");
    }
}

/// The version decides whether the lock section is read: the version 7 payload read as version 6
/// leaves the section (11 bytes, the difference of the two fixtures) unread, and the version 6
/// payload read as version 7 runs out where the section would be
#[test]
fn the_version_decides_whether_the_lock_section_is_read() {
    assert_eq!(
        read(&restamped(LOCKED, Some(6))).err(),
        Some(WorldSnapshotError::TrailingBytes {
            extra: LOCKED.len() - UNLOCKED.len()
        })
    );
    assert_eq!(
        read(&restamped(UNLOCKED, Some(7))).err(),
        Some(WorldSnapshotError::Truncated)
    );
}
