//! A replay recording written by an earlier `alice-db` release stays readable.
//!
//! `ReplayRecorder::to_bytes` hands out the bytes of the `alice-db` store that
//! holds the recording, so a recording kept by a caller is stored data in the
//! format of the `alice-db` release this crate was built with. The fixture
//! pair in `tests/fixtures/` was written once by this crate built against
//! `alice-db` 0.4 (`write_the_fixture_pair`, ignored, run by hand):
//!
//! * `replay_alice_db_0_4.bin`: the recording, `to_bytes()` after 120 frames
//!   of positions and velocities of 3 bodies;
//! * `replay_alice_db_0_4.bits`: what the world held at each of those frames,
//!   as `f32` bit patterns (little endian `u32`, frame by frame, body by body,
//!   position x y z then velocity x y z), read from the world while recording,
//!   not through the replay code.
//!
//! The test reads the recording with the current build and compares every
//! value with the table, so a change of `alice-db` that cannot read the
//! earlier format, or reads it to other values, fails here. The table does not
//! depend on the physics either: it is the recorded truth, not a re-run.

#![cfg(feature = "replay")]

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const BODIES: usize = 3;
const FRAMES: u64 = 120;
/// Values per frame and body: position x y z, velocity x y z.
const PER_BODY: usize = 6;

const RECORDING: &[u8] = include_bytes!("fixtures/replay_alice_db_0_4.bin");
const TABLE: &[u8] = include_bytes!("fixtures/replay_alice_db_0_4.bits");

fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 40, 0),
        Fix128::ONE,
    ));
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 25, -2),
        Fix128::from_int(2),
    ));
    let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(-4, 10, 1), Fix128::ONE);
    b.velocity = Vec3Fix::from_int(2, 7, -1);
    w.add_body(b);
    w
}

fn table_value(frame: u64, body: usize, k: usize) -> u32 {
    let i = ((frame as usize * BODIES + body) * PER_BODY + k) * 4;
    u32::from_le_bytes([TABLE[i], TABLE[i + 1], TABLE[i + 2], TABLE[i + 3]])
}

/// oracle (stored data): the recording written with `alice-db` 0.4 reads back
/// to exactly the recorded values, position and velocity of every body at
/// every frame (720 comparisons each).
#[test]
fn a_recording_written_with_alice_db_0_4_reads_back_bit_for_bit() {
    assert_eq!(TABLE.len(), FRAMES as usize * BODIES * PER_BODY * 4);
    let player = ReplayPlayer::from_bytes(RECORDING, BODIES).expect("the 0.4 recording opens");
    assert_eq!(player.body_count(), BODIES);
    let mut compared = 0;
    for frame in 0..FRAMES {
        for body in 0..BODIES {
            let (px, py, pz) = player
                .get_position(frame, body)
                .expect("read")
                .expect("position present");
            let (vx, vy, vz) = player
                .get_velocity(frame, body)
                .expect("read")
                .expect("velocity present");
            for (k, got) in [px, py, pz, vx, vy, vz].into_iter().enumerate() {
                assert_eq!(
                    got.to_bits(),
                    table_value(frame, body, k),
                    "frame {frame} body {body} value {k}"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, FRAMES as usize * BODIES * PER_BODY);
}

/// Writes the fixture pair (run once, by hand, against the `alice-db` release
/// the pair is named after): `ALICE_PHYSICS_WRITE_FIXTURE=1 cargo test
/// --features replay --test replay_alice_db_format -- --ignored`.
#[test]
#[ignore = "writes tests/fixtures/replay_alice_db_0_4.{bin,bits}; run by hand"]
fn write_the_fixture_pair() {
    if std::env::var_os("ALICE_PHYSICS_WRITE_FIXTURE").is_none() {
        return;
    }
    let mut rec = ReplayRecorder::in_memory(BODIES).expect("recorder");
    let mut w = world();
    let dt = Fix128::from_ratio(1, 60);
    let mut table = Vec::new();
    for _ in 0..FRAMES {
        w.step(dt);
        rec.record_frame(&w).expect("record");
        for b in &w.bodies {
            let (px, py, pz) = b.position.to_f32();
            let (vx, vy, vz) = b.velocity.to_f32();
            for v in [px, py, pz, vx, vy, vz] {
                table.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    std::fs::write(
        dir.join("replay_alice_db_0_4.bin"),
        rec.to_bytes().expect("bytes"),
    )
    .expect("write");
    std::fs::write(dir.join("replay_alice_db_0_4.bits"), table).expect("write");
}
