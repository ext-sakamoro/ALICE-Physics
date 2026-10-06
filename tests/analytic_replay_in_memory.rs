//! In-memory replay: a recording kept in process memory, taken out as bytes
//! and played back from those bytes, reads back the exact bits the world held
//! at every recorded frame. The expected values are captured from the world
//! itself while recording, independently of the replay code.

#![cfg(feature = "replay")]

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const BODIES: usize = 3;
const FRAMES: u64 = 120;

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

type Bits = [u32; 3];

fn bits((x, y, z): (f32, f32, f32)) -> Bits {
    [x.to_bits(), y.to_bits(), z.to_bits()]
}

/// Record `FRAMES` frames into `recorder`, returning (position, velocity)
/// bits per frame and body as read straight from the world.
fn record(recorder: &mut ReplayRecorder, positions_only: bool) -> Vec<Vec<(Bits, Bits)>> {
    let mut w = world();
    let dt = Fix128::from_ratio(1, 60);
    let mut expected = Vec::new();
    for _ in 0..FRAMES {
        w.step(dt);
        if positions_only {
            recorder.record_positions(&w).unwrap();
        } else {
            recorder.record_frame(&w).unwrap();
        }
        expected.push(
            w.bodies
                .iter()
                .map(|b| (bits(b.position.to_f32()), bits(b.velocity.to_f32())))
                .collect(),
        );
    }
    expected
}

fn compare(player: &ReplayPlayer, expected: &[Vec<(Bits, Bits)>], velocities: bool) -> usize {
    let mut compared = 0;
    for (frame, bodies) in expected.iter().enumerate() {
        for (body, (pos, vel)) in bodies.iter().enumerate() {
            let got = player.get_position(frame as u64, body).unwrap().unwrap();
            assert_eq!(bits(got), *pos, "position frame {frame} body {body}");
            compared += 1;
            if velocities {
                let got = player.get_velocity(frame as u64, body).unwrap().unwrap();
                assert_eq!(bits(got), *vel, "velocity frame {frame} body {body}");
                compared += 1;
            }
        }
    }
    compared
}

#[test]
fn memory_recording_round_trips_through_bytes_bit_for_bit() {
    let mut rec = ReplayRecorder::in_memory(BODIES).unwrap();
    let expected = record(&mut rec, false);
    let bytes = rec.to_bytes().unwrap();
    let player = ReplayPlayer::from_bytes(&bytes, BODIES).unwrap();
    let compared = compare(&player, &expected, true);
    assert_eq!(compared, BODIES * FRAMES as usize * 2);
}

#[test]
fn the_layout_travels_with_the_bytes() {
    // positions-only recording, played back with a wrong body count: the
    // layout stored in the recording wins
    let mut rec = ReplayRecorder::in_memory(BODIES).unwrap();
    let expected = record(&mut rec, true);
    let bytes = rec.to_bytes().unwrap();
    let player = ReplayPlayer::from_bytes(&bytes, 99).unwrap();
    assert_eq!(player.body_count(), BODIES);
    let compared = compare(&player, &expected, false);
    assert_eq!(compared, BODIES * FRAMES as usize);
}

#[test]
fn scan_matches_point_reads_after_the_round_trip() {
    let mut rec = ReplayRecorder::in_memory(BODIES).unwrap();
    let expected = record(&mut rec, false);
    let player = ReplayPlayer::from_bytes(&rec.to_bytes().unwrap(), BODIES).unwrap();
    // `body` is the argument of `scan_positions` and the inner index of
    // `expected[frame]`, not an iteration over one slice
    #[allow(clippy::needless_range_loop)]
    for body in 0..BODIES {
        let scan = player.scan_positions(body, 0, FRAMES - 1).unwrap();
        assert_eq!(scan.len(), FRAMES as usize, "body {body}");
        for (frame, x, y, z) in scan {
            assert_eq!(
                bits((x, y, z)),
                expected[frame as usize][body].0,
                "frame {frame} body {body}"
            );
        }
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[test]
fn a_file_recording_exports_the_same_bytes_view_as_its_files() {
    let dir = tempfile::tempdir().unwrap();
    let mut rec = ReplayRecorder::new(dir.path().join("r"), BODIES).unwrap();
    let expected = record(&mut rec, false);
    let bytes = rec.to_bytes().unwrap();
    rec.close().unwrap();
    let from_files = ReplayPlayer::open(dir.path().join("r"), 0).unwrap();
    let from_bytes = ReplayPlayer::from_bytes(&bytes, 0).unwrap();
    assert_eq!(
        compare(&from_files, &expected, true),
        BODIES * FRAMES as usize * 2
    );
    assert_eq!(
        compare(&from_bytes, &expected, true),
        BODIES * FRAMES as usize * 2
    );
}

#[test]
fn damaged_bytes_are_rejected() {
    let mut rec = ReplayRecorder::in_memory(BODIES).unwrap();
    record(&mut rec, false);
    let mut bytes = rec.to_bytes().unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x40;
    assert!(ReplayPlayer::from_bytes(&bytes, BODIES).is_err());
    bytes.truncate(bytes.len() - 1);
    assert!(ReplayPlayer::from_bytes(&bytes, BODIES).is_err());
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[test]
fn re_recording_replaces_only_the_recording_files() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let mut first = ReplayRecorder::new(&path, BODIES).unwrap();
    record(&mut first, false);
    first.close().unwrap();
    std::fs::write(path.join("notes.txt"), b"keep me").unwrap();

    let mut second = ReplayRecorder::new(&path, BODIES).unwrap();
    let mut w = world();
    for _ in 0..3 {
        w.step(Fix128::from_ratio(1, 60));
        second.record_frame(&w).unwrap();
    }
    second.close().unwrap();

    assert_eq!(std::fs::read(path.join("notes.txt")).unwrap(), b"keep me");
    let player = ReplayPlayer::open(&path, BODIES).unwrap();
    assert!(player.get_position(2, 0).unwrap().is_some());
    assert_eq!(
        player.get_position(3, 0).unwrap(),
        None,
        "frame of the old recording shows through"
    );
}
