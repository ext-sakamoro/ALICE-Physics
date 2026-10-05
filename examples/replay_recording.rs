//! Production entry point for `alice_physics::replay::{ReplayRecorder,
//! ReplayPlayer}` (`std,replay` feature — `wiring_guard`: `src/replay.rs`
//! had zero production callers for `close` (both types), `frame_count`,
//! `get_position`, `get_velocity`, `record_frame`, `record_positions` and
//! `scan_positions`; `ReplayRecorder::flush` and `ReplayPlayer::body_count`
//! are reached below as well).
//!
//! Two bodies are driven through four frames with hand-assigned, exactly
//! `f32`-representable integer positions and velocities (no
//! `PhysicsWorld::step`, so what is recorded is exactly what this file
//! wrote — the oracle values below are typed literals built by a plain
//! `as` cast, never a value read back from `to_f32()`), recorded with
//! `record_frame` (`FULL_COMPONENTS = 6`), then read back with
//! `get_position`, `get_velocity` and `scan_positions` and checked for
//! **exact** equality. `src/replay.rs`'s own module doc documents the
//! round trip as exact: the recorder's `open_lossless` sets
//! `FitConfig::lossless = true`, which (as of `alice-db` 0.2.0-beta.3)
//! stores a `ResidualKind::Xor` residual —
//! `f32::from_bits(model.to_bits() ^ residual) == original` for every bit
//! pattern — not the pre-2026-09-17 additive residual that rounds.
//!
//! A second, smaller recording demonstrates `record_positions`
//! (`POSITION_COMPONENTS = 3`, no velocity channel) and both `close`
//! overloads (`ReplayRecorder::close(self)`, `ReplayPlayer::close(self)`).
//!
//! ```bash
//! cargo run --example replay_recording --features std,replay
//! ```

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

/// Body 0's hand-assigned position at frame `k` (`k` in `0..FRAMES`).
fn pos0(k: i64) -> (f32, f32, f32) {
    ((10 + k) as f32, (20 + 2 * k) as f32, (-(5 + k)) as f32)
}

/// Body 0's hand-assigned velocity at frame `k`.
fn vel0(k: i64) -> (f32, f32, f32) {
    ((100 - 10 * k) as f32, 0.0, (7 + k) as f32)
}

/// Body 1's hand-assigned position at frame `k`.
fn pos1(k: i64) -> (f32, f32, f32) {
    ((-(50 + k)) as f32, 0.0, (30 - k) as f32)
}

/// Body 1's hand-assigned velocity at frame `k`.
fn vel1(k: i64) -> (f32, f32, f32) {
    (0.0, (-(3 + k)) as f32, (2 * k) as f32)
}

const FRAMES: i64 = 4;

/// Total size of the regular files under `path` (recursively).
fn dir_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(path).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let meta = entry.metadata().expect("metadata");
        total += if meta.is_dir() {
            dir_bytes(&entry.path())
        } else {
            meta.len()
        };
    }
    total
}

fn main() {
    println!("[replay] hand-built 2-body / {FRAMES}-frame recording via record_frame");

    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));

    let dir = tempfile::tempdir().expect("tempdir");
    let full_path = dir.path().join("full");
    let mut recorder = ReplayRecorder::new(&full_path, 2).expect("ReplayRecorder::new");

    for k in 0..FRAMES {
        world.bodies[0].position = Vec3Fix::from_int(10 + k, 20 + 2 * k, -(5 + k));
        world.bodies[0].velocity = Vec3Fix::from_int(100 - 10 * k, 0, 7 + k);
        world.bodies[1].position = Vec3Fix::from_int(-(50 + k), 0, 30 - k);
        world.bodies[1].velocity = Vec3Fix::from_int(0, -(3 + k), 2 * k);
        recorder.record_frame(&world).expect("record_frame");
        println!(
            "[replay] frame {k}: body0 pos={:?} vel={:?} body1 pos={:?} vel={:?}",
            pos0(k),
            vel0(k),
            pos1(k),
            vel1(k)
        );
    }

    let count = recorder.frame_count();
    println!("[replay] frame_count = {count} (expected {FRAMES})");
    assert_eq!(
        count, FRAMES as u64,
        "frame_count must equal calls to record_frame"
    );
    // `flush` pushes the frames buffered in memory to disk while the
    // recorder stays open (a second handle cannot open the store meanwhile:
    // the writer holds its lock). What it is observable by is the bytes on
    // disk: the store grows on the flush.
    let before = dir_bytes(&full_path);
    recorder.flush().expect("ReplayRecorder::flush");
    let after = dir_bytes(&full_path);
    println!("[replay] bytes on disk before / after flush: {before} / {after}");
    assert!(
        after > before,
        "flush must write the buffered frames ({before} -> {after})"
    );
    recorder.close().expect("ReplayRecorder::close");

    let player = ReplayPlayer::open(&full_path, 2).expect("ReplayPlayer::open");
    // `body_count` is the count the player was opened with (it is not read
    // from the recording).
    assert_eq!(player.body_count(), 2);
    for k in 0..FRAMES {
        let frame = k as u64;
        let got0 = player
            .get_position(frame, 0)
            .expect("get_position io")
            .expect("recorded");
        let gotv0 = player
            .get_velocity(frame, 0)
            .expect("get_velocity io")
            .expect("recorded");
        let got1 = player
            .get_position(frame, 1)
            .expect("get_position io")
            .expect("recorded");
        let gotv1 = player
            .get_velocity(frame, 1)
            .expect("get_velocity io")
            .expect("recorded");
        println!(
            "[replay] playback frame {k}: body0 pos={got0:?} vel={gotv0:?} body1 pos={got1:?} vel={gotv1:?}"
        );
        assert_eq!(
            got0,
            pos0(k),
            "frame {k} body0 position must round-trip exactly"
        );
        assert_eq!(
            gotv0,
            vel0(k),
            "frame {k} body0 velocity must round-trip exactly"
        );
        assert_eq!(
            got1,
            pos1(k),
            "frame {k} body1 position must round-trip exactly"
        );
        assert_eq!(
            gotv1,
            vel1(k),
            "frame {k} body1 velocity must round-trip exactly"
        );
    }

    let scan0 = player
        .scan_positions(0, 0, (FRAMES - 1) as u64)
        .expect("scan_positions body 0");
    println!(
        "[replay] scan_positions(body=0, 0..={}) -> {scan0:?}",
        FRAMES - 1
    );
    assert_eq!(
        scan0.len(),
        FRAMES as usize,
        "scan must return every recorded frame"
    );
    for (frame, x, y, z) in &scan0 {
        let (ex, ey, ez) = pos0(*frame as i64);
        assert_eq!(
            (*x, *y, *z),
            (ex, ey, ez),
            "scan frame {frame} must match pos0 exactly"
        );
    }
    let scan1 = player
        .scan_positions(1, 0, (FRAMES - 1) as u64)
        .expect("scan_positions body 1");
    for (frame, x, y, z) in &scan1 {
        let (ex, ey, ez) = pos1(*frame as i64);
        assert_eq!(
            (*x, *y, *z),
            (ex, ey, ez),
            "scan frame {frame} must match pos1 exactly (channel isolation)"
        );
    }

    player.close().expect("ReplayPlayer::close");

    // --- record_positions: position-only layout (3 components, no velocity) ---
    println!("[replay] record_positions: 1-body / 3-frame position-only recording");
    let pos_path = dir.path().join("positions_only");
    let mut pos_world = PhysicsWorld::new(config);
    pos_world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let mut pos_recorder = ReplayRecorder::new(&pos_path, 1).expect("ReplayRecorder::new");
    for k in 0..3i64 {
        pos_world.bodies[0].position = Vec3Fix::from_int(k, 2 * k, -k);
        pos_recorder
            .record_positions(&pos_world)
            .expect("record_positions");
    }
    assert_eq!(pos_recorder.frame_count(), 3);
    pos_recorder.close().expect("ReplayRecorder::close");

    let pos_player = ReplayPlayer::open(&pos_path, 1).expect("ReplayPlayer::open");
    for k in 0..3u64 {
        let got = pos_player
            .get_position(k, 0)
            .expect("io")
            .expect("recorded");
        let expect = (k as f32, 2.0 * k as f32, -(k as f32));
        println!("[replay] positions-only frame {k}: pos={got:?}");
        assert_eq!(
            got, expect,
            "position-only frame {k} must round-trip exactly"
        );
        let vel = pos_player.get_velocity(k, 0).expect("io");
        assert!(
            vel.is_none(),
            "record_positions must leave no velocity channel to read back"
        );
    }
    pos_player.close().expect("ReplayPlayer::close");

    println!("[replay] done");
}
