//! Audit oracles for replay
//!
//! 期待値は書き込んだ値そのもの (Fix128 -> f32 変換は `to_f32` の定義) と、
//! 鍵配置 `frame * channels + body * components + component` の手計算

#![cfg(all(feature = "std", feature = "replay"))]

use alice_physics::replay::{ReplayPlayer, ReplayRecorder};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

/// xorshift (決定論、外部 crate なし)
fn next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn rnd_f32(state: &mut u64) -> f32 {
    // [-512, 512) の非 dyadic な値 (1/1000 刻み)
    (next(state) % 1_024_000) as f32 / 1000.0 - 512.0
}

fn world_with(n: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    for _ in 0..n {
        w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    }
    w
}

/// 各 body の位置 / 速度を frame ごとに書き換えて記録し、書いた f32 の列を返す
#[allow(clippy::type_complexity)]
fn record_random(
    path: &std::path::Path,
    bodies: usize,
    frames: usize,
    seed: u64,
    full: bool,
    flush_every: usize,
) -> Vec<Vec<([f32; 3], [f32; 3])>> {
    let mut world = world_with(bodies);
    let mut rec = ReplayRecorder::new(path, bodies).unwrap();
    let mut st = seed;
    let mut want = Vec::new();
    for f in 0..frames {
        let mut row = Vec::new();
        for b in 0..bodies {
            let p = [rnd_f32(&mut st), rnd_f32(&mut st), rnd_f32(&mut st)];
            let v = [rnd_f32(&mut st), rnd_f32(&mut st), rnd_f32(&mut st)];
            world.bodies[b].position = Vec3Fix::from_f32(p[0], p[1], p[2]);
            world.bodies[b].velocity = Vec3Fix::from_f32(v[0], v[1], v[2]);
            // 実際に記録される値は Fix128 -> f32 変換後
            let (px, py, pz) = world.bodies[b].position.to_f32();
            let (vx, vy, vz) = world.bodies[b].velocity.to_f32();
            row.push(([px, py, pz], [vx, vy, vz]));
        }
        want.push(row);
        if full {
            rec.record_frame(&world).unwrap();
        } else {
            rec.record_positions(&world).unwrap();
        }
        if flush_every > 0 && (f + 1) % flush_every == 0 {
            rec.flush().unwrap();
        }
    }
    assert_eq!(rec.frame_count(), frames as u64);
    rec.close().unwrap();
    want
}

#[test]
fn random_trajectory_round_trips_bit_exact_through_flush_boundaries() {
    // doc: "a replay of a deterministic engine must read back the bits it wrote" (lossless)
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let want = record_random(&path, 3, 120, 0x9E37_79B9_7F4A_7C15, true, 17);
    let player = ReplayPlayer::open(&path, 3).unwrap();
    assert_eq!(player.body_count(), 3);
    for (f, row) in want.iter().enumerate() {
        for (b, (p, v)) in row.iter().enumerate() {
            let gp = player.get_position(f as u64, b).unwrap().expect("pos");
            let gv = player.get_velocity(f as u64, b).unwrap().expect("vel");
            assert_eq!(
                [gp.0.to_bits(), gp.1.to_bits(), gp.2.to_bits()],
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                "pos f{f} b{b}"
            );
            assert_eq!(
                [gv.0.to_bits(), gv.1.to_bits(), gv.2.to_bits()],
                [v[0].to_bits(), v[1].to_bits(), v[2].to_bits()],
                "vel f{f} b{b}"
            );
        }
    }
    player.close().unwrap();
}

#[test]
fn positions_only_recording_round_trips_and_has_no_velocity() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let want = record_random(&path, 2, 60, 12345, false, 9);
    let player = ReplayPlayer::open(&path, 2).unwrap();
    for (f, row) in want.iter().enumerate() {
        for (b, (p, _)) in row.iter().enumerate() {
            let gp = player.get_position(f as u64, b).unwrap().expect("pos");
            assert_eq!(
                [gp.0.to_bits(), gp.1.to_bits(), gp.2.to_bits()],
                [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                "pos f{f} b{b}"
            );
            assert!(player.get_velocity(f as u64, b).unwrap().is_none());
        }
    }
}

#[test]
fn scan_positions_equals_per_frame_get_for_every_body_and_subrange() {
    for full in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r");
        let want = record_random(&path, 4, 50, 777 + u64::from(full), full, 0);
        let player = ReplayPlayer::open(&path, 4).unwrap();
        for b in 0..4 {
            for (s, e) in [(0u64, 49u64), (7, 7), (10, 31), (49, 49), (0, 0)] {
                let rows = player.scan_positions(b, s, e).unwrap();
                assert_eq!(rows.len() as u64, e - s + 1, "full={full} b{b} {s}..={e}");
                for (i, (f, x, y, z)) in rows.iter().enumerate() {
                    assert_eq!(*f, s + i as u64);
                    let p = want[*f as usize][b].0;
                    assert_eq!(
                        [x.to_bits(), y.to_bits(), z.to_bits()],
                        [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()],
                        "full={full} b{b} f{f}"
                    );
                }
            }
        }
    }
}

#[test]
fn scan_positions_end_frame_beyond_recorded_returns_only_recorded_frames() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let _ = record_random(&path, 2, 10, 5, true, 0);
    let player = ReplayPlayer::open(&path, 2).unwrap();
    let rows = player.scan_positions(1, 5, 1000).unwrap();
    assert_eq!(rows.len(), 5, "frames 5..=9 only: {}", rows.len());
    assert_eq!(rows.last().unwrap().0, 9);
}

#[test]
fn bodies_missing_from_world_are_recorded_as_zero_and_extra_bodies_are_ignored() {
    // body_count = 3 で world は 2 body: 3 番目は 0.0 の位置 / 速度、density なキー列は保たれる
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let mut world = world_with(2);
    world.bodies[0].position = Vec3Fix::from_int(1, 2, 3);
    world.bodies[1].position = Vec3Fix::from_int(4, 5, 6);
    world.bodies[1].velocity = Vec3Fix::from_int(7, 8, 9);
    let mut rec = ReplayRecorder::new(&path, 3).unwrap();
    for _ in 0..4 {
        rec.record_frame(&world).unwrap();
    }
    rec.close().unwrap();
    let player = ReplayPlayer::open(&path, 3).unwrap();
    for f in 0..4u64 {
        assert_eq!(player.get_position(f, 0).unwrap(), Some((1.0, 2.0, 3.0)));
        assert_eq!(player.get_position(f, 1).unwrap(), Some((4.0, 5.0, 6.0)));
        assert_eq!(player.get_velocity(f, 1).unwrap(), Some((7.0, 8.0, 9.0)));
        assert_eq!(player.get_position(f, 2).unwrap(), Some((0.0, 0.0, 0.0)));
        assert_eq!(player.get_velocity(f, 2).unwrap(), Some((0.0, 0.0, 0.0)));
    }
    // 記録した body 数の外は None
    assert_eq!(player.get_position(0, 3).unwrap(), None);
}

#[test]
fn layout_file_records_body_count_and_components() {
    // doc: "The recorder writes the layout (body_count, components) to <path>/replay_layout"
    let dir = tempfile::tempdir().unwrap();
    let p6 = dir.path().join("a");
    let p3 = dir.path().join("b");
    let w = world_with(5);
    let mut r = ReplayRecorder::new(&p6, 5).unwrap();
    r.record_frame(&w).unwrap();
    r.close().unwrap();
    let mut r = ReplayRecorder::new(&p3, 5).unwrap();
    r.record_positions(&w).unwrap();
    r.close().unwrap();
    assert_eq!(
        std::fs::read_to_string(p6.join("replay_layout"))
            .unwrap()
            .trim(),
        "5 6"
    );
    assert_eq!(
        std::fs::read_to_string(p3.join("replay_layout"))
            .unwrap()
            .trim(),
        "5 3"
    );
}

#[test]
fn malformed_layout_file_is_rejected_with_invalid_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let w = world_with(1);
    let mut r = ReplayRecorder::new(&path, 1).unwrap();
    r.record_frame(&w).unwrap();
    r.close().unwrap();
    for bad in [
        "",
        "1",
        "x y",
        "1 4",
        "1 6 extra-ok-ignored-but-first-two-valid",
    ] {
        std::fs::write(path.join("replay_layout"), bad).unwrap();
        let res = ReplayPlayer::open(&path, 1);
        if bad.starts_with("1 6") {
            assert!(res.is_ok(), "two valid leading tokens are accepted");
        } else {
            let e = res
                .err()
                .unwrap_or_else(|| panic!("{bad:?} must be rejected"));
            assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{bad:?}");
        }
    }
}

#[test]
fn re_recording_into_an_existing_replay_does_not_leave_stale_tail_frames() {
    // 10 frame の recording の上に同じ dir で 3 frame を録り直す: frame 3.. は「録っていない」ので None のはず
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let _ = record_random(&path, 2, 10, 1, true, 0);
    let _ = record_random(&path, 2, 3, 2, true, 0);
    let player = ReplayPlayer::open(&path, 2).unwrap();
    assert!(player.get_position(2, 0).unwrap().is_some());
    assert_eq!(
        player.get_position(5, 0).unwrap(),
        None,
        "frame 5 of the first recording is served as if it belonged to the second"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-009: ReplayPlayer::open は存在しない dir を作成し (alice-db の open が create)、存在しない replay でも Ok(player) を返して以後の get が全て None になる"]
fn opening_a_player_on_a_missing_directory_does_not_create_it() {
    // 読み取り専用の open は副作用を持たないはず
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("does_not_exist");
    let _ = ReplayPlayer::open(&path, 1);
    assert!(!path.exists(), "ReplayPlayer::open created {path:?}");
}

#[test]
fn missing_layout_file_falls_back_to_six_components_and_the_callers_body_count() {
    // ReplayPlayer の field doc: "6 when the file is absent"
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("r");
    let want = record_random(&path, 2, 20, 99, true, 0);
    std::fs::remove_file(path.join("replay_layout")).unwrap();
    let player = ReplayPlayer::open(&path, 2).unwrap();
    assert_eq!(player.body_count(), 2);
    for (f, row) in want.iter().enumerate() {
        for (b, (p, v)) in row.iter().enumerate() {
            let gp = player.get_position(f as u64, b).unwrap().unwrap();
            let gv = player.get_velocity(f as u64, b).unwrap().unwrap();
            assert_eq!(gp.0.to_bits(), p[0].to_bits());
            assert_eq!(gv.2.to_bits(), v[2].to_bits());
        }
    }
}
