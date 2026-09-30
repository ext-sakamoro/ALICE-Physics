//! `serialize_state` の被覆に body 外の状態が入っていないことの oracle (WM-08)
//!
//! blob は **per-body の運動状態を全部**持っている
//! (`position 48 + velocity 48 + rotation 64 + angular_velocity 48 = 208 byte`)
//! ⚠️ 欠けているのは **body 外の状態** で、その最初の 1 つが
//! `IslandManager.sleep_data` (`SleepState` + `idle_frames`)
//!
//! `deserialize_state` の末尾が `IslandManager::new(n, config)` なので、
//! **sleep 状態は復元されず 0 に戻る**
//!
//! ⚠️⚠️ **scene 選定が判定の成否を決める** — 既定設定や完全静止 scene では
//! `idle_frames` を確実に失っていても **byte 一致してしまう** (性質でなく
//! scene 依存の偶然) 本 file は
//!
//! 1. `idle_frames` が **実際に軌道へ出る**ことを対照実験で先に示す
//! 2. その上で「往復して復元されない」ことを red で pin する
//!
//! の 2 段で書いてある 1 を省くと、2 が「失っても害が無い状態」を
//! pin しているのか区別できない

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SleepConfig, Vec3Fix};

/// 微小運動が残る scene + 緩い sleep 閾値
///
/// `frames_to_sleep` を小さく取って `idle_frames` が短時間で意味を持つようにし、
/// 閾値を緩くして「動いているが idle 判定される」帯に body を置く
///
/// ⚠️ **重力をゼロにするのが要点** (2026-09-30 実測) 既定重力のままだと body が
/// 加速し続けるので 3 step で速度が閾値を超え、`idle_frames` が 1 度も溜まらない
/// (= 対照実験が成立せず、`rollback_then_continue` が「失っても byte 一致」の
/// 偽 green になる 台帳が警告している scene 依存の偶然そのもの)
/// 等速ドリフトなら idle 帯に留まり、眠った瞬間に速度の積分が止まって
/// **位置が凍る** ので軌道差として観測できる
fn drifting_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    world.set_sleep_config(SleepConfig {
        // 0.5 m/s 未満を idle とみなす = 微小運動でも idle 帯に入る
        linear_threshold: Fix128::from_ratio(1, 2),
        angular_threshold: Fix128::from_ratio(1, 2),
        // 既定 60 では 10 step で効かない 4 step で眠る設定にする
        frames_to_sleep: 4,
    });
    let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
    // 閾値 0.5 未満の微小運動 (0.25 m/s) — idle 帯だが静止していない
    body.velocity = Vec3Fix::new(Fix128::from_ratio(1, 4), Fix128::ZERO, Fix128::ZERO);
    world.add_body(body);
    world
}

fn step_n(world: &mut PhysicsWorld, n: usize) {
    let dt = Fix128::from_ratio(1, 64);
    for _ in 0..n {
        world.step(dt);
    }
}

/// ⚠️ 対照実験 (先に示す) — `idle_frames` を 0 に戻すと軌道が変わる
///
/// `is_sleeping` が両方 false のまま **`idle_frames` だけ**を 0 にする
/// ここが red なら scene が悪い (= `idle_frames` が軌道に出ていない) ので、
/// 下の被覆 test は「失っても害が無い状態」を pin していることになる
#[test]
fn idle_frames_is_load_bearing_in_this_scene() {
    let mut a = drifting_world();
    let mut b = drifting_world();

    // 眠る直前まで進める (frames_to_sleep = 4)
    step_n(&mut a, 3);
    step_n(&mut b, 3);

    assert_eq!(
        a.serialize_state(),
        b.serialize_state(),
        "前提: 同じ scene を同じ step 数だけ進めたら blob は一致する"
    );
    assert!(
        !a.is_sleeping(0) && !b.is_sleeping(0),
        "前提: まだ眠っていない状態で比較する (眠ると別の分岐に入る)"
    );
    assert!(
        a.islands.sleep_data[0].idle_frames > 0,
        "前提: idle_frames が溜まっている scene でないと対照実験にならない"
    );

    // b の idle_frames だけを 0 に戻す (sleep state は触らない)
    b.islands.sleep_data[0].idle_frames = 0;
    // ⚠️ **本行は被覆が入っていることの独立検査** — v1 format で
    // `idle_frames` を blob に入れたので、ここは **差が出なければならない**
    // (旧 format ではこの行が `assert_eq!` で、blob が idle_frames を
    // 持っていなかったことの記録だった 実装が変わったので向きが反転した)
    assert_ne!(
        a.serialize_state(),
        b.serialize_state(),
        "idle_frames が blob の被覆に入っていない — v1 format の sleep 部が \
         書かれていないか、読み側が復元していない"
    );

    // 以降を進めると、眠るタイミングが変わって軌道が割れる
    step_n(&mut a, 12);
    step_n(&mut b, 12);

    assert_ne!(
        a.serialize_state(),
        b.serialize_state(),
        "idle_frames を失うと軌道が変わるべき — 一致するならこの scene では \
         idle_frames が軌道に出ていない (scene を選び直す)"
    );
}

/// ⚠️ 本命 — `SleepState` + `idle_frames` が serialize/deserialize を往復しない
///
/// `deserialize_state` 末尾の `IslandManager::new(n, config)` で 0 に戻るため、
/// **blob から復元した世界は sleep 状態を失う**
#[test]
fn sleep_state_survives_serialize_roundtrip() {
    let mut world = drifting_world();
    step_n(&mut world, 3);

    let idle_before = world.islands.sleep_data[0].idle_frames;
    let state_before = world.islands.sleep_data[0].state;
    assert!(
        idle_before > 0,
        "前提: idle_frames が溜まっている状態を往復させる"
    );

    let blob = world.serialize_state();
    // 壊す
    world.islands.sleep_data[0].idle_frames = 0;
    // 復元
    assert!(world.deserialize_state(&blob), "同 body 数なので復元できる");

    assert_eq!(
        world.islands.sleep_data[0].idle_frames, idle_before,
        "idle_frames が復元されていない — blob の被覆に入っていない"
    );
    assert_eq!(
        world.islands.sleep_data[0].state, state_before,
        "SleepState が復元されていない — blob の被覆に入っていない"
    );
}

/// ⚠️ 本命 2 — 巻き戻して続きを走らせると軌道が一致する (rollback の要件)
///
/// 対照実験が示すとおり `idle_frames` は軌道に出るので、被覆が足りないと
/// 「同じ blob から続けたのに違う未来になる」= lockstep の前提が崩れる
#[test]
fn rollback_then_continue_is_bit_identical() {
    // blob は「3 step 進めた時点」のもの (engine は決定論なので、同じ scene を
    // 同じ step 数進めれば同じ状態になる ⇒ `Clone` は要らない)
    let blob = {
        let mut w = drifting_world();
        step_n(&mut w, 3);
        w.serialize_state()
    };

    // 経路 1: 巻き戻しを挟まずに 3 + 12 step
    let mut a = drifting_world();
    step_n(&mut a, 3 + 12);
    let after_direct = a.serialize_state();

    // 経路 2: 3 step の後さらに進めて状態を汚し、blob で巻き戻してから 12 step
    let mut b = drifting_world();
    step_n(&mut b, 3 + 5);
    assert!(b.deserialize_state(&blob), "巻き戻す");
    step_n(&mut b, 12);
    let after_rollback = b.serialize_state();

    assert_eq!(
        after_direct, after_rollback,
        "巻き戻してから続けた未来が直進した未来と違う — 被覆の穴 \
         (`idle_frames` が 0 に戻るので眠るタイミングがずれる)"
    );
}
