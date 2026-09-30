//! 巻き戻し後の contact event 列が直進時と一致するかの実測 (WM-07 / WM-08 の続き)
//!
//! ⚠️ **`EventCollector` は frame 間に状態を持ち越す** — `begin_frame` が
//! `curr_pairs` を `prev_pairs` に移し、次 frame の begin / persist / end 判定に
//! 使う (`src/event.rs:90`) ところが `prev_pairs` は **blob に入っていない**ので
//! `deserialize_state` で復元されない
//!
//! ⇒ 巻き戻した先の 1 frame 目は「巻き戻し前の `prev_pairs`」と比較されるため、
//! **同じ blob から続けたのに event 列が直進時と違う**可能性がある
//!
//! 本 file は**その差が実在するかを測る**もの (台帳 / doctrine に記載がない)
//! 期待値は実装出力でなく **「同じ状態から続けた未来は同じであるべき」** という
//! 決定論の要求そのもの

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

/// 2 球が接触と分離を繰り返す scene (重力ゼロ)
///
/// ⚠️ **最初から重なっている配置にする** (2026-09-30 実測、試行 2 回で判明)
/// 「離れた位置から近づけて接触させる」形 (中心 ±3 / 半径 1 / 速度 ±1) では
/// **300 step 回しても contact event が 1 本も出なかった** — 既存の
/// `integration_physics.rs::test_drain_events` が使う働く idiom は
/// **半径 2 の球を距離 1 に置く = 最初から重なっている**形
///
/// 重なりから始めると解法が押し戻して離れ、また近づくので begin / persist /
/// end が全部通る (frame 間の `prev_pairs` 依存を踏む scene になる)
fn approaching_pair() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    let r = Fix128::from_int(2);
    let a = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    world.add_body_with_radius(a, r);
    let b = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    );
    world.add_body_with_radius(b, r);
    world
}

/// 1 step 進めて、その step で出た contact event を (body_a, body_b, type) で返す
fn step_events(world: &mut PhysicsWorld) -> Vec<(usize, usize, String)> {
    world.step(Fix128::from_ratio(1, 64));
    world
        .drain_contact_events()
        .into_iter()
        .map(|e| (e.body_a, e.body_b, format!("{:?}", e.event_type)))
        .collect()
}

fn run(world: &mut PhysicsWorld, n: usize) -> Vec<Vec<(usize, usize, String)>> {
    (0..n).map(|_| step_events(world)).collect()
}

/// ⚠️ 対照実験 — この scene で contact event が実際に出ることを先に確かめる
///
/// ここが red なら (event が 0 本なら) 下の parity test は何も測っていない
#[test]
fn this_scene_actually_produces_contact_events() {
    let mut world = approaching_pair();
    let log = run(&mut world, 300);
    let total: usize = log.iter().map(Vec::len).sum();
    assert!(
        total > 0,
        "contact event が 1 本も出ていない — scene を選び直す (300 step で接触しない)"
    );
    // begin が少なくとも 1 本あること (接触の開始を捉えている)
    assert!(
        log.iter().flatten().any(|(_, _, t)| t.contains("Begin")),
        "Begin event が無い — 接触の開始が観測できていない (出た event: {:?})",
        log.iter().flatten().take(4).collect::<Vec<_>>()
    );
}

/// ⚠️ 本命 — 巻き戻してから続けた event 列が直進時と一致するか
///
/// 一致しなければ **`prev_pairs` が blob 外にあることが event 列に出ている**
/// (= 被覆の穴、`SimulationChecksum` は body の状態しか見ないので検出できない)
#[test]
fn rollback_then_continue_reproduces_the_same_contact_events() {
    // 接触が始まる直前まで進めた blob を作る
    let (blob, warmup) = {
        let mut w = approaching_pair();
        let mut n = 0;
        // 最初の Begin が出る step を探し、その **1 step 前** で snapshot を取る
        // (接触の最中に巻き戻すのが最も差が出やすい)
        let mut first_begin = None;
        for i in 0..300 {
            let ev = step_events(&mut w);
            if ev.iter().any(|(_, _, t)| t.contains("Begin")) {
                first_begin = Some(i);
                break;
            }
            n = i + 1;
        }
        assert!(first_begin.is_some(), "Begin が出る前に 300 step 尽きた");
        // n = Begin の 1 step 前までの step 数
        let mut w2 = approaching_pair();
        run(&mut w2, n + 2); // Begin を 1 step 過ぎた地点 (persist 判定が効く)
        (w2.serialize_state(), n + 2)
    };

    // 経路 1: 巻き戻しなしで warmup + 12 step
    let mut a = approaching_pair();
    run(&mut a, warmup);
    let direct = run(&mut a, 12);

    // 経路 2: warmup + 5 step 進めて汚し、blob で巻き戻してから 12 step
    let mut b = approaching_pair();
    run(&mut b, warmup);
    run(&mut b, 5);
    assert!(b.deserialize_state(&blob), "巻き戻す");
    let rolled = run(&mut b, 12);

    assert_eq!(
        direct, rolled,
        "巻き戻してから続けた contact event 列が直進時と違う — \
         frame 間に持ち越される非 blob 状態 (EventCollector の prev_pairs) が \
         復元されていない"
    );
}
