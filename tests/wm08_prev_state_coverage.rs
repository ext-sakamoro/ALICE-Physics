//! 眠っている body の `prev` が **blob の被覆に不要**であることの不変条件 guard
//!
//! # 当初の仮説と、それが否定された経緯 (2026-09-30)
//!
//! `solver.rs:1929` に
//! `// Skip sleeping bodies (preserve prev for zero-velocity derivation)`
//! とあるので、**眠っている body の `prev_position` / `prev_rotation` は
//! blob 外の load-bearing state** だと考えて本 file を書いた
//!
//! ⚠️ **対照実験が否定した** — 眠っている body の `prev` を直接ずらして
//! `wake_body` → 12 step しても **軌道が完全に一致**した
//!
//! **真因**: `wake_body` で起きた body は次 substep 先頭の
//! `// Store previous state` (`:1886` / `:1936`) で `prev_position = position`
//! が再代入されるので、保持されていた `prev` は `update_velocities`
//! (`velocity = (position − prev_position) * inv_dt`) に届く前に上書きされる
//!
//! ⇒ `:1929` のコメントは **眠っている間 `prev == position` を保って速度導出を
//! ゼロに保つ防御的不変条件**であって、**巻き戻しを跨ぐ state ではない**
//!
//! # 本 file の役割 (残す理由)
//!
//! ⚠️ **この性質が失われたら blob の被覆に `prev` (1 body あたり 112 byte) を
//! 足す必要がある**ので、不変条件として pin しておく
//!
//! ⚠️ **教訓**: 対照実験を先に置いたおかげで、**無駄な format 拡張を回避できた**
//! 本命 test だけ書いていたら green を「穴が無い」と読んでしまう

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SleepConfig, Vec3Fix};

const DT: (i64, i64) = (1, 64);

fn dt() -> Fix128 {
    Fix128::from_ratio(DT.0, DT.1)
}

/// すぐ眠る scene (重力ゼロ、初速ゼロ、`frames_to_sleep` を小さく)
fn sleepy_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    world.set_sleep_config(SleepConfig {
        linear_threshold: Fix128::from_ratio(1, 2),
        angular_threshold: Fix128::from_ratio(1, 2),
        frames_to_sleep: 3,
    });
    let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
    world.add_body(body);
    world
}

fn step_n(world: &mut PhysicsWorld, n: usize) {
    for _ in 0..n {
        world.step(dt());
    }
}

/// 眠るまで進める (上限付き、眠らなければ panic して scene の誤りを露呈させる)
fn step_until_asleep(world: &mut PhysicsWorld) -> usize {
    for i in 1..=60 {
        world.step(dt());
        if world.is_sleeping(0) {
            return i;
        }
    }
    panic!("60 step で眠らなかった — scene を選び直す (sleep 条件を満たしていない)");
}

/// ⚠️ 対照実験 1 — この scene で body が実際に眠る
#[test]
fn this_scene_actually_puts_the_body_to_sleep() {
    let mut world = sleepy_world();
    let n = step_until_asleep(&mut world);
    assert!(world.is_sleeping(0), "眠っていない");
    assert!(n <= 60, "眠るまで {n} step");
}

/// ⚠️ **不変条件** — 眠っている body の `prev` は巻き戻しを跨がない
///
/// ⚠️ **当初の仮説は否定された** (2026-09-30 実測): 「眠っている body の `prev` は
/// blob 外の load-bearing state」と考えて対照実験を書いたが、**`prev` をずらして
/// `wake_body` → step しても軌道が完全に一致した**
///
/// **真因**: `wake_body` で起きた body は次 substep 先頭の
/// `// Store previous state` (`solver.rs:1886` / `:1936`) で
/// **`prev_position = position` が再代入される**ので、保持されていた `prev` は
/// `update_velocities` に届く前に上書きされる
///
/// ⇒ `solver.rs:1929` の `// Skip sleeping bodies (preserve prev for
/// zero-velocity derivation)` は **眠っている間 `prev == position` を保って
/// 速度導出をゼロに保つ防御的不変条件**であって、**巻き戻しを跨ぐ state ではない**
///
/// ⚠️ **本 test が red になったら blob の被覆に `prev_position` /
/// `prev_rotation` (1 body あたり 112 byte) を足す必要がある** —
/// 「wake 後に prev が再代入される」性質が失われたことを意味する
#[test]
fn prev_of_a_sleeping_body_does_not_survive_a_wake() {
    let mut a = sleepy_world();
    let mut b = sleepy_world();
    step_until_asleep(&mut a);
    step_until_asleep(&mut b);
    assert_eq!(
        a.serialize_state(),
        b.serialize_state(),
        "前提: 同じ scene を同じだけ進めたら blob は一致する"
    );

    // b だけ `prev_position` をずらす (眠っているので step では上書きされない)
    b.bodies[0].prev_position = b.bodies[0].position + Vec3Fix::from_int(1, 0, 0);
    assert_eq!(
        a.serialize_state(),
        b.serialize_state(),
        "⚠️ prev_position は v1 blob の被覆外なので、この時点では blob は一致する"
    );

    // 起こして進める — wake 後に prev が再代入されるので軌道は一致する
    a.wake_body(0);
    b.wake_body(0);
    step_n(&mut a, 12);
    step_n(&mut b, 12);
    assert_eq!(
        a.serialize_state(),
        b.serialize_state(),
        "眠っている body の prev_position が軌道に出た — wake 後の \
         `prev_position = position` 再代入が失われている ⇒ blob の被覆に \
         prev_position / prev_rotation を足す必要がある"
    );
}

/// 眠った状態で巻き戻して起こしても軌道が一致する (rollback の要件)
///
/// v1 blob は `prev` を持たないが、上の不変条件 (wake 後に再代入される) により
/// **一致する** ⚠️ red になったら `prev` を被覆に足す合図
#[test]
fn rollback_while_asleep_then_wake_is_bit_identical() {
    // 眠った時点の blob
    let blob = {
        let mut w = sleepy_world();
        step_until_asleep(&mut w);
        w.serialize_state()
    };

    // 経路 1: 眠った時点から起こして 12 step
    let mut a = sleepy_world();
    step_until_asleep(&mut a);
    a.wake_body(0);
    step_n(&mut a, 12);
    let direct = a.serialize_state();

    // 経路 2: 眠った後に「別の未来」を作って prev を汚し、blob で巻き戻してから
    //         同じように起こして 12 step
    let mut b = sleepy_world();
    step_until_asleep(&mut b);
    // 汚す: 起こして動かしてから、また眠らせる (prev が別の値になる)
    b.wake_body(0);
    b.bodies[0].velocity = Vec3Fix::new(Fix128::from_ratio(1, 4), Fix128::ZERO, Fix128::ZERO);
    step_n(&mut b, 8);
    step_until_asleep(&mut b);
    assert!(b.deserialize_state(&blob), "巻き戻す");
    b.wake_body(0);
    step_n(&mut b, 12);
    let rolled = b.serialize_state();

    assert_eq!(
        direct, rolled,
        "眠った状態で巻き戻して起こした未来が直進時と違う — \
         prev_position / prev_rotation が blob の被覆に入っていない \
         (solver.rs:1929 が眠っている body の prev を保持するため)"
    );
}
