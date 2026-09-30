//! `SimulationChecksum` の被覆が状態の被覆より狭くないことの oracle (B-13)
//!
//! doctrine の主張「遷移が確定的」は **desync を検出できること** に依っている
//! ところが `SimulationChecksum::from_world` は
//!
//! - position を `.hi` + `.lo` の両方
//! - velocity / rotation を **`.hi` だけ**
//! - `angular_velocity` を **1 語も**
//!
//! で混ぜているので、208 byte/body のうち **角速度 48 byte + 小数語 56 byte が
//! 盲点**になる ⚠️ **`serialize_state` の blob は角速度も小数語も持っている**
//! (`integration_physics.rs` の Test 23 が往復を実証済) ので、検証の道具だけが
//! 状態より狭い = 「一致を検証する道具」が被覆より狭い状態
//!
//! 期待値は実装出力から取っていない: **blob が区別する 2 世界は checksum も
//! 区別しなければならない** という doctrine の要求そのもの
//!
//! ⚠️ 陽性対照を 2 本入れてある (同一世界 / position の小数語) — 入れないと
//! 「常に不一致を返す」実装でも全部 green になる

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, SimulationChecksum, Vec3Fix};

/// 1 body の世界を作る (position / velocity / rotation / angular_velocity を
/// 呼び出し側が自由に潰せる状態で返す)
fn one_body_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let mut body = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);
    body.velocity = Vec3Fix::from_int(4, 5, 6);
    body.angular_velocity = Vec3Fix::from_int(7, 8, 9);
    world.add_body(body);
    world
}

/// blob が違えば checksum も違う、を確かめるための共通判定
///
/// ⚠️ blob 側も一緒に assert する — blob が同一なら「checksum が区別できない」
/// のは正しい挙動なので、test が何を主張しているか曖昧になる
fn assert_blob_and_checksum_differ(a: &PhysicsWorld, b: &PhysicsWorld, what: &str) {
    assert_ne!(
        a.serialize_state(),
        b.serialize_state(),
        "{what}: 前提が崩れている — blob が同一なら checksum を責められない"
    );
    assert_ne!(
        SimulationChecksum::from_world(a),
        SimulationChecksum::from_world(b),
        "{what}: blob は区別するのに checksum が同一 = desync を見逃す"
    );
}

/// 陽性対照 1 — 同一の世界は同一の checksum
#[test]
fn identical_worlds_share_a_checksum() {
    let a = one_body_world();
    let b = one_body_world();
    assert_eq!(a.serialize_state(), b.serialize_state());
    assert_eq!(
        SimulationChecksum::from_world(&a),
        SimulationChecksum::from_world(&b),
        "同一状態で checksum が割れるなら、以下の不一致 test は何も証明しない"
    );
}

/// 陽性対照 2 — position の小数語 1 ulp は現実装でも区別できる
///
/// position だけは `.hi` と `.lo` の両方を混ぜているので、ここが red になったら
/// harness 自体が壊れている
#[test]
fn position_low_word_is_already_covered() {
    let a = one_body_world();
    let mut b = one_body_world();
    b.bodies[0].position.y.lo += 1;
    assert_blob_and_checksum_differ(&a, &b, "position.y.lo 1 ulp");
}

/// ⚠️ 本命 1 — 角速度が完全に別の 2 世界
#[test]
fn checksum_covers_angular_velocity() {
    let a = one_body_world();
    let mut b = one_body_world();
    b.bodies[0].angular_velocity = Vec3Fix::from_int(-70, -80, -90);
    assert_blob_and_checksum_differ(&a, &b, "angular_velocity が全成分別物");
}

/// ⚠️ 本命 2 — velocity の小数語 1 ulp だけ違う 2 世界
///
/// `.hi` しか混ぜていないので、1 ulp の乖離 (= lockstep の desync が始まる
/// 最小単位) がそのまま盲点になる
#[test]
fn checksum_covers_velocity_low_word() {
    let a = one_body_world();
    let mut b = one_body_world();
    b.bodies[0].velocity.y.lo += 1;
    assert_blob_and_checksum_differ(&a, &b, "velocity.y.lo 1 ulp");
}

/// ⚠️ 本命 3 — rotation の小数語 1 ulp だけ違う 2 世界
#[test]
fn checksum_covers_rotation_low_word() {
    let a = one_body_world();
    let mut b = one_body_world();
    b.bodies[0].rotation.z.lo += 1;
    assert_blob_and_checksum_differ(&a, &b, "rotation.z.lo 1 ulp");
}

/// ⚠️ 本命 4 — 角速度の小数語 1 ulp (成分の入れ替えでなく最小単位)
#[test]
fn checksum_covers_angular_velocity_low_word() {
    let a = one_body_world();
    let mut b = one_body_world();
    b.bodies[0].angular_velocity.x.lo += 1;
    assert_blob_and_checksum_differ(&a, &b, "angular_velocity.x.lo 1 ulp");
}
