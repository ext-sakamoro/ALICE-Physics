//! `reset_world()` (WM-07) と rollback の population fingerprint (gap #3) の oracle
//!
//! 背景は `project_alice_physics_world_auditor_engine_gaps` gap #1 / #3
//! (`~/claude-config/memory/`) — ys-1f の裏取りで判明した穴:
//! `deserialize_state` は body **数**しか見ていないので、`remove_body`
//! (swap_remove) + `add_body` で count が元に戻っても population (= どの
//! body がどんな質量・慣性・collider を持つか) が変わっていれば、別の body
//! に誤った状態を書き込んで silent に通ってしまう 本 file はその穴が
//! fingerprint 検査で塞がれていることを確認する

use alice_physics::{
    Fix128, FixedJoint, Joint, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, Vec3Fix,
};

fn quiet_config() -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    }
}

// ---------------------------------------------------------------------------
// WM-07: reset_world()
// ---------------------------------------------------------------------------

/// 同じ初期状態から N step を 2 回実行すると bit 一致する (doctrine WM-07)
///
/// `reset_world()` を挟んで同じ body を再構築し、2 回目の実行が 1 回目と
/// 完全に同じ軌道を辿ることを確認する — `reset_world` が population を
/// 含む全 field を [`PhysicsWorld::new`] と同じ既定値に戻していないと、
/// (例えば overflow flag や islands の残骸が残っていると) 2 回目が 1 回目と
/// 分岐する
#[test]
fn reset_world_then_same_initial_state_two_runs_bit_identical() {
    fn run(world: &mut PhysicsWorld) -> Vec<u8> {
        world.reset_world();
        let body = RigidBody::new_dynamic(Vec3Fix::from_int(5, 50, 3), Fix128::from_ratio(3, 2));
        world.add_body(body);
        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            world.step(dt);
        }
        world.serialize_state()
    }

    let mut world = PhysicsWorld::new(quiet_config());
    let first = run(&mut world);
    let second = run(&mut world);

    assert_eq!(
        first, second,
        "reset_world 後に同じ初期状態から N step を 2 回実行しても blob が一致しない"
    );
}

/// `reset_world()` は population も空にする (= `new` と同一の既定値)
#[test]
fn reset_world_clears_population_and_overflow_flag() {
    let mut world = PhysicsWorld::new(quiet_config());
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));
    world.add_joint(Joint::Fixed(FixedJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )));
    assert_eq!(world.bodies.len(), 2);
    assert_eq!(world.joint_count(), 1);

    world.reset_world();

    assert_eq!(world.bodies.len(), 0);
    assert_eq!(world.joint_count(), 0);
    assert!(!world.overflow_detected());
    // 空 world 同士は population fingerprint も一致する (何も無いので恒等)
    assert_eq!(
        world.population_fingerprint(),
        PhysicsWorld::new(quiet_config()).population_fingerprint()
    );
}

// ---------------------------------------------------------------------------
// gap #3: population fingerprint
// ---------------------------------------------------------------------------

/// 正常系 — caller が population を replay で正しく合わせていれば rollback は成功する
///
/// snapshot 時点の population (body 0 = mass 1, body 1 = mass 2) と同じ
/// population を再構築してから `deserialize_state` を呼ぶと成功する
#[test]
fn rollback_accepts_population_replayed_to_match_snapshot() {
    let mut src = PhysicsWorld::new(quiet_config());
    src.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 2, 3),
        Fix128::ONE,
    ));
    src.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(4, 5, 6),
        Fix128::from_int(2),
    ));
    let snapshot = src.serialize_state();

    // caller が population を replay で再構築 (同じ mass、違う初期位置)
    let mut dst = PhysicsWorld::new(quiet_config());
    dst.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    dst.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));

    assert!(dst.deserialize_state(&snapshot));
    assert_eq!(dst.bodies[0].position, src.bodies[0].position);
    assert_eq!(dst.bodies[1].position, src.bodies[1].position);
}

/// 本命 — count は一致しているが population が違う場合は拒否する (ys-1f の穴)
///
/// `remove_body(0)` (swap_remove) + `add_body` で count は元の 2 に戻るが、
/// index 0 の body は mass 1 → mass 99 に変わっている (= 別の body)
/// このとき `deserialize_state` は **count 一致だけでは population の違いを
/// 検出できず**、fingerprint 検査が無いと body 0 の位置を誤って書き込んで
/// silent に通ってしまう (fail-fast になっていない)
#[test]
fn rollback_rejects_population_mismatch_despite_matching_count() {
    let mut src = PhysicsWorld::new(quiet_config());
    src.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 2, 3),
        Fix128::ONE,
    ));
    src.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(4, 5, 6),
        Fix128::from_int(2),
    ));
    let snapshot = src.serialize_state();

    let mut dst = PhysicsWorld::new(quiet_config());
    dst.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(7, 7, 7),
        Fix128::ONE,
    ));
    dst.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::from_int(2)));

    // body 0 を破棄して別の mass で再生成 (swap_remove なので index 0 に入る)
    // count は 2 のまま、population (mass) は変わっている
    dst.remove_body(0);
    dst.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(7, 7, 7),
        Fix128::from_int(99),
    ));
    assert_eq!(
        dst.bodies.len(),
        2,
        "count は元に戻っている (count だけでは検出不能)"
    );

    let before = dst.bodies[0].position;
    assert!(
        !dst.deserialize_state(&snapshot),
        "population (mass) が違う snapshot を count 一致だけで受理してしまっている"
    );
    // 拒否時は 1 byte も書き込まれていない
    assert_eq!(
        dst.bodies[0].position, before,
        "拒否したのに body 0 の状態が変わっている (fail-fast になっていない)"
    );
}

/// rollback した先で joint が removed index を跨いで remap される経路でも
/// bit-exact (ys-1f oracle 候補 (3))
///
/// 4 body (0,1,2,3) + joint(2,3) を数 step 進めた時点を snapshot に取る
/// 「直進」はそのまま `remove_body(0)` (swap_remove、`remap_joint_indices`
/// が joint を (2,3) → (2,0) に書き換える) + k step 「rollback+replay」は
/// snapshot と同じ population (4 body + joint(2,3)) を再構築し
/// `deserialize_state` で連続状態を復元してから、**同じ** `remove_body(0)` +
/// k step を replay する — 両者が bit-exact であることを確認する
#[test]
fn rollback_then_replaying_a_body_removal_remaps_joints_bit_exact() {
    fn four_body_with_joint() -> PhysicsWorld {
        let mut w = PhysicsWorld::new(quiet_config());
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::ONE,
        ));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(5, 0, 0),
            Fix128::ONE,
        ));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(1, 0, 0),
            Fix128::ONE,
        ));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(1, 1, 0),
            Fix128::ONE,
        ));
        w.add_joint(Joint::Fixed(FixedJoint::new(
            2,
            3,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )));
        w
    }

    let dt = Fix128::from_ratio(1, 60);

    // snapshot 地点まで進める (この時点の population = 4 body + joint(2,3))
    let mut direct = four_body_with_joint();
    for _ in 0..10 {
        direct.step(dt);
    }
    let snapshot = direct.serialize_state();

    // 直進: snapshot の続きで body 0 を破棄 (swap_remove で body 3 が index 0
    // に移り、joint は (2,3) → (2,0) に remap される) + k step
    direct.remove_body(0);
    assert_eq!(
        direct.joint_count(),
        1,
        "remove_body で joint が消えてはいけない"
    );
    for _ in 0..7 {
        direct.step(dt);
    }
    let direct_final = direct.serialize_state();

    // rollback + replay: snapshot と同じ population + joint を再構築してから
    // deserialize_state で連続状態を復元 → 同じ remove_body(0) + k step
    let mut replayed = four_body_with_joint();
    assert!(replayed.deserialize_state(&snapshot));
    replayed.remove_body(0);
    assert_eq!(replayed.joint_count(), 1);
    for _ in 0..7 {
        replayed.step(dt);
    }
    let replayed_final = replayed.serialize_state();

    assert_eq!(
        direct_final, replayed_final,
        "joint remap を跨ぐ body 破棄で rollback+replay が直進と bit-exact でない"
    );
}
