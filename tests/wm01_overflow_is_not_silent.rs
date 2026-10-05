//! `Fix128` の silent 発散を 3 値の `undecided` として表に出す oracle (WM-01 / B-12)
//!
//! ⚠️ **WM-01 の実測 (2026-09-30)**: silent 発散には 2 経路ある
//!
//! **経路 1 位置 wrap**: `v = 2⁵⁸` で step 31 / `2⁶⁰` で 7 / `2⁶²` で 1
//! velocity は正のまま panic も NaN も出ない
//!
//! ⚠️⚠️ **経路 2 静止洗浄**: `Fix128::mul` の積が `2⁶³` 超で 2 の冪だと `hi` が
//! **厳密に 0** (`2⁴⁰ × 2⁴⁰` = exact `1.21e24`) ⇒ `position += velocity*dt` が
//! 0 加算 ⇒ `update_velocities` が `velocity = (position − prev_position) * inv_dt`
//! で速度も 0 に上書き ⇒ **「原点で完全に静止した自己整合な状態」**になる
//! ⚠️ **静止は物理的に妥当なのでどの不変条件でも red にならない**
//!
//! ⇒ 検出は「値が大きい / 値が飛んだ」でなく **`mul` の積が範囲外**を見る
//! (WM-01 の結論) doctrine B-12 の「3 値判定を物理層まで一貫させる」に対応
//!
//! ⚠️ 対照実験を先に置く — 「この scene で実際に洗浄が起きる」ことを確かめてから
//! 検出器の test を書く 順序を逆にすると偽 green を pin する

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// ⚠️ **洗浄が起きる dt は `dt ≫ 1`** (WM-01 の再現条件)
///
/// 実測記録: `g=-2^30 dt=2^20` で `(y=+0.000e0, vy=+0.000e0)` が 4 step 続く
/// (巨大重力なのに一切動かない)
///
/// ⚠️ **`dt < 1` では原理的に起きない** — Q64.64 で `dt.hi = 0` になるので
/// `hh = velocity.hi × 0 = 0`、`mid >> 64` も `velocity.hi × dt.lo ≥ 2¹²⁷` を
/// 要求し `velocity.hi ≤ 2⁶³` では到達不能 (2026-09-30 算術 + 実測)
fn washing_dt() -> Fix128 {
    Fix128::from_int(1 << 20)
}

/// `checked_mul` の単体 oracle — WM-01 が実測した `2⁴⁰ × 2⁴⁰` を拒否する
///
/// 期待値は実装出力でなく **算術** から: Q64.64 で `2⁴⁰` は `hi = 2⁴⁰`、積の
/// `hi` は `2⁸⁰` で **i64 に収まらない** (`2⁸⁰ mod 2⁶⁴ = 0` なので `as i64` は
/// 0 に wrap する = これが洗浄の正体)
#[test]
fn checked_mul_rejects_the_product_that_wraps_to_zero() {
    let big = Fix128::from_int(1 << 40);
    // 素の `*` は 0 に wrap する (現状の挙動、これ自体は変えない)
    let wrapped = big * big;
    assert_eq!(
        wrapped.hi, 0,
        "前提: 2⁴⁰ × 2⁴⁰ の hi は 0 に wrap する (WM-01 実測の洗浄の正体)"
    );
    // checked 版は None を返す
    assert!(
        big.checked_mul(big).is_none(),
        "範囲外の積を None で拒否していない"
    );
}

/// ⚠️ 対照実験 — 範囲内の積では `checked_mul` が `*` と一致する
///
/// ここが red なら `checked_mul` が常に `None` を返す実装でも上の test が通る
#[test]
fn checked_mul_agrees_with_mul_in_range() {
    let cases = [
        (Fix128::from_int(3), Fix128::from_int(7)),
        (Fix128::from_ratio(1, 3), Fix128::from_ratio(-2, 7)),
        (Fix128::from_int(1 << 20), Fix128::from_int(1 << 20)),
        (Fix128::ZERO, Fix128::from_int(5)),
        (Fix128::ONE, Fix128::from_ratio(-1, 64)),
    ];
    for (a, b) in cases {
        assert_eq!(
            a.checked_mul(b),
            Some(a * b),
            "範囲内なのに checked_mul が * と違う ({a:?} × {b:?})"
        );
    }
}

/// 洗浄が起きる world (⚠️ **極端な重力 `-2³⁰`**、`washing_dt` と組で使う)
///
/// WM-01 §(c) の再現条件そのもの 重力で速度が `2³⁰ × 2²⁰ = 2⁵⁰` 級になり、
/// 次の `position += velocity * dt` で `hh = 2⁵⁰ × 2²⁰ = 2⁷⁰` が i64 に収まらず
/// **`hi` が厳密に 0 に wrap する** (`2⁷⁰ mod 2⁶⁴ = 0`)
fn washing_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, -Fix128::from_int(1 << 30), Fix128::ZERO),
        ..PhysicsConfig::default()
    });
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
}

/// ⚠️ 対照実験 — この scene で実際に洗浄が起きる (位置が動かない / 静止に落ちる)
///
/// ここが red なら scene が洗浄を踏んでいないので、検出器の test は空振りになる
#[test]
fn this_scene_actually_washes_to_rest() {
    let mut world = washing_world();
    let v0 = world.bodies[0].velocity.y;
    world.step(washing_dt());
    let p = world.bodies[0].position.y;
    let v = world.bodies[0].velocity.y;
    assert!(
        p == Fix128::ZERO || v == Fix128::ZERO,
        "洗浄が起きていない (position {p:?} / velocity {v:?}、初速 {v0:?}) — \
         scene を選び直す"
    );
}

/// ⚠️ 本命 — 洗浄が起きた step は `undecided` として表に出る
#[test]
fn washing_step_is_reported_as_undecided() {
    let mut world = washing_world();
    assert!(
        !world.overflow_detected(),
        "前提: step 前は overflow flag が立っていない"
    );
    world.step(washing_dt());
    assert!(
        world.overflow_detected(),
        "洗浄が起きた step で overflow flag が立っていない — \
         静止は物理的に妥当なのでどの不変条件でも red にならず、この flag だけが捕まえる"
    );
}

/// ⚠️ 対照実験 — 正常な scene では flag が立たない
///
/// ここが red なら「常に true を返す」実装でも本命が通る
#[test]
fn a_healthy_scene_does_not_set_the_flag() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    });
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    world.add_body(body);
    for _ in 0..64 {
        world.step(dt());
    }
    assert!(
        !world.overflow_detected(),
        "正常な scene で overflow flag が立っている (誤検出)"
    );
    // 1 秒で 1 m 進んでいる (積分が効いている = scene が動いている証跡)
    assert_ne!(
        world.bodies[0].position.x,
        Fix128::ZERO,
        "前提: 正常 scene では位置が動く"
    );
}

/// ⚠️ flag は sticky (1 度立ったら step を重ねても落ちない)
///
/// 探索で「この枝は undecided」を保持するために必要
#[test]
fn the_flag_is_sticky_across_steps() {
    let mut world = washing_world();
    world.step(washing_dt());
    assert!(world.overflow_detected());
    for _ in 0..8 {
        world.step(washing_dt());
    }
    assert!(
        world.overflow_detected(),
        "flag が step で落ちている — sticky でないと巻き戻し前の undecided が消える"
    );
}
