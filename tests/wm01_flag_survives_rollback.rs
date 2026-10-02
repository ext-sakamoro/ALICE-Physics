//! overflow flag が blob の被覆に入っていることの oracle (doctrine B-12 の残件)
//!
//! ⚠️ **doctrine B-12 の逐語指摘**: 「flag は状態の一部なので WM-08 の被覆に
//! 含める — `PhysicsWorld` に持つだけで `serialize_state` / rollback の被覆に
//! 入れないと、**overflow した枝を巻き戻した先で flag が消えて `undecided` が
//! 失われる** = B-12 が自分の目的を達成しない」
//!
//! 探索は「この枝は undecided」を巻き戻しを跨いで保持しなければならない
//! (保持できないと、未決定だった枝を後で決定済と誤認する)

use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn healthy_dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// 洗浄が起きる dt (WM-01 の再現条件、`dt ≫ 1`)
fn washing_dt() -> Fix128 {
    Fix128::from_int(1 << 20)
}

fn washing_world() -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, -Fix128::from_int(1 << 30), Fix128::ZERO),
        ..PhysicsConfig::default()
    });
    world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    world
}

/// ⚠️ 対照実験 — blob の長さに flag の 1 byte が含まれている
///
/// ⚠️ **「立つ前後で blob が変わる」では対照実験にならない** (2026-09-30 実測)
/// 洗浄 step は sleep 状態も変えるので、**flag が被覆外でも blob は変わる**
/// ⇒ 長さで見るのが精密 (v3 = header 12 + body ごと 208 + body ごと sleep 5
/// + world ごと flag 1 + world ごと population fingerprint 8)
#[test]
fn the_blob_length_includes_the_flag_byte() {
    let n = 1; // body 1 個
    let len = washing_world().serialize_state().len();
    assert_eq!(
        len,
        12 + n * 208 + n * 5 + 1 + 8,
        "blob の長さに flag の 1 byte が含まれていない (実測 {len})"
    );
}

/// ⚠️ 本命 1 — overflow した枝を巻き戻しても flag が残る
#[test]
fn rolling_back_an_overflowed_branch_keeps_undecided() {
    let mut w = washing_world();
    w.step(washing_dt());
    assert!(w.overflow_detected());
    let blob = w.serialize_state();

    // 別の world (健全) に復元すると、その world も undecided になる
    let mut fresh = washing_world();
    assert!(!fresh.overflow_detected(), "前提: 復元前は立っていない");
    assert!(fresh.deserialize_state(&blob), "復元できる");
    assert!(
        fresh.overflow_detected(),
        "overflow した枝の blob から復元したのに undecided が失われている — \
         探索が未決定の枝を決定済と誤認する (doctrine B-12)"
    );
}

/// ⚠️ 本命 2 — 健全な枝の blob で復元すると flag は下がる
///
/// sticky は「step を重ねても落ちない」ことで、**別の状態を読み込んだら
/// その状態に従う** 落ちないままだと探索で枝を跨いで汚染する
#[test]
fn restoring_a_healthy_branch_clears_undecided() {
    let healthy = {
        let mut w = washing_world();
        w.step(healthy_dt()); // 健全な dt では洗浄しない
        assert!(!w.overflow_detected(), "前提: 健全 step では立たない");
        w.serialize_state()
    };

    let mut w = washing_world();
    w.step(washing_dt());
    assert!(w.overflow_detected(), "前提: 立っている");
    assert!(w.deserialize_state(&healthy), "健全な blob で復元");
    assert!(
        !w.overflow_detected(),
        "健全な枝を読み込んだのに undecided が残っている — \
         枝を跨いで汚染する (sticky は step に対してであって復元に対してではない)"
    );
}
