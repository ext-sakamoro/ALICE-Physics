//! Oracle: 大変形した布が自分自身を通り抜けてはならない (壁 1/4 後半 = 自己接触)
//!
//! # この file が置かれている理由
//!
//! `src/` の自己接触は `cloth.rs` の **粒子-粒子の距離ばね 1 種だけ**です
//! (`tri_tri` / `triangle_triangle` / `edge_edge` / `vertex_face` / `point_triangle`
//! は `src/` 全体で 0 hit) 粒子どうしの距離だけを見る判定は、**三角形の内部を
//! 通り抜ける頂点を原理的に見られません** — 交点が 3 頂点のどれからも検出半径より
//! 遠ければ、何も起きないまま素通りします
//!
//! この file は `tests/analytic_large_rotation.rs` と同じ **双子構造**を取ります:
//!
//! - **目標 oracle** は現在の solver で **red でなければならず**、その red が
//!   「計器が測りたい性質に届いている」ことの証拠です 実装が入るまで `#[ignore]`
//! - **companion guard** は現在の挙動 (= 自己接触が何もしていない) を pin します
//!   **常にどちらか一方だけが green** であるべきです
//!
//! ⚠️ **反転の契約**: 頂点-面の自己接触が入ると、companion
//! `self_collision_toggle_changes_nothing_in_a_scene_that_self_intersects` の assert は
//! **成立しなくなります** (`pos_on == pos_off` も、前提の `crossings_on > 0` も偽になる)
//!
//! ⚠️ **この companion は「粒子-粒子が三角形重心の頂点を見ない」という構造的事実を
//! pin しているのではありません** 構造的事実を pin しているのは
//! `particle_distance_spring_cannot_see_the_closed_form_crossing` (閉形式なので恒久に真) で、
//! companion が pin しているのは **今の実装が不活性であるという現状**です だから
//! 「削除せず red のまま残す」は CI を恒久 red にするだけで選べません
//!
//! **削除でなく書き換えてください** 壁 2 の `graded_refinement_leaves_hanging_faces` と
//! 同じ decommission の形です (旧実測を doc の `# Before` に残して assert を反転させる)
//! 具体的には 3 箇所:
//!
//! 1. 目標 oracle `a_crumpled_cloth_does_not_pass_through_itself` の `#[ignore]` を外す
//! 2. companion を `..._changes_the_result_in_a_scene_that_self_intersects` に改名し、
//!    `assert_eq!(pos_on, pos_off)` を `assert_ne!` に反転する
//! 3. companion の前提を `crossings_on > 0` から **`crossings_off > 0`** に変える
//!    (自己接触を切れば依然として自己交差する = scene が判別に向いていることの確認)
//!
//! こうすれば companion は実装後も「自己接触が実際に効いている」の恒久 guard として残り、
//! `# Before` に残した旧実測 (貫通 7 回 / ON と OFF が bit 一致) も記録として消えません
//!
//! # 実測 (2026-09-29、`Fix128` のみなので決定論、再実行で同値)
//!
//! ⚠️ **盲点はパラメータ調整では消えません** 検出半径を閉形式の「交点-最近傍頂点
//! `√(1/8) ≈ 0.3536`」より大きくしても貫通は残ります:
//!
//! | `self_collision_distance` | 0.05 | 0.25 | 0.5 | 1.0 |
//! |---|---|---|---|---|
//! | 貫通回数 (`CRUMPLE_STEPS = 120`) | 7 | 5 | 4 | 5 |
//!
//! **単調ですらありません** 半径を 20 倍にしても 5 回残る = 盲点は「半径が小さいだけ」
//! ではなく **構造的**です (粒子-粒子は三角形の内部を通る経路そのものを持たない)
//!
//! ⚠️ **貫通回数は「どれだけ悪いか」の尺度ではありません** crumple は分岐が多く、
//! `CRUMPLE_STEPS` に対しても非単調です:
//!
//! | `CRUMPLE_STEPS` | 15 | 30 | 60 | 90 | 120 (採用) |
//! |---|---|---|---|---|---|
//! | 貫通回数 | 3 | 20 | **1** | 3 | 7 |
//!
//! ⚠️ **`CRUMPLE_STEPS` を触る時に単調だと思わないでください** 最小は `steps = 60` の
//! **1 回**で、companion の前提 `crossings_on > 0` が崩れる寸前です `steps = 30` は
//! 貫通 20 回と数は多いですが、**非単調な曲線の 1 点なので「余裕が 3 倍」にはなりません**
//! (mesher / solver が少し変われば 60 側に落ちます) **実行時間のためだけに動かさないこと**
//!
//! # 閉形式 (出所: 手計算、厳密有理数で検算済)
//!
//! 三角形 `a = (0,0,0)`, `b = (1,0,0)`, `c = (0,0,1)` (XZ 平面)、
//! 頂点が `(1/4, 1/2, 1/4)` から `(1/4, -1/2, 1/4)` へ等速:
//!
//! | 量 | 値 |
//! |---|---|
//! | 交差パラメータ `t*` | `1/2` |
//! | 交点 | `(1/4, 0, 1/4)` |
//! | 重心座標 `(w_a, w_b, w_c)` | `(1/2, 1/4, 1/4)` |
//! | 交点から `a` までの距離² | `1/8` |
//! | 交点から `b` / `c` までの距離² | `5/8` |
//!
//! **全て 2 進小数なので `Fix128` (Q64.64) で厳密に表現でき、許容差なしの等値
//! assert が書けます** 距離は無理数になるので **距離²** で pin します
//!
//! # 精度の扱い — なぜ「厳密構成 + 区間」で、`Mul` の丸めを待たないのか
//!
//! `Fix128::Mul` は 256 bit 積の下位を **切り捨て**ます (`tests/reduction_order_independence.rs`
//! が非結合性を pin 済) 連続時間 (CCD) の同一平面条件は `t` の 3 次式なので、
//! 零点近傍で切り捨てが符号を誤らせます ただし:
//!
//! 1. **最近接丸めにしても 3 次式の符号判定は厳密になりません** 丸めがある限り
//!    零点近傍の符号は不定で、変わるのは偏りが系統的か無偏かだけです
//! 2. oracle の座標は**こちらが選べます** 全ての座標を `2^-16` の整数倍にすれば、
//!    述語に現れる 3 重積は `2^-48` の整数倍になり **Q64.64 で厳密**です
//!    (Shewchuk の exact predicates の縮小版)
//! 3. 厳密構成が成立しない scene (係数が range を溢れる) では **silent に丸めず、
//!    `assert_exact_triple_products` が明示的に落ちます**
//!
//! よって: **係数は厳密に構成し、根の分離は区間で挟む** `Mul` の丸めは独立判断として
//! Backlog に残し、本 file の blocker にしません
//!
//! # ⚠️ 実装への申し送り — 一貫性は構成上達成する
//!
//! CCD の頑健性で効くのは「符号が正しいこと」でなく **「符号が一貫していること」**です
//! 同じ幾何量を 2 箇所で評価して違う符号が出ると、接触集合が非多様体になって貫通や
//! jitter が出ます 逆に符号が (たとえ真の値と違っても) 全箇所で一致していれば、
//! 集合は整合したままです
//!
//! **`Fix128` の切り捨ては決定的なので、符号は再現可能です** つまり
//! **各述語を 1 回だけ計算して使い回せば、一貫性は構成上達成でき、厳密算術は要りません**
//!
//! 同じ理由で **接触法線は必ず 1 回の独立計算で作り、accumulate して作らないでください**
//! (`Mul` は非結合なので、足し込みの順序が法線を変えます)
//!
//! そして応答は **Jacobi 型** (頂点ごとの Δ buffer に `+` で溜めて最後に適用) にしてください
//! `Fix128` の加算は `Z/2¹²⁸` の厳密な群演算なので、**独立に作った積の和は任意の順序で
//! bit 一致**します (同 `reduction_order_independence.rs`) 接触ペアの列挙順を sort で
//! 救うのでなく、**構成上無関係にする**のが本 crate の既定の解法です
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cloth::Cloth;
use alice_physics::math::{Fix128, Vec3Fix};

// ---------------------------------------------------------------------------
// 厳密性の前提 (ys-08 判断 (iii): 係数を厳密に構成し、溢れたら明示的に落とす)
// ---------------------------------------------------------------------------

/// 座標が `2^-16` の整数倍かつ `|v| < 2^10` であることを要求する
///
/// この 2 条件の下で、述語に現れる 3 重積は `2^-48` の整数倍・絶対値 `< 2^30` に
/// 収まり、Q64.64 で **厳密**に表現される (`Fix128::Mul` の切り捨てが効かない)
///
/// ⚠️ 条件を外れた scene では silent に丸まるのでなく、ここで落ちる
fn assert_exact_triple_products(v: Vec3Fix, name: &str) {
    for (axis, c) in [("x", v.x), ("y", v.y), ("z", v.z)] {
        // Q64.64: 値 = hi + lo / 2^64  →  2^-16 の整数倍 ⟺ 下位 48 bit が 0
        assert_eq!(
            c.lo & ((1u64 << 48) - 1),
            0,
            "{name}.{axis} が 2^-16 の整数倍でない (lo = {:#x}) \
             この座標では 3 重積が厳密にならないので scene を選び直す",
            c.lo
        );
        assert!(
            c.hi.abs() < 1024,
            "{name}.{axis} = {} が大きすぎる (|v| < 2^10 が厳密構成の前提)",
            c.hi
        );
    }
}

// ---------------------------------------------------------------------------
// 述語 (除算なし、`Fix128` のみ)
// ---------------------------------------------------------------------------

/// 線分 `p0 → p1` が三角形 `(a, b, c)` の内部を貫くか (Möller–Trumbore、除算なし)
///
/// 除算を避けて det でスケールしたまま符号比較する 交点が必要な呼び出し元向けに
/// `(t_scaled, det)` も返す (`t = t_scaled / det`)
///
/// ⚠️ **各量を 1 度だけ計算して使い回す** (module doc の一貫性の規律)
fn segment_pierces_triangle(
    p0: Vec3Fix,
    p1: Vec3Fix,
    a: Vec3Fix,
    b: Vec3Fix,
    c: Vec3Fix,
) -> Option<(Fix128, Fix128)> {
    let dir = p1 - p0;
    let e1 = b - a;
    let e2 = c - a;
    let h = dir.cross(e2);
    let mut det = e1.dot(h);
    if det.is_zero() {
        return None; // 線分が三角形の面に平行
    }
    let s = p0 - a;
    let q = s.cross(e1);
    let mut u = s.dot(h);
    let mut v = dir.dot(q);
    let mut t = e2.dot(q);

    // det の符号を正に正規化して、以降を符号比較だけで済ませる
    if det < Fix128::ZERO {
        det = Fix128::ZERO - det;
        u = Fix128::ZERO - u;
        v = Fix128::ZERO - v;
        t = Fix128::ZERO - t;
    }
    if u < Fix128::ZERO || v < Fix128::ZERO || u + v > det {
        return None; // 三角形の外
    }
    if t < Fix128::ZERO || t > det {
        return None; // 線分の区間外
    }
    Some((t, det))
}

/// 頂点 `p` と三角形 `(a, b, c)` が頂点を共有しているか
const fn shares_vertex(p: usize, tri: [usize; 3]) -> bool {
    p == tri[0] || p == tri[1] || p == tri[2]
}

// ---------------------------------------------------------------------------
// A. 頂点-面の交差 — 閉形式、厳密等値
// ---------------------------------------------------------------------------

#[test]
fn vertex_face_crossing_matches_the_closed_form_exactly() {
    let a = Vec3Fix::from_int(0, 0, 0);
    let b = Vec3Fix::from_int(1, 0, 0);
    let c = Vec3Fix::from_int(0, 0, 1);
    let quarter = Fix128::from_ratio(1, 4);
    let half = Fix128::from_ratio(1, 2);
    let p0 = Vec3Fix::new(quarter, half, quarter);
    let p1 = Vec3Fix::new(quarter, Fix128::ZERO - half, quarter);

    for (v, n) in [(a, "a"), (b, "b"), (c, "c"), (p0, "p0"), (p1, "p1")] {
        assert_exact_triple_products(v, n);
    }

    let Some((t_scaled, det)) = segment_pierces_triangle(p0, p1, a, b, c) else {
        panic!("閉形式では t* = 1/2 で貫くはずの線分が検出されなかった");
    };

    // t* = 1/2  ⟺  2 * t_scaled == det
    assert_eq!(
        t_scaled * Fix128::from_int(2),
        det,
        "oracle: t* = 1/2 (手計算、厳密有理数で検算済)"
    );

    // 交点 (1/4, 0, 1/4) — t* が厳密なので交点も厳密
    let crossing = p0 + (p1 - p0) * half;
    assert_eq!(crossing, Vec3Fix::new(quarter, Fix128::ZERO, quarter));

    // 重心座標 (w_a, w_b, w_c) = (1/2, 1/4, 1/4)
    // XZ 平面の直角三角形なので w_b = x、w_c = z、w_a = 1 - w_b - w_c
    assert_eq!(crossing.x, quarter, "oracle: w_b = 1/4");
    assert_eq!(crossing.z, quarter, "oracle: w_c = 1/4");
    assert_eq!(
        Fix128::ONE - crossing.x - crossing.z,
        half,
        "oracle: w_a = 1/2"
    );

    // 交点から各頂点までの距離² (無理数を避けて² で pin)
    assert_eq!(
        (crossing - a).length_squared(),
        Fix128::from_ratio(1, 8),
        "oracle: |交点 - a|² = 1/8"
    );
    assert_eq!(
        (crossing - b).length_squared(),
        Fix128::from_ratio(5, 8),
        "oracle: |交点 - b|² = 5/8"
    );
    assert_eq!(
        (crossing - c).length_squared(),
        Fix128::from_ratio(5, 8),
        "oracle: |交点 - c|² = 5/8"
    );
}

/// 粒子-粒子の距離ばねが、上の交差を **原理的に見られない**ことの閉形式
///
/// ⚠️ これは「自己接触が検証されている」の証拠ではありません 現在の実装が
/// **なぜ** 何もしないのかを説明する量です
#[test]
fn particle_distance_spring_cannot_see_the_closed_form_crossing() {
    // `ClothConfig::default()` の検出半径
    let d = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(1),
        Fix128::from_int(1),
        2,
        2,
        Fix128::from_ratio(1, 100),
    )
    .config
    .self_collision_distance;

    // 交点から最も近い頂点 a までの距離² = 1/8
    let nearest_sq = Fix128::from_ratio(1, 8);
    assert!(
        d * d < nearest_sq,
        "検出半径² {} が交点-最近傍頂点の距離² {} 以上なら、この scene は盲点を示さない",
        (d * d).to_f32(),
        nearest_sq.to_f32()
    );
}

/// 負の control — 三角形を外す線分は検出されてはならない
///
/// ⚠️ これが無いと `vertex_face_crossing_matches_the_closed_form_exactly` の green は
/// 「述語が常に Some を返す」でも成立してしまう (正の control だけでは歯が無い)
#[test]
fn the_predicate_rejects_segments_that_miss_the_triangle() {
    let a = Vec3Fix::from_int(0, 0, 0);
    let b = Vec3Fix::from_int(1, 0, 0);
    let c = Vec3Fix::from_int(0, 0, 1);
    let h = Fix128::from_ratio(1, 2);
    let two = Fix128::from_int(2);

    // (a) 三角形の外 (重心座標が範囲外): x = z = 2 は u + v = 4 > 1
    assert!(
        segment_pierces_triangle(
            Vec3Fix::new(two, h, two),
            Vec3Fix::new(two, Fix128::ZERO - h, two),
            a,
            b,
            c
        )
        .is_none(),
        "三角形の外を通る線分が検出された"
    );

    // (b) 斜辺の外側すぐ: x = z = 3/4 は u + v = 3/2 > 1
    let q3 = Fix128::from_ratio(3, 4);
    assert!(
        segment_pierces_triangle(
            Vec3Fix::new(q3, h, q3),
            Vec3Fix::new(q3, Fix128::ZERO - h, q3),
            a,
            b,
            c
        )
        .is_none(),
        "斜辺の外側を通る線分が検出された"
    );

    // (c) 線分が面に届かない (区間外): 面の上方だけを動く
    let q = Fix128::from_ratio(1, 4);
    assert!(
        segment_pierces_triangle(
            Vec3Fix::new(q, Fix128::ONE, q),
            Vec3Fix::new(q, h, q),
            a,
            b,
            c
        )
        .is_none(),
        "面に届かない線分が検出された"
    );

    // (d) 面に平行
    assert!(
        segment_pierces_triangle(
            Vec3Fix::new(Fix128::ZERO, h, Fix128::ZERO),
            Vec3Fix::new(Fix128::ONE, h, Fix128::ZERO),
            a,
            b,
            c
        )
        .is_none(),
        "面に平行な線分が検出された"
    );
}

// ---------------------------------------------------------------------------
// B. 辺-辺の最近接 — 閉形式、厳密等値
// ---------------------------------------------------------------------------

#[test]
fn edge_edge_closest_approach_matches_the_closed_form_exactly() {
    // 線分 A: (0,0,0) → (1,0,0)      線分 B: (1/2, 1/4, -1) → (1/2, 1/4, 1)
    // 閉形式: s* = t* = 1/2、最近接点 (1/2,0,0) と (1/2,1/4,0)、距離² = 1/16
    let half = Fix128::from_ratio(1, 2);
    let quarter = Fix128::from_ratio(1, 4);
    let a0 = Vec3Fix::from_int(0, 0, 0);
    let a1 = Vec3Fix::from_int(1, 0, 0);
    let b0 = Vec3Fix::new(half, quarter, Fix128::from_int(-1));
    let b1 = Vec3Fix::new(half, quarter, Fix128::ONE);

    for (v, n) in [(a0, "a0"), (a1, "a1"), (b0, "b0"), (b1, "b1")] {
        assert_exact_triple_products(v, n);
    }

    let pa = a0 + (a1 - a0) * half;
    let pb = b0 + (b1 - b0) * half;
    assert_eq!(pa, Vec3Fix::new(half, Fix128::ZERO, Fix128::ZERO));
    assert_eq!(pb, Vec3Fix::new(half, quarter, Fix128::ZERO));
    assert_eq!(
        (pb - pa).length_squared(),
        Fix128::from_ratio(1, 16),
        "oracle: 最近接距離² = 1/16"
    );

    // 2 辺は直交し、最近接方向は両辺に直交する (閉形式の裏取り)
    assert!((a1 - a0).dot(b1 - b0).is_zero(), "oracle: 2 辺は直交");
    assert!((pb - pa).dot(a1 - a0).is_zero(), "oracle: 最近接方向 ⊥ A");
    assert!((pb - pa).dot(b1 - b0).is_zero(), "oracle: 最近接方向 ⊥ B");
}

// ---------------------------------------------------------------------------
// C. 連続時間 (CCD) の同一平面条件 — 係数は厳密構成、根は区間で挟む
// ---------------------------------------------------------------------------

/// `det(b(t) - a, c(t) - a, p(t) - a)` — 3 頂点と侵入点が同一平面に載る条件
///
/// scene: `a = (0,0,0)` 固定、`b(t) = (1, t, t)`、`c(t) = (t, t, 1)`、
/// `p(t) = (1/4, 1/4, 1/4 + t)`
///
/// 手で展開すると `P(t) = -t³ + (3/4)t² + (1/2)t - 1/4` (厳密有理数で検算済)
fn coplanarity_det(t: Fix128) -> Fix128 {
    let a = Vec3Fix::ZERO;
    let b = Vec3Fix::new(Fix128::ONE, t, t);
    let c = Vec3Fix::new(t, t, Fix128::ONE);
    let q = Fix128::from_ratio(1, 4);
    let p = Vec3Fix::new(q, q, q + t);
    (p - a).dot((b - a).cross(c - a))
}

#[test]
fn ccd_coplanarity_cubic_is_constructed_exactly_and_its_root_is_bracketed() {
    // 手で導いた 3 次式との厳密一致 (2 進小数の試験点なので許容差なし)
    // 出所: 手計算 → python の Fraction で検算
    let samples = [
        (Fix128::ZERO, Fix128::ZERO - Fix128::from_ratio(1, 4)), // P(0)    = -1/4
        (
            Fix128::from_ratio(1, 4),
            Fix128::ZERO - Fix128::from_ratio(3, 32),
        ), // P(1/4)  = -3/32
        (
            Fix128::from_ratio(3, 8),
            Fix128::ZERO - Fix128::from_ratio(5, 512),
        ), // P(3/8)  = -5/512
        (Fix128::from_ratio(7, 16), Fix128::from_ratio(117, 4096)), // P(7/16) = 117/4096
        (Fix128::from_ratio(1, 2), Fix128::from_ratio(1, 16)),   // P(1/2)  = 1/16
    ];
    for (t, expected) in samples {
        // 係数が厳密に構成できる scene であることを先に確かめる
        assert_exact_triple_products(Vec3Fix::new(t, t, t), "t");
        assert_eq!(
            coplanarity_det(t),
            expected,
            "oracle: P(t) = -t³ + (3/4)t² + (1/2)t - 1/4 を t = {} で評価",
            t.to_f32()
        );
    }

    // 根の分離は厳密値でなく **区間で挟む** (ys-08 判断 (ii))
    let lo = Fix128::from_ratio(3, 8);
    let hi = Fix128::from_ratio(7, 16);
    assert!(coplanarity_det(lo) < Fix128::ZERO, "oracle: 区間左端で負");
    assert!(coplanarity_det(hi) > Fix128::ZERO, "oracle: 区間右端で正");
    assert_eq!(
        hi - lo,
        Fix128::from_ratio(1, 16),
        "根を挟む区間の幅は 1/16"
    );
}

// ---------------------------------------------------------------------------
// D. 自己交差する scene — 目標 oracle と companion の対
// ---------------------------------------------------------------------------

/// 9x9 の布の境界を中心へ縮めて座屈・crumple させる
///
/// 対称性は `RNG` でなく **決定的な 1/8 の持ち上げ**で破る (lockstep 規律)
fn run_crumple(self_collision: bool, steps: usize) -> (usize, Vec<Vec3Fix>) {
    const RES: usize = 9;
    let mut cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        RES,
        RES,
        Fix128::from_ratio(1, 100),
    );
    cloth.config.self_collision = self_collision;
    cloth.config.self_collision_distance = Fix128::from_ratio(5, 100);
    let bump = Fix128::from_ratio(1, 8);
    cloth.positions[4 * RES + 4].y = bump;
    cloth.prev_positions[4 * RES + 4].y = bump;

    let is_boundary = |i: usize| {
        let (r, c) = (i / RES, i % RES);
        r == 0 || c == 0 || r == RES - 1 || c == RES - 1
    };
    let rest = cloth.positions.clone();
    for i in 0..RES * RES {
        if is_boundary(i) {
            cloth.inv_masses[i] = Fix128::ZERO;
        }
    }

    let dt = Fix128::from_ratio(1, 60);
    let center = Fix128::from_int(4);
    let total = Fix128::from_int(steps as i64);
    let mut crossings = 0usize;
    for s in 0..steps {
        let shrink = Fix128::ONE - Fix128::from_ratio(7, 8) * Fix128::from_int(s as i64) / total;
        for (i, r) in rest.iter().enumerate() {
            if !is_boundary(i) {
                continue;
            }
            cloth.positions[i].x = center + (r.x - center) * shrink;
            cloth.positions[i].z = center + (r.z - center) * shrink;
        }
        let before = cloth.positions.clone();
        cloth.step(dt);
        for (p, &bp) in before.iter().enumerate() {
            for tri in &cloth.triangles {
                if shares_vertex(p, *tri) {
                    continue;
                }
                if segment_pierces_triangle(
                    bp,
                    cloth.positions[p],
                    cloth.positions[tri[0]],
                    cloth.positions[tri[1]],
                    cloth.positions[tri[2]],
                )
                .is_some()
                {
                    crossings += 1;
                }
            }
        }
    }
    (crossings, cloth.positions)
}

const CRUMPLE_STEPS: usize = 120;

/// **目標 oracle** — 布は自分自身を通り抜けてはならない
///
/// ⚠️ 現在の solver では **red が正しい** 粒子-粒子の距離ばねは三角形の内部を
/// 通る頂点を見られないので、crumple した布は自分を素通りする
/// 頂点-面の自己接触が入ったら `#[ignore]` を外すこと
#[test]
#[ignore = "the red is correct: cloth.rs has only particle-particle self-collision, which \
            cannot see a vertex passing through a triangle interior. When vertex-face \
            self-collision lands, remove this attribute AND rewrite -- do NOT delete -- the \
            companion `self_collision_toggle_changes_nothing_in_a_scene_that_self_intersects`: \
            rename it to `..._changes_the_result_in_a_scene_that_self_intersects`, flip its \
            `assert_eq!(pos_on, pos_off)` to `assert_ne!`, and change its premise from \
            `crossings_on > 0` to `crossings_off > 0`. Keep the present numbers (7 crossings, \
            on/off bit-identical) in a `# Before` doc section. Leaving it unchanged is not an \
            option: its assert becomes false, which would make CI permanently red"]
fn a_crumpled_cloth_does_not_pass_through_itself() {
    let (crossings, _) = run_crumple(true, CRUMPLE_STEPS);
    assert_eq!(
        crossings, 0,
        "自己交差する crumple scene で頂点が三角形を {crossings} 回貫いた"
    );
}

/// **companion** — 自己交差する scene で、自己接触の on/off が結果を 1 bit も変えない
///
/// ⚠️ **頂点-面の自己接触が入ると、この assert は成立しなくなります**
/// (`pos_on == pos_off` も、前提の `crossings_on > 0` も偽になる)
///
/// **削除せず、書き換えてください**: 名前を
/// `..._changes_the_result_in_a_scene_that_self_intersects` に変え、`assert_eq!` を
/// `assert_ne!` に反転し、前提を `crossings_off > 0` にする そうすれば実装後も
/// 「自己接触が実際に効いている」の恒久 guard として残ります
/// 旧実測 (貫通 7 回 / ON と OFF が bit 一致) は doc の `# Before` に残すこと
/// 壁 2 の `graded_refinement_leaves_hanging_faces` と同じ decommission の形です
#[test]
fn self_collision_toggle_changes_nothing_in_a_scene_that_self_intersects() {
    let (crossings_on, pos_on) = run_crumple(true, CRUMPLE_STEPS);
    let (_, pos_off) = run_crumple(false, CRUMPLE_STEPS);

    // この scene が判別に向いていることを先に確かめる (scene が自己交差しなければ
    // bit 一致は当たり前で、何も言っていない — `feedback_oracle_scene_hits_verifier_limit`)
    assert!(
        crossings_on > 0,
        "scene が自己交差していない この companion は判別に向かない scene では無意味"
    );
    assert_eq!(
        pos_on, pos_off,
        "自己接触の on/off で結果が変わった = 実装が入った \
         ならば a_crumpled_cloth_does_not_pass_through_itself の #[ignore] を外すこと"
    );
}

// ---------------------------------------------------------------------------
// E. 剛体平行移動の不変性
// ---------------------------------------------------------------------------

/// 場面全体を平行移動しても接触判定は変わらない
///
/// ⚠️ **現時点ではこの assert は vacuous です** `cloth.rs` の空間ハッシュは
/// 64³ 格子の外を clamp しますが、clamp はセルを**併合する方向にしか効かない**ので
/// 距離判定が結果を落とすことはありません 新しい空間構造 (BVH / 階層格子) を
/// 入れた時に初めて効く hygiene として置いています
///
/// **これを「自己接触が検証されている」の証拠に数えないでください**
#[test]
fn translating_the_whole_scene_does_not_change_the_predicate() {
    let shift = Vec3Fix::new(
        Fix128::from_ratio(17, 4),
        Fix128::from_ratio(-9, 2),
        Fix128::from_ratio(33, 8),
    );
    let a = Vec3Fix::from_int(0, 0, 0);
    let b = Vec3Fix::from_int(1, 0, 0);
    let c = Vec3Fix::from_int(0, 0, 1);
    let q = Fix128::from_ratio(1, 4);
    let h = Fix128::from_ratio(1, 2);
    let p0 = Vec3Fix::new(q, h, q);
    let p1 = Vec3Fix::new(q, Fix128::ZERO - h, q);

    let base = segment_pierces_triangle(p0, p1, a, b, c);
    let moved = segment_pierces_triangle(p0 + shift, p1 + shift, a + shift, b + shift, c + shift);
    // `Fix128` の加算は厳密なので、差分から作る述語は平行移動で bit 一致する
    assert_eq!(base, moved, "平行移動で述語が変わった");
}
