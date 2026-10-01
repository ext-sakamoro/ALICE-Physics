//! Oracle: 大変形した布が自分自身を通り抜けてはならない (壁 1/4 後半 = 自己接触)
//!
//! # この file が置かれている理由
//!
//! 2026-09-29 時点の `src/` の自己接触は `cloth.rs` の **粒子-粒子の距離ばね 1 種だけ**
//! でした 粒子どうしの距離だけを見る判定は、**三角形の内部を通り抜ける頂点を原理的に
//! 見られません** — 交点が 3 頂点のどれからも検出半径より遠ければ、何も起きないまま
//! 素通りします この file はその盲点を閉形式で示し、目標 oracle を `#[ignore]` 付きの
//! red として置くために作られました
//!
//! **2026-10-01 に頂点-面の自己接触が landing して、目標 oracle は green になりました**
//! (`src/cloth.rs`: substep 内の近接斥力 + frame 単位の掃過線分復元、どちらも Jacobi 蓄積)
//! 以下は decommission の記録です — 契約どおり **どの test も削除していません**
//!
//! | 旧 (2026-09-29) | 新 (2026-10-01) | 変更 |
//! |---|---|---|
//! | `a_crumpled_cloth_does_not_pass_through_itself` (`#[ignore]`、貫通 7) | 同名、green (貫通 0) | `#[ignore]` を外した |
//! | `self_collision_toggle_changes_nothing_in_a_scene_that_self_intersects` | `self_collision_toggle_changes_the_result_in_a_scene_that_self_intersects` | 改名 + `panic!` を `assert!(..is_some())` に反転 + 前提を `crossings_off > 0` に |
//! | `self_collision_moves_particles_without_improving_their_separation` | `self_collision_improves_the_vertex_face_separation_it_constrains` | 改名 + 測る量を頂点-頂点 → **頂点-面**に + 半径 0.5 → 0.05 (理由は下記) |
//! | (無し) | `the_particle_pair_minimum_is_attained_by_two_pinned_vertices` | 旧計器が何を測っていたかを pin (旧数値はここで生きる) |
//! | (無し) | `a_thickness_far_above_the_local_edge_length_is_outside_the_model` | 使用条件 (厚 < 局所辺長の半分) を pin |
//! | (無し) | `the_point_triangle_metric_matches_the_closed_form` | 新計器 (独立実装) 自身の契約 test |
//!
//! **2026-10-01 に辺-辺の自己接触も landing しました** (`src/cloth.rs`:
//! `closest_points_on_segments` + `accumulate_edge_edge_contacts`、頂点-面と**同じ**
//! `Δ`/`hits` buffer に蓄積) 追加した 3 本:
//!
//! | test | 役割 |
//! |---|---|
//! | `the_segment_segment_metric_matches_the_closed_form` | 線分-線分距離の独立実装の契約 test (5 配置、平行と距離 0 を含む) |
//! | `the_vertex_face_metric_cannot_see_the_closed_form_edge_edge_crossing` | ⚠️ **構造的事実**: X 字交差では頂点-面の最小距離² が `65/64` で閾値の外、辺-辺は `1/64` で内側 (閉形式なので恒久に真) |
//! | `self_collision_improves_the_edge_edge_separation_it_constrains` | 辺-辺の段が自分の拘束量を改善することの guard (⚠️ 目標 oracle は辺-辺の段を守りません) |
//!
//! ⚠️ **辺-辺は近接斥力だけで、掃過 (CCD) はありません** frame 単位の復元
//! (`resolve_self_contact_over_frame`) は頂点の掃過線分しか見ないので、1 frame の内側で
//! 交差して `thickness` より離れて終わる辺対は復元されません `src/cloth.rs` の
//! `ClothConfig::self_collision` の doc に逐語で書いてあります
//!
//! ⚠️ **3 本目の反転は doc が当時指定した形 (`min_d2_on > min_d2_off`) では landing
//! できませんでした** 実測すると、**頂点-頂点の最小距離を取る対は 4 run すべて `(7, 17)`
//! = 両方とも pin された境界粒子**で、`run_crumple` が毎 step 座標を代入している量でした
//! (`0.038967 = (0.1396·√2)²` は `shrink(59, 60)` から決まる定数) solver が何をしても
//! 動かないので、その不等号は構造的に成立しません ⚠️ **極値を assert する test は、
//! 「極値を取る要素が実験条件で動くか」を件数の反 vacuous check とは別に確かめること**
//! 詳細は `the_particle_pair_minimum_is_attained_by_two_pinned_vertices` の doc と
//! `[[feedback_cloth_self_contact_metric_measures_the_driver]]`
//!
//! ⚠️ **構造的事実を pin しているのは
//! `particle_distance_spring_cannot_see_the_closed_form_crossing`** (閉形式なので恒久に真)
//! で、これは実装が入った後も残します 粒子-粒子だけに戻せば同じ盲点が戻ることの証明です
//!
//! # 実測 (2026-09-29 = **粒子-粒子だけだった頃**、`Fix128` のみなので決定論、再実行で同値)
//!
//! ⚠️ 以下の表は **landing 前**の値です 現在の実装では `CRUMPLE_STEPS = 120` /
//! 半径 0.05 で貫通は **0** になります (自己接触 OFF は 7 のまま)
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
//! | 貫通回数 (自己接触 ON) | 3 | **20** | **1** | 3 | 7 |
//! | 貫通回数 (自己接触 OFF) | 3 | **13** | **1** | 3 | 7 |
//!
//! ⚠️ **`CRUMPLE_STEPS` を触る時に単調だと思わないでください** 現在 guard が要求するのは
//! **自己接触 OFF 側**の貫通 (`crossings_off > 0`) で、上の表の OFF 行の最小は
//! `steps = 60` の **1 回**です つまり 60 は前提が崩れる寸前で、`steps = 30` は貫通 13 回と
//! 数は多いものの **非単調な曲線の 1 点なので「余裕が 13 倍」にはなりません**
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

/// 2 つの位置列が最初に食い違う index と、その 1 要素だけを返す
///
/// ⚠️ `assert_eq!` / `assert_ne!` に `Vec<Vec3Fix>` をそのまま渡すと、失敗時に
/// **両側の `Debug` が丸ごと出て** (81 粒子 x 3 軸 x 2 side = 24 KB 実測) 書き手の
/// 失敗 message が埋まり、どこが違うのかも読めません (罠 `assert-eq-dumps-large-debug`、
/// `3f4a00e` が `tests/mesh_quality.rs` で一度潰した事象の再発)
/// **反転して `assert_ne!` 相当になった後も、どこが動いたかを出すために要ります**
/// (現在は `the_particle_pair_minimum_is_attained_by_two_pinned_vertices` の失敗時に、
/// 境界でなくなった対の座標を出すのに使っています)
fn first_difference(a: &[Vec3Fix], b: &[Vec3Fix]) -> Option<(usize, Vec3Fix, Vec3Fix)> {
    a.iter()
        .zip(b.iter())
        .position(|(x, y)| x != y)
        .map(|i| (i, a[i], b[i]))
}

/// `Vec3Fix` を 1 行で (`Debug` は hi/lo の生値なので読めない)
fn brief(v: Vec3Fix) -> String {
    let (x, y, z) = v.to_f32();
    format!("({x:.6}, {y:.6}, {z:.6})")
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

/// 線分-線分の最小距離² — **計測用の独立実装**
///
/// `src/cloth.rs` の `closest_points_on_segments` は Ericson §5.1.9 の
/// 「素の解を clamp → `t` を解き直す」形式ですが、こちらは **4 つの端点-線分 clamp 投影 +
/// (両パラメータが開区間に入る時だけ) 内部解** の 5 候補を全部計算して min を取る別形式です
///
/// ⚠️ **実装と違い、平行な対でも答えを返します** 実装側は平行を `None` で捨てます
/// (最小が区間になるので `(s,t)` が幾何で決まらない) が、**計測はその対も測ります** —
/// 計器が実装の盲点に合わせて盲目になってはいけないので、
/// `the_segment_segment_metric_matches_the_closed_form` の (d) で平行の値を pin します
///
/// ⚠️ 独立実装は独立に誤りうるので、閉形式が自明な 5 配置との厳密一致を同 test が固定します
fn segment_segment_distance_squared(p1: Vec3Fix, q1: Vec3Fix, p2: Vec3Fix, q2: Vec3Fix) -> Fix128 {
    fn point_segment_d2(p: Vec3Fix, a: Vec3Fix, b: Vec3Fix) -> Fix128 {
        let ab = b - a;
        let den = ab.length_squared();
        if den.is_zero() {
            return (p - a).length_squared();
        }
        let mut t = (p - a).dot(ab) / den;
        if t < Fix128::ZERO {
            t = Fix128::ZERO;
        }
        if t > Fix128::ONE {
            t = Fix128::ONE;
        }
        (p - (a + ab * t)).length_squared()
    }

    let mut best = point_segment_d2(p1, p2, q2);
    for d in [
        point_segment_d2(q1, p2, q2),
        point_segment_d2(p2, p1, q1),
        point_segment_d2(q2, p1, q1),
    ] {
        if d < best {
            best = d;
        }
    }

    // 内部解は正規方程式を直接解く (clamp でなく、開区間に入ったかで採否を決める)
    //   a·s − b·t = −c       s = (b f − c e)/(a e − b²)
    //   b·s − e·t = −f       t = (a f − b c)/(a e − b²)
    let (d1, d2) = (q1 - p1, q2 - p2);
    let r = p1 - p2;
    let (a, e, b) = (d1.dot(d1), d2.dot(d2), d1.dot(d2));
    let den = a * e - b * b;
    if !den.is_zero() {
        let (c, f) = (d1.dot(r), d2.dot(r));
        let s = (b * f - c * e) / den;
        let t = (a * f - b * c) / den;
        if s > Fix128::ZERO && s < Fix128::ONE && t > Fix128::ZERO && t < Fix128::ONE {
            let d = ((p1 + d1 * s) - (p2 + d2 * t)).length_squared();
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// 計器の契約 test — 上の独立実装が 5 配置で閉形式と厳密一致する
///
/// | | 線分 A | 線分 B | 最近接 | 距離² |
/// |---|---|---|---|---|
/// | (a) 内部-内部 | `(0,0,0)→(1,0,0)` | `(1/2,1/4,-1)→(1/2,1/4,1)` | 両中点 | `1/16` |
/// | (b) 端点-内部 | `(0,0,0)→(1,0,0)` | `(2,1,-1)→(2,1,1)` | `(1,0,0)` と `(2,1,0)` | `2` |
/// | (c) 端点-端点 | `(0,0,0)→(1,0,0)` | `(2,1,2)→(2,1,3)` | `(1,0,0)` と `(2,1,2)` | `6` |
/// | (d) 平行 | `(0,0,0)→(1,0,0)` | `(0,1,0)→(1,1,0)` | 任意の対応点 | `1` |
/// | (e) 交差 | `(-1,0,0)→(1,0,0)` | `(0,0,-1)→(0,0,1)` | 原点 | `0` |
///
/// (b)(c) は端点領域、(d) は実装が `None` で捨てる領域、(e) は距離 0 の退化です
#[test]
fn the_segment_segment_metric_matches_the_closed_form() {
    let half = Fix128::from_ratio(1, 2);
    let quarter = Fix128::from_ratio(1, 4);
    let a0 = Vec3Fix::from_int(0, 0, 0);
    let a1 = Vec3Fix::from_int(1, 0, 0);

    assert_eq!(
        segment_segment_distance_squared(
            a0,
            a1,
            Vec3Fix::new(half, quarter, Fix128::from_int(-1)),
            Vec3Fix::new(half, quarter, Fix128::ONE)
        ),
        Fix128::from_ratio(1, 16),
        "oracle (a): 内部-内部、垂線 1/4 なので距離² = 1/16"
    );
    assert_eq!(
        segment_segment_distance_squared(
            a0,
            a1,
            Vec3Fix::from_int(2, 1, -1),
            Vec3Fix::from_int(2, 1, 1)
        ),
        Fix128::from_int(2),
        "oracle (b): 端点 (1,0,0) から (2,1,0) まで 1² + 1² = 2"
    );
    assert_eq!(
        segment_segment_distance_squared(
            a0,
            a1,
            Vec3Fix::from_int(2, 1, 2),
            Vec3Fix::from_int(2, 1, 3)
        ),
        Fix128::from_int(6),
        "oracle (c): 端点どうし 1² + 1² + 2² = 6"
    );
    assert_eq!(
        segment_segment_distance_squared(
            a0,
            a1,
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(1, 1, 0)
        ),
        Fix128::ONE,
        "oracle (d): 平行、距離² = 1 (実装が None で捨てる領域も計器は測る)"
    );
    assert_eq!(
        segment_segment_distance_squared(
            Vec3Fix::from_int(-1, 0, 0),
            a1,
            Vec3Fix::from_int(0, 0, -1),
            Vec3Fix::from_int(0, 0, 1)
        ),
        Fix128::ZERO,
        "oracle (e): 原点で交差するので距離² = 0"
    );
}

/// **構造的事実** — X 字交差は頂点-面の計器から原理的に見えない (閉形式、恒久に真)
///
/// ```text
///   T0 = [0,1,2]   0 = (-1,0,0)   1 = (1,0,0)   2 = (0,-8,0)      (XY 平面)
///   T1 = [3,4,5]   3 = (0,h,-1)   4 = (0,h,1)   5 = (0,h+8,0)     (YZ 平面)   h = 1/8
/// ```
///
/// 辺 `(0,1)` と辺 `(3,4)` は上から見て直交して交差し、距離は `h = 1/8` です 一方
/// **非接続な (頂点, 三角形) 対の最小距離² は `1 + h² = 65/64`** で、`thickness = 1/4`
/// に対して `65/64 ≫ 1/16 = thickness²` なので頂点-面の対は 1 つも閾値に入りません
///
/// ⚠️ **これは「パラメータが小さいから見えない」ではありません** 検出半径を上げて
/// `65/64` を超えさせると、同じ半径で**面内の普通の対が全部違反になる**ので使用条件の外に
/// 出ます (`a_thickness_far_above_the_local_edge_length_is_outside_the_model` と同じ理由)
/// 頂点-面の述語は**辺どうしの交差という事象そのものを持っていません**
///
/// `particle_distance_spring_cannot_see_the_closed_form_crossing` (粒子-粒子 ⊂ 頂点-面) の
/// 1 段上の盲点で、どちらも実装が入った後も残します 辺-辺の段を外せば同じ盲点が戻ります
#[test]
fn the_vertex_face_metric_cannot_see_the_closed_form_edge_edge_crossing() {
    let h = Fix128::from_ratio(1, 8);
    let eight = Fix128::from_int(8);
    let positions = [
        Vec3Fix::from_int(-1, 0, 0),
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO - eight, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, h, Fix128::from_int(-1)),
        Vec3Fix::new(Fix128::ZERO, h, Fix128::ONE),
        Vec3Fix::new(Fix128::ZERO, h + eight, Fix128::ZERO),
    ];
    for (i, v) in positions.iter().enumerate() {
        assert_exact_triple_products(*v, &format!("p{i}"));
    }
    let triangles = [[0usize, 1, 2], [3, 4, 5]];
    let thickness = Fix128::from_ratio(1, 4);

    // 辺-辺: 交差する 2 辺の距離² は h² = 1/64 で、閾値 (1/16) の内側
    let ee_d2 =
        segment_segment_distance_squared(positions[0], positions[1], positions[3], positions[4]);
    assert_eq!(
        ee_d2,
        Fix128::from_ratio(1, 64),
        "oracle: 辺-辺の距離² = h² = 1/64"
    );
    assert!(
        ee_d2 < thickness * thickness,
        "この scene が辺-辺の接触になっていない"
    );

    // 頂点-面: 非接続な全対の最小距離² は 1 + h² = 65/64 で、閾値の外側
    let (vf_min, vf_viol) = vertex_face_separation(&positions, &triangles, thickness);
    assert_eq!(
        vf_min,
        Fix128::from_ratio(65, 64),
        "oracle: 非接続な (頂点, 三角形) 対の最小距離² = 1 + h² = 65/64 \
         (最近接は頂点 0 / 1 から辺 (3,4) の (0,h,0)、および頂点 3 / 4 から辺 (0,1) の原点)"
    );
    assert_eq!(
        vf_viol, 0,
        "頂点-面の対が閾値を下回った この scene は盲点を示していない"
    );
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
// D. 自己交差する scene — 目標 oracle と、それが効いていることの guard
// ---------------------------------------------------------------------------

/// 9x9 の布の境界を中心へ縮めて座屈・crumple させる
///
/// 対称性は `RNG` でなく **決定的な 1/8 の持ち上げ**で破る (lockstep 規律)
fn run_crumple(
    self_collision: bool,
    steps: usize,
    self_collision_distance: Fix128,
) -> (usize, usize, Vec<Vec3Fix>) {
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
    cloth.config.self_collision_distance = self_collision_distance;
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
    // The crate's own invariant, read through the public API. Kept **alongside** the
    // independent loop below rather than replacing it: the loop is a reimplementation of
    // the vertex-face half and is what makes a `0` from `remaining_self_contact_crossings`
    // mean something, while the public count is the only one that also sees edge-edge.
    let mut invariant = 0usize;
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
        invariant += cloth.remaining_self_contact_crossings(&before);
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
    (crossings, invariant, cloth.positions)
}

const CRUMPLE_STEPS: usize = 120;

/// `run_crumple` の既定の検出半径 (module doc の実測表の 0.05 の列)
///
/// ⚠️ **この値は「良い数字が出るから」ではなく、使用条件から選ばれています** 60 step
/// 時点の mesh 辺長は `shrink(59, 60) = 1 − (7/8)(59/60) = 0.1396` で、0.05 はその
/// **36%** = `src/cloth.rs` が doc に書いた「接触厚は局所辺長の半分未満」の内側です
/// 72% (0.1) と 360% (0.5) が破綻することは
/// `a_thickness_far_above_the_local_edge_length_is_outside_the_model` が測っています
fn default_radius() -> Fix128 {
    Fix128::from_ratio(5, 100)
}

/// 分離の計測に使う step 数 — `CRUMPLE_STEPS` とは別に選んである
const SEPARATION_STEPS: usize = 60;

/// 点 `p` と三角形 `(a, b, c)` の最小距離² — **計測用の独立実装**
///
/// `src/cloth.rs` の `closest_point_on_triangle` は Voronoi 領域を符号判定で **1 つ選ぶ**
/// 形式ですが、こちらは **3 辺への clamp 投影 + 面内投影の 4 候補を全部計算して min を取る**
/// 別形式です (頂点領域は辺の端点への clamp に含まれるので、7 領域が 4 候補で覆えます)
///
/// ⚠️ **実装を呼ばないだけでは計器の正しさは担保されません** 独立実装は独立に誤りうるので、
/// 閉形式が自明な 3 配置 (面 / 辺 / 頂点の各領域) との厳密一致を
/// `the_point_triangle_metric_matches_the_closed_form` が固定します
fn point_triangle_distance_squared(p: Vec3Fix, a: Vec3Fix, b: Vec3Fix, c: Vec3Fix) -> Fix128 {
    fn segment_d2(p: Vec3Fix, a: Vec3Fix, b: Vec3Fix) -> Fix128 {
        let ab = b - a;
        let den = ab.length_squared();
        if den.is_zero() {
            return (p - a).length_squared();
        }
        let mut t = (p - a).dot(ab) / den;
        if t < Fix128::ZERO {
            t = Fix128::ZERO;
        }
        if t > Fix128::ONE {
            t = Fix128::ONE;
        }
        (p - (a + ab * t)).length_squared()
    }

    let mut best = segment_d2(p, a, b);
    for d in [segment_d2(p, b, c), segment_d2(p, c, a)] {
        if d < best {
            best = d;
        }
    }

    // 面内投影は重心座標が三角形の中に落ちた時だけ候補になる
    let (e1, e2) = (b - a, c - a);
    let (d11, d12, d22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
    let det = d11 * d22 - d12 * d12;
    if !det.is_zero() {
        let ap = p - a;
        let (b1, b2) = (ap.dot(e1), ap.dot(e2));
        let u = (b1 * d22 - b2 * d12) / det;
        let v = (b2 * d11 - b1 * d12) / det;
        if u >= Fix128::ZERO && v >= Fix128::ZERO && u + v <= Fix128::ONE {
            let d = (p - (a + e1 * u + e2 * v)).length_squared();
            if d < best {
                best = d;
            }
        }
    }
    best
}

/// 計器の契約 test — 上の brute force が閉形式と厳密に一致する
///
/// 三角形は section A と同じ `a = (0,0,0)`, `b = (1,0,0)`, `c = (0,0,1)` (XZ 平面)
/// 3 領域それぞれで手計算した距離² を pin します (全て 2 進小数なので許容差なし)
///
/// | 領域 | 点 | 最近接点 | 距離² |
/// |---|---|---|---|
/// | 面 | `(1/4, 1/2, 1/4)` | `(1/4, 0, 1/4)` | `1/4` |
/// | 辺 `bc` | `(1, 1/2, 1)` | `(1/2, 0, 1/2)` | `3/4` |
/// | 頂点 `a` | `(-1/2, 1/2, -1/2)` | `(0, 0, 0)` | `3/4` |
#[test]
fn the_point_triangle_metric_matches_the_closed_form() {
    let a = Vec3Fix::from_int(0, 0, 0);
    let b = Vec3Fix::from_int(1, 0, 0);
    let c = Vec3Fix::from_int(0, 0, 1);
    let h = Fix128::from_ratio(1, 2);
    let q = Fix128::from_ratio(1, 4);

    // 面領域: 垂線の足が三角形の中 (u = v = 1/4、u + v = 1/2 < 1)
    assert_eq!(
        point_triangle_distance_squared(Vec3Fix::new(q, h, q), a, b, c),
        Fix128::from_ratio(1, 4),
        "oracle: 面への垂線 1/2 なので距離² = 1/4"
    );

    // 辺 bc 領域: 平面への投影 (1, 0, 1) は u + v = 2 > 1 で三角形の外
    assert_eq!(
        point_triangle_distance_squared(Vec3Fix::new(Fix128::ONE, h, Fix128::ONE), a, b, c),
        Fix128::from_ratio(3, 4),
        "oracle: 斜辺の中点 (1/2, 0, 1/2) までの距離² = (1/2)² x 3 = 3/4"
    );

    // 頂点 a 領域: ab·ap = ac·ap = -1/2 < 0
    let neg_h = Fix128::ZERO - h;
    assert_eq!(
        point_triangle_distance_squared(Vec3Fix::new(neg_h, h, neg_h), a, b, c),
        Fix128::from_ratio(3, 4),
        "oracle: 頂点 a までの距離² = (1/2)² x 3 = 3/4"
    );

    // 負の control: 三角形の上に載った点は距離 0 (4 候補の min が 0 に潰れる)
    assert_eq!(
        point_triangle_distance_squared(Vec3Fix::new(q, Fix128::ZERO, q), a, b, c),
        Fix128::ZERO,
        "面上の点の距離² は 0"
    );
}

/// 非接続な (頂点, 三角形) 対の最小距離² と、閾値を下回る対の数
///
/// **これが `src/cloth.rs` の自己接触が実際に拘束している量です** 頂点-頂点の距離
/// (`nonadjacent_separation`) ではありません
fn vertex_face_separation(
    positions: &[Vec3Fix],
    triangles: &[[usize; 3]],
    threshold: Fix128,
) -> (Fix128, usize) {
    let t2 = threshold * threshold;
    let mut min_d2 = Fix128::from_int(1 << 20);
    let mut violations = 0usize;
    for tri in triangles {
        for (i, p) in positions.iter().enumerate() {
            if shares_vertex(i, *tri) {
                continue;
            }
            let d2 = point_triangle_distance_squared(
                *p,
                positions[tri[0]],
                positions[tri[1]],
                positions[tri[2]],
            );
            if d2 < min_d2 {
                min_d2 = d2;
            }
            if d2 < t2 {
                violations += 1;
            }
        }
    }
    (min_d2, violations)
}

/// **目標 oracle** — 布は自分自身を通り抜けてはならない
///
/// 2026-10-01 に `src/cloth.rs` へ頂点-面の自己接触が入って green になりました
/// (substep 内の近接斥力 + frame 単位の掃過線分復元、どちらも Jacobi 蓄積)
///
/// # Before (2026-09-29、粒子-粒子の距離ばねだけだった頃)
///
/// 同じ scene で **貫通 7 回** 粒子どうしの距離だけを見る判定は三角形の内部を通り抜ける
/// 頂点を原理的に見られないので、検出半径を閉形式の「交点-最近傍頂点 `√(1/8) ≈ 0.3536`」
/// より大きくしても貫通が残り、半径に対して単調ですらありませんでした
/// (`0.05 → 7` / `0.25 → 5` / `0.5 → 4` / `1.0 → 5`)
/// この構造的事実自体は `particle_distance_spring_cannot_see_the_closed_form_crossing`
/// が閉形式で pin し続けています
#[test]
fn a_crumpled_cloth_does_not_pass_through_itself() {
    let (crossings, invariant, _) = run_crumple(true, CRUMPLE_STEPS, default_radius());
    assert_eq!(
        crossings, 0,
        "自己交差する crumple scene で頂点が三角形を {crossings} 回貫いた"
    );
    // 1.5.x で追加 上の独立実装は頂点-面しか見ないので、辺-辺の貫通はここでしか出ません
    // (実装前の実測: 同条件で頂点-面 0 件のまま辺-辺が 22 件残っていた)
    assert_eq!(
        invariant, 0,
        "自己交差する crumple scene で自己貫通が {invariant} 件残った \
         (頂点-面 / 辺-辺 の合算、頂点-面の独立計数は 0)"
    );
}

/// **guard** — 自己交差する scene で、自己接触の on/off が結果を変える
///
/// 目標 oracle が green になったことを「自己接触が実際に効いている」の証拠にするには、
/// **切ったら壊れる**ことを別に測る必要があります (切っても同じなら、green は scene が
/// 自己交差しないことの言い換えでしかない)
///
/// # Before (2026-09-29)
///
/// 旧名は `self_collision_toggle_changes_nothing_in_a_scene_that_self_intersects` で、
/// 主張は正反対でした: `pos_on` と `pos_off` が **bit 一致** し、前提は `crossings_on > 0`
/// (貫通 7 回) 粒子-粒子の距離ばねはこの scene で一度も発火しなかったからです
/// 実装が入って `pos_on == pos_off` も `crossings_on > 0` も偽になったので、doc の指定
/// どおり改名 + assert 反転 + 前提を `crossings_off > 0` に差し替えました
///
/// ⚠️ **`assert_ne!(pos_on, pos_off)` にはしないこと** 失敗時に 81 粒子 x 2 side の
/// `Debug` が 24 KB 出て message が埋まります (罠 `assert-eq-dumps-large-debug`)
#[test]
fn self_collision_toggle_changes_the_result_in_a_scene_that_self_intersects() {
    let (_, _, pos_on) = run_crumple(true, CRUMPLE_STEPS, default_radius());
    let (crossings_off, _, pos_off) = run_crumple(false, CRUMPLE_STEPS, default_radius());

    // この scene が判別に向いていることを先に確かめる (自己接触を切っても自己交差しない
    // scene では、on/off が違うことを示しても何も言っていない)
    assert!(
        crossings_off > 0,
        "自己接触を切っても scene が自己交差しない この guard は判別に向かない scene では無意味"
    );
    assert!(
        first_difference(&pos_on, &pos_off).is_some(),
        "自己接触の on/off で位置が 1 bit も変わらない = 投影が発火していない \
         目標 oracle の green は scene が自己交差しないことの言い換えになっている"
    );
}

/// **guard** — 自己接触は、自分が拘束している量 (頂点-面距離) を実際に改善する
///
/// # 何を測っているか、なぜ量を変えたか
///
/// `src/cloth.rs` が拘束しているのは **非接続な (頂点, 三角形) 対の距離**です
/// この test は `SEPARATION_STEPS = 60` 時点で、自己接触の on/off で
/// **最小の頂点-面距離²** が改善することを assert します
///
/// ⚠️ **違反件数 (`threshold` 未満の対の数) は assert しません** 実測で半径に対して
/// 単調でないからです (60 step、ON / OFF):
///
/// | 半径 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 |
/// |---|---|---|---|---|---|---|
/// | 違反件数 ON | 0 | **6** | 3 | 3 | 2 | 43 |
/// | 違反件数 OFF | 1 | **2** | 3 | 4 | 5 | 11 |
/// | 最小距離² ON | 5.82e-4 | 1.57e-4 | 1.13e-3 | 8.64e-4 | 3.05e-3 | 1.26e-4 |
/// | 最小距離² OFF | 6.08e-5 | 6.08e-5 | 6.08e-5 | 6.08e-5 | 6.08e-5 | 6.08e-5 |
///
/// **件数は 0.03 で逆転し 0.04 で同点**になるので、`viol_on < viol_off` を assert すると
/// 「差が出る半径を選んだ」だけになります (`3 < 4` は margin 1 で、些細な変更で反転する)
/// 一方 **最小距離² は 6 半径すべてで ON が上**で、採用点の 0.05 では **14 倍**あります
/// だから主張の本体は最小距離² 側が持ち、件数は doc の記録に留めます
///
/// # Before (2026-09-29、旧名 `self_collision_moves_particles_without_improving_their_separation`)
///
/// 旧 test は **半径 0.5** で **頂点-頂点** (非連結粒子対) の最小距離を比べ、
/// 「投影は発火するが分離は 1 ミリも改善しない」(`min_on == min_off = 0.197401`、
/// `viol_on == viol_off = 74`) を assert していました 実装後も **その等値は成立します**
/// が、それは欠陥が残っているからではありません:
///
/// 1. ⚠️ **旧計器は solver でなく test 自身の driver を測っていました** 最小値を取る対は
///    4 run すべて `(7, 17)` = **両方とも pin された境界粒子**で、`run_crumple` が毎 step
///    `positions[i].x/.z` を代入している座標です `0.038967 = (0.1396·√2)²` は
///    `shrink(59, 60)` から決まる定数で、solver が何をしても動きません
///    この事実は `the_particle_pair_minimum_is_attained_by_two_pinned_vertices` が
///    引き続き assert します (旧数値はそこで生き続けます)
/// 2. ⚠️ **旧 radius 0.5 は破綻域でした** 60 step 時点の辺長 0.1396 の **360%** で、
///    「全ての非接続対を 0.5 離す」は 8x8 の材料を 1x1 の枠に収めた状態では幾何的に
///    充足できません 実測でも自己接触を入れると貫通が 1 → 651 に増えます
///    (`a_thickness_far_above_the_local_edge_length_is_outside_the_model`)
///
/// # ⚠️ この test を消すと近接斥力段が無防備になる (2026-10-01 破壊試験で実測)
///
/// `src/cloth.rs` の自己接触は **2 段**です (substep 内の近接斥力 + frame 単位の掃過線分
/// 復元) 変異を 1 つずつ入れて測った結果:
///
/// | 変異 | `a_crumpled_cloth_does_not_pass_through_itself` | 本 test |
/// |---|---|---|
/// | frame 単位の CCD 復元段を落とす | **red** | red |
/// | **近接斥力段を落とす** | **green のまま** | **red** |
///
/// ⚠️ **CCD 段だけで貫通 0 には到達します** つまり目標 oracle は近接斥力段を守っていません
/// **近接斥力段の歯は本 test だけ**です 第 3 の pin を頂点-頂点から頂点-面に測り直さなければ、
/// 近接斥力段はどの test からも守られていない状態になっていました
/// **「目標 oracle があるから冗長」と判断して消さないこと**
///
/// つまり本 test は **(a) 測る量** (頂点-頂点 → 頂点-面) と **(b) scene の半径**
/// (0.5 → 0.05) の **2 つ**を変えています どちらも旧 test の前提が実測で偽だったことが
/// 理由で、良い数字の出る条件へ逃げたのではありません 旧条件の数値と、それが何を
/// 測っていたのかは上の 2 点として残してあります
///
/// # 追記 2026-10-01 — 辺-辺の段が入って数値が変わりました (上の表は頂点-面だけだった頃)
///
/// | 半径 0.05、60 step | 頂点-面だけ | **辺-辺も (現在)** |
/// |---|---|---|
/// | 最小距離² ON | 8.64e-4 | **2.134e-3** |
/// | 最小距離² OFF | 6.08e-5 | 6.08e-5 (不変) |
/// | 違反件数 ON / OFF | 3 / 4 | **2** / 4 |
///
/// ⚠️ **辺-辺を足すと、頂点-面の分離も良くなります** (14 倍 → 35 倍) 直感に反しますが、
/// 辺どうしが交差する手前で止まる分だけ頂点が面に押し付けられる状況が減るためです
///
/// ⚠️⚠️ **ただしそれは 2 段を 1 本の `Δ`/`hits` buffer に統合した場合だけです** 段ごとに
/// 別 buffer で平均化した最初の配線では、同じ scene で **ON 1.79e-5 / 違反 21 件** =
/// **自己接触 ON が OFF より悪い**状態になりました (頂点-面 5 件 + 辺-辺 1 件に触られた頂点が
/// `(Σ_vf)/5 + (Σ_ee)/1` を受けて辺-辺が 5 倍過大評価される) **本 test がそれを捕まえた
/// 唯一の test です** 目標 oracle の貫通数は 0 のままでした
/// 詳細 `src/cloth.rs` の `solve_self_collision` の doc と
/// `[[feedback_cloth_edge_edge_two_jacobi_passes_compete]]`
#[test]
fn self_collision_improves_the_vertex_face_separation_it_constrains() {
    let radius = default_radius();
    let (_, _, pos_on) = run_crumple(true, SEPARATION_STEPS, radius);
    let (_, _, pos_off) = run_crumple(false, SEPARATION_STEPS, radius);

    // 三角形の位相は scene に依らないので、どちらの run から取っても同じ
    let cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        9,
        9,
        Fix128::from_ratio(1, 100),
    );
    let (min_on, viol_on) = vertex_face_separation(&pos_on, &cloth.triangles, radius);
    let (min_off, viol_off) = vertex_face_separation(&pos_off, &cloth.triangles, radius);

    // 反 vacuous: 自己接触を切った側に違反が 1 件も無ければ、改善も何も言っていない
    assert!(
        viol_off > 0,
        "自己接触 OFF でも閾値未満の頂点-面対が無い この guard は違反の起きない scene では無意味"
    );
    // 主張の本体: 最小の頂点-面距離² が 4 倍以上に開く (実測 14 倍、ON {min_on} / OFF {min_off})
    assert!(
        min_on > min_off * Fix128::from_int(4),
        "自己接触が最小の頂点-面距離² を 4 倍に開いていない (ON {} / OFF {}、違反件数 {viol_on} / {viol_off})",
        min_on.to_f32(),
        min_off.to_f32()
    );
}

/// **旧計器が何を測っていたかの pin** — 頂点-頂点の最小距離は pin された境界対が取る
///
/// 旧 test `self_collision_moves_particles_without_improving_their_separation` が
/// 実装前も実装後も green だった理由です 最小値を取る対は `(7, 17)` =
/// grid の `(row 0, col 7)` と `(row 1, col 8)` で、三角形分割で辺にならない側の対角
/// なので「非連結」に残りますが、**両方とも `inv_mass = 0` の境界粒子**で、
/// `run_crumple` が毎 step 座標を代入しています
///
/// ⚠️ **極値を assert する test は「極値を取る要素が実験条件で動くか」を別に確かめること**
/// 件数の反 vacuous check (`viol_off > 0`) を通しても、argmin が入力側に固定されていれば
/// 等値 assert は恒久に green です 自由度が固定された要素 (pin / 境界条件 / kinematic
/// driver) は solver の**出力ではなく入力**なので、効果を測る母集団に混ぜてはいけません
///
/// 旧実測 (2026-09-29、半径 0.5): `min_d2 = 0.038967` (= `0.197401²`)、違反 74 件、
/// 自己接触の on/off で両方とも完全一致
#[test]
fn the_particle_pair_minimum_is_attained_by_two_pinned_vertices() {
    let radius = default_radius();
    let (_, _, pos_on) = run_crumple(true, SEPARATION_STEPS, radius);
    let (_, _, pos_off) = run_crumple(false, SEPARATION_STEPS, radius);

    let cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        9,
        9,
        Fix128::from_ratio(1, 100),
    );
    let (min_on, _, pair_on) = nonadjacent_separation(&pos_on, &cloth.triangles, radius);
    let (min_off, _, pair_off) = nonadjacent_separation(&pos_off, &cloth.triangles, radius);

    const RES: usize = 9;
    let is_boundary = |i: usize| {
        let (r, c) = (i / RES, i % RES);
        r == 0 || c == 0 || r == RES - 1 || c == RES - 1
    };

    for (pair, pos, side) in [(pair_on, &pos_on, "ON"), (pair_off, &pos_off, "OFF")] {
        assert!(
            is_boundary(pair.0) && is_boundary(pair.1),
            "{side}: 頂点-頂点の最小対 {pair:?} が境界対でなくなった ({} / {}) \
             solver が動かせる対が最小を取るようになったなら、旧計器を復活させてよい",
            brief(pos[pair.0]),
            brief(pos[pair.1])
        );
    }
    assert_eq!(
        pair_on, pair_off,
        "on/off で最小対が違う = 最小対が driver 由来でなくなった"
    );
    assert_eq!(
        min_on, min_off,
        "pin された対どうしの距離が on/off で違う = driver の外の何かが境界を動かしている"
    );
}

/// **使用条件の pin** — 接触厚が局所辺長を大きく超えると、保証そのものが成り立たない
///
/// `SEPARATION_STEPS = 60` 時点の mesh 辺長は `shrink(59, 60) = 0.1396` です
/// 半径 0.5 はその **360%** で、「全ての非接続 (頂点, 三角形) 対を 0.5 離す」は
/// 8x8 の材料を 1x1 の枠に収めた状態では幾何的に充足できません 充足不能な拘束集合を
/// 投影し続けると補正どうしが打ち消し合って粒子が飛び、**自己接触を入れた方が貫通が増えます**
///
/// ⚠️ **「厚くすれば安全」は逆です** これは `src/cloth.rs` の doc が書いている使用条件
/// (接触厚は局所辺長の半分未満) を実測で pin する test で、`ClothConfig` に実行時の
/// guard を入れるかは別判断として Backlog にあります
///
/// ⚠️ **これは欠陥が続くことを assert する test です** rigid impact zone 等で破綻域まで
/// 扱えるようになったら **red になるのが正しい挙動**で、その時は削除せず不等号を逆に
/// 書き換えてください 旧実測 (ON 651 / OFF 1) は `# Before` として残すこと
#[test]
fn a_thickness_far_above_the_local_edge_length_is_outside_the_model() {
    let out_of_range = Fix128::from_ratio(1, 2);
    let (crossings_on, _, _) = run_crumple(true, SEPARATION_STEPS, out_of_range);
    let (crossings_off, _, _) = run_crumple(false, SEPARATION_STEPS, out_of_range);

    // 反 vacuous: OFF 側で自己交差が起きない scene なら「増えた」は何も言っていない
    assert!(
        crossings_off > 0,
        "自己接触 OFF でこの scene が自己交差しない 破綻域の比較にならない"
    );
    assert!(
        crossings_on > crossings_off,
        "厚 0.5 (辺長の 360%) が破綻域でなくなった (ON {crossings_on} / OFF {crossings_off}) \
         破綻域を扱えるようになったなら不等号を逆に書き換えること"
    );
}

/// 頂点を共有しない mesh 辺対の最小距離² と、閾値を下回る対の数
///
/// **これが辺-辺の段が実際に拘束している量です** 頂点-面距離 (`vertex_face_separation`) でも
/// 頂点-頂点距離 (`nonadjacent_separation`) でもありません
///
/// ⚠️ 頂点を共有する対は距離 0 が構成上の事実なので母集団から外します (実装側の
/// `collect_edge_edge_candidates` と同じ除外) ⚠️ **平行な対は外しません** 実装は平行を
/// 捨てますが、計器が同じ盲点を持つと「捨てた分は測られない」ことになります
fn edge_edge_separation(
    positions: &[Vec3Fix],
    triangles: &[[usize; 3]],
    threshold: Fix128,
) -> (Fix128, usize, (usize, usize)) {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for t in triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();

    let t2 = threshold * threshold;
    let mut min_d2 = Fix128::from_int(1 << 20);
    let mut argmin = (usize::MAX, usize::MAX);
    let mut violations = 0usize;
    for (i, &(a0, a1)) in edges.iter().enumerate() {
        for &(b0, b1) in &edges[i + 1..] {
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                continue; // 頂点を共有する対は距離 0 が構成上の事実
            }
            let d2 = segment_segment_distance_squared(
                positions[a0],
                positions[a1],
                positions[b0],
                positions[b1],
            );
            if d2 < min_d2 {
                min_d2 = d2;
                argmin = (i, usize::MAX);
            }
            if d2 < t2 {
                violations += 1;
            }
        }
    }
    (min_d2, violations, argmin)
}

/// **guard** — 辺-辺の段は、自分が拘束している量 (辺-辺距離) を実際に改善する
///
/// `self_collision_improves_the_vertex_face_separation_it_constrains` の辺-辺版です
/// 頂点-面の段が入った時に「近接斥力段の歯はこの test だけ」になったのと同じ理由で、
/// **辺-辺の段を守るのはこの test だけ**です (目標 oracle `a_crumpled_cloth_...` は
/// 頂点の掃過線分しか見ないので、辺-辺の段を落としても green のままです)
///
/// # 実測 (`SEPARATION_STEPS = 60`、半径 0.05、`Fix128` のみなので再実行で同値)
///
/// | | 自己接触 ON | OFF |
/// |---|---|---|
/// | 最小の辺-辺距離² | **5.225e-3** | 1.550e-3 |
/// | 閾値 (0.05) 未満の辺対 | **0** | 3 |
///
/// 件数 0 は「全ての辺対が接触厚まで離れた」= 段が約束している不変量そのものですが、
/// ⚠️ **assert は `viol_on < viol_off` (margin 3) と `min_on > 2·min_off` (実測 3.37 倍)
/// に置いています** 収束を等値で assert すると反復回数や substep 数の無関係な変更で
/// 折れるので、0 は記録に留めます 件数 0 が崩れたら**弱めずに原因を調べること**
#[test]
fn self_collision_improves_the_edge_edge_separation_it_constrains() {
    let radius = default_radius();
    let (_, _, pos_on) = run_crumple(true, SEPARATION_STEPS, radius);
    let (_, _, pos_off) = run_crumple(false, SEPARATION_STEPS, radius);
    let cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(8),
        Fix128::from_int(8),
        9,
        9,
        Fix128::from_ratio(1, 100),
    );
    let (min_on, viol_on, _) = edge_edge_separation(&pos_on, &cloth.triangles, radius);
    let (min_off, viol_off, _) = edge_edge_separation(&pos_off, &cloth.triangles, radius);

    // 反 vacuous: 自己接触を切った側に違反が 1 件も無ければ、改善も何も言っていない
    assert!(
        viol_off > 0,
        "自己接触 OFF でも閾値未満の辺対が無い この guard は違反の起きない scene では無意味"
    );
    assert!(
        viol_on < viol_off,
        "辺-辺の段が違反件数を減らしていない (ON {viol_on} / OFF {viol_off}、実測は 0 / 3)"
    );
    assert!(
        min_on > min_off * Fix128::from_int(2),
        "辺-辺の段が最小の辺-辺距離² を 2 倍に開いていない (ON {} / OFF {}、実測 3.37 倍)",
        min_on.to_f32(),
        min_off.to_f32()
    );
}

/// 非連結対 (辺で繋がっていない粒子対) の最小距離²、閾値未満の対の数、**最小を取る対**
///
/// 辺集合は `triangles` から作る (`edge_constraints` は private なので同じものを再構成する)
///
/// ⚠️ 最小を取る対を返すのが本体です 値だけ返していた頃、その対が pin された境界粒子で
/// 固定であることが 2 週間見えませんでした
fn nonadjacent_separation(
    positions: &[Vec3Fix],
    triangles: &[[usize; 3]],
    threshold: Fix128,
) -> (Fix128, usize, (usize, usize)) {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for t in triangles {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.sort_unstable();
    edges.dedup();

    let mut min_d2 = Fix128::from_int(1 << 20);
    let mut argmin = (usize::MAX, usize::MAX);
    let mut violations = 0usize;
    let t2 = threshold * threshold;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            if edges.binary_search(&(i, j)).is_ok() {
                continue;
            }
            let d2 = (positions[j] - positions[i]).length_squared();
            if d2 < min_d2 {
                min_d2 = d2;
                argmin = (i, j);
            }
            if d2 < t2 {
                violations += 1;
            }
        }
    }
    (min_d2, violations, argmin)
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
