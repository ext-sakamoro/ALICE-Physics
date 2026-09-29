//! Oracle: `SdfTetMesh` の境界面抽出 — 立方体の表面は 6 面 × 2 三角形 = 12 枚
//!
//! # なぜ「1 回しか使われない面 = 境界面」で済まないか
//!
//! ⚠️ **その同一視が正しいのは mesh が適合的な場合だけです** hanging face も
//! 「1 回しか使われない面」です 実測 (2026-09-29、段階細分後の scene): `z = 0` の面が
//! 共有 0 / 単独使用 3 で、**この 3 面は全部内部**でした 素朴な抽出はこれを境界として
//! 返します
//!
//! したがって `boundary_faces()` は **適合的な mesh を前提とする** API であり、
//! 本 file はその前提が成り立つ入力 (`generate` の出力、適合性は
//! `tests/mesh_conformity.rs` が独立に実測済) でのみ閉形式と突き合わせます
//!
//! # ⚠️ なぜ本 file は境界判定に production API を使わないのか (消さないでください)
//!
//! 境界を**使用回数**で決めると、この oracle は検証したい API と**同じ仮定**の上に
//! 立ちます `boundary_faces()` の正しさは「mesh が適合的である」ことに依存しており、
//! その適合性を「1 回使われる面のうち内部にあるもの」で測るので、**API で census を
//! 実装すると API が hanging face を『境界』と呼び、census は『非適合ゼロ』と報告します**
//! 検出したい欠陥を、検出器が定義ごと消してしまう **循環**です
//!
//! だから本 file の境界判定は **別経路の幾何**で行います — 面の重心から法線方向に
//! ±0.25 cell 動かし、**SDF から独立に再計算した** meshed cell 集合に入るかを見ます
//! (`tests/mesh_conformity.rs` が同じ理由で同じ独立性を持っています)
//!
//! ⚠️ **これは `tests/mesh_conformity.rs` との意図的な重複です** 「重複だから production
//! API に寄せよう」という**正しく見える整理**をした瞬間に、上の循環が発生して検出力が
//! 消えます 複製の理由が書かれていない複製はいずれ統合されるので、ここに書いてあります
//!
//! # 閉形式 (出所: 5 分割の面勘定、下に導出を書く)
//!
//! `generate` は立方体 1 個を **`cube_to_five_tets` で 5 個の四面体**に分けます
//! (Kuhn の 6 分割ではありません) 面の総 slot は `5 × 4 = 20`:
//!
//! - 立方体の 6 つの正方形面がそれぞれ 2 三角形 → **境界面 12 枚** (各 1 回使用)
//! - 残り `20 − 12 = 8` slot が内部で、2 回ずつ使われる → **内部面 4 枚**
//!   (中央の正四面体の 4 面が、4 つの隅の四面体と 1 枚ずつ接する)
//!
//! `n × n × n` の塊では境界は立方体の表面だけなので **`6 n² × 2 = 12 n²` 枚**
//! `n = 2` なら 48 枚、内部面は `(8 × 20 − 48) / 2 = 56` 枚
//!
//! # 現在の状態
//!
//! ⚠️ 本 file は **`SdfTetMesh::boundary_faces()` が存在しない間は compile error で red**
//! です (`error[E0599]: no method named 'boundary_faces' found for struct 'SdfTetMesh'`)
//! compile しない test は CI を通らないので、**実装と同じ commit で landing** します
//! red は local で観測し、commit message に `E0599` の出力を引用してあります
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// 参照実装側は f32 の幾何をそのまま使う (決定論の対象は `Fix128` 側)
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_fem_mesh::{generate, SdfTetMesh};
use std::collections::{HashMap, HashSet};

fn ball_sdf(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - radius,
        |x, y, z| {
            let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

// ---------------------------------------------------------------------------
// 独立な参照実装 (production API を使わない、理由は module doc)
// ---------------------------------------------------------------------------

/// 面 (頂点 index をソートした key) ごとの使用回数
fn face_use_counts(mesh: &SdfTetMesh) -> HashMap<[u32; 3], usize> {
    let mut counts = HashMap::new();
    for tet in &mesh.tets {
        let v = tet.vertices;
        for face in [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[0], v[2], v[3]],
            [v[1], v[2], v[3]],
        ] {
            let mut key = face;
            key.sort_unstable();
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

/// 生成器自身の占有規則 (8 隅すべてが内側なら meshed) を **SDF から再計算**する
///
/// mesh から読み戻さないので、dicing が何をしたかに依存しない
fn meshed_cells<F: SdfField + ?Sized>(
    sdf: &F,
    min: [f32; 3],
    cell: f32,
    counts: [i32; 3],
) -> HashSet<(i32, i32, i32)> {
    let mut cells = HashSet::new();
    for iz in 0..counts[2] {
        for iy in 0..counts[1] {
            for ix in 0..counts[0] {
                let inside = (0..8).all(|c| {
                    let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let p = [
                        min[0] + (ix + dx) as f32 * cell,
                        min[1] + (iy + dy) as f32 * cell,
                        min[2] + (iz + dz) as f32 * cell,
                    ];
                    sdf.distance(p[0], p[1], p[2]) <= 0.0
                });
                if inside {
                    cells.insert((ix, iy, iz));
                }
            }
        }
    }
    cells
}

/// **幾何で**境界面を決める — 面の重心から法線方向に ±0.25 cell 動かし、
/// 着地した格子 cell が meshed でない側があれば境界
fn boundary_faces_by_geometry(
    mesh: &SdfTetMesh,
    min: [f32; 3],
    cell: f32,
    cells: &HashSet<(i32, i32, i32)>,
) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for (face, uses) in face_use_counts(mesh) {
        if uses == 2 {
            continue;
        }
        let p: [[f32; 3]; 3] = [
            mesh.vertices[face[0] as usize],
            mesh.vertices[face[1] as usize],
            mesh.vertices[face[2] as usize],
        ];
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            (p[0][2] + p[1][2] + p[2][2]) / 3.0,
        ];
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = z_len(n);
        let step = 0.25 * cell;
        let mut both_meshed = true;
        for sign in [1.0_f32, -1.0] {
            let q = [
                centroid[0] + sign * step * n[0] / len,
                centroid[1] + sign * step * n[1] / len,
                centroid[2] + sign * step * n[2] / len,
            ];
            let at = (
                ((q[0] - min[0]) / cell).floor() as i32,
                ((q[1] - min[1]) / cell).floor() as i32,
                ((q[2] - min[2]) / cell).floor() as i32,
            );
            if !cells.contains(&at) {
                both_meshed = false;
            }
        }
        if !both_meshed {
            out.push(face);
        }
    }
    out.sort_unstable();
    out
}

fn z_len(n: [f32; 3]) -> f32 {
    n[2].mul_add(n[2], n[0].mul_add(n[0], n[1] * n[1]))
        .sqrt()
        .max(1.0e-12)
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

#[test]
fn a_single_cell_has_twelve_boundary_faces_and_four_interior_ones() {
    let sdf = ball_sdf(2.0);
    let (min, max, cell) = ([-1.0_f32; 3], [1.0_f32; 3], 2.0_f32);
    let mesh = generate(&sdf, min, max, cell);

    // 前提: 1 cell / 5 tet / 8 頂点 であること (閉形式の導出がこれに立っている)
    assert_eq!(
        mesh.tet_count(),
        5,
        "oracle: 立方体 1 個は 5 四面体に分かれる"
    );
    assert_eq!(mesh.vertex_count(), 8, "oracle: 格子の隅 8 個");

    let counts = face_use_counts(&mesh);
    let single: usize = counts.values().filter(|&&u| u == 1).count();
    let shared: usize = counts.values().filter(|&&u| u == 2).count();
    assert_eq!(single, 12, "oracle: 立方体の表面 = 6 面 x 2 三角形");
    assert_eq!(shared, 4, "oracle: (5 x 4 - 12) / 2 = 4 枚が 2 回ずつ");

    // production API と、幾何で独立に取った参照を突き合わせる
    let cells = meshed_cells(&sdf, min, cell, [1, 1, 1]);
    let expected = boundary_faces_by_geometry(&mesh, min, cell, &cells);
    assert_eq!(expected.len(), 12, "参照側も 12 枚を見ているか");

    // ⚠️ **ここで `got` を sort しないこと** doc が約束している昇順は `HashMap` の
    // 反復順を封じるための契約で、test 側が sort し直すとその契約を検証できません
    // (実測: test が sort していた版では、実装から `sort_unstable()` を外しても 3 回とも green)
    let got = mesh.boundary_faces().expect("適合的な mesh なので Ok");
    assert_eq!(
        got, expected,
        "boundary_faces() が幾何の参照と一致しない (昇順であることも含む)"
    );
}

#[test]
fn a_two_by_two_block_has_twelve_n_squared_boundary_faces() {
    let sdf = ball_sdf(2.0);
    let (min, max, cell) = ([-1.0_f32; 3], [1.0_f32; 3], 1.0_f32);
    let mesh = generate(&sdf, min, max, cell);

    assert_eq!(mesh.tet_count(), 8 * 5, "oracle: 2x2x2 = 8 cell x 5 四面体");

    let cells = meshed_cells(&sdf, min, cell, [2, 2, 2]);
    assert_eq!(cells.len(), 8, "oracle: 8 cell すべてが meshed");

    let expected = boundary_faces_by_geometry(&mesh, min, cell, &cells);
    // 12 n^2 = 12 * 4
    assert_eq!(expected.len(), 48, "oracle: 6 面 x (2x2) x 2 三角形");

    let counts = face_use_counts(&mesh);
    let shared: usize = counts.values().filter(|&&u| u == 2).count();
    assert_eq!(shared, (8 * 20 - 48) / 2, "oracle: 内部面 56 枚");

    // sort しない理由は上の test のコメント参照 (昇順の契約を検証するため)
    let got = mesh.boundary_faces().expect("適合的な mesh なので Ok");
    assert_eq!(
        got, expected,
        "boundary_faces() が幾何の参照と一致しない (昇順であることも含む)"
    );
}

#[test]
fn a_non_manifold_mesh_is_rejected() {
    // 同じ面を 3 つの四面体が共有する手組みの mesh
    let mut mesh = SdfTetMesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        tets: Vec::new(),
    };
    for fourth in [3_u32, 4, 5] {
        mesh.tets.push(alice_physics::sdf_fem_mesh::Tetrahedron {
            vertices: [0, 1, 2, fourth],
        });
    }
    assert!(
        mesh.boundary_faces().is_err(),
        "3 つの四面体が共有する面を持つ mesh が Ok で返った"
    );
}
