//! Oracle: Yee FDTD の材料分布 (`YeeGrid::with_materials`)
//!
//! 期待値はすべて実装の外で作る:
//!
//! - **M1 bit 一致**: 材料を入れる前のコード (`origin/main` = `474f5d4`) で採取した
//!   field hash を定数で埋め込む 新旧 2 本の twin 比較は両方が同じ実装を通るので
//!   歯が無い (採取手順は `GOLDEN_*` の doc)
//! - **M2 一様材料の空洞**: sine mode に射影した振幅が従う 3 項漸化式
//!   `q^{n+1} = (1 + C_a − C_b·(S/μ)·λ)·q^n − C_a·q^{n−1}` (Taflove 3 章の係数を
//!   f64 で再導出)
//! - **M3 平均の取り方**: 境界の sample の実効値を手計算 (E 辺 = 算術平均、H 面 = 調和平均)
//! - **M4 反射 / 透過 / 位相速度 / 損失**: 2 次元 TM 導波路の 1 次 mode に pulse を
//!   入れ、時間 DFT で複素振幅比を取り、離散方程式から解いた閉形式 (f64) と突き合わせる
//!   連続体の Fresnel 係数とは「格子分散 + 導波路の遮断」の閉形式補正込みで比べる
//! - **M5 Gauss の法則** `∇·(εE) = ρ` (σ = 0 で保たれ、`∇·E` では保たれない)
//! - **M6 退化入力** の `Err` 経路すべて
//!
//! # なぜ純粋な 1 次元 (TEM) にしないのか
//!
//! 6 面 PEC の箱は矩形導波路で、TEM mode を持ちません 伝搬方向 x に垂直な `Ez` は
//! y 壁で、`Ey` は z 壁で接線成分になり 0 に固定されるので、横方向に一様な場は作れません
//! ⇒ 最低次の TM mode `Ez ∝ sin(πj/NY)` を使い、その横方向固有値
//! `λ_y = 4 sin²(π/2NY)` を閉形式に**厳密に**持ち込みます (材料が y に一様なので
//! mode は混ざらない) `NY = 32` で `λ_y = 9.6e-3`、遮断周波数 0.098 は pulse の
//! 帯域 (中心 0.5) の外です
//!
//! # 離散閉形式 (M4) の導出
//!
//! 時間調和 `E^n = Ê zⁿ`, `H^{n+½} = Ĥ z^{n+½}`, `z = e^{iωS}`, `τ = z^{½} − z^{−½}
//! = 2i·sin(ωS/2)`, `c_h = cos(ωS/2)` と置くと、`Ez` 列 (x 方向の鎖) は
//!
//! ```text
//! Y_i·Ê_i = S·(Ĥ_{i+½} − Ĥ_{i−½})            Z_{i+½}·Ĥ_{i+½} = S·(Ê_{i+1} − Ê_i)
//! Y = ε·τ + σ·S·c_h + S²·λ_y/(μ_x·τ)          Z = μ·τ
//! ```
//!
//! (`(z − C_a)/(C_b z^{½}) = (ε/S)(τ + 2a·c_h)`, `a = σS/2ε` から `σ` の項が出る、
//! `μ_x` は `Hx` 面の調和平均) 一様媒質では `Ê_i = pⁱ` で `Y·Z/S² = p − 2 + 1/p`
//! — 損失なしなら `ε μ sin²(ωS/2)/S² = sin²(k/2) + λ_y/4` (依頼の
//! `sin(ωΔt/2)/S = n·sin(kΔx/2)` に導波路の項を足したもの)
//!
//! 界面 (cell `i0−1` が媒質 1、`i0` から媒質 2) の節点 `i0` は `ε_J = (ε₁+ε₂)/2`,
//! `σ_J = (σ₁+σ₂)/2`, `μ_x = 2/(1/μ₁+1/μ₂)` を持つので、`E_i = p₁ⁱ + r·p₁^{2i0−i}`
//! (`i ≤ i0`)、`E_i = (1 + r)·p₁^{i0}·p₂^{i−i0}` (`i ≥ i0`) を節点 `i0` の式に入れて
//!
//! ```text
//! G = Y_J − S²(p₂ − 1)/Z₂ + S²/Z₁
//! r = (S²/(Z₁ p₁) − G) / (G − S² p₁/Z₁)
//! ```
//!
//! 媒質 1 = 媒質 2 で `r = 0` になることを `discrete_closed_form_is_self_consistent`
//! が確かめます
//!
//! # 測り方 (M4)
//!
//! 同じ source を「媒質 1 一様」の参照格子と「界面あり」の格子で走らせ、probe `IP`
//! の差 = 反射波、参照 = 入射波 (反射が戻る前に完全に通過するので両方とも全体が窓に入る)
//! 線形時不変なので `DFT(反射)/DFT(入射) = r·p₁^{2(i0−IP)}` が**窓の切り方によらず**成り立ちます
//! (pulse が窓内で完全に収まる限り) 媒質 2 側は隣接 probe の比 = `p₂`
//!
//! # 破壊試験 (2026-10-05 実測、1 変異ずつ `src/maxwell_fdtd.rs` に入れて復元)
//!
//! 対象は本 file + 既存 Maxwell oracle 3 file (既存側は全変異で green のまま = 材料の
//! 無い経路に影響が漏れていない)
//!
//! | 変異 | 本 file の red |
//! |---|---|
//! | 実装: `C_a` の `/(1+a)` を落とす | 2 |
//! | 実装: `C_b` の `/(1+a)` を落とす | 2 |
//! | 実装: `a = σS/2ε` の `/2` を落とす | 2 |
//! | 実装: E 辺の ε を調和平均に | 4 |
//! | 実装: E 辺の σ を平均せず 1 cell の値に | 2 |
//! | 実装: H 面の μ を算術平均に | 3 |
//! | 実装: H 更新を `S·μ` に (割らずに掛ける) | 3 |
//! | 実装: E 更新を `S·ε` に | 5 |
//! | 実装: Courant 判定 `>` → `>=` | 1 |
//! | 実装: ε 判定 `<= 0` → `< 0` | 1 |
//! | 配線: `step` が材料を見ない | 7 |
//! | 配線: H の 3 更新が材料を見ない | 3 |
//! | 配線: E の 3 更新が材料を見ない | 5 |
//! | 配線: source 項が `C_b` でなく `S` | 1 |
//! | 配線: `gauss_residual` が `div_d` でなく `div_e` | 1 |
//! | 配線: `with_materials` が係数を格納しない | 9 |
//! | 配線: 吸収層との重なり判定を外す | 1 |
//! | 配線: 真空 sample の判定を外す (全 sample を材料の式で更新) | **0 (等価変異)** |
//!
//! ⚠️ 最後の 1 件は red にならないのが正しい 真空 sample では `C_a = 1`, `C_b = S`,
//! `S/μ = S` が厳密に出るので `1·E + S·curl` と `E + S·curl` は同じ bit になる
//! (`Fix128` の 1 倍は厳密) M1 の golden hash が判定を外した状態でも一致したことが
//! その実測 = bit 一致は判定の有無に依存しない
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{
    Absorber, Component, Material, MaterialError, MaterialMap, YeeGrid, COURANT_3D,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

const ALL: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

// ===========================================================================
// M1: 材料なし (全 sample が真空) = 変更前のコードと bit 一致
// ===========================================================================

/// FNV-1a 64 over `hi` (i64 LE) then `lo` (u64 LE) of each sample.
fn fnv(h: &mut u64, x: Fix128) {
    for b in x.hi.to_le_bytes().iter().chain(x.lo.to_le_bytes().iter()) {
        *h ^= u64::from(*b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

/// Every non-absorbing sample of every component gets `v/1000`, `v ∈ [−1000, 1000]`
/// from a 64-bit LCG (Knuth MMIX constants), visiting components Ex..Hz and
/// indices in `(i, j, k)` order.
fn seed(g: &mut YeeGrid) {
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    s = s
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let v = ((s >> 33) as i64 % 2001) - 1000;
                    if !g.is_absorbing(c, i, j, k) {
                        g.set(c, i, j, k, Fix128::from_ratio(v, 1000));
                    }
                }
            }
        }
    }
}

/// Hash of all six components (Ex..Hz, `(i, j, k)` order) then every interior `ρ`.
fn hash(g: &YeeGrid) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in ALL {
        let (a, b, d) = g.component_dims(c);
        for i in 0..a {
            for j in 0..b {
                for k in 0..d {
                    fnv(&mut h, g.get(c, i, j, k));
                }
            }
        }
    }
    let (a, b, d) = g.interior_node_dims();
    for i in 1..=a {
        for j in 1..=b {
            for k in 1..=d {
                fnv(&mut h, g.charge(i, j, k));
            }
        }
    }
    h
}

/// Scene P: `YeeGrid::new(7, 6, 5, COURANT_3D)`, [`seed`], then
/// `set_current(Ez, 3, 3, 2, 1/3)` and `set_current(Ex, 2, 2, 2, −2/7)`,
/// 200 × `step()`, then [`hash`].
///
/// ⚠️ 採取手順: 2026-10-05、`origin/main` = `474f5d4` (材料の実装を入れる前、
/// `src/` 無変更) の上に本 file と同じ `seed` / `hash` / scene を書いた一時 test を置き、
/// `cargo test -- --nocapture` で出た値を貼った 実装後に取り直した値ではない
const GOLDEN_PLAIN: u64 = 0xf583_cf36_3bc3_2331;

/// Scene A: `YeeGrid::new_with_absorber(12, 10, 8, COURANT_3D,
/// GradedPml { depth: [3, 2, 2], sigma_max: 3 })`, [`seed`] (absorbing samples
/// skipped), 200 × `step()`, then [`hash`]. 採取手順は [`GOLDEN_PLAIN`] と同じ
const GOLDEN_PML: u64 = 0xc273_c162_fc8e_d898;

fn scene_plain(map: bool) -> u64 {
    let mut g = YeeGrid::new(7, 6, 5, COURANT_3D);
    if map {
        g = g.with_materials(&MaterialMap::vacuum(7, 6, 5)).unwrap();
    }
    seed(&mut g);
    g.set_current(Component::Ez, 3, 3, 2, q(1, 3));
    g.set_current(Component::Ex, 2, 2, 2, q(-2, 7));
    for _ in 0..200 {
        g.step();
    }
    hash(&g)
}

fn scene_pml(map: Option<MaterialMap>) -> u64 {
    let mut g = YeeGrid::new_with_absorber(
        12,
        10,
        8,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [3, 2, 2],
            sigma_max: Fix128::from_int(3),
        },
    );
    if let Some(m) = map {
        g = g.with_materials(&m).unwrap();
    }
    seed(&mut g);
    for _ in 0..200 {
        g.step();
    }
    hash(&g)
}

/// Oracle: 変更前のコードの出力 hash (定数) 材料 map を渡した格子も渡さない格子も一致
#[test]
fn a_vacuum_map_is_bit_identical_to_the_code_before_materials() {
    assert_eq!(scene_plain(false), GOLDEN_PLAIN, "no map drifted from main");
    assert_eq!(
        scene_plain(true),
        GOLDEN_PLAIN,
        "vacuum map drifted from main"
    );
    assert_eq!(scene_pml(None), GOLDEN_PML, "PML without map drifted");
    assert_eq!(
        scene_pml(Some(MaterialMap::vacuum(12, 10, 8))),
        GOLDEN_PML,
        "PML + vacuum map drifted"
    );
}

/// 材料が PML 層の外 (sample が吸収層に触れない内部) だけにあれば併用でき、層の外側の
/// 値を持つ map は受理される 内部の材料は場を変えるので hash は golden と違う (歯の確認)
#[test]
fn materials_inside_the_lossless_core_coexist_with_a_pml() {
    let mut m = MaterialMap::vacuum(12, 10, 8);
    // PML depth [3,2,2]: samples touching cells 3..9 × 2..8 × 2..6 may be split
    // only at the layer; cells 5..7 × 4..6 × 3..5 are two cells inside it.
    m.fill(
        [5, 4, 3],
        [7, 6, 5],
        Material::new(q(5, 2), q(3, 2), q(1, 8)),
    );
    let h = scene_pml(Some(m));
    assert_ne!(h, GOLDEN_PML, "an interior material must change the field");
}

// ===========================================================================
// M2: 一様材料の空洞 — 3 項漸化式 (3 向き)
// ===========================================================================

/// `(vector, λ)` with `s[i+1] − 2s[i] + s[i−1] = −λ s[i]`, PEC ends.
const M3_1: (&[i64], i64) = (&[0, 1, 1, 0], 1);
const M4_2: (&[i64], i64) = (&[0, 1, 0, -1, 0], 2);

#[derive(Clone, Copy, Debug)]
enum Orient {
    Ez,
    Ex,
    Ey,
}

fn check_eigen(v: &[i64], lambda: i64) {
    for i in 1..v.len() - 1 {
        assert_eq!(v[i + 1] - 2 * v[i] + v[i - 1], -lambda * v[i]);
    }
}

/// `(dims, component, index of sample (a, b))`.
fn layout(o: Orient, a: usize, b: usize) -> ((usize, usize, usize), Component) {
    match o {
        Orient::Ez => ((a, b, 1), Component::Ez),
        Orient::Ex => ((1, a, b), Component::Ex),
        Orient::Ey => ((b, 1, a), Component::Ey),
    }
}

fn at(o: Orient, x: usize, y: usize) -> (usize, usize, usize) {
    match o {
        Orient::Ez => (x, y, 0),
        Orient::Ex => (0, x, y),
        Orient::Ey => (y, 0, x),
    }
}

/// `Taflove` 3 章の係数を f64 で: `(C_a, C_b)`
fn taflove(eps: f64, sigma: f64, s: f64) -> (f64, f64) {
    let a = sigma * s / (2.0 * eps);
    ((1.0 - a) / (1.0 + a), (s / eps) / (1.0 + a))
}

fn run_uniform_cavity(o: Orient, mat: Material) -> (Vec<f64>, Vec<f64>) {
    let (va, la) = M3_1;
    let (vb, lb) = M4_2;
    check_eigen(va, la);
    check_eigen(vb, lb);
    let (a, b) = (va.len() - 1, vb.len() - 1);
    let ((nx, ny, nz), comp) = layout(o, a, b);
    let mut map = MaterialMap::vacuum(nx, ny, nz);
    map.fill([0, 0, 0], [nx, ny, nz], mat);
    let mut g = YeeGrid::new(nx, ny, nz, COURANT_3D)
        .with_materials(&map)
        .unwrap();
    for x in 0..=a {
        for y in 0..=b {
            let (i, j, k) = at(o, x, y);
            g.set(comp, i, j, k, Fix128::from_int(va[x] * vb[y]));
        }
    }
    let norm: f64 = (0..=a)
        .flat_map(|x| (0..=b).map(move |y| (va[x] * vb[y]) as f64))
        .map(|v| v * v)
        .sum();
    let project = |g: &YeeGrid| -> f64 {
        let mut acc = 0.0;
        for x in 0..=a {
            for y in 0..=b {
                let (i, j, k) = at(o, x, y);
                acc += g.get(comp, i, j, k).to_f64() * (va[x] * vb[y]) as f64;
            }
        }
        acc / norm
    };
    // closed form
    let s = 0.5625;
    let (ca, cb) = taflove(mat.eps_r.to_f64(), mat.sigma.to_f64(), s);
    let kappa = cb * (s / mat.mu_r.to_f64()) * (la + lb) as f64;
    let steps = 60;
    let mut expect = vec![1.0, ca - kappa];
    for n in 1..steps {
        let next = (1.0 + ca - kappa) * expect[n] - ca * expect[n - 1];
        expect.push(next);
    }
    let mut got = vec![project(&g)];
    for _ in 0..steps {
        g.step();
        got.push(project(&g));
    }
    (got, expect)
}

/// Oracle: 一様な (ε, μ, σ) の空洞で、射影振幅が Taflove 係数の 3 項漸化式に一致
/// (`E⁰` = mode、`H^{−½}` = 0 ⇒ `q¹ = (C_a − κ)q⁰`、以降 `q^{n+1} = (1+C_a−κ)qⁿ − C_a q^{n−1}`、
/// `κ = C_b·(S/μ)·λ`、`λ = 1 + 2 = 3`) 3 向きで回すのは 1 向きだと残り 2 成分の更新式が
/// 0 の上しか走らないから (既存 Maxwell oracle と同じ理由)
#[test]
fn a_uniform_material_cavity_follows_the_taflove_recurrence() {
    let cases = [
        (
            "eps2 mu3 sigma1/4",
            Material::new(q(2, 1), q(3, 1), q(1, 4)),
        ),
        (
            "mu4 only",
            Material::new(Fix128::ONE, q(4, 1), Fix128::ZERO),
        ),
        (
            "eps5/2 sigma3",
            Material::new(q(5, 2), Fix128::ONE, q(3, 1)),
        ),
    ];
    for (name, mat) in cases {
        for o in [Orient::Ez, Orient::Ex, Orient::Ey] {
            let (got, expect) = run_uniform_cavity(o, mat);
            let scale = expect.iter().fold(1e-30_f64, |m, v| m.max(v.abs()));
            for (n, (g, e)) in got.iter().zip(&expect).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-12 * scale,
                    "{name} {o:?} step {n}: got {g:.15e}, closed form {e:.15e}"
                );
            }
            // teeth: the vacuum recurrence is distinguishable from this one
            let vac = {
                let k = 0.5625 * 0.5625 * 3.0;
                let mut v = vec![1.0, 1.0 - k];
                for n in 1..60 {
                    let x = (2.0 - k) * v[n] - v[n - 1];
                    v.push(x);
                }
                v
            };
            let gap = got
                .iter()
                .zip(&vac)
                .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(gap > 1e-2, "{name} {o:?}: indistinguishable from vacuum");
        }
    }
}

// ===========================================================================
// M3: 平均の取り方 (手計算)
// ===========================================================================

/// Oracle: E 辺は周囲 4 cell の算術平均 (ε と σ)、H 面は両側 2 cell の調和平均 (μ)、
/// 格子の壁の上では存在する cell だけ、全部同じ値ならその値そのもの (丸めなし)
#[test]
fn edges_take_the_arithmetic_mean_and_faces_the_harmonic_mean() {
    let mut m = MaterialMap::vacuum(4, 4, 4);
    // four distinct cells around the Ez edge (2, 2, 1): cells (1|2, 1|2, 1)
    m.set(1, 1, 1, Material::new(q(1, 1), q(1, 1), q(0, 1)));
    m.set(2, 1, 1, Material::new(q(2, 1), q(3, 1), q(1, 2)));
    m.set(1, 2, 1, Material::new(q(3, 1), q(1, 1), q(1, 4)));
    m.set(2, 2, 1, Material::new(q(6, 1), q(6, 1), q(1, 4)));
    let g = YeeGrid::new(4, 4, 4, COURANT_3D)
        .with_materials(&m)
        .unwrap();
    // ε = (1+2+3+6)/4 = 3, σ = (0 + 1/2 + 1/4 + 1/4)/4 = 1/4
    let e = g.effective_material(Component::Ez, 2, 2, 1);
    assert_eq!(e.eps_r, q(3, 1));
    assert_eq!(e.sigma, q(1, 4));
    assert_eq!(e.mu_r, Fix128::ONE);
    // Hx face (2, 1, 1) separates cells (1,1,1) μ=1 and (2,1,1) μ=3: 2/(1+1/3) = 3/2
    let h = g.effective_material(Component::Hx, 2, 1, 1);
    assert_eq!(h.mu_r, q(3, 2));
    assert_eq!(h.eps_r, Fix128::ONE);
    assert_eq!(h.sigma, Fix128::ZERO);
    // Hy face (2, 2, 1) separates (2,1,1) μ=3 and (2,2,1) μ=6: 2/(1/3+1/6) = 4
    // 2/(1/3+1/6) = 4, but 1/3 and 1/6 truncate: within a few ULP of 4, above it
    let hy = g.effective_material(Component::Hy, 2, 2, 1).mu_r;
    let ulp = hy - q(4, 1);
    assert!(
        !ulp.is_negative() && ulp < Fix128::from_raw(0, 64),
        "harmonic mean {hy:?}"
    );
    // a wall face has one cell: Hx (4, 2, 1) is in cell (3, 2, 1) only (vacuum)
    assert_eq!(
        g.effective_material(Component::Hx, 4, 2, 1).mu_r,
        Fix128::ONE
    );
    // a wall face next to a material cell: Hx (0, 1, 1) touches only cell (0,1,1)
    let mut w = MaterialMap::vacuum(4, 4, 4);
    w.set(0, 1, 1, Material::new(Fix128::ONE, q(1, 3), Fix128::ZERO));
    let gw = YeeGrid::new(4, 4, 4, COURANT_3D / Fix128::from_int(2))
        .with_materials(&w)
        .unwrap();
    assert_eq!(gw.effective_material(Component::Hx, 0, 1, 1).mu_r, q(1, 3));
    // an edge between two cells of 1 and two of 3 is exactly 2 (not a harmonic 3/2)
    let mut h2 = MaterialMap::vacuum(4, 4, 4);
    h2.fill([2, 0, 0], [4, 4, 4], Material::dielectric(q(3, 1)));
    let g2 = YeeGrid::new(4, 4, 4, COURANT_3D)
        .with_materials(&h2)
        .unwrap();
    assert_eq!(g2.effective_material(Component::Ez, 2, 1, 1).eps_r, q(2, 1));
    assert_eq!(g2.effective_material(Component::Ez, 3, 1, 1).eps_r, q(3, 1));
    assert_eq!(g2.effective_material(Component::Ex, 2, 1, 1).eps_r, q(3, 1));
    assert_eq!(
        g2.effective_material(Component::Ex, 1, 1, 1).eps_r,
        Fix128::ONE
    );
    // a lattice without a map is vacuum everywhere
    let plain = YeeGrid::new(4, 4, 4, COURANT_3D);
    assert_eq!(
        plain.effective_material(Component::Hz, 1, 1, 1),
        Material::VACUUM
    );
}

// ===========================================================================
// M4: 反射 / 透過 / 位相速度 / 損失 (2-D TM 導波路の pulse)
// ===========================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
struct C {
    re: f64,
    im: f64,
}
impl C {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    fn div(self, o: Self) -> Self {
        let d = o.re * o.re + o.im * o.im;
        Self::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    fn sqrt(self) -> Self {
        let r = self.abs();
        let re = ((r + self.re) / 2.0).max(0.0).sqrt();
        let im = ((r - self.re) / 2.0).max(0.0).sqrt();
        Self::new(re, if self.im < 0.0 { -im } else { im })
    }
    fn powi(self, n: i32) -> Self {
        let mut acc = Self::new(1.0, 0.0);
        let (mut b, mut e) = (
            if n < 0 {
                Self::new(1.0, 0.0).div(self)
            } else {
                self
            },
            n.unsigned_abs(),
        );
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(b);
            }
            b = b.mul(b);
            e >>= 1;
        }
        acc
    }
}

const S: f64 = 0.5625;
const NX: usize = 272;
const NY: usize = 32;
const IS: usize = 10;
const IP: usize = 110;
const I0: usize = 140;
const JP: usize = 16;
const TAU: f64 = 16.0;
const W0: f64 = 0.6;
const STEPS: usize = 660;
const OMEGAS: [f64; 3] = [0.5, 0.6, 0.7];

/// `λ_y = 4 sin²(π/2NY)`, the discrete transverse eigenvalue of the mode.
const LY: f64 = 0.009_630_546_655_606_228;
/// Continuum `k_y² = (π/NY)²`.
const KY2: f64 = 0.009_638_285_547_938_826;

fn lambda_y() -> f64 {
    let s = (std::f64::consts::PI / (2.0 * NY as f64)).sin();
    4.0 * s * s
}

/// Medium constants as f64 `(ε, μ, σ)`.
#[derive(Clone, Copy, Debug)]
struct Med {
    eps: f64,
    mu: f64,
    sigma: f64,
}

impl Med {
    fn material(self) -> Material {
        Material::new(
            Fix128::from_f64(self.eps),
            Fix128::from_f64(self.mu),
            Fix128::from_f64(self.sigma),
        )
    }
}

fn tau_ch(w: f64) -> (C, f64) {
    let h = w * S / 2.0;
    (C::new(0.0, 2.0 * h.sin()), h.cos())
}

/// Node admittance `Y = ε τ + σ S c_h + S² λ_y/(μ_x τ)`.
fn y_node(eps: f64, sigma: f64, mu_x: f64, w: f64, ly: f64) -> C {
    let (t, ch) = tau_ch(w);
    t.scale(eps)
        .add(C::new(sigma * S * ch, 0.0))
        .add(C::new(S * S * ly / mu_x, 0.0).div(t))
}

fn z_link(mu: f64, w: f64) -> C {
    tau_ch(w).0.scale(mu)
}

/// Forward (`+x`) root of `p + 1/p = 2 + Y Z / S²` in a uniform medium.
fn p_forward(m: Med, w: f64, ly: f64) -> C {
    let beta = C::new(1.0, 0.0).add(
        y_node(m.eps, m.sigma, m.mu, w, ly)
            .mul(z_link(m.mu, w))
            .scale(0.5 / (S * S)),
    );
    let root = beta.mul(beta).sub(C::new(1.0, 0.0)).sqrt();
    let (a, b) = (beta.add(root), beta.sub(root));
    if (a.abs() - b.abs()).abs() > 1e-12 {
        if a.abs() < b.abs() {
            a
        } else {
            b
        }
    } else if a.im < 0.0 {
        a
    } else {
        b
    }
}

/// Discrete reflection coefficient at the `i0` node (header derivation).
fn r_discrete(m1: Med, m2: Med, w: f64, ly: f64) -> C {
    let p1 = p_forward(m1, w, ly);
    let p2 = p_forward(m2, w, ly);
    let z1 = z_link(m1.mu, w);
    let z2 = z_link(m2.mu, w);
    let mux = 2.0 / (1.0 / m1.mu + 1.0 / m2.mu);
    let yj = y_node(
        (m1.eps + m2.eps) / 2.0,
        (m1.sigma + m2.sigma) / 2.0,
        mux,
        w,
        ly,
    );
    let s2 = C::new(S * S, 0.0);
    let g = yj
        .sub(s2.mul(p2.sub(C::new(1.0, 0.0))).div(z2))
        .add(s2.div(z1));
    s2.div(z1.mul(p1)).sub(g).div(g.sub(s2.mul(p1).div(z1)))
}

/// Continuum s-polarised (Ez ⟂ plane of incidence) Fresnel coefficient of the
/// same waveguide mode, `e^{+iωt}` convention, `ε_c = ε − iσ/ω`.
fn r_fresnel_mode(m1: Med, m2: Med, w: f64, ky2: f64) -> C {
    let kx = |m: Med| C::new(m.eps * m.mu * w * w - ky2, -m.mu * m.sigma * w).sqrt();
    let (a, b) = (kx(m1).scale(m2.mu), kx(m2).scale(m1.mu));
    a.sub(b).div(a.add(b))
}

/// Plane-wave normal-incidence Fresnel `r = (η₂ − η₁)/(η₂ + η₁)`, `η = √(μ/ε)`.
fn r_plane(m1: Med, m2: Med) -> f64 {
    let eta = |m: Med| (m.mu / m.eps).sqrt();
    (eta(m2) - eta(m1)) / (eta(m2) + eta(m1))
}

fn source(n: usize) -> f64 {
    let t = n as f64 * S - 4.0 * TAU;
    (-(t / TAU) * (t / TAU)).exp() * (W0 * t).sin()
}

/// Ez(i, JP) at the probes, one entry per step (after the step).
fn run_waveguide(m1: Med, m2: Option<Med>, probes: &[usize]) -> Vec<Vec<f64>> {
    let mut map = MaterialMap::vacuum(NX, NY, 1);
    map.fill([0, 0, 0], [NX, NY, 1], m1.material());
    if let Some(m2) = m2 {
        map.fill([I0, 0, 0], [NX, NY, 1], m2.material());
    }
    let mut g = YeeGrid::new(NX, NY, 1, COURANT_3D)
        .with_materials(&map)
        .unwrap();
    let profile: Vec<f64> = (0..=NY)
        .map(|j| (std::f64::consts::PI * j as f64 / NY as f64).sin())
        .collect();
    let mut out = vec![Vec::with_capacity(STEPS); probes.len()];
    for n in 0..STEPS {
        let a = source(n);
        for j in 1..NY {
            g.set_current(Component::Ez, IS, j, 0, Fix128::from_f64(a * profile[j]));
        }
        g.step();
        for (p, &i) in probes.iter().enumerate() {
            out[p].push(g.get(Component::Ez, i, JP, 0).to_f64());
        }
    }
    out
}

fn dft(x: &[f64], w: f64) -> C {
    x.iter().enumerate().fold(C::new(0.0, 0.0), |acc, (n, &v)| {
        let ph = -w * S * n as f64;
        acc.add(C::new(v * ph.cos(), v * ph.sin()))
    })
}

struct Measured {
    /// `DFT(reflected)/DFT(incident)` at `IP`, divided by `p₁^{2(I0−IP)}`.
    r: Vec<C>,
    /// `DFT(E[I0+11])/DFT(E[I0+10])`.
    p2: Vec<C>,
    /// `DFT(E[IP+1])/DFT(E[IP])` in the reference run.
    p1: Vec<C>,
}

fn measure(m1: Med, m2: Med) -> Measured {
    let probes = [IP, IP + 1, I0 + 10, I0 + 11];
    let reference = run_waveguide(m1, None, &probes);
    let slab = run_waveguide(m1, Some(m2), &probes);
    let refl: Vec<f64> = slab[0]
        .iter()
        .zip(&reference[0])
        .map(|(a, b)| a - b)
        .collect();
    // window sanity: the incident and reflected records have ended inside the window
    let peak = reference[0].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    // The last 20 steps: the pulse has passed. What is left (≤ 2e-6 of the peak,
    // measured 2026-10-05) is the slow near-cut-off ringing; it is what limits
    // the agreement with the closed form to ~1e-7.
    let tail = |x: &[f64]| x[STEPS - 20..].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        tail(&reference[0]) < 1e-5 * peak,
        "incident still at the probe at the end"
    );
    assert!(
        tail(&refl) < 1e-5 * peak,
        "reflection still at the probe at the end"
    );
    let mut out = Measured {
        r: vec![],
        p2: vec![],
        p1: vec![],
    };
    for w in OMEGAS {
        let p1 = p_forward(m1, w, LY);
        let ratio = dft(&refl, w).div(dft(&reference[0], w));
        out.r.push(ratio.div(p1.powi(2 * (I0 - IP) as i32)));
        out.p2.push(dft(&slab[3], w).div(dft(&slab[2], w)));
        out.p1
            .push(dft(&reference[1], w).div(dft(&reference[0], w)));
    }
    out
}

const VAC: Med = Med {
    eps: 1.0,
    mu: 1.0,
    sigma: 0.0,
};

/// Oracle の自己検査: 同じ媒質なら離散の `r` は 0、連続体 mode の `r` も 0、
/// 損失なしなら `|p| = 1`
#[test]
fn discrete_closed_form_is_self_consistent() {
    assert!((LY - lambda_y()).abs() < 1e-17);
    assert!((KY2 - (std::f64::consts::PI / NY as f64).powi(2)).abs() < 1e-17);
    for w in OMEGAS {
        assert!(r_discrete(VAC, VAC, w, LY).abs() < 1e-12);
        let d = Med {
            eps: 4.0,
            mu: 1.0,
            sigma: 0.0,
        };
        assert!(r_discrete(d, d, w, LY).abs() < 1e-12);
        assert!((p_forward(d, w, LY).abs() - 1.0).abs() < 1e-12);
        assert!(r_fresnel_mode(d, d, w, KY2).abs() < 1e-14);
    }
}

/// `|r_disc − r_cont|` of the waveguide mode at `ω`.
fn lattice_gap(m2: Med, w: f64) -> f64 {
    r_discrete(VAC, m2, w, LY)
        .sub(r_fresnel_mode(VAC, m2, w, KY2))
        .abs()
}

/// Order of the lattice correction of the plain 1-D chain (`λ_y = 0`, so the
/// continuum is the plane-wave Fresnel coefficient): `gap(ω)/gap(ω/2)` at small
/// `ω`, where the leading term dominates. `4` means second order.
///
/// A conductivity is scaled with `ω` (`σ/ω` held at its value for `ω = 1/2`) so
/// that the continuum coefficient is the same at both frequencies; otherwise
/// lowering `ω` also moves the medium towards the conductor regime, where the
/// error is measured in `√(ωσ)` and the ratio stops being an order.
fn lattice_order(m2: Med) -> f64 {
    let g = |w: f64| {
        let m = Med {
            sigma: m2.sigma * w / 0.5,
            ..m2
        };
        r_discrete(VAC, m, w, 0.0)
            .sub(r_fresnel_mode(VAC, m, w, 0.0))
            .abs()
    };
    g(0.04) / g(0.02)
}

/// 測定 → 離散閉形式 (厳密) と、離散閉形式 → 連続体 (2 次収束) の 2 段で比べる
///
/// ⚠️ 連続体の Fresnel との差は **格子分解能そのもの** で、`ω = 0.5` (媒質 n = 2 で
/// 1 波長 6 cell) では 13% に達する (`r_disc = −0.2902`, 平面波 Fresnel `−1/3`)
/// これを許容幅に丸ごと入れると 1 桁の bug を素通しするので、許容は
/// 「閉形式が与える補正量 `|r_disc − r_cont|` + 1e-6」とし、その補正量が
/// `ω → ω/2` で 1/4 前後に縮む (= 2 次の離散化誤差であって実装の欠陥ではない) ことを
/// 閉形式の側で確かめる
fn check_medium(name: &str, m2: Med) -> Measured {
    let meas = measure(VAC, m2);
    // the correction is the second-order discretisation error, not a defect:
    // halving ω divides it by four in the asymptotic range (closed form only)
    let order = lattice_order(m2);
    println!("{name}: lattice correction order gap(0.04)/gap(0.02) = {order:.4}");
    assert!(
        (3.8..4.2).contains(&order),
        "{name}: lattice correction is not second order ({order:.3})"
    );
    for (n, &w) in OMEGAS.iter().enumerate() {
        let rd = r_discrete(VAC, m2, w, LY);
        let rm = meas.r[n];
        let rf = r_fresnel_mode(VAC, m2, w, KY2);
        let rp = r_plane(VAC, m2);
        let p1 = p_forward(VAC, w, LY);
        let p2 = p_forward(m2, w, LY);
        println!(
            "{name} ω={w}: r measured {:+.9}{:+.9}i  discrete {:+.9}{:+.9}i  |gap| {:.1e}  continuum mode {:+.6}{:+.6}i  plane {:+.6}  |p2| meas {:.9} disc {:.9}",
            rm.re, rm.im, rd.re, rd.im, rm.sub(rd).abs(), rf.re, rf.im, rp, meas.p2[n].abs(), p2.abs()
        );
        // (1) measured vs the exact discrete closed form. Measured gap 1e-8..1e-7
        // (2026-10-05); the floor is the pulse tail left in the window (≤ 1e-5 of
        // the peak, asserted in `measure`) leaking into the DFT, not truncation.
        assert!(
            rm.sub(rd).abs() < 1e-6,
            "{name} ω={w}: |r_meas − r_disc| = {:.3e}",
            rm.sub(rd).abs()
        );
        assert!(
            meas.p2[n].sub(p2).abs() < 1e-5,
            "{name} ω={w}: p2 gap {:.3e}",
            meas.p2[n].sub(p2).abs()
        );
        assert!(meas.p1[n].sub(p1).abs() < 1e-5, "{name} ω={w}: p1");
        // (2) continuum: within the closed-form lattice correction
        let gap = lattice_gap(m2, w);
        assert!(
            rm.sub(rf).abs() <= gap + 1e-6,
            "{name} ω={w}: measured r is further from the continuum than the lattice correction"
        );
        // (3) the waveguide mode tends to the plane wave: |r_cont − r_plane| is the
        // cut-off term k_y²/k², small for the loss-free media at these ω
        if m2.sigma == 0.0 {
            assert!(
                rf.sub(C::new(rp, 0.0)).abs() < 0.01,
                "{name} ω={w}: cut-off correction too large"
            );
        }
    }
    meas
}

/// Oracle: ε_r = 4 の半空間 — 垂直入射 Fresnel `r = (1 − 2)/(1 + 2) = −1/3`
/// (符号は負、測定は離散閉形式と 1e-6 以内、連続体とは 2 次収束する補正の範囲内)、
/// 媒質中の波数を離散分散関係 `εμ sin²(ωS/2)/S² = sin²(k/2) + λ_y/4` と比べて
/// 位相速度 `ω/k` を出す (連続体 `1/√(εμ) = 1/2` からのずれも同じ閉形式が与える)
#[test]
fn a_dielectric_half_space_reflects_minus_one_third() {
    let m2 = Med {
        eps: 4.0,
        mu: 1.0,
        sigma: 0.0,
    };
    let meas = check_medium("eps4", m2);
    for (n, &w) in OMEGAS.iter().enumerate() {
        assert!(
            meas.r[n].re < -0.2,
            "a denser medium reflects with a negative sign"
        );
        let k_meas = -meas.p2[n].im.atan2(meas.p2[n].re);
        let sh = (w * S / 2.0).sin();
        let k_disc = 2.0 * (4.0 * sh * sh / (S * S) - LY / 4.0).sqrt().asin();
        let k_cont = (4.0 * w * w - (std::f64::consts::PI / NY as f64).powi(2)).sqrt();
        let (v_meas, v_disc, v_cont) = (w / k_meas, w / k_disc, w / k_cont);
        println!("eps4 ω={w}: phase velocity measured {v_meas:.9} discrete {v_disc:.9} continuum mode {v_cont:.6} plane 1/n = 0.5");
        assert!(
            (k_meas - k_disc).abs() < 1e-5,
            "ω={w}: k measured {k_meas:.9} discrete {k_disc:.9}"
        );
        // the lattice slows the wave (numerical dispersion), never speeds it up
        assert!(v_disc < v_cont && v_cont < 0.51, "dispersion sign");
    }
}

/// Oracle: μ_r = 4 の半空間 — `η = 2` なので `r = +1/3` (誘電体と符号が逆)
#[test]
fn a_magnetic_half_space_reflects_plus_one_third() {
    let m2 = Med {
        eps: 1.0,
        mu: 4.0,
        sigma: 0.0,
    };
    let meas = check_medium("mu4", m2);
    for n in 0..OMEGAS.len() {
        assert!(
            meas.r[n].re > 0.2,
            "a magnetic medium reflects with a positive sign"
        );
    }
}

/// Oracle: ε_r = μ_r = 2 (インピーダンス整合) — 連続体の垂直入射で `r = 0`
///
/// ⚠️ 離散では 0 にならない: 界面節点の `ε_J = 3/2` と半 cell ずれた `μ` の段差が
/// 2 次の反射を作り、`ω = 0.5 / 0.6 / 0.7` (媒質内 1 波長 6.0 / 5.0 / 4.3 cell) で
/// `|r| = 0.048 / 0.081 / 0.128` (実測、離散閉形式と 1e-7 で一致) 同じ n = 2 の
/// 非整合 (ε = 4) の `0.29 / 0.26 / 0.22` よりは小さく、`ω → 0` で 2 次に 0 へ向かう
/// (1-D 鎖の閉形式で `ω = 0.04` のとき `|r| < 1e-3`)
#[test]
fn an_impedance_matched_half_space_reflects_only_at_second_order() {
    let m2 = Med {
        eps: 2.0,
        mu: 2.0,
        sigma: 0.0,
    };
    let meas = check_medium("matched", m2);
    let unmatched = Med {
        eps: 4.0,
        mu: 1.0,
        sigma: 0.0,
    };
    for (n, &w) in OMEGAS.iter().enumerate() {
        let r_unmatched = r_discrete(VAC, unmatched, w, LY).abs();
        assert!(
            meas.r[n].abs() < 0.6 * r_unmatched,
            "ω={w}: matched |r| {} vs unmatched {r_unmatched}",
            meas.r[n].abs()
        );
        // and the continuum mode value is the cut-off term alone
        assert!(r_fresnel_mode(VAC, m2, w, KY2).abs() < 0.01);
    }
    let r_low = r_discrete(VAC, m2, 0.04, 0.0).abs();
    assert!(
        r_low < 1e-3,
        "matched 1-D chain at ω = 0.04: |r| = {r_low:.3e}"
    );
}

/// Oracle: 損失媒質 (ε=2, σ=1/2) — 1 cell あたりの伝搬係数 `p₂` (振幅減衰と位相) と
/// 反射係数を離散閉形式と比較 (`Y` の `σ·S·c_h` 項 = Taflove の `C_a` / `C_b` がここで効く)
#[test]
fn a_lossy_half_space_attenuates_by_the_discrete_closed_form() {
    let m2 = Med {
        eps: 2.0,
        mu: 1.0,
        sigma: 0.5,
    };
    let meas = check_medium("lossy", m2);
    for (n, &w) in OMEGAS.iter().enumerate() {
        let a = meas.p2[n].abs();
        let ky2 = (std::f64::consts::PI / NY as f64).powi(2);
        let kc = C::new(2.0 * w * w - ky2, -0.5 * w).sqrt();
        println!(
            "lossy ω={w}: |p2| measured {a:.9} discrete {:.9} continuum e^(−α) {:.6}",
            p_forward(m2, w, LY).abs(),
            (-kc.im.abs()).exp()
        );
        assert!(a < 0.9, "lossy medium must attenuate: |p2| = {a}");
    }
}

// ===========================================================================
// M5: Gauss の法則 ∇·(εE) = ρ
// ===========================================================================

fn gauss_scene(sigma: Fix128) -> YeeGrid {
    let mut m = MaterialMap::vacuum(8, 8, 8);
    m.fill([4, 0, 0], [8, 8, 8], Material::new(q(3, 1), q(2, 1), sigma));
    m.fill([0, 0, 0], [8, 8, 3], Material::dielectric(q(5, 2)));
    let mut g = YeeGrid::new(8, 8, 8, COURANT_3D)
        .with_materials(&m)
        .unwrap();
    // a current that runs for 40 steps on edges at the interfaces, then stops
    g.set_current(Component::Ex, 3, 4, 4, q(1, 3));
    g.set_current(Component::Ez, 4, 3, 2, q(-2, 7));
    g.set_current(Component::Ey, 4, 2, 5, q(3, 5));
    g
}

fn max_div_e_minus_rho(g: &YeeGrid) -> f64 {
    let (a, b, c) = g.interior_node_dims();
    let mut w = 0.0_f64;
    for i in 1..=a {
        for j in 1..=b {
            for k in 1..=c {
                w = w.max((g.div_e(i, j, k) - g.charge(i, j, k)).to_f64().abs());
            }
        }
    }
    w
}

/// Oracle: σ = 0 なら `∇·(εE) − ρ` は ε·C_b = S により切り捨て誤差の範囲で保存
/// (初期値 0)、`∇·E − ρ` (ε で重み付けない) は界面で保存されない
#[test]
fn gauss_law_holds_for_displacement_with_loss_free_materials() {
    let mut g = gauss_scene(Fix128::ZERO);
    for n in 0..200 {
        if n == 40 {
            g.set_current(Component::Ex, 3, 4, 4, Fix128::ZERO);
            g.set_current(Component::Ez, 4, 3, 2, Fix128::ZERO);
            g.set_current(Component::Ey, 4, 2, 5, Fix128::ZERO);
        }
        g.step();
    }
    let d = g.max_abs_gauss_residual().to_f64();
    let e = max_div_e_minus_rho(&g);
    let charge = g.total_charge().to_f64().abs();
    let max_rho = {
        let (a, b, c) = g.interior_node_dims();
        let mut w = 0.0_f64;
        for i in 1..=a {
            for j in 1..=b {
                for k in 1..=c {
                    w = w.max(g.charge(i, j, k).to_f64().abs());
                }
            }
        }
        w
    };
    println!("gauss: max|div D − ρ| = {d:.3e}, max|div E − ρ| = {e:.3e}, max|ρ| = {max_rho:.3e}, |Σρ| = {charge:.3e}");
    assert!(max_rho > 1.0, "the scene must deposit charge");
    // truncation only: a few ULP per step per edge, 200 steps (2⁻⁶⁴ ≈ 5.4e-20)
    assert!(d < 1e-15, "Gauss residual for D is {d:.3e}");
    assert!(
        e > 1e-2,
        "div E alone should not be conserved across ε jumps: {e:.3e}"
    );
}

/// σ > 0 では伝導電流 σE が ρ に計上されない電荷を運ぶので、`∇·(εE) − ρ` は
/// 保存されない (仕様、module doc) 実測値を pin せず「切り捨ての桁を超える」ことだけ見る
#[test]
fn conduction_current_is_not_tracked_by_rho() {
    let mut g = gauss_scene(q(1, 2));
    for _ in 0..200 {
        g.step();
    }
    let d = g.max_abs_gauss_residual().to_f64();
    println!("gauss with σ=1/2: max|div D − ρ| = {d:.3e}");
    assert!(
        d > 1e-3,
        "conduction should relax div D away from ρ: {d:.3e}"
    );
}

// ===========================================================================
// M6: 退化入力
// ===========================================================================

fn err(map: &MaterialMap, courant: Fix128) -> MaterialError {
    let (nx, ny, nz) = (4, 4, 4);
    YeeGrid::new(nx, ny, nz, courant)
        .with_materials(map)
        .unwrap_err()
}

#[test]
fn a_map_with_other_dimensions_is_rejected() {
    let e = err(&MaterialMap::vacuum(4, 4, 3), COURANT_3D);
    assert_eq!(
        e,
        MaterialError::DimensionMismatch {
            lattice: (4, 4, 4),
            map: (4, 4, 3)
        }
    );
    let e0 = err(&MaterialMap::vacuum(0, 4, 4), COURANT_3D);
    assert!(matches!(e0, MaterialError::DimensionMismatch { .. }));
}

#[test]
fn non_positive_constants_are_rejected_with_the_first_cell() {
    for (eps, mu, sigma, want) in [
        (q(0, 1), q(1, 1), q(0, 1), "eps"),
        (q(-1, 2), q(1, 1), q(0, 1), "eps"),
        (q(4, 1), q(0, 1), q(0, 1), "mu"),
        (q(4, 1), q(-3, 1), q(0, 1), "mu"),
        (q(4, 1), q(1, 1), Fix128::from_raw(-1, u64::MAX), "sigma"),
    ] {
        let mut m = MaterialMap::vacuum(4, 4, 4);
        m.set(2, 1, 3, Material::new(eps, mu, sigma));
        m.set(3, 0, 0, Material::new(eps, mu, sigma));
        let e = err(&m, COURANT_3D);
        let cell = (2, 1, 3);
        let expect = match want {
            "eps" => MaterialError::NonPositivePermittivity { cell },
            "mu" => MaterialError::NonPositivePermeability { cell },
            _ => MaterialError::NegativeConductivity { cell },
        };
        assert_eq!(e, expect);
        assert!(!format!("{e}").is_empty());
    }
    // the smallest positive values are accepted (no off-by-one in the comparison)
    let mut ok = MaterialMap::vacuum(4, 4, 4);
    ok.set(1, 1, 1, Material::new(q(1, 1), q(1, 1), Fix128::ZERO));
    assert!(YeeGrid::new(4, 4, 4, COURANT_3D)
        .with_materials(&ok)
        .is_ok());
}

/// `3·S² ≤ ε_min·μ_min`: `S = 9/16` ⇒ `3S² = 243/256` 等号は受理、1 ULP でも下回れば Err
#[test]
fn the_courant_bound_is_checked_with_the_slowest_constants() {
    let edge = q(243, 256);
    let mut m = MaterialMap::vacuum(4, 4, 4);
    m.set(1, 2, 3, Material::dielectric(edge));
    assert!(YeeGrid::new(4, 4, 4, COURANT_3D).with_materials(&m).is_ok());
    let below = edge - Fix128::from_raw(0, 1);
    m.set(1, 2, 3, Material::dielectric(below));
    assert_eq!(
        err(&m, COURANT_3D),
        MaterialError::CourantViolated {
            courant: COURANT_3D,
            eps_min: below,
            mu_min: Fix128::ONE
        }
    );
    // the minima are taken separately: ε_min from one cell, μ_min from another
    let mut m2 = MaterialMap::vacuum(4, 4, 4);
    m2.set(0, 0, 0, Material::new(q(3, 1), q(1, 2), Fix128::ZERO));
    m2.set(3, 3, 3, Material::new(q(1, 2), q(3, 1), Fix128::ZERO));
    assert!(matches!(
        err(&m2, COURANT_3D),
        MaterialError::CourantViolated { .. }
    ));
    // a vacuum map rejects an S above the vacuum bound (new() alone does not)
    let fast = q(5, 8);
    let _unchecked = YeeGrid::new(4, 4, 4, fast);
    assert!(matches!(
        err(&MaterialMap::vacuum(4, 4, 4), fast),
        MaterialError::CourantViolated { .. }
    ));
    // smaller S makes the same slow cell acceptable
    m.set(1, 2, 3, Material::dielectric(q(1, 4)));
    assert!(YeeGrid::new(4, 4, 4, q(1, 4)).with_materials(&m).is_ok());
}

#[test]
fn a_material_sample_inside_the_absorber_is_rejected() {
    let pml = Absorber::GradedPml {
        depth: [2, 2, 2],
        sigma_max: Fix128::from_int(3),
    };
    let mut m = MaterialMap::vacuum(8, 8, 8);
    m.set(1, 4, 4, Material::dielectric(q(2, 1)));
    let e = YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, pml)
        .with_materials(&m)
        .unwrap_err();
    assert!(matches!(e, MaterialError::AbsorberOverlap { .. }), "{e:?}");
    // the reported sample really is absorbing and non-vacuum
    if let MaterialError::AbsorberOverlap { component, index } = e {
        let g = YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, pml);
        assert!(g.is_absorbing(component, index.0, index.1, index.2));
    }
    // a magnetic-only cell in the layer is caught on an H sample
    let mut mm = MaterialMap::vacuum(8, 8, 8);
    mm.set(4, 4, 0, Material::new(Fix128::ONE, q(2, 1), Fix128::ZERO));
    let em = YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, pml)
        .with_materials(&mm)
        .unwrap_err();
    assert!(matches!(
        em,
        MaterialError::AbsorberOverlap {
            component: Component::Hx | Component::Hy | Component::Hz,
            ..
        }
    ));
    // Uniform makes every sample absorbing: any material conflicts, vacuum does not
    let uni = Absorber::Uniform { sigma: q(1, 4) };
    let mut one = MaterialMap::vacuum(8, 8, 8);
    one.set(4, 4, 4, Material::new(Fix128::ONE, Fix128::ONE, q(1, 8)));
    assert!(YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, uni)
        .with_materials(&one)
        .is_err());
    assert!(YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, uni)
        .with_materials(&MaterialMap::vacuum(8, 8, 8))
        .is_ok());
}

fn panics<F: FnOnce()>(f: F) -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f)).is_err();
    std::panic::set_hook(prev);
    r
}

#[test]
fn map_indices_outside_the_map_panic() {
    let mut m = MaterialMap::vacuum(3, 2, 2);
    assert_eq!(m.dims(), (3, 2, 2));
    assert!(panics(|| {
        let _ = MaterialMap::vacuum(3, 2, 2).get(3, 0, 0);
    }));
    assert!(panics(|| MaterialMap::vacuum(3, 2, 2).set(
        0,
        2,
        0,
        Material::VACUUM
    )));
    assert!(panics(|| MaterialMap::vacuum(3, 2, 2).fill(
        [0, 0, 0],
        [4, 2, 2],
        Material::VACUUM
    )));
    assert!(panics(|| MaterialMap::vacuum(3, 2, 2).fill(
        [2, 0, 0],
        [1, 2, 2],
        Material::VACUUM
    )));
    // an empty box is a no-op, a full box sets everything
    m.fill([1, 1, 1], [1, 2, 2], Material::dielectric(q(2, 1)));
    assert_eq!(m, MaterialMap::vacuum(3, 2, 2));
    m.fill([0, 0, 0], [3, 2, 2], Material::dielectric(q(2, 1)));
    assert_eq!(m.get(2, 1, 1).eps_r, q(2, 1));
    assert_eq!(Material::dielectric(q(9, 4)).refractive_index(), q(3, 2));
    let g = YeeGrid::new(3, 2, 2, COURANT_3D);
    assert!(panics(|| {
        let _ = g.effective_material(Component::Ex, 3, 0, 0);
    }));
}
