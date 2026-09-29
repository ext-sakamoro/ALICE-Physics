//! Oracle: source-free Yee FDTD — 厳密離散分散関係 / 矩形空洞共振 / `∇·B` / CFL
//!
//! # なぜ連続体の分散関係と突き合わせないのか
//!
//! ⚠️ FDTD の数値位相速度は連続体の `c` と**一致しません** (数値分散)、なので
//! `ω = c·k` と比べると、許容幅に離散化誤差を丸ごと逃がすことになります 4 cell の
//! 空洞では周期が **6.07% ずれる**ので、それを通す許容幅は 1 桁の bug を素通しします
//!
//! 代わりに **FDTD の厳密離散分散関係**と突き合わせます これも閉形式です:
//!
//! ```text
//! (1/S²)·sin²(ωS/2) = sin²(kx/2) + sin²(ky/2) + sin²(kz/2)      (Δx = 1, c = 1)
//! ```
//!
//! # なぜ test に超越関数が 1 個も出てこないのか
//!
//! `H` を消去すると更新は `E^{n+1} = 2E^n − E^{n−1} − S²·(∇×∇×E)^n` になります
//! PEC 箱の上で離散 curl-curl を**厳密に対角化する**のは sine mode なので、解析
//! mode に射影した振幅 `q` はスカラー 3 項漸化式に従います:
//!
//! ```text
//! q^{n+1} + q^{n−1} = (2 − S²λ)·q^n,   λ = λx + λy + λz
//! λx = 4·sin²(mπ/2a)
//! ```
//!
//! ⚠️ **これは任意の初期条件で成り立ちます** (mode が演算子の固有ベクトルなので、
//! その成分は他の mode と混ざらない) なので「厳密な固有 mode を初期値に置く」という
//! 難しい step が要りません
//!
//! そして `λx` が**有理数になる (a, m) が存在します** — 下表は 2 階差分で直接検算した
//! ものです (`s[i+1] − 2s[i] + s[i−1] = −λx·s[i]` を全 `i` で確認):
//!
//! | a | m | 固有ベクトル (定数倍自由) | λx |
//! |---|---|---|----|
//! | 2 | 1 | `[0, 1, 0]`           | 2  |
//! | 3 | 1 | `[0, 1, 1, 0]`        | 1  |
//! | 3 | 2 | `[0, 1, −1, 0]`       | 3  |
//! | 4 | 2 | `[0, 1, 0, −1, 0]`    | 2  |
//!
//! `λ` が整数で `S` が dyadic なら `2 − S²λ` は **Q64.64 で厳密**です ⇒ 期待値に
//! 丸めが 1 度も入りません ⚠️ `clippy.toml` が `f64::asin` / `f64::sin` を禁止して
//! いるので、これは作法上の好みではなく必要条件でもあります
//!
//! # ⚠️ 期待値の出所
//!
//! 全部**閉形式**です 実装を呼んで作った期待値は 1 つもありません
//! `λ` は上表 (2 階差分で検算)、`2 − S²λ` はそこからの有理演算、周期 6 は
//! `2cos(ωS) = 1 ⇒ ωS = π/3` から、空洞共振 `f_mnp = (c/2)√((m/a)²+(n/b)²+(p/d)²)`
//! は教科書の閉形式です
//!
//! # TM 配置が厳密に 2 次元になる理由
//!
//! `Ez` だけを励起して `nz = 1` にすると `Ex` / `Ey` の更新は `k ∈ [1, nz)` = 空集合
//! なので一度も走らず、`Hz` の更新は `Ex` / `Ey` しか読まないので 0 のままです
//! ⇒ `Ex = Ey = Hz = 0` が**厳密に**保たれます (test `tm_degeneracy_is_exact` が
//! bit 一致で pin します)
//!
//! ⚠️ **だからこそ 1 向きでは足りません** その配置では `Ex` / `Ey` / `Hz` の更新 loop が
//! 0 の上しか走らないので、**中身が何であっても oracle は green のまま**です 破壊試験で
//! 実測しました (2026-09-30): `Ez` 向きだけの版に対し、`Ex` 更新の符号反転 / `Ey` 更新の
//! 符号反転 / `Hz` 更新の符号反転 / `Hz` の curl 2 項の入れ替え の **4 つが全て素通り**
//! しました ⇒ 本 file は同じ閉形式を `Ez` / `Ex` / `Ey` の **3 向き**で回します
//! (`Orientation`) 3 向きにした後は同じ 4 変異が全て red になります
//!
//! # ⚠️ `∇·B` は 0 ではありません
//!
//! 「Yee 格子なら `∇·B` は機械精度で 0」は**連続体の算術の話**です `Fix128` の乗算は
//! 切り捨てるので分配則が壊れ (`(Σf)·S` と `Σ(f·S)` が 1–3 ULP 違う、2026-09-29 実測)、
//! 残差は 0 bit 列にはなりません 本 file は **実測した上界**で pin します
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::math::Fix128;
use alice_physics::maxwell_fdtd::{cfl_limit_3d, Component, YeeGrid, COURANT_3D};

/// Raw Q64.64 value; `1` here is one ULP = 2⁻⁶⁴.
fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | i128::from(x.lo)
}

/// Distance between two `Fix128` in ULP.
fn ulp_gap(a: Fix128, b: Fix128) -> i128 {
    (raw(a) - raw(b)).abs()
}

/// A PEC sine eigenvector of the 1-D second difference, and its eigenvalue.
///
/// Entries are the exactly representable cases from the table in the header.
struct Mode {
    /// `a + 1` samples, index 0 and `a` are the PEC walls and are zero.
    vector: &'static [i64],
    /// `λ = 4·sin²(mπ/2a)`, an integer for these cases.
    lambda: i64,
}

const M2_1: Mode = Mode {
    vector: &[0, 1, 0],
    lambda: 2,
};
const M3_1: Mode = Mode {
    vector: &[0, 1, 1, 0],
    lambda: 1,
};
const M3_2: Mode = Mode {
    vector: &[0, 1, -1, 0],
    lambda: 3,
};
const M4_2: Mode = Mode {
    vector: &[0, 1, 0, -1, 0],
    lambda: 2,
};

/// Independently re-derive `λ` from the second difference, so the table in the
/// header is checked rather than trusted.
fn check_eigenvector(m: &Mode) {
    let s = m.vector;
    let a = s.len() - 1;
    assert_eq!(s[0], 0, "PEC wall must be zero");
    assert_eq!(s[a], 0, "PEC wall must be zero");
    for i in 1..a {
        let second_difference = s[i + 1] - 2 * s[i] + s[i - 1];
        assert_eq!(
            second_difference,
            -m.lambda * s[i],
            "vector {s:?} is not an eigenvector with lambda {} at i={i}",
            m.lambda
        );
    }
}

/// Which electric component carries the transverse-magnetic mode.
///
/// ⚠️ **Three orientations, not one, and that is not decoration.** A single
/// orientation exercises one `E` component and two `H` components, leaving the
/// other three update loops running on nothing but zeros. A destruction test
/// measured this directly on 2026-09-30: with only the `Ez` orientation
/// present, a **sign flip inside the `Ex` update, inside the `Ey` update, or
/// inside the `Hz` update — and a swap of the two `Hz` curl terms — all left
/// every oracle green**. Cycling the orientation covers all six components with
/// the same closed form, and turns the curl's cyclic symmetry into something
/// the oracle checks rather than assumes.
#[derive(Clone, Copy, Debug)]
enum Orientation {
    /// `Ez` in an `a × b × 1` cavity; `Hx`, `Hy` active; `Ex`, `Ey`, `Hz` zero.
    Ez,
    /// `Ex` in a `1 × a × b` cavity; `Hy`, `Hz` active; `Ey`, `Ez`, `Hx` zero.
    Ex,
    /// `Ey` in a `b × 1 × a` cavity; `Hz`, `Hx` active; `Ez`, `Ex`, `Hy` zero.
    Ey,
}

impl Orientation {
    const ALL: [Self; 3] = [Self::Ez, Self::Ex, Self::Ey];

    /// The excited component.
    const fn driven(self) -> Component {
        match self {
            Self::Ez => Component::Ez,
            Self::Ex => Component::Ex,
            Self::Ey => Component::Ey,
        }
    }

    /// The three components that must stay exactly zero.
    const fn silent(self) -> [Component; 3] {
        match self {
            Self::Ez => [Component::Ex, Component::Ey, Component::Hz],
            Self::Ex => [Component::Ey, Component::Ez, Component::Hx],
            Self::Ey => [Component::Ez, Component::Ex, Component::Hy],
        }
    }

    /// Lattice cell counts for a cavity of `la × lb` cells in the mode plane.
    const fn dims(self, la: usize, lb: usize) -> (usize, usize, usize) {
        match self {
            Self::Ez => (la, lb, 1),
            Self::Ex => (1, la, lb),
            Self::Ey => (lb, 1, la),
        }
    }

    /// Sample index of the driven component for mode indices `(a, b)`.
    const fn at(self, a: usize, b: usize) -> (usize, usize, usize) {
        match self {
            Self::Ez => (a, b, 0),
            Self::Ex => (0, a, b),
            Self::Ey => (b, 0, a),
        }
    }
}

/// Build a TM cavity in the given orientation: driven component `= sa·sb`.
fn tm_cavity(o: Orientation, sa: &Mode, sb: &Mode, courant: Fix128) -> YeeGrid {
    let la = sa.vector.len() - 1;
    let lb = sb.vector.len() - 1;
    let (nx, ny, nz) = o.dims(la, lb);
    let mut grid = YeeGrid::new(nx, ny, nz, courant);
    for a in 0..=la {
        for b in 0..=lb {
            let v = Fix128::from_int(sa.vector[a] * sb.vector[b]);
            let (i, j, k) = o.at(a, b);
            grid.set(o.driven(), i, j, k, v);
        }
    }
    grid
}

/// Project the driven component onto the analytic mode shape.
fn project(grid: &YeeGrid, o: Orientation, sa: &Mode, sb: &Mode) -> Fix128 {
    let la = sa.vector.len() - 1;
    let lb = sb.vector.len() - 1;
    let mut acc = Fix128::ZERO;
    for a in 0..=la {
        for b in 0..=lb {
            let w = sa.vector[a] * sb.vector[b];
            if w != 0 {
                let (i, j, k) = o.at(a, b);
                acc = acc + grid.get(o.driven(), i, j, k) * Fix128::from_int(w);
            }
        }
    }
    acc
}

// ---------------------------------------------------------------------------
// O0 — the table the other oracles rest on
// ---------------------------------------------------------------------------

#[test]
fn tabulated_eigenvectors_are_eigenvectors() {
    for m in [&M2_1, &M3_1, &M3_2, &M4_2] {
        check_eigenvector(m);
    }
}

#[test]
fn courant_constant_is_nine_sixteenths_and_under_the_cfl_limit() {
    // 9/16 built two independent ways: the crate constant, and from_ratio.
    assert_eq!(COURANT_3D, Fix128::from_ratio(9, 16));
    // 9/16 = 9·2^60 exactly.
    assert_eq!(raw(COURANT_3D), 9 * (1i128 << 60));
    // Under 1/sqrt(3) = 0.577350269..., and the next dyadic up (19/32 = 0.59375)
    // is over it, so 9/16 is the largest sixteenth that fits.
    let limit = cfl_limit_3d();
    assert!(COURANT_3D < limit, "9/16 must be under the 3-D CFL limit");
    assert!(
        Fix128::from_ratio(19, 32) > limit,
        "19/32 must be over the limit, else 9/16 is not the best dyadic"
    );
}

// ---------------------------------------------------------------------------
// O1 — exact discrete dispersion relation, as a three-term recurrence
// ---------------------------------------------------------------------------

/// `q^{n+1} + q^{n−1} = (2 − S²λ)·q^n` for five (cavity, mode, S) combinations.
///
/// Expected value is closed form: `λ` from the header table, `S` dyadic, so
/// `2 − S²λ` is exact in Q64.64 with no rounding anywhere in the reference.
///
/// ⚠️ Tolerance budget — **constant in `n`, deliberately**. The recurrence
/// relates three *consecutive* states, so only the truncations introduced by
/// the steps between them can contribute; nothing accumulates from earlier
/// steps. One step truncates once per sample in `H` and once in `E`, and the
/// projection is a signed unit-weight sum over `(a+1)(b+1)` samples, so the
/// residual is bounded by `2·(a+1)(b+1)` ULP however long the run is.
///
/// ⚠️ A budget that grew with `n` would defeat the purpose: secular drift is
/// exactly the failure this oracle exists to catch, and a growing bound hides
/// it. Measured worst cases are 0 ULP for the dyadic `S = 1/2` integer-λ cases,
/// 1 ULP for λ = 5, and 3 ULP for `S = 9/16` (2026-09-29).
#[test]
fn projected_amplitude_follows_the_exact_discrete_dispersion_relation() {
    let half = Fix128::from_ratio(1, 2);
    let cases: [(&Mode, &Mode, Fix128, &str); 5] = [
        (&M4_2, &M4_2, half, "4x4 m=n=2, S=1/2, lambda=4"),
        (&M3_1, &M3_1, half, "3x3 m=n=1, S=1/2, lambda=2"),
        (&M3_2, &M3_2, half, "3x3 m=n=2, S=1/2, lambda=6"),
        (&M2_1, &M3_2, half, "2x3, S=1/2, lambda=5"),
        (&M3_1, &M3_2, COURANT_3D, "3x3 m=1,n=2, S=9/16, lambda=4"),
    ];

    for (sa, sb, s, label) in cases {
        let lambda = sa.lambda + sb.lambda;
        // Closed form: 2 - S^2 * lambda.
        let expected = Fix128::from_int(2) - s * s * Fix128::from_int(lambda);
        let samples = sa.vector.len() * sb.vector.len();
        let steps = 40usize;

        // Every orientation must give the same coefficient: the curl operator
        // is cyclic in (x, y, z), so the discrete dispersion relation cannot
        // depend on which axis the mode was built along.
        for o in Orientation::ALL {
            let mut grid = tm_cavity(o, sa, sb, s);
            let mut q_prev = project(&grid, o, sa, sb);
            grid.step();
            let mut q_curr = project(&grid, o, sa, sb);

            let mut worst: i128 = 0;
            for n in 1..=steps {
                grid.step();
                let q_next = project(&grid, o, sa, sb);
                // q_next + q_prev must equal expected * q_curr
                let lhs = q_next + q_prev;
                let rhs = expected * q_curr;
                let gap = ulp_gap(lhs, rhs);
                worst = worst.max(gap);
                let budget = 2 * samples as i128;
                assert!(
                    gap <= budget,
                    "{label} [{o:?}]: step {n} residual {gap} ULP exceeds budget \
                     {budget} ULP (lhs raw {}, rhs raw {})",
                    raw(lhs),
                    raw(rhs)
                );
                q_prev = q_curr;
                q_curr = q_next;
            }
            println!(
                "{label} [{o:?}]: expected coeff raw={}, worst residual {worst} ULP",
                raw(expected)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// O2 — rectangular cavity resonance, tied to f_mnp
// ---------------------------------------------------------------------------

/// A 4×4 cavity driven at `m = n = 2` with `S = 1/2` has `2cos(ωΔt) = 2 − S²λ
/// = 2 − (1/4)(4) = 1`, so `ωΔt = π/3` and the state repeats **every 6 steps
/// exactly**. No transcendental is needed to state that.
///
/// The link to the closed form in the brief: the continuum resonance is
/// `f_mnp = (c/2)·√((m/a)² + (n/b)² + (p/d)²)`, here
/// `f = (1/2)√((2/4)² + (2/4)²)`, so `f² = 1/8` exactly. The discrete period is
/// `T = 6·S = 3`, so `T²·f² = 9/8` — a rational identity that captures the
/// 6.07% numerical dispersion of this coarse lattice **as a number**, instead
/// of hiding it inside a tolerance.
#[test]
fn four_by_four_cavity_repeats_every_six_steps() {
    let s = Fix128::from_ratio(1, 2);
    let lambda = M4_2.lambda + M4_2.lambda;
    assert_eq!(lambda, 4, "this test's arithmetic assumes lambda = 4");
    // 2 - S^2*lambda == 1  =>  omega*dt = pi/3  =>  period 6
    assert_eq!(
        Fix128::from_int(2) - s * s * Fix128::from_int(lambda),
        Fix128::ONE,
        "period-6 argument requires the recurrence coefficient to be exactly 1"
    );

    // Continuum closed form, as rationals: f^2 = (1/4)((2/4)^2 + (2/4)^2) = 1/8.
    let f_sq = Fix128::from_ratio(1, 4)
        * (Fix128::from_ratio(2, 4) * Fix128::from_ratio(2, 4)
            + Fix128::from_ratio(2, 4) * Fix128::from_ratio(2, 4));
    assert_eq!(f_sq, Fix128::from_ratio(1, 8));
    // Discrete period T = 6S = 3, so T^2 f^2 = 9/8.
    let t_discrete = Fix128::from_int(6) * s;
    assert_eq!(t_discrete, Fix128::from_int(3));
    assert_eq!(t_discrete * t_discrete * f_sq, Fix128::from_ratio(9, 8));

    // Now the simulation must actually have that period — and **bit exactly**.
    //
    // ⚠️ Zero tolerance is a measured fact, not optimism. An a-priori argument
    // that the state loses one low bit per step (integer initial data, `S = 1/2`
    // halving each product) predicts drift from about step 64; the measurement
    // says otherwise and the measurement wins. The reason is that a period-6
    // orbit visits a *finite set of exact states* and returns to it, so there is
    // no progressive bit loss to accumulate. Measured 0 ULP out to 144 steps
    // (2026-09-29), so this asserts equality of the whole lattice.
    let periods = 24usize;
    for o in Orientation::ALL {
        let mut grid = tm_cavity(o, &M4_2, &M4_2, s);
        let initial = grid.clone();
        let (i1, j1, k1) = o.at(1, 1);
        for p in 1..=periods {
            for _ in 0..6 {
                grid.step();
            }
            assert_eq!(
                grid, initial,
                "{o:?} period {p}: the lattice must return to its initial state bit-exactly"
            );
            // The claim would be vacuous on a field that had decayed to zero.
            assert!(
                grid.get(o.driven(), i1, j1, k1) != Fix128::ZERO,
                "{o:?}: field went to zero, the period test would be vacuous"
            );
        }
        println!(
            "cavity period 6 [{o:?}]: bit-exact return over {periods} periods ({} steps)",
            periods * 6
        );
    }
}

/// The period-6 claim would be vacuous if the field never moved. Three steps in
/// (half a period) the mode must be exactly inverted: `q^3 = −q^0`.
#[test]
fn half_a_period_inverts_the_mode() {
    let s = Fix128::from_ratio(1, 2);
    for o in Orientation::ALL {
        let mut grid = tm_cavity(o, &M4_2, &M4_2, s);
        let q0 = project(&grid, o, &M4_2, &M4_2);
        assert!(
            q0 != Fix128::ZERO,
            "{o:?}: initial projection must be non-zero"
        );
        for _ in 0..3 {
            grid.step();
        }
        let q3 = project(&grid, o, &M4_2, &M4_2);
        let gap = ulp_gap(q3, Fix128::ZERO - q0);
        assert!(gap <= 2 * 25, "{o:?}: q3 should be -q0, off by {gap} ULP");
    }
}

// ---------------------------------------------------------------------------
// O3 — invariant guards
// ---------------------------------------------------------------------------

/// `∇·B` must stay at a small bounded residual (not zero — see the header).
#[test]
fn div_b_stays_bounded() {
    for o in Orientation::ALL {
        let mut grid = tm_cavity(o, &M4_2, &M4_2, COURANT_3D);
        assert_eq!(
            grid.max_abs_div_b(),
            Fix128::ZERO,
            "the initial condition has H = 0, so div B starts at exactly zero"
        );
        let steps = 500usize;
        for n in 1..=steps {
            grid.step();
            let d = grid.max_abs_div_b();
            // Budget: the only source of a non-zero residual is that multiplying by
            // S truncates and so does not distribute over the face sum (measured
            // 1-3 ULP per application). One step applies it once per face, giving a
            // bound linear in the step count. Measured growth is 0.93 ULP/step,
            // flat from step 100 to step 1000, so 2 ULP/step leaves ~2x of margin
            // while still failing on anything that grows faster than linearly.
            let budget = 2 * n as i128 + 16;
            assert!(
                raw(d) <= budget,
                "{o:?} step {n}: max |div B| = {} ULP exceeds budget {budget} ULP",
                raw(d)
            );
        }
        println!(
            "div B [{o:?}] after {steps} steps: {} ULP",
            raw(grid.max_abs_div_b())
        );
    }
}

/// `Ex`, `Ey`, `Hz` must remain **bit-exactly** zero for a TM_mn0 lattice.
///
/// This is the structural invariant that catches a swapped index or a sign
/// error in the curl: any leak between components shows up here immediately,
/// and it is an exact test with no tolerance at all.
#[test]
fn tm_degeneracy_is_exact() {
    for o in Orientation::ALL {
        let mut grid = tm_cavity(o, &M4_2, &M4_2, COURANT_3D);
        for n in 0..200 {
            grid.step();
            for comp in o.silent() {
                let (ni, nj, nk) = grid.component_dims(comp);
                for i in 0..ni {
                    for j in 0..nj {
                        for k in 0..nk {
                            assert_eq!(
                                grid.get(comp, i, j, k),
                                Fix128::ZERO,
                                "{o:?} step {n}: {comp:?}[{i}][{j}][{k}] left the TM subspace"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// PEC: tangential `E` on every wall must stay **bit-exactly** zero.
#[test]
fn pec_walls_hold_tangential_e_at_zero() {
    let (la, lb) = (M3_2.vector.len() - 1, M3_1.vector.len() - 1);
    for o in Orientation::ALL {
        let mut grid = tm_cavity(o, &M3_2, &M3_1, COURANT_3D);
        for n in 0..200 {
            grid.step();
            // The driven component is tangential to the four walls that bound
            // its mode plane, so it must be zero on both ends of both axes.
            for b in 0..=lb {
                for a in [0, la] {
                    let (i, j, k) = o.at(a, b);
                    assert_eq!(
                        grid.get(o.driven(), i, j, k),
                        Fix128::ZERO,
                        "{o:?} step {n}"
                    );
                }
            }
            for a in 0..=la {
                for b in [0, lb] {
                    let (i, j, k) = o.at(a, b);
                    assert_eq!(
                        grid.get(o.driven(), i, j, k),
                        Fix128::ZERO,
                        "{o:?} step {n}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// O4 — CFL, both sides
// ---------------------------------------------------------------------------

/// Below the limit the field stays bounded; above it, it must blow up.
///
/// ⚠️ The second half is what stops the first half from being vacuous: a solver
/// that quietly damps everything to zero would pass "bounded" and fail here.
#[test]
fn cfl_limit_separates_bounded_from_divergent() {
    fn run(o: Orientation, courant: Fix128, steps: usize) -> Fix128 {
        let (nx, ny, nz) = o.dims(16, 16);
        let mut grid = YeeGrid::new(nx, ny, nz, courant);
        // Point impulse: excites every mode, including the highest one, which is
        // the mode the stability limit is about.
        let (i, j, k) = o.at(7, 9);
        grid.set(o.driven(), i, j, k, Fix128::ONE);
        for _ in 0..steps {
            grid.step();
        }
        grid.max_abs_field()
    }

    // Both sides of the limit, in all three orientations.
    for o in Orientation::ALL {
        let bounded = run(o, COURANT_3D, 500);
        assert!(
            bounded <= Fix128::from_int(2),
            "{o:?}: S = 9/16 is under the limit, field must stay bounded (got {bounded})"
        );
        let blows_up = run(o, Fix128::ONE, 20);
        assert!(
            blows_up > Fix128::from_int(1_000_000_000_000),
            "{o:?}: S = 1 is over the limit, field must diverge (got {blows_up})"
        );
    }

    // The impulse starts at 1.0 and must not grow. Measured peak over 4000
    // steps is 0.313, so 2.0 is a real ceiling rather than a nominal one.
    let stable = run(Orientation::Ez, COURANT_3D, 4000);
    assert!(
        stable <= Fix128::from_int(2),
        "S = 9/16 is under the CFL limit, field must stay bounded (got {stable})"
    );

    // S = 1 exceeds the 2-D limit 1/sqrt(2) = 0.7071 for this lattice, so the
    // amplitude grows by about 5.5x per step.
    //
    // ⚠️ 20 steps, not more: `Fix128` multiplication **wraps** rather than
    // saturating, and this run reaches the 2^63 ceiling at about step 27. Past
    // that the maximum sits at ~9.2e18 forever, which reads like a large number
    // but is wrapped arithmetic, so a threshold test placed there would be
    // asserting on garbage. The upper bound below pins the reading to the
    // pre-wrap regime. Measured 5.5e13 at step 20.
    let divergent = run(Orientation::Ez, Fix128::ONE, 20);
    assert!(
        divergent > Fix128::from_int(1_000_000_000_000),
        "S = 1 is over the CFL limit, field must diverge (got {divergent})"
    );
    assert!(
        divergent < Fix128::from_int(1_000_000_000_000_000_000),
        "reading must stay below the Fix128 wrap point to mean anything (got {divergent})"
    );
    println!("CFL: bounded max = {stable}, divergent max = {divergent}");
}
