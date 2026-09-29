//! Oracle: Yee FDTD — 厳密離散分散関係 / 矩形空洞共振 / `∇·B` / CFL / Gauss / PML
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
use alice_physics::maxwell_fdtd::{
    cfl_limit_3d, loss_coefficients, theoretical_pml_reflection, Absorber, Component, YeeGrid,
    COURANT_3D,
};

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

// ---------------------------------------------------------------------------
// O5 — Gauss's law and charge continuity
// ---------------------------------------------------------------------------

/// A 4×4×4 lattice, the smallest that has a 3×3×3 block of interior nodes.
fn source_cavity(courant: Fix128) -> YeeGrid {
    YeeGrid::new(4, 4, 4, courant)
}

/// Largest `|ρ|` over the interior nodes, to show a test is not vacuous.
fn max_abs_charge(grid: &YeeGrid) -> Fix128 {
    let (ni, nj, nk) = grid.interior_node_dims();
    let mut worst = Fix128::ZERO;
    for i in 1..=ni {
        for j in 1..=nj {
            for k in 1..=nk {
                let q = grid.charge(i, j, k).abs();
                if q > worst {
                    worst = q;
                }
            }
        }
    }
    worst
}

/// `∇·E = ρ` holds **bit-exactly** while `S·J` is exactly representable.
///
/// Gauss's law is never imposed here — it survives because `∇·(∇×H)`
/// telescopes to zero and because `ρ` is marched by the *same* discrete
/// divergence that the `−S·J` term in Ampère's law feeds into `∇·E`. The two
/// cancel term by term, so with `S = 1/2` and integer `J` (no truncation
/// anywhere) the residual must be the zero bit pattern, not a small number.
///
/// ⚠️ The run length is bounded by arithmetic, not by physics: every step adds
/// fractional bits to the field values, so the dyadic exactness eventually runs
/// out and truncation begins. The companion test below pins what happens on the
/// other side of that.
#[test]
fn gauss_law_is_bit_exact_for_an_exactly_representable_current() {
    let s = Fix128::from_ratio(1, 2);
    let mut grid = source_cavity(s);
    // A little of each orientation, with a divergence that is not zero.
    grid.set_current(Component::Ex, 1, 2, 2, Fix128::from_int(3));
    grid.set_current(Component::Ey, 2, 1, 2, Fix128::from_int(-5));
    grid.set_current(Component::Ez, 2, 2, 1, Fix128::from_int(2));
    grid.set_current(Component::Ex, 2, 3, 1, Fix128::from_int(7));

    assert_eq!(
        grid.max_abs_gauss_residual(),
        Fix128::ZERO,
        "E = 0 and rho = 0 satisfy Gauss's law exactly to begin with"
    );

    for n in 1..=30 {
        grid.step();
        assert_eq!(
            grid.max_abs_gauss_residual(),
            Fix128::ZERO,
            "step {n}: div E - rho must stay at the zero bit pattern"
        );
    }

    // ⚠️ Not vacuous: charge really accumulated and the field really moved.
    assert!(
        max_abs_charge(&grid) > Fix128::ZERO,
        "the current must have deposited charge"
    );
    assert!(
        grid.max_abs_field() > Fix128::ZERO,
        "the current must have driven a field"
    );
    println!(
        "Gauss exact run: max |rho| = {}, max |field| = {}",
        max_abs_charge(&grid),
        grid.max_abs_field()
    );
}

/// A current that stops inside the lattice piles charge up at its two ends,
/// at exactly `∓S·J` per step, and nowhere else.
///
/// Closed form with no tolerance: a single `Ex` edge at `(i, j, k)` enters
/// `∇·J` with `+1` at node `(i, j, k)` and `−1` at node `(i+1, j, k)`, so after
/// `n` steps `ρ` is `−n·S·J` and `+n·S·J` there and the zero bit pattern at
/// every other node. Total charge stays exactly zero.
///
/// ⚠️ This is what stops the exactness test above from passing on a broken
/// stencil: that one only asks the two sides to agree, and they would still
/// agree if `∇·J` and the `J` term were wrong in the *same* way. Here the
/// deposited value itself is predicted.
#[test]
fn a_current_that_ends_deposits_exactly_the_charge_that_left() {
    let s = Fix128::from_ratio(1, 2);
    let j = Fix128::from_int(3);
    let mut grid = source_cavity(s);
    grid.set_current(Component::Ex, 1, 2, 2, j);

    let (ni, nj, nk) = grid.interior_node_dims();
    for n in 1..=20i64 {
        grid.step();
        let expect = s * j * Fix128::from_int(n);
        for i in 1..=ni {
            for jj in 1..=nj {
                for k in 1..=nk {
                    let want = match (i, jj, k) {
                        (1, 2, 2) => -expect,
                        (2, 2, 2) => expect,
                        _ => Fix128::ZERO,
                    };
                    assert_eq!(grid.charge(i, jj, k), want, "step {n}: rho[{i}][{jj}][{k}]");
                }
            }
        }
        assert_eq!(
            grid.total_charge(),
            Fix128::ZERO,
            "step {n}: the two ends must cancel exactly"
        );
    }
    assert!(
        grid.charge(2, 2, 2) > Fix128::ZERO,
        "the deposited charge must be non-zero for this to mean anything"
    );
}

/// A closed current loop has `∇·J = 0`, so it must create **no** charge at all.
///
/// The loop is the four edges bounding one cell face, traversed head to tail.
/// Every node it touches receives `+J` from one edge and `−J` from the next, so
/// the charge stays at the zero bit pattern while the loop still drives a
/// field — which is the half that stops this from being a test of a solver that
/// does nothing.
#[test]
fn a_closed_current_loop_creates_no_charge() {
    let s = COURANT_3D;
    let j = Fix128::from_ratio(3, 7); // deliberately not dyadic
    let mut grid = source_cavity(s);
    let (i, jj, k) = (1usize, 1usize, 2usize);
    grid.set_current(Component::Ex, i, jj, k, j);
    grid.set_current(Component::Ey, i + 1, jj, k, j);
    grid.set_current(Component::Ex, i, jj + 1, k, -j);
    grid.set_current(Component::Ey, i, jj, k, -j);

    let (ni, nj, nk) = grid.interior_node_dims();
    for i2 in 1..=ni {
        for j2 in 1..=nj {
            for k2 in 1..=nk {
                assert_eq!(
                    grid.div_j(i2, j2, k2),
                    Fix128::ZERO,
                    "the loop must be divergence free at [{i2}][{j2}][{k2}]"
                );
            }
        }
    }

    for n in 1..=50 {
        grid.step();
        assert_eq!(
            max_abs_charge(&grid),
            Fix128::ZERO,
            "step {n}: a solenoidal current must not create charge"
        );
    }
    assert!(
        grid.max_abs_field() > Fix128::ZERO,
        "the loop must still drive a field"
    );
    println!(
        "solenoidal loop: max |field| after 50 steps = {}",
        grid.max_abs_field()
    );
}

/// With a current whose `S·J` is **not** exact, Gauss's law degrades to a
/// bounded residual — and the bound is linear in the step count.
///
/// ⚠️ This is the companion to the exactness test, and it is the honest half:
/// the `E` update truncates one product per edge while the `ρ` update truncates
/// one product of their sum, so the two no longer cancel bit for bit.
#[test]
fn gauss_residual_is_bounded_when_the_current_is_not_exactly_representable() {
    let mut grid = source_cavity(COURANT_3D);
    grid.set_current(Component::Ex, 1, 2, 2, Fix128::from_ratio(3, 7));
    grid.set_current(Component::Ey, 2, 1, 2, Fix128::from_ratio(-5, 11));
    grid.set_current(Component::Ez, 2, 2, 1, Fix128::from_ratio(2, 13));

    let steps = 500usize;
    for n in 1..=steps {
        grid.step();
        let r = grid.max_abs_gauss_residual();
        let budget = 2 * n as i128 + 16;
        assert!(
            raw(r) <= budget,
            "step {n}: |div E - rho| = {} ULP exceeds budget {budget} ULP",
            raw(r)
        );
    }
    let final_r = raw(grid.max_abs_gauss_residual());
    println!("Gauss residual after {steps} steps: {final_r} ULP");
    // ⚠️ And it must actually be non-zero: if it were zero the exactness test
    // above would be the only one needed and this budget would be decoration.
    assert!(
        final_r > 0,
        "a non-representable current must leave a visible residual"
    );
}

/// Sources must not disturb `∇·B`: the `H` update never sees `J` or `ρ`.
#[test]
fn sources_do_not_disturb_div_b() {
    let mut grid = source_cavity(COURANT_3D);
    grid.set_current(Component::Ez, 2, 2, 1, Fix128::from_ratio(2, 13));
    for n in 1..=500 {
        grid.step();
        let d = grid.max_abs_div_b();
        let budget = 2 * n as i128 + 16;
        assert!(
            raw(d) <= budget,
            "step {n}: max |div B| = {} ULP exceeds budget {budget} ULP",
            raw(d)
        );
    }
    println!(
        "div B with sources after 500 steps: {} ULP",
        raw(grid.max_abs_div_b())
    );
}

/// Allocating zero sources must be **bit-invisible**: a lattice told about a
/// zero current has to step to the same bit pattern as one that was never told.
///
/// This is what keeps the source-free oracles above binding now that `J` and
/// `ρ` exist — the source pass is skipped entirely when there are no sources,
/// and when it does run on zeros it subtracts exact zeros.
#[test]
fn a_zero_source_is_bit_invisible() {
    for o in Orientation::ALL {
        let mut bare = tm_cavity(o, &M4_2, &M4_2, COURANT_3D);
        let mut with_zero = tm_cavity(o, &M4_2, &M4_2, COURANT_3D);
        // ⚠️ A TM cavity is one cell thick, so it has no interior nodes and no
        // `ρ` at all; the allocation is forced through an edge instead.
        let (i, j, k) = o.at(1, 1);
        with_zero.set_current(o.driven(), i, j, k, Fix128::ZERO);
        for n in 0..120 {
            bare.step();
            with_zero.step();
            for comp in [
                Component::Ex,
                Component::Ey,
                Component::Ez,
                Component::Hx,
                Component::Hy,
                Component::Hz,
            ] {
                let (ni, nj, nk) = bare.component_dims(comp);
                for i in 0..ni {
                    for j in 0..nj {
                        for k in 0..nk {
                            assert_eq!(
                                bare.get(comp, i, j, k),
                                with_zero.get(comp, i, j, k),
                                "{o:?} step {n}: {comp:?}[{i}][{j}][{k}]"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// A current on a PEC-tangential edge is rejected rather than discarded.
#[test]
#[should_panic(expected = "tangential to a PEC wall")]
fn a_current_on_a_pec_edge_is_rejected() {
    let mut grid = source_cavity(COURANT_3D);
    // `Ex` at j = 0 lies in the y = 0 PEC wall, so the update never writes it.
    grid.set_current(Component::Ex, 1, 0, 2, Fix128::ONE);
}

/// `J` lives on electric edges; asking for it on a magnetic face is a mistake.
#[test]
#[should_panic(expected = "lives on electric edges")]
fn a_current_on_a_magnetic_face_is_rejected() {
    let mut grid = source_cavity(COURANT_3D);
    grid.set_current(Component::Hz, 1, 1, 1, Fix128::ONE);
}

// ---------------------------------------------------------------------------
// O6 — absorbing boundary (split-field PML) and the lossy update it rests on
// ---------------------------------------------------------------------------

/// Every component, for the sweeps that have to cover all six.
const EVERY: [Component; 6] = [
    Component::Ex,
    Component::Ey,
    Component::Ez,
    Component::Hx,
    Component::Hy,
    Component::Hz,
];

/// A divergence-free seed: the four `E` edges bounding one cell face, head to
/// tail, so `∇·E = 0` at every node.
///
/// ⚠️ **A single-edge impulse is not usable for an absorption measurement.**
/// Setting one edge leaves `∇·E ≠ 0` with `ρ = 0`, which is a curl-free
/// (longitudinal) field; `∇×(∇×E)` annihilates it, so it never propagates and
/// sits in the lattice forever. A PML absorbs outgoing waves and has no
/// business removing a static interior field, so the residual plateaus and the
/// PML measures as if it did nothing. Measured 2026-09-30: a single `Ez` edge
/// leaves `max |field|` at 0.334 for at least 200 steps with a PML on all six
/// walls, identical to PEC, while this seed leaves a late envelope of 1.20e-2
/// against PEC's 0.215. The tell is [`YeeGrid::max_abs_gauss_residual`] being
/// non-zero at step zero, which the absorption test below asserts before it
/// measures.
fn seed_divergence_free_loop(grid: &mut YeeGrid, i: usize, j: usize, k: usize) {
    let v = Fix128::ONE;
    grid.set(Component::Ex, i, j, k, v);
    grid.set(Component::Ey, i + 1, j, k, v);
    grid.set(Component::Ex, i, j + 1, k, -v);
    grid.set(Component::Ey, i, j, k, -v);
}

/// Largest `|field|` over the samples the lossy update does **not** touch.
fn max_abs_field_lossless(grid: &YeeGrid) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for c in EVERY {
        let (ni, nj, nk) = grid.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    if grid.is_absorbing(c, i, j, k) {
                        continue;
                    }
                    let v = grid.get(c, i, j, k).abs();
                    if v > worst {
                        worst = v;
                    }
                }
            }
        }
    }
    worst
}

/// Whether all six `H` faces of cell `(i, j, k)` are on the lossless update.
fn cell_is_lossless(grid: &YeeGrid, i: usize, j: usize, k: usize) -> bool {
    !grid.is_absorbing(Component::Hx, i, j, k)
        && !grid.is_absorbing(Component::Hx, i + 1, j, k)
        && !grid.is_absorbing(Component::Hy, i, j, k)
        && !grid.is_absorbing(Component::Hy, i, j + 1, k)
        && !grid.is_absorbing(Component::Hz, i, j, k)
        && !grid.is_absorbing(Component::Hz, i, j, k + 1)
}

/// Largest `|∇·B|` over the cells whose six faces are all lossless.
fn max_abs_div_b_lossless(grid: &YeeGrid) -> Fix128 {
    let (nx, ny, nz) = grid.dims();
    let mut worst = Fix128::ZERO;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if !cell_is_lossless(grid, i, j, k) {
                    continue;
                }
                let d = grid.div_b(i, j, k).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
    }
    worst
}

/// The loss coefficients are the closed form, with no rounding at all.
///
/// `σ = 12`, `S = 1/2` gives `a = σS/2 = 3`, hence `ca = (1−3)/(1+3) = −1/2`
/// and `cb = S/(1+a) = 1/8`. Both are dyadic, so Q64.64 holds them exactly and
/// the recurrence oracle below has a reference with no rounding in it.
#[test]
fn loss_coefficients_are_the_closed_form() {
    let s = Fix128::from_ratio(1, 2);
    let (ca, cb) = loss_coefficients(Fix128::from_int(12), s);
    assert_eq!(ca, Fix128::from_ratio(-1, 2), "ca = (1 - a)/(1 + a), a = 3");
    assert_eq!(cb, Fix128::from_ratio(1, 8), "cb = S/(1 + a)");
    // sigma = 0 must be the loss-free update exactly, or the lossless arm and
    // the lossy arm would disagree on a zero-conductivity sample.
    let (ca0, cb0) = loss_coefficients(Fix128::ZERO, COURANT_3D);
    assert_eq!(ca0, Fix128::ONE);
    assert_eq!(cb0, COURANT_3D);
}

/// The lossy update follows the exact three-term recurrence
/// `q^{n+1} = (2c − b²λ)·q^n − c²·q^{n−1}`, at **0 ULP**.
///
/// Eliminating `H` from the lossy leap-frog gives that recurrence, with the
/// source-free `q^{n+1} = (2 − S²λ)q^n − q^{n−1}` as the `c = 1`, `b = S` case.
/// With `σ = 12`, `S = 1/2` the coefficients are `c = −1/2` and `b = 1/8`, and
/// with the `4×4` mode `λ = 2 + 2 = 4` the recurrence coefficient is
/// `2c − b²λ = −1 − 4/64 = −17/16` and `c² = 1/4` — every one of them dyadic,
/// so the reference has no rounding in it.
///
/// ⚠️ 16 steps, and that limit is arithmetic rather than physical: `b = 1/8`
/// adds three fractional bits per step, so around step 17 the values stop being
/// exactly representable and the residual becomes a few ULP (measured: 0 ULP
/// through step 16, then within ±4 ULP out to step 40, 2026-09-30). A budget
/// that covered the whole run would hide the thing this oracle exists to see,
/// so the exact window is asserted exactly and the tail is left out.
///
/// ⚠️ All three orientations, for the reason the header gives: one orientation
/// leaves three of the twelve split updates running on nothing but zeros.
#[test]
fn lossy_update_follows_the_exact_three_term_recurrence() {
    let s = Fix128::from_ratio(1, 2);
    let sigma = Fix128::from_int(12);
    let coefficient = Fix128::from_ratio(-17, 16);
    let c_squared = Fix128::from_ratio(1, 4);

    for o in Orientation::ALL {
        let la = M4_2.vector.len() - 1;
        let lb = M4_2.vector.len() - 1;
        let (nx, ny, nz) = o.dims(la, lb);
        let mut grid = YeeGrid::new_with_absorber(nx, ny, nz, s, Absorber::Uniform { sigma });
        for a in 0..=la {
            for b in 0..=lb {
                let v = Fix128::from_int(M4_2.vector[a] * M4_2.vector[b]);
                let (i, j, k) = o.at(a, b);
                grid.set(o.driven(), i, j, k, v);
            }
        }
        let mut q_prev = project(&grid, o, &M4_2, &M4_2);
        grid.step();
        let mut q_now = project(&grid, o, &M4_2, &M4_2);
        for n in 2..=16 {
            grid.step();
            let q_next = project(&grid, o, &M4_2, &M4_2);
            let expected = coefficient * q_now - c_squared * q_prev;
            assert_eq!(
                q_next, expected,
                "{o:?} step {n}: lossy recurrence must hold exactly while the values are dyadic"
            );
            q_prev = q_now;
            q_now = q_next;
        }
        // ⚠️ Not vacuous: the mode has to have actually decayed, or a solver
        // that froze the field would satisfy the recurrence trivially at zero.
        assert!(
            q_now.abs() > Fix128::ZERO && q_now.abs() < Fix128::from_ratio(1, 16),
            "{o:?}: the mode must be alive but heavily damped (got {q_now})"
        );
    }
}

/// A uniformly lossy lattice really does damp, and a lossless one does not.
///
/// **Closed form** — the characteristic roots of the recurrence above have
/// product `c²`, so a mode's amplitude falls off geometrically at `|c|` per
/// step; a lossless lattice conserves energy and cannot fall off at all.
///
/// **Measured** (2026-09-30) — `σ = 2` at `S = 9/16` takes a divergence-free
/// seed from 1 to 1.5e-2 by step 10 and 1.9e-6 by step 60, a ratio of about
/// 0.82 per step, while the loss-free control is still at 0.44. The declared
/// bounds are 1e-5 and 1e-2, leaving roughly 5× and 44× of margin.
#[test]
fn uniform_loss_damps_and_no_loss_does_not() {
    let sigma = Fix128::from_int(2);
    let mut lossy = YeeGrid::new_with_absorber(8, 8, 8, COURANT_3D, Absorber::Uniform { sigma });
    let mut free = YeeGrid::new(8, 8, 8, COURANT_3D);
    seed_divergence_free_loop(&mut lossy, 3, 3, 4);
    seed_divergence_free_loop(&mut free, 3, 3, 4);
    for _ in 0..60 {
        lossy.step();
        free.step();
    }
    let damped = lossy.max_abs_field();
    let undamped = free.max_abs_field();
    assert!(
        damped < Fix128::from_ratio(1, 100_000),
        "sigma = 2 must damp the seed below 1e-5 within 60 steps (got {damped})"
    );
    assert!(
        damped > Fix128::ZERO,
        "and it must decay rather than be annihilated (got {damped})"
    );
    assert!(
        undamped > Fix128::from_ratio(1, 100),
        "the loss-free control must still be ringing (got {undamped})"
    );
    println!("uniform loss: damped = {damped}, control = {undamped}");
}

/// The continuum reflection is the closed form, checked against a hand sum.
///
/// For `depth = 4` and a cubic grading the midpoint samples are `t = 1/8, 3/8,
/// 5/8, 7/8`, so `Σt³ = (1 + 27 + 125 + 343)/512 = 31/32`. With `σ_max = 4`
/// that integral is `31/8`, and the round trip doubles the exponent:
/// `R = exp(−31/4)`. Nothing here calls the implementation to get its own
/// expectation.
#[test]
fn theoretical_reflection_is_the_closed_form() {
    let r = theoretical_pml_reflection(4, Fix128::from_int(4));
    assert_eq!(
        r,
        Fix128::from_ratio(-31, 4).exp(),
        "R = exp(-2 * sigma_max * sum t^3) = exp(-31/4)"
    );
    // No layer reflects everything.
    assert_eq!(
        theoretical_pml_reflection(0, Fix128::from_int(4)),
        Fix128::ONE
    );
    // Monotone in both knobs: more conductivity and more depth both absorb more.
    let a = theoretical_pml_reflection(4, Fix128::from_int(2));
    let b = theoretical_pml_reflection(4, Fix128::from_int(4));
    let c = theoretical_pml_reflection(8, Fix128::from_int(2));
    assert!(b < a, "raising sigma_max must lower R ({b} vs {a})");
    assert!(c < a, "deepening the layer must lower R ({c} vs {a})");
    println!("theoretical R: d4s2 = {a}, d4s4 = {b}, d8s2 = {c}");
}

/// A PML leaves far less field behind than a PEC box, and more conductivity
/// leaves less.
///
/// ⚠️ **What is measured here is the residual field envelope, not a reflection
/// coefficient.** `R = |reflected| / |incident|` needs the two waves separated;
/// the number below is what is still ringing after 200 steps, which mixes
/// several round trips, numerical dispersion and energy that has not reached a
/// wall yet. [`theoretical_pml_reflection`] is deliberately **not** compared
/// against it — relating two different quantities with an inequality would read
/// like a checked result to the next person. That helper has its own closed-form
/// test above.
///
/// The three assertions and where each comes from:
///
/// * **closed form** — a PEC box conserves energy, so the control cannot decay;
///   it is asserted to still be ringing at order 10⁻².
/// * **measured** (2026-09-30) — the PML envelope is 18× below the PEC control
///   at `σ_max = 4` (0.0120 against 0.2148). The declared factor is 10, leaving
///   about 1.8× of margin, so the bound is not a restatement of the reading.
/// * **measured** (2026-09-30) — the envelope falls monotonically with `σ_max`
///   over this range: 2.77e-2 at `σ_max = 1` and 1.20e-2 at 4. ⚠️ Monotone over
///   *this* range only; a large enough `σ_max` makes the front face of the layer
///   a worse discontinuity and the residual rises again.
///
/// ⚠️ The early-step assertion is what stops "it absorbs" from being satisfiable
/// by a solver that destroys the field: the wave has to still be there at step
/// 25 (**measured** 4.7e-2 at `σ_max = 4`) before it is allowed to be gone at
/// step 200.
#[test]
fn a_pml_leaves_far_less_field_than_a_pec_box() {
    // ⚠️ Anisotropic on purpose: a cubic lattice with one depth gives all three
    // axes the same conductivity profile, and then a half field wired to the
    // wrong axis is invisible. Measured 2026-09-30 — on 16³ with depth 4,
    // reading `Exy`'s coefficient from the `z` profile survived every oracle.
    let (nx, ny, nz) = (20usize, 16usize, 12usize);
    let depth = [5usize, 3, 2];
    let steps = 200usize;

    // Returns (field at step 25, envelope over the last 20 steps).
    let run = |absorber: Absorber| -> (Fix128, Fix128) {
        let mut grid = YeeGrid::new_with_absorber(nx, ny, nz, COURANT_3D, absorber);
        seed_divergence_free_loop(&mut grid, nx / 2 - 1, ny / 2 - 1, nz / 2);
        // ⚠️ The seed has to be divergence free or this measures a static
        // longitudinal field that no absorber may remove. See the comment on
        // `seed_divergence_free_loop`.
        assert_eq!(
            grid.max_abs_gauss_residual(),
            Fix128::ZERO,
            "the seed must satisfy Gauss's law, or this measures a static field"
        );
        let mut early = Fix128::ZERO;
        let mut envelope = Fix128::ZERO;
        for step in 1..=steps {
            grid.step();
            if step == 25 {
                early = max_abs_field_lossless(&grid);
            }
            if step > steps - 20 {
                let m = max_abs_field_lossless(&grid);
                if m > envelope {
                    envelope = m;
                }
            }
        }
        (early, envelope)
    };

    let (_, pec) = run(Absorber::None);
    let (early1, s1) = run(Absorber::GradedPml {
        depth,
        sigma_max: Fix128::from_int(1),
    });
    let (early4, s4) = run(Absorber::GradedPml {
        depth,
        sigma_max: Fix128::from_int(4),
    });
    println!(
        "late envelope: pec = {pec}, pml sigma_max 1 = {s1}, 4 = {s4} (step 25: {early1} / {early4})"
    );

    assert!(
        pec > Fix128::from_ratio(1, 100),
        "a PEC box conserves energy, so the control must keep ringing (got {pec})"
    );
    assert!(
        early4 > Fix128::from_ratio(1, 100),
        "the wave must still be present at step 25, or the layer is destroying the field rather than absorbing it (got {early4})"
    );
    assert!(
        s4 * Fix128::from_int(10) < pec,
        "the PML must leave at least 10x less than PEC ({s4} vs {pec})"
    );
    assert!(s4 < s1, "raising sigma_max must absorb more ({s4} vs {s1})");
    assert!(
        s4 > Fix128::ZERO,
        "and it must not have annihilated the field outright"
    );
}

/// `∇·B = 0` still holds — on the part of the lattice where it is a law.
///
/// ⚠️ Restricted to cells whose six faces are all on the lossless update, and
/// that restriction is the point. A split half field is book-keeping, not a
/// component of `B`: inside the layer the two halves of each face decay with
/// different coefficients, so the face sum does not telescope and `∇·B` is
/// **not** expected to vanish there. Asserting over the whole lattice would
/// look like a solver bug and send someone chasing it.
///
/// The assertion below also shows the restriction is load-bearing rather than
/// decorative: **measured** 2026-09-30, after 200 steps the lossless cells sit at
/// 31 ULP while the whole-lattice maximum is 8e-4, which is about 1.5e16 ULP.
#[test]
fn div_b_stays_bounded_outside_the_layer() {
    let (nx, ny, nz) = (20usize, 16usize, 12usize);
    let mut grid = YeeGrid::new_with_absorber(
        nx,
        ny,
        nz,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [5, 3, 2],
            sigma_max: Fix128::from_int(4),
        },
    );
    seed_divergence_free_loop(&mut grid, nx / 2 - 1, ny / 2 - 1, nz / 2);
    assert_eq!(
        grid.max_abs_div_b(),
        Fix128::ZERO,
        "H starts at zero, so div B starts at exactly zero everywhere"
    );
    let steps = 200usize;
    for step in 1..=steps {
        grid.step();
        let d = max_abs_div_b_lossless(&grid);
        let budget = 2 * step as i128 + 16;
        assert!(
            raw(d) <= budget,
            "step {step}: lossless max |div B| = {} ULP exceeds budget {budget} ULP",
            raw(d)
        );
    }
    let inside = grid.max_abs_div_b();
    let outside = max_abs_div_b_lossless(&grid);
    println!(
        "after {steps} steps: lossless max |div B| = {} ULP, whole lattice = {}",
        raw(outside),
        inside
    );
    assert!(
        inside > outside,
        "if the layer kept div B as small as the interior the restriction would be decoration"
    );
}

/// The loss-free core is `∏(n − 2·depth[axis])` cells, exactly.
///
/// ⚠️ And `Ex` inside an `x`-normal slab is **not** absorbing: the layer
/// attenuates through `σy` and `σz` for that component, and a slab normal to
/// `x` has neither. Getting this backwards would put loss on the field
/// component a PML is supposed to leave alone.
#[test]
fn the_absorbing_region_has_the_shape_the_doc_claims() {
    let n = 12usize;
    let depth = [3usize, 2, 4];
    let grid = YeeGrid::new_with_absorber(
        n,
        n,
        n,
        COURANT_3D,
        Absorber::GradedPml {
            depth,
            sigma_max: Fix128::from_int(4),
        },
    );
    let mid = n / 2;
    assert!(
        !grid.is_absorbing(Component::Ex, 0, mid, mid),
        "Ex is the normal component of an x slab, so the layer must not damp it"
    );
    assert!(
        grid.is_absorbing(Component::Ey, 0, mid, mid),
        "Ey is tangential to an x slab, so sigma_x must damp it"
    );
    assert!(
        !grid.is_absorbing(Component::Ez, mid, mid, mid),
        "the centre of the lattice is loss free"
    );
    let mut lossless = 0usize;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if cell_is_lossless(&grid, i, j, k) {
                    lossless += 1;
                }
            }
        }
    }
    let want = (n - 2 * depth[0]) * (n - 2 * depth[1]) * (n - 2 * depth[2]);
    assert_eq!(
        lossless, want,
        "the loss-free core must be the product of (n - 2*depth) per axis"
    );

    // A lattice without an absorber has no absorbing sample at all.
    let plain = YeeGrid::new(4, 4, 4, COURANT_3D);
    for c in EVERY {
        let (ni, nj, nk) = plain.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert!(!plain.is_absorbing(c, i, j, k), "{c:?}[{i}][{j}][{k}]");
                }
            }
        }
    }
}

/// `Absorber::None` and a zero-depth PML must be **bit-identical** to
/// [`YeeGrid::new`], for every component and every step.
///
/// ⚠️ This is condition one of the split-field design: the twelve half fields
/// change the arithmetic wherever they are used, because `S·a + S·b` and
/// `S·(a + b)` differ by 1 ULP in 47.0% of random pairs (measured 2026-09-30).
/// The exact cavity oracles above therefore only stay binding if a lattice
/// without an absorber never enters the split arm.
#[test]
fn no_absorber_is_bit_identical_to_a_plain_lattice() {
    for o in Orientation::ALL {
        let reference = tm_cavity(o, &M4_2, &M4_2, COURANT_3D);
        let (nx, ny, nz) = o.dims(M4_2.vector.len() - 1, M4_2.vector.len() - 1);
        for absorber in [
            Absorber::None,
            Absorber::GradedPml {
                depth: [0, 0, 0],
                sigma_max: Fix128::from_int(4),
            },
            Absorber::Uniform {
                sigma: Fix128::ZERO,
            },
        ] {
            let mut other = YeeGrid::new_with_absorber(nx, ny, nz, COURANT_3D, absorber);
            for a in 0..=(M4_2.vector.len() - 1) {
                for b in 0..=(M4_2.vector.len() - 1) {
                    let v = Fix128::from_int(M4_2.vector[a] * M4_2.vector[b]);
                    let (i, j, k) = o.at(a, b);
                    other.set(o.driven(), i, j, k, v);
                }
            }
            let mut plain = reference.clone();
            for n in 0..120 {
                plain.step();
                other.step();
                for c in EVERY {
                    let (ni, nj, nk) = plain.component_dims(c);
                    for i in 0..ni {
                        for j in 0..nj {
                            for k in 0..nk {
                                assert_eq!(
                                    plain.get(c, i, j, k),
                                    other.get(c, i, j, k),
                                    "{o:?} {absorber:?} step {n}: {c:?}[{i}][{j}][{k}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `set` on a split sample round-trips, is idempotent, and `set(ZERO)` empties.
///
/// The convention is stated in [`YeeGrid::set`]: the whole value goes into the
/// half the update writes first and the other is zeroed. ⚠️ Only the sum is
/// physical, so *some* convention has to be chosen; these three properties are
/// what the chosen one buys, and they are what a future change to it would have
/// to keep.
#[test]
fn set_on_a_split_sample_round_trips_and_is_idempotent() {
    let make = || {
        YeeGrid::new_with_absorber(
            8,
            8,
            8,
            COURANT_3D,
            Absorber::Uniform {
                sigma: Fix128::from_int(2),
            },
        )
    };
    let v = Fix128::from_ratio(3, 7);
    let mut grid = make();
    assert!(grid.is_absorbing(Component::Ez, 4, 4, 4));

    grid.set(Component::Ez, 4, 4, 4, v);
    assert_eq!(grid.get(Component::Ez, 4, 4, 4), v, "set then get");

    // Idempotent: writing the same value twice must not accumulate in a half.
    let mut twice = make();
    twice.set(Component::Ez, 4, 4, 4, v);
    twice.set(Component::Ez, 4, 4, 4, v);
    for _ in 0..8 {
        grid.step();
        twice.step();
    }
    for c in EVERY {
        let (ni, nj, nk) = grid.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    assert_eq!(
                        grid.get(c, i, j, k),
                        twice.get(c, i, j, k),
                        "setting twice must equal setting once: {c:?}[{i}][{j}][{k}]"
                    );
                }
            }
        }
    }
    assert!(
        grid.max_abs_field() > Fix128::ZERO,
        "the value must actually have driven something"
    );

    // set(ZERO) has to empty both halves, not just the visible sum: a lattice
    // seeded and then cleared must step identically to one never seeded.
    let mut cleared = make();
    cleared.set(Component::Ez, 4, 4, 4, v);
    cleared.set(Component::Ez, 4, 4, 4, Fix128::ZERO);
    let mut untouched = make();
    for n in 0..8 {
        cleared.step();
        untouched.step();
        for c in EVERY {
            let (ni, nj, nk) = cleared.component_dims(c);
            for i in 0..ni {
                for j in 0..nj {
                    for k in 0..nk {
                        assert_eq!(
                            cleared.get(c, i, j, k),
                            untouched.get(c, i, j, k),
                            "step {n}: set(ZERO) must empty both halves: {c:?}[{i}][{j}][{k}]"
                        );
                    }
                }
            }
        }
    }
}

/// A current inside a split sample is rejected: the update would discard it.
#[test]
#[should_panic(expected = "split sample")]
fn a_current_inside_the_absorber_is_rejected() {
    let mut grid = YeeGrid::new_with_absorber(
        8,
        8,
        8,
        COURANT_3D,
        Absorber::Uniform {
            sigma: Fix128::from_int(2),
        },
    );
    grid.set_current(Component::Ez, 4, 4, 4, Fix128::ONE);
}

/// A PML layer thicker than half its own axis is rejected rather than folded.
#[test]
#[should_panic(expected = "fit twice into its own axis")]
fn an_oversized_pml_layer_is_rejected() {
    // Fits in x and y but not in z: the guard is per axis, not one minimum.
    let _ = YeeGrid::new_with_absorber(
        8,
        8,
        6,
        COURANT_3D,
        Absorber::GradedPml {
            depth: [4, 4, 4],
            sigma_max: Fix128::from_int(4),
        },
    );
}

// ---------------------------------------------------------------------------
// O7 — the split wiring itself: which axis damps which half, and `set`
// ---------------------------------------------------------------------------

/// A discrete gradient field, whose curl is **exactly** zero on a Yee lattice.
///
/// `E = ∇φ` on node scalars telescopes: `(∇×∇φ) = 0` term by term, and
/// [`Fix128`] addition does not round, so the discrete curl is the zero bit
/// pattern rather than something small. That is what makes the first step of
/// such a state a pure decay — `H` stays exactly zero, so the `E` update sees
/// `∇×H = 0` and each sample is multiplied by its own coefficient and nothing
/// else.
fn seed_gradient(grid: &mut YeeGrid) {
    let phi = |i: usize, j: usize, k: usize| -> i64 {
        let (i, j, k) = (i as i64, j as i64, k as i64);
        i * i + 3 * j * j + 5 * k + 7 * i * j - 2 * j * k
    };
    for (c, (di, dj, dk)) in [
        (Component::Ex, (1usize, 0usize, 0usize)),
        (Component::Ey, (0, 1, 0)),
        (Component::Ez, (0, 0, 1)),
    ] {
        let (ni, nj, nk) = grid.component_dims(c);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let d = phi(i + di, j + dj, k + dk) - phi(i, j, k);
                    grid.set(c, i, j, k, Fix128::from_int(d));
                }
            }
        }
    }
}

/// The cyclic relabelling `x → y → z → x` maps each component to the next.
const fn cyclic(c: Component) -> Component {
    match c {
        Component::Ex => Component::Ey,
        Component::Ey => Component::Ez,
        Component::Ez => Component::Ex,
        Component::Hx => Component::Hy,
        Component::Hy => Component::Hz,
        Component::Hz => Component::Hx,
    }
}

/// Relabelling a lattice's axes relabels its solution, **bit for bit**.
///
/// Maxwell's equations and the Yee lattice are invariant under the cyclic
/// permutation `x → y → z → x`: a sample at `(x, y, z)` moves to `(z, x, y)`,
/// `Ex` becomes `Ey`, `Hz` becomes `Hx`, and an absorbing layer of depth
/// `[dx, dy, dz]` becomes one of depth `[dz, dx, dy]`. So a lattice and its
/// relabelled twin must step to relabelled-identical states forever, with no
/// tolerance: this is the same invariance the three `Orientation` cases above
/// rely on, applied to the absorber instead of the cavity.
///
/// ⚠️ **This is the oracle that catches a half field wired to the wrong axis,
/// and nothing else in this file does.** Each split half is damped by the
/// conductivity of one named axis, and every scalar measurement — absorption
/// against PEC, monotonicity in `σ_max`, `∇·B` outside the layer — still passes
/// when two of those names are swapped, because a wrongly damped field is still
/// a damped field. Measured 2026-09-30: reading `Exy`'s coefficient from the
/// `z` profile instead of the `y` profile survived **every** other test here,
/// including on a deliberately anisotropic layer. Under relabelling it fails
/// immediately, because the mutation moves with the component name while the
/// physics moves with the axis.
///
/// The layer is anisotropic on purpose — with `[d, d, d]` the permutation maps
/// the configuration to itself and the oracle would hold whatever the wiring is.
#[test]
fn relabelling_the_axes_relabels_the_solution_exactly() {
    let (a, b, c) = (4usize, 6usize, 5usize);
    let s = Fix128::from_ratio(1, 2);
    let sigma_max = Fix128::from_int(96);
    let mut grid = YeeGrid::new_with_absorber(
        a,
        b,
        c,
        s,
        Absorber::GradedPml {
            depth: [1, 2, 0],
            sigma_max,
        },
    );
    let mut twin = YeeGrid::new_with_absorber(
        c,
        a,
        b,
        s,
        Absorber::GradedPml {
            depth: [0, 1, 2],
            sigma_max,
        },
    );

    // An arbitrary but reproducible state in all six components, copied into the
    // twin through the relabelling so the mapping is used rather than assumed.
    let mut seeded = 0usize;
    for comp in EVERY {
        let (ni, nj, nk) = grid.component_dims(comp);
        for i in 0..ni {
            for j in 0..nj {
                for k in 0..nk {
                    let t = (i * 37 + j * 17 + k * 7 + comp as usize * 5) % 11;
                    let v = Fix128::from_int(t as i64 - 5);
                    grid.set(comp, i, j, k, v);
                    twin.set(cyclic(comp), k, i, j, v);
                    if !v.is_zero() {
                        seeded += 1;
                    }
                }
            }
        }
    }
    // 957 samples in total across the six components, so most of them must be
    // non-zero for the relabelling to be exercised rather than trivially held.
    assert!(
        seeded > 700,
        "the seed must actually fill the lattice (got {seeded})"
    );

    for n in 0..40 {
        grid.step();
        twin.step();
        for comp in EVERY {
            let (ni, nj, nk) = grid.component_dims(comp);
            for i in 0..ni {
                for j in 0..nj {
                    for k in 0..nk {
                        assert_eq!(
                            grid.get(comp, i, j, k),
                            twin.get(cyclic(comp), k, i, j),
                            "step {n}: {comp:?}[{i}][{j}][{k}] broke the cyclic relabelling"
                        );
                    }
                }
            }
        }
    }
    assert!(
        grid.max_abs_field() > Fix128::ZERO,
        "both lattices must still hold a field"
    );
}

/// `set` must define a sample's whole state, half fields included.
///
/// A lattice driven for a while and then written back to a known state must step
/// on identically to one that was in that state from the start. ⚠️ The visible
/// value would agree either way — [`YeeGrid::get`] returns the primary field —
/// so what this catches is a `set` that updates the primary and the first half
/// but leaves the second half holding history. That mutation survives every
/// other test in this file (measured 2026-09-30), because on a freshly built
/// lattice the second half is already zero.
#[test]
fn set_defines_the_whole_state_of_a_split_sample() {
    let make = || {
        YeeGrid::new_with_absorber(
            4,
            6,
            4,
            Fix128::from_ratio(1, 2),
            Absorber::GradedPml {
                depth: [0, 2, 0],
                sigma_max: Fix128::from_int(96),
            },
        )
    };

    // Give one lattice a history, so its half fields hold something.
    let mut driven = make();
    seed_gradient(&mut driven);
    for _ in 0..4 {
        driven.step();
    }
    assert!(
        driven.max_abs_field() > Fix128::ZERO,
        "the history must have left a field behind"
    );

    // Write both lattices to the same state through `set` alone.
    let mut fresh = make();
    for grid in [&mut driven, &mut fresh] {
        for c in [Component::Hx, Component::Hy, Component::Hz] {
            let (ni, nj, nk) = grid.component_dims(c);
            for i in 0..ni {
                for j in 0..nj {
                    for k in 0..nk {
                        grid.set(c, i, j, k, Fix128::ZERO);
                    }
                }
            }
        }
        seed_gradient(grid);
    }

    for n in 0..6 {
        driven.step();
        fresh.step();
        for c in EVERY {
            let (ni, nj, nk) = driven.component_dims(c);
            for i in 0..ni {
                for j in 0..nj {
                    for k in 0..nk {
                        assert_eq!(
                            driven.get(c, i, j, k),
                            fresh.get(c, i, j, k),
                            "step {n}: set must erase the history in {c:?}[{i}][{j}][{k}]"
                        );
                    }
                }
            }
        }
    }
}

/// Bit-exactness pin for the loss-free arm's **arithmetic form**.
///
/// ⚠️ This is a determinism guard, not a correctness oracle: the expected value
/// is a recorded bit pattern, so it says "this did not change", never "this is
/// right". Correctness is the business of the closed-form oracles above. It
/// exists because the loss-free arm's exact form is otherwise unobservable —
/// `f + S·(a − b)`, `f + S·a − S·b` and `f − S·(b − a)` all agree wherever
/// `S · x` happens to be exact, which is every configuration the exact cavity
/// oracles use. Measured 2026-09-30: rewriting the arm as `+ s * a - s * b`
/// left **every other test in this file green**.
///
/// The configuration is chosen so truncation actually happens: `S = 9/16` and a
/// seed of `3/7`, neither of which is dyadic.
#[test]
fn the_loss_free_arithmetic_form_is_pinned() {
    let mut grid = YeeGrid::new(6, 6, 6, COURANT_3D);
    grid.set(Component::Ez, 2, 3, 1, Fix128::from_ratio(3, 7));
    grid.set(Component::Ex, 1, 2, 4, Fix128::from_ratio(-5, 11));
    for _ in 0..40 {
        grid.step();
    }
    let probe = raw(grid.get(Component::Ez, 3, 3, 2));
    assert_eq!(
        probe, PINNED_LOSS_FREE_PROBE,
        "the loss-free update's bit pattern changed; if that was intended, \
         re-record it and say why in the commit message"
    );
}

/// Recorded 2026-09-30 from the configuration in
/// [`the_loss_free_arithmetic_form_is_pinned`], as a raw Q64.64 bit pattern.
const PINNED_LOSS_FREE_PROBE: i128 = -222_621_341_326_374_155;
