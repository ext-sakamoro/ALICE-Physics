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

// ---------------------------------------------------------------------------
// O7 — 反射係数 `R` そのもの (離散伝達行列の閉形式 vs 定常場からの mode 分離)
// ---------------------------------------------------------------------------
//
// O6 が測っているのは **N step 後の残留場 envelope** で、`R = |反射| / |入射|`
// とは別の量です (複数回の往復 + 数値分散 + まだ壁に届いていない分の混合)
// ここでは単一周波数の定常状態を作り、`R` を **2 経路**で出して突き合わせます:
//
//   (A) 離散 Yee + split-field の更新式から導いた **2×2 伝達行列の閉形式**
//   (B) solver の定常場から 6 点 DFT で複素振幅を取り、`A j^m` / `B j^{−m}` に分離
//
// ⚠️ (A) は実装を 1 つも呼びません `loss_coefficients` / `cubic_graded_sigma` /
// `theoretical_pml_reflection` は全部「実装」なので、係数も σ プロファイルも
// ここで再導出します (`loss_coefficients_are_the_closed_form` が別途 契約 test)
//
// # 1-D 化 — なぜ全部が有理数 + √3 に落ちるのか
//
// `nx × 2 × 1` の格子で `Ez` だけを励起します `nz = 1` なので `Ex` / `Ey` の
// 更新 loop (`for k in 1..nz`) が空集合で `Ex = Ey = Hz ≡ 0`、`ny = 2` なので
// `Ez` は `j = 1` だけ ⇒ 残る自由度は x 方向の 1-D 鎖です:
//
// ```text
// Hx[i][0] ← Hx[i][0] − S·Ez_i          Hx[i][1] ← Hx[i][1] + S·Ez_i
//   ⇒ 0 から始めれば恒に Hx[i][1] = −Hx[i][0] ≡ −X_i   (y 方向の遮断項)
// Hy_{i+½}  ← Hy_{i+½} + S·(Ez_{i+1} − Ez_i)
// Ez_i      ← Ez_i + S·((Hy_i − Hy_{i−1}) + 2 X_i)
// ```
//
// `H` を消去すると `Ez^{n+1} − 2Ez^n + Ez^{n−1} = S²(δ²Ez − λ_y Ez)`、`λ_y = 2`
// なので離散分散関係は
//
// ```text
// 4 sin²(ωS/2) = S²·(4 sin²(kx/2) + λ_y)
// ```
//
// `S = 1/2` と **周期 6** (`ωS = π/3`) を入れると `4·(1/4) = (1/4)(4 sin²(kx/2) + 2)`
// ⇒ `sin²(kx/2) = 1/2` ⇒ **`kx = π/2`** (空間周期 4 cell、`e^{ikx} = i`)
// 時間側は `z = e^{iπ/3} = 1/2 + i√3/2` ⇒ 全部 `Q(√3)[i]` の中で閉じます
// (超越関数は `√3` の `sqrt` 1 回だけ、`clippy.toml` の `f64::sin` 禁止と両立)
//
// # (A) の導出 — 4 成分が 2×2 に落ちる
//
// 時間調和 `X^{n+1} = z X^n` に直し、`H` の半 step 位相を振幅に吸収します
// (`Hy^{n+½} = Ĥ'_{i+½} z^n` と置く = `Ĥ' = Ĥ z^{1/2}`、これで √z が出てこない)
// x のみの PML なので `σy = σz = 0` ⇒ `Ezy` と `Hx` は **損失なし + 空間結合なし**:
//
// ```text
// X̂'_i (1 − 1/z) = −S Ê_i                 Êy_i (z − 1) = 2S X̂'_i
//   ⇒ Êy_i = −λ_y S²·z/(z−1)² · Ê_i,     μ ≡ Êx_i/Ê_i = 1 + λ_y S²·z/(z−1)²
// ```
//
// `(z−1)²/z = z − 2 + 1/z = 2cos θ − 2 = −4 sin²(θ/2)` なので `θ = π/3` では
// `z/(z−1)² = −1` ⇒ **`μ = 1 − 2S² = 1/2`** (有理数、ここで √3 が消える)
//
// 残る 2 成分 (`Êx`, `Ĥ'`) に `ca = (1−a)/(1+a)`, `cb = S/(1+a)`, `a = σS/2` を入れると
// **`(1+a)` が約分で消えます**:
//
// ```text
// w(a) ≡ (z−1) + a(z+1) = (3a−1)/2 + i(√3/2)(1+a)
// F_i     = μ·(z − ca_i)/cb_i  = μ·2·w(a_i)      = w(a_i)        (μ = 1/2)
// v_{i+½} = cb_{i+½}·z/(z − ca_{i+½}) の逆数 ×2 = 2 w(a_{i+½})/z
// ```
//
// これで 1 cell 進む写像が 2×2 になります (`det = 1`、symplectic):
//
// ```text
// Ĥ'_{i+½} = Ĥ'_{i−½} + F_i Ê_i
// Ê_{i+1}  = Ê_i + v_{i+½} Ĥ'_{i+½}        T_i = [[1 + v F, v], [F, 1]]
// ```
//
// PEC (`Ê_0 = 0`) から `Ĥ'_{½} = 1` を正規化として march し、層の外側 (`i ≥ depth`)
// の隣接 2 点で mode 分離します (`j ≡ e^{ikx} = i`、`j^{-1} = −j`):
//
// ```text
// Ê_m = A j^m + B j^{−m},  Ê_{m+1} = j(A j^m − B j^{−m})
//   ⇒ P ≡ A j^m = (Ê_m − j Ê_{m+1})/2,   Q ≡ B j^{−m} = (Ê_m + j Ê_{m+1})/2
//   ⇒ |R| = |Q| / |P|
// ```
//
// `A` が `e^{+ikx}` 側 = 時間因子 `e^{+iωt}` と合わせて **−x 方向 (= 層に向かう)**
// なので入射、`B` が反射です 純進行波なら `Q = 0`、PEC 単体なら `A + B = 0` で `|R| = 1`
// が自動的に出ます (どちらも下の test が実測で確認)
//
// # (B) の測定 — ⚠️ 駐波比 (VSWR) では測れない
//
// ⚠️ `kx = π/2` では `|Ê_i|` が **2 値しか取りません** (`|A+B|` が偶 `i`、`|A−B|` が奇 `i`)
// 包絡線の周期 2 cell を 2 点でしか刻めないので `B/A` の**位相 ψ が測れず**、
//
// ```text
// (VSWR − 1)/(VSWR + 1) ≈ |R·cos ψ|      (|R| ではない)
// ```
//
// になります 実測 (2026-09-30) で真値の **0.31〜0.996 倍**、最悪 3.2 倍ずれ しかも比が
// `σ_max` に対して非単調なので係数で補正もできません
// `the_standing_wave_ratio_is_not_the_reflection_coefficient_here` が この事実を pin します
// ⚠️ 時間方向の「6 step の max」も同型で、位相次第で `cos(π/6) = 0.866` まで潰れます (13% 誤差)
//
// ⇒ 採るのは **6 点 DFT で複素振幅を取り出す**経路です 定常場は厳密に単一周波数なので
// `Ê_i ∝ Σ_{m} Ez_i^{t₀+m} z^{−m}` が厳密に直交します (窓長が 3 の倍数なら
// `Σ z̄^{2m} = 0`、`z̄²` が原始 3 乗根なので) 位相 `z^{t₀}` は全 `i` 共通なので `|B/A|` に効きません
//
// # ⚠️ source は滑らかに立ち上げないと定常に来ない
//
// 駆動は `J(n) = s(n mod 6)`, `s = [0,1,1,0,−1,−1] = (2/√3)·sin(nπ/3)` (厳密に単一周波数)
// ですが、`n = 0` から素で入れると switch-on の広帯域成分が残ります
// ⚠️ **待っても DFT 窓を伸ばしても直りません**: 通過帯域端 (`λ → 2` と `λ → 6`) の成分は
// **群速度が 0 に近く PML まで到達しない**ので、格子の中に居残ります
// 実測 (σ_max = 4、真値 4.4164e-2): settle 300 / 600 / 900 で 4.00e-2 / 3.61e-2 / **6.88e-2**
// = ±70% 振れました (100 周期 DFT でも settle 900 で 0.6% 残る)
// ⇒ 振幅を smoothstep (`3t²−2t³`) で 600 step かけて立ち上げます (注入自体がほぼ単色になる)
// 立ち上げ後の `J` は厳密に整数なので、測定窓の中の算術は ramp の影響を受けません
//
// # ⚠️ 連続体の `theoretical_pml_reflection` とは比較しません
//
// あれは `exp(−2∫σ dx)` = **連続体**の値で、ここで測るのは離散格子の `R` です 別の量なので
// 不等式で結びません (O6 の同じ裁定と同軸) 比は `println!` で出すだけで assert しません
// 実測 (depth 4): `σ_max` = 1 / 2 / 4 / 8 で 離散/連続体 = **1.4 / 2.1 / 102.5 / 4.3e5**
// ⚠️ 4 cell/波長では grading の cell 間段差そのものがインピーダンス不整合になるので、
// `σ_max` を上げると連続体の予測は指数で下がるのに離散の実測は**上がります** (最小は `σ_max ≈ 3`)
//
// # 破壊試験 (2026-09-30 実測、変異は 1 つずつ `src/maxwell_fdtd.rs` に入れて復元)
//
// 5 変異すべてで file が red、復元後は 38 test すべて green です (本 § の 7 test のうち
// `the_discrete_transfer_matrix_...` は閉形式だけを見るので src に依存せず、
// `a_pec_backed_lattice_...` は `Absorber::None` なので PML の変異に反応しません
// = 残る 5 test が変異を捕まえる側です):
//
// | 変異 (`src/maxwell_fdtd.rs`) | 本 § の red | 他 § の red |
// |---|---|---|
// | `Ezx` の損失係数を `ca_e[0][i]` → `ca_e[1][j]` (軸取り違え) | **5** | `relabelling_the_axes_…` |
// | `loss_coefficients` の `cb = S/(1+a)` → `S/(2(1+a))` (分子 2 → 1) | **4** | 契約 test / 損失漸化式 / O6 |
// | `cubic_graded_sigma` の `t*t*t` → `t` (3 乗 → 1 乗) | **5** | `theoretical_reflection_…` |
// | `sigma_at` が `depth[(axis+1)%3]` を読む (層が別の軸に付く) | **5** | **0** |
// | (盲点確認) `Exy` の損失係数を `ca_e[1][j]` → `ca_e[2][k]` | **0** | `relabelling_the_axes_…` |
//
// ⚠️ **`depth[(axis+1)%3]` を捕まえるのは本 § だけ**でした O6 の格子は `depth = [5,3,2]` で
// 3 軸とも層があるので、軸を巡回させても「どこかに層がある」状態が保たれて吸収量が変わりません
// x のみに層を張る配置が、層が**どの軸に付いているか**を初めて観測可能にしています
//
// ⚠️ `cb` 半減で `the_standing_wave_ratio_is_not_the_reflection_coefficient_here` だけが
// green のままなのは**正しい**挙動です あれは同じ場から取った 2 つの測定量の**関係**を
// 述べていて、`|R|` の絶対値には触れていません (絶対値側は他の 4 test が見る)
//
// ⚠️ **本 § の盲点**: x のみ PML + TM なので `Ex` / `Ey` / `Hz` と `Exy` / `Exz` /
// `Eyz` / `Hzx` / `Hzy` は恒等的に 0 の上しか走りません ⇒ **`Exy` 系の軸取り違えは
// ここでは red になりません** (上の表で実測、担当は
// `relabelling_the_axes_relabels_the_solution_exactly`)
// = 対称 / 退化した配置が成分を検証しない、O6 冒頭と同じ構造の盲点です

/// `Q(√3)[i]` の元 この節の閉形式が全部この中で閉じる
///
/// `Fix128` の add / sub / mul / div だけで書けるので、超越関数は `√3` を出す
/// `sqrt` 1 回に限られる
#[derive(Clone, Copy, Debug)]
struct Cx {
    re: Fix128,
    im: Fix128,
}

impl Cx {
    const fn new(re: Fix128, im: Fix128) -> Self {
        Self { re, im }
    }

    const fn zero() -> Self {
        Self::new(Fix128::ZERO, Fix128::ZERO)
    }

    const fn one() -> Self {
        Self::new(Fix128::ONE, Fix128::ZERO)
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

    fn scale(self, s: Fix128) -> Self {
        Self::new(self.re * s, self.im * s)
    }

    fn conj(self) -> Self {
        Self::new(self.re, Fix128::ZERO - self.im)
    }

    fn norm2(self) -> Fix128 {
        self.re * self.re + self.im * self.im
    }

    fn div(self, o: Self) -> Self {
        let n = o.norm2();
        let p = self.mul(o.conj());
        Self::new(p.re / n, p.im / n)
    }

    /// 虚数単位を掛ける (`kx = π/2` なので `e^{ikx} = i`、mode 分離で使う)
    fn mul_i(self) -> Self {
        Self::new(Fix128::ZERO - self.im, self.re)
    }

    fn abs(self) -> Fix128 {
        self.norm2().sqrt()
    }
}

/// `√3/2` = `z` の虚部 この節で唯一の無理数
fn half_root3() -> Fix128 {
    Fix128::from_int(3).sqrt().half()
}

/// `z = e^{iπ/3} = 1/2 + i√3/2` (周期 6 の時間因子)
fn unit_z() -> Cx {
    Cx::new(Fix128::from_ratio(1, 2), half_root3())
}

/// `z^{−m}` の厳密な表、`m = 0..6`
///
/// `z³ = −1` なので `z̄³ = −1` で、6 個全部が `±1`, `±1/2`, `±√3/2` の組み合わせに
/// なる ⚠️ 繰り返し乗算で作ると 1 ULP ずつ drift するので表で持つ
fn inv_z_powers() -> [Cx; 6] {
    let h = half_root3();
    let half = Fix128::from_ratio(1, 2);
    let nh = Fix128::ZERO - half;
    let nr = Fix128::ZERO - h;
    [
        Cx::new(Fix128::ONE, Fix128::ZERO),
        Cx::new(half, nr),
        Cx::new(nh, nr),
        Cx::new(Fix128::NEG_ONE, Fix128::ZERO),
        Cx::new(nh, h),
        Cx::new(half, h),
    ]
}

/// `w(a) = (z−1) + a(z+1) = (3a−1)/2 + i(√3/2)(1+a)`
///
/// 損失係数 `ca = (1−a)/(1+a)`, `cb = S/(1+a)` を伝達行列に入れると `(1+a)` が
/// 約分で消えて、整数座標側は `F = w(a)`、半整数座標側は `v = 2w(a)/z` になる
fn w_of(a: Fix128) -> Cx {
    Cx::new(
        (Fix128::from_int(3) * a - Fix128::ONE).half(),
        half_root3() * (Fix128::ONE + a),
    )
}

/// 左側 x 層の整数座標 `i` での σ、3 乗 grading の**再導出** (実装を呼ばない)
///
/// 層の内縁からの距離は `d − i` cell なので `t = (d−i)/d`、`σ = σ_max t³`
fn layer_sigma_at_edge(i: i64, depth: i64, sigma_max: Fix128) -> Fix128 {
    if depth == 0 || i >= depth {
        return Fix128::ZERO;
    }
    let t = Fix128::from_ratio(depth - i, depth);
    sigma_max * t * t * t
}

/// 同じく半整数座標 `i + ½` での σ 倍した距離で見るので有理数のまま
fn layer_sigma_at_face(i: i64, depth: i64, sigma_max: Fix128) -> Fix128 {
    let into2 = 2 * depth - 2 * i - 1;
    if depth == 0 || into2 <= 0 {
        return Fix128::ZERO;
    }
    let t = Fix128::from_ratio(into2, 2 * depth);
    sigma_max * t * t * t
}

/// PEC から層を抜ける閉形式 march `Ê_0 .. Ê_upto` を返す (`Ĥ'_{½} = 1` が正規化)
fn march_out_of_the_pec(depth: i64, sigma_max: Fix128, courant: Fix128, upto: usize) -> Vec<Cx> {
    assert!(upto >= 2, "need two lossless samples to split the modes");
    let inv_z = unit_z().conj(); // |z| = 1 なので 1/z = z̄
    let two = Fix128::from_int(2);
    let a_edge = |i: i64| (layer_sigma_at_edge(i, depth, sigma_max) * courant).half();
    let a_face = |i: i64| (layer_sigma_at_face(i, depth, sigma_max) * courant).half();
    let v_at = |i: i64| w_of(a_face(i)).scale(two).mul(inv_z);

    let mut e = vec![Cx::zero(); upto + 1];
    // Ê_0 = 0 (PEC)、Ĥ'_{½} は自由なので 1 に取る
    let mut h = Cx::one();
    e[1] = v_at(0).mul(h);
    for i in 1..upto {
        h = h.add(w_of(a_edge(i as i64)).mul(e[i]));
        e[i + 1] = e[i].add(v_at(i as i64).mul(h));
    }
    e
}

/// 隣接 2 点から `(|入射|, |反射|)` を出す (`kx = π/2` 前提)
fn split_modes(e_m: Cx, e_m1: Cx) -> (Fix128, Fix128) {
    let incident = e_m.sub(e_m1.mul_i());
    let reflected = e_m.add(e_m1.mul_i());
    (incident.abs(), reflected.abs())
}

/// `|R| = |反射| / |入射|` を `Ê[m]`, `Ê[m+1]` から
fn reflection_at(e: &[Cx], m: usize) -> Fix128 {
    let (incident, reflected) = split_modes(e[m], e[m + 1]);
    assert!(
        !incident.is_zero(),
        "incident amplitude vanished; the reference plane is not in the lossless region"
    );
    reflected / incident
}

/// 周期 6 の離散正弦 `= (2/√3)·sin(nπ/3)`、整数なので `S·J` が dyadic で厳密
const SRC_PERIOD_6: [i64; 6] = [0, 1, 1, 0, -1, -1];

/// smoothstep `3t² − 2t³` の立ち上げ `n ≥ ramp` では厳密に 1
fn ramp_at(n: usize, ramp: usize) -> Fix128 {
    if ramp == 0 || n >= ramp {
        return Fix128::ONE;
    }
    let t = Fix128::from_ratio(n as i64, ramp as i64);
    Fix128::from_int(3) * t * t - Fix128::from_int(2) * t * t * t
}

/// 導波路を定常まで回して `Ê_i` (`i = 0..=nx`、`j = 1`, `k = 0`) を返す
///
/// `depth = 0` は `Absorber::None` (PEC 終端の control)
fn steady_amplitudes(
    nx: usize,
    depth: usize,
    sigma_max: Fix128,
    settle: usize,
    periods: usize,
) -> Vec<Cx> {
    const RAMP: usize = 600;
    let s = Fix128::from_ratio(1, 2);
    let absorber = if depth == 0 {
        Absorber::None
    } else {
        Absorber::GradedPml {
            depth: [depth, 0, 0],
            sigma_max,
        }
    };
    let mut grid = YeeGrid::new_with_absorber(nx, 2, 1, s, absorber);
    let middle = nx / 2;
    let win = 6 * periods;
    assert!(win % 6 == 0, "the DFT window must be whole periods");
    let powers = inv_z_powers();
    let mut acc = vec![Cx::zero(); nx + 1];
    for n in 0..(settle + win) {
        let drive = ramp_at(n, RAMP) * Fix128::from_int(SRC_PERIOD_6[n % 6]);
        grid.set_current(Component::Ez, middle, 1, 0, drive);
        grid.step();
        if n >= settle {
            let p = powers[(n - settle) % 6];
            for (i, slot) in acc.iter_mut().enumerate() {
                *slot = slot.add(p.scale(grid.get(Component::Ez, i, 1, 0)));
            }
        }
    }
    // Σ Re[Ê z^{t₀+m}]·z^{−m} = (win/2)·Ê z^{t₀}   (窓長が 3 の倍数なので交差項が 0)
    let norm = Fix128::from_ratio(2, win as i64);
    acc.iter().map(|c| c.scale(norm)).collect()
}

/// 定常場の空間包絡線の `(max, min)` 無損失領域だけを見る
fn envelope_extremes(e: &[Cx], lo: usize, hi: usize) -> (Fix128, Fix128) {
    let mut worst = Fix128::ZERO;
    let mut best = Fix128::ZERO;
    for (n, i) in (lo..=hi).enumerate() {
        let a = e[i].abs();
        if n == 0 || a > worst {
            worst = a;
        }
        if n == 0 || a < best {
            best = a;
        }
    }
    (worst, best)
}

/// 測定に使う設定 深さ 4 / `σ_max = 4` が第 1 点 (O6 の理論反射 test と同じ組)
const NX: usize = 64;
const SETTLE: usize = 1200;
const PERIODS: usize = 100;
/// 反射係数の参照面は層の内縁 `i = depth` ここから外は無損失
const PROFILE_CELLS: usize = 14;

/// 相対差を `1e-9` 単位で返す
///
/// ⚠️ `Fix128` の `Display` は小数 4 桁なので、`1e-7` 級の相対差はそのまま出すと
/// `0.0000` になって証跡にならない 落ちた時に読める桁で出す
fn relative_nano(gap: Fix128, reference: Fix128) -> Fix128 {
    (gap / reference) * Fix128::from_int(1_000_000_000)
}

/// (A) の式そのものの oracle — 無損失の伝達行列の固有値が `±i` であること
///
/// ⚠️ **これを先に通します** これが green でなければ `F` / `v` の式が間違っている
/// ので、その後の突合は意味を持ちません 内容は 3 つとも閉形式で、どれも
/// `4 sin²(ωS/2) = S²λ` (= 離散分散関係) の言い換えです:
///
/// * `trace T|_{σ=0} = 2 + v(0)F(0) = 0` ⟺ 固有値が `±i` ⟺ `kx = ±π/2`
///   展開すると `2(z−1)z̄·(z−1) = −2` ⟺ **`(z−1)² = −z`** (`|z| = 1` なので)
/// * `ζ₊ = (z̄/2)(1+i)` を種にした march が `1 → i → −1 → −i → 1` と**周期 4 で
///   厳密に巡回**する (= 進行波が固有解、反射成分が出ない)
/// * 層なし (`depth = 0`) の march は `Ê_0 = 0` から `A + B = 0` になるので `|R| = 1`
///
/// **実測** (2026-09-30) と bound の出所は 2 段に分かれます:
///
/// * **1 式で書ける恒等式**は `√3` の丸め 1 回しか入らないので `≤ 4 ULP`
///   (実測: `(z−1)² + z` と trace のどちらも各成分 **2 ULP**)
/// * **march を経る量**は 1 cell あたり 1-2 ULP 積むので `≤ 64 ULP`
///   (実測: 進行波が 20 cell で worst **21 ULP**、PEC 単体の `|入射| − |反射|` が
///   11 cell で **18 ULP** ⇒ 宣言 64 は約 3 倍の余裕)
/// * 純進行波の反射成分は `|B|/|A| < 1e-9` (実測 4e-19 未満、下限は打ち切りで決まる)
#[test]
fn the_discrete_transfer_matrix_has_the_travelling_wave_as_its_eigensolution() {
    let s = Fix128::from_ratio(1, 2);
    let z = unit_z();
    let two = Fix128::from_int(2);
    // 1 式で書ける恒等式は √3 の丸め 1 回ぶん (実測 2 ULP)
    let budget: i128 = 4;
    // march を経る量は 1 cell あたり 1-2 ULP 積む (実測 21 / 18 ULP、下の 2 箇所で使う)
    let march_budget: i128 = 64;

    // (z−1)² = −z
    let zm1 = z.sub(Cx::one());
    let sq = zm1.mul(zm1);
    assert!(
        ulp_gap(sq.re, Fix128::ZERO - z.re) <= budget
            && ulp_gap(sq.im, Fix128::ZERO - z.im) <= budget,
        "(z-1)^2 must be -z (omega*S = pi/3): got {sq:?}, want {:?}",
        Cx::zero().sub(z)
    );

    // trace T|_{σ=0} = 2 + v(0)·F(0) = 0
    let v0 = w_of(Fix128::ZERO).scale(two).mul(z.conj());
    let f0 = w_of(Fix128::ZERO);
    let trace = Cx::new(two, Fix128::ZERO).add(v0.mul(f0));
    assert!(
        ulp_gap(trace.re, Fix128::ZERO) <= budget && ulp_gap(trace.im, Fix128::ZERO) <= budget,
        "the lossless transfer matrix must have trace 0 (eigenvalues ±i): {trace:?}"
    );

    // 進行波は固有解: ζ₊ = (z̄/2)(1+i) を種にすると 1 → i → −1 → −i と巡回する
    let zeta_plus = z
        .conj()
        .scale(Fix128::from_ratio(1, 2))
        .mul(Cx::new(Fix128::ONE, Fix128::ONE));
    let want = [
        Cx::one(),
        Cx::new(Fix128::ZERO, Fix128::ONE),
        Cx::new(Fix128::NEG_ONE, Fix128::ZERO),
        Cx::new(Fix128::ZERO, Fix128::NEG_ONE),
    ];
    let mut e = Cx::one();
    let mut h = zeta_plus;
    let mut worst: i128 = 0;
    let mut samples = vec![e];
    for m in 0..20usize {
        let w = want[m % 4];
        worst = worst.max(ulp_gap(e.re, w.re)).max(ulp_gap(e.im, w.im));
        h = h.add(f0.mul(e));
        e = e.add(v0.mul(h));
        samples.push(e);
    }
    assert!(
        worst <= march_budget,
        "a travelling wave must stay one over 20 cells; worst deviation {worst} ULP"
    );
    let (incident, reflected) = split_modes(samples[10], samples[11]);
    assert!(
        reflected < incident * Fix128::from_ratio(1, 1_000_000_000),
        "a pure travelling wave must split into no reflected part: {reflected} vs {incident}"
    );
    println!("travelling wave: worst {worst} ULP over 20 cells, |B|/|A| = {reflected}");

    // 層なしの PEC 終端は |R| = 1
    let bare = march_out_of_the_pec(0, Fix128::ZERO, s, 12);
    let (incident, reflected) = split_modes(bare[10], bare[11]);
    println!(
        "bare PEC: |A| = {incident}, |B| = {reflected}, gap = {} ULP",
        ulp_gap(incident, reflected)
    );
    assert!(
        ulp_gap(incident, reflected) <= march_budget,
        "a bare PEC wall must reflect everything: |A| = {incident}, |B| = {reflected}"
    );
}

/// 本体 — (A) 閉形式 と (B) 実測 が相対 `1e-3` 以内で一致する
///
/// **宣言 bound は相対 `1e-3`、`n` には依存させません** (漸化式でなく定常解の比なので
/// step 数に比例する量ではない) **実測** (2026-09-30、`nx = 64` / ramp 600 / settle 1200 /
/// 100 周期):
///
/// | 掃引 | (A) | (B) | 相対差 |
/// |---|---|---|---|
/// | `σ_max = 1`, d=4 | 1.952704454e-1 | 1.952770948e-1 | **3.41e-5** |
/// | `σ_max = 2`, d=4 | 4.459805628e-2 | 4.459804439e-2 | **2.67e-7** |
/// | `σ_max = 4`, d=4 | 4.416421203e-2 | 4.416423313e-2 | **4.78e-7** |
/// | `σ_max = 4`, d=2 | 1.485699435e-1 | 1.485673437e-1 | 1.75e-5 |
/// | `σ_max = 4`, d=6 | 6.607283725e-3 | 6.607271374e-3 | 1.87e-6 |
/// | `σ_max = 4`, d=8 | 1.838820199e-3 | 1.838823487e-3 | 1.79e-6 |
///
/// worst 3.41e-5 に対し 1e-3 は 29 倍の余裕で、bound が実測の言い換えになっていません
/// ⚠️ `nx` を上げると settle が足りなくなります (`nx = 96` で 3.7e-5、往復が 333 step)
/// `nx` は 64 に固定し、残差を `nx` 依存の bound で吸収しません
#[test]
fn the_measured_reflection_matches_the_discrete_transfer_matrix() {
    let s = Fix128::from_ratio(1, 2);
    let tol = Fix128::from_ratio(1, 1000);

    let mut cases: Vec<(usize, i64)> = vec![(4, 1), (4, 2), (4, 4)];
    cases.extend([(2usize, 4i64), (6, 4), (8, 4)]);

    for (depth, sigma) in cases {
        let sigma_max = Fix128::from_int(sigma);
        let closed = march_out_of_the_pec(depth as i64, sigma_max, s, depth + 2);
        let want = reflection_at(&closed, depth);
        let measured = steady_amplitudes(NX, depth, sigma_max, SETTLE, PERIODS);
        let got = reflection_at(&measured, depth);
        let gap = (got - want).abs();
        assert!(
            gap < want * tol,
            "d={depth} sigma_max={sigma}: closed form {want}, measured {got}, gap {gap} \
             exceeds the declared 1e-3 relative bound"
        );
        // 連続体との比は記録だけ (別の量なので assert しない、O6 の裁定と同軸)
        let continuum = theoretical_pml_reflection(depth, sigma_max);
        println!(
            "d={depth} sigma_max={sigma}: (A) {want}  (B) {got}  relative gap {} e-9  \
             continuum {continuum}  discrete/continuum {}",
            relative_nano(gap, want),
            want / continuum
        );
    }
}

/// 層の**中**の複素 profile が閉形式 march の定数倍であること
///
/// source (`i = nx/2`) より左には source が無いので、`Ê_0 = 0` を満たす 2 階漸化式の
/// 解は **1 次元** ⇒ solver の `Ê_i` は march の結果の定数倍でなければなりません
/// ⚠️ scalar な `|R|` は層全体を 1 個の数字に潰しますが、こちらは**層の中の係数を
/// 1 本ずつ pin します** (破壊試験でも先に bite する)
///
/// ⚠️ **この残差 assert が (B) を (A) の仮定から独立にしています** `A j^m` / `B j^{−m}`
/// への分離は `kx = π/2` を**前提**にしているので、実装側の分散関係が違っていても
/// ずれは `A` / `B` に吸収されて `|B/A|` が「もっともらしい値」を返しえます
/// (2 経路突合は**両側が同じだけ間違うと green になる**のが既知の失敗形)
/// profile 一致は `Ê_i` を 1 点ずつ複素数で比べるので `A` / `B` に逃げ場がなく、
/// mode 分離を経由しません ⇒ **print でなく assert、bound も宣言します**
///
/// **実測** (2026-09-30、`σ_max = 4` / d=4、`i = 4` で正規化、場の大きさは 1 前後):
/// worst **7.7e-7** (`σ_max = 1` では 3.9e-5) ⇒ 宣言 bound は `1e-5` で 13 倍の余裕
#[test]
fn the_field_inside_the_layer_is_the_transfer_matrix_solution() {
    let s = Fix128::from_ratio(1, 2);
    let depth = 4usize;
    let sigma_max = Fix128::from_int(4);
    let closed = march_out_of_the_pec(depth as i64, sigma_max, s, PROFILE_CELLS);
    let measured = steady_amplitudes(NX, depth, sigma_max, SETTLE, PERIODS);
    // 正規化は参照面で行う (march の振幅は Ĥ'_{½} = 1 という任意の規格)
    let scale = measured[depth].div(closed[depth]);
    let mut worst = Fix128::ZERO;
    for i in 0..=PROFILE_CELLS {
        let want = closed[i].mul(scale);
        let gap = measured[i].sub(want).abs();
        if gap > worst {
            worst = gap;
        }
    }
    assert!(
        worst < Fix128::from_ratio(1, 100_000),
        "the field inside the layer must be the closed-form march up to one constant; \
         worst deviation {worst}"
    );
    println!(
        "layer profile worst deviation = {} e-9 over {PROFILE_CELLS} cells",
        worst * Fix128::from_int(1_000_000_000)
    );
}

/// `σ_max` の単調性は **この範囲だけ** — 4 cell/波長では最小が `σ_max ≈ 3` で反転する
///
/// **実測** (2026-09-30、(B) 実測値、d=4):
///
/// | `σ_max` | 1 | 2 | 3 | 4 | 6 |
/// |---|---|---|---|---|---|
/// | `\|R\|` | 1.953e-1 | 4.460e-2 | 3.986e-2 | 4.416e-2 | 6.161e-2 |
///
/// ⚠️ **`2 → 4` の余裕は 0.98% しかありません** 決定論なので test は安定しますが、
/// 物理的な主張としては弱いので `σ_max = 6` での**反転も一緒に pin** します
/// (「単調に良くなる」と読まれると `σ_max` を上げる改変が入る)
///
/// 深さ側は強く単調です (**実測**、`σ_max = 4`): d = 2/4/6/8 で
/// 1.486e-1 / 4.416e-2 / 6.607e-3 / 1.839e-3 = 各段 3.3〜6.7 倍
/// ⇒ **`σ_max` を上げるのは depth を増やすのと等価ではありません**
#[test]
fn reflection_falls_with_sigma_max_here_and_rises_beyond_it() {
    let sigma_run: Vec<Fix128> = [1i64, 2, 4, 6]
        .iter()
        .map(|&sigma| {
            let r = reflection_at(
                &steady_amplitudes(NX, 4, Fix128::from_int(sigma), SETTLE, PERIODS),
                4,
            );
            println!("sigma_max {sigma}: |R| = {r}");
            r
        })
        .collect();
    // ⚠️ この 2 本は主 assert ではありません 余裕が薄いので、落ちた時に「退行」でなく
    //    「余裕不足」を先に疑えるよう margin を数値で残します
    //    実測 (2026-09-30) の余裕: 1 -> 2 は 4.4 倍、**2 -> 4 は 0.98% しかない**
    //    (4.459804439e-2 -> 4.416423313e-2) 単調性は法則ではなく この範囲の性質です
    assert!(
        sigma_run[0] > sigma_run[1] && sigma_run[1] > sigma_run[2],
        "|R| must fall over sigma_max 1 -> 2 -> 4 (margin at 2 -> 4 is only 0.98%, \
         measured 2026-09-30, so check the margin before assuming a regression): {sigma_run:?}"
    );
    println!(
        "sigma_max 2 -> 4 margin = {} (measured 0.0097 = 0.98% on 2026-09-30)",
        (sigma_run[1] - sigma_run[2]) / sigma_run[1]
    );
    // 反転の特性化 ⚠️ これがないと「単調に良くなる」と読まれて sigma_max を上げる改変が入る
    assert!(
        sigma_run[3] > sigma_run[2],
        "the trend must reverse by sigma_max 6 (the grading steps become the mismatch): \
         {} at 6 against {} at 4",
        sigma_run[3],
        sigma_run[2]
    );

    // 主 assert — 深さ側は各段が桁で効くので、物理的に頑健な主張はこちら
    let depth_run: Vec<Fix128> = [2usize, 4, 6, 8]
        .iter()
        .map(|&depth| {
            let r = reflection_at(
                &steady_amplitudes(NX, depth, Fix128::from_int(4), SETTLE, PERIODS),
                depth,
            );
            println!("depth {depth}: |R| = {r}");
            r
        })
        .collect();
    let three = Fix128::from_int(3);
    for pair in depth_run.windows(2) {
        assert!(
            pair[0] > pair[1] * three,
            "deepening the layer must cut |R| by at least 3x: {} -> {}",
            pair[0],
            pair[1]
        );
    }
}

/// control — 吸収層が無ければ `|R| = 1`、空間包絡線に節が立つ
///
/// 反証器です 場を 0 にするだけの実装や、`|R|` の抽出が壊れている実装はここで落ちます
///
/// ⚠️ **無損失 cavity は吸収が無いので過渡が永久に残ります** 測定精度は DFT 窓の長さだけで
/// 決まるので、ここだけ窓を 600 周期 (3600 step) に伸ばします **実測** (2026-09-30):
/// 100 周期では `A_min/A_max = 3.93e-3` で `< 1/1000` に届かず、600 周期で **1.60e-4**
/// (`|R|` は 0.995995 → **0.999829**) 2000 周期では 9.56e-6 まで下がります
///
/// ⚠️ 駆動する `Ez[nx/2]` は共振 mode `kx = π/2` (= `m = nx/2` の cavity mode) の
/// **節**なので共振成長は起きません (節に置いた source はその mode を駆動しない)
/// 起きていれば振幅が線形に伸びて測定できません
#[test]
fn a_pec_backed_lattice_reflects_everything() {
    let long_window = 600usize;
    let e = steady_amplitudes(NX, 0, Fix128::ZERO, SETTLE, long_window);
    let r = reflection_at(&e, 4);
    assert!(
        r > Fix128::from_ratio(999, 1000),
        "a lattice with no absorber must reflect everything: |R| = {r}"
    );
    let (worst, best) = envelope_extremes(&e, 8, NX - 8);
    assert!(
        best < worst * Fix128::from_ratio(1, 1000),
        "a full standing wave must have nodes: A_min = {best}, A_max = {worst}"
    );
    println!("control: |R| = {r}, A_min = {best}, A_max = {worst}");
}

/// ⚠️ 駐波比は反射係数ではない — この動作点で使えないことを pin する
///
/// `kx = π/2` (4 cell/波長) では `|Ê_i|` が `|A+B|` (偶 `i`) と `|A−B|` (奇 `i`) の
/// **2 値しか取りません** ⇒ 包絡線の極値がサンプル点に載らず、`B/A` の位相 ψ が測れない
/// ⇒ `(VSWR−1)/(VSWR+1) ≈ |R·cos ψ|` になります
///
/// **実測** (2026-09-30、d=4、真値は mode 分離):
///
/// | `σ_max` | 真の `\|R\|` | `(VSWR−1)/(VSWR+1)` | 比 |
/// |---|---|---|---|
/// | 2 | 4.460e-2 | 3.055e-2 | 0.685 |
/// | 3 | 3.986e-2 | 1.252e-2 | **0.314** |
/// | 4 | 4.416e-2 | 2.679e-2 | 0.607 |
/// | 16 | 1.319e-1 | 6.011e-2 | 0.456 |
///
/// 比が `σ_max` に対して非単調なので、係数で補正することもできません
/// この test があるのは、`|R|` の抽出を「包絡線の max/min を割るだけ」に**簡約する
/// 改変を止める**ためです (見た目は簡単で、数字も同じ桁で返ってくる)
#[test]
fn the_standing_wave_ratio_is_not_the_reflection_coefficient_here() {
    for sigma in [3i64, 4] {
        let e = steady_amplitudes(NX, 4, Fix128::from_int(sigma), SETTLE, PERIODS);
        let truth = reflection_at(&e, 4);
        let (worst, best) = envelope_extremes(&e, 8, NX - 8);
        // (VSWR−1)/(VSWR+1) = (A_max − A_min)/(A_max + A_min)
        let from_vswr = (worst - best) / (worst + best);
        println!(
            "sigma_max {sigma}: |R| = {truth}, from VSWR = {from_vswr}, ratio = {}",
            from_vswr / truth
        );
        assert!(
            from_vswr < truth * Fix128::from_ratio(4, 5),
            "the standing-wave ratio must under-report |R| here (it measures |R cos psi|): \
             {from_vswr} against {truth}"
        );
    }
}

/// 精度 parameter 独立性 — `|R|` が settle 数と DFT 窓長に依存しないこと
///
/// ⚠️ **ramp を入れた効果が「たまたま 600 step で合った」でないことの裏取りです**
/// ramp 無しでは settle 300 / 600 / 900 が 4.00e-2 / 3.61e-2 / 6.88e-2 と ±70% 振れます
/// (§ 冒頭) 同じ sweep を ramp 有りで回して、宣言 bound (相対 `1e-3`) の中に収まる
/// = 測定が settle / 窓長という**解像度 parameter に依存しない**ことを固定します
///
/// `σ_max = 4` / d=4 で settle `{1200, 2400}` × 窓 `{100, 200 周期}` の 4 組を回し、
/// 4 組すべてが閉形式 (4.416421203e-2) から相対 `1e-3` 以内、かつ**互いに** `1e-3` 以内
/// であることを assert します 各組の相対差は `println!` で出るので、落ちた時にどの
/// 組合せで外れたかが分かります **実測** (2026-09-30、相対差): settle 1200 で
/// 4.78e-7 (100 周期) / 2.48e-7 (200 周期)、settle 2400 で 3.82e-7 / 2.31e-7
/// ⇒ 4 組の散らばりが 2.5e-7 程度で、宣言 `1e-3` に対して 3 桁以上の余裕があります
#[test]
fn the_measurement_does_not_depend_on_the_settle_or_the_window() {
    let s = Fix128::from_ratio(1, 2);
    let depth = 4usize;
    let sigma_max = Fix128::from_int(4);
    let tol = Fix128::from_ratio(1, 1000);
    let closed = march_out_of_the_pec(depth as i64, sigma_max, s, depth + 2);
    let want = reflection_at(&closed, depth);

    let mut seen: Vec<Fix128> = Vec::new();
    for settle in [1200usize, 2400] {
        for periods in [100usize, 200] {
            let got = reflection_at(
                &steady_amplitudes(NX, depth, sigma_max, settle, periods),
                depth,
            );
            let gap = (got - want).abs();
            println!(
                "settle {settle} periods {periods}: |R| = {got}, relative gap to the closed \
                 form = {} e-9",
                relative_nano(gap, want)
            );
            assert!(
                gap < want * tol,
                "settle {settle} / {periods} periods: |R| = {got} against the closed form \
                 {want}; the measurement must not depend on either resolution knob"
            );
            seen.push(got);
        }
    }
    // 互いにも 1e-3 以内 (閉形式を経由せずに parameter 独立性そのものを言う)
    for (a, b) in seen.iter().zip(seen.iter().skip(1)) {
        assert!(
            (*a - *b).abs() < *a * tol,
            "two resolution settings disagreed: {a} against {b}"
        );
    }
}
