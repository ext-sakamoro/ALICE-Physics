//! A **non-uniform** closed-form oracle for the hyperelastic P2 / P3 tetrahedra:
//! a manufactured finite-strain solution with the body force it demands.
//!
//! # What this adds over `analytic_{quadratic,cubic}_hyperelastic.rs`
//!
//! Those files can only use a uniform `F`, because for a non-uniform one the
//! element's quadrature of the non-linear internal force is not exact, so "the
//! element reproduces this field" has no non-uniform counterpart. Their module
//! docs record the consequence: a uniform field is affine, lies in the P1 space
//! too, and **does not separate P2 or P3 from P1**. This file measures the
//! thing they cannot, the *discretisation order*, with a smooth non-polynomial
//! field `u = A·(sin πY (1+Z), sin πZ (1+X), sin πX (1+Y))` that lies in no
//! polynomial space, so the error is approximation error and not a quadrature
//! artefact.
//!
//! # Construction
//!
//! The body force is `b = −Div P(F(u))`, with `P = μF + [κJ(J−1) − μ]·F⁻ᵀ` the
//! first Piola-Kirchhoff stress of the module's Neo-Hookean law (`κ = λ`),
//! built here in `f64` from a 4th-order central difference of the exact field,
//! never from the solver's stress routine. The consistent nodal load
//! `∫ b·Nᵢ dV` is integrated with an `8³` collapsed Gauss-Legendre rule, and
//! the Lagrange basis `Nᵢ` is rebuilt from a Vandermonde inverse of the
//! element's own node positions, so **no node-ordering convention of the
//! element under test is relied on**. Interior nodes carry the load; boundary
//! nodes carry the exact displacement.
//!
//! # What is measured (release build)
//!
//! | `A` | element | `h = 1/n` | nodal rms error | slope |
//! |---|---|---|---|---|
//! | 0.02 | P2 | n = 2, 3, 4, 6 | 8.9e-5, 4.9e-5, 2.4e-5, 9.0e-6 | 1.5, 2.4, 2.45 |
//! | 0.02 | P3 | n = 2, 3, 4 | 1.6e-5, 5.1e-6, 1.8e-6 | 2.8, 3.75 |
//!
//! At the same `h = 1/3` the P3 error is **9.4×** smaller than P2's.
//!
//! ⚠️ **`P1` is deliberately absent.** `solve_corotational` with a hyperelastic
//! law is a co-rotational formulation, not the same total-Lagrangian law, so
//! its error against this manufactured solution measures a different model
//! (it came out non-monotone, `4.4e-4 → 1.1e-3 → 4.6e-4`) and says nothing
//! about element order.
//!
//! ⚠️ **The modified Newton iteration does not converge at larger amplitude, and
//! that was a defect of the iteration, not of the oracle.** At `A = 0.06` (P3)
//! and `A = 0.08` (P2, n = 3) the default solve ends `NotConverged`, bit
//! identically at 80 steps, at 400 steps and at 16 increments — the contraction
//! of the iteration is above one, and the step size and budget do not touch it.
//! The tangent has to change, and `with_consistent_tangent` now does that for
//! P2 and P3 too, as a Newton–Krylov step whose tangent action is the central
//! difference of the internal force. It does not move the fixed point:
//! `rms = 7.300e-5` at `A = 0.03` is the same to four digits with and without it.
//! Measured at `A = 0.08`, n = 3: modified `NotConverged`; Newton–Krylov P2
//! `rms = 2.0e-4`, P3 `2.1e-5` (9.4× smaller, the same ratio as at `A = 0.03`).
//!
//! # The teeth
//!
//! `a_flipped_body_force_is_far_from_the_manufactured_solution` runs the same
//! scene with `b → −b`. If the load were silently ignored or its sign were
//! not part of the answer, the error would not move.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]
// The helpers are small dense-linear-algebra loops over fixed 3x3 / 4-vertex
// shapes where index loops read as the maths; the boxed solver closure is a
// one-off in a test.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]
use alice_physics::cubic_elastic_fem::{solve_cubic_hyperelastic, CubicMesh};
use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{solve_quadratic_hyperelastic, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.45;
const SIDE: f64 = 1.0;
// Amplitude `A` of the manufactured field, set per call by [`run`].
//
// A `thread_local` and not a parameter because the field is read by the
// `f64` helpers (`exact`, the finite differences, the load integral) that
// would otherwise all take it; each test runs on its own thread, so each sees
// only the amplitude it set.
thread_local! {
    static AMPLITUDE: std::cell::Cell<f64> = const { std::cell::Cell::new(0.03) };
}

fn amp() -> f64 {
    AMPLITUDE.with(std::cell::Cell::get)
}

/// The amplitude the ordinary oracles run at: inside the range where the
/// modified Newton iteration converges.
const AMP_SMALL: f64 = 0.03;
/// Large enough that the modified iteration fails (`|∇u| ≈ 0.4`).
const AMP_LARGE: f64 = 0.08;
const JITTER: f64 = 0.25;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid")
}
fn lame_f64() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}
fn config(consistent: bool) -> CorotationalConfig {
    let lin = SolverConfig::try_new(500_000, Fix128::from_raw(0, 1 << 34))
        .unwrap()
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .unwrap();
    let (_, mu) = lame_f64();
    let config = CorotationalConfig::try_new(lin, 80, fx(1.0e-7), 4, 64)
        .unwrap()
        .with_hyperelastic(HyperelasticModel::NeoHookean { mu_mpa: fx(mu) });
    if consistent {
        config.with_consistent_tangent()
    } else {
        config
    }
}

// ---- manufactured field and the body force it demands ----
fn exact(p: [f64; 3]) -> [f64; 3] {
    let pi = std::f64::consts::PI;
    [
        amp() * (pi * p[1]).sin() * (1.0 + p[2]),
        amp() * (pi * p[2]).sin() * (1.0 + p[0]),
        amp() * (pi * p[0]).sin() * (1.0 + p[1]),
    ]
}
/// 4th-order central difference of a vector function along `axis`.
fn d4<const N: usize>(
    f: &dyn Fn([f64; 3]) -> [f64; N],
    p: [f64; 3],
    axis: usize,
    h: f64,
) -> [f64; N] {
    let at = |s: f64| {
        let mut q = p;
        q[axis] += s * h;
        f(q)
    };
    let (m2, m1, p1, p2) = (at(-2.0), at(-1.0), at(1.0), at(2.0));
    let mut out = [0.0; N];
    for k in 0..N {
        out[k] = (-p2[k] + 8.0 * p1[k] - 8.0 * m1[k] + m2[k]) / (12.0 * h);
    }
    out
}
fn grad_f(p: [f64; 3]) -> [[f64; 3]; 3] {
    // F[i][j] = delta_ij + du_i/dX_j
    let mut f = [[0.0; 3]; 3];
    for j in 0..3 {
        let d = d4(&exact, p, j, 1.0e-3);
        for i in 0..3 {
            f[i][j] = d[i] + if i == j { 1.0 } else { 0.0 };
        }
    }
    f
}
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}
fn cof3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            c[i][j] = m[i1][j1] * m[i2][j2] - m[i1][j2] * m[i2][j1];
        }
    }
    c
}
/// First Piola-Kirchhoff stress of the module's Neo-Hookean law:
/// `P = mu F + [kappa J (J-1) - mu] F^-T`, `F^-T = cof F / J`, kappa = lambda.
fn piola(p: [f64; 3]) -> [f64; 9] {
    let (lambda, mu) = lame_f64();
    let f = grad_f(p);
    let j = det3(&f);
    let c = cof3(&f);
    let s = (lambda * j * (j - 1.0) - mu) / j;
    let mut out = [0.0; 9];
    for a in 0..3 {
        for b in 0..3 {
            out[3 * a + b] = mu * f[a][b] + s * c[a][b];
        }
    }
    out
}
/// `b = -div P`.
fn body_force(p: [f64; 3]) -> [f64; 3] {
    let mut b = [0.0; 3];
    for j in 0..3 {
        let d = d4(&piola, p, j, 1.0e-2);
        for i in 0..3 {
            b[i] -= d[3 * i + j];
        }
    }
    b
}

// ---- quadrature: collapsed Gauss-Legendre on the reference tetrahedron ----
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (mut p0, mut p1) = (1.0, x);
        for k in 2..=n {
            let kf = k as f64;
            let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
            p0 = p1;
            p1 = p2;
        }
        let dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
        out.push(((x + 1.0) / 2.0, 1.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}
/// (barycentric coordinates, weight summing to 1/6) of an `n^3` collapsed rule.
fn tet_rule(n: usize) -> Vec<([f64; 4], f64)> {
    let g = gauss_legendre(n);
    let mut out = Vec::new();
    for &(u, wu) in &g {
        for &(v, wv) in &g {
            for &(w, ww) in &g {
                let l1 = u;
                let l2 = v * (1.0 - u);
                let l3 = w * (1.0 - u) * (1.0 - v);
                let l0 = 1.0 - l1 - l2 - l3;
                out.push((
                    [l0, l1, l2, l3],
                    wu * wv * ww * (1.0 - u) * (1.0 - u) * (1.0 - v),
                ));
            }
        }
    }
    out
}
fn monomials(p: [f64; 3], degree: usize) -> Vec<f64> {
    let mut m = Vec::new();
    for a in 0..=degree {
        for b in 0..=(degree - a) {
            for c in 0..=(degree - a - b) {
                m.push(p[0].powi(a as i32) * p[1].powi(b as i32) * p[2].powi(c as i32));
            }
        }
    }
    m
}
/// Inverse of a small dense matrix (Gauss-Jordan, partial pivoting).
fn invert(mut a: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for c in 0..n {
        let piv = (c..n)
            .max_by(|&x, &y| a[x][c].abs().partial_cmp(&a[y][c].abs()).unwrap())
            .unwrap();
        a.swap(c, piv);
        inv.swap(c, piv);
        let d = a[c][c];
        for j in 0..n {
            a[c][j] /= d;
            inv[c][j] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = a[r][c];
                for j in 0..n {
                    a[r][j] -= f * a[c][j];
                    inv[r][j] -= f * inv[c][j];
                }
            }
        }
    }
    inv
}
/// Consistent nodal load `∫ b N_i dV` of one element from its node positions alone:
/// the Lagrange basis is rebuilt from a Vandermonde inverse, so no ordering convention
/// of the element under test is relied on.
fn element_load(
    scale: f64,
    corners: [[f64; 3]; 4],
    nodes: &[[f64; 3]],
    degree: usize,
    rule: &[([f64; 4], f64)],
) -> Vec<[f64; 3]> {
    let vand: Vec<Vec<f64>> = nodes.iter().map(|&p| monomials(p, degree)).collect();
    let coef = invert(vand); // column i = monomial coefficients of N_i
    let d = |a: usize| {
        [
            corners[a][0] - corners[0][0],
            corners[a][1] - corners[0][1],
            corners[a][2] - corners[0][2],
        ]
    };
    let (e1, e2, e3) = (d(1), d(2), d(3));
    let det = det3(&[
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ])
    .abs();
    let mut load = vec![[0.0; 3]; nodes.len()];
    for (l, w) in rule {
        let mut x = [0.0; 3];
        for a in 0..4 {
            for k in 0..3 {
                x[k] += l[a] * corners[a][k];
            }
        }
        let mono = monomials(x, degree);
        let b = body_force(x);
        let b = [scale * b[0], scale * b[1], scale * b[2]];
        for i in 0..nodes.len() {
            let ni: f64 = (0..mono.len()).map(|m| mono[m] * coef[m][i]).sum();
            for k in 0..3 {
                load[i][k] += w * det * ni * b[k];
            }
        }
    }
    load
}
fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// Deterministic displacement in `[-1, 1]` for one lattice node and axis.
fn jitter_unit(i: usize, j: usize, k: usize, axis: usize) -> f64 {
    let mut h = 0x9E37_79B9_7F4A_7C15_u64;
    for v in [i as u64, j as u64, k as u64, axis as u64] {
        h ^= v.wrapping_add(0x9E37_79B9_7F4A_7C15);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 31;
    }
    ((h >> 32) as f64) / ((1u64 << 32) as f64) * 2.0 - 1.0
}

/// Kuhn 6-tet cube on `[0, n·h]³`, with the interior vertices displaced.
fn kuhn_cube(n: usize, h: f64, jitter: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let boundary = i == 0 || j == 0 || k == 0 || i == n || j == n || k == n;
                let mut p = [i as f64 * h, j as f64 * h, k as f64 * h];
                if !boundary && jitter != 0.0 {
                    for (axis, c) in p.iter_mut().enumerate() {
                        *c += jitter * h * jitter_unit(i, j, k, axis);
                    }
                }
                mesh.vertices.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(n, i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = node_index(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn on_bdry(p: [f64; 3]) -> bool {
    p.iter().any(|&c| c < 1e-6 || c > SIDE - 1e-6)
}
struct Level {
    h: f64,
    max_err: f64,
    rms: f64,
    secs: f64,
}
fn measure(pos: &[[f64; 3]], d: &[[Fix128; 3]]) -> (f64, f64) {
    let (mut worst, mut sq) = (0.0f64, 0.0f64);
    for (p, u) in pos.iter().zip(d) {
        let e = exact(*p);
        for a in 0..3 {
            let r = (u[a].to_f64() - e[a]).abs();
            worst = worst.max(r);
            sq += r * r;
        }
    }
    (worst, (sq / (3 * pos.len()) as f64).sqrt())
}
fn finish(pos: &[[f64; 3]], loads: Vec<[f64; 3]>) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (n, p) in pos.iter().enumerate() {
        if on_bdry(*p) {
            let u = exact(*p);
            bc.prescribe_all(n as u32, [fx(u[0]), fx(u[1]), fx(u[2])]);
        } else {
            for (k, ax) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
                if loads[n][k] != 0.0 {
                    bc.add_load(n as u32, ax, fx(loads[n][k]));
                }
            }
        }
    }
    bc
}
fn corners_of(m: &SdfTetMesh, t: &Tetrahedron) -> [[f64; 3]; 4] {
    let v = |i: u32| {
        let q = m.vertices[i as usize];
        [q[0] as f64, q[1] as f64, q[2] as f64]
    };
    [
        v(t.vertices[0]),
        v(t.vertices[1]),
        v(t.vertices[2]),
        v(t.vertices[3]),
    ]
}
/// The solve at the ordinary amplitude with the modified iteration.
fn run(order: usize, n: usize, scale: f64) -> Level {
    try_run(order, n, scale, AMP_SMALL, false).expect("converges at the ordinary amplitude")
}

fn try_run(
    order: usize,
    n: usize,
    scale: f64,
    amplitude: f64,
    consistent: bool,
) -> Result<Level, FemError> {
    AMPLITUDE.with(|a| a.set(amplitude));
    let t0 = std::time::Instant::now();
    let h = SIDE / n as f64;
    let sdf = kuhn_cube(n, h, JITTER);
    let rule = tet_rule(8);
    let (pos, bc, solve): (
        Vec<[f64; 3]>,
        BoundaryConditions,
        Box<dyn Fn(&BoundaryConditions) -> Result<Vec<[Fix128; 3]>, FemError>>,
    ) = match order {
        2 => {
            let m = QuadraticMesh::from_tet_mesh(&sdf).unwrap();
            let pos: Vec<[f64; 3]> = (0..m.node_count())
                .map(|i| {
                    let p = m.node_position(i as u32).unwrap();
                    [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
                })
                .collect();
            let mut loads = vec![[0.0; 3]; pos.len()];
            for (e, t) in sdf.tets.iter().enumerate() {
                let c = corners_of(&sdf, t);
                let ids = m.element_nodes(e).unwrap();
                let np: Vec<[f64; 3]> = ids.iter().map(|&i| pos[i as usize]).collect();
                let l = element_load(scale, c, &np, 2, &rule);
                for (s, &id) in ids.iter().enumerate() {
                    for k in 0..3 {
                        loads[id as usize][k] += l[s][k];
                    }
                }
            }
            let bc = finish(&pos, loads);
            (
                pos,
                bc,
                Box::new(move |bc| {
                    solve_quadratic_hyperelastic(&m, &material(), bc, &config(consistent))
                        .map(|s| s.field.displacements)
                }),
            )
        }
        _ => {
            let m = CubicMesh::from_tet_mesh(&sdf).unwrap();
            let pos: Vec<[f64; 3]> = (0..m.node_count())
                .map(|i| {
                    let p = m.node_position(i as u32).unwrap();
                    [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
                })
                .collect();
            let mut loads = vec![[0.0; 3]; pos.len()];
            for (e, t) in sdf.tets.iter().enumerate() {
                let c = corners_of(&sdf, t);
                let ids = m.element_nodes(e).unwrap();
                let np: Vec<[f64; 3]> = ids.iter().map(|&i| pos[i as usize]).collect();
                let l = element_load(scale, c, &np, 3, &rule);
                for (s, &id) in ids.iter().enumerate() {
                    for k in 0..3 {
                        loads[id as usize][k] += l[s][k];
                    }
                }
            }
            let bc = finish(&pos, loads);
            (
                pos,
                bc,
                Box::new(move |bc| {
                    solve_cubic_hyperelastic(&m, &material(), bc, &config(consistent))
                        .map(|s| s.field.displacements)
                }),
            )
        }
    };
    let d = solve(&bc)?;
    let (max_err, rms) = measure(&pos, &d);
    Ok(Level {
        h,
        max_err,
        rms,
        secs: t0.elapsed().as_secs_f64(),
    })
}

fn slope(coarse: &Level, fine: &Level) -> f64 {
    (coarse.rms / fine.rms).ln() / (coarse.h / fine.h).ln()
}

#[test]
fn p2_error_decreases_between_two_levels() {
    let coarse = run(2, 2, 1.0);
    let fine = run(2, 3, 1.0);
    eprintln!("[hyper-mms] P2 rms {:.3e} -> {:.3e}", coarse.rms, fine.rms);
    assert!(
        fine.rms < 0.7 * coarse.rms,
        "refining 2 -> 3 must cut the error: {:.3e} -> {:.3e}",
        coarse.rms,
        fine.rms
    );
}

#[test]
#[ignore = "runtime: about 5 s in release, about 60 s in debug (P2 n = 2, 3, 4 with a Fix128 Newton solve each); run by run_ignored.py"]
fn p2_error_decreases_with_refinement_at_better_than_second_order() {
    let levels: Vec<Level> = [2usize, 3, 4].iter().map(|&n| run(2, n, 1.0)).collect();
    for l in &levels {
        eprintln!(
            "[hyper-mms] P2 h={:.3} max={:.3e} rms={:.3e} {:.1}s",
            l.h, l.max_err, l.rms, l.secs
        );
    }
    assert!(
        levels[0].rms > levels[1].rms && levels[1].rms > levels[2].rms,
        "error must fall monotonically"
    );
    let s = slope(&levels[1], &levels[2]);
    eprintln!("[hyper-mms] P2 slope(3->4) = {s:.2}");
    assert!(
        s > 2.0,
        "P2 must converge faster than second order on the finer pair, got {s:.2}"
    );
    assert!(
        levels[2].rms < 1.0e-4,
        "absolute error at n = 4: {:.3e}",
        levels[2].rms
    );
}

#[test]
fn a_flipped_body_force_is_far_from_the_manufactured_solution() {
    let right = run(2, 3, 1.0);
    let flipped = run(2, 3, -1.0);
    eprintln!(
        "[hyper-mms] right {:.3e}  flipped {:.3e}",
        right.rms, flipped.rms
    );
    assert!(
        flipped.rms > 20.0 * right.rms,
        "the load must be part of the answer: right {:.3e}, flipped {:.3e}",
        right.rms,
        flipped.rms
    );
}

#[test]
#[ignore = "runtime: about 30 s in release (P3 n = 2 and 3 plus P2 n = 3, Fix128 Newton on 20-node elements); run by run_ignored.py"]
fn p3_separates_from_p2_in_order_and_in_error() {
    let p2 = run(2, 3, 1.0);
    let p3_coarse = run(3, 2, 1.0);
    let p3_fine = run(3, 3, 1.0);
    let s3 = slope(&p3_coarse, &p3_fine);
    eprintln!(
        "[hyper-mms] P2(h=1/3) {:.3e}  P3(h=1/2) {:.3e}  P3(h=1/3) {:.3e}  P3 slope {s3:.2}",
        p2.rms, p3_coarse.rms, p3_fine.rms
    );
    assert!(
        p3_fine.rms * 5.0 < p2.rms,
        "P3 must beat P2 at the same h by 5x: {:.3e} vs {:.3e}",
        p3_fine.rms,
        p2.rms
    );
    assert!(s3 > 2.5, "P3 slope {s3:.2}");
}

#[test]
fn the_modified_iteration_still_does_not_converge_at_large_amplitude() {
    // ⚠️ Pins the *reason* the Newton–Krylov step exists. If this starts to
    // converge the claim in the module doc is stale and the test should go.
    let out = try_run(2, 3, 1.0, AMP_LARGE, false);
    assert!(
        matches!(out, Err(FemError::NotConverged { .. })),
        "the modified iteration is expected to fail at A = {AMP_LARGE}, got {:?}",
        out.as_ref().map(|l| l.rms)
    );
}

#[test]
fn the_newton_krylov_step_converges_where_the_modified_one_does_not() {
    let level = try_run(2, 3, 1.0, AMP_LARGE, true).expect("Newton-Krylov converges");
    eprintln!(
        "[hyper-mms] P2 A={AMP_LARGE} newton-krylov rms {:.3e} {:.1}s",
        level.rms, level.secs
    );
    assert!(
        level.rms < 5.0e-4,
        "the answer must still be the manufactured solution, rms {:.3e}",
        level.rms
    );
}

#[test]
fn the_newton_krylov_step_does_not_move_the_fixed_point() {
    let modified = try_run(2, 3, 1.0, AMP_SMALL, false).expect("converges");
    let krylov = try_run(2, 3, 1.0, AMP_SMALL, true).expect("converges");
    eprintln!(
        "[hyper-mms] modified {:.6e}  newton-krylov {:.6e}",
        modified.rms, krylov.rms
    );
    assert!(
        (modified.rms - krylov.rms).abs() < 1.0e-8 * modified.rms.max(1.0e-12) + 1.0e-9,
        "same equilibrium, different tangent: {:.9e} vs {:.9e}",
        modified.rms,
        krylov.rms
    );
}

#[test]
#[ignore = "runtime: about 35 s in release (P3 n = 3 at A = 0.08 with the Newton-Krylov step, plus P2 n = 3); run by run_ignored.py"]
fn p3_separates_from_p2_at_large_amplitude() {
    let p2 = try_run(2, 3, 1.0, AMP_LARGE, true).expect("converges");
    let p3 = try_run(3, 3, 1.0, AMP_LARGE, true).expect("converges");
    eprintln!(
        "[hyper-mms] A={AMP_LARGE} P2 {:.3e} P3 {:.3e}",
        p2.rms, p3.rms
    );
    assert!(
        p3.rms * 5.0 < p2.rms,
        "P3 must beat P2 by 5x: {:.3e} vs {:.3e}",
        p3.rms,
        p2.rms
    );
}
