//! Oracles for `linear_solver` (GMRES(m), BiCGStab, Jacobi / block Jacobi,
//! block equilibration).
//!
//! Every expected value comes from outside the solver: a closed form (the
//! discrete and continuous solutions of 1-D advection–diffusion), the
//! crate's existing private conjugate gradient reached through
//! `linear_elastic_fem::solve`, or an `f64` Gaussian elimination written in
//! this file. No expected value is produced by calling the solver under test.
//!
//! Tolerances: the solvers are run to a relative residual of `2⁻⁴⁰ ≈ 9.1e-13`;
//! an error bound is `cond(A) · tol` relative, and each test states the
//! `cond` it relies on.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::coupled_iteration::{ConfigFault, L2_TERM_FLOOR};
use alice_physics::det_math::exp64;
use alice_physics::linear_elastic_fem::{
    reactions, solve, Axis, BoundaryConditions, ElasticMaterial, FemSolution, SolverConfig,
};
use alice_physics::linear_solver::{
    bicgstab, gmres, solve_equilibrated, BlockEquilibration, BlockJacobiPreconditioner, BlockScale,
    BreakdownKind, DenseMatrix, FnOperator, IdentityPreconditioner, JacobiPreconditioner,
    KrylovConfig, KrylovConfigFault, KrylovMethod, KrylovSolution, LinearOperator,
    LinearSolverError, BREAKDOWN_RELATIVE,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

// ----------------------------------------------------------------------------
// helpers (all independent of the solver under test)
// ----------------------------------------------------------------------------

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// `2⁻⁴⁰`, the relative tolerance used throughout.
fn tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24)
}

fn config(max_iterations: u32) -> KrylovConfig {
    KrylovConfig::try_new(max_iterations, tol()).expect("valid config")
}

fn dense(n: usize, f: impl Fn(usize, usize) -> f64) -> DenseMatrix {
    let mut entries = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            entries.push(fx(f(i, j)));
        }
    }
    DenseMatrix::try_new(n, entries).expect("square")
}

fn dense_exact(n: usize, f: impl Fn(usize, usize) -> Fix128) -> DenseMatrix {
    let mut entries = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            entries.push(f(i, j));
        }
    }
    DenseMatrix::try_new(n, entries).expect("square")
}

/// The matrix of an operator, column by column, in `f64`.
fn to_f64_matrix<A: LinearOperator + ?Sized>(op: &A) -> Vec<Vec<f64>> {
    let n = op.dim();
    let mut m = vec![vec![0.0; n]; n];
    let mut e = vec![Fix128::ZERO; n];
    let mut col = vec![Fix128::ZERO; n];
    for j in 0..n {
        e[j] = Fix128::ONE;
        op.apply(&e, &mut col);
        for i in 0..n {
            m[i][j] = col[i].to_f64();
        }
        e[j] = Fix128::ZERO;
    }
    m
}

/// Reference: Gaussian elimination with partial pivoting in `f64`.
fn gauss_f64(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let p = (k..n)
            .max_by(|&i, &j| a[i][k].abs().partial_cmp(&a[j][k].abs()).expect("finite"))
            .expect("non-empty");
        a.swap(k, p);
        b.swap(k, p);
        for i in k + 1..n {
            let f = a[i][k] / a[k][k];
            let pivot_row = a[k].clone();
            for (aij, akj) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                *aij -= f * akj;
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    x
}

/// `κ∞(A) = ‖A‖∞ ‖A⁻¹‖∞`, with `A⁻¹` from `f64` elimination column by column.
fn cond_inf(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let row_sum = |m: &[Vec<f64>]| {
        m.iter()
            .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = gauss_f64(a.to_vec(), e);
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    row_sum(a) * row_sum(&inv)
}

fn max_rel_err(x: &[Fix128], reference: &[f64]) -> f64 {
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(reference)
        .map(|(a, b)| (a.to_f64() - b).abs() / scale)
        .fold(0.0, f64::max)
}

fn f64s(v: &[Fix128]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

fn ok(result: Result<KrylovSolution, LinearSolverError>, what: &str) -> KrylovSolution {
    match result {
        Ok(s) => s,
        Err(e) => panic!("{what}: {e:?}"),
    }
}

/// Every entry ≤ the previous one, allowing `slack` relative rounding.
fn assert_non_increasing(history: &[Fix128], slack: f64, what: &str) {
    for w in history.windows(2) {
        let (a, b) = (w[0].to_f64(), w[1].to_f64());
        assert!(
            b <= a * (1.0 + slack),
            "{what}: residual rose {a:e} -> {b:e} in {history:?}"
        );
    }
}

// ----------------------------------------------------------------------------
// symmetric positive definite: the crate's private CG through `solve`
// ----------------------------------------------------------------------------

fn node(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    (i + (nx + 1) * (j + (ny + 1) * k)) as u32
}

/// Kuhn-split box (same construction as the FEM oracles).
fn kuhn_box(nx: usize, ny: usize, nz: usize) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    let n = |i, j, k| node(nx, ny, i, j, k);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = [
                    n(i, j, k),
                    n(i + 1, j, k),
                    n(i, j + 1, k),
                    n(i + 1, j + 1, k),
                    n(i, j, k + 1),
                    n(i + 1, j, k + 1),
                    n(i, j + 1, k + 1),
                    n(i + 1, j + 1, k + 1),
                ];
                for t in [
                    [0, 1, 3, 7],
                    [0, 1, 5, 7],
                    [0, 2, 3, 7],
                    [0, 2, 6, 7],
                    [0, 4, 5, 7],
                    [0, 4, 6, 7],
                ] {
                    mesh.tets.push(Tetrahedron {
                        vertices: [c[t[0]], c[t[1]], c[t[2]], c[t[3]]],
                    });
                }
            }
        }
    }
    mesh
}

/// The cantilever's free-block stiffness `K_ff` as an operator, applied
/// through `reactions` (the same `apply_stiffness` the private CG uses) with
/// every degree of freedom prescribed, so the returned "reaction" is `K u`.
struct FreeBlockStiffness<'a> {
    mesh: &'a SdfTetMesh,
    material: &'a ElasticMaterial,
    everything_prescribed: BoundaryConditions,
    free: Vec<usize>,
}

impl LinearOperator for FreeBlockStiffness<'_> {
    fn dim(&self) -> usize {
        self.free.len()
    }

    fn apply(&self, x: &[Fix128], y: &mut [Fix128]) {
        let vertices = self.mesh.vertices.len();
        let mut u = vec![[Fix128::ZERO; 3]; vertices];
        for (&d, &value) in self.free.iter().zip(x) {
            u[d / 3][d % 3] = value;
        }
        let state = FemSolution {
            displacements: u,
            element_stress: Vec::new(),
            iterations: 0,
            relative_residual: Fix128::ZERO,
            effective_relative_tolerance: Fix128::ZERO,
        };
        let force = reactions(
            self.mesh,
            self.material,
            &self.everything_prescribed,
            None,
            &state,
        )
        .expect("valid mesh");
        for (out, &d) in y.iter_mut().zip(&self.free) {
            *out = force[d / 3][d % 3];
        }
    }
}

/// `K_ff u_f = f_f` solved by the private CG (`solve`) and by GMRES / BiCGStab
/// on the same operator must agree, and both must match an `f64` Gaussian
/// elimination of the assembled `K_ff`.
///
/// oracle: the crate's own CG (an independent iteration) + `f64` elimination.
/// `cond(K_ff)` of this 24-unknown cantilever is about `3e2` (measured from the
/// `f64` matrix in the assertion message), so `2⁻⁴⁰` relative residual bounds
/// the relative error by about `3e-10`; the assertion uses `1e-8`.
#[test]
fn spd_cantilever_matches_private_cg_and_f64_elimination() {
    let (nx, ny, nz) = (2, 1, 1);
    let mesh = kuhn_box(nx, ny, nz);
    let material =
        ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(3, 10)).expect("material");
    let mut bc = BoundaryConditions::new();
    let mut fixed = vec![false; mesh.vertices.len()];
    for k in 0..=nz {
        for j in 0..=ny {
            let v = node(nx, ny, 0, j, k);
            bc.fix(v);
            fixed[v as usize] = true;
            bc.add_load(node(nx, ny, nx, j, k), Axis::Z, Fix128::from_int(-1));
        }
    }
    let cg_config = SolverConfig::try_new(10_000, Fix128::from_raw(0, 1 << 20)).expect("config");
    let cg = solve(&mesh, &material, &bc, &cg_config).expect("private CG converges");

    let mut everything = BoundaryConditions::new();
    for v in 0..mesh.vertices.len() as u32 {
        everything.fix(v);
    }
    let free: Vec<usize> = (0..mesh.vertices.len() * 3)
        .filter(|d| !fixed[d / 3])
        .collect();
    let op = FreeBlockStiffness {
        mesh: &mesh,
        material: &material,
        everything_prescribed: everything,
        free: free.clone(),
    };
    let mut b = vec![Fix128::ZERO; free.len()];
    for (slot, &d) in free.iter().enumerate() {
        if d % 3 == 2 && (d / 3) % (nx + 1) == nx {
            b[slot] = Fix128::from_int(-1);
        }
    }
    let k = to_f64_matrix(&op);
    let reference = gauss_f64(k.clone(), f64s(&b));
    let cg_free: Vec<Fix128> = free
        .iter()
        .map(|&d| cg.displacements[d / 3][d % 3])
        .collect();
    assert!(
        max_rel_err(&cg_free, &reference) < 1e-8,
        "private CG vs f64"
    );

    let identity = IdentityPreconditioner::new(free.len());
    let full = config(1000).with_restart(free.len()).expect("restart");
    let g = ok(gmres(&op, &identity, &b, &full), "gmres");
    let s = ok(bicgstab(&op, &identity, &b, &config(1000)), "bicgstab");
    for (name, sol) in [("gmres", &g), ("bicgstab", &s)] {
        assert!(max_rel_err(&sol.x, &reference) < 1e-8, "{name} vs f64");
        let agree = sol
            .x
            .iter()
            .zip(&cg_free)
            .map(|(a, c)| (a.to_f64() - c.to_f64()).abs())
            .fold(0.0, f64::max);
        let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(agree / scale < 1e-8, "{name} vs private CG: {agree:e}");
    }
    // Finite termination: full GMRES on an n-dimensional SPD operator needs at
    // most n Arnoldi steps.
    assert!(
        g.stats.iterations as usize <= free.len(),
        "{} > n",
        g.stats.iterations
    );
}

/// Full GMRES reaches the `2⁻⁴⁰` target within `n` steps on a dense SPD matrix
/// (Krylov finite termination), and the answer matches `f64` elimination.
///
/// `A = tridiag(-1, 4, -1) + 1/(1+|i−j|)` (n = 12), `cond ≈ 10`.
#[test]
fn gmres_terminates_within_n_on_dense_spd() {
    let n = 12;
    let a = dense(n, |i, j| {
        let t = if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        };
        t + 1.0 / (1.0 + i.abs_diff(j) as f64)
    });
    let b: Vec<Fix128> = (0..n)
        .map(|i| Fix128::from_ratio(i as i64 - 5, 7))
        .collect();
    let reference = gauss_f64(to_f64_matrix(&a), f64s(&b));
    let sol = ok(
        gmres(
            &a,
            &IdentityPreconditioner::new(n),
            &b,
            &config(1000).with_restart(n).unwrap(),
        ),
        "gmres",
    );
    assert!(
        sol.stats.iterations as usize <= n,
        "{} > {n}",
        sol.stats.iterations
    );
    assert_eq!(sol.stats.restarts, 0);
    assert!(sol.stats.residual_norm <= tol() * sol.stats.rhs_norm);
    assert!(max_rel_err(&sol.x, &reference) < 1e-10);
}

// ----------------------------------------------------------------------------
// non-symmetric: 1-D steady advection–diffusion, closed form
// ----------------------------------------------------------------------------

/// `−u'' + Pe·u' = 0` on (0,1), `u(0)=0`, `u(1)=1`, central differences on
/// `N = 32` cells, multiplied through by `h²`:
/// `(−1 − P) u_{j−1} + 2 u_j + (−1 + P) u_{j+1} = 0`, `P = Pe·h/2 = Pe/64`
/// (dyadic, so every coefficient is exact in `Fix128`).
fn advection_diffusion(pe: i64) -> (DenseMatrix, Vec<Fix128>) {
    let n = 31;
    let p = Fix128::from_ratio(pe, 64);
    let a = dense_exact(n, |i, j| {
        if i == j {
            Fix128::from_int(2)
        } else if j + 1 == i {
            Fix128::NEG_ONE - p
        } else if i + 1 == j {
            Fix128::NEG_ONE + p
        } else {
            Fix128::ZERO
        }
    });
    let mut b = vec![Fix128::ZERO; n];
    b[n - 1] = Fix128::ONE - p; // −(−1 + P)·u_N with u_N = 1
    (a, b)
}

/// Discrete closed form: `u_j = (1 − ρ^j)/(1 − ρ^N)`, `ρ = (1+P)/(1−P)`.
fn discrete_solution(pe: i64) -> Vec<f64> {
    let p = pe as f64 / 64.0;
    let rho = (1.0 + p) / (1.0 - p);
    // Integer powers by repeated multiplication (no platform libm).
    let pow = |k: i32| (0..k).fold(1.0f64, |acc, _| acc * rho);
    (1..32).map(|j| (1.0 - pow(j)) / (1.0 - pow(32))).collect()
}

/// Continuous closed form: `u(x) = (e^{Pe x} − 1)/(e^{Pe} − 1)`.
fn continuous_solution(pe: i64) -> Vec<f64> {
    let pe = pe as f64;
    (1..32)
        .map(|j| (exp64(pe * j as f64 / 32.0) - 1.0) / (exp64(pe) - 1.0))
        .collect()
}

/// GMRES (full and restarted) and BiCGStab reproduce the discrete closed form
/// for `Pe ∈ {1, 10, 40, 60}` (cell Péclet `P` up to `0.94`), and the discrete
/// solution is within `O(h²)` of the exponential.
///
/// Bound on solver error: `cond` of the `Pe = 60` matrix is about `4e3` in the
/// max norm (measured in `f64`), so `2⁻⁴⁰ · 4e3 ≈ 4e-9`; asserted `1e-7`.
/// Discretisation: `ln ρ = Pe·h + (Pe h)³/12 + …`, so the exponent at `x` is off
/// by `x·Pe³h²/12`; asserted against `2 · Pe³ h² / 12` (the factor 2 covers the
/// same shift in the denominator `ρ^N`).
#[test]
fn advection_diffusion_matches_closed_form_across_peclet() {
    for pe in [1i64, 10, 40, 60] {
        let (a, b) = advection_diffusion(pe);
        let n = b.len();
        let expected = discrete_solution(pe);
        let identity = IdentityPreconditioner::new(n);
        let jacobi = JacobiPreconditioner::from_dense(&a).expect("diag 2");
        let runs = [
            (
                "gmres full",
                gmres(&a, &identity, &b, &config(500).with_restart(n).unwrap()),
            ),
            (
                "gmres(8)",
                gmres(&a, &identity, &b, &config(5000).with_restart(8).unwrap()),
            ),
            (
                "gmres jacobi",
                gmres(&a, &jacobi, &b, &config(500).with_restart(n).unwrap()),
            ),
            ("bicgstab", bicgstab(&a, &identity, &b, &config(500))),
            ("bicgstab jacobi", bicgstab(&a, &jacobi, &b, &config(500))),
        ];
        for (name, run) in runs {
            let sol = ok(run, name);
            let err = max_rel_err(&sol.x, &expected);
            assert!(err < 1e-7, "Pe={pe} {name}: {err:e}");
            assert!(
                sol.stats.residual_norm <= tol() * sol.stats.rhs_norm,
                "Pe={pe} {name}"
            );
        }
        let h = 1.0 / 32.0;
        let bound = 2.0 * (pe as f64) * (pe as f64) * (pe as f64) * h * h / 12.0;
        let disc = expected
            .iter()
            .zip(continuous_solution(pe))
            .map(|(d, c)| (d - c).abs())
            .fold(0.0, f64::max);
        assert!(
            disc <= bound.max(1e-12),
            "Pe={pe}: discrete vs exponential {disc:e} > {bound:e}"
        );
    }
}

// ----------------------------------------------------------------------------
// GMRES restart behaviour
// ----------------------------------------------------------------------------

/// A restart length at or above the step count is full orthogonalisation: the
/// solve is bit-identical to one with a larger restart. Shorter restarts reach
/// the same answer within the stopping tolerance, and the recorded residual is
/// non-increasing (within a cycle exactly by construction; across a restart
/// the recomputed true residual may exceed the Givens estimate by rounding,
/// allowed `1e-6` relative).
#[test]
fn gmres_restart_lengths_agree_and_residual_is_monotone() {
    let (a, b) = advection_diffusion(40);
    let n = b.len();
    let expected = discrete_solution(40);
    let identity = IdentityPreconditioner::new(n);
    let full = ok(
        gmres(&a, &identity, &b, &config(5000).with_restart(n).unwrap()),
        "n",
    );
    let wider = ok(
        gmres(
            &a,
            &identity,
            &b,
            &config(5000).with_restart(4 * n).unwrap(),
        ),
        "4n",
    );
    assert_eq!(full, wider, "restart ≥ n must be full GMRES, bit for bit");
    assert_non_increasing(&full.stats.residual_history, 1e-6, "full");
    for m in [4usize, 8, 16] {
        let sol = ok(
            gmres(&a, &identity, &b, &config(20_000).with_restart(m).unwrap()),
            "m",
        );
        assert!(sol.stats.restarts > 0, "m={m} should restart");
        assert!(max_rel_err(&sol.x, &expected) < 1e-7, "m={m}");
        let diff = sol
            .x
            .iter()
            .zip(&full.x)
            .map(|(p, q)| (p.to_f64() - q.to_f64()).abs())
            .fold(0.0, f64::max);
        assert!(diff < 1e-7, "m={m} differs from full GMRES by {diff:e}");
        assert_non_increasing(&sol.stats.residual_history, 1e-6, "restarted");
    }
}

// ----------------------------------------------------------------------------
// preconditioners
// ----------------------------------------------------------------------------

/// Jacobi cuts the iteration count on a diagonally dominant matrix (n = 64)
/// whose diagonal spans `2⁰ … 2²⁰` (`cond ≈ 2²⁰` unpreconditioned, `≈ 3` after
/// Jacobi: off-diagonals are `±1/4` of the smaller diagonal of their row
/// pair). Measured counts are asserted with the reduction factor.
#[test]
fn jacobi_reduces_iterations_on_a_badly_scaled_diagonal() {
    let n = 64;
    let diag = |i: usize| f64::from(1u32 << ((i * 20) / (n - 1)));
    let a = dense(n, |i, j| {
        if i == j {
            diag(i)
        } else if i.abs_diff(j) == 1 {
            let small = diag(i).min(diag(j));
            if i < j {
                0.25 * small
            } else {
                -0.25 * small
            }
        } else {
            0.0
        }
    });
    let b: Vec<Fix128> = (0..n)
        .map(|i| Fix128::from_int(1 + (i as i64 % 3)))
        .collect();
    let reference = gauss_f64(to_f64_matrix(&a), f64s(&b));
    let identity = IdentityPreconditioner::new(n);
    let jacobi = JacobiPreconditioner::from_dense(&a).unwrap();
    let cfg = config(2000).with_restart(n).unwrap();
    let plain = ok(gmres(&a, &identity, &b, &cfg), "plain");
    let pre = ok(gmres(&a, &jacobi, &b, &cfg), "jacobi");
    assert!(max_rel_err(&pre.x, &reference) < 1e-9);
    assert!(max_rel_err(&plain.x, &reference) < 1e-6);
    // Measured: plain 64 (= n, finite termination is what saves it), Jacobi 19.
    assert!(
        pre.stats.iterations * 3 <= plain.stats.iterations,
        "jacobi {} vs plain {}",
        pre.stats.iterations,
        plain.stats.iterations
    );
    // BiCGStab has no finite termination: measured, plain stagnates at a
    // relative residual of 0.116 after 307 steps, Jacobi converges in 12.
    let pre_b = ok(bicgstab(&a, &jacobi, &b, &config(2000)), "jacobi bicgstab");
    assert!(max_rel_err(&pre_b.x, &reference) < 1e-9);
    match bicgstab(&a, &identity, &b, &config(2000)) {
        Ok(plain_b) => assert!(
            pre_b.stats.iterations * 3 <= plain_b.stats.iterations,
            "bicgstab jacobi {} vs plain {}",
            pre_b.stats.iterations,
            plain_b.stats.iterations
        ),
        // The verdict fires exactly when the window is exhausted.
        Err(LinearSolverError::Stagnated {
            without_improvement,
            ..
        }) => assert_eq!(without_improvement, KrylovConfig::DEFAULT_STAGNATION_WINDOW),
        Err(LinearSolverError::NotConverged { .. }) => {}
        Err(e) => panic!("plain bicgstab: {e:?}"),
    }
}

/// Block Jacobi is the exact inverse of a block-diagonal matrix, so GMRES and
/// BiCGStab converge in one iteration; plain GMRES needs more.
#[test]
fn block_jacobi_solves_a_block_diagonal_matrix_in_one_iteration() {
    let sizes = [3usize, 4, 2, 3];
    let n: usize = sizes.iter().sum();
    let mut block_of = Vec::new();
    for (k, &s) in sizes.iter().enumerate() {
        block_of.extend(std::iter::repeat_n(k, s));
    }
    // Non-symmetric dense blocks with a zero leading entry, so pivoting is
    // exercised.
    let a = dense(n, |i, j| {
        if block_of[i] != block_of[j] || (i == j && i == 3) {
            0.0
        } else {
            let base = (block_of[i] + 1) as f64 * if i == j { 5.0 } else { 1.0 };
            base + 0.5 * (i as f64) - 0.25 * (j as f64)
        }
    });
    let b: Vec<Fix128> = (0..n)
        .map(|i| Fix128::from_ratio(3 - i as i64, 4))
        .collect();
    let reference = gauss_f64(to_f64_matrix(&a), f64s(&b));
    let pre = BlockJacobiPreconditioner::from_dense(&a, &sizes).expect("non-singular blocks");
    let g = ok(gmres(&a, &pre, &b, &config(100)), "gmres");
    let s = ok(bicgstab(&a, &pre, &b, &config(100)), "bicgstab");
    assert_eq!(g.stats.iterations, 1);
    assert_eq!(s.stats.iterations, 1);
    assert!(max_rel_err(&g.x, &reference) < 1e-12);
    assert!(max_rel_err(&s.x, &reference) < 1e-12);
    let plain = ok(
        gmres(&a, &IdentityPreconditioner::new(n), &b, &config(100)),
        "plain",
    );
    assert!(plain.stats.iterations > 1);
}

// ----------------------------------------------------------------------------
// block equilibration (PLA thermo-elasticity block ratio)
// ----------------------------------------------------------------------------

/// A one-way thermo-elastic block system with the PLA / 1 mm block magnitudes
/// of `coupled_iteration` (`E·h = 3.5e6`, `k·h = 1.3e-4`, ratio `2.69e10`):
///
/// ```text
/// [ Eh·L   −Eh·α·I ] [u]   [b_u]
/// [  0      kh·L   ] [T] = [b_T]      L = tridiag(−1, 2, −1), α = 2⁻¹⁴
/// ```
///
/// with a known `(u*, T*)` of O(1) entries. Returns the operator, `b` and the
/// `f64` reference solution of the Fix128 system.
fn pla_system() -> (DenseMatrix, Vec<Fix128>, Vec<f64>) {
    let nb = 32;
    let eh = fx(3.5e6);
    let kh = fx(1.3e-4);
    // α = 2⁻¹⁴ = 6.1e-5 /K, the order of PLA's thermal expansion (6.8e-5 /K).
    let alpha = Fix128::from_raw(0, 1 << 50);
    let lap = |i: usize, j: usize| {
        if i == j {
            Fix128::from_int(2)
        } else if i.abs_diff(j) == 1 {
            Fix128::NEG_ONE
        } else {
            Fix128::ZERO
        }
    };
    let a = dense_exact(2 * nb, |i, j| match (i < nb, j < nb) {
        (true, true) => eh * lap(i, j),
        (true, false) if j - nb == i => -(eh * alpha),
        (false, false) => kh * lap(i - nb, j - nb),
        _ => Fix128::ZERO,
    });
    let mut x_star = Vec::new();
    for i in 0..nb {
        x_star.push(Fix128::from_ratio(i as i64 + 1, nb as i64)); // u* ∈ (0, 1]
    }
    for i in 0..nb {
        x_star.push(Fix128::from_ratio(
            ((i + 1) * (i + 1)) as i64,
            (nb * nb / 4) as i64,
        )); // T* ∈ (0, 4]
    }
    let mut b = vec![Fix128::ZERO; 2 * nb];
    a.apply(&x_star, &mut b);
    let reference = gauss_f64(to_f64_matrix(&a), f64s(&b));
    (a, b, reference)
}

/// Unequilibrated, the thermal block's products in the Krylov inner products
/// truncate: after normalising `b` (every Krylov method does) the thermal
/// entries are `< 2⁻³²` of the mechanical ones, so their squares are exactly
/// zero (asserted directly below). Measured consequence: plain GMRES reports
/// convergence to the `2⁻⁴⁰` residual while its temperature is `4.0e-7`
/// relative off, about `10⁴` times the equilibrated solve's `3.5e-11` — pinned
/// as at least `10³` times. `BlockEquilibration::unscaled` refuses the same
/// system up front. Equilibrated, both methods meet the a-priori bound
/// `√n · κ∞(D A D) · 2⁻⁴⁰` and the temperature is within `1e-9`.
#[test]
fn pla_block_ratio_needs_equilibration() {
    let (a, b, reference) = pla_system();
    let n = b.len();
    let sizes = [n / 2, n / 2];
    let magnitudes = [fx(3.5e6), fx(1.3e-4)];
    let t_err = |x: &[Fix128]| max_rel_err(&x[n / 2..], &reference[n / 2..]);

    // The truncation itself: `b / max|b|` has thermal entries whose squares
    // are zero in Fix128, while every mechanical one survives.
    let b_max = b
        .iter()
        .fold(Fix128::ZERO, |m, v| if v.abs() > m { v.abs() } else { m });
    for (i, &bi) in b.iter().enumerate() {
        let normalised = bi / b_max;
        if i >= n / 2 {
            assert!(
                normalised.abs() < L2_TERM_FLOOR,
                "T entry {i}: {normalised:?}"
            );
            assert_eq!(
                normalised * normalised,
                Fix128::ZERO,
                "T entry {i} survived"
            );
        } else {
            assert!(
                !(normalised * normalised).is_zero(),
                "u entry {i} truncated"
            );
        }
    }

    // Red side, measured: the unscaled solve reports convergence on a
    // temperature far less accurate than the equilibrated one.
    let cfg = config(400).with_restart(n).unwrap();
    let unequilibrated = ok(
        gmres(&a, &IdentityPreconditioner::new(n), &b, &cfg),
        "unscaled",
    );
    let unequilibrated_t = t_err(&unequilibrated.x);
    println!(
        "unequilibrated GMRES: Ok after {}, T error {unequilibrated_t:e}",
        unequilibrated.stats.iterations
    );
    match BlockEquilibration::unscaled(&sizes, &magnitudes) {
        Err(LinearSolverError::BlockBelowProductFloor {
            block,
            relative_magnitude,
        }) => {
            assert_eq!(block, 1);
            assert!(relative_magnitude < L2_TERM_FLOOR);
            // 1 / 2.69e10, to the precision of the f64 conversion.
            assert!((relative_magnitude.to_f64() * 2.692_307_692e10 - 1.0).abs() < 1e-6);
        }
        other => panic!("unscaled PLA must be refused, got {other:?}"),
    }

    let eq = BlockEquilibration::try_new(&sizes, &magnitudes).expect("equilibrates");
    // √3.5e6 = 1870.8 → 2¹¹; √(1/1.3e-4) = 87.7 → 2⁷.
    match (eq.scale(0), eq.scale(1)) {
        (Some(BlockScale::Down(d)), Some(BlockScale::Up(u))) => {
            assert_eq!(d.exponent(), 11);
            assert_eq!(u.exponent(), 7);
        }
        other => panic!("unexpected scales {other:?}"),
    }
    // Error bound of the equilibrated system, independent of the solver:
    // ‖Δx̃‖∞/‖x̃‖∞ ≤ √n · κ∞(D A D) · 2⁻⁴⁰ (the √n converts the 2-norm
    // residual the solver controls to the max norm).
    let d: Vec<f64> = (0..n)
        .map(|i| if i < n / 2 { 1.0 / 2048.0 } else { 128.0 })
        .collect();
    let mut scaled = to_f64_matrix(&a);
    for (i, row) in scaled.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v *= d[i] * d[j];
        }
    }
    let kappa = cond_inf(&scaled);
    let bound = (n as f64).sqrt() * kappa * tol().to_f64();
    let x_tilde_ref: Vec<f64> = reference.iter().zip(&d).map(|(x, di)| x / di).collect();
    let scaled_err = |x: &[Fix128]| {
        let norm = x_tilde_ref.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        x.iter()
            .zip(&reference)
            .zip(&d)
            .map(|((xi, ri), di)| (xi.to_f64() - ri).abs() / di)
            .fold(0.0, f64::max)
            / norm
    };
    println!("κ∞(DAD) = {kappa:e}, bound {bound:e}");
    for method in [KrylovMethod::Gmres, KrylovMethod::BiCgStab] {
        let identity = IdentityPreconditioner::new(n);
        let sol = ok(
            solve_equilibrated(&a, &eq, &identity, &b, method, &cfg),
            "equilibrated",
        );
        let e = scaled_err(&sol.x);
        println!(
            "{method:?}: iterations {}, scaled error {e:e}, T error {:e}",
            sol.stats.iterations,
            t_err(&sol.x)
        );
        assert!(e <= bound, "{method:?}: {e:e} > {bound:e}");
        assert!(t_err(&sol.x) < 1e-9, "{method:?} T: {:e}", t_err(&sol.x));
        if method == KrylovMethod::Gmres {
            assert!(
                unequilibrated_t >= 1e3 * t_err(&sol.x),
                "unequilibrated {unequilibrated_t:e} vs equilibrated {:e}",
                t_err(&sol.x)
            );
        }
    }
    // With a Jacobi preconditioner on the equilibrated diagonal.
    let diag: Vec<Fix128> = (0..n).map(|i| a.get(i, i)).collect();
    let jacobi =
        JacobiPreconditioner::from_diagonal(eq.equilibrate_diagonal(&diag).unwrap()).unwrap();
    let sol = ok(
        solve_equilibrated(&a, &eq, &jacobi, &b, KrylovMethod::Gmres, &cfg),
        "jacobi equilibrated",
    );
    assert!(t_err(&sol.x) < 1e-9);
}

/// The steel / 10 mm row of the table clears the floor by `1.15` (ratio
/// `4.0e9 < 2³² = 4.29e9`), so `unscaled` accepts it; PLA (`2.69e10`) does not.
#[test]
fn unscaled_floor_check_matches_the_material_table() {
    let steel = BlockEquilibration::unscaled(&[3, 3], &[fx(2.0e9), fx(0.5)]);
    assert!(steel.is_ok(), "{steel:?}");
    let aluminium = BlockEquilibration::unscaled(&[3, 3], &[fx(7.0e8), fx(2.0)]);
    assert!(aluminium.is_ok());
    let pla = BlockEquilibration::unscaled(&[3, 3], &[fx(7.0e5), fx(2.6e-5)]);
    assert!(matches!(
        pla,
        Err(LinearSolverError::BlockBelowProductFloor { block: 1, .. })
    ));
    // Equilibrated, the same pair is accepted.
    assert!(BlockEquilibration::try_new(&[3, 3], &[fx(7.0e5), fx(2.6e-5)]).is_ok());
}

// ----------------------------------------------------------------------------
// determinism
// ----------------------------------------------------------------------------

/// Same input ⇒ same bits; and a symmetrically permuted system yields the
/// permuted solution **bit for bit**, because every inner product is a sum of
/// independently truncated products and `Fix128` addition is wrapping integer
/// addition (associative and commutative).
#[test]
fn bit_identical_and_permutation_equivariant() {
    let (a, b) = advection_diffusion(10);
    let n = b.len();
    let perm: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % n).collect(); // 7 ⟂ 31
    let pa = dense_exact(n, |i, j| a.get(perm[i], perm[j]));
    let pb: Vec<Fix128> = perm.iter().map(|&p| b[p]).collect();
    for method in ["gmres", "bicgstab"] {
        let run = |m: &DenseMatrix, rhs: &[Fix128]| {
            let jac = JacobiPreconditioner::from_dense(m).unwrap();
            if method == "gmres" {
                ok(
                    gmres(m, &jac, rhs, &config(500).with_restart(10).unwrap()),
                    method,
                )
            } else {
                ok(bicgstab(m, &jac, rhs, &config(500)), method)
            }
        };
        let first = run(&a, &b);
        let second = run(&a, &b);
        assert_eq!(first, second, "{method}: repeat");
        let permuted = run(&pa, &pb);
        for (i, &p) in perm.iter().enumerate() {
            assert_eq!(permuted.x[i], first.x[p], "{method}: entry {i}");
        }
        assert_eq!(
            permuted.stats.residual_history,
            first.stats.residual_history
        );
    }
}

/// The property the previous test rests on, checked directly: a sum of
/// `Fix128` products is the same in every order.
#[test]
fn fix128_sum_of_products_is_order_independent() {
    let v: Vec<Fix128> = (0..97)
        .map(|i| Fix128::from_ratio((i * 7919 % 1009) as i64 - 504, 37 + i as i64))
        .collect();
    let w: Vec<Fix128> = (0..97)
        .map(|i| Fix128::from_ratio(1 + i as i64, 13))
        .collect();
    let forward = v.iter().zip(&w).fold(Fix128::ZERO, |s, (a, b)| s + *a * *b);
    let backward = v
        .iter()
        .zip(&w)
        .rev()
        .fold(Fix128::ZERO, |s, (a, b)| s + *a * *b);
    let mut strided = Fix128::ZERO;
    for start in 0..5 {
        for i in (start..97).step_by(5) {
            strided = strided + v[i] * w[i];
        }
    }
    assert_eq!(forward, backward);
    assert_eq!(forward, strided);
}

// ----------------------------------------------------------------------------
// degenerate inputs (each with its documented verdict)
// ----------------------------------------------------------------------------

/// `dim = 0`: the empty solution, zero iterations (the right-hand side is the
/// zero vector of the empty space).
#[test]
fn empty_system_returns_the_empty_solution() {
    let a = DenseMatrix::try_new(0, Vec::new()).unwrap();
    for sol in [
        gmres(&a, &IdentityPreconditioner::new(0), &[], &config(5)),
        bicgstab(&a, &IdentityPreconditioner::new(0), &[], &config(5)),
    ] {
        let sol = ok(sol, "empty");
        assert!(sol.x.is_empty());
        assert_eq!(sol.stats.iterations, 0);
    }
}

/// `b = 0`: `x = 0` exactly, zero iterations — even for a singular operator.
#[test]
fn zero_rhs_returns_zero_immediately() {
    let a = dense(3, |_, _| 0.0);
    let b = vec![Fix128::ZERO; 3];
    for sol in [
        gmres(&a, &IdentityPreconditioner::new(3), &b, &config(5)),
        bicgstab(&a, &IdentityPreconditioner::new(3), &b, &config(5)),
    ] {
        let sol = ok(sol, "zero rhs");
        assert_eq!(sol.x, vec![Fix128::ZERO; 3]);
        assert_eq!(sol.stats.iterations, 0);
        assert_eq!(sol.stats.residual_norm, Fix128::ZERO);
    }
}

/// Zero operator, `b ≠ 0`: GMRES reports a singular Hessenberg matrix, and
/// BiCGStab a vanishing `(r̂, A p)`.
#[test]
fn zero_operator_is_a_breakdown() {
    let a = dense(3, |_, _| 0.0);
    let b = vec![Fix128::ONE; 3];
    let g = gmres(&a, &IdentityPreconditioner::new(3), &b, &config(10));
    assert!(
        matches!(
            g,
            Err(LinearSolverError::Breakdown {
                kind: BreakdownKind::SingularHessenberg,
                iterations: 0,
                ..
            })
        ),
        "{g:?}"
    );
    let s = bicgstab(&a, &IdentityPreconditioner::new(3), &b, &config(10));
    assert!(
        matches!(
            s,
            Err(LinearSolverError::Breakdown {
                kind: BreakdownKind::DirectionOrthogonal,
                ..
            })
        ),
        "{s:?}"
    );
}

/// Singular with `b` outside the range (`diag(1, 0)`, `b = (1, 1)`): no
/// solution exists; GMRES must report Breakdown (the second Arnoldi column is
/// dependent) and never `Ok`.
#[test]
fn inconsistent_singular_system_is_not_reported_as_solved() {
    let a = dense(2, |i, j| if i == 0 && j == 0 { 1.0 } else { 0.0 });
    let b = vec![Fix128::ONE, Fix128::ONE];
    let g = gmres(&a, &IdentityPreconditioner::new(2), &b, &config(50));
    assert!(
        matches!(
            g,
            Err(LinearSolverError::Breakdown {
                kind: BreakdownKind::SingularHessenberg,
                ..
            })
        ),
        "{g:?}"
    );
    let s = bicgstab(&a, &IdentityPreconditioner::new(2), &b, &config(50));
    assert!(
        matches!(
            s,
            Err(LinearSolverError::Breakdown { .. } | LinearSolverError::Stagnated { .. })
        ),
        "{s:?}"
    );
}

/// The classic BiCGStab breakdown: `A = [[0,1],[1,0]]`, `b = (1,0)` gives
/// `(r̂, A r) = 0` at the first step. GMRES solves the same system (`x = (0,1)`).
#[test]
fn bicgstab_breakdown_where_gmres_succeeds() {
    let a = dense(2, |i, j| if i == j { 0.0 } else { 1.0 });
    let b = vec![Fix128::ONE, Fix128::ZERO];
    let s = bicgstab(&a, &IdentityPreconditioner::new(2), &b, &config(10));
    assert!(
        matches!(
            s,
            Err(LinearSolverError::Breakdown {
                kind: BreakdownKind::DirectionOrthogonal,
                iterations: 0,
                ..
            })
        ),
        "{s:?}"
    );
    let g = ok(
        gmres(&a, &IdentityPreconditioner::new(2), &b, &config(10)),
        "gmres",
    );
    assert_eq!(g.x, vec![Fix128::ZERO, Fix128::ONE]);
}

/// GMRES(1) on a 90° rotation never reduces the residual (`A v ⟂ v`): it must
/// stop as `Stagnated` after the window, not run out the budget.
#[test]
fn restarted_gmres_on_a_rotation_stagnates() {
    let a = dense(2, |i, j| match (i, j) {
        (0, 1) => 1.0,
        (1, 0) => -1.0,
        _ => 0.0,
    });
    let b = vec![Fix128::ONE, Fix128::ZERO];
    let cfg = config(1000)
        .with_restart(1)
        .unwrap()
        .with_stagnation_window(16)
        .unwrap();
    let g = gmres(&a, &IdentityPreconditioner::new(2), &b, &cfg);
    match g {
        Err(LinearSolverError::Stagnated {
            iterations,
            relative_residual,
            without_improvement,
        }) => {
            assert_eq!(iterations, 16);
            assert_eq!(without_improvement, 16);
            assert_eq!(relative_residual, Fix128::ONE);
        }
        other => panic!("expected Stagnated, got {other:?}"),
    }
    // Full GMRES solves it in two steps: `A x = (x₁, −x₀) = (1, 0)` ⇒ x = (0, 1).
    let full = ok(
        gmres(&a, &IdentityPreconditioner::new(2), &b, &config(10)),
        "full",
    );
    assert_eq!(full.stats.iterations, 2);
    assert!(max_rel_err(&full.x, &[0.0, 1.0]) < 1e-15);
}

/// A budget too small to converge is `NotConverged` (not `Stagnated`): the
/// residual of the advection–diffusion solve is still falling after 5 steps.
#[test]
fn small_budget_is_not_converged() {
    let (a, b) = advection_diffusion(10);
    let n = b.len();
    for r in [
        gmres(
            &a,
            &IdentityPreconditioner::new(n),
            &b,
            &config(5).with_restart(n).unwrap(),
        ),
        bicgstab(&a, &IdentityPreconditioner::new(n), &b, &config(5)),
    ] {
        match r {
            Err(LinearSolverError::NotConverged {
                iterations,
                relative_residual,
            }) => {
                assert_eq!(iterations, 5);
                assert!(relative_residual > tol() && relative_residual < Fix128::ONE);
            }
            other => panic!("expected NotConverged, got {other:?}"),
        }
    }
}

/// Dimension mismatches and invalid configuration are refused with the
/// lengths / fault named.
#[test]
fn mismatches_and_invalid_configuration_are_errors() {
    let a = dense(3, |i, j| if i == j { 1.0 } else { 0.0 });
    assert_eq!(
        gmres(
            &a,
            &IdentityPreconditioner::new(3),
            &[Fix128::ONE; 2],
            &config(5)
        ),
        Err(LinearSolverError::DimensionMismatch {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        bicgstab(
            &a,
            &IdentityPreconditioner::new(4),
            &[Fix128::ONE; 3],
            &config(5)
        ),
        Err(LinearSolverError::DimensionMismatch {
            expected: 3,
            found: 4
        })
    );
    assert_eq!(
        DenseMatrix::try_new(2, vec![Fix128::ONE; 3]),
        Err(LinearSolverError::DimensionMismatch {
            expected: 4,
            found: 3
        })
    );
    assert_eq!(
        KrylovConfig::try_new(0, tol()),
        Err(LinearSolverError::InvalidConfig(
            KrylovConfigFault::ZeroIterationBudget
        ))
    );
    assert_eq!(
        KrylovConfig::try_new(5, Fix128::ZERO),
        Err(LinearSolverError::InvalidConfig(
            KrylovConfigFault::NonPositiveTolerance
        ))
    );
    assert_eq!(
        config(5).with_restart(0),
        Err(LinearSolverError::InvalidConfig(
            KrylovConfigFault::ZeroRestart
        ))
    );
    assert_eq!(
        config(5).with_stagnation_window(0),
        Err(LinearSolverError::InvalidConfig(
            KrylovConfigFault::ZeroStagnationWindow
        ))
    );
    assert_eq!(
        config(5).with_absolute_tolerance(Fix128::NEG_ONE),
        Err(LinearSolverError::InvalidConfig(
            KrylovConfigFault::NegativeAbsoluteTolerance
        ))
    );
    // Jacobi with a zero diagonal entry; block Jacobi with a singular block.
    assert_eq!(
        JacobiPreconditioner::from_diagonal(vec![Fix128::ONE, Fix128::ZERO]),
        Err(LinearSolverError::SingularPreconditioner { index: 1 })
    );
    let singular_block = dense(4, |i, j| if (i < 2 && j < 2) || i == j { 1.0 } else { 0.0 });
    assert_eq!(
        BlockJacobiPreconditioner::from_dense(&singular_block, &[2, 2]),
        Err(LinearSolverError::SingularPreconditioner { index: 0 })
    );
    assert_eq!(
        BlockJacobiPreconditioner::from_dense(&a, &[1, 1]),
        Err(LinearSolverError::DimensionMismatch {
            expected: 3,
            found: 2
        })
    );
    // Equilibration layouts.
    assert_eq!(
        BlockEquilibration::try_new(&[], &[]),
        Err(LinearSolverError::InvalidBlockLayout)
    );
    assert_eq!(
        BlockEquilibration::try_new(&[2, 0], &[Fix128::ONE, Fix128::ONE]),
        Err(LinearSolverError::InvalidBlockLayout)
    );
    assert_eq!(
        BlockEquilibration::try_new(&[2, 1], &[Fix128::ONE, Fix128::ZERO]),
        Err(LinearSolverError::InvalidBlockLayout)
    );
    assert_eq!(
        BlockEquilibration::try_new(&[1], &[Fix128::from_raw(0, 1)]),
        Err(LinearSolverError::Equilibration(
            ConfigFault::ScaleOutOfRange
        ))
    );
    let eq = BlockEquilibration::try_new(&[2, 1], &[Fix128::ONE, Fix128::ONE]).unwrap();
    assert_eq!(eq.dim(), 3);
    assert_eq!(eq.block_count(), 2);
    assert_eq!(eq.scale(2), None);
    assert_eq!(
        solve_equilibrated(
            &dense(4, |i, j| if i == j { 1.0 } else { 0.0 }),
            &eq,
            &IdentityPreconditioner::new(4),
            &[Fix128::ONE; 4],
            KrylovMethod::Gmres,
            &config(5)
        ),
        Err(LinearSolverError::DimensionMismatch {
            expected: 4,
            found: 3
        })
    );
}

/// `FnOperator` is the same operator as the dense matrix it wraps; the
/// absolute tolerance stops a solve whose relative target is unreachable.
#[test]
fn closure_operator_and_absolute_tolerance() {
    let (a, b) = advection_diffusion(1);
    let n = b.len();
    let closure = FnOperator::new(n, |x: &[Fix128], y: &mut [Fix128]| a.apply(x, y));
    let via_dense = ok(
        gmres(&a, &IdentityPreconditioner::new(n), &b, &config(200)),
        "dense",
    );
    let via_closure = ok(
        gmres(&closure, &IdentityPreconditioner::new(n), &b, &config(200)),
        "fn",
    );
    assert_eq!(via_dense, via_closure);
    // An absolute tolerance of 1e-3 stops long before the relative 2⁻⁴⁰.
    let loose = config(200).with_absolute_tolerance(fx(1e-3)).unwrap();
    let early = ok(
        gmres(&a, &IdentityPreconditioner::new(n), &b, &loose),
        "abs",
    );
    assert!(early.stats.iterations < via_dense.stats.iterations);
    assert!(early.stats.residual_norm <= fx(1e-3));
    assert!(BREAKDOWN_RELATIVE < tol().double());
}

/// Rank-deficient matrices whose entries are not exact in `Fix128` — `v vᵀ`
/// with `v = (1/3, 2/3, 2/3)` (rank one) and `v vᵀ + w vᵀ` with
/// `v = (1/7, 3/7, 6/7)`, `w = (1/9, 4/9, −2/3)` — and `b = e_k` outside the
/// range. Rounding leaves the second projected diagonal at the `2⁻⁶⁴` level
/// instead of exactly zero, so only the relative breakdown threshold (against
/// the gain the operator showed at the first step) catches it; compared with
/// zero, the same solves end in `ArithmeticOverflow`, `Stagnated`,
/// `NotConverged` or a late breakdown (measured). Both methods must report the
/// breakdown at the second step: GMRES `SingularHessenberg`, BiCGStab
/// `DirectionOrthogonal` (`A p` is rounding noise of arbitrary direction, so
/// its cosine with `r̂` is not small).
#[test]
fn rounded_rank_deficient_matrices_break_down_at_the_second_step() {
    let q = |pairs: &[(i64, i64)]| -> Vec<Fix128> {
        pairs
            .iter()
            .map(|&(a, b)| Fix128::from_ratio(a, b))
            .collect()
    };
    let rank_one = q(&[(1, 3), (2, 3), (2, 3)]);
    let v = q(&[(1, 7), (3, 7), (6, 7)]);
    let w = q(&[(1, 9), (4, 9), (-2, 3)]);
    let matrices = [
        dense_exact(3, |i, j| rank_one[i] * rank_one[j]),
        dense_exact(3, |i, j| v[i] * v[j] + w[i] * v[j]),
    ];
    for (m, a) in matrices.iter().enumerate() {
        for k in 0..3 {
            let mut b = vec![Fix128::ZERO; 3];
            b[k] = Fix128::ONE;
            let g = gmres(a, &IdentityPreconditioner::new(3), &b, &config(60));
            assert!(
                matches!(
                    g,
                    Err(LinearSolverError::Breakdown {
                        kind: BreakdownKind::SingularHessenberg,
                        iterations: 1,
                        ..
                    })
                ),
                "matrix {m}, e{k}: {g:?}"
            );
            let s = bicgstab(a, &IdentityPreconditioner::new(3), &b, &config(60));
            assert!(
                matches!(
                    s,
                    Err(LinearSolverError::Breakdown {
                        kind: BreakdownKind::DirectionOrthogonal,
                        iterations: 1,
                        ..
                    })
                ),
                "matrix {m}, e{k}: {s:?}"
            );
        }
    }
}
