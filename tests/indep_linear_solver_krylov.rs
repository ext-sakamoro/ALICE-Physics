//! Independent oracles for `linear_solver` (GMRES(m) and `BiCGStab`).
//!
//! The existing oracles in `analytic_linear_solver.rs` use a FEM cantilever,
//! a dense 12×12 SPD matrix, 1-D advection–diffusion with `N = 32`, a
//! badly scaled diagonal, a block-diagonal matrix and the PLA block system.
//! None of those inputs is reused here. The matrices below are built in this
//! file with dyadic entries (exact in both `Fix128` and `f64`) and every
//! expected value is derived in `f64` in this file:
//!
//! - Gaussian elimination with partial pivoting (direct solve),
//! - the minimal-residual recurrences that GMRES(1) and GMRES(2) must follow
//!   cycle by cycle (restart semantics),
//! - the first steps of the van der Vorst recurrence (`BiCGStab`),
//! - Krylov termination bounds from the minimal polynomial (a Jordan block)
//!   and from the number of distinct eigenvalues (2-D Poisson, closed-form
//!   spectrum `4 − 2cos(iπ/7) − 2cos(jπ/7)`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `cos` only builds the reference spectrum; dense loops index matrices.
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::linear_solver::{
    bicgstab, gmres, BreakdownKind, DenseMatrix, FnOperator, IdentityPreconditioner,
    JacobiPreconditioner, KrylovConfig, KrylovSolution, LinearSolverError,
};
use alice_physics::math::Fix128;

// ----------------------------------------------------------------------------
// helpers
// ----------------------------------------------------------------------------

/// `2⁻⁴⁰`.
fn tol40() -> Fix128 {
    Fix128::from_raw(0, 1 << 24)
}

fn cfg(max_iterations: u32, rel: Fix128, restart: usize) -> KrylovConfig {
    KrylovConfig::try_new(max_iterations, rel)
        .expect("config")
        .with_restart(restart)
        .expect("restart")
}

/// Dense matrix from an `f64` generator whose values must be dyadic.
fn dense(n: usize, a: &[Vec<f64>]) -> DenseMatrix {
    let mut e = Vec::with_capacity(n * n);
    for row in a {
        for &v in row {
            let f = Fix128::from_f64(v);
            assert_eq!(f.to_f64(), v, "entry {v} is not exact in Fix128");
            e.push(f);
        }
    }
    DenseMatrix::try_new(n, e).expect("square")
}

fn fvec(v: &[f64]) -> Vec<Fix128> {
    v.iter()
        .map(|&x| {
            let f = Fix128::from_f64(x);
            assert_eq!(f.to_f64(), x, "rhs {x} is not exact");
            f
        })
        .collect()
}

fn f64s(v: &[Fix128]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// Gaussian elimination with partial pivoting.
fn gepp(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if a[i][k].abs() > a[p][k].abs() {
                p = i;
            }
        }
        a.swap(k, p);
        b.swap(k, p);
        assert!(a[k][k].abs() > 1e-300, "singular in reference");
        for i in k + 1..n {
            let l = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= l * a[k][j];
            }
            b[i] -= l * b[k];
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

/// `κ∞(A)` with the inverse from GEPP column by column.
fn cond_inf(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let norm = |m: &dyn Fn(usize, usize) -> f64| {
        (0..n)
            .map(|i| (0..n).map(|j| m(i, j).abs()).sum::<f64>())
            .fold(0.0, f64::max)
    };
    let mut inv = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = gepp(a.to_vec(), e);
        for i in 0..n {
            inv[i][j] = col[i];
        }
    }
    norm(&|i, j| a[i][j]) * norm(&|i, j| inv[i][j])
}

fn matvec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(p, q)| p * q).sum())
        .collect()
}

fn dotf(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(p, q)| p * q).sum()
}

fn norm2(a: &[f64]) -> f64 {
    dotf(a, a).sqrt()
}

fn rel_inf_err(x: &[f64], r: &[f64]) -> f64 {
    let scale = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    x.iter()
        .zip(r)
        .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
        / scale
}

fn ok(r: Result<KrylovSolution, LinearSolverError>, what: &str) -> KrylovSolution {
    match r {
        Ok(s) => s,
        Err(e) => panic!("{what}: {e:?}"),
    }
}

/// Deterministic LCG (Knuth MMIX constants), returning a dyadic in `[-1, 1]`
/// with 8 fractional bits.
struct Lcg(u64);

impl Lcg {
    fn next_dyadic(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let k = ((self.0 >> 33) % 513) as i64 - 256; // −256..=256
        k as f64 / 256.0
    }
}

/// Solve with both methods and check the answer against GEPP.
fn check_against_direct(a: &[Vec<f64>], b: &[f64], restart: usize, what: &str) {
    let n = b.len();
    let op = dense(n, a);
    let rhs = fvec(b);
    let reference = gepp(a.to_vec(), b.to_vec());
    let kappa = cond_inf(a);
    // ‖x − x*‖/‖x*‖ ≤ κ · ‖r‖/‖b‖ (∞ vs 2 norm: factor √n), plus slack 4.
    let bound = 4.0 * (n as f64).sqrt() * kappa * tol40().to_f64();
    let id = IdentityPreconditioner::new(n);
    let jac = JacobiPreconditioner::from_dense(&op).expect("jacobi");
    let runs: [(&str, Result<KrylovSolution, LinearSolverError>); 4] = [
        (
            "gmres(m)",
            gmres(&op, &id, &rhs, &cfg(2000, tol40(), restart)),
        ),
        (
            "gmres(m)+jacobi",
            gmres(&op, &jac, &rhs, &cfg(2000, tol40(), restart)),
        ),
        ("bicgstab", bicgstab(&op, &id, &rhs, &cfg(2000, tol40(), 1))),
        (
            "bicgstab+jacobi",
            bicgstab(&op, &jac, &rhs, &cfg(2000, tol40(), 1)),
        ),
    ];
    for (name, result) in runs {
        let sol = ok(result, &format!("{what} {name}"));
        let x = f64s(&sol.x);
        let err = rel_inf_err(&x, &reference);
        assert!(
            err <= bound,
            "{what} {name}: rel err {err:e} > bound {bound:e} (κ∞ = {kappa:e})"
        );
        // The residual the solver reports must be the true one, recomputed
        // here in f64 from the returned x.
        let ax = matvec(a, &x);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        let rel_res = norm2(&r) / norm2(b);
        assert!(
            rel_res <= 2.0 * tol40().to_f64(),
            "{what} {name}: true relative residual {rel_res:e}"
        );
        // f64 evaluation of `b − A x` is itself only accurate to about
        // `n · ε · ‖A‖∞ ‖x‖∞` per entry.
        let a_inf = a
            .iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
            .fold(0.0, f64::max);
        let x_inf = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let f64_noise = (n as f64) * f64::EPSILON * a_inf * x_inf * (n as f64).sqrt() / norm2(b);
        let reported = sol.stats.residual_norm.to_f64() / norm2(b);
        assert!(
            (reported - rel_res).abs() <= f64_noise,
            "{what} {name}: reported {reported:e} vs recomputed {rel_res:e}"
        );
    }
}

// ----------------------------------------------------------------------------
// direct-solve comparisons on independently built systems
// ----------------------------------------------------------------------------

/// 2-D convection–diffusion on an 8×8 interior grid, central differences,
/// multiplied through by `h²`: centre `4`, west/east `−1 ∓ Px`, south/north
/// `−1 ∓ Py` with `Px = 3/8`, `Py = −1/4` (dyadic). Non-symmetric, n = 64.
#[test]
fn convection_diffusion_2d_matches_direct_solve() {
    let m = 8;
    let n = m * m;
    let (px, py) = (0.375, -0.25);
    let mut a = vec![vec![0.0; n]; n];
    for j in 0..m {
        for i in 0..m {
            let k = j * m + i;
            a[k][k] = 4.0;
            if i > 0 {
                a[k][k - 1] = -1.0 - px;
            }
            if i + 1 < m {
                a[k][k + 1] = -1.0 + px;
            }
            if j > 0 {
                a[k][k - m] = -1.0 - py;
            }
            if j + 1 < m {
                a[k][k + m] = -1.0 + py;
            }
        }
    }
    // Source: a bump in one quadrant plus a sign change elsewhere.
    let b: Vec<f64> = (0..n)
        .map(|k| {
            let (i, j) = (k % m, k / m);
            if i < 4 && j < 4 {
                1.0
            } else if i >= 6 {
                -0.5
            } else {
                0.0
            }
        })
        .collect();
    check_against_direct(&a, &b, 12, "conv-diff 2D");
}

/// A dense non-symmetric, strictly diagonally dominant matrix (n = 40) from a
/// fixed-seed generator: off-diagonals in `[−1, 1]` (8 fractional bits),
/// `a_ii = ±(Σ_j |a_ij| + 1)` with alternating sign so the matrix is
/// indefinite.
#[test]
fn random_diagonally_dominant_matches_direct_solve() {
    let n = 40;
    let mut g = Lcg(0x5EED_2026_1007);
    let mut a = vec![vec![0.0; n]; n];
    for (i, row) in a.iter_mut().enumerate() {
        let mut s = 0.0;
        for (j, v) in row.iter_mut().enumerate() {
            if i != j {
                *v = g.next_dyadic();
                s += v.abs();
            }
        }
        row[i] = if i % 2 == 0 { s + 1.0 } else { -(s + 1.0) };
    }
    let b: Vec<f64> = (0..n).map(|_| g.next_dyadic()).collect();
    check_against_direct(&a, &b, 8, "random diag-dominant");
}

// ----------------------------------------------------------------------------
// Krylov termination from the spectrum / minimal polynomial
// ----------------------------------------------------------------------------

/// `A = diag(J₃(2), J₃(2), J₃(2))` (three 3×3 Jordan blocks with eigenvalue
/// 2) has minimal polynomial `(λ − 2)³`, so for every `b` the Krylov space
/// has dimension ≤ 3 and full GMRES terminates in at most 3 steps; when `b`
/// has a non-zero last entry in some block it needs exactly 3. Not
/// diagonalisable, so this is not covered by any eigenvalue-count argument.
#[test]
fn gmres_terminates_at_minimal_polynomial_degree_of_jordan_matrix() {
    let n = 9;
    let mut a = vec![vec![0.0; n]; n];
    for blk in 0..3 {
        for r in 0..3 {
            let k = 3 * blk + r;
            a[k][k] = 2.0;
            if r < 2 {
                a[k][k + 1] = 1.0;
            }
        }
    }
    let b = vec![0.5, -1.0, 1.0, 0.25, 0.0, -0.75, 1.0, 1.0, 0.125];
    let op = dense(n, &a);
    let sol = ok(
        gmres(
            &op,
            &IdentityPreconditioner::new(n),
            &fvec(&b),
            &cfg(100, tol40(), 50),
        ),
        "jordan",
    );
    assert_eq!(sol.stats.iterations, 3, "degree of (λ−2)³ is 3");
    let reference = gepp(a, b);
    assert!(rel_inf_err(&f64s(&sol.x), &reference) < 1e-11);
}

/// 2-D Poisson (5-point, 6×6 interior, n = 36). The spectrum is
/// `λ_ij = 4 − 2cos(iπ/7) − 2cos(jπ/7)`, `i, j = 1..6`; `cos(kπ/7) =
/// −cos((7−k)π/7)` makes many coincide. A is symmetric, hence
/// diagonalisable, so full GMRES terminates in at most `d` steps where `d` is
/// the number of distinct eigenvalues (computed here: 19), well below n.
#[test]
fn gmres_terminates_within_distinct_eigenvalue_count_on_poisson_2d() {
    let m = 6;
    let n = m * m;
    // Distinct eigenvalues from the closed form.
    let mut eig: Vec<f64> = Vec::new();
    for i in 1..=m {
        for j in 1..=m {
            let l = 4.0
                - 2.0 * (i as f64 * core::f64::consts::PI / 7.0).cos()
                - 2.0 * (j as f64 * core::f64::consts::PI / 7.0).cos();
            if !eig.iter().any(|&e| (e - l).abs() < 1e-9) {
                eig.push(l);
            }
        }
    }
    let distinct = eig.len();
    assert_eq!(distinct, 19, "closed-form count");

    let op = FnOperator::new(n, move |x: &[Fix128], y: &mut [Fix128]| {
        for j in 0..m {
            for i in 0..m {
                let k = j * m + i;
                let mut s = x[k] * Fix128::from_int(4);
                if i > 0 {
                    s = s - x[k - 1];
                }
                if i + 1 < m {
                    s = s - x[k + 1];
                }
                if j > 0 {
                    s = s - x[k - m];
                }
                if j + 1 < m {
                    s = s - x[k + m];
                }
                y[k] = s;
            }
        }
    });
    // Point source in a corner: non-zero projection on every eigenspace.
    let mut b = vec![0.0; n];
    b[0] = 1.0;
    let sol = ok(
        gmres(
            &op,
            &IdentityPreconditioner::new(n),
            &fvec(&b),
            &cfg(200, tol40(), 200),
        ),
        "poisson",
    );
    assert!(
        sol.stats.iterations as usize <= distinct,
        "iterations {} > distinct eigenvalues {distinct}",
        sol.stats.iterations
    );
    assert_eq!(sol.stats.restarts, 0);
    // Direct check.
    let mut a = vec![vec![0.0; n]; n];
    for j in 0..m {
        for i in 0..m {
            let k = j * m + i;
            a[k][k] = 4.0;
            if i > 0 {
                a[k][k - 1] = -1.0;
            }
            if i + 1 < m {
                a[k][k + 1] = -1.0;
            }
            if j > 0 {
                a[k][k - m] = -1.0;
            }
            if j + 1 < m {
                a[k][k + m] = -1.0;
            }
        }
    }
    let reference = gepp(a, b);
    assert!(rel_inf_err(&f64s(&sol.x), &reference) < 1e-10);
}

// ----------------------------------------------------------------------------
// restart semantics, derived from the minimal-residual recurrence
// ----------------------------------------------------------------------------

/// Non-symmetric tridiagonal-plus test matrix with positive definite
/// symmetric part (`4I` dominates), n = 24.
fn restart_matrix() -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = 24;
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = 4.0;
        if i + 1 < n {
            a[i][i + 1] = 1.5;
            a[i + 1][i] = -0.75;
        }
        if i + 3 < n {
            a[i][i + 3] = 0.5;
        }
    }
    let b: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64 - 5.0) / 4.0).collect();
    (a, b)
}

/// GMRES(1) is the minimal-residual iteration `x ← x + α r`,
/// `α = (r, Ar)/(Ar, Ar)`, restarted every step. Each cycle's recorded entry
/// is the recomputed true residual, so the whole history must follow the
/// `f64` recurrence, and `restarts = cycles − 1 = iterations − 1`.
#[test]
fn gmres1_history_follows_minimal_residual_recurrence() {
    let (a, b) = restart_matrix();
    let n = b.len();
    let rel = Fix128::from_f64(1.0 / 1_048_576.0); // 2⁻²⁰
    let sol = ok(
        gmres(
            &dense(n, &a),
            &IdentityPreconditioner::new(n),
            &fvec(&b),
            &cfg(500, rel, 1),
        ),
        "gmres(1)",
    );
    let h = f64s(&sol.stats.residual_history);
    assert_eq!(h.len(), sol.stats.iterations as usize + 1);
    assert_eq!(sol.stats.restarts + 1, sol.stats.iterations);

    let mut x = vec![0.0; n];
    let mut r = b.clone();
    assert!((h[0] - norm2(&r)).abs() <= 1e-12 * norm2(&b));
    for (k, &hk) in h.iter().enumerate().skip(1) {
        let w = matvec(&a, &r);
        let alpha = dotf(&r, &w) / dotf(&w, &w);
        for i in 0..n {
            x[i] += alpha * r[i];
        }
        let ax = matvec(&a, &x);
        r = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        let want = norm2(&r);
        assert!(
            (hk - want).abs() <= 1e-9 * norm2(&b),
            "cycle {k}: history {hk:e} vs MR recurrence {want:e}"
        );
    }
    assert!(rel_inf_err(&f64s(&sol.x), &x) < 1e-8);
}

/// GMRES(2): each cycle minimises `‖r − A(c₁ r + c₂ A r)‖` (solved here by a
/// 2×2 least-squares problem in `f64` via modified Gram–Schmidt on
/// `[A r, A² r]`). The history alternates the in-cycle one-step estimate
/// (= the MR residual from the same `r`) and the cycle-end true residual;
/// `restarts = ⌈iterations / 2⌉ − 1`.
#[test]
fn gmres2_history_follows_two_step_minimal_residual() {
    let (a, b) = restart_matrix();
    let n = b.len();
    let rel = Fix128::from_f64(1.0 / 1_048_576.0);
    let sol = ok(
        gmres(
            &dense(n, &a),
            &IdentityPreconditioner::new(n),
            &fvec(&b),
            &cfg(500, rel, 2),
        ),
        "gmres(2)",
    );
    let it = sol.stats.iterations as usize;
    let h = f64s(&sol.stats.residual_history);
    assert_eq!(h.len(), it + 1);
    assert_eq!(sol.stats.restarts as usize, it.div_ceil(2) - 1);

    let mut x = vec![0.0; n];
    let mut r = b.clone();
    let scale = norm2(&b);
    let mut k = 1;
    while k <= it {
        let w1 = matvec(&a, &r);
        // One-step estimate inside the cycle.
        let a1 = dotf(&r, &w1) / dotf(&w1, &w1);
        let r1: Vec<f64> = r.iter().zip(&w1).map(|(p, q)| p - a1 * q).collect();
        assert!(
            (h[k] - norm2(&r1)).abs() <= 1e-9 * scale,
            "entry {k}: {:e} vs one-step MR {:e}",
            h[k],
            norm2(&r1)
        );
        if k == it {
            // The run ended mid-cycle (converged after one step): the cycle's
            // update is the one-step minimiser.
            for i in 0..n {
                x[i] += a1 * r[i];
            }
            break;
        }
        // Two-step minimiser: columns c1 = A r, c2 = A² r; MGS QR.
        let w2 = matvec(&a, &w1);
        let n1 = norm2(&w1);
        let q1: Vec<f64> = w1.iter().map(|v| v / n1).collect();
        let r12 = dotf(&q1, &w2);
        let u: Vec<f64> = w2.iter().zip(&q1).map(|(p, q)| p - r12 * q).collect();
        let r22 = norm2(&u);
        let q2: Vec<f64> = u.iter().map(|v| v / r22).collect();
        let g1 = dotf(&q1, &r);
        let g2 = dotf(&q2, &r);
        let c2 = g2 / r22;
        let c1 = (g1 - r12 * c2) / n1;
        for i in 0..n {
            x[i] += c1 * r[i] + c2 * w1[i];
        }
        let ax = matvec(&a, &x);
        r = b.iter().zip(&ax).map(|(p, q)| p - q).collect();
        assert!(
            (h[k + 1] - norm2(&r)).abs() <= 1e-9 * scale,
            "entry {}: {:e} vs two-step MR {:e}",
            k + 1,
            h[k + 1],
            norm2(&r)
        );
        k += 2;
    }
    assert!(rel_inf_err(&f64s(&sol.x), &x) < 1e-8);
}

// ----------------------------------------------------------------------------
// BiCGStab recurrence
// ----------------------------------------------------------------------------

/// The first four `BiCGStab` residual norms equal the van der Vorst recurrence
/// evaluated in `f64` here (`r̂ = r₀ = b`, `x₀ = 0`), on the non-symmetric
/// restart matrix.
#[test]
fn bicgstab_first_steps_follow_van_der_vorst_recurrence() {
    let (a, b) = restart_matrix();
    let n = b.len();
    let sol = ok(
        bicgstab(
            &dense(n, &a),
            &IdentityPreconditioner::new(n),
            &fvec(&b),
            &cfg(500, tol40(), 1),
        ),
        "bicgstab",
    );
    let h = f64s(&sol.stats.residual_history);
    assert!(h.len() > 5, "needs at least 5 steps, got {}", h.len() - 1);

    let shadow = b.clone();
    let mut r = b.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let (mut rho_prev, mut alpha, mut omega) = (1.0, 1.0, 1.0);
    let scale = norm2(&b);
    for (k, &hk) in h.iter().enumerate().take(5).skip(1) {
        let rho = dotf(&shadow, &r);
        let beta = (rho / rho_prev) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        v = matvec(&a, &p);
        alpha = rho / dotf(&shadow, &v);
        let s: Vec<f64> = r.iter().zip(&v).map(|(p, q)| p - alpha * q).collect();
        let t = matvec(&a, &s);
        omega = dotf(&t, &s) / dotf(&t, &t);
        r = s.iter().zip(&t).map(|(p, q)| p - omega * q).collect();
        rho_prev = rho;
        let want = norm2(&r);
        assert!(
            (hk - want).abs() <= 1e-10 * scale,
            "step {k}: history {hk:e} vs recurrence {want:e}"
        );
    }
}

// ----------------------------------------------------------------------------
// degenerate inputs
// ----------------------------------------------------------------------------

/// n = 1: `5 x = 3` ⇒ `x = 3/5` within the tolerance, one iteration.
#[test]
fn one_by_one_system() {
    let op = dense(1, &[vec![5.0]]);
    let b = fvec(&[3.0]);
    let id = IdentityPreconditioner::new(1);
    for (name, r) in [
        ("gmres", gmres(&op, &id, &b, &cfg(10, tol40(), 5))),
        ("bicgstab", bicgstab(&op, &id, &b, &cfg(10, tol40(), 1))),
    ] {
        let sol = ok(r, name);
        assert_eq!(sol.stats.iterations, 1, "{name}");
        assert!((sol.x[0].to_f64() - 0.6).abs() <= 1e-12, "{name}");
    }
}

/// Zero right-hand side on a non-singular, non-symmetric system: `x = 0`
/// exactly, zero iterations, for both methods and with a preconditioner.
#[test]
fn zero_rhs_on_nonsingular_system_is_exact_zero() {
    let (a, b) = restart_matrix();
    let n = b.len();
    let op = dense(n, &a);
    let jac = JacobiPreconditioner::from_dense(&op).expect("jacobi");
    let zero = vec![Fix128::ZERO; n];
    for (name, r) in [
        ("gmres", gmres(&op, &jac, &zero, &cfg(10, tol40(), 4))),
        ("bicgstab", bicgstab(&op, &jac, &zero, &cfg(10, tol40(), 1))),
    ] {
        let sol = ok(r, name);
        assert_eq!(sol.stats.iterations, 0, "{name}");
        assert!(sol.x.iter().all(|v| v.is_zero()), "{name}");
    }
}

/// Consistent singular system: `A = [[2,1,0],[1,2,0],[0,0,0]]`,
/// `b = (3,3,0) ∈ range(A)`. The Krylov space lies in `range(A)`, so GMRES
/// from `x₀ = 0` returns the solution in that range, `(1, 1, 0)`.
#[test]
fn consistent_singular_system_is_solved_in_the_range() {
    let a = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 2.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let op = dense(3, &a);
    let sol = ok(
        gmres(
            &op,
            &IdentityPreconditioner::new(3),
            &fvec(&[3.0, 3.0, 0.0]),
            &cfg(20, tol40(), 10),
        ),
        "gmres consistent singular",
    );
    let x = f64s(&sol.x);
    for (got, want) in x.iter().zip([1.0, 1.0, 0.0]) {
        assert!((got - want).abs() <= 1e-11, "{x:?}");
    }
}

/// Same matrix with `b = (3, 3, 1) ∉ range(A)`: no solution exists. The
/// module documents this as `Breakdown` for GMRES; neither method may
/// return `Ok`.
///
/// `A b = (9, 9, 0)` and `A² b = 3 A b`, so the Krylov space stops growing
/// after `{b, A b}`: the first Arnoldi step succeeds and the second column of
/// the projected matrix is singular. The breakdown is therefore reported
/// after exactly one completed step, inside the first cycle.
#[test]
fn inconsistent_rank_two_system_is_refused() {
    let a = vec![
        vec![2.0, 1.0, 0.0],
        vec![1.0, 2.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    let op = dense(3, &a);
    let b = fvec(&[3.0, 3.0, 1.0]);
    let id = IdentityPreconditioner::new(3);
    match gmres(&op, &id, &b, &cfg(50, tol40(), 10)) {
        Err(LinearSolverError::Breakdown {
            kind: BreakdownKind::SingularHessenberg,
            relative_residual,
            iterations,
        }) => {
            assert_eq!(iterations, 1, "Krylov dimension argument");
            // The least-squares residual is the component of b outside the
            // range: 1/‖b‖ = 1/√19.
            let want = 1.0 / 19.0f64.sqrt();
            assert!(
                (relative_residual.to_f64() - want).abs() < 1e-9,
                "{relative_residual:?}"
            );
        }
        other => panic!("gmres: expected SingularHessenberg, got {other:?}"),
    }
    assert!(bicgstab(&op, &id, &b, &cfg(50, tol40(), 1)).is_err());
}
