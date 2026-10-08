//! Oracle for `linear_solver`'s ill-conditioned-system behaviour
//! (`COV-NUM-080`): a system whose condition number exceeds about `2⁴⁰` is
//! reported as `Breakdown` rather than silently solved to digits the
//! `Fix128` representation does not hold (`src/linear_solver.rs`).
//!
//! The reference is the Hilbert matrix `H_n[i][j] = 1/(i+j+1)`, whose
//! condition numbers are a classic, independently published table (e.g.
//! Wilson, "Condition numbers of real and complex Hilbert matrices", Numer.
//! Math. 8 (1970)); they are not computed by the solver under test.
//!
//! | `n` | `cond(H_n)` (published, `f64`) |
//! |---|---|
//! | 9  | `4.9315e11` |
//! | 10 | `1.6025e13` |
//!
//! `2⁴⁰ ≈ 1.0995e12` sits strictly between these two.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::linear_solver::{gmres, DenseMatrix, IdentityPreconditioner, KrylovConfig};
use alice_physics::math::Fix128;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn hilbert(n: usize) -> DenseMatrix {
    let mut entries = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            entries.push(fx(1.0 / (i + j + 1) as f64));
        }
    }
    DenseMatrix::try_new(n, entries).expect("square")
}

/// `2⁻⁴⁰`, the relative tolerance used throughout the solver's own oracles.
fn tol() -> Fix128 {
    Fix128::from_raw(0, 1 << 24)
}

fn config(max_iterations: u32) -> KrylovConfig {
    KrylovConfig::try_new(max_iterations, tol()).expect("valid config")
}

/// Alternating sign so the exact solution has a large component along the
/// Hilbert matrix's worst-conditioned (smallest-eigenvalue) direction: a
/// smooth, same-sign `x_true` barely excites that direction and GMRES
/// reaches machine precision on it up to at least `n = 40` regardless of
/// `cond(H_n)`, which this oracle cannot use to find the documented
/// threshold at all.
fn adversarial_x_true(n: usize) -> Vec<Fix128> {
    (0..n)
        .map(|i| fx(if i % 2 == 0 { 1.0 } else { -1.0 }))
        .collect()
}

fn solve(
    n: usize,
) -> (
    Result<
        alice_physics::linear_solver::KrylovSolution,
        alice_physics::linear_solver::LinearSolverError,
    >,
    Vec<Fix128>,
) {
    let a = hilbert(n);
    let x_true = adversarial_x_true(n);
    let mut b = vec![Fix128::ZERO; n];
    alice_physics::linear_solver::LinearOperator::apply(&a, &x_true, &mut b);
    let identity = IdentityPreconditioner::new(n);
    (gmres(&a, &identity, &b, &config(2000)), x_true)
}

fn solve_error(n: usize) -> Fix128 {
    let (result, x_true) = solve(n);
    let solution = result.unwrap_or_else(|e| panic!("n={n}: expected Ok, got {e:?}"));
    solution
        .x
        .iter()
        .zip(&x_true)
        .map(|(got, want)| (*got - *want).abs())
        .fold(Fix128::ZERO, |m, d| if d > m { d } else { m })
}

/// `n = 9`: `cond(H_9) ≈ 4.93e11`, below `2⁴⁰ ≈ 1.0995e12` — GMRES reaches
/// the true solution to about `2e-8` (well inside the `cond(A) * tol ≈ 0.45`
/// worst case the sibling solver oracle, `tests/analytic_linear_solver.rs`,
/// bounds by). `1e-4` keeps four orders of magnitude of margin over that
/// while staying four orders of magnitude tighter than the `~1.3` error
/// `n = 10` (past the threshold, below) produces with the same construction
/// — so this bound has teeth against the regime this oracle exists to tell
/// apart, not just against total failure.
#[test]
fn hilbert_nine_is_below_the_breakdown_threshold_and_solves_accurately() {
    let err = solve_error(9);
    let bound = fx(1e-4);
    assert!(
        err <= bound,
        "n=9 (cond well below 2^40): max error {err:?} exceeds bound {bound:?}"
    );
}

/// `n = 10`: `cond(H_10) ≈ 1.60e13`, past `2⁴⁰ ≈ 1.0995e12` — the measured
/// boundary is real (error jumps from `~2e-8` at `n=9` with this same
/// adversarial right-hand side to `O(1)` here, three orders of magnitude
/// past where `cond(A) * tol` would bound it), but `src/linear_solver.rs`'s
/// own doc promises this is "reported as breakdown", and it is not: `gmres`
/// returns `Ok` with a solution that is wrong by about `1.3`, not
/// `Err(Breakdown { .. })`. Pinned as a known defect rather than relaxed to
/// match the observed `Ok`, per COV-NUM-080's own text.
#[test]
#[ignore = "known defect: COV-NUM-080: gmres returns Ok(..) with an O(1)-wrong \
            solution for a Hilbert system past the ~2^40 condition-number \
            threshold instead of Err(Breakdown { .. }) as src/linear_solver.rs's \
            own doc comment promises (n=10, cond(H_10)~1.6e13, adversarial \
            alternating-sign right-hand side)"]
fn hilbert_ten_is_past_the_breakdown_threshold_and_should_report_breakdown() {
    let (result, _x_true) = solve(10);
    match result {
        Err(alice_physics::linear_solver::LinearSolverError::Breakdown { .. }) => {}
        other => panic!(
            "n=10 (cond(H_10)~1.6e13, past 2^40) should be Err(Breakdown {{ .. }}), got {other:?}"
        ),
    }
}
