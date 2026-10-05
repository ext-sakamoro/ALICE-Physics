//! Restarted GMRES(m) with modified Gram–Schmidt and Givens rotations.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{
    dot, hypot, negligible, norm, relative, residual, start, BreakdownKind, KrylovConfig,
    KrylovSolution, KrylovStats, LinearOperator, LinearSolverError, Preconditioner, Stagnation,
    Start,
};
use crate::math::Fix128;

/// Solve `A x = b` with right-preconditioned, restarted GMRES(m).
///
/// Each cycle builds an orthonormal basis `V` of the Krylov space of
/// `A M⁻¹` by the Arnoldi process with **modified** Gram–Schmidt, reduces the
/// Hessenberg matrix to triangular form with Givens rotations as it grows (so
/// `|g_{j+1}|` is the residual norm of the least-squares problem after each
/// step), and at the end of the cycle solves the triangular system, updates
/// `x += M⁻¹ V y` and recomputes the **true** residual `b − A x`, which both
/// decides convergence and starts the next cycle. The restart length is
/// [`KrylovConfig::with_restart`]; with `m ≥ n` no restart occurs and the
/// method terminates in at most `n` steps up to rounding.
///
/// Within a cycle the recorded residual is non-increasing (each rotation
/// multiplies it by `|s| ≤ 1`). `max_iterations` counts Arnoldi steps, i.e.
/// operator applications excluding the residual refreshes.
///
/// # Errors
///
/// [`LinearSolverError::DimensionMismatch`] when `b` or the preconditioner
/// disagrees with `op.dim()`; [`LinearSolverError::NotConverged`],
/// [`LinearSolverError::Stagnated`], [`LinearSolverError::Breakdown`]
/// ([`BreakdownKind::SingularHessenberg`]) and
/// [`LinearSolverError::ArithmeticOverflow`] as documented in
/// [`crate::linear_solver`].
pub fn gmres<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
    op: &A,
    precond: &M,
    b: &[Fix128],
    config: &KrylovConfig,
) -> Result<KrylovSolution, LinearSolverError> {
    let (rhs_norm, target) = match start(op, precond, b, config)? {
        Start::Trivial(solution) => return Ok(solution),
        Start::Run { rhs_norm, target } => (rhs_norm, target),
    };
    let n = op.dim();
    let m = config.restart.min(n);
    let window = config
        .stagnation_window
        .max(u32::try_from(m).unwrap_or(u32::MAX).saturating_add(1));

    let mut x = vec![Fix128::ZERO; n];
    let mut r = b.to_vec();
    let mut beta = rhs_norm;
    let mut history = vec![beta];
    let mut stagnation = Stagnation::new(beta, window);
    let mut iterations = 0u32;
    let mut cycles = 0u32;

    // Krylov basis (m + 1 vectors), Hessenberg columns, rotations, rhs `g`.
    let mut basis: Vec<Vec<Fix128>> = vec![vec![Fix128::ZERO; n]; m + 1];
    let mut hess: Vec<Vec<Fix128>> = vec![vec![Fix128::ZERO; m + 1]; m];
    let mut cs = vec![Fix128::ZERO; m];
    let mut sn = vec![Fix128::ZERO; m];
    let mut g = vec![Fix128::ZERO; m + 1];
    let mut z = vec![Fix128::ZERO; n];
    let mut w = vec![Fix128::ZERO; n];

    loop {
        if beta <= target {
            return Ok(KrylovSolution {
                x,
                stats: KrylovStats {
                    iterations,
                    restarts: cycles.saturating_sub(1),
                    residual_history: history,
                    residual_norm: beta,
                    rhs_norm,
                },
            });
        }
        if iterations >= config.max_iterations {
            return Err(LinearSolverError::NotConverged {
                iterations,
                relative_residual: relative(beta, rhs_norm),
            });
        }

        // v₀ = r / β; g = β e₀.
        for (v, &ri) in basis[0].iter_mut().zip(&r) {
            *v = ri / beta;
        }
        g.fill(Fix128::ZERO);
        g[0] = beta;

        let mut cols = 0usize;
        let mut breakdown = false;
        for j in 0..m {
            precond.apply(&basis[j], &mut z);
            op.apply(&z, &mut w);
            let w_scale = norm(&w)?;
            // Modified Gram–Schmidt: project against each basis vector in turn,
            // using the already-reduced `w`.
            for i in 0..=j {
                let h = dot(&w, &basis[i])?;
                hess[j][i] = h;
                for (wk, &vk) in w.iter_mut().zip(&basis[i]) {
                    *wk = *wk - h * vk;
                }
            }
            let h_next = norm(&w)?;
            hess[j][j + 1] = h_next;

            // Apply the previous rotations to the new column.
            for i in 0..j {
                let a = hess[j][i];
                let bb = hess[j][i + 1];
                hess[j][i] = cs[i] * a + sn[i] * bb;
                hess[j][i + 1] = -(sn[i] * a) + cs[i] * bb;
            }
            // New rotation annihilating h_{j+1, j}.
            let a = hess[j][j];
            let d = hypot(a, h_next)?;
            if negligible(d, w_scale) {
                breakdown = true;
                break;
            }
            cs[j] = a / d;
            sn[j] = h_next / d;
            hess[j][j] = d;
            hess[j][j + 1] = Fix128::ZERO;
            g[j + 1] = -(sn[j] * g[j]);
            g[j] = cs[j] * g[j];

            cols = j + 1;
            iterations += 1;
            let estimate = g[j + 1].abs();
            history.push(estimate);
            let stalled = stagnation.record(estimate);

            let invariant = negligible(h_next, w_scale);
            if !invariant {
                for (v, &wk) in basis[j + 1].iter_mut().zip(&w) {
                    *v = wk / h_next;
                }
            }
            if invariant || estimate <= target || iterations >= config.max_iterations || stalled {
                break;
            }
        }

        if cols == 0 {
            // The very first direction of this cycle already broke down.
            return Err(LinearSolverError::Breakdown {
                iterations,
                kind: BreakdownKind::SingularHessenberg,
                relative_residual: relative(beta, rhs_norm),
            });
        }

        // Back substitution on the triangular `R y = g`.
        let mut y = vec![Fix128::ZERO; cols];
        for i in (0..cols).rev() {
            let mut s = g[i];
            for k in i + 1..cols {
                s = s - hess[k][i] * y[k];
            }
            y[i] = s / hess[i][i];
        }
        // x += M⁻¹ (V y).
        w.fill(Fix128::ZERO);
        for (k, &yk) in y.iter().enumerate() {
            for (wk, &vk) in w.iter_mut().zip(&basis[k]) {
                *wk = *wk + yk * vk;
            }
        }
        precond.apply(&w, &mut z);
        for (xi, &zi) in x.iter_mut().zip(&z) {
            *xi = *xi + zi;
        }

        residual(op, &x, b, &mut r);
        beta = norm(&r)?;
        if let Some(last) = history.last_mut() {
            *last = beta;
        }
        cycles += 1;

        if breakdown && beta > target {
            return Err(LinearSolverError::Breakdown {
                iterations,
                kind: BreakdownKind::SingularHessenberg,
                relative_residual: relative(beta, rhs_norm),
            });
        }
        if beta > target && stagnation.since >= stagnation.window {
            return Err(LinearSolverError::Stagnated {
                iterations,
                relative_residual: relative(stagnation.best, rhs_norm),
                without_improvement: stagnation.since,
            });
        }
    }
}
