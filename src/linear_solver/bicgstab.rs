//! van der Vorst `BiCGStab`, right-preconditioned, with breakdown detection.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{
    cosine_vanishes, dot_scaled, norm, norm_scaled, relative, residual, start, BreakdownKind,
    KrylovConfig, KrylovSolution, KrylovStats, LinearOperator, LinearSolverError, OperatorGain,
    Preconditioner, Stagnation, Start,
};
use crate::math::Fix128;

/// Solve `A x = b` with right-preconditioned `BiCGStab` (van der Vorst 1992).
///
/// One iteration is a BiCG step along `p̂ = M⁻¹ p` followed by a one-dimensional
/// minimal-residual step along `ŝ = M⁻¹ s`:
///
/// ```text
/// ρ = (r̂, r);   β = (ρ/ρ₋₁)(α/ω);   p = r + β (p − ω v)
/// v = A M⁻¹ p;  α = ρ / (r̂, v);      s = r − α v
/// t = A M⁻¹ s;  ω = (t, s) / (t, t); x += α M⁻¹p + ω M⁻¹s;  r = s − ω t
/// ```
///
/// Each of `(r̂, r)`, `(r̂, v)` and `(t, s)` is tested against
/// [`super::BREAKDOWN_RELATIVE`] times the product of its operands' norms; a
/// vanishing one ends the solve with the matching [`BreakdownKind`].
///
/// When the recurrence residual meets the target the true residual
/// `b − A x` is recomputed. If the two have drifted apart and the true one does
/// not meet the target, the method restarts from the true residual (with a new
/// shadow vector `r̂ = r`), which counts in [`KrylovStats::restarts`].
///
/// # Errors
///
/// As [`super::gmres`], with the `BiCGStab` breakdown kinds.
pub fn bicgstab<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
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

    let mut x = vec![Fix128::ZERO; n];
    let mut r = b.to_vec();
    let mut r_norm = rhs_norm;
    let mut shadow = r.clone();
    let mut shadow_norm = norm_scaled(&shadow)?;
    let mut p = vec![Fix128::ZERO; n];
    let mut v = vec![Fix128::ZERO; n];
    let mut p_hat = vec![Fix128::ZERO; n];
    let mut s = vec![Fix128::ZERO; n];
    let mut s_hat = vec![Fix128::ZERO; n];
    let mut t = vec![Fix128::ZERO; n];
    let mut rho_prev = super::Scaled {
        mantissa: Fix128::ONE,
        exponent: 0,
    };
    let mut alpha = Fix128::ONE;
    let mut omega = Fix128::ONE;
    let mut gain = OperatorGain::new();
    let mut fresh = true;

    let mut history = vec![rhs_norm];
    let mut stagnation = Stagnation::new(rhs_norm, config.stagnation_window);
    let mut iterations = 0u32;
    let mut restarts = 0u32;

    let breakdown = |iterations, kind, r_norm| LinearSolverError::Breakdown {
        iterations,
        kind,
        relative_residual: relative(r_norm, rhs_norm),
    };

    loop {
        if iterations >= config.max_iterations {
            return Err(LinearSolverError::NotConverged {
                iterations,
                relative_residual: relative(r_norm, rhs_norm),
            });
        }

        let rho = dot_scaled(&shadow, &r)?;
        if cosine_vanishes(rho, shadow_norm, norm_scaled(&r)?) {
            return Err(breakdown(
                iterations,
                BreakdownKind::ShadowResidualOrthogonal,
                r_norm,
            ));
        }
        if fresh {
            p.copy_from_slice(&r);
            fresh = false;
        } else {
            let beta = rho.ratio(rho_prev)? * (alpha / omega);
            for ((pi, &ri), &vi) in p.iter_mut().zip(&r).zip(&v) {
                *pi = ri + beta * (*pi - omega * vi);
            }
        }
        precond.apply(&p, &mut p_hat);
        op.apply(&p_hat, &mut v);
        let shadow_v = dot_scaled(&shadow, &v)?;
        let v_norm = norm_scaled(&v)?;
        let v_negligible = gain.observe(v_norm, norm_scaled(&p)?)?;
        if v_negligible || cosine_vanishes(shadow_v, shadow_norm, v_norm) {
            return Err(breakdown(
                iterations,
                BreakdownKind::DirectionOrthogonal,
                r_norm,
            ));
        }
        alpha = rho.ratio(shadow_v)?;
        for ((si, &ri), &vi) in s.iter_mut().zip(&r).zip(&v) {
            *si = ri - alpha * vi;
        }
        let s_scaled = norm_scaled(&s)?;
        let s_norm = s_scaled.value()?;
        iterations += 1;

        if s_norm <= target {
            for (xi, &ph) in x.iter_mut().zip(&p_hat) {
                *xi = *xi + alpha * ph;
            }
            history.push(s_norm);
            match refresh(op, &x, b, &mut r, target)? {
                Refresh::Converged(true_norm) => {
                    if let Some(last) = history.last_mut() {
                        *last = true_norm;
                    }
                    return Ok(finish(
                        x, iterations, restarts, history, true_norm, rhs_norm,
                    ));
                }
                Refresh::Drifted(true_norm) => {
                    if let Some(last) = history.last_mut() {
                        *last = true_norm;
                    }
                    r_norm = true_norm;
                    shadow.copy_from_slice(&r);
                    shadow_norm = norm_scaled(&shadow)?;
                    fresh = true;
                    restarts += 1;
                    if stagnation.record(r_norm) {
                        return Err(stagnated(iterations, &stagnation, rhs_norm));
                    }
                    continue;
                }
            }
        }

        precond.apply(&s, &mut s_hat);
        op.apply(&s_hat, &mut t);
        let t_scaled = norm_scaled(&t)?;
        let ts = dot_scaled(&t, &s)?;
        let t_negligible = gain.observe(t_scaled, s_scaled)?;
        if t_negligible || cosine_vanishes(ts, t_scaled, s_scaled) {
            return Err(breakdown(
                iterations,
                BreakdownKind::StabilizerVanished,
                s_norm,
            ));
        }
        omega = ts.ratio(dot_scaled(&t, &t)?)?;
        for ((xi, &ph), &sh) in x.iter_mut().zip(&p_hat).zip(&s_hat) {
            *xi = *xi + alpha * ph + omega * sh;
        }
        for ((ri, &si), &ti) in r.iter_mut().zip(&s).zip(&t) {
            *ri = si - omega * ti;
        }
        r_norm = norm(&r)?;
        rho_prev = rho;
        history.push(r_norm);

        if r_norm <= target {
            match refresh(op, &x, b, &mut r, target)? {
                Refresh::Converged(true_norm) => {
                    if let Some(last) = history.last_mut() {
                        *last = true_norm;
                    }
                    return Ok(finish(
                        x, iterations, restarts, history, true_norm, rhs_norm,
                    ));
                }
                Refresh::Drifted(true_norm) => {
                    if let Some(last) = history.last_mut() {
                        *last = true_norm;
                    }
                    r_norm = true_norm;
                    shadow.copy_from_slice(&r);
                    shadow_norm = norm_scaled(&shadow)?;
                    fresh = true;
                    restarts += 1;
                }
            }
        }
        if stagnation.record(r_norm) {
            return Err(stagnated(iterations, &stagnation, rhs_norm));
        }
    }
}

enum Refresh {
    Converged(Fix128),
    Drifted(Fix128),
}

/// Recompute `r = b − A x` and judge it.
fn refresh<A: LinearOperator + ?Sized>(
    op: &A,
    x: &[Fix128],
    b: &[Fix128],
    r: &mut [Fix128],
    target: Fix128,
) -> Result<Refresh, LinearSolverError> {
    residual(op, x, b, r);
    let true_norm = norm(r)?;
    Ok(if true_norm <= target {
        Refresh::Converged(true_norm)
    } else {
        Refresh::Drifted(true_norm)
    })
}

fn finish(
    x: Vec<Fix128>,
    iterations: u32,
    restarts: u32,
    residual_history: Vec<Fix128>,
    residual_norm: Fix128,
    rhs_norm: Fix128,
) -> KrylovSolution {
    KrylovSolution {
        x,
        stats: KrylovStats {
            iterations,
            restarts,
            residual_history,
            residual_norm,
            rhs_norm,
        },
    }
}

fn stagnated(iterations: u32, stagnation: &Stagnation, rhs_norm: Fix128) -> LinearSolverError {
    LinearSolverError::Stagnated {
        iterations,
        relative_residual: relative(stagnation.best, rhs_norm),
        without_improvement: stagnation.since,
    }
}
