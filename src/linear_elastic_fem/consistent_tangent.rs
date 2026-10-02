//! The consistent tangent of a hyperelastic law, applied matrix-free, and the
//! Newton step that uses it.
//!
//! # The tangent
//!
//! `cauchy_stress` gives `P = J σ F⁻ᵀ` with
//!
//! ```text
//! P = 2(W₁ + I₁W₂) F − 2 W₂ F FᵀF + q(J) F⁻ᵀ,       q = κ J (J−1) − p_ref
//! ```
//!
//! and its directional derivative along a displacement gradient `δF` is
//!
//! ```text
//! dP = 2(W₁ + I₁W₂) δF + 4(W₁₁ + W₂)(F:δF) F
//!      − 2 W₂ (δF FᵀF + F δFᵀ F + F Fᵀ δF)
//!      + J q'(J) (F⁻ᵀ:δF) F⁻ᵀ − q F⁻ᵀ δFᵀ F⁻ᵀ
//! ```
//!
//! (`W₁₂ = 0` for every model here). The element carries `V₀ dP ∇₀Nₐ` — the same
//! integral [`element_force_from_piola`] evaluates for the stress. Everything
//! that needs a division (`F⁻ᵀ`, `q`) is formed **once per Newton iteration**
//! and the action per conjugate gradient iteration is multiplications only.
//!
//! The law derives from a strain energy, so the tangent is symmetric; it is not
//! positive definite everywhere (a state with compressive lateral stress has a
//! negative eigenvalue in the lateral shear block), which is why the conjugate
//! gradient below stops on negative curvature instead of treating it as an
//! error.

use super::{
    apply_rotated_stiffness, build_preconditioner, conjugate_gradient, corotational_residual,
    deformation_gradient, dot, element_force_from_piola, gather, max_abs, relative,
    rotated_stiffness_diagonal, stiffness_diagonal, Assembly, Element, FemError, SolverConfig,
    RESIDUAL_NORM_FLOOR,
};
use crate::hyperelastic::{tangent_constants, HyperelasticModel};
use crate::math::{Fix128, Mat3Fix, PolarError, Vec3Fix};

/// `a + b`, entry by entry.
fn add(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(
            a.col0.x + b.col0.x,
            a.col0.y + b.col0.y,
            a.col0.z + b.col0.z,
        ),
        Vec3Fix::new(
            a.col1.x + b.col1.x,
            a.col1.y + b.col1.y,
            a.col1.z + b.col1.z,
        ),
        Vec3Fix::new(
            a.col2.x + b.col2.x,
            a.col2.y + b.col2.y,
            a.col2.z + b.col2.z,
        ),
    )
}

/// `a − b`, entry by entry.
fn sub(a: Mat3Fix, b: Mat3Fix) -> Mat3Fix {
    Mat3Fix::from_cols(
        Vec3Fix::new(
            a.col0.x - b.col0.x,
            a.col0.y - b.col0.y,
            a.col0.z - b.col0.z,
        ),
        Vec3Fix::new(
            a.col1.x - b.col1.x,
            a.col1.y - b.col1.y,
            a.col1.z - b.col1.z,
        ),
        Vec3Fix::new(
            a.col2.x - b.col2.x,
            a.col2.y - b.col2.y,
            a.col2.z - b.col2.z,
        ),
    )
}

/// `A : B = Σ Aᵢⱼ Bᵢⱼ`.
fn contract(a: Mat3Fix, b: Mat3Fix) -> Fix128 {
    a.col0.x * b.col0.x
        + a.col0.y * b.col0.y
        + a.col0.z * b.col0.z
        + a.col1.x * b.col1.x
        + a.col1.y * b.col1.y
        + a.col1.z * b.col1.z
        + a.col2.x * b.col2.x
        + a.col2.y * b.col2.y
        + a.col2.z * b.col2.z
}

/// Everything the tangent action needs at one element, fixed at a state.
struct ElementState {
    f: Mat3Fix,
    /// `F⁻ᵀ`.
    fit: Mat3Fix,
    /// `FᵀF` and `FFᵀ`; only read when `W₂ ≠ 0`.
    ftf: Mat3Fix,
    fft: Mat3Fix,
    /// `2(W₁ + I₁W₂)`.
    a1: Fix128,
    /// `4(W₁₁ + W₂)`.
    a2: Fix128,
    /// `W₂`.
    w2: Fix128,
    /// `q(J)`.
    q: Fix128,
    /// `J q'(J) = κ J (2J−1)`.
    jq: Fix128,
}

/// The tangent of the stress at one displacement field.
pub(super) struct TangentField {
    states: Vec<ElementState>,
}

impl TangentField {
    /// Forms the per-element quantities at `u`.
    ///
    /// # Errors
    ///
    /// [`FemError::RotationFailed`] (`Inverted`) for an element with `det F ≤ 0`,
    /// the same refusal the stress itself gives.
    pub(super) fn at(
        elements: &[Element],
        u: &[Fix128],
        model: &HyperelasticModel,
        bulk_modulus: Fix128,
    ) -> Result<Self, FemError> {
        let mut states = Vec::with_capacity(elements.len());
        for (tet, element) in elements.iter().enumerate() {
            let f = deformation_gradient(element, &gather(element, u));
            let refused = FemError::RotationFailed {
                tet,
                cause: PolarError::Inverted,
            };
            let j = f.determinant();
            if j <= Fix128::ZERO {
                return Err(refused);
            }
            let fit = f.inverse().ok_or(refused)?.transpose();
            let ftf = f.transpose().mul_mat(f);
            let fft = f.mul_mat(f.transpose());
            let i1 = fft.col0.x + fft.col1.y + fft.col2.z;
            let (w1, w2, w11, p_ref) = tangent_constants(model, i1);
            let one = Fix128::ONE;
            states.push(ElementState {
                f,
                fit,
                ftf,
                fft,
                a1: Fix128::from_int(2) * (w1 + i1 * w2),
                a2: Fix128::from_int(4) * (w11 + w2),
                w2,
                q: bulk_modulus * j * (j - one) - p_ref,
                jq: bulk_modulus * j * (Fix128::from_int(2) * j - one),
            });
        }
        Ok(Self { states })
    }

    /// `dP = A : δF` for one element.
    fn d_piola(state: &ElementState, df: Mat3Fix) -> Mat3Fix {
        let mut dp = add(
            df.scale(state.a1),
            state.f.scale(state.a2 * contract(state.f, df)),
        );
        if !state.w2.is_zero() {
            let dft = df.transpose();
            let bracket = add(
                add(df.mul_mat(state.ftf), state.f.mul_mat(dft).mul_mat(state.f)),
                state.fft.mul_mat(df),
            );
            dp = sub(dp, bracket.scale(Fix128::from_int(2) * state.w2));
        }
        dp = add(dp, state.fit.scale(state.jq * contract(state.fit, df)));
        sub(
            dp,
            state
                .fit
                .mul_mat(df.transpose())
                .mul_mat(state.fit)
                .scale(state.q),
        )
    }

    /// `out = K_t p`, the nodal forces of the stress change `p` induces. `out`
    /// is overwritten.
    pub(super) fn apply(&self, elements: &[Element], p: &[Fix128], out: &mut [Fix128]) {
        for slot in out.iter_mut() {
            *slot = Fix128::ZERO;
        }
        for (element, state) in elements.iter().zip(self.states.iter()) {
            // `deformation_gradient` of a field is `I + Σ pᵢ ⊗ ∇Nᵢ`.
            let g = deformation_gradient(element, &gather(element, p));
            let df = sub(g, Mat3Fix::IDENTITY);
            let force = element_force_from_piola(element, Self::d_piola(state, df));
            for (node_force, &node) in force.iter().zip(element.nodes.iter()) {
                let base = node * 3;
                for axis in 0..3 {
                    out[base + axis] = out[base + axis] + node_force[axis];
                }
            }
        }
    }
}

/// `out = K_t(u) p`, applied once. The solver builds a [`TangentField`] and
/// applies it many times; this is the one-shot form the unit tests read.
#[cfg(test)]
pub(super) fn apply_hyperelastic_tangent(
    elements: &[Element],
    u: &[Fix128],
    model: &HyperelasticModel,
    bulk_modulus: Fix128,
    p: &[Fix128],
    out: &mut [Fix128],
) -> Result<(), FemError> {
    TangentField::at(elements, u, model, bulk_modulus)?.apply(elements, p, out);
    Ok(())
}

/// How a truncated conjugate gradient run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ending {
    /// The residual reached its target.
    Converged,
    /// The iteration budget ran out first; the iterate is still a usable step.
    Budget,
    /// `pᵀKp ≤ 0` at the very first direction: nothing was solved.
    NegativeAtStart,
    /// `pᵀKp ≤ 0` after some progress: the iterate so far is the step.
    NegativeLater,
}

struct Truncated {
    x: Vec<Fix128>,
    iterations: u32,
    residual_norm: Fix128,
    b_norm: Fix128,
    target: Fix128,
    ending: Ending,
}

/// Preconditioned conjugate gradient that **stops on negative curvature and
/// returns what it has** instead of failing: the tangent is not positive
/// definite everywhere, and a Newton step built from the iterate before the
/// curvature turned negative is still a descent direction.
fn truncated_cg<A>(
    b: &[Fix128],
    is_free: &[bool],
    precond: &[Fix128],
    config: &SolverConfig,
    mut apply: A,
) -> Truncated
where
    A: FnMut(&[Fix128], &mut [Fix128]),
{
    let ndof = b.len();
    let mut scratch = vec![Fix128::ZERO; ndof];
    let mut x = vec![Fix128::ZERO; ndof];
    let mut r = b.to_vec();
    let mut z = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            z[d] = r[d] * precond[d];
        }
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(b, b).sqrt();
    let requested = config.relative_tolerance() * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };
    let mut iterations = 0u32;
    let mut residual_norm = dot(&r, &r).sqrt();
    let mut ending = Ending::Converged;
    while residual_norm > target {
        if iterations >= config.max_iterations() {
            ending = Ending::Budget;
            break;
        }
        apply(&p, &mut scratch);
        for (d, value) in scratch.iter_mut().enumerate() {
            if !is_free[d] {
                *value = Fix128::ZERO;
            }
        }
        let pkp = dot(&p, &scratch);
        if pkp <= Fix128::ZERO {
            ending = if iterations == 0 {
                Ending::NegativeAtStart
            } else {
                Ending::NegativeLater
            };
            break;
        }
        let alpha = rz / pkp;
        for d in 0..ndof {
            if is_free[d] {
                x[d] = x[d] + alpha * p[d];
                r[d] = r[d] - alpha * scratch[d];
            }
        }
        for d in 0..ndof {
            if is_free[d] {
                z[d] = r[d] * precond[d];
            }
        }
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        for d in 0..ndof {
            if is_free[d] {
                p[d] = z[d] + beta * p[d];
            }
        }
        rz = rz_next;
        residual_norm = dot(&r, &r).sqrt();
        iterations += 1;
    }
    Truncated {
        x,
        iterations,
        residual_norm,
        b_norm,
        target,
        ending,
    }
}

/// What one consistent Newton step reports back to [`super::solve_corotational`].
#[derive(Debug)]
pub(super) struct StepReport {
    pub cg_iterations: u32,
    pub relative_residual: Fix128,
    pub effective_relative_tolerance: Fix128,
}

/// Halvings of the step length tried before the step is refused.
const BACKTRACKS: u32 = 16;

/// One Newton step with the consistent tangent: solve `K_t(u) δ = r(u)` for the
/// free rows, then move `u` by the largest `α = 2⁻ᵏ` that lowers the residual
/// and keeps every element non-inverted.
///
/// A tangent that is not positive definite along the first direction is not
/// solved with; the step falls back to the co-rotational linear surrogate the
/// modified Newton iteration uses, which is positive definite by construction.
pub(super) fn newton_step(
    assembly: &Assembly<'_>,
    u: &mut [Fix128],
    f_ext: &[Fix128],
    law: (HyperelasticModel, Fix128),
    linear: &SolverConfig,
    residual: &mut [Fix128],
) -> Result<StepReport, FemError> {
    let &Assembly {
        elements,
        rotations,
        lame: (lambda, mu),
        is_free,
    } = assembly;
    let (model, bulk) = law;
    let ndof = u.len();
    corotational_residual(assembly, u, f_ext, Some(law), residual)?;
    let before = max_abs(residual);

    let tangent = TangentField::at(elements, u, &model, bulk)?;
    let diag = stiffness_diagonal(elements, lambda, mu, ndof);
    let precond = build_preconditioner(&diag, is_free, linear)?;
    let solved = truncated_cg(residual, is_free, &precond, linear, |p, out| {
        tangent.apply(elements, p, out);
    });
    let (delta, cg_iterations, relative_residual, effective) = match solved.ending {
        Ending::NegativeAtStart => {
            // The surrogate is positive definite; use it for this step only.
            let rotated = rotated_stiffness_diagonal(elements, rotations, lambda, mu, ndof);
            let precond = build_preconditioner(&rotated, is_free, linear)?;
            let cg = conjugate_gradient(residual, is_free, &precond, linear, |p, out| {
                apply_rotated_stiffness(elements, rotations, p, lambda, mu, out);
            })?;
            (
                cg.x,
                cg.iterations,
                relative(cg.residual_norm, cg.b_norm),
                relative(cg.target, cg.b_norm),
            )
        }
        _ => (
            solved.x,
            solved.iterations,
            relative(solved.residual_norm, solved.b_norm),
            relative(solved.target, solved.b_norm),
        ),
    };

    let mut trial_residual = vec![Fix128::ZERO; ndof];
    let accepted = backtrack(before, u, &delta, is_free, |trial| {
        // An inverted element is a refusal, not a failure: a shorter step may
        // still be admissible.
        corotational_residual(assembly, trial, f_ext, Some(law), &mut trial_residual)
            .ok()
            .map(|()| max_abs(&trial_residual))
    });
    match accepted {
        Some(moved) => {
            u.copy_from_slice(&moved);
            Ok(StepReport {
                cg_iterations,
                relative_residual,
                effective_relative_tolerance: effective,
            })
        }
        None => Err(FemError::NotConverged {
            iterations: cg_iterations,
            relative_residual: relative(before, solved.b_norm),
        }),
    }
}

/// The largest `α = 2⁻ᵏ` (`k ≤ BACKTRACKS`) for which `u + α δ` has a residual
/// **strictly below** `before`, as `u` moved on the free rows. `norm_at` returns
/// the residual norm at a trial point, or `None` when the point is refused (an
/// inverted element); a refused or non-improving trial halves the step.
pub(super) fn backtrack<F>(
    before: Fix128,
    u: &[Fix128],
    delta: &[Fix128],
    is_free: &[bool],
    mut norm_at: F,
) -> Option<Vec<Fix128>>
where
    F: FnMut(&[Fix128]) -> Option<Fix128>,
{
    let mut trial = vec![Fix128::ZERO; u.len()];
    let mut alpha = Fix128::ONE;
    for _ in 0..=BACKTRACKS {
        for d in 0..u.len() {
            trial[d] = if is_free[d] {
                u[d] + alpha * delta[d]
            } else {
                u[d]
            };
        }
        if let Some(norm) = norm_at(&trial) {
            if norm < before {
                return Some(trial);
            }
        }
        alpha = alpha * Fix128::from_ratio(1, 2);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperelastic::HyperelasticModel;
    use crate::linear_elastic_fem::{
        deformation_gradient, element_force_from_piola, gather, hyperelastic_stress,
    };

    const MU: f64 = 3.0;
    const BULK: f64 = 100.0;
    type M3 = [[f64; 3]; 3];

    fn fx(v: f64) -> Fix128 {
        Fix128::from_ratio((v * 1_048_576.0).round() as i64, 1_048_576)
    }

    /// `v` as the fixed-point implementation sees it (rounded to 2⁻²⁰). The
    /// closed-form oracle is evaluated on the **same** numbers: `2/3 − 1` is not
    /// representable, and comparing the implementation at its rounded value with
    /// a formula at the exact one measures the rounding, not the tangent.
    fn q(v: f64) -> f64 {
        fx(v).to_f64()
    }

    /// The reference tetrahedron: `X = (0,0,0), (1,0,0), (0,1,0), (0,0,1)`.
    fn unit_tet() -> Element {
        Element {
            nodes: [0, 1, 2, 3],
            grad: [
                [fx(-1.0), fx(-1.0), fx(-1.0)],
                [fx(1.0), fx(0.0), fx(0.0)],
                [fx(0.0), fx(1.0), fx(0.0)],
                [fx(0.0), fx(0.0), fx(1.0)],
            ],
            volume: Fix128::from_ratio(1, 6),
        }
    }

    const REF: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    const GRAD: [[f64; 3]; 4] = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    const VOLUME: f64 = 1.0 / 6.0;

    /// `F` (row `i`, column `J`) for the nodal field `u` of the unit tet.
    fn gradient_of(u: &[f64]) -> M3 {
        let mut f = [[0.0; 3]; 3];
        for (a, g) in GRAD.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    f[i][j] += u[a * 3 + i] * g[j];
                }
            }
        }
        for (i, row) in f.iter_mut().enumerate() {
            row[i] += 1.0;
        }
        f
    }

    fn det(m: &M3) -> f64 {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }

    fn inv(m: &M3) -> M3 {
        let d = det(m);
        let c = |a: usize, b: usize, c: usize, e: usize| m[a][b] * m[c][e];
        [
            [
                (c(1, 1, 2, 2) - c(1, 2, 2, 1)) / d,
                (c(0, 2, 2, 1) - c(0, 1, 2, 2)) / d,
                (c(0, 1, 1, 2) - c(0, 2, 1, 1)) / d,
            ],
            [
                (c(1, 2, 2, 0) - c(1, 0, 2, 2)) / d,
                (c(0, 0, 2, 2) - c(0, 2, 2, 0)) / d,
                (c(0, 2, 1, 0) - c(0, 0, 1, 2)) / d,
            ],
            [
                (c(1, 0, 2, 1) - c(1, 1, 2, 0)) / d,
                (c(0, 1, 2, 0) - c(0, 0, 2, 1)) / d,
                (c(0, 0, 1, 1) - c(0, 1, 1, 0)) / d,
            ],
        ]
    }

    fn mul(a: &M3, b: &M3) -> M3 {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        r
    }

    fn tr(m: &M3) -> M3 {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = m[j][i];
            }
        }
        r
    }

    /// `p(J)` and `c(J) = J p'(J)` of the implemented Neo-Hookean law. The
    /// stress is `P = μ F + (p(J) − μ) F⁻ᵀ` with `W = μ/2 (I₁−3) − μ ln J +
    /// κ/2 (J−1)²`: `cauchy_stress` carries `κ(J−1) − p_ref/J` as one isotropic
    /// term, so `p = κ J (J−1)` and `c = J p' = κ J (2J−1)`. (`ln J` appears only
    /// in the energy; neither the stress nor its tangent has a logarithm.)
    fn p_and_c(j: f64) -> (f64, f64) {
        (BULK * j * (j - 1.0), j * BULK * (2.0 * j - 1.0))
    }

    /// `P(F)` in closed form.
    fn piola(f: &M3) -> M3 {
        let (p, _) = p_and_c(det(f));
        let fit = tr(&inv(f));
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = MU * f[i][j] + (p - MU) * fit[i][j];
            }
        }
        r
    }

    /// `A(F) : dF` in closed form (oracle: `sakamoro-ff`'s derivation):
    /// `dP = μ dF + (μ − p) F⁻ᵀ dFᵀ F⁻ᵀ + c (F⁻ᵀ : dF) F⁻ᵀ`.
    fn dpiola(f: &M3, df: &M3) -> M3 {
        let (p, c) = p_and_c(det(f));
        let fit = tr(&inv(f));
        let mid = mul(&mul(&fit, &tr(df)), &fit);
        let contraction: f64 = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| fit[i][j] * df[i][j])
            .sum();
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = MU * df[i][j] + (MU - p) * mid[i][j] + c * contraction * fit[i][j];
            }
        }
        r
    }

    /// Nodal forces `V P ∇Nₐ` for a stress `P`.
    fn forces(p: &M3) -> Vec<f64> {
        let mut out = vec![0.0; 12];
        for (a, g) in GRAD.iter().enumerate() {
            for i in 0..3 {
                out[a * 3 + i] = VOLUME * (p[i][0] * g[0] + p[i][1] * g[1] + p[i][2] * g[2]);
            }
        }
        out
    }

    fn fix(v: &[f64]) -> Vec<Fix128> {
        v.iter().map(|x| fx(*x)).collect()
    }

    fn model() -> HyperelasticModel {
        HyperelasticModel::NeoHookean { mu_mpa: fx(MU) }
    }

    fn element_clone(e: &Element) -> Element {
        Element {
            nodes: e.nodes,
            grad: e.grad,
            volume: e.volume,
        }
    }

    fn tangent(u: &[f64], p: &[f64]) -> Vec<f64> {
        tangent_with(&model(), u, p)
    }

    fn tangent_with(model: &HyperelasticModel, u: &[f64], p: &[f64]) -> Vec<f64> {
        let element = unit_tet();
        let mut out = vec![Fix128::ZERO; 12];
        apply_hyperelastic_tangent(
            &[element_clone(&element)],
            &fix(u),
            model,
            fx(BULK),
            &fix(p),
            &mut out,
        )
        .expect("a regular state");
        out.iter().map(|x| x.to_f64()).collect()
    }

    /// Nodal fields whose gradient is a chosen `F` (diagonal stretches, a shear,
    /// a state with `J ≠ 1`, a generic non-symmetric one).
    fn states() -> Vec<(&'static str, Vec<f64>)> {
        let diag = |l: [f64; 3]| {
            let mut u = vec![0.0; 12];
            u[3] = q(l[0] - 1.0);
            u[7] = q(l[1] - 1.0);
            u[11] = q(l[2] - 1.0);
            u
        };
        vec![
            ("isochoric a = 3/2", diag([2.25, 2.0 / 3.0, 2.0 / 3.0])),
            ("isochoric a = 2", diag([4.0, 0.5, 0.5])),
            ("J != 1: diag(2, 1, 1)", diag([2.0, 1.0, 1.0])),
            ("J != 1: diag(5/4, 9/10, 11/10)", diag([1.25, 0.9, 1.1])),
            (
                "J != 1, dyadic: diag(5/4, 7/8, 9/8)",
                diag([1.25, 0.875, 1.125]),
            ),
            ("compression diag(0.8, 1.3, 0.95)", diag([0.8, 1.3, 0.95])),
            (
                "generic, non-symmetric",
                vec![
                    0.0, 0.0, 0.0, 0.25, 0.125, -0.0625, 0.0625, 0.125, 0.0, -0.125, 0.0625, 0.1875,
                ],
            ),
        ]
    }

    #[test]
    fn the_closed_form_stress_is_the_implemented_one() {
        // The tangent oracle differentiates `P = μ F + (p(J) − μ) F⁻ᵀ`; if that
        // is not the stress `hyperelastic_stress` returns, every other check here
        // would be measuring the wrong law. No tangent is involved.
        let element = unit_tet();
        for (name, u) in states() {
            let f = gradient_of(&u);
            let gradient = deformation_gradient(&element, &gather(&element, &fix(&u)));
            let (_, piola_impl) =
                hyperelastic_stress(&model(), fx(BULK), gradient).expect("regular");
            let expected = piola(&f);
            let rows = [
                [piola_impl.col0.x, piola_impl.col1.x, piola_impl.col2.x],
                [piola_impl.col0.y, piola_impl.col1.y, piola_impl.col2.y],
                [piola_impl.col0.z, piola_impl.col1.z, piola_impl.col2.z],
            ];
            for i in 0..3 {
                for j in 0..3 {
                    let got = rows[i][j].to_f64();
                    assert!(
                        (got - expected[i][j]).abs() <= 1e-9 * expected[i][j].abs().max(1.0),
                        "{name}: P[{i}][{j}] = {got}, the closed form says {}",
                        expected[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn the_tangent_matches_the_closed_form_for_every_nodal_direction() {
        // oracle: dP = μ dF + (μ−p) F⁻ᵀ dFᵀ F⁻ᵀ + c (F⁻ᵀ:dF) F⁻ᵀ, read through
        // the nodal force it carries. All twelve unit directions are used, so
        // every entry of A is reached, off-diagonal ones included.
        for (name, u) in states() {
            let f = gradient_of(&u);
            for d in 0..12 {
                let mut p = vec![0.0; 12];
                p[d] = 1.0;
                let df = {
                    let g = gradient_of(&p);
                    let mut m = g;
                    for (i, row) in m.iter_mut().enumerate() {
                        row[i] -= 1.0;
                    }
                    m
                };
                let expected = forces(&dpiola(&f, &df));
                let got = tangent(&u, &p);
                for k in 0..12 {
                    assert!(
                        (got[k] - expected[k]).abs() <= 1e-8 * expected[k].abs().max(1.0),
                        "{name}: direction {d}, force component {k}: closed form {} against the tangent {}",
                        expected[k],
                        got[k]
                    );
                }
            }
        }
    }

    #[test]
    fn the_second_and_third_terms_are_not_swapped() {
        // At a diagonal F the entries `A_ijij = μ` and `A_ijji = (μ−p)/(f_i f_j)`
        // differ unless p = 0 and f_i f_j = 1, so a transposed index in the
        // `(μ − p)` term is seen here and not by the symmetry check.
        let u = &states()[3].1; // diag(5/4, 9/10, 11/10), J != 1
        let f = gradient_of(u);
        let (p, _) = p_and_c(det(&f));
        let (fi, fj) = (f[0][0], f[1][1]);
        // dF = e_0 ⊗ e_1 (row 0, column 1): dP_01 = A_0101 = μ, dP_10 = A_1001
        let mut dp = vec![0.0; 12];
        dp[3] = 0.0; // build the nodal field of dF = e_0 ⊗ e_1: u_a = e_0 * X_a[1]
        for (a, x) in REF.iter().enumerate() {
            dp[a * 3] = x[1];
        }
        let out = tangent(u, &dp);
        // node 2 (gradient e_y) carries V·P column 1: its x entry is V·dP_01,
        // node 1 (gradient e_x) carries V·P column 0: its y entry is V·dP_10
        let (a_0101, a_1001) = (out[2 * 3] / VOLUME, out[3 + 1] / VOLUME);
        assert!(
            (a_0101 - MU).abs() < 1e-8,
            "A_0101 = {a_0101}, closed form μ = {MU}"
        );
        let expected = (MU - p) / (fi * fj);
        assert!(
            (a_1001 - expected).abs() < 1e-8 * expected.abs().max(1.0),
            "A_1001 = {a_1001}, closed form (μ−p)/(f_i f_j) = {expected}"
        );
        assert!(
            (expected - MU).abs() > 1e-3,
            "the two entries must differ in this state, or the test cannot tell the terms apart"
        );
    }

    #[test]
    fn the_tangent_obeys_the_differentiated_objectivity_identity() {
        // A(F) : (Ω F) = Ω P(F) for skew Ω — the tangent and the stress are
        // checked against each other, with no finite difference in the oracle.
        let omega: M3 = [
            [0.0, 0.25, -0.125],
            [-0.25, 0.0, 0.0625],
            [0.125, -0.0625, 0.0],
        ];
        // `δF = Ω F` has to be exact for the identity to hold to rounding, so the
        // check reads only states whose products with Ω are exact in 2⁻²⁰: the
        // non-dyadic ones (2/3, 0.9) would be rounded before they got here.
        let exact = |u: &[f64]| u.iter().all(|x| (x * 256.0).fract() == 0.0);
        for (name, u) in states().into_iter().filter(|(_, u)| exact(u)) {
            let f = gradient_of(&u);
            // the field whose gradient is Ω F: p_a = Ω x_a with x_a = X_a + u_a
            let mut p = vec![0.0; 12];
            for a in 0..4 {
                let x: Vec<f64> = (0..3).map(|i| REF[a][i] + u[a * 3 + i]).collect();
                for i in 0..3 {
                    p[a * 3 + i] = q((0..3).map(|j| omega[i][j] * x[j]).sum());
                }
            }
            // (gradient_of(p) − I) is Ω F because x is affine in X with gradient F
            let expected = forces(&mul(&omega, &piola(&f)));
            let got = tangent(&u, &p);
            for k in 0..12 {
                assert!(
                    (got[k] - expected[k]).abs() <= 1e-8 * expected[k].abs().max(1.0),
                    "{name}: component {k}: Ω P gives {}, the tangent {}",
                    expected[k],
                    got[k]
                );
            }
        }
    }

    #[test]
    fn at_the_reference_state_the_tangent_is_the_laws_own_small_strain_limit() {
        // The law's small-strain limit is the linear solid with λ = κ (the
        // Neo-Hookean offset is zero) and the model's own μ, so
        // A_1111(I) = κ + 2μ. With the old volumetric term (`K(J−1) − p_ref`,
        // `K = λ + 2μ/3`) it was K + μ = λ + 5μ/3, μ/3 short of the linear
        // λ + 2μ; this is the oracle that pins the fix on the tangent.
        let u = vec![0.0; 12];
        let mut p = vec![0.0; 12];
        for (a, x) in REF.iter().enumerate() {
            p[a * 3] = x[0]; // dF = e_0 ⊗ e_0
        }
        let out = tangent(&u, &p);
        let a_1111 = out[3] / VOLUME;
        assert!(
            (a_1111 - (BULK + 2.0 * MU)).abs() < 1e-7,
            "A_1111(I) = {a_1111}, expected κ + 2μ = {}",
            BULK + 2.0 * MU
        );
    }

    #[test]
    fn the_tangent_is_symmetric() {
        // The law derives from a strain energy, so `⟨K p, q⟩ = ⟨K q, p⟩`.
        let u = states()[5].1.clone();
        let field = |seed: i64| -> Vec<f64> {
            (0..12)
                .map(|d| (((d as i64 * 7 + seed) % 11) - 5) as f64 / 8.0)
                .collect()
        };
        let (p, q) = (field(3), field(8));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let (a, b) = (dot(&tangent(&u, &p), &q), dot(&tangent(&u, &q), &p));
        assert!(
            (a - b).abs() <= 1e-8 * a.abs().max(1.0),
            "⟨Kp,q⟩ = {a}, ⟨Kq,p⟩ = {b}"
        );
    }

    #[test]
    fn the_residual_slope_halves_into_quarters_for_every_model() {
        // ‖r(x + h d) − r(x) − h K d‖ is O(h²) for a consistent tangent and O(h)
        // for an approximate one, so halving h must quarter it. The residual is
        // built from `hyperelastic_stress` directly, so this measures the closed
        // form against the stress that is implemented — for Mooney-Rivlin (the
        // `W₂` terms) and Yeoh (the `W₁₁` term) as well as Neo-Hookean, which is
        // where the closed form has no independent derivation to lean on.
        let models = [
            ("Neo-Hookean", model()),
            (
                "Mooney-Rivlin",
                HyperelasticModel::MooneyRivlin {
                    c1_mpa: fx(1.0),
                    c2_mpa: fx(0.5),
                },
            ),
            (
                "Yeoh",
                HyperelasticModel::Yeoh {
                    c1_mpa: fx(1.5),
                    c2_mpa: fx(0.25),
                    c3_mpa: fx(0.0625),
                },
            ),
        ];
        let element = unit_tet();
        for (name, model) in models {
            let internal = |u: &[f64]| -> Vec<f64> {
                let g = gather(&element, &fix(u));
                let gradient = deformation_gradient(&element, &g);
                let (_, piola) = hyperelastic_stress(&model, fx(BULK), gradient).expect("regular");
                let f = element_force_from_piola(&element, piola);
                f.iter()
                    .flat_map(|n| n.iter().map(|x| x.to_f64()))
                    .collect()
            };
            for state in [3usize, 5] {
                let u = states()[state].1.clone();
                let d: Vec<f64> = (0..12)
                    .map(|k| (((k as i64 * 5 + 2) % 9) - 4) as f64 / 8.0)
                    .collect();
                let kd = tangent_with(&model, &u, &d);
                let base = internal(&u);
                let err = |h: f64| -> f64 {
                    let moved: Vec<f64> = u.iter().zip(&d).map(|(a, b)| a + h * b).collect();
                    let r = internal(&moved);
                    (0..12)
                        .map(|k| {
                            let e = r[k] - base[k] - h * kd[k];
                            e * e
                        })
                        .sum::<f64>()
                        .sqrt()
                };
                let (e1, e2, e3) = (err(1.0 / 16.0), err(1.0 / 32.0), err(1.0 / 64.0));
                let (r1, r2) = (e2 / e1, e3 / e2);
                assert!(
                    (r1 - 0.25).abs() < 0.06 && (r2 - 0.25).abs() < 0.06,
                    "{name}, state {state}: halving h must quarter the error, got ratios {r1:.3} and {r2:.3} (errors {e1:.3e}, {e2:.3e}, {e3:.3e})"
                );
            }
        }
    }

    #[test]
    fn the_tangent_is_symmetric_for_every_model() {
        let models = [
            HyperelasticModel::MooneyRivlin {
                c1_mpa: fx(1.0),
                c2_mpa: fx(0.5),
            },
            HyperelasticModel::Yeoh {
                c1_mpa: fx(1.5),
                c2_mpa: fx(0.25),
                c3_mpa: fx(0.0625),
            },
        ];
        let u = states()[5].1.clone();
        let field = |seed: i64| -> Vec<f64> {
            (0..12)
                .map(|d| (((d as i64 * 7 + seed) % 11) - 5) as f64 / 8.0)
                .collect()
        };
        let (p, q) = (field(3), field(8));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for model in models {
            let (a, b) = (
                dot(&tangent_with(&model, &u, &p), &q),
                dot(&tangent_with(&model, &u, &q), &p),
            );
            assert!(
                (a - b).abs() <= 1e-8 * a.abs().max(1.0),
                "{model:?}: ⟨Kp,q⟩ = {a}, ⟨Kq,p⟩ = {b}"
            );
        }
    }

    // ---- truncated_cg: every way it can end -------------------------------

    fn cg_config(max_iterations: u32) -> SolverConfig {
        SolverConfig::try_new(max_iterations, Fix128::from_raw(0, 1 << 34)).expect("valid")
    }

    /// `out = K p` in exact fixed-point arithmetic: the matrices below are
    /// integer, so no rounding enters the operator the conjugate gradient sees
    /// (rounding it to 2⁻²⁰ makes it neither exact nor symmetric, and the run
    /// ends on the noise instead of on the solution).
    fn matvec(k: &[[i64; 3]; 3]) -> impl Fn(&[Fix128], &mut [Fix128]) + '_ {
        move |p: &[Fix128], out: &mut [Fix128]| {
            for i in 0..3 {
                out[i] = (0..3).fold(Fix128::ZERO, |acc, j| {
                    acc + Fix128::from_int(k[i][j]) * p[j]
                });
            }
        }
    }

    fn run_cg(k: &[[i64; 3]; 3], b: [f64; 3], max_iterations: u32) -> Truncated {
        let free = [true; 3];
        let ones = vec![Fix128::ONE; 3];
        let b: Vec<Fix128> = b.iter().map(|x| fx(*x)).collect();
        truncated_cg(&b, &free, &ones, &cg_config(max_iterations), matvec(k))
    }

    #[test]
    fn truncated_cg_solves_a_positive_definite_system() {
        // oracle: K = [[4,1,0],[1,3,0],[0,0,2]], b = (1,2,3) has the solution
        // (1/11, 7/11, 3/2), worked by hand from the 2×2 block (det 11).
        let k = [[4, 1, 0], [1, 3, 0], [0, 0, 2]];
        let out = run_cg(&k, [1.0, 2.0, 3.0], 100);
        assert_eq!(out.ending, Ending::Converged);
        for (got, want) in out.x.iter().zip([1.0 / 11.0, 7.0 / 11.0, 1.5]) {
            assert!(
                (got.to_f64() - want).abs() < 1e-8,
                "x = {}, expected {want}",
                got.to_f64()
            );
        }
    }

    #[test]
    fn truncated_cg_stops_at_once_on_negative_curvature_at_the_start() {
        // K = diag(1, −1, 1), b = (0, 1, 0): the first direction is b itself and
        // pᵀKp = −1, so nothing was solved and the step must say so.
        let k = [[1, 0, 0], [0, -1, 0], [0, 0, 1]];
        let out = run_cg(&k, [0.0, 1.0, 0.0], 100);
        assert_eq!(out.ending, Ending::NegativeAtStart);
        assert_eq!(out.iterations, 0);
        assert!(
            out.x.iter().all(|v| *v == Fix128::ZERO),
            "no progress may be returned"
        );
    }

    #[test]
    fn truncated_cg_returns_the_iterate_when_the_curvature_turns_negative_later() {
        // K = diag(2, 1, −1), b = (1,1,1). First direction p₀ = b: pᵀKp = 2 > 0,
        // α = 3/2, x₁ = (3/2)(1,1,1). r₁ = b − (3/2)Kp₀ = (−2, −1/2, 5/2),
        // β = |r₁|²/|r₀|² = (21/2)/3 = 7/2, p₁ = r₁ + (7/2)p₀ = (3/2, 3, 6) and
        // p₁ᵀKp₁ = 2(9/4) + 9 − 36 = −45/2 < 0: the run ends with x₁.
        let k = [[2, 0, 0], [0, 1, 0], [0, 0, -1]];
        let out = run_cg(&k, [1.0, 1.0, 1.0], 100);
        assert_eq!(out.ending, Ending::NegativeLater);
        assert_eq!(out.iterations, 1);
        for v in &out.x {
            assert!(
                (v.to_f64() - 1.5).abs() < 1e-9,
                "the iterate before the negative direction is (3/2)(1,1,1), got {}",
                v.to_f64()
            );
        }
    }

    #[test]
    fn truncated_cg_reports_a_spent_budget_with_the_partial_iterate() {
        let k = [[4, 1, 0], [1, 3, 0], [0, 0, 2]];
        let out = run_cg(&k, [1.0, 2.0, 3.0], 1);
        assert_eq!(out.ending, Ending::Budget);
        assert_eq!(out.iterations, 1);
        assert!(
            out.x.iter().any(|v| *v != Fix128::ZERO),
            "one iteration made progress"
        );
    }

    // ---- backtrack: every way the line search can end ---------------------

    fn v(x: f64) -> Vec<Fix128> {
        vec![fx(x)]
    }

    /// `u = 0`, `δ = 1`: the trial point is `α`, and `norm_at` is the residual
    /// there, so each test states the residual curve it wants.
    fn line_search(before: f64, norm_at: impl Fn(f64) -> Option<f64>) -> Option<f64> {
        backtrack(fx(before), &v(0.0), &v(1.0), &[true], |t| {
            norm_at(t[0].to_f64()).map(fx)
        })
        .map(|moved| moved[0].to_f64())
    }

    #[test]
    fn backtrack_takes_the_full_step_when_it_lowers_the_residual() {
        assert_eq!(line_search(1.0, |a| Some(1.0 - 0.5 * a)), Some(1.0));
    }

    #[test]
    fn backtrack_halves_until_the_residual_is_lower() {
        // |1 − 3α| is above 1 for α = 1 (2) and α = 1/2 (1/2 is lower): the first
        // admissible step is 1/2, not 1 and not 1/4.
        assert_eq!(line_search(1.0, |a| Some((1.0 - 3.0 * a).abs())), Some(0.5));
        // 1 + α is never lower: the search must give up, not accept the first trial.
        assert_eq!(line_search(1.0, |a| Some(1.0 + a)), None);
    }

    #[test]
    fn backtrack_refuses_an_equal_residual_and_a_refused_point() {
        // strictly lower is required: an unchanged residual is no progress
        assert_eq!(line_search(1.0, |_| Some(1.0)), None);
        // a point the residual refuses (an inverted element) is skipped, and the
        // search continues with a shorter step
        assert_eq!(
            line_search(1.0, |a| if a > 0.3 { None } else { Some(0.5) }),
            Some(0.25)
        );
    }

    #[test]
    fn backtrack_only_moves_the_free_rows() {
        let moved = backtrack(
            fx(1.0),
            &[fx(0.0), fx(5.0)],
            &[fx(1.0), fx(1.0)],
            &[true, false],
            |_| Some(fx(0.0)),
        )
        .expect("accepted");
        assert_eq!((moved[0].to_f64(), moved[1].to_f64()), (1.0, 5.0));
    }

    #[test]
    fn backtrack_gives_up_after_sixteen_halvings() {
        let mut trials = 0u32;
        let out = backtrack(fx(1.0), &v(0.0), &v(1.0), &[true], |_| {
            trials += 1;
            Some(fx(2.0))
        });
        assert!(out.is_none());
        assert_eq!(
            trials,
            BACKTRACKS + 1,
            "the first trial and then one per halving"
        );
    }

    // ---- newton_step: the surrogate fallback ------------------------------

    #[test]
    fn newton_step_falls_back_to_the_surrogate_when_the_first_direction_has_negative_curvature() {
        // One tet, nodes 0..2 held, node 3 free, `u = 0`. A negative bulk modulus
        // is not a material, but it is a valid argument and puts the curvature
        // along the load direction at V (μ + K) = (3 − 50)/6 < 0, so the tangent
        // cannot start a conjugate gradient. The co-rotational linear surrogate is
        // positive definite, so the step must be computed with it: the report of
        // the (unavoidable, here) refusal then shows the surrogate's iterations.
        // Without the fallback the step is zero and that count is 0.
        let element = unit_tet();
        let bulk = fx(-50.0);
        let mut u = vec![Fix128::ZERO; 12];
        let mut f_ext = vec![Fix128::ZERO; 12];
        f_ext[11] = Fix128::ONE; // node 3, z
        let is_free = [
            false, false, false, false, false, false, false, false, false, true, true, true,
        ];
        let rotations = [Mat3Fix::IDENTITY];
        let elements = [element_clone(&element)];
        let assembly = Assembly {
            elements: &elements,
            rotations: &rotations,
            lame: (fx(2.0), fx(3.0)),
            is_free: &is_free,
        };
        let law = (model(), bulk);
        let mut residual = vec![Fix128::ZERO; 12];
        corotational_residual(&assembly, &u, &f_ext, Some(law), &mut residual).expect("regular");
        // the premise: the first direction (the residual itself) bends the wrong way
        let tangent = TangentField::at(&elements, &u, &model(), bulk).expect("regular");
        let mut kr = vec![Fix128::ZERO; 12];
        tangent.apply(&elements, &residual, &mut kr);
        assert!(
            dot(&residual, &kr) < Fix128::ZERO,
            "the premise of the test is a negative curvature"
        );
        let outcome = newton_step(
            &assembly,
            &mut u,
            &f_ext,
            law,
            &cg_config(1000),
            &mut residual,
        );
        match outcome {
            Err(FemError::NotConverged { iterations, .. }) => {
                assert!(
                    iterations > 0,
                    "the surrogate must have been solved, but no iteration ran"
                )
            }
            other => panic!("a law that is unstable along the load cannot be reduced: {other:?}"),
        }
    }
}
