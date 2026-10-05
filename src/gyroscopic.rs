//! Implicit gyroscopic term `ω × Iω` for free 3D rigid bodies.
//!
//! A torque-free body obeys `I ω̇ + ω × Iω = 0` in its principal frame. Without
//! the second term the integrators keep `ω` constant in the world frame, so an
//! asymmetric body neither precesses nor shows the intermediate-axis flip, and
//! its world angular momentum `R I Rᵀ ω` is not conserved.
//!
//! The term is integrated implicitly, following E. Catto, "Physics for Game
//! Programmers: Numerical Methods", GDC 2015: in the body frame solve
//!
//! ```text
//! f(ω₂) = I (ω₂ − ω₁) + h ω₂ × (I ω₂) = 0
//! ```
//!
//! with one Newton step from `ω₁`,
//!
//! ```text
//! J  = I + h (skew(ω₁) I − skew(I ω₁))
//! ω₂ = ω₁ − J⁻¹ f(ω₁),   f(ω₁) = h ω₁ × (I ω₁)
//! ```
//!
//! and rotate `ω₂` back to the world frame. The exact implicit solution never
//! gains kinetic energy (dot `f(ω₂) = 0` with `ω₂`: `ω₂ᵀIω₂ = ω₂ᵀIω₁`), which is
//! why the implicit form is used instead of the explicit `ω −= h I⁻¹(ω × Iω)`
//! that gains energy every step.

use crate::math::{Fix128, Mat3Fix, QuatFix, Vec3Fix};

/// Smallest accepted inverse principal moment, `2⁻⁴⁰` (a moment of about
/// `1.1e12 kg·m²`); a smaller one would overflow `1 / inv`.
const INV_INERTIA_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 24);

/// The normalised system is scaled until its largest entry lies in
/// `[2⁻⁸, 2⁸]`, so that the 3x3 determinant stays well inside the range.
const SCALE_HI: Fix128 = Fix128::from_raw(1 << 8, 0);
const SCALE_LO: Fix128 = Fix128::from_raw(0, 1 << 56);

/// Largest accepted right-hand side after normalisation, `2⁴⁰`.
const RHS_LIMIT: Fix128 = Fix128::from_raw(1 << 40, 0);

/// Largest accepted `|Δω_k|` from Cramer's rule, `2⁵⁰` rad/s: a larger quotient
/// means a nearly singular Newton matrix and the step is skipped.
const QUOTIENT_LIMIT: Fix128 = Fix128::from_raw(1 << 50, 0);

/// World angular velocity after one implicit gyroscopic step of length `h`.
///
/// `omega` is the world angular velocity, `rotation` the body orientation and
/// `inv_inertia` the diagonal inverse inertia in the body frame.
///
/// # Claims
///
/// - Returns `omega` unchanged, bit for bit, when the inertia is isotropic
///   (`inv_inertia.x == inv_inertia.y == inv_inertia.z`, for which
///   `ω × Iω = 0`), when `omega` is zero, and when any inverse moment is zero
///   or negative (an infinite moment has no gyroscopic response; static and
///   infinite-inertia bodies fall here).
/// - Returns `omega` unchanged when an intermediate product would leave the
///   fixed-point range (an inverse moment below `2⁻⁴⁰`, or `|ω|` so large that
///   `h |ω|² I` does not fit) or when the Newton matrix is singular: the step
///   is skipped rather than producing a wrapped value.
/// - Accuracy is first order in `h |ω|`. The exact implicit solution never
///   gains energy, but the single Newton step can: measured in f64 on valid
///   inertia tensors (moment ratios up to about 10³), the worst per-step gain
///   is `1e-10` at `h |ω| = 0.01`, `2e-5` at `0.1` and above `1` at `h |ω| = 1`.
///   At large `h |ω|` the step instead dissipates strongly, which is the
///   behaviour of the backward-Euler step it approximates.
/// - Otherwise returns `R (ω₁ − J⁻¹ f(ω₁))` with `ω₁ = Rᵀ omega`, see the
///   module documentation.
#[must_use]
pub(crate) fn gyroscopic_omega(
    omega: Vec3Fix,
    rotation: QuatFix,
    inv_inertia: Vec3Fix,
    h: Fix128,
) -> Vec3Fix {
    let inv = inv_inertia;
    let isotropic = inv.x == inv.y && inv.y == inv.z;
    let at_rest = omega.x.is_zero() && omega.y.is_zero() && omega.z.is_zero();
    if isotropic || at_rest {
        return omega;
    }
    if inv.x < INV_INERTIA_FLOOR || inv.y < INV_INERTIA_FLOOR || inv.z < INV_INERTIA_FLOOR {
        return omega;
    }
    let w_body = rotation.conjugate().rotate_vec(omega);
    match implicit_body_step(w_body, inv, h) {
        Some(w2) => rotation.rotate_vec(w2),
        None => omega,
    }
}

/// One Newton step of the body-frame implicit equation. `None` means "leave
/// `ω` as it is" (out of range or singular).
fn implicit_body_step(w: Vec3Fix, inv: Vec3Fix, h: Fix128) -> Option<Vec3Fix> {
    let i = Vec3Fix::new(
        Fix128::ONE / inv.x,
        Fix128::ONE / inv.y,
        Fix128::ONE / inv.z,
    );
    // Every sum below is a difference of two products with the same sign
    // structure (`h ω_j ω_k I_k − h ω_k ω_j I_j`), so it cannot wrap once both
    // products are in range.
    let m = |a: Fix128, b: Fix128| a.checked_mul(b);
    // I ω, h ω, h I ω
    let l = Vec3Fix::new(m(i.x, w.x)?, m(i.y, w.y)?, m(i.z, w.z)?);
    let hw = Vec3Fix::new(m(h, w.x)?, m(h, w.y)?, m(h, w.z)?);
    let hl = Vec3Fix::new(m(h, l.x)?, m(h, l.y)?, m(h, l.z)?);

    // f = h ω × I ω
    let f = Vec3Fix::new(
        m(hw.y, l.z)? - m(hw.z, l.y)?,
        m(hw.z, l.x)? - m(hw.x, l.z)?,
        m(hw.x, l.y)? - m(hw.y, l.x)?,
    );

    // J = I + skew(h ω) I − skew(h I ω), with skew(a) the matrix of `a ×`.
    // Row-major entries (r, c).
    let j00 = i.x;
    let j01 = hl.z - m(hw.z, i.y)?;
    let j02 = m(hw.y, i.z)? - hl.y;
    let j10 = m(hw.z, i.x)? - hl.z;
    let j11 = i.y;
    let j12 = hl.x - m(hw.x, i.z)?;
    let j20 = hl.y - m(hw.y, i.x)?;
    let j21 = m(hw.x, i.y)? - hl.x;
    let j22 = i.z;
    let mut rows = [[j00, j01, j02], [j10, j11, j12], [j20, j21, j22]];
    let mut rhs = [-f.x, -f.y, -f.z];

    // Normalise by powers of two (exact up to dropped low bits) so the
    // determinant stays in range; the solution is unchanged by the common
    // factor.
    let largest =
        rows.iter().flatten().fold(
            Fix128::ZERO,
            |acc, v| if v.abs() > acc { v.abs() } else { acc },
        );
    let mut s = largest;
    let mut steps = 0;
    while s > SCALE_HI && steps < 16 {
        s = s.shr_bits(8);
        for v in rows.iter_mut().flatten() {
            *v = v.shr_bits(8);
        }
        for v in &mut rhs {
            *v = v.shr_bits(8);
        }
        steps += 1;
    }
    let up = Fix128::from_int(256);
    while s < SCALE_LO && steps < 16 {
        s = s * up;
        for v in rows.iter_mut().flatten() {
            *v = *v * up;
        }
        for v in &mut rhs {
            *v = v.checked_mul(up)?;
        }
        steps += 1;
    }
    if rhs.iter().any(|v| v.abs() > RHS_LIMIT) {
        return None;
    }

    // Cramer's rule: Δω_k = det(J with column k replaced by rhs) / det(J).
    let col = |c: usize| Vec3Fix::new(rows[0][c], rows[1][c], rows[2][c]);
    let b = Vec3Fix::new(rhs[0], rhs[1], rhs[2]);
    let det = Mat3Fix::from_cols(col(0), col(1), col(2)).determinant();
    if det.is_zero() {
        return None;
    }
    let cap = det.abs().checked_mul(QUOTIENT_LIMIT);
    let solve = |num: Fix128| -> Option<Fix128> {
        if let Some(cap) = cap {
            if num.abs() >= cap {
                return None;
            }
        }
        Some(num / det)
    };
    let dx = solve(Mat3Fix::from_cols(b, col(1), col(2)).determinant())?;
    let dy = solve(Mat3Fix::from_cols(col(0), b, col(2)).determinant())?;
    let dz = solve(Mat3Fix::from_cols(col(0), col(1), b).determinant())?;
    Some(Vec3Fix::new(w.x + dx, w.y + dy, w.z + dz))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    #[test]
    fn isotropic_inertia_returns_input_bits() {
        let w = Vec3Fix::new(r(1, 3), r(-2, 7), r(5, 11));
        let q = QuatFix::from_axis_angle(Vec3Fix::new(r(3, 5), r(4, 5), Fix128::ZERO), r(1, 2));
        let inv = Vec3Fix::new(r(1, 4), r(1, 4), r(1, 4));
        assert_eq!(gyroscopic_omega(w, q, inv, r(1, 480)), w);
    }

    #[test]
    fn zero_inverse_moment_returns_input_bits() {
        let w = Vec3Fix::new(r(1, 3), r(-2, 7), r(5, 11));
        let inv = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, r(1, 2));
        assert_eq!(gyroscopic_omega(w, QuatFix::IDENTITY, inv, r(1, 480)), w);
    }

    #[test]
    fn principal_axis_spin_returns_input_bits() {
        let w = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(3), Fix128::ZERO);
        let inv = Vec3Fix::new(Fix128::ONE, r(1, 2), r(1, 3));
        assert_eq!(gyroscopic_omega(w, QuatFix::IDENTITY, inv, r(1, 480)), w);
    }

    /// oracle: one step of Euler's equations for `I = (1, 2, 3)`,
    /// `ω = (1, 1, 1)`: `ω × Iω = (1, −2, 1)`, so `ω₂ ≈ ω − h I⁻¹(1, −2, 1)
    /// = (1 − h, 1 + h, 1 − h/3)` to first order in `h`.
    #[test]
    fn one_step_matches_euler_equations_to_first_order() {
        let h = r(1, 1000);
        let w = Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE);
        let inv = Vec3Fix::new(Fix128::ONE, r(1, 2), r(1, 3));
        let got = gyroscopic_omega(w, QuatFix::IDENTITY, inv, h);
        let hf = 1.0e-3;
        let want = [1.0 - hf, 1.0 + hf, 1.0 - hf / 3.0];
        for (g, e) in [got.x, got.y, got.z].iter().zip(want) {
            assert!((g.to_f64() - e).abs() < 1e-5, "{} vs {e}", g.to_f64());
        }
    }
}
