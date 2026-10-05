//! Gyroscopic term `ω × Iω` for free 3D rigid bodies.
//!
//! A torque-free body obeys `I ω̇ + ω × Iω = 0` in its principal frame. Without
//! the second term the integrators keep `ω` constant in the world frame, so an
//! asymmetric body neither precesses nor shows the intermediate-axis flip, and
//! its world angular momentum `R I Rᵀ ω` is not conserved.
//!
//! [`split_free_rotation`] integrates the free rigid body by the splitting of
//! Dullweber, Leimkuhler and McLachlan, "Symplectic splitting methods for
//! rigid body molecular dynamics", J. Chem. Phys. 107 (1997) 5840 (see also
//! Reich 1994 and McLachlan 1993). With the body angular momentum
//! `L_b = I ω_b`, `H = Σ L_k² / (2 I_k)` splits into `H_k = L_k² / (2 I_k)`;
//! the flow of `H_k` is exact: the body turns about its principal axis `k` by
//! `θ_k = h L_k / I_k` and `L_b` turns about the same axis by `−θ_k`, so
//! `L_k`, `|L_b|` and the world momentum `R L_b` are unchanged. The symmetric
//! (Strang) composition `R₁(h/2) R₂(h/2) R₃(h) R₂(h/2) R₁(h/2)` is second
//! order and time-reversible. Each sub-flow conserves its own `H_k` but not
//! the others (it mixes two components with different moments), so the energy
//! is not conserved exactly: it oscillates with an amplitude of order `h²`
//! and does not drift.
//!
//! Both 3D integrators use it: the XPBD step predicts the rotation with the
//! split and keeps the split's end velocity (plus the rotation the position
//! solve added), the TGS backend does the same per sub-step.

use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Smallest accepted inverse principal moment, `2⁻⁴⁰` (a moment of about
/// `1.1e12 kg·m²`); a smaller one would overflow `1 / inv`.
const INV_INERTIA_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 24);

/// Orientation and world angular velocity of a free body after one step of
/// the symplectic splitting, see [`split_free_rotation`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FreeRotation {
    /// Orientation at the end of the step (unit quaternion).
    pub(crate) rotation: QuatFix,
    /// World angular velocity at the end of the step, `R_end I⁻¹ L_b,end`.
    pub(crate) omega: Vec3Fix,
}

/// Principal axes `(a, b)` that a rotation about axis `k` mixes, in the
/// cyclic order that makes `(a, b, k)` right-handed.
const MIXED_AXES: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];

/// Free-body step of length `h` by the Strang splitting of Dullweber,
/// Leimkuhler and McLachlan (1997), see the module documentation.
///
/// `omega` is the world angular velocity, `rotation` the body orientation and
/// `inv_inertia` the diagonal inverse inertia in the body frame (the crate
/// stores inertia as principal moments, so the body frame is the principal
/// frame).
///
/// # Claims
///
/// - Returns `None` (the caller keeps its gyroscope-free path, bit for bit)
///   for isotropic inertia, zero `omega`, an inverse moment below `2⁻⁴⁰`
///   (zero, negative, infinite moments: static and infinite-inertia bodies),
///   and when `I ω` or a sub-rotation angle leaves the fixed-point range.
/// - Otherwise the world angular momentum `R I Rᵀ ω` and `|L_b|` are
///   conserved up to fixed-point rounding (each sub-rotation is orthogonal to
///   rounding: its sine and cosine are rebuilt from a renormalised half-angle
///   pair), and the energy error is of order `h²` with no drift.
#[must_use]
pub(crate) fn split_free_rotation(
    omega: Vec3Fix,
    rotation: QuatFix,
    inv_inertia: Vec3Fix,
    h: Fix128,
) -> Option<FreeRotation> {
    let inv = inv_inertia;
    let isotropic = inv.x == inv.y && inv.y == inv.z;
    let at_rest = omega.x.is_zero() && omega.y.is_zero() && omega.z.is_zero();
    if isotropic || at_rest {
        return None;
    }
    if inv.x < INV_INERTIA_FLOOR || inv.y < INV_INERTIA_FLOOR || inv.z < INV_INERTIA_FLOOR {
        return None;
    }
    let inv = [inv.x, inv.y, inv.z];
    let w = rotation.conjugate().rotate_vec(omega);
    let mut l = [
        (Fix128::ONE / inv[0]).checked_mul(w.x)?,
        (Fix128::ONE / inv[1]).checked_mul(w.y)?,
        (Fix128::ONE / inv[2]).checked_mul(w.z)?,
    ];
    let half_h = h.half();
    let mut q_inc = QuatFix::IDENTITY;
    for (k, step) in [(0, half_h), (1, half_h), (2, h), (1, half_h), (0, half_h)] {
        // θ = step · L_k / I_k; the sub-flow keeps L_k, so ω_k is current.
        let theta = step.checked_mul(l[k].checked_mul(inv[k])?)?;
        if theta.is_zero() {
            // identity; CORDIC would leave a residue of order 2⁻⁴⁸ instead
            continue;
        }
        let (s2, c2) = theta.half().sin_cos();
        let n = (s2 * s2 + c2 * c2).sqrt();
        if n.is_zero() {
            return None;
        }
        let (s2, c2) = (s2 / n, c2 / n);
        let s = (s2 * c2).double();
        let c = c2 * c2 - s2 * s2;
        // L_b turns by −θ about axis k ...
        let (a, b) = MIXED_AXES[k];
        let (la, lb) = (l[a], l[b]);
        l[a] = c * la + s * lb;
        l[b] = c * lb - s * la;
        // ... while the body turns by +θ about the same body axis.
        let mut axis = [Fix128::ZERO; 3];
        axis[k] = s2;
        q_inc = q_inc.mul(QuatFix::new(axis[0], axis[1], axis[2], c2));
    }
    let end = rotation.mul(q_inc).normalize();
    let w_end = Vec3Fix::new(l[0] * inv[0], l[1] * inv[1], l[2] * inv[2]);
    Some(FreeRotation {
        rotation: end,
        omega: end.rotate_vec(w_end),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64, d: i64) -> Fix128 {
        Fix128::from_ratio(n, d)
    }

    #[test]
    fn split_exempts_isotropic_zero_and_infinite_moments() {
        let w = Vec3Fix::new(r(1, 3), r(-2, 7), r(5, 11));
        let q = QuatFix::from_axis_angle(Vec3Fix::new(r(3, 5), r(4, 5), Fix128::ZERO), r(1, 2));
        let h = r(1, 480);
        let iso = Vec3Fix::new(r(1, 4), r(1, 4), r(1, 4));
        assert_eq!(split_free_rotation(w, q, iso, h), None);
        let inf = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, r(1, 2));
        assert_eq!(split_free_rotation(w, q, inf, h), None);
        let aniso = Vec3Fix::new(Fix128::ONE, r(1, 2), r(1, 3));
        assert_eq!(split_free_rotation(Vec3Fix::ZERO, q, aniso, h), None);
    }

    /// oracle: about a principal axis only the sub-flow of that axis turns
    /// anything, by `θ = h ω`; the momentum and the spin axis are unchanged.
    #[test]
    fn split_principal_axis_spin_turns_by_h_omega() {
        let w = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(3), Fix128::ZERO);
        let inv = Vec3Fix::new(Fix128::ONE, r(1, 2), r(1, 3));
        let h = r(1, 64);
        let got = split_free_rotation(w, QuatFix::IDENTITY, inv, h).unwrap();
        let half: f64 = 3.0 / 128.0;
        let turned = crate::det_math::atan2_64(got.rotation.y.to_f64(), got.rotation.w.to_f64());
        assert!((turned - half).abs() < 1e-13, "{turned} vs {half}");
        assert!(got.rotation.x.is_zero() && got.rotation.z.is_zero());
        assert!(got.omega.x.to_f64().abs() < 1e-15 && got.omega.z.to_f64().abs() < 1e-15);
        assert!((got.omega.y.to_f64() - 3.0).abs() < 1e-15);
    }
}
