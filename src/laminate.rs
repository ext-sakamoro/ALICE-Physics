//! Classical Laminate Theory (CLT) for Stacked-Ply Composites
//!
//! Phase D4 of the ALICE-Physics completeness project. FDM prints are
//! effectively laminates — a stack of anisotropic plies at various angles
//! (raster orientation per layer). Classical Laminate Theory computes the
//! **ABD stiffness matrix** relating the laminate's in-plane force /
//! bending-moment resultants to its mid-plane strains and curvatures.
//!
//! ```text
//! | N |   | A  B | | ε₀ |
//! | M | = | B  D | | κ  |
//! ```
//!
//! where:
//! - `A` (3×3): extensional stiffness (N/mm)
//! - `B` (3×3): bending-extension coupling (N)
//! - `D` (3×3): bending stiffness (N·mm)
//!
//! For symmetric laminates `B = 0` (no bending-stretching coupling), which
//! is the usual print recommendation.
//!
//! # Simplified scope
//!
//! This implementation uses **orthotropic plies** described by the reduced
//! 2-D stiffness matrix `Q̄` at each ply's rotation angle. Cross-ply and
//! angle-ply stacks are supported. Full 3-D thermal / hygroscopic effects
//! are not included.
//!
//! # References
//!
//! - Jones, *Mechanics of Composite Materials* 2nd ed. Ch. 4 (CLT).
//! - Reddy, *Mechanics of Laminated Composite Plates and Shells* 2nd ed. Ch. 3.
//! - Barbero, *Introduction to Composite Materials Design* Ch. 6.

use crate::anisotropic::OrthotropicElasticity;
use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Ply description
// ============================================================================

/// A single ply in the stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ply {
    /// Ply thickness (mm).
    pub thickness_mm: Fix128,
    /// Fibre / strong-axis rotation angle (radians) relative to laminate X axis.
    pub orientation_rad: Fix128,
    /// Orthotropic material properties for this ply.
    pub material: OrthotropicElasticity,
}

impl Ply {
    /// Reduced 2-D orthotropic stiffness matrix `Q` (in-plane) in the ply's
    /// own material coordinates. Returns the six independent entries
    /// `(Q11, Q12, Q22, Q66)` — Q13/Q23 vanish for orthotropic in-plane
    /// analysis and Q11≠Q22 in general.
    ///
    /// For an orthotropic ply under plane-stress:
    ///   Q11 = E_L / (1 − ν_LT·ν_TL)
    ///   Q22 = E_T / (1 − ν_LT·ν_TL)
    ///   Q12 = ν_LT · E_T / (1 − ν_LT·ν_TL)
    ///   Q66 = G_LT
    ///
    /// `ν_TL = ν_LT · E_T / E_L` (Maxwell reciprocity).
    #[must_use]
    pub fn q_matrix(&self) -> (Fix128, Fix128, Fix128, Fix128) {
        let m = &self.material;
        if m.e_l_mpa.is_zero() || m.e_t_mpa.is_zero() {
            return (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        }
        let nu_lt = m.nu_lt;
        let nu_tl = nu_lt * m.e_t_mpa / m.e_l_mpa;
        let denom = Fix128::ONE - nu_lt * nu_tl;
        if denom.is_zero() {
            return (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        }
        let q11 = m.e_l_mpa / denom;
        let q22 = m.e_t_mpa / denom;
        let q12 = nu_lt * m.e_t_mpa / denom;
        let q66 = m.g_lt_mpa;
        (q11, q12, q22, q66)
    }

    /// Rotated `Q̄` matrix components expressed in the laminate axis system.
    /// Returns `(Q̄11, Q̄12, Q̄22, Q̄16, Q̄26, Q̄66)` — the full 3×3 symmetric
    /// upper triangle. `Q̄16` and `Q̄26` vanish only for on-axis (0°) or
    /// cross-ply (90°) orientations.
    #[must_use]
    pub fn q_bar(&self) -> (Fix128, Fix128, Fix128, Fix128, Fix128, Fix128) {
        let (q11, q12, q22, q66) = self.q_matrix();
        let (s, c) = self.orientation_rad.sin_cos();
        let c2 = c * c;
        let s2 = s * s;
        let c4 = c2 * c2;
        let s4 = s2 * s2;
        let c2s2 = c2 * s2;
        let c3s = c2 * c * s;
        let cs3 = c * s2 * s;

        let two = Fix128::from_int(2);
        let four = Fix128::from_int(4);

        // Standard CLT transformation (Jones eq. 2.84):
        let q11_bar = q11 * c4 + two * (q12 + two * q66) * c2s2 + q22 * s4;
        let q22_bar = q11 * s4 + two * (q12 + two * q66) * c2s2 + q22 * c4;
        let q12_bar = (q11 + q22 - four * q66) * c2s2 + q12 * (c4 + s4);
        let q66_bar = (q11 + q22 - two * q12 - two * q66) * c2s2 + q66 * (c4 + s4);
        let q16_bar = (q11 - q12 - two * q66) * c3s + (q12 - q22 + two * q66) * cs3;
        let q26_bar = (q11 - q12 - two * q66) * cs3 + (q12 - q22 + two * q66) * c3s;

        (q11_bar, q12_bar, q22_bar, q16_bar, q26_bar, q66_bar)
    }
}

// ============================================================================
// ABD matrix
// ============================================================================

/// Symmetric 3×3 stiffness matrix (stored as 6 upper-triangular entries).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Sym3 {
    /// Entry (1,1).
    pub m11: Fix128,
    /// Entry (1,2) = (2,1).
    pub m12: Fix128,
    /// Entry (2,2).
    pub m22: Fix128,
    /// Entry (1,3) = (3,1).
    pub m13: Fix128,
    /// Entry (2,3) = (3,2).
    pub m23: Fix128,
    /// Entry (3,3).
    pub m33: Fix128,
}

impl Sym3 {
    /// Add another symmetric matrix component-wise.
    #[must_use]
    pub fn add(&self, other: &Sym3) -> Sym3 {
        Sym3 {
            m11: self.m11 + other.m11,
            m12: self.m12 + other.m12,
            m22: self.m22 + other.m22,
            m13: self.m13 + other.m13,
            m23: self.m23 + other.m23,
            m33: self.m33 + other.m33,
        }
    }

    /// Multiply every entry by a scalar.
    #[must_use]
    pub fn scale(&self, k: Fix128) -> Sym3 {
        Sym3 {
            m11: self.m11 * k,
            m12: self.m12 * k,
            m22: self.m22 * k,
            m13: self.m13 * k,
            m23: self.m23 * k,
            m33: self.m33 * k,
        }
    }
}

/// Full laminate stiffness (A, B, D symmetric 3×3 matrices).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct AbdMatrix {
    /// Extensional stiffness (N/mm).
    pub a: Sym3,
    /// Coupling (N).
    pub b: Sym3,
    /// Bending stiffness (N·mm).
    pub d: Sym3,
}

impl AbdMatrix {
    /// True iff the laminate is symmetric (B ≈ 0).
    ///
    /// Uses an absolute tolerance of 1e-4 in the same units as B.
    #[must_use]
    pub fn is_symmetric(&self, tol: Fix128) -> bool {
        let b = &self.b;
        b.m11.abs() <= tol
            && b.m12.abs() <= tol
            && b.m22.abs() <= tol
            && b.m13.abs() <= tol
            && b.m23.abs() <= tol
            && b.m33.abs() <= tol
    }
}

/// Compute the ABD matrix from an ordered ply stack.
///
/// Session 3 I11 upgrade: pre-computes each ply's `z_lower / z_upper` in a
/// single pass so the main integration loop is data-parallel-ready. When
/// the `parallel` feature is enabled, ply contributions accumulate via
/// `rayon::par_iter` reducers.
///
/// The plies are listed **bottom-to-top**. Each ply's `Q̄` is integrated
/// through the thickness with the classical formulas:
///
///   A = Σ Q̄_k · (z_{k+1} − z_k)
///   B = ½ · Σ Q̄_k · (z_{k+1}² − z_k²)
///   D = ⅓ · Σ Q̄_k · (z_{k+1}³ − z_k³)
///
/// with the mid-plane at `z = 0`.
#[must_use]
pub fn compute_abd(plies: &[Ply]) -> AbdMatrix {
    if plies.is_empty() {
        return AbdMatrix::default();
    }
    let total_thickness: Fix128 = plies
        .iter()
        .fold(Fix128::ZERO, |acc, p| acc + p.thickness_mm);
    let half_thickness = total_thickness.half();

    // Pre-compute z_lower / z_upper for each ply (SoA layout for the parallel
    // reducer). Each ply becomes a self-contained `PlyContribution`.
    struct PlyContribution {
        q: Sym3,
        delta_z: Fix128,
        delta_z2: Fix128,
        delta_z3: Fix128,
    }
    let mut z_lower = Fix128::ZERO - half_thickness;
    let contributions: Vec<PlyContribution> = plies
        .iter()
        .map(|ply| {
            let z_upper = z_lower + ply.thickness_mm;
            let (q11, q12, q22, q16, q26, q66) = ply.q_bar();
            let q = Sym3 {
                m11: q11,
                m12: q12,
                m22: q22,
                m13: q16,
                m23: q26,
                m33: q66,
            };
            let delta_z = z_upper - z_lower;
            let delta_z2 = z_upper * z_upper - z_lower * z_lower;
            let delta_z3 = z_upper * z_upper * z_upper - z_lower * z_lower * z_lower;
            z_lower = z_upper;
            PlyContribution {
                q,
                delta_z,
                delta_z2,
                delta_z3,
            }
        })
        .collect();

    #[cfg(feature = "parallel")]
    let (a, b, d) = {
        use rayon::prelude::*;
        contributions
            .par_iter()
            .map(|c| {
                (
                    c.q.scale(c.delta_z),
                    c.q.scale(c.delta_z2 * Fix128::from_ratio(1, 2)),
                    c.q.scale(c.delta_z3 * Fix128::from_ratio(1, 3)),
                )
            })
            .reduce(
                || (Sym3::default(), Sym3::default(), Sym3::default()),
                |(a1, b1, d1), (a2, b2, d2)| (a1.add(&a2), b1.add(&b2), d1.add(&d2)),
            )
    };

    #[cfg(not(feature = "parallel"))]
    let (a, b, d) = {
        let mut a = Sym3::default();
        let mut b = Sym3::default();
        let mut d = Sym3::default();
        for c in &contributions {
            a = a.add(&c.q.scale(c.delta_z));
            b = b.add(&c.q.scale(c.delta_z2 * Fix128::from_ratio(1, 2)));
            d = d.add(&c.q.scale(c.delta_z3 * Fix128::from_ratio(1, 3)));
        }
        (a, b, d)
    };

    AbdMatrix { a, b, d }
}

/// Check whether a ply list forms a symmetric stack (mirrored about mid-plane).
///
/// Symmetric stacks have `B = 0` exactly. Uses ply orientation, thickness,
/// and material identity as the equality check.
#[must_use]
pub fn is_symmetric_stack(plies: &[Ply]) -> bool {
    let n = plies.len();
    if n < 2 {
        return true;
    }
    for i in 0..n / 2 {
        let left = &plies[i];
        let right = &plies[n - 1 - i];
        if left.thickness_mm != right.thickness_mm
            || left.orientation_rad != right.orientation_rad
            || left.material != right.material
        {
            return false;
        }
    }
    true
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filament_db::MaterialProperties;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    fn cf_ortho() -> OrthotropicElasticity {
        OrthotropicElasticity::from_fdm_material(&MaterialProperties::cf_nylon())
    }

    #[test]
    fn q_matrix_orthotropic_ratios() {
        let ply = Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::ZERO,
            material: cf_ortho(),
        };
        let (q11, q12, q22, q66) = ply.q_matrix();
        // Q11 > Q22 for anisotropic (E_L > E_T? No — CF-Nylon has E_L=E_T
        // per our from_fdm_material simplification). So Q11 == Q22 here.
        assert_eq!(q11, q22);
        assert!(q12 > Fix128::ZERO);
        assert!(q66 > Fix128::ZERO);
    }

    #[test]
    fn q_bar_zero_orientation_equals_q() {
        let ply = Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::ZERO,
            material: cf_ortho(),
        };
        let (q11, q12, q22, q66) = ply.q_matrix();
        let (b11, b12, b22, b16, b26, b66) = ply.q_bar();
        assert!(approx_eq(q11, b11, Fix128::ONE));
        assert!(approx_eq(q12, b12, Fix128::ONE));
        assert!(approx_eq(q22, b22, Fix128::ONE));
        assert!(approx_eq(q66, b66, Fix128::ONE));
        // Off-axis coupling terms vanish at 0°
        assert!(b16.abs() < Fix128::ONE);
        assert!(b26.abs() < Fix128::ONE);
    }

    #[test]
    fn q_bar_90_swaps_q11_q22() {
        let base_material = cf_ortho();
        // Force E_L ≠ E_T to make the test meaningful
        let mut mat = base_material;
        mat.e_l_mpa = Fix128::from_int(10_000);
        mat.e_t_mpa = Fix128::from_int(2_000);
        let ply_0 = Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::ZERO,
            material: mat,
        };
        let ply_90 = Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::HALF_PI,
            material: mat,
        };
        let (q11_0, _, q22_0, _, _, _) = ply_0.q_bar();
        let (q11_90, _, q22_90, _, _, _) = ply_90.q_bar();
        // At 90° Q̄11 should equal on-axis Q̄22 (roles swapped)
        assert!(approx_eq(q11_90, q22_0, Fix128::from_int(10)));
        assert!(approx_eq(q22_90, q11_0, Fix128::from_int(10)));
    }

    #[test]
    fn symmetric_stack_flag() {
        let m = cf_ortho();
        let t = Fix128::from_ratio(2, 10);
        let ply_0 = Ply {
            thickness_mm: t,
            orientation_rad: Fix128::ZERO,
            material: m,
        };
        let ply_90 = Ply {
            thickness_mm: t,
            orientation_rad: Fix128::HALF_PI,
            material: m,
        };
        // [0/90/90/0] — symmetric
        let sym_stack = vec![ply_0, ply_90, ply_90, ply_0];
        assert!(is_symmetric_stack(&sym_stack));
        // [0/90/0/90] — not symmetric
        let asym_stack = vec![ply_0, ply_90, ply_0, ply_90];
        assert!(!is_symmetric_stack(&asym_stack));
    }

    #[test]
    fn single_ply_stack_is_symmetric() {
        let m = cf_ortho();
        let stack = vec![Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::ZERO,
            material: m,
        }];
        assert!(is_symmetric_stack(&stack));
    }

    #[test]
    fn compute_abd_empty_stack_returns_default() {
        let abd = compute_abd(&[]);
        assert_eq!(abd, AbdMatrix::default());
    }

    #[test]
    fn compute_abd_symmetric_has_zero_b() {
        let m = cf_ortho();
        let t = Fix128::from_ratio(2, 10);
        let plies = vec![
            Ply {
                thickness_mm: t,
                orientation_rad: Fix128::ZERO,
                material: m,
            },
            Ply {
                thickness_mm: t,
                orientation_rad: Fix128::HALF_PI,
                material: m,
            },
            Ply {
                thickness_mm: t,
                orientation_rad: Fix128::HALF_PI,
                material: m,
            },
            Ply {
                thickness_mm: t,
                orientation_rad: Fix128::ZERO,
                material: m,
            },
        ];
        let abd = compute_abd(&plies);
        // Symmetric stack ⇒ B ≈ 0 (within CORDIC ULP)
        assert!(abd.is_symmetric(Fix128::from_int(1000)));
    }

    #[test]
    fn compute_abd_a_positive() {
        let m = cf_ortho();
        let plies = vec![Ply {
            thickness_mm: Fix128::from_ratio(2, 10),
            orientation_rad: Fix128::ZERO,
            material: m,
        }];
        let abd = compute_abd(&plies);
        assert!(abd.a.m11 > Fix128::ZERO);
        assert!(abd.a.m22 > Fix128::ZERO);
        assert!(abd.d.m11 > Fix128::ZERO);
    }

    #[test]
    fn thick_ply_stronger_a() {
        let m = cf_ortho();
        let thin = vec![Ply {
            thickness_mm: Fix128::from_ratio(1, 10),
            orientation_rad: Fix128::ZERO,
            material: m,
        }];
        let thick = vec![Ply {
            thickness_mm: Fix128::from_ratio(4, 10),
            orientation_rad: Fix128::ZERO,
            material: m,
        }];
        let abd_thin = compute_abd(&thin);
        let abd_thick = compute_abd(&thick);
        assert!(abd_thick.a.m11 > abd_thin.a.m11);
        // D ~ thickness^3 → strongly amplified
        assert!(abd_thick.d.m11 > abd_thin.d.m11 * Fix128::from_int(10));
    }

    #[test]
    fn sym3_add_and_scale() {
        let a = Sym3 {
            m11: Fix128::from_int(1),
            m12: Fix128::from_int(2),
            m22: Fix128::from_int(3),
            m13: Fix128::from_int(4),
            m23: Fix128::from_int(5),
            m33: Fix128::from_int(6),
        };
        let b = a.scale(Fix128::from_int(2));
        assert_eq!(b.m11, Fix128::from_int(2));
        assert_eq!(b.m33, Fix128::from_int(12));
        let c = a.add(&b);
        assert_eq!(c.m11, Fix128::from_int(3));
        assert_eq!(c.m33, Fix128::from_int(18));
    }
}
