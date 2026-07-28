//! Prestressed / Pretensioned Joint & Cable Analysis
//!
//! Phase D5 of the ALICE-Physics completeness project. Covers the mechanics
//! of joints that carry a **pre-load** (bolts, cables, threaded fasteners) —
//! where the joint stiffness and safety depend on the interplay between the
//! initial tension and the externally applied working load.
//!
//! # Coverage
//!
//! - **Bolt preload**: nominal preload from proof strength and applied torque
//!   (Motosh formula), effective stiffness of the bolt/joint system.
//! - **Load-sharing under external tension**: fraction of the external force
//!   that reaches the bolt vs. that relieves the clamped joint.
//! - **Cable pretension**: axial force required to produce a target sag /
//!   frequency in a suspension cable.
//! - **Separation force**: external load at which the clamped joint separates.
//!
//! # References
//!
//! - Shigley, *Mechanical Engineering Design* 10th ed. Ch. 8 (bolt joints).
//! - Bickford, *Introduction to the Design and Behavior of Bolted Joints*
//!   4th ed. — the canonical treatment.
//! - VDI 2230 Part 1 (systematic calculation of bolted joints).

use crate::math::Fix128;

// ============================================================================
// Bolt preload
// ============================================================================

/// Compute nominal bolt preload from applied torque using Motosh formula:
///
/// `F_i = T / (K · d)`
///
/// where `K` is the nut factor (dimensionless, ~0.20 for dry steel-on-steel,
/// ~0.15 for lubricated, ~0.10 for waxed). Returns preload (N).
#[must_use]
pub fn preload_from_torque(torque_nm: Fix128, nut_factor_k: Fix128, diameter_mm: Fix128) -> Fix128 {
    if nut_factor_k.is_zero() || diameter_mm.is_zero() {
        return Fix128::ZERO;
    }
    // Convert diameter mm → m: F [N] = T [N·m] / (K · d [m])
    let d_m = diameter_mm / Fix128::from_int(1000);
    torque_nm / (nut_factor_k * d_m)
}

/// Recommended installation preload as a fraction of the bolt's proof
/// strength (Shigley §8-8 rule of thumb: `0.75 · S_p · A_t` for non-critical
/// static joints, `0.90` for permanent connections).
#[must_use]
pub fn recommended_preload_n(
    proof_strength_mpa: Fix128,
    tensile_stress_area_mm2: Fix128,
    fraction: Fix128,
) -> Fix128 {
    proof_strength_mpa * tensile_stress_area_mm2 * fraction
}

// ============================================================================
// Load-sharing under external tension
// ============================================================================

/// Fraction of the external tensile load that is carried by the bolt:
///
/// `C = k_b / (k_b + k_m)`
///
/// where `k_b` is the bolt stiffness and `k_m` the effective member
/// (clamped material) stiffness. Values are dimensionless [0, 1].
#[must_use]
pub fn bolt_load_fraction(k_bolt_n_per_mm: Fix128, k_member_n_per_mm: Fix128) -> Fix128 {
    let total = k_bolt_n_per_mm + k_member_n_per_mm;
    if total.is_zero() {
        return Fix128::ZERO;
    }
    k_bolt_n_per_mm / total
}

/// Peak tensile force in the bolt for a given external tensile load:
///
/// `F_b = F_i + C · P_ext`
///
/// with `F_i` = preload, `C` = bolt load fraction, `P_ext` = external load (N).
#[must_use]
pub fn bolt_peak_tension(preload_n: Fix128, bolt_fraction: Fix128, p_ext_n: Fix128) -> Fix128 {
    preload_n + bolt_fraction * p_ext_n
}

/// External tensile load at which the clamped joint separates:
///
/// `P_sep = F_i / (1 − C)`
///
/// Below `P_sep` the joint stays compressed (bolt fully preloaded plus a
/// small increment). Above, the members no longer contact — the bolt must
/// carry the full external load and cyclic fatigue rises sharply.
#[must_use]
pub fn separation_load_n(preload_n: Fix128, bolt_fraction: Fix128) -> Fix128 {
    let one_minus_c = Fix128::ONE - bolt_fraction;
    if one_minus_c.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    preload_n / one_minus_c
}

// ============================================================================
// Cable pretension
// ============================================================================

/// Axial force required to keep a horizontal cable with span `L` at
/// mid-span sag `s` under distributed weight `w` (N/mm):
///
/// `T ≈ w · L² / (8 · s)`
///
/// This is the parabolic-cable approximation (valid for `s ≪ L`).
#[must_use]
pub fn cable_pretension_n(w_n_per_mm: Fix128, span_mm: Fix128, sag_mm: Fix128) -> Fix128 {
    if sag_mm.is_zero() {
        return Fix128::from_int(i64::MAX >> 8);
    }
    let l2 = span_mm * span_mm;
    w_n_per_mm * l2 / (Fix128::from_int(8) * sag_mm)
}

/// Effective stiffness of a tensioned cable in the transverse direction
/// (small displacements). `k_transverse ≈ 8·T / L` for parabolic cable.
#[must_use]
pub fn tensioned_cable_stiffness_n_per_mm(tension_n: Fix128, span_mm: Fix128) -> Fix128 {
    if span_mm.is_zero() {
        return Fix128::ZERO;
    }
    Fix128::from_int(8) * tension_n / span_mm
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn preload_from_torque_zero_diameter_returns_zero() {
        let f = preload_from_torque(
            Fix128::from_int(10),
            Fix128::from_ratio(2, 10),
            Fix128::ZERO,
        );
        assert_eq!(f, Fix128::ZERO);
    }

    #[test]
    fn preload_from_torque_reference() {
        // T=10 N·m, K=0.2, d=8mm → F = 10 / (0.2 · 0.008) = 6250 N
        let f = preload_from_torque(
            Fix128::from_int(10),
            Fix128::from_ratio(2, 10),
            Fix128::from_int(8),
        );
        assert!(approx_eq(f, Fix128::from_int(6250), Fix128::ONE));
    }

    #[test]
    fn preload_scales_inversely_with_diameter() {
        let f_small = preload_from_torque(
            Fix128::from_int(10),
            Fix128::from_ratio(2, 10),
            Fix128::from_int(8),
        );
        let f_large = preload_from_torque(
            Fix128::from_int(10),
            Fix128::from_ratio(2, 10),
            Fix128::from_int(16),
        );
        // Doubling d → half F
        let ratio = f_small / f_large;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn recommended_preload_scales_with_all_inputs() {
        // S_p = 800 MPa, A_t = 20 mm², fraction 0.75 → F = 12000 N
        let f = recommended_preload_n(
            Fix128::from_int(800),
            Fix128::from_int(20),
            Fix128::from_ratio(75, 100),
        );
        assert!(approx_eq(f, Fix128::from_int(12_000), Fix128::from_int(1)));
    }

    #[test]
    fn bolt_fraction_at_extremes() {
        // Very stiff bolt / soft member → C ≈ 1
        let c_stiff_bolt = bolt_load_fraction(Fix128::from_int(1_000_000), Fix128::from_int(1));
        assert!(c_stiff_bolt > Fix128::from_ratio(99, 100));
        // Soft bolt / very stiff member → C ≈ 0
        let c_soft_bolt = bolt_load_fraction(Fix128::from_int(1), Fix128::from_int(1_000_000));
        assert!(c_soft_bolt < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn bolt_fraction_zero_total_returns_zero() {
        let c = bolt_load_fraction(Fix128::ZERO, Fix128::ZERO);
        assert_eq!(c, Fix128::ZERO);
    }

    #[test]
    fn bolt_peak_tension_scales_with_ext_load() {
        // Preload 5000N, C=0.3, external 1000N → 5000 + 300 = 5300
        let f = bolt_peak_tension(
            Fix128::from_int(5000),
            Fix128::from_ratio(3, 10),
            Fix128::from_int(1000),
        );
        assert!(approx_eq(
            f,
            Fix128::from_int(5300),
            Fix128::from_ratio(1, 10)
        ));
    }

    #[test]
    fn separation_load_grows_with_preload() {
        // C = 0.3 → separation = F_i / 0.7
        let sep1 = separation_load_n(Fix128::from_int(1000), Fix128::from_ratio(3, 10));
        let sep2 = separation_load_n(Fix128::from_int(2000), Fix128::from_ratio(3, 10));
        assert!(sep2 > sep1);
        assert!(approx_eq(
            sep2 / sep1,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn separation_load_infinite_when_c_equals_one() {
        // C = 1 → 1-C = 0 → sep = ∞
        let sep = separation_load_n(Fix128::from_int(1000), Fix128::ONE);
        assert!(sep > Fix128::from_int(1_000_000));
    }

    #[test]
    fn cable_pretension_scales_with_length_squared() {
        // w=1 N/mm, L=100mm, s=1mm → T = 1·10000/8 = 1250 N
        let t = cable_pretension_n(Fix128::ONE, Fix128::from_int(100), Fix128::ONE);
        assert!(approx_eq(t, Fix128::from_int(1250), Fix128::ONE));
        // Double L → 4x tension
        let t2 = cable_pretension_n(Fix128::ONE, Fix128::from_int(200), Fix128::ONE);
        assert!(approx_eq(
            t2 / t,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn cable_pretension_zero_sag_returns_sentinel() {
        let t = cable_pretension_n(Fix128::ONE, Fix128::from_int(100), Fix128::ZERO);
        assert!(t > Fix128::from_int(1_000_000));
    }

    #[test]
    fn tensioned_cable_stiffness_scales_with_tension() {
        let k = tensioned_cable_stiffness_n_per_mm(Fix128::from_int(1000), Fix128::from_int(100));
        // k = 8·1000/100 = 80
        assert!(approx_eq(
            k,
            Fix128::from_int(80),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn tensioned_cable_stiffness_zero_span_returns_zero() {
        let k = tensioned_cable_stiffness_n_per_mm(Fix128::from_int(1000), Fix128::ZERO);
        assert_eq!(k, Fix128::ZERO);
    }
}
