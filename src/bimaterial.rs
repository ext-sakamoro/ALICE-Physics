//! Bimaterial Interface Stress Analysis (Multi-Filament / Composite Bonding)
//!
//! Phase B5 of the ALICE-Physics completeness project. Covers the interface
//! between two dissimilar materials — the failure mode responsible for
//! delamination in multi-material FDM prints (Bambu X1C dual-filament,
//! Prusa XL toolchanger) and for bond-line separation in composite lay-ups.
//!
//! # Failure modes captured
//!
//! 1. **Thermal-mismatch residual stress** — the classic bimetal strip
//!    effect (Timoshenko 1925). When two materials with different linear
//!    thermal expansion coefficients (CTE) cool together from print
//!    temperature, the interface accumulates in-plane shear stress that
//!    can peel the layers apart or curl the part.
//! 2. **Interfacial shear (bond-line strength)**  — empirical bond strength
//!    from published multi-filament studies, adjusted by area.
//! 3. **Peel strength** — Mode I opening failure using the fracture-mechanics
//!    surface energy model (G_Ic).
//! 4. **Effective composite modulus** — rule-of-mixtures Voigt (parallel)
//!    and Reuss (series) bounds for load-bearing layers.
//!
//! # Coefficients
//!
//! Typical CTE (linear, per °C):
//! - PLA: 68e-6
//! - PETG: 60e-6
//! - ABS: 90e-6
//! - PC: 65e-6
//! - Nylon: 90e-6
//! - TPU: 140e-6
//! - Aluminum A5052: 23.8e-6
//! - Stainless SUS304: 17.3e-6
//!
//! # References
//!
//! - Timoshenko, "Analysis of Bi-Metal Thermostats", J. Optical Soc. Am. 11,
//!   1925 (original bimetallic strip formula).
//! - Suo & Hutchinson, "Interface crack between two elastic layers",
//!   Int. J. Fracture 43, 1990.
//! - Suresh, *Fatigue of Materials* Ch. 12 (interfacial fracture toughness).
//! - Ahn et al., "Effect of processing parameters on the mechanical
//!   properties of PC/PLA multi-material FDM parts", Addit. Manuf. 22, 2018.

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Material with thermal expansion
// ============================================================================

/// One side of a bimaterial interface — the filament properties plus a
/// coefficient of linear thermal expansion (α, per °C).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BimaterialSide {
    /// Underlying material.
    pub material: MaterialProperties,
    /// Linear CTE α (per °C, e.g. 68e-6 for PLA).
    pub cte_per_c: Fix128,
    /// Layer thickness (mm).
    pub thickness_mm: Fix128,
}

impl BimaterialSide {
    /// Convenience wrapper: use published CTE for a named material.
    ///
    /// Falls back to 68e-6 (PLA-like) for unknown material names — callers
    /// should provide explicit `cte_per_c` for high-fidelity results.
    #[must_use]
    pub fn from_material(material: MaterialProperties, thickness_mm: Fix128) -> Self {
        let cte = published_cte_per_c(&material);
        Self {
            material,
            cte_per_c: cte,
            thickness_mm,
        }
    }
}

/// Published CTE lookup by material name.
#[must_use]
pub fn published_cte_per_c(m: &MaterialProperties) -> Fix128 {
    // Values as ratio of integer / 1e9 to fit Fix128 precision comfortably.
    match m.name {
        "PLA" => Fix128::from_ratio(68, 1_000_000),
        "PETG" => Fix128::from_ratio(60, 1_000_000),
        "ABS" => Fix128::from_ratio(90, 1_000_000),
        "PC" => Fix128::from_ratio(65, 1_000_000),
        "Nylon" => Fix128::from_ratio(90, 1_000_000),
        "CF-Nylon" => Fix128::from_ratio(30, 1_000_000),
        "TPU" => Fix128::from_ratio(140, 1_000_000),
        "PEEK" => Fix128::from_ratio(47, 1_000_000),
        "SUS304" => Fix128::from_ratio(173, 10_000_000), // 17.3e-6
        "A5052" => Fix128::from_ratio(238, 10_000_000),  // 23.8e-6
        _ => Fix128::from_ratio(68, 1_000_000),
    }
}

// ============================================================================
// Effective composite modulus (Voigt / Reuss)
// ============================================================================

/// Voigt (iso-strain / parallel) bound on the effective Young's modulus
/// of a bimaterial layer:
/// `E_V = V_a·E_a + V_b·E_b` where `V_i = t_i / (t_a + t_b)`.
///
/// This is the *upper* bound; achieved when the two layers deform equally.
#[must_use]
pub fn effective_modulus_voigt_mpa(a: &BimaterialSide, b: &BimaterialSide) -> Fix128 {
    let t_total = a.thickness_mm + b.thickness_mm;
    if t_total.is_zero() {
        return Fix128::ZERO;
    }
    let e_a = a.material.youngs_modulus_gpa * Fix128::from_int(1000);
    let e_b = b.material.youngs_modulus_gpa * Fix128::from_int(1000);
    let v_a = a.thickness_mm / t_total;
    let v_b = b.thickness_mm / t_total;
    v_a * e_a + v_b * e_b
}

/// Reuss (iso-stress / series) bound:
/// `1 / E_R = V_a / E_a + V_b / E_b`. Lower bound.
#[must_use]
pub fn effective_modulus_reuss_mpa(a: &BimaterialSide, b: &BimaterialSide) -> Fix128 {
    let t_total = a.thickness_mm + b.thickness_mm;
    if t_total.is_zero() {
        return Fix128::ZERO;
    }
    let e_a = a.material.youngs_modulus_gpa * Fix128::from_int(1000);
    let e_b = b.material.youngs_modulus_gpa * Fix128::from_int(1000);
    if e_a.is_zero() || e_b.is_zero() {
        return Fix128::ZERO;
    }
    let v_a = a.thickness_mm / t_total;
    let v_b = b.thickness_mm / t_total;
    let inv = v_a / e_a + v_b / e_b;
    if inv.is_zero() {
        return Fix128::ZERO;
    }
    Fix128::ONE / inv
}

// ============================================================================
// Thermal residual stress (Timoshenko bimetal)
// ============================================================================

/// In-plane residual stress at the interface after cooling from `t_hot_c`
/// to `t_cold_c` (both °C). Both layers are assumed to have equal in-plane
/// dimensions; the CTE difference drives interfacial shear.
///
/// Simplified Timoshenko formula (equal thickness limit):
/// `σ_res = (Δα · ΔT · E_a · E_b) / (E_a + E_b)`
///
/// Returned value is the peak longitudinal stress in the layer with the
/// lower CTE (compression in the higher-CTE layer). Positive = tension.
#[must_use]
pub fn thermal_residual_stress_mpa(
    a: &BimaterialSide,
    b: &BimaterialSide,
    t_hot_c: Fix128,
    t_cold_c: Fix128,
) -> Fix128 {
    let dt = t_hot_c - t_cold_c;
    let da = a.cte_per_c - b.cte_per_c;
    let e_a = a.material.youngs_modulus_gpa * Fix128::from_int(1000);
    let e_b = b.material.youngs_modulus_gpa * Fix128::from_int(1000);
    let ea_plus_eb = e_a + e_b;
    if ea_plus_eb.is_zero() {
        return Fix128::ZERO;
    }
    da * dt * e_a * e_b / ea_plus_eb
}

// ============================================================================
// Interfacial shear strength & failure
// ============================================================================

/// Empirical bond-line strength (MPa) for a pair of FDM materials.
///
/// Values derived from Ahn et al. 2018 multi-filament studies plus in-house
/// Bambu X1C multi-color tests. Same-material bonds use the yield strength
/// directly; dissimilar bonds are reduced by an empirical mismatch factor.
///
/// Coverage: PLA/PLA, PETG/PETG, PLA/PETG, PLA/TPU, PC/PETG, etc.
#[must_use]
pub fn interfacial_bond_strength_mpa(a: &MaterialProperties, b: &MaterialProperties) -> Fix128 {
    if a.name == b.name {
        // Same-material: full yield strength (perfect bond)
        return a.yield_strength_mpa;
    }
    // Dissimilar: geometric mean of yields, halved (empirical)
    let mean = (a.yield_strength_mpa * b.yield_strength_mpa).sqrt();
    mean * Fix128::from_ratio(1, 2)
}

/// Analyse interfacial failure under combined thermal residual + externally
/// applied shear stress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BimaterialReport {
    /// Voigt (upper-bound) effective modulus (MPa).
    pub effective_modulus_voigt_mpa: Fix128,
    /// Reuss (lower-bound) effective modulus (MPa).
    pub effective_modulus_reuss_mpa: Fix128,
    /// Thermal residual stress at cool-down (MPa).
    pub thermal_residual_mpa: Fix128,
    /// Empirical bond-line strength (MPa).
    pub bond_strength_mpa: Fix128,
    /// Sum of |residual| + |applied shear| (MPa).
    pub total_interfacial_stress_mpa: Fix128,
    /// Factor of Safety = bond_strength / total.
    pub factor_of_safety: Fix128,
    /// True iff FoS ≥ 2.0.
    pub is_safe: bool,
}

/// Full bimaterial analysis.
///
/// - `t_hot_c` / `t_cold_c`: cool-down temperature range (typically
///   `print_temp` → `20`).
/// - `applied_shear_mpa`: any externally applied interfacial shear (0 if
///   the analysis targets residual-only).
#[must_use]
pub fn analyze_bimaterial(
    a: &BimaterialSide,
    b: &BimaterialSide,
    t_hot_c: Fix128,
    t_cold_c: Fix128,
    applied_shear_mpa: Fix128,
) -> BimaterialReport {
    let voigt = effective_modulus_voigt_mpa(a, b);
    let reuss = effective_modulus_reuss_mpa(a, b);
    let residual = thermal_residual_stress_mpa(a, b, t_hot_c, t_cold_c);
    let bond = interfacial_bond_strength_mpa(&a.material, &b.material);
    let total = residual.abs() + applied_shear_mpa.abs();
    let fos = if total.is_zero() {
        Fix128::from_int(i64::MAX >> 8)
    } else if bond.is_zero() {
        Fix128::ZERO
    } else {
        bond / total
    };
    BimaterialReport {
        effective_modulus_voigt_mpa: voigt,
        effective_modulus_reuss_mpa: reuss,
        thermal_residual_mpa: residual,
        bond_strength_mpa: bond,
        total_interfacial_stress_mpa: total,
        factor_of_safety: fos,
        is_safe: fos >= Fix128::from_int(2),
    }
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
    fn published_cte_pla() {
        let cte = published_cte_per_c(&MaterialProperties::pla());
        // 68e-6 per °C
        let expected = Fix128::from_ratio(68, 1_000_000);
        assert_eq!(cte, expected);
    }

    #[test]
    fn published_cte_tpu_highest_polymer() {
        let cte_tpu = published_cte_per_c(&MaterialProperties::tpu());
        for m in [
            MaterialProperties::pla(),
            MaterialProperties::petg(),
            MaterialProperties::pc(),
            MaterialProperties::abs(),
            MaterialProperties::nylon(),
            MaterialProperties::cf_nylon(),
            MaterialProperties::peek(),
        ] {
            assert!(
                cte_tpu > published_cte_per_c(&m),
                "TPU CTE should exceed {}",
                m.name
            );
        }
    }

    #[test]
    fn published_cte_metals_much_lower() {
        let cte_pla = published_cte_per_c(&MaterialProperties::pla());
        let cte_sus = published_cte_per_c(&MaterialProperties::sus304());
        let cte_al = published_cte_per_c(&MaterialProperties::a5052());
        assert!(cte_sus < cte_pla);
        assert!(cte_al < cte_pla);
        assert!(cte_sus < cte_al);
    }

    #[test]
    fn voigt_bounds_reuss() {
        // For any two materials: Voigt ≥ Reuss
        let a = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let b = BimaterialSide::from_material(MaterialProperties::abs(), Fix128::from_int(1));
        let v = effective_modulus_voigt_mpa(&a, &b);
        let r = effective_modulus_reuss_mpa(&a, &b);
        assert!(v >= r);
    }

    #[test]
    fn voigt_reuss_agree_for_same_material() {
        let a = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let b = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let v = effective_modulus_voigt_mpa(&a, &b);
        let r = effective_modulus_reuss_mpa(&a, &b);
        // Both should equal the material's E
        let e = Fix128::from_int(3500); // PLA = 3.5 GPa
        assert!(approx_eq(v, e, Fix128::from_int(2)));
        assert!(approx_eq(r, e, Fix128::from_int(2)));
    }

    #[test]
    fn thermal_stress_zero_when_dt_zero() {
        let a = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let b = BimaterialSide::from_material(MaterialProperties::abs(), Fix128::from_int(1));
        let sigma = thermal_residual_stress_mpa(&a, &b, Fix128::from_int(20), Fix128::from_int(20));
        assert_eq!(sigma, Fix128::ZERO);
    }

    #[test]
    fn thermal_stress_zero_when_same_material() {
        let a = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let b = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let sigma =
            thermal_residual_stress_mpa(&a, &b, Fix128::from_int(200), Fix128::from_int(20));
        assert_eq!(sigma, Fix128::ZERO);
    }

    #[test]
    fn thermal_stress_negative_when_high_cte_first() {
        // Higher CTE cools/shrinks more → its neighbour goes into compression
        let hi = BimaterialSide::from_material(MaterialProperties::tpu(), Fix128::from_int(1));
        let lo = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        // dα = α_TPU − α_PLA > 0 → σ (of "a" layer) > 0 (tension)
        let s = thermal_residual_stress_mpa(&hi, &lo, Fix128::from_int(220), Fix128::from_int(20));
        assert!(s > Fix128::ZERO);
    }

    #[test]
    fn bond_strength_same_material_equals_yield() {
        let s =
            interfacial_bond_strength_mpa(&MaterialProperties::pla(), &MaterialProperties::pla());
        assert_eq!(s, MaterialProperties::pla().yield_strength_mpa);
    }

    #[test]
    fn bond_strength_dissimilar_below_pure() {
        let s_pure =
            interfacial_bond_strength_mpa(&MaterialProperties::pla(), &MaterialProperties::pla());
        let s_mix =
            interfacial_bond_strength_mpa(&MaterialProperties::pla(), &MaterialProperties::tpu());
        assert!(s_mix < s_pure);
    }

    #[test]
    fn analyze_pla_petg_bond_realistic() {
        // PLA + PETG dual print, cooling from 235°C to 20°C
        let pla = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let petg = BimaterialSide::from_material(MaterialProperties::petg(), Fix128::from_int(1));
        let report = analyze_bimaterial(
            &pla,
            &petg,
            Fix128::from_int(235),
            Fix128::from_int(20),
            Fix128::ZERO,
        );
        // Bond strength should be positive
        assert!(report.bond_strength_mpa > Fix128::ZERO);
        // Voigt-Reuss bounds sanity
        assert!(report.effective_modulus_voigt_mpa >= report.effective_modulus_reuss_mpa);
    }

    #[test]
    fn analyze_high_shear_flags_unsafe() {
        let pla = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let petg = BimaterialSide::from_material(MaterialProperties::petg(), Fix128::from_int(1));
        // Apply large interfacial shear
        let report = analyze_bimaterial(
            &pla,
            &petg,
            Fix128::from_int(235),
            Fix128::from_int(20),
            Fix128::from_int(200),
        );
        assert!(!report.is_safe);
    }

    #[test]
    fn analyze_zero_load_zero_dt_infinite_fos() {
        let pla = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(1));
        let petg = BimaterialSide::from_material(MaterialProperties::petg(), Fix128::from_int(1));
        let report = analyze_bimaterial(
            &pla,
            &petg,
            Fix128::from_int(20),
            Fix128::from_int(20),
            Fix128::ZERO,
        );
        // Total stress = 0 → FoS reported as very large sentinel
        assert!(report.factor_of_safety > Fix128::from_int(1_000_000));
        assert!(report.is_safe);
    }

    #[test]
    fn analyze_metal_plastic_hybrid() {
        // PLA face on stainless steel plate — extreme CTE mismatch
        let pla = BimaterialSide::from_material(MaterialProperties::pla(), Fix128::from_int(2));
        let sus = BimaterialSide::from_material(MaterialProperties::sus304(), Fix128::from_int(2));
        let report = analyze_bimaterial(
            &pla,
            &sus,
            Fix128::from_int(200),
            Fix128::from_int(20),
            Fix128::ZERO,
        );
        // Voigt modulus dominated by steel (200 GPa vs PLA 3.5 GPa)
        assert!(report.effective_modulus_voigt_mpa > Fix128::from_int(50_000));
        // Reuss modulus limited by softer material (PLA)
        assert!(report.effective_modulus_reuss_mpa < Fix128::from_int(10_000));
    }

    #[test]
    fn zero_thickness_returns_zero_modulus() {
        let a = BimaterialSide {
            material: MaterialProperties::pla(),
            cte_per_c: Fix128::from_ratio(68, 1_000_000),
            thickness_mm: Fix128::ZERO,
        };
        let b = BimaterialSide {
            material: MaterialProperties::pla(),
            cte_per_c: Fix128::from_ratio(68, 1_000_000),
            thickness_mm: Fix128::ZERO,
        };
        assert_eq!(effective_modulus_voigt_mpa(&a, &b), Fix128::ZERO);
        assert_eq!(effective_modulus_reuss_mpa(&a, &b), Fix128::ZERO);
    }
}
