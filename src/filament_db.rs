//! 3D Printing Material & Sheet Metal Property Database
//!
//! Deterministic material property database for 3D printing safety verification
//! (Phase A of ALICE-Physics completeness project). Extends `material.rs`
//! (friction/restitution for rigid body contact) with **mechanical strength**
//! properties needed for stress analysis, thin-wall detection, print orientation
//! optimization, and warp risk analysis.
//!
//! # Coverage
//!
//! - **FDM Filaments**: PLA, PETG, ABS, PC, TPU, Nylon (PA6/PA12), CF-Nylon (PA-CF), PEEK
//! - **Sheet Metals**: SUS304 stainless, A5052 aluminum
//!
//! # Units
//!
//! Engineering units are used throughout for readability. All fields are `Fix128`
//! for cross-platform determinism.
//!
//! | Property | Unit |
//! |----------|------|
//! | Young's modulus | GPa |
//! | Yield strength | MPa |
//! | Tensile strength | MPa |
//! | Density | g/cm³ |
//! | Print / glass transition temp | °C |
//! | Bridging distance | mm |
//! | Cooling shrinkage | ratio (0.002 = 0.2%) |
//! | Anisotropy Z/XY ratio | dimensionless (0.65 = Z strength is 65% of XY) |
//!
//! # Sources
//!
//! - MatWeb.com material datasheets (accessed 2026-07)
//! - Prusa / Bambu Lab filament technical specs
//! - Ultimaker Cura material profiles
//! - ASM Metals Handbook Volume 2 (nonferrous alloys, SUS304 / A5052)
//! - Ashby & Jones "Engineering Materials" 5th ed.

use crate::math::Fix128;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Constants
// ============================================================================

/// Unit conversion: GPa → Pa (multiplier)
///
/// Young's modulus is stored in GPa for compactness; multiply by this to
/// obtain Pascals (N/m²) when composing stress formulas.
pub const GPA_TO_PA: Fix128 = Fix128 {
    hi: 1_000_000_000,
    lo: 0,
};

/// Unit conversion: MPa → Pa (multiplier)
pub const MPA_TO_PA: Fix128 = Fix128 {
    hi: 1_000_000,
    lo: 0,
};

/// Unit conversion: g/cm³ → kg/m³ (multiplier = 1000)
pub const G_CM3_TO_KG_M3: Fix128 = Fix128 { hi: 1000, lo: 0 };

// ============================================================================
// MaterialCategory
// ============================================================================

/// Material processing category — hints at which properties are meaningful.
///
/// Non-thermal fields (print_temp / glass_transition / bridging_distance) are
/// meaningless for `SheetMetal` and are set to `Fix128::ZERO` in presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MaterialCategory {
    /// Fused Deposition Modeling — thermoplastic filament
    Fdm,
    /// Sheet metal — cutting / bending (no layer-based print process)
    SheetMetal,
    /// Stereolithography / DLP resin printing (future extension)
    Sla,
    /// Selective Laser Sintering / Multi Jet Fusion powder (future extension)
    Powder,
}

// ============================================================================
// MaterialProperties
// ============================================================================

/// Material ID (matches `material::MaterialId` type for cross-reference).
pub type FilamentId = u16;

/// Mechanical + thermal properties for a 3D printing or sheet metal material.
///
/// Fields chosen to support Phase A-C analysis modules (thin_wall, beam_stress,
/// support_volume, layer_adhesion, print_orientation, bridging, warp_risk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialProperties {
    /// Numeric identifier (assigned by `FilamentDb::register`).
    pub id: FilamentId,
    /// Human-readable material name.
    pub name: &'static str,
    /// Material processing category.
    pub category: MaterialCategory,
    /// Young's modulus (GPa). Isotropic value; anisotropic Z direction obtained
    /// by multiplying by `anisotropy_z_ratio`.
    pub youngs_modulus_gpa: Fix128,
    /// Yield strength (MPa). Onset of permanent deformation.
    pub yield_strength_mpa: Fix128,
    /// Ultimate tensile strength (MPa). Break point.
    pub tensile_strength_mpa: Fix128,
    /// Density (g/cm³).
    pub density_g_cm3: Fix128,
    /// Print / process temperature (°C). Zero for non-thermal categories.
    pub print_temp_c: Fix128,
    /// Glass transition temperature Tg (°C). Zero for non-thermal categories.
    pub glass_transition_c: Fix128,
    /// Maximum unsupported bridge distance (mm). Zero for non-FDM categories.
    pub bridging_distance_mm: Fix128,
    /// Linear cooling shrinkage ratio (e.g. 0.002 = 0.2%).
    /// For sheet metal this is the thermal expansion coefficient contribution
    /// during welding / bending; keep zero for stock cold-formed processes.
    pub shrinkage_ratio: Fix128,
    /// Anisotropy Z / XY strength ratio (dimensionless, 0.0-1.0).
    ///
    /// FDM parts show reduced strength across layers (Z axis) versus in-plane
    /// (XY): typical 0.60-0.80 for PLA / ABS, 0.90+ for TPU (higher chain
    /// mobility improves layer bonding), 0.50 for fiber-reinforced (CF-Nylon)
    /// where fibers align in-plane. Sheet metals are 1.0 (isotropic).
    pub anisotropy_z_ratio: Fix128,
}

impl MaterialProperties {
    /// Effective Young's modulus (GPa) along the layer stacking (Z) axis.
    ///
    /// Applies the anisotropy ratio to the isotropic value.
    #[inline]
    #[must_use]
    pub fn youngs_z(&self) -> Fix128 {
        self.youngs_modulus_gpa * self.anisotropy_z_ratio
    }

    /// Effective yield strength (MPa) along the Z axis (layer stacking).
    #[inline]
    #[must_use]
    pub fn yield_z(&self) -> Fix128 {
        self.yield_strength_mpa * self.anisotropy_z_ratio
    }

    /// Effective tensile strength (MPa) along the Z axis.
    #[inline]
    #[must_use]
    pub fn tensile_z(&self) -> Fix128 {
        self.tensile_strength_mpa * self.anisotropy_z_ratio
    }

    /// Young's modulus in Pa (SI base unit) for use in stress formulas.
    #[inline]
    #[must_use]
    pub fn youngs_pa(&self) -> Fix128 {
        self.youngs_modulus_gpa * GPA_TO_PA
    }

    /// Yield strength in Pa.
    #[inline]
    #[must_use]
    pub fn yield_pa(&self) -> Fix128 {
        self.yield_strength_mpa * MPA_TO_PA
    }

    /// Density in SI base units (kg/m³).
    #[inline]
    #[must_use]
    pub fn density_si(&self) -> Fix128 {
        self.density_g_cm3 * G_CM3_TO_KG_M3
    }

    /// Whether this material is a 3D printing filament (FDM).
    #[inline]
    #[must_use]
    pub const fn is_fdm(&self) -> bool {
        matches!(self.category, MaterialCategory::Fdm)
    }

    /// Whether this material is sheet metal.
    #[inline]
    #[must_use]
    pub const fn is_sheet_metal(&self) -> bool {
        matches!(self.category, MaterialCategory::SheetMetal)
    }

    /// Anisotropic effective Young's modulus given a load direction angle
    /// `theta` (radians) from the XY plane.
    ///
    /// Uses a simple squared-cosine mixing model (Reuss-like lower bound):
    /// E(θ) = E_xy · cos²(θ) + E_z · sin²(θ)
    ///
    /// At θ = 0 (pure XY load) returns `youngs_modulus_gpa`; at θ = π/2
    /// (pure Z load) returns `youngs_z()`. This is a first-order engineering
    /// approximation; use `anisotropic.rs` (Phase B1) for the full Hill /
    /// Tsai-Wu criterion.
    #[must_use]
    pub fn youngs_at_angle(&self, theta: Fix128) -> Fix128 {
        let (s, c) = theta.sin_cos();
        let c2 = c * c;
        let s2 = s * s;
        self.youngs_modulus_gpa * c2 + self.youngs_z() * s2
    }

    /// Anisotropic effective yield strength (MPa) given a load direction angle.
    #[must_use]
    pub fn yield_at_angle(&self, theta: Fix128) -> Fix128 {
        let (s, c) = theta.sin_cos();
        let c2 = c * c;
        let s2 = s * s;
        self.yield_strength_mpa * c2 + self.yield_z() * s2
    }
}

// ============================================================================
// Presets
// ============================================================================

impl MaterialProperties {
    /// PLA (polylactic acid) filament — most common FDM material.
    ///
    /// Source: MatWeb PLA generic + Prusa PLA technical datasheet.
    /// Typical rigid but brittle; low glass transition (60°C) means poor
    /// heat resistance.
    #[must_use]
    pub fn pla() -> Self {
        Self {
            id: 0,
            name: "PLA",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(35, 10), // 3.5 GPa
            yield_strength_mpa: Fix128::from_int(50),
            tensile_strength_mpa: Fix128::from_int(60),
            density_g_cm3: Fix128::from_ratio(124, 100), // 1.24
            print_temp_c: Fix128::from_int(200),
            glass_transition_c: Fix128::from_int(60),
            bridging_distance_mm: Fix128::from_int(20),
            shrinkage_ratio: Fix128::from_ratio(2, 1000), // 0.2%
            anisotropy_z_ratio: Fix128::from_ratio(65, 100), // 0.65
        }
    }

    /// PETG (polyethylene terephthalate glycol) — impact resistant, food-safe.
    ///
    /// Source: MatWeb PETG generic + Bambu Lab PETG-CF datasheet (base PETG values).
    #[must_use]
    pub fn petg() -> Self {
        Self {
            id: 0,
            name: "PETG",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(20, 10), // 2.0 GPa
            yield_strength_mpa: Fix128::from_int(50),
            tensile_strength_mpa: Fix128::from_int(53),
            density_g_cm3: Fix128::from_ratio(127, 100), // 1.27
            print_temp_c: Fix128::from_int(235),
            glass_transition_c: Fix128::from_int(80),
            bridging_distance_mm: Fix128::from_int(15),
            shrinkage_ratio: Fix128::from_ratio(4, 1000), // 0.4%
            anisotropy_z_ratio: Fix128::from_ratio(70, 100), // 0.70
        }
    }

    /// ABS (acrylonitrile butadiene styrene) — impact/heat resistant.
    ///
    /// Source: MatWeb ABS injection-molded generic + Stratasys ABSplus datasheet.
    /// Higher warping risk than PLA due to 0.8% shrinkage.
    #[must_use]
    pub fn abs() -> Self {
        Self {
            id: 0,
            name: "ABS",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(23, 10), // 2.3 GPa
            yield_strength_mpa: Fix128::from_int(40),
            tensile_strength_mpa: Fix128::from_int(40),
            density_g_cm3: Fix128::from_ratio(104, 100), // 1.04
            print_temp_c: Fix128::from_int(240),
            glass_transition_c: Fix128::from_int(100),
            bridging_distance_mm: Fix128::from_int(12),
            shrinkage_ratio: Fix128::from_ratio(8, 1000), // 0.8%
            anisotropy_z_ratio: Fix128::from_ratio(68, 100), // 0.68
        }
    }

    /// PC (polycarbonate) — engineering plastic, high strength / heat resistance.
    ///
    /// Source: MatWeb PC injection-molded + Polymaker PC-Max datasheet.
    #[must_use]
    pub fn pc() -> Self {
        Self {
            id: 0,
            name: "PC",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(24, 10), // 2.4 GPa
            yield_strength_mpa: Fix128::from_int(65),
            tensile_strength_mpa: Fix128::from_int(70),
            density_g_cm3: Fix128::from_ratio(120, 100), // 1.20
            print_temp_c: Fix128::from_int(270),
            glass_transition_c: Fix128::from_int(145),
            bridging_distance_mm: Fix128::from_int(15),
            shrinkage_ratio: Fix128::from_ratio(6, 1000), // 0.6%
            anisotropy_z_ratio: Fix128::from_ratio(65, 100), // 0.65
        }
    }

    /// TPU (thermoplastic polyurethane) — elastomer, flexible.
    ///
    /// Source: MatWeb TPU Shore 95A generic + NinjaTek NinjaFlex datasheet.
    /// Young's modulus is highly deformation-dependent; stored value is small-
    /// strain (< 5% elongation). Use `hyperelastic.rs` (Phase B4) for large
    /// deformations.
    #[must_use]
    pub fn tpu() -> Self {
        Self {
            id: 0,
            name: "TPU",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(20, 1000), // 0.020 GPa
            yield_strength_mpa: Fix128::from_int(30),
            tensile_strength_mpa: Fix128::from_int(50),
            density_g_cm3: Fix128::from_ratio(120, 100), // 1.20
            print_temp_c: Fix128::from_int(220),
            glass_transition_c: Fix128::from_int(-30i64),
            bridging_distance_mm: Fix128::from_int(5),
            shrinkage_ratio: Fix128::from_ratio(15, 1000), // 1.5%
            anisotropy_z_ratio: Fix128::from_ratio(90, 100), // 0.90 (elastomer bonds well)
        }
    }

    /// Nylon (PA6 / PA12) — tough, chemical resistant.
    ///
    /// Source: MatWeb PA12 injection-molded + Taulman Nylon 645 datasheet.
    /// Hygroscopic (absorbs moisture) → strength varies with humidity.
    #[must_use]
    pub fn nylon() -> Self {
        Self {
            id: 0,
            name: "Nylon",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(15, 10), // 1.5 GPa
            yield_strength_mpa: Fix128::from_int(40),
            tensile_strength_mpa: Fix128::from_int(60),
            density_g_cm3: Fix128::from_ratio(105, 100), // 1.05
            print_temp_c: Fix128::from_int(260),
            glass_transition_c: Fix128::from_int(50),
            bridging_distance_mm: Fix128::from_int(10),
            shrinkage_ratio: Fix128::from_ratio(15, 1000), // 1.5%
            anisotropy_z_ratio: Fix128::from_ratio(75, 100), // 0.75
        }
    }

    /// CF-Nylon (carbon fiber reinforced nylon) — high stiffness.
    ///
    /// Source: Markforged Onyx datasheet + Bambu Lab PA-CF technical spec.
    /// Highly anisotropic due to fiber alignment during extrusion.
    #[must_use]
    pub fn cf_nylon() -> Self {
        Self {
            id: 0,
            name: "CF-Nylon",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_int(10),
            yield_strength_mpa: Fix128::from_int(85),
            tensile_strength_mpa: Fix128::from_int(100),
            density_g_cm3: Fix128::from_ratio(115, 100), // 1.15
            print_temp_c: Fix128::from_int(270),
            glass_transition_c: Fix128::from_int(60),
            bridging_distance_mm: Fix128::from_int(20),
            shrinkage_ratio: Fix128::from_ratio(5, 1000), // 0.5%
            anisotropy_z_ratio: Fix128::from_ratio(50, 100), // 0.50 (fibers align in XY)
        }
    }

    /// PEEK (polyether ether ketone) — highest-performance thermoplastic.
    ///
    /// Source: MatWeb PEEK Victrex 450G datasheet.
    /// Requires 400°C+ nozzle and 145°C+ chamber; industrial use only.
    #[must_use]
    pub fn peek() -> Self {
        Self {
            id: 0,
            name: "PEEK",
            category: MaterialCategory::Fdm,
            youngs_modulus_gpa: Fix128::from_ratio(40, 10), // 4.0 GPa
            yield_strength_mpa: Fix128::from_int(95),
            tensile_strength_mpa: Fix128::from_int(100),
            density_g_cm3: Fix128::from_ratio(132, 100), // 1.32
            print_temp_c: Fix128::from_int(400),
            glass_transition_c: Fix128::from_int(143),
            bridging_distance_mm: Fix128::from_int(25),
            shrinkage_ratio: Fix128::from_ratio(12, 1000), // 1.2%
            anisotropy_z_ratio: Fix128::from_ratio(70, 100), // 0.70
        }
    }

    /// SUS304 stainless steel — sheet metal (0.3-3mm typical thickness).
    ///
    /// Source: ASM Metals Handbook Vol 1, JIS G4304 SUS304.
    /// Isotropic; no thermal print process. Yield/tensile from annealed condition.
    #[must_use]
    pub fn sus304() -> Self {
        Self {
            id: 0,
            name: "SUS304",
            category: MaterialCategory::SheetMetal,
            youngs_modulus_gpa: Fix128::from_int(200),
            yield_strength_mpa: Fix128::from_int(215),
            tensile_strength_mpa: Fix128::from_int(505),
            density_g_cm3: Fix128::from_ratio(800, 100), // 8.00
            print_temp_c: Fix128::ZERO,
            glass_transition_c: Fix128::ZERO,
            bridging_distance_mm: Fix128::ZERO,
            shrinkage_ratio: Fix128::ZERO,
            anisotropy_z_ratio: Fix128::ONE, // Isotropic
        }
    }

    /// A5052 aluminum alloy — sheet metal (0.5-5mm typical).
    ///
    /// Source: ASM Metals Handbook Vol 2, JIS H4000 A5052 (Al-Mg 2.5%).
    /// Isotropic; corrosion-resistant marine-grade aluminum sheet.
    #[must_use]
    pub fn a5052() -> Self {
        Self {
            id: 0,
            name: "A5052",
            category: MaterialCategory::SheetMetal,
            youngs_modulus_gpa: Fix128::from_int(70),
            yield_strength_mpa: Fix128::from_int(90),
            tensile_strength_mpa: Fix128::from_int(230),
            density_g_cm3: Fix128::from_ratio(268, 100), // 2.68
            print_temp_c: Fix128::ZERO,
            glass_transition_c: Fix128::ZERO,
            bridging_distance_mm: Fix128::ZERO,
            shrinkage_ratio: Fix128::ZERO,
            anisotropy_z_ratio: Fix128::ONE,
        }
    }
}

// ============================================================================
// FilamentDb
// ============================================================================

/// Registry of material properties for 3D printing safety analysis.
///
/// Distinct from `material::MaterialTable` (which handles friction/restitution
/// for rigid body contact). Both databases may co-exist: use `MaterialTable`
/// for dynamics, `FilamentDb` for structural / thermal analysis.
#[derive(Clone, Debug, Default)]
pub struct FilamentDb {
    materials: Vec<MaterialProperties>,
}

impl FilamentDb {
    /// Create an empty database.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            materials: Vec::new(),
        }
    }

    /// Create a database pre-populated with all standard presets
    /// (8 FDM filaments + 2 sheet metals = 10 materials, IDs 0-9).
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut db = Self::new();
        db.register(MaterialProperties::pla());
        db.register(MaterialProperties::petg());
        db.register(MaterialProperties::abs());
        db.register(MaterialProperties::pc());
        db.register(MaterialProperties::tpu());
        db.register(MaterialProperties::nylon());
        db.register(MaterialProperties::cf_nylon());
        db.register(MaterialProperties::peek());
        db.register(MaterialProperties::sus304());
        db.register(MaterialProperties::a5052());
        db
    }

    /// Register a material and return its assigned ID.
    pub fn register(&mut self, mut material: MaterialProperties) -> FilamentId {
        let id = self.materials.len() as FilamentId;
        material.id = id;
        self.materials.push(material);
        id
    }

    /// Lookup by ID. Returns `None` if the ID is out of range.
    #[must_use]
    pub fn get(&self, id: FilamentId) -> Option<&MaterialProperties> {
        self.materials.get(id as usize)
    }

    /// Lookup by material name (case-sensitive). Returns the first match.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&MaterialProperties> {
        self.materials.iter().find(|m| m.name == name)
    }

    /// Number of registered materials.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// Whether the database contains any materials.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Iterate all registered materials.
    pub fn iter(&self) -> core::slice::Iter<'_, MaterialProperties> {
        self.materials.iter()
    }

    /// Filter by processing category. Allocates a `Vec` of references.
    #[must_use]
    pub fn by_category(&self, category: MaterialCategory) -> Vec<&MaterialProperties> {
        self.materials
            .iter()
            .filter(|m| m.category == category)
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pla_preset_values() {
        let pla = MaterialProperties::pla();
        assert_eq!(pla.name, "PLA");
        assert_eq!(pla.category, MaterialCategory::Fdm);
        assert_eq!(pla.youngs_modulus_gpa, Fix128::from_ratio(35, 10));
        assert_eq!(pla.glass_transition_c, Fix128::from_int(60));
        assert!(pla.is_fdm());
        assert!(!pla.is_sheet_metal());
    }

    #[test]
    fn all_fdm_presets_have_thermal_data() {
        for m in [
            MaterialProperties::pla(),
            MaterialProperties::petg(),
            MaterialProperties::abs(),
            MaterialProperties::pc(),
            MaterialProperties::tpu(),
            MaterialProperties::nylon(),
            MaterialProperties::cf_nylon(),
            MaterialProperties::peek(),
        ] {
            assert!(m.is_fdm(), "{} should be FDM", m.name);
            assert!(m.print_temp_c > Fix128::ZERO, "{}: print_temp", m.name);
            // TPU has Tg = -30°C so allow non-zero (including negative)
            if m.name != "TPU" {
                assert!(
                    m.glass_transition_c > Fix128::ZERO,
                    "{}: glass_transition",
                    m.name
                );
            }
            assert!(
                m.bridging_distance_mm > Fix128::ZERO,
                "{}: bridging",
                m.name
            );
            assert!(m.anisotropy_z_ratio > Fix128::ZERO);
            assert!(m.anisotropy_z_ratio <= Fix128::ONE);
        }
    }

    #[test]
    fn sheet_metal_presets_have_zero_thermal() {
        for m in [MaterialProperties::sus304(), MaterialProperties::a5052()] {
            assert!(m.is_sheet_metal());
            assert_eq!(m.print_temp_c, Fix128::ZERO);
            assert_eq!(m.glass_transition_c, Fix128::ZERO);
            assert_eq!(m.bridging_distance_mm, Fix128::ZERO);
            assert_eq!(m.anisotropy_z_ratio, Fix128::ONE);
        }
    }

    #[test]
    fn anisotropy_z_reduces_strength() {
        let pla = MaterialProperties::pla();
        let e_xy = pla.youngs_modulus_gpa;
        let e_z = pla.youngs_z();
        // PLA anisotropy ratio = 0.65 → E_z should be < E_xy
        assert!(e_z < e_xy);
        // E_z / E_xy = anisotropy_ratio (0.65)
        let expected = e_xy * Fix128::from_ratio(65, 100);
        assert_eq!(e_z, expected);
    }

    #[test]
    fn unit_conversion_gpa_to_pa() {
        // 1 GPa == 1_000_000_000 Pa
        let one_gpa = Fix128::ONE;
        let one_gpa_in_pa = one_gpa * GPA_TO_PA;
        assert_eq!(one_gpa_in_pa, Fix128::from_int(1_000_000_000));
    }

    #[test]
    fn density_si_conversion() {
        let pla = MaterialProperties::pla();
        // 1.24 g/cm³ = 1240 kg/m³. Fix128 multiplication of a non-terminating
        // binary fraction (1.24 has infinite binary expansion) will differ from
        // the exact integer by ~1 part in 2^64. Use tolerance ≤ 1 kg/m³.
        let expected = Fix128::from_int(1240);
        let actual = pla.density_si();
        let diff = if actual > expected {
            actual - expected
        } else {
            expected - actual
        };
        assert!(
            diff < Fix128::ONE,
            "density should be within 1 kg/m³ of 1240"
        );
    }

    #[test]
    fn cf_nylon_most_anisotropic() {
        // CF-Nylon should have the lowest Z ratio (0.50) among FDM presets
        // due to in-plane fiber alignment
        let cfn = MaterialProperties::cf_nylon();
        let pla = MaterialProperties::pla();
        let petg = MaterialProperties::petg();
        assert!(cfn.anisotropy_z_ratio < pla.anisotropy_z_ratio);
        assert!(cfn.anisotropy_z_ratio < petg.anisotropy_z_ratio);
    }

    #[test]
    fn tpu_least_anisotropic_fdm() {
        // TPU (elastomer) bonds better between layers → highest Z ratio among FDM
        let tpu = MaterialProperties::tpu();
        let abs = MaterialProperties::abs();
        assert!(tpu.anisotropy_z_ratio > abs.anisotropy_z_ratio);
    }

    #[test]
    fn abs_higher_shrinkage_than_pla() {
        // ABS warps more than PLA (0.8% vs 0.2%)
        let pla = MaterialProperties::pla();
        let abs = MaterialProperties::abs();
        assert!(abs.shrinkage_ratio > pla.shrinkage_ratio);
    }

    #[test]
    fn peek_high_glass_transition_of_fdm() {
        // PEEK Tg = 143°C is one of the highest among common FDM filaments.
        // Polycarbonate (PC) is comparable (145°C) due to its rigid aromatic
        // backbone; the test excludes PC to focus on separation from lower-Tg
        // materials (PLA 60 / ABS 100 / Nylon 50).
        let peek = MaterialProperties::peek();
        for m in [
            MaterialProperties::pla(),
            MaterialProperties::petg(),
            MaterialProperties::abs(),
            MaterialProperties::nylon(),
            MaterialProperties::cf_nylon(),
        ] {
            assert!(
                peek.glass_transition_c > m.glass_transition_c,
                "PEEK Tg (143C) should exceed {} Tg",
                m.name
            );
        }
    }

    #[test]
    fn sus304_stiffest_material() {
        let sus = MaterialProperties::sus304();
        for m in [
            MaterialProperties::pla(),
            MaterialProperties::peek(),
            MaterialProperties::a5052(),
        ] {
            assert!(
                sus.youngs_modulus_gpa > m.youngs_modulus_gpa,
                "SUS304 should be stiffer than {}",
                m.name
            );
        }
    }

    #[test]
    fn db_defaults_registers_all_presets() {
        let db = FilamentDb::with_defaults();
        assert_eq!(db.len(), 10);
        assert!(!db.is_empty());
    }

    #[test]
    fn db_register_assigns_sequential_ids() {
        let mut db = FilamentDb::new();
        let id0 = db.register(MaterialProperties::pla());
        let id1 = db.register(MaterialProperties::petg());
        let id2 = db.register(MaterialProperties::abs());
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(db.get(id0).unwrap().name, "PLA");
        assert_eq!(db.get(id1).unwrap().name, "PETG");
        assert_eq!(db.get(id2).unwrap().name, "ABS");
    }

    #[test]
    fn db_get_out_of_range_returns_none() {
        let db = FilamentDb::with_defaults();
        assert!(db.get(99).is_none());
    }

    #[test]
    fn db_find_by_name() {
        let db = FilamentDb::with_defaults();
        let peek = db.find_by_name("PEEK").expect("PEEK present");
        assert_eq!(peek.category, MaterialCategory::Fdm);
        assert!(db.find_by_name("Nonexistent").is_none());
    }

    #[test]
    fn db_by_category_partitions() {
        let db = FilamentDb::with_defaults();
        let fdm = db.by_category(MaterialCategory::Fdm);
        let sheet = db.by_category(MaterialCategory::SheetMetal);
        assert_eq!(fdm.len(), 8);
        assert_eq!(sheet.len(), 2);
        assert_eq!(fdm.len() + sheet.len(), db.len());
    }

    #[test]
    fn youngs_at_angle_boundary() {
        let pla = MaterialProperties::pla();
        // θ = 0 → pure XY, should approximately equal youngs_modulus_gpa.
        // CORDIC sin_cos(0) has ~2^-64 error accumulated across fixed iterations.
        let e0 = pla.youngs_at_angle(Fix128::ZERO);
        let expected = pla.youngs_modulus_gpa;
        let diff = if e0 > expected {
            e0 - expected
        } else {
            expected - e0
        };
        // Tolerance: 0.01% of E_xy (accounts for CORDIC accumulated error)
        let tol = pla.youngs_modulus_gpa * Fix128::from_ratio(1, 10_000);
        assert!(diff <= tol, "youngs_at_angle(0) should be near E_xy");
    }

    #[test]
    fn youngs_at_angle_ninety_degrees() {
        let pla = MaterialProperties::pla();
        // θ = π/2 → pure Z, should approximately equal youngs_z()
        // (CORDIC may have small error at exact π/2)
        let e90 = pla.youngs_at_angle(Fix128::HALF_PI);
        let e_z = pla.youngs_z();
        // Allow small numerical error
        let diff = if e90 > e_z { e90 - e_z } else { e_z - e90 };
        // Tolerance: 0.1% of E_xy
        let tol = pla.youngs_modulus_gpa * Fix128::from_ratio(1, 1000);
        assert!(diff <= tol, "youngs_at_angle(π/2) should be near youngs_z");
    }

    #[test]
    fn determinism_bit_exact_across_calls() {
        // Preset construction is const-derived from from_int/from_ratio which
        // are pure integer arithmetic → bit-exact between calls.
        let a = MaterialProperties::pla();
        let b = MaterialProperties::pla();
        assert_eq!(a.youngs_modulus_gpa, b.youngs_modulus_gpa);
        assert_eq!(a.yield_strength_mpa, b.yield_strength_mpa);
        assert_eq!(a.density_g_cm3, b.density_g_cm3);
        assert_eq!(a.anisotropy_z_ratio, b.anisotropy_z_ratio);
    }
}
