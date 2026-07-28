//! Beam Stress Analysis for 3D Print Safety
//!
//! Phase A3 of the ALICE-Physics completeness project. Static-load stress and
//! deflection analysis for engineering beams, covering the common load cases
//! that dominate FDM printed part failures (cantilever hooks, shelf brackets,
//! SKADIS pegboard accessories, snap-fit tabs).
//!
//! # Coverage
//!
//! - **Cross sections**: rectangular, circular, hollow rectangular, hollow
//!   circular, I-beam. Provides area, second moment of area (I), and section
//!   modulus (Z = I / c).
//! - **Load cases**: cantilever with end point load or distributed load,
//!   simply-supported with center point load or distributed load.
//! - **Stress**: maximum fibre bending stress σ = M·c / I (MPa).
//! - **Deflection**: elastic tip / centre deflection using classical closed-
//!   form beam theory.
//! - **Euler buckling**: critical axial load `P_cr = π²·E·I / (K·L)²` with
//!   configurable end condition factor `K` (1.0 pin-pin, 0.5 fixed-fixed,
//!   2.0 cantilever, 0.7 fixed-pin).
//! - **Factor of Safety**: yield / actual, plus `is_safe` boolean using a
//!   configurable minimum FoS (default 2.0 per ASME B31 industrial practice).
//!
//! # Unit convention
//!
//! Uses the "engineering" system throughout for numerical stability under
//! Fix128:
//!
//! | Quantity | Unit |
//! |----------|------|
//! | Length | mm |
//! | Force | N |
//! | Bending moment | N·mm |
//! | Stress / modulus | MPa (= N/mm²) |
//! | Area | mm² |
//! | Second moment I | mm⁴ |
//! | Section modulus Z | mm³ |
//!
//! This avoids the ~10⁹ multipliers required by pure SI (Pa, m) and keeps
//! intermediate products within Fix128's ±9.2×10¹⁸ signed integer range.
//!
//! # References
//!
//! - Roark's Formulas for Stress and Strain, 8th ed. Chapter 8 "Beams;
//!   Flexure of Straight Bars", Table 8.1 (loading formulas).
//! - Timoshenko & Gere, "Theory of Elastic Stability" (Euler buckling).
//! - ASME B31.3 "Process Piping" §302.3.5 (minimum factor of safety).

use crate::filament_db::MaterialProperties;
use crate::math::Fix128;

// ============================================================================
// Cross section
// ============================================================================

/// Beam cross-section geometry. Determines area, second moment of area (I),
/// and maximum fibre distance `c` from the neutral axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CrossSection {
    /// Solid rectangle: `width_mm × height_mm`. Neutral axis is horizontal
    /// centreline; bending is about that axis.
    Rectangular {
        /// Width (mm) — dimension normal to the load direction.
        width_mm: Fix128,
        /// Height (mm) — dimension parallel to the load direction.
        height_mm: Fix128,
    },
    /// Solid circle of given diameter (mm).
    Circular {
        /// Outer diameter (mm).
        diameter_mm: Fix128,
    },
    /// Rectangular hollow section (tube).
    HollowRectangular {
        /// Outer width (mm).
        outer_width_mm: Fix128,
        /// Outer height (mm).
        outer_height_mm: Fix128,
        /// Wall thickness on all four sides (mm).
        wall_mm: Fix128,
    },
    /// Circular hollow section (tube).
    HollowCircular {
        /// Outer diameter (mm).
        outer_diameter_mm: Fix128,
        /// Inner diameter (mm).
        inner_diameter_mm: Fix128,
    },
    /// I-beam with equal flanges. `height` is total; `flange_thickness` is
    /// each flange; `web_thickness` is the vertical stem.
    IBeam {
        /// Total flange width (mm).
        flange_width_mm: Fix128,
        /// Total section height (mm).
        height_mm: Fix128,
        /// Thickness of each flange (mm).
        flange_thickness_mm: Fix128,
        /// Web (vertical stem) thickness (mm).
        web_thickness_mm: Fix128,
    },
}

impl CrossSection {
    /// Cross-sectional area (mm²).
    #[must_use]
    pub fn area_mm2(&self) -> Fix128 {
        match *self {
            Self::Rectangular {
                width_mm,
                height_mm,
            } => width_mm * height_mm,
            Self::Circular { diameter_mm } => {
                // A = π/4 · d²
                let d2 = diameter_mm * diameter_mm;
                Fix128::PI * d2 * Fix128::from_ratio(1, 4)
            }
            Self::HollowRectangular {
                outer_width_mm,
                outer_height_mm,
                wall_mm,
            } => {
                let outer = outer_width_mm * outer_height_mm;
                let inner_w = outer_width_mm - wall_mm.double();
                let inner_h = outer_height_mm - wall_mm.double();
                outer - inner_w * inner_h
            }
            Self::HollowCircular {
                outer_diameter_mm,
                inner_diameter_mm,
            } => {
                let do2 = outer_diameter_mm * outer_diameter_mm;
                let di2 = inner_diameter_mm * inner_diameter_mm;
                Fix128::PI * (do2 - di2) * Fix128::from_ratio(1, 4)
            }
            Self::IBeam {
                flange_width_mm,
                height_mm,
                flange_thickness_mm,
                web_thickness_mm,
            } => {
                // 2 flanges + web
                let flange_a = flange_width_mm * flange_thickness_mm;
                let web_h = height_mm - flange_thickness_mm.double();
                let web_a = web_thickness_mm * web_h;
                flange_a.double() + web_a
            }
        }
    }

    /// Second moment of area about the horizontal neutral axis (mm⁴).
    ///
    /// Assumes bending occurs about the axis of maximum stiffness (the
    /// horizontal centreline for rectangular / I-beam sections, the diameter
    /// for circular sections).
    #[must_use]
    pub fn second_moment_of_area_mm4(&self) -> Fix128 {
        match *self {
            Self::Rectangular {
                width_mm,
                height_mm,
            } => {
                // I = b · h³ / 12
                let h3 = height_mm * height_mm * height_mm;
                width_mm * h3 * Fix128::from_ratio(1, 12)
            }
            Self::Circular { diameter_mm } => {
                // I = π / 64 · d⁴
                let d2 = diameter_mm * diameter_mm;
                let d4 = d2 * d2;
                Fix128::PI * d4 * Fix128::from_ratio(1, 64)
            }
            Self::HollowRectangular {
                outer_width_mm,
                outer_height_mm,
                wall_mm,
            } => {
                let oh3 = outer_height_mm * outer_height_mm * outer_height_mm;
                let outer_i = outer_width_mm * oh3 * Fix128::from_ratio(1, 12);
                let inner_w = outer_width_mm - wall_mm.double();
                let inner_h = outer_height_mm - wall_mm.double();
                let ih3 = inner_h * inner_h * inner_h;
                let inner_i = inner_w * ih3 * Fix128::from_ratio(1, 12);
                outer_i - inner_i
            }
            Self::HollowCircular {
                outer_diameter_mm,
                inner_diameter_mm,
            } => {
                let do2 = outer_diameter_mm * outer_diameter_mm;
                let do4 = do2 * do2;
                let di2 = inner_diameter_mm * inner_diameter_mm;
                let di4 = di2 * di2;
                Fix128::PI * (do4 - di4) * Fix128::from_ratio(1, 64)
            }
            Self::IBeam {
                flange_width_mm,
                height_mm,
                flange_thickness_mm,
                web_thickness_mm,
            } => {
                // I = b·h³/12 (bounding rectangle) - (b - t_web) · h_web³ / 12
                // Standard I-beam formula using outer bounding box minus removed side rectangles.
                let h3 = height_mm * height_mm * height_mm;
                let outer_i = flange_width_mm * h3 * Fix128::from_ratio(1, 12);
                let web_h = height_mm - flange_thickness_mm.double();
                let wh3 = web_h * web_h * web_h;
                let removed_w = flange_width_mm - web_thickness_mm;
                let removed_i = removed_w * wh3 * Fix128::from_ratio(1, 12);
                outer_i - removed_i
            }
        }
    }

    /// Distance from neutral axis to the extreme fibre (mm).
    #[must_use]
    pub fn max_c_mm(&self) -> Fix128 {
        match *self {
            Self::Rectangular { height_mm, .. } => height_mm.half(),
            Self::Circular { diameter_mm } => diameter_mm.half(),
            Self::HollowRectangular {
                outer_height_mm, ..
            } => outer_height_mm.half(),
            Self::HollowCircular {
                outer_diameter_mm, ..
            } => outer_diameter_mm.half(),
            Self::IBeam { height_mm, .. } => height_mm.half(),
        }
    }

    /// Section modulus Z = I / c (mm³).
    #[must_use]
    pub fn section_modulus_mm3(&self) -> Fix128 {
        let c = self.max_c_mm();
        if c.is_zero() {
            return Fix128::ZERO;
        }
        self.second_moment_of_area_mm4() / c
    }
}

// ============================================================================
// Load case & end conditions
// ============================================================================

/// Bending load configuration. Enum variants encode both the loading pattern
/// and the support arrangement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadCase {
    /// Cantilever (one end fixed, other free) with a point load at the free end.
    CantileverEndPoint {
        /// Applied load (N) at the free end, directed transversely.
        load_n: Fix128,
        /// Span (mm) from fixed end to free end.
        length_mm: Fix128,
    },
    /// Cantilever with a uniformly distributed load along its length.
    CantileverDistributed {
        /// Load intensity (N per mm of span).
        load_per_mm_n: Fix128,
        /// Span (mm).
        length_mm: Fix128,
    },
    /// Simply supported beam (pin-pin) with a point load at the mid-span.
    SimplySupportedCenter {
        /// Applied load (N) at mid-span.
        load_n: Fix128,
        /// Total span (mm).
        length_mm: Fix128,
    },
    /// Simply supported beam with uniformly distributed load.
    SimplySupportedDistributed {
        /// Load intensity (N per mm).
        load_per_mm_n: Fix128,
        /// Total span (mm).
        length_mm: Fix128,
    },
}

impl LoadCase {
    /// Maximum bending moment (N·mm) that appears anywhere along the beam.
    ///
    /// Location of the maximum:
    /// - Cantilever cases: at the fixed end.
    /// - Simply supported cases: at mid-span.
    #[must_use]
    pub fn max_bending_moment_nmm(&self) -> Fix128 {
        match *self {
            Self::CantileverEndPoint { load_n, length_mm } => load_n * length_mm,
            Self::CantileverDistributed {
                load_per_mm_n,
                length_mm,
            } => {
                // M = w·L² / 2
                let l2 = length_mm * length_mm;
                load_per_mm_n * l2 * Fix128::from_ratio(1, 2)
            }
            Self::SimplySupportedCenter { load_n, length_mm } => {
                // M = F·L / 4
                load_n * length_mm * Fix128::from_ratio(1, 4)
            }
            Self::SimplySupportedDistributed {
                load_per_mm_n,
                length_mm,
            } => {
                // M = w·L² / 8
                let l2 = length_mm * length_mm;
                load_per_mm_n * l2 * Fix128::from_ratio(1, 8)
            }
        }
    }

    /// Beam span for this load case (mm).
    #[must_use]
    pub fn length_mm(&self) -> Fix128 {
        match *self {
            Self::CantileverEndPoint { length_mm, .. }
            | Self::CantileverDistributed { length_mm, .. }
            | Self::SimplySupportedCenter { length_mm, .. }
            | Self::SimplySupportedDistributed { length_mm, .. } => length_mm,
        }
    }

    /// Maximum elastic deflection (mm) given section and Young's modulus (MPa).
    ///
    /// Uses closed-form Roark's formulas; assumes small deflection theory
    /// (δ ≪ L) and homogeneous isotropic material.
    #[must_use]
    pub fn max_deflection_mm(&self, section: &CrossSection, e_mpa: Fix128) -> Fix128 {
        let i = section.second_moment_of_area_mm4();
        let ei = e_mpa * i;
        if ei.is_zero() {
            return Fix128::ZERO;
        }
        match *self {
            Self::CantileverEndPoint { load_n, length_mm } => {
                // δ = F·L³ / (3·E·I)
                let l3 = length_mm * length_mm * length_mm;
                load_n * l3 / (Fix128::from_int(3) * ei)
            }
            Self::CantileverDistributed {
                load_per_mm_n,
                length_mm,
            } => {
                // δ = w·L⁴ / (8·E·I)
                let l2 = length_mm * length_mm;
                let l4 = l2 * l2;
                load_per_mm_n * l4 / (Fix128::from_int(8) * ei)
            }
            Self::SimplySupportedCenter { load_n, length_mm } => {
                // δ = F·L³ / (48·E·I)
                let l3 = length_mm * length_mm * length_mm;
                load_n * l3 / (Fix128::from_int(48) * ei)
            }
            Self::SimplySupportedDistributed {
                load_per_mm_n,
                length_mm,
            } => {
                // δ = 5·w·L⁴ / (384·E·I)
                let l2 = length_mm * length_mm;
                let l4 = l2 * l2;
                Fix128::from_int(5) * load_per_mm_n * l4 / (Fix128::from_int(384) * ei)
            }
        }
    }
}

// ============================================================================
// Buckling
// ============================================================================

/// End-condition factor `K` for Euler's buckling formula
/// `P_cr = π²·E·I / (K·L)²`.
///
/// Values from Timoshenko & Gere "Theory of Elastic Stability" Table 2-1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColumnEndCondition {
    /// Pinned-Pinned (K = 1.0). Most common assumption; conservative.
    PinPin,
    /// Fixed-Fixed (K = 0.5). Both ends rotationally restrained.
    FixedFixed,
    /// Fixed-Free (K = 2.0). Cantilever column — SKADIS pegs, snap-fit tabs.
    Cantilever,
    /// Fixed-Pinned (K = 0.7).
    FixedPin,
}

impl ColumnEndCondition {
    /// Numerical `K` value.
    #[inline]
    #[must_use]
    pub fn k_factor(&self) -> Fix128 {
        match self {
            Self::PinPin => Fix128::ONE,
            Self::FixedFixed => Fix128::from_ratio(1, 2),
            Self::Cantilever => Fix128::from_int(2),
            Self::FixedPin => Fix128::from_ratio(7, 10),
        }
    }
}

/// Critical Euler buckling load (N) for a slender column.
///
/// `P_cr = π²·E·I / (K·L)²`
///
/// If the axial compressive load exceeds this value the column buckles
/// laterally regardless of the yield strength of the material. Applicable
/// to slender members with slenderness ratio (KL/r) > 100 as a rule of
/// thumb; shorter columns need Johnson's formula (not implemented here).
#[must_use]
pub fn euler_critical_load_n(
    section: &CrossSection,
    length_mm: Fix128,
    e_mpa: Fix128,
    end_condition: ColumnEndCondition,
) -> Fix128 {
    let i = section.second_moment_of_area_mm4();
    let kl = end_condition.k_factor() * length_mm;
    let kl2 = kl * kl;
    if kl2.is_zero() {
        return Fix128::ZERO;
    }
    let pi2 = Fix128::PI * Fix128::PI;
    pi2 * e_mpa * i / kl2
}

// ============================================================================
// Beam analysis façade
// ============================================================================

/// Full beam configuration: geometry + support + loading + material.
#[derive(Clone, Copy, Debug)]
pub struct BeamAnalysis {
    /// Cross-section geometry.
    pub section: CrossSection,
    /// Bending load configuration.
    pub load: LoadCase,
    /// Material properties.
    pub material: MaterialProperties,
    /// Column end condition used for buckling check (independent of `load`
    /// which describes bending). For pure bending analyses set to `PinPin`.
    pub end_condition: ColumnEndCondition,
    /// Minimum acceptable factor of safety. Default 2.0 (ASME B31 industrial).
    pub min_factor_of_safety: Fix128,
}

impl BeamAnalysis {
    /// Convenience constructor with default FoS = 2.0 and pin-pin ends.
    #[must_use]
    pub fn new(section: CrossSection, load: LoadCase, material: MaterialProperties) -> Self {
        Self {
            section,
            load,
            material,
            end_condition: ColumnEndCondition::PinPin,
            min_factor_of_safety: Fix128::from_int(2),
        }
    }

    /// Override the buckling end condition.
    #[must_use]
    pub const fn with_end_condition(mut self, end_condition: ColumnEndCondition) -> Self {
        self.end_condition = end_condition;
        self
    }

    /// Override the minimum factor of safety.
    #[must_use]
    pub const fn with_min_fos(mut self, min_fos: Fix128) -> Self {
        self.min_factor_of_safety = min_fos;
        self
    }

    /// Compute all analysis outputs.
    #[must_use]
    pub fn analyze(&self) -> BeamReport {
        let moment = self.load.max_bending_moment_nmm();
        let z = self.section.section_modulus_mm3();
        let stress_mpa = if z.is_zero() {
            Fix128::ZERO
        } else {
            moment / z
        };

        // For anisotropic materials use the yield strength appropriate to
        // the bending direction. Here we assume in-plane loading (XY strength).
        let yield_mpa = self.material.yield_strength_mpa;
        let fos = if stress_mpa.is_zero() {
            Fix128::from_int(i64::MAX >> 8) // "infinite"
        } else {
            yield_mpa / stress_mpa
        };

        // Deflection with material Young's modulus (GPa → MPa: × 1000).
        let e_mpa = self.material.youngs_modulus_gpa * Fix128::from_int(1000);
        let deflection = self.load.max_deflection_mm(&self.section, e_mpa);

        // Buckling load along the beam axis.
        let p_cr = euler_critical_load_n(
            &self.section,
            self.load.length_mm(),
            e_mpa,
            self.end_condition,
        );

        BeamReport {
            max_bending_moment_nmm: moment,
            max_bending_stress_mpa: stress_mpa,
            max_deflection_mm: deflection,
            euler_critical_load_n: p_cr,
            factor_of_safety: fos,
            is_safe: fos >= self.min_factor_of_safety,
        }
    }
}

/// Combined output of a `BeamAnalysis::analyze()` call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeamReport {
    /// Maximum bending moment along the beam (N·mm).
    pub max_bending_moment_nmm: Fix128,
    /// Peak fibre bending stress (MPa) using σ = M / Z.
    pub max_bending_stress_mpa: Fix128,
    /// Maximum elastic deflection (mm).
    pub max_deflection_mm: Fix128,
    /// Critical Euler axial buckling load (N).
    pub euler_critical_load_n: Fix128,
    /// Factor of safety = yield / actual bending stress. Dimensionless.
    pub factor_of_safety: Fix128,
    /// True iff `factor_of_safety >= min_factor_of_safety`.
    pub is_safe: bool,
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
    fn rectangular_area() {
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        // A = 10 × 20 = 200
        assert_eq!(s.area_mm2(), Fix128::from_int(200));
    }

    #[test]
    fn rectangular_second_moment_of_area() {
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        // I = b·h³/12 = 10 · 8000 / 12 = 80000/12 ≈ 6666.67
        let i = s.second_moment_of_area_mm4();
        let expected = Fix128::from_ratio(80_000, 12);
        assert!(approx_eq(i, expected, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn rectangular_section_modulus() {
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        // Z = I/c = (b·h²/6) = 10 · 400 / 6 = 666.67
        let z = s.section_modulus_mm3();
        let expected = Fix128::from_ratio(4000, 6);
        assert!(approx_eq(z, expected, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn circular_second_moment() {
        let s = CrossSection::Circular {
            diameter_mm: Fix128::from_int(10),
        };
        // I = π·d⁴/64 = π · 10000 / 64 ≈ 490.87
        let i = s.second_moment_of_area_mm4();
        // Approx π ≈ 3.14159
        let expected = Fix128::from_ratio(3141_59 * 10_000, 100_000 * 64);
        assert!(
            approx_eq(i, expected, Fix128::from_ratio(1, 10)),
            "got {}, expected {}",
            i.to_f32(),
            expected.to_f32()
        );
    }

    #[test]
    fn hollow_rectangular_area() {
        let s = CrossSection::HollowRectangular {
            outer_width_mm: Fix128::from_int(20),
            outer_height_mm: Fix128::from_int(20),
            wall_mm: Fix128::from_int(2),
        };
        // outer 400 - inner 16×16 = 256, so A = 144
        assert_eq!(s.area_mm2(), Fix128::from_int(144));
    }

    #[test]
    fn hollow_circular_area() {
        let s = CrossSection::HollowCircular {
            outer_diameter_mm: Fix128::from_int(10),
            inner_diameter_mm: Fix128::from_int(8),
        };
        // A = π/4 · (100 - 64) = 9π ≈ 28.27
        let a = s.area_mm2();
        let expected = Fix128::PI * Fix128::from_int(9);
        assert!(approx_eq(a, expected, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn i_beam_geometry() {
        // 100 wide × 200 tall, flange 20 thick, web 10 thick
        let s = CrossSection::IBeam {
            flange_width_mm: Fix128::from_int(100),
            height_mm: Fix128::from_int(200),
            flange_thickness_mm: Fix128::from_int(20),
            web_thickness_mm: Fix128::from_int(10),
        };
        // Area = 2·(100·20) + 10·(200-40) = 4000 + 1600 = 5600
        assert_eq!(s.area_mm2(), Fix128::from_int(5600));
        // c = 100 (half of height)
        assert_eq!(s.max_c_mm(), Fix128::from_int(100));
    }

    #[test]
    fn cantilever_end_point_moment() {
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(100),
            length_mm: Fix128::from_int(500),
        };
        // M = F·L = 100 · 500 = 50000
        assert_eq!(load.max_bending_moment_nmm(), Fix128::from_int(50_000));
    }

    #[test]
    fn cantilever_distributed_moment() {
        let load = LoadCase::CantileverDistributed {
            load_per_mm_n: Fix128::from_ratio(1, 2), // 0.5 N/mm
            length_mm: Fix128::from_int(200),
        };
        // M = w·L²/2 = 0.5 · 40000 / 2 = 10000
        assert_eq!(load.max_bending_moment_nmm(), Fix128::from_int(10_000));
    }

    #[test]
    fn simply_supported_center_moment() {
        let load = LoadCase::SimplySupportedCenter {
            load_n: Fix128::from_int(200),
            length_mm: Fix128::from_int(400),
        };
        // M = F·L/4 = 200 · 400 / 4 = 20000
        assert_eq!(load.max_bending_moment_nmm(), Fix128::from_int(20_000));
    }

    #[test]
    fn simply_supported_distributed_moment() {
        let load = LoadCase::SimplySupportedDistributed {
            load_per_mm_n: Fix128::ONE,
            length_mm: Fix128::from_int(100),
        };
        // M = w·L²/8 = 1 · 10000 / 8 = 1250
        assert_eq!(load.max_bending_moment_nmm(), Fix128::from_int(1_250));
    }

    #[test]
    fn cantilever_deflection_pla_shelf() {
        // Realistic scenario: 10×20 rect, 300mm cantilever, 30N load, PLA
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(30),
            length_mm: Fix128::from_int(300),
        };
        // E = 3.5 GPa = 3500 MPa
        let e_mpa = Fix128::from_int(3500);
        let delta = load.max_deflection_mm(&s, e_mpa);
        // δ = F·L³ / (3·E·I) = 30 · 27_000_000 / (3 · 3500 · 6666.67)
        //   = 810_000_000 / 70_000_035 ≈ 11.57 mm
        let expected = Fix128::from_ratio(1157, 100);
        assert!(
            approx_eq(delta, expected, Fix128::from_ratio(5, 100)),
            "got {} mm, expected ~11.57",
            delta.to_f32()
        );
    }

    #[test]
    fn euler_buckling_pin_pin() {
        // 10mm square, 500mm length, PLA (E=3500 MPa)
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(10),
        };
        // I = 10·1000/12 = 833.33 mm⁴
        // P_cr = π² · 3500 · 833.33 / 500² = 9.8696 · 3500 · 833.33 / 250000
        //      = 2.879e7 / 250000 ≈ 115.14 N
        let p = euler_critical_load_n(
            &s,
            Fix128::from_int(500),
            Fix128::from_int(3500),
            ColumnEndCondition::PinPin,
        );
        let expected = Fix128::from_ratio(11514, 100);
        assert!(
            approx_eq(p, expected, Fix128::from_int(2)),
            "got {} N, expected ~115",
            p.to_f32()
        );
    }

    #[test]
    fn cantilever_end_condition_lowers_pcr() {
        // Cantilever K=2 vs PinPin K=1 → cantilever P_cr should be 1/4 of pin-pin
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(10),
        };
        let l = Fix128::from_int(500);
        let e = Fix128::from_int(3500);
        let p_pin = euler_critical_load_n(&s, l, e, ColumnEndCondition::PinPin);
        let p_cant = euler_critical_load_n(&s, l, e, ColumnEndCondition::Cantilever);
        // Ratio should be 4
        let ratio = p_pin / p_cant;
        assert!(
            approx_eq(ratio, Fix128::from_int(4), Fix128::from_ratio(1, 100)),
            "ratio = {}, expected 4",
            ratio.to_f32()
        );
    }

    #[test]
    fn end_condition_k_factors() {
        assert_eq!(ColumnEndCondition::PinPin.k_factor(), Fix128::ONE);
        assert_eq!(
            ColumnEndCondition::FixedFixed.k_factor(),
            Fix128::from_ratio(1, 2)
        );
        assert_eq!(
            ColumnEndCondition::Cantilever.k_factor(),
            Fix128::from_int(2)
        );
        assert_eq!(
            ColumnEndCondition::FixedPin.k_factor(),
            Fix128::from_ratio(7, 10)
        );
    }

    #[test]
    fn analyze_safe_pla_beam() {
        // PLA 10×20 rectangular, 200mm cantilever, 5N load — very safe
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(20),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(5),
            length_mm: Fix128::from_int(200),
        };
        let ba = BeamAnalysis::new(s, load, MaterialProperties::pla());
        let report = ba.analyze();
        // σ = M/Z = 1000 / 666.67 = 1.5 MPa. Yield = 50 MPa → FoS ≈ 33
        assert!(report.factor_of_safety > Fix128::from_int(20));
        assert!(report.is_safe);
    }

    #[test]
    fn analyze_unsafe_overloaded_beam() {
        // PLA 10×5 rect, 300mm, 100N — should be unsafe
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(5),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(100),
            length_mm: Fix128::from_int(300),
        };
        let ba = BeamAnalysis::new(s, load, MaterialProperties::pla());
        let report = ba.analyze();
        // σ = 100·300 / (10·25/6) = 30000 / 41.67 = 720 MPa vs PLA 50 MPa
        // FoS = 50/720 = 0.07 (very unsafe)
        assert!(report.factor_of_safety < Fix128::ONE);
        assert!(!report.is_safe);
        assert!(report.max_bending_stress_mpa > Fix128::from_int(100));
    }

    #[test]
    fn analyze_with_end_condition_override() {
        let s = CrossSection::Rectangular {
            width_mm: Fix128::from_int(10),
            height_mm: Fix128::from_int(10),
        };
        let load = LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(50),
            length_mm: Fix128::from_int(200),
        };
        let ba = BeamAnalysis::new(s, load, MaterialProperties::pla())
            .with_end_condition(ColumnEndCondition::Cantilever)
            .with_min_fos(Fix128::from_ratio(15, 10));
        let report = ba.analyze();
        // Cantilever K=2 → lower P_cr but doesn't affect bending FoS directly.
        // Just verify override took effect via smaller Euler load
        let baseline = BeamAnalysis::new(s, load, MaterialProperties::pla());
        assert!(report.euler_critical_load_n < baseline.analyze().euler_critical_load_n);
    }

    #[test]
    fn hollow_beam_lighter_than_solid_same_stiffness() {
        // Solid 20×20 vs hollow 20×20 wall 3mm — hollow should have less area
        // but retain most of I (since I ~ h³·b, the removed material is near axis)
        let solid = CrossSection::Rectangular {
            width_mm: Fix128::from_int(20),
            height_mm: Fix128::from_int(20),
        };
        let hollow = CrossSection::HollowRectangular {
            outer_width_mm: Fix128::from_int(20),
            outer_height_mm: Fix128::from_int(20),
            wall_mm: Fix128::from_int(3),
        };
        assert!(hollow.area_mm2() < solid.area_mm2());
        // I ratio: solid = 13333, hollow = 13333 - (14·14³/12) = 13333 - 3201 = 10132
        // → hollow retains ~76% of I with ~36% of the area (efficiency)
        let ratio_i = hollow.second_moment_of_area_mm4() / solid.second_moment_of_area_mm4();
        let ratio_a = hollow.area_mm2() / solid.area_mm2();
        assert!(ratio_i > ratio_a, "hollow section is more area-efficient");
    }

    #[test]
    fn distributed_vs_point_load_comparison() {
        // Same total load, distributed produces half the moment on cantilever
        // (F on end vs same F spread → w = F/L → M_dist = F·L/2 = M_point/2)
        let l = Fix128::from_int(400);
        let f = Fix128::from_int(80);
        let point = LoadCase::CantileverEndPoint {
            load_n: f,
            length_mm: l,
        };
        let dist = LoadCase::CantileverDistributed {
            load_per_mm_n: f / l,
            length_mm: l,
        };
        let m_point = point.max_bending_moment_nmm();
        let m_dist = dist.max_bending_moment_nmm();
        // M_dist should be M_point / 2
        let ratio = m_point / m_dist;
        assert!(approx_eq(
            ratio,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }
}
