//! 3D Print Full Safety Pipeline (Session 3 S3)
//!
//! Aggregates every Session 1-2 print-safety check into a single
//! one-shot analysis, meant to be run before 3MF export. Callers supply
//! the geometry (footprint + optional cross section), material name, and
//! optional load / environment overrides; the solver returns a combined
//! `PrintSafetyReport` structured for CLI display or automated gating.
//!
//! # Checks performed
//!
//! 1. **Material lookup** (`filament_db`)
//! 2. **Warp risk** (`warp_risk`)
//! 3. **Effective layer strength** (`layer_adhesion`)
//! 4. **Optimal print orientation** (`print_orientation`)
//! 5. **Thermal stress at operating temp** (`thermal_stress`)
//! 6. **Beam stress** (if a load case is supplied) (`beam_stress`)
//! 7. **Bridging distance** (if bridge spans supplied) (`bridging`)
//! 8. **Support volume** (if overhang regions supplied) (`support_volume`)
//! 9. **Fillet stress concentration** (if radius / diameter supplied)
//!    (`fillet_stress`)
//! 10. **Bimaterial thermal residual** (if two materials supplied)
//!     (`bimaterial`)

use crate::beam_stress::{BeamAnalysis, BeamReport, CrossSection, LoadCase};
use crate::bimaterial::{analyze_bimaterial, BimaterialReport, BimaterialSide};
use crate::bridging::{analyze_bridges, BridgeSpan, BridgingReport};
use crate::filament_db::{FilamentDb, MaterialProperties};
use crate::fillet_stress::kt_shaft_shoulder_bending;
use crate::layer_adhesion::{EffectiveStrength, PrintOrientation};
use crate::math::Fix128;
use crate::print_orientation::{optimize_analytical, LoadDirection, OrientationReport};
use crate::support_volume::{
    estimate_support_volume, OverhangRegion, SupportConfig, SupportVolumeReport,
};
use crate::thermal_stress::{analyze_thermal_stress, ThermalStressReport};
use crate::warp_risk::{
    analyze_warp_risk, EnvConditions, Footprint, WarpRiskCategory, WarpRiskReport,
};

/// Optional inputs (each can be omitted to skip the corresponding check).
#[derive(Debug, Clone, Default)]
pub struct PrintPipelineInputs {
    /// Beam load case for structural stress check.
    pub beam_load: Option<(CrossSection, LoadCase)>,
    /// Load direction for orientation optimisation.
    pub load_direction: Option<LoadDirection>,
    /// Bridge spans requiring unsupported extrusion.
    pub bridges: Vec<BridgeSpan>,
    /// Overhang regions requiring support material.
    pub overhangs: Vec<OverhangRegion>,
    /// Support configuration (defaults if absent).
    pub support_config: Option<SupportConfig>,
    /// Fillet dimensions for stress concentration `K_t` check
    /// `(fillet_radius_mm, small_dia_mm, large_dia_mm)`.
    pub fillet: Option<(Fix128, Fix128, Fix128)>,
    /// Secondary material for bimaterial residual check (with layer thickness).
    pub secondary_material: Option<(&'static str, Fix128, Fix128)>,
    /// Operating temperature (°C) for thermal stress; default 20.
    pub operating_temp_c: Option<Fix128>,
}

/// Aggregated safety verdict.
#[derive(Debug, Clone)]
pub struct PrintSafetyReport {
    /// Primary material name.
    pub material_name: &'static str,
    /// Warp risk categorisation.
    pub warp: WarpRiskReport,
    /// Effective 6-component strength envelope.
    pub effective_strength: EffectiveStrength,
    /// Best orientation from `print_orientation` (if load_direction supplied).
    pub orientation: Option<OrientationReport>,
    /// Thermal stress at cool-down (print → operating).
    pub thermal_stress: ThermalStressReport,
    /// Beam stress result (if beam_load supplied).
    pub beam: Option<BeamReport>,
    /// Bridge-span check.
    pub bridging: Option<BridgingReport>,
    /// Support-volume estimation.
    pub support: Option<SupportVolumeReport>,
    /// Fillet K_t (if fillet supplied).
    pub fillet_kt: Option<Fix128>,
    /// Bimaterial interface report.
    pub bimaterial: Option<BimaterialReport>,
    /// True iff no CRITICAL warnings raised.
    pub is_safe: bool,
    /// Ordered list of human-readable warning messages.
    pub messages: Vec<String>,
}

impl PrintSafetyReport {
    /// Print a compact CLI summary.
    pub fn print(&self) {
        println!("=== Print Safety Pipeline Report ===");
        println!("  Material: {}", self.material_name);
        println!(
            "  Warp: {:?} (score {:.2})",
            self.warp.category,
            self.warp.score.to_f32()
        );
        println!(
            "  Strength envelope: X={:.1} Y={:.1} Z={:.1} τ_xy={:.1}",
            self.effective_strength.normal_x_mpa.to_f32(),
            self.effective_strength.normal_y_mpa.to_f32(),
            self.effective_strength.normal_z_mpa.to_f32(),
            self.effective_strength.shear_xy_mpa.to_f32(),
        );
        println!(
            "  Thermal stress: {:.2} MPa (FoS {:.1})",
            self.thermal_stress.thermal_stress_mpa.to_f32(),
            self.thermal_stress.factor_of_safety.to_f32(),
        );
        if let Some(o) = &self.orientation {
            println!(
                "  Best orientation angle_to_Z: {:.2} rad (yield {:.1} MPa, +{:.1})",
                o.angle_to_z_axis.to_f32(),
                o.effective_yield_mpa.to_f32(),
                o.improvement_mpa.to_f32()
            );
        }
        if let Some(b) = &self.beam {
            println!(
                "  Beam: σ={:.2} MPa, FoS={:.2}",
                b.max_bending_stress_mpa.to_f32(),
                b.factor_of_safety.to_f32()
            );
        }
        if let Some(br) = &self.bridging {
            println!(
                "  Bridging: {}/{} unsafe, max span {:.1} mm",
                br.unsafe_count,
                br.checks.len(),
                br.max_length_mm.to_f32()
            );
        }
        if let Some(sv) = &self.support {
            println!(
                "  Support volume: {:.0} mm³, filament {:.0} mm, time {:.0} min",
                sv.total_volume_mm3.to_f32(),
                sv.filament_length_mm.to_f32(),
                sv.estimated_time_min.to_f32()
            );
        }
        if let Some(kt) = self.fillet_kt {
            println!("  Fillet K_t: {:.2}", kt.to_f32());
        }
        if let Some(bm) = &self.bimaterial {
            println!(
                "  Bimaterial: σ_residual={:.2} MPa (FoS {:.1})",
                bm.thermal_residual_mpa.to_f32(),
                bm.factor_of_safety.to_f32()
            );
        }
        for m in &self.messages {
            println!("  {}", m);
        }
        println!(
            "  Overall: {}",
            if self.is_safe { "SAFE" } else { "UNSAFE" }
        );
    }
}

/// Look up a material or fall back to PLA.
fn pick_material(name: &str) -> MaterialProperties {
    FilamentDb::with_defaults()
        .find_by_name(name)
        .copied()
        .unwrap_or_else(MaterialProperties::pla)
}

/// Run the full print-safety pipeline for a given footprint and material.
#[must_use]
pub fn analyze_print_pipeline(
    footprint: Footprint,
    material_name: &str,
    inputs: &PrintPipelineInputs,
) -> PrintSafetyReport {
    let material = pick_material(material_name);
    let mut messages = Vec::new();
    let mut is_safe = true;

    // 1. Warp risk
    let warp = analyze_warp_risk(&footprint, &material, &EnvConditions::open_air_pla());
    match warp.category {
        WarpRiskCategory::Critical | WarpRiskCategory::High => {
            messages.push(format!("Warp {:?}: {}", warp.category, warp.recommendation));
            is_safe = false;
        }
        WarpRiskCategory::Medium => {
            messages.push(format!("Warp {:?}: {}", warp.category, warp.recommendation));
        }
        WarpRiskCategory::Low => {}
    }

    // 2. Effective strength envelope
    let effective_strength = EffectiveStrength::for_material(&material, PrintOrientation::XYFlat);

    // 3. Orientation optimisation
    let orientation = inputs
        .load_direction
        .map(|dir| optimize_analytical(&dir, &material));

    // 4. Thermal stress
    let op_temp = inputs
        .operating_temp_c
        .unwrap_or_else(|| Fix128::from_int(20));
    let thermal_stress = analyze_thermal_stress(
        &material,
        Fix128::from_ratio(3, 10),
        material.print_temp_c,
        op_temp,
    );
    if thermal_stress.near_glass_transition {
        messages.push("Operating temperature near Tg".to_string());
    }

    // 5. Beam analysis
    let beam = inputs.beam_load.map(|(section, load)| {
        let analysis = BeamAnalysis::new(section, load, material);
        let r = analysis.analyze();
        if !r.is_safe {
            messages.push(format!("Beam FoS {:.2}", r.factor_of_safety.to_f32()));
            is_safe = false;
        }
        r
    });

    // 6. Bridging
    let bridging = if !inputs.bridges.is_empty() {
        let br = analyze_bridges(&inputs.bridges, &material);
        if br.has_unsafe() {
            messages.push(format!(
                "{} bridge spans exceed material limit",
                br.unsafe_count
            ));
            is_safe = false;
        }
        Some(br)
    } else {
        None
    };

    // 7. Support volume
    let support = if !inputs.overhangs.is_empty() {
        let cfg = inputs.support_config.unwrap_or_default();
        Some(estimate_support_volume(&inputs.overhangs, &cfg))
    } else {
        None
    };

    // 8. Fillet K_t
    let fillet_kt = inputs
        .fillet
        .map(|(r, d, dbig)| kt_shaft_shoulder_bending(r, d, dbig));
    if let Some(kt) = fillet_kt {
        if kt > Fix128::from_int(2) {
            messages.push(format!("Fillet K_t {:.2} > 2 — enlarge R", kt.to_f32()));
        }
    }

    // 9. Bimaterial
    let bimaterial = inputs.secondary_material.map(|(name2, t1, t2)| {
        let m2 = pick_material(name2);
        let a = BimaterialSide::from_material(material, t1);
        let b = BimaterialSide::from_material(m2, t2);
        let report = analyze_bimaterial(&a, &b, material.print_temp_c, op_temp, Fix128::ZERO);
        if !report.is_safe {
            messages.push(format!(
                "Bimaterial FoS {:.1}",
                report.factor_of_safety.to_f32()
            ));
        }
        report
    });

    PrintSafetyReport {
        material_name: material.name,
        warp,
        effective_strength,
        orientation,
        thermal_stress,
        beam,
        bridging,
        support,
        fillet_kt,
        bimaterial,
        is_safe,
        messages,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3Fix;

    fn small_footprint() -> Footprint {
        Footprint {
            area_mm2: Fix128::from_int(1000),
            max_dimension_mm: Fix128::from_int(50),
        }
    }

    fn large_footprint() -> Footprint {
        Footprint {
            area_mm2: Fix128::from_int(70_000),
            max_dimension_mm: Fix128::from_int(280),
        }
    }

    #[test]
    fn small_pla_defaults_safe() {
        let r = analyze_print_pipeline(small_footprint(), "PLA", &PrintPipelineInputs::default());
        assert!(r.is_safe);
        assert_eq!(r.material_name, "PLA");
    }

    #[test]
    fn large_abs_flagged() {
        let r = analyze_print_pipeline(large_footprint(), "ABS", &PrintPipelineInputs::default());
        assert!(!r.is_safe || !r.messages.is_empty());
    }

    #[test]
    fn beam_load_included_in_report() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.beam_load = Some((
            CrossSection::Rectangular {
                width_mm: Fix128::from_int(10),
                height_mm: Fix128::from_int(20),
            },
            LoadCase::CantileverEndPoint {
                load_n: Fix128::from_int(5),
                length_mm: Fix128::from_int(200),
            },
        ));
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.beam.is_some());
    }

    #[test]
    fn orientation_optimized_when_load_direction_supplied() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.load_direction = Some(LoadDirection::axis_z());
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.orientation.is_some());
        assert!(r.orientation.unwrap().improvement_mpa > Fix128::ZERO);
    }

    #[test]
    fn bridging_check_flags_over_limit() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.bridges = vec![
            BridgeSpan {
                start: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
                end: Vec3Fix::new(Fix128::from_int(30), Fix128::ZERO, Fix128::ZERO),
            }, // 30mm > PLA 20mm
        ];
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.bridging.is_some());
        assert!(!r.is_safe);
    }

    #[test]
    fn support_volume_reported() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.overhangs = vec![OverhangRegion {
            projected_area_mm2: Fix128::from_int(100),
            support_height_mm: Fix128::from_int(20),
        }];
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.support.is_some());
        assert!(r.support.unwrap().total_volume_mm3 > Fix128::ZERO);
    }

    #[test]
    fn fillet_kt_reported() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.fillet = Some((
            Fix128::from_ratio(2, 10),
            Fix128::from_int(20),
            Fix128::from_int(40),
        ));
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.fillet_kt.is_some());
    }

    #[test]
    fn bimaterial_reported() {
        let mut inputs = PrintPipelineInputs::default();
        inputs.secondary_material = Some(("PETG", Fix128::from_int(1), Fix128::from_int(1)));
        let r = analyze_print_pipeline(small_footprint(), "PLA", &inputs);
        assert!(r.bimaterial.is_some());
    }

    #[test]
    fn unknown_material_defaults_to_pla() {
        let r = analyze_print_pipeline(
            small_footprint(),
            "Unobtainium",
            &PrintPipelineInputs::default(),
        );
        assert_eq!(r.material_name, "PLA");
    }
}
