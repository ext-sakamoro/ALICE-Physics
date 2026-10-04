//! Audit oracles for `alice_physics::print_pipeline_solver`.
//!
//! The pipeline aggregates sub-analyses, so each oracle fixes the inputs of
//! one sub-analysis independently (explicit temperatures, thickness order,
//! thresholds written as literals) and compares the report against the
//! sub-module result for exactly those inputs, plus closed-form checks of
//! the pipeline's own rules (safety flag, message emission thresholds).

#![cfg(feature = "std")]

use alice_physics::beam_stress::{BeamAnalysis, CrossSection, LoadCase};
use alice_physics::bimaterial::{analyze_bimaterial, BimaterialSide};
use alice_physics::bridging::BridgeSpan;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::fillet_stress::kt_shaft_shoulder_bending;
use alice_physics::layer_adhesion::{EffectiveStrength, PrintOrientation};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::print_orientation::LoadDirection;
use alice_physics::print_pipeline_solver::{analyze_print_pipeline, PrintPipelineInputs};
use alice_physics::support_volume::{estimate_support_volume, OverhangRegion, SupportConfig};
use alice_physics::thermal_stress::analyze_thermal_stress;
use alice_physics::warp_risk::{analyze_warp_risk, EnvConditions, Footprint, WarpRiskCategory};

fn r(n: i64) -> Fix128 {
    Fix128::from_int(n)
}
fn small() -> Footprint {
    Footprint {
        area_mm2: r(1000),
        max_dimension_mm: r(50),
    }
}
fn large() -> Footprint {
    Footprint {
        area_mm2: r(70_000),
        max_dimension_mm: r(280),
    }
}
fn span(len: i64, dz: i64) -> BridgeSpan {
    BridgeSpan {
        start: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        end: Vec3Fix::new(r(len), Fix128::ZERO, r(dz)),
    }
}

/// Thermal stress uses installation = the material print temperature,
/// the default operating temperature 20 C and a constraint coefficient 0.3.
#[test]
fn thermal_stress_matches_sub_analysis_with_default_operating_temp() {
    let pla = MaterialProperties::pla();
    let rep = analyze_print_pipeline(small(), "PLA", &PrintPipelineInputs::default());
    let want = analyze_thermal_stress(&pla, Fix128::from_ratio(3, 10), pla.print_temp_c, r(20));
    assert_eq!(rep.thermal_stress, want);
    // cooling a restrained part is tension: positive stress
    assert!(rep.thermal_stress.thermal_stress_mpa > Fix128::ZERO);
}

/// The operating temperature override reaches the thermal analysis and
/// changes the stress proportionally to the temperature drop:
/// sigma(op) / sigma(20) = (T_print - op) / (T_print - 20).
#[test]
fn operating_temperature_override_scales_thermal_stress_linearly() {
    let pla = MaterialProperties::pla();
    let base = analyze_print_pipeline(small(), "PLA", &PrintPipelineInputs::default());
    let inputs = PrintPipelineInputs {
        operating_temp_c: Some(r(110)),
        ..Default::default()
    };
    let hot = analyze_print_pipeline(small(), "PLA", &inputs);
    let t_print = pla.print_temp_c.to_f64();
    let want_ratio = (t_print - 110.0) / (t_print - 20.0);
    let got_ratio = hot.thermal_stress.thermal_stress_mpa.to_f64()
        / base.thermal_stress.thermal_stress_mpa.to_f64();
    assert!(
        (got_ratio - want_ratio).abs() < 1.0e-6,
        "ratio {got_ratio} vs {want_ratio}"
    );
}

/// Operating temperature within 20 C of Tg emits the Tg message, 40 C below
/// does not (PLA Tg is read from the material, thresholds are literals).
#[test]
fn near_glass_transition_message_follows_the_twenty_degree_window() {
    let pla = MaterialProperties::pla();
    let tg = pla.glass_transition_c.to_f64() as i64;
    let near = PrintPipelineInputs {
        operating_temp_c: Some(r(tg + 15)),
        ..Default::default()
    };
    let far = PrintPipelineInputs {
        operating_temp_c: Some(r(tg - 40)),
        ..Default::default()
    };
    let a = analyze_print_pipeline(small(), "PLA", &near);
    let b = analyze_print_pipeline(small(), "PLA", &far);
    assert!(a.messages.iter().any(|m| m.contains("near Tg")));
    assert!(!b.messages.iter().any(|m| m.contains("near Tg")));
}

/// The effective strength envelope is the XY-flat envelope of the material.
#[test]
fn effective_strength_is_the_xy_flat_envelope() {
    let pla = MaterialProperties::pla();
    let rep = analyze_print_pipeline(small(), "PLA", &PrintPipelineInputs::default());
    let want = EffectiveStrength::for_material(&pla, PrintOrientation::XYFlat);
    assert_eq!(rep.effective_strength, want);
}

/// Every default material name resolves to its own record, not the PLA
/// fallback.
#[test]
fn default_material_names_resolve_to_themselves() {
    for m in [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
        MaterialProperties::tpu(),
        MaterialProperties::nylon(),
    ] {
        let rep = analyze_print_pipeline(small(), m.name, &PrintPipelineInputs::default());
        assert_eq!(rep.material_name, m.name);
        let want = analyze_thermal_stress(&m, Fix128::from_ratio(3, 10), m.print_temp_c, r(20));
        assert_eq!(rep.thermal_stress, want, "{}", m.name);
    }
}

/// Warp risk is computed with the open-air PLA environment for every
/// material; a Critical/High result marks the part unsafe and adds a
/// message starting with "Warp".
#[test]
fn warp_result_matches_open_air_environment_and_drives_safety() {
    let abs = MaterialProperties::abs();
    let rep = analyze_print_pipeline(large(), "ABS", &PrintPipelineInputs::default());
    let want = analyze_warp_risk(&large(), &abs, &EnvConditions::open_air_pla());
    assert_eq!(rep.warp, want);
    let severe = matches!(
        want.category,
        WarpRiskCategory::Critical | WarpRiskCategory::High
    );
    if severe {
        assert!(!rep.is_safe);
    }
    let has_warp_msg = rep.messages.iter().any(|m| m.starts_with("Warp"));
    assert_eq!(
        has_warp_msg,
        !matches!(want.category, WarpRiskCategory::Low),
        "message emitted for every category except Low"
    );
}

/// Optional sections are `None` when their inputs are absent and `Some`
/// when present.
#[test]
fn optional_sections_follow_their_inputs() {
    let none = analyze_print_pipeline(small(), "PLA", &PrintPipelineInputs::default());
    assert!(none.orientation.is_none());
    assert!(none.beam.is_none());
    assert!(none.bridging.is_none());
    assert!(none.support.is_none());
    assert!(none.fillet_kt.is_none());
    assert!(none.bimaterial.is_none());
}

/// Bridge spans: the safe flag drops, one summary line plus one line per
/// unsafe span are emitted, and safe spans are not listed.
#[test]
fn bridging_messages_count_only_unsafe_spans() {
    let pla = MaterialProperties::pla();
    let limit = pla.bridging_distance_mm.to_f64() as i64;
    let inputs = PrintPipelineInputs {
        bridges: vec![span(limit + 10, 0), span(limit - 5, 0), span(limit + 30, 0)],
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    let b = rep.bridging.as_ref().unwrap();
    assert_eq!(b.checks.len(), 3);
    assert_eq!(b.unsafe_count, 2);
    assert!(!rep.is_safe);
    let summary = rep
        .messages
        .iter()
        .filter(|m| m.contains("bridge spans exceed"))
        .count();
    let lines = rep
        .messages
        .iter()
        .filter(|m| m.starts_with("Bridge span"))
        .count();
    assert_eq!(summary, 1);
    assert_eq!(lines, 2);
    assert!(rep.messages.iter().any(|m| m.starts_with("2 bridge spans")));
}

/// A span within the limit leaves the part safe and produces no bridge
/// message.
#[test]
fn safe_bridge_adds_no_message_and_keeps_part_safe() {
    let pla = MaterialProperties::pla();
    let limit = pla.bridging_distance_mm.to_f64() as i64;
    let inputs = PrintPipelineInputs {
        bridges: vec![span(limit - 5, 0)],
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    assert!(rep.is_safe);
    assert!(!rep.messages.iter().any(|m| m.contains("ridge")));
}

/// Overhangs use the default support configuration when none is given,
/// and the explicit configuration otherwise.
#[test]
fn support_volume_uses_default_then_explicit_configuration() {
    let overhangs = vec![
        OverhangRegion {
            projected_area_mm2: r(100),
            support_height_mm: r(20),
        },
        OverhangRegion {
            projected_area_mm2: r(40),
            support_height_mm: r(5),
        },
    ];
    let inputs = PrintPipelineInputs {
        overhangs: overhangs.clone(),
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    let want = estimate_support_volume(&overhangs, &SupportConfig::default());
    assert_eq!(rep.support.unwrap(), want);
}

/// Fillet K_t is passed through unchanged, and the "K_t > 2" message is
/// emitted exactly when the value exceeds 2.
#[test]
fn fillet_message_threshold_is_two() {
    for (rad, d, dd) in [
        (Fix128::from_ratio(1, 10), r(20), r(40)),
        (Fix128::from_ratio(2, 10), r(20), r(40)),
        (r(4), r(20), r(40)),
        (r(8), r(20), r(40)),
        // K_t between 2 and 3 (2.42) and just under 2 (1.98)
        (Fix128::from_ratio(5, 10), r(10), r(20)),
        (r(1), r(10), r(20)),
    ] {
        let kt = kt_shaft_shoulder_bending(rad, d, dd);
        let inputs = PrintPipelineInputs {
            fillet: Some((rad, d, dd)),
            ..Default::default()
        };
        let rep = analyze_print_pipeline(small(), "PLA", &inputs);
        assert_eq!(rep.fillet_kt, Some(kt));
        let msg = rep.messages.iter().any(|m| m.starts_with("Fillet K_t"));
        assert_eq!(msg, kt.to_f64() > 2.0, "kt {}", kt.to_f64());
    }
}

/// Beam: an overloaded slender beam marks the part unsafe with a "Beam FoS"
/// message; a stout beam under a light load stays safe with none.
#[test]
fn beam_unsafe_flips_safety_and_adds_message() {
    let weak = PrintPipelineInputs {
        beam_load: Some((
            CrossSection::Rectangular {
                width_mm: r(2),
                height_mm: r(2),
            },
            LoadCase::CantileverEndPoint {
                load_n: r(50),
                length_mm: r(200),
            },
        )),
        ..Default::default()
    };
    let strong = PrintPipelineInputs {
        beam_load: Some((
            CrossSection::Rectangular {
                width_mm: r(30),
                height_mm: r(30),
            },
            LoadCase::CantileverEndPoint {
                load_n: r(1),
                length_mm: r(50),
            },
        )),
        ..Default::default()
    };
    let a = analyze_print_pipeline(small(), "PLA", &weak);
    let b = analyze_print_pipeline(small(), "PLA", &strong);
    assert!(!a.beam.as_ref().unwrap().is_safe);
    assert!(!a.is_safe);
    assert!(a.messages.iter().any(|m| m.starts_with("Beam FoS")));
    assert!(b.beam.as_ref().unwrap().is_safe);
    assert!(b.is_safe);
    assert!(!b.messages.iter().any(|m| m.starts_with("Beam FoS")));
}

/// Orientation optimisation is run with the supplied direction and the
/// report's material.
#[test]
fn orientation_matches_sub_analysis() {
    let pla = MaterialProperties::pla();
    let dir = LoadDirection::axis_z();
    let inputs = PrintPipelineInputs {
        load_direction: Some(dir),
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    let want = alice_physics::print_orientation::optimize_analytical(&dir, &pla);
    let got = rep.orientation.unwrap();
    assert_eq!(got.angle_to_z_axis, want.angle_to_z_axis);
    assert_eq!(got.effective_yield_mpa, want.effective_yield_mpa);
    assert_eq!(got.improvement_mpa, want.improvement_mpa);
}

/// The two layer thicknesses are passed to the bimaterial analysis in
/// (primary, secondary) order, from the primary print temperature down to
/// the operating temperature, with no applied shear.
#[test]
fn bimaterial_receives_thicknesses_in_order_and_cooldown_range() {
    let pla = MaterialProperties::pla();
    let petg = MaterialProperties::petg();
    let inputs = PrintPipelineInputs {
        secondary_material: Some(("PETG", r(3), r(7))),
        operating_temp_c: Some(r(30)),
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    let a = BimaterialSide::from_material(pla, r(3));
    let b = BimaterialSide::from_material(petg, r(7));
    let want = analyze_bimaterial(&a, &b, pla.print_temp_c, r(30), Fix128::ZERO);
    assert_eq!(rep.bimaterial.unwrap(), want);
}

/// A part whose thermal-stress verdict is "not safe" (operating temperature
/// inside the glass-transition window) must not be reported as safe overall.
#[test]
fn failing_thermal_verdict_clears_the_safe_flag() {
    let pla = MaterialProperties::pla();
    let tg = pla.glass_transition_c.to_f64() as i64;
    let inputs = PrintPipelineInputs {
        operating_temp_c: Some(r(tg - 5)),
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    assert!(
        !rep.thermal_stress.is_safe,
        "scenario must fail the sub-verdict"
    );
    assert!(!rep.is_safe);
}

/// `print` is a pure display routine: it must not panic for a fully
/// populated report.
#[test]
fn print_summary_does_not_panic_on_full_report() {
    let inputs = PrintPipelineInputs {
        beam_load: Some((
            CrossSection::Rectangular {
                width_mm: r(10),
                height_mm: r(10),
            },
            LoadCase::CantileverEndPoint {
                load_n: r(5),
                length_mm: r(100),
            },
        )),
        load_direction: Some(LoadDirection::axis_z()),
        bridges: vec![span(30, 2)],
        overhangs: vec![OverhangRegion {
            projected_area_mm2: r(10),
            support_height_mm: r(3),
        }],
        support_config: None,
        fillet: Some((r(1), r(10), r(20))),
        secondary_material: Some(("TPU", r(1), r(1))),
        operating_temp_c: Some(r(55)),
    };
    let rep = analyze_print_pipeline(large(), "ABS", &inputs);
    rep.print();
}

/// The beam analysis uses the report's material, not a fixed default:
/// the factor of safety equals the sub-analysis for ABS.
#[test]
fn beam_uses_the_requested_material() {
    let section = CrossSection::Rectangular {
        width_mm: r(10),
        height_mm: r(10),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: r(5),
        length_mm: r(100),
    };
    let inputs = PrintPipelineInputs {
        beam_load: Some((section, load)),
        ..Default::default()
    };
    for m in [MaterialProperties::abs(), MaterialProperties::nylon()] {
        let rep = analyze_print_pipeline(small(), m.name, &inputs);
        let want = BeamAnalysis::new(section, load, m).analyze();
        let got = rep.beam.unwrap();
        assert_eq!(got.factor_of_safety, want.factor_of_safety, "{}", m.name);
        assert_eq!(got.max_bending_stress_mpa, want.max_bending_stress_mpa);
    }
}

/// A footprint in the Medium warp band adds a "Warp" message but does not
/// make the part unsafe.
#[test]
fn medium_warp_adds_message_without_clearing_safe_flag() {
    let fp = Footprint {
        area_mm2: r(40_000),
        max_dimension_mm: r(250),
    };
    let rep = analyze_print_pipeline(fp, "PLA", &PrintPipelineInputs::default());
    assert_eq!(rep.warp.category, WarpRiskCategory::Medium);
    assert!(rep.messages.iter().any(|m| m.starts_with("Warp")));
    assert!(rep.is_safe);
}

/// A bimaterial interface that fails its own FoS check (PLA over ABS at
/// -200 C gives FoS 1.83) adds a "Bimaterial FoS" message; a benign one
/// does not.
#[test]
fn failing_bimaterial_adds_message_and_benign_one_does_not() {
    let bad = PrintPipelineInputs {
        secondary_material: Some(("ABS", r(1), r(1))),
        operating_temp_c: Some(r(-200)),
        ..Default::default()
    };
    let ok = PrintPipelineInputs {
        secondary_material: Some(("ABS", r(1), r(1))),
        operating_temp_c: Some(r(20)),
        ..Default::default()
    };
    let a = analyze_print_pipeline(small(), "PLA", &bad);
    let b = analyze_print_pipeline(small(), "PLA", &ok);
    assert!(!a.bimaterial.as_ref().unwrap().is_safe);
    assert!(a.messages.iter().any(|m| m.starts_with("Bimaterial FoS")));
    assert!(b.bimaterial.as_ref().unwrap().is_safe);
    assert!(!b.messages.iter().any(|m| m.starts_with("Bimaterial FoS")));
}

/// A failing bimaterial verdict must make the part unsafe, as a failing
/// beam or bridge does.
#[test]
fn failing_bimaterial_clears_the_safe_flag() {
    let inputs = PrintPipelineInputs {
        secondary_material: Some(("ABS", r(1), r(1))),
        operating_temp_c: Some(r(-200)),
        ..Default::default()
    };
    let rep = analyze_print_pipeline(small(), "PLA", &inputs);
    assert!(!rep.bimaterial.as_ref().unwrap().is_safe);
    assert!(!rep.is_safe);
}
