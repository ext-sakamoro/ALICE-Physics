//! Engineering-module oracles, group A (solid mechanics / print safety).
//!
//! "Deterministic" and "correct" are different properties. Every test here
//! compares a module against a textbook closed form or worked example with
//! a citation, and states the tolerance and where it comes from (Fix128
//! rounding, a rounded constant in the module, f32 sphere tracing, …).
//!
//! Modules that are empirical heuristics (calibrated lookup tables, curated
//! score bands) get invariant tests only (monotonicity, symmetry, limits);
//! they are listed as `validation: none` in the accompanying report.
//!
//! Tests whose name contains `discrepancy` document a place where the
//! module disagrees with a closed form the author is certain of; they are
//! written as failing assertions on purpose (they are *not* skipped).

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic::OrthotropicElasticity;
use alice_physics::beam_stress::{BeamAnalysis, ColumnEndCondition, CrossSection, LoadCase};
use alice_physics::bimaterial::{
    analyze_bimaterial, effective_modulus_reuss_mpa, effective_modulus_voigt_mpa,
    interfacial_bond_strength_mpa, thermal_residual_stress_mpa, BimaterialSide,
};
use alice_physics::bridging::{analyze_bridges, check_span, BridgeSpan};
use alice_physics::buckling::{analyze_column, BucklingRegime};
use alice_physics::creep_longterm::{predict_strain, FindleyParameters};
use alice_physics::damping_rayleigh::{hz_to_omega, omega_to_hz, RayleighCoefficients};
use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::fillet_stress::{
    kt_circular_hole_infinite_plate, kt_elliptical_hole, kt_shaft_shoulder_bending,
    kt_u_notch_axial, recommended_fillet_radius_mm,
};
use alice_physics::hyperelastic::{
    small_strain_shear_modulus, strain_energy_density, uniaxial_cauchy_stress, HyperelasticModel,
    Stretch,
};
use alice_physics::laminate::{compute_abd, is_symmetric_stack, Ply};
use alice_physics::laminate_failure::{
    hashin_failure_mode, puck_failure_mode, tsai_hill_failure_index, tsai_wu_failure_index,
    FailureMode, LaminateStrengths, StressState,
};
use alice_physics::mass_properties::{
    box_mass_properties, capsule_mass_properties, convex_hull_mass_properties,
    cylinder_mass_properties, sphere_mass_properties, translate_inertia,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::modal::{
    beam_natural_frequency_hz, plate_natural_frequency_hz, single_dof_frequency_hz,
    torsional_frequency_hz, BeamBoundary,
};
use alice_physics::plastic::{NortonCreep, PlasticModel, PlasticState};
use alice_physics::prestressed::{
    bolt_load_fraction, bolt_peak_tension, cable_pretension_n, preload_from_torque,
    recommended_preload_n, separation_load_n, tensioned_cable_stiffness_n_per_mm,
};
use alice_physics::print_orientation::{
    angle_to_z_axis, effective_yield_at_angle, optimize_analytical, optimize_grid, LoadDirection,
    OrientationCandidate,
};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::structural_solver::StructuralSolver;
use alice_physics::thermal_stress::{analyze_thermal_stress, yield_temperature_c};
use alice_physics::thin_wall::{measure_thickness_at, ThinWallConfig};
use alice_physics::vibration_wall::{analyze_wall_resonance, ExcitationSource};

use core::f64::consts::PI;

// ============================================================================
// Helpers
// ============================================================================

/// Relative-error assertion against an f64 oracle.
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = ((g - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

/// Absolute-error assertion against an f64 oracle.
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

/// Material with round-number properties so every oracle can be evaluated
/// by hand. `name` matters only for the CTE lookup in `bimaterial`.
fn material(
    name: &'static str,
    e_gpa: (i64, i64),
    yield_mpa: i64,
    uts_mpa: i64,
    density: (i64, i64),
    tg_c: i64,
    anisotropy: (i64, i64),
) -> MaterialProperties {
    MaterialProperties {
        id: 0,
        name,
        category: MaterialCategory::Fdm,
        youngs_modulus_gpa: Fix128::from_ratio(e_gpa.0, e_gpa.1),
        yield_strength_mpa: Fix128::from_int(yield_mpa),
        tensile_strength_mpa: Fix128::from_int(uts_mpa),
        density_g_cm3: Fix128::from_ratio(density.0, density.1),
        print_temp_c: Fix128::from_int(200),
        glass_transition_c: Fix128::from_int(tg_c),
        bridging_distance_mm: Fix128::from_int(20),
        shrinkage_ratio: Fix128::from_ratio(2, 1000),
        anisotropy_z_ratio: Fix128::from_ratio(anisotropy.0, anisotropy.1),
    }
}

/// PLA-like: E = 3.5 GPa, σ_y = 50 MPa, UTS = 60 MPa, ρ = 1.24, T_g = 60 °C,
/// Z ratio 0.65. Named "PLA" so `published_cte_per_c` returns 68e-6.
fn pla_like() -> MaterialProperties {
    material("PLA", (35, 10), 50, 60, (124, 100), 60, (65, 100))
}

/// Steel-like: E = 200 GPa, σ_y = 215, UTS = 505, ρ = 8.0, no T_g, isotropic.
/// Named "SUS304" so `published_cte_per_c` returns 17.3e-6.
fn steel_like() -> MaterialProperties {
    material("SUS304", (200, 1), 215, 505, (8, 1), 0, (1, 1))
}

// Fix128 has 64 fractional bits (≈ 5.4e-20). Products of a handful of
// engineering-magnitude values stay far below 1e-12 relative error; the
// module's π constant is exact to the same resolution.
const FIX_TOL: f64 = 1e-12;

// ============================================================================
// beam_stress — Roark Table 8.1 / Timoshenko & Gere
// ============================================================================

/// Roark's Formulas for Stress and Strain, 8th ed., Table 8.1 case 1a
/// (cantilever, end load): M_max = P L, δ_max = P L³ / (3 E I).
/// σ = M c / I with I = b h³ / 12 (Roark Appendix A, rectangle).
/// Euler: P_cr = π² E I / (K L)², Timoshenko & Gere §2.1, K = 1 pin-pin.
#[test]
fn beam_stress_cantilever_end_load_matches_roark() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(20),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(30),
        length_mm: Fix128::from_int(300),
    };
    let report = BeamAnalysis::new(section, load, pla_like()).analyze();

    let (b, h, l, p, e) = (10.0f64, 20.0f64, 300.0f64, 30.0f64, 3500.0f64);
    let i = b * h * h * h / 12.0; // 6666.67 mm⁴
    let z = i / (h / 2.0); // 666.67 mm³
    let m = p * l; // 9000 N·mm
    assert_rel(report.max_bending_moment_nmm, m, FIX_TOL, "M = P L");
    assert_rel(report.max_bending_stress_mpa, m / z, FIX_TOL, "σ = M / Z");
    assert_rel(
        report.max_deflection_mm,
        p * l * l * l / (3.0 * e * i),
        FIX_TOL,
        "δ = P L³ / (3 E I)",
    );
    assert_rel(
        report.euler_critical_load_n,
        PI * PI * e * i / (l * l),
        FIX_TOL,
        "P_cr = π² E I / L²",
    );
    assert_rel(
        report.factor_of_safety,
        50.0 / (m / z),
        FIX_TOL,
        "FoS = σ_y / σ",
    );
    assert!(report.is_safe, "FoS 3.70 ≥ 2.0");
}

/// Roark Table 8.1 case 2e (simply supported, uniform load):
/// M_max = w L² / 8, δ_max = 5 w L⁴ / (384 E I).
#[test]
fn beam_stress_simply_supported_uniform_load_matches_roark() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let load = LoadCase::SimplySupportedDistributed {
        load_per_mm_n: Fix128::ONE,
        length_mm: Fix128::from_int(100),
    };
    let e = Fix128::from_int(3500);
    let i = 10.0f64 * 1000.0 / 12.0;
    assert_rel(
        load.max_bending_moment_nmm(),
        1250.0,
        FIX_TOL,
        "M = w L² / 8",
    );
    assert_rel(
        load.max_deflection_mm(&section, e),
        5.0 * 1.0 * 1e8 / (384.0 * 3500.0 * i),
        FIX_TOL,
        "δ = 5 w L⁴ / (384 E I)",
    );
    // cantilever distributed: M = w L² / 2, δ = w L⁴ / (8 E I) (Roark case 1e)
    let cant = LoadCase::CantileverDistributed {
        load_per_mm_n: Fix128::ONE,
        length_mm: Fix128::from_int(100),
    };
    assert_rel(
        cant.max_bending_moment_nmm(),
        5000.0,
        FIX_TOL,
        "M = w L² / 2",
    );
    assert_rel(
        cant.max_deflection_mm(&section, e),
        1e8 / (8.0 * 3500.0 * i),
        FIX_TOL,
        "δ = w L⁴ / (8 E I)",
    );
    // simply supported centre load: δ = P L³ / (48 E I) (Roark case 2a)
    let centre = LoadCase::SimplySupportedCenter {
        load_n: Fix128::from_int(200),
        length_mm: Fix128::from_int(400),
    };
    assert_rel(
        centre.max_bending_moment_nmm(),
        20_000.0,
        FIX_TOL,
        "M = P L / 4",
    );
    assert_rel(
        centre.max_deflection_mm(&section, e),
        200.0 * 400.0f64.powi(3) / (48.0 * 3500.0 * i),
        FIX_TOL,
        "δ = P L³ / (48 E I)",
    );
}

/// Section properties against Roark Appendix A: I-beam by parallel-axis sum
/// of flanges + web, hollow circle π (D⁴ − d⁴) / 64, hollow rectangle.
#[test]
fn beam_stress_section_properties_match_roark_appendix() {
    let ibeam = CrossSection::IBeam {
        flange_width_mm: Fix128::from_int(100),
        height_mm: Fix128::from_int(200),
        flange_thickness_mm: Fix128::from_int(20),
        web_thickness_mm: Fix128::from_int(10),
    };
    let (b, h, t, tw) = (100.0f64, 200.0f64, 20.0f64, 10.0f64);
    let flange = b * t * t * t / 12.0 + b * t * ((h - t) / 2.0).powi(2);
    let web = tw * (h - 2.0 * t).powi(3) / 12.0;
    assert_rel(
        ibeam.second_moment_of_area_mm4(),
        2.0 * flange + web,
        FIX_TOL,
        "I-beam I by parallel axis",
    );
    assert_rel(
        ibeam.area_mm2(),
        2.0 * b * t + tw * (h - 2.0 * t),
        FIX_TOL,
        "I-beam A",
    );

    let tube = CrossSection::HollowCircular {
        outer_diameter_mm: Fix128::from_int(12),
        inner_diameter_mm: Fix128::from_int(8),
    };
    assert_rel(
        tube.second_moment_of_area_mm4(),
        PI * (12.0f64.powi(4) - 8.0f64.powi(4)) / 64.0,
        FIX_TOL,
        "hollow circle I",
    );
    assert_rel(
        tube.section_modulus_mm3(),
        PI * (12.0f64.powi(4) - 8.0f64.powi(4)) / 64.0 / 6.0,
        FIX_TOL,
        "hollow circle Z = I / (D/2)",
    );
    let rect_tube = CrossSection::HollowRectangular {
        outer_width_mm: Fix128::from_int(20),
        outer_height_mm: Fix128::from_int(30),
        wall_mm: Fix128::from_int(2),
    };
    assert_rel(
        rect_tube.second_moment_of_area_mm4(),
        (20.0 * 27000.0 - 16.0 * 26.0f64.powi(3)) / 12.0,
        FIX_TOL,
        "hollow rectangle I",
    );
    let circle = CrossSection::Circular {
        diameter_mm: Fix128::from_int(10),
    };
    assert_rel(circle.area_mm2(), PI * 25.0, FIX_TOL, "circle A = π d² / 4");
    assert_rel(
        circle.second_moment_of_area_mm4(),
        PI * 1e4 / 64.0,
        FIX_TOL,
        "circle I = π d⁴ / 64",
    );
}

// ============================================================================
// buckling — Shigley 10th ed. §4-12/4-13 (Euler + Johnson)
// ============================================================================

/// Shigley eq. 4-42 (Euler, σ_cr = π² E / (l/k)²), eq. 4-46 (Johnson,
/// σ_cr = S_y − (S_y l / (2π k))² / E), eq. 4-45 (transition
/// (l/k)₁ = √(2 π² E / S_y)), radius of gyration k = √(I/A).
#[test]
fn buckling_euler_and_johnson_regimes_match_shigley() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let m = pla_like();
    let (e, sy) = (3500.0f64, 50.0f64);
    let k = (10.0f64 * 1000.0 / 12.0 / 100.0).sqrt(); // 2.8868 mm
    let lambda_t = PI * (2.0 * e / sy).sqrt(); // 37.17

    // slender: L = 500 → λ = 173 > λ_t → Euler
    let slender = analyze_column(
        &section,
        Fix128::from_int(500),
        ColumnEndCondition::PinPin,
        &m,
    );
    let lam = 500.0 / k;
    assert_eq!(slender.regime, BucklingRegime::Euler);
    assert_rel(slender.radius_of_gyration_mm, k, FIX_TOL, "k = √(I/A)");
    assert_rel(slender.slenderness, lam, FIX_TOL, "λ = K L / k");
    assert_rel(slender.transition_slenderness, lambda_t, FIX_TOL, "λ_t");
    assert_rel(
        slender.critical_stress_mpa,
        PI * PI * e / (lam * lam),
        FIX_TOL,
        "Euler σ_cr",
    );
    assert_rel(
        slender.critical_load_n,
        PI * PI * e * (10.0 * 1000.0 / 12.0) / (500.0 * 500.0),
        FIX_TOL,
        "Euler P_cr = σ_cr A = π² E I / L²",
    );

    // stocky: L = 60 → λ = 20.8 < λ_t → Johnson parabola
    let stocky = analyze_column(
        &section,
        Fix128::from_int(60),
        ColumnEndCondition::PinPin,
        &m,
    );
    let lam = 60.0 / k;
    assert_eq!(stocky.regime, BucklingRegime::Johnson);
    assert_rel(
        stocky.critical_stress_mpa,
        sy - (sy * lam / (2.0 * PI)).powi(2) / e,
        FIX_TOL,
        "Johnson σ_cr",
    );

    // fixed-free K = 2 (Timoshenko & Gere Table 2-1): λ doubles
    let cant = analyze_column(
        &section,
        Fix128::from_int(500),
        ColumnEndCondition::Cantilever,
        &m,
    );
    assert_rel(cant.slenderness, 2.0 * 500.0 / k, FIX_TOL, "λ with K = 2");
    assert_rel(
        cant.critical_load_n,
        slender.critical_load_n.to_f64() / 4.0,
        FIX_TOL,
        "P_cr / 4",
    );

    // continuity at the transition: both branches give S_y / 2 (Shigley §4-13)
    let l_t = lambda_t * k;
    let at_t = analyze_column(
        &section,
        Fix128::from_f64(l_t),
        ColumnEndCondition::PinPin,
        &m,
    );
    // from_f64 lands on either side of λ_t; both branches agree to Fix128 rounding
    assert_rel(
        at_t.critical_stress_mpa,
        sy / 2.0,
        1e-9,
        "σ_cr(λ_t) = S_y / 2",
    );
}

// ============================================================================
// thin_wall — geometric closed forms (slab, spherical shell)
// ============================================================================

/// Wall thickness of a slab |x| ≤ h is 2h; of a spherical shell it is
/// R_out − R_in. Sphere tracing lands exactly on the opposite surface in
/// real arithmetic; the f32 march can undershoot by one `min_step` (0.001)
/// and the start offset (0.01) is counted, so tolerance 0.005 mm.
#[test]
fn thin_wall_thickness_matches_slab_and_shell_geometry() {
    let cfg = ThinWallConfig::default();
    let slab = ClosureSdf::new(|x, _y, _z| x.abs() - 1.0, |x, _, _| (x.signum(), 0.0, 0.0));
    let p = Vec3Fix::from_int(1, 0, 0);
    let t = measure_thickness_at(&slab, p, (1.0, 0.0, 0.0), &cfg).expect("slab is bounded");
    assert_abs(t, 2.0, 0.005, "slab thickness = 2h");

    let shell = ClosureSdf::new(
        |x, y, z| {
            let r = (x * x + y * y + z * z).sqrt();
            (r - 10.0).max(8.0 - r)
        },
        |x, y, z| {
            let r = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
            (x / r, y / r, z / r)
        },
    );
    let p = Vec3Fix::from_int(10, 0, 0);
    let t = measure_thickness_at(&shell, p, (1.0, 0.0, 0.0), &cfg).expect("shell is bounded");
    assert_abs(t, 2.0, 0.005, "shell thickness = R_out − R_in");
    // and a thinner shell, off-axis sample point (0, 0, 10) with normal +Z
    let thin = ClosureSdf::new(
        |x, y, z| {
            let r = (x * x + y * y + z * z).sqrt();
            (r - 10.0).max(9.5 - r)
        },
        |x, y, z| {
            let r = (x * x + y * y + z * z).sqrt().max(f32::EPSILON);
            (x / r, y / r, z / r)
        },
    );
    let p = Vec3Fix::from_int(0, 0, 10);
    let t = measure_thickness_at(&thin, p, (0.0, 0.0, 1.0), &cfg).expect("shell is bounded");
    assert_abs(t, 0.5, 0.005, "0.5 mm shell");
    assert!(t < cfg.min_thickness_mm, "0.5 mm < 0.8 mm nozzle rule");
}

// ============================================================================
// damping_rayleigh — Chopra 5th ed. §11.4
// ============================================================================

/// Chopra eq. 11.4.5: ζ_n = α / (2 ω_n) + β ω_n / 2. Chopra eq. 11.4.9 for
/// equal ζ at two modes: α = 2 ζ ω_i ω_j / (ω_i + ω_j), β = 2 ζ / (ω_i + ω_j).
#[test]
fn damping_rayleigh_matches_chopra() {
    let c = RayleighCoefficients {
        alpha: Fix128::from_ratio(8, 10),
        beta: Fix128::from_ratio(2, 1000),
    };
    // ω = 20: 0.8/40 + 0.002·20/2 = 0.02 + 0.02
    assert_rel(
        c.damping_ratio(Fix128::from_int(20)),
        0.04,
        FIX_TOL,
        "ζ(20)",
    );
    // minimum of ζ(ω) is at ω = √(α/β) = 20 → ζ_min = √(αβ) = 0.04
    assert!(c.damping_ratio(Fix128::from_int(10)) > Fix128::from_ratio(4, 100));
    assert!(c.damping_ratio(Fix128::from_int(40)) > Fix128::from_ratio(4, 100));

    let (w1, w2, zeta) = (2.0 * PI * 1.0, 2.0 * PI * 5.0, 0.05);
    let fit = RayleighCoefficients::fit_two_modes(
        hz_to_omega(Fix128::ONE),
        Fix128::from_ratio(5, 100),
        hz_to_omega(Fix128::from_int(5)),
        Fix128::from_ratio(5, 100),
    );
    assert_rel(
        fit.alpha,
        2.0 * zeta * w1 * w2 / (w1 + w2),
        1e-11,
        "α (Chopra 11.4.9)",
    );
    assert_rel(fit.beta, 2.0 * zeta / (w1 + w2), 1e-11, "β (Chopra 11.4.9)");
    assert_rel(
        fit.damping_ratio(hz_to_omega(Fix128::ONE)),
        zeta,
        1e-11,
        "ζ(ω₁)",
    );
    assert_rel(
        fit.damping_ratio(hz_to_omega(Fix128::from_int(5))),
        zeta,
        1e-11,
        "ζ(ω₂)",
    );
    // between the two fitted modes Rayleigh damping is below the target
    let w3 = 2.0 * PI * 3.0;
    let want = fit.alpha.to_f64() / (2.0 * w3) + fit.beta.to_f64() * w3 / 2.0;
    assert!(want < zeta);
    assert_rel(
        fit.damping_ratio(hz_to_omega(Fix128::from_int(3))),
        want,
        1e-11,
        "ζ(ω₃)",
    );
    // unit helpers are exact inverses
    assert_rel(
        omega_to_hz(hz_to_omega(Fix128::from_int(7))),
        7.0,
        FIX_TOL,
        "Hz ↔ rad/s",
    );
}

// ============================================================================
// laminate — Jones, Mechanics of Composite Materials 2nd ed.
// ============================================================================

/// T300/5208 graphite-epoxy (Jones Table 2.3): E₁ = 181 GPa, E₂ = 10.3 GPa,
/// ν₁₂ = 0.28, G₁₂ = 7.17 GPa. Values in MPa.
fn t300_5208() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(181_000),
        e_t_mpa: Fix128::from_int(10_300),
        e_z_mpa: Fix128::from_int(10_300),
        nu_lt: Fix128::from_ratio(28, 100),
        nu_lz: Fix128::from_ratio(28, 100),
        nu_tz: Fix128::from_ratio(40, 100),
        g_lt_mpa: Fix128::from_int(7170),
        g_lz_mpa: Fix128::from_int(7170),
        g_tz_mpa: Fix128::from_int(3000),
    }
}

/// Reduced stiffnesses (Jones eq. 2.61), f64 reference.
fn q_ref() -> (f64, f64, f64, f64) {
    let (e1, e2, nu12, g12) = (181_000.0f64, 10_300.0f64, 0.28f64, 7170.0f64);
    let nu21 = nu12 * e2 / e1;
    let d = 1.0 - nu12 * nu21;
    (e1 / d, nu12 * e2 / d, e2 / d, g12)
}

fn ply(angle_rad: Fix128, t_mm: Fix128) -> Ply {
    Ply {
        thickness_mm: t_mm,
        orientation_rad: angle_rad,
        material: t300_5208(),
    }
}

/// Jones eq. 2.61 (Q), Jones §4.3.2 cross-ply [0/90]_s closed forms:
/// A₁₁ = A₂₂ = 2t(Q₁₁+Q₂₂), A₁₂ = 4t Q₁₂, A₆₆ = 4t Q₆₆, B = 0,
/// D₁₁ = ⅔ t³ (7 Q₁₁ + Q₂₂), D₂₂ = ⅔ t³ (Q₁₁ + 7 Q₂₂), D₁₂ = 16/3 t³ Q₁₂.
#[test]
fn laminate_cross_ply_abd_matches_jones() {
    let (q11, q12, q22, q66) = q_ref();
    let p = ply(Fix128::ZERO, Fix128::from_ratio(1, 8));
    let (m11, m12, m22, m66) = p.q_matrix();
    assert_rel(m11, q11, FIX_TOL, "Q11 = E1/(1−ν12ν21)");
    assert_rel(m12, q12, FIX_TOL, "Q12 = ν12 E2/(1−ν12ν21)");
    assert_rel(m22, q22, FIX_TOL, "Q22");
    assert_rel(m66, q66, FIX_TOL, "Q66 = G12");

    let t = 0.125f64;
    let t_fix = Fix128::from_ratio(1, 8);
    let ninety = Fix128::HALF_PI;
    let stack = [
        ply(Fix128::ZERO, t_fix),
        ply(ninety, t_fix),
        ply(ninety, t_fix),
        ply(Fix128::ZERO, t_fix),
    ];
    assert!(is_symmetric_stack(&stack));
    let abd = compute_abd(&stack);
    // cos(π/2) from CORDIC is ~1e-19, so the 90° Q̄ is exact to Fix128 rounding
    let tol = 1e-9;
    assert_rel(abd.a.m11, 2.0 * t * (q11 + q22), tol, "A11");
    assert_rel(abd.a.m22, 2.0 * t * (q11 + q22), tol, "A22");
    assert_rel(abd.a.m12, 4.0 * t * q12, tol, "A12");
    assert_rel(abd.a.m33, 4.0 * t * q66, tol, "A66");
    assert_abs(abd.a.m13, 0.0, 1e-9, "A16 = 0 (cross-ply)");
    assert_abs(abd.a.m23, 0.0, 1e-9, "A26 = 0 (cross-ply)");
    assert!(
        abd.is_symmetric(Fix128::from_ratio(1, 1_000_000_000)),
        "B = 0 for symmetric stack"
    );
    let t3 = t * t * t;
    assert_rel(abd.d.m11, 2.0 / 3.0 * t3 * (7.0 * q11 + q22), tol, "D11");
    assert_rel(abd.d.m22, 2.0 / 3.0 * t3 * (q11 + 7.0 * q22), tol, "D22");
    assert_rel(abd.d.m12, 16.0 / 3.0 * t3 * q12, tol, "D12");
    assert_rel(abd.d.m33, 16.0 / 3.0 * t3 * q66, tol, "D66");
}

/// Jones eq. 2.84 at θ = 45°: Q̄₁₁ = Q̄₂₂ = (Q₁₁+Q₂₂+2Q₁₂+4Q₆₆)/4,
/// Q̄₁₂ = (Q₁₁+Q₂₂−4Q₆₆)/4 + Q₁₂/2, Q̄₆₆ = (Q₁₁+Q₂₂−2Q₁₂−2Q₆₆)/4 + Q₆₆/2,
/// Q̄₁₆ = Q̄₂₆ = (Q₁₁−Q₂₂)/4. Angle-ply [+45/−45]_s: A₁₆ = A₂₆ = 0,
/// D₁₆ ≠ 0, and an unsymmetric [+45/−45] has B₁₆ = −½ Q̄₁₆ t² ≠ 0.
#[test]
fn laminate_angle_ply_transformation_matches_jones() {
    let (q11, q12, q22, q66) = q_ref();
    let p45 = ply(Fix128::HALF_PI.half(), Fix128::ONE);
    let (b11, b12, b22, b16, b26, b66) = p45.q_bar();
    // CORDIC sin/cos at π/4 agree with 1/√2 to ~1e-18; ply products keep 1e-12
    let tol = 1e-11;
    assert_rel(
        b11,
        (q11 + q22 + 2.0 * q12 + 4.0 * q66) / 4.0,
        tol,
        "Q̄11(45°)",
    );
    assert_rel(
        b22,
        (q11 + q22 + 2.0 * q12 + 4.0 * q66) / 4.0,
        tol,
        "Q̄22(45°)",
    );
    assert_rel(
        b12,
        (q11 + q22 - 4.0 * q66) / 4.0 + q12 / 2.0,
        tol,
        "Q̄12(45°)",
    );
    assert_rel(
        b66,
        (q11 + q22 - 2.0 * q12 - 2.0 * q66) / 4.0 + q66 / 2.0,
        tol,
        "Q̄66(45°)",
    );
    assert_rel(b16, (q11 - q22) / 4.0, tol, "Q̄16(45°)");
    assert_rel(b26, (q11 - q22) / 4.0, tol, "Q̄26(45°)");

    let m45 = -Fix128::HALF_PI.half();
    let sym = [
        ply(Fix128::HALF_PI.half(), Fix128::ONE),
        ply(m45, Fix128::ONE),
        ply(m45, Fix128::ONE),
        ply(Fix128::HALF_PI.half(), Fix128::ONE),
    ];
    let abd = compute_abd(&sym);
    assert_abs(abd.a.m13, 0.0, 1e-9, "A16 = 0 for [+45/−45]_s");
    assert_abs(abd.a.m23, 0.0, 1e-9, "A26 = 0 for [+45/−45]_s");
    assert!(abd.is_symmetric(Fix128::from_ratio(1, 1_000_000_000)));
    // D = ⅓ Σ Q̄ (z_u³ − z_l³): outer +45 plies |z| ∈ [1, 2] give 7 each, inner
    // −45 plies |z| ∈ [0, 1] give 1 each → D16 = ⅓ Q̄16 (14 − 2) = 4 Q̄16
    let q16 = (q11 - q22) / 4.0;
    assert_rel(abd.d.m13, 4.0 * q16, tol, "D16 = 4 Q̄16");
    assert_rel(
        abd.a.m11,
        4.0 * (q11 + q22 + 2.0 * q12 + 4.0 * q66) / 4.0,
        tol,
        "A11 = 4 Q̄11",
    );

    // unsymmetric [+45/−45], t = 1 each: z ∈ [−1,0] (+45), [0,1] (−45)
    // B16 = ½ Q̄16 [(0 − 1)·(+1) + (1 − 0)·(−1)] = −Q̄16
    let unsym = [
        ply(Fix128::HALF_PI.half(), Fix128::ONE),
        ply(m45, Fix128::ONE),
    ];
    assert!(!is_symmetric_stack(&unsym));
    let abd_u = compute_abd(&unsym);
    assert_rel(abd_u.b.m13, -q16, tol, "B16 = −Q̄16 for [+45/−45]");
    assert_abs(abd_u.b.m11, 0.0, 1e-9, "B11 = 0 (Q̄11 same in both plies)");
}

// ============================================================================
// laminate_failure — Tsai & Wu 1971, Hashin 1980, Jones §2.9
// ============================================================================

/// Tsai–Wu with F₁₂ = −½√(F₁₁F₂₂) (Jones eq. 2.144): every uniaxial strength
/// lies exactly on the envelope, FI = 1. Same for Tsai–Hill (Jones eq. 2.138).
#[test]
fn laminate_failure_uniaxial_strengths_lie_on_the_envelope() {
    let s = LaminateStrengths::cfrp_ud();
    let cases = [
        (s.xt, Fix128::ZERO, Fix128::ZERO, "σ1 = Xt"),
        (-s.xc, Fix128::ZERO, Fix128::ZERO, "σ1 = −Xc"),
        (Fix128::ZERO, s.yt, Fix128::ZERO, "σ2 = Yt"),
        (Fix128::ZERO, -s.yc, Fix128::ZERO, "σ2 = −Yc"),
        (Fix128::ZERO, Fix128::ZERO, s.s, "τ12 = S"),
    ];
    for (sigma_1, sigma_2, tau_12, what) in cases {
        let st = StressState {
            sigma_1,
            sigma_2,
            tau_12,
        };
        // 1/X products of O(1e3) values: Fix128 keeps 1e-15 relative
        assert_rel(
            tsai_wu_failure_index(s, st),
            1.0,
            1e-12,
            &format!("Tsai–Wu {what}"),
        );
        assert_rel(
            tsai_hill_failure_index(s, st),
            1.0,
            1e-12,
            &format!("Tsai–Hill {what}"),
        );
    }
    // Tsai–Wu at half the strength is quadratic + linear: F1 σ + F11 σ² with
    // σ = Xt/2 → (1/Xt − 1/Xc)(Xt/2) + (Xt/2)²/(Xt Xc) = ½ − Xt/(4 Xc)
    let half = StressState {
        sigma_1: s.xt.half(),
        sigma_2: Fix128::ZERO,
        tau_12: Fix128::ZERO,
    };
    let (xt, xc) = (s.xt.to_f64(), s.xc.to_f64());
    assert_rel(
        tsai_wu_failure_index(s, half),
        0.5 - xt / (4.0 * xc),
        1e-12,
        "Tsai–Wu at Xt/2",
    );
    // Tsai–Hill biaxial σ1 = σ2 = σ: (σ/X)² − (σ/X)² + (σ/Y)² = (σ/Y)²
    let biax = StressState {
        sigma_1: Fix128::from_int(20),
        sigma_2: Fix128::from_int(20),
        tau_12: Fix128::ZERO,
    };
    assert_rel(
        tsai_hill_failure_index(s, biax),
        (20.0 / 40.0f64).powi(2),
        1e-12,
        "Tsai–Hill biaxial",
    );
}

/// Hashin (1980) 2-D criteria: the four uniaxial strengths trip exactly their
/// own mode; matrix compression uses (σ₂/2S)² + [(Yc/2S)² − 1] σ₂/Yc + (τ/S)²
/// which equals 1 at σ₂ = −Yc. Puck reuses the fibre modes.
#[test]
fn laminate_failure_hashin_modes_at_uniaxial_strengths() {
    let s = LaminateStrengths::cfrp_ud();
    let st = |a, b, c| StressState {
        sigma_1: a,
        sigma_2: b,
        tau_12: c,
    };
    let z = Fix128::ZERO;
    assert_eq!(
        hashin_failure_mode(s, st(s.xt, z, z)),
        FailureMode::FibreTension
    );
    assert_eq!(
        hashin_failure_mode(s, st(-s.xc, z, z)),
        FailureMode::FibreCompression
    );
    assert_eq!(
        hashin_failure_mode(s, st(z, s.yt, z)),
        FailureMode::MatrixTension
    );
    assert_eq!(
        hashin_failure_mode(s, st(z, -s.yc, z)),
        FailureMode::MatrixCompression
    );
    assert_eq!(
        hashin_failure_mode(s, st(z, z, s.s)),
        FailureMode::FibreTension,
        "σ1 = 0 ≥ 0 → fibre-tension branch owns τ"
    );
    // strictly inside the envelope: 99 % of each strength
    let f = Fix128::from_ratio(99, 100);
    assert_eq!(
        hashin_failure_mode(s, st(s.xt * f, z, z)),
        FailureMode::Safe
    );
    assert_eq!(
        hashin_failure_mode(s, st(z, -s.yc * f, z)),
        FailureMode::Safe
    );
    assert_eq!(
        hashin_failure_mode(s, st(z, s.yt * f, z)),
        FailureMode::Safe
    );
    // Puck: fibre modes identical to Hashin; IFF A at σ2 = Yt, C at σ2 = −Yc,
    // B for shear-dominated compression (module's simplified classifier)
    assert_eq!(
        puck_failure_mode(s, st(s.xt, z, z)),
        FailureMode::FibreTension
    );
    assert_eq!(
        puck_failure_mode(s, st(-s.xc, z, z)),
        FailureMode::FibreCompression
    );
    assert_eq!(
        puck_failure_mode(s, st(z, s.yt, z)),
        FailureMode::InterFibreA
    );
    assert_eq!(
        puck_failure_mode(s, st(z, -s.yc, z)),
        FailureMode::InterFibreC
    );
    // IFF B needs the shear ratio to dominate without tripping the Hashin
    // fibre branch (τ < S): σ2 = −0.7 Yc, τ = 0.8 S → 0.49 + 0.64 ≥ 1, and
    // Hashin matrix-compression FI = 1.60 − 1.59 + 0.64 = 0.65 → Safe there.
    let b = st(
        z,
        -s.yc * Fix128::from_ratio(7, 10),
        s.s * Fix128::from_ratio(8, 10),
    );
    assert_eq!(hashin_failure_mode(s, b), FailureMode::Safe);
    assert_eq!(puck_failure_mode(s, b), FailureMode::InterFibreB);
    // τ = S with σ1 = 0 is owned by the Hashin fibre-tension branch (τ²/S² = 1)
    assert_eq!(
        puck_failure_mode(s, st(z, -Fix128::ONE, s.s)),
        FailureMode::FibreTension
    );
    assert_eq!(puck_failure_mode(s, st(z, s.yt * f, z)), FailureMode::Safe);
}

// ============================================================================
// prestressed — Shigley 10th ed. §8-7, Irvine Cable Structures
// ============================================================================

/// Shigley eq. 8-13 (C = k_b/(k_b + k_m)), eq. 8-24 (F_b = C P + F_i),
/// separation P₀ = F_i/(1 − C) (Shigley §8-7, F_m = 0), eq. 8-27 (T = K F_i d),
/// eq. 8-31 (F_i = 0.75 F_p, F_p = S_p A_t).
#[test]
fn prestressed_bolt_joint_matches_shigley() {
    let c = bolt_load_fraction(Fix128::from_int(1000), Fix128::from_int(3000));
    assert_rel(c, 0.25, FIX_TOL, "C = k_b/(k_b + k_m)");
    let fi = Fix128::from_int(25_000);
    assert_rel(
        bolt_peak_tension(fi, c, Fix128::from_int(8000)),
        25_000.0 + 0.25 * 8000.0,
        FIX_TOL,
        "F_b = F_i + C P",
    );
    assert_rel(
        separation_load_n(fi, c),
        25_000.0 / 0.75,
        FIX_TOL,
        "P₀ = F_i/(1 − C)",
    );
    // M10, K = 0.2, F_i = 25 kN → T = 0.2 · 25000 · 0.010 = 50 N·m (Shigley 8-27)
    assert_rel(
        preload_from_torque(
            Fix128::from_int(50),
            Fix128::from_ratio(2, 10),
            Fix128::from_int(10),
        ),
        25_000.0,
        FIX_TOL,
        "F_i = T/(K d)",
    );
    // ISO 8.8 M10: S_p = 600 MPa, A_t = 58.0 mm² → 0.75 F_p = 26 100 N
    assert_rel(
        recommended_preload_n(
            Fix128::from_int(600),
            Fix128::from_int(58),
            Fix128::from_ratio(3, 4),
        ),
        0.75 * 600.0 * 58.0,
        FIX_TOL,
        "F_i = 0.75 S_p A_t",
    );
}

/// Parabolic cable (Irvine, *Cable Structures* 1981 §2.2; Meriam & Kraige
/// Statics §5/8): H = w L² / (8 s). The module's "transverse stiffness"
/// 8H/L is the uniform-load stiffness (w L)/s that follows from the same
/// equation — NOT the mid-span point-load stiffness 4H/L of a taut string.
#[test]
fn prestressed_parabolic_cable_matches_irvine() {
    let (w, l, s) = (0.01f64, 1000.0f64, 50.0f64);
    let tension = cable_pretension_n(
        Fix128::from_ratio(1, 100),
        Fix128::from_int(1000),
        Fix128::from_int(50),
    );
    assert_rel(tension, w * l * l / (8.0 * s), FIX_TOL, "H = w L²/(8 s)");
    let k = tensioned_cable_stiffness_n_per_mm(tension, Fix128::from_int(1000));
    assert_rel(k, w * l / s, FIX_TOL, "8H/L = (w L)/s identity");
}

// ============================================================================
// fillet_stress — Kirsch 1898, Inglis 1913 (exact); Peterson fits (empirical)
// ============================================================================

/// Kirsch: K_t = 3 for a circular hole in an infinite plate (Timoshenko &
/// Goodier §35). Inglis: K_t = 1 + 2a/b for an elliptical hole.
#[test]
fn fillet_stress_kirsch_and_inglis_exact() {
    assert_eq!(kt_circular_hole_infinite_plate(), Fix128::from_int(3));
    assert_eq!(
        kt_elliptical_hole(Fix128::from_int(3), Fix128::ONE),
        Fix128::from_int(7)
    );
    assert_eq!(
        kt_elliptical_hole(Fix128::from_int(5), Fix128::from_int(5)),
        Fix128::from_int(3)
    );
    // Inglis in terms of root radius ρ = b²/a: K_t = 1 + 2√(a/ρ); a = 4, b = 2 → ρ = 1 → 5
    assert_eq!(
        kt_elliptical_hole(Fix128::from_int(4), Fix128::from_int(2)),
        Fix128::from_int(5)
    );
}

/// Shoulder-fillet and U-notch K_t are piecewise / curve fits of Peterson
/// charts (Pilkey Chart 3.11, Table 2-8) — empirical, validation: none.
/// Invariants: K_t ≥ 1 in the fitted range, non-increasing with fillet
/// radius, non-decreasing with D/d and with notch depth, and the inverse
/// design helper returns a radius whose K_t does not exceed the target.
#[test]
fn fillet_stress_peterson_fits_invariants() {
    let d = Fix128::from_int(10);
    let big = Fix128::from_int(20);
    let mut prev = Fix128::from_int(100);
    for r_tenths in 1..=50 {
        let r = Fix128::from_ratio(r_tenths, 10);
        let kt = kt_shaft_shoulder_bending(r, d, big);
        assert!(kt >= Fix128::ONE, "K_t ≥ 1 at r = {}", r.to_f64());
        assert!(
            kt <= prev,
            "K_t non-increasing with r at r = {}",
            r.to_f64()
        );
        prev = kt;
    }
    let r = Fix128::ONE;
    assert!(
        kt_shaft_shoulder_bending(r, d, Fix128::from_int(30))
            >= kt_shaft_shoulder_bending(r, d, big)
    );
    assert!(
        kt_shaft_shoulder_bending(r, d, big)
            >= kt_shaft_shoulder_bending(r, d, Fix128::from_int(12))
    );

    let mut prev = Fix128::ZERO;
    for h_tenths in 1..=100 {
        let h = Fix128::from_ratio(h_tenths, 10);
        let kt = kt_u_notch_axial(h, Fix128::ONE);
        assert!(kt >= Fix128::ONE, "K_t ≥ 1 at h/r = {}", h.to_f64());
        assert!(kt >= prev, "K_t non-decreasing with notch depth");
        prev = kt;
    }
    // Pilkey Table 2-8 form: 0.85 + 2√(h/r); at h/r = 4 → 4.85 exactly
    assert_rel(
        kt_u_notch_axial(Fix128::from_int(4), Fix128::ONE),
        4.85,
        FIX_TOL,
        "fit at h/r = 4",
    );

    // Reachable targets only: the fit's floor at D/d = 2 is 1.3 × 1.1 = 1.43
    // (an unreachable target such as 1.2 silently returns d/2 — see report).
    for target_tenths in [15, 20, 25, 30] {
        let target = Fix128::from_ratio(target_tenths, 10);
        let r = recommended_fillet_radius_mm(d, big, target);
        let kt = kt_shaft_shoulder_bending(r, d, big);
        assert!(
            kt <= target,
            "K_t({}) = {} > target {}",
            r.to_f64(),
            kt.to_f64(),
            target.to_f64()
        );
    }
}

// ============================================================================
// plastic — Norton 1929 power law, Simo & Hughes bilinear return
// ============================================================================

/// Norton: ε̇ = A σⁿ, constant stress → ε_c = A σⁿ t (exact for forward
/// Euler because the rate is constant).
#[test]
fn plastic_norton_creep_matches_power_law() {
    let creep = NortonCreep {
        a: Fix128::from_ratio(1, 1_000_000_000),
        n: 3,
    };
    let mut state = PlasticState::default();
    creep.integrate(Fix128::from_int(10), Fix128::from_int(1000), &mut state);
    assert_rel(
        state.creep_strain,
        1e-9 * 1000.0 * 1000.0,
        1e-9,
        "ε = A σ³ t",
    );
    creep.integrate(Fix128::from_int(20), Fix128::from_int(500), &mut state);
    assert_rel(
        state.creep_strain,
        1e-3 + 1e-9 * 8000.0 * 500.0,
        1e-9,
        "accumulates",
    );
    // preset PLA (A = 6.34e-13, n = 3, 1.2.0 calibration) at 10 MPa for
    // 180 days = 15 552 000 s: closed form ε = 6.34e-13 · 1000 · 1.5552e7 =
    // 0.986 % — the "1 % / 6 months at 10 MPa" the docstring cites
    // (Bellehumeur 2004); the pre-1.2.0 A = 3e-10 gave 466 %
    let mut s2 = PlasticState::default();
    NortonCreep::pla_room_temp().integrate(
        Fix128::from_int(10),
        Fix128::from_int(15_552_000),
        &mut s2,
    );
    assert_rel(
        s2.creep_strain,
        6.34e-13 * 1000.0 * 15_552_000.0,
        1e-6, // A = 6.34e-13 is quantised at 2⁻⁶⁴ (rel 8e-8)
        "PLA preset, Norton law",
    );
    // from_fdm_material: H = 0.05 E (module contract)
    let model = PlasticModel::from_fdm_material(&pla_like());
    assert_rel(
        model.hardening_modulus_mpa,
        0.05 * 3500.0,
        FIX_TOL,
        "H = 0.05 E",
    );
    assert_rel(model.youngs_modulus_mpa, 3500.0, FIX_TOL, "E in MPa");
}

// ============================================================================
// hyperelastic — Ogden 1984 §4.3, Bower §3.5, Yeoh 1990
// ============================================================================

/// Incompressible uniaxial Cauchy stress: neo-Hookean σ = μ(λ² − 1/λ);
/// Mooney–Rivlin σ = 2(C₁ + C₂/λ)(λ² − 1/λ) (Bower eq. 3.5.29 / Ogden);
/// Yeoh σ = 2(λ² − 1/λ)(C₁ + 2C₂(I₁−3) + 3C₃(I₁−3)²).
#[test]
fn hyperelastic_uniaxial_stress_matches_closed_forms() {
    let lam = 2.0f64;
    let base = lam * lam - 1.0 / lam; // 3.5
    let l = Fix128::from_int(2);
    let nh = HyperelasticModel::NeoHookean {
        mu_mpa: Fix128::from_int(3),
    };
    assert_rel(
        uniaxial_cauchy_stress(&nh, l),
        3.0 * base,
        FIX_TOL,
        "neo-Hookean",
    );
    let mr = HyperelasticModel::MooneyRivlin {
        c1_mpa: Fix128::from_ratio(1, 10),
        c2_mpa: Fix128::from_ratio(5, 100),
    };
    assert_rel(
        uniaxial_cauchy_stress(&mr, l),
        2.0 * (0.1 + 0.05 / lam) * base,
        FIX_TOL,
        "Mooney–Rivlin",
    );
    // Mooney–Rivlin with C₂ = 0 and C₁ = μ/2 is neo-Hookean (Ogden §4.3.5)
    let mr0 = HyperelasticModel::MooneyRivlin {
        c1_mpa: Fix128::from_ratio(3, 2),
        c2_mpa: Fix128::ZERO,
    };
    assert_eq!(
        uniaxial_cauchy_stress(&mr0, l),
        uniaxial_cauchy_stress(&nh, l)
    );
    let yeoh = HyperelasticModel::Yeoh {
        c1_mpa: Fix128::from_ratio(1, 2),
        c2_mpa: Fix128::from_ratio(-17, 1000),
        c3_mpa: Fix128::from_ratio(62, 100_000),
    };
    let i1 = lam * lam + 2.0 / lam; // 5
    let d = i1 - 3.0;
    let dw = 0.5 + 2.0 * (-0.017) * d + 3.0 * 0.00062 * d * d;
    assert_rel(
        uniaxial_cauchy_stress(&yeoh, l),
        2.0 * dw * base,
        1e-11,
        "Yeoh",
    );
    // Stretch invariants for uniaxial λ = 2: I₁ = 5, I₂ = 1/4 + 2·2 = 4.25, J = 1
    let st = Stretch::uniaxial(l);
    assert_rel(st.i1(), 5.0, 1e-12, "I₁");
    assert_rel(st.i2(), 4.25, 1e-12, "I₂");
    assert_rel(st.volume_ratio(), 1.0, 1e-12, "J = 1");
    // W: neo-Hookean W = μ/2 (I₁ − 3) = 3 at λ = 2, μ = 3
    assert_rel(strain_energy_density(&nh, &st), 3.0, 1e-12, "W neo-Hookean");
    // small-strain μ₀: neo-Hookean μ, MR 2(C₁+C₂), Yeoh 2C₁ (Ogden §4.3)
    assert_eq!(small_strain_shear_modulus(&nh), Fix128::from_int(3));
    assert_rel(
        small_strain_shear_modulus(&mr),
        0.3,
        FIX_TOL,
        "μ₀ = 2(C₁ + C₂)",
    );
    assert_rel(small_strain_shear_modulus(&yeoh), 1.0, FIX_TOL, "μ₀ = 2 C₁");
}

/// Ogden 1984 eq. 4.3.x: for incompressible uniaxial loading with σ₂ = σ₃ = 0,
/// σ₁ = λ dŴ/dλ where Ŵ(λ) = W(λ, λ^−½, λ^−½). Central difference of the
/// module's own W must reproduce its σ (h = 1e-4 → O(h²) ≈ 1e-8 relative).
#[test]
fn hyperelastic_stress_is_derivative_of_strain_energy() {
    let models = [
        HyperelasticModel::tpu_soft(),
        HyperelasticModel::silicone_soft(),
        HyperelasticModel::natural_rubber(),
    ];
    let h = 1e-4;
    for model in &models {
        for lam in [1.2f64, 1.5, 2.0, 3.0] {
            let wp = strain_energy_density(model, &Stretch::uniaxial(Fix128::from_f64(lam + h)))
                .to_f64();
            let wm = strain_energy_density(model, &Stretch::uniaxial(Fix128::from_f64(lam - h)))
                .to_f64();
            let sigma_fd = lam * (wp - wm) / (2.0 * h);
            let sigma = uniaxial_cauchy_stress(model, Fix128::from_f64(lam)).to_f64();
            assert!(
                ((sigma - sigma_fd) / sigma).abs() < 1e-6,
                "{model:?} at λ = {lam}: σ = {sigma}, λ dW/dλ = {sigma_fd}"
            );
        }
        // λ → 1: σ ≈ 3 μ₀ ε (E = 3μ for incompressible); at ε = 1e-3 the
        // neo-Hookean series gives σ/(3ε) = μ₀ (1 + ε + …) → 1e-3 relative
        let eps = 1e-3;
        let sigma = uniaxial_cauchy_stress(model, Fix128::from_f64(1.0 + eps)).to_f64();
        let mu0 = small_strain_shear_modulus(model).to_f64();
        assert!(
            ((sigma / (3.0 * eps) - mu0) / mu0).abs() < 2e-3,
            "{model:?}: μ₀ limit"
        );
    }
}

// ============================================================================
// modal — Blevins Table 8-1, Leissa 1969, Rao Mechanical Vibrations
// ============================================================================

/// Blevins Table 8-1: f₁ = (λ²/(2π L²)) √(EI/(ρA)), λ = 1.87510 (cantilever),
/// π (pinned), 4.73004 (clamped). The module stores λ² rounded to 4 digits
/// (3.516 / 9.870 / 22.373) and 10^4.5 to 8 digits → tolerance 5e-5.
#[test]
fn modal_beam_first_mode_matches_blevins() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let m = pla_like();
    // SI: E = 3.5e9 Pa, I = 8.333e-10 m⁴, ρ = 1240 kg/m³, A = 1e-4 m², L = 0.1 m
    let (e, i, rho, a, l) = (
        3.5e9f64,
        10.0 * 1000.0 / 12.0 * 1e-12,
        1240.0f64,
        1e-4f64,
        0.1f64,
    );
    let base = (e * i / (rho * a)).sqrt() / (2.0 * PI * l * l);
    let cases = [
        (BeamBoundary::Cantilever, 1.875_104_07f64.powi(2)),
        (BeamBoundary::SimplySupported, PI * PI),
        (BeamBoundary::ClampedClamped, 4.730_040_74f64.powi(2)),
    ];
    for (bc, lam2) in cases {
        let f = beam_natural_frequency_hz(&section, Fix128::from_int(100), &m, bc);
        assert_rel(f, lam2 * base, 5e-5, &format!("{bc:?} f₁"));
    }
}

/// Leissa, *Vibration of Plates* (NASA SP-160, 1969) §4.1, SSSS rectangular
/// plate: f₁₁ = (π/2) √(D/(ρh)) (1/a² + 1/b²), D = E h³ / (12(1−ν²)).
/// Module rounds 10^4.5 to 8 digits → 1e-7; use 1e-6.
#[test]
fn modal_plate_first_mode_matches_leissa() {
    let m = pla_like();
    let (e, nu, h, a, b, rho) = (3.5e9f64, 0.35f64, 2e-3f64, 0.1f64, 0.15f64, 1240.0f64);
    let d = e * h * h * h / (12.0 * (1.0 - nu * nu));
    let want = PI / 2.0 * (d / (rho * h)).sqrt() * (1.0 / (a * a) + 1.0 / (b * b));
    let f = plate_natural_frequency_hz(
        &m,
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(150),
    );
    assert_rel(f, want, 1e-6, "Leissa SSSS f₁₁");
}

/// Rao, *Mechanical Vibrations* §2.2: f = (1/2π) √(k/m). k = 1000 N/mm =
/// 1e6 N/m, m = 1 g = 1e-3 kg → 5032.9 Hz. Fix128 sqrt exact to 1e-18.
#[test]
fn modal_single_dof_matches_rao() {
    let f = single_dof_frequency_hz(Fix128::from_int(1000), Fix128::ONE);
    assert_rel(
        f,
        (1e6f64 / 1e-3).sqrt() / (2.0 * PI),
        FIX_TOL,
        "f = √(k/m)/2π",
    );
}

/// Rao §2.2 / Blevins Table 8-15: torsional single-disc shaft
/// ω = √(k_t / I_p), k_t = G J / L. Inputs are MPa, mm⁴, g·mm², mm:
/// k_t [N·mm] → ×1e-3 N·m, I_p [g·mm²] → ×1e-9 kg·m², so ω_SI = 10³ √(k_eng/I_eng).
///
/// DISCREPANCY (documented, failing on purpose): the module multiplies by
/// 10^4.5 = 31 622.8 instead of 10³, so every torsional frequency is
/// 31.6× too high. Worked case: steel shaft d = 10 mm, L = 100 mm,
/// G = 80 000 MPa, J = π d⁴/32 = 981.75 mm⁴, 1 kg disc r = 50 mm
/// (I_p = ½ m r² = 1.25e6 g·mm²) → f = 126.2 Hz (module: 3990 Hz).
#[test]
fn modal_torsional_frequency_si_scale_is_1e3() {
    let (g, l, d) = (80_000.0f64, 100.0f64, 10.0f64);
    let j = PI * d.powi(4) / 32.0;
    let ip = 0.5 * 1000.0 * 50.0 * 50.0; // g·mm²
    let k_si = g * j / l * 1e-3; // N·m/rad
    let ip_si = ip * 1e-9; // kg·m²
    let want = (k_si / ip_si).sqrt() / (2.0 * PI);
    let f = torsional_frequency_hz(
        Fix128::from_int(80_000),
        Fix128::from_f64(j),
        Fix128::from_f64(ip),
        Fix128::from_int(100),
    );
    assert_rel(f, want, 1e-6, "torsional f = √(GJ/(L I_p))/2π in SI");
}

// ============================================================================
// mass_properties — standard inertia tensors, parallel-axis theorem
// ============================================================================

/// Sphere I = ⅖ m r², box I_xx = m/12 (h² + d²), cylinder I_axis = ½ m r²,
/// I_perp = m/12 (3r² + h²) (any statics text, e.g. Meriam & Kraige Table D/4).
#[test]
fn mass_properties_primitives_match_closed_forms() {
    let rho = Fix128::from_int(2);
    let sph = sphere_mass_properties(Fix128::from_int(3), rho);
    let m = 4.0 / 3.0 * PI * 27.0 * 2.0;
    assert_rel(sph.mass, m, FIX_TOL, "sphere mass");
    assert_rel(
        sph.inertia_tensor.col0.x,
        0.4 * m * 9.0,
        FIX_TOL,
        "sphere I",
    );
    assert_eq!(sph.inertia_tensor.col1.y, sph.inertia_tensor.col2.z);
    assert!(sph.inertia_tensor.col0.y.is_zero() && sph.inertia_tensor.col1.z.is_zero());

    let bx = box_mass_properties(Vec3Fix::from_int(1, 2, 3), rho);
    let (w, h, d) = (2.0f64, 4.0f64, 6.0f64);
    let m = w * h * d * 2.0;
    assert_rel(bx.mass, m, FIX_TOL, "box mass");
    assert_rel(
        bx.inertia_tensor.col0.x,
        m / 12.0 * (h * h + d * d),
        FIX_TOL,
        "box Ixx",
    );
    assert_rel(
        bx.inertia_tensor.col1.y,
        m / 12.0 * (w * w + d * d),
        FIX_TOL,
        "box Iyy",
    );
    assert_rel(
        bx.inertia_tensor.col2.z,
        m / 12.0 * (w * w + h * h),
        FIX_TOL,
        "box Izz",
    );

    let cyl = cylinder_mass_properties(Fix128::from_int(2), Fix128::from_int(5), rho);
    let (r, hh) = (2.0f64, 10.0f64);
    let m = PI * r * r * hh * 2.0;
    assert_rel(cyl.mass, m, FIX_TOL, "cylinder mass");
    assert_rel(
        cyl.inertia_tensor.col1.y,
        0.5 * m * r * r,
        FIX_TOL,
        "cylinder I_axis (Y)",
    );
    assert_rel(
        cyl.inertia_tensor.col0.x,
        m / 12.0 * (3.0 * r * r + hh * hh),
        FIX_TOL,
        "cylinder I_perp",
    );
    assert_eq!(cyl.inertia_tensor.col0.x, cyl.inertia_tensor.col2.z);
}

/// Parallel-axis theorem I' = I_cm + m (d²E − d⊗d): sphere shifted by
/// (5, 0, 0): I_xx unchanged, I_yy = I_zz = I + m·25, off-diagonals stay 0;
/// shifted by (1, 2, 0): I_xy = −m·2.
#[test]
fn mass_properties_parallel_axis_matches_steiner() {
    let sph = sphere_mass_properties(Fix128::ONE, Fix128::ONE);
    let m = sph.mass.to_f64();
    let i0 = sph.inertia_tensor.col0.x.to_f64();
    let t = translate_inertia(&sph, Vec3Fix::from_int(5, 0, 0));
    assert_rel(t.col0.x, i0, FIX_TOL, "Ixx unchanged");
    assert_rel(t.col1.y, i0 + 25.0 * m, FIX_TOL, "Iyy + m d²");
    assert_rel(t.col2.z, i0 + 25.0 * m, FIX_TOL, "Izz + m d²");
    assert!(t.col0.y.is_zero() && t.col0.z.is_zero() && t.col1.z.is_zero());
    let t2 = translate_inertia(&sph, Vec3Fix::from_int(1, 2, 0));
    assert_rel(t2.col0.x, i0 + 4.0 * m, FIX_TOL, "Ixx + m(dy²+dz²)");
    assert_rel(t2.col1.y, i0 + 1.0 * m, FIX_TOL, "Iyy + m(dx²+dz²)");
    assert_rel(t2.col2.z, i0 + 5.0 * m, FIX_TOL, "Izz + m(dx²+dy²)");
    assert_rel(t2.col0.y, -2.0 * m, FIX_TOL, "Ixy = −m dx dy");
    assert_eq!(t2.col0.y, t2.col1.x, "symmetric");
}

/// Capsule (cylinder h + two hemispheres of radius r), perpendicular axis:
/// I⊥ = m_cyl (h²/12 + r²/4) + m_sph (⅖ r² + h²/4 + 3hr/8)
/// (hemisphere I_cm = 83/320 m_h r², centroid 3r/8 from the flat face, then
/// Steiner to the capsule centre). Limit h → 0 must recover the sphere.
///
/// DISCREPANCY (documented, failing on purpose): the module applies the
/// parallel-axis shift (h/2 + 3r/8)² to the *full-sphere-about-centre*
/// inertia ⅖ m_sph r² instead of to the hemisphere-about-its-own-centroid
/// value, so I⊥ is over-predicted by m_sph (3r/8)² = 9/64 m_sph r².
/// With r = 1, h = 0 the "capsule" is a sphere but the module returns
/// ⅖ m + 9/64 m instead of ⅖ m.
#[test]
fn mass_properties_capsule_uses_hemisphere_centroid_parallel_axis() {
    let r = 1.0f64;
    let m_sph = 4.0 / 3.0 * PI;
    let degenerate = capsule_mass_properties(Fix128::ONE, Fix128::ZERO, Fix128::ONE);
    assert_rel(
        degenerate.mass,
        m_sph,
        FIX_TOL,
        "h = 0 capsule mass = sphere",
    );
    assert_rel(
        degenerate.inertia_tensor.col1.y,
        0.4 * m_sph * r * r,
        FIX_TOL,
        "I_axis = sphere",
    );
    assert_rel(
        degenerate.inertia_tensor.col0.x,
        0.4 * m_sph * r * r,
        1e-9,
        "h = 0 capsule I⊥ must equal sphere ⅖ m r²",
    );
    let (r, h) = (1.0f64, 2.0f64);
    let m_cyl = PI * r * r * h;
    let cap = capsule_mass_properties(Fix128::ONE, Fix128::ONE, Fix128::ONE);
    assert_rel(cap.mass, m_cyl + m_sph, FIX_TOL, "capsule mass");
    assert_rel(
        cap.inertia_tensor.col1.y,
        0.5 * m_cyl * r * r + 0.4 * m_sph * r * r,
        FIX_TOL,
        "capsule I_axis",
    );
    let want = m_cyl * (h * h / 12.0 + r * r / 4.0)
        + m_sph * (0.4 * r * r + h * h / 4.0 + 3.0 * h * r / 8.0);
    assert_rel(
        cap.inertia_tensor.col0.x,
        want,
        1e-9,
        "capsule I⊥ closed form",
    );
}

/// Convex hull of the unit right tetrahedron: the centroid-to-face fan is an
/// exact tiling, so mass = ρ/6 and CoM = (¼, ¼, ¼) exactly. The inertia is
/// a point-mass approximation per sub-tetrahedron (documented in the
/// module) — not oracle-checked here; see report.
#[test]
fn mass_properties_convex_hull_tetrahedron_mass_and_centroid() {
    let verts = [
        Vec3Fix::from_int(0, 0, 0),
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(0, 1, 0),
        Vec3Fix::from_int(0, 0, 1),
    ];
    let p = convex_hull_mass_properties(&verts, Fix128::from_int(6));
    assert_rel(p.mass, 1.0, FIX_TOL, "V = 1/6, ρ = 6");
    assert_rel(p.center_of_mass.x, 0.25, FIX_TOL, "CoM x");
    assert_rel(p.center_of_mass.y, 0.25, FIX_TOL, "CoM y");
    assert_rel(p.center_of_mass.z, 0.25, FIX_TOL, "CoM z");
}

// ============================================================================
// structural_solver — composition of Roark / Simo & Hughes / Basquin / Shigley
// ============================================================================

/// Each step re-applies σ = M/Z as the trial stress (module contract), so
/// the 1-D return map (Simo & Hughes Box 1.4: Δγ = f_trial/(E + K)) gives a
/// geometric series toward the bilinear-hardening plastic strain
/// (σ − σ_y)/H:  ε_p(k) = (σ − σ_y)/H · (1 − (E/(E+H))^k).
/// Fatigue: Basquin N = N_e (S_e/S)^m with one cycle per step (Miner).
/// Creep: Norton A σ³ Δt per step (Findley at T = T_g stays at ε₀ ≪ Norton).
/// Buckling FoS: Johnson column (Shigley 4-46) P_cr / P_axial.
#[test]
fn structural_solver_step_matches_composed_closed_forms() {
    // material at T_g = 25 °C so the WLF shift is exactly 1 and Findley = ε₀ + m t³
    let m = material("PLA", (35, 10), 50, 60, (124, 100), 25, (65, 100));
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(100),
        length_mm: Fix128::from_int(100),
    };
    let mut solver = StructuralSolver::new(section, load, m);
    solver.axial_load_n = Fix128::from_int(10);

    let (e, h_mod, sy) = (3500.0f64, 175.0f64, 50.0f64);
    let sigma: f64 = 100.0 * 100.0 / (10.0 * 100.0 / 6.0); // 60 MPa > σ_y
    let n_fail = 1e6 * (18.0f64 / 60.0).powi(5); // S_e = 0.3 UTS = 18 MPa
    let norton_per_step = 6.34e-13 * sigma.powi(3) * 3600.0; // 1.2.0 PLA calibration
    let k_gyr = (10.0f64 * 1000.0 / 12.0 / 100.0).sqrt();
    let lam = 100.0 / k_gyr; // 34.6 < λ_t = 37.2 → Johnson
    let p_cr = (sy - (sy * lam / (2.0 * PI)).powi(2) / e) * 100.0;

    for k in 1..=5u32 {
        let r = solver.step();
        assert_rel(r.bending_stress_mpa, sigma, FIX_TOL, "σ = M/Z");
        let ep = (sigma - sy) / h_mod * (1.0 - (e / (e + h_mod)).powi(k as i32));
        assert_rel(r.plastic_strain, ep, 1e-10, &format!("ε_p after {k} steps"));
        // N = 2430 in reals; the module truncates N to u64 after evaluating
        // (18/60)^5 in Fix128 (0.3 is not representable → 2429), so allow
        // one cycle: 1/2430 = 4.1e-4 relative.
        assert_rel(
            r.fatigue_damage,
            f64::from(k) / n_fail,
            1.0 / (n_fail - 1.0),
            "Miner D = k/N(σ)",
        );
        // reported creep = max(short-term Norton over k hours, long-term Findley
        // projection ε₀ + m·k³ at the operating temperature; 1.2.0 no longer
        // freezes the Findley term below T_g, so ε₀ = 0.3 % dominates early on)
        let findley = 3e-3 + 8.3e-14 * f64::from(k).powi(3);
        assert_rel(
            r.creep_strain,
            (norton_per_step * f64::from(k)).max(findley),
            1e-6,
            "creep = max(Norton, Findley)",
        );
        assert_rel(r.buckling_fos, p_cr / 10.0, FIX_TOL, "Johnson P_cr / P");
        assert!(r.failed_this_step, "σ > σ_y yields every step");
        assert!(!r.is_safe);
        assert_rel(r.elapsed_hours, f64::from(k), FIX_TOL, "1 h steps");
    }
    let hist = solver.run(0);
    assert_eq!(hist.failure_step, Some(0));
    assert_eq!(hist.steps, 5);

    // elastic case: σ = 1.5 MPa (5 N on 10×20, L = 200) — no yield, below
    // endurance → zero damage, nothing fails
    let soft = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(5),
        length_mm: Fix128::from_int(200),
    };
    let sec2 = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(20),
    };
    let mut s2 = StructuralSolver::new(sec2, soft, m);
    let h2 = s2.run(3);
    assert_eq!(h2.failure_step, None);
    assert_eq!(h2.plastic_state.equivalent_plastic_strain, Fix128::ZERO);
    assert_eq!(h2.fatigue_damage, Fix128::ZERO);
}

// ============================================================================
// bridging — Euclidean span length (exact); material limits empirical
// ============================================================================

/// Span length is the Euclidean norm; Pythagorean triples make it exact.
/// The per-material bridging limits are a curated table — validation: none.
#[test]
fn bridging_span_geometry_exact() {
    let m = pla_like(); // limit 20 mm
    let span = BridgeSpan {
        start: Vec3Fix::from_int(1, 2, 3),
        end: Vec3Fix::from_int(4, 6, 3),
    };
    assert_eq!(span.length_mm(), Fix128::from_int(5), "3-4-5");
    assert_eq!(span.z_delta_mm(), Fix128::ZERO);
    let c = check_span(&span, &m);
    assert_eq!(c.safety_ratio, Fix128::from_int(4), "20 / 5");
    assert!(c.is_safe);
    let long = BridgeSpan {
        start: Vec3Fix::from_int(0, 0, 0),
        end: Vec3Fix::from_int(12, 16, 15),
    };
    assert_eq!(
        long.length_mm(),
        Fix128::from_int(25),
        "12-16-15-25 quadruple"
    );
    assert_eq!(long.z_delta_mm(), Fix128::from_int(15));
    let c = check_span(&long, &m);
    assert_rel(c.safety_ratio, 0.8, FIX_TOL, "20 / 25");
    assert!(!c.is_safe);
    let report = analyze_bridges(&[span, long], &m);
    assert_eq!(report.unsafe_count, 1);
    assert_eq!(report.max_length_mm, Fix128::from_int(25));
    assert_eq!(report.unsafe_checks().count(), 1);
    // exactly at the limit is safe (ratio = 1)
    let at = BridgeSpan {
        start: Vec3Fix::from_int(0, 0, 0),
        end: Vec3Fix::from_int(20, 0, 0),
    };
    assert!(check_span(&at, &m).is_safe);
}

// ============================================================================
// print_orientation — rotation geometry (exact); cos² mixing rule (heuristic)
// ============================================================================

/// angle_to_z_axis is acos(ẑ·R v): (1,0,1) → π/4, (0,0,1) → 0, (1,0,0) → π/2,
/// and R_x(π/2)(0,0,1) = (0,−1,0) → π/2. CORDIC atan2 / sin_cos ≈ 1e-18.
/// The σ_z cos²θ + σ_xy sin²θ rule itself is an empirical mixing rule
/// (validation: none); its end points and mid-point are checked as stated.
#[test]
fn print_orientation_geometry_and_mixing_rule_endpoints() {
    let id = OrientationCandidate::IDENTITY;
    let diag = LoadDirection {
        x: Fix128::ONE,
        y: Fix128::ZERO,
        z: Fix128::ONE,
    };
    assert_abs(angle_to_z_axis(&diag, &id), PI / 4.0, 1e-12, "acos(1/√2)");
    assert_abs(
        angle_to_z_axis(&LoadDirection::axis_z(), &id),
        0.0,
        1e-12,
        "along Z",
    );
    assert_abs(
        angle_to_z_axis(&LoadDirection::axis_x(), &id),
        PI / 2.0,
        1e-12,
        "along X",
    );
    let rx = OrientationCandidate {
        theta_x: Fix128::HALF_PI,
        theta_y: Fix128::ZERO,
    };
    assert_abs(
        angle_to_z_axis(&LoadDirection::axis_z(), &rx),
        PI / 2.0,
        1e-12,
        "R_x(π/2) ẑ ⊥ ẑ",
    );
    let ry = OrientationCandidate {
        theta_x: Fix128::ZERO,
        theta_y: Fix128::HALF_PI.half(),
    };
    // standard right-handed R_y(θ): x̂ → (cos θ, 0, −sin θ) → angle π/2 + θ
    assert_abs(
        angle_to_z_axis(&LoadDirection::axis_x(), &ry),
        3.0 * PI / 4.0,
        1e-12,
        "R_y(π/4) x̂",
    );

    let m = pla_like(); // σ_xy = 50, σ_z = 32.5
    assert_rel(
        effective_yield_at_angle(&m, Fix128::ZERO),
        32.5,
        1e-12,
        "θ = 0 → σ_z",
    );
    assert_rel(
        effective_yield_at_angle(&m, Fix128::HALF_PI),
        50.0,
        1e-12,
        "θ = π/2 → σ_xy",
    );
    assert_rel(
        effective_yield_at_angle(&m, Fix128::HALF_PI.half()),
        (32.5 + 50.0) / 2.0,
        1e-12,
        "θ = π/4",
    );
    // analytical optimum for a Z load: rotate to the bed → σ_xy, gain 17.5
    let rep = optimize_analytical(&LoadDirection::axis_z(), &m);
    assert_rel(rep.effective_yield_mpa, 50.0, 1e-12, "optimum σ_xy");
    assert_rel(rep.identity_yield_mpa, 32.5, 1e-12, "identity σ_z");
    assert_rel(rep.improvement_mpa, 17.5, 1e-12, "gain");
    // grid search with π/4 steps contains the exact optimum for (0,1,1): θ_x = −π/4
    let yz = LoadDirection {
        x: Fix128::ZERO,
        y: Fix128::ONE,
        z: Fix128::ONE,
    };
    let grid = optimize_grid(&yz, &m, Fix128::HALF_PI.half());
    assert_rel(grid.effective_yield_mpa, 50.0, 1e-9, "grid finds σ_xy");
    assert_abs(grid.angle_to_z_axis, PI / 2.0, 1e-9, "grid optimum ⊥ ẑ");
}

/// The optimum of σ_z cos²θ + σ_xy sin²θ (σ_xy > σ_z) is σ_xy at θ = π/2 and
/// is reachable for every load direction (rotate about the axis ⊥ to both
/// the load and ẑ). `optimize_analytical` claims to return it.
///
/// DISCREPANCY (documented, failing on purpose): the module only rotates
/// about Y by θ_id − π/2, which tilts the XZ projection; for a load with a
/// Y component the rotated angle is not π/2. Load (0,1,1)/√2: θ_id = π/4,
/// R_y(−π/4) gives z = ½ → θ = π/3 → σ = ¼σ_z + ¾σ_xy = 45.6 MPa, not 50.
#[test]
fn print_orientation_analytical_optimum_reaches_plane_for_y_loads() {
    let m = pla_like();
    let yz = LoadDirection {
        x: Fix128::ZERO,
        y: Fix128::ONE,
        z: Fix128::ONE,
    };
    let rep = optimize_analytical(&yz, &m);
    assert_abs(
        rep.angle_to_z_axis,
        PI / 2.0,
        1e-9,
        "analytical optimum must reach θ = π/2",
    );
    assert_rel(
        rep.effective_yield_mpa,
        50.0,
        1e-9,
        "analytical optimum must equal σ_xy",
    );
}

// ============================================================================
// vibration_wall — Leissa plate frequency (exact); ±20 % band (heuristic)
// ============================================================================

/// Wall frequency is Leissa's SSSS f₁₁ (see modal test). The nearest-source
/// selection and the ±band rule are checked at exact ratios; the curated
/// source list and the 20 % band are heuristics — validation: none.
#[test]
fn vibration_wall_frequency_matches_leissa_and_band_logic() {
    let m = pla_like();
    let (e, nu, h, a, b, rho) = (3.5e9f64, 0.35f64, 1.5e-3f64, 0.08f64, 0.12f64, 1240.0f64);
    let d = e * h * h * h / (12.0 * (1.0 - nu * nu));
    let f_wall = PI / 2.0 * (d / (rho * h)).sqrt() * (1.0 / (a * a) + 1.0 / (b * b));
    let nu_f = Fix128::from_ratio(35, 100);
    let (t, sa, sb) = (
        Fix128::from_ratio(3, 2),
        Fix128::from_int(80),
        Fix128::from_int(120),
    );
    let sources = [
        ExcitationSource {
            name: "far",
            frequency_hz: Fix128::from_f64(f_wall * 2.0),
        },
        ExcitationSource {
            name: "near",
            frequency_hz: Fix128::from_f64(f_wall / 1.1),
        },
    ];
    let r = analyze_wall_resonance(&m, nu_f, t, sa, sb, &sources, Fix128::from_ratio(2, 10));
    assert_rel(r.wall_frequency_hz, f_wall, 1e-6, "Leissa f₁₁");
    assert_eq!(r.nearest_source.name, "near");
    assert_rel(r.frequency_ratio, 1.1, 1e-6, "ratio wall/source");
    assert!(r.is_risky, "1.1 inside ±20 %");
    let only_far = [sources[0]];
    let r2 = analyze_wall_resonance(&m, nu_f, t, sa, sb, &only_far, Fix128::from_ratio(2, 10));
    assert_rel(r2.frequency_ratio, 0.5, 1e-6, "ratio 0.5");
    assert!(!r2.is_risky);
    // band edge: ratio exactly 1.2 is NOT risky (strict inequality)
    let edge = [ExcitationSource {
        name: "edge",
        frequency_hz: r.wall_frequency_hz / Fix128::from_ratio(12, 10),
    }];
    let r3 = analyze_wall_resonance(&m, nu_f, t, sa, sb, &edge, Fix128::from_ratio(2, 10));
    assert!(
        !r3.is_risky,
        "ratio {} at the band edge",
        r3.frequency_ratio.to_f64()
    );
}

// ============================================================================
// bimaterial — Voigt/Reuss (Hill 1952), constrained thermal mismatch
// ============================================================================

/// Rule of mixtures (Jones §3.2, Hill 1952): E_V = V_a E_a + V_b E_b,
/// 1/E_R = V_a/E_a + V_b/E_b, E_R ≤ E_V. Equal-thickness bonded layers
/// cooled by ΔT with no bending (force balance σ_a t + σ_b t = 0, strain
/// compatibility): σ_a = (α_a − α_b) ΔT E_a E_b / (E_a + E_b) — tension in
/// the higher-CTE layer (Timoshenko 1925, membrane limit).
#[test]
fn bimaterial_rule_of_mixtures_and_thermal_mismatch() {
    let a = BimaterialSide::from_material(pla_like(), Fix128::from_int(1)); // 68e-6
    let b = BimaterialSide::from_material(steel_like(), Fix128::from_int(3)); // 17.3e-6
    let (ea, eb, va, vb) = (3500.0f64, 200_000.0f64, 0.25f64, 0.75f64);
    assert_rel(
        effective_modulus_voigt_mpa(&a, &b),
        va * ea + vb * eb,
        FIX_TOL,
        "Voigt",
    );
    assert_rel(
        effective_modulus_reuss_mpa(&a, &b),
        1.0 / (va / ea + vb / eb),
        1e-11,
        "Reuss",
    );
    assert!(effective_modulus_reuss_mpa(&a, &b) < effective_modulus_voigt_mpa(&a, &b));
    assert_rel(a.cte_per_c, 68e-6, 1e-12, "PLA CTE lookup");
    assert_rel(b.cte_per_c, 17.3e-6, 1e-12, "SUS304 CTE lookup");
    let dt = 180.0f64;
    let want = (68e-6 - 17.3e-6) * dt * ea * eb / (ea + eb);
    let sigma = thermal_residual_stress_mpa(&a, &b, Fix128::from_int(200), Fix128::from_int(20));
    assert_rel(sigma, want, 1e-10, "σ_res membrane closed form");
    // antisymmetric in the pair order
    let swapped = thermal_residual_stress_mpa(&b, &a, Fix128::from_int(200), Fix128::from_int(20));
    assert_rel(swapped, -want, 1e-10, "σ_res(b, a) = −σ_res(a, b)");
    // bond strength: same material → σ_y; dissimilar → ½√(σ_a σ_b), symmetric (empirical)
    assert_eq!(
        interfacial_bond_strength_mpa(&pla_like(), &pla_like()),
        Fix128::from_int(50)
    );
    let bond = interfacial_bond_strength_mpa(&pla_like(), &steel_like());
    assert_rel(bond, 0.5 * (50.0f64 * 215.0).sqrt(), 1e-12, "½√(σ_a σ_b)");
    assert_eq!(
        bond,
        interfacial_bond_strength_mpa(&steel_like(), &pla_like())
    );
    let rep = analyze_bimaterial(
        &a,
        &b,
        Fix128::from_int(200),
        Fix128::from_int(20),
        Fix128::from_int(5),
    );
    assert_rel(
        rep.total_interfacial_stress_mpa,
        want + 5.0,
        1e-10,
        "|σ_res| + |τ|",
    );
    assert_rel(
        rep.factor_of_safety,
        bond.to_f64() / (want + 5.0),
        1e-10,
        "FoS",
    );
}

// ============================================================================
// creep_longterm — Findley 1989, WLF 1955
// ============================================================================

/// Findley ε(t) = ε₀ + m tⁿ (Findley, Lai & Onaran Ch. 4). With T = T_g the
/// universal WLF shift is exp(0) = 1 exactly. At T = T_g + 10 °C:
/// log₁₀ a_T = −17.44·10/(51.6+10) (WLF 1955 eq. 1), t_eff = t / a_T.
/// exp_fix converges to Fix128 ULP; the module's ln10 has 9 digits → 1e-8.
#[test]
fn creep_longterm_findley_and_wlf_match_closed_forms() {
    let p = FindleyParameters {
        epsilon_0: Fix128::from_ratio(3, 1000),
        m: Fix128::from_ratio(1, 1_000_000_000_000_i64),
        n_int: 3,
    };
    let at_tg = material("PLA", (35, 10), 50, 60, (124, 100), 60, (65, 100));
    let t = Fix128::from_int(4380);
    let want = 0.003 + 1e-12 * 4380.0f64.powi(3);
    // m = 1e-12 is stored as ⌊1e-12 · 2^64⌋ / 2^64 (5.4e-8 relative
    // truncation), which is the whole error budget here → 1e-7.
    assert_rel(
        predict_strain(&p, &at_tg, t, Fix128::from_int(60)),
        want,
        1e-7,
        "Findley at T = T_g",
    );
    assert_rel(
        predict_strain(&p, &at_tg, Fix128::ZERO, Fix128::from_int(60)),
        0.003,
        FIX_TOL,
        "ε(0) = ε₀",
    );
    let log_at = -17.44 * 10.0 / (51.6 + 10.0);
    let t_eff = 1.0 / 10f64.powf(log_at);
    let want_hot = 0.003 + 1e-12 * t_eff.powi(3);
    assert_rel(
        predict_strain(&p, &at_tg, Fix128::ONE, Fix128::from_int(70)),
        want_hot,
        1e-7,
        "WLF-shifted Findley",
    );
    // preset values as documented
    let preset = FindleyParameters::pla_25c_moderate();
    assert_eq!(preset.n_int, 3);
    assert_rel(preset.epsilon_0, 0.003, FIX_TOL, "ε₀ preset");
}

// ============================================================================
// thermal_stress — fully restrained bar σ = −E α ΔT
// ============================================================================

/// Boley & Weiner, *Theory of Thermal Stresses* §9.1 / Timoshenko & Goodier
/// "Thermal stress" chapter: a fully restrained bar heated by ΔT carries
/// |σ| = E α ΔT (compressive on heating, tensile on cooling). Partial
/// restraint scales linearly. Yield temperature: ΔT_y = σ_y / (E α c).
#[test]
fn thermal_stress_magnitude_and_yield_temperature_match_closed_form() {
    let m = pla_like(); // E = 3500 MPa, α = 68e-6, σ_y = 50, T_g = 60
    let (e, alpha, sy) = (3500.0f64, 68e-6f64, 50.0f64);
    let r = analyze_thermal_stress(
        &m,
        Fix128::from_ratio(1, 2),
        Fix128::from_int(20),
        Fix128::from_int(35),
    );
    assert_rel(
        r.thermal_stress_mpa.abs(),
        e * alpha * 15.0 * 0.5,
        1e-11,
        "|σ| = c E α ΔT",
    );
    assert_rel(
        r.factor_of_safety,
        sy / (e * alpha * 15.0 * 0.5),
        1e-11,
        "FoS = σ_y/|σ|",
    );
    assert!(!r.near_glass_transition, "35 °C is 25 °C below T_g");
    assert!(
        r.is_safe,
        "FoS {} ≥ 2 and not near T_g",
        r.factor_of_safety.to_f64()
    );
    let yt = yield_temperature_c(&m, Fix128::ONE, Fix128::from_int(20));
    assert_rel(yt, 20.0 + sy / (e * alpha), 1e-11, "T_y = T_i + σ_y/(E α)");
    // at T_y the stress is exactly σ_y → FoS = 1
    let at_yield = analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(20), yt);
    assert_rel(at_yield.factor_of_safety, 1.0, 1e-9, "FoS(T_y) = 1");
    // glass-transition flag: |T − T_g| ≤ 20 °C
    assert!(
        analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(20), Fix128::from_int(45))
            .near_glass_transition
    );
    assert!(
        !analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(20), Fix128::from_int(39))
            .near_glass_transition
    );
}

/// DISCREPANCY (documented, failing on purpose): the report field is
/// documented "positive = tensile", but the module returns
/// σ = +E α (T_op − T_install) c, i.e. positive on heating. A restrained
/// bar heated by ΔT > 0 is in compression (σ = −E α ΔT); a print cooling
/// from 200 °C to 20 °C while held on the bed is in tension (+E α 180).
#[test]
fn thermal_stress_sign_convention_positive_is_tensile() {
    let m = pla_like();
    let cooled =
        analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(200), Fix128::from_int(20));
    assert!(
        cooled.thermal_stress_mpa > Fix128::ZERO,
        "cooling under restraint must be tensile (positive), got {}",
        cooled.thermal_stress_mpa.to_f64()
    );
    let heated =
        analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(20), Fix128::from_int(40));
    assert!(
        heated.thermal_stress_mpa < Fix128::ZERO,
        "heating under restraint must be compressive (negative), got {}",
        heated.thermal_stress_mpa.to_f64()
    );
}
