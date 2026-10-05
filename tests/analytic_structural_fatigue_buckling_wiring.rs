//! Independent oracles for the per-material fatigue and buckling wiring of
//! `structural_solver::StructuralSolver`.
//!
//! `new` selects the S-N curve from the material: the SUS304 / A5052 metal
//! presets for `MaterialProperties::sus304()` / `a5052()` and the FDM rule of
//! thumb for every other material. `with_sn_curve` overrides it. Every
//! expected value below is a closed form evaluated in `f64` from the material
//! data (numbers, not calls into the code under test).
//!
//! ```text
//! bending stress   sigma = F L / Z,  Z = b h^2 / 6 = 1000 mm^3 (15 x 20)
//! S-N data         SUS304  S_e = 240,   N_e = 1e7, m = 10, UTS 505
//!                  A5052   S_e = 92,    N_e = 5e6, m = 6,  UTS 230
//!                  FDM     S_e = 0.3 UTS, N_e = 1e6, m = 5   (PLA UTS 60)
//! Basquin life     N(S) = floor(N_e (S_e / S)^m)   (S > S_e)
//! Miner per step   D_k = k / N(sigma)              (one cycle per step)
//! Basquin inverse  S(N) = S_e (N_e / N)^(1/m),  S_e for N >= N_e
//! Euler column     P_cr = pi^2 E I / (K L)^2,  I = pi d^4 / 64
//!                  K = 1 (pin-pin), 0.5 (fixed-fixed), 2 (cantilever), 0.7 (fixed-pin)
//! plate            sigma_cr = k pi^2 E / (12 (1 - nu^2)) (t / b)^2
//! snap-through     P = (16 / (3 sqrt 3)) E A (h / L)^3
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{ColumnEndCondition, CrossSection, LoadCase};
use alice_physics::buckling::SnapThroughError;
use alice_physics::fatigue::{FatigueRangeError, SnCurve};
use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

/// S-N data as plain numbers: (S_e, N_e, m).
#[derive(Clone, Copy)]
struct Sn {
    se: f64,
    ne: f64,
    m: i32,
}

const SUS304: Sn = Sn {
    se: 240.0,
    ne: 1e7,
    m: 10,
};
const A5052: Sn = Sn {
    se: 92.0,
    ne: 5e6,
    m: 6,
};
/// FDM rule of thumb for PLA (UTS 60 MPa).
const PLA_FDM: Sn = Sn {
    se: 0.3 * 60.0,
    ne: 1e6,
    m: 5,
};

fn basquin_life(sn: Sn, s: f64) -> f64 {
    (sn.ne * (sn.se / s).powi(sn.m)).floor()
}

/// 15 x 20 cantilever of 200 mm with tip load `load_n`:
/// `sigma = 200 load_n / 1000 = load_n / 5` MPa.
fn beam_solver(load_n: i64, material: MaterialProperties) -> StructuralSolver {
    StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: int(15),
            height_mm: int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: int(load_n),
            length_mm: int(200),
        },
        material,
    )
}

fn rel_close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs(),
        "{what}: got {got}, want {want}"
    );
}

/// Run `steps` steps and check `D_k = k / N(sigma)` after every step.
fn assert_miner_per_step(mut s: StructuralSolver, sigma: f64, sn: Sn, steps: u32, what: &str) {
    let n_fail = basquin_life(sn, sigma);
    for k in 1..=steps {
        let r = s.step();
        rel_close(
            r.bending_stress_mpa.to_f64(),
            sigma,
            1e-12,
            &format!("{what} sigma"),
        );
        rel_close(
            r.fatigue_damage.to_f64(),
            f64::from(k) / n_fail,
            1e-12,
            &format!("{what} D after step {k}"),
        );
    }
}

// ---------------------------------------------------------------------------
// Per-material S-N selection
// ---------------------------------------------------------------------------

#[test]
fn new_selects_the_metal_presets_by_material() {
    let steel = beam_solver(1, MaterialProperties::sus304()).sn_curve;
    assert_eq!(
        (
            steel.ultimate_tensile_mpa,
            steel.endurance_stress_mpa,
            steel.endurance_cycles,
            steel.fatigue_exponent_m
        ),
        (int(505), int(240), 10_000_000, 10)
    );
    let alu = beam_solver(1, MaterialProperties::a5052()).sn_curve;
    assert_eq!(
        (
            alu.ultimate_tensile_mpa,
            alu.endurance_stress_mpa,
            alu.endurance_cycles,
            alu.fatigue_exponent_m
        ),
        (int(230), int(92), 5_000_000, 6)
    );
}

#[test]
fn fdm_and_unnamed_sheet_metal_keep_the_fdm_rule_of_thumb() {
    // PLA, PETG and a sheet metal whose name is not a preset: S_e = 0.3 UTS,
    // N_e = 1e6, m = 5 (the curve every material had before the selection).
    let mut other_metal = MaterialProperties::sus304();
    other_metal.name = "SUS316";
    for (material, uts) in [
        (MaterialProperties::pla(), 60),
        (MaterialProperties::petg(), 53),
        (other_metal, 505),
    ] {
        let c = beam_solver(1, material).sn_curve;
        assert_eq!(c.ultimate_tensile_mpa, int(uts), "{}", material.name);
        assert_eq!(
            c.endurance_stress_mpa,
            int(uts) * Fix128::from_ratio(3, 10),
            "{}",
            material.name
        );
        assert_eq!(
            (c.endurance_cycles, c.fatigue_exponent_m),
            (1_000_000, 5),
            "{}",
            material.name
        );
    }
    // A FDM material that happens to carry a metal preset name is not a metal.
    let mut fdm_named = MaterialProperties::pla();
    fdm_named.name = "SUS304";
    assert_eq!(
        beam_solver(1, fdm_named).sn_curve.fatigue_exponent_m,
        5,
        "category decides"
    );
    assert_eq!(
        MaterialProperties::sus304().category,
        MaterialCategory::SheetMetal
    );
}

// ---------------------------------------------------------------------------
// Basquin / Miner damage per step on the selected curve
// ---------------------------------------------------------------------------

#[test]
fn sus304_damage_per_step_follows_basquin_miner() {
    // F = 1500 N -> sigma = 300 MPa, N = floor(1e7 * 0.8^10) = 1 073 741
    assert_eq!(basquin_life(SUS304, 300.0), 1_073_741.0);
    assert_miner_per_step(
        beam_solver(1500, MaterialProperties::sus304()),
        300.0,
        SUS304,
        4,
        "SUS304",
    );
}

#[test]
fn a5052_damage_per_step_follows_basquin_miner() {
    // F = 600 N -> sigma = 120 MPa, N = floor(5e6 (92/120)^6) = 1 015 335
    assert_eq!(basquin_life(A5052, 120.0), 1_015_335.0);
    assert_miner_per_step(
        beam_solver(600, MaterialProperties::a5052()),
        120.0,
        A5052,
        4,
        "A5052",
    );
}

#[test]
fn pla_damage_per_step_is_unchanged_fdm_rule() {
    // F = 200 N -> sigma = 40 MPa, N = floor(1e6 * 0.45^5) = 18 452
    assert_eq!(basquin_life(PLA_FDM, 40.0), 18_452.0);
    assert_miner_per_step(
        beam_solver(200, MaterialProperties::pla()),
        40.0,
        PLA_FDM,
        4,
        "PLA",
    );
}

#[test]
fn same_stress_gives_material_specific_damage() {
    // sigma = 300 MPa on the three curves: SUS304 1/1 073 741, A5052
    // 1/floor(5e6 (92/300)^6) = 1/4 158, PLA 1 (N < 1 clamps to one cycle).
    let d = |m: MaterialProperties| beam_solver(1500, m).step().fatigue_damage.to_f64();
    let steel = d(MaterialProperties::sus304());
    let alu = d(MaterialProperties::a5052());
    let pla = d(MaterialProperties::pla());
    rel_close(steel, 1.0 / basquin_life(SUS304, 300.0), 1e-12, "SUS304");
    assert_eq!(basquin_life(A5052, 300.0), 4_158.0);
    rel_close(alu, 1.0 / 4_158.0, 1e-12, "A5052");
    assert_eq!(pla, 1.0, "PLA");
}

#[test]
fn with_sn_curve_overrides_the_selection_in_step() {
    // PLA beam carrying the SUS304 data as an explicit curve.
    let curve = SnCurve {
        ultimate_tensile_mpa: int(505),
        endurance_stress_mpa: int(240),
        endurance_cycles: 10_000_000,
        fatigue_exponent_m: 10,
    };
    let s = beam_solver(1500, MaterialProperties::pla()).with_sn_curve(curve);
    assert_eq!(s.sn_curve, curve);
    assert_miner_per_step(s, 300.0, SUS304, 3, "PLA with SUS304 curve");
}

// ---------------------------------------------------------------------------
// Basquin inverse and spectrum report on the selected curve
// ---------------------------------------------------------------------------

#[test]
fn fatigue_strength_inverts_basquin_per_material() {
    let cases: [(MaterialProperties, Sn, u64); 3] = [
        (MaterialProperties::sus304(), SUS304, 100_000),
        (MaterialProperties::a5052(), A5052, 500_000),
        (MaterialProperties::pla(), PLA_FDM, 10_000),
    ];
    for (material, sn, n) in cases {
        let s = beam_solver(1, material);
        let got = s.fatigue_strength_mpa(n).unwrap().to_f64();
        let want = sn.se * (sn.ne / n as f64).powf(1.0 / f64::from(sn.m));
        assert!(
            (got - want).abs() < 1e-6,
            "{} N = {n}: got {got}, want {want}",
            material.name
        );
        // at and beyond N_e the endurance stress is returned
        let at_ne = s.fatigue_strength_mpa(sn.ne as u64).unwrap().to_f64();
        assert!((at_ne - sn.se).abs() < 1e-9, "{}", material.name);
    }
    let steel = beam_solver(1, MaterialProperties::sus304());
    assert_eq!(
        steel.fatigue_strength_mpa(0),
        Err(FatigueRangeError::ZeroCycles)
    );
    assert_eq!(
        steel.fatigue_strength_mpa(999),
        Err(FatigueRangeError::BelowLowCycleBound {
            cycles: 999,
            bound: 1_000
        })
    );
}

#[test]
fn spectrum_report_is_miner_sum_on_the_selected_curve() {
    let spectrum = [(int(300), 100_000u64), (int(350), 1_000)];
    let want_steel =
        100_000.0 / basquin_life(SUS304, 300.0) + 1_000.0 / basquin_life(SUS304, 350.0);
    let r = beam_solver(1, MaterialProperties::sus304()).fatigue_spectrum_report(&spectrum);
    rel_close(r.damage.to_f64(), want_steel, 1e-12, "SUS304 D");
    assert!(r.is_safe);
    rel_close(
        r.safety_factor.to_f64(),
        1.0 / want_steel,
        1e-12,
        "SUS304 SF",
    );

    let alu_spectrum = [(int(120), 203_067u64), (int(150), 66_541)];
    let want_alu = 203_067.0 / basquin_life(A5052, 120.0) + 66_541.0 / basquin_life(A5052, 150.0);
    let r = beam_solver(1, MaterialProperties::a5052()).fatigue_spectrum_report(&alu_spectrum);
    rel_close(r.damage.to_f64(), want_alu, 1e-12, "A5052 D");
    assert!(r.is_safe);

    // the same steel spectrum on the PLA curve: every entry fails in < 1 cycle
    let r = beam_solver(1, MaterialProperties::pla()).fatigue_spectrum_report(&spectrum);
    assert_eq!(r.damage, int(101_000));
    assert!(!r.is_safe);
}

// ---------------------------------------------------------------------------
// Buckling
// ---------------------------------------------------------------------------

#[test]
fn column_buckling_fos_is_euler_with_end_condition_factor() {
    // Solid round bar d = 10 mm, L = 1000 mm: lambda = K L / (d/4) >= 200,
    // above the Euler / Johnson transition of all three materials.
    let d = 10.0_f64;
    let i = core::f64::consts::PI * d.powi(4) / 64.0;
    let axial = 100.0;
    for (material, e_mpa) in [
        (MaterialProperties::sus304(), 200_000.0),
        (MaterialProperties::a5052(), 70_000.0),
        (MaterialProperties::pla(), 3_500.0),
    ] {
        for (end, k) in [
            (ColumnEndCondition::PinPin, 1.0),
            (ColumnEndCondition::FixedFixed, 0.5),
            (ColumnEndCondition::Cantilever, 2.0),
            (ColumnEndCondition::FixedPin, 0.7),
        ] {
            let mut s = StructuralSolver::new(
                CrossSection::Circular {
                    diameter_mm: int(10),
                },
                LoadCase::CantileverEndPoint {
                    load_n: Fix128::ZERO,
                    length_mm: int(1000),
                },
                material,
            );
            s.axial_load_n = int(100);
            s.end_condition = end;
            let kl = k * 1000.0;
            let p_cr = core::f64::consts::PI.powi(2) * e_mpa * i / (kl * kl);
            let r = s.step();
            rel_close(
                r.buckling_fos.to_f64(),
                p_cr / axial,
                1e-9,
                &format!("{} K = {k}", material.name),
            );
        }
    }
}

#[test]
fn plate_buckling_uses_the_material_modulus() {
    // t = 1, b = 40, nu = 0.3, k = 4
    let (t, b, nu, k) = (1.0_f64, 40.0_f64, 0.3_f64, 4.0_f64);
    for (material, e_mpa) in [
        (MaterialProperties::sus304(), 200_000.0),
        (MaterialProperties::a5052(), 70_000.0),
        (MaterialProperties::pla(), 3_500.0),
    ] {
        let s = beam_solver(1, material);
        let got = s
            .plate_buckling_mpa(int(1), int(40), Fix128::from_ratio(3, 10), int(4))
            .to_f64();
        let want =
            k * core::f64::consts::PI.powi(2) * e_mpa / (12.0 * (1.0 - nu * nu)) * (t / b).powi(2);
        rel_close(got, want, 1e-9, material.name);
    }
    let s = beam_solver(1, MaterialProperties::sus304());
    assert_eq!(
        s.plate_buckling_mpa(int(1), Fix128::ZERO, Fix128::from_ratio(3, 10), int(4)),
        Fix128::ZERO
    );
}

#[test]
fn snap_through_uses_material_modulus_and_section_area() {
    // bars of the 15 x 20 section (A = 300 mm^2), h = 5, L = 100
    let coefficient = 16.0 / (3.0 * 3.0_f64.sqrt());
    for (material, e_mpa) in [
        (MaterialProperties::pla(), 3_500.0),
        (MaterialProperties::a5052(), 70_000.0),
    ] {
        let s = beam_solver(1, material);
        let got = s.snap_through_load_n(int(5), int(100)).unwrap().to_f64();
        let want = coefficient * e_mpa * 300.0 * 0.05_f64.powi(3);
        rel_close(got, want, 1e-9, material.name);
    }
    let s = beam_solver(1, MaterialProperties::pla());
    assert_eq!(
        s.snap_through_load_n(int(5), Fix128::ZERO),
        Err(SnapThroughError::NonPositiveSpan)
    );
    assert_eq!(
        s.snap_through_load_n(int(-1), int(100)),
        Err(SnapThroughError::NegativeRise)
    );
}
