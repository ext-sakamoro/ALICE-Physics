//! Independent oracles for the per-material creep selection of
//! `structural_solver::StructuralSolver::{new, with_creep, creep_modelled}`.
//!
//! `new` installs the PLA creep presets only for the PLA preset material;
//! every other material starts with creep not modelled (reported creep
//! strain exactly zero). `with_creep` supplies parameters for any material.
//!
//! Closed forms (step `k = 1, 2, ...`, `dt` in seconds, `t` in hours; the
//! Findley projection is evaluated at the elapsed time *before* the step
//! advances, i.e. `t_k = (k - 1) dt / 3600`):
//!
//! ```text
//! sigma            = M / Z = F L / (b h^2 / 6)
//! Norton (state)   eps_N(k) = k * A sigma^n_N dt
//! Findley          eps_F(k) = e0 + m t_k^n_F           (T <= T_g: no WLF shift)
//! reported creep   max(eps_N(k), eps_F(k))
//! not modelled     reported creep = eps_N = 0 for every k
//! PLA presets      e0 = 0.003, m = 8.3e-14 /h^3, n_F = 3;  A = 6.34e-13 /(MPa^3 s), n_N = 3
//! ```
//!
//! The PLA coefficient values are read from the preset constructors as
//! numbers (they are data, not the code under test) so that the tolerance
//! only has to absorb Fix128 rounding of the arithmetic.

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::creep_longterm::FindleyParameters;
use alice_physics::filament_db::{MaterialCategory, MaterialProperties};
use alice_physics::math::Fix128;
use alice_physics::plastic::NortonCreep;
use alice_physics::structural_solver::StructuralSolver;

/// 1000 h per step (seconds).
const DT_S: i64 = 3_600_000;
const DT_H: f64 = 1000.0;
const STEPS: u32 = 5;

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

/// 15x20 section (`Z = 15 * 400 / 6 = 1000 mm^3`), tip load `load_n` on
/// 200 mm: `sigma = 200 load_n / 1000 = load_n / 5` MPa.
fn solver(load_n: i64, material: MaterialProperties) -> StructuralSolver {
    let mut s = StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: int(15),
            height_mm: int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: int(load_n),
            length_mm: int(200),
        },
        material,
    );
    s.dt_s = int(DT_S);
    s
}

fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let e = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(e <= tol, "{what}: got {g:e}, want {want:e} (err {e:.3e})");
}

/// PLA preset coefficients as plain numbers: `(e0, m, n_F, A, n_N)`.
fn pla_coefficients() -> (f64, f64, i32, f64, i32) {
    let f = FindleyParameters::pla_25c_moderate();
    let n = NortonCreep::pla_room_temp();
    (
        f.epsilon_0.to_f64(),
        f.m.to_f64(),
        f.n_int as i32,
        n.a.to_f64(),
        n.n as i32,
    )
}

/// Closed form of the reported creep at step `k` for parameters
/// `(e0, m, n_F, A, n_N)` under stress `sigma`.
fn reported(k: u32, sigma: f64, c: (f64, f64, i32, f64, i32)) -> f64 {
    let (e0, m, nf, a, nn) = c;
    let norton = f64::from(k) * a * sigma.powi(nn) * (DT_H * 3600.0);
    let t = f64::from(k - 1) * DT_H;
    let findley = e0 + m * t.powi(nf);
    norton.max(findley)
}

fn norton_state(k: u32, sigma: f64, c: (f64, f64, i32, f64, i32)) -> f64 {
    f64::from(k) * c.3 * sigma.powi(c.4) * (DT_H * 3600.0)
}

// ---------------------------------------------------------------------------
// (a) non-PLA: creep not modelled, exactly zero
// ---------------------------------------------------------------------------

/// PETG, ABS (FDM) and SUS304 (sheet metal) under a constant 2 MPa: the
/// reported creep and the Norton state are exactly zero for every step, and
/// the solver says creep is not modelled.
#[test]
fn non_pla_materials_report_exactly_zero_creep_and_not_modelled() {
    for material in [
        MaterialProperties::petg(),
        MaterialProperties::abs(),
        MaterialProperties::sus304(),
        MaterialProperties::a5052(),
    ] {
        let mut s = solver(10, material);
        assert!(!s.creep_modelled(), "{}: creep_modelled", material.name);
        for k in 1..=STEPS {
            let r = s.step();
            assert_rel(r.bending_stress_mpa, 2.0, 1e-12, "sigma = M/Z");
            assert_eq!(
                r.creep_strain,
                Fix128::ZERO,
                "{} step {k}: reported creep",
                material.name
            );
        }
        let h = s.run(0);
        assert_eq!(
            h.plastic_state.creep_strain,
            Fix128::ZERO,
            "{}",
            material.name
        );
    }
}

/// The PLA rule is the exact preset name: a user material called "pla"
/// (lower case) is not the PLA preset and gets no creep; a material derived
/// from `MaterialProperties::pla()` with an edited strength keeps it.
#[test]
fn pla_identification_is_the_preset_name_and_fdm_category() {
    let mut lower = MaterialProperties::pla();
    lower.name = "pla";
    assert!(!solver(10, lower).creep_modelled());

    let mut sheet = MaterialProperties::pla();
    sheet.category = MaterialCategory::SheetMetal;
    assert!(!solver(10, sheet).creep_modelled());

    let mut stronger = MaterialProperties::pla();
    stronger.yield_strength_mpa = int(80);
    assert!(solver(10, stronger).creep_modelled());
    assert!(solver(10, MaterialProperties::pla()).creep_modelled());
}

// ---------------------------------------------------------------------------
// (b) PLA: Findley + Norton closed forms, unchanged
// ---------------------------------------------------------------------------

/// PLA at 25 C (below T_g = 60 C, no WLF shift), 2 MPa, 1000 h steps.
/// Per step `A sigma^3 dt = 6.34e-13 * 8 * 3.6e6 = 1.826e-5` (Norton) and
/// `e0 + m t^3 = 0.003 + 8.3e-5 (k-1)^3` (Findley), so Findley is reported.
#[test]
fn pla_creep_matches_findley_and_norton_closed_forms() {
    let c = pla_coefficients();
    let mut s = solver(10, MaterialProperties::pla());
    for k in 1..=STEPS {
        let r = s.step();
        assert_rel(
            r.creep_strain,
            reported(k, 2.0, c),
            1e-12,
            &format!("PLA reported creep step {k}"),
        );
        assert_rel(
            s.state.creep_strain,
            norton_state(k, 2.0, c),
            1e-12,
            &format!("PLA Norton state step {k}"),
        );
    }
    // the decimal values of the closed form, as a cross-check of the presets
    assert_rel(
        s.state.creep_strain,
        5.0 * 6.34e-13 * 8.0 * 3.6e6,
        1e-6,
        "5 A sigma^3 dt",
    );
}

/// 40 MPa (cantilever 200 N, elastic: below yield 50) and 1 h steps: Norton
/// `k * 6.34e-13 * 64000 * 3600 = 1.461e-4 k` overtakes Findley
/// `0.003 + 8.3e-14 (k-1)^3` from step 21 on, so both branches of the max
/// are exercised for PLA.
#[test]
fn pla_reported_creep_switches_from_findley_to_norton() {
    let (e0, m, nf, a, nn) = pla_coefficients();
    let mut s = solver(200, MaterialProperties::pla());
    s.dt_s = int(3600);
    let mut saw_findley = false;
    let mut saw_norton = false;
    for k in 1..=30u32 {
        let r = s.step();
        let norton = f64::from(k) * a * 40f64.powi(nn) * 3600.0;
        let findley = e0 + m * f64::from(k - 1).powi(nf);
        saw_findley |= findley > norton;
        saw_norton |= norton > findley;
        assert_rel(
            r.creep_strain,
            norton.max(findley),
            1e-12,
            &format!("step {k}"),
        );
    }
    assert!(saw_findley && saw_norton);
}

// ---------------------------------------------------------------------------
// (c) with_creep
// ---------------------------------------------------------------------------

/// PETG + `with_creep(PLA presets)` reports the same creep as the PLA
/// material (bit for bit) and as the closed form: below both glass
/// transitions the material only enters through `sigma = M / Z`.
#[test]
fn non_pla_with_pla_creep_equals_the_pla_material() {
    let c = pla_coefficients();
    let mut petg = solver(10, MaterialProperties::petg()).with_creep(
        FindleyParameters::pla_25c_moderate(),
        NortonCreep::pla_room_temp(),
    );
    assert!(petg.creep_modelled());
    let mut pla = solver(10, MaterialProperties::pla());
    for k in 1..=STEPS {
        let rp = petg.step();
        let rl = pla.step();
        assert_eq!(rp.creep_strain, rl.creep_strain, "step {k}");
        assert_rel(rp.creep_strain, reported(k, 2.0, c), 1e-12, "closed form");
    }
    assert_eq!(petg.state.creep_strain, pla.state.creep_strain);
}

/// SUS304 (T_g field 0, so no WLF) with user parameters
/// `e0 = 0, m = 1e-6 /h, n_F = 1; A = 1e-12 /(MPa s), n_N = 1` at 2 MPa:
/// step 1 reports Norton `1e-12 * 2 * 3.6e6 = 7.2e-6` (Findley is 0 at
/// t = 0), later steps report Findley `1e-6 * 1000 (k-1) = 1e-3 (k-1)`.
#[test]
fn with_creep_installs_the_given_parameters_on_any_material() {
    let findley = FindleyParameters {
        epsilon_0: Fix128::ZERO,
        m: Fix128::from_ratio(1, 1_000_000),
        n_int: 1,
    };
    let norton = NortonCreep {
        a: Fix128::from_ratio(1, 1_000_000_000_000),
        n: 1,
    };
    let c = (0.0, findley.m.to_f64(), 1, norton.a.to_f64(), 1);
    let mut s = solver(10, MaterialProperties::sus304()).with_creep(findley, norton);
    assert!(s.creep_modelled());
    assert_eq!(s.creep_params, findley);
    assert_eq!(s.norton_creep, norton);
    let r1 = s.step();
    assert_rel(r1.creep_strain, 7.2e-6, 1e-6, "step 1 = Norton");
    assert_rel(r1.creep_strain, reported(1, 2.0, c), 1e-12, "step 1");
    for k in 2..=STEPS {
        let r = s.step();
        assert_rel(r.creep_strain, 1e-3 * f64::from(k - 1), 1e-6, "Findley");
        assert_rel(r.creep_strain, reported(k, 2.0, c), 1e-12, "closed form");
    }
}

/// `with_creep` with all-zero coefficients is the not-modelled state, also
/// for PLA.
#[test]
fn with_creep_zero_coefficients_is_not_modelled() {
    let zero_f = FindleyParameters {
        epsilon_0: Fix128::ZERO,
        m: Fix128::ZERO,
        n_int: 3,
    };
    let zero_n = NortonCreep {
        a: Fix128::ZERO,
        n: 3,
    };
    let mut s = solver(10, MaterialProperties::pla()).with_creep(zero_f, zero_n);
    assert!(!s.creep_modelled());
    for _ in 0..STEPS {
        assert_eq!(s.step().creep_strain, Fix128::ZERO);
    }
}

// ---------------------------------------------------------------------------
// (d) degenerate inputs
// ---------------------------------------------------------------------------

/// Zero load: Norton gives zero for every material. The PLA Findley term
/// `e0 + m t^3` does not depend on the load, so PLA still reports it
/// (unchanged behaviour); a non-PLA material reports zero.
#[test]
fn zero_load_gives_zero_norton_and_zero_creep_for_non_pla() {
    let (e0, m, nf, _, _) = pla_coefficients();
    let mut pla = solver(0, MaterialProperties::pla());
    let mut petg = solver(0, MaterialProperties::petg());
    for k in 1..=STEPS {
        let rl = pla.step();
        let rp = petg.step();
        assert_eq!(rl.bending_stress_mpa, Fix128::ZERO);
        assert_eq!(
            pla.state.creep_strain,
            Fix128::ZERO,
            "PLA Norton at sigma 0"
        );
        let t = f64::from(k - 1) * DT_H;
        assert_rel(rl.creep_strain, e0 + m * t.powi(nf), 1e-12, "PLA Findley");
        assert_eq!(rp.creep_strain, Fix128::ZERO);
    }
    assert_eq!(petg.state.creep_strain, Fix128::ZERO);
}

/// `dt = 0`: time does not advance, Norton adds nothing and Findley stays at
/// `t = 0` (`e0` for PLA, 0 for a non-PLA material). Zero steps leave the
/// state untouched.
#[test]
fn zero_dt_and_zero_steps() {
    let (e0, ..) = pla_coefficients();
    let mut pla = solver(10, MaterialProperties::pla());
    let mut petg = solver(10, MaterialProperties::petg());
    pla.dt_s = Fix128::ZERO;
    petg.dt_s = Fix128::ZERO;
    for _ in 0..STEPS {
        let rl = pla.step();
        let rp = petg.step();
        assert_eq!(rl.elapsed_hours, Fix128::ZERO);
        assert_rel(rl.creep_strain, e0, 1e-15, "PLA e0 at t = 0");
        assert_eq!(rp.creep_strain, Fix128::ZERO);
    }
    assert_eq!(pla.state.creep_strain, Fix128::ZERO);

    for material in [MaterialProperties::pla(), MaterialProperties::petg()] {
        let mut s = solver(10, material);
        let h = s.run(0);
        assert_eq!(h.steps, 0);
        assert_eq!(h.plastic_state.creep_strain, Fix128::ZERO);
    }
}
