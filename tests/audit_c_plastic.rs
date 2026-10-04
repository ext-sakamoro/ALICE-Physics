//! Audit oracles for `plastic`: the yield boundary of the radial return,
//! observed through the public `StructuralSolver::step`.
//!
//! The yield function is `f = |sigma - alpha| - sigma_y`. The step is elastic
//! for `f <= 0` (Simo and Hughes, ch. 2: the trial state is admissible on the
//! surface), and the doc of the step result says yielding is reported iff the
//! plastic strain increment is positive. On the surface the increment
//! `f / (E + H)` is exactly zero, so a stress exactly at the yield stress is
//! an elastic step.
//!
//! The solver reports a failure when the radial return yields, when the
//! fatigue damage reaches one, when the beam is unsafe or when the column
//! buckles. The scene below disables the other three so `failed_this_step`
//! is the yield flag alone: the material yield (read by the beam check) is
//! far above the stress, the endurance stress is far above the stress, and
//! there is no axial load. The plastic model's own yield stress is a public
//! field and is set independently of the material.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::plastic::HardeningType;
use alice_physics::structural_solver::StructuralSolver;

fn base_solver(hardening: HardeningType) -> StructuralSolver {
    let mut material = MaterialProperties::pla();
    material.yield_strength_mpa = Fix128::from_int(100_000);
    material.tensile_strength_mpa = Fix128::from_int(100_000);
    let mut s = StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: Fix128::from_int(15),
            height_mm: Fix128::from_int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(100),
            length_mm: Fix128::from_int(200),
        },
        material,
    );
    s.sn_curve.endurance_stress_mpa = Fix128::from_int(100_000);
    s.plastic_model.hardening_type = hardening;
    s
}

/// The bending stress of the scene, read from a probe solver. It is the
/// input the yield stress is set to, not an expected value.
fn bending_stress() -> Fix128 {
    let mut probe = base_solver(HardeningType::Isotropic);
    probe.plastic_model.yield_strength_mpa = Fix128::from_int(100_000);
    let report = probe.step();
    assert!(!report.failed_this_step, "the probe scene must not fail");
    report.bending_stress_mpa
}

fn step_with_yield(hardening: HardeningType, yield_mpa: Fix128) -> (bool, Fix128) {
    let mut s = base_solver(hardening);
    s.plastic_model.yield_strength_mpa = yield_mpa;
    let report = s.step();
    (report.failed_this_step, report.plastic_strain)
}

/// On the surface the step is elastic for every hardening law; a margin of
/// `2^-20` MPa on either side moves it across. For isotropic, kinematic and
/// combined hardening the current yield stress at zero plastic strain and
/// zero back stress is `sigma_y0`, so the same boundary applies to all three.
#[test]
fn trial_stress_exactly_on_the_yield_surface_is_an_elastic_step() {
    let sigma = bending_stress();
    assert!(sigma > Fix128::ZERO);
    let margin = Fix128::from_raw(0, 1 << 44); // 2^-20
    for hardening in [
        HardeningType::Isotropic,
        HardeningType::Kinematic,
        HardeningType::Combined,
    ] {
        let (failed, eps_p) = step_with_yield(hardening, sigma);
        assert!(
            !failed && eps_p.is_zero(),
            "{hardening:?}: on the surface, failed = {failed}, eps_p = {eps_p:?}"
        );

        let (failed, eps_p) = step_with_yield(hardening, sigma + margin);
        assert!(
            !failed && eps_p.is_zero(),
            "{hardening:?}: inside, failed = {failed}, eps_p = {eps_p:?}"
        );

        let (failed, eps_p) = step_with_yield(hardening, sigma - margin);
        assert!(
            failed && eps_p > Fix128::ZERO,
            "{hardening:?}: outside, failed = {failed}, eps_p = {eps_p:?}"
        );
    }
}
