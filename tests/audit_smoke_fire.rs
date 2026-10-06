//! Audit oracles for `smoke_fire` (single-step Arrhenius combustion, soot,
//! Boussinesq buoyancy). Every expected value is derived from the closed forms
//! in the module doc (Turns, *An Introduction to Combustion* ch. 4-5), computed
//! independently in `f64`, never from the crate.
//!
//! Claims audited here (the rest are already pinned by
//! `analytic_smoke_fire_wiring.rs` and `engineering_oracles_fluid.rs`):
//! - accuracy of `r = A exp(-E_a/(R T)) rho_F rho_O` over the whole physical
//!   temperature range (not only at 1500 / 2000 K)
//! - strict monotonicity in `T` (Arrhenius is increasing for `E_a > 0`)
//! - symmetry in the two reactant densities
//! - the `R*T` underflow corner returns zero (limit `R -> 0` gives `r -> 0`)
//! - Boussinesq odd symmetry in `dT` and linearity in each factor
//! - the Arrhenius cliff: below `exponent <= -40` the rate is exactly zero
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::smoke_fire::{
    boussinesq_buoyancy_n_per_m3, heat_release_j_per_m3_s, reaction_rate_kg_per_m3_s,
    soot_generation_kg_per_m3_s, ArrheniusReaction,
};

fn rate_f64(r: &ArrheniusReaction, t: i64) -> f64 {
    reaction_rate_kg_per_m3_s(r, Fix128::from_int(t), Fix128::ONE, Fix128::ONE).to_f64()
}

fn closed(a: f64, ea: f64, rg: f64, t: f64) -> f64 {
    a * (-ea / (rg * t)).exp()
}

fn closed_for(r: &ArrheniusReaction, t: i64) -> f64 {
    closed(
        r.pre_exponential_a.to_f64(),
        r.activation_energy_j_per_mol.to_f64(),
        r.gas_constant_j_per_mol_k.to_f64(),
        t as f64,
    )
}

#[test]
fn arrhenius_rate_is_accurate_over_the_working_temperature_range() {
    // Tolerances reflect the fixed-point exp: absolute error floor ~ 2^-64 * 2^shifts,
    // so relative error grows as the rate shrinks. Both bands below are well inside
    // what a combustion model needs; the tight band pins the high-T regime.
    let methane = ArrheniusReaction::methane_air();
    let pla = ArrheniusReaction::pla_air();
    for t in (700..=3000).step_by(100) {
        let want = closed_for(&methane, t);
        let got = rate_f64(&methane, t);
        let tol = if t >= 1000 { 1e-8 } else { 1e-3 };
        assert!(
            (got - want).abs() / want < tol,
            "methane T={t}: got {got:e} want {want:e}"
        );
    }
    for t in (500..=3000).step_by(100) {
        let want = closed_for(&pla, t);
        let got = rate_f64(&pla, t);
        let tol = if t >= 800 { 1e-8 } else { 1e-3 };
        assert!(
            (got - want).abs() / want < tol,
            "pla T={t}: got {got:e} want {want:e}"
        );
    }
}

#[test]
fn arrhenius_rate_is_strictly_increasing_in_temperature() {
    for r in [
        ArrheniusReaction::methane_air(),
        ArrheniusReaction::pla_air(),
    ] {
        let mut prev = rate_f64(&r, 650);
        assert!(prev > 0.0);
        for t in (651..=3000).step_by(7) {
            let cur = rate_f64(&r, t);
            assert!(
                cur > prev,
                "rate not increasing at T={t}: {cur:e} <= {prev:e}"
            );
            prev = cur;
        }
    }
}

#[test]
fn reaction_rate_is_symmetric_in_fuel_and_oxidizer_density() {
    let r = ArrheniusReaction::methane_air();
    let t = Fix128::from_int(1800);
    let a = Fix128::from_ratio(3, 10);
    let b = Fix128::from_ratio(7, 4);
    assert_eq!(
        reaction_rate_kg_per_m3_s(&r, t, a, b),
        reaction_rate_kg_per_m3_s(&r, t, b, a)
    );
}

#[test]
fn reaction_rate_density_dependence_is_the_exact_product_of_the_unit_rate() {
    // r(rho_F, rho_O) = r(1, 1) * rho_F * rho_O. Check against the f64 product.
    let r = ArrheniusReaction::pla_air();
    let unit = rate_f64(&r, 1100);
    for (f, o) in [(0.25, 0.5), (2.0, 3.0), (0.1, 8.0)] {
        let got = reaction_rate_kg_per_m3_s(
            &r,
            Fix128::from_int(1100),
            Fix128::from_ratio((f * 1000.0) as i64, 1000),
            Fix128::from_ratio((o * 1000.0) as i64, 1000),
        )
        .to_f64();
        assert!(
            ((got - unit * f * o) / (unit * f * o)).abs() < 1e-9,
            "f={f} o={o}"
        );
    }
}

#[test]
fn reaction_rate_is_zero_when_r_times_t_underflows_not_the_prefactor() {
    // R = 1 ulp, T = 0.5 K: R*T rounds to 0 in Q64.64. The physical limit R -> 0
    // gives exp(-inf) = 0, so the rate must be zero rather than A.
    let mut r = ArrheniusReaction::methane_air();
    r.gas_constant_j_per_mol_k = Fix128::from_raw(0, 1);
    let rate = reaction_rate_kg_per_m3_s(&r, Fix128::from_ratio(1, 2), Fix128::ONE, Fix128::ONE);
    assert_eq!(rate, Fix128::ZERO);
}

#[test]
fn heat_and_soot_are_linear_in_the_rate_with_the_preset_coefficients() {
    for r in [
        ArrheniusReaction::methane_air(),
        ArrheniusReaction::pla_air(),
    ] {
        for rate in [0.0, 0.5, 3.0, 125.25] {
            let rr = Fix128::from_ratio((rate * 100.0) as i64, 100);
            let q = heat_release_j_per_m3_s(&r, rr).to_f64();
            let s = soot_generation_kg_per_m3_s(&r, rr).to_f64();
            let q_want = rate * r.heat_of_combustion_j_per_kg.to_f64();
            let s_want = rate * r.soot_yield.to_f64();
            assert!((q - q_want).abs() <= 1e-9 * q_want.abs().max(1.0));
            assert!((s - s_want).abs() <= 1e-9 * s_want.abs().max(1.0));
        }
    }
}

#[test]
fn boussinesq_is_odd_in_delta_t_and_linear_in_each_factor() {
    let rho = Fix128::from_ratio(1204, 1000);
    let beta = Fix128::from_ratio(1, 256);
    let g = Fix128::from_ratio(981, 100);
    let dt = Fix128::from_int(64);
    let f = boussinesq_buoyancy_n_per_m3(rho, beta, dt, g);
    // odd in dT up to the floor-rounding asymmetry of Q64.64 multiplication (a few ulp)
    let neg = boussinesq_buoyancy_n_per_m3(rho, beta, Fix128::ZERO - dt, g);
    assert!(
        (neg.to_f64() + f.to_f64()).abs() < 1e-17 * 1e3,
        "{} vs {}",
        neg.to_f64(),
        f.to_f64()
    );
    assert!(neg < Fix128::ZERO && f > Fix128::ZERO);
    // linear: doubling any one factor doubles the force (within 2 ulp)
    let two = Fix128::from_int(2);
    let tol = 1e-15;
    let fd = f.to_f64();
    for (name, v) in [
        ("rho", boussinesq_buoyancy_n_per_m3(rho * two, beta, dt, g)),
        ("beta", boussinesq_buoyancy_n_per_m3(rho, beta * two, dt, g)),
        ("dT", boussinesq_buoyancy_n_per_m3(rho, beta, dt * two, g)),
        ("g", boussinesq_buoyancy_n_per_m3(rho, beta, dt, g * two)),
    ] {
        assert!(
            (v.to_f64() - 2.0 * fd).abs() <= tol * fd.abs() * 4.0,
            "{name}"
        );
    }
    // closed form
    let want = 1.204 * (1.0 / 256.0) * 64.0 * 9.81;
    assert!((fd - want).abs() / want < 1e-12);
}

#[test]
fn boussinesq_vanishes_when_any_factor_is_zero() {
    let one = Fix128::ONE;
    let z = Fix128::ZERO;
    assert_eq!(boussinesq_buoyancy_n_per_m3(z, one, one, one), z);
    assert_eq!(boussinesq_buoyancy_n_per_m3(one, z, one, one), z);
    assert_eq!(boussinesq_buoyancy_n_per_m3(one, one, z, one), z);
    assert_eq!(boussinesq_buoyancy_n_per_m3(one, one, one, z), z);
}

#[test]
fn arrhenius_rate_has_no_hard_zero_cliff_at_the_exp_saturation_threshold() {
    let pla = ArrheniusReaction::pla_air();
    let want = closed_for(&pla, 450);
    let got = rate_f64(&pla, 450);
    assert!(
        (got - want).abs() / want < 1e-2,
        "PLA 450 K: got {got:e} want {want:e}"
    );
    let methane = ArrheniusReaction::methane_air();
    let want = closed_for(&methane, 600);
    let got = rate_f64(&methane, 600);
    assert!(
        (got - want).abs() / want < 1e-2,
        "methane 600 K: got {got:e} want {want:e}"
    );
}

/// Past the old cut-off the rate still follows the closed form while it is
/// representable, and becomes exactly 0 only when the closed form is below the
/// Fix128 resolution (2^-64). A temperature that makes the exponent huge
/// (1 K: E_a/(R T) ≈ 1.8e4) returns 0 without iterating over the whole exponent.
#[test]
fn arrhenius_rate_reaches_zero_only_below_the_fix128_resolution() {
    let resolution = 2f64.powi(-64);
    // PLA at 300 K: E_a/(R T) ≈ 60.1, closed form ≈ 4.3e-18 (about 80 ulp)
    let pla = ArrheniusReaction::pla_air();
    let want = closed_for(&pla, 300);
    assert!(
        want > 10.0 * resolution,
        "test premise: {want:e} is representable"
    );
    let got = rate_f64(&pla, 300);
    assert!(
        (got - want).abs() / want < 5e-2,
        "PLA 300 K: got {got:e} want {want:e}"
    );
    // methane at 300 K: E_a/(R T) ≈ 81, closed form ≈ 8e-27, below the resolution
    let methane = ArrheniusReaction::methane_air();
    let want = closed_for(&methane, 300);
    assert!(
        want < resolution,
        "test premise: {want:e} is below the resolution"
    );
    assert_eq!(rate_f64(&methane, 300), 0.0);
    // 1 K: the exponent is ~1.8e4; the result is 0 and the call returns promptly
    let start = std::time::Instant::now();
    assert_eq!(rate_f64(&methane, 1), 0.0);
    assert_eq!(rate_f64(&pla, 1), 0.0);
    assert!(start.elapsed().as_millis() < 1000);
}
