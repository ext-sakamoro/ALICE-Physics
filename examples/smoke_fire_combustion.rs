//! Driving the Arrhenius combustion / soot / Boussinesq buoyancy formulas
//! for both material presets through production entry points.
//!
//! `wiring_guard.py` found six zero-production-caller items in
//! `src/smoke_fire.rs`: `ArrheniusReaction::{methane_air, pla_air}`,
//! `reaction_rate_kg_per_m3_s`, `heat_release_j_per_m3_s`,
//! `soot_generation_kg_per_m3_s`, `boussinesq_buoyancy_n_per_m3`. Only the
//! module's own `#[cfg(test)]` block and
//! `tests/engineering_oracles_fluid.rs::smoke_fire_arrhenius_and_boussinesq_closed_forms`
//! (methane preset only, `T` in `{1500, 2000}` K, Boussinesq at
//! `rho=1.204, beta=1/300, dT=100, g=9.81`) called into the module before
//! this example — the guard does not count `tests/` as production (src /
//! examples / benches / fuzz / bindings only).
//!
//! Closed forms, from the module doc (`src/smoke_fire.rs:1-20`; Turns,
//! *An Introduction to Combustion* ch. 5; Kuo, *Principles of Combustion*):
//!
//! ```text
//! reaction rate   : r      = A * exp(-E_a / (R * T)) * rho_fuel * rho_oxidizer
//! heat release    : q_dot  = r * dH_c
//! soot generation : s_dot  = r * Y_s
//! Boussinesq      : f_b    = rho_0 * beta * (T - T_0) * g
//! ```
//!
//! This example extends the existing oracle's methane-only, two-temperature
//! coverage to the `pla_air` preset (not exercised anywhere before this),
//! additional methane temperatures, a bilinear-in-density check against the
//! `pla_air` preset (the existing oracle only checked this for methane),
//! and a Boussinesq scenario distinct from the one above. It drives
//! `heat_release_j_per_m3_s` / `soot_generation_kg_per_m3_s` off the
//! *actual* `reaction_rate_kg_per_m3_s` output for each preset — the way a
//! real combustion-coupled solver would chain them — while still deriving
//! every expected value from the closed form, never from the function
//! under test (the expected reaction rate is computed once, independently,
//! in `f64`, then reused both to check the rate itself and to build the
//! expected heat/soot products).
//!
//! ```bash
//! cargo run --example smoke_fire_combustion --features std
//! ```

// The closed-form reference values below are independent f64 evaluations
// of the Arrhenius formula, not simulation state; this mirrors
// `tests/engineering_oracles_fluid.rs`'s `#![allow(clippy::disallowed_methods)]`.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::smoke_fire::{
    boussinesq_buoyancy_n_per_m3, heat_release_j_per_m3_s, reaction_rate_kg_per_m3_s,
    soot_generation_kg_per_m3_s, ArrheniusReaction,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

/// Arrhenius reaction rate closed form, `r = A exp(-E_a/(R T)) rho_F rho_O`,
/// computed independently in `f64` (never by calling the crate).
fn arrhenius_rate_f64(a: f64, ea: f64, r_gas: f64, t: f64, rho_f: f64, rho_o: f64) -> f64 {
    a * (-ea / (r_gas * t)).exp() * rho_f * rho_o
}

fn main() {
    let methane = ArrheniusReaction::methane_air();
    let pla = ArrheniusReaction::pla_air();
    let r_gas = 8.314_f64;

    // --- preset values match the documented constants exactly -----------
    // These are struct literals, not a derived formula: the "closed form"
    // here is the module doc / README table itself, written down
    // independently of `ArrheniusReaction::{methane_air, pla_air}`.
    assert_eq!(methane.pre_exponential_a, Fix128::from_int(1_300_000_000));
    assert_eq!(
        methane.activation_energy_j_per_mol,
        Fix128::from_int(202_000)
    );
    assert_eq!(
        methane.heat_of_combustion_j_per_kg,
        Fix128::from_int(50_000_000)
    );
    assert_eq!(methane.soot_yield, Fix128::from_ratio(15, 1000));
    assert_eq!(pla.pre_exponential_a, Fix128::from_int(500_000_000));
    assert_eq!(pla.activation_energy_j_per_mol, Fix128::from_int(150_000));
    assert_eq!(
        pla.heat_of_combustion_j_per_kg,
        Fix128::from_int(18_000_000)
    );
    assert_eq!(pla.soot_yield, Fix128::from_ratio(20, 1000));
    println!("[smoke_fire] methane_air / pla_air presets match documented constants exactly");

    // --- reaction_rate_kg_per_m3_s / heat_release / soot, chained off the
    // actual preset, methane at T in {1600, 2200} K (existing oracle used
    // {1500, 2000}) --------------------------------------------------------
    for &t in &[1600.0_f64, 2200.0_f64] {
        let rate_fix = reaction_rate_kg_per_m3_s(
            &methane,
            Fix128::from_int(t as i64),
            Fix128::ONE,
            Fix128::ONE,
        );
        let want_rate = arrhenius_rate_f64(1.3e9, 202_000.0, r_gas, t, 1.0, 1.0);
        let rate = rate_fix.to_f64();
        println!("[smoke_fire] methane r({t} K) = {rate:e} kg/(m3 s) (closed form {want_rate:e})");
        assert!(
            rel_err(rate, want_rate) < 1e-6,
            "methane r({t}) = {rate} vs {want_rate}"
        );

        let want_heat = want_rate * 50_000_000.0;
        let heat = heat_release_j_per_m3_s(&methane, rate_fix).to_f64();
        println!(
            "[smoke_fire] methane q_dot({t} K) = {heat:e} J/(m3 s) (closed form {want_heat:e})"
        );
        assert!(
            rel_err(heat, want_heat) < 1e-6,
            "methane q_dot({t}) = {heat} vs {want_heat}"
        );

        let want_soot = want_rate * 0.015;
        let soot = soot_generation_kg_per_m3_s(&methane, rate_fix).to_f64();
        println!(
            "[smoke_fire] methane s_dot({t} K) = {soot:e} kg/(m3 s) (closed form {want_soot:e})"
        );
        assert!(
            rel_err(soot, want_soot) < 1e-6,
            "methane s_dot({t}) = {soot} vs {want_soot}"
        );
    }

    // --- same chain for pla_air (not exercised anywhere before this), T in
    // {900, 1300} K --------------------------------------------------------
    for &t in &[900.0_f64, 1300.0_f64] {
        let rate_fix =
            reaction_rate_kg_per_m3_s(&pla, Fix128::from_int(t as i64), Fix128::ONE, Fix128::ONE);
        let want_rate = arrhenius_rate_f64(5.0e8, 150_000.0, r_gas, t, 1.0, 1.0);
        let rate = rate_fix.to_f64();
        println!("[smoke_fire] pla r({t} K) = {rate:e} kg/(m3 s) (closed form {want_rate:e})");
        assert!(
            rel_err(rate, want_rate) < 1e-6,
            "pla r({t}) = {rate} vs {want_rate}"
        );

        let want_heat = want_rate * 18_000_000.0;
        let heat = heat_release_j_per_m3_s(&pla, rate_fix).to_f64();
        println!("[smoke_fire] pla q_dot({t} K) = {heat:e} J/(m3 s) (closed form {want_heat:e})");
        assert!(
            rel_err(heat, want_heat) < 1e-6,
            "pla q_dot({t}) = {heat} vs {want_heat}"
        );

        let want_soot = want_rate * 0.020;
        let soot = soot_generation_kg_per_m3_s(&pla, rate_fix).to_f64();
        println!("[smoke_fire] pla s_dot({t} K) = {soot:e} kg/(m3 s) (closed form {want_soot:e})");
        assert!(
            rel_err(soot, want_soot) < 1e-6,
            "pla s_dot({t}) = {soot} vs {want_soot}"
        );
    }

    // --- bilinear in reactant densities, pla preset at T=1300 K (methane's
    // own bilinear-in-density case is already covered in
    // tests/engineering_oracles_fluid.rs) ----------------------------------
    let base =
        reaction_rate_kg_per_m3_s(&pla, Fix128::from_int(1300), Fix128::ONE, Fix128::ONE).to_f64();
    let scaled = reaction_rate_kg_per_m3_s(
        &pla,
        Fix128::from_int(1300),
        Fix128::from_ratio(1, 4),
        Fix128::from_int(8),
    )
    .to_f64();
    println!("[smoke_fire] pla r(1300K, 1/4 fuel, 8x oxidizer) = {scaled:e} (2x base {base:e})");
    assert!(
        rel_err(scaled, 2.0 * base) < 1e-6,
        "bilinear scaling: {scaled} vs {}",
        2.0 * base
    );

    // --- Boussinesq buoyancy, scenario distinct from
    // tests/engineering_oracles_fluid.rs (rho=1.204/beta=1/300/dT=100/g=9.81)
    let (rho0, beta, dt, g) = (0.9_f64, 1.0 / 350.0, 250.0_f64, 9.81_f64);
    let f = boussinesq_buoyancy_n_per_m3(
        Fix128::from_ratio(9, 10),
        Fix128::from_ratio(1, 350),
        Fix128::from_int(250),
        Fix128::from_ratio(981, 100),
    )
    .to_f64();
    let want_f = rho0 * beta * dt * g;
    println!("[smoke_fire] f_b(hot plume) = {f} N/m3 (closed form {want_f})");
    assert!(rel_err(f, want_f) < 1e-9, "f_b = {f} vs {want_f}");

    let cold = boussinesq_buoyancy_n_per_m3(
        Fix128::from_ratio(9, 10),
        Fix128::from_ratio(1, 350),
        Fix128::from_int(-150),
        Fix128::from_ratio(981, 100),
    );
    println!(
        "[smoke_fire] f_b(cold, dT=-150) = {} N/m3 (sign check)",
        cold.to_f64()
    );
    assert!(cold < Fix128::ZERO, "cold plume must sink: f_b < 0");

    // --- degenerate inputs: documented guards return exactly zero --------
    assert_eq!(
        reaction_rate_kg_per_m3_s(&methane, Fix128::ZERO, Fix128::ONE, Fix128::ONE),
        Fix128::ZERO,
        "T=0 -> r=0 exactly (guard)"
    );
    assert_eq!(
        reaction_rate_kg_per_m3_s(&methane, Fix128::from_int(-100), Fix128::ONE, Fix128::ONE),
        Fix128::ZERO,
        "T<0 -> r=0 exactly (guard)"
    );
    assert_eq!(
        reaction_rate_kg_per_m3_s(&methane, Fix128::from_int(2000), Fix128::ZERO, Fix128::ONE),
        Fix128::ZERO,
        "zero fuel density -> r=0 exactly (product, not a guard)"
    );
    let mut zero_r_gas = methane;
    zero_r_gas.gas_constant_j_per_mol_k = Fix128::ZERO;
    assert_eq!(
        reaction_rate_kg_per_m3_s(
            &zero_r_gas,
            Fix128::from_int(2000),
            Fix128::ONE,
            Fix128::ONE
        ),
        Fix128::ZERO,
        "R=0 -> r=0 exactly (guard)"
    );
    println!("[smoke_fire] degenerate guards: T<=0, R=0, zero density all return exactly 0");

    assert_eq!(
        heat_release_j_per_m3_s(&methane, Fix128::ZERO),
        Fix128::ZERO
    );
    assert_eq!(
        soot_generation_kg_per_m3_s(&methane, Fix128::ZERO),
        Fix128::ZERO
    );
    assert_eq!(
        boussinesq_buoyancy_n_per_m3(Fix128::ONE, Fix128::ONE, Fix128::ZERO, Fix128::ONE),
        Fix128::ZERO,
        "dT=0 -> f_b=0 exactly"
    );
    println!(
        "[smoke_fire] zero reaction rate / zero dT -> exactly zero downstream (plain product, \
         no guard needed)"
    );

    // --- out-of-physical-range input: density_fuel/density_oxidizer are not
    // range-checked; the function is a pure product/exponential and
    // propagates whatever sign it is given rather than clamping or
    // panicking. Pinned as observed, not invented. ------------------------
    let neg = reaction_rate_kg_per_m3_s(
        &methane,
        Fix128::from_int(2000),
        Fix128::from_int(-1),
        Fix128::ONE,
    );
    let pos = reaction_rate_kg_per_m3_s(&methane, Fix128::from_int(2000), Fix128::ONE, Fix128::ONE);
    assert_eq!(
        neg,
        Fix128::ZERO - pos,
        "negative density flips sign exactly, unguarded"
    );
    println!("[smoke_fire] negative density_fuel is not range-checked: r flips sign exactly");

    // --- extreme magnitude: Fix128 is a wrapping fixed-point type (mod
    // 2^128 group) -- confirmed to not panic on overflow, not assumed. ----
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);
    let r = catch_unwind(AssertUnwindSafe(|| {
        reaction_rate_kg_per_m3_s(&methane, Fix128::from_int(2000), huge, huge)
    }));
    println!(
        "[smoke_fire] reaction_rate at extreme (i64::MAX, u64::MAX) density: ok={}",
        r.is_ok()
    );
    assert!(
        r.is_ok(),
        "Fix128 multiplication wraps silently, it does not panic"
    );

    println!(
        "[smoke_fire] done: 6 production entry points exercised (methane_air, pla_air, \
         reaction_rate_kg_per_m3_s, heat_release_j_per_m3_s, soot_generation_kg_per_m3_s, \
         boussinesq_buoyancy_n_per_m3)"
    );
}
