//! Oracles for the wiring of `smoke_fire`
//! (`examples/smoke_fire_combustion.rs`):
//! `ArrheniusReaction::{methane_air, pla_air}`, `reaction_rate_kg_per_m3_s`,
//! `heat_release_j_per_m3_s`, `soot_generation_kg_per_m3_s`,
//! `boussinesq_buoyancy_n_per_m3`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! From the module doc (`src/smoke_fire.rs:1-20`; Turns, *An Introduction to
//! Combustion* ch. 5; Kuo, *Principles of Combustion*):
//!
//! ```text
//! reaction rate   : r      = A * exp(-E_a / (R * T)) * rho_fuel * rho_oxidizer
//! heat release    : q_dot  = r * dH_c
//! soot generation : s_dot  = r * Y_s
//! Boussinesq      : f_b    = rho_0 * beta * (T - T_0) * g
//! ```
//!
//! `tests/engineering_oracles_fluid.rs::smoke_fire_arrhenius_and_boussinesq_closed_forms`
//! already covers: the methane preset's Arrhenius rate at `T` in
//! `{1500, 2000}` K and their ratio, bilinear scaling in reactant densities
//! for methane at `T=2000` K, `heat_release`/`soot_generation` as exact
//! products at a literal rate of `10`, and Boussinesq at
//! `rho=1.204, beta=1/300, dT=100, g=9.81` plus a cold (`dT=-100`) sign
//! check. This file does **not** repeat those; it adds: literal preset
//! field values (asserted exactly, not just compared against each other —
//! the module's own `#[cfg(test)]` only compares `pla` against `methane`),
//! the `pla_air` preset's Arrhenius rate and ratio (not covered anywhere
//! else), bilinear density scaling for `pla_air` (the existing oracle only
//! checks this for methane; the example checks it for `pla_air` at
//! `T=1300` K, this file uses `T=900` K instead), a second
//! `heat_release`/`soot_generation` literal-rate case (non-integer, both
//! presets), a second Boussinesq scenario plus a negative-`beta` sign
//! check, and degenerate/out-of-range input coverage (zero/negative
//! temperature, zero gas constant, zero/negative density, negative gas
//! constant driving the Arrhenius exponent past the `exp_fix` saturation
//! threshold, and extreme-magnitude density with `catch_unwind`) that
//! neither existing file touches.
//!
//! # Degenerate / out-of-range input (observed behavior pinned, not invented)
//!
//! `reaction_rate_kg_per_m3_s` returns exactly `Fix128::ZERO` for
//! `temperature_k <= 0` and for `gas_constant_j_per_mol_k == 0` (explicit
//! guards in `src/smoke_fire.rs`); `density_fuel == 0` or
//! `density_oxidizer == 0` also give exactly zero, but as a plain product,
//! not a guard. Neither `density_fuel`/`density_oxidizer` nor
//! `gas_constant_j_per_mol_k`'s *sign* is range-checked: a negative density
//! flips the sign of the product exactly, and a negative gas constant (with
//! positive `T`) flips the sign of the Arrhenius exponent, which can drive
//! it past `exp_fix`'s `x >= 20` saturation threshold
//! (`src/math_util.rs::EXP_OVERFLOW_SENTINEL`, exactly `2_147_483_647`) —
//! pinned below as an exact integer closed form, not a tolerance check.
//! `heat_release_j_per_m3_s`/`soot_generation_kg_per_m3_s`/
//! `boussinesq_buoyancy_n_per_m3` are plain products with no guards at all;
//! zero/negative inputs propagate through the product exactly.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
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

/// Arrhenius reaction rate closed form, computed independently in `f64`.
fn arrhenius_rate_f64(a: f64, ea: f64, r_gas: f64, t: f64, rho_f: f64, rho_o: f64) -> f64 {
    a * (-ea / (r_gas * t)).exp() * rho_f * rho_o
}

// ---------------------------------------------------------------------------
// Preset literal values
// ---------------------------------------------------------------------------

#[test]
fn methane_and_pla_presets_match_documented_constants_exactly() {
    let m = ArrheniusReaction::methane_air();
    assert_eq!(m.pre_exponential_a, Fix128::from_int(1_300_000_000));
    assert_eq!(m.activation_energy_j_per_mol, Fix128::from_int(202_000));
    assert_eq!(m.gas_constant_j_per_mol_k, Fix128::from_ratio(8314, 1000));
    assert_eq!(m.heat_of_combustion_j_per_kg, Fix128::from_int(50_000_000));
    assert_eq!(m.soot_yield, Fix128::from_ratio(15, 1000));

    let p = ArrheniusReaction::pla_air();
    assert_eq!(p.pre_exponential_a, Fix128::from_int(500_000_000));
    assert_eq!(p.activation_energy_j_per_mol, Fix128::from_int(150_000));
    assert_eq!(p.gas_constant_j_per_mol_k, Fix128::from_ratio(8314, 1000));
    assert_eq!(p.heat_of_combustion_j_per_kg, Fix128::from_int(18_000_000));
    assert_eq!(p.soot_yield, Fix128::from_ratio(20, 1000));
}

// ---------------------------------------------------------------------------
// Arrhenius reaction rate — pla_air (methane covered in
// tests/engineering_oracles_fluid.rs)
// ---------------------------------------------------------------------------

#[test]
fn reaction_rate_matches_arrhenius_closed_form_for_pla_air() {
    let p = ArrheniusReaction::pla_air();
    let r_gas = 8.314_f64;
    for &t in &[700.0_f64, 1100.0_f64] {
        let rate =
            reaction_rate_kg_per_m3_s(&p, Fix128::from_int(t as i64), Fix128::ONE, Fix128::ONE)
                .to_f64();
        let want = arrhenius_rate_f64(5.0e8, 150_000.0, r_gas, t, 1.0, 1.0);
        assert!(rel_err(rate, want) < 1e-6, "pla r({t}) = {rate} vs {want}");
    }
}

#[test]
fn reaction_rate_ratio_between_temperatures_matches_closed_form_for_pla_air() {
    let p = ArrheniusReaction::pla_air();
    let (t1, t2) = (700.0_f64, 1100.0_f64);
    let r_gas = 8.314_f64;
    let rate = |t: f64| {
        reaction_rate_kg_per_m3_s(&p, Fix128::from_int(t as i64), Fix128::ONE, Fix128::ONE).to_f64()
    };
    let ratio = rate(t2) / rate(t1);
    let want_ratio = (150_000.0 / r_gas * (1.0 / t1 - 1.0 / t2)).exp();
    assert!(
        rel_err(ratio, want_ratio) < 1e-6,
        "ratio = {ratio} vs {want_ratio}"
    );
}

#[test]
fn reaction_rate_is_bilinear_in_reactant_densities_for_pla_air_at_900k() {
    let p = ArrheniusReaction::pla_air();
    let base =
        reaction_rate_kg_per_m3_s(&p, Fix128::from_int(900), Fix128::ONE, Fix128::ONE).to_f64();
    let scaled = reaction_rate_kg_per_m3_s(
        &p,
        Fix128::from_int(900),
        Fix128::from_ratio(1, 3),
        Fix128::from_int(9),
    )
    .to_f64();
    assert!(
        rel_err(scaled, 3.0 * base) < 1e-6,
        "(1/3 fuel, 9x oxidizer) must be 3x base: {scaled} vs {}",
        3.0 * base
    );
}

// ---------------------------------------------------------------------------
// heat_release_j_per_m3_s / soot_generation_kg_per_m3_s — second (non-integer)
// literal rate, both presets
// ---------------------------------------------------------------------------

#[test]
fn heat_release_and_soot_generation_are_exact_products_for_a_fractional_rate() {
    // rr = 3.7 kg/(m3 s), a literal input never produced by calling
    // reaction_rate_kg_per_m3_s.
    let rr = Fix128::from_ratio(37, 10);
    let m = ArrheniusReaction::methane_air();
    let p = ArrheniusReaction::pla_air();

    let q_m = heat_release_j_per_m3_s(&m, rr).to_f64();
    assert!(
        rel_err(q_m, 3.7 * 50_000_000.0) < 1e-9,
        "methane q_dot(3.7) = {q_m} vs {}",
        3.7 * 50_000_000.0
    );
    let s_m = soot_generation_kg_per_m3_s(&m, rr).to_f64();
    assert!(
        rel_err(s_m, 3.7 * 0.015) < 1e-9,
        "methane s_dot(3.7) = {s_m} vs {}",
        3.7 * 0.015
    );

    let q_p = heat_release_j_per_m3_s(&p, rr).to_f64();
    assert!(
        rel_err(q_p, 3.7 * 18_000_000.0) < 1e-9,
        "pla q_dot(3.7) = {q_p} vs {}",
        3.7 * 18_000_000.0
    );
    let s_p = soot_generation_kg_per_m3_s(&p, rr).to_f64();
    assert!(
        rel_err(s_p, 3.7 * 0.020) < 1e-9,
        "pla s_dot(3.7) = {s_p} vs {}",
        3.7 * 0.020
    );
}

#[test]
fn heat_release_and_soot_generation_zero_and_negative_rate_propagate_exactly() {
    let m = ArrheniusReaction::methane_air();
    assert_eq!(
        heat_release_j_per_m3_s(&m, Fix128::ZERO),
        Fix128::ZERO,
        "zero rate -> zero heat release, exactly"
    );
    assert_eq!(
        soot_generation_kg_per_m3_s(&m, Fix128::ZERO),
        Fix128::ZERO,
        "zero rate -> zero soot, exactly"
    );

    // Negative rate is not physical (rate is a magnitude), but neither
    // function range-checks its input: the product's sign follows exactly.
    let pos_q = heat_release_j_per_m3_s(&m, Fix128::from_int(10));
    let neg_q = heat_release_j_per_m3_s(&m, Fix128::from_int(-10));
    assert_eq!(neg_q, Fix128::ZERO - pos_q, "q_dot(-r) = -q_dot(r) exactly");

    let pos_s = soot_generation_kg_per_m3_s(&m, Fix128::from_int(10));
    let neg_s = soot_generation_kg_per_m3_s(&m, Fix128::from_int(-10));
    assert_eq!(neg_s, Fix128::ZERO - pos_s, "s_dot(-r) = -s_dot(r) exactly");
}

// ---------------------------------------------------------------------------
// Boussinesq buoyancy — second scenario + negative-beta sign check
// ---------------------------------------------------------------------------

#[test]
fn boussinesq_matches_closed_form_for_a_second_scenario() {
    // rho_0=2.5, beta=1/450, dT=75, g=9.8 — distinct from both
    // tests/engineering_oracles_fluid.rs (1.204/1/300/100/9.81) and
    // examples/smoke_fire_combustion.rs (0.9/1/350/250/9.81).
    let f = boussinesq_buoyancy_n_per_m3(
        Fix128::from_ratio(5, 2),
        Fix128::from_ratio(1, 450),
        Fix128::from_int(75),
        Fix128::from_ratio(98, 10),
    )
    .to_f64();
    let want = 2.5 * (1.0 / 450.0) * 75.0 * 9.8;
    assert!(rel_err(f, want) < 1e-9, "f_b = {f} vs {want}");
}

#[test]
fn boussinesq_zero_delta_t_gives_exact_zero_and_negative_beta_flips_sign() {
    assert_eq!(
        boussinesq_buoyancy_n_per_m3(Fix128::ONE, Fix128::ONE, Fix128::ZERO, Fix128::ONE),
        Fix128::ZERO,
        "dT=0 -> f_b=0 exactly"
    );

    // beta_per_k < 0 is not physical for a thermal expansion coefficient,
    // but boussinesq_buoyancy_n_per_m3 does not range-check it: the sign
    // of the product follows exactly.
    let pos = boussinesq_buoyancy_n_per_m3(
        Fix128::ONE,
        Fix128::from_ratio(1, 300),
        Fix128::from_int(100),
        Fix128::from_int(10),
    );
    let neg = boussinesq_buoyancy_n_per_m3(
        Fix128::ONE,
        Fix128::from_ratio(-1, 300),
        Fix128::from_int(100),
        Fix128::from_int(10),
    );
    assert_eq!(
        neg,
        Fix128::ZERO - pos,
        "f_b(-beta) = -f_b(beta) exactly, unguarded"
    );
}

// ---------------------------------------------------------------------------
// Degenerate / out-of-range input
// ---------------------------------------------------------------------------

#[test]
fn reaction_rate_zero_and_negative_temperature_return_exact_zero_for_both_presets() {
    for r in [
        ArrheniusReaction::methane_air(),
        ArrheniusReaction::pla_air(),
    ] {
        assert_eq!(
            reaction_rate_kg_per_m3_s(&r, Fix128::ZERO, Fix128::ONE, Fix128::ONE),
            Fix128::ZERO,
            "T=0 guard"
        );
        assert_eq!(
            reaction_rate_kg_per_m3_s(&r, Fix128::from_int(-1), Fix128::ONE, Fix128::ONE),
            Fix128::ZERO,
            "T<0 guard"
        );
    }
}

#[test]
fn reaction_rate_zero_gas_constant_and_zero_density_return_exact_zero() {
    let mut r = ArrheniusReaction::methane_air();
    r.gas_constant_j_per_mol_k = Fix128::ZERO;
    assert_eq!(
        reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ONE, Fix128::ONE),
        Fix128::ZERO,
        "R=0 guard"
    );

    let r = ArrheniusReaction::methane_air();
    assert_eq!(
        reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ZERO, Fix128::ONE),
        Fix128::ZERO,
        "zero fuel density -> zero rate (product, not a guard)"
    );
    assert_eq!(
        reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
        "zero oxidizer density -> zero rate (product, not a guard)"
    );
}

#[test]
fn reaction_rate_negative_density_is_unguarded_and_flips_sign_exactly() {
    let r = ArrheniusReaction::methane_air();
    let pos = reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ONE, Fix128::ONE);
    let neg_fuel = reaction_rate_kg_per_m3_s(
        &r,
        Fix128::from_int(2000),
        Fix128::from_int(-1),
        Fix128::ONE,
    );
    assert_eq!(
        neg_fuel,
        Fix128::ZERO - pos,
        "negative fuel density flips sign exactly"
    );
    let neg_both = reaction_rate_kg_per_m3_s(
        &r,
        Fix128::from_int(2000),
        Fix128::from_int(-1),
        Fix128::from_int(-1),
    );
    assert_eq!(
        neg_both, pos,
        "two negative densities cancel back to the positive rate exactly"
    );
}

/// `gas_constant_j_per_mol_k`'s *sign* is not range-checked. With
/// `R = -20`, `T = 500`, `E_a = 202_000` (methane's), `denom = R*T = -10000`
/// and `exponent = -(E_a/denom) = -(202000/-10000) = 20.2`, which crosses
/// `exp_fix`'s `x >= 20` saturation threshold
/// (`src/math_util.rs::EXP_OVERFLOW_SENTINEL = Fix128::from_raw(i64::MAX >>
/// 32, 0)`, exactly `2_147_483_647`). Since every factor involved has a
/// zero fractional part, the whole chain is exact integer arithmetic:
/// `r = A * 2_147_483_647 = 1_300_000_000 * 2_147_483_647 =
/// 2_791_728_741_100_000_000` — pinned bit-exact, not by a tolerance.
#[test]
fn negative_gas_constant_drives_the_arrhenius_exponent_past_exp_fix_saturation() {
    let mut r = ArrheniusReaction::methane_air();
    r.gas_constant_j_per_mol_k = Fix128::from_int(-20);
    let rate = reaction_rate_kg_per_m3_s(&r, Fix128::from_int(500), Fix128::ONE, Fix128::ONE);
    assert_eq!(
        rate,
        Fix128::from_int(2_791_728_741_100_000_000),
        "negative R saturates the exponential to EXP_OVERFLOW_SENTINEL, exactly"
    );
}

#[test]
fn reaction_rate_extreme_magnitude_density_wraps_silently_without_panicking() {
    let r = ArrheniusReaction::methane_air();
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);
    let result = catch_unwind(AssertUnwindSafe(|| {
        reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), huge, huge)
    }));
    assert!(
        result.is_ok(),
        "Fix128 multiplication is a wrapping mod-2^128 group operation, it must not panic"
    );
}

#[test]
fn boussinesq_extreme_magnitude_inputs_do_not_panic() {
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);
    let result = catch_unwind(AssertUnwindSafe(|| {
        boussinesq_buoyancy_n_per_m3(huge, huge, huge, huge)
    }));
    assert!(result.is_ok(), "Fix128 wraps silently, it must not panic");
}
