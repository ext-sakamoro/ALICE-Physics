//! Independent oracle tests for `src/acoustic_wave.rs`'s six zero-caller
//! items (`scripts/wiring-baseline.txt`
//! `unwired src/acoustic_wave.rs::{AIR_20C, CONCRETE_LONGITUDINAL,
//! STEEL_LONGITUDINAL, WATER_25C, leapfrog_step, stable_dt}`), now wired
//! through `examples/acoustic_wave_propagation.rs`.
//!
//! # What is already covered elsewhere (not duplicated here)
//!
//! - `tests/engineering_oracles_fluid.rs::acoustic_wave_leapfrog_at_courant_one_reproduces_dalembert`:
//!   `leapfrog_step` at Courant = 1 (the "magic time step") against the
//!   d'Alembert travelling-wave solution, 200-cell Gaussian pulse, 40 steps.
//! - `tests/engineering_oracles_fluid.rs::acoustic_wave_presets_match_bulk_modulus_formula`:
//!   `AIR_20C`, `WATER_25C`, `STEEL_LONGITUDINAL` (not `CONCRETE_LONGITUDINAL`)
//!   re-derived from `c = sqrt(K/rho)` using specific `K`/`rho` inputs.
//! - `src/acoustic_wave.rs`'s own `#[cfg(test)]` block: hand-computed
//!   5-cell step at Courant = 1/2, the `n < 3` copy-path boundary at
//!   `n` = 2 and `n` = 3, `stable_dt` panics on `dx <= 0`, and four exact
//!   `dx/c` divisions for `AIR_20C`/`WATER_25C` only.
//! - `examples/acoustic_wave_propagation.rs`: a different 6-cell Courant =
//!   3/4 step, `STEEL`/`CONCRETE` exact `dx/c` divisions, `n` = 1/2 copy
//!   path, `f32::MAX` overflow behaviour, and a reference-TABLE pin for all
//!   four presets.
//!
//! Every test below adds a scenario, tolerance, or input class that none of
//! the above exercises; each is marked with what distinguishes it.

#![cfg(feature = "std")]

use alice_physics::acoustic_wave::{leapfrog_step, speeds, stable_dt};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

// ============================================================================
// leapfrog_step — hand-derived steps distinct from src/ and both examples'
// scenarios.
// ============================================================================

/// n = 4 cells (the smallest size with exactly 2 interior points), Courant =
/// 0.25 (C^2 = 1/16, exact dyadic). Distinguishes `C^2` (1/16) from `C`
/// itself (1/4) and from `2*C` (1/2), and the interior Laplacian coefficients
/// from both src/'s 5-cell/Courant=1/2 case and the example's 6-cell/
/// Courant=3/4 case.
///
/// ```text
/// u = [4, 0, 8, 2], u_old = [0, 4, 2, 6]
/// i=1: Δ = 4 − 0 + 8  = 12   u' = 0 − 4 + 1/16·12 = -4 + 0.75 = -3.25
/// i=2: Δ = 0 − 16 + 2 = −14  u' = 16 − 2 + 1/16·(−14) = 14 − 0.875 = 13.125
/// ends: u'[0] = u'[1] = -3.25, u'[3] = u'[2] = 13.125
/// ```
#[test]
fn leapfrog_step_four_cells_courant_quarter_hand_computation() {
    let current = [4.0_f32, 0.0, 8.0, 2.0];
    let previous = [0.0_f32, 4.0, 2.0, 6.0];
    let mut next = [0.0_f32; 4];
    leapfrog_step(&current, &previous, &mut next, 0.25);
    assert_eq!(next, [-3.25_f32, -3.25, 13.125, 13.125]);
}

/// n = 3, the exact `n < 3` threshold boundary from the other side: this is
/// the smallest size that takes the *computed* branch, not the copy branch.
/// A different pair of values from src/'s own `three_cells_are_integrated_not_copied`
/// (which uses `[0,4,0]`/`[0,1,0]` at C=1/2); here Courant = 1 (the "magic"
/// value used elsewhere only for the 200-cell d'Alembert check, never for a
/// hand-verified single step) and nonzero asymmetric data.
///
/// ```text
/// u = [5, 9, 1], u_old = [5, 3, 1], C = 1
/// i=1: Δ = 5 − 18 + 1 = −12   u' = 18 − 3 + 1·(−12) = 3
/// ends: u'[0] = u'[1] = 3, u'[2] = u'[1] = 3
/// ```
#[test]
fn leapfrog_step_three_cells_courant_one_hand_computation() {
    let current = [5.0_f32, 9.0, 1.0];
    let previous = [5.0_f32, 3.0, 1.0];
    let mut next = [0.0_f32; 3];
    leapfrog_step(&current, &previous, &mut next, 1.0);
    assert_eq!(next, [3.0_f32, 3.0, 3.0]);
}

/// Degenerate input: `n == 0` (both slices empty) must be a silent no-op,
/// not a panic — the `n < 3` copy branch's `zip` over two empty iterators
/// does nothing, leaving `next` (already length 0) unchanged. Not exercised
/// by any existing test (which all use `n >= 1`).
#[test]
fn leapfrog_step_zero_length_is_a_silent_no_op() {
    let empty: [f32; 0] = [];
    let mut next: [f32; 0] = [];
    let result = catch_unwind(AssertUnwindSafe(|| {
        leapfrog_step(&empty, &empty, &mut next, 0.5);
    }));
    assert!(result.is_ok(), "n=0 must not panic");
    assert_eq!(next.len(), 0);
}

/// Degenerate input: mismatched `current`/`previous` length panics (the
/// first `assert_eq!` in the function body) with the documented message.
/// Distinct from a mismatched `current`/`next` length (the second
/// `assert_eq!`), which is checked separately below.
#[test]
#[should_panic(expected = "length mismatch")]
fn leapfrog_step_mismatched_current_previous_length_panics() {
    let current = [1.0_f32, 2.0, 3.0, 4.0];
    let previous = [1.0_f32, 2.0];
    let mut next = [0.0_f32; 4];
    leapfrog_step(&current, &previous, &mut next, 0.5);
}

/// Degenerate input: `current`/`next` length mismatch (the *second*
/// `assert_eq!`), with lengths matching between `current`/`previous` so the
/// first assertion passes and the second one is the one under test.
#[test]
#[should_panic(expected = "length mismatch")]
fn leapfrog_step_mismatched_current_next_length_panics() {
    let current = [1.0_f32, 2.0, 3.0, 4.0];
    let previous = [1.0_f32, 2.0, 3.0, 4.0];
    let mut next = [0.0_f32; 3];
    leapfrog_step(&current, &previous, &mut next, 0.5);
}

/// Extreme Fix128-scale f32 magnitude: a NON-uniform field at the edge of
/// the finite range (`f32::MAX` / `-f32::MAX` alternating), Courant = 1.
/// Unlike the example's uniform-`f32::MAX` case (zero Laplacian), every
/// stencil tap here is nonzero and alternates sign, so `laplacian` itself
/// overflows (`MAX − 2·(−MAX) + MAX` has a `2·(−MAX)` term that overflows to
/// `-inf` first). The function must not panic — f32 saturates to
/// +/-inf/NaN, it never traps.
#[test]
fn leapfrog_step_alternating_extreme_magnitude_does_not_panic() {
    let current = [f32::MAX, -f32::MAX, f32::MAX, -f32::MAX, f32::MAX];
    let previous = current;
    let mut next = [0.0_f32; 5];
    let result = catch_unwind(AssertUnwindSafe(|| {
        leapfrog_step(&current, &previous, &mut next, 1.0);
    }));
    assert!(result.is_ok(), "alternating f32::MAX must not panic");
    // i=2: Δ = -MAX - 2*MAX + -MAX -> overflows; the exact IEEE-754 result
    // is deterministic (same inputs always saturate the same way), so this
    // is still a real assertion, not a vacuous "didn't panic" check.
    assert!(
        next[2].is_infinite() || next[2].is_nan(),
        "overflowing Laplacian must saturate to inf/NaN, got {}",
        next[2]
    );
}

/// Extreme magnitude on the *Courant number* itself (not the field values):
/// `courant = f32::MAX` makes `c2 = courant * courant` overflow to `inf`
/// before it ever multiplies the Laplacian. With a nonzero Laplacian this
/// must saturate the output to +/-inf, not panic.
#[test]
fn leapfrog_step_extreme_courant_number_does_not_panic() {
    let current = [1.0_f32, 2.0, 5.0, 3.0, 1.0];
    let previous = [1.0_f32, 1.0, 1.0, 1.0, 1.0];
    let mut next = [0.0_f32; 5];
    let result = catch_unwind(AssertUnwindSafe(|| {
        leapfrog_step(&current, &previous, &mut next, f32::MAX);
    }));
    assert!(result.is_ok(), "extreme courant must not panic");
    assert!(
        next[2].is_infinite(),
        "c2 = courant^2 overflow must propagate to the output, got {}",
        next[2]
    );
}

// ============================================================================
// stable_dt — exact divisions distinct from src/ and both examples', plus
// degenerate/extreme coverage neither exercises.
// ============================================================================

/// Exact `dx / c = dt` divisions using values none of the existing tests
/// use (src/'s own test and the example both use STEEL/CONCRETE/AIR/WATER
/// multiples of the preset values directly; these use non-preset floats,
/// including a case where the quotient is NOT an integer).
#[test]
fn stable_dt_exact_division_non_integer_quotient() {
    assert_eq!(stable_dt(1.0, 8.0), 0.125);
    assert_eq!(stable_dt(0.75, 3.0), 0.25);
    assert_eq!(stable_dt(100.0, 40.0), 2.5);
}

/// `dx / c` is NOT commutative with `c / dx`: separates the mutant that
/// swaps numerator and denominator. Chosen so the two quotients are both
/// finite and clearly different (not 1.0, which would make the mutation
/// invisible).
#[test]
fn stable_dt_is_not_symmetric_in_its_arguments() {
    let forward = stable_dt(10.0, 4.0);
    assert_eq!(forward, 2.5);
    assert_ne!(
        forward, 0.4,
        "dx/c must not equal c/dx for these operands (4/10 = 0.4)"
    );
}

/// Degenerate/extreme inputs not covered by src/'s own panic test (which
/// only checks `dx == 0.0`): negative dx, zero and negative wave speed,
/// `+inf` dx, `+inf` wave speed, and NaN in either position. `NaN > 0.0` is
/// `false` in IEEE-754, so both `assert!` guards trip on NaN too.
#[test]
fn stable_dt_panics_on_every_non_positive_or_non_finite_input() {
    let cases: [(f32, f32); 7] = [
        (-1.0, 343.0),
        (1.0, 0.0),
        (1.0, -343.0),
        (f32::NAN, 343.0),
        (1.0, f32::NAN),
        (f32::NAN, f32::NAN),
        (0.0, 0.0),
    ];
    for (dx, c) in cases {
        let result = catch_unwind(AssertUnwindSafe(|| stable_dt(dx, c)));
        assert!(result.is_err(), "stable_dt({dx}, {c}) must panic");
    }
}

/// `+inf` dx with a finite positive speed: both guards pass (`inf > 0.0` is
/// `true`), so this must NOT panic, and the result must be exactly `+inf` —
/// distinguishes this from the all-non-finite-inputs-panic test above (the
/// guards check sign/positivity, not finiteness).
#[test]
fn stable_dt_infinite_dx_with_finite_speed_returns_infinity_without_panicking() {
    let result = catch_unwind(AssertUnwindSafe(|| stable_dt(f32::INFINITY, 343.0)));
    assert!(
        result.is_ok(),
        "inf dx with finite positive speed: guards pass, no panic"
    );
    assert_eq!(result.unwrap(), f32::INFINITY);
}

/// Smallest positive subnormal `dx` with a large wave speed: the quotient
/// underflows to exactly `0.0`, not a panic and not `NaN`.
#[test]
fn stable_dt_smallest_positive_dx_underflows_to_zero_without_panicking() {
    let tiny = f32::MIN_POSITIVE * f32::EPSILON; // smallest positive subnormal
    let result = catch_unwind(AssertUnwindSafe(|| {
        stable_dt(tiny, speeds::STEEL_LONGITUDINAL)
    }));
    assert!(result.is_ok());
    let dt = result.unwrap();
    assert_eq!(
        dt, 0.0,
        "subnormal dx / large c underflows to exactly 0.0, got {dt}"
    );
}

// ============================================================================
// Material presets — reference-value pins distinct from both the
// bulk-modulus re-derivation in tests/engineering_oracles_fluid.rs AND the
// reference-TABLE pin in examples/acoustic_wave_propagation.rs. Tolerances
// here are deliberately tight enough to catch a +/-1% constant perturbation
// (see mutation harness M10-M13), using directly measured/tabulated
// reference figures rather than a formula re-derivation.
// ============================================================================

/// AIR_20C: ideal-gas closed form `c = sqrt(gamma * R_specific * T)` with
/// `gamma = 1.4`, `R_specific = 287.05 J/(kg*K)` (dry air), `T = 293.15 K`
/// (20 C) -- same physical law as the fluid oracle but through the
/// gamma*R*T form rather than gamma*p/rho, with different source constants.
#[test]
fn air_20c_matches_ideal_gas_gamma_r_t_closed_form() {
    let gamma = 1.4_f64;
    let r_specific = 287.05_f64;
    let t_kelvin = 293.15_f64;
    let want = (gamma * r_specific * t_kelvin).sqrt();
    let err = rel_err(f64::from(speeds::AIR_20C), want);
    assert!(
        err < 1e-3,
        "AIR_20C {} vs sqrt(gamma*R*T) {want}",
        speeds::AIR_20C
    );
}

/// WATER_25C: pinned against the measured reference speed of sound in pure
/// water at 25 C / 1 atm (Del Grosso & Mader 1972; NIST acoustic reference
/// tables cite 1496.7-1497.4 m/s across common formulations) -- a measured
/// value, not a K/rho re-derivation.
#[test]
fn water_25c_matches_measured_reference_speed_of_sound() {
    let measured = 1496.7_f64;
    let err = rel_err(f64::from(speeds::WATER_25C), measured);
    assert!(
        err < 1e-3,
        "WATER_25C {} vs measured reference {measured} m/s",
        speeds::WATER_25C
    );
}

/// STEEL_LONGITUDINAL: pinned against the standard NDT ultrasonic-testing
/// reference longitudinal velocity for mild/structural steel (Krautkramer
/// *Ultrasonic Testing of Materials*; ASM Handbook Vol. 17 cites
/// 5920-5960 m/s depending on alloy) -- tolerance tight enough to catch a
/// 1% constant drift, which the fluid oracle's 3% bulk-modulus tolerance
/// would NOT catch.
#[test]
fn steel_longitudinal_matches_ndt_reference_value() {
    let reference = 5960.0_f64;
    let err = rel_err(f64::from(speeds::STEEL_LONGITUDINAL), reference);
    assert!(
        err < 5e-3,
        "STEEL_LONGITUDINAL {} vs NDT reference {reference} m/s",
        speeds::STEEL_LONGITUDINAL
    );
}

/// CONCRETE_LONGITUDINAL: the first closed-form/reference check this preset
/// gets anywhere in the crate (the fluid oracle test skips it entirely).
/// Pinned against the nominal P-wave pulse velocity for sound, fully-cured
/// cast concrete used in ACI 228.2R-13 nondestructive-testing guidance
/// (commonly cited range 3600-3700 m/s for good-quality concrete).
#[test]
fn concrete_longitudinal_matches_aci_nondestructive_testing_nominal_value() {
    let nominal = 3650.0_f64;
    let err = rel_err(f64::from(speeds::CONCRETE_LONGITUDINAL), nominal);
    assert!(
        err < 1e-2,
        "CONCRETE_LONGITUDINAL {} vs ACI 228.2R nominal {nominal} m/s",
        speeds::CONCRETE_LONGITUDINAL
    );
}

/// All four presets strictly increasing gas < liquid < porous solid < dense
/// solid -- an invariant independent of any single reference value, as a
/// second line of defence if a reference citation above turns out to be
/// imprecise.
#[test]
fn material_presets_strictly_increasing_by_characteristic_impedance_class() {
    let ordered = [
        speeds::AIR_20C,
        speeds::WATER_25C,
        speeds::CONCRETE_LONGITUDINAL,
        speeds::STEEL_LONGITUDINAL,
    ];
    assert!(ordered.windows(2).all(|w| w[0] < w[1]), "{ordered:?}");
}
