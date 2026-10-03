//! Oracles for the production entry points of `alice_physics::rolling_contact`
//! driven by `examples/rolling_contact_fatigue.rs`: `HertzianContact`,
//! `basquin_cycles_to_failure`, `materials::{bearing_steel_52100,
//! gear_steel_8620, silicon_nitride}`, `hertzian_sphere_sphere`, and
//! `rolling_contact_life_cycles`.
//!
//! # What this file is and is not
//!
//! `src/rolling_contact.rs`'s own `#[cfg(test)]` block already has closed-form
//! oracles for `hertzian_sphere_sphere` (identical-sphere textbook case,
//! load/stiffness monotonicity, the `a³`/`p_max`/shear-depth relations, body
//! swap symmetry) and for `basquin_cycles_to_failure` (infinite life below
//! the endurance limit, finite life above it, monotonic stress/life
//! trade-off) and for the three presets (range sanity, pairwise distinctness,
//! feeding into Hertz). Those are not repeated here. What that file does
//! **not** cover, and this file does:
//!
//! * the ball-on-flat-rail contact (`radius_2_m = f32::INFINITY`), the
//!   degenerate-radius usage documented on `hertzian_sphere_sphere` — no
//!   existing test ever passes an infinite radius,
//! * `rolling_contact_life_cycles` itself: nothing called it before
//!   `examples/rolling_contact_fatigue.rs` existed, so nothing checked that
//!   it actually composes `hertzian_sphere_sphere` + `basquin_cycles_to_failure`
//!   in that order with the peak pressure as the stress, rather than e.g. a
//!   swapped argument or a hard-coded pressure,
//! * the exact `f32::INFINITY` sentinel at and below the endurance limit
//!   (existing tests only assert `is_infinite()` / `is_finite()`, not the
//!   precise value or the `<=` boundary itself), and
//! * four degenerate/extreme inputs the existing file does not touch: the
//!   five documented zero/negative panics for `hertzian_sphere_sphere`
//!   exercised together with `catch_unwind` (existing tests only check one
//!   panic each, via `#[should_panic]`, never confirming the function does
//!   not panic on anything else), `rolling_contact_life_cycles` propagating
//!   the inner Hertz panic rather than swallowing it, an input that overflows
//!   the contact radius to `f32::INFINITY` (collapsing peak pressure to
//!   exactly `0.0`), and Basquin inputs that cleanly underflow to `0.0`
//!   cycles / overflow to `f32::INFINITY` cycles.
//!
//! # Degenerate input conventions (asserted, not just "did not panic")
//!
//! * `hertzian_sphere_sphere`: `load_n <= 0.0`, `radius_1_m <= 0.0`,
//!   `radius_2_m <= 0.0`, `elastic_modulus_1_pa <= 0.0`, or
//!   `elastic_modulus_2_pa <= 0.0` panics (module doc `# Panics`).
//!   `radius_2_m = f32::INFINITY` is valid (flat-surface usage, module doc on
//!   the function) and collapses `1/R* = 1/R1 + 1/R2` to `1/R1` exactly,
//!   since `1.0_f32 / f32::INFINITY == 0.0`.
//! * `basquin_cycles_to_failure`: `stress_pa < 0.0`, `basquin_coefficient <=
//!   0.0`, or `basquin_exponent <= 0.0` panics. `stress_pa <= endurance_limit_pa`
//!   (the comparison is `<=`, so equality counts) returns `f32::INFINITY`
//!   exactly, not merely "a large finite number" or "`is_infinite()`".
//! * Overflow in `hertzian_sphere_sphere`'s `a³ = 3·P·R*/(4·E*)` division
//!   rounds to `f32::INFINITY` (a plain IEEE-754 division overflow, not an
//!   `inf/inf` — the numerator and denominator are each finite): then
//!   `contact_radius_m = cbrt(∞) = ∞`, and `peak_pressure_pa = 3·P/(2π·∞²) =
//!   0.0` exactly (finite numerator over infinite denominator), not `NaN`.
//!   Measured with `catch_unwind` below, not derived from documentation —
//!   the module's `# Panics` section says nothing about this case.
//! * Basquin underflow (`stress^(-exponent)` smaller than the smallest
//!   subnormal `f32`) rounds to exactly `0.0`, not a panic and not `NaN`.
//!   Basquin overflow (`coefficient · stress^(-exponent)` larger than
//!   `f32::MAX`) rounds to exactly `f32::INFINITY`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::rolling_contact::{
    basquin_cycles_to_failure, hertzian_sphere_sphere, materials, rolling_contact_life_cycles,
    HertzianContact,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Relative-error assertion against an f64 closed-form oracle (f32 keeps
/// ~1e-7 relative precision; the slack below accounts for `det_math::cbrt`'s
/// own approximation error, same convention as the module's own
/// `presets_feed_hertz_and_match_closed_form_relations` unit test).
fn assert_rel(got: f32, want: f64, tol: f64, what: &str) {
    let g = f64::from(got);
    let err = ((g - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

// ============================================================================
// Section 1: material presets — exact published constants (the module's own
// doc comment on `mod materials`), independent of calling the functions.
// ============================================================================

#[test]
fn bearing_steel_52100_matches_documented_values() {
    let (e, nu, c, m, se) = materials::bearing_steel_52100();
    assert_eq!(e, 210.0e9, "E");
    assert_eq!(nu, 0.30, "nu");
    assert_eq!(c, 5.0e34, "Basquin C");
    assert_eq!(m, 3.0, "Basquin m");
    assert_eq!(se, 1.5e9, "endurance limit");
}

#[test]
fn gear_steel_8620_matches_documented_values() {
    let (e, nu, c, m, se) = materials::gear_steel_8620();
    assert_eq!(e, 205.0e9, "E");
    assert_eq!(nu, 0.29, "nu");
    assert_eq!(c, 2.5e34, "Basquin C");
    assert_eq!(m, 3.2, "Basquin m");
    assert_eq!(se, 1.3e9, "endurance limit");
}

#[test]
fn silicon_nitride_matches_documented_values() {
    let (e, nu, c, m, se) = materials::silicon_nitride();
    assert_eq!(e, 310.0e9, "E");
    assert_eq!(nu, 0.27, "nu");
    assert_eq!(c, 8.0e34, "Basquin C");
    assert_eq!(m, 3.5, "Basquin m");
    assert_eq!(se, 2.0e9, "endurance limit");
}

// ============================================================================
// Section 2: ball-on-flat-rail contact (radius_2_m = INFINITY), identical
// radii vs. a very different (infinite) radius. Closed form derived by hand
// from the module's documented Hertz relations:
//   1/R* = 1/R1 + 1/R2  (= 1/R1 exactly when R2 = INFINITY, since 1/INFINITY
//   == 0.0 in IEEE-754)
//   1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2
//   a^3  = 3 P R* / (4 E*)
//   p_max = 3 P / (2 pi a^2)
//   z_max_shear = 0.48 a
// Values (f64 reference): gear_steel_8620 ball (R1=5mm) on a gear_steel_8620
// rail, P=300N: R*=0.005, E*=111911780762.09193, a^3=1.0052560975609756e-11,
// a=0.00021582027356240974, p_max=3075233972.3397856,
// z=0.00010359373130995667.
// ============================================================================

#[test]
fn flat_rail_contact_collapses_effective_radius_to_radius_1_exactly() {
    let (e, nu, _, _, _) = materials::gear_steel_8620();
    let c: HertzianContact = hertzian_sphere_sphere(300.0, 5.0e-3, f32::INFINITY, e, e, nu, nu);

    assert_rel(
        c.contact_radius_m,
        0.00021582027356240974,
        1.0e-4,
        "flat-rail a",
    );
    assert_rel(
        c.peak_pressure_pa,
        3075233972.3397856,
        1.0e-4,
        "flat-rail p_max",
    );
    assert_rel(
        c.max_shear_depth_m,
        0.00010359373130995667,
        1.0e-4,
        "flat-rail z",
    );

    // Independent check of the a^3 relation using R* = R1 directly (no
    // cbrt needed for the oracle side — cube the implementation's `a`).
    let r_star = 5.0e-3_f64;
    let inv_e_star = (1.0 - f64::from(nu) * f64::from(nu)) / f64::from(e) * 2.0;
    let e_star = 1.0 / inv_e_star;
    let expected_a3 = 3.0 * 300.0 * r_star / (4.0 * e_star);
    let a3 = f64::from(c.contact_radius_m).powi(3);
    let rel = (a3 - expected_a3).abs() / expected_a3;
    assert!(rel < 1.0e-3, "a^3 relation: rel err = {rel}");
}

#[test]
fn flat_rail_contact_differs_from_self_mated_contact_at_the_same_radius_1_and_load() {
    let (e, nu, _, _, _) = materials::gear_steel_8620();
    // Same R1, same load, same material on both sides: only R2 changes
    // (5mm ball vs. flat rail). The outputs must differ — a mutation that
    // ignores radius_2_m entirely (e.g. always treats it as equal to
    // radius_1_m) would make this pass by coincidence only if the two
    // geometries happened to agree, so the two R2 values are chosen to be
    // very different (5mm vs. infinite).
    let self_mated = hertzian_sphere_sphere(300.0, 5.0e-3, 5.0e-3, e, e, nu, nu);
    let flat_rail = hertzian_sphere_sphere(300.0, 5.0e-3, f32::INFINITY, e, e, nu, nu);

    assert!(
        flat_rail.contact_radius_m > self_mated.contact_radius_m,
        "flat rail (R*=R1) must give a larger contact patch than self-mated (R*=R1/2): \
         flat={}, self_mated={}",
        flat_rail.contact_radius_m,
        self_mated.contact_radius_m
    );
    assert!(
        flat_rail.peak_pressure_pa < self_mated.peak_pressure_pa,
        "larger patch at the same load must give lower peak pressure: \
         flat={}, self_mated={}",
        flat_rail.peak_pressure_pa,
        self_mated.peak_pressure_pa
    );
}

#[test]
fn flat_rail_contact_is_symmetric_under_body_swap() {
    let (e, nu, _, _, _) = materials::gear_steel_8620();
    let a = hertzian_sphere_sphere(300.0, 5.0e-3, f32::INFINITY, e, e, nu, nu);
    let b = hertzian_sphere_sphere(300.0, f32::INFINITY, 5.0e-3, e, e, nu, nu);
    assert_eq!(
        a, b,
        "swapping which body is the rail must not change the result"
    );
}

// ============================================================================
// Section 3: `rolling_contact_life_cycles` — must compose
// `hertzian_sphere_sphere` + `basquin_cycles_to_failure` (peak pressure as
// the stress), not a swapped/hard-coded value. Checked against a direct
// two-call composition for all three presets at a shared self-mated
// geometry (P=1000N, R1=R2=8mm), and against an independent f64 hand
// derivation of the full chain.
// ============================================================================

#[test]
fn combined_life_matches_direct_two_call_composition_for_every_preset() {
    let presets = [
        materials::bearing_steel_52100(),
        materials::gear_steel_8620(),
        materials::silicon_nitride(),
    ];
    for (e, nu, c, m, se) in presets {
        let contact = hertzian_sphere_sphere(1000.0, 8.0e-3, 8.0e-3, e, e, nu, nu);
        let direct = basquin_cycles_to_failure(contact.peak_pressure_pa, c, m, se);
        let combined = rolling_contact_life_cycles(1000.0, 8.0e-3, 8.0e-3, e, e, nu, nu, c, m, se);
        assert_eq!(
            direct, combined,
            "rolling_contact_life_cycles must equal hertzian_sphere_sphere(..).peak_pressure_pa \
             fed into basquin_cycles_to_failure, for E={e}"
        );
    }
}

#[test]
fn combined_life_matches_hand_derived_closed_form_for_bearing_steel_52100() {
    let (e, nu, c, m, se) = materials::bearing_steel_52100();
    let cycles = rolling_contact_life_cycles(1000.0, 8.0e-3, 8.0e-3, e, e, nu, nu, c, m, se);
    // f64 reference: p_max = 5440337228.969287 Pa, N = C * p_max^(-m) =
    // 310522.1190501148 cycles.
    assert_rel(
        cycles,
        310522.1190501148,
        1.0e-3,
        "bearing_steel_52100 combined life",
    );
}

#[test]
fn combined_life_matches_hand_derived_closed_form_for_gear_steel_8620() {
    let (e, nu, c, m, se) = materials::gear_steel_8620();
    let cycles = rolling_contact_life_cycles(1000.0, 8.0e-3, 8.0e-3, e, e, nu, nu, c, m, se);
    // f64 reference: p_max = 5330620364.969696 Pa, N = 1871.7644538455354 cycles.
    assert_rel(
        cycles,
        1871.7644538455354,
        1.0e-3,
        "gear_steel_8620 combined life",
    );
}

#[test]
fn combined_life_matches_hand_derived_closed_form_for_silicon_nitride() {
    let (e, nu, c, m, se) = materials::silicon_nitride();
    let cycles = rolling_contact_life_cycles(1000.0, 8.0e-3, 8.0e-3, e, e, nu, nu, c, m, se);
    // f64 reference: p_max = 6966221745.672105 Pa, N = 2.8353028592829093 cycles.
    assert_rel(
        cycles,
        2.8353028592829093,
        1.0e-3,
        "silicon_nitride combined life",
    );
}

#[test]
fn combined_life_forwards_the_endurance_limit_argument_not_a_hardcoded_one() {
    // Light load, modest radius: f64 reference p_max = 801694948.5899523 Pa,
    // which is BELOW bearing_steel_52100's endurance limit (1.5e9 Pa) — the
    // composed wrapper must return the exact INFINITY sentinel here. A
    // mutation that hardcodes the inner endurance_limit_pa to 0.0 (instead
    // of forwarding the caller's `se`) would see 801694948.59 > 0.0 and
    // return a large finite cycle count instead, which this test catches.
    let (e, nu, c, m, se) = materials::bearing_steel_52100();
    assert_eq!(
        se, 1.5e9,
        "precondition: bearing_steel_52100 endurance limit"
    );
    let cycles = rolling_contact_life_cycles(5.0, 1.0e-2, 1.0e-2, e, e, nu, nu, c, m, se);
    assert_eq!(
        cycles,
        f32::INFINITY,
        "peak pressure (~8.017e8 Pa) is below the endurance limit (1.5e9 Pa); life must be \
         exactly INFINITY, not a finite value computed against a hardcoded/zero threshold"
    );
}

// ============================================================================
// Section 4: pure Basquin oracle at a shared stress (2 GPa) across two
// presets independent of any Hertz call, plus the exact INFINITY sentinel
// at and below the endurance limit.
// ============================================================================

#[test]
fn basquin_matches_hand_derived_closed_form_at_2gpa_for_bearing_and_gear_steel() {
    let (_, _, c1, m1, se1) = materials::bearing_steel_52100();
    let n1 = basquin_cycles_to_failure(2.0e9, c1, m1, se1);
    assert_rel(n1, 6_250_000.0, 1.0e-4, "bearing_steel_52100 at 2 GPa");

    let (_, _, c2, m2, se2) = materials::gear_steel_8620();
    let n2 = basquin_cycles_to_failure(2.0e9, c2, m2, se2);
    assert_rel(n2, 43116.551920662794, 1.0e-4, "gear_steel_8620 at 2 GPa");

    // The two presets must disagree here (different C and m), otherwise a
    // mutation that ignores the material-specific C/m and uses a fixed pair
    // would be unobservable.
    assert_ne!(
        n1, n2,
        "bearing and gear steel must give different lives at the same stress"
    );
}

#[test]
fn basquin_returns_the_exact_infinity_sentinel_at_and_below_the_endurance_limit() {
    let (_, _, c, m, se) = materials::bearing_steel_52100();
    // Strictly below.
    assert_eq!(
        basquin_cycles_to_failure(se - 1.0e6, c, m, se),
        f32::INFINITY,
        "below se"
    );
    // Exactly at the boundary (the comparison is `<=`).
    assert_eq!(
        basquin_cycles_to_failure(se, c, m, se),
        f32::INFINITY,
        "at se"
    );
    // Strictly above must NOT be infinite.
    assert!(
        basquin_cycles_to_failure(se + 1.0e6, c, m, se).is_finite(),
        "just above se must be finite"
    );
}

// ============================================================================
// Section 5: degenerate input — the five documented nonpositive-input
// panics for `hertzian_sphere_sphere`, exercised together so a change that
// drops one of the five `assert!` guards (and lets that one input through
// silently) is caught even though the other four still panic correctly.
// ============================================================================

#[test]
fn hertzian_sphere_sphere_panics_on_every_documented_nonpositive_input() {
    let ok = (
        100.0_f32,
        1.0e-3_f32,
        1.0e-3_f32,
        210.0e9_f32,
        210.0e9_f32,
        0.3_f32,
        0.3_f32,
    );

    let cases: [(&str, f32, f32, f32, f32, f32); 5] = [
        ("zero load", 0.0, ok.1, ok.3, ok.4, ok.5),
        ("negative load", -100.0, ok.1, ok.3, ok.4, ok.5),
        ("zero radius_1", ok.0, 0.0, ok.3, ok.4, ok.5),
        ("zero elastic_modulus_1", ok.0, ok.1, 0.0, ok.4, ok.5),
        ("negative elastic_modulus_2", ok.0, ok.1, ok.3, -1.0, ok.5),
    ];

    for (label, load, r1, e1, e2, nu1) in cases {
        let result = catch_unwind(AssertUnwindSafe(|| {
            hertzian_sphere_sphere(load, r1, ok.2, e1, e2, nu1, ok.6)
        }));
        assert!(result.is_err(), "{label}: must panic, got {result:?}");
    }

    // radius_2_m uses the same guard as radius_1_m; checked separately since
    // the tuple above only varies one argument per case.
    let r2_zero = catch_unwind(AssertUnwindSafe(|| {
        hertzian_sphere_sphere(ok.0, ok.1, 0.0, ok.3, ok.4, ok.5, ok.6)
    }));
    assert!(r2_zero.is_err(), "zero radius_2: must panic");
}

#[test]
fn basquin_cycles_to_failure_panics_on_every_documented_nonpositive_input() {
    let ok = (1.0e9_f32, 5.0e34_f32, 3.0_f32, 1.0e8_f32);

    let cases: [(&str, f32, f32, f32); 3] = [
        ("negative stress", -1.0, ok.1, ok.2),
        ("zero coefficient", ok.0, 0.0, ok.2),
        ("zero exponent", ok.0, ok.1, 0.0),
    ];
    for (label, stress, coeff, exp) in cases {
        let result = catch_unwind(AssertUnwindSafe(|| {
            basquin_cycles_to_failure(stress, coeff, exp, ok.3)
        }));
        assert!(result.is_err(), "{label}: must panic, got {result:?}");
    }
}

#[test]
fn rolling_contact_life_cycles_propagates_the_inner_hertz_panic_rather_than_swallowing_it() {
    let (e, nu, c, m, se) = materials::bearing_steel_52100();
    let result = catch_unwind(AssertUnwindSafe(|| {
        rolling_contact_life_cycles(0.0, 8.0e-3, 8.0e-3, e, e, nu, nu, c, m, se)
    }));
    let err = result
        .expect_err("zero load must panic through the composed wrapper, not return a silent value");
    let msg = err
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| err.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(
        msg.contains("load must be positive"),
        "panic message must be the inner hertzian_sphere_sphere one, got {msg:?}"
    );
}

// ============================================================================
// Section 6: extreme/overflow inputs, measured with `catch_unwind` (per
// rules/analytic-oracle-tests.md — "panic しない" is not assumed, it is
// checked, and the resulting value is pinned exactly, not merely checked for
// finiteness).
// ============================================================================

#[test]
fn hertzian_sphere_sphere_overflow_contact_radius_goes_to_infinity_pressure_collapses_to_zero() {
    // 3*P*R*/(4*E*) with P=1e30 (well under f32::MAX ~3.4e38) and an
    // extremely soft, large pair of spheres (R=1m, E=1e-30 Pa) makes the
    // division's true mathematical result (~7.5e59) exceed f32::MAX, so the
    // division itself overflows to +inf — not an inf/inf indeterminate form,
    // since both operands of that division are individually finite.
    let result = catch_unwind(AssertUnwindSafe(|| {
        hertzian_sphere_sphere(1.0e30, 1.0, 1.0, 1.0e-30, 1.0e-30, 0.0, 0.0)
    }));
    let c = result.expect("overflow must not panic (no assert! guards against it)");

    assert!(
        c.contact_radius_m.is_infinite(),
        "a must overflow to +inf, got {}",
        c.contact_radius_m
    );
    assert!(c.contact_radius_m > 0.0, "a must be +inf, not -inf");
    assert_eq!(
        c.peak_pressure_pa, 0.0,
        "finite numerator (3*P) over infinite denominator (2*pi*a^2) must be exactly 0.0, not NaN"
    );
    assert!(
        !c.peak_pressure_pa.is_nan(),
        "peak pressure must not be NaN"
    );
    assert!(c.max_shear_depth_m.is_infinite(), "0.48*inf must stay +inf");
}

#[test]
fn basquin_cycles_to_failure_extreme_stress_underflows_cleanly_to_zero_cycles() {
    // stress^(-exponent) = (1e10)^(-50) = 1e-500, far below the smallest
    // subnormal f32 (~1.4e-45), so the product with coefficient=1.0 rounds
    // to exactly 0.0 — not a panic, not NaN, not a negative/garbage value.
    let result = catch_unwind(AssertUnwindSafe(|| {
        basquin_cycles_to_failure(1.0e10, 1.0, 50.0, 0.0)
    }));
    let n = result.expect("underflow must not panic");
    assert_eq!(n, 0.0, "must underflow to exactly 0.0 cycles");
    assert!(n.is_finite(), "0.0 is finite, not NaN or inf");
}

#[test]
fn basquin_cycles_to_failure_extreme_coefficient_overflows_cleanly_to_infinite_cycles() {
    // coefficient = f32::MAX, stress^(-1) = 1000.0 (stress = 0.001 Pa), so
    // the product's true mathematical value (~3.4e41) exceeds f32::MAX and
    // the multiplication overflows to +inf.
    let result = catch_unwind(AssertUnwindSafe(|| {
        basquin_cycles_to_failure(0.001, f32::MAX, 1.0, 0.0)
    }));
    let n = result.expect("overflow must not panic");
    assert_eq!(n, f32::INFINITY, "must overflow to exactly +inf cycles");
}
