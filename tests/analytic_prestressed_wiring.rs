//! Wiring + closed-form + degenerate-input oracle for `prestressed` (Motosh
//! bolt preload / VDI 2230 and parabolic cable pretension).
//!
//! Expected values below are derived by hand from the formulas documented in
//! `src/prestressed.rs`'s module doc (Motosh `F_i = T/(K·d)`, Shigley §8-8
//! `F_i = S_p·A_t·fraction`, bolted-joint diagram `C = k_b/(k_b+k_m)`,
//! `F_b = F_i + C·P_ext`, `P_sep = F_i/(1-C)`, Irvine's parabolic cable
//! `T ≈ w·L²/(8s)`, `k = 8T/L`) — **not** by calling the functions under
//! test. Where the true quotient is a terminating binary fraction the
//! assertion is bit-exact (`assert_eq!`); where it is not (division by a
//! non-power-of-two constant, e.g. the mm→m `/1000` inside
//! `preload_from_torque`, or a sag that does not divide evenly) a tight
//! tolerance is used instead, same convention as the module's own unit
//! tests.
//!
//! No input in this module is an array / collection / dimensioned object
//! (every parameter is a scalar `Fix128`), so the "unexpected dimension"
//! degenerate-input category from the common brief does not apply here;
//! the degenerate categories actually exercised are: zero external load,
//! the separation-load boundary, zero/negative preload or torque inputs,
//! a zero-sag cable, and extreme-magnitude overflow.

#![cfg(feature = "std")]

use alice_physics::math::Fix128;
use alice_physics::prestressed::{
    bolt_load_fraction, bolt_peak_tension, cable_pretension_n, preload_from_torque,
    recommended_preload_n, separation_load_n, tensioned_cable_stiffness_n_per_mm,
};
use std::panic;

fn fix_close(a: Fix128, b: Fix128, tol: Fix128) -> bool {
    let d = if a > b { a - b } else { b - a };
    d <= tol
}

/// `i64::MAX >> 8` is the exact sentinel `preload_from_torque`'s siblings use
/// in place of `+inf` on a degenerate (zero) divisor that is *not* routed
/// through `Fix128::Div`'s own "div by zero returns ZERO" contract —
/// `cable_pretension_n` (sag == 0) and `separation_load_n` (C == 1).
fn sentinel() -> Fix128 {
    Fix128::from_int(i64::MAX >> 8)
}

// ============================================================================
// preload_from_torque: F_i = T / (K · d), d converted mm -> m first.
// The mm->m conversion divides by 1000 (not a power of two), so even a
// "clean" decimal closed form is not bit-exact in Q64.64 — tolerance is used.
// ============================================================================

#[test]
fn preload_from_torque_matches_motosh_formula_closed_form() {
    // T = 18 N*m, K = 0.15 (lubricated), d = 12 mm.
    // F_i = 18 / (0.15 * 0.012) = 18 / 0.0018 = 10_000 N.
    let f = preload_from_torque(
        Fix128::from_int(18),
        Fix128::from_ratio(15, 100),
        Fix128::from_int(12),
    );
    assert!(
        fix_close(f, Fix128::from_int(10_000), Fix128::from_ratio(1, 10)),
        "got {}",
        f.to_f64()
    );
}

#[test]
fn preload_from_torque_zero_torque_is_zero_preload() {
    // No validation on torque_nm itself; T=0 just reads through the formula
    // as F_i = 0/(K*d) = 0 (not a special case, not an Err).
    let f = preload_from_torque(Fix128::ZERO, Fix128::from_ratio(2, 10), Fix128::from_int(8));
    assert_eq!(f, Fix128::ZERO);
}

#[test]
fn preload_from_torque_negative_torque_accepted_as_is_no_err() {
    // There is no Result/Option return and no sign validation anywhere in
    // the module: a negative applied torque (e.g. a loosening torque
    // reading) produces a negative preload, mirroring the positive case
    // exactly negated, and neither panics.
    let pos = preload_from_torque(
        Fix128::from_int(18),
        Fix128::from_ratio(15, 100),
        Fix128::from_int(12),
    );
    let neg = preload_from_torque(
        Fix128::from_int(-18),
        Fix128::from_ratio(15, 100),
        Fix128::from_int(12),
    );
    assert_eq!(neg, -pos);
    assert!(panic::catch_unwind(|| preload_from_torque(
        Fix128::from_int(-18),
        Fix128::from_ratio(15, 100),
        Fix128::from_int(12),
    ))
    .is_ok());
}

// ============================================================================
// recommended_preload_n: F_i = S_p * A_t * fraction (Shigley SS8-8).
// Chosen with an exact binary fraction (0.5) and integer S_p/A_t so every
// intermediate product is a terminating binary fraction - bit exact.
// ============================================================================

#[test]
fn recommended_preload_n_matches_shigley_formula_closed_form_exact() {
    // S_p = 640 MPa, A_t = 15 mm^2, fraction = 0.5 (exact: 1/2).
    // F_i = 640 * 15 * 0.5 = 4800 N exactly.
    let f = recommended_preload_n(
        Fix128::from_int(640),
        Fix128::from_int(15),
        Fix128::from_ratio(1, 2),
    );
    assert_eq!(f, Fix128::from_int(4800));
}

#[test]
fn recommended_preload_n_negative_proof_strength_accepted_as_is_no_err() {
    // Every field/argument is unchecked (no constructor, no Result): a
    // negative S_p (a modeling/data-entry mistake, not a physical value)
    // propagates as a negated result, not an Err and not a panic.
    let neg = recommended_preload_n(
        Fix128::from_int(-640),
        Fix128::from_int(15),
        Fix128::from_ratio(1, 2),
    );
    assert_eq!(neg, Fix128::from_int(-4800));
}

// ============================================================================
// bolt_load_fraction: C = k_b / (k_b + k_m). Chosen so both the reference
// value and the scale-invariance check (C depends only on the *ratio* of
// the two stiffnesses) land on terminating binary fractions.
// ============================================================================

#[test]
fn bolt_load_fraction_matches_stiffness_ratio_closed_form_exact() {
    // k_b = 750_000, k_m = 250_000 -> C = 750000/1000000 = 3/4 = 0.75 exact.
    let c = bolt_load_fraction(Fix128::from_int(750_000), Fix128::from_int(250_000));
    assert_eq!(c, Fix128::from_ratio(3, 4));
}

#[test]
fn bolt_load_fraction_is_scale_invariant_both_directions() {
    // C is a function of the *ratio* k_b : k_m only. Scaling both stiffnesses
    // by the same positive factor must leave C bit-identical -- and a
    // formula that accidentally used a sum/difference instead of a ratio
    // (or that dropped one of the two arguments) would not have this
    // property, so this also catches sum-instead-of-ratio mutations that a
    // single reference value could miss.
    let base = bolt_load_fraction(Fix128::from_int(3), Fix128::from_int(5)); // 3/8
    for scale in [2i64, 4, 8, 16, 1000] {
        let scaled = bolt_load_fraction(Fix128::from_int(3 * scale), Fix128::from_int(5 * scale));
        assert_eq!(
            scaled, base,
            "scale={scale}: C must be scale-invariant (ratio-only)"
        );
    }
}

#[test]
fn bolt_load_fraction_zero_bolt_stiffness_is_zero() {
    // An infinitely soft bolt carries none of the external load.
    let c = bolt_load_fraction(Fix128::ZERO, Fix128::from_int(1_000_000));
    assert_eq!(c, Fix128::ZERO);
}

// ============================================================================
// bolt_peak_tension / separation_load_n: the classic bolted-joint diagram.
// F_b(P_ext) = F_i + C*P_ext, and at the closed-form separation boundary
// P_sep = F_i/(1-C), the bolt tension F_b(P_sep) equals P_sep exactly
// (F_i + C*F_i/(1-C) = F_i*(1 + C/(1-C)) = F_i/(1-C) = P_sep) -- an
// algebraic identity independent of the magnitude of F_i or C. C = 0.5 is
// chosen so every operation (the halving, the doubling) is exact in Q64.64.
// ============================================================================

#[test]
fn bolt_peak_tension_zero_external_load_equals_preload_exactly() {
    // P_ext = 0 => no separation, bolt carries exactly the preload,
    // regardless of the load-sharing fraction.
    let preload = Fix128::from_int(6250);
    for c in [Fix128::ZERO, Fix128::from_ratio(3, 10), Fix128::ONE] {
        let f = bolt_peak_tension(preload, c, Fix128::ZERO);
        assert_eq!(
            f,
            preload,
            "C={}: at rest the bolt carries only preload",
            c.to_f64()
        );
    }
}

#[test]
fn separation_load_and_peak_tension_agree_at_boundary_closed_form_exact() {
    // F_i = 7000 N, C = 0.5 (exact: 1/2).
    let preload = Fix128::from_int(7000);
    let c = Fix128::from_ratio(1, 2);

    // Hand-derived, independent of separation_load_n: P_sep = 7000/0.5 = 14000.
    let hand_derived_sep = Fix128::from_int(14_000);
    assert_eq!(separation_load_n(preload, c), hand_derived_sep);

    // Hand-derived, independent of bolt_peak_tension, using the literal
    // boundary value above (not the function's own output):
    // F_b(14000) = 7000 + 0.5*14000 = 7000 + 7000 = 14000 = P_sep.
    let f_b_at_boundary = bolt_peak_tension(preload, c, hand_derived_sep);
    assert_eq!(
        f_b_at_boundary, hand_derived_sep,
        "at the separation load the bolt tension equals the separation load itself"
    );

    // And just below the boundary the bolt must carry strictly less.
    let just_below = hand_derived_sep - Fix128::from_int(1);
    let f_b_below = bolt_peak_tension(preload, c, just_below);
    assert!(f_b_below < hand_derived_sep);
}

#[test]
fn separation_load_n_c_equals_one_returns_the_exact_sentinel() {
    // 1 - C == 0 (bolt carries 100% of any external load, infinitely stiff
    // relative to the members): the module documents a saturating sentinel
    // rather than Fix128::Div's own "divide by zero -> ZERO" contract.
    let sep = separation_load_n(Fix128::from_int(1000), Fix128::ONE);
    assert_eq!(sep, sentinel());
}

#[test]
fn separation_load_grows_without_bound_as_c_approaches_one() {
    // Both-direction monotonicity: strictly increasing C (below 1) strictly
    // increases P_sep for a fixed, positive preload.
    let preload = Fix128::from_int(1000);
    let mut prev = separation_load_n(preload, Fix128::ZERO);
    for num in [1i64, 2, 3, 4, 5, 6, 7, 8, 9] {
        let c = Fix128::from_ratio(num, 10);
        let sep = separation_load_n(preload, c);
        assert!(sep > prev, "C={num}/10 must raise P_sep monotonically");
        prev = sep;
    }
}

// ============================================================================
// cable_pretension_n: T ~= w*L^2/(8s) (parabolic sag approximation).
// ============================================================================

#[test]
fn cable_pretension_n_matches_parabolic_formula_closed_form_exact() {
    // w = 4 N/mm, L = 100 mm, s = 50 mm.
    // T = 4 * 100^2 / (8*50) = 4*10000/400 = 40000/400 = 100 N exactly
    // (the quotient is an exact integer, so no binary rounding occurs).
    let t = cable_pretension_n(
        Fix128::from_int(4),
        Fix128::from_int(100),
        Fix128::from_int(50),
    );
    assert_eq!(t, Fix128::from_int(100));
}

#[test]
fn cable_pretension_n_realistic_scenario_closed_form_tolerance() {
    // w = 0.125 N/mm, L = 2000 mm, s = 300 mm.
    // T = 0.125 * 2000^2 / (8*300) = 0.125*4_000_000/2400 = 500000/2400
    //   = 208.33... N (not a terminating binary fraction - tolerance used).
    let t = cable_pretension_n(
        Fix128::from_ratio(1, 8),
        Fix128::from_int(2000),
        Fix128::from_int(300),
    );
    let expected = Fix128::from_ratio(625, 3); // 500000/2400 reduced = 625/3
    assert!(
        fix_close(t, expected, Fix128::from_ratio(1, 1000)),
        "got {} expected ~{}",
        t.to_f64(),
        expected.to_f64()
    );
}

#[test]
fn cable_pretension_n_zero_sag_returns_the_exact_sentinel() {
    // A zero-sag cable is a degenerate (taut, infinite-tension) parabola.
    // Pins the *exact* sentinel value, not merely "very large" (the
    // module's own unit test only checks `> 1_000_000`).
    let t = cable_pretension_n(Fix128::ONE, Fix128::from_int(100), Fix128::ZERO);
    assert_eq!(t, sentinel());
}

#[test]
fn cable_pretension_n_negative_sag_not_specially_guarded() {
    // The guard is `sag_mm.is_zero()`, not `sag_mm <= ZERO`: a negative sag
    // (an invalid but not explicitly rejected input) falls through to the
    // ordinary formula and simply flips the sign of the result, it is not
    // routed to the zero-sag sentinel and it does not panic.
    let t = cable_pretension_n(
        Fix128::from_int(4),
        Fix128::from_int(100),
        Fix128::from_int(-50),
    );
    assert_eq!(t, Fix128::from_int(-100));
}

// ============================================================================
// tensioned_cable_stiffness_n_per_mm: k = 8*T/L.
// ============================================================================

#[test]
fn tensioned_cable_stiffness_matches_closed_form_exact() {
    // T = 900 N, L = 50 mm. k = 8*900/50 = 7200/50 = 144 N/mm exactly
    // (zero remainder, regardless of 50 not being a power of two).
    let k = tensioned_cable_stiffness_n_per_mm(Fix128::from_int(900), Fix128::from_int(50));
    assert_eq!(k, Fix128::from_int(144));
}

#[test]
fn tensioned_cable_stiffness_zero_span_is_zero_not_sentinel() {
    // Unlike cable_pretension_n's zero-sag and separation_load_n's C==1,
    // this guard returns ZERO on the degenerate (zero-length cable) input,
    // not the saturating sentinel -- the two degenerate conventions in this
    // module are not interchangeable, pin which one applies here.
    let k = tensioned_cable_stiffness_n_per_mm(Fix128::from_int(1000), Fix128::ZERO);
    assert_eq!(k, Fix128::ZERO);
    assert_ne!(k, sentinel());
}

// ============================================================================
// Extreme-magnitude inputs: every Fix128 operator used in this module
// (`Add`/`Sub`/`Mul`/`Div`) is documented wrapping/panic-free
// (math.rs:421-456 Mul, math.rs:626-679 Div), so "no panic" here is a
// crate-wide contract this module inherits, not something it adds -- pin
// that none of the seven functions breaks that contract.
// ============================================================================

#[test]
fn extreme_magnitude_inputs_never_panic() {
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);
    let tiny = Fix128::from_raw(i64::MIN, 0);

    assert!(panic::catch_unwind(|| preload_from_torque(huge, Fix128::ONE, huge)).is_ok());
    assert!(panic::catch_unwind(|| preload_from_torque(huge, tiny, huge)).is_ok());
    assert!(panic::catch_unwind(|| recommended_preload_n(huge, huge, huge)).is_ok());
    assert!(panic::catch_unwind(|| bolt_load_fraction(huge, huge)).is_ok());
    assert!(panic::catch_unwind(|| bolt_load_fraction(huge, tiny)).is_ok());
    assert!(panic::catch_unwind(|| bolt_peak_tension(huge, huge, huge)).is_ok());
    assert!(panic::catch_unwind(|| separation_load_n(huge, huge)).is_ok());
    assert!(panic::catch_unwind(|| separation_load_n(huge, Fix128::ONE)).is_ok());
    assert!(panic::catch_unwind(|| cable_pretension_n(huge, huge, huge)).is_ok());
    assert!(panic::catch_unwind(|| cable_pretension_n(huge, huge, Fix128::ZERO)).is_ok());
    assert!(panic::catch_unwind(|| tensioned_cable_stiffness_n_per_mm(huge, huge)).is_ok());
}
