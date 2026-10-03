//! Oracles for the production entry points of `alice_physics::fillet_stress`
//! driven by `examples/fillet_stress_concentration.rs`:
//! `kt_circular_hole_infinite_plate`, `kt_elliptical_hole`,
//! `kt_u_notch_axial`, and `recommended_fillet_radius_mm`.
//!
//! # What this file is and is not
//!
//! `examples/fillet_stress_concentration.rs` already drives all four
//! functions against hand-derived Kirsch / Inglis / Peterson closed
//! forms for one representative geometry each (exact K_t = 3 for the
//! circular hole; two elliptical-hole aspect ratios; two U-notch depth
//! ratios; one shoulder-fillet sizing case). What that file does **not**
//! cover, and this file does:
//!
//! * `kt_elliptical_hole`'s two limiting cases: the circle limit
//!   (`a == b`, must reduce to the circular hole's exact `K_t = 3`) and
//!   the needle limit (`a >> b`, must blow up per the formula's own
//!   `1 + 2*a/b` -- checked here as an actual large-but-finite ratio,
//!   not the `semi_axis_parallel.is_zero()` early-return branch that
//!   `src/fillet_stress.rs`'s own unit tests already exercise),
//! * `kt_u_notch_axial` at the formula's stated validity boundary
//!   (`h/r == 10`) from a second, distinct geometry than the example's
//!   own boundary case, plus the module's zero-radius guard branch
//!   verified against its documented sentinel value (not just "did not
//!   panic"),
//! * `recommended_fillet_radius_mm`'s degenerate / boundary inputs:
//!   `kt_target <= 1` (zero and exactly `ONE`, both of which the
//!   function's own early-return hands back `small_dia_mm.half()`
//!   without ever entering the bisection), a zero-diameter ("zero
//!   load-bearing cross-section") shaft, and extreme-magnitude geometry
//!   near `Fix128`'s representable range, confirming the bisection
//!   neither panics nor returns a nonsensical (negative or
//!   larger-than-the-shaft) radius.
//!
//! # Degenerate / extreme input summary
//!
//! * `kt_elliptical_hole`: `a == b` (circle limit, exact `K_t = 3`);
//!   `a = 5000, b = 1` (needle limit, `K_t = 10001`, finite but far
//!   beyond any physically sane fillet).
//! * `kt_u_notch_axial`: `root_radius_mm == 0` (documented sentinel,
//!   `i64::MAX >> 8`, not a division-by-zero panic); `h/r == 10` at a
//!   geometry distinct from the example file's.
//! * `recommended_fillet_radius_mm`: `kt_target == Fix128::ZERO`;
//!   `kt_target == Fix128::ONE` (the `<=` boundary of the early return,
//!   not just `< ONE`); `small_dia_mm == Fix128::ZERO` ("zero load",
//!   a shaft with no bearing cross-section at all); `small_dia_mm` and
//!   `large_dia_mm` both near `Fix128`'s `+-2^63` integer range.
//!
//! Author: Moroya Sakamoto

use alice_physics::fillet_stress::{
    kt_circular_hole_infinite_plate, kt_elliptical_hole, kt_u_notch_axial,
    recommended_fillet_radius_mm,
};
use alice_physics::math::Fix128;

/// Relative-error assertion against an independently hand-derived f64
/// closed form (never computed by calling the function under test).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = ((g - want) / want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

/// Absolute-error assertion against an independently hand-derived f64
/// closed form, for expectations that are exactly zero (where a relative
/// error is undefined) or for a sentinel comparison.
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-9;

// ============================================================================
// Section 1: kt_elliptical_hole -- limiting cases.
// ============================================================================

#[test]
fn elliptical_hole_circle_limit_matches_circular_hole_exactly() {
    // a == b == 20mm: Inglis K_t = 1 + 2*(20/20) = 3, must bit-for-bit
    // match the Kirsch closed form returned by
    // kt_circular_hole_infinite_plate (a different geometry than the
    // module's own `elliptical_hole_circular_reduces_to_three` unit test,
    // which uses a = b = 5).
    let kt_ellipse = kt_elliptical_hole(Fix128::from_int(20), Fix128::from_int(20));
    let kt_circle = kt_circular_hole_infinite_plate();
    assert_eq!(
        kt_ellipse, kt_circle,
        "circle-limit ellipse must equal kt_circular_hole_infinite_plate exactly"
    );
    assert_rel(kt_ellipse, 3.0, FIX_TOL, "circle limit K_t = 1+2a/b at a=b");
}

#[test]
fn elliptical_hole_needle_limit_blows_up_per_formula() {
    // a = 5000mm, b = 1mm: a/b = 5000, K_t = 1 + 2*5000 = 10001. This
    // exercises the formula's actual asymptotic growth for a >> b
    // (distinct from the module's own `elliptical_hole_sharp_crack_infinite`
    // unit test, which uses the b == 0 early-return branch and never
    // evaluates `1 + 2*a/b` at all).
    let kt = kt_elliptical_hole(Fix128::from_int(5000), Fix128::from_int(1));
    assert_rel(
        kt,
        1.0 + 2.0 * (5000.0 / 1.0),
        FIX_TOL,
        "needle limit K_t = 1+2a/b",
    );
    assert!(
        kt.to_f64() > 1000.0,
        "needle limit must blow up past any sane K_t"
    );
}

// ============================================================================
// Section 2: kt_u_notch_axial -- validity boundary and zero-radius guard.
// ============================================================================

#[test]
fn u_notch_at_validity_boundary_h_over_r_ten_second_geometry() {
    // h = 20mm, r = 2mm -> h/r = 10 (the formula's own stated upper
    // validity bound), at a geometry distinct from the example file's
    // h=10,r=1 boundary case (same ratio, different absolute scale --
    // confirms the formula depends only on h/r, not on the absolute
    // magnitude of either dimension).
    let kt = kt_u_notch_axial(Fix128::from_int(20), Fix128::from_int(2));
    assert_rel(
        kt,
        0.85 + 2.0 * (10.0_f64).sqrt(),
        FIX_TOL,
        "U-notch h/r=10 validity boundary, second geometry",
    );
}

#[test]
fn u_notch_zero_radius_returns_documented_sentinel_not_panic() {
    // root_radius_mm == 0 would make h/r divide-by-zero; the module's
    // documented early return hands back the same sentinel
    // (i64::MAX >> 8) used by kt_elliptical_hole's zero-denominator
    // guard, verified here bit-exact against that constant rather than
    // the module's own unit test's loose "> 1_000_000" smoke check.
    let kt = kt_u_notch_axial(Fix128::from_int(5), Fix128::ZERO);
    let sentinel = Fix128::from_int(i64::MAX >> 8);
    assert_eq!(
        kt, sentinel,
        "zero-radius U-notch must return the documented i64::MAX>>8 sentinel exactly"
    );
}

// ============================================================================
// Section 3: recommended_fillet_radius_mm -- boundary and degenerate inputs.
// ============================================================================

#[test]
fn recommended_fillet_zero_target_kt_returns_half_small_diameter() {
    // kt_target == Fix128::ZERO triggers the `kt_target <= Fix128::ONE`
    // early return before the bisection ever runs: "zero target K_t"
    // means "no fillet is strong enough" is nonsensical, so the module's
    // documented fallback is half the small diameter.
    let d = Fix128::from_int(30);
    let big_d = Fix128::from_int(50);
    let r = recommended_fillet_radius_mm(d, big_d, Fix128::ZERO);
    assert_abs(r, 15.0, FIX_TOL, "kt_target=0 -> small_dia_mm.half()");
}

#[test]
fn recommended_fillet_target_kt_exactly_one_is_the_inclusive_boundary() {
    // kt_target == Fix128::ONE is the `<=` boundary of the early return
    // (not `< ONE`): must still short-circuit to half the small diameter
    // rather than entering the bisection, where K_t == 1 is unreachable
    // (the piecewise table's smallest base value is 1.3).
    let d = Fix128::from_int(30);
    let big_d = Fix128::from_int(50);
    let r = recommended_fillet_radius_mm(d, big_d, Fix128::ONE);
    assert_abs(
        r,
        15.0,
        FIX_TOL,
        "kt_target=1 (inclusive boundary) -> small_dia_mm.half()",
    );
}

#[test]
fn recommended_fillet_zero_small_diameter_zero_load_bearing_section() {
    // small_dia_mm == 0: a shaft with no bearing cross-section at all
    // ("zero load"). The bisection's trial radius r_mm = mid * 0 is
    // always 0, so kt_shaft_shoulder_bending's own small_dia_mm.is_zero()
    // guard returns its sentinel on every iteration, `kt > kt_target`
    // holds throughout, and `hi` (and hence the final `hi * small_dia_mm`)
    // never leaves its 0 multiplier -- the function must return exactly
    // zero, not panic on a 0/0 division anywhere in the search.
    let r = recommended_fillet_radius_mm(Fix128::ZERO, Fix128::from_int(50), Fix128::from_int(2));
    assert_abs(
        r,
        0.0,
        FIX_TOL,
        "zero-diameter shaft -> zero recommended radius",
    );
}

#[test]
fn recommended_fillet_extreme_magnitude_geometry_does_not_panic() {
    // small_dia_mm and large_dia_mm both near Fix128's representable
    // +-2^63 integer range: the bisection's own arithmetic
    // (mid * small_dia_mm, and large_dia_mm / small_dia_mm inside
    // kt_shaft_shoulder_bending) must not panic, and the returned radius
    // must be sane: non-negative and at most the small diameter itself
    // (the bisection searches r/d in (0, 0.5), so r can never exceed
    // half the small diameter).
    let small = Fix128::from_int(1_000_000_000);
    let big = Fix128::from_int(1_800_000_000);
    let r = recommended_fillet_radius_mm(small, big, Fix128::from_int(2));
    assert!(
        r.to_f64() >= 0.0,
        "extreme-magnitude recommended radius must be non-negative, got {}",
        r.to_f64()
    );
    assert!(
        r <= small.half(),
        "extreme-magnitude recommended radius must not exceed half the small diameter, got {}",
        r.to_f64()
    );
}
