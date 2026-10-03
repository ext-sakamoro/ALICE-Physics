//! Oracles for the production entry points of `alice_physics::contact_viz`
//! driven by `examples/contact_visualization.rs`: `ContactArrow`,
//! `FrictionCone`, `generate_contact_arrows`, `generate_friction_arrows`,
//! and `generate_friction_cones`.
//!
//! # What this file is and is not
//!
//! `examples/contact_visualization.rs` already drives all five items
//! against hand-derived closed forms for a representative normal case (a
//! 3-4-5 triangle normalize, an already-unit normal, and all three
//! branches of the private `tangent_basis` helper's axis selection) --
//! those are not repeated here. What that file does **not** cover, and
//! this file does:
//!
//! * `generate_contact_arrows` at a second, distinct normalize triangle
//!   (5-12-13, so the formula is checked against more than one rational
//!   normal) and the empty-slice input,
//! * `generate_friction_arrows` at the `mu == 0` boundary (friction force
//!   must be exactly zero regardless of normal force, while the tangent
//!   basis itself is unaffected by `mu`), a negative `normal_force`
//!   (friction force must carry the sign through, not clamp it), the
//!   zero-input-normal degenerate case (tangent basis collapses to the
//!   zero vector but `friction_force` is still well-defined), and the
//!   empty-slice input,
//! * `generate_friction_cones` at the `mu == 0` boundary (checked against
//!   the module's own documented CORDIC residual near zero, not exact
//!   zero -- see `src/math.rs`'s `atan` doc comment), a very large `mu`
//!   boundary (`half_angle` must approach `pi/2` without overflowing),
//!   a negative `normal_force` (height must carry the sign through, not
//!   clamp it, since the module performs no clamping), the
//!   zero-input-normal degenerate case, and the empty-slice input.
//!
//! # Degenerate / boundary input summary
//!
//! * `generate_contact_arrows`: zero-length input slice -> empty output
//!   (no panic, no default element).
//! * `generate_friction_arrows`: `mu == Fix128::ZERO` collapses
//!   `friction_force` to exactly zero for any `normal_force`; a
//!   zero-length input normal makes `tangent_basis` degenerate to
//!   `Vec3Fix::ZERO` for both tangent directions (cross product of the
//!   zero vector with anything is zero), yet `friction_force` is still
//!   computed from `normal_force * mu` with no special-casing; a
//!   negative `normal_force` propagates its sign into `friction_force`
//!   unclamped; zero-length input slice -> empty output.
//! * `generate_friction_cones`: `mu == Fix128::ZERO` leaves `half_angle`
//!   within the documented ~5e-15 CORDIC residual of zero, not bit-exact
//!   zero; an extreme `mu` (`1_000_000`) drives `half_angle` arbitrarily
//!   close to `pi/2` without panicking or wrapping; a negative
//!   `normal_force` propagates unclamped into `height`; zero-length
//!   input slice -> empty output.
//!
//! Author: Moroya Sakamoto

use alice_physics::contact_viz::{
    generate_contact_arrows, generate_friction_arrows, generate_friction_cones,
};
use alice_physics::math::{Fix128, Vec3Fix};

/// Absolute-error assertion against an independently hand-derived f64
/// closed form (never computed by calling the function under test).
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

/// Component-wise absolute-error assertion for a `Vec3Fix`.
fn assert_vec3_abs(got: Vec3Fix, want: (f64, f64, f64), tol: f64, what: &str) {
    let g = (got.x.to_f64(), got.y.to_f64(), got.z.to_f64());
    assert!(
        (g.0 - want.0).abs() <= tol && (g.1 - want.1).abs() <= tol && (g.2 - want.2).abs() <= tol,
        "{what}: got {g:?}, want {want:?} (tol {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-9;

// ============================================================================
// generate_contact_arrows
// ============================================================================

/// A second, distinct normalize triangle (5-12-13: `(5, 12, 0)` has
/// length 13, so the normalized direction is the exact rational
/// `(5/13, 12/13, 0)`), independent of the 3-4-5 triangle already checked
/// in `examples/contact_visualization.rs`.
#[test]
fn contact_arrows_normalize_5_12_13_triangle() {
    let contacts = [(
        Vec3Fix::ZERO,
        Vec3Fix::from_int(5, 12, 0),
        Fix128::from_int(26),
    )];
    let arrows = generate_contact_arrows(&contacts);
    assert_vec3_abs(
        arrows[0].normal,
        (5.0 / 13.0, 12.0 / 13.0, 0.0),
        FIX_TOL,
        "normalize(5,12,0) = (5/13, 12/13, 0)",
    );
    assert_abs(
        arrows[0].force_magnitude,
        26.0,
        FIX_TOL,
        "force_magnitude passes through",
    );
    assert!(!arrows[0].is_friction);
}

/// Empty input slice must produce an empty output, not panic or a
/// default-valued element.
#[test]
fn contact_arrows_empty_slice_is_empty() {
    let arrows = generate_contact_arrows(&[]);
    assert!(arrows.is_empty());
}

// ============================================================================
// generate_friction_arrows
// ============================================================================

/// `mu == 0`: `friction_force = normal_force * 0 = 0` for any
/// `normal_force`, including a large one. The tangent basis itself does
/// not depend on `mu`, so the direction is still the (0.8, -0.6)-style
/// pair in the y-z plane for a y-axis normal, same `tangent_basis`
/// algebra as the module's own `#[cfg(test)]` module (`n = UNIT_Y`:
/// `abs_x=0,abs_y=1,abs_z=0` -> `abs_x<abs_y` true, `abs_x<abs_z` false ->
/// falls to `abs_y<abs_z` (1<0, false) -> helper = UNIT_Z ->
/// `t1 = UNIT_Y x UNIT_Z = (1*1-0*0, 0*0-0*1, 0*0-1*0) = (1,0,0)`,
/// `t2 = UNIT_Y x t1 = (1*0-0*0, 0*1-0*0, 0*0-1*1) = (0,0,-1)`).
#[test]
fn friction_arrows_mu_zero_boundary_forces_are_exactly_zero() {
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(1_000))];
    let arrows = generate_friction_arrows(&contacts, Fix128::ZERO);
    assert_eq!(arrows.len(), 2);
    assert_abs(arrows[0].force_magnitude, 0.0, FIX_TOL, "mu=0 -> force=0");
    assert_abs(arrows[1].force_magnitude, 0.0, FIX_TOL, "mu=0 -> force=0");
    assert_vec3_abs(
        arrows[0].normal,
        (1.0, 0.0, 0.0),
        FIX_TOL,
        "t1 for n=UNIT_Y",
    );
    assert_vec3_abs(
        arrows[1].normal,
        (0.0, 0.0, -1.0),
        FIX_TOL,
        "t2 for n=UNIT_Y",
    );
}

/// A negative `normal_force` must propagate its sign into
/// `friction_force` unchanged (the module performs `normal_force * mu`
/// with no clamping or `abs()`), not be zeroed or flipped positive.
#[test]
fn friction_arrows_negative_normal_force_is_not_clamped() {
    let mu = Fix128::from_ratio(1, 2); // 0.5
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_X, Fix128::from_int(-8))];
    let arrows = generate_friction_arrows(&contacts, mu);
    assert_abs(
        arrows[0].force_magnitude,
        -4.0,
        FIX_TOL,
        "friction_force = -8 * 0.5 = -4 (sign preserved, not clamped)",
    );
    assert_abs(
        arrows[1].force_magnitude,
        -4.0,
        FIX_TOL,
        "same for t2 arrow",
    );
}

/// A zero-length input normal degenerates `tangent_basis`'s internal
/// `n.cross(helper)` to `Vec3Fix::ZERO` (cross product of the zero vector
/// with anything is the zero vector), and that zero vector then
/// normalizes to `Vec3Fix::ZERO` again for `t2`. `friction_force` has no
/// dependency on the normal direction, so it is still well-defined and
/// non-degenerate (`normal_force * mu`).
#[test]
fn friction_arrows_zero_normal_degenerates_tangent_basis_not_force() {
    let mu = Fix128::from_ratio(1, 2); // 0.5
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(10))];
    let arrows = generate_friction_arrows(&contacts, mu);
    assert_vec3_abs(
        arrows[0].normal,
        (0.0, 0.0, 0.0),
        FIX_TOL,
        "t1 degenerates to ZERO when input normal is ZERO",
    );
    assert_vec3_abs(
        arrows[1].normal,
        (0.0, 0.0, 0.0),
        FIX_TOL,
        "t2 degenerates to ZERO when input normal is ZERO",
    );
    assert_abs(
        arrows[0].force_magnitude,
        5.0,
        FIX_TOL,
        "friction_force = 10 * 0.5 = 5, unaffected by the degenerate normal",
    );
}

/// Empty input slice must produce an empty output.
#[test]
fn friction_arrows_empty_slice_is_empty() {
    let arrows = generate_friction_arrows(&[], Fix128::from_ratio(1, 2));
    assert!(arrows.is_empty());
}

// ============================================================================
// generate_friction_cones
// ============================================================================

/// `mu == 0`: per `src/math.rs`'s own documented CORDIC residual
/// (`atan(0)` is "~5e-15", not bit-exact zero -- see
/// `tests/analytic_math_wiring.rs::atan_of_zero_is_within_the_documented_cordic_residual_not_exact_zero`
/// in this same repo), `half_angle` must land within a generous `1e-9` of
/// zero, not exactly zero.
#[test]
fn friction_cones_mu_zero_boundary_half_angle_near_zero() {
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(10))];
    let cones = generate_friction_cones(&contacts, Fix128::ZERO);
    assert_abs(
        cones[0].half_angle,
        0.0,
        1e-9,
        "atan(0) is within the documented CORDIC residual of zero",
    );
}

/// A very large `mu` drives `half_angle = atan(mu)` arbitrarily close to
/// `pi/2` without panicking or wrapping through `Fix128`'s signed range.
/// Independent oracle: platform `f64::atan(1_000_000.0)`, a different
/// algorithm from `Fix128::atan`'s CORDIC implementation.
#[test]
#[allow(clippy::disallowed_methods)] // independent f64 libm oracle, never fed back into Fix128 state
fn friction_cones_very_large_mu_boundary_half_angle_approaches_half_pi() {
    let mu = Fix128::from_int(1_000_000);
    let half_angle_ref = 1_000_000.0_f64.atan();
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(10))];
    let cones = generate_friction_cones(&contacts, mu);
    assert_abs(
        cones[0].half_angle,
        half_angle_ref,
        1e-6,
        "atan(1_000_000) approaches pi/2 without overflow",
    );
    let half_pi_ref = std::f64::consts::FRAC_PI_2;
    assert!(
        (cones[0].half_angle.to_f64() - half_pi_ref).abs() < 1e-5,
        "atan(1_000_000) must be within 1e-5 of pi/2"
    );
}

/// A negative `normal_force` must propagate unchanged into `height` (the
/// module performs no clamping or `abs()`), independent of `half_angle`
/// (which depends only on `mu`, not on `normal_force`'s sign).
#[test]
fn friction_cones_negative_normal_force_height_is_not_clamped() {
    let mu = Fix128::from_ratio(3, 10); // 0.3
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::UNIT_Y, Fix128::from_int(-7))];
    let cones = generate_friction_cones(&contacts, mu);
    assert_abs(
        cones[0].height,
        -7.0,
        FIX_TOL,
        "height = normal_force = -7 (sign preserved, not clamped)",
    );
}

/// A zero-length input normal normalizes to `Vec3Fix::ZERO` (the same
/// documented zero-length contract as `generate_contact_arrows`),
/// independent of `half_angle` and `height`, which still take their
/// ordinary values from `mu` and `normal_force`.
#[test]
fn friction_cones_zero_normal_degenerates_normal_not_half_angle_or_height() {
    let mu = Fix128::from_ratio(3, 10); // 0.3
    #[allow(clippy::disallowed_methods)]
    // independent f64 libm oracle, never fed back into Fix128 state
    let half_angle_ref = 0.3_f64.atan();
    let contacts = [(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::from_int(9))];
    let cones = generate_friction_cones(&contacts, mu);
    assert_vec3_abs(
        cones[0].normal,
        (0.0, 0.0, 0.0),
        FIX_TOL,
        "normalize(ZERO) = ZERO",
    );
    assert_abs(
        cones[0].half_angle,
        half_angle_ref,
        1e-9,
        "half_angle = atan(mu), unaffected",
    );
    assert_abs(
        cones[0].height,
        9.0,
        FIX_TOL,
        "height = normal_force, unaffected",
    );
}

/// Empty input slice must produce an empty output.
#[test]
fn friction_cones_empty_slice_is_empty() {
    let cones = generate_friction_cones(&[], Fix128::from_ratio(1, 2));
    assert!(cones.is_empty());
}
