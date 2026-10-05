//! Oracles for the production entry points of `alice_physics::laminate`
//! driven by `examples/laminate_abd_matrix.rs`: `Ply::{q_matrix, q_bar}`,
//! `compute_abd`, `is_symmetric_stack`, and `AbdMatrix::is_symmetric`.
//!
//! # What this file is and is not
//!
//! `tests/engineering_oracles_solid.rs::laminate_cross_ply_abd_matches_jones`
//! and `::laminate_angle_ply_transformation_matches_jones` already drive
//! `q_matrix` / `q_bar` / `compute_abd` / `is_symmetric_stack` /
//! `is_symmetric` against the Jones (*Mechanics of Composite Materials*,
//! 2nd ed.) closed forms for a T300/5208 material at 0 deg / 45 deg and a
//! `[0/90/90/0]` / `[+45/-45]` stack — those are not repeated here. What
//! that file does **not** cover, and this file does:
//!
//! * `q_matrix` at a second, distinct material (so the formula is checked
//!   against more than one set of engineering constants), and both legs
//!   of its `e_l_mpa.is_zero() || e_t_mpa.is_zero()` guard individually,
//!   plus the separate `denom.is_zero()` guard (the Poisson-reciprocity
//!   singularity) -- none of which any existing test exercises; the
//!   module's own unit tests only exercise the ordinary (non-degenerate)
//!   path,
//! * `q_bar` at 0 deg and 90 deg (the two on-axis special cases, checked
//!   here against an independent closed form rather than the loose
//!   `approx_eq` smoke test in `src/laminate.rs`'s own `#[cfg(test)]`
//!   module) and at a genuinely generic (20 deg) angle, plus what happens
//!   when the underlying material has already collapsed through
//!   `q_matrix`'s zero-modulus guard,
//! * `compute_abd` for a single-ply laminate and for a zero-thickness
//!   ply spliced into a stack (an invariant check: splicing in a ply
//!   that contributes nothing to the integral must not change the
//!   result), and an extreme-magnitude input that makes the cubic `D`
//!   integration term wrap through `Fix128`'s `±2^63` range,
//! * `is_symmetric_stack` for the empty slice, an odd-length palindrome
//!   with an unpaired middle ply, and a stack whose mirrored plies match
//!   on thickness and orientation but differ on material -- the existing
//!   oracle file only exercises even-length stacks that differ (or not)
//!   by angle, and
//! * `AbdMatrix::is_symmetric` as a direct, hand-built `Sym3`/`AbdMatrix`
//!   rather than only as the output of `compute_abd`, isolating the
//!   `<=` tolerance boundary and each of the six `B` components
//!   individually, plus a check that the comparison itself does not
//!   overflow at `Fix128`'s extreme representable magnitude.
//!
//! # Degenerate / extreme input summary
//!
//! * `q_matrix`: `e_l_mpa == 0`, `e_t_mpa == 0`, and the Poisson-
//!   reciprocity singularity `1 - nu_lt*nu_tl == 0` (all three return
//!   `(ZERO, ZERO, ZERO, ZERO)` per the module's documented early return,
//!   not a panic or a division artifact -- verified bit-exact below, not
//!   just "did not panic").
//! * `q_bar`: a material already collapsed to all-zero `Q` propagates
//!   zero regardless of the rotation angle.
//! * `compute_abd`: a zero-thickness ply contributes exactly nothing
//!   (`delta_z = delta_z^2 = delta_z^3 = Fix128::ZERO`, so its `Sym3`
//!   contribution is `Sym3::default()` regardless of its `Q`), and an
//!   extreme total thickness makes `z_upper^3` wrap past `Fix128`'s
//!   `+2^63` integer boundary while `z_lower^3 = -2^63` does not (`-2^63`
//!   is exactly representable as `i64::MIN`, `+2^63` is not), so
//!   `delta_z3` for that ply collapses to exactly zero and the `D`
//!   contribution vanishes -- a documented property of `Fix128`
//!   multiplication's wrapping contract (`src/math.rs`'s `impl Mul for
//!   Fix128` doc comment), not a defect specific to this module, but one
//!   this module's own tests had never exercised: an absurdly thick
//!   laminate silently reports near-zero bending stiffness rather than
//!   an astronomically large one or an error.
//! * `is_symmetric_stack`: the empty slice (`n == 0`, distinct from the
//!   already-tested `n == 1` single-ply case).
//! * `is_symmetric`: `tol == Fix128::ZERO` and `B` at `Fix128`'s extreme
//!   representable magnitude (`i64::MAX`), confirming the comparison
//!   itself (`Fix128`'s `Ord`, not a subtraction or multiplication) does
//!   not need headroom the way an arithmetic op would.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// `q_bar_general_twenty_degree_angle_matches_jones_rotation_formula` needs a
// genuinely generic rotation angle (not the 0/45/90 deg special cases that
// reduce to plain algebra), which means calling `f64::sin`/`f64::cos` for
// the independent oracle side. `clippy.toml`'s `disallowed-methods` bans
// those crate-wide (every transcendental must go through
// `alice_physics::det_math`, not the platform libm) but the ban does not
// apply to a closed-form f64 reference value that is never written back
// into `Fix128` production state -- same convention as
// `tests/engineering_oracles_solid.rs` and
// `tests/analytic_laminate_failure_wiring.rs`.
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic::OrthotropicElasticity;
use alice_physics::laminate::{compute_abd, is_symmetric_stack, AbdMatrix, Ply, Sym3};
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
/// error is undefined).
fn assert_abs(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs();
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-9;

/// A second orthotropic material, distinct from the T300/5208 used by
/// `tests/engineering_oracles_solid.rs` and `examples/laminate_abd_matrix.rs`
/// (E1 = 20 GPa, E2 = 5 GPa, nu12 = 0.25, G12 = 3 GPa -- round numbers with
/// E1 != E2 so the `q_bar` 90 deg axis-swap test below is non-trivial).
fn material_b() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(20_000),
        e_t_mpa: Fix128::from_int(5_000),
        e_z_mpa: Fix128::from_int(5_000),
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        g_lt_mpa: Fix128::from_int(3_000),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    }
}

/// Jones eq. 2.61 reduced-stiffness closed form, independent f64
/// reference (same formula as `examples/laminate_abd_matrix.rs`'s
/// `q_ref_f64`, duplicated here because test binaries cannot import
/// helpers from an example binary).
fn q_ref_f64(e1: f64, e2: f64, nu12: f64, g12: f64) -> (f64, f64, f64, f64) {
    let nu21 = nu12 * e2 / e1;
    let d = 1.0 - nu12 * nu21;
    (e1 / d, nu12 * e2 / d, e2 / d, g12)
}

fn ply_b(angle: Fix128, t: Fix128) -> Ply {
    Ply {
        thickness_mm: t,
        orientation_rad: angle,
        material: material_b(),
    }
}

/// Jones eq. 2.84 axis-transformation closed form, independent f64
/// reference (same formula as `examples/laminate_abd_matrix.rs`'s
/// `q_bar_ref_f64`, duplicated here for the same reason as `q_ref_f64`
/// above). Takes `cos(theta)`/`sin(theta)` rather than `theta` itself so
/// callers choose whether to supply an exact algebraic value (0 deg,
/// 90 deg) or a `f64::cos`/`f64::sin` call (the file-level
/// `#![allow(clippy::disallowed_methods)]` covers the latter).
fn q_bar_ref_f64(
    q11: f64,
    q12: f64,
    q22: f64,
    q66: f64,
    c: f64,
    s: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let c2 = c * c;
    let s2 = s * s;
    let c4 = c2 * c2;
    let s4 = s2 * s2;
    let c2s2 = c2 * s2;
    let c3s = c2 * c * s;
    let cs3 = c * s2 * s;
    let q11_bar = q11 * c4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * s4;
    let q22_bar = q11 * s4 + 2.0 * (q12 + 2.0 * q66) * c2s2 + q22 * c4;
    let q12_bar = (q11 + q22 - 4.0 * q66) * c2s2 + q12 * (c4 + s4);
    let q66_bar = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * c2s2 + q66 * (c4 + s4);
    let q16_bar = (q11 - q12 - 2.0 * q66) * c3s + (q12 - q22 + 2.0 * q66) * cs3;
    let q26_bar = (q11 - q12 - 2.0 * q66) * cs3 + (q12 - q22 + 2.0 * q66) * c3s;
    (q11_bar, q12_bar, q22_bar, q16_bar, q26_bar, q66_bar)
}

// ============================================================================
// Section 1: q_matrix -- a second material, and all three early-return
// (degenerate) branches.
// ============================================================================

#[test]
fn q_matrix_matches_jones_reduced_stiffness_for_a_second_material() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11_ref, q12_ref, q22_ref, q66_ref) = q_ref_f64(e1, e2, nu12, g12);
    let p = ply_b(Fix128::ZERO, Fix128::from_ratio(1, 4));
    let (q11, q12, q22, q66) = p.q_matrix();
    assert_rel(q11, q11_ref, FIX_TOL, "Q11");
    assert_rel(q12, q12_ref, FIX_TOL, "Q12");
    assert_rel(q22, q22_ref, FIX_TOL, "Q22");
    assert_rel(q66, q66_ref, FIX_TOL, "Q66");
}

/// Degenerate: `e_l_mpa == 0` is the first leg of `q_matrix`'s
/// `e_l_mpa.is_zero() || e_t_mpa.is_zero()` guard. `e_t_mpa` is left
/// non-zero so a mutant that drops this leg (or swaps it for an `&&`)
/// would compute a real (non-degenerate) division instead.
#[test]
fn q_matrix_zero_longitudinal_modulus_returns_zero_exactly() {
    let mat = OrthotropicElasticity {
        e_l_mpa: Fix128::ZERO,
        e_t_mpa: Fix128::from_int(5_000),
        e_z_mpa: Fix128::from_int(5_000),
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        g_lt_mpa: Fix128::from_int(3_000),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    };
    let p = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        orientation_rad: Fix128::ZERO,
        material: mat,
    };
    assert_eq!(
        p.q_matrix(),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
}

/// Degenerate: `e_t_mpa == 0` is the second leg of the same guard,
/// exercised with `e_l_mpa` non-zero so the two legs cannot be confused
/// with each other by a mutation that only disables one of them.
#[test]
fn q_matrix_zero_transverse_modulus_returns_zero_exactly() {
    let mat = OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(20_000),
        e_t_mpa: Fix128::ZERO,
        e_z_mpa: Fix128::ZERO,
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        g_lt_mpa: Fix128::from_int(3_000),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    };
    let p = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        orientation_rad: Fix128::ZERO,
        material: mat,
    };
    assert_eq!(
        p.q_matrix(),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
}

/// Degenerate: the Poisson-reciprocity singularity `denom ==
/// 1 - nu_lt*nu_tl == 0`, independent of either modulus being zero.
/// `nu_lt = 1/2` and `e_t/e_l = 4` gives `nu_tl = nu_lt*e_t/e_l = 2`, so
/// `denom = 1 - 0.5*2 = 0` bit-exactly (`nu_lt = 1/2` is an exact power-
/// of-two `Fix128` fraction, so this is not a rounding coincidence --
/// verified against the live `Fix128` computation, not assumed from f64
/// arithmetic).
#[test]
fn q_matrix_poisson_reciprocity_denominator_zero_returns_zero_exactly() {
    let mat = OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(100),
        e_t_mpa: Fix128::from_int(400),
        e_z_mpa: Fix128::from_int(400),
        nu_lt: Fix128::from_ratio(1, 2),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        g_lt_mpa: Fix128::from_int(50),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    };
    let nu_tl = mat.nu_lt * mat.e_t_mpa / mat.e_l_mpa;
    assert_eq!(nu_tl, Fix128::from_int(2), "nu_tl = nu_lt*e_t/e_l");
    let denom = Fix128::ONE - mat.nu_lt * nu_tl;
    assert_eq!(denom, Fix128::ZERO, "denom collapses to exactly zero");

    let p = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        orientation_rad: Fix128::ZERO,
        material: mat,
    };
    assert_eq!(
        p.q_matrix(),
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );
}

// ============================================================================
// Section 2: q_bar -- 0 deg / 90 deg on-axis special cases against an
// independent closed form, a generic 20 deg angle, and zero-modulus
// propagation.
// ============================================================================

/// At theta = 0, `c = 1, s = 0` collapse Jones eq. 2.84 to `Qbar == Q`
/// exactly and the off-axis terms to exactly zero. `src/laminate.rs`'s own
/// `#[cfg(test)]::q_bar_zero_orientation_equals_q` checks this with
/// `approx_eq(..., Fix128::ONE)` -- a tolerance of 1.0 MPa, which is a
/// smoke test, not a closed-form oracle. This checks the same claim at
/// `FIX_TOL` against the independent algebraic reduction.
#[test]
fn q_bar_zero_degrees_equals_q_matrix_exactly() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11_ref, q12_ref, q22_ref, q66_ref) = q_ref_f64(e1, e2, nu12, g12);
    let p = ply_b(Fix128::ZERO, Fix128::from_ratio(1, 4));
    let (b11, b12, b22, b16, b26, b66) = p.q_bar();
    assert_rel(b11, q11_ref, FIX_TOL, "Qbar11(0deg) = Q11");
    assert_rel(b12, q12_ref, FIX_TOL, "Qbar12(0deg) = Q12");
    assert_rel(b22, q22_ref, FIX_TOL, "Qbar22(0deg) = Q22");
    assert_rel(b66, q66_ref, FIX_TOL, "Qbar66(0deg) = Q66");
    assert_abs(b16, 0.0, FIX_TOL, "Qbar16(0deg) = 0");
    assert_abs(b26, 0.0, FIX_TOL, "Qbar26(0deg) = 0");
}

/// At theta = pi/2, `c = 0, s = 1` collapse Jones eq. 2.84 to an exact
/// axis swap: `Qbar11 = Q22`, `Qbar22 = Q11`, `Qbar12 = Q12`,
/// `Qbar66 = Q66`, off-axis terms exactly zero. `material_b` has
/// `E1 != E2` so `Q11 != Q22` and this is a non-trivial swap (unlike
/// `src/laminate.rs`'s own `#[cfg(test)]` module, which independently
/// forces `E1 != E2` for the same reason).
#[test]
fn q_bar_ninety_degrees_swaps_q11_and_q22_exactly() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11_ref, q12_ref, q22_ref, q66_ref) = q_ref_f64(e1, e2, nu12, g12);
    assert!(
        (q11_ref - q22_ref).abs() > 1.0,
        "material_b must have Q11 != Q22 for this swap to be non-trivial"
    );
    let p = ply_b(Fix128::HALF_PI, Fix128::from_ratio(1, 4));
    let (b11, b12, b22, b16, b26, b66) = p.q_bar();
    assert_rel(b11, q22_ref, FIX_TOL, "Qbar11(90deg) = Q22");
    assert_rel(b22, q11_ref, FIX_TOL, "Qbar22(90deg) = Q11");
    assert_rel(b12, q12_ref, FIX_TOL, "Qbar12(90deg) = Q12");
    assert_rel(b66, q66_ref, FIX_TOL, "Qbar66(90deg) = Q66");
    assert_abs(b16, 0.0, FIX_TOL, "Qbar16(90deg) = 0");
    assert_abs(b26, 0.0, FIX_TOL, "Qbar26(90deg) = 0");
}

/// A genuinely generic rotation (20 deg -- not 0/45/90, which all reduce
/// to plain algebra), independent reference via `f64::sin`/`f64::cos`
/// (see the file-level `#![allow(clippy::disallowed_methods)]`) feeding
/// the same Jones eq. 2.84 formula written out again here.
#[test]
fn q_bar_general_twenty_degree_angle_matches_jones_rotation_formula() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11, q12, q22, q66) = q_ref_f64(e1, e2, nu12, g12);
    let theta = 20.0_f64.to_radians();
    let (qb11_ref, qb12_ref, qb22_ref, qb16_ref, qb26_ref, qb66_ref) =
        q_bar_ref_f64(q11, q12, q22, q66, theta.cos(), theta.sin());

    let p = ply_b(Fix128::from_f64(theta), Fix128::from_ratio(1, 4));
    let (b11, b12, b22, b16, b26, b66) = p.q_bar();
    assert_rel(b11, qb11_ref, FIX_TOL, "Qbar11(20deg)");
    assert_rel(b12, qb12_ref, FIX_TOL, "Qbar12(20deg)");
    assert_rel(b22, qb22_ref, FIX_TOL, "Qbar22(20deg)");
    assert_rel(b16, qb16_ref, FIX_TOL, "Qbar16(20deg)");
    assert_rel(b26, qb26_ref, FIX_TOL, "Qbar26(20deg)");
    assert_rel(b66, qb66_ref, FIX_TOL, "Qbar66(20deg)");
}

/// Degenerate: a material already collapsed to all-zero `Q` by
/// `q_matrix`'s zero-modulus guard must propagate zero through `q_bar`
/// regardless of the rotation angle (the rotation formula multiplies
/// every term by some combination of `Q11, Q12, Q22, Q66`, all zero, so
/// every output term is zero times a finite trig factor).
#[test]
fn q_bar_zero_modulus_material_propagates_zero_regardless_of_angle() {
    let mat = OrthotropicElasticity {
        e_l_mpa: Fix128::ZERO,
        e_t_mpa: Fix128::from_int(5_000),
        e_z_mpa: Fix128::from_int(5_000),
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        g_lt_mpa: Fix128::from_int(3_000),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    };
    // An arbitrary non-special angle (37 deg); the claim holds for any
    // angle because every output term has a Q-factor of zero.
    let p = Ply {
        thickness_mm: Fix128::from_ratio(1, 4),
        orientation_rad: Fix128::from_f64(37.0_f64.to_radians()),
        material: mat,
    };
    assert_eq!(
        p.q_bar(),
        (
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO,
            Fix128::ZERO
        )
    );
}

// ============================================================================
// Section 3: compute_abd -- a single ply, a zero-thickness ply spliced
// into a stack, and extreme-magnitude thickness wrapping the D term.
// ============================================================================

/// Degenerate: a single ply is the homogeneous-plate closed form
/// (`Qbar == Q` at 0 deg) with `B` exactly zero regardless of thickness
/// (`z_upper^2 == z_lower^2` for any symmetric-about-midplane single
/// ply): `A = Q*t`, `B = 0`, `D = Q*t^3/12`.
#[test]
fn compute_abd_single_ply_matches_homogeneous_plate_formula() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11, q12, q22, q66) = q_ref_f64(e1, e2, nu12, g12);
    let t = 0.4_f64;
    let plies = [ply_b(Fix128::ZERO, Fix128::from_ratio(2, 5))];
    let abd = compute_abd(&plies);
    assert_rel(abd.a.m11, q11 * t, FIX_TOL, "A11 = Q11*t");
    assert_rel(abd.a.m22, q22 * t, FIX_TOL, "A22 = Q22*t");
    assert_rel(abd.a.m12, q12 * t, FIX_TOL, "A12 = Q12*t");
    assert_rel(abd.a.m33, q66 * t, FIX_TOL, "A66 = Q66*t");
    assert_eq!(abd.b, Sym3::default(), "B = 0 exactly for a single ply");
    let t3_over_12 = t * t * t / 12.0;
    assert_rel(abd.d.m11, q11 * t3_over_12, FIX_TOL, "D11 = Q11*t^3/12");
    assert_rel(abd.d.m22, q22 * t3_over_12, FIX_TOL, "D22 = Q22*t^3/12");
    assert_rel(abd.d.m33, q66 * t3_over_12, FIX_TOL, "D66 = Q66*t^3/12");
}

/// Invariant oracle (not a closed form against an external reference;
/// an invariant guard is a distinct, legitimate kind of oracle): splicing
/// a `thickness_mm = Fix128::ZERO` ply anywhere into a stack must leave
/// `compute_abd`'s result bit-for-bit unchanged, because `z_upper ==
/// z_lower` for that ply makes `delta_z = delta_z2 = delta_z3 =
/// Fix128::ZERO` identically -- and `Sym3::scale(ZERO)` is exactly
/// `Sym3::default()` for *any* `Q`, including one built from an extreme
/// or degenerate material, so this holds regardless of what the
/// zero-thickness ply's own material is.
#[test]
fn compute_abd_zero_thickness_ply_does_not_perturb_the_result() {
    let without_extra_ply = [
        ply_b(Fix128::ZERO, Fix128::from_ratio(1, 4)),
        ply_b(Fix128::HALF_PI, Fix128::from_ratio(1, 4)),
    ];
    let with_zero_thickness_ply_spliced_in = [
        ply_b(Fix128::ZERO, Fix128::from_ratio(1, 4)),
        // Spliced in with an arbitrary non-zero angle and an arbitrary
        // (extreme) material -- neither should matter at zero thickness.
        Ply {
            thickness_mm: Fix128::ZERO,
            orientation_rad: Fix128::from_f64(12.0_f64.to_radians()),
            material: OrthotropicElasticity {
                e_l_mpa: Fix128::from_int(i64::MAX),
                e_t_mpa: Fix128::from_int(i64::MAX),
                e_z_mpa: Fix128::from_int(i64::MAX),
                nu_lt: Fix128::from_ratio(1, 4),
                nu_lz: Fix128::ZERO,
                nu_tz: Fix128::ZERO,
                g_lt_mpa: Fix128::from_int(i64::MAX),
                g_lz_mpa: Fix128::ZERO,
                g_tz_mpa: Fix128::ZERO,
            },
        },
        ply_b(Fix128::HALF_PI, Fix128::from_ratio(1, 4)),
    ];
    let abd_without = compute_abd(&without_extra_ply);
    let abd_with = compute_abd(&with_zero_thickness_ply_spliced_in);
    assert_eq!(
        abd_without, abd_with,
        "a zero-thickness ply must not perturb A/B/D, regardless of its own material"
    );
}

/// Extreme Fix128 magnitude: a single ply thick enough that `z_upper^3`
/// wraps past `Fix128`'s `+2^63` boundary. With `thickness_mm =
/// Fix128::from_int(1i64 << 22)`, `z_upper = Fix128::from_int(1i64 <<
/// 21)` exactly (half-thickness, `half()` is an exact bit shift) and, for
/// `Fix128`s with `lo == 0`, multiplication reduces to plain `i64`
/// `wrapping_mul` on `hi` (verified against the live computation below,
/// not assumed): `z_upper^2 = Fix128::from_int(1i64 << 42)` (no wrap, `2^42
/// < 2^63`), then `z_upper^3 = (1i64<<42).wrapping_mul(1i64<<21)` which is
/// mathematically `2^63` -- one past `i64::MAX` -- and wraps to exactly
/// `i64::MIN`. `z_lower = -z_upper` by symmetry, and `z_lower^3 =
/// -(2^63)` mathematically, which *is* exactly representable as
/// `i64::MIN` (no wrap on that side). Both therefore land on the same
/// `Fix128` bit pattern, so `delta_z3 = z_upper^3 - z_lower^3` is exactly
/// `Fix128::ZERO` -- not a large or small nonzero error, exactly zero --
/// which makes every ply's `D` contribution `Sym3::default()` regardless
/// of its `Q`. `A` (the linear `delta_z` term) and `B` (the quadratic
/// `delta_z2` term, which cancels by symmetry regardless of scale) are
/// unaffected at this magnitude: this wrap is specific to the cubic `D`
/// integration.
#[test]
fn compute_abd_extreme_thickness_wraps_the_cubic_d_term_to_zero() {
    let extreme_t = Fix128::from_int(1i64 << 22);
    let half = extreme_t.half();
    assert_eq!(half, Fix128::from_int(1i64 << 21), "half() is an exact >>1");
    let z_lower = Fix128::ZERO - half;
    let z_upper = z_lower + extreme_t;
    assert_eq!(z_upper, Fix128::from_int(1i64 << 21));
    assert_eq!(z_lower, Fix128::from_int(-(1i64 << 21)));

    let z_upper_cubed = z_upper * z_upper * z_upper;
    let z_lower_cubed = z_lower * z_lower * z_lower;
    assert_eq!(
        z_upper_cubed, z_lower_cubed,
        "+2^63 wraps to the same bit pattern as the exactly-representable -2^63"
    );
    assert_eq!(z_upper_cubed.hi, i64::MIN);

    let plies = [ply_b(Fix128::ZERO, extreme_t)];
    let abd = compute_abd(&plies);
    assert_eq!(
        abd.d,
        Sym3::default(),
        "D collapses to exactly zero when the cubic integration term wraps"
    );
    // A is unaffected: delta_z = 2^22, well within range, so the linear
    // term is the ordinary (non-wrapped) Q*h value.
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11, _, _, _) = q_ref_f64(e1, e2, nu12, g12);
    let h = (1i64 << 22) as f64;
    assert_rel(abd.a.m11, q11 * h, FIX_TOL, "A11 = Q11*h (not wrapped)");
}

/// An unsymmetric 2-ply `[+20/-20]` stack, independent of
/// `tests/engineering_oracles_solid.rs`'s `[+45/-45]` case (different
/// angle, different material). With ply 0 (bottom, `z in [-t, 0]`) at
/// `+20 deg` and ply 1 (top, `z in [0, t]`) at `-20 deg`,
/// `B = 1/2 * Sum Qbar_k*(z_upper_k^2 - z_lower_k^2)` reduces to
/// `(t^2/2)*(Qbar(-20) - Qbar(+20))`. `Qbar11/12/22/66` are even in the
/// rotation angle (their formula only involves `c^2`, `s^2`, `c^4`, `s^4`),
/// so those four components of `B` are exactly zero -- only `Qbar16` and
/// `Qbar26` are odd in the angle (`c^3*s` and `c*s^3` both flip sign under
/// `theta -> -theta`), giving `B16 = -t^2 * Qbar16(+20)` and
/// `B26 = -t^2 * Qbar26(+20)`. This is the oracle that catches a mutant
/// that drops `compute_abd`'s `B` accumulation line entirely (such a
/// mutant leaves every `B` at `Sym3::default()`, which this stack's B16/
/// B26 are not).
#[test]
fn compute_abd_unsymmetric_two_ply_stack_matches_hand_derived_b16() {
    let (e1, e2, nu12, g12) = (20_000.0_f64, 5_000.0_f64, 0.25_f64, 3_000.0_f64);
    let (q11, q12, q22, q66) = q_ref_f64(e1, e2, nu12, g12);
    let theta = 20.0_f64.to_radians();
    let (_, _, _, qb16_ref, qb26_ref, _) =
        q_bar_ref_f64(q11, q12, q22, q66, theta.cos(), theta.sin());
    let t = 0.3_f64;
    let t_fix = Fix128::from_ratio(3, 10);
    let stack = [
        ply_b(Fix128::from_f64(theta), t_fix),
        ply_b(Fix128::from_f64(-theta), t_fix),
    ];
    assert!(
        !is_symmetric_stack(&stack),
        "[+20/-20] differs in orientation between its two plies"
    );
    let abd = compute_abd(&stack);
    assert_abs(abd.b.m11, 0.0, FIX_TOL, "B11 = 0 (Qbar11 is even in theta)");
    assert_abs(abd.b.m22, 0.0, FIX_TOL, "B22 = 0 (Qbar22 is even in theta)");
    assert_abs(abd.b.m12, 0.0, FIX_TOL, "B12 = 0 (Qbar12 is even in theta)");
    assert_abs(abd.b.m33, 0.0, FIX_TOL, "B66 = 0 (Qbar66 is even in theta)");
    assert_rel(
        abd.b.m13,
        -t * t * qb16_ref,
        FIX_TOL,
        "B16 = -t^2*Qbar16(+20)",
    );
    assert_rel(
        abd.b.m23,
        -t * t * qb26_ref,
        FIX_TOL,
        "B26 = -t^2*Qbar26(+20)",
    );
}

// ============================================================================
// Section 4: is_symmetric_stack -- empty slice, odd-length palindrome,
// material-only asymmetry.
// ============================================================================

/// Degenerate: the empty slice. `src/laminate.rs`'s own
/// `#[cfg(test)]::single_ply_stack_is_symmetric` covers `n == 1`; `n == 0`
/// is untested anywhere. By the function's own documented branch
/// (`n < 2` returns `true`), the empty laminate is trivially symmetric.
#[test]
fn is_symmetric_stack_empty_slice_is_trivially_symmetric() {
    let empty: [Ply; 0] = [];
    assert!(is_symmetric_stack(&empty));
}

/// An odd-length palindrome: `[0, 90, 0]`. The loop only pairs indices
/// `0..n/2` against their mirror, so the middle ply (index 1) is never
/// compared against anything -- a mutation that iterated `0..=n/2`
/// instead (comparing the middle ply against itself) would still pass
/// this particular case, so this test alone does not pin that off-by-one;
/// it does pin that an odd-length stack can report symmetric at all.
#[test]
fn is_symmetric_stack_odd_length_palindrome_with_unpaired_middle_ply() {
    let t = Fix128::from_ratio(1, 4);
    let stack = [
        ply_b(Fix128::ZERO, t),
        ply_b(Fix128::HALF_PI, t),
        ply_b(Fix128::ZERO, t),
    ];
    assert!(is_symmetric_stack(&stack));
}

/// Mirrored plies with matching thickness and orientation but different
/// material must report asymmetric: `Ply` derives `PartialEq` structurally
/// (it does not special-case which field differs), so a stack that is a
/// geometric palindrome but not a material palindrome must still fail.
#[test]
fn is_symmetric_stack_material_only_difference_is_asymmetric() {
    let t = Fix128::from_ratio(1, 4);
    let material_c = OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(20_000),
        e_t_mpa: Fix128::from_int(5_000),
        e_z_mpa: Fix128::from_int(5_000),
        nu_lt: Fix128::from_ratio(1, 4),
        nu_lz: Fix128::ZERO,
        nu_tz: Fix128::ZERO,
        // Only g_lt_mpa differs from material_b.
        g_lt_mpa: Fix128::from_int(3_001),
        g_lz_mpa: Fix128::ZERO,
        g_tz_mpa: Fix128::ZERO,
    };
    let stack = [
        ply_b(Fix128::ZERO, t),
        Ply {
            thickness_mm: t,
            orientation_rad: Fix128::ZERO,
            material: material_c,
        },
    ];
    assert!(!is_symmetric_stack(&stack));
}

// ============================================================================
// Section 5: AbdMatrix::is_symmetric -- the tolerance boundary (inclusive
// `<=`), each of the six B components individually, and extreme magnitude.
// ============================================================================

fn abd_with_b(b: Sym3) -> AbdMatrix {
    AbdMatrix {
        a: Sym3::default(),
        b,
        d: Sym3::default(),
    }
}

/// `|B_ij| <= tol` is inclusive: exactly at the tolerance must report
/// symmetric, one `Fix128` fractional unit (`lo = 1`) past it must not.
#[test]
fn is_symmetric_tolerance_boundary_is_inclusive() {
    let tol = Fix128::from_ratio(1, 1_000);
    let at_boundary = abd_with_b(Sym3 {
        m11: tol,
        ..Sym3::default()
    });
    assert!(at_boundary.is_symmetric(tol), "|B11| == tol must be <=");

    let just_past = abd_with_b(Sym3 {
        m11: Fix128 {
            hi: tol.hi,
            lo: tol.lo + 1,
        },
        ..Sym3::default()
    });
    assert!(
        !just_past.is_symmetric(tol),
        "|B11| == tol + one Fix128 ULP must be >"
    );
}

/// Each of the six `B` components is checked individually (not just
/// `m11`): a mutation that dropped, say, the `m23` comparison would pass
/// every other case here but fail this one.
#[test]
fn is_symmetric_checks_every_b_component_individually() {
    let tol = Fix128::from_ratio(1, 1_000);
    let past = Fix128::from_ratio(1, 500); // 2*tol
    let components: [fn(Fix128) -> Sym3; 6] = [
        |v| Sym3 {
            m11: v,
            ..Sym3::default()
        },
        |v| Sym3 {
            m12: v,
            ..Sym3::default()
        },
        |v| Sym3 {
            m22: v,
            ..Sym3::default()
        },
        |v| Sym3 {
            m13: v,
            ..Sym3::default()
        },
        |v| Sym3 {
            m23: v,
            ..Sym3::default()
        },
        |v| Sym3 {
            m33: v,
            ..Sym3::default()
        },
    ];
    for (i, make) in components.iter().enumerate() {
        let abd = abd_with_b(make(past));
        assert!(
            !abd.is_symmetric(tol),
            "component index {i} alone past tol must report asymmetric"
        );
    }
}

/// `tol == Fix128::ZERO`: only an exactly-zero `B` matrix is symmetric.
/// A single `Fix128::ONE` entry must fail.
#[test]
fn is_symmetric_zero_tolerance_requires_exactly_zero_b() {
    assert!(abd_with_b(Sym3::default()).is_symmetric(Fix128::ZERO));
    let nonzero = abd_with_b(Sym3 {
        m11: Fix128::ONE,
        ..Sym3::default()
    });
    assert!(!nonzero.is_symmetric(Fix128::ZERO));
}

/// Extreme Fix128 magnitude: `B11 == tol == Fix128::from_int(i64::MAX)`,
/// at the edge of the representable range. `Fix128`'s `abs()` and
/// ordering compare `(hi, lo)` lexicographically rather than doing
/// arithmetic that could itself overflow, so this boundary case is exact
/// at the extreme just as it was at the ordinary tolerance above.
#[test]
fn is_symmetric_extreme_magnitude_boundary_is_exact() {
    let extreme_tol = Fix128::from_int(i64::MAX);
    let at_boundary = abd_with_b(Sym3 {
        m11: extreme_tol,
        ..Sym3::default()
    });
    assert!(at_boundary.is_symmetric(extreme_tol));

    // One step past i64::MAX overflows what from_int(i64) can even
    // express directly, so instead push the *other* operand down: the
    // same B at a tolerance of zero must fail, confirming the extreme
    // value is not silently treated as "close enough to anything".
    assert!(!at_boundary.is_symmetric(Fix128::ZERO));
}
