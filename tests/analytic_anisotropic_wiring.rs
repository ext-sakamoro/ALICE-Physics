//! Oracles for the production entry points of `alice_physics::anisotropic`
//! driven by `examples/anisotropic_failure_analysis.rs`:
//! `OrthotropicElasticity::{e_at_angle_lt, e_at_angle_lz}`,
//! `OrthotropicStress::axial`, `evaluate_failure`, and its `FailureReport`.
//!
//! # What this file is and is not
//!
//! `src/anisotropic.rs`'s own `#[cfg(test)]` module already carries an
//! extensive set of mutation-killing closed-form tests for this module
//! (boundary angles, the zero-modulus guards, the Hill/Tsai-Wu dyadic
//! closed forms, the `is_safe` boundary at `failure_index == 1`, and the
//! sign-dependent tension/compression allowable selection). This file
//! does not repeat those. What it adds:
//!
//! * `e_at_angle_lt`/`e_at_angle_lz` boundary angles (0/45/90 deg) for a
//!   *second* material, distinct from both `src/anisotropic.rs`'s own
//!   `distinct_ortho()` fixture and `examples/anisotropic_failure_analysis.rs`'s
//!   T300/5208-derived lamina, plus a genuinely generic (20 deg) angle
//!   for each, driven against `f64::sin`/`f64::cos` rather than an exact
//!   algebraic special-case value,
//! * the zero-modulus guard at the one angle where the guarded term is
//!   the *only* nonzero contribution (theta = 0 for `e_at_angle_lt`'s
//!   `e_l_mpa`, theta = 90 deg for `e_at_angle_lz`'s `e_z_mpa`) -- without
//!   the guard, the formula would evaluate the literal division
//!   `Fix128::ONE / Fix128::ZERO` rather than merely attenuating a
//!   nonzero sum, unlike `src/anisotropic.rs`'s own guard test (and this
//!   file's boundary-angle tests above) which all use theta = 45 deg,
//!   where every one of `c^4`, `s^4`, `c^2*s^2` is simultaneously nonzero,
//! * `axial` at `Fix128`'s extreme representable magnitude (`i64::MAX` /
//!   `i64::MIN`), confirming the shear fields are zeroed and the normal
//!   fields are preserved bit-exact at that scale, not just at the small
//!   integer magnitudes `src/anisotropic.rs`'s own
//!   `axial_constructor_zeros_shear` test uses,
//! * `evaluate_failure`'s epsilon-floor reserve-factor branch
//!   (`let denom = if idx > eps { idx } else { eps }`, `eps = 1/1_000_000`)
//!   -- every existing case in `src/anisotropic.rs` has `failure_index`
//!   either exactly zero or comfortably above `eps`, so the floor's
//!   *value* (as opposed to the zero-stress case, which never checks
//!   `reserve_factor` at all) is never pinned anywhere, and
//! * `FailureReport.is_safe`/`reserve_factor` for the Hill and Tsai-Wu
//!   criteria through `evaluate_failure`: `src/anisotropic.rs`'s own
//!   `hill_index_dyadic_closed_form` only exercises a *failing* Hill case,
//!   and `tsai_wu_index_dyadic_closed_form` /
//!   `tsai_wu_unit_at_uniaxial_strengths` only ever check `failure_index`
//!   for Tsai-Wu (never `is_safe` or `reserve_factor`, and never a
//!   *failing* Tsai-Wu case at all) -- this file adds a safe Hill case and
//!   a failing Tsai-Wu case, both with every `FailureReport` field pinned.
//!
//! # Degenerate / extreme input summary
//!
//! * `e_at_angle_lt`/`e_at_angle_lz`: a single zeroed modulus at the
//!   angle where it is the sole denominator in play (not just "some
//!   angle", as `src/anisotropic.rs`'s own guard test uses).
//! * `axial`: `Fix128::from_int(i64::MAX)` / `Fix128::from_int(i64::MIN)`
//!   as normal-stress components, confirming no overflow/wrap in the
//!   trivial field-copying constructor itself.
//! * `evaluate_failure`: a `failure_index` strictly below `eps =
//!   1/1_000_000` (the reserve-factor floor engages) and, separately, the
//!   Hill/Tsai-Wu `FailureReport` field combinations noted above.
//!
//! Author: Moroya Sakamoto

// `e_at_angle_lt_generic_twenty_degree_angle_second_material` and its
// `e_at_angle_lz` counterpart need a genuinely generic rotation angle
// (not the 0/45/90 deg special cases that reduce to plain algebra), which
// means calling `f64::sin`/`f64::cos` for the independent oracle side.
// `clippy.toml`'s `disallowed-methods` bans those crate-wide (every
// transcendental must go through `alice_physics::det_math`, not the
// platform libm) but the ban does not apply to a closed-form f64
// reference value that is never written back into `Fix128` production
// state -- same convention as `tests/analytic_laminate_wiring.rs` and
// `tests/engineering_oracles_solid.rs`.
#![allow(clippy::disallowed_methods)]

use alice_physics::anisotropic::{
    evaluate_failure, AnisotropicStrength, FailureCriterion, FailureReport, OrthotropicElasticity,
    OrthotropicStress,
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

const FIX_TOL: f64 = 1e-9;

/// A second orthotropic material, distinct from `src/anisotropic.rs`'s
/// own `distinct_ortho()` (E_L=200, E_T=100, E_Z=400, nu_LT=1/4,
/// nu_LZ=1/8, G_LT=50, G_LZ=25) and from
/// `examples/anisotropic_failure_analysis.rs`'s T300/5208-derived lamina:
/// E_L = 400, E_T = 200, E_Z = 100, nu_LT = 1/5, nu_LZ = 1/10,
/// G_LT = 80, G_LZ = 40 (MPa).
fn second_material() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(400),
        e_t_mpa: Fix128::from_int(200),
        e_z_mpa: Fix128::from_int(100),
        nu_lt: Fix128::from_ratio(1, 5),
        nu_lz: Fix128::from_ratio(1, 10),
        nu_tz: Fix128::from_ratio(3, 10),
        g_lt_mpa: Fix128::from_int(80),
        g_lz_mpa: Fix128::from_int(40),
        g_tz_mpa: Fix128::from_int(30),
    }
}

/// Jones eq. 2.85 off-axis modulus closed form, independent f64
/// reference (same formula as
/// `examples/anisotropic_failure_analysis.rs`'s `jones_e_ref`, duplicated
/// here because test binaries cannot import helpers from an example
/// binary). `1/E(theta) = c^4/Ea + s^4/Eb + c^2*s^2*(1/Gab - 2*nu_ab/Ea)`.
fn jones_e_ref(e_a: f64, e_b: f64, g_ab: f64, nu_ab: f64, c: f64, s: f64) -> f64 {
    let c2 = c * c;
    let s2 = s * s;
    let c4 = c2 * c2;
    let s4 = s2 * s2;
    let inv_e = c4 / e_a + s4 / e_b + c2 * s2 * (1.0 / g_ab - 2.0 * nu_ab / e_a);
    1.0 / inv_e
}

// ============================================================================
// Section 1: e_at_angle_lt / e_at_angle_lz -- second material at 0/45/90
// deg boundaries, a generic 20 deg angle, and the zero-modulus guard at
// the angle where the guarded term is the sole contribution.
// ============================================================================

#[test]
fn e_at_angle_lt_boundary_zero_and_ninety_second_material() {
    let o = second_material();
    assert_rel(
        o.e_at_angle_lt(Fix128::ZERO),
        400.0,
        FIX_TOL,
        "theta=0 -> E_L",
    );
    assert_rel(
        o.e_at_angle_lt(Fix128::HALF_PI),
        200.0,
        FIX_TOL,
        "theta=90deg -> E_T",
    );
}

#[test]
fn e_at_angle_lt_45deg_second_material_exact_algebraic() {
    let o = second_material();
    let c = 2.0_f64.sqrt() / 2.0;
    let s = c;
    let want = jones_e_ref(400.0, 200.0, 80.0, 0.2, c, s);
    let got = o.e_at_angle_lt(Fix128::HALF_PI.half());
    assert_rel(got, want, FIX_TOL, "e_at_angle_lt(45deg) second material");
}

#[test]
fn e_at_angle_lt_generic_twenty_degree_angle_second_material() {
    let o = second_material();
    let theta = 20.0_f64.to_radians();
    let want = jones_e_ref(400.0, 200.0, 80.0, 0.2, theta.cos(), theta.sin());
    let got = o.e_at_angle_lt(Fix128::from_f64(theta));
    assert_rel(got, want, FIX_TOL, "e_at_angle_lt(20deg) second material");
}

#[test]
fn e_at_angle_lz_boundary_zero_and_ninety_second_material() {
    let o = second_material();
    assert_rel(
        o.e_at_angle_lz(Fix128::ZERO),
        400.0,
        FIX_TOL,
        "theta=0 -> E_L",
    );
    assert_rel(
        o.e_at_angle_lz(Fix128::HALF_PI),
        100.0,
        FIX_TOL,
        "theta=90deg -> E_Z",
    );
}

#[test]
fn e_at_angle_lz_45deg_second_material_exact_algebraic() {
    let o = second_material();
    let c = 2.0_f64.sqrt() / 2.0;
    let s = c;
    let want = jones_e_ref(400.0, 100.0, 40.0, 0.1, c, s);
    let got = o.e_at_angle_lz(Fix128::HALF_PI.half());
    assert_rel(got, want, FIX_TOL, "e_at_angle_lz(45deg) second material");
}

#[test]
fn e_at_angle_lz_generic_twenty_degree_angle_second_material() {
    let o = second_material();
    let theta = 20.0_f64.to_radians();
    let want = jones_e_ref(400.0, 100.0, 40.0, 0.1, theta.cos(), theta.sin());
    let got = o.e_at_angle_lz(Fix128::from_f64(theta));
    assert_rel(got, want, FIX_TOL, "e_at_angle_lz(20deg) second material");
}

/// Guard activation at the one angle where the zeroed modulus would
/// otherwise be the *sole* denominator exercised: theta = 0 makes
/// `c^4 = 1, s^4 = 0`, so without the `e_l_mpa.is_zero() || ...` guard the
/// formula collapses to the literal division `Fix128::ONE / Fix128::ZERO`
/// (not merely an attenuated nonzero sum, as at the 45 deg angle
/// `src/anisotropic.rs`'s own `e_at_angle_zero_modulus_guard_each_operand`
/// test uses).
#[test]
fn e_at_angle_lt_zero_longitudinal_modulus_at_theta_zero() {
    let o = OrthotropicElasticity {
        e_l_mpa: Fix128::ZERO,
        ..second_material()
    };
    assert_eq!(o.e_at_angle_lt(Fix128::ZERO), Fix128::ZERO);
}

/// Same reasoning for `e_at_angle_lz`'s `e_z_mpa.is_zero()` guard leg at
/// theta = 90 deg, where `c^4 = 0, s^4 = 1` makes `e_z_mpa` the sole
/// denominator in the `s^4/e_z_mpa` term.
#[test]
fn e_at_angle_lz_zero_through_thickness_modulus_at_theta_ninety() {
    let o = OrthotropicElasticity {
        e_z_mpa: Fix128::ZERO,
        ..second_material()
    };
    assert_eq!(o.e_at_angle_lz(Fix128::HALF_PI), Fix128::ZERO);
}

// ============================================================================
// Section 2: axial -- extreme Fix128 magnitude.
// ============================================================================

/// `axial` at `Fix128`'s extreme representable magnitude: the normal
/// fields must be preserved bit-exact and the shear fields zeroed, the
/// same contract `src/anisotropic.rs`'s own `axial_constructor_zeros_shear`
/// pins at small integer magnitudes (10/20/30), now at `i64::MAX`/`i64::MIN`.
#[test]
fn axial_preserves_extreme_magnitude_and_sign_exactly() {
    let s = OrthotropicStress::axial(
        Fix128::from_int(i64::MAX),
        Fix128::from_int(i64::MIN),
        Fix128::ZERO,
    );
    assert_eq!(s.sigma_l, Fix128::from_int(i64::MAX));
    assert_eq!(s.sigma_t, Fix128::from_int(i64::MIN));
    assert_eq!(s.sigma_z, Fix128::ZERO);
    assert_eq!(s.tau_lt, Fix128::ZERO);
    assert_eq!(s.tau_lz, Fix128::ZERO);
    assert_eq!(s.tau_tz, Fix128::ZERO);
}

// ============================================================================
// Section 3: evaluate_failure / FailureReport -- the epsilon-floor
// reserve-factor branch, a safe Hill case, and a failing Tsai-Wu case.
// ============================================================================

/// `evaluate_failure`'s reserve-factor denominator floor (`let denom = if
/// idx > eps { idx } else { eps }`, `eps = Fix128::from_ratio(1,
/// 1_000_000)`) is never exercised by `src/anisotropic.rs`'s own tests:
/// every existing case there has `failure_index` either exactly zero (the
/// floor applies, but `reserve_factor` is never checked in that case) or
/// comfortably above `eps`. This picks a `failure_index` strictly between
/// zero and `eps` so the two branches disagree: with the floor in place,
/// `reserve_factor` is 1/eps = 1_000_000 (approximately -- `eps` itself is
/// `Fix128::from_ratio(1, 1_000_000)`, which is not bit-exact because
/// `2^64` is not a multiple of `1_000_000`, so `1/eps` lands a few parts
/// in 1e8 away from the mathematical 1,000,000; the relative-error
/// assertion below allows for that). A mutant that always divided by
/// `idx` instead would give `1/idx = 5_000_000`, five times larger --
/// comfortably outside this test's tolerance.
#[test]
fn evaluate_failure_reserve_factor_floors_below_epsilon() {
    let huge = Fix128::from_int(5_000_000);
    let s = AnisotropicStrength {
        x_l_tension_mpa: huge,
        x_l_compression_mpa: huge,
        x_t_tension_mpa: huge,
        x_t_compression_mpa: huge,
        x_z_tension_mpa: huge,
        x_z_compression_mpa: huge,
        s_lt_mpa: huge,
        s_lz_mpa: huge,
        s_tz_mpa: huge,
    };
    // ratio = 1 / 5_000_000 = 2e-7, strictly below eps = 1e-6.
    let stress = OrthotropicStress::axial(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let r: FailureReport = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
    let eps = Fix128::from_ratio(1, 1_000_000);
    assert_eq!(
        r.failure_index,
        Fix128::from_ratio(1, 5_000_000),
        "failure_index = 1/5_000_000 exactly"
    );
    assert!(
        r.failure_index < eps,
        "failure_index must be strictly below eps for this test to be meaningful"
    );
    assert_rel(
        r.reserve_factor,
        1_000_000.0,
        1e-6,
        "reserve_factor floors at ~1/eps, not 1/idx (which would be 5_000_000)",
    );
    assert!(r.is_safe);
}

/// Hill criterion, sub-threshold (safe) case through `evaluate_failure`.
/// Same strength envelope as `src/anisotropic.rs`'s own
/// `hill_index_dyadic_closed_form` (X_L=2, X_T=4, X_Z=8, S_LT=2, S_LZ=4,
/// S_TZ=8, giving G = 13/128, H = 19/128 per that test's own derivation),
/// but a much smaller, pure-axial-L stress (sigma_L=1, everything else
/// zero) chosen to land below 1 -- that existing test only exercises a
/// failing (f = 55/8) case, so `is_safe`/`reserve_factor` for a safe Hill
/// result are never pinned anywhere. With sigma_T = sigma_Z = 0 and no
/// shear: d_tz = 0, d_zl = -1, d_lt = 1, so
/// `f = F*0 + G*(-1)^2 + H*1^2 = G + H = 13/128 + 19/128 = 32/128 = 1/4`.
/// Hill is homogeneous of degree 2 in the stresses, so the load factor
/// that reaches index 1 is `1/sqrt(f) = 2` (AUD-A-S1W6-003).
#[test]
fn hill_sub_threshold_failure_report_fields_exact() {
    let s = AnisotropicStrength {
        x_l_tension_mpa: Fix128::from_int(2),
        x_l_compression_mpa: Fix128::from_int(2),
        x_t_tension_mpa: Fix128::from_int(4),
        x_t_compression_mpa: Fix128::from_int(4),
        x_z_tension_mpa: Fix128::from_int(8),
        x_z_compression_mpa: Fix128::from_int(8),
        s_lt_mpa: Fix128::from_int(2),
        s_lz_mpa: Fix128::from_int(4),
        s_tz_mpa: Fix128::from_int(8),
    };
    let stress = OrthotropicStress::axial(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let r: FailureReport = evaluate_failure(&stress, &s, FailureCriterion::Hill);
    assert_eq!(
        r.failure_index,
        Fix128::from_ratio(1, 4),
        "hill index = G+H = 1/4"
    );
    assert_eq!(
        r.reserve_factor,
        Fix128::from_int(2),
        "reserve_factor = 1/sqrt(1/4) = 2"
    );
    assert!(r.is_safe, "failure_index 1/4 < 1 must report safe");
}

/// Tsai-Wu criterion, over-threshold (failing) case through
/// `evaluate_failure`. `src/anisotropic.rs`'s own
/// `tsai_wu_index_dyadic_closed_form` and `tsai_wu_unit_at_uniaxial_strengths`
/// tests only ever check `failure_index` for Tsai-Wu -- `is_safe` and
/// `reserve_factor` are never pinned for this criterion anywhere, and no
/// existing Tsai-Wu test uses a failing stress state at all (the former
/// is already > 1 but doesn't check `is_safe`; the latter is a boundary
/// case at exactly 1 via the raw `tsai_wu_index` function, not through
/// `evaluate_failure`).
///
/// With `sigma_L = 2 * X_Lt` (double the tensile strength along L) and
/// `sigma_T = sigma_Z = 0`, no shear, every cross term and every T/Z term
/// vanishes: `f = F_L*sigma_L + F_LL*sigma_L^2`.
/// `F_L = 1/X_Lt - 1/X_Lc = 1/4 - 1/16 = 3/16`,
/// `F_LL = 1/(X_Lt*X_Lc) = 1/64`, `sigma_L = 8`:
/// `f = (3/16)*8 + (1/64)*64 = 3/2 + 1 = 5/2`.
/// The reserve factor `R` solves `a R^2 + b R = 1` with the quadratic part
/// `a = 1` and the linear part `b = 3/2`: `R = 2/(b + sqrt(b^2 + 4a)) =
/// 2/(3/2 + 5/2) = 1/2`, i.e. `sigma_L = 4 = X_Lt` (AUD-A-S1W6-004).
#[test]
fn tsai_wu_over_threshold_failure_report_fields_exact() {
    let s = AnisotropicStrength {
        x_l_tension_mpa: Fix128::from_int(4),
        x_l_compression_mpa: Fix128::from_int(16),
        x_t_tension_mpa: Fix128::from_int(8),
        x_t_compression_mpa: Fix128::from_int(32),
        x_z_tension_mpa: Fix128::from_int(16),
        x_z_compression_mpa: Fix128::from_int(64),
        s_lt_mpa: Fix128::from_int(8),
        s_lz_mpa: Fix128::from_int(16),
        s_tz_mpa: Fix128::from_int(32),
    };
    let stress = OrthotropicStress::axial(Fix128::from_int(8), Fix128::ZERO, Fix128::ZERO);
    let r: FailureReport = evaluate_failure(&stress, &s, FailureCriterion::TsaiWu);
    assert_eq!(
        r.failure_index,
        Fix128::from_ratio(5, 2),
        "tsai-wu index = 5/2"
    );
    assert_eq!(
        r.reserve_factor,
        Fix128::from_ratio(1, 2),
        "reserve_factor = positive root of R^2 + (3/2) R - 1 = 0 = 1/2"
    );
    assert!(
        !r.is_safe,
        "failure_index 5/2 > 1 must report failing -- never checked for Tsai-Wu before this test"
    );
}

/// Maximum-Stress criterion, shear-only over-threshold case through
/// `evaluate_failure`. `src/anisotropic.rs`'s own `shear_stress_flags_failure`
/// exercises an in-plane shear overload but only asserts `!r.is_safe` --
/// `failure_index`/`reserve_factor` are never checked there. In-plane
/// shear allowable = 4, applied tau_lt = 6: `f = 6/4 = 3/2`.
#[test]
fn max_stress_shear_only_over_threshold_failure_report_fields_exact() {
    let big = Fix128::from_int(100);
    let s = AnisotropicStrength {
        x_l_tension_mpa: big,
        x_l_compression_mpa: big,
        x_t_tension_mpa: big,
        x_t_compression_mpa: big,
        x_z_tension_mpa: big,
        x_z_compression_mpa: big,
        s_lt_mpa: Fix128::from_int(4),
        s_lz_mpa: big,
        s_tz_mpa: big,
    };
    let stress = OrthotropicStress {
        sigma_l: Fix128::ZERO,
        sigma_t: Fix128::ZERO,
        sigma_z: Fix128::ZERO,
        tau_lt: Fix128::from_int(6),
        tau_lz: Fix128::ZERO,
        tau_tz: Fix128::ZERO,
    };
    let r: FailureReport = evaluate_failure(&stress, &s, FailureCriterion::MaximumStress);
    assert_eq!(r.failure_index, Fix128::from_ratio(3, 2), "6/4 = 3/2");
    assert_eq!(
        r.reserve_factor,
        Fix128::from_ratio(2, 3),
        "reserve_factor = 1/(3/2) = 2/3"
    );
    assert!(!r.is_safe);
}
