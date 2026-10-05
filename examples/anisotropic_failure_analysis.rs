//! Anisotropic material off-axis modulus and failure-analysis production
//! entry point for `alice_physics::anisotropic`: `OrthotropicElasticity::
//! {e_at_angle_lt, e_at_angle_lz}`, `OrthotropicStress::axial`,
//! `evaluate_failure`, and its `FailureReport` output.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all five as `unwired`.
//! `grep -rn "anisotropic::" src/linear_elastic_fem.rs src/hyperelastic.rs`
//! is empty, and `grep -rln "use.*anisotropic\|anisotropic::" src/*.rs`
//! finds only `src/laminate.rs` (imports `OrthotropicElasticity` as a
//! struct literal for its own `Ply::material` field, never calls any of
//! these five) and `src/filament_db.rs` (references the module name in
//! doc comments only) -- the recently landed corotational-FEM
//! hyperelastic wiring (`4170a1c`/`33fe8a9`) did not reach this module.
//! This example is their first production caller.
//!
//! `tests/analytic_anisotropic_wiring.rs` holds additional closed-form
//! oracles that this file's two "real material" scenarios do not cover:
//! a second, independently chosen material at the same 0/45/90 degree
//! boundary angles plus a genuinely generic (20 degree) angle, the
//! zero-modulus guard interacting with the specific angle at which the
//! guarded term would otherwise be the *only* nonzero contribution,
//! `axial`'s behaviour at `Fix128`'s extreme representable magnitude, and
//! `evaluate_failure`'s epsilon-floor reserve-factor branch together with
//! the Hill and Tsai-Wu criteria (this file only exercises
//! Maximum-Stress).
//!
//! ```bash
//! cargo run --example anisotropic_failure_analysis --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::anisotropic::{
    evaluate_failure, AnisotropicStrength, FailureCriterion, FailureReport, OrthotropicElasticity,
    OrthotropicStress,
};
use alice_physics::math::Fix128;

/// Relative-error check against an independently hand-derived f64 closed
/// form. The expected value (`want`) is computed by a formula written
/// out again in plain f64 arithmetic below -- never by calling the
/// function under test.
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[anisotropic] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[anisotropic] ok {what}: got {g:.6}, want {want:.6} (err {err:.2e})");
}

/// A unidirectional carbon-fibre/epoxy lamina. The in-plane constants
/// (E_L, E_T, nu_LT, G_LT) are T300/5208 (Jones Table 2.3), the same
/// values `examples/laminate_abd_matrix.rs` uses for its CLT scenario.
/// The through-thickness set (E_Z, nu_LZ, G_LZ) is a separate,
/// independently chosen resin-dominated interlaminar estimate -- smaller
/// than E_T/G_LT and using a different Poisson ratio -- so the LT-plane
/// and LZ-plane transformations below are not numerically degenerate
/// with each other.
fn cfrp_lamina() -> OrthotropicElasticity {
    OrthotropicElasticity {
        e_l_mpa: Fix128::from_int(181_000),
        e_t_mpa: Fix128::from_int(10_300),
        e_z_mpa: Fix128::from_int(8_000),
        nu_lt: Fix128::from_ratio(28, 100),
        nu_lz: Fix128::from_ratio(30, 100),
        nu_tz: Fix128::from_ratio(40, 100),
        g_lt_mpa: Fix128::from_int(7_170),
        g_lz_mpa: Fix128::from_int(6_000),
        g_tz_mpa: Fix128::from_int(3_000),
    }
}

/// Strength envelope for the same lamina (round, illustrative MPa values
/// in the typical range for a UD carbon/epoxy tape: strong and stiff
/// along the fibre, matrix-dominated and much weaker transverse to it).
fn cfrp_strength() -> AnisotropicStrength {
    AnisotropicStrength {
        x_l_tension_mpa: Fix128::from_int(1_500),
        x_l_compression_mpa: Fix128::from_int(1_200),
        x_t_tension_mpa: Fix128::from_int(40),
        x_t_compression_mpa: Fix128::from_int(200),
        x_z_tension_mpa: Fix128::from_int(40),
        x_z_compression_mpa: Fix128::from_int(200),
        s_lt_mpa: Fix128::from_int(70),
        s_lz_mpa: Fix128::from_int(70),
        s_tz_mpa: Fix128::from_int(50),
    }
}

/// Jones eq. 2.85 off-axis modulus closed form, independent f64
/// reference: `1/E(theta) = c^4/Ea + s^4/Eb + c^2*s^2*(1/Gab - 2*nu_ab/Ea)`.
/// Takes `cos(theta)`/`sin(theta)` as plain numbers (rather than computing
/// them here) so this function never calls a trigonometric method itself
/// -- `clippy.toml`'s `disallowed-methods` bans `f64::sin`/`f64::cos`
/// crate-wide (determinism gate: every transcendental must go through
/// `alice_physics::det_math`, not the platform libm) and that ban applies
/// to `--all-targets`, including this example.
fn jones_e_ref(e_a: f64, e_b: f64, g_ab: f64, nu_ab: f64, c: f64, s: f64) -> f64 {
    let c2 = c * c;
    let s2 = s * s;
    let c4 = c2 * c2;
    let s4 = s2 * s2;
    let inv_e = c4 / e_a + s4 / e_b + c2 * s2 * (1.0 / g_ab - 2.0 * nu_ab / e_a);
    1.0 / inv_e
}

fn main() {
    let tol = 1e-9;
    let mat = cfrp_lamina();

    // ------------------------------------------------------------------
    // 1. e_at_angle_lt -- 0 deg (= E_L exactly), 90 deg (= E_T exactly),
    //    and 30 deg via Jones eq. 2.85 with the exact algebraic
    //    cos(30deg) = sqrt(3)/2, sin(30deg) = 1/2 (no trig method call).
    // ------------------------------------------------------------------
    let e_lt_0 = mat.e_at_angle_lt(Fix128::ZERO);
    assert_rel(e_lt_0, 181_000.0, tol, "e_at_angle_lt(0deg) = E_L");

    let e_lt_90 = mat.e_at_angle_lt(Fix128::HALF_PI);
    assert_rel(e_lt_90, 10_300.0, tol, "e_at_angle_lt(90deg) = E_T");

    let cos30 = 3.0_f64.sqrt() / 2.0;
    let sin30 = 0.5_f64;
    let e_lt_30_ref = jones_e_ref(181_000.0, 10_300.0, 7_170.0, 0.28, cos30, sin30);
    let theta30 = Fix128::PI / Fix128::from_int(6);
    let e_lt_30 = mat.e_at_angle_lt(theta30);
    assert_rel(
        e_lt_30,
        e_lt_30_ref,
        tol,
        "e_at_angle_lt(30deg) Jones eq. 2.85",
    );

    // ------------------------------------------------------------------
    // 2. e_at_angle_lz -- 0 deg (= E_L exactly), 90 deg (= E_Z exactly),
    //    and 45 deg via the same formula with (E_L, E_Z, G_LZ, nu_LZ),
    //    using the exact algebraic cos(45deg) = sin(45deg) = sqrt(2)/2.
    // ------------------------------------------------------------------
    let e_lz_0 = mat.e_at_angle_lz(Fix128::ZERO);
    assert_rel(e_lz_0, 181_000.0, tol, "e_at_angle_lz(0deg) = E_L");

    let e_lz_90 = mat.e_at_angle_lz(Fix128::HALF_PI);
    assert_rel(e_lz_90, 8_000.0, tol, "e_at_angle_lz(90deg) = E_Z");

    let cos45 = 2.0_f64.sqrt() / 2.0;
    let sin45 = cos45;
    let e_lz_45_ref = jones_e_ref(181_000.0, 8_000.0, 6_000.0, 0.30, cos45, sin45);
    let theta45 = Fix128::HALF_PI.half();
    let e_lz_45 = mat.e_at_angle_lz(theta45);
    assert_rel(
        e_lz_45,
        e_lz_45_ref,
        tol,
        "e_at_angle_lz(45deg) Jones eq. 2.85",
    );

    // ------------------------------------------------------------------
    // 3. axial + evaluate_failure (Maximum-Stress) -- a sub-threshold
    //    (safe) load: sigma_L = 300 MPa, sigma_T = 10 MPa, sigma_Z = 5 MPa,
    //    all tension, no shear. Ratios against the tensile allowables are
    //    300/1500 = 0.2, 10/40 = 0.25, 5/40 = 0.125 -- the worst is the T
    //    axis at 0.25, so failure_index = 0.25 and reserve_factor = 4.
    // ------------------------------------------------------------------
    let strength = cfrp_strength();
    let safe_stress = OrthotropicStress::axial(
        Fix128::from_int(300),
        Fix128::from_int(10),
        Fix128::from_int(5),
    );
    let safe_report: FailureReport =
        evaluate_failure(&safe_stress, &strength, FailureCriterion::MaximumStress);
    assert_rel(
        safe_report.failure_index,
        0.25,
        tol,
        "safe failure_index = max(300/1500, 10/40, 5/40) = 0.25",
    );
    assert_rel(
        safe_report.reserve_factor,
        4.0,
        tol,
        "safe reserve_factor = 1/0.25",
    );
    assert!(
        safe_report.is_safe,
        "[anisotropic] MISMATCH: failure_index 0.25 < 1 must report safe"
    );
    println!(
        "[anisotropic] ok FailureReport (sub-threshold): index={:.6} reserve={:.6} is_safe={}",
        safe_report.failure_index.to_f64(),
        safe_report.reserve_factor.to_f64(),
        safe_report.is_safe
    );

    // ------------------------------------------------------------------
    // 4. Over-threshold (failing) load: sigma_L = 100 MPa, sigma_T =
    //    50 MPa, sigma_Z = 20 MPa, all tension. Ratios are 100/1500 =
    //    0.0667, 50/40 = 1.25, 20/40 = 0.5 -- the T axis now exceeds its
    //    allowable, so failure_index = 1.25 and reserve_factor = 0.8.
    // ------------------------------------------------------------------
    let failing_stress = OrthotropicStress::axial(
        Fix128::from_int(100),
        Fix128::from_int(50),
        Fix128::from_int(20),
    );
    let failing_report: FailureReport =
        evaluate_failure(&failing_stress, &strength, FailureCriterion::MaximumStress);
    assert_rel(
        failing_report.failure_index,
        1.25,
        tol,
        "failing failure_index = max(100/1500, 50/40, 20/40) = 1.25",
    );
    assert_rel(
        failing_report.reserve_factor,
        0.8,
        tol,
        "failing reserve_factor = 1/1.25",
    );
    assert!(
        !failing_report.is_safe,
        "[anisotropic] MISMATCH: failure_index 1.25 > 1 must report failing"
    );
    println!(
        "[anisotropic] ok FailureReport (over-threshold): index={:.6} reserve={:.6} is_safe={}",
        failing_report.failure_index.to_f64(),
        failing_report.reserve_factor.to_f64(),
        failing_report.is_safe
    );

    println!(
        "[anisotropic] all 5 production entry points (e_at_angle_lt, e_at_angle_lz, axial, \
         evaluate_failure, FailureReport) verified against hand-derived closed forms"
    );
}
