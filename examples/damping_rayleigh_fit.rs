//! Rayleigh (Proportional) Damping Model production entry point for
//! `alice_physics::damping_rayleigh`: `hz_to_omega`, `omega_to_hz`,
//! `RayleighCoefficients::fit_two_modes`, and
//! `RayleighCoefficients::damping_ratio`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `src/damping_rayleigh.rs`'s own `#[cfg(test)]` module exercises these
//! but tests do not count as production callers for the wiring guard,
//! and nothing in `src/` / `examples/` / `benches/` called any of the
//! four before this file existed. This example is that caller.
//!
//! Scenario: pick two target vibration modes (frequency in Hz + desired
//! modal damping ratio) and determine the Rayleigh coefficients
//! `(alpha, beta)` of `C = alpha*M + beta*K` such that both targets are
//! hit. Mode 1: 2 Hz at zeta=0.05 (a soft, low-frequency mode -- e.g. a
//! building's fundamental sway). Mode 2: 20 Hz at zeta=0.02 (a stiffer,
//! higher harmonic). This is the standard two-point Rayleigh damping fit
//! (Chopra, *Dynamics of Structures* 5th ed. Ch. 11.4; Clough & Penzien
//! Ch. 12).
//!
//! All expected numbers below are derived independently in plain f64 by
//! re-deriving the closed form, never by calling `hz_to_omega`,
//! `omega_to_hz`, `fit_two_modes`, or `damping_ratio` themselves:
//!
//! * `omega = 2*pi*f` for the Hz <-> rad/s conversion.
//! * `zeta_n = alpha/(2*omega_n) + beta*omega_n/2` at each mode, multiply
//!   through by `2*omega_n` to get `2*omega_n*zeta_n = alpha +
//!   beta*omega_n^2`, then subtract the two mode equations to eliminate
//!   `alpha` and solve for `beta`, then back-substitute for `alpha`.
//!
//! `hz_to_omega` / `omega_to_hz` operate on `Fix128` (the crate's
//! deterministic fixed-point type), not host `f64` -- they are ordinary
//! exact `Fix128` multiply/divide by the precomputed `Fix128::PI`
//! constant (no CORDIC transcendental call is involved), so the
//! crate-wide determinism discipline applies to them exactly as it does
//! to every other `Fix128` function in this file; they are not exempt as
//! a host-side utility.
//!
//! ```bash
//! cargo run --example damping_rayleigh_fit --features std
//! ```

use alice_physics::damping_rayleigh::{hz_to_omega, omega_to_hz, RayleighCoefficients};
use alice_physics::math::Fix128;
use core::f64::consts::PI;

/// Relative-error check against an independently hand-derived f64 closed
/// form (see module doc). The expected value (`want`) is computed by a
/// formula written out again in plain f64 arithmetic in `main` below --
/// never by calling the function under test.
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "[damping_rayleigh] MISMATCH {what}: got {g}, want {want} (err {err:.3e} > {tol:.1e})"
    );
    println!("[damping_rayleigh] ok {what}: got {g:.10}, want {want:.10} (err {err:.2e})");
}

fn main() {
    let tol = 1e-9;

    // ------------------------------------------------------------------
    // 1. hz_to_omega / omega_to_hz -- unit conversion for the two target
    //    modes. Independent reference: omega = 2*pi*f, using
    //    `core::f64::consts::PI` (a value this file never reads from the
    //    crate's `Fix128::PI`), not a self-consistency check against the
    //    implementation's own constant.
    // ------------------------------------------------------------------
    let f1_hz = 2.0_f64;
    let f2_hz = 20.0_f64;
    let omega1_ref = 2.0 * PI * f1_hz;
    let omega2_ref = 2.0 * PI * f2_hz;

    let omega1 = hz_to_omega(Fix128::from_int(2));
    let omega2 = hz_to_omega(Fix128::from_int(20));
    assert_rel(omega1, omega1_ref, tol, "hz_to_omega(2 Hz) = 2*pi*2");
    assert_rel(omega2, omega2_ref, tol, "hz_to_omega(20 Hz) = 2*pi*20");

    let f1_back = omega_to_hz(omega1);
    let f2_back = omega_to_hz(omega2);
    assert_rel(
        f1_back,
        f1_hz,
        tol,
        "omega_to_hz(hz_to_omega(2 Hz)) round-trips to 2 Hz",
    );
    assert_rel(
        f2_back,
        f2_hz,
        tol,
        "omega_to_hz(hz_to_omega(20 Hz)) round-trips to 20 Hz",
    );

    // ------------------------------------------------------------------
    // 2. fit_two_modes -- solve the 2x2 linear system for (alpha, beta)
    //    by hand (see module doc for the derivation): subtract the two
    //    per-mode equations to eliminate alpha and solve for beta, then
    //    back-substitute for alpha.
    // ------------------------------------------------------------------
    let zeta1 = 0.05_f64;
    let zeta2 = 0.02_f64;
    let beta_ref = (2.0 * (omega2_ref * zeta2 - omega1_ref * zeta1))
        / (omega2_ref * omega2_ref - omega1_ref * omega1_ref);
    let alpha_ref = 2.0 * omega1_ref * zeta1 - beta_ref * omega1_ref * omega1_ref;

    let coeffs = RayleighCoefficients::fit_two_modes(
        omega1,
        Fix128::from_ratio(5, 100),
        omega2,
        Fix128::from_ratio(2, 100),
    );
    assert_rel(
        coeffs.alpha,
        alpha_ref,
        tol,
        "fit_two_modes alpha (2x2 linear solve, back-substitution)",
    );
    assert_rel(
        coeffs.beta,
        beta_ref,
        tol,
        "fit_two_modes beta (2x2 linear solve, subtraction-eliminated)",
    );

    // ------------------------------------------------------------------
    // 3. damping_ratio -- verify the fitted (alpha, beta) hit both
    //    original targets. Independent reference:
    //    zeta_n = alpha_ref/(2*omega_n) + beta_ref*omega_n/2, evaluated
    //    in plain f64 from the hand-solved alpha_ref/beta_ref above (not
    //    by calling `damping_ratio` on the `coeffs` under test).
    // ------------------------------------------------------------------
    let zeta1_ref = alpha_ref / (2.0 * omega1_ref) + beta_ref * omega1_ref / 2.0;
    let zeta2_ref = alpha_ref / (2.0 * omega2_ref) + beta_ref * omega2_ref / 2.0;
    let zeta1_check = coeffs.damping_ratio(omega1);
    let zeta2_check = coeffs.damping_ratio(omega2);
    assert_rel(
        zeta1_check,
        zeta1_ref,
        tol,
        "damping_ratio(omega1) matches hand-solved zeta1",
    );
    assert_rel(
        zeta2_check,
        zeta2_ref,
        tol,
        "damping_ratio(omega2) matches hand-solved zeta2",
    );
    assert_rel(
        zeta1_check,
        zeta1,
        tol,
        "damping_ratio(omega1) recovers the 0.05 target directly",
    );
    assert_rel(
        zeta2_check,
        zeta2,
        tol,
        "damping_ratio(omega2) recovers the 0.02 target directly",
    );

    println!(
        "[damping_rayleigh] all 4 production entry points (hz_to_omega, omega_to_hz, \
         fit_two_modes, damping_ratio) verified against a hand-solved 2x2 linear system"
    );
}
