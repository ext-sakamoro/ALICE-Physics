//! Oracles for the production entry points of
//! `alice_physics::damping_rayleigh` driven by
//! `examples/damping_rayleigh_fit.rs`: `hz_to_omega`, `omega_to_hz`,
//! `RayleighCoefficients::fit_two_modes`, and
//! `RayleighCoefficients::damping_ratio`.
//!
//! # What this file is and is not
//!
//! `src/damping_rayleigh.rs`'s own `#[cfg(test)]` module already checks
//! the ordinary (non-degenerate) path of all four functions at one pair
//! of modes (omega=100/500 rad/s), the zero-omega guard for
//! `damping_ratio`, and the two early-return guards of `fit_two_modes`
//! (zero omega, equal omega) -- those exact numeric cases are not
//! repeated here. `examples/damping_rayleigh_fit.rs` drives all four at
//! a 2 Hz / 20 Hz mode pair chosen as a realistic scenario. What neither
//! of those covers, and this file does:
//!
//! * `fit_two_modes`'s 2x2 linear solve at a different, independently
//!   hand-solved mode pair (5 Hz / 50 Hz) -- a second numeric witness
//!   that the subtraction-eliminate-then-back-substitute formula in the
//!   module doc is what the implementation actually computes, not just
//!   that it hits its own target (which `fit_two_modes_recovers_targets`
//!   in `src/damping_rayleigh.rs` already confirms circularly via
//!   `damping_ratio`),
//! * the `alpha=0` (pure stiffness-proportional) and `beta=0` (pure
//!   mass-proportional) boundary cases of `damping_ratio`, checked
//!   against the two halves of the closed form individually and against
//!   the monotonic direction (`beta=0` decreases with frequency, `alpha
//!   = 0` increases with frequency) rather than just "low > high" at
//!   one arbitrary pair of frequencies,
//! * `hz_to_omega` / `omega_to_hz` against an independent
//!   `core::f64::consts::PI` reference (not `Fix128::PI`, and not by
//!   calling the other conversion function) at several frequencies
//!   including a fractional one, plus the round-trip composition,
//! * zero and extreme-magnitude frequency / angular-frequency inputs for
//!   all four functions: `hz_to_omega(0)`, `omega_to_hz(0)`,
//!   `fit_two_modes` with one input omega exactly zero (the first guard
//!   leg, independent of the equal-omega second guard leg already
//!   covered in `src/`), and a 1e12 Hz / combined-alpha-beta extreme
//!   magnitude case for both `hz_to_omega` and `damping_ratio` that
//!   stays well inside `Fix128`'s +-9.2e18 range but far outside the
//!   magnitudes any existing test exercises.
//!
//! All expected numbers are derived independently in plain f64 --
//! `omega = 2*pi*f` (`core::f64::consts::PI`, never `Fix128::PI`), and
//! the `fit_two_modes` / `damping_ratio` closed forms re-derived from
//! `zeta_n = alpha/(2*omega_n) + beta*omega_n/2` -- never by calling the
//! function under test, per this repo's analytic-oracle-tests discipline
//! (`~/claude-config/rules/analytic-oracle-tests.md`).
//!
//! `hz_to_omega` / `omega_to_hz` are ordinary exact `Fix128`
//! multiply/divide by the precomputed `Fix128::PI` constant (no CORDIC
//! transcendental call), so the crate-wide determinism discipline
//! applies to them exactly as it does to every other `Fix128` function
//! here; they are not exempt as a host-side utility.

use alice_physics::damping_rayleigh::{hz_to_omega, omega_to_hz, RayleighCoefficients};
use alice_physics::math::Fix128;
use core::f64::consts::PI;

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

// ============================================================================
// Section 1: hz_to_omega / omega_to_hz -- independent f64::consts::PI
// reference, several frequencies including a fractional one, and the
// round-trip composition.
// ============================================================================

#[test]
fn hz_to_omega_matches_two_pi_times_frequency_independent_reference() {
    for &f_hz in &[1_i64, 2, 5, 50, 1000] {
        let want = 2.0 * PI * (f_hz as f64);
        let got = hz_to_omega(Fix128::from_int(f_hz));
        assert_rel(got, want, FIX_TOL, "hz_to_omega(integer Hz) = 2*pi*f");
    }
}

#[test]
fn hz_to_omega_fractional_frequency_matches_two_pi_times_frequency() {
    // 0.5 Hz: exercises from_ratio rather than from_int.
    let want = 2.0 * PI * 0.5;
    let got = hz_to_omega(Fix128::from_ratio(1, 2));
    assert_rel(got, want, FIX_TOL, "hz_to_omega(0.5 Hz) = 2*pi*0.5");
}

#[test]
fn omega_to_hz_matches_omega_over_two_pi_independent_reference() {
    for &f_hz in &[1.0_f64, 10.0, 100.0] {
        let omega = 2.0 * PI * f_hz;
        let want = omega / (2.0 * PI);
        let got = omega_to_hz(Fix128::from_f64(omega));
        assert_rel(got, want, FIX_TOL, "omega_to_hz(omega) = omega/(2*pi)");
    }
}

#[test]
fn hz_to_omega_omega_to_hz_round_trip_recovers_original_frequency() {
    for &f_hz in &[1_i64, 2, 5, 50, 1000] {
        let want = f_hz as f64;
        let omega = hz_to_omega(Fix128::from_int(f_hz));
        let back = omega_to_hz(omega);
        assert_rel(
            back,
            want,
            FIX_TOL,
            "omega_to_hz(hz_to_omega(f)) round-trips to f",
        );
    }
}

#[test]
fn hz_to_omega_zero_is_zero_exactly() {
    assert_eq!(hz_to_omega(Fix128::ZERO), Fix128::ZERO);
}

#[test]
fn omega_to_hz_zero_is_zero_exactly() {
    assert_eq!(omega_to_hz(Fix128::ZERO), Fix128::ZERO);
}

/// Extreme magnitude: 1e12 Hz is far beyond anything a physical structure
/// would exercise, but stays well inside `Fix128`'s +-9.2e18 range
/// (omega ~= 6.28e12), so this checks the conversion does not lose
/// precision or silently wrap at a magnitude no other test reaches.
#[test]
fn hz_to_omega_extreme_magnitude_frequency_matches_closed_form() {
    let f_hz = 1.0e12_f64;
    let want = 2.0 * PI * f_hz;
    let got = hz_to_omega(Fix128::from_f64(f_hz));
    assert_rel(
        got,
        want,
        FIX_TOL,
        "hz_to_omega(1e12 Hz) = 2*pi*1e12 (extreme magnitude)",
    );
}

#[test]
fn omega_to_hz_extreme_magnitude_omega_matches_closed_form() {
    let omega = 2.0 * PI * 1.0e12_f64;
    let want = omega / (2.0 * PI);
    let got = omega_to_hz(Fix128::from_f64(omega));
    assert_rel(
        got,
        want,
        FIX_TOL,
        "omega_to_hz(6.28e12 rad/s) = omega/(2*pi) (extreme magnitude)",
    );
}

// ============================================================================
// Section 2: fit_two_modes -- a second hand-solved mode pair, and the
// zero-omega guard's first leg in isolation.
// ============================================================================

/// Independent reference for the 2x2 linear solve in the module doc:
/// subtract the two per-mode `2*omega*zeta = alpha + beta*omega^2`
/// equations to eliminate `alpha` and solve for `beta`, then
/// back-substitute for `alpha`.
fn fit_two_modes_ref_f64(w1: f64, z1: f64, w2: f64, z2: f64) -> (f64, f64) {
    let beta = (2.0 * (w2 * z2 - w1 * z1)) / (w2 * w2 - w1 * w1);
    let alpha = 2.0 * w1 * z1 - beta * w1 * w1;
    (alpha, beta)
}

#[test]
fn fit_two_modes_matches_hand_solved_linear_system_for_a_second_mode_pair() {
    // 5 Hz at zeta=0.03, 50 Hz at zeta=0.01 -- distinct from both
    // src/damping_rayleigh.rs's own unit test (omega=100/500 directly)
    // and examples/damping_rayleigh_fit.rs (2 Hz/20 Hz).
    let w1 = 2.0 * PI * 5.0;
    let w2 = 2.0 * PI * 50.0;
    let (z1, z2) = (0.03_f64, 0.01_f64);
    let (alpha_ref, beta_ref) = fit_two_modes_ref_f64(w1, z1, w2, z2);

    let coeffs = RayleighCoefficients::fit_two_modes(
        hz_to_omega(Fix128::from_int(5)),
        Fix128::from_ratio(3, 100),
        hz_to_omega(Fix128::from_int(50)),
        Fix128::from_ratio(1, 100),
    );
    assert_rel(
        coeffs.alpha,
        alpha_ref,
        FIX_TOL,
        "fit_two_modes alpha (5Hz/50Hz)",
    );
    assert_rel(
        coeffs.beta,
        beta_ref,
        FIX_TOL,
        "fit_two_modes beta (5Hz/50Hz)",
    );
}

/// First leg of the `omega_1.is_zero() || omega_2.is_zero()` guard in
/// isolation: `omega_1 == 0`, `omega_2` a genuine nonzero value. A mutant
/// that drops this leg (or swaps `||` for `&&`) would instead attempt a
/// division by `omega_1 == 0` inside the real linear solve.
#[test]
fn fit_two_modes_first_omega_zero_returns_default() {
    let r = RayleighCoefficients::fit_two_modes(
        Fix128::ZERO,
        Fix128::from_ratio(1, 100),
        Fix128::from_int(100),
        Fix128::from_ratio(2, 100),
    );
    assert_eq!(r, RayleighCoefficients::default());
}

/// Second leg of the same guard in isolation: `omega_2 == 0`, `omega_1`
/// nonzero -- the mirror image of the case above.
#[test]
fn fit_two_modes_second_omega_zero_returns_default() {
    let r = RayleighCoefficients::fit_two_modes(
        Fix128::from_int(100),
        Fix128::from_ratio(1, 100),
        Fix128::ZERO,
        Fix128::from_ratio(2, 100),
    );
    assert_eq!(r, RayleighCoefficients::default());
}

// ============================================================================
// Section 3: damping_ratio -- alpha=0 / beta=0 boundary cases checked
// against the individual closed-form halves (not just "low > high"),
// plus a combined extreme-magnitude case.
// ============================================================================

/// `alpha=0` (pure stiffness-proportional damping): `zeta = beta*omega/2`
/// grows linearly with frequency. Checked at two frequencies against the
/// closed form individually, and that the ratio between them matches the
/// ratio of frequencies exactly (linear, not merely monotonic).
#[test]
fn damping_ratio_pure_stiffness_proportional_matches_linear_closed_form() {
    let beta = Fix128::from_ratio(2, 10_000); // 2e-4
    let beta_f64 = 2.0e-4_f64;
    let r = RayleighCoefficients {
        alpha: Fix128::ZERO,
        beta,
    };
    let low = r.damping_ratio(Fix128::from_int(10));
    let high = r.damping_ratio(Fix128::from_int(1000));
    assert_rel(
        low,
        beta_f64 * 10.0 / 2.0,
        FIX_TOL,
        "zeta(omega=10) = beta*10/2",
    );
    assert_rel(
        high,
        beta_f64 * 1000.0 / 2.0,
        FIX_TOL,
        "zeta(omega=1000) = beta*1000/2",
    );
    assert!(
        high > low,
        "beta=0-free (pure stiffness) must increase with frequency"
    );
}

/// `beta=0` (pure mass-proportional damping): `zeta = alpha/(2*omega)`
/// shrinks hyperbolically with frequency. Checked at two frequencies
/// against the closed form individually.
#[test]
fn damping_ratio_pure_mass_proportional_matches_hyperbolic_closed_form() {
    let alpha = Fix128::from_int(5);
    let alpha_f64 = 5.0_f64;
    let r = RayleighCoefficients {
        alpha,
        beta: Fix128::ZERO,
    };
    let low = r.damping_ratio(Fix128::from_int(10));
    let high = r.damping_ratio(Fix128::from_int(1000));
    assert_rel(
        low,
        alpha_f64 / (2.0 * 10.0),
        FIX_TOL,
        "zeta(omega=10) = alpha/(2*10)",
    );
    assert_rel(
        high,
        alpha_f64 / (2.0 * 1000.0),
        FIX_TOL,
        "zeta(omega=1000) = alpha/(2*1000)",
    );
    assert!(
        high < low,
        "alpha-only (pure mass) must decrease with frequency"
    );
}

#[test]
fn damping_ratio_both_zero_coefficients_is_zero_at_any_frequency() {
    let r = RayleighCoefficients::default();
    assert_abs(
        r.damping_ratio(Fix128::from_int(1)),
        0.0,
        FIX_TOL,
        "zeta(omega=1)=0",
    );
    assert_abs(
        r.damping_ratio(Fix128::from_int(10_000)),
        0.0,
        FIX_TOL,
        "zeta(omega=10000)=0",
    );
}

#[test]
fn damping_ratio_zero_omega_is_zero_regardless_of_coefficients() {
    // Independent of src/damping_rayleigh.rs's own unit test: nonzero,
    // distinct alpha and beta, confirming the omega==0 guard fires ahead
    // of the division rather than coincidentally cancelling.
    let r = RayleighCoefficients {
        alpha: Fix128::from_int(7),
        beta: Fix128::from_ratio(1, 3),
    };
    assert_eq!(r.damping_ratio(Fix128::ZERO), Fix128::ZERO);
}

/// Extreme magnitude: omega = 1e6 rad/s combined with both coefficients
/// nonzero, verified against the full two-term closed form.
#[test]
fn damping_ratio_extreme_magnitude_omega_matches_closed_form() {
    let alpha = Fix128::from_ratio(3, 2); // 1.5
    let beta = Fix128::from_f64(3.0e-7);
    let omega = 1.0e6_f64;
    let want = 1.5 / (2.0 * omega) + 3.0e-7 * omega / 2.0;
    let r = RayleighCoefficients { alpha, beta };
    let got = r.damping_ratio(Fix128::from_f64(omega));
    assert_rel(
        got,
        want,
        FIX_TOL,
        "zeta(omega=1e6) = alpha/(2*1e6) + beta*1e6/2",
    );
}
