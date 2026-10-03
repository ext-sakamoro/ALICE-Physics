//! Audit oracles for `alice_physics::damping_rayleigh` (S2-2 audit, axis A and C).
//!
//! Expected values come from the closed forms in Chopra 5th ed. section 11.4
//! (zeta_n = alpha/(2 omega_n) + beta omega_n / 2) re-derived in plain f64,
//! never by calling the function under test.

use alice_physics::damping_rayleigh::{hz_to_omega, omega_to_hz, RayleighCoefficients};
use alice_physics::math::Fix128;

fn rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs() / want.abs().max(1e-300);
    assert!(err <= tol, "{what}: got {g}, want {want}, rel err {err:e}");
}

fn zeta_f64(a: f64, b: f64, w: f64) -> f64 {
    a / (2.0 * w) + b * w / 2.0
}

/// Targets are inputs, so this is not circular: the fit must reproduce the
/// requested damping ratios at both modes to Fix128 precision (inline test
/// tolerates 1e-3 only).
#[test]
fn fit_reproduces_both_targets_to_fix128_precision() {
    let cases: [(i64, i64, i64, i64, i64, i64); 4] = [
        (100, 5, 500, 2, 100, 100),
        (3, 10, 7000, 1, 100, 100),
        (1, 1, 2, 1, 50, 20),
        (60, 3, 61, 3, 100, 100),
    ];
    for (w1, z1n, w2, z2n, z1d, z2d) in cases {
        let (w1f, w2f) = (Fix128::from_int(w1), Fix128::from_int(w2));
        let (z1, z2) = (Fix128::from_ratio(z1n, z1d), Fix128::from_ratio(z2n, z2d));
        let r = RayleighCoefficients::fit_two_modes(w1f, z1, w2f, z2);
        rel(r.damping_ratio(w1f), z1.to_f64(), 1e-9, "zeta at mode 1");
        rel(r.damping_ratio(w2f), z2.to_f64(), 1e-9, "zeta at mode 2");
    }
}

/// The two-row system is symmetric in the mode labels.
#[test]
fn fit_two_modes_is_symmetric_in_mode_order() {
    let (w1, z1) = (Fix128::from_int(100), Fix128::from_ratio(5, 100));
    let (w2, z2) = (Fix128::from_int(500), Fix128::from_ratio(2, 100));
    let a = RayleighCoefficients::fit_two_modes(w1, z1, w2, z2);
    let b = RayleighCoefficients::fit_two_modes(w2, z2, w1, z1);
    rel(a.alpha, b.alpha.to_f64(), 1e-12, "alpha swap");
    rel(a.beta, b.beta.to_f64(), 1e-12, "beta swap");
    // and against the hand-solved system: beta = 2(w2 z2 - w1 z1)/(w2^2 - w1^2)
    let beta = 2.0 * (500.0 * 0.02 - 100.0 * 0.05) / (500.0 * 500.0 - 100.0 * 100.0);
    let alpha = 2.0 * 100.0 * 0.05 - beta * 100.0 * 100.0;
    rel(a.beta, beta, 1e-12, "beta closed form");
    rel(a.alpha, alpha, 1e-12, "alpha closed form");
}

/// zeta(omega) has its minimum at omega = sqrt(alpha/beta) with value
/// sqrt(alpha beta) (AM-GM on alpha/(2w) + beta w/2).
#[test]
fn damping_ratio_minimum_is_sqrt_alpha_beta_at_sqrt_alpha_over_beta() {
    let r = RayleighCoefficients {
        alpha: Fix128::ONE,
        beta: Fix128::from_ratio(1, 400),
    };
    // w* = 20, zeta_min = sqrt(1/400) = 0.05
    rel(
        r.damping_ratio(Fix128::from_int(20)),
        0.05,
        1e-12,
        "zeta_min",
    );
    for w in [2, 5, 10, 15, 19, 21, 30, 80, 400] {
        let z = r.damping_ratio(Fix128::from_int(w)).to_f64();
        let want = zeta_f64(1.0, 1.0 / 400.0, w as f64);
        assert!((z - want).abs() <= 1e-12, "zeta({w}) = {z}, want {want}");
        assert!(
            z >= 0.05 - 1e-12,
            "zeta({w}) = {z} below the AM-GM minimum 0.05"
        );
    }
}

/// Linear in (alpha, beta): zeta(a1+a2, b1+b2) = zeta(a1,b1) + zeta(a2,b2).
#[test]
fn damping_ratio_is_linear_in_the_coefficients() {
    let w = Fix128::from_int(37);
    let r1 = RayleighCoefficients {
        alpha: Fix128::from_ratio(3, 10),
        beta: Fix128::from_ratio(1, 1000),
    };
    let r2 = RayleighCoefficients {
        alpha: Fix128::from_ratio(7, 10),
        beta: Fix128::from_ratio(3, 1000),
    };
    let sum = RayleighCoefficients {
        alpha: r1.alpha + r2.alpha,
        beta: r1.beta + r2.beta,
    };
    let lhs = sum.damping_ratio(w).to_f64();
    let rhs = r1.damping_ratio(w).to_f64() + r2.damping_ratio(w).to_f64();
    assert!((lhs - rhs).abs() <= 1e-12, "{lhs} vs {rhs}");
}

#[test]
fn hz_omega_conversions_are_odd_and_additive() {
    let (a, b) = (Fix128::from_ratio(37, 4), Fix128::from_ratio(5, 3));
    let s = hz_to_omega(a + b).to_f64();
    let t = hz_to_omega(a).to_f64() + hz_to_omega(b).to_f64();
    assert!((s - t).abs() <= 1e-12, "additivity {s} vs {t}");
    // Fix128 mul/div floor, so negation commutes only to 1 raw ULP (2^-64).
    assert!((hz_to_omega(-a) + hz_to_omega(a)).to_f64().abs() <= 1e-15);
    let neg = omega_to_hz(-hz_to_omega(a)).to_f64();
    let pos = omega_to_hz(hz_to_omega(a)).to_f64();
    assert!((neg + pos).abs() <= 1e-15, "{neg} vs {pos}");
    // 1 Hz = 2 pi rad/s
    rel(
        hz_to_omega(Fix128::ONE),
        2.0 * core::f64::consts::PI,
        1e-15,
        "1 Hz",
    );
}

/// Large (ultrasonic) angular frequencies stay inside Fix128 range and the fit
/// still matches the hand-solved system.
#[test]
fn fit_at_ultrasonic_frequencies_matches_closed_form() {
    let (w1, w2) = (1.0e5_f64, 1.0e6_f64);
    let (z1, z2) = (0.02_f64, 0.04_f64);
    let beta = 2.0 * (w2 * z2 - w1 * z1) / (w2 * w2 - w1 * w1);
    let alpha = 2.0 * w1 * z1 - beta * w1 * w1;
    let r = RayleighCoefficients::fit_two_modes(
        Fix128::from_int(100_000),
        Fix128::from_ratio(2, 100),
        Fix128::from_int(1_000_000),
        Fix128::from_ratio(4, 100),
    );
    rel(r.alpha, alpha, 1e-9, "alpha 1e5/1e6");
    rel(r.beta, beta, 1e-9, "beta 1e5/1e6");
}

/// A Rayleigh damping matrix is positive semi-definite only when alpha >= 0 and
/// beta >= 0. fit_two_modes returns a negative beta (energy injection at high
/// frequency) when the high mode is asked for a much smaller ratio, with no
/// clamp and no doc warning.
#[test]
#[ignore = "known defect: AUD-A-S2W2-001: fit_two_modes returns beta=-2.083e-5 for (100 rad/s, z=0.05) and (500 rad/s, z=0.005) so zeta(omega) goes negative at high omega; doc is silent on admissible targets"]
fn fit_two_modes_never_returns_negative_damping_coefficients() {
    let r = RayleighCoefficients::fit_two_modes(
        Fix128::from_int(100),
        Fix128::from_ratio(5, 100),
        Fix128::from_int(500),
        Fix128::from_ratio(5, 1000),
    );
    assert!(r.alpha >= Fix128::ZERO, "alpha = {}", r.alpha.to_f64());
    assert!(r.beta >= Fix128::ZERO, "beta = {}", r.beta.to_f64());
    // consequence: damping ratio at a high frequency must not be negative
    assert!(r.damping_ratio(Fix128::from_int(2000)) >= Fix128::ZERO);
}

/// omega_2 = -omega_1 makes the 2x2 system singular (rows proportional); the doc
/// says singular systems return Default. Only the `denominator_beta` guard
/// catches this, since the coincidence guard compares omega_1 == omega_2.
#[test]
fn fit_two_modes_with_opposite_frequencies_is_singular_and_returns_default() {
    let r = RayleighCoefficients::fit_two_modes(
        Fix128::from_int(100),
        Fix128::from_ratio(5, 100),
        Fix128::from_int(-100),
        Fix128::from_ratio(2, 100),
    );
    assert_eq!(r, RayleighCoefficients::default());
}
