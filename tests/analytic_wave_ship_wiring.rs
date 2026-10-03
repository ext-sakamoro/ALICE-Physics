//! Oracles for the wiring of `wave_ship`
//! (`examples/wave_ship_spectrum.rs`): `WaveComponent`, `Jonswap::north_sea`,
//! `Jonswap::peak_omega`, `Jonswap::spectrum_density`,
//! `free_surface_elevation`, `froude_krylov_vertical_n`.
//!
//! # Closed forms (every expected value is derived here, independently of
//! the crate; see also `src/wave_ship.rs:1-23` and
//! `tests/engineering_oracles_fluid.rs`'s existing `wave_ship_*` tests,
//! which this file does not duplicate -- it adds degenerate / extreme-input
//! coverage those tests do not exercise)
//!
//! ```text
//! peak frequency     : omega_p = 2*pi / T_p                (Tp == 0 -> 0, documented)
//! JONSWAP spectrum   : S(omega) = 5/16 * Hs^2 * omega_p^4 / omega^5
//!                                  * exp(-5/4 * (omega_p/omega)^4) * gamma^r
//!                      r = exp(-(omega - omega_p)^2 / (2 * sigma^2 * omega_p^2))
//!                      sigma = 0.07 for omega <= omega_p, else 0.09
//!                      gamma <= 0 -> gamma^r treated as 1 (documented fallback)
//!                      omega <= 0, or omega_p == 0 -> S = 0 (documented)
//! free surface       : eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)
//! Froude-Krylov      : F_z = rho_w * g * A_wp * (d_mean + eta)
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::wave_ship::{
    free_surface_elevation, froude_krylov_vertical_n, Jonswap, WaveComponent,
};

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn jonswap_reference(hs: f64, omega_p: f64, gamma: f64, omega: f64) -> f64 {
    if omega <= 0.0 || omega_p <= 0.0 {
        return 0.0;
    }
    let sigma = if omega <= omega_p { 0.07 } else { 0.09 };
    let r = (-(omega - omega_p).powi(2) / (2.0 * sigma * sigma * omega_p * omega_p)).exp();
    let pm = 5.0 / 16.0 * hs * hs * omega_p.powi(4) / omega.powi(5);
    let cutoff = (-1.25 * (omega_p / omega).powi(4)).exp();
    let enhancement = if gamma <= 0.0 { 1.0 } else { gamma.powf(r) };
    pm * cutoff * enhancement
}

// ============================================================================
// Jonswap::north_sea
// ============================================================================

#[test]
fn north_sea_preset_matches_hasselmann_1973_fully_developed_values() {
    let j = Jonswap::north_sea();
    assert_eq!(j.significant_wave_height_m, Fix128::from_int(3));
    assert_eq!(j.peak_period_s, Fix128::from_int(9));
    assert!(rel_err(j.gamma.to_f64(), 3.3) < 1e-9);
}

// ============================================================================
// Jonswap::peak_omega
// ============================================================================

#[test]
fn peak_omega_matches_two_pi_over_tp_across_magnitudes() {
    // Ordinary sea states: Tp in [4, 20] s, independently computed 2*pi/Tp.
    for tp in [4.0_f64, 6.5, 9.0, 12.0, 20.0] {
        let j = Jonswap {
            significant_wave_height_m: Fix128::from_int(1),
            peak_period_s: Fix128::from_f64(tp),
            gamma: Fix128::from_ratio(33, 10),
        };
        let got = j.peak_omega().to_f64();
        let want = 2.0 * std::f64::consts::PI / tp;
        assert!(rel_err(got, want) < 1e-9, "Tp={tp}: {got} vs {want}");
    }
}

#[test]
fn peak_omega_extreme_peak_period() {
    // Extremely short period -> very large omega_p.
    let short = Jonswap {
        significant_wave_height_m: Fix128::from_int(1),
        peak_period_s: Fix128::from_ratio(1, 1000), // 0.001 s
        gamma: Fix128::from_ratio(33, 10),
    };
    let got = short.peak_omega().to_f64();
    let want = 2.0 * std::f64::consts::PI / 0.001;
    assert!(rel_err(got, want) < 1e-6, "short Tp: {got} vs {want}");

    // Extremely long period -> omega_p close to zero but nonzero.
    let long = Jonswap {
        significant_wave_height_m: Fix128::from_int(1),
        peak_period_s: Fix128::from_int(1_000_000),
        gamma: Fix128::from_ratio(33, 10),
    };
    let got = long.peak_omega().to_f64();
    let want = 2.0 * std::f64::consts::PI / 1_000_000.0;
    assert!(rel_err(got, want) < 1e-3, "long Tp: {got} vs {want}");
    assert!(
        got > 0.0,
        "long but finite Tp must not collapse to exactly 0"
    );
}

#[test]
fn peak_omega_zero_peak_period_returns_exact_zero() {
    // Documented branch: Tp == 0 -> ZERO (not a division by zero).
    let degenerate = Jonswap {
        significant_wave_height_m: Fix128::from_int(1),
        peak_period_s: Fix128::ZERO,
        gamma: Fix128::from_ratio(33, 10),
    };
    assert_eq!(degenerate.peak_omega(), Fix128::ZERO);
}

// ============================================================================
// Jonswap::spectrum_density
// ============================================================================

#[test]
fn spectrum_density_matches_jonswap_closed_form_default_gamma() {
    let j = Jonswap::north_sea();
    let omega_p = j.peak_omega().to_f64();
    let hs = j.significant_wave_height_m.to_f64();
    let gamma = j.gamma.to_f64();
    for frac in [1, 2, 3, 5, 7, 11, 17, 23] {
        let omega = omega_p * f64::from(frac) / 10.0;
        let got = j.spectrum_density(Fix128::from_f64(omega)).to_f64();
        let want = jonswap_reference(hs, omega_p, gamma, omega);
        // Deep below the peak (frac 0.1, 0.2) the exp(-5/4 (omega_p/omega)^4)
        // cutoff drives the closed form far below Fix128's ~5.4e-20 absolute
        // resolution (e.g. ~1.58e-64 at frac=0.1) -- the crate cannot
        // represent that and correctly collapses to ZERO, so this is an
        // "effectively vanishing" comparison there, not a relative one.
        if want.abs() < 1e-18 {
            assert!(
                got.abs() < 1e-16,
                "S({omega}) should vanish, got {got} (want {want:e})"
            );
        } else {
            assert!(rel_err(got, want) < 1e-3, "S({omega}) = {got} vs {want}");
        }
    }
}

#[test]
fn spectrum_density_gamma_zero_falls_back_to_unity_enhancement() {
    // gamma <= 0 is documented to behave as enhancement = 1 (not gamma^r,
    // which would be ill-defined for gamma = 0 and r != 0 since 0^r != 1
    // in general -- this is a fallback, not the limit of gamma^r).
    let j = Jonswap {
        significant_wave_height_m: Fix128::from_int(3),
        peak_period_s: Fix128::from_int(9),
        gamma: Fix128::ZERO,
    };
    let omega_p = j.peak_omega().to_f64();
    let hs = j.significant_wave_height_m.to_f64();
    for frac in [0.5_f64, 1.0, 2.0] {
        let omega = omega_p * frac;
        let got = j.spectrum_density(Fix128::from_f64(omega)).to_f64();
        // enhancement forced to 1.0 regardless of r
        let want = jonswap_reference(hs, omega_p, 1.0, omega);
        assert!(
            rel_err(got, want) < 1e-3,
            "S({omega}) gamma=0: {got} vs {want}"
        );
    }
}

#[test]
fn spectrum_density_negative_gamma_also_falls_back_to_unity() {
    let j = Jonswap {
        significant_wave_height_m: Fix128::from_int(3),
        peak_period_s: Fix128::from_int(9),
        gamma: Fix128::from_int(-5),
    };
    let omega_p = j.peak_omega().to_f64();
    let hs = j.significant_wave_height_m.to_f64();
    let got = j.spectrum_density(Fix128::from_f64(omega_p)).to_f64();
    let want = jonswap_reference(hs, omega_p, 1.0, omega_p);
    assert!(
        rel_err(got, want) < 1e-3,
        "S(omega_p) gamma<0: {got} vs {want}"
    );
}

#[test]
fn spectrum_density_nonpositive_omega_is_exact_zero() {
    let j = Jonswap::north_sea();
    assert_eq!(j.spectrum_density(Fix128::ZERO), Fix128::ZERO);
    assert_eq!(j.spectrum_density(Fix128::from_int(-1)), Fix128::ZERO);
    assert_eq!(
        j.spectrum_density(Fix128::from_int(-1_000_000)),
        Fix128::ZERO
    );
}

#[test]
fn spectrum_density_zero_peak_period_is_exact_zero_at_any_omega() {
    // omega_p == 0 (via Tp == 0) short-circuits to ZERO before the 1/omega^5
    // term could ever be evaluated.
    let j = Jonswap {
        significant_wave_height_m: Fix128::from_int(3),
        peak_period_s: Fix128::ZERO,
        gamma: Fix128::from_ratio(33, 10),
    };
    for omega in [
        Fix128::from_ratio(1, 1000),
        Fix128::ONE,
        Fix128::from_int(1000),
    ] {
        assert_eq!(j.spectrum_density(omega), Fix128::ZERO);
    }
}

#[test]
fn spectrum_density_extreme_magnitude_does_not_panic_and_decays() {
    let j = Jonswap::north_sea();
    let omega_p = j.peak_omega();
    let peak = j.spectrum_density(omega_p);

    // Moderately far above the peak (mult = 100, 1_000): the closed-form
    // value (5/16 Hs^2 omega_p^4 / omega^5, cutoff ~= 1 here since
    // (omega_p/omega)^4 is tiny) is still many orders of magnitude above
    // Fix128's ~5.4e-20 absolute resolution, so the 1/omega^5 decay must
    // show up as a real, strictly positive, monotonically decreasing
    // sequence -- not noise.
    let s100 = j.spectrum_density(omega_p * Fix128::from_int(100));
    let s1000 = j.spectrum_density(omega_p * Fix128::from_int(1_000));
    assert!(
        s100 > Fix128::ZERO && s100 < peak,
        "S(100*omega_p) must be small and positive, below the peak: {s100:?}"
    );
    assert!(
        s1000 > Fix128::ZERO && s1000 < s100,
        "S(1000*omega_p) must decay further: {s1000:?} vs {s100:?}"
    );

    // Deep tail (mult >= 10_000): the closed-form reference value has
    // already underflowed below Fix128's absolute resolution here (at
    // mult=10_000 it is of order 1/mult^5 below the mult=1_000 value, i.e.
    // ~4e-20 -- right at the resolution floor), so the crate's output is
    // resolution-floor noise rather than a meaningful density. Measured
    // (not assumed): at mult=10_000 the result is a few raw units *below*
    // ZERO (hi=-1, lo=2^64-6, approx -3.25e-19) instead of clamped to it;
    // at mult=100_000 / 1_000_000 it floors to the smallest representable
    // positive value (Fix128::from_raw(0, 1), approx 5.42e-20) instead of
    // true zero. Both are non-blocking fixed-point rounding artifacts at
    // the resolution floor (recorded in the Backlog), not a sign flip or
    // an unbounded blow-up -- the only thing asserted here is "bounded",
    // not "nonnegative" or "monotonic".
    for mult in [10_000, 100_000, 1_000_000] {
        let far = j.spectrum_density(omega_p * Fix128::from_int(mult));
        assert!(
            far.abs() < Fix128::from_raw(0, 1_000_000),
            "deep tail must stay within a tiny band of zero, not blow up: \
             mult={mult} far={far:?} ({})",
            far.to_f64()
        );
    }
}

// ============================================================================
// WaveComponent + free_surface_elevation
// ============================================================================

#[test]
fn free_surface_elevation_matches_cosine_closed_form_pointwise() {
    // The direct closed form eta = A*cos(k*x - omega*t + phi), independently
    // evaluated with std f64 trig, for several (A, k, omega, phi) and
    // several (x, t) -- not the crest-speed / superposition invariants used
    // elsewhere, which hold for any affine transform of the phase argument
    // and therefore cannot tell `k*x - omega*t` apart from `k*x + omega*t`.
    let specs: [(f64, f64, f64, f64); 3] = [
        (2.0, 0.3, 0.7, 0.2),
        (1.5, 1.1, 0.4, -0.9),
        (0.5, 0.05, 2.0, 3.0),
    ];
    for &(amp, k, omega, phase) in &specs {
        let c = WaveComponent {
            amplitude_m: Fix128::from_f64(amp),
            wavenumber_rad_per_m: Fix128::from_f64(k),
            omega_rad_per_s: Fix128::from_f64(omega),
            phase_rad: Fix128::from_f64(phase),
        };
        for &(x, t) in &[(0.0_f64, 0.0_f64), (5.0, 2.0), (-3.0, 7.0), (10.0, -4.0)] {
            let got =
                free_surface_elevation(&[c], Fix128::from_f64(x), Fix128::from_f64(t)).to_f64();
            let want = amp * (k * x - omega * t + phase).cos();
            assert!(
                (got - want).abs() < 1e-6,
                "eta(x={x},t={t}) for (A={amp},k={k},omega={omega},phi={phase}): {got} vs {want}"
            );
        }
    }
}

#[test]
fn free_surface_elevation_zero_amplitude_component_contributes_nothing() {
    let zero_amp = WaveComponent {
        amplitude_m: Fix128::ZERO,
        omega_rad_per_s: Fix128::ONE,
        wavenumber_rad_per_m: Fix128::from_ratio(1, 10),
        phase_rad: Fix128::from_ratio(1, 4),
    };
    let real = WaveComponent {
        amplitude_m: Fix128::from_int(2),
        omega_rad_per_s: Fix128::from_int(2),
        wavenumber_rad_per_m: Fix128::from_ratio(1, 5),
        phase_rad: Fix128::ZERO,
    };
    let with_zero =
        free_surface_elevation(&[zero_amp, real], Fix128::from_int(3), Fix128::from_int(1));
    let without = free_surface_elevation(&[real], Fix128::from_int(3), Fix128::from_int(1));
    assert_eq!(
        with_zero, without,
        "a zero-amplitude component must be a no-op"
    );
}

#[test]
fn free_surface_elevation_zero_frequency_component_is_time_invariant() {
    // omega = 0, k = 0 (consistent with deep-water dispersion k = omega^2/g):
    // the component reduces to a constant offset A*cos(phi) for all (x, t).
    let standing = WaveComponent {
        amplitude_m: Fix128::from_int(2),
        omega_rad_per_s: Fix128::ZERO,
        wavenumber_rad_per_m: Fix128::ZERO,
        phase_rad: Fix128::from_ratio(1, 3),
    };
    let want = 2.0 * (1.0_f64 / 3.0).cos();
    for (x, t) in [(0.0_f64, 0.0_f64), (1e6, -1e6), (-500.0, 9999.0)] {
        let eta =
            free_surface_elevation(&[standing], Fix128::from_f64(x), Fix128::from_f64(t)).to_f64();
        assert!(rel_err(eta, want) < 1e-6, "eta({x},{t}) = {eta} vs {want}");
    }
}

#[test]
fn free_surface_elevation_single_vs_superposition_of_several() {
    // Single-component elevation must equal the term-by-term sum when the
    // same components are superposed, for a list of 5 (not 1 or 2, as in
    // tests/engineering_oracles_fluid.rs).
    let comps: Vec<WaveComponent> = (1..=5)
        .map(|i| WaveComponent {
            amplitude_m: Fix128::from_ratio(i, 2),
            omega_rad_per_s: Fix128::from_int(i),
            wavenumber_rad_per_m: Fix128::from_ratio(i, 3),
            phase_rad: Fix128::from_ratio(i, 7),
        })
        .collect();
    let x = Fix128::from_int(5);
    let t = Fix128::from_int(2);
    let sum_of_singles: f64 = comps
        .iter()
        .map(|&c| free_surface_elevation(&[c], x, t).to_f64())
        .sum();
    let superposed = free_surface_elevation(&comps, x, t).to_f64();
    assert!(
        (sum_of_singles - superposed).abs() < 1e-6,
        "superposition must equal the sum of single-component evaluations: {superposed} vs {sum_of_singles}"
    );
}

#[test]
fn free_surface_elevation_extreme_magnitudes() {
    // Large x and t with a small wavenumber/frequency: the phase argument is
    // still well inside Fix128's range (hi: i64), cos() must stay bounded.
    let c = WaveComponent {
        amplitude_m: Fix128::from_int(1_000_000),
        omega_rad_per_s: Fix128::from_ratio(1, 1_000_000),
        wavenumber_rad_per_m: Fix128::from_ratio(1, 1_000_000),
        phase_rad: Fix128::ZERO,
    };
    let eta = free_surface_elevation(
        &[c],
        Fix128::from_int(1_000_000_000),
        Fix128::from_int(-1_000_000_000),
    )
    .to_f64();
    assert!(
        eta.abs() <= 1_000_000.0 + 1.0,
        "|eta| must stay bounded by the amplitude: {eta}"
    );
    assert!(
        eta.is_finite(),
        "extreme magnitude must not produce NaN/inf: {eta}"
    );
}

#[test]
fn free_surface_elevation_empty_list_is_exact_zero() {
    assert_eq!(
        free_surface_elevation(&[], Fix128::from_int(42), Fix128::from_int(-7)),
        Fix128::ZERO
    );
}

// ============================================================================
// froude_krylov_vertical_n
// ============================================================================

#[test]
fn froude_krylov_vertical_n_matches_hydrostatic_product() {
    for (rho, g, area, draft, eta) in [
        (1000.0_f64, 9.80665, 49.5, 2.5, 0.0),
        (1000.0, 9.80665, 49.5, 2.5, -0.8),
        (1000.0, 9.80665, 49.5, 2.5, 3.0),
        (1025.0, 9.81, 10.0, 0.5, -0.5), // just enough draft to net nonnegative
    ] {
        let f = froude_krylov_vertical_n(
            Fix128::from_f64(rho),
            Fix128::from_f64(g),
            Fix128::from_f64(area),
            Fix128::from_f64(draft),
            Fix128::from_f64(eta),
        )
        .to_f64();
        let want = rho * g * area * (draft + eta);
        assert!(
            rel_err(f, want) < 1e-6,
            "F(rho={rho},area={area},draft={draft},eta={eta}) = {f} vs {want}"
        );
    }
}

#[test]
fn froude_krylov_vertical_n_degenerate_inputs_are_exact_zero() {
    let rho = Fix128::from_int(1000);
    let g = Fix128::from_ratio(981, 100);
    let area = Fix128::from_int(10);
    // zero area -> exactly zero regardless of draft/elevation
    assert_eq!(
        froude_krylov_vertical_n(
            rho,
            g,
            Fix128::ZERO,
            Fix128::from_int(5),
            Fix128::from_int(3)
        ),
        Fix128::ZERO
    );
    // zero density -> exactly zero
    assert_eq!(
        froude_krylov_vertical_n(
            Fix128::ZERO,
            g,
            area,
            Fix128::from_int(5),
            Fix128::from_int(3)
        ),
        Fix128::ZERO
    );
    // zero draft AND zero elevation -> exactly zero
    assert_eq!(
        froude_krylov_vertical_n(rho, g, area, Fix128::ZERO, Fix128::ZERO),
        Fix128::ZERO
    );
    // draft exactly cancels elevation -> exactly zero (d + eta == 0)
    assert_eq!(
        froude_krylov_vertical_n(rho, g, area, Fix128::from_int(4), Fix128::from_int(-4)),
        Fix128::ZERO
    );
}

#[test]
fn froude_krylov_vertical_n_negative_elevation_can_go_negative() {
    // A trough deep enough to exceed the mean draft produces a net downward
    // (negative) force -- not clamped at zero.
    let f = froude_krylov_vertical_n(
        Fix128::from_int(1000),
        Fix128::from_ratio(981, 100),
        Fix128::from_int(10),
        Fix128::from_int(1),
        Fix128::from_int(-5), // trough of 5 m against a 1 m mean draft
    );
    assert!(
        f < Fix128::ZERO,
        "deep trough must net negative force: {f:?}"
    );
    let want = 1000.0 * 9.81 * 10.0 * (1.0 - 5.0);
    assert!(rel_err(f.to_f64(), want) < 1e-6);
}

#[test]
fn froude_krylov_vertical_n_extreme_magnitudes_scale_linearly() {
    // Linear in area: F(2*A) == 2*F(A) exactly (Fix128 multiplication,
    // doubling has no rounding).
    let rho = Fix128::from_int(1025);
    let g = Fix128::from_ratio(981, 100);
    let draft = Fix128::from_int(4);
    let eta = Fix128::from_ratio(3, 2);
    let area = Fix128::from_int(1_000_000);
    let f1 = froude_krylov_vertical_n(rho, g, area, draft, eta);
    let f2 = froude_krylov_vertical_n(rho, g, area + area, draft, eta);
    assert_eq!(f2, f1 + f1, "doubling area must exactly double the force");
}
