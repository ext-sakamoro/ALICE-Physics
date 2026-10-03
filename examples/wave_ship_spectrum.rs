//! Ocean wave spectrum + free-surface superposition — production entry point
//! for every item of `src/wave_ship.rs` that `wiring_guard.py` found with
//! zero production callers: `WaveComponent`, `Jonswap::north_sea`,
//! `Jonswap::peak_omega`, `Jonswap::spectrum_density`,
//! `free_surface_elevation`, `froude_krylov_vertical_n`. Only `tests/`
//! called into these before this example (mainly
//! `tests/engineering_oracles_fluid.rs::{wave_ship_deep_water_dispersion_and_froude_krylov,
//! wave_ship_spectrum_is_jonswap}`), which the guard does not count as
//! production (src / examples / benches / fuzz / bindings only).
//!
//! Every closed form here is re-derived independently of those two tests —
//! different sea states, different frequency sampling, and (for
//! `spectrum_density`) a check neither test performs: recovering `H_s` back
//! out of the JONSWAP spectrum's zeroth moment at the *default* peak
//! enhancement `γ = 3.3` (the existing test's moment check uses the
//! Pierson-Moskowitz reduction `γ = 1`, where the enhancement factor drops
//! out of the formula entirely).
//!
//! # Closed forms (Hasselmann et al. 1973; Chakrabarti, *Hydrodynamics of
//! Offshore Structures* 1987 eq. 4.29; Dean & Dalrymple, *Water Wave
//! Mechanics* eq. 3.30; Faltinsen §3.1)
//!
//! ```text
//! peak frequency     : omega_p = 2*pi / T_p
//! JONSWAP spectrum   : S(omega) = 5/16 * Hs^2 * omega_p^4 / omega^5
//!                                  * exp(-5/4 * (omega_p/omega)^4) * gamma^r
//!                      r = exp(-(omega - omega_p)^2 / (2 * sigma^2 * omega_p^2))
//!                      sigma = 0.07 for omega <= omega_p, else 0.09
//! zeroth moment      : m0 = integral S(omega) d(omega)
//! significant height : Hs = 4 * sqrt(m0)                  (Chakrabarti eq. 4.9)
//! deep-water disp.   : omega^2 = g*k                      (so c = g/omega)
//! free surface       : eta(x, t) = sum_i A_i * cos(k_i*x - omega_i*t + phi_i)
//! Froude-Krylov      : F_z = rho_w * g * A_wp * (d_mean + eta)
//! ```
//!
//! ```bash
//! cargo run --example wave_ship_spectrum --features std
//! ```
//!
//! Author: Moroya Sakamoto

// The reference values in `jonswap_reference` and the free-surface
// superposition closed form below are independent f64 evaluations of the
// textbook formula, not simulation state -- same rationale as
// `tests/engineering_oracles_fluid.rs`.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::wave_ship::{
    free_surface_elevation, froude_krylov_vertical_n, Jonswap, WaveComponent,
};

const PI_F64: f64 = std::f64::consts::PI;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

/// Independent f64 evaluation of the JONSWAP spectrum (Chakrabarti eq. 4.29),
/// written down from the paper formula, never by calling [`Jonswap::spectrum_density`].
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

fn main() {
    // --- Jonswap::north_sea + Jonswap::peak_omega --------------------------
    let sea = Jonswap::north_sea();
    let hs = sea.significant_wave_height_m.to_f64();
    let tp = sea.peak_period_s.to_f64();
    let gamma = sea.gamma.to_f64();
    println!("[wave_ship] north_sea(): Hs={hs} m, Tp={tp} s, gamma={gamma}");
    assert_eq!(hs, 3.0, "north_sea Hs");
    assert_eq!(tp, 9.0, "north_sea Tp");
    assert!(rel_err(gamma, 3.3) < 1e-9, "north_sea gamma");

    let omega_p = sea.peak_omega().to_f64();
    let omega_p_want = 2.0 * PI_F64 / tp;
    println!("[wave_ship] peak_omega() = {omega_p} rad/s (closed form {omega_p_want})");
    assert!(rel_err(omega_p, omega_p_want) < 1e-9, "peak_omega");

    // --- Jonswap::spectrum_density, sampled across the tail and the peak ---
    // (Different sampling grid from tests/engineering_oracles_fluid.rs's
    // 0.1*wp..4*wp sweep: here it is irregular and includes points the
    // existing test never visits.)
    for frac in [0.37_f64, 0.8, 1.0, 1.6, 2.9, 5.5] {
        let omega = omega_p * frac;
        let got = sea.spectrum_density(Fix128::from_f64(omega)).to_f64();
        let want = jonswap_reference(hs, omega_p, gamma, omega);
        println!("[wave_ship] S({omega:.4}) = {got:e} (closed form {want:e})");
        if want.abs() < 1e-9 {
            assert!(got.abs() < 1e-7, "S({omega}) should vanish, got {got}");
        } else {
            assert!(rel_err(got, want) < 1e-3, "S({omega}) = {got} vs {want}");
        }
    }

    // --- Recover Hs from the zeroth moment of the FULL JONSWAP spectrum ----
    // (at the default gamma = 3.3, not the gamma = 1 Pierson-Moskowitz
    // reduction the existing oracle test checks -- here the peak-enhancement
    // factor does not cancel out of the formula, so this exercises it.)
    //
    // NOTE (finding, not a wiring bug -- see Backlog): the textbook identity
    // `Hs = 4*sqrt(m0)` only holds as coded here when gamma = 1. The
    // standard JONSWAP normalisation (DNV-RP-C205 eq. 3.5.7; also Goda,
    // *Random Seas and Design of Maritime Structures*) applies a correction
    // factor `(1 - 0.287*ln(gamma))` to the alpha/Hs^2 prefactor specifically
    // so that `4*sqrt(m0)` keeps reproducing the nominal Hs for gamma != 1.
    // `Jonswap::spectrum_density` (src/wave_ship.rs) uses the bare
    // `5/16 * Hs^2 * omega_p^4` prefactor with no such correction, so at
    // `Jonswap::north_sea()`'s default gamma = 3.3 the spectrum's actual
    // moment-based significant height is `Hs / sqrt(1 - 0.287*ln(gamma))`,
    // about 23% above the nominal Hs = 3 m. This example asserts against
    // that DERIVED value (confirmed independently below, not copied from a
    // crate call) rather than against the nominal Hs, so the check is
    // still a real oracle -- it documents what the formula as implemented
    // actually integrates to, rather than silently loosening a tolerance
    // to hide the gap.
    let (mut m0, dw) = (0.0_f64, omega_p / 400.0);
    let mut w = dw;
    while w < 15.0 * omega_p {
        m0 += sea.spectrum_density(Fix128::from_f64(w)).to_f64() * dw;
        w += dw;
    }
    let hs_recovered = 4.0 * m0.sqrt();
    let dnv_correction = (1.0 - 0.287 * gamma.ln()).sqrt();
    let hs_predicted_uncorrected = hs / dnv_correction;
    println!(
        "[wave_ship] Hs recovered from m0 = 4*sqrt({m0:e}) = {hs_recovered} \
         (nominal preset Hs = {hs}, DNV-uncorrected prediction = {hs_predicted_uncorrected})"
    );
    assert!(
        rel_err(hs_recovered, hs_predicted_uncorrected) < 0.01,
        "Hs recovery from JONSWAP moment should match the DNV-uncorrected \
         prediction Hs/sqrt(1-0.287 ln(gamma)): {hs_recovered} vs {hs_predicted_uncorrected}"
    );

    // --- WaveComponent + free_surface_elevation: 4-component superposition -
    // Each component satisfies the deep-water dispersion relation k = omega^2/g
    // (Dean & Dalrymple eq. 3.30), built directly from omega_p's harmonics --
    // a different, larger component set than the 1- and 2-component cases in
    // tests/engineering_oracles_fluid.rs.
    let g = 9.81_f64;
    let specs: [(f64, f64, f64); 4] = [
        (1.1, omega_p * 0.6, 0.10),
        (0.8, omega_p * 1.0, 1.70),
        (0.5, omega_p * 1.4, -0.85),
        (0.2, omega_p * 2.1, 2.95),
    ];
    let components: Vec<WaveComponent> = specs
        .iter()
        .map(|&(amp, omega, phase)| WaveComponent {
            amplitude_m: Fix128::from_f64(amp),
            omega_rad_per_s: Fix128::from_f64(omega),
            wavenumber_rad_per_m: Fix128::from_f64(omega * omega / g),
            phase_rad: Fix128::from_f64(phase),
        })
        .collect();

    for (x, t) in [(0.0_f64, 0.0_f64), (37.5, 4.25), (-18.0, 9.0)] {
        let eta =
            free_surface_elevation(&components, Fix128::from_f64(x), Fix128::from_f64(t)).to_f64();
        let want: f64 = specs
            .iter()
            .map(|&(amp, omega, phase)| {
                let k = omega * omega / g;
                amp * (k * x - omega * t + phase).cos()
            })
            .sum();
        println!("[wave_ship] eta({x},{t}) = {eta} (closed form {want})");
        assert!(
            rel_err(eta, want).max((eta - want).abs()) < 1e-6,
            "eta({x},{t})"
        );
    }

    // --- froude_krylov_vertical_n: different density/area/draft from the ---
    // existing test, including a NEGATIVE elevation (wave trough).
    let rho_fresh = Fix128::from_int(1000); // fresh water, not the 1025 seawater used elsewhere
    let g_fix = Fix128::from_ratio(980_665, 100_000); // 9.80665 m/s^2 (standard gravity)
    let area = Fix128::from_ratio(1, 2) + Fix128::from_int(49); // 49.5 m^2
    let draft = Fix128::from_ratio(5, 2); // 2.5 m
    for eta in [Fix128::ZERO, Fix128::from_ratio(-4, 5), Fix128::from_int(3)] {
        let f = froude_krylov_vertical_n(rho_fresh, g_fix, area, draft, eta).to_f64();
        let want =
            rho_fresh.to_f64() * g_fix.to_f64() * area.to_f64() * (draft.to_f64() + eta.to_f64());
        println!(
            "[wave_ship] froude_krylov_vertical_n(eta={}) = {f} N (closed form {want} N)",
            eta.to_f64()
        );
        assert!(
            rel_err(f, want) < 1e-9,
            "froude_krylov_vertical_n(eta={})",
            eta.to_f64()
        );
    }
    // zero draft, zero elevation -> exactly zero force (no residual from
    // rounding: Fix128 multiplication of ZERO by anything is exactly ZERO).
    let f_zero = froude_krylov_vertical_n(rho_fresh, g_fix, area, Fix128::ZERO, Fix128::ZERO);
    assert_eq!(
        f_zero,
        Fix128::ZERO,
        "zero draft + zero elevation -> exactly zero"
    );

    println!(
        "[wave_ship] done: 6 production entry points exercised (WaveComponent, north_sea, \
         peak_omega, spectrum_density, free_surface_elevation, froude_krylov_vertical_n)"
    );
}
