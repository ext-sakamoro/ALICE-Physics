//! Independent oracles for `atmosphere::Isa1976`, derived without the layer
//! closed forms used by `analytic_atmosphere_isa1976.rs`.
//!
//! # Derivations used here
//!
//! - Hydrostatic balance `dp/dH = −ρ g₀`: a central difference of the crate's
//!   pressure is compared with the crate's own density times `g₀`. This ties
//!   `p` and `ρ` together through a law the implementation never evaluates.
//! - Numerical integration: `d ln p / dH = −g₀ M₀ / (R* T(H))` is integrated
//!   with composite Simpson quadrature of `1/T` (2 000 panels per layer), so
//!   the expected pressure does not use the power law `(T/T₀)^(g₀M₀/(R*L))`.
//! - Density power law `ρ/ρ₀ = (T/T₀)^(n − 1)`, `n = g₀ M₀ / (R* L)`, a
//!   different exponent from the pressure law.
//! - Speed of sound from the state, `a² = γ p / ρ` (not from `T`).
//! - Altitudes 2 345, 6 543, 8 765, 14 321, 17 777 m and the layer and range
//!   boundaries themselves; none of these points appears in the existing file.
//!
//! # Tolerances
//!
//! `Fix128::powf_pos` / `exp` are documented at relative `1e-6`. Each bound
//! below states the margin it allows over that.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `ln` / `exp` / `powf` compute references outside the crate.
#![allow(clippy::disallowed_methods)]

use alice_physics::atmosphere::{AtmosphereState, Isa1976, IsaError};
use alice_physics::math::Fix128;

const T0: f64 = 288.15;
const P0: f64 = 101_325.0;
const L: f64 = 0.0065;
const G0: f64 = 9.806_65;
const R: f64 = 8.314_32;
const M: f64 = 0.028_964_4;
const GAMMA: f64 = 1.4;
const R0: f64 = 6_356_766.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn at(h: f64) -> AtmosphereState {
    Isa1976::at_geopotential_altitude(fx(h)).expect("in range")
}

fn rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = ((actual - expected) / expected).abs();
    assert!(
        err <= tol,
        "{label}: {actual} vs {expected} (rel err {err:e} > {tol:e})"
    );
}

/// Temperature profile written directly from the layer definition.
fn temperature(h: f64) -> f64 {
    if h <= 11_000.0 {
        T0 - L * h
    } else {
        T0 - L * 11_000.0
    }
}

/// `∫₀^H dh / T(h)` by composite Simpson with `panels` panels per layer.
fn integral_inv_t(h: f64, panels: usize) -> f64 {
    let simpson = |a: f64, b: f64| -> f64 {
        if b <= a {
            return 0.0;
        }
        let n = panels; // even
        let step = (b - a) / n as f64;
        let mut s = 1.0 / temperature(a) + 1.0 / temperature(b);
        for k in 1..n {
            let w = if k % 2 == 1 { 4.0 } else { 2.0 };
            s += w / temperature(a + step * k as f64);
        }
        s * step / 3.0
    };
    simpson(0.0, h.min(11_000.0)) + simpson(11_000.0, h.max(11_000.0))
}

#[test]
fn pressure_matches_numerically_integrated_hydrostatic_equation() {
    // oracle: ln(p/p₀) = −(g₀ M₀ / R*) ∫₀^H dh / T(h), Simpson quadrature.
    // Simpson error for 1/T over 11 km with 2000 panels is below 1e-14
    // relative; the bound 1e-5 is ten times the documented powf/exp error.
    for h in [2_345.0, 8_765.0, 11_000.0, 14_321.0, 17_777.0, 20_000.0] {
        let expected = P0 * (-(G0 * M / R) * integral_inv_t(h, 2_000)).exp();
        rel_close(
            &format!("p({h})"),
            at(h).pressure_pa.to_f64(),
            expected,
            1e-5,
        );
    }
}

#[test]
fn central_difference_of_pressure_obeys_hydrostatic_balance() {
    // oracle: dp/dH = −ρ g₀. Central difference with δ = 50 m: truncation
    // (δ/H_s)²/6 ≈ 1e-5 with scale height H_s ≈ 6.3 km at 20 km; the
    // pressure error 1e-6 relative over a 100 m baseline amplifies to about
    // 1e-6 · H_s / 100 m ≈ 6e-5 (bound 5e-4).
    let delta = 50.0;
    for h in [2_345.0, 6_543.0, 8_765.0, 14_321.0, 17_777.0] {
        let dp = (at(h + delta).pressure_pa.to_f64() - at(h - delta).pressure_pa.to_f64())
            / (2.0 * delta);
        let rho_g = at(h).density_kg_m3.to_f64() * G0;
        rel_close(&format!("dp/dH at {h}"), -dp, rho_g, 5e-4);
    }
}

#[test]
fn hydrostatic_gradient_is_continuous_across_the_tropopause() {
    // ρ is continuous at 11 km, so the one-sided derivatives of p from the
    // two layer formulas must both equal −ρ(11 km) g₀. One-sided difference
    // with δ = 20 m: first-order error δ/(2 H_s) ≈ 1.6e-3 (bound 3e-3).
    let delta = 20.0;
    let p11 = at(11_000.0).pressure_pa.to_f64();
    let rho_g = at(11_000.0).density_kg_m3.to_f64() * G0;
    let below = (p11 - at(11_000.0 - delta).pressure_pa.to_f64()) / delta;
    let above = (at(11_000.0 + delta).pressure_pa.to_f64() - p11) / delta;
    rel_close("left dp/dH", -below, rho_g, 3e-3);
    rel_close("right dp/dH", -above, rho_g, 3e-3);
    // Temperature: left slope −L, right slope 0.
    let t11 = at(11_000.0).temperature_k.to_f64();
    rel_close(
        "left dT/dH",
        (t11 - at(11_000.0 - delta).temperature_k.to_f64()) / delta,
        -L,
        1e-9,
    );
    assert_eq!(
        at(11_000.0 + delta).temperature_k,
        at(11_000.0).temperature_k,
        "isothermal right of the tropopause, bit for bit"
    );
}

#[test]
fn tropopause_temperature_is_the_same_value_from_both_layers() {
    // The layer-1 temperature is the layer-0 expression at 11 000 m, so the
    // values just above the boundary are bit-identical to the boundary value.
    let base = at(11_000.0).temperature_k;
    assert_eq!(
        Isa1976::at_geopotential_altitude(Fix128::from_raw(11_000, 1))
            .unwrap()
            .temperature_k,
        base
    );
    assert!(at(10_999.999).temperature_k > base);
}

#[test]
fn density_follows_its_own_power_law_in_the_troposphere() {
    // oracle: ρ/ρ₀ = (T/T₀)^(n − 1), n = g₀ M₀ / (R* L) ≈ 5.2559.
    let n = G0 * M / (R * L);
    let rho0 = at(0.0).density_kg_m3.to_f64();
    for h in [2_345.0, 6_543.0, 10_000.0] {
        let t = temperature(h);
        let expected = rho0 * (t / T0).powf(n - 1.0);
        rel_close(
            &format!("rho({h})"),
            at(h).density_kg_m3.to_f64(),
            expected,
            1e-5,
        );
    }
}

#[test]
fn speed_of_sound_follows_from_pressure_and_density() {
    // oracle: a² = γ p / ρ (ideal gas), and a ∝ √T.
    let a0 = at(0.0).speed_of_sound_m_s.to_f64();
    for h in [2_345.0, 8_765.0, 14_321.0] {
        let s = at(h);
        let from_state = (GAMMA * s.pressure_pa.to_f64() / s.density_kg_m3.to_f64()).sqrt();
        rel_close(
            &format!("a({h}) vs γp/ρ"),
            s.speed_of_sound_m_s.to_f64(),
            from_state,
            1e-9,
        );
        rel_close(
            &format!("a({h})/a0"),
            s.speed_of_sound_m_s.to_f64() / a0,
            (temperature(h) / T0).sqrt(),
            1e-9,
        );
    }
}

#[test]
fn profiles_are_monotone_over_the_whole_range() {
    // p and ρ strictly fall with altitude; T and a never rise in 0..=20 km.
    let mut prev = at(0.0);
    let mut h = 250;
    while h <= 20_000 {
        let s = Isa1976::at_geopotential_altitude(Fix128::from_int(h)).unwrap();
        assert!(s.pressure_pa < prev.pressure_pa, "p not falling at {h}");
        assert!(s.density_kg_m3 < prev.density_kg_m3, "ρ not falling at {h}");
        assert!(s.temperature_k <= prev.temperature_k, "T rising at {h}");
        assert!(
            s.speed_of_sound_m_s <= prev.speed_of_sound_m_s,
            "a rising at {h}"
        );
        assert!(s.density_kg_m3 > Fix128::ZERO);
        prev = s;
        h += 250;
    }
}

#[test]
fn range_edges_one_raw_unit_outside_are_errors() {
    // One raw unit (2⁻⁶⁴ m) beyond either end is rejected and carries the
    // rejected value; the ends themselves are accepted.
    let above = Fix128::from_raw(20_000, 1);
    let below = Fix128::from_raw(-1, u64::MAX); // −2⁻⁶⁴
    for h in [above, below] {
        assert_eq!(
            Isa1976::at_geopotential_altitude(h),
            Err(IsaError::AltitudeOutOfRange {
                geopotential_altitude_m: h
            })
        );
    }
    assert!(Isa1976::at_geopotential_altitude(Fix128::from_int(20_000)).is_ok());
    assert!(Isa1976::at_geopotential_altitude(Fix128::ZERO).is_ok());
}

#[test]
fn geometric_top_of_range_is_at_the_inverse_conversion() {
    // oracle: Z_top = r₀ H / (r₀ − H) for H = 20 000 m → 20 063.12 m.
    let z_top = R0 * 20_000.0 / (R0 - 20_000.0);
    assert!(Isa1976::at_geometric_altitude(fx(z_top - 0.01)).is_ok());
    assert!(matches!(
        Isa1976::at_geometric_altitude(fx(z_top + 0.01)),
        Err(IsaError::AltitudeOutOfRange { .. })
    ));
    // At Z = 0 the two entry points agree bit for bit.
    assert_eq!(
        Isa1976::at_geometric_altitude(Fix128::ZERO),
        Isa1976::at_geopotential_altitude(Fix128::ZERO)
    );
}

#[test]
fn negative_geometric_altitude_error_carries_the_converted_altitude() {
    // Documented: the error carries the geopotential value of the input.
    let z = Fix128::from_int(-1_234);
    let h = Isa1976::geopotential_altitude_m(z);
    rel_close(
        "H(−1234)",
        h.to_f64(),
        R0 * -1_234.0 / (R0 - 1_234.0),
        1e-12,
    );
    assert_eq!(
        Isa1976::at_geometric_altitude(z),
        Err(IsaError::AltitudeOutOfRange {
            geopotential_altitude_m: h
        })
    );
}
