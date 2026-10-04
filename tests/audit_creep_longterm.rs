//! Audit oracles (S2-1) for `alice_physics::creep_longterm`.
//!
//! Expected values are closed forms evaluated in f64 from the module doc
//! (Findley `e(t) = e0 + m t^n`, WLF `log10 a_T = -C1 dT / (C2 + dT)` with
//! C1 = 17.44, C2 = 51.6, referenced to T_g) -- never from the function under
//! test. `known defect` tests are `#[ignore]`d and recorded in the audit
//! ledger; they are not fixed here.

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::field_reassign_with_default,
    clippy::unnecessary_map_or,
    clippy::needless_range_loop
)]

use alice_physics::creep_longterm::{predict_strain, FindleyParameters};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

fn p(e0: f64, m: f64, n: u32) -> FindleyParameters {
    // exact binary-friendly construction is not needed: tolerances below
    // include the 2^-64 truncation of `m`.
    FindleyParameters {
        epsilon_0: Fix128::from_f64(e0),
        m: Fix128::from_f64(m),
        n_int: n,
    }
}

fn strain(par: &FindleyParameters, t_h: i64, temp_c: i64) -> f64 {
    predict_strain(
        par,
        &MaterialProperties::pla(),
        Fix128::from_int(t_h),
        Fix128::from_int(temp_c),
    )
    .to_f64()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// Doc: PLA preset e0 = 0.3 %, total ~1.0 % at 6 months (4380 h) at 25 C
/// (below T_g = 60 C so no WLF shift), ~6 % at 1 year.
#[test]
fn pla_preset_matches_documented_six_month_and_one_year_strain() {
    let par = FindleyParameters::pla_25c_moderate();
    assert_eq!(par.n_int, 3);
    assert!(rel(par.epsilon_0.to_f64(), 0.003) < 1e-12);
    assert!(
        rel(par.m.to_f64(), 8.3e-14) < 1e-6,
        "m = {}",
        par.m.to_f64()
    );
    let half = strain(&par, 4380, 25);
    let want_half = 0.003 + 8.3e-14 * 4380f64 * 4380f64 * 4380f64;
    assert!(rel(half, want_half) < 1e-5, "6 mo {half} vs {want_half}");
    assert!(
        (0.0099..0.0101).contains(&half),
        "doc says total ~1.0 % at 6 months, got {half}"
    );
    let year = strain(&par, 8760, 25);
    assert!(
        (0.057..0.061).contains(&year),
        "doc says ~6 % at 1 yr: {year}"
    );
}

/// Doc: "under-predicts before and over-predicts after" the 6-month match
/// relative to the literature's 0.25 exponent. Build the same-anchored n=0.25
/// law (e0 + c t^0.25 through 1 % at 4380 h) and check the sign pattern.
#[test]
fn n3_fit_underpredicts_before_and_overpredicts_after_six_months_vs_quarter_power() {
    let par = FindleyParameters::pla_25c_moderate();
    let c = (0.01 - 0.003) / 4380f64.powf(0.25);
    for t in [100, 500, 1000, 2000] {
        let lit = 0.003 + c * (t as f64).powf(0.25);
        assert!(strain(&par, t, 25) < lit, "t={t} should under-predict");
    }
    for t in [8000, 8760, 20000] {
        let lit = 0.003 + c * (t as f64).powf(0.25);
        assert!(strain(&par, t, 25) > lit, "t={t} should over-predict");
    }
}

/// n_int in 1..=4: e0 + m t^n evaluated by closed form (T below T_g).
#[test]
fn findley_power_law_for_each_documented_integer_exponent() {
    for n in 1u32..=4 {
        let par = p(0.002, 1e-6, n);
        for t in [1i64, 3, 10, 25] {
            let want = 0.002 + 1e-6 * (t as f64).powi(n as i32);
            let got = strain(&par, t, 25);
            assert!(rel(got, want) < 1e-9, "n={n} t={t}: {got} vs {want}");
        }
    }
}

/// Doc (via strain_at): t <= 0 returns e0; negative time is not extrapolated.
#[test]
fn nonpositive_time_returns_epsilon_0_at_every_temperature() {
    let par = FindleyParameters::pla_25c_moderate();
    for temp in [25, 60, 70, 90] {
        for t in [0i64, -1, -4380] {
            assert_eq!(
                predict_strain(
                    &par,
                    &MaterialProperties::pla(),
                    Fix128::from_int(t),
                    Fix128::from_int(temp)
                ),
                par.epsilon_0,
                "t={t} T={temp}"
            );
        }
    }
}

/// Doc: "At or below T_g the parameters are used as calibrated": identical
/// strain for any temperature <= T_g (PLA T_g = 60 C).
#[test]
fn at_or_below_glass_transition_strain_is_temperature_independent() {
    let par = FindleyParameters::pla_25c_moderate();
    let base = predict_strain(
        &par,
        &MaterialProperties::pla(),
        Fix128::from_int(4380),
        Fix128::from_int(60),
    );
    for temp in [-40, 0, 25, 59] {
        let got = predict_strain(
            &par,
            &MaterialProperties::pla(),
            Fix128::from_int(4380),
            Fix128::from_int(temp),
        );
        assert_eq!(got, base, "T={temp}");
    }
}

/// Doc: above T_g the effective time is t / a_T with
/// log10 a_T = -C1 dT / (C2 + dT). Closed form in f64 for a range of dT where
/// t_eff^n stays inside Fix128's integer range.
#[test]
fn above_tg_matches_wlf_closed_form() {
    let par = p(0.003, 1e-12, 3);
    let (c1, c2) = (17.44f64, 51.6f64);
    let mut compared = 0;
    for (t, dt) in [
        (1i64, 1i64),
        (1, 5),
        (10, 3),
        (100, 2),
        (100, 5),
        (1000, 1),
        (1000, 2),
        (4380, 1),
    ] {
        let a_t = 10f64.powf(-c1 * dt as f64 / (c2 + dt as f64));
        let t_eff = t as f64 / a_t;
        let want_creep = 1e-12 * t_eff.powi(3);
        let got = strain(&par, t, 60 + dt) - 0.003;
        assert!(
            rel(got, want_creep) < 1e-5,
            "t={t} dT={dt}: creep {got:e} vs {want_creep:e}"
        );
        compared += 1;
    }
    assert_eq!(compared, 8);
}

/// Tight version of the WLF closed form: `m = 2^-40` and `e0 = 0` are exactly
/// representable, so the only error left is the module's 9-digit `ln 10`
/// (1.3e-9 relative -> about 1.5e-8 on a cubed effective time) plus
/// `exp_fix` truncation; tolerance 5e-8 pins the constants' digits.
#[test]
fn above_tg_wlf_closed_form_tight_tolerance() {
    let m = 1.0 / (1u64 << 40) as f64;
    let par = p(0.0, m, 3);
    let (c1, c2) = (17.44f64, 51.6f64);
    for (t, dt) in [(1i64, 3i64), (1, 5), (2, 4), (10, 2), (10, 3)] {
        let log10_a = -c1 * dt as f64 / (c2 + dt as f64);
        let t_eff = t as f64 * 10f64.powf(-log10_a);
        let want = m * t_eff.powi(3);
        let got = strain(&par, t, 60 + dt);
        assert!(
            rel(got, want) < 5e-8,
            "t={t} dT={dt}: {got:e} vs {want:e} (rel {:.2e})",
            rel(got, want)
        );
    }
}

/// Physical sanity within the safely representable range: strain never
/// decreases when the temperature rises (a_T < 1 -> larger effective time).
#[test]
fn strain_is_monotone_in_temperature_just_above_tg() {
    let par = FindleyParameters::pla_25c_moderate();
    for t in [10i64, 100, 1000, 4380] {
        let mut prev = 0.0;
        for temp in 55..=65 {
            let s = strain(&par, t, temp);
            assert!(s >= prev, "t={t} T={temp}: {s} < {prev}");
            prev = s;
        }
    }
}

/// Strain never decreases with time at a fixed temperature.
#[test]
fn strain_is_monotone_in_time() {
    let par = FindleyParameters::pla_25c_moderate();
    for temp in [25, 62, 65] {
        let mut prev = 0.0;
        for t in [0i64, 1, 10, 100, 500, 1000, 4380, 8760] {
            let s = strain(&par, t, temp);
            assert!(s >= prev, "T={temp} t={t}: {s} < {prev}");
            prev = s;
        }
    }
}

/// A material with no glass transition (T_g == 0) skips the WLF shift and
/// uses the calibrated parameters directly (pins the `t_g.is_zero()` guard;
/// not stated in the docs).
#[test]
fn zero_glass_transition_means_no_wlf_shift() {
    let par = FindleyParameters::pla_25c_moderate();
    let mut m = MaterialProperties::pla();
    m.glass_transition_c = Fix128::ZERO;
    let hot = predict_strain(&par, &m, Fix128::from_int(4380), Fix128::from_int(90));
    let cold = predict_strain(&par, &m, Fix128::from_int(4380), Fix128::from_int(25));
    assert_eq!(hot, cold);
}

/// KNOWN DEFECT: the doc gives the WLF domain as T_g <= T <= T_g + 100 C, but
/// `t_eff^n` overflows Fix128's 64-bit integer part from roughly T_g + 15 C
/// (PLA: about 75 C at 100 h, about 71 C at 4380 h) and wraps silently: strain turns
/// non-monotone in T and negative (measured: t = 1000 h, T = 100 C ->
/// -4.4e5; t = 100 h, 70 C -> 25.9, 80 C -> 4.8e5, 100 C -> 5.8e5 against
/// a true 3.4e7 / 5.8e15). A monotone, non-negative result is required of any
/// finite-strain model.
#[test]
fn strain_is_monotone_and_nonnegative_across_documented_wlf_domain() {
    let par = FindleyParameters::pla_25c_moderate();
    for t in [100i64, 1000, 4380] {
        let mut prev = 0.0;
        for temp in (60..=160).step_by(5) {
            let s = strain(&par, t, temp);
            assert!(s >= 0.003 * 0.99, "t={t} T={temp}: strain {s} below e0");
            assert!(s >= prev, "t={t} T={temp}: {s} < {prev} (non-monotone)");
            prev = s;
        }
    }
}
