//! Audit oracles for `creep_longterm`: `predict_strain` far above `T_g`,
//! where the WLF shift factor falls to a few units of `2^-64` and then to
//! zero in the deterministic exponential.
//!
//! Expected values are the closed forms of the module doc, evaluated in f64:
//! Findley `e(t) = e0 + m t^n` at the shifted time `t / a_T`, with WLF
//! `log10 a_T = -C1 dT / (C2 + dT)`, `C1 = 17.44`, `C2 = 51.6`, referenced
//! to `T_g`. `m = 2^-60` and `n = 1` keep the creep term exact and of order
//! one at the shifted times reached here.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::creep_longterm::{predict_strain, FindleyParameters};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

const C1: f64 = 17.44;
const C2: f64 = 51.6;

fn params() -> FindleyParameters {
    FindleyParameters {
        epsilon_0: Fix128::from_ratio(1, 1000),
        m: Fix128::from_raw(0, 1 << 4), // 2^-60
        n_int: 1,
    }
}

/// `e0 + m t / a_T`, the closed form at `T_g + dt`.
fn closed_form(dt: f64, t_h: f64) -> f64 {
    let a_t = 10f64.powf(-C1 * dt / (C2 + dt));
    0.001 + 2f64.powi(-60) * t_h / a_t
}

fn strain_at_offset(dt: i64) -> f64 {
    let pla = MaterialProperties::pla();
    let t = pla.glass_transition_c + Fix128::from_int(dt);
    predict_strain(&params(), &pla, Fix128::ONE, t).to_f64()
}

/// `T_g + 13000`: `log10 a_T = -17.3711`, `a_T = 4.26e-18`, about 79 units
/// of `2^-64`. The shifted time is `2.35e17 h` and the creep term `0.204`.
/// The tolerance is the representation, not the model: `a_T` carries a
/// truncation of up to one unit in 79, so 3 % covers it with margin.
#[test]
fn strain_far_above_tg_matches_the_wlf_closed_form_while_a_t_is_representable() {
    let dt = 13_000;
    let got = strain_at_offset(dt);
    let want = closed_form(dt as f64, 1.0);
    assert!(
        ((got - want) / want).abs() <= 0.03,
        "dT = {dt}: got {got}, closed form {want}"
    );
}

/// `T_g + 13200`: `log10 a_T = -17.3721`, `a_T = 4.24e-18`, still about 78
/// units of `2^-64` and still positive, so the closed form is
/// `e0 + 2^-60 / a_T = 0.2053`. The doc of the shift factor says creep
/// accelerates above the reference temperature. The current source returns
/// `e0`, no creep at all, because the exponential saturates to zero below
/// an argument of -40 and a zero shift factor is mapped to a zero effective
/// time.
#[test]
// AUD-A-S34-031
fn strain_far_above_tg_does_not_collapse_to_the_elastic_strain() {
    let dt = 13_200;
    let got = strain_at_offset(dt);
    let want = closed_form(dt as f64, 1.0);
    assert!(
        ((got - want) / want).abs() <= 0.03,
        "dT = {dt}: got {got}, closed form {want}"
    );
}

/// The doc of the shift factor: above the reference temperature creep
/// accelerates, so at a fixed time the strain does not decrease with
/// temperature. Checked across the point where the shift factor stops being
/// representable.
#[test]
// AUD-A-S34-031
fn strain_is_monotone_in_temperature_across_the_shift_factor_underflow() {
    let mut previous = 0.0_f64;
    for dt in (12_000..=14_000).step_by(100) {
        let got = strain_at_offset(dt);
        assert!(got >= previous, "dT = {dt}: strain {got} after {previous}");
        previous = got;
    }
}

/// At `T_g + 13200` the speedup `1 / a_T` is `2.4e17`; `t = 1e5 h` makes the
/// effective time `2.4e22 h`, past the Fix128 range: it saturates instead of
/// wrapping or dropping to zero, so the strain does not fall as time grows.
#[test]
fn an_effective_time_past_the_range_saturates() {
    let pla = MaterialProperties::pla();
    let t = pla.glass_transition_c + Fix128::from_int(13_200);
    let short = predict_strain(&params(), &pla, Fix128::ONE, t).to_f64();
    let long = predict_strain(&params(), &pla, Fix128::from_int(100_000), t).to_f64();
    assert!(
        long >= short && long > 0.25,
        "t = 1e5 h: {long} after {short}"
    );
}
