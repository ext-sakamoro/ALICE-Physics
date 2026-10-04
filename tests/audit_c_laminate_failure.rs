//! Audit oracles for laminate_failure: the Tsai-Wu index reaches exactly one
//! at each of the five uniaxial strengths, and matches the documented
//! polynomial on combined states, to 1e-9 (AUD-C-S4W2-002).
//!
//! At `sigma_1 = Xt` alone, with `F1 = 1/Xt - 1/Xc` and `F11 = 1/(Xt Xc)`,
//! `FI = (1/Xt - 1/Xc) Xt + Xt^2 / (Xt Xc) = 1 - Xt/Xc + Xt/Xc = 1`; the same
//! cancellation gives one at `-Xc`, `Yt`, `-Yc` and `|tau_12| = S`. A material
//! with `Xt != Xc` and `Yt != Yc` is used so the linear terms do not vanish.

use alice_physics::laminate_failure::{
    failure_index, tsai_wu_failure_index, FailureCriterion, LaminateStrengths, StressState,
};
use alice_physics::math::Fix128;

fn stress(s1: i64, s2: i64, t12: i64) -> StressState {
    StressState {
        sigma_1: Fix128::from_int(s1),
        sigma_2: Fix128::from_int(s2),
        tau_12: Fix128::from_int(t12),
    }
}

fn materials() -> [(&'static str, LaminateStrengths); 3] {
    [
        ("gfrp_ud", LaminateStrengths::gfrp_ud()),
        ("cfrp_ud", LaminateStrengths::cfrp_ud()),
        (
            "custom",
            LaminateStrengths {
                xt: Fix128::from_int(2000),
                xc: Fix128::from_int(1200),
                yt: Fix128::from_int(50),
                yc: Fix128::from_int(200),
                s: Fix128::from_int(70),
            },
        ),
    ]
}

/// The documented polynomial, evaluated in `f64` from the strengths.
fn tsai_wu_reference(st: &LaminateStrengths, s1: f64, s2: f64, t: f64) -> f64 {
    let (xt, xc, yt, yc, s) = (
        st.xt.to_f64(),
        st.xc.to_f64(),
        st.yt.to_f64(),
        st.yc.to_f64(),
        st.s.to_f64(),
    );
    let f1 = 1.0 / xt - 1.0 / xc;
    let f2 = 1.0 / yt - 1.0 / yc;
    let f11 = 1.0 / (xt * xc);
    let f22 = 1.0 / (yt * yc);
    let f66 = 1.0 / (s * s);
    let f12 = -0.5 * (f11 * f22).sqrt();
    f1 * s1 + f2 * s2 + f11 * s1 * s1 + f22 * s2 * s2 + f66 * t * t + 2.0 * f12 * s1 * s2
}

#[test]
fn tsai_wu_is_exactly_one_at_each_uniaxial_strength() {
    for (name, st) in materials() {
        let xt = st.xt.to_f64() as i64;
        let xc = st.xc.to_f64() as i64;
        let yt = st.yt.to_f64() as i64;
        let yc = st.yc.to_f64() as i64;
        let s = st.s.to_f64() as i64;
        for (case, state) in [
            ("sigma_1 = Xt", stress(xt, 0, 0)),
            ("sigma_1 = -Xc", stress(-xc, 0, 0)),
            ("sigma_2 = Yt", stress(0, yt, 0)),
            ("sigma_2 = -Yc", stress(0, -yc, 0)),
            ("tau_12 = S", stress(0, 0, s)),
            ("tau_12 = -S", stress(0, 0, -s)),
        ] {
            let fi = tsai_wu_failure_index(st, state).to_f64();
            assert!(
                (fi - 1.0).abs() <= 1e-9,
                "{name}, {case}: FI = {fi:.15}, closed form 1"
            );
            let via = failure_index(FailureCriterion::TsaiWu, st, state).to_f64();
            assert!(
                (via - fi).abs() <= 1e-12,
                "{name}, {case}: failure_index(TsaiWu) = {via}, direct {fi}"
            );
        }
    }
}

#[test]
fn tsai_wu_at_half_strength_has_the_closed_form_below_one() {
    // sigma_1 = Xt / 2: FI = F1 Xt / 2 + F11 Xt^2 / 4 = 1/2 - Xt/(4 Xc)
    for (name, st) in materials() {
        let xt = st.xt.to_f64();
        let xc = st.xc.to_f64();
        let fi = tsai_wu_failure_index(st, stress(xt as i64 / 2, 0, 0)).to_f64();
        let want = 0.5 - xt / (4.0 * xc);
        assert!(
            (fi - want).abs() <= 1e-9,
            "{name}: FI(Xt/2) = {fi:.15}, closed form {want:.15}"
        );
    }
}

#[test]
fn tsai_wu_matches_the_documented_polynomial_on_combined_states() {
    // Combined states exercise the interaction term 2 F12 sigma_1 sigma_2 and
    // its sign (same-sign and opposite-sign normal stresses).
    for (name, st) in materials() {
        for (s1, s2, t) in [
            (300i64, 20i64, 15i64),
            (300, -60, 15),
            (-400, 25, -30),
            (-250, -90, 40),
            (0, 0, 0),
        ] {
            let fi = tsai_wu_failure_index(st, stress(s1, s2, t)).to_f64();
            let want = tsai_wu_reference(&st, s1 as f64, s2 as f64, t as f64);
            assert!(
                (fi - want).abs() <= 1e-9,
                "{name}, ({s1}, {s2}, {t}): FI = {fi:.15}, polynomial {want:.15}"
            );
        }
    }
}
