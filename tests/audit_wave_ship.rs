//! Audit S2-3 oracles for `alice_physics::wave_ship`.
//! Expected values are hand-evaluated closed forms in f64 (DNV-RP-C205 JONSWAP, linear
//! superposition, Archimedes, damped SDOF), not values produced by the implementation.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::wave_ship::{
    free_surface_elevation, froude_krylov_vertical_n, Jonswap, ShipResponse, WaveComponent,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Independent f64 evaluation of the cited JONSWAP form. Fix128::exp is documented to ~1e-6
/// relative error (measured 4e-8 here), so spectrum comparisons use 1e-6.
fn jonswap_f64(hs: f64, tp: f64, gamma: f64, w: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let wp = 2.0 * std::f64::consts::PI / tp;
    let a = 5.0 / 16.0 * (1.0 - 0.287 * gamma.ln());
    let sigma = if w <= wp { 0.07 } else { 0.09 };
    let rr = (-(w - wp) * (w - wp) / (2.0 * sigma * sigma * wp * wp)).exp();
    a * hs * hs * wp.powi(4) / w.powi(5) * (-1.25 * (wp / w).powi(4)).exp() * gamma.powf(rr)
}

fn rel_close(got: Fix128, want: f64, tol: f64) -> bool {
    let g = got.to_f64();
    (g - want).abs() <= tol * want.abs().max(1e-30)
}

#[test]
fn peak_omega_is_two_pi_over_tp() {
    for tp in [3.0, 8.0, 9.0, 12.5] {
        let j = Jonswap {
            significant_wave_height_m: Fix128::ONE,
            peak_period_s: fx(tp),
            gamma: fx(3.3),
        };
        let want = 2.0 * std::f64::consts::PI / tp;
        assert!(
            (j.peak_omega().to_f64() - want).abs() < 1e-12,
            "tp {tp}: {} vs {want}",
            j.peak_omega().to_f64()
        );
    }
}

#[test]
fn degenerate_period_and_frequency_give_zero() {
    let mut j = Jonswap::north_sea();
    assert_eq!(j.spectrum_density(Fix128::ZERO), Fix128::ZERO);
    assert_eq!(j.spectrum_density(-Fix128::ONE), Fix128::ZERO);
    j.peak_period_s = Fix128::ZERO;
    assert_eq!(j.peak_omega(), Fix128::ZERO);
    assert_eq!(j.spectrum_density(Fix128::ONE), Fix128::ZERO);
}

#[test]
fn spectrum_matches_dnv_closed_form_across_sea_states() {
    let cases = [
        (3.0, 9.0, 3.3),
        (1.5, 6.0, 1.0),
        (6.0, 12.0, 5.0),
        (2.0, 7.0, 2.0),
    ];
    for (hs, tp, g) in cases {
        let j = Jonswap {
            significant_wave_height_m: fx(hs),
            peak_period_s: fx(tp),
            gamma: fx(g),
        };
        let wp = 2.0 * std::f64::consts::PI / tp;
        for k in [0.5, 0.8, 0.95, 1.0, 1.05, 1.3, 2.0, 3.5] {
            let w = wp * k;
            let want = jonswap_f64(hs, tp, g, w);
            let got = j.spectrum_density(fx(w));
            assert!(
                rel_close(got, want, 1e-6),
                "hs {hs} tp {tp} g {g} w/wp {k}: {} vs {want}",
                got.to_f64()
            );
        }
    }
}

/// Peak value S(w_p) = A*Hs^2/w_p * exp(-5/4) * gamma with A = 5/16 (1 - 0.287 ln gamma).
#[test]
fn peak_value_closed_form_and_argmax_at_peak_omega() {
    let j = Jonswap::north_sea();
    let wp = j.peak_omega().to_f64();
    let want = 5.0 / 16.0 * (1.0 - 0.287 * 3.3f64.ln()) * 9.0 / wp * (-1.25f64).exp() * 3.3;
    assert!(rel_close(j.spectrum_density(j.peak_omega()), want, 1e-6));
    let mut best = (0.0, 0.0);
    let n = 2000;
    for i in 1..=n {
        let w = 0.2 * wp + (4.0 - 0.2) * wp * f64::from(i) / f64::from(n);
        let s = j.spectrum_density(fx(w)).to_f64();
        if s > best.1 {
            best = (w, s);
        }
    }
    assert!(
        (best.0 - wp).abs() < 2.0 * 3.8 * wp / f64::from(n),
        "argmax {} vs wp {wp}",
        best.0
    );
}

/// Hs = 4 sqrt(m0) is recovered under every gamma (DNV approximation, 2 % band).
#[test]
fn significant_wave_height_recovered_from_zeroth_moment_for_several_gamma() {
    for g in [1.0, 2.0, 3.3, 5.0, 7.0] {
        let j = Jonswap {
            significant_wave_height_m: fx(3.0),
            peak_period_s: fx(9.0),
            gamma: fx(g),
        };
        let wp = j.peak_omega().to_f64();
        let (lo, hi, n) = (0.05 * wp, 30.0 * wp, 8000);
        let h = (hi - lo) / f64::from(n);
        let mut m0 = 0.0;
        for i in 0..=n {
            let w = lo + h * f64::from(i);
            let s = j.spectrum_density(fx(w)).to_f64();
            m0 += if i == 0 || i == n { 0.5 * s } else { s };
        }
        m0 *= h;
        let hs = 4.0 * m0.sqrt();
        assert!((hs - 3.0).abs() / 3.0 < 0.02, "gamma {g}: Hs {hs}");
    }
}

/// sigma switches 0.07 -> 0.09 at w_p: the spectrum is continuous there and the high side
/// of the peak is wider (S(wp + d) > S(wp - d) after removing the PM asymmetry).
#[test]
fn sigma_switch_is_continuous_and_high_side_is_wider() {
    let j = Jonswap::north_sea();
    let wp = j.peak_omega().to_f64();
    let eps = 1e-6;
    let below = j.spectrum_density(fx(wp - eps)).to_f64();
    let above = j.spectrum_density(fx(wp + eps)).to_f64();
    let at = j.spectrum_density(j.peak_omega()).to_f64();
    assert!((below - at).abs() / at < 1e-5 && (above - at).abs() / at < 1e-5);
    let d = 0.05 * wp;
    let pm = |w: f64| jonswap_f64(3.0, 9.0, 1.0001, w);
    let ratio_hi = j.spectrum_density(fx(wp + d)).to_f64() / pm(wp + d);
    let ratio_lo = j.spectrum_density(fx(wp - d)).to_f64() / pm(wp - d);
    assert!(ratio_hi > ratio_lo);
}

/// gamma <= 0 is documented to behave exactly as gamma = 1 (both places).
#[test]
fn nonpositive_gamma_is_treated_as_one() {
    let mk = |g: Fix128| Jonswap {
        significant_wave_height_m: Fix128::from_int(3),
        peak_period_s: Fix128::from_int(9),
        gamma: g,
    };
    let one = mk(Fix128::ONE);
    for g in [Fix128::ZERO, -Fix128::from_int(2), -r(1, 3)] {
        let j = mk(g);
        for w in [0.4, 0.7, 1.2, 3.0] {
            assert_eq!(
                j.spectrum_density(fx(w)),
                one.spectrum_density(fx(w)),
                "g {g:?} w {w}"
            );
        }
    }
}

/// For w >> w_p the spectrum is the PM tail `A 5/16 Hs^2 wp^4 / w^5`: halving w scales by 32.
#[test]
fn high_frequency_tail_decays_as_omega_minus_five() {
    let j = Jonswap::north_sea();
    let wp = j.peak_omega().to_f64();
    let s1 = j.spectrum_density(fx(8.0 * wp)).to_f64();
    let s2 = j.spectrum_density(fx(16.0 * wp)).to_f64();
    assert!((s1 / s2 - 32.0).abs() / 32.0 < 1e-3, "ratio {}", s1 / s2);
}

/// Low-frequency side: the spectrum vanishes for w -> 0. It must stay in [0, S(w_p)] and
/// essentially zero over many decades (exp(-5/4 (wp/w)^4)).
#[test]
fn low_frequency_side_stays_nonnegative_and_below_peak() {
    let j = Jonswap::north_sea();
    let peak = j.spectrum_density(j.peak_omega());
    let wp = j.peak_omega().to_f64();
    for e in [
        -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0, -7.0,
    ] {
        let w = wp * 10f64.powf(e);
        let s = j.spectrum_density(fx(w));
        assert!(s >= Fix128::ZERO, "w = wp*1e{e}: S = {}", s.to_f64());
        assert!(s < peak, "w = wp*1e{e}: S = {}", s.to_f64());
        assert!(s.to_f64() < 1e-9, "w = wp*1e{e}: S = {}", s.to_f64());
    }
}

fn comp(a: f64, w: f64, k: f64, p: f64) -> WaveComponent {
    WaveComponent {
        amplitude_m: fx(a),
        omega_rad_per_s: fx(w),
        wavenumber_rad_per_m: fx(k),
        phase_rad: fx(p),
    }
}

/// eta = sum A cos(k x - w t + phi): value, sign of the x and t terms, and phase.
#[test]
fn free_surface_matches_cosine_sum_with_signs() {
    let cs = [
        comp(1.5, 1.2, 0.8, 0.3),
        comp(0.7, 2.1, 1.9, -1.1),
        comp(0.2, 0.5, 0.1, 2.0),
    ];
    for &(x, t) in &[
        (0.0, 0.0),
        (1.3, 0.0),
        (0.0, 1.0),
        (2.2, 0.7),
        (-3.1, 4.4),
        (5.0, -2.0),
    ] {
        let want: f64 = cs
            .iter()
            .map(|c| {
                c.amplitude_m.to_f64()
                    * (c.wavenumber_rad_per_m.to_f64() * x - c.omega_rad_per_s.to_f64() * t
                        + c.phase_rad.to_f64())
                    .cos()
            })
            .sum();
        let got = free_surface_elevation(&cs, fx(x), fx(t)).to_f64();
        assert!((got - want).abs() < 1e-8, "x {x} t {t}: {got} vs {want}");
    }
}

/// Progressive wave: eta(x + c tau, t + tau) = eta(x, t) with c = w / k (travels toward +x).
#[test]
fn free_surface_is_a_right_travelling_wave() {
    let c = [comp(1.0, 1.5, 0.5, 0.4)];
    let speed = 1.5 / 0.5;
    let tau = 0.8;
    let a = free_surface_elevation(&c, fx(2.0), fx(1.0)).to_f64();
    let b = free_surface_elevation(&c, fx(2.0 + speed * tau), fx(1.0 + tau)).to_f64();
    assert!((a - b).abs() < 1e-8, "{a} vs {b}");
}

/// Archimedes: at eta = 0 the force equals the weight of displaced water rho g A d.
#[test]
fn froude_krylov_equals_displaced_weight_at_mean_level() {
    let f = froude_krylov_vertical_n(fx(1025.0), fx(9.81), fx(12.0), fx(0.75), Fix128::ZERO);
    assert!(rel_close(f, 1025.0 * 9.81 * 12.0 * 0.75, 1e-9));
    let f = froude_krylov_vertical_n(fx(1025.0), fx(9.81), fx(12.0), fx(0.75), fx(0.25));
    assert!(rel_close(f, 1025.0 * 9.81 * 12.0 * 1.0, 1e-9));
}

/// Module doc: "integrates hydrostatic pressure over the instantaneous wet surface". When the
/// wave trough drops below the keel (eta < -d) nothing is wet and the buoyancy is 0, never
/// negative (buoyancy cannot pull a hull down).
#[test]
#[ignore = "known defect: AUD-A-S2W3-004: froude_krylov_vertical_n returns negative force (rho g A (d+eta) = -2.4e5 N at d=1, eta=-3) when the keel is out of the water; no wet-draft clamp"]
fn froude_krylov_is_not_negative_when_the_keel_is_out_of_the_water() {
    let f = froude_krylov_vertical_n(fx(1000.0), fx(10.0), fx(12.0), fx(1.0), fx(-3.0));
    assert!(f >= Fix128::ZERO, "F = {}", f.to_f64());
}

/// Doc: "Positive = upward buoyancy in excess of the mean." The returned value at the mean
/// level (eta = 0) is rho g A d, not 0, so it is the total upthrust, not the excess over the mean.
#[test]
#[ignore = "known defect: AUD-A-S2W3-005: doc says result is buoyancy in excess of the mean, but eta = 0 returns the full upthrust rho g A d (doc/impl mismatch, doc fix)"]
fn froude_krylov_doc_excess_over_mean_is_zero_at_mean_level() {
    let f = froude_krylov_vertical_n(fx(1000.0), fx(10.0), fx(12.0), fx(1.0), Fix128::ZERO);
    assert_eq!(f, Fix128::ZERO);
}

fn advance(s: &mut ShipResponse, f: f64, m: f64, k: f64, c: f64, dt: f64) {
    s.advance(
        fx(f),
        fx(0.0),
        fx(m),
        fx(1.0),
        fx(k),
        fx(1.0),
        fx(c),
        fx(0.0),
        fx(dt),
    );
}

/// One step by hand: z'' = (F - c z' - k z)/m ; v1 = v0 + a dt ; z1 = z0 + v1 dt (v updated
/// first, so the position uses the new velocity: semi-implicit Euler).
#[test]
fn heave_step_matches_hand_computation() {
    let mut s = ShipResponse {
        heave_m: fx(0.5),
        heave_velocity_m_per_s: fx(-0.25),
        ..ShipResponse::default()
    };
    let (f, m, k, c, dt) = (3.0, 2.0, 8.0, 1.5, 0.125);
    advance(&mut s, f, m, k, c, dt);
    let a = (f - c * -0.25 - k * 0.5) / m;
    let v1 = -0.25 + a * dt;
    let z1 = 0.5 + v1 * dt;
    assert!((s.heave_velocity_m_per_s.to_f64() - v1).abs() < 1e-12);
    assert!((s.heave_m.to_f64() - z1).abs() < 1e-12);
}

#[test]
fn pitch_step_matches_hand_computation_and_is_decoupled_from_heave() {
    let mut s = ShipResponse {
        pitch_rad: fx(0.2),
        pitch_velocity_rad_per_s: fx(0.1),
        heave_m: fx(9.0),
        heave_velocity_m_per_s: fx(4.0),
    };
    let before = s;
    let (mom, inertia, kp, cp, dt) = (5.0, 4.0, 10.0, 2.0, 0.25);
    // zero heave mass -> heave untouched, pitch evolves
    s.advance(
        fx(0.0),
        fx(mom),
        fx(0.0),
        fx(inertia),
        fx(1.0),
        fx(kp),
        fx(1.0),
        fx(cp),
        fx(dt),
    );
    let al = (mom - cp * 0.1 - kp * 0.2) / inertia;
    let w1 = 0.1 + al * dt;
    let p1 = 0.2 + w1 * dt;
    assert!((s.pitch_velocity_rad_per_s.to_f64() - w1).abs() < 1e-12);
    assert!((s.pitch_rad.to_f64() - p1).abs() < 1e-12);
    assert_eq!(s.heave_m, before.heave_m);
    assert_eq!(s.heave_velocity_m_per_s, before.heave_velocity_m_per_s);
}

/// Non-positive mass / inertia freeze the corresponding DOF (no division by zero or sign flip).
#[test]
fn nonpositive_mass_and_inertia_freeze_dofs() {
    for m in [0.0, -5.0] {
        let mut s = ShipResponse {
            heave_m: fx(1.0),
            heave_velocity_m_per_s: fx(2.0),
            pitch_rad: fx(0.3),
            pitch_velocity_rad_per_s: fx(0.4),
        };
        let b = s;
        s.advance(
            fx(10.0),
            fx(10.0),
            fx(m),
            fx(m),
            fx(5.0),
            fx(5.0),
            fx(1.0),
            fx(1.0),
            fx(0.1),
        );
        assert_eq!(s, b, "mass/inertia {m}");
    }
}

/// Undamped SDOF over 20 periods with a small step: |z| tracks cos(wn t) (closed form) and the
/// total energy does not grow (symplectic update), so the response cannot blow up.
#[test]
fn undamped_oscillator_tracks_cosine_and_energy_is_bounded() {
    let (m, k) = (4.0_f64, 100.0_f64);
    let wn = (k / m).sqrt();
    let dt = 1.0e-3;
    let steps = (20.0 * 2.0 * std::f64::consts::PI / wn / dt) as usize;
    let mut s = ShipResponse {
        heave_m: Fix128::ONE,
        ..ShipResponse::default()
    };
    let e0 = 0.5 * k;
    for _ in 0..steps {
        advance(&mut s, 0.0, m, k, 0.0, dt);
        let (z, v) = (s.heave_m.to_f64(), s.heave_velocity_m_per_s.to_f64());
        let e = 0.5 * m * v * v + 0.5 * k * z * z;
        assert!(e < e0 * 1.01, "energy grew: {e} vs {e0}");
    }
    let t = steps as f64 * dt;
    assert!((s.heave_m.to_f64() - (wn * t).cos()).abs() < 0.05);
}

/// Damped SDOF: z(t) = e^{-zeta wn t} (cos wd t + zeta wn / wd sin wd t) for z0 = 1, v0 = 0.
#[test]
fn damped_oscillator_matches_closed_form() {
    let (m, k, c) = (2.0_f64, 50.0_f64, 4.0_f64);
    let wn = (k / m).sqrt();
    let zeta = c / (2.0 * (k * m).sqrt());
    let wd = wn * (1.0 - zeta * zeta).sqrt();
    let dt = 1.0e-4;
    let mut s = ShipResponse {
        heave_m: Fix128::ONE,
        ..ShipResponse::default()
    };
    let steps = 15000; // t = 1.5 s
    for _ in 0..steps {
        advance(&mut s, 0.0, m, k, c, dt);
    }
    let t = steps as f64 * dt;
    let want = (-zeta * wn * t).exp() * ((wd * t).cos() + zeta * wn / wd * (wd * t).sin());
    assert!(
        (s.heave_m.to_f64() - want).abs() < 2e-3,
        "{} vs {want}",
        s.heave_m.to_f64()
    );
}

/// Static forcing: z -> F / k.
#[test]
fn forced_response_converges_to_static_deflection() {
    let mut s = ShipResponse::default();
    for _ in 0..30000 {
        advance(&mut s, 20.0, 1.0, 10.0, 4.0, 1.0e-3);
    }
    assert!(
        (s.heave_m.to_f64() - 2.0).abs() < 1e-3,
        "{}",
        s.heave_m.to_f64()
    );
}
