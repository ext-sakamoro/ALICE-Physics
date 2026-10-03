//! Independent oracles for `aeroelasticity`'s three previously-unwired items
//! (`VivParameters::facchinetti_reference`, `VivState::seeded`, `viv_step`).
//!
//! The wake-oscillator model (module doc, Facchinetti, de Langre & Biolley
//! 2004 eq. 3):
//!
//! ```text
//! wake: q'' + eps*Om*(q^2 - 1)*q' + Om^2*q = A*y''/D       Om = 2*pi*St*U/D
//! body: y'' + 2*zeta*wn*y' + wn^2*y = (C_L*rho*U^2*D/(2m)) * q
//! ```
//!
//! Two independent systems back every number here:
//!
//! 1. closed forms (Strogatz Van der Pol amplitude 2, Rao eq. 3.29 forced
//!    oscillator transfer function, dyadic hand steps), and
//! 2. an `f64` classical RK4 integration of the same ODE written in this file
//!    (a different integrator and a different arithmetic width from the
//!    module's `f32` forward Euler), so a wrong sign or a wrong operand in
//!    `viv_step` cannot be mirrored by the reference.
//!
//! Nothing here touches `src/`.

#![allow(clippy::disallowed_methods)]

use alice_physics::aeroelasticity::{viv_step, VivParameters, VivState};
use core::f64::consts::PI;

/// `f64` mirror of the parameters (read off the struct, not recomputed).
#[derive(Clone, Copy)]
struct P {
    u: f64,
    d: f64,
    rho: f64,
    m: f64,
    wn: f64,
    zeta: f64,
    st: f64,
    eps: f64,
    a: f64,
    cl: f64,
}

fn mirror(p: &VivParameters) -> P {
    P {
        u: f64::from(p.free_stream_velocity_m_s),
        d: f64::from(p.diameter_m),
        rho: f64::from(p.fluid_density_kg_m3),
        m: f64::from(p.mass_per_length_kg_m),
        wn: f64::from(p.natural_frequency_rad_s),
        zeta: f64::from(p.structural_damping_ratio),
        st: f64::from(p.strouhal_number),
        eps: f64::from(p.wake_epsilon),
        a: f64::from(p.wake_coupling_a),
        cl: f64::from(p.lift_coefficient),
    }
}

/// `[y, y', q, q']` derivative from the paper's two equations (eq. 3 and the
/// body equation), `y''` eliminated by substitution into the wake equation.
fn deriv(p: &P, s: [f64; 4]) -> [f64; 4] {
    let om = 2.0 * PI * p.st * p.u / p.d;
    let force = p.cl * p.rho * p.u * p.u * p.d / (2.0 * p.m);
    let ybb = force * s[2] - 2.0 * p.zeta * p.wn * s[1] - p.wn * p.wn * s[0];
    let qbb = -p.eps * om * (s[2] * s[2] - 1.0) * s[3] - om * om * s[2] + p.a * ybb / p.d;
    [s[1], ybb, s[3], qbb]
}

fn rk4(p: &P, s: [f64; 4], dt: f64) -> [f64; 4] {
    let add = |a: [f64; 4], b: [f64; 4], h: f64| {
        [
            a[0] + h * b[0],
            a[1] + h * b[1],
            a[2] + h * b[2],
            a[3] + h * b[3],
        ]
    };
    let k1 = deriv(p, s);
    let k2 = deriv(p, add(s, k1, dt / 2.0));
    let k3 = deriv(p, add(s, k2, dt / 2.0));
    let k4 = deriv(p, add(s, k3, dt));
    [
        s[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        s[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        s[2] + dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        s[3] + dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
    ]
}

fn to_arr(s: &VivState) -> [f64; 4] {
    [
        f64::from(s.displacement_m),
        f64::from(s.velocity_m_s),
        f64::from(s.wake_q),
        f64::from(s.wake_qdot),
    ]
}

/// Peak `|y|`, peak `|q|` and the zero-crossing count of `q` over a window of
/// a trajectory sampled once per step.
struct Window {
    y_peak: f64,
    q_peak: f64,
    crossings: usize,
}

fn run_euler(
    params: &VivParameters,
    start: VivState,
    dt: f32,
    total_s: f32,
    win: (f32, f32),
) -> Window {
    let mut s = start;
    let steps = (total_s / dt) as usize;
    let (w0, w1) = ((win.0 / dt) as usize, (win.1 / dt) as usize);
    let mut w = Window {
        y_peak: 0.0,
        q_peak: 0.0,
        crossings: 0,
    };
    let mut prev = s.wake_q;
    for n in 0..steps {
        viv_step(&mut s, params, dt);
        if n >= w0 && n < w1 {
            w.y_peak = w.y_peak.max(f64::from(s.displacement_m.abs()));
            w.q_peak = w.q_peak.max(f64::from(s.wake_q.abs()));
            if (s.wake_q > 0.0) != (prev > 0.0) {
                w.crossings += 1;
            }
        }
        prev = s.wake_q;
    }
    w
}

fn run_rk4(
    params: &VivParameters,
    start: VivState,
    dt: f64,
    total_s: f64,
    win: (f64, f64),
) -> Window {
    let p = mirror(params);
    let mut s = to_arr(&start);
    let steps = (total_s / dt).round() as usize;
    let (w0, w1) = ((win.0 / dt).round() as usize, (win.1 / dt).round() as usize);
    let mut w = Window {
        y_peak: 0.0,
        q_peak: 0.0,
        crossings: 0,
    };
    let mut prev = s[2];
    for n in 0..steps {
        s = rk4(&p, s, dt);
        if n >= w0 && n < w1 {
            w.y_peak = w.y_peak.max(s[0].abs());
            w.q_peak = w.q_peak.max(s[2].abs());
            if (s[2] > 0.0) != (prev > 0.0) {
                w.crossings += 1;
            }
        }
        prev = s[2];
    }
    w
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

// ---------------------------------------------------------------------------
// presets and seed
// ---------------------------------------------------------------------------

/// Facchinetti 2004 §2: epsilon = 0.3, A = 12 (the values the module doc
/// quotes), lift coefficient amplitude C_L0 = 0.3, Strouhal 0.2.
#[test]
fn facchinetti_reference_carries_the_published_wake_constants() {
    let p = VivParameters::facchinetti_reference();
    assert_eq!(p.wake_epsilon, 0.3);
    assert_eq!(p.wake_coupling_a, 12.0);
    assert_eq!(p.lift_coefficient, 0.3);
    assert_eq!(p.strouhal_number, 0.2);
    // physical sanity of the rest: a heavy cylinder in water, positive and
    // finite everywhere, lightly damped (zeta = 2 %)
    for v in [
        p.free_stream_velocity_m_s,
        p.diameter_m,
        p.fluid_density_kg_m3,
        p.mass_per_length_kg_m,
        p.natural_frequency_rad_s,
        p.structural_damping_ratio,
    ] {
        assert!(v.is_finite() && v > 0.0);
    }
    assert_eq!(p.structural_damping_ratio, 0.02);
}

/// The seed is a pure wake perturbation: zero body state, a small non-zero
/// `q`, zero `q'` (module doc: "all linear terms would otherwise stay at
/// zero").
#[test]
fn seeded_is_a_pure_small_wake_perturbation() {
    let s = VivState::seeded();
    assert_eq!(s.displacement_m, 0.0);
    assert_eq!(s.velocity_m_s, 0.0);
    assert_eq!(s.wake_qdot, 0.0);
    assert!(
        s.wake_q > 0.0 && s.wake_q <= 0.1,
        "small, positive: {}",
        s.wake_q
    );
}

/// With the wake exactly at rest on the unstable fixed point (q = q' = 0 and a
/// quiescent body) nothing moves: the seed is what breaks the symmetry.
#[test]
fn zero_state_is_a_fixed_point_and_the_seed_leaves_it() {
    let params = VivParameters::facchinetti_reference();
    let mut z = VivState {
        displacement_m: 0.0,
        velocity_m_s: 0.0,
        wake_q: 0.0,
        wake_qdot: 0.0,
    };
    for _ in 0..1000 {
        viv_step(&mut z, &params, 1.0e-3);
    }
    assert_eq!(
        z,
        VivState {
            displacement_m: 0.0,
            velocity_m_s: 0.0,
            wake_q: 0.0,
            wake_qdot: 0.0
        }
    );

    let mut s = VivState::seeded();
    viv_step(&mut s, &params, 1.0e-3);
    assert_ne!(s, VivState::seeded());
}

// ---------------------------------------------------------------------------
// single-step oracles
// ---------------------------------------------------------------------------

/// `dt = 0` is the identity (every update is `x += rate * dt`).
#[test]
fn zero_dt_is_the_identity() {
    let params = VivParameters::facchinetti_reference();
    let s0 = VivState {
        displacement_m: 0.013,
        velocity_m_s: -0.4,
        wake_q: 1.7,
        wake_qdot: 9.0,
    };
    let mut s = s0;
    viv_step(&mut s, &params, 0.0);
    assert_eq!(s, s0);
}

/// Hand step on a second set of dyadic parameters (disjoint from the module's
/// own unit tests), including the update order: `y'` first, then `y` with the
/// new `y'` (and likewise for the wake).
///
/// ```text
/// St = 1/2, U = 2, D = 1/2  ->  Om = 2*pi*(1/2)*2/(1/2) = 4*pi
/// rho = 8, m = 2, C_L = 1/4 ->  F = (1/4)*8*4*(1/2)/(2*2) = 1
/// wn = 2, zeta = 1/2        ->  2*zeta*wn = 2,  wn^2 = 4
/// eps = 1/4, A = 2
/// state: y = 1, y' = 1, q = 2, q' = 1, dt = 1/4
///
/// y''  = F q - 2 zeta wn y' - wn^2 y = 2 - 2 - 4 = -4
/// q''  = -eps Om (q^2 - 1) q' - Om^2 q + A y''/D
///      = -(1/4)(4 pi)(3)(1) - 16 pi^2 * 2 + 2*(-4)/(1/2)
///      = -3 pi - 32 pi^2 - 16
/// y'_new = 1 - 4/4 = 0            y_new = 1 + 0/4 = 1       (exact)
/// q'_new = 1 + q''/4 = 1 - 3pi/4 - 8 pi^2 - 4 = -3 - 3pi/4 - 8 pi^2
/// q_new  = 2 + q'_new/4
/// ```
#[test]
fn single_step_on_disjoint_dyadic_parameters() {
    let params = VivParameters {
        free_stream_velocity_m_s: 2.0,
        diameter_m: 0.5,
        fluid_density_kg_m3: 8.0,
        mass_per_length_kg_m: 2.0,
        natural_frequency_rad_s: 2.0,
        structural_damping_ratio: 0.5,
        strouhal_number: 0.5,
        wake_epsilon: 0.25,
        wake_coupling_a: 2.0,
        lift_coefficient: 0.25,
    };
    let mut s = VivState {
        displacement_m: 1.0,
        velocity_m_s: 1.0,
        wake_q: 2.0,
        wake_qdot: 1.0,
    };
    viv_step(&mut s, &params, 0.25);
    assert_eq!(s.velocity_m_s, 0.0);
    assert_eq!(s.displacement_m, 1.0);
    let qd = -3.0 - 3.0 * PI / 4.0 - 8.0 * PI * PI;
    let q = 2.0 + qd / 4.0;
    // f32 carries ~1e-7 relative; |q'| ~ 85 -> 1e-4 absolute is generous
    assert!(
        (f64::from(s.wake_qdot) - qd).abs() < 1.0e-3,
        "q' {} vs {qd}",
        s.wake_qdot
    );
    assert!(
        (f64::from(s.wake_q) - q).abs() < 1.0e-3,
        "q {} vs {q}",
        s.wake_q
    );
}

/// One-step agreement with an `f64` Taylor step over a sweep of parameter
/// sets. The reference is the explicit-Euler map written from the paper's
/// equations; any mistyped operator, sign or operand in `viv_step` moves at
/// least one of the 4 state components by far more than the f32 tolerance.
#[test]
fn single_step_matches_f64_euler_over_a_parameter_sweep() {
    let base = VivParameters::facchinetti_reference();
    let mut n_checked = 0;
    for &u in &[0.2f32, 0.5, 0.9] {
        for &(eps, a) in &[(0.3f32, 12.0f32), (0.1, 4.0), (0.6, 0.0)] {
            for &(zeta, cl) in &[(0.02f32, 0.3f32), (0.1, 0.8)] {
                let params = VivParameters {
                    free_stream_velocity_m_s: u,
                    wake_epsilon: eps,
                    wake_coupling_a: a,
                    structural_damping_ratio: zeta,
                    lift_coefficient: cl,
                    ..base
                };
                let s0 = VivState {
                    displacement_m: 0.004,
                    velocity_m_s: 0.03,
                    wake_q: 0.7,
                    wake_qdot: -1.3,
                };
                let dt = 1.0e-4f32;
                let mut s = s0;
                viv_step(&mut s, &params, dt);

                let p = mirror(&params);
                let st = to_arr(&s0);
                let d = deriv(&p, st);
                let dtf = f64::from(dt);
                // body: y' updated first, y advances with the *new* y'
                let v1 = st[1] + d[1] * dtf;
                let y1 = st[0] + v1 * dtf;
                let qd1 = st[3] + d[3] * dtf;
                let q1 = st[2] + qd1 * dtf;
                let got = to_arr(&s);
                for (g, w, name) in [
                    (got[0], y1, "y"),
                    (got[1], v1, "y'"),
                    (got[2], q1, "q"),
                    (got[3], qd1, "q'"),
                ] {
                    assert!(
                        (g - w).abs() <= 1.0e-4 * w.abs().max(1.0e-3) + 1.0e-7,
                        "{name}: {g} vs {w} (U={u}, eps={eps}, A={a}, zeta={zeta}, C_L={cl})"
                    );
                }
                n_checked += 1;
            }
        }
    }
    assert_eq!(n_checked, 18);
}

// ---------------------------------------------------------------------------
// trajectories
// ---------------------------------------------------------------------------

/// Coupled wake (`A = 12`) at the reference parameters: the Euler trajectory
/// agrees with an `f64` RK4 integration of the same ODE in limit-cycle
/// amplitude of both the wake and the body, and in shedding frequency. This is
/// the only test of the `A*y''/D` coupling term over a cycle.
#[test]
fn coupled_limit_cycle_matches_rk4_reference() {
    let params = VivParameters::facchinetti_reference();
    let start = VivState::seeded();
    // 12 s: well past the transient (growth rate eps*Om/2 ~ 3/s), measure last 4 s
    let e = run_euler(&params, start, 2.0e-5, 12.0, (8.0, 12.0));
    let r = run_rk4(&params, start, 1.0e-4, 12.0, (8.0, 12.0));
    assert!(
        r.q_peak > 1.5,
        "reference must have reached the limit cycle: {}",
        r.q_peak
    );
    assert!(
        rel(e.q_peak, r.q_peak) < 0.04,
        "q peak {} vs RK4 {}",
        e.q_peak,
        r.q_peak
    );
    assert!(
        rel(e.y_peak, r.y_peak) < 0.04,
        "y peak {} vs RK4 {}",
        e.y_peak,
        r.y_peak
    );
    assert!(
        (e.crossings as i64 - r.crossings as i64).abs() <= 1,
        "zero crossings {} vs RK4 {}",
        e.crossings,
        r.crossings
    );
}

/// Lock-in: the coupled wake does not shed at the bare Strouhal frequency. With
/// `A = 12` the shedding frequency is pulled away from `St*U/D` (the RK4
/// reference decides by how much); with `A = 0` it sits on the Strouhal law.
/// Fails if the coupling term is dropped or sign-flipped.
#[test]
fn coupling_pulls_the_shedding_frequency_off_the_strouhal_line() {
    let coupled = VivParameters::facchinetti_reference();
    let free = VivParameters {
        wake_coupling_a: 0.0,
        ..coupled
    };
    let window = (8.0f32, 12.0f32);
    let f_strouhal = 0.2 * 0.5 / 0.03; // Hz
    let hz = |w: &Window| w.crossings as f64 / (2.0 * f64::from(window.1 - window.0));

    let f_free = hz(&run_euler(&free, VivState::seeded(), 2.0e-5, 12.0, window));
    let f_coupled = hz(&run_euler(
        &coupled,
        VivState::seeded(),
        2.0e-5,
        12.0,
        window,
    ));
    let f_coupled_ref = {
        let r = run_rk4(&coupled, VivState::seeded(), 1.0e-4, 12.0, (8.0, 12.0));
        r.crossings as f64 / 8.0
    };

    // decoupled: Van der Pol frequency Om*(1 - eps^2/16), 0.6 % below Strouhal
    assert!(
        rel(f_free, f_strouhal) < 0.02,
        "{f_free} Hz vs St U/D = {f_strouhal} Hz"
    );
    // coupled: agrees with the independent integrator ...
    assert!(
        rel(f_coupled, f_coupled_ref) < 0.02,
        "{f_coupled} Hz vs RK4 {f_coupled_ref} Hz"
    );
    // ... and is measurably displaced from the free wake (>= 2 %)
    assert!(
        rel(f_coupled, f_free) > 0.02,
        "coupled {f_coupled} Hz must differ from the free wake {f_free} Hz"
    );
}

/// Body resonance: with the wake decoupled (`A = 0`) the body is a linear
/// oscillator driven by `q`, whose limit-cycle amplitude is 2 (Strogatz §7.6).
/// At `Om = wn` the steady response is `F q / (2 zeta wn^2)` (Rao eq. 3.29 at
/// resonance), 25x the quasi-static `F q / wn^2` for `zeta = 2 %`. The shedding
/// frequency is set by choosing `U = wn D / (2 pi St)`.
#[test]
fn body_response_peaks_at_resonance_with_the_transfer_function_gain() {
    let base = VivParameters::facchinetti_reference();
    let wn = f64::from(base.natural_frequency_rad_s);
    let d = f64::from(base.diameter_m);
    let u_res = wn * d / (2.0 * PI * 0.2);
    let params = VivParameters {
        wake_coupling_a: 0.0,
        free_stream_velocity_m_s: u_res as f32,
        ..base
    };
    let start = VivState {
        wake_q: 1.0,
        ..VivState::seeded()
    };
    // resonance rings up with time constant 1/(zeta wn) ~ 2 s: run 20 s, read the last 3 s
    let w = run_euler(&params, start, 2.0e-5, 20.0, (17.0, 20.0));
    let p = mirror(&params);
    let force = p.cl * p.rho * p.u * p.u * p.d / (2.0 * p.m);
    let gain_resonance = force / (2.0 * p.zeta * p.wn * p.wn);
    let ratio = w.y_peak / w.q_peak;
    // Van der Pol third harmonic and forward-Euler damping error: 5 %
    assert!(
        rel(ratio, gain_resonance) < 0.05,
        "y/q = {ratio} vs F/(2 zeta wn^2) = {gain_resonance}"
    );
    // and it is far above the off-resonance response at the reference U
    let off = VivParameters {
        wake_coupling_a: 0.0,
        ..base
    };
    let w_off = run_euler(&off, start, 2.0e-5, 20.0, (17.0, 20.0));
    assert!(ratio > 3.0 * (w_off.y_peak / w_off.q_peak));
}

/// A stationary fluid (`U = 0`) makes `Om = 0` and `F = 0`: with the wake
/// decoupled (`A = 0`) the wake equation has no restoring term and no forcing,
/// and the body decays at its structural damping from any start
/// (`zeta wn = 0.48 /s`).
#[test]
fn quiescent_flow_leaves_only_structural_decay() {
    let params = VivParameters {
        free_stream_velocity_m_s: 0.0,
        wake_coupling_a: 0.0,
        ..VivParameters::facchinetti_reference()
    };
    let mut s = VivState {
        displacement_m: 0.01,
        velocity_m_s: 0.0,
        wake_q: 0.0,
        wake_qdot: 0.0,
    };
    let dt = 1.0e-4f32;
    let t_end = 10.0f32;
    for _ in 0..(t_end / dt) as usize {
        viv_step(&mut s, &params, dt);
    }
    // damped oscillator envelope exp(-zeta wn t); wake untouched
    let env = 0.01 * (-0.02f64 * 24.0 * f64::from(t_end)).exp();
    assert!(
        f64::from(s.displacement_m).abs() < 1.6 * env,
        "{} vs envelope {env}",
        s.displacement_m
    );
    assert_eq!(s.wake_q, 0.0);
    assert_eq!(s.wake_qdot, 0.0);
}

// ---------------------------------------------------------------------------
// degenerate inputs
// ---------------------------------------------------------------------------

/// `viv_step` returns `()` and has no error channel: a zero diameter or zero
/// mass divides by zero in `f32` and the state becomes non-finite without a
/// panic. Pinned so a change to either (panic or `Result`) is a visible API
/// decision, not a silent one.
#[test]
fn zero_diameter_or_mass_yields_a_non_finite_state_without_panicking() {
    for broken in [
        VivParameters {
            diameter_m: 0.0,
            ..VivParameters::facchinetti_reference()
        },
        VivParameters {
            mass_per_length_kg_m: 0.0,
            ..VivParameters::facchinetti_reference()
        },
    ] {
        let mut s = VivState::seeded();
        viv_step(&mut s, &broken, 1.0e-3);
        let all = [s.displacement_m, s.velocity_m_s, s.wake_q, s.wake_qdot];
        assert!(all.iter().any(|v| !v.is_finite()), "{s:?}");
    }
}

/// Same inputs, same bits (the module is f32-only deterministic Euler).
#[test]
fn trajectory_is_bit_reproducible() {
    let params = VivParameters::facchinetti_reference();
    let run = || {
        let mut s = VivState::seeded();
        for _ in 0..20_000 {
            viv_step(&mut s, &params, 5.0e-4);
        }
        s
    };
    assert_eq!(run(), run());
}
