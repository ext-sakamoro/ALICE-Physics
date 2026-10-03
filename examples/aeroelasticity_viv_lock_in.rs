//! Vortex-induced vibration of an elastically mounted cylinder, driven through
//! `aeroelasticity::{VivParameters::facchinetti_reference, VivState::seeded,
//! viv_step}`.
//!
//! The wake is a Van der Pol oscillator coupled to a 1-DOF cylinder
//! (Facchinetti, de Langre & Biolley 2004, eq. 3). With the wake decoupled
//! (`A = 0`) the closed forms are:
//!
//! ```text
//! wake amplitude   |q|max = 2                         (Strogatz, section 7.6)
//! shedding freq    f = St * U / D = 0.2*0.5/0.03 = 3.333 Hz   (Strouhal law)
//! body response    |y|/|q| = F / sqrt((wn^2 - Om^2)^2 + (2 zeta wn Om)^2)
//!                  F = C_L rho U^2 D / (2 m)                  (Rao eq. 3.29)
//! ```
//!
//! The same cylinder is then run with the reference wake coupling `A = 12`
//! to show lock-in: the shedding frequency moves off the Strouhal line
//! (toward the natural frequency `wn / 2 pi = 3.82 Hz`).
//!
//! Limitation: the body equation carries structural damping only (no
//! hydrodynamic damping or added mass), so the coupled limit cycle is far
//! larger than the roughly one diameter measured in experiments. The
//! amplitudes printed for the coupled run are the model's own, not a
//! prediction for a real cylinder.
//!
//! ```bash
//! cargo run --release --example aeroelasticity_viv_lock_in --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::aeroelasticity::{viv_step, VivParameters, VivState};
use core::f64::consts::PI;

struct Measured {
    q_peak: f64,
    y_peak: f64,
    freq_hz: f64,
}

/// Integrate `total_s` seconds and read amplitude / frequency over the last
/// `window_s` seconds (zero crossings of `q`, two per period).
fn measure(
    params: &VivParameters,
    start: VivState,
    dt: f32,
    total_s: f32,
    window_s: f32,
) -> Measured {
    let mut s = start;
    let steps = (total_s / dt) as usize;
    let first = ((total_s - window_s) / dt) as usize;
    let mut q_peak = 0.0f64;
    let mut y_peak = 0.0f64;
    let mut crossings = 0usize;
    let mut prev = s.wake_q;
    for n in 0..steps {
        viv_step(&mut s, params, dt);
        if n >= first {
            q_peak = q_peak.max(f64::from(s.wake_q.abs()));
            y_peak = y_peak.max(f64::from(s.displacement_m.abs()));
            if (s.wake_q > 0.0) != (prev > 0.0) {
                crossings += 1;
            }
        }
        prev = s.wake_q;
    }
    Measured {
        q_peak,
        y_peak,
        freq_hz: crossings as f64 / (2.0 * f64::from(window_s)),
    }
}

fn main() {
    let reference = VivParameters::facchinetti_reference();
    let free = VivParameters {
        wake_coupling_a: 0.0,
        ..reference
    };

    // closed forms
    let f_strouhal = f64::from(reference.strouhal_number)
        * f64::from(reference.free_stream_velocity_m_s)
        / f64::from(reference.diameter_m);
    let om = 2.0 * PI * f_strouhal;
    let wn = f64::from(reference.natural_frequency_rad_s);
    let zeta = f64::from(reference.structural_damping_ratio);
    let force = f64::from(reference.lift_coefficient)
        * f64::from(reference.fluid_density_kg_m3)
        * f64::from(reference.free_stream_velocity_m_s).powi(2)
        * f64::from(reference.diameter_m)
        / (2.0 * f64::from(reference.mass_per_length_kg_m));
    let transfer = force / ((wn * wn - om * om).powi(2) + (2.0 * zeta * wn * om).powi(2)).sqrt();

    let start = VivState {
        wake_q: 1.0,
        ..VivState::seeded()
    };
    let m_free = measure(&free, start, 2.0e-5, 40.0, 5.0);
    println!("decoupled wake (A = 0), last 5 s of 40 s");
    println!(
        "  |q|max      measured {:.4}   closed form 2 (Van der Pol)",
        m_free.q_peak
    );
    println!(
        "  frequency   measured {:.3} Hz  closed form St U/D = {:.3} Hz",
        m_free.freq_hz, f_strouhal
    );
    println!(
        "  |y|/|q|     measured {:.5}   closed form {:.5} (Rao 3.29)",
        m_free.y_peak / m_free.q_peak,
        transfer
    );
    assert!((m_free.q_peak - 2.0).abs() < 0.06, "Van der Pol amplitude");
    assert!(
        (m_free.freq_hz - f_strouhal).abs() / f_strouhal < 0.025,
        "Strouhal law"
    );
    assert!(
        (m_free.y_peak / m_free.q_peak - transfer).abs() / transfer < 0.05,
        "body transfer function"
    );

    let m_lock = measure(&reference, VivState::seeded(), 2.0e-5, 12.0, 4.0);
    println!("coupled wake (A = 12), last 4 s of 12 s");
    println!(
        "  frequency   {:.3} Hz   Strouhal line {:.3} Hz   shift {:+.1} %",
        m_lock.freq_hz,
        f_strouhal,
        100.0 * (m_lock.freq_hz - f_strouhal) / f_strouhal
    );
    println!(
        "  |y|max      {:.3} m ({:.0} D)   |q|max {:.2}   (model amplitude, see the limitation above)",
        m_lock.y_peak,
        m_lock.y_peak / f64::from(reference.diameter_m),
        m_lock.q_peak
    );
    assert!(
        (m_lock.freq_hz - f_strouhal).abs() / f_strouhal > 0.02,
        "coupling must move the shedding frequency off the Strouhal line"
    );
}
