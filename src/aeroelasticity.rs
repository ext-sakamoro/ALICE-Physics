//! Vortex-induced vibration (VIV) of an elastically-mounted circular
//! cylinder.
//!
//! Companion of [`crate::wave_ship`] (hydrodynamic hull) for
//! **wake-induced** cross-flow oscillation. The wake dynamics are
//! captured by a Van der Pol oscillator whose amplitude tracks the
//! Strouhal-shed vortex intensity; the wake force is coupled back
//! into a Newton's second-law equation for the cylinder's cross-flow
//! displacement:
//!
//! ```text
//! wake: q̈ + ε (q² − 1) q̇ + Ω_s² q = A · ÿ / D
//! body: ÿ + 2 ζ ω_n ẏ + ω_n² y = C_L · ρ · U² · D / (2 m) · q
//! ```
//!
//! where `Ω_s = 2π · St · U / D` is the shedding angular frequency,
//! `St ≈ 0.2` is the Strouhal number, `U` is the free-stream velocity,
//! `D` is the cylinder diameter, and `A`, `ε`, `C_L`, `m`, `ζ`, and
//! `ω_n` are the standard wake-oscillator constants from Facchinetti
//! et al. 2004.
//!
//! # Scope
//!
//! Single-degree-of-freedom cylinder, small displacement, subcritical
//! Reynolds regime. Full 3-D fluid-structure coupling, mode
//! bifurcation, and lock-in hysteresis remain future work.

/// Van der Pol wake oscillator + cylinder cross-flow state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VivState {
    /// Cross-flow displacement (m).
    pub displacement_m: f32,
    /// Cross-flow velocity (m/s).
    pub velocity_m_s: f32,
    /// Wake-oscillator variable (non-dimensional).
    pub wake_q: f32,
    /// Wake time derivative.
    pub wake_qdot: f32,
}

impl VivState {
    /// Small perturbed initial state so the wake oscillator can grow
    /// from noise. All linear terms would otherwise stay at zero.
    #[must_use]
    pub const fn seeded() -> Self {
        Self {
            displacement_m: 0.0,
            velocity_m_s: 0.0,
            wake_q: 0.01,
            wake_qdot: 0.0,
        }
    }
}

/// Model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VivParameters {
    /// Free-stream velocity `U` (m/s).
    pub free_stream_velocity_m_s: f32,
    /// Cylinder diameter `D` (m).
    pub diameter_m: f32,
    /// Fluid density `ρ` (kg/m³).
    pub fluid_density_kg_m3: f32,
    /// Cylinder mass per unit length `m` (kg/m).
    pub mass_per_length_kg_m: f32,
    /// Natural angular frequency `ω_n` (rad/s).
    pub natural_frequency_rad_s: f32,
    /// Damping ratio `ζ` (structural).
    pub structural_damping_ratio: f32,
    /// Strouhal number `St` — typical 0.2 for subcritical Re.
    pub strouhal_number: f32,
    /// Van der Pol parameter ε (dimensionless; the damping term is
    /// `ε·Ω_f·(q² − 1)·q̇`, Facchinetti et al. 2004 eq. 3). 0.3 in the paper.
    pub wake_epsilon: f32,
    /// Coupling `A · ÿ / D` — Facchinetti 2004 value ~12.
    pub wake_coupling_a: f32,
    /// Sectional lift coefficient `C_L` at lock-in.
    pub lift_coefficient: f32,
}

impl VivParameters {
    /// Facchinetti et al. 2004 — steel cylinder in water tunnel.
    #[must_use]
    pub const fn facchinetti_reference() -> Self {
        Self {
            free_stream_velocity_m_s: 0.5,
            diameter_m: 0.03,
            fluid_density_kg_m3: 1000.0,
            mass_per_length_kg_m: 0.7,
            natural_frequency_rad_s: 24.0,
            structural_damping_ratio: 0.02,
            strouhal_number: 0.2,
            wake_epsilon: 0.3,
            wake_coupling_a: 12.0,
            lift_coefficient: 0.3,
        }
    }
}

/// Advance the wake + body coupled ODE by `dt` seconds using explicit
/// forward Euler.
///
/// Callers who need long-time accuracy should sub-cycle the step or
/// swap in RK4 externally; the module ships forward Euler because it
/// is the simplest scheme that shows correct qualitative behaviour
/// (limit cycle amplitude, lock-in).
pub fn viv_step(state: &mut VivState, params: &VivParameters, dt: f32) {
    let omega_s =
        2.0 * core::f32::consts::PI * params.strouhal_number * params.free_stream_velocity_m_s
            / params.diameter_m;
    let omega_s_sq = omega_s * omega_s;
    let mass = params.mass_per_length_kg_m;
    let vel_sq = params.free_stream_velocity_m_s * params.free_stream_velocity_m_s;
    let fluid_force =
        params.lift_coefficient * params.fluid_density_kg_m3 * vel_sq * params.diameter_m
            / (2.0 * mass);
    let structural_damping = 2.0 * params.structural_damping_ratio * params.natural_frequency_rad_s;
    let natural_sq = params.natural_frequency_rad_s * params.natural_frequency_rad_s;

    let acc_body = fluid_force * state.wake_q
        - structural_damping * state.velocity_m_s
        - natural_sq * state.displacement_m;
    // Facchinetti et al. 2004 eq. (3): q̈ + ε Ω_f (q² − 1) q̇ + Ω_f² q = A ÿ / D.
    // The Van der Pol damping is scaled by the shedding frequency Ω_f, so ε is
    // the dimensionless 0.3 of the paper and the wake locks in within a few
    // shedding periods. Before 1.2.0 the Ω_f factor was missing: ε acted as a
    // rate of 0.3 /s, the limit cycle took ~30 s instead of ~1 s to establish
    // (`tests/engineering_oracles_fluid.rs`).
    let acc_wake =
        -params.wake_epsilon * omega_s * (state.wake_q * state.wake_q - 1.0) * state.wake_qdot
            - omega_s_sq * state.wake_q
            + params.wake_coupling_a * acc_body / params.diameter_m;

    // Explicit Euler.
    state.velocity_m_s += acc_body * dt;
    state.displacement_m += state.velocity_m_s * dt;
    state.wake_qdot += acc_wake * dt;
    state.wake_q += state.wake_qdot * dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_gives_non_trivial_wake() {
        let s = VivState::seeded();
        assert!(s.wake_q.abs() > 0.0);
    }

    #[test]
    fn viv_step_is_deterministic() {
        let params = VivParameters::facchinetti_reference();
        let mut a = VivState::seeded();
        let mut b = VivState::seeded();
        viv_step(&mut a, &params, 1.0e-3);
        viv_step(&mut b, &params, 1.0e-3);
        assert_eq!(a, b);
    }

    #[test]
    fn oscillation_grows_from_seed() {
        let params = VivParameters::facchinetti_reference();
        let mut state = VivState::seeded();
        let mut peak = 0.0_f32;
        for _ in 0..5_000 {
            viv_step(&mut state, &params, 5.0e-4);
            let amp = state.displacement_m.abs();
            if amp > peak {
                peak = amp;
            }
        }
        assert!(peak > 1.0e-4, "expected wake-driven growth, got {peak}");
    }

    #[test]
    fn zero_free_stream_yields_no_wake_forcing() {
        let mut params = VivParameters::facchinetti_reference();
        params.free_stream_velocity_m_s = 0.0;
        let mut state = VivState::seeded();
        state.wake_q = 0.5;
        for _ in 0..100 {
            viv_step(&mut state, &params, 1.0e-3);
        }
        // Without wind, cylinder body should stay very close to origin.
        assert!(state.displacement_m.abs() < 1.0e-3);
    }

    #[test]
    fn increasing_velocity_raises_shedding_frequency() {
        // Sanity check: qualitatively higher U should produce a
        // higher-frequency wake, so the sign of q flips faster.
        let params_low = VivParameters {
            free_stream_velocity_m_s: 0.3,
            ..VivParameters::facchinetti_reference()
        };
        let params_high = VivParameters {
            free_stream_velocity_m_s: 1.5,
            ..VivParameters::facchinetti_reference()
        };
        let count_flips = |params: &VivParameters| -> usize {
            let mut state = VivState::seeded();
            let mut prev_sign = state.wake_q.signum();
            let mut flips = 0;
            for _ in 0..4_000 {
                viv_step(&mut state, params, 5.0e-4);
                let s = state.wake_q.signum();
                if s != 0.0 && s != prev_sign {
                    flips += 1;
                    prev_sign = s;
                }
            }
            flips
        };
        assert!(count_flips(&params_high) > count_flips(&params_low));
    }
}
