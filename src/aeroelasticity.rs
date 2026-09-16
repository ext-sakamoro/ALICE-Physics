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

    /// One explicit-Euler step from a hand-chosen dyadic state, so every
    /// intermediate is exact in f32 except the π-carrying wake terms.
    ///
    /// Params: St = 1/4, U = 4, D = 1/2, ρ = 100, m = 4, ω_n = 4, ζ = 1/4,
    /// ε = 1/2, A = 4, C_L = 1/2. State: y = 1/4, ẏ = 3, q = 1/2, q̇ = 2,
    /// dt = 1/8.
    ///
    /// ```text
    /// Ω_s  = 2π·St·U/D = 4π,   Ω_s² = 16π²
    /// F    = C_L ρ U² D / (2m) = 0.5·100·16·0.5 / 8 = 50
    /// 2ζω_n = 2,   ω_n² = 16
    /// ÿ    = F q − 2ζω_n ẏ − ω_n² y = 25 − 6 − 4 = 15
    /// q̈    = −ε Ω_s (q² − 1) q̇ − Ω_s² q + A ÿ / D
    ///      = −0.5·4π·(−0.75)·2 − 8π² + 120 = 3π − 8π² + 120 ≈ 50.46794
    /// ẏ'   = 3 + 15/8 = 4.875          y' = 1/4 + 4.875/8 = 0.859375   (exact)
    /// q̇'   = 2 + q̈/8 = 17 + 3π/8 − π² ≈ 8.30849
    /// q'   = 1/2 + q̇'/8 = 21/8 + 3π/64 − π²/8 ≈ 1.53856
    /// ```
    ///
    /// No operand is 0 or 1 and no pair coincides under `*`↔`+`, `*`↔`/`,
    /// `-`↔`+`, `/`↔`%`, `+=`↔`-=` (e.g. `2m = 8 ≠ 2 + m = 6`, `U² = 16 ≠
    /// 2U = 8`, `q² − 1 = −0.75 ≠ q² + 1`). The smallest state deviation any
    /// single operator swap produces is 0.34 on q̇ (`·q̇` → `+q̇`), 300× the
    /// 1e-3 tolerance used for the two π-carrying wake components.
    #[test]
    fn viv_step_single_euler_step_matches_hand_computation() {
        let params = VivParameters {
            free_stream_velocity_m_s: 4.0,
            diameter_m: 0.5,
            fluid_density_kg_m3: 100.0,
            mass_per_length_kg_m: 4.0,
            natural_frequency_rad_s: 4.0,
            structural_damping_ratio: 0.25,
            strouhal_number: 0.25,
            wake_epsilon: 0.5,
            wake_coupling_a: 4.0,
            lift_coefficient: 0.5,
        };
        let mut state = VivState {
            displacement_m: 0.25,
            velocity_m_s: 3.0,
            wake_q: 0.5,
            wake_qdot: 2.0,
        };
        viv_step(&mut state, &params, 0.125);

        // body side: dyadic all the way, exact in f32
        assert_eq!(state.velocity_m_s, 4.875, "ẏ' = 3 + 15·(1/8)");
        assert_eq!(state.displacement_m, 0.859375, "y' = 1/4 + 4.875·(1/8)");

        // wake side: carries π, compare against the closed form in f64
        let pi = core::f64::consts::PI;
        let qdot_expected = 17.0 + 3.0 * pi / 8.0 - pi * pi;
        let q_expected = 21.0 / 8.0 + 3.0 * pi / 64.0 - pi * pi / 8.0;
        assert!(
            (f64::from(state.wake_qdot) - qdot_expected).abs() < 1.0e-3,
            "q̇' = {} vs 17 + 3π/8 − π² = {qdot_expected}",
            state.wake_qdot
        );
        assert!(
            (f64::from(state.wake_q) - q_expected).abs() < 1.0e-3,
            "q' = {} vs 21/8 + 3π/64 − π²/8 = {q_expected}",
            state.wake_q
        );
    }

    /// Same state as above with the wake decoupled (`A = 0`) and the body
    /// terms chosen so only one of them survives at a time, pinning the sign
    /// and the operand of each body term separately:
    ///
    /// ```text
    /// (a) ẏ = 0, y = 0:  ÿ = F q = 50·0.5 = 25              → ẏ' = 25/8 = 3.125
    /// (b) q = 0, y = 0:  ÿ = −2ζω_n ẏ = −2·3 = −6           → ẏ' = 3 − 6/8 = 2.25
    /// (c) q = 0, ẏ = 0:  ÿ = −ω_n² y = −16·0.25 = −4        → ẏ' = −4/8 = −0.5,
    ///                                                          y' = 0.25 − 0.5/8 = 0.1875
    /// ```
    ///
    /// In (b) and (c) the wake equation reduces to `q̈ = −ε Ω_s (0 − 1) q̇ + 0
    /// = ε Ω_s q̇ = 0.5·4π·2 = 4π`, so `q̇' = 2 + 4π/8 = 2 + π/2 ≈ 3.5708`
    /// and `q' = 0 + q̇'/8 = 0.25 + π/16 ≈ 0.44635` — the `(q² − 1)` factor
    /// with `q = 0` isolates the `− 1.0` constant and the sign of the
    /// leading `−ε`.
    #[test]
    fn viv_step_isolated_body_terms_and_decoupled_wake() {
        let params = VivParameters {
            free_stream_velocity_m_s: 4.0,
            diameter_m: 0.5,
            fluid_density_kg_m3: 100.0,
            mass_per_length_kg_m: 4.0,
            natural_frequency_rad_s: 4.0,
            structural_damping_ratio: 0.25,
            strouhal_number: 0.25,
            wake_epsilon: 0.5,
            wake_coupling_a: 0.0,
            lift_coefficient: 0.5,
        };
        let pi = core::f64::consts::PI;

        // (a) lift only
        let mut a = VivState {
            displacement_m: 0.0,
            velocity_m_s: 0.0,
            wake_q: 0.5,
            wake_qdot: 2.0,
        };
        viv_step(&mut a, &params, 0.125);
        assert_eq!(a.velocity_m_s, 3.125, "ẏ' = F q dt = 50·0.5/8");
        assert_eq!(a.displacement_m, 0.390625, "y' = 3.125/8");

        // (b) structural damping only
        let mut b = VivState {
            displacement_m: 0.0,
            velocity_m_s: 3.0,
            wake_q: 0.0,
            wake_qdot: 2.0,
        };
        viv_step(&mut b, &params, 0.125);
        assert_eq!(b.velocity_m_s, 2.25, "ẏ' = 3 − 2ζω_n·3/8 = 3 − 0.75");
        assert_eq!(b.displacement_m, 0.28125, "y' = 2.25/8");

        // (c) stiffness only
        let mut c = VivState {
            displacement_m: 0.25,
            velocity_m_s: 0.0,
            wake_q: 0.0,
            wake_qdot: 2.0,
        };
        viv_step(&mut c, &params, 0.125);
        assert_eq!(c.velocity_m_s, -0.5, "ẏ' = −ω_n² y dt = −16·0.25/8");
        assert_eq!(c.displacement_m, 0.1875, "y' = 0.25 − 0.5/8");

        // decoupled wake at q = 0: q̈ = ε Ω_s q̇ = 4π (both (b) and (c))
        for s in [&b, &c] {
            let qdot_expected = 2.0 + pi / 2.0;
            let q_expected = 0.25 + pi / 16.0;
            assert!(
                (f64::from(s.wake_qdot) - qdot_expected).abs() < 1.0e-4,
                "q̇' = {} vs 2 + π/2 = {qdot_expected}",
                s.wake_qdot
            );
            assert!(
                (f64::from(s.wake_q) - q_expected).abs() < 1.0e-4,
                "q' = {} vs 1/4 + π/16 = {q_expected}",
                s.wake_q
            );
        }
    }

    /// Strouhal shedding frequency enters only through `Ω_s = 2π St U / D`.
    /// With `A = 0`, `ε = 0`, `q̇ = 0` the wake reduces to `q̈ = −Ω_s² q`, so
    /// one Euler step gives `q̇' = −Ω_s² q dt` exactly (up to π rounding):
    ///
    /// ```text
    /// St = 1/4, U = 4, D = 1/2 → Ω_s = 4π, Ω_s² = 16π²
    /// q = 1/2, dt = 1/8       → q̇' = −16π²·(1/2)/8 = −π² ≈ −9.8696
    /// ```
    ///
    /// Since `q̇' = −Ω_s²/16` here, every mutant of the `Ω_s` line lands
    /// elsewhere: `(2 + π)·2` → −6.61, `(2/π)·2 = 4/π` → −1/π² = −0.10,
    /// `2π + St·U/D = 2π + 2` → −4.29, `2π·St + U/D = π/2 + 8` → −5.73,
    /// `(2π·St·U) % D = 0.283` → −0.005, `2π·St·U·D = π` → −π²/16 = −0.62,
    /// and `Ω_s + Ω_s` in place of `Ω_s²` → −8π·(1/2)/8 = −π/2 = −1.57.
    #[test]
    fn viv_step_shedding_frequency_squared_drives_undamped_wake() {
        let params = VivParameters {
            free_stream_velocity_m_s: 4.0,
            diameter_m: 0.5,
            fluid_density_kg_m3: 100.0,
            mass_per_length_kg_m: 4.0,
            natural_frequency_rad_s: 4.0,
            structural_damping_ratio: 0.25,
            strouhal_number: 0.25,
            wake_epsilon: 0.0,
            wake_coupling_a: 0.0,
            lift_coefficient: 0.5,
        };
        let mut state = VivState {
            displacement_m: 0.0,
            velocity_m_s: 0.0,
            wake_q: 0.5,
            wake_qdot: 0.0,
        };
        viv_step(&mut state, &params, 0.125);
        let pi = core::f64::consts::PI;
        assert!(
            (f64::from(state.wake_qdot) + pi * pi).abs() < 1.0e-4,
            "q̇' = {} vs −Ω_s² q dt = −π²",
            state.wake_qdot
        );
        // q' = q + q̇' dt = 1/2 − π²/8
        assert!(
            (f64::from(state.wake_q) - (0.5 - pi * pi / 8.0)).abs() < 1.0e-4,
            "q' = {} vs 1/2 − π²/8",
            state.wake_q
        );
        // the body saw F q = 25 for one step and nothing else
        assert_eq!(state.velocity_m_s, 3.125);
    }
}
