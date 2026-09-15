//! Ocean Waves + Ship Dynamics (JONSWAP + Froude-Krylov)
//!
//! Phase G5 of the ALICE-Physics completeness project. Provides the wave
//! generation and vessel-response building blocks needed for maritime
//! simulation:
//!
//! - **JONSWAP** wave spectrum: parameterised by significant wave height
//!   `H_s` and peak period `T_p`.
//! - Wave elevation `η(x, t)` via linear superposition of a discrete set
//!   of frequency components.
//! - **Froude-Krylov buoyancy**: integrates hydrostatic pressure over the
//!   instantaneous wet surface of a floating body — the first-order wave
//!   force in naval hydrodynamics.
//! - Simple 2-DOF (heave + pitch) response equation.
//!
//! # References
//!
//! - Hasselmann et al., "Measurements of wind-wave growth and swell decay
//!   during the Joint North Sea Wave Project (JONSWAP)", Deutsches
//!   Hydrographisches Institut 12, 1973.
//! - Faltinsen, *Sea Loads on Ships and Offshore Structures* Cambridge 1990.
//! - Journee & Massie, *Offshore Hydromechanics* TU Delft 2001.

use crate::math::Fix128;

// ============================================================================
// JONSWAP spectrum
// ============================================================================

/// JONSWAP wave-spectrum parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Jonswap {
    /// Significant wave height `H_s` (m). 3-4 m typical for open North Sea.
    pub significant_wave_height_m: Fix128,
    /// Peak spectral period `T_p` (s). 8-10 s typical.
    pub peak_period_s: Fix128,
    /// Peak-enhancement factor `γ` (dimensionless). Default 3.3 (JONSWAP).
    pub gamma: Fix128,
}

impl Jonswap {
    /// Fully developed North Sea sea-state (H_s = 3, T_p = 9, γ = 3.3).
    #[must_use]
    pub fn north_sea() -> Self {
        Self {
            significant_wave_height_m: Fix128::from_int(3),
            peak_period_s: Fix128::from_int(9),
            gamma: Fix128::from_ratio(33, 10),
        }
    }

    /// Peak angular frequency `ω_p = 2π / T_p` (rad/s).
    #[must_use]
    pub fn peak_omega(&self) -> Fix128 {
        if self.peak_period_s.is_zero() {
            return Fix128::ZERO;
        }
        let two_pi = Fix128::PI + Fix128::PI;
        two_pi / self.peak_period_s
    }

    /// Spectral density `S(ω)` (m²·s / rad) at angular frequency `ω` — the
    /// JONSWAP form (Hasselmann et al. 1973; Chakrabarti 1987 eq. 4.29):
    /// `S(ω) = 5/16 · H_s²·ω_p⁴/ω⁵ · exp(−5/4·(ω_p/ω)⁴) · γ^r`,
    /// `r = exp(−(ω−ω_p)²/(2·σ²·ω_p²))`, `σ = 0.07` for `ω ≤ ω_p`, `0.09` above.
    ///
    /// `γ = 1` reduces to Pierson–Moskowitz. The spectrum peaks at `ω_p`,
    /// vanishes for `ω → 0` and decays as `ω⁻⁵`. Before 1.2.0 the module
    /// returned `5/16·H_s²·ω_p⁴/ω⁵·√γ` without the exponential cut-off or the
    /// peak-enhancement shape — monotone in ω, unbounded as `ω → 0`, so the
    /// zeroth moment diverged and `H_s` could not be recovered from it
    /// (`tests/engineering_oracles_fluid.rs`).
    #[must_use]
    pub fn spectrum_density(&self, omega_rad_per_s: Fix128) -> Fix128 {
        if omega_rad_per_s <= Fix128::ZERO {
            return Fix128::ZERO;
        }
        let omega_p = self.peak_omega();
        if omega_p.is_zero() {
            return Fix128::ZERO;
        }
        let hs_sq = self.significant_wave_height_m * self.significant_wave_height_m;
        let op4 = omega_p * omega_p * omega_p * omega_p;
        let o5 =
            omega_rad_per_s * omega_rad_per_s * omega_rad_per_s * omega_rad_per_s * omega_rad_per_s;
        if o5.is_zero() {
            return Fix128::ZERO;
        }
        let pm = Fix128::from_ratio(5, 16) * hs_sq * op4 / o5;
        // exp(−5/4 (ω_p/ω)⁴)
        let ratio = omega_p / omega_rad_per_s;
        let ratio4 = ratio * ratio * ratio * ratio;
        let cutoff = (Fix128::from_ratio(-5, 4) * ratio4).exp();
        // γ^r, r = exp(−(ω − ω_p)² / (2 σ² ω_p²))
        let sigma = if omega_rad_per_s <= omega_p {
            Fix128::from_ratio(7, 100)
        } else {
            Fix128::from_ratio(9, 100)
        };
        let d = omega_rad_per_s - omega_p;
        let denom = Fix128::from_int(2) * sigma * sigma * omega_p * omega_p;
        let r = if denom.is_zero() {
            Fix128::ZERO
        } else {
            (-(d * d) / denom).exp()
        };
        let enhancement = if self.gamma <= Fix128::ZERO {
            Fix128::ONE
        } else {
            self.gamma.powf_pos(r)
        };
        pm * cutoff * enhancement
    }
}

/// Discrete wave component: amplitude, frequency, phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveComponent {
    /// Amplitude (m).
    pub amplitude_m: Fix128,
    /// Angular frequency (rad/s).
    pub omega_rad_per_s: Fix128,
    /// Wavenumber `k` (rad/m). For deep-water gravity waves: `k = ω²/g`.
    pub wavenumber_rad_per_m: Fix128,
    /// Phase offset (rad).
    pub phase_rad: Fix128,
}

/// Free-surface elevation `η(x, t) = Σ A_i · cos(k_i·x − ω_i·t + φ_i)` (m).
#[must_use]
pub fn free_surface_elevation(components: &[WaveComponent], x_m: Fix128, t_s: Fix128) -> Fix128 {
    let mut eta = Fix128::ZERO;
    for c in components {
        let arg = c.wavenumber_rad_per_m * x_m - c.omega_rad_per_s * t_s + c.phase_rad;
        eta = eta + c.amplitude_m * arg.cos();
    }
    eta
}

// ============================================================================
// Froude-Krylov buoyancy
// ============================================================================

/// Froude-Krylov vertical force (N) on a floating box of horizontal area
/// `A_water` (m²) with mean draft `d_m` (m), given the wave elevation `η`
/// at the box centre and water density `ρ_w`.
///
/// Approximates the pressure integral by treating the box as a small
/// horizontal plate:
///
/// `F_z = ρ_w · g · A_water · (d_mean + η)`
///
/// Positive = upward buoyancy in excess of the mean.
#[must_use]
pub fn froude_krylov_vertical_n(
    density_water: Fix128,
    gravity_m_per_s2: Fix128,
    area_waterplane_m2: Fix128,
    mean_draft_m: Fix128,
    wave_elevation_m: Fix128,
) -> Fix128 {
    density_water * gravity_m_per_s2 * area_waterplane_m2 * (mean_draft_m + wave_elevation_m)
}

// ============================================================================
// 2-DOF ship response
// ============================================================================

/// Ship 2-DOF (heave + pitch) response state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ShipResponse {
    /// Vertical displacement (m).
    pub heave_m: Fix128,
    /// Vertical velocity (m/s).
    pub heave_velocity_m_per_s: Fix128,
    /// Pitch angle (rad).
    pub pitch_rad: Fix128,
    /// Pitch angular velocity (rad/s).
    pub pitch_velocity_rad_per_s: Fix128,
}

impl ShipResponse {
    /// Explicit Euler step of the linear 2-DOF equation `m·z̈ + c·ż + k·z = F(t)`.
    /// Simplified: uses independent SDOF for each DOF (heave/pitch decoupled).
    // 1.0.0 で公開済の signature (crates.io)、引数 struct 化は semver major = 2.0 で実施
    #[allow(clippy::too_many_arguments)]
    pub fn advance(
        &mut self,
        heave_force_n: Fix128,
        pitch_moment_nm: Fix128,
        mass_kg: Fix128,
        pitch_inertia_kg_m2: Fix128,
        heave_stiffness_n_per_m: Fix128,
        pitch_stiffness_nm_per_rad: Fix128,
        heave_damping_ns_per_m: Fix128,
        pitch_damping_nms_per_rad: Fix128,
        dt_s: Fix128,
    ) {
        // Heave equation: m·z̈ = F - c·ż - k·z
        if mass_kg > Fix128::ZERO {
            let accel = (heave_force_n
                - heave_damping_ns_per_m * self.heave_velocity_m_per_s
                - heave_stiffness_n_per_m * self.heave_m)
                / mass_kg;
            self.heave_velocity_m_per_s = self.heave_velocity_m_per_s + accel * dt_s;
            self.heave_m = self.heave_m + self.heave_velocity_m_per_s * dt_s;
        }
        // Pitch equation similar
        if pitch_inertia_kg_m2 > Fix128::ZERO {
            let alpha = (pitch_moment_nm
                - pitch_damping_nms_per_rad * self.pitch_velocity_rad_per_s
                - pitch_stiffness_nm_per_rad * self.pitch_rad)
                / pitch_inertia_kg_m2;
            self.pitch_velocity_rad_per_s = self.pitch_velocity_rad_per_s + alpha * dt_s;
            self.pitch_rad = self.pitch_rad + self.pitch_velocity_rad_per_s * dt_s;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        let d = if a > b { a - b } else { b - a };
        d <= tol
    }

    #[test]
    fn north_sea_preset_reasonable() {
        let j = Jonswap::north_sea();
        assert_eq!(j.significant_wave_height_m, Fix128::from_int(3));
        assert_eq!(j.peak_period_s, Fix128::from_int(9));
        assert!(approx_eq(
            j.gamma,
            Fix128::from_ratio(33, 10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn peak_omega_from_tp() {
        // ω_p = 2π / 9 ≈ 0.698 rad/s
        let j = Jonswap::north_sea();
        let op = j.peak_omega();
        assert!(approx_eq(
            op,
            Fix128::from_ratio(698, 1000),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn spectrum_zero_at_zero_omega() {
        let j = Jonswap::north_sea();
        assert_eq!(j.spectrum_density(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn spectrum_positive_at_peak() {
        let j = Jonswap::north_sea();
        let s = j.spectrum_density(j.peak_omega());
        assert!(s > Fix128::ZERO);
    }

    #[test]
    fn spectrum_falls_off_high_freq() {
        let j = Jonswap::north_sea();
        let s_peak = j.spectrum_density(j.peak_omega());
        let s_high = j.spectrum_density(j.peak_omega() * Fix128::from_int(3));
        assert!(s_high < s_peak);
    }

    #[test]
    fn free_surface_zero_no_components() {
        let eta = free_surface_elevation(&[], Fix128::ZERO, Fix128::ZERO);
        assert_eq!(eta, Fix128::ZERO);
    }

    #[test]
    fn free_surface_single_wave_amplitude_at_zero() {
        // At x=t=0, cos(φ)=cos(0)=1 → η = A
        let c = WaveComponent {
            amplitude_m: Fix128::from_int(2),
            omega_rad_per_s: Fix128::ONE,
            wavenumber_rad_per_m: Fix128::ONE,
            phase_rad: Fix128::ZERO,
        };
        let eta = free_surface_elevation(&[c], Fix128::ZERO, Fix128::ZERO);
        assert!(approx_eq(
            eta,
            Fix128::from_int(2),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn free_surface_superposition_adds_amplitudes() {
        let c1 = WaveComponent {
            amplitude_m: Fix128::from_int(1),
            omega_rad_per_s: Fix128::ONE,
            wavenumber_rad_per_m: Fix128::ONE,
            phase_rad: Fix128::ZERO,
        };
        let c2 = WaveComponent {
            amplitude_m: Fix128::from_int(3),
            omega_rad_per_s: Fix128::from_int(2),
            wavenumber_rad_per_m: Fix128::ONE,
            phase_rad: Fix128::ZERO,
        };
        // Both at t=0, x=0 → 1 + 3 = 4
        let eta = free_surface_elevation(&[c1, c2], Fix128::ZERO, Fix128::ZERO);
        assert!(approx_eq(
            eta,
            Fix128::from_int(4),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn froude_krylov_scales_with_elevation() {
        // Positive wave elevation → larger buoyancy
        let f_low = froude_krylov_vertical_n(
            Fix128::from_int(1025),
            Fix128::from_ratio(981, 100),
            Fix128::from_int(10),
            Fix128::from_int(1),
            Fix128::ZERO,
        );
        let f_high = froude_krylov_vertical_n(
            Fix128::from_int(1025),
            Fix128::from_ratio(981, 100),
            Fix128::from_int(10),
            Fix128::from_int(1),
            Fix128::from_int(2),
        );
        assert!(f_high > f_low);
    }

    #[test]
    fn ship_response_heave_settles_toward_equilibrium() {
        let mut s = ShipResponse {
            heave_m: Fix128::from_int(1),
            heave_velocity_m_per_s: Fix128::ZERO,
            pitch_rad: Fix128::ZERO,
            pitch_velocity_rad_per_s: Fix128::ZERO,
        };
        // Apply zero external force with positive stiffness + damping
        for _ in 0..50 {
            s.advance(
                Fix128::ZERO,
                Fix128::ZERO,
                Fix128::from_int(1000),
                Fix128::from_int(1000),
                Fix128::from_int(100),
                Fix128::from_int(100),
                Fix128::from_int(500),
                Fix128::from_int(500),
                Fix128::from_ratio(1, 100),
            );
        }
        // Amplitude should decrease (damping)
        assert!(s.heave_m.abs() < Fix128::from_int(1));
    }

    #[test]
    fn ship_response_zero_mass_zero_motion() {
        let mut s = ShipResponse::default();
        s.advance(
            Fix128::from_int(100),
            Fix128::ZERO,
            Fix128::ZERO, // zero mass
            Fix128::from_int(1000),
            Fix128::from_int(100),
            Fix128::from_int(100),
            Fix128::from_int(10),
            Fix128::from_int(10),
            Fix128::from_ratio(1, 100),
        );
        assert_eq!(s.heave_m, Fix128::ZERO);
        assert_eq!(s.heave_velocity_m_per_s, Fix128::ZERO);
    }
}
