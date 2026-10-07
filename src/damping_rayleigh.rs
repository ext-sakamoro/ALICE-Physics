//! Rayleigh (Proportional) Damping Model
//!
//! Phase D3 of the ALICE-Physics completeness project. The most widely used
//! damping model in structural dynamics: assumes the damping matrix `C` is a
//! linear combination of the mass matrix `M` and stiffness matrix `K`.
//!
//! `C = α·M + β·K`
//!
//! This form has the crucial property of being **classically damped** —
//! modes decouple into independent SDOF systems with a modal damping ratio
//! that varies with frequency as:
//!
//! `ζ_n = (α / (2·ω_n)) + (β·ω_n / 2)`
//!
//! `α` dominates low-frequency (soft-mode) damping; `β` dominates high-
//! frequency damping. Fit two target modes to determine `(α, β)`.
//!
//! # References
//!
//! - Clough & Penzien, *Dynamics of Structures* 3rd ed. Ch. 12.
//! - Chopra, *Dynamics of Structures* 5th ed. §11.4 (Rayleigh damping).
//! - Craig & Kurdila, *Fundamentals of Structural Dynamics* Ch. 8.

use crate::math::Fix128;

// ============================================================================
// Rayleigh coefficients
// ============================================================================

/// The two Rayleigh coefficients defining `C = α·M + β·K`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RayleighCoefficients {
    /// Mass-proportional coefficient (1/s). Damps low frequencies.
    pub alpha: Fix128,
    /// Stiffness-proportional coefficient (s). Damps high frequencies.
    pub beta: Fix128,
}

impl RayleighCoefficients {
    /// Modal damping ratio `ζ` at a specified angular frequency (rad/s).
    #[must_use]
    pub fn damping_ratio(&self, omega_rad_per_s: Fix128) -> Fix128 {
        if omega_rad_per_s.is_zero() {
            return Fix128::ZERO;
        }
        // ζ = α/(2ω) + β·ω/2
        let two = Fix128::from_int(2);
        self.alpha / (two * omega_rad_per_s) + self.beta * omega_rad_per_s / two
    }

    /// Fit `(α, β)` such that two target frequencies each achieve their
    /// specified damping ratios.
    ///
    /// Solves the 2×2 linear system:
    ///
    /// ```text
    /// | 1/(2ω_1)   ω_1/2 | |α|   |ζ_1|
    /// | 1/(2ω_2)   ω_2/2 | |β| = |ζ_2|
    /// ```
    ///
    /// Returns `Default` (zeros) if the target frequencies coincide (system
    /// singular).
    ///
    /// A Rayleigh damping matrix is positive semi-definite only for `α ≥ 0`
    /// and `β ≥ 0`. When the exact fit needs a negative coefficient (the high
    /// mode asked for a much smaller ratio than the low one, which would make
    /// `ζ` negative at high frequency) the result is the non-negative least
    /// squares fit of the two ratios instead: the better of `(α, 0)` and
    /// `(0, β)` with the free coefficient fitted to both modes, or `(0, 0)`.
    #[must_use]
    pub fn fit_two_modes(
        omega_1_rad_per_s: Fix128,
        zeta_1: Fix128,
        omega_2_rad_per_s: Fix128,
        zeta_2: Fix128,
    ) -> Self {
        if omega_1_rad_per_s.is_zero() || omega_2_rad_per_s.is_zero() {
            return Self::default();
        }
        if omega_1_rad_per_s == omega_2_rad_per_s {
            return Self::default();
        }
        // 2·ω·ζ = α + β·ω²  =>  linear system in (α, β) at each mode
        let two = Fix128::from_int(2);
        let w1 = omega_1_rad_per_s;
        let w2 = omega_2_rad_per_s;
        let w1_sq = w1 * w1;
        let w2_sq = w2 * w2;
        // From subtraction of the two rows: 2·(ω_2·ζ_2 − ω_1·ζ_1) = β·(ω_2² − ω_1²)
        let numerator_beta = two * (w2 * zeta_2 - w1 * zeta_1);
        let denominator_beta = w2_sq - w1_sq;
        if denominator_beta.is_zero() {
            return Self::default();
        }
        let beta = numerator_beta / denominator_beta;
        // Back-substitute: α = 2·ω_1·ζ_1 − β·ω_1²
        let alpha = two * w1 * zeta_1 - beta * w1_sq;
        if alpha >= Fix128::ZERO && beta >= Fix128::ZERO {
            return Self { alpha, beta };
        }
        Self::fit_non_negative([(w1, zeta_1), (w2, zeta_2)])
    }

    /// Least squares fit of `ζ(ω_i) = ζ_i` with `α, β ≥ 0` when the exact fit
    /// is infeasible: the optimum lies on a boundary, so each one-coefficient
    /// fit (`ζ = α/(2ω)`, or `ζ = βω/2`) is solved in closed form, clamped at
    /// 0, and the one with the smaller squared residual is kept (ties go to
    /// mass-proportional).
    fn fit_non_negative(modes: [(Fix128, Fix128); 2]) -> Self {
        let two = Fix128::from_int(2);
        // ζ = α·g with g = 1/(2ω), or ζ = β·h with h = ω/2: c = Σ ζ g / Σ g²
        let one_coefficient = |basis: &dyn Fn(Fix128) -> Fix128| {
            let (mut num, mut den) = (Fix128::ZERO, Fix128::ZERO);
            for &(w, z) in &modes {
                let g = basis(w);
                num = num + z * g;
                den = den + g * g;
            }
            let c = if den.is_zero() {
                Fix128::ZERO
            } else {
                num / den
            };
            let c = c.max(Fix128::ZERO);
            let residual = modes.iter().fold(Fix128::ZERO, |acc, &(w, z)| {
                let e = z - c * basis(w);
                acc + e * e
            });
            (c, residual)
        };
        let (alpha, res_alpha) = one_coefficient(&|w: Fix128| Fix128::ONE / (two * w));
        let (beta, res_beta) = one_coefficient(&|w: Fix128| w / two);
        if res_alpha <= res_beta {
            Self {
                alpha,
                beta: Fix128::ZERO,
            }
        } else {
            Self {
                alpha: Fix128::ZERO,
                beta,
            }
        }
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Convert cycles-per-second (Hz) to angular frequency (rad/s).
#[inline]
#[must_use]
pub fn hz_to_omega(hz: Fix128) -> Fix128 {
    hz * (Fix128::PI + Fix128::PI)
}

/// Convert angular frequency (rad/s) back to Hz.
#[inline]
#[must_use]
pub fn omega_to_hz(omega: Fix128) -> Fix128 {
    let two_pi = Fix128::PI + Fix128::PI;
    if two_pi.is_zero() {
        return Fix128::ZERO;
    }
    omega / two_pi
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
    fn default_coefficients_are_zero() {
        let r = RayleighCoefficients::default();
        assert_eq!(r.alpha, Fix128::ZERO);
        assert_eq!(r.beta, Fix128::ZERO);
    }

    #[test]
    fn damping_ratio_zero_omega_is_zero() {
        let r = RayleighCoefficients {
            alpha: Fix128::from_int(1),
            beta: Fix128::from_ratio(1, 1000),
        };
        assert_eq!(r.damping_ratio(Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn alpha_dominates_low_frequency() {
        let r = RayleighCoefficients {
            alpha: Fix128::from_int(10),
            beta: Fix128::ZERO,
        };
        // ζ = 10 / (2·ω). At ω=1, ζ=5 (very high). At ω=1000, ζ=0.005 (tiny).
        let low = r.damping_ratio(Fix128::from_int(1));
        let high = r.damping_ratio(Fix128::from_int(1000));
        assert!(low > high);
    }

    #[test]
    fn beta_dominates_high_frequency() {
        let r = RayleighCoefficients {
            alpha: Fix128::ZERO,
            beta: Fix128::from_ratio(1, 1000),
        };
        // ζ = β·ω/2. At ω=1, ζ=0.0005. At ω=1000, ζ=0.5.
        let low = r.damping_ratio(Fix128::from_int(1));
        let high = r.damping_ratio(Fix128::from_int(1000));
        assert!(high > low);
    }

    #[test]
    fn fit_two_modes_recovers_targets() {
        // Fit at ω=100 with ζ=0.05, ω=500 with ζ=0.02
        let w1 = Fix128::from_int(100);
        let w2 = Fix128::from_int(500);
        let z1 = Fix128::from_ratio(5, 100);
        let z2 = Fix128::from_ratio(2, 100);
        let r = RayleighCoefficients::fit_two_modes(w1, z1, w2, z2);
        // Verify that the resulting damping ratio at ω_1 and ω_2 matches z1, z2
        let z1_check = r.damping_ratio(w1);
        let z2_check = r.damping_ratio(w2);
        assert!(approx_eq(z1_check, z1, Fix128::from_ratio(1, 1000)));
        assert!(approx_eq(z2_check, z2, Fix128::from_ratio(1, 1000)));
    }

    #[test]
    fn fit_singular_returns_default() {
        // Same frequency → singular
        let w = Fix128::from_int(100);
        let r = RayleighCoefficients::fit_two_modes(
            w,
            Fix128::from_ratio(5, 100),
            w,
            Fix128::from_ratio(2, 100),
        );
        assert_eq!(r, RayleighCoefficients::default());
    }

    #[test]
    fn fit_zero_omega_returns_default() {
        let r = RayleighCoefficients::fit_two_modes(
            Fix128::ZERO,
            Fix128::from_ratio(1, 100),
            Fix128::from_int(100),
            Fix128::from_ratio(1, 100),
        );
        assert_eq!(r, RayleighCoefficients::default());
    }

    #[test]
    fn hz_omega_roundtrip() {
        let f = Fix128::from_int(50);
        let omega = hz_to_omega(f);
        let f_back = omega_to_hz(omega);
        assert!(approx_eq(f, f_back, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn combined_alpha_beta_intermediate_damping() {
        // Fit two modes, damping between them should be lower than either.
        let r = RayleighCoefficients::fit_two_modes(
            Fix128::from_int(100),
            Fix128::from_ratio(5, 100),
            Fix128::from_int(500),
            Fix128::from_ratio(5, 100),
        );
        // Middle frequency should still hit ~0.05 or less (Rayleigh curve
        // dips to a minimum between the two fit points).
        let z_mid = r.damping_ratio(Fix128::from_int(300));
        assert!(z_mid > Fix128::ZERO);
        assert!(z_mid <= Fix128::from_ratio(6, 100));
    }

    #[test]
    fn zero_damping_gives_zero_output() {
        let r = RayleighCoefficients::default();
        assert_eq!(r.damping_ratio(Fix128::from_int(50)), Fix128::ZERO);
    }

    /// `∂/∂c Σ (ζ_i − α/(2ω_i) − β ω_i/2)²` at `(α, β)` for the basis `c`
    /// (`g = 1/(2ω)` for α, `h = ω/2` for β): KKT needs it `≥ 0` where the
    /// coefficient is 0 and `= 0` where it is positive.
    fn residual_gradient(
        r: &RayleighCoefficients,
        modes: &[(Fix128, Fix128)],
        basis: impl Fn(Fix128) -> Fix128,
    ) -> Fix128 {
        modes.iter().fold(Fix128::ZERO, |acc, &(w, z)| {
            let e = z - r.damping_ratio(w);
            acc - Fix128::from_int(2) * e * basis(w)
        })
    }

    #[test]
    fn non_negative_fit_keeps_alpha_when_the_exact_beta_is_negative() {
        // ω = 1, 2 with ζ = 1, 0: the exact fit is β = −2/3 < 0. Mass-only:
        // α = Σζg/Σg² = 0.5/0.3125 = 1.6, residual 0.2² + 0.4² = 0.2;
        // stiffness-only: β = 0.5/1.25 = 0.4, residual 0.8² + 0.4² = 0.8
        let (w1, w2) = (Fix128::ONE, Fix128::from_int(2));
        let (z1, z2) = (Fix128::ONE, Fix128::ZERO);
        let r = RayleighCoefficients::fit_two_modes(w1, z1, w2, z2);
        let tol = Fix128::from_ratio(1, 1_000_000_000);
        assert!(approx_eq(r.alpha, Fix128::from_ratio(8, 5), tol), "{r:?}");
        assert_eq!(r.beta, Fix128::ZERO);
        // KKT: the α gradient vanishes, the β gradient is 0.6 ≥ 0
        let modes = [(w1, z1), (w2, z2)];
        let two = Fix128::from_int(2);
        let g_alpha = residual_gradient(&r, &modes, |w| Fix128::ONE / (two * w));
        let g_beta = residual_gradient(&r, &modes, |w| w / two);
        assert!(approx_eq(g_alpha, Fix128::ZERO, tol), "{g_alpha:?}");
        assert!(
            approx_eq(g_beta, Fix128::from_ratio(3, 5), tol),
            "{g_beta:?}"
        );
    }

    #[test]
    fn non_negative_fit_keeps_beta_when_the_exact_alpha_is_negative() {
        // ω = 1, 2 with ζ = 0, 1: exact β = 4/3, α = −4/3 < 0. Mass-only:
        // α = 0.25/0.3125 = 0.8, residual 0.4² + 0.8² = 0.8; stiffness-only:
        // β = 1/1.25 = 0.8, residual 0.4² + 0.2² = 0.2
        let (w1, w2) = (Fix128::ONE, Fix128::from_int(2));
        let (z1, z2) = (Fix128::ZERO, Fix128::ONE);
        let r = RayleighCoefficients::fit_two_modes(w1, z1, w2, z2);
        let tol = Fix128::from_ratio(1, 1_000_000_000);
        assert_eq!(r.alpha, Fix128::ZERO);
        assert!(approx_eq(r.beta, Fix128::from_ratio(4, 5), tol), "{r:?}");
        // KKT: the β gradient vanishes, the α gradient is 0.3 ≥ 0
        let modes = [(w1, z1), (w2, z2)];
        let two = Fix128::from_int(2);
        let g_alpha = residual_gradient(&r, &modes, |w| Fix128::ONE / (two * w));
        let g_beta = residual_gradient(&r, &modes, |w| w / two);
        assert!(approx_eq(g_beta, Fix128::ZERO, tol), "{g_beta:?}");
        assert!(
            approx_eq(g_alpha, Fix128::from_ratio(3, 10), tol),
            "{g_alpha:?}"
        );
    }

    #[test]
    fn non_negative_fit_of_negative_ratios_is_zero() {
        // ζ = −1 at both modes: both one-coefficient fits clamp at 0, the
        // residuals tie at 2 and the tie goes to mass-proportional (0, 0)
        let r = RayleighCoefficients::fit_two_modes(
            Fix128::ONE,
            Fix128::NEG_ONE,
            Fix128::from_int(2),
            Fix128::NEG_ONE,
        );
        assert_eq!(r, RayleighCoefficients::default());
    }
}
