//! Compressible Fluid Dynamics — Ideal Gas + Shock Relations
//!
//! Phase F3 of the ALICE-Physics completeness project. Covers the essentials
//! of compressible-flow analysis at engineering fidelity:
//!
//! - Ideal-gas equation of state and isentropic (adiabatic) relations.
//! - Speed of sound and Mach number.
//! - Rankine-Hugoniot **normal shock** jump conditions.
//! - Stagnation properties (Bernoulli in compressible form).
//! - 1D Riemann invariants (characteristic velocities).
//!
//! # Units
//!
//! - Pressure `p` in Pa (N/m²).
//! - Density `ρ` in kg/m³.
//! - Temperature `T` in K (absolute).
//! - Specific gas constant `R` in J/(kg·K).  For air R ≈ 287.
//! - Ratio of specific heats `γ` dimensionless.  Air ≈ 1.4, monatomic 5/3.
//!
//! # References
//!
//! - Anderson, *Modern Compressible Flow* 3rd ed. (canonical text).
//! - Liepmann & Roshko, *Elements of Gasdynamics* (Dover 2001).
//! - Shapiro, *The Dynamics and Thermodynamics of Compressible Fluid Flow*.

use crate::math::Fix128;

// ============================================================================
// Ideal gas equation of state
// ============================================================================

/// Ideal gas parameters (species-dependent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdealGas {
    /// Specific gas constant `R = R_universal / M_molar` (J/(kg·K)).
    pub gas_constant: Fix128,
    /// Ratio of specific heats `γ = c_p / c_v`.
    pub gamma: Fix128,
}

impl IdealGas {
    /// Dry air at standard atmosphere (R ≈ 287, γ = 1.4).
    #[must_use]
    pub fn air() -> Self {
        Self {
            gas_constant: Fix128::from_int(287),
            gamma: Fix128::from_ratio(14, 10),
        }
    }

    /// Helium (γ = 5/3, R ≈ 2077).
    #[must_use]
    pub fn helium() -> Self {
        Self {
            gas_constant: Fix128::from_int(2077),
            gamma: Fix128::from_ratio(5, 3),
        }
    }

    /// Pressure from ρ and T via `p = ρ·R·T` (Pa).
    #[inline]
    #[must_use]
    pub fn pressure(&self, density_kg_per_m3: Fix128, temperature_k: Fix128) -> Fix128 {
        density_kg_per_m3 * self.gas_constant * temperature_k
    }

    /// Density from p and T (kg/m³).
    #[inline]
    #[must_use]
    pub fn density(&self, pressure_pa: Fix128, temperature_k: Fix128) -> Fix128 {
        if temperature_k.is_zero() || self.gas_constant.is_zero() {
            return Fix128::ZERO;
        }
        pressure_pa / (self.gas_constant * temperature_k)
    }

    /// Temperature from p and ρ (K).
    #[inline]
    #[must_use]
    pub fn temperature(&self, pressure_pa: Fix128, density_kg_per_m3: Fix128) -> Fix128 {
        if density_kg_per_m3.is_zero() || self.gas_constant.is_zero() {
            return Fix128::ZERO;
        }
        pressure_pa / (density_kg_per_m3 * self.gas_constant)
    }

    /// Speed of sound `a = √(γ·R·T)` (m/s).
    #[must_use]
    pub fn speed_of_sound(&self, temperature_k: Fix128) -> Fix128 {
        (self.gamma * self.gas_constant * temperature_k).sqrt()
    }

    /// Speed of sound from pressure and density: `a = √(γ·p/ρ)`.
    #[must_use]
    pub fn speed_of_sound_from_pd(&self, pressure_pa: Fix128, density_kg_per_m3: Fix128) -> Fix128 {
        if density_kg_per_m3.is_zero() {
            return Fix128::ZERO;
        }
        (self.gamma * pressure_pa / density_kg_per_m3).sqrt()
    }

    /// Mach number `M = |u| / a`.
    #[must_use]
    pub fn mach_number(&self, velocity_m_per_s: Fix128, temperature_k: Fix128) -> Fix128 {
        let a = self.speed_of_sound(temperature_k);
        if a.is_zero() {
            return Fix128::ZERO;
        }
        velocity_m_per_s.abs() / a
    }
}

// ============================================================================
// Stagnation (isentropic) properties
// ============================================================================

/// Isentropic stagnation temperature `T_0 / T = 1 + (γ−1)/2 · M²`.
///
/// A ratio beyond the Fix128 range (`M ≳ 6.8·10⁹` for air) saturates at the
/// largest Fix128 (the most negative for `γ < 1`) instead of wrapping.
#[must_use]
pub fn stagnation_temp_ratio(gas: &IdealGas, mach: Fix128) -> Fix128 {
    let gm1 = gas.gamma - Fix128::ONE;
    let half = Fix128::from_ratio(1, 2);
    // (½(γ−1)·M)·M: the coefficient first, so the product only leaves the
    // range when the ratio itself does (M² alone overflows from M ≈ 3·10⁹)
    let term = (half * gm1)
        .checked_mul(mach)
        .and_then(|k_m| k_m.checked_mul(mach))
        .filter(|&t| t < RATIO_SATURATED - Fix128::ONE)
        .map(|t| t + Fix128::ONE);
    match term {
        Some(r) => r,
        None if gm1.is_negative() => -RATIO_SATURATED,
        None => RATIO_SATURATED,
    }
}

/// Isentropic stagnation pressure ratio `(p_0 / p) = (1 + (γ-1)/2·M²)^(γ/(γ-1))`
/// (Anderson eq. 3.18).
///
/// The exponent `γ/(γ−1)` (3.5 for air, 2.5 for monatomic gases) is applied
/// exactly via [`Fix128::powf_pos`] (deterministic Fix128 throughout). Before
/// 1.2.0 the exponent was rounded to 4 for every gas, which is 9.5 % high at
/// `M = 1` and 34 % at `M = 2` for air (`tests/engineering_oracles_fluid.rs`).
///
/// A ratio beyond the Fix128 range (`M ≳ 3·10³` for air) saturates at the
/// largest Fix128 instead of wrapping, so the ratio stays monotone in `|M|`.
#[must_use]
pub fn stagnation_pressure_ratio(gas: &IdealGas, mach: Fix128) -> Fix128 {
    let gm1 = gas.gamma - Fix128::ONE;
    if gm1.is_zero() {
        // γ = 1 (isothermal limit): the general formula's exponent γ/(γ-1)
        // diverges, but the limit itself is finite —
        // lim_{γ→1} (1 + (γ-1)/2·M²)^(γ/(γ-1)) = exp(M²/2) — so compute that
        // directly instead of silently returning 1.
        let half = Fix128::from_ratio(1, 2);
        // M² itself may not fit; past exp's own range (x > 43) exp saturates
        return match mach.checked_mul(mach) {
            Some(m_sq) => (half * m_sq).exp(),
            None => RATIO_SATURATED,
        };
    }
    let base = stagnation_temp_ratio(gas, mach);
    if gm1 < Fix128::ZERO || base <= Fix128::ZERO {
        return Fix128::ONE;
    }
    let exponent = gas.gamma / gm1;
    // ln of the result; past ln(largest Fix128) = 63 ln 2 the power would wrap
    if exponent * base.ln() >= ln_ratio_saturated() {
        return RATIO_SATURATED;
    }
    base.powf_pos(exponent)
}

/// The largest Fix128, where a ratio beyond the representable range saturates.
const RATIO_SATURATED: Fix128 = Fix128::from_raw(i64::MAX, u64::MAX);

/// `ln` of [`RATIO_SATURATED`], `63 ln 2 = 43.668…`, less `2⁻²⁰` so that the
/// rounding of `exp` / `powf_pos` at the edge cannot step past the range.
fn ln_ratio_saturated() -> Fix128 {
    Fix128::from_int(63) * Fix128::from_int(2).ln() - Fix128::from_raw(0, 1 << 44)
}

/// `x · M²`, or the largest Fix128 when it does not fit (`x ≥ 0`, `M > 1`).
fn times_mach_sq(x: Fix128, mach: Fix128) -> Fix128 {
    x.checked_mul(mach)
        .and_then(|t| t.checked_mul(mach))
        .unwrap_or(RATIO_SATURATED)
}

// ============================================================================
// Rankine-Hugoniot normal shock
// ============================================================================

/// State jump across a stationary normal shock.
///
/// Given upstream (state 1) properties and a shock Mach number `M_1 > 1`,
/// returns downstream (state 2) ratios `(ρ_2/ρ_1, p_2/p_1, T_2/T_1, M_2)`.
///
/// Formulas (Anderson eq. 3.51, 3.53, 3.55):
/// ```text
///   ρ_2/ρ_1 = (γ+1)·M_1² / ((γ-1)·M_1² + 2)
///   p_2/p_1 = 1 + (2γ/(γ+1))·(M_1² - 1)
///   T_2/T_1 = (p_2/p_1) · (ρ_1/ρ_2)
///   M_2²   = ((γ-1)·M_1² + 2) / (2γ·M_1² - (γ-1))
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShockJump {
    /// ρ_2 / ρ_1.
    pub density_ratio: Fix128,
    /// p_2 / p_1.
    pub pressure_ratio: Fix128,
    /// T_2 / T_1.
    pub temperature_ratio: Fix128,
    /// Downstream Mach.
    pub mach_downstream: Fix128,
}

/// Compute the normal-shock jump. Returns unit ratios when `M_1 ≤ 1`
/// (no shock exists for subsonic upstream).
///
/// When an `M_1²` term of the formulas would leave the Fix128 range (from
/// `M_1 ≈ 10⁹`, earlier for a larger `γ`), the same relations are evaluated
/// divided through by `M_1²`: `ρ_2/ρ_1 = (γ+1)/((γ−1) + 2/M_1²)`,
/// `M_2 = √(((γ−1) + 2/M_1²)/(2γ − (γ−1)/M_1²))`, and `p_2/p_1`, `T_2/T_1`
/// (which grow as `M_1²`) saturate at the largest Fix128 once they do not fit.
/// At `γ = 1` (isothermal) `ρ_2/ρ_1 = p_2/p_1 = M_1²` (saturating),
/// `T_2/T_1 = 1` and `M_2 = 1/M_1`. `γ < 1` is not a gas.
#[must_use]
pub fn normal_shock_jump(gas: &IdealGas, mach_upstream: Fix128) -> ShockJump {
    let m1 = mach_upstream;
    if m1 <= Fix128::ONE {
        return ShockJump {
            density_ratio: Fix128::ONE,
            pressure_ratio: Fix128::ONE,
            temperature_ratio: Fix128::ONE,
            mach_downstream: m1,
        };
    }
    let g = gas.gamma;
    let gp1 = g + Fix128::ONE;
    let gm1 = g - Fix128::ONE;
    let two = Fix128::from_int(2);
    let two_g_over_gp1 = two * g / gp1;
    // every M_1² product of the formulas below, or None when one leaves the range
    let fits = m1.checked_mul(m1).filter(|&m_sq| {
        gp1.checked_mul(m_sq).is_some()
            && gm1.checked_mul(m_sq).is_some()
            && g.double().checked_mul(m_sq).is_some()
            && two_g_over_gp1.checked_mul(m_sq).is_some()
    });
    let Some(m1_sq) = fits else {
        return strong_shock_jump(g, m1);
    };

    // ρ_2 / ρ_1
    let num_rho = gp1 * m1_sq;
    let den_rho = gm1 * m1_sq + two;
    let rho_ratio = if den_rho.is_zero() {
        Fix128::ONE
    } else {
        num_rho / den_rho
    };
    // p_2 / p_1
    let p_ratio = Fix128::ONE + two_g_over_gp1 * (m1_sq - Fix128::ONE);
    // T_2 / T_1
    let t_ratio = if rho_ratio.is_zero() {
        Fix128::ONE
    } else {
        p_ratio / rho_ratio
    };
    // M_2² = ((γ-1)M_1² + 2) / (2γM_1² - (γ-1))
    let num_m2 = gm1 * m1_sq + two;
    let den_m2 = two * g * m1_sq - gm1;
    let m2 = if den_m2.is_zero() {
        Fix128::ZERO
    } else {
        (num_m2 / den_m2).sqrt()
    };

    ShockJump {
        density_ratio: rho_ratio,
        pressure_ratio: p_ratio,
        temperature_ratio: t_ratio,
        mach_downstream: m2,
    }
}

/// The normal-shock relations divided through by `M_1²`, for an `M_1` whose
/// square terms do not fit Fix128 (see [`normal_shock_jump`]).
fn strong_shock_jump(g: Fix128, m1: Fix128) -> ShockJump {
    let gp1 = g + Fix128::ONE;
    let gm1 = g - Fix128::ONE;
    let two = Fix128::from_int(2);
    if gm1 <= Fix128::ZERO {
        // γ = 1: ρ_2/ρ_1 = p_2/p_1 = M_1², T_2/T_1 = 1, M_2 = 1/M_1
        return ShockJump {
            density_ratio: times_mach_sq(Fix128::ONE, m1),
            pressure_ratio: times_mach_sq(Fix128::ONE, m1),
            temperature_ratio: Fix128::ONE,
            mach_downstream: Fix128::ONE / m1,
        };
    }
    // 2/M² and (γ−1)/M², below one ulp once M_1 > 2^32
    let two_over = two / m1 / m1;
    let gm1_over = gm1 / m1 / m1;
    let k = two * g / gp1;
    // p = k M² + (1 − k);  T = (2γ − (γ−1)/M²)((γ−1) + 2/M²) M² / (γ+1)²
    let p = times_mach_sq(k, m1);
    let p = if p == RATIO_SATURATED {
        p
    } else {
        p + (Fix128::ONE - k)
    };
    let t_coef = (g.double() - gm1_over) * (gm1 + two_over) / (gp1 * gp1);
    ShockJump {
        density_ratio: gp1 / (gm1 + two_over),
        pressure_ratio: p,
        temperature_ratio: times_mach_sq(t_coef, m1),
        mach_downstream: ((gm1 + two_over) / (g.double() - gm1_over)).sqrt(),
    }
}

// ============================================================================
// Riemann invariants
// ============================================================================

/// Riemann invariants at a state (u, a) for a 1D isentropic flow:
///
/// `J⁺ = u + 2a/(γ−1)`  (right-moving characteristic)
/// `J⁻ = u − 2a/(γ−1)`  (left-moving characteristic)
///
/// At `γ = 1` (isothermal) `2a/(γ−1)` diverges: the term is the saturating
/// sentinel `Fix128::from_int(i64::MAX >> 8)` (signed like `a`), so
/// `J⁺ ≠ J⁻` still; `a = 0` gives `(u, u)`.
#[must_use]
pub fn riemann_invariants(
    gas: &IdealGas,
    velocity_m_per_s: Fix128,
    sound_speed_m_per_s: Fix128,
) -> (Fix128, Fix128) {
    let gm1 = gas.gamma - Fix128::ONE;
    if gm1.is_zero() {
        // γ = 1: 2a/(γ−1) diverges; report it as the saturating sentinel
        // rather than collapsing both invariants onto u
        if sound_speed_m_per_s.is_zero() {
            return (velocity_m_per_s, velocity_m_per_s);
        }
        let inf = Fix128::from_int(i64::MAX >> 8);
        let s = if sound_speed_m_per_s > Fix128::ZERO {
            inf
        } else {
            -inf
        };
        return (velocity_m_per_s + s, velocity_m_per_s - s);
    }
    let two_a_over_gm1 = Fix128::from_int(2) * sound_speed_m_per_s / gm1;
    (
        velocity_m_per_s + two_a_over_gm1,
        velocity_m_per_s - two_a_over_gm1,
    )
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
    fn air_preset_values() {
        let g = IdealGas::air();
        assert_eq!(g.gas_constant, Fix128::from_int(287));
        assert_eq!(g.gamma, Fix128::from_ratio(14, 10));
    }

    #[test]
    fn ideal_gas_roundtrip_pressure_density() {
        let g = IdealGas::air();
        let p = Fix128::from_int(101_325);
        let t = Fix128::from_int(288); // 15 °C
        let rho = g.density(p, t);
        // ρ = 101325 / (287·288) ≈ 1.225 kg/m³
        assert!(approx_eq(
            rho,
            Fix128::from_ratio(1225, 1000),
            Fix128::from_ratio(1, 100)
        ));
        let p_back = g.pressure(rho, t);
        assert!(approx_eq(p_back, p, Fix128::from_int(1)));
    }

    #[test]
    fn speed_of_sound_air_at_288k() {
        let g = IdealGas::air();
        let a = g.speed_of_sound(Fix128::from_int(288));
        // a = √(1.4 · 287 · 288) ≈ √(115718) ≈ 340 m/s
        assert!(approx_eq(a, Fix128::from_int(340), Fix128::from_int(2)));
    }

    #[test]
    fn mach_number_subsonic() {
        let g = IdealGas::air();
        let a = g.speed_of_sound(Fix128::from_int(288));
        let m = g.mach_number(a.half(), Fix128::from_int(288));
        // u = a/2 → M = 0.5
        assert!(approx_eq(
            m,
            Fix128::from_ratio(5, 10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn stagnation_temp_at_m1_is_1_2() {
        let g = IdealGas::air();
        // At M=1, ratio = 1 + 0.2·1 = 1.2
        let r = stagnation_temp_ratio(&g, Fix128::ONE);
        assert!(approx_eq(
            r,
            Fix128::from_ratio(12, 10),
            Fix128::from_ratio(1, 100)
        ));
    }

    #[test]
    fn stagnation_pressure_grows_with_mach() {
        let g = IdealGas::air();
        let p_0 = stagnation_pressure_ratio(&g, Fix128::ONE);
        let p_1 = stagnation_pressure_ratio(&g, Fix128::from_int(2));
        assert!(p_1 > p_0);
    }

    #[test]
    fn no_shock_subsonic_returns_identity() {
        let g = IdealGas::air();
        let s = normal_shock_jump(&g, Fix128::from_ratio(5, 10));
        assert_eq!(s.density_ratio, Fix128::ONE);
        assert_eq!(s.pressure_ratio, Fix128::ONE);
    }

    #[test]
    fn normal_shock_m2_p_ratio_air() {
        // Air, M_1 = 2 → known analytical values (Anderson App. A):
        // ρ_2/ρ_1 = 2.667, p_2/p_1 = 4.500, T_2/T_1 = 1.687, M_2 = 0.5774
        let g = IdealGas::air();
        let s = normal_shock_jump(&g, Fix128::from_int(2));
        assert!(approx_eq(
            s.density_ratio,
            Fix128::from_ratio(2667, 1000),
            Fix128::from_ratio(5, 100)
        ));
        assert!(approx_eq(
            s.pressure_ratio,
            Fix128::from_ratio(45, 10),
            Fix128::from_ratio(5, 100)
        ));
        assert!(approx_eq(
            s.temperature_ratio,
            Fix128::from_ratio(1687, 1000),
            Fix128::from_ratio(5, 100)
        ));
        assert!(approx_eq(
            s.mach_downstream,
            Fix128::from_ratio(5774, 10000),
            Fix128::from_ratio(5, 100)
        ));
    }

    #[test]
    fn strong_shock_asymptotic_density_ratio() {
        // As M_1 → ∞, ρ_2/ρ_1 → (γ+1)/(γ-1) = 6 for γ=1.4
        let g = IdealGas::air();
        let s = normal_shock_jump(&g, Fix128::from_int(100));
        assert!(s.density_ratio > Fix128::from_int(5));
        assert!(s.density_ratio < Fix128::from_int(7));
    }

    #[test]
    fn shock_downstream_subsonic() {
        // Downstream Mach of a normal shock is always < 1 (Anderson thm)
        let g = IdealGas::air();
        for m in [
            Fix128::from_ratio(11, 10),
            Fix128::from_int(2),
            Fix128::from_int(5),
        ] {
            let s = normal_shock_jump(&g, m);
            assert!(s.mach_downstream < Fix128::ONE);
        }
    }

    #[test]
    fn riemann_invariant_sum() {
        // (J⁺ + J⁻)/2 = u
        let g = IdealGas::air();
        let u = Fix128::from_int(50);
        let a = Fix128::from_int(340);
        let (jp, jm) = riemann_invariants(&g, u, a);
        let avg = (jp + jm).half();
        assert!(approx_eq(avg, u, Fix128::from_ratio(1, 100)));
    }

    #[test]
    fn riemann_invariant_difference_sound() {
        // (J⁺ - J⁻)/2 = 2a/(γ-1)
        let g = IdealGas::air();
        let u = Fix128::from_int(50);
        let a = Fix128::from_int(340);
        let (jp, jm) = riemann_invariants(&g, u, a);
        let diff = (jp - jm).half();
        let expected = Fix128::from_int(2) * a / (g.gamma - Fix128::ONE);
        assert!(approx_eq(diff, expected, Fix128::from_ratio(1, 10)));
    }

    #[test]
    fn helium_faster_than_air_same_temp() {
        let air = IdealGas::air();
        let he = IdealGas::helium();
        let t = Fix128::from_int(300);
        let a_air = air.speed_of_sound(t);
        let a_he = he.speed_of_sound(t);
        // Helium has smaller molecular mass → faster sound
        assert!(a_he > a_air);
    }

    #[test]
    fn speed_of_sound_from_pd_matches_sqrt_gamma_p_over_rho() {
        let g = IdealGas::air();

        // Exact case: γ·p/ρ = 1.4 · 100_000 / 1.4 = 100_000 → a = √100000
        // Choose ρ = 1.4 so the ratio is an exact integer in Fix128.
        let p = Fix128::from_int(100_000);
        let rho = Fix128::from_ratio(14, 10);
        let a = g.speed_of_sound_from_pd(p, rho);
        let expected = Fix128::from_int(100_000).sqrt();
        // The only non-exact step is the Fix128 sqrt itself; the argument
        // 1.4·100000/1.4 may differ from 100000 by ~2^-64, so allow 1e-9.
        assert!(
            approx_eq(a, expected, Fix128::from_ratio(1, 1_000_000_000)),
            "a = {} expected = {}",
            a.to_f64(),
            expected.to_f64()
        );
        // √100000 ≈ 316.2278
        assert!(approx_eq(
            a,
            Fix128::from_ratio(3_162_278, 10_000),
            Fix128::from_ratio(1, 1000)
        ));

        // Consistency with the temperature form: for an ideal gas
        // p/ρ = R·T, so a(p, ρ) == a(T) when p = ρ·R·T.
        let t = Fix128::from_int(288);
        let rho2 = Fix128::from_ratio(1225, 1000);
        let p2 = g.pressure(rho2, t);
        let a_pd = g.speed_of_sound_from_pd(p2, rho2);
        let a_t = g.speed_of_sound(t);
        assert!(
            approx_eq(a_pd, a_t, Fix128::from_ratio(1, 1_000_000)),
            "a_pd = {} a_T = {}",
            a_pd.to_f64(),
            a_t.to_f64()
        );
        // ≈ 340 m/s at 15 °C
        assert!(approx_eq(a_pd, Fix128::from_int(340), Fix128::from_int(2)));

        // Helium at the same p, ρ: γ = 5/3 > 1.4 → faster.
        let he = IdealGas::helium();
        assert!(he.speed_of_sound_from_pd(p, rho) > a);

        // Scaling law: a ∝ √p at fixed ρ — quadrupling p doubles a.
        let a4 = g.speed_of_sound_from_pd(p * Fix128::from_int(4), rho);
        assert!(approx_eq(
            a4,
            a * Fix128::from_int(2),
            Fix128::from_ratio(1, 1_000_000)
        ));

        // ρ = 0 guard returns 0 rather than dividing by zero.
        assert_eq!(g.speed_of_sound_from_pd(p, Fix128::ZERO), Fix128::ZERO);
    }

    /// Relative difference `|a − b| / |b|` in f64 (test-side check only).
    fn rel(a: Fix128, b: f64) -> f64 {
        ((a.to_f64() - b) / b).abs()
    }

    #[test]
    fn zero_guards_of_the_state_equation_return_zero() {
        let g = IdealGas::air();
        let p = Fix128::from_int(101_325);
        assert_eq!(g.density(p, Fix128::ZERO), Fix128::ZERO);
        assert_eq!(g.temperature(p, Fix128::ZERO), Fix128::ZERO);
        // T = 0 has no sound speed, so no Mach number either
        assert_eq!(
            g.mach_number(Fix128::from_int(10), Fix128::ZERO),
            Fix128::ZERO
        );
        // T = p / (ρ R) closes the state equation p = ρ R T
        let t = g.temperature(Fix128::from_int(287 * 300), Fix128::ONE);
        assert_eq!(t, Fix128::from_int(300));
    }

    #[test]
    fn stagnation_temperature_ratio_saturates_beyond_the_documented_mach() {
        // air: (γ−1)/2 · M² leaves Fix128 for M ≳ 6.8e9; M = 1e10 saturates at MAX
        let air = IdealGas::air();
        let m = Fix128::from_int(10_000_000_000);
        assert_eq!(stagnation_temp_ratio(&air, m), RATIO_SATURATED);
        // γ < 1: the ratio decreases, so it saturates at the most negative value
        let sub = IdealGas {
            gas_constant: Fix128::from_int(287),
            gamma: Fix128::from_ratio(1, 2),
        };
        assert_eq!(stagnation_temp_ratio(&sub, m), -RATIO_SATURATED);
        // and the pressure ratio of a γ < 1 gas is the documented 1
        assert_eq!(
            stagnation_pressure_ratio(&sub, Fix128::from_int(2)),
            Fix128::ONE
        );
    }

    #[test]
    fn stagnation_pressure_ratio_isothermal_limit_and_saturation() {
        let iso = IdealGas {
            gas_constant: Fix128::from_int(287),
            gamma: Fix128::ONE,
        };
        // γ = 1: p0/p = exp(M²/2); M = 2 gives e² (to Fix128 exp's ~1e-8)
        let e_sq = core::f64::consts::E * core::f64::consts::E;
        let r = stagnation_pressure_ratio(&iso, Fix128::from_int(2));
        assert!(rel(r, e_sq) < 1e-7, "{} vs {e_sq}", r.to_f64());
        // M² beyond Fix128 saturates at MAX
        let m = Fix128::from_int(10_000_000_000);
        assert_eq!(stagnation_pressure_ratio(&iso, m), RATIO_SATURATED);
        // air at M = 1e4: (1 + 0.2·1e8)^3.5 ≈ 3.6e25 > 2^63, so MAX
        let air = IdealGas::air();
        assert_eq!(
            stagnation_pressure_ratio(&air, Fix128::from_int(10_000)),
            RATIO_SATURATED
        );
        // air at M = 1: 1.2^3.5 (Anderson table A.1, 1.8929)
        assert!(
            rel(
                stagnation_pressure_ratio(&air, Fix128::ONE),
                1.892_929_158_737_854
            ) < 1e-9
        );
    }

    #[test]
    fn isothermal_normal_shock_is_m_squared_with_unit_temperature() {
        // γ = 1: ρ2/ρ1 = p2/p1 = M1², T2/T1 = 1, M2 = 1/M1
        let iso = IdealGas {
            gas_constant: Fix128::from_int(287),
            gamma: Fix128::ONE,
        };
        let s = normal_shock_jump(&iso, Fix128::from_int(2));
        assert_eq!(s.density_ratio, Fix128::from_int(4));
        assert_eq!(s.pressure_ratio, Fix128::from_int(4));
        assert_eq!(s.temperature_ratio, Fix128::ONE);
        assert!(rel(s.mach_downstream, 0.5) < 1e-12);
        // beyond the M1² range: density and pressure saturate, M2 = 1/M1
        let m = Fix128::from_int(10_000_000_000);
        let s = normal_shock_jump(&iso, m);
        assert_eq!(s.density_ratio, RATIO_SATURATED);
        assert_eq!(s.pressure_ratio, RATIO_SATURATED);
        assert_eq!(s.temperature_ratio, Fix128::ONE);
        assert!(rel(s.mach_downstream, 1e-10) < 1e-6);
    }

    #[test]
    fn strong_shock_follows_rankine_hugoniot_divided_by_m_squared() {
        let air = IdealGas::air();
        // M1 = 2e9: M1² = 4e18 fits, (γ+1) M1² = 9.6e18 does not
        let m1 = 2_000_000_000_f64;
        let s = normal_shock_jump(&air, Fix128::from_int(2_000_000_000));
        // ρ2/ρ1 = (γ+1)M²/((γ−1)M² + 2) → 6 as M → ∞
        let rho = 2.4 * m1 * m1 / (0.4 * m1 * m1 + 2.0);
        assert!(rel(s.density_ratio, rho) < 1e-9);
        // p2/p1 = 1 + 2γ/(γ+1)(M² − 1)
        let p = 1.0 + 2.8 / 2.4 * (m1 * m1 - 1.0);
        assert!(rel(s.pressure_ratio, p) < 1e-9);
        // T2/T1 = (p2/p1)(ρ1/ρ2)
        assert!(rel(s.temperature_ratio, p / rho) < 1e-9);
        // M2² = ((γ−1)M² + 2)/(2γM² − (γ−1)) → (γ−1)/(2γ) = 1/7
        let m2 = ((0.4 * m1 * m1 + 2.0) / (2.8 * m1 * m1 - 0.4)).sqrt();
        assert!(rel(s.mach_downstream, m2) < 1e-9);

        // M1 = 1e10: p2/p1 and T2/T1 leave the range and saturate, while the
        // density ratio and M2 approach their limits 6 and √(1/7)
        let s = normal_shock_jump(&air, Fix128::from_int(10_000_000_000));
        assert_eq!(s.pressure_ratio, RATIO_SATURATED);
        assert_eq!(s.temperature_ratio, RATIO_SATURATED);
        assert!(rel(s.density_ratio, 6.0) < 1e-9);
        assert!(rel(s.mach_downstream, (1.0_f64 / 7.0).sqrt()) < 1e-9);
    }

    #[test]
    fn isothermal_riemann_invariants_use_the_signed_sentinel() {
        let iso = IdealGas {
            gas_constant: Fix128::from_int(287),
            gamma: Fix128::ONE,
        };
        let u = Fix128::from_int(3);
        // a = 0: both invariants are u
        assert_eq!(riemann_invariants(&iso, u, Fix128::ZERO), (u, u));
        let inf = Fix128::from_int(i64::MAX >> 8);
        assert_eq!(
            riemann_invariants(&iso, u, Fix128::from_int(340)),
            (u + inf, u - inf)
        );
        assert_eq!(
            riemann_invariants(&iso, u, Fix128::from_int(-340)),
            (u - inf, u + inf)
        );
    }
}
