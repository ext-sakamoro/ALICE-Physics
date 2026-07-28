//! Combustion, Smoke and Fire (Arrhenius + Soot + Buoyancy)
//!
//! Phase G4 of the ALICE-Physics completeness project. Implements a
//! simplified single-step combustion model useful for VFX fire and
//! engineering fire-safety analyses:
//!
//! - **Arrhenius reaction rate**: `r = A · exp(−E_a / (R · T)) · [F] · [O]`
//! - **Heat release**: `q̇ = r · Δh_c` (J/(m³·s))
//! - **Soot generation** proportional to burn rate (`s_yield`).
//! - **Buoyancy source** for hot gas via Boussinesq approximation.
//!
//! Numerical `exp` is deterministic (range reduction + Taylor + squaring),
//! duplicated from `creep_longterm` to avoid module coupling.
//!
//! # References
//!
//! - Turns, *An Introduction to Combustion* 3rd ed. Ch. 5.
//! - Kuo, *Principles of Combustion* 2nd ed.
//! - Novozhilov, "Computational fluid dynamics modeling of compartment
//!   fires", Prog. Energy Combust. Sci. 27 (2001).

use crate::math::Fix128;
use crate::math_util::exp_fix;

// ============================================================================
// Reaction rate
// ============================================================================

/// Arrhenius reaction parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrheniusReaction {
    /// Pre-exponential factor `A` (1/s) — reaction attempt frequency.
    pub pre_exponential_a: Fix128,
    /// Activation energy `E_a` (J/mol).
    pub activation_energy_j_per_mol: Fix128,
    /// Universal gas constant `R = 8.314 J/(mol·K)`.
    pub gas_constant_j_per_mol_k: Fix128,
    /// Heat of combustion `Δh_c` (J/kg fuel).
    pub heat_of_combustion_j_per_kg: Fix128,
    /// Soot yield per unit fuel mass consumed (kg soot / kg fuel).
    pub soot_yield: Fix128,
}

impl ArrheniusReaction {
    /// Methane combustion at atmospheric conditions (representative fit,
    /// Turns Table 5-3 simplified single-step).
    #[must_use]
    pub fn methane_air() -> Self {
        Self {
            pre_exponential_a: Fix128::from_int(1_300_000_000),
            activation_energy_j_per_mol: Fix128::from_int(202_000),
            gas_constant_j_per_mol_k: Fix128::from_ratio(8314, 1000),
            heat_of_combustion_j_per_kg: Fix128::from_int(50_000_000),
            soot_yield: Fix128::from_ratio(15, 1000),
        }
    }

    /// Simplified PLA / air (from ALICE-Bamboo enclosure fire scenario).
    #[must_use]
    pub fn pla_air() -> Self {
        Self {
            pre_exponential_a: Fix128::from_int(500_000_000),
            activation_energy_j_per_mol: Fix128::from_int(150_000),
            gas_constant_j_per_mol_k: Fix128::from_ratio(8314, 1000),
            heat_of_combustion_j_per_kg: Fix128::from_int(18_000_000),
            soot_yield: Fix128::from_ratio(20, 1000),
        }
    }
}

/// Compute the volumetric reaction rate `r` (kg fuel / (m³·s)) at a point:
///
/// `r = A · exp(−E_a / (R · T)) · ρ_fuel · ρ_oxidizer / (ρ_fuel_ref²)`
///
/// The reference density normalises the units. `temperature_k` = current gas
/// temperature.  For a well-mixed reactor with `ρ_ref = 1 kg/m³`, this
/// reduces to `r = A · exp(...) · ρ_F · ρ_O`.
#[must_use]
pub fn reaction_rate_kg_per_m3_s(
    reaction: &ArrheniusReaction,
    temperature_k: Fix128,
    density_fuel: Fix128,
    density_oxidizer: Fix128,
) -> Fix128 {
    if temperature_k <= Fix128::ZERO || reaction.gas_constant_j_per_mol_k.is_zero() {
        return Fix128::ZERO;
    }
    let denom = reaction.gas_constant_j_per_mol_k * temperature_k;
    if denom.is_zero() {
        return Fix128::ZERO;
    }
    let exponent = Fix128::ZERO - reaction.activation_energy_j_per_mol / denom;
    let k_arrh = reaction.pre_exponential_a * exp_fix(exponent);
    k_arrh * density_fuel * density_oxidizer
}

/// Heat release per unit volume (J/(m³·s)).
#[must_use]
pub fn heat_release_j_per_m3_s(reaction: &ArrheniusReaction, reaction_rate: Fix128) -> Fix128 {
    reaction_rate * reaction.heat_of_combustion_j_per_kg
}

/// Soot generation rate (kg/(m³·s)).
#[must_use]
pub fn soot_generation_kg_per_m3_s(reaction: &ArrheniusReaction, reaction_rate: Fix128) -> Fix128 {
    reaction_rate * reaction.soot_yield
}

// ============================================================================
// Buoyancy source (Boussinesq)
// ============================================================================

/// Boussinesq buoyancy body force per unit volume (N/m³) for hot fluid:
///
/// `f_b = ρ_0 · β · (T − T_0) · g`
///
/// - `beta`: thermal expansion coefficient (1/K); air ≈ 3.4e-3 at 300 K.
#[must_use]
pub fn boussinesq_buoyancy_n_per_m3(
    reference_density: Fix128,
    beta_per_k: Fix128,
    temperature_delta_k: Fix128,
    gravity_m_per_s2: Fix128,
) -> Fix128 {
    reference_density * beta_per_k * temperature_delta_k * gravity_m_per_s2
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
    fn methane_reaction_positive_at_high_temp() {
        let r = ArrheniusReaction::methane_air();
        let rate = reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ONE, Fix128::ONE);
        assert!(rate > Fix128::ZERO);
    }

    #[test]
    fn reaction_rate_zero_at_zero_temp() {
        let r = ArrheniusReaction::methane_air();
        let rate = reaction_rate_kg_per_m3_s(&r, Fix128::ZERO, Fix128::ONE, Fix128::ONE);
        assert_eq!(rate, Fix128::ZERO);
    }

    #[test]
    fn reaction_rate_zero_at_no_fuel() {
        let r = ArrheniusReaction::methane_air();
        let rate = reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ZERO, Fix128::ONE);
        assert_eq!(rate, Fix128::ZERO);
    }

    #[test]
    fn reaction_rate_scales_with_temperature() {
        let r = ArrheniusReaction::methane_air();
        let rate_low =
            reaction_rate_kg_per_m3_s(&r, Fix128::from_int(1000), Fix128::ONE, Fix128::ONE);
        let rate_hi =
            reaction_rate_kg_per_m3_s(&r, Fix128::from_int(2000), Fix128::ONE, Fix128::ONE);
        assert!(rate_hi > rate_low);
    }

    #[test]
    fn heat_release_proportional() {
        let r = ArrheniusReaction::methane_air();
        // 1 kg/(m³·s) burnt → 50e6 J/(m³·s) released
        let q = heat_release_j_per_m3_s(&r, Fix128::ONE);
        assert!(approx_eq(
            q,
            Fix128::from_int(50_000_000),
            Fix128::from_int(1)
        ));
    }

    #[test]
    fn soot_generation_proportional() {
        let r = ArrheniusReaction::methane_air();
        // yield 0.015; 1 kg burnt → 0.015 kg soot
        let s = soot_generation_kg_per_m3_s(&r, Fix128::ONE);
        assert!(approx_eq(
            s,
            Fix128::from_ratio(15, 1000),
            Fix128::from_ratio(1, 10_000)
        ));
    }

    #[test]
    fn boussinesq_positive_upward_for_hot_fluid() {
        let f = boussinesq_buoyancy_n_per_m3(
            Fix128::from_int(1),
            Fix128::from_ratio(34, 10_000),
            Fix128::from_int(100),
            Fix128::from_int(10),
        );
        assert!(f > Fix128::ZERO);
    }

    #[test]
    fn boussinesq_negative_for_cold_fluid() {
        let f = boussinesq_buoyancy_n_per_m3(
            Fix128::from_int(1),
            Fix128::from_ratio(34, 10_000),
            Fix128::from_int(-50),
            Fix128::from_int(10),
        );
        assert!(f < Fix128::ZERO);
    }

    #[test]
    fn pla_has_lower_heat_than_methane() {
        // PLA 18 MJ/kg vs methane 50 MJ/kg
        let pla = ArrheniusReaction::pla_air();
        let ch4 = ArrheniusReaction::methane_air();
        assert!(pla.heat_of_combustion_j_per_kg < ch4.heat_of_combustion_j_per_kg);
    }

    #[test]
    fn pla_has_higher_soot_yield() {
        // Solid polymer combustion produces more soot than clean gas fuel
        let pla = ArrheniusReaction::pla_air();
        let ch4 = ArrheniusReaction::methane_air();
        assert!(pla.soot_yield > ch4.soot_yield);
    }
}
