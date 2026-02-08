//! Phase Change Simulation Modifier
//!
//! Temperature-driven phase transitions between solid, liquid, and gas:
//! - **Melting** (solid → liquid): Surface softens, material flows downward
//! - **Vaporization** (liquid → gas): Material expands, then dissipates
//! - **Solidification** (liquid → solid): Material grows rigid, surface freezes
//! - **Condensation** (gas → liquid): Gas contracts back to liquid
//!
//! # Physics Model
//!
//! Each cell tracks: temperature, phase state, and latent heat buffer.
//! At phase boundary temperatures, latent heat absorbs/releases energy
//! without changing temperature (plateau behavior). Phase transitions
//! modify the SDF: liquid flows downward, gas expands outward.
//!
//! Author: Moroya Sakamoto

use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Phase State
// ============================================================================

/// Phase of matter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    /// Rigid — no SDF modification
    Solid = 0,
    /// Flows downward under gravity
    Liquid = 1,
    /// Expands outward, dissipates over time
    Gas = 2,
}

// ============================================================================
// Configuration
// ============================================================================

/// Phase change modifier configuration
#[derive(Clone, Copy, Debug)]
pub struct PhaseChangeConfig {
    /// Temperature at which solid melts to liquid
    pub melt_temperature: f32,
    /// Temperature at which liquid vaporizes to gas
    pub boil_temperature: f32,
    /// Latent heat of fusion (energy absorbed during melting)
    pub latent_heat_fusion: f32,
    /// Latent heat of vaporization
    pub latent_heat_vaporization: f32,
    /// Thermal diffusion rate
    pub diffusion_rate: f32,
    /// Ambient temperature
    pub ambient_temperature: f32,
    /// Cooling rate toward ambient
    pub cooling_rate: f32,
    /// Liquid flow speed (gravity-driven downward)
    pub liquid_flow_speed: f32,
    /// Gas expansion rate (SDF offset per second)
    pub gas_expansion_rate: f32,
    /// Gas dissipation rate (material removal per second)
    pub gas_dissipation_rate: f32,
    /// Maximum SDF offset from phase changes
    pub max_offset: f32,
}

impl Default for PhaseChangeConfig {
    fn default() -> Self {
        Self {
            melt_temperature: 200.0,
            boil_temperature: 500.0,
            latent_heat_fusion: 50.0,
            latent_heat_vaporization: 100.0,
            diffusion_rate: 0.5,
            ambient_temperature: 20.0,
            cooling_rate: 0.1,
            liquid_flow_speed: 1.0,
            gas_expansion_rate: 0.5,
            gas_dissipation_rate: 0.3,
            max_offset: 3.0,
        }
    }
}

// ============================================================================
// Phase Change Modifier
// ============================================================================

/// Phase-change SDF modifier with solid/liquid/gas transitions
pub struct PhaseChangeModifier {
    /// Configuration
    pub config: PhaseChangeConfig,
    /// Temperature field
    pub temperature: ScalarField3D,
    /// Phase state per cell (stored as f32: 0=Solid, 1=Liquid, 2=Gas)
    pub phase: ScalarField3D,
    /// Latent heat buffer (energy absorbed/released during phase transition)
    pub latent_heat: ScalarField3D,
    /// Accumulated SDF offset (positive = material removed)
    pub sdf_offset: ScalarField3D,
    /// Whether modifier is enabled
    pub enabled: bool,
}

impl PhaseChangeModifier {
    /// Create a new phase change modifier
    pub fn new(
        config: PhaseChangeConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        Self {
            config,
            temperature: ScalarField3D::new_filled(
                resolution, resolution, resolution,
                min, max,
                config.ambient_temperature,
            ),
            phase: ScalarField3D::new(resolution, resolution, resolution, min, max),
            latent_heat: ScalarField3D::new(resolution, resolution, resolution, min, max),
            sdf_offset: ScalarField3D::new(resolution, resolution, resolution, min, max),
            enabled: true,
        }
    }

    /// Apply heat at a position
    pub fn apply_heat_at(&mut self, x: f32, y: f32, z: f32, amount: f32, radius: f32) {
        self.temperature.splat(x, y, z, amount, radius);
    }

    /// Get temperature at a point
    pub fn temperature_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.temperature.sample(x, y, z)
    }

    /// Get phase at a point
    pub fn phase_at(&self, x: f32, y: f32, z: f32) -> Phase {
        let v = self.phase.sample(x, y, z);
        if v < 0.5 {
            Phase::Solid
        } else if v < 1.5 {
            Phase::Liquid
        } else {
            Phase::Gas
        }
    }

    /// Process phase transitions
    fn process_transitions(&mut self, dt: f32) {
        let melt_t = self.config.melt_temperature;
        let boil_t = self.config.boil_temperature;
        let lh_fusion = self.config.latent_heat_fusion;
        let lh_vaporization = self.config.latent_heat_vaporization;

        let n = self.temperature.cell_count();
        for i in 0..n {
            let temp = self.temperature.data[i];
            let phase_val = self.phase.data[i];
            let lh = self.latent_heat.data[i];

            // Current phase
            let is_solid = phase_val < 0.5;
            let is_liquid = phase_val >= 0.5 && phase_val < 1.5;
            let _is_gas = phase_val >= 1.5;

            // Solid → Liquid transition
            if is_solid && temp >= melt_t {
                let excess = temp - melt_t;
                let new_lh = lh + excess * dt;
                if new_lh >= lh_fusion {
                    self.phase.data[i] = 1.0; // Liquid
                    self.latent_heat.data[i] = 0.0;
                } else {
                    self.latent_heat.data[i] = new_lh;
                }
            }

            // Re-check phase after possible solid→liquid transition
            let phase_val = self.phase.data[i];
            let is_liquid = phase_val >= 0.5 && phase_val < 1.5;

            // Liquid → Gas transition
            if is_liquid && temp >= boil_t {
                let excess = temp - boil_t;
                let new_lh = self.latent_heat.data[i] + excess * dt;
                if new_lh >= lh_vaporization {
                    self.phase.data[i] = 2.0; // Gas
                    self.latent_heat.data[i] = 0.0;
                } else {
                    self.latent_heat.data[i] = new_lh;
                }
            }

            // Re-check phase for cooling transitions
            let phase_val = self.phase.data[i];
            let is_liquid = phase_val >= 0.5 && phase_val < 1.5;
            let is_gas = phase_val >= 1.5;

            // Gas → Liquid transition (cooling)
            if is_gas && temp < boil_t {
                self.phase.data[i] = 1.0; // Condense to liquid
                self.latent_heat.data[i] = 0.0;
            }

            // Liquid → Solid transition (cooling)
            if is_liquid && temp < melt_t {
                self.phase.data[i] = 0.0; // Solidify
                self.latent_heat.data[i] = 0.0;
            }
        }
    }

    /// Accumulate SDF offsets based on phase state
    fn accumulate_offsets(&mut self, dt: f32) {
        let liquid_flow = self.config.liquid_flow_speed;
        let gas_expand = self.config.gas_expansion_rate;
        let gas_dissipate = self.config.gas_dissipation_rate;
        let max_offset = self.config.max_offset;

        let n = self.sdf_offset.cell_count();
        for i in 0..n {
            let phase_val = self.phase.data[i];

            if phase_val >= 1.5 {
                // Gas: expand and dissipate (increase SDF = material removed)
                self.sdf_offset.data[i] =
                    (self.sdf_offset.data[i] + (gas_expand + gas_dissipate) * dt).min(max_offset);
            } else if phase_val >= 0.5 {
                // Liquid: slight material softening
                self.sdf_offset.data[i] =
                    (self.sdf_offset.data[i] + liquid_flow * 0.1 * dt).min(max_offset);
            }
            // Solid: no offset change
        }

        // Liquid gravity flow: shift liquid offset downward
        if liquid_flow > 0.0 {
            let nx = self.sdf_offset.nx;
            let ny = self.sdf_offset.ny;
            let nz = self.sdf_offset.nz;
            for iz in 0..nz {
                for ix in 0..nx {
                    for iy in 1..ny {
                        let idx = self.phase.index(ix, iy, iz);
                        let below_idx = self.phase.index(ix, iy - 1, iz);
                        let phase_above = self.phase.data[idx];
                        // Transfer liquid offset downward
                        if phase_above >= 0.5 && phase_above < 1.5 {
                            let transfer = self.sdf_offset.data[idx] * liquid_flow * dt * 0.5;
                            if transfer > 0.0 {
                                self.sdf_offset.data[below_idx] =
                                    (self.sdf_offset.data[below_idx] + transfer).min(max_offset);
                                self.sdf_offset.data[idx] -= transfer * 0.3;
                            }
                        }
                    }
                }
            }
        }
    }
}

impl PhysicsModifier for PhaseChangeModifier {
    #[inline]
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32 {
        if !self.enabled {
            return original_dist;
        }

        let offset = self.sdf_offset.sample(x, y, z);
        original_dist + offset
    }

    fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        // 1. Diffuse temperature
        self.temperature.diffuse(dt, self.config.diffusion_rate);

        // 2. Cool toward ambient
        self.temperature.decay_toward(
            self.config.ambient_temperature,
            self.config.cooling_rate,
            dt,
        );

        // 3. Process phase transitions
        self.process_transitions(dt);

        // 4. Accumulate SDF offsets
        self.accumulate_offsets(dt);
    }

    fn name(&self) -> &str {
        "phase_change"
    }

    fn is_active(&self) -> bool {
        self.enabled
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_change_no_heat() {
        let config = PhaseChangeConfig::default();
        let modifier = PhaseChangeModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // No heat: everything is solid, no offset
        let d = modifier.modify_distance(0.0, 0.0, 0.0, -1.0);
        assert!((d - (-1.0)).abs() < 0.01, "No heat should not change SDF, got {}", d);
        assert_eq!(modifier.phase_at(0.0, 0.0, 0.0), Phase::Solid);
    }

    #[test]
    fn test_melting_transition() {
        let config = PhaseChangeConfig {
            melt_temperature: 100.0,
            latent_heat_fusion: 10.0,
            ambient_temperature: 20.0,
            cooling_rate: 0.0, // No cooling
            diffusion_rate: 0.0, // No diffusion (prevent re-solidification)
            ..Default::default()
        };
        let mut modifier = PhaseChangeModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Apply heat well above melt temperature (large radius for low-res grid)
        modifier.apply_heat_at(0.0, 0.0, 0.0, 2000.0, 1.5);

        // Simulate enough steps for transition
        for _ in 0..100 {
            modifier.update(0.016);
        }

        let phase = modifier.phase_at(0.0, 0.0, 0.0);
        assert!(
            phase == Phase::Liquid || phase == Phase::Gas,
            "High heat should melt material, got {:?}", phase
        );
    }

    #[test]
    fn test_vaporization_transition() {
        let config = PhaseChangeConfig {
            melt_temperature: 50.0,
            boil_temperature: 100.0,
            latent_heat_fusion: 1.0,
            latent_heat_vaporization: 1.0,
            ambient_temperature: 20.0,
            cooling_rate: 0.0,
            diffusion_rate: 0.0,
            ..Default::default()
        };
        let mut modifier = PhaseChangeModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Apply extreme heat (large radius for low-res grid)
        modifier.apply_heat_at(0.0, 0.0, 0.0, 5000.0, 1.5);

        for _ in 0..200 {
            modifier.update(0.016);
        }

        let phase = modifier.phase_at(0.0, 0.0, 0.0);
        assert_eq!(phase, Phase::Gas, "Extreme heat should vaporize, got {:?}", phase);

        // Gas should have increased SDF offset
        let offset = modifier.sdf_offset.sample(0.0, 0.0, 0.0);
        assert!(offset > 0.0, "Gas should add SDF offset, got {}", offset);
    }

    #[test]
    fn test_solidification() {
        let config = PhaseChangeConfig {
            melt_temperature: 50.0,
            latent_heat_fusion: 1.0,
            ambient_temperature: 20.0,
            cooling_rate: 0.0,
            diffusion_rate: 0.0,
            ..Default::default()
        };
        let mut modifier = PhaseChangeModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Heat to melt (large radius for low-res grid)
        modifier.apply_heat_at(0.0, 0.0, 0.0, 2000.0, 1.5);
        for _ in 0..50 {
            modifier.update(0.016);
        }

        // Should be liquid by now
        let phase_hot = modifier.phase_at(0.0, 0.0, 0.0);
        assert!(
            phase_hot == Phase::Liquid || phase_hot == Phase::Gas,
            "Should be liquid/gas after heating, got {:?}", phase_hot
        );

        // Enable fast cooling and cool down
        modifier.config.cooling_rate = 5.0;
        for _ in 0..500 {
            modifier.update(0.016);
        }

        let phase_cold = modifier.phase_at(0.0, 0.0, 0.0);
        assert_eq!(phase_cold, Phase::Solid, "Should solidify after cooling, got {:?}", phase_cold);
    }
}
