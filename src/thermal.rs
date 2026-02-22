//! Thermal Simulation Modifier
//!
//! Heat diffusion drives SDF shape changes:
//! - **Melt**: Above melt temperature, surface recedes + drips downward
//! - **Thermal Expansion**: Warm areas expand outward
//! - **Freeze**: Below freeze threshold, crystalline growth on surface
//!
//! # Physics Model
//!
//! Temperature field evolves via heat equation: dT/dt = k * laplacian(T)
//! with source terms (heat sources, contact friction, radiation cooling).
//! The temperature field then modulates the SDF distance.
//!
//! Author: Moroya Sakamoto

use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Thermal simulation configuration
#[derive(Clone, Copy, Debug)]
pub struct ThermalConfig {
    /// Thermal diffusion rate (conductivity)
    pub diffusion_rate: f32,
    /// Ambient temperature (environment)
    pub ambient_temperature: f32,
    /// Radiation cooling rate (toward ambient)
    pub cooling_rate: f32,
    /// Temperature at which material begins to melt
    pub melt_temperature: f32,
    /// Rate of surface recession when melting (distance/second/degree above melt)
    pub melt_rate: f32,
    /// Gravity droop strength (how much melt flows downward)
    pub droop_strength: f32,
    /// Thermal expansion coefficient (expansion per degree above ambient)
    pub expansion_coefficient: f32,
    /// Temperature at which freeze growth begins
    pub freeze_temperature: f32,
    /// Rate of surface growth when freezing
    pub freeze_rate: f32,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            diffusion_rate: 0.5,
            ambient_temperature: 20.0,
            cooling_rate: 0.1,
            melt_temperature: 200.0,
            melt_rate: 0.02,
            droop_strength: 0.5,
            expansion_coefficient: 0.0001,
            freeze_temperature: -10.0,
            freeze_rate: 0.01,
        }
    }
}

/// Heat source types
#[derive(Clone, Copy, Debug)]
pub enum HeatSource {
    /// Point heat source
    Point {
        /// Position X (world space)
        x: f32,
        /// Position Y (world space)
        y: f32,
        /// Position Z (world space)
        z: f32,
        /// Heat power (degrees/second at center)
        power: f32,
        /// Influence radius
        radius: f32,
    },
    /// Uniform heat in a volume
    Volume {
        /// Min corner
        min: (f32, f32, f32),
        /// Max corner
        max: (f32, f32, f32),
        /// Heat power
        power: f32,
    },
}

// ============================================================================
// Thermal Modifier
// ============================================================================

/// Thermal simulation that modifies SDF based on temperature
///
/// Supports melting (surface recession + gravity droop),
/// thermal expansion, and freeze growth.
pub struct ThermalModifier {
    /// Configuration
    pub config: ThermalConfig,
    /// Temperature field
    pub temperature: ScalarField3D,
    /// Accumulated melt deformation (positive = material removed)
    pub melt_accumulator: ScalarField3D,
    /// Active heat sources
    pub heat_sources: Vec<HeatSource>,
    /// Whether modifier is enabled
    pub enabled: bool,
}

impl ThermalModifier {
    /// Create a new thermal modifier
    pub fn new(
        config: ThermalConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        Self {
            config,
            temperature: ScalarField3D::new_filled(
                resolution,
                resolution,
                resolution,
                min,
                max,
                config.ambient_temperature,
            ),
            melt_accumulator: ScalarField3D::new(resolution, resolution, resolution, min, max),
            heat_sources: Vec::new(),
            enabled: true,
        }
    }

    /// Add a point heat source
    pub fn add_heat_point(&mut self, x: f32, y: f32, z: f32, power: f32, radius: f32) {
        self.heat_sources.push(HeatSource::Point {
            x,
            y,
            z,
            power,
            radius,
        });
    }

    /// Apply heat at a position (e.g., from friction, laser, etc.)
    pub fn apply_heat_at(&mut self, x: f32, y: f32, z: f32, amount: f32, radius: f32) {
        self.temperature.splat(x, y, z, amount, radius);
    }

    /// Get temperature at a point
    pub fn temperature_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.temperature.sample(x, y, z)
    }

    /// Apply heat sources to the temperature field
    fn apply_heat_sources(&mut self, dt: f32) {
        for source in &self.heat_sources {
            match *source {
                HeatSource::Point {
                    x,
                    y,
                    z,
                    power,
                    radius,
                } => {
                    self.temperature.splat(x, y, z, power * dt, radius);
                }
                HeatSource::Volume { min, max, power } => {
                    // Apply uniform heat to all cells within volume
                    let nx = self.temperature.nx;
                    let ny = self.temperature.ny;
                    let nz = self.temperature.nz;
                    for iz in 0..nz {
                        for iy in 0..ny {
                            for ix in 0..nx {
                                let wx = self.temperature.min.0
                                    + ix as f32 * (self.temperature.max.0 - self.temperature.min.0)
                                        / (nx - 1).max(1) as f32;
                                let wy = self.temperature.min.1
                                    + iy as f32 * (self.temperature.max.1 - self.temperature.min.1)
                                        / (ny - 1).max(1) as f32;
                                let wz = self.temperature.min.2
                                    + iz as f32 * (self.temperature.max.2 - self.temperature.min.2)
                                        / (nz - 1).max(1) as f32;

                                if wx >= min.0
                                    && wx <= max.0
                                    && wy >= min.1
                                    && wy <= max.1
                                    && wz >= min.2
                                    && wz <= max.2
                                {
                                    self.temperature.add(ix, iy, iz, power * dt);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Accumulate melt deformation where temperature exceeds melt point
    fn accumulate_melt(&mut self, dt: f32) {
        let melt_temp = self.config.melt_temperature;
        let melt_rate = self.config.melt_rate;
        let droop = self.config.droop_strength;

        let n = self.temperature.cell_count();
        for i in 0..n {
            let temp = self.temperature.data[i];
            if temp > melt_temp {
                let excess = temp - melt_temp;
                // Surface recession proportional to excess temperature
                self.melt_accumulator.data[i] += excess * melt_rate * dt;
            }
        }

        // Gravity droop: shift melt downward
        if droop > 0.0 {
            let nx = self.melt_accumulator.nx;
            let ny = self.melt_accumulator.ny;
            let nz = self.melt_accumulator.nz;
            // Simple downward diffusion (bottom cells receive from above)
            for iz in 0..nz {
                for ix in 0..nx {
                    for iy in 1..ny {
                        let above_idx = self.melt_accumulator.index(ix, iy, iz);
                        let below_idx = self.melt_accumulator.index(ix, iy - 1, iz);
                        let transfer = self.melt_accumulator.data[above_idx] * droop * dt;
                        if transfer > 0.0 {
                            self.melt_accumulator.data[below_idx] += transfer;
                            self.melt_accumulator.data[above_idx] -= transfer * 0.5;
                        }
                    }
                }
            }
        }
    }
}

impl PhysicsModifier for ThermalModifier {
    #[inline]
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32 {
        if !self.enabled {
            return original_dist;
        }

        let temp = self.temperature.sample(x, y, z);
        let mut d = original_dist;

        // Melt: accumulated surface recession
        let melt = self.melt_accumulator.sample(x, y, z);
        if melt > 0.0 {
            d += melt; // Positive = material removed
        }

        // Thermal expansion: warm areas expand (distance decreases)
        let temp_delta = temp - self.config.ambient_temperature;
        if temp_delta > 0.0 && self.config.expansion_coefficient > 0.0 {
            d -= temp_delta * self.config.expansion_coefficient;
        }

        // Freeze growth: cold areas grow (distance decreases), capped to prevent unbounded expansion
        if temp < self.config.freeze_temperature && self.config.freeze_rate > 0.0 {
            let cold = self.config.freeze_temperature - temp;
            let freeze_offset = (cold * self.config.freeze_rate).min(1.0);
            d -= freeze_offset;
        }

        d
    }

    fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        // 1. Apply heat sources
        self.apply_heat_sources(dt);

        // 2. Diffuse temperature
        self.temperature.diffuse(dt, self.config.diffusion_rate);

        // 3. Cool toward ambient
        self.temperature.decay_toward(
            self.config.ambient_temperature,
            self.config.cooling_rate,
            dt,
        );

        // 4. Accumulate melt
        self.accumulate_melt(dt);
    }

    fn name(&self) -> &'static str {
        "thermal"
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
    use crate::sdf_collider::{ClosureSdf, SdfField};
    use crate::sim_modifier::SingleModifiedSdf;

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        )
    }

    #[test]
    fn test_thermal_no_heat() {
        let config = ThermalConfig::default();
        let modifier = ThermalModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        let modified = SingleModifiedSdf::new(Box::new(unit_sphere()), modifier);

        // No heat applied: distance should be unchanged
        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 1.0).abs() < 0.1,
            "No heat should not change SDF, got {}",
            d
        );
    }

    #[test]
    fn test_thermal_melt() {
        let config = ThermalConfig {
            melt_temperature: 100.0,
            melt_rate: 0.1,
            ambient_temperature: 20.0,
            ..Default::default()
        };
        let mut modifier = ThermalModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Apply extreme heat at surface (radius must be > cell_size for 8-res grid)
        modifier.apply_heat_at(0.0, 0.0, 0.0, 2000.0, 1.5);

        // Simulate several steps
        for _ in 0..50 {
            modifier.update(0.016);
        }

        // Check melt accumulation
        let melt = modifier.melt_accumulator.sample(0.0, 0.0, 0.0);
        assert!(melt > 0.0, "Melt should accumulate, got {}", melt);
    }

    #[test]
    fn test_thermal_expansion() {
        let config = ThermalConfig {
            expansion_coefficient: 0.01,
            ambient_temperature: 20.0,
            melt_temperature: 10000.0, // Won't melt
            ..Default::default()
        };
        let mut modifier = ThermalModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Heat the entire field
        for v in &mut modifier.temperature.data {
            *v = 100.0; // 80 degrees above ambient
        }

        let d_before = 1.0f32; // distance at surface
        let d_after = modifier.modify_distance(2.0, 0.0, 0.0, d_before);

        // Expansion: distance should decrease (object grows)
        assert!(
            d_after < d_before,
            "Expansion should decrease distance, before={}, after={}",
            d_before,
            d_after
        );
    }

    #[test]
    fn test_heat_diffusion() {
        let config = ThermalConfig {
            diffusion_rate: 1.0,
            ambient_temperature: 0.0,
            cooling_rate: 0.0,
            melt_temperature: 10000.0,
            ..Default::default()
        };
        let mut modifier = ThermalModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Set ambient to 0
        for v in &mut modifier.temperature.data {
            *v = 0.0;
        }

        // Hot spot at center (large radius to cover grid cells)
        modifier.apply_heat_at(0.0, 0.0, 0.0, 500.0, 1.5);
        let center_before = modifier.temperature_at(0.0, 0.0, 0.0);

        // Diffuse
        for _ in 0..20 {
            modifier.update(0.01);
        }

        let center_after = modifier.temperature_at(0.0, 0.0, 0.0);
        assert!(
            center_after < center_before,
            "Heat should spread, before={}, after={}",
            center_before,
            center_after
        );
    }
}
