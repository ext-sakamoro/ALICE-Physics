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

use crate::coupled_field::{CoupledField, CoupledFieldError, CoupledScalar};
use crate::math::Fix128;
use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;
use crate::sim_modifier::{observe_max, observe_sum, StateReader, StateWriter};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Thermal simulation configuration
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeatSource {
    /// Point heat source
    Point {
        /// Position X (world space)
        x: f32,
        /// Position Y (world space)
        y: f32,
        /// Position Z (world space)
        z: f32,
        // LIMITATION(COV-THERM-040): Heat power (degrees/second at center)
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
    #[must_use]
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
    #[must_use]
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
// Coupling channel
// ============================================================================

/// The temperature field is shared through a deterministic `Fix128` channel.
///
/// `ThermalModifier` and [`crate::phase_change::PhaseChangeModifier`] each own
/// a temperature field and nothing relates them — `PhysicsModifier::update`
/// receives only `dt`, so neither can read the other's. Going through
/// [`CoupledField`] gives the pair one agreed field; see
/// [`crate::coupled_field::reconcile_mean`].
///
/// This does not change what `update` does: coupling happens only when the
/// caller reconciles, so an existing chain behaves exactly as before.
impl CoupledScalar for ThermalModifier {
    fn coupled_name(&self) -> &'static str {
        "temperature"
    }

    fn coupled_channel(&self) -> Result<CoupledField, CoupledFieldError> {
        CoupledField::try_matching(&self.temperature)
    }

    fn publish(&self, out: &mut CoupledField) -> Result<(), CoupledFieldError> {
        out.copy_from_f32(&self.temperature)
    }

    fn adopt(&mut self, src: &CoupledField) -> Result<(), CoupledFieldError> {
        src.write_to_f32(&mut self.temperature)
    }
}

// ============================================================================
// World participant
// ============================================================================

impl ThermalModifier {
    /// Snapshot tag of this type as a world participant: `"THRM"`, the four
    /// ASCII bytes read big endian. Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"THRM"));

    fn decode_state(bytes: &[u8]) -> Result<Self, StateError> {
        let mut r = StateReader::new(bytes)?;
        let config = ThermalConfig {
            diffusion_rate: r.f32()?,
            ambient_temperature: r.f32()?,
            cooling_rate: r.f32()?,
            melt_temperature: r.f32()?,
            melt_rate: r.f32()?,
            droop_strength: r.f32()?,
            expansion_coefficient: r.f32()?,
            freeze_temperature: r.f32()?,
            freeze_rate: r.f32()?,
        };
        let enabled = r.bool()?;
        let temperature = r.field()?;
        let melt_accumulator = r.field()?;
        let n = r.count(1 + 5 * 4)?;
        let mut heat_sources = Vec::with_capacity(n);
        for _ in 0..n {
            heat_sources.push(match r.u8()? {
                0 => HeatSource::Point {
                    x: r.f32()?,
                    y: r.f32()?,
                    z: r.f32()?,
                    power: r.f32()?,
                    radius: r.f32()?,
                },
                1 => HeatSource::Volume {
                    min: r.vec3()?,
                    max: r.vec3()?,
                    power: r.f32()?,
                },
                _ => return Err(StateError::InvalidValue),
            });
        }
        r.finish()?;
        Ok(Self {
            config,
            temperature,
            melt_accumulator,
            heat_sources,
            enabled,
        })
    }
}

/// One `update(h)` per world substep (`h` converted with
/// [`Fix128::to_f32`]); see the module documentation of
/// [`crate::sim_modifier`] for the payload and what is not coupled yet.
///
/// Observations: channel 0 the highest cell temperature (absent for a grid
/// without cells), channel 1 the total melt (sum of the melt cells).
impl Participant for ThermalModifier {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        PhysicsModifier::update(self, h.to_f32());
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        observe_max(out, 0, &self.temperature);
        observe_sum(out, 1, &self.melt_accumulator);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let mut w = StateWriter::new(out);
        let c = &self.config;
        for v in [
            c.diffusion_rate,
            c.ambient_temperature,
            c.cooling_rate,
            c.melt_temperature,
            c.melt_rate,
            c.droop_strength,
            c.expansion_coefficient,
            c.freeze_temperature,
            c.freeze_rate,
        ] {
            w.f32(v);
        }
        w.bool(self.enabled);
        w.field(&self.temperature);
        w.field(&self.melt_accumulator);
        w.usize(self.heat_sources.len());
        for s in &self.heat_sources {
            match *s {
                HeatSource::Point {
                    x,
                    y,
                    z,
                    power,
                    radius,
                } => {
                    w.u8(0);
                    for v in [x, y, z, power, radius] {
                        w.f32(v);
                    }
                }
                HeatSource::Volume { min, max, power } => {
                    w.u8(1);
                    w.vec3(min);
                    w.vec3(max);
                    w.f32(power);
                }
            }
        }
    }

    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        Self::decode_state(bytes).map(drop)
    }

    fn read_state(&mut self, bytes: &[u8]) {
        match Self::decode_state(bytes) {
            Ok(state) => *self = state,
            Err(e) => panic!("read_state called with a payload check_state refuses: {e:?}"),
        }
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
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
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
            "No heat should not change SDF, got {d}"
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
        assert!(melt > 0.0, "Melt should accumulate, got {melt}");
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
        modifier.temperature.data.fill(100.0); // 80 degrees above ambient

        let d_before = 1.0f32; // distance at surface
        let d_after = modifier.modify_distance(2.0, 0.0, 0.0, d_before);

        // Expansion: distance should decrease (object grows)
        assert!(
            d_after < d_before,
            "Expansion should decrease distance, before={d_before}, after={d_after}"
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
        modifier.temperature.data.fill(0.0);

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
            "Heat should spread, before={center_before}, after={center_after}"
        );
    }
}
