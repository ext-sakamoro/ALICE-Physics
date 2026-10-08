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
//! Each cell tracks: temperature, phase state, and a latent-heat buffer.
//! Transitions use the enthalpy method (Voller & Cross 1981; Carslaw &
// LIMITATION(COV-THERM-069): with the heat capacity normalised to 1 the latent heats are in kelvin-equivalent units
//! Jaeger ch. XI for the Stefan problem): with the heat capacity normalised
//! to 1 the latent heats are in kelvin-equivalent units, and each `update`
//! moves temperature above a transition point into the buffer (or releases
//! it back on cooling) until the buffer reaches the latent heat. The cell
//! therefore sits **on the plateau** `T = T_m` while it melts and
//! `T + latent` is conserved exactly by the transition step; the melt time
//! is set by how fast diffusion / applied heat feed the cell, not by the
//! step size. Phase transitions modify the SDF: liquid flows downward
//! (mass-conserving transfer), gas expands outward.
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseChangeConfig {
    /// Temperature at which solid melts to liquid
    pub melt_temperature: f32,
    /// Temperature at which liquid vaporizes to gas
    pub boil_temperature: f32,
    /// Latent heat of fusion in kelvin-equivalent units (`L_f / c_p`):
    /// the temperature rise a cell must supply, and hold at
    /// `melt_temperature`, before it turns liquid.
    pub latent_heat_fusion: f32,
    /// Latent heat of vaporization in kelvin-equivalent units (`L_v / c_p`).
    pub latent_heat_vaporization: f32,
    /// Thermal diffusion rate
    pub diffusion_rate: f32,
    /// Ambient temperature
    pub ambient_temperature: f32,
    /// Cooling rate toward ambient
    pub cooling_rate: f32,
    /// Liquid flow speed (gravity-driven downward): fraction of a liquid
    /// cell's offset moved to the cell below per second, and (× 0.1) the
    /// rate at which liquid softens the surface (offset per second).
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
    #[must_use]
    pub fn new(
        config: PhaseChangeConfig,
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
    #[must_use]
    pub fn temperature_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.temperature.sample(x, y, z)
    }

    /// Get phase at a point
    #[must_use]
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

    /// Process phase transitions (enthalpy method).
    ///
    /// The latent buffer holds the latent enthalpy stored in the cell, in
    /// kelvin-equivalent units: `0..L_f` while a solid melts,
    /// `L_f..L_f + L_v` while a liquid boils, `L_f + L_v` for gas. Heat above
    /// a transition temperature is moved from `temperature` into the buffer
    /// (melting / boiling) and back out of it on cooling (freezing /
    /// condensing), so `temperature + latent_heat` is conserved cell by cell
    /// and the cell sits on the plateau until the buffer fills or empties.
    /// Independent of `dt`: the step only redistributes enthalpy already
    /// present in the cell.
    fn process_transitions(&mut self) {
        let melt_t = self.config.melt_temperature;
        let boil_t = self.config.boil_temperature;
        let lh_fusion = self.config.latent_heat_fusion.max(0.0);
        let lh_vaporization = self.config.latent_heat_vaporization.max(0.0);
        let lh_total = lh_fusion + lh_vaporization;

        let n = self.temperature.cell_count();
        for i in 0..n {
            let mut temp = self.temperature.data[i];
            let mut phase_val = self.phase.data[i];
            let mut lh = self.latent_heat.data[i];

            // Solid → Liquid: superheat above T_m fills the fusion buffer
            if phase_val < 0.5 && temp > melt_t {
                let absorbed = (temp - melt_t).min(lh_fusion - lh).max(0.0);
                temp -= absorbed;
                lh += absorbed;
                if lh >= lh_fusion {
                    phase_val = 1.0;
                    lh = lh_fusion;
                }
            }

            // Liquid → Gas: superheat above T_b fills the vaporization buffer
            if (0.5..1.5).contains(&phase_val) && temp > boil_t {
                let absorbed = (temp - boil_t).min(lh_total - lh).max(0.0);
                temp -= absorbed;
                lh += absorbed;
                if lh >= lh_total {
                    phase_val = 2.0;
                    lh = lh_total;
                }
            }

            // Gas → Liquid: subcooling below T_b drains the vaporization buffer
            if phase_val >= 1.5 && temp < boil_t && lh > lh_fusion {
                let released = (boil_t - temp).min(lh - lh_fusion);
                temp += released;
                lh -= released;
                if lh <= lh_fusion {
                    phase_val = 1.0;
                    lh = lh_fusion;
                }
            }

            // Liquid: a partly-boiled cell that cools first gives back the
            // vaporization progress, then subcooling below T_m drains the
            // fusion buffer until the cell freezes
            if (0.5..1.5).contains(&phase_val) {
                if temp < boil_t && lh > lh_fusion {
                    let released = (boil_t - temp).min(lh - lh_fusion);
                    temp += released;
                    lh -= released;
                }
                if temp < melt_t && lh > 0.0 {
                    let released = (melt_t - temp).min(lh);
                    temp += released;
                    lh -= released;
                }
                if temp < melt_t && lh <= 0.0 {
                    phase_val = 0.0;
                    lh = 0.0;
                }
            }

            // Solid: a partly-melted cell that cools gives its progress back
            if phase_val < 0.5 && temp < melt_t && lh > 0.0 {
                let released = (melt_t - temp).min(lh);
                temp += released;
                lh -= released;
            }

            self.temperature.data[i] = temp;
            self.phase.data[i] = phase_val;
            self.latent_heat.data[i] = lh;
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
                self.sdf_offset.data[i] = (gas_expand + gas_dissipate)
                    .mul_add(dt, self.sdf_offset.data[i])
                    .min(max_offset);
            } else if phase_val >= 0.5 {
                // Liquid: slight material softening
                self.sdf_offset.data[i] = (liquid_flow * 0.1)
                    .mul_add(dt, self.sdf_offset.data[i])
                    .min(max_offset);
            }
            // Solid: no offset change
        }

        // Liquid gravity flow: move a fraction `liquid_flow · dt` (≤ 1) of
        // each liquid cell's offset to the cell below. The amount removed
        // from the source equals the amount the destination accepts (its
        // `max_offset` head-room), so the column total is conserved.
        if liquid_flow > 0.0 {
            let fraction = (liquid_flow * dt).min(1.0);
            let nx = self.sdf_offset.nx;
            let ny = self.sdf_offset.ny;
            let nz = self.sdf_offset.nz;
            for iz in 0..nz {
                for ix in 0..nx {
                    for iy in 1..ny {
                        let idx = self.phase.index(ix, iy, iz);
                        let below_idx = self.phase.index(ix, iy - 1, iz);
                        if !(0.5..1.5).contains(&self.phase.data[idx]) {
                            continue;
                        }
                        let source = self.sdf_offset.data[idx];
                        if source <= 0.0 {
                            continue;
                        }
                        let room = (max_offset - self.sdf_offset.data[below_idx]).max(0.0);
                        let transfer = (source * fraction).min(room);
                        if transfer > 0.0 {
                            self.sdf_offset.data[below_idx] += transfer;
                            self.sdf_offset.data[idx] = source - transfer;
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

        // 3. Process phase transitions (enthalpy redistribution, dt-free)
        self.process_transitions();

        // 4. Accumulate SDF offsets
        self.accumulate_offsets(dt);
    }

    fn name(&self) -> &'static str {
        "phase_change"
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
/// The counterpart of the impl on [`crate::thermal::ThermalModifier`]: the two
/// modifiers hold independent copies of the same physical quantity, and
/// [`crate::coupled_field::reconcile_mean`] is what makes a point have one
/// temperature instead of two.
impl CoupledScalar for PhaseChangeModifier {
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

    fn check_adopt(&self, src: &CoupledField) -> Result<(), CoupledFieldError> {
        src.require_same_grid_f32(&self.temperature)
    }
}

// ============================================================================
// World participant
// ============================================================================

impl PhaseChangeModifier {
    /// Snapshot tag of this type as a world participant: `"PHAS"`, the four
    /// ASCII bytes read big endian. Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"PHAS"));

    fn decode_state(bytes: &[u8]) -> Result<Self, StateError> {
        let mut r = StateReader::new(bytes)?;
        let config = PhaseChangeConfig {
            melt_temperature: r.f32()?,
            boil_temperature: r.f32()?,
            latent_heat_fusion: r.f32()?,
            latent_heat_vaporization: r.f32()?,
            diffusion_rate: r.f32()?,
            ambient_temperature: r.f32()?,
            cooling_rate: r.f32()?,
            liquid_flow_speed: r.f32()?,
            gas_expansion_rate: r.f32()?,
            gas_dissipation_rate: r.f32()?,
            max_offset: r.f32()?,
        };
        let enabled = r.bool()?;
        let temperature = r.field()?;
        let phase = r.field()?;
        let latent_heat = r.field()?;
        let sdf_offset = r.field()?;
        r.finish()?;
        Ok(Self {
            config,
            temperature,
            phase,
            latent_heat,
            sdf_offset,
            enabled,
        })
    }
}

/// One `update(h)` per world substep (`h` converted with
/// [`Fix128::to_f32`]); see the module documentation of
/// [`crate::sim_modifier`] for the payload and what is not coupled yet.
///
/// Observations: channel 0 the highest cell temperature (absent for a grid
/// without cells), channel 1 the number of cells that are not solid
/// (stored phase ≥ 0.5), channel 2 the total SDF offset.
impl Participant for PhaseChangeModifier {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        PhysicsModifier::update(self, h.to_f32());
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        observe_max(out, 0, &self.temperature);
        let fluid = self.phase.data.iter().filter(|&&p| p >= 0.5).count();
        out.push(1, Fix128::from_int(fluid as i64));
        observe_sum(out, 2, &self.sdf_offset);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let mut w = StateWriter::new(out);
        let c = &self.config;
        for v in [
            c.melt_temperature,
            c.boil_temperature,
            c.latent_heat_fusion,
            c.latent_heat_vaporization,
            c.diffusion_rate,
            c.ambient_temperature,
            c.cooling_rate,
            c.liquid_flow_speed,
            c.gas_expansion_rate,
            c.gas_dissipation_rate,
            c.max_offset,
        ] {
            w.f32(v);
        }
        w.bool(self.enabled);
        w.field(&self.temperature);
        w.field(&self.phase);
        w.field(&self.latent_heat);
        w.field(&self.sdf_offset);
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

    #[test]
    fn test_phase_change_no_heat() {
        let config = PhaseChangeConfig::default();
        let modifier = PhaseChangeModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // No heat: everything is solid, no offset
        let d = modifier.modify_distance(0.0, 0.0, 0.0, -1.0);
        assert!(
            (d - (-1.0)).abs() < 0.01,
            "No heat should not change SDF, got {d}"
        );
        assert_eq!(modifier.phase_at(0.0, 0.0, 0.0), Phase::Solid);
    }

    #[test]
    fn test_melting_transition() {
        let config = PhaseChangeConfig {
            melt_temperature: 100.0,
            latent_heat_fusion: 10.0,
            ambient_temperature: 20.0,
            cooling_rate: 0.0,   // No cooling
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
            "High heat should melt material, got {phase:?}"
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
        assert_eq!(
            phase,
            Phase::Gas,
            "Extreme heat should vaporize, got {phase:?}"
        );

        // Gas should have increased SDF offset
        let offset = modifier.sdf_offset.sample(0.0, 0.0, 0.0);
        assert!(offset > 0.0, "Gas should add SDF offset, got {offset}");
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
            "Should be liquid/gas after heating, got {phase_hot:?}"
        );

        // Enable fast cooling and cool down
        modifier.config.cooling_rate = 5.0;
        for _ in 0..500 {
            modifier.update(0.016);
        }

        let phase_cold = modifier.phase_at(0.0, 0.0, 0.0);
        assert_eq!(
            phase_cold,
            Phase::Solid,
            "Should solidify after cooling, got {phase_cold:?}"
        );
    }

    fn still_config() -> PhaseChangeConfig {
        PhaseChangeConfig {
            melt_temperature: 200.0,
            boil_temperature: 500.0,
            latent_heat_fusion: 50.0,
            latent_heat_vaporization: 100.0,
            diffusion_rate: 0.0,
            ambient_temperature: 20.0,
            cooling_rate: 0.0,
            liquid_flow_speed: 0.0,
            gas_expansion_rate: 0.5,
            gas_dissipation_rate: 0.25,
            max_offset: 3.0,
        }
    }

    /// oracle: a gas cell holding the full buffer `L_f + L_v = 150` at 400
    /// (100 below boiling) condenses: it releases `min(100, 150 − 50) = 100`,
    /// so it ends liquid at 500 with buffer `L_f = 50`.
    #[test]
    fn gas_condenses_to_liquid_closed_form() {
        let mut m = PhaseChangeModifier::new(still_config(), 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.phase.data[0] = 2.0;
        m.latent_heat.data[0] = 150.0;
        m.temperature.data[0] = 400.0;
        m.process_transitions();
        assert_eq!(m.phase.data[0], 1.0);
        assert_eq!(m.temperature.data[0], 500.0);
        assert_eq!(m.latent_heat.data[0], 50.0);
    }

    /// oracle: a gas cell grows its offset by `(expansion + dissipation)·dt =
    /// 0.75·(1/2) = 3/8` per call and stops at `max_offset`; a solid cell
    /// gets nothing.
    #[test]
    fn gas_offset_grows_and_is_capped() {
        let mut m = PhaseChangeModifier::new(still_config(), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.phase.data[0] = 2.0;
        m.accumulate_offsets(0.5);
        assert_eq!(m.sdf_offset.data[0], 0.375);
        assert_eq!(m.sdf_offset.data[1], 0.0);
        m.accumulate_offsets(100.0);
        assert_eq!(m.sdf_offset.data[0], 3.0);
    }

    /// oracle: with every cell liquid, flow speed 1 and `dt = 1/2`, each cell
    /// first gains `0.1·1·(1/2) = 1/20`; then a fraction `1·(1/2)` of the
    /// upper cell flows down: lower `1/20 + 1/40 = 3/40`, upper `1/40`. In a
    /// column whose lower cell is at `max_offset` there is no room, and an
    /// upper cell with a negative offset has nothing to give.
    #[test]
    fn liquid_flows_down_with_room_and_source() {
        let mut m = PhaseChangeModifier::new(
            PhaseChangeConfig {
                liquid_flow_speed: 1.0,
                ..still_config()
            },
            2,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        );
        m.phase.data.fill(1.0);
        let full_below = m.sdf_offset.index(1, 0, 0);
        let full_above = m.sdf_offset.index(1, 1, 0);
        m.sdf_offset.data[full_below] = 3.0;
        let neg_below = m.sdf_offset.index(0, 0, 1);
        let neg_above = m.sdf_offset.index(0, 1, 1);
        m.sdf_offset.data[neg_above] = -1.0;
        m.accumulate_offsets(0.5);
        let below = m.sdf_offset.index(0, 0, 0);
        let above = m.sdf_offset.index(0, 1, 0);
        assert!((m.sdf_offset.data[below] - 0.075).abs() < 1e-6);
        assert!((m.sdf_offset.data[above] - 0.025).abs() < 1e-6);
        assert_eq!(m.sdf_offset.data[full_below], 3.0);
        assert!((m.sdf_offset.data[full_above] - 0.05).abs() < 1e-6);
        assert!((m.sdf_offset.data[neg_above] + 0.95).abs() < 1e-6);
        assert!((m.sdf_offset.data[neg_below] - 0.05).abs() < 1e-6);
    }

    /// oracle: a disabled modifier leaves the distance and its fields as
    /// they are.
    #[test]
    fn disabled_phase_change_is_inert() {
        let mut m = PhaseChangeModifier::new(still_config(), 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.sdf_offset.data[0] = 0.5;
        m.temperature.data[0] = 900.0;
        m.enabled = false;
        assert!(!m.is_active());
        assert_eq!(m.modify_distance(0.5, 0.5, 0.5, 1.0), 1.0);
        m.update(1.0);
        assert_eq!(m.temperature.data[0], 900.0);
        assert_eq!(m.phase.data[0], 0.0);
    }

    fn sample_modifier() -> PhaseChangeModifier {
        let mut m = PhaseChangeModifier::new(
            PhaseChangeConfig {
                melt_temperature: 150.0,
                boil_temperature: 450.0,
                latent_heat_fusion: 40.0,
                latent_heat_vaporization: 80.0,
                diffusion_rate: 0.125,
                ambient_temperature: 22.5,
                cooling_rate: 0.25,
                liquid_flow_speed: 0.5,
                gas_expansion_rate: 0.375,
                gas_dissipation_rate: 0.625,
                max_offset: 2.5,
            },
            2,
            (-1.0, -2.0, -3.0),
            (1.0, 2.0, 3.0),
        );
        for (i, v) in m.temperature.data.iter_mut().enumerate() {
            *v = 100.0 + i as f32;
        }
        m.phase.data[1] = 1.0;
        m.phase.data[2] = 2.0;
        m.latent_heat.data[1] = 40.0;
        m.sdf_offset.data[2] = 0.75;
        m.enabled = false;
        m
    }

    /// oracle: the participant kind is `"PHAS"` read big endian; a payload
    /// written by `write_state` is accepted and read back into a different
    /// modifier, which then has the original config, flag and the four
    /// fields and writes the same bytes. Observations: channel 0 the hottest
    /// cell (107), channel 1 the liquid or gas cells (2), channel 2 the total
    /// offset (3/4).
    #[test]
    fn participant_state_round_trip_and_observations() {
        assert_eq!(
            PhaseChangeModifier::PARTICIPANT_KIND,
            ParticipantKind::new(0x5048_4153)
        );
        let src = sample_modifier();
        assert_eq!(
            Participant::kind(&src),
            PhaseChangeModifier::PARTICIPANT_KIND
        );
        let mut bytes = Vec::new();
        src.write_state(&mut bytes);
        assert_eq!(src.check_state(&bytes), Ok(()));
        let mut dst = PhaseChangeModifier::new(
            PhaseChangeConfig::default(),
            1,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        );
        dst.read_state(&bytes);
        assert_eq!(dst.config, src.config);
        assert!(!dst.enabled);
        assert_eq!(dst.temperature.data, src.temperature.data);
        assert_eq!(dst.phase.data, src.phase.data);
        assert_eq!(dst.latent_heat.data, src.latent_heat.data);
        assert_eq!(dst.sdf_offset.data, src.sdf_offset.data);
        let mut again = Vec::new();
        dst.write_state(&mut again);
        assert_eq!(again, bytes);

        let mut sink = ObservationSink::new();
        dst.observe(&mut sink);
        assert_eq!(
            sink.values(),
            &[
                (0, Fix128::from_int(107)),
                (1, Fix128::from_int(2)),
                (2, Fix128::from_ratio(3, 4))
            ]
        );
    }

    /// oracle: the enabled flag sits after the version (4) and eleven `f32`s
    /// (44), at offset 48; a value other than 0/1 is refused, as is a
    /// payload one byte short or long.
    #[test]
    fn participant_state_refusals() {
        let src = sample_modifier();
        let mut bytes = Vec::new();
        src.write_state(&mut bytes);
        assert_eq!(bytes[48], 0);
        let mut bad = bytes.clone();
        bad[48] = 5;
        assert_eq!(src.check_state(&bad), Err(StateError::InvalidValue));
        assert!(matches!(
            src.check_state(&bytes[..bytes.len() - 1]),
            Err(StateError::Length { .. })
        ));
        let mut long = bytes;
        long.push(0);
        assert!(matches!(
            src.check_state(&long),
            Err(StateError::Length { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "read_state called with a payload check_state refuses")]
    fn participant_read_state_panics_on_a_refused_payload() {
        let mut m = sample_modifier();
        m.read_state(&[1, 0, 0]);
    }

    /// oracle: in a world of one substep `dt = 1/4`, a one-cell solid at 220
    /// (melt 200, fusion buffer 50, no diffusion or cooling) absorbs
    /// `min(20, 50) = 20` into its buffer and stays solid at 200: channel 0
    /// reads 200, channel 1 (fluid cells) 0, channel 2 (offset) 0.
    #[test]
    fn participant_substep_in_a_world() {
        let mut m = PhaseChangeModifier::new(still_config(), 1, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.temperature.data[0] = 220.0;
        let mut world = crate::solver::PhysicsWorld::new(crate::solver::SolverConfig {
            substeps: 1,
            ..Default::default()
        });
        world.add_participant(Box::new(m)).expect("register");
        world.step(Fix128::from_ratio(1, 4));
        let Some(crate::world_participant::Observed::Exact(sink)) = world.observe_participant(0)
        else {
            panic!("observation");
        };
        assert_eq!(
            sink.values(),
            &[
                (0, Fix128::from_int(200)),
                (1, Fix128::ZERO),
                (2, Fix128::ZERO)
            ]
        );
    }
}
