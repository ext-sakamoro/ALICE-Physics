//! Erosion Simulation Modifier
//!
//! Surface material removal from sustained forces:
//! - **Wind Erosion**: Airflow strips exposed surfaces
//! - **Water Erosion**: Flowing water carves channels
//! - **Chemical Corrosion**: Reactive agents eat material
//! - **Ablation**: High-speed flow strips surface layer
//!
//! # Physics Model
//!
//! Empirical rate law, `depth += rate · (1 − hardness) · vⁿ · exposure · dt`,
//! capped at `max_depth`; accumulated erosion increases the SDF distance
//! (surface recedes). The per-type constants are fixed in this version:
//!
//! | type | `n` | prefactor | note |
//! |---|---|---|---|
//! | `Wind` | 1 | 1 | linear in flow speed |
//! | `Water` | 1 | 1.5 | water carries more momentum per unit speed |
//! | `Chemical` | 0 | 1 | speed-independent attack |
//! | `Ablation` | 2 | 1 | kinetic-energy flux |
//!
//! Solid-particle erosion measurements give `n ≈ 2–3` (Finnie 1960,
// LIMITATION(COV-FRACT-085): so `Wind` / `Water` are game-tuned rather than validated
//! Bitter 1963), so `Wind` / `Water` are game-tuned rather than validated;
//! `exposure` is re-read every frame and decays with a fixed `5 /s` when the
//! caller stops supplying it. Making `n`, the prefactor and the exposure
//! decay configurable adds fields to [`ErosionConfig`] and is scheduled for
//! 2.0 (see `docs/ROADMAP.md`).
//!
//! Author: Moroya Sakamoto

use crate::math::Fix128;
use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;
use crate::sim_modifier::{observe_max, observe_sum, StateReader, StateWriter};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

// ============================================================================
// Configuration
// ============================================================================

/// Erosion type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErosionType {
    /// Wind-driven erosion (velocity-dependent)
    Wind,
    /// Water-driven erosion (velocity + gravity channeling)
    Water,
    /// Chemical corrosion (uniform or spatially varying)
    Chemical,
    /// High-speed ablation (velocity-squared dependent)
    Ablation,
}

/// Erosion modifier configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErosionConfig {
    /// Type of erosion
    pub erosion_type: ErosionType,
    /// Base erosion rate (distance/second per unit velocity)
    pub rate: f32,
    /// Material hardness (0 = soft, 1 = hard). Harder = less erosion.
    pub hardness: f32,
    /// Maximum erosion depth
    pub max_depth: f32,
    /// Smoothing rate for erosion field (prevents sharp edges)
    pub smoothing: f32,
    /// Flow direction (for wind/water erosion)
    pub flow_direction: (f32, f32, f32),
    /// Flow speed
    pub flow_speed: f32,
}

impl Default for ErosionConfig {
    fn default() -> Self {
        Self {
            erosion_type: ErosionType::Wind,
            rate: 0.01,
            hardness: 0.5,
            max_depth: 2.0,
            smoothing: 0.1,
            flow_direction: (1.0, 0.0, 0.0),
            flow_speed: 1.0,
        }
    }
}

// ============================================================================
// Erosion Modifier
// ============================================================================

/// Erosion modifier that removes surface material over time
pub struct ErosionModifier {
    /// Configuration
    pub config: ErosionConfig,
    /// Accumulated erosion depth (positive = material removed)
    pub erosion_depth: ScalarField3D,
    /// Exposure field: how exposed each point is to the erosion agent
    /// (computed from SDF surface proximity and flow alignment)
    pub exposure: ScalarField3D,
    /// Whether modifier is enabled
    pub enabled: bool,
}

impl ErosionModifier {
    /// Create a new erosion modifier
    #[must_use]
    pub fn new(
        config: ErosionConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        Self {
            config,
            erosion_depth: ScalarField3D::new(resolution, resolution, resolution, min, max),
            exposure: ScalarField3D::new(resolution, resolution, resolution, min, max),
            enabled: true,
        }
    }

    /// Set exposure at a point (call from physics contact detection)
    ///
    /// `exposure` = how exposed this point is (0..1)
    pub fn set_exposure_at(&mut self, x: f32, y: f32, z: f32, exposure_value: f32, radius: f32) {
        self.exposure.splat(x, y, z, exposure_value, radius);
    }

    /// Mark surface points as exposed based on flow direction alignment
    ///
    /// Points where the SDF normal faces the flow receive higher exposure.
    pub fn compute_exposure_from_normals(
        &mut self,
        sdf: &dyn crate::sdf_collider::SdfField,
        surface_threshold: f32,
    ) {
        let nx = self.exposure.nx;
        let ny = self.exposure.ny;
        let nz = self.exposure.nz;

        let (fx, fy, fz) = self.config.flow_direction;
        let flow_len = fz.mul_add(fz, fx.mul_add(fx, fy * fy)).sqrt();
        let (fx, fy, fz) = if flow_len > 1e-10 {
            (fx / flow_len, fy / flow_len, fz / flow_len)
        } else {
            (1.0, 0.0, 0.0)
        };

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let wx = self.exposure.min.0
                        + ix as f32 * (self.exposure.max.0 - self.exposure.min.0)
                            / (nx - 1).max(1) as f32;
                    let wy = self.exposure.min.1
                        + iy as f32 * (self.exposure.max.1 - self.exposure.min.1)
                            / (ny - 1).max(1) as f32;
                    let wz = self.exposure.min.2
                        + iz as f32 * (self.exposure.max.2 - self.exposure.min.2)
                            / (nz - 1).max(1) as f32;

                    let dist = sdf.distance(wx, wy, wz).abs();

                    if dist < surface_threshold {
                        let (snx, sny, snz) = sdf.normal(wx, wy, wz);
                        // Dot product: how much the surface faces the flow
                        let alignment = -snz.mul_add(fz, snx.mul_add(fx, sny * fy));
                        let exp = alignment.max(0.0); // Only windward faces
                        let idx = self.exposure.index(ix, iy, iz);
                        self.exposure.data[idx] = exp;
                    }
                }
            }
        }
    }

    /// Get current erosion depth at a point
    #[must_use]
    pub fn erosion_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.erosion_depth.sample(x, y, z)
    }

    /// Compute erosion rate based on type
    fn compute_rate(&self, exposure: f32) -> f32 {
        let speed = self.config.flow_speed;
        let base = self.config.rate * (1.0 - self.config.hardness);

        match self.config.erosion_type {
            ErosionType::Wind => base * speed * exposure,
            ErosionType::Water => base * speed * exposure * WATER_PREFACTOR,
            ErosionType::Chemical => base * exposure, // speed-independent
            ErosionType::Ablation => base * speed * speed * exposure, // v² (kinetic flux)
        }
    }
}

/// Water erosion prefactor relative to wind at the same speed (module doc
/// table).
pub const WATER_PREFACTOR: f32 = 1.5;

/// Exposure decay rate (1/s) applied every `update` when the caller does not
/// re-supply exposure.
pub const EXPOSURE_DECAY_PER_S: f32 = 5.0;

impl PhysicsModifier for ErosionModifier {
    #[inline]
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32 {
        if !self.enabled {
            return original_dist;
        }

        let erosion = self.erosion_depth.sample(x, y, z);
        original_dist + erosion // Positive erosion = surface recedes
    }

    fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        // Accumulate erosion based on exposure
        let max_depth = self.config.max_depth;
        let n = self.exposure.cell_count();
        for i in 0..n {
            let exp = self.exposure.data[i];
            if exp > 0.0 {
                let rate = self.compute_rate(exp);
                self.erosion_depth.data[i] =
                    rate.mul_add(dt, self.erosion_depth.data[i]).min(max_depth);
            }
        }

        // Smooth erosion field (prevents jagged edges)
        if self.config.smoothing > 0.0 {
            self.erosion_depth.diffuse(dt, self.config.smoothing);
        }

        // Decay exposure (needs to be re-applied each frame); the rate is
        // fixed until `ErosionConfig` gains an `exposure_decay` field (2.0)
        self.exposure.decay(EXPOSURE_DECAY_PER_S, dt);
    }

    fn name(&self) -> &'static str {
        "erosion"
    }

    fn is_active(&self) -> bool {
        self.enabled
    }
}

// ============================================================================
// World participant
// ============================================================================

impl ErosionType {
    /// Payload tag: `Wind` 0, `Water` 1, `Chemical` 2, `Ablation` 3.
    const fn state_tag(self) -> u8 {
        match self {
            Self::Wind => 0,
            Self::Water => 1,
            Self::Chemical => 2,
            Self::Ablation => 3,
        }
    }

    const fn from_state_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Wind),
            1 => Some(Self::Water),
            2 => Some(Self::Chemical),
            3 => Some(Self::Ablation),
            _ => None,
        }
    }
}

impl ErosionModifier {
    /// Snapshot tag of this type as a world participant: `"EROS"`, the four
    /// ASCII bytes read big endian. Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"EROS"));

    fn decode_state(bytes: &[u8]) -> Result<Self, StateError> {
        let mut r = StateReader::new(bytes)?;
        let config = ErosionConfig {
            erosion_type: ErosionType::from_state_tag(r.u8()?).ok_or(StateError::InvalidValue)?,
            rate: r.f32()?,
            hardness: r.f32()?,
            max_depth: r.f32()?,
            smoothing: r.f32()?,
            flow_direction: r.vec3()?,
            flow_speed: r.f32()?,
        };
        let enabled = r.bool()?;
        let erosion_depth = r.field()?;
        let exposure = r.field()?;
        r.finish()?;
        Ok(Self {
            config,
            erosion_depth,
            exposure,
            enabled,
        })
    }
}

/// One `update(h)` per world substep (`h` converted with
/// [`Fix128::to_f32`]); see the module documentation of
/// [`crate::sim_modifier`] for the payload and what is not coupled yet.
///
/// Observations: channel 0 the deepest erosion of a cell (absent for a grid
/// without cells), channel 1 the total erosion depth.
impl Participant for ErosionModifier {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        PhysicsModifier::update(self, h.to_f32());
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        observe_max(out, 0, &self.erosion_depth);
        observe_sum(out, 1, &self.erosion_depth);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let mut w = StateWriter::new(out);
        let c = &self.config;
        w.u8(c.erosion_type.state_tag());
        w.f32(c.rate);
        w.f32(c.hardness);
        w.f32(c.max_depth);
        w.f32(c.smoothing);
        w.vec3(c.flow_direction);
        w.f32(c.flow_speed);
        w.bool(self.enabled);
        w.field(&self.erosion_depth);
        w.field(&self.exposure);
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
    use crate::sdf_collider::ClosureSdf;

    #[test]
    fn test_erosion_no_exposure() {
        let config = ErosionConfig::default();
        let modifier = ErosionModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // No exposure: distance unchanged
        let d = modifier.modify_distance(1.0, 0.0, 0.0, 0.5);
        assert!((d - 0.5).abs() < 0.01, "No erosion without exposure");
    }

    #[test]
    fn test_erosion_with_exposure() {
        let config = ErosionConfig {
            rate: 1.0,
            hardness: 0.0, // Very soft
            flow_speed: 1.0,
            ..Default::default()
        };
        let mut modifier = ErosionModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Set exposure at a point
        modifier.set_exposure_at(0.0, 0.0, 0.0, 1.0, 0.5);

        // Simulate erosion
        for _ in 0..30 {
            modifier.update(0.016);
        }

        let erosion = modifier.erosion_at(0.0, 0.0, 0.0);
        assert!(erosion > 0.0, "Exposed area should erode, got {erosion}");
    }

    #[test]
    fn test_erosion_max_depth() {
        let config = ErosionConfig {
            rate: 100.0, // Very fast
            hardness: 0.0,
            max_depth: 0.5,
            flow_speed: 1.0,
            ..Default::default()
        };
        let mut modifier = ErosionModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        modifier.set_exposure_at(0.0, 0.0, 0.0, 1.0, 0.5);

        for _ in 0..100 {
            modifier.update(0.016);
        }

        let erosion = modifier.erosion_at(0.0, 0.0, 0.0);
        assert!(
            erosion <= 0.5 + 0.01,
            "Erosion should be capped at max_depth, got {erosion}"
        );
    }

    #[test]
    fn test_compute_exposure_from_normals() {
        let sphere = ClosureSdf::new(
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        );

        let config = ErosionConfig {
            flow_direction: (-1.0, 0.0, 0.0), // Wind from +X
            ..Default::default()
        };
        let mut modifier = ErosionModifier::new(config, 16, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        modifier.compute_exposure_from_normals(&sphere, 0.3);

        // +X face should be exposed (faces into wind)
        let exp_windward = modifier.exposure.sample(1.0, 0.0, 0.0);
        // -X face should be sheltered
        let exp_leeward = modifier.exposure.sample(-1.0, 0.0, 0.0);

        assert!(
            exp_windward > exp_leeward,
            "Windward face should be more exposed: windward={exp_windward}, leeward={exp_leeward}"
        );
    }

    fn chemical(rate: f32) -> ErosionConfig {
        ErosionConfig {
            erosion_type: ErosionType::Chemical,
            rate,
            hardness: 0.0,
            max_depth: 10.0,
            smoothing: 0.0,
            ..Default::default()
        }
    }

    /// oracle (module doc table): with `base = rate·(1 − hardness)`, wind is
    /// `base·v·e`, water `1.5·base·v·e`, chemical `base·e` (no speed),
    /// ablation `base·v²·e`. `rate 2, hardness 1/2, v 3, e 1/2`: base 1, so
    /// 1.5 / 2.25 / 0.5 / 4.5.
    #[test]
    fn compute_rate_per_erosion_type() {
        for (ty, want) in [
            (ErosionType::Wind, 1.5_f32),
            (ErosionType::Water, 2.25),
            (ErosionType::Chemical, 0.5),
            (ErosionType::Ablation, 4.5),
        ] {
            let m = ErosionModifier::new(
                ErosionConfig {
                    erosion_type: ty,
                    rate: 2.0,
                    hardness: 0.5,
                    flow_speed: 3.0,
                    ..Default::default()
                },
                2,
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            );
            assert!((m.compute_rate(0.5) - want).abs() < 1e-6, "{ty:?}");
        }
    }

    /// oracle: a disabled modifier leaves the distance as it is and its
    /// `update` changes nothing, even with full exposure.
    #[test]
    fn disabled_erosion_is_inert() {
        let mut m = ErosionModifier::new(chemical(1.0), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.erosion_depth.data[0] = 0.25;
        m.exposure.data[0] = 1.0;
        m.enabled = false;
        assert!(!m.is_active());
        assert_eq!(m.modify_distance(0.0, 0.0, 0.0, 0.5), 0.5);
        m.update(1.0);
        assert_eq!(m.erosion_depth.data[0], 0.25);
        assert_eq!(m.exposure.data[0], 1.0);
    }

    /// oracle: a zero flow direction falls back to `+x`: on the ground plane
    /// (`distance = y`, normal `+y`) the windward alignment `−n·f = 0`, so
    /// every near-surface cell gets exposure 0, the same as for flow `+x`.
    #[test]
    fn zero_flow_direction_falls_back_to_plus_x() {
        let ground = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
        let mut m = ErosionModifier::new(
            ErosionConfig {
                flow_direction: (0.0, 0.0, 0.0),
                ..Default::default()
            },
            3,
            (-1.0, -1.0, -1.0),
            (1.0, 1.0, 1.0),
        );
        m.exposure.data.fill(7.0);
        m.compute_exposure_from_normals(&ground, 0.5);
        // the middle layer (y = 0) is on the surface: exposure 0 there
        for iz in 0..3 {
            for ix in 0..3 {
                let i = m.exposure.index(ix, 1, iz);
                assert_eq!(m.exposure.data[i], 0.0, "({ix}, 1, {iz})");
            }
        }
        // the layers at y = ±1 are farther than 0.5: untouched
        assert_eq!(m.exposure.data[m.exposure.index(0, 0, 0)], 7.0);
    }

    fn sample_modifier(ty: ErosionType) -> ErosionModifier {
        let mut m = ErosionModifier::new(
            ErosionConfig {
                erosion_type: ty,
                rate: 0.75,
                hardness: 0.25,
                max_depth: 3.5,
                smoothing: 0.125,
                flow_direction: (0.0, 1.0, -1.0),
                flow_speed: 2.5,
            },
            2,
            (-1.0, -2.0, -3.0),
            (1.0, 2.0, 3.0),
        );
        for (i, v) in m.erosion_depth.data.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
        m.exposure.data[3] = 0.875;
        m.enabled = false;
        m
    }

    /// oracle: the participant kind is `"EROS"` read big endian; a payload
    /// written by `write_state` is accepted and read back into a different
    /// modifier, which then writes the same bytes and has the original
    /// config, flag and fields, for every erosion type.
    #[test]
    fn participant_state_round_trip() {
        assert_eq!(
            ErosionModifier::PARTICIPANT_KIND,
            ParticipantKind::new(0x4552_4f53)
        );
        for ty in [
            ErosionType::Wind,
            ErosionType::Water,
            ErosionType::Chemical,
            ErosionType::Ablation,
        ] {
            let src = sample_modifier(ty);
            assert_eq!(Participant::kind(&src), ErosionModifier::PARTICIPANT_KIND);
            let mut bytes = Vec::new();
            src.write_state(&mut bytes);
            assert_eq!(src.check_state(&bytes), Ok(()));
            let mut dst = ErosionModifier::new(
                ErosionConfig::default(),
                1,
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
            );
            dst.read_state(&bytes);
            assert_eq!(dst.config, src.config);
            assert!(!dst.enabled);
            assert_eq!(dst.erosion_depth.data, src.erosion_depth.data);
            assert_eq!(dst.exposure.data, src.exposure.data);
            assert_eq!(
                (dst.exposure.min, dst.exposure.max),
                ((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))
            );
            let mut again = Vec::new();
            dst.write_state(&mut again);
            assert_eq!(again, bytes);
        }
    }

    /// oracle: the erosion type is the byte after the 4-byte version; tag 4
    /// names no type, a missing last byte or an extra byte is a length error.
    #[test]
    fn participant_state_refusals() {
        let src = sample_modifier(ErosionType::Wind);
        let mut bytes = Vec::new();
        src.write_state(&mut bytes);
        let mut bad_tag = bytes.clone();
        bad_tag[4] = 4;
        assert_eq!(src.check_state(&bad_tag), Err(StateError::InvalidValue));
        assert!(matches!(
            src.check_state(&bytes[..bytes.len() - 1]),
            Err(StateError::Length { .. })
        ));
        let mut long = bytes.clone();
        long.push(0);
        assert!(matches!(
            src.check_state(&long),
            Err(StateError::Length { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "read_state called with a payload check_state refuses")]
    fn participant_read_state_panics_on_a_refused_payload() {
        let mut m = sample_modifier(ErosionType::Wind);
        m.read_state(&[1, 0, 0, 0, 9]);
    }

    /// oracle: in a world of one substep, the participant runs `update(dt)`
    /// once: chemical erosion with rate 1, hardness 0, exposure 1 on one cell
    /// and `dt = 1/4` deepens that cell by `1·1·(1/4) = 1/4`, so channel 0
    /// (deepest cell) and channel 1 (sum of cells) both read 1/4.
    #[test]
    fn participant_substep_in_a_world() {
        let mut m = ErosionModifier::new(chemical(1.0), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.exposure.data[0] = 1.0;
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
        let quarter = Fix128::from_ratio(1, 4);
        assert_eq!(sink.values(), &[(0, quarter), (1, quarter)]);
    }
}
