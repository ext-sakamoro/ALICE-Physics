//! Fracture Simulation Modifier
//!
//! Stress accumulation from impacts and sustained loads drives crack
//! propagation through the SDF. When accumulated stress exceeds the
//! material's fracture toughness, cracks form along stress concentration
//! lines and are subtracted from the SDF via CSG.
//!
//! # Model
//!
//! 1. Impacts and contacts accumulate stress in a 3D field
//! 2. Stress above threshold triggers fracture seed points
//! 3. Cracks propagate outward from seeds (thin box CSG subtraction)
//! 4. Voronoi-like pattern from multiple seeds creates realistic fragments
//!
//! Author: Moroya Sakamoto

use crate::math::Fix128;
use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;
use crate::sim_modifier::{observe_max, StateReader, StateWriter};
use crate::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Fracture modifier configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractureConfig {
    /// Stress threshold for crack initiation
    pub fracture_toughness: f32,
    /// Crack width (SDF subtraction thickness)
    pub crack_width: f32,
    // LIMITATION(COV-FRACT-060): a finished crack stays carved into the SDF and keeps occupying its slot
    /// Maximum number of cracks (growing and finished: a finished crack stays
    /// carved into the SDF and keeps occupying its slot)
    pub max_cracks: usize,
    /// Stress diffusion rate
    pub stress_diffusion: f32,
    /// Stress decay rate
    pub stress_decay: f32,
    /// Crack propagation speed
    pub propagation_speed: f32,
    /// Maximum crack length
    pub max_crack_length: f32,
}

impl Default for FractureConfig {
    fn default() -> Self {
        Self {
            fracture_toughness: 50.0,
            crack_width: 0.02,
            max_cracks: 32,
            stress_diffusion: 0.3,
            stress_decay: 0.1,
            propagation_speed: 5.0,
            max_crack_length: 3.0,
        }
    }
}

// ============================================================================
// Crack Representation
// ============================================================================

/// A single crack segment (line from start to end with width)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crack {
    /// Crack start point
    pub start: (f32, f32, f32),
    /// Crack end point (current tip)
    pub end: (f32, f32, f32),
    /// Crack direction (normalized)
    pub direction: (f32, f32, f32),
    /// Current crack length
    pub length: f32,
    /// Whether crack is still propagating
    pub active: bool,
}

impl Crack {
    /// SDF of a crack: capsule with given width
    ///
    /// Returns signed distance to the crack volume
    fn distance(&self, x: f32, y: f32, z: f32, width: f32) -> f32 {
        // Capsule SDF between start and end
        let (ax, ay, az) = self.start;
        let (bx, by, bz) = self.end;

        let pax = x - ax;
        let pay = y - ay;
        let paz = z - az;
        let b_ax = bx - ax;
        let b_ay = by - ay;
        let b_az = bz - az;

        let ba_len_sq = b_az.mul_add(b_az, b_ax.mul_add(b_ax, b_ay * b_ay));
        let h = if ba_len_sq > 1e-10 {
            (paz.mul_add(b_az, pax.mul_add(b_ax, pay * b_ay)) / ba_len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let dx = b_ax.mul_add(-h, pax);
        let dy = b_ay.mul_add(-h, pay);
        let dz = b_az.mul_add(-h, paz);

        (dx * dx + dy * dy + dz * dz).sqrt() - width
    }
}

// ============================================================================
// Fracture Modifier
// ============================================================================

/// Fracture modifier: stress accumulation and crack propagation
pub struct FractureModifier {
    /// Configuration
    pub config: FractureConfig,
    /// Accumulated stress field
    pub stress: ScalarField3D,
    /// Active cracks
    pub cracks: Vec<Crack>,
    /// Whether modifier is enabled
    pub enabled: bool,
}

impl FractureModifier {
    /// Create a new fracture modifier
    #[must_use]
    pub fn new(
        config: FractureConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        Self {
            config,
            stress: ScalarField3D::new(resolution, resolution, resolution, min, max),
            cracks: Vec::new(),
            enabled: true,
        }
    }

    /// Apply stress at a point (from impact or load)
    pub fn apply_stress_at(&mut self, x: f32, y: f32, z: f32, amount: f32, radius: f32) {
        self.stress.splat(x, y, z, amount, radius);
    }

    /// Get stress at a point
    #[must_use]
    pub fn stress_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.stress.sample(x, y, z)
    }

    /// Number of active cracks
    #[must_use]
    pub fn active_crack_count(&self) -> usize {
        self.cracks.iter().filter(|c| c.active).count()
    }

    /// Check for new fracture seeds where stress exceeds threshold
    fn check_fracture_seeds(&mut self) {
        if self.cracks.len() >= self.config.max_cracks {
            return;
        }

        let threshold = self.config.fracture_toughness;
        let nx = self.stress.nx;
        let ny = self.stress.ny;
        let nz = self.stress.nz;

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    if self.cracks.len() >= self.config.max_cracks {
                        return;
                    }

                    let s = self.stress.get(ix, iy, iz);
                    if s > threshold {
                        let wx = self.stress.min.0
                            + ix as f32 * (self.stress.max.0 - self.stress.min.0)
                                / (nx - 1).max(1) as f32;
                        let wy = self.stress.min.1
                            + iy as f32 * (self.stress.max.1 - self.stress.min.1)
                                / (ny - 1).max(1) as f32;
                        let wz = self.stress.min.2
                            + iz as f32 * (self.stress.max.2 - self.stress.min.2)
                                / (nz - 1).max(1) as f32;

                        // Don't seed near existing cracks
                        let too_close = self.cracks.iter().any(|c| {
                            let dx = c.start.0 - wx;
                            let dy = c.start.1 - wy;
                            let dz = c.start.2 - wz;
                            (dx * dx + dy * dy + dz * dz).sqrt() < self.config.crack_width * 10.0
                        });

                        if !too_close {
                            // Crack direction: along stress gradient (perpendicular to max stress)
                            let (gx, gy, gz) = self.stress.gradient(wx, wy, wz);
                            let glen = gz.mul_add(gz, gx.mul_add(gx, gy * gy)).sqrt();

                            // Perpendicular to gradient (crack runs along stress contour)
                            let (dx, dy, dz) = if glen > 1e-5 {
                                // LIMITATION(COV-FRACT-057): Cross with up vector for horizontal crack tendency
                                // Cross with up vector for horizontal crack tendency
                                let cx = gy.mul_add(0.0, -(gz * 1.0));
                                let cy = gz.mul_add(0.0, -(gx * 0.0));
                                let cz = gx.mul_add(1.0, -(gy * 0.0));
                                let clen = (cx * cx + cy * cy + cz * cz).sqrt();
                                if clen > 1e-5 {
                                    (cx / clen, cy / clen, cz / clen)
                                } else {
                                    (1.0, 0.0, 0.0)
                                }
                            } else {
                                // Random-ish direction based on position
                                let hash =
                                    ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) as f32;
                                let angle = hash * 0.0001;
                                (
                                    crate::det_math::cos(angle),
                                    0.0,
                                    crate::det_math::sin(angle),
                                )
                            };

                            self.cracks.push(Crack {
                                start: (wx, wy, wz),
                                end: (wx, wy, wz),
                                direction: (dx, dy, dz),
                                length: 0.0,
                                active: true,
                            });

                            // Consume stress at seed point
                            self.stress.set(ix, iy, iz, 0.0);
                        }
                    }
                }
            }
        }
    }

    /// Propagate active cracks
    fn propagate_cracks(&mut self, dt: f32) {
        let speed = self.config.propagation_speed;
        let max_len = self.config.max_crack_length;

        for crack in &mut self.cracks {
            if !crack.active {
                continue;
            }

            let growth = speed * dt;
            crack.length += growth;

            if crack.length >= max_len {
                crack.length = max_len;
                crack.active = false;
            }

            // Extend tip
            crack.end = (
                crack.direction.0.mul_add(crack.length, crack.start.0),
                crack.direction.1.mul_add(crack.length, crack.start.1),
                crack.direction.2.mul_add(crack.length, crack.start.2),
            );
        }
    }
}

impl PhysicsModifier for FractureModifier {
    #[inline]
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32 {
        if !self.enabled || self.cracks.is_empty() {
            return original_dist;
        }

        let width = self.config.crack_width;
        let mut d = original_dist;

        for crack in &self.cracks {
            if crack.length < 1e-5 {
                continue;
            }
            let crack_dist = crack.distance(x, y, z, width);
            // CSG subtraction: max(original, -crack)
            // Only subtract where inside the original shape
            if d < width * 2.0 {
                d = d.max(-crack_dist);
            }
        }

        d
    }

    fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        // 1. Diffuse stress
        self.stress.diffuse(dt, self.config.stress_diffusion);

        // 2. Decay stress
        self.stress.decay(self.config.stress_decay, dt);

        // 3. Check for new fracture seeds
        self.check_fracture_seeds();

        // 4. Propagate existing cracks
        self.propagate_cracks(dt);
    }

    fn name(&self) -> &'static str {
        "fracture"
    }

    fn is_active(&self) -> bool {
        self.enabled
    }
}

// ============================================================================
// World participant
// ============================================================================

impl FractureModifier {
    /// Snapshot tag of this type as a world participant: `"FRAC"`, the four
    /// ASCII bytes read big endian. Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"FRAC"));

    fn decode_state(bytes: &[u8]) -> Result<Self, StateError> {
        let mut r = StateReader::new(bytes)?;
        let config = FractureConfig {
            fracture_toughness: r.f32()?,
            crack_width: r.f32()?,
            max_cracks: r.usize()?,
            stress_diffusion: r.f32()?,
            stress_decay: r.f32()?,
            propagation_speed: r.f32()?,
            max_crack_length: r.f32()?,
        };
        let enabled = r.bool()?;
        let stress = r.field()?;
        let n = r.count(10 * 4 + 1)?;
        let mut cracks = Vec::with_capacity(n);
        for _ in 0..n {
            cracks.push(Crack {
                start: r.vec3()?,
                end: r.vec3()?,
                direction: r.vec3()?,
                length: r.f32()?,
                active: r.bool()?,
            });
        }
        r.finish()?;
        Ok(Self {
            config,
            stress,
            cracks,
            enabled,
        })
    }
}

/// One `update(h)` per world substep (`h` converted with
/// [`Fix128::to_f32`]); see the module documentation of
/// [`crate::sim_modifier`] for the payload and what is not coupled yet.
///
/// Observations: channel 0 the highest cell stress (absent for a grid
/// without cells), channel 1 the number of cracks, channel 2 the number of
/// cracks still growing.
impl Participant for FractureModifier {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        PhysicsModifier::update(self, h.to_f32());
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        observe_max(out, 0, &self.stress);
        out.push(1, Fix128::from_int(self.cracks.len() as i64));
        let growing = self.cracks.iter().filter(|c| c.active).count();
        out.push(2, Fix128::from_int(growing as i64));
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let mut w = StateWriter::new(out);
        let c = &self.config;
        w.f32(c.fracture_toughness);
        w.f32(c.crack_width);
        w.usize(c.max_cracks);
        w.f32(c.stress_diffusion);
        w.f32(c.stress_decay);
        w.f32(c.propagation_speed);
        w.f32(c.max_crack_length);
        w.bool(self.enabled);
        w.field(&self.stress);
        w.usize(self.cracks.len());
        for k in &self.cracks {
            w.vec3(k.start);
            w.vec3(k.end);
            w.vec3(k.direction);
            w.f32(k.length);
            w.bool(k.active);
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

    #[test]
    fn test_fracture_no_stress() {
        let config = FractureConfig::default();
        let modifier = FractureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        let d = modifier.modify_distance(0.0, 0.0, 0.0, -0.5);
        assert!((d - (-0.5)).abs() < 0.01, "No stress = no cracks");
        assert_eq!(modifier.active_crack_count(), 0);
    }

    #[test]
    fn test_fracture_crack_creation() {
        let config = FractureConfig {
            fracture_toughness: 10.0,
            ..Default::default()
        };
        let mut modifier = FractureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Apply massive stress (radius must cover grid cells)
        modifier.apply_stress_at(0.0, 0.0, 0.0, 500.0, 1.5);

        // Step to trigger crack
        modifier.update(0.016);

        assert!(
            !modifier.cracks.is_empty(),
            "High stress should create cracks"
        );
    }

    #[test]
    fn test_crack_propagation() {
        let config = FractureConfig {
            fracture_toughness: 5.0,
            propagation_speed: 10.0,
            max_crack_length: 1.0,
            ..Default::default()
        };
        let mut modifier = FractureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        modifier.apply_stress_at(0.0, 0.0, 0.0, 500.0, 1.5);
        modifier.update(0.016); // Create crack

        let initial_len = modifier.cracks.first().map_or(0.0, |c| c.length);

        // Propagate
        for _ in 0..10 {
            modifier.update(0.016);
        }

        let final_len = modifier.cracks.first().map_or(0.0, |c| c.length);
        assert!(
            final_len > initial_len,
            "Crack should grow: initial={initial_len}, final={final_len}"
        );
    }

    #[test]
    fn test_crack_sdf_subtraction() {
        let config = FractureConfig {
            crack_width: 0.1,
            ..Default::default()
        };
        let mut modifier = FractureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Manually add a crack
        modifier.cracks.push(Crack {
            start: (-1.0, 0.0, 0.0),
            end: (1.0, 0.0, 0.0),
            direction: (1.0, 0.0, 0.0),
            length: 2.0,
            active: false,
        });

        // Point near the crack should be affected
        let d = modifier.modify_distance(0.0, 0.0, 0.0, -0.05);
        // Original was inside (-0.05), crack distance at (0,0,0) should be near -0.1
        // max(-0.05, -(-0.1)) = max(-0.05, 0.1) = 0.1
        assert!(d > -0.05, "Crack should cut into the SDF, got {d}");
    }

    fn seeding_config(max_cracks: usize) -> FractureConfig {
        FractureConfig {
            fracture_toughness: 50.0,
            crack_width: 0.01,
            max_cracks,
            stress_diffusion: 0.0,
            stress_decay: 0.0,
            propagation_speed: 4.0,
            max_crack_length: 10.0,
        }
    }

    /// oracle: on a one-cell grid at `min = (1, 2, 3)` the stress 100 is above
    /// the toughness 50, so one update seeds a crack at the cell, consumes the
    /// stress (0) and, with a flat stress (no gradient) at cell `(0, 0, 0)`,
    /// runs it along `(cos 0, 0, sin 0) = (1, 0, 0)`; propagation `4·(1/4)`
    /// gives length 1 and tip `(2, 2, 3)`.
    #[test]
    fn one_update_seeds_and_grows_a_crack() {
        let mut m = FractureModifier::new(seeding_config(4), 1, (1.0, 2.0, 3.0), (2.0, 3.0, 4.0));
        m.stress.data[0] = 100.0;
        m.update(0.25);
        assert_eq!(m.cracks.len(), 1);
        let c = m.cracks[0];
        assert_eq!(c.start, (1.0, 2.0, 3.0));
        assert_eq!(c.direction, (1.0, 0.0, 0.0));
        assert_eq!(c.length, 1.0);
        assert_eq!(c.end, (2.0, 2.0, 3.0));
        assert!(c.active);
        assert_eq!(m.stress.data[0], 0.0);
    }

    /// oracle: with every one of the 8 cells over the toughness and the cells
    /// 1 apart (well beyond `10 · width`), `max_cracks` caps the seeds: 1
    /// crack for a cap of 1, none for a cap of 0 (and the stress is then
    /// left untouched).
    #[test]
    fn max_cracks_caps_the_seeds() {
        let mut m = FractureModifier::new(seeding_config(1), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        m.stress.data.fill(100.0);
        m.update(0.25);
        assert_eq!(m.cracks.len(), 1);

        let mut none =
            FractureModifier::new(seeding_config(0), 2, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        none.stress.data.fill(100.0);
        none.update(0.25);
        assert!(none.cracks.is_empty());
        assert!(none.stress.data.iter().all(|&s| s == 100.0));
    }

    /// oracle: a crack growing `4·1 = 4` past the cap 3 stops at length 3,
    /// becomes inactive and its tip sits 3 along its direction; an inactive
    /// crack does not grow. A disabled modifier leaves the distance and the
    /// cracks as they are.
    #[test]
    fn crack_stops_at_the_cap_and_disabled_is_inert() {
        let mut m = FractureModifier::new(
            FractureConfig {
                max_crack_length: 3.0,
                ..seeding_config(4)
            },
            1,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        );
        m.cracks.push(Crack {
            start: (0.0, 0.0, 0.0),
            end: (0.0, 0.0, 0.0),
            direction: (0.0, 1.0, 0.0),
            length: 0.0,
            active: true,
        });
        m.update(1.0);
        assert_eq!(m.cracks[0].length, 3.0);
        assert!(!m.cracks[0].active);
        assert_eq!(m.cracks[0].end, (0.0, 3.0, 0.0));
        m.update(1.0);
        assert_eq!(m.cracks[0].length, 3.0);

        m.enabled = false;
        assert!(!m.is_active());
        assert_eq!(m.modify_distance(0.0, 1.0, 0.0, -0.5), -0.5);
        m.stress.data[0] = 100.0;
        m.update(1.0);
        assert_eq!(m.cracks.len(), 1);
        assert_eq!(m.stress.data[0], 100.0);
    }

    fn sample_modifier() -> FractureModifier {
        let mut m = FractureModifier::new(
            FractureConfig {
                fracture_toughness: 12.5,
                crack_width: 0.25,
                max_cracks: 7,
                stress_diffusion: 0.125,
                stress_decay: 0.375,
                propagation_speed: 1.5,
                max_crack_length: 2.5,
            },
            2,
            (-1.0, -2.0, -3.0),
            (1.0, 2.0, 3.0),
        );
        for (i, v) in m.stress.data.iter_mut().enumerate() {
            *v = i as f32 * 0.5;
        }
        m.cracks.push(Crack {
            start: (0.5, 0.25, -0.5),
            end: (1.0, 0.25, -0.5),
            direction: (1.0, 0.0, 0.0),
            length: 0.5,
            active: true,
        });
        m.cracks.push(Crack {
            start: (0.0, 0.0, 0.0),
            end: (0.0, 0.0, 2.5),
            direction: (0.0, 0.0, 1.0),
            length: 2.5,
            active: false,
        });
        m.enabled = false;
        m
    }

    /// oracle: the participant kind is `"FRAC"` read big endian; a payload
    /// written by `write_state` is accepted and read back into a different
    /// modifier, which then has the original config, flag, stress and cracks
    /// and writes the same bytes. Observations: channel 0 the largest stress
    /// (cell 7: 3.5), channel 1 the cracks (2), channel 2 the growing ones (1).
    #[test]
    fn participant_state_round_trip_and_observations() {
        assert_eq!(
            FractureModifier::PARTICIPANT_KIND,
            ParticipantKind::new(0x4652_4143)
        );
        let src = sample_modifier();
        assert_eq!(Participant::kind(&src), FractureModifier::PARTICIPANT_KIND);
        let mut bytes = Vec::new();
        src.write_state(&mut bytes);
        assert_eq!(src.check_state(&bytes), Ok(()));
        let mut dst = FractureModifier::new(
            FractureConfig::default(),
            1,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        );
        dst.read_state(&bytes);
        assert_eq!(dst.config, src.config);
        assert!(!dst.enabled);
        assert_eq!(dst.stress.data, src.stress.data);
        assert_eq!(dst.cracks, src.cracks);
        let mut again = Vec::new();
        dst.write_state(&mut again);
        assert_eq!(again, bytes);

        let mut sink = ObservationSink::new();
        dst.observe(&mut sink);
        assert_eq!(
            sink.values(),
            &[
                (0, Fix128::from_ratio(7, 2)),
                (1, Fix128::from_int(2)),
                (2, Fix128::ONE)
            ]
        );
    }

    /// oracle: a crack's `active` flag is the last byte of the payload; a
    /// value other than 0/1 is refused, as are a payload one byte short or
    /// long, and a crack count no payload of this length can hold.
    #[test]
    fn participant_state_refusals() {
        let src = sample_modifier();
        let mut bytes = Vec::new();
        src.write_state(&mut bytes);
        let last = bytes.len() - 1;
        assert_eq!(bytes[last], 0);
        let mut bad = bytes.clone();
        bad[last] = 3;
        assert_eq!(src.check_state(&bad), Err(StateError::InvalidValue));
        assert!(matches!(
            src.check_state(&bytes[..last]),
            Err(StateError::Length { .. })
        ));
        let mut long = bytes.clone();
        long.push(0);
        assert!(matches!(
            src.check_state(&long),
            Err(StateError::Length { .. })
        ));
        // the crack count sits right before the two cracks of 41 bytes each
        let at = bytes.len() - 2 * 41 - 8;
        assert_eq!(bytes[at..at + 8], 2_u64.to_le_bytes());
        let mut many = bytes;
        many[at..at + 8].copy_from_slice(&1000_u64.to_le_bytes());
        assert!(matches!(
            src.check_state(&many),
            Err(StateError::Length { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "read_state called with a payload check_state refuses")]
    fn participant_read_state_panics_on_a_refused_payload() {
        let mut m = sample_modifier();
        m.read_state(&[1, 0, 0, 0]);
    }

    /// oracle: in a world of one substep `dt = 1/4`, the participant runs the
    /// same seeding update: one crack, stress consumed (channel 0 reads 0),
    /// one crack still growing.
    #[test]
    fn participant_substep_in_a_world() {
        let mut m = FractureModifier::new(seeding_config(4), 1, (1.0, 2.0, 3.0), (2.0, 3.0, 4.0));
        m.stress.data[0] = 100.0;
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
            &[(0, Fix128::ZERO), (1, Fix128::ONE), (2, Fix128::ONE)]
        );
    }
}
