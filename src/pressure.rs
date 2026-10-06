//! Pressure Simulation Modifier
//!
//! Contact forces and impacts accumulate pressure on the SDF surface:
//! - **Crush**: High pressure compresses the surface inward (permanent dent)
//! - **Bulge**: Internal pressure expands the surface outward
//! - **Dent**: Impact creates localized depression
//!
//! # Physics Model
//!
//! Pressure field accumulates from collision contacts (force × area × time).
//! Above yield threshold, permanent deformation occurs. Below threshold,
//! elastic spring-back (optional).
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

/// Pressure modifier configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PressureConfig {
    /// Pressure diffusion rate (how fast pressure spreads)
    pub diffusion_rate: f32,
    /// Pressure decay rate (elastic recovery)
    pub decay_rate: f32,
    /// Yield threshold (permanent deformation begins above this)
    pub yield_threshold: f32,
    /// Deformation rate (surface recession per unit pressure per second)
    pub deformation_rate: f32,
    /// Maximum deformation depth
    pub max_deformation: f32,
    /// Internal pressure (positive = outward expansion)
    pub internal_pressure: f32,
    /// Internal pressure expansion rate
    pub expansion_rate: f32,
}

impl Default for PressureConfig {
    fn default() -> Self {
        Self {
            diffusion_rate: 0.2,
            decay_rate: 0.5,
            yield_threshold: 10.0,
            deformation_rate: 0.05,
            max_deformation: 1.0,
            internal_pressure: 0.0,
            expansion_rate: 0.01,
        }
    }
}

// ============================================================================
// Pressure Modifier
// ============================================================================

/// Pressure-driven SDF deformation
pub struct PressureModifier {
    /// Configuration
    pub config: PressureConfig,
    /// Current pressure field (transient)
    pub pressure: ScalarField3D,
    /// Accumulated permanent deformation (positive = surface pushed inward)
    pub deformation: ScalarField3D,
    /// Whether modifier is enabled
    pub enabled: bool,
}

impl PressureModifier {
    /// Create a new pressure modifier
    ///
    /// `resolution` is clamped to a minimum of 1: `ScalarField3D` computes
    /// its cell size from `resolution - 1` and clamps grid reads to
    /// `resolution - 1`, both of which underflow `usize` for a resolution
    /// of 0. Clamping here keeps that invariant instead of deferring a
    /// panic to the first call to [`Self::pressure_at`] /
    /// [`Self::deformation_at`].
    #[must_use]
    pub fn new(
        config: PressureConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
        let resolution = resolution.max(1);
        Self {
            config,
            pressure: ScalarField3D::new(resolution, resolution, resolution, min, max),
            deformation: ScalarField3D::new(resolution, resolution, resolution, min, max),
            enabled: true,
        }
    }

    /// Apply pressure at a point (e.g., from collision contact)
    ///
    /// `force` = contact force magnitude, `radius` = contact patch radius
    pub fn apply_pressure_at(&mut self, x: f32, y: f32, z: f32, force: f32, radius: f32) {
        self.pressure.splat(x, y, z, force, radius);
    }

    /// Apply impact at a point (immediate dent)
    ///
    /// `impulse` = impact impulse, `radius` = impact area
    pub fn apply_impact(&mut self, x: f32, y: f32, z: f32, impulse: f32, radius: f32) {
        let dent_depth = impulse * self.config.deformation_rate;
        let clamped = dent_depth.min(self.config.max_deformation);
        self.deformation.splat(x, y, z, clamped, radius);
    }

    /// Get pressure at a point
    #[must_use]
    pub fn pressure_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.pressure.sample(x, y, z)
    }

    /// Get deformation at a point
    #[must_use]
    pub fn deformation_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.deformation.sample(x, y, z)
    }

    /// Accumulate deformation from pressure above yield threshold
    fn yield_deformation(&mut self, dt: f32) {
        let threshold = self.config.yield_threshold;
        let rate = self.config.deformation_rate;
        let max_d = self.config.max_deformation;

        let n = self.pressure.cell_count();
        for i in 0..n {
            let p = self.pressure.data[i];
            if p > threshold {
                let excess = p - threshold;
                let delta = excess * rate * dt;
                self.deformation.data[i] = (self.deformation.data[i] + delta).min(max_d);
            }
        }
    }
}

impl PhysicsModifier for PressureModifier {
    #[inline]
    fn modify_distance(&self, x: f32, y: f32, z: f32, original_dist: f32) -> f32 {
        if !self.enabled {
            return original_dist;
        }

        let mut d = original_dist;

        // Permanent deformation: push surface inward (increase distance)
        let deform = self.deformation.sample(x, y, z);
        if deform > 0.0 {
            d += deform;
        }

        // Internal pressure expansion: push surface outward (decrease distance)
        if self.config.internal_pressure > 0.0 {
            d -= self.config.internal_pressure * self.config.expansion_rate;
        }

        d
    }

    fn update(&mut self, dt: f32) {
        if !self.enabled {
            return;
        }

        // 1. Yield deformation from high pressure
        self.yield_deformation(dt);

        // 2. Diffuse pressure
        self.pressure.diffuse(dt, self.config.diffusion_rate);

        // 3. Decay pressure (elastic recovery for transient pressure)
        self.pressure.decay(self.config.decay_rate, dt);

        // 4. Clamp deformation
        self.deformation.clamp(0.0, self.config.max_deformation);
    }

    fn name(&self) -> &'static str {
        "pressure"
    }

    fn is_active(&self) -> bool {
        self.enabled
    }
}

// ============================================================================
// World participant
// ============================================================================

impl PressureModifier {
    /// Snapshot tag of this type as a world participant: `"PRES"`, the four
    /// ASCII bytes read big endian. Never changes.
    pub const PARTICIPANT_KIND: ParticipantKind =
        ParticipantKind::new(u32::from_be_bytes(*b"PRES"));

    fn decode_state(bytes: &[u8]) -> Result<Self, StateError> {
        let mut r = StateReader::new(bytes)?;
        let config = PressureConfig {
            diffusion_rate: r.f32()?,
            decay_rate: r.f32()?,
            yield_threshold: r.f32()?,
            deformation_rate: r.f32()?,
            max_deformation: r.f32()?,
            internal_pressure: r.f32()?,
            expansion_rate: r.f32()?,
        };
        let enabled = r.bool()?;
        let pressure = r.field()?;
        let deformation = r.field()?;
        r.finish()?;
        Ok(Self {
            config,
            pressure,
            deformation,
            enabled,
        })
    }
}

/// One `update(h)` per world substep (`h` converted with
/// [`Fix128::to_f32`]); see the module documentation of
/// [`crate::sim_modifier`] for the payload and what is not coupled yet.
///
/// Observations: channel 0 the highest cell pressure (absent for a grid
/// without cells), channel 1 the total permanent deformation.
impl Participant for PressureModifier {
    fn kind(&self) -> ParticipantKind {
        Self::PARTICIPANT_KIND
    }

    fn substep(&mut self, _ctx: &mut SubstepCtx<'_>, h: Fix128) -> Result<(), ParticipantFault> {
        PhysicsModifier::update(self, h.to_f32());
        Ok(())
    }

    fn observe(&self, out: &mut ObservationSink) {
        observe_max(out, 0, &self.pressure);
        observe_sum(out, 1, &self.deformation);
    }

    fn write_state(&self, out: &mut Vec<u8>) {
        let mut w = StateWriter::new(out);
        let c = &self.config;
        for v in [
            c.diffusion_rate,
            c.decay_rate,
            c.yield_threshold,
            c.deformation_rate,
            c.max_deformation,
            c.internal_pressure,
            c.expansion_rate,
        ] {
            w.f32(v);
        }
        w.bool(self.enabled);
        w.field(&self.pressure);
        w.field(&self.deformation);
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
    fn test_pressure_no_force() {
        let config = PressureConfig::default();
        let modifier = PressureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        let modified = SingleModifiedSdf::new(Box::new(unit_sphere()), modifier);

        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 1.0).abs() < 0.01,
            "No pressure should not change SDF, got {d}"
        );
    }

    #[test]
    fn test_pressure_dent() {
        let config = PressureConfig {
            deformation_rate: 0.1,
            max_deformation: 2.0,
            ..Default::default()
        };
        let mut modifier = PressureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Impact at surface point
        modifier.apply_impact(1.0, 0.0, 0.0, 5.0, 0.5);

        let deform = modifier.deformation_at(1.0, 0.0, 0.0);
        assert!(
            deform > 0.0,
            "Impact should create deformation, got {deform}"
        );

        // Distance should increase (surface pushed inward)
        let d = modifier.modify_distance(1.0, 0.0, 0.0, 0.0);
        assert!(d > 0.0, "Dent should increase distance, got {d}");
    }

    #[test]
    fn test_pressure_yield() {
        let config = PressureConfig {
            yield_threshold: 5.0,
            deformation_rate: 0.1,
            decay_rate: 0.0, // No recovery
            ..Default::default()
        };
        let mut modifier = PressureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        // Apply high pressure (large radius to cover grid cells)
        modifier.apply_pressure_at(0.0, 0.0, 0.0, 200.0, 1.5);

        // Simulate
        for _ in 0..10 {
            modifier.update(0.016);
        }

        // Deformation should have accumulated
        let deform = modifier.deformation_at(0.0, 0.0, 0.0);
        assert!(
            deform > 0.0,
            "High pressure should cause permanent deformation, got {deform}"
        );
    }

    #[test]
    fn test_pressure_decay() {
        let config = PressureConfig {
            decay_rate: 5.0,          // Fast decay
            yield_threshold: 10000.0, // Won't yield
            ..Default::default()
        };
        let mut modifier = PressureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));

        modifier.apply_pressure_at(0.0, 0.0, 0.0, 100.0, 0.5);
        let before = modifier.pressure_at(0.0, 0.0, 0.0);

        for _ in 0..20 {
            modifier.update(0.016);
        }

        let after = modifier.pressure_at(0.0, 0.0, 0.0);
        assert!(
            after < before * 0.5,
            "Pressure should decay, before={before}, after={after}"
        );
    }
}
