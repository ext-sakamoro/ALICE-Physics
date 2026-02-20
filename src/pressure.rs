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

use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;

// ============================================================================
// Configuration
// ============================================================================

/// Pressure modifier configuration
#[derive(Clone, Copy, Debug)]
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
    pub fn new(
        config: PressureConfig,
        resolution: usize,
        min: (f32, f32, f32),
        max: (f32, f32, f32),
    ) -> Self {
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
    pub fn pressure_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.pressure.sample(x, y, z)
    }

    /// Get deformation at a point
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

    fn name(&self) -> &str {
        "pressure"
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
    fn test_pressure_no_force() {
        let config = PressureConfig::default();
        let modifier = PressureModifier::new(config, 8, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        let modified = SingleModifiedSdf::new(Box::new(unit_sphere()), modifier);

        let d = modified.distance(2.0, 0.0, 0.0);
        assert!(
            (d - 1.0).abs() < 0.01,
            "No pressure should not change SDF, got {}",
            d
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
            "Impact should create deformation, got {}",
            deform
        );

        // Distance should increase (surface pushed inward)
        let d = modifier.modify_distance(1.0, 0.0, 0.0, 0.0);
        assert!(d > 0.0, "Dent should increase distance, got {}", d);
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
            "High pressure should cause permanent deformation, got {}",
            deform
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
            "Pressure should decay, before={}, after={}",
            before,
            after
        );
    }
}
