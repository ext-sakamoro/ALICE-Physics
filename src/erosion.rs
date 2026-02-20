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
//! Erosion rate = velocity × hardness_factor × contact_area × time
//! Accumulated erosion increases SDF distance (surface recedes).
//!
//! Author: Moroya Sakamoto

use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;

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
#[derive(Clone, Copy, Debug)]
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
        let flow_len = (fx * fx + fy * fy + fz * fz).sqrt();
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
                        let alignment = -(snx * fx + sny * fy + snz * fz);
                        let exp = alignment.max(0.0); // Only windward faces
                        let idx = self.exposure.index(ix, iy, iz);
                        self.exposure.data[idx] = exp;
                    }
                }
            }
        }
    }

    /// Get current erosion depth at a point
    pub fn erosion_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.erosion_depth.sample(x, y, z)
    }

    /// Compute erosion rate based on type
    fn compute_rate(&self, exposure: f32) -> f32 {
        let speed = self.config.flow_speed;
        let base = self.config.rate * (1.0 - self.config.hardness);

        match self.config.erosion_type {
            ErosionType::Wind => base * speed * exposure,
            ErosionType::Water => base * speed * exposure * 1.5, // Water is more effective
            ErosionType::Chemical => base * exposure,            // Speed-independent
            ErosionType::Ablation => base * speed * speed * exposure, // v^2 dependent
        }
    }
}

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
                    (self.erosion_depth.data[i] + rate * dt).min(max_depth);
            }
        }

        // Smooth erosion field (prevents jagged edges)
        if self.config.smoothing > 0.0 {
            self.erosion_depth.diffuse(dt, self.config.smoothing);
        }

        // Decay exposure (needs to be re-applied each frame)
        self.exposure.decay(5.0, dt);
    }

    fn name(&self) -> &str {
        "erosion"
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
        assert!(erosion > 0.0, "Exposed area should erode, got {}", erosion);
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
            "Erosion should be capped at max_depth, got {}",
            erosion
        );
    }

    #[test]
    fn test_compute_exposure_from_normals() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
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
            "Windward face should be more exposed: windward={}, leeward={}",
            exp_windward,
            exp_leeward
        );
    }
}
