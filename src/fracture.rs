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

use crate::sim_field::ScalarField3D;
use crate::sim_modifier::PhysicsModifier;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Fracture modifier configuration
#[derive(Clone, Copy, Debug)]
pub struct FractureConfig {
    /// Stress threshold for crack initiation
    pub fracture_toughness: f32,
    /// Crack width (SDF subtraction thickness)
    pub crack_width: f32,
    /// Maximum number of active cracks
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
#[derive(Clone, Copy, Debug)]
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

        let ba_len_sq = b_ax * b_ax + b_ay * b_ay + b_az * b_az;
        let h = if ba_len_sq > 1e-10 {
            ((pax * b_ax + pay * b_ay + paz * b_az) / ba_len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let dx = pax - b_ax * h;
        let dy = pay - b_ay * h;
        let dz = paz - b_az * h;

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
    pub fn stress_at(&self, x: f32, y: f32, z: f32) -> f32 {
        self.stress.sample(x, y, z)
    }

    /// Number of active cracks
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
                            let glen = (gx * gx + gy * gy + gz * gz).sqrt();

                            // Perpendicular to gradient (crack runs along stress contour)
                            let (dx, dy, dz) = if glen > 1e-5 {
                                // Cross with up vector for horizontal crack tendency
                                let cx = gy * 0.0 - gz * 1.0;
                                let cy = gz * 0.0 - gx * 0.0;
                                let cz = gx * 1.0 - gy * 0.0;
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
                                (angle.cos(), 0.0, angle.sin())
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
                crack.start.0 + crack.direction.0 * crack.length,
                crack.start.1 + crack.direction.1 * crack.length,
                crack.start.2 + crack.direction.2 * crack.length,
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
            modifier.cracks.len() > 0,
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

        let initial_len = modifier.cracks.get(0).map(|c| c.length).unwrap_or(0.0);

        // Propagate
        for _ in 0..10 {
            modifier.update(0.016);
        }

        let final_len = modifier.cracks.get(0).map(|c| c.length).unwrap_or(0.0);
        assert!(
            final_len > initial_len,
            "Crack should grow: initial={}, final={}",
            initial_len,
            final_len
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
        assert!(d > -0.05, "Crack should cut into the SDF, got {}", d);
    }
}
