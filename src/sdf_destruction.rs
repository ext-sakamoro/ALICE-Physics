//! SDF Boolean Destruction System
//!
//! Real-time destructible geometry via SDF CSG operations.
//! Physics events (impacts, explosions) create subtraction volumes
//! that modify the SDF in real-time.
//!
//! # How It Works
//!
//! 1. Impact event occurs (collision, explosion, projectile)
//! 2. Create a destruction shape (sphere, box, custom SDF)
//! 3. Apply CSG subtraction: `new_sdf = max(original, -destruction_shape)`
//! 4. Physics re-evaluates against modified SDF
//!
//! Works with ALICE-SDF's CSG operations (Subtract, SmoothSubtract).
//!
//! Author: Moroya Sakamoto

use core::fmt;
use crate::math::{Fix128, Vec3Fix, QuatFix};
use crate::sdf_collider::SdfField;
use crate::collider::Contact;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use std::sync::Arc;

// ============================================================================
// Destruction Shape
// ============================================================================

/// Shape used for carving destruction into an SDF
#[derive(Clone, Debug)]
pub struct DestructionShape {
    /// Center of the destruction volume (world space)
    pub center: Vec3Fix,
    /// Orientation of the destruction volume
    pub rotation: QuatFix,
    /// Type of destruction shape
    pub shape: DestructionType,
    /// Smoothing factor for smooth subtraction (0 = sharp, > 0 = smooth blend)
    pub smooth_factor: f32,
}

/// Types of destruction volumes
pub enum DestructionType {
    /// Spherical crater
    Sphere {
        /// Crater radius
        radius: f32,
    },
    /// Box-shaped cut
    Box {
        /// Half-extents
        half_extents: (f32, f32, f32),
    },
    /// Cylindrical bore
    Cylinder {
        /// Cylinder radius
        radius: f32,
        /// Cylinder half-height
        half_height: f32,
    },
    /// Custom SDF shape
    Custom {
        /// Custom SDF evaluation function
        sdf: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>,
    },
}

impl Clone for DestructionType {
    fn clone(&self) -> Self {
        match self {
            Self::Sphere { radius } => Self::Sphere { radius: *radius },
            Self::Box { half_extents } => Self::Box { half_extents: *half_extents },
            Self::Cylinder { radius, half_height } => Self::Cylinder { radius: *radius, half_height: *half_height },
            Self::Custom { sdf } => Self::Custom { sdf: Arc::clone(sdf) },
        }
    }
}

impl fmt::Debug for DestructionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sphere { radius } => f.debug_struct("Sphere").field("radius", radius).finish(),
            Self::Box { half_extents } => f.debug_struct("Box").field("half_extents", half_extents).finish(),
            Self::Cylinder { radius, half_height } => f.debug_struct("Cylinder").field("radius", radius).field("half_height", half_height).finish(),
            Self::Custom { .. } => f.debug_struct("Custom").field("sdf", &"<fn>").finish(),
        }
    }
}

impl DestructionShape {
    /// Create a spherical crater
    pub fn sphere(center: Vec3Fix, radius: f32) -> Self {
        Self {
            center,
            rotation: QuatFix::IDENTITY,
            shape: DestructionType::Sphere { radius },
            smooth_factor: 0.0,
        }
    }

    /// Create a box cut
    pub fn cube(center: Vec3Fix, half_extents: (f32, f32, f32)) -> Self {
        Self {
            center,
            rotation: QuatFix::IDENTITY,
            shape: DestructionType::Box { half_extents },
            smooth_factor: 0.0,
        }
    }

    /// Create a cylindrical bore
    pub fn cylinder(center: Vec3Fix, radius: f32, half_height: f32) -> Self {
        Self {
            center,
            rotation: QuatFix::IDENTITY,
            shape: DestructionType::Cylinder { radius, half_height },
            smooth_factor: 0.0,
        }
    }

    /// Set smooth subtraction factor
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smooth_factor = factor;
        self
    }

    /// Set orientation
    pub fn with_rotation(mut self, rotation: QuatFix) -> Self {
        self.rotation = rotation;
        self
    }

    /// Evaluate the destruction shape SDF at a world-space point
    fn evaluate(&self, wx: f32, wy: f32, wz: f32) -> f32 {
        // Transform to local space
        let dx = wx - self.center.x.to_f32();
        let dy = wy - self.center.y.to_f32();
        let dz = wz - self.center.z.to_f32();

        let local = self.rotation.conjugate().rotate_vec(
            Vec3Fix::from_f32(dx, dy, dz)
        );
        let (lx, ly, lz) = local.to_f32();

        match &self.shape {
            DestructionType::Sphere { radius } => {
                (lx * lx + ly * ly + lz * lz).sqrt() - radius
            }
            DestructionType::Box { half_extents } => {
                let (hx, hy, hz) = *half_extents;
                let qx = lx.abs() - hx;
                let qy = ly.abs() - hy;
                let qz = lz.abs() - hz;
                let outside = (qx.max(0.0) * qx.max(0.0)
                    + qy.max(0.0) * qy.max(0.0)
                    + qz.max(0.0) * qz.max(0.0))
                .sqrt();
                let inside = qx.max(qy).max(qz).min(0.0);
                outside + inside
            }
            DestructionType::Cylinder { radius, half_height } => {
                let d_radial = (lx * lx + lz * lz).sqrt() - radius;
                let d_axial = ly.abs() - half_height;
                let outside = (d_radial.max(0.0) * d_radial.max(0.0)
                    + d_axial.max(0.0) * d_axial.max(0.0))
                .sqrt();
                let inside = d_radial.max(d_axial).min(0.0);
                outside + inside
            }
            DestructionType::Custom { sdf } => {
                sdf(lx, ly, lz)
            }
        }
    }
}

// ============================================================================
// Destructible SDF
// ============================================================================

/// An SDF with accumulated destruction volumes.
///
/// Wraps an original SDF and applies CSG subtraction for each
/// destruction event. The modified SDF is evaluated on-the-fly.
pub struct DestructibleSdf {
    /// Original (undamaged) SDF
    original: Box<dyn SdfField>,
    /// Accumulated destruction shapes
    destructions: Vec<DestructionShape>,
    /// Total destruction count (for statistics)
    total_destructions: usize,
}

impl DestructibleSdf {
    /// Create a new destructible SDF wrapping an original field
    pub fn new(original: Box<dyn SdfField>) -> Self {
        Self {
            original,
            destructions: Vec::new(),
            total_destructions: 0,
        }
    }

    /// Apply a destruction event
    pub fn apply_destruction(&mut self, shape: DestructionShape) {
        self.destructions.push(shape);
        self.total_destructions += 1;
    }

    /// Number of active destruction volumes
    pub fn destruction_count(&self) -> usize {
        self.destructions.len()
    }

    /// Total destructions applied (including merged)
    pub fn total_destruction_count(&self) -> usize {
        self.total_destructions
    }

    /// Clear all destruction (restore original shape)
    pub fn reset(&mut self) {
        self.destructions.clear();
    }

    /// Remove destruction shapes that are fully contained by newer ones
    /// (optimization to keep evaluation fast)
    pub fn optimize(&mut self) {
        // Simple: if we have too many destructions, merge overlapping ones
        if self.destructions.len() <= 32 {
            return;
        }

        // Keep only the most recent 32 destructions
        // A more sophisticated implementation would merge overlapping volumes
        let drain_count = self.destructions.len() - 32;
        self.destructions.drain(..drain_count);
    }
}

impl SdfField for DestructibleSdf {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut dist = self.original.distance(x, y, z);

        for destruction in &self.destructions {
            let d_shape = destruction.evaluate(x, y, z);

            if destruction.smooth_factor > 0.0 {
                // Smooth subtraction: smooth_max(a, -b)
                dist = smooth_subtraction(dist, d_shape, destruction.smooth_factor);
            } else {
                // Sharp subtraction: max(a, -b)
                let neg_d = -d_shape;
                if neg_d > dist {
                    dist = neg_d;
                }
            }
        }

        dist
    }

    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        // Central difference gradient on the destructed SDF
        let eps = 0.001;
        let dx = self.distance(x + eps, y, z) - self.distance(x - eps, y, z);
        let dy = self.distance(x, y + eps, z) - self.distance(x, y - eps, z);
        let dz = self.distance(x, y, z + eps) - self.distance(x, y, z - eps);

        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len < 1e-10 {
            (0.0, 1.0, 0.0)
        } else {
            (dx / len, dy / len, dz / len)
        }
    }
}

/// Smooth subtraction (CSG): removes shape B from shape A with smooth blend
///
/// `smooth_max(a, -b, k)` where k controls smoothing radius
#[inline]
fn smooth_subtraction(d_a: f32, d_b: f32, k: f32) -> f32 {
    let h = (0.5 - 0.5 * (d_a + d_b) / k).clamp(0.0, 1.0);
    // mix(d_a, -d_b, h) + k*h*(1-h)
    d_a * (1.0 - h) + (-d_b) * h + k * h * (1.0 - h)
}

// ============================================================================
// Impact-Based Destruction Helpers
// ============================================================================

/// Create a destruction event from a physics contact.
///
/// Generates a spherical crater at the contact point with size
/// proportional to impact velocity.
#[cfg(feature = "std")]
pub fn destruction_from_impact(
    contact: &Contact,
    impact_velocity: Fix128,
    velocity_to_radius_scale: f32,
    min_radius: f32,
    max_radius: f32,
) -> DestructionShape {
    let speed = impact_velocity.to_f32().abs();
    let radius = (speed * velocity_to_radius_scale).clamp(min_radius, max_radius);

    DestructionShape::sphere(contact.point_b, radius)
}

/// Create an explosion destruction event.
///
/// Generates a spherical destruction centered at the explosion point.
pub fn destruction_from_explosion(
    center: Vec3Fix,
    radius: f32,
    smooth: f32,
) -> DestructionShape {
    DestructionShape::sphere(center, radius).with_smoothing(smooth)
}

/// Create a projectile bore (cylindrical destruction along a ray).
pub fn destruction_from_projectile(
    entry_point: Vec3Fix,
    direction: Vec3Fix,
    bore_radius: f32,
    bore_depth: f32,
) -> DestructionShape {
    // Cylinder centered at entry + direction * depth/2
    let center = entry_point + direction * Fix128::from_f32(bore_depth * 0.5);

    // Compute rotation to align cylinder Y-axis with direction
    let rotation = rotation_from_direction(direction);

    DestructionShape::cylinder(center, bore_radius, bore_depth * 0.5)
        .with_rotation(rotation)
}

/// Compute quaternion rotation that aligns Y-axis with the given direction
fn rotation_from_direction(dir: Vec3Fix) -> QuatFix {
    let up = Vec3Fix::UNIT_Y;
    let d = dir.normalize();

    let dot = up.dot(d);
    if dot > Fix128::from_f32(0.999) {
        return QuatFix::IDENTITY;
    }
    if dot < Fix128::from_f32(-0.999) {
        // 180 degree rotation around X
        return QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    }

    let axis = up.cross(d).normalize();
    let angle_cos = dot;
    // half angle: cos(a/2) = sqrt((1+cos(a))/2)
    let half_cos = ((Fix128::ONE + angle_cos).half()).sqrt();
    let half_sin = ((Fix128::ONE - angle_cos).half()).sqrt();

    QuatFix::new(
        axis.x * half_sin,
        axis.y * half_sin,
        axis.z * half_sin,
        half_cos,
    ).normalize()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 { (0.0, 1.0, 0.0) } else { (x / len, y / len, z / len) }
            },
        )
    }

    #[test]
    fn test_destructible_sdf_no_destruction() {
        let dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        // Should behave identically to original
        let d = dsdf.distance(0.0, 0.0, 0.0);
        assert!(d < 0.0, "Origin should be inside sphere");

        let d_outside = dsdf.distance(2.0, 0.0, 0.0);
        assert!(d_outside > 0.0, "Point at (2,0,0) should be outside");
    }

    #[test]
    fn test_sphere_destruction() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        // Carve a hole at the +X surface
        dsdf.apply_destruction(DestructionShape::sphere(
            Vec3Fix::from_f32(1.0, 0.0, 0.0),
            0.3,
        ));

        // Point inside the carved region should now be "outside"
        let d = dsdf.distance(1.0, 0.0, 0.0);
        assert!(d > -0.1, "Point at carved region should be near surface or outside, got {}", d);

        // Point far from destruction should be unaffected
        let d_far = dsdf.distance(-0.5, 0.0, 0.0);
        assert!(d_far < 0.0, "Point far from destruction should still be inside");
    }

    #[test]
    fn test_smooth_destruction() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        dsdf.apply_destruction(
            DestructionShape::sphere(Vec3Fix::from_f32(1.0, 0.0, 0.0), 0.5)
                .with_smoothing(0.1),
        );

        // The smoothed destruction should produce valid SDF values
        let d = dsdf.distance(1.0, 0.0, 0.0);
        // With smooth subtraction, the value should be reasonable
        assert!(d.is_finite(), "SDF should produce finite values");
    }

    #[test]
    fn test_box_destruction() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        dsdf.apply_destruction(DestructionShape::cube(
            Vec3Fix::from_f32(0.0, 1.0, 0.0),
            (0.3, 0.3, 0.3),
        ));

        assert_eq!(dsdf.destruction_count(), 1);
    }

    #[test]
    fn test_multiple_destructions() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        for i in 0..5 {
            dsdf.apply_destruction(DestructionShape::sphere(
                Vec3Fix::from_f32(0.5, i as f32 * 0.2, 0.0),
                0.15,
            ));
        }

        assert_eq!(dsdf.destruction_count(), 5);
        assert_eq!(dsdf.total_destruction_count(), 5);
    }

    #[test]
    fn test_reset() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));

        dsdf.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 0.5));
        assert_eq!(dsdf.destruction_count(), 1);

        dsdf.reset();
        assert_eq!(dsdf.destruction_count(), 0);

        // Should behave as original again
        let d = dsdf.distance(0.0, 0.0, 0.0);
        assert!(d < 0.0, "After reset, origin should be inside sphere");
    }

    #[test]
    fn test_explosion_helper() {
        let shape = destruction_from_explosion(
            Vec3Fix::from_f32(1.0, 0.0, 0.0),
            2.0,
            0.1,
        );
        assert!(shape.smooth_factor > 0.0, "Explosion should use smooth subtraction");
    }

    #[test]
    fn test_normal_after_destruction() {
        let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));
        dsdf.apply_destruction(DestructionShape::sphere(
            Vec3Fix::from_f32(1.0, 0.0, 0.0),
            0.3,
        ));

        let (nx, ny, nz) = dsdf.normal(1.0, 0.0, 0.0);
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        assert!((len - 1.0).abs() < 0.1, "Normal should be approximately unit length");
    }
}
