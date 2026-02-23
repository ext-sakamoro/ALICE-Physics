//! SDF Sphere Tracing CCD (Continuous Collision Detection)
//!
//! Uses sphere tracing (ray marching) along the SDF to detect
//! time of impact for arbitrary SDF shapes. Works with concave,
//! CSG, and fractal geometry — anything expressible as an SDF.
//!
//! # Algorithm
//!
//! 1. Start at body position at t=0
//! 2. Evaluate SDF distance at current position
//! 3. Advance along trajectory by `distance - radius` (safe step)
//! 4. Repeat until contact or t > 1
//!
//! This is a generalization of conservative advancement that directly
//! uses the SDF's Lipschitz property for optimal step sizes.
//!
//! Author: Moroya Sakamoto

#[cfg(feature = "std")]
use crate::ccd::TOI;
use crate::math::Fix128;
#[cfg(feature = "std")]
use crate::math::Vec3Fix;
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;

// ============================================================================
// SDF CCD Configuration
// ============================================================================

/// Configuration for SDF sphere tracing CCD
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdfCcdConfig {
    /// Maximum sphere tracing iterations
    pub max_iterations: usize,
    /// Contact tolerance (stop when gap < this)
    pub tolerance: f32,
    /// Minimum velocity magnitude to trigger CCD
    pub velocity_threshold: Fix128,
    /// Safety margin multiplier for step size (< 1.0 for conservative)
    pub step_safety: f32,
}

impl Default for SdfCcdConfig {
    fn default() -> Self {
        Self {
            max_iterations: 64,
            tolerance: 0.001,
            velocity_threshold: Fix128::from_int(5),
            step_safety: 0.9,
        }
    }
}

// ============================================================================
// SDF Sphere Tracing CCD
// ============================================================================

/// Sphere trace a moving sphere against an SDF collider.
///
/// Returns time of impact in [0, 1] where:
/// - 0 = start of timestep (already penetrating)
/// - 1 = end of timestep
///
/// The sphere moves from `start` to `start + displacement` over [0, 1].
#[cfg(feature = "std")]
#[must_use]
pub fn sphere_trace_sdf(
    start: Vec3Fix,
    displacement: Vec3Fix,
    radius: Fix128,
    sdf: &SdfCollider,
    config: &SdfCcdConfig,
) -> Option<TOI> {
    let radius_f32 = radius.to_f32();
    let scale = sdf.scale_f32;
    let disp_len = displacement.length();

    if disp_len.is_zero() {
        return None;
    }

    let inv_disp_len = Fix128::ONE / disp_len;
    let mut t = Fix128::ZERO;

    for _ in 0..config.max_iterations {
        // Current position along trajectory
        let pos = start + displacement * t;

        // SDF distance query in local space
        let (lx, ly, lz) = sdf.world_to_local(pos);
        let dist = sdf.field.distance(lx, ly, lz) * scale;

        // Gap = distance to surface minus sphere radius
        let gap = dist - radius_f32;

        if gap <= config.tolerance {
            // Contact found
            let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
            let normal = sdf.local_normal_to_world(nx, ny, nz);
            let point = pos - normal * Fix128::from_f32(dist);

            return Some(TOI { t, point, normal });
        }

        // Safe advance: the SDF guarantees no surface within `gap` distance
        let safe_step = Fix128::from_f32(gap * config.step_safety);
        let dt = safe_step * inv_disp_len;
        t = t + dt;

        if t > Fix128::ONE {
            return None; // No collision this timestep
        }
    }

    None // Max iterations reached without convergence
}

/// Sphere trace a moving point (zero radius) against an SDF.
///
/// Simpler variant for raycasting against SDF geometry.
#[cfg(feature = "std")]
#[must_use]
pub fn ray_march_sdf(
    origin: Vec3Fix,
    direction: Vec3Fix,
    max_distance: Fix128,
    sdf: &SdfCollider,
    config: &SdfCcdConfig,
) -> Option<TOI> {
    if direction.length_squared().is_zero() {
        return None;
    }
    let scale = sdf.scale_f32;
    let mut t = Fix128::ZERO;

    for _ in 0..config.max_iterations {
        let pos = origin + direction * t;

        let (lx, ly, lz) = sdf.world_to_local(pos);
        let dist = sdf.field.distance(lx, ly, lz) * scale;

        if dist < config.tolerance {
            let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
            let normal = sdf.local_normal_to_world(nx, ny, nz);
            let point = pos;

            return Some(TOI { t, point, normal });
        }

        t = t + Fix128::from_f32(dist * config.step_safety);

        if t > max_distance {
            return None;
        }
    }

    None
}

/// Batch sphere trace: test multiple bodies against multiple SDF colliders.
///
/// Returns (`body_index`, `sdf_index`, TOI) tuples for all detected impacts.
#[cfg(feature = "std")]
#[must_use]
pub fn batch_sphere_trace_sdf(
    bodies: &[crate::solver::RigidBody],
    displacements: &[Vec3Fix],
    radius: Fix128,
    sdf_colliders: &[SdfCollider],
    config: &SdfCcdConfig,
) -> Vec<(usize, usize, TOI)> {
    let mut results = Vec::new();

    for (body_idx, (body, disp)) in bodies.iter().zip(displacements.iter()).enumerate() {
        if body.is_static() {
            continue;
        }

        // Check if body is moving fast enough for CCD
        let speed = body.velocity.length();
        if speed < config.velocity_threshold {
            continue;
        }

        for (sdf_idx, sdf) in sdf_colliders.iter().enumerate() {
            if sdf.body_index == body_idx {
                continue;
            }

            if let Some(toi) = sphere_trace_sdf(body.position, *disp, radius, sdf, config) {
                results.push((body_idx, sdf_idx, toi));
            }
        }
    }

    // Sort by earliest TOI for deterministic processing
    results.sort_by(|a, b| a.2.t.cmp(&b.2.t));
    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::sdf_collider::{ClosureSdf, SdfCollider};

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

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    #[test]
    fn test_sphere_trace_hit() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let start = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
        let displacement = Vec3Fix::from_f32(10.0, 0.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_some(), "Should detect collision with sphere");

        let toi = result.unwrap();
        let t = toi.t.to_f32();
        // Start at -5, sphere at 0, radius 1 + body radius 0.5 = contact at x = -1.5
        // t = (5 - 1.5) / 10 = 0.35
        assert!(t > 0.2 && t < 0.5, "TOI should be ~0.35, got {}", t);
    }

    #[test]
    fn test_sphere_trace_miss() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Moving parallel to sphere, missing it
        let start = Vec3Fix::from_f32(-5.0, 5.0, 0.0);
        let displacement = Vec3Fix::from_f32(10.0, 0.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_none(), "Should miss the sphere");
    }

    #[test]
    fn test_sphere_trace_ground() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Falling sphere toward ground
        let start = Vec3Fix::from_f32(0.0, 5.0, 0.0);
        let displacement = Vec3Fix::from_f32(0.0, -10.0, 0.0);
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default();

        let result = sphere_trace_sdf(start, displacement, radius, &sdf, &config);
        assert!(result.is_some(), "Should detect ground collision");

        let toi = result.unwrap();
        let t = toi.t.to_f32();
        // Start at y=5, ground at y=0, radius=0.5, contact at y=0.5
        // t = (5 - 0.5) / 10 = 0.45
        assert!(t > 0.3 && t < 0.6, "TOI should be ~0.45, got {}", t);
    }

    #[test]
    fn test_ray_march() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let origin = Vec3Fix::from_f32(-5.0, 0.0, 0.0);
        let direction = Vec3Fix::UNIT_X;
        let max_dist = Fix128::from_int(20);
        let config = SdfCcdConfig::default();

        let result = ray_march_sdf(origin, direction, max_dist, &sdf, &config);
        assert!(result.is_some(), "Ray should hit sphere");
    }
}
