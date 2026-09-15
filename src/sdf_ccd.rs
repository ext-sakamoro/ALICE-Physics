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
    results.sort_by_key(|a| a.2.t);
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
        assert!(t > 0.2 && t < 0.5, "TOI should be ~0.35, got {t}");
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
        assert!(t > 0.3 && t < 0.6, "TOI should be ~0.45, got {t}");
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

    #[test]
    fn batch_sphere_trace_sdf_filters_bodies_and_sorts_by_closed_form_toi() {
        use crate::solver::RigidBody;
        let colliders = [
            // 0: 地面 y = 0
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY),
            // 1: 単位球 (原点)
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY),
            // 2: 単位球を body 3 に attach (body 3 に対しては self-collision skip)
            SdfCollider::new_dynamic(Box::new(unit_sphere()), 3),
        ];
        let radius = Fix128::from_f32(0.5);
        let config = SdfCcdConfig::default(); // velocity_threshold 5、tolerance 0.001

        let fast_down = Vec3Fix::from_int(0, -10, 0);
        let fast_right = Vec3Fix::from_int(10, 0, 0);
        let mut bodies = vec![
            // 0: (0, 5, 0) → 下へ 10: 球 top 接触 y = 1.5 → t 0.35 (sdf 1 と 2)、地面 y = 0.5 → t 0.45
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE)
                .with_velocity(fast_down),
            // 1: (-5, 5, 0) → 右へ 10: 球も地面も miss
            RigidBody::new_dynamic(Vec3Fix::from_int(-5, 5, 0), Fix128::ONE)
                .with_velocity(fast_right),
            // 2: static は速くても skip
            RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)).with_velocity(fast_down),
            // 3: (-7, 0, 0) → 右へ 10: 地面は既に貫入 (gap -0.5) → t 0、球 x = -1.5 → t 0.55、sdf 2 は self skip
            RigidBody::new_dynamic(Vec3Fix::from_int(-7, 0, 0), Fix128::ONE)
                .with_velocity(fast_right),
            // 4: body 0 と同じ軌道だが速度 1 < threshold 5 → skip
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 5, 0), Fix128::ONE)
                .with_velocity(Vec3Fix::from_int(0, -1, 0)),
        ];
        let displacements = [fast_down, fast_right, fast_down, fast_right, fast_down];

        let hits = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &config);
        let keys: Vec<(usize, usize)> = hits.iter().map(|(b, s, _)| (*b, *s)).collect();
        // t 昇順、同値 (body 0 の sdf 1 / 2 は同じ field) は stable sort で挿入順
        assert_eq!(keys, vec![(3, 0), (0, 1), (0, 2), (0, 0), (3, 1)]);
        let expected_t = [0.0_f32, 0.35, 0.35, 0.45, 0.55];
        for (k, want) in expected_t.iter().enumerate() {
            let t = hits[k].2.t.to_f32();
            assert!((t - want).abs() < 1e-3, "hit {k} t {t}, want {want}");
        }
        // 法線: 地面 +Y、球 top +Y、球の -x 側 -X
        let n = |k: usize| hits[k].2.normal.to_f32();
        for k in [0usize, 1, 2, 3] {
            let (nx, ny, nz) = n(k);
            assert!(
                nx.abs() < 1e-4 && (ny - 1.0).abs() < 1e-4 && nz.abs() < 1e-4,
                "hit {k}"
            );
        }
        let (nx, ny, nz) = n(4);
        assert!((nx + 1.0).abs() < 1e-4 && ny.abs() < 1e-4 && nz.abs() < 1e-4);
        // 接触点は SDF 表面上: 地面 hit の y ≈ 0、球 hit は |p| ≈ 1
        let (_, py, _) = hits[3].2.point.to_f32();
        assert!(py.abs() < 2e-3, "ground contact point y {py}");
        for k in [1usize, 2, 4] {
            let (px, py, pz) = hits[k].2.point.to_f32();
            let r = (px * px + py * py + pz * pz).sqrt();
            assert!((r - 1.0).abs() < 2e-3, "hit {k} contact point radius {r}");
        }

        // velocity_threshold を 0 にすると body 4 も (body 0 と同じ 3 hit で) 拾われる
        let lenient = SdfCcdConfig {
            velocity_threshold: Fix128::ZERO,
            ..SdfCcdConfig::default()
        };
        let all = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &lenient);
        assert_eq!(all.len(), 8);
        assert_eq!(all.iter().filter(|(b, _, _)| *b == 4).count(), 3);
        // zip: displacement が足りない body は評価されない
        let short =
            batch_sphere_trace_sdf(&bodies, &displacements[..1], radius, &colliders, &config);
        assert_eq!(short.len(), 3);
        assert!(short.iter().all(|(b, _, _)| *b == 0));
        // body 3 を static にすると (3, *) が消える
        bodies[3].inv_mass = Fix128::ZERO;
        let no3 = batch_sphere_trace_sdf(&bodies, &displacements, radius, &colliders, &config);
        assert_eq!(no3.len(), 3);
        assert!(no3.iter().all(|(b, _, _)| *b == 0));
    }
}
