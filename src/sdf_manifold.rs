//! SDF Contact Manifold Generation
//!
//! Multi-point contact manifold from SDF surface sampling.
//! Replaces single-point contacts with stable N-point manifolds
//! for robust stacking, sliding, and resting behavior.
//!
//! # Algorithm
//!
//! 1. Detect initial penetration via SDF distance query
//! 2. Sample SDF surface in a local tangent frame around the contact
//! 3. Build manifold from deepest-penetrating samples
//! 4. Reduce to 4-point manifold (maximizes contact area)
//!
//! Author: Moroya Sakamoto

use crate::collider::Contact;
use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Manifold Configuration
// ============================================================================

/// Configuration for SDF manifold generation
#[derive(Clone, Copy, Debug)]
pub struct ManifoldConfig {
    /// Number of sample points in each tangent direction (total = samples^2)
    pub samples_per_axis: usize,
    /// Sampling radius around the initial contact point
    pub sample_radius: f32,
    /// Maximum number of contacts in the final manifold
    pub max_contacts: usize,
    /// Minimum penetration depth to include in manifold
    pub min_depth: f32,
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            samples_per_axis: 5,
            sample_radius: 0.5,
            max_contacts: 4,
            min_depth: 0.001,
        }
    }
}

// ============================================================================
// Contact Manifold
// ============================================================================

/// Multi-point contact manifold from SDF surface
#[derive(Clone, Debug)]
pub struct SdfManifold {
    /// Contact points (up to max_contacts)
    pub contacts: Vec<Contact>,
    /// Average contact normal
    pub normal: Vec3Fix,
    /// Average penetration depth
    pub avg_depth: Fix128,
}

impl SdfManifold {
    /// Empty manifold
    pub fn empty() -> Self {
        Self {
            contacts: Vec::new(),
            normal: Vec3Fix::ZERO,
            avg_depth: Fix128::ZERO,
        }
    }

    /// Number of contact points
    #[inline]
    pub fn len(&self) -> usize {
        self.contacts.len()
    }

    /// Check if manifold has no contacts
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }

    /// Get the deepest contact
    pub fn deepest(&self) -> Option<&Contact> {
        self.contacts.iter().max_by(|a, b| a.depth.cmp(&b.depth))
    }
}

// ============================================================================
// Manifold Generation
// ============================================================================

/// Build a tangent frame from a normal vector.
///
/// Returns (tangent1, tangent2) orthogonal to the normal.
#[inline]
fn build_tangent_frame(normal: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
    let (nx, _ny, _nz) = normal.to_f32();

    // Choose axis least aligned with normal for stable cross product
    let up = if nx.abs() < 0.9 {
        Vec3Fix::UNIT_X
    } else {
        Vec3Fix::UNIT_Y
    };

    let t1 = normal.cross(up).normalize();
    let t2 = normal.cross(t1).normalize();
    (t1, t2)
}

/// Generate multi-point contact manifold from SDF.
///
/// Samples the SDF surface around an initial contact point to build
/// a stable manifold for constraint solving.
#[cfg(feature = "std")]
pub fn generate_sdf_manifold(
    center: Vec3Fix,
    radius: Fix128,
    sdf: &SdfCollider,
    config: &ManifoldConfig,
) -> SdfManifold {
    // 1. Initial probe: check if there's any contact at all
    let (lx, ly, lz) = sdf.world_to_local(center);
    let dist = sdf.field.distance(lx, ly, lz);
    let scale = sdf.scale_f32;
    let world_dist = dist * scale;
    let radius_f32 = radius.to_f32();

    if radius_f32 - world_dist <= 0.0 {
        return SdfManifold::empty();
    }

    // 2. Get initial normal and build tangent frame
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
    let world_normal = sdf.local_normal_to_world(nx, ny, nz);
    let (t1, t2) = build_tangent_frame(world_normal);

    // 3. Sample grid around contact point
    let n = config.samples_per_axis;
    let half = n / 2;
    let step = config.sample_radius / (half.max(1) as f32);
    let mut candidates: Vec<(Contact, f32)> = Vec::with_capacity(n * n);

    for i in 0..n {
        for j in 0..n {
            let di = (i as f32 - half as f32) * step;
            let dj = (j as f32 - half as f32) * step;

            let offset = t1 * Fix128::from_f32(di) + t2 * Fix128::from_f32(dj);
            let sample_pos = center + offset;

            let (slx, sly, slz) = sdf.world_to_local(sample_pos);
            let sample_dist = sdf.field.distance(slx, sly, slz) * scale;
            let penetration = radius_f32 - sample_dist;

            if penetration > config.min_depth {
                let (snx, sny, snz) = sdf.field.normal(slx, sly, slz);
                let sample_normal = sdf.local_normal_to_world(snx, sny, snz);
                let depth = Fix128::from_f32(penetration);

                let point_a = sample_pos - sample_normal * radius;
                let point_b = sample_pos - sample_normal * Fix128::from_f32(sample_dist);

                candidates.push((
                    Contact {
                        depth,
                        normal: sample_normal,
                        point_a,
                        point_b,
                    },
                    penetration,
                ));
            }
        }
    }

    if candidates.is_empty() {
        return SdfManifold::empty();
    }

    // 4. Reduce to max_contacts using area-maximizing selection
    let contacts = reduce_manifold(&candidates, config.max_contacts);

    // 5. Compute average normal and depth
    let mut sum_normal = Vec3Fix::ZERO;
    let mut sum_depth = Fix128::ZERO;
    for c in &contacts {
        sum_normal = sum_normal + c.normal;
        sum_depth = sum_depth + c.depth;
    }
    let count = Fix128::from_int(contacts.len() as i64);
    let avg_normal = sum_normal.normalize();
    let avg_depth = if contacts.is_empty() {
        Fix128::ZERO
    } else {
        sum_depth / count
    };

    SdfManifold {
        contacts,
        normal: avg_normal,
        avg_depth,
    }
}

/// Reduce contact candidates to N points maximizing contact area.
///
/// Algorithm:
/// 1. Keep the deepest contact
/// 2. Keep the point furthest from it
/// 3. Keep the point maximizing triangle area
/// 4. Keep the point maximizing quadrilateral area
fn reduce_manifold(candidates: &[(Contact, f32)], max_contacts: usize) -> Vec<Contact> {
    if candidates.len() <= max_contacts {
        return candidates.iter().map(|(c, _)| *c).collect();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(max_contacts);

    // 1. Deepest point
    let deepest_idx = candidates
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    selected.push(deepest_idx);

    // 2. Furthest from deepest
    if max_contacts >= 2 {
        let p0 = candidates[deepest_idx].0.point_b;
        let mut best_dist = Fix128::ZERO;
        let mut best_idx = 0;
        for (i, (c, _)) in candidates.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            let d = (c.point_b - p0).length_squared();
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        selected.push(best_idx);
    }

    // 3. Maximize triangle area
    if max_contacts >= 3 && candidates.len() >= 3 {
        let p0 = candidates[selected[0]].0.point_b;
        let p1 = candidates[selected[1]].0.point_b;
        let edge = p1 - p0;
        let mut best_area = Fix128::ZERO;
        let mut best_idx = 0;
        for (i, (c, _)) in candidates.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            let v = c.point_b - p0;
            let cross = edge.cross(v);
            let area = cross.length_squared();
            if area > best_area {
                best_area = area;
                best_idx = i;
            }
        }
        selected.push(best_idx);
    }

    // 4. Maximize quadrilateral area
    if max_contacts >= 4 && candidates.len() >= 4 {
        let p0 = candidates[selected[0]].0.point_b;
        let p1 = candidates[selected[1]].0.point_b;
        let p2 = candidates[selected[2]].0.point_b;
        let tri_normal = (p1 - p0).cross(p2 - p0);
        let mut best_dist = Fix128::ZERO;
        let mut best_idx = 0;
        for (i, (c, _)) in candidates.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            let v = c.point_b - p0;
            let d = v.dot(tri_normal).abs();
            if d > best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        selected.push(best_idx);
    }

    selected.iter().map(|&i| candidates[i].0).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::sdf_collider::ClosureSdf;

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

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
    fn test_manifold_ground_plane() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Sphere penetrating ground
        let center = Vec3Fix::from_f32(0.0, 0.3, 0.0);
        let radius = Fix128::ONE;
        let config = ManifoldConfig::default();

        let manifold = generate_sdf_manifold(center, radius, &sdf, &config);
        assert!(!manifold.is_empty(), "Should generate contacts");
        assert!(
            manifold.len() <= config.max_contacts,
            "Should not exceed max contacts"
        );
    }

    #[test]
    fn test_manifold_no_contact() {
        let sdf =
            SdfCollider::new_static(Box::new(ground_plane()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Sphere above ground
        let center = Vec3Fix::from_f32(0.0, 5.0, 0.0);
        let radius = Fix128::ONE;
        let config = ManifoldConfig::default();

        let manifold = generate_sdf_manifold(center, radius, &sdf, &config);
        assert!(manifold.is_empty(), "Should have no contacts");
    }

    #[test]
    fn test_manifold_sphere_sdf() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Small sphere inside SDF sphere
        let center = Vec3Fix::from_f32(0.5, 0.0, 0.0);
        let radius = Fix128::from_f32(0.3);
        let config = ManifoldConfig {
            samples_per_axis: 3,
            sample_radius: 0.2,
            ..Default::default()
        };

        let manifold = generate_sdf_manifold(center, radius, &sdf, &config);
        assert!(
            !manifold.is_empty(),
            "Should generate contacts inside sphere"
        );
    }

    #[test]
    fn test_tangent_frame() {
        let normal = Vec3Fix::UNIT_Y;
        let (t1, t2) = build_tangent_frame(normal);

        // t1 and t2 should be orthogonal to normal
        let d1 = normal.dot(t1).to_f32();
        let d2 = normal.dot(t2).to_f32();
        assert!(d1.abs() < 0.01, "t1 should be orthogonal to normal");
        assert!(d2.abs() < 0.01, "t2 should be orthogonal to normal");
    }

    #[test]
    fn test_reduce_manifold() {
        let contacts: Vec<(Contact, f32)> = (0..10)
            .map(|i| {
                let angle = (i as f32) * 0.628; // ~2pi/10
                (
                    Contact {
                        depth: Fix128::from_f32(0.1 + i as f32 * 0.01),
                        normal: Vec3Fix::UNIT_Y,
                        point_a: Vec3Fix::from_f32(angle.cos(), 0.0, angle.sin()),
                        point_b: Vec3Fix::from_f32(angle.cos(), -0.1, angle.sin()),
                    },
                    0.1 + i as f32 * 0.01,
                )
            })
            .collect();

        let reduced = reduce_manifold(&contacts, 4);
        assert_eq!(reduced.len(), 4, "Should reduce to 4 contacts");
    }
}
