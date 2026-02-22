//! Convex Decomposition from SDF
//!
//! Approximate convex decomposition of SDF shapes into ConvexHull groups.
//! Allows using fast GJK/EPA for SDF-derived geometry.
//!
//! # Algorithm
//!
//! 1. Sample SDF surface on a voxel grid
//! 2. Extract surface voxels (sign change)
//! 3. Flood-fill connected convex regions
//! 4. Generate ConvexHull per region
//!
//! Author: Moroya Sakamoto

use crate::collider::ConvexHull;
use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfField;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Decomposition Configuration
// ============================================================================

/// Convex decomposition configuration
#[derive(Clone, Copy, Debug)]
pub struct DecomposeConfig {
    /// Voxel grid resolution per axis
    pub resolution: usize,
    /// Maximum convex hulls in decomposition
    pub max_hulls: usize,
    /// Concavity threshold for splitting
    pub concavity_threshold: f32,
    /// Minimum hull volume (skip tiny hulls)
    pub min_volume: f32,
    /// Maximum vertices per hull
    pub max_vertices_per_hull: usize,
}

impl Default for DecomposeConfig {
    fn default() -> Self {
        Self {
            resolution: 32,
            max_hulls: 16,
            concavity_threshold: 0.01,
            min_volume: 0.001,
            max_vertices_per_hull: 64,
        }
    }
}

/// Result of convex decomposition
#[derive(Clone, Debug)]
pub struct DecompositionResult {
    /// Generated convex hulls
    pub hulls: Vec<ConvexHull>,
    /// Center of each hull
    pub centers: Vec<Vec3Fix>,
    /// Volume of each hull
    pub volumes: Vec<Fix128>,
}

// ============================================================================
// Voxel Grid
// ============================================================================

/// 3D voxel grid for SDF sampling
struct VoxelGrid {
    /// SDF distance values
    values: Vec<f32>,
    /// Grid resolution
    res: usize,
    /// World-space minimum corner
    min: Vec3Fix,
    /// World-space cell size
    cell_size: Fix128,
}

impl VoxelGrid {
    fn new(sdf: &dyn SdfField, min: Vec3Fix, max: Vec3Fix, res: usize) -> Self {
        let size = max - min;
        let cell_size = size.x / Fix128::from_int(res as i64);
        let total = res * res * res;
        let mut values = vec![0.0f32; total];

        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let fx = min.x.to_f32() + (x as f32 + 0.5) * cell_size.to_f32();
                    let fy = min.y.to_f32() + (y as f32 + 0.5) * cell_size.to_f32();
                    let fz = min.z.to_f32() + (z as f32 + 0.5) * cell_size.to_f32();
                    values[x + y * res + z * res * res] = sdf.distance(fx, fy, fz);
                }
            }
        }

        Self {
            values,
            res,
            min,
            cell_size,
        }
    }

    #[inline]
    fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.values[x + y * self.res + z * self.res * self.res]
    }

    fn voxel_to_world(&self, x: usize, y: usize, z: usize) -> Vec3Fix {
        let cs = self.cell_size;
        Vec3Fix::new(
            self.min.x + cs * Fix128::from_int(x as i64) + cs.half(),
            self.min.y + cs * Fix128::from_int(y as i64) + cs.half(),
            self.min.z + cs * Fix128::from_int(z as i64) + cs.half(),
        )
    }

    /// Check if voxel is on the surface (sign change with neighbor)
    fn is_surface(&self, x: usize, y: usize, z: usize) -> bool {
        let v = self.get(x, y, z);
        if v > 0.0 {
            return false; // Outside
        }

        // Check 6-connected neighbors for sign change
        let r = self.res;
        if x > 0 && self.get(x - 1, y, z) > 0.0 {
            return true;
        }
        if x + 1 < r && self.get(x + 1, y, z) > 0.0 {
            return true;
        }
        if y > 0 && self.get(x, y - 1, z) > 0.0 {
            return true;
        }
        if y + 1 < r && self.get(x, y + 1, z) > 0.0 {
            return true;
        }
        if z > 0 && self.get(x, y, z - 1) > 0.0 {
            return true;
        }
        if z + 1 < r && self.get(x, y, z + 1) > 0.0 {
            return true;
        }

        false
    }
}

// ============================================================================
// Decomposition
// ============================================================================

/// Decompose an SDF into approximate convex hulls.
///
/// The SDF is sampled on a voxel grid within the given bounds,
/// surface voxels are grouped into connected convex regions,
/// and each region produces a ConvexHull.
pub fn decompose_sdf(
    sdf: &dyn SdfField,
    min: Vec3Fix,
    max: Vec3Fix,
    config: &DecomposeConfig,
) -> DecompositionResult {
    let grid = VoxelGrid::new(sdf, min, max, config.resolution);

    // 1. Extract surface points
    let mut surface_points: Vec<Vec3Fix> = Vec::new();
    for z in 0..config.resolution {
        for y in 0..config.resolution {
            for x in 0..config.resolution {
                if grid.is_surface(x, y, z) {
                    surface_points.push(grid.voxel_to_world(x, y, z));
                }
            }
        }
    }

    if surface_points.is_empty() {
        return DecompositionResult {
            hulls: Vec::new(),
            centers: Vec::new(),
            volumes: Vec::new(),
        };
    }

    // 2. Cluster surface points into groups (simple spatial splitting)
    let clusters = cluster_points(&surface_points, config.max_hulls);

    // 3. Generate convex hull per cluster
    let mut hulls = Vec::new();
    let mut centers = Vec::new();
    let mut volumes = Vec::new();

    for cluster in &clusters {
        if cluster.len() < 4 {
            continue;
        }

        // Limit vertices per hull
        let verts: Vec<Vec3Fix> = if cluster.len() > config.max_vertices_per_hull {
            // Subsample
            let step = cluster.len() / config.max_vertices_per_hull;
            cluster.iter().step_by(step.max(1)).cloned().collect()
        } else {
            cluster.clone()
        };

        // Compute center
        let center = verts.iter().fold(Vec3Fix::ZERO, |acc, &p| acc + p)
            / Fix128::from_int(verts.len() as i64);

        // Compute approximate volume (bounding box)
        let (bb_min, bb_max) = compute_bounds(&verts);
        let extent = bb_max - bb_min;
        let vol = extent.x * extent.y * extent.z;

        if vol.to_f32() < config.min_volume {
            continue;
        }

        hulls.push(ConvexHull::new(verts));
        centers.push(center);
        volumes.push(vol);
    }

    DecompositionResult {
        hulls,
        centers,
        volumes,
    }
}

/// Simple spatial clustering using axis-aligned splitting
fn cluster_points(points: &[Vec3Fix], max_clusters: usize) -> Vec<Vec<Vec3Fix>> {
    if points.len() <= max_clusters || max_clusters <= 1 {
        return vec![points.to_vec()];
    }

    // Split along longest axis
    let (min, max) = compute_bounds(points);
    let extent = max - min;

    let (split_axis, split_val) = if extent.x >= extent.y && extent.x >= extent.z {
        (0, (min.x + max.x).half())
    } else if extent.y >= extent.z {
        (1, (min.y + max.y).half())
    } else {
        (2, (min.z + max.z).half())
    };

    let mut left = Vec::new();
    let mut right = Vec::new();

    for &p in points {
        let val = match split_axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        };

        if val < split_val {
            left.push(p);
        } else {
            right.push(p);
        }
    }

    // Guard against degenerate splits
    if left.is_empty() || right.is_empty() {
        return vec![points.to_vec()];
    }

    let half = max_clusters / 2;
    let mut result = cluster_points(&left, half.max(1));
    result.extend(cluster_points(&right, (max_clusters - half).max(1)));

    result
}

/// Compute AABB bounds of a point set
///
/// # Panics
///
/// Panics if `points` is empty.
fn compute_bounds(points: &[Vec3Fix]) -> (Vec3Fix, Vec3Fix) {
    assert!(
        !points.is_empty(),
        "compute_bounds requires non-empty points"
    );
    let mut min = points[0];
    let mut max = points[0];

    for &p in &points[1..] {
        if p.x < min.x {
            min.x = p.x;
        }
        if p.y < min.y {
            min.y = p.y;
        }
        if p.z < min.z {
            min.z = p.z;
        }
        if p.x > max.x {
            max.x = p.x;
        }
        if p.y > max.y {
            max.y = p.y;
        }
        if p.z > max.z {
            max.z = p.z;
        }
    }

    (min, max)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    #[test]
    fn test_decompose_sphere() {
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

        let config = DecomposeConfig {
            resolution: 16,
            max_hulls: 4,
            ..Default::default()
        };

        let result = decompose_sdf(
            &sphere,
            Vec3Fix::from_f32(-2.0, -2.0, -2.0),
            Vec3Fix::from_f32(2.0, 2.0, 2.0),
            &config,
        );

        assert!(
            !result.hulls.is_empty(),
            "Should generate at least one hull"
        );
    }

    #[test]
    fn test_decompose_empty() {
        // SDF that has no interior (everything positive)
        let empty = ClosureSdf::new(|_, _, _| 10.0, |_, _, _| (0.0, 1.0, 0.0));

        let config = DecomposeConfig {
            resolution: 8,
            ..Default::default()
        };

        let result = decompose_sdf(
            &empty,
            Vec3Fix::from_f32(-1.0, -1.0, -1.0),
            Vec3Fix::from_f32(1.0, 1.0, 1.0),
            &config,
        );

        assert!(result.hulls.is_empty(), "Empty SDF should produce no hulls");
    }

    #[test]
    fn test_cluster_points() {
        let points = vec![
            Vec3Fix::from_f32(-1.0, 0.0, 0.0),
            Vec3Fix::from_f32(-0.9, 0.0, 0.0),
            Vec3Fix::from_f32(1.0, 0.0, 0.0),
            Vec3Fix::from_f32(0.9, 0.0, 0.0),
        ];

        let clusters = cluster_points(&points, 2);
        assert_eq!(clusters.len(), 2, "Should split into 2 clusters");
    }
}
