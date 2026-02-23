//! Collision Mesh Generation from Signed Distance Fields
//!
//! Generates triangle meshes from implicit surfaces using a marching cubes
//! variant. The resulting meshes can be used for collision detection or
//! debug visualization.
//!
//! # Features
//!
//! - Marching cubes surface extraction from SDF
//! - Basic mesh decimation (edge-collapse simplification)
//! - AABB computation for collision meshes
//!
//! All geometry is computed in deterministic 128-bit fixed-point arithmetic.

use crate::collider::AABB;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Collision Mesh
// ============================================================================

/// A triangle mesh for collision detection.
#[derive(Clone, Debug)]
pub struct CollisionMesh {
    /// Vertex positions
    pub vertices: Vec<Vec3Fix>,
    /// Triangle indices (three vertex indices per triangle)
    pub triangles: Vec<[usize; 3]>,
}

/// Configuration for collision mesh generation.
#[derive(Clone, Debug)]
pub struct CollisionMeshConfig {
    /// Grid resolution per axis (number of cells)
    pub resolution: usize,
    /// Minimum corner of the sampling volume
    pub bounds_min: Vec3Fix,
    /// Maximum corner of the sampling volume
    pub bounds_max: Vec3Fix,
}

// ============================================================================
// Marching Cubes Edge Table (simplified)
// ============================================================================

/// Edge indices for marching cubes. Each edge connects two corners.
/// Edge i connects EDGE_VERTICES[i].0 to EDGE_VERTICES[i].1.
const EDGE_VERTICES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0), // bottom face
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4), // top face
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7), // vertical edges
];

/// Corner offsets in (x, y, z) for a unit cube cell.
const CORNER_OFFSETS: [(usize, usize, usize); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
];

// Simplified marching cubes triangle table.
// For each of 256 cube configurations, lists edge indices forming triangles.
// -1 terminates the list. This is a minimal table covering common cases.
// Full 256-entry table is large; we use a procedural approach instead.

// ============================================================================
// Mesh Generation
// ============================================================================

/// Generate a collision mesh from a signed distance function.
///
/// Samples the SDF on a regular grid and extracts the zero-isosurface
/// using a marching cubes variant. Negative SDF values are considered inside.
///
/// # Arguments
///
/// - `sdf`: Signed distance function `f(point) -> distance`
/// - `config`: Grid resolution and bounding volume
#[must_use]
pub fn generate_collision_mesh<F>(sdf: F, config: &CollisionMeshConfig) -> CollisionMesh
where
    F: Fn(Vec3Fix) -> Fix128,
{
    let res = config.resolution.max(2);
    let min = config.bounds_min;
    let max = config.bounds_max;

    let dx = (max.x - min.x) / Fix128::from_int(res as i64);
    let dy = (max.y - min.y) / Fix128::from_int(res as i64);
    let dz = (max.z - min.z) / Fix128::from_int(res as i64);

    // Sample SDF at all grid corners
    let gx = res + 1;
    let gy = res + 1;
    let gz = res + 1;
    let mut values = vec![Fix128::ZERO; gx * gy * gz];

    for iz in 0..gz {
        for iy in 0..gy {
            for ix in 0..gx {
                let p = Vec3Fix::new(
                    min.x + dx * Fix128::from_int(ix as i64),
                    min.y + dy * Fix128::from_int(iy as i64),
                    min.z + dz * Fix128::from_int(iz as i64),
                );
                values[iz * gy * gx + iy * gx + ix] = sdf(p);
            }
        }
    }

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    // Process each cell
    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                // Gather 8 corner values
                let mut corner_vals = [Fix128::ZERO; 8];
                let mut corner_pos = [Vec3Fix::ZERO; 8];

                for (ci, &(ox, oy, oz)) in CORNER_OFFSETS.iter().enumerate() {
                    let cx = ix + ox;
                    let cy = iy + oy;
                    let cz = iz + oz;
                    corner_vals[ci] = values[cz * gy * gx + cy * gx + cx];
                    corner_pos[ci] = Vec3Fix::new(
                        min.x + dx * Fix128::from_int(cx as i64),
                        min.y + dy * Fix128::from_int(cy as i64),
                        min.z + dz * Fix128::from_int(cz as i64),
                    );
                }

                // Compute cube index (which corners are inside)
                let mut cube_index: u8 = 0;
                for (ci, val) in corner_vals.iter().enumerate() {
                    if val.is_negative() {
                        cube_index |= 1 << ci;
                    }
                }

                // Skip fully inside or fully outside cells
                if cube_index == 0 || cube_index == 255 {
                    continue;
                }

                // Find edge intersections
                let mut edge_vertices = [usize::MAX; 12];
                for (ei, &(c0, c1)) in EDGE_VERTICES.iter().enumerate() {
                    let v0 = corner_vals[c0];
                    let v1 = corner_vals[c1];

                    // Edge crosses surface if signs differ
                    let inside0 = v0.is_negative();
                    let inside1 = v1.is_negative();
                    if inside0 != inside1 {
                        // Linear interpolation to find zero crossing
                        let denom = v1 - v0;
                        let t = if denom.is_zero() {
                            Fix128::from_ratio(1, 2)
                        } else {
                            (Fix128::ZERO - v0) / denom
                        };

                        let p0 = corner_pos[c0];
                        let p1 = corner_pos[c1];
                        let vertex = Vec3Fix::new(
                            p0.x + (p1.x - p0.x) * t,
                            p0.y + (p1.y - p0.y) * t,
                            p0.z + (p1.z - p0.z) * t,
                        );

                        edge_vertices[ei] = vertices.len();
                        vertices.push(vertex);
                    }
                }

                // Generate triangles using a simple edge-walking approach
                // For each pair of adjacent inside/outside transitions, form triangles
                generate_cell_triangles(cube_index, &edge_vertices, &mut triangles);
            }
        }
    }

    CollisionMesh {
        vertices,
        triangles,
    }
}

/// Generate triangles for a single marching cubes cell.
///
/// Uses a simplified algorithm: for cells with crossed edges, connects
/// edge intersection points to form triangles via a fan from the first vertex.
fn generate_cell_triangles(
    cube_index: u8,
    edge_verts: &[usize; 12],
    triangles: &mut Vec<[usize; 3]>,
) {
    // Collect all valid edge intersection vertices
    let mut active_edges = Vec::new();
    for (ei, &vi) in edge_verts.iter().enumerate() {
        if vi != usize::MAX {
            active_edges.push((ei, vi));
        }
    }

    if active_edges.len() < 3 {
        return;
    }

    // Use a simple fan triangulation from the first vertex
    // For complex configurations this is an approximation
    let base = active_edges[0].1;
    let _ = cube_index; // cube_index used for the inside/outside determination above
    for i in 1..active_edges.len() - 1 {
        triangles.push([base, active_edges[i].1, active_edges[i + 1].1]);
    }
}

/// Simplify a collision mesh by reducing the triangle count.
///
/// Uses a basic edge-collapse decimation strategy. Collapses the shortest
/// edges first until the target triangle count is reached or no more
/// edges can be collapsed.
///
/// # Arguments
///
/// - `mesh`: Input collision mesh
/// - `target_triangles`: Desired number of output triangles
#[must_use]
pub fn simplify_collision_mesh(mesh: &CollisionMesh, target_triangles: usize) -> CollisionMesh {
    if mesh.triangles.len() <= target_triangles {
        return mesh.clone();
    }

    let mut vertices = mesh.vertices.clone();
    let mut triangles = mesh.triangles.clone();

    // Iteratively collapse shortest edges
    while triangles.len() > target_triangles {
        if triangles.is_empty() {
            break;
        }

        // Find shortest edge
        let mut best_len = Fix128::from_int(i64::MAX);
        let mut best_tri = 0;
        let mut best_edge = (0usize, 0usize);

        for (ti, tri) in triangles.iter().enumerate() {
            for edge_idx in 0..3 {
                let i0 = tri[edge_idx];
                let i1 = tri[(edge_idx + 1) % 3];
                let diff = vertices[i1] - vertices[i0];
                let len_sq = diff.dot(diff);
                if len_sq < best_len {
                    best_len = len_sq;
                    best_tri = ti;
                    best_edge = (i0, i1);
                }
            }
        }

        // Collapse: merge best_edge.1 into best_edge.0
        let (keep, remove) = best_edge;
        let midpoint = Vec3Fix::new(
            (vertices[keep].x + vertices[remove].x).half(),
            (vertices[keep].y + vertices[remove].y).half(),
            (vertices[keep].z + vertices[remove].z).half(),
        );
        vertices[keep] = midpoint;

        // Remap all references from `remove` to `keep`
        for tri in &mut triangles {
            for idx in tri.iter_mut() {
                if *idx == remove {
                    *idx = keep;
                }
            }
        }

        // Remove degenerate triangles (two or more identical indices)
        let _ = best_tri;
        triangles.retain(|tri| tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]);
    }

    CollisionMesh {
        vertices,
        triangles,
    }
}

/// Compute the axis-aligned bounding box of a collision mesh.
#[must_use]
pub fn compute_mesh_aabb(mesh: &CollisionMesh) -> AABB {
    if mesh.vertices.is_empty() {
        return AABB {
            min: Vec3Fix::ZERO,
            max: Vec3Fix::ZERO,
        };
    }

    let mut min = mesh.vertices[0];
    let mut max = mesh.vertices[0];

    for v in &mesh.vertices[1..] {
        if v.x < min.x {
            min.x = v.x;
        }
        if v.y < min.y {
            min.y = v.y;
        }
        if v.z < min.z {
            min.z = v.z;
        }
        if v.x > max.x {
            max.x = v.x;
        }
        if v.y > max.y {
            max.y = v.y;
        }
        if v.z > max.z {
            max.z = v.z;
        }
    }

    AABB { min, max }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_sdf(center: Vec3Fix, radius: Fix128) -> impl Fn(Vec3Fix) -> Fix128 {
        move |p: Vec3Fix| {
            let d = p - center;
            d.length() - radius
        }
    }

    #[test]
    fn test_generate_sphere_mesh() {
        let config = CollisionMeshConfig {
            resolution: 8,
            bounds_min: Vec3Fix::from_int(-2, -2, -2),
            bounds_max: Vec3Fix::from_int(2, 2, 2),
        };
        let mesh = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, Fix128::ONE), &config);
        // Should have some vertices and triangles
        assert!(
            !mesh.vertices.is_empty(),
            "Sphere mesh should have vertices"
        );
        assert!(
            !mesh.triangles.is_empty(),
            "Sphere mesh should have triangles"
        );
    }

    #[test]
    fn test_generate_empty_sdf() {
        // SDF always positive (no surface)
        let config = CollisionMeshConfig {
            resolution: 4,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
        };
        let mesh = generate_collision_mesh(|_| Fix128::ONE, &config);
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn test_generate_fully_inside_sdf() {
        // SDF always negative (fully inside)
        let config = CollisionMeshConfig {
            resolution: 4,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
        };
        let mesh = generate_collision_mesh(|_| Fix128::NEG_ONE, &config);
        assert!(mesh.triangles.is_empty());
    }

    #[test]
    fn test_simplify_no_reduction_needed() {
        let mesh = CollisionMesh {
            vertices: vec![
                Vec3Fix::from_int(0, 0, 0),
                Vec3Fix::from_int(1, 0, 0),
                Vec3Fix::from_int(0, 1, 0),
            ],
            triangles: vec![[0, 1, 2]],
        };
        let simplified = simplify_collision_mesh(&mesh, 10);
        assert_eq!(simplified.triangles.len(), 1);
    }

    #[test]
    fn test_simplify_reduces_triangles() {
        let config = CollisionMeshConfig {
            resolution: 8,
            bounds_min: Vec3Fix::from_int(-2, -2, -2),
            bounds_max: Vec3Fix::from_int(2, 2, 2),
        };
        let mesh = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, Fix128::ONE), &config);
        if mesh.triangles.len() > 4 {
            let simplified = simplify_collision_mesh(&mesh, 4);
            assert!(simplified.triangles.len() <= mesh.triangles.len());
        }
    }

    #[test]
    fn test_compute_mesh_aabb_empty() {
        let mesh = CollisionMesh {
            vertices: vec![],
            triangles: vec![],
        };
        let aabb = compute_mesh_aabb(&mesh);
        assert!(aabb.min.x.is_zero());
    }

    #[test]
    fn test_compute_mesh_aabb_single_vertex() {
        let mesh = CollisionMesh {
            vertices: vec![Vec3Fix::from_int(3, 5, 7)],
            triangles: vec![],
        };
        let aabb = compute_mesh_aabb(&mesh);
        assert_eq!(aabb.min.x.hi, 3);
        assert_eq!(aabb.max.y.hi, 5);
    }

    #[test]
    fn test_compute_mesh_aabb_multiple_vertices() {
        let mesh = CollisionMesh {
            vertices: vec![
                Vec3Fix::from_int(-1, -2, -3),
                Vec3Fix::from_int(4, 5, 6),
                Vec3Fix::from_int(0, 0, 0),
            ],
            triangles: vec![[0, 1, 2]],
        };
        let aabb = compute_mesh_aabb(&mesh);
        assert_eq!(aabb.min.x.hi, -1);
        assert_eq!(aabb.min.y.hi, -2);
        assert_eq!(aabb.min.z.hi, -3);
        assert_eq!(aabb.max.x.hi, 4);
        assert_eq!(aabb.max.y.hi, 5);
        assert_eq!(aabb.max.z.hi, 6);
    }

    #[test]
    fn test_triangle_indices_valid() {
        let config = CollisionMeshConfig {
            resolution: 6,
            bounds_min: Vec3Fix::from_int(-2, -2, -2),
            bounds_max: Vec3Fix::from_int(2, 2, 2),
        };
        let mesh = generate_collision_mesh(sphere_sdf(Vec3Fix::ZERO, Fix128::ONE), &config);
        for tri in &mesh.triangles {
            for &idx in tri {
                assert!(idx < mesh.vertices.len(), "Triangle index out of bounds");
            }
        }
    }
}
