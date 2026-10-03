//! Convex Decomposition from SDF
//!
//! Approximate convex decomposition of SDF shapes into `ConvexHull` groups.
//! Allows using fast GJK/EPA for SDF-derived geometry.
//!
//! # Algorithm
//!
//! 1. Sample the SDF at the centres of a voxel grid
//! 2. Extract the surface cells: inside cells with an outside neighbour
//! 3. Split the surface cells by axis-aligned cuts at the middle of their
//!    bounding box, **only while the hull of a cluster is concave**
//! 4. Generate a `ConvexHull` per cluster
//!
//! # Concavity
//!
//! The concavity of a cluster is the volume of the grid cells that lie inside the
//! hull of its surface centres but outside the solid (SDF > 0), divided by the
//! volume of the hull. A convex solid has concavity 0, so it is never cut; a
//! cluster is cut while its concavity is above
//! [`DecomposeConfig::concavity_threshold`] and the hull budget
//! ([`DecomposeConfig::max_hulls`]) allows.
//!
//! Surface centres are the centres of *inside* cells, so a hull is smaller than
//! the solid by up to one cell on each side.
//!
//! Author: Moroya Sakamoto

use crate::collider::ConvexHull;
use crate::convex_mesh_builder::{build_hull_mesh, HullMesh};
use crate::mass_properties::convex_hull_mass_properties;
use crate::math::{Fix128, Vec3Fix};
use crate::sdf_collider::SdfField;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Decomposition Configuration
// ============================================================================

/// Convex decomposition configuration
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecomposeConfig {
    /// Voxel grid resolution per axis
    pub resolution: usize,
    /// Maximum convex hulls in decomposition
    pub max_hulls: usize,
    /// A cluster is split while its concavity (cells inside its hull but outside
    /// the solid, as a fraction of the hull's volume) is above this
    pub concavity_threshold: f32,
    /// Minimum hull volume: smaller hulls are dropped
    pub min_volume: f32,
    /// Maximum vertices per hull (at least 4 are kept)
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
    /// Volume of each hull (the volume of its solid, not of its bounding box)
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
                    let fx = (x as f32 + 0.5).mul_add(cell_size.to_f32(), min.x.to_f32());
                    let fy = (y as f32 + 0.5).mul_add(cell_size.to_f32(), min.y.to_f32());
                    let fz = (z as f32 + 0.5).mul_add(cell_size.to_f32(), min.z.to_f32());
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

    /// The fraction of the hull of `points` that is not solid: the grid cells whose
    /// centre is inside the hull but outside the SDF (distance > 0), measured as
    /// cell volume over the hull's volume. 0 for a convex solid; 0 too when the
    /// points do not span a volume (nothing to cut).
    fn concavity(&self, points: &[Vec3Fix]) -> f32 {
        let Some(mesh) = build_hull_mesh(points) else {
            return 0.0;
        };
        let hull_volume = convex_hull_mass_properties(&mesh.vertices, Fix128::ONE)
            .mass
            .to_f32();
        // Outward planes `n · p <= d`, in f32 like the SDF samples.
        let planes: Vec<([f32; 3], f32)> = mesh
            .faces
            .iter()
            .map(|f| {
                let [a, b, c] = [
                    mesh.vertices[f[0]],
                    mesh.vertices[f[1]],
                    mesh.vertices[f[2]],
                ];
                let n = (b - a).cross(c - a);
                let n = [n.x.to_f32(), n.y.to_f32(), n.z.to_f32()];
                let d = n[0] * a.x.to_f32() + n[1] * a.y.to_f32() + n[2] * a.z.to_f32();
                (n, d)
            })
            .collect();
        // Cells that can be inside the hull: its bounding box, in cell indices.
        let (lo, hi) = compute_bounds(&mesh.vertices);
        let cell = self.cell_size.to_f32();
        let range = |lo: Fix128, hi: Fix128, origin: Fix128| {
            let first = ((lo - origin).to_f32() / cell).floor().max(0.0) as usize;
            let last = (((hi - origin).to_f32() / cell).ceil().max(0.0) as usize).min(self.res);
            first..last
        };
        let mut outside = 0usize;
        for z in range(lo.z, hi.z, self.min.z) {
            for y in range(lo.y, hi.y, self.min.y) {
                for x in range(lo.x, hi.x, self.min.x) {
                    if self.get(x, y, z) <= 0.0 {
                        continue;
                    }
                    let p = self.voxel_to_world(x, y, z);
                    let p = [p.x.to_f32(), p.y.to_f32(), p.z.to_f32()];
                    let inside = planes
                        .iter()
                        .all(|(n, d)| n[0] * p[0] + n[1] * p[1] + n[2] * p[2] <= *d);
                    if inside {
                        outside += 1;
                    }
                }
            }
        }
        outside as f32 * cell * cell * cell / hull_volume
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
/// and each region produces a `ConvexHull`.
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

    // 2. Cluster surface points into groups (index-based, no point copies)
    let mut cluster_indices: Vec<usize> = Vec::new();
    let mut cluster_offsets: Vec<usize> = Vec::new();
    let point_indices: Vec<usize> = (0..surface_points.len()).collect();
    cluster_points_indexed(
        &grid,
        config,
        &surface_points,
        &point_indices,
        config.max_hulls,
        &mut cluster_indices,
        &mut cluster_offsets,
    );
    cluster_offsets.push(cluster_indices.len());

    // 3. Generate convex hull per cluster
    let mut hulls = Vec::new();
    let mut centers = Vec::new();
    let mut volumes = Vec::new();

    for w in cluster_offsets.windows(2) {
        let start = w[0];
        let end = w[1];
        let cluster_len = end - start;
        if cluster_len < 4 {
            continue;
        }

        // Only the hull's own vertices matter: dropping the points inside it changes
        // nothing, while a blind stride over all the points would drop the corners.
        let points: Vec<Vec3Fix> = cluster_indices[start..end]
            .iter()
            .map(|&i| surface_points[i])
            .collect();
        let extreme = build_hull_mesh(&points).map_or(points, |mesh| extreme_vertices(&mesh));

        // Limit vertices per hull: the `cap` corners furthest from one another, so a
        // curved cluster keeps its outline instead of one side of it
        let verts = spread_subset(extreme, config.max_vertices_per_hull.max(4));

        // The volume and centre of mass of the hull's solid (zero volume when the
        // points do not span one)
        let props = convex_hull_mass_properties(&verts, Fix128::ONE);
        let vol = props.mass;
        let center = props.center_of_mass;

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

/// Spatial clustering using axis-aligned splitting.
///
/// Works on indices into `all_points` rather than copying point data.
/// Results are appended to `out_indices`; `out_offsets` records the start
/// of each cluster (the caller appends `out_indices.len()` as the final sentinel).
fn cluster_points_indexed(
    grid: &VoxelGrid,
    config: &DecomposeConfig,
    all_points: &[Vec3Fix],
    indices: &[usize],
    max_clusters: usize,
    out_indices: &mut Vec<usize>,
    out_offsets: &mut Vec<usize>,
) {
    // A cluster is cut only while it is concave and the hull budget allows.
    let cluster: Vec<Vec3Fix> = indices.iter().map(|&i| all_points[i]).collect();
    if max_clusters <= 1
        || indices.len() < 8
        || grid.concavity(&cluster) <= config.concavity_threshold
    {
        out_offsets.push(out_indices.len());
        out_indices.extend_from_slice(indices);
        return;
    }

    // Compute bounds over this subset
    let first = all_points[indices[0]];
    let mut bmin = first;
    let mut bmax = first;
    for &i in &indices[1..] {
        let p = all_points[i];
        if p.x < bmin.x {
            bmin.x = p.x;
        }
        if p.y < bmin.y {
            bmin.y = p.y;
        }
        if p.z < bmin.z {
            bmin.z = p.z;
        }
        if p.x > bmax.x {
            bmax.x = p.x;
        }
        if p.y > bmax.y {
            bmax.y = p.y;
        }
        if p.z > bmax.z {
            bmax.z = p.z;
        }
    }
    let extent = bmax - bmin;

    let (split_axis, split_val) = if extent.x >= extent.y && extent.x >= extent.z {
        (0, (bmin.x + bmax.x).half())
    } else if extent.y >= extent.z {
        (1, (bmin.y + bmax.y).half())
    } else {
        (2, (bmin.z + bmax.z).half())
    };

    let mut left_idx: Vec<usize> = Vec::new();
    let mut right_idx: Vec<usize> = Vec::new();

    for &i in indices {
        let p = all_points[i];
        let val = match split_axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        };
        if val < split_val {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    // Guard against degenerate splits
    if left_idx.is_empty() || right_idx.is_empty() {
        out_offsets.push(out_indices.len());
        out_indices.extend_from_slice(indices);
        return;
    }

    let half = max_clusters / 2;
    cluster_points_indexed(
        grid,
        config,
        all_points,
        &left_idx,
        half.max(1),
        out_indices,
        out_offsets,
    );
    cluster_points_indexed(
        grid,
        config,
        all_points,
        &right_idx,
        (max_clusters - half).max(1),
        out_indices,
        out_offsets,
    );
}

/// At most `cap` of `points`, chosen to be far from one another: starting from the
/// first point, each next one is the point furthest from those already chosen
/// (farthest-point sampling). All of `points` when there are no more than `cap`.
fn spread_subset(points: Vec<Vec3Fix>, cap: usize) -> Vec<Vec3Fix> {
    if points.len() <= cap {
        return points;
    }
    let mut chosen = vec![points[0]];
    // The squared distance from each point to the nearest chosen one so far.
    let mut nearest: Vec<Fix128> = points
        .iter()
        .map(|&q| (q - points[0]).length_squared())
        .collect();
    while chosen.len() < cap {
        let mut far = 0usize;
        for (i, d) in nearest.iter().enumerate() {
            if *d > nearest[far] {
                far = i;
            }
        }
        let p = points[far];
        chosen.push(p);
        for (d, &q) in nearest.iter_mut().zip(&points) {
            let to_p = (q - p).length_squared();
            if to_p < *d {
                *d = to_p;
            }
        }
    }
    chosen
}

/// The corners of a hull mesh: the vertices whose faces span all three directions.
///
/// [`build_hull_mesh`] also keeps points lying in a face or on an edge (they are
/// vertices of the triangulation), which add nothing to the hull's shape: the
/// normals of the faces around such a point lie in one plane. A corner has faces
/// with three independent normals.
fn extreme_vertices(mesh: &HullMesh) -> Vec<Vec3Fix> {
    let unit = |v: Vec3Fix| -> Option<[f64; 3]> {
        let (x, y, z) = (v.x.to_f64(), v.y.to_f64(), v.z.to_f64());
        let len = (x * x + y * y + z * z).sqrt();
        (len > 0.0).then(|| [x / len, y / len, z / len])
    };
    let mut around: Vec<Vec<[f64; 3]>> = vec![Vec::new(); mesh.vertices.len()];
    for f in &mesh.faces {
        let [a, b, c] = [
            mesh.vertices[f[0]],
            mesh.vertices[f[1]],
            mesh.vertices[f[2]],
        ];
        if let Some(n) = unit((b - a).cross(c - a)) {
            for &i in f {
                around[i].push(n);
            }
        }
    }
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let spans_space = |normals: &[[f64; 3]]| {
        normals.iter().enumerate().any(|(i, &a)| {
            normals[i + 1..].iter().enumerate().any(|(j, &b)| {
                let ab = cross(a, b);
                ab.iter().map(|c| c * c).sum::<f64>() > 1e-12
                    && normals[i + 1 + j + 1..]
                        .iter()
                        .any(|c| (ab[0] * c[0] + ab[1] * c[1] + ab[2] * c[2]).abs() > 1e-6)
            })
        })
    };
    mesh.vertices
        .iter()
        .zip(&around)
        .filter(|(_, normals)| spans_space(normals))
        .map(|(&v, _)| v)
        .collect()
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
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
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

    /// Fewer than eight points are never cut, however concave: the pieces would have
    /// fewer than four points each and be dropped, losing the solid. Seven corners of
    /// a grid with solid cells only at those corners leave nearly all of their hull
    /// outside the solid.
    #[test]
    fn test_cluster_of_fewer_than_eight_points_is_not_cut() {
        let corners = [
            (-0.8f32, -0.8f32, -0.8f32),
            (0.8, -0.8, -0.8),
            (-0.8, 0.8, -0.8),
            (0.8, 0.8, -0.8),
            (-0.8, -0.8, 0.8),
            (0.8, -0.8, 0.8),
            (-0.8, 0.8, 0.8),
        ];
        // Solid (inside) only within 0.2 of a corner: the 8-cell grid's cells there.
        let field = ClosureSdf::new(
            move |x, y, z| {
                if corners.iter().any(|c| {
                    (x - c.0).abs() < 0.2 && (y - c.1).abs() < 0.2 && (z - c.2).abs() < 0.2
                }) {
                    -1.0
                } else {
                    1.0
                }
            },
            |_, _, _| (0.0, 1.0, 0.0),
        );
        let grid = VoxelGrid::new(
            &field,
            Vec3Fix::from_f32(-1.0, -1.0, -1.0),
            Vec3Fix::from_f32(1.0, 1.0, 1.0),
            8,
        );
        let points: Vec<Vec3Fix> = corners
            .iter()
            .map(|c| Vec3Fix::from_f32(c.0, c.1, c.2))
            .collect();
        assert!(grid.concavity(&points) > 0.5, "the hull is mostly outside");
        let indices: Vec<usize> = (0..points.len()).collect();
        let mut out_indices = Vec::new();
        let mut out_offsets = Vec::new();
        cluster_points_indexed(
            &grid,
            &DecomposeConfig::default(),
            &points,
            &indices,
            8,
            &mut out_indices,
            &mut out_offsets,
        );
        assert_eq!(out_offsets.len(), 1, "seven points stay one cluster");
    }

    /// A cluster that spans no volume has no concavity: it is never cut, whatever
    /// the budget (the cuts that matter are covered by `analytic_convex_decompose`).
    #[test]
    fn test_flat_cluster_is_not_split() {
        let sphere = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 0.5,
            |_, _, _| (0.0, 1.0, 0.0),
        );
        let grid = VoxelGrid::new(
            &sphere,
            Vec3Fix::from_f32(-1.0, -1.0, -1.0),
            Vec3Fix::from_f32(1.0, 1.0, 1.0),
            8,
        );
        let points: Vec<Vec3Fix> = (0..10)
            .map(|i| Vec3Fix::from_f32(-1.0 + 0.2 * i as f32, 0.0, 0.0))
            .collect();
        let indices: Vec<usize> = (0..points.len()).collect();
        let mut out_indices = Vec::new();
        let mut out_offsets = Vec::new();
        cluster_points_indexed(
            &grid,
            &DecomposeConfig::default(),
            &points,
            &indices,
            8,
            &mut out_indices,
            &mut out_offsets,
        );
        assert_eq!(out_offsets.len(), 1, "one cluster");
        assert_eq!(out_indices.len(), points.len());
    }
}
