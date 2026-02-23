//! Convex Hull Builder
//!
//! Incremental convex hull construction from a set of 3D points using
//! deterministic 128-bit fixed-point arithmetic.
//!
//! # Features
//!
//! - **Incremental Algorithm**: Builds convex hull by inserting points one at a time
//! - **Deterministic**: All computations use `Fix128` — no floating point
//! - **`no_std` compatible**: Uses `alloc::vec::Vec` when `std` is disabled
//!
//! # Algorithm
//!
//! 1. Find an initial tetrahedron from 4 non-coplanar points
//! 2. For each remaining point, test against all faces
//! 3. If the point is outside any face, remove visible faces and patch the hull
//!
//! # Output
//!
//! Returns a `ConvexHull` from the `collider` module, which can be used
//! directly with GJK/EPA collision detection.
//!
//! Author: Moroya Sakamoto

use crate::collider::ConvexHull;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Face of the convex hull (triangle with outward normal)
#[derive(Clone, Copy, Debug)]
struct HullFace {
    /// Vertex indices
    indices: [usize; 3],
    /// Outward-facing normal (not necessarily unit length)
    normal: Vec3Fix,
}

/// Compute the centroid (average) of a set of points
///
/// Returns the zero vector if the input slice is empty.
#[must_use]
pub fn compute_centroid(points: &[Vec3Fix]) -> Vec3Fix {
    if points.is_empty() {
        return Vec3Fix::ZERO;
    }

    let mut sum = Vec3Fix::ZERO;
    for p in points {
        sum = sum + *p;
    }

    let n = Fix128::from_int(points.len() as i64);
    Vec3Fix::new(sum.x / n, sum.y / n, sum.z / n)
}

/// Build a convex hull from a set of points
///
/// Uses an incremental algorithm:
/// 1. Find an initial tetrahedron from 4 non-coplanar points
/// 2. Insert remaining points, expanding the hull as needed
///
/// # Panics
///
/// Panics if fewer than 4 non-degenerate points are provided (i.e., all
/// points are coplanar or coincident).
///
/// # Returns
///
/// A `ConvexHull` containing the vertices of the convex hull.
#[must_use]
pub fn build_convex_hull(points: &[Vec3Fix]) -> ConvexHull {
    assert!(
        points.len() >= 4,
        "ConvexHull requires at least 4 points, got {}",
        points.len()
    );

    // Step 1: Find initial tetrahedron
    let (tet, remaining) = find_initial_tetrahedron(points);

    let mut hull_verts: Vec<Vec3Fix> = Vec::with_capacity(points.len());
    hull_verts.push(tet[0]);
    hull_verts.push(tet[1]);
    hull_verts.push(tet[2]);
    hull_verts.push(tet[3]);

    // Centroid of the initial tetrahedron for orienting normals
    let centroid = compute_centroid(&hull_verts);

    let mut faces: Vec<HullFace> = Vec::with_capacity(64);
    // Build 4 faces of the tetrahedron
    let face_indices: [[usize; 3]; 4] = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]];

    for fi in &face_indices {
        let a = hull_verts[fi[0]];
        let b = hull_verts[fi[1]];
        let c = hull_verts[fi[2]];
        let normal = (b - a).cross(c - a);

        // Orient normal to point away from centroid
        let to_centroid = centroid - a;
        let normal = if normal.dot(to_centroid) > Fix128::ZERO {
            -normal
        } else {
            normal
        };

        faces.push(HullFace {
            indices: *fi,
            normal,
        });
    }

    // Step 2: Insert remaining points
    for &point in &remaining {
        insert_point(&mut hull_verts, &mut faces, point);
    }

    // Collect unique hull vertices from face indices
    let mut used = Vec::new();
    for face in &faces {
        for &idx in &face.indices {
            if !used.contains(&idx) {
                used.push(idx);
            }
        }
    }
    used.sort_unstable();

    let result_verts: Vec<Vec3Fix> = used.iter().map(|&i| hull_verts[i]).collect();

    // If construction resulted in fewer than 4 unique vertices, the input
    // is degenerate. Return whatever we have (ConvexHull::new asserts >= 1).
    ConvexHull::new(result_verts)
}

/// Find 4 non-coplanar points to form an initial tetrahedron
///
/// Returns `(tetrahedron_points, remaining_points)`.
fn find_initial_tetrahedron(points: &[Vec3Fix]) -> ([Vec3Fix; 4], Vec<Vec3Fix>) {
    let n = points.len();

    // Find two points that are furthest apart (approximate)
    let mut i0 = 0usize;
    let mut i1 = 1usize;
    let mut max_dist = Fix128::ZERO;

    for i in 0..n {
        for j in (i + 1)..n {
            let d = (points[i] - points[j]).length();
            if d > max_dist {
                max_dist = d;
                i0 = i;
                i1 = j;
            }
        }
    }

    // Find the point furthest from the line (i0, i1)
    let line_dir = points[i1] - points[i0];
    let mut i2 = 0usize;
    let mut max_cross_len = Fix128::ZERO;

    for (i, p) in points.iter().enumerate() {
        if i == i0 || i == i1 {
            continue;
        }
        let v = *p - points[i0];
        let cross = line_dir.cross(v);
        let cl = cross.length();
        if cl > max_cross_len {
            max_cross_len = cl;
            i2 = i;
        }
    }

    // Find the point furthest from the plane (i0, i1, i2)
    let tri_normal = (points[i1] - points[i0]).cross(points[i2] - points[i0]);
    let mut i3 = 0usize;
    let mut max_plane_dist = Fix128::ZERO;

    for (i, p) in points.iter().enumerate() {
        if i == i0 || i == i1 || i == i2 {
            continue;
        }
        let v = *p - points[i0];
        let d = v.dot(tri_normal).abs();
        if d > max_plane_dist {
            max_plane_dist = d;
            i3 = i;
        }
    }

    assert!(
        !max_plane_dist.is_zero(),
        "All points are coplanar — cannot build 3D convex hull"
    );

    let tet = [points[i0], points[i1], points[i2], points[i3]];

    let mut remaining = Vec::with_capacity(n.saturating_sub(4));
    for (i, p) in points.iter().enumerate() {
        if i != i0 && i != i1 && i != i2 && i != i3 {
            remaining.push(*p);
        }
    }

    (tet, remaining)
}

/// Insert a point into the convex hull, expanding it if necessary
fn insert_point(verts: &mut Vec<Vec3Fix>, faces: &mut Vec<HullFace>, point: Vec3Fix) {
    // Find all faces visible from the point
    let mut visible = Vec::new();
    for (i, face) in faces.iter().enumerate() {
        let a = verts[face.indices[0]];
        let v = point - a;
        if face.normal.dot(v) > Fix128::ZERO {
            visible.push(i);
        }
    }

    if visible.is_empty() {
        // Point is inside the hull — nothing to do
        return;
    }

    // Collect horizon edges (edges shared by exactly one visible face)
    let mut horizon_edges: Vec<(usize, usize)> = Vec::new();

    for &fi in &visible {
        let face = faces[fi];
        for edge_idx in 0..3 {
            let e0 = face.indices[edge_idx];
            let e1 = face.indices[(edge_idx + 1) % 3];

            // Check if the reversed edge exists in another visible face
            let reverse_exists = visible.iter().any(|&other_fi| {
                if other_fi == fi {
                    return false;
                }
                let other = faces[other_fi];
                for oe in 0..3 {
                    if other.indices[oe] == e1 && other.indices[(oe + 1) % 3] == e0 {
                        return true;
                    }
                }
                false
            });

            if !reverse_exists {
                horizon_edges.push((e0, e1));
            }
        }
    }

    // Remove visible faces (in reverse order to preserve indices)
    visible.sort_unstable();
    for &fi in visible.iter().rev() {
        faces.swap_remove(fi);
    }

    // Add the new point
    let new_idx = verts.len();
    verts.push(point);

    // Create new faces from horizon edges to the new point
    let hull_centroid = compute_centroid(verts);

    for &(e0, e1) in &horizon_edges {
        let a = verts[e0];
        let b = verts[e1];
        let c = point;
        let normal = (b - a).cross(c - a);

        // Orient normal away from centroid
        let to_centroid = hull_centroid - a;
        let normal = if normal.dot(to_centroid) > Fix128::ZERO {
            -normal
        } else {
            normal
        };

        faces.push(HullFace {
            indices: [e0, e1, new_idx],
            normal,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid_basic() {
        let points = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(4, 0, 0),
            Vec3Fix::from_int(0, 4, 0),
            Vec3Fix::from_int(0, 0, 4),
        ];
        let c = compute_centroid(&points);
        assert_eq!(c.x.hi, 1);
        assert_eq!(c.y.hi, 1);
        assert_eq!(c.z.hi, 1);
    }

    #[test]
    fn test_compute_centroid_empty() {
        let c = compute_centroid(&[]);
        assert!(c.x.is_zero());
        assert!(c.y.is_zero());
        assert!(c.z.is_zero());
    }

    #[test]
    fn test_build_convex_hull_tetrahedron() {
        let points = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(10, 0, 0),
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 0, 10),
        ];
        let hull = build_convex_hull(&points);
        assert_eq!(hull.vertices.len(), 4);
    }

    #[test]
    fn test_build_convex_hull_cube() {
        // 8 corners of a cube
        let points = [
            Vec3Fix::from_int(-1, -1, -1),
            Vec3Fix::from_int(1, -1, -1),
            Vec3Fix::from_int(-1, 1, -1),
            Vec3Fix::from_int(1, 1, -1),
            Vec3Fix::from_int(-1, -1, 1),
            Vec3Fix::from_int(1, -1, 1),
            Vec3Fix::from_int(-1, 1, 1),
            Vec3Fix::from_int(1, 1, 1),
        ];
        let hull = build_convex_hull(&points);
        // All 8 corners are on the hull
        assert_eq!(hull.vertices.len(), 8);
    }

    #[test]
    fn test_build_convex_hull_with_interior_point() {
        // Tetrahedron + interior point
        let points = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(10, 0, 0),
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::from_int(0, 0, 10),
            Vec3Fix::from_int(1, 1, 1), // interior
        ];
        let hull = build_convex_hull(&points);
        // Interior point should not appear on hull
        assert_eq!(hull.vertices.len(), 4);
    }

    #[test]
    fn test_convex_hull_support() {
        use crate::collider::Support;
        let points = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(5, 0, 0),
            Vec3Fix::from_int(0, 5, 0),
            Vec3Fix::from_int(0, 0, 5),
        ];
        let hull = build_convex_hull(&points);
        let s = hull.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 5);
    }

    #[test]
    fn test_convex_hull_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let points = [
            Vec3Fix::from_int(-2, -2, -2),
            Vec3Fix::from_int(2, -2, -2),
            Vec3Fix::from_int(0, 2, -2),
            Vec3Fix::from_int(0, 0, 2),
        ];
        let hull = build_convex_hull(&points);
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let result = gjk(&hull, &sphere);
        assert!(
            result.colliding,
            "Hull containing origin should collide with sphere at origin"
        );
    }

    #[test]
    fn test_convex_hull_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let points = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(0, 0, 1),
        ];
        let hull = build_convex_hull(&points);
        let sphere = Sphere::new(Vec3Fix::from_int(20, 20, 20), Fix128::ONE);
        let result = gjk(&hull, &sphere);
        assert!(!result.colliding, "Far sphere should not collide with hull");
    }

    #[test]
    fn test_centroid_single_point() {
        let points = [Vec3Fix::from_int(7, 3, 9)];
        let c = compute_centroid(&points);
        assert_eq!(c.x.hi, 7);
        assert_eq!(c.y.hi, 3);
        assert_eq!(c.z.hi, 9);
    }

    #[test]
    fn test_build_convex_hull_larger() {
        // Cube + additional points outside
        let points = [
            Vec3Fix::from_int(-1, -1, -1),
            Vec3Fix::from_int(1, -1, -1),
            Vec3Fix::from_int(-1, 1, -1),
            Vec3Fix::from_int(1, 1, -1),
            Vec3Fix::from_int(-1, -1, 1),
            Vec3Fix::from_int(1, -1, 1),
            Vec3Fix::from_int(-1, 1, 1),
            Vec3Fix::from_int(1, 1, 1),
            Vec3Fix::from_int(3, 0, 0), // outside along +X
        ];
        let hull = build_convex_hull(&points);
        // Should have 9 vertices (all are on the hull)
        assert!(
            hull.vertices.len() >= 8,
            "Hull should have at least 8 vertices, got {}",
            hull.vertices.len()
        );
    }
}
