//! Collision Mesh Generation from Signed Distance Fields
//!
//! Generates triangle meshes from implicit surfaces by marching tetrahedra. The
//! resulting meshes are closed, consistently oriented (normals out of the solid)
//! and have shared vertices, so they can serve as static colliders
//! ([`CollisionMesh::to_static_collider`]) or for debug visualization.
//!
//! # Features
//!
//! - Marching-tetrahedra surface extraction from an SDF
//! - Edge-collapse simplification that keeps a closed surface closed
//! - AABB computation for collision meshes
//!
//! All geometry is computed in deterministic 128-bit fixed-point arithmetic.

use crate::collider::AABB;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap;
#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Reverse;
#[cfg(feature = "std")]
use std::collections::BTreeMap;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;

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

impl CollisionMesh {
    /// This mesh as an immovable collision surface for a
    /// [`PhysicsWorld`](crate::solver::PhysicsWorld): a
    /// [`StaticCollider::TriMesh`](crate::static_collider::StaticCollider) of its
    /// triangles (see [`PhysicsWorld::add_static_collider`](crate::solver::PhysicsWorld::add_static_collider)).
    #[must_use]
    pub fn to_static_collider(&self) -> crate::static_collider::StaticCollider {
        let indices: Vec<u32> = self
            .triangles
            .iter()
            .flat_map(|t| t.iter().map(|&i| i as u32))
            .collect();
        crate::static_collider::StaticCollider::TriMesh(crate::trimesh::TriMesh::from_indexed(
            &self.vertices,
            &indices,
        ))
    }
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
// Mesh Generation
// ============================================================================

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

/// The six tetrahedra a cell is cut into, all sharing the diagonal from corner 0
/// to corner 6 (each is a path `0 → axis → axis → 6`). Every cell is cut the same
/// way, so the diagonals of the faces two cells share agree and the surface is
/// closed across cell boundaries.
const TETRAHEDRA: [[usize; 4]; 6] = [
    [0, 1, 2, 6],
    [0, 1, 5, 6],
    [0, 3, 2, 6],
    [0, 3, 7, 6],
    [0, 4, 5, 6],
    [0, 4, 7, 6],
];

/// A lattice corner: its index in the grid, its SDF value, its position.
#[derive(Clone, Copy)]
struct Corner {
    index: usize,
    value: Fix128,
    position: Vec3Fix,
}

/// Generate a collision mesh from a signed distance function.
///
/// Samples the SDF on a regular grid and extracts the zero-isosurface by
/// **marching tetrahedra**: every cell is cut into six tetrahedra and each is
/// triangulated by which of its corners are inside (`sdf < 0`). The result is a
/// closed, consistently oriented surface:
///
/// - a surface vertex lies on a lattice edge and is **shared** by every triangle that
///   uses that edge (not repeated per cell), so edges are shared by exactly two
///   triangles and a ball's mesh has `V − E + F = 2`;
/// - every triangle is wound so its normal points out of the solid (towards
///   positive SDF), so the volume the triangles enclose is positive.
///
/// A vertex is placed by linear interpolation of the SDF along its edge, within
/// `ℓ²·κ/8` of the true surface for edge length `ℓ` and SDF curvature `κ`. The
/// mesh covers only the solid inside the bounds: a surface cut by the bounds is not
/// closed there. An SDF that is zero at a lattice point counts as outside.
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
    let position = |ix: usize, iy: usize, iz: usize| {
        Vec3Fix::new(
            min.x + dx * Fix128::from_int(ix as i64),
            min.y + dy * Fix128::from_int(iy as i64),
            min.z + dz * Fix128::from_int(iz as i64),
        )
    };
    let mut values = vec![Fix128::ZERO; gx * gy * gz];
    for iz in 0..gz {
        for iy in 0..gy {
            for ix in 0..gx {
                values[iz * gy * gx + iy * gx + ix] = sdf(position(ix, iy, iz));
            }
        }
    }

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    // The surface vertex on each lattice edge, keyed by its two lattice indices in
    // increasing order (a `BTreeMap`: the vertex order must not depend on a hasher).
    let mut on_edge: BTreeMap<(usize, usize), usize> = BTreeMap::new();

    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                let mut corners = [Corner {
                    index: 0,
                    value: Fix128::ZERO,
                    position: Vec3Fix::ZERO,
                }; 8];
                for (ci, &(ox, oy, oz)) in CORNER_OFFSETS.iter().enumerate() {
                    let (cx, cy, cz) = (ix + ox, iy + oy, iz + oz);
                    let index = cz * gy * gx + cy * gx + cx;
                    corners[ci] = Corner {
                        index,
                        value: values[index],
                        position: position(cx, cy, cz),
                    };
                }
                for tet in &TETRAHEDRA {
                    let tc = [
                        corners[tet[0]],
                        corners[tet[1]],
                        corners[tet[2]],
                        corners[tet[3]],
                    ];
                    triangulate_tetrahedron(&tc, &mut on_edge, &mut vertices, &mut triangles);
                }
            }
        }
    }

    CollisionMesh {
        vertices,
        triangles,
    }
}

/// The surface vertex on the lattice edge between two corners with opposite signs,
/// created on first use and shared afterwards.
fn edge_vertex(
    a: &Corner,
    b: &Corner,
    on_edge: &mut BTreeMap<(usize, usize), usize>,
    vertices: &mut Vec<Vec3Fix>,
) -> usize {
    // Always interpolate from the lower lattice index, so the position does not
    // depend on which cell or tetrahedron asked first.
    let (lo, hi) = if a.index < b.index { (a, b) } else { (b, a) };
    *on_edge.entry((lo.index, hi.index)).or_insert_with(|| {
        // Opposite signs (one value negative, one not): the denominator is nonzero.
        let t = (Fix128::ZERO - lo.value) / (hi.value - lo.value);
        vertices.push(Vec3Fix::new(
            lo.position.x + (hi.position.x - lo.position.x) * t,
            lo.position.y + (hi.position.y - lo.position.y) * t,
            lo.position.z + (hi.position.z - lo.position.z) * t,
        ));
        vertices.len() - 1
    })
}

/// Push the triangles of the surface through one tetrahedron: none when its corners
/// are all inside or all outside, one when one corner is alone on its side, two (a
/// quad) when they split two and two.
fn triangulate_tetrahedron(
    tet: &[Corner; 4],
    on_edge: &mut BTreeMap<(usize, usize), usize>,
    vertices: &mut Vec<Vec3Fix>,
    triangles: &mut Vec<[usize; 3]>,
) {
    let inside: Vec<usize> = (0..4).filter(|&i| tet[i].value.is_negative()).collect();
    let outside: Vec<usize> = (0..4).filter(|&i| !tet[i].value.is_negative()).collect();
    if inside.is_empty() || outside.is_empty() {
        return;
    }
    let mean = |ids: &[usize]| {
        let sum = ids
            .iter()
            .fold(Vec3Fix::ZERO, |acc, &i| acc + tet[i].position);
        sum / Fix128::from_int(ids.len() as i64)
    };
    // From the inside of the solid towards its outside: the way a normal must point.
    let outward = mean(&outside) - mean(&inside);

    let mut push = |a: usize, b: usize, c: usize, vertices: &Vec<Vec3Fix>| {
        let n = (vertices[b] - vertices[a]).cross(vertices[c] - vertices[a]);
        if n.dot(outward) < Fix128::ZERO {
            triangles.push([a, c, b]);
        } else {
            triangles.push([a, b, c]);
        }
    };

    match (inside.len(), outside.len()) {
        (1, 3) | (3, 1) => {
            // One corner alone: the three edges from it to the other side.
            let (lone, others) = if inside.len() == 1 {
                (inside[0], &outside)
            } else {
                (outside[0], &inside)
            };
            let e: Vec<usize> = others
                .iter()
                .map(|&o| edge_vertex(&tet[lone], &tet[o], on_edge, vertices))
                .collect();
            push(e[0], e[1], e[2], vertices);
        }
        _ => {
            // Two and two: the four edges between the sides, around a quad.
            let (i0, i1) = (inside[0], inside[1]);
            let (o0, o1) = (outside[0], outside[1]);
            let e0 = edge_vertex(&tet[i0], &tet[o0], on_edge, vertices);
            let e1 = edge_vertex(&tet[i0], &tet[o1], on_edge, vertices);
            let e2 = edge_vertex(&tet[i1], &tet[o1], on_edge, vertices);
            let e3 = edge_vertex(&tet[i1], &tet[o0], on_edge, vertices);
            push(e0, e1, e2, vertices);
            push(e0, e2, e3, vertices);
        }
    }
}

// ============================================================================
// Simplification
// ============================================================================

/// The vertices next to a vertex, sorted.
fn neighbours(triangles: &[[usize; 3]], incident: &[usize], v: usize) -> Vec<usize> {
    let mut out: Vec<usize> = incident
        .iter()
        .flat_map(|&t| triangles[t])
        .filter(|&w| w != v)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The unnormalised normal of a triangle with the given corner positions.
fn face_normal(a: Vec3Fix, b: Vec3Fix, c: Vec3Fix) -> Vec3Fix {
    (b - a).cross(c - a)
}

/// Where the edge `a`–`b` collapses to, or `None` when it cannot be collapsed
/// without breaking the surface. `inc_a` / `inc_b` are the live triangles at `a` and
/// `b`. An interior edge has two triangles and `a` and `b` have no neighbour in
/// common beyond the two apexes of those triangles (the *link condition*, which
/// keeps the surface a manifold); no triangle that survives turns over or goes flat.
/// The new position is the midpoint, except next to the border of an open mesh: an
/// edge between two border vertices is never collapsed (that would pinch the
/// border), and one with a single border vertex collapses onto it, so the border
/// does not move.
fn collapse_target(
    vertices: &[Vec3Fix],
    triangles: &[[usize; 3]],
    on_border: &[bool],
    (a, inc_a): (usize, &[usize]),
    (b, inc_b): (usize, &[usize]),
) -> Option<Vec3Fix> {
    if on_border[a] && on_border[b] {
        return None;
    }
    let shared: Vec<usize> = inc_a
        .iter()
        .copied()
        .filter(|t| inc_b.contains(t))
        .collect();
    if shared.len() != 2 {
        return None;
    }
    let (na, nb) = (
        neighbours(triangles, inc_a, a),
        neighbours(triangles, inc_b, b),
    );
    // Neither list contains its own vertex, so `a` and `b` are not in the
    // intersection: what the two share must be exactly the two apexes.
    let common = na.iter().filter(|v| nb.contains(v)).count();
    if common != 2 {
        return None;
    }
    let target = if on_border[a] {
        vertices[a]
    } else if on_border[b] {
        vertices[b]
    } else {
        Vec3Fix::new(
            (vertices[a].x + vertices[b].x).half(),
            (vertices[a].y + vertices[b].y).half(),
            (vertices[a].z + vertices[b].z).half(),
        )
    };
    let at = |v: usize| {
        if v == a || v == b {
            target
        } else {
            vertices[v]
        }
    };
    for &t in inc_a.iter().chain(inc_b) {
        if shared.contains(&t) {
            continue;
        }
        let tri = triangles[t];
        let before = face_normal(vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]);
        let after = face_normal(at(tri[0]), at(tri[1]), at(tri[2]));
        if before.dot(after) <= Fix128::ZERO {
            return None;
        }
    }
    Some(target)
}

/// The vertices on the border of an open mesh: those on an edge that belongs to a
/// single triangle (none for a closed mesh).
fn border_vertices(vertex_count: usize, triangles: &[[usize; 3]]) -> Vec<bool> {
    let mut uses: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for tri in triangles {
        for k in 0..3 {
            let (i0, i1) = (tri[k], tri[(k + 1) % 3]);
            *uses.entry((i0.min(i1), i0.max(i1))).or_default() += 1;
        }
    }
    let mut border = vec![false; vertex_count];
    for (&(a, b), &n) in &uses {
        if n == 1 {
            border[a] = true;
            border[b] = true;
        }
    }
    border
}

/// A candidate edge of the simplification queue: squared length, its two vertices,
/// and the versions of those vertices when it was queued.
type EdgeEntry = (Fix128, usize, usize, usize, usize);

/// Simplify a collision mesh by reducing the triangle count.
///
/// Collapses the shortest collapsible edge to its midpoint, repeatedly, until the
/// triangle count is at most `target_triangles` or no edge can be collapsed. An edge
/// is collapsible when doing so keeps the surface manifold (the *link condition*)
/// and no remaining triangle turns over; so a closed, outward oriented mesh stays
/// closed and outward, with the same Euler characteristic. Each collapse of an
/// interior edge removes two triangles, so the result has `target_triangles` or one
/// fewer, unless no edge can be collapsed; a closed mesh is never reduced below four
/// triangles (a tetrahedron). The border of an open mesh does not move:
/// edges between two border vertices are kept, and an edge with one border vertex
/// collapses onto it. Vertices the collapses leave unused are removed and the
/// indices renumbered.
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
    let mut alive_tri = vec![true; triangles.len()];
    let mut alive_count = triangles.len();
    let mut alive_vertex = vec![true; vertices.len()];
    let mut on_border = border_vertices(vertices.len(), &triangles);

    // Triangles at each vertex (dead ones are filtered when read).
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); vertices.len()];
    for (ti, tri) in triangles.iter().enumerate() {
        for &v in tri {
            incident[v].push(ti);
        }
    }
    // Candidate edges, shortest first (ties by vertex index: deterministic). An
    // entry is valid only while both its vertices are unchanged since it was pushed
    // (`version`); a collapse changes the vertices around it and pushes fresh entries.
    let mut version = vec![0usize; vertices.len()];
    let mut queue: BinaryHeap<Reverse<EdgeEntry>> = BinaryHeap::new();
    let push_edges =
        |queue: &mut BinaryHeap<_>, vertices: &[Vec3Fix], version: &[usize], tri: [usize; 3]| {
            for k in 0..3 {
                let (i0, i1) = (tri[k], tri[(k + 1) % 3]);
                let (a, b) = (i0.min(i1), i0.max(i1));
                let d = vertices[b] - vertices[a];
                queue.push(Reverse((d.dot(d), a, b, version[a], version[b])));
            }
        };
    for tri in &triangles {
        push_edges(&mut queue, &vertices, &version, *tri);
    }

    let live = |incident: &[usize], alive_tri: &[bool]| -> Vec<usize> {
        incident.iter().copied().filter(|&t| alive_tri[t]).collect()
    };

    // A closed surface needs four triangles to enclose anything: two would be a flat
    // pillow of zero volume.
    let floor = if on_border.iter().any(|&b| b) { 0 } else { 4 };
    while alive_count > target_triangles.max(floor) {
        let Some(Reverse((_, a, b, va, vb))) = queue.pop() else {
            break;
        };
        if !alive_vertex[a] || !alive_vertex[b] || version[a] != va || version[b] != vb {
            continue;
        }
        let inc_a = live(&incident[a], &alive_tri);
        let inc_b = live(&incident[b], &alive_tri);
        let Some(target) =
            collapse_target(&vertices, &triangles, &on_border, (a, &inc_a), (b, &inc_b))
        else {
            continue;
        };

        // Collapse `b` into `a`: the two triangles on the edge die, the others at `b`
        // move to `a`.
        vertices[a] = target;
        alive_vertex[b] = false;
        // The survivor is on the border if either end was (it sits where the border
        // vertex was).
        on_border[a] = on_border[a] || on_border[b];
        for &t in &inc_b {
            if triangles[t].contains(&a) {
                alive_tri[t] = false;
                alive_count -= 1;
            } else {
                for v in &mut triangles[t] {
                    if *v == b {
                        *v = a;
                    }
                }
                incident[a].push(t);
            }
        }
        // Every vertex around `a` changed: bump it and queue the edges again.
        let around = live(&incident[a], &alive_tri);
        for &t in &around {
            for &v in &triangles[t] {
                version[v] += 1;
            }
        }
        for &t in &around {
            push_edges(&mut queue, &vertices, &version, triangles[t]);
        }
    }

    // Keep the live triangles, drop the vertices nothing uses and renumber.
    let mut renumber = vec![usize::MAX; vertices.len()];
    let mut kept = Vec::new();
    let mut out = Vec::new();
    for (ti, tri) in triangles.iter().enumerate() {
        if !alive_tri[ti] {
            continue;
        }
        let mut t = *tri;
        for v in &mut t {
            if renumber[*v] == usize::MAX {
                renumber[*v] = kept.len();
                kept.push(vertices[*v]);
            }
            *v = renumber[*v];
        }
        out.push(t);
    }

    CollisionMesh {
        vertices: kept,
        triangles: out,
    }
}

/// Compute the axis-aligned bounding box of a collision mesh: of the vertices its
/// triangles use, or of all its vertices when it has no triangles (a point set), or
/// the zero box for an empty mesh.
#[must_use]
pub fn compute_mesh_aabb(mesh: &CollisionMesh) -> AABB {
    let mut points: Vec<Vec3Fix> = if mesh.triangles.is_empty() {
        mesh.vertices.clone()
    } else {
        mesh.triangles
            .iter()
            .flat_map(|t| t.iter().map(|&i| mesh.vertices[i]))
            .collect()
    };
    let Some(first) = points.pop() else {
        return AABB {
            min: Vec3Fix::ZERO,
            max: Vec3Fix::ZERO,
        };
    };
    let (mut min, mut max) = (first, first);
    for v in &points {
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
