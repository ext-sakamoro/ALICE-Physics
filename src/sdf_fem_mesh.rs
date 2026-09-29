//! Cartesian tetrahedral mesh generation from a signed distance field.
//!
//! Companion of [`crate::deformable`], [`crate::linear_elastic_fem`] and
//! [`crate::sdf_collider`]. [`generate`] dices every Cartesian cube that lies
//! fully inside the SDF into five tetrahedra; [`generate_marching_tets`] also
//! clips the cubes the surface crosses.
//!
//! # Conformity
//!
//! Both generators produce a **conforming** mesh: two tetrahedra meet along a
//! shared face, edge or vertex and nothing else. That is a precondition for the
//! FEM — a mismatched face makes the displacement field discontinuous across it
//! and there is no convergence guarantee — and it is not free. Three things
//! have to line up, each of which was wrong here until 2026-09-29:
//!
//! - **The cube dicing alternates** (`CUBE_FIVE_TETS`, selected by
//!   `cell_parity`). A cube has two 5-tet decompositions and a single one used
//!   everywhere puts opposite diagonals on the two faces it shares along each
//!   axis.
//! - **Vertices are interned by topology**, not by position: a lattice corner
//!   by its cell coordinate, a zero crossing by the pair of corners whose
//!   segment carries it. No coordinate comparison and so no threshold to tune.
//! - **Quadrilateral faces of a clipped wedge take the diagonal through their
//!   smallest vertex index** (`prism_to_tets`), so the two sub-tetrahedra that
//!   meet along one make the same choice.
//!
//! `tests/mesh_conformity.rs` counts this directly, by face census. **Do not
//! use the FEM patch test for it**: two triangulations of a square interpolate
//! a linear field identically, so a patch test comes back exact on a
//! non-conforming mesh (measured: 3.6e-15 MPa while 576 of 768 singly-used
//! faces were interior).
//!
//! # Limitations
//!
//! - [`generate`] emits only cubes with all eight corners inside, so it meshes
//!   strictly less than the shape: a 20 mm box at 4 mm cells yields the inner
//!   16 mm. Use [`generate_marching_tets`] when the surface matters.
//! - Edge-based refinement ([`SdfTetMesh::refine_by_max_edge_length`]) is not
//!   Delaunay refinement; aspect ratio can drift.
//! - The generated mesh is intended for downstream FEM callers; it is not tuned
//!   for rendering.

use std::collections::HashMap;

use crate::sdf_collider::SdfField;

/// A single tetrahedron given by four vertex indices into
/// [`SdfTetMesh::vertices`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tetrahedron {
    /// Vertex indices (order defines the outward orientation).
    pub vertices: [u32; 4],
}

/// Output mesh.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SdfTetMesh {
    /// World-space vertex positions.
    pub vertices: Vec<[f32; 3]>,
    /// Tetrahedra emitted by the generator.
    pub tets: Vec<Tetrahedron>,
}

impl SdfTetMesh {
    /// Number of vertices.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of tetrahedra.
    #[must_use]
    pub fn tet_count(&self) -> usize {
        self.tets.len()
    }

    /// Iteratively refine the mesh by splitting every tetrahedron whose
    /// longest edge exceeds `max_edge_length`.
    ///
    /// Each qualifying tetrahedron is decomposed into two new tets by
    /// inserting a midpoint on the longest edge and reconnecting the
    /// remaining vertices. The pass repeats until no tet exceeds the
    /// threshold or `max_passes` is reached (whichever comes first).
    ///
    /// **This is edge-based refinement, not Delaunay refinement** —
    /// aspect ratio can drift as edges shorten unevenly. Callers who
    /// need Delaunay-quality tets should postprocess with an external
    /// remesher; this MVP is aimed at bounded-edge FEM assembly.
    ///
    /// Returns the number of refinement passes actually executed.
    ///
    /// # Panics
    ///
    /// Panics if `max_edge_length <= 0.0`.
    pub fn refine_by_max_edge_length(&mut self, max_edge_length: f32, max_passes: u32) -> u32 {
        assert!(max_edge_length > 0.0, "max_edge_length must be positive");
        let mut passes = 0_u32;
        for _ in 0..max_passes {
            passes += 1;
            let mut changed = false;
            let mut new_tets: Vec<Tetrahedron> = Vec::with_capacity(self.tets.len() * 2);
            let mut edge_midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();
            let tet_snapshot = std::mem::take(&mut self.tets);
            for tet in &tet_snapshot {
                let vs = tet.vertices;
                // Find the longest edge (pair of vertex indices).
                let edges: [(u32, u32); 6] = [
                    (vs[0], vs[1]),
                    (vs[0], vs[2]),
                    (vs[0], vs[3]),
                    (vs[1], vs[2]),
                    (vs[1], vs[3]),
                    (vs[2], vs[3]),
                ];
                let mut best_edge = 0_usize;
                let mut best_len_sq = 0.0_f32;
                for (k, (a, b)) in edges.iter().enumerate() {
                    let pa = self.vertices[*a as usize];
                    let pb = self.vertices[*b as usize];
                    let dx = pa[0] - pb[0];
                    let dy = pa[1] - pb[1];
                    let dz = pa[2] - pb[2];
                    let len_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
                    if len_sq > best_len_sq {
                        best_len_sq = len_sq;
                        best_edge = k;
                    }
                }
                let longest_len = best_len_sq.sqrt();
                if longest_len <= max_edge_length {
                    new_tets.push(*tet);
                    continue;
                }
                changed = true;
                let (a, b) = edges[best_edge];
                // Deduplicated midpoint insertion.
                let key = (a.min(b), a.max(b));
                let mid_index = if let Some(&idx) = edge_midpoint_cache.get(&key) {
                    idx
                } else {
                    let pa = self.vertices[a as usize];
                    let pb = self.vertices[b as usize];
                    let mid = [
                        0.5 * (pa[0] + pb[0]),
                        0.5 * (pa[1] + pb[1]),
                        0.5 * (pa[2] + pb[2]),
                    ];
                    let idx = self.vertices.len() as u32;
                    self.vertices.push(mid);
                    edge_midpoint_cache.insert(key, idx);
                    idx
                };
                // Split the tet by replacing the longest edge's
                // endpoints with the midpoint on each of two child tets.
                // For edge (a, b), the remaining two vertices are `c` and `d`.
                let (c, d) = split_edge_remaining(vs, best_edge);
                new_tets.push(Tetrahedron {
                    vertices: [a, mid_index, c, d],
                });
                new_tets.push(Tetrahedron {
                    vertices: [mid_index, b, c, d],
                });
            }
            self.tets = new_tets;
            if !changed {
                break;
            }
        }
        passes
    }

    /// Maximum edge length over the mesh (`0.0` on an empty mesh).
    #[must_use]
    pub fn max_edge_length(&self) -> f32 {
        let mut best = 0.0_f32;
        for tet in &self.tets {
            let vs = tet.vertices;
            let edges: [(u32, u32); 6] = [
                (vs[0], vs[1]),
                (vs[0], vs[2]),
                (vs[0], vs[3]),
                (vs[1], vs[2]),
                (vs[1], vs[3]),
                (vs[2], vs[3]),
            ];
            for (a, b) in edges {
                let pa = self.vertices[a as usize];
                let pb = self.vertices[b as usize];
                let dx = pa[0] - pb[0];
                let dy = pa[1] - pb[1];
                let dz = pa[2] - pb[2];
                let len = dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt();
                if len > best {
                    best = len;
                }
            }
        }
        best
    }
}

/// For a tetrahedron with vertex list `[v0, v1, v2, v3]` and an
/// enumeration-index `best_edge ∈ 0..6` naming which edge was picked,
/// return the two remaining vertex indices.
fn split_edge_remaining(vs: [u32; 4], best_edge: usize) -> (u32, u32) {
    match best_edge {
        0 => (vs[2], vs[3]), // (v0, v1)
        1 => (vs[1], vs[3]), // (v0, v2)
        2 => (vs[1], vs[2]), // (v0, v3)
        3 => (vs[0], vs[3]), // (v1, v2)
        4 => (vs[0], vs[2]), // (v1, v3)
        5 => (vs[0], vs[1]), // (v2, v3)
        _ => unreachable!("best_edge out of range"),
    }
}

/// Generate a tet mesh by walking a Cartesian grid over the AABB
/// `[min .. max]` at spacing `cell` and dicing every fully-interior
/// cube into five tetrahedra.
///
/// # Panics
///
/// Panics if `cell <= 0`.
#[must_use]
pub fn generate<F: SdfField + ?Sized>(
    sdf: &F,
    min: [f32; 3],
    max: [f32; 3],
    cell: f32,
) -> SdfTetMesh {
    assert!(cell > 0.0, "cell must be positive");
    let mut mesh = SdfTetMesh::default();
    let mut vertex_index: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let nx = ((max[0] - min[0]) / cell).max(1.0) as i32;
    let ny = ((max[1] - min[1]) / cell).max(1.0) as i32;
    let nz = ((max[2] - min[2]) / cell).max(1.0) as i32;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let corner_ids: [(i32, i32, i32); 8] = [
                    (ix, iy, iz),
                    (ix + 1, iy, iz),
                    (ix + 1, iy + 1, iz),
                    (ix, iy + 1, iz),
                    (ix, iy, iz + 1),
                    (ix + 1, iy, iz + 1),
                    (ix + 1, iy + 1, iz + 1),
                    (ix, iy + 1, iz + 1),
                ];
                let corner_positions = corner_ids.map(|(cx, cy, cz)| {
                    [
                        min[0] + (cx as f32) * cell,
                        min[1] + (cy as f32) * cell,
                        min[2] + (cz as f32) * cell,
                    ]
                });
                // Skip cubes that are not fully interior.
                let all_inside = corner_positions
                    .iter()
                    .all(|p| sdf.distance(p[0], p[1], p[2]) <= 0.0);
                if !all_inside {
                    continue;
                }
                let mut cube_verts = [0_u32; 8];
                for (slot, id) in corner_ids.iter().enumerate() {
                    let entry = vertex_index.entry(*id);
                    let next_index = mesh.vertices.len() as u32;
                    let idx = *entry.or_insert_with(|| {
                        mesh.vertices.push(corner_positions[slot]);
                        next_index
                    });
                    cube_verts[slot] = idx;
                }
                for tet in cube_to_five_tets(cube_verts, cell_parity(ix, iy, iz)) {
                    mesh.tets.push(tet);
                }
            }
        }
    }
    mesh
}

/// Generate a **surface-conforming** tet mesh via Marching Tetrahedra.
///
/// Each Cartesian cube is diced into five reference tetrahedra; for each
/// sub-tetrahedron the SDF sign at its four vertices selects an emission
/// pattern from the standard 16-case table (0, 1, 2, 3, or 4 inside),
/// clipping the tetrahedron against the zero level set.
///
/// The resulting mesh conforms to the SDF surface rather than skipping
/// crossing cubes entirely (see [`generate`] for the fully-interior
/// variant). Surface vertices are inserted at exact linear
/// interpolations of the SDF along each crossed edge.
///
/// # Panics
///
/// Panics if `cell <= 0`.
#[must_use]
pub fn generate_marching_tets<F: SdfField + ?Sized>(
    sdf: &F,
    min: [f32; 3],
    max: [f32; 3],
    cell: f32,
) -> SdfTetMesh {
    assert!(cell > 0.0, "cell must be positive");
    let mut mesh = SdfTetMesh::default();
    let mut table: HashMap<VertexKey, u32> = HashMap::new();
    let nx = ((max[0] - min[0]) / cell).max(1.0) as i32;
    let ny = ((max[1] - min[1]) / cell).max(1.0) as i32;
    let nz = ((max[2] - min[2]) / cell).max(1.0) as i32;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let corner_positions: [[f32; 3]; 8] = [
                    corner_pos(min, cell, ix, iy, iz),
                    corner_pos(min, cell, ix + 1, iy, iz),
                    corner_pos(min, cell, ix + 1, iy + 1, iz),
                    corner_pos(min, cell, ix, iy + 1, iz),
                    corner_pos(min, cell, ix, iy, iz + 1),
                    corner_pos(min, cell, ix + 1, iy, iz + 1),
                    corner_pos(min, cell, ix + 1, iy + 1, iz + 1),
                    corner_pos(min, cell, ix, iy + 1, iz + 1),
                ];
                let corner_sdf: [f32; 8] = std::array::from_fn(|i| {
                    sdf.distance(
                        corner_positions[i][0],
                        corner_positions[i][1],
                        corner_positions[i][2],
                    )
                });
                // Skip cubes that are entirely outside.
                if corner_sdf.iter().all(|&d| d > 0.0) {
                    continue;
                }
                let corner_ids: [[i32; 3]; 8] = [
                    [ix, iy, iz],
                    [ix + 1, iy, iz],
                    [ix + 1, iy + 1, iz],
                    [ix, iy + 1, iz],
                    [ix, iy, iz + 1],
                    [ix + 1, iy, iz + 1],
                    [ix + 1, iy + 1, iz + 1],
                    [ix, iy + 1, iz + 1],
                ];
                // The base dicing alternates for the same reason `generate`
                // does: a fixed pattern disagrees with the neighbour on every
                // shared face.
                for sub_tet in CUBE_FIVE_TETS[cell_parity(ix, iy, iz)] {
                    let ids = sub_tet.map(|i| corner_ids[i]);
                    let positions = sub_tet.map(|i| corner_positions[i]);
                    let sdf_values = sub_tet.map(|i| corner_sdf[i]);
                    marching_tet_emit(&mut mesh, &mut table, ids, positions, sdf_values);
                }
            }
        }
    }
    mesh
}

fn corner_pos(min: [f32; 3], cell: f32, ix: i32, iy: i32, iz: i32) -> [f32; 3] {
    [
        min[0] + (ix as f32) * cell,
        min[1] + (iy as f32) * cell,
        min[2] + (iz as f32) * cell,
    ]
}

/// Emit clipped tetrahedra into `mesh` for a single sub-tetrahedron
/// against the SDF represented by `sdf_values` at the four vertices.
fn marching_tet_emit(
    mesh: &mut SdfTetMesh,
    table: &mut HashMap<VertexKey, u32>,
    ids: [[i32; 3]; 4],
    v: [[f32; 3]; 4],
    sdf: [f32; 4],
) {
    // Bitmask of inside vertices (bit i = 1 iff sdf[i] < 0).
    let mut mask: u8 = 0;
    for (i, &d) in sdf.iter().enumerate() {
        if d < 0.0 {
            mask |= 1 << i;
        }
    }
    if mask == 0b0000 {
        return;
    }
    let corner = |mesh: &mut SdfTetMesh, table: &mut HashMap<VertexKey, u32>, i: usize| -> u32 {
        intern(mesh, table, VertexKey::Corner(ids[i]), v[i])
    };
    if mask == 0b1111 {
        let c = [
            corner(mesh, table, 0),
            corner(mesh, table, 1),
            corner(mesh, table, 2),
            corner(mesh, table, 3),
        ];
        push_tet(mesh, c);
        return;
    }

    match mask.count_ones() {
        1 => {
            // One vertex inside; one tet from it to three edge crossings.
            let inside = mask.trailing_zeros() as usize;
            let others = other_three(inside);
            let p = corner(mesh, table, inside);
            let e0 = crossing_vertex(mesh, table, ids, v, sdf, inside, others[0]);
            let e1 = crossing_vertex(mesh, table, ids, v, sdf, inside, others[1]);
            let e2 = crossing_vertex(mesh, table, ids, v, sdf, inside, others[2]);
            push_tet(mesh, [p, e0, e1, e2]);
        }
        3 => {
            // Three inside; the complement of case 1, decomposed as a wedge.
            let outside = (!mask & 0b1111).trailing_zeros() as usize;
            let insides = other_three(outside);
            let a = corner(mesh, table, insides[0]);
            let b = corner(mesh, table, insides[1]);
            let c = corner(mesh, table, insides[2]);
            let e0 = crossing_vertex(mesh, table, ids, v, sdf, insides[0], outside);
            let e1 = crossing_vertex(mesh, table, ids, v, sdf, insides[1], outside);
            let e2 = crossing_vertex(mesh, table, ids, v, sdf, insides[2], outside);
            for tet in prism_to_tets([a, b, c], [e0, e1, e2]) {
                push_tet(mesh, tet);
            }
        }
        2 => {
            // Two inside, two outside: the interior polytope is a wedge with
            // six vertices.
            let mut insides = [0_usize; 2];
            let mut outsides = [0_usize; 2];
            let mut ip = 0;
            let mut op = 0;
            for i in 0..4 {
                if mask & (1 << i) != 0 {
                    insides[ip] = i;
                    ip += 1;
                } else {
                    outsides[op] = i;
                    op += 1;
                }
            }
            let a = corner(mesh, table, insides[0]);
            let b = corner(mesh, table, insides[1]);
            let e_a_c = crossing_vertex(mesh, table, ids, v, sdf, insides[0], outsides[0]);
            let e_a_d = crossing_vertex(mesh, table, ids, v, sdf, insides[0], outsides[1]);
            let e_b_c = crossing_vertex(mesh, table, ids, v, sdf, insides[1], outsides[0]);
            let e_b_d = crossing_vertex(mesh, table, ids, v, sdf, insides[1], outsides[1]);
            // the wedge is a prism with triangular ends (a, e_a_c, e_a_d) and
            // (b, e_b_c, e_b_d), paired corner for corner
            for tet in prism_to_tets([a, e_a_c, e_a_d], [b, e_b_c, e_b_d]) {
                push_tet(mesh, tet);
            }
        }
        _ => unreachable!("count 0 and 4 handled above"),
    }
}

fn other_three(exclude: usize) -> [usize; 3] {
    let mut out = [0_usize; 3];
    let mut cursor = 0;
    for i in 0..4 {
        if i != exclude {
            out[cursor] = i;
            cursor += 1;
        }
    }
    out
}

/// Identity of a Marching Tetrahedra vertex.
///
/// Keyed by **topology, not position**. A crossing vertex always lies on the
/// segment between two lattice corners and is placed by interpolating the same
/// two signed distances, so every sub-tetrahedron that meets that segment
/// computes bit-identical coordinates for it. Keying on the corner pair
/// therefore identifies it exactly, with no distance threshold to tune — the
/// question "are these two floats the same point?" never comes up.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VertexKey {
    /// A lattice corner.
    Corner([i32; 3]),
    /// The zero crossing on the segment between two lattice corners, stored
    /// with the lexicographically smaller corner first so both sides of the
    /// segment produce the same key.
    Crossing([i32; 3], [i32; 3]),
}

impl VertexKey {
    #[inline]
    fn crossing(a: [i32; 3], b: [i32; 3]) -> Self {
        if a <= b {
            Self::Crossing(a, b)
        } else {
            Self::Crossing(b, a)
        }
    }
}

/// Return the index of `key`, inserting `position` on first sight.
fn intern(
    mesh: &mut SdfTetMesh,
    table: &mut HashMap<VertexKey, u32>,
    key: VertexKey,
    position: [f32; 3],
) -> u32 {
    *table.entry(key).or_insert_with(|| {
        let index = mesh.vertices.len() as u32;
        mesh.vertices.push(position);
        index
    })
}

/// Index of the zero crossing on sub-tet edge `(a, b)`.
///
/// Snaps to a corner when the crossing lands exactly on one, so a vertex is
/// never emitted twice under two different keys.
fn crossing_vertex(
    mesh: &mut SdfTetMesh,
    table: &mut HashMap<VertexKey, u32>,
    ids: [[i32; 3]; 4],
    v: [[f32; 3]; 4],
    sdf: [f32; 4],
    a: usize,
    b: usize,
) -> u32 {
    let denom = sdf[a] - sdf[b];
    let t = if denom.abs() < 1.0e-9 {
        0.5
    } else {
        sdf[a] / denom
    };
    if t <= 0.0 {
        return intern(mesh, table, VertexKey::Corner(ids[a]), v[a]);
    }
    if t >= 1.0 {
        return intern(mesh, table, VertexKey::Corner(ids[b]), v[b]);
    }
    let position = [
        v[a][0] + t * (v[b][0] - v[a][0]),
        v[a][1] + t * (v[b][1] - v[a][1]),
        v[a][2] + t * (v[b][2] - v[a][2]),
    ];
    intern(mesh, table, VertexKey::crossing(ids[a], ids[b]), position)
}

/// Split a triangular prism into three tetrahedra, choosing every quadrilateral
/// face's diagonal through that face's smallest vertex index.
///
/// `bottom[k]` is paired with `top[k]`; the three quadrilateral faces are
/// `(bottom[k], bottom[k+1], top[k+1], top[k])`.
///
/// The rule matters for conformity, not for quality. Two clipped
/// sub-tetrahedra that meet along such a face see the same four global indices,
/// so picking the diagonal by index makes them agree — while any fixed choice
/// (an apex hard-coded to one corner, say) makes them disagree and leaves the
/// mesh non-conforming, which is the defect this replaced.
///
/// Rotating the smallest index to `bottom[0]` also avoids the prism
/// configuration that has no three-tetrahedron split at all: the two faces
/// meeting at that corner then both take their diagonal through it, so the three
/// diagonals cannot all circulate the same way (Dompierre et al., *How to
/// Subdivide Pyramids, Prisms and Hexahedra into Tetrahedra*).
fn prism_to_tets(bottom: [u32; 3], top: [u32; 3]) -> [[u32; 4]; 3] {
    let mut b = bottom;
    let mut t = top;
    let all = [b[0], b[1], b[2], t[0], t[1], t[2]];
    let mut lowest = 0usize;
    for (i, v) in all.iter().enumerate() {
        if *v < all[lowest] {
            lowest = i;
        }
    }
    if lowest >= 3 {
        core::mem::swap(&mut b, &mut t);
        lowest -= 3;
    }
    let b = [b[lowest], b[(lowest + 1) % 3], b[(lowest + 2) % 3]];
    let t = [t[lowest], t[(lowest + 1) % 3], t[(lowest + 2) % 3]];
    if b[1].min(t[2]) < b[2].min(t[1]) {
        [
            [b[0], b[1], b[2], t[2]],
            [b[0], b[1], t[2], t[1]],
            [b[0], t[1], t[2], t[0]],
        ]
    } else {
        [
            [b[0], b[1], b[2], t[1]],
            [b[0], t[1], b[2], t[2]],
            [b[0], t[1], t[2], t[0]],
        ]
    }
}

/// Push a tetrahedron, dropping it when two of its vertices coincide (which the
/// corner snapping above can produce on a degenerate crossing).
fn push_tet(mesh: &mut SdfTetMesh, vertices: [u32; 4]) {
    for i in 0..4 {
        for j in (i + 1)..4 {
            if vertices[i] == vertices[j] {
                return;
            }
        }
    }
    mesh.tets.push(Tetrahedron { vertices });
}

/// The two 5-tet decompositions of a cube, as indices into a corner array
/// ordered
///
/// ```text
///   4 ---- 5
///  /|      /|
/// 7 ---- 6 |
/// | 0 ---|-1
/// |/     |/
/// 3 ---- 2
/// ```
///
/// A cube has exactly two such decompositions, one per set of alternating
/// corners: `{0,2,5,7}` and `{1,3,4,6}`. Each is four corner tetrahedra plus the
/// regular tetrahedron on the alternating set.
///
/// **They must alternate.** A single decomposition used for every cube puts
/// opposite diagonals on the two faces it shares with its neighbours along each
/// axis, so the triangulations on a shared face disagree and the mesh is
/// non-conforming — the displacement field of a downstream FEM is then
/// discontinuous across that face. Alternating by the parity of the cell index
/// makes every shared face agree: the `{0,2,5,7}` decomposition puts the
/// diagonal of its `+x` face between corners 2 and 5, and the `{1,3,4,6}` one
/// puts its `−x` face diagonal between corners 3 and 4, which are the same two
/// lattice points.
pub(crate) const CUBE_FIVE_TETS: [[[usize; 4]; 5]; 2] = [
    // even parity: interior tet on {0,2,5,7}
    [
        [0, 1, 2, 5],
        [0, 2, 3, 7],
        [0, 4, 5, 7],
        [2, 5, 6, 7],
        [0, 2, 5, 7],
    ],
    // odd parity: mirrored, interior tet on {1,3,4,6}
    [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ],
];

/// Parity of a lattice cell, selecting which row of [`CUBE_FIVE_TETS`] to use.
///
/// `rem_euclid` rather than `& 1` so that negative cell indices alternate the
/// same way as positive ones.
#[inline]
pub(crate) const fn cell_parity(ix: i32, iy: i32, iz: i32) -> usize {
    ((ix + iy + iz).rem_euclid(2)) as usize
}

/// Dice one cube into five tetrahedra, alternating the decomposition so that
/// neighbouring cubes agree on their shared faces.
fn cube_to_five_tets(v: [u32; 8], parity: usize) -> [Tetrahedron; 5] {
    CUBE_FIVE_TETS[parity].map(|t| Tetrahedron {
        vertices: [v[t[0]], v[t[1]], v[t[2]], v[t[3]]],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_collider::ClosureSdf;

    fn unit_cube() -> ClosureSdf {
        // Cube from (-1, -1, -1) to (+1, +1, +1). Inside = negative.
        ClosureSdf::new(
            |x, y, z| {
                let ax = x.abs() - 1.0;
                let ay = y.abs() - 1.0;
                let az = z.abs() - 1.0;
                ax.max(ay).max(az)
            },
            |_x, _y, _z| (1.0, 0.0, 0.0),
        )
    }

    fn ball_sdf() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 0.9,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
                (x / len, y / len, z / len)
            },
        )
    }

    #[test]
    fn generate_returns_empty_mesh_outside_interior() {
        // Empty AABB that lies entirely outside the surface produces
        // no tets.
        let sdf = ball_sdf();
        let mesh = generate(&sdf, [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], 0.5);
        assert!(mesh.tets.is_empty());
        assert!(mesh.vertices.is_empty());
    }

    #[test]
    fn generate_produces_tets_inside_cube() {
        let sdf = unit_cube();
        let mesh = generate(&sdf, [-1.5, -1.5, -1.5], [1.5, 1.5, 1.5], 0.5);
        assert!(!mesh.tets.is_empty());
        // Every cube contributes 5 tets.
        assert_eq!(mesh.tet_count() % 5, 0);
    }

    #[test]
    fn vertices_are_deduplicated() {
        let sdf = unit_cube();
        let mesh = generate(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 1.0);
        // For a 2×2×2 cube of cells (8 cubes), the vertex count is
        // 3^3 = 27 corners of the enclosing grid; dedup should keep
        // it well below the naive 8 * 8 = 64.
        assert!(mesh.vertex_count() <= 27);
    }

    #[test]
    #[should_panic(expected = "cell must be positive")]
    fn generate_panics_on_zero_cell() {
        let sdf = unit_cube();
        let _ = generate(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0.0);
    }

    #[test]
    fn tet_count_helper_matches_len() {
        let sdf = unit_cube();
        let mesh = generate(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0.5);
        assert_eq!(mesh.tet_count(), mesh.tets.len());
        assert_eq!(mesh.vertex_count(), mesh.vertices.len());
    }

    // ---- Marching Tets tests --------------------------------------------

    #[test]
    fn marching_tets_produces_more_tets_than_interior_only() {
        let sdf = ball_sdf();
        let bbox_min = [-1.5, -1.5, -1.5];
        let bbox_max = [1.5, 1.5, 1.5];
        let cell = 0.5;
        let interior = generate(&sdf, bbox_min, bbox_max, cell);
        let marching = generate_marching_tets(&sdf, bbox_min, bbox_max, cell);
        // Marching tets fills in the surface layer that `generate`
        // skipped, so the tet count must be strictly greater.
        assert!(marching.tet_count() >= interior.tet_count());
    }

    #[test]
    fn marching_tets_yields_empty_outside_geometry() {
        let sdf = ball_sdf();
        let mesh = generate_marching_tets(&sdf, [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], 0.5);
        assert!(mesh.tets.is_empty());
    }

    #[test]
    fn marching_tets_reproduces_interior_when_boundary_absent() {
        // Bounding box entirely inside the ball → no crossings, so the
        // marching-tets output should match the interior-only generator
        // in terms of coverage (equal tet count).
        let sdf = ball_sdf();
        let interior = generate(&sdf, [-0.4, -0.4, -0.4], [0.4, 0.4, 0.4], 0.2);
        let marching = generate_marching_tets(&sdf, [-0.4, -0.4, -0.4], [0.4, 0.4, 0.4], 0.2);
        assert_eq!(marching.tet_count(), interior.tet_count());
    }

    #[test]
    #[should_panic(expected = "cell must be positive")]
    fn marching_tets_panics_on_zero_cell() {
        let sdf = ball_sdf();
        let _ = generate_marching_tets(&sdf, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], 0.0);
    }

    #[test]
    fn other_three_excludes_input() {
        assert_eq!(other_three(0), [1, 2, 3]);
        assert_eq!(other_three(1), [0, 2, 3]);
        assert_eq!(other_three(2), [0, 1, 3]);
        assert_eq!(other_three(3), [0, 1, 2]);
    }

    // ---- Edge-split refinement tests -----------------------------------

    fn one_tet_mesh(side: f32) -> SdfTetMesh {
        let mut m = SdfTetMesh::default();
        m.vertices.push([0.0, 0.0, 0.0]);
        m.vertices.push([side, 0.0, 0.0]);
        m.vertices.push([0.0, side, 0.0]);
        m.vertices.push([0.0, 0.0, side]);
        m.tets.push(Tetrahedron {
            vertices: [0, 1, 2, 3],
        });
        m
    }

    #[test]
    fn refine_halves_edges_after_pass() {
        let mut m = one_tet_mesh(1.0);
        let passes = m.refine_by_max_edge_length(0.7, 4);
        assert!(m.max_edge_length() <= 1.0);
        assert!(passes >= 1);
        // Every refinement pass at least doubles the tet count while
        // shortening the longest edge.
        assert!(m.tet_count() >= 2);
    }

    #[test]
    fn refine_stops_when_no_edge_exceeds_threshold() {
        let mut m = one_tet_mesh(1.0);
        let passes = m.refine_by_max_edge_length(10.0, 8);
        // Threshold larger than any edge — no refinement should occur.
        assert_eq!(m.tet_count(), 1);
        assert!(passes >= 1);
    }

    #[test]
    #[should_panic(expected = "max_edge_length must be positive")]
    fn refine_panics_on_nonpositive_threshold() {
        let mut m = one_tet_mesh(1.0);
        let _ = m.refine_by_max_edge_length(0.0, 4);
    }
}
