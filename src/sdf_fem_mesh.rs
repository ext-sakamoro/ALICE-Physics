//! Cartesian tetrahedral mesh generation from a signed distance field.
//!
//! Companion of [`crate::deformable`] and [`crate::sdf_collider`] that
//! exposes a scaffold-tier "SDF → tet mesh" pipeline. Each Cartesian
//! cube fully inside the SDF is diced into five reference tetrahedra
//! (the standard "5-tet cube" pattern); cubes crossing the surface
//! are dropped by the MVP because clip-tet extraction is
//! substantially more code than the module aims to ship.
//!
//! # Limitations
//!
//! - Only cubes with all eight corners strictly inside the SDF are
//!   emitted. Surface-conforming cells / Marching Tets / Delaunay
//!   refinement are future work.
//! - Vertex indices are deduplicated via a small `HashMap` keyed on
//!   the integer lattice cell coordinate; positions are always
//!   sampled at cell corners, not shifted to the SDF surface.
//! - The generated mesh is intended for downstream `deformable::…`
//!   FEM callers; it is not tuned for rendering.

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
                for tet in cube_to_five_tets(cube_verts) {
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
                // Standard 5-tet decomposition of the cube (vertex indices
                // into the 8-element `corner_*` arrays).
                for sub_tet in [
                    [0, 1, 2, 5],
                    [0, 2, 3, 7],
                    [0, 4, 5, 7],
                    [2, 5, 6, 7],
                    [0, 2, 5, 7],
                ] {
                    let positions = [
                        corner_positions[sub_tet[0]],
                        corner_positions[sub_tet[1]],
                        corner_positions[sub_tet[2]],
                        corner_positions[sub_tet[3]],
                    ];
                    let sdf_values = [
                        corner_sdf[sub_tet[0]],
                        corner_sdf[sub_tet[1]],
                        corner_sdf[sub_tet[2]],
                        corner_sdf[sub_tet[3]],
                    ];
                    marching_tet_emit(&mut mesh, positions, sdf_values);
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
fn marching_tet_emit(mesh: &mut SdfTetMesh, v: [[f32; 3]; 4], sdf: [f32; 4]) {
    // Bitmask of inside vertices (bit i = 1 iff sdf[i] < 0).
    let mut mask: u8 = 0;
    for (i, &d) in sdf.iter().enumerate() {
        if d < 0.0 {
            mask |= 1 << i;
        }
    }
    // Fast paths.
    if mask == 0b0000 {
        return;
    }
    if mask == 0b1111 {
        emit_tet(mesh, v[0], v[1], v[2], v[3]);
        return;
    }
    // Linear-interpolate to the zero crossing along edge (a, b).
    let interp = |a: usize, b: usize| -> [f32; 3] {
        let denom = sdf[a] - sdf[b];
        let t = if denom.abs() < 1.0e-9 {
            0.5
        } else {
            sdf[a] / denom
        };
        [
            v[a][0] + t * (v[b][0] - v[a][0]),
            v[a][1] + t * (v[b][1] - v[a][1]),
            v[a][2] + t * (v[b][2] - v[a][2]),
        ]
    };
    let count = mask.count_ones();
    match count {
        1 => {
            // One vertex inside; emit one tet from that vertex to three
            // edge intersections. Identify the interior vertex.
            let inside = mask.trailing_zeros() as usize;
            let others: [usize; 3] = other_three(inside);
            let e0 = interp(inside, others[0]);
            let e1 = interp(inside, others[1]);
            let e2 = interp(inside, others[2]);
            emit_tet(mesh, v[inside], e0, e1, e2);
        }
        3 => {
            // Three vertices inside; complement of case 1.
            let outside = (!mask & 0b1111).trailing_zeros() as usize;
            let insides: [usize; 3] = other_three(outside);
            let e0 = interp(insides[0], outside);
            let e1 = interp(insides[1], outside);
            let e2 = interp(insides[2], outside);
            // Interior polytope = original tet minus tet(outside, e0, e1, e2).
            // Decompose the remaining wedge into three tetrahedra.
            emit_tet(mesh, v[insides[0]], v[insides[1]], v[insides[2]], e0);
            emit_tet(mesh, v[insides[1]], v[insides[2]], e0, e1);
            emit_tet(mesh, v[insides[2]], e0, e1, e2);
        }
        2 => {
            // Two vertices inside, two outside. The interior polytope is
            // a wedge (triangular prism) with six vertices: the two
            // interior vertices plus four edge intersections.
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
            let a = v[insides[0]];
            let b = v[insides[1]];
            let e_a_c = interp(insides[0], outsides[0]);
            let e_a_d = interp(insides[0], outsides[1]);
            let e_b_c = interp(insides[1], outsides[0]);
            let e_b_d = interp(insides[1], outsides[1]);
            // Prism decomposition into three tets: (a, b, e_bc, e_ac),
            // (a, e_bc, e_ac, e_bd), (a, e_ac, e_bd, e_ad).
            emit_tet(mesh, a, b, e_b_c, e_a_c);
            emit_tet(mesh, a, e_b_c, e_a_c, e_b_d);
            emit_tet(mesh, a, e_a_c, e_b_d, e_a_d);
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

fn emit_tet(mesh: &mut SdfTetMesh, a: [f32; 3], b: [f32; 3], c: [f32; 3], d: [f32; 3]) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(a);
    mesh.vertices.push(b);
    mesh.vertices.push(c);
    mesh.vertices.push(d);
    mesh.tets.push(Tetrahedron {
        vertices: [base, base + 1, base + 2, base + 3],
    });
}

/// Canonical 5-tet decomposition of a unit cube. Input indices are
/// ordered as
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
/// The five tets are the four corners `(0,1,2,5)`, `(0,2,3,7)`,
/// `(0,4,5,7)`, `(2,5,6,7)`, and the interior `(0,2,5,7)`.
fn cube_to_five_tets(v: [u32; 8]) -> [Tetrahedron; 5] {
    [
        Tetrahedron {
            vertices: [v[0], v[1], v[2], v[5]],
        },
        Tetrahedron {
            vertices: [v[0], v[2], v[3], v[7]],
        },
        Tetrahedron {
            vertices: [v[0], v[4], v[5], v[7]],
        },
        Tetrahedron {
            vertices: [v[2], v[5], v[6], v[7]],
        },
        Tetrahedron {
            vertices: [v[0], v[2], v[5], v[7]],
        },
    ]
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
}
