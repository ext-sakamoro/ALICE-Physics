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
}
