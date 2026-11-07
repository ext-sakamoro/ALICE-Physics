//! A column of tetrahedra for the soft-body oracles and measurements: a lattice of
//! `cell`-sized cubes, each split into the six Kuhn tetrahedra (conforming across cubes with
//! no parity rule), so the mesh has the box's exact faces at every cell size and a full layer
//! of vertices on each face. Lumped vertex masses, the face vertex sets and an axis-aligned box
//! distance field live here too.
#![allow(
    dead_code,
    reason = "shared by several test files through #[path]; each uses a subset of these helpers"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: lattice coordinates, lumped masses and face areas are computed in f64 from the box, never by the code under test"
)]

use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// an axis-aligned box, exact signed distance (the same units as its half extents)
pub struct BoxSdf(pub [f32; 3]);

impl SdfField for BoxSdf {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let q = [
            x.abs() - self.0[0],
            y.abs() - self.0[1],
            z.abs() - self.0[2],
        ];
        let outside =
            (q[0].max(0.0).powi(2) + q[1].max(0.0).powi(2) + q[2].max(0.0).powi(2)).sqrt();
        let inside = q[0].max(q[1]).max(q[2]).min(0.0);
        outside + inside
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        let q = [
            x.abs() - self.0[0],
            y.abs() - self.0[1],
            z.abs() - self.0[2],
        ];
        let axis = (0..3)
            .max_by(|&a, &b| q[a].partial_cmp(&q[b]).expect("finite"))
            .expect("three axes");
        let p = [x, y, z];
        let mut n = [0.0f32; 3];
        n[axis] = if p[axis] >= 0.0 { 1.0 } else { -1.0 };
        (n[0], n[1], n[2])
    }
}

/// the vertices of a face lie on its plane to the float rounding of the lattice; the next
/// layer is a cell away
pub const FACE_TOL: f32 = 1e-4;

/// a `2 half_w × 2 half_h × 2 half_w` column centred at the origin, `cell`-sized cubes split
/// into six Kuhn tetrahedra each. With an even number of cubes across, the split is mirrored
/// in the halves `x < 0` and `z < 0`, so the mesh (vertices and connectivity) is symmetric
/// under `x → −x` and `z → −z`: the modal analysis keeps its iterate in the symmetric
/// subspace, which is invariant only then. The mirrored faces on the planes `x = 0`, `z = 0`
/// carry the same diagonal from both sides (the mirror maps the Kuhn face diagonal onto
/// itself), so the mesh stays conforming.
pub fn column_mesh(cell: f32, half_w: f64, half_h: f64) -> SdfTetMesh {
    let nx = ((2.0 * half_w / f64::from(cell)).round() as usize).max(1);
    let ny = ((2.0 * half_h / f64::from(cell)).round() as usize).max(1);
    let nz = nx;
    let at = |i: usize, j: usize, k: usize| -> u32 { ((i * (ny + 1) + j) * (nz + 1) + k) as u32 };
    let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                vertices.push([
                    (i as f64 * f64::from(cell) - half_w) as f32,
                    (j as f64 * f64::from(cell) - half_h) as f32,
                    (k as f64 * f64::from(cell) - half_w) as f32,
                ]);
            }
        }
    }
    let mut tets = Vec::with_capacity(6 * nx * ny * nz);
    // the six tetrahedra of a cube share its main diagonal v000–v111; each is (v000, a, b, v111)
    // along one of the six monotone paths of the cube's edges
    let paths: [[(usize, usize, usize); 2]; 6] = [
        [(1, 0, 0), (1, 1, 0)],
        [(1, 0, 0), (1, 0, 1)],
        [(0, 1, 0), (1, 1, 0)],
        [(0, 1, 0), (0, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, 1), (0, 1, 1)],
    ];
    let mirrored = nx % 2 == 0;
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // local cube coordinates run away from the mirror planes in the far halves
                let flip_x = mirrored && i >= nx / 2;
                let flip_z = mirrored && k >= nz / 2;
                let gi = |dx: usize| if flip_x { i + 1 - dx } else { i + dx };
                let gk = |dz: usize| if flip_z { k + 1 - dz } else { k + dz };
                let v000 = at(gi(0), j, gk(0));
                let v111 = at(gi(1), j + 1, gk(1));
                for [(a0, a1, a2), (b0, b1, b2)] in paths {
                    let a = at(gi(a0), j + a1, gk(a2));
                    let b = at(gi(b0), j + b1, gk(b2));
                    tets.push(Tetrahedron {
                        vertices: [v000, a, b, v111],
                    });
                }
            }
        }
    }
    SdfTetMesh { vertices, tets }
}

pub fn tet_volume(v: &[[f32; 3]], t: [u32; 4]) -> f64 {
    let p = |i: u32| {
        let a = v[i as usize];
        [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
    };
    let (a, b, c, d) = (p(t[0]), p(t[1]), p(t[2]), p(t[3]));
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    ((ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0).abs()
}

/// lumped vertex masses, `density × volume` units: a quarter of each tetrahedron's mass to
/// each of its vertices
pub fn lumped_masses(mesh: &SdfTetMesh, density: f64) -> Vec<f64> {
    let mut m = vec![0.0; mesh.vertices.len()];
    for t in &mesh.tets {
        let share = density * tet_volume(&mesh.vertices, t.vertices) / 4.0;
        for &i in &t.vertices {
            m[i as usize] += share;
        }
    }
    m
}

pub fn min_y(mesh: &SdfTetMesh) -> f32 {
    mesh.vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::INFINITY, f32::min)
}

pub fn max_y(mesh: &SdfTetMesh) -> f32 {
    mesh.vertices
        .iter()
        .map(|v| v[1])
        .fold(f32::NEG_INFINITY, f32::max)
}

pub fn base_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y0 = min_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y0).abs() < FACE_TOL)
        .collect()
}

pub fn top_vertices(mesh: &SdfTetMesh) -> Vec<usize> {
    let y1 = max_y(mesh);
    (0..mesh.vertices.len())
        .filter(|&i| (mesh.vertices[i][1] - y1).abs() < FACE_TOL)
        .collect()
}

pub fn height(mesh: &SdfTetMesh) -> f64 {
    f64::from(max_y(mesh) - min_y(mesh))
}
