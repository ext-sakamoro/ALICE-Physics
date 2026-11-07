//! A flat face of a signed distance field must come out of [`sdf_fem_mesh::generate_marching_tets`]
//! as a face, and the meshed volume must be the shape's volume.
//!
//! The bodies that stand on the ground (the letters of a logo, a block on a plane) touch it
//! with their flat bottoms; a bottom meshed as a ring of chamfers around a smaller face, or
//! as a single vertex, stands on a smaller area or on a point, and both its contact and its
//! sag are wrong. The mass of such a body is `ρ × Σ tet volume`, so a chamfered rim is also
//! a mass error.
//!
//! # The bounds are derived, not chosen
//!
//! - **Axis-aligned box.** Every face and every rim of a box is a plane or a straight line, so
//!   a mesher whose boundary vertices lie on the surface and which puts vertices on the rims
//!   reproduces the box exactly; what is left is f32 rounding of coordinates around `0.1 m`
//!   (`ulp ≈ 7.5e-9 m`, a relative `7.5e-8`). The bound `1e-5` on the face share and on the
//!   relative volume leaves two orders of magnitude for the rounding and nothing for a
//!   chamfer: one cell chamfered off one rim of a `0.2 m` cube at `0.04 m` cells is
//!   already `1e-2` of the face.
//! - **Curved shapes.** With every boundary vertex on the surface, the boundary is a
//!   triangulation inscribed in it, and a triangle of edge `ℓ` on a surface of smallest
//!   principal radius `R` deviates from it by at most the sagitta `ℓ² / (8 R)`. The volume
//!   error is then at most `A · ℓ² / (8 R)` with `A` the surface area and `ℓ` the longest
//!   boundary edge, measured on the mesh under test (not assumed).
//!
//! Measured before the fix (2026-11-08): a `0.2 m` cube at `0.04 m` cells keeps `0.64` of
//! its bottom area (`(0.2 − 0.04)² / 0.2²`, half a cell chamfered off each rim) at both
//! lattice alignments; a `0.0125 × 0.05 × 0.0125 m` column at `0.0125 m` cells has one
//! vertex on its bottom plane and no triangle at all (and an empty mesh when the face lies
//! on a lattice plane).
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: areas, volumes and their bounds are computed in f64 from the shape, never by the mesher under test"
)]

use alice_physics_core::sdf_collider::{ClosureSdf, SdfField};
use alice_physics_core::sdf_fem_mesh::{self, SdfTetMesh};

/// the relative error left to f32 rounding of coordinates on a box (see the module doc)
const BOX_EXACT: f64 = 1e-5;

/// an axis-aligned box, exact signed distance (metres)
struct BoxSdf([f32; 3]);

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

fn box_mesh(half: [f32; 3], cell: f32, pad_cells: f32) -> SdfTetMesh {
    let pad = cell * pad_cells;
    sdf_fem_mesh::generate_marching_tets(
        &BoxSdf(half),
        [-half[0] - pad, -half[1] - pad, -half[2] - pad],
        [half[0] + pad, half[1] + pad, half[2] + pad],
        cell,
    )
}

fn tri_area(p: [[f32; 3]; 3]) -> f64 {
    let a = [0, 1, 2].map(|k| f64::from(p[1][k]) - f64::from(p[0][k]));
    let b = [0, 1, 2].map(|k| f64::from(p[2][k]) - f64::from(p[0][k]));
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

/// (vertices on the bottom plane, area of the boundary triangles lying on it)
fn bottom_face(mesh: &SdfTetMesh, y_face: f32) -> (usize, f64) {
    let on = |v: [f32; 3]| (v[1] - y_face).abs() < 1e-6;
    let n_on = mesh.vertices.iter().filter(|v| on(**v)).count();
    let faces = mesh
        .boundary_faces()
        .expect("a closed mesh has boundary faces");
    let mut area = 0.0f64;
    for f in &faces {
        let p = f.map(|i| mesh.vertices[i as usize]);
        if p.iter().all(|v| on(*v)) {
            area += tri_area(p);
        }
    }
    (n_on, area)
}

/// Σ |tet volume| in f64
fn mesh_volume(mesh: &SdfTetMesh) -> f64 {
    let mut v = 0.0;
    for t in &mesh.tets {
        let p = t.vertices.map(|i| mesh.vertices[i as usize].map(f64::from));
        let e = |i: usize| [0, 1, 2].map(|k| p[i][k] - p[0][k]);
        let (a, b, c) = (e(1), e(2), e(3));
        let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        v += det.abs() / 6.0;
    }
    v
}

/// (largest |d| over boundary vertices, longest boundary edge)
fn boundary_stats<F: SdfField + ?Sized>(mesh: &SdfTetMesh, sdf: &F) -> (f64, f64) {
    let faces = mesh.boundary_faces().expect("manifold");
    let (mut dmax, mut lmax) = (0.0f64, 0.0f64);
    for f in &faces {
        for k in 0..3 {
            let p = mesh.vertices[f[k] as usize];
            let q = mesh.vertices[f[(k + 1) % 3] as usize];
            dmax = dmax.max(f64::from(sdf.distance(p[0], p[1], p[2])).abs());
            let l = (0..3)
                .map(|i| (f64::from(p[i]) - f64::from(q[i])).powi(2))
                .sum::<f64>()
                .sqrt();
            lmax = lmax.max(l);
        }
    }
    (dmax, lmax)
}

/// The bottom face of a `0.2 m` cube at `0.04 m` cells is the whole face, at both lattice
/// alignments (face on a lattice plane, face half a cell off it).
#[test]
#[ignore = "known defect: generate_marching_tets chamfers half a cell off every rim of a flat face (a 0.2 m cube at 0.04 m cells keeps 0.64 of its bottom area)"]
fn a_flat_face_is_meshed_whole() {
    for pad_cells in [2.0f32, 0.5] {
        let mesh = box_mesh([0.1, 0.1, 0.1], 0.04, pad_cells);
        let (n_on, area) = bottom_face(&mesh, -0.1);
        let share = area / (0.2 * 0.2);
        assert!(
            n_on >= 4,
            "pad {pad_cells} cells: {n_on} vertices on the bottom plane"
        );
        assert!(
            share >= 1.0 - BOX_EXACT,
            "pad {pad_cells} cells: bottom face share {share:.6} below 1 - {BOX_EXACT:e}"
        );
    }
}

/// Every boundary vertex of a box mesh lies on the box, and the meshed volume is the box's.
/// This is the mass check: `ρ × Σ tet volume` is the body's mass.
#[test]
#[ignore = "known defect: generate_marching_tets chamfers the rims of a box, so the meshed volume is short and rim vertices sit off the surface"]
fn a_box_mesh_has_the_box_volume_and_its_boundary_on_the_box() {
    for (half, cell) in [([0.1f32, 0.1, 0.1], 0.04f32), ([0.1, 0.03, 0.07], 0.025)] {
        for pad_cells in [2.0f32, 0.5, 0.37] {
            // one spare cell on the far side: the lattice drops a partial last
            // cell, which would otherwise cut the box at offsets like 0.37
            let pad = cell * pad_cells;
            let mesh = sdf_fem_mesh::generate_marching_tets(
                &BoxSdf(half),
                [-half[0] - pad, -half[1] - pad, -half[2] - pad],
                [half[0] + pad + cell, half[1] + pad + cell, half[2] + pad + cell],
                cell,
            );
            let truth = 8.0 * half.iter().map(|&h| f64::from(h)).product::<f64>();
            let rel = (mesh_volume(&mesh) - truth).abs() / truth;
            let (dmax, _) = boundary_stats(&mesh, &BoxSdf(half));
            assert!(
                rel <= BOX_EXACT,
                "half {half:?} cell {cell} pad {pad_cells}: volume off by {rel:.3e} of the box"
            );
            assert!(
                dmax <= BOX_EXACT * f64::from(cell),
                "half {half:?} cell {cell} pad {pad_cells}: a boundary vertex is {dmax:.3e} m off the surface"
            );
        }
    }
}

fn ball(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

fn torus(major: f32, minor: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| {
            let q = (x * x + y * y).sqrt() - major;
            (q * q + z * z).sqrt() - minor
        },
        move |x, y, z| {
            let radial = (x * x + y * y).sqrt().max(1.0e-6);
            let q = radial - major;
            let len = (q * q + z * z).sqrt().max(1.0e-6);
            (q * x / (radial * len), q * y / (radial * len), z / len)
        },
    )
}

/// On a ball and a torus every boundary vertex is on the surface (to f32 rounding of the
/// field, `1e-5` of a cell) and the meshed volume is within the inscribed-triangulation
/// bound `A · ℓ² / (8 R)` of the analytic volume.
#[test]
#[ignore = "known defect: generate_marching_tets places crossings by linear interpolation of the field, which is not on the surface where the field is not linear along the edge"]
fn curved_shapes_have_their_boundary_on_the_surface_and_their_volume_within_the_sagitta_bound()
{
    use std::f64::consts::PI;
    // (name, field, half extent of the box, cell, volume, area, smallest principal radius)
    let scenes: [(&str, ClosureSdf, f32, f32, f64, f64, f64); 2] = [
        ("ball", ball(1.0), 1.5, 0.25, 4.0 / 3.0 * PI, 4.0 * PI, 1.0),
        (
            "torus",
            torus(0.8, 0.35),
            1.5,
            0.125,
            2.0 * PI * PI * 0.8 * 0.35 * 0.35,
            4.0 * PI * PI * 0.8 * 0.35,
            0.35,
        ),
    ];
    for (name, sdf, h, cell, volume, area, r) in scenes {
        let mesh = sdf_fem_mesh::generate_marching_tets(&sdf, [-h; 3], [h; 3], cell);
        let (dmax, lmax) = boundary_stats(&mesh, &sdf);
        let bound = area * lmax * lmax / (8.0 * r);
        let err = (mesh_volume(&mesh) - volume).abs();
        assert!(
            dmax <= 1e-5 * f64::from(cell),
            "{name}: a boundary vertex is {dmax:.3e} off the surface"
        );
        assert!(
            err <= bound,
            "{name}: volume off by {err:.4e}, bound A l^2/(8R) = {bound:.4e} (l = {lmax:.4})"
        );
    }
}

/// A column one cell wide still has a bottom face: at least four vertices on the plane and a
/// positive area. Today it has one vertex and no triangle (and no mesh at all when the face
/// lies on a lattice plane).
#[test]
#[ignore = "known defect: generate_marching_tets reduces the bottom of a one-cell-wide column to a single vertex (0.0125 × 0.05 × 0.0125 m at 0.0125 m cells), or to no mesh when the face lies on a lattice plane"]
fn a_face_one_cell_wide_is_still_a_face() {
    for pad_cells in [1.0f32, 0.5] {
        let mesh = box_mesh([0.00625, 0.025, 0.00625], 0.0125, pad_cells);
        assert!(!mesh.tets.is_empty(), "pad {pad_cells} cells: no mesh");
        let (n_on, area) = bottom_face(&mesh, -0.025);
        assert!(
            n_on >= 4 && area > 0.0,
            "pad {pad_cells} cells: {n_on} vertices on the bottom plane, area {area:.3e}"
        );
    }
}

/// Today's behaviour, pinned: the measured shares and counts above, so a change in the
/// mesher shows here before the oracles turn green.
#[test]
fn today_a_cube_keeps_point_six_four_of_its_bottom_face_and_a_thin_column_a_point() {
    let mesh = box_mesh([0.1, 0.1, 0.1], 0.04, 2.0);
    let (n_on, area) = bottom_face(&mesh, -0.1);
    let share = area / 0.04;
    assert!((share - 0.64).abs() < 1e-3, "share {share:.4}");
    assert!(n_on >= 4);
    let thin = box_mesh([0.00625, 0.025, 0.00625], 0.0125, 0.5);
    let (n_thin, area_thin) = bottom_face(&thin, -0.025);
    assert_eq!((n_thin, area_thin), (1, 0.0));
    assert!(box_mesh([0.00625, 0.025, 0.00625], 0.0125, 1.0)
        .tets
        .is_empty());
}

#[test]
#[ignore = "diagnostic"]
fn zz_diag_bottom() {
    let mesh = box_mesh([0.1, 0.1, 0.1], 0.04, 2.0);
    let on = |v: [f32; 3]| (v[1] + 0.1).abs() < 1e-6;
    let faces = mesh.boundary_faces().unwrap();
    let mut n = 0;
    for f in &faces {
        let p = f.map(|i| mesh.vertices[i as usize]);
        let k = p.iter().filter(|v| on(**v)).count();
        let low = p.iter().all(|v| v[1] < -0.055 && v[0] < -0.055 && v[2] < -0.055);
        if low && n < 40 {
            n += 1;
            eprintln!("{k} on: {:?}", p);
        }
    }
    eprintln!("vol {:.6e} truth 8e-3", mesh_volume(&mesh));
}

#[test]
#[ignore = "diagnostic"]
fn zz_diag_corners() {
    let hb = [0.1f32, 0.03, 0.07];
    for pad in [2.0f32, 0.5, 0.37] {
        let (c, p) = (0.025f32, 0.025f32 * pad);
        let mesh = sdf_fem_mesh::generate_marching_tets(&BoxSdf(hb), [-hb[0] - p, -hb[1] - p, -hb[2] - p], [hb[0] + p + c, hb[1] + p + c, hb[2] + p + c], c);
        for sx in [-1.0f32, 1.0] { for sy in [-1.0f32, 1.0] { for sz in [-1.0f32, 1.0] {
            let c = [hb[0] * sx, hb[1] * sy, hb[2] * sz];
            let best = mesh.vertices.iter().map(|v| ((v[0]-c[0]).powi(2)+(v[1]-c[1]).powi(2)+(v[2]-c[2]).powi(2)).sqrt()).fold(f32::INFINITY, f32::min);
            eprintln!("pad {pad} corner {c:?}: nearest vertex {best:.2e}");
        }}}
        let (_, a) = bottom_face(&mesh, -hb[1]);
        let a = a * 0.04 / (4.0 * f64::from(hb[0] * hb[2]));
        let (dmax, _) = boundary_stats(&mesh, &BoxSdf(hb));
        let bf = mesh.boundary_faces().unwrap();
        let mut seen = std::collections::BTreeSet::new();
        for f in &bf { for &i in f { if seen.insert(i) { let v = mesh.vertices[i as usize]; let d = BoxSdf(hb).distance(v[0], v[1], v[2]); if d.abs() > 1e-6 { eprintln!("  off v{i} {v:?} d {d:.2e}"); } } } }
        let mut worst = 180.0f64;
        for t in &mesh.tets {
            let p = t.vertices.map(|i| mesh.vertices[i as usize].map(f64::from));
            let e = |a: usize, b: usize| [0, 1, 2].map(|k| p[b][k] - p[a][k]);
            let cr = |a: [f64; 3], b: [f64; 3]| [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
            let nrm = |a: [f64; 3]| (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt();
            for (i, j) in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] {
                let o: Vec<usize> = (0..4).filter(|&k| k != i && k != j).collect();
                let ed = e(i, j);
                let n1 = cr(ed, e(i, o[0]));
                let n2 = cr(ed, e(i, o[1]));
                let c = (n1[0]*n2[0]+n1[1]*n2[1]+n1[2]*n2[2]) / (nrm(n1)*nrm(n2));
                worst = worst.min(c.clamp(-1.0, 1.0).acos().to_degrees());
            }
        }
        eprintln!("pad {pad}: worst dihedral {worst:.2} deg, tets {}", mesh.tets.len());
        eprintln!("pad {pad}: share {:.5} vol {:.6e} truth {:.6e} dmax {dmax:.2e}", a / 0.04, mesh_volume(&mesh), 8.0 * f64::from(hb[0] * hb[1] * hb[2]));
    }
}
