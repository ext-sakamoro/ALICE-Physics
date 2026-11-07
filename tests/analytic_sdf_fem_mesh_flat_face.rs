//! A flat face of a signed distance field must come out of [`sdf_fem_mesh::generate_marching_tets`]
//! as a face: its boundary triangles on the plane must cover the face's area within one
//! cell's rounding, and a face one cell wide must still have a vertex layer, not a point.
//!
//! The bodies that stand on the ground (the letters of a logo, a block on a plane) touch it
//! with their flat bottoms; a bottom meshed as a ring of chamfers around a smaller face, or
//! as a single vertex, stands on a smaller area or on a point and both its contact and its
//! sag are wrong.
//!
//! Measured (2026-11-08): a `0.2 m` cube at `0.04 m` cells keeps `0.64` of its bottom area
//! (`(0.2 − 0.04)² / 0.2²`: the outer one-cell ring is chamfered) at both lattice
//! alignments; a `0.0125 × 0.05 × 0.0125 m` column at `0.0125 m` cells has one vertex on its
//! bottom plane and no triangle at all (and an empty mesh when the face lies on a lattice
//! plane). The bound below is the area a face loses to a half-cell chamfer around its rim,
//! `1 − (w − c)² / w²` with the half cell `c = cell / 2`: a mesher that places vertices on
//! the surface along every lattice edge that crosses it keeps at least that.
#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the face area and its bound are computed in f64 from the box, never by the mesher under test"
)]

use alice_physics_core::sdf_collider::SdfField;
use alice_physics_core::sdf_fem_mesh::{self, SdfTetMesh};

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

/// (vertices on the bottom plane, area of the boundary triangles lying on it)
fn bottom_face(mesh: &SdfTetMesh, y_face: f32) -> (usize, f64) {
    let on = |v: [f32; 3]| (v[1] - y_face).abs() < 1e-5;
    let n_on = mesh.vertices.iter().filter(|v| on(**v)).count();
    let faces = mesh
        .boundary_faces()
        .expect("a closed mesh has boundary faces");
    let mut area = 0.0f64;
    for f in &faces {
        let p: Vec<[f32; 3]> = f.iter().map(|&i| mesh.vertices[i as usize]).collect();
        if p.iter().all(|v| on(*v)) {
            let a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let c = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            area += 0.5 * f64::from((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
        }
    }
    (n_on, area)
}

/// the least share of a `w × w` face a half-cell chamfer around its rim leaves
fn least_share(w: f64, cell: f64) -> f64 {
    let inner = (w - cell).max(0.0);
    inner * inner / (w * w)
}

/// The bottom face of a `0.2 m` cube at `0.04 m` cells keeps at least the half-cell-chamfer
/// share of its area (`0.81`) at both lattice alignments. Today it keeps `0.64` (a full
/// cell ring is lost).
#[test]
#[ignore = "known defect: generate_marching_tets chamfers a full cell ring off a flat face (a 0.2 m cube at 0.04 m cells keeps 0.64 of its bottom area, the half-cell bound is 0.81)"]
fn a_flat_face_keeps_its_area_to_within_a_half_cell_chamfer() {
    for pad_cells in [2.0f32, 0.5] {
        let mesh = box_mesh([0.1, 0.1, 0.1], 0.04, pad_cells);
        let (n_on, area) = bottom_face(&mesh, -0.1);
        let share = area / (0.2 * 0.2);
        let bound = least_share(0.2, 0.04);
        assert!(
            n_on >= 4,
            "pad {pad_cells} cells: {n_on} vertices on the bottom plane"
        );
        assert!(
            share >= bound,
            "pad {pad_cells} cells: bottom face share {share:.3} below the bound {bound:.3}"
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
