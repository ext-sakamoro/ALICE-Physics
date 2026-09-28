//! Conformity invariants for `alice_physics::sdf_fem_mesh`.
//!
//! A tetrahedral mesh is **conforming** when any two elements meet along a
//! shared face, edge or vertex and nothing else. Break that and the FEM
//! displacement is discontinuous across the offending face; the solution is no
//! longer the minimiser of the energy over a proper function space and there is
//! no convergence guarantee.
//!
//! # Why this is a face census and not a patch test
//!
//! The obvious check — run the FEM patch test on the generated mesh — **cannot
//! see this defect**, for two independent reasons:
//!
//! 1. The two triangulations of a square interpolate a linear function
//!    identically. With corner values `a, b, c, d` in cyclic order, one diagonal
//!    gives `(a+c)/2` at the centre and the other `(b+d)/2`, and `a+c = b+d`
//!    holds identically for a linear field.
//! 2. The residual at a free node is `σ : Σ_e V_e ∇N_i`, and `V ∇N_i` is the
//!    area vector of the face opposite `i`. If the tetrahedra fill the domain
//!    with no gaps and no overlaps — which a mismatched dicing still does — the
//!    faces opposite `i` form a closed surface and their area vectors sum to
//!    zero by the divergence theorem, whatever the faces look like.
//!
//! Measured on this crate before the fix: the patch test came back at
//! 3.6e-15 MPa while 576 of the 768 singly-used faces were interior.
//!
//! So the invariant is counted directly: **a triangular face may be used once
//! only when it sits on the boundary of the meshed region.** The boundary is
//! worked out here from the generator's own documented occupancy rule (a cube
//! is meshed iff all eight corners are inside), independently of how the cube is
//! diced — which is the part under test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_fem_mesh::{generate, generate_marching_tets, SdfTetMesh};
use std::collections::{HashMap, HashSet};

fn box_sdf(half: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x.abs() - half).max(y.abs() - half).max(z.abs() - half),
        |_x, _y, _z| (1.0, 0.0, 0.0),
    )
}

fn ball_sdf(radius: f32) -> ClosureSdf {
    ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |x, y, z| {
            let len = (x * x + y * y + z * z).sqrt().max(1.0e-6);
            (x / len, y / len, z / len)
        },
    )
}

/// How many times each triangular face appears, keyed by its sorted vertex
/// indices.
fn face_use_counts(mesh: &SdfTetMesh) -> HashMap<[u32; 3], usize> {
    let mut counts = HashMap::new();
    for tet in &mesh.tets {
        let v = tet.vertices;
        for face in [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[0], v[2], v[3]],
            [v[1], v[2], v[3]],
        ] {
            let mut key = face;
            key.sort_unstable();
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

/// Lattice cells the generator meshes, by its own rule: every one of the eight
/// corners is inside. Recomputed here rather than read back from the mesh, so
/// the census does not inherit whatever the dicing did.
fn meshed_cells<F: SdfField + ?Sized>(
    sdf: &F,
    min: [f32; 3],
    cell: f32,
    counts: [i32; 3],
) -> HashSet<(i32, i32, i32)> {
    let mut cells = HashSet::new();
    for iz in 0..counts[2] {
        for iy in 0..counts[1] {
            for ix in 0..counts[0] {
                let inside = (0..8).all(|c| {
                    let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let p = [
                        min[0] + (ix + dx) as f32 * cell,
                        min[1] + (iy + dy) as f32 * cell,
                        min[2] + (iz + dz) as f32 * cell,
                    ];
                    sdf.distance(p[0], p[1], p[2]) <= 0.0
                });
                if inside {
                    cells.insert((ix, iy, iz));
                }
            }
        }
    }
    cells
}

struct Census {
    shared: usize,
    boundary: usize,
    /// Faces used once that are *not* on the boundary of the meshed region.
    /// Any one of these is a conformity break.
    dangling: usize,
    example: Option<([u32; 3], [[f32; 3]; 3])>,
}

/// Classify every singly-used face as boundary or dangling.
///
/// A face is on the boundary of the meshed region when the cell on its far side
/// is not meshed. The far side is found from the face centroid: step half a cell
/// along the face normal and see which lattice cell that lands in.
fn census(mesh: &SdfTetMesh, min: [f32; 3], cell: f32, cells: &HashSet<(i32, i32, i32)>) -> Census {
    let counts = face_use_counts(mesh);
    let mut out = Census {
        shared: 0,
        boundary: 0,
        dangling: 0,
        example: None,
    };
    for (face, uses) in &counts {
        assert!(
            *uses <= 2,
            "a face shared by more than two tetrahedra is never valid: {face:?} used {uses} times"
        );
        if *uses == 2 {
            out.shared += 1;
            continue;
        }
        let p: [[f32; 3]; 3] = [
            mesh.vertices[face[0] as usize],
            mesh.vertices[face[1] as usize],
            mesh.vertices[face[2] as usize],
        ];
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            (p[0][2] + p[1][2] + p[2][2]) / 3.0,
        ];
        let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
        let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let step = 0.25 * cell;
        // Both sides: the face is on the boundary when at least one side has no
        // meshed cell. (For an interface between two meshed cells both sides are
        // meshed, so the face must have been used twice.)
        let mut both_meshed = true;
        for sign in [1.0_f32, -1.0] {
            let q = [
                centroid[0] + sign * step * n[0] / len,
                centroid[1] + sign * step * n[1] / len,
                centroid[2] + sign * step * n[2] / len,
            ];
            let cellof = (
                ((q[0] - min[0]) / cell).floor() as i32,
                ((q[1] - min[1]) / cell).floor() as i32,
                ((q[2] - min[2]) / cell).floor() as i32,
            );
            if !cells.contains(&cellof) {
                both_meshed = false;
            }
        }
        if both_meshed {
            out.dangling += 1;
            if out.example.is_none() {
                out.example = Some((*face, p));
            }
        } else {
            out.boundary += 1;
        }
    }
    out
}

fn check_conforming(name: &str, sdf: &dyn SdfField, min: [f32; 3], max: [f32; 3], cell: f32) {
    let mesh = generate(sdf, min, max, cell);
    assert!(!mesh.tets.is_empty(), "{name}: the scene must produce tets");
    let counts = [
        ((max[0] - min[0]) / cell).max(1.0) as i32,
        ((max[1] - min[1]) / cell).max(1.0) as i32,
        ((max[2] - min[2]) / cell).max(1.0) as i32,
    ];
    let cells = meshed_cells(sdf, min, cell, counts);
    let c = census(&mesh, min, cell, &cells);
    eprintln!(
        "[{name}] {} verts, {} tets, {} meshed cells | faces: shared {}, boundary {}, dangling {}",
        mesh.vertex_count(),
        mesh.tet_count(),
        cells.len(),
        c.shared,
        c.boundary,
        c.dangling
    );
    assert_eq!(
        c.dangling,
        0,
        "{name}: {} interior faces are used by only one tetrahedron, so the dicing \
         of neighbouring cells disagrees. Example face {:?} at {:?}",
        c.dangling,
        c.example.map(|e| e.0),
        c.example.map(|e| e.1)
    );
}

#[test]
fn generate_is_conforming_for_a_box() {
    // a box whose faces land on lattice planes: every interior cell has six
    // meshed neighbours, so every interface must be shared
    check_conforming(
        "box 20mm / cell 4mm",
        &box_sdf(10.0),
        [-12.0, -12.0, -12.0],
        [12.0, 12.0, 12.0],
        4.0,
    );
}

#[test]
fn generate_is_conforming_for_a_box_at_odd_cell_counts() {
    // odd cell counts per axis, so a checkerboard parity scheme cannot rely on
    // the grid being even
    check_conforming(
        "box 20mm / cell 3mm",
        &box_sdf(10.0),
        [-12.0, -12.0, -12.0],
        [12.0, 12.0, 12.0],
        3.0,
    );
}

#[test]
fn generate_is_conforming_for_a_ball() {
    // a curved surface leaves a ragged set of meshed cells, so boundary and
    // interface faces are interleaved rather than sitting on flat planes
    check_conforming(
        "ball r=9mm / cell 2mm",
        &ball_sdf(9.0),
        [-10.0, -10.0, -10.0],
        [10.0, 10.0, 10.0],
        2.0,
    );
}

#[test]
fn generate_is_conforming_for_a_slab() {
    // one cell thick in z: every face of the single layer is boundary, so a
    // scheme that only matches within a layer still has to get this right
    check_conforming(
        "slab / cell 2mm",
        &box_sdf(8.0),
        [-9.0, -9.0, -1.0],
        [9.0, 9.0, 1.0],
        2.0,
    );
}

/// Every vertex the mesh emits must be a distinct position; a mesh that repeats
/// a position under two indices has a hanging node there and is not conforming
/// however the faces are counted.
fn assert_positions_are_deduplicated(name: &str, mesh: &SdfTetMesh) {
    let mut keys: Vec<[i64; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            [
                (f64::from(v[0]) * 1.0e6).round() as i64,
                (f64::from(v[1]) * 1.0e6).round() as i64,
                (f64::from(v[2]) * 1.0e6).round() as i64,
            ]
        })
        .collect();
    let total = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(
        keys.len(),
        total,
        "{name}: {} vertex entries for only {} distinct positions — the repeats are \
         hanging nodes that nothing connects",
        total,
        keys.len()
    );
}

#[test]
fn generate_does_not_repeat_vertex_positions() {
    let mesh = generate(
        &box_sdf(10.0),
        [-12.0, -12.0, -12.0],
        [12.0, 12.0, 12.0],
        4.0,
    );
    assert_positions_are_deduplicated("generate", &mesh);
}

#[test]
fn generate_marching_tets_does_not_repeat_vertex_positions() {
    let mesh = generate_marching_tets(
        &ball_sdf(9.0),
        [-10.0, -10.0, -10.0],
        [10.0, 10.0, 10.0],
        2.0,
    );
    assert_positions_are_deduplicated("generate_marching_tets", &mesh);
}

/// Marching Tetrahedra conformity.
///
/// The boundary of a marching-tets mesh is the zero level set, not a lattice
/// plane, so the cube-occupancy census above does not apply. The criterion that
/// does apply is about where the boundary can *be*: the clipped polytope's
/// outer surface is made entirely of zero crossings, so **every boundary face
/// has all three vertices on the surface**. Contrapositive: a face holding even
/// one lattice corner that is strictly inside cannot be a boundary face, and
/// must therefore be shared by two tetrahedra.
///
/// An earlier version of this check required *all three* vertices to be deep
/// inside. That only ever looked at the fully-interior region, where each
/// sub-tetrahedron is emitted whole, so it never reached the clipping code at
/// all — swapping the prism's diagonal left it green. The criterion below
/// reaches the clipped wedges, which is where the diagonal choices live.
fn assert_interior_faces_are_shared(
    name: &str,
    mesh: &SdfTetMesh,
    sdf: &dyn SdfField,
    min: [f32; 3],
    cell: f32,
) {
    let is_lattice_corner = |p: [f32; 3]| -> bool {
        (0..3).all(|i| {
            let k = (p[i] - min[i]) / cell;
            (k - k.round()).abs() < 1.0e-3
        })
    };
    let depth = 1.0e-3 * cell;
    let counts = face_use_counts(mesh);
    let mut checked = 0usize;
    let mut dangling = 0usize;
    let mut example = None;
    for (face, uses) in &counts {
        let has_interior_corner = face.iter().any(|&n| {
            let p = mesh.vertices[n as usize];
            is_lattice_corner(p) && sdf.distance(p[0], p[1], p[2]) < -depth
        });
        if !has_interior_corner {
            continue;
        }
        checked += 1;
        if *uses == 1 {
            dangling += 1;
            if example.is_none() {
                example = Some((*face, face.map(|n| mesh.vertices[n as usize])));
            }
        }
    }
    eprintln!("[{name}] {checked} interior faces checked, {dangling} dangling");
    assert!(
        checked > 100,
        "{name}: only {checked} faces qualified, so this check is not exercising the \
         clipped region — pick a larger shape or a smaller cell"
    );
    assert_eq!(
        dangling,
        0,
        "{name}: {dangling} faces hold a strictly-interior lattice corner, so they cannot \
         be on the zero level set, yet they belong to only one tetrahedron. \
         Example {:?} at {:?}",
        example.map(|e| e.0),
        example.map(|e| e.1)
    );
}

#[test]
fn generate_marching_tets_is_conforming_inside() {
    let cell = 2.0_f32;
    let sdf = ball_sdf(9.0);
    let min = [-10.0, -10.0, -10.0];
    let mesh = generate_marching_tets(&sdf, min, [10.0, 10.0, 10.0], cell);
    assert_interior_faces_are_shared("marching tets, ball r=9mm", &mesh, &sdf, min, cell);
}

#[test]
fn generate_marching_tets_is_conforming_inside_a_box() {
    let cell = 3.0_f32;
    let sdf = box_sdf(10.0);
    let min = [-12.0, -12.0, -12.0];
    let mesh = generate_marching_tets(&sdf, min, [12.0, 12.0, 12.0], cell);
    assert_interior_faces_are_shared("marching tets, box 20mm", &mesh, &sdf, min, cell);
}
