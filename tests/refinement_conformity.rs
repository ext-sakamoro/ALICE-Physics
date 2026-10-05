//! Oracle: does `refine_by_max_edge_length` keep the mesh conforming?
//!
//! This is the precondition of the adaptive-remeshing wall. Adaptive refinement
//! is only useful if it can split **part** of a mesh, and a partial split is
//! exactly where longest-edge bisection can leave a hanging node: if two tetrahedra
//! share a face and only one of them splits an edge of that face, the shared
//! face is replaced by two half-faces on one side and stays whole on the other.
//! The mesh then no longer partitions the domain along matching faces, which is
//! a variational crime — and one which a patch test
//! is measured to be blind to (3.6e-15 MPa on a mesh with 576 non-conforming
//! interior faces).
//!
//! So conformity has to be measured with a **face census**, never with a patch
//! test. Two scenes are measured here, because they give different answers and
//! the difference is the finding:
//!
//! 1. **Uniform** refinement of a Kuhn box. Every Kuhn tetrahedron has the same
//!    edge multiset `{h, h, h, h√2, h√2, h√3}`, so one threshold splits all of
//!    them on the same kind of edge. Conformity is expected to survive.
//! 2. **Graded** refinement, where one tetrahedron's longest edge belongs to the
//!    face it shares with a neighbour and the neighbour's longest edge does not.
//!    This is the minimal two-element scene that adaptivity has to handle, and
//!    it is where the hanging node appears.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::collections::HashMap;

use alice_physics::sdf_fem_mesh::{RefineError, SdfTetMesh, Tetrahedron};

// ---------------------------------------------------------------------------
// face census
// ---------------------------------------------------------------------------

/// How many tetrahedra use each face, keyed by the sorted vertex indices.
fn face_use_counts(mesh: &SdfTetMesh) -> HashMap<[u32; 3], usize> {
    let mut counts: HashMap<[u32; 3], usize> = HashMap::new();
    for tet in &mesh.tets {
        let v = tet.vertices;
        for face in [
            [v[0], v[1], v[2]],
            [v[0], v[1], v[3]],
            [v[0], v[2], v[3]],
            [v[1], v[2], v[3]],
        ] {
            let mut k = face;
            k.sort_unstable();
            *counts.entry(k).or_insert(0) += 1;
        }
    }
    counts
}

/// Faces that lie entirely in the plane `z = 0`, split by how many tetrahedra
/// use them.
///
/// In the graded scene below, `z = 0` is the *only* plane with material on both
/// sides, so a face there used by one tetrahedron is a hanging face by
/// construction — no geometric inside/outside test is needed, which keeps the
/// census exact.
fn z0_faces(mesh: &SdfTetMesh) -> (usize, usize) {
    let counts = face_use_counts(mesh);
    let mut shared = 0usize;
    let mut once = 0usize;
    for (face, n) in &counts {
        let on_plane = face
            .iter()
            .all(|&i| mesh.vertices[i as usize][2].abs() < 1e-6);
        if !on_plane {
            continue;
        }
        match n {
            1 => once += 1,
            _ => shared += 1,
        }
    }
    (shared, once)
}

/// Interior faces of an axis-aligned box that are used by only one tetrahedron.
///
/// A face of a box-shaped domain is on the boundary exactly when all three of
/// its vertices sit on one of the six bounding planes. Anything else used once is
/// a hanging face. Classifying "used once" as boundary wholesale is the mistake
/// the memory warns about, so the plane test is applied per face.
fn hanging_interior_faces(mesh: &SdfTetMesh, min: [f32; 3], max: [f32; 3]) -> usize {
    let counts = face_use_counts(mesh);
    let mut hanging = 0usize;
    for (face, n) in &counts {
        if *n != 1 {
            continue;
        }
        let mut on_boundary = false;
        for axis in 0..3 {
            let all_min = face
                .iter()
                .all(|&i| (mesh.vertices[i as usize][axis] - min[axis]).abs() < 1e-5);
            let all_max = face
                .iter()
                .all(|&i| (mesh.vertices[i as usize][axis] - max[axis]).abs() < 1e-5);
            if all_min || all_max {
                on_boundary = true;
                break;
            }
        }
        if !on_boundary {
            hanging += 1;
        }
    }
    hanging
}

/// Smallest dihedral angle over the mesh, in degrees — reported so the aspect
/// ratio drift the module doc warns about is a number rather than a caveat.
fn min_dihedral_degrees(mesh: &SdfTetMesh) -> f64 {
    let mut worst = 180.0_f64;
    for tet in &mesh.tets {
        let p: Vec<[f64; 3]> = tet
            .vertices
            .iter()
            .map(|&i| {
                let v = mesh.vertices[i as usize];
                [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
            })
            .collect();
        // the four face normals, each opposite one vertex
        let faces = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
        let mut normals = [[0.0_f64; 3]; 4];
        for (f, idx) in faces.iter().enumerate() {
            let a = p[idx[0]];
            let b = p[idx[1]];
            let c = p[idx[2]];
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len == 0.0 {
                return 0.0;
            }
            normals[f] = [n[0] / len, n[1] / len, n[2] / len];
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d = normals[i][0] * normals[j][0]
                    + normals[i][1] * normals[j][1]
                    + normals[i][2] * normals[j][2];
                // the dihedral along the shared edge is π minus the angle
                // between the outward normals
                let angle = 180.0 - d.clamp(-1.0, 1.0).acos().to_degrees();
                if angle < worst {
                    worst = angle;
                }
            }
        }
    }
    worst
}

// ---------------------------------------------------------------------------
// scenes
// ---------------------------------------------------------------------------

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// Two tetrahedra sharing the triangle `F = (0,0,0) (4,0,0) (0,4,0)` in `z = 0`.
///
/// - The `z > 0` tetrahedron has apex `(0,0,1)`, so its longest edge is the
///   hypotenuse of `F` itself (`4√2 ≈ 5.657`).
/// - The `z < 0` tetrahedron has apex `(0,0,-20)`, so its longest edge is
///   `(4,0,0)–(0,0,-20)` (`≈ 20.396`), which is **not** an edge of `F`.
///
/// A threshold between `5.657` and `20.396` therefore splits an edge of the
/// shared face on one side and a different edge on the other: the minimal graded
/// scene.
fn two_tets_across_a_face() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]); // 0 : F
    mesh.vertices.push([4.0, 0.0, 0.0]); // 1 : F
    mesh.vertices.push([0.0, 4.0, 0.0]); // 2 : F
    mesh.vertices.push([0.0, 0.0, 1.0]); // 3 : near apex
    mesh.vertices.push([0.0, 0.0, -20.0]); // 4 : far apex
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 4],
    });
    mesh
}

// ---------------------------------------------------------------------------
// measurements
// ---------------------------------------------------------------------------

/// Control: the unrefined scenes are conforming, so a later non-conformity is
/// produced by the refinement and not carried in.
#[test]
fn the_scenes_start_conforming() {
    let box_mesh = kuhn_box(2, 2, 2, 1.0);
    let hanging = hanging_interior_faces(&box_mesh, [0.0; 3], [2.0; 3]);
    assert_eq!(
        hanging, 0,
        "the Kuhn box must be conforming before refinement, got {hanging} hanging faces"
    );

    let pair = two_tets_across_a_face();
    let (shared, once) = z0_faces(&pair);
    assert_eq!(
        (shared, once),
        (1, 0),
        "the two-tet scene must share exactly its one z = 0 face before refinement"
    );
    eprintln!(
        "  Kuhn box 2³: {} tets, min dihedral {:.3}°",
        box_mesh.tet_count(),
        min_dihedral_degrees(&box_mesh)
    );
}

/// Uniform refinement of a Kuhn box: every tetrahedron is congruent, so one
/// threshold splits them all on the same kind of edge and the mesh stays
/// conforming.
///
/// Reported alongside: the minimum dihedral angle per pass. The module doc says
/// "aspect ratio can drift"; this puts a number on the drift, which is what a
/// quality gate would have to bound.
#[test]
fn uniform_refinement_stays_conforming() {
    let mut mesh = kuhn_box(2, 2, 2, 1.0);
    eprintln!(
        "  pass 0: {:5} tets  max edge {:.4}  min dihedral {:7.3}°  hanging {}",
        mesh.tet_count(),
        mesh.max_edge_length(),
        min_dihedral_degrees(&mesh),
        hanging_interior_faces(&mesh, [0.0; 3], [2.0; 3])
    );
    for pass in 1..=6 {
        let target = mesh.max_edge_length() * 0.99;
        let ran = mesh
            .try_refine_conforming(target, 1)
            .expect("uniform refinement needs no propagation, so one pass always finishes");
        let hanging = hanging_interior_faces(&mesh, [0.0; 3], [2.0; 3]);
        eprintln!(
            "  pass {pass}: {:5} tets  max edge {:.4}  min dihedral {:7.3}°  hanging {hanging}  \
             (passes run {ran})",
            mesh.tet_count(),
            mesh.max_edge_length(),
            min_dihedral_degrees(&mesh),
        );
        assert_eq!(
            hanging, 0,
            "pass {pass}: uniform refinement produced {hanging} hanging interior faces"
        );
    }
}

/// Graded refinement — the only kind adaptivity needs — must leave the mesh
/// conforming, which means propagating each split to the neighbour that shares
/// the face.
///
/// # Before
///
/// It did not. One pass on the minimal graded scene left a `z = 0` face used by
/// a single tetrahedron, and `tests/hanging_node_effect.rs` measured what that
/// costs: the displacement field tears by 15% of itself on a *linear* exact
/// solution, because the coarse element contributes no force at the hanging node
/// and the shared face behaves traction-free from the fine side. That file keeps
/// the defect reproducible on a hand-built scene, so the measurement survives
/// this test going green.
#[test]
fn graded_refinement_stays_conforming() {
    // One pass is not enough any more, and that is the propagation working: the
    // neighbour learns about the split only once the midpoint exists, so it
    // splits on the pass after. A budget that sufficed without propagation can
    // fall short with it, which is exactly what the error variant is for.
    let mut cramped = two_tets_across_a_face();
    let cramped_result = cramped.try_refine_conforming(5.0, 1);
    let (shared, once) = z0_faces(&cramped);
    eprintln!(
        "  one pass only: {:?}, {} tets, z=0 faces shared {} / used once {}",
        cramped_result,
        cramped.tet_count(),
        shared,
        once
    );
    assert!(
        matches!(cramped_result, Err(RefineError::Unfinished { .. })),
        "with a single pass the propagation cannot reach the neighbour, and the refiner has to \
         say so rather than return a pass count that looks like a clean finish. Got \
         {cramped_result:?}"
    );

    let mut mesh = two_tets_across_a_face();
    // between 4√2 ≈ 5.657 (an edge of the shared face) and 20.396 (not)
    let passes = mesh
        .try_refine_conforming(5.0, 16)
        .expect("sixteen passes is ample for this scene");

    let (shared, once) = z0_faces(&mesh);
    eprintln!(
        "  after {passes} graded passes: {} tets, z=0 faces shared {} / used once {}, \
         min dihedral {:.3}°",
        mesh.tet_count(),
        shared,
        once,
        min_dihedral_degrees(&mesh)
    );

    assert_eq!(
        once, 0,
        "graded longest-edge bisection has to propagate the split to the neighbour that shares \
         the face; the census found {once} hanging face(s) at z = 0 against {shared} shared"
    );
    // `hanging_interior_faces` is deliberately *not* used here. It decides
    // "boundary" by asking whether all three vertices lie on one of six bounding
    // planes, which is exact for the Kuhn box it was written for and wrong for
    // this shape: the two tetrahedra meet at a slanted surface, so most of the
    // genuine boundary lies on no plane and comes back misclassified — 78 faces
    // when first tried here. `z0_faces` is the exact instrument for this scene,
    // because `z = 0` is the only plane with material on both sides.
    assert!(
        min_dihedral_degrees(&mesh) >= 8.0,
        "propagation must not cost more element quality than the mesher's own gate allows; \
         declared before measuring as 8°, the same figure `tests/mesh_quality.rs` holds \
         `generate_marching_tets` to, because the same solver consumes both. Measured {:.3}° — \
         if this is short, the answer is a different propagation rule, not a smaller number",
        min_dihedral_degrees(&mesh)
    );
}

/// What propagation costs in elements.
///
/// The only cost of making the refinement conforming is that it makes more
/// elements than the threshold alone asks for: a split reaches across the shared
/// face and the neighbour splits too, which can cascade. The number is worth
/// having because "more elements" is the argument someone will reach for when
/// proposing to skip it.
///
/// Measured against a reference refiner written here — the algorithm as it was
/// before propagation, kept as twenty lines in the test rather than as a second
/// code path in the crate, since its only remaining use is this comparison.
#[test]
fn propagation_costs_elements_and_the_count_is_bounded() {
    for (name, mesh) in [
        ("two tets (20:1 sizes)", two_tets_across_a_face()),
        ("Kuhn box 2³", kuhn_box(2, 2, 2, 1.0)),
    ] {
        for threshold in [5.0_f32, 1.2] {
            let mut conforming = mesh.clone();
            let Ok(passes) = conforming.try_refine_conforming(threshold, 24) else {
                continue;
            };
            let mut plain = mesh.clone();
            let plain_passes = refine_without_propagation(&mut plain, threshold, 24);
            let ratio = conforming.tet_count() as f64 / plain.tet_count().max(1) as f64;
            eprintln!(
                "  {name:<22} threshold {threshold:<5} | no propagation: {:>5} tets in {} passes \
                 | conforming: {:>5} tets in {} passes | ratio {ratio:.2} | \
                 min dihedral {:.3}° -> {:.3}°",
                plain.tet_count(),
                plain_passes,
                conforming.tet_count(),
                passes,
                min_dihedral_degrees(&plain),
                min_dihedral_degrees(&conforming),
            );
            assert!(
                ratio <= 4.0,
                "{name} at threshold {threshold}: propagation multiplied the element count by \
                 {ratio:.2}, past the 4 declared before measuring. The remedy is a different \
                 propagation rule (red-green closure), not a larger number here"
            );
            assert!(
                min_dihedral_degrees(&conforming) >= 8.0,
                "{name} at threshold {threshold}: {:.3}° is under the 8° declared before \
                 measuring",
                min_dihedral_degrees(&conforming)
            );
        }
    }
}

/// The refiner as it was before propagation: split each tetrahedron on its
/// longest edge if that edge is over the threshold, and leave the neighbour
/// alone.
///
/// Only used to measure what propagation costs. It leaves hanging nodes, which
/// is the defect `tests/hanging_node_effect.rs` measures the consequences of.
fn refine_without_propagation(mesh: &mut SdfTetMesh, max_edge_length: f32, max_passes: u32) -> u32 {
    let limit_sq = max_edge_length * max_edge_length;
    let mut passes = 0;
    for _ in 0..max_passes {
        let mut changed = false;
        let mut out: Vec<Tetrahedron> = Vec::with_capacity(mesh.tets.len() * 2);
        let mut mids: HashMap<(u32, u32), u32> = HashMap::new();
        let snapshot = std::mem::take(&mut mesh.tets);
        for tet in &snapshot {
            let vs = tet.vertices;
            let edges = [
                (vs[0], vs[1]),
                (vs[0], vs[2]),
                (vs[0], vs[3]),
                (vs[1], vs[2]),
                (vs[1], vs[3]),
                (vs[2], vs[3]),
            ];
            let mut best = (0_usize, 0.0_f32);
            for (k, (a, b)) in edges.iter().enumerate() {
                let pa = mesh.vertices[*a as usize];
                let pb = mesh.vertices[*b as usize];
                let d = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
                let len_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if len_sq > best.1 {
                    best = (k, len_sq);
                }
            }
            if best.1 <= limit_sq {
                out.push(*tet);
                continue;
            }
            changed = true;
            let (a, b) = edges[best.0];
            let key = (a.min(b), a.max(b));
            let mid = *mids.entry(key).or_insert_with(|| {
                let pa = mesh.vertices[a as usize];
                let pb = mesh.vertices[b as usize];
                let m = [
                    0.5 * (pa[0] + pb[0]),
                    0.5 * (pa[1] + pb[1]),
                    0.5 * (pa[2] + pb[2]),
                ];
                let idx = u32::try_from(mesh.vertices.len()).expect("fits");
                mesh.vertices.push(m);
                idx
            });
            let others: Vec<u32> = vs.iter().copied().filter(|v| *v != a && *v != b).collect();
            out.push(Tetrahedron {
                vertices: [a, mid, others[0], others[1]],
            });
            out.push(Tetrahedron {
                vertices: [mid, b, others[0], others[1]],
            });
        }
        mesh.tets = out;
        if !changed {
            break;
        }
        passes += 1;
    }
    passes
}
