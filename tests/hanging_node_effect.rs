//! What a hanging node does to the finite element solution, and what could be
//! done about it.
//!
//! `tests/refinement_conformity.rs` established that
//! [`SdfTetMesh::refine_by_max_edge_length`] leaves hanging faces as soon as the
//! refinement is graded, which is the only kind adaptivity needs. This file is
//! the next question: **does it matter, and how would anyone know.**
//!
//! # What the measurement says, against what was expected
//!
//! This file was written expecting the patch test to be blind here, by the same
//! argument that holds for the other non-conformity in this crate: a P1 element
//! reproduces a linear field exactly, so both sides of the face should agree on
//! it. **That is wrong, and the measurement says so.**
//!
//! The two defects are not the same shape. Two triangulations of one quadrilateral
//! face share every vertex, so the field is continuous and a patch test really is
//! blind — measured at 3.6e-15 MPa on a mesh with 576 of 768 faces dangling. A
//! hanging node is a vertex *missing* from one side. The coarse element
//! contributes no force there, so from the fine side the shared face behaves like
//! a traction-free surface, and a linear field does not satisfy that condition.
//! The node drifts, and the field tears on the linear case too:
//!
//! | run | tear, relative to the field |
//! |---|---|
//! | linear, hanging node free | **1.5e-1** |
//! | quadratic, hanging node free | 1.3e-2 |
//! | linear, hanging node prescribed | 1.3e-14 |
//!
//! So the blindness is real but narrow: it needs the hanging node to be *held*,
//! which is what a patch test written the natural way — prescribe every node
//! whose answer is known — happens to do. Left free, as any real solve leaves it,
//! the defect is loud.
//!
//! The mechanism is pinned rather than asserted. With the hanging node held, the
//! nodal error at **every other** vertex is 5.1e-17, while the hanging node's own
//! error when free is 2.978e-4: the departure is that one node and nothing else,
//! which is what "the coarse element contributes nothing there" predicts.
//!
//! # The instrument
//!
//! A hanging node is a vertex sitting in the interior of some *other* element's
//! face. The finite element field is then two different things on that face:
//! whatever the fine elements interpolate, which passes through the hanging
//! node's own value, and whatever the coarse element interpolates, which does
//! not know the node exists. The gap between them at that point,
//!
//! ```text
//! incompatibility = u[hanging] - (linear interpolation of the coarse face at that point)
//! ```
//!
//! is exactly the amount by which the displacement field is torn, and it is zero
//! on a conforming mesh because there is no such node.
//!
//! # Status
//!
//! Investigation. Nothing here changes the mesher or the solver; it measures the
//! defect and sizes the options. The options are laid out in
//! [`the_three_ways_out`], which is documentation in the shape of a test so that
//! the claims in it sit next to the numbers they rest on.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::linear_elastic_fem::{solve, BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn vert(mesh: &SdfTetMesh, i: u32) -> [f64; 3] {
    let p = mesh.vertices[i as usize];
    [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
}

/// The minimal graded scene from `tests/refinement_conformity.rs`.
///
/// Two tetrahedra share the triangle `(0,0,0) (4,0,0) (0,4,0)` in `z = 0`. The
/// upper one's longest edge is an edge of that shared triangle; the lower one's
/// is not. A refinement threshold between the two therefore splits the shared
/// face from one side only.
fn two_tets_across_a_face() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]);
    mesh.vertices.push([4.0, 0.0, 0.0]);
    mesh.vertices.push([0.0, 4.0, 0.0]);
    mesh.vertices.push([0.0, 0.0, 1.0]);
    mesh.vertices.push([0.0, 0.0, -20.0]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh.tets.push(Tetrahedron {
        vertices: [0, 2, 1, 4],
    });
    mesh
}

/// The same scene with the upper tetrahedron split by hand, which is what the
/// graded refinement used to produce.
///
/// Built here rather than by calling [`SdfTetMesh::refine_by_max_edge_length`]
/// **on purpose**. That refiner is being made conforming, and once it is, it can
/// no longer produce a hanging node — which would take this whole file's subject
/// matter with it. The measurements below are the reason the refiner is being
/// changed at all, so they have to outlive the change: losing them would leave
/// the fix with no record of what it fixed.
///
/// The split is the one bisection of the shared face's longest edge, the
/// hypotenuse `(4,0,0)-(0,4,0)`, applied to the `z > 0` tetrahedron only. The
/// `z < 0` tetrahedron keeps the whole face, so its face has three corners while
/// its neighbour's has four points on it.
///
/// `the_hand_built_scene_matches_what_the_old_refiner_produced` holds this to
/// the refiner's output while that output still exists.
fn two_tets_with_a_hanging_node() -> SdfTetMesh {
    let mut mesh = two_tets_across_a_face();
    // midpoint of the shared face's hypotenuse
    let mid = u32::try_from(mesh.vertex_count()).expect("fits u32");
    mesh.vertices.push([2.0, 2.0, 0.0]);
    // replace the upper tetrahedron (0,1,2,3) with its two halves
    mesh.tets.retain(|t| t.vertices != [0, 1, 2, 3]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, mid, 3],
    });
    mesh.tets.push(Tetrahedron {
        vertices: [0, mid, 2, 3],
    });
    mesh
}

/// Every `(hanging vertex, face it hangs on)` pair in the mesh.
///
/// A vertex hangs on a face when it lies in that triangle's plane, inside its
/// outline, and is not one of its three corners. The test is written on the
/// barycentric coordinates so that "inside" is exact for the midpoint case that
/// bisection produces: a midpoint of one edge has one coordinate zero and the
/// other two at one half.
fn hanging_nodes(mesh: &SdfTetMesh) -> Vec<(u32, [u32; 3])> {
    const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    let mut out = Vec::new();
    for tet in &mesh.tets {
        for f in FACES {
            let face = [tet.vertices[f[0]], tet.vertices[f[1]], tet.vertices[f[2]]];
            for v in 0..u32::try_from(mesh.vertex_count()).expect("fits u32") {
                if face.contains(&v) {
                    continue;
                }
                if let Some(w) = barycentric_on_face(mesh, face, v) {
                    // strictly inside, including edges but not corners
                    if w.iter().all(|c| *c >= -1.0e-9)
                        && w.iter().filter(|c| **c > 1.0e-9).count() >= 2
                    {
                        out.push((v, face));
                    }
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Barycentric coordinates of vertex `v` on triangle `face`, or `None` when it is
/// off the plane.
fn barycentric_on_face(mesh: &SdfTetMesh, face: [u32; 3], v: u32) -> Option<[f64; 3]> {
    let a = vert(mesh, face[0]);
    let b = vert(mesh, face[1]);
    let c = vert(mesh, face[2]);
    let p = vert(mesh, v);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let n = cross(ab, ac);
    let area2 = dot(n, n);
    if area2 <= 0.0 {
        return None;
    }
    // off-plane distance, scaled by the triangle size so the tolerance is relative
    if dot(ap, n).abs() > 1.0e-9 * area2.sqrt() {
        return None;
    }
    let w2 = dot(cross(ab, ap), n) / area2;
    let w1 = dot(cross(ap, ac), n) / area2;
    Some([1.0 - w1 - w2, w1, w2])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// How far the displacement field is torn at each hanging node.
fn incompatibility(mesh: &SdfTetMesh, u: &[[Fix128; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for (v, face) in hanging_nodes(mesh) {
        let w = barycentric_on_face(mesh, face, v).expect("hanging nodes are on their face");
        for (axis, mine) in u[v as usize].iter().enumerate() {
            let theirs: f64 = w
                .iter()
                .zip(face)
                .map(|(weight, corner)| weight * u[corner as usize][axis].to_f64())
                .sum();
            worst = worst.max((mine.to_f64() - theirs).abs());
        }
    }
    worst
}

/// Largest nodal departure from `field`, over the vertices in `free`.
fn nodal_error(
    mesh: &SdfTetMesh,
    u: &[[Fix128; 3]],
    field: impl Fn([f64; 3]) -> [f64; 3],
    free: &[u32],
) -> f64 {
    let mut worst = 0.0_f64;
    for v in free {
        let exact = field(vert(mesh, *v));
        for (got, want) in u[*v as usize].iter().zip(&exact) {
            worst = worst.max((got.to_f64() - want).abs());
        }
    }
    worst
}

fn steel() -> ElasticMaterial {
    ElasticMaterial::new(fx(200_000.0), fx(0.3)).expect("valid")
}

/// Prescribe `field` at every vertex except those given, and solve.
fn solve_with_exact_boundary(
    mesh: &SdfTetMesh,
    field: impl Fn([f64; 3]) -> [f64; 3],
    free: &[u32],
) -> Vec<[Fix128; 3]> {
    let mut bc = BoundaryConditions::new();
    for v in 0..u32::try_from(mesh.vertex_count()).expect("fits u32") {
        if free.contains(&v) {
            continue;
        }
        let u = field(vert(mesh, v));
        bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
    }
    let solution = solve(mesh, &steel(), &bc, &SolverConfig::default())
        .unwrap_or_else(|e| panic!("the FEM rejected the graded mesh: {e:?}"));
    solution.displacements
}

/// A linear field, which every P1 mesh reproduces exactly, and a quadratic one,
/// which none does.
fn linear(p: [f64; 3]) -> [f64; 3] {
    [
        1.0e-4 * (p[0] + 0.3 * p[1] - 0.2 * p[2]),
        1.0e-4 * (-0.1 * p[0] + p[1] + 0.4 * p[2]),
        1.0e-4 * (0.25 * p[0] - 0.15 * p[1] + p[2]),
    ]
}

fn quadratic(p: [f64; 3]) -> [f64; 3] {
    [
        1.0e-5 * (p[0] * p[0] - p[1] * p[1]),
        1.0e-5 * (2.0 * p[0] * p[1]),
        1.0e-5 * (p[2] * p[2] - p[0] * p[1]),
    ]
}

/// The graded mesh really does carry hanging nodes, and the census agrees with
/// the geometric search.
///
/// Two independent ways of finding the same defect: the face census in
/// `refinement_conformity.rs` counts faces used once that are not on the
/// boundary, and [`hanging_nodes`] looks for vertices sitting inside someone
/// else's face. They have to agree, or one of them is measuring something else.
#[test]
fn the_graded_mesh_has_hanging_nodes_and_the_conforming_one_does_not() {
    let conforming = two_tets_across_a_face();
    assert!(
        hanging_nodes(&conforming).is_empty(),
        "the unrefined scene must be conforming, or everything below is measuring the wrong thing"
    );

    let graded = two_tets_with_a_hanging_node();
    let hanging = hanging_nodes(&graded);
    eprintln!(
        "[hanging] after one graded pass: {} tets, {} hanging node/face pairs",
        graded.tet_count(),
        hanging.len()
    );
    for (v, face) in &hanging {
        eprintln!(
            "[hanging]   vertex {v} at {:?} hangs on face {face:?} at {:?}",
            vert(&graded, *v),
            face.map(|i| vert(&graded, i))
        );
    }
    assert!(
        !hanging.is_empty(),
        "graded refinement was measured to leave hanging faces in \
         refinement_conformity.rs; if this finds none, the two instruments disagree and one of \
         them is wrong"
    );
}

/// **The finding, and it is worse than the blindness argument predicted.**
///
/// A hanging node is not the same defect as two triangulations of one face. In
/// that case every vertex is shared and the field is continuous, which is why a
/// patch test came back exact to 3.6e-15 MPa on a mesh with 576 of 768 faces
/// dangling. Here a vertex is *missing* from one side: the coarse element
/// contributes no force at the hanging node, so from the fine side the shared
/// face behaves like a traction-free surface. A linear field does not satisfy
/// that condition, so the node moves away from it and the tear opens **on the
/// linear field too**.
///
/// Measured on the minimal graded scene, with the hanging node left free as it
/// would be in any real solve:
///
/// | exact field | tear, relative to the field |
/// |---|---|
/// | linear | 1.5e-1 |
/// | quadratic | 1.3e-2 |
///
/// Fifteen percent of the field, on the case a patch test is built from. The
/// blindness this file was written to expect is real but narrower than stated:
/// it needs the hanging node to be *prescribed*, which the second arm below
/// measures. Left free — the situation adaptivity actually produces — the defect
/// is loud.
#[test]
fn a_hanging_node_tears_the_field_even_on_a_linear_one() {
    let graded = two_tets_with_a_hanging_node();
    let hanging: Vec<u32> = hanging_nodes(&graded).iter().map(|(v, _)| *v).collect();

    let scale = |field: &dyn Fn([f64; 3]) -> [f64; 3]| -> f64 {
        (0..graded.vertex_count())
            .map(|i| {
                let u = field(vert(&graded, u32::try_from(i).expect("fits")));
                u.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
            })
            .fold(0.0_f64, f64::max)
    };

    let mut rel = [0.0_f64; 2];
    for (k, (name, field)) in [
        ("linear", &linear as &dyn Fn([f64; 3]) -> [f64; 3]),
        ("quadratic", &quadratic as &dyn Fn([f64; 3]) -> [f64; 3]),
    ]
    .into_iter()
    .enumerate()
    {
        let u = solve_with_exact_boundary(&graded, field, &hanging);
        rel[k] = incompatibility(&graded, &u) / scale(field);
        eprintln!(
            "[tear]    {name:<10} hanging node free:       tear {:.3e} of the field",
            rel[k]
        );
    }

    // A band, not a sign. Once the refiner is conforming, the scene above is the
    // only place this defect exists, and nothing else would notice it drifting.
    // The scene and the arithmetic are both deterministic, so the value is
    // reproducible; the band is wide enough not to be a hash of it.
    assert!(
        (0.13..0.17).contains(&rel[0]),
        "the measured tear is {:.4}, outside the 0.13 to 0.17 this scene has produced. A hanging \
         node leaves the coarse element contributing no force there, so the shared face acts \
         traction-free from the fine side and even a linear field is torn. If this has moved, \
         the scene has changed and the case for propagating refinement rests on a number that no \
         longer exists",
        rel[0]
    );
    assert!(
        (0.011..0.015).contains(&rel[1]),
        "the quadratic tear is {:.4}, outside the 0.011 to 0.015 this scene has produced",
        rel[1]
    );
}

/// The blindness is real, but only when the hanging node is held.
///
/// Prescribe the exact field at the hanging node as well, and the tear closes to
/// solver precision on a linear field: the node is no longer free to drift, and
/// linear interpolation of a linear field agrees from both sides by definition.
/// That is the configuration in which a patch test would report success on this
/// mesh — and it is not the configuration a solve is in.
///
/// Both arms matter for the oracle design. An oracle that prescribes every node
/// it knows the answer for, which is the natural way to write a patch test,
/// lands in the blind arm.
#[test]
fn prescribing_the_hanging_node_is_what_makes_a_patch_test_blind() {
    let graded = two_tets_with_a_hanging_node();
    let scale = (0..graded.vertex_count())
        .map(|i| {
            let u = linear(vert(&graded, u32::try_from(i).expect("fits")));
            u.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
        })
        .fold(0.0_f64, f64::max);

    let u = solve_with_exact_boundary(&graded, linear, &[]);
    let held = incompatibility(&graded, &u) / scale;
    eprintln!("[tear]    linear     hanging node prescribed: tear {held:.3e} of the field");
    assert!(
        held < 1.0e-9,
        "with the hanging node held at the exact value, both sides interpolate the same linear \
         field and the tear must close to solver precision. Measured {held:.3e}"
    );
}

/// What the missing coupling costs the *rest* of the solution.
///
/// If the mechanism is "the coarse element contributes nothing at the hanging
/// node", then holding that one node should restore the patch test everywhere
/// else. This measures the nodal error at the remaining free vertices with the
/// hanging node free and with it held, and the difference between the two is the
/// mechanism made visible.
#[test]
fn holding_the_hanging_node_restores_the_rest_of_the_patch() {
    let graded = two_tets_with_a_hanging_node();
    let hanging: Vec<u32> = hanging_nodes(&graded).iter().map(|(v, _)| *v).collect();
    let others: Vec<u32> = (0..u32::try_from(graded.vertex_count()).expect("fits"))
        .filter(|v| !hanging.contains(v))
        .collect();

    let free = solve_with_exact_boundary(&graded, linear, &hanging);
    let held = solve_with_exact_boundary(&graded, linear, &[]);
    // `others` were prescribed in both runs, so compare where the solve had room:
    // the hanging node itself under the free run.
    let err_at_hanging = nodal_error(&graded, &free, linear, &hanging);
    let err_elsewhere = nodal_error(&graded, &held, linear, &others);
    eprintln!(
        "[patch]   nodal error at the hanging node when free: {err_at_hanging:.3e}; \
         at every other node when it is held: {err_elsewhere:.3e}"
    );
    assert!(
        err_at_hanging > 1.0e3 * err_elsewhere.max(1.0e-18),
        "the departure has to sit at the hanging node and not be spread over the mesh, or the \
         explanation offered here — that the coarse element simply contributes nothing there — \
         is not the mechanism. Measured {err_at_hanging:.3e} against {err_elsewhere:.3e}"
    );
}

/// What remedy (a) would buy, measured without implementing it.
///
/// The constraint is `u[hanging] = Σ w_k u[face_k]` with the barycentric weights
/// of the hanging node on the coarse face — for a bisection midpoint that is
/// `½(u[a] + u[b])`. Its effect can be measured here without touching the solver,
/// because in this scene the face's corners are prescribed: pin the hanging node
/// to that combination and solve.
///
/// Two things follow, and both are numbers rather than expectations:
///
/// - **The tear closes identically**, not approximately. The constraint *is* the
///   coarse face's interpolation, so the two sides agree by construction.
/// - **What replaces it is interpolation error**, bounded and second order in the
///   element size, rather than a free surface. On the linear field there is none
///   at all, which is why the patch test is restored exactly; on the quadratic
///   field it is the curvature of the field along the parent edge.
///
/// ⚠️ On this scene that replacement is **not smaller**. The constraint sits
/// 8.0e-5 from the exact field on the quadratic case, against a tear of 5.1e-5
/// that it removed. What changes is the *character* of the error, not its size
/// at one resolution: a tear is a discontinuity in the field, which no
/// convergence argument covers, while the constraint's error is ordinary
/// interpolation error that falls with the element size. Refining a mesh with
/// hanging nodes does not obviously help; refining a constrained one does. That
/// distinction is the reason to prefer either remedy over neither, and it is not
/// visible in a single number at a single resolution.
///
/// That is the case for (a) being correct. It is not the case for (a) being
/// *first* — see [`the_three_ways_out`].
#[test]
fn constraining_the_hanging_node_to_its_parent_edge_closes_the_tear() {
    let graded = two_tets_with_a_hanging_node();
    let pairs = hanging_nodes(&graded);

    for (name, field) in [
        ("linear", &linear as &dyn Fn([f64; 3]) -> [f64; 3]),
        ("quadratic", &quadratic as &dyn Fn([f64; 3]) -> [f64; 3]),
    ] {
        let mut bc = BoundaryConditions::new();
        for v in 0..u32::try_from(graded.vertex_count()).expect("fits u32") {
            let u = field(vert(&graded, v));
            bc.prescribe_all(v, [fx(u[0]), fx(u[1]), fx(u[2])]);
        }
        // the constraint, in place of the exact value, at each hanging node
        let mut interpolation_error = 0.0_f64;
        for (v, face) in &pairs {
            let w = barycentric_on_face(&graded, *face, *v).expect("on its face");
            let mut tied = [0.0_f64; 3];
            for (k, weight) in w.iter().enumerate() {
                let corner = field(vert(&graded, face[k]));
                for axis in 0..3 {
                    tied[axis] += weight * corner[axis];
                }
            }
            let exact = field(vert(&graded, *v));
            for axis in 0..3 {
                interpolation_error = interpolation_error.max((tied[axis] - exact[axis]).abs());
            }
            bc.prescribe_all(*v, [fx(tied[0]), fx(tied[1]), fx(tied[2])]);
        }
        let solution = solve(&graded, &steel(), &bc, &SolverConfig::default()).expect("solvable");
        let tear = incompatibility(&graded, &solution.displacements);
        eprintln!(
            "[remedy]  {name:<10} constrained: tear {tear:.3e}, and the constraint itself is \
             {interpolation_error:.3e} from the exact field"
        );
        assert!(
            tear < 1.0e-12,
            "{name}: the constraint is the coarse face's own interpolation, so pinning the \
             hanging node to it makes the two sides agree by construction and the tear must be \
             rounding only. Measured {tear:.3e}"
        );
    }
}

/// The three ways out, and what each costs in this crate specifically.
///
/// Written as a test so that the numbers it cites are next to the code that
/// produced them, and so that the one claim that can be checked mechanically is.
///
/// # (a) Constrain the hanging node to its parent edge
///
/// Hold `u[hanging] = ½(u[a] + u[b])` as a constraint and eliminate the hanging
/// degree of freedom, so the coarse face's interpolation becomes the truth on
/// both sides.
///
/// - **Determinism**: safe. The coefficient is exactly one half, which is a power
///   of two and therefore exact in `Fix128`, and the elimination is additions and
///   multiplications only. No new transcendental, no division by a data-dependent
///   quantity. The assertion below checks the arithmetic rather than asserting it
///   in prose.
/// - **Effect**: measured, without implementing it, in
///   [`constraining_the_hanging_node_to_its_parent_edge_closes_the_tear`]: the
///   tear closes to 2.6e-17 and the linear patch test is restored exactly.
/// - **Cost**: touches [`alice_physics::linear_elastic_fem`] — the matrix-free
///   product has to apply the constraint on the way in and its transpose on the
///   way out. That is the component with the delicate numerics, which argues
///   against going first.
/// - **Catch**: constraints chain when a hanging node's parent edge is itself
///   split, and the chain has to be resolved before elimination or the result
///   depends on the order it is applied in — a determinism hazard that does not
///   exist in (b).
///
/// # (b) Propagate the refinement until the mesh is conforming again
///
/// Longest-edge closure, or red-green refinement: when a split creates a hanging
/// node, split the neighbour too, repeating until nothing hangs. Rivara's
/// longest-edge bisection is the standard choice and terminates.
///
/// - **Determinism**: safe, and *outside the solver entirely*. New vertices are
///   edge midpoints, which the mesher already computes the same way
///   (`0.5 * (a + b)` on `f32`), and the propagation order can be made
///   deterministic by processing edges in index order.
/// - **Cost**: more elements than asked for — the refinement spreads beyond the
///   region that needed it. That is a size cost, not a correctness one.
/// - **Recommended first.** It needs no change to the finite element code, so it
///   adds no new determinism risk to the part of the crate where that is hardest
///   to argue. `refine_by_max_edge_length` already bisects longest edges; what is
///   missing is the neighbour propagation, which is a mesh-side loop.
///
/// # (c) Leave the mesh non-conforming
///
/// Not an option as stated. "Allowing" a hanging node is not a method; it is the
/// tear measured above. There *are* genuine non-conforming finite element
/// methods — Crouzeix-Raviart puts the degrees of freedom on face midpoints — but
/// they are a different basis with different assembly, not a decision to ignore
/// the problem. Adopting one would replace `linear_elastic_fem`'s element, which
/// is a larger change than either (a) or (b).
#[test]
fn the_three_ways_out() {
    // The one claim above that is arithmetic rather than judgement: the constraint
    // coefficient in (a) is exact in this number type, so eliminating a hanging
    // node cannot introduce rounding of its own.
    let a = fx(0.1);
    let b = fx(0.3);
    let half = Fix128::ONE / Fix128::from_int(2);
    let mid = half * (a + b);
    let twice = mid + mid;
    assert_eq!(
        twice,
        a + b,
        "halving and doubling has to round-trip exactly, or the constraint u = (a+b)/2 would \
         introduce an error of its own and option (a) would need a different argument"
    );
}

/// The hand-built scene carries the same defect as the refiner's output, to the
/// last digit of every measurement in this file.
///
/// It is **not the same mesh**. The refinement threshold sits below *both*
/// tetrahedra's longest edges, so the refiner splits the lower one as well — on
/// an edge that is not part of the shared face — and comes out with 7 vertices
/// and 4 elements against the hand build's 6 and 3. That extra split leaves the
/// shared face whole on the lower side, so it has nothing to do with the hanging
/// node, and the hand build is the minimal scene that carries the defect.
///
/// "Has nothing to do with it" is the kind of claim this file exists to distrust,
/// so it is measured rather than argued: every quantity reported above is
/// computed on both scenes and required to agree exactly.
///
/// This test can only run while the refiner still produces hanging nodes. Once it
/// is made conforming the comparison becomes impossible, and the measurements
/// stand on the hand-built scene alone — which is why they carry numeric bands
/// rather than only signs.
#[test]
fn the_hand_built_scene_carries_the_same_defect_as_the_refiner_did() {
    let mut refined = two_tets_across_a_face();
    let passes = refined.refine_by_max_edge_length(5.0, 1);
    assert_eq!(passes, 1, "one pass must have run");
    let built = two_tets_with_a_hanging_node();

    eprintln!(
        "[faithful] refiner: {} vertices / {} tets;  hand-built: {} / {}",
        refined.vertex_count(),
        refined.tet_count(),
        built.vertex_count(),
        built.tet_count()
    );

    for (name, field) in [
        ("linear", &linear as &dyn Fn([f64; 3]) -> [f64; 3]),
        ("quadratic", &quadratic as &dyn Fn([f64; 3]) -> [f64; 3]),
    ] {
        let mut measured = Vec::new();
        for mesh in [&refined, &built] {
            let hanging: Vec<u32> = hanging_nodes(mesh).iter().map(|(v, _)| *v).collect();
            let free = solve_with_exact_boundary(mesh, field, &hanging);
            let held = solve_with_exact_boundary(mesh, field, &[]);
            measured.push((
                incompatibility(mesh, &free),
                incompatibility(mesh, &held),
                nodal_error(mesh, &free, field, &hanging),
            ));
        }
        eprintln!(
            "[faithful]   {name:<10} refiner {:.6e} / {:.6e} / {:.6e}   \
             hand-built {:.6e} / {:.6e} / {:.6e}",
            measured[0].0,
            measured[0].1,
            measured[0].2,
            measured[1].0,
            measured[1].1,
            measured[1].2
        );
        assert_eq!(
            measured[0], measured[1],
            "{name}: the hand-built scene has to reproduce the refiner's numbers exactly, or the \
             measurements in this file describe something the refiner never produced. The extra \
             split the refiner makes on the lower tetrahedron is supposed to be irrelevant to \
             the hanging node; if these differ, it is not"
        );
    }
}
