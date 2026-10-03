//! Closed-form and parity oracles for the `cubic_elastic_fem` mesh-topology
//! helpers: `node_count`, `corner_count`, `edge_node_count`, `face_node_count`,
//! `element_count`, `edge_nodes`, `face_node`, `element_nodes`, `shape_values`
//! and `interior_node_positions_are_exact`.
//!
//! These are pure queries over a mesh `CubicMesh::from_tet_mesh` already
//! built, so every oracle here is either a **closed form** derived from the
//! mesh's own topology (a single tetrahedron has 4 corners, `C(4,2) = 6`
//! edges, 4 faces, and the module's `EDGES`/`FACES` tables fix which slot each
//! one lands in — see the `cubic_elastic_fem` module header) or a **parity**
//! check between two different public calls that must agree by construction
//! (an edge or a face shared by two tetrahedra gets exactly one node, and
//! `edge_nodes`/`face_node` must report the same slot `element_nodes` does
//! for both of them). None of the expected values below comes from calling
//! the function under test.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeSet;

use alice_physics::cubic_elastic_fem::{shape_values, CubicMesh};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Raw two's-complement integer behind a `Fix128`; a difference in it is one
/// ulp.
fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | (x.lo as i128)
}

/// One tetrahedron, corners `(0,0,0)`, `(s,0,0)`, `(0,s,0)`, `(0,0,s)` mm, as
/// tetrahedral-mesh vertices `0..4` in that order.
///
/// With this corner order `tet.vertices == [0, 1, 2, 3]`, the identity
/// permutation, so every `EDGES`/`FACES` pair in the module header is read
/// off the global corner numbers directly — which is what turns the node
/// layout asserted below into a closed form instead of a re-run of the
/// builder.
fn single_tet_mesh(s: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]);
    mesh.vertices.push([s as f32, 0.0, 0.0]);
    mesh.vertices.push([0.0, s as f32, 0.0]);
    mesh.vertices.push([0.0, 0.0, s as f32]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh
}

/// Two tetrahedra glued on the face `(1,2,3)`: corners `0..5`, with corner 4
/// the apex of the second one.
///
/// `tet0 = [0,1,2,3]`, `tet1 = [1,2,3,4]`, so the two share exactly the three
/// edges `(1,2)`, `(1,3)`, `(2,3)` and the one face `(1,2,3)`. Independently:
/// `tet0` has `C(4,2) = 6` edges and 4 faces, `tet1` has 6 edges and 4 faces,
/// and 3 edges + 1 face are shared, so the mesh has `6+6-3 = 9` distinct edges
/// and `4+4-1 = 7` distinct faces — the closed form
/// `the_two_shared_tets_add_up_to_the_independently_counted_edges_and_faces`
/// checks against, and every other test in this file that needs a mesh with
/// sharing uses.
fn two_tet_shared_face_mesh() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]); // 0
    mesh.vertices.push([1.0, 0.0, 0.0]); // 1
    mesh.vertices.push([0.0, 1.0, 0.0]); // 2
    mesh.vertices.push([0.0, 0.0, 1.0]); // 3
    mesh.vertices.push([1.0, 1.0, 1.0]); // 4, the apex of tet1
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh.tets.push(Tetrahedron {
        vertices: [1, 2, 3, 4],
    });
    mesh
}

/// Distinct edges and faces of a tetrahedral mesh, counted independently of
/// `CubicMesh`'s own bookkeeping — straight off the raw corner-index lists.
fn count_edges_and_faces(mesh: &SdfTetMesh) -> (usize, usize) {
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut faces: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
    for tet in &mesh.tets {
        for a in 0..4 {
            for b in (a + 1)..4 {
                let (p, q) = (tet.vertices[a], tet.vertices[b]);
                edges.insert(if p <= q { (p, q) } else { (q, p) });
            }
        }
        for skip in 0..4 {
            let mut tri: Vec<u32> = (0..4)
                .filter(|&i| i != skip)
                .map(|i| tet.vertices[i])
                .collect();
            tri.sort_unstable();
            faces.insert((tri[0], tri[1], tri[2]));
        }
    }
    (edges.len(), faces.len())
}

// ---------------------------------------------------------------------------
// closed-form counts
// ---------------------------------------------------------------------------

/// `corner_count`, `edge_node_count`, `face_node_count`, `element_count` and
/// `node_count` on the one shape whose topology needs no counting at all: a
/// single tetrahedron has exactly 4 corners, `C(4,2) = 6` edges and 4 faces,
/// so a cubic element on it has `4 + 2*6 + 4 = 20` nodes and one element.
#[test]
fn single_tet_counts_match_the_closed_form_tetrahedron_topology() {
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("one non-degenerate tet is well formed");

    assert_eq!(c.corner_count(), 4, "a tetrahedron has 4 corners");
    assert_eq!(c.edge_node_count(), 12, "C(4,2) = 6 edges, 2 nodes each");
    assert_eq!(c.face_node_count(), 4, "4 faces, 1 node each");
    assert_eq!(c.element_count(), 1, "one tetrahedron in the source mesh");
    assert_eq!(c.node_count(), 20, "4 + 12 + 4");
}

/// The two-tet shared-face mesh's counts against the closed form derived from
/// inclusion-exclusion over the shared face, independent of
/// `CubicMesh::from_tet_mesh`'s own bookkeeping.
#[test]
fn two_shared_tets_add_up_to_the_independently_counted_edges_and_faces() {
    let mesh = two_tet_shared_face_mesh();
    let (edges, faces) = count_edges_and_faces(&mesh);
    assert_eq!(
        edges, 9,
        "6 + 6 - 3 shared edges, counted off the raw corner lists"
    );
    assert_eq!(
        faces, 7,
        "4 + 4 - 1 shared face, counted off the raw corner lists"
    );

    let c = CubicMesh::from_tet_mesh(&mesh).expect("two tets glued on a face are well formed");
    assert_eq!(c.corner_count(), 5);
    assert_eq!(c.element_count(), 2);
    assert_eq!(
        c.edge_node_count(),
        2 * edges,
        "sharing must not duplicate an edge's two nodes"
    );
    assert_eq!(
        c.face_node_count(),
        faces,
        "sharing must not duplicate a face's one node"
    );
    assert_eq!(c.node_count(), 5 + 2 * edges + faces);
}

// ---------------------------------------------------------------------------
// element_nodes: closed form on the single tet, degenerate indices
// ---------------------------------------------------------------------------

/// With `tet.vertices == [0,1,2,3]` the identity permutation, every edge and
/// face node is created, in order, on first appearance in `EDGES`/`FACES`
/// order — so element 0's twenty node indices are `0..20` itself, corners
/// first. This is the module's own numbering rule traced by hand, not the
/// output of `element_nodes`.
#[test]
fn element_nodes_is_the_identity_permutation_on_a_single_tet() {
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    let expected: [u32; 20] = core::array::from_fn(|i| i as u32);
    assert_eq!(c.element_nodes(0), Some(expected));
}

#[test]
fn element_nodes_is_none_past_the_element_count() {
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    assert_eq!(c.element_count(), 1);
    assert_eq!(c.element_nodes(1), None, "there is no second element");
    assert_eq!(
        c.element_nodes(usize::MAX),
        None,
        "an absurd index must return None, not panic or wrap"
    );
}

/// Parity: every slot of `element_nodes(t)` for an edge pair must equal
/// `edge_nodes` of the same two corners, and the same for every face slot
/// against `face_node` — checked on the mesh with sharing, so a node reused
/// across tetrahedra has to agree from both directions.
#[test]
fn element_nodes_slots_agree_with_edge_nodes_and_face_node_on_both_shared_tets() {
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];

    let mesh = two_tet_shared_face_mesh();
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");

    for (t, tet) in mesh.tets.iter().enumerate() {
        let nodes = c
            .element_nodes(t)
            .unwrap_or_else(|| panic!("element {t} exists"));
        for (e, &(i, j)) in EDGES.iter().enumerate() {
            let (a, b) = (tet.vertices[i], tet.vertices[j]);
            let pair = c
                .edge_nodes(a, b)
                .unwrap_or_else(|| panic!("edge ({a},{b}) exists"));
            assert_eq!(
                (nodes[4 + 2 * e], nodes[5 + 2 * e]),
                pair,
                "tet {t} edge slot {e} ({a},{b}) must agree with edge_nodes"
            );
        }
        for (f, &(i, j, k)) in FACES.iter().enumerate() {
            let (a, b, k_) = (tet.vertices[i], tet.vertices[j], tet.vertices[k]);
            let node = c
                .face_node(a, b, k_)
                .unwrap_or_else(|| panic!("face ({a},{b},{k_}) exists"));
            assert_eq!(
                nodes[16 + f],
                node,
                "tet {t} face slot {f} ({a},{b},{k_}) must agree with face_node"
            );
        }
    }

    // The shared face's node must be the very same index from both tets'
    // element_nodes, and the shared edges likewise — not merely equal values
    // that happen to coincide, but the one global node both tets reference.
    let n0 = c.element_nodes(0).unwrap();
    let n1 = c.element_nodes(1).unwrap();
    // tet0 local face (1,2,3) is slot 16; tet1 local face (0,1,2) -> global
    // (1,2,3) is slot 19.
    assert_eq!(
        n0[16], n1[19],
        "the face shared by both tets must be exactly one node, referenced identically"
    );
}

// ---------------------------------------------------------------------------
// edge_nodes / face_node: closed-form slot layout, order independence,
// degenerate (absent) queries
// ---------------------------------------------------------------------------

/// Edge `e` of `EDGES` owns slots `4+2e` (nearer the lower-numbered corner)
/// and `5+2e`, on the single tet where corner numbers equal global indices.
#[test]
fn edge_nodes_matches_the_edges_table_slot_layout_in_both_argument_orders() {
    const EDGES: [(u32, u32); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    for (e, &(a, b)) in EDGES.iter().enumerate() {
        let want = (4 + 2 * e as u32, 5 + 2 * e as u32);
        assert_eq!(c.edge_nodes(a, b), Some(want), "edge ({a},{b})");
        assert_eq!(
            c.edge_nodes(b, a),
            Some(want),
            "edge ({b},{a}) must match ({a},{b})"
        );
    }
}

/// Face `f` of `FACES` owns slot `16+f`, on the single tet, in any of the
/// three rotations of its argument order.
#[test]
fn face_node_matches_the_faces_table_slot_layout_in_any_argument_order() {
    const FACES: [(u32, u32, u32); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");
    for (f, &(a, b, k)) in FACES.iter().enumerate() {
        let want = 16 + f as u32;
        for &(x, y, z) in &[(a, b, k), (b, k, a), (k, a, b), (k, b, a)] {
            assert_eq!(
                c.face_node(x, y, z),
                Some(want),
                "face permutation ({x},{y},{z})"
            );
        }
    }
}

#[test]
fn edge_nodes_and_face_node_are_none_for_pairs_and_triples_not_in_the_mesh() {
    let mesh = single_tet_mesh(3.0);
    let c = CubicMesh::from_tet_mesh(&mesh).expect("well formed");

    // A corner has no edge to itself.
    assert_eq!(c.edge_nodes(0, 0), None);
    // Corner 99 does not exist in a 4-corner mesh; the lookup is a BTreeMap
    // key miss, not a bounds check, so it must still return None rather than
    // panic.
    assert_eq!(c.edge_nodes(0, 99), None);
    assert_eq!(c.edge_nodes(99, 100), None);

    // A repeated corner is not a face of this (or any) mesh.
    assert_eq!(c.face_node(0, 0, 1), None);
    // Three corners that are all real but do not form one of this mesh's
    // four faces together (0,1,2,3 is the whole tetrahedron's corner set,
    // not a face of it since a face only has three).
    assert_eq!(c.face_node(0, 99, 2), None);
}

// ---------------------------------------------------------------------------
// interior_node_positions_are_exact
// ---------------------------------------------------------------------------

/// Both answers the flag is allowed to give, and the degenerate boundary
/// between them: 3 mm divides the `(2a+b)/3` and `(a+b+c)/3` thirds exactly,
/// 1 mm does not.
#[test]
fn interior_node_positions_are_exact_iff_the_lattice_is_a_multiple_of_three() {
    let exact = CubicMesh::from_tet_mesh(&single_tet_mesh(3.0)).expect("well formed");
    assert!(exact.interior_node_positions_are_exact());

    let inexact = CubicMesh::from_tet_mesh(&single_tet_mesh(1.0)).expect("well formed");
    assert!(!inexact.interior_node_positions_are_exact());

    // Degenerate boundary case in between: 6 mm is also a multiple of three.
    let also_exact = CubicMesh::from_tet_mesh(&single_tet_mesh(6.0)).expect("well formed");
    assert!(also_exact.interior_node_positions_are_exact());
}

// ---------------------------------------------------------------------------
// shape_values: hand-derived cubic Lagrange formula, not the function under
// test, on the expected side
// ---------------------------------------------------------------------------

/// The twenty cubic shape functions at one barycentric point, from the
/// formulas in the `cubic_elastic_fem` module header. A second, independent
/// implementation of the same published formula — not a call to
/// [`shape_values`].
fn hand_shape_values(l: [f64; 4]) -> [f64; 20] {
    const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];
    let mut n = [0.0f64; 20];
    for (i, &li) in l.iter().enumerate() {
        n[i] = 0.5 * li * (3.0 * li - 1.0) * (3.0 * li - 2.0);
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        let (li, lj) = (l[i], l[j]);
        n[4 + 2 * e] = 4.5 * li * lj * (3.0 * li - 1.0);
        n[5 + 2 * e] = 4.5 * lj * li * (3.0 * lj - 1.0);
    }
    for (f, &(i, j, k)) in FACES.iter().enumerate() {
        n[16 + f] = 27.0 * l[i] * l[j] * l[k];
    }
    n
}

fn assert_shape_values_ulp_exact(label: &str, lambda: [f64; 4], want: &[f64; 20]) {
    let got = shape_values(&[fx(lambda[0]), fx(lambda[1]), fx(lambda[2]), fx(lambda[3])]);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            raw(*g),
            raw(fx(*w)),
            "{label}: slot {i} must evaluate with no rounding at a dyadic abscissa"
        );
    }
}

fn assert_shape_values_close(label: &str, lambda: [f64; 4], want: &[f64; 20], tol: f64) {
    let got = shape_values(&[fx(lambda[0]), fx(lambda[1]), fx(lambda[2]), fx(lambda[3])]);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g.to_f64() - w).abs();
        assert!(
            err < tol,
            "{label}: slot {i} error {err:.3e} exceeds tolerance {tol:.3e}"
        );
    }
}

/// The four corners: a Lagrange basis is exactly 1 at its own node and 0 at
/// the other nineteen — at a dyadic abscissa (0 and 1 are both dyadic), so
/// the match is bit-exact, not approximate.
#[test]
fn shape_values_is_one_hot_at_each_corner() {
    for i in 0..4 {
        let mut lambda = [0.0f64; 4];
        lambda[i] = 1.0;
        let mut want = [0.0f64; 20];
        want[i] = 1.0;
        assert_shape_values_ulp_exact(&format!("corner {i}"), lambda, &want);
    }
}

/// The centroid `(1/4,1/4,1/4,1/4)`: every lambda and every shape-function
/// coefficient is dyadic, so the formula is exact in `Fix128`. By the
/// corner/edge/face formulas the values are `5/128` at each corner,
/// `−9/128` at each edge node, `27/64` at each face node — derived from the
/// published formula, not from calling `shape_values`.
#[test]
fn shape_values_at_the_centroid_matches_the_hand_derived_dyadic_fractions() {
    let mut want = [0.0f64; 20];
    for slot in want.iter_mut().take(4) {
        *slot = 5.0 / 128.0;
    }
    for slot in want.iter_mut().take(16).skip(4) {
        *slot = -9.0 / 128.0;
    }
    for slot in want.iter_mut().skip(16) {
        *slot = 27.0 / 64.0;
    }
    let sum: f64 = want.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1.0e-12,
        "4*(5/128) + 12*(-9/128) + 4*(27/64) must equal 1 (partition of unity)"
    );
    assert_shape_values_ulp_exact("centroid", [0.25; 4], &want);
}

/// A non-dyadic interior point and an "edge-node-equivalent" point `(2/3,
/// 1/3, 0, 0)`: both still have to match the formula, to the tolerance the
/// probe's own construction allows (`2/3` and `1/3` are not representable in
/// `Fix128` either, so the comparison here is against the formula evaluated
/// on the same rounded lambda the implementation receives, not an exact
/// match).
#[test]
fn shape_values_matches_the_formula_at_non_dyadic_points() {
    let interior = [0.5, 0.3, 0.1, 0.1];
    assert_shape_values_close(
        "non-dyadic interior point",
        interior,
        &hand_shape_values(interior),
        1.0e-9,
    );

    // The barycentric point of the edge-0 node nearer corner 0: at an exact
    // 1/3 this would be one-hot at slot 4, but 1/3 truncates in Fix128, so
    // the tolerance is wider than the dyadic checks above.
    let edge_like = [2.0 / 3.0, 1.0 / 3.0, 0.0, 0.0];
    assert_shape_values_close(
        "edge-node-equivalent point",
        edge_like,
        &hand_shape_values(edge_like),
        1.0e-9,
    );
}

/// Degenerate: barycentric coordinates that do not sum to 1.
/// `shape_values` is a pure polynomial evaluation with no validation of
/// `Σλ = 1`, so it must still match the same formula rather than panic,
/// clamp, or silently renormalise.
#[test]
fn shape_values_does_not_validate_that_the_coordinates_sum_to_one() {
    let off = [0.5, 0.3, 0.1, 0.05]; // sums to 0.95, not 1
    assert_shape_values_close(
        "off-simplex (sums to 0.95)",
        off,
        &hand_shape_values(off),
        1.0e-9,
    );

    // Also exercise coordinates outside [0, 1] altogether (a point far
    // outside the tetrahedron, extreme relative to any physical use, but
    // nowhere near Fix128's overflow range) — still a plain polynomial
    // evaluation.
    let extreme = [10.0, -9.0, 0.5, -0.5]; // sums to 1
    assert_shape_values_close(
        "extreme coordinates outside the simplex",
        extreme,
        &hand_shape_values(extreme),
        1.0e-6,
    );
}
