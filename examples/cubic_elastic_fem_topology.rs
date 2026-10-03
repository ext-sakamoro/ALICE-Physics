//! Topology queries of the twenty-node cubic (P3) tetrahedral mesh.
//!
//! [`CubicMesh`] builds the node tables once and answers questions about them
//! — how many nodes of each kind, where an edge's or a face's extra node
//! landed, what the twenty node indices of one element are, and what the
//! cubic shape functions evaluate to at a barycentric point — without ever
//! running a solve. Every check below has a closed-form expectation derived
//! from the mesh's own topology or from the formulas in the
//! [`cubic_elastic_fem`](alice_physics::cubic_elastic_fem) module header, not
//! from calling the function under test to produce its own answer.
//!
//! ```bash
//! cargo run --example cubic_elastic_fem_topology --features std
//! ```

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

/// One tetrahedron, corners `(0,0,0)`, `(s,0,0)`, `(0,s,0)`, `(0,0,s)` mm.
///
/// With this corner order the six [`cubic_elastic_fem`](alice_physics::cubic_elastic_fem)
/// `EDGES` pairs and four `FACES` triples are read off the global corner
/// numbers directly (`tet.vertices` is the identity permutation), which is
/// what makes the node-index layout below a closed form instead of a
/// simulation of the builder.
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

/// The twenty cubic shape functions at one barycentric point, from the
/// formulas in the module header — a second, independent implementation, not
/// a call to [`shape_values`].
///
/// Corner `i`: `½λᵢ(3λᵢ−1)(3λᵢ−2)`. Edge node two thirds of the way to `i`
/// along `(i,j)`: `9/2 λᵢλⱼ(3λᵢ−1)`. Face node of `(i,j,k)`: `27λᵢλⱼλₖ`.
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

fn check_shape_values(label: &str, lambda: [f64; 4], want: &[f64; 20], ulp_exact: bool) {
    let got = shape_values(&[fx(lambda[0]), fx(lambda[1]), fx(lambda[2]), fx(lambda[3])]);
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        worst = worst.max((g.to_f64() - w).abs());
    }
    println!(
        "[cubic_elastic_fem] shape_values at {label} (lambda={lambda:?}): worst |got-want| = \
         {worst:.3e}{}",
        if ulp_exact {
            " (dyadic point, must be 0 ulp)"
        } else {
            ""
        }
    );
    if ulp_exact {
        for (g, w) in got.iter().zip(want.iter()) {
            // `want` is itself a dyadic rational at these probes, representable
            // exactly in f64, so round-tripping through `fx` and comparing raw
            // ulps is a bit-exact check, not an approximate one.
            assert_eq!(
                raw(*g),
                raw(fx(*w)),
                "{label}: dyadic coefficients at a dyadic abscissa must evaluate with no \
                 rounding at all"
            );
        }
    } else {
        assert!(
            worst < 1.0e-9,
            "{label}: worst error {worst:.3e} exceeds the tolerance for a non-dyadic probe"
        );
    }
}

fn main() {
    println!("[cubic_elastic_fem] P3 tetrahedral mesh topology walkthrough");

    // --- A single tetrahedron: every count and index is a closed form -------
    let mesh = single_tet_mesh(3.0); // multiple of 3 mm: interior positions land exactly
    let c = CubicMesh::from_tet_mesh(&mesh).expect("one non-degenerate tet is well formed");

    // 4 corners, C(4,2) = 6 edges (2 nodes each), 4 faces (1 node each), 1
    // element: 4 + 12 + 4 = 20 nodes total.
    assert_eq!(c.corner_count(), 4, "a tetrahedron has 4 corners");
    assert_eq!(c.edge_node_count(), 12, "6 edges * 2 nodes per edge");
    assert_eq!(c.face_node_count(), 4, "4 faces * 1 node per face");
    assert_eq!(c.element_count(), 1, "one tetrahedron in the source mesh");
    assert_eq!(
        c.node_count(),
        20,
        "4 corners + 12 edge nodes + 4 face nodes"
    );
    println!(
        "[cubic_elastic_fem] single tet: {} corners + {} edge nodes + {} face nodes = {} \
         nodes, {} element",
        c.corner_count(),
        c.edge_node_count(),
        c.face_node_count(),
        c.node_count(),
        c.element_count()
    );

    // With corners numbered 0..4 and `tet.vertices == [0,1,2,3]`, every edge
    // and face node is created in `EDGES`/`FACES` order on first appearance,
    // so element 0's layout is the identity permutation 0..20 — a closed
    // form read off the module's own numbering rule, not the output of
    // `element_nodes` itself.
    let nodes = c.element_nodes(0).expect("element 0 exists");
    let expected: [u32; 20] = core::array::from_fn(|i| i as u32);
    assert_eq!(
        nodes, expected,
        "the only tet in the mesh must number its 20 nodes 0..20, corners first"
    );
    println!("[cubic_elastic_fem] element 0 nodes = {nodes:?}");

    // Edge `e` of `EDGES = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]` owns slots
    // `4+2e` and `5+2e`, so its pair is `(4+2e, 5+2e)` — and the corner with
    // the smaller global index is nearer the first of the two.
    const EDGES: [(u32, u32); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    for (e, &(a, b)) in EDGES.iter().enumerate() {
        let want = (4 + 2 * e as u32, 5 + 2 * e as u32);
        assert_eq!(
            c.edge_nodes(a, b),
            Some(want),
            "edge ({a},{b}) must own slots {want:?}"
        );
        assert_eq!(
            c.edge_nodes(b, a),
            Some(want),
            "edge_nodes must not depend on argument order"
        );
    }
    println!("[cubic_elastic_fem] all 6 edge pairs match the EDGES slot layout, both orders");

    // Face `f` of `FACES = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]` owns slot
    // `16+f`.
    const FACES: [(u32, u32, u32); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];
    for (f, &(a, b, c_)) in FACES.iter().enumerate() {
        let want = 16 + f as u32;
        assert_eq!(c.face_node(a, b, c_), Some(want), "face ({a},{b},{c_})");
        assert_eq!(
            c.face_node(c_, a, b),
            Some(want),
            "face_node must not depend on argument order"
        );
    }
    println!("[cubic_elastic_fem] all 4 face triples match the FACES slot layout, both orders");

    // --- Degenerate queries: no such edge/face/element must not panic -------
    assert_eq!(c.edge_nodes(0, 0), None, "a corner has no edge to itself");
    assert_eq!(
        c.edge_nodes(0, 99),
        None,
        "corner 99 is not in this mesh, so there is no edge to it"
    );
    assert_eq!(
        c.face_node(0, 0, 1),
        None,
        "a repeated corner is not a face of this mesh"
    );
    assert_eq!(
        c.element_nodes(1),
        None,
        "only element 0 exists in a one-tet mesh"
    );
    assert_eq!(
        c.element_nodes(usize::MAX),
        None,
        "an absurd index must return None, not panic"
    );
    println!("[cubic_elastic_fem] out-of-range edge/face/element queries return None, not panic");

    // --- interior_node_positions_are_exact: both answers it is allowed to give
    let exact = CubicMesh::from_tet_mesh(&single_tet_mesh(3.0)).expect("well formed");
    let inexact = CubicMesh::from_tet_mesh(&single_tet_mesh(1.0)).expect("well formed");
    assert!(
        exact.interior_node_positions_are_exact(),
        "a 3 mm tetrahedron places (2a+b)/3 and (a+b+c)/3 exactly"
    );
    assert!(
        !inexact.interior_node_positions_are_exact(),
        "a 1 mm tetrahedron does not divide by three exactly"
    );
    println!(
        "[cubic_elastic_fem] interior_node_positions_are_exact: 3 mm tet = {}, 1 mm tet = {}",
        exact.interior_node_positions_are_exact(),
        inexact.interior_node_positions_are_exact()
    );

    // --- shape_values: hand-derived cubic Lagrange formula, not the function
    //     under test, on both sides ------------------------------------------

    // The four corners: a Lagrange basis is 1 at its own node, 0 at the other
    // nineteen. Dyadic abscissae (0 and 1), so the match must be exact.
    for (i, label) in ["corner 0", "corner 1", "corner 2", "corner 3"]
        .into_iter()
        .enumerate()
    {
        let mut lambda = [0.0f64; 4];
        lambda[i] = 1.0;
        let mut want = [0.0f64; 20];
        want[i] = 1.0;
        check_shape_values(label, lambda, &want, true);
    }

    // The centroid (1/4,1/4,1/4,1/4): every coefficient and every lambda is
    // dyadic, so the formula is exact in `Fix128`. By the corner/edge/face
    // formulas: corners 5/128 each, edges −9/128 each, faces 27/64 each —
    // which sum to 4*(5/128) + 12*(−9/128) + 4*(27/64) = 1, the partition of
    // unity, derived independently of calling `shape_values`.
    let mut want_centroid = [0.0f64; 20];
    for slot in want_centroid.iter_mut().take(4) {
        *slot = 5.0 / 128.0;
    }
    for slot in want_centroid.iter_mut().take(16).skip(4) {
        *slot = -9.0 / 128.0;
    }
    for slot in want_centroid.iter_mut().skip(16) {
        *slot = 27.0 / 64.0;
    }
    check_shape_values("centroid", [0.25; 4], &want_centroid, true);
    let sum: f64 = want_centroid.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1.0e-12,
        "the centroid weights must sum to 1 by the partition-of-unity identity"
    );

    // A non-dyadic interior point: the shared formula still has to match, to
    // the tolerance the probe's own `1/3`-flavoured rounding allows.
    let probe = [0.5, 0.3, 0.1, 0.1];
    let want_probe = hand_shape_values(probe);
    check_shape_values("non-dyadic interior point", probe, &want_probe, false);

    // Degenerate: barycentric coordinates that do not sum to 1.
    // `shape_values` is a pure polynomial evaluation with no validation that
    // `Σλ = 1`, so it must still match the same formula rather than panic or
    // silently renormalise.
    let off = [0.5, 0.3, 0.1, 0.05]; // sums to 0.95
    let want_off = hand_shape_values(off);
    check_shape_values(
        "off-simplex point (sums to 0.95, not 1)",
        off,
        &want_off,
        false,
    );

    println!("[cubic_elastic_fem] topology walkthrough complete, all closed-form checks passed");
}
