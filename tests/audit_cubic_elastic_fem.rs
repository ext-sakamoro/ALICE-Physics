//! Audit oracles (S2-1) for `alice_physics::cubic_elastic_fem` (P3, twenty-node
//! cubic tetrahedra).
//!
//! Expected values come from closed forms (Euler-style node counts, the doc's
//! shape-function formulas, the Kronecker / partition-of-unity / polynomial
//! reproduction identities of a cubic Lagrange basis, a linear-field patch test
//! with `sigma = lambda tr(eps) I + 2 mu eps`, and the energy identity
//! `u^T K u = V sigma:eps`), never from the code under test. `known defect`
//! tests are `#[ignore]`d and recorded in the audit ledger; they are not fixed
//! here.

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    clippy::field_reassign_with_default,
    clippy::unnecessary_map_or,
    clippy::needless_range_loop
)]

use alice_physics::cubic_elastic_fem::{
    reactions, shape_values, solve_cubic, solve_cubic_hyperelastic, CubicMesh,
};
use alice_physics::hyperelastic::HyperelasticModel;
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, CorotationalConfig, ElasticMaterial, FemError, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

const E_MPA: f64 = 3500.0;
const NU: f64 = 0.35;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid")
}

fn lame() -> (f64, f64) {
    (
        E_MPA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU)),
        E_MPA / (2.0 * (1.0 + NU)),
    )
}

fn config() -> SolverConfig {
    SolverConfig::try_new(100_000, Fix128::from_raw(0, 1 << 34))
        .expect("valid")
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .expect("valid")
}

fn mesh_of(vertices: &[[f32; 3]], tets: &[[u32; 4]]) -> SdfTetMesh {
    let mut m = SdfTetMesh::default();
    m.vertices.extend_from_slice(vertices);
    for &t in tets {
        m.tets.push(Tetrahedron { vertices: t });
    }
    m
}

const CORNERS5: [[f32; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 3.0],
    [3.0, 3.0, 3.0],
];

/// Two tetrahedra sharing the face (1, 2, 3); every coordinate a multiple of 3.
fn two_tets(order_a: [u32; 4], order_b: [u32; 4]) -> SdfTetMesh {
    mesh_of(&CORNERS5, &[order_a, order_b])
}

fn pos(c: &CubicMesh, n: u32) -> [f64; 3] {
    c.node_position(n).expect("node exists").map(Fix128::to_f64)
}

const EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
const FACES: [(usize, usize, usize); 4] = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)];

// ------------------------------------------------------------------ topology

#[test]
fn node_counts_single_tet_and_shared_face_pair() {
    let one = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[[0, 1, 2, 3]])).unwrap();
    assert_eq!(one.corner_count(), 4);
    assert_eq!(one.edge_node_count(), 12);
    assert_eq!(one.face_node_count(), 4);
    assert_eq!(one.node_count(), 20);
    assert_eq!(one.element_count(), 1);
    // two tets sharing a face: 5 corners, 6 + 6 - 3 = 9 edges, 4 + 4 - 1 = 7
    // faces -> 5 + 18 + 7 = 30
    let two = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    assert_eq!(two.corner_count(), 5);
    assert_eq!(two.edge_node_count(), 18);
    assert_eq!(two.face_node_count(), 7);
    assert_eq!(two.node_count(), 30);
    assert_eq!(two.element_count(), 2);
}

/// Doc: node slots are four corners, twelve edge nodes (two per edge in the
/// order (0,1) (0,2) (0,3) (1,2) (1,3) (2,3), the first nearer the edge's first
/// corner), then four face nodes (face f opposite corner f). Positions are
/// checked against the closed form `(2a+b)/3`, `(a+b)/3`-style centroids, for
/// vertex orders that are ascending and not.
#[test]
fn element_slots_sit_at_thirds_and_centroids_for_any_vertex_order() {
    for order in [[0u32, 1, 2, 3], [3, 1, 0, 2], [2, 3, 1, 0]] {
        let c = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[order])).unwrap();
        let nodes = c.element_nodes(0).unwrap();
        let corner = |i: usize| CORNERS5[order[i] as usize].map(f64::from);
        for i in 0..4 {
            assert_eq!(nodes[i], order[i], "corner slot {i}");
            assert_eq!(pos(&c, nodes[i]), corner(i));
        }
        for (e, &(i, j)) in EDGES.iter().enumerate() {
            let (a, b) = (corner(i), corner(j));
            for axis in 0..3 {
                let near_i = (2.0 * a[axis] + b[axis]) / 3.0;
                let near_j = (a[axis] + 2.0 * b[axis]) / 3.0;
                assert!((pos(&c, nodes[4 + 2 * e])[axis] - near_i).abs() < 1e-12);
                assert!((pos(&c, nodes[5 + 2 * e])[axis] - near_j).abs() < 1e-12);
            }
        }
        for (f, &(i, j, k)) in FACES.iter().enumerate() {
            for axis in 0..3 {
                let want = (corner(i)[axis] + corner(j)[axis] + corner(k)[axis]) / 3.0;
                assert!(
                    (pos(&c, nodes[16 + f])[axis] - want).abs() < 1e-12,
                    "face {f}"
                );
            }
        }
    }
}

/// Doc: corners keep their indices, the sixteen extra nodes follow numbered
/// by first appearance in element order.
#[test]
fn extra_nodes_are_numbered_after_the_corners_in_first_appearance_order() {
    let one = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[[0, 1, 2, 3]])).unwrap();
    let want: Vec<u32> = (0..20).collect();
    assert_eq!(one.element_nodes(0).unwrap().to_vec(), want);
    let two = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let (a, b) = (two.element_nodes(0).unwrap(), two.element_nodes(1).unwrap());
    // first element takes 5..=20, the second only the nodes not already there
    assert_eq!(&a[..], &(0..4).chain(5..21).collect::<Vec<u32>>()[..]);
    let new_in_b: Vec<u32> = b.iter().copied().filter(|n| !a.contains(n)).collect();
    // corner 4 plus the nine edge / face nodes not shared with element 0
    let mut want_new = vec![4u32];
    want_new.extend(21..30);
    assert_eq!(new_in_b, want_new);
    // shared face (1,2,3): face slot 16 of element 0, slot 19 of element 1
    let shared = two.face_node(1, 2, 3).unwrap();
    assert_eq!(a[16], shared);
    assert_eq!(b[19], shared);
    // every node position is distinct
    let mut seen: Vec<[i64; 3]> = (0..30)
        .map(|n| pos(&two, n).map(|v| (v * 1e6).round() as i64))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 30);
}

#[test]
fn lookups_are_order_independent_and_none_for_non_members() {
    let two = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    for (a, b) in [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (1, 4)] {
        let ab = two.edge_nodes(a, b).expect("edge exists");
        assert_eq!(two.edge_nodes(b, a), Some(ab));
        // first node nearer the lower-numbered corner
        let (pa, pb) = (pos(&two, a.min(b)), pos(&two, a.max(b)));
        for axis in 0..3 {
            assert!((pos(&two, ab.0)[axis] - (2.0 * pa[axis] + pb[axis]) / 3.0).abs() < 1e-12);
            assert!((pos(&two, ab.1)[axis] - (pa[axis] + 2.0 * pb[axis]) / 3.0).abs() < 1e-12);
        }
    }
    assert_eq!(two.edge_nodes(0, 4), None, "apexes are not joined");
    let f = two.face_node(1, 2, 3).unwrap();
    for perm in [(1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)] {
        assert_eq!(two.face_node(perm.0, perm.1, perm.2), Some(f));
    }
    assert_eq!(two.face_node(0, 1, 4), None);
    assert!(two.element_nodes(2).is_none());
    assert!(two.node_position(30).is_none());
    assert!(two.node_position(29).is_some());
    for i in 0..5u32 {
        assert_eq!(
            pos(&two, i),
            CORNERS5[i as usize].map(f64::from),
            "corner {i}"
        );
    }
}

/// Doc: the flag is true when every third landed exactly (3 mm lattice), false
/// when any did not (1 mm lattice, or one inexact corner among exact ones).
#[test]
fn interior_exactness_flag_tracks_any_inexact_node() {
    let exact = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    assert!(exact.interior_node_positions_are_exact());
    let unit = CubicMesh::from_tet_mesh(&mesh_of(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        &[[0, 1, 2, 3]],
    ))
    .unwrap();
    assert!(!unit.interior_node_positions_are_exact());
    let mut v = CORNERS5;
    v[4] = [3.0, 3.0, 4.0]; // one inexact apex among exact corners
    let mixed = CubicMesh::from_tet_mesh(&mesh_of(&v, &[[0, 1, 2, 3], [1, 2, 3, 4]])).unwrap();
    assert!(!mixed.interior_node_positions_are_exact());
    // inexact nodes are truncated, never rounded up: error in (-2^-60, 0]
    let n = unit.edge_nodes(0, 1).unwrap().0; // nearer corner 0: x = 1/3
    let x = pos(&unit, n)[0];
    assert!(x <= 1.0 / 3.0 && 1.0 / 3.0 - x < 1e-15, "x = {x}");
}

#[test]
fn from_tet_mesh_error_contract() {
    let empty_v = SdfTetMesh::default();
    assert!(matches!(
        CubicMesh::from_tet_mesh(&empty_v),
        Err(FemError::EmptyMesh)
    ));
    let no_tets = mesh_of(&CORNERS5, &[]);
    assert!(matches!(
        CubicMesh::from_tet_mesh(&no_tets),
        Err(FemError::EmptyMesh)
    ));
    let oob = mesh_of(&CORNERS5[..4], &[[0, 1, 2, 4]]);
    match CubicMesh::from_tet_mesh(&oob) {
        Err(FemError::VertexOutOfRange {
            vertex,
            vertex_count,
        }) => {
            assert_eq!((vertex, vertex_count), (4, 4));
        }
        other => panic!("{other:?}"),
    }
    // index of the first zero-volume tetrahedron is reported
    let flat = mesh_of(
        &[
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        &[[0, 1, 2, 4], [0, 1, 2, 3]],
    );
    match CubicMesh::from_tet_mesh(&flat) {
        Err(FemError::DegenerateElement { tet }) => assert_eq!(tet, 1),
        other => panic!("{other:?}"),
    }
}

// ----------------------------------------------------------- shape functions

fn node_barycentrics() -> [[f64; 4]; 20] {
    let mut out = [[0.0f64; 4]; 20];
    for i in 0..4 {
        out[i][i] = 1.0;
    }
    for (e, &(i, j)) in EDGES.iter().enumerate() {
        out[4 + 2 * e][i] = 2.0 / 3.0;
        out[4 + 2 * e][j] = 1.0 / 3.0;
        out[5 + 2 * e][j] = 2.0 / 3.0;
        out[5 + 2 * e][i] = 1.0 / 3.0;
    }
    for (f, &(i, j, k)) in FACES.iter().enumerate() {
        out[16 + f][i] = 1.0 / 3.0;
        out[16 + f][j] = 1.0 / 3.0;
        out[16 + f][k] = 1.0 / 3.0;
    }
    out
}

fn n_at(l: [f64; 4]) -> [f64; 20] {
    shape_values(&l.map(fx)).map(Fix128::to_f64)
}

/// Kronecker delta `N_a(x_b) = delta_ab` at the 20 nodes.
#[test]
fn shape_values_are_a_nodal_basis() {
    let nodes = node_barycentrics();
    for (b, lb) in nodes.iter().enumerate() {
        let n = n_at(*lb);
        for (a, v) in n.iter().enumerate() {
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-12, "N_{a}(x_{b}) = {v}");
        }
    }
}

/// Doc formulas at rational points, plus partition of unity and reproduction
/// of linear, quadratic and cubic polynomials in the barycentric coordinates.
#[test]
fn shape_values_match_doc_formulas_and_reproduce_cubics() {
    let pts = [
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.0, 0.5, 0.3, 0.2],
        [0.05, 0.15, 0.35, 0.45],
    ];
    let nodes = node_barycentrics();
    for l in pts {
        let n = n_at(l);
        let sum: f64 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "partition of unity {sum}");
        for i in 0..4 {
            let want = 0.5 * l[i] * (3.0 * l[i] - 1.0) * (3.0 * l[i] - 2.0);
            assert!((n[i] - want).abs() < 1e-12, "corner {i}");
        }
        for (e, &(i, j)) in EDGES.iter().enumerate() {
            let near_i = 4.5 * l[i] * l[j] * (3.0 * l[i] - 1.0);
            let near_j = 4.5 * l[i] * l[j] * (3.0 * l[j] - 1.0);
            assert!((n[4 + 2 * e] - near_i).abs() < 1e-12, "edge {e} near i");
            assert!((n[5 + 2 * e] - near_j).abs() < 1e-12, "edge {e} near j");
        }
        for (f, &(i, j, k)) in FACES.iter().enumerate() {
            assert!(
                (n[16 + f] - 27.0 * l[i] * l[j] * l[k]).abs() < 1e-12,
                "face {f}"
            );
        }
        let polys: [fn(&[f64; 4]) -> f64; 4] = [
            |x| 2.0 * x[0] - x[1] + 0.5 * x[3],
            |x| x[0] * x[1] + 3.0 * x[2] * x[2] - x[3] * x[0],
            |x| x[0] * x[0] * x[0] - 2.0 * x[1] * x[1] * x[2] + x[3] * x[2] * x[0],
            |x| x[1] * x[2] * x[3] + x[3] * x[3] * x[3],
        ];
        for (k, p) in polys.iter().enumerate() {
            let interp: f64 = (0..20).map(|a| n[a] * p(&nodes[a])).sum();
            assert!(
                (interp - p(&l)).abs() < 1e-11,
                "poly {k}: {interp} vs {}",
                p(&l)
            );
        }
    }
}

// ------------------------------------------------------------- linear solve

fn linear_field(p: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = p;
    let c = 1.0e-3;
    [
        c * (1.0 + 2.0 * x - y + 0.5 * z),
        c * (3.0 * y + z - 0.25 * x),
        c * (x + y + 4.0 * z),
    ]
}

/// Gradient of `linear_field`: `grad[i][j] = d u_i / d x_j`.
fn linear_gradient() -> [[f64; 3]; 3] {
    let c = 1.0e-3;
    [
        [2.0 * c, -c, 0.5 * c],
        [-0.25 * c, 3.0 * c, c],
        [c, c, 4.0 * c],
    ]
}

/// Closed-form constant stress of the linear field.
fn linear_stress() -> [f64; 6] {
    let g = linear_gradient();
    let (lam, mu) = lame();
    let tr = g[0][0] + g[1][1] + g[2][2];
    [
        lam * tr + 2.0 * mu * g[0][0],
        lam * tr + 2.0 * mu * g[1][1],
        lam * tr + 2.0 * mu * g[2][2],
        mu * (g[0][1] + g[1][0]),
        mu * (g[1][2] + g[2][1]),
        mu * (g[0][2] + g[2][0]),
    ]
}

/// Prescribe the linear field on every node except the one truly interior node,
/// the centroid node of the shared face (1, 2, 3). (The three edge nodes pairs
/// of that face also lie on outer faces of the two tetrahedra, so they belong
/// to the boundary.) Returns the free node ids.
fn patch_boundary(c: &CubicMesh) -> (BoundaryConditions, Vec<u32>) {
    let free: Vec<u32> = vec![c.face_node(1, 2, 3).unwrap()];
    let mut b = BoundaryConditions::new();
    for n in 0..c.node_count() as u32 {
        if !free.contains(&n) {
            let u = linear_field(pos(c, n));
            b.prescribe_all(n, u.map(fx));
        }
    }
    (b, free)
}

/// Patch test: a linear field is in the P3 space, so with every outer node
/// prescribed from it the free interior node must land on it and every element's
/// centroid stress must equal `lambda tr(eps) I + 2 mu eps` -- for either
/// orientation of the tetrahedra (a mirrored element has det < 0).
#[test]
fn linear_patch_is_reproduced_with_the_closed_form_stress() {
    let want = linear_stress();
    for (oa, ob) in [
        ([0u32, 1, 2, 3], [1u32, 2, 3, 4]),
        ([0, 2, 1, 3], [1, 3, 2, 4]),
    ] {
        let c = CubicMesh::from_tet_mesh(&two_tets(oa, ob)).unwrap();
        let (b, free) = patch_boundary(&c);
        let s = solve_cubic(&c, &material(), &b, &config()).unwrap();
        assert!(s.iterations > 0);
        assert!(s.relative_residual <= s.effective_relative_tolerance);
        // floor-adjusted tolerance is never below the requested one (1 ulp slack
        // for the division target / b_norm)
        assert!(s.effective_relative_tolerance.to_f64() >= 2f64.powi(-30) - 1e-18);
        for n in free {
            let u = linear_field(pos(&c, n));
            for axis in 0..3 {
                let got = s.displacements[n as usize][axis].to_f64();
                assert!(
                    (got - u[axis]).abs() < 1e-8,
                    "node {n} axis {axis}: {got} vs {}",
                    u[axis]
                );
            }
        }
        assert_eq!(s.element_stress.len(), 2);
        for st in &s.element_stress {
            let got = [st.xx, st.yy, st.zz, st.xy, st.yz, st.zx].map(Fix128::to_f64);
            for k in 0..6 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-5 * (1.0 + want[k].abs()),
                    "component {k}: {} vs {}",
                    got[k],
                    want[k]
                );
            }
        }
    }
}

/// Energy identity: `sum_n R_n . u_n = u^T K u = V sigma:eps` for the patch
/// field. Pins the quadrature weights (sum to one), the volume factor and the
/// stiffness assembly in one number. Also: reactions vanish on free nodes and
/// sum to zero (no loads), and a load on a prescribed dof shifts exactly that
/// reaction by minus the load.
#[test]
fn reactions_satisfy_the_energy_identity_and_load_bookkeeping() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let (b, free) = patch_boundary(&c);
    let s = solve_cubic(&c, &material(), &b, &config()).unwrap();
    let r = reactions(&c, &material(), &b, None, &s).unwrap();
    // total volume: tet A = 27/6; tet B from the corner (3,0,0): edges
    // (-3,3,0), (-3,0,3), (0,3,3); det = -3*(0*3-3*3) - 3*(-3*3-3*0) + 0 = 54
    let vol = 27.0 / 6.0 + 54.0 / 6.0;
    let g = linear_gradient();
    let st = linear_stress();
    let eps = [
        g[0][0],
        g[1][1],
        g[2][2],
        0.5 * (g[0][1] + g[1][0]),
        0.5 * (g[1][2] + g[2][1]),
        0.5 * (g[0][2] + g[2][0]),
    ];
    let w = st[0] * eps[0]
        + st[1] * eps[1]
        + st[2] * eps[2]
        + 2.0 * (st[3] * eps[3] + st[4] * eps[4] + st[5] * eps[5]);
    let mut dot = 0.0;
    let mut sum = [0.0f64; 3];
    for (n, rn) in r.iter().enumerate() {
        if free.contains(&(n as u32)) {
            assert!(
                rn.iter().all(|v| v.to_f64().abs() < 1e-9),
                "free node {n} reacts"
            );
        }
        let u = linear_field(pos(&c, n as u32));
        for axis in 0..3 {
            dot += rn[axis].to_f64() * u[axis];
            sum[axis] += rn[axis].to_f64();
        }
    }
    assert!(
        (dot - vol * w).abs() < 1e-5 * (vol * w).abs(),
        "R.u = {dot}, V sigma:eps = {}",
        vol * w
    );
    assert!(sum.iter().all(|v| v.abs() < 1e-6), "net reaction {sum:?}");
    // a load on a prescribed dof (node 0, x): reaction is minus that load
    let mut b2 = b.clone();
    b2.add_load(0, Axis::X, fx(5.0));
    let r2 = reactions(&c, &material(), &b2, None, &s).unwrap();
    assert!((r2[0][0].to_f64() - (r[0][0].to_f64() - 5.0)).abs() < 1e-9);
    assert_eq!(r2[0][1], r[0][1]);
    assert_eq!(r2[1][0], r[1][0]);
}

#[test]
fn solve_and_reactions_error_contract() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let (b, _free) = patch_boundary(&c);
    let s = solve_cubic(&c, &material(), &b, &config()).unwrap();
    let mut bad = BoundaryConditions::new();
    bad.prescribe(30, Axis::X, Fix128::ZERO);
    for r in [
        solve_cubic(&c, &material(), &bad, &config()).map(|_| ()),
        reactions(&c, &material(), &bad, None, &s).map(|_| ()),
    ] {
        match r {
            Err(FemError::VertexOutOfRange {
                vertex,
                vertex_count,
            }) => {
                assert_eq!((vertex, vertex_count), (30, 30));
            }
            other => panic!("{other:?}"),
        }
    }
    let mut bad_load = BoundaryConditions::new();
    bad_load.add_load(31, Axis::Y, Fix128::ONE);
    assert!(matches!(
        solve_cubic(&c, &material(), &bad_load, &config()),
        Err(FemError::VertexOutOfRange { vertex: 31, .. })
    ));
    // node 29 is the last valid index
    let mut edge = BoundaryConditions::new();
    for n in [0u32, 1, 2, 3] {
        edge.prescribe_all(n, [Fix128::ZERO; 3]);
    }
    edge.prescribe(29, Axis::X, Fix128::ZERO);
    assert!(solve_cubic(&c, &material(), &edge, &config()).is_ok());
    // a solution from another mesh
    let single = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[[0, 1, 2, 3]])).unwrap();
    assert!(matches!(
        reactions(&single, &material(), &b, None, &s),
        Err(FemError::VertexOutOfRange { .. } | FemError::SolutionDoesNotMatchMesh { .. })
    ));
    let mut ok_single = BoundaryConditions::new();
    ok_single.prescribe(0, Axis::X, Fix128::ZERO);
    match reactions(&single, &material(), &ok_single, None, &s) {
        Err(FemError::SolutionDoesNotMatchMesh {
            nodes,
            vertex_count,
        }) => {
            assert_eq!((nodes, vertex_count), (30, 20));
        }
        other => panic!("{other:?}"),
    }
}

/// Fully prescribed: nothing is solved. The answer is the boundary data, with
/// 0 iterations, a zero residual and the requested tolerance reported back.
#[test]
fn fully_prescribed_returns_the_boundary_data_untouched() {
    let c = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[[0, 1, 2, 3]])).unwrap();
    let mut b = BoundaryConditions::new();
    for n in 0..20u32 {
        b.prescribe_all(n, linear_field(pos(&c, n)).map(fx));
    }
    let cfg = config();
    let s = solve_cubic(&c, &material(), &b, &cfg).unwrap();
    assert_eq!(s.iterations, 0);
    assert_eq!(s.relative_residual, Fix128::ZERO);
    assert_eq!(s.effective_relative_tolerance, cfg.relative_tolerance());
    for n in 0..20usize {
        assert_eq!(s.displacements[n], linear_field(pos(&c, n as u32)).map(fx));
    }
    let want = linear_stress();
    let st = s.element_stress[0];
    assert!((st.xx.to_f64() - want[0]).abs() < 1e-5 * want[0].abs());
    assert!((st.xy.to_f64() - want[3]).abs() < 1e-5 * (1.0 + want[3].abs()));
}

/// Doc (solve_cubic): clamping only the corners of a face "is a different
/// problem that still solves". Pins that behaviour; the `# Errors` paragraph
/// of the same item says the same case yields `UnderConstrained` (see the
/// ledger: the two paragraphs disagree and the code follows the first).
#[test]
fn corner_only_clamp_still_solves() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let mut b = BoundaryConditions::new();
    for v in [0u32, 1, 2] {
        b.prescribe_all(v, [Fix128::ZERO; 3]);
    }
    b.add_load(4, Axis::X, fx(10.0));
    let s = solve_cubic(&c, &material(), &b, &config());
    assert!(s.is_ok(), "{:?}", s.err());
}

/// KNOWN DEFECT: `FemError::UnderConstrained` is documented as "reported when
/// fewer than six degrees of freedom are prescribed", and the P1 solver
/// follows it (`Err(UnderConstrained)` for the same input). `solve_cubic` has
/// no such check: with nothing prescribed and a single nodal load it returns
/// `Ok` with displacements of order 1e11 mm and `relative_residual == 0`
/// (measured: iterations 64, u[0].x = -3.5e11; with a 1-ulp different nu the
/// same body ends in `Stagnated`, so which failure shows is rounding-chaotic).
/// The oracle is the equilibrium
/// every `Ok` answer owes: reactions plus applied loads must balance.
#[test]
#[ignore = "known defect: AUD-A-S2W1-008: solve_cubic returns Ok (u ~ 1e11 mm, relative_residual 0) for a body with no constraints; P1 solve returns Err(UnderConstrained) for the same input"]
fn solve_cubic_never_reports_ok_for_a_body_nothing_holds() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let mut b = BoundaryConditions::new();
    b.add_load(4, Axis::X, fx(10.0));
    // nu built with from_ratio: with `from_f64(0.35)` (1 ulp elsewhere) the same
    // input ends in Err(Stagnated) instead, so the outcome is rounding-chaotic
    let mat = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(35, 100)).unwrap();
    match solve_cubic(&c, &mat, &b, &config()) {
        Err(_) => {}
        Ok(s) => {
            let worst = s
                .displacements
                .iter()
                .flat_map(|d| d.iter())
                .map(|v| v.to_f64().abs())
                .fold(0.0f64, f64::max);
            // a 10 N load on a 3500 MPa, 3 mm body cannot move it by a metre
            assert!(worst < 1.0e3, "Ok with max |u| = {worst} mm");
        }
    }
}

// --------------------------------------------------------------- hyperelastic

#[test]
fn hyperelastic_entry_point_error_contract() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let (full, _) = patch_boundary(&c);
    let lin = SolverConfig::try_new(1000, Fix128::from_raw(0, 1 << 34)).unwrap();
    let no_law = CorotationalConfig::try_new(lin, 5, Fix128::from_raw(0, 1 << 34), 1, 20).unwrap();
    assert!(matches!(
        solve_cubic_hyperelastic(&c, &material(), &full, &no_law),
        Err(FemError::InvalidConfig(_))
    ));
    let law = no_law.with_hyperelastic(HyperelasticModel::NeoHookean {
        mu_mpa: fx(E_MPA / 2.7),
    });
    // 5 prescribed dofs: fewer than the six rigid body modes need
    let mut five = BoundaryConditions::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        five.prescribe(0, axis, Fix128::ZERO);
    }
    five.prescribe(1, Axis::X, Fix128::ZERO);
    five.prescribe(1, Axis::Y, Fix128::ZERO);
    assert!(matches!(
        solve_cubic_hyperelastic(&c, &material(), &five, &law),
        Err(FemError::UnderConstrained)
    ));
    let mut bad = BoundaryConditions::new();
    bad.prescribe(30, Axis::X, Fix128::ZERO);
    assert!(matches!(
        solve_cubic_hyperelastic(&c, &material(), &bad, &law),
        Err(FemError::VertexOutOfRange {
            vertex: 30,
            vertex_count: 30
        })
    ));
}

// ------------------------------------------------------- loads and stress

/// Three corners clamped, a 10 N pull on corner 4 along +x: global balance
/// (`sum of reactions + applied load = 0`) and positive external work
/// `F . u > 0` pin the sign and size of the load vector.
#[test]
fn clamped_pull_balances_and_does_positive_work() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let mut b = BoundaryConditions::new();
    for v in [0u32, 1, 2] {
        b.prescribe_all(v, [Fix128::ZERO; 3]);
    }
    b.add_load(4, Axis::X, fx(10.0));
    let s = solve_cubic(&c, &material(), &b, &config()).unwrap();
    let r = reactions(&c, &material(), &b, None, &s).unwrap();
    let sum_x: f64 = r.iter().map(|n| n[0].to_f64()).sum();
    assert!(
        (sum_x + 10.0).abs() < 1e-5,
        "net reaction {sum_x} must cancel the 10 N pull"
    );
    let work = 10.0 * s.displacements[4][0].to_f64();
    assert!(work > 0.0, "work {work}");
    // reaction moment about the origin balances the load's: R x p summed
    let mut moment = [0.0f64; 3];
    for (n, rn) in r.iter().enumerate() {
        let p = pos(&c, n as u32);
        let f = [rn[0].to_f64(), rn[1].to_f64(), rn[2].to_f64()];
        moment[0] += p[1] * f[2] - p[2] * f[1];
        moment[1] += p[2] * f[0] - p[0] * f[2];
        moment[2] += p[0] * f[1] - p[1] * f[0];
    }
    // load 10 N along x at (3,3,3): moment = p x F = (0, 3*10, -3*10)
    assert!((moment[0]).abs() < 1e-4);
    assert!((moment[1] + 30.0).abs() < 1e-3, "{moment:?}");
    assert!((moment[2] - 30.0).abs() < 1e-3, "{moment:?}");
}

/// Centroid stress of a *quadratic* field prescribed on all twenty nodes of one
/// element (the field is in the P3 space, nothing is solved): the strain at the
/// centroid is the exact gradient there, so `element_stress` must equal
/// `D eps(centroid)`. A linear field cannot tell the centroid from any other
/// point.
#[test]
fn element_stress_is_sampled_at_the_centroid() {
    let c = CubicMesh::from_tet_mesh(&mesh_of(&CORNERS5[..4], &[[0, 1, 2, 3]])).unwrap();
    let k = 1.0e-4;
    let field = |p: [f64; 3]| [k * p[0] * p[0], k * p[1] * p[2], k * p[0] * p[2]];
    let mut b = BoundaryConditions::new();
    for n in 0..20u32 {
        b.prescribe_all(n, field(pos(&c, n)).map(fx));
    }
    let s = solve_cubic(&c, &material(), &b, &config()).unwrap();
    let ctr = [0.75f64, 0.75, 0.75]; // centroid of the 3-3-3 corner tetrahedron
                                     // gradient g[i][j] = d u_i / d x_j at the centroid
    let g = [
        [2.0 * k * ctr[0], 0.0, 0.0],
        [0.0, k * ctr[2], k * ctr[1]],
        [k * ctr[2], 0.0, k * ctr[0]],
    ];
    let (lam, mu) = lame();
    let tr = g[0][0] + g[1][1] + g[2][2];
    let want = [
        lam * tr + 2.0 * mu * g[0][0],
        lam * tr + 2.0 * mu * g[1][1],
        lam * tr + 2.0 * mu * g[2][2],
        mu * (g[0][1] + g[1][0]),
        mu * (g[1][2] + g[2][1]),
        mu * (g[0][2] + g[2][0]),
    ];
    let st = s.element_stress[0];
    let got = [st.xx, st.yy, st.zz, st.xy, st.yz, st.zx].map(Fix128::to_f64);
    for i in 0..6 {
        assert!(
            (got[i] - want[i]).abs() < 1e-6 * (1.0 + want[i].abs()),
            "{i}: {} vs {}",
            got[i],
            want[i]
        );
    }
}

/// `NotConverged` reports the iteration budget it stopped at: with a budget of
/// one iteration the patch problem (which needs several) must stop at exactly
/// `iterations == 1`.
#[test]
fn iteration_budget_is_honoured_exactly() {
    let c = CubicMesh::from_tet_mesh(&two_tets([0, 1, 2, 3], [1, 2, 3, 4])).unwrap();
    let (b, _) = patch_boundary(&c);
    let cfg = SolverConfig::try_new(1, Fix128::from_raw(0, 1 << 34))
        .unwrap()
        .with_stagnation(2_000, Fix128::from_raw(0, 1 << 54))
        .unwrap();
    match solve_cubic(&c, &material(), &b, &cfg) {
        Err(FemError::NotConverged { iterations, .. }) => assert_eq!(iterations, 1),
        other => panic!("{other:?}"),
    }
}
