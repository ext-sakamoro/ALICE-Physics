//! Mesh and Polyline Indexing Example
//!
//! Production entry point for two indexing accessors that no other example
//! reaches: `QuadraticMesh::element_nodes` (`src/quadratic_elastic_fem.rs`)
//! and `Streamlines::is_empty` (`src/flow_viz.rs`).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test:
//! - a ten-node tetrahedron lists its four corners first, in the order of the
//!   linear element, and then one node per edge; each of the six edge nodes
//!   sits at the midpoint `(a + b) / 2` of a distinct corner pair, and an
//!   edge shared by two elements is one node (the mesh is conforming)
//! - two tetrahedra sharing a face have `4 + 1` corners and `6 + 6 − 3 = 9`
//!   edges
//! - a streamline set has one polyline per seed: no seeds is empty, and a
//!   seed in a uniform field `v` traced `n` steps of `dt` moves by `n v dt`
//!
//! Run with: `cargo run --example mesh_and_polyline_indexing`

use alice_physics::flow_viz::generate_streamlines;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::quadratic_elastic_fem::QuadraticMesh;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn quadratic_element_nodes() {
    let mut mesh = SdfTetMesh::default();
    let corners: [[f32; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
    ];
    mesh.vertices.extend_from_slice(&corners);
    let tets = [[0u32, 1, 2, 3], [1, 2, 3, 4]];
    for t in tets {
        mesh.tets.push(Tetrahedron { vertices: t });
    }
    let q = QuadraticMesh::from_tet_mesh(&mesh).expect("two valid tetrahedra");
    assert_eq!(q.element_count(), 2);
    assert_eq!(q.edge_count(), 9, "6 + 6 edges, 3 of them shared");

    let mut all_edge_nodes = Vec::new();
    for (e, tet) in tets.iter().enumerate() {
        let nodes = q.element_nodes(e).expect("element exists");
        assert_eq!(&nodes[..4], tet, "element {e}: corners first, in order");

        let mut pairs_seen = Vec::new();
        for &n in &nodes[4..] {
            let p = q.node_position(n).expect("edge node has a position");
            let p = [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()];
            // Which corner pair is this node the midpoint of?
            let mut found = None;
            for i in 0..4 {
                for j in i + 1..4 {
                    let (a, b) = (corners[tet[i] as usize], corners[tet[j] as usize]);
                    let mid = [
                        (f64::from(a[0]) + f64::from(b[0])) / 2.0,
                        (f64::from(a[1]) + f64::from(b[1])) / 2.0,
                        (f64::from(a[2]) + f64::from(b[2])) / 2.0,
                    ];
                    if mid == p {
                        found = Some((i, j));
                    }
                }
            }
            let pair = found.unwrap_or_else(|| {
                panic!("element {e}: node {n} at {p:?} is no corner pair's midpoint")
            });
            assert_eq!(
                q.edge_node(tet[pair.0], tet[pair.1]),
                Some(n),
                "element {e}: the edge lookup agrees"
            );
            pairs_seen.push(pair);
            all_edge_nodes.push(n);
        }
        pairs_seen.sort_unstable();
        pairs_seen.dedup();
        assert_eq!(pairs_seen.len(), 6, "element {e}: one node per edge");
        println!("element {e}: nodes {nodes:?}");
    }
    all_edge_nodes.sort_unstable();
    all_edge_nodes.dedup();
    assert_eq!(
        all_edge_nodes.len(),
        9,
        "the three shared edges are one node each"
    );
    assert!(q.element_nodes(2).is_none(), "no third element");
}

fn streamline_emptiness() {
    let fluid = [Vec3Fix::ZERO];
    let velocity = [Vec3Fix::from_int(1, 0, 0)];
    let dt = Fix128::from_ratio(1, 4);

    let none = generate_streamlines(&fluid, &velocity, &[], 8, dt);
    assert!(none.is_empty(), "no seeds, no streamlines");
    assert_eq!(none.len(), 0);

    let one = generate_streamlines(&fluid, &velocity, &[Vec3Fix::ZERO], 1, dt);
    assert_eq!(one.len(), 1, "one seed, one streamline");
    assert!(!one.is_empty(), "a single streamline is not empty");

    let seeds = [Vec3Fix::ZERO, Vec3Fix::from_int(5, 0, 0)];
    let lines = generate_streamlines(&fluid, &velocity, &seeds, 2, dt);
    assert!(!lines.is_empty(), "two seeds, two streamlines");
    assert_eq!(lines.len(), seeds.len(), "one polyline per seed");
    // Seed 0 sits on the only particle: two steps of v dt = 1/4 along x.
    let traced = lines.line(0);
    assert_eq!(traced.len(), 3, "the seed plus two steps");
    assert_eq!(
        traced[2],
        Vec3Fix::new(Fix128::from_ratio(1, 2), Fix128::ZERO, Fix128::ZERO)
    );
    // Seed 1 is out of every particle's reach: the line is the seed alone.
    assert_eq!(lines.line(1), &[seeds[1]], "no velocity, no step");

    let no_fluid = generate_streamlines(&[], &[], &seeds, 2, dt);
    assert!(
        !no_fluid.is_empty(),
        "without fluid each seed is still a line"
    );
    assert_eq!(no_fluid.len(), 2);
    println!(
        "Streamlines: 0 seeds -> empty, 2 seeds -> {} lines, line 0 ends at x = {}",
        lines.len(),
        traced[2].x.to_f64()
    );
}

fn main() {
    quadratic_element_nodes();
    streamline_emptiness();
    println!("all closed forms hold");
}
