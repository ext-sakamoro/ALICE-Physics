//! Quadratic (P2) mesh edge nodes: `QuadraticMesh::{edge_count, edge_node}`.
//!
//! Two tetrahedra glued on the face `(1,2,3)` have `6 + 6 - 3 = 9` distinct edges, so the P2 mesh
//! has `5` corners + `9` edge nodes = `14` nodes. Edge nodes are numbered from `corner_count`
//! by first appearance in element order, `edge_node(a, b)` does not care about argument order, and
//! its node sits exactly at the midpoint of the two corners.
//!
//! ```bash
//! cargo run --release --example quadratic_mesh_edge_nodes --features std
//! ```

use alice_physics::quadratic_elastic_fem::QuadraticMesh;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn main() {
    let mut mesh = SdfTetMesh::default();
    for v in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ] {
        mesh.vertices.push(v);
    }
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh.tets.push(Tetrahedron {
        vertices: [1, 2, 3, 4],
    });

    let q = QuadraticMesh::from_tet_mesh(&mesh).expect("two valid tetrahedra");
    println!(
        "corners {}, edge nodes {}, total {}, elements {}",
        q.corner_count(),
        q.edge_count(),
        q.node_count(),
        q.element_count()
    );
    for (a, b) in [(0u32, 1u32), (1, 0), (3, 4), (0, 4)] {
        match q.edge_node(a, b) {
            Some(n) => {
                let p = q.node_position(n).expect("edge node has a position");
                println!(
                    "edge ({a},{b}) -> node {n} at ({:.2}, {:.2}, {:.2})",
                    p[0].to_f64(),
                    p[1].to_f64(),
                    p[2].to_f64()
                );
            }
            None => println!("edge ({a},{b}) is not in the mesh"),
        }
    }
}
