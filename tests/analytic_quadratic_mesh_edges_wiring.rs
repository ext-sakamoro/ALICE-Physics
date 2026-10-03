//! Oracles for `quadratic_elastic_fem::QuadraticMesh::{edge_count, edge_node}`
//! (`examples/quadratic_mesh_edge_nodes.rs`).
//!
//! Counted and numbered independently of the builder, straight off the tetrahedron lists:
//! * distinct edges = `|{ {a,b} : a, b corners of one tet }|` (single tet 6; two tets glued on a face `6+6-3 = 9`;
//!   a fan of `n` tets around one edge: `n + (n+1) ... `, checked by direct set construction)
//! * edge nodes are numbered `corner_count, corner_count+1, ...` in order of first appearance, scanning
//!   elements in order and each element's pairs `(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)`
//! * the node of an edge sits at the exact midpoint of its two corners (`1/2` is exact in binary)
//! * `edge_node` is symmetric and `None` for pairs that are not an edge of any tet
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::QuadraticMesh;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use std::collections::BTreeMap;

fn mesh(vertices: &[[f32; 3]], tets: &[[u32; 4]]) -> SdfTetMesh {
    let mut m = SdfTetMesh::default();
    m.vertices.extend_from_slice(vertices);
    for t in tets {
        m.tets.push(Tetrahedron { vertices: *t });
    }
    m
}

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

/// independent edge numbering: key -> node index, first appearance order
fn reference(m: &SdfTetMesh) -> BTreeMap<(u32, u32), u32> {
    let mut map = BTreeMap::new();
    let mut next = m.vertices.len() as u32;
    for t in &m.tets {
        for &(i, j) in &PAIRS {
            let (a, b) = (t.vertices[i], t.vertices[j]);
            let key = (a.min(b), a.max(b));
            map.entry(key).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
        }
    }
    map
}

fn check(m: &SdfTetMesh) {
    let q = QuadraticMesh::from_tet_mesh(m).unwrap();
    let want = reference(m);
    assert_eq!(q.edge_count(), want.len());
    assert_eq!(q.corner_count(), m.vertices.len());
    assert_eq!(q.node_count(), q.corner_count() + q.edge_count());
    let n = m.vertices.len() as u32;
    for a in 0..n {
        for b in 0..n {
            let expect = if a == b {
                None
            } else {
                want.get(&(a.min(b), a.max(b))).copied()
            };
            assert_eq!(q.edge_node(a, b), expect, "edge ({a},{b})");
            if let Some(node) = expect {
                // midpoint, exactly
                let p = q.node_position(node).unwrap();
                let (pa, pb) = (q.node_position(a).unwrap(), q.node_position(b).unwrap());
                for k in 0..3 {
                    assert_eq!(
                        p[k],
                        (pa[k] + pb[k]) * Fix128::from_ratio(1, 2),
                        "node {node} axis {k}"
                    );
                }
                assert!(node >= n && (node as usize) < q.node_count());
            }
        }
    }
    // element node tables agree with edge_node per slot
    for (e, t) in m.tets.iter().enumerate() {
        let nodes = q.element_nodes(e).unwrap();
        assert_eq!(&nodes[..4], &t.vertices);
        for (slot, &(i, j)) in PAIRS.iter().enumerate() {
            assert_eq!(
                Some(nodes[4 + slot]),
                q.edge_node(t.vertices[i], t.vertices[j])
            );
        }
    }
    // out of range lookups
    assert_eq!(q.edge_node(n, 0), None);
    assert_eq!(q.edge_node(u32::MAX, u32::MAX - 1), None);
}

#[test]
fn single_tet_has_six_edges_numbered_in_pair_order() {
    let m = mesh(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        &[[0, 1, 2, 3]],
    );
    let q = QuadraticMesh::from_tet_mesh(&m).unwrap();
    assert_eq!(q.edge_count(), 6);
    assert_eq!(q.node_count(), 10);
    // closed form: pairs in order -> nodes 4..10
    for (slot, &(i, j)) in PAIRS.iter().enumerate() {
        assert_eq!(q.edge_node(i as u32, j as u32), Some(4 + slot as u32));
        assert_eq!(q.edge_node(j as u32, i as u32), Some(4 + slot as u32));
    }
    assert_eq!(q.edge_node(2, 2), None);
    check(&m);
}

#[test]
fn two_tets_glued_on_a_face_have_nine_edges() {
    let m = mesh(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        &[[0, 1, 2, 3], [1, 2, 3, 4]],
    );
    let q = QuadraticMesh::from_tet_mesh(&m).unwrap();
    assert_eq!(q.edge_count(), 9, "6 + 6 - 3 shared");
    assert_eq!(q.node_count(), 14);
    // hand numbering: tet0 -> 5..=10 as (01)(02)(03)(12)(13)(23); tet1 adds (14)=11 (24)=12 (34)=13
    assert_eq!(q.edge_node(1, 4), Some(11));
    assert_eq!(q.edge_node(4, 2), Some(12));
    assert_eq!(q.edge_node(3, 4), Some(13));
    assert_eq!(q.edge_node(0, 4), None, "0 and 4 never share a tet");
    check(&m);
}

#[test]
fn vertex_order_inside_a_tet_does_not_change_the_edge_set() {
    let v = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let a = QuadraticMesh::from_tet_mesh(&mesh(&v, &[[0, 1, 2, 3], [1, 2, 3, 4]])).unwrap();
    let b = QuadraticMesh::from_tet_mesh(&mesh(&v, &[[0, 1, 2, 3], [4, 3, 2, 1]])).unwrap();
    assert_eq!(a.edge_count(), b.edge_count());
    for x in 0..5u32 {
        for y in 0..5u32 {
            assert_eq!(
                a.edge_node(x, y).is_some(),
                b.edge_node(x, y).is_some(),
                "({x},{y})"
            );
        }
    }
    check(&mesh(&v, &[[0, 1, 2, 3], [4, 3, 2, 1]]));
}

#[test]
fn fan_of_tets_around_a_shared_edge_and_a_disconnected_pair() {
    // axis edge (0,1) shared by 3 tets: apexes 2,3,4,5 around it in sequence
    let v = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let fan = mesh(&v, &[[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 5]]);
    // distinct edges: (01) + each apex to 0 and 1 (4*2) + consecutive apex pairs (23)(34)(45) = 1 + 8 + 3
    assert_eq!(QuadraticMesh::from_tet_mesh(&fan).unwrap().edge_count(), 12);
    check(&fan);
    // two tets that share only a vertex (no common edge): 6 + 6 edges
    let w = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ];
    let touching = mesh(&w, &[[0, 1, 2, 3], [0, 4, 5, 6]]);
    assert_eq!(
        QuadraticMesh::from_tet_mesh(&touching)
            .unwrap()
            .edge_count(),
        12
    );
    check(&touching);
}

#[test]
fn a_larger_mesh_keeps_every_index_dense_and_unique() {
    // Kuhn split of a 2x2x2 cube grid: 6 tets per cube, 8 cubes
    let n = 3usize;
    let id = |x: usize, y: usize, z: usize| (x + n * (y + n * z)) as u32;
    let mut verts = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                verts.push([x as f32, y as f32, z as f32]);
            }
        }
    }
    let mut tets = Vec::new();
    for z in 0..n - 1 {
        for y in 0..n - 1 {
            for x in 0..n - 1 {
                let c = |dx, dy, dz| id(x + dx, y + dy, z + dz);
                for path in [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ] {
                    let mut p = [0usize; 3];
                    let mut cur = [0usize; 3];
                    let mut tet = [c(0, 0, 0), 0, 0, 0];
                    for (k, &axis) in path.iter().enumerate() {
                        cur[axis] += 1;
                        p[k] = axis;
                        tet[k + 1] = c(cur[0], cur[1], cur[2]);
                    }
                    tets.push(tet);
                }
            }
        }
    }
    let m = mesh(&verts, &tets);
    let q = QuadraticMesh::from_tet_mesh(&m).unwrap();
    assert_eq!(q.element_count(), 48);
    check(&m);
    // every edge node index in corner_count..node_count is used exactly once
    let mut seen = vec![0u32; q.node_count()];
    for a in 0..m.vertices.len() as u32 {
        for b in (a + 1)..m.vertices.len() as u32 {
            if let Some(node) = q.edge_node(a, b) {
                seen[node as usize] += 1;
            }
        }
    }
    assert!(seen[..q.corner_count()].iter().all(|&c| c == 0));
    assert!(seen[q.corner_count()..].iter().all(|&c| c == 1));
}
