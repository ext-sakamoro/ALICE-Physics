//! Oracles for collision meshes generated from signed distance fields.
//!
//! # What a collision mesh must be
//!
//! The surface of a solid, to be used as a static collider, must be a **closed
//! orientable 2-manifold with outward normals**: every undirected edge belongs to
//! exactly two triangles that traverse it in opposite directions, the vertices are
//! shared (no two at one place, none unused), a ball's mesh has Euler characteristic
//! `V − E + F = 2`, and the divergence theorem gives its volume as the sum of
//! `a · (b × c) / 6` over triangles (positive when the normals point outward).
//!
//! The closed forms: a ball of radius `R` has volume `4πR³/3`, and every vertex of a
//! mesh made by linear interpolation along an edge of length `ℓ` is within
//! `ℓ²/(8R)` of the sphere (the interpolation error of a function whose second
//! derivative is at most `1/R`). A box whose faces lie half a cell off the lattice has
//! its face planes reproduced exactly (the SDF is linear across a face), so the
//! mesh's bounding box is the box itself.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::collision_mesh_gen::{
    compute_mesh_aabb, generate_collision_mesh, simplify_collision_mesh, CollisionMesh,
    CollisionMeshConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};
use std::collections::BTreeMap;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn p(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn config(res: usize, half: f64) -> CollisionMeshConfig {
    CollisionMeshConfig {
        resolution: res,
        bounds_min: v3(-half, -half, -half),
        bounds_max: v3(half, half, half),
    }
}

/// The ball of radius `r` about the origin, as a fixed-point SDF.
fn ball(r: f64) -> impl Fn(Vec3Fix) -> Fix128 {
    move |q| q.length() - fx(r)
}

/// The box `|x| < hx, |y| < hy, |z| < hz`, exact SDF.
fn cuboid(hx: f64, hy: f64, hz: f64) -> impl Fn(Vec3Fix) -> Fix128 {
    move |q| {
        let d = [q.x.abs() - fx(hx), q.y.abs() - fx(hy), q.z.abs() - fx(hz)];
        let outside = Vec3Fix::new(
            d[0].max(Fix128::ZERO),
            d[1].max(Fix128::ZERO),
            d[2].max(Fix128::ZERO),
        )
        .length();
        let inside = d[0].max(d[1]).max(d[2]).min(Fix128::ZERO);
        outside + inside
    }
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

/// Edge, vertex and orientation facts about a mesh.
struct Topology {
    vertices: usize,
    edges: usize,
    faces: usize,
    /// Every directed edge appears once and its reverse once.
    consistent: bool,
    /// Vertices no triangle uses.
    unused: usize,
}

fn topology(m: &CollisionMesh) -> Topology {
    let mut directed: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut used = vec![false; m.vertices.len()];
    for t in &m.triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            used[a] = true;
            *directed.entry((a, b)).or_default() += 1;
        }
    }
    let consistent = directed
        .iter()
        .all(|(&(a, b), &n)| n == 1 && directed.get(&(b, a)) == Some(&1));
    let edges = directed
        .keys()
        .filter(|&&(a, b)| a < b || !directed.contains_key(&(b, a)))
        .count();
    Topology {
        vertices: m.vertices.len(),
        edges,
        faces: m.triangles.len(),
        consistent,
        unused: used.iter().filter(|u| !**u).count(),
    }
}

/// The volume the triangles enclose (positive for outward normals).
fn signed_volume(m: &CollisionMesh) -> f64 {
    m.triangles
        .iter()
        .map(|t| {
            let (a, b, c) = (
                p(m.vertices[t[0]]),
                p(m.vertices[t[1]]),
                p(m.vertices[t[2]]),
            );
            dot(a, cross(b, c)) / 6.0
        })
        .sum()
}

fn ball_volume(r: f64) -> f64 {
    4.0 / 3.0 * std::f64::consts::PI * r.powi(3)
}

/// A ball's mesh is a closed, consistently oriented sphere: edge-manifold, shared
/// vertices, `V − E + F = 2`.
#[test]
fn a_ball_meshes_to_a_closed_sphere() {
    let m = generate_collision_mesh(ball(1.0), &config(14, 1.5));
    let t = topology(&m);
    assert!(
        t.faces > 100,
        "a ball of 14 cells has many triangles: {}",
        t.faces
    );
    assert!(
        t.consistent,
        "every edge is in two triangles, opposite ways"
    );
    assert_eq!(t.unused, 0, "no vertex is unused");
    assert_eq!(
        t.vertices as i64 - t.edges as i64 + t.faces as i64,
        2,
        "V - E + F of a sphere"
    );
    // No two vertices at the same place.
    let mut places: Vec<[i64; 3]> = m
        .vertices
        .iter()
        .map(|v| {
            let q = p(*v);
            [
                (q[0] * 1e9).round() as i64,
                (q[1] * 1e9).round() as i64,
                (q[2] * 1e9).round() as i64,
            ]
        })
        .collect();
    places.sort_unstable();
    let before = places.len();
    places.dedup();
    assert_eq!(
        places.len(),
        before,
        "vertices are shared, not repeated per cell"
    );
}

/// Every normal points out of the ball, and the enclosed volume converges to
/// `4πR³/3` from inside as the grid is refined.
#[test]
fn the_normals_point_outward_and_the_volume_converges() {
    let r = 1.0;
    let mut errors = Vec::new();
    for res in [10usize, 20, 40] {
        let m = generate_collision_mesh(ball(r), &config(res, 1.5));
        for t in &m.triangles {
            let (a, b, c) = (
                p(m.vertices[t[0]]),
                p(m.vertices[t[1]]),
                p(m.vertices[t[2]]),
            );
            let n = cross(sub(b, a), sub(c, a));
            let centroid = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            assert!(dot(n, centroid) > 0.0, "res {res}: a normal points inward");
        }
        let volume = signed_volume(&m);
        assert!(
            volume > 0.0,
            "res {res}: outward normals enclose a positive volume"
        );
        errors.push((ball_volume(r) - volume) / ball_volume(r));
    }
    assert!(
        errors[2] < errors[1] && errors[1] < errors[0],
        "the volume error shrinks with the cell size: {errors:?}"
    );
    assert!(errors[2].abs() < 0.01, "40 cells give 1%: {}", errors[2]);
    assert!(errors[0].abs() < 0.08, "10 cells give 8%: {}", errors[0]);
}

/// Every vertex is within the interpolation error of the surface: edges are cut
/// by linear interpolation along a lattice edge or a cell diagonal, of length at
/// most `√3·h`, so `|sdf| ≤ (√3 h)² / (8 R)`.
#[test]
fn every_vertex_is_within_the_interpolation_error_of_the_surface() {
    let (r, res, half) = (1.0, 16usize, 1.5);
    let h = 2.0 * half / res as f64;
    let bound = 3.0 * h * h / (8.0 * r);
    let m = generate_collision_mesh(ball(r), &config(res, half));
    let worst = m
        .vertices
        .iter()
        .map(|v| (p(*v).iter().map(|c| c * c).sum::<f64>().sqrt() - r).abs())
        .fold(0.0, f64::max);
    assert!(worst <= bound, "worst {worst} exceeds the bound {bound}");
}

/// A box with faces half a cell off the lattice: the SDF is linear across each face
/// so the face planes are reproduced exactly and the mesh's bounding box is the
/// box. (Faces at ±0.9375 with a cell of 0.125 on `[-1.5, 1.5]`: lattice at multiples
/// of 0.125, faces at odd multiples of 0.0625... 0.9375 = 7.5 cells.)
#[test]
fn a_box_mesh_has_the_boxs_bounding_box() {
    let (hx, hy, hz) = (0.9375, 0.5625, 0.6875);
    let m = generate_collision_mesh(cuboid(hx, hy, hz), &config(24, 1.5));
    let t = topology(&m);
    assert!(t.consistent && t.unused == 0, "closed and consistent");
    assert_eq!(t.vertices as i64 - t.edges as i64 + t.faces as i64, 2);
    let aabb = compute_mesh_aabb(&m);
    let (lo, hi) = (p(aabb.min), p(aabb.max));
    for (k, h) in [hx, hy, hz].into_iter().enumerate() {
        assert!((lo[k] + h).abs() < 1e-9, "min axis {k}: {}", lo[k]);
        assert!((hi[k] - h).abs() < 1e-9, "max axis {k}: {}", hi[k]);
    }
    // The solid is the box with its edges and corners rounded off by at most a cell.
    let box_volume = 8.0 * hx * hy * hz;
    let v = signed_volume(&m);
    assert!(
        v > 0.0 && v <= box_volume + 1e-9,
        "inside the box: {v} of {box_volume}"
    );
    assert!(
        v > 0.93 * box_volume,
        "close to the box: {v} of {box_volume}"
    );
}

/// Nothing to extract: all outside, all inside, or the surface outside the bounds.
#[test]
fn a_field_without_a_surface_in_the_bounds_gives_an_empty_mesh() {
    for (name, field) in [
        (
            "all outside",
            Box::new(|_: Vec3Fix| fx(1.0)) as Box<dyn Fn(Vec3Fix) -> Fix128>,
        ),
        ("all inside", Box::new(|_: Vec3Fix| fx(-1.0))),
        ("far away", Box::new(ball_at_far())),
    ] {
        let m = generate_collision_mesh(field, &config(8, 1.0));
        assert!(m.vertices.is_empty() && m.triangles.is_empty(), "{name}");
    }
}

fn ball_at_far() -> impl Fn(Vec3Fix) -> Fix128 {
    move |q| (q - v3(30.0, 0.0, 0.0)).length() - fx(1.0)
}

/// The bounding box ignores a vertex no triangle uses, and is empty-safe.
#[test]
fn the_bounding_box_covers_the_triangles_not_unused_vertices() {
    let m = CollisionMesh {
        vertices: vec![
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 2.0, 0.0),
            v3(50.0, 50.0, 50.0), // unused
        ],
        triangles: vec![[0, 1, 2]],
    };
    let a = compute_mesh_aabb(&m);
    assert_eq!(p(a.min), [0.0, 0.0, 0.0]);
    assert_eq!(p(a.max), [1.0, 2.0, 0.0]);
    let empty = compute_mesh_aabb(&CollisionMesh {
        vertices: vec![],
        triangles: vec![],
    });
    assert_eq!(p(empty.min), [0.0; 3]);
    assert_eq!(p(empty.max), [0.0; 3]);
}

/// Simplification reaches the target (each collapse of a closed mesh's edge removes
/// two triangles, so the result is the target or one fewer), keeps the surface a
/// closed outward sphere, drops the vertices it collapsed, and keeps the volume.
#[test]
fn simplifying_a_ball_keeps_a_closed_outward_sphere() {
    let m = generate_collision_mesh(ball(1.0), &config(16, 1.5));
    let full = m.triangles.len();
    let target = full / 2;
    let s = simplify_collision_mesh(&m, target);
    assert!(
        s.triangles.len() <= target && s.triangles.len() + 2 >= target,
        "{} triangles for a target of {target}",
        s.triangles.len()
    );
    let t = topology(&s);
    assert!(t.consistent, "still closed and consistently oriented");
    assert_eq!(t.unused, 0, "collapsed vertices are dropped");
    assert_eq!(t.vertices as i64 - t.edges as i64 + t.faces as i64, 2);
    for tri in &s.triangles {
        let (a, b, c) = (
            p(s.vertices[tri[0]]),
            p(s.vertices[tri[1]]),
            p(s.vertices[tri[2]]),
        );
        let n = cross(sub(b, a), sub(c, a));
        let centroid = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        assert!(dot(n, centroid) > 0.0, "a collapsed triangle points inward");
        assert!(
            n.iter().map(|c| c * c).sum::<f64>() > 0.0,
            "no degenerate triangle"
        );
    }
    let (v0, v1) = (signed_volume(&m), signed_volume(&s));
    assert!(
        (v1 - v0).abs() < 0.1 * v0,
        "half the triangles keep the volume to 10%: {v1} against {v0}"
    );
    // A bounding box of the simplified mesh is inside the original's.
    let (a0, a1) = (compute_mesh_aabb(&m), compute_mesh_aabb(&s));
    for (lo0, lo1) in [
        (a0.min.x, a1.min.x),
        (a0.min.y, a1.min.y),
        (a0.min.z, a1.min.z),
    ] {
        assert!(lo1 >= lo0 - fx(1e-9));
    }
}

/// Asking for more triangles than there are returns the mesh unchanged; asking for
/// fewer than a closed mesh can have (4, a tetrahedron) stops at a valid mesh.
#[test]
fn simplifying_never_breaks_the_mesh() {
    let m = generate_collision_mesh(ball(1.0), &config(8, 1.5));
    let same = simplify_collision_mesh(&m, m.triangles.len() + 10);
    assert_eq!(same.triangles, m.triangles);
    assert_eq!(same.vertices.len(), m.vertices.len());
    // Asked for nothing, a closed ball stops at a tetrahedron: four triangles that still
    // enclose a volume, not a flat two-triangle pillow.
    let tiny = simplify_collision_mesh(&m, 0);
    let t = topology(&tiny);
    assert_eq!(tiny.triangles.len(), 4, "a tetrahedron");
    assert!(t.consistent);
    assert_eq!(t.unused, 0);
    assert_eq!(t.vertices as i64 - t.edges as i64 + t.faces as i64, 2);
    assert!(
        signed_volume(&tiny) > 0.1,
        "it encloses a volume: {}",
        signed_volume(&tiny)
    );
}

// ---------------------------------------------------------------------------
// As a static collider
// ---------------------------------------------------------------------------

use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

/// A sphere body above the top of a ball's mesh is lifted to one radius from the
/// surface. The mesh is inscribed in the ball (its vertices are within `3h²/(8R)`
/// of it, a face sags a little more), so the lifted height is the ideal one minus at
/// most that sag.
#[test]
fn a_body_rests_on_the_mesh_of_a_ball() {
    let (big_r, body_r) = (1.0, 0.5);
    let (res, half) = (30usize, 1.5);
    let h = 2.0 * half / res as f64;
    let m = generate_collision_mesh(ball(big_r), &config(res, half));
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 1,
        ..SolverConfig::default()
    });
    w.add_static_collider(m.to_static_collider());
    // 0.1 inside the contact distance above the ball's top.
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, big_r + body_r - 0.1, 0.0), Fix128::ONE),
        fx(body_r),
    );
    w.step(fx(1.0 / 60.0));
    let y = w.get_body(b).expect("body").position.y.to_f64();
    // Ideal: lifted to R + r = 1.5. The mesh surface is up to `sag` below the ball.
    let sag = 3.0 * h * h / 8.0 * 4.0;
    assert!(
        y <= big_r + body_r + 1e-3 && y >= big_r + body_r - sag,
        "lifted to {y}, the ideal height is {} and the mesh sags at most {sag}",
        big_r + body_r
    );
}

// ---------------------------------------------------------------------------
// Topology that simplification must keep
// ---------------------------------------------------------------------------

/// The torus with ring radius `big` and tube radius `small` about the y axis.
fn torus(big: f64, small: f64) -> impl Fn(Vec3Fix) -> Fix128 {
    move |q| {
        let ring = (q.x * q.x + q.z * q.z).sqrt() - fx(big);
        (ring * ring + q.y * q.y).sqrt() - fx(small)
    }
}

/// A torus has Euler characteristic 0 and a hole. Collapsing edges across the tube
/// (which the link condition forbids) would close the hole or pinch the surface, so
/// after heavy simplification the mesh must still be edge-manifold, oriented, and
/// have `V − E + F = 0`; and no triangle may turn over, so it still encloses about the
/// torus's volume `2π²Rr²`.
#[test]
fn simplifying_a_torus_keeps_its_hole() {
    let (big, small) = (1.0, 0.35);
    let m = generate_collision_mesh(torus(big, small), &config(28, 1.6));
    let t0 = topology(&m);
    assert!(t0.consistent && t0.unused == 0);
    assert_eq!(
        t0.vertices as i64 - t0.edges as i64 + t0.faces as i64,
        0,
        "a torus"
    );
    let torus_volume = 2.0 * std::f64::consts::PI.powi(2) * big * small * small;
    for fraction in [4usize, 10, 20] {
        let s = simplify_collision_mesh(&m, m.triangles.len() / fraction);
        // No triangle turned over: every normal agrees with the SDF's gradient at the
        // triangle (the radial direction from the tube's core circle).
        for tri in &s.triangles {
            let (a, b, c) = (
                p(s.vertices[tri[0]]),
                p(s.vertices[tri[1]]),
                p(s.vertices[tri[2]]),
            );
            let n = cross(sub(b, a), sub(c, a));
            let centre = [
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            ];
            let rho = (centre[0] * centre[0] + centre[2] * centre[2]).sqrt();
            let gradient = [
                centre[0] * (1.0 - big / rho),
                centre[1],
                centre[2] * (1.0 - big / rho),
            ];
            assert!(
                dot(n, gradient) > 0.0,
                "1/{fraction}: a triangle turned over"
            );
        }
        let t = topology(&s);
        assert!(
            t.consistent,
            "1/{fraction}: still closed and consistently oriented"
        );
        assert_eq!(t.unused, 0);
        assert_eq!(
            t.vertices as i64 - t.edges as i64 + t.faces as i64,
            0,
            "1/{fraction}: the hole is still there"
        );
        let v = signed_volume(&s);
        assert!(
            (v - torus_volume).abs() < 0.2 * torus_volume,
            "1/{fraction}: volume {v} against {torus_volume}"
        );
    }
}

/// An open mesh: a flat 8×8 square of triangles. Its border does not move: every
/// border vertex keeps its place, the number of border edges is unchanged, and the
/// interior is thinned.
#[test]
fn simplifying_an_open_mesh_keeps_its_border() {
    let n = 8usize;
    let mut vertices = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            vertices.push(v3(i as f64, j as f64, 0.0));
        }
    }
    let at = |i: usize, j: usize| j * (n + 1) + i;
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            triangles.push([at(i, j), at(i + 1, j), at(i + 1, j + 1)]);
            triangles.push([at(i, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    let m = CollisionMesh {
        vertices,
        triangles,
    };
    let border_of = |m: &CollisionMesh| -> (Vec<[i64; 3]>, usize) {
        let mut uses: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for t in &m.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *uses.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let mut points: Vec<[i64; 3]> = uses
            .iter()
            .filter(|(_, &c)| c == 1)
            .flat_map(|(&(a, b), _)| [a, b])
            .map(|i| {
                let q = p(m.vertices[i]);
                [
                    (q[0] * 1e6) as i64,
                    (q[1] * 1e6) as i64,
                    (q[2] * 1e6) as i64,
                ]
            })
            .collect();
        points.sort_unstable();
        points.dedup();
        (points, uses.values().filter(|&&c| c == 1).count())
    };
    let (border_before, edges_before) = border_of(&m);
    let s = simplify_collision_mesh(&m, m.triangles.len() / 3);
    let (border_after, edges_after) = border_of(&s);
    assert!(s.triangles.len() < m.triangles.len() && s.triangles.len() >= 1);
    assert_eq!(
        border_before, border_after,
        "the border vertices did not move"
    );
    assert_eq!(edges_before, edges_after, "no border edge was collapsed");
    assert_eq!(topology(&s).unused, 0);
}

/// The static collider has every triangle of the mesh.
#[test]
fn the_static_collider_holds_every_triangle() {
    use alice_physics::static_collider::StaticCollider;
    let m = generate_collision_mesh(ball(1.0), &config(10, 1.5));
    match m.to_static_collider() {
        StaticCollider::TriMesh(t) => assert_eq!(t.triangle_count(), m.triangles.len()),
        _ => panic!("a mesh is a triangle mesh collider"),
    }
}

/// A resolution below two still gives the two-cell mesh (and does not divide by 0).
#[test]
fn a_resolution_below_two_is_raised_to_two() {
    let two = generate_collision_mesh(ball(1.0), &config(2, 1.5));
    for res in [0usize, 1] {
        let m = generate_collision_mesh(ball(1.0), &config(res, 1.5));
        assert_eq!(m.vertices, two.vertices, "resolution {res}");
        assert_eq!(m.triangles, two.triangles, "resolution {res}");
    }
}

/// A tetrahedron with one face stellated (a vertex `E` added under the base `ABC`):
/// each base edge is the shortest edge of the mesh, and each has a common neighbour
/// that is not an apex of its two triangles (`AB` has `C`, `D` and `E` in common
/// but only `D` and `E` are apexes). Collapsing it would pinch the surface into a
/// non-manifold, so the link condition must skip the base edges and the result must
/// stay a closed, oriented sphere.
#[test]
fn simplification_does_not_collapse_an_edge_that_would_pinch_the_surface() {
    let vertices = vec![
        v3(0.0, 0.0, 0.0),  // A
        v3(1.0, 0.0, 0.0),  // B
        v3(0.5, 0.9, 0.0),  // C
        v3(0.5, 0.3, 1.5),  // D, above the base
        v3(0.5, 0.3, -1.5), // E, below the base
    ];
    let (a, b, c, d, e) = (0usize, 1, 2, 3, 4);
    let mut triangles = vec![
        [a, b, e],
        [b, c, e],
        [c, a, e],
        [a, b, d],
        [b, c, d],
        [c, a, d],
    ];
    // Wind every triangle outward: away from the centroid of the vertices.
    let centre = {
        let sum = vertices.iter().fold([0.0; 3], |acc, v| {
            let q = p(*v);
            [acc[0] + q[0], acc[1] + q[1], acc[2] + q[2]]
        });
        [sum[0] / 5.0, sum[1] / 5.0, sum[2] / 5.0]
    };
    for t in &mut triangles {
        let (pa, pb, pc) = (p(vertices[t[0]]), p(vertices[t[1]]), p(vertices[t[2]]));
        let n = cross(sub(pb, pa), sub(pc, pa));
        let mid = [
            (pa[0] + pb[0] + pc[0]) / 3.0 - centre[0],
            (pa[1] + pb[1] + pc[1]) / 3.0 - centre[1],
            (pa[2] + pb[2] + pc[2]) / 3.0 - centre[2],
        ];
        if dot(n, mid) < 0.0 {
            t.swap(1, 2);
        }
    }
    let m = CollisionMesh {
        vertices,
        triangles,
    };
    let t0 = topology(&m);
    assert!(t0.consistent);
    assert_eq!(t0.vertices as i64 - t0.edges as i64 + t0.faces as i64, 2);
    let s = simplify_collision_mesh(&m, 4);
    let t = topology(&s);
    assert!(t.consistent, "still closed and consistently oriented");
    assert_eq!(t.unused, 0);
    assert_eq!(
        t.vertices as i64 - t.edges as i64 + t.faces as i64,
        2,
        "still a sphere"
    );
}
