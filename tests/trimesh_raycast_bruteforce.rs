//! `TriMesh::raycast` against a brute-force loop over every triangle.
//!
//! oracle (independent of the BVH and of the crate's triangle test): for a ray and a reach, test
//! each triangle in index order with the exact reference `support/exact_ray.rs` (big integers,
//! Cramer's rule) and keep the nearest hit, the smallest index on an exact tie of t; the hit's
//! point is `ray.at(t)` and its normal the triangle's unit normal turned to face the ray. The mesh query
//! must report the same t, triangle index, point and normal, bit for bit, whatever acceleration it
//! uses. Meshes are seeded random triangles at sub-unit scale (where a coarse node bound cannot
//! separate anything) and at a scale of tens of units; rays are aimed at points inside triangles
//! (hits) and at random (mostly misses), plus rays through shared edges and vertices, rays along an
//! axis, and a reach that ends exactly at a hit.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::raycast::{Ray, RayHit};
use alice_physics_core::trimesh::{TriMesh, Triangle};

#[path = "support/exact_ray.rs"]
mod exact_ray;
use exact_ray::{exact_nearest, exact_t};

/// xorshift64*: a fixed sequence, the same on every machine
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// An integer in -range..=range
    fn int(&mut self, range: i64) -> i64 {
        (self.next() % (2 * range as u64 + 1)) as i64 - range
    }

    /// A fixed-point value num / denom, num in -range..=range
    fn fix(&mut self, range: i64, denom: i64) -> Fix128 {
        Fix128::from_ratio(self.int(range), denom)
    }

    fn vec(&mut self, range: i64, denom: i64) -> Vec3Fix {
        Vec3Fix::new(
            self.fix(range, denom),
            self.fix(range, denom),
            self.fix(range, denom),
        )
    }
}

fn brute(tris: &[Triangle], ray: &Ray, max_t: Fix128) -> Option<RayHit> {
    exact_nearest(tris, ray, max_t).map(|(t, i)| {
        let n = tris[i].unit_normal();
        RayHit {
            t,
            point: ray.at(t),
            normal: if n.dot(ray.direction) > Fix128::ZERO {
                -n
            } else {
                n
            },
            body_index: i,
        }
    })
}

/// The nearest t and how many triangles reach it
fn ties(tris: &[Triangle], ray: &Ray, max_t: Fix128) -> usize {
    let Some(b) = brute(tris, ray, max_t) else {
        return 0;
    };
    tris.iter()
        .filter(|t| exact_t(ray, t, max_t) == Some(b.t))
        .count()
}

fn random_mesh(rng: &mut Rng, n: usize, range: i64, denom: i64, size: i64) -> Vec<Triangle> {
    (0..n)
        .map(|_| {
            let c = rng.vec(range, denom);
            Triangle::new(
                c + rng.vec(size, denom),
                c + rng.vec(size, denom),
                c + rng.vec(size, denom),
            )
        })
        .collect()
}

/// Rays aimed at a point inside a random triangle, and rays in random directions
fn rays(rng: &mut Rng, tris: &[Triangle], n: usize, range: i64, denom: i64) -> Vec<Ray> {
    (0..n)
        .map(|k| {
            let o = rng.vec(4 * range, denom);
            let d = if k % 2 == 0 {
                let t = &tris[(rng.next() as usize) % tris.len()];
                let (a, b) = (Fix128::from_ratio(1, 3), Fix128::from_ratio(1, 4));
                let target = t.v0 + (t.v1 - t.v0) * a + (t.v2 - t.v0) * b;
                target - o
            } else {
                rng.vec(1000, 1000)
            };
            Ray::new(o, d)
        })
        .collect()
}

fn same(a: Option<RayHit>, b: Option<RayHit>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.t == b.t && a.body_index == b.body_index && a.point == b.point && a.normal == b.normal
        }
        _ => false,
    }
}

#[test]
fn the_mesh_query_equals_the_brute_force_loop_when_the_nearest_hit_is_unique() {
    let mut compared = 0;
    let mut hits = 0;
    for (seed, n, range, denom, size) in [
        // sub-unit: a figure-sized cloud (coordinates within +-0.8, triangles up to 0.05)
        (0x9e37_79b9_7f4a_7c15u64, 400usize, 800i64, 1000i64, 50i64),
        // tens of units
        (0xd1b5_4a32_d192_ed03, 400, 40, 1, 3),
    ] {
        let mut rng = Rng(seed);
        let tris = random_mesh(&mut rng, n, range, denom, size);
        let mesh = TriMesh::from_triangles(tris.clone());
        let max_t = Fix128::from_int(1000);
        for ray in rays(&mut rng, &tris, 400, range, denom) {
            if ties(&tris, &ray, max_t) > 1 {
                continue;
            }
            let want = brute(&tris, &ray, max_t);
            let got = mesh.raycast(&ray, max_t);
            assert!(same(got, want), "seed {seed:x}: got {got:?} want {want:?}");
            compared += 1;
            hits += usize::from(want.is_some());
        }
    }
    assert!(compared >= 700, "{compared} rays compared");
    assert!(hits >= 300, "{hits} hits among them");
}

#[test]
fn the_reach_and_axis_parallel_rays_match_the_brute_force_loop() {
    // a sub-unit grid of triangles in the plane y = 0.25, cells of 1/16
    let mut tris = Vec::new();
    for i in -8..8 {
        for j in -8..8 {
            let p = |a: i64, b: i64| {
                Vec3Fix::new(
                    Fix128::from_ratio(a, 16),
                    Fix128::from_ratio(1, 4),
                    Fix128::from_ratio(b, 16),
                )
            };
            tris.push(Triangle::new(p(i, j), p(i + 1, j), p(i, j + 1)));
            tris.push(Triangle::new(p(i + 1, j), p(i + 1, j + 1), p(i, j + 1)));
        }
    }
    let mesh = TriMesh::from_triangles(tris.clone());
    let down = Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO);
    let mut compared = 0;
    for k in 0..64 {
        // centres of cells (unique hits), straight down: x and z components exactly 0
        let x = Fix128::from_ratio(2 * (k % 16) - 15, 32) / Fix128::from_int(2);
        let z = Fix128::from_ratio(2 * (k / 16) - 3, 32);
        let ray = Ray::new(Vec3Fix::new(x, Fix128::ONE, z), down);
        for max_t in [
            Fix128::from_int(10),
            Fix128::from_ratio(3, 4),
            Fix128::from_ratio(1, 2),
        ] {
            if ties(&tris, &ray, max_t) > 1 {
                continue;
            }
            let want = brute(&tris, &ray, max_t);
            assert!(
                same(mesh.raycast(&ray, max_t), want),
                "k {k} reach {max_t:?}"
            );
            compared += 1;
        }
    }
    // the reach ending exactly at the plane (t = 3/4) still hits; just short of it does not
    let ray = Ray::new(
        Vec3Fix::new(
            Fix128::from_ratio(1, 64),
            Fix128::ONE,
            Fix128::from_ratio(1, 64),
        ),
        down,
    );
    assert!(brute(&tris, &ray, Fix128::from_ratio(3, 4)).is_some());
    assert!(same(
        mesh.raycast(&ray, Fix128::from_ratio(3, 4)),
        brute(&tris, &ray, Fix128::from_ratio(3, 4))
    ));
    assert!(compared >= 100, "{compared}");
}

// The tie rule: two triangles sharing an edge are met at the same t by a ray through the edge; the
// smaller index is the answer (the brute-force loop's rule)
#[test]
fn an_exact_tie_goes_to_the_smallest_triangle_index() {
    let p = |a: i64, b: i64| {
        Vec3Fix::new(
            Fix128::from_ratio(a, 8),
            Fix128::ZERO,
            Fix128::from_ratio(b, 8),
        )
    };
    // a quad split along its diagonal, in both index orders
    for flip in [false, true] {
        let a = Triangle::new(p(0, 0), p(1, 0), p(1, 1));
        let b = Triangle::new(p(0, 0), p(1, 1), p(0, 1));
        let tris = if flip { vec![b, a] } else { vec![a, b] };
        let mesh = TriMesh::from_triangles(tris.clone());
        // straight down through a point of the shared diagonal
        let ray = Ray::new(
            Vec3Fix::new(
                Fix128::from_ratio(1, 16),
                Fix128::ONE,
                Fix128::from_ratio(1, 16),
            ),
            Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO),
        );
        let max_t = Fix128::from_int(10);
        assert_eq!(
            ties(&tris, &ray, max_t),
            2,
            "both triangles meet the ray at the same t"
        );
        let got = mesh.raycast(&ray, max_t).expect("a hit");
        assert_eq!(got.body_index, 0, "flip {flip}: the smallest index");
        assert!(same(Some(got), brute(&tris, &ray, max_t)));
    }
}

// The walk prunes: the nodes tested per ray are far fewer than the BVH's nodes (and never 0 when
// the mesh has any), on a sub-unit scene and on a scene of tens of units; and the any-hit query
// agrees with the brute-force loop
#[test]
fn the_walk_tests_few_nodes_and_the_any_hit_query_agrees() {
    for (seed, n, range, denom, size) in [
        (0x2545_f491_4f6c_dd1du64, 50_000usize, 800i64, 1000i64, 8i64),
        (0x94d0_49bb_1331_11eb, 50_000, 60, 1, 1),
    ] {
        let mut rng = Rng(seed);
        let tris = random_mesh(&mut rng, n, range, denom, size);
        let mesh = TriMesh::from_triangles(tris.clone());
        let max_t = Fix128::from_int(1000);
        let probe = mesh.raycast_visits(
            &Ray::new(
                Vec3Fix::ZERO,
                Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            ),
            max_t,
        );
        assert!(probe.1 > 0);
        let rs = rays(&mut rng, &tris, 200, range, denom);
        let mut total = 0usize;
        let mut hits = 0usize;
        for ray in &rs {
            let (hit, visits) = mesh.raycast_visits(ray, max_t);
            assert!(visits > 0);
            total += visits;
            hits += usize::from(hit.is_some());
            // the any-hit answer against the brute-force loop (some triangle within reach or none)
            let any = tris.iter().any(|t| exact_t(ray, t, max_t).is_some());
            assert_eq!(mesh.occluded(ray, max_t), any);
        }
        let mean = total / rs.len();
        // a BVH of n triangles has fewer than 2 n nodes; a walk that tested every node would be 2 n
        assert!(
            mean * 50 < n,
            "seed {seed:x}: mean {mean} nodes tested per ray for {n} triangles"
        );
        assert!(hits >= 50, "{hits} hits");
    }
}

// Origins inside the mesh's bounds, and the end of a ray's interval at t = 0: a ray that starts on
// a triangle's vertex meets it at t = 0
#[test]
fn origins_inside_the_bounds_and_a_start_on_a_triangle_match_the_brute_force_loop() {
    let mut rng = Rng(0x6a09_e667_f3bc_c909);
    let tris = random_mesh(&mut rng, 2000, 500, 1000, 40);
    let mesh = TriMesh::from_triangles(tris.clone());
    let max_t = Fix128::from_int(10);
    let mut compared = 0;
    for _ in 0..600 {
        let ray = Ray::new(rng.vec(500, 1000), rng.vec(1000, 1000));
        if ties(&tris, &ray, max_t) > 1 {
            continue;
        }
        assert!(same(mesh.raycast(&ray, max_t), brute(&tris, &ray, max_t)));
        assert_eq!(
            mesh.occluded(&ray, max_t),
            brute(&tris, &ray, max_t).is_some()
        );
        compared += 1;
    }
    assert!(compared >= 500, "{compared}");
    // starting exactly on a triangle's vertex (s = 0, so u = v = t = 0) and leaving along its
    // normal: the vertex is also a corner of the triangle's bounds, the slab's exit at t = 0
    let t = &tris[7];
    let ray = Ray::new(t.v0, t.normal());
    let want = brute(&tris, &ray, max_t);
    assert!(want.is_some_and(|h| h.t.is_zero()), "{want:?}");
    assert!(same(mesh.raycast(&ray, max_t), want));
    assert!(mesh.occluded(&ray, max_t));
}

// A mesh entirely behind the ray's origin: the walk tests the root's bounds and stops there (the
// slab interval ends before t = 0 on some axis), with no hit
#[test]
fn a_ray_pointing_away_from_the_mesh_tests_only_the_root() {
    let mut rng = Rng(0x3c6e_f372_fe94_f82b);
    let tris = random_mesh(&mut rng, 5000, 500, 1000, 30);
    let mesh = TriMesh::from_triangles(tris.clone());
    let max_t = Fix128::from_int(10);
    let mut checked = 0;
    for k in 0..6 {
        // from just outside one face of the bounds, pointing straight away from it
        let (o, d) = match k {
            0 => (
                Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
                Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            ),
            1 => (
                Vec3Fix::new(-Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
                Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            ),
            2 => (
                Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO),
                Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
            ),
            3 => (
                Vec3Fix::new(Fix128::ZERO, -Fix128::from_int(2), Fix128::ZERO),
                Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO),
            ),
            4 => (
                Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE) * Fix128::from_int(2),
                Vec3Fix::new(Fix128::ONE, Fix128::ONE, Fix128::ONE),
            ),
            _ => (
                Vec3Fix::new(Fix128::ONE, -Fix128::ONE, Fix128::ONE) * -Fix128::from_int(2),
                Vec3Fix::new(-Fix128::ONE, Fix128::ONE, -Fix128::ONE),
            ),
        };
        let ray = Ray::new(o, d);
        assert!(brute(&tris, &ray, max_t).is_none());
        let (hit, visits) = mesh.raycast_visits(&ray, max_t);
        assert!(hit.is_none());
        assert_eq!(visits, 1, "case {k}: only the root is tested");
        assert!(!mesh.occluded(&ray, max_t));
        checked += 1;
    }
    assert_eq!(checked, 6);
}

// A ray parallel to an axis that passes beside the mesh (outside its bounds on an axis the ray does
// not move along): the root's slab on that axis does not contain the origin, so only the root is
// tested
#[test]
fn a_ray_beside_the_mesh_along_an_axis_tests_only_the_root() {
    let mut rng = Rng(0xa54f_f53a_5f1d_36f1);
    let tris = random_mesh(&mut rng, 5000, 500, 1000, 30);
    let mesh = TriMesh::from_triangles(tris.clone());
    let max_t = Fix128::from_int(10);
    let two = Fix128::from_int(2);
    for (o, d) in [
        // along +x, 2 above the mesh in y; along +y, 2 beside it in z; along -z, 2 beside it in x
        (
            Vec3Fix::new(-two, two, Fix128::ZERO),
            Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        ),
        (
            Vec3Fix::new(Fix128::ZERO, -two, two),
            Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        ),
        (
            Vec3Fix::new(two, Fix128::ZERO, two),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, -Fix128::ONE),
        ),
    ] {
        let ray = Ray::new(o, d);
        assert!(brute(&tris, &ray, max_t).is_none());
        let (hit, visits) = mesh.raycast_visits(&ray, max_t);
        assert!(hit.is_none());
        assert_eq!(visits, 1, "{o:?} along {d:?}: only the root is tested");
    }
}
