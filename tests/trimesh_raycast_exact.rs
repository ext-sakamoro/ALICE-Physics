//! `ray_triangle` and `TriMesh::raycast` / `occluded` against an exact reference.
//!
//! oracle (independent of the crate's predicate and arithmetic): `support/exact_ray.rs` solves
//! the ray / triangle system by Cramer's rule in `num-bigint` on the raw `Fix128` integers. A ray
//! hits a triangle exactly when the crossing lies in the closed triangle at t in [0, max_t], and
//! the hit's t is the exact t floored to the `Fix128` grid. `ray_triangle` must give that answer
//! and that t for every ray and triangle, the hit's point is `ray.at(t)` and its normal the
//! triangle's unit normal turned to face the ray; the mesh query's nearest hit (smallest index on
//! a tie) and `occluded` must equal the exact nearest over every triangle.
//!
//! The grazing set: a triangle with an edge on the grid plane y = 0, in the tilted plane
//! z = y / 3 (non-dyadic), and rays nearly in that plane crossing it just outside or inside the
//! edge (offset δ = {1, 3, 5, 7, 11} · 2^-20 … 2^-60, tilt ε = 2^-16 … 2^-46). The rounded
//! barycentric test accepted crossings outside the edge by more than the node bounds' padding, so
//! the walk and the loop over triangles disagreed on 522 of these rays.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::raycast::Ray;
use alice_physics_core::trimesh::{ray_triangle, TriMesh, Triangle};

#[path = "support/exact_ray.rs"]
mod exact_ray;
use exact_ray::{exact_nearest, exact_t};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn int(&mut self, range: i64) -> i64 {
        (self.next() % (2 * range as u64 + 1)) as i64 - range
    }

    fn fix(&mut self, range: i64, denom: i64) -> Fix128 {
        Fix128::from_ratio(self.int(range), denom)
    }

    fn vec(&mut self, range: i64, denom: i64) -> Vec3Fix {
        Vec3Fix::new(self.fix(range, denom), self.fix(range, denom), self.fix(range, denom))
    }
}

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

/// `ray_triangle` equals the reference on this ray and triangle: same miss, or same t with the
/// point at t and the normal facing the ray. Returns whether it hit.
fn check_one(ray: &Ray, tri: &Triangle, max_t: Fix128, tag: &str) -> bool {
    let want = exact_t(ray, tri, max_t);
    let got = ray_triangle(ray, tri, max_t);
    assert_eq!(got.map(|h| h.t), want, "{tag}: ray {ray:?} triangle {tri:?} reach {max_t:?}");
    if let Some(h) = got {
        assert_eq!(h.point, ray.at(h.t), "{tag}: point");
        let n = tri.unit_normal();
        let n = if n.dot(ray.direction) > Fix128::ZERO { -n } else { n };
        assert_eq!(h.normal, n, "{tag}: normal");
    }
    want.is_some()
}

/// The mesh query and `occluded` equal the exact nearest over every triangle
fn check_mesh(mesh: &TriMesh, ray: &Ray, max_t: Fix128, tag: &str) -> bool {
    let want = exact_nearest(&mesh.triangles, ray, max_t);
    let got = mesh.raycast(ray, max_t);
    assert_eq!(got.map(|h| (h.t, h.body_index)), want, "{tag}: ray {ray:?} reach {max_t:?}");
    assert_eq!(mesh.occluded(ray, max_t), want.is_some(), "{tag}: occluded, ray {ray:?} reach {max_t:?}");
    want.is_some()
}

/// The grazing rays across one edge of the tilted triangle (tri 0 of the mesh)
fn grazing() -> (Vec<Triangle>, Vec<Ray>) {
    let (one, z, third, half) = (Fix128::ONE, Fix128::ZERO, Fix128::from_ratio(1, 3), Fix128::from_ratio(1, 2));
    let mut tris = vec![Triangle::new(v(z, z, z), v(one, z, z), v(z, one, third))];
    for k in 1..8i64 {
        let c = Fix128::from_int(10 * k);
        tris.push(Triangle::new(v(c, c, c), v(c + one, c, c), v(c, c + one, c)));
    }
    let mut rays = Vec::new();
    for eps_exp in 16..=46 {
        let eps = Fix128::from_ratio(1, 1i64 << eps_exp);
        for off_exp in 20..=60 {
            for m in [1i64, 3, 5, 7, 11] {
                let delta = Fix128::from_ratio(m, 1i64 << off_exp);
                for side in [-delta, delta] {
                    // crossing point on the plane, outside (y < 0) or inside (y > 0) the edge
                    let p = v(half, side, side * third);
                    let d = v(one, -eps * third, eps);
                    rays.push(Ray::new(p - d * Fix128::from_ratio(1, 4), d));
                }
            }
        }
    }
    (tris, rays)
}

#[test]
fn ray_triangle_is_the_exact_closed_triangle_test_on_grazing_rays() {
    let (tris, rays) = grazing();
    let max_t = Fix128::from_int(100);
    let (mut hits, mut misses) = (0, 0);
    for ray in &rays {
        if check_one(ray, &tris[0], max_t, "grazing") {
            hits += 1;
        } else {
            misses += 1;
        }
    }
    assert_eq!(hits + misses, 31 * 41 * 5 * 2);
    // both answers occur (the set straddles the edge)
    assert!(hits > 1000 && misses > 1000, "hits {hits}, misses {misses}");
}

#[test]
fn the_walk_and_occluded_equal_the_exact_nearest_on_grazing_rays() {
    let (tris, rays) = grazing();
    let mesh = TriMesh::from_triangles(tris);
    let mut hits = 0;
    for ray in &rays {
        for max_t in [Fix128::from_int(100), Fix128::from_ratio(1, 4), Fix128::from_ratio(1, 8)] {
            hits += usize::from(check_mesh(&mesh, ray, max_t, "grazing"));
        }
    }
    assert!(hits > 1000, "{hits} hits");
}

/// Random triangles and rays: aimed inside a triangle, at random, through a vertex, nearly
/// parallel to a triangle's plane (a determinant far below the old floor), and with a reach
/// ending at a hit
#[test]
fn ray_triangle_and_the_walk_equal_the_exact_reference_on_random_scenes() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (mut compared, mut hits) = (0, 0);
    for scene in 0..40 {
        let (range, denom, size) = if scene % 2 == 0 { (64, 64, 16) } else { (40, 1, 6) };
        let tris: Vec<Triangle> = (0..48)
            .map(|_| {
                let c = rng.vec(range, denom);
                Triangle::new(c + rng.vec(size, denom), c + rng.vec(size, denom), c + rng.vec(size, denom))
            })
            .collect();
        let mesh = TriMesh::from_triangles(tris.clone());
        let mut rays = Vec::new();
        for _ in 0..24 {
            let t = &tris[(rng.next() % tris.len() as u64) as usize];
            // a point inside the triangle by barycentric weights k/8
            let (a, b) = (rng.int(4).unsigned_abs() as i64, rng.int(4).unsigned_abs() as i64);
            let target = t.v0 + (t.v1 - t.v0) * Fix128::from_ratio(a, 8) + (t.v2 - t.v0) * Fix128::from_ratio(b, 8);
            let o = rng.vec(range * 2, denom);
            rays.push(Ray::new(o, target - o));
            rays.push(Ray::new(o, t.v1 - o));
            rays.push(Ray::new(rng.vec(range * 2, denom), rng.vec(8, 8)));
            // nearly in the triangle's plane: along an edge, tilted by 2^-50 of the normal
            let n = (t.v1 - t.v0).cross(t.v2 - t.v0);
            let tilt = Fix128::from_ratio(1, 1i64 << 50);
            rays.push(Ray::new(t.v0 - (t.v1 - t.v0) + n * tilt, (t.v1 - t.v0) - n * (tilt * Fix128::from_int(2))));
        }
        for ray in &rays {
            for max_t in [Fix128::from_int(1000), Fix128::from_int(3)] {
                for (i, tri) in tris.iter().enumerate() {
                    hits += usize::from(check_one(ray, tri, max_t, &format!("scene {scene} tri {i}")));
                    compared += 1;
                }
                check_mesh(&mesh, ray, max_t, &format!("scene {scene}"));
            }
            // the reach ends one step (2^-64) past the floored nearest t, so past the exact t:
            // the same hit; at the floored t itself, the hit is kept only when t is on the grid
            if let Some((t, i)) = exact_nearest(&tris, ray, Fix128::from_int(1000)) {
                let step = Fix128::from_raw(0, 1);
                assert!(check_mesh(&mesh, ray, t + step, &format!("scene {scene} reach = t + step")));
                assert_eq!(mesh.raycast(ray, t + step).map(|h| (h.t, h.body_index)), Some((t, i)));
                check_mesh(&mesh, ray, t, &format!("scene {scene} reach = t"));
            }
        }
    }
    assert_eq!(compared, 40 * 24 * 4 * 2 * 48);
    assert!(hits > 1000, "{hits} hits");
}

/// A ray parallel to the triangle's plane does not hit it, in the plane or off it, and neither
/// does one tilted toward it by a sine below 2^-48; a sine above it hits
#[test]
fn parallel_rays_do_not_hit() {
    let tri = Triangle::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(4, 0, 0), Vec3Fix::from_int(0, 0, 4));
    for y in [0, 1, -1] {
        let ray = Ray::new(Vec3Fix::from_int(-1, y, 1), Vec3Fix::from_int(1, 0, 0));
        assert_eq!(exact_t(&ray, &tri, Fix128::from_int(100)), None);
        assert!(ray_triangle(&ray, &tri, Fix128::from_int(100)).is_none(), "y {y}");
    }
    // from y = 2^-k above the plane, aimed down by a slope of 2^-(k + 2) (sine ≈ slope)
    // (sine = slope / sqrt(1 + slope²), just below the slope: 2^-48 itself is below the threshold)
    for (k, hit) in [(44, true), (45, true), (47, false), (60, false)] {
        let y = Fix128::from_ratio(1, 1i64 << k);
        let ray = Ray {
            origin: Vec3Fix::new(-Fix128::ONE, y, Fix128::ONE),
            direction: Vec3Fix::new(Fix128::ONE, -(y / Fix128::from_int(4)), Fix128::ZERO),
        };
        let max_t = Fix128::from_int(100);
        assert_eq!(exact_t(&ray, &tri, max_t).is_some(), hit, "reference, slope 2^-{}", k + 2);
        assert_eq!(ray_triangle(&ray, &tri, max_t).map(|h| h.t), exact_t(&ray, &tri, max_t), "slope 2^-{}", k + 2);
    }
}

/// The surface of the cube [0, 1]³, each face 4 × 4 cells of two triangles (closed: every edge in
/// two triangles)
fn closed_cube() -> Vec<Triangle> {
    let mut tris = Vec::new();
    let g = |a: i64| Fix128::from_ratio(a, 4);
    // a face: the axis it is normal to, and its side (0 or 4 in quarters)
    for axis in 0..3 {
        for side in [0i64, 4] {
            let at = |a: i64, b: i64| {
                let mut p = [Fix128::ZERO; 3];
                p[axis] = g(side);
                p[(axis + 1) % 3] = g(a);
                p[(axis + 2) % 3] = g(b);
                Vec3Fix::new(p[0], p[1], p[2])
            };
            for i in 0..4 {
                for j in 0..4 {
                    tris.extend([
                        Triangle::new(at(i, j), at(i + 1, j), at(i + 1, j + 1)),
                        Triangle::new(at(i, j), at(i + 1, j + 1), at(i, j + 1)),
                    ]);
                }
            }
        }
    }
    tris
}

/// Watertight: a ray from outside a closed mesh that passes exactly through a point of its surface
/// (direction = target − origin, unnormalised, so the point is at t = 1) hits it, whether
/// the point is inside a triangle, on an edge two triangles share (the cells' sides and
/// diagonals) or at a vertex several share, including the cube's own edges and corners. Until the
/// exact test, a crossing exactly on a shared edge could round outside both triangles and the ray
/// passed through the closed mesh.
#[test]
fn rays_through_shared_edges_and_vertices_of_a_closed_mesh_hit_it() {
    let tris = closed_cube();
    let mesh = TriMesh::from_triangles(tris.clone());
    let q = |a: i64| Fix128::from_ratio(a, 8);
    let max_t = Fix128::from_int(100);
    let mut rays = Vec::new();
    // targets on the face z = 0 at every eighth (vertices, edge midpoints, diagonal midpoints,
    // cell interiors), from origins below the cube in several directions
    for a in 0..=8 {
        for b in 0..=8 {
            let target = Vec3Fix::new(q(a), q(b), Fix128::ZERO);
            for o in [
                Vec3Fix::new(Fix128::from_ratio(1, 3), Fix128::from_ratio(2, 7), -Fix128::ONE),
                Vec3Fix::new(Fix128::from_int(-2), Fix128::from_ratio(5, 3), Fix128::from_ratio(-1, 9)),
                Vec3Fix::new(Fix128::from_ratio(1, 2), Fix128::from_ratio(1, 2), Fix128::from_int(-3)),
            ] {
                // the direction target − o itself (not normalised): o + 1 · d is the target exactly
                rays.extend([Ray { origin: o, direction: target - o }]);
            }
        }
    }
    let mut on_edges = 0;
    for ray in &rays {
        let want = exact_nearest(&tris, ray, max_t);
        assert!(want.is_some(), "the reference misses the closed mesh: ray {ray:?}");
        assert!(check_mesh(&mesh, ray, max_t, "closed cube"));
        // how many triangles the exact crossing touches (more than one: an edge or a vertex)
        let (t, _) = want.unwrap_or((Fix128::ZERO, 0));
        on_edges += usize::from(tris.iter().filter(|tri| exact_t(ray, tri, max_t) == Some(t)).count() > 1);
    }
    assert_eq!(rays.len(), 81 * 3);
    assert!(on_edges > 100, "only {on_edges} rays through a shared edge or vertex");

    // the shadow rays of examples/trimesh_shadow_rays.rs: the roof [0, 1]² at y = 1 seen from the
    // floor toward (−1, 2, 0) shades the points with x in [1/2, 3/2], z in [0, 1], 8 × 8 of the
    // grid, and every one of those rays crosses the roof on a cell's shared diagonal
    let v = |x: i64, y: i64, z: i64, d: i64| Vec3Fix::new(Fix128::from_ratio(x, d), Fix128::from_ratio(y, d), Fix128::from_ratio(z, d));
    let mut roof = Vec::new();
    for i in 0..32 {
        for j in 0..32 {
            let p = |a: i64, b: i64| v(a, 32, b, 32);
            roof.extend([
                Triangle::new(p(i, j), p(i + 1, j), p(i + 1, j + 1)),
                Triangle::new(p(i, j), p(i + 1, j + 1), p(i, j + 1)),
            ]);
        }
    }
    let roof_mesh = TriMesh::from_triangles(roof);
    let mut shaded = 0;
    for i in 0..24 {
        for j in 0..16 {
            let ray = Ray::new(v(-64 + 5 + 8 * i, 0, -64 + 5 + 8 * j, 64), v(-1, 2, 0, 1));
            let hit = check_mesh(&roof_mesh, &ray, Fix128::from_int(10), "roof");
            let x = Fix128::from_ratio(-64 + 5 + 8 * i, 64);
            let z = Fix128::from_ratio(-64 + 5 + 8 * j, 64);
            let inside = x >= Fix128::from_ratio(1, 2) && x <= Fix128::from_ratio(3, 2) && z >= Fix128::ZERO && z <= Fix128::ONE;
            assert_eq!(hit, inside, "floor point {i}, {j}");
            shaded += usize::from(hit);
        }
    }
    assert_eq!(shaded, 64);
}
