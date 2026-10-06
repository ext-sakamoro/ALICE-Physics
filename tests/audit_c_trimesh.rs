//! Audit oracles for `trimesh`: values of `ray_triangle`, `TriMesh::raycast`,
//! `collide_sphere`, `collide_capsule`, `collide_aabb`, `closest_point` and the
//! bounding boxes against closed-form geometry.
//!
//! Expected values: ray-plane parameter `t = (d - n.o) / (n.dir)` with the hit
//! inside the triangle; sphere / capsule depth `r - |c - p|` with `p` the closest
//! point on the plane, edge or vertex; box depth = plane distance of the lowest
//! face; bounds = componentwise min / max over the vertices (brute force).

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::Ray;
use alice_physics::trimesh::{ray_triangle, TriMesh, Triangle};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (a[k] - b[k]).abs() <= tol,
            "{what}: axis {k} got {a:?} want {b:?}"
        );
    }
}

/// Ground square `[-10, 10]^2` at `y = 0` (two triangles, winding gives `-y`).
fn ground() -> TriMesh {
    let vertices = [
        Vec3Fix::from_int(-10, 0, -10),
        Vec3Fix::from_int(10, 0, -10),
        Vec3Fix::from_int(10, 0, 10),
        Vec3Fix::from_int(-10, 0, 10),
    ];
    TriMesh::from_indexed(&vertices, &[0, 1, 2, 0, 2, 3])
}

fn oblique() -> Triangle {
    // plane x + y + z = 2
    Triangle::new(v3(2.0, 0.0, 0.0), v3(0.0, 2.0, 0.0), v3(0.0, 0.0, 2.0))
}

// ---------------------------------------------------------------------------
// ray_triangle (C-S4W2-013)
// ---------------------------------------------------------------------------

/// Dyadic cases: `t`, hit point and normal exactly (the normal of an axis plane is
/// exact; the oblique unit normal `(1,1,1)/sqrt 3` to 1e-15).
#[test]
fn ray_triangle_axis_and_oblique_hits_are_exact() {
    let flat = Triangle::new(v3(0.0, 0.0, 0.0), v3(4.0, 0.0, 0.0), v3(0.0, 4.0, 0.0));
    let down = Ray::new(v3(1.0, 1.0, 3.0), v3(0.0, 0.0, -1.0));
    let h = ray_triangle(&down, &flat, Fix128::from_int(100)).expect("hit");
    assert_eq!(h.t, Fix128::from_int(3));
    assert_eq!(h.point, v3(1.0, 1.0, 0.0));
    assert_eq!(h.normal, Vec3Fix::UNIT_Z);

    let s3 = 1.0 / 3.0_f64.sqrt();
    // front face: t = 5 - 1.5
    let front = Ray::new(v3(0.25, 0.25, 5.0), v3(0.0, 0.0, -1.0));
    let h = ray_triangle(&front, &oblique(), Fix128::from_int(100)).expect("hit");
    assert_eq!(h.t, fx(3.5));
    assert_eq!(h.point, v3(0.25, 0.25, 1.5));
    close(arr(h.normal), [s3, s3, s3], 1e-15, "front normal");

    // back face: from z = -5 upward, t = 6.5, normal turned toward the ray
    let back = Ray::new(v3(0.25, 0.25, -5.0), v3(0.0, 0.0, 1.0));
    let h = ray_triangle(&back, &oblique(), Fix128::from_int(100)).expect("hit");
    assert_eq!(h.t, fx(6.5));
    assert_eq!(h.point, v3(0.25, 0.25, 1.5));
    close(arr(h.normal), [-s3, -s3, -s3], 1e-15, "back normal");
}

/// Diagonal ray from the origin along `(1,1,1)`: `t = 2 / sqrt 3`, hit `(2/3)(1,1,1)`,
/// relative error below 1e-9 (the inline test accepted 20%).
#[test]
fn ray_triangle_diagonal_hit_is_within_1e_minus_9_relative() {
    let ray = Ray::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0));
    let h = ray_triangle(&ray, &oblique(), Fix128::from_int(100)).expect("hit");
    let want_t = 2.0 / 3.0_f64.sqrt();
    let t = h.t.to_f64();
    assert!(((t - want_t) / want_t).abs() < 1e-9, "t {t} want {want_t}");
    close(arr(h.point), [2.0 / 3.0; 3], 1e-9, "point");
    // a tilted ray off the centroid: o = (0.5, -1, 0.25), dir (0.25, 1, 0.5)
    // n.o = -0.25, n.dir = 1.75 per unnormalised dir: s = (2 + 0.25) / 1.75
    let dir: [f64; 3] = [0.25, 1.0, 0.5];
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let ray = Ray::new(v3(0.5, -1.0, 0.25), v3(dir[0], dir[1], dir[2]));
    let h = ray_triangle(&ray, &oblique(), Fix128::from_int(100)).expect("hit");
    let s = 2.25 / 1.75;
    let want_t = s * len;
    let want_p = [0.5 + s * dir[0], -1.0 + s * dir[1], 0.25 + s * dir[2]];
    let t = h.t.to_f64();
    assert!(((t - want_t) / want_t).abs() < 1e-9, "t {t} want {want_t}");
    close(arr(h.point), want_p, 1e-9, "tilted point");
}

/// `max_t` is inclusive: a hit at exactly `t = 3.5` is reported with `max_t = 3.5`
/// and dropped with `max_t = 3.5 - 2^-40`; outside the triangle (`u + v > 1`) and a
/// ray pointing away (t < 0) miss.
#[test]
fn ray_triangle_max_t_bound_and_misses() {
    let front = Ray::new(v3(0.25, 0.25, 5.0), v3(0.0, 0.0, -1.0));
    assert!(ray_triangle(&front, &oblique(), fx(3.5)).is_some());
    assert!(ray_triangle(&front, &oblique(), fx(3.5 - 1.0 / (1u64 << 40) as f64)).is_none());
    let outside = Ray::new(v3(1.5, 1.5, 5.0), v3(0.0, 0.0, -1.0));
    assert!(ray_triangle(&outside, &oblique(), Fix128::from_int(100)).is_none());
    // u = v = 0.75 (both positive, sum above 1): plane point (-1, 1.5, 1.5)
    let beyond_edge = Ray::new(v3(-1.0, 1.5, 6.5), v3(0.0, 0.0, -1.0));
    assert!(ray_triangle(&beyond_edge, &oblique(), Fix128::from_int(100)).is_none());
    let away = Ray::new(v3(0.25, 0.25, 5.0), v3(0.0, 0.0, 1.0));
    assert!(ray_triangle(&away, &oblique(), Fix128::from_int(100)).is_none());
}

// ---------------------------------------------------------------------------
// TriMesh queries (C-S4W2-012)
// ---------------------------------------------------------------------------

/// Ground plus a raised triangle at `y = 2` over `x, z in [0, 4]`: the downward ray
/// above the raised triangle hits it first (`t = 8`, index 2), elsewhere the ground
/// triangle that contains the point (`t = 10`), normal `+y` toward the ray.
#[test]
fn mesh_raycast_returns_the_nearest_triangle_with_exact_t_point_and_index() {
    let mut tris = ground().triangles.clone();
    tris.push(Triangle::new(
        v3(0.0, 2.0, 0.0),
        v3(0.0, 2.0, 4.0),
        v3(4.0, 2.0, 0.0),
    ));
    let mesh = TriMesh::from_triangles(tris);
    let down = |x: f64, z: f64| Ray::new(v3(x, 10.0, z), v3(0.0, -1.0, 0.0));
    for (x, z, t, idx) in [
        (1.0, 1.0, 8.0, 2usize),
        (5.0, -5.0, 10.0, 0),
        (-5.0, 5.0, 10.0, 1),
    ] {
        let h = mesh
            .raycast(&down(x, z), Fix128::from_int(100))
            .expect("hit");
        // Moller-Trumbore divides by det: a few hundred 2^-64 ulp
        assert!((h.t.to_f64() - t).abs() < 1e-15, "t at ({x}, {z})");
        close(arr(h.point), [x, 10.0 - t, z], 1e-15, "point");
        close(arr(h.normal), [0.0, 1.0, 0.0], 1e-15, "normal");
        assert_eq!(h.body_index, idx, "index at ({x}, {z})");
    }
    // from below: the ground is nearer (t = 5, tri 1, normal -y), the raised
    // triangle behind it (t = 7) must not replace it
    let up = Ray::new(v3(1.0, -5.0, 2.0), v3(0.0, 1.0, 0.0));
    let h = mesh.raycast(&up, Fix128::from_int(100)).expect("hit");
    assert!((h.t.to_f64() - 5.0).abs() < 1e-15, "t from below {:?}", h.t);
    assert_eq!(h.body_index, 1);
    close(arr(h.normal), [0.0, -1.0, 0.0], 1e-15, "normal from below");
    assert!(mesh.raycast(&down(1.0, 1.0), Fix128::from_int(7)).is_none());
    assert!(mesh
        .raycast(&down(20.0, 0.0), Fix128::from_int(100))
        .is_none());
}

/// Sphere against the ground: face region (above and below the plane), edge region
/// and vertex region, all from the closest-point distance.
#[test]
fn collide_sphere_matches_face_edge_and_vertex_closed_forms() {
    let mesh = ground();
    // face, above: depth 0.5, normal +y
    let c = mesh
        .collide_sphere(v3(3.0, 0.5, -2.0), Fix128::ONE)
        .expect("hit");
    assert!((c.depth.to_f64() - 0.5).abs() < 1e-15, "face depth");
    close(arr(c.normal), [0.0, 1.0, 0.0], 1e-15, "face normal");
    close(arr(c.point_a), [3.0, -0.5, -2.0], 1e-15, "face point_a");
    close(arr(c.point_b), [3.0, 0.0, -2.0], 1e-15, "face point_b");
    // face, below: normal -y (from the mesh toward the centre)
    let c = mesh
        .collide_sphere(v3(3.0, -0.5, -2.0), Fix128::ONE)
        .expect("hit");
    assert!((c.depth.to_f64() - 0.5).abs() < 1e-15, "below depth");
    close(arr(c.normal), [0.0, -1.0, 0.0], 1e-15, "below normal");
    close(arr(c.point_b), [3.0, 0.0, -2.0], 1e-15, "below point_b");
    // edge x = 10: closest (10, 0, 0), delta (1, 0.5, 0)
    let c = mesh
        .collide_sphere(v3(11.0, 0.5, 0.0), fx(2.0))
        .expect("hit");
    let d = 1.25_f64.sqrt();
    assert!((c.depth.to_f64() - (2.0 - d)).abs() < 1e-15, "edge depth");
    close(arr(c.normal), [1.0 / d, 0.5 / d, 0.0], 1e-15, "edge normal");
    close(arr(c.point_b), [10.0, 0.0, 0.0], 1e-15, "edge point_b");
    close(
        arr(c.point_a),
        [11.0 - 2.0 / d, 0.5 - 1.0 / d, 0.0],
        1e-15,
        "edge point_a",
    );
    // vertex (10, 0, 10): delta (1, 1, 2), |delta| = sqrt 6
    let c = mesh
        .collide_sphere(v3(11.0, 1.0, 12.0), fx(3.0))
        .expect("hit");
    let d = 6.0_f64.sqrt();
    assert!((c.depth.to_f64() - (3.0 - d)).abs() < 1e-15, "vertex depth");
    close(
        arr(c.normal),
        [1.0 / d, 1.0 / d, 2.0 / d],
        1e-15,
        "vertex normal",
    );
    close(arr(c.point_b), [10.0, 0.0, 10.0], 1e-15, "vertex point_b");
    // just out of reach: |delta| = sqrt 6 > 2.4
    assert!(mesh.collide_sphere(v3(11.0, 1.0, 12.0), fx(2.4)).is_none());
}

/// Capsule above the ground: horizontal segment at height 0.25 and a tilted
/// segment whose lower end is at 0.25: depth `r - 0.25`, normal +y, `point_b` on
/// the plane below a nearest segment point, `point_a = point_b - n (r - depth)`.
#[test]
fn collide_capsule_depth_normal_and_points_match_the_closed_form() {
    let mesh = ground();
    let r = fx(0.5);
    let c = mesh
        .collide_capsule(v3(-1.0, 0.25, 2.0), v3(1.0, 0.25, 2.0), r)
        .expect("hit");
    assert!((c.depth.to_f64() - 0.25).abs() < 1e-15, "depth");
    close(arr(c.normal), [0.0, 1.0, 0.0], 1e-15, "normal");
    let (pa, pb) = (arr(c.point_a), arr(c.point_b));
    assert!(pb[0].abs() <= 1.0, "point_b under the segment {pb:?}");
    close(pb, [pb[0], 0.0, 2.0], 1e-15, "point_b on the plane");
    close(pa, [pb[0], -0.25, 2.0], 1e-15, "point_a");

    let c = mesh
        .collide_capsule(v3(3.0, 0.25, -1.0), v3(4.0, 2.0, -1.0), r)
        .expect("hit");
    assert!((c.depth.to_f64() - 0.25).abs() < 1e-12, "tilted depth");
    close(arr(c.normal), [0.0, 1.0, 0.0], 1e-12, "tilted normal");
    close(arr(c.point_b), [3.0, 0.0, -1.0], 1e-12, "tilted point_b");
    close(arr(c.point_a), [3.0, -0.25, -1.0], 1e-12, "tilted point_a");
    // lifted clear of the plane
    assert!(mesh
        .collide_capsule(v3(-1.0, 0.75, 2.0), v3(1.0, 0.75, 2.0), r)
        .is_none());
}

/// Box resting into the ground: depth = how far the bottom face is below `y = 0`,
/// normal +y (from the mesh to the box), `point_b` a point of the plane inside the
/// footprint of the box.
#[test]
fn collide_aabb_depth_and_normal_follow_the_bottom_face() {
    let mesh = ground();
    for (c, h, depth) in [
        ([2.0, 0.5, 3.0], [1.0, 1.0, 1.0], 0.5),
        ([0.0, 1.0, 0.0], [0.5, 2.0, 0.25], 1.0),
        ([-4.0, 0.125, 6.0], [2.0, 0.25, 1.0], 0.125),
    ] {
        let aabb = AABB::new(
            v3(c[0] - h[0], c[1] - h[1], c[2] - h[2]),
            v3(c[0] + h[0], c[1] + h[1], c[2] + h[2]),
        );
        let got = mesh.collide_aabb(&aabb).expect("hit");
        assert!(
            (got.depth.to_f64() - depth).abs() < 1e-15,
            "depth for centre {c:?}"
        );
        close(arr(got.normal), [0.0, 1.0, 0.0], 1e-15, "normal");
        // point_b is a mesh point on the bottom face footprint
        let pb = arr(got.point_b);
        assert!(pb[1].abs() < 1e-15, "point_b on the plane {pb:?}");
        assert!(
            (pb[0] - c[0]).abs() <= h[0] && (pb[2] - c[2]).abs() <= h[2],
            "point_b {pb:?} centre {c:?} normal {:?} depth {}",
            arr(got.normal),
            got.depth.to_f64()
        );
    }
    let above = AABB::new(v3(-1.0, 0.5, -1.0), v3(1.0, 2.0, 1.0));
    assert!(mesh.collide_aabb(&above).is_none());
}

/// Closest point: projection inside the square, clamp to the edge / vertex
/// outside; the index is the triangle that contains it.
#[test]
fn mesh_closest_point_projects_and_clamps() {
    let mesh = ground();
    let (p, idx) = mesh.closest_point(v3(3.0, 5.0, -4.0));
    close(arr(p), [3.0, 0.0, -4.0], 1e-15, "inside tri 0");
    assert_eq!(idx, 0);
    let (p, idx) = mesh.closest_point(v3(-3.0, -2.0, 4.0));
    close(arr(p), [-3.0, 0.0, 4.0], 1e-15, "inside tri 1");
    assert_eq!(idx, 1);
    let (p, _) = mesh.closest_point(v3(15.0, 2.0, 0.0));
    close(arr(p), [10.0, 0.0, 0.0], 1e-15, "edge clamp");
    let (p, _) = mesh.closest_point(v3(-13.0, 1.0, -12.0));
    close(arr(p), [-10.0, 0.0, -10.0], 1e-15, "vertex clamp");
}

/// `Triangle::aabb` and `TriMesh::bounds` equal the componentwise min / max of the
/// vertices (brute force), for a mesh with vertices spread on all axes.
#[test]
fn triangle_aabb_and_mesh_bounds_are_the_vertex_min_max() {
    let pts = [
        [-3.0, 1.5, 2.0],
        [4.25, -2.0, 0.5],
        [0.0, 6.0, -7.5],
        [1.0, -0.25, 9.0],
        [-8.0, 3.0, -1.0],
    ];
    let vertices: Vec<Vec3Fix> = pts.iter().map(|p| v3(p[0], p[1], p[2])).collect();
    let indices = [0u32, 1, 2, 1, 3, 2, 4, 0, 3];
    let mesh = TriMesh::from_indexed(&vertices, &indices);
    assert_eq!(mesh.triangle_count(), 3);
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for tri in indices.chunks(3) {
        let mut tlo = [f64::INFINITY; 3];
        let mut thi = [f64::NEG_INFINITY; 3];
        for &i in tri {
            for k in 0..3 {
                tlo[k] = tlo[k].min(pts[i as usize][k]);
                thi[k] = thi[k].max(pts[i as usize][k]);
                lo[k] = lo[k].min(pts[i as usize][k]);
                hi[k] = hi[k].max(pts[i as usize][k]);
            }
        }
        let t = Triangle::new(
            vertices[tri[0] as usize],
            vertices[tri[1] as usize],
            vertices[tri[2] as usize],
        );
        let b = t.aabb();
        assert_eq!(arr(b.min), tlo);
        assert_eq!(arr(b.max), thi);
    }
    assert_eq!(arr(mesh.bounds.min), lo);
    assert_eq!(arr(mesh.bounds.max), hi);
}
