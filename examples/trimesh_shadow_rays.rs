//! Shadow rays against a triangle mesh: which points of a floor are hidden from a light by a
//! square roof above them, and how much of the mesh's BVH each ray tests.
//!
//! The roof is the square [0, 1] x [0, 1] at height 1 over a floor at height 0, tiled in small
//! triangles so the mesh has a real BVH; the light is toward (-1, 2, 0), so a floor point (x, 0, z)
//! is in shadow when 0 < x - 1/2 < 1 and 0 < z < 1: 64 of the 384 floor points below. Each of
//! those rays meets the roof exactly on the diagonal two triangles of a cell share, and the
//! closed triangle test counts that as a hit.
//!
//! Run: `cargo run --example trimesh_shadow_rays`

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::raycast::Ray;
use alice_physics_core::trimesh::{TriMesh, Triangle};

fn v(x: i64, y: i64, z: i64, d: i64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_ratio(x, d),
        Fix128::from_ratio(y, d),
        Fix128::from_ratio(z, d),
    )
}

fn main() {
    // the roof, 32 x 32 cells of 1/32, two triangles each
    let mut tris = Vec::new();
    for i in 0..32 {
        for j in 0..32 {
            let p = |a: i64, b: i64| v(a, 32, b, 32);
            tris.push(Triangle::new(p(i, j), p(i + 1, j), p(i + 1, j + 1)));
            tris.push(Triangle::new(p(i, j), p(i + 1, j + 1), p(i, j + 1)));
        }
    }
    let mesh = TriMesh::from_triangles(tris);
    let toward_light = v(-1, 2, 0, 1);
    let max_t = Fix128::from_int(10);
    let mut shadowed = 0;
    let mut visits = 0;
    let mut points = 0;
    // floor points on a grid of 1/8 from an offset of 1/16 + 1/64 (none on the shadow's border)
    for i in 0..24 {
        for j in 0..16 {
            let ray = Ray::new(v(-64 + 5 + 8 * i, 0, -64 + 5 + 8 * j, 64), toward_light);
            let (_, n) = mesh.raycast_visits(&ray, max_t);
            visits += n;
            points += 1;
            if mesh.occluded(&ray, max_t) {
                shadowed += 1;
            }
        }
    }
    println!(
        "{points} floor points, {shadowed} in shadow; {} triangles, {:.1} BVH nodes tested per ray",
        mesh.triangle_count(),
        visits as f64 / points as f64
    );
}
