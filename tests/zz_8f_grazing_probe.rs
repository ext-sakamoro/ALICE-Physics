//! Throwaway verification probe (not committed): grazing rays just outside a triangle edge on the
//! node box face, walk vs brute force, plus random scenes and the occluded agreement.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::raycast::{Ray, RayHit};
use alice_physics_core::trimesh::{ray_triangle, TriMesh, Triangle};

fn brute(tris: &[Triangle], ray: &Ray, max_t: Fix128) -> Option<RayHit> {
    let mut best: Option<RayHit> = None;
    for (i, tri) in tris.iter().enumerate() {
        if let Some(mut h) = ray_triangle(ray, tri, max_t) {
            h.body_index = i;
            if best.is_none_or(|b| h.t < b.t) {
                best = Some(h);
            }
        }
    }
    best
}

fn key(h: Option<RayHit>) -> Option<(Fix128, usize, Vec3Fix)> {
    h.map(|h| (h.t, h.body_index, h.point))
}

fn v(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, y, z)
}

#[test]
fn grazing_rays_outside_an_edge_on_the_box_face() {
    let one = Fix128::ONE;
    let z = Fix128::ZERO;
    // triangle 0: the unit right triangle in z = 0, its edge y = 0 lies on its leaf box face;
    // triangles 1..: far away, so the root is internal and the unit triangle has its own leaf
    let mut tris = vec![Triangle::new(v(z, z, z), v(one, z, z), v(z, one, z))];
    for k in 1..8i64 {
        let c = Fix128::from_int(10 * k);
        tris.push(Triangle::new(v(c, c, c), v(c + one, c, c), v(c, c + one, c)));
    }
    let mesh = TriMesh::from_triangles(tris.clone());
    let max_t = Fix128::from_int(100);
    let (mut compared, mut brute_hits, mut walk_miss, mut occ_bad) = (0, 0, 0, 0);
    let mut first = None;
    for slope_exp in 20..=47 {
        let s = Fix128::from_ratio(1, 1i64 << slope_exp.min(62));
        for off_exp in 20..=62 {
            for m in [1i64, 3, 7] {
                // y just outside the edge (negative), the ray along +x descending through z = 0
                let y = -Fix128::from_ratio(m, 1i64 << off_exp);
                let o = v(Fix128::from_ratio(1, 4), y, s * Fix128::from_ratio(1, 4));
                let d = v(one, z, -s);
                let ray = Ray::new(o, d);
                let b = brute(&tris, &ray, max_t);
                let w = mesh.raycast(&ray, max_t);
                compared += 1;
                if b.is_some() {
                    brute_hits += 1;
                }
                if key(b) != key(w) {
                    walk_miss += 1;
                    if first.is_none() {
                        first = Some((slope_exp, off_exp, m, key(b).is_some(), key(w).is_some()));
                    }
                }
                if mesh.occluded(&ray, max_t) != b.is_some() {
                    occ_bad += 1;
                }
            }
        }
    }
    eprintln!(
        "PROBE grazing: compared {compared}, brute hits {brute_hits}, walk != brute {walk_miss}, occluded != brute {occ_bad}, first {first:?}"
    );
    assert!(compared > 0);
    assert_eq!((walk_miss, occ_bad), (0, 0));
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> i64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as i64
    }
    fn f(&mut self, r: i64, denom: i64) -> Fix128 {
        Fix128::from_ratio(self.next() % (2 * r + 1) - r, denom)
    }
}

/// Grazing rays outside an edge that lies on the leaf box face, in a tilted plane whose products
/// round (v2 has a third in z)
#[test]
fn grazing_rays_in_a_tilted_plane() {
    let one = Fix128::ONE;
    let z = Fix128::ZERO;
    let third = Fix128::from_ratio(1, 3);
    // edge v0v1 along x at y = z = 0 (the box face y = 0), v2 lifted by a third
    let mut tris = vec![Triangle::new(v(z, z, z), v(one, z, z), v(z, one, third))];
    for k in 1..8i64 {
        let c = Fix128::from_int(10 * k);
        tris.push(Triangle::new(v(c, c, c), v(c + one, c, c), v(c, c + one, c)));
    }
    let mesh = TriMesh::from_triangles(tris.clone());
    let max_t = Fix128::from_int(100);
    let half = Fix128::from_ratio(1, 2);
    let (mut compared, mut brute_hits, mut outside_pad_hits, mut bad, mut occ_bad) = (0, 0, 0, 0, 0);
    let mut first = None;
    for eps_exp in 16..=46 {
        let eps = Fix128::from_ratio(1, 1i64 << eps_exp);
        for off_exp in 20..=60 {
            for m in [1i64, 3, 5, 7, 11] {
                let delta = Fix128::from_ratio(m, 1i64 << off_exp);
                // plane: z = y / 3; crossing point outside the edge by delta, on the plane
                let p = v(half, -delta, -delta * third);
                // in-plane direction along x plus eps of the normal (0, -1/3, 1)
                let d = v(one, -eps * third, eps);
                let o = p - d * Fix128::from_ratio(1, 4);
                let ray = Ray::new(o, d);
                let b = brute(&tris, &ray, max_t);
                let w = mesh.raycast(&ray, max_t);
                compared += 1;
                if b.is_some() {
                    brute_hits += 1;
                    if off_exp < 32 {
                        outside_pad_hits += 1;
                    }
                }
                if key(b) != key(w) {
                    bad += 1;
                    if first.is_none() {
                        first = Some((eps_exp, off_exp, m, b.is_some(), w.is_some()));
                    }
                }
                if mesh.occluded(&ray, max_t) != b.is_some() {
                    occ_bad += 1;
                }
            }
        }
    }
    eprintln!("PROBE tilted: compared {compared}, brute hits {brute_hits} (outside the pad {outside_pad_hits}), walk != brute {bad}, occluded != brute {occ_bad}, first {first:?}");
    assert_eq!((bad, occ_bad), (0, 0));
}

/// Random scenes of my own: random triangles at several scales, random rays and rays aimed at
/// random points of random triangles, every max_t from a list; walk == brute bit for bit and
/// occluded == brute.is_some().
#[test]
fn random_scenes_match_the_brute_force_loop() {
    let mut g = Lcg(0x8f8f_1234_5678_9abc);
    let (mut compared, mut hits, mut bad, mut occ_bad, mut ties) = (0, 0, 0, 0, 0);
    for scene in 0..60 {
        let denom = [1i64, 16, 1024, 1 << 20][scene % 4];
        let range = [5i64, 40, 300][scene % 3];
        let n = 20 + (scene % 7) * 30;
        let mut tris = Vec::new();
        for _ in 0..n {
            let c = v(g.f(range, denom), g.f(range, denom), g.f(range, denom));
            tris.push(Triangle::new(c + v(g.f(3, denom), g.f(3, denom), g.f(3, denom)), c + v(g.f(3, denom), g.f(3, denom), g.f(3, denom)), c + v(g.f(3, denom), g.f(3, denom), g.f(3, denom))));
        }
        // duplicates for exact ties
        for k in 0..n / 10 {
            tris.push(tris[k * 3 % n]);
        }
        let mesh = TriMesh::from_triangles(tris.clone());
        for r in 0..120 {
            let o = v(g.f(4 * range, denom), g.f(4 * range, denom), g.f(4 * range, denom));
            let d = if r % 2 == 0 {
                let t = tris[(g.next() as usize) % tris.len()];
                let a = Fix128::from_ratio(g.next() % 1000, 3000);
                let b = Fix128::from_ratio(g.next() % 1000, 3000);
                t.v0 + (t.v1 - t.v0) * a + (t.v2 - t.v0) * b - o
            } else {
                v(g.f(1000, denom), g.f(1000, denom), g.f(1000, denom))
            };
            let ray = Ray::new(o, d);
            for max_t in [1i64, 7, 50, 400, 100_000] {
                let max_t = Fix128::from_int(max_t);
                let b = brute(&tris, &ray, max_t);
                let w = mesh.raycast(&ray, max_t);
                compared += 1;
                if let Some(bh) = b {
                    hits += 1;
                    let n_at = tris
                        .iter()
                        .filter(|t| ray_triangle(&ray, t, max_t).is_some_and(|h| h.t == bh.t))
                        .count();
                    if n_at > 1 {
                        ties += 1;
                    }
                }
                if key(b) != key(w) || b.map(|h| h.normal) != w.map(|h| h.normal) {
                    bad += 1;
                }
                if mesh.occluded(&ray, max_t) != b.is_some() {
                    occ_bad += 1;
                }
            }
        }
    }
    eprintln!("PROBE random: compared {compared}, hits {hits}, ties {ties}, walk != brute {bad}, occluded != brute {occ_bad}");
    assert!(compared > 0 && hits > 0 && ties > 0);
    assert_eq!((bad, occ_bad), (0, 0));
}
