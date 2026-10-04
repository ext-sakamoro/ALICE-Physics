//! Audit oracles for trimesh
//!
//! 期待値は (1) 全三角形の総当たり (BVH 経由と同値)、(2) f64 の独立実装
//! (平面交点 / 重心座標の格子探索 / SAT) から導く
//! 座標は 1/16 刻みの dyadic なので Fix128 と f64 で入力が厳密に一致する

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::Ray;
use alice_physics::trimesh::{ray_triangle, TriMesh, Triangle};

// ---------------------------------------------------------------- 補助

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// [lo, hi] を 1/16 刻みで
    fn grid(&mut self, lo: i64, hi: i64) -> i64 {
        lo * 16 + (self.next() % ((hi - lo) as u64 * 16 + 1)) as i64
    }
}

fn fx(n16: i64) -> Fix128 {
    Fix128::from_ratio(n16, 16)
}
fn v3(a: i64, b: i64, c: i64) -> Vec3Fix {
    Vec3Fix::new(fx(a), fx(b), fx(c))
}
fn f(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn rand_tri(r: &mut Rng, span: i64) -> Triangle {
    loop {
        let t = Triangle::new(
            v3(
                r.grid(-span, span),
                r.grid(-span, span),
                r.grid(-span, span),
            ),
            v3(
                r.grid(-span, span),
                r.grid(-span, span),
                r.grid(-span, span),
            ),
            v3(
                r.grid(-span, span),
                r.grid(-span, span),
                r.grid(-span, span),
            ),
        );
        let n = cross(sub(f(t.v1), f(t.v0)), sub(f(t.v2), f(t.v0)));
        if norm(n) > 4.0 {
            return t;
        }
    }
}

fn rand_mesh(r: &mut Rng, n: usize, span: i64) -> TriMesh {
    TriMesh::from_triangles((0..n).map(|_| rand_tri(r, span)).collect())
}

fn rand_dir(r: &mut Rng) -> Vec3Fix {
    loop {
        let d = v3(r.grid(-4, 4), r.grid(-4, 4), r.grid(-4, 4));
        if norm(f(d)) > 0.5 {
            return d;
        }
    }
}

/// 三角形上の点 (重心座標 s, t) を格子で総当たりした距離の下限側 (f64)
fn grid_dist(t: &Triangle, p: [f64; 3], n: usize) -> f64 {
    let a = f(t.v0);
    let ab = sub(f(t.v1), a);
    let ac = sub(f(t.v2), a);
    let mut best = f64::MAX;
    for i in 0..=n {
        for j in 0..=(n - i) {
            let (s, u) = (i as f64 / n as f64, j as f64 / n as f64);
            let q = [
                a[0] + ab[0] * s + ac[0] * u,
                a[1] + ab[1] * s + ac[1] * u,
                a[2] + ab[2] * s + ac[2] * u,
            ];
            let d = norm(sub(q, p));
            if d < best {
                best = d;
            }
        }
    }
    best
}

// ---------------------------------------------------------------- Triangle

#[test]
fn triangle_aabb_normal_unit_normal_closed_form() {
    let mut r = Rng(0x1234_5678_9ABC_DEF1);
    for _ in 0..200 {
        let t = rand_tri(&mut r, 20);
        let (a, b, c) = (f(t.v0), f(t.v1), f(t.v2));
        let bb = t.aabb();
        for ax in 0..3 {
            let lo = a[ax].min(b[ax]).min(c[ax]);
            let hi = a[ax].max(b[ax]).max(c[ax]);
            assert_eq!(f(bb.min)[ax], lo, "min axis {ax}");
            assert_eq!(f(bb.max)[ax], hi, "max axis {ax}");
        }
        let n = cross(sub(b, a), sub(c, a));
        assert_eq!(f(t.normal()), n, "normal = e1 x e2");
        let un = f(t.unit_normal());
        assert!((norm(un) - 1.0).abs() < 1e-9);
        let nn = norm(n);
        for ax in 0..3 {
            assert!((un[ax] - n[ax] / nn).abs() < 1e-9, "unit normal axis {ax}");
        }
    }
}

#[test]
fn triangle_closest_point_is_feasible_and_optimal_in_every_voronoi_region() {
    let mut r = Rng(0xDEAD_BEEF_0000_0001);
    let mut regions = [0usize; 7];
    for _ in 0..120 {
        let t = rand_tri(&mut r, 8);
        let (a, b, c) = (f(t.v0), f(t.v1), f(t.v2));
        let ab = sub(b, a);
        let ac = sub(c, a);
        let nrm = cross(ab, ac);
        for _ in 0..8 {
            let p = v3(r.grid(-14, 14), r.grid(-14, 14), r.grid(-14, 14));
            let cp = t.closest_point(p);
            let q = f(cp);
            // 実行可能性: 三角形の平面上かつ重心座標が [0,1]
            let pq = sub(q, a);
            assert!(
                dot(pq, nrm).abs() / norm(nrm) < 1e-9,
                "off plane: {}",
                dot(pq, nrm) / norm(nrm)
            );
            let (d00, d01, d11) = (dot(ab, ab), dot(ab, ac), dot(ac, ac));
            let (d20, d21) = (dot(pq, ab), dot(pq, ac));
            let den = d00 * d11 - d01 * d01;
            let s = (d11 * d20 - d01 * d21) / den;
            let u = (d00 * d21 - d01 * d20) / den;
            assert!(
                s > -1e-9 && u > -1e-9 && s + u < 1.0 + 1e-9,
                "outside triangle: s={s} u={u}"
            );
            // 最適性: 格子の最小距離以下
            let d_impl = norm(sub(q, f(p)));
            let d_grid = grid_dist(&t, f(p), 120);
            assert!(
                d_impl <= d_grid + 1e-9,
                "not nearest: impl {d_impl} grid {d_grid} tri {t:?} p {p:?}"
            );
            // 領域の分類 (統計用)
            let on = |x: [f64; 3], y: [f64; 3]| norm(sub(x, y)) < 1e-9;
            if on(q, a) || on(q, b) || on(q, c) {
                regions[0] += 1;
            } else if s.abs() < 1e-9 || u.abs() < 1e-9 || (s + u - 1.0).abs() < 1e-9 {
                regions[1] += 1;
            } else {
                regions[2] += 1;
            }
        }
    }
    assert!(
        regions[0] > 20 && regions[1] > 20 && regions[2] > 20,
        "{regions:?}"
    );
}

#[test]
fn triangle_closest_point_returns_the_point_itself_on_the_triangle() {
    let t = Triangle::new(v3(0, 0, 0), v3(64, 0, 0), v3(0, 64, 0));
    for (x, y) in [(16, 16), (32, 16), (1, 1), (0, 32), (32, 0), (20, 20)] {
        let p = v3(x, y, 0);
        assert_eq!(f(t.closest_point(p)), f(p), "({x},{y})");
    }
    // 真上の点は射影
    assert_eq!(f(t.closest_point(v3(16, 16, 80))), f(v3(16, 16, 0)));
}

// ---------------------------------------------------------------- 構築

#[test]
fn from_indexed_builds_triangles_in_order_and_ignores_trailing_indices() {
    let verts = vec![v3(0, 0, 0), v3(16, 0, 0), v3(0, 16, 0), v3(16, 16, 32)];
    let m = TriMesh::from_indexed(&verts, &[0, 1, 2, 1, 3, 2, 0, 3]);
    assert_eq!(m.triangle_count(), 2);
    assert_eq!(m.triangles[0], Triangle::new(verts[0], verts[1], verts[2]));
    assert_eq!(m.triangles[1], Triangle::new(verts[1], verts[3], verts[2]));
    assert_eq!(f(m.bounds.min), [0.0, 0.0, 0.0]);
    assert_eq!(f(m.bounds.max), [1.0, 1.0, 2.0]);
    let none = TriMesh::from_indexed(&verts, &[0, 1]);
    assert_eq!(none.triangle_count(), 0);
}

#[test]
fn from_triangles_bounds_are_the_union_of_triangle_boxes() {
    let mut r = Rng(42);
    for n in [1usize, 2, 7, 40] {
        let m = rand_mesh(&mut r, n, 15);
        assert_eq!(m.triangle_count(), n);
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for t in &m.triangles {
            for v in [t.v0, t.v1, t.v2] {
                for ax in 0..3 {
                    lo[ax] = lo[ax].min(f(v)[ax]);
                    hi[ax] = hi[ax].max(f(v)[ax]);
                }
            }
        }
        assert_eq!(f(m.bounds.min), lo);
        assert_eq!(f(m.bounds.max), hi);
    }
}

#[test]
fn indexed_and_triangle_builders_answer_queries_identically() {
    let mut r = Rng(7);
    let tris: Vec<Triangle> = (0..30).map(|_| rand_tri(&mut r, 10)).collect();
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    for t in &tris {
        for v in [t.v0, t.v1, t.v2] {
            idx.push(verts.len() as u32);
            verts.push(v);
        }
    }
    let a = TriMesh::from_triangles(tris);
    let b = TriMesh::from_indexed(&verts, &idx);
    for _ in 0..100 {
        let ray = Ray::new(
            v3(r.grid(-12, 12), r.grid(-12, 12), r.grid(-12, 12)),
            rand_dir(&mut r),
        );
        let ha = a.raycast(&ray, Fix128::from_int(40));
        let hb = b.raycast(&ray, Fix128::from_int(40));
        assert_eq!(
            ha.map(|h| (h.t, h.body_index)),
            hb.map(|h| (h.t, h.body_index))
        );
    }
}

// ---------------------------------------------------------------- ray_triangle

/// 独立な f64 解 (平面交点 + 重心座標、Cramer)。`None` は命中なし、
/// `Some(Err)` は境界・平行に近く判定を避ける
fn ref_ray(o: [f64; 3], d: [f64; 3], t: &Triangle) -> Option<Result<f64, ()>> {
    let (a, b, c) = (f(t.v0), f(t.v1), f(t.v2));
    let n = cross(sub(b, a), sub(c, a));
    let nd = dot(n, d);
    if nd.abs() / norm(n) < 1e-3 {
        return Some(Err(()));
    }
    let tt = dot(n, sub(a, o)) / nd;
    let p = [o[0] + d[0] * tt, o[1] + d[1] * tt, o[2] + d[2] * tt];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let (d00, d01, d11, d20, d21) = (
        dot(ab, ab),
        dot(ab, ac),
        dot(ac, ac),
        dot(ap, ab),
        dot(ap, ac),
    );
    let den = d00 * d11 - d01 * d01;
    let s = (d11 * d20 - d01 * d21) / den;
    let u = (d00 * d21 - d01 * d20) / den;
    let margin = s.min(u).min(1.0 - s - u);
    if margin.abs() < 1e-5 || tt.abs() < 1e-5 {
        return Some(Err(()));
    }
    if margin < 0.0 || tt < 0.0 {
        return None;
    }
    Some(Ok(tt))
}

#[test]
fn ray_triangle_matches_independent_plane_intersection() {
    let mut r = Rng(0xABCD_EF01_2345_6789);
    let (mut hit, mut miss) = (0, 0);
    for _ in 0..3000 {
        let t = rand_tri(&mut r, 6);
        let o = v3(r.grid(-9, 9), r.grid(-9, 9), r.grid(-9, 9));
        // 三角形の重心の方へ向け、方向に揺らぎを足す (命中と外れを半々にする)
        let cen = (t.v0 + t.v1 + t.v2) * Fix128::from_ratio(1, 3);
        let jitter = v3(r.grid(-3, 3), r.grid(-3, 3), r.grid(-3, 3));
        let dir = cen - o + jitter;
        if norm(f(dir)) < 0.5 {
            continue;
        }
        let ray = Ray::new(o, dir);
        let want = ref_ray(f(ray.origin), f(ray.direction), &t);
        let got = ray_triangle(&ray, &t, Fix128::from_int(1000));
        match want {
            Some(Err(())) => {}
            None => {
                miss += 1;
                assert!(got.is_none(), "false hit: {t:?} {ray:?} -> {got:?}");
            }
            Some(Ok(tt)) => {
                hit += 1;
                let h = got.unwrap_or_else(|| panic!("missed: {t:?} {ray:?} want t={tt}"));
                assert!(
                    (h.t.to_f64() - tt).abs() < 1e-6 * (1.0 + tt.abs()),
                    "t {} vs {tt}",
                    h.t.to_f64()
                );
                // 命中点 = origin + t d
                let want_p = [
                    f(ray.origin)[0] + f(ray.direction)[0] * h.t.to_f64(),
                    f(ray.origin)[1] + f(ray.direction)[1] * h.t.to_f64(),
                    f(ray.origin)[2] + f(ray.direction)[2] * h.t.to_f64(),
                ];
                assert!(norm(sub(f(h.point), want_p)) < 1e-9);
                // 法線は単位長で、ray に向く (n . d <= 0)、三角形の法線と平行
                let n = f(h.normal);
                assert!((norm(n) - 1.0).abs() < 1e-6);
                assert!(dot(n, f(ray.direction)) <= 1e-9);
                let tn = cross(sub(f(t.v1), f(t.v0)), sub(f(t.v2), f(t.v0)));
                assert!(norm(cross(n, tn)) / norm(tn) < 1e-6);
            }
        }
    }
    assert!(hit > 200 && miss > 200, "hit {hit} miss {miss}");
}

#[test]
fn ray_triangle_max_t_cutoff_and_behind_origin_misses() {
    let t = Triangle::new(v3(-32, -32, 0), v3(64, -32, 0), v3(-32, 64, 0));
    let down = Ray::new(v3(0, 0, 80), v3(0, 0, -16));
    let five = Fix128::from_int(5);
    let eps = Fix128::from_ratio(1, 1_000_000);
    // t は 5 (除算の丸めで 1e-15 程度ずれる)。max_t が少し上なら命中、少し下なら外れ
    let h = ray_triangle(&down, &t, five + eps).expect("hit below max_t");
    assert!((h.t - five).abs() < eps, "t = {}", h.t.to_f64());
    assert!(ray_triangle(&down, &t, five - eps).is_none());
    // 三角形の背後から離れる向き
    let away = Ray::new(v3(0, 0, 80), v3(0, 0, 16));
    assert!(ray_triangle(&away, &t, Fix128::from_int(100)).is_none());
    // 裏面からも命中し、法線は ray に向く
    let up = Ray::new(v3(0, 0, -80), v3(0, 0, 16));
    let hb = ray_triangle(&up, &t, Fix128::from_int(100)).expect("back face");
    assert!((hb.t - five).abs() < eps);
    assert!(
        f(hb.normal)[2] < 0.0,
        "normal faces the ray: {:?}",
        hb.normal
    );
    assert!(f(h.normal)[2] > 0.0);
}

#[test]
fn ray_triangle_includes_edges_and_vertices() {
    let t = Triangle::new(v3(0, 0, 0), v3(64, 0, 0), v3(0, 64, 0));
    let down = |x: i64, y: i64| Ray::new(v3(x, y, 80), v3(0, 0, -16));
    let max = Fix128::from_int(100);
    for (x, y) in [
        (0, 0),
        (64, 0),
        (0, 64),
        (32, 0),
        (0, 32),
        (32, 32),
        (16, 0),
        (0, 48),
    ] {
        assert!(
            ray_triangle(&down(x, y), &t, max).is_some(),
            "edge/vertex ({x},{y})"
        );
    }
    for (x, y) in [(-1, 0), (0, -1), (33, 32), (32, 33), (65, 0)] {
        assert!(
            ray_triangle(&down(x, y), &t, max).is_none(),
            "outside ({x},{y})"
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-012: MT_EPSILON (2^-24) との比較が絶対値で、det = 2 * 面積 * |cos| が小さい三角形 (辺 1e-4 m、面積 5e-9) は真上からの ray でも常に None"]
fn small_triangles_are_not_invisible_to_rays() {
    // 辺 1e-4 m (0.1 mm) の三角形に真上から ray。面積 5e-9 < MT_EPSILON (2^-24 = 6e-8) で det が閾値を下回る
    let e = Fix128::from_ratio(1, 10_000);
    let t = Triangle::new(
        Vec3Fix::ZERO,
        Vec3Fix::new(e, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(Fix128::ZERO, e, Fix128::ZERO),
    );
    let third = Fix128::from_ratio(1, 30_000);
    let ray = Ray::new(Vec3Fix::new(third, third, Fix128::ONE), v3(0, 0, -16));
    assert!(
        ray_triangle(&ray, &t, Fix128::from_int(10)).is_some(),
        "0.1 mm triangle is missed by a ray through its centroid"
    );
}

#[test]
fn rays_through_shared_diagonals_do_not_leak_between_two_triangles() {
    // 2 三角形で作る四角形の対角線の中点 (dyadic) を狙う: 少なくとも一方が命中する (水密)
    let mut r = Rng(99);
    let mut leaks = 0;
    for _ in 0..300 {
        let (ax, ay) = (r.grid(-6, 0), r.grid(-6, 0));
        let (bx, by) = (r.grid(1, 7), r.grid(-6, 0));
        let (cx, cy) = (r.grid(1, 7), r.grid(1, 7));
        let (dx, dy) = (r.grid(-6, 0), r.grid(1, 7));
        let z = |r: &mut Rng| r.grid(-2, 2);
        let (za, zb, zc, zd) = (z(&mut r), z(&mut r), z(&mut r), z(&mut r));
        let (a, b, c, d) = (
            v3(ax, ay, za),
            v3(bx, by, zb),
            v3(cx, cy, zc),
            v3(dx, dy, zd),
        );
        let m = TriMesh::from_triangles(vec![Triangle::new(a, b, c), Triangle::new(a, c, d)]);
        // 共有辺 a-c の中点: 辺上の点を真上から
        let mid = (a + c) * Fix128::from_ratio(1, 2);
        let ray = Ray::new(
            Vec3Fix::new(mid.x, mid.y, Fix128::from_int(20)),
            v3(0, 0, -16),
        );
        if m.raycast(&ray, Fix128::from_int(60)).is_none() {
            leaks += 1;
        }
    }
    assert_eq!(
        leaks, 0,
        "{leaks} / 300 rays through a shared edge hit neither triangle"
    );
}

// ---------------------------------------------------------------- raycast (BVH vs 総当たり)

#[test]
fn raycast_equals_brute_force_over_all_triangles() {
    let mut r = Rng(0x0F0F_1234_ABCD_0001);
    let mut hits = 0;
    for mesh_i in 0..6 {
        let n = [1usize, 3, 10, 40, 90, 200][mesh_i];
        let m = rand_mesh(&mut r, n, 12);
        for _ in 0..200 {
            let ray = Ray::new(
                v3(r.grid(-14, 14), r.grid(-14, 14), r.grid(-14, 14)),
                rand_dir(&mut r),
            );
            let max_t = Fix128::from_int([2, 8, 30, 100][(r.next() % 4) as usize]);
            // 総当たり
            let mut best: Option<(Fix128, usize)> = None;
            let mut second = f64::MAX;
            for (i, t) in m.triangles.iter().enumerate() {
                if let Some(h) = ray_triangle(&ray, t, max_t) {
                    match best {
                        Some((bt, _)) if h.t >= bt => {
                            second = second.min(h.t.to_f64());
                        }
                        Some((bt, _)) => {
                            second = second.min(bt.to_f64());
                            best = Some((h.t, i));
                        }
                        None => best = Some((h.t, i)),
                    }
                }
            }
            let got = m.raycast(&ray, max_t);
            match (best, got) {
                (None, None) => {}
                (Some((bt, bi)), Some(h)) => {
                    hits += 1;
                    assert_eq!(h.t, bt, "closest t");
                    if second - bt.to_f64() > 1e-9 {
                        assert_eq!(h.body_index, bi, "closest triangle index");
                    }
                    assert!(h.t <= max_t);
                }
                (b, g) => panic!("BVH {g:?} vs brute {b:?} (n={n}, ray={ray:?}, max_t={max_t:?})"),
            }
        }
    }
    assert!(hits > 100, "hits {hits}");
}

#[test]
fn raycast_returns_the_nearest_of_stacked_triangles_and_respects_max_t() {
    let mut tris = Vec::new();
    for k in 1..=5 {
        let z = k * 16 * 2;
        tris.push(Triangle::new(v3(-64, -64, z), v3(64, -64, z), v3(0, 64, z)));
    }
    let m = TriMesh::from_triangles(tris);
    let ray = Ray::new(v3(0, 0, 0), v3(0, 0, 16));
    let h = m.raycast(&ray, Fix128::from_int(100)).unwrap();
    assert_eq!(h.t, Fix128::from_int(2));
    assert_eq!(h.body_index, 0);
    assert_eq!(
        m.raycast(&ray, Fix128::from_int(2)).unwrap().t,
        Fix128::from_int(2)
    );
    assert!(m.raycast(&ray, Fix128::ONE).is_none());
}

fn far_offset_raycast_equals_brute_force(offset_m: i64) {
    let off = offset_m * 16;
    let mut r = Rng(31337);
    let tris: Vec<Triangle> = (0..40)
        .map(|_| {
            let t = rand_tri(&mut r, 8);
            let sh = |v: Vec3Fix| v + v3(off, off, off);
            Triangle::new(sh(t.v0), sh(t.v1), sh(t.v2))
        })
        .collect();
    let m = TriMesh::from_triangles(tris);
    let mut hits = 0;
    for _ in 0..200 {
        let o = v3(
            off + r.grid(-10, 10),
            off + r.grid(-10, 10),
            off + r.grid(-10, 10),
        );
        let ray = Ray::new(o, rand_dir(&mut r));
        let max_t = Fix128::from_int(40);
        let brute = m
            .triangles
            .iter()
            .filter_map(|t| ray_triangle(&ray, t, max_t).map(|h| h.t))
            .min();
        let got = m.raycast(&ray, max_t).map(|h| h.t);
        assert_eq!(got, brute);
        hits += usize::from(got.is_some());
    }
    assert!(hits > 20, "{hits}");
}

#[test]
fn raycast_on_far_offset_mesh_equals_brute_force() {
    // 原点から遠い (1e6 m) メッシュでも BVH 経由と総当たりが一致する
    far_offset_raycast_equals_brute_force(1_000_000);
}

#[test]
fn raycast_on_mesh_beyond_i32_range_equals_brute_force() {
    // BVH の query は座標を i32 に圧縮する。3e9 m は i32 の範囲 (2.1e9) の外
    far_offset_raycast_equals_brute_force(3_000_000_000);
}

// ---------------------------------------------------------------- closest_point

fn brute_dist(m: &TriMesh, p: Vec3Fix) -> f64 {
    m.triangles
        .iter()
        .map(|t| norm(sub(f(t.closest_point(p)), f(p))))
        .fold(f64::MAX, f64::min)
}

#[test]
fn mesh_closest_point_equals_brute_force_for_nearby_queries() {
    let mut r = Rng(0xC0FF_EE00_1111_2222);
    for n in [1usize, 4, 25, 120] {
        let m = rand_mesh(&mut r, n, 10);
        for _ in 0..80 {
            let p = v3(r.grid(-30, 30), r.grid(-30, 30), r.grid(-30, 30));
            let (cp, idx) = m.closest_point(p);
            let d = norm(sub(f(cp), f(p)));
            let want = brute_dist(&m, p);
            assert!((d - want).abs() < 1e-9, "n={n} p={p:?}: {d} vs {want}");
            // 返った index の三角形上の最近点であること
            assert_eq!(f(m.triangles[idx].closest_point(p)), f(cp));
        }
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-010: 候補を点の ±1000 の箱で絞るため、メッシュから 1000 より遠い点では候補が 0 件になり triangles[0].v0 (index 0) を返す (点 (5000,0,0): 返値までの距離 5006.44、最近は 4990.32)"]
fn mesh_closest_point_is_correct_for_queries_far_from_the_mesh() {
    // doc: "Closest point on mesh to a given point" に距離の制限は無い。実装は ±1000 の箱で候補を絞る
    let mut r = Rng(5150);
    let m = rand_mesh(&mut r, 20, 10);
    for p in [
        v3(5000 * 16, 0, 0),
        v3(0, -3000 * 16, 2000 * 16),
        v3(1500 * 16, 1500 * 16, 1500 * 16),
    ] {
        let (cp, _) = m.closest_point(p);
        let d = norm(sub(f(cp), f(p)));
        let want = brute_dist(&m, p);
        assert!(
            (d - want).abs() < 1e-6,
            "far query {p:?}: {d} vs nearest {want}"
        );
    }
}

#[test]
fn mesh_closest_point_on_an_empty_mesh_does_not_panic() {
    let m = TriMesh::from_triangles(Vec::new());
    let res = std::panic::catch_unwind(|| m.closest_point(Vec3Fix::ZERO));
    assert!(
        res.is_ok(),
        "closest_point panics on an empty mesh (triangles[0])"
    );
}

// ---------------------------------------------------------------- collide_sphere

#[test]
fn collide_sphere_equals_brute_force_and_contact_is_self_consistent() {
    let mut r = Rng(0x5EED_0000_AAAA_BBBB);
    let mut contacts = 0;
    for n in [2usize, 12, 60, 150] {
        let m = rand_mesh(&mut r, n, 8);
        for _ in 0..150 {
            let c = v3(r.grid(-10, 10), r.grid(-10, 10), r.grid(-10, 10));
            let rad = fx(8 + (r.next() % 40) as i64); // 0.5 .. 3.0
            let want_d = brute_dist(&m, c);
            let got = m.collide_sphere(c, rad);
            let r64 = rad.to_f64();
            if want_d < r64 - 1e-9 {
                let ct = got.unwrap_or_else(|| panic!("missed contact: dist {want_d} < r {r64}"));
                contacts += 1;
                assert!((ct.depth.to_f64() - (r64 - want_d)).abs() < 1e-9, "depth");
                // 法線は単位長、center = point_b + normal * (r - depth)
                let nn = f(ct.normal);
                assert!((norm(nn) - 1.0).abs() < 1e-9);
                let pb = f(ct.point_b);
                let d = r64 - ct.depth.to_f64();
                for ax in 0..3 {
                    assert!(
                        (pb[ax] + nn[ax] * d - f(c)[ax]).abs() < 1e-9,
                        "point_b + n d = center"
                    );
                }
                // point_a は球の表面: center - normal * r
                let pa = f(ct.point_a);
                for ax in 0..3 {
                    assert!((pa[ax] - (f(c)[ax] - nn[ax] * r64)).abs() < 1e-9);
                }
            } else if want_d > r64 + 1e-9 {
                assert!(got.is_none(), "false contact: dist {want_d} > r {r64}");
            }
        }
    }
    assert!(contacts > 60, "{contacts}");
}

#[test]
fn collide_sphere_center_on_the_triangle_uses_the_face_normal() {
    let t = Triangle::new(v3(-32, -32, 0), v3(32, -32, 0), v3(0, 32, 0));
    let m = TriMesh::from_triangles(vec![t]);
    let ct = m.collide_sphere(v3(0, 0, 0), Fix128::ONE).unwrap();
    assert_eq!(ct.depth, Fix128::ONE);
    assert_eq!(f(ct.normal), [0.0, 0.0, 1.0]);
}

#[test]
fn collide_sphere_touching_exactly_is_not_a_contact() {
    let t = Triangle::new(v3(-32, -32, 0), v3(32, -32, 0), v3(0, 32, 0));
    let m = TriMesh::from_triangles(vec![t]);
    assert!(m.collide_sphere(v3(0, 0, 16), Fix128::ONE).is_none());
    assert!(m.collide_sphere(v3(0, 0, 15), Fix128::ONE).is_some());
}

// ---------------------------------------------------------------- collide_capsule

fn seg_tri_dist_sampled(a: Vec3Fix, b: Vec3Fix, t: &Triangle, n: usize) -> f64 {
    let (fa, fb) = (f(a), f(b));
    let mut best = f64::MAX;
    for i in 0..=n {
        let s = i as f64 / n as f64;
        let p = [
            fa[0] + (fb[0] - fa[0]) * s,
            fa[1] + (fb[1] - fa[1]) * s,
            fa[2] + (fb[2] - fa[2]) * s,
        ];
        // 点 p の三角形への距離 (検証済みの closest_point を使うため Fix128 に戻す)
        let pf = Vec3Fix::new(
            Fix128::from_f64(p[0]),
            Fix128::from_f64(p[1]),
            Fix128::from_f64(p[2]),
        );
        let d = norm(sub(f(t.closest_point(pf)), p));
        if d < best {
            best = d;
        }
    }
    best
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-014: 線分-三角形の最近点を 2 回の交互射影で近似するため距離を過大評価する (250 配置中 63 件で標本より悪い、最悪 +0.55)"]
fn collide_capsule_depth_equals_radius_minus_segment_triangle_distance() {
    let mut r = Rng(0xCA95_0000_0000_0001);
    let mut contacts = 0;
    let mut worst = 0.0f64;
    let mut bad = Vec::new();
    for _ in 0..250 {
        let t = rand_tri(&mut r, 6);
        let m = TriMesh::from_triangles(vec![t]);
        let a = v3(r.grid(-8, 8), r.grid(-8, 8), r.grid(-8, 8));
        let b = v3(r.grid(-8, 8), r.grid(-8, 8), r.grid(-8, 8));
        let rad = fx(8 + (r.next() % 40) as i64);
        let n = 3000;
        let step = norm(sub(f(a), f(b))) / n as f64;
        let d_true_ub = seg_tri_dist_sampled(a, b, &t, n); // 真の距離 + 最大 step
        let r64 = rad.to_f64();
        let got = m.collide_capsule(a, b, rad);
        if d_true_ub + step < r64 - 1e-6 {
            // 確実に貫入している
            match got {
                None => bad.push(format!(
                    "missed contact: tri {t:?} a {a:?} b {b:?} r {r64} dist<= {d_true_ub}"
                )),
                Some(c) => {
                    contacts += 1;
                    let d_impl = r64 - c.depth.to_f64();
                    // 実装距離が標本 (上界) より step を超えて悪い = 精度不足
                    let excess = d_impl - d_true_ub;
                    worst = worst.max(excess);
                    if excess > 1e-6 {
                        bad.push(format!("shallow: impl dist {d_impl} sampled {d_true_ub} tri {t:?} a {a:?} b {b:?}"));
                    }
                }
            }
        } else if d_true_ub - step > r64 + 1e-6 {
            assert!(got.is_none(), "false contact: dist {d_true_ub} > r {r64}");
        }
    }
    assert!(contacts > 40, "{contacts}");
    assert!(
        bad.is_empty(),
        "{} defects (worst excess {worst}); first: {}",
        bad.len(),
        bad.first().unwrap()
    );
}

#[test]
fn collide_capsule_degenerate_segment_equals_sphere() {
    let mut r = Rng(11);
    for _ in 0..80 {
        let m = rand_mesh(&mut r, 6, 6);
        let c = v3(r.grid(-8, 8), r.grid(-8, 8), r.grid(-8, 8));
        let rad = fx(8 + (r.next() % 40) as i64);
        let s = m.collide_sphere(c, rad);
        let k = m.collide_capsule(c, c, rad);
        assert_eq!(s.map(|x| x.depth), k.map(|x| x.depth));
    }
}

// ---------------------------------------------------------------- collide_aabb

/// 三角形と AABB の分離軸テスト (Akenine-Möller)。重なっていれば true
fn sat_tri_box(t: &Triangle, c: [f64; 3], h: [f64; 3]) -> bool {
    let v = [sub(f(t.v0), c), sub(f(t.v1), c), sub(f(t.v2), c)];
    let e = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    let ax_ = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut axes: Vec<[f64; 3]> = ax_.to_vec();
    axes.push(cross(e[0], e[1]));
    for ei in e {
        for a in ax_ {
            axes.push(cross(ei, a));
        }
    }
    for ax in axes {
        if norm(ax) < 1e-12 {
            continue;
        }
        let p: Vec<f64> = v.iter().map(|x| dot(*x, ax)).collect();
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        let rr = h[0] * ax[0].abs() + h[1] * ax[1].abs() + h[2] * ax[2].abs();
        if lo > rr + 1e-9 || hi < -rr - 1e-9 {
            return false;
        }
    }
    true
}

/// 三角形上の標本点 (重心座標の格子) が箱の内側に 1 つでもあるか (SAT とは独立な確認)
fn sample_inside(t: &Triangle, c: [f64; 3], h: [f64; 3]) -> bool {
    let n = 80;
    let a = f(t.v0);
    let ab = sub(f(t.v1), a);
    let ac = sub(f(t.v2), a);
    for i in 0..=n {
        for j in 0..=(n - i) {
            let (s, u) = (i as f64 / n as f64, j as f64 / n as f64);
            let ok = (0..3).all(|k| (a[k] + ab[k] * s + ac[k] * u - c[k]).abs() < h[k] - 1e-3);
            if ok {
                return true;
            }
        }
    }
    false
}

/// 重なりが明確 (分離軸の余裕が 0.05 以上) か
fn sat_margin(t: &Triangle, c: [f64; 3], h: [f64; 3]) -> f64 {
    // 各軸での (重なり量) の最小値。正なら貫入
    let v = [sub(f(t.v0), c), sub(f(t.v1), c), sub(f(t.v2), c)];
    let e = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    let ax_ = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut axes: Vec<[f64; 3]> = ax_.to_vec();
    axes.push(cross(e[0], e[1]));
    for ei in e {
        for a in ax_ {
            axes.push(cross(ei, a));
        }
    }
    let mut m = f64::MAX;
    for ax in axes {
        let l = norm(ax);
        if l < 1e-12 {
            continue;
        }
        let p: Vec<f64> = v.iter().map(|x| dot(*x, ax) / l).collect();
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        let rr = (h[0] * ax[0].abs() + h[1] * ax[1].abs() + h[2] * ax[2].abs()) / l;
        m = m.min((rr - lo).min(hi + rr));
    }
    m
}

#[test]
fn collide_aabb_never_reports_contact_for_a_triangle_separated_from_the_box() {
    // 分離軸が明確 (余裕 0.05 以上) な配置では接触なし。接触を返すときは法線が単位長で depth > 0
    let mut r = Rng(0xBA5E_BA11_0000_0002);
    let (mut sep, mut contacts) = (0, 0);
    for _ in 0..800 {
        let t = rand_tri(&mut r, 5);
        let m = TriMesh::from_triangles(vec![t]);
        let c = v3(r.grid(-6, 6), r.grid(-6, 6), r.grid(-6, 6));
        let hh = |r: &mut Rng| 16 + (r.next() % 33) as i64;
        let h = v3(hh(&mut r), hh(&mut r), hh(&mut r));
        let got = m.collide_aabb(&AABB::new(c - h, c + h));
        let mg = sat_margin(&t, f(c), f(h));
        if mg < -0.05 {
            sep += 1;
            assert!(
                got.is_none(),
                "false contact: tri {t:?} center {c:?} half {h:?} margin {mg}"
            );
        }
        if let Some(ct) = got {
            contacts += 1;
            assert!(ct.depth > Fix128::ZERO);
            assert!((norm(f(ct.normal)) - 1.0).abs() < 1e-9);
        }
    }
    assert!(sep > 200 && contacts > 50, "sep {sep} contacts {contacts}");
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-017: collide_aabb の point_a が center - normal * depth (箱の表面でも最深点でもない)。床の上の箱で point_a == point_b (距離 0) となり、Contact の不変条件 point_a - point_b = -normal * depth を満たさない (collide_sphere / collide_capsule は満たす)"]
fn collide_aabb_contact_points_are_separated_by_depth_along_the_normal() {
    let m = TriMesh::from_triangles(vec![
        Triangle::new(v3(-160, 0, -160), v3(160, 0, -160), v3(160, 0, 160)),
        Triangle::new(v3(-160, 0, -160), v3(160, 0, 160), v3(-160, 0, 160)),
    ]);
    let c = v3(0, 8, 0);
    let h = v3(16, 16, 16);
    let ct = m.collide_aabb(&AABB::new(c - h, c + h)).unwrap();
    // 箱の最下点 (0, -0.5, 0) が床 (0, 0, 0) の下 0.5 に入っている
    let pa = f(ct.point_a);
    let pb = f(ct.point_b);
    let n = f(ct.normal);
    let d = ct.depth.to_f64();
    for ax in 0..3 {
        assert!(
            (pa[ax] - pb[ax] + n[ax] * d).abs() < 1e-9,
            "axis {ax}: point_a {pa:?} point_b {pb:?} normal {n:?} depth {d}"
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-013: 重なる箱の 30 / 239 (12.6%) を接触なしと答える (箱中心への最近点が箱の外なら棄却するため)。標本点で箱内を確認済"]
fn collide_aabb_reports_contact_iff_the_triangle_overlaps_the_box() {
    let mut r = Rng(0xBA5E_BA11_0000_0001);
    let (mut overlap, mut agree) = (0, 0);
    let mut false_neg = Vec::new();
    let mut false_pos = Vec::new();
    for _ in 0..600 {
        let t = rand_tri(&mut r, 5);
        let m = TriMesh::from_triangles(vec![t]);
        let c = v3(r.grid(-4, 4), r.grid(-4, 4), r.grid(-4, 4));
        let hh = |r: &mut Rng| 16 + (r.next() % 33) as i64;
        let h = v3(hh(&mut r), hh(&mut r), hh(&mut r));
        let aabb = AABB::new(c - h, c + h);
        let (cf, hf) = (f(c), f(h));
        let mg = sat_margin(&t, cf, hf);
        let got = m.collide_aabb(&aabb).is_some();
        if mg > 0.05 {
            overlap += 1;
            if got {
                agree += 1;
            } else if sample_inside(&t, cf, hf) {
                // 三角形上の点が箱の内側にあるのに接触なし = 確実な取りこぼし
                false_neg.push(format!("tri {t:?} center {c:?} half {h:?} margin {mg}"));
            }
        } else if mg < -0.05 && got {
            false_pos.push(format!("tri {t:?} center {c:?} half {h:?} margin {mg}"));
        }
        let _ = sat_tri_box;
    }
    assert!(overlap > 100, "overlap {overlap}");
    assert!(
        false_neg.is_empty() && false_pos.is_empty(),
        "overlapping boxes missed: {} / {} (agree {agree}); false contacts {}; first miss: {}",
        false_neg.len(),
        overlap,
        false_pos.len(),
        false_neg.first().cloned().unwrap_or_default()
    );
}

#[test]
fn collide_aabb_ground_plane_penetration_depth_and_normal() {
    // 床 (y = 0) に箱 (中心 y = 0.5, 半幅 1): 貫入 0.5、法線は +y (箱が上)
    let m = TriMesh::from_triangles(vec![
        Triangle::new(v3(-160, 0, -160), v3(160, 0, -160), v3(160, 0, 160)),
        Triangle::new(v3(-160, 0, -160), v3(160, 0, 160), v3(-160, 0, 160)),
    ]);
    let c = v3(0, 8, 0);
    let h = v3(16, 16, 16);
    let ct = m.collide_aabb(&AABB::new(c - h, c + h)).expect("contact");
    assert!(
        (ct.depth.to_f64() - 0.5).abs() < 1e-9,
        "depth {}",
        ct.depth.to_f64()
    );
    assert!((f(ct.normal)[1] - 1.0).abs() < 1e-9 || (f(ct.normal)[1] + 1.0).abs() < 1e-9);
    assert!(
        f(ct.normal)[1] > 0.0,
        "normal points from the mesh to the box: {:?}",
        ct.normal
    );
}

#[test]
fn collide_aabb_separated_box_has_no_contact() {
    let m = TriMesh::from_triangles(vec![Triangle::new(
        v3(-160, 0, -160),
        v3(160, 0, -160),
        v3(160, 0, 160),
    )]);
    let c = v3(0, 40, 0);
    let h = v3(16, 16, 16);
    assert!(m.collide_aabb(&AABB::new(c - h, c + h)).is_none());
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-013: 平面 x + 0.1y = 1.05 の三角形が箱 [-1,1]^3 と交わるのに接触なし (中心からの垂線の足 x = 1.0396 が箱の外)"]
fn collide_aabb_slanted_triangle_clipping_a_box_corner_is_a_contact() {
    // 平面 x + 0.1 y = 1.05 上の大きな三角形。箱 [-1, 1]^3 とは (0.99, 0.6, 0) などで交わるが、
    // 箱の中心から平面への垂線の足 (1.0396, 0.104, 0) は箱の外 (x > 1)
    let t = Triangle::new(
        Vec3Fix::new(
            Fix128::from_ratio(105, 100),
            Fix128::ZERO,
            Fix128::from_int(-10),
        ),
        Vec3Fix::new(
            Fix128::from_ratio(105, 100),
            Fix128::ZERO,
            Fix128::from_int(10),
        ),
        Vec3Fix::new(
            Fix128::from_ratio(5, 100),
            Fix128::from_int(10),
            Fix128::ZERO,
        ),
    );
    let m = TriMesh::from_triangles(vec![t]);
    let one = Fix128::ONE;
    let h = Vec3Fix::new(one, one, one);
    let inside = Vec3Fix::new(
        Fix128::from_ratio(99, 100),
        Fix128::from_ratio(6, 10),
        Fix128::ZERO,
    );
    // 前提: この点は三角形の平面上にあり箱の内側
    assert!((f(inside)[0] + 0.1 * f(inside)[1] - 1.05).abs() < 1e-9);
    assert!(
        m.collide_aabb(&AABB::new(-h, h)).is_some(),
        "box overlapping the triangle reports no contact"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-014: 整数例 (線分 (6,8,-7)-(-5,0,5)、三角形 (6,2,-2) (-5,-2,5) (6,4,5)、半径 2.5) で真の距離 1.756 < r なのに接触なし (実装の近似距離 3.377)"]
fn collide_capsule_integer_example_with_oblique_segment_is_a_contact() {
    // 手計算の例: 三角形 (6,2,-2) (-5,-2,5) (6,4,5)、線分 (6,8,-7)-(-5,0,5)、半径 2.5
    // 線分上の標本 3000 点の最小距離は 1.7558 (< 2.5) だが、実装の近似距離は 3.377
    let t = Triangle::new(
        Vec3Fix::from_int(6, 2, -2),
        Vec3Fix::from_int(-5, -2, 5),
        Vec3Fix::from_int(6, 4, 5),
    );
    let m = TriMesh::from_triangles(vec![t]);
    let a = Vec3Fix::from_int(6, 8, -7);
    let b = Vec3Fix::from_int(-5, 0, 5);
    let d = seg_tri_dist_sampled(a, b, &t, 3000);
    assert!(d < 1.76 && d > 1.75, "reference distance {d}");
    assert!(m.collide_capsule(a, b, Fix128::from_ratio(5, 2)).is_some());
}

#[test]
fn ray_starting_on_the_triangle_hits_at_t_zero() {
    let t = Triangle::new(v3(-32, -32, 0), v3(64, -32, 0), v3(-32, 64, 0));
    let ray = Ray::new(v3(0, 0, 0), v3(0, 0, 16));
    let h = ray_triangle(&ray, &t, Fix128::from_int(10)).expect("origin on the triangle");
    assert_eq!(h.t, Fix128::ZERO);
}

#[test]
fn triangle_count_and_vertices_are_exposed_in_order() {
    let mut r = Rng(2024);
    for n in [0usize, 1, 5, 33] {
        let tris: Vec<Triangle> = (0..n).map(|_| rand_tri(&mut r, 9)).collect();
        let m = TriMesh::from_triangles(tris.clone());
        assert_eq!(m.triangle_count(), n);
        assert_eq!(m.triangles, tris);
    }
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-016: triangles は pub field だが BVH は構築時の AABB のまま。triangles[i] を書き換えると raycast / collide_sphere が新しい位置を見つけられない (BVH の再構築 API も無い)"]
fn editing_the_public_triangles_keeps_queries_consistent() {
    let t0 = Triangle::new(v3(-16, -16, 0), v3(16, -16, 0), v3(0, 16, 0));
    let mut m = TriMesh::from_triangles(vec![t0]);
    let shift = v3(1600, 0, 0); // x + 100
    m.triangles[0] = Triangle::new(t0.v0 + shift, t0.v1 + shift, t0.v2 + shift);
    let ray = Ray::new(v3(1600, 0, 80), v3(0, 0, -16));
    let brute = ray_triangle(&ray, &m.triangles[0], Fix128::from_int(100));
    assert!(brute.is_some());
    assert!(
        m.raycast(&ray, Fix128::from_int(100)).is_some(),
        "raycast misses the edited triangle (stale BVH)"
    );
}

#[test]
fn degenerate_collinear_triangle_closest_point_is_the_nearest_point_on_the_segment() {
    let mut r = Rng(606);
    for _ in 0..200 {
        let a = v3(r.grid(-4, 4), r.grid(-4, 4), r.grid(-4, 4));
        let d = v3(r.grid(-3, 3), r.grid(-3, 3), r.grid(-3, 3));
        if norm(f(d)) < 0.5 {
            continue;
        }
        // 共線: a, a + d, a + 2 d
        let t = Triangle::new(a, a + d, a + d * Fix128::from_int(2));
        let p = v3(r.grid(-9, 9), r.grid(-9, 9), r.grid(-9, 9));
        let cp = f(t.closest_point(p));
        // 線分 a .. a + 2d への最近点 (閉形式)
        let (fa, fd, fp) = (f(a), f(d), f(p));
        let s = (dot(sub(fp, fa), fd) / dot(fd, fd) / 2.0).clamp(0.0, 1.0);
        let want = [
            fa[0] + 2.0 * fd[0] * s,
            fa[1] + 2.0 * fd[1] * s,
            fa[2] + 2.0 * fd[2] * s,
        ];
        assert!(
            norm(sub(cp, want)) < 1e-9,
            "tri {t:?} p {p:?}: {cp:?} vs {want:?}"
        );
    }
}

fn ground_triangles(x0: i64, x1: i64) -> TriMesh {
    // y = 0 の平面 (x0..x1, z = -10..10)
    TriMesh::from_triangles(vec![
        Triangle::new(
            v3(x0 * 16, 0, -160),
            v3(x1 * 16, 0, -160),
            v3(x1 * 16, 0, 160),
        ),
        Triangle::new(
            v3(x0 * 16, 0, -160),
            v3(x1 * 16, 0, 160),
            v3(x0 * 16, 0, 160),
        ),
    ])
}

#[test]
fn collide_capsule_parallel_above_a_plane_has_closed_form_contact() {
    // 線分 (-1, 0.5, 0)-(1, 0.5, 0)、半径 1、床 y = 0: depth 0.5、法線 +y、point_a = (x, -0.5, 0)、point_b = (x, 0, 0)
    let m = ground_triangles(-10, 10);
    let ct = m
        .collide_capsule(v3(-16, 8, 0), v3(16, 8, 0), Fix128::ONE)
        .expect("contact");
    assert!((ct.depth.to_f64() - 0.5).abs() < 1e-9);
    assert_eq!(f(ct.normal), [0.0, 1.0, 0.0]);
    let (pa, pb) = (f(ct.point_a), f(ct.point_b));
    assert!((pa[1] + 0.5).abs() < 1e-9 && pb[1].abs() < 1e-9);
    assert!((pa[0] - pb[0]).abs() < 1e-9 && (pa[2] - pb[2]).abs() < 1e-9);
}

#[test]
fn collide_capsule_endpoint_beyond_the_segment_end_uses_the_clamped_end() {
    // 線分 (0, 0.5, 0)-(1, 0.5, 0)、床は x = 4..6 だけ。最近点は端点 b = (1, 0.5, 0): 距離 sqrt(9 + 0.25) = 3.0414
    let m = ground_triangles(4, 6);
    let near = m.collide_capsule(v3(0, 8, 0), v3(16, 8, 0), Fix128::from_ratio(32, 10));
    let ct = near.expect("end point is within the radius 3.2");
    let want = 3.2 - (9.25f64).sqrt();
    assert!(
        (ct.depth.to_f64() - want).abs() < 1e-9,
        "depth {} want {want}",
        ct.depth.to_f64()
    );
    // 半径 3.0 では届かない
    assert!(m
        .collide_capsule(v3(0, 8, 0), v3(16, 8, 0), Fix128::from_int(3))
        .is_none());
}
