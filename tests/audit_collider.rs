//! Audit S3W3 oracles for `src/collider.rs` (shapes, support functions, GJK,
//! EPA, `contact`).
//!
//! Every expected value is a closed form computed in `f64` outside the crate:
//! * GJK colliding  <=>  the shapes overlap, which for the shape pairs below is
//!   a distance comparison (centre distance, point-to-box distance, segment to
//!   segment distance) or the separating axis test (boxes);
//! * `contact`: sphere pair `depth = r1 + r2 - |c1 - c2|`, box pair
//!   `depth = min over axes (h1 + h2 - |c1 - c2|_axis)`, normal pointing from
//!   B to A;
//! * support: `max_p <p, d>` over the shape (sphere `c.d + r|d|`, hull the
//!   maximum vertex dot, box the sum of per-axis maxima).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::{
    contact, epa, gjk, Capsule, ConvexHull, ScaledShape, Sphere, Support, AABB,
};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::metric::MetricWeights;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Deterministic LCG in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    fn vec(&mut self, lo: f64, hi: f64) -> [f64; 3] {
        [self.range(lo, hi), self.range(lo, hi), self.range(lo, hi)]
    }
}

fn aabb(c: [f64; 3], h: [f64; 3]) -> AABB {
    AABB::new(
        v3(c[0] - h[0], c[1] - h[1], c[2] - h[2]),
        v3(c[0] + h[0], c[1] + h[1], c[2] + h[2]),
    )
}

fn point_box_dist(p: [f64; 3], c: [f64; 3], h: [f64; 3]) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        let d = ((p[i] - c[i]).abs() - h[i]).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time
/// Collision Detection 5.1.9).
fn seg_seg_dist(p1: [f64; 3], q1: [f64; 3], p2: [f64; 3], q2: [f64; 3]) -> f64 {
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let (s, t);
    if a <= 1e-12 && e <= 1e-12 {
        return norm(r);
    }
    if a <= 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            let mut s_ = if denom > 1e-12 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t_ = (b * s_ + f) / e;
            if t_ < 0.0 {
                t_ = 0.0;
                s_ = (-c / a).clamp(0.0, 1.0);
            } else if t_ > 1.0 {
                t_ = 1.0;
                s_ = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s_;
            t = t_;
        }
    }
    let c1 = [p1[0] + d1[0] * s, p1[1] + d1[1] * s, p1[2] + d1[2] * s];
    let c2 = [p2[0] + d2[0] * t, p2[1] + d2[1] * t, p2[2] + d2[2] * t];
    norm(sub(c1, c2))
}

// ---------------------------------------------------------------------------
// GJK == overlap test
// ---------------------------------------------------------------------------

#[test]
fn gjk_sphere_pairs_agree_with_centre_distance() {
    let mut g = Lcg(1);
    let mut n_hit = 0;
    for _ in 0..300 {
        let (ra, rb) = (g.range(0.2, 1.5), g.range(0.2, 1.5));
        let ca = g.vec(-3.0, 3.0);
        let cb = g.vec(-3.0, 3.0);
        let d = norm(sub(ca, cb));
        if (d - (ra + rb)).abs() < 1e-3 {
            continue;
        }
        let want = d < ra + rb;
        n_hit += usize::from(want);
        let got = gjk(
            &Sphere::new(v3(ca[0], ca[1], ca[2]), fx(ra)),
            &Sphere::new(v3(cb[0], cb[1], cb[2]), fx(rb)),
        )
        .colliding;
        assert_eq!(got, want, "spheres d {d} ra+rb {}", ra + rb);
    }
    assert!(
        n_hit > 15 && n_hit < 285,
        "scene must mix hits and misses ({n_hit})"
    );
}

#[test]
fn gjk_box_pairs_agree_with_the_separating_axis_test() {
    let mut g = Lcg(2);
    let mut n_hit = 0;
    for _ in 0..300 {
        let (ha, hb) = (g.vec(0.2, 1.2), g.vec(0.2, 1.2));
        let (ca, cb) = (g.vec(-2.5, 2.5), g.vec(-2.5, 2.5));
        let mut want = true;
        let mut near_edge = false;
        for i in 0..3 {
            let gap = (ca[i] - cb[i]).abs() - (ha[i] + hb[i]);
            if gap.abs() < 1e-3 {
                near_edge = true;
            }
            if gap >= 0.0 {
                want = false;
            }
        }
        if near_edge {
            continue;
        }
        n_hit += usize::from(want);
        assert_eq!(
            gjk(&aabb(ca, ha), &aabb(cb, hb)).colliding,
            want,
            "boxes {ca:?} {ha:?} / {cb:?} {hb:?}"
        );
    }
    assert!(
        n_hit > 15 && n_hit < 285,
        "scene must mix hits and misses ({n_hit})"
    );
}

#[test]
fn gjk_sphere_box_agrees_with_point_to_box_distance() {
    let mut g = Lcg(3);
    let mut n_hit = 0;
    for _ in 0..300 {
        let (c, h) = (g.vec(-1.0, 1.0), g.vec(0.3, 1.2));
        let (p, r) = (g.vec(-3.0, 3.0), g.range(0.2, 1.4));
        let d = point_box_dist(p, c, h);
        if (d - r).abs() < 1e-3 {
            continue;
        }
        let want = d < r;
        n_hit += usize::from(want);
        let got = gjk(&Sphere::new(v3(p[0], p[1], p[2]), fx(r)), &aabb(c, h)).colliding;
        assert_eq!(got, want, "sphere {p:?} r {r} box {c:?} {h:?} dist {d}");
    }
    assert!(n_hit > 15 && n_hit < 285, "({n_hit})");
}

#[test]
fn gjk_capsule_pairs_agree_with_segment_distance() {
    let mut g = Lcg(4);
    let mut n_hit = 0;
    for _ in 0..300 {
        let (a1, b1) = (g.vec(-2.0, 2.0), g.vec(-2.0, 2.0));
        let (a2, b2) = (g.vec(-2.0, 2.0), g.vec(-2.0, 2.0));
        let (r1, r2) = (g.range(0.1, 0.8), g.range(0.1, 0.8));
        let d = seg_seg_dist(a1, b1, a2, b2);
        if (d - (r1 + r2)).abs() < 1e-3 {
            continue;
        }
        let want = d < r1 + r2;
        n_hit += usize::from(want);
        let ca = Capsule::new(v3(a1[0], a1[1], a1[2]), v3(b1[0], b1[1], b1[2]), fx(r1));
        let cb = Capsule::new(v3(a2[0], a2[1], a2[2]), v3(b2[0], b2[1], b2[2]), fx(r2));
        assert_eq!(
            gjk(&ca, &cb).colliding,
            want,
            "capsules dist {d} radii {}",
            r1 + r2
        );
    }
    assert!(n_hit > 15 && n_hit < 285, "({n_hit})");
}

#[test]
fn gjk_hull_cube_agrees_with_box_and_sphere_closed_forms() {
    let cube: Vec<Vec3Fix> = (0..8)
        .map(|i| {
            v3(
                if i & 1 == 0 { -1.0 } else { 1.0 },
                if i & 2 == 0 { -1.0 } else { 1.0 },
                if i & 4 == 0 { -1.0 } else { 1.0 },
            )
        })
        .collect();
    let hull = ConvexHull::new(cube);
    let mut g = Lcg(5);
    for _ in 0..200 {
        let (p, r) = (g.vec(-3.0, 3.0), g.range(0.2, 1.2));
        let d = point_box_dist(p, [0.0; 3], [1.0; 3]);
        if (d - r).abs() < 1e-3 {
            continue;
        }
        let got = gjk(&hull, &Sphere::new(v3(p[0], p[1], p[2]), fx(r))).colliding;
        assert_eq!(got, d < r, "hull vs sphere {p:?} r {r} dist {d}");
    }
}

#[test]
fn gjk_far_from_the_origin_still_matches_the_closed_form() {
    for base in [1.0e3, 1.0e5, 1.0e7] {
        let a = Sphere::new(v3(base, 0.0, 0.0), fx(1.0));
        let hit = Sphere::new(v3(base + 1.5, 0.3, -0.2), fx(1.0));
        let miss = Sphere::new(v3(base + 2.5, 0.3, -0.2), fx(1.0));
        assert!(gjk(&a, &hit).colliding, "overlapping spheres at {base}");
        assert!(!gjk(&a, &miss).colliding, "separated spheres at {base}");
    }
}

#[test]
fn gjk_small_shapes_are_not_reported_colliding_when_apart() {
    // 0.1 mm radius spheres (3D-printing scale, in metres) 1 mm apart.
    let a = Sphere::new(v3(0.0, 0.0, 0.0), fx(1.0e-4));
    let apart = Sphere::new(v3(1.0e-3, 0.0, 0.0), fx(1.0e-4));
    let near = Sphere::new(v3(1.5e-4, 0.0, 0.0), fx(1.0e-4));
    assert!(
        !gjk(&a, &apart).colliding,
        "spheres 0.8 mm apart reported colliding"
    );
    assert!(gjk(&a, &near).colliding, "overlapping micro spheres");
}

// ---------------------------------------------------------------------------
// contact / epa
// ---------------------------------------------------------------------------

#[test]
fn contact_sphere_pairs_match_closed_form_depth_normal_and_points() {
    let mut g = Lcg(6);
    let mut tested = 0;
    for _ in 0..200 {
        let (ra, rb) = (g.range(0.4, 1.5), g.range(0.4, 1.5));
        let ca = g.vec(-1.5, 1.5);
        let cb = g.vec(-1.5, 1.5);
        let dv = sub(ca, cb);
        let d = norm(dv);
        if d > ra + rb - 0.05 || d < 0.5 {
            continue;
        }
        tested += 1;
        let c = contact(
            &Sphere::new(v3(ca[0], ca[1], ca[2]), fx(ra)),
            &Sphere::new(v3(cb[0], cb[1], cb[2]), fx(rb)),
        )
        .expect("overlapping spheres must produce a contact");
        let depth = ra + rb - d;
        // documented: "a few parts in a thousand for spheres"
        assert!(
            (c.depth.to_f64() - depth).abs() < 0.01 * (ra + rb),
            "depth {} vs {depth}",
            c.depth.to_f64()
        );
        let n = arr(c.normal);
        let want_n = [dv[0] / d, dv[1] / d, dv[2] / d]; // B -> A
        assert!(dot(n, want_n) > 0.999, "normal {n:?} vs B->A {want_n:?}");
        // translating A by depth * normal separates the pair
        let moved = Sphere::new(
            v3(
                ca[0] + n[0] * c.depth.to_f64() * 1.02,
                ca[1] + n[1] * c.depth.to_f64() * 1.02,
                ca[2] + n[2] * c.depth.to_f64() * 1.02,
            ),
            fx(ra),
        );
        assert!(
            !gjk(&moved, &Sphere::new(v3(cb[0], cb[1], cb[2]), fx(rb))).colliding
                || c.depth.to_f64() < 1e-2
        );
        // point_a is the deepest point of A (towards B), point_b of B (towards A)
        let pa = arr(c.point_a);
        let pb = arr(c.point_b);
        assert!((norm(sub(pa, ca)) - ra).abs() < 1e-2, "point_a on sphere A");
        assert!((norm(sub(pb, cb)) - rb).abs() < 1e-2, "point_b on sphere B");
    }
    assert!(tested > 30, "({tested})");
}

#[test]
fn contact_box_pairs_match_the_minimum_translation_exactly() {
    let mut g = Lcg(7);
    let mut tested = 0;
    for _ in 0..300 {
        let (ha, hb) = (g.vec(0.4, 1.2), g.vec(0.4, 1.2));
        let (ca, cb) = (g.vec(-1.2, 1.2), g.vec(-1.2, 1.2));
        let mut depth = f64::INFINITY;
        let mut axis = 0;
        let mut ok = true;
        let mut gaps = [0.0; 3];
        for i in 0..3 {
            let pen = ha[i] + hb[i] - (ca[i] - cb[i]).abs();
            gaps[i] = pen;
            if pen <= 0.02 {
                ok = false;
            }
            if pen < depth {
                depth = pen;
                axis = i;
            }
        }
        // need a clear minimum axis (no ties within 1e-2)
        let mut sorted = gaps;
        sorted.sort_by(f64::total_cmp);
        if !ok || sorted[1] - sorted[0] < 1e-2 {
            continue;
        }
        tested += 1;
        let c = contact(&aabb(ca, ha), &aabb(cb, hb)).expect("overlapping boxes");
        assert!(
            (c.depth.to_f64() - depth).abs() < 1e-4,
            "depth {} vs {depth}",
            c.depth.to_f64()
        );
        let n = arr(c.normal);
        let sign = if ca[axis] - cb[axis] >= 0.0 {
            1.0
        } else {
            -1.0
        };
        for i in 0..3 {
            let want = if i == axis { sign } else { 0.0 };
            assert!(
                (n[i] - want).abs() < 1e-4,
                "normal {n:?} axis {axis} sign {sign}"
            );
        }
    }
    assert!(tested > 60, "({tested})");
}

#[test]
fn contact_sphere_box_matches_closed_form() {
    let c0 = [0.0, 0.0, 0.0];
    let h = [1.0, 0.8, 0.6];
    let b = aabb(c0, h);
    let mut g = Lcg(8);
    let mut tested = 0;
    for _ in 0..300 {
        let p = g.vec(-2.2, 2.2);
        let r = g.range(0.3, 1.0);
        let d = point_box_dist(p, c0, h);
        if d < 0.05 || d > r - 0.05 {
            continue; // outside-centre overlaps only
        }
        tested += 1;
        let hit = contact(&Sphere::new(v3(p[0], p[1], p[2]), fx(r)), &b).expect("overlap");
        assert!(
            (hit.depth.to_f64() - (r - d)).abs() < 0.01 * r,
            "depth {} vs {}",
            hit.depth.to_f64(),
            r - d
        );
        // normal = (sphere centre - closest box point)/d, B -> A
        let cp = [
            p[0].clamp(-h[0], h[0]),
            p[1].clamp(-h[1], h[1]),
            p[2].clamp(-h[2], h[2]),
        ];
        let want = sub(p, cp);
        let want = [want[0] / d, want[1] / d, want[2] / d];
        assert!(
            dot(arr(hit.normal), want) > 0.98,
            "normal {:?} vs {want:?}",
            arr(hit.normal)
        );
    }
    assert!(tested > 30, "({tested})");
}

#[test]
fn contact_is_none_for_separated_shapes_and_normal_flips_with_argument_order() {
    let a = aabb([0.0; 3], [1.0; 3]);
    let far = aabb([3.0, 0.0, 0.0], [1.0; 3]);
    assert!(contact(&a, &far).is_none());
    let near = aabb([1.5, 0.2, -0.1], [1.0; 3]);
    let ab = contact(&a, &near).expect("overlap");
    let ba = contact(&near, &a).expect("overlap");
    assert!((ab.depth.to_f64() - 0.5).abs() < 1e-6);
    assert!((ab.depth.to_f64() - ba.depth.to_f64()).abs() < 1e-6);
    // B -> A normal: A is at -x of near, so contact(a, near).normal = -x
    assert!(ab.normal.x.to_f64() < -0.999);
    assert!(ba.normal.x.to_f64() > 0.999);
}

#[test]
// AUD-A-S3W3-016
fn contact_concentric_sphere_pair_reports_the_radius_sum() {
    let a = Sphere::new(v3(0.0, 0.0, 0.0), fx(1.0));
    let b = Sphere::new(v3(0.0, 0.0, 0.0), fx(0.5));
    let c = contact(&a, &b).expect("concentric spheres overlap");
    assert!(
        (c.depth.to_f64() - 1.5).abs() < 0.005 * 1.5,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(
        (norm(arr(c.normal)) - 1.0).abs() < 1e-6,
        "normal must be unit"
    );
}

#[test]
fn epa_rejects_a_simplex_with_fewer_than_four_points() {
    let a = Sphere::new(v3(0.0, 0.0, 0.0), fx(1.0));
    let b = Sphere::new(v3(0.5, 0.0, 0.0), fx(1.0));
    let tri = [v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)];
    assert!(epa(&a, &b, &tri).is_none());
    assert!(epa(&a, &b, &[]).is_none());
}

#[test]
fn epa_expands_a_hand_built_tetrahedron_around_the_origin_to_the_exact_box_depth() {
    // A = [-1,1]^3 at origin, B = [-1,1]^3 shifted +1.5 in x: A - B spans
    // x in [-3.5, 0.5], y,z in [-2, 2]; the tetrahedron below uses points of
    // A - B and contains the origin; the nearest face is x = 0.5 (depth 0.5).
    let a = aabb([0.0; 3], [1.0; 3]);
    let b = aabb([1.5, 0.0, 0.0], [1.0; 3]);
    let tet = [
        v3(0.5, 2.0, 2.0),
        v3(0.5, -2.0, 2.0),
        v3(0.5, 0.0, -2.0),
        v3(-3.5, 0.0, 0.0),
    ];
    let c = epa(&a, &b, &tet).expect("contact");
    assert!(
        (c.depth.to_f64() - 0.5).abs() < 1e-6,
        "depth {}",
        c.depth.to_f64()
    );
    assert!(c.normal.x.to_f64() < -0.999);
}

// ---------------------------------------------------------------------------
// support functions
// ---------------------------------------------------------------------------

fn dirs(g: &mut Lcg, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let d = g.vec(-1.0, 1.0);
            let l = norm(d);
            if l < 0.2 {
                [1.0, 0.0, 0.0]
            } else {
                // random scale in [0.5, 8] so non-unit directions are covered
                let s = g.range(0.5, 8.0) / l;
                [d[0] * s, d[1] * s, d[2] * s]
            }
        })
        .collect()
}

#[test]
fn sphere_support_attains_c_dot_d_plus_r_norm_d() {
    let c = [1.0, -2.0, 0.5];
    let r = 1.7;
    let s = Sphere::new(v3(c[0], c[1], c[2]), fx(r));
    let mut g = Lcg(9);
    for d in dirs(&mut g, 200) {
        let p = arr(s.support(v3(d[0], d[1], d[2])));
        let want = dot(c, d) + r * norm(d);
        assert!(
            (dot(p, d) - want).abs() < 1e-7,
            "dot {} vs {want}",
            dot(p, d)
        );
        assert!(
            (norm(sub(p, c)) - r).abs() < 1e-7,
            "support point must lie on the sphere"
        );
    }
}

#[test]
fn capsule_support_attains_the_endpoint_maximum_plus_r_norm_d() {
    let (a, b, r) = ([-1.0, 0.5, 2.0], [2.0, -1.0, 0.0], 0.6);
    let cap = Capsule::new(v3(a[0], a[1], a[2]), v3(b[0], b[1], b[2]), fx(r));
    let mut g = Lcg(10);
    for d in dirs(&mut g, 200) {
        let p = arr(cap.support(v3(d[0], d[1], d[2])));
        let want = dot(a, d).max(dot(b, d)) + r * norm(d);
        assert!(
            (dot(p, d) - want).abs() < 1e-7,
            "dot {} vs {want} (d {d:?})",
            dot(p, d)
        );
    }
}

#[test]
fn hull_support_attains_the_maximum_vertex_dot() {
    let verts = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 2.0],
    ];
    let hull = ConvexHull::new(verts.iter().map(|v| v3(v[0], v[1], v[2])).collect());
    let mut g = Lcg(11);
    for d in dirs(&mut g, 200) {
        let p = arr(hull.support(v3(d[0], d[1], d[2])));
        let want = verts
            .iter()
            .map(|v| dot(*v, d))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((dot(p, d) - want).abs() < 1e-9);
        assert!(
            verts.iter().any(|v| norm(sub(*v, p)) < 1e-9),
            "support must be a vertex"
        );
    }
}

#[test]
fn aabb_support_is_the_corner_chosen_per_axis_by_direction_sign() {
    let b = AABB::new(v3(-1.0, -2.0, -3.0), v3(4.0, 5.0, 6.0));
    let mut g = Lcg(12);
    for d in dirs(&mut g, 100) {
        let p = arr(b.support(v3(d[0], d[1], d[2])));
        let want = [
            if d[0] >= 0.0 { 4.0 } else { -1.0 },
            if d[1] >= 0.0 { 5.0 } else { -2.0 },
            if d[2] >= 0.0 { 6.0 } else { -3.0 },
        ];
        assert_eq!(p, want);
    }
    // a zero component picks the max face (>= 0)
    let p = arr(b.support(v3(0.0, -1.0, 0.0)));
    assert_eq!(p, [4.0, -2.0, 6.0]);
}

#[test]
fn scaled_shape_support_scales_the_inner_extreme_point_for_positive_scale() {
    let inner = Sphere::new(v3(1.0, 2.0, 3.0), fx(0.5));
    let sc = ScaledShape::new(inner, fx(3.0));
    let mut g = Lcg(13);
    for d in dirs(&mut g, 100) {
        let p = arr(sc.support(v3(d[0], d[1], d[2])));
        let want = 3.0 * (dot([1.0, 2.0, 3.0], d) + 0.5 * norm(d));
        assert!((dot(p, d) - want).abs() < 1e-6);
    }
}

#[test]
fn scaled_shape_with_negative_scale_is_still_a_support_function() {
    // Trait contract: "the point on the shape furthest in the given
    // direction". The shape scaled by -2 about the origin is the inner box
    // mirrored and doubled, so its support in d is -2 * inner.support(-d).
    let inner = AABB::new(v3(1.0, 1.0, 1.0), v3(2.0, 3.0, 4.0));
    let sc = ScaledShape::new(inner, fx(-2.0));
    let d = [1.0, 0.5, -0.25];
    let p = arr(sc.support(v3(d[0], d[1], d[2])));
    // mirrored box spans x in [-4,-2], y in [-6,-2], z in [-8,-2]: max <p,d>
    let want = dot([-2.0, -2.0, -8.0], d);
    assert!(
        (dot(p, d) - want).abs() < 1e-9,
        "support dot {} vs mirrored-box maximum {want}",
        dot(p, d)
    );
}

#[test]
fn sphere_and_capsule_support_handle_a_tiny_direction() {
    // The trait takes any non-zero direction. A direction of 1e-11 is far
    // below 2.3e-10 where `normalize` returns zero, so the support collapses
    // to the centre instead of centre + r * d/|d|.
    let s = Sphere::new(v3(0.0, 0.0, 0.0), fx(1.0));
    let p = arr(s.support(v3(1.0e-11, 0.0, 0.0)));
    assert!(
        (p[0] - 1.0).abs() < 1e-6,
        "sphere support for a tiny +x direction is {p:?}"
    );
    let cap = Capsule::new(v3(0.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), fx(0.5));
    let q = arr(cap.support(v3(1.0e-11, 0.0, 0.0)));
    assert!(
        (q[0] - 0.5).abs() < 1e-6,
        "capsule support for a tiny +x direction is {q:?}"
    );
}

#[test]
#[should_panic(expected = "at least one vertex")]
fn empty_hull_is_rejected() {
    let _ = ConvexHull::new(Vec::new());
}

// ---------------------------------------------------------------------------
// AABB::from_metric_ball
// ---------------------------------------------------------------------------

fn metric_norm(w: (f64, f64, f64), v: [f64; 3]) -> f64 {
    let l1 = v[0].abs() + v[1].abs() + v[2].abs();
    let l2 = norm(v);
    let li = v[0].abs().max(v[1].abs()).max(v[2].abs());
    w.0 * l1 + w.1 * l2 + w.2 * li
}

#[test]
fn from_metric_ball_is_the_tight_box_of_the_ball() {
    let centre = v3(1.0, -2.0, 0.5);
    let r = 1.5;
    let cases = [
        (MetricWeights::L1, (1.0, 0.0, 0.0)),
        (MetricWeights::L2, (0.0, 1.0, 0.0)),
        (MetricWeights::LINF, (0.0, 0.0, 1.0)),
        (
            MetricWeights::new(fx(1.0), fx(1.0), fx(0.0)).unwrap(),
            (1.0, 1.0, 0.0),
        ),
        (
            MetricWeights::new(fx(0.5), fx(0.25), fx(2.0)).unwrap(),
            (0.5, 0.25, 2.0),
        ),
    ];
    for (m, w) in cases {
        let b = AABB::from_metric_ball(centre, fx(r), m);
        let half = r / (w.0 + w.1 + w.2);
        let (mn, mx) = (arr(b.min), arr(b.max));
        let c = [1.0, -2.0, 0.5];
        for i in 0..3 {
            assert!(
                (mx[i] - (c[i] + half)).abs() < 1e-9,
                "max[{i}] {} vs {}",
                mx[i],
                c[i] + half
            );
            assert!((mn[i] - (c[i] - half)).abs() < 1e-9, "min[{i}]");
        }
        // every boundary point g(x) = r lies inside the box, and the axis
        // points touch it (tightness)
        let mut g = Lcg(14);
        let mut touched = 0.0f64;
        for _ in 0..400 {
            let d = g.vec(-1.0, 1.0);
            let gd = metric_norm(w, d);
            if gd < 1e-6 {
                continue;
            }
            let t = r / gd;
            let p = [d[0] * t, d[1] * t, d[2] * t];
            for pi in p {
                assert!(
                    pi.abs() <= half + 1e-9,
                    "boundary point {p:?} outside box half {half}"
                );
                touched = touched.max(pi.abs());
            }
        }
        assert!(
            touched > 0.9 * half,
            "box not tight: farthest boundary point {touched} of {half}"
        );
        let e = [1.0, 0.0, 0.0];
        assert!(
            (metric_norm(w, e) * half - r).abs() < 1e-9,
            "axis point sits on the ball surface"
        );
    }
}

#[test]
fn from_metric_ball_euclidean_equals_from_center_half_bit_for_bit() {
    let c = v3(0.3, 0.7, -1.1);
    let r = fx(0.9);
    let a = AABB::from_metric_ball(c, r, MetricWeights::L2);
    let b = AABB::from_center_half(c, Vec3Fix::new(r, r, r));
    assert_eq!(a, b);
}

// ---------------------------------------------------------------------------
// Additional pins: AABB::intersects faces, EPA independence from the start tetrahedron
// ---------------------------------------------------------------------------

#[test]
fn aabb_intersects_is_inclusive_on_every_face_and_exclusive_one_ulp_beyond() {
    let a = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let ulp = Fix128::from_raw(0, 1);
    for axis in 0..3 {
        for side in [-1.0, 1.0] {
            // box touching `a` on this face
            let mut lo = [0.0, 0.0, 0.0];
            let mut hi = [1.0, 1.0, 1.0];
            if side > 0.0 {
                lo[axis] = 1.0;
                hi[axis] = 2.0;
            } else {
                lo[axis] = -1.0;
                hi[axis] = 0.0;
            }
            let b = AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));
            assert!(
                a.intersects(&b) && b.intersects(&a),
                "touching on axis {axis} side {side}"
            );
            // move b one ulp further away along that axis
            let mut bmin = b.min;
            let mut bmax = b.max;
            let shift = |v: &mut Vec3Fix| match axis {
                0 => v.x = if side > 0.0 { v.x + ulp } else { v.x - ulp },
                1 => v.y = if side > 0.0 { v.y + ulp } else { v.y - ulp },
                _ => v.z = if side > 0.0 { v.z + ulp } else { v.z - ulp },
            };
            shift(&mut bmin);
            shift(&mut bmax);
            let c = AABB::new(bmin, bmax);
            assert!(
                !a.intersects(&c) && !c.intersects(&a),
                "one ulp apart on axis {axis} side {side}"
            );
        }
    }
}

#[test]
fn epa_depth_and_normal_do_not_depend_on_the_enclosing_start_tetrahedron() {
    // A = [-1,1]^3, B = [-1,1]^3 shifted (0.3, 0.1, -0.2): A - B spans
    // x in [-1.3, 2.3], y in [-1.1, 2.1], z in [-0.8, 2.2]... take points of
    // A - B (corners shifted by the box offset) and keep tetrahedra that
    // contain the origin. Exact minimum translation = 2 - 0.3 = 1.7 (x axis);
    // A sits at -0.3 in x relative to B, so the B -> A normal is -x.
    let a = aabb([0.0; 3], [1.0; 3]);
    let b = aabb([0.3, 0.1, -0.2], [1.0; 3]);
    let mut g = Lcg(15);
    let mut used = 0;
    for _ in 0..20000 {
        if used >= 24 {
            break;
        }
        // random points of A - B: a_corner - b_corner with random corner signs
        let mut pts = Vec::new();
        for _ in 0..4 {
            let sa = [g.range(-1.0, 1.0), g.range(-1.0, 1.0), g.range(-1.0, 1.0)];
            let sb = [g.range(-1.0, 1.0), g.range(-1.0, 1.0), g.range(-1.0, 1.0)];
            // pick extreme corners half of the time so the tetra is large
            let q = |s: f64| if s > 0.0 { 1.0 } else { -1.0 };
            let (pa, pb) = if g.next() < 0.7 {
                (
                    [q(sa[0]), q(sa[1]), q(sa[2])],
                    [0.3 + q(sb[0]), 0.1 + q(sb[1]), -0.2 + q(sb[2])],
                )
            } else {
                (sa, [0.3 + sb[0], 0.1 + sb[1], -0.2 + sb[2]])
            };
            pts.push(sub(pa, pb));
        }
        // every other candidate: one vertex close to the origin (centroid far
        // from the origin relative to the tetrahedron's thin end)
        if used % 2 == 0 {
            pts[0] = [
                g.range(-0.15, 0.15),
                g.range(-0.15, 0.15),
                g.range(-0.15, 0.15),
            ];
        }
        if !tetra_contains_origin_f64(&pts) {
            continue;
        }
        used += 1;
        let tet: Vec<Vec3Fix> = pts.iter().map(|p| v3(p[0], p[1], p[2])).collect();
        let c = epa(&a, &b, &tet).expect("contact");
        assert!(
            (c.depth.to_f64() - 1.7).abs() < 1e-4,
            "depth {} from tetra {pts:?}",
            c.depth.to_f64()
        );
        let n = arr(c.normal);
        // A is at -0.3 in x relative to B: translating A by depth * normal (B -> A) along -x separates
        assert!(
            n[0] < -0.999 && n[1].abs() < 1e-3 && n[2].abs() < 1e-3,
            "normal {n:?}"
        );
    }
    assert!(used >= 16, "only {used} enclosing tetrahedra generated");
}

fn tetra_contains_origin_f64(p: &[[f64; 3]]) -> bool {
    // barycentric via signed volumes (origin inside iff all five volumes share a sign)
    fn vol(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
        let (u, v, w) = (sub(b, a), sub(c, a), sub(d, a));
        dot(
            u,
            [
                v[1] * w[2] - v[2] * w[1],
                v[2] * w[0] - v[0] * w[2],
                v[0] * w[1] - v[1] * w[0],
            ],
        )
    }
    let o = [0.0; 3];
    let v0 = vol(p[0], p[1], p[2], p[3]);
    if v0.abs() < 1e-3 {
        return false;
    }
    let parts = [
        vol(o, p[1], p[2], p[3]),
        vol(p[0], o, p[2], p[3]),
        vol(p[0], p[1], o, p[3]),
        vol(p[0], p[1], p[2], o),
    ];
    parts.iter().all(|x| x * v0 > 1e-4)
}
