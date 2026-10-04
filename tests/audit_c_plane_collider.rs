//! Audit oracles for `plane_collider`: depth, normal and both contact points of
//! `intersect_sphere` and `intersect_aabb` against the plane distance formula.
//!
//! Expected values: signed distance `n.c - d`; sphere depth `r - |dist|`, normal
//! `sign(dist) n`, `point_a = c - normal r`, `point_b = c - n dist`; box depth =
//! minus the smallest signed corner distance (brute force over the 8 corners).
//! Axis-aligned cases use dyadic inputs, so the expected values are exact.

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;

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

fn axes() -> Vec<[f64; 3]> {
    vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
}

/// The two inline scenarios (sphere r = 1 at y = +0.5 and y = -0.5 on the y = 0
/// plane) with every field of the result.
#[test]
fn sphere_on_the_ground_plane_reports_exact_depth_normal_and_points() {
    let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
    let front = plane.intersect_sphere(v3(0.0, 0.5, 0.0), Fix128::ONE);
    assert!(front.colliding);
    assert_eq!(front.depth, fx(0.5));
    assert_eq!(front.normal, Vec3Fix::UNIT_Y);
    assert_eq!(front.point_a, v3(0.0, -0.5, 0.0));
    assert_eq!(front.point_b, Vec3Fix::ZERO);

    let back = plane.intersect_sphere(v3(0.0, -0.5, 0.0), Fix128::ONE);
    assert!(back.colliding);
    assert_eq!(back.depth, fx(0.5));
    assert_eq!(back.normal, v3(0.0, -1.0, 0.0));
    assert_eq!(back.point_a, v3(0.0, 0.5, 0.0));
    assert_eq!(back.point_b, Vec3Fix::ZERO);
}

/// Sweep: 6 axis normals x offsets x sphere centres x radii on a dyadic lattice,
/// every field equal to the closed form (or NONE when `|dist| >= r`).
#[test]
fn sphere_sweep_matches_the_plane_distance_formula_exactly() {
    let mut hits = 0;
    let mut misses = 0;
    for n in axes() {
        for d in [-1.5, 0.0, 0.75, 2.0] {
            let plane = PlaneCollider::new(v3(n[0], n[1], n[2]), fx(d));
            assert_eq!(arr(plane.normal), n, "normal kept");
            for cx in [-2.0, -0.25, 1.0] {
                for cy in [-1.75, 0.5, 2.25] {
                    for cz in [-0.5, 0.0, 1.5] {
                        for r in [0.5, 1.25, 3.0] {
                            let c = [cx, cy, cz];
                            let dist = dot(n, c) - d;
                            let got = plane.intersect_sphere(v3(cx, cy, cz), fx(r));
                            if dist.abs() >= r {
                                assert!(!got.colliding, "n {n:?} d {d} c {c:?} r {r}");
                                misses += 1;
                                continue;
                            }
                            hits += 1;
                            let s = if dist >= 0.0 { 1.0 } else { -1.0 };
                            let normal = [s * n[0], s * n[1], s * n[2]];
                            let pa = [cx - normal[0] * r, cy - normal[1] * r, cz - normal[2] * r];
                            let pb = [cx - n[0] * dist, cy - n[1] * dist, cz - n[2] * dist];
                            let ctx = format!("n {n:?} d {d} c {c:?} r {r}");
                            assert!(got.colliding, "{ctx}");
                            assert_eq!(got.depth.to_f64(), r - dist.abs(), "depth {ctx}");
                            assert_eq!(arr(got.normal), normal, "normal {ctx}");
                            assert_eq!(arr(got.point_a), pa, "point_a {ctx}");
                            assert_eq!(arr(got.point_b), pb, "point_b {ctx}");
                        }
                    }
                }
            }
        }
    }
    assert!(hits > 100 && misses > 100, "hits {hits} misses {misses}");
}

/// The inline straddling box `[-1, 1]^3` on `y = 0`: depth 1, normal +y, the
/// deepest corner as `point_a` and its projection as `point_b`.
#[test]
fn straddling_box_on_the_ground_plane_reports_exact_values() {
    let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
    let r = plane.intersect_aabb(&aabb);
    assert!(r.colliding);
    assert_eq!(r.depth, Fix128::ONE);
    assert_eq!(r.normal, Vec3Fix::UNIT_Y);
    assert_eq!(r.point_a.y, Fix128::NEG_ONE);
    assert_eq!(r.point_b.y, Fix128::ZERO);
    assert_eq!(r.point_b.x, r.point_a.x);
    assert_eq!(r.point_b.z, r.point_a.z);
}

/// Oblique normals (unit after `new`), boxes straddling or in front of the plane:
/// depth = `-min corner distance` (brute force over 8 corners), normal = plane
/// normal, `point_a` is a corner at that minimum and `point_b` its projection.
#[test]
fn box_sweep_depth_and_points_match_the_brute_force_corner_distance() {
    let normals = [
        [1.0, 2.0, 2.0],
        [-2.0, 1.0, 2.0],
        [2.0, -2.0, -1.0],
        [0.0, 3.0, -4.0],
        [-4.0, 0.0, -3.0],
    ];
    let mut hits = 0;
    for raw in normals {
        let len = dot(raw, raw).sqrt();
        let n = [raw[0] / len, raw[1] / len, raw[2] / len];
        for d in [-0.5, 0.0, 1.0] {
            let plane = PlaneCollider::new(v3(raw[0], raw[1], raw[2]), fx(d));
            for (lo, hi) in [
                ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
                ([0.25, -0.5, 0.0], [1.5, 2.0, 0.75]),
                ([-2.0, 0.5, -1.0], [-0.5, 1.0, 3.0]),
                ([3.0, 3.0, 3.0], [4.0, 4.0, 4.0]),
            ] {
                let aabb = AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));
                let mut dmin = f64::INFINITY;
                let mut dmax = f64::NEG_INFINITY;
                for k in 0..8 {
                    let c = [
                        if k & 1 == 0 { lo[0] } else { hi[0] },
                        if k & 2 == 0 { lo[1] } else { hi[1] },
                        if k & 4 == 0 { lo[2] } else { hi[2] },
                    ];
                    let dist = dot(n, c) - d;
                    dmin = dmin.min(dist);
                    dmax = dmax.max(dist);
                }
                let got = plane.intersect_aabb(&aabb);
                let ctx = format!("n {raw:?} d {d} box {lo:?}..{hi:?}");
                if dmin >= 1e-12 {
                    assert!(!got.colliding, "{ctx}");
                    continue;
                }
                assert!(got.colliding, "{ctx}");
                hits += 1;
                assert!((got.depth.to_f64() + dmin).abs() < 1e-12, "depth {ctx}");
                let gn = arr(got.normal);
                for k in 0..3 {
                    assert!((gn[k] - n[k]).abs() < 1e-15, "normal {ctx}");
                }
                let pa = arr(got.point_a);
                for k in 0..3 {
                    assert!(pa[k] == lo[k] || pa[k] == hi[k], "point_a corner {ctx}");
                }
                assert!(
                    (dot(n, pa) - d - dmin).abs() < 1e-12,
                    "point_a deepest {ctx}"
                );
                if dmax > 1e-9 {
                    // partial overlap (clear of the touching boundary): point_b is the
                    // projection of point_a
                    let pb = arr(got.point_b);
                    for k in 0..3 {
                        assert!(
                            (pb[k] - (pa[k] - n[k] * dmin)).abs() < 1e-12,
                            "point_b {ctx}"
                        );
                    }
                }
            }
        }
    }
    assert!(hits >= 15, "hits {hits}");
}
