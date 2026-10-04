//! Audit oracles for `collider::epa`: signed contact normal (from B to A, the
//! documented convention), exact depth and contact points for overlapping boxes.
//!
//! Expected values: for `A = [-1, 1]^3` and `B` overlapping A by `delta` along one
//! axis direction `s e_k` (and by 2 on the other axes), the minimum translation of A
//! is `delta` along `-s e_k`, so `normal = -s e_k`, `depth = delta`; the deepest
//! point of A lies on its `s e_k` face and that of B on its `-s e_k` face.

use alice_physics::collider::{epa, gjk, Sphere, Support, AABB};
use alice_physics::math::{Fix128, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn unit_box() -> AABB {
    AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0))
}

/// B shifted along `s e_k` so that it overlaps A by `delta` on axis k (lateral
/// extents `[-1.5, 1.5]` cover A, so the other axes overlap by 2).
fn shifted(k: usize, s: f64, delta: f64) -> AABB {
    let mut lo = [-1.5, -1.5, -1.5];
    let mut hi = [1.5, 1.5, 1.5];
    if s > 0.0 {
        lo[k] = 1.0 - delta;
        hi[k] = 3.0 - delta;
    } else {
        lo[k] = -3.0 + delta;
        hi[k] = -1.0 + delta;
    }
    AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]))
}

/// Minkowski-difference tetrahedron enclosing the origin, built from the public
/// support functions in four fixed directions (independent of GJK).
fn tetra<A: Support, B: Support>(a: &A, b: &B) -> [Vec3Fix; 4] {
    let dirs = [
        v3(1.0, 1.0, 1.0),
        v3(-1.0, -1.0, 1.0),
        v3(-1.0, 1.0, -1.0),
        v3(1.0, -1.0, -1.0),
    ];
    let mut out = [Vec3Fix::ZERO; 4];
    for (i, d) in dirs.iter().enumerate() {
        out[i] = a.support(*d) - b.support(-*d);
    }
    out
}

fn translated(b: &AABB, t: Vec3Fix) -> AABB {
    AABB::new(b.min + t, b.max + t)
}

/// All 6 directions x 3 depths: signed normal `-s e_k` exactly, depth `delta`
/// exactly, `point_a` on A's `s e_k` face and `point_b` on B's `-s e_k` face.
#[test]
fn epa_box_normal_points_from_b_to_a_with_exact_depth_and_face_points() {
    let a = unit_box();
    for k in 0..3 {
        for s in [1.0, -1.0] {
            for delta in [0.5, 0.25, 0.125] {
                let b = shifted(k, s, delta);
                let c = epa(&a, &b, &tetra(&a, &b)).expect("overlapping boxes");
                let mut want_n = [0.0; 3];
                want_n[k] = -s;
                let ctx = format!("axis {k} side {s} delta {delta}");
                assert_eq!(arr(c.normal), want_n, "normal {ctx}");
                assert_eq!(c.depth, fx(delta), "depth {ctx}");
                assert_eq!(arr(c.point_a)[k], s, "point_a on A's face {ctx}");
                assert_eq!(
                    arr(c.point_b)[k],
                    s * (1.0 - delta),
                    "point_b on B's face {ctx}"
                );
            }
        }
    }
}

/// The convention is directional: swapping the arguments negates the normal and
/// keeps the depth.
#[test]
fn epa_swapping_the_shapes_negates_the_normal() {
    let a = unit_box();
    for k in 0..3 {
        for s in [1.0, -1.0] {
            let b = shifted(k, s, 0.25);
            let ab = epa(&a, &b, &tetra(&a, &b)).expect("overlap");
            let ba = epa(&b, &a, &tetra(&b, &a)).expect("overlap");
            assert_eq!(ba.normal, -ab.normal, "axis {k} side {s}");
            assert_eq!(ba.depth, ab.depth, "axis {k} side {s}");
        }
    }
}

/// The documented meaning of the sign: moving A by `depth * normal` (plus a margin)
/// separates the shapes, while moving it the same amount the other way does not.
#[test]
fn epa_translating_a_by_depth_times_normal_separates_the_shapes() {
    let a = unit_box();
    let margin = fx(1.0 / 1024.0);
    for k in 0..3 {
        for s in [1.0, -1.0] {
            let b = shifted(k, s, 0.375);
            let c = epa(&a, &b, &tetra(&a, &b)).expect("overlap");
            let t = c.normal * (c.depth + margin);
            assert!(
                !gjk(&translated(&a, t), &b).colliding,
                "axis {k} side {s}: still colliding after the push"
            );
            assert!(
                gjk(&translated(&a, -t), &b).colliding,
                "axis {k} side {s}: the opposite push separated them"
            );
        }
    }
}

/// Unit spheres with centre offset `d e` along each axis direction: the B to A
/// normal is `-e`, depth `2 - d`, `point_a = e` and `point_b = (d - 1) e`. EPA
/// approximates a curved Minkowski sum, so the tolerance is 1% (documented as a
/// few parts in a thousand); the sign of the normal is exact information.
#[test]
fn epa_sphere_pair_normal_is_signed_and_depth_matches_the_closed_form() {
    for k in 0..3 {
        for s in [1.0, -1.0] {
            for d in [1.25, 1.5, 1.75] {
                let mut e = [0.0; 3];
                e[k] = s;
                let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
                let b = Sphere::new(v3(d * e[0], d * e[1], d * e[2]), Fix128::ONE);
                let c = epa(&a, &b, &tetra(&a, &b)).expect("overlapping spheres");
                let n = arr(c.normal);
                let ctx = format!("axis {k} side {s} d {d}");
                for i in 0..3 {
                    assert!((n[i] + e[i]).abs() < 1e-2, "normal {n:?} {ctx}");
                }
                let depth = c.depth.to_f64();
                assert!(
                    (depth - (2.0 - d)).abs() < 1e-2 * (2.0 - d),
                    "depth {depth} want {} {ctx}",
                    2.0 - d
                );
                let pa = arr(c.point_a);
                let pb = arr(c.point_b);
                for i in 0..3 {
                    assert!((pa[i] - e[i]).abs() < 2e-2, "point_a {pa:?} {ctx}");
                    assert!(
                        (pb[i] - (d - 1.0) * e[i]).abs() < 2e-2,
                        "point_b {pb:?} {ctx}"
                    );
                }
            }
        }
    }
}

/// Boxes of unequal size with B offset laterally: the minimum translation is still
/// along the axis of least overlap, exact, signed from B to A.
#[test]
fn epa_offset_unequal_boxes_report_the_least_overlap_axis_signed() {
    // A = [-1, 2] x [-0.5, 0.5] x [-2, 1]; B overlaps it by 0.25 on +y and by more on x, z
    let a = AABB::new(v3(-1.0, -0.5, -2.0), v3(2.0, 0.5, 1.0));
    let cases = [
        (
            v3(-0.5, 0.25, -1.5),
            v3(3.0, 2.0, 0.5),
            [0.0, -1.0, 0.0],
            0.25,
        ),
        (
            v3(1.5, -3.0, -1.0),
            v3(4.0, 0.25, 2.0),
            [-1.0, 0.0, 0.0],
            0.5,
        ),
        (
            v3(-3.0, -2.0, 0.875),
            v3(0.5, 0.25, 4.0),
            [0.0, 0.0, -1.0],
            0.125,
        ),
        (
            v3(-0.75, -0.75, -4.0),
            v3(1.75, 0.75, -1.625),
            [0.0, 0.0, 1.0],
            0.375,
        ),
    ];
    for (lo, hi, n, depth) in cases {
        let b = AABB::new(lo, hi);
        let c = epa(&a, &b, &tetra(&a, &b)).expect("overlap");
        assert_eq!(arr(c.normal), n, "box {:?}..{:?}", arr(lo), arr(hi));
        assert_eq!(c.depth, fx(depth), "box {:?}..{:?}", arr(lo), arr(hi));
    }
}
