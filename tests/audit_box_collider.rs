//! Audit oracles for `box_collider::OrientedBox`.
//!
//! Expectations come from textbook closed forms (box volume, surface area,
//! solid-box inertia) and from brute force over the eight corners computed with
//! an independent f64 Rodrigues rotation. Nothing is obtained by calling the
//! box's own methods to produce the expected value.

#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::Support;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// Rodrigues rotation of `v` about unit `axis` by `ang`, in f64.
fn rodrigues(axis: [f64; 3], ang: f64, v: [f64; 3]) -> [f64; 3] {
    let (s, c) = (ang.sin(), ang.cos());
    let k = axis;
    let kxv = [
        k[1] * v[2] - k[2] * v[1],
        k[2] * v[0] - k[0] * v[2],
        k[0] * v[1] - k[1] * v[0],
    ];
    let kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    [
        v[0] * c + kxv[0] * s + k[0] * kd * (1.0 - c),
        v[1] * c + kxv[1] * s + k[1] * kd * (1.0 - c),
        v[2] * c + kxv[2] * s + k[2] * kd * (1.0 - c),
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

struct Case {
    center: [f64; 3],
    half: [f64; 3],
    axis: [f64; 3],
    ang: f64,
}

fn cases() -> Vec<Case> {
    let axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, -2.0, 3.0],
        [-0.3, 0.8, 0.5],
    ];
    let angs = [0.0, 0.3, std::f64::consts::FRAC_PI_4, 1.2, 2.5, -1.1, 3.0];
    let halves = [[1.0, 1.0, 1.0], [2.0, 0.5, 1.5], [0.25, 3.0, 0.75]];
    let centers = [[0.0, 0.0, 0.0], [3.0, -2.0, 5.0], [-10.5, 7.25, 0.125]];
    let mut out = Vec::new();
    for (i, ax) in axes.iter().enumerate() {
        for (j, &ang) in angs.iter().enumerate() {
            out.push(Case {
                center: centers[(i + j) % 3],
                half: halves[(i * 3 + j) % 3],
                axis: unit(*ax),
                ang,
            });
        }
    }
    out
}

fn build(c: &Case) -> OrientedBox {
    let rot = QuatFix::from_axis_angle(v3(c.axis[0], c.axis[1], c.axis[2]), fx(c.ang));
    OrientedBox::new(
        v3(c.center[0], c.center[1], c.center[2]),
        v3(c.half[0], c.half[1], c.half[2]),
        rot,
    )
}

/// Reference corner `index` (bit0 -> x, bit1 -> y, bit2 -> z; set bit = negative).
fn ref_corner(c: &Case, index: usize) -> [f64; 3] {
    let s = |bit: usize, h: f64| if index & bit == 0 { h } else { -h };
    let local = [s(1, c.half[0]), s(2, c.half[1]), s(4, c.half[2])];
    let r = rodrigues(c.axis, c.ang, local);
    [c.center[0] + r[0], c.center[1] + r[1], c.center[2] + r[2]]
}

const TOL: f64 = 1e-9;

#[test]
fn corner_matches_independent_rotation_and_bit_convention() {
    for c in cases() {
        let b = build(&c);
        for i in 0..8 {
            let got = arr(b.corner(i));
            let want = ref_corner(&c, i);
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < TOL,
                    "corner {i} axis {k}: {} vs {}",
                    got[k],
                    want[k]
                );
            }
        }
    }
}

#[test]
fn corners_equals_corner_per_index() {
    let c = &cases()[17];
    let b = build(c);
    let all = b.corners();
    for (i, p) in all.iter().enumerate() {
        assert_eq!(*p, b.corner(i), "corners()[{i}]");
    }
}

#[test]
fn aabb_is_tight_hull_of_the_eight_corners() {
    for c in cases() {
        let b = build(&c);
        let bb = b.aabb();
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for i in 0..8 {
            let p = ref_corner(&c, i);
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let (mn, mx) = (arr(bb.min), arr(bb.max));
        for k in 0..3 {
            assert!(
                (mn[k] - lo[k]).abs() < TOL,
                "min axis {k}: {} vs {}",
                mn[k],
                lo[k]
            );
            assert!(
                (mx[k] - hi[k]).abs() < TOL,
                "max axis {k}: {} vs {}",
                mx[k],
                hi[k]
            );
        }
    }
}

#[test]
fn aabb_axis_aligned_is_center_pm_half() {
    let b = OrientedBox::axis_aligned(v3(1.0, -2.0, 3.0), v3(0.5, 2.0, 4.0));
    let bb = b.aabb();
    assert_eq!(arr(bb.min), [0.5, -4.0, -1.0]);
    assert_eq!(arr(bb.max), [1.5, 0.0, 7.0]);
}

#[test]
fn axis_aligned_constructor_is_identity_rotation() {
    let a = OrientedBox::axis_aligned(v3(1.0, 2.0, 3.0), v3(1.0, 1.0, 1.0));
    let n = OrientedBox::new(v3(1.0, 2.0, 3.0), v3(1.0, 1.0, 1.0), QuatFix::IDENTITY);
    assert_eq!(a, n);
}

#[test]
fn volume_is_product_of_full_edges() {
    for c in cases() {
        let b = build(&c);
        let want = 8.0 * c.half[0] * c.half[1] * c.half[2];
        assert!((b.volume().to_f64() - want).abs() < 1e-9 * want.max(1.0));
    }
}

#[test]
fn surface_area_is_two_sum_of_face_products() {
    for c in cases() {
        let b = build(&c);
        let (lx, ly, lz) = (2.0 * c.half[0], 2.0 * c.half[1], 2.0 * c.half[2]);
        let want = 2.0 * (lx * ly + ly * lz + lz * lx);
        assert!((b.surface_area().to_f64() - want).abs() < 1e-9 * want);
    }
    // unit cube: 6
    let cube = OrientedBox::axis_aligned(Vec3Fix::ZERO, v3(0.5, 0.5, 0.5));
    assert!((cube.surface_area().to_f64() - 6.0).abs() < 1e-12);
}

#[test]
fn inertia_diagonal_is_solid_box_formula() {
    // I_x = m/12 (Ly^2 + Lz^2) with full edges L = 2h
    let b = OrientedBox::axis_aligned(Vec3Fix::ZERO, v3(1.0, 2.0, 3.0));
    let m = 7.0;
    let i = arr(b.inertia_diagonal(fx(m)));
    let (lx, ly, lz) = (2.0, 4.0, 6.0);
    let want = [
        m / 12.0 * (ly * ly + lz * lz),
        m / 12.0 * (lx * lx + lz * lz),
        m / 12.0 * (lx * lx + ly * ly),
    ];
    for k in 0..3 {
        assert!(
            (i[k] - want[k]).abs() < 1e-9,
            "axis {k}: {} vs {}",
            i[k],
            want[k]
        );
    }
}

#[test]
fn inertia_diagonal_ignores_rotation_and_center() {
    // The documented tensor is the body-frame diagonal: rotation / center must not change it.
    let h = v3(1.0, 2.0, 3.0);
    let a = OrientedBox::axis_aligned(Vec3Fix::ZERO, h).inertia_diagonal(fx(5.0));
    for c in cases().iter().take(10) {
        let b = OrientedBox::new(
            v3(c.center[0], c.center[1], c.center[2]),
            h,
            QuatFix::from_axis_angle(v3(c.axis[0], c.axis[1], c.axis[2]), fx(c.ang)),
        );
        assert_eq!(b.inertia_diagonal(fx(5.0)), a);
    }
}

fn directions() -> Vec<[f64; 3]> {
    let mut d = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    // deterministic LCG sweep
    let mut s: u64 = 0x2545F4914F6CDD1D;
    for _ in 0..40 {
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        d.push([next(), next(), next()]);
    }
    d
}

#[test]
fn support_point_attains_maximum_over_corners() {
    for c in cases() {
        let b = build(&c);
        for d in directions() {
            let sp = arr(b.support(v3(d[0], d[1], d[2])));
            let got = sp[0] * d[0] + sp[1] * d[1] + sp[2] * d[2];
            let mut best = f64::MIN;
            for i in 0..8 {
                let p = ref_corner(&c, i);
                best = best.max(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]);
            }
            assert!(
                (got - best).abs() < 1e-8,
                "support dot {got} vs max corner dot {best} for dir {d:?}"
            );
        }
    }
}

#[test]
fn support_point_is_a_corner() {
    for c in cases().iter().take(12) {
        let b = build(c);
        for d in directions().into_iter().take(20) {
            let sp = arr(b.support(v3(d[0], d[1], d[2])));
            let mut nearest = f64::MAX;
            for i in 0..8 {
                let p = ref_corner(c, i);
                let dd = (0..3).map(|k| (p[k] - sp[k]).abs()).fold(0.0, f64::max);
                nearest = nearest.min(dd);
            }
            assert!(
                nearest < 1e-8,
                "support {sp:?} is not a corner (off by {nearest})"
            );
        }
    }
}

#[test]
fn support_is_scale_invariant_in_direction() {
    let c = &cases()[23];
    let b = build(c);
    for d in directions().into_iter().take(20) {
        let a = b.support(v3(d[0], d[1], d[2]));
        let s = b.support(v3(d[0] * 1000.0, d[1] * 1000.0, d[2] * 1000.0));
        assert_eq!(a, s);
    }
}

#[test]
fn support_opposite_directions_are_antipodal_about_center() {
    let c = &cases()[9];
    let b = build(c);
    for d in directions().into_iter().skip(6).take(20) {
        let p = arr(b.support(v3(d[0], d[1], d[2])));
        let q = arr(b.support(v3(-d[0], -d[1], -d[2])));
        for k in 0..3 {
            assert!((p[k] + q[k] - 2.0 * c.center[k]).abs() < 1e-8);
        }
    }
}

#[test]
// AUD-A-S5W2-018
fn non_unit_rotation_must_not_scale_the_box() {
    let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, fx(2.0));
    let b = OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), q);
    let c = arr(b.corner(0));
    assert!(c.iter().all(|v| (v.abs() - 1.0).abs() < 1e-9), "{c:?}");
}
