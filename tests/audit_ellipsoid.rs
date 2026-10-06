//! Audit oracles for `ellipsoid::Ellipsoid`.
//!
//! Expectations: the implicit surface equation `(x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1`,
//! the closed-form support point `r^2 d / |r d|` checked against a dense
//! parametric sample of the surface, the textbook volume and inertia, and an
//! independent f64 Rodrigues rotation.

#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::collider::Support;
use alice_physics::ellipsoid::Ellipsoid;
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

fn rodrigues(axis: [f64; 3], ang: f64, v: [f64; 3]) -> [f64; 3] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / n, axis[1] / n, axis[2] / n];
    let (s, c) = (ang.sin(), ang.cos());
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

struct Case {
    center: [f64; 3],
    radii: [f64; 3],
    axis: [f64; 3],
    ang: f64,
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let axes = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, -2.0, 3.0],
    ];
    let angs = [0.0, 0.5, 1.2, 2.5, -0.9];
    let radii = [
        [1.0, 2.0, 3.0],
        [3.0, 1.0, 0.5],
        [0.5, 2.5, 1.5],
        [2.0, 2.0, 2.0],
    ];
    let centers = [[0.0, 0.0, 0.0], [4.0, -3.0, 2.0], [-7.5, 1.25, 9.0]];
    for (i, ax) in axes.iter().enumerate() {
        for (j, &ang) in angs.iter().enumerate() {
            out.push(Case {
                center: centers[(i + j) % 3],
                radii: radii[(i + 2 * j) % 4],
                axis: *ax,
                ang,
            });
        }
    }
    out
}

fn build(c: &Case) -> Ellipsoid {
    Ellipsoid::with_rotation(
        v3(c.center[0], c.center[1], c.center[2]),
        v3(c.radii[0], c.radii[1], c.radii[2]),
        QuatFix::from_axis_angle(v3(c.axis[0], c.axis[1], c.axis[2]), fx(c.ang)),
    )
}

/// World-space surface point for the local unit direction `u`.
fn surface_point(c: &Case, u: [f64; 3]) -> [f64; 3] {
    let local = [c.radii[0] * u[0], c.radii[1] * u[1], c.radii[2] * u[2]];
    let r = rodrigues(c.axis, c.ang, local);
    [c.center[0] + r[0], c.center[1] + r[1], c.center[2] + r[2]]
}

fn sphere_samples(n: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for i in 0..=n {
        let th = std::f64::consts::PI * i as f64 / n as f64;
        for j in 0..2 * n {
            let ph = std::f64::consts::PI * j as f64 / n as f64;
            out.push([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()]);
        }
    }
    out
}

fn dirs() -> Vec<[f64; 3]> {
    let mut d = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut s: u64 = 0x9E3779B97F4A7C15;
    for _ in 0..30 {
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
fn constructors_set_fields() {
    let e = Ellipsoid::new(v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0));
    assert_eq!(e.rotation, QuatFix::IDENTITY);
    assert_eq!((e.center, e.radii), (v3(1.0, 2.0, 3.0), v3(4.0, 5.0, 6.0)));
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(0.5));
    let w = Ellipsoid::with_rotation(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), q);
    assert_eq!(w.rotation, q);
}

#[test]
fn bounding_sphere_radius_is_the_largest_semi_axis_in_every_position() {
    for r in [
        [5.0, 1.0, 3.0],
        [1.0, 5.0, 3.0],
        [1.0, 3.0, 5.0],
        [2.0, 2.0, 2.0],
        [4.0, 4.0, 1.0],
        [0.5, 4.0, 4.0],
    ] {
        let e = Ellipsoid::new(Vec3Fix::ZERO, v3(r[0], r[1], r[2]));
        let want = r[0].max(r[1]).max(r[2]);
        assert_eq!(e.bounding_sphere_radius(), fx(want), "{r:?}");
    }
}

#[test]
fn bounding_sphere_contains_every_surface_point() {
    for c in cases() {
        let e = build(&c);
        let rad = e.bounding_sphere_radius().to_f64();
        for u in sphere_samples(12) {
            let p = surface_point(&c, u);
            let d = (0..3)
                .map(|k| (p[k] - c.center[k]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d <= rad + 1e-9, "|p-c| = {d} > bounding radius {rad}");
        }
    }
}

#[test]
fn aabb_encloses_the_whole_surface() {
    for c in cases() {
        let bb = build(&c).aabb();
        let (mn, mx) = (arr(bb.min), arr(bb.max));
        for u in sphere_samples(16) {
            let p = surface_point(&c, u);
            for k in 0..3 {
                assert!(
                    p[k] >= mn[k] - 1e-9 && p[k] <= mx[k] + 1e-9,
                    "axis {k}: {} not in [{}, {}]",
                    p[k],
                    mn[k],
                    mx[k]
                );
            }
        }
    }
}

#[test]
fn aabb_is_exact_for_axis_aligned_ellipsoids() {
    let e = Ellipsoid::new(v3(1.0, -2.0, 3.0), v3(2.0, 3.0, 0.5));
    let bb = e.aabb();
    assert_eq!(arr(bb.min), [-1.0, -5.0, 2.5]);
    assert_eq!(arr(bb.max), [3.0, 1.0, 3.5]);
}

#[test]
fn aabb_of_a_rotated_sphere_is_tight() {
    // a sphere of radius 2 is rotation invariant: its AABB half-extent is 2 on every axis
    let e = Ellipsoid::with_rotation(
        Vec3Fix::ZERO,
        v3(2.0, 2.0, 2.0),
        QuatFix::from_axis_angle(v3(1.0, 1.0, 1.0), fx(1.0)),
    );
    let bb = e.aabb();
    let (mn, mx) = (arr(bb.min), arr(bb.max));
    for k in 0..3 {
        assert!(
            (mx[k] - 2.0).abs() < 1e-6 && (mn[k] + 2.0).abs() < 1e-6,
            "axis {k}: [{}, {}]",
            mn[k],
            mx[k]
        );
    }
}

#[test]
fn aabb_of_a_tilted_ellipsoid_equals_the_exact_extent() {
    let c = Case {
        center: [0.0; 3],
        radii: [1.0, 2.0, 3.0],
        axis: [1.0, 1.0, 0.0],
        ang: 0.7,
    };
    let bb = build(&c).aabb();
    // exact half extent on axis i = sqrt(sum_j (R_ij r_j)^2)
    let mut want = [0.0; 3];
    for j in 0..3 {
        let mut ej = [0.0; 3];
        ej[j] = c.radii[j];
        let col = rodrigues(c.axis, c.ang, ej);
        for i in 0..3 {
            want[i] += col[i] * col[i];
        }
    }
    let mx = arr(bb.max);
    for i in 0..3 {
        assert!(
            (mx[i] - want[i].sqrt()).abs() < 1e-6,
            "axis {i}: {} vs {}",
            mx[i],
            want[i].sqrt()
        );
    }
}

#[test]
fn volume_is_four_thirds_pi_abc() {
    for c in cases() {
        let v = build(&c).volume().to_f64();
        let want = 4.0 / 3.0 * std::f64::consts::PI * c.radii[0] * c.radii[1] * c.radii[2];
        assert!((v - want).abs() < 1e-12 * want, "{v} vs {want}");
    }
}

#[test]
fn inertia_diagonal_matches_numeric_integration() {
    // solid ellipsoid at unit mass: I_xx = m/5 (b^2 + c^2); verify with midpoint quadrature
    let (a, b, c) = (1.0_f64, 2.0, 0.5);
    let n = 80;
    let (mut vol, mut ixx, mut iyy, mut izz) = (0.0, 0.0, 0.0, 0.0);
    let cell = (2.0 * a / n as f64) * (2.0 * b / n as f64) * (2.0 * c / n as f64);
    for i in 0..n {
        let x = -a + (i as f64 + 0.5) * 2.0 * a / n as f64;
        for j in 0..n {
            let y = -b + (j as f64 + 0.5) * 2.0 * b / n as f64;
            for k in 0..n {
                let z = -c + (k as f64 + 0.5) * 2.0 * c / n as f64;
                if (x / a).powi(2) + (y / b).powi(2) + (z / c).powi(2) <= 1.0 {
                    vol += cell;
                    ixx += cell * (y * y + z * z);
                    iyy += cell * (x * x + z * z);
                    izz += cell * (x * x + y * y);
                }
            }
        }
    }
    let m = 3.0;
    let e = Ellipsoid::new(Vec3Fix::ZERO, v3(a, b, c));
    let got = arr(e.inertia_diagonal(fx(m)));
    for (g, q) in got.iter().zip([ixx, iyy, izz]) {
        let want = m * q / vol;
        assert!((g - want).abs() < 0.01 * want, "{g} vs quadrature {want}");
    }
    // and the closed form to full precision
    let closed = [
        m / 5.0 * (b * b + c * c),
        m / 5.0 * (a * a + c * c),
        m / 5.0 * (a * a + b * b),
    ];
    for k in 0..3 {
        assert!((got[k] - closed[k]).abs() < 1e-12, "axis {k}");
    }
}

#[test]
fn support_point_lies_on_the_surface() {
    for c in cases() {
        let e = build(&c);
        for d in dirs() {
            let p = arr(e.support(v3(d[0], d[1], d[2])));
            // to local: subtract center, rotate by -ang
            let rel = [p[0] - c.center[0], p[1] - c.center[1], p[2] - c.center[2]];
            let l = rodrigues(c.axis, -c.ang, rel);
            let f = (l[0] / c.radii[0]).powi(2)
                + (l[1] / c.radii[1]).powi(2)
                + (l[2] / c.radii[2]).powi(2);
            assert!((f - 1.0).abs() < 1e-8, "implicit value {f} for dir {d:?}");
        }
    }
}

#[test]
fn support_point_maximises_the_dot_product_over_the_surface() {
    let samples = sphere_samples(40);
    for c in cases().iter().step_by(3) {
        let e = build(c);
        for d in dirs() {
            let p = arr(e.support(v3(d[0], d[1], d[2])));
            let got = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
            let mut best = f64::MIN;
            for u in &samples {
                let q = surface_point(c, *u);
                best = best.max(q[0] * d[0] + q[1] * d[1] + q[2] * d[2]);
            }
            assert!(
                got >= best - 1e-9,
                "support dot {got} < sampled max {best} for {d:?}"
            );
            // closed form of the maximum: c.d + sqrt(sum (r_j (R^T d)_j)^2)
            let l = rodrigues(c.axis, -c.ang, [d[0], d[1], d[2]]);
            let h = ((c.radii[0] * l[0]).powi(2)
                + (c.radii[1] * l[1]).powi(2)
                + (c.radii[2] * l[2]).powi(2))
            .sqrt();
            let want = c.center[0] * d[0] + c.center[1] * d[1] + c.center[2] * d[2] + h;
            assert!(
                (got - want).abs() < 1e-8,
                "support dot {got} vs closed form {want}"
            );
        }
    }
}

#[test]
fn support_is_invariant_to_direction_scale() {
    let c = &cases()[7];
    let e = build(c);
    for d in dirs().into_iter().take(20) {
        let a = arr(e.support(v3(d[0], d[1], d[2])));
        let b = arr(e.support(v3(d[0] * 500.0, d[1] * 500.0, d[2] * 500.0)));
        for k in 0..3 {
            assert!((a[k] - b[k]).abs() < 1e-8, "axis {k}");
        }
    }
}

#[test]
fn support_of_a_zero_direction_is_a_surface_point() {
    // documented degenerate fallback: a point on the local X semi-axis
    let e = Ellipsoid::new(v3(1.0, 1.0, 1.0), v3(3.0, 2.0, 1.0));
    assert_eq!(arr(e.support(Vec3Fix::ZERO)), [4.0, 1.0, 1.0]);
}

#[test]
fn support_of_a_tiny_direction_is_still_the_extreme_point() {
    let e = Ellipsoid::new(Vec3Fix::ZERO, v3(1.0, 2.0, 3.0));
    let p = arr(e.support(v3(0.0, 1e-11, 0.0)));
    assert!(
        (p[1] - 2.0).abs() < 1e-6 && p[0].abs() < 1e-6 && p[2].abs() < 1e-6,
        "{p:?}"
    );
}

#[test]
fn support_of_a_huge_direction_is_still_the_extreme_point() {
    // |r d|^2 would exceed the Fix128 integer range for d = 1e10, but the
    // support point is still the +X pole
    let e = Ellipsoid::new(Vec3Fix::ZERO, v3(3.0, 1.0, 1.0));
    let p = arr(e.support(v3(1e10, 0.0, 0.0)));
    assert!(
        (p[0] - 3.0).abs() < 1e-6 && p[1].abs() < 1e-6 && p[2].abs() < 1e-6,
        "{p:?}"
    );
}

#[test]
// AUD-A-S5W2-019
fn non_unit_rotation_must_not_scale_the_ellipsoid() {
    let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, fx(2.0));
    let e = Ellipsoid::with_rotation(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), q);
    let p = arr(e.support(Vec3Fix::UNIT_X));
    assert!((p[0] - 1.0).abs() < 1e-9, "{p:?}");
    // the bounding box too: the unit sphere's box is [-1, 1]^3
    let b = e.aabb();
    assert!((arr(b.max)[0] - 1.0).abs() < 1e-9, "{:?}", arr(b.max));
    assert!((arr(b.min)[1] + 1.0).abs() < 1e-9, "{:?}", arr(b.min));
}
