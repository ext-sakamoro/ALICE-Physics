//! Audit oracles for `wedge::Wedge` (triangular prism).
//!
//! Expected values: the six vertex coordinates written out by hand, the
//! shoelace / prism volume, the centroid of the cross-section, a midpoint
//! quadrature of the solid for the centre of mass and the inertia about it, and
//! an independent f64 Rodrigues rotation for the oriented cases.

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Support;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::wedge::Wedge;

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
    w: f64,
    h: f64,
    d: f64,
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
    let angs = [0.0, 0.4, 1.3, 2.7, -1.0];
    let dims = [
        [4.0, 6.0, 2.0],
        [1.0, 3.0, 5.0],
        [2.5, 0.5, 1.5],
        [3.0, 3.0, 3.0],
    ];
    let centers = [[0.0; 3], [2.0, -3.0, 4.0], [-6.5, 1.25, 8.0]];
    for (i, ax) in axes.iter().enumerate() {
        for (j, &ang) in angs.iter().enumerate() {
            let d = dims[(i + j) % 4];
            out.push(Case {
                center: centers[(i + 2 * j) % 3],
                w: d[0],
                h: d[1],
                d: d[2],
                axis: *ax,
                ang,
            });
        }
    }
    out
}

fn build(c: &Case) -> Wedge {
    Wedge::with_rotation(
        v3(c.center[0], c.center[1], c.center[2]),
        fx(c.w),
        fx(c.h),
        fx(c.d),
        QuatFix::from_axis_angle(v3(c.axis[0], c.axis[1], c.axis[2]), fx(c.ang)),
    )
}

fn local_vertices(c: &Case) -> [[f64; 3]; 6] {
    let (hw, hh, hd) = (c.w / 2.0, c.h / 2.0, c.d / 2.0);
    [
        [-hw, -hh, -hd],
        [hw, -hh, -hd],
        [0.0, hh, -hd],
        [-hw, -hh, hd],
        [hw, -hh, hd],
        [0.0, hh, hd],
    ]
}

fn world_vertices(c: &Case) -> [[f64; 3]; 6] {
    let lv = local_vertices(c);
    let mut out = [[0.0; 3]; 6];
    for i in 0..6 {
        let r = rodrigues(c.axis, c.ang, lv[i]);
        out[i] = [c.center[0] + r[0], c.center[1] + r[1], c.center[2] + r[2]];
    }
    out
}

#[test]
fn constructors_set_fields_and_new_is_unrotated() {
    let w = Wedge::new(v3(1.0, 2.0, 3.0), fx(4.0), fx(5.0), fx(6.0));
    assert_eq!(w.rotation, QuatFix::IDENTITY);
    assert_eq!((w.width, w.height, w.depth), (fx(4.0), fx(5.0), fx(6.0)));
    let q = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(0.3));
    let r = Wedge::with_rotation(v3(1.0, 2.0, 3.0), fx(4.0), fx(5.0), fx(6.0), q);
    assert_eq!(r.rotation, q);
    assert_eq!(r.center, v3(1.0, 2.0, 3.0));
}

#[test]
fn vertices_match_the_documented_layout_unrotated() {
    let w = Wedge::new(v3(10.0, 20.0, 30.0), fx(4.0), fx(6.0), fx(2.0));
    let v = w.vertices();
    let want = [
        [8.0, 17.0, 29.0],
        [12.0, 17.0, 29.0],
        [10.0, 23.0, 29.0],
        [8.0, 17.0, 31.0],
        [12.0, 17.0, 31.0],
        [10.0, 23.0, 31.0],
    ];
    for i in 0..6 {
        assert_eq!(arr(v[i]), want[i], "vertex {i}");
    }
}

#[test]
fn vertices_match_independent_rotation() {
    for c in cases() {
        let got = build(&c).vertices();
        let want = world_vertices(&c);
        for i in 0..6 {
            let g = arr(got[i]);
            for k in 0..3 {
                assert!(
                    (g[k] - want[i][k]).abs() < 1e-9,
                    "vertex {i} axis {k}: {} vs {}",
                    g[k],
                    want[i][k]
                );
            }
        }
    }
}

#[test]
fn aabb_is_the_min_max_hull_of_the_vertices() {
    for c in cases() {
        let bb = build(&c).aabb();
        let wv = world_vertices(&c);
        let (mn, mx) = (arr(bb.min), arr(bb.max));
        for k in 0..3 {
            let lo = wv.iter().map(|p| p[k]).fold(f64::MAX, f64::min);
            let hi = wv.iter().map(|p| p[k]).fold(f64::MIN, f64::max);
            assert!((mn[k] - lo).abs() < 1e-9, "min axis {k}");
            assert!((mx[k] - hi).abs() < 1e-9, "max axis {k}");
        }
    }
}

#[test]
fn volume_is_half_width_height_depth() {
    for c in cases() {
        let v = build(&c).volume().to_f64();
        let want = 0.5 * c.w * c.h * c.d;
        assert!((v - want).abs() < 1e-12 * want.max(1.0), "{v} vs {want}");
    }
}

/// Midpoint quadrature of the unrotated solid (triangle in XY, extruded in Z).
/// Returns (volume, centre of mass, inertia diagonal about the CoM, unit density).
fn quadrature(w: f64, h: f64, d: f64, n: usize) -> (f64, [f64; 3], [f64; 3]) {
    let (mut vol, mut cy) = (0.0, 0.0);
    let (mut sxx, mut syy, mut szz) = (0.0, 0.0, 0.0);
    let cell = (w / n as f64) * (h / n as f64) * (d / n as f64);
    let inside = |x: f64, y: f64| {
        // triangle (-w/2,-h/2), (w/2,-h/2), (0, h/2)
        let t = (y + h / 2.0) / h;
        x.abs() <= (1.0 - t) * w / 2.0
    };
    // first pass: centroid
    let mut pts = Vec::new();
    for i in 0..n {
        let x = -w / 2.0 + (i as f64 + 0.5) * w / n as f64;
        for j in 0..n {
            let y = -h / 2.0 + (j as f64 + 0.5) * h / n as f64;
            if !inside(x, y) {
                continue;
            }
            for k in 0..n {
                let z = -d / 2.0 + (k as f64 + 0.5) * d / n as f64;
                pts.push((x, y, z));
                vol += cell;
                cy += cell * y;
            }
        }
    }
    let cy = cy / vol;
    for (x, y, z) in pts {
        let y = y - cy;
        sxx += cell * (y * y + z * z);
        syy += cell * (x * x + z * z);
        szz += cell * (x * x + y * y);
    }
    (vol, [0.0, cy, 0.0], [sxx, syy, szz])
}

#[test]
fn center_of_mass_offset_is_minus_height_over_six_below_the_geometric_centre() {
    for (w, h, d) in [(4.0, 6.0, 2.0), (1.0, 3.0, 5.0), (2.0, 0.5, 1.0)] {
        let wedge = Wedge::new(Vec3Fix::ZERO, fx(w), fx(h), fx(d));
        let off = arr(wedge.center_of_mass_offset());
        let (_, com, _) = quadrature(w, h, d, 60);
        assert_eq!(off[0], 0.0);
        assert_eq!(off[2], 0.0);
        assert!(
            (off[1] - (-h / 6.0)).abs() < 1e-12,
            "closed form: {}",
            off[1]
        );
        assert!(
            (off[1] - com[1]).abs() < 0.01 * h,
            "quadrature: {} vs {}",
            off[1],
            com[1]
        );
    }
}

#[test]
fn inertia_about_the_centre_of_mass_matches_quadrature_and_the_documented_formulas() {
    let (w, h, d, m) = (4.0, 6.0, 2.0, 5.0);
    let wedge = Wedge::new(Vec3Fix::ZERO, fx(w), fx(h), fx(d));
    let i = arr(wedge.inertia_diagonal(fx(m)));
    let closed = [
        m * (h * h / 18.0 + d * d / 12.0),
        m * (w * w / 24.0 + d * d / 12.0),
        m * (w * w / 24.0 + h * h / 18.0),
    ];
    let (vol, _, q) = quadrature(w, h, d, 80);
    for k in 0..3 {
        assert!(
            (i[k] - closed[k]).abs() < 1e-9,
            "closed form axis {k}: {} vs {}",
            i[k],
            closed[k]
        );
        let qk = m * q[k] / vol;
        assert!(
            (i[k] - qk).abs() < 0.02 * qk,
            "quadrature axis {k}: {} vs {}",
            i[k],
            qk
        );
    }
}

#[test]
fn inertia_is_linear_in_mass() {
    let wedge = Wedge::new(Vec3Fix::ZERO, fx(2.0), fx(3.0), fx(4.0));
    let a = arr(wedge.inertia_diagonal(fx(1.0)));
    let b = arr(wedge.inertia_diagonal(fx(7.0)));
    for k in 0..3 {
        assert!((b[k] - 7.0 * a[k]).abs() < 1e-9);
    }
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
    let mut s: u64 = 0xD1B54A32D192ED03;
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
fn support_attains_the_maximum_dot_over_the_vertices_and_is_a_vertex() {
    for c in cases() {
        let w = build(&c);
        let wv = world_vertices(&c);
        for d in dirs() {
            let p = arr(w.support(v3(d[0], d[1], d[2])));
            let got = p[0] * d[0] + p[1] * d[1] + p[2] * d[2];
            let best = wv
                .iter()
                .map(|q| q[0] * d[0] + q[1] * d[1] + q[2] * d[2])
                .fold(f64::MIN, f64::max);
            assert!((got - best).abs() < 1e-8, "{got} vs {best} for {d:?}");
            let near = wv
                .iter()
                .map(|q| (0..3).map(|k| (q[k] - p[k]).abs()).fold(0.0, f64::max))
                .fold(f64::MAX, f64::min);
            assert!(near < 1e-8, "support is not a vertex");
        }
    }
}

#[test]
fn support_of_an_axis_aligned_wedge_picks_the_documented_vertices() {
    let w = Wedge::new(Vec3Fix::ZERO, fx(4.0), fx(6.0), fx(2.0));
    // apex edge for +Y
    assert_eq!(arr(w.support(Vec3Fix::UNIT_Y))[1], 3.0);
    // +X: base corner x = 2, y = -3
    let s = arr(w.support(Vec3Fix::UNIT_X));
    assert_eq!((s[0], s[1]), (2.0, -3.0));
    // -X: base corner x = -2
    let s = arr(w.support(-Vec3Fix::UNIT_X));
    assert_eq!((s[0], s[1]), (-2.0, -3.0));
    // far along the sloped face normal (6, 2) direction picks the +x base or apex by dot
    let s = arr(w.support(v3(1.0, 1.0, 0.0)));
    // dots: base (2,-3) -> -1 ; apex (0,3) -> 3
    assert_eq!((s[0], s[1]), (0.0, 3.0));
}

#[test]
fn support_picks_the_correct_depth_side() {
    let w = Wedge::new(Vec3Fix::ZERO, fx(4.0), fx(6.0), fx(2.0));
    assert_eq!(arr(w.support(Vec3Fix::UNIT_Z))[2], 1.0);
    assert_eq!(arr(w.support(-Vec3Fix::UNIT_Z))[2], -1.0);
}

#[test]
#[ignore = "known defect: AUD-A-S5W2-020: the orientation quaternion is not normalised or checked, so a non-unit rotation scales the wedge (rotation (0,0,0,2) maps vertex (1,-1,-1) to (4,-4,-4) about the centre; vertices, aabb and support inherit the factor 4)"]
fn non_unit_rotation_must_not_scale_the_wedge() {
    let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, fx(2.0));
    let w = Wedge::with_rotation(Vec3Fix::ZERO, fx(2.0), fx(2.0), fx(2.0), q);
    let v = arr(w.vertices()[1]);
    assert!((v[0] - 1.0).abs() < 1e-9, "{v:?}");
}
