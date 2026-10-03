//! Audit oracles for `alice_physics::cylinder::Cylinder`.
//!
//! References are independent of the implementation: the support function is
//! compared with the closed form `c.d + hh |d.a| + r sqrt(|d|^2 - (d.a)^2)`
//! (axis `a` = rotated local Y) and with a brute-force maximum over rim
//! points; the AABB must enclose a dense rim sampling; the surface area is the
//! textbook `2 pi r (r + 2 hh)`.

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Support;
use alice_physics::cylinder::Cylinder;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

const PI: f64 = std::f64::consts::PI;

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

/// Rotation matrix (row-major) from the quaternion components actually stored
/// in `q`, normalised in f64.
fn rot_matrix(q: QuatFix) -> [[f64; 3]; 3] {
    let (x, y, z, w) = (q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64());
    let n = (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = (x / n, y / n, z / n, w / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn mul(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

fn quat(axis: [f64; 3], angle: f64) -> QuatFix {
    let n = dot(axis, axis).sqrt();
    QuatFix::from_axis_angle(v3(axis[0] / n, axis[1] / n, axis[2] / n), fx(angle))
}

/// Deterministic LCG in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn sample_cylinders() -> Vec<Cylinder> {
    let mut rng = Lcg(0x5EED);
    let mut v = vec![Cylinder::new(v3(1.0, -2.0, 0.5), fx(1.5), fx(0.75))];
    for _ in 0..12 {
        let axis = [rng.next() + 0.01, rng.next(), rng.next()];
        let q = quat(axis, rng.next() * PI);
        let c = v3(rng.next() * 5.0, rng.next() * 5.0, rng.next() * 5.0);
        let hh = 0.2 + (rng.next() + 1.0) * 1.5;
        let r = 0.2 + (rng.next() + 1.0) * 1.5;
        v.push(Cylinder::with_rotation(c, fx(hh), fx(r), q));
    }
    v
}

fn rim_points(c: &Cylinder) -> Vec<[f64; 3]> {
    let m = rot_matrix(c.rotation);
    let (hh, r) = (c.half_height.to_f64(), c.radius.to_f64());
    let ctr = arr(c.center);
    let mut pts = Vec::new();
    for k in 0..72 {
        let th = 2.0 * PI * f64::from(k) / 72.0;
        for &s in &[-1.0, 1.0] {
            let w = mul(m, [r * th.cos(), s * hh, r * th.sin()]);
            pts.push([ctr[0] + w[0], ctr[1] + w[1], ctr[2] + w[2]]);
        }
    }
    pts
}

#[test]
fn support_value_matches_the_closed_form_for_random_orientations() {
    let mut rng = Lcg(77);
    for c in sample_cylinders() {
        let m = rot_matrix(c.rotation);
        let a = mul(m, [0.0, 1.0, 0.0]);
        let (hh, r) = (c.half_height.to_f64(), c.radius.to_f64());
        let ctr = arr(c.center);
        for _ in 0..20 {
            let d = [rng.next(), rng.next(), rng.next()];
            let s = arr(c.support(v3(d[0], d[1], d[2])));
            let da = dot(d, a);
            let want = dot(d, ctr) + hh * da.abs() + r * (dot(d, d) - da * da).max(0.0).sqrt();
            let got = dot(d, s);
            assert!((got - want).abs() < 1.0e-8, "got {got} want {want}");
        }
    }
}

#[test]
fn support_is_a_rim_point_and_maximal_over_a_rim_sampling() {
    let mut rng = Lcg(4242);
    for c in sample_cylinders() {
        let m = rot_matrix(c.rotation);
        let inv = [
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ];
        let (hh, r) = (c.half_height.to_f64(), c.radius.to_f64());
        let ctr = arr(c.center);
        let pts = rim_points(&c);
        for _ in 0..20 {
            let d = [rng.next(), rng.next(), rng.next()];
            let s = arr(c.support(v3(d[0], d[1], d[2])));
            // on the rim: |local y| = hh, local radial = r
            let loc = mul(inv, [s[0] - ctr[0], s[1] - ctr[1], s[2] - ctr[2]]);
            assert!((loc[1].abs() - hh).abs() < 1.0e-8, "cap plane");
            assert!(
                ((loc[0] * loc[0] + loc[2] * loc[2]).sqrt() - r).abs() < 1.0e-8,
                "radius"
            );
            let best = pts.iter().map(|p| dot(d, *p)).fold(f64::MIN, f64::max);
            assert!(dot(d, s) >= best - 1.0e-9, "support not maximal");
        }
    }
}

#[test]
fn support_is_invariant_to_positive_scaling_of_the_direction() {
    let c = Cylinder::with_rotation(
        v3(0.5, 0.25, -1.0),
        fx(1.0),
        fx(0.5),
        quat([1.0, 2.0, 3.0], 0.9),
    );
    let d = [0.3, -0.8, 0.5];
    let base = arr(c.support(v3(d[0], d[1], d[2])));
    for &k in &[1.0e-3, 0.5, 2.0, 1.0e3, 1.0e6] {
        let s = arr(c.support(v3(d[0] * k, d[1] * k, d[2] * k)));
        for i in 0..3 {
            assert!(
                (s[i] - base[i]).abs() < 1.0e-6,
                "k={k} axis {i}: {} vs {}",
                s[i],
                base[i]
            );
        }
    }
}

/// `xz_len_sq = dx^2 + dz^2` is computed in `Fix128`, whose multiplication
/// wraps on overflow, so a direction with |component| ~ 1e10 (square 1e20 >
/// 2^63) yields a different support point than the same direction scaled down.
#[test]
#[ignore = "known defect: AUD-A-S6W1-007: support() is not scale invariant for |dir| >= ~3.04e9 (Fix128 square wraps): k=4e9 gives (0,1,0), k=1e10 gives (4.31,1,5.74) instead of (1.2,1,1.6)"]
fn support_is_invariant_for_very_large_directions() {
    let c = Cylinder::new(Vec3Fix::ZERO, fx(1.0), fx(2.0));
    let d = [0.6, 0.3, 0.8];
    let base = arr(c.support(v3(d[0], d[1], d[2])));
    let k = 1.0e10;
    let s = arr(c.support(v3(d[0] * k, d[1] * k, d[2] * k)));
    for i in 0..3 {
        assert!(
            (s[i] - base[i]).abs() < 1.0e-6,
            "axis {i}: {} vs {}",
            s[i],
            base[i]
        );
    }
}

#[test]
fn rotated_cylinder_support_has_exact_closed_form_points() {
    // local Y maps to world -X for +90 degrees about Z.
    let rot = quat([0.0, 0.0, 1.0], PI / 2.0);
    let c = Cylinder::with_rotation(Vec3Fix::ZERO, fx(3.0), fx(1.0), rot);
    let s = arr(c.support(v3(1.0, 0.0, 0.0)));
    assert!(
        (s[0] - 3.0).abs() < 1.0e-9 && s[1].abs() < 1.0e-9 && s[2].abs() < 1.0e-9,
        "{s:?}"
    );
    let s = arr(c.support(v3(1.0, 1.0, 0.0)));
    assert!(
        (s[0] - 3.0).abs() < 1.0e-9 && (s[1] - 1.0).abs() < 1.0e-9 && s[2].abs() < 1.0e-9,
        "{s:?}"
    );
    let s = arr(c.support(v3(-1.0, 0.0, 0.0)));
    assert!((s[0] + 3.0).abs() < 1.0e-9 && s[1].abs() < 1.0e-9, "{s:?}");
}

#[test]
fn support_translates_with_the_center_and_zero_direction_is_a_cap_point() {
    let a = Cylinder::new(Vec3Fix::ZERO, fx(2.0), fx(1.0));
    let b = Cylinder::new(v3(10.0, -4.0, 7.0), fx(2.0), fx(1.0));
    let d = v3(0.2, -0.9, 0.4);
    let sa = arr(a.support(d));
    let sb = arr(b.support(d));
    assert!((sb[0] - sa[0] - 10.0).abs() < 1.0e-9);
    assert!((sb[1] - sa[1] + 4.0).abs() < 1.0e-9);
    assert!((sb[2] - sa[2] - 7.0).abs() < 1.0e-9);
    let z = arr(a.support(Vec3Fix::ZERO));
    assert_eq!(z, [0.0, 2.0, 0.0]);
}

#[test]
fn aabb_encloses_every_rim_point_and_obeys_the_conservative_formula() {
    for c in sample_cylinders() {
        let bb = c.aabb();
        let m = rot_matrix(c.rotation);
        let a = mul(m, [0.0, 1.0, 0.0]);
        let (hh, r) = (c.half_height.to_f64(), c.radius.to_f64());
        let ctr = arr(c.center);
        let (lo, hi) = (arr(bb.min), arr(bb.max));
        for p in rim_points(&c) {
            for i in 0..3 {
                assert!(
                    p[i] >= lo[i] - 1.0e-9 && p[i] <= hi[i] + 1.0e-9,
                    "rim point outside axis {i}"
                );
            }
        }
        for i in 0..3 {
            let exact = hh * a[i].abs() + r * (1.0 - a[i] * a[i]).max(0.0).sqrt();
            let loose = hh * a[i].abs() + r;
            let half = 0.5 * (hi[i] - lo[i]);
            assert!(
                half >= exact - 1.0e-9,
                "smaller than the exact extent on axis {i}"
            );
            assert!(
                half <= loose + 1.0e-9,
                "larger than documented formula on axis {i}"
            );
            assert!(
                ((hi[i] + lo[i]) * 0.5 - ctr[i]).abs() < 1.0e-9,
                "AABB not centred"
            );
        }
    }
}

#[test]
fn aabb_is_exact_for_the_unrotated_cylinder() {
    let c = Cylinder::new(v3(5.0, 0.0, 0.0), fx(2.0), fx(1.0));
    let bb = c.aabb();
    assert_eq!(arr(bb.min), [4.0, -3.0, -1.0]);
    assert_eq!(arr(bb.max), [6.0, 3.0, 1.0]);
}

#[test]
fn surface_area_matches_the_textbook_closed_form() {
    for &(hh, r) in &[(1.0, 1.0), (2.5, 0.5), (0.1, 3.0), (7.0, 2.0)] {
        let c = Cylinder::new(Vec3Fix::ZERO, fx(hh), fx(r));
        let want = 2.0 * PI * r * r + 2.0 * PI * r * (2.0 * hh); // two caps + lateral
        let got = c.surface_area().to_f64();
        assert!(
            (got - want).abs() < 1.0e-11 * want,
            "hh {hh} r {r}: {got} vs {want}"
        );
    }
}

#[test]
fn volume_and_inertia_scale_laws() {
    let a = Cylinder::new(Vec3Fix::ZERO, fx(1.0), fx(1.0));
    let b = Cylinder::new(Vec3Fix::ZERO, fx(2.0), fx(3.0));
    // V ~ r^2 hh : 9 * 2 = 18
    assert!((b.volume().to_f64() / a.volume().to_f64() - 18.0).abs() < 1.0e-9);
    // inertia is linear in mass
    let i1 = arr(a.inertia_diagonal(fx(1.0)));
    let i5 = arr(a.inertia_diagonal(fx(5.0)));
    for k in 0..3 {
        assert!((i5[k] - 5.0 * i1[k]).abs() < 1.0e-12);
    }
    // thin disc (hh -> 0): Iyy = m r^2 / 2 and Ixx = m r^2 / 4
    let disc = Cylinder::new(Vec3Fix::ZERO, fx(0.0), fx(2.0));
    let i = arr(disc.inertia_diagonal(fx(8.0)));
    assert!((i[1] - 16.0).abs() < 1.0e-12 && (i[0] - 8.0).abs() < 1.0e-12 && i[0] == i[2]);
}

#[test]
fn with_rotation_and_new_store_their_fields() {
    let q = quat([0.0, 1.0, 0.0], 0.4);
    let c = Cylinder::with_rotation(v3(1.0, 2.0, 3.0), fx(0.5), fx(0.25), q);
    assert_eq!(c.center, v3(1.0, 2.0, 3.0));
    assert_eq!(c.half_height, fx(0.5));
    assert_eq!(c.radius, fx(0.25));
    assert_eq!(c.rotation, q);
    let n = Cylinder::new(Vec3Fix::ZERO, fx(0.5), fx(0.25));
    assert_eq!(n.rotation, QuatFix::IDENTITY);
}

/// Neither constructor checks the sign of its dimensions. A negative
/// `half_height` gives a negative volume and a negative `radius` flips the rim
/// point to the far side, so `support` is no longer the maximum.
#[test]
#[ignore = "known defect: AUD-A-S6W1-008: half_height=-1 gives volume -6.283, radius=-2 gives support x=-2 for +X (no dimension check)"]
fn negative_dimensions_do_not_break_volume_or_support() {
    let c = Cylinder::new(Vec3Fix::ZERO, fx(-1.0), fx(1.0));
    assert!(c.volume().to_f64() >= 0.0);
    let n = Cylinder::new(Vec3Fix::ZERO, fx(1.0), fx(-2.0));
    let s = arr(n.support(v3(1.0, 0.0, 0.0)));
    assert!(s[0] >= 1.999, "support x {} should be +radius", s[0]);
}

/// `rotation` is a public field with no normalisation. A quaternion of norm 2
/// scales `rotate_vec` by |q|^2 = 4 in both the direction transform and the
/// back transform, so the support point lands 4x too far from the centre.
#[test]
#[ignore = "known defect: AUD-A-S6W1-009: rotation quaternion (0,0,0,2) gives support (4,4,0) instead of (1,1,0), 4x too far from the center (no normalisation)"]
fn non_unit_rotation_does_not_scale_the_support_point() {
    let unit = QuatFix::IDENTITY;
    let two = QuatFix::new(fx(0.0), fx(0.0), fx(0.0), fx(2.0));
    let a = Cylinder::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(1.0), unit);
    let b = Cylinder::with_rotation(Vec3Fix::ZERO, fx(1.0), fx(1.0), two);
    let d = v3(1.0, 0.0, 0.0);
    assert_eq!(arr(a.support(d)), arr(b.support(d)));
}
