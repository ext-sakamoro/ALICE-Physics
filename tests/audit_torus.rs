//! Audit oracles for `torus`.
//!
//! Closed forms (independent of the crate):
//!
//! * volume `2 pi^2 R r^2`, surface area `4 pi^2 R r`;
//! * support function of a torus (Minkowski sum of a ring of radius `R` in the
//!   local XZ plane and a ball of radius `r`):
//!   `h(d) = R * |d_xz| + r * |d|`, and the support point lies on the surface
//!   `(sqrt(x^2 + z^2) - R)^2 + y^2 = r^2` in the local frame;
//! * inertia about the principal axes from a midpoint quadrature over the tube
//!   cross-section (`dV = (R + s cos phi) s ds dphi dtheta`);
//! * world-space half extent along axis `i` of a rotated torus:
//!   `R * sqrt(1 - w_i^2) + r` with `w` the world image of the symmetry axis.
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Support;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::torus::Torus;
use std::f64::consts::PI;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn f(v: Fix128) -> f64 {
    v.to_f64()
}
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// Unit quaternion (x, y, z, w) from axis and angle, in f64.
fn quat_f64(axis: [f64; 3], ang: f64) -> [f64; 4] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let s = (ang / 2.0).sin() / n;
    [axis[0] * s, axis[1] * s, axis[2] * s, (ang / 2.0).cos()]
}
fn quat_fix(q: [f64; 4]) -> QuatFix {
    QuatFix::new(fx(q[0]), fx(q[1]), fx(q[2]), fx(q[3]))
}
/// Rotate v by unit quaternion q (f64): v + 2 w (u x v) + 2 u x (u x v).
fn rot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let u = [q[0], q[1], q[2]];
    let w = q[3];
    let c = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let t = c(u, v);
    let t2 = c(u, t);
    [
        v[0] + 2.0 * (w * t[0] + t2[0]),
        v[1] + 2.0 * (w * t[1] + t2[1]),
        v[2] + 2.0 * (w * t[2] + t2[2]),
    ]
}
fn conj(q: [f64; 4]) -> [f64; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

const R: f64 = 3.0;
const RR: f64 = 0.75;

fn sample_torus() -> (Torus, [f64; 4]) {
    let q = quat_f64([1.0, 2.0, -0.5], 1.1);
    (
        Torus::with_rotation(v3(2.0, -1.0, 0.5), fx(R), fx(RR), quat_fix(q)),
        q,
    )
}

#[test]
fn support_value_equals_ring_plus_ball_support_function() {
    let (t, q) = sample_torus();
    let c = [2.0, -1.0, 0.5];
    let mut s = 7u64;
    for _ in 0..300 {
        let d = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if dn < 0.1 {
            continue;
        }
        let p = t.support(v3(d[0], d[1], d[2]));
        let rel_p = [f(p.x) - c[0], f(p.y) - c[1], f(p.z) - c[2]];
        let got = rel_p[0] * d[0] + rel_p[1] * d[1] + rel_p[2] * d[2];
        let dl = rot(conj(q), d);
        let want = R * (dl[0] * dl[0] + dl[2] * dl[2]).sqrt() + RR * dn;
        assert!((got - want).abs() < 1e-9, "d={d:?} got {got} want {want}");
    }
}

#[test]
fn support_point_lies_on_the_torus_surface() {
    let (t, q) = sample_torus();
    let c = [2.0, -1.0, 0.5];
    let mut s = 11u64;
    for _ in 0..300 {
        let d = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if dn < 0.1 {
            continue;
        }
        let p = t.support(v3(d[0], d[1], d[2]));
        let l = rot(conj(q), [f(p.x) - c[0], f(p.y) - c[1], f(p.z) - c[2]]);
        let ring = (l[0] * l[0] + l[2] * l[2]).sqrt();
        let resid = (ring - R) * (ring - R) + l[1] * l[1] - RR * RR;
        assert!(resid.abs() < 1e-9, "d={d:?} resid {resid}");
    }
}

#[test]
fn support_is_the_maximiser_over_a_dense_surface_sampling() {
    // Brute force over (u, v): p = ((R + r cos v) cos u, r sin v, (R + r cos v) sin u).
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    for d in [[0.3, 0.9, -0.2], [-1.0, 0.1, 0.4], [0.0, -1.0, 0.05]] {
        let mut best = f64::NEG_INFINITY;
        for iu in 0..720 {
            let u = 2.0 * PI * f64::from(iu) / 720.0;
            for iv in 0..360 {
                let v = 2.0 * PI * f64::from(iv) / 360.0;
                let p = [
                    (R + RR * v.cos()) * u.cos(),
                    RR * v.sin(),
                    (R + RR * v.cos()) * u.sin(),
                ];
                best = best.max(p[0] * d[0] + p[1] * d[1] + p[2] * d[2]);
            }
        }
        let p = t.support(v3(d[0], d[1], d[2]));
        let got = f(p.x) * d[0] + f(p.y) * d[1] + f(p.z) * d[2];
        assert!(got >= best - 1e-9, "support {got} below sampled max {best}");
        assert!(
            (got - best).abs() < 1e-3,
            "support {got} vs sampled max {best}"
        );
    }
}

#[test]
fn support_translates_exactly_with_the_center() {
    let q = quat_f64([0.0, 0.0, 1.0], 0.4);
    let a = Torus::with_rotation(Vec3Fix::ZERO, fx(R), fx(RR), quat_fix(q));
    let b = Torus::with_rotation(v3(5.0, -7.0, 2.0), fx(R), fx(RR), quat_fix(q));
    let d = v3(0.3, -0.8, 0.5);
    let pa = a.support(d);
    let pb = b.support(d);
    assert_eq!(pb, pa + v3(5.0, -7.0, 2.0));
}

#[test]
fn support_is_point_symmetric_for_a_centered_torus() {
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    for d in [
        [0.3, 0.9, -0.2],
        [-1.0, 0.1, 0.4],
        [1.0, 0.0, 0.0],
        [0.2, 1.0, 0.7],
    ] {
        let p = t.support(v3(d[0], d[1], d[2]));
        let m = t.support(v3(-d[0], -d[1], -d[2]));
        assert!((f(p.x) + f(m.x)).abs() < 1e-12);
        assert!((f(p.y) + f(m.y)).abs() < 1e-12);
        assert!((f(p.z) + f(m.z)).abs() < 1e-12);
    }
}

#[test]
fn support_y_direction_reaches_height_minor_radius() {
    // d = +Y: h = R*0 + r*1 = r, and the point is above a ring point.
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    let p = t.support(Vec3Fix::UNIT_Y);
    assert!((f(p.y) - RR).abs() < 1e-12);
    assert!((f(p.x) * f(p.x) + f(p.z) * f(p.z)).sqrt() - R < 1e-12);
}

#[test]
fn support_is_invariant_to_direction_magnitude_in_a_sane_range() {
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    let base = t.support(v3(0.3, 0.5, -0.7));
    for k in [1e-3, 1e-2, 10.0, 1e3] {
        let p = t.support(v3(0.3 * k, 0.5 * k, -0.7 * k));
        assert!((f(p.x) - f(base.x)).abs() < 1e-9, "k={k}");
        assert!((f(p.y) - f(base.y)).abs() < 1e-9, "k={k}");
        assert!((f(p.z) - f(base.z)).abs() < 1e-9, "k={k}");
    }
}

#[test]
fn volume_and_surface_area_match_closed_forms() {
    for (rr, r) in [(3.0, 1.0), (10.0, 2.5), (0.5, 0.25), (100.0, 40.0)] {
        let t = Torus::new(Vec3Fix::ZERO, fx(rr), fx(r));
        assert!(
            rel(f(t.volume()), 2.0 * PI * PI * rr * r * r) < 1e-12,
            "vol {rr} {r}"
        );
        assert!(
            rel(f(t.surface_area()), 4.0 * PI * PI * rr * r) < 1e-12,
            "area {rr} {r}"
        );
    }
}

#[test]
fn volume_matches_pappus_quadrature() {
    // V = 2 pi * (area of disc) * R = 2 pi R * pi r^2, integrated numerically
    // as 2 pi * integral over the tube cross-section of (R + s cos phi) dA.
    let (rr, r) = (4.0, 1.2);
    let (ns, nphi) = (200, 400);
    let mut sum = 0.0;
    for i in 0..ns {
        let s = (f64::from(i) + 0.5) / f64::from(ns) * r;
        for j in 0..nphi {
            let ph = (f64::from(j) + 0.5) / f64::from(nphi) * 2.0 * PI;
            sum += (rr + s * ph.cos()) * s * (r / f64::from(ns)) * (2.0 * PI / f64::from(nphi));
        }
    }
    let t = Torus::new(Vec3Fix::ZERO, fx(rr), fx(r));
    assert!(rel(f(t.volume()), 2.0 * PI * sum) < 1e-4);
}

#[test]
fn inertia_matches_quadrature_about_each_principal_axis() {
    let (rr, r, m) = (4.0, 1.2, 7.0);
    let (ns, nphi) = (300, 600);
    let (mut vol, mut iyy_i, mut ixx_i) = (0.0, 0.0, 0.0);
    for i in 0..ns {
        let s = (f64::from(i) + 0.5) / f64::from(ns) * r;
        for j in 0..nphi {
            let ph = (f64::from(j) + 0.5) / f64::from(nphi) * 2.0 * PI;
            let da = s * (r / f64::from(ns)) * (2.0 * PI / f64::from(nphi));
            let rho = rr + s * ph.cos(); // distance from symmetry axis
            let y = s * ph.sin();
            vol += rho * da;
            iyy_i += rho * rho * rho * da; // integral of rho^2 dV / (2 pi)
                                           // theta-average of (y^2 + z^2), z = rho sin(theta): y^2 + rho^2 / 2
            ixx_i += (y * y + rho * rho / 2.0) * rho * da;
        }
    }
    let t = Torus::new(Vec3Fix::ZERO, fx(rr), fx(r));
    let i = t.inertia_diagonal(fx(m));
    assert!(
        rel(f(i.y), m * iyy_i / vol) < 1e-4,
        "Iyy {} vs {}",
        f(i.y),
        m * iyy_i / vol
    );
    assert!(
        rel(f(i.x), m * ixx_i / vol) < 1e-4,
        "Ixx {} vs {}",
        f(i.x),
        m * ixx_i / vol
    );
    assert_eq!(i.x, i.z);
}

#[test]
fn inertia_thin_ring_limit_and_linearity_in_mass() {
    // r -> 0: Iyy = m R^2, Ixx = Izz = m R^2 / 2.
    let t = Torus::new(Vec3Fix::ZERO, fx(2.0), fx(1e-6));
    let i = t.inertia_diagonal(fx(3.0));
    assert!(rel(f(i.y), 3.0 * 4.0) < 1e-9);
    assert!(rel(f(i.x), 3.0 * 2.0) < 1e-9);
    let i2 = t.inertia_diagonal(fx(6.0));
    assert!(rel(f(i2.y), 2.0 * f(i.y)) < 1e-12);
}

#[test]
fn aabb_encloses_every_support_point_for_rotated_torus() {
    let (t, _q) = sample_torus();
    let bb = t.aabb();
    let mut s = 3u64;
    for _ in 0..2000 {
        let d = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if dn < 0.1 {
            continue;
        }
        let p = t.support(v3(d[0], d[1], d[2]));
        assert!(p.x >= bb.min.x && p.x <= bb.max.x);
        assert!(p.y >= bb.min.y && p.y <= bb.max.y);
        assert!(p.z >= bb.min.z && p.z <= bb.max.z);
    }
}

#[test]
fn aabb_is_centered_and_at_least_the_closed_form_half_extent() {
    let (t, q) = sample_torus();
    let bb = t.aabb();
    let c = [2.0, -1.0, 0.5];
    let w = rot(q, [0.0, 1.0, 0.0]);
    let lo = [f(bb.min.x), f(bb.min.y), f(bb.min.z)];
    let hi = [f(bb.max.x), f(bb.max.y), f(bb.max.z)];
    for i in 0..3 {
        assert!(
            (lo[i] + hi[i] - 2.0 * c[i]).abs() < 1e-9,
            "axis {i} not centered"
        );
        let need = R * (1.0 - w[i] * w[i]).sqrt() + RR;
        assert!(
            hi[i] - c[i] >= need - 1e-9,
            "axis {i}: half extent {} < {need}",
            hi[i] - c[i]
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-001: Torus::aabb is not tight along the symmetry axis (R=5, r=1, identity: Y half extent 6, closed form r = 1)"]
fn aabb_half_extent_along_symmetry_axis_is_minor_radius() {
    let t = Torus::new(Vec3Fix::ZERO, fx(5.0), fx(1.0));
    let bb = t.aabb();
    assert!(
        (f(bb.max.y) - 1.0).abs() < 1e-9,
        "Y half extent {}",
        f(bb.max.y)
    );
}

#[test]
fn aabb_of_axis_aligned_torus_has_exact_ring_plane_extent() {
    let t = Torus::new(Vec3Fix::ZERO, fx(5.0), fx(1.0));
    let bb = t.aabb();
    assert!((f(bb.max.x) - 6.0).abs() < 1e-12);
    assert!((f(bb.max.z) - 6.0).abs() < 1e-12);
    assert!((f(bb.min.x) + 6.0).abs() < 1e-12);
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-002: support() does not tolerate a tiny direction (1e-10 scale squares to 0 in Fix128, support(1e-10,0,1e-10).x = 3.75 = R+r instead of 2.6517 = (R+r)/sqrt2 for R=3, r=0.75)"]
fn support_is_invariant_to_tiny_direction_magnitude() {
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    let a = t.support(v3(1.0, 0.0, 1.0));
    let b = t.support(v3(1e-10, 0.0, 1e-10));
    assert!((f(a.x) - f(b.x)).abs() < 1e-6, "x {} vs {}", f(a.x), f(b.x));
    assert!((f(a.z) - f(b.z)).abs() < 1e-6, "z {} vs {}", f(a.z), f(b.z));
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-003: support() with a huge direction (|d| ~ 1e10) wraps in Fix128 squares and returns a point that is not the maximiser"]
fn support_is_invariant_to_huge_direction_magnitude() {
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    let a = t.support(v3(1.0, 0.0, 1.0));
    let b = t.support(v3(1e10, 0.0, 1e10));
    assert!((f(a.x) - f(b.x)).abs() < 1e-6, "x {} vs {}", f(a.x), f(b.x));
    assert!((f(a.z) - f(b.z)).abs() < 1e-6, "z {} vs {}", f(a.z), f(b.z));
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-004: Torus accepts a non-unit orientation and then scales support points by |q|^2 (q = (0,0,0,2): support(+X) = (24, 0, 0) instead of (R+r, 0, 0) = (6, 0, 0))"]
fn non_unit_rotation_does_not_scale_the_torus() {
    let q = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, fx(2.0));
    let t = Torus::with_rotation(Vec3Fix::ZERO, fx(5.0), fx(1.0), q);
    let p = t.support(Vec3Fix::UNIT_X);
    assert!((f(p.x) - 6.0).abs() < 1e-9, "x = {}", f(p.x));
}

#[test]
#[ignore = "known defect: AUD-A-S5W3-005: Torus::new accepts a negative minor radius; support(+X) with R=5, r=-1 returns x = 4 (the surface point nearest the axis, minimiser not maximiser)"]
fn negative_minor_radius_is_rejected_or_still_maximises() {
    let t = Torus::new(Vec3Fix::ZERO, fx(5.0), fx(-1.0));
    let p = t.support(Vec3Fix::UNIT_X);
    assert!((f(p.x) - 6.0).abs() < 1e-9, "x = {}", f(p.x));
}

#[test]
fn zero_direction_returns_a_finite_point_on_the_surface() {
    let t = Torus::new(Vec3Fix::ZERO, fx(R), fx(RR));
    let p = t.support(Vec3Fix::ZERO);
    let ring = (f(p.x) * f(p.x) + f(p.z) * f(p.z)).sqrt();
    assert!(((ring - R) * (ring - R) + f(p.y) * f(p.y) - RR * RR).abs() < 1e-9);
}

#[test]
fn constructors_store_fields_and_new_is_identity_rotation() {
    let a = Torus::new(v3(1.0, 2.0, 3.0), fx(5.0), fx(2.0));
    assert_eq!(a.rotation, QuatFix::IDENTITY);
    assert_eq!(a.center, v3(1.0, 2.0, 3.0));
    assert_eq!(a.major_radius, fx(5.0));
    assert_eq!(a.minor_radius, fx(2.0));
    let q = quat_fix(quat_f64([0.0, 1.0, 0.0], 0.5));
    let b = Torus::with_rotation(v3(1.0, 2.0, 3.0), fx(5.0), fx(2.0), q);
    assert_eq!(b.rotation, q);
}

#[test]
fn rotated_support_follows_the_rotated_symmetry_axis() {
    // Rotating by 90 degrees about X sends local +Y to world +Z; the highest point
    // in world +Z is the tube top, at distance r.
    let q = quat_fix(quat_f64([1.0, 0.0, 0.0], PI / 2.0));
    let t = Torus::with_rotation(Vec3Fix::ZERO, fx(R), fx(RR), q);
    let p = t.support(Vec3Fix::UNIT_Z);
    assert!((f(p.z) - RR).abs() < 1e-9, "z = {}", f(p.z));
}
