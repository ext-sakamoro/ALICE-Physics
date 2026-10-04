//! Audit oracles for `alice_physics::cone`.
//!
//! Expected values come from closed forms (volume, surface area), from numerical
//! integration of the solid in `f64` (centroid, inertia) and from a brute-force
//! sample of the boundary (support point, AABB). None of them is read back from
//! the code under test. The rotation is done in `f64` by Rodrigues' formula.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::collider::Support;
use alice_physics::cone::Cone;
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
/// Rodrigues rotation of `v` about unit `k` by `t`.
fn rot(k: [f64; 3], t: f64, v: [f64; 3]) -> [f64; 3] {
    let (s, c) = (t.sin(), t.cos());
    let kxv = cross(k, v);
    let kv = dot(k, v);
    let mut o = [0.0; 3];
    for i in 0..3 {
        o[i] = v[i] * c + kxv[i] * s + k[i] * kv * (1.0 - c);
    }
    o
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}
fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-12)
}

fn oriented(center: [f64; 3], r: f64, hh: f64, axis: [f64; 3], ang: f64) -> Cone {
    let k = unit(axis);
    let q = QuatFix::from_axis_angle(v3(k[0], k[1], k[2]), fx(ang));
    Cone::with_rotation(v3(center[0], center[1], center[2]), fx(r), fx(hh), q)
}

/// Boundary samples of the cone in world space (apex + 360 rim points + 40 slant points).
fn boundary(center: [f64; 3], r: f64, hh: f64, axis: [f64; 3], ang: f64) -> Vec<[f64; 3]> {
    let k = unit(axis);
    let to_world = |p: [f64; 3]| {
        let q = rot(k, ang, p);
        [center[0] + q[0], center[1] + q[1], center[2] + q[2]]
    };
    let mut pts = vec![to_world([0.0, hh, 0.0])];
    for i in 0..360 {
        let a = (i as f64) * std::f64::consts::PI / 180.0;
        pts.push(to_world([r * a.cos(), -hh, r * a.sin()]));
        if i % 9 == 0 {
            for j in 1..5 {
                let t = j as f64 / 5.0;
                // slant line from rim to apex
                pts.push(to_world([
                    r * a.cos() * (1.0 - t),
                    -hh + 2.0 * hh * t,
                    r * a.sin() * (1.0 - t),
                ]));
            }
        }
    }
    pts
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

// ---------------------------------------------------------------- A: closed forms

#[test]
fn volume_matches_pi_r2_h_over_3() {
    let c = Cone::new(Vec3Fix::ZERO, fx(1.5), fx(2.0));
    let h = 4.0;
    let want = std::f64::consts::PI * 1.5 * 1.5 * h / 3.0;
    assert!(
        rel(c.volume().to_f64(), want) < 1e-12,
        "{}",
        c.volume().to_f64()
    );
    // non-symmetric r, hh (a swap of r and h would show)
    let c = Cone::new(Vec3Fix::ZERO, fx(0.7), fx(3.1));
    let want = std::f64::consts::PI * 0.7 * 0.7 * 6.2 / 3.0;
    assert!(rel(c.volume().to_f64(), want) < 1e-12);
}

#[test]
fn surface_area_matches_pi_r_r_plus_slant() {
    // r = 3, h = 4 (hh = 2): slant 5, area = pi*3*(3+5) = 24 pi
    let c = Cone::new(Vec3Fix::ZERO, fx(3.0), fx(2.0));
    assert!(rel(c.surface_area().to_f64(), 24.0 * std::f64::consts::PI) < 1e-12);
    // r = 1.2, h = 5: slant sqrt(1.44+25)
    let c = Cone::new(Vec3Fix::ZERO, fx(1.2), fx(2.5));
    let want = std::f64::consts::PI * 1.2 * (1.2 + (1.44f64 + 25.0).sqrt());
    assert!(rel(c.surface_area().to_f64(), want) < 1e-12);
}

#[test]
fn surface_area_equals_base_plus_lateral_by_integration() {
    // Independent: base disc pi r^2 + lateral = integral of 2 pi rho(t) sqrt(1 + rho'^2) dt.
    let (r, hh) = (0.8f64, 1.7f64);
    let h = 2.0 * hh;
    let n = 100_000;
    let mut lat = 0.0;
    for i in 0..n {
        let t = (i as f64 + 0.5) / n as f64 * h;
        let rho = r * (1.0 - t / h);
        let drho = -r / h;
        lat += 2.0 * std::f64::consts::PI * rho * (1.0 + drho * drho).sqrt() * (h / n as f64);
    }
    let want = std::f64::consts::PI * r * r + lat;
    let c = Cone::new(Vec3Fix::ZERO, fx(r), fx(hh));
    assert!(rel(c.surface_area().to_f64(), want) < 1e-8);
}

/// Numerical centroid (distance from the base along the axis) and inertia of the solid.
fn integrate(r: f64, hh: f64, m: f64) -> (f64, f64, f64) {
    let h = 2.0 * hh;
    let n = 200_000;
    let dt = h / n as f64;
    let (mut vol, mut mom) = (0.0, 0.0);
    for i in 0..n {
        let t = (i as f64 + 0.5) * dt;
        let rho = r * (1.0 - t / h);
        let dv = std::f64::consts::PI * rho * rho * dt;
        vol += dv;
        mom += t * dv;
    }
    let tc = mom / vol;
    let (mut iyy, mut ixx) = (0.0, 0.0);
    for i in 0..n {
        let t = (i as f64 + 0.5) * dt;
        let rho = r * (1.0 - t / h);
        let dm = m * std::f64::consts::PI * rho * rho * dt / vol;
        iyy += 0.5 * dm * rho * rho;
        ixx += dm * (0.25 * rho * rho + (t - tc) * (t - tc));
    }
    (tc, ixx, iyy)
}

#[test]
fn center_of_mass_offset_matches_integrated_centroid() {
    for &(r, hh) in &[(1.0, 1.0), (0.6, 2.2), (2.5, 0.4)] {
        let (tc, _, _) = integrate(r, hh, 1.0);
        let c = Cone::new(Vec3Fix::ZERO, fx(r), fx(hh));
        let off = arr(c.center_of_mass_offset());
        // centroid y (geometric centre = origin) = -hh + tc
        assert!(
            (off[1] - (-hh + tc)).abs() < 1e-6,
            "r={r} hh={hh} off={off:?}"
        );
        assert!(off[0].abs() < 1e-15 && off[2].abs() < 1e-15);
    }
}

#[test]
fn inertia_diagonal_matches_integration_about_com() {
    for &(r, hh, m) in &[(1.0, 1.0, 1.0), (0.6, 2.2, 3.5), (2.5, 0.4, 0.8)] {
        let (_, ixx, iyy) = integrate(r, hh, m);
        let c = Cone::new(Vec3Fix::ZERO, fx(r), fx(hh));
        let i = arr(c.inertia_diagonal(fx(m)));
        assert!(
            rel(i[0], ixx) < 1e-6,
            "Ixx {} vs {} (r={r} hh={hh})",
            i[0],
            ixx
        );
        assert!(rel(i[1], iyy) < 1e-6, "Iyy {} vs {}", i[1], iyy);
        assert!(rel(i[2], ixx) < 1e-6, "Izz {} vs {}", i[2], ixx);
    }
}

#[test]
fn inertia_is_linear_in_mass() {
    let c = Cone::new(Vec3Fix::ZERO, fx(0.9), fx(1.3));
    let a = arr(c.inertia_diagonal(fx(1.0)));
    let b = arr(c.inertia_diagonal(fx(4.0)));
    for i in 0..3 {
        assert!(rel(b[i], 4.0 * a[i]) < 1e-12);
    }
}

// ---------------------------------------------------------------- A: apex / base / support / aabb

#[test]
fn apex_and_base_follow_the_rotation() {
    let (r, hh) = (1.0, 1.5);
    let center = [0.5, -1.0, 2.0];
    let axis = [1.0, 2.0, -0.5];
    let ang = 0.9;
    let c = oriented(center, r, hh, axis, ang);
    let k = unit(axis);
    let ap = rot(k, ang, [0.0, hh, 0.0]);
    let ba = rot(k, ang, [0.0, -hh, 0.0]);
    let a = arr(c.apex());
    let b = arr(c.base_center());
    for i in 0..3 {
        assert!((a[i] - (center[i] + ap[i])).abs() < 1e-9);
        assert!((b[i] - (center[i] + ba[i])).abs() < 1e-9);
    }
}

#[test]
fn support_is_the_maximum_over_the_boundary() {
    // Property: for random directions and orientations, no sampled boundary point
    // beats the support point, and the support point itself lies on the boundary
    // (apex or on the base rim).
    let mut s = 12345u64;
    for case in 0..40 {
        let r = 0.5 + 1.5 * (lcg(&mut s) * 0.5 + 0.5);
        let hh = 0.5 + 1.5 * (lcg(&mut s) * 0.5 + 0.5);
        let center = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let axis = [lcg(&mut s) + 0.1, lcg(&mut s), lcg(&mut s)];
        let ang = lcg(&mut s) * 3.0;
        let cone = oriented(center, r, hh, axis, ang);
        let pts = boundary(center, r, hh, axis, ang);
        let d = [lcg(&mut s), lcg(&mut s), lcg(&mut s)];
        let sup = arr(cone.support(v3(d[0], d[1], d[2])));
        let best = pts.iter().map(|p| dot(*p, d)).fold(f64::MIN, f64::max);
        assert!(
            dot(sup, d) >= best - 1e-6,
            "case {case}: support {} < sampled max {}",
            dot(sup, d),
            best
        );
        // on the boundary: apex or base rim
        let k = unit(axis);
        let local = rot(
            k,
            -ang,
            [sup[0] - center[0], sup[1] - center[1], sup[2] - center[2]],
        );
        let is_apex =
            local[0].abs() < 1e-8 && (local[1] - hh).abs() < 1e-8 && local[2].abs() < 1e-8;
        let on_rim = (local[1] + hh).abs() < 1e-8
            && ((local[0] * local[0] + local[2] * local[2]).sqrt() - r).abs() < 1e-8;
        assert!(
            is_apex || on_rim,
            "case {case}: support {local:?} not on boundary"
        );
    }
}

#[test]
fn support_picks_apex_for_steep_up_and_rim_for_shallow() {
    // r = 2, hh = 1: apex (0,1,0), rim (+-2,-1,0). Direction (sin a, cos a): apex dot = cos a,
    // rim dot = 2 sin a - cos a; they tie at tan a = 1 (45 deg).
    let c = Cone::new(Vec3Fix::ZERO, fx(2.0), fx(1.0));
    let up = arr(c.support(v3(0.6, 0.8, 0.0)));
    assert!(up[0].abs() < 1e-12 && (up[1] - 1.0).abs() < 1e-12, "{up:?}");
    let side = arr(c.support(v3(0.8, 0.6, 0.0)));
    // 2*0.8 - 0.6 = 1.0 > apex 0.6 -> rim
    assert!(
        (side[0] - 2.0).abs() < 1e-9 && (side[1] + 1.0).abs() < 1e-12,
        "{side:?}"
    );
}

#[test]
fn support_is_scale_invariant_in_the_direction() {
    let cone = oriented([0.0, 0.0, 0.0], 1.0, 1.0, [0.0, 0.0, 1.0], 0.0);
    let d = [0.3, -0.4, 0.5];
    let base = arr(cone.support(v3(d[0], d[1], d[2])));
    for &s in &[1e-3, 1e3, 1e6] {
        let got = arr(cone.support(v3(d[0] * s, d[1] * s, d[2] * s)));
        for i in 0..3 {
            assert!(
                (got[i] - base[i]).abs() < 1e-6,
                "scale {s}: {got:?} vs {base:?}"
            );
        }
    }
}

#[test]
fn support_tiny_direction_still_picks_the_right_rim_point() {
    // Direction -X scaled to 1e-12: the true support is the rim point (-r, -hh, 0).
    let cone = Cone::new(Vec3Fix::ZERO, fx(1.0), fx(1.0));
    let s = arr(cone.support(v3(-1e-12, 0.0, 0.0)));
    assert!((s[0] + 1.0).abs() < 1e-9, "tiny -X direction gave {s:?}");
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-002: support() with |dir|=5e9 (xz_len_sq wraps in Fix128) returns x=1.953 instead of the rim radius 1.0"]
fn support_huge_direction_does_not_wrap() {
    // |d| = 5e9 : the squared XZ length (2.5e19) exceeds the i64 integer range of Fix128.
    let cone = Cone::new(Vec3Fix::ZERO, fx(1.0), fx(1.0));
    let s = arr(cone.support(v3(5.0e9, 0.0, 0.0)));
    assert!((s[0] - 1.0).abs() < 1e-6, "huge +X direction gave {s:?}");
}

#[test]
fn aabb_encloses_every_boundary_point() {
    let mut s = 777u64;
    for case in 0..30 {
        let r = 0.3 + 1.7 * (lcg(&mut s) * 0.5 + 0.5);
        let hh = 0.3 + 1.7 * (lcg(&mut s) * 0.5 + 0.5);
        let center = [lcg(&mut s) * 3.0, lcg(&mut s) * 3.0, lcg(&mut s) * 3.0];
        let axis = [lcg(&mut s) + 0.1, lcg(&mut s), lcg(&mut s)];
        let ang = lcg(&mut s) * 3.0;
        let cone = oriented(center, r, hh, axis, ang);
        let bb = cone.aabb();
        let (lo, hi) = (arr(bb.min), arr(bb.max));
        for p in boundary(center, r, hh, axis, ang) {
            for i in 0..3 {
                assert!(
                    p[i] >= lo[i] - 1e-9 && p[i] <= hi[i] + 1e-9,
                    "case {case}: axis {i} point {} outside [{}, {}]",
                    p[i],
                    lo[i],
                    hi[i]
                );
            }
        }
        // symmetric about the centre
        for i in 0..3 {
            assert!(((lo[i] + hi[i]) * 0.5 - center[i]).abs() < 1e-9);
        }
    }
}

#[test]
fn aabb_x_extent_of_upright_cone_is_the_radius() {
    let cone = Cone::new(v3(5.0, 0.0, 0.0), fx(2.0), fx(3.0));
    let bb = cone.aabb();
    assert!((bb.min.x.to_f64() - 3.0).abs() < 1e-12);
    assert!((bb.max.x.to_f64() - 7.0).abs() < 1e-12);
    assert!((bb.min.z.to_f64() + 2.0).abs() < 1e-12);
}

#[test]
fn zero_radius_degenerate_cone_is_a_segment() {
    let cone = Cone::new(Vec3Fix::ZERO, Fix128::ZERO, fx(1.0));
    assert_eq!(cone.volume().to_f64(), 0.0);
    // support in +X: apex dot 0, base point (0,-1,0) dot 0 : tie -> apex, on the axis
    let s = arr(cone.support(v3(1.0, 0.0, 0.0)));
    assert!(s[0].abs() < 1e-15);
}

#[test]
fn support_along_the_axis_is_a_point_of_the_rim_or_the_apex() {
    // Direction -Y: every rim point ties; the returned point must still lie on the rim.
    let cone = Cone::new(v3(1.0, 2.0, 3.0), fx(2.0), fx(1.0));
    let s = arr(cone.support(v3(0.0, -1.0, 0.0)));
    let dx = s[0] - 1.0;
    let dz = s[2] - 3.0;
    assert!((s[1] - 1.0).abs() < 1e-12, "y {}", s[1]);
    assert!(
        ((dx * dx + dz * dz).sqrt() - 2.0).abs() < 1e-9,
        "radial {dx} {dz}"
    );
}

#[test]
fn with_rotation_stores_every_field() {
    let q = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(0.5));
    let c = Cone::with_rotation(v3(1.0, 2.0, 3.0), fx(0.25), fx(0.75), q);
    assert_eq!(c.center, v3(1.0, 2.0, 3.0));
    assert_eq!(c.radius, fx(0.25));
    assert_eq!(c.half_height, fx(0.75));
    assert_eq!(c.rotation, q);
    let n = Cone::new(v3(1.0, 2.0, 3.0), fx(0.25), fx(0.75));
    assert_eq!(n.rotation, QuatFix::IDENTITY);
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-013: Cone::aabb is conservative, not tight: along the cone axis it adds the base radius to the half-height (upright r=2, hh=3 gives y in [-5, 5], the cone spans [-3, 3]); it encloses every point, but the inline test pins the loose value"]
fn aabb_of_an_upright_cone_is_tight_along_the_axis() {
    let cone = Cone::new(Vec3Fix::ZERO, fx(2.0), fx(3.0));
    let bb = cone.aabb();
    assert!(
        (bb.max.y.to_f64() - 3.0).abs() < 1e-9,
        "max y {}",
        bb.max.y.to_f64()
    );
    assert!((bb.min.y.to_f64() + 3.0).abs() < 1e-9);
}
