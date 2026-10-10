//! A cylinder's and a cone's support point for a direction along the solid's
//! axis, turned by many rotations: the point on the axis (the centre of the cap,
//! the apex) or, for the cone's base, the base point the mapping takes for a
//! direction with no part across the axis, exactly as before the support
//! mappings scaled up a short part across the axis; and a direction a little
//! off the axis gets the rim point.
//!
//! # Expected values
//!
//! The direction is the turned axis, in `f64` (as a caller would give it) or
//! turned by the solid's own quaternion: its part across the axis is the
//! rounding of the turn, below the bound of `Vec3Fix::rescaled_xz`. The
//! expected point is the closed form: the centre plus the turned local point
//! `(0, ±h, 0)` (cylinder), `(0, h, 0)` (apex) or `(r, −h, 0)` (the base point
//! the mapping takes for a direction straight along `−Y`). Off the axis by
//! `θ` rad, the support is the rim point toward the direction across the axis,
//! whose support value exceeds the axis point's by `r·sin θ`.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Support;
use alice_physics::cone::Cone;
use alice_physics::cylinder::Cylinder;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(p: [f64; 3]) -> Vec3Fix {
    Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2]))
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

struct Rng(u64);

impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.u()
    }
}

/// `v` turned by the unit quaternion `q = (w, x, y, z)`, in `f64`.
fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

/// The rotations: about the coordinate axes by multiples of `π/12` and about
/// random axes by random angles, all built as a caller would (axis and angle in
/// `f64`).
fn rotations() -> Vec<QuatFix> {
    let mut out = Vec::new();
    for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        for k in -12..=12 {
            let angle = std::f64::consts::PI * f64::from(k) / 12.0;
            out.push(QuatFix::from_axis_angle(v3(axis), fx(angle)));
        }
    }
    let mut rng = Rng(0x5a1e_a715);
    for _ in 0..400 {
        let a = [rng.r(-1.0, 1.0), rng.r(-1.0, 1.0), rng.r(-1.0, 1.0)];
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        if l < 0.1 {
            continue;
        }
        let axis = [a[0] / l, a[1] / l, a[2] / l];
        out.push(QuatFix::from_axis_angle(v3(axis), fx(rng.r(-3.1, 3.1))));
    }
    out
}

/// Directions along the turned axis `±Y` (the turn in `f64` and by the
/// quaternion), at several lengths.
fn axial(q: QuatFix, sign: f64) -> Vec<Vec3Fix> {
    let qf = [q.w.to_f64(), q.x.to_f64(), q.y.to_f64(), q.z.to_f64()];
    let mut out = Vec::new();
    for scale in [1.0, 3.0, 1.0 / 1024.0, 4096.0] {
        out.push(v3(rotate(qf, [0.0, sign * scale, 0.0])));
        out.push(q.rotate_vec(v3([0.0, sign * scale, 0.0])));
    }
    out
}

#[test]
fn a_cylinder_takes_its_cap_centre_for_a_direction_along_its_axis() {
    let mut n = 0;
    for q in rotations() {
        let c = Cylinder::with_rotation(v3([0.3, -1.2, 2.5]), fx(0.7), fx(0.9), q);
        for (sign, hh) in [(1.0, 0.7), (-1.0, -0.7)] {
            let want = c.center + q.rotate_vec(Vec3Fix::new(Fix128::ZERO, fx(hh), Fix128::ZERO));
            for d in axial(q, sign) {
                let got = c.support(d);
                assert_eq!(
                    got,
                    want,
                    "{q:?} {:?}: {:?} != {:?}",
                    f3(d),
                    f3(got),
                    f3(want)
                );
                n += 1;
            }
        }
    }
    assert!(n > 3000, "{n} directions");
}

#[test]
fn a_cone_takes_its_apex_or_its_base_point_for_a_direction_along_its_axis() {
    let mut n = 0;
    for q in rotations() {
        let c = Cone::with_rotation(v3([0.3, -1.2, 2.5]), fx(1.0), fx(0.8), q);
        let apex = c.center + q.rotate_vec(Vec3Fix::new(Fix128::ZERO, fx(0.8), Fix128::ZERO));
        let base = c.center + q.rotate_vec(Vec3Fix::new(fx(1.0), fx(-0.8), Fix128::ZERO));
        for (sign, want) in [(1.0, apex), (-1.0, base)] {
            for d in axial(q, sign) {
                let got = c.support(d);
                assert_eq!(
                    got,
                    want,
                    "{q:?} {:?}: {:?} != {:?}",
                    f3(d),
                    f3(got),
                    f3(want)
                );
                n += 1;
            }
        }
    }
    assert!(n > 3000, "{n} directions");
}

/// Off the axis by `θ` (from `2⁻²⁰` to `2⁻⁸` rad, the directions of a contact
/// near a rim), the support is the rim point toward the part across the axis:
/// its support value exceeds the axis point's by `r·sin θ`, to `2⁻⁴⁰`.
#[test]
fn a_direction_a_little_off_the_axis_gets_the_rim_point() {
    let mut rng = Rng(0x0ff_a715);
    let r = 0.9;
    for q in rotations().into_iter().step_by(7) {
        let qf = [q.w.to_f64(), q.x.to_f64(), q.y.to_f64(), q.z.to_f64()];
        let c = Cylinder::with_rotation(Vec3Fix::ZERO, fx(0.7), fx(r), q);
        for e in [20, 16, 12, 8] {
            let theta = 2f64.powi(-e);
            let phi = rng.r(0.0, std::f64::consts::TAU);
            let local = [
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ];
            let d = rotate(qf, local);
            let got = f3(c.support(v3(d)));
            let value = got[0] * d[0] + got[1] * d[1] + got[2] * d[2];
            // oracle: the rim point (r cos φ, 0.7, r sin φ) dotted with the local direction
            let want = r * theta.sin() + 0.7 * theta.cos();
            assert!(
                (value - want).abs() < 2f64.powi(-40),
                "θ = 2^-{e}: support value {value} != {want}"
            );
        }
    }
}
