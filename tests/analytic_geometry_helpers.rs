//! Oracles for the small geometry queries on the collider primitives: the corners
//! of an oriented box, the apex and base of a cone, the box of a metric ball.
//!
//! The expectations are written from the geometry, with the rotation done in `f64`
//! by Rodrigues' formula (the code under test rotates with a fixed-point
//! quaternion), and the metric ball's box is checked against a brute-force sample
//! of the ball's boundary.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Support, AABB};
use alice_physics::cone::Cone;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
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

/// `v` turned by `angle` about the unit `axis` (Rodrigues).
fn rodrigues(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    let (s, c) = angle.sin_cos();
    let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
    ]
}

fn close(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() <= tol)
}

/// The eight sign combinations, in the order the box documents: bit 0 flips x, bit
/// 1 flips y, bit 2 flips z.
fn sign_corner(h: [f64; 3], index: usize) -> [f64; 3] {
    [
        if index & 1 == 0 { h[0] } else { -h[0] },
        if index & 2 == 0 { h[1] } else { -h[1] },
        if index & 4 == 0 { h[2] } else { -h[2] },
    ]
}

/// An axis-aligned box is an identity-rotated one, and its corners are `centre ± h`
/// in every sign combination, in index order.
#[test]
fn an_axis_aligned_box_has_the_corners_centre_plus_minus_half_extents() {
    let b = OrientedBox::axis_aligned(v3(1.0, -2.0, 3.0), v3(0.5, 1.5, 2.5));
    assert_eq!(b.rotation, QuatFix::IDENTITY);
    assert_eq!(
        b,
        OrientedBox::new(v3(1.0, -2.0, 3.0), v3(0.5, 1.5, 2.5), QuatFix::IDENTITY)
    );
    let corners = b.corners();
    for (i, c) in corners.iter().enumerate() {
        let h = sign_corner([0.5, 1.5, 2.5], i);
        let want = [1.0 + h[0], -2.0 + h[1], 3.0 + h[2]];
        assert!(close(arr(*c), want, 1e-12), "corner {i}: {:?}", arr(*c));
        assert_eq!(*c, b.corner(i), "corners()[{i}] is corner({i})");
    }
}

/// A turned box: each corner is the centre plus the turned half-extent vector.
#[test]
fn a_turned_box_has_the_turned_corners() {
    let centre = [0.5, 1.0, -1.5];
    let h = [1.0, 2.0, 0.5];
    let axis = [1.0 / 2f64.sqrt(), 1.0 / 2f64.sqrt(), 0.0];
    let angle = 0.7;
    let q = QuatFix::from_axis_angle(v3(axis[0], axis[1], axis[2]), fx(angle));
    let b = OrientedBox::new(v3(centre[0], centre[1], centre[2]), v3(h[0], h[1], h[2]), q);
    for i in 0..8 {
        let r = rodrigues(sign_corner(h, i), axis, angle);
        let want = [centre[0] + r[0], centre[1] + r[1], centre[2] + r[2]];
        assert!(
            close(arr(b.corner(i)), want, 1e-9),
            "corner {i}: {:?} but the turned corner is {want:?}",
            arr(b.corner(i))
        );
    }
}

/// The box's AABB is the box of its corners (the AABB is computed from the
/// rotated axes, the corners from the rotated corners).
#[test]
fn the_aabb_of_a_turned_box_is_the_box_of_its_corners() {
    let q = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(0.6));
    let b = OrientedBox::new(v3(2.0, 0.0, 1.0), v3(1.0, 0.5, 0.25), q);
    let corners = b.corners().map(arr);
    let aabb: AABB = b.aabb();
    for k in 0..3 {
        let lo = corners.iter().map(|c| c[k]).fold(f64::MAX, f64::min);
        let hi = corners.iter().map(|c| c[k]).fold(f64::MIN, f64::max);
        assert!((arr(aabb.min)[k] - lo).abs() < 1e-9, "min axis {k}");
        assert!((arr(aabb.max)[k] - hi).abs() < 1e-9, "max axis {k}");
    }
}

/// An upright cone: the apex is `half_height` above the centre, the base centre
/// `half_height` below.
#[test]
fn an_upright_cone_has_its_apex_above_and_its_base_below() {
    let c = Cone::new(v3(1.0, 2.0, 3.0), fx(0.75), fx(1.25));
    assert!(close(arr(c.apex()), [1.0, 3.25, 3.0], 1e-12));
    assert!(close(arr(c.base_center()), [1.0, 0.75, 3.0], 1e-12));
}

/// A turned cone: apex and base centre are the centre plus/minus the turned
/// `(0, h, 0)`, the apex is the farthest point along the axis, and the base circle
/// is perpendicular to it.
#[test]
fn a_turned_cone_has_the_turned_apex_and_base() {
    let centre = [0.5, -1.0, 2.0];
    let (radius, h) = (0.6, 1.5);
    let axis = [0.0, 0.0, 1.0];
    let angle = std::f64::consts::FRAC_PI_2;
    let q = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), fx(angle));
    let cone = Cone::with_rotation(v3(centre[0], centre[1], centre[2]), fx(radius), fx(h), q);
    let up = rodrigues([0.0, h, 0.0], axis, angle);
    let apex = [centre[0] + up[0], centre[1] + up[1], centre[2] + up[2]];
    let base = [centre[0] - up[0], centre[1] - up[1], centre[2] - up[2]];
    assert!(
        close(arr(cone.apex()), apex, 1e-9),
        "apex {:?}",
        arr(cone.apex())
    );
    assert!(close(arr(cone.base_center()), base, 1e-9));
    // Along the axis the apex is the support point, the base circle's centre is
    // the middle of the opposite face.
    let along = cone.apex() - cone.base_center();
    assert!(close(arr(cone.support(along)), arr(cone.apex()), 1e-9));
    let mid = (cone.apex() + cone.base_center()) * Fix128::from_ratio(1, 2);
    assert!(close(arr(mid), centre, 1e-9), "the centre is midway");
}

/// The box of a metric ball: half-width `r / (w₁ + w₂ + w∞)` on each axis,
/// centred on the ball, and tight (an axis point of the ball is on its boundary;
/// no boundary point lies outside the box).
#[test]
fn the_box_of_a_metric_ball_is_tight() {
    let r = 2.0;
    let centre = [1.0, -1.0, 0.5];
    let metrics = [
        ("L1", MetricWeights::L1, [1.0, 0.0, 0.0]),
        ("L2", MetricWeights::L2, [0.0, 1.0, 0.0]),
        ("LINF", MetricWeights::LINF, [0.0, 0.0, 1.0]),
        (
            "mixed",
            MetricWeights::new(fx(1.0), fx(2.0), fx(3.0)).expect("a convex metric"),
            [1.0, 2.0, 3.0],
        ),
    ];
    for (name, metric, w) in metrics {
        let aabb = AABB::from_metric_ball(v3(centre[0], centre[1], centre[2]), fx(r), metric);
        let e = r / (w[0] + w[1] + w[2]);
        for k in 0..3 {
            assert!(
                (arr(aabb.min)[k] - (centre[k] - e)).abs() < 1e-9
                    && (arr(aabb.max)[k] - (centre[k] + e)).abs() < 1e-9,
                "{name}: axis {k} is [{}, {}], the ball reaches {e} each side",
                arr(aabb.min)[k],
                arr(aabb.max)[k]
            );
        }
        // g(x) = w₁‖x‖₁ + w₂‖x‖₂ + w∞‖x‖∞. An axis point on the box face is on the
        // boundary of the ball ...
        let g = |x: [f64; 3]| {
            let l1 = x.iter().map(|c| c.abs()).sum::<f64>();
            let l2 = x.iter().map(|c| c * c).sum::<f64>().sqrt();
            let li = x.iter().map(|c| c.abs()).fold(0.0, f64::max);
            w[0] * l1 + w[1] * l2 + w[2] * li
        };
        assert!(
            (g([e, 0.0, 0.0]) - r).abs() < 1e-9,
            "{name}: the axis point"
        );
        // ... and no point of the ball is outside the box: scale many directions
        // to the boundary and look at the largest coordinate.
        let mut widest = 0.0f64;
        let n = 40;
        for i in 0..n {
            for j in 0..n {
                let (u, v) = (
                    std::f64::consts::PI * (i as f64 + 0.5) / n as f64,
                    2.0 * std::f64::consts::PI * j as f64 / n as f64,
                );
                let d = [u.sin() * v.cos(), u.sin() * v.sin(), u.cos()];
                let scale = r / g(d);
                widest = widest.max(d.iter().map(|c| (c * scale).abs()).fold(0.0, f64::max));
            }
        }
        assert!(
            widest <= e + 1e-9,
            "{name}: a boundary point reaches {widest}, outside the box half-width {e}"
        );
    }
}
