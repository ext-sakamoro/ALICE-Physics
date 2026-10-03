//! Audit S2-3 oracles for `alice_physics::interpolation`.
//! Expected values are closed forms in f64 (affine blend, NLERP/SLERP angle formulas).
#![allow(clippy::disallowed_methods)]

use alice_physics::interpolation::{
    lerp_fix128, lerp_vec3, slerp, BodySnapshot, InterpolationState, WorldSnapshot,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// Unit quaternion for a rotation by `deg` about Y.
fn roty(deg: f64) -> QuatFix {
    let h = deg.to_radians() / 2.0;
    QuatFix::new(fx(0.0), fx(h.sin()), fx(0.0), fx(h.cos()))
}

fn q64(q: QuatFix) -> [f64; 4] {
    [q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64()]
}

/// Rotation angle (deg) between two unit quaternions (shortest path), via the relative
/// rotation conj(a) * b so that it stays accurate near 0 (acos of the dot loses 1e-8).
fn angle_between(a: QuatFix, b: QuatFix) -> f64 {
    let (a, b) = (q64(a), q64(b));
    let (ax, ay, az, aw) = (-a[0], -a[1], -a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    let w = aw * bw - ax * bx - ay * by - az * bz;
    let x = aw * bx + ax * bw + ay * bz - az * by;
    let y = aw * by - ax * bz + ay * bw + az * bx;
    let z = aw * bz + ax * by - ay * bx + az * bw;
    (2.0 * (x * x + y * y + z * z).sqrt().atan2(w.abs())).to_degrees()
}

#[test]
fn lerp_is_affine_and_exact_on_dyadic_inputs() {
    let a = Fix128::from_int(-3);
    let b = Fix128::from_int(13);
    for (n, d) in [
        (0, 1),
        (1, 8),
        (1, 4),
        (1, 2),
        (3, 4),
        (7, 8),
        (1, 1),
        (-1, 4),
        (5, 4),
    ] {
        let t = Fix128::from_ratio(n, d);
        let want = -3.0 + 16.0 * (n as f64 / d as f64);
        assert_eq!(lerp_fix128(a, b, t).to_f64(), want, "t = {n}/{d}");
    }
    let va = Vec3Fix::new(a, Fix128::ZERO, b);
    let vb = Vec3Fix::new(b, Fix128::from_int(8), a);
    let v = lerp_vec3(va, vb, Fix128::from_ratio(1, 4));
    assert_eq!((v.x.to_f64(), v.y.to_f64(), v.z.to_f64()), (1.0, 2.0, 9.0));
}

fn state_with(prev: [f64; 3], curr: [f64; 3], rp: QuatFix, rc: QuatFix) -> InterpolationState {
    let mk = |p: [f64; 3], r: QuatFix| WorldSnapshot {
        bodies: vec![BodySnapshot {
            position: Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2])),
            rotation: r,
            velocity: Vec3Fix::ZERO,
        }],
    };
    InterpolationState::new(mk(prev, rp), mk(curr, rc))
}

/// alpha = 0 is the previous state, alpha = 1 the current one, per component (distinct values
/// on every axis so an axis or prev/curr swap shows).
#[test]
fn position_blend_endpoints_and_quarter_point_per_axis() {
    let s = state_with([1.0, -2.0, 4.0], [5.0, 6.0, -4.0], roty(0.0), roty(0.0));
    let p0 = s.interpolate_position(0, Fix128::ZERO);
    assert_eq!(
        (p0.x.to_f64(), p0.y.to_f64(), p0.z.to_f64()),
        (1.0, -2.0, 4.0)
    );
    let p1 = s.interpolate_position(0, Fix128::ONE);
    assert_eq!(
        (p1.x.to_f64(), p1.y.to_f64(), p1.z.to_f64()),
        (5.0, 6.0, -4.0)
    );
    let q = s.interpolate_position(0, Fix128::from_ratio(1, 4));
    assert_eq!((q.x.to_f64(), q.y.to_f64(), q.z.to_f64()), (2.0, 0.0, 2.0));
}

/// Index out of range for either snapshot -> ZERO / IDENTITY; the boundary index (== len) too.
#[test]
fn out_of_range_index_gives_zero_and_identity_at_the_boundary() {
    let s = state_with([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], roty(10.0), roty(20.0));
    let half = Fix128::from_ratio(1, 2);
    assert_eq!(s.interpolate_position(1, half), Vec3Fix::ZERO);
    assert_eq!(s.interpolate_rotation(1, half), QuatFix::IDENTITY);
    assert_eq!(s.interpolate_position(99, half), Vec3Fix::ZERO);
    assert_ne!(s.interpolate_position(0, half), Vec3Fix::ZERO);
    // one snapshot shorter than the other: the shorter bound applies on either side
    let mut a = state_with([1.0; 3], [2.0; 3], roty(0.0), roty(0.0));
    a.prev.bodies.clear();
    assert_eq!(a.body_count(), 0);
    assert_eq!(a.interpolate_position(0, half), Vec3Fix::ZERO);
    let mut b = state_with([1.0; 3], [2.0; 3], roty(0.0), roty(0.0));
    b.current.bodies.clear();
    assert_eq!(b.body_count(), 0);
    assert_eq!(b.interpolate_position(0, half), Vec3Fix::ZERO);
    assert_eq!(b.interpolate_rotation(0, half), QuatFix::IDENTITY);
    assert!(b.interpolate_all(half).is_empty());
}

/// The result of slerp is always a unit quaternion, also for non-unit inputs (drifted state).
#[test]
fn slerp_result_is_unit_even_for_unnormalised_inputs() {
    let a = QuatFix::new(fx(0.0), fx(0.0), fx(0.0), fx(2.0));
    let b = QuatFix::new(fx(0.0), fx(3.0), fx(0.0), fx(3.0));
    for k in 0..=10 {
        let t = fx(f64::from(k) / 10.0);
        let q = q64(slerp(a, b, t));
        let n: f64 = q.iter().map(|c| c * c).sum();
        assert!((n - 1.0).abs() < 1e-12, "t = {k}/10: |q|^2 = {n}");
    }
}

/// Shortest path: with dot(a, b) < 0 the result stays near a (dot(a, result) >= 0 for all t),
/// and t = 1 reaches b's rotation (as +b or -b) by the short way.
#[test]
fn slerp_takes_the_short_way_for_negative_dot() {
    let a = roty(10.0);
    let b = roty(60.0);
    let neg_b = QuatFix::new(-b.x, -b.y, -b.z, -b.w);
    for k in 0..=10 {
        let t = fx(f64::from(k) / 10.0);
        let direct = q64(slerp(a, b, t));
        let flipped = q64(slerp(a, neg_b, t));
        for i in 0..4 {
            assert!((direct[i] - flipped[i]).abs() < 1e-12, "t {k} comp {i}");
        }
        let qa = q64(a);
        let dot: f64 = (0..4).map(|i| qa[i] * flipped[i]).sum();
        assert!(dot >= 0.0);
    }
    let end = slerp(a, neg_b, Fix128::ONE);
    assert!(angle_between(end, b) < 1e-9);
}

/// Midpoint of two rotations about the same axis is the mean angle (exact for NLERP by symmetry),
/// and t = 0 / t = 1 reproduce the end rotations.
#[test]
fn slerp_midpoint_and_endpoints_about_a_common_axis() {
    let (a, b) = (roty(20.0), roty(100.0));
    assert!(angle_between(slerp(a, b, Fix128::ZERO), a) < 1e-9);
    assert!(angle_between(slerp(a, b, Fix128::ONE), b) < 1e-9);
    let mid = slerp(a, b, Fix128::from_ratio(1, 2));
    assert!(
        angle_between(mid, roty(60.0)) < 1e-9,
        "{}",
        angle_between(mid, roty(60.0))
    );
}

/// Symmetry: slerp(a, b, t) == slerp(b, a, 1 - t) as rotations.
#[test]
fn slerp_is_symmetric_under_swap() {
    let (a, b) = (roty(-30.0), roty(75.0));
    for (n, d) in [(1, 8), (1, 4), (3, 8), (5, 8)] {
        let t = Fix128::from_ratio(n, d);
        let x = slerp(a, b, t);
        let y = slerp(b, a, Fix128::ONE - t);
        assert!(angle_between(x, y) < 1e-9);
    }
}

/// Doc: `slerp` / "interpolate_rotation (SLERP)". True SLERP has constant angular velocity:
/// a 90 degree arc at t = 1/4 is exactly 22.5 degrees. The NLERP actually used gives ~21.6.
#[test]
#[ignore = "known defect: AUD-A-S2W3-006: slerp is NLERP; 90 deg arc at t=1/4 is 21.6 deg not 22.5 (angular speed non-uniform, 4% at 90 deg, growing with the arc); doc discloses NLERP but names/describes it slerp"]
fn slerp_has_constant_angular_velocity() {
    let a = roty(0.0);
    let b = roty(90.0);
    for k in 1..8 {
        let t = f64::from(k) / 8.0;
        let got = angle_between(a, slerp(a, b, fx(t)));
        assert!(
            (got - 90.0 * t).abs() < 0.05,
            "t = {t}: {got} deg vs {}",
            90.0 * t
        );
    }
}

/// NLERP closed form (what the implementation promises in its doc): the angle at t is
/// 2 atan( t sin(th/2) / ((1-t) + t cos(th/2)) ). Pinning it checks the weights and normalisation.
#[test]
fn slerp_matches_the_nlerp_angle_formula() {
    for theta in [30.0f64, 90.0, 150.0] {
        let a = roty(0.0);
        let b = roty(theta);
        for t in [0.125f64, 0.25, 0.5, 0.75] {
            let h = theta.to_radians() / 2.0;
            let want = (2.0 * (t * h.sin() / ((1.0 - t) + t * h.cos())).atan()).to_degrees();
            let got = angle_between(a, slerp(a, b, fx(t)));
            assert!(
                (got - want).abs() < 1e-8,
                "theta {theta} t {t}: {got} vs {want}"
            );
        }
    }
}

/// Monotone in t (never moves backwards along the arc).
#[test]
fn slerp_angle_is_monotone_in_t() {
    let a = roty(0.0);
    let b = roty(170.0);
    let mut prev = -1.0;
    for k in 0..=16 {
        let g = angle_between(a, slerp(a, b, fx(f64::from(k) / 16.0)));
        assert!(g >= prev - 1e-9, "k {k}: {g} < {prev}");
        prev = g;
    }
}

/// `BodySnapshot::velocity` is documented "for extrapolation" but the module offers no
/// extrapolation: the interpolated position ignores the velocity entirely.
#[test]
fn snapshot_velocity_does_not_influence_interpolated_pose() {
    let mut s = state_with([0.0; 3], [4.0, 0.0, 0.0], roty(0.0), roty(0.0));
    let base = s.interpolate(0, Fix128::from_ratio(1, 2));
    s.current.bodies[0].velocity = Vec3Fix::from_int(100, 100, 100);
    s.prev.bodies[0].velocity = Vec3Fix::from_int(-7, 8, 9);
    assert_eq!(s.interpolate(0, Fix128::from_ratio(1, 2)), base);
}

/// capture() copies pose and velocity per body in index order; `from_body` is a pure copy.
#[test]
fn capture_preserves_order_and_all_three_fields() {
    use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let mut b0 = RigidBody::new_dynamic(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);
    b0.velocity = Vec3Fix::from_int(4, 5, 6);
    b0.rotation = roty(33.0);
    let b1 = RigidBody::new_static(Vec3Fix::from_int(-1, -2, -3));
    w.add_body(b0);
    w.add_body(b1);
    let snap = WorldSnapshot::capture(&w);
    assert_eq!(snap.len(), 2);
    assert_eq!(snap.bodies[0].position, Vec3Fix::from_int(1, 2, 3));
    assert_eq!(snap.bodies[0].velocity, Vec3Fix::from_int(4, 5, 6));
    assert_eq!(snap.bodies[0].rotation, roty(33.0));
    assert_eq!(snap.bodies[1].position, Vec3Fix::from_int(-1, -2, -3));
    assert!(!snap.is_empty());
    assert!(WorldSnapshot { bodies: vec![] }.is_empty());
}

/// `push` keeps exactly two generations: after three pushes prev is the second, current the third.
#[test]
fn push_keeps_two_generations() {
    let snap = |x: f64| WorldSnapshot {
        bodies: vec![BodySnapshot {
            position: Vec3Fix::new(fx(x), Fix128::ZERO, Fix128::ZERO),
            rotation: QuatFix::IDENTITY,
            velocity: Vec3Fix::ZERO,
        }],
    };
    let mut s = InterpolationState::empty();
    s.push(snap(1.0));
    s.push(snap(2.0));
    s.push(snap(3.0));
    assert_eq!(s.prev.bodies[0].position.x.to_f64(), 2.0);
    assert_eq!(s.current.bodies[0].position.x.to_f64(), 3.0);
    // alpha 1/2 -> 2.5
    assert_eq!(
        s.interpolate_position(0, Fix128::from_ratio(1, 2))
            .x
            .to_f64(),
        2.5
    );
}

/// `interpolate_all` returns one transform per body in order, equal to `interpolate(i)`.
#[test]
fn interpolate_all_matches_per_body_interpolate_in_order() {
    let mk = |off: f64, deg: f64| WorldSnapshot {
        bodies: (0..3)
            .map(|i| BodySnapshot {
                position: Vec3Fix::new(
                    fx(off + f64::from(i)),
                    fx(f64::from(i) * 2.0),
                    Fix128::ZERO,
                ),
                rotation: roty(deg + 10.0 * f64::from(i)),
                velocity: Vec3Fix::ZERO,
            })
            .collect(),
    };
    let s = InterpolationState::new(mk(0.0, 0.0), mk(8.0, 40.0));
    let all = s.interpolate_all(Fix128::from_ratio(1, 4));
    assert_eq!(all.len(), 3);
    for (i, got) in all.iter().enumerate() {
        assert_eq!(*got, s.interpolate(i, Fix128::from_ratio(1, 4)));
        assert_eq!(got.0.x.to_f64(), 2.0 + i as f64);
        let want = 10.0 * i as f64 + 10.0;
        assert!(
            angle_between(got.1, roty(want)) < 0.5 * 40.0 / 40.0,
            "body {i}"
        );
    }
}

fn rot_axis(axis: [f64; 3], deg: f64) -> QuatFix {
    let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let h = deg.to_radians() / 2.0;
    let s = h.sin() / l;
    QuatFix::new(
        fx(axis[0] * s),
        fx(axis[1] * s),
        fx(axis[2] * s),
        fx(h.cos()),
    )
}

/// The shortest-path flip must negate all four components: for general (all-components-non-zero)
/// quaternions, slerp(a, -b, t) is the same rotation as slerp(a, b, t) at every t.
#[test]
fn slerp_flip_is_component_complete_for_general_axes() {
    let a = rot_axis([2.0, -1.0, 0.5], 25.0);
    for arc_deg in [20.0, 100.0, 150.0] {
        let b = rot_axis([1.0, 2.0, 3.0], arc_deg);
        let nb = QuatFix::new(-b.x, -b.y, -b.z, -b.w);
        for k in 0..=8 {
            let t = fx(f64::from(k) / 8.0);
            let p = slerp(a, b, t);
            let q = slerp(a, nb, t);
            assert!(
                angle_between(p, q) < 1e-6,
                "arc {arc_deg} k {k}: {}",
                angle_between(p, q)
            );
            let (pv, qv) = (q64(p), q64(q));
            let dot: f64 = (0..4).map(|i| pv[i] * qv[i]).sum();
            assert!(dot > 0.0, "same hemisphere, not just the same rotation");
        }
    }
}

/// The flip decision uses the full 4-D dot product: from +128 to -128 degrees about Y the dot
/// is negative only because of the y components (ay*by = -0.81, aw*bw = +0.19); the short way
/// passes through 180 degrees, the long way through 0.
#[test]
fn slerp_flip_decision_uses_every_component_of_the_dot() {
    let a = roty(128.0);
    let b = roty(-128.0);
    let mid = slerp(a, b, Fix128::from_ratio(1, 2));
    assert!(
        angle_between(mid, roty(180.0)) < 1e-6,
        "{}",
        angle_between(mid, roty(180.0))
    );
}

/// slerp of a quaternion with itself returns that quaternion including its sign, also when the
/// vector part is zero (-identity must not collapse to +identity).
#[test]
fn slerp_of_equal_inputs_preserves_the_representation() {
    let m = QuatFix::new(fx(0.0), fx(0.0), fx(0.0), fx(-1.0));
    for k in 0..=4 {
        let out = slerp(m, m, fx(f64::from(k) / 4.0));
        assert_eq!(q64(out), [0.0, 0.0, 0.0, -1.0], "k {k}");
    }
}
