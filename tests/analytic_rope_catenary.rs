//! Closed-form oracle for a hanging rope against the classical catenary
//! (COV-SOFT-087). `tests/integration_physics.rs::test_rope_sag_under_gravity`
//! only checks the sag direction; nothing compares node positions to the
//! catenary curve (see docs/coverage/soft.toml COV-SOFT-087 evidence before
//! this test landed).
//!
//! A uniform chain of length `L` pinned at two points a span `D < L` apart
//! settles to `y(x) = a (cosh(x/a) - cosh(D/(2a)))` (y=0 at the pins,
//! x=0 at the lowest point, symmetric about x=0), where `a` solves
//! `a sinh(D/(2a)) = L/2` (Irvine, Cable Structures ch.1). Particles are
//! spaced at equal *rest length* (equal arc length along the curve), not
//! equal x, since `Rope::new` gives every segment the same rest length.
//!
//! `Rope::new`'s total particle mass is `mpu * L * (N+1)/N`, not `mpu * L`
//! (known defect AUD-A-S3W1-003, tests/audit_rope.rs) -- every particle is
//! off by the same constant factor, though, and a catenary's *shape* for a
//! chain of given length between given pins does not depend on the
//! absolute mass (only uniformity along the chain, which this still is),
//! so that defect does not affect this oracle.

use alice_physics::det_math;
use alice_physics::rope::PinConstraint;
use alice_physics::{Fix128, Rope, Vec3Fix};

fn ffx(v: f32) -> Fix128 {
    Fix128::from_f64(v as f64)
}

fn sinh(x: f32) -> f32 {
    (det_math::exp(x) - det_math::exp(-x)) / 2.0
}

fn cosh(x: f32) -> f32 {
    (det_math::exp(x) + det_math::exp(-x)) / 2.0
}

fn asinh(z: f32) -> f32 {
    det_math::ln(z + (z * z + 1.0).sqrt())
}

/// Solves `a * sinh(half_span / a) = half_length` for `a > 0` by bisection;
/// `f(a) = a sinh(half_span/a) - half_length` is monotonically decreasing
/// in `a` (a -> 0+ sends it to +inf, a -> inf sends it to half_span -
/// half_length < 0 since half_length > half_span here), so any bracket
/// with f(lo) > 0 > f(hi) converges.
fn solve_catenary_a(half_span: f32, half_length: f32) -> f32 {
    let f = |a: f32| a * sinh(half_span / a) - half_length;
    let mut lo = 0.1_f32;
    let mut hi = 100.0_f32;
    assert!(
        f(lo) > 0.0 && f(hi) < 0.0,
        "bracket does not straddle the root"
    );
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Expected (x, y) of the point at arc length `s` from the left pin
/// (x = -half_span), y = 0 at pin height.
fn catenary_point(a: f32, half_span: f32, s_from_left_pin: f32) -> (f32, f32) {
    // s(x) = a sinh(x/a), s(-half_span) = -half_length by construction of a;
    // x such that s(x) = s(-half_span) + s_from_left_pin.
    let s_at_x = -a * sinh(half_span / a) + s_from_left_pin;
    let x = a * asinh(s_at_x / a);
    let y = a * cosh(x / a) - a * cosh(half_span / a);
    (x, y)
}

#[test]
fn hanging_rope_settles_to_the_catenary_shape() {
    let span = 6.0_f32; // distance between pins
    let length = 8.0_f32; // total rope length, slack
    let num_segments = 20;
    let half_span = span / 2.0;
    let half_length = length / 2.0;
    let a = solve_catenary_a(half_span, half_length);

    // Build at the rope's own full length so segment rest lengths sum to
    // `length`, then pin the ends closer together (span < length) to put
    // the chain under slack instead of taut.
    let mut rope = Rope::new(
        Vec3Fix::new(ffx(-half_length), Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(ffx(half_length), Fix128::ZERO, Fix128::ZERO),
        num_segments,
        Fix128::ONE,
    );
    let last = rope.particle_count() - 1;
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target: Vec3Fix::new(ffx(-half_span), Fix128::ZERO, Fix128::ZERO),
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
    rope.add_pin(PinConstraint {
        particle_index: last,
        target: Vec3Fix::new(ffx(half_span), Fix128::ZERO, Fix128::ZERO),
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
    // Start the free particles roughly along the sagging shape instead of
    // the straight line `Rope::new` lays down at the full (slack) length,
    // so the solver settles from a reasonable initial guess rather than a
    // badly stretched initial configuration.
    for i in 0..rope.particle_count() {
        let s = (i as f32 / num_segments as f32) * length;
        let (x, y) = catenary_point(a, half_span, s);
        rope.positions[i] = Vec3Fix::new(ffx(x), ffx(y), Fix128::ZERO);
        rope.prev_positions[i] = rope.positions[i];
    }

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..600 {
        // 10 s, settle to static equilibrium
        rope.step(dt);
    }

    let segment_rest_length = length / num_segments as f32;
    let mut max_err = 0.0_f32;
    for i in 0..rope.particle_count() {
        let s = i as f32 * segment_rest_length;
        let (x_expected, y_expected) = catenary_point(a, half_span, s);
        let got = rope.positions[i];
        let dx = got.x.to_f64() as f32 - x_expected;
        let dy = got.y.to_f64() as f32 - y_expected;
        let err = (dx * dx + dy * dy).sqrt();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 0.05,
        "max node position error vs catenary closed form: {max_err:.4} (span {span}, length {length}, a={a:.4})"
    );
}
