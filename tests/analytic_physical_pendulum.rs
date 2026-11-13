//! A body hung from a static anchor by a ball joint is a physical pendulum.
//!
//! oracle (closed form): a rigid body of mass `m` and moment of inertia `I` about its centre,
//! pivoting about a point at distance `L` from its centre, swings with the period
//! `T(θ₀) = 4·√(I_p / (m·g·L))·K(sin(θ₀/2))`, `I_p = I + m·L²` (parallel axis), `K` the
//! complete elliptic integral of the first kind (computed here with the arithmetic-geometric
//! mean, `K(k) = π / (2·AGM(1, √(1 − k²)))`, independent of the crate). The body is
//! `RigidBody::new_dynamic` (a unit sphere's inertia, `I = 2/5·m`), `m = 1`, `L = 1`,
//! `g = 10`, so `I_p = 7/5`.
//!
//! Also: the total energy `½·m·v² + ½·I·ω² + m·g·y` never rises above its starting value, and
//! the solver's loss of it is first order in the substep (XPBD's position projection
//! dissipates): it falls by about a factor 4 for 4 times as many substeps.
//!
//! Measured (1200 frames of 1/60 s): at θ₀ = 0.1 the period error against the closed form is
//! 5.4e-5 / 2.0e-5 / 4.1e-6 at 8 / 32 / 128 substeps; at θ₀ = π/2 the energy loss of
//! 3.64 / 1.13 / 0.29 J shrinks the swing, and the period error is 4.1 % / 1.3 % / 0.30 %.
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: the closed-form period K(sin(θ₀/2)) and the starting angle in plain f64, compared with tolerances, never pinned to bits"
)]
use alice_physics_core::joint::{BallJoint, Joint};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

const I_CM: f64 = 0.4;
const I_P: f64 = 1.4;

fn elliptic_k(k: f64) -> f64 {
    let (mut a, mut b) = (1.0f64, (1.0 - k * k).sqrt());
    for _ in 0..40 {
        let (an, bn) = ((a + b) / 2.0, (a * b).sqrt());
        a = an;
        b = bn;
    }
    std::f64::consts::PI / (2.0 * a)
}

/// period of the swing (mean spacing of the downward crossings of x = 0, interpolated
/// linearly between frames), the largest `E − E₀` seen, and `E_end − E₀`
fn swing(substeps: usize, theta0: f64, frames: usize) -> (f64, usize, f64, f64) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        substeps,
        ..PhysicsConfig::default()
    });
    let g = -w.config.gravity.y.to_f64();
    assert_eq!(g, 10.0);
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let (s, c) = theta0.sin_cos();
    let pos = Vec3Fix::new(Fix128::from_f64(s), Fix128::from_f64(-c), Fix128::ZERO);
    let b = w.add_body(RigidBody::new_dynamic(pos, Fix128::ONE));
    // the pivot is the anchor's centre; on the body it is at −pos (rotation the identity)
    w.add_joint(Joint::Ball(BallJoint::new(anchor, b, Vec3Fix::ZERO, -pos)));
    let energy = |w: &PhysicsWorld| {
        let body = &w.bodies[b];
        0.5 * body.velocity.length_squared().to_f64()
            + 0.5 * I_CM * body.angular_velocity.length_squared().to_f64()
            + g * body.position.y.to_f64()
    };
    let e0 = energy(&w);
    let mut rise = f64::MIN;
    let mut crossings = Vec::new();
    let mut px = w.bodies[b].position.x.to_f64();
    let dt = 1.0 / 60.0;
    for f in 0..frames {
        w.step(Fix128::from_ratio(1, 60));
        let x = w.bodies[b].position.x.to_f64();
        if px > 0.0 && x <= 0.0 {
            crossings.push((f as f64 + px / (px - x)) * dt);
        }
        px = x;
        rise = rise.max(energy(&w) - e0);
    }
    let n = crossings.len();
    assert!(n >= 3, "only {n} swings measured");
    let period = (crossings[n - 1] - crossings[0]) / (n - 1) as f64;
    (period, n, rise, energy(&w) - e0)
}

fn closed_form_period(theta0: f64) -> f64 {
    4.0 * (I_P / 10.0).sqrt() * elliptic_k((theta0 / 2.0).sin())
}

#[test]
fn the_elliptic_k_reference_is_right() {
    // K(0) = π/2, K(1/√2) = Γ(1/4)² / (4·√π) = 1.854074677301372
    assert!((elliptic_k(0.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    assert!((elliptic_k(std::f64::consts::FRAC_1_SQRT_2) - 1.854_074_677_301_372).abs() < 1e-14);
}

#[test]
fn a_small_swing_has_the_closed_form_period_and_converges_with_the_substep() {
    let theta0 = 0.1;
    let t = closed_form_period(theta0);
    let mut errors = Vec::new();
    for substeps in [8usize, 32, 128] {
        let (period, n, rise, _) = swing(substeps, theta0, 1200);
        let err = ((period - t) / t).abs();
        assert!(rise <= 0.0, "substeps {substeps}: energy rose by {rise:e}");
        assert!(n >= 8);
        errors.push(err);
    }
    // the default 8 substeps is within 1e-4 of the closed form, and the error falls with
    // the substep (measured 5.4e-5, 2.0e-5, 4.1e-6)
    assert!(
        errors[0] < 1e-4,
        "period error {:e} at 8 substeps",
        errors[0]
    );
    assert!(errors[1] < errors[0] && errors[2] < errors[1], "{errors:?}");
}

#[test]
fn a_wide_swing_loses_energy_at_first_order_and_its_period_converges() {
    let theta0 = std::f64::consts::FRAC_PI_2;
    let t = closed_form_period(theta0);
    let mut errors = Vec::new();
    let mut losses = Vec::new();
    for substeps in [8usize, 32, 128] {
        let (period, _, rise, drift) = swing(substeps, theta0, 1200);
        assert!(rise <= 0.0, "substeps {substeps}: energy rose by {rise:e}");
        assert!(drift < 0.0, "substeps {substeps}: no loss measured");
        errors.push(((period - t) / t).abs());
        losses.push(-drift);
    }
    // first order: 4× the substeps takes the loss down by about 4 (measured 3.23, 3.82)
    for k in 0..2 {
        let ratio = losses[k] / losses[k + 1];
        assert!(
            (2.5..=5.0).contains(&ratio),
            "loss ratio {ratio} ({losses:?})"
        );
    }
    assert!(errors[1] < errors[0] && errors[2] < errors[1], "{errors:?}");
    assert!(
        errors[2] < 0.005,
        "period error {:e} at 128 substeps",
        errors[2]
    );
}
