//! A solid cylinder rolling down an incline with its axis across the slope
//! (Erleben et al., Physics-Based Animation ch. 6; the benchmark of
//! COV-RIGID-113): `I = m r² / 2` about the axis, so without slip
//! `a = g sin θ / (1 + 1/2) = (2/3) g sin θ`, and the friction it needs,
//! `(1/3) m g sin θ`, stays inside the cone while `tan θ ≤ 3 μ`; above it the
//! centre slides at `g (sin θ − μ cos θ)` and the spin rate is `α r = 2 μ g cos θ`.
//!
//! Known defect (the tests are ignored until the fix lands): the contact of a
//! shaped body is one EPA support vertex, so a cylinder lying on a plane has
//! a point contact instead of its line of contact, and the friction torque
//! about that point does not give the rolling law. Fix: a contact manifold
//! for cylinder / capsule line contact (Backlog `cylinder line contact`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 incline closed forms"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::PhysicsMaterial;

const DT: f64 = 1.0 / 60.0;
const R: f64 = 0.5;
const HALF_LEN: f64 = 0.5;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn g() -> f64 {
    -PhysicsConfig::default().gravity.y.to_f64()
}

fn substep() -> f64 {
    DT / PhysicsConfig::default().substeps as f64
}

/// Static box turned by `θ` about `z` and a solid cylinder (`m = 1`, `r = 1/2`,
/// length 1) lying on it with its axis along `z` (the shape's `Y` axis turned
/// onto `z`), released from rest.
fn incline(mu: f64, theta: f64) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        damping_rate: Fix128::ZERO,
        ..PhysicsConfig::default()
    });
    let slab = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.bodies[slab].rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, fx(theta)).normalize();
    assert!(w.set_body_shape(
        slab,
        &Shape::Box {
            half_extents: v3(20.0, 0.5, 2.0),
        }
    ));
    let normal = [-theta.sin(), theta.cos(), 0.0];
    let up = 0.5 + R;
    let cyl = w.add_body(RigidBody::new_dynamic(
        v3(normal[0] * up, normal[1] * up, 0.0),
        Fix128::ONE,
    ));
    let shape = Shape::Cylinder {
        radius: fx(R),
        half_height: fx(HALF_LEN),
    };
    assert!(w.set_body_shape(cyl, &shape));
    // axis Y -> z: a quarter turn about x
    w.bodies[cyl].rotation =
        QuatFix::from_axis_angle(Vec3Fix::UNIT_X, fx(core::f64::consts::FRAC_PI_2)).normalize();
    let inertia = shape.inertia_diagonal(Fix128::ONE);
    w.bodies[cyl].inv_inertia = Vec3Fix::new(
        Fix128::ONE / inertia.x,
        Fix128::ONE / inertia.y,
        Fix128::ONE / inertia.z,
    );
    // the axial moment is m r² / 2
    assert!(
        (inertia.y.to_f64() - 0.5 * R * R).abs() < 1e-9,
        "axial moment {}",
        inertia.y.to_f64()
    );
    let id = w
        .material_table
        .register(PhysicsMaterial::new(0, fx(mu), Fix128::ZERO));
    w.set_body_material(slab, id);
    w.set_body_material(cyl, id);
    (w, cyl)
}

/// `(downhill displacement, downhill speed, ω_z r)` after `frames`
fn run(mu: f64, theta: f64, frames: usize) -> (f64, f64, f64) {
    let (mut w, cyl) = incline(mu, theta);
    let start = w.bodies[cyl].position;
    for _ in 0..frames {
        w.step(fx(DT));
    }
    let b = &w.bodies[cyl];
    let d = b.position - start;
    let down = -(d.x.to_f64() * theta.cos() + d.y.to_f64() * theta.sin());
    let speed = -(b.velocity.x.to_f64() * theta.cos() + b.velocity.y.to_f64() * theta.sin());
    (down, speed, b.angular_velocity.z.to_f64() * R)
}

/// oracle: `θ = 0.3`, `μ = 0.5` (`tan θ = 0.309 ≤ 3 μ = 1.5`): rolls without
/// slip, `d = (2/3) g sin θ / 2` after 1 s within one substep's `a h t`, and
/// `v = ω r` within the same budget.
#[test]
#[ignore = "known defect: cylinder line contact"]
fn a_cylinder_below_the_slip_limit_rolls_at_two_thirds_of_g_sin_theta() {
    let (mu, theta) = (0.5, 0.3_f64);
    let a = 2.0 / 3.0 * g() * theta.sin();
    let (down, speed, wr) = run(mu, theta, 60);
    let want = a * 0.5;
    println!("cylinder θ=0.3 μ=0.5 t=1: d {down:.6} (want {want:.6}) v {speed:.6} ω_z r {wr:.6}");
    assert!(
        (down - want).abs() <= a * substep() && (speed - wr).abs() <= a * substep(),
        "rolled {down:.6} (want {want:.6}), v {speed:.6}, ω r {wr:.6}"
    );
}

/// oracle: `θ = 0.3`, `μ = 0.05` (`tan θ = 0.309 > 3 μ = 0.15`): slides,
/// `d = g (sin θ − μ cos θ) / 2` and `ω_z r = 2 μ g cos θ · t` after 1 s within
/// one substep's `a h t`, and `v − ω r > 0`.
#[test]
#[ignore = "known defect: cylinder line contact"]
fn a_cylinder_above_the_slip_limit_slides_and_spins_at_the_kinetic_rates() {
    let (mu, theta) = (0.05, 0.3_f64);
    let a = g() * (theta.sin() - mu * theta.cos());
    let alpha_r = 2.0 * mu * g() * theta.cos();
    let (down, speed, wr) = run(mu, theta, 60);
    let want = a * 0.5;
    println!("cylinder θ=0.3 μ=0.05 t=1: d {down:.6} (want {want:.6}) v {speed:.6} ω_z r {wr:.6} (want {alpha_r:.6})");
    assert!(
        (down - want).abs() <= a * substep()
            && (wr - alpha_r).abs() <= alpha_r * substep()
            && speed - wr > 0.0,
        "slid {down:.6} (want {want:.6}), ω r {wr:.6} (want {alpha_r:.6}), v {speed:.6}"
    );
}
