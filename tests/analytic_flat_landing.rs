//! A box falling flat onto a floor lands flat: its face hits with all four corners
//! at once, so the contact pushes it straight up and does not turn it.
//!
//! oracle (symmetry): the four corner contacts are mirror images about the box's
//! two vertical symmetry planes; the reaction is then a pure force through the
//! centre of mass, so the box keeps `ω = 0` and its face level. The tolerance is
//! the rounding of the four arms (`1e-9` rad of tilt, `1e-9` rad/s).
//!
//! What it catches: correcting the four corners one after another in a single
//! iteration. The first corner's correction `λ₁ = d / w₁` (`d ≈ v h` at impact)
//! also turns the box by `I⁻¹ (r × n) λ₁`, which the later corners only partly
//! undo, so the box is left turning, by up to `d / r` (here about `1e-2` rad).

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 acos of the tilt of the box's up axis"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// The largest tilt (rad) and the largest `|ω|` of the box over 0.5 s after it is
/// dropped flat from `drop` above the floor, at `hz` with one iteration.
fn landing(hz: f64, drop: f64) -> (f64, f64) {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-10.0), Fix128::ZERO),
        iterations: 1,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    )));
    let b = w
        .add_shaped_body(
            &Shape::Box {
                half_extents: Vec3Fix::new(fx(0.45), fx(0.06), fx(0.3)),
            },
            fx(500.0),
            Vec3Fix::new(Fix128::ZERO, fx(0.06 + drop), Fix128::ZERO),
        )
        .expect("box");
    let (mut tilt, mut spin) = (0.0f64, 0.0f64);
    for _ in 0..(0.5 * hz) as usize {
        w.step(fx(1.0 / hz));
        let body = w.get_body(b).expect("box");
        let up = body
            .rotation
            .rotate_vec(Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO));
        tilt = tilt.max(up.y.to_f64().clamp(-1.0, 1.0).acos());
        spin = spin.max(body.angular_velocity.length().to_f64());
    }
    (tilt, spin)
}

/// Dropped from 0.45 m (3 m/s at impact) at 240 and 480 Hz.
#[test]
fn a_box_falling_flat_lands_without_turning() {
    let mut wrong = Vec::new();
    for hz in [240.0, 480.0] {
        let (tilt, spin) = landing(hz, 0.45);
        eprintln!("{hz} Hz: tilt {tilt:.3e} rad, |ω| {spin:.3e} rad/s");
        if tilt > 1e-9 || spin > 1e-9 {
            wrong.push(format!(
                "{hz} Hz: tilt {tilt:.3e} rad, |ω| {spin:.3e} rad/s"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
