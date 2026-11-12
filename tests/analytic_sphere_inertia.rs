//! A body added with a radius is a solid sphere of that radius: its inertia is
//! `I = 2/5 m r²`, so it rolls with `k = I / (m r²) = 2/5` whatever its radius.
//!
//! oracle: the solid-sphere moment of inertia `2/5 m r²` (any mechanics text), and
//! rolling without slip down a slope `θ`: `v = g sin θ t / (1 + k)`, `ω r = v`.
//! Before, every such body kept the inertia [`RigidBody::new`] gives, that of a
//! sphere of radius 1 (`k = 1.6` at `r = 0.5`), and rolled as the wrong solid.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 slope direction (sin, cos of the slope angle) for the rolling closed form"
)]

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::SolverBackend;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

/// `k = I / (m r²)` of every axis of body `b`, from its inverse inertia.
fn k_of(w: &PhysicsWorld, b: usize, r: f64) -> [f64; 3] {
    let body = w.get_body(b).expect("body");
    let m = 1.0 / body.inv_mass.to_f64();
    let i = body.inv_inertia;
    [i.x, i.y, i.z].map(|c| 1.0 / (c.to_f64() * m * r * r))
}

/// Every radius and mass gives `k = 2/5` on every axis (the Fix128 quotient is
/// within `1e-12` relative of it).
#[test]
fn a_ball_of_any_radius_and_mass_has_the_solid_sphere_inertia() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    for r in [0.05, 0.25, 0.5, 1.0, 2.0, 7.5] {
        for m in [0.2, 1.0, 3.0, 48.6] {
            let b = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), fx(m)), fx(r));
            for k in k_of(&w, b, r) {
                assert!((k - 0.4).abs() < 0.4e-12, "r {r} m {m}: k = {k}, want 2/5");
            }
        }
    }
}

/// A ball dropped onto a slope rolls at `g sin θ t / (1 + 2/5)`, on both radii
/// and at substeps 4 and 8 (TGS, the backend whose contact turns a ball).
/// Tolerance `1e-3` m/s, as the tilted-plane test (the drop keeps the angular
/// momentum about the contact point, so the ball rolls as from the start).
#[test]
fn a_ball_rolls_down_a_slope_at_five_sevenths_of_g_sin_theta_t() {
    let theta: f64 = 0.3;
    let n = [-theta.sin(), theta.cos(), 0.0];
    let mut wrong = Vec::new();
    for r in [0.25, 0.5, 1.5] {
        for substeps in [4, 8] {
            let cfg = SolverConfig {
                substeps,
                damping_rate: Fix128::ZERO,
                solver_backend: SolverBackend::Tgs,
                ..SolverConfig::default()
            };
            let g = -cfg.gravity.y.to_f64();
            let mut w = PhysicsWorld::new(cfg);
            w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
                v3(n[0], n[1], n[2]),
                Fix128::ZERO,
            )));
            let h = r + 0.01;
            let b = w.add_body_with_radius(
                RigidBody::new_dynamic(v3(n[0] * h, n[1] * h, 0.0), Fix128::ONE),
                fx(r),
            );
            for _ in 0..120 {
                w.step(fx(1.0 / 60.0));
            }
            let body = w.get_body(b).expect("ball");
            let along =
                -(body.velocity.x.to_f64() * theta.cos() + body.velocity.y.to_f64() * theta.sin());
            let spin = body.angular_velocity.z.to_f64() * r;
            let want = g * theta.sin() * 2.0 / 1.4;
            if (along - want).abs() > 1e-3 || (spin - want).abs() > 1e-3 {
                wrong.push(format!(
                    "r {r} s {substeps}: downhill {along:.5}, ω r {spin:.5}, want {want:.5}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The keeping variant leaves a preset inertia bit for bit; a body without
/// inverse mass, a radius that is not positive, and a radius so small that
/// `inv_mass · 5 / (2 r²)` reaches `2⁶¹` keep theirs on the sphere variant too
/// (`r = 8e-10` gives `3.9e18`, between `2⁶¹` and `2⁶³`).
#[test]
fn the_inertia_is_kept_where_it_is_not_a_solid_sphere_of_that_radius() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let mut boxy = RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE);
    boxy.inv_inertia = v3(0.25, 3.0, 0.5);
    let b = w.add_body_with_radius_keeping_inertia(boxy, fx(0.5));
    assert_eq!(w.get_body(b).expect("box").inv_inertia, boxy.inv_inertia);

    for body in [
        RigidBody::new_static(v3(0.0, 0.0, 0.0)),
        RigidBody::new_kinematic(v3(0.0, 0.0, 0.0)),
    ] {
        let i = w.add_body_with_radius(body, fx(0.5));
        assert_eq!(
            w.get_body(i).expect("no mass").inv_inertia,
            body.inv_inertia
        );
    }
    let unit = RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE);
    for radius in [Fix128::ZERO, fx(-0.5), fx(1e-10), fx(8e-10)] {
        let i = w.add_body_with_radius(unit, radius);
        assert_eq!(
            w.get_body(i).expect("unit").inv_inertia,
            unit.inv_inertia,
            "radius {radius:?}"
        );
    }
    // Just inside the bound: r = 1e-8 gives 2.5e16 < 2⁶¹, so it is replaced.
    let i = w.add_body_with_radius(unit, fx(1e-8));
    assert_ne!(w.get_body(i).expect("small").inv_inertia, unit.inv_inertia);
}
