//! The TGS backend detects contacts once per tick; inside the tick each
//! position iteration measures the penetration at the bodies' present poses and
//! corrects it through the full effective mass, moving and turning the bodies.
//!
//! oracle (closed form, no gravity, no velocity, so the velocity iterations do
//! nothing and only the position iterations act): a position iteration removes
//! the fraction `β` of the excess `d − slop` at the contact point (the positional
//! impulse `λ = β (d − slop) m_eff` moves the point by `λ / m_eff`), so after one
//! tick of `N = substeps × position iterations` iterations
//!
//! `d_N = slop + (d₀ − slop) (1 − β)^N`
//!
//! (β = 0.2, slop = 0.005 m, two position iterations per substep: the TGS
//! defaults). For a contact whose arm is not along the normal (a tilted box on
//! its corner) the point moves by the turn as well as the shift; the decay holds
//! to first order in the turn `θ`, whose second-order part `r θ²` is the
//! tolerance.
//!
//! An overlap the dynamics did not make never reaches these iterations: a
//! body moved into the plane from outside the step (a pose write between
//! ticks) has its points counted as new, and the overlap is pushed out by
//! position, without velocity, before the tick solves (the start-of-substep
//! push). The two tests here pin that for the temporal Gauss–Seidel backend on
//! a contact that persists (a first tick finds it): the ball and the tilted
//! plank come out of the plane in one tick at rest. The decay law above is
//! pinned on the position correction itself
//! (`solver_tgs_hooks_6dof_oriented_scoped`,
//! `position_correction_moves_by_baumgarte_times_excess_per_pass`).

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 sin / cos of the box tilt"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::SolverBackend;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn world(substeps: usize) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        damping_rate: Fix128::ZERO,
        solver_backend: SolverBackend::Tgs,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        Fix128::ZERO,
    )));
    w
}

/// Put body `b` at rest at `position`, `rotation` (the pose of the next tick's
/// start), keeping the contacts the last tick found.
fn sink_to(w: &mut PhysicsWorld, b: usize, position: Vec3Fix, rotation: QuatFix) {
    let body = w.get_body_mut(b).expect("body");
    body.position = position;
    body.prev_position = position;
    body.rotation = rotation;
    body.prev_rotation = rotation;
    body.velocity = Vec3Fix::ZERO;
    body.angular_velocity = Vec3Fix::ZERO;
}

/// A ball whose resting contact persists is written `d₀ = 0.1` into the plane
/// between ticks: one tick puts it back on the surface, exactly (a single point
/// along the normal through the centre: the least-constraint push moves the
/// ball by `d₀`), at rest (no gravity here, and the push adds no velocity), at
/// substeps 1, 4 and 8.
#[test]
fn a_ball_written_into_the_plane_comes_out_at_rest_in_one_tick() {
    let (r, d0) = (0.5, 0.1);
    for substeps in [1, 4, 8] {
        let mut w = world(substeps);
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, r - 1e-4, 0.0), Fix128::ONE),
            fx(r),
        );
        w.step(fx(1.0 / 60.0));
        sink_to(&mut w, b, v3(0.0, r - d0, 0.0), QuatFix::IDENTITY);
        w.step(fx(1.0 / 60.0));
        let body = w.get_body(b).expect("ball");
        assert_eq!(body.position.y, fx(r), "s {substeps}: on the surface");
        assert_eq!(body.velocity, Vec3Fix::ZERO, "s {substeps}: at rest");
        assert_eq!(body.angular_velocity, Vec3Fix::ZERO, "s {substeps}");
    }
}

/// A plank tilted by `θ₀ = 0.05` rad about `z` (and `ψ = 0.002` rad about `x`,
/// so one corner is the lowest alone), resting on its low edge, is written so
/// that corner is `d₀ = 0.01` into the plane: one tick pushes it out, turning,
/// until no corner is deeper than the push's stopping tolerance `2⁻⁵⁰` m, and
/// leaves it at rest (no velocity, no spin), at substeps 4 and 8.
#[test]
fn a_tilted_plank_written_into_the_plane_comes_out_at_rest_in_one_tick() {
    let (hx, hy, hz) = (0.45, 0.06, 0.01);
    let (theta, psi, d0) = (0.05f64, 0.002f64, 0.01);
    let shape = Shape::Box {
        half_extents: v3(hx, hy, hz),
    };
    let about_z = QuatFix::new(
        fx(0.0),
        fx(0.0),
        fx(-(theta / 2.0).sin()),
        fx((theta / 2.0).cos()),
    );
    let about_x = QuatFix::new(
        fx((psi / 2.0).sin()),
        fx(0.0),
        fx(0.0),
        fx((psi / 2.0).cos()),
    );
    let q0 = about_z.mul(about_x);
    let corners: Vec<Vec3Fix> = [-1.0, 1.0]
        .iter()
        .flat_map(|&sx| [-1.0, 1.0].map(move |sz| v3(sx * hx, -hy, sz * hz)))
        .collect();
    let height = |q: QuatFix, c: Vec3Fix| q.rotate_vec(c).y.to_f64();
    let low = corners
        .iter()
        .copied()
        .min_by(|a, b| height(q0, *a).total_cmp(&height(q0, *b)))
        .expect("corner");
    let tolerance = 1.0 / 1_125_899_906_842_624.0; // 2^-50
    for substeps in [4, 8] {
        let mut w = world(substeps);
        let y0 = -height(q0, low) - d0;
        let b = w
            .add_shaped_body(&shape, fx(500.0), v3(0.0, y0 + d0 - 1e-4, 0.0))
            .expect("box");
        w.get_body_mut(b).expect("box").rotation = q0;
        w.get_body_mut(b).expect("box").prev_rotation = q0;
        w.step(fx(1.0 / 60.0));
        sink_to(&mut w, b, v3(0.0, y0, 0.0), q0);
        w.step(fx(1.0 / 60.0));
        let body = w.get_body(b).expect("box");
        let deepest = corners
            .iter()
            .map(|c| -(body.position.y.to_f64() + height(body.rotation, *c)))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            deepest <= tolerance,
            "s {substeps}: a corner {deepest} m deep"
        );
        assert_eq!(body.velocity, Vec3Fix::ZERO, "s {substeps}: at rest");
        assert_eq!(
            body.angular_velocity,
            Vec3Fix::ZERO,
            "s {substeps}: no spin"
        );
    }
}
