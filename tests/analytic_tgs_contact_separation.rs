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
//! What it catches: reading the detection-time depth in every iteration pushes
//! `N β (d₀ − slop)` in total (the ball pops far above the plane); moving the
//! centre of mass only (no turn) moves a corner by `λ / m < β (d − slop)`, so it
//! decays slower than `(1 − β)^N`.

#![allow(
    clippy::disallowed_methods,
    reason = "analytic reference: f64 sin / cos of the box tilt and the powi of the geometric decay"
)]

use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics_core::static_collider::StaticCollider;
use alice_physics_core::SolverBackend;

const BETA: f64 = 0.2;
const SLOP: f64 = 0.005;
const POSITION_ITERS: usize = 2;

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

fn decayed(d0: f64, n: usize) -> f64 {
    SLOP + (d0 - SLOP) * (1.0 - BETA).powi(n as i32)
}

/// A ball overlapping the plane by `d₀ = 0.1` comes out to
/// `slop + (d₀ − slop)(1 − β)^N` in one tick, at substeps 1, 4 and 8 (within
/// `1e-6` m: the contact arm is along the normal, the decay is exact up to
/// rounding).
#[test]
fn an_overlapping_ball_comes_out_by_the_geometric_decay_in_one_tick() {
    let (r, d0) = (0.5, 0.1);
    let mut wrong = Vec::new();
    for substeps in [1, 4, 8] {
        let mut w = world(substeps);
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, r - d0, 0.0), Fix128::ONE),
            fx(r),
        );
        w.step(fx(1.0 / 60.0));
        let depth = r - w.get_body(b).expect("ball").position.y.to_f64();
        let want = decayed(d0, substeps * POSITION_ITERS);
        if (depth - want).abs() > 1e-6 {
            wrong.push(format!("s {substeps}: depth {depth:.6}, want {want:.6}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A plank tilted by `θ₀ = 0.05` rad about `z` (and `ψ = 0.002` rad about `x`,
/// so one corner is the lowest alone), that corner overlapping the plane by
/// `d₀ = 0.01`, brings it to `slop + (d₀ − slop)(1 − β)^N` in one tick
/// (substeps 4 and 8), turning as it rises. The plank is thin along `z` (half
/// depth 0.01 m), so the turn about `x` that the corner's arm along `z` gives is
/// small. Tolerance: the second-order part of the turn, `r θ²` with `r` the
/// corner's arm (0.46 m) and `θ` the total turn (below `d₀ / r`), `2e-4` m.
#[test]
fn a_tilted_box_brings_its_corner_out_by_the_geometric_decay_turning() {
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
    let second = corners
        .iter()
        .map(|c| height(q0, *c))
        .filter(|h| *h > height(q0, low))
        .fold(f64::INFINITY, f64::min);
    assert!(
        second - height(q0, low) > 1e-5,
        "one corner is the lowest alone"
    );
    let mut wrong = Vec::new();
    for substeps in [4, 8] {
        let mut w = world(substeps);
        let y0 = -height(q0, low) - d0;
        let b = w
            .add_shaped_body(&shape, fx(500.0), v3(0.0, y0, 0.0))
            .expect("box");
        w.get_body_mut(b).expect("box").rotation = q0;
        w.get_body_mut(b).expect("box").prev_rotation = q0;
        w.step(fx(1.0 / 60.0));
        let body = w.get_body(b).expect("box");
        let depth = -(body.position.y.to_f64() + height(body.rotation, low));
        let want = decayed(d0, substeps * POSITION_ITERS);
        if (depth - want).abs() > 2e-4 {
            wrong.push(format!(
                "s {substeps}: corner depth {depth:.6}, want {want:.6}"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
