//! Oracles for the parts of a `PhysicsWorld` the TGS backend has to honour
//! besides contacts and distance constraints: the world's joints
//! ([`Joint`]), its static colliders (plane / height field / triangle mesh)
//! and its contact filters (pre-solve hooks and contact modifiers).
//!
//! Every scene runs through the production entry point, `PhysicsWorld::step`
//! with `solver_backend: SolverBackend::Tgs`; the rest of the configuration is
//! `SolverConfig::default()` (gravity, damping, substeps and iterations) except
//! where a scene states otherwise. Expected values are closed forms written
//! from the scene's geometry and the default frame damping, never obtained by
//! running the solver:
//!
//! * frame damping: `step` multiplies every dynamic body's linear and angular
//!   velocity by `d = 0.99` once per frame of `dt`, i.e. a decay rate
//!   `β = −ln(d) / dt` per second in the continuous limit;
//! * a body pendulum whose bob has the default inertia `I = 2/5 · m` about its
//!   centre (`RigidBody::new_dynamic`) and hangs at arm length `L` from a fixed
//!   pivot obeys `θ'' + β θ' + ω₀² θ = 0` for small `θ`, with
//!   `ω₀² = m g L / (I + m L²)`, so its period is `T = 2π / √(ω₀² − β²/4)`.
//!   A solver that moved the bob as a point mass (ignoring its rotation) would
//!   give `ω₀² = g / L`, a period about 18 % shorter at `L = 1`;
//! * a body on a frictionless axis with acceleration `a` along it:
//!   `x(t) = (a/β) t − (a/β²)(1 − e^{−βt})`;
//! * a free fall: the same with `a = −g`;
//! * a sliding body that cannot rotate, under kinetic friction `μ`:
//!   `v(t) = (v₀ + μg/β) e^{−βt} − μg/β`.
//!
//! The tolerances state their own budget next to each assert.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::{TriMesh, Triangle};
use alice_physics::{ContactModifier, SolverBackend};

const DT: f64 = 1.0 / 60.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn tgs(substeps: usize) -> SolverConfig {
    SolverConfig {
        substeps,
        solver_backend: SolverBackend::Tgs,
        ..SolverConfig::default()
    }
}

/// Gravity magnitude and frame-damping rate of `SolverConfig::default()`.
fn g_and_beta() -> (f64, f64) {
    let d = SolverConfig::default();
    (-d.gravity.y.to_f64(), -(d.damping.to_f64()).ln() / DT)
}

/// Displacement along an axis with constant acceleration `a` from rest under
/// the frame damping `beta`, after `t` seconds.
fn damped_displacement(a: f64, beta: f64, t: f64) -> f64 {
    (a / beta) * t - (a / (beta * beta)) * (1.0 - (-beta * t).exp())
}

// ---------------------------------------------------------------------------
// Static colliders
// ---------------------------------------------------------------------------

fn quad_floor() -> StaticCollider {
    let a = v3(-5.0, 0.0, -5.0);
    let b = v3(-5.0, 0.0, 5.0);
    let c = v3(5.0, 0.0, 5.0);
    let d = v3(5.0, 0.0, -5.0);
    StaticCollider::TriMesh(TriMesh::from_triangles(vec![
        Triangle::new(a, b, c),
        Triangle::new(a, c, d),
    ]))
}

fn floors() -> [(&'static str, StaticCollider); 3] {
    [
        (
            "plane",
            StaticCollider::Plane(PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO)),
        ),
        (
            "height field",
            StaticCollider::HeightField(HeightField::flat(
                11,
                11,
                Fix128::ONE,
                v3(-5.0, 0.0, -5.0),
                Fix128::ZERO,
            )),
        ),
        ("triangle mesh", quad_floor()),
    ]
}

/// A sphere of radius `r = 0.5` dropped from `0.5 m` above a floor comes to
/// rest on it: the static equilibrium of a body on a surface is a height of
/// `r` and a contact that supplies `m g dt` of upward impulse per frame, so
/// the vertical velocity stops changing.
///
/// Tolerances: the body may sink below `r` by at most `2e-2` (the TGS contact
/// keeps a slop of `5e-3` and corrects the rest over a few frames) and float
/// above it by at most `5e-3`; over the last 30 frames the vertical velocity
/// stays within `5 %` of `g·dt` (a contact missing for one frame would leave
/// `g·dt` there, a body falling through would reach several `g·dt`).
#[test]
fn a_body_rests_on_a_plane_a_height_field_and_a_triangle_mesh() {
    let (g, _) = g_and_beta();
    let r = 0.5;
    for k in 0..3 {
        for substeps in [4, 8] {
            let (name, floor) = floors().into_iter().nth(k).expect("floor");
            let mut w = PhysicsWorld::new(tgs(substeps));
            w.add_static_collider(floor);
            let b = w.add_body_with_radius(
                RigidBody::new_dynamic(v3(0.3, 1.0, -0.2), Fix128::ONE),
                fx(r),
            );
            let mut max_vy_late: f64 = 0.0;
            let mut lowest: f64 = f64::MAX;
            for n in 0..180 {
                w.step(fx(DT));
                let body = w.get_body(b).expect("body");
                if n >= 60 {
                    lowest = lowest.min(body.position.y.to_f64());
                }
                if n >= 150 {
                    max_vy_late = max_vy_late.max(body.velocity.y.to_f64().abs());
                }
            }
            let y = w.get_body(b).expect("body").position.y.to_f64();
            assert!(
                lowest > r - 2e-2 && y < r + 5e-3,
                "{name}, substeps {substeps}: rest height {y:.5} (lowest {lowest:.5}), \
                 closed form {r}"
            );
            assert!(
                max_vy_late < 5e-2 * g * DT,
                "{name}, substeps {substeps}: vertical velocity {max_vy_late:.3e} at rest \
                 (g·dt = {:.3e})",
                g * DT
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Contact filters
// ---------------------------------------------------------------------------

/// A large static sphere whose top is the plane `y = 0` (curvature radius
/// `R = 1e4`: over the metres these scenes cover the surface tilts by under
/// `3e-4 rad`), and a dynamic ball of radius `0.5` resting on it.
fn ball_on_ground(substeps: usize, ball_y: f64) -> (PhysicsWorld, usize, usize) {
    let mut w = PhysicsWorld::new(tgs(substeps));
    let big = 1.0e4;
    let ground = w.add_body_with_radius(RigidBody::new_static(v3(0.0, -big, 0.0)), fx(big));
    let ball = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, ball_y, 0.0), Fix128::ONE),
        fx(0.5),
    );
    (w, ground, ball)
}

/// A pre-solve hook that rejects the ball-ground pair leaves no impulse on it:
/// the ball falls through the ground as a free body, `y(t) = y₀ +
/// x_damped(−g, t)`. Without the hook the same ball stays on the ground (the
/// control, so the scene does have the contact the hook removes).
///
/// Budget for the `2e-2` tolerance: the sub-step integration offset `g·t·h/2
/// ≤ 1.6e-2` at `substeps = 4` (`h = dt/4`, `t = 0.5 s`) and per-frame damping
/// `O(β·dt)` of the `≈ 0.12 m` damping term.
#[test]
fn a_rejecting_hook_lets_the_body_fall_through() {
    let (g, beta) = g_and_beta();
    let y0 = 0.45;
    for substeps in [4, 8] {
        let (mut w, ground, ball) = ball_on_ground(substeps, y0);
        w.add_pre_solve_hook(Box::new(move |a, b, _c: &Contact| {
            !((a == ground && b == ball) || (a == ball && b == ground))
        }));
        for _ in 0..30 {
            w.step(fx(DT));
        }
        let y = w.get_body(ball).expect("ball").position.y.to_f64();
        let want = y0 + damped_displacement(-g, beta, 0.5);
        assert!(
            (y - want).abs() < 2e-2,
            "substeps {substeps}: y = {y:.5}, free fall {want:.5}"
        );

        let (mut control, _, ball) = ball_on_ground(substeps, y0);
        for _ in 0..30 {
            control.step(fx(DT));
        }
        let y = control.get_body(ball).expect("ball").position.y.to_f64();
        assert!(
            y > 0.4,
            "substeps {substeps}: without the hook the ball should rest on the ground, y = {y:.5}"
        );
    }
}

/// Sets every contact's friction to an absolute value.
struct AbsoluteFriction(Fix128);

impl ContactModifier for AbsoluteFriction {
    fn modify_contact(
        &self,
        _a: usize,
        _b: usize,
        _contact: &mut Contact,
        friction: &mut Fix128,
        _restitution: &mut Fix128,
    ) -> bool {
        *friction = self.0;
        true
    }
}

/// A contact modifier that sets the friction to `μ = 0.1` makes a sliding
/// ball (rotation locked, so it slides and does not roll) decelerate at `μ g`
/// plus the frame damping: `v(t) = (v₀ + μg/β) e^{−βt} − μg/β`.
///
/// Budget for the 3 % tolerance on `v(1 s) ≈ 0.89`: the friction impulse per
/// sub-step is capped by `μ` times the normal impulse, which equals `m g h`
/// up to the surface tilt (`< 3e-4`); per-frame rather than continuous
/// damping changes `v` by `O(β·dt)` of the damping term (`≈ 2e-2` absolute).
/// Without the modifier the material friction (`0.5` by default) stops the
/// ball well before 1 s, so the scene separates the two.
#[test]
fn an_absolute_friction_modifier_sets_the_sliding_deceleration() {
    let (g, beta) = g_and_beta();
    let mu = 0.1;
    let v0 = 3.0;
    let want = (v0 + mu * g / beta) * (-beta).exp() - mu * g / beta;
    for substeps in [4, 8] {
        let run = |with_modifier: bool| {
            let (mut w, _, ball) = ball_on_ground(substeps, 0.49);
            w.bodies[ball].inv_inertia = Vec3Fix::ZERO;
            w.bodies[ball].velocity = v3(v0, 0.0, 0.0);
            if with_modifier {
                w.add_contact_modifier(Box::new(AbsoluteFriction(fx(mu))));
            }
            for _ in 0..60 {
                w.step(fx(DT));
            }
            w.get_body(ball).expect("ball").velocity.x.to_f64()
        };
        let v = run(true);
        assert!(
            (v - want).abs() < 3e-2 * want,
            "substeps {substeps}: v(1 s) = {v:.5}, closed form {want:.5}"
        );
        let v_material = run(false);
        assert!(
            v_material < 0.5 * want,
            "substeps {substeps}: without the modifier v(1 s) = {v_material:.5}"
        );
    }
}

// ---------------------------------------------------------------------------
// World integration: removing a body mid-simulation
// ---------------------------------------------------------------------------

/// Removing a body that rests on a plane moves the last body, which also
/// rests on the plane but is three times heavier, into its index. The moved
/// body must stay at rest: the contact state the TGS solve carries from frame
/// to frame must follow the bodies, not their indices.
#[test]
fn removing_a_resting_body_does_not_disturb_the_body_moved_into_its_index() {
    let (g, _) = g_and_beta();
    let mut w = PhysicsWorld::new(tgs(8));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let light = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(-2.0, 0.5, 0.0), Fix128::ONE),
        fx(0.5),
    );
    let heavy = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(2.0, 0.5, 0.0), Fix128::from_int(3)),
        fx(0.5),
    );
    for _ in 0..120 {
        w.step(fx(DT));
    }
    assert_eq!(heavy, 1);
    assert!(w.remove_body(light).is_some());
    let mut max_vy: f64 = 0.0;
    let mut lowest: f64 = f64::MAX;
    for _ in 0..30 {
        w.step(fx(DT));
        let b = w.get_body(light).expect("moved body");
        max_vy = max_vy.max(b.velocity.y.to_f64().abs());
        lowest = lowest.min(b.position.y.to_f64());
    }
    assert!(
        max_vy < 5e-2 * g * DT,
        "the moved body's vertical velocity reached {max_vy:.3e} (g·dt = {:.3e})",
        g * DT
    );
    assert!(lowest > 0.5 - 2e-2, "the moved body sank to {lowest:.5}");
}
