//! Oracles for the parts of a `PhysicsWorld` the TGS backend has to honour
//! besides contacts and distance constraints: the world's joints ([`Joint`],
//! solved inside the substep loop) and its static colliders (plane / height
//! field / triangle mesh).
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
//! * a damped harmonic oscillator: period `2π / √(k/m − β²/4)`.
//!
//! The tolerances state their own budget next to each assert.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::heightfield::HeightField;
use alice_physics::joint::{
    BallJoint, ConeTwistJoint, D6Joint, D6Motion, FixedJoint, HingeJoint, Joint, SliderJoint,
    SpringJoint,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::{TriMesh, Triangle};
use alice_physics::SolverBackend;

const DT: f64 = 1.0 / 60.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
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
// Joints
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Kind {
    Ball,
    Hinge,
    /// All three linear axes locked, all angular axes free: a ball joint.
    D6,
    /// The cone (π/2) and twist (π) limits are never reached by a `0.1 rad`
    /// swing, so it moves as a ball joint.
    ConeTwist,
}

/// A static pivot at the origin and a unit-mass bob at arm length `L`,
/// released at `θ₀` from the vertical, joined by `kind` with the pivot point
/// as the anchor on both bodies. Returns the world, the bob index and the
/// bob-local anchor.
fn pendulum(kind: Kind, substeps: usize, l: f64, theta0: f64) -> (PhysicsWorld, usize, Vec3Fix) {
    let mut w = PhysicsWorld::new(tgs(substeps));
    let pivot = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let bob = w.add_body(RigidBody::new_dynamic(
        v3(l * theta0.sin(), -l * theta0.cos(), 0.0),
        Fix128::ONE,
    ));
    let anchor_b = v3(-l * theta0.sin(), l * theta0.cos(), 0.0);
    let joint = match kind {
        Kind::Ball => Joint::Ball(BallJoint::new(pivot, bob, Vec3Fix::ZERO, anchor_b)),
        Kind::Hinge => Joint::Hinge(HingeJoint::new(
            pivot,
            bob,
            Vec3Fix::ZERO,
            anchor_b,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
        Kind::D6 => Joint::D6(
            D6Joint::new(pivot, bob, Vec3Fix::ZERO, anchor_b).with_linear_motion(
                D6Motion::Locked,
                D6Motion::Locked,
                D6Motion::Locked,
            ),
        ),
        Kind::ConeTwist => Joint::ConeTwist(ConeTwistJoint::new(
            pivot,
            bob,
            Vec3Fix::ZERO,
            anchor_b,
            v3(0.0, -1.0, 0.0),
            v3(theta0.sin(), -theta0.cos(), 0.0),
        )),
    };
    w.add_joint(joint);
    (w, bob, anchor_b)
}

/// World position of the bob's joint anchor.
fn anchor_world(w: &PhysicsWorld, bob: usize, local: Vec3Fix) -> [f64; 3] {
    let b = w.get_body(bob).expect("bob");
    arr(b.position + b.rotation.rotate_vec(local))
}

/// Steps the pendulum for `seconds` and returns (the mean interval between
/// successive left-to-right crossings of the vertical, the largest distance
/// between the bob's anchor and the pivot).
fn swing(w: &mut PhysicsWorld, bob: usize, local: Vec3Fix, seconds: f64) -> (f64, f64) {
    let steps = (seconds / DT).round() as usize;
    let mut crossings = Vec::new();
    let mut prev_x = w.get_body(bob).expect("bob").position.x.to_f64();
    let mut max_sep: f64 = 0.0;
    for n in 1..=steps {
        w.step(fx(DT));
        let x = w.get_body(bob).expect("bob").position.x.to_f64();
        if prev_x < 0.0 && x >= 0.0 {
            let frac = -prev_x / (x - prev_x);
            crossings.push((n as f64 - 1.0 + frac) * DT);
        }
        prev_x = x;
        let a = anchor_world(w, bob, local);
        max_sep = max_sep.max((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt());
    }
    assert!(
        crossings.len() >= 3,
        "the bob crossed the vertical {} times in {seconds} s: it is not swinging",
        crossings.len()
    );
    let mean = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    (mean, max_sep)
}

/// Period of the damped body pendulum (module doc), unit mass.
fn pendulum_period(l: f64) -> f64 {
    let (g, beta) = g_and_beta();
    let inertia = 0.4;
    let w0_sq = g * l / (inertia + l * l);
    2.0 * std::f64::consts::PI / (w0_sq - beta * beta / 4.0).sqrt()
}

/// Ball, hinge, D6 (linear axes locked) and cone-twist pendulums keep the
/// closed-form period and stay attached.
///
/// Budget for the 1 % period tolerance: the finite amplitude adds `θ₀²/16 =
/// 6e-4` of the period; the crossing times are interpolated linearly inside a
/// frame (error well under `dt²` per crossing); the per-frame (not
/// continuous) damping shifts `β` by `O(β·dt) = 1 %` of its own, a `1e-4`
/// effect on the period. The point-mass period differs by 18 %, so a solver
/// that drops the bob's rotation fails by an order of magnitude more than
/// the tolerance. The anchor may drift from the pivot by at most `1e-2 L`.
#[test]
fn pendulums_of_every_pivoting_joint_keep_the_closed_form_period_and_stay_attached() {
    let l = 1.0;
    let expected = pendulum_period(l);
    for kind in [Kind::Ball, Kind::Hinge, Kind::D6, Kind::ConeTwist] {
        for substeps in [4, 8] {
            let (mut w, bob, local) = pendulum(kind, substeps, l, 0.1);
            let (period, max_sep) = swing(&mut w, bob, local, 3.0 * expected);
            let rel = (period - expected).abs() / expected;
            assert!(
                rel < 1e-2,
                "{kind:?} substeps {substeps}: period {period:.5} s, closed form {expected:.5} s \
                 (relative error {rel:.2e})"
            );
            assert!(
                max_sep < 1e-2 * l,
                "{kind:?} substeps {substeps}: the bob's anchor drifted {max_sep:.3e} from the pivot"
            );
        }
    }
}

/// A hinge about `z` keeps the bob in the `xy` plane even when the bob is
/// pushed along `z`: the `z` coordinate stays within `1e-2` of 0 while a free
/// body would travel `v t = 1 m`.
#[test]
fn a_hinge_keeps_the_bob_in_its_plane() {
    for substeps in [4, 8] {
        let (mut w, bob, _) = pendulum(Kind::Hinge, substeps, 1.0, 0.1);
        w.bodies[bob].velocity = v3(0.0, 0.0, 1.0);
        let mut max_z: f64 = 0.0;
        for _ in 0..60 {
            w.step(fx(DT));
            max_z = max_z.max(w.get_body(bob).expect("bob").position.z.to_f64().abs());
        }
        assert!(
            max_z < 1e-2,
            "substeps {substeps}: the hinged bob left its plane by {max_z:.3e}"
        );
    }
}

/// A body welded to a static body by a fixed joint keeps its pose under
/// gravity: a cantilever that does not sag or turn. The weld point is half a
/// metre from the body's centre, so the joint has to resist a torque too. Closed form: position
/// `(1, 0, 0)` and identity rotation; tolerance `1e-2` (a free body would fall
/// `g t²/2 ≈ 20 m` in the 2 s).
#[test]
fn a_fixed_joint_holds_the_relative_pose() {
    for substeps in [4, 8] {
        let mut w = PhysicsWorld::new(tgs(substeps));
        let a = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let b = w.add_body(RigidBody::new_dynamic(v3(1.0, 0.0, 0.0), Fix128::ONE));
        w.add_joint(Joint::Fixed(FixedJoint::new(
            a,
            b,
            v3(0.5, 0.0, 0.0),
            v3(-0.5, 0.0, 0.0),
            QuatFix::IDENTITY,
        )));
        for _ in 0..120 {
            w.step(fx(DT));
        }
        let body = w.get_body(b).expect("body");
        let p = arr(body.position);
        let dp = ((p[0] - 1.0).powi(2) + p[1].powi(2) + p[2].powi(2)).sqrt();
        assert!(
            dp < 1e-2,
            "substeps {substeps}: the welded body moved {dp:.3e}"
        );
        let q = body.rotation;
        let turn =
            2.0 * (q.x.to_f64().powi(2) + q.y.to_f64().powi(2) + q.z.to_f64().powi(2)).sqrt();
        assert!(
            turn < 1e-2,
            "substeps {substeps}: the welded body turned {turn:.3e} rad"
        );
    }
}

/// A slider along `x` lets the body move only along `x`. With gravity tilted
/// to `(a, −g, 0)` the body slides along the axis by the damped closed form
/// `x(t) = (a/β) t − (a/β²)(1 − e^{−βt})` and stays on the axis.
///
/// Budget for the 2 % tolerance on `x`: the symplectic sub-step integration
/// adds `a·t·h/2` (`h = dt/substeps`, at most `1.25e-2·t` here, 1.3 % of
/// `x(1 s) ≈ 1.2`), per-frame damping `O(β·dt)` of the damping term. Off the
/// axis the body may drift `1e-2`; unconstrained it would fall `≈ 4.8 m`.
#[test]
fn a_slider_keeps_the_motion_on_its_axis() {
    let (g, beta) = g_and_beta();
    let a = 3.0;
    for substeps in [4, 8] {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: v3(a, -g, 0.0),
            ..tgs(substeps)
        });
        let base = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let body = w.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
        w.add_joint(Joint::Slider(SliderJoint::new(
            base,
            body,
            Vec3Fix::UNIT_X,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
        let mut off_axis: f64 = 0.0;
        for _ in 0..60 {
            w.step(fx(DT));
            let p = arr(w.get_body(body).expect("body").position);
            off_axis = off_axis.max((p[1] * p[1] + p[2] * p[2]).sqrt());
        }
        let x = w.get_body(body).expect("body").position.x.to_f64();
        let want = damped_displacement(a, beta, 1.0);
        assert!(
            (x - want).abs() < 2e-2 * want,
            "substeps {substeps}: x = {x:.5}, closed form {want:.5}"
        );
        assert!(
            off_axis < 1e-2,
            "substeps {substeps}: the slider body left its axis by {off_axis:.3e}"
        );
    }
}

/// A body hanging from a static body on a spring joint (stiffness `k`, unit
/// mass) oscillates about its equilibrium with the damped harmonic period
/// `T = 2π / √(k/m − β²/4)`; gravity only shifts the equilibrium.
///
/// Budget for the 2 % tolerance: the position-level spring update is first
/// order in `h²k/m` (`≤ 1.1e-3` at `substeps = 4`), per-frame damping shifts
/// `β` by `O(β·dt)`, and the crossings are interpolated within a frame
/// (`dt/T = 1.7 %` per crossing, averaged over several periods).
#[test]
fn a_spring_joint_oscillates_with_the_harmonic_period() {
    let (g, beta) = g_and_beta();
    let k = 40.0;
    let rest = 1.0;
    let expected = 2.0 * std::f64::consts::PI / (k - beta * beta / 4.0).sqrt();
    let y_eq = -(rest + g / k);
    for substeps in [4, 8] {
        let mut w = PhysicsWorld::new(tgs(substeps));
        let top = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let body = w.add_body(RigidBody::new_dynamic(
            v3(0.0, y_eq + 0.1, 0.0),
            Fix128::ONE,
        ));
        w.add_joint(Joint::Spring(SpringJoint::new(
            top,
            body,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            fx(rest),
            fx(k),
            Fix128::ZERO,
        )));
        let mut crossings = Vec::new();
        let mut prev = 0.1;
        for n in 1..=(4.0 * expected / DT) as usize {
            w.step(fx(DT));
            let d = w.get_body(body).expect("body").position.y.to_f64() - y_eq;
            if prev > 0.0 && d <= 0.0 {
                crossings.push((n as f64 - 1.0 + prev / (prev - d)) * DT);
            }
            prev = d;
        }
        assert!(
            crossings.len() >= 3,
            "substeps {substeps}: {} crossings",
            crossings.len()
        );
        let period = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
        assert!(
            (period - expected).abs() < 2e-2 * expected,
            "substeps {substeps}: period {period:.5} s, closed form {expected:.5} s"
        );
    }
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
/// Tolerances: the body may sink below `r` by at most `2e-2` (the static
/// contact pushes the sphere out along the normal once per tick, after the
/// TGS solve) and float above it by at most `5e-3`; over the last 30 frames
/// the vertical velocity stays below `5 %` of `g·dt` (a contact missing for
/// one frame would leave `g·dt` there, a body falling through would reach
/// several `g·dt`).
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
// World integration: removing a body mid-simulation
// ---------------------------------------------------------------------------

/// `remove_body` swap-removes: the last body takes the removed index and the
/// joints referring to it are remapped. A pendulum whose bob is the last body
/// keeps swinging with the closed-form period and stays attached after an
/// unrelated body is removed in the middle of the run, and a body resting on
/// a plane that is not moved by the removal stays at rest.
#[test]
fn removing_a_body_mid_run_keeps_joints_and_resting_contacts_consistent() {
    let l = 1.0;
    let expected = pendulum_period(l);
    let mut w = PhysicsWorld::new(tgs(8));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        fx(-5.0),
    )));
    let pivot = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let resting = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(3.0, -4.5, 0.0), Fix128::ONE),
        fx(0.5),
    );
    let filler = w.add_body(RigidBody::new_dynamic(v3(-3.0, 10.0, 0.0), Fix128::ONE));
    let theta0: f64 = 0.1;
    let bob = w.add_body(RigidBody::new_dynamic(
        v3(l * theta0.sin(), -l * theta0.cos(), 0.0),
        Fix128::ONE,
    ));
    let local = v3(-l * theta0.sin(), l * theta0.cos(), 0.0);
    w.add_joint(Joint::Ball(BallJoint::new(
        pivot,
        bob,
        Vec3Fix::ZERO,
        local,
    )));
    for _ in 0..30 {
        w.step(fx(DT));
    }
    assert!(w.remove_body(filler).is_some());
    let bob = filler; // swap_remove moved the last body (the bob) here
    assert_eq!(w.joint_count(), 1);
    let (period, max_sep) = swing(&mut w, bob, local, 3.0 * expected);
    let rel = (period - expected).abs() / expected;
    assert!(
        rel < 1e-2,
        "period after the removal {period:.5}, closed form {expected:.5}"
    );
    assert!(
        max_sep < 1e-2,
        "the bob's anchor drifted {max_sep:.3e} after the removal"
    );
    let rest = w.get_body(resting).expect("resting body");
    let y = rest.position.y.to_f64();
    assert!(
        y > -4.5 - 2e-2 && y < -4.5 + 5e-3,
        "the resting body is at {y:.5}, its rest height is -4.5"
    );
    assert!(rest.velocity.y.to_f64().abs() < 5e-2 * g_and_beta().0 * DT);
}

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
