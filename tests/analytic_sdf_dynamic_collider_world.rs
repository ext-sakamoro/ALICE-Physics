//! Oracles for an SDF collider attached to a body, stepped through
//! `PhysicsWorld::step` (and `step_parallel`, and the TGS backend).
//!
//! # What is measured
//!
//! A collider made by `SdfCollider::new_dynamic(field, k)` puts the field's
//! local origin at body `k`'s position and its local axes along body `k`'s
//! axes. The world copies that pose into the collider before it resolves SDF
//! overlap, so every push-out is the push-out of the field in the body's
//! current frame. The scenes use no gravity and one substep, so each push has
//! a closed form:
//!
//! - **translation**: the local half-space `y < 0` on a kinematic body moved
//!   from `y = 0` to `y = h` is the world half-space `y < h`; a sphere of
//!   radius `r` resting on top at `y = r` is pushed to `y = h + r`. A unit
//!   ball on a body moved from the origin to `(a, 0, 0)` pushes a sphere of
//!   radius `r` centred at `(c, 0, 0)` (`c − a < 1 + r`) to `x = a + 1 + r`;
//! - **rotation**: the local half-space `y < 0` on a body turned by +90°
//!   about `z` has world normal `R ŷ = −x̂`: the solid is `x > P_x`. A sphere
//!   of radius `r` at `x = P_x + e` (`e > −r`) is pushed along `−x̂` to
//!   `x = P_x − r`, its `y` and `z` unchanged;
//! - **self**: a body is never pushed out of its own field. A dynamic body
//!   carrying the local half-space `y < −1/4` with velocity `(0, 6, 0)` moves
//!   by `v·dt` exactly, as it would with no collider; a free sphere of radius
//!   `1/2` at `y = 0` next to it is pushed out of the field placed at the
//!   carrier's new position: to `y = v·dt − 1/4 + 1/2`;
//! - after a step every dynamic collider holds its body's pose (what
//!   `sdf_contacts` and `sdf_ccd_hits` read between steps); static colliders
//!   keep the pose they were created with, bit for bit.
//!
//! The kinematic bodies reach their target in the one substep, so their
//! final pose is the target exactly.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// The local half-space `y < offset` (solid below).
fn half_space(offset: f32) -> Box<dyn SdfField> {
    Box::new(ClosureSdf::new(
        move |_, y, _| y - offset,
        |_, _, _| (0.0, 1.0, 0.0),
    ))
}

fn unit_ball() -> Box<dyn SdfField> {
    Box::new(ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-6);
            (x / l, y / l, z / l)
        },
    ))
}

#[derive(Clone, Copy, Debug)]
enum Runner {
    Step,
    Tgs,
    #[cfg(feature = "parallel")]
    Parallel,
}

fn runners() -> Vec<Runner> {
    vec![
        Runner::Step,
        Runner::Tgs,
        #[cfg(feature = "parallel")]
        Runner::Parallel,
    ]
}

fn world(runner: Runner) -> PhysicsWorld {
    let mut config = PhysicsConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    if matches!(runner, Runner::Tgs) {
        config.solver_backend = SolverBackend::Tgs;
    }
    let mut w = PhysicsWorld::new(config);
    w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
    w
}

fn step(w: &mut PhysicsWorld, runner: Runner) {
    match runner {
        Runner::Step | Runner::Tgs => w.step(dt()),
        #[cfg(feature = "parallel")]
        Runner::Parallel => w.step_parallel(dt()),
    }
}

fn near(a: Fix128, b: f64, tol: f64, what: &str) {
    assert!(
        (a.to_f64() - b).abs() < tol,
        "{what}: {} vs closed form {b}",
        a.to_f64()
    );
}

fn assert_collider_on_body(w: &PhysicsWorld, collider: usize, body: usize, runner: Runner) {
    let c = &w.sdf_colliders[collider];
    let b = &w.bodies[body];
    assert_eq!(c.position, b.position, "{runner:?}: collider position");
    assert_eq!(c.rotation, b.rotation, "{runner:?}: collider rotation");
}

#[test]
fn a_lifted_platform_pushes_the_resting_sphere_up_by_the_lift() {
    for runner in runners() {
        let mut w = world(runner);
        let mut platform = RigidBody::new_kinematic(Vec3Fix::ZERO);
        platform.set_kinematic_target(v3(0.0, 0.75, 0.0), QuatFix::IDENTITY);
        let p = w.add_body(platform);
        let s = w.add_body(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE));
        let k = w.add_sdf_collider(SdfCollider::new_dynamic(half_space(0.0), p));

        step(&mut w, runner);

        assert_eq!(w.bodies[p].position, v3(0.0, 0.75, 0.0), "{runner:?}");
        // Plane at y = 0.75, radius 0.5: centre at 1.25.
        near(w.bodies[s].position.y, 1.25, 1e-5, &format!("{runner:?} y"));
        near(w.bodies[s].position.x, 0.0, 1e-9, &format!("{runner:?} x"));
        assert_collider_on_body(&w, k, p, runner);
    }
}

#[test]
fn a_translated_ball_pushes_a_sphere_ahead_of_it() {
    for runner in runners() {
        let mut w = world(runner);
        let mut carrier = RigidBody::new_kinematic(Vec3Fix::ZERO);
        carrier.set_kinematic_target(v3(2.0, 0.0, 0.0), QuatFix::IDENTITY);
        let c = w.add_body(carrier);
        let s = w.add_body(RigidBody::new_dynamic(v3(3.0, 0.0, 0.0), Fix128::ONE));
        w.add_sdf_collider(SdfCollider::new_dynamic(unit_ball(), c));

        step(&mut w, runner);

        // Ball of radius 1 at x = 2, sphere radius 0.5: x = 3.5.
        near(w.bodies[s].position.x, 3.5, 1e-5, &format!("{runner:?} x"));
        near(w.bodies[s].position.y, 0.0, 1e-9, &format!("{runner:?} y"));
    }
}

#[test]
fn a_rotated_half_space_pushes_along_its_rotated_normal() {
    for runner in runners() {
        let mut w = world(runner);
        let pivot = [1.0, 0.0, 0.0];
        let mut carrier = RigidBody::new_kinematic(v3(pivot[0], pivot[1], pivot[2]));
        let turn = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI);
        carrier.set_kinematic_target(v3(pivot[0], pivot[1], pivot[2]), turn);
        let c = w.add_body(carrier);
        // 0.25 inside the turned solid x > 1, high above the unturned one.
        let s = w.add_body(RigidBody::new_dynamic(v3(1.25, 5.0, -2.0), Fix128::ONE));
        let k = w.add_sdf_collider(SdfCollider::new_dynamic(half_space(0.0), c));

        step(&mut w, runner);

        // Pushed along −x̂ to x = P_x − r = 0.5; y and z unchanged.
        near(w.bodies[s].position.x, 0.5, 1e-4, &format!("{runner:?} x"));
        near(w.bodies[s].position.y, 5.0, 1e-4, &format!("{runner:?} y"));
        near(w.bodies[s].position.z, -2.0, 1e-9, &format!("{runner:?} z"));
        assert_collider_on_body(&w, k, c, runner);
    }
}

/// The carrier moves as it would with no collider at all (bit for bit), and
/// a free sphere beside it is pushed out of the field at the carrier's pose.
#[test]
fn the_carrier_is_not_pushed_by_its_own_field() {
    for runner in runners() {
        let build = |with_field: bool| {
            let mut w = world(runner);
            let mut carrier = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
            carrier.velocity = v3(0.0, 6.0, 0.0);
            let c = w.add_body(carrier);
            let s = w.add_body(RigidBody::new_dynamic(v3(3.0, 0.0, 0.0), Fix128::ONE));
            if with_field {
                w.add_sdf_collider(SdfCollider::new_dynamic(half_space(-0.25), c));
            }
            (w, c, s)
        };
        let (mut a, c, s) = build(true);
        let (mut b, _, _) = build(false);
        step(&mut a, runner);
        step(&mut b, runner);

        assert_eq!(a.bodies[c].position, b.bodies[c].position, "{runner:?}");
        assert_eq!(a.bodies[c].rotation, b.bodies[c].rotation, "{runner:?}");
        assert_collider_on_body(&a, 0, c, runner);

        if matches!(runner, Runner::Tgs) {
            // TGS resolves SDF overlap once, before it integrates: the field
            // is still where the carrier started (plane y = −1/4).
            near(a.bodies[s].position.y, 0.25, 1e-5, &format!("{runner:?} y"));
        } else {
            // XPBD resolves after the carrier moved by v·dt = 0.1: plane at
            // y = −0.15, sphere centre at 0.35.
            near(
                a.bodies[c].position.y,
                0.1,
                1e-12,
                &format!("{runner:?} carrier"),
            );
            near(a.bodies[s].position.y, 0.35, 1e-5, &format!("{runner:?} y"));
        }

        // Between steps the world's queries read the synced pose: a sphere
        // centred on the plane y = P_y − 1/4 of the carrier's field is r deep.
        a.bodies[s].position = Vec3Fix::new(
            a.bodies[s].position.x,
            a.bodies[c].position.y - fx(0.25),
            Fix128::ZERO,
        );
        a.bodies[s].velocity = Vec3Fix::ZERO;
        let contacts = a.sdf_contacts();
        assert_eq!(contacts.len(), 1, "{runner:?}: {contacts:?}");
        assert_eq!(contacts[0].0, s, "{runner:?}");
        near(contacts[0].1.depth, 0.5, 1e-5, &format!("{runner:?} depth"));
    }
}

#[test]
fn static_colliders_keep_their_pose_and_resolve_as_before() {
    for runner in runners() {
        let mut w = world(runner);
        let at = v3(0.0, -1.0, 0.0);
        let k = w.add_sdf_collider(SdfCollider::new_static(
            half_space(0.0),
            at,
            QuatFix::IDENTITY,
        ));
        // A dynamic collider whose body index names no body stays put too.
        let orphan = w.add_sdf_collider(SdfCollider::new_dynamic(unit_ball(), 99));
        let s = w.add_body(RigidBody::new_dynamic(v3(10.0, -0.75, 0.0), Fix128::ONE));

        step(&mut w, runner);

        assert_eq!(w.sdf_colliders[k].position, at, "{runner:?}");
        assert_eq!(w.sdf_colliders[k].rotation, QuatFix::IDENTITY, "{runner:?}");
        assert_eq!(
            w.sdf_colliders[orphan].position,
            Vec3Fix::ZERO,
            "{runner:?}"
        );
        // Plane y = −1, radius 0.5: centre at −0.5 (the orphan unit ball at
        // the origin is 10 away).
        near(w.bodies[s].position.y, -0.5, 1e-5, &format!("{runner:?} y"));
    }
}

#[test]
fn a_collider_added_for_an_existing_body_starts_at_its_pose() {
    let mut w = world(Runner::Step);
    let mut body = RigidBody::new_kinematic(v3(4.0, -2.0, 1.0));
    body.rotation = QuatFix::from_axis_angle(v3(0.0, 1.0, 0.0), Fix128::HALF_PI);
    let b = w.add_body(body);
    let k = w.add_sdf_collider(SdfCollider::new_dynamic(unit_ball(), b));
    assert_collider_on_body(&w, k, b, Runner::Step);
    // The ball at (4, −2, 1) reaches a sphere 1.25 above its centre.
    let s = w.add_body(RigidBody::new_dynamic(v3(4.0, -0.75, 1.0), Fix128::ONE));
    let contacts = w.sdf_contacts();
    assert_eq!(contacts.len(), 1, "{contacts:?}");
    assert_eq!(contacts[0].0, s);
    near(contacts[0].1.depth, 0.25, 1e-5, "depth");
}

/// A sphere asleep on a platform is lifted with it in the step the platform
/// starts to rise (the sleep skip must not leave it behind), bit for bit as
/// with the sleep skip off.
#[test]
fn a_sleeping_sphere_is_lifted_in_the_first_step_of_the_lift() {
    let run = |sleep_skip: bool| {
        let mut w = world(Runner::Step);
        w.set_sleep_skip(sleep_skip);
        w.set_sleep_config(alice_physics::SleepConfig {
            frames_to_sleep: 2,
            ..alice_physics::SleepConfig::default()
        });
        let p = w.add_body(RigidBody::new_kinematic(Vec3Fix::ZERO));
        let s = w.add_body(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE));
        w.add_sdf_collider(SdfCollider::new_dynamic(half_space(0.0), p));
        for _ in 0..4 {
            w.step(dt());
        }
        assert!(
            w.is_sleeping(s),
            "sleep_skip {sleep_skip}: the sphere sleeps"
        );
        w.bodies[p].set_kinematic_target(v3(0.0, 0.75, 0.0), QuatFix::IDENTITY);
        w.step(dt());
        near(
            w.bodies[s].position.y,
            1.25,
            1e-5,
            &format!("sleep_skip {sleep_skip} y"),
        );
        w.bodies[s].position
    };
    assert_eq!(run(true), run(false));
}
