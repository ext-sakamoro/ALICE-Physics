//! Oracles for an SDF collider attached to a body, stepped through
//! `PhysicsWorld::step` (and `step_parallel`, and the TGS backend).
//!
//! # What is measured
//!
//! A collider made by `SdfCollider::new_dynamic(field, k)` puts the field's
//! local origin at body `k`'s position and its local axes along body `k`'s
//! axes, so every contact is a contact with the field in the body's frame.
//! The scenes use no gravity and one substep:
//!
//! - **translation** (impact closed form): a kinematic body driven to its
//!   target over the step `dt` moves at `V`. The field reaches a free sphere
//!   at `t_c` (the gap over `V`) and the sphere leaves at `(1 + e) V`, ending
//!   at `x_c + (1 + e) V (dt − t_c)` (`e` the pair's restitution, here the
//!   default 0.3). The step resolves the impact to the substep it falls in
//!   (the gap is closed there, the bounce acts on the velocity after it), so
//!   the speed is exact and the position trails the closed form by at most
//!   `e V h + α d / h²` (`h` the substep; the second term is the compliant
//!   contact's give, `α` the pair's compliance, `d ≤ V h` the overlap the
//!   position-based solve removes, `tests/analytic_compliant_rest.rs`): the
//!   position-based step bounces after its
//!   substep's positions, and the temporal Gauss–Seidel step does too for a
//!   gap that closes within the substep (for a field touching from the start
//!   it bounces from the start, and is exact). Scenes: the local half-space
//!   `y < 0` lifted from `y = 0` to `y = 3/4` under a sphere of radius `1/2`
//!   resting on it (`t_c = 0`), and a unit ball moved from the origin to
//!   `x = 2` towards a sphere at `x = 3` (`t_c = 3/4 dt`);
//! - **rotation**: the local half-space `y < 0` on a body turned by +90°
//!   about `z` has world normal `R ŷ = −x̂`: the solid is `x > P_x`. A sphere
//!   of radius `r` at `x = P_x + e` (`e > −r`) is pushed along `−x̂` to
//!   `x = P_x − r`, its `y` and `z` unchanged. The position-based step detects
//!   after the turn and pushes it out in the step. The temporal Gauss–Seidel
//!   step is not run here: it detects at the start of the tick and treats a
//!   contact to first order in the turn of the tick, which a quarter turn is
//!   far outside (the field's turn is not seen within the tick, and the
//!   overlap found at the next tick is corrected over the ticks that follow);
//! - **reaction**: a field attached to a dynamic body pushes a free body as
//!   the body it is attached to, so the push reacts on its carrier: the pair's
//!   linear momentum, and its angular momentum about any point, are kept (the
//!   impulses are equal and opposite at one point; the overlap the sphere
//!   starts in is pushed out without velocity). A body is never pushed out of
//!   its own field (it would gain momentum the pair does not have);
//! - after a step every dynamic collider holds its body's pose (what
//!   `sdf_contacts` and `sdf_ccd_hits` read between steps); static colliders
//!   keep the pose they were created with, bit for bit.
//!
//! The kinematic bodies reach their target in the one substep, so their
//! final pose is the target exactly.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::material::DEFAULT_CONTACT_COMPLIANCE;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider, SdfField};
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend};

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

/// The impact closed form on axis `axis` (0 = x, 1 = y): the speed is exact,
/// the position trails `want` by `0 ..= lag` (see the module doc).
fn impact(
    w: &PhysicsWorld,
    s: usize,
    axis: usize,
    want: f64,
    speed: f64,
    lag: f64,
    runner: Runner,
) {
    let b = &w.bodies[s];
    let (x, v) = if axis == 0 {
        (b.position.x.to_f64(), b.velocity.x.to_f64())
    } else {
        (b.position.y.to_f64(), b.velocity.y.to_f64())
    };
    assert!((v - speed).abs() < 1e-6, "{runner:?}: speed {v} vs {speed}");
    // the compliant give: α = 2 · DEFAULT_CONTACT_COMPLIANCE (two materials in
    // series), d ≤ V h = speed / (1 + e) · h, h = dt (one substep)
    let h = 1.0 / 60.0;
    let alpha = 2.0 * DEFAULT_CONTACT_COMPLIANCE.to_f64();
    let lag = lag + alpha * speed * h / (h * h);
    let behind = want - x;
    assert!(
        (-1e-9..=lag + 1e-9).contains(&behind),
        "{runner:?}: at {x}, closed form {want}, behind by {behind} (at most {lag})"
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
    // V = 45, t_c = 0: leaves at 1.3 V = 58.5, ends at 1/2 + 58.5 dt = 1.475.
    let (e, big_v, dt) = (0.3, 45.0, 1.0 / 60.0);
    for runner in runners() {
        let mut w = world(runner);
        let mut platform = RigidBody::new_kinematic(Vec3Fix::ZERO);
        platform.set_kinematic_target(v3(0.0, 0.75, 0.0), QuatFix::IDENTITY);
        let p = w.add_body(platform);
        let s = w.add_body(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE));
        let k = w.add_sdf_collider(SdfCollider::new_dynamic(half_space(0.0), p));

        step(&mut w, runner);

        assert_eq!(w.bodies[p].position, v3(0.0, 0.75, 0.0), "{runner:?}");
        impact(
            &w,
            s,
            1,
            0.5 + (1.0 + e) * big_v * dt,
            (1.0 + e) * big_v,
            e * big_v * dt,
            runner,
        );
        near(w.bodies[s].position.x, 0.0, 1e-9, &format!("{runner:?} x"));
        assert_collider_on_body(&w, k, p, runner);
    }
}

#[test]
fn a_translated_ball_pushes_a_sphere_ahead_of_it() {
    // V = 120, gap 3 − 1 − 1/2 = 3/2, t_c = 3/4 dt: leaves at 156, ends at
    // 3 + 156 · dt / 4 = 3.65.
    let (e, big_v, dt) = (0.3, 120.0, 1.0 / 60.0);
    for runner in runners() {
        let mut w = world(runner);
        let mut carrier = RigidBody::new_kinematic(Vec3Fix::ZERO);
        carrier.set_kinematic_target(v3(2.0, 0.0, 0.0), QuatFix::IDENTITY);
        let c = w.add_body(carrier);
        let s = w.add_body(RigidBody::new_dynamic(v3(3.0, 0.0, 0.0), Fix128::ONE));
        w.add_sdf_collider(SdfCollider::new_dynamic(unit_ball(), c));

        step(&mut w, runner);

        impact(
            &w,
            s,
            0,
            3.0 + (1.0 + e) * big_v * dt / 4.0,
            (1.0 + e) * big_v,
            e * big_v * dt,
            runner,
        );
        near(w.bodies[s].position.y, 0.0, 1e-9, &format!("{runner:?} y"));
    }
}

#[test]
fn a_rotated_half_space_pushes_along_its_rotated_normal() {
    // Not the temporal Gauss–Seidel runner (see the module doc).
    for runner in runners().into_iter().filter(|r| !matches!(r, Runner::Tgs)) {
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

/// The push of a field on a free body reacts on the dynamic body carrying it:
/// momentum is kept (see the module doc), and the carrier is not pushed by its
/// own field.
#[test]
fn the_carrier_is_not_pushed_by_its_own_field() {
    let momentum = |b: &RigidBody| -> (f64, f64) {
        let m = 1.0 / b.inv_mass.to_f64();
        let iz = 1.0 / b.inv_inertia.z.to_f64();
        let (x, y) = (b.position.x.to_f64(), b.position.y.to_f64());
        let (vx, vy) = (b.velocity.x.to_f64(), b.velocity.y.to_f64());
        (
            m * vy,
            iz * b.angular_velocity.z.to_f64() + m * (x * vy - y * vx),
        )
    };
    for runner in runners() {
        let mut w = world(runner);
        let mut carrier = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
        carrier.velocity = v3(0.0, 6.0, 0.0);
        let c = w.add_body(carrier);
        // 1/4 into the field y < −1/4 of the carrier, 3 to its side.
        let s = w.add_body(RigidBody::new_dynamic(v3(3.0, 0.0, 0.0), Fix128::ONE));
        w.add_sdf_collider(SdfCollider::new_dynamic(half_space(-0.25), c));
        step(&mut w, runner);

        let ((pc, lc), (ps, ls)) = (momentum(&w.bodies[c]), momentum(&w.bodies[s]));
        assert!(
            (pc + ps - 6.0).abs() < 1e-12,
            "{runner:?}: p_y {} vs 6",
            pc + ps
        );
        assert!(ps > 0.0, "{runner:?}: the sphere is pushed up ({ps})");
        // Initially L_z = 0 about the origin; the bound is rounding over the
        // components (each about 1).
        assert!((lc + ls).abs() < 1e-8, "{runner:?}: L_z {}", lc + ls);
        assert_collider_on_body(&w, 0, c, runner);

        // Between steps the world's queries read the synced pose: a sphere
        // centred on the plane of the carrier's field (local y = −1/4, turned
        // with the carrier, which the push has set spinning) is r deep.
        let carrier = w.bodies[c];
        w.bodies[s].position = carrier.position
            + carrier
                .rotation
                .rotate_vec(Vec3Fix::new(Fix128::ZERO, -fx(0.25), Fix128::ZERO));
        w.bodies[s].velocity = Vec3Fix::ZERO;
        let contacts = w.sdf_contacts();
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
        w.set_sleep_config(alice_physics_core::SleepConfig {
            time_to_sleep: Fix128::from_ratio(2, 60),
            ..alice_physics_core::SleepConfig::default()
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
