//! A kinematic body driven to a target hits a free body during the step, on
//! both backends.
//!
//! oracle (impact closed form, no gravity): a kinematic ball of radius `R`
//! driven from `x = 0` to `x = D` over one step `dt` moves at `V = D / dt`. A
//! free sphere of radius `r` at rest at `x = c`, with a gap `g = c − R − r`
//! ahead of the ball, is reached at `t_c = g / V`; with restitution 0 it then
//! moves with the ball, so at the end of the step it touches the ball at its
//! target: `x = D + R + r`, velocity `V`. With restitution `e` it leaves at
//! `(1 + e) V` and ends at `x = c + (1 + e) V (dt − t_c)`.
//!
//! The step resolves the impact to the substep it falls in: in that substep
//! the gap is closed exactly (the free body ends it touching), and the bounce
//! is applied to the velocity after the substep's positions advance. So with
//! restitution 0 the end state is exact whatever `t_c` is (the sphere touches
//! the ball at its target, at speed `V`), and with restitution `e` the speed
//! `(1 + e) V` is exact. The position depends on where the bounce is placed in
//! the impact substep: the temporal Gauss–Seidel step applies it after the
//! substep, so the position trails the closed form by at most `e V h` (`h` the
//! substep); the position-based step takes it from the approach of a
//! speculative point whose gap closes within the substep (Box2D v3's rule),
//! so it bounces from at most that gap short of the surface, up to a substep
//! early: it leads by at most `(1 + e) V h`. Scenes: `R = 1`, `r = 1/2`, `D = 2`, `dt = 1/60` (`V = 120`), 4
//! substeps; `c = 3` (`t_c = 3/4 dt`) and `c = 11/4` (`t_c = 5/8 dt`, the
//! middle of the third substep).
//!
//! What it catches: the kinematic body snapped to its target before contacts
//! are detected, so that the contact already carries the step's motion, and
//! the free body's solved velocity then integrated over the whole step again
//! (the sphere ended at 5.478 instead of 3.5 under the temporal Gauss–Seidel
//! backend).

use alice_physics_core::material::PhysicsMaterial;
use alice_physics_core::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics_core::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody, SolverBackend};

fn scene(backend: SolverBackend, e: Fix128, c: Fix128) -> (PhysicsWorld, usize, usize) {
    let mut config = PhysicsConfig {
        substeps: 4,
        iterations: 4,
        gravity: Vec3Fix::ZERO,
        ..PhysicsConfig::default()
    };
    config.solver_backend = backend;
    let mut w = PhysicsWorld::new(config);
    // Rigid contacts (compliance 0): the closed form is the rigid impact. The
    // default compliance leaves `α F` of each overlap, which is the compliant
    // contact's own law (tests/analytic_compliant_rest.rs).
    let rigid = w.material_table.register(
        PhysicsMaterial::new(0, Fix128::from_ratio(1, 2), e).with_contact_compliance(Fix128::ZERO),
    );
    let mut ball = RigidBody::new_kinematic(Vec3Fix::ZERO);
    ball.restitution = e;
    ball.set_kinematic_target(
        Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
        QuatFix::IDENTITY,
    );
    let k = w.add_body_with_radius(ball, Fix128::ONE);
    w.set_body_material(k, rigid);
    let mut s = RigidBody::new_dynamic(Vec3Fix::new(c, Fix128::ZERO, Fix128::ZERO), Fix128::ONE);
    s.restitution = e;
    let s = w.add_body_with_radius(s, Fix128::from_ratio(1, 2));
    w.set_body_material(s, rigid);
    (w, k, s)
}

const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];

#[test]
fn a_kinematic_ball_carries_the_sphere_it_reaches_during_the_step() {
    let mut compared = 0;
    for (backend, c) in [
        (SolverBackend::Xpbd, Fix128::from_int(3)),
        (SolverBackend::Tgs, Fix128::from_int(3)),
        (SolverBackend::Xpbd, Fix128::from_ratio(11, 4)),
        (SolverBackend::Tgs, Fix128::from_ratio(11, 4)),
    ] {
        let (mut w, k, s) = scene(backend, Fix128::ZERO, c);
        w.step(Fix128::from_ratio(1, 60));
        let (kb, sb) = (&w.bodies[k], &w.bodies[s]);
        assert_eq!(kb.position.x, Fix128::from_int(2), "{backend:?} target");
        let x = sb.position.x.to_f64();
        let v = sb.velocity.x.to_f64();
        assert!((x - 3.5).abs() < 1e-9, "{backend:?}: x {x} vs 3.5");
        assert!((v - 120.0).abs() < 1e-6, "{backend:?}: v {v} vs 120");
        assert!(
            sb.position.y.is_zero() && sb.position.z.is_zero(),
            "{backend:?}"
        );
        compared += 1;
    }
    assert_eq!(compared, 4);
}

#[test]
fn with_restitution_it_leaves_at_one_plus_e_times_the_speed() {
    // e = 1/2 on both bodies (the pair's restitution is then 1/2 whatever the
    // combine rule): leaves at 180. c = 11/4: t_c = 5/8 dt, the closed form
    // ends at 11/4 + 180 (dt − t_c) = 3.875; the bound is e V h = 1/4.
    let (e, v_big, h) = (0.5, 120.0, 1.0 / 240.0);
    let x_closed = 2.75 + (1.0 + e) * v_big * (1.0 / 60.0) * (3.0 / 8.0);
    let mut compared = 0;
    for backend in BACKENDS {
        let (mut w, _, s) = scene(backend, Fix128::from_ratio(1, 2), Fix128::from_ratio(11, 4));
        w.step(Fix128::from_ratio(1, 60));
        let sb = &w.bodies[s];
        let x = sb.position.x.to_f64();
        let v = sb.velocity.x.to_f64();
        assert!((v - 180.0).abs() < 1e-6, "{backend:?}: v {v} vs 180");
        let lag = x_closed - x;
        let (behind, ahead) = match backend {
            SolverBackend::Tgs => (e * v_big * h, 0.0),
            _ => (0.0, (1.0 + e) * v_big * h),
        };
        assert!(
            lag >= -ahead - 1e-9 && lag <= behind + 1e-9,
            "{backend:?}: x {x}, closed form {x_closed}, lag {lag} (behind ≤ {behind}, ahead ≤ {ahead})"
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// oracle (first order in the turn): the local half-space `y < 0` carried by
/// a kinematic body at the origin, turned by a small `θ` about `z` over the
/// step, under a frictionless sphere of radius `r` resting on it at `x = L`
/// (restitution 0). The surface under the sphere rises at `ω L` (`ω = θ / dt`)
/// along the normal, so the sphere leaves at `ω L` and ends `L θ` higher; the
/// exact rigid answer differs by terms of order `θ²` (the turned normal tilts
/// by `θ`): `|y − (r + L θ)| ≤ 2 L θ²` and `|v − ω L| ≤ 2 L θ² / dt`. Run on
/// both backends at 1 and 4 substeps (the temporal Gauss–Seidel step moves the
/// kinematic body by its angular velocity within the substeps).
#[test]
fn a_turning_kinematic_field_lifts_the_sphere_by_the_lever_times_the_turn() {
    let (l, theta, dt) = (2.0f64, 1.0e-3f64, 1.0 / 60.0);
    let mut compared = 0;
    for backend in BACKENDS {
        for substeps in [1usize, 4] {
            let mut config = PhysicsConfig {
                substeps,
                iterations: 4,
                gravity: Vec3Fix::ZERO,
                ..PhysicsConfig::default()
            };
            config.solver_backend = backend;
            let mut w = PhysicsWorld::new(config);
            w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
            let slick = w.material_table.register(
                PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ZERO)
                    .with_contact_compliance(Fix128::ZERO),
            );
            let mut carrier = RigidBody::new_kinematic(Vec3Fix::ZERO);
            carrier.set_kinematic_target(
                Vec3Fix::ZERO,
                QuatFix::from_axis_angle(
                    Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
                    Fix128::from_f64(theta),
                ),
            );
            let c = w.add_body(carrier);
            w.set_body_material(c, slick);
            let mut sphere = RigidBody::new_dynamic(
                Vec3Fix::new(Fix128::from_f64(l), Fix128::from_ratio(1, 2), Fix128::ZERO),
                Fix128::ONE,
            );
            sphere.restitution = Fix128::ZERO;
            sphere.friction = Fix128::ZERO;
            let s = w.add_body(sphere);
            w.set_body_material(s, slick);
            w.add_sdf_collider(SdfCollider::new_dynamic(
                Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
                c,
            ));
            w.step(Fix128::from_ratio(1, 60));
            let sb = &w.bodies[s];
            let (y, vy) = (sb.position.y.to_f64(), sb.velocity.y.to_f64());
            let bound = 2.0 * l * theta * theta;
            assert!(
                (y - (0.5 + l * theta)).abs() <= bound,
                "{backend:?} substeps {substeps}: y {y} vs {} (bound {bound})",
                0.5 + l * theta
            );
            assert!(
                (vy - l * theta / dt).abs() <= bound / dt,
                "{backend:?} substeps {substeps}: v {vy} vs {} (bound {})",
                l * theta / dt,
                bound / dt
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 4);
}

/// oracle (first order in the turn): as above, but the sphere starts a gap
/// `g = L θ / 2` above the surface, which the turning field closes half way
/// through the step. With restitution 0 the sphere ends touching the turned
/// plane: its distance to the plane through the pivot with normal
/// `(−sin θ, cos θ, 0)` is `r`, to within `2 L θ²`. The contact is speculative
/// for the first half of the step; its gap is measured at the carrier's turned
/// pose in each substep (the temporal Gauss–Seidel step turns the carrier's
/// contact arm with it), 4 substeps.
#[test]
fn a_turning_kinematic_field_meets_a_sphere_above_it_half_way_and_carries_it() {
    let (l, theta) = (2.0f64, 1.0e-2f64);
    let mut compared = 0;
    for backend in BACKENDS {
        let mut config = PhysicsConfig {
            substeps: 4,
            iterations: 4,
            gravity: Vec3Fix::ZERO,
            ..PhysicsConfig::default()
        };
        config.solver_backend = backend;
        let mut w = PhysicsWorld::new(config);
        w.set_sdf_collision_radius(Fix128::from_ratio(1, 2));
        let slick = w.material_table.register(
            PhysicsMaterial::new(0, Fix128::ZERO, Fix128::ZERO)
                .with_contact_compliance(Fix128::ZERO),
        );
        let mut carrier = RigidBody::new_kinematic(Vec3Fix::ZERO);
        carrier.set_kinematic_target(
            Vec3Fix::ZERO,
            QuatFix::from_axis_angle(
                Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
                Fix128::from_f64(theta),
            ),
        );
        let c = w.add_body(carrier);
        w.set_body_material(c, slick);
        let mut sphere = RigidBody::new_dynamic(
            Vec3Fix::new(
                Fix128::from_f64(l),
                Fix128::from_f64(0.5 + l * theta / 2.0),
                Fix128::ZERO,
            ),
            Fix128::ONE,
        );
        sphere.restitution = Fix128::ZERO;
        sphere.friction = Fix128::ZERO;
        let s = w.add_body(sphere);
        w.set_body_material(s, slick);
        w.add_sdf_collider(SdfCollider::new_dynamic(
            Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
            c,
        ));
        w.step(Fix128::from_ratio(1, 60));
        let sb = &w.bodies[s];
        let (x, y) = (sb.position.x.to_f64(), sb.position.y.to_f64());
        // sin and cos by their series (the platform libm is not allowed); the
        // first omitted terms, θ⁵/120 and θ⁴/24, are far below the bound
        let (sn, cs) = (
            theta - theta * theta * theta / 6.0,
            1.0 - theta * theta / 2.0,
        );
        let distance = -x * sn + y * cs;
        let bound = 2.0 * l * theta * theta;
        assert!(
            (distance - 0.5).abs() <= bound,
            "{backend:?}: distance to the turned plane {distance} vs 0.5 (bound {bound})"
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}
