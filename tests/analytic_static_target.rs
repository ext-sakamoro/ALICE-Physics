//! The static target of a contact (`STATIC_BODY`) stays internal: a contact
//! against a static collider adds no body to the world, a collider that touches
//! nothing changes no body, and a snapshot holding such contacts restores and
//! steps on bit for bit (the sentinel is written as a tag, version 8).
//!
//! oracle: the same world without the collider (bodies and their bits), and the
//! unrestored run of the same world (snapshot round trip).

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::solver::{
    PhysicsWorld, RigidBody, SolverBackend, SolverConfig, STATIC_BODY,
};
use alice_physics_core::static_collider::StaticCollider;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn world(backend: SolverBackend, floor_y: f64) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, fx(-10.0), Fix128::ZERO),
        solver_backend: backend,
        ..SolverConfig::default()
    });
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        fx(floor_y),
    )));
    w
}

fn bits(w: &PhysicsWorld) -> String {
    w.bodies
        .iter()
        .map(|b| {
            format!(
                "{:?}{:?}{:?}{:?}",
                b.position, b.velocity, b.rotation, b.angular_velocity
            )
        })
        .collect()
}

#[test]
fn a_static_contact_adds_no_body_and_names_the_static_target() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut w = world(backend, 0.0);
        w.add_body_with_radius(
            RigidBody::new_dynamic(
                Vec3Fix::new(Fix128::ZERO, fx(0.4), Fix128::ZERO),
                Fix128::ONE,
            ),
            fx(0.5),
        );
        let mut seen = false;
        for _ in 0..30 {
            w.step(fx(1.0 / 60.0));
            assert_eq!(w.bodies.len(), 1, "{backend:?}");
            seen |= w
                .contact_constraints
                .iter()
                .any(|c| c.body_b == STATIC_BODY);
        }
        assert!(
            seen,
            "{backend:?}: the ball on the plane had a contact against the static target"
        );
    }
}

#[test]
fn a_collider_that_touches_nothing_changes_no_body() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut with = world(backend, -100.0);
        let mut without = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::new(Fix128::ZERO, fx(-10.0), Fix128::ZERO),
            solver_backend: backend,
            ..SolverConfig::default()
        });
        for w in [&mut with, &mut without] {
            w.add_body_with_radius(
                RigidBody::new_dynamic(
                    Vec3Fix::new(Fix128::ZERO, fx(5.0), Fix128::ZERO),
                    Fix128::ONE,
                ),
                fx(0.5),
            );
        }
        for _ in 0..60 {
            with.step(fx(1.0 / 60.0));
            without.step(fx(1.0 / 60.0));
        }
        assert_eq!(with.bodies.len(), without.bodies.len());
        assert_eq!(bits(&with), bits(&without), "{backend:?}");
    }
}

#[test]
fn a_snapshot_with_static_contacts_restores_and_steps_on() {
    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        let mut a = world(backend, 0.0);
        a.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::new(fx(0.3), fx(0.45), Fix128::ZERO), Fix128::ONE),
            fx(0.5),
        );
        // step until the snapshot holds a contact against the static target
        let mut held = false;
        for _ in 0..60 {
            a.step(fx(1.0 / 60.0));
            if a.contact_constraints
                .iter()
                .any(|c| c.body_b == STATIC_BODY)
            {
                held = true;
                break;
            }
        }
        assert!(held, "{backend:?}");
        let snap = a.snapshot_world();
        assert_eq!(u16::from_le_bytes([snap[4], snap[5]]), 8, "{backend:?}");
        let mut b = world(backend, 0.0);
        b.add_body_with_radius(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE), fx(0.5));
        b.restore_world(&snap).expect("restore");
        assert_eq!(b.snapshot_world(), snap);
        assert!(b
            .contact_constraints
            .iter()
            .any(|c| c.body_b == STATIC_BODY));
        for _ in 0..20 {
            a.step(fx(1.0 / 60.0));
            b.step(fx(1.0 / 60.0));
        }
        assert_eq!(a.state_digest(), b.state_digest(), "{backend:?}");
    }
}
