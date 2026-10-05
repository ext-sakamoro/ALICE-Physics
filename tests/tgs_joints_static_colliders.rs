//! `SolverBackend::Tgs` must honour the world's joints (`PhysicsWorld::add_joint`)
//! and static colliders (`PhysicsWorld::add_static_collider`) the way the XPBD
//! path does.
//!
//! Both are position-level corrections in this engine. TGS owns velocities, so
//! a correction that moves a body must also show up in its velocity —
//! otherwise a resting body keeps the velocity gravity gave it and sinks again
//! the next tick, and a pendulum gains energy every tick.
//!
//! oracles (closed form, computed here, never by calling `step`):
//! - a sphere of radius `r` resting on the plane `y = 0` sits at `y = r` with
//!   zero vertical velocity;
//! - a ball-jointed pendulum keeps its arm length `L`, its bob turns with the
//!   arm (the anchor is fixed in the bob), and its translational energy
//!   `½|v|² + g·y` (per unit mass) cannot grow above its start value.

#![cfg(feature = "std")]

use alice_physics::joint::{BallJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::SolverBackend;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Every oracle runs at several substep counts: the corrections happen once
/// per tick, after TGS's own substeps, and must not depend on how many there
/// are.
const SUBSTEPS: [usize; 3] = [1, 4, 8];

fn tgs_world(substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        substeps,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    })
}

#[test]
fn a_sphere_dropped_on_a_static_plane_rests_on_it_under_tgs() {
    for substeps in SUBSTEPS {
        sphere_rests_on_plane(substeps);
    }
}

fn sphere_rests_on_plane(substeps: usize) {
    let dt = r(1, 60);
    let radius = r(1, 2);
    let mut w = tgs_world(substeps);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 3, 0),
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, radius);
    let mut lowest = w.bodies[ball].position.y;
    for _ in 0..180 {
        w.step(dt);
        lowest = lowest.min(w.bodies[ball].position.y);
    }
    let b = &w.bodies[ball];
    let tol = r(1, 100);
    assert!(
        (b.position.y - radius).abs() <= tol,
        "substeps {substeps}: rest height must be the radius {radius:?}, got {:?}",
        b.position.y
    );
    assert!(
        lowest >= radius - tol,
        "substeps {substeps}: the sphere must not sink into the plane on the way (lowest {lowest:?})"
    );
    assert!(
        b.velocity.y.abs() <= tol,
        "substeps {substeps}: a resting sphere must have no vertical velocity, got {:?}",
        b.velocity.y
    );
}

#[test]
fn a_ball_jointed_pendulum_keeps_its_length_and_does_not_gain_energy_under_tgs() {
    for substeps in SUBSTEPS {
        pendulum_keeps_length_and_energy(substeps);
    }
}

fn pendulum_keeps_length_and_energy(substeps: usize) {
    let dt = r(1, 60);
    let arm = Fix128::from_int(2);
    let g = Fix128::from_int(10);
    let mut w = tgs_world(substeps);
    let pivot = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let bob = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 0, 0),
        Fix128::ONE,
    ));
    w.add_joint(Joint::Ball(BallJoint::new(
        pivot,
        bob,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
    )));
    let energy = |b: &RigidBody| {
        let v = b.velocity;
        (v.x * v.x + v.y * v.y + v.z * v.z) / Fix128::from_int(2) + g * b.position.y
    };
    let e0 = energy(&w.bodies[bob]);
    let len_tol = r(1, 20);
    let e_tol = r(1, 20);
    let mut swung_down = false;
    for frame in 0..240 {
        w.step(dt);
        let b = &w.bodies[bob];
        // The bob's anchor sits 2 units from its centre, so measure the arm
        // to the anchor's world position.
        let anchor = b.position + b.rotation.rotate_vec(Vec3Fix::from_int(-2, 0, 0));
        let reach = anchor.length();
        assert!(
            reach <= len_tol,
            "substeps {substeps}, frame {frame}: the bob's anchor must stay on the pivot (off by {reach:?})"
        );
        // The anchor is the bob's local `-x` at distance 2, and it sits on
        // the pivot, so the bob's local `+x` axis must point from the pivot
        // to the bob: orientation and position swing together.
        let axis = b.rotation.rotate_vec(Vec3Fix::UNIT_X);
        let radial = b.position / arm;
        let misalign = (axis - radial).length();
        assert!(
            misalign <= len_tol,
            "substeps {substeps}, frame {frame}: the bob must turn with the arm (axis off by {misalign:?})"
        );
        let d = b.position.length();
        assert!(
            (d - arm).abs() <= len_tol,
            "substeps {substeps}, frame {frame}: arm length must stay {arm:?}, got {d:?}"
        );
        assert!(
            energy(b) <= e0 + e_tol,
            "substeps {substeps}, frame {frame}: energy grew from {e0:?} to {:?}",
            energy(b)
        );
        swung_down |= b.position.y < -Fix128::ONE;
    }
    assert!(
        swung_down,
        "substeps {substeps}: the pendulum must actually swing under gravity"
    );
}
