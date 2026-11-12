//! Body axis locks: a dynamic body whose world axes are locked keeps the
//! locked coordinates of its pose exactly and moves on the free axes as a
//! body that never had a velocity along the locked ones.
//!
//! oracle (closed form, independent of the projection code): a locked
//! position coordinate is the value it had when the lock was set, bit for bit;
//! the locked velocity components are exactly zero; on the free axes the
//! locked body is bit for bit the twin that started with the locked velocity
//! components already zero (integration does not couple the axes of a free
//! body under uniform gravity); with both tilting axes locked the orientation
//! stays a rotation about the free axis (its other two vector components are
//! exactly zero from the identity); a lock of all three angular axes keeps the
//! orientation bit for bit.
//!
//! The side table follows `remove_body` (swap-remove), the snapshot keeps the
//! locks, a world without locks writes the snapshot and law identifier it
//! wrote before locks existed, and a lock changes both.

use alice_physics_core::{
    AxisLock, BallJoint, Fix128, Joint, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody,
    SolverBackend, Vec3Fix,
};

fn i(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

fn world(backend: SolverBackend) -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(-9.81), Fix128::ZERO),
        solver_backend: backend,
        ..PhysicsConfig::default()
    })
}

const BACKENDS: [SolverBackend; 2] = [SolverBackend::Xpbd, SolverBackend::Tgs];

fn thrown(v: Vec3Fix, w: Vec3Fix) -> RigidBody {
    let mut b = RigidBody::new_dynamic(Vec3Fix::new(i(1), i(20), i(3)), Fix128::ONE);
    b.velocity = v;
    b.angular_velocity = w;
    b
}

/// A locked position axis stays at its value bit for bit and its velocity is
/// zero; the free axes are the twin's that started without that velocity.
#[test]
fn a_locked_position_axis_stays_exact_and_the_free_axes_follow_the_twin() {
    for backend in BACKENDS {
        let mut locked = world(backend);
        let a = locked.add_body(thrown(Vec3Fix::new(i(2), i(5), i(-7)), Vec3Fix::ZERO));
        assert!(locked.set_body_axis_lock(a, AxisLock::LINEAR_Z));
        let mut twin = world(backend);
        let b = twin.add_body(thrown(
            Vec3Fix::new(i(2), i(5), Fix128::ZERO),
            Vec3Fix::ZERO,
        ));
        for _ in 0..90 {
            locked.step(dt());
            twin.step(dt());
            let (p, q) = (&locked.bodies[a], &twin.bodies[b]);
            assert_eq!(p.position.z, i(3), "{backend:?}");
            assert_eq!(p.velocity.z, Fix128::ZERO, "{backend:?}");
            assert_eq!((p.position.x, p.position.y), (q.position.x, q.position.y));
            assert_eq!((p.velocity.x, p.velocity.y), (q.velocity.x, q.velocity.y));
        }
        // the free axes moved: the lock is not a freeze
        assert_ne!(locked.bodies[a].position.y, i(20));
    }
}

/// With the x and y rotations locked the body turns about z only, as the twin
/// that spun about z alone; all three locked keeps the orientation.
#[test]
fn locked_rotation_axes_leave_a_turn_about_the_free_axis() {
    for backend in BACKENDS {
        let mut locked = world(backend);
        let a = locked.add_body(thrown(Vec3Fix::ZERO, Vec3Fix::new(i(1), i(2), i(3))));
        assert!(locked.set_body_axis_lock(a, AxisLock::ANGULAR_X | AxisLock::ANGULAR_Y));
        let mut twin = world(backend);
        let b = twin.add_body(thrown(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, i(3)),
        ));
        for _ in 0..90 {
            locked.step(dt());
            twin.step(dt());
            let (p, q) = (&locked.bodies[a], &twin.bodies[b]);
            assert_eq!(
                (p.rotation.x, p.rotation.y),
                (Fix128::ZERO, Fix128::ZERO),
                "{backend:?}"
            );
            assert_eq!(
                (p.angular_velocity.x, p.angular_velocity.y),
                (Fix128::ZERO, Fix128::ZERO)
            );
            assert_eq!(p.rotation, q.rotation, "{backend:?}");
            assert_eq!(p.angular_velocity, q.angular_velocity, "{backend:?}");
        }
        assert_ne!(locked.bodies[a].rotation, QuatFix::IDENTITY);

        let mut all = world(backend);
        let mut body = thrown(Vec3Fix::ZERO, Vec3Fix::new(i(1), i(2), i(3)));
        body.rotation =
            QuatFix::from_axis_angle(Vec3Fix::new(i(1), i(1), Fix128::ZERO), Fix128::ONE);
        let c = all.add_body(body);
        let start = all.bodies[c].rotation;
        all.set_body_axis_lock(c, AxisLock::ANGULAR);
        for _ in 0..60 {
            all.step(dt());
            assert_eq!(all.bodies[c].rotation, start, "{backend:?}");
            assert_eq!(all.bodies[c].angular_velocity, Vec3Fix::ZERO);
        }
    }
}

/// A contact and a joint that push along a locked axis do not move the body on
/// it: a ball dropped on a locked-y body leaves its y exact, and a joint that
/// pulls a locked-x body sideways leaves its x exact.
#[test]
fn contacts_and_joints_do_not_move_a_locked_axis() {
    for backend in BACKENDS {
        let mut w = world(backend);
        let r = Fix128::ONE;
        let shelf = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, i(2), Fix128::ZERO), i(4)),
            r,
        );
        w.set_body_axis_lock(shelf, AxisLock::LINEAR);
        let ball = w.add_body_with_radius(
            RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, i(5), Fix128::ZERO), Fix128::ONE),
            r,
        );
        for _ in 0..120 {
            w.step(dt());
            assert_eq!(
                w.bodies[shelf].position,
                Vec3Fix::new(Fix128::ZERO, i(2), Fix128::ZERO)
            );
        }
        // the ball rests on the shelf: it was pushed back by the locked body
        let top = w.bodies[ball].position.y;
        assert!(top > i(3) && top < i(5), "{backend:?}: ball at {top:?}");

        let mut j = world(backend);
        let anchor = j.add_body(RigidBody::new_static(Vec3Fix::new(
            i(4),
            i(10),
            Fix128::ZERO,
        )));
        let bob = j.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, i(10), Fix128::ZERO),
            Fix128::ONE,
        ));
        j.set_body_axis_lock(bob, AxisLock::LINEAR_X);
        j.add_joint(Joint::Ball(BallJoint::new(
            anchor,
            bob,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
        )));
        for _ in 0..60 {
            j.step(dt());
            assert_eq!(j.bodies[bob].position.x, Fix128::ZERO, "{backend:?}");
            assert_eq!(j.bodies[bob].velocity.x, Fix128::ZERO, "{backend:?}");
        }
    }
}

/// A bounce along a locked axis leaves no velocity on it: a free ball dropped with full
/// restitution onto a body whose y is locked is pushed back, and the restitution of the
/// contact, which the velocity update adds after the pose projection, is removed from the
/// locked body too (its y velocity is exactly zero after every step, in the snapshot too).
#[test]
fn a_bounce_along_a_locked_axis_leaves_no_velocity_on_it() {
    for backend in BACKENDS {
        let mut w = world(backend);
        let mut base = RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, i(2), Fix128::ZERO), i(4));
        base.restitution = Fix128::ONE;
        let locked = w.add_body_with_radius(base, Fix128::ONE);
        w.set_body_axis_lock(locked, AxisLock::LINEAR_Y);
        let mut ball =
            RigidBody::new_dynamic(Vec3Fix::new(Fix128::ZERO, i(5), Fix128::ZERO), Fix128::ONE);
        ball.restitution = Fix128::ONE;
        ball.velocity = Vec3Fix::new(Fix128::ZERO, i(-6), Fix128::ZERO);
        let ball = w.add_body_with_radius(ball, Fix128::ONE);
        let mut touched = false;
        for _ in 0..60 {
            w.step(dt());
            let b = &w.bodies[locked];
            assert_eq!(b.position.y, i(2), "{backend:?}");
            assert_eq!(b.velocity.y, Fix128::ZERO, "{backend:?}: {:?}", b.velocity);
            touched |= w.bodies[ball].velocity.y > Fix128::ZERO;
        }
        assert!(
            touched,
            "{backend:?}: the ball never bounced off the locked body"
        );
        let mut back = world(backend);
        back.restore_world(&w.snapshot_world()).expect("restore");
        assert_eq!(back.bodies[locked].velocity.y, Fix128::ZERO);
    }
}

/// The lock table follows the swap-remove: removing the first of three bodies
/// moves the last body, and its lock, into index 0.
#[test]
fn remove_body_keeps_the_lock_table_aligned() {
    let mut w = world(SolverBackend::Xpbd);
    for k in 0..3 {
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(i(k * 5), i(10), Fix128::ZERO),
            Fix128::ONE,
        ));
    }
    assert!(w.set_body_axis_lock(2, AxisLock::LINEAR_Y));
    assert!(w.set_body_axis_lock(1, AxisLock::ANGULAR_Z));
    assert!(!w.set_body_axis_lock(3, AxisLock::LINEAR_Y));
    assert_eq!(w.body_axis_lock(3), None);
    w.remove_body(0);
    assert_eq!(w.body_axis_lock(0), Some(AxisLock::LINEAR_Y));
    assert_eq!(w.body_axis_lock(1), Some(AxisLock::ANGULAR_Z));
    for _ in 0..30 {
        w.step(dt());
    }
    assert_eq!(w.bodies[0].position.y, i(10));
    assert!(w.bodies[1].position.y < i(10));
    // a body pushed onto the public field has no lock
    w.bodies
        .push(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    assert_eq!(w.body_axis_lock(2), Some(AxisLock::NONE));
    w.remove_body(0);
    assert_eq!(w.body_axis_lock(0), Some(AxisLock::NONE));
    assert_eq!(w.body_axis_lock(1), Some(AxisLock::ANGULAR_Z));
}

/// Without locks the law identifier is the one written before locks existed and
/// the snapshot holds a lock count of 0 (version 8 onward always writes the
/// section); a lock changes both and survives a snapshot round trip, and
/// clearing it gives the lock-free bytes back.
#[test]
fn the_snapshot_and_law_id_carry_locks_only_when_there_are_some() {
    let build = || {
        let mut w = world(SolverBackend::Xpbd);
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(i(0), i(10), i(0)),
            Fix128::ONE,
        ));
        w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(i(3), i(10), i(0)),
            Fix128::ONE,
        ));
        w
    };
    let sem = [7u8; 32];
    let plain = build();
    let blob = plain.snapshot_world();
    assert_eq!(
        u16::from_le_bytes([blob[4], blob[5]]),
        10,
        "a lock-free world writes the current version"
    );

    let mut locked = build();
    locked.set_body_axis_lock(1, AxisLock::LINEAR_Z | AxisLock::ANGULAR_X);
    assert_ne!(locked.law_id(&sem), plain.law_id(&sem));
    let mut other = build();
    other.set_body_axis_lock(1, AxisLock::LINEAR_Z);
    assert_ne!(other.law_id(&sem), locked.law_id(&sem));
    let mut moved = build();
    moved.set_body_axis_lock(0, AxisLock::LINEAR_Z | AxisLock::ANGULAR_X);
    assert_ne!(moved.law_id(&sem), locked.law_id(&sem));

    let lblob = locked.snapshot_world();
    assert_eq!(u16::from_le_bytes([lblob[4], lblob[5]]), 10);
    assert_ne!(lblob, blob);
    assert_ne!(locked.state_digest(), plain.state_digest());
    let mut back = build();
    back.restore_world(&lblob).expect("restore");
    assert_eq!(
        back.body_axis_lock(1),
        Some(AxisLock::LINEAR_Z | AxisLock::ANGULAR_X)
    );
    assert_eq!(back.snapshot_world(), lblob);

    // a lock-free blob restores as a world without locks
    locked.restore_world(&blob).expect("restore lock-free");
    assert_eq!(locked.body_axis_lock(1), Some(AxisLock::NONE));
    assert_eq!(locked.snapshot_world(), blob);
    assert_eq!(locked.law_id(&sem), plain.law_id(&sem));

    locked.set_body_axis_lock(1, AxisLock::LINEAR_Z);
    locked.set_body_axis_lock(1, AxisLock::NONE);
    assert_eq!(locked.snapshot_world(), blob);
}

/// The mask is six bits; any other bit is refused.
#[test]
fn the_mask_is_six_bits() {
    assert_eq!(AxisLock::from_bits(0), Some(AxisLock::NONE));
    assert_eq!(AxisLock::from_bits(63), Some(AxisLock::ALL));
    assert_eq!(AxisLock::from_bits(64), None);
    assert_eq!(AxisLock::LINEAR | AxisLock::ANGULAR, AxisLock::ALL);
    assert_eq!(AxisLock::LINEAR_X.bits(), 1);
    assert_eq!(AxisLock::ANGULAR_Z.bits(), 32);
    assert!(AxisLock::ALL.contains(AxisLock::ANGULAR_Y));
    assert!(!AxisLock::LINEAR.contains(AxisLock::ANGULAR_Y));
}
