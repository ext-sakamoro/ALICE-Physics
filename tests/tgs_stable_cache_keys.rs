//! `SolverBackend::Tgs` warm-starts each contact and distance constraint from
//! the impulse it carried last tick. The cache key must name the same contact
//! / constraint from one tick to the next even when the world's vectors are
//! reordered (a body removed with `swap_remove`, a constraint removed from the
//! middle of `distance_constraints`).
//!
//! oracle: twin worlds. World X carries an extra element and drops it
//! mid-run; world Y never had it. The extra element never interacts with
//! anything, so from the removal on the two worlds hold the same bodies, the
//! same constraints and the same cached impulses, and must step bit for bit
//! the same. A key built from vector positions hands a cached impulse to a
//! different contact / constraint after the reorder, and the twins diverge.

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn tgs_world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        substeps: 4,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    })
}

fn snapshot(w: &PhysicsWorld) -> Vec<(Vec3Fix, Vec3Fix)> {
    w.bodies.iter().map(|b| (b.position, b.velocity)).collect()
}

/// Balls of different masses resting on their own static ground spheres, far
/// apart: four independent contacts, each with its own steady impulse (the
/// masses differ, so a contact handed another contact's impulse is wrong).
fn add_resting_stacks(w: &mut PhysicsWorld) {
    for k in 0..4i64 {
        let x = 100 * k;
        let ground = w.add_body(RigidBody::new_static(Vec3Fix::from_int(x, 0, 0)));
        w.set_body_collision_radius(ground, Fix128::from_int(10));
        let ball = w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(x, 11, 0),
            Fix128::from_int(k + 1),
        ));
        w.set_body_collision_radius(ball, Fix128::ONE);
    }
}

/// World X has one more ball, at the lowest indices, that falls onto its own
/// ground after the other contacts have warmed up. Its contact is detected
/// first, so from that tick on every other contact sits one place further
/// down the contact list. The extra pair never touches the others, and the
/// remaining bodies keep their relative order (so detection orients every
/// other pair the same way in both worlds).
#[test]
fn a_new_contact_ahead_in_the_list_does_not_shuffle_the_others() {
    let dt = r(1, 60);
    let mut x = tgs_world();
    let extra_ground = x.add_body(RigidBody::new_static(Vec3Fix::from_int(-1000, 0, 0)));
    x.set_body_collision_radius(extra_ground, Fix128::from_int(10));
    let extra_ball = x.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(-1000, 13, 0),
        Fix128::from_int(7),
    ));
    x.set_body_collision_radius(extra_ball, Fix128::ONE);
    add_resting_stacks(&mut x);
    let mut y = tgs_world();
    add_resting_stacks(&mut y);

    let mut touched = false;
    for frame in 0..90 {
        x.step(dt);
        y.step(dt);
        touched |= x.bodies[extra_ball].position.y < Fix128::from_int(11);
        assert_eq!(
            snapshot(&x)[2..],
            snapshot(&y)[..],
            "frame {frame}: a contact warm-started from another contact's impulse"
        );
    }
    assert!(
        touched,
        "the extra ball must reach its ground inside the run"
    );
}

/// Two pendulums with different bob masses, each on its own distance
/// constraint, plus a first constraint between two far, free bodies that is
/// removed mid-run.
#[test]
fn removing_a_distance_constraint_keeps_the_others_on_their_own_impulse() {
    let dt = r(1, 60);
    let build = |with_extra: bool| {
        let mut w = tgs_world();
        let a = w.add_body(RigidBody::new_static(Vec3Fix::from_int(-500, 0, 0)));
        let b = w.add_body(RigidBody::new_static(Vec3Fix::from_int(-505, 0, 0)));
        if with_extra {
            w.add_distance_constraint(DistanceConstraint::new(
                a,
                b,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Fix128::from_int(5),
            ));
        }
        for k in 0..2i64 {
            let pivot = w.add_body(RigidBody::new_static(Vec3Fix::from_int(100 * k, 0, 0)));
            let mut bob = RigidBody::new_dynamic(
                Vec3Fix::from_int(100 * k + 5, 0, 0),
                Fix128::from_int(k + 1),
            );
            bob.velocity = Vec3Fix::from_int(0, 3 * (k + 1), 0);
            let bob = w.add_body(bob);
            w.add_distance_constraint(DistanceConstraint::new(
                pivot,
                bob,
                Vec3Fix::ZERO,
                Vec3Fix::ZERO,
                Fix128::from_int(5),
            ));
        }
        w
    };
    let mut x = build(true);
    let mut y = build(false);
    for _ in 0..20 {
        x.step(dt);
        y.step(dt);
    }
    assert_eq!(
        snapshot(&x),
        snapshot(&y),
        "the twins must agree before the removal"
    );
    x.distance_constraints.remove(0);
    for frame in 0..30 {
        x.step(dt);
        y.step(dt);
        assert_eq!(
            snapshot(&x),
            snapshot(&y),
            "frame {frame} after the removal: a constraint warm-started from another constraint's impulse"
        );
    }
}

/// Removing a body can flip the index order of two bodies that stay in
/// contact (`swap_remove` moves the last body to the front). The contact keeps
/// its stable ids, so its warm start must still hit.
#[test]
fn a_pair_whose_index_order_flips_still_hits_its_cached_impulse() {
    let dt = r(1, 60);
    let mut w = tgs_world();
    let float = w.add_body(RigidBody::new_static(Vec3Fix::from_int(-1000, 500, 0)));
    w.set_body_collision_radius(float, Fix128::ONE);
    let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.set_body_collision_radius(ground, Fix128::from_int(10));
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 11, 0),
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, Fix128::ONE);
    // Let the ball settle into its resting penetration (TGS detects contacts
    // at start-of-tick positions; a sphere exactly touching is not a contact,
    // so the first ticks alternate between touching and not).
    for _ in 0..120 {
        w.step(dt);
    }
    w.reset_tgs_cache_stats();
    w.step(dt);
    let before = w.tgs_cache_stats();
    assert!(
        before.hits > 0 && before.misses == 0,
        "control tick: {before:?}"
    );
    assert!(w.remove_body(float).is_some());
    // Now the ball is body 0 and the ground body 1: detection orders the pair
    // the other way round.
    assert_eq!(w.bodies[0].position.x, Fix128::ZERO);
    assert!(w.bodies[1].is_static());
    w.reset_tgs_cache_stats();
    w.step(dt);
    let stats = w.tgs_cache_stats();
    assert!(
        stats.hits > 0,
        "the contact must have been looked up: {stats:?}"
    );
    assert_eq!(
        stats.misses, 0,
        "the flipped pair must warm-start from its own impulse: {stats:?}"
    );
}

/// A pendulum keeps hitting its own cached impulse when `remove_body`
/// reshuffles the world around it. Two layouts, both removing an unrelated
/// body at index 0:
/// - nothing else added: the bob moves from index 2 to 0, so the constraint is
///   renumbered;
/// - a body pushed onto the public `bodies` field just before (it has no id
///   yet): it moves to index 0, and the pendulum's ids must not shift.
#[test]
fn a_pendulum_keeps_its_impulse_across_remove_body() {
    let dt = r(1, 60);
    for push_unregistered in [false, true] {
        let mut w = tgs_world();
        w.add_body(RigidBody::new_static(Vec3Fix::from_int(-1000, 0, 0)));
        let pivot = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        let mut bob = RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
        bob.velocity = Vec3Fix::from_int(0, 3, 0);
        let bob = w.add_body(bob);
        w.add_distance_constraint(DistanceConstraint::new(
            pivot,
            bob,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Fix128::from_int(5),
        ));
        for _ in 0..5 {
            w.step(dt);
        }
        if push_unregistered {
            w.bodies
                .push(RigidBody::new_static(Vec3Fix::from_int(1000, 0, 0)));
        }
        assert!(w.remove_body(0).is_some());
        let c = w.distance_constraints[0];
        if !push_unregistered {
            assert_eq!(
                (c.body_a, c.body_b),
                (pivot, 0),
                "the bob must have moved to index 0"
            );
        }
        w.reset_tgs_cache_stats();
        w.step(dt);
        let stats = w.tgs_cache_stats();
        assert!(
            stats.hits > 0,
            "push_unregistered={push_unregistered}: {stats:?}"
        );
        assert_eq!(
            stats.misses, 0,
            "push_unregistered={push_unregistered}: the pendulum must warm-start from its own impulse: {stats:?}"
        );
    }
}
