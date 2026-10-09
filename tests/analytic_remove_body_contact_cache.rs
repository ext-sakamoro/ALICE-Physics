//! `PhysicsWorld::remove_body` must leave `contact_cache` exactly as a world
//! that never contained the removed body would have it.
//!
//! The cache is keyed by body index pairs (`BodyPairKey`, canonical
//! `body_a < body_b`). `remove_body` swap-removes: the last body moves into
//! the freed slot `idx`. So, per manifold:
//! - a pair that involves `idx` describes the removed body; no body in the
//!   control world has those contacts, so it is dropped;
//! - a pair that involves the old last index describes the moved body; in
//!   the control world that body already sits at `idx`, so the same data is
//!   cached under the pair with `last` replaced by `idx`, re-sorted by
//!   `BodyPairKey::new` (the cache never reorders the stored points / normal
//!   by the key, so the data is carried over unchanged);
//! - every other pair is untouched.
//!
//! Oracle: the control world receives the same contacts (indices mapped,
//! the removed body's contacts skipped) in the same order, with the same
//! stored impulses. Manifold list, lookups by every pair and the velocities
//! produced by `apply_warm_start` must agree bit for bit. Before the fix the
//! removed body's manifolds stayed (and a later contact for the moved body
//! at the removed body's contact point picked up the removed body's
//! impulses), and the moved body's manifolds kept its old index.
//!
//! `step()` does not read the cache; it is reached through `add_contact`,
//! `find` / `get_or_create`, `warm_start_impulse` and `apply_warm_start`.

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::contact_cache::{BodyPairKey, ContactManifold};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsWorld, RigidBody, SolverConfig};

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3r(x: i64, y: i64, z: i64, d: i64) -> Vec3Fix {
    Vec3Fix::new(fr(x, d), fr(y, d), fr(z, d))
}

/// Body `k` of the original world: dynamic, distinct mass and position.
fn body(k: i64) -> RigidBody {
    RigidBody::new_dynamic(v3r(3 * k, 1, -k, 1), fr(k + 2, 2))
}

/// A contact recorded for the `k`-th call: point, normal and depth all
/// distinct, so a manifold picked up for the wrong pair is visible.
fn contact_k(k: i64) -> Contact {
    Contact {
        depth: fr(k + 1, 100),
        normal: v3r(1, 2 + k, -2, 7).normalize(),
        point_a: v3r(k, 1, 0, 4),
        point_b: v3r(-k, 0, 1, 4),
    }
}

/// Stored impulses for the `k`-th call (what a solver would write back).
fn lambdas_k(k: i64) -> (Fix128, Fix128, Fix128) {
    (fr(k + 1, 3), fr(-(k + 2), 11), fr(k + 5, 13))
}

/// Add contact `k` between `a` and `b` (in that order) and store its
/// impulses on the manifold's first point.
fn add(w: &mut PhysicsWorld, k: i64, a: usize, b: usize) {
    let c = contact_k(k);
    w.add_contact(ContactConstraint {
        body_a: a,
        body_b: b,
        contact: c,
        friction: fr(k + 1, 5),
        restitution: fr(k, 9),
        cached_lambda: Fix128::ZERO,
    });
    let (n, t1, t2) = lambdas_k(k);
    w.contact_cache
        .get_or_create(BodyPairKey::new(a, b), Fix128::ZERO, Fix128::ZERO)
        .store_impulses(0, n, t1, t2);
}

/// Every field of a manifold, comparable bit for bit.
fn snapshot(m: &ContactManifold) -> String {
    format!(
        "{:?} {:?} {:?} {:?} {:?} {}",
        m.pair, m.points, m.normal, m.friction, m.restitution, m.stale_frames
    )
}

fn assert_same_cache(got: &PhysicsWorld, want: &PhysicsWorld, n: usize) {
    let g: Vec<String> = got.contact_cache.manifolds.iter().map(snapshot).collect();
    let w: Vec<String> = want.contact_cache.manifolds.iter().map(snapshot).collect();
    assert_eq!(g, w, "manifold list differs from the control world");
    for a in 0..n {
        for b in a..n {
            let key = BodyPairKey::new(a, b);
            assert_eq!(
                got.contact_cache.find(&key).map(snapshot),
                want.contact_cache.find(&key).map(snapshot),
                "find({a}, {b}) differs"
            );
        }
    }
    let mut vg = got.bodies.clone();
    let mut vw = want.bodies.clone();
    got.contact_cache.apply_warm_start(&mut vg);
    want.contact_cache.apply_warm_start(&mut vw);
    for (i, (x, y)) in vg.iter().zip(&vw).enumerate() {
        assert_eq!(x.velocity, y.velocity, "warm-started velocity of body {i}");
    }
}

/// The original world's contacts, as (k, body_a, body_b) in call order.
/// Remove idx = 1 of 5 bodies (last = 4):
/// - k0 (0,1), k3 (1,3), k5 (1,4): involve the removed body;
/// - k1 (0,4): moved body, other index below idx, order kept → (0,1);
/// - k2 (2,4) and k6 (4,3): moved body, other index above idx, the
///   canonical order flips → (1,2) / (1,3);
/// - k4 (2,3): untouched.
const CALLS: [(i64, usize, usize); 7] = [
    (0, 0, 1),
    (1, 0, 4),
    (2, 2, 4),
    (3, 1, 3),
    (4, 2, 3),
    (5, 1, 4),
    (6, 4, 3),
];

fn original() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    for k in 0..5 {
        w.add_body(body(k));
    }
    for &(k, a, b) in &CALLS {
        add(&mut w, k, a, b);
    }
    w
}

/// Control world: never had body 1; body 4 sits at index 1 from the start.
fn control() -> PhysicsWorld {
    let map = |i: usize| -> Option<usize> {
        match i {
            0 => Some(0),
            1 => None,
            2 => Some(2),
            3 => Some(3),
            4 => Some(1),
            _ => unreachable!(),
        }
    };
    let mut w = PhysicsWorld::new(SolverConfig::default());
    for k in [0, 4, 2, 3] {
        w.add_body(body(k));
    }
    for &(k, a, b) in &CALLS {
        if let (Some(a), Some(b)) = (map(a), map(b)) {
            add(&mut w, k, a, b);
        }
    }
    w
}

#[test]
fn removing_a_middle_body_leaves_the_cache_of_a_world_without_it() {
    let mut w = original();
    assert_eq!(w.contact_cache.manifold_count(), 7);
    w.remove_body(1).expect("index 1 exists");
    let c = control();
    assert_eq!(c.contact_cache.manifold_count(), 4);
    assert_same_cache(&w, &c, 4);
}

#[test]
fn a_later_contact_for_the_moved_body_does_not_pick_up_the_removed_bodys_impulses() {
    // Body 4 (moved to 1) now touches body 0 at the point where the removed
    // body touched it (k0's point). The removed body's manifold (0,1) must
    // be gone: the contact lands in the moved body's own manifold (k1's,
    // remapped from (0,4)) as a new point with zero impulses.
    let mut w = original();
    w.remove_body(1).expect("index 1 exists");
    let mut c = control();
    for world in [&mut w, &mut c] {
        world.add_contact(ContactConstraint {
            body_a: 0,
            body_b: 1,
            contact: contact_k(0),
            friction: Fix128::ONE,
            restitution: Fix128::ZERO,
            cached_lambda: Fix128::ZERO,
        });
    }
    let m = w
        .contact_cache
        .find(&BodyPairKey::new(0, 1))
        .expect("pair (0,1) cached");
    assert_eq!(m.point_count(), 2, "k1's point plus the new one");
    assert_eq!(m.points[0].local_point_a, contact_k(1).point_a);
    let fresh = m.warm_start_impulse(1);
    assert_eq!(
        fresh,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        "removed body's impulses {:?} picked up",
        lambdas_k(0)
    );
    assert_same_cache(&w, &c, 4);
}

#[test]
fn removing_the_last_body_only_drops_its_pairs() {
    let mut w = original();
    w.remove_body(4).expect("index 4 exists");
    let mut c = PhysicsWorld::new(SolverConfig::default());
    for k in 0..4 {
        c.add_body(body(k));
    }
    for &(k, a, b) in &CALLS {
        if a != 4 && b != 4 {
            add(&mut c, k, a, b);
        }
    }
    assert_eq!(c.contact_cache.manifold_count(), 3);
    assert_same_cache(&w, &c, 4);
}

#[test]
fn the_cache_stays_consistent_for_lookups_and_frames_after_a_removal() {
    // After the removal, get_or_create on a remapped pair must find the
    // existing manifold (no duplicate), and aging must treat it like any
    // other manifold.
    let mut w = original();
    w.remove_body(1).expect("index 1 exists");
    let before = w.contact_cache.manifold_count();
    let key = BodyPairKey::new(1, 2); // k2, remapped from (2,4)
    let lambda = w
        .contact_cache
        .get_or_create(key, Fix128::ZERO, Fix128::ZERO)
        .warm_start_impulse(0);
    assert_eq!(lambda, lambdas_k(2));
    assert_eq!(w.contact_cache.manifold_count(), before, "no duplicate");
    let mut c = control();
    for world in [&mut w, &mut c] {
        for _ in 0..4 {
            world.begin_frame();
            world.end_frame();
        }
    }
    assert_eq!(w.contact_cache.manifold_count(), 0);
    assert_same_cache(&w, &c, 4);
}
