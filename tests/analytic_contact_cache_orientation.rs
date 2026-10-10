//! `PhysicsWorld::add_contact` must store a contact in the contact cache
//! the same way whichever of the two bodies the caller names first.
//!
//! The cache keys a manifold by the sorted pair (`BodyPairKey`, `body_a <
//! body_b`), and `ContactCache::apply_warm_start` pushes the impulse along
//! `+normal` on `pair.body_a` and along `-normal` on `pair.body_b` (normal
//! pointing from B to A). A contact given as `(body_a = j, body_b = i)` with
//! `j > i` describes the same touch as `(i, j)` with the normal negated and
//! the two contact points swapped, so the cache must hold the same data for
//! both, expressed with `pair.body_a` as A.
//!
//! Oracle: hand-computed velocities for an axis-aligned normal (every value
//! dyadic, warm start factor 1, so the expected numbers are exact), and bit
//! equality of the cached data and of the warm-started velocities between
//! the two call orders for general normals.

#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::contact_cache::BodyPairKey;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{ContactConstraint, PhysicsWorld, RigidBody, SolverConfig};

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn v3r(x: i64, y: i64, z: i64, d: i64) -> Vec3Fix {
    Vec3Fix::new(fr(x, d), fr(y, d), fr(z, d))
}

/// Two bodies at rest: body 0 of mass 2 (inverse 1/2), body 1 of mass 4
/// (inverse 1/4); warm start factor 1.
fn world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body(RigidBody::new_dynamic(v3r(0, 0, 0, 1), fr(2, 1)));
    w.add_body(RigidBody::new_dynamic(v3r(0, 1, 0, 1), fr(4, 1)));
    w.contact_cache.warm_start_factor = Fix128::ONE;
    w
}

/// Add the touch "body 0 is A, body 1 is B, normal `n` from B to A" in the
/// canonical order `(0, 1)` or, with `swapped`, as `(1, 0)` with the normal
/// negated and the points swapped (the same touch described from body 1).
fn add_touch(w: &mut PhysicsWorld, swapped: bool, n: Vec3Fix, p0: Vec3Fix, p1: Vec3Fix) {
    let depth = fr(1, 64);
    let (body_a, body_b, contact) = if swapped {
        (
            1,
            0,
            Contact {
                depth,
                normal: -n,
                point_a: p1,
                point_b: p0,
            },
        )
    } else {
        (
            0,
            1,
            Contact {
                depth,
                normal: n,
                point_a: p0,
                point_b: p1,
            },
        )
    };
    w.add_contact(ContactConstraint {
        body_a,
        body_b,
        contact,
        friction: fr(1, 2),
        restitution: Fix128::ZERO,
        cached_lambda: Fix128::ZERO,
    });
}

/// Store (λn, λt1, λt2) on point 0 of the (0, 1) manifold and warm start.
fn warm_start(w: &PhysicsWorld, l: (Fix128, Fix128, Fix128)) -> Vec<Vec3Fix> {
    let mut cache_bodies = w.bodies.clone();
    let mut cache = alice_physics::contact_cache::ContactCache::new();
    cache.warm_start_factor = w.contact_cache.warm_start_factor;
    let m = w
        .contact_cache
        .find(&BodyPairKey::new(0, 1))
        .expect("pair cached")
        .clone();
    cache.manifolds.push(m);
    cache.manifolds[0].store_impulses(0, l.0, l.1, l.2);
    cache.apply_warm_start(&mut cache_bodies);
    cache_bodies.iter().map(|b| b.velocity).collect()
}

#[test]
fn either_call_order_warm_starts_to_the_hand_computed_velocities() {
    // oracle (hand): n = +y, tangent frame of +y = (t1, t2) = (-z, -x)
    // (reference axis x; t1 = y × x = -z, t2 = y × t1 = -x).
    // λ = (3/4, 1/2, -1/4): T = 3/4·y + 1/2·(-z) - 1/4·(-x) = (1/4, 3/4, -1/2)
    // body 0 (A): +T/2 = (1/8, 3/8, -1/4); body 1 (B): -T/4 = (-1/16, -3/16, 1/8)
    let lambdas = (fr(3, 4), fr(1, 2), fr(-1, 4));
    let want = [v3r(2, 6, -4, 16), v3r(-1, -3, 2, 16)];
    for swapped in [false, true] {
        let mut w = world();
        add_touch(
            &mut w,
            swapped,
            Vec3Fix::UNIT_Y,
            v3r(0, 1, 0, 2),
            v3r(0, -1, 0, 2),
        );
        let got = warm_start(&w, lambdas);
        assert_eq!(got, want, "swapped = {swapped}");
    }
}

#[test]
fn either_call_order_caches_the_same_manifold() {
    // general normals: the cached data (normal, both points, depth) and the
    // warm-started velocities are bit-identical between the two orders
    for k in 0..6i64 {
        let n = v3r(1 + k, 2 - k, 3, 7).normalize();
        let p0 = v3r(k, 1, -2, 8);
        let p1 = v3r(-k, 3, 1, 8);
        let mut a = world();
        let mut b = world();
        add_touch(&mut a, false, n, p0, p1);
        add_touch(&mut b, true, n, p0, p1);
        let ma = a.contact_cache.find(&BodyPairKey::new(0, 1)).unwrap();
        let mb = b.contact_cache.find(&BodyPairKey::new(0, 1)).unwrap();
        assert_eq!(ma.points, mb.points, "k = {k}");
        assert_eq!(ma.normal, mb.normal, "k = {k}");
        assert_eq!(ma.points[0].local_point_a, p0, "point on pair.body_a");
        assert_eq!(
            ma.points[0].normal, n,
            "normal from pair.body_b to pair.body_a"
        );
        let l = (fr(5, 3), fr(-2, 7), fr(4, 9));
        assert_eq!(warm_start(&a, l), warm_start(&b, l), "k = {k}");
    }
}

#[test]
fn a_touch_reported_in_both_orders_updates_one_point_and_keeps_its_impulses() {
    // the same touch reported first as (0, 1) and next as (1, 0) (a detector
    // that does not sort its pairs) is one persistent point: the second
    // report matches it and keeps the stored impulses for warm starting
    let n = v3r(1, 4, -2, 5).normalize();
    let (p0, p1) = (v3r(1, 2, 0, 4), v3r(1, -2, 0, 4));
    let mut w = world();
    add_touch(&mut w, false, n, p0, p1);
    let l = (fr(7, 8), fr(1, 16), fr(-3, 32));
    w.contact_cache
        .get_or_create(BodyPairKey::new(0, 1), Fix128::ZERO, Fix128::ZERO)
        .store_impulses(0, l.0, l.1, l.2);
    add_touch(&mut w, true, n, p0, p1);
    let m = w.contact_cache.find(&BodyPairKey::new(0, 1)).unwrap();
    assert_eq!(m.point_count(), 1, "matched, not added as a second point");
    assert_eq!(m.warm_start_impulse(0), l);
    assert_eq!(m.points[0].age, 1);
    assert_eq!(m.points[0].normal, n);
    assert_eq!(m.points[0].local_point_a, p0);
}
