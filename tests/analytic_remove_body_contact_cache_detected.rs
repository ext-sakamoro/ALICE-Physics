//! `PhysicsWorld::remove_body` against a control world built from
//! detection: the control holds the surviving bodies in the order the
//! removal leaves them (`swap_remove`: the last body moves into `idx`) and
//! is stepped once, so its contact cache is what the built-in detection
//! (which reports pairs as `body_a < body_b`) stores for that order. The
//! original world is stepped once with all bodies, then `remove_body(idx)`.
//! The cache contents must agree bit for bit; `serialize_state` (which
//! does not hold the cache) must agree bit for bit where the pair keeps its
//! order. Where the re-sort exchanges A and B, the step itself solved the
//! touching pair with the roles exchanged in the two worlds, which leaves
//! those two bodies a few hundred ulps apart regardless of the cache; there
//! the other bodies are compared bit for bit and the touching ones to 2^-44.
//!
//! Partner positions swept (moved body = old last, touching `partner`):
//! - `partner < idx`: the re-keyed pair keeps its order;
//! - `idx < partner < last`: the re-keyed pair is re-sorted, so A and B
//!   exchange and the stored data must turn around (normal, points,
//!   tangent impulses);
//! - `partner == idx` (the moved body touched the removed one): the pair is
//!   dropped; `serialize_state` is not compared here, the control never had
//!   the touch;
//! - `idx == last`: nothing moves, the removed body's pairs are dropped
//!   (cache only, as for `partner == idx`).
//!   (`partner > last` does not exist: `last` is the largest index.)
//!
//! Impulses: detection stores none (the CPU step does not write them
//! back), so a physical impulse is stored by hand on the moved body's
//! manifold before the removal and its expression in the control's frame
//! is computed by hand from the axis-aligned normal (see
//! `stored_impulses_follow_the_turned_frame`).

#![cfg(feature = "std")]

use alice_physics::contact_cache::{BodyPairKey, ContactManifold};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

/// Equal to within 2^-56 per component.
fn near(a: Vec3Fix, b: Vec3Fix) -> bool {
    let tol = Fix128::from_raw(0, 1 << 8);
    let d = a - b;
    d.x.abs() <= tol && d.y.abs() <= tol && d.z.abs() <= tol
}

/// Equal to within 2^-44 per component (the contact solve's dependence on
/// the index order, measured at a few hundred ulps of 2^-64).
fn near_loose(a: Vec3Fix, b: Vec3Fix) -> bool {
    let tol = Fix128::from_raw(0, 1 << 20);
    let d = a - b;
    d.x.abs() <= tol && d.y.abs() <= tol && d.z.abs() <= tol
}

fn fr(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn x_at(x: i64) -> Vec3Fix {
    Vec3Fix::new(fr(x, 2), Fix128::ZERO, Fix128::ZERO)
}

/// Five unit spheres on the x axis (positions in half units). Body 1 sits
/// far away; the last body (4) overlaps `partner` by half a unit, sitting
/// on its +x side.
fn positions(partner: usize) -> [Vec3Fix; 5] {
    let base = [0i64, 200, 20, 40, 0];
    let mut p = base.map(x_at);
    p[4] = x_at(base[partner] + 3);
    p
}

fn world_of(order: &[usize], pos: &[Vec3Fix; 5]) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    for &k in order {
        w.add_body_with_radius(
            RigidBody::new_dynamic(pos[k], fr(k as i64 + 2, 2)),
            Fix128::ONE,
        );
    }
    w
}

fn dump(m: &ContactManifold) -> String {
    format!(
        "{:?} {:?} {:?} {:?} {:?} {}",
        m.pair, m.points, m.normal, m.friction, m.restitution, m.stale_frames
    )
}

fn cache(w: &PhysicsWorld) -> Vec<String> {
    w.contact_cache.manifolds.iter().map(dump).collect()
}

/// Order of the surviving bodies after `remove_body(idx)` on 5 bodies.
fn order_after(idx: usize) -> Vec<usize> {
    let mut o: Vec<usize> = (0..5).collect();
    o.swap_remove(idx);
    o
}

fn removed_then_control(idx: usize, partner: usize) -> (PhysicsWorld, PhysicsWorld) {
    let pos = positions(partner);
    let dt = fr(1, 60);
    let mut w = world_of(&[0, 1, 2, 3, 4], &pos);
    w.step(dt);
    assert_eq!(
        w.contact_cache.manifold_count(),
        1,
        "the overlap is detected"
    );
    w.remove_body(idx).expect("idx exists");
    let mut c = world_of(&order_after(idx), &pos);
    c.step(dt);
    (w, c)
}

#[test]
fn partner_below_idx_keeps_the_order() {
    let (w, c) = removed_then_control(1, 0);
    assert_eq!(cache(&w), cache(&c));
    assert_eq!(w.serialize_state(), c.serialize_state());
}

#[test]
fn partner_between_idx_and_last_turns_the_manifold() {
    for partner in [2, 3] {
        let (w, c) = removed_then_control(1, partner);
        assert_eq!(
            c.contact_cache.manifolds[0].pair,
            BodyPairKey::new(1, partner)
        );
        assert_eq!(cache(&w), cache(&c), "partner {partner}");
        // `serialize_state` does not hold the cache. The step solved the
        // touching pair with A and B exchanged in the two worlds (the
        // contact solve depends on the index order), so the two touching
        // bodies differ by a few hundred ulps; every other body is
        // bit-identical
        for i in 0..4 {
            let (a, b) = (&w.bodies[i], &c.bodies[i]);
            if i == 1 || i == partner {
                assert!(near_loose(a.velocity, b.velocity), "body {i}");
                assert!(near_loose(a.position, b.position), "body {i}");
            } else {
                assert_eq!(
                    (a.position, a.velocity, a.rotation, a.angular_velocity),
                    (b.position, b.velocity, b.rotation, b.angular_velocity),
                    "body {i}"
                );
            }
        }
    }
}

#[test]
fn partner_equal_to_idx_drops_the_pair() {
    let (w, c) = removed_then_control(2, 2);
    assert_eq!(w.contact_cache.manifold_count(), 0);
    assert_eq!(c.contact_cache.manifold_count(), 0);
}

#[test]
fn removing_the_last_body_drops_its_pairs() {
    // the removed body touched body 2 during the step, so only the cache is
    // compared (the control never had that touch)
    let (w, c) = removed_then_control(4, 2);
    assert_eq!(w.contact_cache.manifold_count(), 0);
    assert_eq!(cache(&w), cache(&c));
}

#[test]
fn stored_impulses_follow_the_turned_frame() {
    // partner 2 lies between idx = 1 and last = 4. Before the removal the
    // pair is (2, 4): A = body 2, B = body 4 at +x of it, so the normal
    // (B to A) is -x. Its tangent frame (reference y, see
    // `tangent_frame`): t1 = (-x) × y = -z, t2 = (-x) × (-z) = -y.
    // After the removal the pair is (1, 2) with A = the moved body: normal
    // +x, frame (reference y): t1 = x × y = z, t2 = x × z = -y.
    // So t1' = -t1, t2' = t2, n' = -n. The impulse on the new A must be
    // minus the impulse on the old A: -(n λn + t1 λt1 + t2 λt2)
    //   = n' λn + t1' λt1 + t2' (-λt2), i.e. (λn, λt1, λt2) -> (λn, λt1, -λt2).
    let l = (fr(3, 4), fr(1, 2), fr(-1, 4));
    let pos = positions(2);
    let dt = fr(1, 60);
    let mut w = world_of(&[0, 1, 2, 3, 4], &pos);
    w.step(dt);
    {
        let m = &mut w.contact_cache.manifolds[0];
        assert_eq!(m.pair, BodyPairKey::new(2, 4));
        assert!(
            near(m.normal, -Vec3Fix::UNIT_X),
            "hand: B (4) is at +x of A (2)"
        );
        m.store_impulses(0, l.0, l.1, l.2);
    }
    w.remove_body(1).expect("idx exists");
    let mut c = world_of(&order_after(1), &pos);
    c.step(dt);
    {
        let m = &mut c.contact_cache.manifolds[0];
        assert_eq!(m.pair, BodyPairKey::new(1, 2));
        assert!(
            near(m.normal, Vec3Fix::UNIT_X),
            "hand: B (2) is at -x of A (1)"
        );
        m.store_impulses(0, l.0, l.1, -l.2);
    }
    assert_eq!(cache(&w), cache(&c));
    let mut bw = w.bodies.clone();
    let mut bc = c.bodies.clone();
    w.contact_cache.warm_start_factor = Fix128::ONE;
    c.contact_cache.warm_start_factor = Fix128::ONE;
    w.contact_cache.apply_warm_start(&mut bw);
    c.contact_cache.apply_warm_start(&mut bc);
    // hand: T on body 2 before = n λn + t1 λt1 + t2 λt2
    //   = (-x)(3/4) + (-z)(1/2) + (-y)(-1/4) = (-3/4, 1/4, -1/2);
    // body 2 (inverse mass 1/2) gains T/2 = (-3/8, 1/8, -1/4) in both worlds
    // (the detected normal is -x to within a few ulps, so this is checked
    // to 2^-56; the two worlds are compared bit for bit below)
    let dv = bw[2].velocity - w.bodies[2].velocity;
    assert!(
        near(dv, Vec3Fix::new(fr(-3, 8), fr(1, 8), fr(-1, 4))),
        "{dv:?}"
    );
    // the step left the touching bodies a few ulps apart (see above), so
    // the warm-start increments are compared: they depend only on the
    // cache and the masses, and Fix128 addition is exact
    for i in 0..4 {
        let dw = bw[i].velocity - w.bodies[i].velocity;
        let dc = bc[i].velocity - c.bodies[i].velocity;
        assert_eq!(dw, dc, "body {i}");
    }
}
