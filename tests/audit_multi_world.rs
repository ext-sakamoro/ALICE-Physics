//! Audit oracles for `multi_world`.
//!
//! Closed forms:
//!
//! ```text
//! q = (1/2, 1/2, 1/2, 1/2) is the 120 deg rotation about (1,1,1): (x,y,z) -> (z,x,y)
//!     (every component is dyadic, so rotate_vec is exact in Fix128)
//! Portal:  A->B  p' = R p + t        B->A  p = R^T (p' - t)
//! transfer_body: conserves the total body count; the moved body keeps
//!     everything the world stores about it (material, filter, collision radius)
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::filter::CollisionFilter;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::multi_world::{MultiWorld, Portal};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn half() -> Fix128 {
    Fix128::from_ratio(1, 2)
}
fn q120() -> QuatFix {
    QuatFix::new(half(), half(), half(), half())
}
fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

/// 120 deg about (1,1,1) is the cyclic permutation (x,y,z) -> (z,x,y).
#[test]
fn portal_cyclic_rotation_closed_form() {
    let p = Portal::new(0, 1, v(10, 20, 30), q120());
    assert_eq!(p.transform_a_to_b(v(1, 2, 3)), v(3 + 10, 1 + 20, 2 + 30));
    // B->A is the inverse: R^T (p' - t)
    assert_eq!(p.transform_b_to_a(v(13, 21, 32)), v(1, 2, 3));
    // a point at the translation maps back to the origin
    assert_eq!(p.transform_b_to_a(v(10, 20, 30)), Vec3Fix::ZERO);
    // the origin of A lands exactly on the translation
    assert_eq!(p.transform_a_to_b(Vec3Fix::ZERO), v(10, 20, 30));
}

/// A->B->A and B->A->B are the identity for a unit rotation (exact here).
#[test]
fn portal_round_trips_both_ways() {
    let p = Portal::new(2, 5, v(-7, 4, 9), q120());
    for pt in [v(0, 0, 0), v(1, -2, 3), v(100, 50, -25)] {
        assert_eq!(p.transform_b_to_a(p.transform_a_to_b(pt)), pt);
        assert_eq!(p.transform_a_to_b(p.transform_b_to_a(pt)), pt);
    }
    assert_eq!((p.world_a, p.world_b), (2, 5));
}

/// The map is rigid: distances between points are preserved (|R(a-b)| = |a-b|).
#[test]
fn portal_preserves_distances() {
    let p = Portal::new(0, 1, v(3, 3, 3), q120());
    let (a, b) = (v(1, 2, 3), v(-4, 0, 7));
    let d0 = (a - b).length_squared();
    let d1 = (p.transform_a_to_b(a) - p.transform_a_to_b(b)).length_squared();
    assert_eq!(d0, d1);
}

/// AUD-A-S4W1-004 (known defect, precondition unchecked): `Portal::new` stores
/// the rotation as given and `transform_a_to_b` rotates with `q v q*`, which
/// scales by |q|^2 when q is not a unit quaternion. A portal given
/// q = (0,0,0,2) (a "2x identity", w = 2) maps v to 4 v + t, and the pair
/// A->B->A returns |q|^4 v = 16 v. Neither a normalization nor a rejection
/// is present, so a mis-scaled quaternion silently scales the world.
#[test]
#[ignore = "known defect: AUD-A-S4W1-004: Portal::new keeps a non-unit rotation; transform_a_to_b(1,2,3) with q=(0,0,0,2) returns 4*(1,2,3)+t, not (1,2,3)+t"]
fn portal_with_non_unit_rotation_is_not_a_scaling() {
    let q2 = QuatFix::new(
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::ZERO,
        Fix128::from_int(2),
    );
    let p = Portal::new(0, 1, v(10, 0, 0), q2);
    assert_eq!(p.transform_a_to_b(v(1, 2, 3)), v(11, 2, 3));
}

fn body(x: i64, y: i64, z: i64) -> RigidBody {
    RigidBody::new_dynamic(v(x, y, z), Fix128::ONE)
}

/// Transferring a body conserves the total count and changes only the
/// position/prev_position; mass, velocity, rotation stay as they were.
#[test]
fn transfer_keeps_body_state_except_position() {
    let mut mw = MultiWorld::new();
    mw.add_world(PhysicsConfig::default());
    mw.add_world(PhysicsConfig::default());
    let mut b = RigidBody::new_dynamic(v(1, 2, 3), Fix128::from_int(7));
    b.velocity = v(4, 5, 6);
    b.angular_velocity = v(1, 0, 2);
    b.rotation = q120();
    let snapshot = b;
    mw.worlds[0].add_body(b);
    mw.worlds[0].add_body(body(9, 9, 9));
    let before = mw.total_body_count();
    let id = mw.transfer_body(0, 0, 1, v(-5, 6, -7)).expect("valid");
    assert_eq!(mw.total_body_count(), before);
    assert_eq!(id, 0);
    let moved = &mw.worlds[1].bodies[id];
    assert_eq!(moved.position, v(-5, 6, -7));
    assert_eq!(moved.prev_position, v(-5, 6, -7));
    assert_eq!(moved.velocity, snapshot.velocity);
    assert_eq!(moved.angular_velocity, snapshot.angular_velocity);
    assert_eq!(moved.rotation, snapshot.rotation);
    assert_eq!(moved.inv_mass, snapshot.inv_mass);
    // the swap-removed survivor is the other body, untouched
    assert_eq!(mw.worlds[0].bodies.len(), 1);
    assert_eq!(mw.worlds[0].bodies[0].position, v(9, 9, 9));
}

/// A failed transfer (bad body id / bad world) changes nothing, including
/// the destination.
#[test]
fn failed_transfer_is_atomic() {
    let mut mw = MultiWorld::new();
    mw.add_world(PhysicsConfig::default());
    mw.add_world(PhysicsConfig::default());
    mw.worlds[0].add_body(body(1, 1, 1));
    assert!(mw.transfer_body(0, 1, 1, Vec3Fix::ZERO).is_none());
    assert!(mw.transfer_body(2, 0, 1, Vec3Fix::ZERO).is_none());
    assert!(mw.transfer_body(0, 0, 2, Vec3Fix::ZERO).is_none());
    assert!(mw.transfer_body(0, 0, 0, Vec3Fix::ZERO).is_none());
    assert_eq!(mw.worlds[0].bodies.len(), 1);
    assert_eq!(mw.worlds[1].bodies.len(), 0);
    assert_eq!(mw.worlds[0].bodies[0].position, v(1, 1, 1));
}

/// AUD-A-S4W1-005 (known defect): the doc says transfer "removes the body from
/// `from_world` and adds it to `to_world`". The world stores per-body material,
/// collision filter and collision radius in side tables that `remove_body`
/// pops and `transfer_body` drops; `add_body` re-creates them as defaults.
#[test]
#[ignore = "known defect: AUD-A-S4W1-005: transfer_body drops the body's material id (3 -> 0)"]
fn transfer_preserves_material() {
    let mut mw = MultiWorld::new();
    mw.add_world(PhysicsConfig::default());
    mw.add_world(PhysicsConfig::default());
    mw.worlds[0].add_body(body(0, 0, 0));
    mw.worlds[0].set_body_material(0, 3);
    let id = mw.transfer_body(0, 0, 1, Vec3Fix::ZERO).expect("valid");
    assert_eq!(mw.worlds[1].body_materials[id], 3);
}

/// AUD-A-S4W1-005 (collision filter half).
#[test]
#[ignore = "known defect: AUD-A-S4W1-005: transfer_body drops the body's collision filter (custom -> DEFAULT)"]
fn transfer_preserves_collision_filter() {
    let mut mw = MultiWorld::new();
    mw.add_world(PhysicsConfig::default());
    mw.add_world(PhysicsConfig::default());
    mw.worlds[0].add_body(body(0, 0, 0));
    let f = CollisionFilter {
        layer: 4,
        mask: 6,
        group: 9,
    };
    mw.worlds[0].set_body_filter(0, f);
    let id = mw.transfer_body(0, 0, 1, Vec3Fix::ZERO).expect("valid");
    assert_eq!(mw.worlds[1].body_filter(id), f);
}

fn settle(w: &mut PhysicsWorld) -> Vec<Vec3Fix> {
    for _ in 0..8 {
        w.step(Fix128::from_ratio(1, 60));
    }
    w.bodies.iter().map(|b| b.position).collect()
}

/// AUD-A-S4W1-005 (collision radius half), differential: a body with a collision
/// radius that is moved into a world must interact exactly like the same body
/// added there directly with that radius. Control world: A and B added with
/// radius 1 at overlapping positions. Test world: B is created with radius 1 in
/// another world and transferred.
#[test]
#[ignore = "known defect: AUD-A-S4W1-005: transfer_body drops the collision radius, so the transferred body no longer collides (positions differ from the directly-added control)"]
fn transfer_preserves_collision_radius() {
    let r = Fix128::ONE;
    let mut control = PhysicsWorld::new(PhysicsConfig::default());
    control.add_body_with_radius(body(0, 0, 0), r);
    control.add_body_with_radius(
        RigidBody::new_dynamic(
            Vec3Fix::new(half(), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ),
        r,
    );
    let want = settle(&mut control);

    let mut mw = MultiWorld::new();
    mw.add_world(PhysicsConfig::default());
    mw.add_world(PhysicsConfig::default());
    mw.worlds[1].add_body_with_radius(body(0, 0, 0), r);
    mw.worlds[0].add_body_with_radius(body(5, 0, 0), r);
    let id = mw
        .transfer_body(0, 0, 1, Vec3Fix::new(half(), Fix128::ZERO, Fix128::ZERO))
        .expect("valid");
    assert_eq!(id, 1);
    let got = settle(&mut mw.worlds[1]);
    assert_eq!(got, want);
}

/// step_all advances every world by one `step(dt)` exactly: a MultiWorld of three
/// worlds equals three standalone worlds stepped separately (differential).
#[test]
fn step_all_equals_independent_steps() {
    let dt = Fix128::from_ratio(1, 60);
    let mut mw = MultiWorld::new();
    let mut solo: Vec<PhysicsWorld> = Vec::new();
    for k in 0..3 {
        mw.add_world(PhysicsConfig::default());
        let mut s = PhysicsWorld::new(PhysicsConfig::default());
        for j in 0..=k {
            let mut b = body(j, 10 + k, 0);
            b.velocity = v(k, 0, j);
            mw.worlds[k as usize].add_body(b);
            s.add_body(b);
        }
        solo.push(s);
    }
    for _ in 0..5 {
        mw.step_all(dt);
        for s in &mut solo {
            s.step(dt);
        }
    }
    for (w, s) in mw.worlds.iter().zip(solo.iter()) {
        for (a, b) in w.bodies.iter().zip(s.bodies.iter()) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.velocity, b.velocity);
        }
    }
    // Free fall closed form for world 0 body: y decreases (g acts), x moves at vx = 0
    assert!(mw.worlds[0].bodies[0].position.y < Fix128::from_int(10));
}
