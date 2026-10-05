//! Oracles for the direction of contact normals: every producer reports the
//! normal pointing from B to A (the crate-wide `Contact::normal` contract),
//! i.e. translating A by `depth · normal` separates the pair.
//!
//! * `ContactEvent::normal` points from `body_b` toward `body_a` of the event
//!   itself. `EventCollector::report_contact` orders the pair as
//!   `(min, max)`; when it swaps the bodies it must also negate the normal.
//!   `relative_velocity = (v_a − v_b) · normal` is unchanged by that swap
//!   (both factors change sign), so it is negative while the bodies approach.
//! * `CollisionResult::normal` (plane producers) points from the plane (B)
//!   toward the sphere / box (A).
//!
//! Expected values are analytic: two spheres on the x axis, a plane `y = 0`.

use alice_physics::collider::{self, Sphere, AABB};
use alice_physics::event::{ContactEventType, EventCollector};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;

const TOL: f64 = 1e-12;

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn half(n: i64) -> Fix128 {
    Fix128::from_ratio(n, 2)
}

fn assert_dir(got: Vec3Fix, want: [f64; 3], what: &str) {
    let g = [got.x.to_f64(), got.y.to_f64(), got.z.to_f64()];
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() <= TOL,
            "{what}: normal {g:?}, want {want:?}"
        );
    }
}

/// Two unit spheres overlapping on the x axis (centres 3/2 apart), moving
/// toward each other at 1 m/s. `lower_index_on_left` picks which body gets
/// index 0. Returns the Begin event and the analytic B→A direction for the
/// event's (body_a, body_b).
fn world_event(lower_index_on_left: bool) -> (alice_physics::event::ContactEvent, [f64; 3]) {
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        ..Default::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let left = (Vec3Fix::ZERO, v(1, 0, 0));
    let right = (
        Vec3Fix::new(half(3), Fix128::ZERO, Fix128::ZERO),
        v(-1, 0, 0),
    );
    let order = if lower_index_on_left {
        [left, right]
    } else {
        [right, left]
    };
    let mut xs = [0.0; 2];
    for (i, (p, vel)) in order.into_iter().enumerate() {
        let mut body = RigidBody::new_dynamic(p, Fix128::ONE);
        body.set_velocity(vel);
        w.add_body_with_radius(body, Fix128::ONE);
        xs[i] = p.x.to_f64();
    }
    w.step(Fix128::from_ratio(1, 60));
    let begins: Vec<_> = w
        .contact_events()
        .iter()
        .copied()
        .filter(|e| e.event_type == ContactEventType::Begin)
        .collect();
    assert_eq!(begins.len(), 1, "one Begin event: {:?}", w.contact_events());
    let e = begins[0];
    // B→A of the event's own pair: sign of x_a − x_b along the x axis.
    let s = (xs[e.body_a] - xs[e.body_b]).signum();
    (e, [s, 0.0, 0.0])
}

#[test]
fn world_step_event_normal_points_from_body_b_to_body_a_lower_index_left() {
    let (e, want) = world_event(true);
    assert_eq!((e.body_a, e.body_b), (0, 1));
    assert_eq!(want, [-1.0, 0.0, 0.0]);
    assert_dir(e.normal, want, "index 0 on the left");
    assert!(
        e.relative_velocity < Fix128::ZERO,
        "approaching: {}",
        e.relative_velocity.to_f64()
    );
}

#[test]
fn world_step_event_normal_points_from_body_b_to_body_a_lower_index_right() {
    let (e, want) = world_event(false);
    assert_eq!((e.body_a, e.body_b), (0, 1));
    assert_eq!(want, [1.0, 0.0, 0.0]);
    assert_dir(e.normal, want, "index 0 on the right");
    assert!(
        e.relative_velocity < Fix128::ZERO,
        "approaching: {}",
        e.relative_velocity.to_f64()
    );
}

/// Body 3 at the origin, body 1 at x = 2: reported as (3, 1) with the B→A
/// normal of that order (from body 1 toward body 3 = −x). The event is
/// `(1, 3)`, whose B→A normal is from body 3 toward body 1 = +x.
#[test]
fn report_contact_flips_the_normal_when_it_swaps_the_pair() {
    let mut ev = EventCollector::new();
    ev.begin_frame();
    let rel = Fix128::from_int(-2);
    ev.report_contact(3, 1, v(-1, 0, 0), Vec3Fix::ZERO, half(1), rel);
    ev.end_frame();
    let e = ev.contact_events()[0];
    assert_eq!((e.body_a, e.body_b), (1, 3));
    assert_eq!(e.event_type, ContactEventType::Begin);
    assert_eq!(e.normal, v(1, 0, 0), "swapped pair keeps B→A");
    // (v_b − v_a) · (−n) = (v_a − v_b) · n: unchanged by the swap.
    assert_eq!(e.relative_velocity, rel);
    assert_eq!(e.depth, half(1));

    // Persist frame, reported in the swapped order again.
    ev.begin_frame();
    ev.report_contact(3, 1, v(-1, 0, 0), Vec3Fix::ZERO, half(1), rel);
    ev.end_frame();
    let e = ev.contact_events()[0];
    assert_eq!(e.event_type, ContactEventType::Persist);
    assert_eq!(e.normal, v(1, 0, 0));
}

#[test]
fn report_contact_keeps_the_normal_when_the_pair_is_already_ordered() {
    let mut ev = EventCollector::new();
    ev.begin_frame();
    ev.report_contact(1, 3, v(1, 0, 0), Vec3Fix::ZERO, half(1), Fix128::ZERO);
    ev.end_frame();
    let e = ev.contact_events()[0];
    assert_eq!((e.body_a, e.body_b), (1, 3));
    assert_eq!(e.normal, v(1, 0, 0));
}

/// Both orders of the same physical contact give the same event.
#[test]
fn report_contact_is_symmetric_in_argument_order() {
    let n = Vec3Fix::new(half(1), Fix128::from_ratio(-3, 4), Fix128::ZERO);
    let mut fwd = EventCollector::new();
    fwd.begin_frame();
    fwd.report_contact(2, 7, n, Vec3Fix::ZERO, half(1), Fix128::ONE);
    let mut rev = EventCollector::new();
    rev.begin_frame();
    rev.report_contact(7, 2, -n, Vec3Fix::ZERO, half(1), Fix128::ONE);
    assert_eq!(fwd.contact_events(), rev.contact_events());
}

/// `CollisionResult` from `PlaneCollider::intersect_sphere`: A = sphere,
/// B = plane `y = 0` with normal +y.
#[test]
fn plane_sphere_collision_result_normal_points_from_plane_to_sphere() {
    let plane = PlaneCollider::new(v(0, 1, 0), Fix128::ZERO);
    let front = plane.intersect_sphere(
        Vec3Fix::new(Fix128::ZERO, half(1), Fix128::ZERO),
        Fix128::ONE,
    );
    assert!(front.colliding);
    assert_dir(front.normal, [0.0, 1.0, 0.0], "sphere in front");
    // Translating A by depth · normal leaves the centre at distance r.
    assert_eq!(front.depth, half(1));

    let back = plane.intersect_sphere(
        Vec3Fix::new(Fix128::ZERO, -half(1), Fix128::ZERO),
        Fix128::ONE,
    );
    assert!(back.colliding);
    assert_dir(back.normal, [0.0, -1.0, 0.0], "sphere behind");
    assert_eq!(back.depth, half(1));

    // Same convention through `StaticCollider::collide_sphere` → `Contact`.
    let c = StaticCollider::Plane(plane)
        .collide_sphere(
            Vec3Fix::new(Fix128::ZERO, half(1), Fix128::ZERO),
            Fix128::ONE,
        )
        .expect("overlap");
    assert_dir(c.normal, [0.0, 1.0, 0.0], "static plane contact");
}

/// `CollisionResult` from `PlaneCollider::intersect_aabb`: A = box, B = plane.
#[test]
fn plane_aabb_collision_result_normal_points_from_plane_to_box() {
    let plane = PlaneCollider::new(v(0, 1, 0), Fix128::ZERO);
    // Straddling the plane.
    let partial = plane.intersect_aabb(&AABB::new(v(-1, -1, -1), v(1, 1, 1)));
    assert!(partial.colliding);
    assert_dir(partial.normal, [0.0, 1.0, 0.0], "box straddling");
    assert_eq!(partial.depth, Fix128::ONE);
    // Entirely behind: y ∈ [−3, −2], moving it 3 along +y clears the plane.
    let behind = plane.intersect_aabb(&AABB::new(v(-1, -3, -1), v(1, -2, 1)));
    assert!(behind.colliding);
    assert_dir(behind.normal, [0.0, 1.0, 0.0], "box behind");
    assert_eq!(behind.depth, Fix128::from_int(3));
}

/// `collider::contact` (GJK + EPA) on two unit spheres: A at the origin, B at
/// x = 3/2. B→A is −x, depth 1/2.
#[test]
fn gjk_epa_contact_normal_points_from_b_to_a() {
    let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let b = Sphere::new(
        Vec3Fix::new(half(3), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    );
    let c = collider::contact(&a, &b).expect("overlap");
    let n = [
        c.normal.x.to_f64(),
        c.normal.y.to_f64(),
        c.normal.z.to_f64(),
    ];
    assert!(
        n[0] < -0.99 && n[1].abs() < 0.1 && n[2].abs() < 0.1,
        "normal {n:?}"
    );
    let ba = collider::contact(&b, &a).expect("overlap");
    assert!(ba.normal.x.to_f64() > 0.99, "swapped: {:?}", ba.normal);
}
