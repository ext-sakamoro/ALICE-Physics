//! World shape queries: `PhysicsWorld::cast_sphere`, `cast_capsule`,
//! `overlap_sphere` and `overlap_aabb` against the geometry the world collides
//! with (a box body, a ground plane), next to the bounding-sphere answer of
//! `alice_physics::query::overlap_sphere`.
//!
//! Scene: a unit box (half extents 1) at the origin and the plane `y = 0` under
//! it, shifted so the box rests on it (centre at `y = 1`).
//!
//! Expected values (closed forms, see `tests/analytic_world_shape_query.rs`):
//!
//! * a sphere of radius 0.5 from `(−10, 1.2, 0)` along `+X` meets the box face
//!   `x = −1` at `t = 10 − 1.5 = 8.5`;
//! * a capsule from `(−0.5, 5, 0)` to `(0.5, 5, 0)`, radius 0.5, moving down meets
//!   the box top `y = 2` at `t = 5 − 2 − 0.5 = 2.5`;
//! * the sphere `(1.4, 2.4, 0)`, radius 0.5, is `0.4·√2 = 0.566` from the box edge
//!   `(1, 2)`: no overlap, though the box's bounding sphere reaches it;
//! * the box `[−3, −0.2, −3]..[−2, 0.1, −2]` crosses the plane only.
//!
//! Run: `cargo run --example world_shape_query`.
//!
//! Author: Moroya Sakamoto

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics::static_collider::StaticCollider;

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn main() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let shape = Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    };
    let body = world
        .add_shaped_body(&shape, Fix128::ONE, v3(0.0, 1.0, 0.0))
        .expect("valid box");
    world.bodies[body].rotation = QuatFix::IDENTITY;
    let ground = world.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let filter = RayFilter::default();

    let hit = world
        .cast_sphere(
            v3(-10.0, 1.2, 0.0),
            Fix128::from_f64(0.5),
            v3(1.0, 0.0, 0.0),
            Fix128::from_int(100),
            &filter,
        )
        .expect("the sphere meets the box");
    println!(
        "cast_sphere:  {:?} at t = {:.6}, normal {:?}",
        hit.target,
        hit.t.to_f64(),
        hit.normal.to_f32()
    );
    assert_eq!(hit.target, RayTarget::Body(body));
    assert!((hit.t.to_f64() - 8.5).abs() < 1e-12);

    let hit = world
        .cast_capsule(
            v3(-0.5, 5.0, 0.0),
            v3(0.5, 5.0, 0.0),
            Fix128::from_f64(0.5),
            v3(0.0, -1.0, 0.0),
            Fix128::from_int(100),
            &filter,
        )
        .expect("the capsule meets the box");
    println!(
        "cast_capsule: {:?} at t = {:.6}, point {:?}",
        hit.target,
        hit.t.to_f64(),
        hit.point.to_f32()
    );
    assert_eq!(hit.target, RayTarget::Body(body));
    assert!((hit.t.to_f64() - 2.5).abs() < 1e-9);

    let center = v3(1.4, 2.4, 0.0);
    let radius = Fix128::from_f64(0.5);
    let found = world.overlap_sphere(center, radius, &filter);
    let old = alice_physics::query::overlap_sphere(
        center,
        radius,
        &world.bodies,
        shape.bounding_radius(),
    );
    println!(
        "overlap_sphere: {found:?} (bounding spheres: {} bodies)",
        old.len()
    );
    assert!(found.is_empty());
    assert_eq!(old.len(), 1);

    let found = world.overlap_aabb(
        &AABB::new(v3(-3.0, -0.2, -3.0), v3(-2.0, 0.1, -2.0)),
        &filter,
    );
    println!("overlap_aabb:   {found:?}");
    assert_eq!(found, vec![RayTarget::StaticCollider(ground)]);
}
