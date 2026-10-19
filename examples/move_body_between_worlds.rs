//! Moving a body from one `PhysicsWorld` into another with its per-body
//! settings (material, collision radius, collider, collision filter):
//! `remove_body_with_attachments` takes them out with the body and
//! `add_body_with_attachments` gives them to it in the destination, so it
//! collides there as it did in the source.
//!
//! Run: `cargo run --example move_body_between_worlds`

use alice_physics_core::filter::CollisionFilter;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape::Shape;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    let half = Fix128::from_ratio(1, 2);
    let cube = Shape::Box {
        half_extents: Vec3Fix::new(half, half, half),
    };

    // the source world: a rubber cube on layer 2
    let mut source = PhysicsWorld::new(PhysicsConfig::default());
    let rubber = source.material_table.register_rubber();
    let cube_idx = source
        .add_shaped_body(&cube, Fix128::ONE, Vec3Fix::from_int(0, 3, 0))
        .expect("a box with positive half-extents");
    source.set_body_material(cube_idx, rubber);
    source.set_body_filter(cube_idx, CollisionFilter::new(2, u32::MAX));

    // the destination registers the same materials in the same order, so the
    // carried material id names the same material there
    let mut destination = PhysicsWorld::new(PhysicsConfig::default());
    let rubber_there = destination.material_table.register_rubber();
    assert_eq!(rubber_there, rubber);
    let floor = destination.add_body(RigidBody::new_static(Vec3Fix::from_int(0, -1, 0)));
    let slab = Shape::Box {
        half_extents: Vec3Fix::new(Fix128::from_int(8), half, Fix128::from_int(8)),
    };
    assert!(destination.set_body_shape(floor, &slab));

    let (body, attachments) = source
        .remove_body_with_attachments(cube_idx)
        .expect("the index is in range");
    let moved = destination.add_body_with_attachments(body, attachments);
    assert_eq!(source.bodies.len(), 0);
    assert_eq!(
        destination.body_filter(moved),
        CollisionFilter::new(2, u32::MAX)
    );

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..240 {
        destination.step(dt);
    }
    let rest = destination.get_body(moved).expect("moved body").position;
    println!(
        "after 4 s the cube is at y = {:.3} (on the floor of the destination world: its top is y = -0.5, the cube's half-height 0.5)",
        rest.y.to_f64()
    );
}
