//! Convex shapes colliding as shapes, not as spheres
//!
//! Two unit boxes whose bounding spheres overlap but whose faces do not are not in
//! contact; move one in and the contact is the separating-axis answer: the depth is
//! the overlap along the shortest axis and the normal points from B to A. The
//! example asks the same of a cone and a wedge, whose position is their *centre of
//! mass* (not their geometric centre), and lets a box drop onto a static one.
//!
//! ```bash
//! cargo run --example convex_contacts --features std
//! ```

use alice_physics::collider::contact;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::{PosedShape, Shape};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn main() {
    let unit_box = Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    };
    let mut world = PhysicsWorld::new(SolverConfig::default());
    let b = world
        .add_shaped_body(&unit_box, Fix128::ONE, v3(0.0, 0.0, 0.0))
        .expect("a unit box");
    let a = world
        .add_shaped_body(&unit_box, Fix128::ONE, v3(2.2, 2.2, 0.0))
        .expect("a unit box");
    println!(
        "boxes at (2.2, 2.2): bounding spheres overlap, colliders overlap: {}",
        world.colliders_overlap(a, b)
    );
    world.get_body_mut(a).expect("a").position = v3(0.5, 1.7, 0.0);
    println!(
        "boxes at (0.5, 1.7): colliders overlap: {}",
        world.colliders_overlap(a, b)
    );
    let posed = |p: Vec3Fix| PosedShape {
        shape: unit_box,
        position: p,
        rotation: alice_physics::math::QuatFix::IDENTITY,
    };
    if let Some(hit) = contact(&posed(v3(0.5, 1.7, 0.0)), &posed(Vec3Fix::ZERO)) {
        println!(
            "  contact depth {:.4}, normal ({:.1}, {:.1}, {:.1}) (from B to A)",
            hit.depth.to_f64(),
            hit.normal.x.to_f64(),
            hit.normal.y.to_f64(),
            hit.normal.z.to_f64()
        );
    }

    // A cone and a wedge dropped on a static box floor: their positions are their
    // centres of mass, so the flat base ends up that far below it.
    let shapes = [
        (
            "cone",
            Shape::Cone {
                radius: Fix128::ONE,
                half_height: Fix128::from_int(2),
            },
            1.0,
        ),
        (
            "wedge",
            Shape::Wedge {
                width: Fix128::from_int(2),
                height: Fix128::from_int(3),
                depth: Fix128::from_int(2),
            },
            1.0,
        ),
    ];
    for (name, shape, base_below_com) in shapes {
        let mut w = PhysicsWorld::new(SolverConfig::default());
        let floor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        assert!(w.set_body_shape(floor, &unit_box));
        let body = w
            .add_shaped_body(&shape, Fix128::ONE, v3(0.0, 4.0, 0.0))
            .expect("a valid solid");
        for _ in 0..400 {
            w.step(Fix128::from_ratio(1, 60));
        }
        let y = w.get_body(body).expect("body").position.y.to_f64();
        println!(
            "{name:<5} rests with its centre of mass at y = {y:.4} \
             (floor top 1 + base {base_below_com} below the centre of mass)"
        );
    }
}
