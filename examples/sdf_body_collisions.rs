//! Bodies with a shape against an SDF
//!
//! A floor given as a signed distance field. A plain body meets it as a sphere of
//! the world's collision radius; a body with a shape meets it as that shape, so a
//! box turned on its corner sinks by its lowest corner and not by a bounding ball.
//! `sdf_contacts` lists what `step` would push each body out of; `with_scale`
//! scales the field; `detect_sdf_contacts` is the same query for plain bodies on
//! their own.
//!
//! ```bash
//! cargo run --example sdf_body_collisions --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{detect_sdf_contacts, ClosureSdf, SdfCollider};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// The half-space `y < offset`, scaled by `scale` (so its surface is at
/// `y = scale · offset`).
fn floor(offset: f32, scale: f64) -> SdfCollider {
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            move |_, y, _| y - offset,
            |_, _, _| (0.0, 1.0, 0.0),
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
    .with_scale(Fix128::from_f64(scale))
}

fn main() {
    let mut world = PhysicsWorld::new(SolverConfig::default());
    world.add_sdf_collider(floor(0.0, 1.0));
    world.set_sdf_collision_radius(Fix128::from_f64(0.5));

    // A plain body 0.3 above the floor: a sphere of radius 0.5 reaches 0.2 in.
    world.add_body(RigidBody::new(v3(-4.0, 0.3, 0.0), Fix128::ONE));

    // A box (half-extents 1, 0.5, 0.75) turned 0.9 rad about z: its half-height is
    // sin 0.9 · 1 + cos 0.9 · 0.5 = 1.0941, so its lowest corner is 0.2 below the
    // floor when its centre is at 0.8941.
    let density = Fix128::from_int(1000);
    let slab = world
        .add_shaped_body(
            &Shape::Box {
                half_extents: v3(1.0, 0.5, 0.75),
            },
            density,
            v3(0.0, 0.8941, 0.0),
        )
        .expect("a valid solid");
    world
        .get_body_mut(slab)
        .expect("the body just added")
        .rotation = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::from_f64(0.9));

    // An upright cone: its base is half a half-height below its centre of mass.
    world
        .add_shaped_body(
            &Shape::Cone {
                radius: Fix128::from_f64(0.8),
                half_height: Fix128::from_f64(1.2),
            },
            density,
            v3(4.0, 0.5, 0.0),
        )
        .expect("a valid solid");

    for (body, contact) in world.sdf_contacts() {
        println!(
            "body {body}: depth {:.4} along ({:.1}, {:.1}, {:.1})",
            contact.depth.to_f64(),
            contact.normal.x.to_f64(),
            contact.normal.y.to_f64(),
            contact.normal.z.to_f64()
        );
    }
    println!("expected: body 0 0.2000 (sphere), body 1 0.2000 (corner), body 2 0.1000 (base)");

    // The same plain body, asked of the free function: it knows only spheres.
    let plain = detect_sdf_contacts(&world.bodies, &world.sdf_colliders, Fix128::from_f64(0.5));
    println!(
        "detect_sdf_contacts, every body as a sphere of radius 0.5: {:?}",
        plain
            .iter()
            .map(|(i, c)| (*i, (c.depth.to_f64() * 1e4).round() / 1e4))
            .collect::<Vec<_>>()
    );

    // One step pushes each body out along the normal.
    world.step(Fix128::from_ratio(1, 60));
    println!(
        "after a step: {} contacts left",
        world
            .sdf_contacts()
            .iter()
            .filter(|(_, c)| c.depth.to_f64() > 1e-3)
            .count()
    );

    // The field scaled by 2.5 has its surface at y = 2.5.
    let mut scaled = PhysicsWorld::new(SolverConfig::default());
    scaled.add_sdf_collider(floor(1.0, 2.5));
    scaled
        .add_shaped_body(
            &Shape::Ellipsoid {
                radii: v3(0.5, 0.5, 0.5),
            },
            density,
            v3(0.0, 2.3, 0.0),
        )
        .expect("a valid solid");
    println!(
        "scaled floor (surface y = 2.5), sphere of radius 0.5 at y = 2.3: depth {:.4} (0.7000)",
        scaled.sdf_contacts()[0].1.depth.to_f64()
    );
}
