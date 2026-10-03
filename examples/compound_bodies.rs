//! A body made of several shapes
//!
//! Two boxes joined by nothing but their positions, a dumbbell with a gap between
//! them. `add_compound_body` gives it the mass and inertia of the two boxes
//! together (their centre of mass is *between* the boxes, not at either of them),
//! stores it in its principal frame, and makes it collide as its children: a probe
//! in the gap touches nothing, although the gap lies inside the convex hull of the
//! pair, while a probe on either box collides.
//!
//! ```bash
//! cargo run --example compound_bodies --features std
//! ```

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere};
use alice_physics::compound::CompoundShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn main() {
    let unit_box = OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), QuatFix::IDENTITY);
    let mut dumbbell = CompoundShape::new();
    dumbbell.add_box(unit_box, v3(-3.0, 0.0, 0.0), QuatFix::IDENTITY);
    dumbbell.add_box(unit_box, v3(5.0, 0.0, 0.0), QuatFix::IDENTITY);

    let density = Fix128::from_int(1000);
    let props = dumbbell.mass_properties(density);
    println!(
        "authored: mass {:.1} kg, centre of mass x = {:.3} (boxes at -3 and +5)",
        props.mass.to_f64(),
        props.center_of_mass.x.to_f64()
    );

    let mut world = PhysicsWorld::new(SolverConfig::default());
    let body = world
        .add_compound_body(&dumbbell, density, Vec3Fix::ZERO)
        .expect("two boxes have volume");
    let b = *world.get_body(body).expect("the body just added");
    println!(
        "body: mass {:.1} kg, principal-frame inertia diag ({:.1}, {:.1}, {:.1})",
        b.mass().to_f64(),
        1.0 / b.inv_inertia.x.to_f64(),
        1.0 / b.inv_inertia.y.to_f64(),
        1.0 / b.inv_inertia.z.to_f64()
    );

    // The boxes now sit at x = -4 and x = +4 about the centre of mass.
    let probe = Shape::Ellipsoid {
        radii: v3(0.5, 0.5, 0.5),
    };
    for (what, x) in [
        ("in the gap", 0.0),
        ("on the left box", -4.0),
        ("on the right box", 4.0),
    ] {
        let p = world
            .add_shaped_body(&probe, density, v3(x, 0.0, 0.0))
            .expect("a valid solid");
        println!(
            "probe {what:<17} overlaps: {}",
            world.colliders_overlap(body, p)
        );
    }

    // Spheres, capsules and convex hulls join a compound the same way.
    let mut mixed = CompoundShape::new();
    mixed.add_sphere(
        Sphere::new(Vec3Fix::ZERO, Fix128::from_f64(0.8)),
        v3(0.5, 2.0, 0.5),
        QuatFix::IDENTITY,
    );
    mixed.add_capsule(
        Capsule::new(v3(0.0, -1.0, 0.0), v3(0.0, 1.0, 0.0), Fix128::from_f64(0.5)),
        v3(2.5, -1.5, -0.5),
        QuatFix::IDENTITY,
    );
    mixed.add_convex_hull(
        ConvexHull::new(vec![
            v3(0.0, 0.0, 0.0),
            v3(1.0, 0.0, 0.0),
            v3(0.0, 1.0, 0.0),
            v3(0.0, 0.0, 1.0),
        ]),
        v3(-2.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let mixed_props = mixed.mass_properties(density);
    println!(
        "sphere + capsule + hull: mass {:.1} kg, centre of mass ({:.3}, {:.3}, {:.3})",
        mixed_props.mass.to_f64(),
        mixed_props.center_of_mass.x.to_f64(),
        mixed_props.center_of_mass.y.to_f64(),
        mixed_props.center_of_mass.z.to_f64()
    );
}
