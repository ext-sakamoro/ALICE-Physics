//! Bodies with the inertia of their shape
//!
//! Six solids of the same density take the same torque about each of their
//! principal axes for the same time. A body built with `RigidBody::new` would spin
//! at one rate whatever it is; a body built from a `Shape` spins at `τ·dt / I`,
//! with `I` the closed-form moment of inertia of that solid about its centre of
//! mass. The printout lists the mass, the centre-of-mass offset (non-zero for the
//! cone and the wedge), the bounding radius used as the collision radius, and the
//! angular velocity about each axis.
//!
//! ```bash
//! cargo run --example shaped_bodies --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
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
    let shapes = [
        (
            "box",
            Shape::Box {
                half_extents: v3(0.5, 1.0, 1.5),
            },
        ),
        (
            "cylinder",
            Shape::Cylinder {
                radius: Fix128::from_f64(0.75),
                half_height: Fix128::from_f64(1.25),
            },
        ),
        (
            "cone",
            Shape::Cone {
                radius: Fix128::from_f64(1.5),
                half_height: Fix128::from_f64(1.0),
            },
        ),
        (
            "ellipsoid",
            Shape::Ellipsoid {
                radii: v3(1.0, 0.5, 1.5),
            },
        ),
        (
            "wedge",
            Shape::Wedge {
                width: Fix128::from_f64(2.0),
                height: Fix128::from_f64(1.5),
                depth: Fix128::from_f64(1.0),
            },
        ),
        (
            "torus",
            Shape::Torus {
                major_radius: Fix128::from_f64(1.5),
                minor_radius: Fix128::from_f64(0.5),
            },
        ),
    ];
    let density = Fix128::from_int(1000);
    let dt = Fix128::from_ratio(1, 60);
    let torque = 100.0;

    let mut world = PhysicsWorld::new(SolverConfig::default());
    println!(
        "{:<10} {:>10} {:>10} {:>8}   ω about x / y / z for τ = {torque} N·m over {:.4} s",
        "shape",
        "mass kg",
        "com y",
        "radius",
        dt.to_f64()
    );
    for (name, shape) in &shapes {
        let index = world
            .add_shaped_body(shape, density, Vec3Fix::ZERO)
            .expect("these are valid solids");
        let body = *world.get_body(index).expect("the body just added");
        let mut spin = [0.0; 3];
        for (axis, slot) in spin.iter_mut().enumerate() {
            let mut b = body;
            let mut t = [0.0; 3];
            t[axis] = torque;
            b.add_torque(v3(t[0], t[1], t[2]), dt);
            *slot = [
                b.angular_velocity.x,
                b.angular_velocity.y,
                b.angular_velocity.z,
            ][axis]
                .to_f64();
        }
        println!(
            "{name:<10} {:>10.1} {:>10.4} {:>8.4}   {:.5} / {:.5} / {:.5}",
            body.mass().to_f64(),
            shape.center_of_mass_offset().y.to_f64(),
            shape.bounding_radius().to_f64(),
            spin[0],
            spin[1],
            spin[2],
        );
    }
}
