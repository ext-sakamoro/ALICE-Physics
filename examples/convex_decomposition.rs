//! A concave solid from an SDF, as a body of convex pieces
//!
//! An L made of two boxes is given as a signed distance field. `decompose_sdf`
//! cuts it where it is concave, `CompoundShape::from_sdf` turns the pieces into a
//! compound, and `add_compound_body` makes a body that collides as an L: a probe in
//! the notch (inside the L's convex hull, outside the L) touches nothing.
//!
//! ```bash
//! cargo run --example convex_decomposition --features std
//! ```

use alice_physics::compound::CompoundShape;
use alice_physics::convex_decompose::{decompose_sdf, DecomposeConfig};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::shape::Shape;
use alice_physics::solver::{PhysicsWorld, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

/// Signed distance to the box `[lo, hi]` (negative inside).
fn box_sd(p: [f32; 3], lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let mut outside = 0.0f32;
    let mut inside = f32::MIN;
    for k in 0..3 {
        let d = (p[k] - 0.5 * (lo[k] + hi[k])).abs() - 0.5 * (hi[k] - lo[k]);
        outside += d.max(0.0) * d.max(0.0);
        inside = inside.max(d);
    }
    outside.sqrt() + inside.min(0.0)
}

fn main() {
    // [0,2]x[0,1]x[0,1] and [0,1]x[1,2]x[0,1]: volume 3.
    let l_shape = ClosureSdf::new(
        |x, y, z| {
            let p = [x, y, z];
            box_sd(p, [0.0, 0.0, 0.0], [2.0, 1.0, 1.0]).min(box_sd(
                p,
                [0.0, 1.0, 0.0],
                [1.0, 2.0, 1.0],
            ))
        },
        |_, _, _| (0.0, 1.0, 0.0),
    );
    let (lo, hi) = (v3(-1.0, -1.0, -1.0), v3(3.0, 3.0, 3.0));
    let config = DecomposeConfig {
        resolution: 32,
        ..DecomposeConfig::default()
    };

    let pieces = decompose_sdf(&l_shape, lo, hi, &config);
    println!(
        "the L is cut into {} convex hulls (volumes {:?}; a hull is the solid shrunk by half a cell)",
        pieces.hulls.len(),
        pieces
            .volumes
            .iter()
            .map(|v| format!("{:.3}", v.to_f64()))
            .collect::<Vec<_>>()
    );

    let density = Fix128::from_int(1000);
    let compound = CompoundShape::from_sdf(&l_shape, lo, hi, &config);
    let props = compound.mass_properties(density);
    println!(
        "compound: {} children, mass {:.1} kg, centre of mass ({:.3}, {:.3}, {:.3})",
        compound.len(),
        props.mass.to_f64(),
        props.center_of_mass.x.to_f64(),
        props.center_of_mass.y.to_f64(),
        props.center_of_mass.z.to_f64()
    );

    let mut world = PhysicsWorld::new(SolverConfig::default());
    let body = world
        .add_compound_body(&compound, density, Vec3Fix::ZERO)
        .expect("the L has volume");
    let probe = Shape::Ellipsoid {
        radii: v3(0.15, 0.15, 0.15),
    };
    let com = props.center_of_mass;
    for (what, x, y) in [
        ("in the notch", 1.3, 1.3),
        ("on the vertical bar", 0.5, 1.8),
        ("on the horizontal bar", 1.8, 0.5),
    ] {
        let at = v3(x, y, 0.5) - com;
        let p = world
            .add_shaped_body(&probe, density, at)
            .expect("a valid solid");
        println!(
            "probe {what:<22} overlaps: {}",
            world.colliders_overlap(body, p)
        );
    }
}
