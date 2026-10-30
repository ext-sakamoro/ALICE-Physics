//! A magnet pulling a steel ball
//!
//! A 1 cm steel ball flies past a permanent magnet (point dipole, moment 1 A·m² along +y, about a
//! strong 1 cm³ neodymium magnet) with no gravity. The magnet is a `ForceField::InducedDipole` that pulls only the bodies of the metal
//! material: the ball's moment is induced by the field, so the pull grows as 1/r⁷ far away and as
//! 1/r⁴ once the steel saturates. The force is conservative, so the printed energy ½mv² + U stays
//! put up to the integrator's error.
//!
//! ```bash
//! cargo run --example magnet_induced_dipole
//! ```

use alice_physics_core::electromagnetic::{dipole_sample, induced_dipole_energy, MagneticResponse};
use alice_physics_core::force::apply_force_fields;
use alice_physics_core::{
    Fix128, ForceField, ForceFieldInstance, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix,
};

fn main() {
    let config = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    };
    let mut world = PhysicsWorld::new(config);

    // Soft iron / steel: effective susceptibility 3 (a sphere), saturation 1.7e6 A/m, 7850 kg/m³
    let steel = MagneticResponse {
        chi_eff: Fix128::from_int(3),
        saturation_magnetization: Fix128::from_int(1_700_000),
    };
    let density = Fix128::from_int(7850);
    let metal = world.material_table.register_metal();

    // A 1 cm ball (radius 5 mm, mass 4/3·π·r³·ρ ≈ 4.11 g) moving at 1 m/s along +x, 4 cm above the
    // magnet's axis line
    let mass = Fix128::from_ratio(4_110, 1_000_000);
    let mut ball = RigidBody::new_dynamic(
        Vec3Fix::new(
            Fix128::from_ratio(-15, 100),
            Fix128::from_ratio(4, 100),
            Fix128::ZERO,
        ),
        mass,
    );
    ball.velocity = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let b = world.add_body_with_radius(ball, Fix128::from_ratio(5, 1000));
    world.set_body_material(b, metal);

    let (position, moment) = (
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
    );
    world.add_force_field(ForceFieldInstance::new(ForceField::InducedDipole {
        position,
        moment,
        radius: Fix128::from_ratio(1, 100),
        material: metal,
        response: steel,
        density,
    }));

    let volume = mass / density;
    let dt = Fix128::from_ratio(1, 1000);
    println!("  t [ms]   x [cm]   y [cm]   |v| [m/s]   ½mv² + U [µJ]");
    for step in 0..=300 {
        if step % 30 == 0 {
            let body = &world.bodies[b];
            let kinetic = body.velocity.length_squared() * mass / Fix128::from_int(2);
            let field = dipole_sample(position, moment, body.position).field;
            let energy = kinetic + induced_dipole_energy(field, &steel, volume);
            println!(
                "{:>8} {:>8.3} {:>8.3} {:>11.4} {:>15.4}",
                step,
                body.position.x.to_f64() * 100.0,
                body.position.y.to_f64() * 100.0,
                body.velocity.length().to_f64(),
                energy.to_f64() * 1e6,
            );
        }
        world.step(dt);
    }

    // Outside a world the force fields act on a slice of bodies, with each body's material and
    // collision radius: the magnet pulls the bodies of its material, the wind pulls every body.
    let wind = ForceFieldInstance::new(ForceField::Directional {
        direction: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        strength: Fix128::ONE,
    });
    let mut fields = world.force_fields.clone();
    fields.push(wind);
    let radii = vec![Some(Fix128::from_ratio(5, 1000)); world.bodies.len()];
    let mut bodies = world.bodies.clone();
    let before = bodies[b].velocity;
    apply_force_fields(&fields, &world.body_materials, &radii, &mut bodies, dt);
    let dv = bodies[b].velocity - before;
    println!(
        "one step of magnet + 1 N wind on a slice: dv = ({:.6}, {:.6}, {:.4}) m/s",
        dv.x.to_f64(),
        dv.y.to_f64(),
        dv.z.to_f64()
    );
}
