//! Mutual gravity among a world's bodies as a participant: two bodies on a
//! circular orbit, attracted inside every substep of `try_step`.
//!
//! The pair starts at separation 1 with `G = 1`, masses 1/2 and speeds 1/2,
//! a circular orbit of period `2π`. The world integrates the attraction
//! with the substep width, so the separation stays within `±h/2` of 1
//! (`h = dt / substeps`). The example prints the separation range, the
//! potential energy the participant observed and how many bodies took part.
//!
//! Run with `cargo run --example participant_nbody`.

use alice_physics_core::nbody::{
    DirectSum, NBodyParticipant, NBODY_OBS_BODIES, NBODY_OBS_POTENTIAL,
};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let substeps = 4;
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: Fix128::ZERO,
        substeps,
        ..PhysicsConfig::default()
    });
    let half = Fix128::from_ratio(1, 2);
    for s in [Fix128::ONE, -Fix128::ONE] {
        let mut body =
            RigidBody::new_dynamic(Vec3Fix::new(half * s, Fix128::ZERO, Fix128::ZERO), half);
        body.velocity = Vec3Fix::new(Fix128::ZERO, half * s, Fix128::ZERO);
        world.add_body(body);
    }
    let resident = NBodyParticipant::new(DirectSum::new(Fix128::ONE, Fix128::ZERO)?);
    println!(
        "G = {:.3}, softening = {:.3}",
        resident.law().gravitational_constant().to_f64(),
        resident.law().softening().to_f64()
    );
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("register: {e:?}"))?;

    let frames = 400;
    let dt = Fix128::TWO_PI / Fix128::from_int(frames);
    let (mut rmin, mut rmax) = (f64::MAX, 0.0_f64);
    for _ in 0..frames {
        world.try_step(dt).map_err(|e| format!("step: {e:?}"))?;
        let d = world.bodies[0].position - world.bodies[1].position;
        let r = d.length().to_f64();
        rmin = rmin.min(r);
        rmax = rmax.max(r);
    }
    let h = 2.0 * core::f64::consts::PI / (frames as f64 * substeps as f64);
    println!(
        "separation over one period: [{rmin:.6}, {rmax:.6}] (closed form 1 ± {:.6})",
        h / 2.0
    );

    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            match channel {
                NBODY_OBS_POTENTIAL => println!(
                    "potential energy: {:.6} (closed form -1/4)",
                    -(-value).to_f64()
                ),
                NBODY_OBS_BODIES => println!("bodies attracting: {}", value.to_f64()),
                _ => {}
            }
        }
    }
    Ok(())
}
