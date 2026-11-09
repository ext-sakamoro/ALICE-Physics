//! A bounded fluid as a participant: balls lighter than the fluid float where
//! Archimedes says, inside every substep of `try_step`.
//!
//! A sphere of relative density `s` floats with `h²(3r − h)/(4r³) = s`, `h` the
//! submerged height of the cap. The example finds the root by bisection and
//! prints it beside the height each ball settled at, then rewinds to a snapshot
//! and shows that the second run lands on the same bits.
//!
//! Run with `cargo run --example participant_buoyancy_zone`.

use alice_physics_core::buoyancy_zone::{
    BuoyancyZone, BuoyancyZoneParticipant, ZoneShape, BUOYANCY_ZONE_OBS_BODIES,
};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

/// The height of a floating centre above the surface, for a radius of 1.
fn closed_form(relative_density: f64) -> f64 {
    let (mut low, mut high) = (0.0_f64, 2.0_f64);
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if mid * mid * (3.0 - mid) / 4.0 < relative_density {
            low = mid;
        } else {
            high = mid;
        }
    }
    1.0 - low
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gravity = Fix128::from_f64(9.81);
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, -gravity, Fix128::ZERO),
        damping_rate: Fix128::ZERO,
        substeps: 8,
        ..PhysicsConfig::default()
    });
    let densities = [0.25, 0.5, 0.75];
    let volume = 4.0 * core::f64::consts::PI / 3.0;
    let balls: Vec<usize> = densities
        .iter()
        .enumerate()
        .map(|(i, density)| {
            let at = Vec3Fix::new(
                Fix128::from_int(5 * i as i64),
                Fix128::from_ratio(1, 4),
                Fix128::ZERO,
            );
            let mass = Fix128::from_f64(density * volume);
            world.add_body_with_radius(RigidBody::new_dynamic(at, mass), Fix128::ONE)
        })
        .collect();
    let reach = Fix128::from_int(1000);
    let zone = BuoyancyZone {
        shape: ZoneShape::Aabb {
            min: Vec3Fix::new(-reach, -reach, -reach),
            max: Vec3Fix::new(reach, Fix128::ZERO, reach),
        },
        density_kg_m3: Fix128::ONE,
        gravity,
        drag_linear: Fix128::from_int(8),
        drag_quadratic: Fix128::ZERO,
    };
    let resident = BuoyancyZoneParticipant::new(zone);
    println!(
        "fluid density {:.1}, bodies in it before the first step: {}",
        resident.zone().density_kg_m3.to_f64(),
        resident.bodies_in_fluid()
    );
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("register: {e:?}"))?;

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..1800 {
        world.try_step(dt).map_err(|e| format!("step: {e:?}"))?;
    }
    for (&ball, density) in balls.iter().zip(densities) {
        let y = world.bodies[ball].position.y;
        let got = if y.is_negative() {
            -(-y).to_f64()
        } else {
            y.to_f64()
        };
        println!(
            "density {density:.2}: centre at {got:+.6} (closed form {:+.6})",
            closed_form(density)
        );
        // At rest the fluid's force carries the ball's weight.
        let body = &world.bodies[ball];
        let lift = zone
            .try_force_on(body, Fix128::ONE)
            .ok_or("the force left the fixed-point range")?;
        let weight = (gravity / body.inv_mass).to_f64();
        println!("  upward force {:.6}, weight {weight:.6}", lift.y.to_f64());
    }
    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            if channel == BUOYANCY_ZONE_OBS_BODIES {
                println!("bodies in the fluid: {:.0}", value.to_f64());
            }
        }
    }

    let snapshot = world.snapshot_world();
    let run = |w: &mut PhysicsWorld| -> Result<Vec3Fix, String> {
        for _ in 0..120 {
            w.try_step(dt).map_err(|e| format!("step: {e:?}"))?;
        }
        Ok(w.bodies[balls[0]].position)
    };
    let first = run(&mut world)?;
    world
        .restore_world(&snapshot)
        .map_err(|e| format!("restore: {e}"))?;
    let second = run(&mut world)?;
    println!("rewound run lands on the same bits: {}", first == second);
    Ok(())
}
