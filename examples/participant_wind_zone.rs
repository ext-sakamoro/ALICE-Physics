//! A bounded wind as a participant: a ball released at rest in a steady wind
//! is pushed inside every substep of `try_step`.
//!
//! With no gravity the speed follows `dv/dt = c·(U − v)²`, `c = ½·ρ·C_d·A/m`,
//! whose solution is `v(t) = U − U/(1 + c·U·t)`. The example prints the speed
//! beside the closed form, then rewinds to a snapshot and shows that the second
//! run lands on the same bits.
//!
//! Run with `cargo run --example participant_wind_zone`.

use alice_physics_core::buoyancy_zone::ZoneShape;
use alice_physics_core::wind_zone::{WindZone, WindZoneParticipant, WIND_ZONE_OBS_TIME};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping_rate: Fix128::ZERO,
        substeps: 8,
        ..PhysicsConfig::default()
    });
    let mass = 2.0;
    let ball = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::ZERO,
        Fix128::from_f64(mass),
    ));
    let (speed, density, drag, area) = (10.0, 1.0, 1.0, 1.0);
    let reach = Fix128::from_int(1000);
    let zone = WindZone {
        shape: ZoneShape::Aabb {
            min: Vec3Fix::new(-reach, -reach, -reach),
            max: Vec3Fix::new(reach, reach, reach),
        },
        air_density_kg_m3: Fix128::from_f64(density),
        direction: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        base_speed_m_s: Fix128::from_f64(speed),
        turbulence_amplitude: Fix128::ZERO,
        gust_frequency_hz: Fix128::ZERO,
        drag_coefficient: Fix128::from_f64(drag),
        reference_area_m2: Fix128::from_f64(area),
    };
    let index = world
        .add_participant(Box::new(WindZoneParticipant::new(zone)))
        .map_err(|e| format!("register: {e:?}"))?;

    let dt = Fix128::from_ratio(1, 60);
    let c = 0.5 * density * drag * area / mass;
    for second in 1..=3 {
        for _ in 0..60 {
            world.try_step(dt).map_err(|e| format!("step: {e:?}"))?;
        }
        let t = f64::from(second);
        let want = speed - speed / (1.0 + c * speed * t);
        let got = world.bodies[ball].velocity.x.to_f64();
        println!("t = {t:.0} s: speed {got:.5} m/s (closed form {want:.5})");
    }

    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            if channel == WIND_ZONE_OBS_TIME {
                println!("zone clock: {:.4} s", value.to_f64());
            }
        }
    }

    let snapshot = world.snapshot_world();
    let run = |w: &mut PhysicsWorld| -> Result<Vec3Fix, String> {
        for _ in 0..120 {
            w.try_step(dt).map_err(|e| format!("step: {e:?}"))?;
        }
        Ok(w.bodies[ball].position)
    };
    let first = run(&mut world)?;
    world
        .restore_world(&snapshot)
        .map_err(|e| format!("restore: {e}"))?;
    let second = run(&mut world)?;
    println!("rewound run lands on the same bits: {}", first == second);
    Ok(())
}
