//! A 1-D Euler finite-volume solver as a world participant: the Sod shock
//! tube advanced through the world's substep loop, each substep split into
//! the solver's own CFL steps.
//!
//! Run: `cargo run --example participant_euler_fv`

use alice_physics_core::euler_fv::{
    EulerConfig, EulerFv1d, EulerFv1dParticipant, Primitive, EULER_FV_OBS_LAST_STEPS,
    EULER_FV_OBS_MASS, EULER_FV_OBS_STEPS, EULER_FV_OBS_TIME,
};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 100;
    let left = Primitive {
        rho: Fix128::ONE,
        u: Fix128::ZERO,
        p: Fix128::ONE,
    };
    let right = Primitive {
        rho: Fix128::from_ratio(1, 8),
        u: Fix128::ZERO,
        p: Fix128::from_ratio(1, 10),
    };
    let init: Vec<Primitive> = (0..n)
        .map(|i| if 2 * i < n { left } else { right })
        .collect();
    let gas = EulerFv1d::new(
        EulerConfig::godunov(Fix128::from_ratio(7, 5)),
        Fix128::from_ratio(1, n as i64),
        &init,
    )
    .map_err(|e| format!("solver: {e:?}"))?;
    let resident = EulerFv1dParticipant::new(gas);
    println!(
        "cells {}, steps so far {} (last substep {})",
        resident.solver().cells().len(),
        resident.steps(),
        resident.last_steps()
    );

    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 2,
        ..PhysicsConfig::default()
    });
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("register: {e:?}"))?;
    for _ in 0..4 {
        world
            .try_step(Fix128::from_ratio(1, 16))
            .map_err(|e| format!("step: {e:?}"))?;
    }
    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            let name = match channel {
                EULER_FV_OBS_TIME => "time (s)",
                EULER_FV_OBS_LAST_STEPS => "CFL steps in the last substep",
                EULER_FV_OBS_STEPS => "CFL steps in all",
                EULER_FV_OBS_MASS => "total mass / dx",
                _ => continue,
            };
            println!("{name}: {:.6}", value.to_f64());
        }
    }
    Ok(())
}
