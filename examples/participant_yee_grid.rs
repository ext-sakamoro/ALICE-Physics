//! A Yee (FDTD) lattice as a world participant: a 4×4 PEC cavity mode that
//! returns to its start every 6 lattice steps, run through the world's substep
//! loop on its own fixed step.
//!
//! Run: `cargo run --example participant_yee_grid`

use alice_physics_core::maxwell_fdtd::{
    Component, YeeGrid, YeeGridParticipant, YEE_GRID_OBS_MAX_FIELD, YEE_GRID_OBS_STEPS,
    YEE_GRID_OBS_TIME,
};
use alice_physics_core::world_participant::Observed;
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = [0_i64, 1, 0, -1, 0];
    let mut grid = YeeGrid::new(4, 4, 1, Fix128::from_ratio(1, 2));
    for (a, sa) in mode.iter().enumerate() {
        for (b, sb) in mode.iter().enumerate() {
            grid.set(Component::Ez, a, b, 0, Fix128::from_int(sa * sb));
        }
    }
    let step = Fix128::from_ratio(1, 64);
    let resident = YeeGridParticipant::new(grid, step)?;
    println!(
        "lattice {} steps at {:.6} s, one step = {:.6} s",
        resident.steps(),
        resident.time().to_f64(),
        resident.step().to_f64()
    );
    let start = resident.grid().get(Component::Ez, 1, 1, 0);

    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        substeps: 2,
        ..PhysicsConfig::default()
    });
    let index = world
        .add_participant(Box::new(resident))
        .map_err(|e| format!("register: {e:?}"))?;
    let period = Fix128::from_int(6) * step;
    for _ in 0..10 {
        world.try_step(period).map_err(|e| format!("step: {e:?}"))?;
    }
    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            let name = match channel {
                YEE_GRID_OBS_TIME => "time (s)",
                YEE_GRID_OBS_STEPS => "lattice steps",
                YEE_GRID_OBS_MAX_FIELD => "max |field|",
                _ => "?",
            };
            println!("{name}: {:.6}", value.to_f64());
        }
    }
    println!(
        "Ez(1,1) at start: {:.3} (returns every 6 steps)",
        start.to_f64()
    );
    Ok(())
}
