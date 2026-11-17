//! Smoke as a world participant: a small burner on the floor of a closed box of air
//! (16 × 32 × 16 cells of 1 cm) heats the air above it and makes soot; the hot, sooty air
//! rises as a plume. Heat and soot are held as integer quanta and the ledger holds to the
//! last one, and a twin that visits the cells in reverse order ends with the same bits.
//!
//! Run: `cargo run --release --example participant_smoke_plume`

use alice_physics_core::smoke_fire::ArrheniusReaction;
use alice_physics_core::smoke_plume::{
    from_quanta, source_rates, CellBox, SmokeConfig, SmokePlume, SmokeSource, SourceKind,
    HEAT_QUANTUM_BITS, SMOKE_OBS_HEAT, SMOKE_OBS_MAX_EXCESS, SMOKE_OBS_SOOT, SOOT_QUANTUM_BITS,
};
use alice_physics_core::world_participant::{Observed, Participant};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, ny, nz) = (16, 32, 16);
    let dx = Fix128::from_ratio(1, 100);
    let mut config = SmokeConfig::air();
    config.step_max = Fix128::from_ratio(1, 128);
    let mut plume = SmokePlume::new(nx, ny, nz, dx, config).map_err(|e| format!("{e:?}"))?;
    let burner = SourceKind::Burner {
        reaction: ArrheniusReaction::pla_air(),
        fuel_kg_m3: Fix128::from_ratio(1, 100),
        oxidizer_kg_m3: Fix128::from_ratio(23, 100),
    };
    plume
        .add_source(SmokeSource {
            cells: CellBox {
                lo: [7, 1, 7],
                hi: [9, 2, 9],
            },
            kind: burner,
        })
        .map_err(|e| format!("{e:?}"))?;
    // light it: the burner cells start 500 K above the air
    for (i, k) in [(7, 7), (8, 7), (7, 8), (8, 8)] {
        plume
            .set_initial(i, 1, k, 500 << HEAT_QUANTUM_BITS, 0)
            .map_err(|e| format!("{e:?}"))?;
    }
    let (heat_rate, soot_rate) = source_rates(&burner, plume.config(), Fix128::from_int(500));
    println!(
        "burner at 800 K: heats {:.1} K/s, makes {:.3e} kg/(m³·s) of soot",
        heat_rate.to_f64(),
        soot_rate.to_f64()
    );

    let mut twin = plume.clone();
    twin.set_reverse_cells(true);
    let step = Fix128::from_ratio(1, 128);
    for _ in 0..64 {
        plume.advance(step).map_err(|e| format!("advance: {e:?}"))?;
        twin.advance(step).map_err(|e| format!("twin: {e:?}"))?;
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    plume.write_state(&mut a);
    twin.write_state(&mut b);
    assert_eq!(a, b, "cell order changed the state");
    let ledger = plume.ledger();
    assert_eq!(
        plume.heat().total(),
        Some(ledger.heat_expected()),
        "heat ledger"
    );
    assert_eq!(
        plume.soot().total(),
        Some(ledger.soot_expected()),
        "soot ledger"
    );
    // how high the plume has risen: the topmost layer holding 1 K above the air
    let f = plume.heat();
    let top = (0..ny)
        .filter(|&j| {
            (0..nz).any(|k| (0..nx).any(|i| f.quanta()[f.index(i, j, k)] >= 1 << HEAT_QUANTUM_BITS))
        })
        .max()
        .unwrap_or(0);
    println!(
        "after 0.5 s: {} velocity steps, {} transport and {} diffusion substeps in the last, plume top at {:.2} m, burner cell {:.0} K above the air",
        plume.steps(),
        plume.last_transport_substeps(),
        plume.last_diffusion_substeps(),
        (top as f64 + 0.5) * dx.to_f64(),
        plume.excess_k(7, 1, 7).to_f64(),
    );
    // the last advance: velocity steps it took, quanta it moved, and the fastest rise
    let grid = plume.velocity();
    let rise = grid
        .v
        .iter()
        .copied()
        .fold(Fix128::ZERO, |m, v| if v > m { v } else { m });
    assert!(
        plume.last_steps() >= 1,
        "an advance takes at least one velocity step"
    );
    assert!(plume.last_moved() > 0, "a lit plume moves quanta");
    println!(
        "last advance: {} velocity steps, {} quanta moved, fastest rise {:.3} m/s on a {}×{}×{} grid",
        plume.last_steps(),
        plume.last_moved(),
        rise.to_f64(),
        grid.nx,
        grid.ny,
        grid.nz,
    );
    println!(
        "heat {:.4e} K·cell, soot {:.4e} kg/m³·cell (ledger exact)",
        from_quanta(ledger.heat_expected(), HEAT_QUANTUM_BITS).to_f64(),
        from_quanta(ledger.soot_expected(), SOOT_QUANTUM_BITS).to_f64(),
    );

    // the same plume stepping inside a world
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(
            Fix128::ZERO,
            -Fix128::from_ratio(980_665, 100_000),
            Fix128::ZERO,
        ),
        ..PhysicsConfig::default()
    });
    let index = world
        .add_participant(Box::new(plume))
        .map_err(|e| format!("register: {e:?}"))?;
    for _ in 0..8 {
        world
            .try_step(Fix128::from_ratio(1, 64))
            .map_err(|e| format!("step: {e:?}"))?;
    }
    if let Some(Observed::Exact(sink)) = world.observe_participant(index) {
        for &(channel, value) in sink.values() {
            match channel {
                SMOKE_OBS_HEAT => println!("inside a world: heat {:.4e} K·m³", value.to_f64()),
                SMOKE_OBS_SOOT => println!("inside a world: soot {:.4e} kg", value.to_f64()),
                SMOKE_OBS_MAX_EXCESS => {
                    println!(
                        "inside a world: hottest cell {:.1} K above the air",
                        value.to_f64()
                    );
                }
                _ => {}
            }
        }
    }
    Ok(())
}
