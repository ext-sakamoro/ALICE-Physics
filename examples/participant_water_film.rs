//! A water film as a world participant: a groove 3 cm deep cut in flat ground (1 cm cells)
//! fills under 20 mm/h of rain. The film runs off the ground into the groove, the groove's
//! water stands level as a pool, and the volume ledger holds to the last quantum.
//!
//! Run: `cargo run --example participant_water_film`

use alice_physics_core::water_film::{
    metres_to_quanta, quanta_to_metres, FilmBed, FilmConfig, FilmEdge, FilmScheme, WaterFilm,
};
use alice_physics_core::{Fix128, PhysicsConfig, PhysicsWorld, Vec3Fix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (nx, nz) = (24, 9);
    let cell = Fix128::from_ratio(1, 100);
    // the groove: 3 cm deep, cells 6..18 along x, rows 3..6 across
    let groove = |x: f32, z: f32| -> f32 {
        if (0.06..0.18).contains(&x) && (0.03..0.06).contains(&z) {
            -0.03
        } else {
            0.0
        }
    };
    let bed = FilmBed::from_height_fn(nx, nz, cell, (Fix128::ZERO, Fix128::ZERO), 1, groove)
        .map_err(|e| format!("bed: {e:?}"))?;
    let mut cfg = FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Closed);
    cfg.rain = FilmConfig::rain_mm_per_hour(Fix128::from_int(20));
    let mut film = WaterFilm::new(cfg, bed).map_err(|e| format!("film: {e:?}"))?;
    // a damp start: a 20 µm film on the ground (below the rest residue, it stays put), and a
    // spoonful of water in the groove
    for k in 0..nz {
        for i in 0..nx {
            if film.bed().heights()[i + k * nx] == 0 {
                film.set_initial_depth(i, k, metres_to_quanta(Fix128::from_ratio(2, 100_000)));
            }
        }
    }
    let (x, z) = film.bed().centre(12, 4);
    film.deposit(x, z, Fix128::from_ratio(1, 1_000_000_000));
    // the same ground as a new generation (as after the ground is re-cut): water kept
    let regen = FilmBed::from_height_fn(nx, nz, cell, (Fix128::ZERO, Fix128::ZERO), 2, groove)
        .map_err(|e| format!("bed: {e:?}"))?;
    film.set_bed(regen).map_err(|e| format!("set_bed: {e:?}"))?;

    // a twin that visits the faces in reverse order: the state does not depend on it
    let mut twin = film.clone();
    twin.set_reverse_faces(true);

    let step = Fix128::from_ratio(1, 6);
    for _ in 0..(6 * 60) {
        if let Err(e) = film.advance(step) {
            return Err(format!(
                "advance: {e:?}, out-of-range cells {:?}",
                film.last_band_cells()
            )
            .into());
        }
        twin.advance(step).map_err(|e| format!("twin: {e:?}"))?;
    }
    assert_eq!(film.depths(), twin.depths(), "face order changed the state");
    let ledger = film.ledger();
    assert_eq!(film.total(), ledger.expected(), "volume ledger");
    let pooled = film.last_pooled_cells().len();
    let dry = film.dry_ages().iter().filter(|a| **a > 0).count();
    println!(
        "after 60 s: groove depth {:.3} mm, {pooled} pooled cells, {dry} dry cells, last substep moved {} quanta, {} capped faces, {} out-of-range cells",
        film.depth_m(12, 4).to_f64() * 1000.0,
        film.last_moved(),
        film.last_cap_faces(),
        film.last_band_cells().len(),
    );
    println!(
        "water {:.3e} m·cell (rain {:.3e}, deposits {:.3e})",
        quanta_to_metres(film.total()).to_f64(),
        quanta_to_metres(ledger.rain).to_f64(),
        quanta_to_metres(ledger.deposits).to_f64(),
    );

    // the same film stepping inside a world
    let mut world = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(
            Fix128::ZERO,
            -Fix128::from_ratio(980_665, 100_000),
            Fix128::ZERO,
        ),
        ..PhysicsConfig::default()
    });
    world
        .add_participant(Box::new(film))
        .map_err(|e| format!("register: {e:?}"))?;
    for _ in 0..60 {
        world
            .try_step(Fix128::from_ratio(1, 60))
            .map_err(|e| format!("step: {e:?}"))?;
    }
    println!("stepped one more second inside a world");
    Ok(())
}
