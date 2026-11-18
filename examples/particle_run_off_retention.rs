//! Drops Held by a Surface Example
//!
//! Rain falls on a gently tilted plate. Contact-angle hysteresis holds each drop with the
//! Furmidge force: on this plate the tilt is below the critical angle, so every drop stops
//! where it lands, is pinned, and hands its water to the water film over the plate. The
//! water born, the water the film took, the part below one film quantum still held and the
//! water still in the air add up exactly.
//!
//! ```bash
//! cargo run --example particle_run_off_retention --features std
//! ```

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::particle::{
    retention_acceleration, FilmHandOff, LandingTarget, Particle, ParticleSystem, RunOff,
};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::ClosureSdf;
use alice_physics_core::water_film::{FilmBed, FilmConfig, FilmEdge, FilmScheme, WaterFilm};

fn main() {
    // water on a surface with advancing 80° and receding 50° (cosines), a drop 3 mm wide of
    // 25 mm³: the retention per unit mass, and the critical tilt it gives
    let retention = retention_acceleration(
        Fix128::from_ratio(72, 1000),
        Fix128::from_int(1000),
        Fix128::from_ratio(25, 1_000_000_000),
        Fix128::from_ratio(3, 1000),
        Fix128::from_ratio(6428, 10000),
        Fix128::from_ratio(1736, 10000),
    );
    let g = 10.0_f32;
    println!(
        "retention {:.3} m/s², critical tilt sin α = {:.3}",
        retention.to_f64(),
        retention.to_f64() / f64::from(g)
    );

    // a plate tilted at sin α = 0.05, below the critical tilt
    let (sx, sy) = (0.05_f32, (1.0_f32 - 0.05 * 0.05).sqrt());
    let plate = ClosureSdf::new(move |x, y, _| sx * x + sy * y, move |_, _, _| (sx, sy, 0.0));
    let targets = [LandingTarget::Field(&plate)];

    let cell = Fix128::from_ratio(1, 100);
    let bed = FilmBed::from_height_fn(30, 30, cell, (Fix128::ZERO, Fix128::ZERO), 1, |x, _| {
        -sx / sy * x
    })
    .expect("bed");
    let mut film = WaterFilm::new(
        FilmConfig::water(FilmScheme::DiffusionWave, FilmEdge::Open),
        bed,
    )
    .expect("film");

    let mut rain = ParticleSystem::new(usize::MAX, Vec3Fix::from_int(0, -10, 0));
    let mut rng = DeterministicRng::new(7);
    let run_off = RunOff {
        drag: Fix128::from_int(2),
        drip_cos: Fix128::from_ratio(1, 2),
        retention,
    };
    let volume = Fix128::from_ratio(25, 1_000_000_000);
    let mut hand = FilmHandOff::default();
    let mut born = Fix128::ZERO;
    let step = Fix128::from_ratio(1, 60);
    for n in 0..300_i64 {
        if n % 2 == 0 {
            let k = n / 2;
            let x = Fix128::from_ratio(10 + (k * 37) % 280, 1000);
            let z = Fix128::from_ratio(10 + (k * 53) % 280, 1000);
            rain.particles.push(Particle::new(
                Vec3Fix::new(x, Fix128::from_ratio(3, 10), z),
                Vec3Fix::ZERO,
                Fix128::from_int(5),
                Fix128::ONE,
            ));
            born = born + volume;
        }
        let events =
            rain.step_with_run_off(step, &mut rng, &targets, Fix128::from_ratio(1, 4), run_off);
        hand.hand_off(&mut film, &events.pinned, volume);
    }
    let in_air = rain.particles.iter().filter(|p| p.alive).count();
    let air = volume * Fix128::from_int(in_air as i64);
    println!(
        "born {:.3e} m³ = film {:.3e} + held {:.3e} + off the bed {:.3e} + in the air {:.3e} ({})",
        born.to_f64(),
        hand.deposited().to_f64(),
        hand.held().to_f64(),
        hand.off_bed().to_f64(),
        air.to_f64(),
        if hand.deposited() + hand.held() + hand.off_bed() + air == born {
            "exact"
        } else {
            "NOT exact"
        }
    );
    println!(
        "film quanta {} (ledger {})",
        hand.quanta(),
        film.ledger().expected()
    );
}
