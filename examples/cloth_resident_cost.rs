//! Frame cost of a cloth on the world's own cloth list (today) and as a resident
//! ([`ClothParticipant`], `StepRule::PerFrame`), for the same thrown letter E.
//!
//! `cargo run --release --example cloth_resident_cost` prints one line per case: world
//! substeps, cloth steps per frame, iterations per step, milliseconds per frame (the
//! mean of 240 frames after 60 warm-up frames). Only the timing reads the wall clock;
//! no simulated value depends on it.
#![allow(
    clippy::disallowed_methods,
    reason = "a cost measurement: the wall clock times the frames and never feeds the simulation"
)]

use alice_physics_core::cloth::{Cloth, ClothConfig, OutlineDesc};
use alice_physics_core::cloth_participant::ClothParticipant;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn letter(steps_per_frame: usize, iterations: usize) -> Cloth {
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    let outline = vec![vec![
        p(0, 0),
        p(10, 0),
        p(10, 2),
        p(2, 2),
        p(2, 4),
        p(10, 4),
        p(10, 6),
        p(2, 6),
        p(2, 8),
        p(10, 8),
        p(10, 10),
        p(0, 10),
    ]];
    let omega = Fix128::from_ratio(12_566, 500);
    let tether = (omega * omega, omega * Fix128::from_ratio(84, 100));
    let mut c = Cloth::from_outline(&OutlineDesc {
        origin: Vec3Fix::ZERO,
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours: outline,
        spacing: r(1, 10),
        mass_per_particle: Fix128::ONE,
        config: ClothConfig {
            iterations,
            substeps: steps_per_frame,
            gravity: Vec3Fix::ZERO,
            damping: r(985, 1000),
            stretch_compliance: r(1, 100),
            bend_compliance: r(1, 4),
            ..ClothConfig::default()
        },
        tether: Some(tether),
    })
    .expect("E");
    c.set_sleep(false);
    for v in &mut c.velocities {
        *v = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    }
    c
}

fn world(substeps: usize) -> PhysicsWorld {
    PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        substeps,
        ..SolverConfig::default()
    })
}

fn ms_per_frame(mut w: PhysicsWorld) -> f64 {
    let dt = r(1, 60);
    for _ in 0..60 {
        w.step(dt);
    }
    let t = std::time::Instant::now();
    for _ in 0..240 {
        w.step(dt);
    }
    t.elapsed().as_secs_f64() * 1e3 / 240.0
}

fn main() {
    for n in [2usize, 8] {
        let mut today = world(n);
        today.add_cloth(letter(1, 4));
        println!(
            "[cloth cost] substeps {n}: today (1 step x 4 it) {:.3} ms/frame",
            ms_per_frame(today)
        );
        for (m, it) in [(1usize, 4usize), (2, 2), (2, 4)] {
            let mut w = world(n);
            w.add_participant(Box::new(ClothParticipant::new(letter(m, it))))
                .expect("register");
            println!(
                "[cloth cost] substeps {n}: resident ({m} steps x {it} it) {:.3} ms/frame",
                ms_per_frame(w)
            );
        }
    }
}
