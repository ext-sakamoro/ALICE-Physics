//! Drops Running Off a Ball Example
//!
//! Drops fall on a ball, run down its surface under gravity with a viscous drag, and
//! drip from the point where the surface turns under. The digest at the end covers the
//! whole state of the rain, so two runs can be compared bit for bit.
//!
//! ```bash
//! cargo run --example particle_rain_run_off --features std
//! ```

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::particle::{LandingTarget, ParticleEmitter, ParticleSystem, RunOff};
use alice_physics_core::rng::DeterministicRng;
use alice_physics_core::sdf_collider::ClosureSdf;

fn main() {
    let ball = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    );
    let targets = [LandingTarget::Field(&ball)];

    let mut rain = ParticleSystem::new(64, Vec3Fix::from_int(0, -10, 0));
    rain.add_emitter(ParticleEmitter::new(
        Vec3Fix::from_int(0, 3, 0),
        Vec3Fix::from_int(0, -1, 0),
        Fix128::from_ratio(1, 2),
        Fix128::from_int(60),
        Fix128::ONE,
        Fix128::from_int(4),
        Fix128::ONE,
    ));
    let mut rng = DeterministicRng::new(7);
    let run_off = RunOff {
        drag: Fix128::from_int(3),
        drip_cos: Fix128::from_ratio(1, 2),
    };

    let (mut landed, mut dripped) = (0, 0);
    for _ in 0..240 {
        let events = rain.step_with_run_off(
            Fix128::from_ratio(1, 60),
            &mut rng,
            &targets,
            Fix128::from_ratio(1, 20),
            run_off,
        );
        landed += events.landed.len();
        dripped += events.dripped.len();
    }
    let on_ball = (0..rain.particles.len())
        .filter(|&i| rain.particle_on_surface(i).is_some())
        .count();
    println!("landed {landed}, dripped {dripped}, on the ball now {on_ball}");
    assert!(landed > 0 && dripped > 0, "the drops must land and drip");
    let hex: String = rain
        .state_digest(&rng)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("digest {hex}");
}
