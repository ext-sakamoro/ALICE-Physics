//! Particle emitters and force fields.
//!
//! Two emitters feed one `ParticleSystem` (`add_emitter` returns their indices);
//! the emission count follows `floor(rate * t)`. A directional wind is then applied
//! with `apply_force_field`, which adds `F / m * (1/60)` to every live particle's
//! velocity, so a 12 N wind on 3 kg particles adds 4/60 m/s along its direction.
//!
//! ```bash
//! cargo run --release --example particle_emitter_forces --features std
//! ```

use alice_physics::force::ForceField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::particle::{ParticleEmitter, ParticleSystem};
use alice_physics::rng::DeterministicRng;

fn main() {
    let mut ps = ParticleSystem::new(1000, Vec3Fix::from_int(0, -10, 0));
    let spark = ParticleEmitter::new(
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
        Fix128::from_int(32),
        Fix128::from_int(5),
        Fix128::from_int(10),
        Fix128::from_int(3),
    );
    let smoke = ParticleEmitter::new(
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::UNIT_Y,
        Fix128::from_ratio(1, 2),
        Fix128::from_int(16),
        Fix128::from_int(1),
        Fix128::from_int(10),
        Fix128::from_int(3),
    );
    let (a, b) = (ps.add_emitter(spark), ps.add_emitter(smoke));
    println!("emitter indices: {a}, {b}");
    assert_eq!((a, b), (0, 1));

    let mut rng = DeterministicRng::new(42);
    let dt = Fix128::from_ratio(1, 16);
    for _ in 0..16 {
        ps.step(dt, &mut rng);
    }
    // 1 s of (32 + 16) particles per second, dyadic dt so the count is exact
    println!("alive after 1 s: {} (closed form 48)", ps.alive_count());
    assert_eq!(ps.alive_count(), 48);

    let before: Vec<Vec3Fix> = ps.particles.iter().map(|p| p.velocity).collect();
    ps.apply_force_field(&ForceField::Directional {
        direction: Vec3Fix::UNIT_X,
        strength: Fix128::from_int(12),
    });
    let expected = 12.0 / 3.0 / 60.0;
    for (p, v0) in ps.particles.iter().zip(&before) {
        let dv = (p.velocity - *v0).x.to_f64();
        assert!((dv - expected).abs() < 1e-12, "dv = {dv}");
    }
    println!("wind added {expected:.6} m/s to every live particle");
}
