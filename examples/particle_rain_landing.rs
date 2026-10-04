//! Particles Landing on Shapes Example
//!
//! Rain-like particles fall from an emitter onto a floor (an SDF field), a placed
//! ball (an SDF collider) and a non-SDF occupant (a point query). Each landing is
//! reported with its position, outward normal and particle index, and the particle is
//! re-emitted, so the number of live drops stays constant.
//!
//! ```bash
//! cargo run --example particle_rain_landing --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::particle::{LandingEvent, LandingTarget, ParticleEmitter, ParticleSystem};
use alice_physics::rng::DeterministicRng;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

fn main() {
    let floor = ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
    let ball = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::from_int(0, 2, 0),
        QuatFix::IDENTITY,
    );
    // stand-in for cloth particles: the slab 1 < x < 2, 0 < y < 1
    let slab = |p: Vec3Fix| {
        let inside = p.x > Fix128::ONE
            && p.x < Fix128::from_int(2)
            && p.y > Fix128::ZERO
            && p.y < Fix128::ONE;
        inside.then_some(Vec3Fix::UNIT_Y)
    };
    let targets = [
        LandingTarget::Field(&floor),
        LandingTarget::Collider(&ball),
        LandingTarget::Query(&slab),
    ];

    let mut rain = ParticleSystem::new(200, Vec3Fix::from_int(0, -10, 0));
    rain.add_emitter(ParticleEmitter::new(
        Vec3Fix::from_int(0, 6, 0),
        Vec3Fix::from_int(0, -1, 0),
        Fix128::from_ratio(3, 2),
        Fix128::from_int(600),
        Fix128::from_int(4),
        Fix128::from_int(100),
        Fix128::ONE,
    ));
    let mut rng = DeterministicRng::new(42);
    let dt = Fix128::from_ratio(1, 60);
    // no sample of a drop's path is more than 5 cm from the next one
    let max_travel = Fix128::from_ratio(1, 20);

    let mut per_target = [0usize; 3];
    let mut last: Option<LandingEvent> = None;
    for _ in 0..600 {
        for e in rain.step_with_landing(dt, &mut rng, &targets, max_travel, 0) {
            per_target[e.target] += 1;
            last = Some(e);
        }
    }
    println!("live drops: {}", rain.alive_count());
    println!(
        "landings: floor {}, ball {}, slab {}",
        per_target[0], per_target[1], per_target[2]
    );
    if let Some(e) = last {
        let (px, py, pz) = e.position.to_f32();
        let (nx, ny, nz) = e.normal.to_f32();
        println!(
            "last: particle {} at ({px:.3}, {py:.3}, {pz:.3}) normal ({nx:.2}, {ny:.2}, {nz:.2}) after {:.4} s",
            e.particle,
            e.time_in_step.to_f32()
        );
    }
}
