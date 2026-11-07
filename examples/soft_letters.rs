//! Soft letters: a block E cut into a soft body, grabbed, thrown and left to recover
//! and fall asleep inside a [`PhysicsWorld`].
//!
//! Run: `cargo run --example soft_letters`

use alice_physics_core::cloth::ClothConfig;
use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::soft_body::SoftBodyDesc;
use alice_physics_core::solver::{PhysicsWorld, SolverConfig};

fn main() {
    let r = |n: i64, d: i64| Fix128::from_ratio(n, d);
    let p = |x: i64, y: i64| [r(x, 10), r(y, 10)];
    let omega = Fix128::from_f32(core::f32::consts::TAU / 0.5);
    let desc = SoftBodyDesc {
        origin: Vec3Fix::ZERO,
        u: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        v: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        contours: vec![vec![
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
        ]],
        spacing: r(1, 10),
        mass_per_particle: Fix128::ONE,
        config: ClothConfig {
            iterations: 4,
            substeps: 1,
            gravity: Vec3Fix::ZERO,
            damping: r(985, 1000),
            stretch_compliance: r(1, 100),
            bend_compliance: r(1, 4),
            ..ClothConfig::default()
        },
        tether: Some((omega * omega, omega * Fix128::from_f32(2.0 * 0.42))),
    };
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    let e = world.add_soft_body(&desc).expect("the E keeps triangles");
    let body = &world.soft_bodies[e];
    println!(
        "E: {} particles, {} triangles",
        body.particle_count(),
        body.cloth().triangles.len()
    );
    let dt = r(1, 60);
    let grab = world
        .soft_grab(e, Vec3Fix::ZERO, r(34, 100))
        .expect("corner in reach");
    for _ in 0..30 {
        world.soft_move_grab(grab, Vec3Fix::from_f32(-0.9, 0.0, -0.5));
        world.step(dt);
    }
    world.soft_release_with_velocity(grab, Vec3Fix::from_f32(40.0, 0.0, -25.0));
    let mut asleep_at = None;
    for f in 1..=900u32 {
        world.step(dt);
        if world.soft_bodies[e].is_asleep() {
            asleep_at = Some(f);
            break;
        }
    }
    let c = world.soft_bodies[e].cloth();
    let dev = c
        .positions
        .iter()
        .zip(c.rest_positions())
        .map(|(p, q)| (*p - *q).length().to_f32())
        .fold(0.0, f32::max);
    println!("asleep after frame {asleep_at:?}, max deviation from rest {dev:.6}");
    let digest: String = world
        .state_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("state digest {digest}");
}
