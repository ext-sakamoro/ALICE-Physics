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
    // a second pointer drags the top-right corner and lets go: the throw velocity is
    // the pointer's motion over the last step
    let second = world
        .soft_grab(e, Vec3Fix::from_f32(1.0, 0.0, 1.0), r(34, 100))
        .expect("corner in reach");
    let held = world.soft_bodies[e].grab_particle(second).expect("listed");
    assert_eq!(world.soft_bodies[e].grabs().len(), 1);
    assert!(world.has_soft_grab(second));
    let mut target = Vec3Fix::from_f32(1.0, 0.0, 1.0);
    for _ in 0..10 {
        target = target + Vec3Fix::from_f32(0.02, 0.0, 0.01);
        world.soft_move_grab(second, target);
        world.step(dt);
    }
    world.soft_release(second);
    let v = world.soft_bodies[e].cloth().velocities[held];
    println!(
        "thrown at ({:.3}, {:.3}) em/s after a step of {:.5} s",
        v.x.to_f32(),
        v.z.to_f32(),
        world.soft_bodies[e].last_dt().to_f32()
    );
    assert!(!world.has_soft_grab(second));
    // the nearest free particle to a point, as a cursor would show it
    let near = world.soft_bodies[e]
        .nearest_free(Vec3Fix::from_f32(0.5, 0.0, 0.5), r(34, 100))
        .expect("the middle arm is in reach");
    println!("nearest free particle to the middle: {near}");
    // the exact sleep can be switched off for a body that must step every frame
    world.soft_bodies[e].set_sleep(false);
    assert!(!world.soft_bodies[e].sleep_enabled());
    world.soft_bodies[e].set_sleep(true);
    // writing to the cloth directly is seen by the next step (a sleeping body wakes)
    let c = world.soft_bodies[e].cloth_mut();
    c.positions[near].x = c.positions[near].x + r(1, 100);
    assert!(c.pin_released().is_empty() && c.sleep_enabled());
    world.step(dt);
    // a cloth over any triangle mesh: one triangle sheet, as the lattice is built
    let sheet = alice_physics_core::cloth::Cloth::from_mesh(
        vec![
            Vec3Fix::ZERO,
            Vec3Fix::from_f32(1.0, 0.0, 0.0),
            Vec3Fix::from_f32(0.0, 0.0, 1.0),
        ],
        vec![[0, 1, 2]],
        Fix128::ONE,
    );
    assert_eq!(sheet.particle_count(), 3);
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
