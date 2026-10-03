//! SDF continuous collision detection: a fast sphere against a thin wall.
//!
//! A wall of half-thickness 0.05 (`f = |x| - 0.05`) and a sphere of radius
//! 0.5 fired at 300 m/s: one 1/60 s frame moves it 5 m, from x = -3 to x = 2,
//! so a discrete overlap test at the frame's end sees nothing. The sweep
//! (`PhysicsWorld::sdf_ccd_hits`, sphere tracing) finds the impact at
//! x = -0.05 - 0.5 = -0.55, i.e. t = (3 - 0.55) / 5 = 0.49, and `ray_march_sdf`
//! finds the wall surface at x = -0.05 along the same line.
//!
//! ```bash
//! cargo run --release --example sdf_ccd_sweep --features std
//! ```

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_ccd::{ray_march_sdf, SdfCcdConfig};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    let wall = ClosureSdf::new(
        |x, _y, _z| x.abs() - 0.05,
        |x, _y, _z| (if x < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
    );
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    world.set_sdf_collision_radius(Fix128::from_f32(0.5));
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(wall),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    let bullet = world.add_body(
        RigidBody::new_dynamic(Vec3Fix::from_f32(-3.0, 0.0, 0.0), Fix128::ONE)
            .with_velocity(Vec3Fix::from_f32(300.0, 0.0, 0.0)),
    );
    let dt = Fix128::from_ratio(1, 60);
    let cfg = SdfCcdConfig::default();

    let end_x = world.bodies[bullet].position.x.to_f32() + 300.0 / 60.0;
    println!("[sdf_ccd] frame end x = {end_x} (past the wall: a discrete test misses it)");
    assert!(end_x > 0.55);

    let hits = world.sdf_ccd_hits(dt, &cfg);
    assert_eq!(hits.len(), 1, "one sweep hit expected");
    let (body, sdf, toi) = hits[0];
    let t = toi.t.to_f32();
    println!("[sdf_ccd] body {body} vs sdf {sdf}: t = {t:.5} (closed form 0.49)");
    assert_eq!((body, sdf), (bullet, 0));
    assert!(t <= 0.49 + 1e-6 && 0.49 - t < 1e-3);
    let (nx, _, _) = toi.normal.to_f32();
    assert!((nx + 1.0).abs() < 1e-4);

    // A direction of length 2 gives the same wall distance (4 / 2 -> distance 2.95).
    let wall2 = SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, _y, _z| x.abs() - 0.05,
            |x, _y, _z| (if x < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let ray = ray_march_sdf(
        Vec3Fix::from_f32(-3.0, 0.0, 0.0),
        Vec3Fix::from_f32(2.0, 0.0, 0.0),
        Fix128::from_int(10),
        &wall2,
        &cfg,
    )
    .expect("ray hits the wall");
    let (px, _, _) = ray.point.to_f32();
    println!(
        "[sdf_ccd] ray distance {:.4} (closed form 2.95), point x = {px:.4}",
        ray.t.to_f32()
    );
    assert!((ray.t.to_f32() - 2.95).abs() < 2e-3);
    assert!((px + 0.05).abs() < 2e-3);
}
