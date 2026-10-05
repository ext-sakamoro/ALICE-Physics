//! SDF continuous collision detection: a fast sphere against a thin wall.
//!
//! A wall of half-thickness 0.05 (`f = |x| - 0.05`) and a sphere of radius
//! 0.5 fired at 300 m/s: one 1/60 s frame moves it 5 m, from x = -3 to x = 2,
//! so a discrete overlap test at the frame's end sees nothing. The sweep
//! (`PhysicsWorld::sdf_ccd_hits`, sphere tracing) finds the impact at
//! x = -0.05 - 0.5 = -0.55, i.e. t = (3 - 0.55) / 5 = 0.49, and `ray_march_sdf`
//! finds the wall surface at x = -0.05 along the same line.
//!
//! The same wall as a borrowed field: `sphere_trace_sdf_field` with a
//! `ClosureSdfQuery` whose closures borrow a local half-thickness (no `Box`,
//! no `'static`), placed by an `SdfFrame` 2 m along x, gives t = (5 - 0.55) / 5
//! = 0.89; `collide_point_sdf_field` / `collide_sphere_sdf_field` test a point
//! inside it and a sphere overlapping it.
//!
//! ```bash
//! cargo run --release --example sdf_ccd_sweep --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_ccd::{
    ray_march_sdf, sphere_trace_sdf, sphere_trace_sdf_field, SdfCcdConfig,
};
use alice_physics::sdf_collider::{
    collide_point_sdf_field, collide_sphere_sdf_field, ClosureSdf, ClosureSdfQuery, SdfCollider,
    SdfFrame,
};
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

    // The wall as a borrowed field: the closures capture a local.
    let half = 0.05_f32;
    let borrowed = ClosureSdfQuery::new(
        |x: f32, _y: f32, _z: f32| x.abs() - half,
        |x: f32, _y: f32, _z: f32| (if x < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
    );
    let start = Vec3Fix::from_f32(-3.0, 0.0, 0.0);
    let disp = Vec3Fix::from_f32(5.0, 0.0, 0.0);
    let r = Fix128::from_f32(0.5);
    let at_origin = sphere_trace_sdf_field(start, disp, r, &borrowed, &SdfFrame::IDENTITY, &cfg)
        .expect("borrowed wall hit");
    // same answer, bit for bit, as the boxed collider at the origin
    assert_eq!(
        Some(at_origin),
        sphere_trace_sdf(start, disp, r, &wall2, &cfg)
    );
    let moved = SdfFrame::new(
        Vec3Fix::from_f32(2.0, 0.0, 0.0),
        QuatFix::IDENTITY,
        Fix128::ONE,
    );
    let toi =
        sphere_trace_sdf_field(start, disp, r, &borrowed, &moved, &cfg).expect("moved wall hit");
    let t = toi.t.to_f32();
    println!("[sdf_ccd] borrowed wall at x = 2: t = {t:.5} (closed form 0.89)");
    assert!(t <= 0.89 + 1e-6 && 0.89 - t < 1e-3);

    let inside = collide_point_sdf_field(Vec3Fix::from_f32(2.02, 0.0, 0.0), &borrowed, &moved)
        .expect("point inside the wall");
    println!(
        "[sdf_ccd] point depth {:.4} (closed form 0.03)",
        inside.depth.to_f32()
    );
    assert!((inside.depth.to_f32() - 0.03).abs() < 1e-4);
    let overlap = collide_sphere_sdf_field(Vec3Fix::from_f32(1.6, 0.0, 0.0), r, &borrowed, &moved)
        .expect("sphere overlaps the wall");
    println!(
        "[sdf_ccd] sphere depth {:.4} (closed form 0.15)",
        overlap.depth.to_f32()
    );
    assert!((overlap.depth.to_f32() - 0.15).abs() < 1e-4);
}
