//! An SDF collider carried by a body
//!
//! Reaches `SdfCollider::new_dynamic`, `sync_to_body`, `set_pose`,
//! `update_cache` and `sdf_collider::sync_dynamic_sdf_colliders`.
//!
//! A dynamic SDF collider evaluates its field in the frame of the body it is
//! attached to, but it holds its own copy of that pose: the copy is refreshed
//! by `sync_dynamic_sdf_colliders(&mut world.sdf_colliders, &world.bodies)`,
//! which has to run after the bodies move and before the colliders are
//! queried. This file moves a carrier body by hand, syncs, and checks the
//! contact a probe body reports against closed forms:
//!
//! - a ball of radius 1 on a carrier at `P`: a probe sphere of radius `r`
//!   centred at `P + d·e` penetrates by `r − (d − 1)` along `e`;
//! - a local half-space `y < 0` on a carrier turned +90° about `z`: the solid
//!   is the world side `x > P_x`, outward normal `−x`.
//!
//! Run with: `cargo run --example sdf_dynamic_collider_pose`

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{
    collide_point_sdf, sync_dynamic_sdf_colliders, ClosureSdf, SdfCollider, SdfFrame,
};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn main() {
    let mut world = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    world.sdf_collision_radius = Fix128::from_f64(0.5);
    let carrier = world.add_body(RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE));
    let probe = world.add_body(RigidBody::new_dynamic(v3(4.0, 1.25, -2.0), Fix128::ONE));
    world.add_sdf_collider(SdfCollider::new_dynamic(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt().max(1e-6);
                (x / l, y / l, z / l)
            },
        )),
        carrier,
    ));

    // The carrier moves under the probe; the collider is still at the origin.
    world.bodies[carrier].position = v3(4.0, 0.0, -2.0);
    assert!(world.sdf_contacts().is_empty());

    let moved = sync_dynamic_sdf_colliders(&mut world.sdf_colliders, &world.bodies);
    assert_eq!(moved, 1);
    assert_eq!(
        world.sdf_colliders[0].frame(),
        SdfFrame::new(v3(4.0, 0.0, -2.0), QuatFix::IDENTITY, Fix128::ONE)
    );
    // Probe 1.25 above the ball centre, radius 0.5: penetration 0.25, normal +y.
    // The carrier sits inside its own ball and is skipped.
    let contacts = world.sdf_contacts();
    assert_eq!(contacts.len(), 1);
    let (who, c) = &contacts[0];
    assert_eq!(*who, probe);
    assert!(
        near(c.depth.to_f64(), 0.25, 1e-6),
        "depth {}",
        c.depth.to_f64()
    );
    let (nx, ny, nz) = c.normal.to_f32();
    assert!(near(f64::from(nx), 0.0, 1e-6) && near(f64::from(ny), 1.0, 1e-6));
    assert!(near(f64::from(nz), 0.0, 1e-6));
    println!("ball on the carrier: probe depth {:.6}", c.depth.to_f64());

    // A wall (local half-space y < 0) on a turned carrier.
    let mut bodies = vec![RigidBody::new_dynamic(v3(1.0, 2.0, 3.0), Fix128::ONE)];
    bodies[0].rotation = QuatFix::from_axis_angle(v3(0.0, 0.0, 1.0), Fix128::HALF_PI);
    let mut wall = SdfCollider::new_dynamic(
        Box::new(ClosureSdf::new(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0))),
        0,
    );
    assert!(wall.sync_to_body(&bodies));
    let c = collide_point_sdf(v3(1.5, -7.0, 3.0), &wall).expect("inside the turned wall");
    assert!(
        near(c.depth.to_f64(), 0.5, 1e-5),
        "depth {}",
        c.depth.to_f64()
    );
    let (nx, ny, _) = c.normal.to_f32();
    assert!(near(f64::from(nx), -1.0, 1e-5) && near(f64::from(ny), 0.0, 1e-5));
    assert!(collide_point_sdf(v3(0.5, -7.0, 3.0), &wall).is_none());
    println!("turned wall: depth {:.6} normal -x", c.depth.to_f64());

    // Placing by hand: back to the world plane y = 0 (unturned).
    wall.set_pose(v3(0.0, 0.0, 0.0), QuatFix::IDENTITY);
    let c = collide_point_sdf(v3(0.0, -0.25, 0.0), &wall).expect("below the plane");
    assert!(
        near(c.depth.to_f64(), 0.25, 1e-6),
        "depth {}",
        c.depth.to_f64()
    );
    println!("placed wall: depth {:.6}", c.depth.to_f64());

    // A scale written directly takes effect after `update_cache`: the ball
    // scaled by 2 has radius 2, so the probe 1.25 above its centre
    // penetrates by 0.5 + (2 - 1.25) = 1.25.
    world.sdf_colliders[0].scale = Fix128::from_int(2);
    world.sdf_colliders[0].update_cache();
    let contacts = world.sdf_contacts();
    assert_eq!(contacts.len(), 1);
    let depth = contacts[0].1.depth.to_f64();
    assert!(near(depth, 1.25, 1e-6), "depth {depth}");
    println!("ball scaled by 2: probe depth {depth:.6}");
}
