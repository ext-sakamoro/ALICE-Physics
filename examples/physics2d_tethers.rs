//! Returning bodies to a rest pose in 2D with `Tethers2D`.
//!
//! - A dynamic tile is pulled back to its rest position by a `Joint2D::Mouse` and to its
//!   rest angle by an `AngularTether2D` (both critically damped).
//! - Two kinematic tiles are flung away and returned by `KinematicDrive2D`; being
//!   kinematic they pass through each other on the way back.
//!
//! Run: `cargo run --example physics2d_tethers --features std`

use alice_physics::math::Fix128;
use alice_physics::physics2d::{
    AngularTether2D, Joint2D, KinematicDrive2D, PhysicsConfig2D, PhysicsWorld2D, RigidBody2D,
    Shape2D, Tethers2D, Vec2Fix,
};

fn main() {
    let mut world = PhysicsWorld2D::new(PhysicsConfig2D {
        gravity: Vec2Fix::ZERO,
        substeps: 4,
        iterations: 4,
        damping: Fix128::ONE,
    });
    let tile = Shape2D::Circle {
        radius: Fix128::from_ratio(1, 4),
    };
    let omega = Fix128::from_int(8);

    // dynamic tile, displaced and spun away from its rest pose (0, 0), angle 0
    let mut body = RigidBody2D::new_dynamic(Vec2Fix::from_int(2, 1), Fix128::ONE, tile.clone());
    body.angle = Fix128::from_int(2);
    body.angular_velocity = Fix128::from_int(6);
    let inertia = Fix128::ONE / body.inv_inertia;
    let dynamic = world.add_body(body);
    world.add_joint(Joint2D::Mouse {
        body: dynamic,
        target: Vec2Fix::ZERO,
        max_force: Fix128::from_int(1_000),
        stiffness: omega * omega,
        damping: Fix128::from_int(2) * omega,
    });

    // two kinematic tiles that swap sides on the way back to their rest poses
    let left = world.add_body(RigidBody2D::new_kinematic(
        Vec2Fix::from_int(3, 4),
        tile.clone(),
    ));
    let right = world.add_body(RigidBody2D::new_kinematic(Vec2Fix::from_int(-3, 4), tile));

    let mut tethers = Tethers2D::new();
    let spin = tethers.add_angular(AngularTether2D::critically_damped(
        dynamic,
        Fix128::ZERO,
        inertia,
        omega,
    ));
    let drive_left = tethers.add_drive(KinematicDrive2D::new(
        left,
        Vec2Fix::from_int(-3, 4),
        Fix128::ZERO,
        omega,
    ));
    let drive_right = tethers.add_drive(KinematicDrive2D::new(
        right,
        Vec2Fix::from_int(3, 4),
        Fix128::ZERO,
        omega,
    ));
    println!(
        "[physics2d_tethers] {} angular tether(s), {} kinematic drive(s)",
        tethers.angular_count(),
        tethers.drive_count()
    );

    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..=120 {
        if frame % 20 == 0 {
            let d = &world.bodies[dynamic];
            println!(
                "[physics2d_tethers] t={:.2}s tile=({:+.4},{:+.4}) angle={:+.4} | left x={:+.4} right x={:+.4}",
                f64::from(frame) / 60.0,
                d.position.x.to_f64(),
                d.position.y.to_f64(),
                d.angle.to_f64(),
                world.bodies[left].position.x.to_f64(),
                world.bodies[right].position.x.to_f64(),
            );
        }
        world.step_with_tethers(dt, &tethers);
    }

    // re-aim: tilt the tile's rest angle and send the kinematic tiles home again
    if let Some(t) = tethers.angular_mut(spin) {
        t.target_angle = Fix128::from_ratio(1, 2);
    }
    if let Some(d) = tethers.drive_mut(drive_left) {
        d.target_position = Vec2Fix::from_int(3, 4);
    }
    for _ in 0..120 {
        world.step_with_tethers(dt, &tethers);
    }
    println!(
        "[physics2d_tethers] re-aimed: angle={:+.4} (target {:+.4}), left x={:+.4} (target {:+.4})",
        world.bodies[dynamic].angle.to_f64(),
        tethers
            .angular(spin)
            .map_or(0.0, |t| t.target_angle.to_f64()),
        world.bodies[left].position.x.to_f64(),
        tethers
            .drive(drive_left)
            .map_or(0.0, |d| d.target_position.x.to_f64()),
    );

    // releasing a drive leaves the body coasting with the last velocity it was given
    tethers.remove_drive(drive_right);
    tethers.remove_angular(spin);
    println!(
        "[physics2d_tethers] released: {} angular tether(s), {} kinematic drive(s)",
        tethers.angular_count(),
        tethers.drive_count()
    );
}
