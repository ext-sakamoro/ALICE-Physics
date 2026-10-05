//! Joint motors in the world — production entry point for
//! `PhysicsWorld::add_joint_motor` / `add_joint_motor_3d` and their setters.
//!
//! A hinge motor spins a wheel up to a target rate, is retargeted and then
//! switched off; a ball-joint rotation motor turns a second body to a target
//! orientation. `PhysicsWorld::step` applies both every substep. The closed
//! forms are pinned in `tests/analytic_joint_motor_world.rs`.
//!
//! ```bash
//! cargo run --example joint_motor_world
//! ```

use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::motor::PdController;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

fn main() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let dt = Fix128::from_ratio(1, 60);

    // wheel on a hinge (axis z), held by a static frame
    let frame = world.add_body(RigidBody::new_static(Vec3Fix::from_int(0, 10, 0)));
    let wheel = world.add_body(RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE));
    let hinge = world.add_joint(Joint::Hinge(HingeJoint::new(
        frame,
        wheel,
        Vec3Fix::from_int(0, -10, 0),
        Vec3Fix::ZERO,
        Vec3Fix::from_int(0, 0, 1),
        Vec3Fix::from_int(0, 0, 1),
    )));
    let motor = world.add_joint_motor(
        hinge,
        PdController::new(Fix128::from_int(2), Fix128::ZERO, Fix128::from_int(5)),
    );
    world.set_joint_motor_velocity_target(motor, Fix128::from_int(3));
    for _ in 0..240 {
        world.step(dt);
    }
    let spun = world.bodies[wheel].angular_velocity.z;
    println!(
        "wheel rate after 4 s at target 3 rad/s: {:.4}",
        spun.to_f64()
    );
    assert!(spun > Fix128::from_int(2));

    // stiffer gain, then off: the wheel coasts down under the frame damping
    if let Some(m) = world.joint_motor_mut(motor) {
        m.controller.kp = Fix128::from_int(4);
    }
    world.disable_joint_motor(motor);
    for _ in 0..60 {
        world.step(dt);
    }
    let coasting = world.bodies[wheel].angular_velocity.z;
    println!("wheel rate 1 s after disable: {:.4}", coasting.to_f64());
    assert!(coasting < spun);

    // a ball-jointed body turned to 0.5 rad about y by a 3-axis motor
    let anchor = world.add_body(RigidBody::new_static(Vec3Fix::from_int(20, 10, 0)));
    let head = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(20, 0, 0),
        Fix128::ONE,
    ));
    let ball = world.add_joint(Joint::Ball(BallJoint::new(
        anchor,
        head,
        Vec3Fix::from_int(0, -10, 0),
        Vec3Fix::ZERO,
    )));
    let gains = Vec3Fix::from_int(4, 4, 4);
    let turn = world.add_joint_motor_3d(ball, gains, gains, Fix128::from_int(100));
    if let Some(c) = world.joint_motor_3d_mut(turn) {
        c.kd = Vec3Fix::new(
            Fix128::from_ratio(8, 5),
            Fix128::from_ratio(8, 5),
            Fix128::from_ratio(8, 5),
        );
    }
    let target = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 1, 0), Fix128::from_ratio(1, 2));
    world.set_joint_motor_3d_rotation_target(turn, target);
    for _ in 0..300 {
        world.step(dt);
    }
    let q = world.bodies[head].rotation;
    let angle = Fix128::atan2(q.y, q.w).double().to_f64();
    println!("head angle about y after 5 s (target 0.5 rad): {angle:.4}");
    assert!((angle - 0.5).abs() < 0.01);
}
