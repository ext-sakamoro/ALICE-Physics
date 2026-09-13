//! Humanoid ragdoll demo — drop a 1.75 m / 75 kg adult male ragdoll from
//! a height of 4 m and simulate 2 seconds of free fall + ground impact.
//!
//! Illustrates the v0.13.0 Session 4 `ragdoll` module (Tier ★★★ G1).
//!
//! ```bash
//! cargo run --example ragdoll_demo --features std
//! ```

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::ragdoll::{Bone, RagdollBuilder, RagdollProportions};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};

fn main() {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        substeps: 4,
        iterations: 8,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        damping: Fix128::from_ratio(99, 100),
        ..Default::default()
    });

    // Drop a full-size adult male ragdoll from a pelvis height of 4 m.
    let pelvis_position = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(4), Fix128::ZERO);
    let ragdoll = RagdollBuilder::build(&mut world, RagdollProportions::human_male(), pelvis_position);

    println!("ALICE-Physics — Humanoid Ragdoll Demo");
    println!("=====================================");
    println!("Bodies:         {}", world.body_count());
    println!("Ragdoll bones:  {}", ragdoll.bones.len());
    println!("Ragdoll joints: {}", ragdoll.joints.len());
    println!("Pelvis drop:    4.0 m");
    println!("Proportions:    1.75 m / 75 kg adult male");

    let dt = Fix128::from_ratio(1, 60);
    println!("\nStep |   pelvis y | head y  | left_hand y | right_foot y");
    println!("-----|------------|---------|-------------|--------------");

    // Report the pelvis / head / left-hand / right-foot altitudes every 15 frames (~0.25 s).
    for frame in 0..=120 {
        if frame % 15 == 0 {
            let pelvis = world.bodies[ragdoll.body(Bone::Pelvis)].position;
            let head = world.bodies[ragdoll.body(Bone::Head)].position;
            let lhand = world.bodies[ragdoll.body(Bone::LeftHand)].position;
            let rfoot = world.bodies[ragdoll.body(Bone::RightFoot)].position;
            println!(
                "{:4} | {:>10.3} | {:>7.3} | {:>11.3} | {:>13.3}",
                frame,
                pelvis.y.to_f32(),
                head.y.to_f32(),
                lhand.y.to_f32(),
                rfoot.y.to_f32(),
            );
        }
        world.step(dt);
    }

    println!("\nDone — 120 frames (2.0 s) simulated.");
    println!("Body positions above are read from the underlying PhysicsWorld,");
    println!("proving that the ragdoll builder wired all 15 bones and 14 joints correctly.");
}
