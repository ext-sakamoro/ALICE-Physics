//! Ragdoll presets and IK position targets.
//!
//! Builds the female and child presets, prints each preset's pelvis-to-head
//! span (0.94 x height, hand-derived from the segment fractions) and steers the
//! right hand with `IkTargetSet`: a half-weight target covers half the
//! residual per pass (8 -> 4 -> 2 ... geometric), a full-weight one snaps.
//!
//! ```bash
//! cargo run --release --example ragdoll_ik_targets --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::ik_physics_bridge::{IkTarget, IkTargetSet};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::ragdoll::{Bone, RagdollBuilder, RagdollProportions};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};

fn main() {
    for (name, p, height) in [
        ("female", RagdollProportions::human_female(), 1.62_f64),
        ("child", RagdollProportions::child(), 1.30),
    ] {
        let mut world = PhysicsWorld::new(PhysicsConfig::default());
        let rag = RagdollBuilder::build(&mut world, p, Vec3Fix::ZERO);
        let head = world.bodies[rag.body(Bone::Head)].position.y.to_f64();
        let foot = world.bodies[rag.body(Bone::LeftFoot)].position.y.to_f64();
        println!(
            "[ragdoll] {name}: head - foot = {:.4} m (closed form {:.4}), {} bones, {} joints",
            head - foot,
            0.94 * height,
            rag.bones.len(),
            rag.joints.len()
        );
        assert!((head - foot - 0.94 * height).abs() < 1e-6);
    }

    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let rag = RagdollBuilder::build(
        &mut world,
        RagdollProportions::child(),
        Vec3Fix::from_int(0, 2, 0),
    );
    let hand = rag.body(Bone::RightHand);
    let start = world.bodies[hand].position.x.to_f64();
    let goal_x = start + 8.0;
    let goal = Vec3Fix::new(
        Fix128::from_f64(goal_x),
        world.bodies[hand].position.y,
        world.bodies[hand].position.z,
    );
    let mut ik = IkTargetSet::new();
    ik.push(IkTarget::blended(hand, goal));
    for pass in 1..=3 {
        ik.apply(&mut world);
        let x = world.bodies[hand].position.x.to_f64();
        let want = goal_x - 8.0 * 0.5_f64.powi(pass);
        println!("[ik] blended pass {pass}: x = {x:.6} (closed form {want:.6})");
        assert!((x - want).abs() < 1e-6);
    }
    let mut snap = IkTargetSet::new();
    snap.push(IkTarget::snap(hand, goal));
    snap.apply(&mut world);
    println!(
        "[ik] snap: hand at goal = {}",
        world.bodies[hand].position == goal
    );
    assert_eq!(world.bodies[hand].position, goal);
}
