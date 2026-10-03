//! Ragdoll animation blend state machine — keyframed walk clip crossfaded
//! into physics ragdoll and back, then driven through the motor-targets
//! ("Powered") mode.
//!
//! This is the production entry point for the `animation_blend` items that
//! otherwise have no caller outside their own unit tests
//! (`add_keyframe` / `bone_count` / `get_motor_targets` / `go_animated` /
//! `go_powered` / `go_ragdoll` / `is_transitioning` / `lerp` /
//! `set_animated` / `set_ragdoll` / `update`): every one of them is called
//! below, and each step prints the closed-form value next to what the
//! blender actually produced.
//!
//! ```bash
//! cargo run --example animation_blend_state_machine --features std
//! ```

use alice_physics::animation_blend::{
    AnimationBlender, AnimationClip, BlendMode, Keyframe, SkeletonPose,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

const TAG: &str = "[animation_blend]";

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn print_pose(label: &str, pose: &SkeletonPose) {
    for (i, b) in pose.bones.iter().enumerate() {
        println!(
            "{TAG} {label} bone {i}: pos = ({:.3}, {:.3}, {:.3})",
            b.position.x.to_f32(),
            b.position.y.to_f32(),
            b.position.z.to_f32(),
        );
    }
}

fn main() {
    // --- A small 2-bone skeleton driven by a 2-second walk clip --------
    let num_bones = 2;
    let duration = Fix128::from_int(2);
    let mut clip = AnimationClip::new(num_bones, duration);
    clip.looping = false;

    // bone 0: a straight slide from x=0 to x=8 (ALLOW-UNWIRED item:
    // `add_keyframe` is the only way production code populates a clip).
    clip.add_keyframe(
        0,
        Keyframe {
            time: Fix128::ZERO,
            pose: alice_physics::animation_blend::BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            },
        },
    );
    clip.add_keyframe(
        0,
        Keyframe {
            time: duration,
            pose: alice_physics::animation_blend::BonePose {
                position: Vec3Fix::from_int(8, 0, 0),
                rotation: QuatFix::IDENTITY,
            },
        },
    );
    // bone 1: a small rotation only, no translation.
    clip.add_keyframe(
        1,
        Keyframe {
            time: Fix128::ZERO,
            pose: alice_physics::animation_blend::BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            },
        },
    );
    clip.add_keyframe(
        1,
        Keyframe {
            time: duration,
            pose: alice_physics::animation_blend::BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::new(
                    q(1, 1024),
                    -q(1, 1024),
                    q(1, 1024),
                    Fix128::ONE - q(1, 1024),
                ),
            },
        },
    );

    println!(
        "{TAG} clip: {} bones, duration = {:.1}s",
        clip.num_bones(),
        clip.duration.to_f32()
    );

    // Sample the clip halfway (t=1s of 2s) — closed form: bone 0 is exactly
    // x = 4 (midpoint of a linear 0 -> 8 ramp).
    let animation_pose = clip.sample(Fix128::ONE);
    println!(
        "{TAG} clip.sample(1.0) bone 0 x = {:.3} (closed form: 4.000)",
        animation_pose.bones[0].position.x.to_f32()
    );

    // --- bone_count: a plain size accessor, exercised on every pose kind
    println!(
        "{TAG} bone_count: animation_pose = {}, clip-declared = {}",
        animation_pose.bone_count(),
        clip.num_bones()
    );

    // A stand-in ragdoll (physics) pose: bone 0 settles further along x,
    // bone 1 unrotated (identity).
    let physics_pose = SkeletonPose {
        bones: vec![
            alice_physics::animation_blend::BonePose {
                position: Vec3Fix::from_int(12, 0, 0),
                rotation: QuatFix::IDENTITY,
            },
            alice_physics::animation_blend::BonePose::default(),
        ],
    };

    // --- lerp: the blend primitive, called directly (closed form: the
    // t=1/2 blend of bone 0's x is (4+12)/2 = 8).
    let half = SkeletonPose::lerp(&animation_pose, &physics_pose, q(1, 2));
    println!(
        "{TAG} SkeletonPose::lerp(anim, physics, 1/2) bone 0 x = {:.3} (closed form: 8.000)",
        half.bones[0].position.x.to_f32()
    );

    // --- Drive the state machine --------------------------------------
    let mut blender = AnimationBlender::new(num_bones);
    blender.animation_pose = animation_pose;
    blender.physics_pose = physics_pose;
    blender.transition_speed = Fix128::ONE;
    assert_eq!(blender.mode, BlendMode::Animated);
    println!(
        "{TAG} start: mode = {:?}, is_transitioning = {}",
        blender.mode,
        blender.is_transitioning()
    );

    // go_powered: physics-driven with animation-sourced motor targets.
    blender.go_powered();
    println!(
        "{TAG} go_powered(): mode = {:?}, is_transitioning = {}",
        blender.mode,
        blender.is_transitioning()
    );

    let dt = Fix128::from_ratio(1, 4);
    for step in 0..4 {
        blender.update(dt);
        println!(
            "{TAG} powered update {step}: blend_weight = {:.3}, is_transitioning = {}, output bone0 x = {:.3} (= physics x), motor bone0 x = {:.3} (= animation x)",
            blender.blend_weight.to_f32(),
            blender.is_transitioning(),
            blender.output_pose.bones[0].position.x.to_f32(),
            blender.get_motor_targets().bones[0].position.x.to_f32(),
        );
    }
    // Powered never auto-switches to Ragdoll even once blend_weight == 1.
    assert_eq!(blender.mode, BlendMode::Powered);

    // set_animated: immediate cut, no crossfade — boundary α=0 exactly.
    blender.set_animated();
    println!(
        "{TAG} set_animated(): mode = {:?}, blend_weight = {:.3}, is_transitioning = {}",
        blender.mode,
        blender.blend_weight.to_f32(),
        blender.is_transitioning()
    );

    // go_ragdoll: gradual crossfade toward the physics pose.
    blender.go_ragdoll();
    println!(
        "{TAG} go_ragdoll(): mode = {:?}, is_transitioning = {}",
        blender.mode,
        blender.is_transitioning()
    );
    for step in 0..4 {
        blender.update(dt);
        println!(
            "{TAG} ragdoll crossfade update {step}: blend_weight = {:.3}, mode = {:?}, output bone0 x = {:.3}",
            blender.blend_weight.to_f32(),
            blender.mode,
            blender.output_pose.bones[0].position.x.to_f32(),
        );
    }
    // transition_speed=1, dt=1/4 => step 1/4 per update: lands exactly on
    // the boundary after 4 updates and auto-switches Blend -> Ragdoll.
    assert_eq!(blender.mode, BlendMode::Ragdoll);
    assert!(!blender.is_transitioning());

    // go_animated: gradual crossfade back the other way.
    blender.go_animated();
    println!(
        "{TAG} go_animated(): mode = {:?}, is_transitioning = {}",
        blender.mode,
        blender.is_transitioning()
    );
    for step in 0..4 {
        blender.update(dt);
        println!(
            "{TAG} animated crossfade update {step}: blend_weight = {:.3}, mode = {:?}, output bone0 x = {:.3}",
            blender.blend_weight.to_f32(),
            blender.mode,
            blender.output_pose.bones[0].position.x.to_f32(),
        );
    }
    assert_eq!(blender.mode, BlendMode::Animated);
    assert!(!blender.is_transitioning());

    // set_ragdoll: immediate cut the other way — boundary α=1 exactly.
    // Note: `set_ragdoll` only snaps `mode` / `blend_weight` / `target_weight`;
    // `output_pose` is only ever (re)computed inside `update`, so it still
    // holds the previous (Animated) value until the next `update` call.
    blender.set_ragdoll();
    println!(
        "{TAG} set_ragdoll(): mode = {:?}, blend_weight = {:.3}, is_transitioning = {}",
        blender.mode,
        blender.blend_weight.to_f32(),
        blender.is_transitioning()
    );
    blender.update(dt);

    print_pose("final output_pose", &blender.output_pose);
    println!("{TAG} done — all 11 previously-unwired animation_blend items exercised above.");
}
