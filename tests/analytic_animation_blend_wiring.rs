//! Oracles for the wiring of `animation_blend`'s state-machine surface:
//! `add_keyframe`, `bone_count`, `get_motor_targets`, `go_animated`,
//! `go_powered`, `go_ragdoll`, `is_transitioning`, `lerp`, `set_animated`,
//! `set_ragdoll`, `update`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * **`update`'s smooth-weight step** adds `transition_speed` times `dt`
//!   to `blend_weight` when `target_weight` is greater, or subtracts it
//!   when `target_weight` is smaller, clamped so it never crosses
//!   `target_weight` in the direction of travel — there is no clamp
//!   against overshooting in the other direction, which is what the
//!   negative-`dt` case below exploits. Snapping replaces this arithmetic
//!   once `|target_weight - blend_weight| <= 1/1000`.
//! * **`is_transitioning`** is exactly `|blend_weight - target_weight| >
//!   1/1000`, independent of `mode`.
//! * **Mode auto-switch** (`Blend` -> `Ragdoll`/`Animated`) fires only when
//!   `mode == Blend` and `blend_weight` crosses `>= 1` or `<= 0`; `Powered`
//!   never auto-switches.
//! * **`SkeletonPose::lerp(a, b, t)`** is `a*(1-t) + b*t` per bone,
//!   component-wise for position and NLERP/SLERP for rotation; it does not
//!   clamp `t` to `[0, 1]` — extrapolation at `t=2`: `a=(3,5,7)`,
//!   `b=(11,13,15)` gives `-a + 2b = (19, 21, 23)`; at `t=-1` gives
//!   `2a - b = (-5, -3, -1)`.
//! * **NLERP rotation blend** (the `dot > 999/1000` fast path): with
//!   `a = IDENTITY = (0,0,0,1)` and `b = (d, -d, d, 1-d)` where
//!   `d = 1/1024` (dyadic, exact in `Fix128`), `dot = a·b = 1 - d =
//!   1023/1024 > 999/1000`, so the NLERP path is used. At `t = 1/4` the
//!   un-normalised blend is `((1-t)*0+t*d, -t*d, t*d, (1-t)+t*(1-d)) =
//!   (d/4, -d/4, d/4, 1 - d/4)`, normalised by dividing by its Euclidean
//!   norm (computed independently in `f64` below, not by calling the
//!   crate's `quat_slerp`).
//! * **`get_motor_targets`** always returns the current `animation_pose`
//!   verbatim — never blended, never lagged — regardless of `mode` or
//!   `blend_weight`.
//! * **`output_pose`** is only (re)computed inside `update`; `set_ragdoll` /
//!   `set_animated` / `go_*` change `mode` / `blend_weight` / the (private)
//!   target but do not touch `output_pose` until the next `update` call.
//!
//! # Degenerate inputs (documented result, not "no panic")
//!
//! * **Zero bones**: every item above operates on empty `Vec<BonePose>`;
//!   `bone_count() == 0` throughout and no panic (checked via
//!   `catch_unwind`).
//! * **Re-triggering a transition that is already in progress overwrites
//!   the target** (restart semantics) — it is neither queued nor ignored:
//!   §`retrigger_mid_transition_overwrites_target_not_queue_or_ignore`
//!   proves this by driving `blend_weight` to a value that is only
//!   reachable if the second call replaced the first call's target.
//! * **Negative `dt`** is not guarded: the smooth-step branch is selected by
//!   the *sign of `target_weight - blend_weight`* before `dt` is applied, so
//!   a negative `dt` moves `blend_weight` *away* from `target_weight`,
//!   including below `0` with no lower clamp. This can flip `mode` to
//!   `Animated` (via the `blend_weight <= 0` auto-switch) while
//!   `is_transitioning()` still reports `true`, because `is_transitioning`
//!   compares against `target_weight`, not `0`.
//! * **Zero `dt`** is a true no-op only while `|diff| > 1/1000`; once within
//!   that band `update` snaps `blend_weight` to `target_weight` regardless
//!   of `dt` (existing in-module behaviour, re-asserted here through the
//!   public surface only).
//! * **Extreme transition_speed / huge step** overshoots are clamped to
//!   land exactly on `target_weight` (no wraparound), confirmed via
//!   `catch_unwind` with `transition_speed = Fix128::from_raw(i64::MAX,
//!   u64::MAX)`.
//! * **Out-of-order `add_keyframe` calls** still produce a time-sorted
//!   buffer (existing invariant), re-checked here via `sample` through the
//!   public surface with keyframes inserted scrambled.

#![cfg(feature = "std")]

use alice_physics::animation_blend::{
    AnimationBlender, AnimationClip, BlendMode, BonePose, Keyframe, SkeletonPose,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use std::panic::catch_unwind;

fn r(num: i64, denom: i64) -> Fix128 {
    Fix128::from_ratio(num, denom)
}

fn pos_pose(n: usize, x: i64) -> SkeletonPose {
    SkeletonPose {
        bones: (0..n)
            .map(|i| BonePose {
                position: Vec3Fix::from_int(x, i as i64, 0),
                rotation: QuatFix::IDENTITY,
            })
            .collect(),
    }
}

fn kf(time: Fix128, x: i64) -> Keyframe {
    Keyframe {
        time,
        pose: BonePose {
            position: Vec3Fix::from_int(x, 0, 0),
            rotation: QuatFix::IDENTITY,
        },
    }
}

// ---------------------------------------------------------------------
// boundary α = 0 / 1: both the immediate (`set_*`) and gradual (`go_*` +
// `update`-to-completion) paths land exactly on the source/target pose.
// ---------------------------------------------------------------------

#[test]
fn immediate_set_ragdoll_and_set_animated_are_exact_boundaries() {
    let mut b = AnimationBlender::new(2);
    b.animation_pose = pos_pose(2, 1);
    b.physics_pose = pos_pose(2, 9);

    b.set_ragdoll();
    assert_eq!(b.mode, BlendMode::Ragdoll);
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert!(!b.is_transitioning());
    b.update(r(1, 4)); // output_pose is only refreshed inside `update`
    assert_eq!(b.output_pose.bones, b.physics_pose.bones, "α=1 boundary");

    b.set_animated();
    assert_eq!(b.mode, BlendMode::Animated);
    assert_eq!(b.blend_weight, Fix128::ZERO);
    assert!(!b.is_transitioning());
    b.update(r(1, 4));
    assert_eq!(b.output_pose.bones, b.animation_pose.bones, "α=0 boundary");
}

#[test]
fn gradual_crossfade_lands_exactly_on_boundary_and_auto_switches_mode() {
    let mut b = AnimationBlender::new(2);
    b.animation_pose = pos_pose(2, 0);
    b.physics_pose = pos_pose(2, 8);
    b.transition_speed = Fix128::ONE;
    let dt = r(1, 4); // step = 1 * 1/4 = 1/4 per update

    b.go_ragdoll();
    assert_eq!(b.mode, BlendMode::Blend);
    assert_eq!(
        b.blend_weight,
        Fix128::ZERO,
        "go_* never moves weight immediately"
    );
    assert!(b.is_transitioning());

    for (step_idx, want_weight) in [(0, r(1, 4)), (1, r(1, 2)), (2, r(3, 4))] {
        b.update(dt);
        assert_eq!(b.blend_weight, want_weight, "step {step_idx}");
        assert_eq!(b.mode, BlendMode::Blend, "step {step_idx} not yet complete");
        assert!(b.is_transitioning(), "step {step_idx}");
    }
    b.update(dt); // 4th step: 3/4 + 1/4 = 1 exactly, no overshoot-clamp needed
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert_eq!(b.mode, BlendMode::Ragdoll, "α=1 auto-switch");
    assert!(!b.is_transitioning());
    assert_eq!(b.output_pose.bones, b.physics_pose.bones);
}

// ---------------------------------------------------------------------
// mid-crossfade lerp: exact per-bone position (linear) and rotation
// (NLERP closed form, hand-derived in f64, not via `quat_slerp`).
// ---------------------------------------------------------------------

#[test]
fn mid_crossfade_matches_hand_derived_linear_and_nlerp_blend() {
    let d = r(1, 1024);
    let q_b = QuatFix::new(d, -d, d, Fix128::ONE - d);
    // dot(IDENTITY, q_b) = 1 - d = 1023/1024 > 999/1000: NLERP path.
    let dot = QuatFix::IDENTITY.x * q_b.x
        + QuatFix::IDENTITY.y * q_b.y
        + QuatFix::IDENTITY.z * q_b.z
        + QuatFix::IDENTITY.w * q_b.w;
    assert!(dot > r(999, 1000), "dot = {}", dot.to_f64());

    let mut b = AnimationBlender::new(2);
    b.animation_pose = SkeletonPose {
        bones: vec![
            BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            },
            BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            },
        ],
    };
    b.physics_pose = SkeletonPose {
        bones: vec![
            BonePose {
                position: Vec3Fix::from_int(8, 0, 0),
                rotation: QuatFix::IDENTITY,
            },
            BonePose {
                position: Vec3Fix::ZERO,
                rotation: q_b,
            },
        ],
    };
    b.transition_speed = Fix128::ONE;
    b.go_ragdoll();
    b.update(r(1, 4)); // exactly t = 1/4

    assert_eq!(b.blend_weight, r(1, 4));
    // bone 0: pure linear position blend, closed form (1-1/4)*0 + 1/4*8 = 2.
    assert_eq!(b.output_pose.bones[0].position, Vec3Fix::from_int(2, 0, 0));
    assert_eq!(b.output_pose.bones[0].rotation, QuatFix::IDENTITY);

    // bone 1: NLERP closed form, un-normalised then normalised in f64.
    let raw = [
        0.25 * (d.to_f64()),
        -0.25 * (d.to_f64()),
        0.25 * (d.to_f64()),
        1.0 - 0.25 * (d.to_f64()),
    ];
    let norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2] + raw[3] * raw[3]).sqrt();
    let want = [raw[0] / norm, raw[1] / norm, raw[2] / norm, raw[3] / norm];
    let got = b.output_pose.bones[1].rotation;
    let got_f64 = [
        got.x.to_f64(),
        got.y.to_f64(),
        got.z.to_f64(),
        got.w.to_f64(),
    ];
    for i in 0..4 {
        assert!(
            (got_f64[i] - want[i]).abs() < 1e-12,
            "component {i}: got {} want {}",
            got_f64[i],
            want[i]
        );
    }
    assert_eq!(b.output_pose.bones[1].position, Vec3Fix::ZERO);
}

// ---------------------------------------------------------------------
// `lerp` called directly: no clamp on `t` (extrapolation).
// ---------------------------------------------------------------------

#[test]
fn lerp_extrapolates_outside_zero_one_with_no_clamp() {
    let a = SkeletonPose {
        bones: vec![BonePose {
            position: Vec3Fix::from_int(3, 5, 7),
            rotation: QuatFix::IDENTITY,
        }],
    };
    let b = SkeletonPose {
        bones: vec![BonePose {
            position: Vec3Fix::from_int(11, 13, 15),
            rotation: QuatFix::IDENTITY,
        }],
    };

    let at_2 = catch_unwind(|| SkeletonPose::lerp(&a, &b, Fix128::from_int(2)));
    assert!(at_2.is_ok(), "t=2 should not panic");
    assert_eq!(
        at_2.unwrap().bones[0].position,
        Vec3Fix::from_int(19, 21, 23),
        "-a + 2b"
    );

    let at_neg1 = catch_unwind(|| SkeletonPose::lerp(&a, &b, Fix128::NEG_ONE));
    assert!(at_neg1.is_ok(), "t=-1 should not panic");
    assert_eq!(
        at_neg1.unwrap().bones[0].position,
        Vec3Fix::from_int(-5, -3, -1),
        "2a - b"
    );
}

// ---------------------------------------------------------------------
// `get_motor_targets`: always the live `animation_pose`, in both Powered
// and mid-crossfade-to-Ragdoll states — never the blended output.
// ---------------------------------------------------------------------

#[test]
fn motor_targets_always_mirror_live_animation_pose_never_blended() {
    let mut b = AnimationBlender::new(1);
    b.animation_pose = pos_pose(1, 1);
    b.physics_pose = pos_pose(1, 9);
    b.transition_speed = Fix128::ONE;
    let dt = r(1, 2);

    b.go_powered();
    assert_eq!(b.mode, BlendMode::Powered);
    assert!(b.is_transitioning());
    assert_eq!(b.get_motor_targets().bones, pos_pose(1, 1).bones);

    b.update(dt); // blend_weight = 1/2, still mid-"transition" by weight
    assert_eq!(b.blend_weight, r(1, 2));
    assert!(
        b.is_transitioning(),
        "weight has not caught up to target yet"
    );
    assert_eq!(b.mode, BlendMode::Powered, "Powered never auto-switches");
    // Output is physics_pose regardless of blend_weight in Powered mode.
    assert_eq!(b.output_pose.bones, pos_pose(1, 9).bones);
    assert_eq!(
        b.get_motor_targets().bones,
        pos_pose(1, 1).bones,
        "unblended"
    );

    b.update(dt); // blend_weight = 1 exactly
    assert!(!b.is_transitioning());
    assert_eq!(b.mode, BlendMode::Powered, "still Powered, not Ragdoll");
    assert_eq!(b.output_pose.bones, pos_pose(1, 9).bones);

    // Feeding fresh animation data is reflected immediately, with no lag
    // and no blending, even though mode/weight are unchanged.
    b.animation_pose = pos_pose(1, 5);
    assert_eq!(b.get_motor_targets().bones, pos_pose(1, 5).bones);
    assert_eq!(
        b.output_pose.bones,
        pos_pose(1, 9).bones,
        "output still physics"
    );
}

// ---------------------------------------------------------------------
// retrigger mid-transition: overwrite semantics (not queue, not ignore).
// ---------------------------------------------------------------------

#[test]
fn retrigger_mid_transition_overwrites_target_not_queue_or_ignore() {
    let mut b = AnimationBlender::new(1);
    b.transition_speed = Fix128::ONE;
    let dt = r(1, 2);

    b.go_ragdoll(); // target = 1
    b.update(dt); // blend_weight = 1/2, still heading to 1
    assert_eq!(b.blend_weight, r(1, 2));
    assert_eq!(b.mode, BlendMode::Blend);

    // Retrigger towards Animated while still mid-crossfade.
    b.go_animated(); // target overwritten to 0; blend_weight untouched
    assert_eq!(
        b.blend_weight,
        r(1, 2),
        "go_* never snaps weight immediately"
    );
    assert!(b.is_transitioning(), "now heading the other way");

    b.update(dt); // if the retrigger were ignored this would be 1/2+1/2=1
    assert_eq!(
        b.blend_weight,
        Fix128::ZERO,
        "weight moved toward the NEW target (0), proving overwrite not ignore/queue"
    );
    assert_eq!(
        b.mode,
        BlendMode::Animated,
        "α=0 auto-switch after overwrite"
    );
    assert!(!b.is_transitioning());
}

// ---------------------------------------------------------------------
// negative dt: unguarded, moves weight away from target, no lower clamp.
// ---------------------------------------------------------------------

#[test]
fn negative_dt_moves_weight_away_from_target_with_no_lower_clamp() {
    let mut b = AnimationBlender::new(1);
    b.transition_speed = Fix128::ONE;

    b.go_ragdoll(); // target = 1
    b.update(r(1, 4)); // blend_weight = 1/4
    assert_eq!(b.blend_weight, r(1, 4));

    // Negative dt: diff = 1 - 1/4 = 3/4 > 0 selects the "add step" branch,
    // but step = speed * dt = 1 * (-1/2) = -1/2, so weight DECREASES.
    b.update(-r(1, 2));
    assert_eq!(
        b.blend_weight,
        -r(1, 4),
        "1/4 + (-1/2) = -1/4, unclamped below zero"
    );
    // the mode switches only when the weight reaches its target (AUD-A-S2W3-011),
    // so a weight pushed below 0 with the target still at 1 stays in Blend and
    // agrees with is_transitioning (it used to switch to Animated on weight <= 0)
    assert_eq!(
        b.mode,
        BlendMode::Blend,
        "no auto-switch before the target is reached"
    );
    assert!(
        b.is_transitioning(),
        "is_transitioning compares to target_weight (1), not 0: |-1/4 - 1| = 5/4 > 1/1000"
    );
}

#[test]
fn zero_dt_is_a_no_op_far_from_target_but_snaps_once_within_epsilon() {
    let mut b = AnimationBlender::new(1);
    b.transition_speed = Fix128::ONE;
    b.go_ragdoll();
    b.update(r(1, 4)); // blend_weight = 1/4, diff = 3/4 (> 1/1000)

    b.update(Fix128::ZERO);
    assert_eq!(
        b.blend_weight,
        r(1, 4),
        "dt=0 is a true no-op far from target"
    );

    // Push to just inside the snap epsilon, then dt=0 snaps exactly.
    b.blend_weight = Fix128::ONE - r(1, 2000); // diff = 1/2000 < 1/1000
    b.update(Fix128::ZERO);
    assert_eq!(
        b.blend_weight,
        Fix128::ONE,
        "within epsilon, update always snaps"
    );
}

// ---------------------------------------------------------------------
// extreme transition_speed: clamps exactly to target, no wraparound.
// ---------------------------------------------------------------------

#[test]
fn huge_transition_speed_clamps_exactly_to_target_without_panicking() {
    let mut b = AnimationBlender::new(1);
    b.transition_speed = Fix128::from_raw(i64::MAX, u64::MAX);
    b.go_ragdoll();

    let result = catch_unwind(move || {
        b.update(Fix128::ONE);
        (b.blend_weight, b.mode, b.is_transitioning())
    });
    let (weight, mode, transitioning) = result.expect("huge step must not panic");
    assert_eq!(
        weight,
        Fix128::ONE,
        "clamped exactly to target, no wraparound"
    );
    assert_eq!(mode, BlendMode::Ragdoll);
    assert!(!transitioning);
}

// ---------------------------------------------------------------------
// zero bones: every item is a no-op over an empty Vec, no panic.
// ---------------------------------------------------------------------

#[test]
fn zero_bones_every_item_is_a_no_op_without_panicking() {
    let result = catch_unwind(|| {
        let mut b = AnimationBlender::new(0);
        assert_eq!(b.animation_pose.bone_count(), 0);
        b.go_powered();
        b.update(r(1, 4));
        b.go_ragdoll();
        b.update(r(1, 4));
        b.go_animated();
        b.update(r(1, 4));
        b.set_ragdoll();
        b.set_animated();
        let _ = b.is_transitioning();
        let _ = b.get_motor_targets().bone_count();
        let mut clip = AnimationClip::new(0, Fix128::from_int(2));
        clip.add_keyframe(0, kf(Fix128::ZERO, 1)); // bone_idx >= num_bones: ignored
        assert_eq!(clip.sample(Fix128::ZERO).bone_count(), 0);
        (
            b.animation_pose.bone_count(),
            b.physics_pose.bone_count(),
            b.output_pose.bone_count(),
        )
    });
    let (a, p, o) = result.expect("zero bones must not panic anywhere in the state machine");
    assert_eq!((a, p, o), (0, 0, 0));
}

// ---------------------------------------------------------------------
// add_keyframe out of time order still produces a sorted buffer, checked
// through `sample` on the public surface only.
// ---------------------------------------------------------------------

#[test]
fn add_keyframe_out_of_order_still_samples_correctly() {
    let mut clip = AnimationClip::new(1, Fix128::from_int(4));
    clip.looping = false;
    // Scrambled insertion order: 4, 0, 2.
    clip.add_keyframe(0, kf(Fix128::from_int(4), 16));
    clip.add_keyframe(0, kf(Fix128::ZERO, 0));
    clip.add_keyframe(0, kf(Fix128::from_int(2), 8));

    // Closed form: piecewise-linear 0 -> 8 -> 16 over [0,2] and [2,4].
    assert_eq!(
        clip.sample(Fix128::ONE).bones[0].position,
        Vec3Fix::from_int(4, 0, 0)
    );
    assert_eq!(
        clip.sample(Fix128::from_int(3)).bones[0].position,
        Vec3Fix::from_int(12, 0, 0)
    );
    assert_eq!(
        clip.sample(Fix128::from_int(2)).bones[0].position,
        Vec3Fix::from_int(8, 0, 0)
    );
}
