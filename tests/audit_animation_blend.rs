//! Audit oracles for `alice_physics::animation_blend`.
//! Reference values: linear interpolation of keyframes, constant-angular-velocity SLERP about a
//! fixed axis (closed form: rotation by `t * arc`), loop/clamp time arithmetic, and the stated
//! transition speed (weight units per second).
#![allow(clippy::disallowed_methods)]

use alice_physics::animation_blend::{
    AnimationBlender, AnimationClip, BlendMode, BonePose, Keyframe, SkeletonPose,
};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// Unit quaternion for a rotation of `deg` about the (normalised) axis.
fn rot(axis: [f64; 3], deg: f64) -> QuatFix {
    let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let h = deg.to_radians() / 2.0;
    let s = h.sin() / l;
    QuatFix::new(
        fx(axis[0] * s),
        fx(axis[1] * s),
        fx(axis[2] * s),
        fx(h.cos()),
    )
}

fn q64(q: QuatFix) -> [f64; 4] {
    [q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64()]
}

/// Angle (deg) of the relative rotation conj(a) * b, accurate near zero.
fn angle_between(a: QuatFix, b: QuatFix) -> f64 {
    let (a, b) = (q64(a), q64(b));
    let (ax, ay, az, aw) = (-a[0], -a[1], -a[2], a[3]);
    let w = aw * b[3] - ax * b[0] - ay * b[1] - az * b[2];
    let x = aw * b[0] + ax * b[3] + ay * b[2] - az * b[1];
    let y = aw * b[1] - ax * b[2] + ay * b[3] + az * b[0];
    let z = aw * b[2] + ax * b[1] - ay * b[0] + az * b[3];
    (2.0 * (x * x + y * y + z * z).sqrt().atan2(w.abs())).to_degrees()
}

fn pose1(p: Vec3Fix, q: QuatFix) -> SkeletonPose {
    SkeletonPose {
        bones: vec![BonePose {
            position: p,
            rotation: q,
        }],
    }
}

/// `SkeletonPose::lerp` is documented as a pose lerp whose rotations are SLERPed: the angle swept
/// grows linearly with t (constant angular velocity), for arcs on both sides of the NLERP
/// threshold (dot > 0.999 is about 5.1 degrees) and for an arbitrary axis.
#[test]
fn pose_lerp_rotation_has_constant_angular_velocity() {
    for axis in [[0.0, 1.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]] {
        for arc in [4.0f64, 5.4, 30.0, 100.0, 170.0] {
            let a = pose1(Vec3Fix::ZERO, rot(axis, 0.0));
            let b = pose1(Vec3Fix::ZERO, rot(axis, arc));
            for k in 0..=8 {
                let t = f64::from(k) / 8.0;
                let q = SkeletonPose::lerp(&a, &b, fx(t)).bones[0].rotation;
                let want = rot(axis, arc * t);
                assert!(
                    angle_between(q, want) < 2e-4,
                    "axis {axis:?} arc {arc} t {t}: off by {} deg",
                    angle_between(q, want)
                );
                let n: f64 = q64(q).iter().map(|c| c * c).sum();
                assert!((n - 1.0).abs() < 1e-9, "|q|^2 = {n}");
            }
        }
    }
}

/// Antipodal representation of the same rotation gives that rotation for every t.
#[test]
fn pose_lerp_between_q_and_minus_q_stays_put() {
    let q = rot([0.0, 1.0, 0.0], 70.0);
    let nq = QuatFix::new(-q.x, -q.y, -q.z, -q.w);
    for k in 0..=4 {
        let t = fx(f64::from(k) / 4.0);
        let out = SkeletonPose::lerp(&pose1(Vec3Fix::ZERO, q), &pose1(Vec3Fix::ZERO, nq), t).bones
            [0]
        .rotation;
        assert!(angle_between(out, q) < 1e-4);
    }
}

/// Shortest path across the double cover: from 0 to 350 degrees goes the 10 degree way round.
#[test]
fn pose_lerp_takes_the_short_arc() {
    let a = pose1(Vec3Fix::ZERO, rot([0.0, 1.0, 0.0], 0.0));
    let b = pose1(Vec3Fix::ZERO, rot([0.0, 1.0, 0.0], 350.0));
    let mid = SkeletonPose::lerp(&a, &b, r(1, 2)).bones[0].rotation;
    assert!(angle_between(mid, rot([0.0, 1.0, 0.0], -5.0)) < 1e-4);
}

fn kf(t: f64, p: [f64; 3], q: QuatFix) -> Keyframe {
    Keyframe {
        time: fx(t),
        pose: BonePose {
            position: Vec3Fix::new(fx(p[0]), fx(p[1]), fx(p[2])),
            rotation: q,
        },
    }
}

fn two_bone_clip(looping: bool) -> AnimationClip {
    let mut c = AnimationClip::new(2, Fix128::from_int(4));
    c.looping = looping;
    // bone 0: non-uniform keys 0, 1, 4 (inserted out of order)
    c.add_keyframe(0, kf(4.0, [8.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 120.0)));
    c.add_keyframe(0, kf(0.0, [0.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 0.0)));
    c.add_keyframe(0, kf(1.0, [4.0, 2.0, -2.0], rot([0.0, 1.0, 0.0], 60.0)));
    // bone 1: single key, constant pose
    c.add_keyframe(1, kf(2.0, [-1.0, -2.0, -3.0], rot([1.0, 0.0, 0.0], 45.0)));
    c
}

fn p(pose: &SkeletonPose, bone: usize) -> [f64; 3] {
    let v = pose.bones[bone].position;
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// add_keyframe keeps per-bone time order, the offset table consistent for every bone, ignores an
/// out-of-range bone, and appends an equal-time key after the existing one.
#[test]
fn add_keyframe_maintains_order_and_offsets() {
    let mut c = two_bone_clip(true);
    assert_eq!(c.num_bones(), 2);
    assert_eq!(c.bone_offsets, vec![0, 3, 4]);
    let times: Vec<f64> = c.keyframes.iter().map(|k| k.time.to_f64()).collect();
    assert_eq!(times, vec![0.0, 1.0, 4.0, 2.0]);
    let before = c.clone().keyframes.len();
    c.add_keyframe(2, kf(0.0, [9.0; 3], rot([0.0, 1.0, 0.0], 0.0)));
    c.add_keyframe(usize::MAX - 1, kf(0.0, [9.0; 3], rot([0.0, 1.0, 0.0], 0.0)));
    assert_eq!(c.keyframes.len(), before);
    // equal time: new key goes after the old one (position marker distinguishes them)
    c.add_keyframe(0, kf(1.0, [100.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 60.0)));
    assert_eq!(c.bone_offsets, vec![0, 4, 5]);
    assert_eq!(c.keyframes[2].pose.position.x.to_f64(), 100.0);
    // bone 1's key is still where the offsets say
    assert_eq!(c.keyframes[4].time.to_f64(), 2.0);
    // insertion at the front
    c.add_keyframe(0, kf(-1.0, [-4.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 0.0)));
    assert_eq!(c.keyframes[0].time.to_f64(), -1.0);
    assert_eq!(c.bone_offsets, vec![0, 5, 6]);
}

/// Linear position interpolation between non-uniform keys, per bone and per axis.
#[test]
fn clip_sample_interpolates_each_segment_linearly() {
    let c = two_bone_clip(false);
    // segment [0,1] : (0,0,0) -> (4,2,-2)
    let s = c.sample(r(1, 4));
    assert_eq!(p(&s, 0), [1.0, 0.5, -0.5]);
    // segment [1,4] : (4,2,-2) -> (8,0,0), t = 2.5 -> local 0.5
    let s = c.sample(fx(2.5));
    let got = p(&s, 0);
    assert!(
        (got[0] - 6.0).abs() < 1e-9 && (got[1] - 1.0).abs() < 1e-9 && (got[2] + 1.0).abs() < 1e-9,
        "{got:?}"
    );
    // keys hit exactly
    assert_eq!(p(&c.sample(Fix128::ONE), 0), [4.0, 2.0, -2.0]);
    // single-key bone is constant at every time
    for t in [0.0, 2.0, 3.5] {
        assert_eq!(p(&c.sample(fx(t)), 1), [-1.0, -2.0, -3.0]);
    }
    assert!(
        angle_between(
            c.sample(fx(0.7)).bones[1].rotation,
            rot([1.0, 0.0, 0.0], 45.0)
        ) < 1e-9
    );
}

/// Rotation along a segment is the constant-angular-velocity arc (here 60 -> 120 degrees over [1,4]).
#[test]
fn clip_sample_rotation_follows_the_slerp_arc() {
    let c = two_bone_clip(false);
    for (t, deg) in [
        (0.5, 30.0),
        (1.0, 60.0),
        (2.5, 90.0),
        (3.25, 105.0),
        (4.0, 120.0),
    ] {
        let q = c.sample(fx(t)).bones[0].rotation;
        assert!(angle_between(q, rot([0.0, 1.0, 0.0], deg)) < 2e-4, "t {t}");
    }
}

/// Time handling: before the first key hold it; non-looping past the duration clamps (holds the
/// last key); looping wraps modulo the duration also for negative times and exactly at the period.
#[test]
fn clip_sample_time_handling_hold_clamp_and_wrap() {
    let nl = two_bone_clip(false);
    assert_eq!(p(&nl.sample(fx(-3.0)), 0), [0.0, 0.0, 0.0]);
    assert_eq!(p(&nl.sample(fx(4.0)), 0), [8.0, 0.0, 0.0]);
    assert_eq!(p(&nl.sample(fx(1000.0)), 0), [8.0, 0.0, 0.0]);
    let lp = two_bone_clip(true);
    // duration 4: t = 4.25 == 0.25, t = -3.75 == 0.25, t = 8 + 0.25
    let want = p(&lp.sample(r(1, 4)), 0);
    for t in [4.25f64, -3.75, 8.25, 400.25, -399.75] {
        let got = p(&lp.sample(fx(t)), 0);
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < 1e-9,
                "t {t}: {got:?} vs {want:?}"
            );
        }
    }
    // exactly one period: back at the start pose
    assert_eq!(p(&lp.sample(Fix128::from_int(4)), 0), [0.0, 0.0, 0.0]);
    assert_eq!(p(&lp.sample(Fix128::from_int(-4)), 0), [0.0, 0.0, 0.0]);
    // looping with a zero duration falls back to the clamp branch (no division by zero)
    let mut z = AnimationClip::new(1, Fix128::ZERO);
    z.add_keyframe(0, kf(0.0, [1.0, 1.0, 1.0], rot([0.0, 1.0, 0.0], 0.0)));
    z.add_keyframe(0, kf(1.0, [3.0, 3.0, 3.0], rot([0.0, 1.0, 0.0], 0.0)));
    assert_eq!(p(&z.sample(fx(5.0)), 0), [1.0, 1.0, 1.0]);
}

/// Duplicate key times: the later one wins at and after that time (step discontinuity), and
/// there is no division by the zero-length segment.
#[test]
fn clip_sample_duplicate_times_step_without_nan() {
    let mut c = AnimationClip::new(1, Fix128::from_int(4));
    c.looping = false;
    c.add_keyframe(0, kf(0.0, [0.0; 3], rot([0.0, 1.0, 0.0], 0.0)));
    c.add_keyframe(0, kf(1.0, [2.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 0.0)));
    c.add_keyframe(0, kf(1.0, [10.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 0.0)));
    c.add_keyframe(0, kf(3.0, [14.0, 0.0, 0.0], rot([0.0, 1.0, 0.0], 0.0)));
    assert_eq!(p(&c.sample(r(1, 2)), 0), [1.0, 0.0, 0.0]);
    assert_eq!(p(&c.sample(Fix128::ONE), 0)[0], 10.0);
    assert_eq!(p(&c.sample(Fix128::from_int(2)), 0)[0], 12.0);
}

/// `transition_speed` is in weight units per second: speed 2, dt 1/8 moves 1/4 per update, in
/// either direction, clamps at the target and then switches the mode.
#[test]
fn transition_moves_at_speed_times_dt_and_clamps() {
    let mut b = AnimationBlender::new(1);
    b.go_ragdoll();
    let dt = r(1, 8);
    for step in 1..=3 {
        b.update(dt);
        assert_eq!(b.blend_weight, r(step, 4), "step {step}");
        assert_eq!(b.mode, BlendMode::Blend);
        assert!(b.is_transitioning());
    }
    b.update(dt);
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert_eq!(b.mode, BlendMode::Ragdoll);
    assert!(!b.is_transitioning());
    b.go_animated();
    for step in (0..4).rev() {
        b.update(dt);
        assert_eq!(b.blend_weight, r(step, 4));
    }
    assert_eq!(b.mode, BlendMode::Animated);
    // a larger step than the distance does not overshoot
    b.transition_speed = Fix128::from_int(100);
    b.go_ragdoll();
    b.update(Fix128::ONE);
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert_eq!(b.mode, BlendMode::Ragdoll);
}

/// Output pose per mode: Animated -> animation pose, Ragdoll/Powered -> physics pose, Blend ->
/// weight 0 = animation ... weight 1 = physics (linear in position at weight 1/4).
#[test]
fn output_pose_follows_mode_and_weight() {
    let mut b = AnimationBlender::new(1);
    b.animation_pose = pose1(Vec3Fix::from_int(0, 0, 0), rot([0.0, 1.0, 0.0], 0.0));
    b.physics_pose = pose1(Vec3Fix::from_int(8, 4, 0), rot([0.0, 1.0, 0.0], 80.0));
    b.update(Fix128::ZERO);
    assert_eq!(b.output_pose.bones, b.animation_pose.bones);
    b.set_ragdoll();
    b.update(Fix128::ZERO);
    assert_eq!(b.output_pose.bones, b.physics_pose.bones);
    b.mode = BlendMode::Blend;
    b.blend_weight = r(1, 4);
    b.go_animated(); // target 0, mode Blend; one update of dt = 0 would snap nothing (diff 1/4)
    b.update(Fix128::ZERO);
    // weight stays 1/4: position 2, 1, 0 and rotation 20 degrees
    let o = b.output_pose.bones[0];
    assert_eq!((o.position.x.to_f64(), o.position.y.to_f64()), (2.0, 1.0));
    assert!(angle_between(o.rotation, rot([0.0, 1.0, 0.0], 20.0)) < 2e-4);
}

/// A transition started and then advanced with a zero time step must not be cancelled: the
/// auto-switch looks only at the weight (<= 0 -> Animated, >= 1 -> Ragdoll), so with dt = 0 the
/// weight is still at its starting end and the mode snaps back, after which the weight creeps to
/// the target while the output stays on the old pose forever.
#[test]
#[ignore = "known defect: AUD-A-S2W3-011: go_ragdoll() then update(dt = 0) (or transition_speed 0) flips mode Blend -> Animated immediately; later updates move the weight to 1 but mode stays Animated and the physics pose never appears (mirror: go_animated() from weight 1)"]
fn a_zero_dt_update_does_not_cancel_a_started_transition() {
    let mut b = AnimationBlender::new(1);
    b.physics_pose = pose1(Vec3Fix::from_int(8, 0, 0), rot([0.0, 1.0, 0.0], 0.0));
    b.go_ragdoll();
    b.update(Fix128::ZERO);
    for _ in 0..40 {
        b.update(r(1, 16));
    }
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert_eq!(b.mode, BlendMode::Ragdoll, "ragdoll never reached");
    assert_eq!(b.output_pose.bones, b.physics_pose.bones);
}

#[test]
#[ignore = "known defect: AUD-A-S2W3-011: mirror case, go_animated() from full ragdoll then update(dt = 0) flips mode to Ragdoll"]
fn a_zero_dt_update_does_not_cancel_a_return_to_animation() {
    let mut b = AnimationBlender::new(1);
    b.animation_pose = pose1(Vec3Fix::from_int(1, 0, 0), rot([0.0, 1.0, 0.0], 0.0));
    b.physics_pose = pose1(Vec3Fix::from_int(8, 0, 0), rot([0.0, 1.0, 0.0], 0.0));
    b.set_ragdoll();
    b.go_animated();
    b.update(Fix128::ZERO);
    for _ in 0..40 {
        b.update(r(1, 16));
    }
    assert_eq!(b.mode, BlendMode::Animated, "animation never reached");
    assert_eq!(b.output_pose.bones, b.animation_pose.bones);
}

/// `go_*` do not touch the weight; `set_*` are immediate; `go_powered` keeps Powered through the
/// transition and `get_motor_targets` is the live animation pose.
#[test]
fn mode_commands_and_motor_targets() {
    let mut b = AnimationBlender::new(2);
    assert_eq!(
        (b.mode, b.blend_weight),
        (BlendMode::Animated, Fix128::ZERO)
    );
    assert_eq!(b.transition_speed, Fix128::from_int(2));
    assert_eq!(b.motor_stiffness, Fix128::from_int(100));
    b.go_ragdoll();
    assert_eq!((b.mode, b.blend_weight), (BlendMode::Blend, Fix128::ZERO));
    b.set_ragdoll();
    assert_eq!((b.mode, b.blend_weight), (BlendMode::Ragdoll, Fix128::ONE));
    assert!(!b.is_transitioning());
    b.go_powered();
    assert_eq!(b.mode, BlendMode::Powered);
    assert!(!b.is_transitioning(), "already at weight 1");
    b.set_animated();
    assert_eq!(
        (b.mode, b.blend_weight),
        (BlendMode::Animated, Fix128::ZERO)
    );
    assert!(!b.is_transitioning());
    b.animation_pose.bones[1].position = Vec3Fix::from_int(3, 4, 5);
    assert_eq!(
        b.get_motor_targets().bones[1].position,
        Vec3Fix::from_int(3, 4, 5)
    );
    assert_eq!(b.get_motor_targets().bone_count(), 2);
}

/// `SkeletonPose::new` is identity poses; `lerp` of two poses of different length uses the
/// shorter; extrapolation outside [0, 1] is unclamped on position.
#[test]
fn skeleton_pose_basics() {
    let p3 = SkeletonPose::new(3);
    assert!(p3
        .bones
        .iter()
        .all(|b| b.position == Vec3Fix::ZERO && b.rotation == QuatFix::IDENTITY));
    let mut a = SkeletonPose::new(1);
    a.bones[0].position = Vec3Fix::from_int(1, 2, 3);
    let mut b = SkeletonPose::new(1);
    b.bones[0].position = Vec3Fix::from_int(5, 6, 7);
    let e = SkeletonPose::lerp(&a, &b, Fix128::from_int(2));
    assert_eq!(e.bones[0].position, Vec3Fix::from_int(9, 10, 11));
    let n = SkeletonPose::lerp(&a, &b, -Fix128::ONE);
    assert_eq!(n.bones[0].position, Vec3Fix::from_int(-3, -2, -1));
}

/// The snap-to-target window is 1/1000: 5/1000 away from the target the weight still moves by
/// exactly `speed * dt`, it does not jump to the target.
#[test]
fn weight_snaps_only_inside_the_one_thousandth_window() {
    let mut b = AnimationBlender::new(1);
    b.transition_speed = Fix128::ONE;
    b.go_ragdoll();
    b.blend_weight = Fix128::ONE - r(1, 200);
    b.update(r(1, 1024));
    assert_eq!(b.blend_weight, Fix128::ONE - r(1, 200) + r(1, 1024));
    assert!(b.blend_weight < Fix128::ONE);
    assert!(b.is_transitioning());
    // 1/2000 away: snaps exactly (and the mode completes)
    b.blend_weight = Fix128::ONE - r(1, 2000);
    b.update(Fix128::ZERO);
    assert_eq!(b.blend_weight, Fix128::ONE);
    assert_eq!(b.mode, BlendMode::Ragdoll);
}

/// The rotation output is a unit quaternion also for scaled (drifted) input rotations, on the
/// full-SLERP path (dot <= 0.999) as well as the near-parallel NLERP path.
#[test]
fn pose_lerp_rotation_is_unit_for_scaled_inputs() {
    let scale = |q: QuatFix, k: f64| {
        QuatFix::new(
            fx(q.x.to_f64() * k),
            fx(q.y.to_f64() * k),
            fx(q.z.to_f64() * k),
            fx(q.w.to_f64() * k),
        )
    };
    for (arc, k) in [(60.0f64, 0.5f64), (100.0, 0.7), (2.0, 0.5), (30.0, 3.0)] {
        let a = pose1(Vec3Fix::ZERO, scale(rot([1.0, 2.0, 3.0], 0.0), k));
        let b = pose1(Vec3Fix::ZERO, scale(rot([1.0, 2.0, 3.0], arc), k));
        for i in 0..=4 {
            let q = SkeletonPose::lerp(&a, &b, fx(f64::from(i) / 4.0)).bones[0].rotation;
            let n: f64 = q64(q).iter().map(|c| c * c).sum();
            assert!(
                (n - 1.0).abs() < 1e-12,
                "arc {arc} scale {k} i {i}: |q|^2 = {n}"
            );
        }
    }
}
