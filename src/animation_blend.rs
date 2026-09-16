//! Ragdoll Animation Blending
//!
//! Blends between keyframe animation and physics ragdoll simulation.
//! Extends neural.rs `RagdollController` with smooth transitions.
//!
//! # Modes
//!
//! - **Animated**: Pure keyframe animation (physics inactive)
//! - **Ragdoll**: Pure physics simulation (animation inactive)
//! - **Blend**: Weighted mix of animation and physics
//! - **Powered**: Physics with motor targets from animation
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Blend Mode
// ============================================================================

/// Animation-physics blend mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Pure animation (ignore physics)
    Animated,
    /// Pure physics (ignore animation)
    Ragdoll,
    /// Weighted blend between animation and physics
    Blend,
    /// Physics with animation-driven motor targets
    Powered,
}

// ============================================================================
// Animation Pose
// ============================================================================

/// A single bone pose (position + rotation)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BonePose {
    /// Bone position (local space)
    pub position: Vec3Fix,
    /// Bone rotation (local space)
    pub rotation: QuatFix,
}

impl Default for BonePose {
    fn default() -> Self {
        Self {
            position: Vec3Fix::ZERO,
            rotation: QuatFix::IDENTITY,
        }
    }
}

/// Complete skeleton pose (all bones)
#[derive(Clone, Debug)]
pub struct SkeletonPose {
    /// Bone poses indexed by bone ID
    pub bones: Vec<BonePose>,
}

impl SkeletonPose {
    /// Create empty pose for N bones
    #[must_use]
    pub fn new(num_bones: usize) -> Self {
        Self {
            bones: vec![BonePose::default(); num_bones],
        }
    }

    /// Number of bones
    #[inline]
    #[must_use]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Lerp between two poses
    #[must_use]
    pub fn lerp(a: &Self, b: &Self, t: Fix128) -> Self {
        let n = a.bones.len().min(b.bones.len());
        let one_minus_t = Fix128::ONE - t;

        let bones: Vec<BonePose> = (0..n)
            .map(|i| BonePose {
                position: Vec3Fix::new(
                    a.bones[i].position.x * one_minus_t + b.bones[i].position.x * t,
                    a.bones[i].position.y * one_minus_t + b.bones[i].position.y * t,
                    a.bones[i].position.z * one_minus_t + b.bones[i].position.z * t,
                ),
                rotation: quat_slerp(a.bones[i].rotation, b.bones[i].rotation, t),
            })
            .collect();

        Self { bones }
    }
}

// ============================================================================
// Animation Clip
// ============================================================================

/// Keyframe for a single bone
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keyframe {
    /// Time in seconds
    pub time: Fix128,
    /// Bone pose at this time
    pub pose: BonePose,
}

/// Animation clip (sequence of keyframes per bone).
///
/// Uses a flat `keyframes` buffer with a `bone_offsets` table so that all
/// keyframe data is stored contiguously. `bone_offsets[b]..bone_offsets[b+1]`
/// gives the keyframe slice for bone `b`.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Clip name
    pub name: Vec<u8>,
    /// Duration in seconds
    pub duration: Fix128,
    /// All keyframes packed flat (sorted per-bone by time)
    pub keyframes: Vec<Keyframe>,
    /// Start index in `keyframes` for each bone; length = num_bones + 1
    pub bone_offsets: Vec<usize>,
    /// Whether clip loops
    pub looping: bool,
}

impl AnimationClip {
    /// Create empty clip for `num_bones` bones.
    #[must_use]
    pub fn new(num_bones: usize, duration: Fix128) -> Self {
        Self {
            name: Vec::new(),
            duration,
            keyframes: Vec::new(),
            bone_offsets: vec![0; num_bones + 1],
            looping: true,
        }
    }

    /// Number of bones this clip was created for.
    #[inline]
    #[must_use]
    pub fn num_bones(&self) -> usize {
        self.bone_offsets.len().saturating_sub(1)
    }

    /// Add a keyframe for a bone.
    ///
    /// Inserts in time-sorted order and shifts subsequent bone offsets.
    pub fn add_keyframe(&mut self, bone_idx: usize, keyframe: Keyframe) {
        let num_bones = self.num_bones();
        if bone_idx >= num_bones {
            return;
        }
        let start = self.bone_offsets[bone_idx];
        let end = self.bone_offsets[bone_idx + 1];

        // Find insertion position (maintain sorted order by time)
        let insert_pos = start
            + self.keyframes[start..end]
                .iter()
                .position(|kf| kf.time > keyframe.time)
                .unwrap_or(end - start);

        self.keyframes.insert(insert_pos, keyframe);

        // Shift offsets for all subsequent bones
        for off in &mut self.bone_offsets[bone_idx + 1..] {
            *off += 1;
        }
    }

    /// Sample pose at given time.
    #[must_use]
    pub fn sample(&self, time: Fix128) -> SkeletonPose {
        let t = if self.looping && !self.duration.is_zero() {
            let cycles = time / self.duration;
            time - self.duration * Fix128::from_int(cycles.hi)
        } else if time > self.duration {
            self.duration
        } else {
            time
        };

        let num_bones = self.num_bones();
        let mut pose = SkeletonPose::new(num_bones);

        for bone_idx in 0..num_bones {
            let start = self.bone_offsets[bone_idx];
            let end = self.bone_offsets[bone_idx + 1];
            let bone_keyframes = &self.keyframes[start..end];

            if bone_keyframes.is_empty() {
                continue;
            }

            if bone_keyframes.len() == 1 {
                pose.bones[bone_idx] = bone_keyframes[0].pose;
                continue;
            }

            // Before the first keyframe: hold its pose (before 1.2.0 the
            // first segment was extrapolated backwards with a negative
            // `local_t`, e.g. keys at t = 1 / 3 sampled at t = 0 gave
            // `2·p₀ − p₁`).
            if t < bone_keyframes[0].time {
                pose.bones[bone_idx] = bone_keyframes[0].pose;
                continue;
            }

            // Find surrounding keyframes
            let mut prev_idx = 0;
            for (i, kf) in bone_keyframes.iter().enumerate() {
                if kf.time <= t {
                    prev_idx = i;
                }
            }

            let next_idx = (prev_idx + 1).min(bone_keyframes.len() - 1);
            let prev = &bone_keyframes[prev_idx];
            let next = &bone_keyframes[next_idx];

            if prev_idx == next_idx {
                pose.bones[bone_idx] = prev.pose;
            } else {
                let segment_duration = next.time - prev.time;
                let local_t = if segment_duration.is_zero() {
                    Fix128::ZERO
                } else {
                    (t - prev.time) / segment_duration
                };

                pose.bones[bone_idx] = BonePose {
                    position: Vec3Fix::new(
                        prev.pose.position.x
                            + (next.pose.position.x - prev.pose.position.x) * local_t,
                        prev.pose.position.y
                            + (next.pose.position.y - prev.pose.position.y) * local_t,
                        prev.pose.position.z
                            + (next.pose.position.z - prev.pose.position.z) * local_t,
                    ),
                    rotation: quat_slerp(prev.pose.rotation, next.pose.rotation, local_t),
                };
            }
        }

        pose
    }
}

// ============================================================================
// Animation Blender
// ============================================================================

/// Animation-physics blender
pub struct AnimationBlender {
    /// Current blend mode
    pub mode: BlendMode,
    /// Blend weight (0 = full animation, 1 = full physics)
    pub blend_weight: Fix128,
    /// Target blend weight (for smooth transitions)
    target_weight: Fix128,
    /// Blend transition speed (weight units per second)
    pub transition_speed: Fix128,
    /// Current animation pose
    pub animation_pose: SkeletonPose,
    /// Current physics pose
    pub physics_pose: SkeletonPose,
    /// Output blended pose
    pub output_pose: SkeletonPose,
    /// Motor stiffness when in Powered mode
    pub motor_stiffness: Fix128,
}

impl AnimationBlender {
    /// Create a new blender for N bones
    #[must_use]
    pub fn new(num_bones: usize) -> Self {
        Self {
            mode: BlendMode::Animated,
            blend_weight: Fix128::ZERO,
            target_weight: Fix128::ZERO,
            transition_speed: Fix128::from_int(2),
            animation_pose: SkeletonPose::new(num_bones),
            physics_pose: SkeletonPose::new(num_bones),
            output_pose: SkeletonPose::new(num_bones),
            motor_stiffness: Fix128::from_int(100),
        }
    }

    /// Transition to ragdoll mode
    pub fn go_ragdoll(&mut self) {
        self.mode = BlendMode::Blend;
        self.target_weight = Fix128::ONE;
    }

    /// Transition to animated mode
    pub fn go_animated(&mut self) {
        self.mode = BlendMode::Blend;
        self.target_weight = Fix128::ZERO;
    }

    /// Set to powered ragdoll (physics with animation targets)
    pub fn go_powered(&mut self) {
        self.mode = BlendMode::Powered;
        self.target_weight = Fix128::ONE;
    }

    /// Immediately set ragdoll (no transition)
    pub fn set_ragdoll(&mut self) {
        self.mode = BlendMode::Ragdoll;
        self.blend_weight = Fix128::ONE;
        self.target_weight = Fix128::ONE;
    }

    /// Immediately set animated (no transition)
    pub fn set_animated(&mut self) {
        self.mode = BlendMode::Animated;
        self.blend_weight = Fix128::ZERO;
        self.target_weight = Fix128::ZERO;
    }

    /// Update blender state
    pub fn update(&mut self, dt: Fix128) {
        // Smooth weight transition
        let diff = self.target_weight - self.blend_weight;
        if diff.abs() > Fix128::from_ratio(1, 1000) {
            let step = self.transition_speed * dt;
            if diff > Fix128::ZERO {
                self.blend_weight = self.blend_weight + step;
                if self.blend_weight > self.target_weight {
                    self.blend_weight = self.target_weight;
                }
            } else {
                self.blend_weight = self.blend_weight - step;
                if self.blend_weight < self.target_weight {
                    self.blend_weight = self.target_weight;
                }
            }
        } else {
            self.blend_weight = self.target_weight;
        }

        // Auto-switch mode when transition completes
        if self.mode == BlendMode::Blend {
            if self.blend_weight >= Fix128::ONE {
                self.mode = BlendMode::Ragdoll;
            } else if self.blend_weight <= Fix128::ZERO {
                self.mode = BlendMode::Animated;
            }
        }

        // Compute output pose
        match self.mode {
            BlendMode::Animated => {
                self.output_pose.clone_from(&self.animation_pose);
            }
            BlendMode::Ragdoll | BlendMode::Powered => {
                self.output_pose.clone_from(&self.physics_pose);
            }
            BlendMode::Blend => {
                self.output_pose =
                    SkeletonPose::lerp(&self.animation_pose, &self.physics_pose, self.blend_weight);
            }
        }
    }

    /// Get motor targets for powered mode (animation poses as motor goals)
    #[must_use]
    pub const fn get_motor_targets(&self) -> &SkeletonPose {
        &self.animation_pose
    }

    /// Check if currently transitioning
    #[must_use]
    pub fn is_transitioning(&self) -> bool {
        (self.blend_weight - self.target_weight).abs() > Fix128::from_ratio(1, 1000)
    }
}

// ============================================================================
// Quaternion SLERP
// ============================================================================

/// Spherical linear interpolation between two quaternions
fn quat_slerp(a: QuatFix, b: QuatFix, t: Fix128) -> QuatFix {
    let mut dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

    // Ensure shortest path
    let b = if dot < Fix128::ZERO {
        dot = -dot;
        QuatFix::new(-b.x, -b.y, -b.z, -b.w)
    } else {
        b
    };

    // If quaternions are very close, use NLERP (faster, avoids division by near-zero)
    if dot > Fix128::from_ratio(999, 1000) {
        let one_minus_t = Fix128::ONE - t;
        return QuatFix::new(
            a.x * one_minus_t + b.x * t,
            a.y * one_minus_t + b.y * t,
            a.z * one_minus_t + b.z * t,
            a.w * one_minus_t + b.w * t,
        )
        .normalize();
    }

    // Full SLERP: theta = acos(dot) via atan2(sqrt(1 - dot^2), dot)
    let one_minus_dot_sq = Fix128::ONE - dot * dot;
    // Clamp to avoid negative values from numerical error
    let sin_half = if one_minus_dot_sq.is_negative() {
        Fix128::ZERO
    } else {
        one_minus_dot_sq.sqrt()
    };
    let theta = Fix128::atan2(sin_half, dot);
    let sin_theta = sin_half;

    if sin_theta.is_zero() {
        return a;
    }

    let one_minus_t = Fix128::ONE - t;
    let s0 = (one_minus_t * theta).sin() / sin_theta;
    let s1 = (t * theta).sin() / sin_theta;

    QuatFix::new(
        a.x * s0 + b.x * s1,
        a.y * s0 + b.y * s1,
        a.z * s0 + b.z * s1,
        a.w * s0 + b.w * s1,
    )
    .normalize()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blend_modes() {
        let mut blender = AnimationBlender::new(3);
        assert_eq!(blender.mode, BlendMode::Animated);

        blender.go_ragdoll();
        assert_eq!(blender.mode, BlendMode::Blend);
        assert_eq!(blender.target_weight, Fix128::ONE);
    }

    #[test]
    fn test_blend_transition() {
        let mut blender = AnimationBlender::new(2);
        blender.go_ragdoll();

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..120 {
            // 2 seconds at 60fps
            blender.update(dt);
        }

        // After transition, should be in ragdoll mode
        assert_eq!(blender.mode, BlendMode::Ragdoll);
        assert_eq!(blender.blend_weight, Fix128::ONE);
    }

    #[test]
    fn test_pose_lerp() {
        let a = SkeletonPose {
            bones: vec![BonePose {
                position: Vec3Fix::ZERO,
                rotation: QuatFix::IDENTITY,
            }],
        };
        let b = SkeletonPose {
            bones: vec![BonePose {
                position: Vec3Fix::from_int(10, 0, 0),
                rotation: QuatFix::IDENTITY,
            }],
        };

        let mid = SkeletonPose::lerp(&a, &b, Fix128::from_ratio(1, 2));
        let x = mid.bones[0].position.x.to_f32();
        assert!((x - 5.0).abs() < 0.1, "Midpoint should be at x=5, got {x}");
    }

    #[test]
    fn test_animation_clip() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(2));
        clip.add_keyframe(
            0,
            Keyframe {
                time: Fix128::ZERO,
                pose: BonePose {
                    position: Vec3Fix::ZERO,
                    rotation: QuatFix::IDENTITY,
                },
            },
        );
        clip.add_keyframe(
            0,
            Keyframe {
                time: Fix128::from_int(2),
                pose: BonePose {
                    position: Vec3Fix::from_int(10, 0, 0),
                    rotation: QuatFix::IDENTITY,
                },
            },
        );

        let pose = clip.sample(Fix128::ONE); // t=1s, midpoint
        let x = pose.bones[0].position.x.to_f32();
        assert!((x - 5.0).abs() < 0.5, "At t=1s, x should be ~5, got {x}");
    }

    #[test]
    fn test_immediate_mode_switch() {
        let mut blender = AnimationBlender::new(1);

        blender.set_ragdoll();
        assert_eq!(blender.mode, BlendMode::Ragdoll);
        assert!(!blender.is_transitioning());

        blender.set_animated();
        assert_eq!(blender.mode, BlendMode::Animated);
    }

    fn pose_at(n: usize, x: i64) -> SkeletonPose {
        let mut p = SkeletonPose::new(n);
        for (i, b) in p.bones.iter_mut().enumerate() {
            b.position = Vec3Fix::from_int(x, i as i64, 0);
        }
        p
    }

    #[test]
    fn bone_count_follows_construction_lerp_and_sampling() {
        assert_eq!(SkeletonPose::new(0).bone_count(), 0);
        let five = SkeletonPose::new(5);
        assert_eq!(five.bone_count(), 5);
        assert_eq!(five.bone_count(), five.bones.len());
        // lerp は短い方の本数
        let two = SkeletonPose::new(2);
        assert_eq!(
            SkeletonPose::lerp(&five, &two, Fix128::from_ratio(1, 2)).bone_count(),
            2
        );
        assert_eq!(
            SkeletonPose::lerp(&two, &five, Fix128::from_ratio(1, 2)).bone_count(),
            2
        );
        // clip.sample は clip の bone 数
        let clip = AnimationClip::new(3, Fix128::ONE);
        assert_eq!(clip.sample(Fix128::ZERO).bone_count(), 3);
        // bones を伸ばせば追従
        let mut grown = two;
        grown.bones.push(BonePose::default());
        assert_eq!(grown.bone_count(), 3);
        // blender の 3 pose は同じ本数で始まる
        let blender = AnimationBlender::new(4);
        assert_eq!(blender.animation_pose.bone_count(), 4);
        assert_eq!(blender.physics_pose.bone_count(), 4);
        assert_eq!(blender.output_pose.bone_count(), 4);
    }

    #[test]
    fn go_animated_blends_from_ragdoll_back_to_animation_at_transition_speed() {
        let mut blender = AnimationBlender::new(2);
        blender.animation_pose = pose_at(2, 0);
        blender.physics_pose = pose_at(2, 8);
        blender.set_ragdoll();
        assert_eq!(blender.blend_weight, Fix128::ONE);

        blender.go_animated();
        assert_eq!(blender.mode, BlendMode::Blend);
        assert_eq!(blender.target_weight, Fix128::ZERO);
        assert_eq!(
            blender.blend_weight,
            Fix128::ONE,
            "go_* は即時に weight を変えない"
        );
        assert!(blender.is_transitioning());

        // transition_speed 2 × dt 1/4 = 1/2 per update: 1 → 1/2 (Blend、中点) → 0 (Animated)
        let dt = Fix128::from_ratio(1, 4);
        blender.update(dt);
        assert_eq!(blender.blend_weight, Fix128::from_ratio(1, 2));
        assert_eq!(blender.mode, BlendMode::Blend);
        assert!(blender.is_transitioning());
        for (i, b) in blender.output_pose.bones.iter().enumerate() {
            assert_eq!(b.position, Vec3Fix::from_int(4, i as i64, 0), "bone {i}");
            assert_eq!(b.rotation, QuatFix::IDENTITY);
        }

        blender.update(dt);
        assert_eq!(blender.blend_weight, Fix128::ZERO);
        assert_eq!(blender.mode, BlendMode::Animated);
        assert!(!blender.is_transitioning());
        assert_eq!(blender.output_pose.bones, blender.animation_pose.bones);

        // 以後 update しても Animated のまま animation pose を出力 (physics を変えても無関係)
        blender.physics_pose = pose_at(2, -100);
        blender.update(dt);
        assert_eq!(blender.mode, BlendMode::Animated);
        assert_eq!(blender.output_pose.bones, pose_at(2, 0).bones);

        // 逆方向 (Animated → go_ragdoll) も同じ速度で 2 update で Ragdoll
        blender.go_ragdoll();
        blender.update(dt);
        assert_eq!(blender.blend_weight, Fix128::from_ratio(1, 2));
        blender.update(dt);
        assert_eq!(blender.mode, BlendMode::Ragdoll);
        assert_eq!(blender.output_pose.bones, pose_at(2, -100).bones);
    }

    #[test]
    fn go_powered_outputs_physics_pose_and_exposes_animation_as_motor_targets() {
        let mut blender = AnimationBlender::new(3);
        blender.animation_pose = pose_at(3, 1);
        blender.physics_pose = pose_at(3, 9);

        blender.go_powered();
        assert_eq!(blender.mode, BlendMode::Powered);
        assert_eq!(blender.target_weight, Fix128::ONE);
        // motor target は animation pose そのもの
        assert_eq!(blender.get_motor_targets().bones, pose_at(3, 1).bones);
        assert_eq!(blender.get_motor_targets().bone_count(), 3);

        let dt = Fix128::from_ratio(1, 4);
        // Powered は途中 weight (1/2) でも lerp せず physics pose を出力、mode も自動遷移しない
        blender.update(dt);
        assert_eq!(blender.blend_weight, Fix128::from_ratio(1, 2));
        assert_eq!(blender.mode, BlendMode::Powered);
        assert_eq!(blender.output_pose.bones, pose_at(3, 9).bones);
        blender.update(dt);
        assert_eq!(blender.blend_weight, Fix128::ONE);
        assert_eq!(
            blender.mode,
            BlendMode::Powered,
            "Powered は Ragdoll に変わらない"
        );
        assert!(!blender.is_transitioning());
        blender.update(dt);
        assert_eq!(blender.mode, BlendMode::Powered);
        assert_eq!(blender.output_pose.bones, pose_at(3, 9).bones);

        // animation pose を差し替えると motor target も追従 (出力は physics のまま)
        blender.animation_pose = pose_at(3, 5);
        blender.update(dt);
        assert_eq!(blender.get_motor_targets().bones, pose_at(3, 5).bones);
        assert_eq!(blender.output_pose.bones, pose_at(3, 9).bones);

        // Animated モードでも get_motor_targets は animation pose を返す
        blender.set_animated();
        assert_eq!(blender.get_motor_targets().bones, pose_at(3, 5).bones);
    }

    // ------------------------------------------------------------------
    // Mutation-kill tests (cargo-mutants 42.9 % → ≥ 85 %)
    // ------------------------------------------------------------------

    fn r(num: i64, denom: i64) -> Fix128 {
        Fix128::from_ratio(num, denom)
    }

    fn kf(time: Fix128, position: Vec3Fix) -> Keyframe {
        Keyframe {
            time,
            pose: BonePose {
                position,
                rotation: QuatFix::IDENTITY,
            },
        }
    }

    /// Unit quaternion (1, 2, 2, 4) / 5 — all four components non-zero.
    fn qa() -> QuatFix {
        QuatFix::new(r(1, 5), r(2, 5), r(2, 5), r(4, 5))
    }

    /// Unit quaternion (3, 1, 1, 5) / 6 — all four components non-zero,
    /// `qa · qb = 27/30 = 0.9` (θ ≈ 25.84°, full SLERP path).
    fn qb() -> QuatFix {
        QuatFix::new(r(3, 6), r(1, 6), r(1, 6), r(5, 6))
    }

    fn q_neg(q: QuatFix) -> QuatFix {
        QuatFix::new(-q.x, -q.y, -q.z, -q.w)
    }

    fn q_f64(q: QuatFix) -> [f64; 4] {
        [q.x.to_f64(), q.y.to_f64(), q.z.to_f64(), q.w.to_f64()]
    }

    fn norm4(v: [f64; 4]) -> [f64; 4] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
        [v[0] / len, v[1] / len, v[2] / len, v[3] / len]
    }

    /// Textbook SLERP in f64 (independent oracle): shortest path, θ = acos(a·b),
    /// `(sin((1-t)θ)·a + sin(tθ)·b) / sin θ`, renormalised.
    #[allow(clippy::disallowed_methods)] // the f64 libm values are the oracle on purpose
    fn slerp_ref(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
        let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        let b = if dot < 0.0 {
            dot = -dot;
            [-b[0], -b[1], -b[2], -b[3]]
        } else {
            b
        };
        let theta = dot.clamp(-1.0, 1.0).acos();
        let s = theta.sin();
        let s0 = ((1.0 - t) * theta).sin() / s;
        let s1 = (t * theta).sin() / s;
        norm4([
            a[0] * s0 + b[0] * s1,
            a[1] * s0 + b[1] * s1,
            a[2] * s0 + b[2] * s1,
            a[3] * s0 + b[3] * s1,
        ])
    }

    /// Normalised LERP in f64 (the documented fast path for `dot > 0.999`).
    fn nlerp_ref(a: [f64; 4], b: [f64; 4], t: f64) -> [f64; 4] {
        let u = 1.0 - t;
        norm4([
            a[0] * u + b[0] * t,
            a[1] * u + b[1] * t,
            a[2] * u + b[2] * t,
            a[3] * u + b[3] * t,
        ])
    }

    fn assert_quat_close(got: QuatFix, want: [f64; 4], tol: f64, ctx: &str) {
        let g = q_f64(got);
        for i in 0..4 {
            assert!(
                (g[i] - want[i]).abs() <= tol,
                "{ctx}: component {i} got {} want {} (|Δ| = {:e} > {tol:e})",
                g[i],
                want[i],
                (g[i] - want[i]).abs()
            );
        }
    }

    /// Kills 93:43 `*`→`/`, 95:57 `+`→`-`/`*`, 95:43 `*`→`/`, 95:81 `*`→`/`
    /// (SkeletonPose::lerp x / z 成分の算術).
    ///
    /// t = 1/4 で a=(3,5,7), b=(11,13,15) → (5, 7, 9) が dyadic 厳密解
    #[test]
    fn pose_lerp_exact_values_all_axes_at_quarter() {
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
        let q = SkeletonPose::lerp(&a, &b, r(1, 4));
        assert_eq!(q.bones[0].position, Vec3Fix::from_int(5, 7, 9));
        assert_eq!(q.bones[0].rotation, QuatFix::IDENTITY);
        // 端点は厳密に一致
        assert_eq!(
            SkeletonPose::lerp(&a, &b, Fix128::ZERO).bones[0].position,
            Vec3Fix::from_int(3, 5, 7)
        );
        assert_eq!(
            SkeletonPose::lerp(&a, &b, Fix128::ONE).bones[0].position,
            Vec3Fix::from_int(11, 13, 15)
        );
        // 3/4 も dyadic 厳密解 (9, 11, 13)
        assert_eq!(
            SkeletonPose::lerp(&a, &b, r(3, 4)).bones[0].position,
            Vec3Fix::from_int(9, 11, 13)
        );
    }

    /// Kills 172:40 `>`→`==` / `>`→`>=` と 173:32 `-`→`+` (add_keyframe の挿入位置).
    #[test]
    fn add_keyframe_keeps_time_order_and_appends_equal_times_after() {
        let mut clip = AnimationClip::new(2, Fix128::from_int(4));
        // bone 0: 0, 2 を入れてから 1 を割り込ませる → [0, 1, 2]
        clip.add_keyframe(0, kf(Fix128::ZERO, Vec3Fix::from_int(0, 0, 0)));
        clip.add_keyframe(0, kf(Fix128::from_int(2), Vec3Fix::from_int(2, 0, 0)));
        clip.add_keyframe(0, kf(Fix128::ONE, Vec3Fix::from_int(1, 0, 0)));
        let times: Vec<i64> = clip.keyframes.iter().map(|k| k.time.hi).collect();
        assert_eq!(times, vec![0, 1, 2], "`>` → `==` なら 1 が末尾に付く");
        assert_eq!(clip.bone_offsets, vec![0, 3, 3]);

        // 同時刻 keyframe は既存の後ろに入る (stable、`>=` だと前に入る)
        clip.add_keyframe(0, kf(Fix128::ONE, Vec3Fix::from_int(7, 0, 0)));
        assert_eq!(clip.keyframes[1].pose.position, Vec3Fix::from_int(1, 0, 0));
        assert_eq!(clip.keyframes[2].pose.position, Vec3Fix::from_int(7, 0, 0));
        assert_eq!(clip.bone_offsets, vec![0, 4, 4]);

        // bone 1 (start = 4 ≠ 0): 空 slice への追加は start 位置、`end + start` だと
        // insert index 8 > len 4 で panic
        clip.add_keyframe(1, kf(Fix128::from_int(3), Vec3Fix::from_int(0, 3, 0)));
        assert_eq!(clip.bone_offsets, vec![0, 4, 5]);
        assert_eq!(clip.keyframes[4].pose.position, Vec3Fix::from_int(0, 3, 0));
        // bone 1 の末尾追加 (position None → unwrap_or(end - start) = 1)
        clip.add_keyframe(1, kf(Fix128::from_int(4), Vec3Fix::from_int(0, 4, 0)));
        assert_eq!(clip.bone_offsets, vec![0, 4, 6]);
        assert_eq!(clip.keyframes[5].pose.position, Vec3Fix::from_int(0, 4, 0));
        // 範囲外 bone は無視
        clip.add_keyframe(2, kf(Fix128::ZERO, Vec3Fix::ZERO));
        assert_eq!(clip.keyframes.len(), 6);
    }

    /// Kills 227:50 `-`→`+`, 231:24 `-`→`+`, 237:53, 239:29 / 239:53 / 239:77,
    /// 241:29 / 241:53 / 241:77 (sample の区間補間算術、prev.time ≠ 0 / prev.pos ≠ 0).
    ///
    /// keyframes t=1: (2, 4, 6), t=3: (10, 12, 14)  → t=2 で (6, 8, 10)、t=3/2 で (4, 6, 8)
    #[test]
    fn sample_interpolates_between_nonzero_keyframes_exactly() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(4));
        clip.looping = false;
        clip.add_keyframe(0, kf(Fix128::ONE, Vec3Fix::from_int(2, 4, 6)));
        clip.add_keyframe(0, kf(Fix128::from_int(3), Vec3Fix::from_int(10, 12, 14)));

        assert_eq!(
            clip.sample(Fix128::from_int(2)).bones[0].position,
            Vec3Fix::from_int(6, 8, 10),
            "local_t = (2-1)/(3-1) = 1/2"
        );
        assert_eq!(
            clip.sample(r(3, 2)).bones[0].position,
            Vec3Fix::from_int(4, 6, 8),
            "local_t = 1/4"
        );
        // keyframe 時刻ちょうどでは keyframe pose そのもの
        assert_eq!(
            clip.sample(Fix128::ONE).bones[0].position,
            Vec3Fix::from_int(2, 4, 6)
        );
        assert_eq!(
            clip.sample(Fix128::from_int(3)).bones[0].position,
            Vec3Fix::from_int(10, 12, 14)
        );
        // 最初の keyframe より前は最初の pose を hold する (1.2.0、
        // `sample_before_first_keyframe_holds_first_pose`)
    }

    /// Kills 220:68 `-`→`+` / `-`→`/` (next_idx clamp、末尾 keyframe 以降で index out of bounds).
    #[test]
    fn sample_past_last_keyframe_holds_last_pose_without_panic() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(2));
        clip.add_keyframe(0, kf(Fix128::ZERO, Vec3Fix::from_int(1, 1, 1)));
        clip.add_keyframe(0, kf(Fix128::ONE, Vec3Fix::from_int(9, 9, 9)));
        // looping, duration 2: t = 3/2 は最後の keyframe (t=1) より後 → prev = next = 1
        assert_eq!(
            clip.sample(r(3, 2)).bones[0].position,
            Vec3Fix::from_int(9, 9, 9)
        );
        // 非 loop でも同じ (clamp 後 t = 2 > 1)
        clip.looping = false;
        assert_eq!(
            clip.sample(Fix128::from_int(5)).bones[0].position,
            Vec3Fix::from_int(9, 9, 9)
        );
    }

    /// Kills 186:33 `&&`→`||`, 186:36 delete `!`, 188:18 `-`→`+`, 188:34 `*`→`/`
    /// (loop wrap: t = time − duration·floor(time/duration)).
    ///
    /// keyframes t=0: (0,0,0), t=2: (8,0,0), duration 2 → 線形 x = 4·t
    #[test]
    fn sample_looping_wraps_time_modulo_duration() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(2));
        clip.add_keyframe(0, kf(Fix128::ZERO, Vec3Fix::ZERO));
        clip.add_keyframe(0, kf(Fix128::from_int(2), Vec3Fix::from_int(8, 0, 0)));
        assert!(clip.looping);

        // time = 3 → 1 → x = 4 (`-`→`+` なら t = 5 → 末尾 8、`!` 削除なら clamp → 8)
        assert_eq!(
            clip.sample(Fix128::from_int(3)).bones[0].position,
            Vec3Fix::from_int(4, 0, 0)
        );
        // time = 5 → cycles.hi = 2 → t = 1 → x = 4 (`*`→`/` なら 5 − 2/2 = 4 → 末尾 8)
        assert_eq!(
            clip.sample(Fix128::from_int(5)).bones[0].position,
            Vec3Fix::from_int(4, 0, 0)
        );
        // time = duration ちょうどは wrap して先頭 (x = 0)
        assert_eq!(
            clip.sample(Fix128::from_int(2)).bones[0].position,
            Vec3Fix::ZERO
        );
        // duration + 1/2 → 1/2 → x = 2
        assert_eq!(
            clip.sample(r(5, 2)).bones[0].position,
            Vec3Fix::from_int(2, 0, 0)
        );
        // duration 直前 (1 ulp 手前) は wrap しない: x = 8 − 8 ulp (local_t = 1 − ulp、×8)
        let just_before = Fix128::from_int(2) - Fix128::from_raw(0, 1);
        let x = clip.sample(just_before).bones[0].position.x;
        assert!(x < Fix128::from_int(8));
        assert!(
            x >= Fix128::from_int(8) - Fix128::from_raw(0, 16),
            "x = {x:?}"
        );
        // duration 直後 (1 ulp 先) は wrap して先頭付近 (x ≤ 16 ulp)
        let just_after = Fix128::from_int(2) + Fix128::from_raw(0, 1);
        let x = clip.sample(just_after).bones[0].position.x;
        assert!(x <= Fix128::from_raw(0, 16), "x = {x:?}");
    }

    /// Kills 186:33 `&&`→`||`, 189:24 `>`→`==` / `>`→`<` (非 loop の clamp).
    ///
    /// 最後の keyframe (t=4) が duration (2) より先にあるので、clamp の有無が
    /// 補間位置に出る: time 3 → clamp 2 → local_t = 1/2 → x = 8 (clamp 無しなら 12)
    #[test]
    fn sample_non_looping_clamps_time_to_duration() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(2));
        clip.looping = false;
        clip.add_keyframe(0, kf(Fix128::ZERO, Vec3Fix::ZERO));
        clip.add_keyframe(0, kf(Fix128::from_int(4), Vec3Fix::from_int(16, 0, 0)));

        // time 3 > duration 2 → clamp → x = 16 · (2/4) = 8
        assert_eq!(
            clip.sample(Fix128::from_int(3)).bones[0].position,
            Vec3Fix::from_int(8, 0, 0)
        );
        // time = duration ちょうど → x = 8 (loop なら 0 に wrap するはず)
        assert_eq!(
            clip.sample(Fix128::from_int(2)).bones[0].position,
            Vec3Fix::from_int(8, 0, 0)
        );
        // 1 ulp 先も clamp → 厳密に 8
        assert_eq!(
            clip.sample(Fix128::from_int(2) + Fix128::from_raw(0, 1))
                .bones[0]
                .position,
            Vec3Fix::from_int(8, 0, 0)
        );
        // time 1 < duration → clamp されない → x = 4 (`>`→`<` なら clamp → 8)
        assert_eq!(
            clip.sample(Fix128::ONE).bones[0].position,
            Vec3Fix::from_int(4, 0, 0)
        );
        // duration 0 かつ非 loop: 全て clamp → 先頭 pose
        clip.duration = Fix128::ZERO;
        assert_eq!(
            clip.sample(Fix128::from_int(3)).bones[0].position,
            Vec3Fix::ZERO
        );
    }

    /// Kills 328:23 `>`→`>=` (snap 閾値ちょうど) と 378:56 `>`→`>=` (is_transitioning 閾値).
    #[test]
    fn update_snaps_when_diff_equals_threshold_exactly() {
        let eps = Fix128::from_ratio(1, 1000);
        let mut blender = AnimationBlender::new(1);
        blender.set_animated();
        blender.target_weight = eps; // diff = 1/1000 ちょうど
        blender.transition_speed = Fix128::ONE;

        // |diff| == 1/1000 は「遷移中ではない」
        assert!(!blender.is_transitioning());
        blender.blend_weight = eps;
        blender.target_weight = Fix128::ZERO;
        assert!(!blender.is_transitioning());
        // 1 ulp 超えると遷移中
        blender.blend_weight = eps + Fix128::from_raw(0, 1);
        assert!(blender.is_transitioning());

        // step = 1/4000 < diff: `>` なら else 枝で snap → weight = target
        blender.blend_weight = Fix128::ZERO;
        blender.target_weight = eps;
        blender.update(Fix128::from_ratio(1, 4000));
        assert_eq!(blender.blend_weight, eps, "`>=` なら 1/4000 に留まる");
    }

    /// Kills 332:38 `>`→`==` と 337:38 `<`→`==` (overshoot clamp、両方向).
    #[test]
    fn update_clamps_overshoot_to_target_in_both_directions() {
        let mut blender = AnimationBlender::new(1);
        blender.transition_speed = Fix128::from_int(2);
        // 0 → 1、step = 2 × 1 = 2 で overshoot → clamp 1
        blender.go_ragdoll();
        blender.update(Fix128::ONE);
        assert_eq!(blender.blend_weight, Fix128::ONE);
        assert_eq!(blender.mode, BlendMode::Ragdoll);

        // 1 → 0、step 2 で overshoot → clamp 0
        blender.go_animated();
        blender.update(Fix128::ONE);
        assert_eq!(blender.blend_weight, Fix128::ZERO);
        assert_eq!(blender.mode, BlendMode::Animated);

        // 部分 step は clamp されない: 0 → 3/8 (step = 2 × 3/16)
        blender.go_ragdoll();
        blender.update(r(3, 16));
        assert_eq!(blender.blend_weight, r(3, 8));
        assert_eq!(blender.mode, BlendMode::Blend);
        // 3/8 → 1 は step 1 で overshoot → 1
        blender.update(r(1, 2));
        assert_eq!(blender.blend_weight, Fix128::ONE);
    }

    /// SLERP 端点 + 閉形式の中点 / 四分点 (identity → 90° about z).
    ///
    /// t=1/2: (0, 0, sin π/8, cos π/8)、t=1/4: (0, 0, sin π/16, cos π/16)
    #[test]
    #[allow(clippy::disallowed_methods)] // closed-form f64 sin / cos are the oracle
    fn quat_slerp_identity_to_90deg_closed_form() {
        let a = QuatFix::IDENTITY;
        let b = QuatFix::from_axis_angle(Vec3Fix::from_int(0, 0, 1), Fix128::PI.half());
        let tol = 1e-12;
        assert_quat_close(quat_slerp(a, b, Fix128::ZERO), q_f64(a), tol, "t=0");
        assert_quat_close(quat_slerp(a, b, Fix128::ONE), q_f64(b), tol, "t=1");
        let pi = core::f64::consts::PI;
        assert_quat_close(
            quat_slerp(a, b, r(1, 2)),
            [0.0, 0.0, (pi / 8.0).sin(), (pi / 8.0).cos()],
            tol,
            "t=1/2 → 45°",
        );
        assert_quat_close(
            quat_slerp(a, b, r(1, 4)),
            [0.0, 0.0, (pi / 16.0).sin(), (pi / 16.0).cos()],
            tol,
            "t=1/4 → 22.5°",
        );
        assert_quat_close(
            quat_slerp(a, b, r(3, 4)),
            [0.0, 0.0, (3.0 * pi / 16.0).sin(), (3.0 * pi / 16.0).cos()],
            tol,
            "t=3/4 → 67.5°",
        );
    }

    /// Kills 388:* (dot 積の算術 8 site)、411:40 / 411:46 (1 − dot²)、425:35 (1 − t)、
    /// 426:27 / 426:42 / 427:17 / 427:32 (sin 重み)、430–433 (成分合成 24 site)、
    /// 399:12 `>`→`<` (dot 0.9 で NLERP に落ちると 1e-3 ずれる).
    ///
    /// a = (1,2,2,4)/5、b = (3,1,1,5)/6、dot = 0.9、t = 1/4 (s0 ≠ s1 なので θ 依存)
    #[test]
    fn quat_slerp_full_path_matches_textbook_oracle_at_quarter() {
        let (a, b) = (qa(), qb());
        let af = q_f64(a);
        let bf = q_f64(b);
        let tol = 1e-12;
        for (num, denom) in [(1, 4), (3, 4), (1, 8), (5, 8)] {
            let t = f64::from(num) / f64::from(denom);
            assert_quat_close(
                quat_slerp(a, b, r(i64::from(num), i64::from(denom))),
                slerp_ref(af, bf, t),
                tol,
                &format!("slerp(qa, qb, {num}/{denom})"),
            );
        }
        // 端点
        assert_quat_close(quat_slerp(a, b, Fix128::ZERO), af, tol, "t=0");
        assert_quat_close(quat_slerp(a, b, Fix128::ONE), bf, tol, "t=1");
        // 中点は (a+b) の正規化 (θ に依らない閉形式)
        assert_quat_close(
            quat_slerp(a, b, r(1, 2)),
            norm4([af[0] + bf[0], af[1] + bf[1], af[2] + bf[2], af[3] + bf[3]]),
            tol,
            "t=1/2",
        );
    }

    /// Kills 391:20 `<`→`==` / `<`→`>`、392:15 delete `-`、393:22/28/34/40 delete `-`
    /// (最短経路の b 反転、4 成分全て非零).
    #[test]
    fn quat_slerp_negative_dot_flips_b_to_shortest_path() {
        let (a, b) = (qa(), qb());
        let t = r(1, 4);
        let want = quat_slerp(a, b, t);
        // −b は同じ回転 → 反転後は同じ経路 (負数積の丸めで 1 ulp 差、1e-15 以内)
        assert_quat_close(
            quat_slerp(a, q_neg(b), t),
            q_f64(want),
            1e-15,
            "slerp(a, −b) = slerp(a, b)",
        );
        // f64 oracle とも一致
        assert_quat_close(
            quat_slerp(a, q_neg(b), t),
            slerp_ref(q_f64(a), q_f64(q_neg(b)), 0.25),
            1e-12,
            "slerp(a, −b)",
        );
        // 逆向き (b → a) も対称
        assert_quat_close(
            quat_slerp(q_neg(b), a, t),
            slerp_ref(q_f64(q_neg(b)), q_f64(a), 0.25),
            1e-12,
            "slerp(−b, a)",
        );
    }

    /// Kills 391:20 `<`→`<=` (dot == 0 ちょうどで b を反転してはいけない).
    ///
    /// a = identity、b = (1,2,2,0)/3 (180° 回転、w = 0) → 固定小数点でも dot は厳密に 0
    /// (0·b.xyz = 0、1·0 = 0)、θ = 90°:
    /// t = 1/2 → (a + b)/√2 = (1/3, 2/3, 2/3, 1)/√2、
    /// t = 1/4 → 22.5°: (sin(π/8)·(1,2,2)/3, cos(π/8))
    ///
    /// (1,2,2,4)/5 と (4,2,−2,−1)/5 のような有理直交対は truncation で dot = −2 ulp になり
    /// `Fix128::atan2(≈1, 2 ulp)` が壊れる (math.rs 側 bug、report 済) ので identity を使う
    #[test]
    #[allow(clippy::disallowed_methods)] // closed-form f64 sin / cos are the oracle
    fn quat_slerp_orthogonal_pair_does_not_flip() {
        let a = QuatFix::IDENTITY;
        let b = QuatFix::new(r(1, 3), r(2, 3), r(2, 3), Fix128::ZERO);
        assert!((a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w).is_zero());
        let s = 1.0 / 2f64.sqrt();
        assert_quat_close(
            quat_slerp(a, b, r(1, 2)),
            [s / 3.0, 2.0 * s / 3.0, 2.0 * s / 3.0, s],
            1e-12,
            "dot = 0、t = 1/2",
        );
        // `<=` なら (a − b)/√2 = (−1/3, −2/3, −2/3, 1)/√2 になる
        let pi = core::f64::consts::PI;
        let (sn, cs) = ((pi / 8.0).sin(), (pi / 8.0).cos());
        assert_quat_close(
            quat_slerp(a, b, r(1, 4)),
            [sn / 3.0, 2.0 * sn / 3.0, 2.0 * sn / 3.0, cs],
            1e-12,
            "dot = 0、t = 1/4",
        );
        assert_quat_close(
            quat_slerp(a, b, r(1, 4)),
            slerp_ref(q_f64(a), q_f64(b), 0.25),
            1e-12,
            "dot = 0、t = 1/4 (oracle)",
        );
    }

    /// Kills 399:12 `>`→`==` / `>`→`>=` (dot が閾値 999/1000 に厳密一致 → full SLERP を使う).
    ///
    /// a = identity、b = (x, 0, 0, w) で w = from_ratio(999, 1000) → dot = 1·w が厳密に閾値
    #[test]
    fn quat_slerp_dot_exactly_at_nlerp_threshold_uses_full_slerp() {
        let w = Fix128::from_ratio(999, 1000);
        let x = (Fix128::ONE - w * w).sqrt();
        let a = QuatFix::IDENTITY;
        let b = QuatFix::new(x, Fix128::ZERO, Fix128::ZERO, w);
        assert_eq!(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w, w);
        let got = quat_slerp(a, b, r(1, 4));
        // NLERP と SLERP は θ ≈ 0.0447 rad で ≈ 1.5e-6 rad ずれる、oracle は SLERP
        assert_quat_close(
            got,
            slerp_ref(q_f64(a), q_f64(b), 0.25),
            1e-12,
            "dot = 0.999",
        );
        let nl = nlerp_ref(q_f64(a), q_f64(b), 0.25);
        assert!(
            (got.x.to_f64() - nl[0]).abs() > 1e-7,
            "NLERP に落ちている (x = {}, nlerp x = {})",
            got.x.to_f64(),
            nl[0]
        );
    }

    /// Kills 399:12 `>`→`==` / `>`→`<`、400:39 `-`→`+`、402–405 (NLERP 成分算術 14 site).
    ///
    /// b = a + (1, −1, 1, −1)/1000 → dot = 1 − 0.6/1000 = 0.9994 > 0.999 → NLERP 契約
    /// `normalize((1−t)·a + t·b)`、全成分非零
    #[test]
    fn quat_slerp_near_identical_uses_normalised_lerp() {
        let a = qa();
        let d = r(1, 1000);
        let b = QuatFix::new(a.x + d, a.y - d, a.z + d, a.w - d);
        let af = q_f64(a);
        let bf = q_f64(b);
        let dot = af[0] * bf[0] + af[1] * bf[1] + af[2] * bf[2] + af[3] * bf[3];
        assert!(dot > 0.999 && dot < 1.0, "dot = {dot}");
        let tol = 1e-12;
        for (num, denom) in [(1, 4), (3, 4), (1, 8)] {
            let t = f64::from(num) / f64::from(denom);
            assert_quat_close(
                quat_slerp(a, b, r(i64::from(num), i64::from(denom))),
                nlerp_ref(af, bf, t),
                tol,
                &format!("nlerp(a, b, {num}/{denom})"),
            );
        }
        // 端点 (b は非単位なので t=1 は normalize(b))
        assert_quat_close(quat_slerp(a, b, Fix128::ZERO), af, tol, "t=0");
        assert_quat_close(quat_slerp(a, b, Fix128::ONE), norm4(bf), tol, "t=1");
        // 同一 quaternion は不変 (normalize の 1 ulp 丸めのみ)
        assert_quat_close(quat_slerp(a, a, r(1, 4)), af, 1e-15, "slerp(a, a)");
    }

    /// SkeletonPose::lerp の rotation は quat_slerp と bit-exact に一致 (line 97 経路).
    #[test]
    fn pose_lerp_rotation_goes_through_quat_slerp() {
        let a = SkeletonPose {
            bones: vec![BonePose {
                position: Vec3Fix::ZERO,
                rotation: qa(),
            }],
        };
        let b = SkeletonPose {
            bones: vec![BonePose {
                position: Vec3Fix::ZERO,
                rotation: qb(),
            }],
        };
        let t = r(1, 4);
        assert_eq!(
            SkeletonPose::lerp(&a, &b, t).bones[0].rotation,
            quat_slerp(qa(), qb(), t)
        );
        // clip.sample の rotation も同じ経路
        let mut clip = AnimationClip::new(1, Fix128::from_int(4));
        clip.looping = false;
        clip.add_keyframe(
            0,
            Keyframe {
                time: Fix128::ZERO,
                pose: BonePose {
                    position: Vec3Fix::ZERO,
                    rotation: qa(),
                },
            },
        );
        clip.add_keyframe(
            0,
            Keyframe {
                time: Fix128::from_int(4),
                pose: BonePose {
                    position: Vec3Fix::ZERO,
                    rotation: qb(),
                },
            },
        );
        assert_eq!(
            clip.sample(Fix128::ONE).bones[0].rotation,
            quat_slerp(qa(), qb(), t)
        );
    }
    /// Sampling before the first keyframe holds the first pose instead of
    /// extrapolating the first segment backwards: keys at t = 1 (2, 4, 6)
    /// and t = 3 (10, 12, 14) sampled at t = 0 gave (−2, 0, 2) before 1.2.0.
    #[test]
    fn sample_before_first_keyframe_holds_first_pose() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(4));
        clip.looping = false;
        let p0 = Vec3Fix::from_int(2, 4, 6);
        let mut k1 = kf(Fix128::from_int(3), Vec3Fix::from_int(10, 12, 14));
        k1.pose.rotation = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 2));
        clip.add_keyframe(0, kf(Fix128::ONE, p0));
        clip.add_keyframe(0, k1);
        for t in [
            Fix128::ZERO,
            Fix128::from_ratio(1, 2),
            Fix128::ONE - Fix128::from_raw(0, 1),
        ] {
            let pose = clip.sample(t);
            assert_eq!(pose.bones[0].position, p0, "t = {:?}", t.to_f64());
            assert_eq!(pose.bones[0].rotation, QuatFix::IDENTITY);
        }
        // exactly at the first key and inside the segment: interpolation as before
        assert_eq!(clip.sample(Fix128::ONE).bones[0].position, p0);
        assert_eq!(
            clip.sample(Fix128::from_int(2)).bones[0].position,
            Vec3Fix::from_int(6, 8, 10)
        );
    }
}
