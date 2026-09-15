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
}
