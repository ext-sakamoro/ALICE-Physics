//! Ragdoll Animation Blending
//!
//! Blends between keyframe animation and physics ragdoll simulation.
//! Extends neural.rs RagdollController with smooth transitions.
//!
//! # Modes
//!
//! - **Animated**: Pure keyframe animation (physics inactive)
//! - **Ragdoll**: Pure physics simulation (animation inactive)
//! - **Blend**: Weighted mix of animation and physics
//! - **Powered**: Physics with motor targets from animation
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix, QuatFix};

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
#[derive(Clone, Copy, Debug)]
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
    pub fn new(num_bones: usize) -> Self {
        Self {
            bones: vec![BonePose::default(); num_bones],
        }
    }

    /// Number of bones
    #[inline]
    pub fn bone_count(&self) -> usize {
        self.bones.len()
    }

    /// Lerp between two poses
    pub fn lerp(a: &SkeletonPose, b: &SkeletonPose, t: Fix128) -> SkeletonPose {
        let n = a.bones.len().min(b.bones.len());
        let one_minus_t = Fix128::ONE - t;

        let bones: Vec<BonePose> = (0..n)
            .map(|i| {
                BonePose {
                    position: Vec3Fix::new(
                        a.bones[i].position.x * one_minus_t + b.bones[i].position.x * t,
                        a.bones[i].position.y * one_minus_t + b.bones[i].position.y * t,
                        a.bones[i].position.z * one_minus_t + b.bones[i].position.z * t,
                    ),
                    rotation: quat_slerp(a.bones[i].rotation, b.bones[i].rotation, t),
                }
            })
            .collect();

        SkeletonPose { bones }
    }
}

// ============================================================================
// Animation Clip
// ============================================================================

/// Keyframe for a single bone
#[derive(Clone, Copy, Debug)]
pub struct Keyframe {
    /// Time in seconds
    pub time: Fix128,
    /// Bone pose at this time
    pub pose: BonePose,
}

/// Animation clip (sequence of keyframes per bone)
#[derive(Clone, Debug)]
pub struct AnimationClip {
    /// Clip name
    pub name: Vec<u8>,
    /// Duration in seconds
    pub duration: Fix128,
    /// Keyframes per bone: keyframes[bone_idx] = sorted list
    pub keyframes: Vec<Vec<Keyframe>>,
    /// Whether clip loops
    pub looping: bool,
}

impl AnimationClip {
    /// Create empty clip
    pub fn new(num_bones: usize, duration: Fix128) -> Self {
        Self {
            name: Vec::new(),
            duration,
            keyframes: vec![Vec::new(); num_bones],
            looping: true,
        }
    }

    /// Add keyframe for a bone
    pub fn add_keyframe(&mut self, bone_idx: usize, keyframe: Keyframe) {
        if bone_idx < self.keyframes.len() {
            self.keyframes[bone_idx].push(keyframe);
            self.keyframes[bone_idx].sort_by(|a, b| a.time.cmp(&b.time));
        }
    }

    /// Sample pose at given time
    pub fn sample(&self, time: Fix128) -> SkeletonPose {
        let t = if self.looping && !self.duration.is_zero() {
            // Modular time
            let cycles = time / self.duration;
            time - self.duration * Fix128::from_int(cycles.hi)
        } else {
            if time > self.duration { self.duration } else { time }
        };

        let mut pose = SkeletonPose::new(self.keyframes.len());

        for (bone_idx, bone_keyframes) in self.keyframes.iter().enumerate() {
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
                        prev.pose.position.x + (next.pose.position.x - prev.pose.position.x) * local_t,
                        prev.pose.position.y + (next.pose.position.y - prev.pose.position.y) * local_t,
                        prev.pose.position.z + (next.pose.position.z - prev.pose.position.z) * local_t,
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
                self.output_pose = SkeletonPose::lerp(
                    &self.animation_pose,
                    &self.physics_pose,
                    self.blend_weight,
                );
            }
        }
    }

    /// Get motor targets for powered mode (animation poses as motor goals)
    pub fn get_motor_targets(&self) -> &SkeletonPose {
        &self.animation_pose
    }

    /// Check if currently transitioning
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
        ).normalize();
    }

    // Full SLERP
    let theta = dot.atan(); // acos approximation: atan(sqrt(1-d^2)/d)
    let sin_theta = theta.sin();

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
    ).normalize()
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
        for _ in 0..120 { // 2 seconds at 60fps
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
        assert!((x - 5.0).abs() < 0.1, "Midpoint should be at x=5, got {}", x);
    }

    #[test]
    fn test_animation_clip() {
        let mut clip = AnimationClip::new(1, Fix128::from_int(2));
        clip.add_keyframe(0, Keyframe {
            time: Fix128::ZERO,
            pose: BonePose { position: Vec3Fix::ZERO, rotation: QuatFix::IDENTITY },
        });
        clip.add_keyframe(0, Keyframe {
            time: Fix128::from_int(2),
            pose: BonePose { position: Vec3Fix::from_int(10, 0, 0), rotation: QuatFix::IDENTITY },
        });

        let pose = clip.sample(Fix128::ONE); // t=1s, midpoint
        let x = pose.bones[0].position.x.to_f32();
        assert!((x - 5.0).abs() < 0.5, "At t=1s, x should be ~5, got {}", x);
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
}
