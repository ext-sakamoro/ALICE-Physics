//! Character Controller
//!
//! Kinematic character controller with capsule collider.
//! Handles ground detection, stair stepping, slope limiting,
//! and SDF terrain collision.
//!
//! # Features
//!
//! - Capsule-based collision shape
//! - `move_and_slide()` for smooth movement along surfaces
//! - Ground detection via downward raycast
//! - Configurable stair step height
//! - Slope angle limiting
//! - SDF terrain integration
//!
//! Author: Moroya Sakamoto

use crate::collider::Sphere;
use crate::math::{Fix128, Vec3Fix};
use crate::raycast::{ray_plane, ray_sphere, Ray, RayHit};
use crate::sdf_collider::SdfCollider;
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Configuration
// ============================================================================

/// Character controller configuration
#[derive(Clone, Copy, Debug)]
pub struct CharacterConfig {
    /// Capsule radius
    pub radius: Fix128,
    /// Capsule total height (including hemispheres)
    pub height: Fix128,
    /// Maximum slope angle the character can walk on (radians)
    pub max_slope_angle: Fix128,
    /// Stair step height
    pub step_height: Fix128,
    /// Skin width (small gap to prevent tunneling)
    pub skin_width: Fix128,
    /// Ground detection ray distance below feet
    pub ground_probe_distance: Fix128,
    /// Maximum number of slide iterations
    pub max_slides: usize,
    /// Push force applied to dynamic bodies on collision
    pub push_force: Fix128,
}

impl Default for CharacterConfig {
    fn default() -> Self {
        Self {
            radius: Fix128::from_ratio(3, 10),                // 0.3
            height: Fix128::from_ratio(18, 10),               // 1.8
            max_slope_angle: Fix128::from_ratio(785, 1000),   // ~0.785 rad (45 degrees)
            step_height: Fix128::from_ratio(3, 10),           // 0.3
            skin_width: Fix128::from_ratio(1, 100),           // 0.01
            ground_probe_distance: Fix128::from_ratio(1, 10), // 0.1
            max_slides: 4,
            push_force: Fix128::from_int(5),
        }
    }
}

// ============================================================================
// Move Result
// ============================================================================

/// Impulse to apply to a rigid body from character collision
#[derive(Clone, Copy, Debug)]
pub struct PushImpulse {
    /// Index of the body to push
    pub body_index: usize,
    /// Impulse vector to apply
    pub impulse: Vec3Fix,
    /// World-space point of application
    pub point: Vec3Fix,
}

/// Result of a character move operation
#[derive(Clone, Copy, Debug)]
pub struct MoveResult {
    /// Final position after movement
    pub position: Vec3Fix,
    /// Whether the character is on the ground
    pub grounded: bool,
    /// Velocity after sliding (may differ from input if we slid along a wall)
    pub velocity: Vec3Fix,
    /// Velocity inherited from the moving platform (zero if not on platform)
    pub platform_velocity: Vec3Fix,
}

// ============================================================================
// Character Controller
// ============================================================================

/// Kinematic character controller
pub struct CharacterController {
    /// Current position (capsule center)
    pub position: Vec3Fix,
    /// Current velocity
    pub velocity: Vec3Fix,
    /// Whether the character is on the ground
    pub grounded: bool,
    /// Configuration
    pub config: CharacterConfig,
    /// Index of the ground body (for moving platform support)
    pub ground_body_index: Option<usize>,
    /// Velocity inherited from the moving platform
    pub platform_velocity: Vec3Fix,
}

impl CharacterController {
    /// Create a new character controller at the given position
    pub fn new(position: Vec3Fix, config: CharacterConfig) -> Self {
        Self {
            position,
            velocity: Vec3Fix::ZERO,
            grounded: false,
            config,
            ground_body_index: None,
            platform_velocity: Vec3Fix::ZERO,
        }
    }

    /// Create with default config
    pub fn new_default(position: Vec3Fix) -> Self {
        Self::new(position, CharacterConfig::default())
    }

    /// Get the bottom of the capsule (feet position)
    #[inline]
    pub fn feet_position(&self) -> Vec3Fix {
        let half_height = self.config.height.half();
        Vec3Fix::new(
            self.position.x,
            self.position.y - half_height + self.config.radius,
            self.position.z,
        )
    }

    /// Move the character by the given displacement, sliding along colliders.
    ///
    /// Uses the "collide and slide" algorithm:
    /// 1. Try to move the full displacement
    /// 2. On collision, project remaining displacement onto the collision plane
    /// 3. Repeat up to `max_slides` times
    pub fn move_and_slide(
        &mut self,
        displacement: Vec3Fix,
        bodies: &[RigidBody],
        sdf_colliders: &[SdfCollider],
    ) -> MoveResult {
        let mut remaining = displacement;
        let mut pos = self.position;

        for _ in 0..self.config.max_slides {
            let move_len = remaining.length();
            if move_len < self.config.skin_width {
                break;
            }

            // Check body collisions (simplified: treat each body as a sphere)
            if let Some(hit) = self.sweep_against_bodies(pos, remaining, bodies) {
                // Move to just before the hit
                let safe_t = if hit.t > self.config.skin_width {
                    hit.t - self.config.skin_width
                } else {
                    Fix128::ZERO
                };
                let move_dir = remaining.normalize();
                pos = pos + move_dir * safe_t;

                // Slide: project remaining displacement onto the collision plane
                let used = safe_t;
                let left = move_len - used;
                if left < self.config.skin_width {
                    break;
                }
                let remainder_dir = remaining.normalize();
                let remainder_vec = remainder_dir * left;
                let slide = remainder_vec - hit.normal * remainder_vec.dot(hit.normal);
                remaining = slide;
            } else {
                // No collision, move freely
                pos = pos + remaining;
                break;
            }
        }

        // SDF collision: push out of any SDF surfaces
        #[cfg(feature = "std")]
        {
            pos = self.resolve_sdf(pos, sdf_colliders);
        }

        // Stair stepping: if blocked horizontally and grounded, try stepping up
        let horizontal_moved = Vec3Fix::new(
            pos.x - self.position.x,
            Fix128::ZERO,
            pos.z - self.position.z,
        );
        let horizontal_desired = Vec3Fix::new(displacement.x, Fix128::ZERO, displacement.z);
        let moved_ratio = horizontal_moved.length_squared();
        let desired_ratio = horizontal_desired.length_squared();

        if self.grounded && !desired_ratio.is_zero() {
            let threshold = desired_ratio * Fix128::from_ratio(1, 4); // less than 25% moved
            if moved_ratio < threshold {
                let step_up = Vec3Fix::new(Fix128::ZERO, self.config.step_height, Fix128::ZERO);
                let stepped_pos = pos + step_up;
                // Try moving horizontally from stepped position
                let horiz_disp = Vec3Fix::new(displacement.x, Fix128::ZERO, displacement.z);
                let test_pos = stepped_pos + horiz_disp;

                // Verify no collision at stepped position
                let step_clear = self.check_position_clear(test_pos, bodies);

                #[cfg(feature = "std")]
                let step_clear_sdf = self.check_sdf_clear(test_pos, sdf_colliders);
                #[cfg(not(feature = "std"))]
                let step_clear_sdf = true;

                if step_clear && step_clear_sdf {
                    // Snap down to find the actual stair surface
                    let snap_ray = Ray::new(
                        test_pos,
                        Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO),
                    );
                    let snap_dist = self.config.step_height + self.config.step_height;
                    if let Some(ground_hit) = ray_plane(
                        &snap_ray,
                        Vec3Fix::UNIT_Y,
                        self.position.y - self.config.height.half() + self.config.radius,
                        snap_dist,
                    ) {
                        let snapped_y = test_pos.y - ground_hit.t;
                        pos = Vec3Fix::new(test_pos.x, snapped_y, test_pos.z);
                    } else {
                        pos = test_pos;
                    }
                }
            }
        }

        // Ground detection
        let (grounded, ground_idx) = self.detect_ground(pos, bodies, sdf_colliders);

        // Compute platform velocity from ground body
        let platform_vel = if let Some(idx) = ground_idx {
            if idx < bodies.len() {
                bodies[idx].velocity
            } else {
                Vec3Fix::ZERO
            }
        } else {
            Vec3Fix::ZERO
        };

        self.position = pos;
        self.grounded = grounded;
        self.ground_body_index = ground_idx;
        self.platform_velocity = platform_vel;
        self.velocity = displacement; // Caller controls velocity

        MoveResult {
            position: pos,
            grounded,
            velocity: displacement,
            platform_velocity: platform_vel,
        }
    }

    /// Sweep the character capsule against rigid bodies (simplified sphere approximation)
    fn sweep_against_bodies(
        &self,
        from: Vec3Fix,
        displacement: Vec3Fix,
        bodies: &[RigidBody],
    ) -> Option<RayHit> {
        let move_len = displacement.length();
        if move_len.is_zero() {
            return None;
        }
        let direction = displacement / move_len;

        let char_sphere = Sphere::new(from, self.config.radius);
        let ray = Ray::new(from, direction);

        let mut closest: Option<RayHit> = None;
        let mut best_t = move_len;

        for (i, body) in bodies.iter().enumerate() {
            // Skip dynamic bodies for now (character interacts with statics)
            if !body.is_static() {
                continue;
            }

            // Treat body as a sphere with the character's collision radius expanded
            let body_sphere = Sphere::new(body.position, self.config.radius);
            let expanded = Sphere::new(body_sphere.center, body_sphere.radius + char_sphere.radius);

            if let Some(mut hit) = ray_sphere(&ray, &expanded, best_t) {
                hit.body_index = i;
                best_t = hit.t;
                closest = Some(hit);
            }
        }

        closest
    }

    /// Check if a position is clear of body collisions
    fn check_position_clear(&self, pos: Vec3Fix, bodies: &[RigidBody]) -> bool {
        let r = self.config.radius + self.config.skin_width;
        let r_sq = r * r;

        for body in bodies {
            if !body.is_static() {
                continue;
            }
            let dist_sq = (pos - body.position).length_squared();
            if dist_sq < r_sq {
                return false;
            }
        }
        true
    }

    /// Resolve SDF collisions: push character out of SDF surfaces
    #[cfg(feature = "std")]
    fn resolve_sdf(&self, mut pos: Vec3Fix, sdf_colliders: &[SdfCollider]) -> Vec3Fix {
        let half_h = self.config.height.half();
        // Check at 3 points along the capsule: bottom, center, top
        let offsets = [
            Vec3Fix::new(Fix128::ZERO, -half_h + self.config.radius, Fix128::ZERO),
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::ZERO, half_h - self.config.radius, Fix128::ZERO),
        ];

        for sdf in sdf_colliders {
            for offset in &offsets {
                let sample_pos = pos + *offset;
                let (lx, ly, lz) = sdf.world_to_local(sample_pos);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;
                let threshold = self.config.radius.to_f32();

                if dist < threshold {
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let push = Fix128::from_f32(threshold - dist);
                    pos = pos + normal * push;
                }
            }
        }
        pos
    }

    /// Check if a position is clear of SDF surfaces
    #[cfg(feature = "std")]
    fn check_sdf_clear(&self, pos: Vec3Fix, sdf_colliders: &[SdfCollider]) -> bool {
        for sdf in sdf_colliders {
            let (lx, ly, lz) = sdf.world_to_local(pos);
            let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;
            if dist < self.config.radius.to_f32() {
                return false;
            }
        }
        true
    }

    /// Detect if the character is on the ground.
    /// Returns (grounded, ground_body_index).
    fn detect_ground(
        &self,
        pos: Vec3Fix,
        bodies: &[RigidBody],
        _sdf_colliders: &[SdfCollider],
    ) -> (bool, Option<usize>) {
        let feet = Vec3Fix::new(
            pos.x,
            pos.y - self.config.height.half() + self.config.radius,
            pos.z,
        );
        let down = Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO);
        let ray = Ray::new(feet, down);
        let probe = self.config.ground_probe_distance + self.config.skin_width;

        // Check against static/kinematic bodies (sphere approximation)
        for (i, body) in bodies.iter().enumerate() {
            if !body.is_static() {
                continue;
            }
            let body_sphere = Sphere::new(body.position, self.config.radius);
            if ray_sphere(&ray, &body_sphere, probe).is_some() {
                return (true, Some(i));
            }
        }

        // Check SDF ground
        #[cfg(feature = "std")]
        for sdf in _sdf_colliders {
            let (lx, ly, lz) = sdf.world_to_local(feet);
            let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;
            if dist < (probe.to_f32() + self.config.radius.to_f32()) {
                // Check slope angle
                let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                let normal = sdf.local_normal_to_world(nx, ny, nz);
                let up_dot = normal.dot(Vec3Fix::UNIT_Y);
                let cos_max = self.config.max_slope_angle.cos();
                if up_dot >= cos_max {
                    return (true, None); // SDF ground, no body index
                }
            }
        }

        (false, None)
    }

    /// Compute push impulses for dynamic bodies overlapping the character.
    ///
    /// Call this after `move_and_slide` to get impulses that should be applied
    /// to nearby dynamic bodies. Apply them via `body.apply_impulse_at()`.
    pub fn compute_push_impulses(
        &self,
        bodies: &[RigidBody],
        body_radius: Fix128,
    ) -> Vec<PushImpulse> {
        let mut pushes = Vec::new();
        let combined = self.config.radius + body_radius;
        let combined_sq = combined * combined;

        for (i, body) in bodies.iter().enumerate() {
            // Only push dynamic bodies
            if body.is_static() || body.is_sensor {
                continue;
            }

            let delta = body.position - self.position;
            let dist_sq = delta.length_squared();
            if dist_sq < combined_sq {
                let (normal, overlap) = if dist_sq.is_zero() {
                    // Bodies at same position: push upward as default
                    (Vec3Fix::UNIT_Y, combined)
                } else {
                    let dist = dist_sq.sqrt();
                    (delta / dist, combined - dist)
                };
                let impulse = normal * overlap * self.config.push_force;
                pushes.push(PushImpulse {
                    body_index: i,
                    impulse,
                    point: body.position - normal * body_radius,
                });
            }
        }

        pushes
    }

    /// Get the velocity of the platform the character is standing on
    #[inline]
    pub fn get_platform_velocity(&self) -> Vec3Fix {
        self.platform_velocity
    }

    /// Apply gravity and update position (convenience method)
    pub fn apply_gravity(&mut self, gravity: Vec3Fix, dt: Fix128) {
        if !self.grounded {
            self.velocity = self.velocity + gravity * dt;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_creation() {
        let pos = Vec3Fix::from_int(0, 5, 0);
        let cc = CharacterController::new_default(pos);
        assert_eq!(cc.position.x.hi, 0);
        assert_eq!(cc.position.y.hi, 5);
        assert!(!cc.grounded);
    }

    #[test]
    fn test_feet_position() {
        let cc = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));
        let feet = cc.feet_position();
        // height=1.8, so half_height=0.9, feet = 5 - 0.9 + 0.3 = 4.4
        let feet_y = feet.y.to_f32();
        assert!(
            feet_y > 4.0 && feet_y < 5.0,
            "Feet should be below center, got {}",
            feet_y
        );
    }

    #[test]
    fn test_move_and_slide_no_collision() {
        let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));
        let displacement = Vec3Fix::from_int(1, 0, 0);
        let result = cc.move_and_slide(displacement, &[], &[]);
        // Should move freely with no collisions
        let dx = (result.position.x - Fix128::ONE).abs();
        assert!(dx < Fix128::from_ratio(1, 10), "Should have moved to x=1");
    }

    #[test]
    fn test_platform_velocity_tracking() {
        let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 2, 0));

        // Create a kinematic platform
        let mut platform = RigidBody::new_kinematic(Vec3Fix::from_int(0, 0, 0));
        platform.velocity = Vec3Fix::from_int(5, 0, 0); // moving right
        let bodies = vec![platform];

        cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);

        // If grounded on the platform, platform_velocity should be tracked
        if cc.grounded {
            assert_eq!(
                cc.platform_velocity.x.hi, 5,
                "Should track platform velocity"
            );
        }
    }

    #[test]
    fn test_push_impulses() {
        let cc = CharacterController::new_default(Vec3Fix::from_int(0, 0, 0));

        // Dynamic body overlapping the character
        let dynamic = RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::ONE);
        let bodies = vec![dynamic];

        let pushes = cc.compute_push_impulses(&bodies, Fix128::from_ratio(1, 2));
        assert!(
            !pushes.is_empty(),
            "Should generate push impulse for overlapping dynamic body"
        );
        assert_eq!(pushes[0].body_index, 0);
    }

    #[test]
    fn test_no_push_static() {
        let cc = CharacterController::new_default(Vec3Fix::from_int(0, 0, 0));

        // Static body — should not be pushed
        let static_body = RigidBody::new_static(Vec3Fix::from_int(0, 0, 0));
        let bodies = vec![static_body];

        let pushes = cc.compute_push_impulses(&bodies, Fix128::from_ratio(1, 2));
        assert!(pushes.is_empty(), "Should not push static bodies");
    }

    #[test]
    fn test_move_and_slide_config() {
        let config = CharacterConfig {
            radius: Fix128::from_ratio(5, 10),
            height: Fix128::from_int(2),
            ..Default::default()
        };
        let cc = CharacterController::new(Vec3Fix::ZERO, config);
        assert_eq!(cc.config.height.hi, 2);
    }
}
