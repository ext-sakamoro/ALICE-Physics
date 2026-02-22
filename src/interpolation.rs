//! Substep Interpolation Output
//!
//! Provides smooth rendering positions between physics substeps.
//! Physics runs at fixed timestep; rendering interpolates between
//! the previous and current physics state using alpha blending.
//!
//! # Usage
//!
//! ```rust,ignore
//! let alpha = accumulator / fixed_dt; // 0.0..1.0
//! let render_pos = interpolation::lerp_state(&prev_state, &current_state, alpha);
//! ```

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Snapshot of a single body's render-relevant state
#[derive(Clone, Copy, Debug)]
pub struct BodySnapshot {
    /// Position
    pub position: Vec3Fix,
    /// Rotation
    pub rotation: QuatFix,
    /// Linear velocity (for extrapolation)
    pub velocity: Vec3Fix,
}

impl BodySnapshot {
    /// Capture snapshot from a rigid body
    #[inline]
    pub fn from_body(body: &RigidBody) -> Self {
        Self {
            position: body.position,
            rotation: body.rotation,
            velocity: body.velocity,
        }
    }
}

/// World snapshot for interpolation
#[derive(Clone, Debug)]
pub struct WorldSnapshot {
    /// Per-body snapshots
    pub bodies: Vec<BodySnapshot>,
}

impl WorldSnapshot {
    /// Capture snapshot from physics world
    pub fn capture(world: &crate::solver::PhysicsWorld) -> Self {
        Self {
            bodies: world.bodies.iter().map(BodySnapshot::from_body).collect(),
        }
    }

    /// Number of bodies in snapshot
    #[inline]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }
}

/// Interpolation state manager
///
/// Stores two snapshots (previous and current) and provides
/// alpha-blended positions/rotations for rendering.
pub struct InterpolationState {
    /// Previous physics state
    pub prev: WorldSnapshot,
    /// Current physics state
    pub current: WorldSnapshot,
}

impl InterpolationState {
    /// Create from two snapshots
    pub fn new(prev: WorldSnapshot, current: WorldSnapshot) -> Self {
        Self { prev, current }
    }

    /// Create empty
    pub fn empty() -> Self {
        Self {
            prev: WorldSnapshot { bodies: Vec::new() },
            current: WorldSnapshot { bodies: Vec::new() },
        }
    }

    /// Push new physics state (current becomes prev)
    pub fn push(&mut self, new_state: WorldSnapshot) {
        self.prev = core::mem::replace(&mut self.current, new_state);
    }

    /// Capture and push from physics world
    pub fn capture_and_push(&mut self, world: &crate::solver::PhysicsWorld) {
        let new_state = WorldSnapshot::capture(world);
        self.push(new_state);
    }

    /// Get interpolated position for a body
    ///
    /// `alpha` is in [0, 1]: 0 = previous state, 1 = current state
    pub fn interpolate_position(&self, body_idx: usize, alpha: Fix128) -> Vec3Fix {
        if body_idx >= self.current.bodies.len() || body_idx >= self.prev.bodies.len() {
            return Vec3Fix::ZERO;
        }

        let prev = self.prev.bodies[body_idx].position;
        let curr = self.current.bodies[body_idx].position;

        lerp_vec3(prev, curr, alpha)
    }

    /// Get interpolated rotation for a body (SLERP)
    pub fn interpolate_rotation(&self, body_idx: usize, alpha: Fix128) -> QuatFix {
        if body_idx >= self.current.bodies.len() || body_idx >= self.prev.bodies.len() {
            return QuatFix::IDENTITY;
        }

        let prev = self.prev.bodies[body_idx].rotation;
        let curr = self.current.bodies[body_idx].rotation;

        slerp(prev, curr, alpha)
    }

    /// Get both interpolated position and rotation
    pub fn interpolate(&self, body_idx: usize, alpha: Fix128) -> (Vec3Fix, QuatFix) {
        (
            self.interpolate_position(body_idx, alpha),
            self.interpolate_rotation(body_idx, alpha),
        )
    }

    /// Get all interpolated transforms
    pub fn interpolate_all(&self, alpha: Fix128) -> Vec<(Vec3Fix, QuatFix)> {
        let count = self.current.bodies.len().min(self.prev.bodies.len());
        (0..count).map(|i| self.interpolate(i, alpha)).collect()
    }

    /// Number of bodies available for interpolation
    pub fn body_count(&self) -> usize {
        self.current.bodies.len().min(self.prev.bodies.len())
    }
}

/// Linear interpolation for Vec3Fix
#[inline]
pub fn lerp_vec3(a: Vec3Fix, b: Vec3Fix, t: Fix128) -> Vec3Fix {
    let one_minus_t = Fix128::ONE - t;
    Vec3Fix::new(
        a.x * one_minus_t + b.x * t,
        a.y * one_minus_t + b.y * t,
        a.z * one_minus_t + b.z * t,
    )
}

/// Linear interpolation for Fix128
#[inline]
pub fn lerp_fix128(a: Fix128, b: Fix128, t: Fix128) -> Fix128 {
    a * (Fix128::ONE - t) + b * t
}

/// Spherical linear interpolation for quaternions
///
/// Uses NLERP (normalized linear interpolation) which is deterministic
/// and provides near-identical results to SLERP for interpolation.
/// NLERP is preferred for fixed-point as it avoids acos/sin.
pub fn slerp(a: QuatFix, b: QuatFix, t: Fix128) -> QuatFix {
    // Compute dot product
    let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

    // Ensure shortest path
    let b = if dot < Fix128::ZERO {
        QuatFix::new(-b.x, -b.y, -b.z, -b.w)
    } else {
        b
    };

    // Use NLERP for deterministic fixed-point interpolation
    nlerp(a, b, t)
}

/// Normalized linear interpolation (NLERP) — faster approximate SLERP
fn nlerp(a: QuatFix, b: QuatFix, t: Fix128) -> QuatFix {
    let one_minus_t = Fix128::ONE - t;
    let result = QuatFix::new(
        a.x * one_minus_t + b.x * t,
        a.y * one_minus_t + b.y * t,
        a.z * one_minus_t + b.z * t,
        a.w * one_minus_t + b.w * t,
    );
    let len_sq =
        result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w;
    if len_sq.is_zero() {
        return QuatFix::IDENTITY;
    }
    result.normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_vec3() {
        let a = Vec3Fix::from_int(0, 0, 0);
        let b = Vec3Fix::from_int(10, 20, 30);
        let half = Fix128::from_ratio(5, 10);

        let result = lerp_vec3(a, b, half);
        assert_eq!(result.x.hi, 5);
        assert_eq!(result.y.hi, 10);
        assert_eq!(result.z.hi, 15);
    }

    #[test]
    fn test_lerp_endpoints() {
        let a = Vec3Fix::from_int(1, 2, 3);
        let b = Vec3Fix::from_int(4, 5, 6);

        let at_zero = lerp_vec3(a, b, Fix128::ZERO);
        assert_eq!(at_zero.x.hi, 1);

        let at_one = lerp_vec3(a, b, Fix128::ONE);
        assert_eq!(at_one.x.hi, 4);
    }

    #[test]
    fn test_slerp_identity() {
        let q = QuatFix::IDENTITY;
        let result = slerp(q, q, Fix128::from_ratio(5, 10));
        // Should still be approximately identity
        let dot = result.w;
        assert!(dot > Fix128::from_ratio(99, 100));
    }

    #[test]
    fn test_nlerp() {
        let a = QuatFix::IDENTITY;
        let b = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::from_ratio(1, 10));

        let result = nlerp(a, b, Fix128::from_ratio(5, 10));
        // Result should be between a and b
        let len_sq =
            result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w;
        // Should be normalized (length ~= 1)
        let error = (len_sq - Fix128::ONE).abs();
        assert!(error < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn test_interpolation_state() {
        use crate::solver::{PhysicsConfig, PhysicsWorld};

        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 10, 0),
            Fix128::ONE,
        ));

        let snap1 = WorldSnapshot::capture(&world);

        // Simulate
        world.step(Fix128::from_ratio(1, 60));
        let snap2 = WorldSnapshot::capture(&world);

        let interp = InterpolationState::new(snap1, snap2);
        let half = Fix128::from_ratio(5, 10);

        let pos = interp.interpolate_position(0, half);
        // Should be between initial and simulated position
        assert!(pos.y < Fix128::from_int(10));
        assert!(pos.y > Fix128::from_int(-100)); // Sanity check
    }

    #[test]
    fn test_world_snapshot() {
        use crate::solver::{PhysicsConfig, PhysicsWorld};

        let config = PhysicsConfig::default();
        let mut world = PhysicsWorld::new(config);
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(1, 2, 3),
            Fix128::ONE,
        ));
        world.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        let snap = WorldSnapshot::capture(&world);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap.bodies[0].position.x.hi, 1);
    }
}
