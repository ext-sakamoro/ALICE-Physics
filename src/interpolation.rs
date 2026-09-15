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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    #[must_use]
    pub const fn from_body(body: &RigidBody) -> Self {
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
    #[must_use]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
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
    #[must_use]
    pub const fn new(prev: WorldSnapshot, current: WorldSnapshot) -> Self {
        Self { prev, current }
    }

    /// Create empty
    #[must_use]
    pub const fn empty() -> Self {
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
    #[must_use]
    pub fn interpolate_position(&self, body_idx: usize, alpha: Fix128) -> Vec3Fix {
        if body_idx >= self.current.bodies.len() || body_idx >= self.prev.bodies.len() {
            return Vec3Fix::ZERO;
        }

        let prev = self.prev.bodies[body_idx].position;
        let curr = self.current.bodies[body_idx].position;

        lerp_vec3(prev, curr, alpha)
    }

    /// Get interpolated rotation for a body (SLERP)
    #[must_use]
    pub fn interpolate_rotation(&self, body_idx: usize, alpha: Fix128) -> QuatFix {
        if body_idx >= self.current.bodies.len() || body_idx >= self.prev.bodies.len() {
            return QuatFix::IDENTITY;
        }

        let prev = self.prev.bodies[body_idx].rotation;
        let curr = self.current.bodies[body_idx].rotation;

        slerp(prev, curr, alpha)
    }

    /// Get both interpolated position and rotation
    #[must_use]
    pub fn interpolate(&self, body_idx: usize, alpha: Fix128) -> (Vec3Fix, QuatFix) {
        (
            self.interpolate_position(body_idx, alpha),
            self.interpolate_rotation(body_idx, alpha),
        )
    }

    /// Get all interpolated transforms
    #[must_use]
    pub fn interpolate_all(&self, alpha: Fix128) -> Vec<(Vec3Fix, QuatFix)> {
        let count = self.current.bodies.len().min(self.prev.bodies.len());
        (0..count).map(|i| self.interpolate(i, alpha)).collect()
    }

    /// Number of bodies available for interpolation
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.current.bodies.len().min(self.prev.bodies.len())
    }
}

/// Linear interpolation for `Vec3Fix`
#[inline]
#[must_use]
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
#[must_use]
pub fn lerp_fix128(a: Fix128, b: Fix128, t: Fix128) -> Fix128 {
    a * (Fix128::ONE - t) + b * t
}

/// Spherical linear interpolation for quaternions
///
/// Uses NLERP (normalized linear interpolation) which is deterministic
/// and provides near-identical results to SLERP for interpolation.
/// NLERP is preferred for fixed-point as it avoids acos/sin.
#[must_use]
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

    #[test]
    fn lerp_fix128_is_affine_in_t_with_exact_dyadic_results() {
        let a = Fix128::from_int(-4);
        let b = Fix128::from_int(12);
        // 端点は正確に一致
        assert_eq!(lerp_fix128(a, b, Fix128::ZERO), a);
        assert_eq!(lerp_fix128(a, b, Fix128::ONE), b);
        // 2 冪の t は exact: 1/2 → 4、1/4 → 0、3/4 → 8
        assert_eq!(
            lerp_fix128(a, b, Fix128::from_ratio(1, 2)),
            Fix128::from_int(4)
        );
        assert_eq!(lerp_fix128(a, b, Fix128::from_ratio(1, 4)), Fix128::ZERO);
        assert_eq!(
            lerp_fix128(a, b, Fix128::from_ratio(3, 4)),
            Fix128::from_int(8)
        );
        // a == b なら t に依らず a
        assert_eq!(lerp_fix128(b, b, Fix128::from_ratio(3, 7)), b);
        // 対称性: lerp(a, b, t) == lerp(b, a, 1 - t)
        let t = Fix128::from_ratio(3, 8);
        assert_eq!(lerp_fix128(a, b, t), lerp_fix128(b, a, Fix128::ONE - t));
        // lerp_vec3 は成分毎の lerp_fix128
        let va = Vec3Fix::new(a, Fix128::ZERO, b);
        let vb = Vec3Fix::new(b, Fix128::from_int(8), a);
        let v = lerp_vec3(va, vb, t);
        assert_eq!(v.x, lerp_fix128(a, b, t));
        assert_eq!(v.y, lerp_fix128(Fix128::ZERO, Fix128::from_int(8), t));
        assert_eq!(v.z, lerp_fix128(b, a, t));
        assert_eq!(v.y, Fix128::from_int(3));
        // 外挿 (t > 1) も同じ式: t = 2 → a + 2(b - a) = 28
        assert_eq!(lerp_fix128(a, b, Fix128::from_int(2)), Fix128::from_int(28));
    }

    #[test]
    fn capture_and_push_shifts_current_into_prev_and_interpolate_all_gives_midpoints() {
        use crate::solver::{PhysicsConfig, PhysicsWorld};

        // 重力なし、速度一定 → 1 step (dt 1/4、damping 1) で位置が exact に v/4 進む
        let config = PhysicsConfig {
            gravity: Vec3Fix::ZERO,
            damping: Fix128::ONE,
            ..PhysicsConfig::default()
        };
        let mut world = PhysicsWorld::new(config);
        world.add_body(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 0, 0), Fix128::ONE)
                .with_velocity(Vec3Fix::from_int(8, 0, 0)),
        );
        world.add_body(
            RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE)
                .with_velocity(Vec3Fix::from_int(0, -4, 16)),
        );
        world.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 5, 5)));

        let mut interp = InterpolationState::empty();
        assert_eq!(interp.body_count(), 0);
        assert!(interp.interpolate_all(Fix128::from_ratio(1, 2)).is_empty());

        // 1 回目: current だけ埋まり prev は空 → 補間対象 0
        interp.capture_and_push(&world);
        assert_eq!(interp.current.len(), 3);
        assert!(interp.prev.is_empty());
        assert_eq!(interp.body_count(), 0);
        assert!(interp.interpolate_all(Fix128::from_ratio(1, 2)).is_empty());
        let first = interp.current.clone();

        // step して 2 回目: 旧 current が prev へ、新 capture が current
        world.step(Fix128::from_ratio(1, 4));
        interp.capture_and_push(&world);
        assert_eq!(interp.prev.bodies, first.bodies);
        assert_eq!(
            interp.current.bodies[0].position,
            Vec3Fix::from_int(2, 0, 0)
        );
        assert_eq!(
            interp.current.bodies[1].position,
            Vec3Fix::from_int(0, 9, 4)
        );
        assert_eq!(
            interp.current.bodies[2].position,
            Vec3Fix::from_int(5, 5, 5)
        );
        assert_eq!(interp.body_count(), 3);

        // alpha 1/2 → 中点、alpha 0 → prev、alpha 1 → current
        let mid = interp.interpolate_all(Fix128::from_ratio(1, 2));
        assert_eq!(mid.len(), 3);
        assert_eq!(mid[0].0, Vec3Fix::from_int(1, 0, 0));
        assert_eq!(
            mid[1].0,
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(19, 2), Fix128::from_int(2))
        );
        assert_eq!(mid[2].0, Vec3Fix::from_int(5, 5, 5));
        for (i, (p, q)) in mid.iter().enumerate() {
            assert_eq!(*p, interp.interpolate_position(i, Fix128::from_ratio(1, 2)));
            // 回転は両 snapshot とも identity → identity
            assert_eq!(*q, QuatFix::IDENTITY);
        }
        let at_prev = interp.interpolate_all(Fix128::ZERO);
        let at_curr = interp.interpolate_all(Fix128::ONE);
        for i in 0..3 {
            assert_eq!(at_prev[i].0, interp.prev.bodies[i].position);
            assert_eq!(at_curr[i].0, interp.current.bodies[i].position);
        }

        // body を追加してもう一度 push: 補間数は prev/current の小さい方 (3)
        world.add_body(RigidBody::new_static(Vec3Fix::from_int(9, 9, 9)));
        interp.capture_and_push(&world);
        assert_eq!(interp.current.len(), 4);
        assert_eq!(interp.prev.len(), 3);
        assert_eq!(interp.body_count(), 3);
        assert_eq!(interp.interpolate_all(Fix128::from_ratio(1, 2)).len(), 3);
        // 範囲外 index は ZERO / IDENTITY
        assert_eq!(
            interp.interpolate(3, Fix128::from_ratio(1, 2)),
            (Vec3Fix::ZERO, QuatFix::IDENTITY)
        );
    }
}
