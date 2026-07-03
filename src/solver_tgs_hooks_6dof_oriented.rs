//! Orientation-tracking extension for the sub-stepping TGS solver.
//!
//! Adds a unit-quaternion orientation to the 6-DOF body state and
//! provides the standard quaternion integration primitive:
//!
//! ```text
//! q_new = normalize(q + 0.5 · dt · (ω × q))
//! ```
//!
//! Here `ω × q` is the Hamilton product between the pure quaternion
//! `(ω.x, ω.y, ω.z, 0)` and the current orientation `q`. Integrating
//! the orientation at the end of a sub-step and then rebuilding the
//! world-frame inertia (`R · I_local · Rᵀ`) is the usual recipe for
//! layering full 6-DOF simulation on top of the existing hooks.
//!
//! The inertia is kept diagonal for now, matching
//! [`crate::solver_tgs_hooks_6dof::Body6DofState`]; consumers that
//! need a full symmetric world-frame inertia can rebuild one before
//! each sub-step using [`crate::math::Mat3Fix`].

use crate::math::{Fix128, QuatFix, Vec3Fix};
use crate::solver_tgs::{BodyLike, ContactLike};

// ---------------------------------------------------------------------------
// Small `[Fix128; 3]` scratch layout used by the hooks family.
// ---------------------------------------------------------------------------

type Vec3 = [Fix128; 3];

const V_ZERO: Vec3 = [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO];

/// Convert a `[Fix128; 3]` triple to a [`Vec3Fix`].
#[inline]
#[must_use]
pub fn to_vec3fix(v: Vec3) -> Vec3Fix {
    Vec3Fix {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

/// Convert a [`Vec3Fix`] to a `[Fix128; 3]` triple.
#[inline]
#[must_use]
pub fn from_vec3fix(v: Vec3Fix) -> Vec3 {
    [v.x, v.y, v.z]
}

// ---------------------------------------------------------------------------
// Orientation integration primitive
// ---------------------------------------------------------------------------

/// Integrate a unit quaternion by an angular velocity `omega` over the
/// interval `dt` using the classic first-order rule
/// `q_new = normalize(q + 0.5 · dt · (ω_q × q))`.
#[must_use]
pub fn integrate_orientation(q: QuatFix, omega: Vec3, dt: Fix128) -> QuatFix {
    let omega_q = QuatFix::new(omega[0], omega[1], omega[2], Fix128::ZERO);
    let q_dot = omega_q.mul(q);
    let half_dt = Fix128::from_f32(0.5) * dt;
    let candidate = QuatFix::new(
        q.x + q_dot.x * half_dt,
        q.y + q_dot.y * half_dt,
        q.z + q_dot.z * half_dt,
        q.w + q_dot.w * half_dt,
    );
    candidate.normalize()
}

// ---------------------------------------------------------------------------
// Body state with orientation
// ---------------------------------------------------------------------------

/// A [`Body6DofState`](crate::solver_tgs_hooks_6dof::Body6DofState)
/// augmented with a unit quaternion orientation. Callers that want
/// full 6-DOF simulation typically build a shim hook that mirrors the
/// existing [`Pgs6DofHooks`](crate::solver_tgs_hooks_6dof::Pgs6DofHooks)
/// logic against this type and finishes each sub-step by
/// [`integrate_orientation`] over `end_substep`'s `sub_dt`.
#[derive(Debug, Clone, Copy)]
pub struct Body6DofOrientedState {
    pub position: Vec3,
    pub orientation: QuatFix,
    pub linear_velocity: Vec3,
    pub angular_velocity: Vec3,
    pub inv_mass: Fix128,
    /// Reciprocals of the three principal moments of inertia in the
    /// body's local frame.
    pub inv_inertia_local: Vec3,
    pub is_dynamic: bool,
    pub stable_id: u64,
}

impl Default for Body6DofOrientedState {
    fn default() -> Self {
        Self {
            position: V_ZERO,
            orientation: QuatFix::IDENTITY,
            linear_velocity: V_ZERO,
            angular_velocity: V_ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia_local: V_ZERO,
            is_dynamic: false,
            stable_id: 0,
        }
    }
}

impl BodyLike for Body6DofOrientedState {
    fn stable_id(&self) -> u64 {
        self.stable_id
    }
    fn is_dynamic(&self) -> bool {
        self.is_dynamic
    }
}

impl Body6DofOrientedState {
    /// Advance the body's linear position, angular velocity is left
    /// intact, and the orientation is integrated by
    /// [`integrate_orientation`].
    pub fn advance(&mut self, sub_dt: Fix128) {
        if !self.is_dynamic {
            return;
        }
        // Linear position: p += v · dt.
        self.position = [
            self.position[0] + self.linear_velocity[0] * sub_dt,
            self.position[1] + self.linear_velocity[1] * sub_dt,
            self.position[2] + self.linear_velocity[2] * sub_dt,
        ];
        // Orientation: standard quaternion integration.
        self.orientation = integrate_orientation(self.orientation, self.angular_velocity, sub_dt);
    }

    /// Rotate a body-local vector into the world frame.
    #[must_use]
    pub fn local_to_world(&self, local: Vec3) -> Vec3 {
        from_vec3fix(self.orientation.rotate_vec(to_vec3fix(local)))
    }
}

// ---------------------------------------------------------------------------
// Contact (mirrors the diagonal-inertia contact, kept independent so
// callers can migrate to orientation tracking incrementally).
// ---------------------------------------------------------------------------

/// A pairwise contact for oriented bodies.
#[derive(Debug, Clone, Copy)]
pub struct ContactOriented {
    pub body_a: usize,
    pub body_b: usize,
    pub stable_id: u64,
    pub normal: Vec3,
    pub tangent1: Vec3,
    pub tangent2: Vec3,
    pub r_a: Vec3,
    pub r_b: Vec3,
    pub penetration: Fix128,
    pub friction: Fix128,
    pub restitution: Fix128,
    pub accum_normal: Fix128,
    pub accum_tangent1: Fix128,
    pub accum_tangent2: Fix128,
}

impl ContactLike for ContactOriented {
    fn body_a(&self) -> usize {
        self.body_a
    }
    fn body_b(&self) -> usize {
        self.body_b
    }
    fn stable_id(&self) -> u64 {
        self.stable_id
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_with_zero_omega_stays_identity() {
        let q = integrate_orientation(QuatFix::IDENTITY, V_ZERO, Fix128::from_f32(1.0 / 60.0));
        // Should remain the identity quaternion within Fix128 precision.
        assert_eq!(q.x, Fix128::ZERO);
        assert_eq!(q.y, Fix128::ZERO);
        assert_eq!(q.z, Fix128::ZERO);
        assert_eq!(q.w, Fix128::ONE);
    }

    #[test]
    fn spin_about_y_produces_positive_y_component() {
        // ω = 1 rad/s about +Y, integrated over a small step, gives a
        // quaternion with a small positive y component and w ≈ 1.
        let omega = [Fix128::ZERO, Fix128::ONE, Fix128::ZERO];
        let q = integrate_orientation(
            QuatFix::IDENTITY,
            omega,
            Fix128::from_f32(0.02), // 20 ms
        );
        assert!(q.y.to_f32() > 0.0, "y component should be positive");
        // Unit-length invariant.
        let mag = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).to_f32();
        assert!((mag - 1.0).abs() < 1e-3, "quaternion must stay unit-length");
    }

    #[test]
    fn integrate_orientation_is_bit_perfect_deterministic() {
        let q = QuatFix::IDENTITY;
        let omega = [
            Fix128::from_f32(0.3),
            Fix128::from_f32(-0.5),
            Fix128::from_f32(0.7),
        ];
        let dt = Fix128::from_f32(1.0 / 60.0);
        let a = integrate_orientation(q, omega, dt);
        let b = integrate_orientation(q, omega, dt);
        assert_eq!(a.x, b.x);
        assert_eq!(a.y, b.y);
        assert_eq!(a.z, b.z);
        assert_eq!(a.w, b.w);
    }

    #[test]
    fn advance_moves_position_and_rotates() {
        let mut body = Body6DofOrientedState {
            is_dynamic: true,
            linear_velocity: [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            angular_velocity: [Fix128::ZERO, Fix128::from_f32(0.5), Fix128::ZERO],
            inv_mass: Fix128::ONE,
            inv_inertia_local: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            stable_id: 1,
            ..Default::default()
        };
        let start_pos = body.position;
        let start_q = body.orientation;
        body.advance(Fix128::from_f32(1.0 / 60.0));
        assert!(
            body.position[0].to_f32() > start_pos[0].to_f32(),
            "position should advance along +X"
        );
        assert!(
            body.orientation.y.to_f32() > start_q.y.to_f32(),
            "orientation should pick up +Y component"
        );
    }

    #[test]
    fn static_body_advance_is_a_noop() {
        let mut body = Body6DofOrientedState {
            linear_velocity: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            angular_velocity: [Fix128::ONE, Fix128::ONE, Fix128::ONE],
            is_dynamic: false,
            ..Default::default()
        };
        let snapshot = body;
        body.advance(Fix128::from_f32(1.0 / 60.0));
        assert_eq!(body.position, snapshot.position);
        assert_eq!(body.orientation.x, snapshot.orientation.x);
        assert_eq!(body.orientation.w, snapshot.orientation.w);
    }

    #[test]
    fn local_to_world_uses_the_orientation() {
        let mut body = Body6DofOrientedState {
            is_dynamic: true,
            stable_id: 1,
            ..Default::default()
        };
        // Rotate 90° about +Y using the axis-angle helper.
        body.orientation =
            QuatFix::from_axis_angle(Vec3Fix::from_f32(0.0, 1.0, 0.0), Fix128::from_f32(1.5708));
        // +X in the body frame becomes ≈ -Z in the world frame after a
        // 90° rotation about +Y.
        let world = body.local_to_world([Fix128::ONE, Fix128::ZERO, Fix128::ZERO]);
        assert!(world[2].to_f32() < -0.9, "expected ≈ -Z, got {:?}", world);
    }
}
