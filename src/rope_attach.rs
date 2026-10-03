//! Rope Attachment to Rigid Bodies
//!
//! Provides XPBD-style constraints that attach rope particles to rigid body
//! anchor points. Supports compliant (soft) attachments and breakable
//! connections with configurable break force thresholds.
//!
//! # Features
//!
//! - Attach any rope particle to a rigid body local anchor point
//! - Configurable compliance for soft/hard attachments
//! - Optional break force for destructible connections
//! - Returns indices of broken attachments each solve step

use crate::math::{Fix128, Vec3Fix};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Rope Attachment
// ============================================================================

/// Attachment constraint connecting a rope particle to a rigid body anchor.
#[derive(Clone, Debug)]
pub struct RopeAttachment {
    /// Index of the rope particle
    pub rope_particle: usize,
    /// Index of the rigid body in the world
    pub body_index: usize,
    /// Anchor point in the body's local space
    pub local_anchor: Vec3Fix,
    /// Constraint compliance (0 = rigid, higher = softer)
    pub compliance: Fix128,
    /// If set, the attachment breaks when the constraint force exceeds this value
    pub break_force: Option<Fix128>,
}

impl RopeAttachment {
    /// Create a new rigid attachment (zero compliance, unbreakable).
    #[must_use]
    pub const fn new(rope_particle: usize, body_index: usize, local_anchor: Vec3Fix) -> Self {
        Self {
            rope_particle,
            body_index,
            local_anchor,
            compliance: Fix128::ZERO,
            break_force: None,
        }
    }

    /// Create an attachment with a break force threshold.
    #[must_use]
    pub const fn with_break_force(
        rope_particle: usize,
        body_index: usize,
        local_anchor: Vec3Fix,
        break_force: Fix128,
    ) -> Self {
        Self {
            rope_particle,
            body_index,
            local_anchor,
            compliance: Fix128::ZERO,
            break_force: Some(break_force),
        }
    }

    /// Set compliance and return self (builder pattern).
    #[must_use]
    pub const fn compliance(mut self, c: Fix128) -> Self {
        self.compliance = c;
        self
    }
}

// ============================================================================
// Solver
// ============================================================================

/// Result of evaluating one attachment against the current rope / body state.
enum AttachmentOutcome {
    /// Nothing to do (coincident, zero weight sum)
    Skip,
    /// Constraint force exceeded the break threshold
    Break,
    /// Move: unit direction rope -> anchor, correction magnitude `lambda`, and the body weight
    Move {
        direction: Vec3Fix,
        lambda: Fix128,
        w_body: Fix128,
    },
}

/// Evaluate one attachment (shared by the one-way and two-way solvers).
fn evaluate_attachment(
    att: &RopeAttachment,
    rope_pos: Vec3Fix,
    body: &RigidBody,
    dt_sq: Fix128,
) -> AttachmentOutcome {
    // Transform local anchor to world space
    let world_anchor = body.position + body.rotation.rotate_vec(att.local_anchor);

    // Constraint error: distance between rope particle and anchor
    let delta = rope_pos - world_anchor;
    let dist_sq = delta.dot(delta);

    if dist_sq.is_zero() {
        return AttachmentOutcome::Skip;
    }

    let dist = dist_sq.sqrt();

    // Effective compliance: alpha = compliance / dt^2
    let alpha = if dt_sq.is_zero() {
        Fix128::ZERO
    } else {
        att.compliance / dt_sq
    };

    // Inverse masses
    let w_rope = Fix128::ONE; // rope particles have unit mass by default
    let w_body = body.inv_mass;
    let w_sum = w_rope + w_body + alpha;

    if w_sum.is_zero() {
        return AttachmentOutcome::Skip;
    }

    // Constraint correction magnitude
    let lambda = dist / w_sum;

    // Check break force
    let force_estimate = lambda / dt_sq;
    if let Some(max_force) = att.break_force {
        if force_estimate > max_force {
            return AttachmentOutcome::Break;
        }
    }

    AttachmentOutcome::Move {
        direction: delta / dist,
        lambda,
        w_body,
    }
}

/// Solve rope-to-body attachment constraints, moving only the rope particles.
///
/// For each attachment, projects the rope particle toward the body's anchor
/// point (transformed to world space). The rope particle takes the share
/// `w_rope / (w_rope + w_body + alpha)` of the gap (`w_rope = 1`). **The body is
/// not modified** (`bodies` is read-only): for a dynamic body the remaining share
/// `w_body / (w_rope + w_body + alpha)` of the correction is discarded, so there
/// is no action-reaction on the body. Use [`solve_rope_attachments_two_way`] when the
/// body must receive its share.
///
/// Returns a list of attachment indices that broke during this solve step
/// (constraint force exceeded the break threshold). Callers should remove
/// these from the attachment list.
///
/// # Arguments
///
/// - `attachments`: Active attachment constraints
/// - `rope_positions`: Mutable rope particle positions
/// - `rope_velocities`: Mutable rope particle velocities
/// - `bodies`: Rigid body array (positions and rotations used for anchor transforms)
/// - `dt`: Time step
pub fn solve_rope_attachments(
    attachments: &[RopeAttachment],
    rope_positions: &mut [Vec3Fix],
    rope_velocities: &mut [Vec3Fix],
    bodies: &[RigidBody],
    dt: Fix128,
) -> Vec<Option<usize>> {
    if dt.is_zero() {
        return Vec::new();
    }

    let dt_sq = dt * dt;
    let mut broken = Vec::new();

    for (ai, att) in attachments.iter().enumerate() {
        // Validate indices
        if att.rope_particle >= rope_positions.len() || att.body_index >= bodies.len() {
            continue;
        }

        let rope_pos = rope_positions[att.rope_particle];
        match evaluate_attachment(att, rope_pos, &bodies[att.body_index], dt_sq) {
            AttachmentOutcome::Skip => {}
            AttachmentOutcome::Break => broken.push(Some(ai)),
            AttachmentOutcome::Move {
                direction, lambda, ..
            } => {
                // Apply corrections (rope weight is 1)
                let rope_correction = direction * lambda;
                rope_positions[att.rope_particle] = rope_pos - rope_correction;

                // Update rope velocity to reflect position change
                let vel_change = rope_correction / dt;
                rope_velocities[att.rope_particle] =
                    rope_velocities[att.rope_particle] - vel_change;
            }
        }
    }

    broken
}

/// Two-way variant of [`solve_rope_attachments`]: the body also receives its share.
///
/// With gap `C = |rope - anchor|`, unit direction `n` from the anchor to the rope
/// particle, `w_rope = 1`, `w_b = body.inv_mass` and `alpha = compliance / dt^2`:
///
/// - rope particle moves by `-n * C * w_rope / (w_rope + w_b + alpha)`
/// - body position moves by `+n * C * w_b / (w_rope + w_b + alpha)`
///
/// so that `m_rope * dx_rope + m_body * dx_body = 0` for a rigid (`alpha = 0`)
/// dynamic attachment (momentum is conserved). Only the body **position** is
/// changed (translation of the centre of mass; the anchor offset's torque is not
/// applied) and the body velocity is left to the solver to derive from
/// `prev_position`. Static / kinematic bodies have `inv_mass = 0` and are not moved,
/// so the result then matches [`solve_rope_attachments`]. Attachments are processed
/// in order, so a later attachment sees the body positions already corrected by an
/// earlier one. Returns the indices of the attachments that broke (same convention).
pub fn solve_rope_attachments_two_way(
    attachments: &[RopeAttachment],
    rope_positions: &mut [Vec3Fix],
    rope_velocities: &mut [Vec3Fix],
    bodies: &mut [RigidBody],
    dt: Fix128,
) -> Vec<Option<usize>> {
    if dt.is_zero() {
        return Vec::new();
    }

    let dt_sq = dt * dt;
    let mut broken = Vec::new();

    for (ai, att) in attachments.iter().enumerate() {
        if att.rope_particle >= rope_positions.len() || att.body_index >= bodies.len() {
            continue;
        }

        let rope_pos = rope_positions[att.rope_particle];
        match evaluate_attachment(att, rope_pos, &bodies[att.body_index], dt_sq) {
            AttachmentOutcome::Skip => {}
            AttachmentOutcome::Break => broken.push(Some(ai)),
            AttachmentOutcome::Move {
                direction,
                lambda,
                w_body,
            } => {
                let rope_correction = direction * lambda;
                rope_positions[att.rope_particle] = rope_pos - rope_correction;
                let vel_change = rope_correction / dt;
                rope_velocities[att.rope_particle] =
                    rope_velocities[att.rope_particle] - vel_change;

                let body = &mut bodies[att.body_index];
                body.position = body.position + direction * (lambda * w_body);
            }
        }
    }

    broken
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::solver::BodyType;

    const fn make_static_body(pos: Vec3Fix) -> RigidBody {
        RigidBody {
            position: pos,
            velocity: Vec3Fix::ZERO,
            inv_mass: Fix128::ZERO,
            inv_inertia: Vec3Fix::ZERO,
            prev_position: pos,
            rotation: QuatFix::IDENTITY,
            angular_velocity: Vec3Fix::ZERO,
            prev_rotation: QuatFix::IDENTITY,
            restitution: Fix128::ZERO,
            friction: Fix128::ZERO,
            gravity_scale: Fix128::ONE,
            linear_damping: Fix128::ONE,
            angular_damping: Fix128::ONE,
            is_sensor: false,
            body_type: BodyType::Static,
            kinematic_target: None,
        }
    }

    #[test]
    fn test_new_attachment() {
        let att = RopeAttachment::new(0, 1, Vec3Fix::UNIT_X);
        assert_eq!(att.rope_particle, 0);
        assert_eq!(att.body_index, 1);
        assert!(att.compliance.is_zero());
        assert!(att.break_force.is_none());
    }

    #[test]
    fn test_with_break_force() {
        let att = RopeAttachment::with_break_force(2, 3, Vec3Fix::ZERO, Fix128::from_int(100));
        assert_eq!(att.rope_particle, 2);
        assert!(att.break_force.is_some());
    }

    #[test]
    fn test_compliance_builder() {
        let att = RopeAttachment::new(0, 0, Vec3Fix::ZERO).compliance(Fix128::from_ratio(1, 10));
        assert!(!att.compliance.is_zero());
    }

    #[test]
    fn test_solve_pulls_rope_to_anchor() {
        let body = make_static_body(Vec3Fix::from_int(10, 0, 0));
        let bodies = vec![body];
        let att = RopeAttachment::new(0, 0, Vec3Fix::ZERO);

        let mut rope_pos = vec![Vec3Fix::from_int(0, 0, 0)];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let dt = Fix128::from_ratio(1, 60);
        let _ = solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, dt);

        // Rope should have moved toward the body
        assert!(rope_pos[0].x > Fix128::ZERO);
    }

    #[test]
    fn test_solve_already_at_anchor() {
        let body = make_static_body(Vec3Fix::from_int(5, 5, 5));
        let bodies = vec![body];
        let att = RopeAttachment::new(0, 0, Vec3Fix::ZERO);

        let mut rope_pos = vec![Vec3Fix::from_int(5, 5, 5)];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let dt = Fix128::from_ratio(1, 60);
        let broken = solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, dt);

        // No breaking, no movement needed
        assert!(broken.is_empty());
        assert_eq!(rope_pos[0].x.hi, 5);
    }

    #[test]
    fn test_break_force_exceeded() {
        let body = make_static_body(Vec3Fix::from_int(1000, 0, 0));
        let bodies = vec![body];
        // Very low break force, large distance -> should break
        let att =
            RopeAttachment::with_break_force(0, 0, Vec3Fix::ZERO, Fix128::from_ratio(1, 10000));

        let mut rope_pos = vec![Vec3Fix::ZERO];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let dt = Fix128::from_ratio(1, 60);
        let broken = solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, dt);

        assert!(!broken.is_empty());
    }

    #[test]
    fn test_break_force_not_exceeded() {
        let body = make_static_body(Vec3Fix::from_int(0, 0, 0));
        let bodies = vec![body];
        // Large break force, zero distance -> should not break
        let att = RopeAttachment::with_break_force(0, 0, Vec3Fix::ZERO, Fix128::from_int(1000000));

        let mut rope_pos = vec![Vec3Fix::ZERO];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let dt = Fix128::from_ratio(1, 60);
        let broken = solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, dt);

        assert!(broken.is_empty());
    }

    #[test]
    fn test_zero_dt_no_effect() {
        let body = make_static_body(Vec3Fix::from_int(10, 0, 0));
        let bodies = vec![body];
        let att = RopeAttachment::new(0, 0, Vec3Fix::ZERO);

        let mut rope_pos = vec![Vec3Fix::ZERO];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let broken =
            solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, Fix128::ZERO);

        assert!(broken.is_empty());
        assert!(rope_pos[0].x.is_zero());
    }

    #[test]
    fn test_invalid_indices_skipped() {
        let body = make_static_body(Vec3Fix::ZERO);
        let bodies = vec![body];
        let att = RopeAttachment::new(99, 0, Vec3Fix::ZERO); // invalid particle index

        let mut rope_pos = vec![Vec3Fix::ZERO];
        let mut rope_vel = vec![Vec3Fix::ZERO];

        let dt = Fix128::from_ratio(1, 60);
        // Should not panic
        let _ = solve_rope_attachments(&[att], &mut rope_pos, &mut rope_vel, &bodies, dt);
    }
}
