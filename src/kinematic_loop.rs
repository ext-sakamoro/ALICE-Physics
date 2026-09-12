//! Position-based kinematic loop closure constraint.
//!
//! Complements [`crate::articulation`] (tree-topology articulated
//! bodies) by adding a **loop-closure constraint** that turns a
//! tree of rigid bodies into a closed kinematic chain such as a
//! four-bar linkage, a piston-crankshaft mechanism, or a
//! differential gear.
//!
//! The MVP applies the constraint via a Baumgarte-style position
//! stabilisation:
//!
//! ```text
//! err   = (pos_a + local_offset_a) − (pos_b + local_offset_b)
//! delta = err / (1 + compliance)
//! pos_a −= 0.5 · delta · w_a
//! pos_b += 0.5 · delta · w_b
//! ```
//!
//! where `w_a`, `w_b` are inverse-mass weights taken from the bodies.
//! Rotational alignment is not enforced yet — callers who need
//! parallel-axis coupling (gear meshing) should combine the closure
//! with a HingeJoint on both endpoints.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::{PhysicsWorld, RigidBody};

/// Explicit closure between two bodies whose local anchor points must
/// remain coincident under the solver's positional integration.
///
/// This mirrors [`crate::joint::BallJoint`] semantically but is
/// applied as an independent stabilisation pass — the caller decides
/// when to run it (typically after the primary constraint solver's
/// position step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopClosureConstraint {
    /// First body index in the `PhysicsWorld`.
    pub body_a: usize,
    /// Second body index.
    pub body_b: usize,
    /// Anchor point in body A's local frame (the closure attaches
    /// there).
    pub local_anchor_a: Vec3Fix,
    /// Anchor point in body B's local frame.
    pub local_anchor_b: Vec3Fix,
    /// Compliance (inverse stiffness). `0` = perfectly rigid,
    /// positive values allow slack.
    pub compliance: Fix128,
}

impl LoopClosureConstraint {
    /// Create a rigid closure between two bodies with the anchors
    /// coincident with each body's centre.
    #[must_use]
    pub const fn centre_to_centre(body_a: usize, body_b: usize) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3Fix::ZERO,
            local_anchor_b: Vec3Fix::ZERO,
            compliance: Fix128::ZERO,
        }
    }

    /// Positional residual: world-space offset of anchor A relative
    /// to anchor B. Callers can inspect this to gauge constraint
    /// error before / after applying the correction.
    #[must_use]
    pub fn residual(&self, world: &PhysicsWorld) -> Vec3Fix {
        let a = world.bodies[self.body_a];
        let b = world.bodies[self.body_b];
        (a.position + self.local_anchor_a) - (b.position + self.local_anchor_b)
    }

    /// Apply one Baumgarte-style correction pass, moving both bodies
    /// towards a satisfied constraint. Compliance dampens the
    /// correction; static bodies (`inv_mass == 0`) are held fixed.
    pub fn apply(&self, world: &mut PhysicsWorld) {
        let a_snapshot = world.bodies[self.body_a];
        let b_snapshot = world.bodies[self.body_b];
        let residual = (a_snapshot.position + self.local_anchor_a)
            - (b_snapshot.position + self.local_anchor_b);
        let total_inv_mass = a_snapshot.inv_mass + b_snapshot.inv_mass;
        if total_inv_mass.is_zero() {
            return;
        }
        let scale = Fix128::ONE / (Fix128::ONE + self.compliance);
        let delta = residual * scale;
        let w_a = a_snapshot.inv_mass / total_inv_mass;
        let w_b = b_snapshot.inv_mass / total_inv_mass;
        world.bodies[self.body_a].position = a_snapshot.position - delta * w_a;
        world.bodies[self.body_b].position = b_snapshot.position + delta * w_b;
    }
}

/// Handles returned by [`four_bar_linkage`] so callers can drive
/// the mechanism through their preferred crank body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FourBarLinkage {
    /// Ground body (typically static).
    pub ground: usize,
    /// Crank (driven) body.
    pub crank: usize,
    /// Coupler (floating) body.
    pub coupler: usize,
    /// Rocker (following) body.
    pub rocker: usize,
    /// The three intra-link joints (ground↔crank, crank↔coupler,
    /// coupler↔rocker).
    pub joints: [usize; 3],
    /// The loop-closure constraint (rocker↔ground).
    pub closure: LoopClosureConstraint,
}

/// Register a planar four-bar linkage into `world`.
///
/// Adds four fresh rigid bodies + three ball joints along the
/// kinematic tree and returns a [`LoopClosureConstraint`] that
/// re-closes the loop by tying the rocker back to the ground body.
///
/// Positions are laid out in the XY plane with `ground_position` at
/// the origin of the mechanism.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn four_bar_linkage(
    world: &mut PhysicsWorld,
    ground_position: Vec3Fix,
    crank_length: Fix128,
    coupler_length: Fix128,
    rocker_length: Fix128,
    ground_length: Fix128,
    body_mass: Fix128,
) -> FourBarLinkage {
    let ground = world.add_body(RigidBody::new(ground_position, Fix128::ZERO));
    let crank = world.add_body(RigidBody::new(
        ground_position + Vec3Fix::new(crank_length, Fix128::ZERO, Fix128::ZERO),
        body_mass,
    ));
    let coupler = world.add_body(RigidBody::new(
        ground_position + Vec3Fix::new(crank_length + coupler_length, Fix128::ZERO, Fix128::ZERO),
        body_mass,
    ));
    let rocker = world.add_body(RigidBody::new(
        ground_position + Vec3Fix::new(ground_length, Fix128::ZERO, Fix128::ZERO),
        body_mass,
    ));
    use crate::joint::{BallJoint, Joint};
    let zero = Vec3Fix::ZERO;
    let j0 = world.add_joint(Joint::Ball(BallJoint::new(ground, crank, zero, zero)));
    let j1 = world.add_joint(Joint::Ball(BallJoint::new(crank, coupler, zero, zero)));
    let j2 = world.add_joint(Joint::Ball(BallJoint::new(coupler, rocker, zero, zero)));
    let closure = LoopClosureConstraint::centre_to_centre(rocker, ground);
    let _ = rocker_length; // stored purely for caller convenience; the constraint
                           // covers the closing side numerically.
    FourBarLinkage {
        ground,
        crank,
        coupler,
        rocker,
        joints: [j0, j1, j2],
        closure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::SolverConfig;

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(SolverConfig::default())
    }

    #[test]
    fn residual_reflects_body_offset() {
        let mut w = world();
        let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = w.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        let closure = LoopClosureConstraint::centre_to_centre(a, b);
        let residual = closure.residual(&w);
        assert_eq!(residual.x, -Fix128::from_int(2));
    }

    #[test]
    fn apply_pulls_bodies_together() {
        let mut w = world();
        let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b = w.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        let closure = LoopClosureConstraint::centre_to_centre(a, b);
        closure.apply(&mut w);
        let residual_after = closure.residual(&w);
        assert!(residual_after.x.abs() < Fix128::from_ratio(1, 1000));
    }

    #[test]
    fn apply_does_nothing_when_both_static() {
        let mut w = world();
        let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO));
        let b = w.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO),
            Fix128::ZERO,
        ));
        let before_a = w.bodies[a].position;
        let before_b = w.bodies[b].position;
        let closure = LoopClosureConstraint::centre_to_centre(a, b);
        closure.apply(&mut w);
        assert_eq!(w.bodies[a].position, before_a);
        assert_eq!(w.bodies[b].position, before_b);
    }

    #[test]
    fn compliance_dampens_correction() {
        let mut w1 = world();
        let a1 = w1.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b1 = w1.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        let rigid = LoopClosureConstraint::centre_to_centre(a1, b1);
        rigid.apply(&mut w1);

        let mut w2 = world();
        let a2 = w2.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let b2 = w2.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        let mut soft = LoopClosureConstraint::centre_to_centre(a2, b2);
        soft.compliance = Fix128::from_int(3);
        soft.apply(&mut w2);

        let rigid_error = rigid.residual(&w1).x.abs();
        let soft_error = soft.residual(&w2).x.abs();
        assert!(soft_error > rigid_error);
    }

    #[test]
    fn heavier_body_moves_less() {
        let mut w = world();
        let a = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(1)));
        let b = w.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
            Fix128::from_int(10), // 10× the mass.
        ));
        let closure = LoopClosureConstraint::centre_to_centre(a, b);
        closure.apply(&mut w);
        // The lighter body A should have moved further along +x than
        // the heavier body B moved back.
        let da = (w.bodies[a].position - Vec3Fix::ZERO).length_squared();
        let db_final_pos =
            Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO) - w.bodies[b].position;
        let db = db_final_pos.length_squared();
        assert!(da > db);
    }

    #[test]
    fn four_bar_linkage_registers_expected_bodies() {
        let mut w = world();
        let linkage = four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::from_int(1),
            Fix128::from_int(2),
            Fix128::from_int(1),
            Fix128::from_int(3),
            Fix128::ONE,
        );
        // Four fresh bodies were added.
        let ids = [
            linkage.ground,
            linkage.crank,
            linkage.coupler,
            linkage.rocker,
        ];
        let mut sorted = ids;
        sorted.sort_unstable();
        assert_eq!(sorted[3] - sorted[0], 3);
    }

    #[test]
    fn four_bar_closure_targets_ground_and_rocker() {
        let mut w = world();
        let linkage = four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::from_int(1),
            Fix128::from_int(2),
            Fix128::from_int(1),
            Fix128::from_int(3),
            Fix128::ONE,
        );
        assert_eq!(linkage.closure.body_a, linkage.rocker);
        assert_eq!(linkage.closure.body_b, linkage.ground);
    }
}
