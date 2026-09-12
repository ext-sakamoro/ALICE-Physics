//! Bridge between an inverse-kinematics (IK) target chain and the
//! physics rigid-body world.
//!
//! Complements [`crate::ragdoll`] (fully passive humanoid) by exposing
//! a lightweight position-space constraint that steers named ragdoll
//! bones toward IK-computed target positions. Callers who own an IK
//! solver (e.g. `alice-kinematics::fabrik`) can post the solved
//! per-frame chain into a `IkTargetSet` and have the physics
//! stabiliser apply proportional position corrections that respect
//! the underlying rigid-body inverse masses.
//!
//! The MVP is purely kinematic: the correction is Baumgarte-style
//! position stabilisation, not force-based tracking. Downstream
//! systems that need PD-controlled joint motors can layer
//! [`crate::motor`] on top.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::PhysicsWorld;

/// Named IK target: pair of body index and world-space goal position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IkTarget {
    /// Body index in the source `PhysicsWorld`.
    pub body: usize,
    /// Desired world-space position (m).
    pub position: Vec3Fix,
    /// Blend factor in `[0, 1]`. `0` disables tracking, `1` snaps to
    /// the target (subject to inverse-mass weighting).
    pub weight: Fix128,
}

impl IkTarget {
    /// Full-weight target (`weight = 1`).
    #[must_use]
    pub const fn snap(body: usize, position: Vec3Fix) -> Self {
        Self {
            body,
            position,
            weight: Fix128::ONE,
        }
    }

    /// Half-weight target — typical setting for a blended IK layer.
    #[must_use]
    pub fn blended(body: usize, position: Vec3Fix) -> Self {
        Self {
            body,
            position,
            weight: Fix128::from_ratio(1, 2),
        }
    }
}

/// Collection of IK targets applied together each physics step.
#[derive(Debug, Clone, Default)]
pub struct IkTargetSet {
    /// All targets driven this frame.
    pub targets: Vec<IkTarget>,
    /// Global compliance factor (higher = softer tracking).
    pub compliance: Fix128,
}

impl IkTargetSet {
    /// Construct an empty target set with zero compliance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a target.
    pub fn push(&mut self, target: IkTarget) -> &mut Self {
        self.targets.push(target);
        self
    }

    /// Apply one position-stabilisation pass on every target. Static
    /// bodies (`inv_mass == 0`) are held fixed.
    pub fn apply(&self, world: &mut PhysicsWorld) {
        let scale = Fix128::ONE / (Fix128::ONE + self.compliance);
        for target in &self.targets {
            if target.body >= world.bodies.len() {
                continue;
            }
            let body = world.bodies[target.body];
            if body.inv_mass.is_zero() {
                continue;
            }
            let residual = target.position - body.position;
            let correction = residual * (target.weight * scale);
            world.bodies[target.body].position = body.position + correction;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{PhysicsWorld, RigidBody, SolverConfig};

    fn world() -> PhysicsWorld {
        PhysicsWorld::new(SolverConfig::default())
    }

    #[test]
    fn empty_set_is_no_op() {
        let mut w = world();
        let b = w.add_body(RigidBody::new(
            Vec3Fix::new(Fix128::from_int(1), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        let before = w.bodies[b].position;
        IkTargetSet::new().apply(&mut w);
        assert_eq!(w.bodies[b].position, before);
    }

    #[test]
    fn full_weight_snaps_dynamic_body_toward_target() {
        let mut w = world();
        let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let target_pos = Vec3Fix::new(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO);
        let mut set = IkTargetSet::new();
        set.push(IkTarget::snap(b, target_pos));
        set.apply(&mut w);
        assert_eq!(w.bodies[b].position, target_pos);
    }

    #[test]
    fn blended_weight_moves_half_way() {
        let mut w = world();
        let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let target_pos = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        let mut set = IkTargetSet::new();
        set.push(IkTarget::blended(b, target_pos));
        set.apply(&mut w);
        // Position should be approximately target * 0.5.
        let expected = target_pos.x * Fix128::from_ratio(1, 2);
        let diff = w.bodies[b].position.x - expected;
        assert!(diff.abs() < Fix128::from_ratio(1, 100));
    }

    #[test]
    fn static_body_is_not_moved() {
        let mut w = world();
        let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ZERO));
        let mut set = IkTargetSet::new();
        set.push(IkTarget::snap(
            b,
            Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
        ));
        set.apply(&mut w);
        assert_eq!(w.bodies[b].position, Vec3Fix::ZERO);
    }

    #[test]
    fn out_of_range_target_is_ignored() {
        let mut w = world();
        let mut set = IkTargetSet::new();
        set.push(IkTarget::snap(
            999,
            Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
        ));
        // No panic.
        set.apply(&mut w);
    }

    #[test]
    fn compliance_dampens_correction() {
        let mut w = world();
        let b = w.add_body(RigidBody::new(Vec3Fix::ZERO, Fix128::ONE));
        let target_pos = Vec3Fix::new(Fix128::from_int(4), Fix128::ZERO, Fix128::ZERO);
        let mut soft = IkTargetSet::new();
        soft.compliance = Fix128::from_int(3);
        soft.push(IkTarget::snap(b, target_pos));
        soft.apply(&mut w);
        assert!(w.bodies[b].position.x < target_pos.x);
    }
}
