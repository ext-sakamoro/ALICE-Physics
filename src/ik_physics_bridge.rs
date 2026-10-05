//! Bridge between an inverse-kinematics (IK) target chain and the
//! physics rigid-body world.
//!
//! Complements [`crate::ragdoll`] (fully passive humanoid) by exposing
//! a lightweight position-space constraint that steers named ragdoll
//! bones toward IK-computed target positions. Callers who own an IK
//! solver (e.g. `alice-kinematics::fabrik`) can post the solved
//! per-frame chain into a `IkTargetSet` and have the physics
//! stabiliser apply proportional position corrections (immovable bodies,
//! `inv_mass == 0`, are held fixed; the correction is not otherwise
//! weighted by mass).
//!
// LIMITATION(COV-MBD-101): The MVP is purely kinematic: the correction is Baumgarte-style position stabilisation, not force-based tracking.
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
    /// Blend factor in `[0, 1]` (values outside are clamped). `0` disables
    /// tracking, `1` snaps to the target. Bodies with `inv_mass == 0` are
    /// never moved.
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
    /// Global compliance factor (higher = softer tracking). Each correction is
    /// scaled by `1 / (1 + compliance)`; a negative value is treated as `0`.
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
        // `compliance` is "higher = softer", so a negative value has no meaning:
        // taken as written it would flip the correction (or divide by zero at -1).
        let compliance = if self.compliance < Fix128::ZERO {
            Fix128::ZERO
        } else {
            self.compliance
        };
        let scale = Fix128::ONE / (Fix128::ONE + compliance);
        for target in &self.targets {
            if target.body >= world.bodies.len() {
                continue;
            }
            let body = world.bodies[target.body];
            if body.inv_mass.is_zero() {
                continue;
            }
            let residual = target.position - body.position;
            // The blend factor is documented as `[0, 1]`; outside it the body
            // would overshoot the target (weight > 1) or move away from it.
            let weight = if target.weight < Fix128::ZERO {
                Fix128::ZERO
            } else if target.weight > Fix128::ONE {
                Fix128::ONE
            } else {
                target.weight
            };
            let correction = residual * (weight * scale);
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
