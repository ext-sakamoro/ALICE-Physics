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

use crate::error::PhysicsError;
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
    /// Ground body (static): the ground–crank pin `O₂`.
    pub ground: usize,
    /// Crank body: the crank–coupler pin `A`.
    pub crank: usize,
    /// Coupler body: the coupler–rocker pin `B`.
    pub coupler: usize,
    /// Rocker body: the rocker–ground pin `O₄`, static (a ground pin), so
    /// the loop is closed by construction; `closure` re-closes it if the
    /// caller frees the body.
    pub rocker: usize,
    /// The three link constraints (ground↔crank, crank↔coupler,
    /// coupler↔rocker): indices into `PhysicsWorld::distance_constraints`,
    /// each holding one link length.
    pub joints: [usize; 3],
    /// The loop-closure constraint (rocker pin ↔ ground link end).
    pub closure: LoopClosureConstraint,
}

/// Register a planar four-bar linkage into `world`.
///
/// Norton, *Design of Machinery* §4.5: the loop-closure equation
/// `r₂ + r₃ = r₁ + r₄` is written in link *lengths*, so the mechanism is
/// modelled as four pin bodies (`O₂` = ground, `A` = crank pin, `B` =
/// coupler pin, `O₄` = rocker pin) joined by three rigid
/// [`DistanceConstraint`](crate::solver::DistanceConstraint)s carrying the
/// crank, coupler and rocker lengths. `O₄` is a static ground pin at
/// `ground_position + (ground_length, 0, 0)`, so the loop is closed by
/// construction and the returned [`LoopClosureConstraint`] (rocker pin ↔
/// that ground anchor) has zero residual; it is what a caller applies
/// after freeing the rocker pin. `body_mass` is the mass of the crank and
/// coupler pins. Ball joints would demand coincident pins, which is why the
/// links are distance constraints (before 1.2.0 the helper registered
/// zero-anchor ball joints and the solver collapsed every link to length 0).
///
/// Layout is in the XY plane with the crank along `+x` (crank angle 0);
/// the coupler pin is the circle–circle intersection of the coupler and
/// rocker lengths with positive `y` (the "open" configuration).
///
/// # Panics
///
/// If the four lengths cannot close at crank angle 0 (the coupler and
/// rocker circles do not intersect); use [`try_four_bar_linkage`] to get
/// the error instead.
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
    match try_four_bar_linkage(
        world,
        ground_position,
        crank_length,
        coupler_length,
        rocker_length,
        ground_length,
        body_mass,
    ) {
        Ok(linkage) => linkage,
        Err(e) => panic!("four_bar_linkage: {e}"),
    }
}

/// [`four_bar_linkage`] returning `Err(PhysicsError::InvalidConfiguration)`
/// instead of panicking when the lengths cannot form a closed loop at
/// crank angle 0. Nothing is added to `world` on error.
///
/// # Errors
///
/// `InvalidConfiguration` when a length is not positive or when the
/// coupler / rocker circles about the crank pin and the rocker ground pin
/// do not intersect (`|r₃ − r₄| ≤ |A O₄| ≤ r₃ + r₄` violated).
#[allow(clippy::too_many_arguments)]
pub fn try_four_bar_linkage(
    world: &mut PhysicsWorld,
    ground_position: Vec3Fix,
    crank_length: Fix128,
    coupler_length: Fix128,
    rocker_length: Fix128,
    ground_length: Fix128,
    body_mass: Fix128,
) -> Result<FourBarLinkage, PhysicsError> {
    if crank_length <= Fix128::ZERO
        || coupler_length <= Fix128::ZERO
        || rocker_length <= Fix128::ZERO
        || ground_length <= Fix128::ZERO
    {
        return Err(PhysicsError::InvalidConfiguration {
            reason: "four-bar link lengths must be positive",
        });
    }
    // Pins at crank angle 0: O2 = origin, A = (r2, 0), O4 = (r1, 0); B is on
    // the circle of radius r3 about A and radius r4 about O4.
    let d = (ground_length - crank_length).abs();
    let sum = coupler_length + rocker_length;
    let diff = (coupler_length - rocker_length).abs();
    if d > sum || d < diff || d.is_zero() {
        return Err(PhysicsError::InvalidConfiguration {
            reason: "four-bar links cannot close at crank angle 0 (coupler / rocker circles do not intersect)",
        });
    }
    // Along-axis distance from A to the chord, then the chord half-height.
    let two = Fix128::from_int(2);
    let a = (d * d + coupler_length * coupler_length - rocker_length * rocker_length) / (two * d);
    let h_sq = coupler_length * coupler_length - a * a;
    let h = if h_sq.is_negative() {
        Fix128::ZERO
    } else {
        h_sq.sqrt()
    };
    // A → O4 direction along x: +1 when O4 is right of A, −1 otherwise.
    let dir = if ground_length >= crank_length {
        Fix128::ONE
    } else {
        -Fix128::ONE
    };
    let pin_a = ground_position + Vec3Fix::new(crank_length, Fix128::ZERO, Fix128::ZERO);
    let pin_o4 = ground_position + Vec3Fix::new(ground_length, Fix128::ZERO, Fix128::ZERO);
    let pin_b = pin_a + Vec3Fix::new(dir * a, h, Fix128::ZERO);

    let ground = world.add_body(RigidBody::new(ground_position, Fix128::ZERO));
    let crank = world.add_body(RigidBody::new(pin_a, body_mass));
    let coupler = world.add_body(RigidBody::new(pin_b, body_mass));
    // O4 is a ground pin: static, so the loop is closed by construction and
    // the returned closure has zero residual (it is still the constraint a
    // caller re-applies after freeing the rocker pin, e.g. to animate the
    // ground link).
    let rocker = world.add_body(RigidBody::new(pin_o4, Fix128::ZERO));
    let _ = body_mass;

    use crate::solver::DistanceConstraint;
    let zero = Vec3Fix::ZERO;
    let base = world.distance_constraints.len();
    world.add_distance_constraint(DistanceConstraint::new(
        ground,
        crank,
        zero,
        zero,
        crank_length,
    ));
    world.add_distance_constraint(DistanceConstraint::new(
        crank,
        coupler,
        zero,
        zero,
        coupler_length,
    ));
    world.add_distance_constraint(DistanceConstraint::new(
        coupler,
        rocker,
        zero,
        zero,
        rocker_length,
    ));
    // O4 sits on the ground link: rocker pin ↔ ground body anchor (r1, 0, 0)
    let closure = LoopClosureConstraint {
        body_a: rocker,
        body_b: ground,
        local_anchor_a: zero,
        local_anchor_b: Vec3Fix::new(ground_length, Fix128::ZERO, Fix128::ZERO),
        compliance: Fix128::ZERO,
    };
    Ok(FourBarLinkage {
        ground,
        crank,
        coupler,
        rocker,
        joints: [base, base + 1, base + 2],
        closure,
    })
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
        // O4 is held on the ground link, r1 from O2
        assert_eq!(
            linkage.closure.local_anchor_b,
            Vec3Fix::new(Fix128::from_int(3), Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(linkage.closure.residual(&w), Vec3Fix::ZERO);
    }

    /// Initial pins satisfy every link length exactly (circle–circle
    /// intersection), and the lengths survive motion: kick the crank pin
    /// and let the solver run.
    #[test]
    fn four_bar_link_lengths_hold_at_rest_and_in_motion() {
        let mut w = PhysicsWorld::new(SolverConfig {
            gravity: Vec3Fix::ZERO,
            ..SolverConfig::default()
        });
        let (r2, r3, r4, r1) = (1.0, 2.0, 1.5, 2.5);
        let l = four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::from_f64(r2),
            Fix128::from_f64(r3),
            Fix128::from_f64(r4),
            Fix128::from_f64(r1),
            Fix128::ONE,
        );
        let dist = |w: &PhysicsWorld, i: usize, j: usize| {
            (w.bodies[i].position - w.bodies[j].position)
                .length()
                .to_f64()
        };
        assert!((dist(&w, l.crank, l.ground) - r2).abs() < 1e-9);
        assert!((dist(&w, l.coupler, l.crank) - r3).abs() < 1e-9);
        assert!((dist(&w, l.rocker, l.coupler) - r4).abs() < 1e-9);
        assert!((dist(&w, l.rocker, l.ground) - r1).abs() < 1e-9);
        assert!(
            w.bodies[l.coupler].position.y > Fix128::ZERO,
            "open configuration"
        );
        // drive the crank pin upward and integrate
        w.bodies[l.crank].velocity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO);
        for _ in 0..60 {
            w.step(Fix128::from_ratio(1, 60));
            l.closure.apply(&mut w);
        }
        assert!(
            w.bodies[l.crank].position.y.to_f64() > 0.1,
            "crank rotated: {:?}",
            w.bodies[l.crank].position
        );
        assert!((dist(&w, l.crank, l.ground) - r2).abs() < 1e-3);
        assert!((dist(&w, l.coupler, l.crank) - r3).abs() < 1e-3);
        assert!((dist(&w, l.rocker, l.coupler) - r4).abs() < 1e-3);
        assert!((dist(&w, l.rocker, l.ground) - r1).abs() < 1e-3);
    }

    #[test]
    fn four_bar_rejects_lengths_that_cannot_close() {
        let mut w = world();
        // |A O4| = 4 > r3 + r4 = 2
        let r = try_four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::from_int(5),
            Fix128::ONE,
        );
        assert!(matches!(r, Err(PhysicsError::InvalidConfiguration { .. })));
        assert_eq!(w.bodies.len(), 0, "nothing added on error");
        // zero length
        let r = try_four_bar_linkage(
            &mut w,
            Vec3Fix::ZERO,
            Fix128::ZERO,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
            Fix128::ONE,
        );
        assert!(matches!(r, Err(PhysicsError::InvalidConfiguration { .. })));
    }
}
