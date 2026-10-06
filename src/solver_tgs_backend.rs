//! Adapter between [`crate::solver`]'s `RigidBody` / `ContactConstraint`
//! world representation and the generic TGS engine in
//! [`crate::solver_tgs`] / [`crate::solver_tgs_hooks_6dof_oriented`].
//!
//! This is the production entry point for [`crate::solver::SolverBackend::Tgs`]:
//! [`crate::solver::PhysicsWorld::step`] calls into this module's conversion
//! functions, then [`crate::solver_tgs_hooks_6dof_oriented_scoped::solve_oriented_islands_serial`]
//! directly (no new per-body math is introduced here — the existing,
//! previously-unwired `solver_tgs*` family already implements the full
//! 6-DOF oriented Gauss-Seidel solve; this module only translates between
//! the two body/contact layouts).
//!
//! # Visibility
//!
//! `pub(crate)` and `std`-gated, matching the rest of the `solver_tgs*`
//! family (see [`crate::solver_tgs`] module doc for the Option-C rationale).
//!
//! Unlike the rest of the `solver_tgs*` family this module has no
//! `#![allow(dead_code)]`: every item here has a production caller
//! ([`crate::solver::PhysicsWorld::step_tgs`]), so none of it is dead.

use crate::math::{Fix128, Vec3Fix};
use crate::solver::{ContactConstraint, RigidBody};
use crate::solver_tgs_hooks_6dof_oriented::{Body6DofOrientedState, ContactOriented};

/// Converts a [`RigidBody`] into the TGS engine's oriented body
/// representation. `stable_id` is the caller-assigned warm-start identity
/// (the world body index is used by [`crate::solver::PhysicsWorld::step_tgs`],
/// which is stable across frames as long as bodies are not removed/reordered —
/// the same caveat [`crate::solver_tgs::BodyRef`] documents).
#[must_use]
pub(crate) fn body_to_tgs(body: &RigidBody, stable_id: u64) -> Body6DofOrientedState {
    Body6DofOrientedState {
        position: [body.position.x, body.position.y, body.position.z],
        orientation: body.rotation,
        linear_velocity: [body.velocity.x, body.velocity.y, body.velocity.z],
        angular_velocity: [
            body.angular_velocity.x,
            body.angular_velocity.y,
            body.angular_velocity.z,
        ],
        inv_mass: body.inv_mass,
        inv_inertia_local: [body.inv_inertia.x, body.inv_inertia.y, body.inv_inertia.z],
        is_dynamic: body.is_dynamic(),
        stable_id,
        overflow: false,
    }
}

/// Writes a solved [`Body6DofOrientedState`] back onto a [`RigidBody`].
///
/// Only the fields the TGS solver can change are touched (position,
/// orientation, linear/angular velocity); mass, inertia, body type,
/// material properties and `kinematic_target` are left untouched (the TGS
/// path does not read or drive `kinematic_target`, see
/// [`crate::solver::SolverBackend`]'s documented gaps).
pub(crate) fn tgs_to_body(state: &Body6DofOrientedState, body: &mut RigidBody) {
    body.position = Vec3Fix::new(state.position[0], state.position[1], state.position[2]);
    body.rotation = state.orientation;
    body.velocity = Vec3Fix::new(
        state.linear_velocity[0],
        state.linear_velocity[1],
        state.linear_velocity[2],
    );
    body.angular_velocity = Vec3Fix::new(
        state.angular_velocity[0],
        state.angular_velocity[1],
        state.angular_velocity[2],
    );
}

/// Builds an orthonormal tangent basis `(t1, t2)` perpendicular to a unit
/// `normal`, used for the TGS contact's friction axes (XPBD's own contact
/// solver has no tangent basis — it only ever applies impulses along
/// `normal` — so there is nothing to port here; this is new, minimal code).
///
/// Picks the world axis least aligned with `normal` to cross against, which
/// avoids the degenerate (near-zero-length) cross product that choosing a
/// fixed axis (e.g. always `+Y`) would hit when `normal` is close to that
/// axis. Degenerate input (`normal` not unit length, e.g. `ZERO`) is the
/// caller's responsibility to avoid; see `tests/analytic_tgs_wiring.rs` for
/// the behavior at exactly `ZERO` (does not panic, returns a defined but
/// unspecified-magnitude basis — never fed a zero normal in practice since
/// [`crate::collider`] only emits unit normals for real contacts).
#[must_use]
pub(crate) fn tangent_basis(normal: Vec3Fix) -> (Vec3Fix, Vec3Fix) {
    // The axis whose |component along `normal`| is smallest is the safest
    // to cross against (crossing against an axis nearly parallel to
    // `normal` produces a near-zero-length vector before normalization).
    let ax = normal.x.to_f32().abs();
    let ay = normal.y.to_f32().abs();
    let az = normal.z.to_f32().abs();
    let helper = if ax <= ay && ax <= az {
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    } else if ay <= az {
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO)
    } else {
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE)
    };
    let t1 = normal.cross(helper).normalize();
    let t2 = normal.cross(t1).normalize();
    (t1, t2)
}

/// Converts a [`ContactConstraint`] (XPBD's `Contact::normal` points from B
/// into A, see `src/collider.rs` doc) into the TGS engine's
/// [`ContactOriented`] (normal points from A into B, see its own field doc)
/// — the two families disagree on this convention, so the sign flip below
/// is load-bearing: getting it backwards makes the contact solver push
/// bodies together instead of apart (`tests/analytic_tgs_wiring.rs::resting_contact_*`
/// pins the correct sign by checking the bodies separate, not interpenetrate,
/// over time).
#[must_use]
pub(crate) fn contact_to_tgs(
    c: &ContactConstraint,
    bodies: &[RigidBody],
    stable_id: u64,
) -> ContactOriented {
    let normal_a_to_b = -c.contact.normal;
    let (t1, t2) = tangent_basis(normal_a_to_b);
    let r_a = c.contact.point_a - bodies[c.body_a].position;
    let r_b = c.contact.point_b - bodies[c.body_b].position;
    ContactOriented {
        body_a: c.body_a,
        body_b: c.body_b,
        stable_id,
        normal: [normal_a_to_b.x, normal_a_to_b.y, normal_a_to_b.z],
        tangent1: [t1.x, t1.y, t1.z],
        tangent2: [t2.x, t2.y, t2.z],
        r_a: [r_a.x, r_a.y, r_a.z],
        r_b: [r_b.x, r_b.y, r_b.z],
        penetration: c.contact.depth,
        friction: c.friction,
        restitution: c.restitution,
        accum_normal: Fix128::ZERO,
        accum_tangent1: Fix128::ZERO,
        accum_tangent2: Fix128::ZERO,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collider::Contact;
    use crate::solver::BodyType;

    fn body(pos: Vec3Fix, dynamic: bool) -> RigidBody {
        let mut b = RigidBody::new(pos, Fix128::ONE);
        if !dynamic {
            b.body_type = BodyType::Static;
            b.inv_mass = Fix128::ZERO;
        }
        b
    }

    // --- tangent_basis -----------------------------------------------------

    /// oracle: for any of the 3 world axes as `normal`, the returned basis
    /// must be unit length and mutually orthogonal (dot products == 0,
    /// computed independently of `tangent_basis` itself from the textbook
    /// orthonormal-basis definition, not by calling the function under test
    /// to produce its own expected value).
    #[test]
    fn tangent_basis_is_orthonormal_for_each_axis() {
        for n in [
            Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE),
        ] {
            let (t1, t2) = tangent_basis(n);
            assert_eq!(n.dot(t1), Fix128::ZERO, "t1 must be perpendicular to n");
            assert_eq!(n.dot(t2), Fix128::ZERO, "t2 must be perpendicular to n");
            assert_eq!(t1.dot(t2), Fix128::ZERO, "t1 must be perpendicular to t2");
            // Unit length within Fix128 CORDIC tolerance (not exact: both
            // `cross` and `normalize` round).
            let tol = Fix128::from_ratio(1, 1000);
            assert!(
                (t1.length() - Fix128::ONE).abs() < tol,
                "t1 len={:?}",
                t1.length()
            );
            assert!(
                (t2.length() - Fix128::ONE).abs() < tol,
                "t2 len={:?}",
                t2.length()
            );
        }
    }

    /// Degenerate input: the zero vector is not a unit normal, but the
    /// helper-axis selection only ever reads `normal`'s components to pick
    /// an axis and crosses/normalizes — `Vec3Fix::normalize` on a
    /// zero-length cross product is the only panic surface, exercised here
    /// via `catch_unwind` per the analytic-oracle-tests rule (no assertion
    /// on the *value* — only that degenerate input is not silently treated
    /// as a hidden success path).
    #[test]
    fn tangent_basis_zero_normal_does_not_panic() {
        let result = std::panic::catch_unwind(|| tangent_basis(Vec3Fix::ZERO));
        assert!(
            result.is_ok(),
            "tangent_basis(ZERO) must not panic (cross(ZERO, axis) = ZERO, \
             normalize(ZERO) is defined to return ZERO, not divide by zero)"
        );
    }

    // --- body_to_tgs / tgs_to_body round-trip ------------------------------

    /// oracle: round-tripping a dynamic body through `body_to_tgs` then
    /// `tgs_to_body` must reproduce every field `tgs_to_body` is documented
    /// to touch, exactly (Fix128 fields are exact integers under this
    /// identity transform, no rounding is introduced by either direction).
    #[test]
    fn body_round_trip_preserves_motion_fields() {
        let mut b = body(
            Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(2),
                Fix128::from_int(3),
            ),
            true,
        );
        b.velocity = Vec3Fix::new(
            Fix128::from_int(4),
            Fix128::from_int(5),
            Fix128::from_int(6),
        );
        b.angular_velocity = Vec3Fix::new(
            Fix128::from_int(7),
            Fix128::from_int(8),
            Fix128::from_int(9),
        );
        let original = b;

        let state = body_to_tgs(&b, 42);
        assert!(state.is_dynamic);
        assert_eq!(state.stable_id, 42);
        assert_eq!(state.inv_mass, b.inv_mass);
        assert_eq!(
            state.inv_inertia_local,
            [b.inv_inertia.x, b.inv_inertia.y, b.inv_inertia.z]
        );

        // Mutate before write-back to prove `tgs_to_body` overwrites rather
        // than reading stale state.
        b.position = Vec3Fix::ZERO;
        b.velocity = Vec3Fix::ZERO;
        tgs_to_body(&state, &mut b);
        assert_eq!(b.position, original.position);
        assert_eq!(b.velocity, original.velocity);
        assert_eq!(b.angular_velocity, original.angular_velocity);
        assert_eq!(b.rotation, original.rotation);
        // Fields `tgs_to_body` must NOT touch.
        assert_eq!(b.inv_mass, original.inv_mass);
        assert_eq!(b.body_type, original.body_type);
    }

    /// Degenerate input: a static body (`inv_mass == ZERO`) round-trips
    /// without panicking and `is_dynamic` is correctly reported `false` so
    /// the island builder treats it as a non-propagating sink.
    #[test]
    fn static_body_round_trip_is_dynamic_false() {
        let b = body(Vec3Fix::ZERO, false);
        let state = body_to_tgs(&b, 7);
        assert!(!state.is_dynamic);
        assert_eq!(state.inv_mass, Fix128::ZERO);
    }

    // --- contact_to_tgs -----------------------------------------------------

    /// oracle: XPBD's `Contact::normal` points from B into A (see
    /// `src/collider.rs` doc and `solve_contact_constraints`'s `+correction`
    /// on A / `-correction` on B). `ContactOriented::normal` is documented
    /// to point from A into B. `contact_to_tgs` must therefore negate it —
    /// checked here against the hand-picked value `(0, 1, 0)` rather than by
    /// calling XPBD's own contact-detection code (independent of the
    /// production path under test).
    #[test]
    fn contact_normal_is_negated_from_xpbd_convention() {
        let bodies = [body(Vec3Fix::ZERO, true), body(Vec3Fix::ZERO, false)];
        let c = ContactConstraint::new(
            0,
            1,
            Contact {
                depth: Fix128::from_ratio(1, 10),
                normal: Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
        );
        let oriented = contact_to_tgs(&c, &bodies, 1);
        assert_eq!(
            oriented.normal,
            [Fix128::ZERO, -Fix128::ONE, Fix128::ZERO],
            "ContactOriented.normal must be the negation of Contact.normal"
        );
    }

    /// oracle: `r_a` / `r_b` are the lever arms from each body's centre to
    /// the contact point — `point - body.position`, computed directly here
    /// (not by calling `contact_to_tgs` for the expected value) for a
    /// hand-picked asymmetric scene.
    #[test]
    fn contact_lever_arms_are_point_minus_body_position() {
        let bodies = [
            body(
                Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::ZERO),
                true,
            ),
            body(
                Vec3Fix::new(Fix128::ZERO, Fix128::from_int(5), Fix128::ZERO),
                false,
            ),
        ];
        let c = ContactConstraint::new(
            0,
            1,
            Contact {
                depth: Fix128::from_ratio(1, 10),
                normal: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
                point_a: Vec3Fix::new(Fix128::from_int(9), Fix128::ZERO, Fix128::ZERO),
                point_b: Vec3Fix::new(Fix128::from_int(1), Fix128::from_int(5), Fix128::ZERO),
            },
        );
        let oriented = contact_to_tgs(&c, &bodies, 2);
        assert_eq!(
            oriented.r_a,
            [-Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            "r_a = point_a - bodies[body_a].position"
        );
        assert_eq!(
            oriented.r_b,
            [Fix128::ONE, Fix128::ZERO, Fix128::ZERO],
            "r_b = point_b - bodies[body_b].position"
        );
    }

    /// oracle: friction / restitution / penetration pass through unchanged,
    /// and the fresh-contact accumulators start at zero (warm-starting is
    /// the hook's job via `ImpulseCache`, not this conversion).
    #[test]
    fn contact_scalar_fields_pass_through_and_accumulators_start_zero() {
        let bodies = [body(Vec3Fix::ZERO, true), body(Vec3Fix::ZERO, false)];
        let mut c = ContactConstraint::new(
            0,
            1,
            Contact {
                depth: Fix128::from_ratio(3, 10),
                normal: Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
                point_a: Vec3Fix::ZERO,
                point_b: Vec3Fix::ZERO,
            },
        );
        c.friction = Fix128::from_ratio(7, 10);
        c.restitution = Fix128::from_ratio(2, 10);
        let oriented = contact_to_tgs(&c, &bodies, 3);
        assert_eq!(oriented.friction, c.friction);
        assert_eq!(oriented.restitution, c.restitution);
        assert_eq!(oriented.penetration, c.contact.depth);
        assert_eq!(oriented.accum_normal, Fix128::ZERO);
        assert_eq!(oriented.accum_tangent1, Fix128::ZERO);
        assert_eq!(oriented.accum_tangent2, Fix128::ZERO);
        assert_eq!(oriented.stable_id, 3);
    }
}
