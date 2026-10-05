//! Positional compliance of ball and hinge joints, one solve at a time
//!
//! Reaches `BallJoint::with_compliance` and `HingeJoint::with_compliance`.
//!
//! The point constraint of both joints is solved in compliance form: with
//! the anchors at the body centres (no lever arm, so no rotation) and a gap
//! `d` between them, one `solve_joints` call moves the bodies by
//! `λ = d / (w_a + w_b + α / h²)` along the gap, each by its inverse mass
//! `w`. The gap left is therefore
//!
//! `d' = d · (α / h²) / (w_a + w_b + α / h²)`.
//!
//! With unit masses (`w_a = w_b = 1`), `h = 1/2` and `α = 1/2` the
//! compliance term is `2`, so the gap halves: 2 → 1. A rigid joint
//! (`α = 0`) closes it in one call. The values are dyadic, so the positions
//! are exact.
//!
//! The compliance is not accumulated across calls (there is no running
//! multiplier): each call removes the same fraction `(w_a + w_b) /
//! (w_a + w_b + α / h²)` of whatever gap is left, so `n` calls leave
//! `d · (1/2)ⁿ` here, and the stiffness a scene sees depends on how many
//! times per step the joints are solved.
//!
//! Run with: `cargo run --example joint_compliance_closed_form`

use alice_physics::joint::{solve_joints, BallJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn pair() -> Vec<RigidBody> {
    vec![
        RigidBody::new_dynamic(Vec3Fix::from_int(-1, 0, 0), Fix128::ONE),
        RigidBody::new_dynamic(Vec3Fix::from_int(1, 0, 0), Fix128::ONE),
    ]
}

fn gap(bodies: &[RigidBody]) -> Fix128 {
    bodies[1].position.x - bodies[0].position.x
}

fn main() {
    let h = Fix128::from_ratio(1, 2);
    let alpha = Fix128::from_ratio(1, 2);
    let z_axis = Vec3Fix::from_int(0, 0, 1);

    let ball = |a: Fix128| {
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO).with_compliance(a))
    };
    let hinge = |a: Fix128| {
        Joint::Hinge(
            HingeJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO, z_axis, z_axis).with_compliance(a),
        )
    };

    for (name, make) in [
        ("ball", &ball as &dyn Fn(Fix128) -> Joint),
        ("hinge", &hinge),
    ] {
        // Compliant: the gap 2 halves, symmetrically about the origin.
        let mut bodies = pair();
        solve_joints(&[make(alpha)], &mut bodies, h);
        assert_eq!(
            gap(&bodies),
            Fix128::ONE,
            "{name}: d' = 2 · 2 / (1 + 1 + 2)"
        );
        assert_eq!(
            bodies[0].position,
            Vec3Fix::new(-h, Fix128::ZERO, Fix128::ZERO)
        );
        assert_eq!(bodies[1].position.x, h);
        // Three more calls: 1 → 1/8, the same fraction each time.
        for _ in 0..3 {
            solve_joints(&[make(alpha)], &mut bodies, h);
        }
        assert_eq!(
            gap(&bodies),
            Fix128::from_ratio(1, 8),
            "{name}: 2 · (1/2)^4"
        );

        // Rigid: closed in one call, both bodies meeting at the origin.
        let mut bodies = pair();
        solve_joints(&[make(Fix128::ZERO)], &mut bodies, h);
        assert_eq!(gap(&bodies), Fix128::ZERO, "{name}: rigid");
        assert_eq!(bodies[0].position, Vec3Fix::ZERO);

        // A static body takes nothing: w_a = 0, so λ = 2 / (0 + 1 + 2) = 2/3
        // and only the dynamic body moves, leaving d' = 2 · 2 / 3 = 4/3.
        let mut bodies = pair();
        bodies[0] = RigidBody::new_static(Vec3Fix::from_int(-1, 0, 0));
        solve_joints(&[make(alpha)], &mut bodies, h);
        let left = gap(&bodies).to_f64();
        assert!(
            (left - 4.0 / 3.0).abs() < 1e-12,
            "{name}: d' = 4/3, got {left}"
        );
        assert_eq!(bodies[0].position, Vec3Fix::from_int(-1, 0, 0));
        println!("[joint_compliance] {name}: gap 2 -> 1 (alpha/h^2 = 2), rigid -> 0, static anchor -> 4/3");
    }
}
