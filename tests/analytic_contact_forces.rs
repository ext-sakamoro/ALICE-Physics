//! Oracles for the contact forces a `PhysicsWorld` reports and draws.
//!
//! # What is measured
//!
//! A body at rest under gravity carries its weight: the normal force at its contact
//! is `m·g`, and in a stack each contact carries the weight of everything above it
//! (`F = Σ mᵢ·g`). The solver's own quantity is the separation applied in the last
//! substep (`cached_lambda`); `contact_forces` divides it by `(wₐ + w_b)·h²` to get a
//! force. Friction arrows have magnitude `μ·F` along two tangents that are unit,
//! orthogonal to each other and to the normal; a friction cone has half-angle
//! `atan μ` (`π/4` for `μ = 1`) and height `F`.
//!
//! The scenes settle for 30–40 frames and no longer: a body at rest goes to sleep
//! after about 60 frames, and a sleeping body has no contacts to report.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

const G: f64 = 9.81;

/// Spheres of radius 0.5 stacked on a static one (radius 0.5) at the origin, each
/// resting just touching the one below, with the given masses bottom to top.
fn stack(masses: &[f64]) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: v3(0.0, -G, 0.0),
        substeps: 8,
        iterations: 4,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
    for (i, &m) in masses.iter().enumerate() {
        w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, (i + 1) as f64, 0.0), fx(m)),
            fx(0.5),
        );
    }
    w
}

fn settle(w: &mut PhysicsWorld, steps: usize) {
    for _ in 0..steps {
        w.step(dt());
    }
}

/// The force on each contact, ordered by the lower body's index.
fn forces_by_height(w: &PhysicsWorld) -> Vec<f64> {
    let mut f: Vec<(f64, f64)> = w
        .contact_forces(dt())
        .iter()
        .map(|(p, _, force)| (p.y.to_f64(), force.to_f64()))
        .collect();
    f.sort_by(|a, b| a.0.total_cmp(&b.0));
    f.into_iter().map(|(_, force)| force).collect()
}

/// One sphere resting on a static one carries its weight `m·g`.
#[test]
fn a_resting_body_carries_its_weight() {
    for &m in &[1.0, 2.5] {
        let mut w = stack(&[m]);
        settle(&mut w, 30);
        let forces = forces_by_height(&w);
        assert_eq!(forces.len(), 1, "one contact");
        let want = m * G;
        assert!(
            (forces[0] - want).abs() < 0.02 * want,
            "m = {m}: force {} but the weight is {want}",
            forces[0]
        );
    }
}

/// In a stack each contact carries the weight above it: the top one `m₂·g`, the
/// lower one `(m₁ + m₂)·g`.
#[test]
fn in_a_stack_each_contact_carries_the_weight_above() {
    let (m1, m2) = (2.0, 3.0);
    let mut w = stack(&[m1, m2]);
    settle(&mut w, 40);
    let forces = forces_by_height(&w);
    assert_eq!(forces.len(), 2, "two contacts");
    let (lower, upper) = (forces[0], forces[1]);
    assert!(
        (lower - (m1 + m2) * G).abs() < 0.05 * (m1 + m2) * G,
        "lower contact {lower}, the weight above is {}",
        (m1 + m2) * G
    );
    assert!(
        (upper - m2 * G).abs() < 0.05 * m2 * G,
        "upper contact {upper}, the weight above is {}",
        m2 * G
    );
}

/// The arrows point along the contact normal, unit length, with the force as
/// magnitude. The normal runs from B to A and the force acts on A: here A is the
/// static body added first (below) and B the ball, so the arrow points down, the way
/// the ball presses on the ground.
#[test]
fn the_arrows_carry_the_normal_force_along_the_normal() {
    let mut w = stack(&[2.0]);
    settle(&mut w, 30);
    let arrows = w.contact_arrows(dt());
    let forces = w.contact_forces(dt());
    assert_eq!(arrows.len(), 1);
    assert!(!arrows[0].is_friction);
    assert_eq!(arrows[0].force_magnitude, forces[0].2);
    let n = arrows[0].normal;
    assert!((n.length().to_f64() - 1.0).abs() < 1e-9, "unit");
    assert!(
        (n.y.to_f64() + 1.0).abs() < 1e-6,
        "down: the ball presses on the static body"
    );
}

/// Friction arrows: two per contact, magnitude `μ·F` with the contact's own `μ`,
/// along unit tangents orthogonal to each other and to the normal.
#[test]
fn friction_arrows_are_tangent_and_scaled_by_the_contacts_own_mu() {
    let mut w = stack(&[2.0]);
    // A friction coefficient different from the default 0.3, through the material
    // of the dynamic body.
    w.get_body_mut(1).expect("the ball").friction = fx(0.8);
    settle(&mut w, 30);
    let force = w.contact_forces(dt())[0].2.to_f64();
    let mu = w.contact_constraints[0].friction.to_f64();
    let arrows = w.contact_friction_arrows(dt());
    assert_eq!(arrows.len(), 2, "two tangents per contact");
    let n = w.contact_forces(dt())[0].1;
    for a in &arrows {
        assert!(a.is_friction);
        assert!(
            (a.force_magnitude.to_f64() - mu * force).abs() < 1e-9 * force.max(1.0),
            "magnitude {} but μ·F = {}",
            a.force_magnitude.to_f64(),
            mu * force
        );
        assert!(
            (a.normal.length().to_f64() - 1.0).abs() < 1e-9,
            "unit tangent"
        );
        assert!(
            a.normal.dot(n).to_f64().abs() < 1e-9,
            "tangent to the contact"
        );
    }
    assert!(
        arrows[0].normal.dot(arrows[1].normal).to_f64().abs() < 1e-9,
        "the two tangents are orthogonal"
    );
    assert!(mu > 0.0, "the contact has a friction coefficient: {mu}");
}

/// A friction cone has half-angle `atan μ` and the normal force as height.
#[test]
fn a_friction_cone_has_half_angle_atan_mu_and_the_force_as_height() {
    let mut w = stack(&[2.0]);
    settle(&mut w, 30);
    let mu = w.contact_constraints[0].friction.to_f64();
    let force = w.contact_forces(dt())[0].2;
    let cones = w.contact_friction_cones(dt());
    assert_eq!(cones.len(), 1);
    assert!(
        (cones[0].half_angle.to_f64() - mu.atan()).abs() < 1e-9,
        "atan μ"
    );
    assert_eq!(cones[0].height, force);
    assert!(
        (cones[0].normal.y.to_f64() + 1.0).abs() < 1e-6,
        "axis along the normal, B to A"
    );
}

/// Nothing is drawn without contacts, and a contact between two immovable bodies
/// carries no force and is left out.
#[test]
fn no_contacts_no_forces_and_immovable_pairs_are_left_out() {
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.0, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.step(dt());
    assert!(w.contact_forces(dt()).is_empty());
    assert!(w.contact_arrows(dt()).is_empty());
    assert!(w.contact_friction_arrows(dt()).is_empty());
    assert!(w.contact_friction_cones(dt()).is_empty());

    // Two static bodies overlapping: a contact the solver never resolves, no force.
    let mut w = PhysicsWorld::new(SolverConfig::default());
    w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
    w.add_body_with_radius(RigidBody::new_static(v3(0.4, 0.0, 0.0)), fx(0.5));
    w.step(dt());
    assert!(
        w.contact_forces(dt()).is_empty(),
        "no force between immovable bodies"
    );
}

/// A contact is drawn where it is on body A: on A's surface, 0.5 from its centre,
/// not where B's surface reaches (the two differ by the penetration, here ~1e-3).
#[test]
fn a_contact_is_at_the_surface_of_body_a() {
    let mut w = stack(&[2.0]);
    settle(&mut w, 30);
    let (point, _, _) = w.contact_forces(dt())[0];
    // A is the static body at the origin, radius 0.5.
    let distance = point.length().to_f64();
    assert!(
        (distance - 0.5).abs() < 1e-6,
        "the point is {distance} from A's centre, A's radius is 0.5"
    );
}

/// A contact between two immovable bodies carries no force: put by hand, the solver
/// never resolves it.
#[test]
fn a_hand_made_contact_of_two_immovable_bodies_has_no_force() {
    use alice_physics::collider::Contact;
    use alice_physics::solver::ContactConstraint;
    let mut w = PhysicsWorld::new(SolverConfig::default());
    let a = w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
    let b = w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.9, 0.0)), fx(0.5));
    let mut c = ContactConstraint::new(
        a,
        b,
        Contact {
            depth: fx(0.1),
            normal: v3(0.0, -1.0, 0.0),
            point_a: v3(0.0, 0.5, 0.0),
            point_b: v3(0.0, 0.4, 0.0),
        },
    );
    c.cached_lambda = fx(0.1);
    w.add_contact(c);
    assert_eq!(w.contact_constraints.len(), 1, "the contact is there");
    assert!(w.contact_forces(dt()).is_empty(), "but it carries no force");
    assert!(w.contact_arrows(dt()).is_empty());
}

/// The substep length is the frame step over the number of substeps, and a world
/// with zero substeps counts as one (no division by zero): a hand-made contact with
/// a known multiplier gives `λ / (w·dt²)`.
#[test]
fn the_force_is_the_multiplier_over_w_times_the_substep_squared() {
    use alice_physics::collider::Contact;
    use alice_physics::solver::ContactConstraint;
    for (substeps, h) in [(0usize, 1.0 / 60.0), (1, 1.0 / 60.0), (4, 1.0 / 240.0)] {
        let mut w = PhysicsWorld::new(SolverConfig {
            substeps,
            ..SolverConfig::default()
        });
        // A dynamic body of mass 2 (w = 0.5) on a static one.
        let a = w.add_body_with_radius(RigidBody::new_dynamic(v3(0.0, 1.0, 0.0), fx(2.0)), fx(0.5));
        let b = w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.0, 0.0)), fx(0.5));
        let mut c = ContactConstraint::new(
            a,
            b,
            Contact {
                depth: fx(0.01),
                normal: v3(0.0, 1.0, 0.0),
                point_a: v3(0.0, 0.5, 0.0),
                point_b: v3(0.0, 0.49, 0.0),
            },
        );
        c.cached_lambda = fx(0.002);
        w.add_contact(c);
        let force = w.contact_forces(dt())[0].2.to_f64();
        let want = 0.002 / (0.5 * h * h);
        assert!(
            (force - want).abs() < 1e-6 * want,
            "{substeps} substeps: force {force}, closed form {want}"
        );
    }
}
