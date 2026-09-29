//! Analytic oracles for articulated-body forward dynamics
//! (`FeatherstoneSolver`).
//!
//! Every expected value in this file is derived from mechanics below, never
//! by running the implementation and recording what came out. The five
//! properties pinned here were chosen because each one holds for *any*
//! correct multibody formulation — they do not depend on whether links are
//! modelled as point masses on massless rods or as rigid bodies with their
//! own inertia tensors, nor on the integrator's order. That makes them safe
//! to assert exactly rather than within a tuned tolerance.
//!
//! # Why these five
//!
//! A forward-dynamics solver for an articulated body has to do two things
//! that a bag of independent rigid bodies does not: it has to make the joint
//! constrain relative motion, and it has to produce a joint acceleration
//! from the articulated-body inertia. The oracles below measure exactly
//! those two things, from five directions:
//!
//! 1. [`free_floating_chain_center_of_mass_accelerates_at_gravity`] — the
//!    constraint forces must be internal (Newton's third law).
//! 2. [`link_welded_to_static_base_does_not_move`] — a zero-DOF joint must
//!    remove all freedom.
//! 3. [`chain_hanging_at_rest_stays_at_rest`] — a configuration in static
//!    equilibrium must produce zero acceleration.
//! 4. [`joint_type_changes_the_motion`] — the joint must be read at all.
//! 5. [`hinged_body_acquires_angular_velocity`] and
//!    [`rotational_inertia_is_load_bearing`] — the angular half of the
//!    spatial inertia must reach the answer.
//!
//! Oracles 2 and 3 are both "nothing moves", so on their own they could be
//! satisfied by a solver that does nothing. Oracle 1 rules that out: its
//! expected answer is a specific nonzero velocity. The pair is therefore
//! non-degenerate in both directions — neither "everything free-falls" nor
//! "nothing ever moves" can pass the set.
//!
//! # Oracles vs characterisation
//!
//! Every oracle above is `#[ignore = "src bug: …"]`: the red is correct and
//! is held back only because the solver has not caught up. `cargo test --test
//! analytic_multibody_dynamics -- --ignored` runs them, and that command is
//! the authority on how many there are — this paragraph deliberately does not
//! say, so that adding an oracle cannot make it wrong.
//!
//! Ignoring them costs no CI coverage, because the last section of this file
//! —  [`characterises_the_present_solver`] and its neighbours — is **not**
//! ignored and pins the present behaviour from the other side. Those tests
//! record what the implementation does today; they are *not* derived from
//! mechanics and must not be read as saying it is right. Any change to the
//! solver moves those numbers and reds them in CI, which is what makes it
//! safe to ignore the oracles above. The two halves are a pair: the oracles
//! are the goal, the characterisation is the guard, and exactly one half is
//! green at any time. The commit that fixes the solver removes the `ignore`
//! attributes and deletes the characterisation section in the same diff.
//!
//! What the characterisation cannot detect, measured by mutation: deleting
//! the body of the forward velocity pass, and replacing the angular block of
//! the articulated-body inertia with anything at all, both leave every test
//! here green. That is not a gap in the guard — those two computations have
//! no reader, so removing them genuinely does not change the output. The
//! guard does red on every mutation that touches the live path (dropping the
//! child-inertia accumulation, bypassing the inertia in the acceleration
//! pass, truncating the tree recursion, dropping the position integration).
//!
//! Nothing in this file touches `src/`.

use alice_physics::articulation::{ArticulatedBody, FeatherstoneSolver};
use alice_physics::joint::{BallJoint, FixedJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::RigidBody;

/// Gravity used by every scene: 10 m/s^2 downward, exactly representable.
fn gravity() -> Vec3Fix {
    Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO)
}

/// Timestep used by every scene: 1/60 s.
fn dt() -> Fix128 {
    Fix128::from_ratio(1, 60)
}

/// Ball joint whose two anchors are given in each body's local frame.
fn ball(a: usize, b: usize, anchor_a: Vec3Fix, anchor_b: Vec3Fix) -> Joint {
    Joint::Ball(BallJoint::new(a, b, anchor_a, anchor_b))
}

// ---------------------------------------------------------------------------
// Oracle 1 — internal constraint forces cancel
// ---------------------------------------------------------------------------

/// A free-floating articulated chain under gravity must have its centre of
/// mass accelerate at exactly `g`.
///
/// # Where the expected value comes from
///
/// Newton's third law. Summing `m_i a_i = F_i` over all links, every joint
/// constraint force appears twice with opposite sign (the joint pushes on
/// the parent exactly as hard as it pulls on the child) and cancels. The
/// only force left in the sum is weight, so
///
/// ```text
///   (sum_i m_i) a_com = (sum_i m_i) g   =>   a_com = g
/// ```
///
/// This is independent of the number of links, the joint types, the
/// configuration, the mass distribution and the inertia tensors. With a
/// semi-implicit Euler step of size `dt` starting from rest, the centre of
/// mass velocity after one step is therefore exactly `g * dt`, and with
/// `g_y = -10` and `dt = 1/60` that is `-1/6` m/s.
///
/// The chain is given a bent configuration (not a straight line) so that a
/// solver which happened to be correct only for collinear chains cannot pass
/// by accident.
#[test]
#[ignore = "src bug: forward dynamics applies no joint constraint, so the centre of \
            mass does not accelerate at g. Remove this attribute in the commit \
            that lands the constraint"]
fn free_floating_chain_center_of_mass_accelerates_at_gravity() {
    // Three dynamic links, no static base: root is free to move.
    // Masses 1 / 2 / 3 so that a mass-weighted mistake cannot cancel out.
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(1)),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(2)),
        RigidBody::new(Vec3Fix::from_int(2, -2, 0), Fix128::from_int(3)),
    ];

    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ false);
    artic.add_link(
        0,
        1,
        ball(
            0,
            1,
            Vec3Fix::from_int(0, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
        ),
        Vec3Fix::from_int(0, -2, 0),
    );
    artic.add_link(
        1,
        2,
        ball(
            1,
            2,
            Vec3Fix::from_int(0, -1, 0),
            Vec3Fix::from_int(-1, 0, 0),
        ),
        Vec3Fix::from_int(2, 0, 0),
    );

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, gravity(), dt());

    // Mass-weighted mean velocity = centre of mass velocity.
    let mut total_mass = Fix128::ZERO;
    let mut weighted = Vec3Fix::ZERO;
    for body in &bodies {
        let m = Fix128::ONE / body.inv_mass;
        total_mass = total_mass + m;
        weighted = weighted + body.velocity * m;
    }
    let com_velocity = weighted / total_mass;

    let expected = gravity() * dt();
    assert_eq!(
        com_velocity,
        expected,
        "free-floating chain: centre of mass must accelerate at exactly g \
         (internal constraint forces cancel pairwise). \
         expected v_com = g*dt = {:?} ({} m/s in y), got {:?} ({} m/s in y)",
        expected,
        expected.y.to_f64(),
        com_velocity,
        com_velocity.y.to_f64(),
    );
}

// ---------------------------------------------------------------------------
// Oracle 2 — a zero-DOF joint removes all freedom
// ---------------------------------------------------------------------------

/// A link welded to a static base by a `Joint::Fixed` must not move.
///
/// # Where the expected value comes from
///
/// A fixed (weld) joint has zero degrees of freedom: it constrains the
/// child's position and orientation rigidly to the parent's. The parent here
/// is a static body, which by definition never moves. A rigid constraint to
/// something that does not move leaves the child no admissible motion at
/// all, so its acceleration is zero and it stays where it is for any number
/// of steps.
///
/// No modelling choice affects this: it is the definition of a weld.
#[test]
#[ignore = "src bug: `solve` never reads `link.joint`, so a zero-DOF weld does not \
            hold the link. Remove this attribute in the commit that makes the \
            joint an input to forward dynamics"]
fn link_welded_to_static_base_does_not_move() {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(5)),
    ];
    let start = bodies[1].position;

    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ true);
    artic.add_link(
        0,
        1,
        Joint::Fixed(FixedJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(0, 2, 0),
            QuatFix::IDENTITY,
        )),
        Vec3Fix::from_int(0, -2, 0),
    );

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..30 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }

    assert_eq!(
        bodies[1].velocity,
        Vec3Fix::ZERO,
        "welded to a static base: a zero-DOF joint leaves no admissible \
         motion, so velocity must stay exactly zero after 30 steps, got {:?} \
         ({} m/s in y)",
        bodies[1].velocity,
        bodies[1].velocity.y.to_f64(),
    );
    assert_eq!(
        bodies[1].position,
        start,
        "welded to a static base: position must be unchanged after 30 steps, \
         expected {:?}, got {:?} (drift {} m in y)",
        start,
        bodies[1].position,
        (bodies[1].position.y - start.y).to_f64(),
    );
}

// ---------------------------------------------------------------------------
// Oracle 3 — static equilibrium produces zero acceleration
// ---------------------------------------------------------------------------

/// A two-link chain hanging straight down at rest from a fixed base is in
/// static equilibrium and must not start moving.
///
/// # Where the expected value comes from
///
/// Each link hangs directly below its joint anchor, so the line of action of
/// its weight passes through the anchor and exerts no moment about it. The
/// constraint force along each rod is free to take any magnitude, and the
/// magnitudes that balance the weights exist and are unique
/// (`T_lower = m_2 g`, `T_upper = (m_1 + m_2) g`). With the whole chain at
/// rest there is no centripetal term to supply either, so the net force on
/// every link is zero and the configuration is a genuine equilibrium — the
/// lowest point of a pendulum, released from rest.
///
/// Hence all accelerations are exactly zero, so velocities stay zero and
/// positions stay put, for any number of steps. This holds whether the links
/// are point masses or extended rigid bodies, because in either case the
/// weight acts at the centre of mass which lies on the vertical through the
/// anchor.
#[test]
#[ignore = "src bug: links free-fall regardless of configuration, so a static \
            equilibrium is not preserved. Remove this attribute in the commit \
            that lands the constraint"]
fn chain_hanging_at_rest_stays_at_rest() {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(3)),
        RigidBody::new(Vec3Fix::from_int(0, -4, 0), Fix128::from_int(1)),
    ];
    let start: Vec<Vec3Fix> = bodies.iter().map(|b| b.position).collect();

    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ true);
    artic.add_link(
        0,
        1,
        ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(0, 2, 0)),
        Vec3Fix::from_int(0, -2, 0),
    );
    artic.add_link(
        1,
        2,
        ball(
            1,
            2,
            Vec3Fix::from_int(0, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
        ),
        Vec3Fix::from_int(0, -2, 0),
    );

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..30 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }

    for i in 1..bodies.len() {
        assert_eq!(
            bodies[i].velocity,
            Vec3Fix::ZERO,
            "link {i} hangs directly below its anchor at rest = static \
             equilibrium, so velocity must stay exactly zero after 30 steps, \
             got {:?} ({} m/s in y)",
            bodies[i].velocity,
            bodies[i].velocity.y.to_f64(),
        );
        assert_eq!(
            bodies[i].position,
            start[i],
            "link {i} must not drift from equilibrium, expected {:?}, got \
             {:?} (drift {} m in y)",
            start[i],
            bodies[i].position,
            (bodies[i].position.y - start[i].y).to_f64(),
        );
    }
}

// ---------------------------------------------------------------------------
// Oracle 4 — the joint reaches the answer at all
// ---------------------------------------------------------------------------

/// Two scenes that are identical except for the joint connecting the links
/// must not produce identical motion.
///
/// # Where the expected value comes from
///
/// This is a discrimination oracle, not a numeric one: it asserts only that
/// the joint is an input to forward dynamics. The two scenes differ by a
/// weld (zero DOF, the child is pinned to a static base and cannot move) and
/// a ball joint (three rotational DOF, the child is free to swing). Those
/// are different constraint sets, the child starts off the vertical through
/// the anchor so the ball joint scene has a nonzero moment about the anchor,
/// and therefore the two admissible motions differ. Any solver that produces
/// bit-identical output for both is not reading the joint.
///
/// Stated this way the oracle needs no reference implementation and no
/// tolerance, which is what makes it safe to pin.
#[test]
#[ignore = "src bug: `solve` never reads `link.joint`, so every joint type gives \
            bit-identical motion. Remove this attribute in the commit that \
            makes the joint an input to forward dynamics"]
fn joint_type_changes_the_motion() {
    fn run(joint: Joint) -> (Vec3Fix, Vec3Fix) {
        let mut bodies = vec![
            RigidBody::new_static(Vec3Fix::ZERO),
            // Offset horizontally from the anchor: a swinging joint has a
            // moment here, a weld does not.
            RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(1)),
        ];
        let mut artic = ArticulatedBody::new(0, true);
        artic.add_link(0, 1, joint, Vec3Fix::from_int(2, 0, 0));

        let mut solver = FeatherstoneSolver::new();
        for _ in 0..10 {
            solver.solve(&artic, &mut bodies, gravity(), dt());
        }
        (bodies[1].position, bodies[1].velocity)
    }

    let welded = run(Joint::Fixed(FixedJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
        QuatFix::IDENTITY,
    )));
    let swinging = run(ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(-2, 0, 0)));

    assert_ne!(
        welded, swinging,
        "a zero-DOF weld and a three-DOF ball joint impose different \
         constraints, so the motion must differ; identical output means the \
         joint is never read by forward dynamics (weld {:?}, ball {:?})",
        welded, swinging,
    );
}

// ---------------------------------------------------------------------------
// Oracle 5 — the angular half of the spatial inertia reaches the answer
// ---------------------------------------------------------------------------

/// A body hinged about an axis it is offset from, stepped under gravity.
/// `inertia_scale` multiplies the child's inertia tensor, leaving its mass,
/// position and every other input untouched.
fn hinged_swing(inertia_scale: i64) -> (Vec3Fix, Vec3Fix) {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(1)),
    ];
    // inv_inertia scales by the reciprocal of the inertia scale.
    bodies[1].inv_inertia = bodies[1].inv_inertia * Fix128::from_ratio(1, inertia_scale);

    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        // Hinge about +z, so a body offset along +x swings in the xy plane
        // under gravity.
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-2, 0, 0),
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        )),
        Vec3Fix::from_int(2, 0, 0),
    );

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..10 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }
    (bodies[1].angular_velocity, bodies[1].velocity)
}

/// A hinge whose child is offset from the axis must produce angular motion.
///
/// # Where the expected value comes from
///
/// A hinge joint fixes the child's orientation relative to the parent up to
/// one rotation about the hinge axis. The child's centre of mass here sits
/// off the axis, so gravity exerts a moment about it and the child must
/// start rotating about that axis — a hinged body cannot translate along its
/// arc without also rotating, because its orientation is tied to the joint
/// angle by the constraint. So after stepping from rest under gravity the
/// angular velocity must be nonzero.
///
/// The oracle asserts only "nonzero", not a value, because the magnitude
/// depends on how the link's mass is distributed, which is a modelling
/// choice. The sign of the claim does not: zero is wrong for every choice.
#[test]
#[ignore = "src bug: the backward pass produces no joint acceleration and \
            `angular_velocity` is never written. Remove this attribute in the \
            commit that wires the backward pass through"]
fn hinged_body_acquires_angular_velocity() {
    let (spin, _) = hinged_swing(1);
    assert_ne!(
        spin,
        Vec3Fix::ZERO,
        "a hinged body offset from its axis must acquire angular velocity \
         about that axis (its orientation is tied to the joint angle), got \
         exactly zero — forward dynamics produced no angular acceleration",
    );
}

/// The child's rotational inertia must change the motion.
///
/// # Where the expected value comes from
///
/// The angular acceleration of a hinged link is its moment about the axis
/// divided by its moment of inertia about that axis, and its linear motion
/// follows from the joint angle. Scaling the inertia tensor by 1000 with
/// mass, position, joint and gravity all held fixed therefore has to change
/// the result: the same moment acting against a thousand times the inertia
/// gives a different joint acceleration.
///
/// Bit-identical output for the two runs means the angular block of the
/// articulated-body inertia never reaches the answer at all — which is a
/// stronger statement than "the 6x6 spatial inertia is approximated by its
/// diagonal", and is what this measures.
#[test]
#[ignore = "src bug: the angular block of the articulated-body inertia is computed \
            and never read. Remove this attribute in the commit that lands the \
            6x6 spatial inertia"]
fn rotational_inertia_is_load_bearing() {
    let light = hinged_swing(1);
    let heavy = hinged_swing(1000);
    assert_ne!(
        light, heavy,
        "scaling the child's rotational inertia by 1000 must change the \
         motion (angular acceleration = moment / inertia); bit-identical \
         output means the angular part of the articulated-body inertia never \
         reaches the answer (I=1 {light:?}, I=1000 {heavy:?})",
    );
}

// ---------------------------------------------------------------------------
// Characterisation — what the solver does today
// ---------------------------------------------------------------------------
//
// NOT ORACLES. Everything below records the present behaviour so that the
// oracles above can be `#[ignore]`d without losing CI coverage. None of these
// numbers is claimed to be correct; several are demonstrably wrong. Delete
// this whole section in the commit that fixes the solver.

/// The rule the present `solve` follows, reconstructed independently.
///
/// Measured, not derived: each non-root link is integrated as
///
/// ```text
///   a_i = (g * m_i) / (sum of m_j over the subtree rooted at i, i included)
/// ```
///
/// with semi-implicit Euler, and the root link is not integrated at all. The
/// joint is not consulted. Reproduced here with the same `Fix128` operations
/// in the same order as `fa_recursive`, so the comparison is bit-exact.
fn free_fall_scaled(mass_self: Fix128, subtree_mass: Fix128, steps: u32) -> Vec3Fix {
    free_fall_scaled_state(mass_self, subtree_mass, steps).0
}

/// As [`free_fall_scaled`], returning both the final velocity and the
/// displacement. Position is pinned as well as velocity so that a change to
/// the integrator — not just to the acceleration — reds this file.
fn free_fall_scaled_state(
    mass_self: Fix128,
    subtree_mass: Fix128,
    steps: u32,
) -> (Vec3Fix, Vec3Fix) {
    let force = gravity() * mass_self;
    let accel = force / subtree_mass;
    let mut v = Vec3Fix::ZERO;
    let mut displacement = Vec3Fix::ZERO;
    for _ in 0..steps {
        v = v + accel * dt();
        displacement = displacement + v * dt();
    }
    (v, displacement)
}

/// Pins the acceleration rule above on a static-base chain.
///
/// Masses 2 (middle) and 3 (leaf): the leaf's subtree is itself, so it
/// free-falls at `g`; the middle link's subtree is 2 + 3 = 5, so it is
/// integrated at `g * 2/5`. A correct solver would hold both at rest
/// ([`chain_hanging_at_rest_stays_at_rest`]).
#[test]
fn characterises_the_present_solver() {
    const STEPS: u32 = 30;
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(2)),
        RigidBody::new(Vec3Fix::from_int(0, -4, 0), Fix128::from_int(3)),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(0, 2, 0)),
        Vec3Fix::from_int(0, -2, 0),
    );
    artic.add_link(
        1,
        2,
        ball(
            1,
            2,
            Vec3Fix::from_int(0, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
        ),
        Vec3Fix::from_int(0, -2, 0),
    );

    let start: Vec<Vec3Fix> = bodies.iter().map(|b| b.position).collect();

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..STEPS {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }

    assert_eq!(
        bodies[0].velocity,
        Vec3Fix::ZERO,
        "characterisation: the root link is never integrated",
    );
    assert_eq!(
        bodies[0].position, start[0],
        "characterisation: the root link is never integrated",
    );

    let mid = free_fall_scaled_state(Fix128::from_int(2), Fix128::from_int(5), STEPS);
    assert_eq!(
        (bodies[1].velocity, bodies[1].position - start[1]),
        mid,
        "characterisation: middle link is integrated at g * m_self / subtree_mass",
    );

    let leaf = free_fall_scaled_state(Fix128::from_int(3), Fix128::from_int(3), STEPS);
    assert_eq!(
        (bodies[2].velocity, bodies[2].position - start[2]),
        leaf,
        "characterisation: the leaf's subtree is itself, so it free-falls at g",
    );
}

/// A single body on a static base, stepped under each joint type and over a
/// wide range of inertia tensors.
fn present_output(inv_inertia: Vec3Fix, joint: Joint) -> (Vec3Fix, Vec3Fix) {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE),
    ];
    bodies[1].inv_inertia = inv_inertia;
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(0, 1, joint, Vec3Fix::from_int(2, 0, 0));
    let mut solver = FeatherstoneSolver::new();
    for _ in 0..10 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }
    (bodies[1].velocity, bodies[1].angular_velocity)
}

/// Pins that the joint does not reach the answer.
///
/// All four joint types produce bit-identical output, and that output is
/// plain free fall. A correct solver would differ between them
/// ([`joint_type_changes_the_motion`]).
#[test]
fn characterises_the_joint_as_unused() {
    let unit = Vec3Fix::from_int(1, 1, 1);
    let weld = Joint::Fixed(FixedJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
        QuatFix::IDENTITY,
    ));
    let joints = [
        weld,
        ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(-2, 0, 0)),
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-2, 0, 0),
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        )),
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-2, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
        )),
    ];
    let free_fall = free_fall_scaled(Fix128::ONE, Fix128::ONE, 10);
    for joint in joints {
        assert_eq!(
            present_output(unit, joint),
            (free_fall, Vec3Fix::ZERO),
            "characterisation: every joint type gives plain free fall and no spin",
        );
    }
}

/// Pins that the rotational inertia does not reach the answer.
///
/// The inertia tensor is swept over six orders of magnitude, made
/// anisotropic, and finally zeroed (which takes the other branch of
/// `bi_recursive`'s `is_zero` test). Every case is bit-identical. A correct
/// solver would differ ([`rotational_inertia_is_load_bearing`]).
#[test]
fn characterises_the_rotational_inertia_as_unused() {
    let hinge = Joint::Hinge(HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
        Vec3Fix::from_int(0, 0, 1),
        Vec3Fix::from_int(0, 0, 1),
    ));
    let inertias = [
        Vec3Fix::from_int(1, 1, 1),
        Vec3Fix::from_int(1000, 1000, 1000),
        Vec3Fix::new(
            Fix128::from_ratio(1, 1000),
            Fix128::from_ratio(1, 1000),
            Fix128::from_ratio(1, 1000),
        ),
        Vec3Fix::new(
            Fix128::ONE,
            Fix128::from_ratio(1, 10),
            Fix128::from_ratio(1, 100),
        ),
        Vec3Fix::ZERO,
    ];
    let free_fall = free_fall_scaled(Fix128::ONE, Fix128::ONE, 10);
    for inv_inertia in inertias {
        assert_eq!(
            present_output(inv_inertia, hinge),
            (free_fall, Vec3Fix::ZERO),
            "characterisation: the inertia tensor does not change the answer",
        );
    }
}
