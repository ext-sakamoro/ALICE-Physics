//! Analytic oracles for articulated-body forward dynamics
//! (`FeatherstoneSolver`).
//!
//! Every expected value in this file is derived from mechanics below, never
//! by running the implementation and recording what came out. The properties
//! pinned here were chosen because each one holds for *any* correct multibody
//! formulation — they do not depend on whether links are modelled as point
//! masses on massless rods or as rigid bodies with their own inertia tensors,
//! nor on the integrator's order. That makes them safe to assert exactly rather
//! than within a tuned tolerance.
//!
//! # Why these
//!
//! A forward-dynamics solver for an articulated body has to do two things
//! that a bag of independent rigid bodies does not: it has to make the joint
//! constrain relative motion, and it has to produce a joint acceleration
//! from the articulated-body inertia. The oracles below measure exactly
//! those two things, from these directions:
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
//! 6. [`free_floating_chain_links_each_accelerate_at_gravity`] — oracle 1 link
//!    by link, so that two compensating errors cannot hide inside the mean.
//! 7. [`joint_axis_selects_which_moments_can_act`] — the joint's *axis*, not
//!    merely its type, decides which moments can drive it.
//! 8. [`heavier_rotational_inertia_swings_more_slowly`] — the inertia reaches
//!    the answer in the right direction, and the infinite-inertia case is the
//!    slowest of all.
//! 9. [`momentum_of_an_articulating_free_chain_changes_only_by_weight`] —
//!    oracle 1 again, but with the chain in motion, which is the only case
//!    where the bias force and the velocity-product term are nonzero.
//! 10. [`outboard_joint_changes_what_the_inboard_link_feels`] — the rank-`n`
//!     reduction `-U D⁻¹ Uᵀ` reaches the answer, which needs two moving links
//!     and is therefore invisible to every other oracle here.
//! 11. [`compound_pendulum_matches_its_closed_form_angular_acceleration`] — the
//!     one closed-form number in the file, with the hinge deliberately away
//!     from the world origin.
//!
//! Oracles 2 and 3 are both "nothing moves", so on their own they could be
//! satisfied by a solver that does nothing. Oracle 1 rules that out: its
//! expected answer is a specific nonzero velocity. The pair is therefore
//! non-degenerate in both directions — neither "everything free-falls" nor
//! "nothing ever moves" can pass the set. Oracle 7 carries the same pairing
//! inside a single test.
//!
//! # History
//!
//! Oracles 1 to 5 were landed red, each under `#[ignore = "src bug: …"]`, while
//! `FeatherstoneSolver::solve` still ignored `Link::joint` and integrated every
//! link as a scaled free fall. A characterisation section pinned that behaviour
//! from the other side so the ignores cost no CI coverage. The commit that
//! replaced `solve` with the Articulated Body Algorithm removed the ignore
//! attributes, as the ignore reasons said it would. The characterisation tests
//! did not simply go away with it: each one was turned around into the oracle
//! that measures the same quantity from the correct side — the free-fall rule
//! became oracle 6, "every joint type gives the same answer" became oracle 7,
//! and "the inertia tensor changes nothing" became oracle 8. Nothing in this
//! file records present behaviour any more.
//!
//! # Why they are exact
//!
//! The equalities here are exact `Fix128` comparisons rather than tolerances,
//! and that survives the matrix inversions inside the solver for a structural
//! reason, not a lucky one. Gravity enters the solver through Featherstone's
//! base-acceleration substitution, so a chain in free fall and a chain in static
//! equilibrium both have `p^A = 0` at every link; the joint acceleration is then
//! `D⁻¹ · 0`, which is exactly zero however `D⁻¹` rounded. Oracles 1, 2, 3, 6
//! and the two motionless cases of 7 ride on that. The rest assert inequalities
//! or carry a derived bound, and rounding can manufacture neither.
//!
//! # What this set does not see, measured
//!
//! Nine mutations were put into `FeatherstoneSolver` one at a time and the
//! whole set was run against each. Seven are caught:
//!
//! | mutation | oracles red |
//! |---|---|
//! | a weld treated as a ball joint | 4, 7, 10 |
//! | the rank-`n` update `-U D⁻¹ Uᵀ` dropped | 9 |
//! | the bias term `U D⁻¹ u` dropped | 9 |
//! | the gravity substitution at a held base dropped | 2, 3, 4, 5, 7, 8, 10, 11 |
//! | angular velocity never integrated | 5, 7, 8, 11 |
//! | `D⁻¹` replaced by the identity in pass 3 | 5, 8, 9, 11 |
//! | the joint anchor forced to the world origin | 11 |
//!
//! Two are **not** caught, and both are gaps in these scenes rather than in the
//! solver:
//!
//! * **Dropping the velocity-product acceleration `c = v × (v - v_parent)`.**
//!   `v × v` is identically zero, so `c` vanishes for every link whose parent is
//!   at rest — which is every link in every scene here that has a static base.
//!   The one scene with a moving parent is oracle 9, and dropping `c` there
//!   solves a different but still self-consistent system, so momentum is still
//!   conserved and oracle 9 cannot see it either. Closing this needs a closed
//!   form for a *moving* multi-link chain: a rigid assembly spinning about a
//!   fixed hinge is the obvious candidate, where the centrifugal coupling
//!   between the links is analytic.
//! * **Not rotating the inertia tensor into world axes (`R I Rᵀ` → `I`).**
//!   Every body here has an isotropic inertia — `RigidBody::new` builds
//!   `diag(2m/5)` and oracle 8 scales it uniformly — and `R (k·1) Rᵀ = k·1`
//!   exactly, so the rotation has nothing to act on. Closing this needs an
//!   anisotropic inertia on a body that turns appreciably.
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
// Oracle 6 — free fall is free fall link by link, not only on average
// ---------------------------------------------------------------------------

/// Free fall under semi-implicit Euler: `v ← v + g dt`, then `x ← x + v dt`.
///
/// Derived, not measured: with no constraint force acting, every link obeys
/// `a = g`, and the two lines below are the integrator this solver uses written
/// out in the same order, so the comparison is bit-exact rather than a
/// tolerance. Returns the velocity and the displacement after `steps` steps.
fn free_fall_state(steps: u32) -> (Vec3Fix, Vec3Fix) {
    let mut v = Vec3Fix::ZERO;
    let mut displacement = Vec3Fix::ZERO;
    for _ in 0..steps {
        v = v + gravity() * dt();
        displacement = displacement + v * dt();
    }
    (v, displacement)
}

/// Every link of a free-floating chain accelerates at exactly `g`, and none of
/// them starts spinning.
///
/// # Where the expected value comes from
///
/// [`free_floating_chain_center_of_mass_accelerates_at_gravity`] pins the
/// mass-weighted mean, which a solver could satisfy by moving one link too fast
/// and another too slow. The stronger statement is available here because a
/// rigid chain in free fall admits a constraint-force solution that is
/// identically zero: if every link translates at `g` with no rotation, every
/// joint's two anchors keep coinciding and every relative orientation is
/// preserved, so no joint has to push at all. The solution of a constrained
/// system is unique, so that is *the* answer — each link at `g`, no spin, for
/// as many steps as you like.
///
/// This is also where the run is long enough for the velocity-product term to
/// matter: after the first step the links are moving, so `p^A = v ×* I v` is no
/// longer trivially zero by "nothing moves". It is still exactly zero, because
/// in pure translation it reduces to `v_O × m v_O`.
#[test]
fn free_floating_chain_links_each_accelerate_at_gravity() {
    const STEPS: u32 = 30;
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(1)),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(2)),
        RigidBody::new(Vec3Fix::from_int(2, -2, 0), Fix128::from_int(3)),
    ];
    let start: Vec<Vec3Fix> = bodies.iter().map(|b| b.position).collect();

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
    for _ in 0..STEPS {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }

    let (expected_velocity, expected_displacement) = free_fall_state(STEPS);
    for (i, body) in bodies.iter().enumerate() {
        assert_eq!(
            body.velocity,
            expected_velocity,
            "link {i} of a free-floating chain is in free fall, so its velocity \
             after {STEPS} steps must be exactly {} m/s in y, got {}",
            expected_velocity.y.to_f64(),
            body.velocity.y.to_f64(),
        );
        assert_eq!(
            body.position - start[i],
            expected_displacement,
            "link {i} must have fallen exactly {} m in y, got {}",
            expected_displacement.y.to_f64(),
            (body.position.y - start[i].y).to_f64(),
        );
        assert_eq!(
            body.angular_velocity,
            Vec3Fix::ZERO,
            "link {i} has no moment acting on it in free fall, so it must not \
             acquire angular velocity, got {:?}",
            body.angular_velocity,
        );
    }
}

// ---------------------------------------------------------------------------
// Oracle 7 — the joint axis decides which moments can act
// ---------------------------------------------------------------------------

/// One dynamic link on a static base, offset along `+x` from the anchor at the
/// origin, stepped ten times under gravity with the given joint.
fn offset_link_under(joint: Joint) -> (Vec3Fix, Vec3Fix) {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(0, 1, joint, Vec3Fix::from_int(2, 0, 0));
    let mut solver = FeatherstoneSolver::new();
    for _ in 0..10 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }
    (bodies[1].velocity, bodies[1].angular_velocity)
}

fn weld_at_origin() -> Joint {
    Joint::Fixed(FixedJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
        QuatFix::IDENTITY,
    ))
}

fn hinge_about(axis: Vec3Fix) -> Joint {
    Joint::Hinge(HingeJoint::new(
        0,
        1,
        Vec3Fix::ZERO,
        Vec3Fix::from_int(-2, 0, 0),
        axis,
        axis,
    ))
}

/// A joint only admits the motion its own axis allows, so which joint is fitted
/// decides whether gravity can move the link at all.
///
/// # Where the expected value comes from
///
/// The link's centre of mass is at `(2,0,0)` and the anchor is at the origin, so
/// the gravitational moment about the anchor is
///
/// ```text
///   r × F = (2,0,0) × (0,-mg,0) = (0,0,-2mg)
/// ```
///
/// which points purely along `-z`. From that single vector all four cases
/// follow without any reference implementation:
///
/// * **weld** — zero degrees of freedom, so no moment can move it, whatever its
///   direction. Exactly at rest.
/// * **hinge about x** — the joint admits rotation about `x` only, and the
///   moment has no `x` component, so nothing drives it. Exactly at rest, and
///   for a completely different reason than the weld.
/// * **hinge about z** — the moment lies along the hinge axis, so it drives the
///   joint. Moves.
/// * **ball** — three rotational degrees of freedom, one of which is the `z`
///   rotation above. Moves.
///
/// The two that move are not asserted to differ from each other: a ball joint
/// leaves the `x` and `y` rotations unexcited here, so agreeing with the `z`
/// hinge is the correct answer, not a missing distinction.
#[test]
fn joint_axis_selects_which_moments_can_act() {
    let unit_x = Vec3Fix::from_int(1, 0, 0);
    let unit_z = Vec3Fix::from_int(0, 0, 1);

    for (name, joint) in [
        ("weld", weld_at_origin()),
        ("hinge about x", hinge_about(unit_x)),
    ] {
        let (velocity, spin) = offset_link_under(joint);
        assert_eq!(
            (velocity, spin),
            (Vec3Fix::ZERO, Vec3Fix::ZERO),
            "{name}: no admissible motion is driven by a moment along -z, so \
             the link must stay exactly at rest, got velocity {velocity:?} and \
             spin {spin:?}",
        );
    }

    for (name, joint) in [
        ("hinge about z", hinge_about(unit_z)),
        (
            "ball",
            ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(-2, 0, 0)),
        ),
    ] {
        let (velocity, spin) = offset_link_under(joint);
        assert!(
            spin.z < Fix128::ZERO,
            "{name}: the moment about the anchor points along -z and the joint \
             admits that rotation, so the link must turn that way, got {}",
            spin.z.to_f64(),
        );
        assert!(
            velocity.y < Fix128::ZERO,
            "{name}: the link must fall, got {} m/s in y",
            velocity.y.to_f64(),
        );
    }
}

// ---------------------------------------------------------------------------
// Oracle 8 — more rotational inertia means less angular acceleration
// ---------------------------------------------------------------------------

/// [`rotational_inertia_is_load_bearing`] asserts only that the inertia reaches
/// the answer. This asserts the direction in which it does.
///
/// # Where the expected value comes from
///
/// The angular acceleration of the hinged link is the moment about the axis
/// divided by the moment of inertia about that axis. The moment is fixed here:
/// mass, position, gravity and joint are identical across the sweep, and only
/// the inertia tensor changes. So `|α|` must fall strictly as the inertia rises,
/// over any range, and the swing must get slower rather than merely different.
///
/// The last entry sweeps to `inv_inertia == 0`, which takes the other branch of
/// the articulated-body inertia's reciprocal: a zero inverse inertia means the
/// link cannot rotate, so it must come out slower than every finite case. That
/// pins the contract of the stand-in value used for an infinite inertia, which
/// is otherwise only visible from inside the solver.
#[test]
fn heavier_rotational_inertia_swings_more_slowly() {
    // Ascending inertia: 1/1000, 1, 1000, then infinite.
    let inertia_scales = [1, 1000, 1_000_000];
    let mut previous: Option<Fix128> = None;
    for scale in inertia_scales {
        let (spin, _) = hinged_swing(scale);
        let magnitude = spin.z.abs();
        assert!(
            magnitude > Fix128::ZERO,
            "inertia scale {scale}: a finite inertia must still let the link \
             turn, got exactly zero",
        );
        if let Some(previous) = previous {
            assert!(
                magnitude < previous,
                "inertia scale {scale}: angular acceleration is moment over \
                 inertia and the moment is unchanged, so raising the inertia \
                 must lower |spin|; got {} against the previous {}",
                magnitude.to_f64(),
                previous.to_f64(),
            );
        }
        previous = Some(magnitude);
    }

    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE),
    ];
    bodies[1].inv_inertia = Vec3Fix::ZERO;
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        hinge_about(Vec3Fix::from_int(0, 0, 1)),
        Vec3Fix::from_int(2, 0, 0),
    );
    let mut solver = FeatherstoneSolver::new();
    for _ in 0..10 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }
    let locked = bodies[1].angular_velocity.z.abs();
    assert!(
        locked < previous.expect("the sweep ran at least once"),
        "a zero inverse inertia means the link cannot rotate, so it must turn \
         more slowly than every finite inertia in the sweep; got {} against {}",
        locked.to_f64(),
        previous.expect("the sweep ran at least once").to_f64(),
    );
}

// ---------------------------------------------------------------------------
// Oracle 9 — Newton's third law survives once the chain really articulates
// ---------------------------------------------------------------------------

/// A three-link free-floating chain with a genuinely non-rigid initial motion,
/// so that the joints have to do work instead of riding along.
fn articulating_free_chain() -> (Vec<RigidBody>, ArticulatedBody) {
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::from_int(0, 0, 0), Fix128::from_int(1)),
        RigidBody::new(Vec3Fix::from_int(0, -2, 0), Fix128::from_int(2)),
        RigidBody::new(Vec3Fix::from_int(2, -2, 0), Fix128::from_int(3)),
    ];
    // Not a rigid-body motion of the whole chain: the joints are loaded.
    bodies[1].velocity = Vec3Fix::from_int(1, 0, 0);
    bodies[2].angular_velocity = Vec3Fix::from_int(0, 0, 2);

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
    (bodies, artic)
}

/// The total linear momentum of a free-floating chain changes at exactly the
/// total weight, whatever the links are doing to each other.
///
/// # Where the expected value comes from
///
/// The same pairwise cancellation as
/// [`free_floating_chain_center_of_mass_accelerates_at_gravity`], but with the
/// chain in motion rather than at rest, which is the case that actually
/// exercises the articulated-body inertia and the bias force: here `p^A` is
/// nonzero, the velocity-product term is nonzero, and the rank-`n` update
/// decides how much of each link's inertia its parent feels. Summing
/// `m_i a_i = F_i` still leaves only weight, so after `n` semi-implicit Euler
/// steps the momentum change is `n · M g dt`.
///
/// # Why this one has a tolerance and the others do not
///
/// The other equalities are exact because they ride on `p^A = 0`, which makes
/// the joint acceleration `D⁻¹ · 0` regardless of how `D⁻¹` rounded. Nothing
/// protects this one: the constraint forces here are genuinely nonzero and
/// cancel only after passing through matrix inversions, so the sum carries a
/// few units in the last place of `Fix128`.
///
/// The bound is derived rather than tuned. `Fix128` keeps 64 fractional bits,
/// the solver performs on the order of a hundred operations per link per step
/// on quantities of order ten, so the residue per step is a few `2⁻⁶⁴` and over
/// thirty steps stays far below `2⁻⁴⁰`. The measured residue at thirty steps is
/// 74 units in the last place, about 2⁻⁵⁷. A structural error — a missing rank
/// update, a dropped bias term, a dropped Coriolis term — moves the sum by a
/// fraction of the momentum itself, which is more than twenty binary orders of
/// magnitude above this bound. The bound therefore separates rounding from
/// mechanics without being fitted to either.
#[test]
fn momentum_of_an_articulating_free_chain_changes_only_by_weight() {
    const STEPS: u32 = 30;
    // 2^-40: far above the rounding residue, far below any mechanical error.
    let tolerance = Fix128::from_ratio(1, 1i64 << 40);

    let (mut bodies, artic) = articulating_free_chain();
    let initial: Vec<Vec3Fix> = bodies.iter().map(|b| b.velocity).collect();

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..STEPS {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }

    let mut total_mass = Fix128::ZERO;
    let mut momentum_change = Vec3Fix::ZERO;
    for (i, body) in bodies.iter().enumerate() {
        let m = Fix128::ONE / body.inv_mass;
        total_mass = total_mass + m;
        momentum_change = momentum_change + (body.velocity - initial[i]) * m;
    }

    let mut expected = Vec3Fix::ZERO;
    for _ in 0..STEPS {
        expected = expected + gravity() * dt() * total_mass;
    }

    let error = momentum_change - expected;
    for (axis, value) in [("x", error.x), ("y", error.y), ("z", error.z)] {
        assert!(
            value.abs() < tolerance,
            "the joints of a free-floating chain can only exchange momentum \
             between its links, never add any, so the total change must be the \
             weight impulse; {axis} is off by {} against a bound of {} \
             (change {:?}, expected {:?})",
            value.to_f64(),
            tolerance.to_f64(),
            momentum_change,
            expected,
        );
    }
}

// ---------------------------------------------------------------------------
// Oracle 10 — the articulated-body inertia is what makes the algorithm O(n)
// ---------------------------------------------------------------------------

/// A two-link arm on a static base, swinging in the xy plane, whose outboard
/// joint is `outboard`. Returns the inboard link's motion, not the outboard's.
fn inboard_link_of_arm(outboard: Joint) -> (Vec3Fix, Vec3Fix) {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::ONE),
        RigidBody::new(Vec3Fix::from_int(4, 0, 0), Fix128::from_int(3)),
    ];
    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        ball(0, 1, Vec3Fix::ZERO, Vec3Fix::from_int(-2, 0, 0)),
        Vec3Fix::from_int(2, 0, 0),
    );
    artic.add_link(1, 2, outboard, Vec3Fix::from_int(2, 0, 0));

    let mut solver = FeatherstoneSolver::new();
    for _ in 0..10 {
        solver.solve(&artic, &mut bodies, gravity(), dt());
    }
    (bodies[1].velocity, bodies[1].angular_velocity)
}

/// What a link feels from the limb hanging off it depends on the joint in
/// between — that is the whole content of the word "articulated" in
/// articulated-body inertia.
///
/// # Where the expected value comes from
///
/// Two arms identical in every mass, length and position, differing only in the
/// joint at the elbow. Welded, the forearm is rigidly carried: the shoulder has
/// to accelerate the whole limb as one body, and the inertia it feels is the
/// combined inertia about the shoulder. Hinged or balled, the forearm can
/// accelerate differently from the upper arm, so the shoulder feels strictly
/// less — that reduction is exactly the `-U D⁻¹ Uᵀ` term, which is zero for a
/// zero-degree-of-freedom joint and positive definite otherwise.
///
/// So the *inboard* link must move differently in the two cases, even though
/// nothing about the inboard link itself was changed. A solver that passed each
/// child's plain rigid inertia up to its parent — no rank update at all — would
/// give the two arms the same shoulder motion and fail here. That is the case
/// no other oracle in this file covers, because every other scene that moves
/// has only one moving link.
#[test]
fn outboard_joint_changes_what_the_inboard_link_feels() {
    let welded = inboard_link_of_arm(Joint::Fixed(FixedJoint::new(
        1,
        2,
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(-2, 0, 0),
        QuatFix::IDENTITY,
    )));
    let hinged = inboard_link_of_arm(Joint::Hinge(HingeJoint::new(
        1,
        2,
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(-2, 0, 0),
        Vec3Fix::from_int(0, 0, 1),
        Vec3Fix::from_int(0, 0, 1),
    )));
    let balled = inboard_link_of_arm(ball(
        1,
        2,
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(-2, 0, 0),
    ));

    assert_ne!(
        welded, hinged,
        "a welded forearm and a hinged one present different inertias to the \
         shoulder, so the shoulder must move differently; identical output \
         means the child's inertia is passed up unreduced and the `-U D⁻¹ Uᵀ` \
         term never reaches the answer (welded {welded:?}, hinged {hinged:?})",
    );
    assert_ne!(
        welded, balled,
        "same for a ball elbow (welded {welded:?}, balled {balled:?})",
    );
}

// ---------------------------------------------------------------------------
// Oracle 11 — the closed-form angular acceleration of a compound pendulum
// ---------------------------------------------------------------------------

/// The first step of a compound pendulum matches its closed-form angular
/// acceleration, with the hinge deliberately away from the world origin.
///
/// # Where the expected value comes from
///
/// A rigid body on a hinge has one degree of freedom, so its equation of motion
/// is the scalar `I_hinge α = M`, where `M` is the moment of the applied forces
/// about the hinge axis and `I_hinge` is the moment of inertia about that axis.
/// With the axis along `z`, the centre of mass at `c`, the hinge at `r`, and the
/// parallel-axis theorem for the inertia,
///
/// ```text
///   M       = ((c - r) × m g)_z
///   I_hinge = I_cm,zz + m |c - r|²
///   α       = M / I_hinge
/// ```
///
/// Released from rest there is no centrifugal term, so this is the whole answer
/// for the first step, and `ω = α dt` after it. Every quantity on the right is
/// an input to the scene, so nothing here is read back out of the solver.
///
/// # Why the hinge is at `(0, 5, 0)`
///
/// Every other scene in this file puts its anchor at the world origin, where
/// `r` drops out of the arithmetic. This solver expresses spatial quantities
/// about the world origin, so an anchor that is *not* there is the only way to
/// check that the anchor is read at all rather than assumed. Mutating the
/// subspace to always anchor at the origin leaves every other oracle here
/// green; it reds this one.
///
/// # Why this one has a tolerance
///
/// The solver reaches `α` through `D⁻¹ (u - Uᵀ a')`, assembling `D` out of the
/// spatial inertia rather than by the parallel-axis formula above, so the two
/// routes to the same real number round differently in their last bits. The
/// bound is the same `2⁻⁴⁰` as oracle 9 and for the same reason: about twenty
/// binary orders above the rounding, and far below the fraction of `α` that any
/// wrong moment arm or wrong inertia would produce.
#[test]
fn compound_pendulum_matches_its_closed_form_angular_acceleration() {
    let hinge_position = Vec3Fix::from_int(0, 5, 0);
    let mut bodies = vec![
        RigidBody::new_static(hinge_position),
        RigidBody::new(Vec3Fix::from_int(4, 5, 0), Fix128::from_int(3)),
    ];

    let mut artic = ArticulatedBody::new(0, true);
    artic.add_link(
        0,
        1,
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            Vec3Fix::from_int(-4, 0, 0),
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        )),
        Vec3Fix::from_int(4, 0, 0),
    );

    // Inputs, read from the scene before it is stepped.
    let mass = Fix128::ONE / bodies[1].inv_mass;
    let inertia_about_com = Fix128::ONE / bodies[1].inv_inertia.z;
    let arm = bodies[1].position - hinge_position;

    let moment = arm.cross(gravity() * mass).z;
    let inertia_about_hinge = inertia_about_com + mass * arm.length_squared();
    let expected = moment / inertia_about_hinge * dt();

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, gravity(), dt());

    let tolerance = Fix128::from_ratio(1, 1i64 << 40);
    let error = bodies[1].angular_velocity.z - expected;
    assert!(
        error.abs() < tolerance,
        "a compound pendulum released from rest turns at M / I_hinge, so after \
         one step omega_z must be {} rad/s (M = {}, I_hinge = {}); got {}, off \
         by {} against a bound of {}",
        expected.to_f64(),
        moment.to_f64(),
        inertia_about_hinge.to_f64(),
        bodies[1].angular_velocity.z.to_f64(),
        error.to_f64(),
        tolerance.to_f64(),
    );
    assert_eq!(
        bodies[1].angular_velocity.x,
        Fix128::ZERO,
        "a hinge about z admits no rotation about x",
    );
    assert_eq!(
        bodies[1].angular_velocity.y,
        Fix128::ZERO,
        "a hinge about z admits no rotation about y",
    );
}
