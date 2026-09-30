//! Analytic oracles for articulated-body forward dynamics
//! (`FeatherstoneSolver`).
//!
//! Every expected value in this file is derived from mechanics below, never
//! by running the implementation and recording what came out. The properties
//! pinned here were chosen because each one holds for *any* correct multibody
//! formulation — they do not depend on whether links are modelled as point
//! masses on massless rods or as rigid bodies with their own inertia tensors,
//! nor on the integrator's order. That makes them safe to assert exactly rather
//! than within a tuned tolerance. The exceptions are the closed-form oracles
//! 11, 12, 13 and 14, which do read the scene's mass and inertia; each states
//! its derivation and carries a bound instead of an exact equality.
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
//! 12. [`centrifugal_coupling_to_a_rotating_parent_matches_its_closed_form`] —
//!     the velocity-product acceleration `c = v × (v - v_parent)`, which is
//!     identically zero in every scene whose parent is at rest.
//! 13. [`anisotropic_inertia_is_carried_into_world_axes`] — the conjugation
//!     `R I Rᵀ`, which is invisible to every scene whose bodies are spheres.
//! 14. [`double_pendulum_rank_reduction_matches_its_closed_form`] — oracle 10's
//!     subject, the rank-`n` reduction `-U D⁻¹ Uᵀ`, against a closed form over
//!     the rationals rather than a qualitative "these two differ".
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
//! Two escaped that set, and both were gaps in the scenes rather than in the
//! solver. Oracles 12 and 13 were written to close them, and the mutations were
//! re-run to confirm that they do:
//!
//! | mutation | oracles red |
//! |---|---|
//! | `c = v × (v - v_parent)` forced to zero | 12 only |
//! | the two operands of that cross product swapped | 12 only |
//! | `R I Rᵀ` → `I` | 13 only |
//! | `R I Rᵀ` → `Rᵀ I R` | 13 only |
//!
//! Why the other eleven were blind to each:
//!
//! * **Dropping the velocity-product acceleration `c = v × (v - v_parent)`.**
//!   `v × v` is identically zero, so `c` vanishes for every link whose parent is
//!   at rest — which is every link in every scene here that has a static base.
//!   The one scene with a moving parent is oracle 9, and dropping `c` there
//!   solves a different but still self-consistent system, so momentum is still
//!   conserved and oracle 9 cannot see it either. Oracle 12 closes it with a
//!   *kinematic* parent: an infinitely massive base carrying a prescribed
//!   angular velocity, which the solver reads and never writes, so the link's
//!   hinge point accelerates centripetally by an amount fixed by the scene.
//!   Under the mutation that scene answers exactly zero.
//! * **Not rotating the inertia tensor into world axes (`R I Rᵀ` → `I`).**
//!   Every body in oracles 1 to 12, and in oracle 14, has an isotropic inertia
//!   — `RigidBody::new` builds `diag(2m/5)`, oracle 8 scales it uniformly and
//!   oracle 14 replaces it with `diag(2, 2, 2)` — and
//!   `R (k·1) Rᵀ = k·1` exactly, so the rotation has nothing to act on. Oracle
//!   13 gives one body `diag(2, 5, 10)` and turns it by the 120° rotation about
//!   `(1,1,1)`, whose quaternion `(½,½,½,½)` is exact in `Fix128`; the world
//!   hinge axis then samples `I_yy` rather than `I_zz`, and the mutation lands
//!   on the unturned body's answer instead.
//!
//! Both of these were still-open gaps when this file was first landed, which is
//! the general lesson they carry: a term that is *identically* zero across a
//! whole test set is not thereby verified. A scene has to be built for which it
//! is the only nonzero thing in the answer.
//!
//! Oracle 14 was then written to turn oracle 10's qualitative claim into a
//! closed form, and four more mutations were run against the whole set. Three
//! are caught and one is not:
//!
//! | mutation | oracles red | oracle 10 | oracle 14 | where the mutant lands |
//! |---|---|---|---|---|
//! | `-U D⁻¹ Uᵀ` dropped (`inertia.sub(correction)` → `inertia`) | 9, 14 | **green** | red | `ω₁ = -0.0326797385620915`, exactly the welded `-5/153`; `ω₂` flips to `-5/306` |
//! | the bias term `U D⁻¹ u` dropped | 9 | green | **green** | — |
//! | the child's spatial inertia never added to the parent's | 9, 14 | **green** | red | `ω₁ = -0.0757575757575758` |
//! | the gravity substitution `a₀ := -a_g` dropped | 2, 3, 4, 5, 7, 8, 10, 11, 13, 14 | red | red | everything stops: `ω₁ = 0` |
//!
//! Read the oracle 10 column: it is red for one of those four, and *not* for the
//! rank-`n` deletion, which is the term oracle 10 is about. A qualitative
//! `assert_ne!` asks whether two scenes differ, so it cannot see a mutation that
//! leaves them differing while moving both by the wrong amount. That is the
//! whole reason oracle 14 exists, and the two columns above are the measurement
//! of it rather than an argument for it.
//!
//! The bias term escapes, and for the same structural reason as the two above:
//! `u = τ - Sᵀ p^A` has `τ = 0` (no motors anywhere in this file), a leaf's
//! `p^A` is `v ×* (I v)` which vanishes at rest, and gravity never enters `p^A`
//! because the base-acceleration substitution carries it. So `u` is
//! *identically* zero in every scene released from rest — which is oracles 11,
//! 12, 13 and 14, all of them first-step oracles. The only thing that sees the
//! term is oracle 9, whose chain is already moving. Closing it needs a
//! double pendulum released with nonzero angular velocity, whose closed form
//! carries the velocity-product terms this file has so far been able to drop;
//! that is a new scene rather than a change here, and it is filed in the
//! backlog rather than fixed. The narrower lesson is that a first-step oracle
//! released from rest buys its cheap closed form by killing every term
//! proportional to velocity.
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
/// give the two arms the same shoulder motion and fail here. Every other scene
/// in this file that moves has only one moving link, so this is the only place
/// two moving links meet — but the claim here is only that the two arms differ.
/// [`double_pendulum_rank_reduction_matches_its_closed_form`] pins the same
/// reduction quantitatively, and the two are not interchangeable: dropping the
/// rank update leaves *this* test green, measured. The mutated solver still
/// gives the two elbows different shoulder motions, so "they differ" still
/// holds while both numbers are wrong.
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

// ---------------------------------------------------------------------------
// Oracle 12 — the velocity-product term, seen from a moving parent
// ---------------------------------------------------------------------------

/// A link hinged to a **rotating** base picks up the angular acceleration that
/// the centripetal motion of its own hinge point demands, and nothing else.
///
/// # Why this scene exists
///
/// The velocity-product acceleration is `c = v × (v - v_parent)`. The spatial
/// motion cross product is alternating, so `v × v` is identically zero and `c`
/// collapses whenever the parent is at rest. Every other scene in this file
/// hangs its chain from a base that never moves, so every one of them would
/// pass with `c` deleted outright. The only way to make `c` load-bearing is a
/// parent that is genuinely moving while the joint to it is free to turn.
///
/// The base here is a body of infinite mass carrying a prescribed constant
/// angular velocity `Omega` about the world `z` axis through the origin — a
/// kinematic base, the standard way to impose a motion rather than solve for
/// one. The solver never writes to it (it is the held root), so `Omega` is a
/// constant of the scene rather than a state that has to be integrated
/// correctly for the oracle to mean anything. Its spatial acceleration about
/// the origin is exactly zero, which is what a held root is given, so the
/// scene is self-consistent.
///
/// # Where the expected value comes from
///
/// Take moments about the hinge point `P`, which is a point of the rotating
/// base at radius `d` and therefore accelerates centripetally,
/// `a_P = -Omega^2 P`. For a rigid body of mass `m`, centroidal inertia `I_c`
/// about the hinge axis, and `r = c - P` the vector from hinge to centre of
/// mass,
///
/// ```text
///   sum M_P = I_c alpha + m (r x a_c)_z ,   a_c = a_P + alpha x r - omega^2 r
/// ```
///
/// The hinge force acts at `P` and so contributes no moment about it, and there
/// is no gravity in this scene, so `sum M_P = 0`. The body is released with
/// zero angular velocity, so the `omega^2 r` term drops; `r x (alpha x r)` is
/// `alpha |r|^2`; and with `r = (0, L, 0)` perpendicular to `a_P = (-Omega^2 d,
/// 0, 0)` the remaining cross product is `L Omega^2 d` along `z`. Hence
///
/// ```text
///   alpha = - m L Omega^2 d / (I_c + m L^2)
/// ```
///
/// With `m = 5`, `L = 3`, `d = 2`, `Omega = 4` and `I_c = 2 m / 5 = 2` that is
/// `-480 / 47 rad/s^2`, and after one semi-implicit Euler step of `1/60 s` the
/// link turns at `-8/47 rad/s`. Every quantity on the right is an input to the
/// scene.
///
/// Note that `alpha` does not depend on the link's own angular velocity at all
/// in this configuration, which is why the link may be released from rest: the
/// entire answer is the centrifugal coupling to the parent.
///
/// # Why one step
///
/// The closed form above is evaluated at the initial configuration. The scene's
/// kinematic base keeps its velocity but not its orientation (a held root is
/// never integrated), so the prescribed motion is exact at `t = 0` and the
/// oracle is a first-step oracle, like oracle 11.
///
/// # Why the companion case is here
///
/// With `Omega = 0` the same scene is a link hanging off a motionless base in
/// zero gravity, whose acceleration is exactly zero. Asserting both halves
/// makes the pair non-degenerate: a solver that always answers zero fails the
/// first, and one that invents motion fails the second. Deleting `c` makes the
/// first half answer exactly zero — the link's bias force `v x* (I v)` vanishes
/// for pure translation, so with `c` gone there is no nonzero term left
/// anywhere in the scene.
#[test]
fn centrifugal_coupling_to_a_rotating_parent_matches_its_closed_form() {
    let spin = Fix128::from_int(4);
    let moving = spun_base_scene(spin);
    let still = spun_base_scene(Fix128::ZERO);

    // Inputs, all read from the scene description rather than from a run.
    let mass = Fix128::from_int(5);
    let hinge_radius = Fix128::from_int(2); // d
    let arm = Fix128::from_int(3); // L
    let inertia_about_com = mass * Fix128::from_ratio(2, 5); // RigidBody::new
    let expected = (Fix128::ZERO - mass * arm * spin * spin * hinge_radius)
        / (inertia_about_com + mass * arm * arm)
        * dt();

    let tolerance = Fix128::from_ratio(1, 1i64 << 40);
    let error = moving.z - expected;
    assert!(
        error.abs() < tolerance,
        "a link hinged to a base spinning at Omega = {} feels its hinge point \
         accelerate centripetally, so after one step omega_z must be \
         -m L Omega^2 d / (I_c + m L^2) * dt = {} rad/s; got {}, off by {} \
         against a bound of {}. An exact zero here means the velocity-product \
         acceleration c = v x (v - v_parent) never reached the answer",
        spin.to_f64(),
        expected.to_f64(),
        moving.z.to_f64(),
        error.to_f64(),
        tolerance.to_f64(),
    );
    assert_eq!(
        Vec3Fix::new(moving.x, moving.y, Fix128::ZERO),
        Vec3Fix::ZERO,
        "a hinge about z admits no rotation about x or y, got {moving:?}",
    );
    assert_eq!(
        still,
        Vec3Fix::ZERO,
        "with the base at rest and no gravity the same scene is in \
         equilibrium, so the link must not start turning, got {still:?}",
    );
}

/// One step of a link hinged about `z` to a base spinning at `spin`, in zero
/// gravity. Returns the link's angular velocity afterwards.
///
/// The base sits at the origin with infinite mass and carries the prescribed
/// angular velocity. The hinge point is at `(2, 0, 0)`, a point of the base at
/// radius `d = 2`, and the link's centre of mass is at `(2, 3, 0)`, i.e. an arm
/// of `L = 3` perpendicular to that radius. The link starts with the velocity
/// that rigid attachment demands and no rotation of its own: every point of it
/// moves at `Omega z x P`, the velocity of the hinge point.
fn spun_base_scene(spin: Fix128) -> Vec3Fix {
    let hinge = Vec3Fix::from_int(2, 0, 0);
    let mut base = RigidBody::new_static(Vec3Fix::ZERO);
    base.angular_velocity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, spin);

    let mut link = RigidBody::new(Vec3Fix::from_int(2, 3, 0), Fix128::from_int(5));
    // v = Omega z x hinge = (0, 2 Omega, 0), shared by every point of the link
    // because it is not rotating.
    link.velocity = Vec3Fix::new(Fix128::ZERO, spin * hinge.x, Fix128::ZERO);

    let mut bodies = vec![base, link];
    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ true);
    artic.add_link(
        0,
        1,
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            hinge,
            Vec3Fix::from_int(0, -3, 0),
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        )),
        Vec3Fix::from_int(0, 3, 0),
    );

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, Vec3Fix::ZERO, dt());
    bodies[1].angular_velocity
}

// ---------------------------------------------------------------------------
// Oracle 13 — the inertia tensor is carried into world axes
// ---------------------------------------------------------------------------

/// A compound pendulum whose body is anisotropic and turned swings at the rate
/// set by the inertia **about the world hinge axis**, not by the stored
/// diagonal.
///
/// # Why this scene exists
///
/// `RigidBody` stores its inertia as a diagonal in body axes, so the spatial
/// inertia has to conjugate it into world axes as `R I R^T`. Every other body
/// in this file is a sphere — `RigidBody::new` builds `diag(2m/5)` and oracle 8
/// scales it uniformly — and `R (k 1) R^T = k 1` exactly for any `R`, so
/// dropping the conjugation changes nothing anywhere else in the file. Making
/// it load-bearing needs an anisotropic inertia on a body that is actually
/// turned relative to the world.
///
/// # Where the expected value comes from
///
/// A rigid body on a hinge has one degree of freedom, so released from rest
/// (no gyroscopic term) with a stationary hinge point it obeys the scalar
///
/// ```text
///   alpha  = M_axis / I_axis
///   M_axis = ((c - P) x m g) . n
///   I_axis = n^T (R I_local R^T) n + m |(c - P) perpendicular to n|^2
/// ```
///
/// the second term being the parallel-axis theorem for the distance from the
/// centre of mass to the hinge *line*.
///
/// The rotation used is `q = (1/2, 1/2, 1/2, 1/2)`, the 120-degree turn about
/// `(1, 1, 1)` that cycles `x -> y -> z -> x`. It is a unit quaternion whose
/// every component is a power of two, so it is exact in `Fix128` and so is the
/// conjugation. Its inverse carries the world `z` axis onto the body `y` axis,
/// so `n^T (R I_local R^T) n` is exactly `I_yy` — the anisotropy is selected by
/// the rotation, not merely scaled by it.
///
/// With `I_local = diag(2, 5, 10)`, `m = 3`, the hinge at the origin and the
/// centre of mass at `(4, 0, 0)`:
///
/// ```text
///   turned:    I_axis = I_yy + m |r|^2 = 5 + 48 = 53 ,  alpha = -120 / 53
///   upright:   I_axis = I_zz + m |r|^2 = 10 + 48 = 58 , alpha = -120 / 58
/// ```
///
/// Both halves are asserted. The turned one fails if the conjugation is
/// dropped, because it then answers with `I_zz`; the upright one fails if the
/// conjugation is applied where it should not be, and it holds the scene
/// honest by showing that the two answers differ only by the rotation.
#[test]
fn anisotropic_inertia_is_carried_into_world_axes() {
    // q = (1/2, 1/2, 1/2, 1/2): 120 degrees about (1, 1, 1), x -> y -> z -> x.
    let half = Fix128::from_ratio(1, 2);
    let turned = QuatFix::new(half, half, half, half);

    let mass = Fix128::from_int(3);
    let arm = Fix128::from_int(4);
    // I_local = diag(2, 5, 10); the world hinge axis picks out I_yy when the
    // body is turned and I_zz when it is not.
    let local_inertia = Vec3Fix::from_int(2, 5, 10);
    let moment = Fix128::ZERO - mass * arm * Fix128::from_int(10); // (r x m g)_z

    let leverage = mass * arm * arm;
    let expect_turned = moment / (local_inertia.y + leverage) * dt();
    let expect_upright = moment / (local_inertia.z + leverage) * dt();

    assert_ne!(
        expect_turned, expect_upright,
        "the scene is only a test of the conjugation if the two rotations \
         give different answers",
    );

    let tolerance = Fix128::from_ratio(1, 1i64 << 40);
    for (label, rotation, expected, other) in [
        ("turned", turned, expect_turned, expect_upright),
        ("upright", QuatFix::IDENTITY, expect_upright, expect_turned),
    ] {
        let got = anisotropic_pendulum(rotation, mass, arm, local_inertia);
        let error = got.z - expected;
        assert!(
            error.abs() < tolerance,
            "{label} pendulum: a hinge about world z swings at M / I_axis with \
             I_axis = n^T R I_local R^T n + m |r|^2, so omega_z after one step \
             must be {} rad/s; got {}, off by {} against a bound of {}. The \
             other rotation's answer is {} rad/s — landing on that one means \
             the inertia was never carried into world axes",
            expected.to_f64(),
            got.z.to_f64(),
            error.to_f64(),
            tolerance.to_f64(),
            other.to_f64(),
        );
        assert_eq!(
            Vec3Fix::new(got.x, got.y, Fix128::ZERO),
            Vec3Fix::ZERO,
            "{label} pendulum: a hinge about z admits no rotation about x or \
             y, got {got:?}",
        );
    }
}

/// One step of an anisotropic compound pendulum hinged about world `z` at the
/// origin, released from rest under gravity, with the body pre-rotated by
/// `rotation`. Returns its angular velocity afterwards.
fn anisotropic_pendulum(
    rotation: QuatFix,
    mass: Fix128,
    arm: Fix128,
    local_inertia: Vec3Fix,
) -> Vec3Fix {
    let mut link = RigidBody::new(Vec3Fix::new(arm, Fix128::ZERO, Fix128::ZERO), mass);
    link.rotation = rotation;
    link.inv_inertia = Vec3Fix::new(
        Fix128::ONE / local_inertia.x,
        Fix128::ONE / local_inertia.y,
        Fix128::ONE / local_inertia.z,
    );

    // The hinge in the link's own frame is `R^-1 (-arm, 0, 0)`. Only the
    // parent-side anchor enters the motion subspace, so this is bookkeeping
    // for `forward_kinematics` rather than an input to the oracle.
    let local_anchor = rotation.conjugate().rotate_vec(Vec3Fix::new(
        Fix128::ZERO - arm,
        Fix128::ZERO,
        Fix128::ZERO,
    ));

    let mut bodies = vec![RigidBody::new_static(Vec3Fix::ZERO), link];
    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ true);
    artic.add_link(
        0,
        1,
        Joint::Hinge(HingeJoint::new(
            0,
            1,
            Vec3Fix::ZERO,
            local_anchor,
            Vec3Fix::from_int(0, 0, 1),
            Vec3Fix::from_int(0, 0, 1),
        )),
        Vec3Fix::new(arm, Fix128::ZERO, Fix128::ZERO),
    );

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, gravity(), dt());
    bodies[1].angular_velocity
}

// ---------------------------------------------------------------------------
// Oracle 14 — the rank-`n` reduction, against a closed form over the rationals
// ---------------------------------------------------------------------------

/// A planar two-link arm released from rest, horizontal along `+x`, hinged
/// about world `z` at the origin, with `outboard` as the elbow. Returns the
/// angular velocities of the upper arm and the forearm after one step.
///
/// Both links get an inertia of exactly 2 about every axis, written as the
/// reciprocal `1/2` rather than left at `RigidBody::new`'s default `2m/5`:
/// `2/5` is not a dyadic rational, so the default is 2 only up to the rounding
/// of `from_ratio(2, 5)`, and the closed form below wants it exact.
fn horizontal_double_pendulum(outboard: Joint) -> (Vec3Fix, Vec3Fix) {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new(Vec3Fix::from_int(2, 0, 0), Fix128::from_int(5)),
        RigidBody::new(Vec3Fix::from_int(6, 0, 0), Fix128::from_int(5)),
    ];
    let half = Fix128::from_ratio(1, 2);
    bodies[1].inv_inertia = Vec3Fix::new(half, half, half);
    bodies[2].inv_inertia = Vec3Fix::new(half, half, half);

    let mut artic = ArticulatedBody::new(0, /* fixed_base = */ true);
    let upper = artic.add_link(
        0,
        1,
        // Shoulder at the world origin, hinged about +z.
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
    // Elbow at (4, 0, 0) = body 1's position plus its local anchor. Only the
    // parent-side anchor enters the motion subspace; the child-side one is
    // bookkeeping for `forward_kinematics`, so it is set to the value that
    // agrees with it in world coordinates rather than being load-bearing here.
    artic.add_link(upper, 2, outboard, Vec3Fix::from_int(4, 0, 0));

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, gravity(), dt());
    (bodies[1].angular_velocity, bodies[2].angular_velocity)
}

/// The elbow of the arm in [`horizontal_double_pendulum`], as a hinge about
/// world `z`.
fn elbow_hinge() -> Joint {
    Joint::Hinge(HingeJoint::new(
        1,
        2,
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(-2, 0, 0),
        Vec3Fix::from_int(0, 0, 1),
        Vec3Fix::from_int(0, 0, 1),
    ))
}

/// The same elbow, welded.
fn elbow_weld() -> Joint {
    Joint::Fixed(FixedJoint::new(
        1,
        2,
        Vec3Fix::from_int(2, 0, 0),
        Vec3Fix::from_int(-2, 0, 0),
        QuatFix::IDENTITY,
    ))
}

/// The rank-`n` reduction `-U D⁻¹ Uᵀ` matches a closed form over the rationals,
/// and its sign is the thing that tells it apart from having no reduction at
/// all: with a free elbow the forearm's angular acceleration comes out
/// *against* gravity.
///
/// # Where the expected values come from
///
/// The scene is a planar double pendulum released from rest, so the Lagrangian
/// route gives the whole answer in two lines. Take `θ₁` and `θ₂` as the two
/// links' absolute angles from straight down, `c₁` and `c₂` as each link's
/// hinge-to-centre distance, `L₁` as the shoulder-to-elbow distance, and `I₁`,
/// `I₂` as the centroidal inertias about `z`. Then
///
/// ```text
///   T = ½(I₁ + m₁c₁² + m₂L₁²) θ̇₁² + ½(I₂ + m₂c₂²) θ̇₂²
///       + m₂ L₁ c₂ cos(θ₁ - θ₂) θ̇₁ θ̇₂
///   V = -m₁ g c₁ cos θ₁ - m₂ g (L₁ cos θ₁ + c₂ cos θ₂)
/// ```
///
/// so the mass matrix and the gravity gradient are
///
/// ```text
///   M₁₁ = I₁ + m₁c₁² + m₂L₁²      M₁₂ = M₂₁ = m₂ L₁ c₂ cos(θ₁ - θ₂)
///   M₂₂ = I₂ + m₂c₂²
///   G₁  = (m₁c₁ + m₂L₁) g sin θ₁   G₂ = m₂ c₂ g sin θ₂
/// ```
///
/// Released from rest every term quadratic in `θ̇` drops out, leaving
/// `M θ̈ = -G`, which is a 2×2 solve. Here the arm lies along `+x` so both
/// angles are exactly a right angle — `sin = 1`, `cos = 0`, so `θ₁ - θ₂ = 0`
/// and `cos(θ₁ - θ₂) = 1` — and with `m₁ = m₂ = 5`, `I₁ = I₂ = 2`, `c₁ = c₂ = 2`,
/// `L₁ = 4`, `g = 10`:
///
/// ```text
///   M₁₁ = 2 + 20 + 80 = 102   M₁₂ = 5·4·2 = 40   M₂₂ = 2 + 20 = 22
///   G₁  = (10 + 20)·10 = 300  G₂  = 10·10 = 100
///   det = 102·22 - 40² = 2244 - 1600 = 644
///   θ̈₁ = -(M₂₂G₁ - M₁₂G₂)/det = -(6600 - 4000)/644 = -650/161
///   θ̈₂ = -(M₁₁G₂ - M₁₂G₁)/det = -(10200 - 12000)/644 = +450/161
/// ```
///
/// After one step of `dt = 1/60`, `ω = θ̈ dt`, so `ω₁ = -65/966` and
/// `ω₂ = +15/322`. A rotation about `+z` carries `+x` towards `+y`, and
/// increasing `θ` from a right angle does the same, so `θ̇` is `ω_z` with no
/// sign flip.
///
/// The derivation was checked against the textbook case of two uniform rods of
/// length `L` (`I = mL²/12`, `c = L/2`, `L₁ = L`) in the same formulae, which
/// gives `θ̈₁ = -9g/(7L)` and `θ̈₂ = +3g/(7L)`; both come out of the algebra
/// above.
///
/// # Why the sign of `ω₂` is the answer
///
/// `θ̈₂` is *positive* while gravity pulls the forearm down. The forearm is not
/// being lifted by gravity: the shoulder is dropping fast enough that, in the
/// frame of the falling upper arm, the elbow is thrown the other way. That is
/// the coupling `M₁₂` doing its work, and on the solver's side it is exactly
/// the `-U D⁻¹ Uᵀ` reduction plus its bias `U D⁻¹ u`. A solver that passed each
/// child's unreduced rigid inertia up to its parent has no such coupling and
/// lands on the welded answer, where both links turn the same way as gravity.
/// So the sign is asserted on its own line, not just inside the tolerance.
///
/// # The welded twin
///
/// With the elbow welded the arm is one rigid body on the shoulder hinge, and
/// the reduction term is identically zero because a zero-degree-of-freedom
/// joint has no `U` at all. Then `I_hinge α = M` about `z`:
///
/// ```text
///   I_hinge = I₁ + m₁c₁² + I₂ + m₂(L₁ + c₂)² = 2 + 20 + 2 + 180 = 204
///   M       = -(m₁ c₁ + m₂(L₁ + c₂)) g = -(10 + 30)·10 = -400
///   α       = -400/204 = -100/51        ω = α dt = -5/153
/// ```
///
/// and both links share that `ω`, being rigidly joined. Asserting the two
/// scenes together is what makes the pair non-degenerate: the hinged numbers
/// alone could be met by a solver that gets the magnitudes right for the wrong
/// reason, and the welded numbers alone are what the *unreduced* solver returns
/// for both. [`outboard_joint_changes_what_the_inboard_link_feels`] already
/// asserts that the two scenes differ; this one says by how much.
///
/// # Why this one has a tolerance
///
/// Same reason as oracle 11. The solver reaches these accelerations through
/// `D⁻¹ (u - Uᵀ a')` with `D` assembled out of 6×6 spatial inertias about the
/// world origin, not by the 2×2 solve above, so the two routes to the same real
/// numbers round differently in their last bits. The bound is the same `2⁻⁴⁰`:
/// about twenty binary orders above the rounding, and orders below the shift
/// that a missing reduction, a missing bias or a wrong moment arm would make —
/// each of those moves `ω₂` by more than `0.04`, which is `2³⁴` times the bound.
#[test]
fn double_pendulum_rank_reduction_matches_its_closed_form() {
    let tolerance = Fix128::from_ratio(1, 1i64 << 40);

    // --- hinged elbow: the reduction is active -----------------------------
    let (upper, forearm) = horizontal_double_pendulum(elbow_hinge());

    let want_upper = Fix128::from_ratio(-65, 966);
    let want_forearm = Fix128::from_ratio(15, 322);

    let upper_error = upper.z - want_upper;
    assert!(
        upper_error.abs() < tolerance,
        "the upper arm of a horizontal double pendulum released from rest turns \
         at theta1ddot = -650/161 rad/s^2, so after one step omega_z must be \
         {} rad/s; got {}, off by {} against a bound of {}",
        want_upper.to_f64(),
        upper.z.to_f64(),
        upper_error.to_f64(),
        tolerance.to_f64(),
    );

    let forearm_error = forearm.z - want_forearm;
    assert!(
        forearm_error.abs() < tolerance,
        "the forearm turns at theta2ddot = +450/161 rad/s^2, so after one step \
         omega_z must be {} rad/s; got {}, off by {} against a bound of {}",
        want_forearm.to_f64(),
        forearm.z.to_f64(),
        forearm_error.to_f64(),
        tolerance.to_f64(),
    );

    // The sign, on its own line: this is what a solver without the rank-`n`
    // reduction cannot produce, whatever its magnitudes.
    assert!(
        forearm.z > Fix128::ZERO,
        "the forearm's angular acceleration is against gravity, which only the \
         coupling M12 -- the `-U D^-1 U^T` reduction and its bias -- can \
         produce; a solver passing unreduced child inertias upward turns it the \
         same way as gravity, got {}",
        forearm.z.to_f64(),
    );

    for (name, spin) in [("upper arm", upper), ("forearm", forearm)] {
        assert_eq!(
            spin.x,
            Fix128::ZERO,
            "{name}: a hinge about z admits no rotation about x",
        );
        assert_eq!(
            spin.y,
            Fix128::ZERO,
            "{name}: a hinge about z admits no rotation about y",
        );
    }

    // --- welded elbow: the reduction is identically zero -------------------
    let (welded_upper, welded_forearm) = horizontal_double_pendulum(elbow_weld());

    let want_welded = Fix128::from_ratio(-5, 153);
    for (name, spin) in [("upper arm", welded_upper), ("forearm", welded_forearm)] {
        let error = spin.z - want_welded;
        assert!(
            error.abs() < tolerance,
            "welded, the arm is one rigid body on the shoulder hinge with \
             I_hinge = 204 and M = -400, so alpha = -100/51 and after one step \
             {name} omega_z must be {} rad/s; got {}, off by {} against a bound \
             of {}",
            want_welded.to_f64(),
            spin.z.to_f64(),
            error.to_f64(),
            tolerance.to_f64(),
        );
        assert_eq!(
            spin.x,
            Fix128::ZERO,
            "welded {name}: a hinge about z admits no rotation about x",
        );
        assert_eq!(
            spin.y,
            Fix128::ZERO,
            "welded {name}: a hinge about z admits no rotation about y",
        );
    }
}
