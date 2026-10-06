//! Articulated body chain — production entry point for `articulation`'s
//! public API.
//!
//! `ArticulatedBody` (`add_link` / `link_count` / `dof_count` / `body_indices`
//! / `joints` / `forward_kinematics` / `apply_motors` / `set_motor`),
//! `build_ragdoll` and `FeatherstoneSolver::{solve, solve_with_mass_splitting}`
//! had no caller outside their own `#[cfg(test)]` module before this file —
//! the physics `solve` produces is already pinned exhaustively by
//! `tests/analytic_multibody_dynamics.rs` (14 closed-form oracles); this
//! example is the one production path that actually calls all eleven items,
//! and the companion `tests/analytic_articulation_wiring.rs` covers the API
//! surface that file does not (`forward_kinematics`, `apply_motors`,
//! `set_motor`, `body_indices`, `joints`, `build_ragdoll`) plus degenerate
//! input.
//!
//! ```bash
//! cargo run --example articulated_body_chain --features std
//! ```

use alice_physics::articulation::{build_ragdoll, ArticulatedBody, FeatherstoneSolver};
use alice_physics::joint::{BallJoint, HingeJoint, Joint};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::motor::PdController;
use alice_physics::solver::RigidBody;

/// Assert `got` is within `tol` of `expected`, printing both on failure.
fn assert_close(label: &str, got: Fix128, expected: Fix128, tol: Fix128) {
    let err = (got - expected).abs();
    assert!(
        err < tol,
        "{label}: got {} expected {} (|err| = {} >= tol {})",
        got.to_f64(),
        expected.to_f64(),
        err.to_f64(),
        tol.to_f64(),
    );
}

fn main() {
    section_chain_topology();
    section_forward_kinematics();
    section_motors();
    section_solve_pendulum();
    section_solve_with_mass_splitting();
    section_build_ragdoll();
    println!("[articulation] all sections passed");
}

/// `add_link` / `link_count` / `dof_count` / `body_indices` / `joints`: build
/// a 3-link chain (root + 2 children, a Ball then a Hinge joint) and check
/// the bookkeeping each accessor reports about it.
fn section_chain_topology() {
    let mut artic = ArticulatedBody::new(0, false);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );
    let _link2 = artic.add_link(
        link1,
        2,
        Joint::Hinge(HingeJoint::new(
            1,
            2,
            Vec3Fix::ZERO,
            Vec3Fix::ZERO,
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
        Vec3Fix::from_int(0, 2, 0),
    );

    assert_eq!(artic.link_count(), 3, "root + 2 add_link calls = 3 links");
    // dof_count sums the joints' freedom: ball 3 + hinge 1
    assert_eq!(artic.dof_count(), 4);
    assert_eq!(artic.body_indices(), vec![0, 1, 2]);
    assert_eq!(artic.joints().len(), 2);

    println!(
        "[articulation] chain_topology: link_count={} dof_count={} body_indices={:?} joints={}",
        artic.link_count(),
        artic.dof_count(),
        artic.body_indices(),
        artic.joints().len(),
    );
}

/// `forward_kinematics`: a root rotated 180 degrees about `z` — the exact
/// unit quaternion `(x=0, y=0, z=1, w=0)`, constructed directly (no
/// `sin_cos` CORDIC call, so the rotation itself is bit-exact) — must rotate
/// `(x, y, 0)` to `(-x, -y, 0)`. Direct quaternion algebra:
/// `q*v*q⁻¹` with `q=(0,0,1,0)`, `v=(0,2,0,0)`:
///   `q*v = (-2, 0, 0, 0)`, `q⁻¹ = (0,0,-1,0)`, `(q*v)*q⁻¹ = (0, -2, 0, 0)`.
/// So link 1 (offset `(0,2,0)` from the rotated root) lands at `(0,-2,0)`,
/// and link 2 (offset `(0,2,0)` from link 1, whose own rotation is left at
/// `IDENTITY`) lands at `(0,-2,0) + (0,2,0) = (0,0,0)`.
fn section_forward_kinematics() {
    let mut bodies = vec![
        RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(5)),
        RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(3)),
        RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2)),
    ];
    bodies[0].rotation = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

    let mut artic = ArticulatedBody::new(0, true);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );
    artic.add_link(
        link1,
        2,
        Joint::Ball(BallJoint::new(1, 2, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(0, 2, 0),
    );

    artic.forward_kinematics(&mut bodies);

    assert_eq!(bodies[1].position, Vec3Fix::from_int(0, -2, 0));
    assert_eq!(bodies[2].position, Vec3Fix::ZERO);
    println!(
        "[articulation] forward_kinematics: link1={:?} link2={:?} (expected (0,-2,0), (0,0,0))",
        (bodies[1].position.x.to_f64(), bodies[1].position.y.to_f64()),
        (bodies[2].position.x.to_f64(), bodies[2].position.y.to_f64()),
    );
}

/// `set_motor` / `apply_motors`: static root (body 0) with one Ball-jointed
/// dynamic link (body 1, `inv_mass = 1`) at `(3,0,0)`. A Position-mode motor
/// with `kp=4`, `kd=0`, target separation `7`: `error = 7-3 = 4`,
/// `force = kp*error = 16` (within `max_force`), `direction = (1,0,0)`,
/// `dt = 1/2` ⇒ `impulse = 16 * 1/2 = 8` along `(1,0,0)`. The static root has
/// `inv_mass = 0` so it is not pushed; the dynamic link's velocity becomes
/// exactly `(8,0,0)`.
fn section_motors() {
    let mut bodies = vec![
        RigidBody::new_static(Vec3Fix::ZERO),
        RigidBody::new_dynamic(Vec3Fix::from_int(3, 0, 0), Fix128::ONE),
    ];

    let mut artic = ArticulatedBody::new(0, true);
    let link1 = artic.add_link(
        0,
        1,
        Joint::Ball(BallJoint::new(0, 1, Vec3Fix::ZERO, Vec3Fix::ZERO)),
        Vec3Fix::from_int(3, 0, 0),
    );
    let mut motor = PdController::new(Fix128::from_int(4), Fix128::ZERO, Fix128::from_int(1000));
    motor.set_position_target(Fix128::from_int(7));
    artic.set_motor(link1, motor);
    // Out-of-range link index must be a silent no-op (documented behavior,
    // re-checked independently of the src unit test in
    // `tests/analytic_articulation_wiring.rs`).
    artic.set_motor(99, motor);

    let dt = Fix128::from_ratio(1, 2);
    artic.apply_motors(&mut bodies, dt);

    assert_eq!(
        bodies[0].velocity,
        Vec3Fix::ZERO,
        "static root must not move"
    );
    assert_eq!(bodies[1].velocity, Vec3Fix::from_int(8, 0, 0));
    println!(
        "[articulation] motors: body1 velocity = {:?} (expected (8,0,0))",
        (
            bodies[1].velocity.x.to_f64(),
            bodies[1].velocity.y.to_f64(),
            bodies[1].velocity.z.to_f64()
        )
    );
}

/// `FeatherstoneSolver::solve`: a single-link pendulum hinged about `z`,
/// released from rest — the textbook compound-pendulum angular acceleration
/// `alpha = M / I_hinge` with `M` the gravitational moment about the hinge
/// and `I_hinge = I_com + m*|arm|^2` (parallel-axis theorem). Values are read
/// from the scene before stepping, never from the solver, and the physics
/// itself is already pinned to 2^-40 by
/// `tests/analytic_multibody_dynamics.rs::compound_pendulum_matches_its_closed_form_angular_acceleration`
/// — this section exists to give `solve` an actual production caller.
fn section_solve_pendulum() {
    let hinge = Vec3Fix::from_int(0, 5, 0);
    let mut bodies = vec![
        RigidBody::new_static(hinge),
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
            Vec3Fix::UNIT_Z,
            Vec3Fix::UNIT_Z,
        )),
        Vec3Fix::from_int(4, 0, 0),
    );

    let mass = Fix128::ONE / bodies[1].inv_mass;
    let inertia_com = Fix128::ONE / bodies[1].inv_inertia.z;
    let arm = bodies[1].position - hinge;
    let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let moment = arm.cross(gravity * mass).z;
    let inertia_hinge = inertia_com + mass * arm.length_squared();
    let dt = Fix128::from_ratio(1, 60);
    let expected_omega_z = moment / inertia_hinge * dt;

    let mut solver = FeatherstoneSolver::new();
    solver.solve(&artic, &mut bodies, gravity, dt);

    assert_close(
        "pendulum omega_z",
        bodies[1].angular_velocity.z,
        expected_omega_z,
        Fix128::from_ratio(1, 1i64 << 40),
    );
    println!(
        "[articulation] solve: omega_z = {:.6} expected = {:.6}",
        bodies[1].angular_velocity.z.to_f64(),
        expected_omega_z.to_f64(),
    );
}

/// `FeatherstoneSolver::solve_with_mass_splitting`: a single free (unjointed)
/// root body under gravity has `bias = 0` identically (no rotation is ever
/// introduced — see the doc derivation in
/// `tests/analytic_articulation_wiring.rs`), so its acceleration is exactly
/// `gravity` at every re-solve. Semi-implicit Euler integrated as **two**
/// half-steps of `dt/2` therefore reaches a different position than **one**
/// full step of `dt`, by exactly `-(1/4) * g * dt^2`, while both reach the
/// same final velocity `g*dt`. With a single body, `min_inv_mass ==
/// max_inv_mass`, so `mass_ratio_split_threshold <= 1` activates splitting
/// and `> 1` leaves it inert — both branches are exercised below.
fn section_solve_with_mass_splitting() {
    let gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 4);

    let scene = || {
        let artic = ArticulatedBody::new(0, false);
        let bodies = vec![RigidBody::new(Vec3Fix::ZERO, Fix128::from_int(2))];
        (artic, bodies)
    };

    let (artic_full, mut bodies_full) = scene();
    FeatherstoneSolver::new().solve(&artic_full, &mut bodies_full, gravity, dt);

    let (artic_split, mut bodies_split) = scene();
    FeatherstoneSolver::new().solve_with_mass_splitting(
        &artic_split,
        &mut bodies_split,
        gravity,
        dt,
        Fix128::ONE, // threshold 1, ratio 1 (single body) -> splitting ACTIVE
    );

    let (artic_inert, mut bodies_inert) = scene();
    FeatherstoneSolver::new().solve_with_mass_splitting(
        &artic_inert,
        &mut bodies_inert,
        gravity,
        dt,
        Fix128::from_int(2), // threshold 2 > ratio 1 -> splitting INERT
    );

    let expected_diff = Fix128::from_ratio(-1, 4) * gravity.y * dt * dt;
    let actual_diff = bodies_split[0].position.y - bodies_full[0].position.y;
    assert_close(
        "split vs full position difference",
        actual_diff,
        expected_diff,
        Fix128::from_ratio(1, 1i64 << 60),
    );
    assert_eq!(
        bodies_split[0].velocity, bodies_full[0].velocity,
        "constant acceleration: both schemes reach the same final velocity"
    );
    assert_eq!(
        bodies_inert[0].position, bodies_full[0].position,
        "below-threshold ratio must forward to a single plain solve, bit-exact"
    );

    println!(
        "[articulation] solve_with_mass_splitting: split.y={:.6} full.y={:.6} diff={:.6} (expected {:.6})",
        bodies_split[0].position.y.to_f64(),
        bodies_full[0].position.y.to_f64(),
        actual_diff.to_f64(),
        expected_diff.to_f64(),
    );
}

/// `build_ragdoll`: a nonzero, non-default `body_start_index` must offset
/// every produced body index — a mutation that ignores the argument would
/// still pass a default (`0`) call.
fn section_build_ragdoll() {
    let start = 1000;
    let (artic, bodies) = build_ragdoll(Vec3Fix::ZERO, start);

    assert_eq!(bodies.len(), 12, "ragdoll has 12 bodies");
    assert_eq!(artic.link_count(), 12, "ragdoll has 12 links");
    let indices = artic.body_indices();
    assert_eq!(
        indices[0], start,
        "pelvis body index must honor body_start_index"
    );
    assert!(
        indices.iter().all(|&i| (start..start + 12).contains(&i)),
        "every body index must fall in [start, start+12), got {indices:?}"
    );
    assert_eq!(
        artic.joints().len(),
        11,
        "11 non-root links, each with a joint"
    );
    // 7 ball joints * 3 + 4 hinges * 1
    assert_eq!(artic.dof_count(), 25);

    println!(
        "[articulation] build_ragdoll: bodies={} links={} dof={} indices[0..3]={:?}",
        bodies.len(),
        artic.link_count(),
        artic.dof_count(),
        &indices[..3],
    );
}
