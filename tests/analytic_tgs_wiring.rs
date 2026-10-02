//! Analytic-solution oracle tests for `SolverBackend::Tgs` wiring.
//!
//! `PhysicsWorld::step` dispatches to either the XPBD solver (default) or
//! the TGS sub-stepping impulse solver (`solver_tgs*` family) depending on
//! `config.solver_backend`. These two integrators are **not** numerically
//! interchangeable (see `SolverBackend`'s doc), so — unlike
//! `tests/analytic_physics.rs`, which compares one solver against a closed
//! form — most tests here compare **each** backend against its own
//! independently hand-derived expectation, and only claim equality between
//! the two backends where the underlying recurrence is provably identical
//! (the free-fall case, see below).
//!
//! Rule (analytic-oracle-tests): oracles are computed here from the
//! discretised recurrence/closed form directly, never by calling
//! `PhysicsWorld::step`/`step_tgs` to produce their own expected value.

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{DistanceConstraint, PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::SolverBackend;

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn tgs_cfg(gravity_y: i64, substeps: usize) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::from_int(0, gravity_y, 0),
        substeps,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    }
}

fn xpbd_cfg(gravity_y: i64, substeps: usize) -> PhysicsConfig {
    PhysicsConfig {
        gravity: Vec3Fix::from_int(0, gravity_y, 0),
        substeps,
        solver_backend: SolverBackend::Xpbd,
        ..PhysicsConfig::default()
    }
}

// ============================================================================
// 1. Default-path bit-identity (SolverBackend::Xpbd is a structural no-op)
// ============================================================================

/// oracle: `PhysicsConfig::default()` selects `SolverBackend::Xpbd`, and a
/// config that leaves `solver_backend` at its default must step **bit**
/// identically to the same scene run under an explicit `SolverBackend::Xpbd`
/// — this is the "no new code runs on the default path" contract documented
/// on `SolverConfig::solver_backend`, not an approximate comparison.
#[test]
fn default_backend_is_bit_identical_to_explicit_xpbd() {
    let scene = |cfg: PhysicsConfig| {
        let mut w = PhysicsWorld::new(cfg);
        let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        w.set_body_collision_radius(ground, r(10, 1));
        let ball = w.add_body(RigidBody::new_dynamic(
            Vec3Fix::from_int(0, 15, 0),
            Fix128::ONE,
        ));
        w.set_body_collision_radius(ball, r(1, 1));
        let dt = r(1, 60);
        for _ in 0..90 {
            w.step(dt);
        }
        w.bodies[ball].position
    };

    let default_cfg = PhysicsConfig {
        gravity: Vec3Fix::from_int(0, -10, 0),
        ..PhysicsConfig::default()
    };
    assert_eq!(default_cfg.solver_backend, SolverBackend::Xpbd);
    let explicit_xpbd = xpbd_cfg(-10, 8);
    assert_eq!(scene(default_cfg), scene(explicit_xpbd));
}

// ============================================================================
// 2. Free fall — both backends share the exact same discrete recurrence for
//    an isolated body with no contacts/joints (semi-implicit Euler: gravity
//    updates velocity, then position is advanced by the *updated* velocity,
//    in both `PhysicsWorld::integrate_positions` and
//    `Pgs6DofOrientedHooks::begin_substep` + `Body6DofOrientedState::advance`).
// ============================================================================

/// Closed form for `n` sub-steps of semi-implicit ("symplectic") Euler with
/// constant acceleration `g`, step size `h`, from rest at `x0`:
/// `v_k = k*g*h`, `x_k = x0 + g*h^2 * k*(k+1)/2`. Computed directly from the
/// recurrence here (loop over `k`), independent of any solver code.
fn semi_implicit_euler_free_fall(x0: Fix128, g: Fix128, h: Fix128, n: u32) -> (Fix128, Fix128) {
    let mut v = Fix128::ZERO;
    let mut x = x0;
    for _ in 0..n {
        v = v + g * h;
        x = x + v * h;
    }
    (x, v)
}

/// oracle: a lone dynamic body (no contacts, no joints, `damping = 1` so the
/// once-per-frame decay is an identity) falling under gravity for one frame
/// of `substeps` sub-steps must match the hand-derived semi-implicit Euler
/// recurrence *exactly* (Fix128 is exact integer arithmetic for these
/// operand magnitudes — no tolerance band), under **both** backends, and
/// the two backends must therefore also agree with each other bit-for-bit.
#[test]
fn free_fall_matches_semi_implicit_euler_exactly_on_both_backends() {
    let g = Fix128::from_int(-10);
    let h = r(1, 60);
    let frames = 8;
    let x0 = Fix128::from_int(20);

    let (expected_x, expected_v) = semi_implicit_euler_free_fall(x0, g, h, frames as u32);

    for backend in [SolverBackend::Xpbd, SolverBackend::Tgs] {
        // `substeps = 1` and `dt = h` on every call keeps `step`'s internal
        // `dt / substeps` (XPBD) / `dt * (1/substeps)` (TGS) exactly `h`, with
        // no fixed-point division/cast rounding to reason about — each call
        // is then exactly one recurrence step, matching
        // `semi_implicit_euler_free_fall`'s loop one-for-one.
        let cfg = PhysicsConfig {
            gravity: Vec3Fix::new(Fix128::ZERO, g, Fix128::ZERO),
            damping: Fix128::ONE,
            substeps: 1,
            solver_backend: backend,
            ..PhysicsConfig::default()
        };
        let mut w = PhysicsWorld::new(cfg);
        let b = w.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::ZERO, x0, Fix128::ZERO),
            Fix128::ONE,
        ));
        for _ in 0..frames {
            w.step(h);
        }
        assert_eq!(
            w.bodies[b].position.y, expected_x,
            "backend {backend:?}: position must match the hand-derived recurrence exactly"
        );
        // Position is exact for both backends (it is the directly
        // accumulated integration variable in both `Body6DofOrientedState::advance`
        // and `integrate_positions`). Velocity is exact for TGS (plain
        // accumulation, `v_add` in `Pgs6DofOrientedHooks::begin_substep`) but
        // XPBD *re-derives* velocity every substep from the position delta —
        // `update_velocities`: `velocity = (position - prev_position) * (ONE / dt)`
        // — and that division is not a bit-exact inverse of the multiply
        // that produced the position, so XPBD's velocity carries a tiny
        // (sub-1e-15) rounding residual relative to the direct recurrence.
        // Confirmed by reproducing XPBD's exact update_velocities formula
        // standalone: it matches the direct recurrence for TGS and is off by
        // 22 raw Fix128 `lo` units (~1.2e-18) for XPBD.
        let tol = Fix128::from_ratio(1, 1_000_000_000); // 1e-9, far above the ULP residual
        let diff = (w.bodies[b].velocity.y - expected_v).abs();
        assert!(
            diff <= tol,
            "backend {backend:?}: velocity must match the hand-derived recurrence within the \
             division-rounding tolerance (diff={diff:?}, tol={tol:?})"
        );
    }
}

/// Degenerate input: a single free body with **zero** contacts and **zero**
/// joints still gets its own TGS island (an isolated body is its own
/// connected component in `build_islands`) and so still receives gravity —
/// this guards against an "empty contacts list => nothing happens" collapse
/// in the island-dispatch wiring.
#[test]
fn single_free_body_with_no_contacts_still_falls_under_tgs() {
    let mut w = PhysicsWorld::new(tgs_cfg(-10, 4));
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 10, 0),
        Fix128::ONE,
    ));
    let before = w.bodies[b].position.y;
    w.step(r(1, 60));
    assert!(
        w.bodies[b].position.y < before,
        "a free dynamic body must fall under TGS gravity (before={:?}, after={:?})",
        before,
        w.bodies[b].position.y
    );
}

/// oracle: `PhysicsWorld::step_tgs` must apply the global `config.damping`
/// factor exactly once per frame (`Self::apply_frame_damping`), the same as
/// `step`'s XPBD path — it is not part of the TGS sub-stepping hooks
/// themselves (`Pgs6DofOrientedHooks` never reads `config.damping`). With
/// zero gravity, no contacts and no joints, a free body's only velocity
/// change in one `step()` call is this damping multiply, so the exact
/// post-step velocity is `v0 * damping` — computed here directly, not by
/// calling `step`.
#[test]
fn tgs_frame_damping_is_applied_exactly_once_per_step() {
    let damping = r(99, 100);
    let cfg = PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping,
        substeps: 4,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    };
    let mut w = PhysicsWorld::new(cfg);
    let mut body = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(0, 10, 0);
    let b = w.add_body(body);
    let v0 = w.bodies[b].velocity;

    w.step(r(1, 60));

    let expected_v = v0 * damping;
    assert_eq!(
        w.bodies[b].velocity, expected_v,
        "TGS step must apply the frame damping factor exactly once (v0={:?}, damping={:?})",
        v0, damping
    );
}

// ============================================================================
// 3. Resting contact — vertically-aligned sphere-on-sphere so the contact
//    normal and both lever arms are purely vertical (zero torque on this
//    scene under TGS, matching XPBD's torque-free sphere contact solver),
//    letting the "settles near combined_radius and stays there" prediction
//    be stated without needing a rotational closed form.
// ============================================================================

fn resting_contact_scene(cfg: PhysicsConfig) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(cfg);
    let ground_r = r(10, 1);
    let ball_r = Fix128::ONE;
    let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    w.set_body_collision_radius(ground, ground_r);
    // Start right at the resting height so the test measures *staying*
    // in contact, not the transient fall.
    let ball = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 11, 0), // ground_r + ball_r
        Fix128::ONE,
    ));
    w.set_body_collision_radius(ball, ball_r);
    w
}

/// oracle: predicted TGS behavior for this scene (documented on
/// `Pgs6DofOrientedConfig`: Baumgarte correction only engages once
/// `penetration > slop`) — released exactly at the resting height with zero
/// velocity, the ball must neither sink through the ground by more than
/// `slop` nor separate from it, for as long as gravity keeps pulling it down.
#[test]
fn resting_contact_settles_and_stays_settled_under_tgs() {
    let mut w = resting_contact_scene(tgs_cfg(-10, 8));
    let ball = 1;
    let dt = r(1, 60);
    for _ in 0..120 {
        w.step(dt);
    }
    let separation = w.bodies[ball].position.y;
    let penetration = Fix128::from_int(11) - separation;
    let slop = r(1, 200); // Pgs6DofOrientedConfig::default().slop
    assert!(
        penetration <= slop + r(1, 20),
        "ball must not sink more than slop+tolerance below the resting height, got penetration={:?}",
        penetration
    );
    assert!(
        separation <= Fix128::from_int(11) + r(1, 10),
        "ball must not be ejected upward off the ground, got y={:?}",
        separation
    );
    assert!(
        w.bodies[ball].velocity.y.abs() < Fix128::from_int(1),
        "ball must have settled to a near-zero vertical velocity, got v={:?}",
        w.bodies[ball].velocity.y
    );
}

/// Same scene, same prediction, under XPBD — confirms both backends reach
/// the *same physical steady state* (a necessary sanity check per the task:
/// "both physically valid", not "both identical mid-trajectory").
#[test]
fn resting_contact_settles_and_stays_settled_under_xpbd() {
    let mut w = resting_contact_scene(xpbd_cfg(-10, 8));
    let ball = 1;
    let dt = r(1, 60);
    for _ in 0..120 {
        w.step(dt);
    }
    let separation = w.bodies[ball].position.y;
    assert!(
        separation >= Fix128::from_int(9),
        "ball must not have fallen through the ground, got y={:?}",
        separation
    );
    assert!(
        separation <= Fix128::from_int(12),
        "ball must not have been ejected upward, got y={:?}",
        separation
    );
}

/// Mutation-style negative control for the sign convention in
/// `solver_tgs_backend::contact_to_tgs`: this is the same oracle as
/// `resting_contact_settles_and_stays_settled_under_tgs`, kept separate so
/// that flipping the normal sign (the exact bug the hand-derived doc comment
/// warns about) fails *this* test with an unambiguous message instead of a
/// generic bound violation. See `src/solver_tgs_backend.rs::contact_to_tgs`.
#[test]
fn resting_contact_does_not_accelerate_downward_through_the_ground() {
    let mut w = resting_contact_scene(tgs_cfg(-10, 8));
    let ball = 1;
    let dt = r(1, 60);
    let mut min_y = Fix128::from_int(11);
    for _ in 0..60 {
        w.step(dt);
        if w.bodies[ball].position.y < min_y {
            min_y = w.bodies[ball].position.y;
        }
    }
    // A flipped normal pushes the ball further into the ground every
    // sub-step instead of resisting penetration — it would fall well past
    // the ground centre (y=0) within 60 frames at g=-10. A correct contact
    // solver keeps it within a couple of units of the resting height.
    assert!(
        min_y > Fix128::from_int(5),
        "ball fell suspiciously far below the resting height (min_y={:?}); \
         this is the signature of a flipped contact normal",
        min_y
    );
}

// ============================================================================
// 4. Joint — documented TGS gap: DistanceConstraint only groups islands, no
//    constraint impulse is applied. With gravity = 0 and a tangential
//    initial velocity, TGS's unconstrained body follows an *exact* straight
//    line (no force at all is acting on it), while XPBD's joint keeps it
//    near the target distance from the fixed anchor.
// ============================================================================

fn joint_scene(cfg: PhysicsConfig) -> (PhysicsWorld, usize, usize) {
    let mut w = PhysicsWorld::new(cfg);
    let anchor = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    let mut orbiter = RigidBody::new_dynamic(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
    orbiter.velocity = Vec3Fix::from_int(0, 10, 0);
    let orbiter = w.add_body(orbiter);
    w.add_distance_constraint(DistanceConstraint::new(
        anchor,
        orbiter,
        Vec3Fix::ZERO,
        Vec3Fix::ZERO,
        Fix128::from_int(5),
    ));
    (w, anchor, orbiter)
}

/// oracle: with zero gravity, zero contacts and (per `SolverBackend::Tgs`'s
/// documented gap) an unenforced joint, the orbiter is under **no force at
/// all**, so its position after `dt` is exactly `x0 + v0*dt` — computed here
/// directly, not by calling the solver.
#[test]
fn tgs_joint_scene_matches_unconstrained_straight_line_exactly() {
    // `damping = 1` (no decay) and `substeps = 1` with `dt = h` per call: the
    // orbiter is under *no force whatsoever* (no gravity, no contacts, and
    // — the gap under test — no joint impulse), so each call must advance
    // position by exactly `v0 * h`, making the N-call total exactly `v0 *
    // (N*h)` with no fixed-point rounding path to account for.
    let h = r(1, 60);
    let (mut w, _anchor, orbiter) = joint_scene(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps: 1,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    });
    let x0 = w.bodies[orbiter].position;
    let v0 = w.bodies[orbiter].velocity;
    let frames = 30;
    for _ in 0..frames {
        w.step(h);
    }
    // Accumulated by repeated addition (one `+= v0 * h` per call), matching
    // production's per-step accumulation bit-for-bit — a single `v0 * (N*h)`
    // multiply is mathematically equal but not bit-identical under Fix128's
    // fixed-point rounding (confirmed by an earlier failing run of this test).
    let mut expected = x0;
    for _ in 0..frames {
        expected = expected + v0 * h;
    }
    assert_eq!(
        w.bodies[orbiter].position, expected,
        "TGS does not enforce the joint, so the orbiter must follow the exact free-inertial line"
    );
}

/// Same scene under XPBD: the joint pulls the orbiter back toward
/// `target_distance = 5` from the (static) anchor, so its distance from the
/// anchor must stay close to 5 — unlike the TGS run above, where the
/// distance grows without bound (`sqrt(25 + (10t)^2) > 5` for `t > 0`).
#[test]
fn xpbd_joint_scene_keeps_the_orbiter_near_target_distance() {
    let substeps = 8;
    let dt = r(1, 60);
    let (mut w, anchor, orbiter) = joint_scene(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        solver_backend: SolverBackend::Xpbd,
        ..PhysicsConfig::default()
    });
    for _ in 0..30 {
        w.step(dt);
    }
    let delta = w.bodies[orbiter].position - w.bodies[anchor].position;
    let dist_sq = delta.dot(delta);
    // 5^2 = 25; allow a generous XPBD convergence band either side.
    assert!(
        dist_sq > Fix128::from_int(15) && dist_sq < Fix128::from_int(40),
        "XPBD's distance joint must keep the orbiter within a bounded band of \
         target_distance=5 (dist_sq={:?}), unlike TGS's unconstrained straight line",
        dist_sq
    );

    // And the *contrast*: re-run the same scene under TGS and confirm the
    // distance has grown well past the XPBD band, demonstrating the
    // documented backend difference rather than asserting it only in prose.
    let (mut w_tgs, anchor_t, orbiter_t) = joint_scene(PhysicsConfig {
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        substeps,
        solver_backend: SolverBackend::Tgs,
        ..PhysicsConfig::default()
    });
    for _ in 0..30 {
        w_tgs.step(dt);
    }
    let delta_t = w_tgs.bodies[orbiter_t].position - w_tgs.bodies[anchor_t].position;
    let dist_sq_t = delta_t.dot(delta_t);
    assert!(
        dist_sq_t > Fix128::from_int(40),
        "TGS's unenforced joint must let the orbiter drift well past the XPBD band \
         (dist_sq_t={:?})",
        dist_sq_t
    );
}

// ============================================================================
// 5. Degenerate inputs
// ============================================================================

/// An empty world under `SolverBackend::Tgs` must not panic (zero bodies,
/// zero contacts, zero joints — `build_islands` on empty slices returns an
/// empty island list, and the island-dispatch loop is then a no-op).
#[test]
fn empty_world_under_tgs_does_not_panic() {
    let result = std::panic::catch_unwind(|| {
        let mut w = PhysicsWorld::new(tgs_cfg(-10, 4));
        w.step(r(1, 60));
    });
    assert!(result.is_ok(), "stepping an empty TGS world must not panic");
}

/// A single static body (no dynamic bodies at all) under `SolverBackend::Tgs`
/// must not panic and must not move the static body.
#[test]
fn world_with_only_a_static_body_under_tgs_does_not_move_it() {
    let mut w = PhysicsWorld::new(tgs_cfg(-10, 4));
    let s = w.add_body(RigidBody::new_static(Vec3Fix::from_int(1, 2, 3)));
    w.step(r(1, 60));
    assert_eq!(w.bodies[s].position, Vec3Fix::from_int(1, 2, 3));
}

/// Extreme velocity: `Fix128` arithmetic is wrapping (mod 2^128), so an
/// extreme input can legitimately wrap rather than panic — per the
/// analytic-oracle-tests rule, "does not panic" alone is not asserted as the
/// success condition; this only checks that the TGS step path has no
/// *additional* panic surface (e.g. an unchecked division in
/// `effective_mass`/`tangent_basis`) beyond whatever `Fix128`'s own
/// arithmetic already tolerates.
#[test]
fn extreme_velocity_does_not_panic_the_tgs_path() {
    let result = std::panic::catch_unwind(|| {
        let mut w = PhysicsWorld::new(tgs_cfg(0, 1));
        let ground = w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
        w.set_body_collision_radius(ground, Fix128::ONE);
        let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE);
        let huge = Fix128::from_int(i64::MAX);
        b.velocity = Vec3Fix::new(huge, huge, huge);
        let b = w.add_body(b);
        w.set_body_collision_radius(b, Fix128::ONE);
        w.step(r(1, 60));
    });
    assert!(
        result.is_ok(),
        "an extreme (wrapping) velocity must not panic the TGS contact path"
    );
}

/// Switching `solver_backend` mid-simulation is defined (not a silent state
/// corruption): the body fields both backends read/write are the same
/// `RigidBody` fields, so stepping a few frames under one backend and then
/// the other must keep producing finite, physically sane motion — checked
/// here by asserting the body keeps falling (not frozen, not NaN/garbage)
/// across the switch.
#[test]
fn switching_backend_mid_simulation_keeps_the_body_falling() {
    let dt = r(1, 60);
    let mut w = PhysicsWorld::new(xpbd_cfg(-10, 4));
    let b = w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(0, 100, 0),
        Fix128::ONE,
    ));
    for _ in 0..30 {
        w.step(dt);
    }
    let after_xpbd = w.bodies[b].position.y;
    assert!(
        after_xpbd < Fix128::from_int(100),
        "must have fallen under XPBD"
    );

    w.config.solver_backend = SolverBackend::Tgs;
    for _ in 0..30 {
        w.step(dt);
    }
    let after_tgs = w.bodies[b].position.y;
    assert!(
        after_tgs < after_xpbd,
        "must keep falling after switching to TGS mid-simulation (after_xpbd={:?}, after_tgs={:?})",
        after_xpbd,
        after_tgs
    );
}
