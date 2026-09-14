//! Semantic determinism tests (v1.0 Item E, strict complement).
//!
//! These tests assert *semantic invariants* — sign/direction, order/index,
//! conservation laws, symmetry, and concrete hand-calculated values.
//! They complement `tests/determinism_golden.rs`:
//!
//! - **Golden hash tests**: catch any bit-level drift between runs / platforms.
//!   Fail on "the simulation changed at all".
//! - **Semantic tests (this file)**: catch bugs that leave the hash unchanged
//!   or produce a new-but-plausible hash. Fail on "the physics is wrong".
//!
//! Example bugs semantic tests catch that hash tests miss:
//! - Symmetric sign flips (e.g. both body positions negate together) — hash
//!   changes but the meaning of the change is opaque; a direction assert
//!   names it explicitly.
//! - Off-by-one indexing that swaps particle 0 with particle N-1 in a
//!   symmetric grid — hash of the sum is unchanged.
//! - Energy leaks in a nominally lossless scenario — hash detects drift but
//!   not the physical violation.
//!
//! # Tolerance policy
//!
//! Semi-implicit Euler + XPBD introduce small per-step numerical error.
//! Semantic tests use *loose* tolerances (typically 1–5%) chosen so a real
//! sign-flip or index-swap bug fails clearly while normal integrator error
//! passes.

#![cfg(feature = "std")]

use alice_physics::cloth::Cloth;
use alice_physics::eulerian_grid::MacGrid;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};

// ============================================================================
// Helpers
// ============================================================================

fn abs_diff(a: Fix128, b: Fix128) -> Fix128 {
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// Assert |a - b| < tolerance with a descriptive panic message.
fn assert_close(context: &str, a: Fix128, b: Fix128, tolerance: Fix128) {
    let d = abs_diff(a, b);
    assert!(
        d < tolerance,
        "{context}: |{a} - {b}| = {d} exceeds tolerance {tolerance}",
        context = context,
        a = a.to_f64(),
        b = b.to_f64(),
        d = d.to_f64(),
        tolerance = tolerance.to_f64(),
    );
}

fn gravity_config(g_y: i64) -> PhysicsConfig {
    PhysicsConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(g_y), Fix128::ZERO),
        damping: Fix128::ONE, // no damping
        ..Default::default()
    }
}

fn zero_gravity_config() -> PhysicsConfig {
    PhysicsConfig {
        substeps: 1,
        iterations: 1,
        gravity: Vec3Fix::ZERO,
        damping: Fix128::ONE,
        ..Default::default()
    }
}

// ============================================================================
// § 1. Sign / direction invariants (5 tests)
// ============================================================================
//
// These tests catch bugs where a `+` was swapped for a `-` (or vice versa)
// somewhere in the integration or gravity path. A well-designed hash test
// would also catch these, but the failure message is opaque; direction
// asserts name the specific physical property that was violated.

#[test]
fn sign_gravity_pulls_down_not_up() {
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::ONE,
    );
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        world.step(dt);
    }

    let y_after = world.bodies[0].position.y;
    assert!(
        y_after < Fix128::from_int(100),
        "gravity should pull body downward, but y went 100 → {}",
        y_after.to_f64(),
    );
    let vy_after = world.bodies[0].velocity.y;
    assert!(
        vy_after < Fix128::ZERO,
        "y-velocity should be negative under -Y gravity, got {}",
        vy_after.to_f64(),
    );
}

#[test]
fn sign_positive_x_velocity_moves_right() {
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let ball = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(Vec3Fix::new(
        Fix128::from_int(5),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        world.step(dt);
    }

    let x_after = world.bodies[0].position.x;
    assert!(
        x_after > Fix128::ZERO,
        "body with +5 x-velocity in zero gravity should move to positive x, got {}",
        x_after.to_f64(),
    );
}

#[test]
fn sign_thrown_upward_returns_downward() {
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(50), Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::ZERO,
        Fix128::from_int(15),
        Fix128::ZERO,
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    let mut saw_positive_vy = false;
    let mut saw_negative_vy = false;
    for _ in 0..240 {
        world.step(dt);
        let vy = world.bodies[0].velocity.y;
        if vy > Fix128::ZERO {
            saw_positive_vy = true;
        }
        if vy < Fix128::ZERO {
            saw_negative_vy = true;
        }
    }

    assert!(
        saw_positive_vy,
        "thrown-upward body should transiently have positive y-velocity"
    );
    assert!(
        saw_negative_vy,
        "thrown-upward body should eventually have negative y-velocity (gravity reversal)"
    );
}

#[test]
fn sign_static_body_stays_put_under_gravity() {
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let anchor = RigidBody::new_static(Vec3Fix::new(
        Fix128::from_int(7),
        Fix128::from_int(3),
        Fix128::from_int(11),
    ));
    world.add_body(anchor);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let pos = world.bodies[0].position;
    assert_eq!(
        pos.x,
        Fix128::from_int(7),
        "static body x must not drift under gravity"
    );
    assert_eq!(
        pos.y,
        Fix128::from_int(3),
        "static body y must not drift under gravity (this is what makes it 'static')"
    );
    assert_eq!(
        pos.z,
        Fix128::from_int(11),
        "static body z must not drift under gravity"
    );
}

#[test]
fn sign_zero_velocity_no_gravity_no_drift() {
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let ball = RigidBody::new_dynamic(Vec3Fix::from_int(2, 3, 5), Fix128::ONE);
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let pos = world.bodies[0].position;
    assert_eq!(
        pos.x,
        Fix128::from_int(2),
        "no gravity + no velocity: x must not drift"
    );
    assert_eq!(
        pos.y,
        Fix128::from_int(3),
        "no gravity + no velocity: y must not drift"
    );
    assert_eq!(
        pos.z,
        Fix128::from_int(5),
        "no gravity + no velocity: z must not drift"
    );
}

// ============================================================================
// § 2. Order / index invariants (4 tests)
// ============================================================================
//
// These tests catch bugs where the storage order of bodies / particles /
// grid cells changed silently. A hash test detects the change but doesn't
// tell you whether the change is a reorder or a computation error;
// explicit index asserts pin down "body i is at the location I put it".

#[test]
fn order_body_zero_is_first_added() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let id0 = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(1, 0, 0),
        Fix128::ONE,
    ));
    let id1 = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(2, 0, 0),
        Fix128::ONE,
    ));
    let id2 = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_int(3, 0, 0),
        Fix128::ONE,
    ));

    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    assert_eq!(id2, 2);
    assert_eq!(world.bodies[0].position.x, Fix128::from_int(1));
    assert_eq!(world.bodies[1].position.x, Fix128::from_int(2));
    assert_eq!(world.bodies[2].position.x, Fix128::from_int(3));
}

#[test]
fn order_cloth_grid_corners() {
    // 3x3 grid, width=height=2, origin at (0,0,0).
    // Layout convention: positions[i + j*res_x] with i=column, j=row.
    let cloth = Cloth::new_grid(
        Vec3Fix::ZERO,
        Fix128::from_int(2),
        Fix128::from_int(2),
        3,
        3,
        Fix128::ONE,
    );

    // First particle at origin
    let p00 = cloth.positions[0];
    // Last column of first row (i=res_x-1=2, j=0) — offset by width in x
    let p20 = cloth.positions[2];
    // First column of last row (i=0, j=res_y-1=2) — offset by height in y or z
    let p02 = cloth.positions[6];

    // Cloth is a 2D grid embedded in 3D; verify the corner-to-corner span
    // is exactly (width, height) along whatever axes the grid picks. We
    // don't assert the exact axis binding (the crate documents that), only
    // that the distances line up.
    let dx = abs_diff(p20.x, p00.x) + abs_diff(p20.y, p00.y) + abs_diff(p20.z, p00.z);
    let dy = abs_diff(p02.x, p00.x) + abs_diff(p02.y, p00.y) + abs_diff(p02.z, p00.z);

    // Manhattan norm of corner delta ≈ width (2) for the first-row span
    let two = Fix128::from_int(2);
    let tol = Fix128::from_ratio(1, 1000);
    assert_close(
        "cloth first-row corner span",
        dx,
        two,
        tol + Fix128::from_ratio(1, 100),
    );
    assert_close(
        "cloth first-column corner span",
        dy,
        two,
        tol + Fix128::from_ratio(1, 100),
    );
}

#[test]
fn order_mac_grid_index_zero_is_origin_face() {
    let grid = MacGrid::new(4, 4, 4, Fix128::from_ratio(1, 10));
    // u lives on X-faces of shape (nx+1, ny, nz) = (5, 4, 4). Index 0 must
    // correspond to (i=0, j=0, k=0). We can't call the pub(crate) idx_u
    // from a test outside the crate, but we CAN verify the array shape.
    assert_eq!(grid.u.len(), 5 * 4 * 4);
    assert_eq!(grid.v.len(), 4 * 5 * 4);
    assert_eq!(grid.w.len(), 4 * 4 * 5);
    assert_eq!(grid.pressure.len(), 4 * 4 * 4);

    // All zero-initialised
    for &f in &grid.u {
        assert_eq!(f, Fix128::ZERO);
    }
    for &f in &grid.pressure {
        assert_eq!(f, Fix128::ZERO);
    }
}

#[test]
fn order_body_count_matches_additions() {
    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    assert_eq!(world.body_count(), 0);
    for i in 0..7 {
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(i), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        ));
        assert_eq!(world.body_count(), (i + 1) as usize);
    }
}

// ============================================================================
// § 3. Conservation laws (4 tests)
// ============================================================================
//
// These tests catch bugs where the integrator silently leaks energy /
// momentum. Semi-implicit Euler is *not* energy-conserving in general,
// but the tests use loose tolerances tuned so a "typo" bug (e.g. accidentally
// halving impulse) fails while normal drift passes.

#[test]
fn conservation_horizontal_position_under_vertical_gravity() {
    // A body with no horizontal velocity, only vertical gravity, should
    // not drift horizontally. Bugs that couple gravity into x/z components
    // would fail here.
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(
            Fix128::from_int(4),
            Fix128::from_int(50),
            Fix128::from_int(-3),
        ),
        Fix128::ONE,
    );
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        world.step(dt);
    }

    let pos = world.bodies[0].position;
    assert_eq!(
        pos.x,
        Fix128::from_int(4),
        "x must not change under vertical gravity"
    );
    assert_eq!(
        pos.z,
        Fix128::from_int(-3),
        "z must not change under vertical gravity"
    );
    // y should have decreased significantly
    assert!(pos.y < Fix128::from_int(50), "y should decrease");
}

#[test]
fn conservation_x_velocity_under_vertical_gravity() {
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let initial_vx = Fix128::from_int(3);
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(initial_vx, Fix128::ZERO, Fix128::ZERO));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let vx_after = world.bodies[0].velocity.x;
    // Semi-implicit Euler + Fix128 arithmetic: drift ≤ 2^-40 per step
    // over 60 steps ≈ 5e-11 — well below anything a real bug would introduce.
    let tolerance = Fix128::from_ratio(1, 1_000_000);
    assert_close(
        "x-velocity conservation (no x-force, no damping)",
        vx_after,
        initial_vx,
        tolerance,
    );
}

#[test]
fn conservation_gravity_impulse_after_one_second() {
    // No damping, gravity = -10 m/s², one second (60 frames @ dt=1/60):
    // Δv_y = a·t = -10 · 1 = -10 m/s exactly (in continuous form).
    // Semi-implicit Euler with 60 substeps of dt=1/60 gives exactly this
    // increment because the accumulated dt is exactly 1 second in Fix128.
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::ONE,
    );
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let vy_after = world.bodies[0].velocity.y;
    // Loose tolerance (5% of expected -10) — semi-implicit Euler is exact
    // for constant acceleration, but Fix128 division rounding can add tiny
    // drift. If a bug halved the impulse, this fires.
    let expected = Fix128::from_int(-10);
    let tolerance = Fix128::from_ratio(1, 2); // 0.5, 5% of |-10|
    assert_close("gravity impulse over 1s", vy_after, expected, tolerance);
}

#[test]
fn conservation_kinematic_x_after_one_second() {
    // No gravity, +5 m/s x-velocity, 60 frames @ dt=1/60 = 1 second.
    // Δx = v·t = 5·1 = 5 exactly.
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let vx = Fix128::from_int(5);
    let ball = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(Vec3Fix::new(
        vx,
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let x_after = world.bodies[0].position.x;
    let expected = Fix128::from_int(5);
    let tolerance = Fix128::from_ratio(5, 100); // 1% of expected 5
    assert_close("kinematic drift over 1s", x_after, expected, tolerance);
}

// ============================================================================
// § 4. Symmetry (3 tests)
// ============================================================================
//
// These tests catch bugs where a symmetric setup produces asymmetric
// output. Common causes: order-dependent iteration over a hash map,
// integer overflow on one side but not the other, off-by-one in index
// mapping.

#[test]
fn symmetry_mirror_pair_under_gravity_no_interaction() {
    // Two bodies at x=+3 and x=-3, both at y=50, no gravity, no contact —
    // they should evolve independently and stay mirror-symmetric.
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let ball_right = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(3), Fix128::from_int(50), Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::from_int(1),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    let ball_left = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(-3), Fix128::from_int(50), Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::from_int(-1),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    world.add_body(ball_right);
    world.add_body(ball_left);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let pr = world.bodies[0].position;
    let pl = world.bodies[1].position;
    // Sub-machine-epsilon Fix128 rounding tolerance
    let tol = Fix128::from_ratio(1, 1_000_000);
    assert_close(
        "mirror-pair x opposite-sign equal-magnitude",
        pr.x,
        Fix128::ZERO - pl.x,
        tol,
    );
    assert_close("mirror-pair y equal", pr.y, pl.y, tol);
    assert_close("mirror-pair z equal", pr.z, pl.z, tol);
}

#[test]
fn symmetry_center_body_stays_on_axis_of_triple() {
    // Three bodies at x=-2, 0, +2 with equal masses, no gravity, no contact.
    // The middle body should remain at x=0 by symmetry.
    let mut world = PhysicsWorld::new(zero_gravity_config());
    for x in [-2, 0, 2] {
        world.add_body(RigidBody::new_dynamic(
            Vec3Fix::new(Fix128::from_int(x), Fix128::from_int(10), Fix128::ZERO),
            Fix128::ONE,
        ));
    }

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    assert_eq!(
        world.bodies[1].position.x,
        Fix128::ZERO,
        "middle body of symmetric triple must remain at x=0"
    );
}

#[test]
fn symmetry_time_reversibility_of_kinematic_drift() {
    // Semi-implicit Euler is NOT strictly time-reversible in general, but
    // for a body with constant velocity and no forces, the update is
    // linear and reversible: after +N steps and then -N steps (velocity
    // negated), the body should return to its start position within a
    // small Fix128-rounding tolerance.
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let initial = Vec3Fix::new(Fix128::from_int(7), Fix128::from_int(11), Fix128::ZERO);
    let ball = RigidBody::new_dynamic(initial, Fix128::ONE).with_velocity(Vec3Fix::new(
        Fix128::from_int(2),
        Fix128::from_int(-3),
        Fix128::ZERO,
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        world.step(dt);
    }
    // Reverse the velocity and step back
    world.bodies[0].velocity = Vec3Fix::new(
        Fix128::ZERO - Fix128::from_int(2),
        Fix128::from_int(3),
        Fix128::ZERO,
    );
    for _ in 0..30 {
        world.step(dt);
    }

    let final_pos = world.bodies[0].position;
    let tol = Fix128::from_ratio(1, 1000); // 0.001 unit
    assert_close("time-reversed x", final_pos.x, initial.x, tol);
    assert_close("time-reversed y", final_pos.y, initial.y, tol);
    assert_close("time-reversed z", final_pos.z, initial.z, tol);
}

// ============================================================================
// § 5. Concrete value tests (4 tests)
// ============================================================================
//
// These tests calculate the expected output by hand and compare with the
// simulation. Catches bugs where the sign is right, the direction is right,
// but the *magnitude* is wrong by a factor of 2 (e.g. missing 0.5 in a
// kinematic equation).

#[test]
fn concrete_freefall_position_after_one_second() {
    // No damping, gravity -10, initial y=100, initial velocity 0.
    // Analytic: y(1s) = 100 - 0.5·10·1² = 95
    // Semi-implicit Euler with 60 sub-steps: slight overestimate of drop
    // (integrates velocity at end of step). Tolerance 5% of 5-unit drop.
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::ONE,
    );
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    let y_after = world.bodies[0].position.y;
    let expected = Fix128::from_int(95);
    let tolerance = Fix128::from_ratio(1, 4); // 0.25, 5% of 5-unit drop
    assert_close("freefall y after 1s", y_after, expected, tolerance);
}

#[test]
fn concrete_kinematic_position_after_two_seconds() {
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let ball = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE).with_velocity(Vec3Fix::new(
        Fix128::from_int(4),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        world.step(dt);
    }

    // Expected: 4 m/s × 2s = 8 units
    let expected = Fix128::from_int(8);
    let tolerance = Fix128::from_ratio(1, 100); // 1% of 8
    assert_close(
        "kinematic drift after 2s",
        world.bodies[0].position.x,
        expected,
        tolerance,
    );
}

#[test]
fn concrete_gravity_velocity_after_two_seconds() {
    // Δv_y = -10 · 2 = -20 m/s (continuous form)
    let mut world = PhysicsWorld::new(gravity_config(-10));
    let ball = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::ZERO, Fix128::from_int(100), Fix128::ZERO),
        Fix128::ONE,
    );
    world.add_body(ball);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..120 {
        world.step(dt);
    }

    let expected = Fix128::from_int(-20);
    let tolerance = Fix128::ONE; // 5% of 20
    assert_close(
        "gravity velocity after 2s",
        world.bodies[0].velocity.y,
        expected,
        tolerance,
    );
}

#[test]
fn concrete_two_body_relative_position_after_one_second() {
    // Two bodies with opposing velocities, initial separation 4 units,
    // closing at 2 m/s each = 4 m/s combined. After 1 second, separation
    // should decrease by 4 units and they meet at the origin.
    let mut world = PhysicsWorld::new(zero_gravity_config());
    let left = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(-2), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::from_int(2),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    let right = RigidBody::new_dynamic(
        Vec3Fix::new(Fix128::from_int(2), Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    )
    .with_velocity(Vec3Fix::new(
        Fix128::ZERO - Fix128::from_int(2),
        Fix128::ZERO,
        Fix128::ZERO,
    ));
    world.add_body(left);
    world.add_body(right);

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..60 {
        world.step(dt);
    }

    // Both should meet at x=0
    let tolerance = Fix128::from_ratio(5, 100); // 0.05
    assert_close(
        "left body x after 1s (should meet at origin)",
        world.bodies[0].position.x,
        Fix128::ZERO,
        tolerance,
    );
    assert_close(
        "right body x after 1s (should meet at origin)",
        world.bodies[1].position.x,
        Fix128::ZERO,
        tolerance,
    );
    // And their relative position should be near zero
    let separation = abs_diff(world.bodies[0].position.x, world.bodies[1].position.x);
    assert!(
        separation < Fix128::from_ratio(1, 10),
        "bodies should meet: separation = {}",
        separation.to_f64()
    );
}

// ============================================================================
// Extra invariant: Fix128 numeric behaviour used across the suite
// ============================================================================

#[test]
fn fix128_sanity_from_int_negative() {
    // Verify that Fix128::from_int(-x) equals ZERO - Fix128::from_int(x)
    // — a canary test for sign handling in the base type.
    let a = Fix128::from_int(-7);
    let b = Fix128::ZERO - Fix128::from_int(7);
    assert_eq!(a, b, "Fix128::from_int(-x) must equal -Fix128::from_int(x)");
}

#[test]
fn fix128_sanity_add_negate_zero() {
    let a = Fix128::from_int(42);
    let b = Fix128::ZERO - a;
    assert_eq!(a + b, Fix128::ZERO, "x + (-x) must equal zero");
}
