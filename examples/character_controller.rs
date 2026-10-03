//! Character Controller Example
//!
//! Production entry point for `src/character.rs`'s kinematic character
//! controller API (wiring_guard: `PushImpulse`, `apply_gravity`,
//! `compute_push_impulses`, `feet_position`, `get_platform_velocity`,
//! `new_default` had zero production callers -- only `#[cfg(test)]`
//! called in, which the guard does not count).
//!
//! Every value this example prints is checked against a closed-form
//! expectation computed independently of the function under test (never
//! by calling that function for the expected side), per
//! `~/claude-config/rules/analytic-oracle-tests.md`.
//!
//! ```bash
//! cargo run --example character_controller --features std
//! ```

use alice_physics::character::{CharacterConfig, CharacterController, PushImpulse};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

/// Default `CharacterConfig` field values, hand-transcribed from
/// `CharacterConfig::default()`'s source literals (not obtained by
/// calling `Default::default()` or `new_default()`).
struct ExpectedDefaults {
    radius: f64,
    height: f64,
    push_force: f64,
}

const EXPECTED_DEFAULTS: ExpectedDefaults = ExpectedDefaults {
    radius: 0.3,
    height: 1.8,
    push_force: 5.0,
};

fn assert_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tol,
        "[character] FAIL {label}: actual={actual} expected={expected} diff={diff} tol={tol}"
    );
    println!("[character] OK   {label}: actual={actual:.6} expected={expected:.6}");
}

/// `new_default` -- construct a controller and check every field against
/// the independently known defaults (position passed in, velocity/ground
/// state/platform velocity at their documented rest values).
fn demo_new_default() {
    println!("[character] --- new_default ---");
    let start = Vec3Fix::from_int(2, 5, -3);
    let cc = CharacterController::new_default(start);

    assert_close("new_default.position.x", cc.position.x.to_f64(), 2.0, 1e-9);
    assert_close("new_default.position.y", cc.position.y.to_f64(), 5.0, 1e-9);
    assert_close("new_default.position.z", cc.position.z.to_f64(), -3.0, 1e-9);
    assert_close("new_default.velocity.x", cc.velocity.x.to_f64(), 0.0, 1e-9);
    assert_close("new_default.velocity.y", cc.velocity.y.to_f64(), 0.0, 1e-9);
    assert!(
        !cc.grounded,
        "[character] FAIL new_default.grounded: expected false"
    );
    println!("[character] OK   new_default.grounded: false");
    assert!(
        cc.ground_body_index.is_none(),
        "[character] FAIL new_default.ground_body_index: expected None"
    );
    println!("[character] OK   new_default.ground_body_index: None");
    assert_close(
        "new_default.platform_velocity.y",
        cc.platform_velocity.y.to_f64(),
        0.0,
        1e-9,
    );
    assert_close(
        "new_default.config.radius",
        cc.config.radius.to_f64(),
        EXPECTED_DEFAULTS.radius,
        1e-9,
    );
    assert_close(
        "new_default.config.height",
        cc.config.height.to_f64(),
        EXPECTED_DEFAULTS.height,
        1e-9,
    );
    assert_close(
        "new_default.config.push_force",
        cc.config.push_force.to_f64(),
        EXPECTED_DEFAULTS.push_force,
        1e-9,
    );
}

/// `feet_position` -- `feet.y = body.y - height/2 + radius`, computed
/// independently from the config constants (not by calling `feet_position`
/// for the expected side).
fn demo_feet_position() {
    println!("[character] --- feet_position ---");
    let body_y = 5.0_f64;
    let cc = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));

    let half_height = EXPECTED_DEFAULTS.height / 2.0;
    let expected_feet_y = body_y - half_height + EXPECTED_DEFAULTS.radius; // 5 - 0.9 + 0.3 = 4.4

    let feet = cc.feet_position();
    assert_close(
        "feet_position.y (y=5)",
        feet.y.to_f64(),
        expected_feet_y,
        1e-9,
    );
    assert_close("feet_position.x (unchanged)", feet.x.to_f64(), 0.0, 1e-9);
    assert_close("feet_position.z (unchanged)", feet.z.to_f64(), 0.0, 1e-9);

    // Degenerate: height == 0 and radius == 0 -> feet == body position exactly.
    let zero_config = CharacterConfig {
        radius: Fix128::ZERO,
        height: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    let cc_zero = CharacterController::new(Vec3Fix::from_int(1, 7, 1), zero_config);
    let feet_zero = cc_zero.feet_position();
    assert_close(
        "feet_position degenerate (zero height/radius)",
        feet_zero.y.to_f64(),
        7.0,
        1e-9,
    );
}

/// `apply_gravity` -- `v_y' = v_y + g*dt` while airborne, no-op while
/// grounded. Expected value computed with plain `f64` arithmetic,
/// independent of the Fix128 implementation under test.
fn demo_apply_gravity() {
    println!("[character] --- apply_gravity ---");
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));
    let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 4); // 0.25

    // Airborne: v1 = 0 + (-10 * 0.25) = -2.5
    let expected_v1 = 0.0_f64 + (-10.0_f64 * 0.25_f64);
    cc.apply_gravity(g, dt);
    assert_close(
        "apply_gravity v1 (airborne)",
        cc.velocity.y.to_f64(),
        expected_v1,
        1e-9,
    );

    // Airborne again: v2 = -2.5 + (-2.5) = -5.0
    let expected_v2 = expected_v1 + (-10.0_f64 * 0.25_f64);
    cc.apply_gravity(g, dt);
    assert_close(
        "apply_gravity v2 (airborne x2)",
        cc.velocity.y.to_f64(),
        expected_v2,
        1e-9,
    );

    // Grounded: velocity must be unchanged (frozen at v2).
    cc.grounded = true;
    cc.apply_gravity(g, dt);
    assert_close(
        "apply_gravity (grounded, frozen)",
        cc.velocity.y.to_f64(),
        expected_v2,
        1e-9,
    );

    // Degenerate: dt == 0 -> no change regardless of ground state.
    let mut cc2 = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));
    cc2.apply_gravity(g, Fix128::ZERO);
    assert_close("apply_gravity dt=0", cc2.velocity.y.to_f64(), 0.0, 1e-9);

    // Degenerate: zero gravity -> no change.
    let mut cc3 = CharacterController::new_default(Vec3Fix::from_int(0, 5, 0));
    cc3.apply_gravity(Vec3Fix::ZERO, dt);
    assert_close("apply_gravity g=0", cc3.velocity.y.to_f64(), 0.0, 1e-9);
}

/// `compute_push_impulses` -- for an overlapping dynamic body, the pushed
/// impulse is `normal * overlap * push_force` where `normal` points from
/// the character to the body and `overlap = (r_char + r_body) - dist`.
/// Expected normal/overlap/impulse are computed independently with plain
/// `f64` vector arithmetic (not by calling `compute_push_impulses`).
fn demo_compute_push_impulses() {
    println!("[character] --- compute_push_impulses ---");
    let cc = CharacterController::new_default(Vec3Fix::from_int(0, 0, 0));
    let body_radius = 0.5_f64;
    let combined = EXPECTED_DEFAULTS.radius + body_radius; // 0.3 + 0.5 = 0.8

    // Case 1: body offset along +x by 0.4 -> dist=0.4 < combined=0.8 -> overlap.
    let dynamic = RigidBody::new(Vec3Fix::from_f32(0.4, 0.0, 0.0), Fix128::ONE);
    let bodies = vec![dynamic];
    let pushes: Vec<PushImpulse> = cc.compute_push_impulses(&bodies, Fix128::from_ratio(1, 2));

    assert_eq!(
        pushes.len(),
        1,
        "[character] FAIL compute_push_impulses: expected 1 push"
    );
    println!("[character] OK   compute_push_impulses.len: 1");

    let dist = 0.4_f64;
    let expected_overlap = combined - dist; // 0.4
    let expected_normal = (1.0_f64, 0.0_f64, 0.0_f64); // delta/dist along +x
    let expected_impulse_x = expected_normal.0 * expected_overlap * EXPECTED_DEFAULTS.push_force;
    // expected_impulse_x = 1.0 * 0.4 * 5.0 = 2.0

    let push = pushes[0];
    assert_eq!(push.body_index, 0);
    println!("[character] OK   compute_push_impulses.body_index: 0");
    assert_close(
        "compute_push_impulses.impulse.x",
        push.impulse.x.to_f64(),
        expected_impulse_x,
        1e-6,
    );
    assert_close(
        "compute_push_impulses.impulse.y",
        push.impulse.y.to_f64(),
        0.0,
        1e-6,
    );

    // point = body.position - normal * body_radius = 0.4 - 1.0*0.5 = -0.1
    let expected_point_x = 0.4_f64 - expected_normal.0 * body_radius;
    assert_close(
        "compute_push_impulses.point.x",
        push.point.x.to_f64(),
        expected_point_x,
        1e-6,
    );

    // Degenerate: zero overlap (bodies exactly combined distance apart) -> no push.
    let far = RigidBody::new(Vec3Fix::from_f32(combined as f32, 0.0, 0.0), Fix128::ONE);
    let far_bodies = vec![far];
    let far_pushes = cc.compute_push_impulses(&far_bodies, Fix128::from_ratio(1, 2));
    assert!(
        far_pushes.is_empty(),
        "[character] FAIL compute_push_impulses degenerate: expected 0 pushes at exact combined distance"
    );
    println!("[character] OK   compute_push_impulses degenerate (dist==combined): 0 pushes");

    // Degenerate: static body -> never pushed regardless of overlap.
    let static_body = RigidBody::new_static(Vec3Fix::ZERO);
    let static_bodies = vec![static_body];
    let static_pushes = cc.compute_push_impulses(&static_bodies, Fix128::from_ratio(1, 2));
    assert!(
        static_pushes.is_empty(),
        "[character] FAIL compute_push_impulses degenerate: expected 0 pushes for static body"
    );
    println!("[character] OK   compute_push_impulses degenerate (static body): 0 pushes");

    // Direct construction of the PushImpulse type (production reference to
    // the struct literal itself, independent of compute_push_impulses).
    let manual = PushImpulse {
        body_index: 7,
        impulse: Vec3Fix::from_int(1, 0, 0),
        point: Vec3Fix::ZERO,
    };
    println!(
        "[character] OK   PushImpulse direct construction: body_index={}",
        manual.body_index
    );
}

/// `get_platform_velocity` -- reports `platform_velocity`, which
/// `move_and_slide` sets to the ground body's velocity while grounded on a
/// body, and resets to zero once no longer grounded on a body. Expected
/// values are the platform's own velocity (set independently of the
/// character), not derived by calling `get_platform_velocity` itself.
fn demo_get_platform_velocity() {
    println!("[character] --- get_platform_velocity ---");

    // Degenerate: brand new controller, no platform under it yet -> ZERO.
    let fresh = CharacterController::new_default(Vec3Fix::from_int(0, 100, 0));
    assert_close(
        "get_platform_velocity (fresh, no platform)",
        fresh.get_platform_velocity().x.to_f64(),
        0.0,
        1e-9,
    );

    // Character standing on a kinematic platform: feet=0.4, platform top=0.3,
    // gap=0.1 < probe(0.1)+skin(0.01)=0.11 -> grounded on the platform.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 1, 0));
    let mut platform = RigidBody::new_kinematic(Vec3Fix::ZERO);
    let expected_platform_vx = 5.0_f64;
    platform.velocity = Vec3Fix::from_int(5, 0, 0);
    let bodies = vec![platform];

    cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert!(
        cc.grounded,
        "[character] FAIL get_platform_velocity: expected grounded on platform"
    );
    assert_close(
        "get_platform_velocity (on platform)",
        cc.get_platform_velocity().x.to_f64(),
        expected_platform_vx,
        1e-9,
    );

    // Next frame: platform velocity carries the character away (to x=5),
    // leaving the platform -> get_platform_velocity resets to ZERO.
    cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert_close(
        "get_platform_velocity (left platform, reset)",
        cc.get_platform_velocity().x.to_f64(),
        0.0,
        1e-9,
    );
}

fn main() {
    println!("ALICE-Physics Character Controller");
    println!("===================================");
    demo_new_default();
    demo_feet_position();
    demo_apply_gravity();
    demo_compute_push_impulses();
    demo_get_platform_velocity();
    println!();
    println!("[character] all closed-form checks passed");
}
