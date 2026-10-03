//! Analytic-solution oracle tests for `src/character.rs`'s wiring_guard
//! targets: `PushImpulse`, `apply_gravity`, `compute_push_impulses`,
//! `feet_position`, `get_platform_velocity`, `new_default`.
//!
//! Every expected value below is derived independently of the function
//! under test (plain `f64` / raw integer arithmetic, or the documented
//! `Fix128` wrapping-add contract applied by hand) — never by calling the
//! function under test for the expected side, per
//! `~/claude-config/rules/analytic-oracle-tests.md`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController, PushImpulse};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

const TOL: f64 = 1e-9;

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// ============================================================================
// new_default
// ============================================================================

#[test]
fn new_default_sets_position_and_rest_defaults() {
    // Oracle: new_default(p) == new(p, CharacterConfig::default()), and the
    // rest-state fields (velocity/grounded/ground_body_index/platform_velocity)
    // are their documented zero/false/None values, independent of the
    // constructor's own code path.
    let p = Vec3Fix::from_int(-4, 11, 7);
    let cc = CharacterController::new_default(p);

    assert!(approx(cc.position.x.to_f64(), -4.0, TOL));
    assert!(approx(cc.position.y.to_f64(), 11.0, TOL));
    assert!(approx(cc.position.z.to_f64(), 7.0, TOL));
    assert!(approx(cc.velocity.x.to_f64(), 0.0, TOL));
    assert!(approx(cc.velocity.y.to_f64(), 0.0, TOL));
    assert!(approx(cc.velocity.z.to_f64(), 0.0, TOL));
    assert!(!cc.grounded);
    assert_eq!(cc.ground_body_index, None);
    assert!(approx(cc.platform_velocity.x.to_f64(), 0.0, TOL));
    assert!(approx(cc.platform_velocity.y.to_f64(), 0.0, TOL));
    assert!(approx(cc.platform_velocity.z.to_f64(), 0.0, TOL));
}

#[test]
fn new_default_config_matches_known_default_constants() {
    // Oracle: the literal constants from CharacterConfig::default()'s source
    // (radius=0.3, height=1.8, max_slope_angle~0.785, step_height=0.3,
    // skin_width=0.01, ground_probe_distance=0.1, max_slides=4, push_force=5),
    // transcribed independently of calling Default::default() or new_default().
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    assert!(approx(cc.config.radius.to_f64(), 0.3, TOL));
    assert!(approx(cc.config.height.to_f64(), 1.8, TOL));
    assert!(approx(cc.config.max_slope_angle.to_f64(), 0.785, TOL));
    assert!(approx(cc.config.step_height.to_f64(), 0.3, TOL));
    assert!(approx(cc.config.skin_width.to_f64(), 0.01, TOL));
    assert!(approx(cc.config.ground_probe_distance.to_f64(), 0.1, TOL));
    assert_eq!(cc.config.max_slides, 4);
    assert!(approx(cc.config.push_force.to_f64(), 5.0, TOL));
}

#[test]
fn new_default_degenerate_zero_position() {
    // Degenerate input: position at the origin.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    assert!(approx(cc.position.x.to_f64(), 0.0, TOL));
    assert!(approx(cc.position.y.to_f64(), 0.0, TOL));
    assert!(approx(cc.position.z.to_f64(), 0.0, TOL));
}

// ============================================================================
// feet_position
// ============================================================================

#[test]
fn feet_position_subtracts_half_height_adds_radius() {
    // Oracle: feet.y = body.y - height/2 + radius, body.x/z unchanged.
    // Default config: height=1.8 (half=0.9), radius=0.3.
    for &y in &[0.0_f64, 5.0, -5.0, 100.0] {
        let cc = CharacterController::new_default(Vec3Fix::from_f32(1.0, y as f32, -2.0));
        let expected_y = y - 0.9 + 0.3;
        let feet = cc.feet_position();
        assert!(
            approx(feet.y.to_f64(), expected_y, 1e-5),
            "y={y} actual={} expected={expected_y}",
            feet.y.to_f64()
        );
        assert!(approx(feet.x.to_f64(), 1.0, 1e-5));
        assert!(approx(feet.z.to_f64(), -2.0, 1e-5));
    }
}

#[test]
fn feet_position_degenerate_zero_height_and_radius() {
    // Degenerate: height == 0, radius == 0 -> feet == body position exactly.
    let config = CharacterConfig {
        radius: Fix128::ZERO,
        height: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    let cc = CharacterController::new(Vec3Fix::from_int(3, 9, -1), config);
    let feet = cc.feet_position();
    assert!(approx(feet.y.to_f64(), 9.0, TOL));
}

#[test]
fn feet_position_degenerate_extreme_magnitude() {
    // Degenerate: extreme Fix128 magnitude for the body position's y, well
    // within i64 range, verified via plain f64 arithmetic on the same
    // closed form (no call to feet_position for the expected side).
    let body_y_hi: i64 = 1_000_000_000;
    let cc = CharacterController::new_default(Vec3Fix::new(
        Fix128::ZERO,
        Fix128 {
            hi: body_y_hi,
            lo: 0,
        },
        Fix128::ZERO,
    ));
    let expected_y = body_y_hi as f64 - 0.9 + 0.3;
    let feet = cc.feet_position();
    assert!(approx(feet.y.to_f64(), expected_y, 1.0));
}

// ============================================================================
// apply_gravity
// ============================================================================

#[test]
fn apply_gravity_integrates_while_airborne() {
    // Oracle: v' = v + g*dt (plain f64), only while !grounded.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 10, 0));
    let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 3); // 1/3 s

    let mut v = 0.0_f64;
    for _ in 0..5 {
        v += -10.0 * (1.0 / 3.0);
        cc.apply_gravity(g, dt);
        assert!(
            approx(cc.velocity.y.to_f64(), v, 1e-6),
            "actual={} expected={v}",
            cc.velocity.y.to_f64()
        );
    }
}

#[test]
fn apply_gravity_frozen_while_grounded() {
    // Oracle: once grounded, velocity.y must not change regardless of g*dt.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 10, 0));
    let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 4);

    cc.apply_gravity(g, dt); // airborne: v = -2.5
    let frozen_at = cc.velocity.y.to_f64();
    assert!(approx(frozen_at, -2.5, 1e-9));

    cc.grounded = true;
    for _ in 0..10 {
        cc.apply_gravity(g, dt);
        assert!(approx(cc.velocity.y.to_f64(), frozen_at, 1e-9));
    }
}

#[test]
fn apply_gravity_degenerate_zero_dt() {
    // Degenerate: dt == 0 -> no velocity change regardless of gravity magnitude.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 10, 0));
    let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-999), Fix128::ZERO);
    cc.apply_gravity(g, Fix128::ZERO);
    assert!(approx(cc.velocity.y.to_f64(), 0.0, TOL));
}

#[test]
fn apply_gravity_degenerate_zero_gravity() {
    // Degenerate: zero gravity vector -> no velocity change regardless of dt.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 10, 0));
    cc.apply_gravity(Vec3Fix::ZERO, Fix128::from_int(1000));
    assert!(approx(cc.velocity.y.to_f64(), 0.0, TOL));
}

#[test]
fn apply_gravity_preserves_horizontal_velocity() {
    // Oracle: apply_gravity only touches velocity.y; x/z must be the input
    // velocity unchanged (checked against the literal value we set).
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 10, 0));
    cc.velocity = Vec3Fix::from_int(7, 0, -3);
    let g = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    cc.apply_gravity(g, Fix128::from_ratio(1, 2));
    assert!(approx(cc.velocity.x.to_f64(), 7.0, TOL));
    assert!(approx(cc.velocity.z.to_f64(), -3.0, TOL));
    assert!(approx(cc.velocity.y.to_f64(), -5.0, TOL));
}

#[test]
fn apply_gravity_degenerate_extreme_fix128_magnitude_wraps_per_documented_add_contract() {
    // Degenerate: extreme Fix128 magnitude (near i64::MAX) that overflows the
    // 128-bit representation on addition. Fix128::Add is documented
    // (src/math.rs) as a wrapping add: lo via `overflowing_add` producing a
    // carry, hi via `wrapping_add` of both his and the carry. We replicate
    // that documented rule here with raw i64/u64 arithmetic -- independent
    // of calling Fix128::add/apply_gravity for the expected side -- then
    // check apply_gravity (v + g*1) against it.
    let v_hi = i64::MAX;
    let v_lo: u64 = 0xFFFF_FFFF_FFFF_FFFF;
    let g_hi = i64::MAX;
    let g_lo: u64 = 1;

    let (expected_lo, carry) = v_lo.overflowing_add(g_lo);
    let expected_hi = v_hi.wrapping_add(g_hi).wrapping_add(carry as i64);

    let mut cc = CharacterController::new_default(Vec3Fix::ZERO);
    cc.velocity = Vec3Fix::new(Fix128::ZERO, Fix128 { hi: v_hi, lo: v_lo }, Fix128::ZERO);
    let g = Vec3Fix::new(Fix128::ZERO, Fix128 { hi: g_hi, lo: g_lo }, Fix128::ZERO);

    cc.apply_gravity(g, Fix128::ONE); // dt=1 -> g*dt == g exactly (Mul identity)

    assert_eq!(
        cc.velocity.y.hi, expected_hi,
        "hi mismatch (wraparound expected)"
    );
    assert_eq!(cc.velocity.y.lo, expected_lo, "lo mismatch");
    // Sanity: this really did wrap (two positive near-max values summing to
    // something that is not itself near-max-positive).
    assert!(
        expected_hi < 0,
        "test setup should have produced a wrapped (negative) hi"
    );
}

// ============================================================================
// compute_push_impulses
// ============================================================================

#[test]
fn compute_push_impulses_momentum_conserving_single_overlap() {
    // Oracle (hand-derived momentum-conserving impulse for a known overlap):
    // normal = delta/|delta|, overlap = (r_char + r_body) - |delta|,
    // impulse = normal * overlap * push_force (default push_force = 5.0).
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let body_radius = 0.5_f64;
    let combined = 0.3_f64 + body_radius; // char radius default 0.3

    let dx = 0.4_f64;
    let body = RigidBody::new(Vec3Fix::from_f32(dx as f32, 0.0, 0.0), Fix128::ONE);
    let pushes = cc.compute_push_impulses(&[body], Fix128::from_ratio(1, 2));

    assert_eq!(pushes.len(), 1);
    let overlap = combined - dx; // 0.4
    let expected_impulse_x = 1.0_f64 * overlap * 5.0; // normal.x=1
    let expected_point_x = dx - 1.0 * body_radius; // -0.1

    let p: PushImpulse = pushes[0];
    assert_eq!(p.body_index, 0);
    assert!(approx(p.impulse.x.to_f64(), expected_impulse_x, 1e-5));
    assert!(approx(p.impulse.y.to_f64(), 0.0, 1e-5));
    assert!(approx(p.point.x.to_f64(), expected_point_x, 1e-5));
}

#[test]
fn compute_push_impulses_degenerate_zero_overlap_at_exact_combined_distance() {
    // Degenerate: distance exactly equal to combined radius -> dist_sq is
    // NOT < combined_sq, so no push (strict inequality in the source).
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let combined = 0.8_f32; // 0.3 + 0.5
    let body = RigidBody::new(Vec3Fix::from_f32(combined, 0.0, 0.0), Fix128::ONE);
    let pushes = cc.compute_push_impulses(&[body], Fix128::from_ratio(1, 2));
    assert!(pushes.is_empty());
}

#[test]
fn compute_push_impulses_degenerate_coincident_positions_push_up() {
    // Degenerate: body at the exact same position as the character
    // (dist_sq == 0) -> documented fallback: push upward (+Y) by `combined`.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let body_radius = 0.5_f64;
    let combined = 0.3_f64 + body_radius;
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let pushes = cc.compute_push_impulses(&[body], Fix128::from_ratio(1, 2));

    assert_eq!(pushes.len(), 1);
    let expected_impulse_y = 1.0_f64 * combined * 5.0; // normal=+Y, overlap=combined
    assert!(approx(
        pushes[0].impulse.y.to_f64(),
        expected_impulse_y,
        1e-6
    ));
    assert!(approx(pushes[0].impulse.x.to_f64(), 0.0, 1e-9));
}

#[test]
fn compute_push_impulses_ignores_static_bodies() {
    // Oracle: static bodies are never pushed, regardless of overlap.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let body = RigidBody::new_static(Vec3Fix::ZERO);
    let pushes = cc.compute_push_impulses(&[body], Fix128::from_ratio(1, 2));
    assert!(pushes.is_empty());
}

#[test]
fn compute_push_impulses_ignores_sensor_bodies() {
    // Oracle: sensor bodies (is_sensor=true) are never pushed, even when
    // dynamic and fully overlapping.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.is_sensor = true;
    let pushes = cc.compute_push_impulses(&[body], Fix128::from_ratio(1, 2));
    assert!(pushes.is_empty());
}

#[test]
fn compute_push_impulses_degenerate_empty_bodies() {
    // Degenerate: empty body slice -> empty push list.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let pushes = cc.compute_push_impulses(&[], Fix128::from_ratio(1, 2));
    assert!(pushes.is_empty());
}

#[test]
fn compute_push_impulses_degenerate_zero_body_radius() {
    // Degenerate: body_radius == 0 -> combined == character radius only.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let combined = 0.3_f64; // body_radius = 0
    let dx = 0.1_f64;
    let body = RigidBody::new(Vec3Fix::from_f32(dx as f32, 0.0, 0.0), Fix128::ONE);
    let pushes = cc.compute_push_impulses(&[body], Fix128::ZERO);
    assert_eq!(pushes.len(), 1);
    let overlap = combined - dx;
    let expected_impulse_x = overlap * 5.0;
    assert!(approx(
        pushes[0].impulse.x.to_f64(),
        expected_impulse_x,
        1e-5
    ));
    // point = body.position - normal * body_radius(0) == body.position
    assert!(approx(pushes[0].point.x.to_f64(), dx, 1e-5));
}

#[test]
fn compute_push_impulses_multiple_bodies_mixed_overlap() {
    // Oracle: among 3 dynamic bodies, only the overlapping one is pushed;
    // a non-overlapping one and a static one are excluded.
    let cc = CharacterController::new_default(Vec3Fix::ZERO);
    let overlapping = RigidBody::new(Vec3Fix::from_f32(0.2, 0.0, 0.0), Fix128::ONE); // dist=0.2 < 0.8
    let far = RigidBody::new(Vec3Fix::from_f32(10.0, 0.0, 0.0), Fix128::ONE); // dist=10 >= 0.8
    let statik = RigidBody::new_static(Vec3Fix::from_f32(0.0, 0.1, 0.0));
    let pushes = cc.compute_push_impulses(&[overlapping, far, statik], Fix128::from_ratio(1, 2));
    assert_eq!(pushes.len(), 1);
    assert_eq!(pushes[0].body_index, 0);
}

#[test]
fn push_impulse_direct_construction_round_trips_fields() {
    // PushImpulse (type) -- construct directly and verify field round-trip,
    // independent of compute_push_impulses.
    let imp = PushImpulse {
        body_index: 42,
        impulse: Vec3Fix::from_int(1, 2, 3),
        point: Vec3Fix::from_int(-1, -2, -3),
    };
    assert_eq!(imp.body_index, 42);
    assert!(approx(imp.impulse.x.to_f64(), 1.0, TOL));
    assert!(approx(imp.point.z.to_f64(), -3.0, TOL));
}

// ============================================================================
// get_platform_velocity
// ============================================================================

#[test]
fn get_platform_velocity_zero_when_fresh() {
    // Degenerate: brand-new controller, no platform under it yet -> ZERO.
    let cc = CharacterController::new_default(Vec3Fix::from_int(0, 50, 0));
    assert!(approx(cc.get_platform_velocity().x.to_f64(), 0.0, TOL));
    assert!(approx(cc.get_platform_velocity().y.to_f64(), 0.0, TOL));
    assert!(approx(cc.get_platform_velocity().z.to_f64(), 0.0, TOL));
}

#[test]
fn get_platform_velocity_reports_ground_body_velocity_then_clears() {
    // Oracle: move_and_slide sets platform_velocity to the ground body's own
    // velocity (set independently of the character) while grounded on it,
    // then clears to ZERO the frame the character leaves that body.
    // feet=0.4 (y=1 - 0.9 + 0.3), platform top=0.3 (sphere radius=0.3),
    // gap=0.1 < probe(0.1)+skin(0.01)=0.11 -> grounded.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 1, 0));
    let mut platform = RigidBody::new_kinematic(Vec3Fix::ZERO);
    let expected_vx = 6.0_f64;
    platform.velocity = Vec3Fix::from_int(6, 0, 0);
    let bodies = vec![platform];

    cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert!(cc.grounded);
    assert!(approx(
        cc.get_platform_velocity().x.to_f64(),
        expected_vx,
        TOL
    ));

    // Next frame: the platform velocity carries the character to x=6,
    // leaving the platform underneath it -> resets to ZERO.
    cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert!(!cc.grounded);
    assert!(approx(cc.get_platform_velocity().x.to_f64(), 0.0, TOL));
}

#[test]
fn get_platform_velocity_degenerate_no_platform_under_character() {
    // Degenerate: no bodies and no SDF colliders at all (free fall) ->
    // never grounded, platform velocity stays ZERO regardless of movement.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 1000, 0));
    cc.move_and_slide(Vec3Fix::from_int(0, -5, 0), &[], &[]);
    assert!(!cc.grounded);
    assert!(approx(cc.get_platform_velocity().x.to_f64(), 0.0, TOL));
    assert!(approx(cc.get_platform_velocity().y.to_f64(), 0.0, TOL));
    assert!(approx(cc.get_platform_velocity().z.to_f64(), 0.0, TOL));
}

#[test]
fn get_platform_velocity_degenerate_dynamic_body_underneath_never_grounds() {
    // Degenerate: a body is directly underneath but it is dynamic, not
    // static -- detect_ground only considers `body.is_static()`, so the
    // character must not be considered grounded on it, and platform
    // velocity must stay ZERO even though the dynamic body has nonzero
    // velocity.
    let mut cc = CharacterController::new_default(Vec3Fix::from_int(0, 1, 0));
    let mut dynamic = RigidBody::new_dynamic(Vec3Fix::ZERO, Fix128::ONE);
    dynamic.velocity = Vec3Fix::from_int(9, 0, 0);
    let bodies = vec![dynamic];

    cc.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert!(!cc.grounded);
    assert_eq!(cc.ground_body_index, None);
    assert!(approx(cc.get_platform_velocity().x.to_f64(), 0.0, TOL));
}
