//! Audit S3W3 oracles for `src/character.rs`.
//!
//! The existing `analytic_character_wiring.rs` covers the six helpers
//! (`new_default`, `feet_position`, `apply_gravity`, `compute_push_impulses`,
//! `get_platform_velocity`, `PushImpulse`) but none of the main
//! `move_and_slide` behaviour (sweep, slide, stair step, slope limit, SDF
//! push-out). Expected values below come from closed forms (sphere sweep of
//! a point against a sphere inflated by the character radius, the capsule
//! geometry `bottom = centre - height/2`, plane SDF `n . p`).

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn static_at(x: f64, y: f64, z: f64) -> RigidBody {
    RigidBody::new(v3(x, y, z), Fix128::ZERO)
}

fn plane_sdf(nx: f32, ny: f32, offset: f32) -> SdfCollider {
    // signed distance of the half space  n . p = offset  (n unit)
    let f = ClosureSdf::new(
        move |x, y, z| nx * x + ny * y - offset + 0.0 * z,
        move |_x, _y, _z| (nx, ny, 0.0),
    );
    SdfCollider::new_static(Box::new(f), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn cfg() -> CharacterConfig {
    CharacterConfig::default() // r 0.3, h 1.8, skin 0.01, probe 0.1, step 0.3, max slope 0.785 rad
}

// ---------------------------------------------------------------------------
// move_and_slide: free movement and the sweep against static bodies
// ---------------------------------------------------------------------------

#[test]
fn free_move_translates_exactly_by_the_displacement() {
    let mut c = CharacterController::new(v3(1.0, 5.0, -2.0), cfg());
    let r = c.move_and_slide(v3(0.5, -0.25, 1.0), &[], &[]);
    assert!((r.position.x.to_f64() - 1.5).abs() < 1e-9);
    assert!((r.position.y.to_f64() - 4.75).abs() < 1e-9);
    assert!((r.position.z.to_f64() - -1.0).abs() < 1e-9);
    assert_eq!(r.position, c.position);
    assert!(!r.grounded);
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-004: move_and_slide breaks out when |displacement| < skin_width before moving, so a 5 mm free move advances 0 m (walking < 0.6 m/s at 60 Hz never moves)"]
fn displacement_shorter_than_skin_width_is_not_discarded() {
    // "Try to move the full displacement": 5 mm is below skin_width (1 cm) but
    // nothing is in the way, so the character must advance 5 mm.
    let mut c = CharacterController::new(v3(0.0, 5.0, 0.0), cfg());
    let r = c.move_and_slide(v3(0.005, 0.0, 0.0), &[], &[]);
    assert!(
        (r.position.x.to_f64() - 0.005).abs() < 1e-9,
        "moved {} m instead of 0.005 m",
        r.position.x.to_f64()
    );
}

#[test]
fn head_on_static_body_stops_one_skin_before_the_inflated_sphere() {
    // Static body at (3,0,0). The character is swept as a point against a
    // sphere of radius r_body + r_char = 0.6, so the surface is at x = 2.4
    // and the stop is `skin` short of it: x = 2.39. y/z untouched.
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let bodies = [static_at(3.0, 0.0, 0.0)];
    let r = c.move_and_slide(v3(5.0, 0.0, 0.0), &bodies, &[]);
    assert!(
        (r.position.x.to_f64() - 2.39).abs() < 1e-6,
        "x = {}",
        r.position.x.to_f64()
    );
    assert!(r.position.y.to_f64().abs() < 1e-9);
    assert!(r.position.z.to_f64().abs() < 1e-9);
}

#[test]
fn dynamic_bodies_do_not_block_the_sweep() {
    // doc: "Skip dynamic bodies for now (character interacts with statics)"
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let bodies = [RigidBody::new(v3(3.0, 0.0, 0.0), Fix128::ONE)];
    let r = c.move_and_slide(v3(5.0, 0.0, 0.0), &bodies, &[]);
    assert!((r.position.x.to_f64() - 5.0).abs() < 1e-9);
}

#[test]
fn nearer_of_two_static_bodies_wins() {
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let bodies = [static_at(6.0, 0.0, 0.0), static_at(3.0, 0.0, 0.0)];
    let r = c.move_and_slide(v3(8.0, 0.0, 0.0), &bodies, &[]);
    assert!((r.position.x.to_f64() - 2.39).abs() < 1e-6);
}

#[test]
fn oblique_hit_never_ends_inside_the_inflated_sphere() {
    // body at (3, 0.3, 0): glancing contact; whatever the slide does, the
    // final point must stay on or outside the 0.6 inflated sphere (minus a
    // skin of numerical slack).
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let bodies = [static_at(3.0, 0.3, 0.0)];
    let r = c.move_and_slide(v3(5.0, 0.0, 0.0), &bodies, &[]);
    let dx = r.position.x.to_f64() - 3.0;
    let dy = r.position.y.to_f64() - 0.3;
    let dz = r.position.z.to_f64();
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(
        dist >= 0.6 - 0.011,
        "ended inside the obstacle: dist {dist}"
    );
    // and it made progress along +x (slid around, not bounced back)
    assert!(r.position.x.to_f64() > 2.4);
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-005: MoveResult.velocity doc says velocity after sliding but move_and_slide always returns the input displacement (head-on into a wall: 5, want 0)"]
fn result_velocity_has_no_component_into_the_wall_after_a_head_on_slide() {
    // MoveResult::velocity is documented as "Velocity after sliding (may
    // differ from input if we slid along a wall)". Head-on into a sphere the
    // slid velocity along the wall plane is zero.
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let bodies = [static_at(3.0, 0.0, 0.0)];
    let r = c.move_and_slide(v3(5.0, 0.0, 0.0), &bodies, &[]);
    assert!(
        r.velocity.x.to_f64().abs() < 1e-6,
        "velocity into the wall is still {} (input returned verbatim)",
        r.velocity.x.to_f64()
    );
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-007: starting inside the inflated body sphere the ray_sphere far hit (exit) is treated as an obstacle with outward normal, the slide cancels the remainder and the character sticks at x = 0.59 forever instead of walking away"]
fn character_overlapping_a_static_body_can_walk_away() {
    // Start 0.5 m from a static body (inside the 0.6 inflated sphere) and
    // walk directly away at 0.1 m per frame for 30 frames: the character
    // must end clear of the body (>= 0.6) instead of sticking at the exit
    // boundary.
    let mut c = CharacterController::new(v3(0.5, 0.0, 0.0), cfg());
    let bodies = [static_at(0.0, 0.0, 0.0)];
    for _ in 0..30 {
        c.move_and_slide(v3(0.1, 0.0, 0.0), &bodies, &[]);
    }
    assert!(
        c.position.x.to_f64() > 3.0,
        "stuck at x = {} after 30 frames of walking away",
        c.position.x.to_f64()
    );
}

// ---------------------------------------------------------------------------
// Ground detection
// ---------------------------------------------------------------------------

#[test]
fn flat_sdf_floor_grounds_within_probe_of_the_capsule_bottom() {
    // capsule bottom = centre - h/2 = y - 0.9. Gap to y = 0 floor:
    // 0.05 < probe + skin (0.11) -> grounded; 0.20 -> airborne.
    let floor = [plane_sdf(0.0, 1.0, 0.0)];
    let mut near = CharacterController::new(v3(0.0, 0.95, 0.0), cfg());
    assert!(near.move_and_slide(Vec3Fix::ZERO, &[], &floor).grounded);
    let mut far = CharacterController::new(v3(0.0, 1.10, 0.0), cfg());
    assert!(!far.move_and_slide(Vec3Fix::ZERO, &[], &floor).grounded);
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-006: detect_ground body branch rays from the hemisphere centre against a sphere of radius r, so a capsule bottom 5 cm above a static body is not grounded while the SDF branch (dist < probe + r) grounds a 5 cm gap; inconsistent by one radius (0.3 m)"]
fn static_body_grounds_within_probe_of_the_capsule_bottom_like_an_sdf_floor() {
    // Body sphere radius = character radius = 0.3 at the origin: top at y = 0.3.
    // Capsule bottom 0.05 above it: centre y = 0.3 + 0.05 + 0.9 = 1.25.
    // The SDF floor test above counts a 0.05 gap as grounded, so the body
    // branch must as well.
    let bodies = [static_at(0.0, 0.0, 0.0)];
    let mut c = CharacterController::new(v3(0.0, 1.25, 0.0), cfg());
    let r = c.move_and_slide(Vec3Fix::ZERO, &bodies, &[]);
    assert!(
        r.grounded,
        "capsule bottom 5 cm above a static body is not grounded"
    );
    assert_eq!(c.ground_body_index, Some(0));
}

#[test]
fn slope_limit_decides_grounded_on_a_tilted_sdf_plane() {
    // plane normal n = (sin t, cos t, 0): walkable iff n_y >= cos(0.785).
    let place = |theta: f64| {
        let (s, co) = (theta.sin() as f32, theta.cos() as f32);
        let floor = [plane_sdf(s, co, 0.0)];
        // feet (hemisphere centre) at signed distance 0.35 from the plane,
        // x = 0 so  n.p = y_feet * cos t = 0.35
        let y_feet = 0.35 / theta.cos();
        let mut c = CharacterController::new(v3(0.0, y_feet + 0.6, 0.0), cfg());
        c.move_and_slide(Vec3Fix::ZERO, &[], &floor).grounded
    };
    assert!(
        place(30.0_f64.to_radians()),
        "30 deg slope must be walkable"
    );
    assert!(
        place(40.0_f64.to_radians()),
        "40 deg slope must be walkable"
    );
    assert!(
        !place(50.0_f64.to_radians()),
        "50 deg slope exceeds 45 deg limit"
    );
    assert!(
        !place(60.0_f64.to_radians()),
        "60 deg slope exceeds 45 deg limit"
    );
}

#[test]
fn sdf_push_out_lifts_a_sunken_capsule_to_radius_clearance() {
    // Floor y=0. Capsule centre at 0.5 -> bottom hemisphere centre at -0.1,
    // inside the floor. resolve_sdf pushes along the normal until
    // dist(sample) >= radius at the bottom sample point: feet y -> 0.3,
    // i.e. centre y = 0.9. Samples at centre/top are already clear.
    let floor = [plane_sdf(0.0, 1.0, 0.0)];
    let mut c = CharacterController::new(v3(0.0, 0.5, 0.0), cfg());
    c.move_and_slide(Vec3Fix::ZERO, &[], &floor);
    assert!(
        (c.position.y.to_f64() - 0.9).abs() < 1e-3,
        "y = {}",
        c.position.y.to_f64()
    );
}

#[test]
fn platform_velocity_is_added_to_the_next_displacement() {
    let mut c = CharacterController::new(v3(0.0, 5.0, 0.0), cfg());
    c.platform_velocity = v3(2.0, 0.0, 0.0);
    let r = c.move_and_slide(v3(1.0, 0.0, 0.0), &[], &[]);
    assert!((r.position.x.to_f64() - 3.0).abs() < 1e-9);
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-003: feet_position doc says capsule bottom but returns the lower hemisphere centre (centre y - h/2 + r): centre y=1, h=1.8 -> 0.4, capsule bottom is 0.1"]
fn feet_position_is_the_capsule_bottom() {
    // doc: "Get the bottom of the capsule (feet position)". Capsule centre
    // (0,1,0), height 1.8: bottom is y = 0.1.
    let c = CharacterController::new(v3(0.0, 1.0, 0.0), cfg());
    assert!(
        (c.feet_position().y.to_f64() - 0.1).abs() < 1e-9,
        "feet_position.y = {} (capsule bottom is 0.1)",
        c.feet_position().y.to_f64()
    );
}

// ---------------------------------------------------------------------------
// Stair step
// ---------------------------------------------------------------------------

/// Axis-aligned box SDF (exact for outside points, interior approximated by
/// the max-axis distance) used as a stair block.
fn block(min: [f32; 3], max: [f32; 3]) -> SdfCollider {
    let c = [
        (min[0] + max[0]) / 2.0,
        (min[1] + max[1]) / 2.0,
        (min[2] + max[2]) / 2.0,
    ];
    let h = [
        (max[0] - min[0]) / 2.0,
        (max[1] - min[1]) / 2.0,
        (max[2] - min[2]) / 2.0,
    ];
    let dist = move |x: f32, y: f32, z: f32| {
        let q = [
            (x - c[0]).abs() - h[0],
            (y - c[1]).abs() - h[1],
            (z - c[2]).abs() - h[2],
        ];
        let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
        (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
    };
    let d2 = dist;
    let normal = move |x: f32, y: f32, z: f32| {
        let e = 1e-3;
        let gx = d2(x + e, y, z) - d2(x - e, y, z);
        let gy = d2(x, y + e, z) - d2(x, y - e, z);
        let gz = d2(x, y, z + e) - d2(x, y, z - e);
        let l = (gx * gx + gy * gy + gz * gz).sqrt().max(1e-9);
        (gx / l, gy / l, gz / l)
    };
    SdfCollider::new_static(
        Box::new(ClosureSdf::new(dist, normal)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    )
}

fn walk_into_block(height: f32) -> f64 {
    // floor + a block of the given height starting at x = 1; the character is
    // grounded and walks +x at 0.2 m per frame for 20 frames.
    let sdfs = [
        plane_sdf(0.0, 1.0, 0.0),
        block([1.0, 0.0, -5.0], [3.0, height, 5.0]),
    ];
    let mut c = CharacterController::new(v3(0.0, 0.92, 0.0), cfg());
    c.move_and_slide(Vec3Fix::ZERO, &[], &sdfs); // settle: grounded = true
    for _ in 0..20 {
        c.move_and_slide(v3(0.2, 0.0, 0.0), &[], &sdfs);
    }
    c.position.x.to_f64()
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-008: stair step check samples only the capsule centre at test_pos so step_height = 0.3 does not limit climbing: 0.35 / 0.5 / 0.7 m blocks are all climbed (blocked only from ~0.8 m)"]
fn step_height_limit_blocks_obstacles_taller_than_step_height() {
    // step_height = 0.3. A 0.2 m ledge is climbable; a 0.5 m wall is not.
    let climbed_low = walk_into_block(0.2);
    let climbed_tall = walk_into_block(0.5);
    assert!(
        climbed_low > 2.0,
        "0.2 m ledge not climbed: x = {climbed_low}"
    );
    assert!(
        climbed_tall < 1.0,
        "0.5 m wall (step_height 0.3) was climbed: x = {climbed_tall}"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-009: sweep_against_bodies casts a point-sphere from the capsule centre, so landing on a static body stops with the centre 0.61 above it and the capsule bottom 0.3 m inside the body (bottom y = -0.29, top at 0.3)"]
fn capsule_bottom_does_not_sink_into_a_static_body_it_lands_on() {
    // Falling straight onto a static body at the origin (sphere proxy radius
    // 0.3, top at y = 0.3). The capsule bottom is centre - 0.9, so a correct
    // landing keeps centre y >= 0.3 + 0.9 - skin.
    let mut c = CharacterController::new(v3(0.0, 3.0, 0.0), cfg());
    let bodies = [static_at(0.0, 0.0, 0.0)];
    c.move_and_slide(v3(0.0, -5.0, 0.0), &bodies, &[]);
    let bottom = c.position.y.to_f64() - 0.9;
    assert!(
        bottom >= 0.3 - 0.011,
        "capsule bottom at y = {bottom} is inside the body (top at 0.3)"
    );
}

// ---------------------------------------------------------------------------
// Closed-form slide, sweep ordering, stair step, SDF hysteresis, slope config
// ---------------------------------------------------------------------------

#[test]
fn slide_along_an_oblique_contact_matches_the_collide_and_slide_closed_form() {
    // Static body at (3, 0, 0.3), inflated radius R = 0.6, character moves
    // (5,0,0) from the origin. Hand derivation:
    //   hit      x_h = 3 - sqrt(R^2 - 0.3^2) = 2.480384, n = ((x_h-3), 0, -0.3)/R
    //   stop     x_s = x_h - skin
    //   left     = 5 - x_s,  rem = (left, 0, 0)
    //   slide    = rem - n (rem . n)
    //   2nd sweep from (x_s,0,0) along slide misses the sphere (discriminant < 0)
    //   so the final position is (x_s,0,0) + slide.
    let (r, d, skin) = (0.6_f64, 0.3_f64, 0.01_f64);
    let x_h = 3.0 - (r * r - d * d).sqrt();
    let n = [(x_h - 3.0) / r, 0.0, -d / r];
    let x_s = x_h - skin;
    let left = 5.0 - x_s;
    let rn = left * n[0];
    let slide = [left - n[0] * rn, -n[1] * rn, -n[2] * rn];
    let want = [x_s + slide[0], slide[1], slide[2]];
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let res = c.move_and_slide(v3(5.0, 0.0, 0.0), &[static_at(3.0, 0.0, d)], &[]);
    let got = [
        res.position.x.to_f64(),
        res.position.y.to_f64(),
        res.position.z.to_f64(),
    ];
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "axis {i}: got {got:?} want {want:?}"
        );
    }
}

#[test]
fn max_slides_one_stops_after_the_first_contact() {
    let mut conf = cfg();
    conf.max_slides = 1;
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), conf);
    let r = c.move_and_slide(v3(5.0, 0.0, 0.0), &[static_at(3.0, 0.0, 0.3)], &[]);
    let x_h = 3.0 - (0.36_f64 - 0.09).sqrt();
    assert!((r.position.x.to_f64() - (x_h - 0.01)).abs() < 1e-4);
    assert!(
        r.position.z.to_f64().abs() < 1e-9,
        "no slide with a one-slide budget"
    );
}

#[test]
fn nearer_body_wins_when_listed_first_and_far_bodies_beyond_the_displacement_do_not_block() {
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let near_first = [static_at(3.0, 0.0, 0.0), static_at(6.0, 0.0, 0.0)];
    let r = c.move_and_slide(v3(8.0, 0.0, 0.0), &near_first, &[]);
    assert!((r.position.x.to_f64() - 2.39).abs() < 1e-6);
    // displacement 2.0 ends before the inflated surface at x = 2.4
    let mut c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let r = c.move_and_slide(v3(2.0, 0.0, 0.0), &[static_at(3.0, 0.0, 0.0)], &[]);
    assert!(
        (r.position.x.to_f64() - 2.0).abs() < 1e-9,
        "x = {}",
        r.position.x.to_f64()
    );
}

fn stepper(x0: f64) -> CharacterController {
    let mut c = CharacterController::new(v3(x0, 0.0, 0.0), cfg());
    c.grounded = true;
    c
}

#[test]
fn stair_step_lifts_by_step_height_and_advances_the_full_displacement_when_clear() {
    // Body at (1,0,0). Start 0.38, move 0.9: the sweep stops at 0.39 (blocked,
    // moved < 25 %), the stepped test point (1.29, 0.3, 0) is 0.417 from the
    // body centre (> radius + skin = 0.31) so the step is taken.
    let mut c = stepper(0.38);
    let r = c.move_and_slide(v3(0.9, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!(
        (r.position.x.to_f64() - 1.29).abs() < 1e-6,
        "x = {}",
        r.position.x.to_f64()
    );
    assert!(
        (r.position.y.to_f64() - 0.3).abs() < 1e-6,
        "y = {}",
        r.position.y.to_f64()
    );
}

#[test]
fn stair_step_needs_a_grounded_character() {
    let mut c = stepper(0.38);
    c.grounded = false;
    let r = c.move_and_slide(v3(0.9, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!((r.position.x.to_f64() - 0.39).abs() < 1e-6);
    assert!(r.position.y.to_f64().abs() < 1e-9);
}

#[test]
fn stair_step_is_refused_when_the_stepped_point_is_inside_the_body_clearance() {
    // displacement 0.6: stepped point (0.99, 0.3, 0) is 0.3002 from the body
    // centre, inside radius + skin = 0.31 -> blocked, no step.
    let mut c = stepper(0.38);
    let r = c.move_and_slide(v3(0.6, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!((r.position.x.to_f64() - 0.39).abs() < 1e-6);
    assert!(r.position.y.to_f64().abs() < 1e-9);
    // 0.555: stepped point is 0.305 from the centre: still inside r + skin (0.31)
    let mut c = stepper(0.38);
    let r = c.move_and_slide(v3(0.555, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!(
        r.position.y.to_f64().abs() < 1e-9,
        "step taken inside the skin clearance"
    );
}

#[test]
fn stair_step_is_not_attempted_when_the_character_moved_more_than_a_quarter() {
    // start 0, move 0.7: stops at 0.39, moved^2 / desired^2 = 0.31 > 1/4
    let mut c = stepper(0.0);
    let r = c.move_and_slide(v3(0.7, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!((r.position.x.to_f64() - 0.39).abs() < 1e-6);
    assert!(
        r.position.y.to_f64().abs() < 1e-9,
        "step attempted at 31 % progress"
    );
}

#[test]
fn stair_step_is_refused_when_an_sdf_wall_occupies_the_stepped_point() {
    // tall wall x in [1,3]; start 0.65 grounded, move +0.5: pushed back to the
    // 0.3 clearance (x = 0.7); stepped point (1.2, 0.3, 0) is inside the wall.
    let wall = block([1.0, -50.0, -50.0], [3.0, 50.0, 50.0]);
    let mut c = stepper(0.65);
    let r = c.move_and_slide(v3(0.5, 0.0, 0.0), &[], &[wall]);
    assert!(
        (r.position.x.to_f64() - 0.7).abs() < 1e-3,
        "x = {}",
        r.position.x.to_f64()
    );
    assert!(
        r.position.y.to_f64().abs() < 1e-3,
        "y = {}",
        r.position.y.to_f64()
    );
}

#[test]
fn sdf_push_out_applies_below_the_radius_by_the_exact_shortfall() {
    // bottom sample (hemisphere centre) at distance 0.28 < radius 0.3: pushed
    // up by 0.02 -> centre y = 0.88 + 0.02 = 0.9
    let floor = [plane_sdf(0.0, 1.0, 0.0)];
    let mut c = CharacterController::new(v3(0.0, 0.88, 0.0), cfg());
    c.move_and_slide(Vec3Fix::ZERO, &[], &floor);
    assert!(
        (c.position.y.to_f64() - 0.9).abs() < 1e-3,
        "y = {}",
        c.position.y.to_f64()
    );
}

#[test]
fn sdf_push_out_uses_the_top_hemisphere_centre_for_a_ceiling() {
    // ceiling at y = 2 (normal pointing down). Top hemisphere centre = y + 0.6;
    // y = 1.2 -> 1.8 -> distance 0.2 < 0.3 -> pushed down by 0.1 to y = 1.1.
    let ceiling = [plane_sdf(0.0, -1.0, -2.0)];
    let mut c = CharacterController::new(v3(0.0, 1.2, 0.0), cfg());
    c.move_and_slide(Vec3Fix::ZERO, &[], &ceiling);
    assert!(
        (c.position.y.to_f64() - 1.1).abs() < 1e-3,
        "y = {}",
        c.position.y.to_f64()
    );
}

#[test]
fn static_body_beyond_the_probe_does_not_ground() {
    // hemisphere centre 0.2 above the body-sphere top: farther than
    // probe + skin = 0.11 -> airborne.
    let bodies = [static_at(0.0, 0.0, 0.0)];
    // feet = y - 0.6; top at 0.3; feet = 0.5 -> y = 1.1
    let mut c = CharacterController::new(v3(0.0, 1.1, 0.0), cfg());
    assert!(!c.move_and_slide(Vec3Fix::ZERO, &bodies, &[]).grounded);
}

#[test]
fn max_slope_angle_is_compared_through_its_cosine() {
    // limit 0.5 rad (28.6 deg): cos = 0.8776, sin = 0.4794. A 40 deg slope
    // (n_y = 0.766) is steeper than the limit, a 20 deg slope (0.940) is not.
    let mut conf = cfg();
    conf.max_slope_angle = fx(0.5);
    let grounded_on = |theta: f64| {
        let (s, co) = (theta.sin() as f32, theta.cos() as f32);
        let floor = [plane_sdf(s, co, 0.0)];
        let y_feet = 0.35 / theta.cos();
        let mut c = CharacterController::new(v3(0.0, y_feet + 0.6, 0.0), conf);
        c.move_and_slide(Vec3Fix::ZERO, &[], &floor).grounded
    };
    assert!(grounded_on(20.0_f64.to_radians()));
    assert!(!grounded_on(40.0_f64.to_radians()));
}

#[test]
fn push_impulse_threshold_is_strict_at_the_exact_combined_distance() {
    // body exactly radius + body_radius away in fixed point: not a push
    let c = CharacterController::new(v3(0.0, 0.0, 0.0), cfg());
    let br = fx(0.5);
    let body = RigidBody::new(
        Vec3Fix::new(c.config.radius + br, Fix128::ZERO, Fix128::ZERO),
        Fix128::ONE,
    );
    assert!(c.compute_push_impulses(&[body], br).is_empty());
    // one ulp inside: a push with a tiny overlap
    let inside = RigidBody::new(
        Vec3Fix::new(
            c.config.radius + br - Fix128::from_raw(0, 1 << 20),
            Fix128::ZERO,
            Fix128::ZERO,
        ),
        Fix128::ONE,
    );
    assert_eq!(c.compute_push_impulses(&[inside], br).len(), 1);
}

#[test]
fn result_and_controller_velocity_equal_the_displacement_argument() {
    // pins the current contract: `velocity` is set to the displacement passed
    // in (a per-frame distance, see AUD-A-S3W3-005 / the units note)
    let mut c = CharacterController::new(v3(0.0, 9.0, 0.0), cfg());
    let d = v3(0.25, -0.5, 0.125);
    let r = c.move_and_slide(d, &[], &[]);
    assert_eq!(c.velocity, d);
    assert_eq!(r.velocity, d);
}

#[test]
fn stair_snap_down_settles_the_character_on_the_ledge() {
    // Walk a grounded character onto a 0.2 m ledge (floor y = 0 plus a block
    // x in [1,3], top 0.2). The controller documents "Snap down to find the
    // actual stair surface"; the capsule bottom must end within a skin of the
    // ledge top rather than hovering one step_height above it.
    let sdfs = [
        plane_sdf(0.0, 1.0, 0.0),
        block([1.0, 0.0, -5.0], [3.0, 0.2, 5.0]),
    ];
    let mut c = CharacterController::new(v3(0.0, 0.92, 0.0), cfg());
    c.move_and_slide(Vec3Fix::ZERO, &[], &sdfs);
    for _ in 0..20 {
        c.move_and_slide(v3(0.2, 0.0, 0.0), &[], &sdfs);
    }
    assert!(c.position.x.to_f64() > 2.0, "ledge not climbed");
    let hover = c.position.y.to_f64() - 0.9 - 0.2;
    assert!(
        hover <= 0.02,
        "capsule bottom hovers {hover} m above the ledge top (step_height - ledge = 0.1)"
    );
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-019: the stair snap-down ray is cast against the character's own previous feet plane (y - h/2 + r), so for a capsule with h/2 - r < step_height the step ends h/2 - r below the start (h 0.8, r 0.3: y = -0.1 from 0.0) instead of settling on the stair; unreachable with the default 1.8 m capsule where the ray never hits"]
fn stair_step_with_a_short_capsule_does_not_lower_the_character() {
    // Capsule height 0.8, radius 0.3 (h/2 - r = 0.1 < step_height 0.3): the
    // step-up point is 0.3 above the start, and the "snap down to the stair
    // surface" ray is cast against the character's own previous feet plane.
    // A character that steps onto something must not end below where it
    // started.
    let mut conf = cfg();
    conf.height = fx(0.8);
    let mut c = CharacterController::new(v3(0.38, 0.0, 0.0), conf);
    c.grounded = true;
    let r = c.move_and_slide(v3(0.9, 0.0, 0.0), &[static_at(1.0, 0.0, 0.0)], &[]);
    assert!(
        (r.position.x.to_f64() - 1.29).abs() < 1e-6,
        "step not taken: x = {}",
        r.position.x.to_f64()
    );
    assert!(
        r.position.y.to_f64() >= -1e-9,
        "stepped character ended at y = {} below its start 0.0",
        r.position.y.to_f64()
    );
}
