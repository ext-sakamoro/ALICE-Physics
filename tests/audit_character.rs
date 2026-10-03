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
    let d2 = dist.clone();
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
