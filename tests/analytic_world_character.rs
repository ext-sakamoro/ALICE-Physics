//! Oracles for `PhysicsWorld::move_character`: move-and-slide of a
//! `CharacterController` capsule against the geometry the world collides with
//! (static colliders, body shapes), compared with closed forms.
//!
//! # Expected values
//!
//! Every expected position is written from the geometry by hand; the closed form
//! is in a comment next to each assertion. None calls the code under test.
//!
//! The controller is a capsule: centre `c`, total height `h` (hemispheres
//! included), radius `r`, so its segment runs from `c − (h/2 − r)·Y` to
//! `c + (h/2 − r)·Y`. With `CharacterConfig::default()` (`h = 1.8`, `r = 0.3`,
//! skin `s = 0.01`, step height `0.3`, slope limit `0.785` rad, 4 slides) a
//! capsule resting on the floor `y = y0` keeps a gap `s` under it, so its centre
//! is at `y0 + h/2 + s = y0 + (h/2 − r) + r + s = y0 + 0.91`.
//!
//! One slide step: the capsule moves along the unit direction `d` toward a
//! contact at distance `t` with normal `n` only as far as leaves a gap `s` along
//! `n`, `t − s / (−d·n)` (at least 0), and the rest of the displacement is
//! projected onto the contact plane, `rest − n (rest · n)`. Starting at gap `s`
//! from a plane, the result after the whole move is therefore `c + D − n (D · n)`
//! (the normal part is cancelled, the tangential part is kept whole).
//!
//! # Tolerances
//!
//! - `EXACT = 1e-12`: planes (closed-form capsule cast).
//! - `ITER = 1e-8`: boxes, triangle meshes and height fields (conservative
//!   advancement stops at a gap below `2⁻³²`, a few of them per move).
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;

const EXACT: f64 = 1e-12;
const ITER: f64 = 1e-8;

/// Default config: centre height above the floor, `h/2 + s`.
const STAND: f64 = 0.91;
const R: f64 = 0.3;
const SKIN: f64 = 0.01;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

#[track_caller]
fn assert_pos(got: Vec3Fix, want: [f64; 3], tol: f64) {
    let g = f3(got);
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() < tol,
            "position {g:?} but the closed form is {want:?}"
        );
    }
}

fn floor_plane(w: &mut PhysicsWorld, y: f64) -> usize {
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        fx(y),
    )))
}

/// A static body with a box shape.
fn static_box(w: &mut PhysicsWorld, center: [f64; 3], half: [f64; 3]) -> usize {
    let i = w.add_body(RigidBody::new_static(v3(center[0], center[1], center[2])));
    assert!(w.set_body_shape(
        i,
        &Shape::Box {
            half_extents: v3(half[0], half[1], half[2]),
        }
    ));
    i
}

fn ctrl_at(p: [f64; 3]) -> CharacterController {
    CharacterController::new_default(v3(p[0], p[1], p[2]))
}

/// A controller resting on a surface: one zero move sets `grounded`.
fn settled(w: &PhysicsWorld, p: [f64; 3]) -> CharacterController {
    let mut c = ctrl_at(p);
    w.move_character(&mut c, Vec3Fix::ZERO);
    c
}

// ============================================================ floors

#[test]
fn walking_on_a_plane_keeps_the_standing_height() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    let mut c = ctrl_at([0.0, STAND, 0.0]);
    // oracle: a horizontal move is parallel to the floor: nothing is hit, the
    // centre moves by the full displacement at height h/2 + s.
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_pos(res.position, [1.0, STAND, 0.0], EXACT);
    assert_eq!(res.position, c.position);
    assert!(res.grounded && c.grounded, "standing on the plane");
    assert_eq!(c.ground_body_index, None, "a static collider has no body");

    // oracle: falling while walking, D = (1, −0.5, 0) from 0.29 above the
    // standing height: the floor normal (0, 1, 0) cancels the vertical part,
    // x = 1 is kept whole, y = h/2 + s.
    let mut c = ctrl_at([0.0, STAND + 0.29, 0.0]);
    let res = w.move_character(&mut c, v3(1.0, -0.5, 0.0));
    assert_pos(res.position, [1.0, STAND, 0.0], EXACT);
    assert!(res.grounded);
    assert_eq!(res.velocity, v3(1.0, -0.5, 0.0), "velocity = input");
}

#[test]
fn walking_on_a_triangle_mesh_floor_keeps_the_standing_height() {
    let mut w = world();
    let verts = [
        v3(-10.0, 0.0, -10.0),
        v3(10.0, 0.0, -10.0),
        v3(10.0, 0.0, 10.0),
        v3(-10.0, 0.0, 10.0),
    ];
    w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 2, 1, 0, 3, 2],
    )));
    // oracle: landing from 0.29 above with D = (2, −0.5, 0.5), then the floor
    // y = 0 gives centre y = 0.91 and the horizontal part (2, 0.5) kept whole
    // (the move crosses the diagonal edge of the two triangles).
    let mut c = ctrl_at([-1.0, STAND + 0.29, 0.0]);
    let res = w.move_character(&mut c, v3(2.0, -0.5, 0.5));
    assert_pos(res.position, [1.0, STAND, 0.5], ITER);
    assert!(res.grounded);
    // oracle: walking on: parallel, the full displacement.
    let res = w.move_character(&mut c, v3(-0.5, 0.0, 1.0));
    assert_pos(res.position, [0.5, STAND, 1.5], ITER);
    assert!(res.grounded);
}

#[test]
fn walking_on_a_height_field_floor_keeps_the_standing_height() {
    let mut w = world();
    // flat field at y = 0.25 over [0, 8]²
    w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
        9,
        9,
        fx(1.0),
        Vec3Fix::ZERO,
        fx(0.25),
    )));
    // oracle: landing: centre y = 0.25 + 0.91, horizontal (2, 0) kept.
    let mut c = ctrl_at([2.0, 0.25 + STAND + 0.29, 2.0]);
    let res = w.move_character(&mut c, v3(2.0, -0.5, 0.0));
    assert_pos(res.position, [4.0, 0.25 + STAND, 2.0], ITER);
    assert!(res.grounded);
    // oracle: walking on: parallel, the full displacement.
    let res = w.move_character(&mut c, v3(0.0, 0.0, 1.5));
    assert_pos(res.position, [4.0, 0.25 + STAND, 3.5], ITER);
    assert!(res.grounded);
}

// ============================================================ walls

#[test]
fn moving_into_a_wall_stops_at_radius_plus_skin_and_keeps_the_tangential_part() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // wall x = 2 (normal −X toward the character): dot((−1,0,0), p) = −2
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(-1.0, 0.0, 0.0),
        fx(-2.0),
    )));
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    assert!(c.grounded);
    // oracle: D = (3, 0, 1): x stops at 2 − r − s, the tangential z = 1 is kept
    // whole (slide = D − n (D·n) for the part after contact), y unchanged.
    let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
    assert_pos(res.position, [2.0 - R - SKIN, STAND, 1.0], EXACT);
    assert!(res.grounded);
}

#[test]
fn a_box_body_wall_is_its_box_not_a_sphere() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // box face x = 2 (box x ∈ [2, 4], y ∈ [0, 4], z ∈ [−5, 5])
    static_box(&mut w, [3.0, 2.0, 0.0], [1.0, 2.0, 5.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    // oracle: as the plane wall: x = 2 − r − s, z = 1 kept.
    let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
    assert_pos(res.position, [2.0 - R - SKIN, STAND, 1.0], ITER);
}

/// A vertical box edge where the old sphere approximation (a sphere of the
/// character's radius at the body's position) sees nothing.
#[test]
fn a_box_corner_stops_the_capsule_where_the_old_sphere_approximation_passes_through() {
    let mut w = world();
    // box x, z ∈ [−1, 1], y ∈ [0, 2]
    let b = static_box(&mut w, [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]);
    let config = CharacterConfig {
        max_slides: 1,
        ..CharacterConfig::default()
    };
    let start = v3(-3.0, 1.0, 1.2);
    let d = v3(6.0, 0.0, 0.0);

    // oracle: the capsule's segment y ∈ [0.4, 1.6] lies beside the vertical edge
    // x = −1, z = 1; in the xz plane its centre moving along z = 1.2 reaches
    // distance r from (−1, 1) at x = −1 − √(r² − 0.2²); normal
    // n = (x + 1, 0, 0.2)/r, −d·n = √(r² − 0.2²)/r; one slide (max_slides = 1):
    // stops s / (−d·n) short of the contact along d.
    let mut c = CharacterController::new(start, config);
    let res = w.move_character(&mut c, d);
    let dx = (R * R - 0.04f64).sqrt();
    assert_pos(res.position, [-1.0 - dx - SKIN * R / dx, 1.0, 1.2], ITER);

    // characterisation: `move_and_slide` treats the body as a sphere of radius
    // r at (0, 1, 0) grown by r: |z| = 1.2 > 0.6, no hit, the capsule passes
    // through the box to x = 3.
    let mut old = CharacterController::new(start, config);
    let res_old = old.move_and_slide(d, &w.bodies, &[]);
    assert_eq!(res_old.position, v3(3.0, 1.0, 1.2));
    assert!(w.bodies[b].is_static());
}

/// Two vertical walls meeting at x = 2 at 60° (normals (−1/2, 0, ±√3/2)): a
/// move into the corner ends a skin width from both.
#[test]
fn an_acute_corner_stops_a_skin_width_from_both_walls() {
    let mut w = world();
    let h = 3f64.sqrt() / 2.0;
    // dot(n, p) = dot(n, (2, 0, 0)) = −1 for both walls
    for n in [[-0.5, 0.0, h], [-0.5, 0.0, -h]] {
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            v3(n[0], n[1], n[2]),
            fx(-1.0),
        )));
    }
    let mut c = ctrl_at([0.0, 5.0, 0.3]);
    // oracle: the point on the bisector z = 0 at distance r + s from both walls:
    // 0.5 (2 − x) = r + s, x = 2 − 2 (r + s); y unchanged (walls are vertical).
    for d in [[3.0, 0.0, 0.5], [3.0, 0.0, 0.0], [3.0, 0.0, -1.0]] {
        let res = w.move_character(&mut c, v3(d[0], d[1], d[2]));
        assert_pos(res.position, [2.0 - 2.0 * (R + SKIN), 5.0, 0.0], ITER);
        assert!(!res.grounded);
    }
}

// ============================================================ slopes

/// A plane through the origin with unit normal `n` and a controller resting on
/// it: the bottom hemisphere centre at `(r + s)·n`.
fn slope_world(n: [f64; 3]) -> (PhysicsWorld, [f64; 3]) {
    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(n[0], n[1], n[2]),
        Fix128::ZERO,
    )));
    let k = R + SKIN;
    let c = [k * n[0], k * n[1] + (0.9 - R), k * n[2]];
    (w, c)
}

#[test]
fn a_slope_below_the_limit_is_walkable_and_grounded() {
    // 30° slope rising toward +X: n = (−sin 30°, cos 30°, 0); cos 30° ≥ cos 0.785
    let a = 30f64.to_radians();
    let n = [-a.sin(), a.cos(), 0.0];
    let (w, c0) = slope_world(n);
    let mut c = settled(&w, c0);
    assert_pos(c.position, c0, EXACT);
    assert!(c.grounded, "30° is below the 45° limit");
    // oracle: walking up, D = (1, 0, 0): c + D − n (D·n), D·n = −sin 30°.
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    let dn = -a.sin();
    assert_pos(
        res.position,
        [c0[0] + 1.0 - n[0] * dn, c0[1] - n[1] * dn, c0[2]],
        EXACT,
    );
    assert!(res.grounded);
}

#[test]
fn a_slope_above_the_limit_is_not_grounded_and_slides() {
    // 60° slope rising toward +X: n = (−sin 60°, cos 60°, 0); cos 60° < cos 0.785
    let a = 60f64.to_radians();
    let n = [-a.sin(), a.cos(), 0.0];
    let (w, c0) = slope_world(n);
    let mut c = settled(&w, c0);
    assert!(!c.grounded, "60° is above the 45° limit");

    // oracle: gravity, D = (0, −1, 0): slides down the plane,
    // c + D − n (D·n) with D·n = −cos 60°.
    let res = w.move_character(&mut c, v3(0.0, -1.0, 0.0));
    let dn = -a.cos();
    assert_pos(
        res.position,
        [c0[0] - n[0] * dn, c0[1] - 1.0 - n[1] * dn, c0[2]],
        EXACT,
    );
    assert!(!res.grounded);

    // oracle: walking up a non-walkable slope is blocked like a wall: the contact
    // is at t = s / sin 60° and −d·n = sin 60°, so the capsule moves
    // t − s / sin 60° = 0; the rest (pushing up the slope) is projected onto the
    // horizontal wall normal (−1, 0, 0): nothing is left, the capsule stays.
    let mut c = settled(&w, c0);
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_pos(res.position, c0, EXACT);
    assert!(!res.grounded);
}

// ============================================================ steps

#[test]
fn a_ledge_lower_than_the_step_height_is_stepped_onto() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // ledge top y = 0.2 < step height 0.3, face x = 1
    let b = static_box(&mut w, [3.0, 0.1, 0.0], [2.0, 0.1, 2.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    assert!(c.grounded);
    // oracle: up by the step height, across, down onto the top: centre
    // y = 0.2 + h/2 + s, x = 2 (the full horizontal displacement).
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [2.0, 0.2 + STAND, 0.0], ITER);
    assert!(res.grounded);
    assert_eq!(c.ground_body_index, Some(b), "standing on the ledge body");
}

#[test]
fn a_ledge_higher_than_the_step_height_blocks() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // ledge top y = 0.5 > step height 0.3, face x = 1
    static_box(&mut w, [3.0, 0.25, 0.0], [2.0, 0.25, 2.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    // oracle: the bottom hemisphere centre (y = 0.31) is below the top: the face
    // x = 1 stops the centre at x = 1 − r − s.
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [1.0 - R - SKIN, STAND, 0.0], ITER);
    assert!(res.grounded);
}

#[test]
fn without_a_step_height_the_low_ledge_edge_blocks() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    static_box(&mut w, [3.0, 0.1, 0.0], [2.0, 0.1, 2.0]);
    let config = CharacterConfig {
        step_height: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    let mut c = CharacterController::new(v3(0.0, STAND, 0.0), config);
    w.move_character(&mut c, Vec3Fix::ZERO);
    // oracle: the bottom hemisphere centre (y = 0.31, 0.11 above the edge
    // (1, 0.2)) touches the edge at x = 1 − √(r² − 0.11²), normal
    // n = (x − 1, 0.11, 0)/r (steeper than the limit: a wall), −d·n = √(r² −
    // 0.11²)/r: stops s / (−d·n) short of the contact along d = (1, 0, 0).
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    let dx = (R * R - 0.11f64 * 0.11).sqrt();
    assert_pos(res.position, [1.0 - dx - SKIN * R / dx, STAND, 0.0], ITER);
}

#[test]
fn a_low_ceiling_limits_the_step_rise() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // ceiling y = 1.96: the capsule top (1.81) is 0.15 below it
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, -1.0, 0.0),
        fx(-1.96),
    )));
    // ledge top y = 0.12, face x = 1: the edge contact (bottom hemisphere centre
    // 0.19 above it, n·Y = 0.19/0.3 < cos 0.785) is too steep to walk
    let b = static_box(&mut w, [3.0, 0.06, 0.0], [2.0, 0.06, 2.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    // oracle: the rise is the ceiling gap less a skin, 0.15 − s = 0.14 (> 0.12 − s,
    // enough to clear the ledge); across, then down onto the top: centre
    // y = 0.12 + h/2 + s, x = 2.
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    assert_pos(res.position, [2.0, 0.12 + STAND, 0.0], ITER);
    assert_eq!(c.ground_body_index, Some(b));
}

#[test]
fn a_raised_move_that_gets_less_far_is_not_taken() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // the low ledge of `without_a_step_height_the_low_ledge_edge_blocks`
    static_box(&mut w, [3.0, 0.1, 0.0], [2.0, 0.1, 2.0]);
    // a beam over it, bottom y = 1.85 (0.04 above the capsule top), from x = 0.3:
    // the raised capsule (top 2.11) runs into the beam almost at once
    static_box(&mut w, [2.3, 2.85, 0.0], [2.0, 1.0, 2.0]);
    let mut c = settled(&w, [0.0, STAND, 0.0]);
    assert!(c.grounded);
    // oracle: the raised move lands back on the floor near x = 0, short of the
    // plain sweep, so the plain sweep stands: stopped at the ledge edge as with no
    // step height, x = 1 − √(r² − 0.11²) − s·r/√(r² − 0.11²).
    let res = w.move_character(&mut c, v3(2.0, 0.0, 0.0));
    let dx = (R * R - 0.11f64 * 0.11).sqrt();
    assert_pos(res.position, [1.0 - dx - SKIN * R / dx, STAND, 0.0], ITER);
    assert!(res.grounded);
}

// ============================================================ bodies

#[test]
fn the_own_body_is_excluded_by_the_filter_and_ground_bodies_carry_their_velocity() {
    let mut w = world();
    // a kinematic platform box, top y = 0, moving +X
    let mut platform = RigidBody::new_kinematic(v3(0.0, -0.5, 0.0));
    platform.velocity = v3(2.0, 0.0, 0.0);
    let p = w.add_body(platform);
    assert!(w.set_body_shape(
        p,
        &Shape::Box {
            half_extents: v3(5.0, 0.5, 5.0),
        }
    ));
    // the character's own body: a sphere at the capsule centre
    let own = w.add_body_with_radius(RigidBody::new_kinematic(v3(0.0, STAND, 0.0)), fx(0.3));

    // oracle: excluding its own body, the character stands on the platform:
    // grounded on body `p`, platform velocity = (2, 0, 0).
    let mut c = ctrl_at([0.0, STAND, 0.0]);
    let filter = RayFilter::default().excluding_body(own);
    let res = w.move_character_with_filter(&mut c, Vec3Fix::ZERO, &filter);
    assert_pos(res.position, [0.0, STAND, 0.0], EXACT);
    assert!(res.grounded);
    assert_eq!(c.ground_body_index, Some(p));
    assert_eq!(res.platform_velocity, v3(2.0, 0.0, 0.0));
    // oracle: the next move adds the platform velocity (the existing
    // `move_and_slide` semantics): D + platform = (1, 0, 0) + (2, 0, 0).
    let res = w.move_character_with_filter(&mut c, v3(1.0, 0.0, 0.0), &filter);
    assert_pos(res.position, [3.0, STAND, 0.0], ITER);

    // oracle: without the filter the character starts inside its own body and
    // does not move (see `starting_overlap_does_not_move`).
    let mut c = ctrl_at([0.0, STAND, 0.0]);
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_pos(res.position, [0.0, STAND, 0.0], EXACT);
}

// ============================================================ old behaviour

/// Characterisation of `CharacterController::move_and_slide`, kept to document
/// why `PhysicsWorld::move_character` exists: the old method sees bodies only (as
/// spheres) and SDF colliders, not the world's static colliders, so a character
/// falls through a plane floor. Asserts the measured old behaviour; the new
/// method is checked on the same scene in
/// `walking_on_a_plane_keeps_the_standing_height`.
#[test]
fn characterisation_old_move_and_slide_falls_through_a_plane_floor() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    let mut c = ctrl_at([0.0, STAND, 0.0]);
    let res = c.move_and_slide(v3(0.0, -2.0, 0.0), &w.bodies, &[]);
    // the full displacement, 2 below the start: through the floor y = 0
    assert_pos(res.position, [0.0, STAND - 2.0, 0.0], EXACT);
    assert!(!res.grounded);

    let mut c = ctrl_at([0.0, STAND, 0.0]);
    let res = w.move_character(&mut c, v3(0.0, -2.0, 0.0));
    assert_pos(res.position, [0.0, STAND, 0.0], EXACT);
    assert!(res.grounded);
}

// ============================================================ degenerate inputs

#[test]
fn zero_displacement_does_not_move() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // expected: the position is unchanged bit for bit; grounded is detected.
    let mut c = ctrl_at([0.25, STAND, -0.5]);
    let before = c.position;
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, before);
    assert!(res.grounded);
    // expected: in the air: unchanged, not grounded.
    let mut c = ctrl_at([0.25, 5.0, -0.5]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, v3(0.25, 5.0, -0.5));
    assert!(!res.grounded);
}

#[test]
fn an_empty_world_moves_the_full_displacement() {
    let w = world();
    // expected: nothing to hit: position + D exactly, not grounded.
    let mut c = ctrl_at([1.0, 2.0, 3.0]);
    let res = w.move_character(&mut c, v3(-4.0, 5.5, 0.25));
    assert_eq!(res.position, v3(-3.0, 7.5, 3.25));
    assert!(!res.grounded);
    assert_eq!(c.ground_body_index, None);
}

#[test]
fn large_displacements_are_swept_whole() {
    // expected: 1e9 in an empty world: exactly position + D.
    let w = world();
    let mut c = ctrl_at([0.0, 0.0, 0.0]);
    let res = w.move_character(&mut c, v3(1e9, 0.0, 0.0));
    assert_eq!(res.position, v3(1e9, 0.0, 0.0));
    // oracle: a fall of 1e9 onto the plane y = 0 lands at h/2 + s (no tunnelling).
    let mut w = world();
    floor_plane(&mut w, 0.0);
    let mut c = ctrl_at([0.0, 10.0, 0.0]);
    let res = w.move_character(&mut c, v3(0.0, -1e9, 0.0));
    assert_pos(res.position, [0.0, STAND, 0.0], EXACT);
    assert!(res.grounded);
}

/// Design: a capsule that starts overlapping a collider does not move. The
/// cast reports such a contact at `t = 0` with no surface normal (`−direction`),
/// so there is no plane to slide along; the controller keeps its position and
/// such a contact does not count as ground.
#[test]
fn starting_overlap_does_not_move() {
    let mut w = world();
    // box x, z ∈ [−1, 1], y ∈ [0, 2]; the capsule centre (0.5, 1, 0) is inside
    static_box(&mut w, [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]);
    let mut c = ctrl_at([0.5, 1.0, 0.0]);
    let res = w.move_character(&mut c, v3(3.0, 0.0, 0.0));
    assert_eq!(res.position, v3(0.5, 1.0, 0.0));
    assert!(!res.grounded);
    let res = w.move_character(&mut c, v3(0.0, -3.0, 0.0));
    assert_eq!(res.position, v3(0.5, 1.0, 0.0));
}

#[test]
fn huge_coordinates_give_the_same_answers() {
    let o = 1e6;
    let mut w = world();
    floor_plane(&mut w, o);
    static_box(&mut w, [o + 3.0, o + 2.0, o], [1.0, 2.0, 5.0]);
    let mut c = settled(&w, [o, o + STAND, o]);
    assert!(c.grounded);
    // oracle: as at the origin: x = o + 2 − r − s, z = o + 1, y = o + 0.91.
    let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
    assert_pos(res.position, [o + 2.0 - R - SKIN, o + STAND, o + 1.0], ITER);
    assert!(res.grounded);
}

// ============================================================ starting overlap

// A capsule that starts overlapping a collider by less than its radius (its
// segment outside every solid) is pushed out first: each push goes along the
// collider's normal at the segment point nearest it, by the penetration depth
// plus the skin width, so the capsule ends a skin width clear of the surface. The
// expected values are the same closed forms as for a resting capsule: the core
// segment at distance `r + s` from the surface along its normal.

/// A sphere body of radius `r` (no shape: the body's collision sphere).
fn sphere_body(w: &mut PhysicsWorld, center: [f64; 3], r: f64) -> usize {
    w.add_body_with_radius(
        RigidBody::new_static(v3(center[0], center[1], center[2])),
        fx(r),
    )
}

#[test]
fn a_capsule_embedded_in_a_plane_floor_is_pushed_up_to_the_standing_height() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // embedded by d = 0.1: centre y = h/2 − d = 0.8, segment bottom at
    // 0.8 − (h/2 − r) = 0.2 < r.
    // oracle: pushed along the floor normal +Y by d + s: y = (h/2 − r) + r + s =
    // h/2 + s = 0.91, x and z unchanged (zero displacement), on the ground.
    let mut c = ctrl_at([0.25, 0.8, -0.5]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [0.25, STAND, -0.5], EXACT);
    assert_eq!(res.position, c.position);
    assert!(res.grounded, "a skin width above the floor");

    // oracle: depth 0.25 (segment bottom 0.05 above the floor) ends at the same
    // height.
    let mut c = ctrl_at([0.0, 0.65, 0.0]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [0.0, STAND, 0.0], EXACT);
}

#[test]
fn after_the_push_out_the_move_is_made_from_the_freed_position() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // oracle: pushed to y = h/2 + s, then D = (1, 0, 0.5) is parallel to the
    // floor and taken whole: (x + 1, 0.91, z + 0.5).
    let mut c = ctrl_at([0.0, 0.8, 0.0]);
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.5));
    assert_pos(res.position, [1.0, STAND, 0.5], EXACT);
    assert!(res.grounded);
}

#[test]
fn a_capsule_embedded_in_a_box_wall_from_the_side_ends_radius_plus_skin_from_the_face() {
    let mut w = world();
    // box x ∈ [2, 4], y ∈ [−4, 4], z ∈ [−5, 5]; face x = 2, normal −X
    static_box(&mut w, [3.0, 0.0, 0.0], [1.0, 4.0, 5.0]);
    // embedded by 0.1: centre x = 2 − r + 0.1 = 1.8 (segment at distance 0.2)
    // oracle: pushed along −X by 0.1 + s: x = 2 − r − s = 1.69, y and z kept.
    let mut c = ctrl_at([1.8, 1.0, 0.5]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [2.0 - R - SKIN, 1.0, 0.5], ITER);
    assert!(!res.grounded, "a wall is not ground");
}

#[test]
fn a_capsule_embedded_in_a_sphere_body_is_pushed_out_along_the_centre_line() {
    let mut w = world();
    // sphere body of radius 1 at (0, 1, 0); the capsule centre is level with it,
    // so the segment point nearest the sphere centre is the capsule centre.
    sphere_body(&mut w, [0.0, 1.0, 0.0], 1.0);
    // centre (0.9, 1, 0.9): distance 0.9·√2 ≈ 1.2728 < R + r = 1.3.
    // oracle: pushed along the centre line (1, 0, 1)/√2 to distance R + r + s =
    // 1.31: x = z = 1.31/√2.
    let mut c = ctrl_at([0.9, 1.0, 0.9]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    let k = (1.0 + R + SKIN) / 2f64.sqrt();
    assert_pos(res.position, [k, 1.0, k], EXACT);

    // oracle: from the side along −X: centre (−1.2, 1, 0) ends at x = −1.31.
    let mut c = ctrl_at([-1.2, 1.0, 0.0]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [-(1.0 + R + SKIN), 1.0, 0.0], EXACT);
}

#[test]
fn a_capsule_embedded_in_a_floor_and_a_wall_is_freed_from_both() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // wall x = 2 (normal −X): dot((−1,0,0), p) = −2
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(-1.0, 0.0, 0.0),
        fx(-2.0),
    )));
    // embedded 0.1 in the floor (y = 0.8) and 0.05 in the wall (x = 1.75)
    // oracle: deepest first (the floor), then the wall: y = h/2 + s,
    // x = 2 − r − s.
    let mut c = ctrl_at([1.75, 0.8, 0.0]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [2.0 - R - SKIN, STAND, 0.0], EXACT);
    assert!(res.grounded);
}

/// A capsule whose segment is inside a solid (deeper than its radius) has no
/// push-out direction from the distance queries: it keeps its position.
#[test]
fn a_capsule_deep_inside_a_large_box_is_left_blocked() {
    let mut w = world();
    // box x, y, z ∈ [−10, 10]: the whole capsule is inside
    static_box(&mut w, [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut c = ctrl_at([1.0, 2.0, 3.0]);
    // expected: unchanged bit for bit, for a zero and a non-zero displacement
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, v3(1.0, 2.0, 3.0));
    assert!(!res.grounded);
    let res = w.move_character(&mut c, v3(0.0, 5.0, 0.0));
    assert_eq!(res.position, v3(1.0, 2.0, 3.0));

    // expected: a capsule whose segment crosses a plane floor (centre y = 0.3,
    // segment y ∈ [−0.3, 0.9]) is blocked too.
    let mut w = world();
    floor_plane(&mut w, 0.0);
    let mut c = ctrl_at([0.0, 0.3, 0.0]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, v3(0.0, 0.3, 0.0));
}

#[test]
fn a_capsule_clear_of_everything_is_not_pushed() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    sphere_body(&mut w, [5.0, 1.0, 0.0], 1.0);
    // expected: exactly r + s from the sphere (x = 5 − 1.31) and a skin above the
    // floor: no overlap, nothing moves (bit for bit).
    let p = v3(5.0 - 1.31, STAND, 0.0);
    let mut c = CharacterController::new_default(p);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, p);
}

#[test]
fn a_capsule_taller_than_the_gap_between_floor_and_ceiling_is_left_blocked() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // ceiling y = 1.5 (normal −Y): dot((0,−1,0), p) = −1.5; the capsule is 1.8 tall
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, -1.0, 0.0),
        fx(-1.5),
    )));
    // centre y = 0.75 overlaps both by 0.15; pushing out of either drives the
    // segment into the other (no position is clear).
    // expected: unchanged bit for bit, also for a move.
    let mut c = ctrl_at([0.0, 0.75, 0.0]);
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_eq!(res.position, v3(0.0, 0.75, 0.0));
    assert!(!res.grounded);
}

#[test]
fn overlaps_are_pushed_out_deepest_first() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    // sphere body R = 1 at (−1.2, 0, 0)
    sphere_body(&mut w, [-1.2, 0.0, 0.0], 1.0);
    // centre (0, 0.8, 0): segment y ∈ [0.2, 1.4]; floor depth r − 0.2 = 0.1,
    // sphere depth R + r − |(1.2, 0.2)| ≈ 0.0834 (nearest segment point is the
    // bottom end (0, 0.2)).
    // oracle, deepest (the floor) first: up by 0.1 + s to y = 0.91 (bottom end
    // (0, 0.31)); then along u = (1.2, 0.31)/|·| by R + r + s − |(1.2, 0.31)|,
    // which leaves the floor clear.
    let mut c = ctrl_at([0.0, 0.8, 0.0]);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    let (ux, uy) = (1.2_f64, 0.31_f64);
    let len = ux.hypot(uy);
    let push = 1.0 + R + SKIN - len;
    let want = [push * ux / len, STAND + push * uy / len, 0.0];
    assert_pos(res.position, want, 1e-9);
}

#[test]
fn at_most_max_slides_pushes_are_made() {
    let mut w = world();
    floor_plane(&mut w, 0.0);
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(-1.0, 0.0, 0.0),
        fx(-2.0),
    )));
    // the floor-and-wall overlap of
    // `a_capsule_embedded_in_a_floor_and_a_wall_is_freed_from_both` needs two
    // pushes.
    // expected: with max_slides = 1 it is not freed and keeps its position.
    let config = CharacterConfig {
        max_slides: 1,
        ..CharacterConfig::default()
    };
    let mut c = CharacterController::new(v3(1.75, 0.8, 0.0), config);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_eq!(res.position, v3(1.75, 0.8, 0.0));
    // oracle: with max_slides = 2 it is freed: (2 − r − s, h/2 + s, 0).
    let config = CharacterConfig {
        max_slides: 2,
        ..CharacterConfig::default()
    };
    let mut c = CharacterController::new(v3(1.75, 0.8, 0.0), config);
    let res = w.move_character(&mut c, Vec3Fix::ZERO);
    assert_pos(res.position, [2.0 - R - SKIN, STAND, 0.0], EXACT);
}
