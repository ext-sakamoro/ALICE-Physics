//! Oracles for static colliders in a `PhysicsWorld`: planes, height fields and
//! triangle meshes that bodies rest on.
//!
//! # What is measured
//!
//! A body is a sphere of a given collision radius. A static collider is an
//! immovable surface. After one `step` the position-based solver leaves a
//! penetrating sphere **exactly one radius away from the surface, along the
//! surface normal**: `p' = p + n·(r − d)` with `d` the signed distance from the
//! centre to the surface. The expectations below are that closed form, written
//! from the geometry of each scene (a plane's normal and offset, a ramp's slope, a
//! valley's half-angle) and not obtained from the colliders' own methods.
//!
//! Gravity is zero in those scenes, so the single step moves the body only by the
//! correction. A separate scene with gravity checks the resting height of a body
//! dropped on a floor.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::{TriMesh, Triangle};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn arr(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

/// One substep per step, so a `step` is one collision pass: the position-based
/// solver turns a correction into a velocity (`Δx / dt`) and the next substep
/// would carry the body on by that much again. With a single substep the position
/// after the step is the correction alone, which is what the closed forms state.
fn config(gravity: f64) -> SolverConfig {
    SolverConfig {
        substeps: 1,
        gravity: v3(0.0, gravity, 0.0),
        ..SolverConfig::default()
    }
}

fn assert_vec(got: [f64; 3], want: [f64; 3], tol: f64, what: &str) {
    for k in 0..3 {
        assert!(
            (got[k] - want[k]).abs() <= tol,
            "{what}: axis {k} is {} but the closed form says {} (|Δ| = {:.3e} > {tol:.0e})",
            got[k],
            want[k],
            (got[k] - want[k]).abs()
        );
    }
}

/// One step of a zero-gravity world holding one sphere body of `radius` at `at`,
/// and the position it ends at.
fn corrected(collider: StaticCollider, at: [f64; 3], radius: f64) -> [f64; 3] {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(collider);
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(at[0], at[1], at[2]), Fix128::ONE),
        fx(radius),
    );
    w.step(fx(1.0 / 60.0));
    arr(w.get_body(b).expect("body").position)
}

const TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// A horizontal floor: a sphere 0.2 above it with radius 0.5 is lifted to 0.5.
#[test]
fn a_floor_lifts_a_penetrating_sphere_to_one_radius() {
    let floor = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    let got = corrected(StaticCollider::Plane(floor), [1.0, 0.2, -2.0], 0.5);
    assert_vec(got, [1.0, 0.5, -2.0], TOL, "floor");
}

/// A plane tilted 45°: `n = (0, 1, 1)/√2`, passing through the origin. A sphere at
/// distance `d` along the normal ends at distance `r`, moved along the normal.
#[test]
fn a_tilted_plane_pushes_along_its_normal() {
    let s = 1.0 / 2f64.sqrt();
    let plane = PlaneCollider::new(v3(0.0, 1.0, 1.0), fx(0.0));
    let at = [0.0, 0.3, 0.1];
    // d = n·p = (0.3 + 0.1)/√2.
    let d = (at[1] + at[2]) * s;
    let r = 0.5;
    let got = corrected(StaticCollider::Plane(plane), at, r);
    let push = r - d;
    assert_vec(
        got,
        [at[0], at[1] + s * push, at[2] + s * push],
        TOL,
        "tilted plane",
    );
}

/// A plane with a non-zero offset: `y = 3`.
#[test]
fn a_plane_with_an_offset_holds_the_sphere_above_it() {
    let plane = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(3.0));
    let got = corrected(StaticCollider::Plane(plane), [0.0, 3.1, 0.0], 0.5);
    assert_vec(got, [0.0, 3.5, 0.0], TOL, "offset plane");
}

/// A plane is two-sided, as the primitive documents: a sphere whose centre is
/// behind it is held on that side, `r` behind.
#[test]
fn a_plane_holds_a_sphere_on_the_side_it_is_on() {
    let plane = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    let got = corrected(StaticCollider::Plane(plane), [0.0, -0.2, 0.0], 0.5);
    assert_vec(got, [0.0, -0.5, 0.0], TOL, "sphere behind the plane");
}

/// A sphere that is clear of the plane is not touched.
#[test]
fn a_sphere_clear_of_every_collider_does_not_move() {
    let plane = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    let got = corrected(StaticCollider::Plane(plane), [0.0, 2.0, 0.0], 0.5);
    assert_eq!(got, [0.0, 2.0, 0.0]);
}

// ---------------------------------------------------------------------------
// Height field
// ---------------------------------------------------------------------------

/// A flat field at height 2.
#[test]
fn a_flat_height_field_lifts_the_sphere_to_the_height_plus_the_radius() {
    let field = HeightField::flat(8, 8, fx(1.0), v3(0.0, 0.0, 0.0), fx(2.0));
    let got = corrected(StaticCollider::HeightField(field), [3.0, 2.2, 3.0], 0.5);
    assert_vec(got, [3.0, 2.5, 3.0], TOL, "flat height field");
}

/// A planar ramp `y = s·x` (s = 1/2). The sphere ends one radius from the ramp
/// **along the ramp's normal** `(−s, 1, 0)/√(1+s²)`, i.e. `r/cosθ` above it
/// vertically — not `r` above it, which would bury it in the slope.
#[test]
fn a_height_field_ramp_pushes_along_the_slope_normal() {
    let s = 0.5;
    let mut field = HeightField::flat(12, 12, fx(1.0), v3(0.0, 0.0, 0.0), fx(0.0));
    for gz in 0..12u32 {
        for gx in 0..12u32 {
            field.set_height(gx, gz, fx(s * f64::from(gx)));
        }
    }
    let at = [5.0, s * 5.0 + 0.3, 5.0];
    let r = 0.5;
    let norm = (1.0 + s * s).sqrt();
    let n = [-s / norm, 1.0 / norm, 0.0];
    // d = n·(p − p_surface), p_surface = (5, 2.5, 5), so d = 0.3·n_y.
    let d = 0.3 * n[1];
    let push = r - d;
    let got = corrected(StaticCollider::HeightField(field), at, r);
    assert_vec(
        got,
        [
            at[0] + n[0] * push,
            at[1] + n[1] * push,
            at[2] + n[2] * push,
        ],
        1e-6,
        "ramp",
    );
}

/// Degenerate fields do not panic and do not move a body: a field with no cells,
/// and a single-cell field (no surface between grid points).
#[test]
fn a_degenerate_height_field_is_harmless() {
    let empty = HeightField::new(Vec::new(), 0, 0, fx(1.0), Vec3Fix::ZERO);
    let got = corrected(StaticCollider::HeightField(empty), [0.0, 0.1, 0.0], 0.5);
    assert_eq!(got, [0.0, 0.1, 0.0], "an empty field has no surface");
    let zero_spacing = HeightField::flat(4, 4, fx(0.0), Vec3Fix::ZERO, fx(0.0));
    let _ = corrected(
        StaticCollider::HeightField(zero_spacing),
        [0.0, 0.1, 0.0],
        0.5,
    );
}

// ---------------------------------------------------------------------------
// Triangle mesh
// ---------------------------------------------------------------------------

fn quad(a: Vec3Fix, b: Vec3Fix, c: Vec3Fix, d: Vec3Fix) -> [Triangle; 2] {
    [Triangle::new(a, b, c), Triangle::new(a, c, d)]
}

/// A 20 × 20 ground quad at `y = 1`.
#[test]
fn a_triangle_mesh_floor_lifts_the_sphere_to_the_height_plus_the_radius() {
    let tris = quad(
        v3(-10.0, 1.0, -10.0),
        v3(10.0, 1.0, -10.0),
        v3(10.0, 1.0, 10.0),
        v3(-10.0, 1.0, 10.0),
    );
    let mesh = TriMesh::from_triangles(tris.to_vec());
    let got = corrected(StaticCollider::TriMesh(mesh), [0.0, 1.2, 0.0], 0.5);
    assert_vec(got, [0.0, 1.5, 0.0], TOL, "mesh floor");
}

/// The right face of a valley `y = t·|x|` (t = 1/2): a sphere above it is pushed
/// along that face's normal `(−t, 1, 0)/√(1+t²)`.
#[test]
fn a_triangle_mesh_valley_pushes_along_the_face_normal() {
    let t = 0.5;
    let (zl, zr) = (-10.0, 10.0);
    let right = quad(
        v3(0.0, 0.0, zl),
        v3(10.0, 10.0 * t, zl),
        v3(10.0, 10.0 * t, zr),
        v3(0.0, 0.0, zr),
    );
    let left = quad(
        v3(-10.0, 10.0 * t, zl),
        v3(0.0, 0.0, zl),
        v3(0.0, 0.0, zr),
        v3(-10.0, 10.0 * t, zr),
    );
    let mut tris = right.to_vec();
    tris.extend(left);
    let mesh = TriMesh::from_triangles(tris);
    let at = [2.0, t * 2.0 + 0.3, 0.0];
    let r = 0.5;
    let norm = (1.0 + t * t).sqrt();
    let n = [-t / norm, 1.0 / norm, 0.0];
    let d = 0.3 * n[1];
    let push = r - d;
    let got = corrected(StaticCollider::TriMesh(mesh), at, r);
    assert_vec(
        got,
        [at[0] + n[0] * push, at[1] + n[1] * push, at[2]],
        1e-6,
        "valley face",
    );
}

/// An empty mesh has no surface: no panic, no movement.
#[test]
fn an_empty_triangle_mesh_is_harmless() {
    let mesh = TriMesh::from_triangles(Vec::new());
    let got = corrected(StaticCollider::TriMesh(mesh), [0.0, 0.0, 0.0], 0.5);
    assert_eq!(got, [0.0, 0.0, 0.0]);
}

// ---------------------------------------------------------------------------
// The world
// ---------------------------------------------------------------------------

/// Two orthogonal planes, a floor and a wall, resolve independently: the sphere
/// ends one radius from each.
#[test]
fn two_planes_resolve_in_turn_into_the_corner() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(1.0, 0.0, 0.0),
        fx(0.0),
    )));
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.2, 0.2, 7.0), Fix128::ONE),
        fx(0.5),
    );
    w.step(fx(1.0 / 60.0));
    assert_vec(
        arr(w.get_body(b).expect("body").position),
        [0.5, 0.5, 7.0],
        TOL,
        "corner",
    );
}

/// Dropped on a floor under gravity, a body comes to rest one radius above it.
#[test]
fn a_body_dropped_on_a_floor_rests_one_radius_above_it() {
    let mut w = PhysicsWorld::new(config(-10.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 3.0, 0.0), Fix128::ONE),
        fx(0.5),
    );
    for _ in 0..300 {
        w.step(fx(1.0 / 60.0));
    }
    let y = w.get_body(b).expect("body").position.y.to_f64();
    assert!((y - 0.5).abs() < 1e-6, "rests at {y}, one radius is 0.5");
}

/// A body with no collision radius of its own uses the world's default one
/// (`set_sdf_collision_radius`), as it does for SDF colliders.
#[test]
fn a_body_without_a_radius_uses_the_default_collision_radius() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.set_sdf_collision_radius(fx(1.25));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    let b = w.add_body(RigidBody::new_dynamic(v3(0.0, 0.5, 0.0), Fix128::ONE));
    w.step(fx(1.0 / 60.0));
    assert_vec(
        arr(w.get_body(b).expect("body").position),
        [0.0, 1.25, 0.0],
        TOL,
        "default radius",
    );
}

/// Static bodies and sensors are not pushed.
#[test]
fn static_bodies_and_sensors_are_not_moved() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    let s = w.add_body_with_radius(RigidBody::new_static(v3(0.0, 0.1, 0.0)), fx(0.5));
    let q = w.add_body_with_radius(RigidBody::new_sensor(v3(5.0, 0.1, 0.0)), fx(0.5));
    w.step(fx(1.0 / 60.0));
    assert_eq!(
        arr(w.get_body(s).expect("static").position),
        [0.0, 0.1, 0.0]
    );
    assert_eq!(
        arr(w.get_body(q).expect("sensor").position),
        [5.0, 0.1, 0.0]
    );
}

/// Colliders can be removed again; a removed collider no longer acts, and the
/// removal reports what it removed or `None` for a bad index.
#[test]
fn a_removed_collider_no_longer_acts() {
    let mut w = PhysicsWorld::new(config(0.0));
    assert_eq!(w.static_collider_count(), 0);
    let i = w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    assert_eq!((i, w.static_collider_count()), (0, 1));
    assert!(
        w.remove_static_collider(5).is_none(),
        "a bad index removes nothing"
    );
    assert_eq!(w.static_collider_count(), 1);
    assert!(matches!(
        w.remove_static_collider(i),
        Some(StaticCollider::Plane(_))
    ));
    assert_eq!(w.static_collider_count(), 0);
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.1, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.step(fx(1.0 / 60.0));
    assert_vec(
        arr(w.get_body(b).expect("body").position),
        [0.0, 0.1, 0.0],
        TOL,
        "after removal",
    );
}

/// `reset_world` returns the world to its construction state, colliders included.
#[test]
fn reset_world_clears_the_static_colliders() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    w.reset_world();
    assert_eq!(w.static_collider_count(), 0);
}

// ===========================================================================
// The primitives' own queries, against closed forms
// ===========================================================================

use alice_physics::collider::AABB;

/// `from_point_normal` normalises the normal and sets `offset = n·p`; the signed
/// distance, projection, side test and `flip` follow.
#[test]
fn a_plane_from_a_point_and_a_normal_has_the_hessian_form() {
    // normal (0, 3, 4)/5 = (0, 0.6, 0.8); point (1, 2, 3): offset 0.6·2 + 0.8·3 = 3.6.
    let plane = PlaneCollider::from_point_normal(v3(1.0, 2.0, 3.0), v3(0.0, 3.0, 4.0));
    assert_vec(arr(plane.normal), [0.0, 0.6, 0.8], 1e-12, "normal");
    assert!((plane.offset.to_f64() - 3.6).abs() < 1e-12);
    let p = v3(0.0, 5.0, 0.0);
    // distance = 0.6·5 − 3.6 = −0.6, so the point is behind the plane.
    assert!((plane.distance_to_point(p).to_f64() + 0.6).abs() < 1e-12);
    assert!(!plane.is_front(p));
    assert!(plane.is_front(v3(0.0, 9.0, 0.0)), "0.6·9 − 3.6 = 1.8 > 0");
    // The projection lies on the plane: its distance is 0 and it differs from the
    // point by (−d)·n = 0.6·n.
    let q = plane.project_point(p);
    assert!(plane.distance_to_point(q).to_f64().abs() < 1e-12);
    assert_vec(arr(q), [0.0, 5.0 + 0.36, 0.48], 1e-12, "projection");
    // Flipping reverses the sides and keeps the surface.
    let flipped = plane.flip();
    assert!((flipped.distance_to_point(p).to_f64() - 0.6).abs() < 1e-12);
    assert!(flipped.is_front(p));
    assert!(flipped.distance_to_point(q).to_f64().abs() < 1e-12);
    // A zero normal falls back to Y-up, as documented.
    let fallback = PlaneCollider::from_point_normal(v3(0.0, 2.0, 0.0), Vec3Fix::ZERO);
    assert_vec(
        arr(fallback.normal),
        [0.0, 1.0, 0.0],
        0.0,
        "fallback normal",
    );
}

/// A box against the floor `y = 0`: its lowest corner is below by `−y_min`, which is
/// the depth; entirely above is no hit.
#[test]
fn a_plane_against_a_box_reports_the_deepest_corner() {
    let floor = PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0));
    let partly = AABB::new(v3(-1.0, -0.5, -1.0), v3(1.0, 1.0, 1.0));
    let hit = floor.intersect_aabb(&partly);
    assert!(hit.colliding);
    assert!(
        (hit.depth.to_f64() - 0.5).abs() < 1e-12,
        "depth {}",
        hit.depth.to_f64()
    );
    assert_vec(arr(hit.normal), [0.0, 1.0, 0.0], 1e-12, "normal");
    let above = AABB::new(v3(-1.0, 0.25, -1.0), v3(1.0, 2.0, 1.0));
    assert!(!floor.intersect_aabb(&above).colliding);
    let below = AABB::new(v3(-1.0, -3.0, -1.0), v3(1.0, -1.0, 1.0));
    let sunk = floor.intersect_aabb(&below);
    assert!(sunk.colliding && (sunk.depth.to_f64() - 3.0).abs() < 1e-12);
}

/// Bilinear interpolation on one cell: h00 = 0, h10 = 1, h01 = 2, h11 = 4.
#[test]
fn a_height_field_interpolates_bilinearly_and_clamps_outside_its_grid() {
    let mut f = HeightField::flat(2, 2, fx(1.0), Vec3Fix::ZERO, fx(0.0));
    f.set_height(1, 0, fx(1.0));
    f.set_height(0, 1, fx(2.0));
    f.set_height(1, 1, fx(4.0));
    assert_eq!(f.get_height(1, 1).to_f64(), 4.0);
    assert_eq!(f.get_height(9, 9).to_f64(), 4.0, "grid indices clamp");
    // (u, v) = (0.25, 0.5): 0.375·0 + 0.125·1 + 0.375·2 + 0.125·4 = 1.375.
    assert!((f.sample_height(fx(0.25), fx(0.5)).to_f64() - 1.375).abs() < 1e-12);
    // Out of range: set_height ignores it, and sampling past the edge holds the edge.
    f.set_height(5, 5, fx(100.0));
    assert!(f.sample_height(fx(0.25), fx(0.5)).to_f64() < 5.0);
    assert!((f.sample_height(fx(3.0), fx(3.0)).to_f64() - 4.0).abs() < 1e-12);
}

/// The normal of the ramp `y = s·x` is `(−s, 1, 0)/√(1+s²)` and the signed distance
/// is the vertical offset from the surface.
#[test]
fn a_height_field_ramp_has_the_slope_normal_and_a_vertical_signed_distance() {
    let s = 0.5;
    let mut f = HeightField::flat(8, 8, fx(1.0), Vec3Fix::ZERO, fx(0.0));
    for gz in 0..8u32 {
        for gx in 0..8u32 {
            f.set_height(gx, gz, fx(s * f64::from(gx)));
        }
    }
    let norm = (1.0 + s * s).sqrt();
    assert_vec(
        arr(f.sample_normal(fx(3.0), fx(3.0))),
        [-s / norm, 1.0 / norm, 0.0],
        1e-9,
        "ramp normal",
    );
    assert!((f.signed_distance(v3(3.0, 1.5 + 0.25, 3.0)).to_f64() - 0.25).abs() < 1e-12);
    assert!((f.signed_distance(v3(3.0, 1.5 - 0.5, 3.0)).to_f64() + 0.5).abs() < 1e-12);
}

/// A capsule lying along `x` over a floor at `y = 1`, radius 0.5, its axis at 1.3: the
/// penetration is `0.5 − 0.3`, along the floor's normal.
#[test]
fn a_triangle_mesh_reports_the_capsule_and_box_penetration_of_a_floor() {
    let tris = quad(
        v3(-10.0, 1.0, -10.0),
        v3(10.0, 1.0, -10.0),
        v3(10.0, 1.0, 10.0),
        v3(-10.0, 1.0, 10.0),
    );
    let mesh = TriMesh::from_triangles(tris.to_vec());
    assert_eq!(mesh.triangle_count(), 2);
    let hit = mesh
        .collide_capsule(v3(-1.0, 1.3, 0.0), v3(1.0, 1.3, 0.0), fx(0.5))
        .expect("the capsule penetrates");
    assert!(
        (hit.depth.to_f64() - 0.2).abs() < 1e-9,
        "capsule depth {}",
        hit.depth.to_f64()
    );
    assert_vec(arr(hit.normal), [0.0, 1.0, 0.0], 1e-9, "capsule normal");
    assert!(mesh
        .collide_capsule(v3(-1.0, 1.6, 0.0), v3(1.0, 1.6, 0.0), fx(0.5))
        .is_none());
    // A box of half-extent 1 whose centre is at height 1.5: its bottom is at 0.5, 0.5
    // below the floor, so the depth is 1 − 0.5 = 0.5 along +y.
    let b = AABB::new(v3(-1.0, 0.5, -1.0), v3(1.0, 2.5, 1.0));
    let hit = mesh.collide_aabb(&b).expect("the box penetrates");
    assert!(
        (hit.depth.to_f64() - 0.5).abs() < 1e-9,
        "box depth {}",
        hit.depth.to_f64()
    );
    assert_vec(arr(hit.normal), [0.0, 1.0, 0.0], 1e-9, "box normal");
    let clear = AABB::new(v3(-1.0, 1.5, -1.0), v3(1.0, 3.5, 1.0));
    assert!(mesh.collide_aabb(&clear).is_none());
}

/// The same correction through the GPU-bridge step: static colliders are resolved
/// on the CPU in `step_with_bridge` exactly as in `step`. (The bridge is a no-op
/// that leaves every buffer alone, so the position is the collision correction.)
#[cfg(feature = "gpu-solver-bridge")]
mod bridged {
    use super::*;
    use alice_physics::gpu_bridge::{DiffFixture, GpuDivergence, GpuSolverBridge};
    use alice_physics::solver::ContactConstraint;

    struct NoOpBridge;

    impl GpuSolverBridge for NoOpBridge {
        fn send_island(&mut self, _p: &[[Fix128; 3]], _v: &[[Fix128; 3]]) {}
        fn dispatch_iterations(&mut self, _iters: u32, _dt: Fix128) {}
        fn recv_island(&self, _p: &mut [[Fix128; 3]], _v: &mut [[Fix128; 3]]) {}
        fn assert_bit_exact_vs_cpu(&self, _fixture: &DiffFixture) -> Result<(), GpuDivergence> {
            Ok(())
        }
        fn send_contact_constraints(&mut self, _c: &[ContactConstraint]) {}
        fn send_body_state(&mut self, _p: &[[Fix128; 3]], _i: &[Fix128]) {}
        fn dispatch_contact_solve_iteration(&mut self, _warm_start_factor: Fix128) {}
        fn recv_contact_constraints(&self, _c: &mut [ContactConstraint]) {}
        fn recv_body_positions(&self, _p: &mut [[Fix128; 3]]) {}
    }

    #[test]
    fn step_with_bridge_resolves_static_colliders_like_step() {
        let mut w = PhysicsWorld::new(config(0.0));
        w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
            v3(0.0, 1.0, 0.0),
            fx(0.0),
        )));
        let b = w.add_body_with_radius(
            RigidBody::new_dynamic(v3(0.0, 0.2, 0.0), Fix128::ONE),
            fx(0.5),
        );
        w.step_with_bridge(&mut NoOpBridge, fx(1.0 / 60.0));
        assert_vec(
            arr(w.get_body(b).expect("body").position),
            [0.0, 0.5, 0.0],
            TOL,
            "bridged step",
        );
    }
}

/// A body's own collision radius is the sphere tested, not the world's default
/// (0.5): radius 0.75 on a floor ends at 0.75.
#[test]
fn a_body_s_own_collision_radius_is_the_sphere_tested() {
    let got = corrected(
        StaticCollider::Plane(PlaneCollider::new(v3(0.0, 1.0, 0.0), fx(0.0))),
        [0.0, 0.1, 0.0],
        0.75,
    );
    assert_vec(got, [0.0, 0.75, 0.0], TOL, "own radius");
}

/// Removing at exactly `count` is out of range: nothing removed, nothing panics.
#[test]
fn removing_at_the_first_unused_index_removes_nothing() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    assert!(w.remove_static_collider(1).is_none());
    assert_eq!(w.static_collider_count(), 1);
}

/// A field with no grid points answers zero to every height query instead of
/// underflowing its own index arithmetic.
#[test]
fn an_empty_height_field_answers_zero_to_height_queries() {
    let empty = HeightField::new(Vec::new(), 0, 0, fx(1.0), Vec3Fix::ZERO);
    assert_eq!(empty.get_height(0, 0).to_f64(), 0.0);
    assert_eq!(empty.get_height(7, 3).to_f64(), 0.0);
    assert_eq!(empty.sample_height(fx(1.0), fx(1.0)).to_f64(), 0.0);
    assert!(empty.collide_sphere(v3(0.0, 0.1, 0.0), fx(0.5)).is_none());
}

/// The graph-coloured parallel step resolves static colliders too.
#[cfg(feature = "parallel")]
#[test]
fn step_parallel_resolves_static_colliders_like_step() {
    let mut w = PhysicsWorld::new(config(0.0));
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        v3(0.0, 1.0, 0.0),
        fx(0.0),
    )));
    let b = w.add_body_with_radius(
        RigidBody::new_dynamic(v3(0.0, 0.2, 0.0), Fix128::ONE),
        fx(0.5),
    );
    w.step_parallel(fx(1.0 / 60.0));
    assert_vec(
        arr(w.get_body(b).expect("body").position),
        [0.0, 0.5, 0.0],
        TOL,
        "parallel step",
    );
}
