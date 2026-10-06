//! Oracles for world casts that start exactly touching a collider: a contact at
//! `t = 0` is reported only when the motion goes into the surface (`d·n < 0`);
//! moving away from it or along it is not a hit, and the cast goes on to what
//! lies further along the path. The character controller relies on this to
//! walk on a surface it touches and to slide with a zero skin width.
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion. Nothing here calls the code under test to
//! make an expected value.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::character::{CharacterConfig, CharacterController};
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::world_shape_query::WorldShapeHit;

const EXACT: f64 = 1e-12;
const ITER: f64 = 1e-9;
const MOVE_TOL: f64 = 1e-8;
const R: f64 = 0.3;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn p3(p: [f64; 3]) -> Vec3Fix {
    v3(p[0], p[1], p[2])
}

fn f3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn world() -> PhysicsWorld {
    PhysicsWorld::new(PhysicsConfig::default())
}

fn sphere_cast(
    w: &PhysicsWorld,
    c: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_sphere(p3(c), fx(r), p3(d), fx(max), &RayFilter::default())
}

fn capsule_cast(
    w: &PhysicsWorld,
    a: [f64; 3],
    b: [f64; 3],
    r: f64,
    d: [f64; 3],
    max: f64,
) -> Option<WorldShapeHit> {
    w.cast_capsule(p3(a), p3(b), fx(r), p3(d), fx(max), &RayFilter::default())
}

#[track_caller]
fn assert_vec(got: Vec3Fix, want: [f64; 3], tol: f64, what: &str) {
    let g = f3(got);
    for k in 0..3 {
        assert!(
            (g[k] - want[k]).abs() < tol,
            "{what} {g:?} but the closed form is {want:?}"
        );
    }
}

#[track_caller]
fn assert_t(hit: Option<WorldShapeHit>, t: f64, tol: f64) -> WorldShapeHit {
    let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
    assert!(
        (h.t.to_f64() - t).abs() < tol,
        "t = {} but the closed form is {t}",
        h.t.to_f64()
    );
    h
}

fn shaped(w: &mut PhysicsWorld, shape: Shape, pos: Vec3Fix) -> usize {
    w.add_shaped_body(&shape, Fix128::ONE, pos)
        .expect("valid shape")
}

fn unit_box(w: &mut PhysicsWorld) -> usize {
    shaped(
        w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
    )
}

/// One triangle mesh: a floor `y = 0` over `x, z ∈ [−20, 20]` and a wall `x = 12`
/// (`y ∈ [0, 5]`) facing −X.
fn room() -> PhysicsWorld {
    let mut w = world();
    let verts = [
        v3(-20.0, 0.0, -20.0),
        v3(20.0, 0.0, -20.0),
        v3(20.0, 0.0, 20.0),
        v3(-20.0, 0.0, 20.0),
        v3(12.0, 0.0, -20.0),
        v3(12.0, 0.0, 20.0),
        v3(12.0, 5.0, 20.0),
        v3(12.0, 5.0, -20.0),
    ];
    w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7],
    )));
    w
}

/// A 21 × 5 field of spacing 1: height 0 for `x ≤ 12`, 5 for `x ≥ 13`: between
/// them the plane `y = 5(x − 12)`.
fn ramp_world() -> PhysicsWorld {
    let mut heights = vec![];
    for _z in 0..5 {
        for x in 0..21 {
            heights.push(if x >= 13 { fx(5.0) } else { Fix128::ZERO });
        }
    }
    let mut w = world();
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        heights,
        21,
        5,
        fx(1.0),
        Vec3Fix::ZERO,
    )));
    w
}

#[test]
fn a_cast_that_starts_touching_and_moves_away_or_along_does_not_hit() {
    let mut w = world();
    unit_box(&mut w);
    // oracle: the sphere of radius 0.5 at (1.5, 0, 0) touches the face x = 1;
    // moving +X (away) or along the face its distance never drops below 0.5.
    assert!(sphere_cast(&w, [1.5, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).is_none());
    assert!(sphere_cast(&w, [0.0, 1.5, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).is_none());
    assert!(sphere_cast(&w, [0.0, 1.5, 0.0], 0.5, [0.0, 0.0, -1.0], 100.0).is_none());
    // oracle: the capsule (−0.5..0.5, 1.5, 0) of radius 0.5 touches the top face;
    // moving up or along it never comes nearer.
    let (a, b) = ([-0.5, 1.5, 0.0], [0.5, 1.5, 0.0]);
    assert!(capsule_cast(&w, a, b, 0.5, [0.0, 1.0, 0.0], 100.0).is_none());
    assert!(capsule_cast(&w, a, b, 0.5, [1.0, 0.0, 0.0], 100.0).is_none());
    assert!(capsule_cast(&w, a, b, 0.5, [0.0, 0.0, 1.0], 100.0).is_none());

    // oracle: the same for a cylinder side (convex): touching at (1.5, 0, 0),
    // moving out or along the tangent.
    let mut w = world();
    shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
    );
    assert!(sphere_cast(&w, [1.5, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).is_none());
    assert!(sphere_cast(&w, [1.5, 0.0, 0.0], 0.5, [0.0, 0.0, 1.0], 100.0).is_none());
    assert!(capsule_cast(
        &w,
        [1.5, -0.5, 0.0],
        [1.5, 0.5, 0.0],
        0.5,
        [0.0, 0.0, 1.0],
        100.0
    )
    .is_none());

    // oracle: a flat height field, the sphere resting on it moving along it.
    let mut w = world();
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        vec![Fix128::ZERO; 16],
        4,
        4,
        fx(1.0),
        Vec3Fix::ZERO,
    )));
    assert!(sphere_cast(&w, [0.5, 0.25, 1.5], 0.25, [1.0, 0.0, 0.0], 2.0).is_none());
    assert!(capsule_cast(
        &w,
        [0.5, 0.25, 1.0],
        [0.5, 0.25, 2.0],
        0.25,
        [1.0, 0.0, 0.0],
        2.0
    )
    .is_none());
}

#[test]
fn a_cast_that_starts_touching_and_moves_in_hits_at_zero() {
    let mut w = world();
    let b = unit_box(&mut w);
    // oracle: touching the face x = 1 and moving −X: contact at once, t = 0,
    // point (1, 0, 0), normal +X (the face, not −direction: nothing overlaps).
    let h = sphere_cast(&w, [1.5, 0.0, 0.0], 0.5, [-1.0, 0.0, 0.0], 100.0).expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert_eq!(h.t, Fix128::ZERO);
    assert_vec(h.normal, [1.0, 0.0, 0.0], EXACT, "normal");
    assert_vec(h.point, [1.0, 0.0, 0.0], EXACT, "point");
    // oracle: touching the top face and moving down and sideways: t = 0, +Y.
    let h = capsule_cast(
        &w,
        [-0.5, 1.5, 0.0],
        [0.5, 1.5, 0.0],
        0.5,
        [1.0, -1.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert!(h.t.to_f64().abs() < ITER, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [0.0, 1.0, 0.0], ITER, "normal");
}

#[test]
fn a_cast_resting_on_a_surface_reaches_the_wall_of_the_same_collider() {
    let r = 0.3;
    let w = room();
    // oracle: resting on the floor (gap 0) and moving along it, the first contact
    // is the wall x = 12: t = 12 − r, normal −X.
    let h = assert_t(
        sphere_cast(&w, [0.0, r, 0.0], r, [1.0, 0.0, 0.0], 20.0),
        12.0 - r,
        ITER,
    );
    assert_vec(h.normal, [-1.0, 0.0, 0.0], ITER, "sphere normal");
    let h = assert_t(
        capsule_cast(
            &w,
            [0.0, r, 0.0],
            [0.0, r + 1.2, 0.0],
            r,
            [1.0, 0.0, 0.0],
            20.0,
        ),
        12.0 - r,
        ITER,
    );
    assert_vec(h.normal, [-1.0, 0.0, 0.0], ITER, "capsule normal");
    // oracle: resting on the flat part of the ramp field, the ramp
    // y = 5(x − 12) is met at x = 12 + (r − r√26)/5.
    let w = ramp_world();
    let x = 12.0 + (r - r * 26f64.sqrt()) / 5.0;
    assert_t(
        sphere_cast(&w, [0.0, r, 2.0], r, [1.0, 0.0, 0.0], 20.0),
        x,
        ITER,
    );
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

fn ctrl(p: [f64; 3], config: CharacterConfig) -> CharacterController {
    CharacterController::new(v3(p[0], p[1], p[2]), config)
}

#[test]
fn exactly_touching_a_box_top_walks_and_jumps() {
    let mut w = world();
    static_box(&mut w, [0.0, 0.5, 0.0], [5.0, 0.5, 5.0]);
    // oracle: the capsule bottom touches the top y = 1 (centre 1 + h/2 = 1.9);
    // touching is not overlapping, so it is not pushed, walks the whole
    // horizontal move and is grounded (the probe moves into the top at t = 0).
    let config = CharacterConfig::default();
    // 1 + h/2 in Fix128: the lower end 1 + r is exactly r above the top
    let y = Fix128::ONE + config.height.half();
    let start = Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO);
    let mut c = CharacterController::new(start, config);
    let res = w.move_character(&mut c, v3(1.0, 0.0, 0.0));
    assert_pos(res.position, [1.0, 1.9, 0.0], MOVE_TOL);
    assert!(res.grounded, "standing on the top");
    let mut c = CharacterController::new(start, config);
    let res = w.move_character(&mut c, v3(0.0, 1.0, 0.0));
    assert_pos(res.position, [0.0, 2.9, 0.0], MOVE_TOL);
}

#[test]
fn a_zero_skin_width_slides_along_a_wall() {
    let mut w = world();
    static_box(&mut w, [3.0, 2.0, 0.0], [1.0, 2.0, 5.0]);
    let config = CharacterConfig {
        skin_width: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    let mut c = ctrl([0.0, 1.0, 0.0], config);
    // oracle: D = (3, 0, 1) meets the face x = 2 when the centre is at x = 1.7
    // (z = 1.7/3); the rest (1.3, 0, 1 − 1.7/3) projected on the face keeps its z:
    // the capsule, touching the wall, slides on to z = 1.
    let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
    assert_pos(res.position, [2.0 - R, 1.0, 1.0], MOVE_TOL);
    // oracle: a move along the wall from touching it: the whole move.
    let res = w.move_character(&mut c, v3(0.0, 0.0, 1.0));
    assert_pos(res.position, [2.0 - R, 1.0, 2.0], MOVE_TOL);
}

#[test]
fn resting_on_a_floor_of_many_pieces_crosses_their_shared_edges() {
    // a floor of two triangles sharing the diagonal (−5, 0, −5)–(5, 0, 5)
    let mut w = world();
    let verts = [
        v3(-5.0, 0.0, -5.0),
        v3(5.0, 0.0, -5.0),
        v3(5.0, 0.0, 5.0),
        v3(-5.0, 0.0, 5.0),
    ];
    w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 2, 1, 0, 3, 2],
    )));
    // oracle: resting on the floor (gap 0) and moving along it across the
    // diagonal, the distance to the floor stays r: no hit.
    assert!(sphere_cast(&w, [-2.0, 0.25, 1.0], 0.25, [1.0, 0.0, 0.0], 4.0).is_none());
    assert!(capsule_cast(
        &w,
        [-2.0, 0.25, 1.0],
        [-2.0, 1.25, 1.0],
        0.25,
        [1.0, 0.0, 0.0],
        4.0
    )
    .is_none());
    // oracle: the same move drifting down by 1e-6 per unit goes into the floor
    // at once: a hit at t = 0 with the floor normal.
    let h = sphere_cast(&w, [-2.0, 0.25, 1.0], 0.25, [1.0, -1e-6, 0.0], 4.0).expect("hit");
    assert!(h.t.to_f64().abs() < ITER, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [0.0, 1.0, 0.0], ITER, "normal");
    let h = capsule_cast(
        &w,
        [-2.0, 0.25, 1.0],
        [-2.0, 1.25, 1.0],
        0.25,
        [1.0, -1e-6, 0.0],
        4.0,
    )
    .expect("hit");
    assert!(h.t.to_f64().abs() < ITER, "t = {}", h.t.to_f64());
    // oracle: a controller with a zero skin width walks across the diagonal: the
    // whole move, at the height it rests at (h/2 above the floor).
    let config = CharacterConfig {
        skin_width: Fix128::ZERO,
        ..CharacterConfig::default()
    };
    let y = config.height.half();
    let mut c = CharacterController::new(Vec3Fix::new(fx(-2.0), y, fx(1.0)), config);
    let res = w.move_character(&mut c, v3(3.0, 0.0, 0.0));
    assert_pos(res.position, [1.0, 0.9, 1.0], MOVE_TOL);
    assert!(res.grounded);
}
