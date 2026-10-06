//! Oracles for the boundary rules of world shape queries and the character
//! controller: touching is not overlapping for every kind of geometry (a point on
//! a surface with radius 0, a sphere or box exactly touching), huge direction
//! components, and the configurations the controller refuses or adjusts.
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
use alice_physics::collider::AABB;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;

const EXACT: f64 = 1e-12;
const MOVE_TOL: f64 = 1e-8;
const R: f64 = 0.3;
const SKIN: f64 = 0.01;

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

fn overlap_s(w: &PhysicsWorld, c: [f64; 3], r: f64) -> Vec<RayTarget> {
    w.overlap_sphere(p3(c), fx(r), &RayFilter::default())
}

fn overlap_b(w: &PhysicsWorld, lo: [f64; 3], hi: [f64; 3]) -> Vec<RayTarget> {
    w.overlap_aabb(&AABB::new(p3(lo), p3(hi)), &RayFilter::default())
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

/// One twisted cell, heights `1, −1, −1, 1` at `(0,0), (2,0), (0,2), (2,2)`:
/// with `u = x/2`, `v = z/2` the surface is `y = (1 − 2u)(1 − 2v) = (1 − x)(1 − z)`,
/// a saddle through `(1, 0, 1)` with principal curvatures `±1` there.
fn saddle_world() -> (PhysicsWorld, usize) {
    let mut w = world();
    let h = w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        vec![fx(1.0), fx(-1.0), fx(-1.0), fx(1.0)],
        2,
        2,
        fx(2.0),
        Vec3Fix::ZERO,
    )));
    (w, h)
}

#[test]
fn touching_is_not_overlapping_for_every_kind_of_geometry() {
    // oracle: a point on the surface of a solid (radius 0) or a sphere exactly
    // touching it is not inside: the module rule "touching is not overlapping".
    let mut w = world();
    unit_box(&mut w);
    assert!(overlap_s(&w, [1.0, 0.0, 0.0], 0.0).is_empty(), "box face");
    assert!(overlap_s(&w, [1.0, 1.0, 1.0], 0.0).is_empty(), "box corner");
    assert!(
        overlap_s(&w, [1.5, 0.0, 0.0], 0.5).is_empty(),
        "sphere touching a box face"
    );
    assert!(
        overlap_b(&w, [1.0, -0.5, -0.5], [2.0, 0.5, 0.5]).is_empty(),
        "box touching a box face"
    );
    assert!(
        overlap_b(&w, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]).is_empty(),
        "box touching a box corner"
    );
    assert_eq!(
        overlap_b(&w, [0.75, -0.5, -0.5], [2.0, 0.5, 0.5]).len(),
        1,
        "a box reaching in"
    );

    let mut w = world();
    shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
    );
    assert!(
        overlap_s(&w, [1.0, 0.0, 0.0], 0.0).is_empty(),
        "cylinder side"
    );
    assert!(
        overlap_s(&w, [0.0, 1.0, 0.0], 0.0).is_empty(),
        "cylinder cap"
    );

    let mut w = world();
    w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), fx(1.0));
    assert!(
        overlap_s(&w, [1.0, 0.0, 0.0], 0.0).is_empty(),
        "sphere surface"
    );
    assert!(
        overlap_b(&w, [1.0, -0.5, -0.5], [2.0, 0.5, 0.5]).is_empty(),
        "box touching a sphere"
    );

    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    assert!(
        overlap_s(&w, [1.0, 0.0, 0.0], 0.0).is_empty(),
        "point on a plane"
    );
    assert!(
        overlap_b(&w, [-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]).is_empty(),
        "box resting on a plane"
    );
    assert!(
        overlap_b(&w, [-1.0, -1.0, -1.0], [1.0, 0.0, 1.0]).is_empty(),
        "box under a plane"
    );

    let mut w = world();
    let verts = [v3(0.0, 0.0, 0.0), v3(4.0, 0.0, 0.0), v3(0.0, 0.0, 4.0)];
    w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 1, 2],
    )));
    assert!(
        overlap_s(&w, [1.0, 0.0, 1.0], 0.0).is_empty(),
        "point on a triangle"
    );

    let mut w = world();
    shaped(
        &mut w,
        Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Vec3Fix::ZERO,
    );
    assert!(
        overlap_s(&w, [2.5, 0.0, 0.0], 0.0).is_empty(),
        "torus surface"
    );

    let (w, _) = saddle_world();
    assert!(
        overlap_s(&w, [1.0, 0.0, 1.0], 0.0).is_empty(),
        "point on a height field"
    );
    // oracle: the saddle over x, z ∈ [0.9, 1.1] stays within |y| ≤ 0.01: a box
    // above y = 0.05 or below y = −0.05 does not meet it.
    assert!(
        overlap_b(&w, [0.9, 0.05, 0.9], [1.1, 0.2, 1.1]).is_empty(),
        "box above"
    );
    assert!(
        overlap_b(&w, [0.9, -0.2, 0.9], [1.1, -0.05, 1.1]).is_empty(),
        "box below"
    );
    assert_eq!(
        overlap_b(&w, [0.9, -0.05, 0.9], [1.1, 0.05, 1.1]).len(),
        1,
        "box across"
    );
}

#[test]
fn huge_direction_components_are_normalised() {
    let mut w = world();
    w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), fx(1.0));
    let huge = Vec3Fix::new(Fix128::from_int(i64::MAX / 2), Fix128::ZERO, Fix128::ZERO);
    // oracle: the direction is +X: radius 1 + 0.5 from (−10, 0, 0), t = 8.5.
    let h = w
        .cast_sphere(
            v3(-10.0, 0.0, 0.0),
            fx(0.5),
            huge,
            fx(100.0),
            &RayFilter::default(),
        )
        .expect("hit");
    assert!((h.t.to_f64() - 8.5).abs() < EXACT, "t = {}", h.t.to_f64());
    let h = w
        .cast_capsule(
            v3(-10.0, -0.25, 0.0),
            v3(-10.0, 0.25, 0.0),
            fx(0.5),
            huge,
            fx(100.0),
            &RayFilter::default(),
        )
        .expect("hit");
    assert!((h.t.to_f64() - 8.5).abs() < EXACT, "t = {}", h.t.to_f64());
    // oracle: radius 0 is cast_ray: the unit sphere at t = 9.
    let h = w
        .cast_sphere(
            v3(-10.0, 0.0, 0.0),
            Fix128::ZERO,
            huge,
            fx(100.0),
            &RayFilter::default(),
        )
        .expect("hit");
    assert!((h.t.to_f64() - 9.0).abs() < EXACT, "t = {}", h.t.to_f64());
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
fn an_invalid_configuration_does_not_move() {
    let mut w = world();
    static_box(&mut w, [3.0, 2.0, 0.0], [1.0, 2.0, 5.0]);
    for config in [
        CharacterConfig {
            radius: fx(-0.3),
            ..CharacterConfig::default()
        },
        CharacterConfig {
            skin_width: fx(-0.05),
            ..CharacterConfig::default()
        },
        CharacterConfig {
            height: fx(-1.0),
            ..CharacterConfig::default()
        },
    ] {
        // oracle: a negative radius, skin width or height describes no capsule:
        // the move is refused and the controller keeps its position.
        let mut c = ctrl([0.0, 1.0, 0.0], config);
        let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
        assert_pos(res.position, [0.0, 1.0, 0.0], EXACT);
        assert_pos(c.position, [0.0, 1.0, 0.0], EXACT);
        assert!(!res.grounded);
    }
}

#[test]
fn zero_slides_is_one_slide() {
    let mut w = world();
    static_box(&mut w, [3.0, 2.0, 0.0], [1.0, 2.0, 5.0]);
    let config = CharacterConfig {
        max_slides: 0,
        ..CharacterConfig::default()
    };
    // oracle: one sweep, no slide: D = (3, 0, 1)/√10 · |D| stops a skin width from
    // the face x = 2 along its normal: travel = (1.7 − s)·√10/3, so the centre is
    // at (1.69, 1, 1.69/3).
    let mut c = ctrl([0.0, 1.0, 0.0], config);
    let res = w.move_character(&mut c, v3(3.0, 0.0, 1.0));
    assert_pos(
        res.position,
        [2.0 - R - SKIN, 1.0, (2.0 - R - SKIN) / 3.0],
        MOVE_TOL,
    );
    // oracle: a free move with zero slides goes the whole way.
    let mut c = ctrl([-5.0, 1.0, 0.0], config);
    let res = w.move_character(&mut c, v3(-1.0, 0.0, 0.0));
    assert_pos(res.position, [-6.0, 1.0, 0.0], EXACT);
}
