//! Oracles for world casts whose path runs along one part of a collider before
//! it meets another part of the same collider (a mesh floor and wall, a flat
//! height field and a ramp), and for grazing approaches. Conservative
//! advancement along a surface at gap `g` moves `g` per step; these casts need
//! far more steps than any fixed budget, and must still report the contact.
//!
//! Grazing approaches are ill-conditioned (`dt = dgap / (d·n)`): `t` is checked to
//! `1e-6` and never later than the closed form.
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
use alice_physics::shape_raycast::RayFilter;
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::world_shape_query::WorldShapeHit;

const ITER: f64 = 1e-9;
const MOVE_TOL: f64 = 1e-8;
const STAND: f64 = 0.91;
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

#[test]
fn a_cast_along_a_mesh_floor_reaches_the_wall_of_the_same_mesh() {
    let w = room();
    let r = 0.3;
    for gap in [0.5, 0.01, 0.0001] {
        let y = r + gap;
        // oracle: the floor is parallel to the motion at distance r + gap ≥ r,
        // so the first contact is the wall x = 12: the centre stops at x = 12 − r,
        // t = 11.7, normal −X.
        let h = assert_t(
            sphere_cast(&w, [0.0, y, 0.0], r, [1.0, 0.0, 0.0], 20.0),
            12.0 - r,
            ITER,
        );
        assert_vec(h.normal, [-1.0, 0.0, 0.0], ITER, "sphere normal");
        let h = assert_t(
            capsule_cast(
                &w,
                [0.0, y, 0.0],
                [0.0, y + 1.2, 0.0],
                r,
                [1.0, 0.0, 0.0],
                20.0,
            ),
            12.0 - r,
            ITER,
        );
        assert_vec(h.normal, [-1.0, 0.0, 0.0], ITER, "capsule normal");
    }
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
fn a_cast_along_a_height_field_reaches_the_ramp() {
    let w = ramp_world();
    let r = 0.3;
    for gap in [0.1, 0.01, 0.0001] {
        let c = r + gap;
        // oracle: the centre at height c moving +X meets the offset of the ramp
        // plane 5x − y − 60 = 0 when (5(x − 12) − c)/√26 = −r, i.e.
        // x = 12 + (c − r√26)/5; the contact (x + 5r/√26, c − r/√26) lies on the
        // ramp (12.05, 0.25 for gap 0.01), normal (−5, 1)/√26.
        let x = 12.0 + (c - r * 26f64.sqrt()) / 5.0;
        let h = assert_t(
            sphere_cast(&w, [0.0, c, 2.0], r, [1.0, 0.0, 0.0], 20.0),
            x,
            ITER,
        );
        let s = 26f64.sqrt();
        assert_vec(h.normal, [-5.0 / s, 1.0 / s, 0.0], 1e-6, "normal");
    }
}

#[test]
fn grazing_sphere_cast_onto_an_ellipsoid_still_hits() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Ellipsoid {
            radii: v3(2.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
    );
    let r = 0.5;
    for eps in [1e-2, 1e-4, 1e-6] {
        let c = 1.5 - eps;
        // oracle: in the plane z = 0 the ellipse (2cos θ, sin θ) has normal
        // (cos θ, 2 sin θ)/N; the swept centre path y = c first meets the offset
        // curve (2cos θ, sin θ) + r·n at the θ ∈ (π/2, π) where its y is c
        // (solved by bisection in f64): t = 10 + x(θ).
        let offset = |th: f64| {
            let n = (th.cos().powi(2) + 4.0 * th.sin().powi(2)).sqrt();
            (
                2.0 * th.cos() + r * th.cos() / n,
                th.sin() + r * 2.0 * th.sin() / n,
            )
        };
        let (mut lo, mut hi) = (std::f64::consts::FRAC_PI_2, std::f64::consts::PI);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if offset(mid).1 > c {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = 10.0 + offset(lo).0;
        let h = sphere_cast(&w, [-10.0, c, 0.0], r, [1.0, 0.0, 0.0], 100.0)
            .unwrap_or_else(|| panic!("eps = {eps}: the sphere grazes the ellipsoid at t = {t}"));
        assert!(
            h.t.to_f64() <= t + 1e-7,
            "eps = {eps}: t = {} after {t}",
            h.t.to_f64()
        );
        assert!(
            (h.t.to_f64() - t).abs() < 1e-6,
            "eps = {eps}: t = {} but {t}",
            h.t.to_f64()
        );
    }
}

#[test]
fn grazing_capsule_cast_onto_a_box_edge_still_hits() {
    let mut w = world();
    unit_box(&mut w);
    for eps in [1e-3, 1e-5, 1e-6] {
        // oracle: the lower end (−10, 1.5 − eps) moving +X meets the top edge
        // (x = −1, y = 1) when (x + 1)² + (0.5 − eps)² = 0.25:
        // t = 9 − √(0.25 − (0.5 − eps)²).
        let t = 9.0 - (0.25f64 - (0.5 - eps) * (0.5 - eps)).sqrt();
        let h = capsule_cast(
            &w,
            [-10.0, 1.5 - eps, 0.0],
            [-10.0, 3.0, 0.0],
            0.5,
            [1.0, 0.0, 0.0],
            100.0,
        )
        .unwrap_or_else(|| panic!("eps = {eps}: the capsule grazes the edge at t = {t}"));
        assert!(
            h.t.to_f64() <= t + 1e-9,
            "eps = {eps}: t = {} after {t}",
            h.t.to_f64()
        );
        assert!(
            (h.t.to_f64() - t).abs() < 1e-6,
            "eps = {eps}: t = {} but {t}",
            h.t.to_f64()
        );
    }
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

fn ctrl(p: [f64; 3], config: CharacterConfig) -> CharacterController {
    CharacterController::new(v3(p[0], p[1], p[2]), config)
}

fn settled(w: &PhysicsWorld, p: [f64; 3], config: CharacterConfig) -> CharacterController {
    let mut c = ctrl(p, config);
    w.move_character(&mut c, Vec3Fix::ZERO);
    assert!(c.grounded, "the scene's floor holds the controller");
    c
}

#[test]
fn a_single_long_move_stops_at_a_mesh_wall_beyond_the_floor() {
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
    let mut c = settled(&w, [0.0, STAND, 0.0], CharacterConfig::default());
    // oracle: the floor is parallel to the move; the wall x = 12 stops the
    // capsule a skin width short: x = 12 − r − s = 11.69, y = h/2 + s.
    let res = w.move_character(&mut c, v3(15.0, 0.0, 0.0));
    assert_pos(res.position, [12.0 - R - SKIN, STAND, 0.0], MOVE_TOL);
    assert!(res.grounded);
}
