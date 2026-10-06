//! Oracles for world shape queries near contact: gaps of `2⁻¹⁰` and below,
//! far under the bound GJK uses to decide that two convex sets intersect
//! (`|v| ≤ 2⁻³²`).
//!
//! # Expected values
//!
//! Every expected value is written from the geometry by hand; the closed form is
//! in a comment next to each assertion. Nothing here calls the code under test to
//! make an expected value.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};
use alice_physics::world_shape_query::WorldShapeHit;

const ITER: f64 = 1e-9;

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

fn overlap_s(w: &PhysicsWorld, c: [f64; 3], r: f64) -> Vec<RayTarget> {
    w.overlap_sphere(p3(c), fx(r), &RayFilter::default())
}

fn overlap_b(w: &PhysicsWorld, lo: [f64; 3], hi: [f64; 3]) -> Vec<RayTarget> {
    w.overlap_aabb(&AABB::new(p3(lo), p3(hi)), &RayFilter::default())
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

#[test]
fn near_contact_is_not_overlap() {
    let g = 1.0 / 1024.0;
    let mut w = world();
    let e = shaped(
        &mut w,
        Shape::Ellipsoid {
            radii: v3(2.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
    );
    // oracle: the ellipsoid's top is y = 1; a box from y = 1 + 2⁻¹⁰ is 2⁻¹⁰ above
    // it, and a sphere of radius 2⁻¹² at (0, 1 + 2⁻¹⁰, 0) is 2⁻¹⁰ − 2⁻¹² clear.
    assert!(overlap_b(&w, [-0.1, 1.0 + g, -0.1], [0.1, 2.0, 0.1]).is_empty());
    assert!(overlap_s(&w, [0.0, 1.0 + g, 0.0], g / 4.0).is_empty());
    // oracle: a box from y = 1 − 2⁻¹⁰ reaches into it.
    assert_eq!(
        overlap_b(&w, [-0.1, 1.0 - g, -0.1], [0.1, 2.0, 0.1]),
        vec![RayTarget::Body(e)]
    );

    let mut w = world();
    let b = unit_box(&mut w);
    // oracle: a segment (radius 0) 2⁻¹⁰ above the top face moving down touches
    // it after 2⁻¹⁰: t = 2⁻¹⁰, normal +Y (not a start overlap at t = 0).
    let h = capsule_cast(
        &w,
        [-0.5, 1.0 + g, 0.0],
        [0.5, 1.0 + g, 0.0],
        0.0,
        [0.0, -1.0, 0.0],
        10.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert!((h.t.to_f64() - g).abs() < ITER, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [0.0, 1.0, 0.0], ITER, "normal");
    // oracle: a capsule of radius 2⁻¹² with its segment 2⁻¹⁰ above: t = 2⁻¹⁰ − 2⁻¹².
    let h = capsule_cast(
        &w,
        [-0.5, 1.0 + g, 0.0],
        [0.5, 1.0 + g, 0.0],
        g / 4.0,
        [0.0, -1.0, 0.0],
        10.0,
    )
    .expect("hit");
    assert!(
        (h.t.to_f64() - 0.75 * g).abs() < ITER,
        "t = {}",
        h.t.to_f64()
    );
}
