//! A body whose stored orientation is not a unit quaternion (norm 2 or 1/2) is
//! placed by the rotation that quaternion stands for: compound children, world
//! shape queries and ray casts give the same answer as for the normalized
//! orientation.
//!
//! `q v q*` scales `v` by `|q|^2`, so applying the stored quaternion as is moves
//! and resizes what it turns (a compound child at body-local `x = 1` would sit at
//! `x = 4` for a doubled orientation).
//!
//! # Expected values
//!
//! For each scaled orientation `s·q` the answer must be bit for bit the answer for
//! `normalize(s·q)`: the code applies a unit rotation, and a quaternion within
//! `2^-32` of unit length is applied unchanged, so both inputs reach the same
//! rotation. Against the original unit `q` (which differs from `normalize(s·q)` by
//! the rounding of the normalization) the answers agree to `1e-12` on closed-form
//! paths and to `1e-8` on the iterative ones (capsule casts against a solid
//! stop when the gap is below `2^-32`).
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere, AABB};
use alice_physics::compound::CompoundShape;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld};

const NEAR: f64 = 1e-12;
const ITER: f64 = 1e-8;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn scaled(q: QuatFix, s: Fix128) -> QuatFix {
    QuatFix::new(q.x * s, q.y * s, q.z * s, q.w * s)
}

/// A turn about a slanted axis, so every component of the quaternion is non-zero.
fn turn() -> QuatFix {
    QuatFix::from_axis_angle(v3(1.0, 2.0, 3.0).normalize(), fx(0.7))
}

/// The two non-unit scalings tested, each with the unit quaternion it stands for.
fn cases() -> Vec<(f64, QuatFix, QuatFix)> {
    let q = turn();
    [2.0, 0.5]
        .into_iter()
        .map(|s| {
            let qs = scaled(q, fx(s));
            (s, qs, qs.normalize())
        })
        .collect()
}

fn near(a: Fix128, b: Fix128, what: &str) {
    near_tol(a, b, NEAR, what);
}

fn near_tol(a: Fix128, b: Fix128, tol: f64, what: &str) {
    let d = (a - b).abs().to_f64();
    assert!(
        d < tol,
        "{what}: {} vs {} (|d| = {d:e})",
        a.to_f64(),
        b.to_f64()
    );
}

fn near_v(a: Vec3Fix, b: Vec3Fix, what: &str) {
    near(a.x, b.x, what);
    near(a.y, b.y, what);
    near(a.z, b.z, what);
}

fn near_box(a: &AABB, b: &AABB, what: &str) {
    near_v(a.min, b.min, what);
    near_v(a.max, b.max, what);
}

/// A sphere, a capsule, a box and a hull child, each off the body origin and
/// turned in the body.
fn compound() -> CompoundShape {
    let mut c = CompoundShape::new();
    let local = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(0.3));
    c.add_sphere(
        Sphere::new(v3(0.25, 0.0, 0.0), fx(0.5)),
        v3(1.5, 0.0, 0.0),
        local,
    );
    c.add_capsule(
        Capsule::new(v3(-0.5, 0.0, 0.0), v3(0.5, 0.0, 0.0), fx(0.25)),
        v3(0.0, 1.5, 0.0),
        local,
    );
    c.add_box(
        OrientedBox::new(
            v3(0.0, 0.0, 0.25),
            v3(0.5, 0.25, 0.75),
            QuatFix::from_axis_angle(Vec3Fix::UNIT_X, fx(0.2)),
        ),
        v3(0.0, 0.0, -1.5),
        local,
    );
    c.add_convex_hull(
        ConvexHull::new(vec![
            v3(0.0, -0.5, 0.0),
            v3(0.5, 0.0, 0.0),
            v3(-0.5, 0.0, 0.0),
            v3(0.0, 0.0, 0.5),
            v3(0.0, 0.5, 0.0),
        ]),
        v3(-1.5, -0.5, 0.0),
        local,
    );
    c
}

fn directions() -> Vec<Vec3Fix> {
    vec![
        Vec3Fix::UNIT_X,
        -Vec3Fix::UNIT_X,
        Vec3Fix::UNIT_Y,
        -Vec3Fix::UNIT_Y,
        Vec3Fix::UNIT_Z,
        -Vec3Fix::UNIT_Z,
        v3(1.0, 1.0, 1.0),
        v3(-1.0, 0.5, 2.0),
    ]
}

/// `child_world_aabb`, `world_aabb`, `overlapping_children` and both
/// `support_world`s of a compound turned by a scaled quaternion.
#[test]
fn compound_children_are_placed_by_the_unit_rotation() {
    let c = compound();
    let pos = v3(1.0, 2.0, 3.0);
    let q = turn();
    for (s, qs, qn) in cases() {
        for i in 0..c.children.len() {
            let got = c.child_world_aabb(i, pos, qs);
            assert_eq!(
                got,
                c.child_world_aabb(i, pos, qn),
                "child {i} box, s = {s}"
            );
            near_box(&got, &c.child_world_aabb(i, pos, q), "child box vs unit q");
        }
        let got = c.world_aabb(pos, qs);
        assert_eq!(got, c.world_aabb(pos, qn), "compound box, s = {s}");
        near_box(&got, &c.world_aabb(pos, q), "compound box vs unit q");
        // A box around the sphere child only (near body-local (1.75, 0, 0)).
        let probe = c.child_world_aabb(0, pos, q);
        assert_eq!(
            c.overlapping_children(&probe, pos, qs),
            c.overlapping_children(&probe, pos, q),
            "overlapping children, s = {s}"
        );
        for d in directions() {
            let got = c.support_world(d, pos, qs);
            assert_eq!(got, c.support_world(d, pos, qn), "support, s = {s}");
            near_v(got, c.support_world(d, pos, q), "support vs unit q");
            for (i, child) in c.children.iter().enumerate() {
                let got = child.support_world(d, pos, qs);
                assert_eq!(
                    got,
                    child.support_world(d, pos, qn),
                    "child {i} support, s = {s}"
                );
                near_v(
                    got,
                    child.support_world(d, pos, q),
                    "child support vs unit q",
                );
            }
        }
    }
}

/// A world holding one body made by `add`, turned to `rotation`.
fn world_with(
    add: impl Fn(&mut PhysicsWorld) -> usize,
    rotation: QuatFix,
) -> (PhysicsWorld, usize) {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let b = add(&mut w);
    w.bodies[b].rotation = rotation;
    (w, b)
}

/// Every query the scene asks, as comparable values: ray and shape casts toward
/// the body from six sides and from slanted origins, sphere and box overlaps.
#[derive(Debug, PartialEq)]
struct Answers {
    rays: Vec<Option<(Fix128, Vec3Fix, Vec3Fix, RayTarget)>>,
    spheres: Vec<Option<(Fix128, Vec3Fix, Vec3Fix, RayTarget)>>,
    capsules: Vec<Option<(Fix128, Vec3Fix, Vec3Fix, RayTarget)>>,
    overlaps: Vec<Vec<RayTarget>>,
    boxes: Vec<Vec<RayTarget>>,
}

fn origins(center: Vec3Fix) -> Vec<(Vec3Fix, Vec3Fix)> {
    let mut out = Vec::new();
    for d in directions() {
        let dir = d.normalize();
        for offset in [v3(0.0, 0.0, 0.0), v3(0.3, -0.2, 0.1), v3(-0.6, 0.4, 0.5)] {
            out.push((center - dir * fx(10.0) + offset, dir));
        }
    }
    out
}

fn answers(w: &PhysicsWorld, center: Vec3Fix) -> Answers {
    let f = RayFilter::default();
    let max = fx(100.0);
    let mut a = Answers {
        rays: Vec::new(),
        spheres: Vec::new(),
        capsules: Vec::new(),
        overlaps: Vec::new(),
        boxes: Vec::new(),
    };
    for (o, d) in origins(center) {
        a.rays.push(
            w.cast_ray(o, d, max, &f)
                .map(|h| (h.t, h.point, h.normal, h.target)),
        );
        a.spheres.push(
            w.cast_sphere(o, fx(0.25), d, max, &f)
                .map(|h| (h.t, h.point, h.normal, h.target)),
        );
        let side = if d.cross(Vec3Fix::UNIT_Y).length() > fx(0.1) {
            d.cross(Vec3Fix::UNIT_Y).normalize() * fx(0.3)
        } else {
            Vec3Fix::UNIT_X * fx(0.3)
        };
        a.capsules.push(
            w.cast_capsule(o - side, o + side, fx(0.2), d, max, &f)
                .map(|h| (h.t, h.point, h.normal, h.target)),
        );
    }
    // Probes on a shell around the body, inside the doubled reach of a child and
    // outside its true reach, and closer in.
    for r in [0.5, 1.0, 1.6, 2.2, 3.0] {
        for d in directions() {
            let p = center + d.normalize() * fx(r);
            a.overlaps.push(w.overlap_sphere(p, fx(0.3), &f));
            let h = v3(0.2, 0.2, 0.2);
            a.boxes.push(w.overlap_aabb(&AABB::new(p - h, p + h), &f));
        }
    }
    a
}

fn hit_near(
    a: &Option<(Fix128, Vec3Fix, Vec3Fix, RayTarget)>,
    b: &Option<(Fix128, Vec3Fix, Vec3Fix, RayTarget)>,
    tol: f64,
    what: &str,
) {
    let near_v = |p: Vec3Fix, q: Vec3Fix| {
        near_tol(p.x, q.x, tol, what);
        near_tol(p.y, q.y, tol, what);
        near_tol(p.z, q.z, tol, what);
    };
    match (a, b) {
        (None, None) => {}
        (Some(x), Some(y)) => {
            near_tol(x.0, y.0, tol, what);
            near_v(x.1, y.1);
            near_v(x.2, y.2);
            assert_eq!(x.3, y.3, "{what}");
        }
        _ => panic!("{what}: {a:?} vs {b:?}"),
    }
}

fn check_scene(
    name: &str,
    add: impl Fn(&mut PhysicsWorld) -> usize,
    center: Vec3Fix,
    scales: &[f64],
    gjk: bool,
) {
    let answers = |w: &PhysicsWorld| {
        let mut a = answers(w, center);
        if !gjk {
            a.capsules.clear();
            a.boxes.clear();
        }
        a
    };
    let q = turn();
    let (w, _) = world_with(&add, q);
    let unit = answers(&w);
    let hits = unit.rays.iter().filter(|h| h.is_some()).count();
    let overlaps = unit.overlaps.iter().filter(|o| !o.is_empty()).count();
    assert!(hits > 0 && overlaps > 0, "{name}: the scene tests nothing");
    for (s, qs, qn) in cases().into_iter().filter(|c| scales.contains(&c.0)) {
        let (ws, _) = world_with(&add, qs);
        let (wn, _) = world_with(&add, qn);
        let got = answers(&ws);
        assert_eq!(got, answers(&wn), "{name}, s = {s}");
        for (k, (g, u)) in got.rays.iter().zip(&unit.rays).enumerate() {
            hit_near(g, u, NEAR, &format!("{name} ray {k}, s = {s}"));
        }
        for (k, (g, u)) in got.spheres.iter().zip(&unit.spheres).enumerate() {
            hit_near(g, u, NEAR, &format!("{name} sphere cast {k}, s = {s}"));
        }
        for (k, (g, u)) in got.capsules.iter().zip(&unit.capsules).enumerate() {
            hit_near(g, u, ITER, &format!("{name} capsule cast {k}, s = {s}"));
        }
        assert_eq!(
            got.overlaps, unit.overlaps,
            "{name} sphere overlaps, s = {s}"
        );
        assert_eq!(got.boxes, unit.boxes, "{name} box overlaps, s = {s}");
    }
}

/// Ray casts, sphere and capsule casts and overlaps against a compound body.
#[test]
fn world_queries_on_a_compound_use_the_unit_rotation() {
    let pos = v3(1.0, 2.0, 3.0);
    check_scene(
        "compound",
        |w| {
            w.add_compound_body(&compound(), Fix128::ONE, pos)
                .expect("valid compound")
        },
        pos,
        &[2.0, 0.5],
        true,
    );
}

/// The same queries against a box body and a cylinder body, for the doubled
/// norm. With norm 1/2 the body's broad-phase box (`PosedShape::world_aabb`, not
/// one of these query paths) must itself use the unit rotation for the body to be
/// found at all. A capsule cast and a box overlap against the cylinder run on
/// `PosedShape`'s support (convex time of impact, GJK), likewise outside these
/// paths, so the cylinder is tested with rays, sphere casts and sphere overlaps
/// (its closed forms).
#[test]
fn world_queries_on_a_posed_shape_use_the_unit_rotation() {
    let pos = v3(-1.0, 0.5, 2.0);
    for (name, shape, gjk) in [
        (
            "box",
            Shape::Box {
                half_extents: v3(1.0, 0.5, 0.75),
            },
            true,
        ),
        (
            "cylinder",
            Shape::Cylinder {
                radius: fx(0.5),
                half_height: fx(1.0),
            },
            false,
        ),
    ] {
        check_scene(
            name,
            |w| {
                w.add_shaped_body(&shape, Fix128::ONE, pos)
                    .expect("valid shape")
            },
            pos,
            &[2.0],
            gjk,
        );
    }
}
