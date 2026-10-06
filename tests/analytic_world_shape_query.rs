//! Oracles for `PhysicsWorld::cast_sphere`, `cast_capsule`, `overlap_sphere` and
//! `overlap_aabb`: swept and overlap queries against the geometry the world
//! collides with, compared with closed forms.
//!
//! # Expected values
//!
//! Every expected distance, normal and contact point is written from the geometry
//! by hand; the closed form is in a comment next to each assertion. None calls the
//! code under test. A sphere of radius `r` swept along a unit direction first
//! touches a solid `S` where its centre reaches the boundary of the Minkowski sum
//! `S ⊕ ball(r)`: a sphere of radius `R + r` for a sphere, a box with its faces
//! pushed out by `r`, its edges turned into cylinders of radius `r` and its
//! corners into spheres of radius `r` (a rounded box), and so on.
//!
//! # Tolerances
//!
//! - `EXACT = 1e-12`: closed-form paths (sphere, box, capsule child, plane, triangle
//!   mesh, torus). Each `Fix128` operation truncates by at most `2⁻⁶⁴`; a query is
//!   a few hundred operations on numbers below 100, so the error is far below
//!   `1e-12`, and a wrong formula errs by `1e-2` or more in every scene here.
//! - `ITER = 1e-9`: iterative paths (GJK distance and conservative advancement for
//!   cylinders, cones, ellipsoids, capsule casts against solids, height fields).
//!   The iteration stops when the gap is below `2⁻³²` (`2.3e-10`).
//! - SDF: the field is `f32`; a hit is where the field minus the radius drops
//!   below `SdfCcdConfig::tolerance` (`1e-3`), so `t ∈ [d − tol, d + F32]`.
//!
//! # Against the bounding-sphere API
//!
//! `alice_physics::query::{sphere_cast, overlap_sphere, overlap_aabb}` test each
//! body's bounding sphere or position. The corner-gap scenes below show a case the
//! old functions get wrong and assert the world query answers correctly.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, Sphere, AABB};
use alice_physics::compound::CompoundShape;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget};
use alice_physics::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;
use alice_physics::world_shape_query::WorldShapeHit;

const EXACT: f64 = 1e-12;
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

fn overlap_s(w: &PhysicsWorld, c: [f64; 3], r: f64) -> Vec<RayTarget> {
    w.overlap_sphere(p3(c), fx(r), &RayFilter::default())
}

fn overlap_b(w: &PhysicsWorld, lo: [f64; 3], hi: [f64; 3]) -> Vec<RayTarget> {
    w.overlap_aabb(&AABB::new(p3(lo), p3(hi)), &RayFilter::default())
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
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

/// Assert a hit at `t` with `normal` (normalized here) and contact `point`.
#[track_caller]
fn assert_hit(
    hit: Option<WorldShapeHit>,
    target: RayTarget,
    t: f64,
    normal: [f64; 3],
    point: [f64; 3],
    tol: f64,
) -> WorldShapeHit {
    let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
    assert_eq!(h.target, target, "target");
    assert!(
        (h.t.to_f64() - t).abs() < tol,
        "t = {} but the closed form is {t}",
        h.t.to_f64()
    );
    assert_vec(h.normal, unit(normal), tol, "normal");
    assert_vec(h.point, point, tol, "point");
    h
}

/// A body of `shape` with its centre of mass at `pos`, turned by `rot`.
fn shaped(w: &mut PhysicsWorld, shape: Shape, pos: Vec3Fix, rot: QuatFix) -> usize {
    let i = w
        .add_shaped_body(&shape, Fix128::ONE, pos)
        .expect("valid shape");
    w.bodies[i].rotation = rot;
    i
}

fn unit_box_shape() -> Shape {
    Shape::Box {
        half_extents: v3(1.0, 1.0, 1.0),
    }
}

fn unit_box(w: &mut PhysicsWorld) -> usize {
    shaped(w, unit_box_shape(), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn sphere_body(w: &mut PhysicsWorld, pos: Vec3Fix, r: f64) -> usize {
    w.add_body_with_radius(RigidBody::new_static(pos), fx(r))
}

fn ground_plane(w: &mut PhysicsWorld) -> usize {
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )))
}

fn rot_y(deg: f64) -> QuatFix {
    QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(deg.to_radians()))
}

// ============================================================ sphere cast: bodies

#[test]
fn sphere_cast_against_sphere_body() {
    let mut w = world();
    let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
    // oracle: sphere-sphere sweep, centre reaches |c| = R + r = 1.5 on the line
    // y = 0.3: x = −√(1.5² − 0.3²) = −√2.16, t = 10 − √2.16; normal = c/1.5,
    // contact = normal · R.
    let s = 2.16f64.sqrt();
    let n = [-s / 1.5, 0.3 / 1.5, 0.0];
    assert_hit(
        sphere_cast(&w, [-10.0, 0.3, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        10.0 - s,
        n,
        n,
        EXACT,
    );
    // oracle: the line y = 1.6 passes 1.6 > 1.5 from the centre: no contact.
    assert_eq!(
        sphere_cast(&w, [-10.0, 1.6, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        None
    );
}

#[test]
fn sphere_cast_against_box_face_edge_and_corner() {
    let mut w = world();
    let b = unit_box(&mut w);
    // oracle: rounded box, face region: the face x = −1 pushed out to x = −1.5,
    // t = 10 − 1.5 = 8.5, normal −X, contact (−1, 0.2, 0.3).
    assert_hit(
        sphere_cast(&w, [-10.0, 0.2, 0.3], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.5,
        [-1.0, 0.0, 0.0],
        [-1.0, 0.2, 0.3],
        EXACT,
    );
    // oracle: rounded box, edge region (edge x = −1, y = 1): the centre at
    // (x, 1.3) is 0.5 from (−1, 1) when x = −1 − √(0.25 − 0.09) = −1.4, t = 8.6,
    // normal (−0.4, 0.3, 0)/0.5, contact (−1, 1, 0).
    assert_hit(
        sphere_cast(&w, [-10.0, 1.3, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.6,
        [-0.8, 0.6, 0.0],
        [-1.0, 1.0, 0.0],
        EXACT,
    );
    // oracle: rounded box, corner region (corner (−1, 1, 1)): the centre at
    // (x, 1.2, 1.2) is 0.5 from the corner when x = −1 − √(0.25 − 0.08),
    // t = 9 − √0.17, normal (−√0.17, 0.2, 0.2)/0.5.
    let s = 0.17f64.sqrt();
    assert_hit(
        sphere_cast(&w, [-10.0, 1.2, 1.2], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.0 - s,
        [-s, 0.2, 0.2],
        [-1.0, 1.0, 1.0],
        EXACT,
    );
}

/// The line `y = z = 1.4` passes the box edge `(y, z) = (1, 1)` at `0.4·√2 = 0.566`,
/// more than the radius 0.5: no contact. The bounding sphere (radius `√3`) plus
/// 0.5 reaches `2.23 > 1.4·√2 = 1.98`, so the bounding-sphere sweep reports a hit.
#[test]
fn sphere_cast_through_box_corner_gap_misses() {
    let mut w = world();
    unit_box(&mut w);
    let r = unit_box_shape().bounding_radius();
    let old = alice_physics::query::sphere_cast(
        v3(-10.0, 1.4, 1.4),
        fx(0.5),
        v3(1.0, 0.0, 0.0),
        fx(100.0),
        &w.bodies,
        r,
    );
    assert!(old.is_some(), "the bounding-sphere sweep reports the gap");
    // oracle: distance from the line to the edge 0.4·√2 > 0.5.
    assert_eq!(
        sphere_cast(&w, [-10.0, 1.4, 1.4], 0.5, [1.0, 0.0, 0.0], 100.0),
        None
    );
}

/// A box turned 90° about `Y` has the same faces; a box turned 45° presents the
/// vertical edge at `x = −√2`.
#[test]
fn sphere_cast_against_turned_box() {
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        rot_y(45.0),
    );
    // oracle: the vertical edge at (−√2, y, 0) is the nearest feature on the axis
    // z = 0; centre reaches x = −√2 − 0.5, t = 10 − √2 − 0.5, normal −X.
    assert_hit(
        sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        10.0 - 2f64.sqrt() - 0.5,
        [-1.0, 0.0, 0.0],
        [-(2f64.sqrt()), 0.0, 0.0],
        ITER,
    );
}

#[test]
fn sphere_cast_against_cylinder_cone_ellipsoid_torus() {
    // oracle: cylinder (radius 1, half height 1, axis Y): side pushed out to
    // x = −1.5 ⇒ t = 8.5; rim circle (radius 1, y = 1): centre (x, 1.3) is 0.5 from
    // (−1, 1) at x = −1.4 ⇒ t = 8.6, normal (−0.8, 0.6, 0).
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.5,
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        ITER,
    );
    assert_hit(
        sphere_cast(&w, [-10.0, 1.3, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.6,
        [-0.8, 0.6, 0.0],
        [-1.0, 1.0, 0.0],
        ITER,
    );

    // oracle: cone (radius 1, half height 1) with its centre of mass at the
    // origin: the geometric centre is h/2 = 0.5 above it, the apex at y = 1.5.
    // Straight down the axis the apex is nearest: t = 10 − 1.5 − 0.5 = 8.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Cone {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        sphere_cast(&w, [0.0, 10.0, 0.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.0,
        [0.0, 1.0, 0.0],
        [0.0, 1.5, 0.0],
        ITER,
    );

    // oracle: ellipsoid (2, 1, 1): along the major axis the vertex (−2, 0, 0) is
    // the nearest point of every point (x < −2, 0, 0): t = 10 − 2.5 = 7.5.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Ellipsoid {
            radii: v3(2.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        sphere_cast(&w, [-10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        7.5,
        [-1.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        ITER,
    );

    // oracle: torus (ring 2 in XZ, tube 0.5) swept by a sphere of 0.25 is a ray
    // against the torus of tube 0.75: down at x = 2 the tube top y = 0.75,
    // t = 9.25, contact (2, 0.5, 0). Down the axis the hole: the axis is
    // √(4 + y²) − 0.5 ≥ 1.5 > 0.25 from the tube, no contact.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        sphere_cast(&w, [2.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.25,
        [0.0, 1.0, 0.0],
        [2.0, 0.5, 0.0],
        ITER,
    );
    assert_eq!(
        sphere_cast(&w, [0.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        None
    );
}

/// Two box children (half 0.5) at `x = ±3` and nothing between them.
#[test]
fn sphere_cast_against_compound_children() {
    let mut w = world();
    let mut c = CompoundShape::new();
    for x in [3.0, -3.0] {
        c.add_box(
            OrientedBox::new(Vec3Fix::ZERO, v3(0.5, 0.5, 0.5), QuatFix::IDENTITY),
            v3(x, 0.0, 0.0),
            QuatFix::IDENTITY,
        );
    }
    let b = w
        .add_compound_body(&c, Fix128::ONE, Vec3Fix::ZERO)
        .expect("valid compound");
    // oracle: rounded-box face of the child at x = 3: top y = 0.5 pushed to 0.75,
    // t = 10 − 0.75 = 9.25, contact (3, 0.5, 0).
    assert_hit(
        sphere_cast(&w, [3.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.25,
        [0.0, 1.0, 0.0],
        [3.0, 0.5, 0.0],
        EXACT,
    );
    // oracle: rounded-box edge (x = 3.5, y = 0.5) of the same child: the centre at
    // x = 3.6 is 0.25 from it when y = 0.5 + √(0.0625 − 0.01), t = 9.5 − √0.0525.
    let s = 0.0525f64.sqrt();
    assert_hit(
        sphere_cast(&w, [3.6, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.5 - s,
        [0.1, s, 0.0],
        [3.5, 0.5, 0.0],
        EXACT,
    );
    // oracle: x = 0 is 2.5 from either child: the gap, no contact.
    assert_eq!(
        sphere_cast(&w, [0.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        None
    );
    // Sphere children at z = ±3 and a capsule child along X through the middle:
    // symmetric, so the centre of mass (the body origin) stays at the origin.
    let mut w = world();
    let mut c = CompoundShape::new();
    for z in [3.0, -3.0] {
        c.add_sphere(
            Sphere::new(Vec3Fix::ZERO, fx(0.5)),
            v3(0.0, 0.0, z),
            QuatFix::IDENTITY,
        );
    }
    c.add_capsule(
        Capsule::new(v3(-2.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let b = w
        .add_compound_body(&c, Fix128::ONE, Vec3Fix::ZERO)
        .expect("valid compound");
    // oracle: sphere-sphere, child radius 0.5 + 0.25 = 0.75 straight above (0, 0, 3).
    assert_hit(
        sphere_cast(&w, [0.0, 10.0, 3.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.25,
        [0.0, 1.0, 0.0],
        [0.0, 0.5, 3.0],
        EXACT,
    );
    // oracle: capsule child side, radius 0.5 + 0.25 above the segment at x = 1.
    assert_hit(
        sphere_cast(&w, [1.0, 10.0, 0.0], 0.25, [0.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        9.25,
        [0.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        EXACT,
    );
    // oracle: (0, y, 1.5) is at least 1.5 from the capsule segment and from the
    // sphere centre (0, 0, 3): more than 0.75 from both, no contact (inside the
    // bounding sphere).
    assert_eq!(
        sphere_cast(&w, [0.0, 10.0, 1.5], 0.25, [0.0, -1.0, 0.0], 100.0),
        None
    );
}

// ============================================================ sphere cast: static

#[test]
fn sphere_cast_against_plane() {
    let mut w = world();
    let p = ground_plane(&mut w);
    // oracle: sphere-plane, the centre reaches y = r: t = 5 − 0.5 = 4.5, contact
    // below the centre.
    assert_hit(
        sphere_cast(&w, [1.0, 5.0, 2.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        RayTarget::StaticCollider(p),
        4.5,
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],
        EXACT,
    );
    // oracle: oblique (1, −1, 0)/√2: the height drops 4.5 after 4.5·√2.
    let t = 4.5 * 2f64.sqrt();
    assert_hit(
        sphere_cast(&w, [0.0, 5.0, 0.0], 0.5, [1.0, -1.0, 0.0], 100.0),
        RayTarget::StaticCollider(p),
        t,
        [0.0, 1.0, 0.0],
        [4.5, 0.0, 0.0],
        EXACT,
    );
    // oracle: the plane is two-sided: from below the centre reaches y = −r.
    assert_hit(
        sphere_cast(&w, [0.0, -5.0, 0.0], 0.5, [0.0, 1.0, 0.0], 100.0),
        RayTarget::StaticCollider(p),
        4.5,
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
        EXACT,
    );
    // Parallel to the plane 1 above it: never closer than 1 > 0.5.
    assert_eq!(
        sphere_cast(&w, [0.0, 1.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        None
    );
}

#[test]
fn sphere_cast_against_triangle_mesh_face_and_edge() {
    let mut w = world();
    let verts = [v3(0.0, 0.0, 0.0), v3(4.0, 0.0, 0.0), v3(0.0, 0.0, 4.0)];
    let m = w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 1, 2],
    )));
    // oracle: face region: the triangle plane y = 0 reached at y = r, t = 2.5.
    assert_hit(
        sphere_cast(&w, [1.0, 3.0, 1.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        RayTarget::StaticCollider(m),
        2.5,
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        EXACT,
    );
    // oracle: edge region: the edge x = 0, y = 0 (z ∈ [0, 4]); the centre
    // (x, 0.3, 1) is 0.5 from (0, 0, 1) at x = −0.4, t = 9.6, normal (−0.8, 0.6, 0).
    assert_hit(
        sphere_cast(&w, [-10.0, 0.3, 1.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::StaticCollider(m),
        9.6,
        [-0.8, 0.6, 0.0],
        [0.0, 0.0, 1.0],
        EXACT,
    );
    // oracle: beside the hypotenuse x + z = 4 at (3, y, 3): distance to the
    // triangle is |(3,3) − (2,2)| = √2 > 0.5 for every y: no contact.
    assert_eq!(
        sphere_cast(&w, [3.0, 5.0, 3.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        None
    );
}

#[test]
fn sphere_cast_against_height_fields() {
    // oracle: flat field at height 0.25 over [0, 4]²: t = 5 − 0.25 − 0.5.
    let mut w = world();
    let h = w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
        5,
        5,
        fx(1.0),
        Vec3Fix::ZERO,
        fx(0.25),
    )));
    assert_hit(
        sphere_cast(&w, [2.0, 5.0, 2.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        RayTarget::StaticCollider(h),
        4.25,
        [0.0, 1.0, 0.0],
        [2.0, 0.25, 2.0],
        ITER,
    );
    // oracle: heights 0.5·x (every bilinear cell is the plane y = 0.5·x, normal
    // (−0.5, 1, 0)/√1.25): the centre (4, y, 4) is 0.5 from the plane when
    // y = 2 + 0.5·√1.25; contact = centre − 0.5·normal.
    let mut heights = vec![];
    for _z in 0..9 {
        for x in 0..9 {
            heights.push(fx(0.5 * f64::from(x)));
        }
    }
    let mut w = world();
    let h = w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        heights,
        9,
        9,
        fx(1.0),
        Vec3Fix::ZERO,
    )));
    let s = 1.25f64.sqrt();
    let n = [-0.5 / s, 1.0 / s, 0.0];
    let y = 2.0 + 0.5 * s;
    assert_hit(
        sphere_cast(&w, [4.0, 6.0, 4.0], 0.5, [0.0, -1.0, 0.0], 100.0),
        RayTarget::StaticCollider(h),
        6.0 - y,
        n,
        [4.0 - 0.5 * n[0], y - 0.5 * n[1], 4.0],
        ITER,
    );
}

#[cfg(feature = "std")]
#[test]
fn sphere_cast_against_sdf() {
    let mut w = world();
    w.sdf_colliders.push(SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    // oracle: unit sphere field: the centre reaches |c| = 1.5, t = 5 − 1.5 = 3.5.
    let h = sphere_cast(&w, [-5.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).expect("hit");
    assert_eq!(h.target, RayTarget::Sdf(0));
    let tol = f64::from(RayFilter::default().sdf.tolerance);
    assert!(
        h.t.to_f64() > 3.5 - tol - 1e-5 && h.t.to_f64() < 3.5 + 1e-5,
        "t = {}",
        h.t.to_f64()
    );
    assert_vec(h.normal, [-1.0, 0.0, 0.0], 1e-3, "normal");
}

// ============================================================ capsule cast

#[test]
fn capsule_cast_against_plane() {
    let mut w = world();
    let p = ground_plane(&mut w);
    // oracle: capsule-plane, the lower end (−1, 5, 0) reaches y = r first:
    // t = 5 − 0.5 = 4.5, contact (−1, 0, 0).
    assert_hit(
        capsule_cast(
            &w,
            [-1.0, 5.0, 0.0],
            [1.0, 6.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::StaticCollider(p),
        4.5,
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        EXACT,
    );
}

#[test]
fn capsule_cast_against_sphere_body() {
    let mut w = world();
    let b = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
    // oracle: the segment (−2..2, y, 0) passes over the centre: its nearest point
    // (0, y, 0) reaches distance 1.5 at y = 1.5, t = 3.5, contact (0, 1, 0).
    assert_hit(
        capsule_cast(
            &w,
            [-2.0, 5.0, 0.0],
            [2.0, 5.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::Body(b),
        3.5,
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        EXACT,
    );
    // oracle: the segment (1..3, y, 0): nearest point (1, y, 0) reaches 1.5 at
    // y = √1.25, t = 5 − √1.25, normal (1, √1.25, 0)/1.5.
    let s = 1.25f64.sqrt();
    let n = [1.0 / 1.5, s / 1.5, 0.0];
    assert_hit(
        capsule_cast(
            &w,
            [1.0, 5.0, 0.0],
            [3.0, 5.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::Body(b),
        5.0 - s,
        n,
        n,
        EXACT,
    );
    // oracle: the segment (2..4, y, 0) is never closer than 2 > 1.5.
    assert_eq!(
        capsule_cast(
            &w,
            [2.0, 5.0, 0.0],
            [4.0, 5.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0
        ),
        None
    );
}

#[test]
fn capsule_cast_against_box_and_triangle_mesh() {
    let mut w = world();
    let b = unit_box(&mut w);
    // oracle: horizontal capsule over the top face y = 1: t = 5 − 1 − 0.5 = 3.5.
    let h = capsule_cast(
        &w,
        [-0.5, 5.0, 0.0],
        [0.5, 5.0, 0.0],
        0.5,
        [0.0, -1.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert!((h.t.to_f64() - 3.5).abs() < ITER, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [0.0, 1.0, 0.0], ITER, "normal");
    assert!((h.point.y.to_f64() - 1.0).abs() < ITER);

    let mut w = world();
    let verts = [
        v3(-10.0, 0.0, -10.0),
        v3(10.0, 0.0, -10.0),
        v3(0.0, 0.0, 10.0),
    ];
    let m = w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 1, 2],
    )));
    // oracle: tilted capsule, lower end (0, 3, 0) reaches y = 0.5: t = 2.5,
    // contact (0, 0, 0).
    assert_hit(
        capsule_cast(
            &w,
            [0.0, 3.0, 0.0],
            [1.0, 4.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::StaticCollider(m),
        2.5,
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        ITER,
    );
}

/// A capsule whose two ends coincide is a sphere.
#[test]
fn capsule_cast_with_coincident_ends_is_sphere_cast() {
    let mut w = world();
    unit_box(&mut w);
    let a = capsule_cast(
        &w,
        [-10.0, 1.3, 0.0],
        [-10.0, 1.3, 0.0],
        0.5,
        [1.0, 0.0, 0.0],
        100.0,
    )
    .expect("hit");
    // oracle: the rounded-box edge case above, t = 8.6.
    assert!((a.t.to_f64() - 8.6).abs() < EXACT, "t = {}", a.t.to_f64());
}

// ============================================================ overlap

#[test]
fn overlap_sphere_against_box_edge_gap() {
    let mut w = world();
    let b = unit_box(&mut w);
    let r = unit_box_shape().bounding_radius();
    // The bounding-sphere overlap: |(1.4, 1.4, 0)| = 1.98 < √3 + 0.5.
    let old = alice_physics::query::overlap_sphere(v3(1.4, 1.4, 0.0), fx(0.5), &w.bodies, r);
    assert_eq!(old.len(), 1, "the bounding-sphere overlap reports the gap");
    // oracle: distance from (1.4, 1.4, 0) to the box edge (1, 1, z) is 0.4·√2 =
    // 0.566 > 0.5: no overlap; with radius 0.6 > 0.566 it overlaps.
    assert_eq!(overlap_s(&w, [1.4, 1.4, 0.0], 0.5), vec![]);
    assert_eq!(
        overlap_s(&w, [1.4, 1.4, 0.0], 0.6),
        vec![RayTarget::Body(b)]
    );
    // oracle: elongated box (3, 1, 1), corner (3, 1, 1): the centre (3.4, 1.4, 1.4)
    // is 0.4·√3 = 0.693 from it; radius 0.65 does not reach, 0.7 does. The
    // bounding sphere √11 + 0.65 = 3.967 > |c| = 3.934 does.
    let mut w = world();
    let long = Shape::Box {
        half_extents: v3(3.0, 1.0, 1.0),
    };
    let b = shaped(&mut w, long, Vec3Fix::ZERO, QuatFix::IDENTITY);
    let r = long.bounding_radius();
    let old = alice_physics::query::overlap_sphere(v3(3.4, 1.4, 1.4), fx(0.65), &w.bodies, r);
    assert_eq!(old.len(), 1);
    assert_eq!(overlap_s(&w, [3.4, 1.4, 1.4], 0.65), vec![]);
    assert_eq!(
        overlap_s(&w, [3.4, 1.4, 1.4], 0.7),
        vec![RayTarget::Body(b)]
    );
}

#[test]
fn overlap_aabb_against_box_edge_gap() {
    let mut w = world();
    let b = unit_box(&mut w);
    let r = unit_box_shape().bounding_radius();
    // The bounding-sphere overlap: the box [1.2, 2]² × [−0.5, 0.5] is
    // |(1.2, 1.2, 0)| = 1.70 < √3 from the centre.
    let q = AABB::new(v3(1.2, 1.2, -0.5), v3(2.0, 2.0, 0.5));
    let old = alice_physics::query::overlap_aabb_expanded(&q, &w.bodies, r);
    assert_eq!(old.len(), 1, "the bounding-sphere overlap reports the gap");
    // oracle: box-box separating axis X: 1.2 > 1.
    assert_eq!(overlap_b(&w, [1.2, 1.2, -0.5], [2.0, 2.0, 0.5]), vec![]);
    // oracle: [0.9, 2] overlaps [−1, 1] on every axis.
    assert_eq!(
        overlap_b(&w, [0.9, 0.9, -0.5], [2.0, 2.0, 0.5]),
        vec![RayTarget::Body(b)]
    );
    // oracle: a box turned 45° about Y reaches x = √2 at z = 0: [1.3, 2] meets it,
    // [1.5, 2] does not.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        rot_y(45.0),
    );
    assert_eq!(
        overlap_b(&w, [1.3, -0.1, -0.05], [2.0, 0.1, 0.05]),
        vec![RayTarget::Body(b)]
    );
    assert_eq!(overlap_b(&w, [1.5, -0.1, -0.05], [2.0, 0.1, 0.05]), vec![]);
}

#[test]
fn overlap_against_static_colliders() {
    let mut w = world();
    let p = ground_plane(&mut w);
    let verts = [v3(10.0, 0.0, 0.0), v3(14.0, 0.0, 0.0), v3(10.0, 0.0, 4.0)];
    let m = w.add_static_collider(StaticCollider::TriMesh(TriMesh::from_indexed(
        &verts,
        &[0, 1, 2],
    )));
    // oracle: plane y = 0, |y| < r.
    assert_eq!(
        overlap_s(&w, [0.0, 0.4, 0.0], 0.5),
        vec![RayTarget::StaticCollider(p)]
    );
    assert_eq!(overlap_s(&w, [0.0, 0.6, 0.0], 0.5), vec![]);
    assert_eq!(
        overlap_s(&w, [0.0, -0.4, 0.0], 0.5),
        vec![RayTarget::StaticCollider(p)]
    );
    // oracle: the triangle (and the plane under it): (11, 0.4, 1) is 0.4 above
    // both.
    assert_eq!(
        overlap_s(&w, [11.0, 0.4, 1.0], 0.5),
        vec![RayTarget::StaticCollider(p), RayTarget::StaticCollider(m)]
    );
    // oracle: AABB vs plane: [−1, 0.1, −1]..[1, 1, 1] is above y = 0.
    assert_eq!(overlap_b(&w, [-1.0, 0.1, -1.0], [1.0, 1.0, 1.0]), vec![]);
    assert_eq!(
        overlap_b(&w, [-1.0, -0.1, -1.0], [1.0, 1.0, 1.0]),
        vec![RayTarget::StaticCollider(p)]
    );
    // oracle: AABB vs triangle: the box [13, 14]² over y ∈ [−1, 1] has x + z ≥ 26,
    // beyond the hypotenuse x + z = 14: the triangle is not in it, the plane is.
    // The box [11, 11.5] × [1, 1.5] lies over the triangle's interior.
    assert_eq!(
        overlap_b(&w, [13.0, -1.0, 13.0], [14.0, 1.0, 14.0]),
        vec![RayTarget::StaticCollider(p)]
    );
    assert_eq!(
        overlap_b(&w, [11.0, -1.0, 1.0], [11.5, 1.0, 1.5]),
        vec![RayTarget::StaticCollider(p), RayTarget::StaticCollider(m)]
    );
}

// ============================================================ conventions

/// A sphere of radius 0 is a ray: the same target and distance as `cast_ray`.
#[test]
fn radius_zero_sphere_cast_equals_cast_ray() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 0.5, 2.0),
        },
        v3(0.0, 0.0, 0.0),
        rot_y(30.0),
    );
    shaped(
        &mut w,
        Shape::Cone {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        v3(5.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(0.7),
            half_height: fx(1.0),
        },
        v3(-5.0, 0.0, 0.0),
        rot_y(10.0),
    );
    ground_plane(&mut w);
    for (o, d) in [
        ([-10.0, 0.2, 0.1], [1.0, 0.0, 0.0]),
        ([5.2, 10.0, 0.1], [0.0, -1.0, 0.0]),
        ([-5.3, 10.0, 0.2], [0.1, -1.0, 0.0]),
        ([20.0, 3.0, 0.0], [-1.0, -0.2, 0.0]),
        ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ] {
        let ray = w.cast_ray(p3(o), p3(d), fx(100.0), &RayFilter::default());
        let sweep = w.cast_sphere(p3(o), Fix128::ZERO, p3(d), fx(100.0), &RayFilter::default());
        assert_eq!(
            ray.map(|h| (h.target, h.t)),
            sweep.map(|h| (h.target, h.t)),
            "origin {o:?} direction {d:?}"
        );
        assert!(ray.is_some());
    }
}

#[test]
fn starting_overlap_reports_t_zero() {
    let mut w = world();
    let b = unit_box(&mut w);
    // oracle: centre (0, 1.2, 0) is 0.2 < 0.5 above the top face: overlapping at
    // the start, t = 0, normal −direction (the `cast_ray` convention).
    let h = sphere_cast(&w, [0.0, 1.2, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0).expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert_eq!(h.t, Fix128::ZERO);
    assert_vec(h.normal, [-1.0, 0.0, 0.0], EXACT, "normal");
    let h = capsule_cast(
        &w,
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        0.1,
        [0.0, 1.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert_eq!(h.t, Fix128::ZERO);
}

#[test]
fn filter_excludes_bodies_and_static_colliders() {
    let mut w = world();
    let near = sphere_body(&mut w, v3(0.0, 0.0, 0.0), 1.0);
    let far = sphere_body(&mut w, v3(5.0, 0.0, 0.0), 1.0);
    let p = w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_X,
        fx(20.0),
    )));
    let c = v3(-10.0, 0.0, 0.0);
    let d = v3(1.0, 0.0, 0.0);
    // oracle: t = 10 − 1.5 = 8.5 for the near sphere, 13.5 for the far one, the
    // plane x = 20 at 29.5.
    let hit = |f: &RayFilter| {
        w.cast_sphere(c, fx(0.5), d, fx(100.0), f)
            .map(|h| (h.target, h.t))
    };
    assert_eq!(
        hit(&RayFilter::default()),
        Some((RayTarget::Body(near), fx(8.5)))
    );
    assert_eq!(
        hit(&RayFilter::default().excluding_body(near)),
        Some((RayTarget::Body(far), fx(13.5)))
    );
    assert_eq!(
        hit(&RayFilter::default().excluding_body(near).with_layer_mask(0)),
        Some((RayTarget::StaticCollider(p), fx(29.5)))
    );
    assert_eq!(
        hit(&RayFilter::default().with_layer_mask(0).with_static(false)),
        None
    );
    assert_eq!(
        w.overlap_sphere(v3(2.5, 0.0, 0.0), fx(2.0), &RayFilter::default()),
        vec![RayTarget::Body(near), RayTarget::Body(far)]
    );
    assert_eq!(
        w.overlap_sphere(
            v3(2.5, 0.0, 0.0),
            fx(2.0),
            &RayFilter::default().excluding_body(far)
        ),
        vec![RayTarget::Body(near)]
    );
}

/// Two spheres the cast reaches at the same `t`: the lower body index, whatever
/// order the bodies were added in.
#[test]
fn ties_are_broken_by_target() {
    for flip in [false, true] {
        let mut w = world();
        let zs = if flip { [-1.5, 1.5] } else { [1.5, -1.5] };
        for z in zs {
            sphere_body(&mut w, v3(0.0, 0.0, z), 1.0);
        }
        // oracle: both at |c − (0, 0, ±1.5)| = 1.6: x = −√(1.6² − 1.5²), symmetric.
        let h = sphere_cast(&w, [-10.0, 0.0, 0.0], 0.6, [1.0, 0.0, 0.0], 100.0).expect("hit");
        assert_eq!(h.target, RayTarget::Body(0));
        let x = (1.6f64 * 1.6 - 1.5 * 1.5).sqrt();
        assert!((h.t.to_f64() - (10.0 - x)).abs() < EXACT);
    }
}

// ============================================================ degenerate inputs

#[test]
fn degenerate_inputs() {
    let mut w = world();
    unit_box(&mut w);
    ground_plane(&mut w);
    let f = RayFilter::default();
    let c = v3(-10.0, 0.5, 0.0);
    let d = v3(1.0, 0.0, 0.0);
    // Zero direction: None (as `cast_ray`).
    assert_eq!(
        w.cast_sphere(c, fx(0.5), Vec3Fix::ZERO, fx(100.0), &f),
        None
    );
    assert_eq!(
        w.cast_capsule(c, c + d, fx(0.5), Vec3Fix::ZERO, fx(100.0), &f),
        None
    );
    // max_t ≤ 0: None, even when overlapping at the start.
    assert_eq!(w.cast_sphere(c, fx(0.5), d, Fix128::ZERO, &f), None);
    assert_eq!(
        w.cast_sphere(v3(0.0, 0.0, 0.0), fx(0.5), d, fx(-1.0), &f),
        None
    );
    assert_eq!(w.cast_capsule(c, c + d, fx(0.5), d, fx(-1.0), &f), None);
    // Negative radius: not a sphere, None / empty.
    assert_eq!(w.cast_sphere(c, fx(-0.5), d, fx(100.0), &f), None);
    assert_eq!(w.cast_capsule(c, c + d, fx(-0.5), d, fx(100.0), &f), None);
    assert_eq!(w.overlap_sphere(Vec3Fix::ZERO, fx(-0.5), &f), vec![]);
    // An inverted AABB (min > max) contains nothing: empty.
    assert_eq!(
        w.overlap_aabb(&AABB::new(v3(1.0, 1.0, 1.0), v3(-1.0, -1.0, -1.0)), &f),
        vec![]
    );
    // Radius 0 overlap: a point strictly inside the box.
    assert_eq!(
        w.overlap_sphere(v3(0.5, 0.5, 0.5), Fix128::ZERO, &f),
        vec![RayTarget::Body(0)]
    );

    // Empty world: None / empty.
    let e = world();
    assert_eq!(e.cast_sphere(c, fx(0.5), d, fx(100.0), &f), None);
    assert_eq!(e.cast_capsule(c, c + d, fx(0.5), d, fx(100.0), &f), None);
    assert_eq!(e.overlap_sphere(c, fx(0.5), &f), vec![]);
    assert_eq!(
        e.overlap_aabb(&AABB::new(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0)), &f),
        vec![]
    );
}

/// Coordinates of `1e6`: the same distances as at the origin.
#[test]
fn large_coordinates() {
    let mut w = world();
    let s = sphere_body(&mut w, v3(1e6, 0.0, 0.0), 1.0);
    let b = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(1e6, 0.0, 1e6),
        QuatFix::IDENTITY,
    );
    // oracle: sphere-sphere head on, t = 10 − 1.5.
    assert_hit(
        sphere_cast(&w, [1e6 - 10.0, 0.0, 0.0], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(s),
        8.5,
        [-1.0, 0.0, 0.0],
        [1e6 - 1.0, 0.0, 0.0],
        1e-9,
    );
    // oracle: rounded-box edge as at the origin, t = 8.6.
    assert_hit(
        sphere_cast(&w, [1e6 - 10.0, 1.3, 1e6], 0.5, [1.0, 0.0, 0.0], 100.0),
        RayTarget::Body(b),
        8.6,
        [-0.8, 0.6, 0.0],
        [1e6 - 1.0, 1.0, 1e6],
        1e-9,
    );
    assert_eq!(
        overlap_s(&w, [1e6 + 1.4, 1.4, 1e6], 0.5),
        vec![],
        "edge gap at 1e6"
    );
    assert_eq!(
        overlap_s(&w, [1e6 + 1.4, 1.4, 1e6], 0.6),
        vec![RayTarget::Body(b)]
    );
}

// ============================================================ iterative paths

fn torus_world() -> (PhysicsWorld, usize) {
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    (w, b)
}

fn cone_world() -> (PhysicsWorld, usize) {
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Cone {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    (w, b)
}

/// Heights `0.5·x` on a 9 × 9 grid of spacing 1: every cell is the plane
/// `y = 0.5·x`.
fn sloped_field_world() -> (PhysicsWorld, usize) {
    let mut heights = vec![];
    for _z in 0..9 {
        for x in 0..9 {
            heights.push(fx(0.5 * f64::from(x)));
        }
    }
    let mut w = world();
    let h = w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        heights,
        9,
        9,
        fx(1.0),
        Vec3Fix::ZERO,
    )));
    (w, h)
}

#[cfg(feature = "std")]
fn sdf_world() -> PhysicsWorld {
    let mut w = world();
    w.sdf_colliders.push(SdfCollider::new_static(
        Box::new(ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let l = (x * x + y * y + z * z).sqrt();
                (x / l, y / l, z / l)
            },
        )),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    w
}

/// An approach 45° off the face normal converges geometrically (each step keeps
/// `1 − cos 45°` of the gap) and stops within `2⁻³²`.
#[test]
fn sphere_cast_oblique_onto_cone_base() {
    let (w, b) = cone_world();
    // oracle: the base disc is the plane y = −0.5 (geometric centre 0.5, half
    // height 1) of radius 1; the centre from (−4, −5, 0) along (1, 1, 0)/√2
    // reaches y = −1 at t = 4·√2, x = 0: contact (0, −0.5, 0), normal −Y.
    assert_hit(
        sphere_cast(&w, [-4.0, -5.0, 0.0], 0.5, [1.0, 1.0, 0.0], 100.0),
        RayTarget::Body(b),
        4.0 * 2f64.sqrt(),
        [0.0, -1.0, 0.0],
        [0.0, -0.5, 0.0],
        ITER,
    );
    // oracle: head on along the line through the apex (0, 1.5, 0) from
    // (−5, 6.5, 0): t = 5·√2 − 0.5, normal (−1, 1, 0)/√2 (within the apex's normal
    // cone, 63.4° from +Y).
    assert_hit(
        sphere_cast(&w, [-5.0, 6.5, 0.0], 0.5, [1.0, -1.0, 0.0], 100.0),
        RayTarget::Body(b),
        5.0 * 2f64.sqrt() - 0.5,
        [-1.0, 1.0, 0.0],
        [0.0, 1.5, 0.0],
        ITER,
    );
}

#[test]
fn capsule_cast_against_iterative_geometry() {
    // oracle: torus (2, 0.5): the segment x ∈ [1.5, 2.5] at y is nearest the ring
    // at (2, y, 0); its distance to the tube y − 0.5 reaches 0.25 at y = 0.75:
    // t = 4.25, contact (2, 0.5, 0).
    let (w, b) = torus_world();
    assert_hit(
        capsule_cast(
            &w,
            [1.5, 5.0, 0.0],
            [2.5, 5.0, 0.0],
            0.25,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::Body(b),
        4.25,
        [0.0, 1.0, 0.0],
        [2.0, 0.5, 0.0],
        ITER,
    );
    // oracle: cone apex (0, 1.5, 0) is the nearest point of the cone to the
    // segment x ∈ [−1, 1] above it (the end (1, y) projects onto the apex along the
    // lateral line): t = 10 − 1.5 − 0.5 = 8.
    let (w, b) = cone_world();
    assert_hit(
        capsule_cast(
            &w,
            [-1.0, 10.0, 0.0],
            [1.0, 10.0, 0.0],
            0.5,
            [0.0, -1.0, 0.0],
            100.0,
        ),
        RayTarget::Body(b),
        8.0,
        [0.0, 1.0, 0.0],
        [0.0, 1.5, 0.0],
        ITER,
    );
    // oracle: cylinder (radius 1): a segment along Z moving +X meets the side at
    // x = −1.5: t = 8.5.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let h = capsule_cast(
        &w,
        [-10.0, 0.0, -0.5],
        [-10.0, 0.0, 0.5],
        0.5,
        [1.0, 0.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::Body(b));
    assert!((h.t.to_f64() - 8.5).abs() < ITER, "t = {}", h.t.to_f64());
    assert_vec(h.normal, [-1.0, 0.0, 0.0], ITER, "normal");
    // oracle: the sloped field (plane y = 0.5·x): a segment along Z (parallel to
    // the plane's level lines) at x = 4 is 0.5 from it when y = 2 + 0.5·√1.25.
    let (w, f) = sloped_field_world();
    let s = 1.25f64.sqrt();
    let h = capsule_cast(
        &w,
        [4.0, 6.0, 3.0],
        [4.0, 6.0, 5.0],
        0.5,
        [0.0, -1.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::StaticCollider(f));
    assert!(
        (h.t.to_f64() - (4.0 - 0.5 * s)).abs() < ITER,
        "t = {}",
        h.t.to_f64()
    );
    assert_vec(h.normal, [-0.5 / s, 1.0 / s, 0.0], ITER, "normal");
}

#[cfg(feature = "std")]
#[test]
fn capsule_cast_against_sdf() {
    let w = sdf_world();
    // oracle: unit sphere field; the segment x ∈ [−1, 1] at height y is nearest
    // the origin at (0, y, 0): field y − 1 = 0.5 at y = 1.5, t = 3.5.
    let h = capsule_cast(
        &w,
        [-1.0, 5.0, 0.0],
        [1.0, 5.0, 0.0],
        0.5,
        [0.0, -1.0, 0.0],
        100.0,
    )
    .expect("hit");
    assert_eq!(h.target, RayTarget::Sdf(0));
    let tol = f64::from(RayFilter::default().sdf.tolerance);
    assert!(
        h.t.to_f64() > 3.5 - tol - 1e-5 && h.t.to_f64() < 3.5 + 1e-5,
        "t = {}",
        h.t.to_f64()
    );
}

#[test]
fn overlap_sphere_against_iterative_geometry() {
    // oracle: torus tube: (2, 0.7, 0) is 0.7 − 0.5 = 0.2 from it.
    let (w, b) = torus_world();
    assert_eq!(overlap_s(&w, [2.0, 0.7, 0.0], 0.15), vec![]);
    assert_eq!(
        overlap_s(&w, [2.0, 0.7, 0.0], 0.25),
        vec![RayTarget::Body(b)]
    );
    // oracle: the hole: the axis point (0, 0, 0) is 2 − 0.5 = 1.5 from the tube.
    assert_eq!(overlap_s(&w, [0.0, 0.0, 0.0], 1.4), vec![]);
    assert_eq!(
        overlap_s(&w, [0.0, 0.0, 0.0], 1.6),
        vec![RayTarget::Body(b)]
    );
    // oracle: cylinder rim (1, 1): (1.3, 1.3, 0) is 0.3·√2 = 0.424 from it.
    let mut w = world();
    let b = shaped(
        &mut w,
        Shape::Cylinder {
            radius: fx(1.0),
            half_height: fx(1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(overlap_s(&w, [1.3, 1.3, 0.0], 0.4), vec![]);
    assert_eq!(
        overlap_s(&w, [1.3, 1.3, 0.0], 0.45),
        vec![RayTarget::Body(b)]
    );
    // oracle: cone apex (0, 1.5, 0): (0, 2, 0) is 0.5 from it.
    let (w, b) = cone_world();
    assert_eq!(overlap_s(&w, [0.0, 2.0, 0.0], 0.49), vec![]);
    assert_eq!(
        overlap_s(&w, [0.0, 2.0, 0.0], 0.51),
        vec![RayTarget::Body(b)]
    );
    // oracle: plane y = 0.5·x: (4, 2.5, 4) is 0.5/√1.25 = 0.447 from it.
    let (w, f) = sloped_field_world();
    assert_eq!(overlap_s(&w, [4.0, 2.5, 4.0], 0.44), vec![]);
    assert_eq!(
        overlap_s(&w, [4.0, 2.5, 4.0], 0.45),
        vec![RayTarget::StaticCollider(f)]
    );
}

#[cfg(feature = "std")]
#[test]
fn overlap_against_sdf() {
    let w = sdf_world();
    // oracle: unit sphere field: (1.4, 0, 0) is 0.4 from it.
    assert_eq!(overlap_s(&w, [1.4, 0.0, 0.0], 0.39), vec![]);
    assert_eq!(
        overlap_s(&w, [1.4, 0.0, 0.0], 0.41),
        vec![RayTarget::Sdf(0)]
    );
    // oracle: the box [0.9, 1.2] × [−0.1, 0.1]² holds (0.95, 0, 0), inside.
    assert_eq!(
        overlap_b(&w, [0.9, -0.1, -0.1], [1.2, 0.1, 0.1]),
        vec![RayTarget::Sdf(0)]
    );
    // oracle: the box [0.75, 1]² × [−0.05, 0.05]: its nearest point to the origin
    // (0.75, 0.75, 0) is 1.0607 > 1 from it, outside the unit sphere.
    assert_eq!(overlap_b(&w, [0.75, 0.75, -0.05], [1.0, 1.0, 0.05]), vec![]);
}

#[test]
fn overlap_aabb_against_iterative_geometry() {
    // oracle: the box [1.9, 2.1] × [−0.1, 0.1]² holds the ring point (2, 0, 0).
    // The box of half 0.5 about the origin: its farthest point in the ring plane,
    // the corner (0.5, ·, 0.5), is 0.707 from the axis, so every ring point is at
    // least 2 − 0.707 = 1.29 > 0.5 from it: no overlap.
    let (w, b) = torus_world();
    assert_eq!(
        overlap_b(&w, [1.9, -0.1, -0.1], [2.1, 0.1, 0.1]),
        vec![RayTarget::Body(b)]
    );
    assert_eq!(overlap_b(&w, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]), vec![]);
    // oracle: plane y = 0.5·x over x ∈ [3.9, 4.1]: heights [1.95, 2.05].
    let (w, f) = sloped_field_world();
    assert_eq!(overlap_b(&w, [3.9, 1.0, 3.9], [4.1, 1.9, 4.1]), vec![]);
    assert_eq!(
        overlap_b(&w, [3.9, 1.0, 3.9], [4.1, 1.96, 4.1]),
        vec![RayTarget::StaticCollider(f)]
    );
    // oracle: sphere body R = 1: the box from (0.6, 0.6, 0.6) is 1.039 from the
    // centre, from (0.5, 0.5, 0.5) 0.866.
    let mut w = world();
    let s = sphere_body(&mut w, Vec3Fix::ZERO, 1.0);
    assert_eq!(overlap_b(&w, [0.6, 0.6, 0.6], [1.0, 1.0, 1.0]), vec![]);
    assert_eq!(
        overlap_b(&w, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        vec![RayTarget::Body(s)]
    );
    // oracle: compound children: sphere (0, 0, ±3) radius 0.5 and capsule along X
    // radius 0.5; a box corner (0.4, 0.4) off either axis is 0.566 > 0.5 from it,
    // (0.3, 0.3) is 0.424.
    let mut w = world();
    let mut c = CompoundShape::new();
    for z in [3.0, -3.0] {
        c.add_sphere(
            Sphere::new(Vec3Fix::ZERO, fx(0.5)),
            v3(0.0, 0.0, z),
            QuatFix::IDENTITY,
        );
    }
    c.add_capsule(
        Capsule::new(v3(-2.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let b = w
        .add_compound_body(&c, Fix128::ONE, Vec3Fix::ZERO)
        .expect("valid compound");
    assert_eq!(overlap_b(&w, [0.4, 0.4, 2.6], [1.0, 1.0, 3.0]), vec![]);
    assert_eq!(
        overlap_b(&w, [0.3, 0.3, 2.6], [1.0, 1.0, 3.0]),
        vec![RayTarget::Body(b)]
    );
    assert_eq!(overlap_b(&w, [-1.0, 0.4, 0.4], [1.0, 1.0, 1.0]), vec![]);
    assert_eq!(
        overlap_b(&w, [-1.0, 0.3, 0.3], [1.0, 1.0, 1.0]),
        vec![RayTarget::Body(b)]
    );
}
