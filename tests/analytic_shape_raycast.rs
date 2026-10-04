//! Oracles for `PhysicsWorld::cast_ray` and its variants: rays against the
//! geometry the world collides with, compared with closed forms.
//!
//! # Expected values
//!
//! Every expected distance and normal is written from the geometry by hand (the
//! closed form is in each test's doc), or for a turned box by an independent `f64`
//! slab test; none calls the code under test.
//!
//! # Tolerances
//!
//! - `EXACT = 1e-12`: shapes that are not turned. Each `Fix128` operation
//!   truncates by at most `2⁻⁶⁴ ≈ 5.4e-20`; a query is a few hundred operations on
//!   numbers below 100 (one `sqrt`, a handful of divisions), so the error is below
//!   `1e-15`. `1e-12` leaves room without hiding a wrong formula, which errs by
//!   `1e-2` or more in every scene here.
//! - `TURNED = 1e-9`: shapes turned by a quaternion from
//!   `QuatFix::from_axis_angle`, whose CORDIC sine and cosine are good to about
//!   `1e-14` (measured worst `7e-15` over `[−π, π]`, rotated unit vector `2.4e-15`);
//!   the error is carried through distances of 10 and the hull's plane build.
//! - SDF: a hit is where the `f32` field drops below `SdfCcdConfig::tolerance`
//!   (`1e-3`), so `t ∈ [d − tol, d + F32]` with `F32 = 1e-5` for `f32` rounding of
//!   positions near 5.
//!
//! # Against the bounding-sphere API
//!
//! Each shape has a scene where `PhysicsWorld::raycast` (bounding sphere) reports
//! a hit the shape does not have, and the static colliders and SDF have one where
//! it reports nothing; the new query is asserted to answer correctly in both.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::box_collider::OrientedBox;
use alice_physics::collider::{Capsule, ConvexHull, Sphere};
use alice_physics::compound::CompoundShape;
use alice_physics::heightfield::HeightField;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::plane_collider::PlaneCollider;
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};
use alice_physics::shape::Shape;
use alice_physics::shape_raycast::{RayFilter, RayTarget, WorldRayHit};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use alice_physics::static_collider::StaticCollider;
use alice_physics::trimesh::TriMesh;

const EXACT: f64 = 1e-12;
const TURNED: f64 = 1e-9;

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
    PhysicsWorld::new(SolverConfig::default())
}

fn cast(w: &PhysicsWorld, o: [f64; 3], d: [f64; 3], max: f64) -> Option<WorldRayHit> {
    w.cast_ray(
        v3(o[0], o[1], o[2]),
        v3(d[0], d[1], d[2]),
        fx(max),
        &RayFilter::default(),
    )
}

/// The bounding-sphere answer of the existing API, for the comparison scenes.
fn old_cast(w: &PhysicsWorld, o: [f64; 3], d: [f64; 3], max: f64) -> Option<(usize, Fix128)> {
    w.raycast(v3(o[0], o[1], o[2]), v3(d[0], d[1], d[2]), fx(max))
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

#[track_caller]
fn assert_hit(hit: Option<WorldRayHit>, t: f64, normal: [f64; 3], tol: f64) -> WorldRayHit {
    let h = hit.unwrap_or_else(|| panic!("expected a hit at t = {t}, got none"));
    assert!(
        (h.t.to_f64() - t).abs() < tol,
        "t = {} but the closed form is {t}",
        h.t.to_f64()
    );
    let n = f3(h.normal);
    let want = unit(normal);
    for k in 0..3 {
        assert!(
            (n[k] - want[k]).abs() < tol,
            "normal {n:?} but the closed form is {want:?}"
        );
    }
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

fn rot_y(deg: f64) -> QuatFix {
    QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, fx(deg.to_radians()))
}

// ---------------------------------------------------------------- sphere body

/// A body with only a collision radius is a sphere: from `(−10, 0.3, 0)` along `+X`
/// to a unit sphere at the origin, `t = 10 − √(1 − 0.3²)`, normal `(−√0.91, 0.3, 0)`.
#[test]
fn sphere_body_matches_closed_form() {
    let mut w = world();
    let i = w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    let h = assert_hit(
        cast(&w, [-10.0, 0.3, 0.0], [1.0, 0.0, 0.0], 100.0),
        10.0 - 0.91f64.sqrt(),
        [-(0.91f64.sqrt()), 0.3, 0.0],
        EXACT,
    );
    assert_eq!(h.target, RayTarget::Body(i));
    assert_eq!(h.body, Some(i));
    let p = f3(h.point);
    assert!((p[0] + 0.91f64.sqrt()).abs() < EXACT && (p[1] - 0.3).abs() < EXACT);
}

/// A body with neither radius nor shape takes part in no collision: not hit.
#[test]
fn body_without_collision_radius_is_not_hit() {
    let mut w = world();
    w.add_body(RigidBody::new_static(Vec3Fix::ZERO));
    assert_eq!(cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- box

/// Box of half-extents (2, 1, 0.5) at the origin: from `(−10, 0.4, 0.2)` along `+X`
/// the face `x = −2` at `t = 8`, normal `−X`; from above, `y = 1` at `t = 9`.
#[test]
fn box_matches_closed_form() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(2.0, 1.0, 0.5),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [-10.0, 0.4, 0.2], [1.0, 0.0, 0.0], 100.0),
        8.0,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [1.5, 10.0, -0.3], [0.0, -1.0, 0.0], 100.0),
        9.0,
        [0.0, 1.0, 0.0],
        EXACT,
    );
}

/// The bounding sphere of a (2, 0.2, 0.2) box has radius √4.08 ≈ 2.02: a ray along
/// `X` at `y = 1` meets the sphere but passes 0.8 above the box.
#[test]
fn box_corner_gap_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(2.0, 0.2, 0.2),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert!(old_cast(&w, [-10.0, 1.0, 0.0], [1.0, 0.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [-10.0, 1.0, 0.0], [1.0, 0.0, 0.0], 100.0), None);
}

/// Independent `f64` slab test of a box turned about `Y` by `deg`.
fn f64_turned_box(o: [f64; 3], d: [f64; 3], half: [f64; 3], deg: f64) -> (f64, [f64; 3]) {
    let (s, c) = deg.to_radians().sin_cos();
    // Local = R(−θ) world about Y: x' = c·x − s·z, z' = s·x + c·z.
    let to_local = |v: [f64; 3]| [c * v[0] - s * v[2], v[1], s * v[0] + c * v[2]];
    let to_world = |v: [f64; 3]| [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
    let (ol, dl) = (to_local(o), to_local(d));
    let mut t_in = f64::NEG_INFINITY;
    let mut n = [0.0; 3];
    let mut t_out = f64::INFINITY;
    for k in 0..3 {
        let (a, b) = ((-half[k] - ol[k]) / dl[k], (half[k] - ol[k]) / dl[k]);
        let (lo, hi, sign) = if a < b { (a, b, -1.0) } else { (b, a, 1.0) };
        if lo > t_in {
            t_in = lo;
            n = [0.0; 3];
            n[k] = sign;
        }
        t_out = t_out.min(hi);
    }
    assert!(
        t_in <= t_out && t_in > 0.0,
        "reference scene must hit from outside"
    );
    (t_in, to_world(n))
}

/// A (1.5, 1, 0.5) box turned 30° about `Y`, hit from `(−10, 0.2, 0.7)` along a
/// skew direction: distance and normal agree with the `f64` slab test.
#[test]
fn turned_box_matches_f64_reference() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.5, 1.0, 0.5),
        },
        Vec3Fix::ZERO,
        rot_y(30.0),
    );
    let o = [-10.0, 0.2, 0.7];
    let d = unit([1.0, 0.05, -0.1]);
    let (t, n) = f64_turned_box(o, d, [1.5, 1.0, 0.5], 30.0);
    assert_hit(cast(&w, o, d, 100.0), t, n, TURNED);
}

// ---------------------------------------------------------------- cylinder

/// Cylinder radius 1, half-height 2: the top cap from above at `t = 8` (normal
/// `+Y`); the side from `(−10, 1.5, 0.6)` along `+X` at `x = −0.8`, `t = 9.2`,
/// normal `(−0.8, 0, 0.6)`.
#[test]
fn cylinder_matches_closed_form() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Cylinder {
            radius: Fix128::ONE,
            half_height: fx(2.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [0.3, 10.0, 0.4], [0.0, -1.0, 0.0], 100.0),
        8.0,
        [0.0, 1.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [-10.0, 1.5, 0.6], [1.0, 0.0, 0.0], 100.0),
        9.2,
        [-0.8, 0.0, 0.6],
        EXACT,
    );
    // From below: the bottom cap.
    assert_hit(
        cast(&w, [0.1, -10.0, 0.0], [0.0, 1.0, 0.0], 100.0),
        8.0,
        [0.0, -1.0, 0.0],
        EXACT,
    );
}

/// The bounding sphere has radius √5 ≈ 2.24; a ray along `X` at `y = 2.1` is above
/// the cap.
#[test]
fn cylinder_above_cap_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Cylinder {
            radius: Fix128::ONE,
            half_height: fx(2.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert!(old_cast(&w, [-10.0, 2.1, 0.0], [1.0, 0.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [-10.0, 2.1, 0.0], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- cone

/// Cone radius 1, half-height 1, centre of mass at the origin: its centroid is
/// `h/2` below the geometric centre, so the apex is at `y = 1.5` and the base at
/// `y = −0.5`, and the lateral radius is `(1.5 − y)/2`. Down at `x = 0.2`:
/// `y = 1.1`, `t = 8.9`, normal ∝ `(x, k²(1.5 − y), 0) = (0.2, 0.1, 0)`. Up at
/// `x = 0.3`: the base, `t = 9.5`, normal `−Y`.
#[test]
fn cone_matches_closed_form() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Cone {
            radius: Fix128::ONE,
            half_height: Fix128::ONE,
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [0.2, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        8.9,
        [2.0, 1.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [0.3, -10.0, 0.0], [0.0, 1.0, 0.0], 100.0),
        9.5,
        [0.0, -1.0, 0.0],
        EXACT,
    );
}

/// Bounding radius 1.5 (the apex); a ray along `X` at `y = 1.3, z = 0.5` is within
/// it but the cone there has radius 0.1.
#[test]
fn cone_beside_tip_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Cone {
            radius: Fix128::ONE,
            half_height: Fix128::ONE,
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert!(old_cast(&w, [-10.0, 1.3, 0.5], [1.0, 0.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [-10.0, 1.3, 0.5], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- ellipsoid

/// Ellipsoid (3, 1, 2): along `+X` at `y = 0.5`, `x = −3√0.75`, normal ∝
/// `(x/9, y/1, 0)`.
#[test]
fn ellipsoid_matches_closed_form() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Ellipsoid {
            radii: v3(3.0, 1.0, 2.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let x = -3.0 * 0.75f64.sqrt();
    assert_hit(
        cast(&w, [-10.0, 0.5, 0.0], [1.0, 0.0, 0.0], 100.0),
        10.0 + x,
        [x / 9.0, 0.5, 0.0],
        EXACT,
    );
}

/// Bounding radius 3; a ray along `X` at `z = 2.5` is outside the `z` semi-axis 2.
#[test]
fn ellipsoid_side_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Ellipsoid {
            radii: v3(3.0, 1.0, 2.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert!(old_cast(&w, [-10.0, 0.0, 2.5], [1.0, 0.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [-10.0, 0.0, 2.5], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- wedge

/// Wedge 2 × 2 × 2 with its geometric centre at the origin (centre of mass at
/// `y = −1/3`): base `y = −1`, apex edge `y = 1`, right face `y = 1 − 2x`. Down at
/// `x = 0.5`: `y = 0`, `t = 10`, normal ∝ `(h, w/2, 0) = (2, 1, 0)`. Up: the base,
/// `t = 9`, normal `−Y`.
#[test]
fn wedge_matches_closed_form() {
    let mut w = world();
    let shape = Shape::Wedge {
        width: fx(2.0),
        height: fx(2.0),
        depth: fx(2.0),
    };
    let com = shape.center_of_mass_offset();
    shaped(&mut w, shape, com, QuatFix::IDENTITY);
    assert_hit(
        cast(&w, [0.5, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        10.0,
        [2.0, 1.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [0.2, -10.0, 0.3], [0.0, 1.0, 0.0], 100.0),
        9.0,
        [0.0, -1.0, 0.0],
        EXACT,
    );
}

/// Bounding radius ≈ 1.56 (a base corner from the centroid); a ray along `X` at
/// `z = 1.2` is outside the depth ±1.
#[test]
fn wedge_beyond_depth_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    let shape = Shape::Wedge {
        width: fx(2.0),
        height: fx(2.0),
        depth: fx(2.0),
    };
    let com = shape.center_of_mass_offset();
    shaped(&mut w, shape, com, QuatFix::IDENTITY);
    assert!(old_cast(&w, [-10.0, 0.0, 1.2], [1.0, 0.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [-10.0, 0.0, 1.2], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- torus

/// Torus R = 2, r = 0.5 around `Y`: along `+X` the outer equator `x = −2.5`,
/// `t = 7.5`, normal `−X`; down at `x = 2` the tube top `y = 0.5`, `t = 9.5`,
/// normal `+Y`.
#[test]
fn torus_matches_closed_form() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0),
        7.5,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [2.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        9.5,
        [0.0, 1.0, 0.0],
        EXACT,
    );
    // From the centre of the hole at 10° above X: the point s·(cos a, sin a, 0) is
    // on the tube when (s cos a − 2)² + (s sin a)² = 0.25, i.e.
    // s² − 4s·cos a + 3.75 = 0, s = 2cos a − √(4cos²a − 3.75); the normal points
    // from the tube centre (2, 0, 0) to the hit.
    let (sa, ca) = 10f64.to_radians().sin_cos();
    let s = 2.0 * ca - (4.0 * ca * ca - 3.75).sqrt();
    assert_hit(
        cast(&w, [0.0, 0.0, 0.0], [ca, sa, 0.0], 100.0),
        s,
        [s * ca - 2.0, s * sa, 0.0],
        TURNED,
    );
}

/// The ray down the hole meets the bounding sphere (radius 2.5) but not the torus.
#[test]
fn torus_hole_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Torus {
            major_radius: fx(2.0),
            minor_radius: fx(0.5),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert!(old_cast(&w, [0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- compound children

fn compound_body(w: &mut PhysicsWorld, c: &CompoundShape) -> usize {
    w.add_compound_body(c, Fix128::ONE, Vec3Fix::ZERO)
        .expect("valid compound")
}

/// Capsule child from (−2, 0, 0) to (2, 0, 0), radius 0.5: down at `x = 1` the side,
/// `t = 9.5`, `+Y`; along `+X` the end cap `x = −2.5`, `t = 7.5`, `−X`; down at
/// `x = 2.3` the cap sphere, `y = 0.4`, `t = 9.6`, normal `(0.3, 0.4, 0)/0.5`.
#[test]
fn capsule_child_matches_closed_form() {
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(-2.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    assert_hit(
        cast(&w, [1.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        9.5,
        [0.0, 1.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0),
        7.5,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_hit(
        cast(&w, [2.3, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        9.6,
        [0.3, 0.4, 0.0],
        EXACT,
    );
}

/// The capsule's bounding sphere (radius 2.5) contains `(0, y, 1)`; the capsule
/// does not.
#[test]
fn capsule_child_beside_side_is_hit_by_bounding_sphere_but_not_by_shape() {
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(-2.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    assert!(old_cast(&w, [0.0, 10.0, 1.0], [0.0, -1.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [0.0, 10.0, 1.0], [0.0, -1.0, 0.0], 100.0), None);
}

/// Two sphere children of radius 0.5 at `x = ±3`: down at `x = 3` the sphere top,
/// `t = 9.5`; down at `x = 0` the gap between them (inside the bounding sphere).
#[test]
fn sphere_children_and_the_gap_between_them() {
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(0.5)),
        v3(3.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    c.add_sphere(
        Sphere::new(Vec3Fix::ZERO, fx(0.5)),
        v3(-3.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    assert_hit(
        cast(&w, [3.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        9.5,
        [0.0, 1.0, 0.0],
        EXACT,
    );
    assert!(old_cast(&w, [0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0), None);
    // Along X both spheres lie on the ray: the nearer one.
    assert_hit(
        cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0),
        6.5,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
}

/// Box child (1, 1, 1) turned 90° about `Y` inside the compound and offset to
/// `x = 2` with a second box at `x = −2` (centre of mass at 0): down at `x = 2.5`
/// the top `y = 1`, `t = 9`; along `X` at `y = 0.5` the near face `x = −3`. Then
/// two (2, 1, 0.5) boxes at `x = ±3`, the one at `+3` turned 90° about `Y` so it
/// spans `x ∈ [2.5, 3.5]`, `z ∈ [−2, 2]`: down at `(3.2, ·, 1.5)` its top at
/// `t = 9` (unturned it would span `z ∈ [−0.5, 0.5]` and be missed).
#[test]
fn box_children_match_closed_form() {
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), QuatFix::IDENTITY),
        v3(2.0, 0.0, 0.0),
        rot_y(90.0),
    );
    c.add_box(
        OrientedBox::new(Vec3Fix::ZERO, v3(1.0, 1.0, 1.0), QuatFix::IDENTITY),
        v3(-2.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    assert_hit(
        cast(&w, [2.5, 10.0, 0.2], [0.0, -1.0, 0.0], 100.0),
        9.0,
        [0.0, 1.0, 0.0],
        TURNED,
    );
    assert_hit(
        cast(&w, [-10.0, 0.5, 0.0], [1.0, 0.0, 0.0], 100.0),
        7.0,
        [-1.0, 0.0, 0.0],
        TURNED,
    );

    let mut w = world();
    let mut c = CompoundShape::new();
    let slab = OrientedBox::new(Vec3Fix::ZERO, v3(2.0, 1.0, 0.5), QuatFix::IDENTITY);
    c.add_box(slab, v3(3.0, 0.0, 0.0), rot_y(90.0));
    c.add_box(slab, v3(-3.0, 0.0, 0.0), QuatFix::IDENTITY);
    compound_body(&mut w, &c);
    assert_hit(
        cast(&w, [3.2, 10.0, 1.5], [0.0, -1.0, 0.0], 100.0),
        9.0,
        [0.0, 1.0, 0.0],
        TURNED,
    );
    assert_eq!(
        cast(&w, [3.2, 10.0, 0.0 + 2.2], [0.0, -1.0, 0.0], 100.0),
        None
    );
}

/// Octahedron hull `|x| + |y| + |z| ≤ 1`: down at `(0.2, ·, 0.1)` the face
/// `x + y + z = 1` at `y = 0.7`, `t = 9.3`, normal `(1, 1, 1)/√3`. At `(0.6, ·, 0.6)`
/// it is missed (`|x| + |z| = 1.2`) though inside the bounding sphere (√3).
#[test]
fn convex_hull_child_matches_closed_form() {
    let mut w = world();
    let mut c = CompoundShape::new();
    let one = Fix128::ONE;
    let z = Fix128::ZERO;
    c.add_convex_hull(
        ConvexHull::new(vec![
            Vec3Fix::new(one, z, z),
            Vec3Fix::new(-one, z, z),
            Vec3Fix::new(z, one, z),
            Vec3Fix::new(z, -one, z),
            Vec3Fix::new(z, z, one),
            Vec3Fix::new(z, z, -one),
        ]),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    assert_hit(
        cast(&w, [0.2, 10.0, 0.1], [0.0, -1.0, 0.0], 100.0),
        9.3,
        [1.0, 1.0, 1.0],
        TURNED,
    );
    assert!(old_cast(&w, [0.6, 10.0, 0.6], [0.0, -1.0, 0.0], 100.0).is_some());
    assert_eq!(cast(&w, [0.6, 10.0, 0.6], [0.0, -1.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- static colliders

/// Plane `y = 0` from `(0, 10, 0)` along `(1, −1, 0)`: `t = 10√2`, normal `+Y`; from
/// below the normal faces the ray (`−Y`). The bounding-sphere API sees no static
/// collider at all.
#[test]
fn plane_matches_closed_form_and_old_api_misses_it() {
    let mut w = world();
    let j = w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    let h = assert_hit(
        cast(&w, [0.0, 10.0, 0.0], [1.0, -1.0, 0.0], 100.0),
        10.0 * 2f64.sqrt(),
        [0.0, 1.0, 0.0],
        EXACT,
    );
    assert_eq!(h.target, RayTarget::StaticCollider(j));
    assert_eq!(h.body, None);
    assert_hit(
        cast(&w, [0.0, -5.0, 0.0], [0.0, 1.0, 0.0], 100.0),
        5.0,
        [0.0, -1.0, 0.0],
        EXACT,
    );
    assert_eq!(
        old_cast(&w, [0.0, 10.0, 0.0], [0.0, -1.0, 0.0], 100.0),
        None
    );
}

/// A height field sampling the plane `y = 0.5x + 1` on an 11 × 11 grid of spacing 1
/// (bilinear interpolation reproduces a plane exactly). Down at `(3.3, ·, 4.7)`:
/// `y = 2.65`, `t = 7.35`, normal `(−0.5, 1, 0)/√1.25`. Along `(2, −1, 0.5)` from
/// `(−1, 6, 2)`: `6 − s = 0.5(−1 + 2s) + 1` ⇒ `s = 2.75`, `t = 2.75·√5.25`.
#[test]
fn sloped_heightfield_matches_closed_form() {
    let mut w = world();
    let mut heights = Vec::new();
    for _z in 0..11 {
        for x in 0..11 {
            heights.push(fx(0.5 * x as f64 + 1.0));
        }
    }
    let field = HeightField::new(heights, 11, 11, Fix128::ONE, Vec3Fix::ZERO);
    w.add_static_collider(StaticCollider::HeightField(field.clone()));
    let n = [-0.5, 1.0, 0.0];
    let h = assert_hit(
        cast(&w, [3.3, 10.0, 4.7], [0.0, -1.0, 0.0], 100.0),
        7.35,
        n,
        EXACT,
    );
    // The hit lies on the surface the field's own sampler describes.
    assert!(
        (h.point.y - field.sample_height(h.point.x, h.point.z))
            .to_f64()
            .abs()
            < EXACT
    );
    assert_hit(
        cast(&w, [-1.0, 6.0, 2.0], [2.0, -1.0, 0.5], 100.0),
        2.75 * 5.25f64.sqrt(),
        n,
        EXACT,
    );
    assert_eq!(
        old_cast(&w, [3.3, 10.0, 4.7], [0.0, -1.0, 0.0], 100.0),
        None
    );
    // Outside the footprint: nothing.
    assert_eq!(cast(&w, [12.0, 10.0, 4.7], [0.0, -1.0, 0.0], 100.0), None);
}

/// The plane `y = 0.5x + 0.3z + 1` sampled on the same grid (slope along both axes,
/// so every term of the cell quadratic is non-zero): along `(2, −1, 0.5)` from
/// `(−1, 6, 2)`, `6 − s = 0.5(−1 + 2s) + 0.3(2 + 0.5s) + 1` ⇒ `s = 4.9 / 2.15`,
/// `t = s·√5.25`, normal `(−0.5, 1, −0.3)/|·|`.
#[test]
fn heightfield_sloped_along_both_axes_matches_closed_form() {
    let mut w = world();
    let mut heights = Vec::new();
    for z in 0..11 {
        for x in 0..11 {
            heights.push(fx(0.5 * x as f64 + 0.3 * z as f64 + 1.0));
        }
    }
    w.add_static_collider(StaticCollider::HeightField(HeightField::new(
        heights,
        11,
        11,
        Fix128::ONE,
        Vec3Fix::ZERO,
    )));
    let s = 4.9 / 2.15;
    assert_hit(
        cast(&w, [-1.0, 6.0, 2.0], [2.0, -1.0, 0.5], 100.0),
        s * 5.25f64.sqrt(),
        [-0.5, 1.0, -0.3],
        EXACT,
    );
}

/// One cell with corner heights 0, 0, 0, 1 is the saddle `y = x·z`. Along the
/// diagonal at height 0.25 from outside the footprint, `(−1, 0.25, −1) + s(1, 0, 1)`
/// reaches `s² … ` the surface where `(s − 1)² = 0.25`, `s = 1.5`, `t = 1.5√2`.
/// The normal is the sampler's: ∝ `(−z, 1, −x) = (−0.5, 1, −0.5)` at (0.5, 0.5).
#[test]
fn curved_heightfield_cell_is_solved_exactly() {
    let mut w = world();
    let field = HeightField::new(
        vec![Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ONE],
        2,
        2,
        Fix128::ONE,
        Vec3Fix::ZERO,
    );
    w.add_static_collider(StaticCollider::HeightField(field));
    assert_hit(
        cast(&w, [-1.0, 0.25, -1.0], [1.0, 0.0, 1.0], 100.0),
        1.5 * 2f64.sqrt(),
        [-0.5, 1.0, -0.5],
        EXACT,
    );
}

/// Two triangles making the square `[0, 2]²` at `y = 0`: down at `(0.5, 3, 0.7)`
/// `t = 3`, normal `+Y`.
#[test]
fn trimesh_matches_closed_form() {
    let mut w = world();
    let verts = [
        v3(0.0, 0.0, 0.0),
        v3(2.0, 0.0, 0.0),
        v3(2.0, 0.0, 2.0),
        v3(0.0, 0.0, 2.0),
    ];
    let mesh = TriMesh::from_indexed(&verts, &[0, 1, 2, 0, 2, 3]);
    w.add_static_collider(StaticCollider::TriMesh(mesh));
    assert_hit(
        cast(&w, [0.5, 3.0, 0.7], [0.0, -1.0, 0.0], 100.0),
        3.0,
        [0.0, 1.0, 0.0],
        EXACT,
    );
    assert_eq!(old_cast(&w, [0.5, 3.0, 0.7], [0.0, -1.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- SDF

fn unit_sphere_sdf() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

/// A unit-sphere SDF at `(5, 0, 0)`: along `+X` the surface is at `t = 4`; the march
/// stops within `tolerance = 1e-3` before it. The bounding-sphere API does not see
/// SDF colliders.
#[test]
fn sdf_sphere_matches_closed_form_within_tolerance() {
    const F32: f64 = 1e-5;
    let mut w = world();
    let k = w.add_sdf_collider(SdfCollider::new_static(
        Box::new(unit_sphere_sdf()),
        v3(5.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    ));
    let h = cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0).expect("hit");
    let t = h.t.to_f64();
    assert!((4.0 - 1e-3 - F32..=4.0 + F32).contains(&t), "t = {t}");
    let n = f3(h.normal);
    assert!(
        (n[0] + 1.0).abs() < F32 && n[1].abs() < F32 && n[2].abs() < F32,
        "normal {n:?}"
    );
    assert_eq!(h.target, RayTarget::Sdf(k));
    assert_eq!(h.body, None);
    assert_eq!(old_cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- inside origins

/// A ray starting strictly inside a solid hits at `t = 0`, at the origin, with
/// normal `−direction` (module convention), for every solid kind.
#[test]
fn origin_inside_every_solid_hits_at_zero_against_the_direction() {
    let d = [0.0, 0.0, 1.0];
    let check = |w: &PhysicsWorld, o: [f64; 3]| {
        let h = cast(w, o, d, 100.0).expect("inside is a hit");
        assert_eq!(h.t, Fix128::ZERO, "origin {o:?}");
        assert_eq!(h.point, v3(o[0], o[1], o[2]));
        assert_eq!(h.normal, -Vec3Fix::UNIT_Z);
    };
    let shapes = [
        (
            Shape::Box {
                half_extents: v3(1.0, 1.0, 1.0),
            },
            [0.2, 0.1, 0.0],
        ),
        (
            Shape::Cylinder {
                radius: Fix128::ONE,
                half_height: Fix128::ONE,
            },
            [0.2, 0.1, 0.0],
        ),
        (
            Shape::Cone {
                radius: Fix128::ONE,
                half_height: Fix128::ONE,
            },
            [0.1, 0.2, 0.0],
        ),
        (
            Shape::Ellipsoid {
                radii: v3(1.0, 2.0, 3.0),
            },
            [0.2, 0.1, 0.0],
        ),
        (
            Shape::Wedge {
                width: fx(2.0),
                height: fx(2.0),
                depth: fx(2.0),
            },
            [0.0, 0.0, 0.0],
        ),
        (
            Shape::Torus {
                major_radius: fx(2.0),
                minor_radius: fx(0.5),
            },
            [2.0, 0.1, 0.0],
        ),
    ];
    for (shape, o) in shapes {
        let mut w = world();
        shaped(&mut w, shape, Vec3Fix::ZERO, QuatFix::IDENTITY);
        check(&w, o);
    }
    let mut w = world();
    w.add_body_with_radius(RigidBody::new_static(Vec3Fix::ZERO), Fix128::ONE);
    check(&w, [0.3, 0.0, 0.0]);
    let mut w = world();
    let mut c = CompoundShape::new();
    c.add_capsule(
        Capsule::new(v3(-2.0, 0.0, 0.0), v3(2.0, 0.0, 0.0), fx(0.5)),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    compound_body(&mut w, &c);
    check(&w, [1.5, 0.1, 0.0]);
    let mut w = world();
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(unit_sphere_sdf()),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    check(&w, [0.2, 0.0, 0.0]);
}

// ---------------------------------------------------------------- degenerate input

/// Zero direction, `max_t ≤ 0`, an empty world: no hit, an empty list, `false`.
#[test]
fn degenerate_queries_return_nothing() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    let f = RayFilter::default();
    let o = v3(-10.0, 0.0, 0.0);
    for (d, max) in [
        (Vec3Fix::ZERO, fx(100.0)),
        (Vec3Fix::UNIT_X, Fix128::ZERO),
        (Vec3Fix::UNIT_X, fx(-5.0)),
    ] {
        assert_eq!(w.cast_ray(o, d, max, &f), None);
        assert!(w.cast_ray_all(o, d, max, &f).is_empty());
        assert!(!w.cast_ray_any(o, d, max, &f));
        assert!(w.ray_caster(f).candidates(o, d, max).is_empty());
    }
    let empty = world();
    assert_eq!(empty.cast_ray(o, Vec3Fix::UNIT_X, fx(100.0), &f), None);
    assert!(empty
        .cast_ray_all(o, Vec3Fix::UNIT_X, fx(100.0), &f)
        .is_empty());
}

/// `max_t` is inclusive: the box face at exactly `t = 9` is hit with `max_t = 9`
/// and missed with `max_t = 8.5`.
#[test]
fn max_t_is_an_inclusive_bound() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 9.0),
        9.0,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_eq!(cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 8.5), None);
    // The same for a quadratic surface: a unit sphere at x = 10 is met at t = 9.
    let mut w = world();
    w.add_body_with_radius(RigidBody::new_static(v3(10.0, 0.0, 0.0)), Fix128::ONE);
    assert_hit(
        cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 9.0),
        9.0,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_eq!(cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 8.5), None);
}

/// `max_t = 0` is an empty segment even from inside a solid (whose hit would be at
/// `t = 0`): no hit; with any positive `max_t` the inside hit is reported.
#[test]
fn zero_max_t_is_empty_even_inside_a_solid() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_eq!(cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0), None);
    assert!(!w.cast_ray_any(
        Vec3Fix::ZERO,
        Vec3Fix::UNIT_X,
        Fix128::ZERO,
        &RayFilter::default()
    ));
    assert_eq!(
        cast(&w, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1e-6).map(|h| h.t),
        Some(Fix128::ZERO)
    );
}

/// A ray lying in the top face `y = 1` of a box enters through the side face it
/// meets (`x = −1`, `t = 9`); just above the face it misses. A ray in a plane
/// collider or in a flat height field does not hit it.
#[test]
fn rays_parallel_to_faces() {
    let mut w = world();
    shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    );
    assert_hit(
        cast(&w, [-10.0, 1.0, 0.0], [1.0, 0.0, 0.0], 100.0),
        9.0,
        [-1.0, 0.0, 0.0],
        EXACT,
    );
    assert_eq!(cast(&w, [-10.0, 1.0001, 0.0], [1.0, 0.0, 0.0], 100.0), None);

    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_Y,
        Fix128::ZERO,
    )));
    assert_eq!(cast(&w, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0), None);

    let mut w = world();
    w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
        4,
        4,
        Fix128::ONE,
        Vec3Fix::ZERO,
        Fix128::ZERO,
    )));
    assert_eq!(cast(&w, [-1.0, 0.0, 1.5], [1.0, 0.0, 0.0], 100.0), None);
    // A grid with a single row has no cell, so no surface.
    let mut w = world();
    w.add_static_collider(StaticCollider::HeightField(HeightField::flat(
        4,
        1,
        Fix128::ONE,
        Vec3Fix::ZERO,
        Fix128::ZERO,
    )));
    assert_eq!(cast(&w, [1.0, 5.0, 0.0], [0.0, -1.0, 0.0], 100.0), None);
}

// ---------------------------------------------------------------- query modes, filter, BVH

/// Three unit boxes on the `X` axis at 0, 5, 10: `all` returns the three entries in
/// order, `closest` the first, `any` is true; off the axis all three are empty.
#[test]
fn closest_all_any_agree() {
    let mut w = world();
    let ids: Vec<usize> = [10.0, 0.0, 5.0]
        .iter()
        .map(|&x| {
            shaped(
                &mut w,
                Shape::Box {
                    half_extents: v3(1.0, 1.0, 1.0),
                },
                v3(x, 0.0, 0.0),
                QuatFix::IDENTITY,
            )
        })
        .collect();
    let f = RayFilter::default();
    let o = v3(-10.0, 0.0, 0.0);
    let all = w.cast_ray_all(o, Vec3Fix::UNIT_X, fx(100.0), &f);
    let ts: Vec<f64> = all.iter().map(|h| h.t.to_f64()).collect();
    assert_eq!(ts, vec![9.0, 14.0, 19.0]);
    let order: Vec<Option<usize>> = all.iter().map(|h| h.body).collect();
    assert_eq!(order, vec![Some(ids[1]), Some(ids[2]), Some(ids[0])]);
    assert_eq!(w.cast_ray(o, Vec3Fix::UNIT_X, fx(100.0), &f), Some(all[0]));
    assert!(w.cast_ray_any(o, Vec3Fix::UNIT_X, fx(100.0), &f));
    let off = v3(-10.0, 5.0, 0.0);
    assert!(!w.cast_ray_any(off, Vec3Fix::UNIT_X, fx(100.0), &f));
}

/// Layer mask, excluded body, sensors, static and SDF switches each remove exactly
/// what they name.
#[test]
fn filter_removes_what_it_names() {
    let mut w = world();
    let near = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(0.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let far = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(5.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    w.set_body_filter(
        near,
        alice_physics::filter::CollisionFilter::new(1 << 3, u32::MAX),
    );
    let o = v3(-10.0, 0.0, 0.0);
    let d = Vec3Fix::UNIT_X;
    let m = fx(100.0);
    let hit_body = |w: &PhysicsWorld, f: RayFilter| w.cast_ray(o, d, m, &f).and_then(|h| h.body);
    assert_eq!(hit_body(&w, RayFilter::default()), Some(near));
    assert_eq!(
        hit_body(&w, RayFilter::default().with_layer_mask(1)),
        Some(far)
    );
    assert_eq!(
        hit_body(&w, RayFilter::default().excluding_body(near)),
        Some(far)
    );

    w.bodies[near].is_sensor = true;
    assert_eq!(hit_body(&w, RayFilter::default()), Some(far));
    assert_eq!(
        hit_body(&w, RayFilter::default().with_sensors(true)),
        Some(near)
    );

    let mut w = world();
    w.add_static_collider(StaticCollider::Plane(PlaneCollider::new(
        Vec3Fix::UNIT_X,
        Fix128::ZERO,
    )));
    w.add_sdf_collider(SdfCollider::new_static(
        Box::new(unit_sphere_sdf()),
        v3(5.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    ));
    let target = |f: RayFilter| w.cast_ray(o, d, m, &f).map(|h| h.target);
    assert_eq!(
        target(RayFilter::default()),
        Some(RayTarget::StaticCollider(0))
    );
    assert_eq!(
        target(RayFilter::default().with_static(false)),
        Some(RayTarget::Sdf(0))
    );
    assert_eq!(
        target(RayFilter::default().with_static(false).with_sdf(false)),
        None
    );
}

/// An SDF attached to a body is filtered with that body.
#[test]
fn sdf_attached_to_a_body_follows_the_body_filter() {
    let mut w = world();
    let b = w.add_body(RigidBody::new_static(v3(5.0, 0.0, 0.0)));
    let mut sdf = SdfCollider::new_dynamic(Box::new(unit_sphere_sdf()), b);
    sdf.position = v3(5.0, 0.0, 0.0);
    w.add_sdf_collider(sdf);
    let o = v3(0.0, 0.0, 0.0);
    let h = w
        .cast_ray(o, Vec3Fix::UNIT_X, fx(100.0), &RayFilter::default())
        .expect("hit");
    assert_eq!((h.target, h.body), (RayTarget::Sdf(0), Some(b)));
    assert_eq!(
        w.cast_ray(
            o,
            Vec3Fix::UNIT_X,
            fx(100.0),
            &RayFilter::default().excluding_body(b)
        ),
        None
    );
}

/// The BVH walk hands the narrow phase the bodies along the ray only: of three
/// boxes, the one 50 away from the ray is not a candidate.
#[test]
fn bvh_culls_bodies_off_the_ray() {
    let mut w = world();
    let a = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(0.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let b = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(8.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    );
    let off = shaped(
        &mut w,
        Shape::Box {
            half_extents: v3(1.0, 1.0, 1.0),
        },
        v3(4.0, 0.0, 50.0),
        QuatFix::IDENTITY,
    );
    let caster = w.ray_caster(RayFilter::default());
    let mut c = caster.candidates(v3(-10.0, 0.0, 0.0), Vec3Fix::UNIT_X, fx(100.0));
    c.sort_unstable();
    assert_eq!(c, vec![a, b]);
    assert!(!c.contains(&off));
    // A short segment does not reach the second box.
    assert_eq!(
        caster.candidates(v3(-10.0, 0.0, 0.0), Vec3Fix::UNIT_X, fx(9.5)),
        vec![a]
    );
    // A filtered body is not in the tree.
    let only_far = w.ray_caster(RayFilter::default().excluding_body(a));
    assert_eq!(
        only_far.candidates(v3(-10.0, 0.0, 0.0), Vec3Fix::UNIT_X, fx(100.0)),
        vec![b]
    );
}

/// The same scene gives the same bits on every run, and the closest hit does not
/// depend on the order bodies were added.
#[test]
fn results_are_reproducible_and_order_independent() {
    let build = |order: &[f64]| {
        let mut w = world();
        for &x in order {
            shaped(
                &mut w,
                Shape::Cylinder {
                    radius: fx(0.7),
                    half_height: Fix128::ONE,
                },
                v3(x, 0.1, 0.0),
                rot_y(17.0),
            );
        }
        w
    };
    let o = v3(-9.0, 0.3, 0.05);
    let d = v3(1.0, 0.01, 0.02);
    let a = build(&[0.0, 3.0, 6.0]);
    let b = build(&[6.0, 0.0, 3.0]);
    let ha = a
        .cast_ray(o, d, fx(100.0), &RayFilter::default())
        .expect("hit");
    let hb = b
        .cast_ray(o, d, fx(100.0), &RayFilter::default())
        .expect("hit");
    assert_eq!((ha.t, ha.point, ha.normal), (hb.t, hb.point, hb.normal));
    assert_eq!(
        ha,
        a.cast_ray(o, d, fx(100.0), &RayFilter::default())
            .expect("hit")
    );
}
