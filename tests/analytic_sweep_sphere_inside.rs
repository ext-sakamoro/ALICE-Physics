//! Analytic oracles for `raycast::sweep_sphere` when the swept sphere starts
//! overlapping the target.
//!
//! `sweep_sphere(a, d, b, max_t)` is a ray from `a.center` along `d` against
//! the Minkowski sphere of radius `R = a.radius + b.radius` around `b.center`.
//! With `oc = a.center - b.center`, `c = |oc|^2 - R^2` and `k = oc . d`
//! (`d` normalised by `Ray::new`):
//!
//! - `c > 0` (separated): first contact at `t = -k - sqrt(k^2 - c)`; on the
//!   line of centres this is `t = |oc| - R` (distance, independent of `|d|`);
//! - `c <= 0` (overlapping or touching) and `k < 0` (moving deeper): an
//!   initial overlap at `t = 0`, point `a.center`, normal `oc / |oc|`;
//! - `c <= 0` and `k >= 0` (moving out or tangentially): no contact.
//!
//! This is the convention `query::sphere_cast` adopted for AUD-A-S3W3-007 /
//! AUD-A-S3W3-013; the far root (the exit point) is never reported as a
//! contact. Expected values are worked by hand in each comment.

#![cfg(feature = "std")]

use alice_physics::collider::Sphere;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::query::sphere_cast;
use alice_physics::raycast::sweep_sphere;
use alice_physics::solver::RigidBody;

fn r(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn sphere(x: i64, y: i64, z: i64, radius: i64) -> Sphere {
    Sphere::new(v(x, y, z), r(radius))
}

fn neg_x() -> Vec3Fix {
    Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
}

/// Mover r=1 at (4,0,0), target r=1 at (5,0,0): R=2, oc=(-1,0,0), c=1-4=-3,
/// k=-1 (moving deeper). The mover's centre is inside the target. Contact at
/// t=0 at (4,0,0) with normal (-1,0,0); the exit root t=1+2=3 is not a contact.
#[test]
fn centre_inside_target_moving_deeper_is_initial_overlap() {
    let hit = sweep_sphere(
        &sphere(4, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(5, 0, 0, 1),
        r(100),
    )
    .expect("initial overlap");
    assert_eq!(hit.t, Fix128::ZERO, "initial overlap, not the exit t=3");
    assert_eq!(hit.point, v(4, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// Mover r=2 at (3,0,0), target r=1 at (5,0,0): |oc|=2 > 1 so the mover's
/// centre is outside the target, but 2 < R=3 so the spheres overlap.
/// c=4-9=-5, k=-2. Contact at t=0, normal (-1,0,0); exit root -k+3=5 rejected.
#[test]
fn overlapping_with_centre_outside_target_is_initial_overlap() {
    let hit = sweep_sphere(
        &sphere(3, 0, 0, 2),
        Vec3Fix::UNIT_X,
        &sphere(5, 0, 0, 1),
        r(100),
    )
    .expect("initial overlap");
    assert_eq!(hit.t, Fix128::ZERO, "initial overlap, not the exit t=5");
    assert_eq!(hit.point, v(3, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// Overlap with an off-axis direction still moving deeper: mover r=1 at
/// (4,0,0), d=(1,1,0)/sqrt2, oc=(-1,0,0), k=-1/sqrt2 < 0, c=-3. t=0 and the
/// normal is oc/|oc|=(-1,0,0), not tied to the direction.
#[test]
fn overlap_off_axis_moving_deeper_is_initial_overlap() {
    let hit = sweep_sphere(&sphere(4, 0, 0, 1), v(1, 1, 0), &sphere(5, 0, 0, 1), r(100))
        .expect("initial overlap");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(4, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// Same overlap as the first case but moving away (k=+1 >= 0): the cast is
/// leaving the overlap, no contact (the old code reported the exit t=1).
#[test]
fn overlap_moving_out_reports_no_contact() {
    assert!(sweep_sphere(&sphere(4, 0, 0, 1), neg_x(), &sphere(5, 0, 0, 1), r(100)).is_none());
}

/// Coincident centres: oc=0, the deepest overlap. Contact at t=0 at the
/// start centre with the fallback normal -d (the old code reported the exit
/// t=R=2; the shared helper before this fix returned None because k=0 was
/// read as moving out).
#[test]
fn coincident_centres_are_initial_overlap_with_normal_minus_d() {
    let hit = sweep_sphere(
        &sphere(0, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(0, 0, 0, 1),
        r(100),
    )
    .expect("initial overlap");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(0, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// The same coincident start through `query::sphere_cast` (shared helper):
/// mover r=1 at (5,0,0) cast along +y against a body at (5,0,0) of radius 1.
/// t=0, point (5,0,0), normal -d=(0,-1,0).
#[test]
fn sphere_cast_coincident_centres_is_initial_overlap() {
    let bodies = [RigidBody::new_static(v(5, 0, 0))];
    let hit = sphere_cast(v(5, 0, 0), r(1), Vec3Fix::UNIT_Y, r(100), &bodies, r(1))
        .expect("initial overlap");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(5, 0, 0));
    assert_eq!(
        hit.normal,
        Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO)
    );
    assert_eq!(hit.body_index, 0);
}

/// Just touching: mover r=1 at (3,0,0), target r=1 at (5,0,0): |oc|=R=2,
/// c=0, k=-2. Contact at t=0 at (3,0,0), normal (-1,0,0).
#[test]
fn just_touching_moving_in_is_contact_at_t0() {
    let hit = sweep_sphere(
        &sphere(3, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(5, 0, 0, 1),
        r(100),
    )
    .expect("touching contact");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(3, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// Just touching and moving apart: c=0, k=+2. No contact.
#[test]
fn just_touching_moving_apart_reports_no_contact() {
    assert!(sweep_sphere(&sphere(3, 0, 0, 1), neg_x(), &sphere(5, 0, 0, 1), r(100)).is_none());
}

/// Separated on the line of centres: d=10, r_a=1, r_b=2, |v|=3. In distance
/// units t = d - r_a - r_b = 7 (the time t/|v| = 7/3); contact centre (7,0,0),
/// normal (-1,0,0). The direction's length does not scale t.
#[test]
fn separated_on_line_of_centres_hits_at_d_minus_radii() {
    let hit = sweep_sphere(
        &sphere(0, 0, 0, 1),
        v(3, 0, 0),
        &sphere(10, 0, 0, 2),
        r(100),
    )
    .expect("first contact");
    assert_eq!(hit.t, r(7));
    assert_eq!(hit.point, v(7, 0, 0));
    assert_eq!(hit.normal, neg_x());
}

/// Separated off the line of centres: target (10,3,0), R=5, oc=(-10,-3,0),
/// k=-10, c=109-25=84, k^2-c=16, t=10-4=6; contact centre (6,0,0), normal
/// ((6-10)/5, (0-3)/5, 0) = (-0.8, -0.6, 0).
#[test]
fn separated_off_axis_hits_at_closed_form() {
    let hit = sweep_sphere(
        &sphere(0, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(10, 3, 0, 4),
        r(100),
    )
    .expect("first contact");
    assert_eq!(hit.t, r(6));
    assert_eq!(hit.point, v(6, 0, 0));
    let n = hit.normal;
    let eps = 1e-12;
    assert!((n.x.to_f64() + 0.8).abs() < eps, "{}", n.x.to_f64());
    assert!((n.y.to_f64() + 0.6).abs() < eps, "{}", n.y.to_f64());
    assert!(n.z.to_f64().abs() < eps);
}

/// Misses: perpendicular offset 5 > R=3; target behind the mover; and a
/// max_t shorter than the contact distance 7.
#[test]
fn misses_report_none() {
    assert!(sweep_sphere(
        &sphere(0, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(10, 5, 0, 2),
        r(100)
    )
    .is_none());
    assert!(sweep_sphere(
        &sphere(0, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(-10, 0, 0, 2),
        r(100)
    )
    .is_none());
    assert!(sweep_sphere(
        &sphere(0, 0, 0, 1),
        Vec3Fix::UNIT_X,
        &sphere(10, 0, 0, 2),
        r(6)
    )
    .is_none());
}

/// (mover centre, mover radius, direction, target centre, target radius)
type Case = ((i64, i64, i64), i64, (i64, i64, i64), (i64, i64, i64), i64);

/// Same inputs through `query::sphere_cast` (already fixed for
/// AUD-A-S3W3-007 / 013) and `sweep_sphere` give the same answer, including
/// every overlap case above.
#[test]
fn agrees_with_query_sphere_cast() {
    let cases: [Case; 10] = [
        ((4, 0, 0), 1, (1, 0, 0), (5, 0, 0), 1),
        ((3, 0, 0), 2, (1, 0, 0), (5, 0, 0), 1),
        ((4, 0, 0), 1, (1, 1, 0), (5, 0, 0), 1),
        ((4, 0, 0), 1, (-1, 0, 0), (5, 0, 0), 1),
        ((0, 0, 0), 1, (1, 0, 0), (0, 0, 0), 1),
        ((3, 0, 0), 1, (1, 0, 0), (5, 0, 0), 1),
        ((3, 0, 0), 1, (-1, 0, 0), (5, 0, 0), 1),
        ((0, 0, 0), 1, (3, 0, 0), (10, 0, 0), 2),
        ((0, 0, 0), 1, (1, 0, 0), (10, 3, 0), 4),
        ((0, 0, 0), 1, (1, 0, 0), (10, 5, 0), 2),
    ];
    for (a, ra, d, b, rb) in cases {
        let dir = v(d.0, d.1, d.2);
        let swept = sweep_sphere(
            &sphere(a.0, a.1, a.2, ra),
            dir,
            &sphere(b.0, b.1, b.2, rb),
            r(100),
        );
        let bodies = [RigidBody::new_static(v(b.0, b.1, b.2))];
        let cast = sphere_cast(v(a.0, a.1, a.2), r(ra), dir, r(100), &bodies, r(rb));
        match (swept, cast) {
            (None, None) => {}
            (Some(s), Some(c)) => {
                assert_eq!(s.t, c.t, "t for case {a:?} {d:?} {b:?}");
                assert_eq!(s.point, c.point, "point for case {a:?} {d:?} {b:?}");
                assert_eq!(s.normal, c.normal, "normal for case {a:?} {d:?} {b:?}");
            }
            (s, c) => panic!("case {a:?} {d:?} {b:?}: sweep_sphere {s:?} vs sphere_cast {c:?}"),
        }
    }
}
