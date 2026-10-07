//! Audit oracles for `alice_physics::raycast`.
//!
//! Expected values are closed-form geometry (sphere / capsule / plane
//! intersection written out by hand), compared in f64 with a tolerance far
//! below fixed-point resolution effects on the chosen scenes.

use alice_physics::collider::{Capsule, Sphere, AABB};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::{
    ray_aabb, ray_capsule, ray_plane, ray_sphere, raycast_aabbs, raycast_all_aabbs,
    raycast_all_spheres, raycast_any_aabbs, raycast_any_spheres, raycast_spheres, sweep_sphere,
    Ray,
};

fn fx(f: f64) -> Fix128 {
    Fix128::from_f64(f)
}
fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}
fn close(a: Fix128, b: f64) -> bool {
    (a.to_f64() - b).abs() < 1.0e-8
}
fn vclose(a: Vec3Fix, x: f64, y: f64, z: f64) -> bool {
    close(a.x, x) && close(a.y, y) && close(a.z, z)
}

/// Ray normalises its direction: a (3, 4, 0) direction hits a sphere at the
/// Euclidean distance, not at distance / 5.
#[test]
fn ray_new_normalises_direction_so_t_is_euclidean_distance() {
    let ray = Ray::new(v(-3.0, 0.0, 0.0), v(3.0, 0.0, 0.0));
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let hit = ray_sphere(&ray, &s, fx(100.0)).unwrap();
    assert!(close(hit.t, 2.0));
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(3.0, 4.0, 0.0));
    assert!(vclose(ray.direction, 0.6, 0.8, 0.0));
    assert!(vclose(ray.at(fx(5.0)), 3.0, 4.0, 0.0));
}

/// Off-centre sphere hit: chord at height 0.6 of a unit sphere is at
/// x = -0.8, so t = 2.2 from x = -3 and the normal is (-0.8, 0.6, 0).
#[test]
fn ray_sphere_off_centre_hit_point_and_normal() {
    let ray = Ray::new(v(-3.0, 0.6, 0.0), v(1.0, 0.0, 0.0));
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let hit = ray_sphere(&ray, &s, fx(100.0)).unwrap();
    assert!(close(hit.t, 2.2));
    assert!(vclose(hit.point, -0.8, 0.6, 0.0));
    assert!(vclose(hit.normal, -0.8, 0.6, 0.0));
}

/// A ray starting inside a sphere reports the far (exit) intersection.
#[test]
fn ray_sphere_from_inside_reports_exit() {
    let ray = Ray::new(v(0.5, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let hit = ray_sphere(&ray, &s, fx(10.0)).unwrap();
    assert!(close(hit.t, 0.5));
    assert!(vclose(hit.point, 1.0, 0.0, 0.0));
}

/// Sphere entirely behind the ray origin is a miss.
#[test]
fn ray_sphere_behind_origin_misses() {
    let ray = Ray::new(v(5.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    assert!(ray_sphere(&ray, &s, fx(100.0)).is_none());
}

/// Capsule hit on a side perpendicular to an oblique axis: the normal is the
/// perpendicular unit vector and t = distance - radius.
#[test]
fn ray_capsule_oblique_axis_side_hit() {
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let cap = Capsule::new(v(0.0, 0.0, 0.0), v(2.0, 2.0, 0.0), fx(0.5));
    // midpoint (1,1,0), n = (1,-1,0)/sqrt2, origin = mid + 5 n, direction -n
    let origin = v(1.0 + 5.0 * h, 1.0 - 5.0 * h, 0.0);
    let ray = Ray::new(origin, v(-h, h, 0.0));
    let hit = ray_capsule(&ray, &cap, fx(100.0)).unwrap();
    assert!(
        (hit.t.to_f64() - 4.5).abs() < 1.0e-6,
        "t {}",
        hit.t.to_f64()
    );
    assert!((hit.normal.x.to_f64() - h).abs() < 1.0e-6);
    assert!((hit.normal.y.to_f64() + h).abs() < 1.0e-6);
}

/// A ray along the capsule axis hits the near hemispherical cap: capsule
/// (-1,0,0)-(1,0,0), radius 0.5, ray from x = -5: t = 5 - 1 - 0.5 = 3.5.
#[test]
fn ray_capsule_along_axis_hits_near_cap() {
    let cap = Capsule::new(v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), fx(0.5));
    let ray = Ray::new(v(-5.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_capsule(&ray, &cap, fx(100.0)).expect("axis-parallel ray must hit the cap");
    assert!(close(hit.t, 3.5));
    assert!(vclose(hit.point, -1.5, 0.0, 0.0));
    assert!(vclose(hit.normal, -1.0, 0.0, 0.0));
}

/// Axis-parallel ray offset inside the radius hits the cap sphere at
/// x = -1 - sqrt(0.5^2 - 0.3^2) = -1.4 with normal (-0.8, 0.6, 0).
#[test]
fn ray_capsule_parallel_offset_inside_radius_hits_cap() {
    let cap = Capsule::new(v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), fx(0.5));
    let ray = Ray::new(v(-5.0, 0.3, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_capsule(&ray, &cap, fx(100.0)).expect("must hit");
    assert!(close(hit.t, 3.6));
    assert!(vclose(hit.normal, -0.8, 0.6, 0.0));
}

/// Axis-parallel ray outside the radius misses.
#[test]
fn ray_capsule_parallel_outside_radius_misses() {
    let cap = Capsule::new(v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0), fx(0.5));
    let ray = Ray::new(v(-5.0, 0.8, 0.0), v(1.0, 0.0, 0.0));
    assert!(ray_capsule(&ray, &cap, fx(100.0)).is_none());
}

/// Plane y = 2: a ray descending from above sees the +Y normal, one rising
/// from below sees -Y (the normal faces the incoming ray).
#[test]
fn ray_plane_normal_faces_the_incoming_ray() {
    let n = v(0.0, 1.0, 0.0);
    let down = Ray::new(v(0.0, 5.0, 0.0), v(0.0, -1.0, 0.0));
    let hit = ray_plane(&down, n, fx(2.0), fx(100.0)).unwrap();
    assert!(close(hit.t, 3.0));
    assert!(vclose(hit.normal, 0.0, 1.0, 0.0));
    let up = Ray::new(v(0.0, 0.0, 0.0), v(0.0, 1.0, 0.0));
    let hit = ray_plane(&up, n, fx(2.0), fx(100.0)).unwrap();
    assert!(close(hit.t, 2.0));
    assert!(vclose(hit.normal, 0.0, -1.0, 0.0));
}

/// Oblique incidence: ray along (1,1,0)/sqrt2 from the origin reaches y = 2
/// at t = 2 sqrt 2, point (2, 2, 0).
#[test]
fn ray_plane_oblique_incidence_closed_form() {
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(1.0, 1.0, 0.0));
    let hit = ray_plane(&ray, v(0.0, 1.0, 0.0), fx(2.0), fx(100.0)).unwrap();
    assert!((hit.t.to_f64() - 2.0 * 2.0_f64.sqrt()).abs() < 1.0e-6);
    assert!((hit.point.x.to_f64() - 2.0).abs() < 1.0e-6);
    assert!((hit.point.y.to_f64() - 2.0).abs() < 1.0e-6);
}

/// Plane behind the ray, parallel ray, and beyond `max_t` all miss; the
/// boundary `t == max_t` hits.
#[test]
fn ray_plane_miss_cases_and_max_t_boundary() {
    let n = v(0.0, 1.0, 0.0);
    let away = Ray::new(v(0.0, 5.0, 0.0), v(0.0, 1.0, 0.0));
    assert!(ray_plane(&away, n, fx(2.0), fx(100.0)).is_none());
    let parallel = Ray::new(v(0.0, 5.0, 0.0), v(1.0, 0.0, 0.0));
    assert!(ray_plane(&parallel, n, fx(2.0), fx(100.0)).is_none());
    let down = Ray::new(v(0.0, 5.0, 0.0), v(0.0, -1.0, 0.0));
    assert!(ray_plane(&down, n, fx(2.0), fx(2.9)).is_none());
    assert!(ray_plane(&down, n, fx(2.0), fx(3.0)).is_some());
}

/// Two identical spheres at equal distance: the single-hit query and the
/// first element of the all-hits query must agree on which body is nearest.
#[test]
fn closest_and_all_agree_on_equidistant_spheres() {
    let ray = Ray::new(v(-5.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let list = [(s, 3usize), (s, 7usize)];
    let one = raycast_spheres(&ray, &list, fx(100.0)).unwrap();
    let all = raycast_all_spheres(&ray, &list, fx(100.0));
    assert_eq!(one.body_index, all[0].body_index);
}

/// Same tie-break question for AABBs.
#[test]
fn closest_and_all_agree_on_equidistant_aabbs() {
    let ray = Ray::new(v(-5.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    let list = [(b, 3usize), (b, 7usize)];
    let one = raycast_aabbs(&ray, &list, fx(100.0)).unwrap();
    let all = raycast_all_aabbs(&ray, &list, fx(100.0));
    assert_eq!(one.body_index, all[0].body_index);
}

/// A box farther than 1e6 away is hit when `max_t` allows it.
#[test]
fn ray_aabb_beyond_one_million_units_hits() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    let ray = Ray::new(v(-2_000_000.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_aabb(&ray, &b, fx(3_000_000.0)).expect("box at t = 1999999 must be hit");
    assert!(close(hit.t, 1_999_999.0));
}

/// Origin inside a very large box (half-extent 2e6): the exit is at the
/// real box face, not at the 1e6 clamp.
#[test]
fn ray_aabb_exit_of_box_larger_than_one_million_units() {
    let b = AABB::new(v(-2_000_000.0, -1.0, -1.0), v(2_000_000.0, 1.0, 1.0));
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_aabb(&ray, &b, fx(3_000_000.0)).expect("hit");
    assert!(close(hit.t, 2_000_000.0));
}

/// The any-hit queries agree with the existence of a hit in the all-hit
/// queries for a range of `max_t` (including exact touching).
#[test]
fn any_hit_matches_all_hits_emptiness_across_max_t() {
    let ray = Ray::new(v(-5.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let spheres = [
        (Sphere::new(v(0.0, 0.0, 0.0), fx(1.0)), 0usize),
        (Sphere::new(v(10.0, 0.0, 0.0), fx(1.0)), 1usize),
    ];
    let boxes = [(AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0)), 0usize)];
    for max_t in [0.0, 3.9, 4.0, 4.1, 6.0, 20.0] {
        let m = fx(max_t);
        assert_eq!(
            raycast_any_spheres(&ray, &spheres, m),
            !raycast_all_spheres(&ray, &spheres, m).is_empty(),
            "max_t {max_t}"
        );
        assert_eq!(
            raycast_any_aabbs(&ray, &boxes, m),
            !raycast_all_aabbs(&ray, &boxes, m).is_empty(),
            "max_t {max_t}"
        );
    }
}

/// Swept sphere against an off-axis target: combined radius 1.0 at height
/// 0.6 gives the same chord as the unit-sphere case: t = 5 - 0.8 = 4.2.
#[test]
fn sweep_sphere_off_axis_contact_time() {
    let mover = Sphere::new(v(-5.0, 0.6, 0.0), fx(0.2));
    let target = Sphere::new(v(0.0, 0.0, 0.0), fx(0.8));
    let hit = sweep_sphere(&mover, v(2.0, 0.0, 0.0), &target, fx(100.0)).unwrap();
    assert!(close(hit.t, 4.2));
    assert!(vclose(hit.point, -0.8, 0.6, 0.0));
}

/// The swept distance limit `max_t` is a distance, independent of the
/// magnitude of the direction vector.
#[test]
fn sweep_sphere_max_t_is_distance_independent_of_direction_length() {
    let mover = Sphere::new(v(-5.0, 0.0, 0.0), fx(0.5));
    let target = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    // contact at t = 3.5
    assert!(sweep_sphere(&mover, v(100.0, 0.0, 0.0), &target, fx(3.4)).is_none());
    assert!(sweep_sphere(&mover, v(100.0, 0.0, 0.0), &target, fx(3.6)).is_some());
}

// ---------------------------------------------------------------------------
// Boundary and face-selection oracles
// ---------------------------------------------------------------------------

/// Origin exactly on the sphere surface, moving inward: the entry point is
/// the origin itself (t = 0), not the far side.
#[test]
fn ray_sphere_origin_on_surface_entering_hits_at_zero() {
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let ray = Ray::new(v(-1.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_sphere(&ray, &s, fx(10.0)).unwrap();
    assert!(close(hit.t, 0.0), "t {}", hit.t.to_f64());
    assert!(vclose(hit.normal, -1.0, 0.0, 0.0));
}

/// Origin on the surface moving outward: the exit intersection is the
/// origin (t = 0) and is reported.
#[test]
fn ray_sphere_origin_on_surface_leaving_hits_at_zero() {
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let ray = Ray::new(v(1.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_sphere(&ray, &s, fx(10.0)).unwrap();
    assert!(close(hit.t, 0.0));
    assert!(vclose(hit.normal, 1.0, 0.0, 0.0));
}

/// From the centre the exit is at t = radius; `max_t == radius` includes it
/// and the normal points outward.
#[test]
fn ray_sphere_exit_at_exactly_max_t_is_included_with_outward_normal() {
    let s = Sphere::new(v(0.0, 0.0, 0.0), fx(1.0));
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_sphere(&ray, &s, fx(1.0)).unwrap();
    assert!(close(hit.t, 1.0));
    assert!(vclose(hit.normal, 1.0, 0.0, 0.0));
    assert!(ray_sphere(&ray, &s, fx(0.999)).is_none());
}

/// Axis-aligned hits on all six faces of a box report the outward face
/// normal and the entry distance.
#[test]
fn ray_aabb_all_six_faces_report_outward_normals() {
    let b = AABB::new(v(-1.0, -2.0, -3.0), v(1.0, 2.0, 3.0));
    let cases = [
        (v(-9.0, 0.0, 0.0), v(1.0, 0.0, 0.0), 8.0, (-1.0, 0.0, 0.0)),
        (v(9.0, 0.0, 0.0), v(-1.0, 0.0, 0.0), 8.0, (1.0, 0.0, 0.0)),
        (v(0.0, -9.0, 0.0), v(0.0, 1.0, 0.0), 7.0, (0.0, -1.0, 0.0)),
        (v(0.0, 9.0, 0.0), v(0.0, -1.0, 0.0), 7.0, (0.0, 1.0, 0.0)),
        (v(0.0, 0.0, -9.0), v(0.0, 0.0, 1.0), 6.0, (0.0, 0.0, -1.0)),
        (v(0.0, 0.0, 9.0), v(0.0, 0.0, -1.0), 6.0, (0.0, 0.0, 1.0)),
    ];
    for (o, d, t, n) in cases {
        let hit = ray_aabb(&Ray::new(o, d), &b, fx(100.0)).unwrap();
        assert!(close(hit.t, t), "t {} vs {t}", hit.t.to_f64());
        assert!(
            vclose(hit.normal, n.0, n.1, n.2),
            "normal for origin {:?}",
            n
        );
    }
}

/// An oblique ray whose nearest exit is the z face: box x,y in [-10, 10],
/// z in [-1, 1], from the centre along (0.6, 0, 0.8): t = 1 / 0.8 = 1.25.
#[test]
fn ray_aabb_exit_through_z_face_of_oblique_ray() {
    let b = AABB::new(v(-10.0, -10.0, -1.0), v(10.0, 10.0, 1.0));
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(0.6, 0.0, 0.8));
    let hit = ray_aabb(&ray, &b, fx(100.0)).unwrap();
    assert!(
        (hit.t.to_f64() - 1.25).abs() < 1.0e-8,
        "t {}",
        hit.t.to_f64()
    );
    assert!(vclose(hit.normal, 0.0, 0.0, 1.0));
}

/// Origin on the exit face moving outward: t = 0 hit (t_max == 0).
#[test]
fn ray_aabb_origin_on_exit_face_moving_out_hits_at_zero() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    let ray = Ray::new(v(1.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_aabb(&ray, &b, fx(10.0)).unwrap();
    assert!(close(hit.t, 0.0));
}

/// A ray parallel to a slab and outside it misses, for each axis.
#[test]
fn ray_aabb_parallel_to_each_axis_outside_misses() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    // along Y, offset in x
    assert!(ray_aabb(
        &Ray::new(v(5.0, -9.0, 0.0), v(0.0, 1.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_none());
    // along X, offset in y
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 5.0, 0.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_none());
    // along X, offset in z (both signs)
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 0.0, 5.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_none());
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 0.0, -5.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_none());
}

/// A ray running exactly along a face of the box (parallel, on the boundary)
/// counts as touching: boundaries are inclusive for y = min, z = max.
#[test]
fn ray_aabb_parallel_on_boundary_plane_hits() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    assert!(ray_aabb(
        &Ray::new(v(-9.0, -1.0, 0.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_some());
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 0.0, 1.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_some());
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 1.0, 0.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_some());
    assert!(ray_aabb(
        &Ray::new(v(-9.0, 0.0, -1.0), v(1.0, 0.0, 0.0)),
        &b,
        fx(100.0)
    )
    .is_some());
}

/// A zero-thickness box (a rectangle) is hit by a ray that crosses it, for
/// each axis.
#[test]
fn ray_aabb_flat_box_is_hit_along_each_axis() {
    let fx_flat = AABB::new(v(0.0, -1.0, -1.0), v(0.0, 1.0, 1.0));
    let hit = ray_aabb(
        &Ray::new(v(-5.0, 0.0, 0.0), v(1.0, 0.0, 0.0)),
        &fx_flat,
        fx(100.0),
    );
    assert!(hit.is_some(), "flat in x");
    let fy = AABB::new(v(-1.0, 0.0, -1.0), v(1.0, 0.0, 1.0));
    let hit = ray_aabb(
        &Ray::new(v(0.0, -5.0, 0.0), v(0.0, 1.0, 0.0)),
        &fy,
        fx(100.0),
    );
    assert!(hit.is_some(), "flat in y");
    let fz = AABB::new(v(-1.0, -1.0, 0.0), v(1.0, 1.0, 0.0));
    let hit = ray_aabb(
        &Ray::new(v(0.0, 0.0, -5.0), v(0.0, 0.0, 1.0)),
        &fz,
        fx(100.0),
    );
    assert!(hit.is_some(), "flat in z");
}

/// Capsule hit on the cylinder side by an oblique ray that is not
/// perpendicular to the axis: capsule (0,0,0)-(0,2,0) radius 0.5, ray from
/// (-3, 3, 0) along (1,-1,0)/sqrt2 meets x = -0.5 at t = 2.5 sqrt2,
/// y = 0.5.
#[test]
fn ray_capsule_oblique_ray_not_perpendicular_to_axis() {
    let cap = Capsule::new(v(0.0, 0.0, 0.0), v(0.0, 2.0, 0.0), fx(0.5));
    let ray = Ray::new(v(-3.0, 3.0, 0.0), v(1.0, -1.0, 0.0));
    let hit = ray_capsule(&ray, &cap, fx(100.0)).unwrap();
    assert!((hit.t.to_f64() - 2.5 * 2.0_f64.sqrt()).abs() < 1.0e-6);
    assert!((hit.point.x.to_f64() + 0.5).abs() < 1.0e-6);
    assert!((hit.point.y.to_f64() - 0.5).abs() < 1.0e-6);
    assert!((hit.normal.x.to_f64() + 1.0).abs() < 1.0e-6);
}

/// Origin exactly on the plane: hit at t = 0.
#[test]
fn ray_plane_origin_on_plane_hits_at_zero() {
    let ray = Ray::new(v(3.0, 2.0, 0.0), v(0.0, -1.0, 0.0));
    let hit = ray_plane(&ray, v(0.0, 1.0, 0.0), fx(2.0), fx(10.0)).unwrap();
    assert!(close(hit.t, 0.0));
}

/// A grazing ray with a tiny but non-negligible slope (1e-8, far above the
/// 2^-32 parallel threshold) still intersects the plane far away:
/// t = 2 / 1e-8 = 2e8.
#[test]
fn ray_plane_shallow_ray_hits_far_away() {
    let ray = Ray::new(v(0.0, 0.0, 0.0), v(1.0, 1.0e-8, 0.0));
    let hit = ray_plane(&ray, v(0.0, 1.0, 0.0), fx(2.0), fx(1.0e9));
    let hit = hit.expect("slope 1e-8 is not parallel");
    assert!(
        (hit.t.to_f64() - 2.0e8).abs() < 1.0e6,
        "t {}",
        hit.t.to_f64()
    );
}

/// Box hit near the 1e6 clamp boundary from a nearby origin still works
/// (the clamp only matters beyond 1e6).
#[test]
fn ray_aabb_moderately_far_box_hits() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    let ray = Ray::new(v(-500_000.0, 0.0, 0.0), v(1.0, 0.0, 0.0));
    let hit = ray_aabb(&ray, &b, fx(1.0e6)).unwrap();
    assert!(close(hit.t, 499_999.0));
}

/// Characterisation of the tie-break at an edge shared by the +Y and +Z
/// faces (ray along (0,-1,-1) hitting the edge y = 1, z = 1): the reported
/// normal is a single axis, and the z face wins the y/z tie.
#[test]
fn ray_aabb_edge_hit_between_y_and_z_faces_reports_z_normal() {
    let b = AABB::new(v(-1.0, -1.0, -1.0), v(1.0, 1.0, 1.0));
    let ray = Ray::new(v(0.0, 5.0, 5.0), v(0.0, -1.0, -1.0));
    let hit = ray_aabb(&ray, &b, fx(100.0)).unwrap();
    assert!(vclose(hit.normal, 0.0, 0.0, 1.0), "{:?}", hit.normal);
}
