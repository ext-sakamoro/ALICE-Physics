//! Independent oracle tests for the 9 previously-unwired items in
//! `src/raycast.rs`: `ray_aabb`, `ray_capsule`, `raycast_aabbs`,
//! `raycast_all_aabbs`, `raycast_all_spheres`, `raycast_any_aabbs`,
//! `raycast_any_spheres`, `raycast_spheres`, `sweep_sphere`.
//!
//! Every expected value is hand-derived from the primitive types (`Ray`,
//! `Vec3Fix`, `Fix128`, `AABB`, `Sphere`, `Capsule`) and elementary geometry
//! (slab intersection by hand, quadratic ray/sphere roots, Pythagorean
//! combined-radius TOI) -- never by calling the function under test for the
//! expected side. See each test's doc comment for the derivation.
//!
//! `sweep_sphere` has semantic overlap with `src/ccd.rs`'s already-wired
//! `sphere_sphere_toi` (both reduce to the same closed form when the CCD
//! target's velocity is zero) and with `swept_aabb` / `conservative_advancement`
//! (still unwired here, being wired in a separate parallel task on this
//! repo) -- noted, not silently resolved; see the wiring report.

use alice_physics::collider::{Capsule, Sphere, AABB};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::raycast::{
    ray_aabb, ray_capsule, raycast_aabbs, raycast_all_aabbs, raycast_all_spheres,
    raycast_any_aabbs, raycast_any_spheres, raycast_spheres, sweep_sphere, Ray, RayHit,
};

fn r(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

// ===========================================================================
// ray_aabb
// ===========================================================================

/// oracle: AABB min=(-2,-3,-4) max=(2,3,4), ray from (-10,1,2) along +X.
/// x slab t0=8,t1=12 (no swap); y,z pass through (dir=0, origin in range).
/// t=t_min=8, point=(-2,1,2). Normal: local=(-2,1,2), half=(2,3,4):
/// dx=|2-2|=0, dy=|1-3|=2, dz=|2-4|=2 -> dx<dy && dx<dz -> -X face.
#[test]
fn ray_aabb_hits_offset_box_on_entry_face() {
    let aabb = AABB::new(v(-2, -3, -4), v(2, 3, 4));
    let ray = Ray::new(v(-10, 1, 2), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("must hit");
    assert_eq!(
        hit,
        RayHit {
            t: r(8),
            point: v(-2, 1, 2),
            normal: v(-1, 0, 0),
            body_index: 0
        }
    );
}

/// oracle: same box, origin (0,1,2) is INSIDE (x slab (t_min,t_max)=(-2,2)).
/// t_min<0 so the exit point (t_max=2) is used, not t=0.
#[test]
fn ray_aabb_from_inside_reports_exit_point_not_zero() {
    let aabb = AABB::new(v(-2, -3, -4), v(2, 3, 4));
    let ray = Ray::new(v(0, 1, 2), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("origin inside must still hit");
    assert_eq!(hit.t, r(2), "must be the exit point, not t=0");
    assert_eq!(hit.point, v(2, 1, 2));
    assert_eq!(hit.normal, v(1, 0, 0));
}

/// oracle: ray parallel to the z-slab (dir.z=0) with origin.z=5 outside
/// [-2,2] must miss regardless of the x/y intersection.
#[test]
fn ray_aabb_parallel_to_slab_face_outside_range_misses() {
    let aabb = AABB::new(v(-2, -2, -2), v(2, 2, 2));
    let ray = Ray::new(v(-10, 0, 5), Vec3Fix::UNIT_X);
    assert!(ray_aabb(&ray, &aabb, r(100)).is_none());
}

/// oracle: ray parallel to the z-slab (dir.z=0) with origin.z=0 inside
/// [-2,2] passes through unconstrained; the x-slab alone determines the hit.
#[test]
fn ray_aabb_parallel_to_slab_face_inside_range_hits() {
    let aabb = AABB::new(v(-2, -2, -2), v(2, 2, 2));
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("z in-range must pass through");
    assert_eq!(hit.t, r(8));
}

/// oracle: max_t=0 with the ray origin exactly on the surface pointing
/// inward is a valid hit at t=0 (closed interval [0, max_t]).
#[test]
fn ray_aabb_max_t_zero_exact_touch_hits() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(-1, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, Fix128::ZERO).expect("t=0 is inside the closed interval");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(-1, 0, 0));
}

/// oracle: max_t=0 with the ray strictly outside (real hit at t=4) must miss.
#[test]
fn ray_aabb_max_t_zero_no_touch_misses() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(-5, 0, 0), Vec3Fix::UNIT_X);
    assert!(ray_aabb(&ray, &aabb, Fix128::ZERO).is_none());
}

/// oracle: non-normalized direction (-2,0,0), `Ray::new` normalizes to
/// (-1,0,0). Negative direction forces t0>t1 before the min/max swap on the
/// x slab: inv_d=-1, t0=(-1-5)*-1=6, t1=(1-5)*-1=4 -> swap -> (4,6). t=4,
/// point=(1,0,0).
#[test]
fn ray_aabb_non_normalized_negative_direction_exercises_slab_swap() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(5, 0, 0), v(-2, 0, 0));
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("must hit via the min/max swap");
    assert_eq!(hit.t, r(4));
    assert_eq!(hit.point, v(1, 0, 0));
    assert_eq!(hit.normal, v(1, 0, 0));
}

/// oracle: corner tie dx==dy. origin=(-5,1,0) sits exactly on the y=max=1
/// boundary (dir.y=0, boundary inclusive). point=(-1,1,0): local=(-1,1,0),
/// half=(1,1,1): dx=0, dy=0, dz=1. `dx<dy` is false (tie) so the face
/// selector falls through to `dy<dz` (0<1, true) -> +Y face, not -X.
#[test]
fn ray_aabb_corner_tie_resolves_to_y_face() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(-5, 1, 0), Vec3Fix::UNIT_X);
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("must hit");
    assert_eq!(hit.t, r(4));
    assert_eq!(
        hit.normal,
        v(0, 1, 0),
        "dx==dy tie must resolve to +Y, not -X"
    );
}

/// oracle: Y-slab min/max swap. AABB [-1,1]^3, ray from (0,5,0) along -Y
/// (unit). y slab: inv_d=-1, t0=(-1-5)*-1=6, t1=(1-5)*-1=4 -> t0>t1 ->
/// swap -> (t_min,t_max)=(4,6). x,z pass through (origin 0 in range). t=4,
/// point=(0,1,0).
#[test]
fn ray_aabb_y_slab_swap() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(0, 5, 0), v(0, -1, 0));
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("must hit via the Y-slab swap");
    assert_eq!(hit.t, r(4));
    assert_eq!(hit.point, v(0, 1, 0));
}

/// oracle: Z-slab min/max swap, mirror of the Y case above along -Z.
#[test]
fn ray_aabb_z_slab_swap() {
    let aabb = AABB::new(v(-1, -1, -1), v(1, 1, 1));
    let ray = Ray::new(v(0, 0, 5), v(0, 0, -1));
    let hit = ray_aabb(&ray, &aabb, r(100)).expect("must hit via the Z-slab swap");
    assert_eq!(hit.t, r(4));
    assert_eq!(hit.point, v(0, 0, 1));
}

// ===========================================================================
// ray_capsule
// ===========================================================================

fn vertical_capsule() -> Capsule {
    Capsule::new(v(0, -2, 0), v(0, 2, 0), Fix128::ONE)
}

/// oracle: vertical capsule (0,-2,0)-(0,2,0) r=1, ray along +X at y=0,z=0.
/// Perpendicular distance from the ray's line to the capsule axis is 0, so
/// the entry is at x=-radius=-1: t = -1-(-5) = 4.
#[test]
fn ray_capsule_hits_cylinder_side() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-5, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100)).expect("must hit cylinder side");
    assert_eq!(hit.t, r(4));
    assert_eq!(hit.point, v(-1, 0, 0));
}

/// oracle: zero-length capsule (a==b) is documented to collapse to a
/// sphere of the same radius. Sphere center (0,0,0) r=2, ray from
/// (-10,0,0): entry at x=-2 -> t = -2-(-10) = 8.
#[test]
fn ray_capsule_degenerate_zero_length_equals_sphere() {
    let capsule = Capsule::new(Vec3Fix::ZERO, Vec3Fix::ZERO, r(2));
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100)).expect("degenerate capsule == sphere");
    assert_eq!(hit.t, r(8));
    assert_eq!(hit.point, v(-2, 0, 0));
}

/// oracle: ray at y=5 clears the cylinder (perpendicular distance 5 > r=1)
/// and both hemisphere caps (distance to either cap center >= 5 > 1).
#[test]
fn ray_capsule_misses_everything_when_far_off_axis() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-5, 5, 0), Vec3Fix::UNIT_X);
    assert!(ray_capsule(&ray, &capsule, r(100)).is_none());
}

/// oracle: ray at y=-3 (the bottom cap's south-pole height) is tangent to
/// the bottom cap sphere (oc=(-10,-1,0), b=-10, c=101-1=100, disc=0,
/// t=10, point=(0,-3,0)) and the infinite-cylinder quadratic's projection
/// onto the axis comes out to -0.25 (outside [0,1]), forcing the fallthrough
/// to the hemisphere-cap test rather than the cylinder-side branch.
#[test]
fn ray_capsule_proj_below_segment_falls_through_to_bottom_cap() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-10, -3, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100)).expect("bottom cap fallback");
    assert_eq!(hit.t, r(10));
    assert_eq!(hit.point, v(0, -3, 0));
    assert_eq!(hit.normal, v(0, -1, 0));
}

/// oracle: mirror of the above at y=+3 -- proj > 1, top cap's north pole.
#[test]
fn ray_capsule_proj_above_segment_falls_through_to_top_cap() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-10, 3, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100)).expect("top cap fallback");
    assert_eq!(hit.t, r(10));
    assert_eq!(hit.point, v(0, 3, 0));
    assert_eq!(hit.normal, v(0, 1, 0));
}

/// oracle: max_t narrowing -- the real hit is at t=4, so max_t=3 must miss
/// and max_t=4 must hit (boundary inclusive).
#[test]
fn ray_capsule_max_t_boundary() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-5, 0, 0), Vec3Fix::UNIT_X);
    assert!(ray_capsule(&ray, &capsule, r(3)).is_none());
    assert_eq!(ray_capsule(&ray, &capsule, r(4)).map(|h| h.t), Some(r(4)));
}

/// oracle: ray exactly tangent to the (infinite) cylinder's lateral surface,
/// at a point within the finite segment (proj=0.5, in range). Ray at
/// z=1=radius, y=0 (mid-segment), x along +X: the perpendicular-distance
/// quadratic has disc=0 (a genuine double root, not a disc<0 miss), and
/// the point (0,0,1) is within the segment, so this must be a cylinder-side
/// hit at t=10, not a fallthrough to the hemisphere caps (which both miss
/// at this position: oc has magnitude^2 = 100+1+1=102, c=102-1=101,
/// disc=100-101=-1<0 for each cap).
///
/// This is the oracle that kills the `discriminant < ZERO -> <= ZERO`
/// mutation found by the mutation-testing pass (see the wiring report):
/// that mutant misroutes an exact disc=0 tangent to the (missing) cap
/// fallback and returns None instead of the correct tangent hit.
#[test]
fn ray_capsule_tangent_to_cylinder_side_is_not_a_cap_fallback() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-10, 0, 1), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100))
        .expect("disc=0 tangent within the segment must hit the cylinder side");
    assert_eq!(hit.t, r(10));
    assert_eq!(hit.point, v(0, 0, 1));
    assert_eq!(hit.normal, v(0, 0, 1));
}

/// oracle: ray origin exactly ON the cylinder's lateral surface (at
/// radius-distance from the axis), pointing inward, so the near root is
/// exactly t=0. Must be a valid hit at t=0 (closed interval), not a
/// fallthrough to the caps (which both miss here: oc magnitude^2=1+4=5,
/// c=5-1=4, disc=1-4=-3<0).
///
/// This is the oracle that kills the `t >= ZERO -> t > ZERO` mutation
/// found by the mutation-testing pass: that mutant rejects the valid t=0
/// root and falls through to the (missing) cap fallback, returning None.
#[test]
fn ray_capsule_origin_on_cylinder_surface_at_t_zero() {
    let capsule = vertical_capsule();
    let ray = Ray::new(v(-1, 0, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100))
        .expect("t=0 on the cylinder surface is inside the closed interval");
    assert_eq!(hit.t, Fix128::ZERO);
    assert_eq!(hit.point, v(-1, 0, 0));
    assert_eq!(hit.normal, v(-1, 0, 0));
}

/// oracle: both hemisphere caps hit with DIFFERENT t, exercising
/// `closer_hit`'s tie-break. Capsule a=(0,-1,0) b=(0,1,0) r=5 (short
/// segment, large radius so both end caps reach far past the segment).
/// Ray (-20,4,0)+X: the infinite-cylinder quadratic is real (disc=100>=0)
/// but its near-root's axial projection is 2.5 (out of [0,1]), so this
/// falls through past the cylinder-side branch to the caps:
///   bottom cap (0,-1,0) r5: disc = r^2-(y0-cap_y)^2 = 25-25 = 0 (tangent),
///     t = 20, point (0,4,0).
///   top cap (0,1,0) r5: disc = 25-9 = 16, sqrt=4, t = 20-4 = 16,
///     point (-4,4,0).
/// The nearer hit is the TOP cap (t=16 < t=20): this is the oracle that
/// kills the `ha.t < hb.t -> >` mutation found by the mutation-testing
/// pass, which would instead return the farther bottom-cap hit (t=20).
#[test]
fn ray_capsule_both_caps_hit_closer_hit_picks_nearer() {
    let capsule = Capsule::new(v(0, -1, 0), v(0, 1, 0), r(5));
    let ray = Ray::new(v(-20, 4, 0), Vec3Fix::UNIT_X);
    let hit = ray_capsule(&ray, &capsule, r(100)).expect("both caps are real hits");
    assert_eq!(
        hit.t,
        r(16),
        "closer_hit must pick the nearer (top) cap, not the farther one"
    );
    assert_eq!(hit.point, v(-4, 4, 0));
}

// ===========================================================================
// raycast_aabbs / raycast_all_aabbs / raycast_any_aabbs
// ===========================================================================

// Deliberately NOT in ascending-t order (18, 4, miss, 12, miss): this is
// the scene a missing `sort_by_key` mutation in `raycast_all_aabbs` would
// fail to catch if the list were already sorted by construction.
fn aabb_scene() -> Vec<(AABB, usize)> {
    vec![
        (AABB::new(v(8, -1, -1), v(9, 1, 1)), 2),     // t=18
        (AABB::new(v(-6, -1, -1), v(-4, 1, 1)), 0),   // t=4
        (AABB::new(v(0, 5, -1), v(1, 6, 1)), 3),      // off-axis miss
        (AABB::new(v(2, -1, -1), v(4, 1, 1)), 1),     // t=12
        (AABB::new(v(-20, -1, -1), v(-15, 1, 1)), 4), // behind, miss (t_max<0)
    ]
}

/// oracle: nearest of the 5 AABBs above is idx0 at t=4 (hand-derived per
/// AABB via the slab formula, independent of iteration order).
#[test]
fn raycast_aabbs_returns_nearest_regardless_of_order() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let max_t = r(100);
    let scene = aabb_scene();
    let expected = RayHit {
        t: r(4),
        point: v(-6, 0, 0),
        normal: v(-1, 0, 0),
        body_index: 0,
    };
    assert_eq!(raycast_aabbs(&ray, &scene, max_t), Some(expected));
    let mut reversed = scene.clone();
    reversed.reverse();
    assert_eq!(raycast_aabbs(&ray, &reversed, max_t), Some(expected));
}

#[test]
fn raycast_all_aabbs_sorted_by_t() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = aabb_scene();
    let all = raycast_all_aabbs(&ray, &scene, r(100));
    assert_eq!(
        all.iter().map(|h| h.body_index).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        all.iter().map(|h| h.t).collect::<Vec<_>>(),
        vec![r(4), r(12), r(18)]
    );
}

#[test]
fn raycast_any_aabbs_true_and_false() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = aabb_scene();
    assert!(raycast_any_aabbs(&ray, &scene, r(100)));
    let miss_only: Vec<(AABB, usize)> = scene
        .iter()
        .copied()
        .filter(|(_, idx)| *idx == 3 || *idx == 4)
        .collect();
    assert!(!raycast_any_aabbs(&ray, &miss_only, r(100)));
}

/// oracle: empty candidate list is the identity case for all three
/// batch-AABB functions.
#[test]
fn raycast_aabbs_empty_candidate_list() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let empty: Vec<(AABB, usize)> = Vec::new();
    assert!(raycast_aabbs(&ray, &empty, r(100)).is_none());
    assert!(raycast_all_aabbs(&ray, &empty, r(100)).is_empty());
    assert!(!raycast_any_aabbs(&ray, &empty, r(100)));
}

/// oracle: max_t narrowing. The nearest AABB hit is at t=4: max_t=3 must
/// miss entirely, max_t=5 must still find idx0 (the farther AABBs are
/// still out of budget).
#[test]
fn raycast_aabbs_max_t_narrowing() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = aabb_scene();
    assert!(raycast_aabbs(&ray, &scene, r(3)).is_none());
    assert_eq!(
        raycast_aabbs(&ray, &scene, r(5)).map(|h| h.body_index),
        Some(0)
    );
}

// ===========================================================================
// raycast_spheres / raycast_all_spheres / raycast_any_spheres
// ===========================================================================

// Deliberately NOT in ascending-t order (13, miss, 5, miss): same rationale
// as `aabb_scene` above, for `raycast_all_spheres`'s sort.
fn sphere_scene() -> Vec<(Sphere, usize)> {
    vec![
        (Sphere::new(v(5, 0, 0), r(2)), 1),          // t=13
        (Sphere::new(v(0, 4, 0), Fix128::ONE), 2),   // off-axis miss (disc<0)
        (Sphere::new(v(-4, 0, 0), Fix128::ONE), 0),  // t=5
        (Sphere::new(v(-20, 0, 0), Fix128::ONE), 3), // behind, both roots <0
    ]
}

#[test]
fn raycast_spheres_returns_nearest_regardless_of_order() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let max_t = r(100);
    let scene = sphere_scene();
    let expected = RayHit {
        t: r(5),
        point: v(-5, 0, 0),
        normal: v(-1, 0, 0),
        body_index: 0,
    };
    assert_eq!(raycast_spheres(&ray, &scene, max_t), Some(expected));
    let mut reversed = scene.clone();
    reversed.reverse();
    assert_eq!(raycast_spheres(&ray, &reversed, max_t), Some(expected));
}

#[test]
fn raycast_all_spheres_sorted_by_t() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = sphere_scene();
    let all = raycast_all_spheres(&ray, &scene, r(100));
    assert_eq!(
        all.iter().map(|h| h.body_index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        all.iter().map(|h| h.t).collect::<Vec<_>>(),
        vec![r(5), r(13)]
    );
}

#[test]
fn raycast_any_spheres_true_and_false() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = sphere_scene();
    assert!(raycast_any_spheres(&ray, &scene, r(100)));
    let miss_only: Vec<(Sphere, usize)> = scene
        .iter()
        .copied()
        .filter(|(_, idx)| *idx == 2 || *idx == 3)
        .collect();
    assert!(!raycast_any_spheres(&ray, &miss_only, r(100)));
}

#[test]
fn raycast_spheres_empty_candidate_list() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let empty: Vec<(Sphere, usize)> = Vec::new();
    assert!(raycast_spheres(&ray, &empty, r(100)).is_none());
    assert!(raycast_all_spheres(&ray, &empty, r(100)).is_empty());
    assert!(!raycast_any_spheres(&ray, &empty, r(100)));
}

#[test]
fn raycast_spheres_max_t_narrowing() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = sphere_scene();
    assert!(raycast_spheres(&ray, &scene, r(4)).is_none());
    assert_eq!(
        raycast_spheres(&ray, &scene, r(5)).map(|h| h.body_index),
        Some(0)
    );
}

/// oracle: ray origin exactly at a sphere's center (oc=(0,0,0), b=0,
/// c=-r^2, disc=r^2, sqrt=r) rejects the near root (t=-r<0) and uses the
/// far root (t=+r): the exit point, not t=0.
#[test]
fn raycast_spheres_origin_at_center_uses_far_exit() {
    let ray = Ray::new(v(-10, 0, 0), Vec3Fix::UNIT_X);
    let scene = vec![(Sphere::new(v(-10, 0, 0), r(3)), 0usize)];
    let hit = raycast_spheres(&ray, &scene, r(100)).expect("origin==center must still hit");
    assert_eq!(hit.t, r(3));
    assert_eq!(hit.point, v(-7, 0, 0));
}

/// oracle: non-normalized direction (0,5,0), `Ray::new` normalizes to
/// (0,1,0). Sphere (0,10,0) r=2, origin (0,0,0): oc=(0,-10,0), b=-10,
/// c=100-4=96, disc=4, sqrt=2, t=10-2=8.
#[test]
fn raycast_spheres_non_normalized_direction() {
    let ray = Ray::new(Vec3Fix::ZERO, v(0, 5, 0));
    let scene = vec![(Sphere::new(v(0, 10, 0), r(2)), 0usize)];
    let hit = raycast_spheres(&ray, &scene, r(100)).expect("non-normalized direction must hit");
    assert_eq!(hit.t, r(8));
    assert_eq!(hit.point, v(0, 8, 0));
}

// ===========================================================================
// sweep_sphere
// ===========================================================================

/// oracle: moving sphere (-10,0,0) r=1 along +X vs static target (0,0,0)
/// r=2. Equivalent to a ray vs the expanded sphere (center (0,0,0),
/// radius 3): surfaces touch when center separation == 3, i.e.
/// t = 10 - 3 = 7.
#[test]
fn sweep_sphere_toi_against_static_target() {
    let moving = Sphere::new(v(-10, 0, 0), Fix128::ONE);
    let target = Sphere::new(v(0, 0, 0), r(2));
    let hit = sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, r(100)).expect("must hit");
    assert_eq!(hit.t, r(7));
    assert_eq!(hit.point, v(-3, 0, 0));
}

/// oracle: the real contact above is at t=7; max_t=0 must find nothing.
#[test]
fn sweep_sphere_max_t_zero_before_contact_misses() {
    let moving = Sphere::new(v(-10, 0, 0), Fix128::ONE);
    let target = Sphere::new(v(0, 0, 0), r(2));
    assert!(sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, Fix128::ZERO).is_none());
}

/// oracle: two unit spheres already coincident at the start. Expanded
/// sphere center (0,0,0) r=2 (1+1), ray from (0,0,0): oc=(0,0,0). The sweep
/// starts in the deepest overlap, so it is an initial overlap at t=0 with the
/// fallback normal -d=(-1,0,0); the exit root t=2 is never reported (the
/// sphere-cast convention of AUD-A-S3W3-007 / 013).
#[test]
fn sweep_sphere_already_overlapping_is_initial_overlap_at_t0() {
    let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let b = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
    let hit = sweep_sphere(&a, Vec3Fix::UNIT_X, &b, r(100)).expect("must report the overlap");
    assert_eq!(hit.t, Fix128::ZERO, "initial overlap, not the exit t=2");
    assert_eq!(hit.point, v(0, 0, 0));
    assert_eq!(hit.normal, v(-1, 0, 0));
}

/// oracle: zero direction. `Ray::new`'s documented fallback for a
/// zero-length direction is `Vec3Fix::UNIT_X`, so this must equal the
/// explicit +X case exactly.
#[test]
fn sweep_sphere_zero_direction_falls_back_to_unit_x() {
    let moving = Sphere::new(v(-10, 0, 0), Fix128::ONE);
    let target = Sphere::new(v(0, 0, 0), r(2));
    let explicit = sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, r(100)).expect("must hit");
    let via_zero = sweep_sphere(&moving, Vec3Fix::ZERO, &target, r(100))
        .expect("zero direction falls back to UNIT_X per Ray::new's documented policy");
    assert_eq!(via_zero, explicit);
}

/// oracle: moving sphere missing the target entirely (perpendicular offset
/// greater than the combined radius). Moving sphere (-10,5,0) r=1 along
/// +X vs target (0,0,0) r=1: expanded radius 2, perpendicular distance 5
/// > 2 -> the expanded-sphere quadratic has negative discriminant -> miss.
#[test]
fn sweep_sphere_misses_when_offset_exceeds_combined_radius() {
    let moving = Sphere::new(v(-10, 5, 0), Fix128::ONE);
    let target = Sphere::new(v(0, 0, 0), Fix128::ONE);
    assert!(sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, r(100)).is_none());
}

// ===========================================================================
// Extreme Fix128 magnitudes: Fix128 arithmetic is a mod-2^128 group (wraps),
// so every item here must not panic on extreme inputs.
// ===========================================================================

#[test]
fn all_nine_items_wrap_on_extreme_magnitudes_without_panicking() {
    let extreme = Vec3Fix::new(
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX,
        },
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let extreme_aabb = AABB::new(extreme, extreme);
    let extreme_sphere = Sphere::new(extreme, Fix128::ONE);
    let extreme_capsule = Capsule::new(extreme, Vec3Fix::ZERO, Fix128::ONE);
    let ray = Ray::new(Vec3Fix::ZERO, Vec3Fix::UNIT_X);
    let max_t = r(100);

    let outcome = std::panic::catch_unwind(|| {
        let _ = ray_aabb(&ray, &extreme_aabb, max_t);
        let _ = ray_capsule(&ray, &extreme_capsule, max_t);
        let _ = raycast_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_all_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_any_aabbs(&ray, &[(extreme_aabb, 0)], max_t);
        let _ = raycast_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = raycast_all_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = raycast_any_spheres(&ray, &[(extreme_sphere, 0)], max_t);
        let _ = sweep_sphere(&extreme_sphere, Vec3Fix::UNIT_X, &extreme_sphere, max_t);
    });
    assert!(
        outcome.is_ok(),
        "extreme Fix128 magnitudes must wrap, not panic"
    );
}
