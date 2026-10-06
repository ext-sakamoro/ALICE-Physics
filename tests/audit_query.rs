//! Audit S3W3 oracles for `src/query.rs` (shape casts, overlaps, batch and
//! BVH-accelerated variants).
//!
//! Closed forms:
//! * sphere cast of radius `r` against a body of radius `b` is a ray against
//!   the sphere of radius `r + b`: first contact `t = -d.oc - sqrt((d.oc)^2 -
//!   (|oc|^2 - (r+b)^2))`, normal `(p - c)/|p - c|`;
//! * overlap_sphere reports exactly the bodies with `|p - c| < R + b` and
//!   `depth = R + b - |p - c|` (brute force in f64, independent of the code);
//! * the BVH variants must return the same set as the brute-force variants.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::bvh::{BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh,
    overlap_aabb_expanded, overlap_sphere, overlap_sphere_bvh, sphere_cast, BatchRayQuery,
};
use alice_physics::solver::RigidBody;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn body(x: f64, y: f64, z: f64) -> RigidBody {
    RigidBody::new_static(v3(x, y, z))
}

fn close(label: &str, got: Fix128, want: f64) {
    let g = got.to_f64();
    assert!((g - want).abs() < 1e-8, "{label}: got {g} want {want}");
}

// ---------------------------------------------------------------------------
// sphere_cast
// ---------------------------------------------------------------------------

#[test]
fn sphere_cast_axis_aligned_matches_closed_form() {
    // origin (-10,0,0), r = 0.5, body r = 1 at origin: contact at x = -1.5.
    let hit = sphere_cast(
        v3(-10.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    )
    .expect("must hit");
    close("t", hit.t, 8.5);
    close("point.x", hit.point.x, -1.5);
    close("normal.x", hit.normal.x, -1.0);
    close("normal.y", hit.normal.y, 0.0);
    assert_eq!(hit.body_index, 0);
}

#[test]
fn sphere_cast_oblique_hit_matches_quadratic_solution() {
    // body at (0,1,0), combined radius 1.5: lateral offset 1 -> t = 10 - sqrt(1.25)
    let hit = sphere_cast(
        v3(-10.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(0.0, 1.0, 0.0)],
        fx(1.0),
    )
    .expect("must hit");
    let t = 10.0 - 1.25_f64.sqrt();
    close("t", hit.t, t);
    close("normal.x", hit.normal.x, (-10.0 + t - 0.0) / 1.5);
    close("normal.y", hit.normal.y, (0.0 - 1.0) / 1.5);
}

#[test]
fn sphere_cast_non_unit_direction_reports_euclidean_distance() {
    let hit = sphere_cast(
        v3(-10.0, 0.0, 0.0),
        fx(0.5),
        v3(3.0, 0.0, 0.0),
        fx(100.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    )
    .expect("must hit");
    close("t", hit.t, 8.5);
}

#[test]
fn sphere_cast_zero_direction_is_none() {
    assert!(sphere_cast(
        Vec3Fix::ZERO,
        fx(0.5),
        Vec3Fix::ZERO,
        fx(10.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0)
    )
    .is_none());
}

#[test]
fn sphere_cast_respects_max_distance() {
    let bodies = [body(0.0, 0.0, 0.0)];
    let near = sphere_cast(
        v3(-10.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(8.4999),
        &bodies,
        fx(1.0),
    );
    assert!(
        near.is_none(),
        "contact at 8.5 is beyond max_distance 8.4999"
    );
    let ok = sphere_cast(
        v3(-10.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(8.5001),
        &bodies,
        fx(1.0),
    );
    assert!(ok.is_some());
}

#[test]
fn sphere_cast_returns_the_nearest_body_in_either_order() {
    let ab = [body(0.0, 0.0, 0.0), body(20.0, 0.0, 0.0)];
    let ba = [body(20.0, 0.0, 0.0), body(0.0, 0.0, 0.0)];
    let args = (v3(-10.0, 0.0, 0.0), fx(0.5), Vec3Fix::UNIT_X, fx(100.0));
    let h1 = sphere_cast(args.0, args.1, args.2, args.3, &ab, fx(1.0)).unwrap();
    let h2 = sphere_cast(args.0, args.1, args.2, args.3, &ba, fx(1.0)).unwrap();
    assert_eq!(h1.body_index, 0);
    assert_eq!(h2.body_index, 1);
    close("t1", h1.t, 8.5);
    close("t2", h2.t, 8.5);
}

#[test]
fn sphere_cast_miss_when_lateral_offset_exceeds_combined_radius() {
    let hit = sphere_cast(
        v3(-10.0, 1.6, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    );
    assert!(hit.is_none());
}

#[test]
fn sphere_cast_behind_the_origin_is_not_a_hit() {
    let hit = sphere_cast(
        v3(10.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    );
    assert!(hit.is_none(), "body is behind the cast origin");
}

#[test]
fn sphere_cast_from_inside_reports_a_surface_facing_the_ray() {
    // Origin 0.5 from the body centre, combined radius 1.5: the cast sphere
    // already overlaps. A first-contact report must have a normal that opposes
    // (or is perpendicular to) the cast direction; the exit point of the
    // Minkowski sphere has normal . d > 0.
    let hit = sphere_cast(
        v3(0.5, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    );
    if let Some(h) = hit {
        assert!(
            h.normal.x.to_f64() <= 1e-9,
            "initially-overlapping cast reports the far exit (t = {}, normal.x = {})",
            h.t.to_f64(),
            h.normal.x.to_f64()
        );
    }
}

// ---------------------------------------------------------------------------
// capsule_cast
// ---------------------------------------------------------------------------

#[test]
fn capsule_cast_endpoint_hit_matches_closed_form() {
    // capsule (0,0,0)-(0,4,0) r 0.5, cast +x, body r 0.5 at (5,4,0):
    // the top end hits at t = 5 - 1 = 4.
    let hit = capsule_cast(
        v3(0.0, 0.0, 0.0),
        v3(0.0, 4.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(5.0, 4.0, 0.0)],
        fx(0.5),
    )
    .expect("hit");
    close("t", hit.t, 4.0);
}

#[test]
// AUD-A-S3W3-014
fn capsule_cast_misses_nothing_along_the_segment() {
    // Long horizontal capsule (-5,0,0)-(5,0,0), r 0.5, swept +y. A body at
    // (2.5, 5, 0) with r 0.5 lies over the quarter point of the segment: the
    // swept capsule touches it when the segment has risen 5 - 1.0 = 4.
    let hit = capsule_cast(
        v3(-5.0, 0.0, 0.0),
        v3(5.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_Y,
        fx(100.0),
        &[body(2.5, 5.0, 0.0)],
        fx(0.5),
    )
    .expect("capsule sweep must hit a body above its quarter point");
    close("t", hit.t, 4.0);
}

#[test]
fn capsule_cast_picks_the_earliest_of_the_three_sample_hits() {
    // vertical capsule x=-10, y in [-1,1]; two bodies ahead at different x;
    // the nearer one defines t.
    let bodies = [body(10.0, 0.0, 0.0), body(0.0, 0.0, 0.0)];
    let hit = capsule_cast(
        v3(-10.0, -1.0, 0.0),
        v3(-10.0, 1.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &bodies,
        fx(1.0),
    )
    .expect("hit");
    assert_eq!(hit.body_index, 1);
    close("t", hit.t, 8.5);
}

#[test]
fn capsule_cast_midpoint_sample_catches_a_centre_body() {
    // capsule (0,-2,0)-(0,2,0) swept +x; body at (5,0,0) r 0.5, capsule r 0.5:
    // only the midpoint ray (y=0) passes within 1.0 of the body centre
    // (endpoints are 2 away). t = 5 - 1 = 4.
    let hit = capsule_cast(
        v3(0.0, -2.0, 0.0),
        v3(0.0, 2.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(5.0, 0.0, 0.0)],
        fx(0.5),
    )
    .expect("hit");
    close("t", hit.t, 4.0);
}

// ---------------------------------------------------------------------------
// overlaps vs brute force
// ---------------------------------------------------------------------------

fn lattice() -> Vec<RigidBody> {
    let mut v = Vec::new();
    for x in -3..=3 {
        for y in -3..=3 {
            for z in -3..=3 {
                v.push(body(
                    f64::from(x) * 0.9,
                    f64::from(y) * 1.1,
                    f64::from(z) * 0.7,
                ));
            }
        }
    }
    v
}

fn pos(b: &RigidBody) -> [f64; 3] {
    [
        b.position.x.to_f64(),
        b.position.y.to_f64(),
        b.position.z.to_f64(),
    ]
}

#[test]
fn overlap_sphere_matches_brute_force_set_and_depth() {
    let bodies = lattice();
    let (c, r, br) = ([0.5, 0.25, -0.75], 2.3, 0.4);
    let got = overlap_sphere(v3(c[0], c[1], c[2]), fx(r), &bodies, fx(br));
    let mut want = Vec::new();
    for (i, b) in bodies.iter().enumerate() {
        let p = pos(b);
        let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
        if d < r + br {
            want.push((i, r + br - d));
        }
    }
    assert!(want.len() > 10 && want.len() < bodies.len());
    assert_eq!(got.len(), want.len());
    for (g, (i, depth)) in got.iter().zip(want.iter()) {
        assert_eq!(g.body_index, *i);
        assert!(
            (g.depth.to_f64() - depth).abs() < 1e-7,
            "depth {} vs {depth}",
            g.depth.to_f64()
        );
    }
}

#[test]
fn overlap_sphere_touching_is_not_overlapping() {
    // body exactly at distance R + b: contact without penetration
    let bodies = [body(4.0, 0.0, 0.0)];
    assert!(overlap_sphere(Vec3Fix::ZERO, fx(3.0), &bodies, fx(1.0)).is_empty());
    let bodies = [body(3.9, 0.0, 0.0)];
    let r = overlap_sphere(Vec3Fix::ZERO, fx(3.0), &bodies, fx(1.0));
    assert_eq!(r.len(), 1);
    close("depth", r[0].depth, 0.1);
}

#[test]
fn overlap_aabb_point_test_is_inclusive_on_every_face() {
    let bb = AABB::new(v3(-1.0, -2.0, -3.0), v3(1.0, 2.0, 3.0));
    let inside = [
        body(0.0, 0.0, 0.0),
        body(1.0, 0.0, 0.0),
        body(-1.0, 0.0, 0.0),
        body(0.0, 2.0, 0.0),
        body(0.0, -2.0, 0.0),
        body(0.0, 0.0, 3.0),
        body(0.0, 0.0, -3.0),
    ];
    let r = overlap_aabb(&bb, &inside);
    assert_eq!(r.len(), inside.len());
    for x in &r {
        assert!(x.depth.is_zero());
    }
    let eps = 1.0 / 1024.0;
    let outside = [
        body(1.0 + eps, 0.0, 0.0),
        body(-1.0 - eps, 0.0, 0.0),
        body(0.0, 2.0 + eps, 0.0),
        body(0.0, -2.0 - eps, 0.0),
        body(0.0, 0.0, 3.0 + eps),
        body(0.0, 0.0, -3.0 - eps),
    ];
    assert!(overlap_aabb(&bb, &outside).is_empty());
}

#[test]
fn overlap_aabb_expanded_face_distances() {
    // AABB [0,1]^3, body radius 0.5: face-adjacent bodies within 0.5 overlap.
    let bb = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let near = [
        body(1.4, 0.5, 0.5),
        body(-0.4, 0.5, 0.5),
        body(0.5, 1.4, 0.5),
        body(0.5, 0.5, -0.4),
    ];
    assert_eq!(overlap_aabb_expanded(&bb, &near, fx(0.5)).len(), 4);
    let far = [
        body(1.6, 0.5, 0.5),
        body(-0.6, 0.5, 0.5),
        body(0.5, 1.6, 0.5),
        body(0.5, 0.5, 1.6),
    ];
    assert!(overlap_aabb_expanded(&bb, &far, fx(0.5)).is_empty());
}

#[test]
// AUD-A-S3W3-015
fn overlap_aabb_expanded_does_not_report_spheres_that_miss_the_corner() {
    // Doc: "treated as a sphere of body_radius ... to detect sphere-vs-AABB
    // overlap". Body centre (1.9,1.9,1.9), radius 1: nearest AABB point is the
    // corner (1,1,1), distance 0.9*sqrt(3) = 1.559 > 1, so no overlap.
    let bb = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let r = overlap_aabb_expanded(&bb, &[body(1.9, 1.9, 1.9)], fx(1.0));
    assert!(r.is_empty(), "corner-region sphere reported as overlapping");
}

// ---------------------------------------------------------------------------
// batch
// ---------------------------------------------------------------------------

#[test]
fn batch_raycast_equals_zero_radius_sphere_cast_per_query() {
    let bodies = [
        body(0.0, 0.0, 0.0),
        body(5.0, 2.0, 0.0),
        body(-4.0, -1.0, 3.0),
    ];
    let qs = [
        BatchRayQuery {
            origin: v3(-10.0, 0.0, 0.0),
            direction: Vec3Fix::UNIT_X,
            max_distance: fx(100.0),
        },
        BatchRayQuery {
            origin: v3(5.0, -10.0, 0.0),
            direction: Vec3Fix::UNIT_Y,
            max_distance: fx(100.0),
        },
        BatchRayQuery {
            origin: v3(-4.0, -1.0, -10.0),
            direction: Vec3Fix::UNIT_Z,
            max_distance: fx(5.0),
        },
        BatchRayQuery {
            origin: v3(0.0, 9.0, 0.0),
            direction: Vec3Fix::UNIT_X,
            max_distance: fx(100.0),
        },
    ];
    let got = batch_raycast(&qs, &bodies, fx(1.0));
    assert_eq!(got.len(), 4);
    // closed forms: unit spheres
    close("q0", got[0].unwrap().t, 9.0);
    assert_eq!(got[0].unwrap().body_index, 0);
    close("q1", got[1].unwrap().t, 11.0); // y: -10 -> 2 - 1
    assert_eq!(got[1].unwrap().body_index, 1);
    assert!(
        got[2].is_none(),
        "contact at 12 beyond max_distance 5 ... z: -10 -> 3-1 = 12"
    );
    assert!(got[3].is_none());
}

#[test]
fn batch_sphere_cast_matches_single_casts_and_checks_lengths() {
    let bodies = [body(0.0, 0.0, 0.0), body(0.0, 8.0, 0.0)];
    let origins = [v3(-10.0, 0.0, 0.0), v3(0.0, -10.0, 0.0), v3(0.0, 20.0, 0.0)];
    let dirs = [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_X];
    let got = batch_sphere_cast(&origins, fx(0.5), &dirs, fx(100.0), &bodies, fx(1.0));
    close("c0", got[0].unwrap().t, 8.5);
    close("c1", got[1].unwrap().t, 8.5);
    assert_eq!(got[1].unwrap().body_index, 0);
    assert!(got[2].is_none());
}

#[test]
#[should_panic]
fn batch_sphere_cast_panics_on_length_mismatch() {
    let _ = batch_sphere_cast(
        &[Vec3Fix::ZERO],
        fx(0.5),
        &[],
        fx(1.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    );
}

// ---------------------------------------------------------------------------
// BVH variants == brute force
// ---------------------------------------------------------------------------

fn build_bvh(bodies: &[RigidBody], br: f64, extra_index: Option<u32>) -> LinearBvh {
    let mut prims: Vec<BvhPrimitive> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let p = pos(b);
            BvhPrimitive {
                aabb: AABB::new(
                    v3(p[0] - br, p[1] - br, p[2] - br),
                    v3(p[0] + br, p[1] + br, p[2] + br),
                ),
                index: i as u32,
                morton: 0,
            }
        })
        .collect();
    if let Some(idx) = extra_index {
        prims.push(BvhPrimitive {
            aabb: AABB::new(v3(-100.0, -100.0, -100.0), v3(100.0, 100.0, 100.0)),
            index: idx,
            morton: 0,
        });
    }
    LinearBvh::build(prims)
}

#[test]
fn overlap_sphere_bvh_equals_brute_force() {
    let bodies = lattice();
    let br = 0.4;
    let bvh = build_bvh(&bodies, br, None);
    for (c, r) in [
        ([0.5, 0.25, -0.75], 2.3),
        ([-2.7, 3.1, 0.2], 1.1),
        ([9.0, 9.0, 9.0], 1.0),
        ([0.0, 0.0, 0.0], 0.1),
    ] {
        let center = v3(c[0], c[1], c[2]);
        let mut a = overlap_sphere(center, fx(r), &bodies, fx(br));
        let mut b = overlap_sphere_bvh(center, fx(r), &bodies, fx(br), &bvh);
        a.sort_by_key(|x| x.body_index);
        b.sort_by_key(|x| x.body_index);
        assert_eq!(a, b, "centre {c:?} radius {r}");
    }
}

#[test]
fn overlap_aabb_bvh_equals_brute_force() {
    let bodies = lattice();
    let bvh = build_bvh(&bodies, 0.4, None);
    for (lo, hi) in [
        ([-1.3, -0.7, -0.2], [1.9, 2.2, 0.9]),
        ([-0.45, -0.55, -0.35], [0.45, 0.55, 0.35]),
        ([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
        ([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]),
    ] {
        let bb = AABB::new(v3(lo[0], lo[1], lo[2]), v3(hi[0], hi[1], hi[2]));
        let mut a = overlap_aabb(&bb, &bodies);
        let mut b = overlap_aabb_bvh(&bb, &bodies, &bvh);
        a.sort_by_key(|x| x.body_index);
        b.sort_by_key(|x| x.body_index);
        assert_eq!(a, b, "aabb {lo:?}..{hi:?}");
    }
}

#[test]
fn bvh_queries_ignore_leaf_indices_beyond_the_body_slice() {
    let bodies = [body(0.0, 0.0, 0.0), body(1.0, 0.0, 0.0)];
    let bvh = build_bvh(&bodies, 0.4, Some(99));
    let s = overlap_sphere_bvh(Vec3Fix::ZERO, fx(5.0), &bodies, fx(0.4), &bvh);
    assert_eq!(s.len(), 2);
    let bb = AABB::new(v3(-5.0, -5.0, -5.0), v3(5.0, 5.0, 5.0));
    let a = overlap_aabb_bvh(&bb, &bodies, &bvh);
    assert_eq!(a.len(), 2);
}

#[test]
fn capsule_cast_tilted_side_contact_is_the_closed_form_toi() {
    // capsule (-1,-2,-2)-(1,2,2): axis through the origin along u = (1,2,2)/3,
    // half-length 3; ends are 2.83 from the x axis. The body at (5,0,0) is
    // reached by the capsule's side: the distance from (5 - t, 0, 0) to the axis
    // line is (5 - t) sqrt(8)/3, equal to the reach 1 at t = 5 - 3/sqrt(8), with
    // the axial coordinate (5 - t)/3 = 0.354 inside the segment.
    let hit = capsule_cast(
        v3(-1.0, -2.0, -2.0),
        v3(1.0, 2.0, 2.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(5.0, 0.0, 0.0)],
        fx(0.5),
    )
    .expect("hit");
    close("t", hit.t, 5.0 - 3.0 / 8.0_f64.sqrt());
}

#[test]
fn overlap_aabb_expanded_inflates_all_six_faces() {
    let bb = AABB::new(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0));
    let near = [
        body(0.5, -0.4, 0.5),
        body(0.5, 0.5, 1.4),
        body(1.4, 0.5, 0.5),
        body(-0.4, 0.5, 0.5),
        body(0.5, 1.4, 0.5),
        body(0.5, 0.5, -0.4),
    ];
    assert_eq!(overlap_aabb_expanded(&bb, &near, fx(0.5)).len(), 6);
    let far = [body(0.5, -0.6, 0.5), body(0.5, 0.5, 1.6)];
    assert!(overlap_aabb_expanded(&bb, &far, fx(0.5)).is_empty());
}

#[test]
#[should_panic]
fn batch_sphere_cast_panics_when_origins_are_fewer_than_directions() {
    let _ = batch_sphere_cast(
        &[],
        fx(0.5),
        &[Vec3Fix::UNIT_X],
        fx(1.0),
        &[body(0.0, 0.0, 0.0)],
        fx(1.0),
    );
}

#[test]
fn bvh_queries_ignore_a_leaf_index_equal_to_the_body_count() {
    let bodies = [body(0.0, 0.0, 0.0), body(1.0, 0.0, 0.0)];
    let bvh = build_bvh(&bodies, 0.4, Some(2));
    assert_eq!(
        overlap_sphere_bvh(Vec3Fix::ZERO, fx(5.0), &bodies, fx(0.4), &bvh).len(),
        2
    );
    let bb = AABB::new(v3(-5.0, -5.0, -5.0), v3(5.0, 5.0, 5.0));
    assert_eq!(overlap_aabb_bvh(&bb, &bodies, &bvh).len(), 2);
}

/// Capsule sweep, end cap: capsule (0,0,0)-(2,0,0), reach 0.5 + 0.5 = 1, swept
/// +x; the body at (5, 0.6, 0) is met by the end sphere at (2,0,0) when
/// (3 - t)^2 + 0.6^2 = 1, t = 2.2. The contact point is the moved end
/// (4.2, 0, 0) and the normal points from the body to it.
#[test]
fn capsule_cast_end_cap_toi_and_normal() {
    let hit = capsule_cast(
        v3(0.0, 0.0, 0.0),
        v3(2.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_X,
        fx(100.0),
        &[body(5.0, 0.6, 0.0)],
        fx(0.5),
    )
    .expect("hit");
    close("t", hit.t, 2.2);
    close("px", hit.point.x, 4.2);
    close("py", hit.point.y, 0.0);
    close("nx", hit.normal.x, -0.8);
    close("ny", hit.normal.y, -0.6);
}

/// A body beside the line of the axis but past the segment's end is missed:
/// capsule (0,0,0)-(2,0,0) swept +y passes the body at (3.5, 5, 0) at 1.5 from
/// the end, beyond the reach 1 (the infinite cylinder would hit at t = 4).
#[test]
fn capsule_cast_does_not_hit_past_the_segment_end() {
    let hit = capsule_cast(
        v3(0.0, 0.0, 0.0),
        v3(2.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_Y,
        fx(100.0),
        &[body(3.5, 5.0, 0.0)],
        fx(0.5),
    );
    assert!(hit.is_none(), "{hit:?}");
}

/// Overlap at t = 0 beside the cylinder: moving towards the body is a contact
/// at t = 0, moving away is none (the rule of sphere_cast).
#[test]
fn capsule_cast_starting_in_overlap_follows_the_sweep_direction() {
    let cast = |dir: Vec3Fix| {
        capsule_cast(
            v3(-2.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            fx(0.5),
            dir,
            fx(100.0),
            &[body(0.0, 0.8, 0.0)],
            fx(0.5),
        )
    };
    let toward = cast(Vec3Fix::UNIT_Y).expect("contact now");
    close("t", toward.t, 0.0);
    close("ny", toward.normal.y, -1.0);
    assert!(cast(-Vec3Fix::UNIT_Y).is_none());
}

/// Two bodies met at the same t (side hits at t = 4): the lower index wins.
#[test]
fn capsule_cast_ties_go_to_the_lowest_body_index() {
    let hit = capsule_cast(
        v3(-1.0, 0.0, 0.0),
        v3(1.0, 0.0, 0.0),
        fx(0.5),
        Vec3Fix::UNIT_Y,
        fx(100.0),
        &[body(-0.5, 5.0, 0.0), body(0.5, 5.0, 0.0)],
        fx(0.5),
    )
    .expect("hit");
    close("t", hit.t, 4.0);
    assert_eq!(hit.body_index, 0);
}

/// Overlap at t = 0 and a sweep that does not close in: sideways at a constant
/// distance (+x along the axis, the body 0.8 from it) or slowly away
/// ((1, -0.1, 0)) is no contact, not a later entry through an end sphere.
#[test]
fn capsule_cast_starting_in_overlap_and_not_closing_in_is_no_contact() {
    let cast = |dir: Vec3Fix| {
        capsule_cast(
            v3(-2.0, 0.0, 0.0),
            v3(2.0, 0.0, 0.0),
            fx(0.5),
            dir,
            fx(100.0),
            &[body(0.0, 0.8, 0.0)],
            fx(0.5),
        )
    };
    assert!(cast(Vec3Fix::UNIT_X).is_none());
    assert!(cast(v3(1.0, -0.1, 0.0)).is_none());
    assert!(cast(v3(1.0, 0.1, 0.0)).is_some_and(|h| h.t.is_zero()));
}
