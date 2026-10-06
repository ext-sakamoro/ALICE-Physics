//! Spatial query wiring: shape casts, overlap tests, batch queries, BVH.
//!
//! Exercises every previously-unwired item in `query`:
//! - `overlap_sphere` / `overlap_aabb` / `overlap_aabb_expanded` against a
//!   small scene, each closed form derived by hand in the comments below
//! - `overlap_sphere_bvh` / `overlap_aabb_bvh`, cross-checked against their
//!   brute-force counterparts (the accelerated path must agree exactly with
//!   the reference path on every scene)
//! - `sphere_cast` / `capsule_cast` against a known geometry, with the
//!   time-of-impact derived from the ray/sphere quadratic by hand
//! - `batch_raycast` / `batch_sphere_cast`, cross-checked against the
//!   per-ray `sphere_cast` loop they must be equivalent to
//! - `BatchRayQuery` / `ShapeCastHit` / `OverlapResult`, whose fields are
//!   read directly below (construction + field access)
//!
//! ```bash
//! cargo run --example spatial_queries --features std
//! ```

use alice_physics::bvh::{BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh,
    overlap_aabb_expanded, overlap_sphere, overlap_sphere_bvh, sphere_cast, BatchRayQuery,
    OverlapResult, ShapeCastHit,
};
use alice_physics::raycast::{ray_sphere, Ray};
use alice_physics::solver::RigidBody;

/// Four static bodies used for the overlap tests:
/// B0 (0,0,0), B1 (6,0,0), B2 (0,6,0), B3 (-4,0,0).
fn overlap_scene() -> Vec<RigidBody> {
    vec![
        RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)),
        RigidBody::new_static(Vec3Fix::from_int(6, 0, 0)),
        RigidBody::new_static(Vec3Fix::from_int(0, 6, 0)),
        RigidBody::new_static(Vec3Fix::from_int(-4, 0, 0)),
    ]
}

fn build_bvh(bodies: &[RigidBody], body_radius: Fix128) -> LinearBvh {
    let prims: Vec<BvhPrimitive> = bodies
        .iter()
        .enumerate()
        .map(|(i, b)| BvhPrimitive {
            aabb: AABB::new(
                Vec3Fix::new(
                    b.position.x - body_radius,
                    b.position.y - body_radius,
                    b.position.z - body_radius,
                ),
                Vec3Fix::new(
                    b.position.x + body_radius,
                    b.position.y + body_radius,
                    b.position.z + body_radius,
                ),
            ),
            index: i as u32,
            morton: 0,
        })
        .collect();
    LinearBvh::build(prims)
}

fn sorted_indices(results: &[OverlapResult]) -> Vec<usize> {
    let mut v: Vec<usize> = results.iter().map(|r| r.body_index).collect();
    v.sort_unstable();
    v
}

fn main() {
    let bodies = overlap_scene();
    let body_radius = Fix128::ONE;

    // --- overlap_sphere ---------------------------------------------------
    //
    // center (0,0,0), radius 3, body_radius 1 -> combined_radius 4,
    // combined_sq 16.
    //   B0 dist   0 ->   0 < 16  overlap, depth = 4 - 0 = 4
    //   B1 dist   6 ->  36 < 16  no
    //   B2 dist   6 ->  36 < 16  no
    //   B3 dist   4 ->  16 < 16  no (strict `<`: touching is excluded)
    let sphere_center = Vec3Fix::from_int(0, 0, 0);
    let sphere_radius = Fix128::from_int(3);
    let sphere_results: Vec<OverlapResult> =
        overlap_sphere(sphere_center, sphere_radius, &bodies, body_radius);
    assert_eq!(sorted_indices(&sphere_results), vec![0], "overlap_sphere");
    assert_eq!(sphere_results[0].depth, Fix128::from_int(4));
    println!(
        "[spatial_queries] overlap_sphere: {} hit(s), body0 depth={}",
        sphere_results.len(),
        sphere_results[0].depth.to_f64()
    );

    // --- overlap_aabb -------------------------------------------------------
    //
    // aabb [-1,-1,-1] .. [7,1,1]:
    //   B0 (0,0,0)  -> x,y,z all in range -> inside
    //   B1 (6,0,0)  -> x=6 in [-1,7]      -> inside
    //   B2 (0,6,0)  -> y=6 not in [-1,1]  -> outside
    //   B3 (-4,0,0) -> x=-4 not in [-1,7] -> outside
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(7, 1, 1));
    let aabb_results: Vec<OverlapResult> = overlap_aabb(&aabb, &bodies);
    assert_eq!(sorted_indices(&aabb_results), vec![0, 1], "overlap_aabb");
    println!(
        "[spatial_queries] overlap_aabb: bodies {:?}",
        sorted_indices(&aabb_results)
    );

    // --- overlap_aabb_expanded: zero expansion == overlap_aabb exactly ------
    let expanded_zero = overlap_aabb_expanded(&aabb, &bodies, Fix128::ZERO);
    assert_eq!(
        expanded_zero, aabb_results,
        "zero expansion must equal overlap_aabb exactly"
    );

    // --- overlap_aabb_expanded: known margin adds exactly the reached bodies -
    //
    // aabb2 [2,-1,-1] .. [4,1,1] (no body inside: B0 x=0, B1 x=6 both
    // outside [2,4]). Expand by 2 -> [0,-3,-3] .. [6,3,3]:
    //   B0 (0,0,0)  -> x=0 in [0,6]  -> now inside  (added)
    //   B1 (6,0,0)  -> x=6 in [0,6]  -> now inside  (added)
    //   B2 (0,6,0)  -> y=6 not in [-3,3] -> still outside
    //   B3 (-4,0,0) -> x=-4 not in [0,6] -> still outside
    let aabb2 = AABB::new(Vec3Fix::from_int(2, -1, -1), Vec3Fix::from_int(4, 1, 1));
    let aabb2_unexpanded = overlap_aabb(&aabb2, &bodies);
    assert!(
        aabb2_unexpanded.is_empty(),
        "aabb2 should contain no body before expansion"
    );
    let margin = Fix128::from_int(2);
    let aabb2_expanded = overlap_aabb_expanded(&aabb2, &bodies, margin);
    assert_eq!(
        sorted_indices(&aabb2_expanded),
        vec![0, 1],
        "margin of 2 should reach exactly bodies 0 and 1"
    );
    println!(
        "[spatial_queries] overlap_aabb_expanded: margin 2 adds {:?}",
        sorted_indices(&aabb2_expanded)
    );

    // --- overlap_sphere_bvh / overlap_aabb_bvh: BVH must agree with brute ---
    let bvh = build_bvh(&bodies, body_radius);
    let sphere_bvh_results =
        overlap_sphere_bvh(sphere_center, sphere_radius, &bodies, body_radius, &bvh);
    let mut brute_sorted = sphere_results.clone();
    brute_sorted.sort_by_key(|r| r.body_index);
    let mut bvh_sorted = sphere_bvh_results.clone();
    bvh_sorted.sort_by_key(|r| r.body_index);
    assert_eq!(
        bvh_sorted, brute_sorted,
        "overlap_sphere_bvh must agree exactly with overlap_sphere"
    );

    let aabb_bvh_results = overlap_aabb_bvh(&aabb, &bodies, &bvh);
    let mut aabb_bvh_sorted = aabb_bvh_results.clone();
    aabb_bvh_sorted.sort_by_key(|r| r.body_index);
    let mut aabb_brute_sorted = aabb_results.clone();
    aabb_brute_sorted.sort_by_key(|r| r.body_index);
    assert_eq!(
        aabb_bvh_sorted, aabb_brute_sorted,
        "overlap_aabb_bvh must agree exactly with overlap_aabb"
    );
    println!(
        "[spatial_queries] overlap_*_bvh: BVH-accelerated and brute-force agree \
         ({} sphere hits, {} aabb hits)",
        bvh_sorted.len(),
        aabb_bvh_sorted.len()
    );

    // --- sphere_cast: closest-of-N selection ---------------------------------
    //
    // bodies at (5,0,0) and (15,0,0), body_radius 1, cast_radius 1
    // (combined_radius 2 for both). Ray from origin along +X:
    //   body0: oc=(-5,0,0) b=-5 c=25-4=21 disc=4 sqrt=2 t=5-2=3
    //   body1: oc=(-15,0,0) b=-15 c=225-4=221 disc=4 sqrt=2 t=15-2=13
    // closest is body0 at t=3, point (3,0,0), normal (3-5,0,0)/2=(-1,0,0).
    let cast_bodies = vec![
        RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)),
        RigidBody::new_static(Vec3Fix::from_int(15, 0, 0)),
    ];
    let cast_body_radius = Fix128::ONE;
    let cast_radius = Fix128::ONE;
    let hit: Option<ShapeCastHit> = sphere_cast(
        Vec3Fix::ZERO,
        cast_radius,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &cast_bodies,
        cast_body_radius,
    );
    let hit = hit.expect("sphere_cast should hit body0");
    assert_eq!(hit.body_index, 0);
    assert_eq!(hit.t, Fix128::from_int(3));
    assert_eq!(hit.point, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(
        hit.normal,
        Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
    println!(
        "[spatial_queries] sphere_cast: body={} t={} point={:?}",
        hit.body_index,
        hit.t.to_f64(),
        (
            hit.point.x.to_f64(),
            hit.point.y.to_f64(),
            hit.point.z.to_f64()
        )
    );

    // --- sphere_cast: zero-radius cast behaves like a raycast ----------------
    //
    // Same body0, cast_radius ZERO (combined_radius = body_radius = 1):
    //   oc=(-5,0,0) b=-5 c=25-1=24 disc=1 sqrt=1 t=5-1=4
    // Cross-checked against `Ray` + `ray_sphere` directly (the production
    // raycast path), not against `sphere_cast` itself.
    let zero_radius_hit = sphere_cast(
        Vec3Fix::ZERO,
        Fix128::ZERO,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &cast_bodies,
        cast_body_radius,
    )
    .expect("zero-radius sphere_cast should hit body0");
    assert_eq!(zero_radius_hit.t, Fix128::from_int(4));
    let ray = Ray::new(Vec3Fix::ZERO, Vec3Fix::UNIT_X);
    let plain_ray_hit = ray_sphere(
        &ray,
        &alice_physics::collider::Sphere::new(cast_bodies[0].position, cast_body_radius),
        Fix128::from_int(100),
    )
    .expect("plain ray_sphere should also hit body0");
    assert_eq!(
        zero_radius_hit.t, plain_ray_hit.t,
        "zero-radius sphere_cast must equal a plain raycast"
    );
    println!(
        "[spatial_queries] sphere_cast(radius=0) == raycast: t={}",
        zero_radius_hit.t.to_f64()
    );

    // --- sphere_cast: ray starting inside a body -----------------------------
    //
    // body at (5,0,0), body_radius 1, cast_radius 1 (combined_radius 2).
    // origin (4,0,0) is inside (distance 1 < 2). oc=(-1,0,0) b=oc.d=-1 < 0
    // (moving deeper) c=1-4=-3 <= 0: the cast starts in an initial overlap,
    // reported at t=0 at the origin with normal oc/|oc|=(-1,0,0). The exit
    // root t=1+2=3 is never a contact.
    let inside_body = vec![RigidBody::new_static(Vec3Fix::from_int(5, 0, 0))];
    let inside_hit = sphere_cast(
        Vec3Fix::from_int(4, 0, 0),
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &inside_body,
        Fix128::ONE,
    )
    .expect("sphere_cast from inside the body reports the initial overlap");
    assert_eq!(
        inside_hit.t,
        Fix128::ZERO,
        "interior origin moving deeper is an initial overlap, not the exit t=3"
    );
    assert_eq!(inside_hit.point, Vec3Fix::from_int(4, 0, 0));
    assert_eq!(inside_hit.normal, Vec3Fix::from_int(-1, 0, 0));
    println!(
        "[spatial_queries] sphere_cast from inside a body: initial overlap t={} normal={:?}",
        inside_hit.t.to_f64(),
        (
            inside_hit.normal.x.to_f64(),
            inside_hit.normal.y.to_f64(),
            inside_hit.normal.z.to_f64()
        )
    );

    // --- capsule_cast: three sub-casts, closest wins -------------------------
    //
    // capsule (4,-2,0)-(4,2,0), capsule_radius 1, body at (10,0,0),
    // body_radius 1 (combined_radius 2 for every sub-cast):
    //   from (4,-2,0): oc=(-6,-2,0) b=-6 c=40-4=36 disc=36-36=0 sqrt=0
    //                  t = 6 - 0 = 6 (tangent)
    //   from (4, 2,0): oc=(-6, 2,0) b=-6 c=40-4=36 disc=0  t = 6 (tangent)
    //   from midpoint (4,0,0): oc=(-6,0,0) b=-6 c=36-4=32 disc=4 sqrt=2
    //                  t = 6 - 2 = 4   <- closest, wins
    let capsule_body = vec![RigidBody::new_static(Vec3Fix::from_int(10, 0, 0))];
    let capsule_hit = capsule_cast(
        Vec3Fix::from_int(4, -2, 0),
        Vec3Fix::from_int(4, 2, 0),
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &capsule_body,
        Fix128::ONE,
    )
    .expect("capsule_cast should hit");
    assert_eq!(
        capsule_hit.t,
        Fix128::from_int(4),
        "the midpoint sub-cast is the closest of the three"
    );
    assert_eq!(capsule_hit.point, Vec3Fix::from_int(8, 0, 0));
    println!(
        "[spatial_queries] capsule_cast: closest sub-cast t={} (midpoint)",
        capsule_hit.t.to_f64()
    );

    // --- batch_raycast == loop of sphere_cast(radius=0) ---------------------
    //
    // Q0 origin (0,0,0) dir +X  -> hits body0 at t=4 (see zero-radius case)
    // Q1 origin (0,0,0) dir +Y  -> misses both bodies (neither body lies
    //                              on the y axis)
    // Q2 origin (20,0,0) dir -X -> hits body1 (15,0,0) first:
    //    oc=(5,0,0) b=-5 c=25-1=24 disc=1 sqrt=1 t=5-1=4, point (16,0,0)
    let queries = vec![
        BatchRayQuery {
            origin: Vec3Fix::ZERO,
            direction: Vec3Fix::UNIT_X,
            max_distance: Fix128::from_int(100),
        },
        BatchRayQuery {
            origin: Vec3Fix::ZERO,
            direction: Vec3Fix::UNIT_Y,
            max_distance: Fix128::from_int(100),
        },
        BatchRayQuery {
            origin: Vec3Fix::from_int(20, 0, 0),
            direction: -Vec3Fix::UNIT_X,
            max_distance: Fix128::from_int(100),
        },
    ];
    let batch_results: Vec<Option<ShapeCastHit>> =
        batch_raycast(&queries, &cast_bodies, cast_body_radius);
    let loop_results: Vec<Option<ShapeCastHit>> = queries
        .iter()
        .map(|q| {
            sphere_cast(
                q.origin,
                Fix128::ZERO,
                q.direction,
                q.max_distance,
                &cast_bodies,
                cast_body_radius,
            )
        })
        .collect();
    assert_eq!(
        batch_results, loop_results,
        "batch_raycast must equal calling sphere_cast once per ray"
    );
    assert!(batch_results[0].is_some());
    assert!(batch_results[1].is_none());
    let q2_hit = batch_results[2].expect("Q2 should hit body1");
    assert_eq!(q2_hit.body_index, 1);
    assert_eq!(q2_hit.t, Fix128::from_int(4));
    assert_eq!(q2_hit.point, Vec3Fix::from_int(16, 0, 0));
    println!(
        "[spatial_queries] batch_raycast: {} queries, hits={:?}",
        queries.len(),
        batch_results
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
    );

    // --- batch_sphere_cast == loop of sphere_cast ----------------------------
    //
    // origins/directions both (0,0,0)+X and (0,0,0)+Y, radius 1
    // (combined_radius 2): first hits body0 at t=3 (see closest-of-N case
    // above), second misses (same geometry as the Q1 miss above, now with
    // the larger combined_radius 2 -- still misses: disc = 0-21 < 0 for
    // body0, 0-221 < 0 for body1).
    let origins = vec![Vec3Fix::ZERO, Vec3Fix::ZERO];
    let directions = vec![Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y];
    let batch_sphere_results: Vec<Option<ShapeCastHit>> = batch_sphere_cast(
        &origins,
        cast_radius,
        &directions,
        Fix128::from_int(100),
        &cast_bodies,
        cast_body_radius,
    );
    let loop_sphere_results: Vec<Option<ShapeCastHit>> = origins
        .iter()
        .zip(directions.iter())
        .map(|(o, d)| {
            sphere_cast(
                *o,
                cast_radius,
                *d,
                Fix128::from_int(100),
                &cast_bodies,
                cast_body_radius,
            )
        })
        .collect();
    assert_eq!(
        batch_sphere_results, loop_sphere_results,
        "batch_sphere_cast must equal calling sphere_cast once per (origin, direction) pair"
    );
    assert!(batch_sphere_results[0].is_some());
    assert!(batch_sphere_results[1].is_none());
    println!(
        "[spatial_queries] batch_sphere_cast: {} queries, hits={:?}",
        origins.len(),
        batch_sphere_results
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>()
    );

    // --- degenerate: empty world ----------------------------------------------
    let empty: Vec<RigidBody> = Vec::new();
    assert!(overlap_sphere(sphere_center, sphere_radius, &empty, body_radius).is_empty());
    assert!(overlap_aabb(&aabb, &empty).is_empty());
    assert!(overlap_aabb_expanded(&aabb, &empty, margin).is_empty());
    let empty_bvh = build_bvh(&empty, body_radius);
    assert!(overlap_sphere_bvh(
        sphere_center,
        sphere_radius,
        &empty,
        body_radius,
        &empty_bvh
    )
    .is_empty());
    assert!(overlap_aabb_bvh(&aabb, &empty, &empty_bvh).is_empty());
    assert!(sphere_cast(
        Vec3Fix::ZERO,
        cast_radius,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &empty,
        cast_body_radius
    )
    .is_none());
    assert!(capsule_cast(
        Vec3Fix::from_int(0, -1, 0),
        Vec3Fix::from_int(0, 1, 0),
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &empty,
        cast_body_radius
    )
    .is_none());
    assert!(batch_raycast(&[], &empty, cast_body_radius).is_empty());
    assert!(batch_sphere_cast(
        &[],
        cast_radius,
        &[],
        Fix128::from_int(100),
        &empty,
        cast_body_radius
    )
    .is_empty());
    println!("[spatial_queries] degenerate: empty world returns empty results, no panic");

    // --- degenerate: zero-length ray / direction -----------------------------
    assert!(sphere_cast(
        Vec3Fix::ZERO,
        cast_radius,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        &cast_bodies,
        cast_body_radius
    )
    .is_none());
    assert!(capsule_cast(
        Vec3Fix::from_int(0, -1, 0),
        Vec3Fix::from_int(0, 1, 0),
        Fix128::ONE,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        &cast_bodies,
        cast_body_radius
    )
    .is_none());
    println!("[spatial_queries] degenerate: zero-length direction returns None, no panic");

    // --- degenerate: zero-size AABB / zero-radius sphere ---------------------
    //
    // A degenerate point AABB only matches a body at that exact point.
    let point_aabb = AABB::new(Vec3Fix::from_int(6, 0, 0), Vec3Fix::from_int(6, 0, 0));
    let point_hit = overlap_aabb(&point_aabb, &bodies);
    assert_eq!(
        sorted_indices(&point_hit),
        vec![1],
        "point AABB matches only body1"
    );
    // Zero sphere radius AND zero body radius: combined_sq is 0, and the
    // strict `<` comparison excludes dist_sq == 0, so an exact coincidence
    // does *not* register as an overlap (same strict-boundary semantics as
    // the B3-touching case above, now visible at distance 0 too).
    let zero_zero = overlap_sphere(
        Vec3Fix::from_int(0, 0, 0),
        Fix128::ZERO,
        &bodies,
        Fix128::ZERO,
    );
    assert!(
        zero_zero.is_empty(),
        "zero-radius sphere query never matches, even an exact coincidence (strict <)"
    );
    println!(
        "[spatial_queries] degenerate: zero-size queries -- point AABB matches 1, \
         zero-radius sphere matches 0 even at exact coincidence"
    );

    // --- degenerate: extreme coordinates (wrapping, not panicking) ----------
    let extreme = Vec3Fix::new(
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX,
        },
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let extreme_bodies = vec![RigidBody::new_static(extreme)];
    let outcome = std::panic::catch_unwind(|| {
        let _ = overlap_sphere(extreme, sphere_radius, &extreme_bodies, body_radius);
        let _ = sphere_cast(
            Vec3Fix::ZERO,
            cast_radius,
            Vec3Fix::UNIT_X,
            Fix128::from_int(100),
            &extreme_bodies,
            cast_body_radius,
        );
    });
    assert!(
        outcome.is_ok(),
        "extreme coordinates wrap (Fix128 arithmetic is a mod-2^128 group), they must not panic"
    );
    println!("[spatial_queries] degenerate: extreme coordinates wrap, no panic");

    println!("[spatial_queries] all 12 previously-unwired query items exercised");
}
