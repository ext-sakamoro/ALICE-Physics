//! Analytic-oracle tests for `query`'s wiring pass.
//!
//! `query` had 12 items (`BatchRayQuery`/`OverlapResult`/`ShapeCastHit`/
//! `batch_raycast`/`batch_sphere_cast`/`capsule_cast`/`overlap_aabb`/
//! `overlap_aabb_bvh`/`overlap_aabb_expanded`/`overlap_sphere`/
//! `overlap_sphere_bvh`/`sphere_cast`) that production never called outside
//! the module's own `#[cfg(test)]` block. `examples/spatial_queries.rs` is
//! the production entry point; this file pins the closed forms
//! independently of that example and of the implementation under test.
//! Expected values are the ray/sphere quadratic (`t = -b - sqrt(b^2 - c)`,
//! `b = oc.dot(dir)`, `c = oc.dot(oc) - r^2`, from `src/raycast.rs`'s
//! `ray_sphere`) and the point-in-box / sphere-overlap formulas from
//! `src/query.rs`, worked by hand in each test's comment -- not produced by
//! calling the functions being checked.

#![cfg(feature = "std")]

use alice_physics::bvh::{BvhPrimitive, LinearBvh};
use alice_physics::collider::AABB;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::query::{
    batch_raycast, batch_sphere_cast, capsule_cast, overlap_aabb, overlap_aabb_bvh,
    overlap_aabb_expanded, overlap_sphere, overlap_sphere_bvh, sphere_cast, BatchRayQuery,
    OverlapResult, ShapeCastHit,
};
use alice_physics::solver::RigidBody;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn body_at(x: i64, y: i64, z: i64) -> RigidBody {
    RigidBody::new_static(Vec3Fix::from_int(x, y, z))
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

fn sorted_by_index(mut results: Vec<OverlapResult>) -> Vec<OverlapResult> {
    results.sort_by_key(|r| r.body_index);
    results
}

// ---------------------------------------------------------------------------
// overlap_sphere: strict-`<` boundary, and a non-trivial sqrt depth
// ---------------------------------------------------------------------------

/// center (0,0,0), radius 3, body_radius 1 -> combined_radius 4,
/// combined_sq 16. B0 (0,0,0) dist 0 -> overlap depth 4-0=4. B3 (-4,0,0)
/// dist 4, dist_sq 16 == combined_sq 16 -> excluded (strict `<`, touching
/// is not overlapping).
#[test]
fn overlap_sphere_excludes_exact_touch_and_pins_depth() {
    let bodies = vec![
        body_at(0, 0, 0),
        body_at(6, 0, 0),
        body_at(0, 6, 0),
        body_at(-4, 0, 0),
    ];
    let body_radius = Fix128::ONE;
    let results = overlap_sphere(Vec3Fix::ZERO, Fix128::from_int(3), &bodies, body_radius);
    assert_eq!(
        results.len(),
        1,
        "only body0 overlaps, B3 touches exactly and is excluded"
    );
    assert_eq!(results[0].body_index, 0);
    assert_eq!(results[0].depth, Fix128::from_int(4));
}

/// center (0,0,0), body at (3,4,0): dist = 5 (3-4-5 triangle, dist_sq=25
/// exact). radius 6 + body_radius 4 = combined_radius 10, combined_sq 100.
/// 25 < 100 -> overlap. depth = combined_radius - sqrt(dist_sq)
///           = 10 - sqrt(25) = 10 - 5 = 5 (exact, no rounding).
#[test]
fn overlap_sphere_depth_uses_sqrt_of_squared_distance() {
    let bodies = vec![body_at(3, 4, 0)];
    let body_radius = Fix128::from_int(4);
    let results = overlap_sphere(Vec3Fix::ZERO, Fix128::from_int(6), &bodies, body_radius);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].body_index, 0);
    assert_eq!(results[0].depth, Fix128::from_int(5));
}

// ---------------------------------------------------------------------------
// overlap_aabb / overlap_aabb_expanded
// ---------------------------------------------------------------------------

/// aabb [-1,-1,-1]..[7,1,1]: B0 (0,0,0) and B1 (6,0,0) are inside (x within
/// range, y=0 and z=0 within [-1,1]); B2 (0,6,0) fails the y bound; B3
/// (-4,0,0) fails the x bound.
#[test]
fn overlap_aabb_pins_exact_included_set() {
    let bodies = vec![
        body_at(0, 0, 0),
        body_at(6, 0, 0),
        body_at(0, 6, 0),
        body_at(-4, 0, 0),
    ];
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(7, 1, 1));
    let results = overlap_aabb(&aabb, &bodies);
    let mut idx: Vec<usize> = results.iter().map(|r| r.body_index).collect();
    idx.sort_unstable();
    assert_eq!(idx, vec![0, 1]);
    for r in &results {
        assert_eq!(r.depth, Fix128::ZERO, "point-in-AABB carries no depth");
    }
}

/// Zero expansion must equal `overlap_aabb` exactly (same elements, same
/// order, same depth values) -- not merely the same set.
#[test]
fn overlap_aabb_expanded_zero_margin_is_identity() {
    let bodies = vec![
        body_at(0, 0, 0),
        body_at(6, 0, 0),
        body_at(0, 6, 0),
        body_at(-4, 0, 0),
    ];
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(7, 1, 1));
    let plain = overlap_aabb(&aabb, &bodies);
    let expanded = overlap_aabb_expanded(&aabb, &bodies, Fix128::ZERO);
    assert_eq!(expanded, plain);
}

/// aabb2 [2,-1,-1]..[4,1,1] contains no body (B0 x=0, B1 x=6 both outside
/// [2,4]). Expanding by margin 2 -> [0,-3,-3]..[6,3,3]: B0 x=0 in [0,6] and
/// B1 x=6 in [0,6] are now inside; B2/B3 remain outside (y=6 / x=-4).
#[test]
fn overlap_aabb_expanded_adds_exactly_the_reached_bodies() {
    let bodies = vec![
        body_at(0, 0, 0),
        body_at(6, 0, 0),
        body_at(0, 6, 0),
        body_at(-4, 0, 0),
    ];
    let aabb2 = AABB::new(Vec3Fix::from_int(2, -1, -1), Vec3Fix::from_int(4, 1, 1));
    assert!(overlap_aabb(&aabb2, &bodies).is_empty());
    let expanded = overlap_aabb_expanded(&aabb2, &bodies, Fix128::from_int(2));
    let mut idx: Vec<usize> = expanded.iter().map(|r| r.body_index).collect();
    idx.sort_unstable();
    assert_eq!(idx, vec![0, 1]);
}

// ---------------------------------------------------------------------------
// overlap_sphere_bvh / overlap_aabb_bvh: BVH must agree exactly with brute
// ---------------------------------------------------------------------------

/// Same scene and queries as the brute-force tests above: the BVH-pruned
/// candidate set, after the identical narrow-phase test, must produce the
/// exact same (sorted) result set as the brute-force scan. This is the
/// accelerated-path-matches-reference-path oracle, checked on two different
/// scenes (a sphere query and an AABB query) rather than any fixed closed
/// form of the BVH's internal structure.
#[test]
fn overlap_bvh_matches_brute_force_on_sphere_and_aabb_queries() {
    let bodies = vec![
        body_at(0, 0, 0),
        body_at(6, 0, 0),
        body_at(0, 6, 0),
        body_at(-4, 0, 0),
    ];
    let body_radius = Fix128::ONE;
    let bvh = build_bvh(&bodies, body_radius);

    let sphere_center = Vec3Fix::ZERO;
    let sphere_radius = Fix128::from_int(3);
    let brute_sphere = sorted_by_index(overlap_sphere(
        sphere_center,
        sphere_radius,
        &bodies,
        body_radius,
    ));
    let bvh_sphere = sorted_by_index(overlap_sphere_bvh(
        sphere_center,
        sphere_radius,
        &bodies,
        body_radius,
        &bvh,
    ));
    assert_eq!(bvh_sphere, brute_sphere);

    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(7, 1, 1));
    let brute_aabb = sorted_by_index(overlap_aabb(&aabb, &bodies));
    let bvh_aabb = sorted_by_index(overlap_aabb_bvh(&aabb, &bodies, &bvh));
    assert_eq!(bvh_aabb, brute_aabb);
}

/// A second, denser scene (8 bodies on a grid) so the BVH actually branches
/// (more than one leaf), still checked against brute force.
#[test]
fn overlap_bvh_matches_brute_force_on_a_denser_grid() {
    let bodies: Vec<RigidBody> = (0..8).map(|i| body_at(i * 3, (i % 3) * 3, 0)).collect();
    let body_radius = Fix128::ONE;
    let bvh = build_bvh(&bodies, body_radius);

    let sphere_center = Vec3Fix::from_int(9, 3, 0);
    let sphere_radius = Fix128::from_int(5);
    let brute = sorted_by_index(overlap_sphere(
        sphere_center,
        sphere_radius,
        &bodies,
        body_radius,
    ));
    let accel = sorted_by_index(overlap_sphere_bvh(
        sphere_center,
        sphere_radius,
        &bodies,
        body_radius,
        &bvh,
    ));
    assert_eq!(accel, brute);
    assert!(
        !brute.is_empty(),
        "scene must produce at least one overlap to be a non-vacuous check"
    );

    let aabb = AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(12, 6, 0));
    let brute_aabb = sorted_by_index(overlap_aabb(&aabb, &bodies));
    let accel_aabb = sorted_by_index(overlap_aabb_bvh(&aabb, &bodies, &bvh));
    assert_eq!(accel_aabb, brute_aabb);
    assert!(!brute_aabb.is_empty());
}

/// Empty BVH (built from zero primitives) must agree with brute force on
/// an empty world: both empty, neither panics.
#[test]
fn overlap_bvh_on_empty_world_matches_brute_force() {
    let bodies: Vec<RigidBody> = Vec::new();
    let body_radius = Fix128::ONE;
    let bvh = build_bvh(&bodies, body_radius);
    assert!(overlap_sphere(Vec3Fix::ZERO, Fix128::from_int(3), &bodies, body_radius).is_empty());
    assert!(overlap_sphere_bvh(
        Vec3Fix::ZERO,
        Fix128::from_int(3),
        &bodies,
        body_radius,
        &bvh
    )
    .is_empty());
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(7, 1, 1));
    assert!(overlap_aabb(&aabb, &bodies).is_empty());
    assert!(overlap_aabb_bvh(&aabb, &bodies, &bvh).is_empty());
}

// ---------------------------------------------------------------------------
// sphere_cast / capsule_cast: ray/sphere quadratic, derived in-test
// ---------------------------------------------------------------------------

/// Two bodies on the X axis, body_radius 1, cast_radius 1 (combined 2):
///   body0 (5,0,0):  oc=(-5,0,0) b=-5 c=25-4=21 disc=4  sqrt=2 t=5-2=3
///   body1 (15,0,0): oc=(-15,0,0) b=-15 c=225-4=221 disc=4 sqrt=2 t=15-2=13
/// Closest is body0 at t=3.
#[test]
fn sphere_cast_picks_the_closest_of_multiple_bodies() {
    let bodies = vec![body_at(5, 0, 0), body_at(15, 0, 0)];
    let hit: ShapeCastHit = sphere_cast(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    )
    .expect("must hit body0");
    assert_eq!(hit.body_index, 0);
    assert_eq!(hit.t, Fix128::from_int(3));
    assert_eq!(hit.point, Vec3Fix::from_int(3, 0, 0));
    assert_eq!(
        hit.normal,
        Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
}

/// Ray that starts inside a body (distance 1 < combined_radius 2) and moves
/// deeper is an initial overlap: oc=(-1,0,0) b=oc.d=-1<0 c=1-4=-3<0, so the
/// contact is at t=0 at the origin with normal oc/|oc|=(-1,0,0), opposing the
/// cast. The exit root t=1+2=3 is never a contact (AUD-A-S3W3-013).
#[test]
fn sphere_cast_from_inside_a_body_moving_deeper_reports_contact_at_t0() {
    let bodies = vec![body_at(5, 0, 0)];
    let hit = sphere_cast(
        Vec3Fix::from_int(4, 0, 0),
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    )
    .expect("interior origin moving deeper reports the initial overlap");
    assert_eq!(hit.t, Fix128::ZERO, "initial overlap, not the exit t=3");
    assert_eq!(hit.point, Vec3Fix::from_int(4, 0, 0));
    assert_eq!(
        hit.normal,
        Vec3Fix::new(-Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
}

/// The same interior origin moving away from the centre (b=+1>=0) is leaving
/// the overlap: no contact (AUD-A-S3W3-013).
#[test]
fn sphere_cast_from_inside_a_body_moving_out_reports_no_contact() {
    let bodies = vec![body_at(5, 0, 0)];
    let hit = sphere_cast(
        Vec3Fix::from_int(4, 0, 0),
        Fix128::ONE,
        -Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    );
    assert!(hit.is_none(), "leaving the overlap is not a contact");
}

/// Zero-radius sphere cast must equal a plain raycast: combined_radius
/// collapses to body_radius alone. oc=(-5,0,0) b=-5 c=25-1=24 disc=1
/// sqrt=1 t=5-1=4.
#[test]
fn sphere_cast_zero_radius_behaves_like_a_raycast() {
    let bodies = vec![body_at(5, 0, 0)];
    let hit = sphere_cast(
        Vec3Fix::ZERO,
        Fix128::ZERO,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    )
    .expect("zero-radius cast still hits the body surface");
    assert_eq!(hit.t, Fix128::from_int(4));
}

/// Capsule (4,-2,0)-(4,2,0), capsule_radius 1, body (10,0,0), body_radius 1
/// (combined_radius 2 for every sub-cast):
///   from (4,-2,0): oc=(-6,-2,0) b=-6 c=40-4=36 disc=0  t=6 (tangent)
///   from (4, 2,0): oc=(-6, 2,0) b=-6 c=40-4=36 disc=0  t=6 (tangent)
///   from midpoint (4,0,0): oc=(-6,0,0) b=-6 c=36-4=32 disc=4 sqrt=2 t=4
/// The midpoint sub-cast is strictly closest and must win.
///
/// (Mutation note: a `<` -> `<=` tie-break mutation in `capsule_cast`'s
/// final selection survives every test here. For any capsule whose two
/// endpoints share the cast-direction coordinate (the common case: the
/// capsule is perpendicular to the cast direction, as here), the ray/sphere
/// linear coefficient `b` is identical across all three sub-casts, so `t`
/// is a strictly decreasing function of squared perpendicular offset. By
/// the strict triangle inequality, the midpoint offset's magnitude is
/// strictly less than that of two *distinct* endpoints with equal offset
/// magnitude, so a tie between the endpoints while excluding the midpoint
/// requires the degenerate case `capsule_a == capsule_b`. We did not find a
/// non-degenerate construction (e.g. a capsule tilted along the cast
/// direction, where `b` differs per sub-cast) within this pass's time
/// budget; reported as a probable equivalent mutation for the common case,
/// unverified for tilted capsules.)
#[test]
fn capsule_cast_picks_the_closest_of_its_three_sub_casts() {
    let bodies = vec![body_at(10, 0, 0)];
    let hit = capsule_cast(
        Vec3Fix::from_int(4, -2, 0),
        Vec3Fix::from_int(4, 2, 0),
        Fix128::ONE,
        Vec3Fix::UNIT_X,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    )
    .expect("capsule_cast should hit");
    assert_eq!(hit.t, Fix128::from_int(4));
    assert_eq!(hit.point, Vec3Fix::from_int(8, 0, 0));
}

/// `overlap_aabb` must reject a body that is inside the x/y range but
/// outside the z range -- a mutation that drops the z bound only would
/// survive every other test in this file (every other scene has z=0 and a
/// z range that covers it). Body0 (0,0,0) is inside; body1 (0,0,10) has
/// x=0,y=0 inside [-1,1] but z=10 outside [-1,1].
#[test]
fn overlap_aabb_rejects_z_outside_range_even_with_xy_inside() {
    let bodies = vec![body_at(0, 0, 0), body_at(0, 0, 10)];
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
    let results = overlap_aabb(&aabb, &bodies);
    let idx: Vec<usize> = results.iter().map(|r| r.body_index).collect();
    assert_eq!(idx, vec![0], "body1 must be rejected by the z bound alone");
}

/// Same z-exclusion, but through the BVH path, with the BVH built using a
/// deliberately large `body_radius` (2) so that body1's *bounding box*
/// still overlaps the query AABB in z (passing broad-phase as a candidate)
/// even though its exact position does not satisfy the narrow-phase z
/// range. This is required to actually exercise the narrow-phase z check:
/// with a tight BVH radius, broad-phase pruning alone would already
/// exclude body1 and the mutation would be unreachable.
///   body1 z=2, bvh body_radius=2 -> leaf AABB z in [0,4], which overlaps
///   the query AABB's z=[-1,1] (max(0,-1)=0 <= min(4,1)=1), so body1 is a
///   candidate; its exact z=2 is still outside [-1,1].
#[test]
fn overlap_aabb_bvh_rejects_z_outside_range_when_broad_phase_still_admits_it() {
    let bodies = vec![body_at(0, 0, 0), body_at(0, 0, 2)];
    let bvh_radius = Fix128::from_int(2);
    let bvh = build_bvh(&bodies, bvh_radius);
    let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));

    // Sanity: body1 actually survives broad-phase pruning as a candidate.
    let candidates = bvh.query(&aabb);
    assert!(
        candidates.contains(&1),
        "test setup must keep body1 as a broad-phase candidate, or this test is vacuous"
    );

    let results = overlap_aabb_bvh(&aabb, &bodies, &bvh);
    let idx: Vec<usize> = results.iter().map(|r| r.body_index).collect();
    assert_eq!(
        idx,
        vec![0],
        "body1 must still be rejected by the narrow-phase z bound"
    );
}

/// `batch_raycast` must use the caller's `body_radius`, not a hardcoded
/// value: with body_radius 2 (not the 1 used everywhere else in this
/// file), oc=(-5,0,0) b=-5 c=25-4=21 disc=4 sqrt=2 t=5-2=3 -- distinct
/// from the t=4 that body_radius=1 would give (every other batch_raycast
/// test in this file uses body_radius=1, so a mutation hardcoding it to 1
/// would otherwise survive).
#[test]
fn batch_raycast_uses_the_given_body_radius_not_a_constant() {
    let bodies = vec![body_at(5, 0, 0)];
    let queries = vec![BatchRayQuery {
        origin: Vec3Fix::ZERO,
        direction: Vec3Fix::UNIT_X,
        max_distance: Fix128::from_int(100),
    }];
    let results = batch_raycast(&queries, &bodies, Fix128::from_int(2));
    let hit = results[0].expect("should hit");
    assert_eq!(hit.t, Fix128::from_int(3));
}

// ---------------------------------------------------------------------------
// batch_raycast / batch_sphere_cast: batch == loop of singles
// ---------------------------------------------------------------------------

/// Three rays with distinct outcomes (hit nearest body, miss, hit from the
/// other side) must produce exactly what calling `sphere_cast` once per ray
/// produces.
#[test]
fn batch_raycast_equals_loop_of_sphere_cast_zero_radius() {
    let bodies = vec![body_at(5, 0, 0), body_at(15, 0, 0)];
    let body_radius = Fix128::ONE;
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
    let batch: Vec<Option<ShapeCastHit>> = batch_raycast(&queries, &bodies, body_radius);
    let looped: Vec<Option<ShapeCastHit>> = queries
        .iter()
        .map(|q| {
            sphere_cast(
                q.origin,
                Fix128::ZERO,
                q.direction,
                q.max_distance,
                &bodies,
                body_radius,
            )
        })
        .collect();
    assert_eq!(batch, looped);
    assert!(batch[0].is_some());
    assert!(batch[1].is_none());
    assert_eq!(batch[2].expect("Q2 hits body1").body_index, 1);
}

/// Empty batch returns an empty vec, not a panic.
#[test]
fn batch_raycast_on_empty_queries_and_empty_world() {
    let bodies: Vec<RigidBody> = Vec::new();
    assert!(batch_raycast(&[], &bodies, Fix128::ONE).is_empty());
    let non_empty = vec![body_at(5, 0, 0)];
    assert!(batch_raycast(&[], &non_empty, Fix128::ONE).is_empty());
}

/// Two (origin, direction) pairs, one hits, one misses, must equal the
/// per-pair `sphere_cast` loop.
#[test]
fn batch_sphere_cast_equals_loop_of_sphere_cast() {
    let bodies = vec![body_at(5, 0, 0), body_at(15, 0, 0)];
    let body_radius = Fix128::ONE;
    let radius = Fix128::ONE;
    let origins = vec![Vec3Fix::ZERO, Vec3Fix::ZERO];
    let directions = vec![Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y];
    let batch = batch_sphere_cast(
        &origins,
        radius,
        &directions,
        Fix128::from_int(100),
        &bodies,
        body_radius,
    );
    let looped: Vec<Option<ShapeCastHit>> = origins
        .iter()
        .zip(directions.iter())
        .map(|(o, d)| sphere_cast(*o, radius, *d, Fix128::from_int(100), &bodies, body_radius))
        .collect();
    assert_eq!(batch, looped);
    assert!(batch[0].is_some());
    assert!(batch[1].is_none());
}

/// Degenerate: mismatched `origins`/`directions` lengths panic (documented
/// in `src/query.rs`'s `# Panics` section), they do not silently truncate.
#[test]
fn batch_sphere_cast_panics_on_mismatched_lengths() {
    let bodies = vec![body_at(5, 0, 0)];
    let body_radius = Fix128::ONE;
    let origins = vec![Vec3Fix::ZERO, Vec3Fix::ZERO];
    let directions = vec![Vec3Fix::UNIT_X]; // one short
    let result = catch_unwind(AssertUnwindSafe(|| {
        batch_sphere_cast(
            &origins,
            Fix128::ONE,
            &directions,
            Fix128::from_int(100),
            &bodies,
            body_radius,
        )
    }));
    assert!(
        result.is_err(),
        "mismatched lengths must panic, not silently zip-truncate"
    );
}

// ---------------------------------------------------------------------------
// Degenerate inputs: zero-length direction, zero-size queries, overflow
// ---------------------------------------------------------------------------

/// Zero-length direction is documented to return `None` (the early-return
/// guard in `sphere_cast`), not to fall back to an arbitrary axis.
#[test]
fn sphere_cast_zero_length_direction_returns_none() {
    let bodies = vec![body_at(5, 0, 0)];
    let hit = sphere_cast(
        Vec3Fix::ZERO,
        Fix128::ONE,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    );
    assert!(
        hit.is_none(),
        "zero-length direction must not hit via a fallback axis"
    );
}

#[test]
fn capsule_cast_zero_length_direction_returns_none() {
    let bodies = vec![body_at(5, 0, 0)];
    let hit = capsule_cast(
        Vec3Fix::from_int(0, -1, 0),
        Vec3Fix::from_int(0, 1, 0),
        Fix128::ONE,
        Vec3Fix::ZERO,
        Fix128::from_int(100),
        &bodies,
        Fix128::ONE,
    );
    assert!(hit.is_none());
}

/// A degenerate (zero-size, point) AABB matches only a body at that exact
/// point.
#[test]
fn overlap_aabb_zero_size_box_matches_only_the_exact_point() {
    let bodies = vec![body_at(0, 0, 0), body_at(6, 0, 0)];
    let point_aabb = AABB::new(Vec3Fix::from_int(6, 0, 0), Vec3Fix::from_int(6, 0, 0));
    let results = overlap_aabb(&point_aabb, &bodies);
    let idx: Vec<usize> = results.iter().map(|r| r.body_index).collect();
    assert_eq!(idx, vec![1]);
}

/// Zero sphere radius AND zero body radius: combined_sq is 0, and the
/// strict `<` comparison in `overlap_sphere` excludes `dist_sq == 0`, so an
/// exact coincidence does *not* register as an overlap. This is the same
/// strict-boundary semantics as the touching case above, now visible at
/// distance 0.
#[test]
fn overlap_sphere_zero_radius_never_matches_even_exact_coincidence() {
    let bodies = vec![body_at(0, 0, 0)];
    let results = overlap_sphere(Vec3Fix::ZERO, Fix128::ZERO, &bodies, Fix128::ZERO);
    assert!(
        results.is_empty(),
        "zero combined radius excludes dist_sq == 0 under strict <"
    );
}

/// Extreme coordinates (near `Fix128::MAX`) must wrap (the documented
/// mod-2^128 group semantics of `Fix128` arithmetic), never panic.
#[test]
fn extreme_coordinates_wrap_without_panicking() {
    let extreme = Vec3Fix::new(
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX,
        },
        Fix128::ZERO,
        Fix128::ZERO,
    );
    let bodies = vec![RigidBody::new_static(extreme)];
    let body_radius = Fix128::ONE;
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let _ = overlap_sphere(extreme, Fix128::from_int(3), &bodies, body_radius);
        let _ = overlap_aabb(
            &AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1)),
            &bodies,
        );
        let _ = sphere_cast(
            Vec3Fix::ZERO,
            Fix128::ONE,
            Vec3Fix::UNIT_X,
            Fix128::from_int(100),
            &bodies,
            body_radius,
        );
    }));
    assert!(outcome.is_ok(), "extreme coordinates must wrap, not panic");
}
