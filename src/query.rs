//! Shape Cast and Overlap Queries
//!
//! Spatial queries for game logic: sphere/capsule cast, overlap tests.
//! BVH-accelerated broad phase with exact narrow phase.
//!
//! # Features
//!
//! - `sphere_cast`: Sweep a sphere along a direction
//! - `capsule_cast`: Sweep a capsule along a direction
//! - `overlap_sphere`: Find all bodies overlapping a sphere
//! - `overlap_aabb`: Find all bodies overlapping an AABB
//!
//! Author: Moroya Sakamoto

use crate::collider::{Sphere, AABB};
use crate::math::{Fix128, Vec3Fix};
use crate::raycast::{sweep_ray_sphere, Ray};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// Query Results
// ============================================================================

/// Result of a shape cast query
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeCastHit {
    /// Distance along the cast direction
    pub t: Fix128,
    /// World-space hit point
    pub point: Vec3Fix,
    /// Surface normal at hit point
    pub normal: Vec3Fix,
    /// Index of the hit body
    pub body_index: usize,
}

/// Result of an overlap query
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OverlapResult {
    /// Index of the overlapping body
    pub body_index: usize,
    /// Overlap depth (penetration distance)
    pub depth: Fix128,
}

// ============================================================================
// Shape Cast Functions
// ============================================================================

/// Cast a sphere along a direction against rigid bodies.
///
/// Returns the closest hit, or None if no intersection.
///
/// This is equivalent to a "thick raycast" — useful for character collision,
/// projectile sweeps, and visibility checks.
#[inline]
#[must_use]
pub fn sphere_cast(
    origin: Vec3Fix,
    radius: Fix128,
    direction: Vec3Fix,
    max_distance: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Option<ShapeCastHit> {
    let dir_len = direction.length();
    if dir_len.is_zero() {
        return None;
    }
    let dir_norm = direction / dir_len;
    let ray = Ray::new(origin, dir_norm);

    let mut closest: Option<ShapeCastHit> = None;
    let mut best_t = max_distance;

    for (i, body) in bodies.iter().enumerate() {
        // Expand the body sphere by the cast sphere radius (Minkowski sum)
        let expanded = Sphere::new(body.position, body_radius + radius);

        if let Some(hit) = sweep_ray_sphere(&ray, &expanded, best_t) {
            best_t = hit.t;
            closest = Some(ShapeCastHit {
                t: hit.t,
                point: hit.point,
                normal: hit.normal,
                body_index: i,
            });
        }
    }

    closest
}

/// Cast a capsule along a direction against rigid bodies.
///
/// The capsule (segment `capsule_a`–`capsule_b`, radius `capsule_radius`) is
/// swept along `direction` up to `max_distance`; each body is a sphere of
/// `body_radius` at its position. The hit is the earliest `t` at which the
/// swept segment comes within `capsule_radius + body_radius` of a body centre
/// — the exact time of impact, computed as a ray from the body centre along
/// `−direction` against the capsule grown by `body_radius`: its cylinder
/// (a quadratic in the plane across the axis) and its two end spheres. A
/// body already within reach at `t = 0` is hit at `t = 0` if the sweep moves
/// the capsule towards it, as for [`sphere_cast`].
///
/// `point` is the point of the moved axis nearest the body centre (the centre
/// of the cast sphere that touches it) and `normal` points from the body
/// centre to it; ties in `t` go to the lowest body index.
// LIMITATION(COV-RIGID-098): Bodies are treated as their bounding spheres of `body_radius`, not their shapes.
#[must_use]
pub fn capsule_cast(
    capsule_a: Vec3Fix,
    capsule_b: Vec3Fix,
    capsule_radius: Fix128,
    direction: Vec3Fix,
    max_distance: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Option<ShapeCastHit> {
    let dir_len = direction.length();
    if dir_len.is_zero() {
        return None;
    }
    let d = direction / dir_len;
    let reach = capsule_radius + body_radius;
    let axis = capsule_b - capsule_a;
    let len = axis.length();
    if len.is_zero() {
        return sphere_cast(
            capsule_a,
            capsule_radius,
            direction,
            max_distance,
            bodies,
            body_radius,
        );
    }
    let u = axis / len;

    let mut best: Option<ShapeCastHit> = None;
    for (i, body) in bodies.iter().enumerate() {
        let limit = best.map_or(max_distance, |b| b.t);
        let Some(t) = capsule_toi(capsule_a, capsule_b, u, len, reach, d, body.position, limit)
        else {
            continue;
        };
        if best.is_some_and(|b| t >= b.t) {
            continue;
        }
        // the moved axis point nearest the body centre
        let c = body.position;
        let s = ((c - capsule_a - d * t).dot(u)).max(Fix128::ZERO).min(len);
        let point = capsule_a + d * t + u * s;
        let to = point - c;
        let normal = if to.length_squared().is_zero() {
            -d
        } else {
            to.normalize()
        };
        best = Some(ShapeCastHit {
            t,
            point,
            normal,
            body_index: i,
        });
    }
    best
}

/// Earliest `t` in `[0, max_t]` at which the segment `a + t d`–`b + t d`
/// (unit axis `u`, length `len`) comes within `reach` of `c`: a ray from `c`
/// along `−d` against the capsule of radius `reach` around `a`–`b`.
#[allow(clippy::too_many_arguments)]
fn capsule_toi(
    a: Vec3Fix,
    b: Vec3Fix,
    u: Vec3Fix,
    len: Fix128,
    reach: Fix128,
    d: Vec3Fix,
    c: Vec3Fix,
    max_t: Fix128,
) -> Option<Fix128> {
    // Already within reach at t = 0: the distance from c to the moving segment
    // is convex in t, so the sweep either closes in now (contact at t = 0) or
    // never comes back within reach (no contact), the rule of sphere_cast;
    // the end spheres would otherwise report a later "entry" from inside
    let s0 = (c - a).dot(u).max(Fix128::ZERO).min(len);
    let to_body = c - (a + u * s0);
    if to_body.length_squared() <= reach * reach {
        return if to_body.length_squared().is_zero() || to_body.dot(d) > Fix128::ZERO {
            Some(Fix128::ZERO)
        } else {
            None
        };
    }
    let ray = Ray::new(c, -d);
    let mut best: Option<Fix128> = None;
    let mut take = |t: Fix128| {
        if t >= Fix128::ZERO && t <= max_t && best.is_none_or(|b| t < b) {
            best = Some(t);
        }
    };
    // the end spheres (their overlap rule at t = 0 is sphere_cast's)
    for end in [a, b] {
        if let Some(h) = sweep_ray_sphere(&ray, &Sphere::new(end, reach), max_t) {
            take(h.t);
        }
    }
    // the cylinder: components across the axis
    let rd = -d;
    let oa = c - a;
    let oa_p = oa - u * oa.dot(u);
    let rd_p = rd - u * rd.dot(u);
    let qa = rd_p.dot(rd_p);
    let qb = oa_p.dot(rd_p);
    let qc = oa_p.dot(oa_p) - reach * reach;
    let along = |t: Fix128| oa.dot(u) + rd.dot(u) * t;
    let on_side = |t: Fix128| {
        let y = along(t);
        y >= Fix128::ZERO && y <= len
    };
    if qc > Fix128::ZERO && !qa.is_zero() {
        let disc = qb * qb - qa * qc;
        if disc >= Fix128::ZERO {
            let t = (-qb - disc.sqrt()) / qa;
            if on_side(t) {
                take(t);
            }
        }
    }
    best
}

// ============================================================================
// Overlap Functions
// ============================================================================

/// Find all bodies whose bounding sphere overlaps with the given sphere.
// LIMITATION(COV-RIGID-100): Find all bodies whose bounding sphere overlaps with the given sphere.
///
/// `body_radius` is the assumed radius for each body.
#[inline]
#[must_use]
pub fn overlap_sphere(
    center: Vec3Fix,
    radius: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Vec<OverlapResult> {
    let mut results = Vec::new();
    let combined_radius = radius + body_radius;
    let combined_sq = combined_radius * combined_radius;

    for (i, body) in bodies.iter().enumerate() {
        let delta = body.position - center;
        let dist_sq = delta.length_squared();

        if dist_sq < combined_sq {
            let dist = dist_sq.sqrt();
            let depth = combined_radius - dist;
            results.push(OverlapResult {
                body_index: i,
                depth,
            });
        }
    }

    results
}

/// Find all bodies whose position falls within the given AABB.
///
/// Uses the body position as a point test (no body extent considered).
/// For volume overlap, expand the AABB by the body radius first.
#[inline]
#[must_use]
pub fn overlap_aabb(aabb: &AABB, bodies: &[RigidBody]) -> Vec<OverlapResult> {
    let mut results = Vec::new();

    for (i, body) in bodies.iter().enumerate() {
        let p = body.position;
        if p.x >= aabb.min.x
            && p.x <= aabb.max.x
            && p.y >= aabb.min.y
            && p.y <= aabb.max.y
            && p.z >= aabb.min.z
            && p.z <= aabb.max.z
        {
            results.push(OverlapResult {
                body_index: i,
                depth: Fix128::ZERO, // Point-in-AABB doesn't have a depth
            });
        }
    }

    results
}

/// Find all bodies overlapping an AABB, with body radius consideration.
///
/// Each body is treated as a sphere of `body_radius`: it overlaps when the
/// point of the AABB nearest its centre is within `body_radius` (boundary
/// included). `depth` is zero, as for [`overlap_aabb`].
#[must_use]
pub fn overlap_aabb_expanded(
    aabb: &AABB,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Vec<OverlapResult> {
    let r_sq = body_radius * body_radius;
    bodies
        .iter()
        .enumerate()
        .filter(|(_, body)| {
            let p = body.position;
            let nearest = Vec3Fix::new(
                p.x.max(aabb.min.x).min(aabb.max.x),
                p.y.max(aabb.min.y).min(aabb.max.y),
                p.z.max(aabb.min.z).min(aabb.max.z),
            );
            (p - nearest).length_squared() <= r_sq
        })
        .map(|(i, _)| OverlapResult {
            body_index: i,
            depth: Fix128::ZERO,
        })
        .collect()
}

// ============================================================================
// Batch Queries
// ============================================================================

/// A single raycast query for batch execution
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchRayQuery {
    /// Ray origin
    pub origin: Vec3Fix,
    /// Ray direction (will be normalized)
    pub direction: Vec3Fix,
    /// Maximum cast distance
    pub max_distance: Fix128,
}

/// Execute multiple raycasts in batch against rigid bodies.
///
/// Returns one result per query (closest hit or None).
/// Each ray is treated as a zero-radius sphere cast.
/// When `parallel` feature is enabled, queries execute in parallel via Rayon.
///
/// Every body is a sphere of `body_radius` (a bounding-sphere approximation): the
/// body's own shape is not consulted. To cast against the geometry a
/// [`PhysicsWorld`](crate::solver::PhysicsWorld) collides with, use
/// [`PhysicsWorld::ray_caster`](crate::solver::PhysicsWorld::ray_caster) and cast
/// each ray with the BVH built once.
#[must_use]
pub fn batch_raycast(
    queries: &[BatchRayQuery],
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Vec<Option<ShapeCastHit>> {
    #[cfg(feature = "parallel")]
    {
        queries
            .par_iter()
            .map(|q| {
                sphere_cast(
                    q.origin,
                    Fix128::ZERO,
                    q.direction,
                    q.max_distance,
                    bodies,
                    body_radius,
                )
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        queries
            .iter()
            .map(|q| {
                sphere_cast(
                    q.origin,
                    Fix128::ZERO,
                    q.direction,
                    q.max_distance,
                    bodies,
                    body_radius,
                )
            })
            .collect()
    }
}

/// Execute multiple sphere casts in batch against rigid bodies.
///
/// `origins` and `directions` must have the same length.
/// Returns one result per query (closest hit or None).
/// When `parallel` feature is enabled, queries execute in parallel via Rayon.
///
/// # Panics
///
/// Panics if `origins` and `directions` have different lengths.
#[must_use]
pub fn batch_sphere_cast(
    origins: &[Vec3Fix],
    radius: Fix128,
    directions: &[Vec3Fix],
    max_distance: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Vec<Option<ShapeCastHit>> {
    assert_eq!(origins.len(), directions.len());

    #[cfg(feature = "parallel")]
    {
        origins
            .par_iter()
            .zip(directions.par_iter())
            .map(|(origin, direction)| {
                sphere_cast(
                    *origin,
                    radius,
                    *direction,
                    max_distance,
                    bodies,
                    body_radius,
                )
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        origins
            .iter()
            .zip(directions.iter())
            .map(|(origin, direction)| {
                sphere_cast(
                    *origin,
                    radius,
                    *direction,
                    max_distance,
                    bodies,
                    body_radius,
                )
            })
            .collect()
    }
}

// ============================================================================
// BVH-Accelerated Queries
// ============================================================================

/// Find all bodies overlapping a sphere, using a BVH for broad-phase pruning.
///
/// The BVH must be built from per-body AABBs (expanded by `body_radius`).
/// Only candidates whose BVH leaf overlaps the query sphere AABB are tested
/// in the narrow phase, reducing cost from O(n) to O(log n + k).
#[must_use]
pub fn overlap_sphere_bvh(
    center: Vec3Fix,
    radius: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
    bvh: &crate::bvh::LinearBvh,
) -> Vec<OverlapResult> {
    let combined_radius = radius + body_radius;
    let query_aabb = AABB::new(
        Vec3Fix::new(
            center.x - combined_radius,
            center.y - combined_radius,
            center.z - combined_radius,
        ),
        Vec3Fix::new(
            center.x + combined_radius,
            center.y + combined_radius,
            center.z + combined_radius,
        ),
    );

    let candidates = bvh.query(&query_aabb);
    let combined_sq = combined_radius * combined_radius;
    let mut results = Vec::new();

    for &idx in &candidates {
        let i = idx as usize;
        if i >= bodies.len() {
            continue;
        }
        let delta = bodies[i].position - center;
        let dist_sq = delta.length_squared();
        if dist_sq < combined_sq {
            let dist = dist_sq.sqrt();
            let depth = combined_radius - dist;
            results.push(OverlapResult {
                body_index: i,
                depth,
            });
        }
    }

    results
}

/// Find all bodies overlapping an AABB, using a BVH for broad-phase pruning.
///
/// Only candidates whose BVH leaf overlaps the query AABB are tested,
/// reducing cost from O(n) to O(log n + k).
#[must_use]
pub fn overlap_aabb_bvh(
    aabb: &AABB,
    bodies: &[RigidBody],
    bvh: &crate::bvh::LinearBvh,
) -> Vec<OverlapResult> {
    let candidates = bvh.query(aabb);
    let mut results = Vec::new();

    for &idx in &candidates {
        let i = idx as usize;
        if i >= bodies.len() {
            continue;
        }
        let p = bodies[i].position;
        if p.x >= aabb.min.x
            && p.x <= aabb.max.x
            && p.y >= aabb.min.y
            && p.y <= aabb.max.y
            && p.z >= aabb.min.z
            && p.z <= aabb.max.z
        {
            results.push(OverlapResult {
                body_index: i,
                depth: Fix128::ZERO,
            });
        }
    }

    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn make_bodies() -> Vec<RigidBody> {
        vec![
            RigidBody::new_static(Vec3Fix::from_int(0, 0, 0)),
            RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)),
            RigidBody::new_static(Vec3Fix::from_int(10, 0, 0)),
            RigidBody::new_static(Vec3Fix::from_int(0, 5, 0)),
        ]
    }

    #[test]
    fn test_sphere_cast_hit() {
        let bodies = make_bodies();
        let hit = sphere_cast(
            Vec3Fix::from_int(-10, 0, 0),
            Fix128::from_ratio(1, 2), // radius 0.5
            Vec3Fix::UNIT_X,
            Fix128::from_int(100),
            &bodies,
            Fix128::ONE,
        );
        assert!(hit.is_some(), "Sphere cast should hit body at origin");
        let h = hit.expect("sphere cast should hit");
        assert_eq!(h.body_index, 0);
        // Should hit at approximately t = 10 - 1.5 = 8.5 (distance minus combined radii)
        assert!(h.t > Fix128::from_int(5), "Hit should be before the body");
    }

    #[test]
    fn test_sphere_cast_miss() {
        let bodies = make_bodies();
        let hit = sphere_cast(
            Vec3Fix::from_int(-10, 10, 0), // above all bodies
            Fix128::from_ratio(1, 2),
            Vec3Fix::UNIT_X,
            Fix128::from_int(100),
            &bodies,
            Fix128::ONE,
        );
        assert!(hit.is_none(), "Sphere cast should miss (y offset)");
    }

    #[test]
    fn test_capsule_cast() {
        let bodies = make_bodies();
        let hit = capsule_cast(
            Vec3Fix::from_int(-10, -1, 0),
            Vec3Fix::from_int(-10, 1, 0),
            Fix128::from_ratio(1, 2),
            Vec3Fix::UNIT_X,
            Fix128::from_int(100),
            &bodies,
            Fix128::ONE,
        );
        assert!(hit.is_some(), "Capsule cast should hit");
    }

    #[test]
    fn test_overlap_sphere() {
        let bodies = make_bodies();
        let results = overlap_sphere(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::from_int(3),
            &bodies,
            Fix128::ONE,
        );
        // Body 0 at origin (dist=0 < 4), body 3 at (0,5,0) (dist=5 < 4? no, 5>4)
        // Body 1 at (5,0,0) (dist=5 < 4? no)
        // Only body 0 should overlap
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].body_index, 0);
    }

    #[test]
    fn test_overlap_sphere_multiple() {
        let bodies = make_bodies();
        let results = overlap_sphere(
            Vec3Fix::from_int(2, 0, 0),
            Fix128::from_int(5),
            &bodies,
            Fix128::ONE,
        );
        // Body 0 at (0,0,0): dist=2, combined=6 → overlap
        // Body 1 at (5,0,0): dist=3, combined=6 → overlap
        // Body 2 at (10,0,0): dist=8, combined=6 → no
        // Body 3 at (0,5,0): dist=sqrt(4+25)≈5.4, combined=6 → overlap
        assert!(
            results.len() >= 2,
            "Should find at least 2 overlaps, got {}",
            results.len()
        );
    }

    #[test]
    fn test_overlap_aabb() {
        let bodies = make_bodies();
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(6, 1, 1));
        let results = overlap_aabb(&aabb, &bodies);
        // Body 0 at (0,0,0) → inside
        // Body 1 at (5,0,0) → inside
        // Body 2 at (10,0,0) → outside
        // Body 3 at (0,5,0) → outside
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_raycast() {
        let bodies = make_bodies();
        let queries = vec![
            BatchRayQuery {
                origin: Vec3Fix::from_int(-10, 0, 0),
                direction: Vec3Fix::UNIT_X,
                max_distance: Fix128::from_int(100),
            },
            BatchRayQuery {
                origin: Vec3Fix::from_int(-10, 10, 0), // misses all
                direction: Vec3Fix::UNIT_X,
                max_distance: Fix128::from_int(100),
            },
        ];
        let results = batch_raycast(&queries, &bodies, Fix128::ONE);
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some(), "First ray should hit");
        assert!(results[1].is_none(), "Second ray should miss");
    }

    #[test]
    fn test_batch_sphere_cast() {
        let bodies = make_bodies();
        let origins = vec![Vec3Fix::from_int(-10, 0, 0), Vec3Fix::from_int(-10, 0, 0)];
        let directions = vec![
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Y, // up — misses bodies at y=0
        ];
        let results = batch_sphere_cast(
            &origins,
            Fix128::from_ratio(1, 2),
            &directions,
            Fix128::from_int(100),
            &bodies,
            Fix128::ONE,
        );
        assert_eq!(results.len(), 2);
        assert!(results[0].is_some(), "X-direction should hit");
    }

    #[test]
    fn test_overlap_aabb_expanded() {
        let bodies = make_bodies();
        let aabb = AABB::new(Vec3Fix::from_int(4, -1, -1), Vec3Fix::from_int(6, 1, 1));
        let results = overlap_aabb_expanded(&aabb, &bodies, Fix128::from_int(2));
        // Expanded by 2: (2,-3,-3) to (8,3,3)
        // Body 0 at (0,0,0) → outside (x=0 < 2)
        // Body 1 at (5,0,0) → inside
        // Body 2 at (10,0,0) → outside (x=10 > 8)
        // Body 3 at (0,5,0) → outside
        assert!(!results.is_empty(), "Should find body at (5,0,0)");
    }

    #[test]
    fn test_overlap_sphere_bvh() {
        use crate::bvh::{BvhPrimitive, LinearBvh};

        let bodies = make_bodies();
        let body_radius = Fix128::ONE;

        // Build BVH from body AABBs
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
        let bvh = LinearBvh::build(prims);

        let results = overlap_sphere_bvh(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::from_int(3),
            &bodies,
            body_radius,
            &bvh,
        );
        // Same as the linear test: only body 0 overlaps
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].body_index, 0);
    }

    #[test]
    fn test_overlap_aabb_bvh() {
        use crate::bvh::{BvhPrimitive, LinearBvh};

        let bodies = make_bodies();
        let body_radius = Fix128::ONE;

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
        let bvh = LinearBvh::build(prims);

        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(6, 1, 1));
        let results = overlap_aabb_bvh(&aabb, &bodies, &bvh);
        // Bodies at (0,0,0) and (5,0,0) are inside
        assert_eq!(results.len(), 2);
    }
}
