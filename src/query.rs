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
use crate::raycast::{ray_sphere, Ray};
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ============================================================================
// Query Results
// ============================================================================

/// Result of a shape cast query
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
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

        if let Some(hit) = ray_sphere(&ray, &expanded, best_t) {
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
/// Approximates capsule-vs-sphere as two sphere casts
/// (top and bottom hemispheres) and returns the closest hit.
pub fn capsule_cast(
    capsule_a: Vec3Fix,
    capsule_b: Vec3Fix,
    capsule_radius: Fix128,
    direction: Vec3Fix,
    max_distance: Fix128,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Option<ShapeCastHit> {
    // Cast from both endpoints of the capsule and take closest
    let hit_a = sphere_cast(
        capsule_a,
        capsule_radius,
        direction,
        max_distance,
        bodies,
        body_radius,
    );
    let hit_b = sphere_cast(
        capsule_b,
        capsule_radius,
        direction,
        max_distance,
        bodies,
        body_radius,
    );

    // Also cast from the midpoint for better coverage
    let mid = Vec3Fix::new(
        (capsule_a.x + capsule_b.x).half(),
        (capsule_a.y + capsule_b.y).half(),
        (capsule_a.z + capsule_b.z).half(),
    );
    let hit_mid = sphere_cast(
        mid,
        capsule_radius,
        direction,
        max_distance,
        bodies,
        body_radius,
    );

    // Return closest of the three
    let mut best: Option<ShapeCastHit> = None;
    for h in [hit_a, hit_b, hit_mid].into_iter().flatten() {
        if best.is_none() || h.t < best.unwrap().t {
            best = Some(h);
        }
    }

    best
}

// ============================================================================
// Overlap Functions
// ============================================================================

/// Find all bodies whose bounding sphere overlaps with the given sphere.
///
/// `body_radius` is the assumed radius for each body.
#[inline]
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
/// Each body is treated as a sphere of `body_radius`. The AABB is expanded
/// by `body_radius` to detect sphere-vs-AABB overlap.
pub fn overlap_aabb_expanded(
    aabb: &AABB,
    bodies: &[RigidBody],
    body_radius: Fix128,
) -> Vec<OverlapResult> {
    let expanded = AABB::new(
        Vec3Fix::new(
            aabb.min.x - body_radius,
            aabb.min.y - body_radius,
            aabb.min.z - body_radius,
        ),
        Vec3Fix::new(
            aabb.max.x + body_radius,
            aabb.max.y + body_radius,
            aabb.max.z + body_radius,
        ),
    );

    overlap_aabb(&expanded, bodies)
}

// ============================================================================
// Batch Queries
// ============================================================================

/// A single raycast query for batch execution
#[derive(Clone, Copy, Debug)]
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
// Tests
// ============================================================================

#[cfg(test)]
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
        let h = hit.unwrap();
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
        assert!(results.len() >= 1, "Should find body at (5,0,0)");
    }
}
