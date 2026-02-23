//! Infinite Plane Collider
//!
//! Infinite plane for collision detection. Planes do not use GJK (they are
//! not bounded convex shapes); instead specialized intersection tests are
//! provided for sphere and AABB queries.
//!
//! # Features
//!
//! - **Half-space**: Defined by normal and signed offset from origin
//! - **Point queries**: Signed distance, projection onto plane
//! - **Intersection tests**: Plane-sphere, plane-AABB
//!
//! # Representation
//!
//! The plane is represented in Hessian normal form: `dot(normal, p) = offset`.
//! Points with `dot(normal, p) > offset` are on the positive (front) side.
//!
//! Author: Moroya Sakamoto

use crate::collider::{CollisionResult, AABB};
use crate::math::{Fix128, Vec3Fix};

/// Infinite plane collider
///
/// Defined in Hessian normal form: `dot(normal, p) = offset`.
/// The normal should be unit length.
#[derive(Clone, Copy, Debug)]
pub struct PlaneCollider {
    /// Plane normal (unit length)
    pub normal: Vec3Fix,
    /// Signed distance from origin along normal
    pub offset: Fix128,
}

impl PlaneCollider {
    /// Create a new plane from normal and offset
    ///
    /// The normal is normalized internally. If the input normal is zero-length,
    /// the Y-up direction is used as a fallback.
    #[must_use]
    pub fn new(normal: Vec3Fix, offset: Fix128) -> Self {
        let n = if normal.length_squared().is_zero() {
            Vec3Fix::UNIT_Y
        } else {
            normal.normalize()
        };
        Self { normal: n, offset }
    }

    /// Create a plane from a point on the plane and a normal direction
    ///
    /// The normal is normalized internally. The offset is computed as
    /// `dot(normal, point)`.
    #[must_use]
    pub fn from_point_normal(point: Vec3Fix, normal: Vec3Fix) -> Self {
        let n = if normal.length_squared().is_zero() {
            Vec3Fix::UNIT_Y
        } else {
            normal.normalize()
        };
        let offset = n.dot(point);
        Self { normal: n, offset }
    }

    /// Signed distance from a point to the plane
    ///
    /// Positive: point is on the front (normal) side.
    /// Zero: point is on the plane.
    /// Negative: point is on the back side.
    #[inline]
    #[must_use]
    pub fn distance_to_point(&self, point: Vec3Fix) -> Fix128 {
        self.normal.dot(point) - self.offset
    }

    /// Project a point onto the plane (closest point on plane)
    #[inline]
    #[must_use]
    pub fn project_point(&self, point: Vec3Fix) -> Vec3Fix {
        let dist = self.distance_to_point(point);
        point - self.normal * dist
    }

    /// Test intersection with a sphere
    ///
    /// Returns a `CollisionResult` with the contact point, normal, and depth.
    /// If the sphere does not intersect, returns `CollisionResult::NONE`.
    #[must_use]
    pub fn intersect_sphere(
        &self,
        sphere_center: Vec3Fix,
        sphere_radius: Fix128,
    ) -> CollisionResult {
        let dist = self.distance_to_point(sphere_center);

        // Sphere intersects if |dist| < radius (using signed distance for
        // the check to handle spheres on either side of the plane)
        let abs_dist = dist.abs();
        if abs_dist >= sphere_radius {
            return CollisionResult::NONE;
        }

        let depth = sphere_radius - abs_dist;

        // Contact normal: from plane toward sphere center
        let contact_normal = if dist >= Fix128::ZERO {
            self.normal
        } else {
            -self.normal
        };

        let point_on_plane = self.project_point(sphere_center);
        let point_on_sphere = sphere_center - contact_normal * sphere_radius;

        CollisionResult::new(depth, contact_normal, point_on_sphere, point_on_plane)
    }

    /// Test intersection with an AABB
    ///
    /// Returns a `CollisionResult` if any part of the AABB is below the plane.
    /// The penetration depth is the maximum penetration of any AABB corner.
    #[must_use]
    pub fn intersect_aabb(&self, aabb: &AABB) -> CollisionResult {
        // Find the AABB vertex most in the negative normal direction (n-vertex)
        // and the vertex most in the positive normal direction (p-vertex)
        let p_vertex = Vec3Fix::new(
            if self.normal.x >= Fix128::ZERO {
                aabb.max.x
            } else {
                aabb.min.x
            },
            if self.normal.y >= Fix128::ZERO {
                aabb.max.y
            } else {
                aabb.min.y
            },
            if self.normal.z >= Fix128::ZERO {
                aabb.max.z
            } else {
                aabb.min.z
            },
        );

        let n_vertex = Vec3Fix::new(
            if self.normal.x >= Fix128::ZERO {
                aabb.min.x
            } else {
                aabb.max.x
            },
            if self.normal.y >= Fix128::ZERO {
                aabb.min.y
            } else {
                aabb.max.y
            },
            if self.normal.z >= Fix128::ZERO {
                aabb.min.z
            } else {
                aabb.max.z
            },
        );

        let p_dist = self.distance_to_point(p_vertex);
        let n_dist = self.distance_to_point(n_vertex);

        // If the p-vertex is on the back side, the entire AABB is behind the plane
        // If the n-vertex is on the front side, the entire AABB is in front (no collision)
        if n_dist >= Fix128::ZERO {
            // Entirely in front of plane — no collision
            return CollisionResult::NONE;
        }

        if p_dist < Fix128::ZERO {
            // Entire AABB is behind the plane
            let depth = -n_dist;
            let center = Vec3Fix::new(
                (aabb.min.x + aabb.max.x).half(),
                (aabb.min.y + aabb.max.y).half(),
                (aabb.min.z + aabb.max.z).half(),
            );
            let point_on_plane = self.project_point(center);
            return CollisionResult::new(depth, self.normal, n_vertex, point_on_plane);
        }

        // Partial intersection
        let depth = -n_dist;
        let point_on_plane = self.project_point(n_vertex);
        CollisionResult::new(depth, self.normal, n_vertex, point_on_plane)
    }

    /// Check which side of the plane a point is on
    ///
    /// Returns `true` if the point is on the front (positive) side.
    #[inline]
    #[must_use]
    pub fn is_front(&self, point: Vec3Fix) -> bool {
        self.distance_to_point(point) >= Fix128::ZERO
    }

    /// Flip the plane (reverse normal, negate offset)
    #[inline]
    #[must_use]
    pub fn flip(&self) -> Self {
        Self {
            normal: -self.normal,
            offset: -self.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_new() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        assert_eq!(plane.normal.y.hi, 1);
        assert_eq!(plane.offset.hi, 0);
    }

    #[test]
    fn test_plane_from_point_normal() {
        let plane = PlaneCollider::from_point_normal(Vec3Fix::from_int(0, 5, 0), Vec3Fix::UNIT_Y);
        assert_eq!(plane.normal.y.hi, 1);
        assert_eq!(plane.offset.hi, 5);
    }

    #[test]
    fn test_plane_distance_to_point() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        // Point above plane
        let d1 = plane.distance_to_point(Vec3Fix::from_int(0, 3, 0));
        assert_eq!(d1.hi, 3);
        // Point below plane
        let d2 = plane.distance_to_point(Vec3Fix::from_int(0, -2, 0));
        assert_eq!(d2.hi, -2);
        // Point on plane
        let d3 = plane.distance_to_point(Vec3Fix::from_int(5, 0, 7));
        assert_eq!(d3.hi, 0);
    }

    #[test]
    fn test_plane_project_point() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        let projected = plane.project_point(Vec3Fix::from_int(3, 7, 5));
        assert_eq!(projected.x.hi, 3);
        assert_eq!(projected.y.hi, 0);
        assert_eq!(projected.z.hi, 5);
    }

    #[test]
    fn test_plane_sphere_intersection() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        // Sphere at y=0.5 with r=1 penetrates the plane
        let result = plane.intersect_sphere(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(1, 2), Fix128::ZERO),
            Fix128::ONE,
        );
        assert!(result.colliding, "Sphere should intersect plane");
        // depth = 1 - 0.5 = 0.5
        assert_eq!(result.depth.hi, 0); // 0.5 has hi=0
    }

    #[test]
    fn test_plane_sphere_no_intersection() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        // Sphere at y=5 with r=1 does NOT intersect
        let result = plane.intersect_sphere(Vec3Fix::from_int(0, 5, 0), Fix128::ONE);
        assert!(!result.colliding, "Sphere should not intersect plane");
    }

    #[test]
    fn test_plane_aabb_intersection() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        // AABB from (-1,-1,-1) to (1,1,1) straddles the plane
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
        let result = plane.intersect_aabb(&aabb);
        assert!(result.colliding, "AABB should intersect plane");
        assert_eq!(result.depth.hi, 1); // n-vertex at y=-1, dist=-1, depth=1
    }

    #[test]
    fn test_plane_aabb_no_intersection() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        // AABB entirely above plane
        let aabb = AABB::new(Vec3Fix::from_int(-1, 2, -1), Vec3Fix::from_int(1, 4, 1));
        let result = plane.intersect_aabb(&aabb);
        assert!(!result.colliding, "AABB above plane should not intersect");
    }

    #[test]
    fn test_plane_is_front() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        assert!(plane.is_front(Vec3Fix::from_int(0, 1, 0)));
        assert!(plane.is_front(Vec3Fix::ZERO)); // on plane counts as front
        assert!(!plane.is_front(Vec3Fix::from_int(0, -1, 0)));
    }

    #[test]
    fn test_plane_flip() {
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::from_int(3));
        let flipped = plane.flip();
        assert_eq!(flipped.normal.y.hi, -1);
        assert_eq!(flipped.offset.hi, -3);
    }

    #[test]
    fn test_plane_sphere_behind() {
        // Sphere partially behind the plane (center at y=-0.5, radius=1)
        let plane = PlaneCollider::new(Vec3Fix::UNIT_Y, Fix128::ZERO);
        let result = plane.intersect_sphere(
            Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-1, 2), Fix128::ZERO),
            Fix128::ONE,
        );
        assert!(result.colliding, "Sphere crossing plane should collide");
    }
}
