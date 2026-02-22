//! Raycasting and Shape Casting
//!
//! Deterministic ray queries against physics bodies and BVH.
//!
//! # Features
//!
//! - Ray-Sphere, Ray-AABB, Ray-Capsule intersection
//! - BVH-accelerated ray queries
//! - Closest hit and any-hit modes
//! - Shape casting (swept sphere)

use crate::collider::{Capsule, Sphere, AABB};
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A ray defined by origin and direction
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin point
    pub origin: Vec3Fix,
    /// Ray direction (normalized internally)
    pub direction: Vec3Fix,
}

/// Ray-parallel epsilon threshold
const RAY_PARALLEL_EPSILON: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000000100000000,
};

impl Ray {
    /// Create a new ray (direction is normalized internally)
    #[inline]
    pub fn new(origin: Vec3Fix, direction: Vec3Fix) -> Self {
        let len = direction.length();
        Self {
            origin,
            direction: if len.is_zero() { Vec3Fix::UNIT_X } else { direction / len },
        }
    }

    /// Get point along the ray at parameter t
    #[inline]
    pub fn at(&self, t: Fix128) -> Vec3Fix {
        self.origin + self.direction * t
    }
}

/// Result of a ray intersection test
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Parameter along the ray (distance from origin)
    pub t: Fix128,
    /// World-space hit point
    pub point: Vec3Fix,
    /// Surface normal at hit point
    pub normal: Vec3Fix,
    /// Index of the hit body (if applicable)
    pub body_index: usize,
}

/// Ray-Sphere intersection
///
/// Returns parameter t along the ray, or None if no hit.
#[inline]
pub fn ray_sphere(ray: &Ray, sphere: &Sphere, max_t: Fix128) -> Option<RayHit> {
    let oc = ray.origin - sphere.center;
    let b = oc.dot(ray.direction);
    let c = oc.dot(oc) - sphere.radius * sphere.radius;
    let discriminant = b * b - c;

    if discriminant < Fix128::ZERO {
        return None;
    }

    let sqrt_d = discriminant.sqrt();

    // Try near intersection first
    let t = -b - sqrt_d;
    if t >= Fix128::ZERO && t <= max_t {
        let point = ray.at(t);
        let normal = (point - sphere.center).normalize();
        return Some(RayHit {
            t,
            point,
            normal,
            body_index: 0,
        });
    }

    // Try far intersection
    let t = -b + sqrt_d;
    if t >= Fix128::ZERO && t <= max_t {
        let point = ray.at(t);
        let normal = (point - sphere.center).normalize();
        return Some(RayHit {
            t,
            point,
            normal,
            body_index: 0,
        });
    }

    None
}

/// Ray-AABB intersection (slab method)
///
/// Returns (t_min, t_max) interval or None if no hit.
#[inline]
pub fn ray_aabb(ray: &Ray, aabb: &AABB, max_t: Fix128) -> Option<RayHit> {
    let (t_min, t_max) = ray_aabb_interval(ray, aabb)?;

    if t_min > max_t || t_max < Fix128::ZERO {
        return None;
    }

    let t = if t_min >= Fix128::ZERO { t_min } else { t_max };
    if t > max_t {
        return None;
    }

    let point = ray.at(t);

    // Compute normal from closest face
    let normal = aabb_hit_normal(point, aabb);

    Some(RayHit {
        t,
        point,
        normal,
        body_index: 0,
    })
}

/// Ray-AABB interval computation (returns (t_min, t_max))
#[inline]
fn ray_aabb_interval(ray: &Ray, aabb: &AABB) -> Option<(Fix128, Fix128)> {
    let mut t_min = Fix128::from_int(-1000000);
    let mut t_max = Fix128::from_int(1000000);

    // X slab
    if !ray.direction.x.is_zero() {
        let inv_d = Fix128::ONE / ray.direction.x;
        let mut t0 = (aabb.min.x - ray.origin.x) * inv_d;
        let mut t1 = (aabb.max.x - ray.origin.x) * inv_d;
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
        }
        t_min = if t0 > t_min { t0 } else { t_min };
        t_max = if t1 < t_max { t1 } else { t_max };
        if t_min > t_max {
            return None;
        }
    } else if ray.origin.x < aabb.min.x || ray.origin.x > aabb.max.x {
        return None;
    }

    // Y slab
    if !ray.direction.y.is_zero() {
        let inv_d = Fix128::ONE / ray.direction.y;
        let mut t0 = (aabb.min.y - ray.origin.y) * inv_d;
        let mut t1 = (aabb.max.y - ray.origin.y) * inv_d;
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
        }
        t_min = if t0 > t_min { t0 } else { t_min };
        t_max = if t1 < t_max { t1 } else { t_max };
        if t_min > t_max {
            return None;
        }
    } else if ray.origin.y < aabb.min.y || ray.origin.y > aabb.max.y {
        return None;
    }

    // Z slab
    if !ray.direction.z.is_zero() {
        let inv_d = Fix128::ONE / ray.direction.z;
        let mut t0 = (aabb.min.z - ray.origin.z) * inv_d;
        let mut t1 = (aabb.max.z - ray.origin.z) * inv_d;
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
        }
        t_min = if t0 > t_min { t0 } else { t_min };
        t_max = if t1 < t_max { t1 } else { t_max };
        if t_min > t_max {
            return None;
        }
    } else if ray.origin.z < aabb.min.z || ray.origin.z > aabb.max.z {
        return None;
    }

    Some((t_min, t_max))
}

/// Compute AABB hit normal from the closest face
fn aabb_hit_normal(point: Vec3Fix, aabb: &AABB) -> Vec3Fix {
    let center = Vec3Fix::new(
        (aabb.min.x + aabb.max.x).half(),
        (aabb.min.y + aabb.max.y).half(),
        (aabb.min.z + aabb.max.z).half(),
    );
    let half = Vec3Fix::new(
        (aabb.max.x - aabb.min.x).half(),
        (aabb.max.y - aabb.min.y).half(),
        (aabb.max.z - aabb.min.z).half(),
    );
    let local = point - center;

    // Find axis with largest relative penetration
    let dx = (local.x.abs() - half.x).abs();
    let dy = (local.y.abs() - half.y).abs();
    let dz = (local.z.abs() - half.z).abs();

    if dx < dy && dx < dz {
        Vec3Fix::new(
            if local.x >= Fix128::ZERO {
                Fix128::ONE
            } else {
                Fix128::NEG_ONE
            },
            Fix128::ZERO,
            Fix128::ZERO,
        )
    } else if dy < dz {
        Vec3Fix::new(
            Fix128::ZERO,
            if local.y >= Fix128::ZERO {
                Fix128::ONE
            } else {
                Fix128::NEG_ONE
            },
            Fix128::ZERO,
        )
    } else {
        Vec3Fix::new(
            Fix128::ZERO,
            Fix128::ZERO,
            if local.z >= Fix128::ZERO {
                Fix128::ONE
            } else {
                Fix128::NEG_ONE
            },
        )
    }
}

/// Ray-Capsule intersection
pub fn ray_capsule(ray: &Ray, capsule: &Capsule, max_t: Fix128) -> Option<RayHit> {
    // Test against the infinite cylinder first, then cap with hemispheres
    let ab = capsule.b - capsule.a;
    let ao = ray.origin - capsule.a;

    // Project ray direction and origin offset perpendicular to capsule axis
    let ab_dot_ab = ab.dot(ab);
    if ab_dot_ab.is_zero() {
        // Degenerate capsule = sphere
        let sphere = Sphere::new(capsule.a, capsule.radius);
        return ray_sphere(ray, &sphere, max_t);
    }

    let ab_dot_d = ab.dot(ray.direction);
    let ab_dot_ao = ab.dot(ao);

    let d_perp = ray.direction - ab * (ab_dot_d / ab_dot_ab);
    let ao_perp = ao - ab * (ab_dot_ao / ab_dot_ab);

    let a_coeff = d_perp.dot(d_perp);
    let b_coeff = d_perp.dot(ao_perp).double();
    let c_coeff = ao_perp.dot(ao_perp) - capsule.radius * capsule.radius;

    let discriminant = b_coeff * b_coeff - Fix128::from_int(4) * a_coeff * c_coeff;
    if discriminant < Fix128::ZERO {
        // Try hemisphere caps
        let sphere_a = Sphere::new(capsule.a, capsule.radius);
        let sphere_b = Sphere::new(capsule.b, capsule.radius);
        let hit_a = ray_sphere(ray, &sphere_a, max_t);
        let hit_b = ray_sphere(ray, &sphere_b, max_t);
        return closer_hit(hit_a, hit_b);
    }

    let sqrt_d = discriminant.sqrt();
    let two_a = a_coeff.double();
    if two_a.is_zero() {
        return None;
    }

    let t = (-b_coeff - sqrt_d) / two_a;
    if t >= Fix128::ZERO && t <= max_t {
        let point = ray.at(t);
        // Check if within cylinder bounds
        let ap = point - capsule.a;
        let proj = ap.dot(ab) / ab_dot_ab;
        if proj >= Fix128::ZERO && proj <= Fix128::ONE {
            let center_on_axis = capsule.a + ab * proj;
            let normal = (point - center_on_axis).normalize();
            return Some(RayHit {
                t,
                point,
                normal,
                body_index: 0,
            });
        }
    }

    // Test hemisphere caps
    let sphere_a = Sphere::new(capsule.a, capsule.radius);
    let sphere_b = Sphere::new(capsule.b, capsule.radius);
    let hit_a = ray_sphere(ray, &sphere_a, max_t);
    let hit_b = ray_sphere(ray, &sphere_b, max_t);
    closer_hit(hit_a, hit_b)
}

/// Return the closer of two optional hits
#[inline]
fn closer_hit(a: Option<RayHit>, b: Option<RayHit>) -> Option<RayHit> {
    match (a, b) {
        (Some(ha), Some(hb)) => {
            if ha.t < hb.t {
                Some(ha)
            } else {
                Some(hb)
            }
        }
        (Some(h), None) | (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

/// Ray-Plane intersection
///
/// Plane defined by normal and offset: dot(normal, point) = offset
pub fn ray_plane(
    ray: &Ray,
    plane_normal: Vec3Fix,
    plane_offset: Fix128,
    max_t: Fix128,
) -> Option<RayHit> {
    let denom = ray.direction.dot(plane_normal);

    if denom.abs() < RAY_PARALLEL_EPSILON {
        return None; // Ray parallel to plane
    }

    let t = (plane_offset - ray.origin.dot(plane_normal)) / denom;

    if t >= Fix128::ZERO && t <= max_t {
        let point = ray.at(t);
        let normal = if denom < Fix128::ZERO {
            plane_normal
        } else {
            -plane_normal
        };
        Some(RayHit {
            t,
            point,
            normal,
            body_index: 0,
        })
    } else {
        None
    }
}

/// Cast a ray against multiple AABBs and return the closest hit
pub fn raycast_aabbs(ray: &Ray, aabbs: &[(AABB, usize)], max_t: Fix128) -> Option<RayHit> {
    let mut closest: Option<RayHit> = None;
    let mut best_t = max_t;

    for &(ref aabb, body_idx) in aabbs {
        if let Some(mut hit) = ray_aabb(ray, aabb, best_t) {
            hit.body_index = body_idx;
            best_t = hit.t;
            closest = Some(hit);
        }
    }

    closest
}

/// Cast a ray against multiple Spheres and return the closest hit
pub fn raycast_spheres(ray: &Ray, spheres: &[(Sphere, usize)], max_t: Fix128) -> Option<RayHit> {
    let mut closest: Option<RayHit> = None;
    let mut best_t = max_t;

    for &(ref sphere, body_idx) in spheres {
        if let Some(mut hit) = ray_sphere(ray, sphere, best_t) {
            hit.body_index = body_idx;
            best_t = hit.t;
            closest = Some(hit);
        }
    }

    closest
}

/// Cast a ray against multiple Spheres and return ALL hits (sorted by t, closest first)
pub fn raycast_all_spheres(ray: &Ray, spheres: &[(Sphere, usize)], max_t: Fix128) -> Vec<RayHit> {
    let mut hits = Vec::new();

    for &(ref sphere, body_idx) in spheres {
        if let Some(mut hit) = ray_sphere(ray, sphere, max_t) {
            hit.body_index = body_idx;
            hits.push(hit);
        }
    }

    // Deterministic stable sort by t
    hits.sort_by(|a, b| a.t.cmp(&b.t));
    hits
}

/// Cast a ray against multiple AABBs and return ALL hits (sorted by t, closest first)
pub fn raycast_all_aabbs(ray: &Ray, aabbs: &[(AABB, usize)], max_t: Fix128) -> Vec<RayHit> {
    let mut hits = Vec::new();

    for &(ref aabb, body_idx) in aabbs {
        if let Some(mut hit) = ray_aabb(ray, aabb, max_t) {
            hit.body_index = body_idx;
            hits.push(hit);
        }
    }

    hits.sort_by(|a, b| a.t.cmp(&b.t));
    hits
}

/// Test if ANY sphere is hit by the ray (early-out on first hit)
#[inline]
pub fn raycast_any_spheres(ray: &Ray, spheres: &[(Sphere, usize)], max_t: Fix128) -> bool {
    for (sphere, _) in spheres {
        if ray_sphere(ray, sphere, max_t).is_some() {
            return true;
        }
    }
    false
}

/// Test if ANY AABB is hit by the ray (early-out on first hit)
#[inline]
pub fn raycast_any_aabbs(ray: &Ray, aabbs: &[(AABB, usize)], max_t: Fix128) -> bool {
    for (aabb, _) in aabbs {
        if ray_aabb(ray, aabb, max_t).is_some() {
            return true;
        }
    }
    false
}

/// Swept sphere (shape cast): move a sphere along a ray direction
pub fn sweep_sphere(
    sphere: &Sphere,
    direction: Vec3Fix,
    target_sphere: &Sphere,
    max_t: Fix128,
) -> Option<RayHit> {
    // Equivalent to ray vs expanded sphere
    let expanded = Sphere::new(target_sphere.center, target_sphere.radius + sphere.radius);
    let ray = Ray::new(sphere.center, direction);
    ray_sphere(&ray, &expanded, max_t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_sphere_hit() {
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);

        let hit = ray_sphere(&ray, &sphere, Fix128::from_int(100)).unwrap();
        // Should hit at x = -1 (near side of unit sphere at origin)
        let expected_t = Fix128::from_int(4); // -5 + 4 = -1
        let error = (hit.t - expected_t).abs();
        assert!(error < Fix128::ONE, "t should be ~4, got {:?}", hit.t);
    }

    #[test]
    fn test_ray_sphere_miss() {
        let ray = Ray::new(Vec3Fix::from_int(-5, 5, 0), Vec3Fix::UNIT_X);
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);

        assert!(ray_sphere(&ray, &sphere, Fix128::from_int(100)).is_none());
    }

    #[test]
    fn test_ray_aabb_hit() {
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));

        let hit = ray_aabb(&ray, &aabb, Fix128::from_int(100)).unwrap();
        let expected_t = Fix128::from_int(4); // -5 + 4 = -1
        let error = (hit.t - expected_t).abs();
        assert!(error < Fix128::ONE, "t should be ~4");
    }

    #[test]
    fn test_ray_aabb_miss() {
        let ray = Ray::new(Vec3Fix::from_int(-5, 5, 0), Vec3Fix::UNIT_X);
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));

        assert!(ray_aabb(&ray, &aabb, Fix128::from_int(100)).is_none());
    }

    #[test]
    fn test_ray_plane() {
        let ray = Ray::new(
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO), // downward
        );

        let hit = ray_plane(&ray, Vec3Fix::UNIT_Y, Fix128::ZERO, Fix128::from_int(100)).unwrap();
        let expected_t = Fix128::from_int(10);
        let error = (hit.t - expected_t).abs();
        assert!(error < Fix128::ONE, "Should hit ground plane at t=10");
    }

    #[test]
    fn test_sweep_sphere() {
        let moving = Sphere::new(Vec3Fix::from_int(-5, 0, 0), Fix128::ONE);
        let target = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);

        let hit = sweep_sphere(&moving, Vec3Fix::UNIT_X, &target, Fix128::from_int(100));
        assert!(hit.is_some(), "Swept sphere should hit target");
        let t = hit.unwrap().t;
        // Should hit at distance 5 - 2 = 3 (center distance minus combined radii)
        let error = (t - Fix128::from_int(3)).abs();
        assert!(error < Fix128::ONE, "Should hit at approximately t=3");
    }

    #[test]
    fn test_raycast_all_spheres() {
        let spheres = vec![
            (Sphere::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE), 0),
            (Sphere::new(Vec3Fix::from_int(6, 0, 0), Fix128::ONE), 1),
            (Sphere::new(Vec3Fix::from_int(9, 0, 0), Fix128::ONE), 2),
        ];
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        let hits = raycast_all_spheres(&ray, &spheres, Fix128::from_int(100));
        assert_eq!(hits.len(), 3, "Should hit all 3 spheres");
        // Sorted by t — closest first
        assert_eq!(hits[0].body_index, 0);
        assert_eq!(hits[1].body_index, 1);
        assert_eq!(hits[2].body_index, 2);
    }

    #[test]
    fn test_raycast_any_spheres() {
        let spheres = vec![(Sphere::new(Vec3Fix::from_int(3, 0, 0), Fix128::ONE), 0)];
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        assert!(raycast_any_spheres(&ray, &spheres, Fix128::from_int(100)));

        let miss_ray = Ray::new(Vec3Fix::from_int(-5, 10, 0), Vec3Fix::UNIT_X);
        assert!(!raycast_any_spheres(
            &miss_ray,
            &spheres,
            Fix128::from_int(100)
        ));
    }

    #[test]
    fn test_raycast_all_aabbs() {
        let aabbs = vec![
            (
                AABB::new(Vec3Fix::from_int(2, -1, -1), Vec3Fix::from_int(4, 1, 1)),
                0,
            ),
            (
                AABB::new(Vec3Fix::from_int(6, -1, -1), Vec3Fix::from_int(8, 1, 1)),
                1,
            ),
        ];
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        let hits = raycast_all_aabbs(&ray, &aabbs, Fix128::from_int(100));
        assert_eq!(hits.len(), 2, "Should hit both AABBs");
        assert_eq!(hits[0].body_index, 0);
        assert_eq!(hits[1].body_index, 1);
    }

    #[test]
    fn test_ray_capsule() {
        let ray = Ray::new(Vec3Fix::from_int(-5, 0, 0), Vec3Fix::UNIT_X);
        let capsule = Capsule::new(
            Vec3Fix::from_int(0, -2, 0),
            Vec3Fix::from_int(0, 2, 0),
            Fix128::ONE,
        );

        let hit = ray_capsule(&ray, &capsule, Fix128::from_int(100));
        assert!(hit.is_some(), "Ray should hit capsule");
    }
}
