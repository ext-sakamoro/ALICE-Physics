//! Triangle Mesh Collision
//!
//! Static triangle mesh collider with BVH acceleration.
//! Supports ray queries and closest-point queries against arbitrary meshes.

use crate::math::{Fix128, Vec3Fix};
use crate::collider::{AABB, Contact};
use crate::bvh::{LinearBvh, BvhPrimitive};
use crate::raycast::{Ray, RayHit};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A single triangle
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub v0: Vec3Fix,
    pub v1: Vec3Fix,
    pub v2: Vec3Fix,
}

impl Triangle {
    #[inline]
    pub fn new(v0: Vec3Fix, v1: Vec3Fix, v2: Vec3Fix) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute face normal (not normalized)
    #[inline]
    pub fn normal(&self) -> Vec3Fix {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2)
    }

    /// Compute normalized face normal
    #[inline]
    pub fn unit_normal(&self) -> Vec3Fix {
        self.normal().normalize()
    }

    /// Compute AABB enclosing this triangle
    pub fn aabb(&self) -> AABB {
        let min = Vec3Fix::new(
            min3(self.v0.x, self.v1.x, self.v2.x),
            min3(self.v0.y, self.v1.y, self.v2.y),
            min3(self.v0.z, self.v1.z, self.v2.z),
        );
        let max = Vec3Fix::new(
            max3(self.v0.x, self.v1.x, self.v2.x),
            max3(self.v0.y, self.v1.y, self.v2.y),
            max3(self.v0.z, self.v1.z, self.v2.z),
        );
        AABB::new(min, max)
    }

    /// Closest point on triangle to a given point
    pub fn closest_point(&self, p: Vec3Fix) -> Vec3Fix {
        let ab = self.v1 - self.v0;
        let ac = self.v2 - self.v0;
        let ap = p - self.v0;

        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= Fix128::ZERO && d2 <= Fix128::ZERO {
            return self.v0; // Vertex region A
        }

        let bp = p - self.v1;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= Fix128::ZERO && d4 <= d3 {
            return self.v1; // Vertex region B
        }

        let vc = d1 * d4 - d3 * d2;
        if vc <= Fix128::ZERO && d1 >= Fix128::ZERO && d3 <= Fix128::ZERO {
            let v = d1 / (d1 - d3);
            return self.v0 + ab * v; // Edge AB
        }

        let cp = p - self.v2;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= Fix128::ZERO && d5 <= d6 {
            return self.v2; // Vertex region C
        }

        let vb = d5 * d2 - d1 * d6;
        if vb <= Fix128::ZERO && d2 >= Fix128::ZERO && d6 <= Fix128::ZERO {
            let w = d2 / (d2 - d6);
            return self.v0 + ac * w; // Edge AC
        }

        let va = d3 * d6 - d5 * d4;
        if va <= Fix128::ZERO {
            let d4_d3 = d4 - d3;
            let d5_d6 = d5 - d6;
            if d4_d3 >= Fix128::ZERO && d5_d6 >= Fix128::ZERO {
                let w = d4_d3 / (d4_d3 + d5_d6);
                return self.v1 + (self.v2 - self.v1) * w; // Edge BC
            }
        }

        // Inside triangle
        let denom = va + vb + vc;
        if denom.is_zero() {
            return self.v0;
        }
        let v = vb / denom;
        let w = vc / denom;
        self.v0 + ab * v + ac * w
    }
}

/// Triangle mesh with BVH acceleration
pub struct TriMesh {
    /// Triangles
    pub triangles: Vec<Triangle>,
    /// BVH for acceleration
    bvh: LinearBvh,
    /// Overall AABB
    pub bounds: AABB,
}

impl TriMesh {
    /// Build from vertices and triangle indices
    pub fn from_indexed(vertices: &[Vec3Fix], indices: &[u32]) -> Self {
        let mut triangles = Vec::with_capacity(indices.len() / 3);
        let mut bvh_prims = Vec::with_capacity(indices.len() / 3);

        for i in (0..indices.len()).step_by(3) {
            if i + 2 >= indices.len() {
                break;
            }
            let v0 = vertices[indices[i] as usize];
            let v1 = vertices[indices[i + 1] as usize];
            let v2 = vertices[indices[i + 2] as usize];

            let tri = Triangle::new(v0, v1, v2);
            let aabb = tri.aabb();
            let tri_idx = triangles.len() as u32;

            triangles.push(tri);
            bvh_prims.push(BvhPrimitive {
                aabb,
                index: tri_idx,
                morton: 0,
            });
        }

        let bvh = LinearBvh::build(bvh_prims);
        let bounds = bvh.bounds;

        Self { triangles, bvh, bounds }
    }

    /// Build from raw triangles
    pub fn from_triangles(triangles: Vec<Triangle>) -> Self {
        let bvh_prims: Vec<BvhPrimitive> = triangles
            .iter()
            .enumerate()
            .map(|(i, tri)| BvhPrimitive {
                aabb: tri.aabb(),
                index: i as u32,
                morton: 0,
            })
            .collect();

        let bvh = LinearBvh::build(bvh_prims);
        let bounds = bvh.bounds;

        Self { triangles, bvh, bounds }
    }

    /// Ray query: find closest triangle intersection
    pub fn raycast(&self, ray: &Ray, max_t: Fix128) -> Option<RayHit> {
        let ray_aabb = ray_to_aabb(ray, max_t);
        let candidates = self.bvh.query(&ray_aabb);

        let mut best: Option<RayHit> = None;
        let mut best_t = max_t;

        for tri_idx in candidates {
            let tri = &self.triangles[tri_idx as usize];
            if let Some(mut hit) = ray_triangle(ray, tri, best_t) {
                hit.body_index = tri_idx as usize;
                best_t = hit.t;
                best = Some(hit);
            }
        }

        best
    }

    /// Closest point on mesh to a given point
    pub fn closest_point(&self, point: Vec3Fix) -> (Vec3Fix, usize) {
        // Query BVH with a large AABB centered on point
        let half = Fix128::from_int(1000);
        let query = AABB::new(
            point - Vec3Fix::new(half, half, half),
            point + Vec3Fix::new(half, half, half),
        );
        let candidates = self.bvh.query(&query);

        let mut best_point = self.triangles[0].v0;
        let mut best_dist_sq = Fix128::from_int(i64::MAX / 2);
        let mut best_idx = 0;

        for tri_idx in candidates {
            let tri = &self.triangles[tri_idx as usize];
            let cp = tri.closest_point(point);
            let dist_sq = (cp - point).length_squared();
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_point = cp;
                best_idx = tri_idx as usize;
            }
        }

        (best_point, best_idx)
    }

    /// Sphere vs TriMesh collision: find deepest penetrating contact
    pub fn collide_sphere(&self, center: Vec3Fix, radius: Fix128) -> Option<Contact> {
        let query = AABB::new(
            center - Vec3Fix::new(radius, radius, radius),
            center + Vec3Fix::new(radius, radius, radius),
        );
        let candidates = self.bvh.query(&query);

        let mut deepest: Option<Contact> = None;
        let mut max_depth = Fix128::ZERO;

        for tri_idx in candidates {
            let tri = &self.triangles[tri_idx as usize];
            let cp = tri.closest_point(center);
            let delta = center - cp;
            let dist_sq = delta.length_squared();
            let r_sq = radius * radius;

            if dist_sq < r_sq {
                let dist = dist_sq.sqrt();
                let depth = radius - dist;

                if depth > max_depth {
                    max_depth = depth;
                    let normal = if dist.is_zero() {
                        tri.unit_normal()
                    } else {
                        delta / dist
                    };
                    deepest = Some(Contact {
                        depth,
                        normal,
                        point_a: center - normal * radius,
                        point_b: cp,
                    });
                }
            }
        }

        deepest
    }

    /// Number of triangles
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// Ray-Triangle intersection (Moller-Trumbore algorithm)
pub fn ray_triangle(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Option<RayHit> {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = ray.direction.cross(e2);
    let det = e1.dot(h);

    // Parallel check
    let epsilon = Fix128::from_raw(0, 0x0000010000000000);
    if det.abs() < epsilon {
        return None;
    }

    let inv_det = Fix128::ONE / det;
    let s = ray.origin - tri.v0;
    let u = s.dot(h) * inv_det;

    if u < Fix128::ZERO || u > Fix128::ONE {
        return None;
    }

    let q = s.cross(e1);
    let v = ray.direction.dot(q) * inv_det;

    if v < Fix128::ZERO || u + v > Fix128::ONE {
        return None;
    }

    let t = e2.dot(q) * inv_det;

    if t >= Fix128::ZERO && t <= max_t {
        let point = ray.at(t);
        let normal = tri.unit_normal();
        // Ensure normal faces the ray
        let normal = if normal.dot(ray.direction) > Fix128::ZERO { -normal } else { normal };
        Some(RayHit { t, point, normal, body_index: 0 })
    } else {
        None
    }
}

/// Create an AABB enclosing a ray segment
fn ray_to_aabb(ray: &Ray, max_t: Fix128) -> AABB {
    let end = ray.at(max_t);
    let one = Fix128::ONE;
    AABB::new(
        Vec3Fix::new(
            if ray.origin.x < end.x { ray.origin.x - one } else { end.x - one },
            if ray.origin.y < end.y { ray.origin.y - one } else { end.y - one },
            if ray.origin.z < end.z { ray.origin.z - one } else { end.z - one },
        ),
        Vec3Fix::new(
            if ray.origin.x > end.x { ray.origin.x + one } else { end.x + one },
            if ray.origin.y > end.y { ray.origin.y + one } else { end.y + one },
            if ray.origin.z > end.z { ray.origin.z + one } else { end.z + one },
        ),
    )
}

#[inline]
fn min3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let ab = if a < b { a } else { b };
    if ab < c { ab } else { c }
}

#[inline]
fn max3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let ab = if a > b { a } else { b };
    if ab > c { ab } else { c }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ground_mesh() -> TriMesh {
        // Simple ground plane as two triangles
        let vertices = vec![
            Vec3Fix::from_int(-10, 0, -10),
            Vec3Fix::from_int(10, 0, -10),
            Vec3Fix::from_int(10, 0, 10),
            Vec3Fix::from_int(-10, 0, 10),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        TriMesh::from_indexed(&vertices, &indices)
    }

    #[test]
    fn test_build_trimesh() {
        let mesh = make_ground_mesh();
        assert_eq!(mesh.triangle_count(), 2);
    }

    #[test]
    fn test_ray_triangle_hit() {
        let tri = Triangle::new(
            Vec3Fix::from_int(-1, 0, -1),
            Vec3Fix::from_int(1, 0, -1),
            Vec3Fix::from_int(0, 0, 1),
        );
        let ray = Ray::new(
            Vec3Fix::from_int(0, 5, 0),
            Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO),
        );

        let hit = ray_triangle(&ray, &tri, Fix128::from_int(100));
        assert!(hit.is_some(), "Ray should hit triangle");
        let t = hit.unwrap().t;
        let error = (t - Fix128::from_int(5)).abs();
        assert!(error < Fix128::ONE, "t should be ~5");
    }

    #[test]
    fn test_ray_triangle_miss() {
        let tri = Triangle::new(
            Vec3Fix::from_int(-1, 0, -1),
            Vec3Fix::from_int(1, 0, -1),
            Vec3Fix::from_int(0, 0, 1),
        );
        let ray = Ray::new(
            Vec3Fix::from_int(5, 5, 0),
            Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO),
        );

        assert!(ray_triangle(&ray, &tri, Fix128::from_int(100)).is_none());
    }

    #[test]
    fn test_mesh_raycast() {
        let mesh = make_ground_mesh();
        let ray = Ray::new(
            Vec3Fix::from_int(0, 10, 0),
            Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO),
        );

        let hit = mesh.raycast(&ray, Fix128::from_int(100));
        assert!(hit.is_some(), "Should hit ground mesh");
    }

    #[test]
    fn test_sphere_trimesh_collision() {
        let mesh = make_ground_mesh();
        // Sphere overlapping with ground plane
        let contact = mesh.collide_sphere(
            Vec3Fix::from_int(0, 0, 0), // Center at ground level
            Fix128::ONE,                  // Radius 1 => penetrating
        );
        assert!(contact.is_some(), "Sphere should collide with ground mesh");
    }

    #[test]
    fn test_closest_point_on_triangle() {
        let tri = Triangle::new(
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(10, 0, 0),
            Vec3Fix::from_int(0, 0, 10),
        );

        // Point above triangle center
        let cp = tri.closest_point(Vec3Fix::from_int(3, 5, 3));
        assert_eq!(cp.y.hi, 0, "Closest point should be on triangle plane (y=0)");
    }
}
