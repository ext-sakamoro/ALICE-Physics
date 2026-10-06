//! Triangle Mesh Collision
//!
//! Static triangle mesh collider with BVH acceleration.
//! Supports ray queries and closest-point queries against arbitrary meshes.

use crate::bvh::{BvhPrimitive, LinearBvh};
use crate::collider::{Contact, AABB};
use crate::math::{Fix128, Vec3Fix};
use crate::raycast::{Ray, RayHit};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A single triangle
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Triangle {
    /// First vertex
    pub v0: Vec3Fix,
    /// Second vertex
    pub v1: Vec3Fix,
    /// Third vertex
    pub v2: Vec3Fix,
}

impl Triangle {
    /// Create a new triangle from three vertices
    #[inline]
    #[must_use]
    pub const fn new(v0: Vec3Fix, v1: Vec3Fix, v2: Vec3Fix) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute face normal (not normalized)
    #[inline]
    #[must_use]
    pub fn normal(&self) -> Vec3Fix {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(e2)
    }

    /// Compute normalized face normal
    #[inline]
    #[must_use]
    pub fn unit_normal(&self) -> Vec3Fix {
        self.normal().normalize()
    }

    /// Compute AABB enclosing this triangle
    #[must_use]
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
    #[must_use]
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
    pub(crate) bvh: LinearBvh,
    /// Overall AABB
    pub bounds: AABB,
}

impl TriMesh {
    /// Build from vertices and triangle indices
    #[must_use]
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

        Self {
            triangles,
            bvh,
            bounds,
        }
    }

    /// Build from raw triangles
    #[must_use]
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

        Self {
            triangles,
            bvh,
            bounds,
        }
    }

    /// Ray query: find closest triangle intersection
    #[must_use]
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
    ///
    /// An empty mesh (no triangles) has no closest point to report: returns
    /// the query point itself with index 0.
    #[must_use]
    pub fn closest_point(&self, point: Vec3Fix) -> (Vec3Fix, usize) {
        if self.triangles.is_empty() {
            return (point, 0);
        }

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

    /// Sphere vs `TriMesh` collision: find deepest penetrating contact
    #[must_use]
    pub fn collide_sphere(&self, center: Vec3Fix, radius: Fix128) -> Option<Contact> {
        let query = AABB::new(
            center - Vec3Fix::new(radius, radius, radius),
            center + Vec3Fix::new(radius, radius, radius),
        );
        let mut deepest: Option<Contact> = None;
        let mut max_depth = Fix128::ZERO;

        // Candidates are visited in the same order `LinearBvh::query` would
        // return them, without collecting them into a `Vec` first.
        self.bvh.query_callback(&query, |tri_idx| {
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
        });

        deepest
    }

    /// Capsule vs `TriMesh` collision: find deepest contact
    #[must_use]
    pub fn collide_capsule(&self, a: Vec3Fix, b: Vec3Fix, radius: Fix128) -> Option<Contact> {
        let cap_min = Vec3Fix::new(
            if a.x < b.x { a.x } else { b.x } - radius,
            if a.y < b.y { a.y } else { b.y } - radius,
            if a.z < b.z { a.z } else { b.z } - radius,
        );
        let cap_max = Vec3Fix::new(
            if a.x > b.x { a.x } else { b.x } + radius,
            if a.y > b.y { a.y } else { b.y } + radius,
            if a.z > b.z { a.z } else { b.z } + radius,
        );
        let query = AABB::new(cap_min, cap_max);
        let candidates = self.bvh.query(&query);

        let mut deepest: Option<Contact> = None;
        let mut max_depth = Fix128::ZERO;

        for tri_idx in candidates {
            let tri = &self.triangles[tri_idx as usize];
            // Find closest point on capsule segment to closest point on triangle
            let tri_cp = closest_point_segment_triangle(a, b, tri);
            let seg_cp = closest_point_on_segment(a, b, tri_cp);
            let delta = seg_cp - tri_cp;
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
                        point_a: seg_cp - normal * radius,
                        point_b: tri_cp,
                    });
                }
            }
        }
        deepest
    }

    /// AABB (box) vs `TriMesh` collision: the deepest contact over the
    /// triangles that overlap the box.
    ///
    /// Each candidate triangle is tested against the box with the separating
    /// axis theorem on the 13 axes of a triangle-box pair (the 3 box faces,
    /// the triangle normal and the 9 cross products of box and triangle
    /// edges): a triangle overlaps the box exactly when no axis separates them.
    /// The contact is the axis of least overlap: `depth` is that overlap and
    /// `normal` the direction that pushes the box out (from the mesh to the
    /// box); `point_b` is the point of the triangle closest to the box's
    /// deepest point along `-normal` (on a tie in depth, the triangle whose
    /// point is nearest it), and `point_a = point_b - normal * depth`
    /// (the invariant every `Contact` of this module keeps).
    #[must_use]
    pub fn collide_aabb(&self, aabb: &AABB) -> Option<Contact> {
        let candidates = self.bvh.query(aabb);
        let center = (aabb.min + aabb.max) * Fix128::from_ratio(1, 2);
        let half = (aabb.max - aabb.min) * Fix128::from_ratio(1, 2);

        // the deepest contact; on equal depth, the triangle whose point is nearest
        // the box's support point (two coplanar triangles of a floor tie)
        let mut deepest: Option<(Contact, Fix128)> = None;
        for tri_idx in candidates {
            let tri = &self.triangles[tri_idx as usize];
            let Some((depth, normal)) = triangle_box_overlap(tri, center, half) else {
                continue;
            };
            if depth <= Fix128::ZERO {
                continue;
            }
            // the box's deepest point along -normal (the face / edge centre on the
            // axes the normal does not lean along), and the triangle point nearest it
            let lean = |n: Fix128, h: Fix128| {
                if n > Fix128::ZERO {
                    h
                } else if n < Fix128::ZERO {
                    -h
                } else {
                    Fix128::ZERO
                }
            };
            let support = center
                - Vec3Fix::new(
                    lean(normal.x, half.x),
                    lean(normal.y, half.y),
                    lean(normal.z, half.z),
                );
            let point_b = tri.closest_point(support);
            let gap = (support - point_b).length_squared();
            let better = match &deepest {
                None => true,
                Some((best, best_gap)) => {
                    depth > best.depth || (depth == best.depth && gap < *best_gap)
                }
            };
            if better {
                deepest = Some((
                    Contact {
                        depth,
                        normal,
                        point_a: point_b - normal * depth,
                        point_b,
                    },
                    gap,
                ));
            }
        }
        deepest.map(|(c, _)| c)
    }

    /// Number of triangles
    #[inline]
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }
}

/// Separating-axis test of a triangle against the box `center ± half`.
/// `None` when one of the 13 axes separates them (or the triangle has no
/// area); otherwise the overlap along the triangle's own normal and that unit
/// normal, signed to push the box away from the triangle. The contact uses the
/// face axis rather than the axis of least overlap: on a mesh the least-overlap
/// axis of one triangle is often an edge shared with its neighbour (an
/// internal edge), which would push the box sideways off a flat floor.
fn triangle_box_overlap(
    tri: &Triangle,
    center: Vec3Fix,
    half: Vec3Fix,
) -> Option<(Fix128, Vec3Fix)> {
    let v = [tri.v0 - center, tri.v1 - center, tri.v2 - center];
    let edges = [v[1] - v[0], v[2] - v[1], v[0] - v[2]];
    let face = edges[0].cross(edges[1]);
    let face_len = face.length();
    if face_len.is_zero() {
        return None;
    }
    // overlap of the box and the triangle along a unit axis: (push along +l, along -l)
    let pushes = |l: Vec3Fix| {
        let r = half.x * l.x.abs() + half.y * l.y.abs() + half.z * l.z.abs();
        let p = [v[0].dot(l), v[1].dot(l), v[2].dot(l)];
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        if lo > r || hi < -r {
            None
        } else {
            Some((hi + r, r - lo))
        }
    };
    let axes_box = [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z];
    for b in axes_box {
        pushes(b)?;
        for e in edges {
            let axis = b.cross(e);
            let len = axis.length();
            if !len.is_zero() {
                pushes(axis / len)?;
            }
        }
    }
    let n = face / face_len;
    let (up, down) = pushes(n)?;
    Some(if up <= down { (up, n) } else { (down, -n) })
}

/// Moller-Trumbore parallel-check epsilon (~2^-24), relative: the ray counts
/// as parallel to the triangle when `|det| < MT_EPSILON * |e1| * |e2|`, i.e.
/// when the sine of the angle between the ray and the plane times the sine of
/// the triangle's corner angle is below it, whatever the triangle's size
const MT_EPSILON: Fix128 = Fix128 {
    hi: 0,
    lo: 0x0000010000000000,
};

/// Ray-Triangle intersection (Moller-Trumbore algorithm)
#[must_use]
pub fn ray_triangle(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Option<RayHit> {
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = ray.direction.cross(e2);
    let det = e1.dot(h);
    // relative to the triangle's size (an absolute threshold made every
    // triangle below about 2.4e-4 m on a side invisible to every ray)
    if det.is_zero() || det.abs() < MT_EPSILON * e1.length() * e2.length() {
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
        let normal = if normal.dot(ray.direction) > Fix128::ZERO {
            -normal
        } else {
            normal
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

/// Create an AABB enclosing a ray segment
fn ray_to_aabb(ray: &Ray, max_t: Fix128) -> AABB {
    let end = ray.at(max_t);
    let one = Fix128::ONE;
    AABB::new(
        Vec3Fix::new(
            if ray.origin.x < end.x {
                ray.origin.x - one
            } else {
                end.x - one
            },
            if ray.origin.y < end.y {
                ray.origin.y - one
            } else {
                end.y - one
            },
            if ray.origin.z < end.z {
                ray.origin.z - one
            } else {
                end.z - one
            },
        ),
        Vec3Fix::new(
            if ray.origin.x > end.x {
                ray.origin.x + one
            } else {
                end.x + one
            },
            if ray.origin.y > end.y {
                ray.origin.y + one
            } else {
                end.y + one
            },
            if ray.origin.z > end.z {
                ray.origin.z + one
            } else {
                end.z + one
            },
        ),
    )
}

/// Closest point on a line segment to a target point
fn closest_point_on_segment(a: Vec3Fix, b: Vec3Fix, p: Vec3Fix) -> Vec3Fix {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq.is_zero() {
        return a;
    }
    let t = (p - a).dot(ab) / len_sq;
    let t = if t < Fix128::ZERO {
        Fix128::ZERO
    } else if t > Fix128::ONE {
        Fix128::ONE
    } else {
        t
    };
    a + ab * t
}

/// Closest point on triangle to a segment — iteratively refines the closest pair
/// between the segment and triangle for better convergence.
fn closest_point_segment_triangle(seg_a: Vec3Fix, seg_b: Vec3Fix, tri: &Triangle) -> Vec3Fix {
    // Start from segment midpoint
    let mid = (seg_a + seg_b) * Fix128::from_ratio(1, 2);
    let mut tri_cp = tri.closest_point(mid);

    // Iterate: segment→triangle→segment→triangle (2 refinement passes)
    for _ in 0..2 {
        let seg_cp = closest_point_on_segment(seg_a, seg_b, tri_cp);
        tri_cp = tri.closest_point(seg_cp);
    }

    tri_cp
}

#[inline]
fn min3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let ab = if a < b { a } else { b };
    if ab < c {
        ab
    } else {
        c
    }
}

#[inline]
fn max3(a: Fix128, b: Fix128, c: Fix128) -> Fix128 {
    let ab = if a > b { a } else { b };
    if ab > c {
        ab
    } else {
        c
    }
}

#[cfg(all(test, feature = "std"))]
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
            Fix128::ONE,                // Radius 1 => penetrating
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
        assert_eq!(
            cp.y.hi, 0,
            "Closest point should be on triangle plane (y=0)"
        );
    }

    #[test]
    fn test_capsule_trimesh_collision() {
        let mesh = make_ground_mesh();
        // Capsule straddling the ground plane
        let contact = mesh.collide_capsule(
            Vec3Fix::from_int(0, -1, 0),
            Vec3Fix::from_int(0, 1, 0),
            Fix128::from_ratio(5, 10), // radius 0.5
        );
        assert!(contact.is_some(), "Capsule should collide with ground mesh");
    }

    #[test]
    fn test_aabb_trimesh_collision() {
        let mesh = make_ground_mesh();
        // AABB overlapping with ground plane
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));
        let contact = mesh.collide_aabb(&aabb);
        assert!(contact.is_some(), "AABB should collide with ground mesh");
    }

    #[test]
    fn from_triangles_matches_from_indexed_and_covers_every_triangle() {
        // 地面 2 枚 + 離れた高台 1 枚 (BVH が全 primitive を包むことを確認)
        let tris = vec![
            Triangle::new(
                Vec3Fix::from_int(-10, 0, -10),
                Vec3Fix::from_int(10, 0, -10),
                Vec3Fix::from_int(10, 0, 10),
            ),
            Triangle::new(
                Vec3Fix::from_int(-10, 0, -10),
                Vec3Fix::from_int(10, 0, 10),
                Vec3Fix::from_int(-10, 0, 10),
            ),
            Triangle::new(
                Vec3Fix::from_int(99, 5, 99),
                Vec3Fix::from_int(101, 5, 99),
                Vec3Fix::from_int(100, 5, 101),
            ),
        ];
        let mesh = TriMesh::from_triangles(tris.clone());
        assert_eq!(mesh.triangle_count(), 3);
        assert_eq!(mesh.triangles, tris);
        // bounds = 全 triangle AABB の union (独立計算)
        let mut want = tris[0].aabb();
        for t in &tris[1..] {
            want = want.union(&t.aabb());
        }
        assert_eq!(mesh.bounds, want);
        assert_eq!(mesh.bounds.min, Vec3Fix::from_int(-10, 0, -10));
        assert_eq!(mesh.bounds.max, Vec3Fix::from_int(101, 5, 101));

        // 同じ geometry を from_indexed で組んだ mesh と query 結果が一致する
        let vertices = vec![
            Vec3Fix::from_int(-10, 0, -10),
            Vec3Fix::from_int(10, 0, -10),
            Vec3Fix::from_int(10, 0, 10),
            Vec3Fix::from_int(-10, 0, 10),
            Vec3Fix::from_int(99, 5, 99),
            Vec3Fix::from_int(101, 5, 99),
            Vec3Fix::from_int(100, 5, 101),
        ];
        let indexed = TriMesh::from_indexed(&vertices, &[0, 1, 2, 0, 2, 3, 4, 5, 6]);
        assert_eq!(indexed.triangles, mesh.triangles);
        assert_eq!(indexed.bounds, mesh.bounds);

        let down = Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO);
        // (5, 5, -5) の真下は triangle 0、(-5, 5, 5) は triangle 1、(100, 10, 100) は triangle 2
        let cases = [
            (Vec3Fix::from_int(5, 5, -5), 0usize, 5i64),
            (Vec3Fix::from_int(-5, 5, 5), 1, 5),
            (Vec3Fix::from_int(100, 10, 100), 2, 5),
        ];
        for (origin, tri_idx, t_want) in cases {
            let ray = Ray::new(origin, down);
            let hit = mesh.raycast(&ray, Fix128::from_int(100));
            let hit_indexed = indexed.raycast(&ray, Fix128::from_int(100));
            match (hit, hit_indexed) {
                (Some(h), Some(hi)) => {
                    assert_eq!(h.body_index, tri_idx, "origin {origin:?}");
                    // Möller–Trumbore の除算で数 ulp 丸まる
                    let err = (h.t - Fix128::from_int(t_want)).abs();
                    assert!(
                        err < Fix128 { hi: 0, lo: 1 << 24 },
                        "origin {origin:?}: t {:?}",
                        h.t
                    );
                    assert_eq!(
                        (h.t, h.body_index, h.point),
                        (hi.t, hi.body_index, hi.point)
                    );
                }
                other => panic!("both meshes must hit for {origin:?}: {other:?}"),
            }
        }
        // 何もない場所は miss
        let miss = Ray::new(Vec3Fix::from_int(50, 5, 50), down);
        assert!(mesh.raycast(&miss, Fix128::from_int(100)).is_none());
        // closest_point も一致し、高台の真上では triangle 2 の頂点面 (y = 5) に落ちる
        let q = Vec3Fix::from_int(100, 8, 100);
        assert_eq!(mesh.closest_point(q), indexed.closest_point(q));
        let (cp, idx) = mesh.closest_point(q);
        assert_eq!(idx, 2);
        assert_eq!(cp, Vec3Fix::from_int(100, 5, 100));

        // 空 mesh: triangle 0、bounds は退化 AABB、raycast は None
        let empty = TriMesh::from_triangles(Vec::new());
        assert_eq!(empty.triangle_count(), 0);
        assert_eq!(empty.bounds, AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO));
        assert!(empty
            .raycast(
                &Ray::new(Vec3Fix::from_int(0, 5, 0), down),
                Fix128::from_int(100)
            )
            .is_none());
    }

    /// `collide_sphere` as it was written before it switched to
    /// `LinearBvh::query_callback`: collect the candidates with `query`, then
    /// walk them in that order. Kept here as the reference the allocation-free
    /// version must reproduce to the bit.
    fn collide_sphere_via_query(
        mesh: &TriMesh,
        center: Vec3Fix,
        radius: Fix128,
    ) -> Option<Contact> {
        let query = AABB::new(
            center - Vec3Fix::new(radius, radius, radius),
            center + Vec3Fix::new(radius, radius, radius),
        );
        let mut deepest: Option<Contact> = None;
        let mut max_depth = Fix128::ZERO;
        for tri_idx in mesh.bvh.query(&query) {
            let tri = &mesh.triangles[tri_idx as usize];
            let cp = tri.closest_point(center);
            let delta = center - cp;
            let dist_sq = delta.length_squared();
            if dist_sq < radius * radius {
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

    fn contacts_bit_equal(a: Option<Contact>, b: Option<Contact>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.depth == b.depth
                    && a.normal == b.normal
                    && a.point_a == b.point_a
                    && a.point_b == b.point_b
            }
            _ => false,
        }
    }

    /// A bumpy 8x8 terrain (128 triangles, several BVH leaves) plus two
    /// horizontal triangles at y = +1/2 and y = -1/2 over the same footprint,
    /// so a sphere at the origin touches both at exactly the same depth and the
    /// winner is decided by visiting order alone.
    fn terrain_with_tied_pair() -> TriMesh {
        let n = 8i64;
        let mut vertices = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                let y = Fix128::from_ratio((x * 7 + z * 3) % 5, 8);
                vertices.push(Vec3Fix::new(
                    Fix128::from_int(x + 10),
                    y,
                    Fix128::from_int(z),
                ));
            }
        }
        let mut triangles = Vec::new();
        let w = (n + 1) as usize;
        for z in 0..n as usize {
            for x in 0..n as usize {
                let a = vertices[z * w + x];
                let b = vertices[z * w + x + 1];
                let c = vertices[(z + 1) * w + x];
                let d = vertices[(z + 1) * w + x + 1];
                triangles.push(Triangle::new(a, c, b));
                triangles.push(Triangle::new(b, c, d));
            }
        }
        let half = Fix128::from_ratio(1, 2);
        let two = Fix128::from_int(2);
        for y in [half, -half] {
            triangles.push(Triangle::new(
                Vec3Fix::new(-two, y, -two),
                Vec3Fix::new(-two, y, two),
                Vec3Fix::new(two, y, -two),
            ));
        }
        TriMesh::from_triangles(triangles)
    }

    #[test]
    fn collide_sphere_matches_the_collect_then_walk_reference_to_the_bit() {
        let mesh = terrain_with_tied_pair();
        // Every candidate order matters for the tie: both slabs are hit at
        // depth 1/4 by a sphere of radius 3/4 at the origin.
        let tie = mesh.collide_sphere(Vec3Fix::ZERO, Fix128::from_ratio(3, 4));
        let tie_ref = collide_sphere_via_query(&mesh, Vec3Fix::ZERO, Fix128::from_ratio(3, 4));
        assert!(tie.is_some());
        assert!(
            contacts_bit_equal(tie, tie_ref),
            "tie: {tie:?} vs {tie_ref:?}"
        );
        assert_eq!(tie.map(|c| c.depth), Some(Fix128::from_ratio(1, 4)));

        let mut hits = 0usize;
        for zi in -1..=18i64 {
            for xi in -6..=38i64 {
                for yi in [-1i64, 0, 1, 3] {
                    let center = Vec3Fix::new(
                        Fix128::from_ratio(xi, 2),
                        Fix128::from_ratio(yi, 4),
                        Fix128::from_ratio(zi, 2),
                    );
                    for radius in [Fix128::from_ratio(1, 4), Fix128::from_ratio(3, 4)] {
                        let got = mesh.collide_sphere(center, radius);
                        let want = collide_sphere_via_query(&mesh, center, radius);
                        assert!(
                            contacts_bit_equal(got, want),
                            "center {center:?} radius {radius:?}: {got:?} vs {want:?}"
                        );
                        hits += usize::from(got.is_some());
                    }
                }
            }
        }
        // The sweep must actually exercise contacts, not compare None to None.
        assert!(hits > 100, "only {hits} contacts in the sweep");
    }
}
