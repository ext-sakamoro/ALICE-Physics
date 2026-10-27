//! Triangle Mesh Collision
//!
//! Static triangle mesh collider with BVH acceleration.
//! Supports ray queries and closest-point queries against arbitrary meshes.

use crate::bvh::{BvhPrimitive, LinearBvh};
use crate::collider::{Contact, AABB};
use crate::math::{Fix128, Vec3Fix};
use crate::raycast::{Ray, RayHit};

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

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
    /// Exact bounds of each BVH node (the min / max of its triangles' vertices, no rounding), in
    /// the order of `bvh.nodes`: what a ray is tested against (the BVH's own node bounds are whole
    /// units, too coarse to separate geometry smaller than a unit)
    pub(crate) node_bounds: Vec<AABB>,
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

        Self::from_parts(triangles, bvh, bounds)
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

        Self::from_parts(triangles, bvh, bounds)
    }

    /// A mesh from its triangles, their BVH and the overall bounds (the exact node bounds are
    /// derived here, so a restored mesh carries them without storing them)
    pub(crate) fn from_parts(triangles: Vec<Triangle>, bvh: LinearBvh, bounds: AABB) -> Self {
        let node_bounds = refit_node_bounds(&bvh, &triangles);
        Self {
            triangles,
            bvh,
            bounds,
            node_bounds,
        }
    }

    /// Ray query: the nearest triangle the ray meets within `max_t`. The BVH is walked along the
    /// ray (a node is skipped when the ray cannot meet its exact bounds before the nearest hit so
    /// far), so the answer is the one a test of every triangle in index order would give: the
    /// smallest t, and on an exact tie of t the smallest triangle index
    #[must_use]
    pub fn raycast(&self, ray: &Ray, max_t: Fix128) -> Option<RayHit> {
        self.raycast_visits(ray, max_t).0
    }

    /// [`TriMesh::raycast`] with the number of BVH nodes whose bounds were tested (a measure of the
    /// work done: far below the node count when the walk prunes)
    #[must_use]
    pub fn raycast_visits(&self, ray: &Ray, max_t: Fix128) -> (Option<RayHit>, usize) {
        let mut best: Option<RayHit> = None;
        let mut best_t = max_t;
        let mut visits = 0usize;
        let nodes = &self.bvh.nodes;
        let mut idx = 0usize;
        while idx < nodes.len() {
            let node = &nodes[idx];
            visits += 1;
            // inclusive at best_t: a triangle met at the same t may still have a smaller index
            if ray_meets_box(ray, &self.node_bounds[idx], best_t) {
                if node.is_leaf() {
                    let start = node.first_child_or_prim as usize;
                    for &prim in &self.bvh.primitives[start..start + node.prim_count() as usize] {
                        let tri_idx = prim as usize;
                        if let Some(mut hit) = ray_triangle(ray, &self.triangles[tri_idx], best_t) {
                            let nearer = match &best {
                                None => true,
                                Some(b) => hit.t < b.t || (hit.t == b.t && tri_idx < b.body_index),
                            };
                            if nearer {
                                hit.body_index = tri_idx;
                                best_t = hit.t;
                                best = Some(hit);
                            }
                        }
                    }
                    idx = node.escape_idx() as usize;
                } else {
                    idx = node.first_child_or_prim as usize;
                }
            } else {
                idx = node.escape_idx() as usize;
            }
        }
        (best, visits)
    }

    /// Whether the ray meets any triangle within `max_t` (a shadow ray): the same answer as
    /// `raycast(ray, max_t).is_some()`, returning at the first triangle met
    #[must_use]
    pub fn occluded(&self, ray: &Ray, max_t: Fix128) -> bool {
        let nodes = &self.bvh.nodes;
        let mut idx = 0usize;
        while idx < nodes.len() {
            let node = &nodes[idx];
            if ray_meets_box(ray, &self.node_bounds[idx], max_t) {
                if node.is_leaf() {
                    let start = node.first_child_or_prim as usize;
                    for &prim in &self.bvh.primitives[start..start + node.prim_count() as usize] {
                        if ray_triangle(ray, &self.triangles[prim as usize], max_t).is_some() {
                            return true;
                        }
                    }
                    idx = node.escape_idx() as usize;
                } else {
                    idx = node.first_child_or_prim as usize;
                }
            } else {
                idx = node.escape_idx() as usize;
            }
        }
        false
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

        // Grow a box around the point until the best candidate is no farther
        // than the box's half-size: every triangle nearer than that intersects
        // the box, so the answer is exact at any distance (a fixed 1000 box
        // found nothing past it and answered triangles[0].v0). Ties go to the
        // smallest triangle index, whatever order the BVH visits them in.
        let span =
            (self.bounds.max - self.bounds.min).length() + (point - self.bounds.min).length();
        let mut half = Fix128::ONE;
        loop {
            let query = AABB::new(
                point - Vec3Fix::new(half, half, half),
                point + Vec3Fix::new(half, half, half),
            );
            let mut best: Option<(Fix128, usize, Vec3Fix)> = None;
            for tri_idx in self.bvh.query(&query) {
                let idx = tri_idx as usize;
                let cp = self.triangles[idx].closest_point(point);
                let dist_sq = (cp - point).length_squared();
                let better = match best {
                    None => true,
                    Some((d, i, _)) => dist_sq < d || (dist_sq == d && idx < i),
                };
                if better {
                    best = Some((dist_sq, idx, cp));
                }
            }
            if let Some((dist_sq, idx, cp)) = best {
                if dist_sq <= half * half || half > span {
                    return (cp, idx);
                }
            }
            if half > span {
                // the box holds the whole mesh and still found nothing: an empty BVH
                return (self.triangles[0].v0, 0);
            }
            half = half.double();
        }
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
            // the exact closest pair between the capsule's segment and the triangle
            let (seg_cp, tri_cp) = closest_points_segment_triangle(a, b, tri);
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
    /// The contact of an overlapping triangle is taken along its face normal
    /// (not the axis of least overlap, which on a flat mesh can be an internal
    /// edge): `depth` is the box's overlap along the face normal, the smaller
    /// of the two ways out, and `normal` that way (from the mesh to the box).
    /// A box hanging over a convex edge (a step or a ledge) therefore overlaps
    /// the riser's face by more than its minimum translation and can be
    /// pushed sideways; `point_b` is the point of the triangle closest to the box's
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

/// The parallel threshold of the ray / triangle test (2^-48), relative: the ray counts as
/// parallel to the triangle when the sine of the angle between the ray and the triangle's plane
/// is below it, |det| < 2^-48 · |e1 × e2| · |d| (`det = −d · (e1 × e2)`), for any shape of
/// triangle and any length of `d`. Evaluated exactly, squared: det² · 2^96 < |e1 × e2|² · |d|².
/// It used to be 2^-24, which dropped real grazing hits (a sphere resting on a mesh floor and cast
/// at a slope of 2^-31 found nothing); a ray lying in a triangle's plane up to the rounding of its
/// direction (a sine near 2^-64) is still parallel.
const MT_PARALLEL_LOG2: u32 = 48;

/// Limbs of [`Exact`]: 1024 bits. The widest value the ray / triangle test forms is det² · 2^96
/// (det is a triple product of 129-bit differences, under 2^390), under 2^880.
const EXACT_LIMBS: usize = 16;

/// A two's-complement integer of `64 · N` bits: the ray / triangle test on the raw `Fix128` values
/// (each a 128-bit integer, the value times 2^64), without rounding. Every value formed stays far
/// inside the range, so additions and products never wrap.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Exact<const N: usize>([u64; N]);

impl<const N: usize> Exact<N> {
    /// The raw integer of `x` (`hi · 2^64 + lo`)
    const fn from_fix(x: Fix128) -> Self {
        let ext = if x.hi < 0 { u64::MAX } else { 0 };
        let mut w = [ext; N];
        w[0] = x.lo;
        w[1] = x.hi as u64;
        Self(w)
    }

    const fn is_negative(&self) -> bool {
        (self.0[N - 1] as i64) < 0
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }

    fn add(&self, o: &Self) -> Self {
        let mut w = [0u64; N];
        let mut carry = 0u64;
        for (i, out) in w.iter_mut().enumerate() {
            let (s1, c1) = self.0[i].overflowing_add(o.0[i]);
            let (s2, c2) = s1.overflowing_add(carry);
            *out = s2;
            carry = u64::from(c1) + u64::from(c2);
        }
        Self(w)
    }

    fn neg(&self) -> Self {
        let mut inv = [0u64; N];
        for (out, &l) in inv.iter_mut().zip(&self.0) {
            *out = !l;
        }
        let mut one = [0u64; N];
        one[0] = 1;
        Self(inv).add(&Self(one))
    }

    fn sub(&self, o: &Self) -> Self {
        self.add(&o.neg())
    }

    fn abs(&self) -> Self {
        if self.is_negative() {
            self.neg()
        } else {
            *self
        }
    }

    /// The product, schoolbook on the magnitudes, kept to 1024 bits (the operands' magnitudes
    /// add up to fewer bits)
    fn mul(&self, o: &Self) -> Self {
        let (a, b) = (self.abs(), o.abs());
        // limbs in use (the magnitudes' high limbs are 0)
        let len = |x: &Self| x.0.iter().rposition(|&l| l != 0).map_or(0, |k| k + 1);
        let (la, lb) = (len(&a), len(&b));
        let mut w = [0u64; N];
        for i in 0..la {
            if a.0[i] == 0 {
                continue;
            }
            let mut carry = 0u128;
            for j in 0..lb.min(N - i) {
                let cur = u128::from(w[i + j]) + u128::from(a.0[i]) * u128::from(b.0[j]) + carry;
                w[i + j] = cur as u64;
                carry = cur >> 64;
            }
            // the carry out of the last used limb of b
            let mut k = i + lb;
            while carry != 0 && k < N {
                let cur = u128::from(w[k]) + carry;
                w[k] = cur as u64;
                carry = cur >> 64;
                k += 1;
            }
        }
        let p = Self(w);
        if self.is_negative() == o.is_negative() {
            p
        } else {
            p.neg()
        }
    }

    /// `self` · 2^bits, for a non-negative value and bits < 1024
    fn shl(&self, bits: u32) -> Self {
        let (limbs, rest) = ((bits / 64) as usize, bits % 64);
        let mut w = [0u64; N];
        for i in (limbs..N).rev() {
            let lo = self.0[i - limbs];
            let below = if rest == 0 || i == limbs {
                0
            } else {
                self.0[i - limbs - 1] >> (64 - rest)
            };
            w[i] = if rest == 0 { lo } else { (lo << rest) | below };
        }
        Self(w)
    }

    /// self < o
    fn lt(&self, o: &Self) -> bool {
        self.sub(o).is_negative()
    }

    /// floor(self / d) as [`Exact::div_floor_fix`], starting from a guess of the quotient and
    /// correcting it by comparing exact products (q · d ≤ self < (q + 1) · d); a guess more than
    /// 64 steps off falls back to the long division
    fn div_floor_fix_near(&self, d: &Self, guess: Fix128) -> Fix128 {
        let raw = |q: i128| Fix128::from_raw((q >> 64) as i64, q as u64);
        let mut q = (i128::from(guess.hi) << 64) | i128::from(guess.lo);
        for _ in 0..64 {
            let below = Self::from_fix(raw(q)).mul(d);
            if self.lt(&below) {
                q -= 1;
                continue;
            }
            let above = below.add(d);
            if !self.lt(&above) {
                q += 1;
                continue;
            }
            return raw(q);
        }
        self.div_floor_fix(d)
    }

    /// floor(self / d) for self ≥ 0, d > 0 and a quotient below 2^127: the `Fix128` whose raw
    /// integer it is
    fn div_floor_fix(&self, d: &Self) -> Fix128 {
        let mut rem = *self;
        let mut q = 0u128;
        for bit in (0..127).rev() {
            let shifted = d.shl(bit);
            if !rem.lt(&shifted) {
                rem = rem.sub(&shifted);
                q |= 1 << bit;
            }
        }
        Fix128::from_raw((q >> 64) as i64, q as u64)
    }
}

/// A `Vec3Fix` on the raw integers
#[derive(Clone, Copy)]
struct ExactVec<const N: usize>([Exact<N>; 3]);

impl<const N: usize> ExactVec<N> {
    const fn from_vec(v: Vec3Fix) -> Self {
        Self([Exact::from_fix(v.x), Exact::from_fix(v.y), Exact::from_fix(v.z)])
    }

    fn sub(&self, o: &Self) -> Self {
        Self([self.0[0].sub(&o.0[0]), self.0[1].sub(&o.0[1]), self.0[2].sub(&o.0[2])])
    }

    fn dot(&self, o: &Self) -> Exact<N> {
        self.0[0].mul(&o.0[0]).add(&self.0[1].mul(&o.0[1])).add(&self.0[2].mul(&o.0[2]))
    }

    fn cross(&self, o: &Self) -> Self {
        let [a, b, c] = &self.0;
        let [x, y, z] = &o.0;
        Self([
            b.mul(z).sub(&c.mul(y)),
            c.mul(x).sub(&a.mul(z)),
            a.mul(y).sub(&b.mul(x)),
        ])
    }
}

/// Exact bounds of every BVH node: a leaf's are the min / max of its triangles' vertices, an
/// internal node's the union of its children's (children are after their parent in the array, so
/// one pass from the end has every child done before its parent; a node's children are its first
/// child and that child's escape chain up to the node's own escape)
fn refit_node_bounds(bvh: &LinearBvh, triangles: &[Triangle]) -> Vec<AABB> {
    let nodes = &bvh.nodes;
    let empty = AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO);
    let mut out = vec![empty; nodes.len()];
    for i in (0..nodes.len()).rev() {
        let node = &nodes[i];
        let mut acc: Option<AABB> = None;
        let mut add = |b: AABB| {
            acc = Some(match acc {
                None => b,
                Some(a) => AABB::new(
                    Vec3Fix::new(
                        min3(a.min.x, b.min.x, b.min.x),
                        min3(a.min.y, b.min.y, b.min.y),
                        min3(a.min.z, b.min.z, b.min.z),
                    ),
                    Vec3Fix::new(
                        max3(a.max.x, b.max.x, b.max.x),
                        max3(a.max.y, b.max.y, b.max.y),
                        max3(a.max.z, b.max.z, b.max.z),
                    ),
                ),
            });
        };
        if node.is_leaf() {
            let start = node.first_child_or_prim as usize;
            for &prim in &bvh.primitives[start..start + node.prim_count() as usize] {
                add(triangles[prim as usize].aabb());
            }
        } else {
            let end = node.escape_idx() as usize;
            let mut c = node.first_child_or_prim as usize;
            while c < nodes.len() && c != end && c > i {
                add(out[c]);
                c = nodes[c].escape_idx() as usize;
            }
        }
        out[i] = acc.unwrap_or(empty);
    }
    out
}

/// How far a node's bounds are widened before a ray is tested against them: 2^-32 units, far
/// below any geometry and far above the rounding of the triangle test, so a ray the triangle test
/// accepts at a vertex or along an edge is never refused by its node
const BOX_PAD: Fix128 = Fix128 { hi: 0, lo: 1 << 32 };

/// The slack of a compared product: `Fix128` multiplication keeps the middle 128 bits of the
/// 256-bit product, which floors (rounds toward negative infinity) by less than one step of
/// 2^-64; each side of a comparison is off by less than one step, so a pad of 4 steps toward
/// acceptance keeps the test conservative
const PRODUCT_SLACK: Fix128 = Fix128 { hi: 0, lo: 4 };

/// Whether the ray can meet the box at some t in [0, max_t] (conservative: true whenever the exact
/// ray does). Per axis with a direction component d != 0 the ray is inside the slab for t between
/// enter = (near - o) / |d| and exit = (far - o) / |d|; the ray meets the box when the largest
/// enter is at most the smallest exit, 0 and max_t included. Without dividing: enter_k <= exit_m
/// as enter_num_k * |d_m| <= exit_num_m * |d_k| for every pair of axes, enter_k <= max_t as
/// enter_num_k <= max_t * |d_k|, exit_m >= 0 as exit_num_m >= 0. An axis with d = 0 needs the
/// origin inside the slab (bounds included). A product that overflows counts as met
fn ray_meets_box(ray: &Ray, b: &AABB, max_t: Fix128) -> bool {
    let o = [ray.origin.x, ray.origin.y, ray.origin.z];
    let d = [ray.direction.x, ray.direction.y, ray.direction.z];
    let lo = [b.min.x - BOX_PAD, b.min.y - BOX_PAD, b.min.z - BOX_PAD];
    let hi = [b.max.x + BOX_PAD, b.max.y + BOX_PAD, b.max.z + BOX_PAD];
    // (enter numerator, exit numerator, |d|) of the axes that move
    let mut axes: [(Fix128, Fix128, Fix128); 3] = [(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO); 3];
    let mut n = 0;
    for k in 0..3 {
        if d[k].is_zero() {
            if o[k] < lo[k] || o[k] > hi[k] {
                return false;
            }
        } else if d[k].is_negative() {
            axes[n] = (o[k] - hi[k], o[k] - lo[k], -d[k]);
            n += 1;
        } else {
            axes[n] = (lo[k] - o[k], hi[k] - o[k], d[k]);
            n += 1;
        }
    }
    let axes = &axes[..n];
    let le = |a: Fix128, b: Fix128, c: Fix128, e: Fix128| -> bool {
        // a * b <= c * e (with the slack toward true)
        match (a.checked_mul(b), c.checked_mul(e)) {
            (Some(l), Some(r)) => l <= r + PRODUCT_SLACK,
            _ => true,
        }
    };
    for &(enter, exit, dk) in axes {
        if exit < -PRODUCT_SLACK {
            return false;
        }
        if !le(enter, Fix128::ONE, max_t, dk) {
            return false;
        }
        for &(_, exit_m, dm) in axes {
            if !le(enter, dm, exit_m, dk) {
                return false;
            }
        }
    }
    true
}

/// Ray-Triangle intersection (Moller-Trumbore algorithm), exact
///
/// The ray hits the triangle exactly when its crossing with the triangle's plane lies in the
/// closed triangle (edges and vertices included) at t in [0, max_t], and the ray is not parallel
/// to the plane ([`MT_PARALLEL_LOG2`]). Every quantity is formed on the raw `Fix128` integers in
/// [`Exact`], so nothing rounds: with the barycentric numerators rounded, a ray grazing an edge
/// was accepted up to ~2^-63 / sinθ outside the triangle, farther than a node's padded bounds,
/// and the mesh query disagreed with a loop over the triangles. The hit's t is the exact t
/// floored to a step of 2^-64, its point `ray.at(t)` and its normal the triangle's unit normal
/// turned to face the ray.
///
/// Most calls are decided by [`certainly_misses`] in `Fix128` arithmetic with a proven bound on
/// its rounding; the rest (every hit, and every miss within the bound of an edge, of t = 0, of
/// t = max_t or of the parallel threshold) run the exact test, so the answer and the t are the
/// exact ones in every case.
#[must_use]
pub fn ray_triangle(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Option<RayHit> {
    match fast_test(ray, tri, max_t) {
        Fast::Miss => None,
        Fast::Hit { guess } => Some(hit_at(ray, tri, exact_hit_t(ray, tri, guess))),
        Fast::Unsure => ray_triangle_exact(ray, tri, max_t),
    }
}

/// One step of `Fix128`, 2^-64
const ULP: Fix128 = Fix128 { hi: 0, lo: 1 };

/// Coordinates (vertices, origin, direction) below 2^15 in magnitude keep every value the fast
/// test forms below 2^51 (differences < 2^16, cross products < 2^33, dot products < 3 · 2^49),
/// so no `Fix128` addition wraps; outside it the exact test runs
const FAST_RANGE: i64 = 1 << 15;

/// The fast test's bound on the rounding of a dot product `a · b` where `b` is a cross product,
/// in steps of 2^-64. Every `Fix128` product is off by less than one step (it keeps the middle
/// 128 bits of the 256-bit product); additions and subtractions are exact in range. A cross
/// product's component is a difference of two products, off by less than 2 steps; the dot
/// product a · b_rounded is off by less than Σ |a_i| · 2 (from b) + 3 (its own three products),
/// so by less than 2 · ceil(‖a‖₁) + 3 steps. Taken as 2 · (⌊‖a‖₁⌋ + 1) + 3.
fn dot_bound(a: Vec3Fix) -> Fix128 {
    let l1 = a.x.abs() + a.y.abs() + a.z.abs();
    Fix128::from_raw(0, 2 * (l1.hi as u64 + 1) + 3)
}

/// The fast test's verdict
enum Fast {
    /// The exact test certainly reports a miss
    Miss,
    /// The exact test certainly reports a hit; `guess` is the rounded t
    Hit { guess: Fix128 },
    /// Within a bound of a decision, or out of [`FAST_RANGE`]: the exact test decides
    Unsure,
}

/// Coordinates in [`FAST_RANGE`]
fn in_fast_range(ray: &Ray, tri: &Triangle) -> bool {
    let small = |v: Vec3Fix| [v.x, v.y, v.z].iter().all(|c| c.hi > -FAST_RANGE && c.hi < FAST_RANGE);
    small(ray.origin) && small(ray.direction) && small(tri.v0) && small(tri.v1) && small(tri.v2)
}

/// The exact test's verdict where the `Fix128` numerators decide it ([`dot_bound`]): `det` =
/// e1 · (d × e2), `u` = s · (d × e2), `v` = d · (s × e1) and `t_num` = e2 · (s × e1), each within
/// its bound of the exact value. When |det| exceeds its bound its sign is the exact sign; then a
/// numerator beyond its bound on the wrong side of 0, of det (u, u + v, bounds added) or of
/// max_t · det (t_num) fails that inequality exactly as well (a miss), and all of them beyond
/// their bounds on the right side pass exactly (a hit, when the ray is also certainly not
/// parallel). Anything nearer, or out of [`FAST_RANGE`], is [`Fast::Unsure`].
fn fast_test(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Fast {
    if !in_fast_range(ray, tri) {
        return Fast::Unsure;
    }
    let d = ray.direction;
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let s = ray.origin - tri.v0;
    let h = d.cross(e2);
    let det = e1.dot(h);
    let det_bound = dot_bound(e1);
    if det.abs() <= det_bound {
        return Fast::Unsure;
    }
    let flip = det.is_negative();
    let signed = |x: Fix128| if flip { -x } else { x };
    let det = det.abs();
    let (u, u_bound) = (signed(s.dot(h)), dot_bound(s));
    if u < -u_bound || u - det > u_bound + det_bound {
        return Fast::Miss;
    }
    let q = s.cross(e1);
    let (v, v_bound) = (signed(d.dot(q)), dot_bound(d));
    if v < -v_bound || u + v - det > u_bound + v_bound + det_bound {
        return Fast::Miss;
    }
    let (t_num, t_bound) = (signed(e2.dot(q)), dot_bound(e2));
    if t_num < -t_bound {
        return Fast::Miss;
    }
    // t_num against max_t · det: the product floors by less than a step, and det's error scales
    // by max_t (kept below 2^20 so it stays in range); a negative reach is decided exactly
    if max_t.hi < 0 || max_t.hi >= (1 << 20) {
        return Fast::Unsure;
    }
    let Some(limit) = max_t.checked_mul(det) else {
        return Fast::Unsure;
    };
    let t_margin = t_bound + Fix128::from_raw(0, det_bound.lo * (max_t.hi as u64 + 1)) + ULP;
    if t_num - limit > t_margin {
        return Fast::Miss;
    }
    // a hit: every inequality clears its bound on the passing side ...
    let clear = u > u_bound
        && det - u > u_bound + det_bound
        && v > v_bound
        && det - u - v > u_bound + v_bound + det_bound
        && t_num > t_bound
        && limit - t_num > t_margin;
    // ... and the ray is certainly not parallel: |det| > 2^-48 · |n| · |d| exactly, from
    // |n| · |d| ≤ ‖n‖₁ · ‖d‖₁ with n = e1 × e2 rounded (each component within 2 steps, so ‖n‖₁
    // within 6) and each floored product raised by a step
    let l1 = |a: Vec3Fix| a.x.abs() + a.y.abs() + a.z.abs();
    let n = e1.cross(e2);
    let parallel_limit = (l1(n) + Fix128::from_raw(0, 6)) * l1(d) + ULP;
    let parallel_limit = parallel_limit * Fix128::from_raw(0, 1 << (64 - MT_PARALLEL_LOG2)) + ULP;
    if clear && det - det_bound > parallel_limit {
        if let Some(guess) = t_num.checked_div(det) {
            return Fast::Hit { guess };
        }
    }
    Fast::Unsure
}

/// Limbs for the exact t of a hit decided by [`fast_test`]: with coordinates in [`FAST_RANGE`] the
/// raw integers are below 2^79, det and t_num below 2^243, t_num · 2^64 below 2^307 and q · det
/// (q ≤ max_t < 2^84 in raw steps) below 2^328, all inside 384 bits
const HIT_LIMBS: usize = 6;

/// The exact t of a hit the fast test decided: t_num · 2^64 / det floored, from the guess
fn exact_hit_t(ray: &Ray, tri: &Triangle, guess: Fix128) -> Fix128 {
    let d = ExactVec::<HIT_LIMBS>::from_vec(ray.direction);
    let v0 = ExactVec::<HIT_LIMBS>::from_vec(tri.v0);
    let e1 = ExactVec::<HIT_LIMBS>::from_vec(tri.v1).sub(&v0);
    let e2 = ExactVec::<HIT_LIMBS>::from_vec(tri.v2).sub(&v0);
    let s = ExactVec::<HIT_LIMBS>::from_vec(ray.origin).sub(&v0);
    let det = e1.dot(&d.cross(&e2));
    let t_num = e2.dot(&s.cross(&e1));
    let (det, t_num) = if det.is_negative() { (det.neg(), t_num.neg()) } else { (det, t_num) };
    t_num.shl(64).div_floor_fix_near(&det, guess)
}

/// The hit at t: the point `ray.at(t)` and the triangle's unit normal turned to face the ray
fn hit_at(ray: &Ray, tri: &Triangle, t: Fix128) -> RayHit {
    let normal = tri.unit_normal();
    let normal = if normal.dot(ray.direction) > Fix128::ZERO {
        -normal
    } else {
        normal
    };
    RayHit {
        t,
        point: ray.at(t),
        normal,
        body_index: 0,
    }
}

/// A guess of the hit's t from the `Fix128` numerators (rounded), for coordinates in
/// [`FAST_RANGE`] and a rounded det away from 0; only the starting point of the exact division
fn rounded_t(ray: &Ray, tri: &Triangle) -> Option<Fix128> {
    if !in_fast_range(ray, tri) {
        return None;
    }
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let s = ray.origin - tri.v0;
    let det = e1.dot(ray.direction.cross(e2));
    if det.abs() <= dot_bound(e1) {
        return None;
    }
    let t_num = e2.dot(s.cross(e1));
    t_num.checked_div(det)
}

/// The exact ray / triangle test ([`ray_triangle`] without the fast filter)
fn ray_triangle_exact(ray: &Ray, tri: &Triangle, max_t: Fix128) -> Option<RayHit> {
    let d = ExactVec::<EXACT_LIMBS>::from_vec(ray.direction);
    let v0 = ExactVec::<EXACT_LIMBS>::from_vec(tri.v0);
    let e1 = ExactVec::<EXACT_LIMBS>::from_vec(tri.v1).sub(&v0);
    let e2 = ExactVec::<EXACT_LIMBS>::from_vec(tri.v2).sub(&v0);
    let h = d.cross(&e2);
    let det = e1.dot(&h);
    // parallel: det² · 2^96 < |e1 × e2|² · |d|² (0 included)
    let n = e1.cross(&e2);
    if det.mul(&det).shl(2 * MT_PARALLEL_LOG2).lt(&n.dot(&n).mul(&d.dot(&d))) || det.is_zero() {
        return None;
    }

    // the barycentric and distance numerators against det, its sign folded in
    let flip = det.is_negative();
    let signed = |x: Exact<EXACT_LIMBS>| if flip { x.neg() } else { x };
    let det = det.abs();
    let s = ExactVec::<EXACT_LIMBS>::from_vec(ray.origin).sub(&v0);
    let u = signed(s.dot(&h));
    if u.is_negative() || det.lt(&u) {
        return None;
    }
    let q = s.cross(&e1);
    let v = signed(d.dot(&q));
    if v.is_negative() || det.lt(&u.add(&v)) {
        return None;
    }
    let t_num = signed(e2.dot(&q));
    // 0 <= t = t_num / det <= max_t, with t's raw integer t_num · 2^64 / det
    let scaled = t_num.shl(64);
    if t_num.is_negative() || Exact::<EXACT_LIMBS>::from_fix(max_t).mul(&det).lt(&scaled) {
        return None;
    }
    let t = match rounded_t(ray, tri) {
        Some(guess) => scaled.div_floor_fix_near(&det, guess),
        None => scaled.div_floor_fix(&det),
    };
    Some(hit_at(ray, tri, t))
}

/// Closest pair `(point on segment, point on triangle)` between the segment
/// `seg_a`–`seg_b` and `tri`, exact: zero distance where the segment crosses
/// the triangle, otherwise the nearest of the two endpoints against the
/// triangle and the segment against each of the three edges (the distance
/// between a segment and a triangle that do not cross is attained at one of
/// these). Ties keep the first candidate in that order.
fn closest_points_segment_triangle(
    seg_a: Vec3Fix,
    seg_b: Vec3Fix,
    tri: &Triangle,
) -> (Vec3Fix, Vec3Fix) {
    if let Some(p) = segment_crosses_triangle(seg_a, seg_b, tri) {
        return (p, p);
    }
    let mut best = (seg_a, tri.closest_point(seg_a));
    let mut best_d = (best.0 - best.1).length_squared();
    let mut consider = |pair: (Vec3Fix, Vec3Fix)| {
        let d = (pair.0 - pair.1).length_squared();
        if d < best_d {
            best_d = d;
            best = pair;
        }
    };
    consider((seg_b, tri.closest_point(seg_b)));
    for (e0, e1) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)] {
        consider(closest_points_segment_segment(seg_a, seg_b, e0, e1));
    }
    best
}

/// The point where the segment `a`–`b` crosses the triangle, if it does
/// (Moller-Trumbore with the unnormalised direction, parameter in `[0, 1]`).
fn segment_crosses_triangle(a: Vec3Fix, b: Vec3Fix, tri: &Triangle) -> Option<Vec3Fix> {
    let d = b - a;
    let e1 = tri.v1 - tri.v0;
    let e2 = tri.v2 - tri.v0;
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.is_zero() {
        return None;
    }
    let inv = Fix128::ONE / det;
    let s = a - tri.v0;
    let u = s.dot(h) * inv;
    if u < Fix128::ZERO || u > Fix128::ONE {
        return None;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < Fix128::ZERO || u + v > Fix128::ONE {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t < Fix128::ZERO || t > Fix128::ONE {
        return None;
    }
    Some(a + d * t)
}

/// Closest points between the segments `p1`–`q1` and `p2`–`q2` (Ericson,
/// Real-Time Collision Detection 5.1.9), each parameter clamped to `[0, 1]`.
fn closest_points_segment_segment(
    p1: Vec3Fix,
    q1: Vec3Fix,
    p2: Vec3Fix,
    q2: Vec3Fix,
) -> (Vec3Fix, Vec3Fix) {
    let clamp01 = |x: Fix128| x.max(Fix128::ZERO).min(Fix128::ONE);
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let (s, t) = if a.is_zero() && e.is_zero() {
        (Fix128::ZERO, Fix128::ZERO)
    } else if a.is_zero() {
        (Fix128::ZERO, clamp01(f / e))
    } else {
        let c = d1.dot(r);
        if e.is_zero() {
            (clamp01(-c / a), Fix128::ZERO)
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s = if denom.is_zero() {
                Fix128::ZERO
            } else {
                clamp01((b * f - c * e) / denom)
            };
            let mut t = (b * s + f) / e;
            if t < Fix128::ZERO {
                t = Fix128::ZERO;
                s = clamp01(-c / a);
            } else if t > Fix128::ONE {
                t = Fix128::ONE;
                s = clamp01((b - c) / a);
            }
            (s, t)
        }
    };
    (p1 + d1 * s, p2 + d2 * t)
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

    /// A ray lying in the plane of a triangle 1e-9 m across, up to the rounding
    /// of its direction, is parallel to it (a sine near 2^-64, far below
    /// [`MT_PARALLEL_LOG2`]). Before the threshold 1 in 6 of such rays was a hit.
    #[test]
    fn a_ray_in_the_plane_of_a_tiny_triangle_is_parallel() {
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            Fix128::from_raw(0, seed) - Fix128::from_ratio(1, 2)
        };
        let tiny = Fix128::from_f64(1e-9);
        let mut hits = 0;
        for _ in 0..2000 {
            let v0 = Vec3Fix::new(rnd(), rnd(), rnd());
            let a = Vec3Fix::new(rnd() * tiny, rnd() * tiny, rnd() * tiny);
            let b = Vec3Fix::new(rnd() * tiny, rnd() * tiny, rnd() * tiny);
            let tri = Triangle {
                v0,
                v1: v0 + a,
                v2: v0 + b,
            };
            let direction = (a + b * Fix128::from_ratio(3, 10)).normalize();
            let ray = Ray {
                origin: v0 + (a + b) * Fix128::from_ratio(1, 4) - direction,
                direction,
            };
            if ray_triangle(&ray, &tri, Fix128::from_int(10)).is_some() {
                hits += 1;
            }
        }
        assert_eq!(
            hits, 0,
            "{hits} of 2000 rays in a tiny triangle's plane hit it"
        );
    }

    /// Whether a grazing ray counts as parallel depends on the angle alone: the
    /// same ray with its direction scaled by 4 or 1/4 (as a `Ray` built without
    /// `Ray::new` may carry) gets the same answer on both sides of `2^-48`.
    #[test]
    fn the_parallel_threshold_does_not_depend_on_the_direction_length() {
        let tri = Triangle {
            v0: Vec3Fix::ZERO,
            v1: Vec3Fix::UNIT_X,
            v2: Vec3Fix::UNIT_Y,
        };
        let target = Vec3Fix::new(
            Fix128::from_ratio(1, 4),
            Fix128::from_ratio(1, 4),
            Fix128::ZERO,
        );
        let hits = |sine: Fix128, scale: Fix128| {
            let direction = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, sine);
            let ray = Ray {
                origin: target - direction,
                direction: direction * scale,
            };
            ray_triangle(&ray, &tri, Fix128::from_int(100)).is_some()
        };
        let below = Fix128::ONE / Fix128::from_int(1i64 << 49);
        let above = Fix128::ONE / Fix128::from_int(1i64 << 47);
        for scale in [Fix128::ONE, Fix128::from_int(4), Fix128::from_ratio(1, 4)] {
            assert!(
                !hits(below, scale),
                "sine 2^-25 at direction scale {scale:?} hit"
            );
            assert!(
                hits(above, scale),
                "sine 2^-23 at direction scale {scale:?} missed"
            );
        }
    }

    #[test]
    fn segment_segment_closest_points_clamp_both_ends() {
        let v = |x: i64, y: i64, z: i64| Vec3Fix::from_int(x, y, z);
        // the lines meet at (4, 0, 0), past the second segment's end: t clamps to
        // 1 at (3, 1, 0) and s is re-solved for that point, (3, 0, 0)
        let (a, b) = closest_points_segment_segment(v(0, 0, 0), v(4, 0, 0), v(1, 3, 0), v(3, 1, 0));
        assert_eq!((a, b), (v(3, 0, 0), v(3, 1, 0)));
        // crossing skew segments: the common perpendicular, (1, 0, 0) and (1, 0, 2)
        let (a, b) =
            closest_points_segment_segment(v(0, 0, 0), v(2, 0, 0), v(1, -1, 2), v(1, 1, 2));
        assert_eq!((a, b), (v(1, 0, 0), v(1, 0, 2)));
        // the first segment ends before the foot (s clamps to 1), then t follows
        let (a, b) =
            closest_points_segment_segment(v(0, 0, 0), v(1, 0, 0), v(3, -1, 0), v(3, 1, 0));
        assert_eq!((a, b), (v(1, 0, 0), v(3, 0, 0)));
    }

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

    fn half() -> Fix128 {
        Fix128::from_ratio(1, 2)
    }

    fn quarter() -> Fix128 {
        Fix128::from_ratio(1, 4)
    }

    fn vq(x: Fix128, y: Fix128, z: Fix128) -> Vec3Fix {
        Vec3Fix::new(x, y, z)
    }

    /// within 2^-40 per component (the closest point on a triangle rounds in
    /// its last bits)
    fn assert_near(got: Vec3Fix, want: Vec3Fix) {
        let tol = Fix128 { hi: 0, lo: 1 << 24 };
        let d = got - want;
        assert!(
            d.x.abs() <= tol && d.y.abs() <= tol && d.z.abs() <= tol,
            "{got:?} vs {want:?}"
        );
    }

    #[test]
    fn trailing_indices_are_ignored_and_an_empty_mesh_answers_the_query_point() {
        let vertices = [
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 0, 1),
        ];
        let mesh = TriMesh::from_indexed(&vertices, &[0, 1, 2, 0, 1]);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.triangles[0].v2, Vec3Fix::from_int(0, 0, 1));

        let empty = TriMesh::from_indexed(&vertices, &[]);
        assert_eq!(empty.triangle_count(), 0);
        let p = Vec3Fix::from_int(3, -2, 7);
        assert_eq!(empty.closest_point(p), (p, 0));
    }

    #[test]
    fn capsule_parallel_above_the_floor_has_depth_radius_minus_gap() {
        // the segment runs along x a quarter above y = 0, parallel to two
        // floor edges: depth 1/2 - 1/4, normal +y, contact under the segment
        let mesh = make_ground_mesh();
        let c = mesh
            .collide_capsule(
                vq(Fix128::from_int(-1), quarter(), Fix128::ZERO),
                vq(Fix128::from_int(1), quarter(), Fix128::ZERO),
                half(),
            )
            .expect("a quarter gap under a half radius is a contact");
        assert_eq!(c.depth, quarter());
        assert_near(c.normal, Vec3Fix::UNIT_Y);
        assert_eq!(c.point_b.y, Fix128::ZERO);
        assert_eq!(c.point_a, c.point_b - c.normal * c.depth);
        // a gap equal to the radius is not a contact
        assert!(mesh
            .collide_capsule(
                vq(Fix128::from_int(-1), half(), Fix128::ZERO),
                vq(Fix128::from_int(1), half(), Fix128::ZERO),
                half(),
            )
            .is_none());
    }

    #[test]
    fn capsule_above_or_crossing_the_floor() {
        let mesh = make_ground_mesh();
        // vertical, bottom end a quarter above the floor: the plane lies
        // before the segment's start, depth 1/2 - 1/4 at (2, 0, -3)
        let c = mesh
            .collide_capsule(
                vq(Fix128::from_int(2), quarter(), Fix128::from_int(-3)),
                Vec3Fix::from_int(2, 3, -3),
                half(),
            )
            .unwrap();
        assert_eq!(c.depth, quarter());
        assert_near(c.normal, Vec3Fix::UNIT_Y);
        assert_near(c.point_b, Vec3Fix::from_int(2, 0, -3));

        // crossing the floor at (5, 0, -5) (inside one triangle, outside the
        // other): zero distance, depth = radius along the triangle's normal
        let c = mesh
            .collide_capsule(
                Vec3Fix::from_int(5, -1, -5),
                Vec3Fix::from_int(5, 1, -5),
                half(),
            )
            .unwrap();
        assert_eq!(c.depth, half());
        assert_near(c.point_b, Vec3Fix::from_int(5, 0, -5));
        assert_eq!(c.normal.x, Fix128::ZERO);
        assert_eq!(c.normal.y.abs(), Fix128::ONE);
        assert_eq!(c.normal.z, Fix128::ZERO);

        // a zero-length capsule is a sphere: depth radius - height
        let p = vq(Fix128::from_int(-3), quarter(), Fix128::from_int(4));
        let c = mesh.collide_capsule(p, p, half()).unwrap();
        assert_eq!(c.depth, quarter());
        assert_near(c.normal, Vec3Fix::UNIT_Y);
        assert_near(c.point_b, Vec3Fix::from_int(-3, 0, 4));
    }

    #[test]
    fn segment_segment_degenerate_and_parallel_cases() {
        let v = |x: i64, y: i64, z: i64| Vec3Fix::from_int(x, y, z);
        // both segments points
        assert_eq!(
            closest_points_segment_segment(v(1, 2, 3), v(1, 2, 3), v(4, 5, 6), v(4, 5, 6)),
            (v(1, 2, 3), v(4, 5, 6))
        );
        // first a point: its foot on the second, clamped to the end
        assert_eq!(
            closest_points_segment_segment(v(1, 5, 0), v(1, 5, 0), v(0, 0, 0), v(4, 0, 0)),
            (v(1, 5, 0), v(1, 0, 0))
        );
        assert_eq!(
            closest_points_segment_segment(v(9, 5, 0), v(9, 5, 0), v(0, 0, 0), v(4, 0, 0)),
            (v(9, 5, 0), v(4, 0, 0))
        );
        // second a point: its foot on the first
        assert_eq!(
            closest_points_segment_segment(v(0, 0, 0), v(4, 0, 0), v(3, 2, 0), v(3, 2, 0)),
            (v(3, 0, 0), v(3, 2, 0))
        );
        // parallel: s starts at 0, t = -1/2 clamps to 0 and s is re-solved
        // for the second segment's start (distance 1, the parallel gap)
        assert_eq!(
            closest_points_segment_segment(v(0, 0, 0), v(4, 0, 0), v(1, 1, 0), v(3, 1, 0)),
            (v(1, 0, 0), v(1, 1, 0))
        );
        // t below 0 clamps to the second segment's start and s is re-solved
        assert_eq!(
            closest_points_segment_segment(v(0, 0, 0), v(4, 0, 0), v(2, 1, 0), v(2, 3, 0)),
            (v(2, 0, 0), v(2, 1, 0))
        );
    }

    #[test]
    fn segment_crossing_rejects_parallel_and_outside_segments() {
        let tri = Triangle::new(
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::from_int(0, 0, 2),
        );
        let v = |x: i64, y: i64, z: i64| Vec3Fix::from_int(x, y, z);
        // through the interior
        assert_eq!(
            segment_crosses_triangle(v(1, 1, 0), v(1, -1, 0), &tri),
            Some(v(1, 0, 0))
        );
        // parallel to the plane
        assert_eq!(segment_crosses_triangle(v(0, 1, 0), v(2, 1, 0), &tri), None);
        // through the plane beside the triangle on either barycentric side
        assert_eq!(
            segment_crosses_triangle(v(-1, 1, 1), v(-1, -1, 1), &tri),
            None
        );
        assert_eq!(
            segment_crosses_triangle(v(1, 1, -1), v(1, -1, -1), &tri),
            None
        );
        assert_eq!(
            segment_crosses_triangle(v(2, 1, 2), v(2, -1, 2), &tri),
            None
        );
        // short of the plane on either end
        assert_eq!(segment_crosses_triangle(v(1, 3, 0), v(1, 1, 0), &tri), None);
        assert_eq!(
            segment_crosses_triangle(v(1, -1, 0), v(1, -3, 0), &tri),
            None
        );
    }

    #[test]
    fn box_resting_mostly_above_the_floor_is_pushed_up() {
        // centre y = 1/2, half 1: the box reaches 1/2 below the floor
        let mesh = make_ground_mesh();
        let aabb = AABB::new(
            vq(Fix128::NEG_ONE, -half(), Fix128::NEG_ONE),
            vq(Fix128::ONE, Fix128::from_ratio(3, 2), Fix128::ONE),
        );
        let c = mesh.collide_aabb(&aabb).unwrap();
        assert_eq!(c.depth, half());
        assert_eq!(c.normal, Vec3Fix::UNIT_Y);
        assert_eq!(c.point_b, Vec3Fix::ZERO);
        assert_eq!(c.point_a, vq(Fix128::ZERO, -half(), Fix128::ZERO));
    }

    #[test]
    fn box_touching_separated_by_a_slanted_face_or_degenerate_has_no_contact() {
        // resting exactly on the floor: zero depth is not a contact
        let mesh = make_ground_mesh();
        let touching = AABB::new(Vec3Fix::from_int(-1, 0, -1), Vec3Fix::from_int(1, 2, 1));
        assert!(mesh.collide_aabb(&touching).is_none());

        // x + y + z = 2: the boxes overlap but the face normal separates them
        // (the triangle is 2/sqrt3 from the centre, the box reaches 1.5/sqrt3)
        let slanted = TriMesh::from_indexed(
            &[
                Vec3Fix::from_int(2, 0, 0),
                Vec3Fix::from_int(0, 2, 0),
                Vec3Fix::from_int(0, 0, 2),
            ],
            &[0, 1, 2],
        );
        let cube = AABB::new(vq(-half(), -half(), -half()), vq(half(), half(), half()));
        assert!(slanted.bounds.min.x < cube.max.x);
        assert!(slanted.collide_aabb(&cube).is_none());

        // a zero-area triangle through the box has no face to push along
        let sliver = TriMesh::from_indexed(
            &[
                Vec3Fix::from_int(-1, 0, 0),
                Vec3Fix::from_int(0, 0, 0),
                Vec3Fix::from_int(1, 0, 0),
            ],
            &[0, 1, 2],
        );
        assert!(sliver.collide_aabb(&cube).is_none());
    }

    #[test]
    fn ray_hit_with_a_max_t_whose_product_overflows() {
        // |det| = 400 for this floor triangle; max_t * 400 overflows Fix128
        let tri = Triangle::new(
            Vec3Fix::from_int(-10, 0, -10),
            Vec3Fix::from_int(10, 0, -10),
            Vec3Fix::from_int(10, 0, 10),
        );
        let ray = Ray::new(
            Vec3Fix::from_int(1, 5, -1),
            Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO),
        );
        let max_t = Fix128::from_int(1_i64 << 60);
        assert!(max_t.checked_mul(Fix128::from_int(400)).is_none());
        let hit = ray_triangle(&ray, &tri, max_t).unwrap();
        assert_eq!(hit.t, Fix128::from_int(5));
        assert_eq!(hit.point, Vec3Fix::from_int(1, 0, -1));
        assert_eq!(hit.normal, Vec3Fix::UNIT_Y);
    }

    /// The nearest hit by the loop over every triangle (ties: the smallest index)
    fn loop_hit(mesh: &TriMesh, ray: &Ray, max_t: Fix128) -> Option<RayHit> {
        let mut best: Option<RayHit> = None;
        for (i, tri) in mesh.triangles.iter().enumerate() {
            if let Some(mut h) = ray_triangle(ray, tri, max_t) {
                h.body_index = i;
                if best.is_none_or(|b| h.t < b.t) {
                    best = Some(h);
                }
            }
        }
        best
    }

    /// A 12 × 12 grid of floor cells (two triangles each) at heights 0..3, so the BVH has
    /// internal nodes: the walk (nearest hit, and the any-hit `occluded`) equals the loop over
    /// every triangle for rays straight down onto each cell, slanted rays across the grid, rays
    /// beside or away from it, a reach that stops above the floor, and a reach of 2^60 whose
    /// slab products overflow (taken as a meet, so the triangles decide)
    #[test]
    fn the_walk_over_a_deep_tree_equals_the_loop_and_the_any_hit_query_agrees() {
        let mut tris = Vec::new();
        for i in 0..12_i64 {
            for j in 0..12_i64 {
                let y = (i * 3 + j * 5) % 4;
                let (x0, z0) = (i * 2, j * 2);
                let p = |x, z| Vec3Fix::from_int(x, y, z);
                tris.push(Triangle::new(p(x0, z0), p(x0 + 2, z0), p(x0 + 2, z0 + 2)));
                tris.push(Triangle::new(p(x0, z0), p(x0 + 2, z0 + 2), p(x0, z0 + 2)));
            }
        }
        let mesh = TriMesh::from_triangles(tris);
        assert!(
            mesh.bvh.nodes.iter().any(|n| !n.is_leaf()),
            "the tree has no internal node"
        );
        let down = Vec3Fix::new(Fix128::ZERO, Fix128::NEG_ONE, Fix128::ZERO);
        let mut rays = Vec::new();
        for i in 0..12_i64 {
            for j in 0..12_i64 {
                // straight down onto the cell's first triangle, and a slanted ray across cells
                let o = Vec3Fix::new(
                    Fix128::from_ratio(i * 4 + 3, 2),
                    Fix128::from_int(10),
                    Fix128::from_ratio(j * 4 + 1, 2),
                );
                rays.push(Ray::new(o, down));
                let d = Vec3Fix::new(
                    Fix128::from_ratio(i - 6, 7),
                    Fix128::NEG_ONE,
                    Fix128::from_ratio(5 - j, 9),
                );
                rays.push(Ray::new(o, d.normalize()));
            }
        }
        // beside the grid along x, and pointing away from it
        rays.push(Ray::new(Vec3Fix::from_int(-5, 1, 7), Vec3Fix::UNIT_Y));
        rays.push(Ray::new(Vec3Fix::from_int(7, 10, 7), Vec3Fix::UNIT_Y));
        // the fields are public: a direction of length 2^40 (not normalised), where the slab
        // products with a reach of 2^60 overflow
        rays.push(Ray {
            origin: Vec3Fix::from_int(7, 10, 5),
            direction: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-(1_i64 << 40)), Fix128::ZERO),
        });
        let mut hits = 0;
        for ray in &rays {
            for max_t in [
                Fix128::from_int(5),
                Fix128::from_int(100),
                Fix128::from_int(1_i64 << 60),
            ] {
                let want = loop_hit(&mesh, ray, max_t);
                let (got, visits) = mesh.raycast_visits(ray, max_t);
                assert_eq!(got, want, "ray {ray:?} reach {max_t:?}");
                assert_eq!(mesh.raycast(ray, max_t), want);
                assert_eq!(
                    mesh.occluded(ray, max_t),
                    want.is_some(),
                    "ray {ray:?} reach {max_t:?}"
                );
                assert!(visits >= 1);
                hits += usize::from(want.is_some());
            }
        }
        // every downward ray hits within 100, no unit-length ray within 5 of a start 10 above
        // the floor (the last ray's t is in units of 2^40)
        assert!(hits >= 2 * 144, "{hits} hits");
        assert_eq!(
            rays[..rays.len() - 1]
                .iter()
                .filter(|r| loop_hit(&mesh, r, Fix128::from_int(5)).is_some())
                .count(),
            0
        );
    }
}
