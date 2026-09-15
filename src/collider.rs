//! Collision Detection using GJK and EPA
//!
//! Deterministic collision detection using fixed-point math.
//!
//! # Algorithms
//!
//! - **GJK (Gilbert-Johnson-Keerthi)**: Determines if two convex shapes intersect
//! - **EPA (Expanding Polytope Algorithm)**: Computes penetration depth and normal

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Collision Result
// ============================================================================

/// Result of a collision detection query
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollisionResult {
    /// Whether the shapes are colliding
    pub colliding: bool,
    /// Penetration depth (if colliding)
    pub depth: Fix128,
    /// Collision normal (from A to B)
    pub normal: Vec3Fix,
    /// Contact point on shape A
    pub point_a: Vec3Fix,
    /// Contact point on shape B
    pub point_b: Vec3Fix,
}

impl CollisionResult {
    /// No collision
    pub const NONE: Self = Self {
        colliding: false,
        depth: Fix128::ZERO,
        normal: Vec3Fix::ZERO,
        point_a: Vec3Fix::ZERO,
        point_b: Vec3Fix::ZERO,
    };

    /// Create a new collision result
    #[must_use]
    pub const fn new(depth: Fix128, normal: Vec3Fix, point_a: Vec3Fix, point_b: Vec3Fix) -> Self {
        Self {
            colliding: true,
            depth,
            normal,
            point_a,
            point_b,
        }
    }
}

// ============================================================================
// Collider Shapes
// ============================================================================

/// Support function trait for GJK
pub trait Support {
    /// Returns the point on the shape furthest in the given direction
    fn support(&self, direction: Vec3Fix) -> Vec3Fix;
}

/// Sphere collider
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sphere {
    /// Center position
    pub center: Vec3Fix,
    /// Sphere radius
    pub radius: Fix128,
}

impl Sphere {
    /// Create a new sphere from center and radius
    #[must_use]
    pub const fn new(center: Vec3Fix, radius: Fix128) -> Self {
        Self { center, radius }
    }
}

impl Support for Sphere {
    #[inline(always)]
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let dir_norm = direction.normalize();
        self.center + dir_norm * self.radius
    }
}

/// Axis-Aligned Bounding Box
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AABB {
    /// Minimum corner
    pub min: Vec3Fix,
    /// Maximum corner
    pub max: Vec3Fix,
}

impl AABB {
    /// Create a new AABB from min and max corners
    #[must_use]
    pub const fn new(min: Vec3Fix, max: Vec3Fix) -> Self {
        Self { min, max }
    }

    /// Create AABB from center and half-extents
    #[must_use]
    pub fn from_center_half(center: Vec3Fix, half: Vec3Fix) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    /// Check if two AABBs intersect (broad phase)
    #[inline]
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Compute union of two AABBs
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vec3Fix::new(
                if self.min.x < other.min.x {
                    self.min.x
                } else {
                    other.min.x
                },
                if self.min.y < other.min.y {
                    self.min.y
                } else {
                    other.min.y
                },
                if self.min.z < other.min.z {
                    self.min.z
                } else {
                    other.min.z
                },
            ),
            max: Vec3Fix::new(
                if self.max.x > other.max.x {
                    self.max.x
                } else {
                    other.max.x
                },
                if self.max.y > other.max.y {
                    self.max.y
                } else {
                    other.max.y
                },
                if self.max.z > other.max.z {
                    self.max.z
                } else {
                    other.max.z
                },
            ),
        }
    }

    /// Surface area (for BVH heuristics)
    #[must_use]
    pub fn surface_area(&self) -> Fix128 {
        let d = self.max - self.min;
        let two = Fix128::from_int(2);
        two * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

impl Support for AABB {
    #[inline(always)]
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        Vec3Fix::new(
            if direction.x >= Fix128::ZERO {
                self.max.x
            } else {
                self.min.x
            },
            if direction.y >= Fix128::ZERO {
                self.max.y
            } else {
                self.min.y
            },
            if direction.z >= Fix128::ZERO {
                self.max.z
            } else {
                self.min.z
            },
        )
    }
}

/// Convex hull (array of vertices)
#[derive(Clone, Debug)]
pub struct ConvexHull {
    /// Hull vertices
    pub vertices: Vec<Vec3Fix>,
}

impl ConvexHull {
    /// Create a new convex hull from vertices.
    ///
    /// # Panics
    ///
    /// Panics if `vertices` is empty, since `support()` requires at least one vertex.
    #[must_use]
    pub fn new(vertices: Vec<Vec3Fix>) -> Self {
        assert!(
            !vertices.is_empty(),
            "ConvexHull requires at least one vertex"
        );
        Self { vertices }
    }
}

impl Support for ConvexHull {
    #[inline(always)]
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let mut best = self.vertices[0];
        let mut best_dot = best.dot(direction);

        for &v in &self.vertices[1..] {
            let d = v.dot(direction);
            if d > best_dot {
                best = v;
                best_dot = d;
            }
        }

        best
    }
}

/// Capsule (line segment with radius)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capsule {
    /// Start point of the capsule segment
    pub a: Vec3Fix,
    /// End point of the capsule segment
    pub b: Vec3Fix,
    /// Capsule radius
    pub radius: Fix128,
}

impl Capsule {
    /// Create a new capsule from two endpoints and a radius
    #[must_use]
    pub const fn new(a: Vec3Fix, b: Vec3Fix, radius: Fix128) -> Self {
        Self { a, b, radius }
    }
}

impl Support for Capsule {
    #[inline(always)]
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        let da = self.a.dot(direction);
        let db = self.b.dot(direction);
        let base = if da > db { self.a } else { self.b };
        base + direction.normalize() * self.radius
    }
}

/// Uniformly scaled shape wrapper
///
/// Wraps any `Support`-implementing shape with a uniform scale factor.
/// The support function scales the inner shape's support point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaledShape<S> {
    /// Inner shape
    pub shape: S,
    /// Uniform scale factor
    pub scale: Fix128,
}

impl<S> ScaledShape<S> {
    /// Create a new scaled shape
    pub const fn new(shape: S, scale: Fix128) -> Self {
        Self { shape, scale }
    }
}

impl<S: Support> Support for ScaledShape<S> {
    #[inline(always)]
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        self.shape.support(direction) * self.scale
    }
}

// ============================================================================
// GJK Algorithm
// ============================================================================

/// Minkowski difference support function
#[inline(always)]
fn minkowski_support<A: Support, B: Support>(a: &A, b: &B, direction: Vec3Fix) -> Vec3Fix {
    a.support(direction) - b.support(-direction)
}

/// Simplex for GJK (up to 4 points in 3D)
#[derive(Clone, Debug)]
struct Simplex {
    points: [Vec3Fix; 4],
    size: usize,
}

impl Simplex {
    const fn new() -> Self {
        Self {
            points: [Vec3Fix::ZERO; 4],
            size: 0,
        }
    }

    fn push(&mut self, point: Vec3Fix) {
        // Shift existing points
        for i in (1..4).rev() {
            self.points[i] = self.points[i - 1];
        }
        self.points[0] = point;
        self.size = (self.size + 1).min(4);
    }

    fn set(&mut self, points: &[Vec3Fix]) {
        for (i, &p) in points.iter().enumerate().take(4) {
            self.points[i] = p;
        }
        self.size = points.len().min(4);
    }
}

/// GJK collision result
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GjkResult {
    /// Whether the two shapes are colliding
    pub colliding: bool,
    /// Closest point on the Minkowski difference boundary
    pub closest_point: Vec3Fix,
}

/// GJK algorithm for collision detection
///
/// Returns true if the two shapes are intersecting.
/// Deterministic: fixed iteration count.
pub fn gjk<A: Support, B: Support>(a: &A, b: &B) -> GjkResult {
    const MAX_ITERATIONS: usize = 64;

    // Initial direction
    let mut direction = Vec3Fix::UNIT_X;

    // Get initial support point
    let mut simplex = Simplex::new();
    simplex.push(minkowski_support(a, b, direction));

    // Direction toward origin
    direction = -simplex.points[0];

    for _ in 0..MAX_ITERATIONS {
        if direction.length_squared().is_zero() {
            // Origin is on the simplex
            return GjkResult {
                colliding: true,
                closest_point: Vec3Fix::ZERO,
            };
        }

        let new_point = minkowski_support(a, b, direction);

        // Check if we passed the origin
        if new_point.dot(direction) < Fix128::ZERO {
            return GjkResult {
                colliding: false,
                closest_point: new_point,
            };
        }

        simplex.push(new_point);

        // Update simplex and direction
        if do_simplex(&mut simplex, &mut direction) {
            return GjkResult {
                colliding: true,
                closest_point: Vec3Fix::ZERO,
            };
        }
    }

    // Assume no collision if max iterations reached
    GjkResult {
        colliding: false,
        closest_point: simplex.points[0],
    }
}

/// Process simplex and update direction toward origin
fn do_simplex(simplex: &mut Simplex, direction: &mut Vec3Fix) -> bool {
    match simplex.size {
        2 => do_simplex_line(simplex, direction),
        3 => do_simplex_triangle(simplex, direction),
        4 => do_simplex_tetrahedron(simplex, direction),
        _ => false,
    }
}

fn do_simplex_line(simplex: &mut Simplex, direction: &mut Vec3Fix) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let ab = b - a;
    let ao = -a;

    if ab.dot(ao) > Fix128::ZERO {
        // Origin is between A and B
        let mut new_dir = ab.cross(ao).cross(ab);
        // If ab and ao are parallel, the triple cross product is zero.
        // Pick a perpendicular fallback direction.
        if new_dir.length_squared().is_zero() {
            // Choose an axis least aligned with ab for the perpendicular
            let abs_x = ab.x.abs();
            let abs_y = ab.y.abs();
            let abs_z = ab.z.abs();
            let perp = if abs_x <= abs_y && abs_x <= abs_z {
                Vec3Fix::UNIT_X
            } else if abs_y <= abs_z {
                Vec3Fix::UNIT_Y
            } else {
                Vec3Fix::UNIT_Z
            };
            new_dir = ab.cross(perp);
        }
        *direction = new_dir;
    } else {
        // Origin is beyond A
        simplex.set(&[a]);
        *direction = ao;
    }

    false
}

fn do_simplex_triangle(simplex: &mut Simplex, direction: &mut Vec3Fix) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];

    let ab = b - a;
    let ac = c - a;
    let ao = -a;

    let abc = ab.cross(ac);

    if abc.cross(ac).dot(ao) > Fix128::ZERO {
        if ac.dot(ao) > Fix128::ZERO {
            simplex.set(&[a, c]);
            *direction = ac.cross(ao).cross(ac);
        } else {
            simplex.set(&[a, b]);
            return do_simplex_line(simplex, direction);
        }
    } else if ab.cross(abc).dot(ao) > Fix128::ZERO {
        simplex.set(&[a, b]);
        return do_simplex_line(simplex, direction);
    } else if abc.dot(ao) > Fix128::ZERO {
        *direction = abc;
    } else {
        simplex.set(&[a, c, b]);
        *direction = -abc;
    }

    false
}

fn do_simplex_tetrahedron(simplex: &mut Simplex, direction: &mut Vec3Fix) -> bool {
    let a = simplex.points[0];
    let b = simplex.points[1];
    let c = simplex.points[2];
    let d = simplex.points[3];

    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if abc.dot(ao) > Fix128::ZERO {
        simplex.set(&[a, b, c]);
        return do_simplex_triangle(simplex, direction);
    }

    if acd.dot(ao) > Fix128::ZERO {
        simplex.set(&[a, c, d]);
        return do_simplex_triangle(simplex, direction);
    }

    if adb.dot(ao) > Fix128::ZERO {
        simplex.set(&[a, d, b]);
        return do_simplex_triangle(simplex, direction);
    }

    // Origin is inside the tetrahedron
    true
}

// ============================================================================
// EPA Algorithm (Expanding Polytope Algorithm)
// ============================================================================

/// Contact information from EPA
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contact {
    /// Penetration depth
    pub depth: Fix128,
    /// Contact normal (pointing from B to A)
    pub normal: Vec3Fix,
    /// Contact point on A
    pub point_a: Vec3Fix,
    /// Contact point on B
    pub point_b: Vec3Fix,
}

/// EPA face (triangle)
#[derive(Clone, Copy, Debug, PartialEq)]
struct EpaFace {
    indices: [usize; 3],
    normal: Vec3Fix,
    distance: Fix128,
}

/// EPA algorithm for penetration depth
///
/// Call this after GJK returns a collision; `initial_simplex` must be a
/// tetrahedron of Minkowski-difference points that encloses the origin.
/// Deterministic: fixed iteration count.
///
/// The returned [`Contact::normal`] points from B to A (the crate-wide
/// contact contract): translating A by `depth · normal` separates the
/// shapes. `point_a` is the deepest point of A inside B and `point_b` the
/// deepest point of B inside A. Before 1.2.0 the normal was the outward
/// face normal of A⊖B, i.e. A→B, the opposite of the documented sign, and
/// faces through the origin (touching shapes) kept their vertex winding,
/// so a touching pair could report the far side of the polytope.
pub fn epa<A: Support, B: Support>(a: &A, b: &B, initial_simplex: &[Vec3Fix]) -> Option<Contact> {
    const MAX_ITERATIONS: usize = 64;
    const EPSILON: Fix128 = Fix128 {
        hi: 0,
        lo: 0x0001000000000000,
    }; // Small threshold

    if initial_simplex.len() < 4 {
        return None;
    }

    let mut vertices: Vec<Vec3Fix> = initial_simplex.to_vec();
    let mut faces: Vec<EpaFace> = Vec::with_capacity(64);

    // Interior reference point: the centroid of the initial tetrahedron
    // stays inside the polytope as it grows (convex), so every face normal
    // can be oriented away from it — independent of vertex winding and of
    // whether the origin lies exactly on the face.
    let interior =
        (vertices[0] + vertices[1] + vertices[2] + vertices[3]) * Fix128::from_ratio(1, 4);

    // Initialize with tetrahedron faces
    add_face(&mut faces, &vertices, interior, 0, 1, 2);
    add_face(&mut faces, &vertices, interior, 0, 3, 1);
    add_face(&mut faces, &vertices, interior, 0, 2, 3);
    add_face(&mut faces, &vertices, interior, 1, 3, 2);

    for _ in 0..MAX_ITERATIONS {
        // Find face closest to origin
        let (_closest_idx, closest_face) = faces
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.distance.cmp(&b.distance))
            .map(|(i, f)| (i, *f))?;

        // Get support point in face normal direction
        let support = minkowski_support(a, b, closest_face.normal);
        let distance = support.dot(closest_face.normal);

        // Check for convergence
        if distance - closest_face.distance < EPSILON {
            // Found the closest face: its outward normal is the A→B
            // penetration direction, so the B→A contact normal is its negation
            let outward = closest_face.normal;
            let depth = closest_face.distance;

            // Deepest points (simplified): A furthest along A→B, B along B→A
            let point_a = a.support(outward);
            let point_b = b.support(-outward);

            return Some(Contact {
                depth,
                normal: -outward,
                point_a,
                point_b,
            });
        }

        // Add new vertex
        let new_idx = vertices.len();
        vertices.push(support);

        // Remove faces visible from new vertex and add new faces
        let mut edges: Vec<(usize, usize)> = Vec::new();

        faces.retain(|face| {
            let v = vertices[face.indices[0]];
            let to_new = support - v;

            if face.normal.dot(to_new) > Fix128::ZERO {
                // Face is visible, collect edges
                for i in 0..3 {
                    let edge = (face.indices[i], face.indices[(i + 1) % 3]);
                    // Check if edge already exists (shared edge)
                    if let Some(pos) = edges.iter().position(|&e| e == (edge.1, edge.0)) {
                        edges.remove(pos);
                    } else {
                        edges.push(edge);
                    }
                }
                false
            } else {
                true
            }
        });

        // Add new faces from edges
        for (i, j) in edges {
            add_face(&mut faces, &vertices, interior, i, j, new_idx);
        }
    }

    None
}

fn add_face(
    faces: &mut Vec<EpaFace>,
    vertices: &[Vec3Fix],
    interior: Vec3Fix,
    i: usize,
    j: usize,
    k: usize,
) {
    let a = vertices[i];
    let b = vertices[j];
    let c = vertices[k];

    let ab = b - a;
    let ac = c - a;
    let cross = ab.cross(ac);

    // Skip degenerate faces where vertices are collinear or coincident
    if cross.length_squared().is_zero() {
        return;
    }

    let normal = cross.normalize();

    // Orient the normal away from the polytope interior (not away from the
    // origin: a face through the origin has distance 0 either way and the
    // origin gives no orientation)
    let normal = if (a - interior).dot(normal) < Fix128::ZERO {
        -normal
    } else {
        normal
    };
    let distance = a.dot(normal);

    faces.push(EpaFace {
        indices: [i, j, k],
        normal,
        distance,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_intersection() {
        let a = AABB::new(Vec3Fix::from_int(0, 0, 0), Vec3Fix::from_int(2, 2, 2));
        let b = AABB::new(Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(3, 3, 3));
        let c = AABB::new(Vec3Fix::from_int(5, 5, 5), Vec3Fix::from_int(6, 6, 6));

        assert!(a.intersects(&b), "a and b should intersect");
        assert!(!a.intersects(&c), "a and c should not intersect");
    }

    #[test]
    fn test_sphere_support() {
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);

        let support = sphere.support(Vec3Fix::UNIT_X);
        assert_eq!(support.x.hi, 1);
        assert!(support.y.is_zero());
        assert!(support.z.is_zero());
    }

    #[test]
    fn test_gjk_spheres_colliding() {
        let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let b = Sphere::new(
            Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );

        let result = gjk(&a, &b);
        assert!(result.colliding, "Overlapping spheres should collide");
    }

    #[test]
    fn test_gjk_spheres_not_colliding() {
        let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let b = Sphere::new(
            Vec3Fix::new(Fix128::from_int(5), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );

        let result = gjk(&a, &b);
        assert!(!result.colliding, "Separated spheres should not collide");
    }

    #[test]
    fn test_aabb_support() {
        let aabb = AABB::new(Vec3Fix::from_int(-1, -1, -1), Vec3Fix::from_int(1, 1, 1));

        let support = aabb.support(Vec3Fix::UNIT_X);
        assert_eq!(support.x.hi, 1);
    }

    #[test]
    fn test_scaled_shape() {
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let scaled = ScaledShape::new(sphere, Fix128::from_int(3));

        let support = scaled.support(Vec3Fix::UNIT_X);
        assert_eq!(support.x.hi, 3, "Scaled sphere support should be at 3");
    }

    #[test]
    fn test_scaled_collision() {
        let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let b = Sphere::new(Vec3Fix::from_int(4, 0, 0), Fix128::ONE);

        // Unscaled: distance=4, combined radius=2 → no collision
        let result = gjk(&a, &b);
        assert!(!result.colliding, "Unscaled spheres should not collide");

        // Scaled: scale=3, effective radius=3, combined=3+1=4 → overlap at boundary
        let scaled_a = ScaledShape::new(a, Fix128::from_int(3));
        let result = gjk(&scaled_a, &b);
        assert!(result.colliding, "Scaled sphere should collide");
    }

    // ---- mutation-score tests (2026-09-15) ----------------------------

    fn fi(n: i64) -> Fix128 {
        Fix128::from_int(n)
    }

    fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    fn near(a: Fix128, b: Fix128) -> bool {
        (a - b).abs() < Fix128 { hi: 0, lo: 1 << 24 }
    }

    #[test]
    fn aabb_from_center_half_union_and_surface_area_exact() {
        let a = AABB::from_center_half(v3i(1, 2, 3), v3i(1, 2, 3));
        assert_eq!(a.min, v3i(0, 0, 0));
        assert_eq!(a.max, v3i(2, 4, 6));
        // 2 (xy + yz + zx) = 2 (8 + 24 + 12) = 88
        assert_eq!(a.surface_area(), fi(88));
        let b = AABB::new(v3i(-5, 1, 2), v3i(1, 3, 9));
        // union: min は成分ごとの小さい方、max は大きい方 (各軸で self / other が勝つ組合せを混ぜる)
        let u = a.union(&b);
        assert_eq!(u.min, v3i(-5, 0, 0));
        assert_eq!(u.max, v3i(2, 4, 9));
        assert_eq!(b.union(&a), u, "対称");
        assert_eq!(a.union(&a), a);
        // surface_area の 3 項が独立に効く: 各軸を伸ばすと増分が閉形式
        let base = AABB::new(Vec3Fix::ZERO, v3i(1, 2, 3)); // 2(2+6+3) = 22
        assert_eq!(base.surface_area(), fi(22));
        assert_eq!(
            AABB::new(Vec3Fix::ZERO, v3i(2, 2, 3)).surface_area(),
            fi(32)
        ); // 2(4+6+6)
        assert_eq!(
            AABB::new(Vec3Fix::ZERO, v3i(1, 3, 3)).surface_area(),
            fi(30)
        ); // 2(3+9+3)
        assert_eq!(
            AABB::new(Vec3Fix::ZERO, v3i(1, 2, 4)).surface_area(),
            fi(28)
        ); // 2(2+8+4)
        assert_eq!(
            AABB::new(v3i(1, 1, 1), v3i(1, 1, 1)).surface_area(),
            Fix128::ZERO
        );
    }

    #[test]
    fn aabb_intersects_boundary_semantics() {
        let a = AABB::new(Vec3Fix::ZERO, v3i(2, 2, 2));
        assert!(a.intersects(&AABB::new(v3i(1, 1, 1), v3i(3, 3, 3))));
        assert!(!a.intersects(&AABB::new(v3i(3, 0, 0), v3i(4, 2, 2))));
        // 各軸単独で離れている
        assert!(!a.intersects(&AABB::new(v3i(0, 3, 0), v3i(2, 4, 2))));
        assert!(!a.intersects(&AABB::new(v3i(0, 0, -5), v3i(2, 2, -3))));
        let touching = a.intersects(&AABB::new(v3i(2, 0, 0), v3i(3, 2, 2)));
        // 接触ちょうどの真偽は実装依存だが、少し重なれば必ず true、少し離れれば必ず false
        assert!(a.intersects(&AABB::new(
            Vec3Fix::new(fi(2) - Fix128::from_ratio(1, 8), Fix128::ZERO, Fix128::ZERO),
            v3i(3, 2, 2)
        )));
        assert!(!a.intersects(&AABB::new(
            Vec3Fix::new(fi(2) + Fix128::from_ratio(1, 8), Fix128::ZERO, Fix128::ZERO),
            v3i(3, 2, 2)
        )));
        let _ = touching;
    }

    #[test]
    fn support_functions_return_extreme_points() {
        let aabb = AABB::new(v3i(-1, -2, -3), v3i(4, 5, 6));
        assert_eq!(aabb.support(v3i(1, 1, 1)), v3i(4, 5, 6));
        assert_eq!(aabb.support(v3i(-1, -1, -1)), v3i(-1, -2, -3));
        assert_eq!(aabb.support(v3i(1, -1, 1)), v3i(4, -2, 6));
        assert_eq!(aabb.support(v3i(-1, 1, -1)), v3i(-1, 5, -3));
        // 成分 0 は `>=` で max 側
        assert_eq!(aabb.support(Vec3Fix::ZERO), v3i(4, 5, 6));
        let sphere = Sphere::new(v3i(1, 1, 1), fi(2));
        assert_eq!(sphere.support(v3i(0, 5, 0)), v3i(1, 3, 1));
        assert_eq!(sphere.support(v3i(-3, 0, 0)), v3i(-1, 1, 1));
        let capsule = Capsule::new(v3i(0, -2, 0), v3i(0, 2, 0), Fix128::ONE);
        assert_eq!(capsule.support(v3i(0, 1, 0)), v3i(0, 3, 0));
        assert_eq!(capsule.support(v3i(0, -1, 0)), v3i(0, -3, 0));
        assert_eq!(
            capsule.support(v3i(1, 0, 0)),
            v3i(1, 2, 0),
            "da == db は b 側 (`>` false)"
        );
        assert_eq!(
            capsule.support(v3i(4, 3, 0)),
            Vec3Fix::new(
                Fix128::from_ratio(4, 5),
                fi(2) + Fix128::from_ratio(3, 5),
                Fix128::ZERO
            )
        );
        let hull = ConvexHull::new(vec![v3i(0, 0, 0), v3i(3, 0, 0), v3i(0, 4, 0), v3i(0, 0, 5)]);
        assert_eq!(hull.support(v3i(1, 0, 0)), v3i(3, 0, 0));
        assert_eq!(hull.support(v3i(0, 1, 0)), v3i(0, 4, 0));
        assert_eq!(hull.support(v3i(0, 0, 1)), v3i(0, 0, 5));
        assert_eq!(hull.support(v3i(-1, -1, -1)), v3i(0, 0, 0));
        // 同点 (dot が等しい) は最初の頂点 (`>` false)
        assert_eq!(hull.support(v3i(0, 0, -1)), v3i(0, 0, 0));
    }

    #[test]
    fn gjk_box_box_and_sphere_box_configurations() {
        let a = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        // 重なり (各軸)、接触前後、完全包含、離れている
        assert!(
            gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(Fix128::from_ratio(1, 2), fi(-1), fi(-1)),
                    v3i(3, 1, 1)
                )
            )
            .colliding
        );
        assert!(
            gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(fi(-1), Fix128::from_ratio(1, 2), fi(-1)),
                    v3i(1, 3, 1)
                )
            )
            .colliding
        );
        assert!(
            gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(fi(-1), fi(-1), Fix128::from_ratio(1, 2)),
                    v3i(1, 1, 3)
                )
            )
            .colliding
        );
        assert!(!gjk(&a, &AABB::new(v3i(2, -1, -1), v3i(4, 1, 1))).colliding);
        assert!(!gjk(&a, &AABB::new(v3i(-1, 2, -1), v3i(1, 4, 1))).colliding);
        assert!(!gjk(&a, &AABB::new(v3i(-1, -1, -4), v3i(1, 1, -2))).colliding);
        assert!(
            gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(
                        Fix128::from_ratio(-1, 4),
                        Fix128::from_ratio(-1, 4),
                        Fix128::from_ratio(-1, 4)
                    ),
                    Vec3Fix::new(
                        Fix128::from_ratio(1, 4),
                        Fix128::from_ratio(1, 4),
                        Fix128::from_ratio(1, 4)
                    )
                )
            )
            .colliding,
            "包含"
        );
        // 対角方向にずれた箱 (line / triangle / tetrahedron 全経路)
        assert!(
            gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(
                        Fix128::from_ratio(1, 2),
                        Fix128::from_ratio(1, 2),
                        Fix128::from_ratio(1, 2)
                    ),
                    v3i(3, 3, 3)
                )
            )
            .colliding
        );
        assert!(
            !gjk(
                &a,
                &AABB::new(
                    Vec3Fix::new(
                        Fix128::from_ratio(3, 2),
                        Fix128::from_ratio(3, 2),
                        Fix128::from_ratio(3, 2)
                    ),
                    v3i(3, 3, 3)
                )
            )
            .colliding
        );
        // 球 vs 箱: 角の外側 (距離 > r) と内側
        let corner_out = Sphere::new(v3i(2, 2, 2), Fix128::ONE); // 角 (1,1,1) まで √3 ≈ 1.73 > 1
        assert!(!gjk(&a, &corner_out).colliding);
        let corner_in = Sphere::new(v3i(2, 2, 2), fi(2));
        assert!(gjk(&a, &corner_in).colliding);
        let face = Sphere::new(
            Vec3Fix::new(fi(2) - Fix128::from_ratio(1, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        assert!(gjk(&a, &face).colliding);
        let face_out = Sphere::new(
            Vec3Fix::new(fi(2) + Fix128::from_ratio(1, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        assert!(!gjk(&a, &face_out).colliding);
        // capsule vs 箱
        let cap = Capsule::new(v3i(-3, 0, 0), v3i(3, 0, 0), Fix128::from_ratio(1, 2));
        assert!(gjk(&a, &cap).colliding);
        let cap_far = Capsule::new(v3i(-3, 3, 0), v3i(3, 3, 0), Fix128::from_ratio(1, 2));
        assert!(!gjk(&a, &cap_far).colliding);
        // 対称
        assert_eq!(gjk(&a, &corner_in).colliding, gjk(&corner_in, &a).colliding);
    }

    /// 2 shape の Minkowski 差の 4 方向 support から EPA 用初期四面体を作る
    fn tetra<A: Support, B: Support>(a: &A, b: &B) -> [Vec3Fix; 4] {
        let dirs = [v3i(1, 1, 1), v3i(-1, -1, 1), v3i(-1, 1, -1), v3i(1, -1, -1)];
        let mut out = [Vec3Fix::ZERO; 4];
        for (i, d) in dirs.iter().enumerate() {
            out[i] = a.support(*d) - b.support(-*d);
        }
        out
    }

    #[test]
    fn epa_reports_minimum_penetration_axis_and_depth() {
        let a = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        // B が +x 側に 0.5 だけ重なる (y/z は 2 重なる) → 最小侵入 0.5、法線 ±x
        let b = AABB::new(
            Vec3Fix::new(Fix128::from_ratio(1, 2), fi(-1), fi(-1)),
            v3i(3, 1, 1),
        );
        let c = epa(&a, &b, &tetra(&a, &b)).expect("overlapping boxes");
        assert!(
            near(c.depth, Fix128::from_ratio(1, 2)),
            "depth {:?}",
            c.depth
        );
        assert!(
            near(c.normal.x.abs(), Fix128::ONE)
                && near(c.normal.y, Fix128::ZERO)
                && near(c.normal.z, Fix128::ZERO),
            "{:?}",
            c.normal
        );
        // +y 側に 0.25 重なる → depth 0.25、法線 ±y
        let by = AABB::new(
            Vec3Fix::new(fi(-1), Fix128::from_ratio(3, 4), fi(-1)),
            v3i(1, 3, 1),
        );
        let cy = epa(&a, &by, &tetra(&a, &by)).expect("overlapping boxes");
        assert!(
            near(cy.depth, Fix128::from_ratio(1, 4)),
            "depth {:?}",
            cy.depth
        );
        assert!(
            near(cy.normal.y.abs(), Fix128::ONE) && near(cy.normal.x, Fix128::ZERO),
            "{:?}",
            cy.normal
        );
        // -z 側に 0.125 重なる
        let bz = AABB::new(
            v3i(-1, -1, -3),
            Vec3Fix::new(fi(1), fi(1), fi(-1) + Fix128::from_ratio(1, 8)),
        );
        let cz = epa(&a, &bz, &tetra(&a, &bz)).expect("overlapping boxes");
        assert!(
            near(cz.depth, Fix128::from_ratio(1, 8)),
            "depth {:?}",
            cz.depth
        );
        assert!(near(cz.normal.z.abs(), Fix128::ONE), "{:?}", cz.normal);
        // 法線は単位長、point_a と point_b は法線方向に depth 離れている
        for cc in [c, cy, cz] {
            assert!(near(cc.normal.length(), Fix128::ONE));
            let sep = (cc.point_a - cc.point_b).dot(cc.normal).abs();
            assert!(near(sep, cc.depth), "sep {sep:?} vs depth {:?}", cc.depth);
        }
        // 四面体未満は None
        assert!(epa(&a, &b, &tetra(&a, &b)[..3]).is_none());
        assert!(epa(&a, &b, &[]).is_none());
    }

    // ---- mutation-score tests, round 2 (2026-09-15) -------------------
    //
    // Each test below names the cargo-mutants mutant it kills (file line of
    // the mutated expression) and derives the expected value by hand in a
    // comment. The GJK simplex helpers are private, so they are driven
    // directly with hand-built simplices; every coordinate is a small
    // integer (or a dyadic fraction), so cross / dot products are exact in
    // Fix128 and the assertions use exact equality unless stated otherwise.

    /// Build a simplex whose `points[i]` is `points[i]` (push in reverse).
    fn simplex_of(points: &[Vec3Fix]) -> Simplex {
        let mut s = Simplex::new();
        for &p in points.iter().rev() {
            s.push(p);
        }
        assert_eq!(s.size, points.len());
        s
    }

    fn near_tol(a: Fix128, b: Fix128, tol: Fix128) -> bool {
        (a - b).abs() < tol
    }

    /// Origin strictly inside a tetrahedron: every sub-tetrahedron obtained by
    /// replacing one vertex with the origin has the same signed volume sign.
    fn tetra_contains_origin(t: &[Vec3Fix; 4]) -> bool {
        fn vol(a: Vec3Fix, b: Vec3Fix, c: Vec3Fix, d: Vec3Fix) -> Fix128 {
            (b - a).cross(c - a).dot(d - a)
        }
        let o = Vec3Fix::ZERO;
        let v0 = vol(t[0], t[1], t[2], t[3]);
        let parts = [
            vol(o, t[1], t[2], t[3]),
            vol(t[0], o, t[2], t[3]),
            vol(t[0], t[1], o, t[3]),
            vol(t[0], t[1], t[2], o),
        ];
        !v0.is_zero()
            && parts
                .iter()
                .all(|v| !v.is_zero() && v.is_negative() == v0.is_negative())
    }

    /// Kills `Simplex::set` → `()` (line 333): `set` must overwrite both the
    /// points and the size.
    #[test]
    fn simplex_set_overwrites_points_and_size() {
        let mut s = simplex_of(&[v3i(1, 1, 1), v3i(2, 2, 2), v3i(3, 3, 3)]);
        assert_eq!(s.size, 3);
        s.set(&[v3i(7, 8, 9), v3i(-1, -2, -3)]);
        assert_eq!(s.size, 2);
        assert_eq!(s.points[0], v3i(7, 8, 9));
        assert_eq!(s.points[1], v3i(-1, -2, -3));
        s.set(&[v3i(5, 5, 5)]);
        assert_eq!(s.size, 1);
        assert_eq!(s.points[0], v3i(5, 5, 5));
        // More than 4 points are truncated to 4.
        s.set(&[v3i(1, 0, 0); 6]);
        assert_eq!(s.size, 4);
    }

    /// Kills `direction = -simplex.points[0]` → `simplex.points[0]` (line 364).
    ///
    /// A = sphere r=1 at origin, B = sphere r=1 at (5,0,0). A ⊖ B is a sphere
    /// of radius 2 centred at (-5,0,0); its closest point to the origin is
    /// (-3,0,0). Trace: first support (dir +x) = (1,0,0) - (4,0,0) = (-3,0,0);
    /// direction = (3,0,0); support(+x) = (-3,0,0) again, dot = -9 < 0 → not
    /// colliding with `closest_point = (-3,0,0)` (all steps exact: the only
    /// normalisation is of an axis-aligned vector). With the mutant the second
    /// direction is (-3,0,0), the walk goes through (-7,0,0) and the parallel
    /// fallback, and the returned point acquires a z component.
    #[test]
    fn gjk_closest_point_for_separated_spheres_is_exact() {
        let a = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let b = Sphere::new(v3i(5, 0, 0), Fix128::ONE);
        let r = gjk(&a, &b);
        assert!(!r.colliding);
        assert_eq!(r.closest_point, v3i(-3, 0, 0));
    }

    /// Kills `new_point.dot(direction) < 0` → `<= 0` (line 378).
    ///
    /// Shapes that exactly touch put the origin on the boundary of A ⊖ B, so
    /// the support in the search direction has dot exactly 0. `AABB::intersects`
    /// treats touching as intersecting (`<=`), and GJK must agree.
    ///
    /// Box A = [-1,1]^3, sphere B centre (-2,1,1) r=1 touches A at the corner
    /// (-1,1,1): first support (dir +x) = (1,1,1) - (-3,1,1) = (4,0,0);
    /// direction = (-4,0,0); support = (-1,1,1) - (-1,1,1) = (0,0,0), dot = 0.
    /// Original: push, line case `ab·ao = 0` keeps only a = origin, direction
    /// = 0 → colliding. Mutant returns `colliding: false` at the dot == 0 step.
    /// Box B = [-3,-1]^3 touches A at the corner (-1,-1,-1) the same way:
    /// first support (4,2,2), second (-1,-1,-1) - (-1,-1,-1) = 0.
    #[test]
    fn gjk_touching_shapes_report_collision() {
        let a = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        let sphere = Sphere::new(v3i(-2, 1, 1), Fix128::ONE);
        assert!(gjk(&a, &sphere).colliding, "sphere touching box corner");
        let corner = AABB::new(v3i(-3, -3, -3), v3i(-1, -1, -1));
        assert!(gjk(&a, &corner).colliding, "box touching box corner");
    }

    /// Kills `ab.dot(ao) > 0` → `>= 0` (line 419).
    ///
    /// a = (2,0,0), b = (2,3,0): ab = (0,3,0), ao = (-2,0,0), ab·ao = 0 exactly.
    /// The documented side of the boundary is "origin beyond A": the simplex
    /// shrinks to [a] and the direction becomes ao = (-2,0,0). The mutant keeps
    /// both points and sets direction (ab × ao) × ab = (-18,0,0).
    #[test]
    fn simplex_line_origin_perpendicular_to_ab_keeps_only_a() {
        let a = v3i(2, 0, 0);
        let b = v3i(2, 3, 0);
        let mut s = simplex_of(&[a, b]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_line(&mut s, &mut dir));
        assert_eq!(s.size, 1);
        assert_eq!(s.points[0], a);
        assert_eq!(dir, v3i(-2, 0, 0));
    }

    /// Kills the parallel-fallback axis selection mutants (line 429:
    /// `&&` → `||`, `abs_x <= abs_y` → `>`, `abs_x <= abs_z` → `>`; line 431:
    /// `abs_y <= abs_z` → `>`).
    ///
    /// With a = -2·ab and b = -ab the origin lies on the line through a and b
    /// (ab ∥ ao, ab·ao = 2|ab|² > 0), so (ab × ao) × ab = 0 and the fallback
    /// picks the axis least aligned with ab; new_dir = ab × axis.
    ///   ab = (1,2,3): x smallest → UNIT_X → (1,2,3) × (1,0,0) = (0,3,-2)
    ///     (mutants `abs_x > abs_y` / `abs_x > abs_z` fall through to UNIT_Y
    ///     → (-3,0,1))
    ///   ab = (2,3,1): z smallest → UNIT_Z → (2,3,1) × (0,0,1) = (3,-2,0)
    ///     (`||` picks UNIT_X → (0,1,-3); `abs_y > abs_z` picks UNIT_Y →
    ///     (-1,0,2))
    ///   ab = (3,1,2): y smallest → UNIT_Y → (3,1,2) × (0,1,0) = (-2,0,3)
    ///     (`abs_y > abs_z` picks UNIT_Z → (1,-3,0))
    #[test]
    fn simplex_line_parallel_fallback_picks_least_aligned_axis() {
        let cases = [
            (v3i(1, 2, 3), v3i(0, 3, -2)),
            (v3i(2, 3, 1), v3i(3, -2, 0)),
            (v3i(3, 1, 2), v3i(-2, 0, 3)),
        ];
        for (ab, expected) in cases {
            let a = -ab - ab;
            let b = -ab;
            assert!(ab.cross(-a).length_squared().is_zero(), "ab ∥ ao");
            let mut s = simplex_of(&[a, b]);
            let mut dir = Vec3Fix::ZERO;
            assert!(!do_simplex_line(&mut s, &mut dir));
            assert_eq!(s.size, 2, "ab = {ab:?}");
            assert_eq!(s.points[0], a);
            assert_eq!(s.points[1], b);
            assert_eq!(dir, expected, "ab = {ab:?}");
        }
    }

    /// Kills `ac = c - a` → `c + a` (line 454) and the `>` → `==` / `<`
    /// mutants of lines 459 and 460 (origin in the Voronoi region of edge AC).
    ///
    /// a = (1,-2,2), b = (4,-2,2), c = (1,3,2): ab = (3,0,0), ac = (0,5,0),
    /// ao = (-1,2,-2), abc = ab × ac = (0,0,15).
    /// (abc × ac)·ao = (-75,0,0)·ao = 75 > 0 and ac·ao = 10 > 0 → simplex
    /// becomes [a, c] and direction = (ac × ao) × ac = (-10,0,5) × (0,5,0)
    /// = (-25,0,-50) (∝ (-1,0,-2): from the edge at x=1, z=2 toward the origin).
    #[test]
    fn simplex_triangle_edge_ac_region() {
        let a = v3i(1, -2, 2);
        let b = v3i(4, -2, 2);
        let c = v3i(1, 3, 2);
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 2);
        assert_eq!(s.points[0], a);
        assert_eq!(s.points[1], c);
        assert_eq!(dir, v3i(-25, 0, -50));
    }

    /// Kills the `>` → `>=` mutants of `do_simplex_triangle` (lines 459, 460,
    /// 467, 470) by putting the origin exactly on each decision plane.
    #[test]
    fn simplex_triangle_boundary_cases() {
        // Line 459: (abc × ac)·ao == 0. a = (0,-2,2), b = (3,-2,2), c = (0,3,2):
        // ab = (3,0,0), ac = (0,5,0), ao = (0,2,-2), abc = (0,0,15),
        // (abc × ac)·ao = (-75,0,0)·ao = 0; (ab × abc)·ao = (0,-45,0)·ao = -90;
        // abc·ao = -30 → last branch: simplex [a, c, b], direction -abc.
        // (`>=` enters the first branch and shrinks the simplex to [a, c].)
        let (a, b, c) = (v3i(0, -2, 2), v3i(3, -2, 2), v3i(0, 3, 2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, c, b]);
        assert_eq!(dir, v3i(0, 0, -15));

        // Line 460: ac·ao == 0 with (abc × ac)·ao > 0. a = (1,0,2), b = (4,0,2),
        // c = (1,5,2): ab = (3,0,0), ac = (0,5,0), ao = (-1,0,-2),
        // (abc × ac)·ao = (-75,0,0)·ao = 75 > 0, ac·ao = 0 → simplex [a, b]
        // then line case: ab·ao = -3 ≤ 0 → [a], direction ao = (-1,0,-2).
        // (`>=` keeps [a, c] with size 2.)
        let (a, b, c) = (v3i(1, 0, 2), v3i(4, 0, 2), v3i(1, 5, 2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 1);
        assert_eq!(s.points[0], a);
        assert_eq!(dir, v3i(-1, 0, -2));

        // Line 467: (ab × abc)·ao == 0. a = (-2,0,2), b = (3,0,2), c = (-2,4,2):
        // ab = (5,0,0), ac = (0,4,0), ao = (2,0,-2), abc = (0,0,20),
        // (abc × ac)·ao = (-80,0,0)·ao = -160; (ab × abc)·ao = (0,-100,0)·ao = 0;
        // abc·ao = -40 → [a, c, b], direction (0,0,-20).
        // (`>=` goes to the line case with [a, b] and direction (0,0,-50).)
        let (a, b, c) = (v3i(-2, 0, 2), v3i(3, 0, 2), v3i(-2, 4, 2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, c, b]);
        assert_eq!(dir, v3i(0, 0, -20));

        // Line 470: abc·ao == 0 (origin in the triangle plane, inside the
        // triangle). a = (-2,-2,0), b = (4,-2,0), c = (-2,4,0): ab = (6,0,0),
        // ac = (0,6,0), ao = (2,2,0), abc = (0,0,36), (abc × ac)·ao = -432,
        // (ab × abc)·ao = -432, abc·ao = 0 → [a, c, b], direction (0,0,-36).
        // (`>=` keeps [a, b, c] and sets direction (0,0,36).)
        let (a, b, c) = (v3i(-2, -2, 0), v3i(4, -2, 0), v3i(-2, 4, 0));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, c, b]);
        assert_eq!(dir, v3i(0, 0, -36));
    }

    /// Kills line 467 `>` → `==`, line 470 `>` → `==` / `<`, and line 474
    /// `-abc` → `abc`.
    #[test]
    fn simplex_triangle_edge_ab_region_and_above_below_plane() {
        // Edge AB Voronoi region. a = (-2,1,2), b = (3,1,2), c = (-2,5,2):
        // ab = (5,0,0), ac = (0,4,0), ao = (2,-1,-2), abc = (0,0,20),
        // (abc × ac)·ao = (-80,0,0)·ao = -160; (ab × abc)·ao = (0,-100,0)·ao
        // = 100 > 0 → [a, b] then line case: ab·ao = 10 > 0,
        // (ab × ao) × ab = (0,10,-5) × (5,0,0) = (0,-25,-50).
        // (`==` falls through to the last branch: size 3, direction (0,0,-20).)
        let (a, b, c) = (v3i(-2, 1, 2), v3i(3, 1, 2), v3i(-2, 5, 2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 2);
        assert_eq!([s.points[0], s.points[1]], [a, b]);
        assert_eq!(dir, v3i(0, -25, -50));

        // Origin above the triangle (abc·ao > 0). a = (-2,-2,-2), b = (4,-2,-2),
        // c = (-2,4,-2): abc = (6,0,0) × (0,6,0) = (0,0,36), ao = (2,2,2),
        // (abc × ac)·ao = (-216,0,0)·ao = -432, (ab × abc)·ao = (0,-216,0)·ao
        // = -432, abc·ao = 72 > 0 → simplex unchanged, direction abc.
        // (`==` / `<` take the last branch: [a, c, b], direction (0,0,-36).)
        let (a, b, c) = (v3i(-2, -2, -2), v3i(4, -2, -2), v3i(-2, 4, -2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, b, c]);
        assert_eq!(dir, v3i(0, 0, 36));

        // Origin below the triangle (abc·ao < 0): same triangle lifted to z = 2,
        // ao = (2,2,-2), abc·ao = -72 → winding flipped to [a, c, b] and
        // direction = -abc = (0,0,-36). (Deleting the `-` yields (0,0,36).)
        let (a, b, c) = (v3i(-2, -2, 2), v3i(4, -2, 2), v3i(-2, 4, 2));
        let mut s = simplex_of(&[a, b, c]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_triangle(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, c, b]);
        assert_eq!(dir, v3i(0, 0, -36));
    }

    /// Kills `do_simplex_tetrahedron` → `true` (line 481) and the `>` → `==` /
    /// `>=` mutants of lines 495 (face abc) and 505 (face adb).
    ///
    /// Base tetrahedron (apex a above a base triangle b, c, d, wound so that
    /// ab × ac, ac × ad, ad × ab all point outward):
    ///   a = (0,0,4), b = (-3,-3,0), c = (3,-3,0), d = (0,3,0)
    ///   abc = (0,-24,18), acd = (24,12,9), adb = (-24,12,9)
    /// The tetrahedron is translated so that the origin lands in the wanted
    /// region; translation does not change the face normals.
    #[test]
    fn simplex_tetrahedron_face_regions_and_containment() {
        let base = [v3i(0, 0, 4), v3i(-3, -3, 0), v3i(3, -3, 0), v3i(0, 3, 0)];
        let shifted = |t: Vec3Fix| [base[0] + t, base[1] + t, base[2] + t, base[3] + t];

        // Origin outside face abc: shift by (0,4,0) → a = (0,4,4), ao = (0,-4,-4),
        // abc·ao = 96 - 72 = 24 > 0 → [a, b, c] handed to the triangle case,
        // which (both edge tests negative, abc·ao = 24 > 0) keeps the triangle
        // and sets direction abc = (0,-24,18). Return value false.
        // (`→ true` returns true; `==` skips to acd·ao = -84, adb·ao = -84 and
        // returns true.)
        let [a, b, c, d] = shifted(v3i(0, 4, 0));
        let mut s = simplex_of(&[a, b, c, d]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_tetrahedron(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, b, c]);
        assert_eq!(dir, v3i(0, -24, 18));

        // Origin exactly on face abc: shift by (0,3,0) → a = (0,3,4),
        // ao = (0,-3,-4), abc·ao = 72 - 72 = 0, acd·ao = -72, adb·ao = -72
        // → origin counts as inside (true, simplex untouched).
        // (`>=` hands [a, b, c] to the triangle case and returns false.)
        let [a, b, c, d] = shifted(v3i(0, 3, 0));
        let mut s = simplex_of(&[a, b, c, d]);
        let mut dir = v3i(9, 9, 9);
        assert!(do_simplex_tetrahedron(&mut s, &mut dir));
        assert_eq!(s.size, 4);
        assert_eq!(dir, v3i(9, 9, 9));

        // Origin outside face adb: shift by (3,0,0) → a = (3,0,4), ao = (-3,0,-4),
        // abc·ao = -72, acd·ao = -72 - 36 = -108, adb·ao = 72 - 36 = 36 > 0
        // → [a, d, b] handed to the triangle case: with b' = d, c' = b the
        // triangle normal is ab' × ac' = adb = (-24,12,9); (abc' × ac')·ao =
        // (-21,-123,108)·ao = -369, (ab' × abc')·ao = (75,96,72)·ao = -513,
        // abc'·ao = 36 > 0 → direction (-24,12,9), size 3. Return false.
        let [a, b, c, d] = shifted(v3i(3, 0, 0));
        let mut s = simplex_of(&[a, b, c, d]);
        let mut dir = Vec3Fix::ZERO;
        assert!(!do_simplex_tetrahedron(&mut s, &mut dir));
        assert_eq!(s.size, 3);
        assert_eq!([s.points[0], s.points[1], s.points[2]], [a, d, b]);
        assert_eq!(dir, v3i(-24, 12, 9));

        // Origin exactly on face adb (scaled ×2 to keep integers): a = (3,0,8),
        // b = (-3,-6,0), c = (9,-6,0), d = (3,6,0): abc = (0,-96,72),
        // acd = (96,48,36), adb = (-96,48,36), ao = (-3,0,-8):
        // abc·ao = -576, acd·ao = -576, adb·ao = 288 - 288 = 0 → true.
        // (`>=` hands [a, d, b] to the triangle case and returns false.)
        let (a, b, c, d) = (v3i(3, 0, 8), v3i(-3, -6, 0), v3i(9, -6, 0), v3i(3, 6, 0));
        let mut s = simplex_of(&[a, b, c, d]);
        let mut dir = v3i(9, 9, 9);
        assert!(do_simplex_tetrahedron(&mut s, &mut dir));
        assert_eq!(s.size, 4);
        assert_eq!(dir, v3i(9, 9, 9));

        // Origin strictly inside: shift by (0,0,-1) → ao = (0,0,-3),
        // abc·ao = -54, acd·ao = -27, adb·ao = -27 → true.
        let [a, b, c, d] = shifted(v3i(0, 0, -1));
        let mut s = simplex_of(&[a, b, c, d]);
        let mut dir = Vec3Fix::ZERO;
        assert!(do_simplex_tetrahedron(&mut s, &mut dir));
    }

    /// Kills `distance - closest_face.distance < EPSILON` → `<=` (line 576).
    ///
    /// A ⊖ B is the hull itself (B is the single point at the origin). The
    /// initial tetrahedron has its closest face in the plane x = 1 (distance
    /// exactly 1, normal exactly (1,0,0): cross = (64,0,0), |cross| = 64 is a
    /// perfect square); the other three faces are at 10/√13 and 10/7 > 1. The
    /// hull has one extra vertex at (1 + 2^-16, 0, 0), i.e. exactly EPSILON
    /// beyond that face. The strict test does not converge yet: the polytope
    /// is expanded through the extra vertex and the reported depth exceeds 1
    /// (by a little less than 2^-16). The `<=` mutant stops immediately with
    /// depth exactly 1.
    #[test]
    fn epa_convergence_threshold_is_strict() {
        let eps = Fix128 {
            hi: 0,
            lo: 0x0001000000000000,
        };
        let beyond = v3i(1, 0, 0) + Vec3Fix::new(eps, Fix128::ZERO, Fix128::ZERO);
        let simplex = [v3i(1, 4, 4), v3i(1, -4, 4), v3i(1, 0, -4), v3i(-5, 0, 0)];
        assert!(tetra_contains_origin(&simplex));
        let mut verts = simplex.to_vec();
        verts.push(beyond);
        let a = ConvexHull::new(verts);
        let b = ConvexHull::new(vec![Vec3Fix::ZERO]);
        assert_eq!(a.support(Vec3Fix::UNIT_X), beyond);
        let c = epa(&a, &b, &simplex).expect("origin inside hull");
        let half_eps = Fix128 {
            hi: 0,
            lo: 0x0000800000000000,
        };
        assert!(c.depth > Fix128::ONE + half_eps, "depth {:?}", c.depth);
        assert!(c.depth <= Fix128::ONE + eps, "depth {:?}", c.depth);
        assert!(c.normal.x < Fix128::ONE, "{:?}", c.normal);
    }

    /// Box-box EPA from hand-built, origin-enclosing initial tetrahedra whose
    /// closest face is *not* the answer, so the polytope has to be expanded.
    /// Kills `add_face` `distance < 0` → `== 0` (line 648, config A: the
    /// inward-facing initial faces are never flipped → EPA fails), the
    /// `(-normal, -distance)` → `(normal, -distance)` mutant (line 649,
    /// config B: reports 1/4 along y instead of 1/8 along x) and
    /// `(i + 1) % 3` → `/ 3` (line 607, config C: the polytope loses faces and
    /// EPA fails).
    ///
    /// A = [-1,1]^3 throughout; with B = [bx,3]×[by,3]×[bz,3] the Minkowski
    /// difference A ⊖ B is [-4, 1-bx]×[-4, 1-by]×[-4, 1-bz], whose corners are
    /// exact support points, and the penetration depth is the smallest of the
    /// three positive extents. The contact points are `A.support(n)` and
    /// `B.support(-n)`, so `(point_a - point_b)·n` equals the depth.
    #[test]
    fn epa_box_box_exact_depth_from_hand_built_simplices() {
        let a = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        let r = Fix128::from_ratio;
        let check = |b: &AABB, simplex: &[Vec3Fix; 4], depth: Fix128, axis: usize| {
            assert!(tetra_contains_origin(simplex), "{simplex:?}");
            let c = epa(&a, b, simplex).expect("overlapping boxes");
            assert_eq!(c.depth, depth, "depth {:?}", c.depth);
            let n = [c.normal.x, c.normal.y, c.normal.z];
            for (i, comp) in n.iter().enumerate() {
                if i == axis {
                    assert_eq!(comp.abs(), Fix128::ONE, "{:?}", c.normal);
                } else {
                    assert_eq!(*comp, Fix128::ZERO, "{:?}", c.normal);
                }
            }
            // normal is B→A: A's deepest point lies along −normal (A→B)
            assert_eq!(c.point_a, a.support(-c.normal));
            assert_eq!(c.point_b, b.support(c.normal));
            assert_eq!((c.point_a - c.point_b).dot(c.normal).abs(), depth);
            // B→A: the normal points from B's centre towards A's centre
            let ab = b.min + b.max - (a.min + a.max);
            assert!(ab.dot(c.normal) <= Fix128::ZERO, "B→A sign: {:?}", c.normal);
        };

        // Config A: B = [5/8,3]×[1/2,3]×[1/4,3] → A ⊖ B = [-4,3/8]×[-4,1/2]×[-4,3/4],
        // depth 3/8 along x. Tetra = corners (3/8,1/2,-4), (-4,-4,3/4),
        // (3/8,-4,3/4), (3/8,1/2,3/4).
        let b = AABB::new(Vec3Fix::new(r(5, 8), r(1, 2), r(1, 4)), v3i(3, 3, 3));
        let t = [
            Vec3Fix::new(r(3, 8), r(1, 2), fi(-4)),
            Vec3Fix::new(fi(-4), fi(-4), r(3, 4)),
            Vec3Fix::new(r(3, 8), fi(-4), r(3, 4)),
            Vec3Fix::new(r(3, 8), r(1, 2), r(3, 4)),
        ];
        check(&b, &t, r(3, 8), 0);

        // Config B: B = [7/8,3]×[3/4,3]×[5/8,3] → A ⊖ B = [-4,1/8]×[-4,1/4]×[-4,3/8],
        // depth 1/8 along x. Tetra = corners (1/8,-4,-4), (-4,1/4,-4),
        // (-4,-4,3/8), (1/8,1/4,3/8).
        let b = AABB::new(Vec3Fix::new(r(7, 8), r(3, 4), r(5, 8)), v3i(3, 3, 3));
        let t = [
            Vec3Fix::new(r(1, 8), fi(-4), fi(-4)),
            Vec3Fix::new(fi(-4), r(1, 4), fi(-4)),
            Vec3Fix::new(fi(-4), fi(-4), r(3, 8)),
            Vec3Fix::new(r(1, 8), r(1, 4), r(3, 8)),
        ];
        check(&b, &t, r(1, 8), 0);

        // Config C: B = [5/8,3]×[1/2,3]×[3/4,3] → A ⊖ B = [-4,3/8]×[-4,1/2]×[-4,1/4],
        // depth 1/4 along z. Tetra = corners (-4,-4,-4), (3/8,-4,1/4),
        // (-4,1/2,1/4), (3/8,1/2,1/4).
        let b = AABB::new(Vec3Fix::new(r(5, 8), r(1, 2), r(3, 4)), v3i(3, 3, 3));
        let t = [
            v3i(-4, -4, -4),
            Vec3Fix::new(r(3, 8), fi(-4), r(1, 4)),
            Vec3Fix::new(fi(-4), r(1, 2), r(1, 4)),
            Vec3Fix::new(r(3, 8), r(1, 2), r(1, 4)),
        ];
        check(&b, &t, r(1, 4), 2);
    }

    /// Sphere-box EPA with a curved Minkowski boundary, so the polytope is
    /// refined over several expansions. Kills `to_new = support - v` →
    /// `support + v` (line 602: every face becomes "visible", the edge
    /// bookkeeping breaks and EPA returns `None`) and, again, line 607.
    ///
    /// Box A = [-1,1]^3, sphere B centre (-3/2,0,0) r=1: the sphere reaches
    /// x = -1/2 past the box face x = -1, so the penetration depth is 1/2
    /// along x (analytic). Initial tetrahedron = Minkowski support points in
    /// the directions (1,0,0), (-1,1,0), (-1,-1,1), (-1,-1,-1); origin
    /// containment is checked with signed volumes. EPA converges to within
    /// its EPSILON (2^-16), so the comparison uses 2^-13.
    #[test]
    fn epa_sphere_box_face_penetration_is_analytic() {
        let a = AABB::new(v3i(-1, -1, -1), v3i(1, 1, 1));
        let b = Sphere::new(
            Vec3Fix::new(Fix128::from_ratio(-3, 2), Fix128::ZERO, Fix128::ZERO),
            Fix128::ONE,
        );
        let dirs = [v3i(1, 0, 0), v3i(-1, 1, 0), v3i(-1, -1, 1), v3i(-1, -1, -1)];
        let mut simplex = [Vec3Fix::ZERO; 4];
        for (i, d) in dirs.iter().enumerate() {
            simplex[i] = minkowski_support(&a, &b, *d);
        }
        assert!(tetra_contains_origin(&simplex));
        let c = epa(&a, &b, &simplex).expect("sphere overlaps box");
        let tol = Fix128::from_ratio(1, 8192);
        assert!(
            near_tol(c.depth, Fix128::from_ratio(1, 2), tol),
            "depth {:?}",
            c.depth
        );
        assert!(
            near_tol(c.normal.x.abs(), Fix128::ONE, tol),
            "{:?}",
            c.normal
        );
        assert!(
            c.normal.y.abs() < tol && c.normal.z.abs() < tol,
            "{:?}",
            c.normal
        );
        // sphere B is on −x of box A: B→A normal is +x
        assert!(c.normal.x > Fix128::ZERO, "B→A sign: {:?}", c.normal);
        assert_eq!(c.point_a, a.support(-c.normal));
        assert_eq!(c.point_b, b.support(c.normal));
    }

    /// Touching boxes: the origin lies on a face of the initial simplex
    /// (distance exactly 0) and on the boundary of A ⊖ B. Kills `add_face`
    /// `distance < 0` → `<= 0` (line 648: the zero-distance face gets its
    /// normal flipped to point into the polytope, EPA expands the wrong way
    /// and reports depth 4 = the far face) and again `→ == 0`.
    ///
    /// A = [0,2]×[-1,1]^2, B = [-2,0]×[-1,1]^2 share the plane x = 0, so
    /// A ⊖ B = [0,4]×[-2,2]^2 has the origin on its face x = 0. Initial
    /// simplex (all corners / face points of A ⊖ B): p0 = (0,-2,2),
    /// p1 = (0,2,2), p2 = (0,0,-2), p3 = (4,0,0). Face (p0,p1,p2):
    /// (p1-p0) × (p2-p0) = (0,4,0) × (0,2,-4) = (-16,0,0) → normal (-1,0,0)
    /// exactly, distance p0·n = 0. It is the closest face; the support in
    /// (-1,0,0) is A.support(-x) - B.support(+x) = (0,1,1) - (0,1,1) = 0 with
    /// dot 0, so EPA converges at once: depth 0, normal (-1,0,0), and both
    /// contact points are (0,1,1).
    ///
    /// Note: a zero-distance face keeps the orientation given by its vertex
    /// winding (there is no "away from the origin" side); this simplex is
    /// wound so that the normal points out of the polytope.
    #[test]
    fn epa_touching_boxes_zero_depth_face_through_origin() {
        let a = AABB::new(v3i(0, -1, -1), v3i(2, 1, 1));
        let b = AABB::new(v3i(-2, -1, -1), v3i(0, 1, 1));
        // A is on +x of B: B→A normal = +x, depth 0, for either winding of
        // the face through the origin (1.2.0: faces are oriented by the
        // polytope interior; before that this winding reported depth 4)
        for simplex in [
            [v3i(0, -2, 2), v3i(0, 2, 2), v3i(0, 0, -2), v3i(4, 0, 0)],
            [v3i(0, 2, 2), v3i(0, -2, 2), v3i(0, 0, -2), v3i(4, 0, 0)],
        ] {
            let c = epa(&a, &b, &simplex).expect("touching boxes");
            assert_eq!(c.depth, Fix128::ZERO, "{simplex:?}");
            assert_eq!(c.normal, v3i(1, 0, 0), "{simplex:?}");
            assert_eq!(c.point_a, v3i(0, 1, 1));
            assert_eq!(c.point_b, v3i(0, 1, 1));
        }
    }
}
