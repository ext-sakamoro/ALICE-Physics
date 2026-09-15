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
/// Call this after GJK returns a collision.
/// Deterministic: fixed iteration count.
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

    // Initialize with tetrahedron faces
    add_face(&mut faces, &vertices, 0, 1, 2);
    add_face(&mut faces, &vertices, 0, 3, 1);
    add_face(&mut faces, &vertices, 0, 2, 3);
    add_face(&mut faces, &vertices, 1, 3, 2);

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
            // Found the closest face
            let normal = closest_face.normal;
            let depth = closest_face.distance;

            // Compute contact points (simplified)
            let point_a = a.support(normal);
            let point_b = b.support(-normal);

            return Some(Contact {
                depth,
                normal,
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
            add_face(&mut faces, &vertices, i, j, new_idx);
        }
    }

    None
}

fn add_face(faces: &mut Vec<EpaFace>, vertices: &[Vec3Fix], i: usize, j: usize, k: usize) {
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

    // Ensure normal points away from origin
    let distance = a.dot(normal);
    let (normal, distance) = if distance < Fix128::ZERO {
        (-normal, -distance)
    } else {
        (normal, distance)
    };

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
}
