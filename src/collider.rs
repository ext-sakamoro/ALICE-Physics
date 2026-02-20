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
#[derive(Clone, Copy, Debug)]
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
    pub fn new(depth: Fix128, normal: Vec3Fix, point_a: Vec3Fix, point_b: Vec3Fix) -> Self {
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
#[derive(Clone, Copy, Debug)]
pub struct Sphere {
    /// Center position
    pub center: Vec3Fix,
    /// Sphere radius
    pub radius: Fix128,
}

impl Sphere {
    /// Create a new sphere from center and radius
    pub fn new(center: Vec3Fix, radius: Fix128) -> Self {
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
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    /// Minimum corner
    pub min: Vec3Fix,
    /// Maximum corner
    pub max: Vec3Fix,
}

impl AABB {
    /// Create a new AABB from min and max corners
    pub fn new(min: Vec3Fix, max: Vec3Fix) -> Self {
        Self { min, max }
    }

    /// Create AABB from center and half-extents
    pub fn from_center_half(center: Vec3Fix, half: Vec3Fix) -> Self {
        Self {
            min: center - half,
            max: center + half,
        }
    }

    /// Check if two AABBs intersect (broad phase)
    #[inline]
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Compute union of two AABBs
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
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
    /// Create a new convex hull from vertices
    pub fn new(vertices: Vec<Vec3Fix>) -> Self {
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
#[derive(Clone, Copy, Debug)]
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
    pub fn new(a: Vec3Fix, b: Vec3Fix, radius: Fix128) -> Self {
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
#[derive(Clone, Copy, Debug)]
pub struct ScaledShape<S> {
    /// Inner shape
    pub shape: S,
    /// Uniform scale factor
    pub scale: Fix128,
}

impl<S> ScaledShape<S> {
    /// Create a new scaled shape
    pub fn new(shape: S, scale: Fix128) -> Self {
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
    fn new() -> Self {
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
#[derive(Clone, Copy, Debug)]
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
        *direction = ab.cross(ao).cross(ab);
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
    } else {
        if ab.cross(abc).dot(ao) > Fix128::ZERO {
            simplex.set(&[a, b]);
            return do_simplex_line(simplex, direction);
        } else {
            if abc.dot(ao) > Fix128::ZERO {
                *direction = abc;
            } else {
                simplex.set(&[a, c, b]);
                *direction = -abc;
            }
        }
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
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone, Copy, Debug)]
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
    let normal = ab.cross(ac).normalize();

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
}
