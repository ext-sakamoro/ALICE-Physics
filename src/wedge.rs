//! Wedge (Triangular Prism) Collider
//!
//! Wedge collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Wedge**: Triangular prism with 6 vertices
//! - **GJK Support**: Implements `Support` trait by iterating all vertices
//! - **Fast AABB**: Compute world-space AABB from all vertices
//!
//! # Geometry
//!
//! The wedge is a triangular prism defined by width (X), height (Y), and
//! depth (Z) dimensions. In local space the triangular cross-section lies
//! in the XY plane:
//!
//! ```text
//!     (0, +height/2) ---- apex
//!    /                 \
//! (-width/2, -height/2) -- (+width/2, -height/2)
//! ```
//!
//! The prism extends from `-depth/2` to `+depth/2` along the Z axis.
//!
//! Author: Moroya Sakamoto

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Wedge (triangular prism) collider
///
/// A triangular prism defined by center, width, height, depth, and rotation.
/// The triangular cross-section is in the local XY plane, extruded along Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wedge {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Full width along local X axis
    pub width: Fix128,
    /// Full height along local Y axis
    pub height: Fix128,
    /// Full depth along local Z axis (extrusion direction)
    pub depth: Fix128,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Wedge {
    /// Create a new axis-aligned wedge
    #[inline]
    #[must_use]
    pub fn new(center: Vec3Fix, width: Fix128, height: Fix128, depth: Fix128) -> Self {
        Self {
            center,
            width,
            height,
            depth,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented wedge
    #[inline]
    #[must_use]
    pub fn with_rotation(
        center: Vec3Fix,
        width: Fix128,
        height: Fix128,
        depth: Fix128,
        rotation: QuatFix,
    ) -> Self {
        Self {
            center,
            width,
            height,
            depth,
            rotation,
        }
    }

    /// Get all 6 vertices of the wedge in world space
    ///
    /// Local vertices (triangular cross-section extruded along Z):
    /// - Front triangle (z = -depth/2):
    ///   v0 = (-width/2, -height/2, -depth/2)
    ///   v1 = (+width/2, -height/2, -depth/2)
    ///   v2 = (0, +height/2, -depth/2)
    /// - Back triangle (z = +depth/2):
    ///   v3 = (-width/2, -height/2, +depth/2)
    ///   v4 = (+width/2, -height/2, +depth/2)
    ///   v5 = (0, +height/2, +depth/2)
    #[must_use]
    pub fn vertices(&self) -> [Vec3Fix; 6] {
        let hw = self.width.half();
        let hh = self.height.half();
        let hd = self.depth.half();

        let local_verts = [
            Vec3Fix::new(-hw, -hh, -hd),
            Vec3Fix::new(hw, -hh, -hd),
            Vec3Fix::new(Fix128::ZERO, hh, -hd),
            Vec3Fix::new(-hw, -hh, hd),
            Vec3Fix::new(hw, -hh, hd),
            Vec3Fix::new(Fix128::ZERO, hh, hd),
        ];

        let mut result = [Vec3Fix::ZERO; 6];
        for (i, lv) in local_verts.iter().enumerate() {
            result[i] = self.center + self.rotation.rotate_vec(*lv);
        }
        result
    }

    /// Compute world-space AABB enclosing this wedge
    #[must_use]
    pub fn aabb(&self) -> AABB {
        let verts = self.vertices();
        let mut min = verts[0];
        let mut max = verts[0];

        for v in &verts[1..] {
            if v.x < min.x {
                min.x = v.x;
            }
            if v.y < min.y {
                min.y = v.y;
            }
            if v.z < min.z {
                min.z = v.z;
            }
            if v.x > max.x {
                max.x = v.x;
            }
            if v.y > max.y {
                max.y = v.y;
            }
            if v.z > max.z {
                max.z = v.z;
            }
        }

        AABB::new(min, max)
    }

    /// Volume of the wedge: (1/2) * width * height * depth
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        let two = Fix128::from_int(2);
        self.width * self.height * self.depth / two
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// Approximation treating the wedge as a triangular prism.
    /// For a uniform triangular prism:
    /// - Ixx = m/18 * (h^2) + m/12 * d^2
    /// - Iyy = m/18 * (w^2) + m/12 * d^2
    /// - Izz = m/18 * (w^2 + h^2)
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let eighteen = Fix128::from_int(18);
        let twelve = Fix128::from_int(12);
        let w2 = self.width * self.width;
        let h2 = self.height * self.height;
        let d2 = self.depth * self.depth;

        Vec3Fix::new(
            mass * h2 / eighteen + mass * d2 / twelve,
            mass * w2 / eighteen + mass * d2 / twelve,
            mass * (w2 + h2) / eighteen,
        )
    }
}

impl Support for Wedge {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // For a convex polytope, the support is the vertex with the max
        // dot product with the direction. Since vertices are already in
        // world space from `vertices()`, we iterate and pick the best.
        let verts = self.vertices();
        let mut best = verts[0];
        let mut best_dot = best.dot(direction);

        for v in &verts[1..] {
            let d = v.dot(direction);
            if d > best_dot {
                best = *v;
                best_dot = d;
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wedge_new() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(3),
            Fix128::from_int(2),
        );
        assert_eq!(w.width.hi, 4);
        assert_eq!(w.height.hi, 3);
        assert_eq!(w.depth.hi, 2);
    }

    #[test]
    fn test_wedge_vertices_count() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            Fix128::from_int(2),
        );
        let verts = w.vertices();
        assert_eq!(verts.len(), 6);
    }

    #[test]
    fn test_wedge_vertices_positions() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(6),
            Fix128::from_int(2),
        );
        let verts = w.vertices();
        // v0 = (-2, -3, -1)
        assert_eq!(verts[0].x.hi, -2);
        assert_eq!(verts[0].y.hi, -3);
        assert_eq!(verts[0].z.hi, -1);
        // v1 = (2, -3, -1)
        assert_eq!(verts[1].x.hi, 2);
        assert_eq!(verts[1].y.hi, -3);
        // v2 = (0, 3, -1) — apex
        assert_eq!(verts[2].x.hi, 0);
        assert_eq!(verts[2].y.hi, 3);
    }

    #[test]
    fn test_wedge_support_up() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(6),
            Fix128::from_int(2),
        );
        let s = w.support(Vec3Fix::UNIT_Y);
        // Max dot with +Y: apex vertices at y=3, so support y=3
        assert_eq!(s.y.hi, 3);
    }

    #[test]
    fn test_wedge_support_down() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(6),
            Fix128::from_int(2),
        );
        let s = w.support(-Vec3Fix::UNIT_Y);
        // Max dot with -Y: base vertices at y=-3
        assert_eq!(s.y.hi, -3);
    }

    #[test]
    fn test_wedge_aabb() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(6),
            Fix128::from_int(2),
        );
        let aabb = w.aabb();
        assert_eq!(aabb.min.x.hi, -2);
        assert_eq!(aabb.max.x.hi, 2);
        assert_eq!(aabb.min.y.hi, -3);
        assert_eq!(aabb.max.y.hi, 3);
        assert_eq!(aabb.min.z.hi, -1);
        assert_eq!(aabb.max.z.hi, 1);
    }

    #[test]
    fn test_wedge_volume() {
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(6),
            Fix128::from_int(2),
        );
        // V = (1/2) * 4 * 6 * 2 = 24
        assert_eq!(w.volume().hi, 24);
    }

    #[test]
    fn test_wedge_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(4),
            Fix128::from_int(4),
            Fix128::from_int(4),
        );
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let result = gjk(&w, &sphere);
        assert!(
            result.colliding,
            "Overlapping wedge and sphere should collide"
        );
    }

    #[test]
    fn test_wedge_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let w = Wedge::new(
            Vec3Fix::ZERO,
            Fix128::from_int(2),
            Fix128::from_int(2),
            Fix128::from_int(2),
        );
        let sphere = Sphere::new(Vec3Fix::from_int(20, 0, 0), Fix128::ONE);
        let result = gjk(&w, &sphere);
        assert!(
            !result.colliding,
            "Separated wedge and sphere should not collide"
        );
    }

    #[test]
    fn test_wedge_offset_center() {
        let w = Wedge::new(
            Vec3Fix::from_int(10, 20, 30),
            Fix128::from_int(2),
            Fix128::from_int(2),
            Fix128::from_int(2),
        );
        let verts = w.vertices();
        // v0 = (10-1, 20-1, 30-1) = (9, 19, 29)
        assert_eq!(verts[0].x.hi, 9);
        assert_eq!(verts[0].y.hi, 19);
        assert_eq!(verts[0].z.hi, 29);
    }
}
