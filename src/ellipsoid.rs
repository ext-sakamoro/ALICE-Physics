//! Ellipsoid Collider
//!
//! Ellipsoid collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Ellipsoid**: Three independent semi-axis radii (rx, ry, rz)
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Bounding Sphere**: Conservative bounding sphere from max semi-axis
//!
//! # Geometry
//!
//! The ellipsoid is defined by a center, three semi-axis radii stored as a
//! `Vec3Fix`, and an orientation quaternion. In local space the surface
//! satisfies `(x/rx)^2 + (y/ry)^2 + (z/rz)^2 = 1`.
//!
//! Author: Moroya Sakamoto

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Ellipsoid collider
///
/// An ellipsoid defined by center position, three semi-axis radii, and an
/// orientation quaternion. The radii are stored as `Vec3Fix` where
/// `radii.x`, `radii.y`, `radii.z` are the semi-axes along local X, Y, Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ellipsoid {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Semi-axis radii (rx, ry, rz) in local space
    pub radii: Vec3Fix,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Ellipsoid {
    /// Create a new axis-aligned ellipsoid
    #[inline]
    #[must_use]
    pub fn new(center: Vec3Fix, radii: Vec3Fix) -> Self {
        Self {
            center,
            radii,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented ellipsoid
    #[inline]
    #[must_use]
    pub fn with_rotation(center: Vec3Fix, radii: Vec3Fix, rotation: QuatFix) -> Self {
        Self {
            center,
            radii,
            rotation,
        }
    }

    /// Bounding sphere radius (maximum semi-axis)
    #[must_use]
    pub fn bounding_sphere_radius(&self) -> Fix128 {
        let mut max = self.radii.x;
        if self.radii.y > max {
            max = self.radii.y;
        }
        if self.radii.z > max {
            max = self.radii.z;
        }
        max
    }

    /// Compute world-space AABB enclosing this ellipsoid
    #[must_use]
    pub fn aabb(&self) -> AABB {
        // Transform each local axis scaled by its radius to world space
        let local_x = Vec3Fix::new(self.radii.x, Fix128::ZERO, Fix128::ZERO);
        let local_y = Vec3Fix::new(Fix128::ZERO, self.radii.y, Fix128::ZERO);
        let local_z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, self.radii.z);

        let world_x = self.rotation.rotate_vec(local_x);
        let world_y = self.rotation.rotate_vec(local_y);
        let world_z = self.rotation.rotate_vec(local_z);

        // For an ellipsoid, the extent on each world axis is:
        // sqrt(wx_i^2 + wy_i^2 + wz_i^2) per component i
        // Conservative approximation using abs sums:
        let extent = Vec3Fix::new(
            world_x.x.abs() + world_y.x.abs() + world_z.x.abs(),
            world_x.y.abs() + world_y.y.abs() + world_z.y.abs(),
            world_x.z.abs() + world_y.z.abs() + world_z.z.abs(),
        );

        AABB::new(self.center - extent, self.center + extent)
    }

    /// Volume of the ellipsoid: (4/3) * pi * rx * ry * rz
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let four_thirds = Fix128::from_ratio(4, 3);
        four_thirds * pi * self.radii.x * self.radii.y * self.radii.z
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// For a solid ellipsoid:
    /// - Ixx = m/5 * (ry^2 + rz^2)
    /// - Iyy = m/5 * (rx^2 + rz^2)
    /// - Izz = m/5 * (rx^2 + ry^2)
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let five = Fix128::from_int(5);
        let rx2 = self.radii.x * self.radii.x;
        let ry2 = self.radii.y * self.radii.y;
        let rz2 = self.radii.z * self.radii.z;

        Vec3Fix::new(
            mass * (ry2 + rz2) / five,
            mass * (rx2 + rz2) / five,
            mass * (rx2 + ry2) / five,
        )
    }
}

impl Support for Ellipsoid {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Transform direction to local space
        let local_dir = self.rotation.conjugate().rotate_vec(direction);

        // Scale direction by radii (componentwise)
        let scaled = Vec3Fix::new(
            local_dir.x * self.radii.x,
            local_dir.y * self.radii.y,
            local_dir.z * self.radii.z,
        );

        let len = scaled.length();
        let local_support = if len.is_zero() {
            // Degenerate: return point along X semi-axis
            Vec3Fix::new(self.radii.x, Fix128::ZERO, Fix128::ZERO)
        } else {
            // Normalize the scaled direction and scale by radii again
            Vec3Fix::new(
                self.radii.x * scaled.x / len,
                self.radii.y * scaled.y / len,
                self.radii.z * scaled.z / len,
            )
        };

        // Transform back to world space
        self.center + self.rotation.rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ellipsoid_new() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 2, 3));
        assert_eq!(e.radii.x.hi, 1);
        assert_eq!(e.radii.y.hi, 2);
        assert_eq!(e.radii.z.hi, 3);
    }

    #[test]
    fn test_ellipsoid_bounding_sphere_radius() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 5, 3));
        assert_eq!(e.bounding_sphere_radius().hi, 5);
    }

    #[test]
    fn test_ellipsoid_support_x() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(3, 1, 1));
        let s = e.support(Vec3Fix::UNIT_X);
        // Support along +X for axis-aligned ellipsoid should be (rx, 0, 0)
        assert_eq!(s.x.hi, 3);
        assert!(s.y.abs().hi == 0);
        assert!(s.z.abs().hi == 0);
    }

    #[test]
    fn test_ellipsoid_support_y() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 4, 1));
        let s = e.support(Vec3Fix::UNIT_Y);
        assert_eq!(s.y.hi, 4);
    }

    #[test]
    fn test_ellipsoid_support_neg_z() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 7));
        let s = e.support(-Vec3Fix::UNIT_Z);
        assert_eq!(s.z.hi, -7);
    }

    #[test]
    fn test_ellipsoid_sphere_degenerates() {
        // An ellipsoid with equal radii is a sphere
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 2, 2));
        let s = e.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 2);

        let s2 = e.support(Vec3Fix::UNIT_Y);
        assert_eq!(s2.y.hi, 2);
    }

    #[test]
    fn test_ellipsoid_volume() {
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        // V = (4/3)*pi*1*1*1 = 4*pi/3 ≈ 4.189
        let v = e.volume();
        assert!(
            v.hi >= 4 && v.hi <= 5,
            "Volume should be ~4.189, got {}",
            v.hi
        );
    }

    #[test]
    fn test_ellipsoid_aabb() {
        let e = Ellipsoid::new(Vec3Fix::from_int(5, 0, 0), Vec3Fix::from_int(2, 3, 1));
        let aabb = e.aabb();
        assert_eq!(aabb.min.x.hi, 3); // 5 - 2
        assert_eq!(aabb.max.x.hi, 7); // 5 + 2
        assert_eq!(aabb.min.y.hi, -3); // 0 - 3
        assert_eq!(aabb.max.y.hi, 3); // 0 + 3
    }

    #[test]
    fn test_ellipsoid_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(2, 2, 2));
        let sphere = Sphere::new(Vec3Fix::from_int(1, 0, 0), Fix128::ONE);
        let result = gjk(&e, &sphere);
        assert!(
            result.colliding,
            "Overlapping ellipsoid and sphere should collide"
        );
    }

    #[test]
    fn test_ellipsoid_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let e = Ellipsoid::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        let sphere = Sphere::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let result = gjk(&e, &sphere);
        assert!(
            !result.colliding,
            "Separated ellipsoid and sphere should not collide"
        );
    }

    #[test]
    fn test_ellipsoid_offset_center() {
        let e = Ellipsoid::new(Vec3Fix::from_int(10, 20, 30), Vec3Fix::from_int(1, 1, 1));
        let s = e.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 11);
        assert_eq!(s.y.hi, 20);
        assert_eq!(s.z.hi, 30);
    }
}
