//! Cylinder Collider
//!
//! Cylinder collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Cylinder**: Y-axis aligned cylinder with half-height and radius
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Fast AABB**: Compute world-space AABB from oriented cylinder
//!
//! Author: Moroya Sakamoto

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Cylinder collider
///
/// A cylinder defined by center position, half-height (along local Y axis),
/// radius, and an orientation quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cylinder {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Half-height (from center to cap along local Y axis)
    pub half_height: Fix128,
    /// Radius of the circular cross-section
    pub radius: Fix128,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Cylinder {
    /// Create a new axis-aligned cylinder (Y-up)
    #[inline]
    #[must_use]
    pub fn new(center: Vec3Fix, half_height: Fix128, radius: Fix128) -> Self {
        Self {
            center,
            half_height,
            radius,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented cylinder
    #[inline]
    #[must_use]
    pub fn with_rotation(
        center: Vec3Fix,
        half_height: Fix128,
        radius: Fix128,
        rotation: QuatFix,
    ) -> Self {
        Self {
            center,
            half_height,
            radius,
            rotation,
        }
    }

    /// Compute world-space AABB enclosing this cylinder
    #[must_use]
    pub fn aabb(&self) -> AABB {
        // Local Y axis in world space
        let local_y = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let world_y = self.rotation.rotate_vec(local_y);

        // The cylinder extends half_height along world_y
        let axis_extent = Vec3Fix::new(
            (world_y.x * self.half_height).abs(),
            (world_y.y * self.half_height).abs(),
            (world_y.z * self.half_height).abs(),
        );

        // Conservative radial extent on each world axis
        let radial_extent = Vec3Fix::new(self.radius, self.radius, self.radius);

        let total = Vec3Fix::new(
            axis_extent.x + radial_extent.x,
            axis_extent.y + radial_extent.y,
            axis_extent.z + radial_extent.z,
        );

        AABB::new(self.center - total, self.center + total)
    }

    /// Volume of the cylinder: pi * r^2 * 2 * `half_height`
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        // pi ≈ 355/113 (Milü approximation, accurate to 7 digits)
        let pi = Fix128::from_ratio(355, 113);
        let two = Fix128::from_int(2);
        pi * self.radius * self.radius * two * self.half_height
    }

    /// Surface area: 2 * pi * r * (r + 2 * `half_height`)
    #[inline]
    #[must_use]
    pub fn surface_area(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let two = Fix128::from_int(2);
        two * pi * self.radius * (self.radius + two * self.half_height)
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// For a solid cylinder aligned along Y:
    /// - Ixx = Izz = m/12 * (3*r^2 + h^2) where h = 2*`half_height`
    /// - Iyy = m * r^2 / 2
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let r2 = self.radius * self.radius;
        let h = self.half_height * Fix128::from_int(2);
        let h2 = h * h;
        let three = Fix128::from_int(3);
        let twelve = Fix128::from_int(12);
        let two = Fix128::from_int(2);

        let ixx = mass * (three * r2 + h2) / twelve;
        let iyy = mass * r2 / two;

        Vec3Fix::new(ixx, iyy, ixx)
    }
}

impl Support for Cylinder {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Transform direction to local space
        let local_dir = self.rotation.conjugate().rotate_vec(direction);

        // In local space (Y-up cylinder):
        // Along Y axis: sign(local_dir.y) * half_height
        // On XZ plane: radius * normalize(local_dir.x, local_dir.z)
        let y_support = if local_dir.y >= Fix128::ZERO {
            self.half_height
        } else {
            -self.half_height
        };

        let xz_len_sq = local_dir.x * local_dir.x + local_dir.z * local_dir.z;
        let local_support = if xz_len_sq.is_zero() {
            Vec3Fix::new(Fix128::ZERO, y_support, Fix128::ZERO)
        } else {
            let xz_len = xz_len_sq.sqrt();
            let x_support = self.radius * local_dir.x / xz_len;
            let z_support = self.radius * local_dir.z / xz_len;
            Vec3Fix::new(x_support, y_support, z_support)
        };

        // Transform back to world space
        self.center + self.rotation.rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cylinder_support_up() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::ONE);
        let s = cyl.support(Vec3Fix::UNIT_Y);
        assert_eq!(s.y.hi, 2);
    }

    #[test]
    fn test_cylinder_support_side() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::from_int(3));
        let s = cyl.support(Vec3Fix::UNIT_X);
        // Pure X direction: support at (radius, half_height, 0)
        assert_eq!(s.x.hi, 3); // radius
        assert_eq!(s.y.hi, 2); // half_height (y >= 0)
    }

    #[test]
    fn test_cylinder_support_down() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::ONE);
        let s = cyl.support(-Vec3Fix::UNIT_Y);
        assert_eq!(s.y.hi, -2);
    }

    #[test]
    fn test_cylinder_aabb() {
        let cyl = Cylinder::new(Vec3Fix::from_int(5, 0, 0), Fix128::from_int(2), Fix128::ONE);
        let aabb = cyl.aabb();
        // Y-up: axis_extent=(0,2,0), radial=(1,1,1), total=(1,3,1)
        assert_eq!(aabb.min.x.hi, 4); // 5 - 1
        assert_eq!(aabb.max.x.hi, 6); // 5 + 1
        assert_eq!(aabb.min.y.hi, -3); // 0 - 3
        assert_eq!(aabb.max.y.hi, 3); // 0 + 3
    }

    #[test]
    fn test_cylinder_volume() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        // V = pi * 1^2 * 2 * 1 = 2*pi ≈ 6.28
        let v = cyl.volume();
        assert!(
            v.hi >= 6 && v.hi <= 7,
            "Volume should be ~6.28, got {}",
            v.hi
        );
    }

    #[test]
    fn test_cylinder_inertia() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        let mass = Fix128::from_int(12);
        let inertia = cyl.inertia_diagonal(mass);
        // Ixx = m/12 * (3*1 + 4) = 12/12 * 7 = 7
        assert_eq!(inertia.x.hi, 7);
        // Iyy = m * 1 / 2 = 6
        assert_eq!(inertia.y.hi, 6);
    }

    #[test]
    fn test_cylinder_surface_area() {
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        // SA = 2*pi*1*(1 + 2*1) = 6*pi ≈ 18.85
        let sa = cyl.surface_area();
        assert!(
            sa.hi >= 18 && sa.hi <= 19,
            "Surface area should be ~18.85, got {}",
            sa.hi
        );
    }

    #[test]
    fn test_rotated_cylinder_support() {
        let rot = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
        let cyl = Cylinder::with_rotation(Vec3Fix::ZERO, Fix128::from_int(3), Fix128::ONE, rot);
        // After 90° rotation around Z, local Y maps to world -X
        let s = cyl.support(Vec3Fix::UNIT_X);
        // Support in +X should pick up the half_height (3) projected onto X
        assert!(
            s.x > Fix128::ONE,
            "Rotated cylinder support should extend along X"
        );
    }

    #[test]
    fn test_cylinder_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::ONE);
        let sphere = Sphere::new(Vec3Fix::from_int(0, 0, 0), Fix128::ONE);
        let result = gjk(&cyl, &sphere);
        assert!(
            result.colliding,
            "Overlapping cylinder and sphere should collide"
        );
    }

    #[test]
    fn test_cylinder_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let cyl = Cylinder::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        let sphere = Sphere::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let result = gjk(&cyl, &sphere);
        assert!(
            !result.colliding,
            "Separated cylinder and sphere should not collide"
        );
    }
}
