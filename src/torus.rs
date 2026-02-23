//! Torus Collider
//!
//! Torus collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Torus**: Defined by major radius (ring center) and minor radius (tube)
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Volume**: Analytic volume formula
//!
//! # Geometry
//!
//! The torus is centered at `center` with the ring lying in the local XZ plane.
//! The major radius is the distance from the center to the tube center, and
//! the minor radius is the tube radius. The local Y axis is the symmetry axis.
//!
//! Author: Moroya Sakamoto

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Torus collider
///
/// A torus defined by center position, major radius (distance from center to
/// tube center line), minor radius (tube cross-section radius), and an
/// orientation quaternion. The ring lies in the local XZ plane.
#[derive(Clone, Copy, Debug)]
pub struct Torus {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Major radius (from center to tube center)
    pub major_radius: Fix128,
    /// Minor radius (tube cross-section radius)
    pub minor_radius: Fix128,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Torus {
    /// Create a new axis-aligned torus (ring in XZ plane)
    #[inline]
    #[must_use]
    pub fn new(center: Vec3Fix, major_radius: Fix128, minor_radius: Fix128) -> Self {
        Self {
            center,
            major_radius,
            minor_radius,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented torus
    #[inline]
    #[must_use]
    pub fn with_rotation(
        center: Vec3Fix,
        major_radius: Fix128,
        minor_radius: Fix128,
        rotation: QuatFix,
    ) -> Self {
        Self {
            center,
            major_radius,
            minor_radius,
            rotation,
        }
    }

    /// Volume of the torus: 2 * pi^2 * R * r^2
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let two = Fix128::from_int(2);
        two * pi * pi * self.major_radius * self.minor_radius * self.minor_radius
    }

    /// Surface area: 4 * pi^2 * R * r
    #[inline]
    #[must_use]
    pub fn surface_area(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let four = Fix128::from_int(4);
        four * pi * pi * self.major_radius * self.minor_radius
    }

    /// Compute world-space AABB enclosing this torus
    #[must_use]
    pub fn aabb(&self) -> AABB {
        let outer = self.major_radius + self.minor_radius;

        // Local Y axis in world space (symmetry axis)
        let local_y = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let world_y = self.rotation.rotate_vec(local_y);

        // Along the symmetry axis, extent is minor_radius
        let axis_extent = Vec3Fix::new(
            (world_y.x * self.minor_radius).abs(),
            (world_y.y * self.minor_radius).abs(),
            (world_y.z * self.minor_radius).abs(),
        );

        // In the ring plane, extent is major + minor
        // Conservative: use outer on all axes and add axis extent
        let radial_extent = Vec3Fix::new(outer, outer, outer);

        let total = Vec3Fix::new(
            if axis_extent.x > radial_extent.x {
                axis_extent.x
            } else {
                radial_extent.x
            },
            if axis_extent.y > radial_extent.y {
                axis_extent.y
            } else {
                radial_extent.y
            },
            if axis_extent.z > radial_extent.z {
                axis_extent.z
            } else {
                radial_extent.z
            },
        );

        AABB::new(self.center - total, self.center + total)
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// For a solid torus with major radius R and minor radius r:
    /// - Ixx = Izz = m * (5/8 * r^2 + 1/2 * R^2)
    /// - Iyy  = m * (3/4 * r^2 + R^2)
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let r2 = self.minor_radius * self.minor_radius;
        let big_r2 = self.major_radius * self.major_radius;

        let ixx = mass * (Fix128::from_ratio(5, 8) * r2 + Fix128::from_ratio(1, 2) * big_r2);
        let iyy = mass * (Fix128::from_ratio(3, 4) * r2 + big_r2);

        Vec3Fix::new(ixx, iyy, ixx)
    }
}

impl Support for Torus {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Transform direction to local space
        let local_dir = self.rotation.conjugate().rotate_vec(direction);

        // A torus is the Minkowski sum of a circle (major radius R in XZ
        // plane) and a sphere (minor radius r). Therefore:
        //   support(d) = support_circle(d) + support_sphere(d)
        //
        // support_circle(d) = R * normalize(d projected onto XZ)
        // support_sphere(d) = r * normalize(d)

        // 1) Major circle support: project direction onto XZ, normalize,
        //    scale by major radius
        let xz_len_sq = local_dir.x * local_dir.x + local_dir.z * local_dir.z;
        let ring_point = if xz_len_sq.is_zero() {
            // Direction is purely along Y — pick any point on the ring
            Vec3Fix::new(self.major_radius, Fix128::ZERO, Fix128::ZERO)
        } else {
            let xz_len = xz_len_sq.sqrt();
            Vec3Fix::new(
                self.major_radius * local_dir.x / xz_len,
                Fix128::ZERO,
                self.major_radius * local_dir.z / xz_len,
            )
        };

        // 2) Minor sphere support: normalize full direction, scale by
        //    minor radius
        let dir_len = local_dir.length();
        let sphere_offset = if dir_len.is_zero() {
            Vec3Fix::new(self.minor_radius, Fix128::ZERO, Fix128::ZERO)
        } else {
            Vec3Fix::new(
                self.minor_radius * local_dir.x / dir_len,
                self.minor_radius * local_dir.y / dir_len,
                self.minor_radius * local_dir.z / dir_len,
            )
        };

        let local_support = ring_point + sphere_offset;

        // Transform back to world space
        self.center + self.rotation.rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_torus_new() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(2));
        assert_eq!(t.major_radius.hi, 5);
        assert_eq!(t.minor_radius.hi, 2);
    }

    #[test]
    fn test_torus_support_x() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(1));
        let s = t.support(Vec3Fix::UNIT_X);
        // Support in +X: ring point at (5,0,0), extend by 1 in +X → (6,0,0)
        assert_eq!(s.x.hi, 6);
        assert_eq!(s.y.hi, 0);
        assert_eq!(s.z.hi, 0);
    }

    #[test]
    fn test_torus_support_neg_x() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(1));
        let s = t.support(-Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, -6);
    }

    #[test]
    fn test_torus_support_y() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(2));
        let s = t.support(Vec3Fix::UNIT_Y);
        // Direction is along Y. Ring point at (5,0,0), extend by 2 in direction
        // towards (0,1,0) from ring point: to_dir = (-5,1,0), len = sqrt(26)
        // Result: (5 - 2*5/sqrt(26), 2*1/sqrt(26), 0)
        // The y component should be positive
        assert!(s.y > Fix128::ZERO, "Support Y should be positive");
    }

    #[test]
    fn test_torus_volume() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(3), Fix128::from_int(1));
        // V = 2 * pi^2 * 3 * 1^2 = 6*pi^2 ≈ 59.22
        let v = t.volume();
        assert!(
            v.hi >= 59 && v.hi <= 60,
            "Volume should be ~59.22, got {}",
            v.hi
        );
    }

    #[test]
    fn test_torus_surface_area() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(3), Fix128::from_int(1));
        // SA = 4 * pi^2 * 3 * 1 = 12*pi^2 ≈ 118.44
        let sa = t.surface_area();
        assert!(
            sa.hi >= 118 && sa.hi <= 119,
            "Surface area should be ~118.44, got {}",
            sa.hi
        );
    }

    #[test]
    fn test_torus_aabb() {
        let t = Torus::new(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::from_int(5),
            Fix128::from_int(1),
        );
        let aabb = t.aabb();
        // Outer radius = 5+1 = 6. For axis-aligned torus in XZ plane:
        // X: [-6, 6], Y: [-1, 1] (only minor radius in Y), Z: [-6, 6]
        // Conservative AABB will be at least this large
        assert!(aabb.min.x.hi <= -6);
        assert!(aabb.max.x.hi >= 6);
    }

    #[test]
    fn test_torus_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(2));
        // Sphere at (5,0,0) with r=1 overlaps the tube
        let sphere = Sphere::new(Vec3Fix::from_int(5, 0, 0), Fix128::ONE);
        let result = gjk(&t, &sphere);
        assert!(result.colliding, "Sphere inside torus tube should collide");
    }

    #[test]
    fn test_torus_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(5), Fix128::from_int(1));
        let sphere = Sphere::new(Vec3Fix::from_int(20, 0, 0), Fix128::ONE);
        let result = gjk(&t, &sphere);
        assert!(
            !result.colliding,
            "Far sphere should not collide with torus"
        );
    }

    #[test]
    fn test_torus_support_z() {
        let t = Torus::new(Vec3Fix::ZERO, Fix128::from_int(4), Fix128::from_int(1));
        let s = t.support(Vec3Fix::UNIT_Z);
        // Ring point at (0,0,4), extend by 1 in +Z → (0,0,5)
        assert_eq!(s.z.hi, 5);
        assert_eq!(s.x.hi, 0);
    }
}
