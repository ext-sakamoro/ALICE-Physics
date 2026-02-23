//! Cone Collider
//!
//! Cone collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **Cone**: Y-axis aligned cone with apex at +Y and base circle at -Y
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Fast AABB**: Compute world-space AABB from oriented cone
//!
//! # Geometry
//!
//! The cone is defined by a center (midpoint between apex and base center),
//! half-height (distance from center to apex or base), and base radius.
//! In local space the apex is at `(0, +half_height, 0)` and the base
//! circle center is at `(0, -half_height, 0)`.
//!
//! Author: Moroya Sakamoto

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Cone collider
///
/// A cone defined by center position, half-height (from center to apex along
/// local Y axis), base radius, and an orientation quaternion. The apex sits
/// at local `+Y` and the base circle at local `-Y`.
#[derive(Clone, Copy, Debug)]
pub struct Cone {
    /// Center position in world space (midpoint between apex and base center)
    pub center: Vec3Fix,
    /// Base radius of the circular cross-section
    pub radius: Fix128,
    /// Half-height (from center to apex along local Y axis)
    pub half_height: Fix128,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl Cone {
    /// Create a new axis-aligned cone (Y-up, apex at +Y)
    #[inline]
    #[must_use]
    pub fn new(center: Vec3Fix, radius: Fix128, half_height: Fix128) -> Self {
        Self {
            center,
            radius,
            half_height,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Create a new oriented cone
    #[inline]
    #[must_use]
    pub fn with_rotation(
        center: Vec3Fix,
        radius: Fix128,
        half_height: Fix128,
        rotation: QuatFix,
    ) -> Self {
        Self {
            center,
            radius,
            half_height,
            rotation,
        }
    }

    /// World-space position of the apex (+Y in local space)
    #[must_use]
    pub fn apex(&self) -> Vec3Fix {
        let local_apex = Vec3Fix::new(Fix128::ZERO, self.half_height, Fix128::ZERO);
        self.center + self.rotation.rotate_vec(local_apex)
    }

    /// World-space position of the base circle center (-Y in local space)
    #[must_use]
    pub fn base_center(&self) -> Vec3Fix {
        let local_base = Vec3Fix::new(Fix128::ZERO, -self.half_height, Fix128::ZERO);
        self.center + self.rotation.rotate_vec(local_base)
    }

    /// Compute world-space AABB enclosing this cone
    #[must_use]
    pub fn aabb(&self) -> AABB {
        // Local Y axis in world space
        let local_y = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let world_y = self.rotation.rotate_vec(local_y);

        // Axis extent covers both apex and base
        let axis_extent = Vec3Fix::new(
            (world_y.x * self.half_height).abs(),
            (world_y.y * self.half_height).abs(),
            (world_y.z * self.half_height).abs(),
        );

        // Conservative radial extent (base circle)
        let radial_extent = Vec3Fix::new(self.radius, self.radius, self.radius);

        let total = Vec3Fix::new(
            axis_extent.x + radial_extent.x,
            axis_extent.y + radial_extent.y,
            axis_extent.z + radial_extent.z,
        );

        AABB::new(self.center - total, self.center + total)
    }

    /// Volume of the cone: (1/3) * pi * r^2 * h where h = 2 * `half_height`
    #[inline]
    #[must_use]
    pub fn volume(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let two = Fix128::from_int(2);
        let three = Fix128::from_int(3);
        pi * self.radius * self.radius * two * self.half_height / three
    }

    /// Surface area: pi * r * (r + sqrt(r^2 + h^2))
    #[must_use]
    pub fn surface_area(&self) -> Fix128 {
        let pi = Fix128::from_ratio(355, 113);
        let two = Fix128::from_int(2);
        let h = two * self.half_height;
        let slant = (self.radius * self.radius + h * h).sqrt();
        pi * self.radius * (self.radius + slant)
    }

    /// Compute inertia tensor (diagonal) for given mass
    ///
    /// For a solid cone aligned along Y with height h = 2*`half_height`:
    /// - Ixx = Izz = m * (3/80 * r^2 + 3/20 * h^2)
    /// - Iyy = 3/10 * m * r^2
    ///
    /// Note: center of mass offset is not applied here; inertia is about
    /// the geometric center.
    #[must_use]
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let r2 = self.radius * self.radius;
        let two = Fix128::from_int(2);
        let h = two * self.half_height;
        let h2 = h * h;

        let ixx = mass * (Fix128::from_ratio(3, 80) * r2 + Fix128::from_ratio(3, 20) * h2);
        let iyy = Fix128::from_ratio(3, 10) * mass * r2;

        Vec3Fix::new(ixx, iyy, ixx)
    }
}

impl Support for Cone {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Transform direction to local space
        let local_dir = self.rotation.conjugate().rotate_vec(direction);

        // The cone has:
        //   apex at (0, half_height, 0)
        //   base circle at (0, -half_height, 0) with given radius
        //
        // Support is the max of:
        //   1) The apex point
        //   2) The furthest point on the base circle edge

        // Dot product of apex with local direction
        let apex_dot = local_dir.y * self.half_height;

        // Furthest point on base circle: project direction onto XZ plane,
        // normalize, scale by radius, set Y = -half_height
        let xz_len_sq = local_dir.x * local_dir.x + local_dir.z * local_dir.z;
        let base_support = if xz_len_sq.is_zero() {
            // Direction is purely along Y — any point on circle edge works
            Vec3Fix::new(self.radius, -self.half_height, Fix128::ZERO)
        } else {
            let xz_len = xz_len_sq.sqrt();
            Vec3Fix::new(
                self.radius * local_dir.x / xz_len,
                -self.half_height,
                self.radius * local_dir.z / xz_len,
            )
        };

        let base_dot = base_support.dot(local_dir);

        let local_support = if apex_dot >= base_dot {
            Vec3Fix::new(Fix128::ZERO, self.half_height, Fix128::ZERO)
        } else {
            base_support
        };

        // Transform back to world space
        self.center + self.rotation.rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_new() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::from_int(3));
        assert_eq!(cone.center.x.hi, 0);
        assert_eq!(cone.radius.hi, 2);
        assert_eq!(cone.half_height.hi, 3);
    }

    #[test]
    fn test_cone_apex() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::from_int(5));
        let apex = cone.apex();
        assert_eq!(apex.x.hi, 0);
        assert_eq!(apex.y.hi, 5);
        assert_eq!(apex.z.hi, 0);
    }

    #[test]
    fn test_cone_base_center() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::from_int(5));
        let base = cone.base_center();
        assert_eq!(base.x.hi, 0);
        assert_eq!(base.y.hi, -5);
        assert_eq!(base.z.hi, 0);
    }

    #[test]
    fn test_cone_support_up() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::from_int(3));
        let s = cone.support(Vec3Fix::UNIT_Y);
        // Support in +Y should be the apex
        assert_eq!(s.y.hi, 3);
        assert_eq!(s.x.hi, 0);
    }

    #[test]
    fn test_cone_support_down() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::from_int(3));
        let s = cone.support(-Vec3Fix::UNIT_Y);
        // Support in -Y should be on the base circle edge
        assert_eq!(s.y.hi, -3);
    }

    #[test]
    fn test_cone_support_side() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::from_int(4), Fix128::from_int(1));
        let s = cone.support(Vec3Fix::UNIT_X);
        // Support in +X: base circle point at (radius, -half_height, 0)
        // vs apex at (0, half_height, 0). Dot with UNIT_X: radius vs 0.
        // Since radius(4) > 0, base circle wins.
        assert_eq!(s.x.hi, 4);
        assert_eq!(s.y.hi, -1);
    }

    #[test]
    fn test_cone_aabb() {
        let cone = Cone::new(
            Vec3Fix::from_int(5, 0, 0),
            Fix128::from_int(2),
            Fix128::from_int(3),
        );
        let aabb = cone.aabb();
        // Y-up: axis_extent=(0,3,0), radial=(2,2,2), total=(2,5,2)
        assert_eq!(aabb.min.x.hi, 3); // 5 - 2
        assert_eq!(aabb.max.x.hi, 7); // 5 + 2
        assert_eq!(aabb.min.y.hi, -5); // 0 - 5
        assert_eq!(aabb.max.y.hi, 5); // 0 + 5
    }

    #[test]
    fn test_cone_volume() {
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        // V = (1/3) * pi * 1^2 * 2*1 = 2*pi/3 ≈ 2.094
        let v = cone.volume();
        assert!(
            v.hi >= 2 && v.hi <= 3,
            "Volume should be ~2.094, got {}",
            v.hi
        );
    }

    #[test]
    fn test_cone_gjk_collision() {
        use crate::collider::{gjk, Sphere};
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::from_int(2), Fix128::from_int(2));
        let sphere = Sphere::new(Vec3Fix::ZERO, Fix128::ONE);
        let result = gjk(&cone, &sphere);
        assert!(
            result.colliding,
            "Overlapping cone and sphere should collide"
        );
    }

    #[test]
    fn test_cone_gjk_no_collision() {
        use crate::collider::{gjk, Sphere};
        let cone = Cone::new(Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE);
        let sphere = Sphere::new(Vec3Fix::from_int(10, 0, 0), Fix128::ONE);
        let result = gjk(&cone, &sphere);
        assert!(
            !result.colliding,
            "Separated cone and sphere should not collide"
        );
    }

    #[test]
    fn test_cone_offset_center() {
        let cone = Cone::new(
            Vec3Fix::from_int(10, 20, 30),
            Fix128::ONE,
            Fix128::from_int(5),
        );
        let apex = cone.apex();
        assert_eq!(apex.x.hi, 10);
        assert_eq!(apex.y.hi, 25);
        assert_eq!(apex.z.hi, 30);

        let base = cone.base_center();
        assert_eq!(base.x.hi, 10);
        assert_eq!(base.y.hi, 15);
        assert_eq!(base.z.hi, 30);
    }
}
