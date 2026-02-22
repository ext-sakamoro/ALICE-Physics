//! Oriented Box (OBB) Collider
//!
//! Box collider with arbitrary orientation, supporting GJK/EPA collision detection.
//!
//! # Features
//!
//! - **OBB**: Oriented Bounding Box with half-extents and rotation
//! - **GJK Support**: Implements `Support` trait for GJK/EPA pipeline
//! - **Fast AABB**: Compute world-space AABB from OBB for broadphase

use crate::collider::{Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

/// Oriented Bounding Box (OBB) collider
///
/// A box defined by center position, half-extents (size/2 on each axis),
/// and an orientation quaternion.
#[derive(Clone, Copy, Debug)]
pub struct OrientedBox {
    /// Center position in world space
    pub center: Vec3Fix,
    /// Half-extents (half-size on each local axis)
    pub half_extents: Vec3Fix,
    /// Orientation quaternion
    pub rotation: QuatFix,
}

impl OrientedBox {
    /// Create a new oriented box
    #[inline]
    pub fn new(center: Vec3Fix, half_extents: Vec3Fix, rotation: QuatFix) -> Self {
        Self {
            center,
            half_extents,
            rotation,
        }
    }

    /// Create an axis-aligned box (no rotation)
    #[inline]
    pub fn axis_aligned(center: Vec3Fix, half_extents: Vec3Fix) -> Self {
        Self {
            center,
            half_extents,
            rotation: QuatFix::IDENTITY,
        }
    }

    /// Compute world-space AABB enclosing this OBB
    pub fn aabb(&self) -> AABB {
        // Transform each local axis to world space and compute extents
        let local_x = Vec3Fix::new(self.half_extents.x, Fix128::ZERO, Fix128::ZERO);
        let local_y = Vec3Fix::new(Fix128::ZERO, self.half_extents.y, Fix128::ZERO);
        let local_z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, self.half_extents.z);

        let world_x = self.rotation.rotate_vec(local_x);
        let world_y = self.rotation.rotate_vec(local_y);
        let world_z = self.rotation.rotate_vec(local_z);

        // Extent on each world axis is sum of absolute projections
        let extent = Vec3Fix::new(
            world_x.x.abs() + world_y.x.abs() + world_z.x.abs(),
            world_x.y.abs() + world_y.y.abs() + world_z.y.abs(),
            world_x.z.abs() + world_y.z.abs() + world_z.z.abs(),
        );

        AABB::new(self.center - extent, self.center + extent)
    }

    /// Get a corner vertex by index (0..8)
    pub fn corner(&self, index: usize) -> Vec3Fix {
        let sx = if index & 1 == 0 {
            self.half_extents.x
        } else {
            -self.half_extents.x
        };
        let sy = if index & 2 == 0 {
            self.half_extents.y
        } else {
            -self.half_extents.y
        };
        let sz = if index & 4 == 0 {
            self.half_extents.z
        } else {
            -self.half_extents.z
        };
        let local = Vec3Fix::new(sx, sy, sz);
        self.center + self.rotation.rotate_vec(local)
    }

    /// Get all 8 corner vertices
    pub fn corners(&self) -> [Vec3Fix; 8] {
        let mut result = [Vec3Fix::ZERO; 8];
        for (i, item) in result.iter_mut().enumerate() {
            *item = self.corner(i);
        }
        result
    }

    /// Volume of the box
    #[inline]
    pub fn volume(&self) -> Fix128 {
        let eight = Fix128::from_int(8);
        self.half_extents.x * self.half_extents.y * self.half_extents.z * eight
    }

    /// Surface area of the box
    #[inline]
    pub fn surface_area(&self) -> Fix128 {
        let two = Fix128::from_int(2);
        let hx = self.half_extents.x * two;
        let hy = self.half_extents.y * two;
        let hz = self.half_extents.z * two;
        two * (hx * hy + hy * hz + hz * hx)
    }

    /// Compute inertia tensor (diagonal) for given mass
    pub fn inertia_diagonal(&self, mass: Fix128) -> Vec3Fix {
        let three = Fix128::from_int(3);
        let ex2 = self.half_extents.x * self.half_extents.x;
        let ey2 = self.half_extents.y * self.half_extents.y;
        let ez2 = self.half_extents.z * self.half_extents.z;
        Vec3Fix::new(
            mass * (ey2 + ez2) / three,
            mass * (ex2 + ez2) / three,
            mass * (ex2 + ey2) / three,
        )
    }
}

impl Support for OrientedBox {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        // Transform direction to local space
        let local_dir = self.rotation.conjugate().rotate_vec(direction);

        // In local space, support is simply sign(dir) * half_extents
        let local_support = Vec3Fix::new(
            if local_dir.x >= Fix128::ZERO {
                self.half_extents.x
            } else {
                -self.half_extents.x
            },
            if local_dir.y >= Fix128::ZERO {
                self.half_extents.y
            } else {
                -self.half_extents.y
            },
            if local_dir.z >= Fix128::ZERO {
                self.half_extents.z
            } else {
                -self.half_extents.z
            },
        );

        // Transform back to world space
        self.center + self.rotation.rotate_vec(local_support)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis_aligned_box_support() {
        let b = OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(2, 3, 4));

        let s = b.support(Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, 2);
        assert_eq!(s.y.hi, 3);
        assert_eq!(s.z.hi, 4);

        let s = b.support(-Vec3Fix::UNIT_X);
        assert_eq!(s.x.hi, -2);
    }

    #[test]
    fn test_box_aabb() {
        let b = OrientedBox::axis_aligned(Vec3Fix::from_int(5, 0, 0), Vec3Fix::from_int(1, 1, 1));
        let aabb = b.aabb();
        assert_eq!(aabb.min.x.hi, 4);
        assert_eq!(aabb.max.x.hi, 6);
    }

    #[test]
    fn test_box_corners() {
        let b = OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        let corners = b.corners();
        assert_eq!(corners.len(), 8);

        // All corners should be at distance sqrt(3) from center
        for c in &corners {
            let dist_sq = c.length_squared();
            // 1^2 + 1^2 + 1^2 = 3
            assert_eq!(dist_sq.hi, 3);
        }
    }

    #[test]
    fn test_rotated_box_support() {
        // 90-degree rotation around Y axis
        let rot = QuatFix::from_axis_angle(Vec3Fix::UNIT_Y, Fix128::HALF_PI);
        let b = OrientedBox::new(
            Vec3Fix::ZERO,
            Vec3Fix::new(Fix128::from_int(2), Fix128::ONE, Fix128::ONE),
            rot,
        );

        // After 90° Y rotation, local X axis maps to world -Z
        // So support in +Z direction should give -half_extents.x on local X
        let s = b.support(Vec3Fix::UNIT_Z);
        // The z-support should be roughly 2 (the long axis rotated)
        assert!(
            s.z > Fix128::ONE,
            "Rotated box support should reflect orientation"
        );
    }

    #[test]
    fn test_box_volume() {
        let b = OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 2, 3));
        // Volume = 2*1 * 2*2 * 2*3 = 48
        assert_eq!(b.volume().hi, 48);
    }

    #[test]
    fn test_box_inertia() {
        let b = OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        let inertia = b.inertia_diagonal(Fix128::from_int(6));
        // I_x = m/3 * (hy^2 + hz^2) = 6/3 * (1+1) = 4
        assert_eq!(inertia.x.hi, 4);
    }
}
