//! Height Field Collision
//!
//! Grid-based terrain collider with bilinear interpolation.
//! Efficient for large flat terrain with vertical displacement.

use crate::collider::Contact;
use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Height field terrain collider
///
/// Heights stored as a 2D grid in row-major order (Z-major):
/// `index = x + z * width`
pub struct HeightField {
    /// Height values (Fix128)
    pub heights: Vec<Fix128>,
    /// Grid width (number of columns, X axis)
    pub width: u32,
    /// Grid depth (number of rows, Z axis)
    pub depth: u32,
    /// World-space spacing between grid points
    pub spacing: Fix128,
    /// World-space origin (min corner)
    pub origin: Vec3Fix,
}

impl HeightField {
    /// Create a height field from a grid of heights
    pub fn new(
        heights: Vec<Fix128>,
        width: u32,
        depth: u32,
        spacing: Fix128,
        origin: Vec3Fix,
    ) -> Self {
        debug_assert_eq!(heights.len(), (width * depth) as usize);
        Self {
            heights,
            width,
            depth,
            spacing,
            origin,
        }
    }

    /// Create a flat height field at a given Y level
    pub fn flat(width: u32, depth: u32, spacing: Fix128, origin: Vec3Fix, height: Fix128) -> Self {
        let heights = vec![height; (width * depth) as usize];
        Self::new(heights, width, depth, spacing, origin)
    }

    /// Get height at grid coordinates (clamped to bounds)
    #[inline]
    pub fn get_height(&self, gx: u32, gz: u32) -> Fix128 {
        let gx = gx.min(self.width - 1);
        let gz = gz.min(self.depth - 1);
        self.heights[(gx + gz * self.width) as usize]
    }

    /// Set height at grid coordinates
    #[inline]
    pub fn set_height(&mut self, gx: u32, gz: u32, height: Fix128) {
        if gx < self.width && gz < self.depth {
            self.heights[(gx + gz * self.width) as usize] = height;
        }
    }

    /// World-space to grid coordinates (fractional)
    fn world_to_grid(&self, world_pos: Vec3Fix) -> (Fix128, Fix128) {
        let local_x = world_pos.x - self.origin.x;
        let local_z = world_pos.z - self.origin.z;

        if self.spacing.is_zero() {
            return (Fix128::ZERO, Fix128::ZERO);
        }

        (local_x / self.spacing, local_z / self.spacing)
    }

    /// Sample height at world-space XZ with bilinear interpolation
    pub fn sample_height(&self, world_x: Fix128, world_z: Fix128) -> Fix128 {
        let local_x = world_x - self.origin.x;
        let local_z = world_z - self.origin.z;

        if self.spacing.is_zero() {
            return Fix128::ZERO;
        }

        let gx_f = local_x / self.spacing;
        let gz_f = local_z / self.spacing;

        // Integer grid coords (clamped to valid range for u32 conversion)
        let max_gx = (self.width as i64).saturating_sub(2).max(0);
        let max_gz = (self.depth as i64).saturating_sub(2).max(0);
        let gx0 = if gx_f.is_negative() {
            0u32
        } else {
            (gx_f.hi.min(max_gx).max(0)) as u32
        };
        let gz0 = if gz_f.is_negative() {
            0u32
        } else {
            (gz_f.hi.min(max_gz).max(0)) as u32
        };

        let gx1 = gx0 + 1;
        let gz1 = gz0 + 1;

        // Fractional part for interpolation
        let fx = gx_f - Fix128::from_int(gx0 as i64);
        let fz = gz_f - Fix128::from_int(gz0 as i64);

        // Clamp fractions to [0, 1]
        let fx = clamp01(fx);
        let fz = clamp01(fz);

        // Sample 4 corners
        let h00 = self.get_height(gx0, gz0);
        let h10 = self.get_height(gx1, gz0);
        let h01 = self.get_height(gx0, gz1);
        let h11 = self.get_height(gx1, gz1);

        // Bilinear interpolation
        let one = Fix128::ONE;
        let h0 = h00 * (one - fx) + h10 * fx;
        let h1 = h01 * (one - fx) + h11 * fx;
        h0 * (one - fz) + h1 * fz
    }

    /// Compute surface normal at world-space XZ via central difference
    pub fn sample_normal(&self, world_x: Fix128, world_z: Fix128) -> Vec3Fix {
        let eps = self.spacing.half();
        let hx_neg = self.sample_height(world_x - eps, world_z);
        let hx_pos = self.sample_height(world_x + eps, world_z);
        let hz_neg = self.sample_height(world_x, world_z - eps);
        let hz_pos = self.sample_height(world_x, world_z + eps);

        let dx = hx_pos - hx_neg;
        let dz = hz_pos - hz_neg;

        // Normal = (-dh/dx, 1, -dh/dz) normalized
        // But we need to account for the spacing
        let inv_eps2 = Fix128::ONE / eps.double();
        Vec3Fix::new(-(dx * inv_eps2), Fix128::ONE, -(dz * inv_eps2)).normalize()
    }

    /// Sphere vs HeightField collision
    pub fn collide_sphere(&self, center: Vec3Fix, radius: Fix128) -> Option<Contact> {
        let (gx_f, gz_f) = self.world_to_grid(center);

        // Check if within bounds (with margin)
        let margin = Fix128::from_int(2);
        if gx_f < -margin || gx_f > Fix128::from_int(self.width as i64) + margin {
            return None;
        }
        if gz_f < -margin || gz_f > Fix128::from_int(self.depth as i64) + margin {
            return None;
        }

        let ground_height = self.sample_height(center.x, center.z);
        let dist = center.y - ground_height;

        if dist < radius {
            let depth = radius - dist;
            let normal = self.sample_normal(center.x, center.z);
            let point_on_surface = Vec3Fix::new(center.x, ground_height, center.z);

            Some(Contact {
                depth,
                normal,
                point_a: center - normal * radius,
                point_b: point_on_surface,
            })
        } else {
            None
        }
    }

    /// Point vs HeightField: get signed distance (positive = above)
    #[inline]
    pub fn signed_distance(&self, point: Vec3Fix) -> Fix128 {
        let ground_height = self.sample_height(point.x, point.z);
        point.y - ground_height
    }

    /// Get world-space AABB of the height field
    pub fn aabb(&self) -> crate::collider::AABB {
        let mut min_h = self.heights[0];
        let mut max_h = self.heights[0];
        for &h in &self.heights[1..] {
            if h < min_h {
                min_h = h;
            }
            if h > max_h {
                max_h = h;
            }
        }

        let max_x = self.origin.x + self.spacing * Fix128::from_int((self.width - 1) as i64);
        let max_z = self.origin.z + self.spacing * Fix128::from_int((self.depth - 1) as i64);

        crate::collider::AABB::new(
            Vec3Fix::new(self.origin.x, min_h, self.origin.z),
            Vec3Fix::new(max_x, max_h, max_z),
        )
    }
}

#[inline]
fn clamp01(v: Fix128) -> Fix128 {
    if v < Fix128::ZERO {
        Fix128::ZERO
    } else if v > Fix128::ONE {
        Fix128::ONE
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_heightfield() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
        let h = hf.sample_height(Fix128::from_int(5), Fix128::from_int(5));
        assert!(
            h.abs() < Fix128::from_ratio(1, 10),
            "Flat field should have height ~0"
        );
    }

    #[test]
    fn test_heightfield_interpolation() {
        // Create a simple slope: heights increase with X
        let mut heights = Vec::new();
        for z in 0..4u32 {
            for x in 0..4u32 {
                let _ = z;
                heights.push(Fix128::from_int(x as i64));
            }
        }
        let hf = HeightField::new(heights, 4, 4, Fix128::ONE, Vec3Fix::ZERO);

        // At x=0, height should be 0
        let h0 = hf.sample_height(Fix128::ZERO, Fix128::ONE);
        assert!(h0.abs() < Fix128::ONE, "Height at x=0 should be ~0");

        // At x=2, height should be ~2
        let h2 = hf.sample_height(Fix128::from_int(2), Fix128::ONE);
        let error = (h2 - Fix128::from_int(2)).abs();
        assert!(error < Fix128::ONE, "Height at x=2 should be ~2");
    }

    #[test]
    fn test_sphere_heightfield_collision() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);

        // Sphere penetrating ground
        let contact = hf.collide_sphere(
            Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_ratio(1, 2),
                Fix128::from_int(5),
            ),
            Fix128::ONE,
        );

        assert!(
            contact.is_some(),
            "Sphere at y=0.5 with r=1 should penetrate ground at y=0"
        );
        let c = contact.unwrap();
        assert!(c.depth > Fix128::ZERO);
    }

    #[test]
    fn test_sphere_above_heightfield() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);

        let contact = hf.collide_sphere(
            Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(5),
                Fix128::from_int(5),
            ),
            Fix128::ONE,
        );

        assert!(
            contact.is_none(),
            "Sphere at y=5 should not collide with ground"
        );
    }

    #[test]
    fn test_normal_on_flat() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);
        let normal = hf.sample_normal(Fix128::from_int(5), Fix128::from_int(5));

        // On flat terrain, normal should point up (0, 1, 0)
        assert!(
            normal.y > Fix128::from_ratio(9, 10),
            "Normal should point up on flat terrain"
        );
    }

    #[test]
    fn test_signed_distance() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::ZERO);

        let above = hf.signed_distance(Vec3Fix::from_int(5, 3, 5));
        assert!(
            above > Fix128::ZERO,
            "Point above should have positive distance"
        );

        let below = hf.signed_distance(Vec3Fix::from_int(5, -2, 5));
        assert!(
            below < Fix128::ZERO,
            "Point below should have negative distance"
        );
    }

    #[test]
    fn test_aabb() {
        let hf = HeightField::flat(10, 10, Fix128::ONE, Vec3Fix::ZERO, Fix128::from_int(5));
        let aabb = hf.aabb();
        assert_eq!(aabb.min.y.hi, 5);
        assert_eq!(aabb.max.y.hi, 5);
    }
}
