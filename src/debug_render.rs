//! Debug Visualization API
//!
//! Abstract debug rendering interface for visualizing physics state.
//! Outputs wireframe geometry (lines, points) that can be rendered
//! by any graphics backend.
//!
//! # Usage
//!
//! Implement `DebugRenderer` trait for your graphics backend, then call
//! `debug_draw_world()` each frame.

use crate::collider::AABB;
use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// RGBA color for debug rendering (0-255 per channel)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DebugColor {
    /// Red channel
    pub r: u8,
    /// Green channel
    pub g: u8,
    /// Blue channel
    pub b: u8,
    /// Alpha channel
    pub a: u8,
}

impl DebugColor {
    /// Create a new color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Predefined colors
    pub const RED: Self = Self::new(255, 50, 50, 255);
    /// Green color
    pub const GREEN: Self = Self::new(50, 255, 50, 255);
    /// Blue color
    pub const BLUE: Self = Self::new(50, 50, 255, 255);
    /// Yellow color
    pub const YELLOW: Self = Self::new(255, 255, 50, 255);
    /// Cyan color
    pub const CYAN: Self = Self::new(50, 255, 255, 255);
    /// Magenta color
    pub const MAGENTA: Self = Self::new(255, 50, 255, 255);
    /// White color
    pub const WHITE: Self = Self::new(255, 255, 255, 255);
    /// Gray color
    pub const GRAY: Self = Self::new(128, 128, 128, 255);
    /// Orange color
    pub const ORANGE: Self = Self::new(255, 165, 0, 255);
}

/// A debug line segment
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugLine {
    /// Start point
    pub start: Vec3Fix,
    /// End point
    pub end: Vec3Fix,
    /// Color
    pub color: DebugColor,
}

/// A debug point
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugPoint {
    /// Position
    pub position: Vec3Fix,
    /// Color
    pub color: DebugColor,
    /// Size (for rendering)
    pub size: Fix128,
}

/// What to draw in debug mode
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct DebugDrawFlags {
    /// Draw body AABBs
    pub draw_aabbs: bool,
    /// Draw body centers of mass
    pub draw_centers: bool,
    /// Draw velocity vectors
    pub draw_velocities: bool,
    /// Draw contact points
    pub draw_contacts: bool,
    /// Draw contact normals
    pub draw_contact_normals: bool,
    /// Draw joint connections
    pub draw_joints: bool,
    /// Draw BVH nodes
    pub draw_bvh: bool,
    /// Draw body axes (local coordinate frame)
    pub draw_axes: bool,
}

impl Default for DebugDrawFlags {
    fn default() -> Self {
        Self {
            draw_aabbs: true,
            draw_centers: true,
            draw_velocities: false,
            draw_contacts: true,
            draw_contact_normals: true,
            draw_joints: true,
            draw_bvh: false,
            draw_axes: false,
        }
    }
}

/// Collected debug geometry for a frame
#[derive(Clone, Debug, Default)]
pub struct DebugDrawData {
    /// Line segments to draw
    pub lines: Vec<DebugLine>,
    /// Points to draw
    pub points: Vec<DebugPoint>,
}

impl DebugDrawData {
    /// Create empty draw data
    #[must_use]
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            points: Vec::new(),
        }
    }

    /// Clear all geometry
    pub fn clear(&mut self) {
        self.lines.clear();
        self.points.clear();
    }

    /// Add a line
    #[inline]
    pub fn line(&mut self, start: Vec3Fix, end: Vec3Fix, color: DebugColor) {
        self.lines.push(DebugLine { start, end, color });
    }

    /// Add a point
    #[inline]
    pub fn point(&mut self, position: Vec3Fix, color: DebugColor, size: Fix128) {
        self.points.push(DebugPoint {
            position,
            color,
            size,
        });
    }

    /// Draw an AABB wireframe (12 edges)
    pub fn aabb(&mut self, aabb: &AABB, color: DebugColor) {
        let min = aabb.min;
        let max = aabb.max;

        let corners = [
            Vec3Fix::new(min.x, min.y, min.z),
            Vec3Fix::new(max.x, min.y, min.z),
            Vec3Fix::new(max.x, max.y, min.z),
            Vec3Fix::new(min.x, max.y, min.z),
            Vec3Fix::new(min.x, min.y, max.z),
            Vec3Fix::new(max.x, min.y, max.z),
            Vec3Fix::new(max.x, max.y, max.z),
            Vec3Fix::new(min.x, max.y, max.z),
        ];

        // Bottom face
        self.line(corners[0], corners[1], color);
        self.line(corners[1], corners[2], color);
        self.line(corners[2], corners[3], color);
        self.line(corners[3], corners[0], color);
        // Top face
        self.line(corners[4], corners[5], color);
        self.line(corners[5], corners[6], color);
        self.line(corners[6], corners[7], color);
        self.line(corners[7], corners[4], color);
        // Verticals
        self.line(corners[0], corners[4], color);
        self.line(corners[1], corners[5], color);
        self.line(corners[2], corners[6], color);
        self.line(corners[3], corners[7], color);
    }

    /// Draw a sphere wireframe (3 rings: XY, XZ, YZ)
    pub fn sphere(&mut self, center: Vec3Fix, radius: Fix128, color: DebugColor) {
        let segments = 16;
        self.draw_circle(
            center,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Y,
            radius,
            segments,
            color,
        );
        self.draw_circle(
            center,
            Vec3Fix::UNIT_X,
            Vec3Fix::UNIT_Z,
            radius,
            segments,
            color,
        );
        self.draw_circle(
            center,
            Vec3Fix::UNIT_Y,
            Vec3Fix::UNIT_Z,
            radius,
            segments,
            color,
        );
    }

    /// Draw a circle (ring)
    fn draw_circle(
        &mut self,
        center: Vec3Fix,
        axis_a: Vec3Fix,
        axis_b: Vec3Fix,
        radius: Fix128,
        segments: usize,
        color: DebugColor,
    ) {
        let mut prev = center + axis_a * radius;
        for i in 1..=segments {
            let angle = Fix128::TWO_PI * Fix128::from_ratio(i as i64, segments as i64);
            let c = angle.cos();
            let s = angle.sin();
            let point = center + axis_a * (radius * c) + axis_b * (radius * s);
            self.line(prev, point, color);
            prev = point;
        }
    }

    /// Draw an arrow (line + arrowhead)
    pub fn arrow(&mut self, start: Vec3Fix, end: Vec3Fix, color: DebugColor) {
        self.line(start, end, color);

        let dir = end - start;
        let len = dir.length();
        if len.is_zero() {
            return;
        }

        let head_len = len * Fix128::from_ratio(2, 10);
        let head_point = end - dir.normalize() * head_len;
        self.point(end, color, head_len);
        let _ = head_point; // Arrow tip marker
    }

    /// Draw local coordinate axes at a position
    pub fn axes(&mut self, position: Vec3Fix, rotation: QuatFix, scale: Fix128) {
        let x = rotation.rotate_vec(Vec3Fix::UNIT_X) * scale;
        let y = rotation.rotate_vec(Vec3Fix::UNIT_Y) * scale;
        let z = rotation.rotate_vec(Vec3Fix::UNIT_Z) * scale;

        self.arrow(position, position + x, DebugColor::RED);
        self.arrow(position, position + y, DebugColor::GREEN);
        self.arrow(position, position + z, DebugColor::BLUE);
    }

    /// Total primitive count
    #[must_use]
    pub fn primitive_count(&self) -> usize {
        self.lines.len() + self.points.len()
    }
}

/// Draw debug visualization for a physics world
pub fn debug_draw_world(
    world: &crate::solver::PhysicsWorld,
    flags: &DebugDrawFlags,
    data: &mut DebugDrawData,
) {
    data.clear();

    for body in &world.bodies {
        let is_static = body.is_static();
        let body_color = if is_static {
            DebugColor::GRAY
        } else {
            DebugColor::GREEN
        };

        if flags.draw_centers {
            data.point(body.position, body_color, Fix128::from_ratio(1, 10));
        }

        if flags.draw_velocities && !is_static {
            let vel_end = body.position + body.velocity;
            data.arrow(body.position, vel_end, DebugColor::YELLOW);
        }

        if flags.draw_axes {
            data.axes(body.position, body.rotation, Fix128::ONE);
        }
    }

    if flags.draw_contacts {
        for contact in &world.contact_constraints {
            let c = &contact.contact;
            data.point(c.point_a, DebugColor::RED, Fix128::from_ratio(5, 100));
            data.point(c.point_b, DebugColor::BLUE, Fix128::from_ratio(5, 100));

            if flags.draw_contact_normals {
                data.arrow(c.point_a, c.point_a + c.normal * c.depth, DebugColor::RED);
            }
        }
    }

    if flags.draw_joints {
        // Draw joint connections as colored lines
        for constraint in &world.distance_constraints {
            let pos_a = world.bodies[constraint.body_a].position;
            let pos_b = world.bodies[constraint.body_b].position;
            data.line(pos_a, pos_b, DebugColor::CYAN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_draw_data() {
        let mut data = DebugDrawData::new();
        data.line(Vec3Fix::ZERO, Vec3Fix::UNIT_X, DebugColor::RED);
        data.point(Vec3Fix::ZERO, DebugColor::GREEN, Fix128::ONE);

        assert_eq!(data.lines.len(), 1);
        assert_eq!(data.points.len(), 1);
        assert_eq!(data.primitive_count(), 2);

        data.clear();
        assert_eq!(data.primitive_count(), 0);
    }

    #[test]
    fn test_debug_aabb() {
        let mut data = DebugDrawData::new();
        let aabb = AABB::new(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1));
        data.aabb(&aabb, DebugColor::WHITE);
        assert_eq!(data.lines.len(), 12); // 12 edges of a box
    }

    #[test]
    fn test_debug_sphere() {
        let mut data = DebugDrawData::new();
        data.sphere(Vec3Fix::ZERO, Fix128::ONE, DebugColor::GREEN);
        // 3 rings × 16 segments = 48 lines
        assert_eq!(data.lines.len(), 48);
    }

    #[test]
    fn test_debug_axes() {
        let mut data = DebugDrawData::new();
        data.axes(Vec3Fix::ZERO, QuatFix::IDENTITY, Fix128::ONE);
        // 3 arrows = 3 lines + 3 points
        assert_eq!(data.lines.len(), 3);
        assert_eq!(data.points.len(), 3);
    }

    #[test]
    fn test_debug_flags_default() {
        let flags = DebugDrawFlags::default();
        assert!(flags.draw_aabbs);
        assert!(flags.draw_contacts);
        assert!(!flags.draw_bvh);
    }

    #[test]
    fn test_debug_color() {
        assert_eq!(DebugColor::RED, DebugColor::new(255, 50, 50, 255));
    }
}
