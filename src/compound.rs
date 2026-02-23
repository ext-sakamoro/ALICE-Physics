//! Compound Shape
//!
//! Combines multiple collision shapes into a single rigid body.
//! Each child shape has a local offset (position + rotation) relative to the body.
//!
//! # Usage
//!
//! ```
//! use alice_physics::compound::CompoundShape;
//! use alice_physics::{Vec3Fix, Fix128};
//! use alice_physics::collider::Sphere;
//! use alice_physics::math::QuatFix;
//!
//! let mut compound = CompoundShape::new();
//! compound.add_sphere(Sphere::new(Vec3Fix::ZERO, Fix128::ONE), Vec3Fix::ZERO, QuatFix::IDENTITY);
//! compound.add_sphere(Sphere::new(Vec3Fix::ZERO, Fix128::ONE), Vec3Fix::from_int(3, 0, 0), QuatFix::IDENTITY);
//! ```

use crate::box_collider::OrientedBox;
use crate::collider::{Capsule, Sphere, Support, AABB};
use crate::math::{Fix128, QuatFix, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Enumeration of supported child shape types
#[derive(Clone, Debug)]
pub enum ShapeRef {
    /// Sphere child
    Sphere(Sphere),
    /// Capsule child
    Capsule(Capsule),
    /// Oriented box child
    Box(OrientedBox),
}

impl ShapeRef {
    /// Compute AABB of this shape
    #[must_use]
    pub fn aabb(&self) -> AABB {
        match self {
            ShapeRef::Sphere(s) => {
                let r = Vec3Fix::new(s.radius, s.radius, s.radius);
                AABB::new(s.center - r, s.center + r)
            }
            ShapeRef::Capsule(c) => {
                let r = Vec3Fix::new(c.radius, c.radius, c.radius);
                let min = Vec3Fix::new(
                    if c.a.x < c.b.x { c.a.x } else { c.b.x },
                    if c.a.y < c.b.y { c.a.y } else { c.b.y },
                    if c.a.z < c.b.z { c.a.z } else { c.b.z },
                );
                let max = Vec3Fix::new(
                    if c.a.x > c.b.x { c.a.x } else { c.b.x },
                    if c.a.y > c.b.y { c.a.y } else { c.b.y },
                    if c.a.z > c.b.z { c.a.z } else { c.b.z },
                );
                AABB::new(min - r, max + r)
            }
            ShapeRef::Box(b) => b.aabb(),
        }
    }
}

/// A child shape within a compound, with local transform
#[derive(Clone, Debug)]
pub struct CompoundChild {
    /// The shape
    pub shape: ShapeRef,
    /// Local position offset from compound center
    pub local_position: Vec3Fix,
    /// Local rotation offset
    pub local_rotation: QuatFix,
}

/// Compound shape: multiple child shapes combined into one body
#[derive(Clone, Debug)]
pub struct CompoundShape {
    /// Child shapes with their local transforms
    pub children: Vec<CompoundChild>,
    /// Cached world-space AABB (recomputed on transform)
    cached_aabb: AABB,
    /// Whether the cached AABB needs recomputation
    dirty: bool,
}

impl CompoundShape {
    /// Create an empty compound shape
    #[must_use]
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            cached_aabb: AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO),
            dirty: true,
        }
    }

    /// Add a sphere child
    pub fn add_sphere(&mut self, sphere: Sphere, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Sphere(sphere),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Add a capsule child
    pub fn add_capsule(&mut self, capsule: Capsule, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Capsule(capsule),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Add a box child
    pub fn add_box(&mut self, obox: OrientedBox, position: Vec3Fix, rotation: QuatFix) {
        self.children.push(CompoundChild {
            shape: ShapeRef::Box(obox),
            local_position: position,
            local_rotation: rotation,
        });
        self.dirty = true;
    }

    /// Number of children
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Compute the AABB enclosing all children (in local space)
    pub fn compute_aabb(&mut self) -> AABB {
        if self.children.is_empty() {
            return AABB::new(Vec3Fix::ZERO, Vec3Fix::ZERO);
        }

        let first = self.child_world_aabb(0, Vec3Fix::ZERO, QuatFix::IDENTITY);
        let mut result = first;

        for i in 1..self.children.len() {
            let child_aabb = self.child_world_aabb(i, Vec3Fix::ZERO, QuatFix::IDENTITY);
            result = result.union(&child_aabb);
        }

        self.cached_aabb = result;
        self.dirty = false;
        result
    }

    /// Get world-space AABB for a specific child, given body transform
    #[must_use]
    pub fn child_world_aabb(&self, child_idx: usize, body_pos: Vec3Fix, body_rot: QuatFix) -> AABB {
        let child = &self.children[child_idx];
        let world_pos = body_pos + body_rot.rotate_vec(child.local_position);

        match &child.shape {
            ShapeRef::Sphere(s) => {
                let r = Vec3Fix::new(s.radius, s.radius, s.radius);
                let center = world_pos + body_rot.rotate_vec(s.center);
                AABB::new(center - r, center + r)
            }
            ShapeRef::Capsule(c) => {
                let child_rot = body_rot.mul(child.local_rotation);
                let a = world_pos + child_rot.rotate_vec(c.a);
                let b = world_pos + child_rot.rotate_vec(c.b);
                let r = Vec3Fix::new(c.radius, c.radius, c.radius);
                let min = Vec3Fix::new(
                    if a.x < b.x { a.x } else { b.x },
                    if a.y < b.y { a.y } else { b.y },
                    if a.z < b.z { a.z } else { b.z },
                );
                let max = Vec3Fix::new(
                    if a.x > b.x { a.x } else { b.x },
                    if a.y > b.y { a.y } else { b.y },
                    if a.z > b.z { a.z } else { b.z },
                );
                AABB::new(min - r, max + r)
            }
            ShapeRef::Box(ob) => {
                let child_rot = body_rot.mul(child.local_rotation);
                let transformed = OrientedBox::new(
                    world_pos + child_rot.rotate_vec(ob.center),
                    ob.half_extents,
                    child_rot.mul(ob.rotation),
                );
                transformed.aabb()
            }
        }
    }

    /// Get world-space AABB for entire compound given body transform
    #[must_use]
    pub fn world_aabb(&self, body_pos: Vec3Fix, body_rot: QuatFix) -> AABB {
        if self.children.is_empty() {
            return AABB::new(body_pos, body_pos);
        }

        let mut result = self.child_world_aabb(0, body_pos, body_rot);
        for i in 1..self.children.len() {
            let child_aabb = self.child_world_aabb(i, body_pos, body_rot);
            result = result.union(&child_aabb);
        }
        result
    }

    /// Support function for the compound shape (selects the child with maximum support)
    #[must_use]
    pub fn support_world(
        &self,
        direction: Vec3Fix,
        body_pos: Vec3Fix,
        body_rot: QuatFix,
    ) -> Vec3Fix {
        if self.children.is_empty() {
            return body_pos;
        }
        let mut best = Vec3Fix::ZERO;
        let mut best_dot = Fix128::from_int(-1000000);

        for child in &self.children {
            let child_rot = body_rot.mul(child.local_rotation);
            let child_pos = body_pos + body_rot.rotate_vec(child.local_position);

            let s = match &child.shape {
                ShapeRef::Sphere(sphere) => {
                    let shifted = Sphere::new(
                        child_pos + child_rot.rotate_vec(sphere.center),
                        sphere.radius,
                    );
                    shifted.support(direction)
                }
                ShapeRef::Capsule(cap) => {
                    let a = child_pos + child_rot.rotate_vec(cap.a);
                    let b = child_pos + child_rot.rotate_vec(cap.b);
                    let shifted = Capsule::new(a, b, cap.radius);
                    shifted.support(direction)
                }
                ShapeRef::Box(ob) => {
                    let shifted = OrientedBox::new(
                        child_pos + child_rot.rotate_vec(ob.center),
                        ob.half_extents,
                        child_rot.mul(ob.rotation),
                    );
                    shifted.support(direction)
                }
            };

            let d = s.dot(direction);
            if d > best_dot {
                best_dot = d;
                best = s;
            }
        }

        best
    }
}

impl Default for CompoundShape {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for using `CompoundShape` with GJK (needs body transform context)
pub struct TransformedCompound<'a> {
    /// Reference to the compound shape
    pub compound: &'a CompoundShape,
    /// Body position
    pub position: Vec3Fix,
    /// Body rotation
    pub rotation: QuatFix,
}

impl Support for TransformedCompound<'_> {
    fn support(&self, direction: Vec3Fix) -> Vec3Fix {
        self.compound
            .support_world(direction, self.position, self.rotation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compound_basic() {
        let mut compound = CompoundShape::new();
        assert!(compound.is_empty());

        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(5, 0, 0),
            QuatFix::IDENTITY,
        );

        assert_eq!(compound.len(), 2);
    }

    #[test]
    fn test_compound_aabb() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );

        let aabb = compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
        // Should span from -1 to 11 on X
        assert!(aabb.min.x <= -Fix128::ONE);
        assert!(aabb.max.x >= Fix128::from_int(11));
    }

    #[test]
    fn test_compound_support() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(10, 0, 0),
            QuatFix::IDENTITY,
        );

        // Support in +X should come from the far sphere (center=10, radius=1 → 11)
        let s = compound.support_world(Vec3Fix::UNIT_X, Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(
            s.x >= Fix128::from_int(10),
            "Support should be from far sphere"
        );
    }

    #[test]
    fn test_compound_with_box() {
        let mut compound = CompoundShape::new();
        compound.add_box(
            OrientedBox::axis_aligned(Vec3Fix::ZERO, Vec3Fix::from_int(1, 1, 1)),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        );
        compound.add_capsule(
            Capsule::new(
                Vec3Fix::from_int(0, -2, 0),
                Vec3Fix::from_int(0, 2, 0),
                Fix128::from_ratio(5, 10),
            ),
            Vec3Fix::from_int(3, 0, 0),
            QuatFix::IDENTITY,
        );

        assert_eq!(compound.len(), 2);

        let aabb = compound.world_aabb(Vec3Fix::ZERO, QuatFix::IDENTITY);
        assert!(aabb.min.y <= -Fix128::from_int(2));
    }

    #[test]
    fn test_transformed_compound_support() {
        let mut compound = CompoundShape::new();
        compound.add_sphere(
            Sphere::new(Vec3Fix::ZERO, Fix128::ONE),
            Vec3Fix::from_int(5, 0, 0),
            QuatFix::IDENTITY,
        );

        let tc = TransformedCompound {
            compound: &compound,
            position: Vec3Fix::from_int(10, 0, 0),
            rotation: QuatFix::IDENTITY,
        };

        let s = tc.support(Vec3Fix::UNIT_X);
        // Body at 10, child offset 5, sphere radius 1 → 16
        assert!(s.x >= Fix128::from_int(15));
    }
}
