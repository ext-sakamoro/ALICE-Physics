//! Immovable collision surfaces for a [`PhysicsWorld`](crate::solver::PhysicsWorld).
//!
//! A [`StaticCollider`] is a plane, a height field or a triangle mesh that bodies
//! rest on. It is the counterpart of the SDF colliders for explicit geometry: the
//! world resolves each non-static, non-sensor body's collision sphere against
//! every collider after the substep's collision detection, pushing the body out
// LIMITATION(COV-RIGID-065): world resolves each non-static, non-sensor body's collision sphere against every collider after the substep's collision detection
//! along the contact normal by the penetration depth — the correction the
//! position-based solver turns into a velocity.
//!
//! # Which sphere
//!
//! A body's collision radius (`add_body_with_radius`, `set_body_collision_radius`)
//! is the sphere tested. A body with none uses the world's default
//! (`set_sdf_collision_radius`), exactly as it does against an SDF collider.
//!
//! # Sides
//!
//! A [`PlaneCollider`] is **two-sided**, as the primitive documents: a sphere
//! whose centre is behind the plane is held on that side. A [`HeightField`] and a
//! [`TriMesh`] have no inside; a sphere is pushed away from the surface point
//! nearest it.
//!
//! Author: Moroya Sakamoto

use crate::collider::Contact;
use crate::heightfield::HeightField;
use crate::math::{Fix128, Vec3Fix};
use crate::plane_collider::PlaneCollider;
use crate::trimesh::TriMesh;

/// A surface that does not move.
pub enum StaticCollider {
    /// An infinite plane.
    Plane(PlaneCollider),
    /// A grid of heights over the `XZ` plane.
    HeightField(HeightField),
    /// A triangle soup with a BVH.
    TriMesh(TriMesh),
}

impl StaticCollider {
    /// The contact of a sphere with this surface: the penetration depth and the
    /// normal pointing from the surface toward the sphere's centre, or `None` when
    /// the sphere is clear of it.
    #[must_use]
    pub fn collide_sphere(&self, center: Vec3Fix, radius: Fix128) -> Option<Contact> {
        match self {
            Self::Plane(plane) => {
                let hit = plane.intersect_sphere(center, radius);
                hit.colliding.then_some(Contact {
                    depth: hit.depth,
                    normal: hit.normal,
                    point_a: hit.point_a,
                    point_b: hit.point_b,
                })
            }
            Self::HeightField(field) => field.collide_sphere(center, radius),
            Self::TriMesh(mesh) => mesh.collide_sphere(center, radius),
        }
    }
}
