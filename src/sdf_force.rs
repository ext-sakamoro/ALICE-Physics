//! SDF-Driven Force Fields
//!
//! Uses SDF gradient vectors as force field directions.
//! Bodies are pushed along the SDF surface or attracted/repelled
//! based on their distance to the surface.
//!
//! # Use Cases
//!
//! - Containment: keep particles inside an SDF volume
//! - Surface flow: move bodies along SDF surface (tangent to gradient)
//! - Attraction: pull bodies toward SDF surface
//! - Repulsion: push bodies away from SDF boundary
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;
#[cfg(feature = "std")]
use crate::solver::RigidBody;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// SDF Force Field Types
// ============================================================================

/// Type of SDF-driven force
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SdfForceType {
    /// Push toward SDF surface (attraction)
    /// Strength scales with distance from surface
    Attract {
        /// Force strength
        strength: Fix128,
        /// Maximum force magnitude
        max_force: Fix128,
    },

    /// Push away from SDF surface (repulsion)
    /// Strength scales inversely with distance (stronger near surface)
    Repel {
        /// Force strength
        strength: Fix128,
        /// Maximum effective distance
        range: Fix128,
    },

    /// Containment: zero force inside, strong push inward when outside
    Contain {
        /// Inward push strength when outside
        strength: Fix128,
        /// Damping coefficient for containment
        damping: Fix128,
    },

    /// Surface flow: force tangent to SDF surface
    /// Bodies slide along the surface in a specified direction
    SurfaceFlow {
        /// Flow direction (will be projected onto surface tangent)
        flow_direction: Vec3Fix,
        /// Flow strength
        strength: Fix128,
        /// Maximum distance from surface where flow applies
        influence_distance: Fix128,
    },

    /// Vortex around SDF: rotational force following SDF contours
    SdfVortex {
        /// Rotation axis
        axis: Vec3Fix,
        /// Rotational strength
        strength: Fix128,
        /// Maximum distance from surface where vortex applies
        influence_distance: Fix128,
    },
}

/// SDF-driven force field instance
#[derive(Clone, Debug)]
pub struct SdfForceField {
    /// SDF collider index in the physics world
    pub sdf_index: usize,
    /// Force type
    pub force_type: SdfForceType,
    /// Whether this field is active
    pub enabled: bool,
    /// Optional body filter (None = affects all)
    pub affected_bodies: Option<Vec<usize>>,
}

impl SdfForceField {
    /// Create a new SDF force field
    #[must_use]
    pub fn new(sdf_index: usize, force_type: SdfForceType) -> Self {
        Self {
            sdf_index,
            force_type,
            enabled: true,
            affected_bodies: None,
        }
    }

    /// Restrict to specific bodies
    #[must_use]
    pub fn with_affected_bodies(mut self, bodies: Vec<usize>) -> Self {
        self.affected_bodies = Some(bodies);
        self
    }

    /// Check if this field affects a given body
    #[cfg(feature = "std")]
    #[inline]
    fn affects(&self, body_index: usize) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.affected_bodies {
            None => true,
            Some(list) => list.contains(&body_index),
        }
    }
}

// ============================================================================
// Force Computation
// ============================================================================

/// Compute SDF-driven force on a body.
///
/// Evaluates the SDF at the body's position and computes force
/// based on the distance and gradient.
#[cfg(feature = "std")]
#[must_use]
pub fn compute_sdf_force(
    body: &RigidBody,
    sdf: &SdfCollider,
    force_type: &SdfForceType,
) -> Vec3Fix {
    let (lx, ly, lz) = sdf.world_to_local(body.position);
    let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
    let normal = sdf.local_normal_to_world(nx, ny, nz);

    let dist_fix = Fix128::from_f32(dist);

    match force_type {
        SdfForceType::Attract {
            strength,
            max_force,
        } => {
            // Force toward surface, proportional to distance
            let force_mag = (*strength * dist_fix.abs()).min(*max_force);
            if dist > 0.0 {
                // Outside: push inward (negative normal direction)
                -normal * force_mag
            } else {
                // Inside: push outward
                normal * force_mag
            }
        }

        SdfForceType::Repel { strength, range } => {
            let dist_abs = dist_fix.abs();
            if dist_abs > *range {
                return Vec3Fix::ZERO;
            }
            // Inverse distance: stronger near surface
            let factor = Fix128::ONE - dist_abs / *range;
            let force_mag = *strength * factor * factor;
            if dist > 0.0 {
                normal * force_mag // Push outward
            } else {
                -normal * force_mag // Push outward from inside
            }
        }

        SdfForceType::Contain { strength, damping } => {
            if dist <= 0.0 {
                // Inside: apply damping only
                body.velocity * (-*damping)
            } else {
                // Outside: strong inward push + damping
                let push = -normal * (*strength * dist_fix);
                let damp = body.velocity * (-*damping);
                push + damp
            }
        }

        SdfForceType::SurfaceFlow {
            flow_direction,
            strength,
            influence_distance,
        } => {
            let dist_abs = dist_fix.abs();
            if dist_abs > *influence_distance {
                return Vec3Fix::ZERO;
            }

            // Project flow direction onto surface tangent plane
            let dot = flow_direction.dot(normal);
            let tangent = *flow_direction - normal * dot;
            let tangent_len = tangent.length();
            if tangent_len.is_zero() {
                return Vec3Fix::ZERO;
            }
            let tangent_norm = tangent / tangent_len;

            // Falloff with distance from surface
            let falloff = Fix128::ONE - dist_abs / *influence_distance;
            tangent_norm * (*strength * falloff)
        }

        SdfForceType::SdfVortex {
            axis,
            strength,
            influence_distance,
        } => {
            let dist_abs = dist_fix.abs();
            if dist_abs > *influence_distance {
                return Vec3Fix::ZERO;
            }

            // Tangent direction: cross(axis, gradient)
            let tangent = axis.cross(normal).normalize();

            let falloff = Fix128::ONE - dist_abs / *influence_distance;
            tangent * (*strength * falloff)
        }
    }
}

/// Apply all SDF force fields to bodies for one timestep.
#[cfg(feature = "std")]
pub fn apply_sdf_force_fields(
    fields: &[SdfForceField],
    sdf_colliders: &[SdfCollider],
    bodies: &mut [RigidBody],
    dt: Fix128,
) {
    for (body_idx, body) in bodies.iter_mut().enumerate() {
        if body.is_static() {
            continue;
        }

        let mut total_force = Vec3Fix::ZERO;

        for field in fields {
            if !field.affects(body_idx) {
                continue;
            }
            if field.sdf_index >= sdf_colliders.len() {
                continue;
            }

            let sdf = &sdf_colliders[field.sdf_index];
            total_force = total_force + compute_sdf_force(body, sdf, &field.force_type);
        }

        // F = ma, a = F * inv_mass, v += a * dt
        let acceleration = total_force * body.inv_mass;
        body.velocity = body.velocity + acceleration * dt;
    }
}

// ============================================================================
// Convenience constructors
// ============================================================================

impl SdfForceField {
    /// Create attraction toward SDF surface
    #[must_use]
    pub fn attract(sdf_index: usize, strength: Fix128) -> Self {
        Self::new(
            sdf_index,
            SdfForceType::Attract {
                strength,
                max_force: strength * Fix128::from_int(10),
            },
        )
    }

    /// Create repulsion from SDF surface
    #[must_use]
    pub fn repel(sdf_index: usize, strength: Fix128, range: Fix128) -> Self {
        Self::new(sdf_index, SdfForceType::Repel { strength, range })
    }

    /// Create containment field
    #[must_use]
    pub fn contain(sdf_index: usize, strength: Fix128) -> Self {
        Self::new(
            sdf_index,
            SdfForceType::Contain {
                strength,
                damping: Fix128::from_ratio(1, 10),
            },
        )
    }

    /// Create surface flow
    #[must_use]
    pub fn surface_flow(sdf_index: usize, direction: Vec3Fix, strength: Fix128) -> Self {
        Self::new(
            sdf_index,
            SdfForceType::SurfaceFlow {
                flow_direction: direction,
                strength,
                influence_distance: Fix128::from_int(2),
            },
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::math::QuatFix;
    use crate::sdf_collider::{ClosureSdf, SdfCollider};

    fn unit_sphere() -> ClosureSdf {
        ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len < 1e-10 {
                    (0.0, 1.0, 0.0)
                } else {
                    (x / len, y / len, z / len)
                }
            },
        )
    }

    #[test]
    fn test_attract_outside() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let body = RigidBody::new(Vec3Fix::from_f32(3.0, 0.0, 0.0), Fix128::ONE);
        let force_type = SdfForceType::Attract {
            strength: Fix128::from_int(10),
            max_force: Fix128::from_int(100),
        };

        let force = compute_sdf_force(&body, &sdf, &force_type);
        // Body outside sphere, should be pulled inward (negative X)
        assert!(force.x < Fix128::ZERO, "Should pull toward sphere surface");
    }

    #[test]
    fn test_contain_inside() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let body = RigidBody::new(Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::ONE);
        let force_type = SdfForceType::Contain {
            strength: Fix128::from_int(10),
            damping: Fix128::from_ratio(1, 10),
        };

        let force = compute_sdf_force(&body, &sdf, &force_type);
        // Body inside containment: only damping, no push
        // Velocity is zero, so force should be near zero
        let mag = force.length().to_f32();
        assert!(
            mag < 0.01,
            "Inside containment with zero velocity should have minimal force"
        );
    }

    #[test]
    fn test_contain_outside() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let body = RigidBody::new(Vec3Fix::from_f32(3.0, 0.0, 0.0), Fix128::ONE);
        let force_type = SdfForceType::Contain {
            strength: Fix128::from_int(10),
            damping: Fix128::from_ratio(1, 10),
        };

        let force = compute_sdf_force(&body, &sdf, &force_type);
        // Body outside containment: should push inward
        assert!(force.x < Fix128::ZERO, "Should push body inward");
    }

    #[test]
    fn test_surface_flow() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        // Body near surface, flow in +Z direction
        let body = RigidBody::new(Vec3Fix::from_f32(1.1, 0.0, 0.0), Fix128::ONE);
        let force_type = SdfForceType::SurfaceFlow {
            flow_direction: Vec3Fix::UNIT_Z,
            strength: Fix128::from_int(10),
            influence_distance: Fix128::from_int(2),
        };

        let force = compute_sdf_force(&body, &sdf, &force_type);
        // At (1.1, 0, 0), normal is +X. Flow in Z projected onto tangent plane should give Z force
        assert!(
            force.z.to_f32().abs() > 0.01,
            "Surface flow should produce tangential force"
        );
    }

    #[test]
    fn test_repel() {
        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);

        let body = RigidBody::new(Vec3Fix::from_f32(1.5, 0.0, 0.0), Fix128::ONE);
        let force_type = SdfForceType::Repel {
            strength: Fix128::from_int(10),
            range: Fix128::from_int(5),
        };

        let force = compute_sdf_force(&body, &sdf, &force_type);
        // Body outside, should push further away
        assert!(
            force.x > Fix128::ZERO,
            "Repel should push away from surface"
        );
    }
}
