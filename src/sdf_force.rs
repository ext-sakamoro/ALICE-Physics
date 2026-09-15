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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub const fn new(sdf_index: usize, force_type: SdfForceType) -> Self {
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
        self.affected_bodies
            .as_ref()
            .is_none_or(|list| list.contains(&body_index))
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
    pub const fn repel(sdf_index: usize, strength: Fix128, range: Fix128) -> Self {
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
    pub const fn surface_flow(sdf_index: usize, direction: Vec3Fix, strength: Fix128) -> Self {
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
            |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
            |x, y, z| {
                let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
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

    fn ground_plane() -> ClosureSdf {
        ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
    }

    fn near_v(v: Vec3Fix, x: f64, y: f64, z: f64) -> bool {
        (v.x.to_f64() - x).abs() < 1e-6
            && (v.y.to_f64() - y).abs() < 1e-6
            && (v.z.to_f64() - z).abs() < 1e-6
    }

    #[test]
    fn surface_flow_constructor_projects_flow_onto_tangent_with_linear_falloff() {
        let field = SdfForceField::surface_flow(3, Vec3Fix::from_int(1, 0, 1), Fix128::from_int(8));
        assert_eq!(field.sdf_index, 3);
        assert!(field.enabled);
        assert!(field.affected_bodies.is_none());
        assert_eq!(
            field.force_type,
            SdfForceType::SurfaceFlow {
                flow_direction: Vec3Fix::from_int(1, 0, 1),
                strength: Fix128::from_int(8),
                influence_distance: Fix128::from_int(2),
            }
        );

        let sdf =
            SdfCollider::new_static(Box::new(unit_sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY);
        // (1.5, 0, 0): dist 0.5、normal +X → flow (1,0,1) の接線成分は +Z、
        // falloff 1 - 0.5/2 = 3/4 → force = (0, 0, 8 * 3/4) = (0, 0, 6)
        let body = RigidBody::new(Vec3Fix::from_f32(1.5, 0.0, 0.0), Fix128::ONE);
        let f = compute_sdf_force(&body, &sdf, &field.force_type);
        assert!(near_v(f, 0.0, 0.0, 6.0), "{f:?}");
        // 表面上 (dist 0) は falloff 1 → (0, 0, 8)
        let on = RigidBody::new(Vec3Fix::from_f32(1.0, 0.0, 0.0), Fix128::ONE);
        let f_on = compute_sdf_force(&on, &sdf, &field.force_type);
        assert!(near_v(f_on, 0.0, 0.0, 8.0), "{f_on:?}");
        // 内側 (0.5, 0, 0) も |dist| 0.5 で同じ大きさ、法線 +X → (0, 0, 6)
        let inside = RigidBody::new(Vec3Fix::from_f32(0.5, 0.0, 0.0), Fix128::ONE);
        let f_in = compute_sdf_force(&inside, &sdf, &field.force_type);
        assert!(near_v(f_in, 0.0, 0.0, 6.0), "{f_in:?}");
        // influence_distance (2) の外は 0
        let far = RigidBody::new(Vec3Fix::from_f32(4.0, 0.0, 0.0), Fix128::ONE);
        assert_eq!(
            compute_sdf_force(&far, &sdf, &field.force_type),
            Vec3Fix::ZERO
        );
        // flow が法線と平行なら接線成分なし → 0
        let radial = SdfForceField::surface_flow(0, Vec3Fix::UNIT_X, Fix128::from_int(8));
        assert_eq!(
            compute_sdf_force(&body, &sdf, &radial.force_type),
            Vec3Fix::ZERO
        );
    }

    #[test]
    fn apply_sdf_force_fields_sums_fields_and_integrates_with_inverse_mass() {
        let colliders = [SdfCollider::new_static(
            Box::new(ground_plane()),
            Vec3Fix::ZERO,
            QuatFix::IDENTITY,
        )];
        // Attract strength 3: body (0, 2, 0) は dist 2 → force -ŷ * 6
        let attract = SdfForceField::new(
            0,
            SdfForceType::Attract {
                strength: Fix128::from_int(3),
                max_force: Fix128::from_int(100),
            },
        );
        // Contain strength 4 / damping 1/2: push -ŷ * 8、damp = v * -1/2 = (-2, 0, 0)
        let contain = SdfForceField::new(
            0,
            SdfForceType::Contain {
                strength: Fix128::from_int(4),
                damping: Fix128::from_ratio(1, 2),
            },
        );
        let dt = Fix128::from_ratio(1, 4);
        let make = || {
            let mut b = RigidBody::new_dynamic(Vec3Fix::from_int(0, 2, 0), Fix128::ONE);
            b.inv_mass = Fix128::from_int(2);
            b.velocity = Vec3Fix::from_int(4, 0, 0);
            b
        };

        // 合力 (-2, -14, 0) → a = F * 2 = (-4, -28, 0) → Δv = a/4 = (-1, -7, 0) → v = (3, -7, 0)
        let mut bodies = vec![make(), RigidBody::new_static(Vec3Fix::from_int(0, 2, 0))];
        bodies[1].velocity = Vec3Fix::from_int(4, 0, 0);
        apply_sdf_force_fields(
            &[attract.clone(), contain.clone()],
            &colliders,
            &mut bodies,
            dt,
        );
        assert!(
            near_v(bodies[0].velocity, 3.0, -7.0, 0.0),
            "{:?}",
            bodies[0].velocity
        );
        // static は力を受けない
        assert_eq!(bodies[1].velocity, Vec3Fix::from_int(4, 0, 0));

        // disabled field は無視 → attract のみ: Δv = (0, -6*2/4, 0) = (0, -3, 0)
        let mut off = contain.clone();
        off.enabled = false;
        let mut b2 = vec![make()];
        apply_sdf_force_fields(&[attract.clone(), off], &colliders, &mut b2, dt);
        assert!(
            near_v(b2[0].velocity, 4.0, -3.0, 0.0),
            "{:?}",
            b2[0].velocity
        );

        // sdf_index 範囲外の field は無視
        let dangling = SdfForceField::new(
            5,
            SdfForceType::Attract {
                strength: Fix128::from_int(3),
                max_force: Fix128::from_int(100),
            },
        );
        let mut b3 = vec![make()];
        apply_sdf_force_fields(&[dangling], &colliders, &mut b3, dt);
        assert_eq!(b3[0].velocity, Vec3Fix::from_int(4, 0, 0));

        // affected_bodies filter: body 1 だけに contain を適用
        let only_one = contain.with_affected_bodies(vec![1]);
        let mut b4 = vec![make(), make()];
        apply_sdf_force_fields(&[only_one], &colliders, &mut b4, dt);
        assert_eq!(b4[0].velocity, Vec3Fix::from_int(4, 0, 0));
        // contain のみ: F = (-2, -8, 0) → Δv = (-1, -4, 0) → (3, -4, 0)
        assert!(
            near_v(b4[1].velocity, 3.0, -4.0, 0.0),
            "{:?}",
            b4[1].velocity
        );

        // field なし → 不変
        let mut b5 = vec![make()];
        apply_sdf_force_fields(&[], &colliders, &mut b5, dt);
        assert_eq!(b5[0].velocity, Vec3Fix::from_int(4, 0, 0));
    }
}
