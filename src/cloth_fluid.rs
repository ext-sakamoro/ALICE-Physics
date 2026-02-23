//! Cloth-Fluid Coupling
//!
//! Two-way interaction between cloth particles and fluid particles:
//!
//! - **Fluid-to-Cloth**: Drag and buoyancy forces on cloth particles that
//!   are near fluid particles.
//! - **Cloth-to-Fluid**: Cloth surface acts as a boundary that repels
//!   nearby fluid particles.
//!
//! All computations use deterministic 128-bit fixed-point arithmetic.

use crate::math::{Fix128, Vec3Fix};

// ============================================================================
// Cloth-Fluid Coupling Configuration
// ============================================================================

/// Configuration for cloth-fluid interaction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothFluidCoupling {
    /// Drag coefficient applied to cloth particles submerged in fluid.
    /// Higher values cause stronger velocity damping.
    pub drag_coefficient: Fix128,
    /// Buoyancy factor controlling upward force on submerged cloth.
    pub buoyancy_factor: Fix128,
    /// Surface tension factor at the fluid-cloth interface.
    pub surface_tension: Fix128,
}

impl Default for ClothFluidCoupling {
    fn default() -> Self {
        Self {
            drag_coefficient: Fix128::from_ratio(1, 2),
            buoyancy_factor: Fix128::from_ratio(1, 10),
            surface_tension: Fix128::from_ratio(1, 100),
        }
    }
}

// ============================================================================
// Fluid-to-Cloth Forces
// ============================================================================

/// Apply drag and buoyancy forces from fluid particles to cloth particles.
///
/// For each cloth particle, finds nearby fluid particles and applies:
/// - **Drag**: Force proportional to the relative velocity between cloth
///   and fluid, scaled by the number of nearby fluid particles.
/// - **Buoyancy**: Upward force proportional to the local fluid density
///   (approximated by nearby fluid particle count).
///
/// Cloth velocities are modified in place.
pub fn apply_fluid_forces_to_cloth(
    coupling: &ClothFluidCoupling,
    cloth_positions: &[Vec3Fix],
    cloth_velocities: &mut [Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    fluid_density: Fix128,
    dt: Fix128,
) {
    if dt.is_zero() || fluid_positions.is_empty() || cloth_positions.is_empty() {
        return;
    }

    // Interaction radius: larger radius catches more fluid neighbors
    let interaction_radius = Fix128::from_ratio(1, 2);
    let radius_sq = interaction_radius * interaction_radius;
    let buoyancy_dir = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

    for ci in 0..cloth_positions.len() {
        let cp = cloth_positions[ci];
        let cv = cloth_velocities[ci];

        let mut avg_fluid_vel = Vec3Fix::ZERO;
        let mut neighbor_count = Fix128::ZERO;

        // Find fluid particles within interaction radius
        for fi in 0..fluid_positions.len() {
            let diff = fluid_positions[fi] - cp;
            let dist_sq = diff.dot(diff);

            if dist_sq < radius_sq {
                avg_fluid_vel = avg_fluid_vel + fluid_velocities[fi];
                neighbor_count = neighbor_count + Fix128::ONE;
            }
        }

        if neighbor_count.is_zero() {
            continue;
        }

        // Average fluid velocity near this cloth particle
        avg_fluid_vel = avg_fluid_vel / neighbor_count;

        // Drag force: F_drag = -C_d * (v_cloth - v_fluid) * density_factor
        let relative_vel = cv - avg_fluid_vel;
        let density_factor = neighbor_count * fluid_density;
        let drag_force = relative_vel * (coupling.drag_coefficient * density_factor);

        // Buoyancy force: F_buoy = buoyancy_factor * neighbor_count * up
        let buoyancy_force = buoyancy_dir * (coupling.buoyancy_factor * neighbor_count);

        // Surface tension: pulls cloth toward local fluid center
        let tension_force = avg_fluid_vel * coupling.surface_tension;

        // Apply forces as velocity change (F * dt)
        cloth_velocities[ci] = cv - drag_force * dt + buoyancy_force * dt + tension_force * dt;
    }
}

// ============================================================================
// Cloth-to-Fluid Boundary
// ============================================================================

/// Apply cloth boundary forces to fluid particles.
///
/// Each cloth particle acts as a boundary that repels nearby fluid particles
/// along the cloth surface normal direction. This prevents fluid from
/// passing through the cloth.
///
/// `cloth_normals` should contain per-particle surface normals.
pub fn apply_cloth_boundary_to_fluid(
    cloth_positions: &[Vec3Fix],
    cloth_normals: &[Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &mut [Vec3Fix],
    repulsion_strength: Fix128,
) {
    if cloth_positions.is_empty() || fluid_positions.is_empty() {
        return;
    }

    let repulsion_radius = Fix128::from_ratio(1, 4);
    let radius_sq = repulsion_radius * repulsion_radius;

    for fi in 0..fluid_positions.len() {
        let fp = fluid_positions[fi];

        for ci in 0..cloth_positions.len() {
            let cp = cloth_positions[ci];
            let normal = if ci < cloth_normals.len() {
                cloth_normals[ci]
            } else {
                Vec3Fix::UNIT_Y
            };

            let diff = fp - cp;
            let dist_sq = diff.dot(diff);

            if dist_sq >= radius_sq || dist_sq.is_zero() {
                continue;
            }

            // Project onto cloth normal to determine which side
            let proj = diff.dot(normal);

            // Repulsion: push fluid along normal, strength inversely proportional to distance
            let dist = dist_sq.sqrt();
            let inv_dist = Fix128::ONE / dist;
            let overlap = repulsion_radius - dist;
            let force_mag = repulsion_strength * overlap * inv_dist;

            if proj.is_negative() {
                // Fluid is on the back side: push away along -normal
                fluid_velocities[fi] = fluid_velocities[fi] - normal * force_mag;
            } else {
                // Fluid is on the front side: push away along +normal
                fluid_velocities[fi] = fluid_velocities[fi] + normal * force_mag;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn test_default_coupling() {
        let coupling = ClothFluidCoupling::default();
        assert!(!coupling.drag_coefficient.is_zero());
        assert!(!coupling.buoyancy_factor.is_zero());
    }

    #[test]
    fn test_no_fluid_no_effect() {
        let coupling = ClothFluidCoupling::default();
        let cloth_pos = vec![Vec3Fix::ZERO];
        let mut cloth_vel = vec![Vec3Fix::from_int(1, 0, 0)];
        let original_vel = cloth_vel[0];

        apply_fluid_forces_to_cloth(
            &coupling,
            &cloth_pos,
            &mut cloth_vel,
            &[],
            &[],
            Fix128::ONE,
            Fix128::from_ratio(1, 60),
        );

        assert_eq!(cloth_vel[0].x.hi, original_vel.x.hi);
    }

    #[test]
    fn test_no_cloth_no_effect() {
        let coupling = ClothFluidCoupling::default();
        apply_fluid_forces_to_cloth(
            &coupling,
            &[],
            &mut [],
            &[Vec3Fix::ZERO],
            &[Vec3Fix::ZERO],
            Fix128::ONE,
            Fix128::from_ratio(1, 60),
        );
        // Should not panic
    }

    #[test]
    fn test_drag_slows_cloth() {
        let coupling = ClothFluidCoupling {
            drag_coefficient: Fix128::ONE,
            buoyancy_factor: Fix128::ZERO,
            surface_tension: Fix128::ZERO,
        };

        let cloth_pos = vec![Vec3Fix::ZERO];
        // Cloth moving fast in +X, fluid stationary
        let mut cloth_vel = vec![Vec3Fix::from_int(10, 0, 0)];
        let fluid_pos = vec![Vec3Fix::ZERO]; // Right on top
        let fluid_vel = vec![Vec3Fix::ZERO];

        apply_fluid_forces_to_cloth(
            &coupling,
            &cloth_pos,
            &mut cloth_vel,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            Fix128::from_ratio(1, 60),
        );

        // Drag should reduce velocity in +X direction
        assert!(cloth_vel[0].x < Fix128::from_int(10));
    }

    #[test]
    fn test_buoyancy_adds_upward_force() {
        let coupling = ClothFluidCoupling {
            drag_coefficient: Fix128::ZERO,
            buoyancy_factor: Fix128::ONE,
            surface_tension: Fix128::ZERO,
        };

        let cloth_pos = vec![Vec3Fix::ZERO];
        let mut cloth_vel = vec![Vec3Fix::ZERO];
        let fluid_pos = vec![Vec3Fix::ZERO];
        let fluid_vel = vec![Vec3Fix::ZERO];

        apply_fluid_forces_to_cloth(
            &coupling,
            &cloth_pos,
            &mut cloth_vel,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            Fix128::from_ratio(1, 10),
        );

        // Buoyancy should add upward (positive Y) velocity
        assert!(cloth_vel[0].y > Fix128::ZERO);
    }

    #[test]
    fn test_boundary_repulsion() {
        let cloth_pos = vec![Vec3Fix::ZERO];
        let cloth_normals = vec![Vec3Fix::UNIT_Y];
        // Fluid particle slightly above cloth
        let fluid_pos = vec![Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_ratio(1, 10),
            Fix128::ZERO,
        )];
        let mut fluid_vel = vec![Vec3Fix::new(
            Fix128::ZERO,
            Fix128::from_int(-5),
            Fix128::ZERO,
        )];

        apply_cloth_boundary_to_fluid(
            &cloth_pos,
            &cloth_normals,
            &fluid_pos,
            &mut fluid_vel,
            Fix128::from_int(10),
        );

        // Fluid should be pushed upward (repelled from cloth)
        assert!(fluid_vel[0].y > Fix128::from_int(-5));
    }

    #[test]
    fn test_boundary_no_effect_far_away() {
        let cloth_pos = vec![Vec3Fix::ZERO];
        let cloth_normals = vec![Vec3Fix::UNIT_Y];
        // Fluid particle far from cloth
        let fluid_pos = vec![Vec3Fix::from_int(100, 100, 100)];
        let mut fluid_vel = vec![Vec3Fix::from_int(1, -1, 0)];
        let original_vel = fluid_vel[0];

        apply_cloth_boundary_to_fluid(
            &cloth_pos,
            &cloth_normals,
            &fluid_pos,
            &mut fluid_vel,
            Fix128::ONE,
        );

        assert_eq!(fluid_vel[0].x.hi, original_vel.x.hi);
        assert_eq!(fluid_vel[0].y.hi, original_vel.y.hi);
    }

    #[test]
    fn test_zero_dt_no_effect() {
        let coupling = ClothFluidCoupling::default();
        let cloth_pos = vec![Vec3Fix::ZERO];
        let mut cloth_vel = vec![Vec3Fix::from_int(5, 0, 0)];
        let original = cloth_vel[0];

        apply_fluid_forces_to_cloth(
            &coupling,
            &cloth_pos,
            &mut cloth_vel,
            &[Vec3Fix::ZERO],
            &[Vec3Fix::ZERO],
            Fix128::ONE,
            Fix128::ZERO,
        );

        assert_eq!(cloth_vel[0].x.hi, original.x.hi);
    }
}
