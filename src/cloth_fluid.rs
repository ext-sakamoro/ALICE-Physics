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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
///
/// This is a thin wrapper over [`apply_fluid_forces_to_cloth_with_residual`],
/// which does the work and additionally reports the interface force. Callers
/// that sub-iterate the coupling want that one; this one discards the report.
/// There is deliberately only one implementation, so the two cannot drift.
pub fn apply_fluid_forces_to_cloth(
    coupling: &ClothFluidCoupling,
    cloth_positions: &[Vec3Fix],
    cloth_velocities: &mut [Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    fluid_density: Fix128,
    dt: Fix128,
) {
    let _ = apply_fluid_forces_to_cloth_with_residual(
        coupling,
        cloth_positions,
        cloth_velocities,
        fluid_positions,
        fluid_velocities,
        fluid_density,
        dt,
    );
}

/// Apply the fluid-to-cloth forces and report the interface force norm.
///
/// Identical in effect to [`apply_fluid_forces_to_cloth`]; the return value is
/// `‖F_interface‖_∞` over the cloth particles, where `F` is the net force the
/// fluid exerts (drag, buoyancy and surface tension combined) **before** it is
/// multiplied by `dt`.
///
/// # ⚠️ This is the force, not the iteration residual
///
/// The value returned is the **absolute** interface force at this sweep. It
/// does **not** go to zero as a sub-iteration converges — it converges to the
/// equilibrium force, which is generally non-zero. Handing it straight to a
/// convergence test would compare a physical magnitude against a tolerance and
/// never settle.
///
/// The residual of the fixed-point iteration is the **change** in interface
/// force between sweeps, `‖F^(j) − F^(j−1)‖_∞`. Because each sweep restarts the
/// cloth velocities from the step's initial state and adds `F·dt`, the iterate
/// difference satisfies
///
/// ```text
/// ‖v^(j+1) − v^(j)‖_∞ = dt · ‖F^(j) − F^(j−1)‖_∞
/// ```
///
/// exactly, so a driver obtains the force residual by differencing the state,
/// and the constant `dt` cancels in the relative stopping rules of
/// [`crate::coupled_iteration`]. `tests/cloth_fluid_sub_iteration.rs` does
/// exactly that.
///
/// What this return value is good for is a `dt`-independent reading of how hard
/// the interface is pushing — the velocity change would shrink with `dt` and so
/// could not distinguish "the interface is balanced" from "the step is small".
///
/// # What this supports, and what it does not
///
/// Driving this coupling under [`crate::coupled_iteration::run_sub_iteration`]
/// measures the **contraction ratio of the splitting** and detects divergence,
/// including the silent `Fix128` wrap. ⚠️ It does **not** establish that the
/// converged state is physically correct: this coupling is particle-based and
/// has no closed form to compare against. The closed-form claims live in
/// `tests/analytic_added_mass_coupling.rs` on the one-degree-of-freedom piston,
/// and are deliberately not transferred here.
///
/// The drag part of the applied update is one implicit Euler step on the
/// relative velocity `u = v_cloth − v_fluid`: `u' = u / (1 + c·dt)` with
/// `c = C_d · ρ · N`. Buoyancy and surface tension keep their explicit `F·dt`
/// terms. The reported norm is still the force evaluated at the incoming
/// velocity (`c·|u|` for drag), so for `c·dt > 0` the velocity change of a
/// sweep is the reported drag force times `dt / (1 + c·dt)`, not times `dt`.
///
/// # Claims
///
/// - Drag never reverses the relative velocity and never increases its
///   magnitude, for any `dt >= 0`, `C_d >= 0`, `ρ >= 0` (per component,
///   `u' = u / (1 + c·dt)` has the sign of `u` and `|u'| <= |u|`).
/// - For `c·dt -> 0` the update tends to the explicit `u − c·dt·u`.
/// - A zero `dt`, empty inputs or no neighbour leave the velocities unchanged.
/// - A negative `c·dt` is outside the model; the explicit step is used.
pub fn apply_fluid_forces_to_cloth_with_residual(
    coupling: &ClothFluidCoupling,
    cloth_positions: &[Vec3Fix],
    cloth_velocities: &mut [Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    fluid_density: Fix128,
    dt: Fix128,
) -> Fix128 {
    let mut interface_force = Fix128::ZERO;
    if dt.is_zero() || fluid_positions.is_empty() || cloth_positions.is_empty() {
        return interface_force;
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

        // Drag is advanced with one implicit Euler step on the relative
        // velocity: u' = u / (1 + c dt), i.e. u - u c dt / (1 + c dt), with
        // c = C_d rho N. The explicit step u - c dt u overshoots (and flips
        // the sign of u) once c dt > 1. Outside c dt >= 0 the closed form
        // above is not defined, so the explicit step is kept there.
        // Buoyancy and surface tension keep their explicit F * dt terms.
        let drag_rate_dt = coupling.drag_coefficient * density_factor * dt;
        let drag_delta = if drag_rate_dt >= Fix128::ZERO {
            relative_vel * (drag_rate_dt / (Fix128::ONE + drag_rate_dt))
        } else {
            drag_force * dt
        };
        cloth_velocities[ci] = cv - drag_delta + buoyancy_force * dt + tension_force * dt;

        // Report the net interface force. Additions are exact in `Fix128`
        // (wrapping two's complement), so summing the terms here cannot
        // perturb what was applied above.
        let net = buoyancy_force + tension_force - drag_force;
        for component in [net.x, net.y, net.z] {
            let magnitude = component.abs();
            if magnitude > interface_force {
                interface_force = magnitude;
            }
        }
    }

    interface_force
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
///
/// A thin wrapper over [`apply_cloth_boundary_to_fluid_with_residual`], which
/// does the work and additionally reports the interface correction.
pub fn apply_cloth_boundary_to_fluid(
    cloth_positions: &[Vec3Fix],
    cloth_normals: &[Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &mut [Vec3Fix],
    repulsion_strength: Fix128,
) {
    let _ = apply_cloth_boundary_to_fluid_with_residual(
        cloth_positions,
        cloth_normals,
        fluid_positions,
        fluid_velocities,
        repulsion_strength,
    );
}

/// Apply the cloth-to-fluid boundary repulsion and report its norm.
///
/// Identical in effect to [`apply_cloth_boundary_to_fluid`]; the return value
/// is `‖Δv‖_∞` over the fluid particles, where `Δv` is each fluid particle's
/// **net** velocity change from this call (the sum of the pushes from every
/// cloth particle in range). Two pushes on the same side add, two from
/// opposite sides cancel, and the report follows the sum in both cases.
///
/// ⚠️ Unlike the fluid-to-cloth direction, this reports a **velocity
/// correction rather than a force**, because `repulsion_strength` already
/// carries the step: this API has no `dt` to divide out. The two directions
/// are therefore not in the same units, and a caller combining them into one
/// residual must scale them itself. Feeding either to
/// [`crate::coupled_iteration::run_sub_iteration`] on its own is unaffected,
/// since every stopping rule there is relative to the first residual.
///
/// The applied update is byte-for-byte what
/// [`apply_cloth_boundary_to_fluid`] has always produced.
pub fn apply_cloth_boundary_to_fluid_with_residual(
    cloth_positions: &[Vec3Fix],
    cloth_normals: &[Vec3Fix],
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &mut [Vec3Fix],
    repulsion_strength: Fix128,
) -> Fix128 {
    let mut interface_correction = Fix128::ZERO;
    if cloth_positions.is_empty() || fluid_positions.is_empty() {
        return interface_correction;
    }

    let repulsion_radius = Fix128::from_ratio(1, 4);
    let radius_sq = repulsion_radius * repulsion_radius;

    for fi in 0..fluid_positions.len() {
        let fp = fluid_positions[fi];
        let before = fluid_velocities[fi];

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

            let correction = normal * force_mag;
            if proj.is_negative() {
                // Fluid is on the back side: push away along -normal
                fluid_velocities[fi] = fluid_velocities[fi] - correction;
            } else {
                // Fluid is on the front side: push away along +normal
                fluid_velocities[fi] = fluid_velocities[fi] + correction;
            }
        }

        // The report is the net change of this particle's velocity, measured
        // after every cloth particle has pushed it: pushes from different
        // cloth particles add (same side) or cancel (opposite sides), and the
        // residual must see the sum, not the largest single push
        let dv = fluid_velocities[fi] - before;
        for component in [dv.x, dv.y, dv.z] {
            let magnitude = component.abs();
            if magnitude > interface_correction {
                interface_correction = magnitude;
            }
        }
    }

    interface_correction
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

    // ------------------------------------------------------------------
    // The reporting variants: what the prose above claims, as assertions
    // ------------------------------------------------------------------

    /// A scene with all three force terms live, so none of them can be
    /// dropped without the assertions noticing.
    fn coupled_scene() -> (
        ClothFluidCoupling,
        Vec<Vec3Fix>,
        Vec<Vec3Fix>,
        Vec<Vec3Fix>,
        Vec<Vec3Fix>,
    ) {
        let coupling = ClothFluidCoupling::default();
        let quarter = Fix128::from_ratio(1, 4);
        let eighth = Fix128::from_ratio(1, 8);
        let cloth_pos = vec![
            Vec3Fix::ZERO,
            Vec3Fix::new(quarter, Fix128::ZERO, Fix128::ZERO),
        ];
        let cloth_vel = vec![Vec3Fix::from_int(3, 0, 0), Vec3Fix::from_int(0, -2, 0)];
        let fluid_pos = vec![
            Vec3Fix::ZERO,
            Vec3Fix::new(eighth, Fix128::ZERO, Fix128::ZERO),
        ];
        let fluid_vel = vec![Vec3Fix::from_int(0, 1, 0), Vec3Fix::from_int(1, 0, 0)];
        (coupling, cloth_pos, cloth_vel, fluid_pos, fluid_vel)
    }

    #[test]
    fn the_wrapper_and_the_reporting_variant_leave_identical_state() {
        // The wrapper exists so that there is only one implementation. If it
        // ever grows its own copy, this goes red on the first divergence.
        let (coupling, cloth_pos, cloth_vel, fluid_pos, fluid_vel) = coupled_scene();
        let dt = Fix128::from_ratio(1, 60);

        let mut through_wrapper = cloth_vel.clone();
        apply_fluid_forces_to_cloth(
            &coupling,
            &cloth_pos,
            &mut through_wrapper,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            dt,
        );

        let mut through_reporting = cloth_vel.clone();
        let force = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &cloth_pos,
            &mut through_reporting,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            dt,
        );

        assert_eq!(through_wrapper, through_reporting);
        assert!(
            force > Fix128::ZERO,
            "this scene must exert a force, or the comparison above is vacuous"
        );
        // The state must actually have moved, or two no-ops would match.
        assert_ne!(through_wrapper, cloth_vel);
    }

    #[test]
    fn the_boundary_wrapper_and_its_reporting_variant_leave_identical_state() {
        let (_, cloth_pos, _, fluid_pos, fluid_vel) = coupled_scene();
        let normals = vec![Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Y];
        let strength = Fix128::from_ratio(1, 4);

        let mut through_wrapper = fluid_vel.clone();
        apply_cloth_boundary_to_fluid(
            &cloth_pos,
            &normals,
            &fluid_pos,
            &mut through_wrapper,
            strength,
        );

        let mut through_reporting = fluid_vel.clone();
        let correction = apply_cloth_boundary_to_fluid_with_residual(
            &cloth_pos,
            &normals,
            &fluid_pos,
            &mut through_reporting,
            strength,
        );

        assert_eq!(through_wrapper, through_reporting);
        assert!(
            correction > Fix128::ZERO,
            "this scene must repel, or the comparison above is vacuous"
        );
        assert_ne!(through_wrapper, fluid_vel);
    }

    #[test]
    fn the_reported_force_does_not_move_when_the_step_does() {
        // This is the reason the fluid-to-cloth direction reports a force and
        // not a velocity change. Halving `dt` must halve what is applied while
        // leaving the reported force alone; a velocity change would halve too,
        // and so could not tell "the interface is balanced" from "the step is
        // small".
        let (coupling, cloth_pos, cloth_vel, fluid_pos, fluid_vel) = coupled_scene();

        let mut coarse_state = cloth_vel.clone();
        let coarse_force = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &cloth_pos,
            &mut coarse_state,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            Fix128::from_ratio(1, 60),
        );

        let mut fine_state = cloth_vel.clone();
        let fine_force = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &cloth_pos,
            &mut fine_state,
            &fluid_pos,
            &fluid_vel,
            Fix128::ONE,
            Fix128::from_ratio(1, 120),
        );

        assert_eq!(
            coarse_force, fine_force,
            "the reported force must not depend on the step"
        );

        // And the applied change must, or the test above says nothing.
        let coarse_change = (coarse_state[0] - cloth_vel[0]).x.abs();
        let fine_change = (fine_state[0] - cloth_vel[0]).x.abs();
        assert!(
            fine_change < coarse_change,
            "halving the step must shrink what is applied: {coarse_change:?} -> {fine_change:?}"
        );
    }

    #[test]
    fn an_empty_scene_reports_no_interface_force() {
        let (coupling, cloth_pos, cloth_vel, _, _) = coupled_scene();
        let mut state = cloth_vel.clone();
        let force = apply_fluid_forces_to_cloth_with_residual(
            &coupling,
            &cloth_pos,
            &mut state,
            &[],
            &[],
            Fix128::ONE,
            Fix128::from_ratio(1, 60),
        );
        assert_eq!(force, Fix128::ZERO);
        assert_eq!(state, cloth_vel);
    }
}
