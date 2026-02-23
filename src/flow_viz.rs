//! Fluid Flow Visualization
//!
//! Generates visualization data from fluid particle simulations:
//!
//! - **Flow Arrows**: Grid-sampled velocity arrows for vector field display
//! - **Streamlines**: Particle traces through the velocity field
//!
//! All computations use deterministic 128-bit fixed-point arithmetic.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Flow Visualization Types
// ============================================================================

/// A velocity arrow for flow visualization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlowArrow {
    /// Arrow base position
    pub position: Vec3Fix,
    /// Velocity direction (normalized)
    pub direction: Vec3Fix,
    /// Velocity magnitude
    pub magnitude: Fix128,
}

/// Configuration for flow visualization.
#[derive(Clone, Debug)]
pub struct FlowVizConfig {
    /// Grid resolution (cells per axis)
    pub grid_resolution: usize,
    /// Minimum corner of the visualization volume
    pub bounds_min: Vec3Fix,
    /// Maximum corner of the visualization volume
    pub bounds_max: Vec3Fix,
    /// Scale factor for arrow lengths
    pub arrow_scale: Fix128,
}

// ============================================================================
// Flow Arrow Generation
// ============================================================================

/// Generate flow arrows from fluid particle data.
///
/// Samples velocity on a regular 3D grid by averaging nearby fluid particle
/// velocities using a radial kernel. Each grid cell with non-zero velocity
/// produces a flow arrow.
///
/// # Arguments
///
/// - `fluid_positions`: Fluid particle positions
/// - `fluid_velocities`: Fluid particle velocities
/// - `config`: Visualization configuration
#[must_use]
pub fn generate_flow_arrows(
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    config: &FlowVizConfig,
) -> Vec<FlowArrow> {
    if fluid_positions.is_empty() {
        return Vec::new();
    }

    let res = config.grid_resolution.max(1);
    let min = config.bounds_min;
    let max = config.bounds_max;

    let dx = (max.x - min.x) / Fix128::from_int(res as i64);
    let dy = (max.y - min.y) / Fix128::from_int(res as i64);
    let dz = (max.z - min.z) / Fix128::from_int(res as i64);

    // Cell size determines the search radius for averaging
    let cell_size = dx;
    let radius_sq = cell_size * cell_size;

    let mut arrows = Vec::new();

    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                // Cell center
                let center = Vec3Fix::new(
                    min.x + dx * Fix128::from_int(ix as i64) + dx.half(),
                    min.y + dy * Fix128::from_int(iy as i64) + dy.half(),
                    min.z + dz * Fix128::from_int(iz as i64) + dz.half(),
                );

                // Average velocity of nearby particles
                let mut avg_vel = Vec3Fix::ZERO;
                let mut count = Fix128::ZERO;

                for (pi, pos) in fluid_positions.iter().enumerate() {
                    let diff = *pos - center;
                    let dist_sq = diff.dot(diff);

                    if dist_sq < radius_sq {
                        if pi < fluid_velocities.len() {
                            avg_vel = avg_vel + fluid_velocities[pi];
                        }
                        count = count + Fix128::ONE;
                    }
                }

                if count.is_zero() {
                    continue;
                }

                avg_vel = avg_vel / count;
                let magnitude = avg_vel.length();

                if magnitude.is_zero() {
                    continue;
                }

                let direction = avg_vel / magnitude;

                arrows.push(FlowArrow {
                    position: center,
                    direction,
                    magnitude: magnitude * config.arrow_scale,
                });
            }
        }
    }

    arrows
}

// ============================================================================
// Streamline Generation
// ============================================================================

/// Generate streamlines by tracing particles through a velocity field.
///
/// Starting from each seed point, advances forward through the velocity
/// field (interpolated from nearby fluid particles) for the given number
/// of steps. Each streamline is a polyline of traced positions.
///
/// # Arguments
///
/// - `fluid_positions`: Fluid particle positions
/// - `fluid_velocities`: Fluid particle velocities
/// - `seed_points`: Starting positions for streamlines
/// - `steps`: Number of integration steps per streamline
/// - `dt`: Time step for Euler integration
#[must_use]
pub fn generate_streamlines(
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    seed_points: &[Vec3Fix],
    steps: usize,
    dt: Fix128,
) -> Vec<Vec<Vec3Fix>> {
    if fluid_positions.is_empty() || dt.is_zero() {
        return seed_points.iter().map(|s| vec![*s]).collect();
    }

    let influence_radius = Fix128::ONE;
    let radius_sq = influence_radius * influence_radius;

    seed_points
        .iter()
        .map(|&seed| {
            let mut line = Vec::with_capacity(steps + 1);
            let mut pos = seed;
            line.push(pos);

            for _ in 0..steps {
                // Interpolate velocity at current position
                let vel = interpolate_velocity(
                    pos,
                    fluid_positions,
                    fluid_velocities,
                    radius_sq,
                    influence_radius,
                );

                if vel.length_squared().is_zero() {
                    break; // No velocity, stop tracing
                }

                // Euler step
                pos = pos + vel * dt;
                line.push(pos);
            }

            line
        })
        .collect()
}

/// Interpolate velocity at a point using inverse-distance weighting.
fn interpolate_velocity(
    pos: Vec3Fix,
    fluid_positions: &[Vec3Fix],
    fluid_velocities: &[Vec3Fix],
    radius_sq: Fix128,
    radius: Fix128,
) -> Vec3Fix {
    let mut weighted_vel = Vec3Fix::ZERO;
    let mut total_weight = Fix128::ZERO;

    for (i, fp) in fluid_positions.iter().enumerate() {
        let diff = pos - *fp;
        let dist_sq = diff.dot(diff);

        if dist_sq >= radius_sq || i >= fluid_velocities.len() {
            continue;
        }

        // Linear weight: 1 - dist/radius
        let dist = dist_sq.sqrt();
        let weight = Fix128::ONE - dist / radius;

        if weight > Fix128::ZERO {
            weighted_vel = weighted_vel + fluid_velocities[i] * weight;
            total_weight = total_weight + weight;
        }
    }

    if total_weight.is_zero() {
        Vec3Fix::ZERO
    } else {
        weighted_vel / total_weight
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_config() -> FlowVizConfig {
        FlowVizConfig {
            grid_resolution: 2,
            bounds_min: Vec3Fix::from_int(-2, -2, -2),
            bounds_max: Vec3Fix::from_int(2, 2, 2),
            arrow_scale: Fix128::ONE,
        }
    }

    #[test]
    fn test_flow_arrows_empty_fluid() {
        let config = basic_config();
        let arrows = generate_flow_arrows(&[], &[], &config);
        assert!(arrows.is_empty());
    }

    #[test]
    fn test_flow_arrows_uniform_field() {
        let config = FlowVizConfig {
            grid_resolution: 2,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
            arrow_scale: Fix128::ONE,
        };
        // Particles everywhere moving in +X
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        for ix in -1..=1 {
            for iy in -1..=1 {
                for iz in -1..=1 {
                    positions.push(Vec3Fix::from_int(ix, iy, iz));
                    velocities.push(Vec3Fix::from_int(1, 0, 0));
                }
            }
        }

        let arrows = generate_flow_arrows(&positions, &velocities, &config);
        // Should produce some arrows
        assert!(!arrows.is_empty());

        // All arrows should point in +X direction
        for arrow in &arrows {
            assert!(arrow.direction.x > Fix128::ZERO);
            assert!(arrow.magnitude > Fix128::ZERO);
        }
    }

    #[test]
    fn test_flow_arrows_magnitude_scaled() {
        let config = FlowVizConfig {
            grid_resolution: 1,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
            arrow_scale: Fix128::from_int(2),
        };
        let positions = vec![Vec3Fix::ZERO];
        let velocities = vec![Vec3Fix::from_int(3, 0, 0)];

        let arrows = generate_flow_arrows(&positions, &velocities, &config);
        if !arrows.is_empty() {
            // Magnitude should be velocity * scale
            assert!(arrows[0].magnitude > Fix128::from_int(5));
        }
    }

    #[test]
    fn test_streamlines_empty_fluid() {
        let seeds = vec![Vec3Fix::ZERO];
        let lines = generate_streamlines(&[], &[], &seeds, 10, Fix128::from_ratio(1, 60));
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), 1); // Only seed point
    }

    #[test]
    fn test_streamlines_single_particle() {
        let positions = vec![Vec3Fix::ZERO];
        let velocities = vec![Vec3Fix::from_int(1, 0, 0)];
        let seeds = vec![Vec3Fix::ZERO];

        let lines = generate_streamlines(
            &positions,
            &velocities,
            &seeds,
            5,
            Fix128::from_ratio(1, 10),
        );
        assert_eq!(lines.len(), 1);
        // Should have seed + some trace points
        assert!(lines[0].len() > 1);
        // Should have moved in +X direction
        let last = lines[0].last().unwrap();
        assert!(last.x > Fix128::ZERO);
    }

    #[test]
    fn test_streamlines_zero_dt() {
        let positions = vec![Vec3Fix::ZERO];
        let velocities = vec![Vec3Fix::from_int(1, 0, 0)];
        let seeds = vec![Vec3Fix::ZERO];

        let lines = generate_streamlines(&positions, &velocities, &seeds, 10, Fix128::ZERO);
        assert_eq!(lines[0].len(), 1); // Only seed
    }

    #[test]
    fn test_streamlines_multiple_seeds() {
        let positions = vec![Vec3Fix::ZERO, Vec3Fix::from_int(1, 0, 0)];
        let velocities = vec![Vec3Fix::UNIT_X, Vec3Fix::UNIT_X];
        let seeds = vec![
            Vec3Fix::from_int(-1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(0, 0, -1),
        ];

        let lines = generate_streamlines(
            &positions,
            &velocities,
            &seeds,
            3,
            Fix128::from_ratio(1, 10),
        );
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_interpolate_velocity_no_neighbors() {
        let vel = interpolate_velocity(
            Vec3Fix::from_int(100, 100, 100),
            &[Vec3Fix::ZERO],
            &[Vec3Fix::from_int(1, 0, 0)],
            Fix128::ONE,
            Fix128::ONE,
        );
        assert!(vel.length_squared().is_zero());
    }

    #[test]
    fn test_flow_arrow_direction_normalized() {
        let config = FlowVizConfig {
            grid_resolution: 1,
            bounds_min: Vec3Fix::from_int(-1, -1, -1),
            bounds_max: Vec3Fix::from_int(1, 1, 1),
            arrow_scale: Fix128::ONE,
        };
        let positions = vec![Vec3Fix::ZERO];
        let velocities = vec![Vec3Fix::from_int(3, 4, 0)];

        let arrows = generate_flow_arrows(&positions, &velocities, &config);
        if !arrows.is_empty() {
            let len = arrows[0].direction.length();
            let eps = Fix128::from_ratio(1, 100);
            let diff = (len - Fix128::ONE).abs();
            assert!(diff < eps, "Direction should be normalized");
        }
    }
}
