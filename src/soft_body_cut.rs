//! Soft Body Cutting (Topology Change)
//!
//! Cut deformable bodies and cloth along a plane, splitting particles into
//! two groups and removing constraints that cross the cut boundary.
//!
//! # Algorithm
//!
//! 1. Classify all particles as side A or side B based on signed distance
//!    to the cut plane.
//! 2. For constraints (edges) that span both sides, create new particles
//!    at the intersection point and remove the original constraint.
//! 3. Return a `CutResult` describing the topology change.
//!
//! All computations use deterministic 128-bit fixed-point arithmetic.

use crate::math::{Fix128, Vec3Fix};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Cut Types
// ============================================================================

/// A plane defined by a point and normal for cutting operations.
#[derive(Clone, Copy, Debug)]
pub struct CutPlane {
    /// A point on the plane
    pub point: Vec3Fix,
    /// Normal vector of the plane (should be normalized)
    pub normal: Vec3Fix,
}

/// Result of a cutting operation.
#[derive(Clone, Debug)]
pub struct CutResult {
    /// Particle indices on the positive side of the plane (side A)
    pub side_a_particles: Vec<usize>,
    /// Particle indices on the negative side of the plane (side B)
    pub side_b_particles: Vec<usize>,
    /// New particles created at cut intersections
    pub new_particles: Vec<Vec3Fix>,
    /// Constraints (edge pairs) that were removed because they cross the cut
    pub removed_constraints: Vec<(usize, usize)>,
}

// ============================================================================
// Cutting Operations
// ============================================================================

/// Signed distance from a point to a plane.
///
/// Positive means on the side the normal points to (side A),
/// negative means on the opposite side (side B).
#[inline]
fn signed_distance_to_plane(point: Vec3Fix, plane: &CutPlane) -> Fix128 {
    let diff = point - plane.point;
    diff.dot(plane.normal)
}

/// Cut a deformable body along a plane.
///
/// Splits particles into two groups and identifies constraints (edge pairs)
/// that cross the plane. For each crossing constraint, a new particle is
/// created at the intersection point.
///
/// # Arguments
///
/// - `particles`: Particle positions of the deformable body
/// - `constraints`: Edge constraints as pairs of particle indices `(i, j)`
/// - `plane`: The cutting plane
#[must_use]
pub fn cut_deformable(
    particles: &[Vec3Fix],
    constraints: &[(usize, usize)],
    plane: &CutPlane,
) -> CutResult {
    // Classify particles
    let mut side_a = Vec::new();
    let mut side_b = Vec::new();
    let mut signed_dists = Vec::with_capacity(particles.len());

    for (i, p) in particles.iter().enumerate() {
        let d = signed_distance_to_plane(*p, plane);
        signed_dists.push(d);
        if d >= Fix128::ZERO {
            side_a.push(i);
        } else {
            side_b.push(i);
        }
    }

    let mut new_particles = Vec::new();
    let mut removed_constraints = Vec::new();

    // Find constraints that cross the plane
    for &(i, j) in constraints {
        if i >= particles.len() || j >= particles.len() {
            continue;
        }

        let di = signed_dists[i];
        let dj = signed_dists[j];

        // Crossing occurs when signs differ
        let i_positive = di >= Fix128::ZERO;
        let j_positive = dj >= Fix128::ZERO;

        if i_positive != j_positive {
            removed_constraints.push((i, j));

            // Compute intersection point via linear interpolation
            let denom = di - dj;
            if !denom.is_zero() {
                let t = di / denom;
                let pi = particles[i];
                let pj = particles[j];
                let intersection = Vec3Fix::new(
                    pi.x + (pj.x - pi.x) * t,
                    pi.y + (pj.y - pi.y) * t,
                    pi.z + (pj.z - pi.z) * t,
                );
                new_particles.push(intersection);
            }
        }
    }

    CutResult {
        side_a_particles: side_a,
        side_b_particles: side_b,
        new_particles,
        removed_constraints,
    }
}

/// Cut a cloth mesh along a plane.
///
/// Similar to `cut_deformable` but operates on cloth-style edge lists
/// (pairs of particle indices representing mesh edges).
///
/// # Arguments
///
/// - `cloth_particles`: Cloth particle positions
/// - `cloth_edges`: Edge connectivity as pairs of particle indices
/// - `plane`: The cutting plane
#[must_use]
pub fn cut_cloth(
    cloth_particles: &[Vec3Fix],
    cloth_edges: &[(usize, usize)],
    plane: &CutPlane,
) -> CutResult {
    // Cloth cutting uses the same algorithm as deformable cutting
    cut_deformable(cloth_particles, cloth_edges, plane)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn horizontal_plane(y: Fix128) -> CutPlane {
        CutPlane {
            point: Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO),
            normal: Vec3Fix::UNIT_Y,
        }
    }

    #[test]
    fn test_signed_distance_positive() {
        let plane = horizontal_plane(Fix128::ZERO);
        let point = Vec3Fix::from_int(0, 5, 0);
        let d = signed_distance_to_plane(point, &plane);
        assert!(d > Fix128::ZERO);
    }

    #[test]
    fn test_signed_distance_negative() {
        let plane = horizontal_plane(Fix128::ZERO);
        let point = Vec3Fix::from_int(0, -5, 0);
        let d = signed_distance_to_plane(point, &plane);
        assert!(d.is_negative());
    }

    #[test]
    fn test_signed_distance_on_plane() {
        let plane = horizontal_plane(Fix128::from_int(3));
        let point = Vec3Fix::from_int(10, 3, -7);
        let d = signed_distance_to_plane(point, &plane);
        assert!(d.is_zero());
    }

    #[test]
    fn test_cut_splits_particles() {
        let particles = vec![
            Vec3Fix::from_int(0, 5, 0),  // above
            Vec3Fix::from_int(0, -5, 0), // below
            Vec3Fix::from_int(0, 3, 0),  // above
            Vec3Fix::from_int(0, -1, 0), // below
        ];
        let constraints = vec![(0, 1), (2, 3)];
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_deformable(&particles, &constraints, &plane);

        assert_eq!(result.side_a_particles.len(), 2); // particles 0, 2
        assert_eq!(result.side_b_particles.len(), 2); // particles 1, 3
    }

    #[test]
    fn test_cut_removes_crossing_constraints() {
        let particles = vec![
            Vec3Fix::from_int(0, 5, 0),  // above
            Vec3Fix::from_int(0, -5, 0), // below
        ];
        let constraints = vec![(0, 1)];
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_deformable(&particles, &constraints, &plane);

        assert_eq!(result.removed_constraints.len(), 1);
        assert_eq!(result.removed_constraints[0], (0, 1));
    }

    #[test]
    fn test_cut_creates_intersection_particles() {
        let particles = vec![Vec3Fix::from_int(0, 10, 0), Vec3Fix::from_int(0, -10, 0)];
        let constraints = vec![(0, 1)];
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_deformable(&particles, &constraints, &plane);

        // Should create one new particle at the intersection (y=0)
        assert_eq!(result.new_particles.len(), 1);
        // Intersection should be near y=0
        let eps = Fix128::from_ratio(1, 100);
        assert!(result.new_particles[0].y.abs() < eps);
    }

    #[test]
    fn test_cut_no_crossing() {
        // All particles on the same side
        let particles = vec![Vec3Fix::from_int(0, 5, 0), Vec3Fix::from_int(0, 10, 0)];
        let constraints = vec![(0, 1)];
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_deformable(&particles, &constraints, &plane);

        assert!(result.removed_constraints.is_empty());
        assert!(result.new_particles.is_empty());
        assert_eq!(result.side_a_particles.len(), 2);
        assert!(result.side_b_particles.is_empty());
    }

    #[test]
    fn test_cut_cloth_same_as_deformable() {
        let particles = vec![Vec3Fix::from_int(0, 1, 0), Vec3Fix::from_int(0, -1, 0)];
        let edges = vec![(0, 1)];
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_cloth(&particles, &edges, &plane);

        assert_eq!(result.removed_constraints.len(), 1);
        assert_eq!(result.new_particles.len(), 1);
    }

    #[test]
    fn test_empty_particles() {
        let result = cut_deformable(&[], &[], &horizontal_plane(Fix128::ZERO));
        assert!(result.side_a_particles.is_empty());
        assert!(result.side_b_particles.is_empty());
        assert!(result.new_particles.is_empty());
    }

    #[test]
    fn test_invalid_constraint_indices() {
        let particles = vec![Vec3Fix::from_int(0, 1, 0)];
        let constraints = vec![(0, 99)]; // 99 out of bounds
        let plane = horizontal_plane(Fix128::ZERO);

        let result = cut_deformable(&particles, &constraints, &plane);
        // Invalid constraint is skipped
        assert!(result.removed_constraints.is_empty());
    }
}
