//! Oracles for the production entry points of `alice_physics::soft_body_cut`
//! driven by `examples/soft_body_cutting.rs`: `CutPlane`, `CutResult`,
//! `cut_cloth`, and `cut_deformable`.
//!
//! # What this file is and is not
//!
//! The module's own `#[cfg(all(test, feature = "std"))]` block (in
//! `src/soft_body_cut.rs`) already covers: signed-distance sign for a point
//! above/below/on a plane, a 4-particle/2-edge split into two sides of two
//! each, a single crossing edge being removed with its intersection
//! particle created near the plane (epsilon-bounded), a fully-one-sided
//! 2-particle mesh producing no cut, `cut_cloth` matching `cut_deformable`
//! on a 2-particle mesh, the empty mesh, and an out-of-bounds constraint
//! index being skipped. None of that is repeated here. What this file adds:
//!
//! * `CutPlane` / `CutResult` as directly constructed struct literals,
//!   confirming the public fields round-trip unchanged,
//! * a multi-edge (6-edge tetrahedron) hand-counted split where *some* but
//!   not all edges from a single vertex cross the plane, with exact
//!   (not epsilon-bounded) intersection coordinates -- the plane and mesh
//!   are chosen so every interpolation parameter is a power-of-two
//!   fraction (`5/8`), which `Fix128` (binary fixed-point) represents with
//!   zero rounding error,
//! * the on-plane-vertex degeneracy: a vertex with signed distance exactly
//!   zero counts as the positive side (`d >= ZERO`), so an edge from that
//!   vertex to another positive-side vertex does *not* cross, while an
//!   edge from it to a negative-side vertex *does* cross with interpolation
//!   parameter `t = 0` exactly -- producing a "new" particle bit-identical
//!   to the already-existing on-plane vertex,
//! * a plane that misses an entire multi-particle, multi-edge mesh (every
//!   vertex strictly on one side), confirming the mesh passes through
//!   unsplit: no removed constraints, no new particles, every particle on
//!   the one side the plane leaves it on, and
//! * two cross-scenario invariants that do not depend on the specific
//!   geometry: every particle appears in exactly one of `side_a_particles`
//!   / `side_b_particles` (partition conservation), and
//!   `removed_constraints.len() == new_particles.len()` (the crossing
//!   denominator `d_i - d_j` can never be zero for a crossing edge, since
//!   crossing requires `d_i >= 0` and `d_j < 0` or vice versa, so the two
//!   signed distances can never be equal).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::soft_body_cut::{cut_cloth, cut_deformable, CutPlane, CutResult};

fn horizontal_plane(y: Fix128) -> CutPlane {
    CutPlane {
        point: Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO),
        normal: Vec3Fix::UNIT_Y,
    }
}

// ============================================================================
// CutPlane / CutResult -- struct literal round-trip
// ============================================================================

#[test]
fn cut_plane_fields_round_trip() {
    let plane = CutPlane {
        point: Vec3Fix::from_int(1, 2, 3),
        normal: Vec3Fix::from_int(0, 1, 0),
    };
    assert_eq!(plane.point, Vec3Fix::from_int(1, 2, 3));
    assert_eq!(plane.normal, Vec3Fix::from_int(0, 1, 0));
}

#[test]
fn cut_result_fields_round_trip() {
    let result = CutResult {
        side_a_particles: vec![0, 2],
        side_b_particles: vec![1],
        new_particles: vec![Vec3Fix::from_int(5, 5, 5)],
        removed_constraints: vec![(0, 1)],
    };
    assert_eq!(result.side_a_particles, vec![0, 2]);
    assert_eq!(result.side_b_particles, vec![1]);
    assert_eq!(result.new_particles, vec![Vec3Fix::from_int(5, 5, 5)]);
    assert_eq!(result.removed_constraints, vec![(0, 1)]);
}

// ============================================================================
// cut_deformable -- multi-edge hand-counted split, exact intersections
// ============================================================================

/// A tetrahedron: one vertex above the plane, three below. The three edges
/// from the apex cross; the three base-to-base edges do not. Every crossing
/// edge has d_apex = 5, d_base = -3, so t = 5/8 exactly (see the doc comment
/// in examples/soft_body_cutting.rs for the full derivation).
#[test]
fn cut_deformable_tetrahedron_hand_counted_split_and_intersections() {
    let particles = vec![
        Vec3Fix::from_int(10, 5, 0),
        Vec3Fix::from_int(9, -3, 0),
        Vec3Fix::from_int(11, -3, 0),
        Vec3Fix::from_int(10, -3, 2),
    ];
    let edges = vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let plane = horizontal_plane(Fix128::ZERO);

    let result = cut_deformable(&particles, &edges, &plane);

    assert_eq!(result.side_a_particles, vec![0]);
    assert_eq!(result.side_b_particles, vec![1, 2, 3]);
    assert_eq!(result.removed_constraints, vec![(0, 1), (0, 2), (0, 3)]);
    assert_eq!(
        result.new_particles,
        vec![
            Vec3Fix::new(Fix128::from_ratio(75, 8), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(Fix128::from_ratio(85, 8), Fix128::ZERO, Fix128::ZERO),
            Vec3Fix::new(Fix128::from_int(10), Fix128::ZERO, Fix128::from_ratio(5, 4)),
        ]
    );
}

// ============================================================================
// On-plane vertex degeneracy
// ============================================================================

/// q0 sits exactly on the plane (d = 0). Per `d >= Fix128::ZERO`, zero counts
/// as the positive side. Edge (0,2) -- on-plane to above -- does not cross
/// (both positive). Edge (0,1) -- on-plane to below -- does cross, with
/// t = d0 / (d0 - d1) = 0 / 4 = 0 exactly, so its "new" particle is
/// bit-identical to q0 itself.
#[test]
fn cut_plane_through_existing_vertex_degenerate_edges() {
    let particles = vec![
        Vec3Fix::from_int(0, 0, 0),  // q0, exactly on the plane
        Vec3Fix::from_int(0, -4, 0), // q1, below
        Vec3Fix::from_int(0, 3, 0),  // q2, above
    ];
    let edges = vec![(0, 1), (0, 2)];
    let plane = horizontal_plane(Fix128::ZERO);

    let result = cut_deformable(&particles, &edges, &plane);

    assert_eq!(result.side_a_particles, vec![0, 2]);
    assert_eq!(result.side_b_particles, vec![1]);
    assert_eq!(result.removed_constraints, vec![(0, 1)]);
    assert_eq!(result.new_particles, vec![Vec3Fix::from_int(0, 0, 0)]);
}

// ============================================================================
// Plane misses the mesh entirely
// ============================================================================

/// A multi-particle, multi-edge mesh entirely below a plane far above it:
/// the mesh must pass through unsplit.
#[test]
fn cut_plane_missing_mesh_entirely_leaves_topology_unsplit() {
    let particles = vec![
        Vec3Fix::from_int(0, 5, 0),
        Vec3Fix::from_int(1, 3, 0),
        Vec3Fix::from_int(-1, -5, 2),
        Vec3Fix::from_int(2, 0, -3),
    ];
    let edges = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
    let plane = horizontal_plane(Fix128::from_int(100));

    let result = cut_deformable(&particles, &edges, &plane);

    assert!(result.side_a_particles.is_empty());
    assert_eq!(result.side_b_particles, vec![0, 1, 2, 3]);
    assert!(result.removed_constraints.is_empty());
    assert!(result.new_particles.is_empty());
}

// ============================================================================
// Cross-scenario invariants: partition conservation, 1:1 crossing count
// ============================================================================

fn assert_conservation(particles: &[Vec3Fix], edges: &[(usize, usize)], plane: &CutPlane) {
    let result = cut_deformable(particles, edges, plane);
    assert_eq!(
        result.side_a_particles.len() + result.side_b_particles.len(),
        particles.len(),
        "partition conservation violated"
    );
    assert_eq!(
        result.removed_constraints.len(),
        result.new_particles.len(),
        "removed_constraints/new_particles 1:1 invariant violated"
    );
}

#[test]
fn cut_deformable_conservation_holds_across_scenarios() {
    // Tetrahedron, 1-vs-3 split.
    assert_conservation(
        &[
            Vec3Fix::from_int(10, 5, 0),
            Vec3Fix::from_int(9, -3, 0),
            Vec3Fix::from_int(11, -3, 0),
            Vec3Fix::from_int(10, -3, 2),
        ],
        &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        &horizontal_plane(Fix128::ZERO),
    );
    // Quad + diagonal, 2-vs-2 split.
    assert_conservation(
        &[
            Vec3Fix::from_int(0, 2, 0),
            Vec3Fix::from_int(2, 2, 0),
            Vec3Fix::from_int(0, -2, 0),
            Vec3Fix::from_int(2, -2, 0),
        ],
        &[(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)],
        &horizontal_plane(Fix128::ZERO),
    );
    // On-plane vertex degeneracy.
    assert_conservation(
        &[
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(0, -4, 0),
            Vec3Fix::from_int(0, 3, 0),
        ],
        &[(0, 1), (0, 2)],
        &horizontal_plane(Fix128::ZERO),
    );
    // Plane misses entirely.
    assert_conservation(
        &[
            Vec3Fix::from_int(0, 5, 0),
            Vec3Fix::from_int(1, 3, 0),
            Vec3Fix::from_int(-1, -5, 2),
            Vec3Fix::from_int(2, 0, -3),
        ],
        &[(0, 1), (1, 2), (2, 3), (0, 3)],
        &horizontal_plane(Fix128::from_int(100)),
    );
}

// ============================================================================
// cut_cloth -- hand-counted split on a 2x2 cloth patch, exact intersections
// ============================================================================

#[test]
fn cut_cloth_quad_hand_counted_split_and_intersections() {
    let particles = vec![
        Vec3Fix::from_int(0, 2, 0),
        Vec3Fix::from_int(2, 2, 0),
        Vec3Fix::from_int(0, -2, 0),
        Vec3Fix::from_int(2, -2, 0),
    ];
    let edges = vec![(0, 1), (2, 3), (0, 2), (1, 3), (0, 3)];
    let plane = horizontal_plane(Fix128::ZERO);

    let result = cut_cloth(&particles, &edges, &plane);

    assert_eq!(result.side_a_particles, vec![0, 1]);
    assert_eq!(result.side_b_particles, vec![2, 3]);
    assert_eq!(result.removed_constraints, vec![(0, 2), (1, 3), (0, 3)]);
    assert_eq!(
        result.new_particles,
        vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::from_int(1, 0, 0),
        ]
    );
}
