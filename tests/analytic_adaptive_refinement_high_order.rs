//! Oracles for error-driven adaptive refinement on the quadratic (P2) and cubic
//! (P3) tetrahedra.
//!
//! `analytic_adaptive_refinement.rs` pins the P1 driver. The roadmap recorded the
//! higher-order version as "not started: the refinement of `QuadraticMesh` /
//! `CubicMesh` is a separate piece of work". It turns out to need no new mesh
//! refinement at all: both meshes are *built from* a straight-edged `SdfTetMesh`,
//! so the driver refines that corner mesh with the conforming
//! `try_refine_marked` and rebuilds the high-order mesh each round.
//!
//! # What is pinned
//!
//! | claim | test |
//! |---|---|
//! | an exact (affine) solution has nothing to refine: one solve, mesh untouched | `an_exact_solution_stops_the_{quadratic,cubic}_driver_immediately` |
//! | on a smooth problem the total indicator falls as the driver refines | `the_total_indicator_falls_for_{quadratic,cubic}` |
//! | the driver actually refines (more elements, more than one round) and the solution is indexed by the rebuilt high-order mesh | same tests |
//!
//! ⚠️ **What is *not* measured here:** that adaptive P2 / P3 beats uniform
//! refinement per degree of freedom. The P1 driver has that test
//! (`adaptive_refinement_beats_uniform_per_degree_of_freedom`); the higher-order
//! one reuses the P1 recovery estimator on the centroid stress, which for an
//! element whose true stress is linear *within* the element is a coarser signal,
//! and its payoff has not been measured. Do not read the passing tests here as
//! that claim.
//!
//! ⚠️ **The boundary-condition closure receives the high-order mesh.** A
//! higher-order condition must constrain the edge (and face) nodes of the
//! clamped faces as well, and only that mesh knows where they are; the helper
//! below prescribes by node *position*, so it covers them.

#![allow(clippy::disallowed_methods)]

use alice_physics::cubic_elastic_fem::{solve_adaptive_cubic, CubicMesh};
use alice_physics::linear_elastic_fem::{
    AdaptiveConfig, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{solve_adaptive_quadratic, QuadraticMesh};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn material() -> ElasticMaterial {
    ElasticMaterial::new(fx(1024.0), fx(0.25)).expect("valid")
}

fn node_index(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("fits")
}

/// Kuhn 6-tet cube `[0, n]³`, unit cells.
fn kuhn_cube(n: usize) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(n, i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = node_index(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn config(rounds: u32) -> AdaptiveConfig {
    AdaptiveConfig::try_new(SolverConfig::default(), Fix128::ONE, rounds, 32).expect("valid")
}

/// Positions of every node of a high-order mesh, in `f64`.
fn positions(count: usize, at: impl Fn(u32) -> [Fix128; 3]) -> Vec<[f64; 3]> {
    (0..count)
        .map(|n| {
            let p = at(u32::try_from(n).expect("fits"));
            [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()]
        })
        .collect()
}

/// Every node prescribed to the affine field `u = (ε x, 0, 0)`, which lies in
/// every polynomial space.
fn affine_bc(pos: &[[f64; 3]], strain: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (n, p) in pos.iter().enumerate() {
        bc.prescribe_all(
            u32::try_from(n).expect("fits"),
            [fx(strain * p[0]), Fix128::ZERO, Fix128::ZERO],
        );
    }
    bc
}

/// `u = (s·y², 0, 0)` on the surface only, the interior free: smooth, with no
/// singularity, and outside the P1 / P2 spaces.
fn smooth_bc(pos: &[[f64; 3]], extent: f64, scale: f64) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (n, p) in pos.iter().enumerate() {
        if p.iter()
            .any(|&c| c.abs() < 1e-6 || (c - extent).abs() < 1e-6)
        {
            bc.prescribe_all(
                u32::try_from(n).expect("fits"),
                [fx(scale * p[1] * p[1]), Fix128::ZERO, Fix128::ZERO],
            );
        }
    }
    bc
}

fn quad_pos(m: &QuadraticMesh) -> Vec<[f64; 3]> {
    positions(m.node_count(), |n| m.node_position(n).expect("in range"))
}

fn cubic_pos(m: &CubicMesh) -> Vec<[f64; 3]> {
    positions(m.node_count(), |n| m.node_position(n).expect("in range"))
}

#[test]
fn an_exact_solution_stops_the_quadratic_driver_immediately() {
    let base = kuhn_cube(2);
    let out = solve_adaptive_quadratic(
        &base,
        &material(),
        |m| affine_bc(&quad_pos(m), 0.125),
        &config(5),
    )
    .expect("adaptive solve");
    assert_eq!(out.rounds, 1, "one solve, nothing to refine");
    assert_eq!(
        out.mesh.tets, base.tets,
        "the mesh must come back unrefined"
    );
    assert_eq!(out.mesh.vertices, base.vertices);
}

#[test]
fn an_exact_solution_stops_the_cubic_driver_immediately() {
    let base = kuhn_cube(2);
    let out = solve_adaptive_cubic(
        &base,
        &material(),
        |m| affine_bc(&cubic_pos(m), 0.125),
        &config(5),
    )
    .expect("adaptive solve");
    assert_eq!(out.rounds, 1);
    assert_eq!(out.mesh.tets, base.tets);
}

#[test]
fn the_total_indicator_falls_for_quadratic() {
    let base = kuhn_cube(2);
    let out = solve_adaptive_quadratic(
        &base,
        &material(),
        |m| smooth_bc(&quad_pos(m), 2.0, 1.0 / 64.0),
        &config(4),
    )
    .expect("adaptive solve");
    let h = &out.total_indicator_history;
    eprintln!(
        "[adaptive-p2] history {:?}",
        h.iter().map(|v| v.to_f64()).collect::<Vec<_>>()
    );
    assert!(
        h.len() >= 2 && out.rounds >= 2,
        "the driver must refine at least once"
    );
    assert!(
        out.mesh.tets.len() > base.tets.len(),
        "refinement must add elements"
    );
    assert_eq!(
        out.field.displacements.len(),
        out.high_order.node_count(),
        "the solution is indexed by the rebuilt high-order mesh"
    );
    assert!(
        h[0] > Fix128::ZERO,
        "a quadratic boundary field is not exact for P2 recovery"
    );
    assert!(
        *h.last().expect("non-empty") < h[0],
        "the estimate must fall: {h:?}"
    );
}

#[test]
fn the_total_indicator_falls_for_cubic() {
    let base = kuhn_cube(2);
    let out = solve_adaptive_cubic(
        &base,
        &material(),
        |m| smooth_bc(&cubic_pos(m), 2.0, 1.0 / 64.0),
        &config(3),
    )
    .expect("adaptive solve");
    let h = &out.total_indicator_history;
    eprintln!(
        "[adaptive-p3] history {:?}",
        h.iter().map(|v| v.to_f64()).collect::<Vec<_>>()
    );
    assert!(h.len() >= 2 && out.rounds >= 2);
    assert!(out.mesh.tets.len() > base.tets.len());
    assert_eq!(out.field.displacements.len(), out.high_order.node_count());
    assert!(
        *h.last().expect("non-empty") < h[0],
        "the estimate must fall: {h:?}"
    );
}
