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
//! | adaptive beats uniform refinement per node | `adaptive_{quadratic,cubic}_beats_uniform_per_node` (`runtime:`) |
//! | the driver actually refines (more elements, more than one round) and the solution is indexed by the rebuilt high-order mesh | same tests |
//!
//! # The payoff, measured
//!
//! `adaptive_{quadratic,cubic}_beats_uniform_per_node` repeat the P1 payoff test
//! (`adaptive_refinement_beats_uniform_per_degree_of_freedom`) on the same block
//! and the same statically equivalent patch load, with Dörfler `θ = 1/2`. The
//! measure is the strain energy and the reference is a uniformly refined solve of
//! the same element, so the estimator does not grade itself.
//!
//! | element | uniform error (nodes) | adaptive error (nodes) |
//! |---|---|---|
//! | P2 | 4.941 (1053) | 1.961 (279) |
//! | P3 | 8.923 (3211) | 5.973 (710) |
//!
//! ⚠️ **`θ = 1` is uniform refinement under another name.** A first version of the
//! P2 test used the `config()` helper's `θ = 1`, marked every element and landed
//! on exactly the reference mesh (`3805` nodes, error `0`) — a result that would
//! have read as "adaptive is perfect". The reason is in the test.
//!
//! ⚠️ The patch load is a fixed set of four nodal forces on corner vertices, so the
//! continuum limit has infinite energy; the comparison is between fixed discrete
//! problems, as in the P1 test, and says nothing about convergence to a limit.
//!
//! ⚠️ **The boundary-condition closure receives the high-order mesh.** A
//! higher-order condition must constrain the edge (and face) nodes of the
//! clamped faces as well, and only that mesh knows where they are; the helper
//! below prescribes by node *position*, so it covers them.

#![allow(clippy::disallowed_methods)]

use alice_physics::cubic_elastic_fem::{solve_adaptive_cubic, solve_cubic, CubicMesh};
use alice_physics::linear_elastic_fem::{
    AdaptiveConfig, BoundaryConditions, ElasticMaterial, SolverConfig,
};
use alice_physics::math::Fix128;
use alice_physics::quadratic_elastic_fem::{
    solve_adaptive_quadratic, solve_quadratic, QuadraticMesh,
};
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

/// Kuhn 6-tet box `[0, nx] x [0, ny] x [0, nz]`, unit cells.
fn kuhn_box(nx: usize, ny: usize, nz: usize) -> SdfTetMesh {
    let idx = |i: usize, j: usize, k: usize| {
        u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("fits")
    };
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
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
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = idx(i, j, k);
                    for (m, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[m + 1] = idx(i + step[0], j + step[1], k + step[2]);
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

// ---------------------------------------------------------------------------
// the payoff: adaptive against uniform refinement, per node
// ---------------------------------------------------------------------------

const TRACTION: f64 = 64.0;
const BOX: [usize; 3] = [3, 2, 2];

/// The block of the P1 payoff test: clamped on `x = 0`, pulled by the statically
/// equivalent nodal load of a unit traction patch on the far face. The loaded
/// nodes are corner vertices, whose indices survive refinement (vertices are
/// appended), so one definition serves every mesh.
fn patch_bc(pos: &[[f64; 3]]) -> (BoundaryConditions, Vec<(u32, f64)>) {
    let mut bc = BoundaryConditions::new();
    for (n, p) in pos.iter().enumerate() {
        if p[0].abs() < 0.5 {
            bc.prescribe_all(
                u32::try_from(n).expect("fits"),
                [Fix128::ZERO, Fix128::ZERO, Fix128::ZERO],
            );
        }
    }
    let (nx, ny) = (BOX[0], BOX[1]);
    let corner = |j: usize, k: usize| {
        u32::try_from(nx + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("fits")
    };
    let loaded = vec![
        (corner(0, 0), 1.0 / 3.0),
        (corner(1, 1), 1.0 / 3.0),
        (corner(1, 0), 1.0 / 6.0),
        (corner(0, 1), 1.0 / 6.0),
    ];
    for &(node, share) in &loaded {
        bc.add_load(
            node,
            alice_physics::linear_elastic_fem::Axis::X,
            fx(TRACTION * share),
        );
    }
    (bc, loaded)
}

fn work(displacements: &[[Fix128; 3]], loaded: &[(u32, f64)]) -> f64 {
    0.5 * loaded
        .iter()
        .map(|&(n, share)| TRACTION * share * displacements[n as usize][0].to_f64())
        .sum::<f64>()
}

#[test]
#[ignore = "runtime: about 20 s in release (P2 reference at two uniform passes, a uniform coarse solve and a six-round adaptive run); run by run_ignored.py"]
fn adaptive_quadratic_beats_uniform_per_node() {
    let base = kuhn_box(BOX[0], BOX[1], BOX[2]);
    let solve_on = |mesh: &SdfTetMesh| {
        let m = QuadraticMesh::from_tet_mesh(mesh).expect("mesh");
        let (bc, loaded) = patch_bc(&quad_pos(&m));
        let f = solve_quadratic(&m, &material(), &bc, &SolverConfig::default()).expect("solve");
        (work(&f.displacements, &loaded), m.node_count())
    };
    let mut reference_mesh = base.clone();
    reference_mesh
        .try_refine_conforming(0.6, 24)
        .expect("refine");
    let (reference, ref_nodes) = solve_on(&reference_mesh);
    let mut uniform_mesh = base.clone();
    uniform_mesh
        .try_refine_conforming(0.95, 24)
        .expect("refine");
    let (uniform, uniform_nodes) = solve_on(&uniform_mesh);

    // Dörfler θ = 1/2 as in the P1 payoff test. `config()` above uses θ = 1, which
    // marks every element and is uniform refinement under another name.
    let half = AdaptiveConfig::try_new(SolverConfig::default(), Fix128::from_ratio(1, 2), 6, 24)
        .expect("valid");
    let out = solve_adaptive_quadratic(&base, &material(), |m| patch_bc(&quad_pos(m)).0, &half)
        .expect("adaptive");
    let (_, loaded) = patch_bc(&quad_pos(&out.high_order));
    let adaptive = work(&out.field.displacements, &loaded);
    let adaptive_nodes = out.high_order.node_count();

    let (u_err, a_err) = ((reference - uniform).abs(), (reference - adaptive).abs());
    eprintln!(
        "[adaptive-p2] reference {reference:.9} ({ref_nodes} nodes)\n  uniform  err {u_err:.3e} at {uniform_nodes} nodes\n  adaptive err {a_err:.3e} at {adaptive_nodes} nodes"
    );
    assert!(
        adaptive_nodes <= uniform_nodes,
        "adaptive spent more nodes: {adaptive_nodes} vs {uniform_nodes}"
    );
    assert!(
        a_err < u_err,
        "adaptive must be closer at no more cost: {a_err:.3e} vs {u_err:.3e}"
    );
}

#[test]
#[ignore = "runtime: about 40 s in release (P3 reference at two uniform passes, a uniform coarse solve and a six-round adaptive run); run by run_ignored.py"]
fn adaptive_cubic_beats_uniform_per_node() {
    let base = kuhn_box(BOX[0], BOX[1], BOX[2]);
    let solve_on = |mesh: &SdfTetMesh| {
        let m = CubicMesh::from_tet_mesh(mesh).expect("mesh");
        let (bc, loaded) = patch_bc(&cubic_pos(&m));
        let f = solve_cubic(&m, &material(), &bc, &SolverConfig::default()).expect("solve");
        (work(&f.displacements, &loaded), m.node_count())
    };
    let mut reference_mesh = base.clone();
    reference_mesh
        .try_refine_conforming(0.6, 24)
        .expect("refine");
    let (reference, ref_nodes) = solve_on(&reference_mesh);
    let mut uniform_mesh = base.clone();
    uniform_mesh
        .try_refine_conforming(0.95, 24)
        .expect("refine");
    let (uniform, uniform_nodes) = solve_on(&uniform_mesh);

    let half = AdaptiveConfig::try_new(SolverConfig::default(), Fix128::from_ratio(1, 2), 6, 24)
        .expect("valid");
    let out = solve_adaptive_cubic(&base, &material(), |m| patch_bc(&cubic_pos(m)).0, &half)
        .expect("adaptive");
    let (_, loaded) = patch_bc(&cubic_pos(&out.high_order));
    let adaptive = work(&out.field.displacements, &loaded);
    let adaptive_nodes = out.high_order.node_count();

    let (u_err, a_err) = ((reference - uniform).abs(), (reference - adaptive).abs());
    eprintln!(
        "[adaptive-p3] reference {reference:.9} ({ref_nodes} nodes)\n  uniform  err {u_err:.3e} at {uniform_nodes} nodes\n  adaptive err {a_err:.3e} at {adaptive_nodes} nodes"
    );
    assert!(
        adaptive_nodes <= uniform_nodes,
        "adaptive spent more nodes: {adaptive_nodes} vs {uniform_nodes}"
    );
    assert!(
        a_err < u_err,
        "adaptive must be closer at no more cost: {a_err:.3e} vs {u_err:.3e}"
    );
}
