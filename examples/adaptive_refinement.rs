//! Error-driven adaptive refinement: fewer nodes, less error
//!
//! A steel block is clamped on one face and pulled by a uniform traction on a
//! single cell of the opposite one. The traction jumps at the edge of that patch,
//! so the stress concentrates there and a uniform mesh spends most of its nodes
//! where nothing is happening.
//!
//! The run prints three solves of the same problem:
//!
//! 1. a **reference**, uniformly refined until it is much finer than the others
//! 2. a **uniform** refinement of the base mesh, one step
//! 3. an **adaptive** run, which scores each element, marks a bulk set and splits
//!    only those
//!
//! and then does the same adaptive step by hand — indicators, marking,
//! refinement — so the three pieces are visible separately.
//!
//! The measure is the strain energy `½ uᵀf`, which has a finite limit here
//! because the traction is distributed. ⚠️ A **point** load would be the obvious
//! way to localize the error and it is the wrong one: the solution for a Dirac
//! force in three dimensions has infinite energy, so neither the energy nor the
//! error in the energy norm converges, and comparing meshes means nothing.
//!
//! ```bash
//! cargo run --example adaptive_refinement --features std
//! ```

use alice_physics::linear_elastic_fem::{
    error_indicators_squared, mark_bulk, solve, solve_adaptive, AdaptiveConfig, AdaptiveSolution,
    Axis, BoundaryConditions, ElasticMaterial, FemSolution, SolverConfig, StressTensor,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Young's modulus, MPa (mild steel, rounded to a power of two).
const E_MPA: f64 = 204_800.0;
/// Poisson's ratio.
const NU: f64 = 0.25;
/// Traction on the loaded patch, MPa.
const TRACTION: f64 = 64.0;
/// Cells along each axis of the base block.
const CELLS: (usize, usize, usize) = (3, 2, 2);

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn half() -> Fix128 {
    Fix128::from_raw(0, 1 << 63)
}

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet subdivision of a box, conforming across cell faces.
fn kuhn_box(nx: usize, ny: usize, nz: usize) -> SdfTetMesh {
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
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

/// The four nodes of the loaded patch and their share of a uniform traction on a
/// square face split into two triangles: `A/3` on the diagonal pair, `A/6` on the
/// other two.
fn patch_nodes() -> [(u32, f64); 4] {
    let (nx, ny, _) = CELLS;
    [
        (node_index(nx, ny, nx, 0, 0), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 1), 1.0 / 3.0),
        (node_index(nx, ny, nx, 1, 0), 1.0 / 6.0),
        (node_index(nx, ny, nx, 0, 1), 1.0 / 6.0),
    ]
}

/// Clamp `x = 0`, pull the patch. Rebuilt for every mesh — see
/// `solve_adaptive` on why the driver insists on that.
fn boundary_for(mesh: &SdfTetMesh) -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[0].abs() < 0.5 {
            let node = u32::try_from(v).expect("fits u32");
            bc.prescribe(node, Axis::X, Fix128::ZERO);
            bc.prescribe(node, Axis::Y, Fix128::ZERO);
            bc.prescribe(node, Axis::Z, Fix128::ZERO);
        }
    }
    for (node, share) in patch_nodes() {
        bc.add_load(node, Axis::X, fx(TRACTION * share));
    }
    bc
}

/// `½ uᵀf` over the loaded patch.
fn strain_energy(sol: &FemSolution) -> Fix128 {
    let mut work = Fix128::ZERO;
    for (node, share) in patch_nodes() {
        work = work + fx(TRACTION * share) * sol.displacements[node as usize][0];
    }
    half() * work
}

fn main() {
    let (nx, ny, nz) = CELLS;
    let base = kuhn_box(nx, ny, nz);
    let material = ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid steel");
    let linear = SolverConfig::default();

    println!(
        "block {nx}x{ny}x{nz} cells, {} tets, {} nodes; traction {TRACTION} MPa on one face cell\n",
        base.tets.len(),
        base.vertices.len()
    );

    // --- the three solves ---------------------------------------------------
    let mut reference_mesh = base.clone();
    reference_mesh
        .try_refine_conforming(0.6, 24)
        .expect("reference refinement finishes");
    let reference = solve(
        &reference_mesh,
        &material,
        &boundary_for(&reference_mesh),
        &linear,
    )
    .expect("reference solve succeeds");
    let reference_energy = strain_energy(&reference);

    let mut uniform_mesh = base.clone();
    uniform_mesh
        .try_refine_conforming(0.95, 24)
        .expect("uniform refinement finishes");
    let uniform = solve(
        &uniform_mesh,
        &material,
        &boundary_for(&uniform_mesh),
        &linear,
    )
    .expect("uniform solve succeeds");

    let config = AdaptiveConfig::try_new(linear, half(), 6, 24).expect("valid adaptive config");
    let adaptive: AdaptiveSolution =
        solve_adaptive(&base, &material, boundary_for, &config).expect("adaptive solve succeeds");

    println!(
        "config: θ = {}, at most {} rounds, {} refinement passes each, CG tolerance {:e}",
        config.bulk_fraction().to_f64(),
        config.max_rounds(),
        config.max_refine_passes(),
        config.linear().relative_tolerance().to_f64()
    );
    println!();
    println!(
        "{:<12} {:>7} {:>7} {:>16} {:>12}",
        "run", "tets", "nodes", "energy", "error"
    );
    let row = |name: &str, mesh: &SdfTetMesh, sol: &FemSolution| {
        println!(
            "{name:<12} {:>7} {:>7} {:>16.9} {:>12.3e}",
            mesh.tets.len(),
            mesh.vertices.len(),
            strain_energy(sol).to_f64(),
            (reference_energy - strain_energy(sol)).abs().to_f64()
        );
    };
    row("reference", &reference_mesh, &reference);
    row("uniform", &uniform_mesh, &uniform);
    row("adaptive", &adaptive.mesh, &adaptive.field);
    println!(
        "\nadaptive used {} rounds; total indicator per round: {:?}",
        adaptive.rounds,
        adaptive
            .total_indicator_history
            .iter()
            .map(|v| v.to_f64())
            .collect::<Vec<_>>()
    );

    // --- the same step by hand ----------------------------------------------
    println!("\none adaptive step, piece by piece:");
    let first = solve(&base, &material, &boundary_for(&base), &linear).expect("solve succeeds");
    let indicators =
        error_indicators_squared(&base, &material, &first).expect("indicators computed");
    let total = indicators.iter().fold(Fix128::ZERO, |a, &b| a + b);
    let worst = indicators
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.cmp(b.1))
        .expect("non-empty");
    println!(
        "  Σ η² = {:.6}; worst element {} carries {:.1}% of it",
        total.to_f64(),
        worst.0,
        100.0 * worst.1.to_f64() / total.to_f64()
    );

    let marked = mark_bulk(&indicators, half()).expect("valid marking");
    let marked_count = marked.iter().filter(|&&m| m).count();
    println!(
        "  bulk marking at θ = 1/2 chose {marked_count} of {} elements",
        marked.len()
    );

    let mut refined = base.clone();
    let passes = refined
        .try_refine_marked(&marked, 24)
        .expect("refinement finishes");
    println!(
        "  refining them took {passes} passes: {} -> {} tets, {} -> {} nodes",
        base.tets.len(),
        refined.tets.len(),
        base.vertices.len(),
        refined.vertices.len()
    );
    println!(
        "  ⚠️ more elements split than were marked ({} vs {marked_count}): the extra ones are \
         propagation keeping the mesh conforming",
        refined.tets.len() - base.tets.len()
    );

    // --- the norm the indicator is built on ---------------------------------
    println!("\nthe energy norm σ:C⁻¹:σ, on three states with closed forms:");
    let s = Fix128::from_int(8);
    for (name, stress, closed) in [
        (
            "uniaxial",
            StressTensor {
                xx: s,
                ..Default::default()
            },
            "s²/E",
        ),
        (
            "pure shear",
            StressTensor {
                xy: s,
                ..Default::default()
            },
            "2(1+ν)s²/E",
        ),
        (
            "hydrostatic",
            StressTensor {
                xx: s,
                yy: s,
                zz: s,
                ..Default::default()
            },
            "3(1−2ν)p²/E",
        ),
    ] {
        println!(
            "  {name:<12} {:.9}   ({closed})",
            stress.complementary_energy_density(&material).to_f64()
        );
    }
}
