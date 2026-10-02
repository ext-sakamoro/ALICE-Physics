//! Configuration knobs and solve diagnostics for the linear elastic FEM
//! solver, none of which `solve`'s own smoke tests exercise because they all
//! default to a value that already works.
//!
//! - [`ElasticMaterial::from_filament`] / [`ElasticMaterial::default_poissons_ratio`]
//!   / [`ElasticMaterial::with_poisson`]: building a material from the
//!   filament database on a **non-FDM** category, so a category-table lookup
//!   that silently fell back to the FDM default would show up as the wrong
//!   number — SheetMetal's 0.30 is used here, not FDM's 0.35.
//! - [`SolverConfig::with_preconditioner`] / [`SolverConfig::with_stagnation_fraction`]:
//!   read back from the config that carried them into the solve.
//! - [`BoundaryConditions::prescribed_count`] / [`BoundaryConditions::load_count`]:
//!   counted after one vertex receives two loads on the same axis, which
//!   accumulate into one entry rather than two.
//! - [`stiffness_diagonal_stats`] / [`DiagonalStats`]: the conditioning
//!   diagnostic, computed without running an iteration.
//! - [`ThermalExpansion::alpha_per_k`]: read back from the type that carries
//!   the coefficient into the eigenstrain term of the residual.
//! - [`StressTensor::hydrostatic`]: the mean normal stress of the worst
//!   element in the solved field.
//!
//! ```bash
//! cargo run --example linear_elastic_fem_config_diagnostics --features std
//! ```

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::linear_elastic_fem::{
    solve_with_eigenstrain, stiffness_diagonal_stats, Axis, BoundaryConditions, ElasticMaterial,
    Preconditioner, SolverConfig, StressTensor, ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Node index within an `(n+1)³` lattice.
fn node(n: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (n + 1) + k * (n + 1) * (n + 1)).expect("lattice fits u32")
}

/// A cube of side `n·h`, split into Kuhn 6-tet cells (same construction as
/// `examples/thermoelastic_rise.rs`).
fn kuhn_cube(n: usize, h: f64) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                mesh.vertices.push([
                    (i as f64 * h) as f32,
                    (j as f64 * h) as f32,
                    (k as f64 * h) as f32,
                ]);
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
                    corners[0] = node(n, i, j, k);
                    for (slot, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[slot + 1] = node(n, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

fn main() {
    println!("[linear_elastic_fem] config + diagnostics walkthrough");

    // --- Material: filament database, a non-FDM category -------------------
    //
    // SUS304 is `MaterialCategory::SheetMetal`, whose table default is 0.30 —
    // distinct from the `Fdm` default (0.35) that every existing
    // `from_filament` scene in this crate happens to use. A wiring mutation
    // that ignored the category and always read the `Fdm` row would be
    // invisible on an all-FDM scene and visible on this one.
    let sheet = MaterialProperties::sus304();
    let table_nu = ElasticMaterial::default_poissons_ratio(sheet.category);
    let from_table = ElasticMaterial::from_filament(&sheet).expect("SUS304 is a valid material");
    println!(
        "[linear_elastic_fem] from_filament(SUS304): E = {:.1} MPa, ν = {:.3} \
         (default_poissons_ratio(SheetMetal) = {:.3})",
        from_table.youngs_modulus_mpa().to_f64(),
        from_table.poissons_ratio().to_f64(),
        table_nu.to_f64()
    );

    // A measured value overrides the table default without disturbing E.
    let measured = from_table
        .with_poisson(Fix128::from_ratio(33, 100))
        .expect("0.33 is a valid Poisson's ratio");
    println!(
        "[linear_elastic_fem] with_poisson(0.33): E = {:.1} MPa (unchanged), ν = {:.3}",
        measured.youngs_modulus_mpa().to_f64(),
        measured.poissons_ratio().to_f64()
    );

    // --- Solver config -------------------------------------------------------
    let config = SolverConfig::default()
        .with_preconditioner(Preconditioner::JacobiScaled)
        .with_stagnation_fraction(Fix128::from_ratio(3, 10))
        .expect("0.3 is strictly between 0 and 1");
    println!(
        "[linear_elastic_fem] solver config: preconditioner = {:?}, \
         stagnation_window_fraction = {:.2}",
        config.preconditioner(),
        config.stagnation_window_fraction().to_f64()
    );

    // --- Mesh + boundary conditions ------------------------------------------
    let n = 2usize;
    let side = 2.0;
    let mesh = kuhn_cube(n, side / n as f64);

    let mut bc = BoundaryConditions::new();
    for (v, p) in mesh.vertices.iter().enumerate() {
        if p[2] <= 0.0 {
            bc.fix(u32::try_from(v).expect("fits"));
        }
    }
    // Two loads at the same (vertex, axis): they accumulate into one entry,
    // so `load_count` reports 1, not 2.
    let top = node(n, 1, 1, n);
    bc.add_load(top, Axis::Z, Fix128::from_int(-10));
    bc.add_load(top, Axis::Z, Fix128::from_int(-5));
    println!(
        "[linear_elastic_fem] boundary conditions: prescribed_count = {}, load_count = {} \
         (two add_load calls at the same DOF, so load_count stays 1)",
        bc.prescribed_count(),
        bc.load_count()
    );

    // --- Stiffness-diagonal diagnostic, no iteration run --------------------
    let stats = stiffness_diagonal_stats(&mesh, &measured, &bc).expect("well posed");
    println!(
        "[linear_elastic_fem] stiffness diagonal: free_dofs = {}, min = {:.4e}, \
         mean = {:.4e}, max = {:.4e} MPa/mm",
        stats.free_dofs,
        stats.min.to_f64(),
        stats.mean.to_f64(),
        stats.max.to_f64()
    );

    // --- Thermal eigenstrain --------------------------------------------------
    let absolute = CoupledField::try_new_filled(
        2,
        2,
        2,
        (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        (
            Fix128::from_int(side as i64),
            Fix128::from_int(side as i64),
            Fix128::from_int(side as i64),
        ),
        Fix128::from_int(45),
    )
    .expect("a 2^3 grid over the cube is valid");
    let rise = TemperatureRise::from_absolute(&absolute, Fix128::from_int(25));
    let alpha_per_k = Fix128::from_ratio(12, 1_000_000);
    let thermal = ThermalExpansion::from_rise(&rise, alpha_per_k);
    println!(
        "[linear_elastic_fem] thermal eigenstrain: alpha_per_k = {:.2e} K^-1 (read back from \
         the type that carries it into the residual)",
        thermal.alpha_per_k().to_f64()
    );

    let out = solve_with_eigenstrain(&mesh, &measured, &bc, &config, Some(thermal))
        .expect("clamped base, loaded top, well posed");

    let worst_hydrostatic = out
        .element_stress
        .iter()
        .map(StressTensor::hydrostatic)
        .fold(
            Fix128::ZERO,
            |worst, h| {
                if h.abs() > worst.abs() {
                    h
                } else {
                    worst
                }
            },
        );
    println!(
        "[linear_elastic_fem] solve: {} CG iterations, relative residual {:.3e}, \
         worst hydrostatic stress = {:.4} MPa",
        out.iterations,
        out.relative_residual.to_f64(),
        worst_hydrostatic.to_f64()
    );
}
