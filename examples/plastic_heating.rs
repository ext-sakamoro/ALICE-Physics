//! Plastic work becomes heat, and the heat deforms the body
//!
//! The return leg of the thermo-mechanical coupling whose forward leg is
//! `solve_with_eigenstrain`: a steel bar is pulled past yield, the plastic work
//! it dissipates is converted by the Taylor-Quinney fraction into a
//! temperature rise, that rise is deposited on a grid, and the grid drives a
//! thermal eigenstrain that moves the bar.
//!
//! Two things are worth watching in the output.
//!
//! **The dissipation depends on how the load path was cut.** Integrating the
//! hardening law over the discrete path gives
//! `W_p = σ_y·ε̄_p + (H/2)(ε̄_p² + Σ Δε̄_k²)`, which exceeds the continuous
//! `σ_y·ε̄_p + (H/2)·ε̄_p²` by `(H/2)·Σ Δε̄_k²`. That term is first order in the
//! step, so the same end state reports a different plastic work depending on
//! the number of steps, while `ε̄_p` itself does not move at all. The table
//! prints both so the difference is visible.
//!
//! **The deposit conserves energy, not temperature.** Each element's rise is
//! scaled by `V_e / V_cell` before it is splatted, because a temperature is
//! intensive: without the scaling a sliver tetrahedron would warm the grid as
//! much as a fat one.
//!
//! ```bash
//! cargo run --example plastic_heating --features std
//! ```

use alice_physics::coupled_field::{CoupledField, TemperatureRise};
use alice_physics::linear_elastic_fem::{
    deposit_plastic_heat, plastic_temperature_rise, solve_elastoplastic, solve_with_eigenstrain,
    Axis, BoundaryConditions, ElasticMaterial, ElastoplasticConfig, PlasticHeating, SolverConfig,
    ThermalExpansion,
};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Young's modulus, MPa (mild steel, rounded to a power of two).
const E_MPA: f64 = 204_800.0;
/// Poisson's ratio.
const NU: f64 = 0.25;
/// Initial yield stress, MPa.
const SIGMA_Y: f64 = 256.0;
/// Plastic modulus `H = dσ_y/dε̄_p`, MPa.
const H_PLASTIC: f64 = 2048.0;
/// Taylor-Quinney fraction: the share of plastic work that becomes heat.
const BETA: f64 = 0.9;
/// Volumetric heat capacity `ρ c_p`, MPa/K (steel: 7850 kg/m³ x 486 J/(kg K)).
const C_V_MPA_PER_K: f64 = 3.82;
/// Linear expansion coefficient, K⁻¹ (steel).
const ALPHA_PER_K: f64 = 12.0 / 1_000_000.0;
/// Cell edge of the bar, mm; the bar is 2 x 1 x 1 cells.
const H_MM: f32 = 2.0;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("lattice fits u32")
}

/// Kuhn 6-tet subdivision of a box, conforming across cell faces.
fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
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

/// Stretch the bar to a uniform axial strain, with the lateral motion left
/// free apart from the rigid body modes.
fn stretch_to(strain: Fix128) -> BoundaryConditions {
    let (nx, ny, nz) = (2usize, 1usize, 1usize);
    let length = Fix128::from_f32(H_MM) * Fix128::from_int(nx as i64);
    let mut bc = BoundaryConditions::new();
    for k in 0..=nz {
        for j in 0..=ny {
            bc.prescribe(node_index(nx, ny, 0, j, k), Axis::X, Fix128::ZERO);
            bc.prescribe(node_index(nx, ny, nx, j, k), Axis::X, strain * length);
        }
    }
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(nx, ny, 0, ny, 0), Axis::Z, Fix128::ZERO);
    bc
}

/// Only the rigid body modes removed, so a thermal expansion shows up as
/// displacement rather than as stress.
fn float_free() -> BoundaryConditions {
    let mut bc = BoundaryConditions::new();
    bc.prescribe(node_index(2, 1, 0, 0, 0), Axis::X, Fix128::ZERO);
    bc.prescribe(node_index(2, 1, 0, 0, 0), Axis::Y, Fix128::ZERO);
    bc.prescribe(node_index(2, 1, 0, 0, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(2, 1, 0, 1, 0), Axis::X, Fix128::ZERO);
    bc.prescribe(node_index(2, 1, 0, 1, 0), Axis::Z, Fix128::ZERO);
    bc.prescribe(node_index(2, 1, 0, 0, 1), Axis::X, Fix128::ZERO);
    bc
}

/// `n` equal load steps ending at factor 1.
fn uniform_path(n: i64) -> Vec<Fix128> {
    (1..=n)
        .map(|k| Fix128::from_int(k) / Fix128::from_int(n))
        .collect()
}

/// Volume of a tetrahedron, from the mesh.
fn tet_volume(mesh: &SdfTetMesh, t: usize) -> f64 {
    let v = mesh.tets[t].vertices;
    let p = |i: usize| {
        let a = mesh.vertices[v[i] as usize];
        [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
    };
    let (p0, p1, p2, p3) = (p(0), p(1), p(2), p(3));
    let e = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (a, b, c) = (e(p1, p0), e(p2, p0), e(p3, p0));
    let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0]);
    det.abs() / 6.0
}

fn main() {
    let mesh = kuhn_box(2, 1, 1, H_MM);
    let material = ElasticMaterial::new(fx(E_MPA), fx(NU)).expect("valid steel");
    let config = ElastoplasticConfig::try_new(
        SolverConfig::default(),
        60,
        Fix128::from_raw(0, 1 << 24), // 2^-40
        fx(SIGMA_Y),
        fx(H_PLASTIC),
    )
    .expect("valid elastoplastic config");
    let heating =
        PlasticHeating::try_new(fx(BETA), fx(C_V_MPA_PER_K)).expect("valid Taylor-Quinney pair");

    // Yield strain is σ_y / E = 1/800; pull to four times that.
    let target_strain = fx(4.0 / 800.0);
    let bc = stretch_to(target_strain);

    println!(
        "Steel bar, {} tets, pulled to ε = {:.5} (yield at {:.5})",
        mesh.tets.len(),
        target_strain.to_f64(),
        SIGMA_Y / E_MPA
    );
    println!("β = {BETA}, c_v = {C_V_MPA_PER_K} MPa/K, H = {H_PLASTIC} MPa\n");

    println!(
        "{:>6} {:>14} {:>16} {:>14} {:>12}",
        "steps", "ε̄_p", "W_p (MPa)", "continuous", "ΔT (K)"
    );
    for n in [1i64, 2, 8, 64, 512] {
        let sol = solve_elastoplastic(&mesh, &material, &bc, &config, &uniform_path(n))
            .expect("elastoplastic solve succeeds");
        let rise = plastic_temperature_rise(&sol, &heating);
        let idx = sol
            .equivalent_plastic_strain
            .iter()
            .position(|e| !e.is_zero())
            .expect("the bar yields");
        let eq = sol.equivalent_plastic_strain[idx].to_f64();
        let continuous = SIGMA_Y * eq + 0.5 * H_PLASTIC * eq * eq;
        println!(
            "{n:>6} {:>14.9} {:>16.9} {:>14.9} {:>12.6}",
            eq,
            sol.dissipation[idx].to_f64(),
            continuous,
            rise[idx].to_f64()
        );
    }
    println!("\n⚠️ ε̄_p is the same for every step count; W_p is not. The excess over the");
    println!("   continuous form is (H/2)·Σ Δε̄_k², which is first order in the step.\n");

    // --- the loop: dissipation -> grid -> eigenstrain -> displacement -------
    let sol = solve_elastoplastic(&mesh, &material, &bc, &config, &uniform_path(8))
        .expect("solve succeeds");

    // One millimetre cells covering the 4 x 2 x 2 bar with a margin.
    let lo = -Fix128::ONE;
    let mut field = CoupledField::try_new(
        9,
        7,
        7,
        (lo, lo, lo),
        (
            Fix128::from_int(7),
            Fix128::from_int(5),
            Fix128::from_int(5),
        ),
    )
    .expect("valid grid");
    deposit_plastic_heat(&mesh, &sol, &heating, &mut field).expect("deposit succeeds");

    let (cx, cy, cz) = field.cell_size();
    let on_grid = field.sum().to_f64()
        * (cx * cy * cz).to_f64()
        * heating.volumetric_heat_capacity_mpa_per_k().to_f64();
    let in_solution: f64 = sol
        .dissipation
        .iter()
        .enumerate()
        .map(|(e, &w)| heating.taylor_quinney().to_f64() * w.to_f64() * tet_volume(&mesh, e))
        .sum();
    println!("heat on the grid     = {on_grid:.9} mJ");
    println!(
        "heat in the solution = {in_solution:.9} mJ   (relative gap {:.2e})",
        (on_grid - in_solution).abs() / in_solution
    );

    let free = float_free();
    let rise_field = TemperatureRise::from_absolute(&field, Fix128::ZERO);
    let hot = solve_with_eigenstrain(
        &mesh,
        &material,
        &free,
        &SolverConfig::default(),
        Some(ThermalExpansion::from_rise(&rise_field, fx(ALPHA_PER_K))),
    )
    .expect("thermoelastic solve succeeds");
    let cold = solve_with_eigenstrain(&mesh, &material, &free, &SolverConfig::default(), None)
        .expect("reference solve succeeds");

    let moved = hot
        .displacements
        .iter()
        .zip(cold.displacements.iter())
        .map(|(a, b)| {
            (a[0] - b[0])
                .abs()
                .max((a[1] - b[1]).abs())
                .max((a[2] - b[2]).abs())
        })
        .fold(Fix128::ZERO, Fix128::max);
    println!(
        "\nlargest thermal displacement from the plastic heat = {:.6e} mm",
        moved.to_f64()
    );
    println!("(α = {ALPHA_PER_K:e} /K, so the bar grows by α·ΔT·L once the heat has spread)");
}
