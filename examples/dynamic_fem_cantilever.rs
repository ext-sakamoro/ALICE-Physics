//! Transient FEM: a clamped bar released under a suddenly applied end load.
//!
//! The bar is clamped at `x = 0` and loaded along its axis at `x = L` at
//! `t = 0`. An undamped elastic body answers a step load by oscillating about
//! its static deflection and overshooting it, so the tip reaches roughly twice
//! the static value — the dynamic amplification factor — and the wave turns
//! around at `t = 2L/c` with `c = √(E/ρ)`. A static solve returns one number
//! and cannot show either.
//!
//! ```bash
//! cargo run --example dynamic_fem_cantilever --features std
//! ```

use alice_physics::dynamic_fem::{DynamicsConfig, MassLumping, TransientSolver};
use alice_physics::linear_elastic_fem::{Axis, BoundaryConditions, ElasticMaterial, SolverConfig};
use alice_physics::math::Fix128;
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};

/// Node index within an `(nx+1) × 2 × 2` lattice.
fn node(nx: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * 2).expect("lattice fits u32")
}

/// `[0, nx·h] × [0, h] × [0, h]` split into Kuhn six-tetrahedron cells.
fn bar(nx: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..2 {
        for j in 0..2 {
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
    for i in 0..nx {
        for path in PATHS {
            let mut step = [0usize; 3];
            let mut corners = [0u32; 4];
            corners[0] = node(nx, i, 0, 0);
            for (n, axis) in path.into_iter().enumerate() {
                step[axis] = 1;
                corners[n + 1] = node(nx, i + step[0], step[1], step[2]);
            }
            mesh.tets.push(Tetrahedron { vertices: corners });
        }
    }
    mesh
}

fn main() -> Result<(), alice_physics::linear_elastic_fem::FemError> {
    let (nx, h) = (8usize, 1.0_f32);
    let mesh = bar(nx, h);

    // mm / N / MPa / s, so density is in tonne/mm³. ν = 0 keeps the rod wave
    // speed exactly √(E/ρ); 2⁻³⁰ ≈ 9.3e-10 is close to PLA and is a power of
    // two, which keeps every Newmark coefficient exact.
    let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::ZERO)?;
    let dynamics = DynamicsConfig::try_new(
        Fix128::from_raw(0, 1 << 34), // ρ = 2⁻³⁰ tonne/mm³
        Fix128::from_raw(0, 1 << 41), // dt = 2⁻²³ s
        MassLumping::Consistent,
    )?;
    let solver_config = SolverConfig::try_new(20_000, Fix128::from_f64(1e-12))?;

    let mut boundary = BoundaryConditions::new();
    for j in 0..2 {
        for k in 0..2 {
            boundary.fix(node(nx, 0, j, k));
            boundary.add_load(node(nx, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }

    let mut solver = TransientSolver::new(&mesh, &material, &boundary, &dynamics, &solver_config)?;

    let tip = node(nx, nx, 0, 0) as usize * 3;
    let mut peak = 0.0_f64;
    let mut peak_step = 0usize;
    let mut iterations = 0u64;
    for n in 1..=200usize {
        iterations += u64::from(solver.step()?);
        let u = solver.displacements()[tip].to_f64();
        if u > peak {
            peak = u;
            peak_step = n;
        }
    }

    let dt = 1.0 / f64::from(1u32 << 23);
    let length = f64::from(nx as u32) * f64::from(h);
    let wave_speed = (3500.0 / (1.0 / 1_073_741_824.0_f64)).sqrt();
    println!(
        "bar: {} nodes, {} tetrahedra",
        mesh.vertices.len(),
        mesh.tets.len()
    );
    println!(
        "tip peak        {peak:.6e} mm at step {peak_step} (t = {:.4e} s)",
        peak_step as f64 * dt
    );
    println!("closed form 2L/c = {:.4e} s", 2.0 * length / wave_speed);
    println!(
        "tip velocity    {:.6e} mm/s",
        solver.velocities()[tip].to_f64()
    );
    println!(
        "tip accel       {:.6e} mm/s²",
        solver.accelerations()[tip].to_f64()
    );
    println!("conjugate gradient iterations over 200 steps: {iterations}");
    Ok(())
}
