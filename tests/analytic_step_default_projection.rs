//! Oracles for the projection `CfdSolver::step` uses by default.
//!
//! `step` projects with multigrid (`DEFAULT_CYCLES` W-cycles) when every grid
//! extent is a power of two, and with the Gauss-Seidel sweeps
//! (`jacobi_iterations`) otherwise. `step_multigrid(dt, 0)` is the explicit way
//! to keep the Gauss-Seidel projection on a grid multigrid could solve.
//!
//! Each check is bit for bit against a composition built from pieces that do
//! not themselves depend on the default: the step with a zero-sweep
//! projection (which leaves the field exactly as the projection starts), then
//! the named projection function.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{project_pressure, project_pressure_multigrid, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};

/// Cycles of the default projection. Measured (`probe`: a 4-cycle-stride sweep
/// on 8^3 / 16^3 / 32^3) as the smallest count whose post-step `max|div|` is at
/// or below that of the 30 Gauss-Seidel sweeps it replaces on every size: 6
/// cycles give 6.3e-4 / 1.9e-3 / 2.6e-2 against 7.9e-4 / 1.4e-2 / 6.9e-1.
const DEFAULT_CYCLES: u32 = 6;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

fn scene(nx: usize, ny: usize, nz: usize) -> CfdSolver {
    let mut s = CfdSolver::new(nx, ny, nz, Fix128::ONE);
    s.gravity = Vec3Fix::default();
    let pat =
        |i: usize, off: usize| Fix128::from_int(((3 * i + 5 * (i / 3) + off) % 11) as i64 - 5);
    for (i, x) in s.grid.u.iter_mut().enumerate() {
        *x = pat(i, 1);
    }
    for (i, x) in s.grid.v.iter_mut().enumerate() {
        *x = pat(i, 4);
    }
    for (i, x) in s.grid.w.iter_mut().enumerate() {
        *x = pat(i, 7);
    }
    s
}

fn same_grid(a: &MacGrid, b: &MacGrid, what: &str) {
    for (name, x, y) in [
        ("u", &a.u, &b.u),
        ("v", &a.v, &b.v),
        ("w", &a.w, &b.w),
        ("pressure", &a.pressure, &b.pressure),
    ] {
        assert_eq!(x.len(), y.len(), "{what}: {name} length");
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(p, q, "{what}: {name}[{i}]");
        }
    }
}

/// The step up to the projection, then `project`.
fn composed(
    nx: usize,
    ny: usize,
    nz: usize,
    project: impl Fn(&mut MacGrid, Fix128, Fix128),
) -> CfdSolver {
    let mut s = scene(nx, ny, nz);
    // `cycles == 0` selects the Gauss-Seidel projection on every grid, and
    // zero sweeps make it a no-op, whatever the default is
    s.jacobi_iterations = 0;
    s.step_multigrid(dt(), 0);
    project(&mut s.grid, dt(), s.density_kg_m3);
    s
}

#[test]
fn step_on_a_power_of_two_grid_projects_with_the_default_multigrid_cycles() {
    for (nx, ny, nz) in [(4, 4, 4), (8, 8, 8), (8, 4, 2), (16, 16, 16)] {
        let mut by_step = scene(nx, ny, nz);
        by_step.step(dt());
        let reference = composed(nx, ny, nz, |g, t, rho| {
            project_pressure_multigrid(g, t, rho, DEFAULT_CYCLES);
        });
        same_grid(
            &by_step.grid,
            &reference.grid,
            &format!("{nx}x{ny}x{nz} default"),
        );
        assert_eq!(by_step.step_count, 1);
    }
}

#[test]
fn step_on_a_grid_with_a_non_power_of_two_extent_keeps_the_gauss_seidel_projection() {
    // the offending extent goes on each axis in turn, so a guard that checked
    // only some axes would still be caught
    for (nx, ny, nz) in [(6, 6, 6), (6, 8, 8), (8, 6, 8), (8, 8, 6), (5, 5, 5)] {
        let mut by_step = scene(nx, ny, nz);
        by_step.step(dt());
        let reference = composed(nx, ny, nz, |g, t, rho| project_pressure(g, t, rho, 30));
        same_grid(
            &by_step.grid,
            &reference.grid,
            &format!("{nx}x{ny}x{nz} fallback"),
        );
    }
}

#[test]
fn zero_cycles_is_the_explicit_gauss_seidel_projection_on_a_power_of_two_grid() {
    let mut by_step = scene(8, 8, 8);
    by_step.step_multigrid(dt(), 0);
    let reference = composed(8, 8, 8, |g, t, rho| project_pressure(g, t, rho, 30));
    same_grid(&by_step.grid, &reference.grid, "8^3 cycles 0");
    // and it is not the default
    let mut default = scene(8, 8, 8);
    default.step(dt());
    assert_ne!(by_step.grid.u, default.grid.u, "default must not be GS");
}

#[test]
fn jacobi_iterations_still_counts_gauss_seidel_sweeps_and_the_default_ignores_it() {
    // power-of-two grid: the default projection does not read the sweep count
    let mut a = scene(8, 8, 8);
    let mut b = scene(8, 8, 8);
    a.jacobi_iterations = 1;
    b.jacobi_iterations = 300;
    a.step(dt());
    b.step(dt());
    same_grid(&a.grid, &b.grid, "default ignores jacobi_iterations");
    // the explicit Gauss-Seidel route reads it as the number of sweeps
    let mut c = scene(8, 8, 8);
    c.jacobi_iterations = 7;
    c.step_multigrid(dt(), 0);
    let reference = composed(8, 8, 8, |g, t, rho| project_pressure(g, t, rho, 7));
    same_grid(&c.grid, &reference.grid, "7 sweeps");
    // fallback grid: it is the sweep count
    let mut d = scene(6, 6, 6);
    d.jacobi_iterations = 7;
    d.step(dt());
    let reference = composed(6, 6, 6, |g, t, rho| project_pressure(g, t, rho, 7));
    same_grid(&d.grid, &reference.grid, "6^3 7 sweeps");
}
