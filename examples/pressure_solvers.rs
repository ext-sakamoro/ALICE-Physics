//! Choosing the pressure projection of a CFD step
//!
//! Runs the same divergent seed through each of the four single-process
//! pressure solvers `CfdSolver::step_with_pressure_solver` can be asked for,
//! prints how much divergence each one leaves, and shows the two refusals a
//! caller is most likely to meet: multigrid on a grid that is not a power of
//! two, and a BiCGStab request with no tolerance.
//!
//! ```bash
//! cargo run --example pressure_solvers --features std
//! ```

use alice_physics::cfd_solver::{CfdSolver, PressureSolver};
use alice_physics::eulerian_grid::MacGrid;
use alice_physics::math::{Fix128, Vec3Fix};

/// `u(i, j, k) = i · dx`, which has unit divergence in every cell.
fn seed(grid: &mut MacGrid) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                grid.u[i + (nx + 1) * (j + ny * k)] = Fix128::from_int(i as i64) * grid.dx;
            }
        }
    }
}

fn max_abs_divergence(grid: &MacGrid) -> f64 {
    let mut worst = 0.0_f64;
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                worst = worst.max(grid.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    worst
}

fn solver(nx: usize) -> CfdSolver {
    let mut s = CfdSolver::new(nx, 8, 8, Fix128::from_ratio(1, 8));
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    seed(&mut s.grid);
    s
}

fn main() {
    let dt = Fix128::from_ratio(1, 100);
    let choices = [
        (
            "red-black Gauss-Seidel, 400 sweeps",
            PressureSolver::RedBlackGs { sweeps: 400 },
        ),
        (
            "multigrid, 30 W-cycles",
            PressureSolver::Multigrid { cycles: 30 },
        ),
        (
            "Jacobi, 800 iterations",
            PressureSolver::Jacobi { iterations: 800 },
        ),
        (
            "BiCGStab to 2^-24",
            PressureSolver::BiCgStab {
                max_iterations: 200,
                tolerance: Fix128::from_raw(0, 1 << 40),
            },
        ),
    ];
    for (label, choice) in choices {
        let mut s = solver(8);
        let before = max_abs_divergence(&s.grid);
        let report = s
            .step_with_pressure_solver(dt, choice)
            .expect("8x8x8 accepts every solver");
        let after = max_abs_divergence(&s.grid);
        match report.bicgstab {
            Some(stats) => println!(
                "{label:<36} max|div u| {before:.3} -> {after:.3e}  ({} iterations, converged = {})",
                stats.iterations, stats.converged
            ),
            None => println!("{label:<36} max|div u| {before:.3} -> {after:.3e}"),
        }
    }

    // The refusals: a chosen solver that cannot run is an error, not a
    // silent fallback to another one.
    let mut narrow = solver(7);
    match narrow.step_with_pressure_solver(dt, PressureSolver::Multigrid { cycles: 6 }) {
        Err(e) => println!(
            "7x8x8 with multigrid: refused ({e}); step_count stays {}",
            narrow.step_count
        ),
        Ok(_) => println!("7x8x8 with multigrid: unexpectedly accepted"),
    }
    let mut s = solver(8);
    let zero_tolerance = PressureSolver::BiCgStab {
        max_iterations: 50,
        tolerance: Fix128::ZERO,
    };
    match s.step_with_pressure_solver(dt, zero_tolerance) {
        Err(e) => println!("BiCGStab with tolerance 0: refused ({e})"),
        Ok(_) => println!("BiCGStab with tolerance 0: unexpectedly accepted"),
    }
}
