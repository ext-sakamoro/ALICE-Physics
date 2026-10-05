//! Which projection a CFD step runs, and editing a RANS state cell by cell
//!
//! Reaches `CfdSolver::step_multigrid` and `RansState::set`.
//!
//! `step_multigrid(dt, cycles)` is `step(dt)` with the pressure projection
//! done by `cycles` multigrid W-cycles. Its documented contract, checked here
//! bit for bit against the other entry points on the same seed:
//!
//! - on a power-of-two grid, `step(dt)` is `step_multigrid(dt, 6)` and
//!   `step_multigrid(dt, c)` is `step_with_pressure_solver(dt, Multigrid { c })`;
//! - `step_multigrid(dt, 0)`, and any cycle count on a grid that is not a
//!   power of two, falls back to `jacobi_iterations` Gauss-Seidel sweeps,
//!   i.e. `step_with_pressure_solver(dt, RedBlackGs { jacobi_iterations })`;
//! - `dt = 0` leaves the solver untouched.
//!
//! The seed `u(i) = i·dx` has unit divergence in every cell; 30 W-cycles
//! leave less than `1e-6` of it.
//!
//! `RansState::set(i, j, k, k_value, epsilon)` writes one cell; the k-ε eddy
//! viscosity of that cell is then `C_μ k² / ε` with `C_μ = 0.09`, every other
//! cell keeps its value, and a write outside the grid is ignored.
//!
//! Run with: `cargo run --example cfd_projection_and_rans_state`

use alice_physics::cfd_solver::{CfdSolver, PressureSolver, RansState, TurbulenceModel};
use alice_physics::math::Fix128;

type Fields = (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>, Vec<Fix128>);

fn seeded(nx: usize) -> CfdSolver {
    let mut s = CfdSolver::new(nx, 8, 8, Fix128::from_ratio(1, 8));
    let (ny, nz) = (s.grid.ny, s.grid.nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                s.grid.u[i + (nx + 1) * (j + ny * k)] = Fix128::from_int(i as i64) * s.grid.dx;
            }
        }
    }
    s
}

fn fields(s: &CfdSolver) -> Fields {
    (
        s.grid.u.clone(),
        s.grid.v.clone(),
        s.grid.w.clone(),
        s.grid.pressure.clone(),
    )
}

fn max_abs_divergence(s: &CfdSolver) -> f64 {
    let g = &s.grid;
    let mut worst = 0.0_f64;
    for k in 0..g.nz {
        for j in 0..g.ny {
            for i in 0..g.nx {
                worst = worst.max(g.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    worst
}

fn after(nx: usize, run: impl FnOnce(&mut CfdSolver)) -> Fields {
    let mut s = seeded(nx);
    run(&mut s);
    assert_eq!(s.step_count, 1);
    fields(&s)
}

fn main() {
    let dt = Fix128::from_ratio(1, 100);

    // Power-of-two grid (8³).
    let default = after(8, |s| s.step(dt));
    assert!(default == after(8, |s| s.step_multigrid(dt, 6)));
    let three = after(8, |s| s.step_multigrid(dt, 3));
    assert!(
        three
            == after(8, |s| {
                s.step_with_pressure_solver(dt, PressureSolver::Multigrid { cycles: 3 })
                    .expect("8³ is a power of two");
            })
    );
    assert!(three != default, "3 and 6 cycles differ");
    let gs = after(8, |s| {
        let sweeps = s.jacobi_iterations;
        s.step_with_pressure_solver(dt, PressureSolver::RedBlackGs { sweeps })
            .expect("Gauss-Seidel accepts any grid");
    });
    assert!(gs == after(8, |s| s.step_multigrid(dt, 0)));
    assert!(gs != default, "Gauss-Seidel and multigrid differ on 8³");
    println!("[cfd] 8³: step = step_multigrid(6), step_multigrid(0) = Gauss-Seidel");

    // Not a power of two (6 x 8 x 8): every entry is Gauss-Seidel.
    let gs6 = after(6, |s| {
        let sweeps = s.jacobi_iterations;
        s.step_with_pressure_solver(dt, PressureSolver::RedBlackGs { sweeps })
            .expect("Gauss-Seidel accepts any grid");
    });
    assert!(gs6 == after(6, |s| s.step_multigrid(dt, 6)));
    assert!(gs6 == after(6, |s| s.step(dt)));
    println!("[cfd] 6x8x8: step_multigrid(6) falls back to Gauss-Seidel");

    // dt = 0 is a no-op.
    let mut s = seeded(8);
    let before = fields(&s);
    s.step_multigrid(Fix128::ZERO, 6);
    assert!(fields(&s) == before && s.step_count == 0);

    // 30 W-cycles remove the unit divergence of the seed.
    let mut s = seeded(8);
    assert!((max_abs_divergence(&s) - 1.0).abs() < 1e-12);
    s.step_multigrid(dt, 30);
    let left = max_abs_divergence(&s);
    assert!(left < 1e-6, "max |div| after 30 cycles: {left:e}");
    println!("[cfd] 30 W-cycles: max |div| 1 -> {left:.3e}");

    // RansState::set: one cell of a uniform k-ε state.
    let k0 = Fix128::from_ratio(1, 2);
    let e0 = Fix128::from_ratio(1, 4);
    let mut rans = RansState::uniform(4, 4, 1, TurbulenceModel::KEpsilon, k0, e0);
    let (k1, e1) = (Fix128::from_int(2), Fix128::from_ratio(9, 25));
    rans.set(1, 2, 0, k1, e1);
    assert_eq!((rans.k(1, 2, 0), rans.epsilon(1, 2, 0)), (k1, e1));
    // 0.09 · 2² / 0.36 = 1.
    let nu = rans.eddy_viscosity(1, 2, 0).to_f64();
    assert!((nu - 1.0).abs() < 1e-12, "nu_t = {nu}");
    // 0.09 · 0.25 / 0.25 = 0.09 elsewhere, bit-equal to an untouched state.
    let untouched = RansState::uniform(4, 4, 1, TurbulenceModel::KEpsilon, k0, e0);
    for (i, j) in [(0, 0), (2, 2), (1, 3), (3, 1)] {
        assert_eq!(
            rans.eddy_viscosity(i, j, 0),
            untouched.eddy_viscosity(i, j, 0)
        );
    }
    assert!((untouched.eddy_viscosity(0, 0, 0).to_f64() - 0.09).abs() < 1e-12);
    // Outside the grid: ignored, and reads there are zero.
    let snapshot = rans.clone();
    rans.set(4, 0, 0, k1, e1);
    rans.set(0, 0, 1, k1, e1);
    assert_eq!(rans, snapshot);
    assert_eq!(rans.k(4, 0, 0), Fix128::ZERO);
    println!("[cfd] RansState::set: nu_t of the edited cell = {nu:.12}");
}
