//! Oracles for `CfdSolver::step_with_pressure_solver` — the entry that lets a
//! caller pick the pressure projection (red-black Gauss-Seidel, multigrid,
//! Jacobi, BiCGStab) and is refused on the inputs the fixed-count solvers
//! would otherwise answer silently.
//!
//! # What is pinned, and why each is an oracle and not a restatement
//!
//! 1. **Every solver removes divergence.** On a field seeded with
//!    `∇·u = 1` everywhere, each solver's projected field has
//!    `max|∇·u|` below a bound that is *its own* claim: for BiCGStab the
//!    bound is `tolerance / scale` (the residual it stopped at, converted to
//!    divergence units — the identity `r = scale · ∇·u_after` is derived in
//!    the module below); for the fixed-count solvers the bound is a fraction of
//!    the seeded divergence.
//! 2. **All solvers agree.** Two projected fields differ by at most
//!    `(n²/4) · dx · (d_a + d_b)` where `d` is each one's remaining
//!    `max|∇·u|` — derived from the maximum principle for the Dirichlet
//!    Laplacian (`‖L⁻¹‖_∞ ≤ n²/8`), not measured. ⚠️ Without this bound a
//!    "they agree to 1e-6" assertion would be a number pulled from a run.
//! 3. **A solenoidal field is a fixed point of every solver, to the bit.**
//!    A uniform flow through an open box has zero divergence, so the Poisson
//!    right-hand side is exactly zero, every solver leaves `p = 0` exactly and
//!    the velocity correction is exactly zero.
//! 4. **The named entry reproduces [`CfdSolver::step`] to the bit** when it
//!    is handed the solver `step` would have chosen (multigrid on a
//!    power-of-two grid, Gauss-Seidel otherwise). This is the wiring oracle:
//!    the new entry runs the same body, not a copy.
//! 5. **BiCGStab's verdict is honest.** A budget of one iteration reports
//!    `converged == false` with a residual at or above the tolerance and a
//!    field whose divergence is above the converged one's; a generous budget
//!    reports `converged` and the result does not depend on how generous
//!    (stopping on the tolerance, not on the budget — bit identical).
//! 6. **Refusals.** Zero `dt`, density, count, a non-positive tolerance and
//!    multigrid off a power-of-two grid each return the named error **and
//!    leave the solver untouched** (grid and `step_count` bit identical).
//!
//! # The residual identity used by oracles 1 and 2
//!
//! The solvers write `L p = b` with `b = scale · ∇·u*`, `scale = ρ dx²/dt`,
//! `L` the masked 7-point Laplacian at unit spacing, and then correct
//! `u ← u* − (dt/(ρ dx)) ∇p`. Taking the discrete divergence of the
//! correction gives `∇·u = ∇·u* − (dt/(ρ dx²)) L p = (b − L p)/scale`, so
//! the post-projection divergence **is** the residual over `scale`. For two
//! solutions `L (p_a − p_b) = r_b − r_a`, and on the open box (`p = 0`
//! outside, so `L` is the Dirichlet Laplacian) the discrete maximum principle
//! bounds `‖p_a − p_b‖_∞ ≤ (n²/8) ‖r_a − r_b‖_∞`. A face velocity differs by
//! `(dt/(ρ dx))` times a pressure difference across the face, at most
//! `2 ‖Δp‖_∞`, which gives `|Δu| ≤ (n²/4) dx (d_a + d_b)`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// Bounds are closed-form f64 evaluations of the grid parameters, not state.
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::{CfdSolver, PressureSolver, PressureSolverError};
use alice_physics::eulerian_grid::MacGrid;
use alice_physics::math::{Fix128, Vec3Fix};

const N: usize = 8;

/// Spacing `1/8` m, so the box is the unit cube.
fn dx() -> Fix128 {
    Fix128::from_ratio(1, 8)
}

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 100)
}

/// A solver with no body force and no viscosity, so the only thing the step
/// does to the velocity besides the projection is the (identity on these
/// fields) advection.
fn quiet_solver(nx: usize, ny: usize, nz: usize) -> CfdSolver {
    let mut s = CfdSolver::new(nx, ny, nz, dx());
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s
}

/// `u(i, j, k) = i · dx`: a linear profile with `∇·u = 1` in every cell.
fn seed_divergent(grid: &mut MacGrid) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let ix = i + (nx + 1) * (j + ny * k);
                grid.u[ix] = Fix128::from_int(i as i64) * grid.dx;
            }
        }
    }
}

/// `u = 1` everywhere: solenoidal.
fn seed_uniform(grid: &mut MacGrid) {
    for value in grid.u.iter_mut() {
        *value = Fix128::ONE;
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

fn max_face_difference(a: &MacGrid, b: &MacGrid) -> f64 {
    a.u.iter()
        .zip(&b.u)
        .chain(a.v.iter().zip(&b.v))
        .chain(a.w.iter().zip(&b.w))
        .map(|(x, y)| (x.to_f64() - y.to_f64()).abs())
        .fold(0.0, f64::max)
}

fn grids_bit_equal(a: &MacGrid, b: &MacGrid) -> bool {
    a.u == b.u && a.v == b.v && a.w == b.w && a.pressure == b.pressure
}

/// The four solvers at budgets that leave each one well converged on `8³`.
fn solvers() -> [(&'static str, PressureSolver); 4] {
    [
        ("red-black GS", PressureSolver::RedBlackGs { sweeps: 400 }),
        ("multigrid", PressureSolver::Multigrid { cycles: 30 }),
        ("Jacobi", PressureSolver::Jacobi { iterations: 800 }),
        (
            "BiCGStab",
            PressureSolver::BiCgStab {
                max_iterations: 200,
                tolerance: Fix128::from_raw(0, 1 << 40), // 2⁻²⁴ in rhs units, about 4e-11 in divergence
            },
        ),
    ]
}

/// `scale = ρ dx² / dt` of the Poisson right-hand side.
fn scale(s: &CfdSolver) -> f64 {
    let d = s.grid.dx.to_f64();
    s.density_kg_m3.to_f64() * d * d / dt().to_f64()
}

// ---------------------------------------------------------------------------
// oracle 1 — every solver removes the seeded divergence
// ---------------------------------------------------------------------------

#[test]
fn every_solver_removes_the_seeded_divergence() {
    for (name, solver) in solvers() {
        let mut s = quiet_solver(N, N, N);
        seed_divergent(&mut s.grid);
        let before = max_abs_divergence(&s.grid);
        assert!(
            (before - 1.0).abs() < 1e-12,
            "the seed has unit divergence, got {before}"
        );
        let report = s
            .step_with_pressure_solver(dt(), solver)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let after = max_abs_divergence(&s.grid);
        let bound = match report.bicgstab {
            Some(stats) => {
                assert!(stats.converged, "{name}: did not converge in budget");
                // r = scale · ∇·u_after, and the solver stopped at r < tolerance.
                stats.final_residual.to_f64().max(2f64.powi(-24)) / scale(&s) * 1.5
            }
            // Fixed-count solvers make no claim; at these budgets each reduces
            // the seed by at least six decades (measured), and oracle 2 is the
            // derived statement that they land on the same field.
            None => 1e-6 * before,
        };
        assert!(
            after <= bound,
            "{name}: max|div u| after projection {after:.3e} exceeds its bound {bound:.3e}"
        );
        eprintln!("  {name}: max|div u| {before:.3} -> {after:.3e} (bound {bound:.3e})");
    }
}

// ---------------------------------------------------------------------------
// oracle 2 — all solvers agree, within a bound derived from their residuals
// ---------------------------------------------------------------------------

#[test]
fn all_solvers_agree_on_the_projected_field_within_the_residual_bound() {
    let mut results = Vec::new();
    for (name, solver) in solvers() {
        let mut s = quiet_solver(N, N, N);
        seed_divergent(&mut s.grid);
        s.step_with_pressure_solver(dt(), solver)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let d = max_abs_divergence(&s.grid);
        results.push((name, s, d));
    }
    let d_x = dx().to_f64();
    for a in 0..results.len() {
        for b in (a + 1)..results.len() {
            let (na, sa, da) = &results[a];
            let (nb, sb, db) = &results[b];
            let got = max_face_difference(&sa.grid, &sb.grid);
            // ‖Δu‖ ≤ (n²/4) · dx · (d_a + d_b), from the maximum principle.
            let bound = (N * N) as f64 / 4.0 * d_x * (da + db);
            assert!(
                got <= bound,
                "{na} vs {nb}: fields differ by {got:.3e}, more than the residual bound {bound:.3e}"
            );
            eprintln!("  {na} vs {nb}: |Δu| {got:.3e} <= {bound:.3e}");
        }
    }
    // The bound must have been a real constraint, not a vacuous one: the
    // least converged pair's bound stays well below the seed's own scale.
    let worst_bound = results
        .iter()
        .map(|(_, _, d)| d)
        .fold(0.0_f64, |m, d| m.max(*d))
        * 2.0
        * (N * N) as f64
        / 4.0
        * d_x;
    assert!(
        worst_bound < 1e-2,
        "the residual bound {worst_bound:.3e} is loose enough to accept a wrong solver"
    );
}

// ---------------------------------------------------------------------------
// oracle 3 — a solenoidal field is a fixed point, to the bit
// ---------------------------------------------------------------------------

#[test]
fn a_uniform_flow_is_a_fixed_point_of_every_solver_to_the_bit() {
    for (name, solver) in solvers() {
        let mut s = quiet_solver(N, N, N);
        seed_uniform(&mut s.grid);
        let before = s.grid.clone();
        let report = s
            .step_with_pressure_solver(dt(), solver)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            grids_bit_equal(&before, &s.grid),
            "{name}: a divergence-free field was changed by the projection"
        );
        assert!(
            s.grid.pressure.iter().all(|p| p.is_zero()),
            "{name}: pressure is not exactly zero on a solenoidal field"
        );
        if let Some(stats) = report.bicgstab {
            assert_eq!(stats.iterations, 0, "{name}: nothing to iterate on");
            assert!(stats.converged);
        }
    }
}

// ---------------------------------------------------------------------------
// oracle 4 — the named entry is the same body as `step`, to the bit
// ---------------------------------------------------------------------------

#[test]
fn the_named_entry_reproduces_step_to_the_bit_with_the_default_choice() {
    // Power-of-two grid: `step` projects with 6 multigrid cycles.
    let mut by_step = quiet_solver(N, N, N);
    seed_divergent(&mut by_step.grid);
    by_step.step(dt());
    let mut by_name = quiet_solver(N, N, N);
    seed_divergent(&mut by_name.grid);
    by_name
        .step_with_pressure_solver(dt(), PressureSolver::Multigrid { cycles: 6 })
        .expect("8³ supports multigrid");
    assert!(
        grids_bit_equal(&by_step.grid, &by_name.grid),
        "Multigrid {{ cycles: 6 }} must be what `step` does on 8³"
    );
    assert_eq!(by_step.step_count, by_name.step_count);

    // Non-power-of-two grid: `step` falls back to `jacobi_iterations` sweeps.
    let mut by_step = quiet_solver(7, N, N);
    seed_divergent(&mut by_step.grid);
    by_step.step(dt());
    let mut by_name = quiet_solver(7, N, N);
    seed_divergent(&mut by_name.grid);
    let sweeps = by_name.jacobi_iterations;
    by_name
        .step_with_pressure_solver(dt(), PressureSolver::RedBlackGs { sweeps })
        .expect("Gauss-Seidel runs on any grid");
    assert!(
        grids_bit_equal(&by_step.grid, &by_name.grid),
        "RedBlackGs {{ sweeps: jacobi_iterations }} must be what `step` does on 7×8×8"
    );
    // And the two are not the same thing: Jacobi on the same scene lands
    // elsewhere (it is a different iteration), which is the teeth of the
    // dispatch — a wiring that sent `Jacobi` to the Gauss-Seidel solver would
    // make this pass trivially.
    let mut by_jacobi = quiet_solver(7, N, N);
    seed_divergent(&mut by_jacobi.grid);
    by_jacobi
        .step_with_pressure_solver(dt(), PressureSolver::Jacobi { iterations: sweeps })
        .expect("Jacobi runs on any grid");
    assert!(
        !grids_bit_equal(&by_step.grid, &by_jacobi.grid),
        "Jacobi at the same count reproduced Gauss-Seidel to the bit: the dispatch is not \
         reaching a different solver"
    );
}

// ---------------------------------------------------------------------------
// oracle 5 — BiCGStab's verdict is honest and its stop is the tolerance
// ---------------------------------------------------------------------------

#[test]
fn bicgstab_reports_an_exhausted_budget_and_stops_on_the_tolerance() {
    let tolerance = Fix128::from_raw(0, 1 << 40); // 2⁻²⁴
    let mut starved = quiet_solver(N, N, N);
    seed_divergent(&mut starved.grid);
    let report = starved
        .step_with_pressure_solver(
            dt(),
            PressureSolver::BiCgStab {
                max_iterations: 1,
                tolerance,
            },
        )
        .expect("valid request");
    let stats = report.bicgstab.expect("BiCGStab reports");
    assert!(
        !stats.converged,
        "one iteration cannot converge the seeded field"
    );
    assert_eq!(stats.iterations, 1);
    assert!(stats.final_residual >= tolerance);

    let run = |budget: u32| {
        let mut s = quiet_solver(N, N, N);
        seed_divergent(&mut s.grid);
        let report = s
            .step_with_pressure_solver(
                dt(),
                PressureSolver::BiCgStab {
                    max_iterations: budget,
                    tolerance,
                },
            )
            .expect("valid request");
        (s, report.bicgstab.expect("BiCGStab reports"))
    };
    let (fed, stats_fed) = run(200);
    assert!(stats_fed.converged);
    assert!(stats_fed.final_residual < tolerance);
    assert!(
        max_abs_divergence(&fed.grid) < max_abs_divergence(&starved.grid),
        "the converged run must leave less divergence than the starved one"
    );
    // Stopping on the tolerance: a bigger budget changes nothing.
    let (fed_more, stats_more) = run(2000);
    assert_eq!(stats_fed, stats_more);
    assert!(grids_bit_equal(&fed.grid, &fed_more.grid));
    eprintln!(
        "  BiCGStab: {} iterations to r < 2^-24 (starved: residual {:.3e})",
        stats_fed.iterations,
        stats.final_residual.to_f64()
    );
}

// ---------------------------------------------------------------------------
// refusals — each named, and each leaves the solver untouched
// ---------------------------------------------------------------------------

fn refuse(
    nx: usize,
    dt_s: Fix128,
    density: Fix128,
    solver: PressureSolver,
    want: PressureSolverError,
) {
    let mut s = quiet_solver(nx, N, N);
    s.density_kg_m3 = density;
    seed_divergent(&mut s.grid);
    let before = s.grid.clone();
    let got = s.step_with_pressure_solver(dt_s, solver);
    assert_eq!(
        got,
        Err(want),
        "{solver:?} on {nx}×{N}×{N}, dt {dt_s:?}, ρ {density:?}"
    );
    assert!(
        grids_bit_equal(&before, &s.grid),
        "a refused step must not touch the grid ({want:?})"
    );
    assert_eq!(s.step_count, 0, "a refused step must not count ({want:?})");
}

#[test]
fn degenerate_inputs_are_refused_and_leave_the_solver_untouched() {
    let rho = Fix128::from_int(1000);
    let gs = PressureSolver::RedBlackGs { sweeps: 10 };
    refuse(N, Fix128::ZERO, rho, gs, PressureSolverError::ZeroTimeStep);
    refuse(N, dt(), Fix128::ZERO, gs, PressureSolverError::ZeroDensity);
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::RedBlackGs { sweeps: 0 },
        PressureSolverError::ZeroIterations,
    );
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::Multigrid { cycles: 0 },
        PressureSolverError::ZeroIterations,
    );
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::Jacobi { iterations: 0 },
        PressureSolverError::ZeroIterations,
    );
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::BiCgStab {
            max_iterations: 0,
            tolerance: Fix128::ONE,
        },
        PressureSolverError::ZeroIterations,
    );
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::BiCgStab {
            max_iterations: 10,
            tolerance: Fix128::ZERO,
        },
        PressureSolverError::NonPositiveTolerance,
    );
    refuse(
        N,
        dt(),
        rho,
        PressureSolver::BiCgStab {
            max_iterations: 10,
            tolerance: -Fix128::ONE,
        },
        PressureSolverError::NonPositiveTolerance,
    );
    refuse(
        7,
        dt(),
        rho,
        PressureSolver::Multigrid { cycles: 6 },
        PressureSolverError::MultigridNeedsPowerOfTwoExtents { extents: (7, N, N) },
    );
    // The zero-spacing refusal needs a grid built with dx = 0.
    let mut s = CfdSolver::new(N, N, N, Fix128::ZERO);
    s.gravity = Vec3Fix::ZERO;
    assert_eq!(
        s.step_with_pressure_solver(dt(), gs),
        Err(PressureSolverError::ZeroSpacing)
    );
    assert_eq!(s.step_count, 0);
    // Refusals are ordered: dt first, then density, then spacing, then the
    // solver's own parameter — so the message names the first thing to fix.
    let mut s = CfdSolver::new(N, N, N, Fix128::ZERO);
    s.density_kg_m3 = Fix128::ZERO;
    assert_eq!(
        s.step_with_pressure_solver(Fix128::ZERO, PressureSolver::Jacobi { iterations: 0 }),
        Err(PressureSolverError::ZeroTimeStep)
    );
}

/// The error type displays every variant (a `Display` that panics or prints
/// nothing would make a logged refusal unreadable).
#[test]
fn every_refusal_displays_itself() {
    use alice_physics::cfd_solver::PressureSolverError as E;
    for e in [
        E::ZeroTimeStep,
        E::ZeroDensity,
        E::ZeroSpacing,
        E::ZeroIterations,
        E::MultigridNeedsPowerOfTwoExtents { extents: (7, 8, 8) },
        E::NonPositiveTolerance,
    ] {
        let text = e.to_string();
        assert!(!text.is_empty(), "{e:?} displays nothing");
    }
    assert!(E::MultigridNeedsPowerOfTwoExtents { extents: (7, 8, 8) }
        .to_string()
        .contains("7x8x8"));
}
