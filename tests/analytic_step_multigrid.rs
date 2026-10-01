//! Oracles for `CfdSolver::step_multigrid`: the time step whose pressure
//! projection is the multigrid solver.
//!
//! `project_pressure_multigrid` is only worth having if a solver step can reach
//! it, so these pin the **wiring**, not the solver (`analytic_multigrid.rs`
//! pins that):
//!
//! 1. composition: `step_multigrid` is exactly "the step up to the projection,
//!    then `project_pressure_multigrid`" — compared bit for bit against that
//!    composition, built from `step` with a zero-iteration projection. A step
//!    that quietly kept the Gauss-Seidel projection would fail this and still
//!    pass the physics checks below, because both solvers converge to the same
//!    field
//! 2. physics: with enough cycles the result matches the converged `step`, and
//!    is divergence-free
//! 3. fallback: an extent that is not a power of two, or `cycles == 0`, cannot
//!    be solved by the multigrid projection; the step then uses the Gauss-Seidel
//!    projection (bit-identical to `step`) instead of silently skipping the
//!    projection and leaving a compressible field
//! 4. no new panic path: for every degenerate input, `step_multigrid` panics
//!    exactly when `step` does
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::{project_pressure_multigrid, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Largest relative difference between a 24-cycle `step_multigrid` and a
/// converged (600-sweep) `step`, over `|u|max`. Measured 2.7e-14 at 24 cycles
/// (`tolerance_measurement`: 2.2e-6 at 8, 2.1e-10 at 16, 2.7e-14 at 24), so this
/// leaves a factor of about 3700 and still rejects a solver that stops early.
const PHYSICS_TOL_REL: f64 = 1e-10;
/// Largest `max|div|` after 24 cycles relative to before the step. Measured
/// 2.4e-14 (same measurement), a margin of about 4000.
const DIVERGENCE_TOL: f64 = 1e-10;

fn dt() -> Fix128 {
    Fix128::from_ratio(1, 64)
}

/// Integer velocities that are neither symmetric nor divergence-free.
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

fn same_grid(a: &MacGrid, b: &MacGrid) {
    for (name, x, y) in [
        ("u", &a.u, &b.u),
        ("v", &a.v, &b.v),
        ("w", &a.w, &b.w),
        ("pressure", &a.pressure, &b.pressure),
    ] {
        assert_eq!(x.len(), y.len(), "{name} length");
        for (i, (p, q)) in x.iter().zip(y).enumerate() {
            assert_eq!(p, q, "{name}[{i}]");
        }
    }
}

fn max_abs(v: &[Fix128]) -> f64 {
    v.iter().map(|x| x.to_f64().abs()).fold(0.0, f64::max)
}

fn max_div(g: &MacGrid) -> f64 {
    let mut m = 0.0f64;
    for k in 0..g.nz {
        for j in 0..g.ny {
            for i in 0..g.nx {
                m = m.max(g.divergence(i, j, k).to_f64().abs());
            }
        }
    }
    m
}

#[test]
fn step_multigrid_is_the_step_up_to_the_projection_then_the_multigrid_projection() {
    // oracle: composition. `jacobi_iterations = 0` turns the step's own
    // projection into a no-op (zero pressure, nothing to subtract), leaving the
    // field exactly as it is when the projection starts.
    for n in [4usize, 8] {
        for cycles in [1u32, 3] {
            let mut by_step = scene(n, n, n);
            let mut reference = scene(n, n, n);
            by_step.step_multigrid(dt(), cycles);
            reference.jacobi_iterations = 0;
            reference.step(dt());
            project_pressure_multigrid(&mut reference.grid, dt(), reference.density_kg_m3, cycles);
            same_grid(&by_step.grid, &reference.grid);
            assert_eq!(by_step.step_count, reference.step_count);
        }
    }
}

#[test]
fn a_converged_step_multigrid_matches_a_converged_step() {
    let n = 8;
    let mut mg = scene(n, n, n);
    let mut gs = scene(n, n, n);
    gs.jacobi_iterations = 600;
    mg.step_multigrid(dt(), 24);
    gs.step(dt());
    let scale = max_abs(&gs.grid.u)
        .max(max_abs(&gs.grid.v))
        .max(max_abs(&gs.grid.w));
    for (a, b) in [
        (&mg.grid.u, &gs.grid.u),
        (&mg.grid.v, &gs.grid.v),
        (&mg.grid.w, &gs.grid.w),
    ] {
        for (x, y) in a.iter().zip(b) {
            assert!(
                (x.to_f64() - y.to_f64()).abs() <= PHYSICS_TOL_REL * scale,
                "{x:?} vs {y:?}"
            );
        }
    }
}

#[test]
fn step_multigrid_leaves_a_divergence_free_field() {
    let n = 8;
    let mut s = scene(n, n, n);
    let before = max_div(&s.grid);
    s.step_multigrid(dt(), 24);
    assert!(
        max_div(&s.grid) <= DIVERGENCE_TOL * before.max(1.0),
        "divergence {} (before {before})",
        max_div(&s.grid)
    );
}

#[test]
fn a_zero_dt_is_a_no_op_and_a_real_step_counts() {
    let mut s = scene(8, 8, 8);
    let before = s.grid.clone();
    s.step_multigrid(Fix128::ZERO, 4);
    same_grid(&s.grid, &before);
    assert_eq!(s.step_count, 0);
    s.step_multigrid(dt(), 4);
    assert_eq!(s.step_count, 1);
}

#[test]
fn unsupported_extents_and_zero_cycles_use_the_gauss_seidel_projection() {
    // 6 is not a power of two; 8 with cycles == 0 cannot run a cycle. The
    // non-cubic cases put the offending extent on each axis in turn, so a guard
    // that checked only some of the axes would still be caught.
    let cases: [(usize, usize, usize, u32); 6] = [
        (6, 6, 6, 4),
        (6, 8, 8, 4),
        (8, 6, 8, 4),
        (8, 8, 6, 4),
        (8, 8, 8, 0),
        (4, 4, 4, 0),
    ];
    for (nx, ny, nz, cycles) in cases {
        let mut mg = scene(nx, ny, nz);
        let mut gs = scene(nx, ny, nz);
        mg.step_multigrid(dt(), cycles);
        gs.step(dt());
        same_grid(&mg.grid, &gs.grid);
        assert_eq!(
            mg.step_count, gs.step_count,
            "{nx}x{ny}x{nz} cycles {cycles}"
        );
    }
}

#[test]
fn the_result_is_bit_deterministic() {
    let mut a = scene(8, 8, 8);
    let mut b = scene(8, 8, 8);
    for _ in 0..3 {
        a.step_multigrid(dt(), 4);
        b.step_multigrid(dt(), 4);
    }
    same_grid(&a.grid, &b.grid);
}

#[test]
fn degenerate_inputs_never_add_a_panic_path() {
    // `step_multigrid` must panic exactly when `step` does: whatever the step
    // already cannot survive is not this entry point's doing, and anything it
    // survives, this one survives too.
    type Make = fn() -> (CfdSolver, Fix128, u32);
    let cases: [(&str, Make); 7] = [
        ("zero dt", || (scene(8, 8, 8), Fix128::ZERO, 4)),
        ("1x1x1", || (scene(1, 1, 1), dt(), 4)),
        ("8x8x1", || (scene(8, 8, 1), dt(), 4)),
        ("6x6x6", || (scene(6, 6, 6), dt(), 4)),
        ("cycles 0", || (scene(8, 8, 8), dt(), 0)),
        ("huge dt", || (scene(8, 8, 8), Fix128::from_int(1 << 40), 4)),
        ("zero dx", || {
            let mut s = scene(8, 8, 8);
            s.grid.dx = Fix128::ZERO;
            (s, dt(), 4)
        }),
    ];
    for (name, make) in cases {
        let (mut a, t, c) = make();
        let (mut b, _, _) = make();
        let mg = catch_unwind(AssertUnwindSafe(|| a.step_multigrid(t, c))).is_err();
        let gs = catch_unwind(AssertUnwindSafe(|| b.step(t))).is_err();
        assert_eq!(
            mg, gs,
            "{name}: step_multigrid panics = {mg}, step panics = {gs}"
        );
    }
}

#[test]
#[ignore = "diagnostic: the measurements the two tolerances above are fixed from"]
fn tolerance_measurement() {
    for cycles in [8u32, 16, 24, 32] {
        let n = 8;
        let mut mg = scene(n, n, n);
        let mut gs = scene(n, n, n);
        gs.jacobi_iterations = 600;
        let before = max_div(&mg.grid);
        mg.step_multigrid(dt(), cycles);
        gs.step(dt());
        let scale = max_abs(&gs.grid.u)
            .max(max_abs(&gs.grid.v))
            .max(max_abs(&gs.grid.w));
        let mut worst = 0.0f64;
        for (a, b) in [
            (&mg.grid.u, &gs.grid.u),
            (&mg.grid.v, &gs.grid.v),
            (&mg.grid.w, &gs.grid.w),
        ] {
            for (x, y) in a.iter().zip(b) {
                worst = worst.max((x.to_f64() - y.to_f64()).abs());
            }
        }
        eprintln!(
            "[step-mg] cycles={cycles:2} rel diff vs converged GS = {:.3e}  max|div| = {:.3e} (before {before:.3e}, ratio {:.3e})  |u|max = {scale:.3e}",
            worst / scale,
            max_div(&mg.grid),
            max_div(&mg.grid) / before
        );
    }
}
