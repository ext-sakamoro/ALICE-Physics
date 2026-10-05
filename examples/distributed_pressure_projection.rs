//! The slab-decomposed pressure projection with one thread per rank.
//!
//! `project_pressure_distributed` splits the grid into `ranks` contiguous `z`
//! slabs; every rank runs on its own thread, holds only what a process of a
//! distributed run would hold, and reaches its neighbours only through byte
//! messages. This example checks the two closed-form properties of the result:
//!
//! 1. it is **bit-identical** to the single-process projection
//!    (`project_pressure` / `project_pressure_multigrid`) for every rank count,
//!    including counts that do not divide `nz` and counts above it;
//! 2. the projected field is **discretely divergence-free**: the divergence,
//!    computed here from the raw face arrays, is a vanishing fraction of the
//!    seeded one.
//!
//! Run with `cargo run --release --example distributed_pressure_projection`.

use alice_physics::cfd_solver::PressureSolver;
use alice_physics::eulerian_grid::{
    project_pressure, project_pressure_distributed, project_pressure_multigrid, MacGrid,
};
use alice_physics::math::Fix128;

/// A closed 16³ box with a deterministic, structureless velocity field.
fn seeded(n: usize) -> MacGrid {
    let mut g = MacGrid::new(n, n, n, Fix128::from_ratio(1, 8));
    let wiggle = |a: usize, salt: u64| {
        let x = (a as u64).wrapping_mul(0x9E37_79B9).wrapping_add(salt) % 61;
        Fix128::from_ratio(x as i64 - 30, 16)
    };
    for (a, u) in g.u.iter_mut().enumerate() {
        *u = wiggle(a, 1);
    }
    for (a, v) in g.v.iter_mut().enumerate() {
        *v = wiggle(a, 2);
    }
    for (a, w) in g.w.iter_mut().enumerate() {
        *w = wiggle(a, 3);
    }
    g.set_closed_box_walls();
    g.enforce_face_boundaries();
    g
}

/// `max |∇·u|` from the raw face arrays.
fn max_divergence(g: &MacGrid) -> f64 {
    let (nx, ny, nz) = (g.nx, g.ny, g.nz);
    let mut worst = 0f64;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let u = |i: usize| g.u[i + (nx + 1) * (j + ny * k)].to_f64();
                let v = |j: usize| g.v[i + nx * (j + (ny + 1) * k)].to_f64();
                let w = |k: usize| g.w[i + nx * (j + ny * k)].to_f64();
                let d = (u(i + 1) - u(i) + v(j + 1) - v(j) + w(k + 1) - w(k)) / g.dx.to_f64();
                worst = worst.max(d.abs());
            }
        }
    }
    worst
}

fn same_bits(a: &MacGrid, b: &MacGrid) -> bool {
    a.pressure == b.pressure && a.u == b.u && a.v == b.v && a.w == b.w
}

fn main() {
    let n = 16;
    let dt = Fix128::from_ratio(1, 100);
    let rho = Fix128::from_int(1000);
    let base = seeded(n);
    let seeded_div = max_divergence(&base);
    println!("=== Distributed pressure projection, {n}³, closed box ===");
    println!("seeded max|div u| = {seeded_div:.3e}");

    let sweeps = 200;
    let cycles = 20;
    let mut gs_ref = base.clone();
    project_pressure(&mut gs_ref, dt, rho, sweeps);
    let mut mg_ref = base.clone();
    project_pressure_multigrid(&mut mg_ref, dt, rho, cycles);

    for ranks in [1usize, 2, 3, 4, 8, 20] {
        for (solver, reference) in [
            (PressureSolver::DecomposedGs { ranks, sweeps }, &gs_ref),
            (PressureSolver::BandedGs { ranks, sweeps }, &gs_ref),
            (
                PressureSolver::DecomposedMultigrid { ranks, cycles },
                &mg_ref,
            ),
        ] {
            let mut g = base.clone();
            project_pressure_distributed(&mut g, dt, rho, solver)
                .expect("a decomposed solver with a non-zero rank and iteration count");
            assert!(
                same_bits(&g, reference),
                "{solver:?} is not the single-process projection"
            );
            let div = max_divergence(&g);
            println!(
                "{solver:?}: bit-identical to the single-process solve, max|div u| = {div:.3e}"
            );
        }
    }

    // The closed form of the projection is ∇·u = 0; the multigrid solve reaches
    // it to round-off, the Gauss-Seidel one to its sweep count.
    assert!(max_divergence(&mg_ref) < seeded_div * 1e-10);
    assert!(max_divergence(&gs_ref) < seeded_div * 1e-3);
    println!("all rank counts reproduce the single-process answer; divergence removed");
}
