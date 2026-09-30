//! Scale probe for the HPC-parallel work (research wall "HPC parallel", step A-5).
//!
//! Answers one question with measurements instead of estimates: **how far is the
//! current single-process engine from the "hundreds of millions of elements"
//! target, in time and in memory?**
//!
//! Two element kinds are probed, because "element" means different things in the
//! two halves of the crate:
//!
//! - **Eulerian cells** — `MacGrid` of `nx · ny · nz` cells, driven through
//!   [`eulerian_grid::project_pressure`] (the order-dependent Gauss-Seidel
//!   pressure solve that the parallel work has to replace).
//! - **Rigid bodies** — `PhysicsWorld` stepped through `step()`.
//!
//! Memory is reported from `size_of` plus the live allocation counts, not from a
//! guess: `MacGrid` stores four `Vec<Fix128>` whose lengths are fixed by the
//! grid dimensions, so bytes-per-cell is exact.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --features std --example hpc_scale_probe
//! ```
//!
//! The probe prints a table and an extrapolation to 1e8 elements. It performs no
//! assertions — it is a measurement harness, not a test. The numbers it produces
//! belong in the wall record, not in a golden file, because they are
//! machine-specific by construction.

use alice_physics::eulerian_grid::{project_pressure, MacGrid};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};
use std::time::Instant;

/// Elements in the stated target ("hundreds of millions").
const TARGET_ELEMENTS: f64 = 1.0e8;

/// Bytes per Eulerian cell, derived from the four `Vec<Fix128>` a `MacGrid`
/// keeps. The face-velocity arrays are one cell longer along their own axis, so
/// this is a lower bound that becomes exact as the grid grows.
fn macgrid_bytes(nx: usize, ny: usize, nz: usize) -> usize {
    let f = size_of::<Fix128>();
    let u = (nx + 1) * ny * nz;
    let v = nx * (ny + 1) * nz;
    let w = nx * ny * (nz + 1);
    let p = nx * ny * nz;
    (u + v + w + p) * f
}

fn probe_eulerian() {
    println!("## Eulerian cells (MacGrid + project_pressure, 1 iteration)");
    println!();
    println!("| n (per axis) | cells | bytes | bytes/cell | project_pressure | ns/cell |");
    println!("|---|---|---|---|---|---|");

    let dt = Fix128::from_f64(1.0 / 60.0);
    let density = Fix128::from_int(1000);
    let dx = Fix128::from_f64(0.01);

    for &n in &[8usize, 16, 32, 48, 64, 96, 128] {
        let cells = n * n * n;
        let bytes = macgrid_bytes(n, n, n);
        let mut grid = MacGrid::new(n, n, n, dx);

        // Seed a divergent field so the solve has work to do; a zero field would
        // converge immediately and measure nothing.
        for (i, slot) in grid.u.iter_mut().enumerate() {
            *slot = Fix128::from_f64(((i % 7) as f64 - 3.0) * 0.1);
        }

        let t0 = Instant::now();
        project_pressure(&mut grid, dt, density, 1);
        let elapsed = t0.elapsed();

        let ns_per_cell = elapsed.as_secs_f64() * 1.0e9 / cells as f64;
        println!(
            "| {n} | {cells} | {:.1} MiB | {} | {:?} | {ns_per_cell:.1} |",
            bytes as f64 / (1024.0 * 1024.0),
            bytes / cells,
            elapsed,
        );
    }
    println!();

    // Extrapolate from the largest point measured above. Memory is exact (the
    // arrays are sized by construction); time assumes the per-cell cost stays at
    // its largest-measured value, which the table above is there to justify.
    let n = 128usize;
    let cells = n * n * n;
    let bytes_per_cell = macgrid_bytes(n, n, n) / cells;
    let target_bytes = bytes_per_cell as f64 * TARGET_ELEMENTS;
    println!(
        "Extrapolated to {TARGET_ELEMENTS:e} cells: **{:.1} GiB** of grid state alone \
         ({bytes_per_cell} B/cell, pressure + 3 face-velocity arrays, no solver scratch).",
        target_bytes / (1024.0 * 1024.0 * 1024.0),
    );
    println!(
        "Time at the last measured ns/cell, for ONE Gauss-Seidel iteration: see the \
         table's final row times {:.0}x.",
        TARGET_ELEMENTS / cells as f64,
    );
    println!();
}

fn probe_rigid_bodies() {
    println!("## Rigid bodies (PhysicsWorld::step)");
    println!();
    println!(
        "size_of::<RigidBody>() = {} B, size_of::<Fix128>() = {} B",
        size_of::<RigidBody>(),
        size_of::<Fix128>(),
    );
    println!();
    println!("| bodies | body bytes | step() | µs/body |");
    println!("|---|---|---|---|");

    let dt = Fix128::from_f64(1.0 / 60.0);

    for &count in &[1_000usize, 4_000, 16_000, 64_000, 256_000, 1_024_000] {
        let mut world = PhysicsWorld::new(SolverConfig::default());
        for i in 0..count {
            // Spread the bodies out so broad-phase is not degenerate: a single
            // overlapping pile would measure narrow-phase, not scaling.
            let x = (i % 100) as i64;
            let y = ((i / 100) % 100) as i64;
            let z = (i / 10_000) as i64;
            world.add_body(RigidBody::new_dynamic(
                Vec3Fix::from_int(x * 4, y * 4, z * 4),
                Fix128::from_int(1),
            ));
        }
        let bytes = count * size_of::<RigidBody>();

        let t0 = Instant::now();
        world.step(dt);
        let elapsed = t0.elapsed();

        let us_per_body = elapsed.as_secs_f64() * 1.0e6 / count as f64;
        println!(
            "| {count} | {:.1} MiB | {:?} | {us_per_body:.3} |",
            bytes as f64 / (1024.0 * 1024.0),
            elapsed,
        );
    }
    println!();

    let target_bytes = size_of::<RigidBody>() as f64 * TARGET_ELEMENTS;
    println!(
        "Extrapolated to {TARGET_ELEMENTS:e} bodies: **{:.1} GiB** of body state alone \
         (constraints, contact cache and BVH excluded).",
        target_bytes / (1024.0 * 1024.0 * 1024.0),
    );
    println!();
}

fn main() {
    println!("# ALICE-Physics HPC scale probe (A-5)");
    println!();
    println!(
        "Target for the wall: {TARGET_ELEMENTS:e} elements. Everything below is a \
         single-process, single-machine measurement on the host that ran it."
    );
    println!();

    probe_eulerian();
    probe_rigid_bodies();
}
