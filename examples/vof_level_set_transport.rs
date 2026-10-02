//! Interface transport of the `multiphase` module: a VOF slab under rigid
//! convection, and a level-set sphere reinitialised by the two schemes
//!
//! Part 1 runs the translation test of a fraction field with
//! `advect_vof_rigid`: a two-cell slab of fraction `1` is carried by a
//! uniform velocity for `dt = dx / u` (one cell per step) under the upwind
//! and the semi-Lagrangian scheme, and the volume and the distance to the
//! exactly translated profile are printed after every step; the same slab is
//! then carried half a cell per step, where both schemes diffuse. Part 2
//! seeds `CfdSolver::level_set` with `initialize_level_set_sphere`, steps a
//! resting fluid with the fast-sweeping default and with
//! `LevelSetReinit::PseudoTime` through `StepOptions::with_level_set_reinit`,
//! and prints the enclosed volume against `4/3 π r³` and how far the
//! central-difference `|∇φ|` is from `1` over the interior. The last lines
//! show the refusal of a zero reinitialisation count.
//!
//! ```bash
//! cargo run --example vof_level_set_transport --features std
//! ```

use alice_physics::cfd_solver::{CfdSolver, LevelSetReinit, PressureSolver, StepOptions};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::{advect_vof_rigid, initialize_level_set_sphere, Grid3d, VofScheme};

/// An `n`-cell line along `x` with the fraction `1` in `[lo, hi)`.
fn slab(n: usize, dx: Fix128, lo: usize, hi: usize) -> Grid3d {
    let mut g = Grid3d::new(n, 1, 1, dx, Fix128::ZERO);
    for i in lo..hi {
        g.set(i, 0, 0, Fix128::ONE);
    }
    g
}

/// `Σ |f − g|` over the cells.
fn l1_distance(a: &Grid3d, b: &Grid3d) -> Fix128 {
    a.data
        .iter()
        .zip(&b.data)
        .fold(Fix128::ZERO, |acc, (&x, &y)| acc + (x - y).abs())
}

fn translation_test(scheme: VofScheme, cells_per_step: i64, steps: usize) {
    let n = 12;
    let dx = Fix128::from_ratio(1, 4);
    let u = Fix128::from_ratio(1, 2);
    // dt = cells_per_step · dx / u
    let dt = Fix128::from_int(cells_per_step) * dx / u;
    let mut f = slab(n, dx, 2, 4);
    let v0 = Fix128::from_int(2) * dx * dx * dx;
    println!(
        "  {scheme:?}, {cells_per_step} cell(s) per step: V0 = {:.6} m^3",
        v0.to_f64()
    );
    for step in 1..=steps {
        let volume = advect_vof_rigid(
            &mut f,
            scheme,
            Vec3Fix::new(u, Fix128::ZERO, Fix128::ZERO),
            dt,
        );
        let shift = (cells_per_step as usize) * step;
        let exact = slab(n, dx, 2 + shift, 4 + shift);
        println!(
            "    step {step}: V = {:.6} m^3 (V/V0 = {:.4}), L1 distance to the exact slab = {:.6}",
            volume.to_f64(),
            (volume / v0).to_f64(),
            l1_distance(&f, &exact).to_f64()
        );
    }
}

fn fractional_test(scheme: VofScheme) {
    let n = 12;
    let dx = Fix128::from_ratio(1, 4);
    let u = Fix128::from_ratio(1, 2);
    // half a cell per step
    let dt = dx / (u + u);
    let mut f = slab(n, dx, 2, 4);
    let v0 = Fix128::from_int(2) * dx * dx * dx;
    let mut volume = Fix128::ZERO;
    for _ in 0..4 {
        volume = advect_vof_rigid(
            &mut f,
            scheme,
            Vec3Fix::new(u, Fix128::ZERO, Fix128::ZERO),
            dt,
        );
    }
    let exact = slab(n, dx, 4, 6);
    let peak = f
        .data
        .iter()
        .copied()
        .fold(Fix128::ZERO, |a, b| if b > a { b } else { a });
    println!(
        "  {scheme:?}, half a cell per step, 4 steps (2 cells): V/V0 = {:.4}, L1 distance to the exact slab = {:.4}, peak fraction = {:.4}",
        (volume / v0).to_f64(),
        l1_distance(&f, &exact).to_f64(),
        peak.to_f64()
    );
}

/// Largest `| |∇φ| − 1 |` over the interior cells within two spacings of the
/// interface (`|φ| ≤ 2 dx`, where the reinitialisation matters; at the
/// sphere's centre the central difference is zero by symmetry and would
/// dominate the number), central differences.
fn eikonal_defect(ls: &Grid3d) -> Fix128 {
    let two_dx = ls.dx + ls.dx;
    let mut worst = Fix128::ZERO;
    for k in 1..ls.nz.saturating_sub(1) {
        for j in 1..ls.ny.saturating_sub(1) {
            for i in 1..ls.nx.saturating_sub(1) {
                if ls.get(i, j, k).abs() > two_dx {
                    continue;
                }
                let gx = (ls.get(i + 1, j, k) - ls.get(i - 1, j, k)) / two_dx;
                let gy = (ls.get(i, j + 1, k) - ls.get(i, j - 1, k)) / two_dx;
                let gz = (ls.get(i, j, k + 1) - ls.get(i, j, k - 1)) / two_dx;
                let d = ((gx * gx + gy * gy + gz * gz).sqrt() - Fix128::ONE).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
    }
    worst
}

/// Cells with `φ < 0` times the cell volume.
fn enclosed_volume(ls: &Grid3d) -> Fix128 {
    let inside = ls.data.iter().filter(|&&phi| phi < Fix128::ZERO).count();
    Fix128::from_int(inside as i64) * ls.dx * ls.dx * ls.dx
}

fn resting_sphere(n: usize) -> CfdSolver {
    let dx = Fix128::from_ratio(1, n as i64);
    let mut s = CfdSolver::new(n, n, n, dx);
    s.gravity = Vec3Fix::ZERO;
    s.surface_tension_n_m = Fix128::ZERO;
    s.reinit_every_n_steps = 1;
    let mut ls = Grid3d::new(n, n, n, dx, Fix128::ZERO);
    let centre = Fix128::from_ratio(1, 2);
    initialize_level_set_sphere(&mut ls, centre, centre, centre, Fix128::from_ratio(1, 4));
    s.level_set = Some(ls);
    s
}

fn main() {
    println!("Part 1: rigid convection of a two-cell VOF slab (n = 12, dx = 1/4 m, u = 1/2 m/s)");
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        translation_test(scheme, 1, 3);
    }
    for scheme in [VofScheme::Upwind, VofScheme::SemiLagrangian] {
        fractional_test(scheme);
    }

    println!();
    let n = 16;
    println!("Part 2: level-set sphere (r = 1/4 m in a unit box, {n}^3 cells) in a resting fluid");
    let exact_volume = 4.0 / 3.0 * core::f64::consts::PI * 0.25_f64 * 0.25 * 0.25;
    let dt = Fix128::from_ratio(1, 64);
    let options = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 30 });
    let schemes = [
        ("fast sweeping (default)", options),
        (
            "pseudo-time, 4 iterations",
            options.with_level_set_reinit(LevelSetReinit::PseudoTime { iterations: 4 }),
        ),
        (
            "pseudo-time, 16 iterations",
            options.with_level_set_reinit(LevelSetReinit::PseudoTime { iterations: 16 }),
        ),
    ];
    for (name, opts) in schemes {
        let mut s = resting_sphere(n);
        let ls0 = s.level_set.clone().expect("seeded");
        println!(
            "  {name}: seed       V = {:.6} m^3 (exact 4/3 pi r^3 = {exact_volume:.6}), max | |grad phi| - 1 | = {:.6}",
            enclosed_volume(&ls0).to_f64(),
            eikonal_defect(&ls0).to_f64()
        );
        // The first step never reinitialises (step_count = 0); the second does.
        for _ in 0..2 {
            s.step_with_options(dt, &opts)
                .expect("a resting fluid steps");
        }
        let ls = s.level_set.as_ref().expect("still seeded");
        println!(
            "  {name}: 2 steps    V = {:.6} m^3, max | |grad phi| - 1 | = {:.6}, step_count = {}",
            enclosed_volume(ls).to_f64(),
            eikonal_defect(ls).to_f64(),
            s.step_count
        );
    }

    println!();
    let mut s = resting_sphere(n);
    for reinit in [
        LevelSetReinit::FastSweeping { sweeps: 0 },
        LevelSetReinit::PseudoTime { iterations: 0 },
    ] {
        let err = s
            .step_with_options(dt, &options.with_level_set_reinit(reinit))
            .expect_err("a zero count is refused");
        println!(
            "  {reinit:?}: refused, {err}; step_count stays {}",
            s.step_count
        );
    }
}
