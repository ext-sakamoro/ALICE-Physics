//! Oracle for `CfdSolver`'s time-stepping order (`COV-CFD-026`):
//! `src/cfd_solver.rs` documents this as "a first-order operator-splitting
//! scheme", and the oracle this file pins is that claim itself — halving
//! `dt` halves the time-discretisation error.
//!
//! The scenario is the same shear-diffusion eigenmode as
//! `tests/engineering_oracles_fluid.rs::cfd_solver_shear_mode_decays_at_the_viscous_rate`
//! (zero divergence, zero advection, pure `∂u/∂t = ν ∂²u/∂y²`), run to the
//! same total time `T` at four geometrically halved `dt`. Comparing each run
//! directly against the continuous closed form `exp(−ν π² t / L²)` does not
//! isolate the time-stepping error: the fixed spatial grid (`ny = 16`) has
//! its own `O(dx²)` error that does not shrink with `dt`, and at these grid
//! sizes it is the same order of magnitude as the time error, so the two
//! partially cancel at some `dt` and add at others (confirmed empirically:
//! the raw closed-form error is *not* monotonic in `dt` here).
//!
//! Comparing **successive runs to each other** instead removes the spatial
//! term, which is identical (same `dx`) at every `dt`: if the time error is
//! `C·dt` to leading order, run(`dt`) and run(`dt/2`) differ from the exact
//! solution by `C·dt` and `C·dt/2`, so they differ from *each other* by
//! `C·dt/2` — and run(`dt/2`) − run(`dt/4`) by `C·dt/4`, a ratio of exactly
//! 2 between successive differences, with no reference to any closed form
//! at all. This is the standard grid/time refinement study (Richardson
//! extrapolation's diagnostic half), not a comparison this module's own
//! output invented.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle is the ratio between successive runs, not a simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::{Fix128, Vec3Fix};
use core::f64::consts::PI;

const NX: usize = 64;
const NY: usize = 16;
const NZ: usize = 2;

/// The centre-column `u` profile (`i = nx/2`) after running the shear
/// eigenmode scenario for `steps` steps of `dt_f`, each.
fn centre_column(dt_f: f64, steps: usize) -> Vec<f64> {
    let dx = Fix128::from_ratio(1, 128);
    let mut solver = CfdSolver::new(NX, NY, NZ, dx);
    solver.gravity = Vec3Fix::ZERO;
    solver.density_kg_m3 = Fix128::from_int(1000);
    solver.dynamic_viscosity_pas = Fix128::ONE;
    let amp = 1e-3f64;
    for k in 0..NZ {
        for j in 0..NY {
            let value = Fix128::from_f64(amp * (PI * (j as f64 + 0.5) / NY as f64).cos());
            for i in 0..=NX {
                let ix = i + (NX + 1) * (j + NY * k);
                solver.grid.u[ix] = value;
            }
        }
    }
    let dt = Fix128::from_f64(dt_f);
    for _ in 0..steps {
        solver.step(dt);
    }
    (0..NY)
        .map(|j| solver.grid.u(NX / 2, j, 0).to_f64())
        .collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Four geometrically halved `dt`, same total time `T = 0.5` s, same `dx`:
/// three successive-difference ratios, each expected near 2 for a
/// first-order scheme. `0.02` is the coarsest step that stays in the
/// explicit-diffusion stability region used by the sibling decay test
/// (`r = ν dt / dx² ≈ 0.33` at `dt = 0.02`, under the `1/6`-per-axis limit
/// noted there once halved twice more for the finer runs).
#[test]
fn halving_dt_halves_the_successive_time_stepping_difference() {
    let total_t = 0.5f64;
    let dts = [0.02f64, 0.01, 0.005, 0.0025, 0.00125];
    let fields: Vec<Vec<f64>> = dts
        .iter()
        .map(|&dt_f| {
            let steps = (total_t / dt_f).round() as u32 as usize;
            centre_column(dt_f, steps)
        })
        .collect();

    let diffs: Vec<f64> = fields.windows(2).map(|w| max_diff(&w[0], &w[1])).collect();

    for d in &diffs {
        assert!(
            *d > 1e-9,
            "successive-run difference {d:e} is too small to measure a ratio from \
             (dts: {dts:?})"
        );
    }

    for (i, w) in diffs.windows(2).enumerate() {
        let ratio = w[0] / w[1];
        assert!(
            (1.7..=2.3).contains(&ratio),
            "diff[{i}]/diff[{}] = {ratio:.3}, expected close to 2 for a \
             first-order scheme (diffs: {diffs:?}, dts: {dts:?})",
            i + 1
        );
    }
}
