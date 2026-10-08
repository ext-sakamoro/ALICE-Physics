//! Oracle for `CfdSolver`'s time-stepping order (`COV-CFD-026`):
//! `src/cfd_solver.rs` documents this as "a first-order operator-splitting
//! scheme", and the oracle this file pins is that claim itself — halving
//! `dt` halves the time-discretisation error.
//!
//! The scenario is the same shear-diffusion eigenmode as
//! `tests/engineering_oracles_fluid.rs::cfd_solver_shear_mode_decays_at_the_viscous_rate`
//! (zero divergence, zero advection, pure `∂u/∂t = ν ∂²u/∂y²`), but the
//! reference is not that test's continuous closed form
//! `exp(−ν π² t / L²)`: comparing against it conflates the solver's time
//! error with its *spatial* discretisation error, which does not shrink as
//! `dt` shrinks and is the same order of magnitude here (confirmed
//! empirically: the raw closed-form error is not monotonic in `dt`).
//!
//! The discrete central-difference Laplacian has its own, exactly known
//! eigenvalue for this cosine mode under the mirror (zero-flux) boundary:
//! `μ = (4/dx²) sin²(π dx / 2L)` (the standard discrete-Laplacian
//! eigenvalue for a wavenumber-`π/L` mode on a cell-centred grid with
//! Neumann ends). The solver's own per-step update is then exactly forward
//! Euler on that single mode, `aₙ₊₁ = aₙ (1 − ν μ dt)`, which the first
//! test below confirms to the representation's own floor (`4e-17`, pure
//! rounding — not an approximation). That isolates the question
//! "first-order in dt" to a comparison with **no solver and no spatial
//! term in it at all**: `(1 − ν μ dt)ⁿ` against the continuous-time
//! `exp(−ν μ n dt)` at the same `μ`, which differ by `O(dt)` and nothing
//! else — a textbook forward-Euler truncation error, not a measurement.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::{Fix128, Vec3Fix};
use core::f64::consts::PI;

const NX: usize = 64;
const NY: usize = 16;
const NZ: usize = 2;
const DX_F: f64 = 1.0 / 128.0;
const NU: f64 = 1e-3;
const AMP: f64 = 1e-3;

/// `L = ny · dx`, the mode's half-wavelength.
fn length() -> f64 {
    NY as f64 * DX_F
}

/// The discrete central-difference Laplacian's eigenvalue for the cosine
/// mode `cos(π y / L)` under the mirror (zero-flux) boundary this grid
/// uses — not the continuous `(π/L)²`, which is a different number and
/// the source of the spatial term this file exists to remove from the
/// comparison.
fn discrete_eigenvalue() -> f64 {
    let l = length();
    (4.0 / (DX_F * DX_F)) * (PI * DX_F / (2.0 * l)).sin().powi(2)
}

/// The centre-column `u` profile (`i = nx/2`) after running the shear
/// eigenmode scenario for `steps` steps of `dt_f`, each.
fn centre_column(dt_f: f64, steps: usize) -> Vec<f64> {
    let dx = Fix128::from_ratio(1, 128);
    let mut solver = CfdSolver::new(NX, NY, NZ, dx);
    solver.gravity = Vec3Fix::ZERO;
    solver.density_kg_m3 = Fix128::from_int(1000);
    solver.dynamic_viscosity_pas = Fix128::ONE;
    for k in 0..NZ {
        for j in 0..NY {
            let value = Fix128::from_f64(AMP * (PI * (j as f64 + 0.5) / NY as f64).cos());
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

/// Every `dt` below stays inside the explicit-diffusion stability region
/// `src/cfd_solver.rs` documents (`r = ν dt / dx² ≤ 1/6`, the same limit
/// `step_with_options` enforces as `DiffusionUnstable`); `r(0.01) ≈ 0.164`,
/// just under the limit, and the finer runs only shrink `r` further.
const DTS: [f64; 5] = [0.01, 0.005, 0.0025, 0.00125, 0.000625];
const TOTAL_T: f64 = 0.5;

/// The solver's own step, restricted to this single diffusion mode, is
/// exactly forward Euler on `μ`: `aₙ₊₁ = aₙ (1 − ν μ dt)`. No tolerance
/// softens this — the two are expected to agree to the `f64` floor.
#[test]
fn the_solver_step_is_exactly_forward_euler_on_the_discrete_eigenvalue() {
    let mu = discrete_eigenvalue();
    for &dt_f in &DTS {
        let steps = (TOTAL_T / dt_f).round() as u32 as usize;
        let got = centre_column(dt_f, steps);
        let factor = (1.0 - NU * mu * dt_f).powi(steps as i32);
        let mut worst = 0.0f64;
        for (j, &g) in got.iter().enumerate() {
            let want = AMP * (PI * (j as f64 + 0.5) / NY as f64).cos() * factor;
            worst = worst.max((g - want).abs());
        }
        assert!(
            worst < 1e-12,
            "dt={dt_f}: solver departs from (1-nu*mu*dt)^n by {worst:e} (expected ~f64 rounding)"
        );
    }
}

/// `(1 − ν μ dt)ⁿ` (what the solver computes, per the test above) against
/// `exp(−ν μ n dt)` (the exact solution of the same discrete-space ODE,
/// continuous in time) at the same total time and the same `μ`: no solver
/// call, no spatial term, just the forward-Euler truncation error, which
/// must halve when `dt` halves.
#[test]
fn halving_dt_halves_the_pure_time_stepping_truncation_error() {
    let mu = discrete_eigenvalue();
    let errs: Vec<f64> = DTS
        .iter()
        .map(|&dt_f| {
            let steps = (TOTAL_T / dt_f).round() as u32 as usize;
            let discrete = (1.0 - NU * mu * dt_f).powi(steps as i32);
            let continuous = (-NU * mu * steps as f64 * dt_f).exp();
            (discrete - continuous).abs()
        })
        .collect();

    for w in errs.windows(2) {
        let ratio = w[0] / w[1];
        assert!(
            (1.9..=2.1).contains(&ratio),
            "successive dt-halving error ratio {ratio:.4}, expected close to 2 \
             for a first-order scheme (errs: {errs:?}, dts: {DTS:?})"
        );
    }
}
