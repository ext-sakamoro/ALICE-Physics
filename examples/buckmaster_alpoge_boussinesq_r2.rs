//! Alpöge–Buckmaster forced 2D Boussinesq blow-up scaffolding.
//!
//! Evolves the initial data (1.4) from Alpöge–Buckmaster (2026 preprint,
//! `cims.nyu.edu/~tristanb/boussinesq.pdf`) on the ALICE-Physics
//! `CfdSolver`, using a thin `nz = 4` slab as an R² approximation.
//!
//! ```text
//! θ_in(x₁, x₂) = -(A₀ / λ₀) · sin(λ₀ · x₂) · χ₀(|x|)
//! u_in = 0
//! ```
//!
//! Three scenarios sweep the amplitude parameter:
//!
//! | A₀ | λ₀ | -A₀/λ₀ |
//! |----|----|--------|
//! |  1 |  1 |    -1  |
//! |  4 |  1 |    -4  |
//! | 16 |  2 |    -8  |
//!
//! # Scope
//!
//! This example demonstrates the ALICE-Physics `CfdSolver` faithfully
//! integrates the inviscid Boussinesq system with the paper's initial
//! data on a bounded thin slab, and reports the sup-norm diagnostics
//! needed to track potential gradient growth:
//!
//! - `‖θ‖_∞` — temperature anomaly (paper asserts this stays bounded)
//! - `‖∇θ‖_∞` — temperature gradient (paper: → ∞ as t ↑ T∗)
//! - `‖ω‖_∞` — vorticity (paper: lim sup → ∞ as t ↑ T∗)
//!
//! The example does NOT reproduce the multiscale layer construction that
//! produces the actual finite-time blow-up. That construction requires
//! an infinite nested sequence of oscillatory perturbations tuned by the
//! amplitude equations (paper Section 3) and is future work.
//!
//! # Physical convention
//!
//! The paper's system reads `∂_t u + u·∇u + ∇p = θ e₂ + f_u`, meaning
//! buoyancy acts upward for positive `θ` (there is no separate uniform
//! gravity term). ALICE-Physics' `CfdSolver` always applies uniform
//! gravity, and its Boussinesq buoyancy uses `f = ρ·β·(T - T_ref)·g_y`.
//!
//! To match the paper's equation with unit buoyancy coefficient we set:
//! - `density_kg_m3 = 1`, `beta_per_k = 1`, `dynamic_viscosity_pas = 0`
//! - `gravity = (0, -1, 0)`, `reference_temp_k = 0`
//! - `T(cell) = 1 + θ_paper(x)`
//!
//! Verification: with `θ_paper = 0`, uniform gravity contributes `-Δt`
//! to `v` while buoyancy contributes `+Δt` (from `dt_temp = 1`), so the
//! quiescent state is a solver fixed point. For general `θ_paper` the
//! net vertical body force per unit time is exactly `θ_paper`.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example buckmaster_alpoge_boussinesq_r2
//! ```

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::Grid3d;

const NX: usize = 64;
const NY: usize = 64;
const NZ: usize = 4;
const DX_F32: f32 = 0.1;
const DT_F32: f32 = 0.01;
const N_STEPS: usize = 500;
const LOG_EVERY: usize = 25;
const R_INNER: f32 = 2.0; // χ₀ = 1 for |x| ≤ R_INNER
const R_OUTER: f32 = 3.0; // χ₀ = 0 for |x| ≥ R_OUTER

fn main() {
    println!("scenario,step,time_s,l_inf_theta,l_inf_grad_theta,l_inf_vorticity,max_divergence");
    run_scenario(1.0, 1.0);
    run_scenario(4.0, 1.0);
    run_scenario(16.0, 2.0);
}

fn run_scenario(a0: f32, lambda0: f32) {
    let dx = Fix128::from_f32(DX_F32);
    let dt = Fix128::from_f32(DT_F32);
    let mut solver = CfdSolver::new(NX, NY, NZ, dx);
    configure_inviscid_boussinesq(&mut solver);

    let mut temp = Grid3d::new(NX, NY, NZ, dx, Fix128::ONE);
    fill_initial_temperature(&mut temp, a0, lambda0);
    solver.temperature = Some(temp);

    let scenario = format!("A0={a0:.0}_lam0={lambda0:.0}");
    log_diagnostics(&solver, &scenario, 0);
    for step in 1..=N_STEPS {
        solver.step(dt);
        if step % LOG_EVERY == 0 {
            log_diagnostics(&solver, &scenario, step);
        }
    }
}

/// Apply the paper's unit-buoyancy convention with zero viscosity.
fn configure_inviscid_boussinesq(solver: &mut CfdSolver) {
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::ZERO;
    solver.gravity = Vec3Fix::new(Fix128::ZERO, -Fix128::ONE, Fix128::ZERO);
    solver.beta_per_k = Fix128::ONE;
    solver.reference_temp_k = Fix128::ZERO;
    solver.jacobi_iterations = 60;
    solver.reinit_every_n_steps = 0;
    solver.use_turbulence = false;
}

/// Write `T(i, j, k) = 1 + θ_paper(x_i, y_j)` into the cell-centred grid.
fn fill_initial_temperature(temp: &mut Grid3d, a0: f32, lambda0: f32) {
    let scale = -a0 / lambda0;
    let cx = (NX as f32) * 0.5;
    let cy = (NY as f32) * 0.5;
    for k in 0..NZ {
        for j in 0..NY {
            let y = (j as f32 + 0.5 - cy) * DX_F32;
            for i in 0..NX {
                let x = (i as f32 + 0.5 - cx) * DX_F32;
                let r = x.hypot(y);
                let theta = scale * (lambda0 * y).sin() * cutoff(r);
                let t_val = 1.0 + theta;
                temp.set(i, j, k, Fix128::from_f32(t_val));
            }
        }
    }
}

/// Smooth radial cutoff: 1 on `|x| ≤ R_INNER`, 0 outside `R_OUTER`,
/// half-cosine ramp between them. Not `C^∞` but adequate for numerical
/// demonstration; the paper's `χ₀ ∈ C_c^∞` is only required for the
/// analytic blow-up construction which this example does not reproduce.
fn cutoff(r: f32) -> f32 {
    if r <= R_INNER {
        1.0
    } else if r >= R_OUTER {
        0.0
    } else {
        let s = (r - R_INNER) / (R_OUTER - R_INNER);
        0.5 * (1.0 + (std::f32::consts::PI * s).cos())
    }
}

fn log_diagnostics(solver: &CfdSolver, scenario: &str, step: usize) {
    let time = step as f32 * DT_F32;
    let l_inf_theta = l_infty_theta(solver);
    let l_inf_grad = l_infty_grad_theta(solver);
    let l_inf_vort = l_infty_vorticity(solver);
    let max_div = l_infty_divergence(solver);
    println!(
        "{scenario},{step},{time:.4},{l_inf_theta:.6e},{l_inf_grad:.6e},{l_inf_vort:.6e},{max_div:.3e}"
    );
}

fn l_infty_theta(solver: &CfdSolver) -> f32 {
    let Some(temp) = solver.temperature.as_ref() else {
        return 0.0;
    };
    let mut best = 0.0_f32;
    for k in 0..NZ {
        for j in 0..NY {
            for i in 0..NX {
                let theta = temp.get(i, j, k).to_f32() - 1.0;
                let mag = theta.abs();
                if mag > best {
                    best = mag;
                }
            }
        }
    }
    best
}

fn l_infty_grad_theta(solver: &CfdSolver) -> f32 {
    let Some(temp) = solver.temperature.as_ref() else {
        return 0.0;
    };
    let inv_two_dx = 1.0 / (2.0 * DX_F32);
    let k = NZ / 2;
    let mut best = 0.0_f32;
    for j in 1..NY - 1 {
        for i in 1..NX - 1 {
            let t_ip = temp.get(i + 1, j, k).to_f32();
            let t_im = temp.get(i - 1, j, k).to_f32();
            let t_jp = temp.get(i, j + 1, k).to_f32();
            let t_jm = temp.get(i, j - 1, k).to_f32();
            let d_dx = (t_ip - t_im) * inv_two_dx;
            let d_dy = (t_jp - t_jm) * inv_two_dx;
            let mag = d_dx.hypot(d_dy);
            if mag > best {
                best = mag;
            }
        }
    }
    best
}

fn l_infty_vorticity(solver: &CfdSolver) -> f32 {
    let inv_two_dx = 1.0 / (2.0 * DX_F32);
    let k = NZ / 2;
    let mut best = 0.0_f32;
    for j in 1..NY - 1 {
        for i in 1..NX - 1 {
            let v_ip = solver.grid.v(i + 1, j, k).to_f32();
            let v_im = solver.grid.v(i - 1, j, k).to_f32();
            let u_jp = solver.grid.u(i, j + 1, k).to_f32();
            let u_jm = solver.grid.u(i, j - 1, k).to_f32();
            let d_vdx = (v_ip - v_im) * inv_two_dx;
            let d_udy = (u_jp - u_jm) * inv_two_dx;
            let omega = d_vdx - d_udy;
            let mag = omega.abs();
            if mag > best {
                best = mag;
            }
        }
    }
    best
}

fn l_infty_divergence(solver: &CfdSolver) -> f32 {
    let k = NZ / 2;
    let mut best = 0.0_f32;
    for j in 1..NY - 1 {
        for i in 1..NX - 1 {
            let mag = solver.grid.divergence(i, j, k).abs().to_f32();
            if mag > best {
                best = mag;
            }
        }
    }
    best
}
