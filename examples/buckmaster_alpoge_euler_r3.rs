//! Alpöge–Buckmaster forced 3D Euler blow-up scaffolding.
//!
//! Evolves axisymmetric initial data patterned after the Theorem 1.1
//! setup of Alpöge–Buckmaster (2026 preprint,
//! `cims.nyu.edu/~tristanb/euler.pdf`) on the ALICE-Physics `CfdSolver`
//! full 3D MAC grid. The paper's construction is normalized to unit
//! ring radius `r₀ = 1`, `z₀ = 0` (Lemma 2.2); we adopt this scaling.
//!
//! ```text
//! T_R = {(r, φ, z) : (z - z₀)² + (r - r₀)² < R²},    R < r₀/2
//! u_in ∈ C_c^∞(T_R; R³),   axisymmetric,   nonzero swirl,   zero meridional
//! ```
//!
//! # Initial data
//!
//! Set `Γ_in(r, z) = Γ₀ · χ(d)` where `d = √((r - r₀)² + z²)` measures
//! the meridional distance from the initial ring. The physical velocity
//! is pure swirl: `u_φ = Γ_in / r`, `u_r = u_z = 0`. In Cartesian
//! coordinates:
//!
//! ```text
//! u_x(x, y, z) = -(y / r²) · Γ_in
//! u_y(x, y, z) =  (x / r²) · Γ_in
//! u_z(x, y, z) =  0
//! ```
//!
//! # Two scenarios
//!
//! | Γ₀ | Notes |
//! |----|-------|
//! |  1 | mild swirl, well-resolved on 40³ |
//! |  4 | stronger swirl, larger vorticity build-up |
//!
//! # Diagnostics per Theorem 1.1
//!
//! - `‖Γ‖_∞`, `‖u_r‖_∞`, `‖u_z‖_∞` (bounded per (1.2))
//! - `‖∇Γ‖_∞` (paper: → ∞ as `t ↑ T∗`)
//! - `‖ω‖_∞` (paper: → ∞ as `t ↑ T∗`)
//! - `∫ ‖ω(t)‖_∞ dt` (paper: divergent per (1.4), Beale-Kato-Majda)
//! - `max_divergence` (projection health)
//!
//! # Scope
//!
//! Solver + diagnostics infrastructure only. The complete finite family
//! construction of Section 5 (successive amplifications with material
//! coordinates and higher-order corrections) is future work; here we
//! demonstrate the base swirl evolves as expected with vorticity
//! amplification.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example buckmaster_alpoge_euler_r3
//! ```

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::{Fix128, Vec3Fix};

const N: usize = 40;
const DX_F32: f32 = 0.05; // physical domain [-1.0, 1.0]^3 → 2m / 40 = 0.05m per cell
const DT_F32: f32 = 5.0e-3;
const N_STEPS: usize = 400;
const LOG_EVERY: usize = 20;
const R0: f32 = 1.0; // ring radius (normalized per Lemma 2.2)
const Z0: f32 = 0.0;
const R_INNER: f32 = 0.30; // χ = 1 for d ≤ R_INNER
const R_OUTER: f32 = 0.40; // χ = 0 for d ≥ R_OUTER (< r₀/2 = 0.5)

fn main() {
    println!(
        "scenario,step,time_s,l_inf_gamma,l_inf_u_r,l_inf_u_z,l_inf_grad_gamma,l_inf_vorticity,vorticity_time_integral,max_divergence"
    );
    run_scenario(1.0);
    run_scenario(4.0);
}

fn run_scenario(gamma_0: f32) {
    let dx = Fix128::from_f32(DX_F32);
    let dt = Fix128::from_f32(DT_F32);
    let mut solver = CfdSolver::new(N, N, N, dx);
    configure_inviscid_euler(&mut solver);
    initialize_axisymmetric_swirl(&mut solver, gamma_0);

    let scenario = format!("Gamma0={gamma_0:.0}");
    let mut vort_integral = 0.0_f32;
    let mut prev_vort = 0.0_f32;
    log_diagnostics(&solver, &scenario, 0, vort_integral);
    for step in 1..=N_STEPS {
        solver.step(dt);
        // Trapezoidal integration of |ω|_∞ over time.
        let curr_vort = l_infty_vorticity(&solver);
        vort_integral += 0.5 * (prev_vort + curr_vort) * DT_F32;
        prev_vort = curr_vort;
        if step % LOG_EVERY == 0 {
            log_diagnostics(&solver, &scenario, step, vort_integral);
        }
    }
}

fn configure_inviscid_euler(solver: &mut CfdSolver) {
    solver.density_kg_m3 = Fix128::ONE;
    solver.dynamic_viscosity_pas = Fix128::ZERO;
    solver.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
    solver.jacobi_iterations = 60;
    solver.reinit_every_n_steps = 0;
    solver.use_turbulence = false;
    // Leave `temperature` and `level_set` as None — Euler system only.
}

/// Set the MAC velocity field to a smooth axisymmetric swirl supported in
/// a torus tube of radius `R_OUTER` around the initial ring.
fn initialize_axisymmetric_swirl(solver: &mut CfdSolver, gamma_0: f32) {
    let cx = (N as f32) * 0.5;
    let cy = (N as f32) * 0.5;
    let cz = (N as f32) * 0.5;

    // u faces at (i, j+0.5, k+0.5) physically
    for k in 0..N {
        let z = (k as f32 + 0.5 - cz) * DX_F32;
        for j in 0..N {
            let y = (j as f32 + 0.5 - cy) * DX_F32;
            for i in 0..=N {
                let x = (i as f32 - cx) * DX_F32;
                let r = x.hypot(y);
                let u_val = if r < 1.0e-6 {
                    0.0
                } else {
                    let d = ((r - R0).hypot(z - Z0)).max(0.0);
                    let gamma = gamma_0 * cutoff(d);
                    // u_x = -(y / r²) · Γ = -y · Γ / r²
                    -(y / (r * r)) * gamma
                };
                let ix = i + (N + 1) * (j + N * k);
                if ix < solver.grid.u.len() {
                    solver.grid.u[ix] = Fix128::from_f32(u_val);
                }
            }
        }
    }
    // v faces at (i+0.5, j, k+0.5)
    for k in 0..N {
        let z = (k as f32 + 0.5 - cz) * DX_F32;
        for j in 0..=N {
            let y = (j as f32 - cy) * DX_F32;
            for i in 0..N {
                let x = (i as f32 + 0.5 - cx) * DX_F32;
                let r = x.hypot(y);
                let v_val = if r < 1.0e-6 {
                    0.0
                } else {
                    let d = ((r - R0).hypot(z - Z0)).max(0.0);
                    let gamma = gamma_0 * cutoff(d);
                    (x / (r * r)) * gamma
                };
                let ix = i + N * (j + (N + 1) * k);
                if ix < solver.grid.v.len() {
                    solver.grid.v[ix] = Fix128::from_f32(v_val);
                }
            }
        }
    }
    // w faces stay zero (u_z = 0)
}

fn cutoff(d: f32) -> f32 {
    if d <= R_INNER {
        1.0
    } else if d >= R_OUTER {
        0.0
    } else {
        let s = (d - R_INNER) / (R_OUTER - R_INNER);
        0.5 * (1.0 + (std::f32::consts::PI * s).cos())
    }
}

fn log_diagnostics(solver: &CfdSolver, scenario: &str, step: usize, vort_integral: f32) {
    let time = step as f32 * DT_F32;
    let (l_inf_gamma, l_inf_ur, l_inf_uz) = bounded_quantities(solver);
    let l_inf_grad_gamma = l_infty_grad_gamma(solver);
    let l_inf_vort = l_infty_vorticity(solver);
    let max_div = l_infty_divergence(solver);
    println!(
        "{scenario},{step},{time:.4},{l_inf_gamma:.6e},{l_inf_ur:.6e},{l_inf_uz:.6e},{l_inf_grad_gamma:.6e},{l_inf_vort:.6e},{vort_integral:.6e},{max_div:.3e}"
    );
}

fn cell_center_velocity(solver: &CfdSolver, i: usize, j: usize, k: usize) -> (f32, f32, f32) {
    let (ux, uy, uz) = solver.grid.cell_velocity(i, j, k);
    (ux.to_f32(), uy.to_f32(), uz.to_f32())
}

/// Return `(‖Γ‖_∞, ‖u_r‖_∞, ‖u_z‖_∞)` in Cartesian sampling.
/// - `Γ = r · u_φ`, and `u_φ = (-y · ux + x · uy) / r`
///   so `Γ = -y·ux + x·uy`
/// - `u_r = (x·ux + y·uy) / r`
fn bounded_quantities(solver: &CfdSolver) -> (f32, f32, f32) {
    let cx = (N as f32) * 0.5;
    let cy = (N as f32) * 0.5;
    let mut best_gamma = 0.0_f32;
    let mut best_ur = 0.0_f32;
    let mut best_uz = 0.0_f32;
    for k in 0..N {
        for j in 0..N {
            let y = (j as f32 + 0.5 - cy) * DX_F32;
            for i in 0..N {
                let x = (i as f32 + 0.5 - cx) * DX_F32;
                let r = x.hypot(y);
                let (ux, uy, uz) = cell_center_velocity(solver, i, j, k);
                let gamma = (-y).mul_add(ux, x * uy);
                let ur = if r < 1.0e-6 {
                    0.0
                } else {
                    x.mul_add(ux, y * uy) / r
                };
                if gamma.abs() > best_gamma {
                    best_gamma = gamma.abs();
                }
                if ur.abs() > best_ur {
                    best_ur = ur.abs();
                }
                if uz.abs() > best_uz {
                    best_uz = uz.abs();
                }
            }
        }
    }
    (best_gamma, best_ur, best_uz)
}

fn l_infty_grad_gamma(solver: &CfdSolver) -> f32 {
    let cx = (N as f32) * 0.5;
    let cy = (N as f32) * 0.5;
    let inv_two_dx = 1.0 / (2.0 * DX_F32);
    let mut best = 0.0_f32;
    for k in 1..N - 1 {
        for j in 1..N - 1 {
            for i in 1..N - 1 {
                let gamma_x = |ii: usize, jj: usize, kk: usize| {
                    let x = (ii as f32 + 0.5 - cx) * DX_F32;
                    let y = (jj as f32 + 0.5 - cy) * DX_F32;
                    let (ux, uy, _) = cell_center_velocity(solver, ii, jj, kk);
                    (-y).mul_add(ux, x * uy)
                };
                let d_dx = (gamma_x(i + 1, j, k) - gamma_x(i - 1, j, k)) * inv_two_dx;
                let d_dy = (gamma_x(i, j + 1, k) - gamma_x(i, j - 1, k)) * inv_two_dx;
                let d_dz = (gamma_x(i, j, k + 1) - gamma_x(i, j, k - 1)) * inv_two_dx;
                let mag = (d_dx * d_dx + d_dy * d_dy + d_dz * d_dz).sqrt();
                if mag > best {
                    best = mag;
                }
            }
        }
    }
    best
}

fn l_infty_vorticity(solver: &CfdSolver) -> f32 {
    let inv_two_dx = 1.0 / (2.0 * DX_F32);
    let mut best = 0.0_f32;
    for k in 1..N - 1 {
        for j in 1..N - 1 {
            for i in 1..N - 1 {
                let (_ux_jp, _uy_jp, uz_jp) = cell_center_velocity(solver, i, j + 1, k);
                let (_ux_jm, _uy_jm, uz_jm) = cell_center_velocity(solver, i, j - 1, k);
                let (_ux_ip, uy_ip, uz_ip) = cell_center_velocity(solver, i + 1, j, k);
                let (_ux_im, uy_im, uz_im) = cell_center_velocity(solver, i - 1, j, k);
                let (ux_kp, uy_kp, _uz_kp) = cell_center_velocity(solver, i, j, k + 1);
                let (ux_km, uy_km, _uz_km) = cell_center_velocity(solver, i, j, k - 1);
                let (ux_jp2, _, _) = cell_center_velocity(solver, i, j + 1, k);
                let (ux_jm2, _, _) = cell_center_velocity(solver, i, j - 1, k);
                let d_uz_dy = (uz_jp - uz_jm) * inv_two_dx;
                let d_uy_dz = (uy_kp - uy_km) * inv_two_dx;
                let d_ux_dz = (ux_kp - ux_km) * inv_two_dx;
                let d_uz_dx = (uz_ip - uz_im) * inv_two_dx;
                let d_uy_dx = (uy_ip - uy_im) * inv_two_dx;
                let d_ux_dy = (ux_jp2 - ux_jm2) * inv_two_dx;
                let omega_x = d_uz_dy - d_uy_dz;
                let omega_y = d_ux_dz - d_uz_dx;
                let omega_z = d_uy_dx - d_ux_dy;
                let mag = (omega_x * omega_x + omega_y * omega_y + omega_z * omega_z).sqrt();
                if mag > best {
                    best = mag;
                }
            }
        }
    }
    best
}

fn l_infty_divergence(solver: &CfdSolver) -> f32 {
    let mut best = 0.0_f32;
    for k in 1..N - 1 {
        for j in 1..N - 1 {
            for i in 1..N - 1 {
                let mag = solver.grid.divergence(i, j, k).abs().to_f32();
                if mag > best {
                    best = mag;
                }
            }
        }
    }
    best
}
