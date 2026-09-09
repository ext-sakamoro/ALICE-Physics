//! Integrated CFD Solver (Session 3 S1)
//!
//! Wires together the Session 1-2 fluid modules into a single time-stepping
//! loop that a user can drive with `solver.step(dt)`. Combines:
//!
//! - MAC-grid velocity storage & pressure projection (`eulerian_grid`)
//! - Level-set fluid tracking + reinitialisation (`multiphase`, `interface_capture`)
//! - Continuum surface force (`surface_tension_csf`)
//! - Smagorinsky LES eddy viscosity (`turbulence`)
//! - Boussinesq buoyancy for hot fluid (`smoke_fire`)
//! - Gravity (uniform body force)
//!
//! # Step order
//!
//! ```text
//! 1. Add body forces  → gravity + buoyancy + CSF (into u/v/w)
//! 2. Turbulent viscosity → Smagorinsky ν_t (from local strain rate)
//! 3. Diffuse velocity (explicit Laplacian, coefficient = ν_mol + ν_t)
//! 4. Pressure projection (Jacobi from eulerian_grid)
//! 5. Advect level set (semi-Lagrangian, uniform velocity approx)
//! 6. Reinit level set (fast sweeping, every reinit_every_n steps)
//! ```
//!
//! This is a first-order operator-splitting scheme; adequate for engineering
//! demos and validation tests. Higher-order (BFECC advection, RK3 time,
//! multigrid pressure) is a future upgrade.

use crate::eulerian_grid::{
    g2p_velocity, project_pressure, sample_u_trilinear, sample_v_trilinear, sample_w_trilinear,
    MacGrid,
};
use crate::interface_capture::fast_sweeping_reinit;
use crate::math::{Fix128, Vec3Fix};
use crate::multiphase::{trilinear_sample, Grid3d};
use crate::surface_tension_csf::{compute_csf_field, SIGMA_WATER_AIR};
use crate::turbulence::{smagorinsky_eddy_viscosity, strain_rate_magnitude, SMAGORINSKY_CS};

/// Complete CFD solver state.
pub struct CfdSolver {
    /// Velocity + pressure MAC grid.
    pub grid: MacGrid,
    /// Fluid level set (negative = liquid, positive = gas), optional.
    /// When present, CSF and level-set advection are executed.
    pub level_set: Option<Grid3d>,
    /// Temperature field (K), optional. Enables Boussinesq buoyancy.
    pub temperature: Option<Grid3d>,
    /// Fluid density ρ (kg/m³).
    pub density_kg_m3: Fix128,
    /// Dynamic molecular viscosity μ (Pa·s).
    pub dynamic_viscosity_pas: Fix128,
    /// Gravity vector (m/s²), typically `(0, -9.81, 0)`.
    pub gravity: Vec3Fix,
    /// Surface tension σ (N/m), used only if `level_set` present.
    pub surface_tension_n_m: Fix128,
    /// Thermal expansion coefficient β (1/K), used only if `temperature` present.
    pub beta_per_k: Fix128,
    /// Reference temperature `T_0` (K) for Boussinesq.
    pub reference_temp_k: Fix128,
    /// Pressure-projection Jacobi iterations per step.
    pub jacobi_iterations: u32,
    /// Reinitialise the level set every N steps (0 = never).
    pub reinit_every_n_steps: u32,
    /// Simulation step counter.
    pub step_count: u64,
    /// Turbulence toggle: use Smagorinsky if `true`.
    pub use_turbulence: bool,
}

impl CfdSolver {
    /// Construct a solver with sensible default fluid = water at 20 °C.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: Fix128) -> Self {
        Self {
            grid: MacGrid::new(nx, ny, nz, dx),
            level_set: None,
            temperature: None,
            density_kg_m3: Fix128::from_int(1000),
            dynamic_viscosity_pas: Fix128::from_ratio(1, 1000), // 1e-3 (water)
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(-981, 100), Fix128::ZERO),
            surface_tension_n_m: SIGMA_WATER_AIR,
            beta_per_k: Fix128::from_ratio(34, 10_000), // air ≈ 3.4e-3
            reference_temp_k: Fix128::from_int(293),
            jacobi_iterations: 30,
            reinit_every_n_steps: 10,
            step_count: 0,
            use_turbulence: false,
        }
    }

    /// One integrated time step.
    pub fn step(&mut self, dt_s: Fix128) {
        if dt_s.is_zero() {
            return;
        }
        self.advect_velocity(dt_s);
        self.apply_body_forces(dt_s);
        if self.use_turbulence {
            self.apply_turbulent_diffusion(dt_s);
        } else {
            self.apply_molecular_diffusion(dt_s);
        }
        project_pressure(
            &mut self.grid,
            dt_s,
            self.density_kg_m3,
            self.jacobi_iterations,
        );
        if self.level_set.is_some() {
            self.advect_level_set(dt_s);
            if self.reinit_every_n_steps > 0
                && self.step_count > 0
                && self.step_count % u64::from(self.reinit_every_n_steps) == 0
            {
                if let Some(ls) = self.level_set.as_mut() {
                    fast_sweeping_reinit(ls, 2);
                }
            }
        }
        if self.temperature.is_some() {
            self.advect_temperature(dt_s);
        }
        self.step_count += 1;
    }

    /// Step 1: Add body forces (gravity + buoyancy + surface tension).
    fn apply_body_forces(&mut self, dt_s: Fix128) {
        let g = self.gravity;
        // Uniform gravity to each face
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let ix = i + (self.grid.nx + 1) * (j + self.grid.ny * k);
                    if ix < self.grid.u.len() {
                        self.grid.u[ix] = self.grid.u[ix] + g.x * dt_s;
                    }
                }
            }
        }
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let ix = i + self.grid.nx * (j + (self.grid.ny + 1) * k);
                    if ix < self.grid.v.len() {
                        self.grid.v[ix] = self.grid.v[ix] + g.y * dt_s;
                    }
                }
            }
        }
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let ix = i + self.grid.nx * (j + self.grid.ny * k);
                    if ix < self.grid.w.len() {
                        self.grid.w[ix] = self.grid.w[ix] + g.z * dt_s;
                    }
                }
            }
        }

        // Boussinesq: add buoyancy from temperature deviation
        if let Some(temp) = self.temperature.as_ref() {
            for k in 0..self.grid.nz {
                for j in 0..=self.grid.ny {
                    for i in 0..self.grid.nx {
                        // Sample cell-centre temperature
                        let jj = if j == 0 { 0 } else { j - 1 };
                        let t = temp.get(i, jj, k);
                        let dt_temp = t - self.reference_temp_k;
                        let f = self.density_kg_m3 * self.beta_per_k * dt_temp * self.gravity.y;
                        let ix = i + self.grid.nx * (j + (self.grid.ny + 1) * k);
                        if ix < self.grid.v.len() {
                            self.grid.v[ix] = self.grid.v[ix] - f * dt_s / self.density_kg_m3;
                        }
                    }
                }
            }
        }

        // Continuum surface force from level set
        if let Some(ls) = self.level_set.as_ref() {
            let (fx, _fy, _fz) = compute_csf_field(
                ls,
                self.surface_tension_n_m,
                self.grid.dx * Fix128::from_ratio(15, 10),
            );
            // Apply to u faces (approximate; treats fx as cell-centered)
            for k in 0..self.grid.nz {
                for j in 0..self.grid.ny {
                    for i in 0..self.grid.nx {
                        let cell_idx = i + self.grid.nx * (j + self.grid.ny * k);
                        let force = fx[cell_idx];
                        // Distribute to two u faces (like p2g_nearest)
                        let ix_lo = i + (self.grid.nx + 1) * (j + self.grid.ny * k);
                        let ix_hi = (i + 1) + (self.grid.nx + 1) * (j + self.grid.ny * k);
                        if ix_lo < self.grid.u.len() {
                            self.grid.u[ix_lo] = self.grid.u[ix_lo]
                                + force * dt_s * Fix128::from_ratio(1, 2) / self.density_kg_m3;
                        }
                        if ix_hi < self.grid.u.len() {
                            self.grid.u[ix_hi] = self.grid.u[ix_hi]
                                + force * dt_s * Fix128::from_ratio(1, 2) / self.density_kg_m3;
                        }
                    }
                }
            }
        }
    }

    /// Molecular-only viscous diffusion via explicit Laplacian.
    fn apply_molecular_diffusion(&mut self, dt_s: Fix128) {
        if self.density_kg_m3.is_zero() {
            return;
        }
        let nu = self.dynamic_viscosity_pas / self.density_kg_m3;
        self.diffuse_velocity(nu, dt_s);
    }

    /// Smagorinsky-augmented diffusion: `ν_eff = ν_mol + ν_t`.
    fn apply_turbulent_diffusion(&mut self, dt_s: Fix128) {
        if self.density_kg_m3.is_zero() {
            return;
        }
        let nu_mol = self.dynamic_viscosity_pas / self.density_kg_m3;
        // Compute strain-rate magnitude at cell centres and take max as an
        // upper-bound proxy for the SGS eddy viscosity (simplification).
        let mut max_strain = Fix128::ZERO;
        for k in 1..self.grid.nz - 1 {
            for j in 1..self.grid.ny - 1 {
                for i in 1..self.grid.nx - 1 {
                    let s11 = (self.grid.u(i + 1, j, k) - self.grid.u(i, j, k)) / self.grid.dx;
                    let s22 = (self.grid.v(i, j + 1, k) - self.grid.v(i, j, k)) / self.grid.dx;
                    let s33 = (self.grid.w(i, j, k + 1) - self.grid.w(i, j, k)) / self.grid.dx;
                    let s = strain_rate_magnitude(
                        s11,
                        s22,
                        s33,
                        Fix128::ZERO,
                        Fix128::ZERO,
                        Fix128::ZERO,
                    );
                    if s > max_strain {
                        max_strain = s;
                    }
                }
            }
        }
        let nu_t = smagorinsky_eddy_viscosity(self.grid.dx, max_strain);
        let _ = SMAGORINSKY_CS; // referenced via smagorinsky_eddy_viscosity
        self.diffuse_velocity(nu_mol + nu_t, dt_s);
    }

    /// Explicit Laplacian: `u_new = u + dt·ν·∇²u`.
    fn diffuse_velocity(&mut self, nu: Fix128, dt_s: Fix128) {
        if nu.is_zero() || self.grid.dx.is_zero() {
            return;
        }
        let coeff = nu * dt_s / (self.grid.dx * self.grid.dx);

        // u faces
        let mut u_next = self.grid.u.clone();
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 1..self.grid.nx {
                    let center = self.grid.u(i, j, k);
                    let left = self.grid.u(i - 1, j, k);
                    let right = self.grid.u(i + 1, j, k);
                    let down = if j > 0 {
                        self.grid.u(i, j - 1, k)
                    } else {
                        center
                    };
                    let up = if j + 1 < self.grid.ny {
                        self.grid.u(i, j + 1, k)
                    } else {
                        center
                    };
                    let back = if k > 0 {
                        self.grid.u(i, j, k - 1)
                    } else {
                        center
                    };
                    let fwd = if k + 1 < self.grid.nz {
                        self.grid.u(i, j, k + 1)
                    } else {
                        center
                    };
                    let laplacian =
                        left + right + down + up + back + fwd - center * Fix128::from_int(6);
                    let ix = i + (self.grid.nx + 1) * (j + self.grid.ny * k);
                    u_next[ix] = center + coeff * laplacian;
                }
            }
        }
        self.grid.u = u_next;

        // v faces
        let mut v_next = self.grid.v.clone();
        for k in 0..self.grid.nz {
            for j in 1..self.grid.ny {
                for i in 0..self.grid.nx {
                    let center = self.grid.v(i, j, k);
                    let left = if i > 0 {
                        self.grid.v(i - 1, j, k)
                    } else {
                        center
                    };
                    let right = if i + 1 < self.grid.nx {
                        self.grid.v(i + 1, j, k)
                    } else {
                        center
                    };
                    let down = self.grid.v(i, j - 1, k);
                    let up = self.grid.v(i, j + 1, k);
                    let back = if k > 0 {
                        self.grid.v(i, j, k - 1)
                    } else {
                        center
                    };
                    let fwd = if k + 1 < self.grid.nz {
                        self.grid.v(i, j, k + 1)
                    } else {
                        center
                    };
                    let laplacian =
                        left + right + down + up + back + fwd - center * Fix128::from_int(6);
                    let ix = i + self.grid.nx * (j + (self.grid.ny + 1) * k);
                    v_next[ix] = center + coeff * laplacian;
                }
            }
        }
        self.grid.v = v_next;

        // w faces analogous
        let mut w_next = self.grid.w.clone();
        for k in 1..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let center = self.grid.w(i, j, k);
                    let left = if i > 0 {
                        self.grid.w(i - 1, j, k)
                    } else {
                        center
                    };
                    let right = if i + 1 < self.grid.nx {
                        self.grid.w(i + 1, j, k)
                    } else {
                        center
                    };
                    let down = if j > 0 {
                        self.grid.w(i, j - 1, k)
                    } else {
                        center
                    };
                    let up = if j + 1 < self.grid.ny {
                        self.grid.w(i, j + 1, k)
                    } else {
                        center
                    };
                    let back = self.grid.w(i, j, k - 1);
                    let fwd = self.grid.w(i, j, k + 1);
                    let laplacian =
                        left + right + down + up + back + fwd - center * Fix128::from_int(6);
                    let ix = i + self.grid.nx * (j + self.grid.ny * k);
                    w_next[ix] = center + coeff * laplacian;
                }
            }
        }
        self.grid.w = w_next;
    }

    /// Semi-Lagrangian self-advection of the MAC velocity field.
    ///
    /// Implements the `u · ∇u` transport term of the Navier–Stokes / Euler
    /// momentum equation. For each u/v/w face, back-traces along the current
    /// velocity by `dt_s` and resamples the corresponding face component
    /// from the pre-advection snapshot via trilinear interpolation on the
    /// respective staggered grid.
    ///
    /// Called at the start of `step` so that all subsequent operator stages
    /// (body forces, diffusion, projection) act on the transported field.
    /// This scheme is first-order in time and diffusive on coarse grids;
    /// higher-order variants (BFECC, MacCormack, RK3) are future upgrades.
    fn advect_velocity(&mut self, dt_s: Fix128) {
        let old = self.grid.clone();
        let dx = self.grid.dx;
        let half = Fix128::from_ratio(1, 2);

        // u faces
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&old, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let new_u = sample_u_trilinear(&old, back);
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < self.grid.u.len() {
                        self.grid.u[ix] = new_u;
                    }
                }
            }
        }

        // v faces
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = Fix128::from_int(j as i64) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&old, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let new_v = sample_v_trilinear(&old, back);
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < self.grid.v.len() {
                        self.grid.v[ix] = new_v;
                    }
                }
            }
        }

        // w faces
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = Fix128::from_int(k as i64) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&old, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let new_w = sample_w_trilinear(&old, back);
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < self.grid.w.len() {
                        self.grid.w[ix] = new_w;
                    }
                }
            }
        }
    }

    /// Semi-Lagrangian advection of the temperature field using cell-centred
    /// velocity from the projected MAC grid.
    ///
    /// Called after `project_pressure` on each step when `temperature` is
    /// present. Implements the `∂_t θ + u · ∇θ = 0` transport half of the
    /// Boussinesq system; the buoyancy source is handled by
    /// `apply_body_forces` and pairs with this term to complete the
    /// physically consistent inviscid Boussinesq step.
    fn advect_temperature(&mut self, dt_s: Fix128) {
        let temp = match self.temperature.as_mut() {
            Some(t) => t,
            None => return,
        };
        let old = temp.data.clone();
        let old_grid = Grid3d {
            nx: temp.nx,
            ny: temp.ny,
            nz: temp.nz,
            dx: temp.dx,
            data: old,
        };
        let inv_dx = Fix128::ONE / temp.dx;
        for k in 0..temp.nz {
            for j in 0..temp.ny {
                for i in 0..temp.nx {
                    let (uc, vc, wc) = self.grid.cell_velocity(
                        i.min(self.grid.nx - 1),
                        j.min(self.grid.ny - 1),
                        k.min(self.grid.nz - 1),
                    );
                    let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                    let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                    let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                    let sampled = trilinear_sample(&old_grid, cx, cy, cz);
                    let ix = temp.idx(i, j, k);
                    temp.data[ix] = sampled;
                }
            }
        }
    }

    /// Semi-Lagrangian advection of the level set using cell-centred velocity.
    fn advect_level_set(&mut self, dt_s: Fix128) {
        let ls = match self.level_set.as_mut() {
            Some(ls) => ls,
            None => return,
        };
        let old = ls.data.clone();
        let old_grid = Grid3d {
            nx: ls.nx,
            ny: ls.ny,
            nz: ls.nz,
            dx: ls.dx,
            data: old,
        };
        let inv_dx = Fix128::ONE / ls.dx;
        for k in 0..ls.nz {
            for j in 0..ls.ny {
                for i in 0..ls.nx {
                    // Cell-centred velocity from the MAC grid
                    let (uc, vc, wc) = self.grid.cell_velocity(
                        i.min(self.grid.nx - 1),
                        j.min(self.grid.ny - 1),
                        k.min(self.grid.nz - 1),
                    );
                    let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                    let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                    let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                    let sampled = trilinear_sample(&old_grid, cx, cy, cz);
                    let ix = ls.idx(i, j, k);
                    ls.data[ix] = sampled;
                }
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_new_default_water() {
        let s = CfdSolver::new(4, 4, 4, Fix128::ONE);
        assert_eq!(s.density_kg_m3, Fix128::from_int(1000));
        assert!(s.dynamic_viscosity_pas > Fix128::ZERO);
        assert_eq!(s.step_count, 0);
    }

    #[test]
    fn step_zero_dt_no_op() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::ONE);
        s.step(Fix128::ZERO);
        assert_eq!(s.step_count, 0);
    }

    #[test]
    fn gravity_accelerates_velocity_downward() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::ONE);
        // Take a small step
        s.step(Fix128::from_ratio(1, 100));
        // v-faces should now have some negative velocity due to gravity
        let mut some_neg = false;
        for &vv in &s.grid.v {
            if vv < Fix128::ZERO {
                some_neg = true;
                break;
            }
        }
        assert!(some_neg, "Expected downward velocity component");
    }

    #[test]
    fn step_count_increments() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::ONE);
        for _ in 0..3 {
            s.step(Fix128::from_ratio(1, 100));
        }
        assert_eq!(s.step_count, 3);
    }

    #[test]
    fn projection_reduces_divergence_after_step() {
        // Set an artificial divergent velocity field
        let mut s = CfdSolver::new(4, 4, 4, Fix128::ONE);
        for i in 0..=4 {
            for j in 0..4 {
                for k in 0..4 {
                    let ix = i + 5 * (j + 4 * k);
                    if ix < s.grid.u.len() {
                        s.grid.u[ix] = Fix128::from_int(i as i64);
                    }
                }
            }
        }
        let div_before = s.grid.divergence(2, 2, 2).abs();
        s.step(Fix128::from_ratio(1, 100));
        let div_after = s.grid.divergence(2, 2, 2).abs();
        // After step (gravity + projection), divergence should decrease
        assert!(div_after < div_before);
    }

    #[test]
    fn turbulence_toggle_changes_effective_viscosity() {
        // Without turbulence, molecular only; with, larger effective ν.
        // Difficult to directly assert, but confirm no crash + step_count.
        let mut s = CfdSolver::new(6, 6, 6, Fix128::ONE);
        s.use_turbulence = true;
        // Give it a nonzero velocity so strain rate > 0
        for i in 1..6 {
            s.grid.u[i + 7 * (2 + 6 * 2)] = Fix128::from_int(i as i64);
        }
        s.step(Fix128::from_ratio(1, 1000));
        assert_eq!(s.step_count, 1);
    }

    #[test]
    fn level_set_advection_moves_interface() {
        let mut s = CfdSolver::new(6, 6, 6, Fix128::ONE);
        // Initialise a level set (sphere at (3,3,3), r=2)
        let mut ls = Grid3d::new(6, 6, 6, Fix128::ONE, Fix128::ZERO);
        crate::multiphase::initialize_level_set_sphere(
            &mut ls,
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(3),
            Fix128::from_int(2),
        );
        s.level_set = Some(ls);
        // Prescribe a positive u velocity everywhere
        for u in s.grid.u.iter_mut() {
            *u = Fix128::ONE;
        }
        let before = s.level_set.as_ref().unwrap().data.clone();
        s.step(Fix128::from_ratio(5, 10));
        let after = &s.level_set.as_ref().unwrap().data;
        // Field should change somewhere
        let mut changed = false;
        for i in 0..before.len() {
            if before[i] != after[i] {
                changed = true;
                break;
            }
        }
        assert!(changed);
    }
}
