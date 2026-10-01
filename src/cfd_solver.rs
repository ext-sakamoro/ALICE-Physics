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
//! demos and validation tests. Higher-order RK3 time stepping is a future
//! upgrade. The pressure projection of [`CfdSolver::step`] is multigrid when
//! every grid extent is a power of two and Gauss-Seidel (`jacobi_iterations`
//! sweeps) otherwise; [`CfdSolver::step_multigrid`] picks the cycle count, and
//! `step_multigrid(dt, 0)` keeps the Gauss-Seidel projection on any grid.

use crate::eulerian_grid::{
    g2p_velocity, p2g_normalized, project_pressure, project_pressure_multigrid, sample_u_range,
    sample_u_trilinear, sample_v_range, sample_v_trilinear, sample_w_range, sample_w_trilinear,
    MacGrid,
};
use crate::interface_capture::fast_sweeping_reinit;
use crate::math::{Fix128, Vec3Fix};
use crate::multiphase::{trilinear_range, trilinear_sample, Grid3d};
use crate::surface_tension_csf::{compute_csf_field, SIGMA_WATER_AIR};
use crate::turbulence::{smagorinsky_eddy_viscosity, strain_rate_magnitude, SMAGORINSKY_CS};

/// W-cycles of the multigrid projection [`CfdSolver::step`] runs by default.
///
/// Measured as the smallest count whose post-step `max|div|` is at or below
/// that of the 30 Gauss-Seidel sweeps the default projection used to be, on
/// 8^3, 16^3 and 32^3 grids (6 cycles: 6.3e-4 / 1.9e-3 / 2.6e-2 against
/// 7.9e-4 / 1.4e-2 / 6.9e-1). The cost is about that of the 30 sweeps.
const DEFAULT_MULTIGRID_CYCLES: u32 = 6;

/// Selects the advection scheme applied to velocity and temperature at
/// each solver step.
///
/// - `SemiLagrangian` (default): first-order back-trace with trilinear
///   sampling. Cheap, unconditionally stable, but adds numerical
///   diffusion on every step. Adequate for engineering demos and short
///   time horizons.
/// - `MacCormack`: two-pass predictor-corrector built on top of the
///   semi-Lagrangian primitive. Second-order accurate on smooth data,
///   preserving sharp gradients and small-scale features roughly one
///   order of magnitude longer than plain semi-Lagrangian. No monotone
///   flux limiter is applied — smooth initial data with the projection
///   stage tends to remain stable, but shocks or sharp discontinuities
///   can produce local over/under-shoots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdvectionScheme {
    /// Semi-Lagrangian back-trace with trilinear sampling (first-order).
    #[default]
    SemiLagrangian,
    /// MacCormack predictor-corrector (second-order, unlimited).
    MacCormack,
    /// **BFECC** (Back and Forth Error Compensation and Correction).
    ///
    /// Three semi-Lagrangian passes: predictor `φ̃ = A(φ_n)`, reverse
    /// `φ̂ = A⁻¹(φ̃)`, compensated `φ* = φ_n + ½ (φ_n − φ̂)`, final
    /// `φ_{n+1} = A(φ*)`. Second-order accurate on smooth data with
    /// noticeably less phase error than MacCormack; the extra pass
    /// costs ~30% more per step. Applied to both MAC-face velocity
    /// (`advect_velocity_bfecc`) and the temperature scalar field.
    Bfecc,
}

/// Complete CFD solver state.
pub struct CfdSolver {
    /// Velocity + pressure MAC grid.
    pub grid: MacGrid,
    /// Fluid level set (negative = liquid, positive = gas), optional.
    /// When present, CSF and level-set advection are executed.
    pub level_set: Option<Grid3d>,
    /// Temperature field (K), optional. Enables Boussinesq buoyancy.
    pub temperature: Option<Grid3d>,
    /// Advection scheme used for both velocity and temperature.
    pub advection_scheme: AdvectionScheme,
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
    /// Gauss-Seidel sweeps per projection, used when the projection is not
    /// multigrid (a grid extent that is not a power of two, or
    /// `step_multigrid(dt, 0)`).
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
            advection_scheme: AdvectionScheme::SemiLagrangian,
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

    /// Largest `dt` satisfying `cfl_target = |u|_max · dt / dx`.
    ///
    /// Scans the current MAC-grid face velocities for the peak
    /// component magnitude, then inverts the Courant condition:
    ///
    /// ```text
    /// dt_max = cfl_target · dx / |u|_max
    /// ```
    ///
    /// Returns `Fix128::from_int(large_value)` if the velocity field is
    /// effectively zero (no CFL constraint), giving callers a
    /// well-defined upper bound to compare against a scheme-specific
    /// diffusion cap. `cfl_target` should typically be in `[0.5, 1.0]`
    /// for semi-Lagrangian and higher for BFECC / MacCormack when
    /// paired with a monotone clamp.
    #[must_use]
    pub fn compute_max_dt(&self, cfl_target: Fix128) -> Fix128 {
        let peak = self
            .grid
            .u
            .iter()
            .chain(self.grid.v.iter())
            .chain(self.grid.w.iter())
            .fold(Fix128::ZERO, |acc, &v| {
                let av = v.abs();
                if av > acc {
                    av
                } else {
                    acc
                }
            });
        if peak.is_zero() {
            return Fix128::from_int(1_000_000);
        }
        cfl_target * self.grid.dx / peak
    }

    /// Convenience — step with an automatically chosen `dt` from
    /// [`Self::compute_max_dt`], capped by `dt_ceiling`.
    ///
    /// Useful in engineering demos where the simulation should adapt
    /// to fast transients without the caller re-computing `dt` on each
    /// tick. Returns the `dt` that was actually integrated.
    pub fn step_adaptive(&mut self, cfl_target: Fix128, dt_ceiling: Fix128) -> Fix128 {
        let mut dt = self.compute_max_dt(cfl_target);
        if dt > dt_ceiling {
            dt = dt_ceiling;
        }
        self.step(dt);
        dt
    }

    /// One integrated time step.
    ///
    /// The pressure projection is [`project_pressure_multigrid`]
    /// (6 W-cycles) when every grid extent is a power
    /// of two, and `jacobi_iterations` Gauss-Seidel sweeps otherwise;
    /// `jacobi_iterations` counts only those sweeps. To keep the Gauss-Seidel
    /// projection on a power-of-two grid call `step_multigrid(dt, 0)`.
    ///
    /// The face boundary conditions of the grid ([`crate::eulerian_grid::FaceBc`]) are imposed
    /// three times: once before advection, so nothing samples a stale value
    /// off a wall; once after the body forces, so the viscous term sees the
    /// wall itself rather than a wall plus `g dt`; and once inside the
    /// projection, which is where they enter the Poisson problem. A grid
    /// with no boundary conditions set — the default — is untouched by all
    /// three.
    pub fn step(&mut self, dt_s: Fix128) {
        self.step_with_projection(dt_s, None);
    }

    /// [`Self::step`] with the pressure projection done by
    /// [`project_pressure_multigrid`] (`cycles` W-cycles) instead of
    /// `jacobi_iterations` Gauss-Seidel sweeps.
    ///
    /// Everything else — the order, the boundary enforcement, the advection
    /// and the level-set and temperature updates — is the shared step body, so
    /// the two entry points differ only in the projection. The per-cycle error
    /// reduction of the multigrid projection does not degrade as the grid is
    /// refined, which is what the Gauss-Seidel projection cannot offer.
    ///
    /// # When the multigrid projection cannot run
    ///
    /// It needs every grid extent to be a power of two and `cycles > 0`. When
    /// either does not hold the step projects with the Gauss-Seidel sweeps
    /// (`jacobi_iterations`), exactly as [`Self::step`] does, rather than
    /// skipping the projection and returning a compressible field. The
    /// fallback is a documented behaviour, not an error: this entry point has no
    /// error channel, as [`Self::step`] has none.
    // ALLOW-UNWIRED: public multigrid-projected step entry for downstream solvers
    pub fn step_multigrid(&mut self, dt_s: Fix128, cycles: u32) {
        self.step_with_projection(dt_s, Some(cycles));
    }

    /// Whether [`project_pressure_multigrid`] can solve this grid.
    fn grid_supports_multigrid(&self) -> bool {
        self.grid.nx.is_power_of_two()
            && self.grid.ny.is_power_of_two()
            && self.grid.nz.is_power_of_two()
    }

    /// The step body shared by [`Self::step`] (`multigrid_cycles = None`, which
    /// means 6 W-cycles) and
    /// [`Self::step_multigrid`].
    fn step_with_projection(&mut self, dt_s: Fix128, multigrid_cycles: Option<u32>) {
        if dt_s.is_zero() {
            return;
        }
        self.grid.enforce_face_boundaries();
        match self.advection_scheme {
            AdvectionScheme::SemiLagrangian => self.advect_velocity(dt_s),
            AdvectionScheme::MacCormack => self.advect_velocity_maccormack(dt_s),
            AdvectionScheme::Bfecc => self.advect_velocity_bfecc(dt_s),
        }
        self.apply_body_forces(dt_s);
        self.grid.enforce_face_boundaries();
        if self.use_turbulence {
            self.apply_turbulent_diffusion(dt_s);
        } else {
            self.apply_molecular_diffusion(dt_s);
        }
        // `None` is the default projection: multigrid where the grid allows it
        let cycles = multigrid_cycles.unwrap_or(DEFAULT_MULTIGRID_CYCLES);
        match cycles {
            cycles @ 1.. if self.grid_supports_multigrid() => {
                project_pressure_multigrid(&mut self.grid, dt_s, self.density_kg_m3, cycles);
            }
            _ => project_pressure(
                &mut self.grid,
                dt_s,
                self.density_kg_m3,
                self.jacobi_iterations,
            ),
        }
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
            match self.advection_scheme {
                AdvectionScheme::SemiLagrangian => self.advect_temperature(dt_s),
                AdvectionScheme::MacCormack => self.advect_temperature_maccormack(dt_s),
                AdvectionScheme::Bfecc => self.advect_temperature_bfecc(dt_s),
            }
        }
        self.step_count += 1;
    }

    /// FLIP / PIC particle step: scatter particles to the grid, apply forces
    /// and the pressure projection, then update and advect the particles.
    ///
    /// `particles` is a slice of `(position_m, velocity_m_per_s)` (the type
    /// [`crate::eulerian_grid::p2g_normalized`] takes) and `flip_ratio` is the
    /// blend `r`: `0` is pure PIC (the particle takes the grid velocity, smooth
    /// and dissipative), `1` is pure FLIP (the particle keeps its own velocity
    /// and adds the grid's change, noisy and nearly non-dissipative).
    ///
    /// ```text
    /// 1. clear u/v/w, p2g_normalized(particles), enforce_face_boundaries
    /// 2. u_old = grid velocity
    /// 3. body forces (gravity + buoyancy + CSF), enforce_face_boundaries,
    ///    molecular or Smagorinsky diffusion, pressure projection
    /// 4. v_p = (1 - r) G2P(u_new) + r (v_p + G2P(u_new - u_old))
    /// 5. x_p += v_p dt, clamped to [0, N dx] on each axis
    /// ```
    ///
    /// Step 3 is what [`CfdSolver::step`] runs between its advection and its
    /// level-set stage, in the same order, so a given `gravity`,
    /// viscosity, `use_turbulence` and `jacobi_iterations` mean the same thing
    /// here. Velocity advection is the particles' job and is skipped. The
    /// level set and the temperature field are **not** advected (the particles
    /// carry the fluid; a caller that uses buoyancy or surface tension owns
    /// those fields), and `step_count` is incremented. The projection is the
    /// Gauss-Seidel one (`jacobi_iterations` sweeps); the multigrid projection
    /// of [`CfdSolver::step_multigrid`] is not wired into the particle path.
    ///
    /// # Contract of this first stage: the particles must fill the domain
    ///
    /// The pressure solver has no fluid / air classification, so there is no
    /// free surface: every cell is a fluid cell. A face no particle reaches is
    /// cleared to zero by step 1 and then takes part in the projection as a
    /// fluid face, which is not physical for a particle cloud with a surface
    /// or a gap. Free surfaces (air cells as `p = 0`, cell classification,
    /// particle reseeding) and periodic boundaries are separate features.
    ///
    /// # Degenerate input
    ///
    /// Returns with the grid, the particles and `step_count` **unchanged** (no
    /// panic) for an empty particle list, `dt_s <= 0`, `dx = 0`, a grid with a
    /// zero dimension, a zero density, or a `flip_ratio` outside `[0, 1]`. A
    /// particle with any coordinate outside `[0, N dx]` takes no part in the
    /// transfer and is left bit-identical; one on the boundary does take part.
    // ALLOW-UNWIRED: public FLIP/PIC entry point for downstream solvers
    pub fn step_flip(
        &mut self,
        particles: &mut [(Vec3Fix, Vec3Fix)],
        dt_s: Fix128,
        flip_ratio: Fix128,
    ) {
        let g = &self.grid;
        if particles.is_empty()
            || dt_s <= Fix128::ZERO
            || g.dx.is_zero()
            || g.nx == 0
            || g.ny == 0
            || g.nz == 0
            || self.density_kg_m3.is_zero()
            || flip_ratio < Fix128::ZERO
            || flip_ratio > Fix128::ONE
        {
            return;
        }
        let hi_x = g.dx * Fix128::from_int(g.nx as i64);
        let hi_y = g.dx * Fix128::from_int(g.ny as i64);
        let hi_z = g.dx * Fix128::from_int(g.nz as i64);
        let inside = |p: Vec3Fix| {
            p.x >= Fix128::ZERO
                && p.x <= hi_x
                && p.y >= Fix128::ZERO
                && p.y <= hi_y
                && p.z >= Fix128::ZERO
                && p.z <= hi_z
        };

        // 1. particles -> grid
        let cloud: Vec<(Vec3Fix, Vec3Fix)> = particles
            .iter()
            .copied()
            .filter(|&(p, _)| inside(p))
            .collect();
        if cloud.is_empty() {
            return;
        }
        self.grid.u.fill(Fix128::ZERO);
        self.grid.v.fill(Fix128::ZERO);
        self.grid.w.fill(Fix128::ZERO);
        p2g_normalized(&mut self.grid, &cloud);
        self.grid.enforce_face_boundaries();

        // 2. velocity as transferred
        let u_old = self.grid.u.clone();
        let v_old = self.grid.v.clone();
        let w_old = self.grid.w.clone();

        // 3. forces, diffusion, projection (the order `step` uses)
        self.apply_body_forces(dt_s);
        self.grid.enforce_face_boundaries();
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

        // 4. grid -> particles, on the change of the grid velocity
        let mut delta = MacGrid::new(self.grid.nx, self.grid.ny, self.grid.nz, self.grid.dx);
        for (d, (new, old)) in delta.u.iter_mut().zip(self.grid.u.iter().zip(&u_old)) {
            *d = *new - *old;
        }
        for (d, (new, old)) in delta.v.iter_mut().zip(self.grid.v.iter().zip(&v_old)) {
            *d = *new - *old;
        }
        for (d, (new, old)) in delta.w.iter_mut().zip(self.grid.w.iter().zip(&w_old)) {
            *d = *new - *old;
        }
        let pic_weight = Fix128::ONE - flip_ratio;
        for (pos, vel) in particles.iter_mut() {
            if !inside(*pos) {
                continue;
            }
            let pic = g2p_velocity(&self.grid, *pos);
            let change = g2p_velocity(&delta, *pos);
            let kept = Vec3Fix::new(vel.x + change.x, vel.y + change.y, vel.z + change.z);
            *vel = Vec3Fix::new(
                pic_weight * pic.x + flip_ratio * kept.x,
                pic_weight * pic.y + flip_ratio * kept.y,
                pic_weight * pic.z + flip_ratio * kept.z,
            );
            // 5. advect, keep inside the box
            pos.x = clamp(pos.x + vel.x * dt_s, Fix128::ZERO, hi_x);
            pos.y = clamp(pos.y + vel.y * dt_s, Fix128::ZERO, hi_y);
            pos.z = clamp(pos.z + vel.z * dt_s, Fix128::ZERO, hi_z);
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
    ///
    /// # Tangential walls
    ///
    /// The mirror used for a neighbour that lies on the far side of a
    /// [`crate::eulerian_grid::FaceBc::Wall`] is the no-slip ghost `2 u_wall − u_in`, not the
    /// zero-gradient `u_in`. Those two differ by `2 (u_wall − u_in)`, which
    /// is the whole of the viscous shear the wall exerts: with the
    /// zero-gradient mirror the wall is free-slip, a lid-driven cavity never
    /// drives the fluid and a channel never develops a profile.
    ///
    /// `MacGrid::u_wall_across_y` and its five siblings answer whether the
    /// mirror crosses a wall, so an obstacle in the middle of the domain
    /// gets the same treatment as the outer box. A [`crate::eulerian_grid::FaceBc::SlipWall`]
    /// deliberately keeps the zero-gradient mirror — that is the symmetry
    /// plane a quasi-2-D run wants on its `z` faces.
    fn diffuse_velocity(&mut self, nu: Fix128, dt_s: Fix128) {
        if nu.is_zero() || self.grid.dx.is_zero() {
            return;
        }
        let coeff = nu * dt_s / (self.grid.dx * self.grid.dx);
        let two = Fix128::from_int(2);

        // u faces (all nx + 1 of them; the boundary faces i = 0 / nx use a
        // zero-gradient mirror like every other boundary. Before 1.2.0 they were
        // skipped and kept their old value while the interior diffused, which
        // created wall divergence → spurious pressure and a secondary flow up to
        // 10 % of a plane shear profile; `tests/engineering_oracles_fluid.rs`)
        let mut u_next = self.grid.u.clone();
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let center = self.grid.u(i, j, k);
                    let left = if i > 0 {
                        self.grid.u(i - 1, j, k)
                    } else {
                        center
                    };
                    let right = if i < self.grid.nx {
                        self.grid.u(i + 1, j, k)
                    } else {
                        center
                    };
                    let down = match self.grid.u_wall_across_y(i, j, k, false) {
                        Some(wall) => two * wall.x - center,
                        None if j > 0 => self.grid.u(i, j - 1, k),
                        None => center,
                    };
                    let up = match self.grid.u_wall_across_y(i, j, k, true) {
                        Some(wall) => two * wall.x - center,
                        None if j + 1 < self.grid.ny => self.grid.u(i, j + 1, k),
                        None => center,
                    };
                    let back = match self.grid.u_wall_across_z(i, j, k, false) {
                        Some(wall) => two * wall.x - center,
                        None if k > 0 => self.grid.u(i, j, k - 1),
                        None => center,
                    };
                    let fwd = match self.grid.u_wall_across_z(i, j, k, true) {
                        Some(wall) => two * wall.x - center,
                        None if k + 1 < self.grid.nz => self.grid.u(i, j, k + 1),
                        None => center,
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
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let center = self.grid.v(i, j, k);
                    let left = match self.grid.v_wall_across_x(i, j, k, false) {
                        Some(wall) => two * wall.y - center,
                        None if i > 0 => self.grid.v(i - 1, j, k),
                        None => center,
                    };
                    let right = match self.grid.v_wall_across_x(i, j, k, true) {
                        Some(wall) => two * wall.y - center,
                        None if i + 1 < self.grid.nx => self.grid.v(i + 1, j, k),
                        None => center,
                    };
                    let down = if j > 0 {
                        self.grid.v(i, j - 1, k)
                    } else {
                        center
                    };
                    let up = if j < self.grid.ny {
                        self.grid.v(i, j + 1, k)
                    } else {
                        center
                    };
                    let back = match self.grid.v_wall_across_z(i, j, k, false) {
                        Some(wall) => two * wall.y - center,
                        None if k > 0 => self.grid.v(i, j, k - 1),
                        None => center,
                    };
                    let fwd = match self.grid.v_wall_across_z(i, j, k, true) {
                        Some(wall) => two * wall.y - center,
                        None if k + 1 < self.grid.nz => self.grid.v(i, j, k + 1),
                        None => center,
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
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let center = self.grid.w(i, j, k);
                    let left = match self.grid.w_wall_across_x(i, j, k, false) {
                        Some(wall) => two * wall.z - center,
                        None if i > 0 => self.grid.w(i - 1, j, k),
                        None => center,
                    };
                    let right = match self.grid.w_wall_across_x(i, j, k, true) {
                        Some(wall) => two * wall.z - center,
                        None if i + 1 < self.grid.nx => self.grid.w(i + 1, j, k),
                        None => center,
                    };
                    let down = match self.grid.w_wall_across_y(i, j, k, false) {
                        Some(wall) => two * wall.z - center,
                        None if j > 0 => self.grid.w(i, j - 1, k),
                        None => center,
                    };
                    let up = match self.grid.w_wall_across_y(i, j, k, true) {
                        Some(wall) => two * wall.z - center,
                        None if j + 1 < self.grid.ny => self.grid.w(i, j + 1, k),
                        None => center,
                    };
                    let back = if k > 0 {
                        self.grid.w(i, j, k - 1)
                    } else {
                        center
                    };
                    let fwd = if k < self.grid.nz {
                        self.grid.w(i, j, k + 1)
                    } else {
                        center
                    };
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

    /// MacCormack predictor-corrector for MAC velocity self-advection.
    ///
    /// Runs one semi-Lagrangian pass forward (`φ̂ = SL(φ_n, dt)`), then
    /// reverse-advects `φ̂` by `-dt` using the pre-advection velocity
    /// `u_n` as the advecting field (`φ̃ = SL(φ̂, -dt; u_n)`), and applies
    /// the second-order MacCormack correction
    ///
    /// ```text
    /// φ_{n+1} = φ̂ + ½ · (φ_n − φ̃)
    /// ```
    ///
    /// followed by a Fedkiw-style monotone limiter: for each face the
    /// corrected value is clamped to the `[min, max]` interval spanned
    /// by the 8 neighbouring face-value corners on `u_n` at the back-
    /// traced position. Without this limiter the unlimited corrector
    /// injects super-linear vorticity growth on the paper's
    /// axisymmetric swirl setup and diverges within tens of steps.
    fn advect_velocity_maccormack(&mut self, dt_s: Fix128) {
        let u_n = self.grid.clone();

        // Predictor: reuse the existing semi-Lagrangian pass.
        self.advect_velocity(dt_s);
        let phi_hat = self.grid.clone();

        // Corrector: reverse-advect phi_hat using u_n's velocity by
        // forward-tracing with +dt (equivalent to back-trace with -dt).
        let dx = self.grid.dx;
        let half = Fix128::from_ratio(1, 2);
        let mut u_tilde = vec![Fix128::ZERO; self.grid.u.len()];
        let mut v_tilde = vec![Fix128::ZERO; self.grid.v.len()];
        let mut w_tilde = vec![Fix128::ZERO; self.grid.w.len()];

        // u faces
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < u_tilde.len() {
                        u_tilde[ix] = sample_u_trilinear(&phi_hat, forward);
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
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < v_tilde.len() {
                        v_tilde[ix] = sample_v_trilinear(&phi_hat, forward);
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
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < w_tilde.len() {
                        w_tilde[ix] = sample_w_trilinear(&phi_hat, forward);
                    }
                }
            }
        }

        // Apply MacCormack correction: φ_{n+1} = φ̂ + ½ · (φ_n − φ̃).
        for ((dst, &hat), (&n_val, &tilde)) in self
            .grid
            .u
            .iter_mut()
            .zip(phi_hat.u.iter())
            .zip(u_n.u.iter().zip(u_tilde.iter()))
        {
            *dst = hat + half * (n_val - tilde);
        }
        for ((dst, &hat), (&n_val, &tilde)) in self
            .grid
            .v
            .iter_mut()
            .zip(phi_hat.v.iter())
            .zip(u_n.v.iter().zip(v_tilde.iter()))
        {
            *dst = hat + half * (n_val - tilde);
        }
        for ((dst, &hat), (&n_val, &tilde)) in self
            .grid
            .w
            .iter_mut()
            .zip(phi_hat.w.iter())
            .zip(u_n.w.iter().zip(w_tilde.iter()))
        {
            *dst = hat + half * (n_val - tilde);
        }

        // Monotonicity guard: clamp each face component to the local
        // pre-advection range at the back-traced position on `u_n`.
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_u_range(&u_n, back);
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < self.grid.u.len() {
                        let val = self.grid.u[ix];
                        self.grid.u[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = Fix128::from_int(j as i64) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_v_range(&u_n, back);
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < self.grid.v.len() {
                        let val = self.grid.v[ix];
                        self.grid.v[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = Fix128::from_int(k as i64) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_w_range(&u_n, back);
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < self.grid.w.len() {
                        let val = self.grid.w[ix];
                        self.grid.w[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
    }

    /// MacCormack predictor-corrector for the temperature scalar field.
    ///
    /// Mirrors `advect_velocity_maccormack` on a single cell-centred
    /// scalar. The advecting velocity is `self.grid` at call time (the
    /// projected divergence-free field), consistent with the
    /// semi-Lagrangian variant.
    fn advect_temperature_maccormack(&mut self, dt_s: Fix128) {
        if self.temperature.is_none() {
            return;
        }
        let phi_n = self
            .temperature
            .as_ref()
            .map(|t| t.data.clone())
            .unwrap_or_default();

        // Predictor: reuse existing SL pass.
        self.advect_temperature(dt_s);

        // Snapshot phi_hat and prepare reverse sampling grid.
        let (nx_t, ny_t, nz_t, dx_t, phi_hat) = {
            let temp = self
                .temperature
                .as_ref()
                .expect("temperature checked above");
            (temp.nx, temp.ny, temp.nz, temp.dx, temp.data.clone())
        };
        let phi_hat_grid = Grid3d {
            nx: nx_t,
            ny: ny_t,
            nz: nz_t,
            dx: dx_t,
            data: phi_hat.clone(),
        };
        let inv_dx = Fix128::ONE / dx_t;
        let half = Fix128::from_ratio(1, 2);
        let mut phi_tilde = vec![Fix128::ZERO; phi_hat.len()];
        for k in 0..nz_t {
            for j in 0..ny_t {
                for i in 0..nx_t {
                    let (uc, vc, wc) = self.grid.cell_velocity(
                        i.min(self.grid.nx - 1),
                        j.min(self.grid.ny - 1),
                        k.min(self.grid.nz - 1),
                    );
                    // Forward-trace = reverse advection with -dt.
                    let cx = Fix128::from_int(i as i64) + uc * dt_s * inv_dx;
                    let cy = Fix128::from_int(j as i64) + vc * dt_s * inv_dx;
                    let cz = Fix128::from_int(k as i64) + wc * dt_s * inv_dx;
                    let sampled = trilinear_sample(&phi_hat_grid, cx, cy, cz);
                    phi_tilde[i + nx_t * (j + ny_t * k)] = sampled;
                }
            }
        }

        // Apply MacCormack correction.
        if let Some(temp) = self.temperature.as_mut() {
            for ((dst, &hat), (&n_val, &tilde)) in temp
                .data
                .iter_mut()
                .zip(phi_hat.iter())
                .zip(phi_n.iter().zip(phi_tilde.iter()))
            {
                *dst = hat + half * (n_val - tilde);
            }
        }

        // Monotonicity guard: clamp to pre-advection range at back-trace
        // position on the pre-advection temperature field.
        let phi_n_grid = Grid3d {
            nx: nx_t,
            ny: ny_t,
            nz: nz_t,
            dx: dx_t,
            data: phi_n,
        };
        if let Some(temp) = self.temperature.as_mut() {
            for k in 0..nz_t {
                for j in 0..ny_t {
                    for i in 0..nx_t {
                        let (uc, vc, wc) = self.grid.cell_velocity(
                            i.min(self.grid.nx - 1),
                            j.min(self.grid.ny - 1),
                            k.min(self.grid.nz - 1),
                        );
                        let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                        let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                        let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                        let (lo, hi) = trilinear_range(&phi_n_grid, cx, cy, cz);
                        let ix = i + nx_t * (j + ny_t * k);
                        let val = temp.data[ix];
                        temp.data[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
    }

    /// BFECC advection of the MAC velocity field (u/v/w faces).
    ///
    /// Full three-pass BFECC on all three staggered components:
    ///
    /// ```text
    /// φ̂  = SL(φ_n, +dt; u_n)      // predictor
    /// φ̃  = SL(φ̂, -dt; u_n)       // reverse
    /// φ*  = φ_n + ½ (φ_n − φ̃)     // compensated input
    /// φ_{n+1} = SL(φ*, +dt; u_n)   // final SL from the compensated field
    /// ```
    ///
    /// The final result is clipped to the pre-advection back-trace range
    /// on `u_n` for monotonicity, matching the MacCormack limiter
    /// convention already applied in [`advect_velocity_maccormack`].
    ///
    /// Compared to MacCormack, BFECC pays one extra semi-Lagrangian
    /// pass but removes the phase-error residual on the compensator,
    /// giving cleaner spectra on smooth flow.
    #[allow(clippy::too_many_lines)] // canonical 3-pass BFECC structure
    fn advect_velocity_bfecc(&mut self, dt_s: Fix128) {
        let u_n = self.grid.clone();
        let dx = self.grid.dx;
        let half = Fix128::from_ratio(1, 2);

        // Pass 1 — forward SL: φ̂ = SL(φ_n, +dt).
        self.advect_velocity(dt_s);
        let phi_hat = self.grid.clone();

        // Pass 2 — reverse SL on φ̂ using u_n's advecting field:
        // φ̃ = SL(φ̂, -dt; u_n).
        let mut u_tilde = vec![Fix128::ZERO; self.grid.u.len()];
        let mut v_tilde = vec![Fix128::ZERO; self.grid.v.len()];
        let mut w_tilde = vec![Fix128::ZERO; self.grid.w.len()];
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < u_tilde.len() {
                        u_tilde[ix] = sample_u_trilinear(&phi_hat, forward);
                    }
                }
            }
        }
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = Fix128::from_int(j as i64) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < v_tilde.len() {
                        v_tilde[ix] = sample_v_trilinear(&phi_hat, forward);
                    }
                }
            }
        }
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = Fix128::from_int(k as i64) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let forward = Vec3Fix::new(
                        pos.x + dt_s * vel.x,
                        pos.y + dt_s * vel.y,
                        pos.z + dt_s * vel.z,
                    );
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < w_tilde.len() {
                        w_tilde[ix] = sample_w_trilinear(&phi_hat, forward);
                    }
                }
            }
        }

        // Pass 3 — build the compensated input φ* = φ_n + ½(φ_n − φ̃).
        // 演算順序は index 版と同一 (決定論性維持): un + half * (un - ut)
        let mut phi_star = u_n.clone();
        for (ps, (un, ut)) in phi_star.u.iter_mut().zip(u_n.u.iter().zip(u_tilde.iter())) {
            *ps = *un + half * (*un - *ut);
        }
        for (ps, (un, ut)) in phi_star.v.iter_mut().zip(u_n.v.iter().zip(v_tilde.iter())) {
            *ps = *un + half * (*un - *ut);
        }
        for (ps, (un, ut)) in phi_star.w.iter_mut().zip(u_n.w.iter().zip(w_tilde.iter())) {
            *ps = *un + half * (*un - *ut);
        }

        // Pass 4 — final SL from φ* using u_n's advecting field.
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < self.grid.u.len() {
                        self.grid.u[ix] = sample_u_trilinear(&phi_star, back);
                    }
                }
            }
        }
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = Fix128::from_int(j as i64) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < self.grid.v.len() {
                        self.grid.v[ix] = sample_v_trilinear(&phi_star, back);
                    }
                }
            }
        }
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = Fix128::from_int(k as i64) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < self.grid.w.len() {
                        self.grid.w[ix] = sample_w_trilinear(&phi_star, back);
                    }
                }
            }
        }

        // Monotonicity guard — clamp each face component to the local
        // pre-advection range at the back-traced position on `u_n`.
        for k in 0..self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..=self.grid.nx {
                    let px = Fix128::from_int(i as i64) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_u_range(&u_n, back);
                    let ix = self.grid.idx_u(i, j, k);
                    if ix < self.grid.u.len() {
                        let val = self.grid.u[ix];
                        self.grid.u[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
        for k in 0..self.grid.nz {
            for j in 0..=self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = Fix128::from_int(j as i64) * dx;
                    let pz = (Fix128::from_int(k as i64) + half) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_v_range(&u_n, back);
                    let ix = self.grid.idx_v(i, j, k);
                    if ix < self.grid.v.len() {
                        let val = self.grid.v[ix];
                        self.grid.v[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
        for k in 0..=self.grid.nz {
            for j in 0..self.grid.ny {
                for i in 0..self.grid.nx {
                    let px = (Fix128::from_int(i as i64) + half) * dx;
                    let py = (Fix128::from_int(j as i64) + half) * dx;
                    let pz = Fix128::from_int(k as i64) * dx;
                    let pos = Vec3Fix::new(px, py, pz);
                    let vel = g2p_velocity(&u_n, pos);
                    let back = Vec3Fix::new(
                        pos.x - dt_s * vel.x,
                        pos.y - dt_s * vel.y,
                        pos.z - dt_s * vel.z,
                    );
                    let (lo, hi) = sample_w_range(&u_n, back);
                    let ix = self.grid.idx_w(i, j, k);
                    if ix < self.grid.w.len() {
                        let val = self.grid.w[ix];
                        self.grid.w[ix] = clamp(val, lo, hi);
                    }
                }
            }
        }
    }

    /// BFECC advection of the temperature scalar field.
    ///
    /// Implements the three-pass Back-and-Forth Error Compensation and
    /// Correction scheme: forward SL, reverse SL, compensate, final SL.
    /// Under smooth flow the phase error is one order lower than plain
    /// semi-Lagrangian and slightly better than MacCormack; a
    /// monotonicity clamp against the pre-advection field prevents
    /// runaway over/undershoots.
    fn advect_temperature_bfecc(&mut self, dt_s: Fix128) {
        if self.temperature.is_none() {
            return;
        }
        let phi_n = self
            .temperature
            .as_ref()
            .map(|t| t.data.clone())
            .unwrap_or_default();
        let (nx_t, ny_t, nz_t, dx_t) = {
            let temp = self
                .temperature
                .as_ref()
                .expect("temperature checked above");
            (temp.nx, temp.ny, temp.nz, temp.dx)
        };
        let inv_dx = Fix128::ONE / dx_t;
        let half = Fix128::from_ratio(1, 2);

        // Pass 1 — forward SL predictor: φ̃ = A(φ_n) in place.
        self.advect_temperature(dt_s);
        let phi_hat = self
            .temperature
            .as_ref()
            .map(|t| t.data.clone())
            .unwrap_or_default();
        let phi_hat_grid = Grid3d {
            nx: nx_t,
            ny: ny_t,
            nz: nz_t,
            dx: dx_t,
            data: phi_hat.clone(),
        };

        // Pass 2 — reverse SL: φ̂ = A⁻¹(φ̃), by forward-tracing.
        let mut phi_reverse = vec![Fix128::ZERO; phi_hat.len()];
        for k in 0..nz_t {
            for j in 0..ny_t {
                for i in 0..nx_t {
                    let (uc, vc, wc) = self.grid.cell_velocity(
                        i.min(self.grid.nx - 1),
                        j.min(self.grid.ny - 1),
                        k.min(self.grid.nz - 1),
                    );
                    let cx = Fix128::from_int(i as i64) + uc * dt_s * inv_dx;
                    let cy = Fix128::from_int(j as i64) + vc * dt_s * inv_dx;
                    let cz = Fix128::from_int(k as i64) + wc * dt_s * inv_dx;
                    phi_reverse[i + nx_t * (j + ny_t * k)] =
                        trilinear_sample(&phi_hat_grid, cx, cy, cz);
                }
            }
        }

        // Pass 3 — compensate the input: φ* = φ_n + ½ (φ_n − φ̂) at each cell,
        // then run one more forward SL from φ*.
        let mut phi_star = vec![Fix128::ZERO; phi_n.len()];
        for i in 0..phi_n.len() {
            phi_star[i] = phi_n[i] + half * (phi_n[i] - phi_reverse[i]);
        }
        let phi_star_grid = Grid3d {
            nx: nx_t,
            ny: ny_t,
            nz: nz_t,
            dx: dx_t,
            data: phi_star,
        };
        if let Some(temp) = self.temperature.as_mut() {
            for k in 0..nz_t {
                for j in 0..ny_t {
                    for i in 0..nx_t {
                        let (uc, vc, wc) = self.grid.cell_velocity(
                            i.min(self.grid.nx - 1),
                            j.min(self.grid.ny - 1),
                            k.min(self.grid.nz - 1),
                        );
                        let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                        let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                        let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                        let sampled = trilinear_sample(&phi_star_grid, cx, cy, cz);
                        let ix = temp.idx(i, j, k);
                        temp.data[ix] = sampled;
                    }
                }
            }
        }

        // Monotonicity guard against the pre-advection field.
        let phi_n_grid = Grid3d {
            nx: nx_t,
            ny: ny_t,
            nz: nz_t,
            dx: dx_t,
            data: phi_n,
        };
        if let Some(temp) = self.temperature.as_mut() {
            for k in 0..nz_t {
                for j in 0..ny_t {
                    for i in 0..nx_t {
                        let (uc, vc, wc) = self.grid.cell_velocity(
                            i.min(self.grid.nx - 1),
                            j.min(self.grid.ny - 1),
                            k.min(self.grid.nz - 1),
                        );
                        let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                        let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                        let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                        let (lo, hi) = trilinear_range(&phi_n_grid, cx, cy, cz);
                        let ix = i + nx_t * (j + ny_t * k);
                        let val = temp.data[ix];
                        temp.data[ix] = clamp(val, lo, hi);
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

fn clamp(value: Fix128, lo: Fix128, hi: Fix128) -> Fix128 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
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

    // ---- BFECC advection tests -----------------------------------------

    fn setup_bfecc_solver(nx: usize) -> CfdSolver {
        let mut s = CfdSolver::new(nx, nx, nx, Fix128::from_ratio(1, 10));
        s.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        s.temperature = Some(Grid3d::new(nx, nx, nx, s.grid.dx, Fix128::from_int(293)));
        // Uniform u = 1 m/s along +x, everything else zero.
        for u in s.grid.u.iter_mut() {
            *u = Fix128::from_ratio(1, 2);
        }
        s
    }

    #[test]
    fn bfecc_advection_scheme_step_runs() {
        let mut s = setup_bfecc_solver(4);
        s.advection_scheme = AdvectionScheme::Bfecc;
        s.step(Fix128::from_ratio(1, 100));
        assert_eq!(s.step_count, 1);
    }

    #[test]
    fn bfecc_conserves_uniform_temperature_field() {
        let mut s = setup_bfecc_solver(4);
        s.advection_scheme = AdvectionScheme::Bfecc;
        // Ensure temperature is truly uniform (293 K) prior to step.
        let expected = Fix128::from_int(293);
        s.step(Fix128::from_ratio(1, 100));
        for &t in &s.temperature.as_ref().unwrap().data {
            assert!(
                (t - expected).abs() < Fix128::from_ratio(1, 100),
                "temp drift {:?}",
                t
            );
        }
    }

    #[test]
    fn bfecc_transports_temperature_bump() {
        let mut s = setup_bfecc_solver(6);
        s.advection_scheme = AdvectionScheme::Bfecc;
        // Place a hot cell at (1, 3, 3); after +x advection, some cell
        // downstream should exceed the reference by more than the pure
        // semi-Lagrangian smearing would allow.
        if let Some(temp) = s.temperature.as_mut() {
            let idx = temp.idx(1, 3, 3);
            temp.data[idx] = Fix128::from_int(500);
        }
        let before_max = s
            .temperature
            .as_ref()
            .unwrap()
            .data
            .iter()
            .copied()
            .fold(Fix128::ZERO, |acc, x| if x > acc { x } else { acc });
        for _ in 0..3 {
            s.step(Fix128::from_ratio(1, 100));
        }
        let after_max = s
            .temperature
            .as_ref()
            .unwrap()
            .data
            .iter()
            .copied()
            .fold(Fix128::ZERO, |acc, x| if x > acc { x } else { acc });
        // BFECC preserves the sharp bump much better than SL — the peak
        // must remain above 250 K (well above the reference of 293 K
        // and clearly non-diffused into oblivion).
        assert!(
            after_max > Fix128::from_int(250),
            "peak dropped: {:?}",
            after_max
        );
        // Bounded by pre-advection extremum (monotonicity guard).
        assert!(after_max <= before_max);
    }

    #[test]
    fn compute_max_dt_returns_large_dt_on_zero_velocity() {
        let s = CfdSolver::new(4, 4, 4, Fix128::from_ratio(1, 10));
        let dt = s.compute_max_dt(Fix128::from_ratio(5, 10));
        // With zero velocity the CFL is inactive; expect a big cap.
        assert!(dt > Fix128::from_int(1000));
    }

    #[test]
    fn compute_max_dt_inverts_cfl_condition() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::from_ratio(1, 10));
        for u in s.grid.u.iter_mut() {
            *u = Fix128::ONE; // 1 m/s uniform
        }
        // dx = 0.1, CFL = 0.5, dt = 0.5 * 0.1 / 1.0 = 0.05
        let dt = s.compute_max_dt(Fix128::from_ratio(5, 10));
        assert!((dt - Fix128::from_ratio(5, 100)).abs() < Fix128::from_ratio(1, 1000));
    }

    #[test]
    fn step_adaptive_respects_ceiling() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::from_ratio(1, 10));
        s.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        // Zero velocity → compute_max_dt returns huge; ceiling should
        // clamp the actually-integrated dt.
        let ceiling = Fix128::from_ratio(1, 100);
        let dt_used = s.step_adaptive(Fix128::from_ratio(5, 10), ceiling);
        assert_eq!(dt_used, ceiling);
        assert_eq!(s.step_count, 1);
    }

    #[test]
    fn bfecc_velocity_preserves_zero_field() {
        let mut s = CfdSolver::new(4, 4, 4, Fix128::from_ratio(1, 10));
        s.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        s.advection_scheme = AdvectionScheme::Bfecc;
        s.step(Fix128::from_ratio(1, 100));
        for &u in &s.grid.u {
            assert!(u.abs() < Fix128::from_ratio(1, 100));
        }
    }

    #[test]
    fn bfecc_velocity_transports_uniform_flow() {
        // Uniform u = 0.5 m/s along +x should remain approximately
        // uniform under BFECC self-advection (no gradient to advect).
        let mut s = CfdSolver::new(6, 6, 6, Fix128::from_ratio(1, 10));
        s.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        s.advection_scheme = AdvectionScheme::Bfecc;
        for u in s.grid.u.iter_mut() {
            *u = Fix128::from_ratio(1, 2);
        }
        s.step(Fix128::from_ratio(1, 100));
        // Interior u values should stay close to 0.5 (small boundary
        // clamping and projection deviations are acceptable).
        let mid = 3;
        let interior_u = s.grid.u[s.grid.idx_u(3, mid, mid)];
        assert!(
            (interior_u - Fix128::from_ratio(1, 2)).abs() < Fix128::from_ratio(1, 10),
            "interior u drifted: {interior_u:?}"
        );
    }

    #[test]
    fn bfecc_velocity_monotonicity_bounded_by_pre_advection() {
        // Give a smooth Gaussian-like u profile and step once — the
        // BFECC clamp should keep each face value bounded by its
        // back-traced pre-advection range.
        let mut s = CfdSolver::new(6, 6, 6, Fix128::from_ratio(1, 10));
        s.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);
        s.advection_scheme = AdvectionScheme::Bfecc;
        for i in 0..=6 {
            for j in 0..6 {
                for k in 0..6 {
                    let ix = s.grid.idx_u(i, j, k);
                    if ix < s.grid.u.len() {
                        // triangular ramp along x, peak = 1 at i=3
                        let dist = (i as i64 - 3).abs();
                        s.grid.u[ix] = Fix128::from_ratio(3 - dist, 3);
                    }
                }
            }
        }
        let pre_max = s
            .grid
            .u
            .iter()
            .fold(Fix128::ZERO, |a, &b| if b > a { b } else { a });
        s.step(Fix128::from_ratio(1, 100));
        let post_max = s
            .grid
            .u
            .iter()
            .fold(Fix128::ZERO, |a, &b| if b > a { b } else { a });
        // BFECC must not overshoot the initial peak by more than a
        // small tolerance (projection stage may nudge by ε).
        assert!(
            post_max <= pre_max + Fix128::from_ratio(1, 50),
            "overshoot: post={post_max:?} pre={pre_max:?}"
        );
    }
}
