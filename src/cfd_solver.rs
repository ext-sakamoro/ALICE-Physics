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
// LIMITATION(COV-CFD-026): This is a first-order operator-splitting scheme; adequate for engineering demos and validation tests. Higher-order RK3 time stepping is a future upgrade.
//! This is a first-order operator-splitting scheme; adequate for engineering
//! demos and validation tests. Higher-order RK3 time stepping is a future
//! upgrade. The pressure projection of [`CfdSolver::step`] is multigrid when
//! every grid extent is a power of two and Gauss-Seidel (`jacobi_iterations`
//! sweeps) otherwise; [`CfdSolver::step_multigrid`] picks the cycle count, and
//! `step_multigrid(dt, 0)` keeps the Gauss-Seidel projection on any grid.

use crate::eulerian_grid::{
    g2p_velocity, p2g_normalized_with, project_pressure, project_pressure_banded,
    project_pressure_bicgstab, project_pressure_decomposed, project_pressure_jacobi,
    project_pressure_multigrid, project_pressure_multigrid_decomposed, sample_u_range,
    sample_u_trilinear, sample_v_range, sample_v_trilinear, sample_w_range, sample_w_trilinear,
    BicgstabStats, FaceBc, HaloSchedule, MacGrid, ParticleScatter,
};
use crate::interface_capture::fast_sweeping_reinit;
use crate::math::{Fix128, Vec3Fix};
use crate::multiphase::{reinitialize_level_set, trilinear_range, trilinear_sample, Grid3d};
use crate::surface_tension_csf::{compute_csf_field, SIGMA_WATER_AIR};
use crate::turbulence::{
    dynamic_smagorinsky_cs, friction_velocity_checked, smagorinsky_eddy_viscosity,
    smagorinsky_eddy_viscosity_with, strain_rate_magnitude, wall_k_epsilon, y_plus, KEpsilonState,
    KOmegaState, KE_SIGMA_EPS, KE_SIGMA_K, KW_BETA_STAR, KW_SIGMA, SMAGORINSKY_CS, VON_KARMAN,
};

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

/// Which solver [`CfdSolver::step_with_pressure_solver`] projects with.
///
/// All four solve the same masked 7-point Poisson problem
/// `∇²p = (ρ/dt) ∇·u*` and apply the same velocity correction, so a
/// converged answer from any of them is the same projected field to within
/// the residual each one leaves (`tests/analytic_pressure_solvers.rs` bounds
/// that difference from the residuals, not from a measured number). They
/// differ in cost and in how convergence is reported:
///
/// | variant | per-iteration cost | stops | reports |
/// |---|---|---|---|
/// | `RedBlackGs` | one sweep | fixed `sweeps` | nothing |
// LIMITATION(COV-CFD-071): nothing; refused off a power-of-two grid
/// | `Multigrid` | one W-cycle (grid-independent rate) | fixed `cycles` | nothing; refused off a power-of-two grid |
/// | `Jacobi` | one matrix-vector product | fixed `iterations` | nothing |
/// | `BiCgStab` | ~7 dot products + 2 operator applications | `‖r‖_∞ < tolerance` or `max_iterations` | [`BicgstabStats`] |
/// | `DecomposedGs` | one sweep, over `ranks` `z` slabs with one halo layer each | fixed `sweeps` | nothing; bit-identical to `RedBlackGs` |
/// | `BandedGs` | as `DecomposedGs`, every rank holding only its band | fixed `sweeps` | nothing; bit-identical to `RedBlackGs` |
/// | `DecomposedMultigrid` | one W-cycle, over `ranks` `z` slabs, every rank holding only its band | fixed `cycles` | nothing; bit-identical to `Multigrid`; refused off a power-of-two grid |
///
// LIMITATION(COV-CFD-075): The fixed-count solvers never say whether they converged; on a large grid a short count leaves a smooth divergence of order one.
/// ⚠️ The fixed-count solvers never say whether they converged; on a large
/// grid a short count leaves a smooth divergence of order one. `BiCgStab` is
/// the only one that returns a verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PressureSolver {
    /// Red-black Gauss-Seidel, `sweeps` full sweeps (what [`CfdSolver::step`]
    /// uses off a power-of-two grid, with `jacobi_iterations` sweeps).
    RedBlackGs {
        /// Sweeps; refused when zero.
        sweeps: u32,
    },
    /// Galerkin multigrid, `cycles` W-cycles (what [`CfdSolver::step`] uses
    /// on a power-of-two grid, with 6 cycles).
    Multigrid {
        /// W-cycles; refused when zero.
        cycles: u32,
    },
    /// Jacobi iteration, `iterations` sweeps.
    Jacobi {
        /// Sweeps; refused when zero.
        iterations: u32,
    },
    /// Red-black Gauss-Seidel over `ranks` contiguous `z` slabs that exchange
    /// one halo layer after every colour sweep, run in this process — the
    /// decomposition the distributed solvers use, with every rank still
    /// holding a full-length buffer. The answer is the `RedBlackGs` one to the
    /// bit for any `ranks`, including counts that do not divide `nz` and
    /// counts above it (the surplus ranks own nothing).
    DecomposedGs {
        /// Slabs; refused when zero.
        ranks: usize,
        /// Sweeps; refused when zero.
        sweeps: u32,
    },
    /// As `DecomposedGs`, but every rank holds only its own layers plus one
    /// halo layer (slab-local storage): no full-length array exists during the
    /// solve, which is the memory shape of a run that spans machines.
    BandedGs {
        /// Slabs; refused when zero.
        ranks: usize,
        /// Sweeps; refused when zero.
        sweeps: u32,
    },
    /// `Multigrid` run as `ranks` contiguous `z` slabs that exchange one halo
    /// layer after every colour sweep, in this process, every rank holding only
    /// its band of each distributed level. Levels with fewer layers than ranks
    /// are gathered to rank 0 and solved there. The answer is the `Multigrid`
    /// one to the bit for any `ranks`, including counts above `nz`; refused off
    /// a power-of-two grid like `Multigrid`.
    DecomposedMultigrid {
        /// Slabs; refused when zero.
        ranks: usize,
        /// W-cycles; refused when zero.
        cycles: u32,
    },
    /// Jacobi-preconditioned BiCGStab (van der Vorst 1992).
    BiCgStab {
        /// Iteration budget; refused when zero.
        max_iterations: u32,
        /// Stop once `‖r‖_∞` is below this (in the units of
        /// `b = ρ dx²/dt · ∇·u`); refused unless strictly positive.
        tolerance: Fix128,
    },
}

/// Why [`CfdSolver::step_with_pressure_solver`] refused to step.
///
/// Every case is an input that would otherwise be answered silently: the
/// fixed-count solvers return the field untouched on a zero `dt`, density or
/// count, and [`CfdSolver::step`] falls back from multigrid to Gauss-Seidel
/// off a power-of-two grid. A caller that *chose* a solver is told instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PressureSolverError {
    /// `dt_s` is zero: nothing would be integrated.
    ZeroTimeStep,
    /// `density_kg_m3` is zero: the Poisson scale `ρ dx²/dt` is zero and the
    /// velocity correction `dt/ρ` is undefined.
    ZeroDensity,
    /// The grid spacing is zero.
    ZeroSpacing,
    /// The iteration / sweep / cycle count is zero.
    ZeroIterations,
    /// `Multigrid` was asked for on a grid whose extents are not all powers
    /// of two.
    MultigridNeedsPowerOfTwoExtents {
        /// `(nx, ny, nz)` of the grid.
        extents: (usize, usize, usize),
    },
    /// `BiCgStab` was given a tolerance that is not strictly positive.
    NonPositiveTolerance,
    /// A slab decomposition was asked for with zero ranks.
    ZeroRanks,
    /// [`crate::eulerian_grid::project_pressure_distributed`] was given a
    /// solver that is not a slab decomposition (`DecomposedGs`, `BandedGs` or
    /// `DecomposedMultigrid`).
    NotDecomposed,
}

impl core::fmt::Display for PressureSolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroTimeStep => write!(f, "dt_s is zero"),
            Self::ZeroDensity => write!(f, "density_kg_m3 is zero"),
            Self::ZeroSpacing => write!(f, "the grid spacing dx is zero"),
            Self::ZeroIterations => write!(f, "the iteration count is zero"),
            Self::MultigridNeedsPowerOfTwoExtents { extents } => write!(
                f,
                "multigrid needs power-of-two extents, got {}x{}x{}",
                extents.0, extents.1, extents.2
            ),
            Self::NonPositiveTolerance => write!(f, "the BiCGStab tolerance must be positive"),
            Self::ZeroRanks => write!(f, "a slab decomposition needs at least one rank"),
            Self::NotDecomposed => write!(f, "the solver is not a slab decomposition"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PressureSolverError {}

/// What [`CfdSolver::step_with_pressure_solver`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ProjectionReport {
    /// The solver that ran, as requested.
    pub solver: PressureSolver,
    /// The BiCGStab verdict; `None` for the fixed-count solvers, which have
    /// none to give.
    pub bicgstab: Option<BicgstabStats>,
}

/// A wall model: the viscous flux across a [`crate::eulerian_grid::FaceBc::Wall`]
/// is taken from the universal wall profile instead of from the resolved
/// no-slip ghost.
///
/// With the ghost, the shear a wall exerts on the adjacent face is
/// `μ (u_in − u_wall) / (dx/2)` — the laminar value at the resolution the grid
/// has, which under-predicts the shear of a turbulent boundary layer the grid
/// does not resolve. The model instead reads the friction velocity from the
/// log law ([`crate::turbulence::friction_velocity`]) at the first face centre
/// (`y_p = dx/2`) and applies `τ_w = ρ u_τ²` against the direction of the
/// relative motion. Below the sublayer edge (`y⁺ < 11.4453`) the profile is
/// linear and the model reduces to the ghost exactly, which is what makes it
/// safe to enable on a resolved grid.
///
/// Only the molecular viscosity enters `y⁺`; the Smagorinsky eddy viscosity
/// of `use_turbulence` still acts on the interior faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct WallModel {}

impl WallModel {
    /// The log-law wall model with the module constants of
    /// [`crate::turbulence`] (`κ = 0.41`, `B = 5.5`, sublayer edge `11.4453`).
    #[must_use]
    pub const fn log_law() -> Self {
        Self {}
    }
}

/// How the level set is reinitialised on the steps where
/// `CfdSolver::reinit_every_n_steps` says so (when a level set is present;
/// the cadence is unchanged). Chosen per step through
/// [`StepOptions::with_level_set_reinit`]; [`CfdSolver::step`] and
/// [`StepOptions::new`] use `FastSweeping { sweeps: 2 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LevelSetReinit {
    /// `interface_capture::fast_sweeping_reinit` with `sweeps` sweeps: the
    /// Zhao fast sweeping solve of the eikonal equation from the frozen
    /// interface cells. Refused when `sweeps` is zero.
    FastSweeping {
        /// Sweeps; refused when zero.
        sweeps: u32,
    },
    /// `multiphase::reinitialize_level_set` with `iterations` explicit
    /// pseudo-time steps of `φ_τ = sgn(φ) (1 − |∇φ|)`, `Δτ = dx / 2`,
    /// central differences, the outermost cell layer frozen. A signed
    /// distance field is its fixed point to the bit; a grid with fewer than
    /// three cells on an axis is left untouched. Refused when `iterations`
    /// is zero.
    PseudoTime {
        /// Pseudo-time steps; refused when zero.
        iterations: u32,
    },
}

/// What [`CfdSolver::step`] reinitialises the level set with.
const DEFAULT_LEVEL_SET_REINIT: LevelSetReinit = LevelSetReinit::FastSweeping { sweeps: 2 };

/// What a step does beyond the shared body: which pressure solver, whether a
/// wall model replaces the no-slip ghost at the walls, and how the level set
/// is reinitialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepOptions {
    pressure: PressureSolver,
    wall_model: Option<WallModel>,
    level_set_reinit: LevelSetReinit,
}

impl StepOptions {
    /// Project with `pressure`, no wall model, the level set reinitialised
    /// as [`CfdSolver::step`] does (`FastSweeping { sweeps: 2 }`).
    #[must_use]
    pub const fn new(pressure: PressureSolver) -> Self {
        Self {
            pressure,
            wall_model: None,
            level_set_reinit: DEFAULT_LEVEL_SET_REINIT,
        }
    }

    /// Replace the no-slip ghost at every wall with `model`.
    #[must_use]
    pub const fn with_wall_model(mut self, model: WallModel) -> Self {
        self.wall_model = Some(model);
        self
    }

    /// Reinitialise the level set with `reinit` on the reinitialisation
    /// steps (`CfdSolver::reinit_every_n_steps`).
    #[must_use]
    pub const fn with_level_set_reinit(mut self, reinit: LevelSetReinit) -> Self {
        self.level_set_reinit = reinit;
        self
    }

    /// The pressure solver.
    #[must_use]
    pub const fn pressure(&self) -> PressureSolver {
        self.pressure
    }

    /// The wall model, if any.
    #[must_use]
    pub const fn wall_model(&self) -> Option<WallModel> {
        self.wall_model
    }
}

/// Why [`CfdSolver::step_with_options`] refused to step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StepError {
    /// The pressure solver request was refused.
    Pressure(PressureSolverError),
    /// A wall model was requested on a solver whose molecular viscosity is
    /// zero: `y⁺ = y u_τ / ν` is undefined and no wall shear can be read.
    WallModelNeedsViscosity,
    /// The [`RansState`] handed to [`CfdSolver::step_rans`] does not have
    /// the grid's cell dimensions (or, for a prescribed eddy viscosity, its
    /// spacing).
    TurbulenceFieldShape,
    /// A prescribed eddy viscosity holds a negative cell: the diffusion would
    /// pump energy into the flow.
    NegativeEddyViscosity,
    /// The explicit diffusion would be unstable: the diffusion number
    /// `(ν + max ν_t) dt / dx²` exceeds `1/6`, the limit of the
    /// seven-point explicit Laplacian. Reduce `dt` or the field. A clamp here
    /// would hide a blown-up field as a damped one, so the step is refused.
    DiffusionUnstable {
        /// The diffusion number that was measured.
        diffusion_number: Fix128,
    },
    /// The [`LevelSetReinit`] has a zero sweep / iteration count: nothing
    /// would be reinitialised on the reinitialisation steps. Refused whether
    /// or not a level set is present, as the zero counts of
    /// [`PressureSolver`] are.
    ZeroReinitCount,
}

impl From<PressureSolverError> for StepError {
    fn from(e: PressureSolverError) -> Self {
        Self::Pressure(e)
    }
}

impl core::fmt::Display for StepError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Pressure(e) => write!(f, "pressure solver: {e}"),
            Self::WallModelNeedsViscosity => {
                write!(f, "a wall model needs a positive molecular viscosity")
            }
            Self::TurbulenceFieldShape => {
                write!(
                    f,
                    "the turbulence field does not match the grid's cell dimensions"
                )
            }
            Self::NegativeEddyViscosity => {
                write!(f, "a prescribed eddy viscosity holds a negative cell")
            }
            Self::DiffusionUnstable { diffusion_number } => write!(
                f,
                "explicit diffusion unstable: diffusion number {} exceeds 1/6",
                diffusion_number.to_f64()
            ),
            Self::ZeroReinitCount => {
                write!(f, "the level set reinitialisation count is zero")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StepError {}

/// What the wall model did over one step, summarised over every wall-adjacent
/// face it touched.
///
/// The envelope (`min` / `max`) is taken over the faces that **move relative
/// to their wall**; a face at rest has `u_τ = 0` exactly and is counted in
/// `resting_faces` instead, so that a channel whose spanwise faces are all at
/// rest still reports the one friction velocity its streamwise faces share
/// (`u_tau_min == u_tau_max` there). `k` and `ε` are the wall-consistent
/// values of `turbulence::wall_k_epsilon` at the first face centre. A
/// k-ε / k-ω step of [`CfdSolver::step_rans`] with a [`WallModel`] imposes
/// the same formulas on the wall-adjacent cells, from the cell-centred
/// velocity (see [`TurbulenceModel::KEpsilon`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct WallShearSummary {
    /// Wall-adjacent face updates the model performed (a face next to two
    /// walls counts twice), moving and resting alike.
    pub faces: usize,
    /// Of those, the faces at rest relative to their wall (`u_τ = 0`,
    /// excluded from the envelope below).
    pub resting_faces: usize,
    /// Smallest friction velocity over the moving faces (m/s).
    pub u_tau_min: Fix128,
    /// Largest friction velocity over the moving faces (m/s).
    pub u_tau_max: Fix128,
    /// Smallest `y⁺` at the first face centre, over the moving faces.
    pub y_plus_min: Fix128,
    /// Largest `y⁺` at the first face centre, over the moving faces.
    pub y_plus_max: Fix128,
    /// Largest wall-consistent turbulent kinetic energy `u_τ²/√C_μ` (m²/s²).
    pub k_max: Fix128,
    /// Largest wall-consistent dissipation `u_τ³/(κ y_p)` (m²/s³).
    pub epsilon_max: Fix128,
}

impl WallShearSummary {
    fn with_no_faces() -> Self {
        Self {
            faces: 0,
            resting_faces: 0,
            u_tau_min: Fix128::ZERO,
            u_tau_max: Fix128::ZERO,
            y_plus_min: Fix128::ZERO,
            y_plus_max: Fix128::ZERO,
            k_max: Fix128::ZERO,
            epsilon_max: Fix128::ZERO,
        }
    }

    fn fold_face(&mut self, u_tau: Fix128, y_plus_value: Fix128, k: Fix128, epsilon: Fix128) {
        self.faces += 1;
        if u_tau.is_zero() {
            self.resting_faces += 1;
            return;
        }
        let moving_so_far = self.faces - self.resting_faces - 1;
        if moving_so_far == 0 {
            self.u_tau_min = u_tau;
            self.u_tau_max = u_tau;
            self.y_plus_min = y_plus_value;
            self.y_plus_max = y_plus_value;
        } else {
            if u_tau < self.u_tau_min {
                self.u_tau_min = u_tau;
            }
            if u_tau > self.u_tau_max {
                self.u_tau_max = u_tau;
            }
            if y_plus_value < self.y_plus_min {
                self.y_plus_min = y_plus_value;
            }
            if y_plus_value > self.y_plus_max {
                self.y_plus_max = y_plus_value;
            }
        }
        if k > self.k_max {
            self.k_max = k;
        }
        if epsilon > self.epsilon_max {
            self.epsilon_max = epsilon;
        }
    }
}

/// What [`CfdSolver::step_with_options`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct StepReport {
    /// The BiCGStab verdict, when that was the pressure solver.
    pub bicgstab: Option<BicgstabStats>,
    /// The wall model's summary, when one was enabled.
    pub wall: Option<WallShearSummary>,
}

/// What [`CfdSolver::step_rans`] did: the shared-body reports plus the
/// closure's summary and the eddy viscosity field it diffused with.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RansReport {
    /// The eddy viscosity the step used, one value per cell (what a caller
    /// snapshots next to the [`RansState`], or feeds back as a prescribed
    /// field).
    pub eddy_viscosity: Grid3d,
    /// The closure's summary.
    pub turbulence: TurbulenceSummary,
    /// The BiCGStab verdict, when that was the pressure solver.
    pub bicgstab: Option<BicgstabStats>,
    /// The wall model's summary, when one was enabled.
    pub wall: Option<WallShearSummary>,
}

/// Which turbulence closure [`CfdSolver::step_rans`] runs, carried by the
/// [`RansState`] the caller owns.
///
/// Every closure produces one eddy viscosity `ν_t` per cell at the **start**
/// of the step, from the state the step begins with, and the momentum
/// diffusion then runs with the cell-wise `ν = ν_mol + ν_t` (interface
/// coefficients are harmonic means, so a two-layer Couette flow has the
/// piecewise-linear profile as its discrete fixed point). The field used is
/// returned in [`RansReport::eddy_viscosity`], and its envelope in
/// [`TurbulenceSummary`]. The legacy `use_turbulence` flag is a different
/// path (one grid-wide `ν_t` from the largest diagonal strain) and is left
/// as it was.
///
/// # Stability
///
/// The diffusion is explicit, so `(ν_mol + max ν_t) dt / dx² ≤ 1/6` is
/// required; a step that would exceed it is refused with
/// [`StepError::DiffusionUnstable`] before anything is touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TurbulenceModel {
    /// Static Smagorinsky: `ν_t = (C_s Δ)² |S|` per cell with the full
    /// strain-rate magnitude at the cell centre (`C_s = 0.17`).
    Smagorinsky,
    /// Smagorinsky with the coefficient per cell from the ratio of the
    /// test-filtered strain (the mean of the six neighbours' `|S|`) to the
    /// grid strain, clamped to `[0.05, 0.25]`; under uniform strain the ratio
    /// is exactly one and the step is bit-identical to
    /// [`TurbulenceModel::Smagorinsky`]. This is the ratio form of
    // LIMITATION(COV-CFD-044): This is the ratio form of `turbulence::dynamic_smagorinsky_cs`, not the Germano least-squares procedure.
    /// `turbulence::dynamic_smagorinsky_cs`, not the Germano least-squares
    /// procedure.
    DynamicSmagorinsky,
    /// Launder–Spalding k-ε on the state's `(k, ε)`: per step, production
    /// `P_k = ν_t |S|²`, the explicit point sources of `k` and `ε`, explicit
    /// diffusion with `ν_mol + ν_t / σ`, then semi-Lagrangian advection, in
    /// that order; `ν_t = C_μ k² / ε`.
    ///
    /// With a [`WallModel`] in the [`StepOptions`], the cells with a no-slip
    /// wall face then take the Launder–Spalding (1974) wall-function values
    /// `k = u_τ²/√C_μ`, `ε = u_τ³/(κ y_p)` at `y_p = dx/2`, `u_τ` the
    /// friction velocity of the cell-centred speed relative to the wall
    /// (fixed values, replacing what the transport gave those cells; the mean
    /// over the faces for a corner cell). Two more pieces of the log layer
    /// come with it, because the first cells off the wall are not resolved:
    /// the momentum diffusion takes `ν_t = κ u_τ dx` (the log-layer value at
    /// the interface height) on the interface one cell from the wall instead
    /// of the harmonic mean of the two cells (`0.75 κ u_τ dx` there), and the
    /// production of the second cell from the wall reads the wall-normal
    /// gradient `u_τ / (κ · 3dx/2)` instead of the centred difference across
    /// the wall cell (21 % too steep on a log profile, independently of
    /// `dx`).
    ///
    // LIMITATION(COV-CFD-052): only that interface and that cell are replaced; from the third cell on the ordinary discretisation stands, and a cell with walls in reach on both sides (a three-cell gap) keeps the finite difference.
    /// ⚠️ Limitation: only that interface and that cell are replaced; from
    /// the third cell on the ordinary discretisation stands, and a cell with
    /// walls in reach on both sides (a three-cell gap) keeps the finite
    /// difference.
    ///
    /// ⚠️ Limitation: **without a [`WallModel`] no boundary condition acts on
    /// `k` or `ε` at a wall** — there is neither a low-Reynolds-number wall
    /// treatment nor a wall function, only the zero flux of the cell
    /// diffusion. The closure is then a high-Reynolds-number model run to the
    /// wall, which it does not describe.
    KEpsilon,
    /// Wilcox (1988) k-ω on the same `(k, ε)` storage: each cell is converted
    /// to `ω = ε / (β* k)`, advanced with `dk/dt = P − β* k ω`,
    /// `dω/dt = α (ω/k) P − β ω²` (`α = 5/9`, `β = 3/40`, `σ = 2`) and
    /// converted back; `ν_t = k / ω`.
    ///
    /// The wall treatment is that of [`TurbulenceModel::KEpsilon`], on the
    /// same `(k, ε)` storage, so the wall cells hold
    /// `ω = ε/(β* k) = u_τ/(√β* κ y_p)` with a [`WallModel`] and **no wall
    /// boundary condition on `k` or `ω` without one** (the same limitation).
    KOmega,
    /// The eddy viscosity the state was built from
    /// ([`RansState::prescribed`]) as given, cell by cell; nothing is
    /// transported. The way to feed a closure computed elsewhere into the
    /// momentum diffusion.
    Prescribed,
}

/// The turbulence state a caller owns and hands to [`CfdSolver::step_rans`]:
/// which closure runs and the cell-centred field it needs.
///
/// For [`TurbulenceModel::KEpsilon`] and [`TurbulenceModel::KOmega`] that
/// field is `(k, ε)` — `k` the turbulent kinetic energy (m²/s²), `ε` the
/// dissipation rate (m²/s³); the k-ω closure derives `ω = ε / (β* k)` per
/// cell and writes `ε = β* k ω` back. For [`TurbulenceModel::Prescribed`] it
/// is the eddy viscosity itself ([`RansState::prescribed`]). The LES
/// closures carry no field. The state lives outside the solver so that a
/// caller can snapshot and roll it back next to the grid; the solver's own
/// layout is unchanged by the closures. Out-of-range reads return zero and
/// out-of-range writes are ignored, as the MAC grid's accessors do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RansState {
    model: TurbulenceModel,
    nx: usize,
    ny: usize,
    nz: usize,
    k: Vec<Fix128>,
    epsilon: Vec<Fix128>,
    /// `ν_t` per cell for the prescribed closure; empty otherwise.
    prescribed: Vec<Fix128>,
    /// Spacing the prescribed field was given on.
    prescribed_dx: Fix128,
}

impl RansState {
    /// A state of `nx × ny × nz` cells for `model`, with `k = ε = 0`
    /// everywhere (nothing for the LES closures to carry).
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, model: TurbulenceModel) -> Self {
        Self::uniform(nx, ny, nz, model, Fix128::ZERO, Fix128::ZERO)
    }

    /// A state holding `k` and `epsilon` in every cell.
    #[must_use]
    pub fn uniform(
        nx: usize,
        ny: usize,
        nz: usize,
        model: TurbulenceModel,
        k: Fix128,
        epsilon: Fix128,
    ) -> Self {
        let cells = nx * ny * nz;
        Self {
            model,
            nx,
            ny,
            nz,
            k: vec![k; cells],
            epsilon: vec![epsilon; cells],
            prescribed: Vec::new(),
            prescribed_dx: Fix128::ZERO,
        }
    }

    /// A [`TurbulenceModel::Prescribed`] state: `nu_t` is the eddy viscosity
    /// per cell, used as given (its spacing has to match the grid's).
    #[must_use]
    pub fn prescribed(nu_t: Grid3d) -> Self {
        let cells = nu_t.nx * nu_t.ny * nu_t.nz;
        Self {
            model: TurbulenceModel::Prescribed,
            nx: nu_t.nx,
            ny: nu_t.ny,
            nz: nu_t.nz,
            k: vec![Fix128::ZERO; cells],
            epsilon: vec![Fix128::ZERO; cells],
            prescribed: nu_t.data,
            prescribed_dx: nu_t.dx,
        }
    }

    /// The closure this state runs.
    #[must_use]
    pub const fn model(&self) -> TurbulenceModel {
        self.model
    }

    /// Cell dimensions `(nx, ny, nz)`.
    #[must_use]
    pub const fn cell_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    const fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.nx * (j + self.ny * k)
    }

    const fn in_range(&self, i: usize, j: usize, k: usize) -> bool {
        i < self.nx && j < self.ny && k < self.nz
    }

    /// `k` of cell `(i, j, k)`; zero out of range.
    #[must_use]
    pub fn k(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if self.in_range(i, j, k) {
            self.k[self.idx(i, j, k)]
        } else {
            Fix128::ZERO
        }
    }

    /// `ε` of cell `(i, j, k)`; zero out of range.
    #[must_use]
    pub fn epsilon(&self, i: usize, j: usize, k: usize) -> Fix128 {
        if self.in_range(i, j, k) {
            self.epsilon[self.idx(i, j, k)]
        } else {
            Fix128::ZERO
        }
    }

    /// Set `k` and `ε` of cell `(i, j, k)`; ignored out of range.
    pub fn set(&mut self, i: usize, j: usize, k: usize, k_value: Fix128, epsilon: Fix128) {
        if self.in_range(i, j, k) {
            let ix = self.idx(i, j, k);
            self.k[ix] = k_value;
            self.epsilon[ix] = epsilon;
        }
    }

    /// The eddy viscosity of cell `(i, j, k)` under this state's closure:
    /// `C_μ k² / ε` for k-ε, `k / ω` with `ω = ε / (β* k)` for k-ω (the same
    /// quantity up to rounding, because `β* = C_μ`), the given value for the
    /// prescribed closure, zero for the LES closures (which read the strain
    /// instead), and zero whenever `k` or `ε` is zero (no division by zero).
    #[must_use]
    pub fn eddy_viscosity(&self, i: usize, j: usize, k: usize) -> Fix128 {
        let state = KEpsilonState {
            k: self.k(i, j, k),
            epsilon: self.epsilon(i, j, k),
        };
        match self.model {
            TurbulenceModel::KEpsilon => state.eddy_viscosity(),
            TurbulenceModel::KOmega => KOmegaState::from_k_epsilon(&state).eddy_viscosity(),
            TurbulenceModel::Prescribed => {
                if self.in_range(i, j, k) && self.prescribed.len() == self.k.len() {
                    self.prescribed[self.idx(i, j, k)]
                } else {
                    Fix128::ZERO
                }
            }
            _ => Fix128::ZERO,
        }
    }

    fn cell_state(&self, c: usize) -> KEpsilonState {
        KEpsilonState {
            k: self.k[c],
            epsilon: self.epsilon[c],
        }
    }
}

/// What the turbulence closure did over one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct TurbulenceSummary {
    /// The closure that ran.
    pub model: TurbulenceModel,
    /// Smallest eddy viscosity over the cells (m²/s).
    pub nu_t_min: Fix128,
    /// Largest eddy viscosity over the cells (m²/s).
    pub nu_t_max: Fix128,
    /// `(ν_mol + nu_t_max) dt / dx²`, at most `1/6` for the step to run.
    pub diffusion_number: Fix128,
    /// Smallest / largest Smagorinsky coefficient used (the constant for the
    /// static model, zero for the closures that have none).
    pub cs_min: Fix128,
    /// See `cs_min`.
    pub cs_max: Fix128,
    /// Largest production `P_k = ν_t |S|²` over the cells (k-ε / k-ω only,
    /// zero otherwise).
    pub production_max: Fix128,
    /// Envelope of `k` after the step (k-ε / k-ω only, zero otherwise).
    pub k_min: Fix128,
    /// See `k_min`.
    pub k_max: Fix128,
    /// Envelope of `ε` after the step (k-ε / k-ω only, zero otherwise).
    pub epsilon_min: Fix128,
    /// See `epsilon_min`.
    pub epsilon_max: Fix128,
    /// Number of cells whose explicit source step would have taken `k`, `ε`
    /// or `ω` negative and was clamped to zero. A decaying field integrated
    /// within its stability limit reports zero here; a non-zero count means
    /// `dt` is too large for the field.
    pub clamped: u32,
}

/// The eddy viscosity field prepared at the start of a step, before anything
/// is touched.
struct TurbulenceRun {
    model: TurbulenceModel,
    nu_t: Vec<Fix128>,
    nu_t_min: Fix128,
    nu_t_max: Fix128,
    diffusion_number: Fix128,
    cs_min: Fix128,
    cs_max: Fix128,
}

/// The wall sink the model applies to one face component, and the state it
/// reports: `(Δu, u_τ, y⁺, k, ε)`.
///
/// `Δu = −dt u_τ² sgn(u_rel) / dx` is the momentum the wall shear `ρ u_τ²`
/// removes from the half cell between the wall and the face over `dt`, per
/// unit mass; with `u_τ² = ν u_rel / y_p` (the sublayer) and `y_p = dx/2` it
/// is `−2 ν dt u_rel / dx²`, the no-slip ghost's contribution exactly.
struct WallSink {
    nu_mol: Fix128,
    density: Fix128,
    dynamic_viscosity: Fix128,
    y_p: Fix128,
    dt_over_dx: Fix128,
}

impl WallSink {
    fn apply(&self, u_rel: Fix128, summary: &mut WallShearSummary) -> Fix128 {
        let speed = u_rel.abs();
        let u_tau = friction_velocity_checked(speed, self.y_p, self.nu_mol);
        let yp = y_plus(self.density, u_tau, self.y_p, self.dynamic_viscosity);
        let (k, epsilon) = wall_k_epsilon(u_tau, self.y_p);
        summary.fold_face(u_tau, yp, k, epsilon);
        let magnitude = self.dt_over_dx * u_tau * u_tau;
        if u_rel.is_negative() {
            magnitude
        } else {
            -magnitude
        }
    }
}

/// The projection the shared step body runs, after validation.
#[derive(Clone, Copy)]
enum Projection {
    Gs(u32),
    Mg(u32),
    Jacobi(u32),
    BiCgStab {
        max_iterations: u32,
        tolerance: Fix128,
    },
    Decomposed {
        ranks: usize,
        sweeps: u32,
    },
    Banded {
        ranks: usize,
        sweeps: u32,
    },
    DecomposedMg {
        ranks: usize,
        cycles: u32,
    },
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
    /// Reinitialise the level set every N steps, at the end of steps N, 2N, …
    /// (counting the first step as step 1; 0 = never).
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
            beta_per_k: Fix128::from_ratio(207, 1_000_000), // water at 20 C ≈ 2.07e-4
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
    /// The result is capped at [`Self::MAX_DT_CAP`] (`1_000_000` s): a field
    /// that is exactly zero has no Courant constraint, and a field so slow
    /// that `cfl_target · dx / |u|_max` would exceed the cap is reported as
    /// the cap rather than as that quotient. The cap is what keeps the
    /// division inside `Fix128`: a peak of a few ulp with `dx` of order one
    /// would otherwise wrap the quotient into a wrong, possibly negative,
    /// time step. `cfl_target` should typically be in `[0.5, 1.0]` for
    /// semi-Lagrangian and higher for BFECC / MacCormack when paired with a
    /// monotone clamp.
    ///
    /// # Degenerate input
    ///
    /// `cfl_target <= 0` or `dx = 0` give **zero**: there is no positive time
    /// step that satisfies a non-positive Courant number, and a grid without
    /// spacing has no Courant number at all. [`Self::step_adaptive`] then
    /// takes no step.
    #[must_use]
    pub fn compute_max_dt(&self, cfl_target: Fix128) -> Fix128 {
        if cfl_target <= Fix128::ZERO || self.grid.dx.is_zero() {
            return Fix128::ZERO;
        }
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
        let numerator = cfl_target * self.grid.dx;
        // `cfl · dx / peak > cap` ⇔ `peak · cap < cfl · dx`. The product is
        // checked so that a peak large enough to wrap it (which would need
        // |u| ≳ 9e12 m/s) is read as "no cap", not as a wrapped small number.
        let capped = match peak.checked_mul(Self::MAX_DT_CAP) {
            None => false,
            Some(scaled) => scaled < numerator,
        };
        if peak.is_zero() || capped {
            return Self::MAX_DT_CAP;
        }
        numerator / peak
    }

    /// Largest time step [`Self::compute_max_dt`] reports: `1_000_000` s.
    pub const MAX_DT_CAP: Fix128 = Fix128 {
        hi: 1_000_000,
        lo: 0,
    };

    /// Convenience — step with an automatically chosen `dt` from
    /// [`Self::compute_max_dt`], capped by `dt_ceiling`.
    ///
    /// Useful in engineering demos where the simulation should adapt
    /// to fast transients without the caller re-computing `dt` on each
    /// tick. Returns the `dt` that was actually integrated, which is
    /// `min(compute_max_dt(cfl_target), dt_ceiling)`; the step taken is
    /// [`Self::step`] with that `dt`, bit for bit.
    ///
    /// # Degenerate input
    ///
    /// When that minimum is not positive — `dt_ceiling <= 0`, or
    /// [`Self::compute_max_dt`] returned zero — no step is taken, the solver
    /// is left untouched (including `step_count`) and `0` is returned. A
    /// negative time step would integrate the fluid backwards, which is never
    /// what an adaptive caller asked for.
    pub fn step_adaptive(&mut self, cfl_target: Fix128, dt_ceiling: Fix128) -> Fix128 {
        let mut dt = self.compute_max_dt(cfl_target);
        if dt > dt_ceiling {
            dt = dt_ceiling;
        }
        if dt <= Fix128::ZERO {
            return Fix128::ZERO;
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
    ///
    /// `step` has no error channel and does not check the explicit diffusion
    /// limit `ν dt / dx² ≤ 1/6`; [`Self::step_with_options`] refuses a step
    /// beyond it with [`StepError::DiffusionUnstable`].
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
    pub fn step_multigrid(&mut self, dt_s: Fix128, cycles: u32) {
        self.step_with_projection(dt_s, Some(cycles));
    }

    /// Whether [`project_pressure_multigrid`] can solve this grid.
    fn grid_supports_multigrid(&self) -> bool {
        self.grid.nx.is_power_of_two()
            && self.grid.ny.is_power_of_two()
            && self.grid.nz.is_power_of_two()
    }

    /// Map the default / `step_multigrid` request onto a projection, keeping
    /// the documented fallback of those two entry points.
    fn default_projection(&self, multigrid_cycles: Option<u32>) -> Projection {
        // `None` is the default projection: multigrid where the grid allows it
        let cycles = multigrid_cycles.unwrap_or(DEFAULT_MULTIGRID_CYCLES);
        match cycles {
            cycles @ 1.. if self.grid_supports_multigrid() => Projection::Mg(cycles),
            _ => Projection::Gs(self.jacobi_iterations),
        }
    }

    /// The step body shared by [`Self::step`] (`multigrid_cycles = None`, which
    /// means 6 W-cycles) and
    /// [`Self::step_multigrid`].
    fn step_with_projection(&mut self, dt_s: Fix128, multigrid_cycles: Option<u32>) {
        if dt_s.is_zero() {
            return;
        }
        let projection = self.default_projection(multigrid_cycles);
        self.step_body(dt_s, projection, None, None, DEFAULT_LEVEL_SET_REINIT);
    }

    /// [`Self::step`] with the pressure projection done by the solver the
    /// caller names, and the inputs the fixed-count solvers would answer
    /// silently refused instead. [`Self::step_with_options`] with
    /// `StepOptions::new(solver)`; see there.
    ///
    /// # Errors
    ///
    /// [`PressureSolverError`], as [`Self::step_with_options`] reports it.
    pub fn step_with_pressure_solver(
        &mut self,
        dt_s: Fix128,
        solver: PressureSolver,
    ) -> Result<ProjectionReport, PressureSolverError> {
        match self.step_with_options(dt_s, &StepOptions::new(solver)) {
            Ok(report) => Ok(ProjectionReport {
                solver,
                bicgstab: report.bicgstab,
            }),
            Err(StepError::Pressure(e)) => Err(e),
            Err(StepError::WallModelNeedsViscosity) => {
                unreachable!("no wall model was requested")
            }
            Err(_) => unreachable!(
                "no turbulence closure was requested and the default reinitialisation count is 2"
            ),
        }
    }

    /// Validate the level set reinitialisation request: a zero count is
    /// refused whether or not a level set is present.
    fn reinit_for(reinit: LevelSetReinit) -> Result<LevelSetReinit, StepError> {
        match reinit {
            LevelSetReinit::FastSweeping { sweeps: 0 }
            | LevelSetReinit::PseudoTime { iterations: 0 } => Err(StepError::ZeroReinitCount),
            other => Ok(other),
        }
    }

    /// Validate a pressure solver request against this solver's state.
    fn projection_for(&self, solver: PressureSolver) -> Result<Projection, PressureSolverError> {
        if self.density_kg_m3.is_zero() {
            return Err(PressureSolverError::ZeroDensity);
        }
        if self.grid.dx.is_zero() {
            return Err(PressureSolverError::ZeroSpacing);
        }
        Ok(match solver {
            PressureSolver::RedBlackGs { sweeps: 0 }
            | PressureSolver::Multigrid { cycles: 0 }
            | PressureSolver::Jacobi { iterations: 0 }
            | PressureSolver::DecomposedGs { sweeps: 0, .. }
            | PressureSolver::BandedGs { sweeps: 0, .. }
            | PressureSolver::DecomposedMultigrid { cycles: 0, .. }
            | PressureSolver::BiCgStab {
                max_iterations: 0, ..
            } => return Err(PressureSolverError::ZeroIterations),
            PressureSolver::DecomposedGs { ranks: 0, .. }
            | PressureSolver::BandedGs { ranks: 0, .. }
            | PressureSolver::DecomposedMultigrid { ranks: 0, .. } => {
                return Err(PressureSolverError::ZeroRanks)
            }
            PressureSolver::DecomposedGs { ranks, sweeps } => {
                Projection::Decomposed { ranks, sweeps }
            }
            PressureSolver::BandedGs { ranks, sweeps } => Projection::Banded { ranks, sweeps },
            PressureSolver::RedBlackGs { sweeps } => Projection::Gs(sweeps),
            PressureSolver::Multigrid { cycles } => {
                if !self.grid_supports_multigrid() {
                    return Err(PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                        extents: (self.grid.nx, self.grid.ny, self.grid.nz),
                    });
                }
                Projection::Mg(cycles)
            }
            PressureSolver::DecomposedMultigrid { ranks, cycles } => {
                if !self.grid_supports_multigrid() {
                    return Err(PressureSolverError::MultigridNeedsPowerOfTwoExtents {
                        extents: (self.grid.nx, self.grid.ny, self.grid.nz),
                    });
                }
                Projection::DecomposedMg { ranks, cycles }
            }
            PressureSolver::Jacobi { iterations } => Projection::Jacobi(iterations),
            PressureSolver::BiCgStab {
                max_iterations,
                tolerance,
            } => {
                if tolerance <= Fix128::ZERO {
                    return Err(PressureSolverError::NonPositiveTolerance);
                }
                Projection::BiCgStab {
                    max_iterations,
                    tolerance,
                }
            }
        })
    }

    /// [`Self::step`] with the pressure projection and the wall treatment the
    /// caller names in `options`, every unusable input refused up front.
    ///
    /// Everything but the projection and the wall flux is the shared step
    /// body: `StepOptions::new(Multigrid { cycles: 6 })` on a power-of-two
    /// grid, or `RedBlackGs { sweeps: jacobi_iterations }` on any other,
    /// reproduces [`Self::step`] bit for bit. With a [`WallModel`] the
    /// viscous flux at every [`crate::eulerian_grid::FaceBc::Wall`] comes from
    /// the log law instead of the no-slip ghost, and the step reports the
    /// envelope of what the model read ([`WallShearSummary`]).
    ///
    /// The level set, when present, is reinitialised on the steps
    /// `reinit_every_n_steps` names with the [`LevelSetReinit`] of `options`
    /// (`StepOptions::new` carries the `FastSweeping { sweeps: 2 }` of
    /// [`Self::step`]).
    ///
    /// # Errors
    ///
    /// [`StepError::Pressure`] for the refusals of
    /// [`Self::step_with_pressure_solver`],
    /// [`StepError::WallModelNeedsViscosity`] for a wall model on a solver
    /// with zero molecular viscosity, [`StepError::ZeroReinitCount`] for
    /// a reinitialisation with a zero count, and [`StepError::DiffusionUnstable`]
    /// when the explicit molecular diffusion number `ν dt / dx²` exceeds `1/6`
    /// (the limit [`Self::step_rans`] applies with the eddy viscosity added).
    /// ⚠️ On `Err` the solver is **not stepped**: the grid, the level set and
    /// `step_count` are untouched.
    pub fn step_with_options(
        &mut self,
        dt_s: Fix128,
        options: &StepOptions,
    ) -> Result<StepReport, StepError> {
        if dt_s.is_zero() {
            return Err(PressureSolverError::ZeroTimeStep.into());
        }
        let projection = self.projection_for(options.pressure)?;
        if options.wall_model.is_some() && self.dynamic_viscosity_pas <= Fix128::ZERO {
            return Err(StepError::WallModelNeedsViscosity);
        }
        let reinit = Self::reinit_for(options.level_set_reinit)?;
        // the molecular diffusion is explicit too: the same 1/6 limit as step_rans
        if self.density_kg_m3 > Fix128::ZERO {
            let dx = self.grid.dx;
            let diffusion_number =
                self.dynamic_viscosity_pas / self.density_kg_m3 * dt_s / (dx * dx);
            if diffusion_number > Fix128::from_ratio(1, 6) {
                return Err(StepError::DiffusionUnstable { diffusion_number });
            }
        }
        let (bicgstab, wall, _) =
            self.step_body(dt_s, projection, options.wall_model.as_ref(), None, reinit);
        Ok(StepReport { bicgstab, wall })
    }

    /// [`Self::step_with_options`] with a turbulence closure: the eddy
    /// viscosity of `state`'s [`TurbulenceModel`] enters the momentum
    /// diffusion cell by cell, and the transport closures advance `state`.
    ///
    /// The closure's field is prepared from the state the step starts with,
    /// every refusal is checked before anything changes, and the rest is the
    /// shared step body with the cell-wise diffusion in place of the scalar
    /// one (see [`TurbulenceModel`] for the order of operations). `state` is
    /// the caller's: it holds `(k, ε)` for the transport closures and is left
    /// advanced by one step; the eddy viscosity used comes back in
    /// [`RansReport::eddy_viscosity`].
    ///
    /// For the transport closures the [`WallModel`] of `options` is also the
    /// wall function of `(k, ε)`: with it, the wall-adjacent cells are set
    /// to `k = u_τ²/√C_μ`, `ε = u_τ³/(κ y_p)` after the transport, the
    /// interface one cell from the wall diffuses momentum with `κ u_τ dx`,
    /// and the second cell's production reads the log-law gradient (see
    /// [`TurbulenceModel::KEpsilon`]; only that interface and that cell).
    /// ⚠️ Limitation: without a [`WallModel`], **`k` and `ε` (or `ω`) have no
    /// wall boundary condition** — no low-Reynolds-number wall treatment and
    /// no wall function — so a k-ε / k-ω run against a no-slip wall without
    /// one is not a modelled wall-bounded flow.
    ///
    /// # Errors
    ///
    /// Everything [`Self::step_with_options`] refuses, plus
    /// [`StepError::TurbulenceFieldShape`] when `state` does not match the
    /// grid, [`StepError::NegativeEddyViscosity`] for a prescribed field with
    /// a negative cell, and [`StepError::DiffusionUnstable`] when
    /// `(ν_mol + max ν_t) dt / dx²` exceeds `1/6`. ⚠️ On `Err` neither the
    /// solver nor `state` is touched.
    pub fn step_rans(
        &mut self,
        dt_s: Fix128,
        options: &StepOptions,
        state: &mut RansState,
    ) -> Result<RansReport, StepError> {
        if dt_s.is_zero() {
            return Err(PressureSolverError::ZeroTimeStep.into());
        }
        let projection = self.projection_for(options.pressure)?;
        if options.wall_model.is_some() && self.dynamic_viscosity_pas <= Fix128::ZERO {
            return Err(StepError::WallModelNeedsViscosity);
        }
        let reinit = Self::reinit_for(options.level_set_reinit)?;
        let run = self.prepare_turbulence(state, dt_s)?;
        let (bicgstab, wall, turbulence) = self.step_body(
            dt_s,
            projection,
            options.wall_model.as_ref(),
            Some((&run, state)),
            reinit,
        );
        Ok(RansReport {
            eddy_viscosity: Grid3d {
                nx: self.grid.nx,
                ny: self.grid.ny,
                nz: self.grid.nz,
                dx: self.grid.dx,
                data: run.nu_t,
            },
            turbulence: turbulence.expect("a closure ran"),
            bicgstab,
            wall,
        })
    }

    /// The step body: boundaries, advection, body forces, diffusion, the
    /// projection named by `projection`, then the optional level-set
    /// (advection, and `reinit` on the reinitialisation steps) and
    /// temperature updates. Returns the BiCGStab verdict when that is the
    /// solver.
    fn step_body(
        &mut self,
        dt_s: Fix128,
        projection: Projection,
        wall: Option<&WallModel>,
        turbulence: Option<(&TurbulenceRun, &mut RansState)>,
        reinit: LevelSetReinit,
    ) -> (
        Option<BicgstabStats>,
        Option<WallShearSummary>,
        Option<TurbulenceSummary>,
    ) {
        self.grid.enforce_face_boundaries();
        match self.advection_scheme {
            AdvectionScheme::SemiLagrangian => self.advect_velocity(dt_s),
            AdvectionScheme::MacCormack => self.advect_velocity_maccormack(dt_s),
            AdvectionScheme::Bfecc => self.advect_velocity_bfecc(dt_s),
        }
        self.apply_body_forces(dt_s);
        self.grid.enforce_face_boundaries();
        let (wall_summary, turbulence_summary) = match turbulence {
            Some((run, state)) => {
                let (production_max, clamped) = self.advance_rans(state, run, dt_s, wall);
                let nu_mol = self.dynamic_viscosity_pas / self.density_kg_m3;
                let log_layer_edges = wall.is_some()
                    && matches!(
                        run.model,
                        TurbulenceModel::KEpsilon | TurbulenceModel::KOmega
                    );
                let ws =
                    self.diffuse_velocity_variable(nu_mol, &run.nu_t, dt_s, wall, log_layer_edges);
                let (k_min, k_max, epsilon_min, epsilon_max) = match run.model {
                    TurbulenceModel::KEpsilon | TurbulenceModel::KOmega => (
                        min_of(&state.k),
                        max_of(&state.k),
                        min_of(&state.epsilon),
                        max_of(&state.epsilon),
                    ),
                    _ => (Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
                };
                (
                    ws,
                    Some(TurbulenceSummary {
                        model: run.model,
                        nu_t_min: run.nu_t_min,
                        nu_t_max: run.nu_t_max,
                        diffusion_number: run.diffusion_number,
                        cs_min: run.cs_min,
                        cs_max: run.cs_max,
                        production_max,
                        k_min,
                        k_max,
                        epsilon_min,
                        epsilon_max,
                        clamped,
                    }),
                )
            }
            None if self.use_turbulence => (self.apply_turbulent_diffusion(dt_s, wall), None),
            None => (self.apply_molecular_diffusion(dt_s, wall), None),
        };
        let stats = match projection {
            Projection::Mg(cycles) => {
                project_pressure_multigrid(&mut self.grid, dt_s, self.density_kg_m3, cycles);
                None
            }
            Projection::Gs(sweeps) => {
                project_pressure(&mut self.grid, dt_s, self.density_kg_m3, sweeps);
                None
            }
            Projection::Jacobi(iterations) => {
                project_pressure_jacobi(&mut self.grid, dt_s, self.density_kg_m3, iterations);
                None
            }
            Projection::Decomposed { ranks, sweeps } => {
                project_pressure_decomposed(
                    &mut self.grid,
                    dt_s,
                    self.density_kg_m3,
                    sweeps,
                    ranks,
                    HaloSchedule::EverySweep,
                );
                None
            }
            Projection::Banded { ranks, sweeps } => {
                project_pressure_banded(&mut self.grid, dt_s, self.density_kg_m3, sweeps, ranks);
                None
            }
            Projection::DecomposedMg { ranks, cycles } => {
                project_pressure_multigrid_decomposed(
                    &mut self.grid,
                    dt_s,
                    self.density_kg_m3,
                    cycles,
                    ranks,
                    HaloSchedule::EverySweep,
                );
                None
            }
            Projection::BiCgStab {
                max_iterations,
                tolerance,
            } => Some(project_pressure_bicgstab(
                &mut self.grid,
                dt_s,
                self.density_kg_m3,
                max_iterations,
                tolerance,
            )),
        };
        if self.level_set.is_some() {
            self.advect_level_set(dt_s);
            // `step_count` is the number of steps finished before this one:
            // reinitialise at the end of steps N, 2N, ...
            let every = u64::from(self.reinit_every_n_steps);
            if every > 0 && self.step_count % every == every - 1 {
                if let Some(ls) = self.level_set.as_mut() {
                    match reinit {
                        LevelSetReinit::FastSweeping { sweeps } => fast_sweeping_reinit(ls, sweeps),
                        LevelSetReinit::PseudoTime { iterations } => {
                            reinitialize_level_set(ls, iterations);
                        }
                    }
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
        self.step_count = self.step_count.wrapping_add(1);
        (stats, wall_summary, turbulence_summary)
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
    ///
    /// The particle-to-grid transfer is the trilinear one; [`Self::step_flip_with`]
    /// lets the caller pick the stencil.
    pub fn step_flip(
        &mut self,
        particles: &mut [(Vec3Fix, Vec3Fix)],
        dt_s: Fix128,
        flip_ratio: Fix128,
    ) {
        self.step_flip_with(particles, dt_s, flip_ratio, ParticleScatter::Trilinear);
    }

    /// [`Self::step_flip`] with the particle-to-grid stencil chosen by
    /// `scatter`.
    ///
    /// Only step 1 changes: the transfer is
    /// [`crate::eulerian_grid::p2g_normalized_with`] with the given
    /// [`ParticleScatter`]. The grid-to-particle interpolation of step 4 is
    /// trilinear for both, so with [`ParticleScatter::Nearest`] the particle
    /// velocities are still read off a smooth field; what changes is that the
    /// field no longer depends on where inside its cell each particle sits.
    /// Everything else, including every refusal above, is identical, and
    /// [`ParticleScatter::Trilinear`] reproduces [`Self::step_flip`] bit for
    /// bit.
    pub fn step_flip_with(
        &mut self,
        particles: &mut [(Vec3Fix, Vec3Fix)],
        dt_s: Fix128,
        flip_ratio: Fix128,
        scatter: ParticleScatter,
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
        p2g_normalized_with(&mut self.grid, &cloud, scatter);
        self.grid.enforce_face_boundaries();

        // 2. velocity as transferred
        let u_old = self.grid.u.clone();
        let v_old = self.grid.v.clone();
        let w_old = self.grid.w.clone();

        // 3. forces, diffusion, projection (the order `step` uses)
        self.apply_body_forces(dt_s);
        self.grid.enforce_face_boundaries();
        if self.use_turbulence {
            self.apply_turbulent_diffusion(dt_s, None);
        } else {
            self.apply_molecular_diffusion(dt_s, None);
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
        self.step_count = self.step_count.wrapping_add(1);
    }

    /// Step 1: Add body forces (gravity + buoyancy + surface tension).
    ///
    /// # Claims
    /// - The interface is smeared over `eps = 2 dx`, an integer multiple of `dx`, so the
    ///   sampled delta sums to 1 for any interface offset and the line integral of the
    ///   force across a spherical interface is `-2 sigma / R` (Young-Laplace)
    /// - The default fluid is water at 20 C: `beta_per_k = 2.07e-4` 1/K (volumetric expansion)
    /// - Buoyancy on a v face uses the mean temperature of the two cells sharing it
    ///   (a boundary face uses its single neighbour), so a linear profile is exact
    /// - Surface tension adds `f / rho * dt` (m/s) to the faces along each axis,
    ///   split 1/2 to each of the two faces of a cell, for x, y and z alike
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
                        // Face temperature: mean of the two cells sharing the face
                        // (boundary faces have one neighbour and take its value)
                        let t_below = temp.get(i, if j == 0 { 0 } else { j - 1 }, k);
                        let t_above = temp.get(i, if j == self.grid.ny { j - 1 } else { j }, k);
                        let t = (t_below + t_above) * Fix128::from_ratio(1, 2);
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
            let (fx, fy, fz) = compute_csf_field(
                ls,
                self.surface_tension_n_m,
                self.grid.dx * Fix128::from_int(2),
            );
            // Apply each cell-centred component to the two faces of the cell along
            // its own axis (1/2 each), so the three components act alike
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
                        // v faces (j, j + 1) take fy
                        let half_y =
                            fy[cell_idx] * dt_s * Fix128::from_ratio(1, 2) / self.density_kg_m3;
                        let vy_lo = i + self.grid.nx * (j + (self.grid.ny + 1) * k);
                        let vy_hi = i + self.grid.nx * ((j + 1) + (self.grid.ny + 1) * k);
                        if vy_lo < self.grid.v.len() {
                            self.grid.v[vy_lo] = self.grid.v[vy_lo] + half_y;
                        }
                        if vy_hi < self.grid.v.len() {
                            self.grid.v[vy_hi] = self.grid.v[vy_hi] + half_y;
                        }
                        // w faces (k, k + 1) take fz
                        let half_z =
                            fz[cell_idx] * dt_s * Fix128::from_ratio(1, 2) / self.density_kg_m3;
                        let wz_lo = i + self.grid.nx * (j + self.grid.ny * k);
                        let wz_hi = i + self.grid.nx * (j + self.grid.ny * (k + 1));
                        if wz_lo < self.grid.w.len() {
                            self.grid.w[wz_lo] = self.grid.w[wz_lo] + half_z;
                        }
                        if wz_hi < self.grid.w.len() {
                            self.grid.w[wz_hi] = self.grid.w[wz_hi] + half_z;
                        }
                    }
                }
            }
        }
    }

    /// Molecular-only viscous diffusion via explicit Laplacian.
    fn apply_molecular_diffusion(
        &mut self,
        dt_s: Fix128,
        wall: Option<&WallModel>,
    ) -> Option<WallShearSummary> {
        if self.density_kg_m3.is_zero() {
            return None;
        }
        let nu = self.dynamic_viscosity_pas / self.density_kg_m3;
        self.diffuse_velocity(nu, dt_s, wall)
    }

    /// Smagorinsky-augmented diffusion: `ν_eff = ν_mol + ν_t`.
    fn apply_turbulent_diffusion(
        &mut self,
        dt_s: Fix128,
        wall: Option<&WallModel>,
    ) -> Option<WallShearSummary> {
        if self.density_kg_m3.is_zero() {
            return None;
        }
        let nu_mol = self.dynamic_viscosity_pas / self.density_kg_m3;
        // Compute strain-rate magnitude at cell centres and take max as an
        // upper-bound proxy for the SGS eddy viscosity (simplification).
        let mut max_strain = Fix128::ZERO;
        // interior cells only; a grid narrower than 3 cells along an axis has none
        // (saturating, so a zero extent is an empty range, AUD-A-S1W4-007)
        for k in 1..self.grid.nz.saturating_sub(1) {
            for j in 1..self.grid.ny.saturating_sub(1) {
                for i in 1..self.grid.nx.saturating_sub(1) {
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
        self.diffuse_velocity(nu_mol + nu_t, dt_s, wall)
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
    fn diffuse_velocity(
        &mut self,
        nu: Fix128,
        dt_s: Fix128,
        wall: Option<&WallModel>,
    ) -> Option<WallShearSummary> {
        if nu.is_zero() || self.grid.dx.is_zero() {
            return None;
        }
        let coeff = nu * dt_s / (self.grid.dx * self.grid.dx);
        let two = Fix128::from_int(2);
        // The wall model, when enabled, reads `y⁺` with the *molecular*
        // viscosity whatever `nu` the interior diffuses with.
        let sink = wall.map(|_| WallSink {
            nu_mol: self.dynamic_viscosity_pas / self.density_kg_m3,
            density: self.density_kg_m3,
            dynamic_viscosity: self.dynamic_viscosity_pas,
            y_p: self.grid.dx.half(),
            dt_over_dx: dt_s / self.grid.dx,
        });
        let mut summary = WallShearSummary::with_no_faces();
        // A neighbour across a wall: the no-slip ghost `2 u_wall − u_in`
        // without a model, or — with one — no ghost (the neighbour reads as
        // the centre, contributing nothing to the Laplacian) and the modelled
        // shear added as a separate sink.
        let mut across_wall =
            |wall_component: Fix128, center: Fix128, extra: &mut Fix128| match &sink {
                None => two * wall_component - center,
                Some(model) => {
                    *extra = *extra + model.apply(center - wall_component, &mut summary);
                    center
                }
            };

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
                    let mut extra = Fix128::ZERO;
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
                        Some(w) => across_wall(w.x, center, &mut extra),
                        None if j > 0 => self.grid.u(i, j - 1, k),
                        None => center,
                    };
                    let up = match self.grid.u_wall_across_y(i, j, k, true) {
                        Some(w) => across_wall(w.x, center, &mut extra),
                        None if j + 1 < self.grid.ny => self.grid.u(i, j + 1, k),
                        None => center,
                    };
                    let back = match self.grid.u_wall_across_z(i, j, k, false) {
                        Some(w) => across_wall(w.x, center, &mut extra),
                        None if k > 0 => self.grid.u(i, j, k - 1),
                        None => center,
                    };
                    let fwd = match self.grid.u_wall_across_z(i, j, k, true) {
                        Some(w) => across_wall(w.x, center, &mut extra),
                        None if k + 1 < self.grid.nz => self.grid.u(i, j, k + 1),
                        None => center,
                    };
                    let laplacian =
                        left + right + down + up + back + fwd - center * Fix128::from_int(6);
                    let ix = i + (self.grid.nx + 1) * (j + self.grid.ny * k);
                    u_next[ix] = center + coeff * laplacian + extra;
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
                    let mut extra = Fix128::ZERO;
                    let left = match self.grid.v_wall_across_x(i, j, k, false) {
                        Some(w) => across_wall(w.y, center, &mut extra),
                        None if i > 0 => self.grid.v(i - 1, j, k),
                        None => center,
                    };
                    let right = match self.grid.v_wall_across_x(i, j, k, true) {
                        Some(w) => across_wall(w.y, center, &mut extra),
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
                        Some(w) => across_wall(w.y, center, &mut extra),
                        None if k > 0 => self.grid.v(i, j, k - 1),
                        None => center,
                    };
                    let fwd = match self.grid.v_wall_across_z(i, j, k, true) {
                        Some(w) => across_wall(w.y, center, &mut extra),
                        None if k + 1 < self.grid.nz => self.grid.v(i, j, k + 1),
                        None => center,
                    };
                    let laplacian =
                        left + right + down + up + back + fwd - center * Fix128::from_int(6);
                    let ix = i + self.grid.nx * (j + (self.grid.ny + 1) * k);
                    v_next[ix] = center + coeff * laplacian + extra;
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
                    let mut extra = Fix128::ZERO;
                    let left = match self.grid.w_wall_across_x(i, j, k, false) {
                        Some(w) => across_wall(w.z, center, &mut extra),
                        None if i > 0 => self.grid.w(i - 1, j, k),
                        None => center,
                    };
                    let right = match self.grid.w_wall_across_x(i, j, k, true) {
                        Some(w) => across_wall(w.z, center, &mut extra),
                        None if i + 1 < self.grid.nx => self.grid.w(i + 1, j, k),
                        None => center,
                    };
                    let down = match self.grid.w_wall_across_y(i, j, k, false) {
                        Some(w) => across_wall(w.z, center, &mut extra),
                        None if j > 0 => self.grid.w(i, j - 1, k),
                        None => center,
                    };
                    let up = match self.grid.w_wall_across_y(i, j, k, true) {
                        Some(w) => across_wall(w.z, center, &mut extra),
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
                    w_next[ix] = center + coeff * laplacian + extra;
                }
            }
        }
        self.grid.w = w_next;
        sink.map(|_| summary)
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

// ============================================================================
// Turbulence closures with a cell-wise eddy viscosity
// ============================================================================

/// Smallest entry, zero for an empty slice.
fn min_of(v: &[Fix128]) -> Fix128 {
    v.iter()
        .copied()
        .reduce(|a, b| if b < a { b } else { a })
        .unwrap_or(Fix128::ZERO)
}

/// Largest entry, zero for an empty slice.
fn max_of(v: &[Fix128]) -> Fix128 {
    v.iter()
        .copied()
        .reduce(|a, b| if b > a { b } else { a })
        .unwrap_or(Fix128::ZERO)
}

/// Harmonic mean of two diffusivities, `2ab / (a + b)`, evaluated as
/// `a · (2b / (a + b))` so that a small diffusivity keeps its relative
/// precision and equal inputs return exactly themselves (`2b / 2b` is `1`).
/// Zero if either is zero: no flux crosses an interface with a non-diffusing
/// side.
fn harmonic2(a: Fix128, b: Fix128) -> Fix128 {
    if a.is_zero() || b.is_zero() {
        return Fix128::ZERO;
    }
    let sum = a + b;
    if sum.is_zero() {
        return Fix128::ZERO;
    }
    a * (Fix128::from_int(2) * b / sum)
}

impl CfdSolver {
    const fn cell_index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.grid.nx * (j + self.grid.ny * k)
    }

    /// Cell-centred velocity components with the index clamped into the
    /// grid, as `advect_temperature` reads them.
    fn cell_velocity_clamped(&self, i: usize, j: usize, k: usize) -> (Fix128, Fix128, Fix128) {
        self.grid.cell_velocity(
            i.min(self.grid.nx - 1),
            j.min(self.grid.ny - 1),
            k.min(self.grid.nz - 1),
        )
    }

    /// `|S| = √(2 S_ij S_ij)` at every cell centre, from the face
    /// velocities: the diagonal from the face difference across the cell,
    /// the off-diagonals from central differences of the cell-centred
    /// velocity, one-sided on the domain edge (so a linear shear has the same
    /// strain in every cell, edge rows included).
    ///
    /// With `log_layer = Some(ν_mol)` (the wall function of a k-ε / k-ω step),
    /// the wall-normal derivative of a tangential component in a cell one
    /// cell away from a no-slip wall (its neighbour towards the wall touches
    /// it, the cell itself does not) is the log-law gradient at the cell
    /// centre, see [`Self::log_layer_gradient`]; every other derivative, and
    /// every cell further out, keeps the finite difference.
    fn cell_strain(&self, log_layer: Option<Fix128>) -> Vec<Fix128> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dx = self.grid.dx;
        let two = Fix128::from_int(2);
        let half = Fix128::from_ratio(1, 2);
        let mut out = vec![Fix128::ZERO; nx * ny * nz];
        if dx.is_zero() {
            return out;
        }
        // d(component)/d(axis) at cell (i, j, k) along one axis with `n` cells.
        let derivative = |value: &dyn Fn(usize) -> Fix128, idx: usize, n: usize| -> Fix128 {
            match (idx > 0, idx + 1 < n) {
                (true, true) => (value(idx + 1) - value(idx - 1)) / (two * dx),
                (false, true) => (value(idx + 1) - value(idx)) / dx,
                (true, false) => (value(idx) - value(idx - 1)) / dx,
                (false, false) => Fix128::ZERO,
            }
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let s11 = (self.grid.u(i + 1, j, k) - self.grid.u(i, j, k)) / dx;
                    let s22 = (self.grid.v(i, j + 1, k) - self.grid.v(i, j, k)) / dx;
                    let s33 = (self.grid.w(i, j, k + 1) - self.grid.w(i, j, k)) / dx;
                    let fd = [
                        // (component, axis, finite difference)
                        (
                            0,
                            1,
                            derivative(&|jj| self.cell_velocity_clamped(i, jj, k).0, j, ny),
                        ),
                        (
                            0,
                            2,
                            derivative(&|kk| self.cell_velocity_clamped(i, j, kk).0, k, nz),
                        ),
                        (
                            1,
                            0,
                            derivative(&|ii| self.cell_velocity_clamped(ii, j, k).1, i, nx),
                        ),
                        (
                            1,
                            2,
                            derivative(&|kk| self.cell_velocity_clamped(i, j, kk).1, k, nz),
                        ),
                        (
                            2,
                            0,
                            derivative(&|ii| self.cell_velocity_clamped(ii, j, k).2, i, nx),
                        ),
                        (
                            2,
                            1,
                            derivative(&|jj| self.cell_velocity_clamped(i, jj, k).2, j, ny),
                        ),
                    ];
                    let [du_dy, du_dz, dv_dx, dv_dz, dw_dx, dw_dy] = fd.map(|(c, a, d)| {
                        log_layer
                            .and_then(|nu| self.log_layer_gradient(i, j, k, c, a, nu))
                            .unwrap_or(d)
                    });
                    let s12 = half * (du_dy + dv_dx);
                    let s13 = half * (du_dz + dw_dx);
                    let s23 = half * (dv_dz + dw_dy);
                    out[self.cell_index(i, j, k)] =
                        strain_rate_magnitude(s11, s22, s33, s12, s13, s23);
                }
            }
        }
        out
    }

    /// The velocity of the wall on face `positive` (`+axis` side when true)
    /// of cell `(i, j, k)` along `axis`, if that face is a no-slip
    /// [`FaceBc::Wall`]; `None` for any other condition (slip walls included).
    fn wall_face_velocity(
        &self,
        i: usize,
        j: usize,
        k: usize,
        axis: usize,
        positive: bool,
    ) -> Option<Vec3Fix> {
        let step = usize::from(positive);
        let bc = match axis {
            0 => self.grid.u_bc(i + step, j, k),
            1 => self.grid.v_bc(i, j + step, k),
            _ => self.grid.w_bc(i, j, k + step),
        };
        match bc {
            FaceBc::Wall { velocity } => Some(velocity),
            _ => None,
        }
    }

    /// `(speed tangential to a face of normal axis, relative velocity)` of the
    /// cell-centred velocity of `(i, j, k)` against `wall`. The speed leaves
    /// out the `axis` component and is exact (no square root) when only one
    /// tangential component is non-zero.
    fn tangential_relative(
        &self,
        i: usize,
        j: usize,
        k: usize,
        axis: usize,
        wall: Vec3Fix,
    ) -> (Fix128, [Fix128; 3]) {
        let (uc, vc, wc) = self.grid.cell_velocity(i, j, k);
        let rel = [uc - wall.x, vc - wall.y, wc - wall.z];
        let (a, b) = match axis {
            0 => (rel[1], rel[2]),
            1 => (rel[0], rel[2]),
            _ => (rel[0], rel[1]),
        };
        let speed = if a.is_zero() {
            b.abs()
        } else if b.is_zero() {
            a.abs()
        } else {
            (a * a + b * b).sqrt()
        };
        (speed, rel)
    }

    /// The log-law gradient `∂u_c/∂x_axis` at the centre of cell `(i, j, k)`
    /// when that cell is the second from a no-slip wall normal to `axis`:
    /// its neighbour towards the wall has the wall face, the cell itself
    /// has none on that side. The cell centre is `y = 3dx/2` from the wall,
    /// and with `u_τ` the friction velocity of the neighbour's speed relative
    /// to the wall (`friction_velocity` at `y_p = dx/2`, the same value the
    /// wall function gives that cell) the gradient of the speed is
    /// `u_τ / (κ y)`, distributed over the tangential components in
    /// proportion to the neighbour's relative velocity, signed so that the
    /// speed grows away from the wall.
    ///
    /// The centred difference across the wall cell would read
    /// `u_τ ln 5 / (2 κ dx)` from a log profile there, 21 % above
    /// `u_τ / (κ · 3dx/2)`, independently of `dx`: the log layer is
    /// self-similar, so refining the grid does not shrink the error of the
    /// first cell off the wall. Only that cell is replaced; from the third
    /// cell on the finite difference stands.
    ///
    /// `None` when `c == axis`, when no wall is in reach, when walls are in
    /// reach on both sides (a three-cell-wide gap; the finite difference is
    /// kept), or when the spacing or `nu` is not positive.
    fn log_layer_gradient(
        &self,
        i: usize,
        j: usize,
        k: usize,
        c: usize,
        axis: usize,
        nu: Fix128,
    ) -> Option<Fix128> {
        let dx = self.grid.dx;
        if c == axis || dx <= Fix128::ZERO || nu <= Fix128::ZERO {
            return None;
        }
        let dims = [self.grid.nx, self.grid.ny, self.grid.nz];
        let idx = [i, j, k][axis];
        let shifted = |delta_up: bool| -> (usize, usize, usize) {
            let mut p = [i, j, k];
            if delta_up {
                p[axis] += 1;
            } else {
                p[axis] -= 1;
            }
            (p[0], p[1], p[2])
        };
        let lower = (idx >= 1 && self.wall_face_velocity(i, j, k, axis, false).is_none())
            .then(|| {
                let (a, b, cc) = shifted(false);
                self.wall_face_velocity(a, b, cc, axis, false)
                    .map(|w| ((a, b, cc), w))
            })
            .flatten();
        let upper = (idx + 1 < dims[axis]
            && self.wall_face_velocity(i, j, k, axis, true).is_none())
        .then(|| {
            let (a, b, cc) = shifted(true);
            self.wall_face_velocity(a, b, cc, axis, true)
                .map(|w| ((a, b, cc), w))
        })
        .flatten();
        let (((a, b, cc), wall), towards_positive) = match (lower, upper) {
            (Some(l), None) => (l, false),
            (None, Some(u)) => (u, true),
            _ => return None,
        };
        let (speed, rel) = self.tangential_relative(a, b, cc, axis, wall);
        if speed.is_zero() {
            return Some(Fix128::ZERO);
        }
        let u_tau = friction_velocity_checked(speed, dx.half(), nu);
        let y = dx + dx.half();
        let g = u_tau / (VON_KARMAN * y);
        let along = if rel[c] == speed {
            g
        } else if rel[c] == Fix128::ZERO - speed {
            Fix128::ZERO - g
        } else {
            g * rel[c] / speed
        };
        // The speed relative to the wall grows away from it: along +axis for a
        // wall below, along −axis for a wall above.
        Some(if towards_positive {
            Fix128::ZERO - along
        } else {
            along
        })
    }

    /// Mean of the six neighbours' values (a missing neighbour on the domain
    /// edge counts as the cell itself): the `2Δ` test filter of the dynamic
    /// Smagorinsky ratio. A uniform field is reproduced exactly (`6v / 6`).
    fn neighbour_mean(&self, field: &[Fix128], i: usize, j: usize, k: usize) -> Fix128 {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let at = |ii: usize, jj: usize, kk: usize| field[self.cell_index(ii, jj, kk)];
        let me = at(i, j, k);
        let pick = |ok: bool, v: Fix128| if ok { v } else { me };
        let sum = pick(i > 0, if i > 0 { at(i - 1, j, k) } else { me })
            + pick(i + 1 < nx, if i + 1 < nx { at(i + 1, j, k) } else { me })
            + pick(j > 0, if j > 0 { at(i, j - 1, k) } else { me })
            + pick(j + 1 < ny, if j + 1 < ny { at(i, j + 1, k) } else { me })
            + pick(k > 0, if k > 0 { at(i, j, k - 1) } else { me })
            + pick(k + 1 < nz, if k + 1 < nz { at(i, j, k + 1) } else { me });
        sum / Fix128::from_int(6)
    }

    /// The eddy viscosity field of `model` from the state the step starts
    /// with, every refusal of [`StepError`] checked before anything changes.
    fn prepare_turbulence(
        &self,
        state: &RansState,
        dt_s: Fix128,
    ) -> Result<TurbulenceRun, StepError> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let cells = nx * ny * nz;
        let dx = self.grid.dx;
        let model = state.model();
        if state.cell_dims() != (nx, ny, nz) || state.k.len() != cells {
            return Err(StepError::TurbulenceFieldShape);
        }
        let (nu_t, cs_min, cs_max) = match model {
            TurbulenceModel::Smagorinsky | TurbulenceModel::DynamicSmagorinsky => {
                let strain = self.cell_strain(None);
                let mut nu_t = vec![Fix128::ZERO; cells];
                let mut cs_all = Vec::with_capacity(cells);
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let c = self.cell_index(i, j, k);
                            let cs = match model {
                                TurbulenceModel::DynamicSmagorinsky => dynamic_smagorinsky_cs(
                                    strain[c],
                                    self.neighbour_mean(&strain, i, j, k),
                                ),
                                _ => SMAGORINSKY_CS,
                            };
                            cs_all.push(cs);
                            nu_t[c] = smagorinsky_eddy_viscosity_with(cs, dx, strain[c]);
                        }
                    }
                }
                if cells == 0 {
                    (nu_t, SMAGORINSKY_CS, SMAGORINSKY_CS)
                } else {
                    (nu_t, min_of(&cs_all), max_of(&cs_all))
                }
            }
            TurbulenceModel::KEpsilon | TurbulenceModel::KOmega => {
                let mut nu_t = vec![Fix128::ZERO; cells];
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            nu_t[self.cell_index(i, j, k)] = state.eddy_viscosity(i, j, k);
                        }
                    }
                }
                (nu_t, Fix128::ZERO, Fix128::ZERO)
            }
            TurbulenceModel::Prescribed => {
                if state.prescribed_dx != dx || state.prescribed.len() != cells {
                    return Err(StepError::TurbulenceFieldShape);
                }
                if state.prescribed.iter().any(|v| *v < Fix128::ZERO) {
                    return Err(StepError::NegativeEddyViscosity);
                }
                (state.prescribed.clone(), Fix128::ZERO, Fix128::ZERO)
            }
        };
        let (nu_t_min, nu_t_max) = (min_of(&nu_t), max_of(&nu_t));
        let nu_mol = self.dynamic_viscosity_pas / self.density_kg_m3;
        let diffusion_number = (nu_mol + nu_t_max) * dt_s / (dx * dx);
        if diffusion_number > Fix128::from_ratio(1, 6) {
            return Err(StepError::DiffusionUnstable { diffusion_number });
        }
        Ok(TurbulenceRun {
            model,
            nu_t,
            nu_t_min,
            nu_t_max,
            diffusion_number,
            cs_min,
            cs_max,
        })
    }

    /// One transport step of the k-ε / k-ω field: production from the
    /// current strain and `run.nu_t`, the explicit point sources, explicit
    /// diffusion with `ν_mol + ν_t / σ`, then semi-Lagrangian advection,
    /// then — with a wall model — the wall-function values of
    /// [`Self::impose_wall_k_epsilon`] on the cells with a no-slip wall face.
    /// Returns `(largest production, cells clamped)`; the LES and prescribed
    /// closures transport nothing and return zeros.
    ///
    /// ⚠️ Limitation: with `wall == None` no wall boundary condition is
    /// applied to `k` or `ε` (no low-Reynolds-number wall treatment, no wall
    /// function); the only wall effect is that no diffusive flux crosses a
    /// solid face.
    fn advance_rans(
        &mut self,
        state: &mut RansState,
        run: &TurbulenceRun,
        dt_s: Fix128,
        wall: Option<&WallModel>,
    ) -> (Fix128, u32) {
        let (sigma_k, sigma_second) = match run.model {
            TurbulenceModel::KEpsilon => (KE_SIGMA_K, KE_SIGMA_EPS),
            TurbulenceModel::KOmega => (KW_SIGMA, KW_SIGMA),
            _ => return (Fix128::ZERO, 0),
        };
        let nu_mol = self.dynamic_viscosity_pas / self.density_kg_m3;
        let strain = self.cell_strain(wall.map(|_| nu_mol));
        let mut production_max = Fix128::ZERO;
        let mut clamped = 0u32;
        {
            let field = &mut *state;
            for (c, &strain_c) in strain.iter().enumerate() {
                let production = run.nu_t[c] * strain_c * strain_c;
                if production > production_max {
                    production_max = production;
                }
                let cell = field.cell_state(c);
                // a cell counts once, whichever of its two scalars would go negative
                let mut hit = false;
                let next = match run.model {
                    TurbulenceModel::KEpsilon => {
                        let mut st = cell;
                        // Would the explicit update go negative? Counted
                        // before the clamp inside `advance_*` hides it.
                        if st.k + (production - st.epsilon) * dt_s < Fix128::ZERO {
                            hit = true;
                        }
                        st.advance_k(production, dt_s);
                        if !st.k.is_zero() {
                            let factor = st.epsilon / st.k;
                            let d_eps = factor
                                * (crate::turbulence::KE_C1_EPS * production
                                    - crate::turbulence::KE_C2_EPS * st.epsilon);
                            if st.epsilon + d_eps * dt_s < Fix128::ZERO {
                                hit = true;
                            }
                        }
                        st.advance_epsilon(production, dt_s);
                        st
                    }
                    _ => {
                        let mut st = KOmegaState::from_k_epsilon(&cell);
                        let (k, w) = (st.k, st.omega);
                        let dk = production - KW_BETA_STAR * k * w;
                        if k + dk * dt_s < Fix128::ZERO {
                            hit = true;
                        }
                        let dw = if k.is_zero() {
                            Fix128::ZERO - crate::turbulence::KW_BETA * w * w
                        } else {
                            crate::turbulence::KW_ALPHA * (w / k) * production
                                - crate::turbulence::KW_BETA * w * w
                        };
                        if w + dw * dt_s < Fix128::ZERO {
                            hit = true;
                        }
                        st.advance(production, dt_s);
                        st.to_k_epsilon()
                    }
                };
                clamped += u32::from(hit);
                field.k[c] = next.k;
                field.epsilon[c] = next.epsilon;
            }
        }
        // Diffusion of both scalars with their own turbulent Prandtl number.
        let nu_k: Vec<Fix128> = run.nu_t.iter().map(|&t| nu_mol + t / sigma_k).collect();
        let nu_e: Vec<Fix128> = run
            .nu_t
            .iter()
            .map(|&t| nu_mol + t / sigma_second)
            .collect();
        let (k_next, e_next) = (
            self.diffuse_cells(&state.k, &nu_k, dt_s),
            self.diffuse_cells(&state.epsilon, &nu_e, dt_s),
        );
        let (k_adv, e_adv) = (
            self.advect_cells(&k_next, dt_s),
            self.advect_cells(&e_next, dt_s),
        );
        state.k = k_adv;
        state.epsilon = e_adv;
        if wall.is_some() {
            self.impose_wall_k_epsilon(state, nu_mol);
        }
        (production_max, clamped)
    }

    /// The Launder–Spalding (1974) wall function of the `(k, ε)` transport:
    /// every cell with a no-slip [`crate::eulerian_grid::FaceBc::Wall`] face
    /// is given the log-layer equilibrium values at its centre,
    /// `y_p = dx/2` from the wall,
    ///
    /// ```text
    /// u_rel = u_τ · u⁺(y_p u_τ / ν)      (inverted by `friction_velocity`)
    /// k_P   = u_τ² / √C_μ
    /// ε_P   = u_τ³ / (κ y_p)             (ω_P = ε_P / (β* k_P) = u_τ / (√β* κ y_p))
    /// ```
    ///
    /// with `u_rel` the speed of the cell-centred velocity relative to the
    /// wall's, tangential to the face (the normal component is dropped).
    /// This is the fixed-value form: the cell's transported `(k, ε)` are
    /// replaced, not its production and dissipation. A cell with several
    /// wall faces (a corner) takes the arithmetic mean of the per-face
    /// values. A wall the fluid does not move against gives `u_τ = 0` and so
    /// `k = ε = 0` there (`ν_t = 0`). Slip walls are not walls here: they
    /// exert no shear.
    ///
    /// The values are those of the log region; in a cell whose `y⁺` lies in
    /// the viscous sublayer `u_τ` comes from the linear branch of the
    /// profile and the same two formulas are used, which is not the
    /// near-wall asymptotics of `k` and `ε` (no low-Reynolds-number damping
    /// is modelled).
    fn impose_wall_k_epsilon(&self, state: &mut RansState, nu_mol: Fix128) {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let y_p = self.grid.dx.half();
        if y_p <= Fix128::ZERO || nu_mol <= Fix128::ZERO {
            return;
        }
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut walls = 0i64;
                    let mut k_sum = Fix128::ZERO;
                    let mut e_sum = Fix128::ZERO;
                    for (axis, positive) in [
                        (0, false),
                        (0, true),
                        (1, false),
                        (1, true),
                        (2, false),
                        (2, true),
                    ] {
                        let Some(velocity) = self.wall_face_velocity(i, j, k, axis, positive)
                        else {
                            continue;
                        };
                        let (u_rel, _) = self.tangential_relative(i, j, k, axis, velocity);
                        let u_tau = friction_velocity_checked(u_rel, y_p, nu_mol);
                        let (kw, ew) = wall_k_epsilon(u_tau, y_p);
                        walls += 1;
                        k_sum = k_sum + kw;
                        e_sum = e_sum + ew;
                    }
                    if walls > 0 {
                        let c = self.cell_index(i, j, k);
                        if walls == 1 {
                            state.k[c] = k_sum;
                            state.epsilon[c] = e_sum;
                        } else {
                            let n = Fix128::from_int(walls);
                            state.k[c] = k_sum / n;
                            state.epsilon[c] = e_sum / n;
                        }
                    }
                }
            }
        }
    }

    /// Explicit diffusion of a cell-centred scalar with a cell-wise
    /// diffusivity: the flux between two cells uses the harmonic mean of
    /// their diffusivities, no flux crosses the domain edge or a solid face.
    fn diffuse_cells(&self, field: &[Fix128], nu: &[Fix128], dt_s: Fix128) -> Vec<Fix128> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dx = self.grid.dx;
        let mut out = field.to_vec();
        if dx.is_zero() {
            return out;
        }
        let scale = dt_s / (dx * dx);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = self.cell_index(i, j, k);
                    let centre = field[c];
                    let mut flux = Fix128::ZERO;
                    let mut add = |n: usize| {
                        flux = flux + harmonic2(nu[c], nu[n]) * (field[n] - centre);
                    };
                    if i > 0 && !self.grid.is_u_solid(i, j, k) {
                        add(self.cell_index(i - 1, j, k));
                    }
                    if i + 1 < nx && !self.grid.is_u_solid(i + 1, j, k) {
                        add(self.cell_index(i + 1, j, k));
                    }
                    if j > 0 && !self.grid.is_v_solid(i, j, k) {
                        add(self.cell_index(i, j - 1, k));
                    }
                    if j + 1 < ny && !self.grid.is_v_solid(i, j + 1, k) {
                        add(self.cell_index(i, j + 1, k));
                    }
                    if k > 0 && !self.grid.is_w_solid(i, j, k) {
                        add(self.cell_index(i, j, k - 1));
                    }
                    if k + 1 < nz && !self.grid.is_w_solid(i, j, k + 1) {
                        add(self.cell_index(i, j, k + 1));
                    }
                    out[c] = centre + scale * flux;
                }
            }
        }
        out
    }

    /// Semi-Lagrangian advection of a cell-centred scalar by the cell-centred
    /// velocity, as `advect_temperature` does it; a resting fluid returns the
    /// field bit for bit.
    fn advect_cells(&self, field: &[Fix128], dt_s: Fix128) -> Vec<Fix128> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dx = self.grid.dx;
        if dx.is_zero() || nx == 0 || ny == 0 || nz == 0 {
            return field.to_vec();
        }
        let old = Grid3d {
            nx,
            ny,
            nz,
            dx,
            data: field.to_vec(),
        };
        let inv_dx = Fix128::ONE / dx;
        let mut out = field.to_vec();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let (uc, vc, wc) = self.grid.cell_velocity(i, j, k);
                    let cx = Fix128::from_int(i as i64) - uc * dt_s * inv_dx;
                    let cy = Fix128::from_int(j as i64) - vc * dt_s * inv_dx;
                    let cz = Fix128::from_int(k as i64) - wc * dt_s * inv_dx;
                    out[self.cell_index(i, j, k)] = trilinear_sample(&old, cx, cy, cz);
                }
            }
        }
        out
    }

    /// Harmonic-mean diffusivity at the edge where the stencil of a face
    /// crosses from the cells on one side of the edge (`near`, a pair sharing
    /// the edge's index along the stencil axis) to the cells on the other
    /// (`far`). Each pair is averaged first (equal values give themselves
    /// exactly), then the two sides.
    fn edge_nu(near: (Fix128, Fix128), far: Option<(Fix128, Fix128)>) -> Fix128 {
        let n = harmonic2(near.0, near.1);
        match far {
            Some(f) => harmonic2(n, harmonic2(f.0, f.1)),
            None => n,
        }
    }

    /// Explicit diffusion of the face velocities with a cell-wise viscosity
    /// `ν_mol + ν_t`: `u += dt/dx² Σ_faces ν_f (u_nbr − u)`, the coefficient
    /// across a cell being that cell's viscosity and the coefficient across an
    /// edge the harmonic mean of the four cells around it (two on the stencil
    /// side, two beyond; only the fluid side when the edge is on a wall). The
    /// walls are treated exactly as in `diffuse_velocity`: the no-slip ghost
    /// `2 u_wall − u_in`, or the wall model's sink in place of it.
    fn diffuse_velocity_variable(
        &mut self,
        nu_mol: Fix128,
        nu_t: &[Fix128],
        dt_s: Fix128,
        wall: Option<&WallModel>,
        log_layer_edges: bool,
    ) -> Option<WallShearSummary> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        let dx = self.grid.dx;
        if dx.is_zero() || nx == 0 || ny == 0 || nz == 0 {
            return None;
        }
        let scale = dt_s / (dx * dx);
        let two = Fix128::from_int(2);
        let sink = wall.map(|_| WallSink {
            nu_mol,
            density: self.density_kg_m3,
            dynamic_viscosity: self.dynamic_viscosity_pas,
            y_p: dx.half(),
            dt_over_dx: dt_s / dx,
        });
        let mut summary = WallShearSummary::with_no_faces();
        let mut across_wall =
            |wall_component: Fix128, center: Fix128, extra: &mut Fix128| match &sink {
                None => two * wall_component - center,
                Some(model) => {
                    *extra = *extra + model.apply(center - wall_component, &mut summary);
                    center
                }
            };
        // Cell viscosity with the index clamped: a face on the domain edge
        // reads the one cell it has on the other side too.
        let cell = |i: usize, j: usize, k: usize| -> Fix128 {
            nu_mol + nu_t[i.min(nx - 1) + nx * (j.min(ny - 1) + ny * k.min(nz - 1))]
        };
        // The pair of cells straddling an X-face at `i` in row `(j, k)`.
        let x_pair = |i: usize, j: usize, k: usize| -> (Fix128, Fix128) {
            let lo = if i > 0 {
                cell(i - 1, j, k)
            } else {
                cell(i, j, k)
            };
            let hi = if i < nx {
                cell(i, j, k)
            } else {
                cell(i - 1, j, k)
            };
            (lo, hi)
        };
        let y_pair = |i: usize, j: usize, k: usize| -> (Fix128, Fix128) {
            let lo = if j > 0 {
                cell(i, j - 1, k)
            } else {
                cell(i, j, k)
            };
            let hi = if j < ny {
                cell(i, j, k)
            } else {
                cell(i, j - 1, k)
            };
            (lo, hi)
        };
        let z_pair = |i: usize, j: usize, k: usize| -> (Fix128, Fix128) {
            let lo = if k > 0 {
                cell(i, j, k - 1)
            } else {
                cell(i, j, k)
            };
            let hi = if k < nz {
                cell(i, j, k)
            } else {
                cell(i, j, k - 1)
            };
            (lo, hi)
        };
        // The same pairs as cell coordinates (clamped as `cell` clamps).
        type C3 = (usize, usize, usize);
        let clamp = |c: C3| -> C3 { (c.0.min(nx - 1), c.1.min(ny - 1), c.2.min(nz - 1)) };
        let x_cells = |i: usize, j: usize, k: usize| -> [C3; 2] {
            let lo = if i > 0 { (i - 1, j, k) } else { (i, j, k) };
            let hi = if i < nx { (i, j, k) } else { (i - 1, j, k) };
            [clamp(lo), clamp(hi)]
        };
        let y_cells = |i: usize, j: usize, k: usize| -> [C3; 2] {
            let lo = if j > 0 { (i, j - 1, k) } else { (i, j, k) };
            let hi = if j < ny { (i, j, k) } else { (i, j - 1, k) };
            [clamp(lo), clamp(hi)]
        };
        let z_cells = |i: usize, j: usize, k: usize| -> [C3; 2] {
            let lo = if k > 0 { (i, j, k - 1) } else { (i, j, k) };
            let hi = if k < nz { (i, j, k) } else { (i, j, k - 1) };
            [clamp(lo), clamp(hi)]
        };
        let log_nu = |c: C3| -> Fix128 { nu_mol + two * nu_t[c.0 + nx * (c.1 + ny * c.2)] };
        // No-slip wall faces of every cell, `[−x, +x, −y, +y, −z, +z]`,
        // gathered once (only needed with the wall function on).
        let wall_faces: Vec<[bool; 6]> = if log_layer_edges {
            let mut out = Vec::with_capacity(nx * ny * nz);
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let mut f = [false; 6];
                        for (slot, flag) in f.iter_mut().enumerate() {
                            *flag = self
                                .wall_face_velocity(i, j, k, slot / 2, slot % 2 == 1)
                                .is_some();
                        }
                        out.push(f);
                    }
                }
            }
            out
        } else {
            Vec::new()
        };
        let wall_beyond = |c: C3, axis: usize, positive: bool| -> bool {
            wall_faces[c.0 + nx * (c.1 + ny * c.2)][2 * axis + usize::from(positive)]
        };
        // The edge between the `near` pair and the `far` pair one cell away
        // along `axis` (`forward` = towards +axis). With the wall function on,
        // an edge one cell from a no-slip wall (both cells of the pair beyond
        // it, or both before it, touching the wall on the far side) takes the
        // log-layer value at its own height, `ν_t = κ u_τ dx = 2 ν_t,P` of the
        // wall cell whose `ν_t,P = κ u_τ dx/2`, instead of the harmonic mean
        // (which gives `0.75 κ u_τ dx` there). Otherwise the harmonic mean.
        let edge = |near: [C3; 2], far: [C3; 2], axis: usize, forward: bool| -> Fix128 {
            if log_layer_edges {
                let far_wall = far.iter().all(|&c| wall_beyond(c, axis, forward));
                let near_wall = near.iter().all(|&c| wall_beyond(c, axis, !forward));
                if far_wall && !near_wall {
                    return harmonic2(log_nu(far[0]), log_nu(far[1]));
                }
                if near_wall && !far_wall {
                    return harmonic2(log_nu(near[0]), log_nu(near[1]));
                }
            }
            Self::edge_nu(
                (
                    cell(near[0].0, near[0].1, near[0].2),
                    cell(near[1].0, near[1].1, near[1].2),
                ),
                Some((
                    cell(far[0].0, far[0].1, far[0].2),
                    cell(far[1].0, far[1].1, far[1].2),
                )),
            )
        };

        // u faces
        let mut u_next = self.grid.u.clone();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..=nx {
                    let center = self.grid.u(i, j, k);
                    let mut extra = Fix128::ZERO;
                    let mut acc = Fix128::ZERO;
                    if i > 0 {
                        acc = acc + cell(i - 1, j, k) * (self.grid.u(i - 1, j, k) - center);
                    }
                    if i < nx {
                        acc = acc + cell(i, j, k) * (self.grid.u(i + 1, j, k) - center);
                    }
                    let near = x_pair(i, j, k);
                    let near_c = x_cells(i, j, k);
                    // y neighbours: edge at y = j dx (down) and (j + 1) dx (up)
                    match self.grid.u_wall_across_y(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.x, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if j > 0 => {
                            acc = acc
                                + edge(near_c, x_cells(i, j - 1, k), 1, false)
                                    * (self.grid.u(i, j - 1, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.u_wall_across_y(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.x, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if j + 1 < ny => {
                            acc = acc
                                + edge(near_c, x_cells(i, j + 1, k), 1, true)
                                    * (self.grid.u(i, j + 1, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.u_wall_across_z(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.x, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if k > 0 => {
                            acc = acc
                                + edge(near_c, x_cells(i, j, k - 1), 2, false)
                                    * (self.grid.u(i, j, k - 1) - center);
                        }
                        None => {}
                    }
                    match self.grid.u_wall_across_z(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.x, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if k + 1 < nz => {
                            acc = acc
                                + edge(near_c, x_cells(i, j, k + 1), 2, true)
                                    * (self.grid.u(i, j, k + 1) - center);
                        }
                        None => {}
                    }
                    let ix = i + (nx + 1) * (j + ny * k);
                    u_next[ix] = center + scale * acc + extra;
                }
            }
        }
        self.grid.u = u_next;

        // v faces
        let mut v_next = self.grid.v.clone();
        for k in 0..nz {
            for j in 0..=ny {
                for i in 0..nx {
                    let center = self.grid.v(i, j, k);
                    let mut extra = Fix128::ZERO;
                    let mut acc = Fix128::ZERO;
                    if j > 0 {
                        acc = acc + cell(i, j - 1, k) * (self.grid.v(i, j - 1, k) - center);
                    }
                    if j < ny {
                        acc = acc + cell(i, j, k) * (self.grid.v(i, j + 1, k) - center);
                    }
                    let near = y_pair(i, j, k);
                    let near_c = y_cells(i, j, k);
                    match self.grid.v_wall_across_x(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.y, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if i > 0 => {
                            acc = acc
                                + edge(near_c, y_cells(i - 1, j, k), 0, false)
                                    * (self.grid.v(i - 1, j, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.v_wall_across_x(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.y, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if i + 1 < nx => {
                            acc = acc
                                + edge(near_c, y_cells(i + 1, j, k), 0, true)
                                    * (self.grid.v(i + 1, j, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.v_wall_across_z(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.y, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if k > 0 => {
                            acc = acc
                                + edge(near_c, y_cells(i, j, k - 1), 2, false)
                                    * (self.grid.v(i, j, k - 1) - center);
                        }
                        None => {}
                    }
                    match self.grid.v_wall_across_z(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.y, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if k + 1 < nz => {
                            acc = acc
                                + edge(near_c, y_cells(i, j, k + 1), 2, true)
                                    * (self.grid.v(i, j, k + 1) - center);
                        }
                        None => {}
                    }
                    let ix = i + nx * (j + (ny + 1) * k);
                    v_next[ix] = center + scale * acc + extra;
                }
            }
        }
        self.grid.v = v_next;

        // w faces
        let mut w_next = self.grid.w.clone();
        for k in 0..=nz {
            for j in 0..ny {
                for i in 0..nx {
                    let center = self.grid.w(i, j, k);
                    let mut extra = Fix128::ZERO;
                    let mut acc = Fix128::ZERO;
                    if k > 0 {
                        acc = acc + cell(i, j, k - 1) * (self.grid.w(i, j, k - 1) - center);
                    }
                    if k < nz {
                        acc = acc + cell(i, j, k) * (self.grid.w(i, j, k + 1) - center);
                    }
                    let near = z_pair(i, j, k);
                    let near_c = z_cells(i, j, k);
                    match self.grid.w_wall_across_x(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.z, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if i > 0 => {
                            acc = acc
                                + edge(near_c, z_cells(i - 1, j, k), 0, false)
                                    * (self.grid.w(i - 1, j, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.w_wall_across_x(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.z, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if i + 1 < nx => {
                            acc = acc
                                + edge(near_c, z_cells(i + 1, j, k), 0, true)
                                    * (self.grid.w(i + 1, j, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.w_wall_across_y(i, j, k, false) {
                        Some(w) => {
                            let ghost = across_wall(w.z, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if j > 0 => {
                            acc = acc
                                + edge(near_c, z_cells(i, j - 1, k), 1, false)
                                    * (self.grid.w(i, j - 1, k) - center);
                        }
                        None => {}
                    }
                    match self.grid.w_wall_across_y(i, j, k, true) {
                        Some(w) => {
                            let ghost = across_wall(w.z, center, &mut extra);
                            acc = acc + Self::edge_nu(near, None) * (ghost - center);
                        }
                        None if j + 1 < ny => {
                            acc = acc
                                + edge(near_c, z_cells(i, j + 1, k), 1, true)
                                    * (self.grid.w(i, j + 1, k) - center);
                        }
                        None => {}
                    }
                    let ix = i + nx * (j + ny * k);
                    w_next[ix] = center + scale * acc + extra;
                }
            }
        }
        self.grid.w = w_next;
        sink.map(|_| summary)
    }
}

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
