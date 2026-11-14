//! 2D material point method for soft solids (the mochi): particles carry mass, velocity, an
//! affine velocity and the deformation gradient; a regular grid of nodes carries the momentum
//! and the stress force each step.
//!
//! The method is the moving least squares material point method (MLS-MPM, Hu et al. 2018,
//! "A moving least squares material point method with displacement discontinuity and
//! two-way rigid body coupling", ACM TOG 37(4)) with quadratic B-spline weights and the
//! affine particle-in-cell transfer (APIC, Jiang et al. 2015), in plane strain, in `Q64.64`
//! throughout, so a run is bit for bit the same on every machine.
//!
//! # MODEL
//!
//! - **Equations.** Momentum balance of a continuum, `ρ Dv/Dt = ∇·σ + ρ g`, discretised by
//!   MLS-MPM. Particle to grid: `m_i = Σ_p w_ip m_p`, `(m v)_i = Σ_p w_ip (m_p v_p + Δt f_p + A_p
//!   (x_i − x_p))` (`f_p` the external force on the particle, [`Mpm2d::step_with_loads`], 0 in
//!   [`Mpm2d::step`]). An elastic foundation (Winkler backing, [`Mpm2d::foundation_loads`]):
//!   `f_p = A⁰_p (κ (a_p − x_p) − c v_p)`, `κ` N/m⁴, `c` N·s/m⁴, anchor `a_p`, damping ratio of
//!   the whole body `ζ = c / (2 √(κ ρ))`; a hand's patch ([`Mpm2d::particles_within`] a fixed
//!   radius, [`Mpm2d::shared_loads`]) shares its force `F` by rest area, `f_p = F A⁰_p / Σ A⁰`,
//!   adding up to `F` exactly; under a uniform load `ρ g` the body
//!   settles by `u = ρ g / κ` as a whole, and anchors `X_p − ρ g / κ` hold it at `X_p`. In the
//!   transfer `A_p = m_p C_p − Δt A⁰_p (4 / Δx²) τ_p` (`A⁰_p` the particle's rest
//!   area, `τ = P Fᵀ` the Kirchhoff stress). Grid: `v_i = (m v)_i / m_i + Δt g`, then the
//!   ground. Grid to particle: `v_p = Σ_i w_ip v_i`, `C_p = (4 / Δx²) Σ_i w_ip v_i (x_i −
//!   x_p)ᵀ`, `x_p += Δt v_p`, `F_p ← (I + Δt C_p) F_p`. Weights: quadratic B-splines,
//!   `w(f) = ½(1.5 − f)², ¾ − (f − 1)², ½(f − ½)²` per axis. Material: fixed co-rotated,
//!   `P = 2μ (F − R) + λ (J − 1) J F⁻ᵀ`, energy density `μ |F − R|² + λ (J − 1)² / 2`, `R` the
//!   rotation of the polar decomposition; or compressible neo-Hookean, `P = μ (F − F⁻ᵀ) + λ
//!   ln J F⁻ᵀ`, energy density `μ (tr FᵀF − 2) / 2 − μ ln J + λ (ln J)² / 2` ([`Mpm2dModel`]);
//!   `μ = E / (2 (1 + ν))`, `λ = E ν / ((1 + ν)(1 − 2ν))`. Viscoelastic (one Maxwell branch,
//!   [`Maxwell`]): `τ = R ((1 − g) τ̂ + g ĥ) Rᵀ` with `τ̂ = Rᵀ τ_e R` the elastic Kirchhoff
//!   stress in the unrotated frame and `ĥ + T dĥ/dt = T dτ̂/dt`, integrated by
//!   `ĥ ← e^(−Δt/T) ĥ + (1 − e^(−Δt/T)) (T / Δt) (τ̂ − τ̂_last)` (exact for `τ̂` linear over
//!   the step); under a held load the material creeps like the standard linear solid,
//!   relaxed modulus `E (1 − g)`, creep time `T / (1 − g)`.
//!   Implicit ([`Mpm2d::step_implicit`], backward Euler): the grid velocities `v` minimise the
//!   incremental potential `Σ_i ½ m_i |v_i − v*_i|² + Σ_p (A⁰_p k ψ(F_p(v)) + Δt K_p : C_p(v))`,
//!   `v*` the grid velocity without the stress (momentum, gravity, loads), `F_p(v) = (I + Δt
//!   C_p(v)) F_p`, `C_p(v)` the APIC matrix the grid to particle transfer builds from `v`; with
//!   a Maxwell branch `k = 1 − g + g γ` and `K_p` the branch's held stress (rotated with the
//!   start rotation, the one approximation). Solved by Newton iterations: the direction by the
//!   conjugate gradient with the exact Hessian (each particle's `∂P/∂F` along the direction by
//!   a central difference) and the Jacobi preconditioner, falling back to the isotropic
//!   linear stiffness where the curvature is not positive; a backtracking line search on the
//!   potential (near convergence, a full step that lowers the residual); the ground's nodes as
//!   a contact set (held at `v_y = 0` while pressed, released when the potential pulls them
//!   up). It stops when the residual's mass-weighted RMS velocity is at most the tolerance
//!   ([`ImplicitSolve`]); then the explicit step's grid to particle transfer.
//! - **Units.** SI: m, s, kg, Pa; plane strain, so masses, momenta and energies are per metre
//!   of depth (kg/m, kg/s, J/m).
//! - **Conservation.** Mass is carried by the particles and conserved exactly. Linear momentum
//!   and the APIC angular momentum `Σ m (x × v) + m (Δx² / 4)(C₁₀ − C₀₁)` are conserved by the
//!   transfers and the internal stress in exact arithmetic; in `Q64.64` the grid's division of
//!   momentum by mass floors, and the drift stays within a rounding budget (measured 1e-15
//!   kg/s per metre after 48 steps of a 64-particle blob). Energy is not conserved: symplectic
//!   Euler's energy oscillates within a period and APIC dissipates (measured on the column:
//!   above the start by 3.6e-5 of it in the first period only, then about 3e-5 of it lost per
//!   period); the ground removes the downward velocity of the nodes on it.
//! - **Range.** Explicit: stable for a step below the cell crossing time of the elastic wave
//!   (`Δt < Δx / c`, the oracles use a third of it). Every particle's 3×3 stencil must lie in
//!   the grid ([`Mpm2dError::OutOfGrid`], the step does nothing); a step that would make
//!   `det F ≤ 0` is refused ([`Mpm2dError::Inverted`]); a refused step changes nothing. 2D plane strain only; `0 ≤ ν < 0.5`.
//!   Implicit: stable for any step; it damps the modes the step does not resolve (backward
//!   Euler's amplitude factor `1 / √(1 + (ωΔt)²)` per step), so a step of a frame settles a
//!   soft body quasi-statically; it converges at first order to the explicit run as the step
//!   shrinks. A solve that does not reach the tolerance is refused
//!   ([`Mpm2dError::NotConverged`]). Cost (measured, a 0.91 m square block, E 2e6 Pa,
//!   Jacobi preconditioner): about 10 times the explicit step's per frame at cells 0.02 and
//!   0.04 m (`tests/analytic_mpm2d_implicit.rs::measure_the_cost_per_frame_of_a_letter_sized_block`).
//! - **Approximations.** Plane strain (no out-of-plane strain). The fixed co-rotated model
//!   is linear elasticity under rotation at small strain and its own large-strain law beyond
//!   it. One material per world. The ground is frictionless and only stops downward motion
//!   (no adhesion); there are no side walls. No plasticity. The viscosity is one Maxwell
//!   branch that relaxes shear and volume alike (Poisson's ratio stays fixed); several
//!   branches (a Prony series) are not offered. The energy reported for a viscoelastic
//!   material leaves out the branch's stored part. Contact between two bodies of particles is
//!   the grid's (they stick where their stencils overlap).
//! - **Validation.** `tests/analytic_mpm2d.rs::a_free_blob_falls_on_the_discrete_parabola::a_spinning_free_blob_keeps_its_momenta::a_column_settles_by_the_plane_strain_bar_value::a_column_oscillates_with_the_quarter_wave_period::the_energy_of_the_column_never_grows::two_runs_end_in_the_same_state::invalid_worlds_are_refused::a_refused_step_changes_nothing::a_seeded_disk_has_the_disk_area::a_restored_snapshot_replays_bit_for_bit::a_viscoelastic_column_creeps_like_the_standard_linear_solid::a_zero_fraction_branch_is_the_elastic_material::a_force_on_some_particles_moves_the_centre_of_mass_by_newtons_law::no_loads_is_the_plain_step::a_load_on_no_particle_is_refused::a_uniform_load_on_a_foundation_moves_the_body_by_load_over_stiffness::anchors_at_the_hanging_equilibrium_hold_the_body_at_rest::foundation_anchors_must_match_the_particles::a_damped_foundation_decays_by_the_logarithmic_decrement::a_patch_takes_the_particles_within_its_radius::a_shared_force_adds_up_to_the_force_exactly`, `tests/analytic_mpm2d_implicit.rs::a_free_blob_falls_on_the_discrete_parabola::a_column_at_frame_steps_settles_by_the_plane_strain_bar_value::backward_euler_does_not_create_energy::the_implicit_run_approaches_the_explicit_one_as_the_step_shrinks::a_viscoelastic_column_at_frame_steps_creeps_like_the_standard_linear_solid::a_spinning_free_blob_keeps_its_momentum_within_the_tolerance::implicit_runs_are_deterministic_and_replay_from_a_snapshot::a_solve_that_cannot_converge_is_refused_and_changes_nothing`
//!   (in order: the centre of mass on symplectic Euler's discrete parabola to 1e-12 m; momenta
//!   within the rounding budget; the time-averaged settlement of a self-weight column within
//!   5 % of `ρ g H² (1 − ν²) / (2 E)`; its first axial period within 5 % of `4 H / c`; each
//!   period's largest energy at most the previous one's; two runs give the same digest;
//!   invalid worlds refused; a refused step leaves every particle as it was; a seeded disk has the disk's area; a snapshot restored mid-run replays bit for bit; the viscoelastic column's settlement within 5 % of `δ∞ (1 − g e^(−t/τc))` at four times; a branch of fraction 0 is bit for bit the elastic material; a force on part of a free blob gives the momentum `n Δt F` and the centre of mass `c0 + Δt² (F / M) n (n + 1) / 2` to 1e-12, loads on one particle add; no loads is the plain step bit for bit; a load on no particle is refused; a body on a foundation under gravity settles by `ρ g / κ` on average over four periods of `√(κ / ρ)` within 2 %, unstrained; anchors at the hanging equilibrium hold it within 1e-9 m; anchors not one a particle are refused; with damping, the free decay's logarithmic decrement is `2π ζ / √(1 − ζ²)` within 3 %; a patch takes exactly the particles within its radius; a shared force's parts are in proportion to area and add up to the force bit for bit).
//! - **Tiers.** L0 only.
//!

use crate::math::Fix128;

mod implicit;
pub use implicit::{ImplicitReport, ImplicitSolve};

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A 2D vector `[x, y]`
pub type V2 = [Fix128; 2];
/// A 2×2 matrix, row major: `m[r][c]`
pub type M2 = [[Fix128; 2]; 2];

/// The grid and the step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpm2dConfig {
    /// The position of node `(0, 0)`, m
    pub origin: V2,
    /// The node spacing, m
    pub cell: Fix128,
    /// Nodes along x and y
    pub nodes: [usize; 2],
    /// Gravity, m/s²
    pub gravity: V2,
    /// The step, s
    pub dt: Fix128,
    /// A frictionless ground at this height, m: the nodes at or below it cannot move down
    pub ground: Option<Fix128>,
}

/// The hyperelastic law of a material.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mpm2dModel {
    /// Fixed co-rotated: `P = 2μ (F − R) + λ (J − 1) J F⁻ᵀ`, energy density
    /// `μ |F − R|² + λ (J − 1)² / 2`
    #[default]
    FixedCorotated,
    /// Compressible neo-Hookean (plane strain): `P = μ (F − F⁻ᵀ) + λ ln J F⁻ᵀ`, energy density
    /// `μ (tr FᵀF − 2) / 2 − μ ln J + λ (ln J)² / 2`
    NeoHookean,
}

/// One Maxwell branch of a viscoelastic material: the stress is
/// `(1 − fraction) τ_e + fraction h`, where `τ_e` is the elastic Kirchhoff stress of the
/// current deformation and `h` relaxes towards it, `ḣ + h / relaxation_time = τ̇_e`. Under a
/// held deformation the stress relaxes from `τ_e` to `(1 − fraction) τ_e` with the time
/// constant `relaxation_time`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Maxwell {
    /// The part of the instantaneous stiffness that relaxes, in `[0, 1)`
    pub fraction: Fix128,
    /// The relaxation time, s, `> 0`
    pub relaxation_time: Fix128,
}

/// An isotropic elastic material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mpm2dMaterial {
    /// Young's modulus, Pa
    pub youngs: Fix128,
    /// Poisson's ratio, in `[0, 0.5)`
    pub poisson: Fix128,
    /// Density, kg/m³ (per unit thickness: masses are kg per metre of depth)
    pub density: Fix128,
    /// The hyperelastic law
    pub model: Mpm2dModel,
    /// A Maxwell branch (viscoelastic), or none (elastic)
    pub maxwell: Option<Maxwell>,
}

/// One material point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MpmParticle2 {
    /// Position, m
    pub x: V2,
    /// Velocity, m/s
    pub v: V2,
    /// Affine velocity (the APIC matrix), 1/s
    pub c: M2,
    /// Deformation gradient
    pub f: M2,
    /// Mass per metre of depth, kg/m
    pub mass: Fix128,
    /// Area at rest, m²
    pub area0: Fix128,
    /// The Maxwell branch's stress in the particle's unrotated frame (`Rᵀ h R`), Pa
    pub branch: M2,
    /// The elastic Kirchhoff stress of the last step in the unrotated frame, Pa
    pub branch_ref: M2,
}

/// Why a world cannot be made or stepped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mpm2dError {
    /// A configuration or material value out of its range (the field named)
    Invalid(&'static str),
    /// A particle's 3×3 stencil leaves the grid
    OutOfGrid {
        /// The particle index
        particle: usize,
    },
    /// A particle's deformation gradient has `det F <= 0`
    Inverted {
        /// The particle index
        particle: usize,
    },
    /// The implicit step's solve did not reach its tolerance ([`ImplicitSolve`])
    NotConverged {
        /// The Newton iterations it ran
        newton: u32,
    },
}

/// A 2D MPM world: one material, its particles and the grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mpm2d {
    config: Mpm2dConfig,
    material: Mpm2dMaterial,
    particles: Vec<MpmParticle2>,
}

/// The 3×3 quadratic B-spline stencil of one particle: the lower-left node and the weights
/// along each axis.
struct Stencil {
    base: [usize; 2],
    /// the particle's position in cell units from `base`, in `[0.5, 1.5)`
    frac: V2,
    w: [[Fix128; 3]; 2],
}

impl Stencil {
    fn new(cfg: &Mpm2dConfig, inv_dx: Fix128, x: V2) -> Option<Self> {
        let half = Fix128::from_ratio(1, 2);
        let mut base = [0usize; 2];
        let mut frac = [Fix128::ZERO; 2];
        let mut w = [[Fix128::ZERO; 3]; 2];
        for k in 0..2 {
            let g = (x[k] - cfg.origin[k]) * inv_dx;
            let b = (g - half).hi;
            if b < 0 || (b as u64).checked_add(3)? > cfg.nodes[k] as u64 {
                return None;
            }
            base[k] = b as usize;
            let f = g - Fix128::from_int(b);
            frac[k] = f;
            let a = Fix128::from_ratio(3, 2) - f;
            let m = f - Fix128::ONE;
            let c = f - half;
            w[k] = [half * a * a, Fix128::from_ratio(3, 4) - m * m, half * c * c];
        }
        Some(Self { base, frac, w })
    }

    /// node `(base + (a, b))` minus the particle, m
    fn dpos(&self, cfg: &Mpm2dConfig, a: usize, b: usize) -> V2 {
        [
            (Fix128::from_int(a as i64) - self.frac[0]) * cfg.cell,
            (Fix128::from_int(b as i64) - self.frac[1]) * cfg.cell,
        ]
    }
}

fn mul(a: M2, b: M2) -> M2 {
    let mut m = [[Fix128::ZERO; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            m[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j];
        }
    }
    m
}

fn det(f: M2) -> Fix128 {
    f[0][0] * f[1][1] - f[0][1] * f[1][0]
}

/// The rotation of the polar decomposition `F = R S` of a 2×2 `F` with `det F > 0`:
/// `R = [[c, −s], [s, c]]`, `(c, s) = (F₀₀ + F₁₁, F₁₀ − F₀₁) / |…|`.
fn rotation(f: M2) -> M2 {
    let c = f[0][0] + f[1][1];
    let s = f[1][0] - f[0][1];
    let n = (c * c + s * s).sqrt();
    if n == Fix128::ZERO {
        return identity();
    }
    let (c, s) = (c / n, s / n);
    [[c, -s], [s, c]]
}

/// The Kirchhoff stress `τ = P Fᵀ` of `model` (`J F⁻ᵀ` is the cofactor matrix of `F`).
/// Neo-Hookean: `τ = μ (F Fᵀ − I) + λ ln J I`.
fn kirchhoff(model: Mpm2dModel, f: M2, mu: Fix128, lambda: Fix128) -> M2 {
    let j = det(f);
    if model == Mpm2dModel::NeoHookean {
        let ft = [[f[0][0], f[1][0]], [f[0][1], f[1][1]]];
        let b = mul(f, ft);
        let l = lambda * j.ln();
        return [
            [mu * (b[0][0] - Fix128::ONE) + l, mu * b[0][1]],
            [mu * b[1][0], mu * (b[1][1] - Fix128::ONE) + l],
        ];
    }
    let r = rotation(f);
    let cof = [[f[1][1], -f[1][0]], [-f[0][1], f[0][0]]];
    let two_mu = Fix128::from_int(2) * mu;
    let l = lambda * (j - Fix128::ONE);
    let mut p = [[Fix128::ZERO; 2]; 2];
    for i in 0..2 {
        for k in 0..2 {
            p[i][k] = two_mu * (f[i][k] - r[i][k]) + l * cof[i][k];
        }
    }
    let ft = [[f[0][0], f[1][0]], [f[0][1], f[1][1]]];
    mul(p, ft)
}

const fn identity() -> M2 {
    [[Fix128::ONE, Fix128::ZERO], [Fix128::ZERO, Fix128::ONE]]
}

impl Mpm2d {
    /// A world with no particle. Refused: a cell or step `<= 0`, fewer than 3 nodes along an
    /// axis, `E <= 0`, `ν` outside `[0, 0.5)`, density `<= 0`.
    pub fn new(config: Mpm2dConfig, material: Mpm2dMaterial) -> Result<Self, Mpm2dError> {
        if config.cell <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("cell"));
        }
        if config.dt <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("dt"));
        }
        if config.nodes[0] < 3 || config.nodes[1] < 3 {
            return Err(Mpm2dError::Invalid("nodes"));
        }
        if material.youngs <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("youngs"));
        }
        if material.poisson < Fix128::ZERO || material.poisson >= Fix128::from_ratio(1, 2) {
            return Err(Mpm2dError::Invalid("poisson"));
        }
        if material.density <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("density"));
        }
        if let Some(m) = material.maxwell {
            if m.fraction < Fix128::ZERO || m.fraction >= Fix128::ONE {
                return Err(Mpm2dError::Invalid("maxwell.fraction"));
            }
            if m.relaxation_time <= Fix128::ZERO {
                return Err(Mpm2dError::Invalid("maxwell.relaxation_time"));
            }
        }
        Ok(Self {
            config,
            material,
            particles: Vec::new(),
        })
    }

    /// The configuration
    #[must_use]
    pub const fn config(&self) -> &Mpm2dConfig {
        &self.config
    }

    /// The material
    #[must_use]
    pub const fn material(&self) -> &Mpm2dMaterial {
        &self.material
    }

    /// The particles, in the order they were added
    #[must_use]
    pub fn particles(&self) -> &[MpmParticle2] {
        &self.particles
    }

    /// Add a particle at rest shape (`F = I`, `C = 0`) of area `area0` at `x` moving at `v`;
    /// its mass is `density × area0`. Returns its index.
    pub fn add_particle(&mut self, x: V2, v: V2, area0: Fix128) -> usize {
        self.particles.push(MpmParticle2 {
            x,
            v,
            c: [[Fix128::ZERO; 2]; 2],
            f: identity(),
            mass: self.material.density * area0,
            area0,
            branch: [[Fix128::ZERO; 2]; 2],
            branch_ref: [[Fix128::ZERO; 2]; 2],
        });
        self.particles.len() - 1
    }

    /// Seed `per_axis × per_axis` particles in every cell whose sample points lie inside the
    /// signed distance `sdf` (negative inside), at rest; returns the number added.
    pub fn seed<F: Fn(V2) -> Fix128>(&mut self, sdf: F, per_axis: usize) -> usize {
        if per_axis == 0 {
            return 0;
        }
        let k = Fix128::from_int(per_axis as i64);
        let h = self.config.cell / k;
        let area = h * h;
        let before = self.particles.len();
        for i in 0..self.config.nodes[0] - 1 {
            for j in 0..self.config.nodes[1] - 1 {
                for a in 0..per_axis {
                    for b in 0..per_axis {
                        let x = [
                            self.config.origin[0]
                                + self.config.cell * Fix128::from_int(i as i64)
                                + h * Fix128::from_int(a as i64)
                                + h / Fix128::from_int(2),
                            self.config.origin[1]
                                + self.config.cell * Fix128::from_int(j as i64)
                                + h * Fix128::from_int(b as i64)
                                + h / Fix128::from_int(2),
                        ];
                        if sdf(x) < Fix128::ZERO {
                            self.add_particle(x, [Fix128::ZERO; 2], area);
                        }
                    }
                }
            }
        }
        self.particles.len() - before
    }

    /// Advance one step of [`Mpm2dConfig::dt`]: particle to grid (mass, APIC momentum, MLS
    /// stress force), grid update (gravity, ground), grid to particle (velocity, affine
    /// velocity, position, deformation gradient).
    pub fn step(&mut self) -> Result<(), Mpm2dError> {
        self.step_with_loads(&[])
    }

    /// The particles within `radius` of `centre` (a hand's contact patch), by index: the patch is a
    /// fixed physical size, so what it holds does not depend on the grid's cell.
    #[must_use]
    pub fn particles_within(&self, centre: V2, radius: Fix128) -> Vec<usize> {
        let r2 = radius * radius;
        self.particles
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let (dx, dy) = (p.x[0] - centre[0], p.x[1] - centre[1]);
                dx * dx + dy * dy <= r2
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// A force (N per metre of depth) shared over particles in proportion to their rest areas,
    /// as loads for [`Self::step_with_loads`]: a hand pulling on a patch. The parts add up to the
    /// force exactly (the last takes what the rounding of the others left), so the total applied
    /// is the hand's force at any resolution.
    ///
    /// # Errors
    ///
    /// [`Mpm2dError::Invalid`] for no particles, a particle that is not there, or no area.
    pub fn shared_loads(
        &self,
        particles: &[usize],
        force: V2,
    ) -> Result<Vec<(usize, V2)>, Mpm2dError> {
        let Some((&last, rest)) = particles.split_last() else {
            return Err(Mpm2dError::Invalid("shared: no particles"));
        };
        let mut area = Fix128::ZERO;
        for &i in particles {
            area = area
                + self
                    .particles
                    .get(i)
                    .ok_or(Mpm2dError::Invalid("shared: no such particle"))?
                    .area0;
        }
        if area <= Fix128::ZERO {
            return Err(Mpm2dError::Invalid("shared: no area"));
        }
        let mut loads = Vec::with_capacity(particles.len());
        let mut given = [Fix128::ZERO; 2];
        for &i in rest {
            let share = self.particles[i].area0 / area;
            let part = [force[0] * share, force[1] * share];
            given = [given[0] + part[0], given[1] + part[1]];
            loads.push((i, part));
        }
        loads.push((last, [force[0] - given[0], force[1] - given[1]]));
        Ok(loads)
    }

    /// The forces of an elastic foundation (a Winkler backing) on every particle: the backing
    /// pulls each particle toward its anchor and resists its motion, `A⁰ (κ (a − x) − c v)` (N per
    /// metre of depth), `κ` the foundation's stiffness per volume, N/m⁴, `c` its damping per
    /// volume, N·s/m⁴ (a viscous layer; 0 for none), `A⁰` the particle's rest area. The damping
    /// ratio of the body moving as a whole on it is `ζ = c / (2 √(κ ρ))`. Pass them to
    /// [`Self::step_with_loads`] (with any other loads). Under a uniform load `ρ g` the body
    /// settles by `ρ g / κ` as a whole; anchors set `ρ g / κ` against gravity from where the
    /// body rests carry its weight and hold it there.
    ///
    /// # Errors
    ///
    /// [`Mpm2dError::Invalid`] when there is not one anchor a particle, or `κ` or `c` is
    /// negative.
    pub fn foundation_loads(
        &self,
        anchors: &[V2],
        stiffness: Fix128,
        damping: Fix128,
    ) -> Result<Vec<(usize, V2)>, Mpm2dError> {
        if anchors.len() != self.particles.len() {
            return Err(Mpm2dError::Invalid("foundation: one anchor a particle"));
        }
        if stiffness < Fix128::ZERO {
            return Err(Mpm2dError::Invalid("foundation: stiffness"));
        }
        if damping < Fix128::ZERO {
            return Err(Mpm2dError::Invalid("foundation: damping"));
        }
        Ok(self
            .particles
            .iter()
            .zip(anchors)
            .enumerate()
            .map(|(k, (p, a))| {
                let (s, d) = (stiffness * p.area0, damping * p.area0);
                (
                    k,
                    [
                        s * (a[0] - p.x[0]) - d * p.v[0],
                        s * (a[1] - p.x[1]) - d * p.v[1],
                    ],
                )
            })
            .collect())
    }

    /// One step with external forces on particles: `(particle, force)`, N per metre of depth,
    /// held over the step. A load enters the grid with the particle's momentum (`Δt f` spread by
    /// the particle's weights), so a total force `F` changes the momentum by `Δt F` and the
    /// stress carries it through the body; several loads on one particle add. A load on a
    /// particle that is not there is refused ([`Mpm2dError::Invalid`]), and the step does
    /// nothing; otherwise as [`Self::step`].
    ///
    /// # Errors
    ///
    /// As [`Self::step`], and [`Mpm2dError::Invalid`] for a load on no particle.
    pub fn step_with_loads(&mut self, loads: &[(usize, V2)]) -> Result<(), Mpm2dError> {
        let mut force = vec![[Fix128::ZERO; 2]; self.particles.len()];
        for &(k, f) in loads {
            let at = force
                .get_mut(k)
                .ok_or(Mpm2dError::Invalid("loads: no such particle"))?;
            at[0] = at[0] + f[0];
            at[1] = at[1] + f[1];
        }
        let cfg = self.config;
        let (nx, ny) = (cfg.nodes[0], cfg.nodes[1]);
        let inv_dx = Fix128::ONE / cfg.cell;
        let four_inv_dx2 = Fix128::from_int(4) * inv_dx * inv_dx;
        let (mu, lambda) = self.lame();
        let dt = cfg.dt;
        // every particle's stencil first: a step either runs whole or not at all
        let mut stencils = Vec::with_capacity(self.particles.len());
        for (k, p) in self.particles.iter().enumerate() {
            let st =
                Stencil::new(&cfg, inv_dx, p.x).ok_or(Mpm2dError::OutOfGrid { particle: k })?;
            stencils.push(st);
        }
        // particle to grid: mass, APIC momentum and the MLS stress force
        let mut gm = vec![Fix128::ZERO; nx * ny];
        let mut gv = vec![[Fix128::ZERO; 2]; nx * ny];
        // the Maxwell branch: h ← e^(−Δt/T) h + γ (τ̂ − τ̂_last), γ = (1 − e^(−Δt/T)) T / Δt
        // (exact for τ̂ linear in time over the step), in the unrotated frame
        let decay = self
            .material
            .maxwell
            .filter(|m| m.fraction > Fix128::ZERO)
            .map(|m| ((-(dt / m.relaxation_time)).exp(), m));
        let mut branches = Vec::with_capacity(self.particles.len());
        for ((p, st), load) in self.particles.iter().zip(&stencils).zip(&force) {
            let mut tau = kirchhoff(self.material.model, p.f, mu, lambda);
            if let Some((e, m)) = decay {
                let r = rotation(p.f);
                let rt = [[r[0][0], r[1][0]], [r[0][1], r[1][1]]];
                let hat = mul(mul(rt, tau), r);
                let gamma = (Fix128::ONE - e) * m.relaxation_time / dt;
                let mut h = [[Fix128::ZERO; 2]; 2];
                let mut mix = [[Fix128::ZERO; 2]; 2];
                for i in 0..2 {
                    for k in 0..2 {
                        h[i][k] = e * p.branch[i][k] + gamma * (hat[i][k] - p.branch_ref[i][k]);
                        mix[i][k] = (Fix128::ONE - m.fraction) * hat[i][k] + m.fraction * h[i][k];
                    }
                }
                tau = mul(mul(r, mix), rt);
                branches.push((h, hat));
            } else {
                branches.push((p.branch, p.branch_ref));
            }
            // affine = −dt · area0 · 4/Δx² · τ + m C
            let s = -(dt * p.area0 * four_inv_dx2);
            let aff = [
                [
                    s * tau[0][0] + p.mass * p.c[0][0],
                    s * tau[0][1] + p.mass * p.c[0][1],
                ],
                [
                    s * tau[1][0] + p.mass * p.c[1][0],
                    s * tau[1][1] + p.mass * p.c[1][1],
                ],
            ];
            // the particle's momentum and the impulse of its load over the step
            let mv = [
                p.mass * p.v[0] + dt * load[0],
                p.mass * p.v[1] + dt * load[1],
            ];
            for a in 0..3 {
                for b in 0..3 {
                    let w = st.w[0][a] * st.w[1][b];
                    let d = st.dpos(&cfg, a, b);
                    let idx = (st.base[0] + a) * ny + st.base[1] + b;
                    gm[idx] = gm[idx] + w * p.mass;
                    gv[idx][0] = gv[idx][0] + w * (mv[0] + aff[0][0] * d[0] + aff[0][1] * d[1]);
                    gv[idx][1] = gv[idx][1] + w * (mv[1] + aff[1][0] * d[0] + aff[1][1] * d[1]);
                }
            }
        }
        // grid: momentum to velocity, gravity, the ground
        for i in 0..nx {
            for j in 0..ny {
                let idx = i * ny + j;
                if gm[idx] == Fix128::ZERO {
                    continue;
                }
                let mut v = [gv[idx][0] / gm[idx], gv[idx][1] / gm[idx]];
                v[0] = v[0] + dt * cfg.gravity[0];
                v[1] = v[1] + dt * cfg.gravity[1];
                if let Some(g) = cfg.ground {
                    let y = cfg.origin[1] + cfg.cell * Fix128::from_int(j as i64);
                    if y <= g && v[1] < Fix128::ZERO {
                        v[1] = Fix128::ZERO;
                    }
                }
                gv[idx] = v;
            }
        }
        // grid to particle: velocity, affine velocity, position, deformation gradient
        let mut next = Vec::with_capacity(self.particles.len());
        for (k, ((p, st), &(branch, branch_ref))) in self
            .particles
            .iter()
            .zip(&stencils)
            .zip(&branches)
            .enumerate()
        {
            let mut v = [Fix128::ZERO; 2];
            let mut b = [[Fix128::ZERO; 2]; 2];
            for a in 0..3 {
                for c in 0..3 {
                    let w = st.w[0][a] * st.w[1][c];
                    let d = st.dpos(&cfg, a, c);
                    let gvi = gv[(st.base[0] + a) * ny + st.base[1] + c];
                    v[0] = v[0] + w * gvi[0];
                    v[1] = v[1] + w * gvi[1];
                    for r in 0..2 {
                        for q in 0..2 {
                            b[r][q] = b[r][q] + w * gvi[r] * d[q];
                        }
                    }
                }
            }
            let c = [
                [four_inv_dx2 * b[0][0], four_inv_dx2 * b[0][1]],
                [four_inv_dx2 * b[1][0], four_inv_dx2 * b[1][1]],
            ];
            // F ← (I + dt C) F
            let g = [
                [Fix128::ONE + dt * c[0][0], dt * c[0][1]],
                [dt * c[1][0], Fix128::ONE + dt * c[1][1]],
            ];
            let f = mul(g, p.f);
            if det(f) <= Fix128::ZERO {
                return Err(Mpm2dError::Inverted { particle: k });
            }
            next.push(MpmParticle2 {
                x: [p.x[0] + dt * v[0], p.x[1] + dt * v[1]],
                v,
                c,
                f,
                branch,
                branch_ref,
                ..*p
            });
        }
        self.particles = next;
        Ok(())
    }

    /// The Lamé parameters `(μ, λ)` of the material (plane strain: `λ = E ν / ((1 + ν)(1 − 2ν))`)
    fn lame(&self) -> (Fix128, Fix128) {
        let (e, nu) = (self.material.youngs, self.material.poisson);
        let mu = e / (Fix128::from_int(2) * (Fix128::ONE + nu));
        let lambda = e * nu / ((Fix128::ONE + nu) * (Fix128::ONE - Fix128::from_int(2) * nu));
        (mu, lambda)
    }

    /// Total mass, kg/m
    #[must_use]
    pub fn mass(&self) -> Fix128 {
        self.particles.iter().fold(Fix128::ZERO, |a, p| a + p.mass)
    }

    /// Total linear momentum, kg/s per metre of depth
    #[must_use]
    pub fn momentum(&self) -> V2 {
        self.particles.iter().fold([Fix128::ZERO; 2], |a, p| {
            [a[0] + p.mass * p.v[0], a[1] + p.mass * p.v[1]]
        })
    }

    /// Total angular momentum about the origin, kg·m²/s per metre of depth, including the
    /// affine part of APIC: `m (x × v) + m D (C₁₀ − C₀₁)` with `D = cell² / 4` for quadratic
    /// B-splines (the momentum the particle hands to the grid).
    #[must_use]
    pub fn angular_momentum(&self) -> Fix128 {
        let d = self.config.cell * self.config.cell / Fix128::from_int(4);
        self.particles.iter().fold(Fix128::ZERO, |a, p| {
            a + p.mass * (p.x[0] * p.v[1] - p.x[1] * p.v[0]) + p.mass * d * (p.c[1][0] - p.c[0][1])
        })
    }

    /// Kinetic energy, J per metre of depth
    #[must_use]
    pub fn kinetic_energy(&self) -> Fix128 {
        self.particles.iter().fold(Fix128::ZERO, |a, p| {
            a + p.mass * (p.v[0] * p.v[0] + p.v[1] * p.v[1]) / Fix128::from_int(2)
        })
    }

    /// Stored elastic energy, J per metre of depth. With a Maxwell branch of fraction `g`,
    /// the long-term part `(1 − g) ψ(F)` only: the branch's own stored energy is not counted.
    #[must_use]
    pub fn elastic_energy(&self) -> Fix128 {
        let full = self.instantaneous_energy();
        match self.material.maxwell {
            Some(m) if m.fraction > Fix128::ZERO => (Fix128::ONE - m.fraction) * full,
            _ => full,
        }
    }

    /// `Σ area0 ψ(F)` with the instantaneous moduli
    fn instantaneous_energy(&self) -> Fix128 {
        let (mu, lambda) = self.lame();
        let model = self.material.model;
        self.particles.iter().fold(Fix128::ZERO, |acc, p| {
            if model == Mpm2dModel::NeoHookean {
                let tr = p.f[0][0] * p.f[0][0]
                    + p.f[0][1] * p.f[0][1]
                    + p.f[1][0] * p.f[1][0]
                    + p.f[1][1] * p.f[1][1];
                let lj = det(p.f).ln();
                let two = Fix128::from_int(2);
                return acc + p.area0 * (mu * (tr - two) / two - mu * lj + lambda * lj * lj / two);
            }
            let r = rotation(p.f);
            let mut dev = Fix128::ZERO;
            for (fr, rr) in p.f.iter().zip(&r) {
                for (&a, &b) in fr.iter().zip(rr) {
                    let d = a - b;
                    dev = dev + d * d;
                }
            }
            let j1 = det(p.f) - Fix128::ONE;
            acc + p.area0 * (mu * dev + lambda * j1 * j1 / Fix128::from_int(2))
        })
    }

    /// The first bytes of [`Mpm2d::snapshot`]. The layout, little endian: magic `b"AMP2"`,
    /// version `u16`; twelve `Fix128` (`hi: i64`, `lo: u64`): origin x and y, cell, gravity x
    /// and y, dt, ground (0 when none), Young's modulus, Poisson's ratio, density, the Maxwell
    /// fraction and relaxation time (0 when none); per particle twenty-two `Fix128`: x, v,
    /// mass, area0, C, F, the branch stress and its reference (each row major); then the
    /// nodes along x and y (`u64` each), the ground flag (`u8`), the model tag (`u8`, 0 fixed
    /// co-rotated, 1 neo-Hookean), the Maxwell flag (`u8`) and the particle count (`u64`).
    pub const SNAPSHOT_MAGIC: [u8; 4] = *b"AMP2";
    /// The snapshot format version
    pub const SNAPSHOT_VERSION: u16 = 1;

    /// The whole world as bytes: [`Self::restore`] makes a world that steps on bit for bit
    /// as this one.
    #[must_use]
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&Self::SNAPSHOT_MAGIC);
        out.extend_from_slice(&Self::SNAPSHOT_VERSION.to_le_bytes());
        let mut put = |x: Fix128| {
            out.extend_from_slice(&x.hi.to_le_bytes());
            out.extend_from_slice(&x.lo.to_le_bytes());
        };
        let c = &self.config;
        put(c.origin[0]);
        put(c.origin[1]);
        put(c.cell);
        put(c.gravity[0]);
        put(c.gravity[1]);
        put(c.dt);
        put(c.ground.unwrap_or(Fix128::ZERO));
        put(self.material.youngs);
        put(self.material.poisson);
        put(self.material.density);
        let mx = self.material.maxwell;
        put(mx.map_or(Fix128::ZERO, |m| m.fraction));
        put(mx.map_or(Fix128::ZERO, |m| m.relaxation_time));
        for p in &self.particles {
            for x in [p.x[0], p.x[1], p.v[0], p.v[1], p.mass, p.area0] {
                put(x);
            }
            for m in [p.c, p.f, p.branch, p.branch_ref] {
                for row in m {
                    for x in row {
                        put(x);
                    }
                }
            }
        }
        // the fixed-size tail holds the integers and the count that sizes the particle block
        let mut tail = Vec::new();
        tail.extend_from_slice(&(c.nodes[0] as u64).to_le_bytes());
        tail.extend_from_slice(&(c.nodes[1] as u64).to_le_bytes());
        tail.push(u8::from(c.ground.is_some()));
        tail.push(match self.material.model {
            Mpm2dModel::FixedCorotated => 0,
            Mpm2dModel::NeoHookean => 1,
        });
        tail.push(u8::from(mx.is_some()));
        tail.extend_from_slice(&(self.particles.len() as u64).to_le_bytes());
        out.extend_from_slice(&tail);
        out
    }

    /// The world [`Self::snapshot`] wrote. `None` for a wrong magic or version, a length that
    /// does not match the particle count, a ground flag or model tag out of range, or a
    /// configuration or material [`Self::new`] refuses.
    #[must_use]
    pub fn restore(data: &[u8]) -> Option<Self> {
        const HEAD: usize = 6;
        const NUMS: usize = 12;
        const PER: usize = 22;
        const TAIL: usize = 8 + 8 + 1 + 1 + 1 + 8;
        if data.len() < HEAD + NUMS * 16 + TAIL || data[0..4] != Self::SNAPSHOT_MAGIC {
            return None;
        }
        if u16::from_le_bytes([data[4], data[5]]) != Self::SNAPSHOT_VERSION {
            return None;
        }
        let tail = &data[data.len() - TAIL..];
        let u = |b: &[u8]| u64::from_le_bytes(b.try_into().expect("8 bytes"));
        let nodes = [
            usize::try_from(u(&tail[0..8])).ok()?,
            usize::try_from(u(&tail[8..16])).ok()?,
        ];
        let has_ground = match tail[16] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let model = match tail[17] {
            0 => Mpm2dModel::FixedCorotated,
            1 => Mpm2dModel::NeoHookean,
            _ => return None,
        };
        let has_maxwell = match tail[18] {
            0 => false,
            1 => true,
            _ => return None,
        };
        let count = usize::try_from(u(&tail[19..27])).ok()?;
        let want = count
            .checked_mul(PER * 16)?
            .checked_add(HEAD + NUMS * 16 + TAIL)?;
        if data.len() != want {
            return None;
        }
        let body = &data[HEAD..data.len() - TAIL];
        let fix = |k: usize| {
            let b = &body[16 * k..16 * k + 16];
            Fix128 {
                hi: i64::from_le_bytes(b[0..8].try_into().expect("8 bytes")),
                lo: u64::from_le_bytes(b[8..16].try_into().expect("8 bytes")),
            }
        };
        let ground = fix(6);
        if !has_ground && ground != Fix128::ZERO {
            return None;
        }
        let config = Mpm2dConfig {
            origin: [fix(0), fix(1)],
            cell: fix(2),
            nodes,
            gravity: [fix(3), fix(4)],
            dt: fix(5),
            ground: has_ground.then_some(ground),
        };
        let material = Mpm2dMaterial {
            youngs: fix(7),
            poisson: fix(8),
            density: fix(9),
            model,
            maxwell: if has_maxwell {
                Some(Maxwell {
                    fraction: fix(10),
                    relaxation_time: fix(11),
                })
            } else {
                if fix(10) != Fix128::ZERO || fix(11) != Fix128::ZERO {
                    return None;
                }
                None
            },
        };
        let mut world = Self::new(config, material).ok()?;
        for i in 0..count {
            let k = NUMS + PER * i;
            world.particles.push(MpmParticle2 {
                x: [fix(k), fix(k + 1)],
                v: [fix(k + 2), fix(k + 3)],
                mass: fix(k + 4),
                area0: fix(k + 5),
                c: [[fix(k + 6), fix(k + 7)], [fix(k + 8), fix(k + 9)]],
                f: [[fix(k + 10), fix(k + 11)], [fix(k + 12), fix(k + 13)]],
                branch: [[fix(k + 14), fix(k + 15)], [fix(k + 16), fix(k + 17)]],
                branch_ref: [[fix(k + 18), fix(k + 19)], [fix(k + 20), fix(k + 21)]],
            });
        }
        Some(world)
    }

    /// SHA-256 of [`Self::snapshot`]: equal for two worlds exactly when their configuration,
    /// material and particles are bit for bit equal
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        Sha256::digest(self.snapshot()).into()
    }
}
