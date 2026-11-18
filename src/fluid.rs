//! Position-Based Fluids (PBF)
//!
//! SPH-based fluid simulation using XPBD density constraints.
//! Supports SDF boundary containment and surface tension.
//!
//! # Algorithm (Macklin & Muller 2013)
//!
//! 1. Predict particle positions
//! 2. Iterate: rebuild the neighbour grid on the current predicted positions, compute
//!    density, solve density constraints
//! 3. Update velocities
//! 4. Apply viscosity and vorticity confinement
//!
//! # MODEL
//!
//! **Equations.** Macklin & Muller 2013, with the changes listed under "approximations":
//!
//! - density `rho_i = sum_j m W(x_i - x_j)` over fluid neighbours and fixed boundary
//!   particles (Akinci et al. 2012), poly6 kernel normalised to `integral W dV = 1`
//! - constraint `C_i = rho_i / rho_0 - 1`, gradient `grad_k C_i = (m / rho_0) grad W_ik`
//! - multiplier `lambda_i = -C_i / (S_i + eps * S_ref)`, `S_i = sum_k |grad_k C_i|^2`,
//!   `S_ref` = the same sum on the rest lattice (`rest_spacing`), so `eps` is unit-free
//! - correction `dx_i = (m / rho_0) sum_j (lambda_i + lambda_j) grad W_ij` (boundary
//!   particles contribute `lambda_j = 0`), applied Jacobi-style after every particle's
//!   `lambda` is known
//! - XSPH viscosity `v_i += c sum_j (m / rho_0) W_ij (v_j - v_i)` over fluid neighbours,
//!   an added kinematic viscosity of order `nu = c h^2 / dt`
//!
//! **Units.** SI (m, s, kg); `eps` and the surface weights are unit-free.
//!
//! **Conservation.** Total fluid momentum is conserved by the pairwise corrections; boundary
//! particles and density-map solids are fixed and take momentum out, as a wall does (the
//! force they take is reported by [`Fluid::boundary_force`] and [`Fluid::body_forces`]).
//! Mass is conserved exactly (particles are neither made nor removed). Kinetic energy is not
//! conserved: XSPH dissipates it at the rate of the added viscosity below.
//!
//! **Approximations.** Departures from the paper, each measured on a 4x8x4 column at
//! `h = 2 s` with the default iteration and substep counts:
//!
//! - lattice-calibrated mass: the default `particle_mass` makes the rest lattice
//!   exactly `rho_0` (`m s^3 = rho_0 s^3` gives `1.010 rho_0` at `h = 2 s`, and that 1 %
//!   pushes a free block upward without bound); see [`FluidConfig::lattice_mass`]
//! - `C >= 0` clamped everywhere, as in the paper: a liquid carries no tension beyond its
//!   cohesion, so a stretched region is pulled together only by the cohesion term, never by
//!   a projection of `C < 0`. Measured 2026-11-18 with a weighted two-sided branch (since
//!   removed): projecting `C < 0` on a block laid out 15 % wide gave it a kinetic energy of
//!   239 J after one frame, 5e4 times the 0.0047 J the cohesion potential can release, and
//!   threw particles out at up to 4.5 m/s; with the clamp the kinetic energy stays below the
//!   cohesion energy released at every frame and the block contracts monotonically.
//! - XSPH strength `c = 0.3` (paper `0.01`): measured as the largest speed over the
//!   last 60 of 300 frames of a column under gravity, `c = 0.1` gives 0.03-0.13 m/s and
//!   bursts to 0.5 m/s over 900 frames, `c = 0.2` gives 0.04-0.13 m/s, `c = 0.3` gives
//!   0.024 m/s (`s = 0.05`). More iterations (20) or substeps (4, 8) did not remove the
//!   chatter at `c = 0.1`.
//! - settling time grows with the scale at `c = 0.3`, `h = 2 s`, 2 substeps: largest
//!   speed over the last 60 frames of the 4x8x4 column after 300 / 600 / 900 frames is
//!   0.003 / 0.0000 / 0.0000 m/s at `s = 0.05`, 0.22 / 0.026 / 0.036 at `s = 0.075`,
//!   0.16 / 0.13 / 0.025 at `s = 0.1` (the default). Density error (0.2-0.3 %) and the
//!   floor reaction (weight to 0.3 %) do not depend on it.
//! - artificial pressure off by default: `s_corr` pushes every pair apart and makes a
//!   free block expand (1.68x its height in 60 frames); kept as `artificial_pressure`
//!   (`k`, with `n = 4` and `dq = 0.2 h`, in `lambda` units).
//! - contact with solids: a particle centre inside a density-map solid
//!   ([`Fluid::with_sdf_body`]) is put back on the solid's surface along its SDF normal and
//!   loses its normal velocity (inelastic; a liquid at a wall does not bounce), and the
//!   momentum change is credited to that solid (each pass to its own solid). A centre inside
//!   two solids, or in a concave corner of one, is resolved one axis per pass: the passes
//!   repeat until no centre is inside, at most 4 per substep (2 resolve an orthogonal corner,
//!   one axis each; 4 cover a corner where two solids meet a wall), and
//!   [`Fluid::contact_left_inside`] counts what the cap leaves. Without it the map density alone left the
//!   layer next to a wall 0.013-0.015 m inside it at rest (measured 2026-11-18). Container
//!   colliders ([`Fluid::step_with_sdf`]) keep their restitution 0.5 for now. Measured: with
//!   restitution 0.5 on the solids too, the bounce during settling changes the packing and
//!   the cube two kernel radii above the floor reads `F_y / (rho_0 g V)` = 0.924; inelastic,
//!   0.995.
//! - Newton 3 between the fluid and the solids holds exactly: the neighbour grid is rebuilt
//!   on the predicted positions at the start of every constraint iteration, so every pair
//!   is found from both sides and its push is equal and opposite; the fluid-side total
//!   ([`Fluid::boundary_force`]) and the solids' forces ([`Fluid::body_forces`]) then sum to
//!   zero in `Fix128`, on every frame. Measured (2026-11-18): with the grid built once per
//!   substep, the corrections moved the particles off the cells the grid was built on, a
//!   pair near a cell edge was found from one side only, and the sum was 0 to 3.3e-9 of the
//!   fluid-side total depending on the initial placement (3.4e-5 N of 2.98e4 N); rebuilt
//!   every iteration, exactly 0 on all 900 frames, at no measurable cost.
//! - the side-wall load is not isotropic at the floor/wall junction and under the free
//!   surface (measured 2026-11-18 on the 4x8x4 column after 900 frames): the x+ wall load
//!   in 0.1 m bins, divided by the vertical load `rho_0 g (H - y)` on the same bin, is
//!   1.203 / 1.136 in the bottom two bins, 1.00-1.01 at 0.2-0.5 m and 0.899 / 0.804 / 0.488
//!   under the surface (with the weighted two-sided branch: 1.150 / 1.130, 0.99-1.01,
//!   0.876 / 0.766 / 0.480). 4x the iterations cut the bottom density residual 6x and 2x the
//!   substeps 4x while the ratios stay within 0.05 of these values, so it is a property of
//!   the converged projection (one-sided support under the surface, the junction at the
//!   floor), not of convergence. The total over the wall (1.054; 1.035 two-sided) is read
//!   as a sum of these and hides the profile; the per-bin oracle reads the bins. The boundary itself is complete: two
//!   layers fill the kernel support at walls, edges and corners
//!   (`tests/analytic_fluid_boundary_support.rs`).
//! - frame damping is an artificial drag: `damping` (0.99 per frame) removes 1 % of the
//!   velocity every frame. Where the only driving force is weak (cohesion pulling a stretched
//!   block together, about 1e-3 m/s^2) it sets the speed (terminal drift `a dt / (1 - d)`,
//!   about 2e-3 m/s) and the time scale (a 0.05 m contraction takes about 3000 frames instead
//!   of about 560 under cohesion alone).
//!
//! **Range.** Weakly compressible water-like fluids at `h ~ 2 s`, frame `dt` up to 1/60 s
//! with two substeps; density error of a resting column under gravity is ~0.3 %. Not
//! valid for free surfaces without boundary particles below them (particles near an
//! unsupported wall see too few neighbours and sink).
//!
//! **Validation.** `tests/analytic_fluid_pbf_density.rs::rest_block_without_gravity_stays_at_rest::column_under_gravity_settles_and_the_floor_carries_its_weight::floor_reaction_is_zero_without_boundary_particles::default_mass_makes_the_rest_lattice_exactly_rho0::box_boundary_counts_match_the_closed_form::stretched_block_contracts_under_cohesion_within_its_energy_budget::stretched_block_drift_without_xsph_follows_the_damped_closed_form::floor_reaction_scales_with_the_particle_mass::side_wall_load_per_height_bin_matches_the_hydrostatic_profile::settled_column_does_not_depend_on_the_substep_count`: hydrostatics two ways, the floor
//! reaction against the weight M g (1.00006 at 900 frames) and the force on a side
//! wall against the integral of the linear pressure, rho_0 g H^2 D / 2 with
//! H = M / (rho_0 W D) (1.035; the floor balance holds for any pressure profile, the
//! wall force only for a linear one); column density; a resting block staying at rest;
//! a block laid out 15 % wide contracting to rho_0; the same answer at 4 substeps and
//! at 2.5x the mass.
//!
//! **Tiers.** L0 only.
//!
//! **Known defects.** `tests/analytic_fluid_pbf_density.rs::submerged_cube_feels_the_archimedes_force`:
//! a cube of boundary particles in place of fluid is pushed up by 1.535 rho_0 g V.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Vec3Fix};
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;
use crate::spatial::SpatialGrid;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Fluid Configuration
// ============================================================================

/// Fluid simulation configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FluidConfig {
    /// Rest density (kg/m^3)
    pub rest_density: Fix128,
    /// Smoothing kernel radius
    pub kernel_radius: Fix128,
    /// Number of solver iterations
    pub iterations: usize,
    /// Number of substeps
    pub substeps: usize,
    /// Gravity vector
    pub gravity: Vec3Fix,
    /// Velocity retention per frame (`step()` call), applied once per frame
    /// since 1.2.0 (per substep before, which made the result depend on `substeps`)
    pub damping: Fix128,
    /// Viscosity coefficient (XSPH)
    pub viscosity: Fix128,
    /// Vorticity confinement strength ε (`Δv = ε · dt · N × ω`, PBF §5)
    pub vorticity_strength: Fix128,
    /// Surface tension / cohesion coefficient κ
    /// (`Δv_i = κ · dt · Σ_j (m_j/ρ₀) (x_j − x_i)/r · W_ij`)
    pub surface_tension: Fix128,
    /// Particle mass (default: [`FluidConfig::lattice_mass`] of the rest lattice)
    pub particle_mass: Fix128,
    /// Constraint relaxation parameter (scales the position correction)
    pub relaxation: Fix128,
    /// Rest particle spacing `s`: sets the reference `S_ref` that makes the constraint
    /// softening unit-free (default `h / 2`)
    pub rest_spacing: Fix128,
    /// Constraint softening for `C >= 0`, relative to `S_ref`
    pub epsilon: Fix128,
    /// Artificial pressure `k` of `s_corr = -k (W(r) / W(0.2 h))^4`, in units of the
    /// reference multiplier `1 / S_ref`; `0` (default) disables it
    pub artificial_pressure: Fix128,
}

impl FluidConfig {
    /// Particle mass that makes a simple cubic lattice of spacing `spacing` exactly
    /// `rest_density`: `rho_0 / sum_k W(|k| spacing)` over the lattice points within
    /// `kernel_radius` (the origin included).
    ///
    /// Returns `rest_density * spacing^3` when the sum is zero (a kernel radius of
    /// zero) or `spacing` is not positive.
    #[must_use]
    pub fn lattice_mass(rest_density: Fix128, kernel_radius: Fix128, spacing: Fix128) -> Fix128 {
        if spacing <= Fix128::ZERO {
            return rest_density * spacing * spacing * spacing;
        }
        let sum = lattice_sum(kernel_radius, spacing, |r_sq| poly6(r_sq, kernel_radius));
        if sum.is_zero() {
            rest_density * spacing * spacing * spacing
        } else {
            rest_density / sum
        }
    }
}

impl Default for FluidConfig {
    fn default() -> Self {
        let rest_density = Fix128::from_int(1000);
        let kernel_radius = Fix128::from_ratio(1, 5); // 0.2
        let rest_spacing = Fix128::from_ratio(1, 10); // h / 2
        Self {
            rest_density,
            kernel_radius,
            iterations: 10,
            substeps: 2,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            viscosity: Fix128::from_ratio(3, 10),
            vorticity_strength: Fix128::from_ratio(1, 1000),
            surface_tension: Fix128::from_ratio(1, 100),
            particle_mass: Self::lattice_mass(rest_density, kernel_radius, rest_spacing),
            relaxation: Fix128::ONE,
            rest_spacing,
            epsilon: Fix128::from_ratio(1, 100),
            artificial_pressure: Fix128::ZERO,
        }
    }
}

/// `sum_k f(|k|^2 spacing^2)` over the integer lattice points `k` with
/// `|k| spacing < radius` (the origin included).
fn lattice_sum(radius: Fix128, spacing: Fix128, f: impl Fn(Fix128) -> Fix128) -> Fix128 {
    if spacing <= Fix128::ZERO || radius <= Fix128::ZERO {
        return Fix128::ZERO;
    }
    let r_sq = radius * radius;
    let mut reach: i64 = 0;
    while Fix128::from_int(reach + 1) * spacing < radius {
        reach += 1;
    }
    let mut sum = Fix128::ZERO;
    for a in -reach..=reach {
        for b in -reach..=reach {
            for c in -reach..=reach {
                let d_sq = Fix128::from_int(a * a + b * b + c * c) * spacing * spacing;
                if d_sq < r_sq {
                    sum = sum + f(d_sq);
                }
            }
        }
    }
    sum
}

// ============================================================================
// SPH Kernels (Deterministic)
// ============================================================================

/// Poly6 kernel (density estimation)
#[inline(always)]
fn poly6(r_sq: Fix128, h: Fix128) -> Fix128 {
    let h_sq = h * h;
    if r_sq >= h_sq {
        return Fix128::ZERO;
    }
    let diff = h_sq - r_sq;
    // W = 315 / (64 * pi * h^9) * (h^2 - r^2)^3, normalised so that the integral over
    // the ball of radius h is 1 (before 1.2.x the 1/pi was omitted and every density
    // came out pi times too large)
    let h9 = h_sq * h_sq * h_sq * h_sq * h;
    let coeff = Fix128::from_ratio(315, 64) / (h9 * Fix128::PI);
    coeff * diff * diff * diff
}

/// Spiky kernel gradient magnitude (pressure)
#[inline(always)]
fn spiky_grad(r: Fix128, h: Fix128) -> Fix128 {
    if r >= h || r.is_zero() {
        return Fix128::ZERO;
    }
    let diff = h - r;
    // grad_W = -45 / (pi * h^6) * (h - r)^2
    let h6 = h * h * h * h * h * h;
    let coeff = Fix128::from_ratio(45, 1) / (h6 * Fix128::PI);
    -coeff * diff * diff
}

/// The poly6 kernel, for the boundary density maps
#[cfg(feature = "std")]
pub(crate) fn poly6_kernel(r_sq: Fix128, h: Fix128) -> Fix128 {
    poly6(r_sq, h)
}

/// The spiky kernel gradient magnitude, for the boundary density maps
#[cfg(feature = "std")]
pub(crate) fn spiky_grad_kernel(r: Fix128, h: Fix128) -> Fix128 {
    spiky_grad(r, h)
}

// ============================================================================
// Fluid Simulation
// ============================================================================

/// Position-Based Fluid simulation
#[repr(C, align(64))]
pub struct Fluid {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Predicted positions
    predicted: Vec<Vec3Fix>,
    /// Velocities
    pub velocities: Vec<Vec3Fix>,
    /// Per-particle density
    pub densities: Vec<Fix128>,
    /// Per-particle lambda (constraint multiplier)
    lambdas: Vec<Fix128>,
    /// Spatial hash grid
    grid: SpatialGrid,
    /// Configuration
    pub config: FluidConfig,
    /// Cached reciprocal of `rest_density` to avoid repeated division in hot loops
    inv_rest_density: Fix128,
    /// Fixed boundary particles (Akinci et al. 2012): they add to the density and the
    /// constraint gradients of fluid particles near a wall but never move
    pub boundary: Vec<Vec3Fix>,
    /// `S_ref`: the constraint-gradient sum on the rest lattice
    reference_gradient_sq: Fix128,
    /// Force the boundary particles exerted on the fluid in the last substep
    boundary_force: Vec3Fix,
    /// Per boundary particle: the force it exerted on the fluid in the last substep
    boundary_forces: Vec<Vec3Fix>,
    /// Static solids given by their SDF, coupled through density maps
    /// (see [`crate::fluid_boundary_map`])
    #[cfg(feature = "std")]
    bodies: Vec<crate::fluid_boundary_map::FluidBody>,
    /// Per solid: the force the fluid exerted on it in the last substep
    body_forces: Vec<Vec3Fix>,
    /// Per particle and solid (`i * body_count + k`): the force the particle exerted on the
    /// solid in the last substep
    body_particle_forces: Vec<Vec3Fix>,
    /// Particle centres still inside a density-map solid after the last substep's contact
    contact_left_inside: usize,
}

impl Fluid {
    /// Create fluid with initial particle positions
    #[must_use]
    pub fn new(positions: Vec<Vec3Fix>, config: FluidConfig) -> Self {
        let n = positions.len();
        let grid_dim = 32;
        let inv_rest_density = if config.rest_density.is_zero() {
            Fix128::ONE
        } else {
            Fix128::ONE / config.rest_density
        };
        let h = config.kernel_radius;
        let vol = config.particle_mass * inv_rest_density;
        let reference_gradient_sq = lattice_sum(h, config.rest_spacing, |r_sq| {
            if r_sq.is_zero() {
                Fix128::ZERO
            } else {
                let g = spiky_grad(r_sq.sqrt(), h);
                g * g
            }
        }) * vol
            * vol;

        Self {
            predicted: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            densities: vec![Fix128::ZERO; n],
            lambdas: vec![Fix128::ZERO; n],
            grid: SpatialGrid::new(config.kernel_radius, grid_dim),
            positions,
            config,
            inv_rest_density,
            boundary: Vec::new(),
            reference_gradient_sq,
            boundary_force: Vec3Fix::ZERO,
            boundary_forces: Vec::new(),
            #[cfg(feature = "std")]
            bodies: Vec::new(),
            body_forces: Vec::new(),
            body_particle_forces: Vec::new(),
            contact_left_inside: 0,
        }
    }

    /// Add a static solid given by its SDF: its boundary density map is built over the box
    /// `[lo, hi]` (world space; it must contain every point within the kernel radius of the
    /// solid that fluid can reach) with grid step `rest_spacing / 4`. The solid then adds to
    /// the density and constraint gradient of nearby fluid particles, and the force the fluid
    /// exerts on it is read with [`Fluid::body_forces`] (see the MODEL of
    /// [`crate::fluid_boundary_map`]).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn with_sdf_body(mut self, sdf: SdfCollider, lo: Vec3Fix, hi: Vec3Fix) -> Self {
        let step = self.config.rest_spacing / Fix128::from_int(4);
        let map = crate::fluid_boundary_map::BoundaryMap::from_sdf(
            &sdf,
            lo,
            hi,
            step,
            self.config.rest_density,
            self.config.kernel_radius,
        );
        self.bodies
            .push(crate::fluid_boundary_map::FluidBody::new(sdf, map));
        self
    }

    /// Force (N) the fluid exerted on each solid added with [`Fluid::with_sdf_body`] during
    /// the last substep, in the order they were added. Their sum is minus the solids' part of
    /// [`Fluid::boundary_force`]. Empty before the first step.
    #[must_use]
    pub fn body_forces(&self) -> &[Vec3Fix] {
        &self.body_forces
    }

    /// Force (N) each particle exerted on each solid added with [`Fluid::with_sdf_body`]
    /// during the last substep (density map and contact), at index `i * body_count() + k`
    /// for particle `i` and solid `k`; summed over the particles it is
    /// [`Fluid::body_forces`] up to rounding. Empty before the first step.
    #[must_use]
    pub fn body_particle_forces(&self) -> &[Vec3Fix] {
        &self.body_particle_forces
    }

    /// Particle centres still inside a solid added with [`Fluid::with_sdf_body`] after the
    /// contact of the last substep: the contact repeats its passes until no centre is inside,
    /// at most a fixed number of times, and this is 0 unless that cap ran out
    #[must_use]
    pub fn contact_left_inside(&self) -> usize {
        self.contact_left_inside
    }

    /// Number of solids added with [`Fluid::with_sdf_body`]
    #[must_use]
    pub fn body_count(&self) -> usize {
        #[cfg(feature = "std")]
        {
            self.bodies.len()
        }
        #[cfg(not(feature = "std"))]
        {
            0
        }
    }

    /// `rho_B` and `grad rho_B` of every solid at `p`, appended to `out`
    #[allow(
        unused_variables,
        reason = "the arguments are read only with feature `std`, which holds the solids"
    )]
    fn sample_bodies(&self, p: Vec3Fix, out: &mut Vec<(Fix128, Vec3Fix)>) {
        #[cfg(feature = "std")]
        for body in &self.bodies {
            out.push(body.map().sample(p));
        }
    }

    /// Unit outward normal of solid `k` at `p`
    #[allow(
        unused_variables,
        reason = "the arguments are read only with feature `std`, which holds the solids"
    )]
    fn body_normal(&self, k: usize, p: Vec3Fix) -> Vec3Fix {
        #[cfg(feature = "std")]
        {
            crate::fluid_boundary_map::sdf_normal(self.bodies[k].sdf(), p)
        }
        #[cfg(not(feature = "std"))]
        {
            Vec3Fix::ZERO
        }
    }

    /// The share of `total` that solid `b` takes when two solids are near a particle (see
    /// [`crate::fluid_boundary_map::split_two`]); `None` keeps each solid's own push
    #[allow(
        unused_variables,
        reason = "the arguments are read only with feature `std`, which holds the solids"
    )]
    fn split_pair(total: Vec3Fix, n_a: Vec3Fix, n_b: Vec3Fix, share_b: Fix128) -> Option<Vec3Fix> {
        #[cfg(feature = "std")]
        {
            crate::fluid_boundary_map::split_two(total, n_a, n_b, share_b)
        }
        #[cfg(not(feature = "std"))]
        {
            None
        }
    }

    /// Force (N) the boundary particles exerted on the fluid during the last substep:
    /// `sum_i m (dx_i) / dt^2` over the constraint corrections. Fluid-fluid corrections
    /// are pairwise antisymmetric and cancel, so only the wall terms remain; for fluid at
    /// rest in a box this is the floor reaction, `-M g` (the integral of `rho g h` over
    /// the floor). Zero before the first step.
    #[must_use]
    pub fn boundary_force(&self) -> Vec3Fix {
        self.boundary_force
    }

    /// Force (N) each boundary particle exerted on the fluid during the last substep,
    /// in the order of [`Fluid::boundary`]: `sum_i m (dx_i from that particle) / dt^2`.
    /// Summing a wall's entries gives the pressure force on that wall (for fluid at rest,
    /// `1/2 rho g H^2 D` on a side wall of depth `D`); the sum over all entries is
    /// [`Fluid::boundary_force`] up to rounding. Empty before the first step.
    #[must_use]
    pub fn boundary_forces(&self) -> &[Vec3Fix] {
        &self.boundary_forces
    }

    /// Replace the fixed boundary particles (see [`Fluid::box_boundary`])
    #[must_use]
    pub fn with_boundary(mut self, boundary: Vec<Vec3Fix>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Boundary particles for an open-topped box `[min, max]` filled on the lattice
    /// `min + (i + 1/2) spacing`: `layers` shells of lattice points outside the box on
    /// the floor and the four walls, up to `max.y` (no lid).
    ///
    /// Returns no particles when `spacing` is not positive, `layers` is zero or the box
    /// is inverted on some axis.
    #[must_use]
    pub fn box_boundary(
        min: Vec3Fix,
        max: Vec3Fix,
        spacing: Fix128,
        layers: usize,
    ) -> Vec<Vec3Fix> {
        let mut out = Vec::new();
        if spacing <= Fix128::ZERO || layers == 0 || min.x > max.x || min.y > max.y || min.z > max.z
        {
            return out;
        }
        let cells = |lo: Fix128, hi: Fix128| -> i64 {
            let mut k: i64 = 0;
            while lo + Fix128::from_int(k + 1) * spacing <= hi {
                k += 1;
            }
            k
        };
        let (nx, ny, nz) = (
            cells(min.x, max.x),
            cells(min.y, max.y),
            cells(min.z, max.z),
        );
        let l = layers as i64;
        let half = Fix128::from_ratio(1, 2);
        let at = |o: Fix128, i: i64| o + (Fix128::from_int(i) + half) * spacing;
        for i in -l..nx + l {
            for j in -l..ny {
                for k in -l..nz + l {
                    let inside = (0..nx).contains(&i) && j >= 0 && (0..nz).contains(&k);
                    if !inside {
                        out.push(Vec3Fix::new(at(min.x, i), at(min.y, j), at(min.z, k)));
                    }
                }
            }
        }
        out
    }

    /// Create a block of fluid particles on a regular lattice
    ///
    /// Particles sit at `min + (i, j, k) * spacing` for every lattice point not beyond `max`
    /// (a point exactly on `max` is included), ordered x-major (x outer loop, z inner).
    /// An inverted box (`min > max` on some axis) yields no particles.
    ///
    /// # Panics
    ///
    /// Panics if `spacing` is not strictly positive: a zero or negative step never leaves
    /// the `<= max` loop and would exhaust memory.
    #[must_use]
    pub fn new_block(min: Vec3Fix, max: Vec3Fix, spacing: Fix128, config: FluidConfig) -> Self {
        assert!(
            spacing > Fix128::ZERO,
            "Fluid::new_block: spacing must be strictly positive"
        );
        let mut positions = Vec::new();

        let mut x = min.x;
        while x <= max.x {
            let mut y = min.y;
            while y <= max.y {
                let mut z = min.z;
                while z <= max.z {
                    positions.push(Vec3Fix::new(x, y, z));
                    z = z + spacing;
                }
                y = y + spacing;
            }
            x = x + spacing;
        }

        Self::new(positions, config)
    }

    /// Number of particles
    #[inline(always)]
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Step fluid simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
        self.apply_frame_damping();
    }

    /// Step with SDF boundary
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_boundary(sdf_colliders);
        }
        self.apply_frame_damping();
    }

    /// The spatial grid on the predicted fluid positions and the fixed boundary particles
    fn rebuild_grid(&mut self) {
        let n = self.particle_count();
        self.grid.clear();
        for i in 0..n {
            self.grid.insert(i, self.predicted[i]);
        }
        for (k, b) in self.boundary.iter().enumerate() {
            self.grid.insert(n + k, *b);
        }
        self.grid.build();
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let vol = self.particle_volume();

        let mut neighbors_buf = Vec::new();
        let mut body_buf: Vec<(Fix128, Vec3Fix)> = Vec::new();
        let n_bodies = self.body_count();

        // 1. Predict positions (damping is applied once per frame in `step`,
        //    1.2.0 — per substep it made the terminal velocity depend on
        //    `substeps`, the same defect as the rigid-body solver's R2-1)
        for i in 0..n {
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.predicted[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 3. Density constraint iterations (Jacobi: every lambda, then every correction)
        let s_ref = self.reference_gradient_sq;
        let artificial = self.config.artificial_pressure;
        let w_dq = poly6(h_sq * Fix128::from_ratio(1, 25), h);
        let mut corrections = vec![Vec3Fix::ZERO; n];
        let mut total_correction = Vec3Fix::ZERO;
        let mut wall_corrections = vec![Vec3Fix::ZERO; self.boundary.len()];
        let mut body_corrections = vec![Vec3Fix::ZERO; n_bodies];
        let mut particle_corrections = vec![Vec3Fix::ZERO; n * n_bodies];
        let mut body_samples = vec![(Fix128::ZERO, Vec3Fix::ZERO); n * n_bodies];
        if self.config.iterations == 0 {
            // the steps after the constraints (viscosity, cohesion, vorticity) read the grid
            self.rebuild_grid();
        }
        for _ in 0..self.config.iterations {
            // 2. Spatial grid on the predicted fluid positions and the boundary, rebuilt
            //    every iteration: the corrections move `predicted`, and a grid built on the
            //    positions of an earlier iteration finds a pair near a cell edge from one side
            //    only, so the pair's push is not equal and opposite (the fluid-side total and
            //    the solids' forces then differ; see MODEL "Newton 3")
            self.rebuild_grid();
            for i in 0..n {
                self.grid
                    .query_neighbors_into(self.predicted[i], h_sq, &mut neighbors_buf);
                let mut density = Fix128::ZERO;
                let mut sum_grad_sq = Fix128::ZERO;
                let mut grad_i = Vec3Fix::ZERO;

                for &j in &neighbors_buf {
                    let delta = self.predicted[i] - self.predicted_point(j);
                    let r_sq = delta.length_squared();

                    density = density + self.config.particle_mass * poly6(r_sq, h);

                    if j != i && r_sq < h_sq && !r_sq.is_zero() {
                        let r = r_sq.sqrt();
                        let grad = delta / r * (spiky_grad(r, h) * vol);
                        grad_i = grad_i + grad;
                        sum_grad_sq = sum_grad_sq + grad.length_squared();
                    }
                }

                if n_bodies > 0 {
                    body_buf.clear();
                    self.sample_bodies(self.predicted[i], &mut body_buf);
                    for (k, &(r_b, g_b)) in body_buf.iter().enumerate() {
                        density = density + r_b;
                        grad_i = grad_i + g_b * self.inv_rest_density;
                        body_samples[i * n_bodies + k] = (r_b, g_b);
                    }
                }

                self.densities[i] = density;
                sum_grad_sq = sum_grad_sq + grad_i.length_squared();

                // `C >= 0` (a liquid carries no tension beyond its cohesion)
                let mut constraint = density * self.inv_rest_density - Fix128::ONE;
                if constraint < Fix128::ZERO {
                    constraint = Fix128::ZERO;
                }
                let denom = sum_grad_sq + self.config.epsilon * s_ref;
                self.lambdas[i] = if sum_grad_sq.is_zero() || denom.is_zero() {
                    Fix128::ZERO
                } else {
                    -constraint / denom
                };
            }

            for (i, slot) in corrections.iter_mut().enumerate() {
                self.grid
                    .query_neighbors_into(self.predicted[i], h_sq, &mut neighbors_buf);
                let mut correction = Vec3Fix::ZERO;
                for &j in &neighbors_buf {
                    if j == i {
                        continue;
                    }
                    let delta = self.predicted[i] - self.predicted_point(j);
                    let r_sq = delta.length_squared();
                    if r_sq >= h_sq || r_sq.is_zero() {
                        continue;
                    }
                    let r = r_sq.sqrt();
                    let grad = delta / r * spiky_grad(r, h);
                    let mut lambda_sum = self.lambdas[i];
                    if j < n {
                        lambda_sum = lambda_sum + self.lambdas[j];
                    } else {
                        wall_corrections[j - n] = wall_corrections[j - n]
                            + grad * self.lambdas[i] * vol * self.config.relaxation;
                    }
                    if !artificial.is_zero() && !w_dq.is_zero() && !s_ref.is_zero() {
                        let ratio = poly6(r_sq, h) / w_dq;
                        let r2 = ratio * ratio;
                        lambda_sum = lambda_sum - artificial * r2 * r2 / s_ref;
                    }
                    correction = correction + grad * lambda_sum;
                }
                let mut correction = correction * vol * self.config.relaxation;
                if n_bodies > 0 {
                    let scale = self.lambdas[i] * self.inv_rest_density * self.config.relaxation;
                    let samples = &body_samples[i * n_bodies..(i + 1) * n_bodies];
                    let mut total = Vec3Fix::ZERO;
                    let mut near = [usize::MAX; 2];
                    let mut near_count = 0usize;
                    for (k, &(r_b, g_b)) in samples.iter().enumerate() {
                        total = total + g_b * scale;
                        if !r_b.is_zero() {
                            if near_count < 2 {
                                near[near_count] = k;
                            }
                            near_count += 1;
                        }
                    }
                    correction = correction + total;
                    // share the solids' push (see the fluid_boundary_map MODEL); the last
                    // share is the total minus the others, so the shares sum to it exactly
                    let mut split = None;
                    if near_count == 2 {
                        let (a, b) = (near[0], near[1]);
                        let (r_a, _) = samples[a];
                        let (r_b, _) = samples[b];
                        let share_b = r_b / (r_a + r_b);
                        let n_a = self.body_normal(a, self.predicted[i]);
                        let n_b = self.body_normal(b, self.predicted[i]);
                        if let Some(to_b) = Self::split_pair(total, n_a, n_b, share_b) {
                            split = Some((a, b, to_b));
                        }
                    }
                    if let Some((a, b, to_b)) = split {
                        body_corrections[b] = body_corrections[b] + to_b;
                        body_corrections[a] = body_corrections[a] + (total - to_b);
                        particle_corrections[i * n_bodies + b] =
                            particle_corrections[i * n_bodies + b] + to_b;
                        particle_corrections[i * n_bodies + a] =
                            particle_corrections[i * n_bodies + a] + (total - to_b);
                    } else {
                        let mut credited = Vec3Fix::ZERO;
                        let last = samples.len() - 1;
                        for (k, &(_, g_b)) in samples.iter().enumerate() {
                            let own = if k == last {
                                total - credited
                            } else {
                                g_b * scale
                            };
                            credited = credited + own;
                            body_corrections[k] = body_corrections[k] + own;
                            particle_corrections[i * n_bodies + k] =
                                particle_corrections[i * n_bodies + k] + own;
                        }
                    }
                }
                *slot = correction;
            }
            for (p, c) in self.predicted.iter_mut().zip(&corrections) {
                *p = *p + *c;
                total_correction = total_correction + *c;
            }
        }
        let force_per_correction = if dt.is_zero() {
            Fix128::ZERO
        } else {
            self.config.particle_mass / (dt * dt)
        };
        self.boundary_force = total_correction * force_per_correction;
        self.boundary_forces = wall_corrections
            .into_iter()
            .map(|c| c * force_per_correction)
            .collect();
        // the solids receive the reaction of the corrections they gave the fluid
        self.body_particle_forces = particle_corrections
            .into_iter()
            .map(|c| -(c * force_per_correction))
            .collect();
        self.body_forces = body_corrections
            .into_iter()
            .map(|c| -(c * force_per_correction))
            .collect();

        // 4. Update velocities and positions
        let inv_dt = Fix128::ONE / dt;
        for i in 0..n {
            self.velocities[i] = (self.predicted[i] - self.positions[i]) * inv_dt;
            self.positions[i] = self.predicted[i];
        }

        // 4b. Contact with the density-map solids: a centre inside a solid is put back on its
        //     surface and loses its normal velocity; the fluid's momentum change is the
        //     solid's reaction (Newton 3, credited per solid and per particle)
        #[cfg(feature = "std")]
        self.resolve_body_contacts(inv_dt);

        // 5. Apply viscosity (XSPH)
        self.apply_viscosity();

        // 6. Surface tension (cohesion acceleration toward neighbors)
        if !self.config.surface_tension.is_zero() {
            self.apply_surface_tension(dt);
        }

        // 7. Vorticity confinement (re-inject rotational energy lost to damping)
        if !self.config.vorticity_strength.is_zero() {
            self.apply_vorticity_confinement(dt);
        }
    }

    /// Predicted position of grid entry `j` (boundary particles do not move)
    #[inline]
    fn predicted_point(&self, j: usize) -> Vec3Fix {
        let n = self.predicted.len();
        if j < n {
            self.predicted[j]
        } else {
            self.boundary[j - n]
        }
    }

    /// `damping` once per frame (velocity retention per `step()` call).
    fn apply_frame_damping(&mut self) {
        let d = self.config.damping;
        for v in &mut self.velocities {
            *v = *v * d;
        }
    }

    /// Particle volume `m / ρ₀` — the SPH normalisation that turns a kernel
    /// sum `Σ_j W_ij` into a dimensionless number of order 1 (and `Σ_j ∇W_ij`
    /// into `O(1/h)`), so the cohesion and confinement terms below are
    /// accelerations with the documented coefficients, not raw kernel sums.
    #[inline]
    fn particle_volume(&self) -> Fix128 {
        self.config.particle_mass * self.inv_rest_density
    }

    /// XSPH viscosity: `v_i += c * sum_j (m/rho_0) W_ij (v_j - v_i)` over fluid
    /// neighbours (paper eq. 17 with the SPH volume, so `c` is unit-free)
    fn apply_viscosity(&mut self) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let c = self.config.viscosity * self.particle_volume();

        let velocities_copy: Vec<Vec3Fix> = self.velocities.clone();
        let mut neighbors_buf = Vec::new();

        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut sum = Vec3Fix::ZERO;

            for &j in &neighbors_buf {
                if i == j || j >= n {
                    continue;
                }
                let delta = self.positions[i] - self.positions[j];
                let r_sq = delta.length_squared();
                let w = poly6(r_sq, h);
                sum = sum + (velocities_copy[j] - velocities_copy[i]) * w;
            }

            self.velocities[i] = self.velocities[i] + sum * c;
        }
    }

    /// Surface tension via pairwise cohesion accelerations
    /// (Becker & Teschner 2007 form):
    /// `Δv_i = κ · dt · Σ_j (m_j / ρ₀) · (x_j − x_i)/r · W(r)`.
    ///
    /// Pairwise antisymmetric, so the centre of mass is untouched. Before
    /// 1.2.0 the un-normalised kernel sum (`W ≈ 260` at `r = h/2`,
    /// `h = 0.2`) was added straight to the velocity without `m/ρ₀` or `dt`,
    /// which threw a resting 5×5×5 block apart by ±4 m in a single frame at
    /// the default coefficient (`tests/default_configs.rs`).
    fn apply_surface_tension(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let coeff = self.config.surface_tension * dt * self.particle_volume();
        let mut neighbors_buf = Vec::new();

        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut force = Vec3Fix::ZERO;

            for &j in &neighbors_buf {
                if i == j || j >= n {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let w = poly6(r_sq, h);
                force = force + delta / r * w;
            }

            self.velocities[i] = self.velocities[i] + force * coeff;
        }
    }

    /// Vorticity confinement (Macklin & Müller 2013, §5):
    /// `ω_i = Σ_j (m_j/ρ₀) (v_j − v_i) × ∇W_ij`,
    /// `η_i = Σ_j (m_j/ρ₀) |ω_j| ∇W_ij`, `N = η/|η|`,
    /// `Δv_i = ε · dt · (N × ω_i)`.
    ///
    /// Same 1.2.0 normalisation fix as `apply_surface_tension`: the kernel
    /// sums carry `m/ρ₀` and the confinement acceleration is integrated over
    /// `dt` instead of being added to the velocity as-is.
    fn apply_vorticity_confinement(&mut self, dt: Fix128) {
        let n = self.particle_count();
        let h = self.config.kernel_radius;
        let h_sq = h * h;
        let vol = self.particle_volume();
        let epsilon = self.config.vorticity_strength * dt;
        let mut neighbors_buf = Vec::new();

        // Compute per-particle curl of velocity
        let mut curls: Vec<Vec3Fix> = vec![Vec3Fix::ZERO; n];
        for (i, curl_out) in curls.iter_mut().enumerate() {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut curl = Vec3Fix::ZERO;
            for &j in &neighbors_buf {
                if i == j || j >= n {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let grad_mag = spiky_grad(r, h) * vol;
                let grad = delta / r * grad_mag;
                let vel_diff = self.velocities[j] - self.velocities[i];
                curl = curl + vel_diff.cross(grad);
            }
            *curl_out = curl;
        }

        // Apply confinement force: f = epsilon * (N x omega)
        // where N = normalize(gradient of |omega|)
        for i in 0..n {
            self.grid
                .query_neighbors_into(self.positions[i], h_sq, &mut neighbors_buf);
            let mut grad_mag_curl = Vec3Fix::ZERO;
            for &j in &neighbors_buf {
                if i == j || j >= n {
                    continue;
                }
                let delta = self.positions[j] - self.positions[i];
                let r_sq = delta.length_squared();
                if r_sq.is_zero() || r_sq >= h_sq {
                    continue;
                }
                let r = r_sq.sqrt();
                let grad_w = spiky_grad(r, h) * vol;
                // PBF eq. 16 uses |ω_j| at the neighbour
                grad_mag_curl = grad_mag_curl + delta / r * (grad_w * curls[j].length());
            }
            if grad_mag_curl.length_squared().is_zero() {
                continue;
            }
            let n_vec = grad_mag_curl.normalize();
            let force = n_vec.cross(curls[i]) * epsilon;
            self.velocities[i] = self.velocities[i] + force;
        }
    }

    /// Resolve SDF boundary collisions
    #[cfg(feature = "std")]
    fn resolve_sdf_boundary(&mut self, sdf_colliders: &[SdfCollider]) {
        for i in 0..self.particle_count() {
            for sdf in sdf_colliders {
                // For containment: push particles inside if they escape
                if let Some((p, v)) = sdf_contact(
                    sdf,
                    self.positions[i],
                    self.velocities[i],
                    ContactSide::Container,
                ) {
                    self.positions[i] = p;
                    self.predicted[i] = p;
                    self.velocities[i] = v;
                }
            }
        }
    }

    /// Contact with the solids added by [`Fluid::with_sdf_body`]: a centre inside a solid
    /// (`SDF < 0`) goes back to its surface along the solid's normal and its normal velocity
    /// is removed (inelastic); `m dv / dt` of each contact is added to the
    /// force on the fluid and taken from that solid's force
    #[cfg(feature = "std")]
    fn resolve_body_contacts(&mut self, inv_dt: Fix128) {
        let m = self.config.particle_mass;
        let n_bodies = self.bodies.len();
        // a centre inside two solids at once (a concave corner, or where two solids meet)
        // leaves one projection on its other side; passes repeat until no centre is inside,
        // at most CONTACT_PASSES times (a fixed count, so the result is deterministic)
        for _ in 0..CONTACT_PASSES {
            let mut moved = false;
            for k in 0..n_bodies {
                let mut on_fluid = Vec3Fix::ZERO;
                for i in 0..self.particle_count() {
                    let sdf = self.bodies[k].sdf();
                    if let Some((p, v)) = sdf_contact(
                        sdf,
                        self.positions[i],
                        self.velocities[i],
                        ContactSide::Solid,
                    ) {
                        let push = (v - self.velocities[i]) * (m * inv_dt);
                        on_fluid = on_fluid + push;
                        if let Some(f) = self.body_particle_forces.get_mut(i * n_bodies + k) {
                            *f = *f - push;
                        }
                        self.positions[i] = p;
                        self.predicted[i] = p;
                        self.velocities[i] = v;
                        moved = true;
                    }
                }
                if !on_fluid.length_squared().is_zero() {
                    self.boundary_force = self.boundary_force + on_fluid;
                    if let Some(f) = self.body_forces.get_mut(k) {
                        *f = *f - on_fluid;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        // what the passes left: 0 unless the cap ran out (read by `contact_left_inside`)
        let mut left = 0;
        for i in 0..self.particle_count() {
            let inside = self.bodies.iter().any(|b| {
                sdf_contact(
                    b.sdf(),
                    self.positions[i],
                    self.velocities[i],
                    ContactSide::Solid,
                )
                .is_some()
            });
            if inside {
                left += 1;
            }
        }
        self.contact_left_inside = left;
    }
}

/// Contact passes over the density-map solids per substep (see `resolve_body_contacts`)
#[cfg(feature = "std")]
const CONTACT_PASSES: usize = 4;

/// Which side of an SDF the fluid keeps to
#[cfg(feature = "std")]
#[derive(Clone, Copy)]
enum ContactSide {
    /// Inside (`d <= 0`): a container
    Container,
    /// Outside (`d >= 0`): a solid
    Solid,
}

/// The contact rule of container colliders and solids: a centre on the wrong side of the
/// surface is moved back onto it along the normal (by `-d`); a container reflects its normal
/// velocity with restitution 0.5 (`v - 1.5 (v . n) n`), a solid removes it (`v - (v . n) n`);
/// `None` when the centre is on its side
#[cfg(feature = "std")]
fn sdf_contact(
    sdf: &SdfCollider,
    p: Vec3Fix,
    v: Vec3Fix,
    side: ContactSide,
) -> Option<(Vec3Fix, Vec3Fix)> {
    let (lx, ly, lz) = sdf.world_to_local(p);
    let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;
    let wrong_side = match side {
        ContactSide::Container => dist > 0.0,
        ContactSide::Solid => dist < 0.0,
    };
    if !wrong_side {
        return None;
    }
    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
    let normal = sdf.local_normal_to_world(nx, ny, nz);
    let p = p + normal * Fix128::from_f32(-dist);
    let vn = normal * v.dot(normal);
    match side {
        // restitution 0.5, kept for the container colliders (see the fluid MODEL)
        ContactSide::Container => Some((p, v - vn * Fix128::from_ratio(15, 10))),
        // inelastic: a liquid at a solid wall keeps no normal velocity
        ContactSide::Solid => Some((p, v - vn)),
    }
}

impl core::fmt::Debug for Fluid {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Fluid")
            .field(
                "positions",
                &format_args!("[{} items]", self.positions.len()),
            )
            .field(
                "predicted",
                &format_args!("[{} items]", self.predicted.len()),
            )
            .field(
                "velocities",
                &format_args!("[{} items]", self.velocities.len()),
            )
            .field(
                "densities",
                &format_args!("[{} items]", self.densities.len()),
            )
            .field("lambdas", &format_args!("[{} items]", self.lambdas.len()))
            .field("grid", &"<SpatialGrid>")
            .field("config", &self.config)
            .field("inv_rest_density", &self.inv_rest_density)
            .field("boundary", &format_args!("[{} items]", self.boundary.len()))
            .field("bodies", &self.body_count())
            .field("reference_gradient_sq", &self.reference_gradient_sq)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluid_creation() {
        let config = FluidConfig::default();
        let fluid = Fluid::new_block(
            Vec3Fix::from_f32(-0.5, 0.0, -0.5),
            Vec3Fix::from_f32(0.5, 1.0, 0.5),
            Fix128::from_ratio(1, 5),
            config,
        );

        assert!(fluid.particle_count() > 0, "Should create particles");
    }

    /// Small boxed block for the branch tests below (`n`^3 at the rest spacing, 2 wall
    /// layers, walls twice the block height)
    fn small_boxed(config: FluidConfig, n: i64) -> Fluid {
        let sp = config.rest_spacing;
        let half = sp / Fix128::from_int(2);
        let side = Fix128::from_int(n) * sp;
        Fluid::new_block(
            Vec3Fix::new(half, half, half),
            Vec3Fix::new(side - half, side - half, side - half),
            sp,
            config,
        )
        .with_boundary(Fluid::box_boundary(
            Vec3Fix::ZERO,
            Vec3Fix::new(side, side + side, side),
            sp,
            2,
        ))
    }

    #[test]
    fn boxed_block_under_gravity_is_carried_by_the_floor() {
        // oracle: a fluid at rest on a floor is carried by its weight M g
        let config = FluidConfig::default();
        let mut f = small_boxed(config, 3);
        for _ in 0..240 {
            f.step(Fix128::from_ratio(1, 60));
        }
        let weight = f.particle_count() as f64 * config.particle_mass.to_f64() * 10.0;
        let support = f.boundary_force().y.to_f64() / weight;
        assert!(
            (support - 1.0).abs() < 0.05,
            "floor reaction / weight {support}"
        );
        assert!(
            f.positions.iter().all(|p| p.y > Fix128::ZERO),
            "fell through the floor"
        );
    }

    #[test]
    fn artificial_pressure_pushes_apart() {
        // C >= 0 everywhere (paper); s_corr adds a pairwise repulsion
        let a = FluidConfig {
            gravity: Vec3Fix::ZERO,
            ..FluidConfig::default()
        };
        let mut b = a;
        b.artificial_pressure = Fix128::from_ratio(1, 10);
        let mut fa = small_boxed(a, 2);
        let mut fb = small_boxed(b, 2);
        for _ in 0..10 {
            fa.step(Fix128::from_ratio(1, 60));
            fb.step(Fix128::from_ratio(1, 60));
        }
        let top = |f: &Fluid| {
            f.positions
                .iter()
                .map(|p| p.y.to_f64())
                .fold(f64::MIN, f64::max)
        };
        assert!(
            top(&fb) > top(&fa),
            "s_corr must lift the free top: {} vs {}",
            top(&fb),
            top(&fa)
        );
    }

    #[test]
    fn lattice_mass_falls_back_on_degenerate_inputs() {
        let rho = Fix128::from_int(1000);
        let s = Fix128::from_ratio(1, 10);
        assert_eq!(
            FluidConfig::lattice_mass(rho, Fix128::ZERO, s),
            rho * s * s * s
        );
        assert_eq!(
            FluidConfig::lattice_mass(rho, Fix128::ONE, Fix128::ZERO),
            Fix128::ZERO
        );
        assert!(FluidConfig::lattice_mass(rho, Fix128::from_ratio(1, 5), s) > Fix128::ZERO);
    }

    #[test]
    fn test_fluid_gravity() {
        let config = FluidConfig {
            iterations: 2,
            substeps: 1,
            ..Default::default()
        };
        let mut fluid = Fluid::new(vec![Vec3Fix::from_f32(0.0, 5.0, 0.0)], config);

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            fluid.step(dt);
        }

        let y = fluid.positions[0].y.to_f32();
        assert!(y < 5.0, "Particle should fall under gravity");
    }

    /// radial quadrature over the ball of radius h with the given integrand
    fn ball_integral(h: Fix128, f: impl Fn(Fix128) -> Fix128) -> f64 {
        let n = 4000i64;
        let dr = h / Fix128::from_int(n);
        let mut sum = 0.0;
        for i in 0..n {
            let r = dr * (Fix128::from_int(i) + Fix128::from_ratio(1, 2));
            sum += (f(r) * Fix128::from_int(4) * Fix128::PI * r * r * dr).to_f64();
        }
        sum
    }

    #[test]
    fn poly6_integrates_to_one_over_its_support() {
        for h in [
            Fix128::from_ratio(1, 5),
            Fix128::from_ratio(3, 2),
            Fix128::ONE,
        ] {
            let v = ball_integral(h, |r| poly6(r * r, h));
            assert!((v - 1.0).abs() < 1e-4, "h = {}: {v}", h.to_f64());
        }
    }

    #[test]
    fn poly6_has_compact_support() {
        let h = Fix128::from_ratio(1, 5);
        for k in [10, 11, 15, 30] {
            let r = h * Fix128::from_ratio(k, 10);
            assert_eq!(poly6(r * r, h), Fix128::ZERO, "r = {k}/10 h");
        }
    }

    #[test]
    fn spiky_gradient_matches_its_closed_form() {
        let h = Fix128::from_ratio(1, 5);
        let hf = 0.2f64;
        for k in 1..10 {
            let r = h * Fix128::from_ratio(k, 10);
            let d = hf - r.to_f64();
            let h6 = hf * hf * hf * hf * hf * hf;
            let want = -45.0 / (core::f64::consts::PI * h6) * d * d;
            let got = spiky_grad(r, h).to_f64();
            assert!(got < 0.0, "gradient points inward");
            assert!(
                ((got - want) / want).abs() < 1e-9,
                "k = {k}: {got} vs {want}"
            );
        }
        assert_eq!(spiky_grad(h, h), Fix128::ZERO);
        assert_eq!(spiky_grad(h * Fix128::from_ratio(3, 2), h), Fix128::ZERO);
        assert_eq!(spiky_grad(Fix128::ZERO, h), Fix128::ZERO);
        // closed form: integral of |grad W| over the ball is 6 / h
        let v = ball_integral(h, |r| -spiky_grad(r, h));
        assert!((v - 6.0 / hf).abs() / (6.0 / hf) < 1e-3, "{v}");
    }

    #[test]
    fn test_poly6_kernel() {
        let h = Fix128::from_ratio(1, 5);
        let zero = poly6(Fix128::ZERO, h);
        assert!(zero > Fix128::ZERO, "Poly6 at r=0 should be positive");

        let far = poly6(h * h, h);
        assert!(
            far.is_zero() || far <= Fix128::ZERO,
            "Poly6 at r=h should be zero"
        );
    }

    #[test]
    fn test_spatial_grid() {
        let mut grid = SpatialGrid::new(Fix128::from_ratio(1, 5), 32);
        grid.insert(0, Vec3Fix::ZERO);
        grid.insert(1, Vec3Fix::from_f32(0.1, 0.0, 0.0));
        grid.insert(2, Vec3Fix::from_f32(10.0, 0.0, 0.0));
        grid.build();

        let h_sq = Fix128::from_ratio(1, 5) * Fix128::from_ratio(1, 5);
        let mut neighbors = Vec::new();
        grid.query_neighbors_into(Vec3Fix::ZERO, h_sq, &mut neighbors);
        assert!(neighbors.contains(&0), "Should find self");
        assert!(neighbors.contains(&1), "Should find nearby particle");
    }

    /// 原点中心の単位球 SDF (f32 sqrt のみ、det_math gate 対象外)
    #[cfg(feature = "std")]
    fn unit_sphere_collider() -> crate::sdf_collider::SdfCollider {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        let field = ClosureSdf::new(
            |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
            |x, y, z| {
                let len = (x * x + y * y + z * z).sqrt();
                if len > 1e-6 {
                    (x / len, y / len, z / len)
                } else {
                    (0.0, 1.0, 0.0)
                }
            },
        );
        SdfCollider::new_static(
            Box::new(field),
            Vec3Fix::ZERO,
            crate::math::QuatFix::IDENTITY,
        )
    }

    /// Static solid from an exact SDF and its unit normal, both in world space
    #[cfg(feature = "std")]
    fn static_solid(
        d: impl Fn(f32, f32, f32) -> f32 + Send + Sync + 'static,
        n: impl Fn(f32, f32, f32) -> (f32, f32, f32) + Send + Sync + 'static,
    ) -> crate::sdf_collider::SdfCollider {
        use crate::sdf_collider::{ClosureSdf, SdfCollider};
        SdfCollider::new_static(
            Box::new(ClosureSdf::new(d, n)),
            Vec3Fix::ZERO,
            crate::math::QuatFix::IDENTITY,
        )
    }

    #[cfg(feature = "std")]
    #[test]
    fn a_floor_and_walls_as_two_solids_carry_the_weight_and_balance_the_fluid_side() {
        // oracle: a block at rest in a square well, its floor one density-map solid and its
        // four side walls another, is carried by its weight M g (the walls carry no vertical
        // load at rest); the solids' forces sum to minus the fluid-side total (Newton 3, the
        // particles at the foot of a wall go through the two-solid split); at no frame is a
        // particle centre inside a solid (the contact with the map solids)
        let config = FluidConfig {
            iterations: 4,
            ..FluidConfig::default()
        };
        let sp = config.rest_spacing;
        let h = config.kernel_radius;
        let half = sp / Fix128::from_int(2);
        let side = Fix128::from_int(3) * sp;
        let w = side.to_f64() as f32;
        let floor = static_solid(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
        // distance to the nearest side wall of the well [0, w]^2 in x and z
        let walls = static_solid(
            move |x, _, z| x.min(w - x).min(z).min(w - z),
            move |x, _, z| {
                let m = x.min(w - x).min(z).min(w - z);
                if m == x {
                    (1.0, 0.0, 0.0)
                } else if m == w - x {
                    (-1.0, 0.0, 0.0)
                } else if m == z {
                    (0.0, 0.0, 1.0)
                } else {
                    (0.0, 0.0, -1.0)
                }
            },
        );
        let lo = Vec3Fix::new(-h, -h, -h);
        let hi = Vec3Fix::new(side + h, side + h, side + h);
        let mut f = Fluid::new_block(
            Vec3Fix::new(half, half, half),
            Vec3Fix::new(side - half, side - half, side - half),
            sp,
            config,
        )
        .with_sdf_body(floor, lo, hi)
        .with_sdf_body(walls, lo, hi);
        assert_eq!(f.body_count(), 2);
        assert!(f.body_forces().is_empty(), "no force before the first step");
        let frames = 240;
        let mut carried = 0.0;
        let wd = side.to_f64();
        for frame in 0..frames {
            f.step(Fix128::from_ratio(1, 60));
            let bf = f.body_forces();
            assert_eq!(bf.len(), 2);
            // no centre inside either solid at any frame (the contact puts it back on the
            // surface; 1e-6 m covers the f32 distance the contact reads)
            let deepest = f
                .positions
                .iter()
                .map(|p| {
                    let (x, y, z) = (p.x.to_f64(), p.y.to_f64(), p.z.to_f64());
                    y.min(x).min(z).min(wd - x).min(wd - z)
                })
                .fold(f64::MAX, f64::min);
            assert_eq!(
                f.contact_left_inside(),
                0,
                "frame {frame}: the contact passes ran out"
            );
            assert!(
                deepest > -1e-6,
                "frame {frame}: a particle is {} m inside a solid",
                -deepest
            );
            let sum = bf[0] + bf[1] + f.boundary_force();
            assert_eq!(sum, Vec3Fix::ZERO, "frame {frame}: Newton 3");
            if frame >= frames - 60 {
                carried += -(bf[0].y + bf[1].y).to_f64() / 60.0;
            }
        }
        let weight = f.particle_count() as f64 * config.particle_mass.to_f64() * 10.0;
        assert!(
            (carried / weight - 1.0).abs() < 0.05,
            "weight carried by the solids / M g = {}",
            carried / weight
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn the_contact_with_a_map_solid_is_its_reaction() {
        // oracle: momentum balance of the fluid over one frame with one substep, no damping
        // and no vorticity confinement (the other velocity terms are pairwise and cancel):
        // sum m dv = M g dt + F dt, F the force the solids exerted on the fluid
        // (`boundary_force`, corrections and contacts); a block dropped onto a floor solid,
        // so the contact acts in the first frames
        let config = FluidConfig {
            substeps: 1,
            damping: Fix128::ONE,
            vorticity_strength: Fix128::ZERO,
            ..FluidConfig::default()
        };
        let sp = config.rest_spacing;
        let h = config.kernel_radius;
        let lift = sp * Fix128::from_int(2);
        let floor = static_solid(|_, y, _| y, |_, _, _| (0.0, 1.0, 0.0));
        let mut f = Fluid::new_block(
            Vec3Fix::new(Fix128::ZERO, lift, Fix128::ZERO),
            Vec3Fix::new(
                sp * Fix128::from_int(2),
                lift + sp * Fix128::from_int(2),
                sp * Fix128::from_int(2),
            ),
            sp,
            config,
        )
        .with_sdf_body(
            floor,
            Vec3Fix::new(-h, -h, -h),
            Vec3Fix::new(sp * Fix128::from_int(4), h, sp * Fix128::from_int(4)),
        );
        let m = config.particle_mass.to_f64();
        let dt = 1.0 / 60.0;
        let mut contacts = 0;
        for frame in 0..90 {
            let before: Vec<Vec3Fix> = f.velocities.clone();
            let lowest_before = f
                .positions
                .iter()
                .map(|p| p.y.to_f64())
                .fold(f64::MAX, f64::min);
            f.step(Fix128::from_ratio(1, 60));
            let mut dp = [0.0; 3];
            for (a, b) in f.velocities.iter().zip(&before) {
                let d = *a - *b;
                dp[0] += m * d.x.to_f64();
                dp[1] += m * d.y.to_f64();
                dp[2] += m * d.z.to_f64();
            }
            let fb = f.boundary_force();
            let weight = f.particle_count() as f64 * m * config.gravity.y.to_f64();
            let expect = [
                fb.x.to_f64() * dt,
                (weight + fb.y.to_f64()) * dt,
                fb.z.to_f64() * dt,
            ];
            let scale = (weight * dt).abs();
            for k in 0..3 {
                assert!(
                    (dp[k] - expect[k]).abs() <= 1e-6 * scale,
                    "frame {frame} axis {k}: dp {} vs {}",
                    dp[k],
                    expect[k]
                );
            }
            // the solid's force is minus the fluid-side total
            let sum = f.body_forces()[0] + fb;
            assert_eq!(sum, Vec3Fix::ZERO, "frame {frame}: Newton 3");
            let lowest = f
                .positions
                .iter()
                .map(|p| p.y.to_f64())
                .fold(f64::MAX, f64::min);
            assert!(
                lowest > -1e-6,
                "frame {frame}: a centre at y = {lowest} is inside the floor"
            );
            if lowest_before > 1e-6 && lowest < 1e-5 {
                contacts += 1;
            }
        }
        assert!(contacts > 0, "the block never reached the floor");
    }

    /// 単位球からの符号付き距離 (f32 oracle)
    #[cfg(feature = "std")]
    fn sphere_dist(p: Vec3Fix) -> f32 {
        let (x, y, z) = p.to_f32();
        (x * x + y * y + z * z).sqrt() - 1.0
    }

    #[cfg(feature = "std")]
    #[test]
    fn step_with_sdf_contains_fluid_particles_inside_unit_sphere() {
        // kernel 半径 0.2 より離した 3 粒子 (相互作用なし) を球内に置き、重力で落とす
        // fluid の SDF は「容器」: 外に出た粒子 (dist > 0) を表面へ戻し法線速度を反射
        let make = || {
            Fluid::new(
                vec![
                    Vec3Fix::from_f32(0.0, 0.5, 0.0),
                    Vec3Fix::from_f32(0.3, 0.2, 0.0),
                    Vec3Fix::from_f32(-0.3, 0.1, 0.3),
                ],
                FluidConfig::default(),
            )
        };
        let sphere = [unit_sphere_collider()];
        let mut fluid = make();
        let dt = Fix128::from_ratio(1, 60);
        let mut max_dist = f32::MIN;
        for frame in 0..120 {
            fluid.step_with_sdf(dt, &sphere);
            for (i, p) in fluid.positions.iter().enumerate() {
                let d = sphere_dist(*p);
                max_dist = max_dist.max(d);
                assert!(
                    d <= 1e-3,
                    "frame {frame} particle {i} escaped sphere: dist {d}"
                );
            }
        }
        // 実際に壁に当たっている (2 s で 20 m 落ちるはずが半径 1 の球内)
        assert!(
            max_dist > -0.05,
            "never reached the wall: max dist {max_dist}"
        );
        for (i, p) in fluid.positions.iter().enumerate() {
            let (_, y, _) = p.to_f32();
            assert!(y >= -1.0 - 1e-3, "particle {i} below sphere bottom: y {y}");
        }

        // SDF なしでは粒子は落下して球外へ
        let mut free = make();
        for _ in 0..120 {
            free.step(dt);
        }
        for (i, p) in free.positions.iter().enumerate() {
            let d = sphere_dist(*p);
            assert!(d > 1.0, "particle {i} should have fallen out: dist {d}");
        }

        // collider が空なら step と bit 一致
        let mut a = make();
        let mut b = make();
        for _ in 0..10 {
            a.step_with_sdf(dt, &[]);
            b.step(dt);
        }
        assert_eq!(a.positions, b.positions);
        assert_eq!(a.velocities, b.velocities);
    }
}
