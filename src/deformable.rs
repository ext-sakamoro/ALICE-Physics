//! Deformable Body Simulation (FEM-XPBD)
//!
//! Tetrahedral mesh deformable bodies using XPBD volume + shape constraints.
//! Supports SDF collision for deformable-rigid interaction.
//!
//! # Algorithm
//!
//! - Volume constraints preserve tetrahedral volume
//! - Shape matching constraints resist deformation
//! - Edge constraints prevent over-stretching
//! - SDF collision for surface contacts
//!
//! # MODEL — tetrahedral soft body
//!
//! **Equations.** Particles of mass `m` under XPBD: a predicted position `x* = x + v·h + a·h²`
//! per substep `h`, constraints projected over `iterations` Gauss–Seidel passes, velocities
//! `v = (x − x_prev) / h`. Two constitutive laws ([`Material`]):
//! `Material::NeoHookean(law)` is the stable Neo-Hookean solid on every tetrahedron: with
//! the deformation gradient `F = Ds·Dm⁻¹` (`Ds`, `Dm` the current and rest edge matrices),
//! `I_C = tr(FᵀF)`, `J = det F`, the energy per unit rest volume
//! `ψ = μ/2 (I_C − 3) − μ (J − 1) + λ̂/2 (J − 1)²` (Smith, De Goes & Kim, ACM TOG 2018,
//! without the log term; zero at every rotation, positive elsewhere near rest), with
//! `μ = E / (2(1+ν))`, `λ = Eν / ((1+ν)(1−2ν))` and `λ̂ = λ + μ` (the energy expands to
//! `μ tr(ε²) + (λ̂ − μ)/2 (tr ε)²` at small strain, so this `λ̂` makes that limit Hooke's law
//! with `μ`, `λ`), is the XPBD constraint `C = √(2 V ψ)`
//! of unit compliance (`½ C² = V ψ`, `V` the rest volume; Macklin, Müller & Chentanez 2016)
//! with gradient `∇_i C = V (P Dm⁻ᵀ) e_i / C`, `P = ∂ψ/∂F = μ F + (λ̂ (J − 1) − μ) cof F`,
//! projected with `Δλ = −(C + α̃λ) / (Σ w_i|∇_i C|² + α̃)`, `α̃ = 1 / h²`, the multiplier `λ`
//! kept over the iterations of a substep (`solve_neo_hookean`). (The two-constraint split `C_D = ‖F‖`, `C_H = J − γ`
//! of Macklin & Müller 2021 is not used: in the stiff regime `α̃ ≪ Σ w|∇C|²` its deviatoric
//! projection sends `F` to zero in one step and the body collapses (`solve_neo_hookean` keeps
//! the energy form); measured.) `Material::Constraints` is the earlier geometric set: edge
//! length `|x_i − x_j| = l_0` (compliance `edge_compliance`), tetrahedral volume
//! `V(x) = V_0` (compliance `volume_compliance`) and a blend of every particle toward its
//! rest offset from the centroid by `shape_stiffness` per iteration.
//!
//! **Units.** SI: positions m, masses kg, time s, `E`, `μ`, `λ` Pa, `ν` dimensionless,
//! compliances m/N (the geometric set: `edge_compliance` m/N, `volume_compliance` m³/N·m);
//! every quantity is `Fix128` (Q64.64), so a run replays bit for bit.
//!
//! **Conservation.** Every constraint moves its particles along the gradient in proportion to
//! their inverse masses, so the particles' linear momentum is unchanged by a projection
//! (exactly, `Fix128` addition being exact); the Neo-Hookean and volume constraints are
//! translation invariant and the Neo-Hookean set rotation invariant, so they conserve angular
//! momentum too, while the shape blend of `Material::Constraints` is not rotation invariant
//! (it pulls toward the rest orientation and exchanges angular momentum with nothing).
//! Energy: the elastic energy is a potential (`lambda_energy` gives its coefficients, nothing
//! tracks it as a quantity); `damping` (`apply_frame_damping`) is a velocity retention per
//! frame (numerical, frame-rate dependent) that dissipates, and the Gauss–Seidel passes before
//! convergence dissipate as well.
//!
//! **Valid range.** Neo-Hookean: `0 < E ≤ 2⁴⁰ Pa` and `−1 < ν ≤ 0.5 − 2⁻¹⁶`, refused by
//! [`NeoHookeanLaw::new`] outside (`λ = Eν/((1+ν)(1−2ν))` diverges at `ν = 0.5`); any
//! strain including inverted tetrahedra (`J ≤ 0`, the energy stays finite and the shape
//! recovers, `deformation_gradient`); rest tetrahedra of non-zero volume (a degenerate one is
//! skipped by `build_constraints`). The
//! stiffness converges with `iterations` and `substeps` (the stiff regime
//! `α̃ = 1/h² ≪ Σ w|∇C|²` is handled: a projection then restores the rest shape and the
//! multiplier carries the material's compliance to convergence). Positions, `V` and `h` must
//! keep `V ψ`, `V P Dm⁻¹` and `Σ w|∇C|²` inside `Fix128` (|value| < 2^63); out of that the
//! arithmetic wraps and the result is not a simulation (no detection in `solve_neo_hookean`).
//! The geometric set has no material range: its stiffness is the iteration count's
//! (`iterations`).
//!
//! **Approximations.** P1 (constant strain) tetrahedra (`deformation_gradient`); XPBD
//! Gauss–Seidel projection, which
//! converges first order in the substep `h` only once `α̃ = 1/h²` is comparable to
//! `Σ w|∇C|²` (≈ `μ / (ρ L²)`): below that the body is far softer than the material, and
//! more iterations do not help because the multipliers reset every substep (measured: a
//! 1.25 cm column at 64 substeps per frame sags 18 % more than the finite element solve, at
//! 256 substeps 2.8 %, at 4 substeps it collapses). For a stiff solid an implicit solve
//! (`crate::dynamic_fem`) reaches 1 % at one step per frame, 17–100 times cheaper (measured
//! by `tests/analytic_deformable_integrators.rs`); no plasticity, viscosity, damage or thermal
//! strain (`solve_neo_hookean` has only the elastic law); contact
//! with SDF colliders (`resolve_sdf_collisions`) is a position projection to the surface with
//! the normal velocity dropped
//! and the tangential one scaled by `1 − sdf_friction` (no Coulomb law, no reaction on the
//! collider), with rigid bodies a mass-weighted push-out; particle masses are what the caller
//! sets (`new` gives every particle the same mass; a lumped distribution is the caller's,
//! `inv_masses`). The geometric set is not a material model: no `E`, no `ν`, its sag depends
//! on the iteration count and the compliances (`solve_edge_constraints`,
//! `solve_volume_constraints`).
//!
//! **Validation.** `tests/analytic_deformable_static_sag.rs::the_neo_hookean_body_settles_to_the_finite_element_sag`
//! (a fixed-base column's sag under its own weight equals the linear elastic finite element
//! solve on the same tetrahedra within 5 % at 256 substeps per frame, the bound derived from
//! the measured first-order convergence),
//! `::the_neo_hookean_sag_converges_to_the_finite_element_sag_as_the_substeps_double` (32 to
//! 256 substeps: 1.78, 0.18, 0.092, 0.028 of the finite element sag, shrinking at every
//! doubling), `::an_inverted_tetrahedron_recovers_its_rest_shape`,
//! `::a_neo_hookean_run_replays_to_the_same_bits`,
//! `::the_finite_element_sag_of_a_column_lies_below_rho_g_h_squared_over_two_e_by_at_most_the_base_layer`
//! (the reference itself against the closed form: the fixed base takes at most the bottom
//! layer's share, 6.0 % at `H/w = 8`, 5.4 % at 15),
//! `::the_default_material_is_constraints_and_neo_hookean_is_a_different_law`;
//! `tests/analytic_deformable_wiring.rs` (the cube constructor, centre of mass in free fall,
//! rigid-body push-out).
//!
//! **Tiers.** This body is not the reference tier for a stiff solid. Measured on the same
//! Kuhn-split column of 1.25 cm tetrahedra (`E = 1.27e6 Pa`, `ν = 0.49`, `ρ = 1100 kg/m³`,
//! base fixed, reference = the static linear finite element sag on the same mesh,
//! `tests/analytic_deformable_integrators.rs`): the Neo-Hookean XPBD body needs 64 substeps
//! per frame for 21 % and 256 for 2.8 % of the sag (0.013 and 0.05 s per frame on 24
//! tetrahedra; 0.24 s per frame at 64 substeps on 384), while the implicit Newmark solve of
//! `crate::dynamic_fem` on the same tetrahedra reaches 0.6–1 % at one step per frame for
//! 0.0002 and 0.026 s per frame: 17 to 100 times cheaper at better accuracy. So L0 for a
//! stiff elastic solid is the implicit finite element step (co-rotational for large rotation),
//! and this body is the coupling path (contact with the rest of the world through its
//! particles) and a candidate for a coarse cage tier, its error against L0 to be measured by
//! the same sag and vibration problems when that tier exists.
//!
//! Author: Moroya Sakamoto

use crate::math::{Fix128, Mat3Fix, Vec3Fix};
#[cfg(feature = "std")]
use crate::sdf_collider::SdfCollider;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Deformable Configuration
// ============================================================================

/// The constitutive law a [`DeformableBody`] solves per substep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Material {
    /// The geometric constraints: edge length (`edge_compliance`), tetrahedral volume
    /// (`volume_compliance`) and the blend toward the rest shape (`shape_stiffness`). No
    /// material parameter: the stiffness depends on the iteration count and the compliances.
    Constraints,
    /// The stable Neo-Hookean solid with Young's modulus `E` (Pa) and Poisson's ratio `ν`
    /// ([`NeoHookeanLaw`], checked on construction): one energy constraint per tetrahedron,
    /// see the module's MODEL section. Checked by `tests/analytic_deformable_static_sag.rs`.
    NeoHookean(NeoHookeanLaw),
}

/// Why a [`NeoHookeanLaw`] could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialError {
    /// `E ≤ 0`
    YoungModulusNotPositive,
    /// `ν` outside `(−1, 0.5 − 2⁻¹⁶]`: at `ν = 0.5` the Lamé parameter `λ` diverges, and
    /// closer than `2⁻¹⁶` to it `λ = Eν / ((1+ν)(1−2ν))` leaves the `Fix128` range for the
    /// moduli this law is meant for (`E` up to `2⁴⁰ Pa`)
    PoissonRatioOutOfRange,
    /// `E > 2⁴⁰ Pa`: `μ V`, `λ V` and the stress `P` must stay inside `Fix128`
    YoungModulusTooLarge,
}

impl core::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::YoungModulusNotPositive => f.write_str("Young's modulus must be positive"),
            Self::PoissonRatioOutOfRange => {
                f.write_str("Poisson's ratio must be in (-1, 0.5 - 2^-16]")
            }
            Self::YoungModulusTooLarge => f.write_str("Young's modulus must be at most 2^40 Pa"),
        }
    }
}

impl core::error::Error for MaterialError {}

/// Young's modulus and Poisson's ratio of a [`Material::NeoHookean`] body, with the Lamé
/// parameters `μ = E / (2(1+ν))`, `λ = Eν / ((1+ν)(1−2ν))` they give. Built only through
/// [`NeoHookeanLaw::new`], which refuses the values the constraint cannot carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NeoHookeanLaw {
    mu: Fix128,
    lambda: Fix128,
}

impl NeoHookeanLaw {
    /// Largest Young's modulus accepted, Pa (`2⁴⁰`).
    pub const MAX_YOUNG_MODULUS: Fix128 = Fix128::from_int(1 << 40);
    /// How far below `0.5` Poisson's ratio must stay (`2⁻¹⁶`).
    pub const POISSON_MARGIN: Fix128 = Fix128::from_raw(0, 1 << 48);

    /// `E` in Pa, `ν` dimensionless.
    ///
    /// # Errors
    ///
    /// [`MaterialError::YoungModulusNotPositive`] for `E ≤ 0`,
    /// [`MaterialError::YoungModulusTooLarge`] above [`Self::MAX_YOUNG_MODULUS`],
    /// [`MaterialError::PoissonRatioOutOfRange`] for `ν ≤ −1` or `ν > 0.5 − 2⁻¹⁶`.
    pub fn new(young_modulus: Fix128, poisson_ratio: Fix128) -> Result<Self, MaterialError> {
        if young_modulus <= Fix128::ZERO {
            return Err(MaterialError::YoungModulusNotPositive);
        }
        if young_modulus > Self::MAX_YOUNG_MODULUS {
            return Err(MaterialError::YoungModulusTooLarge);
        }
        let one_plus = Fix128::ONE + poisson_ratio;
        let one_minus_two = Fix128::ONE - (poisson_ratio + poisson_ratio);
        if one_plus <= Fix128::ZERO || one_minus_two < Self::POISSON_MARGIN + Self::POISSON_MARGIN {
            return Err(MaterialError::PoissonRatioOutOfRange);
        }
        let mu = young_modulus / (one_plus + one_plus);
        let lambda = young_modulus * poisson_ratio / (one_plus * one_minus_two);
        Ok(Self { mu, lambda })
    }

    /// The Lamé parameters `(μ, λ)`, Pa
    #[must_use]
    pub const fn lame(&self) -> (Fix128, Fix128) {
        (self.mu, self.lambda)
    }

    /// The energy's volumetric coefficient `λ̂ = λ + μ`: the stable Neo-Hookean energy without
    /// its log term expands to `μ tr(ε²) + (λ̂ − μ)/2 (tr ε)²` at small strain, so `λ̂ = λ + μ`
    /// makes that limit Hooke's law with this `μ` and `λ`.
    #[must_use]
    pub fn lambda_energy(&self) -> Fix128 {
        self.lambda + self.mu
    }
}

/// Deformable body configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformableConfig {
    /// Constitutive law (`Material::Constraints` by default)
    pub material: Material,
    /// Solver iterations
    pub iterations: usize,
    /// Number of substeps
    pub substeps: usize,
    /// Gravity
    pub gravity: Vec3Fix,
    /// Velocity retention per frame (`step()` call), applied once per frame since 1.2.0
    pub damping: Fix128,
    /// Volume preservation compliance (0 = rigid volume)
    pub volume_compliance: Fix128,
    /// Edge stretch compliance
    pub edge_compliance: Fix128,
    /// Shape matching stiffness (0..1, 1 = rigid)
    pub shape_stiffness: Fix128,
    /// SDF collision friction
    pub sdf_friction: Fix128,
}

impl Default for DeformableConfig {
    fn default() -> Self {
        Self {
            material: Material::Constraints,
            iterations: 8,
            substeps: 4,
            gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
            damping: Fix128::from_ratio(99, 100),
            volume_compliance: Fix128::from_ratio(1, 1000),
            edge_compliance: Fix128::ZERO,
            shape_stiffness: Fix128::from_ratio(5, 10),
            sdf_friction: Fix128::from_ratio(3, 10),
        }
    }
}

// ============================================================================
// Constraints
// ============================================================================

/// Tetrahedral volume constraint
#[derive(Clone, Copy, Debug, PartialEq)]
struct TetConstraint {
    /// Four vertex indices
    indices: [usize; 4],
    /// Rest volume (signed, preserves orientation)
    rest_volume: Fix128,
}

/// Edge length constraint
#[derive(Clone, Copy, Debug, PartialEq)]
struct EdgeConstraint {
    i0: usize,
    i1: usize,
    rest_length: Fix128,
}

// ============================================================================
// Deformable Body
// ============================================================================

/// Deformable body simulation
pub struct DeformableBody {
    /// Particle positions
    pub positions: Vec<Vec3Fix>,
    /// Previous positions
    prev_positions: Vec<Vec3Fix>,
    /// Velocities
    pub velocities: Vec<Vec3Fix>,
    /// Inverse mass per particle
    pub inv_masses: Vec<Fix128>,
    /// Tetrahedra (4 vertex indices each)
    pub tetrahedra: Vec<[usize; 4]>,
    /// Surface triangles (for rendering/collision)
    pub surface_triangles: Vec<[usize; 3]>,
    /// Tetrahedral volume constraints
    tet_constraints: Vec<TetConstraint>,
    /// Edge constraints
    edge_constraints: Vec<EdgeConstraint>,
    /// Rest positions relative to center
    rest_relative: Vec<Vec3Fix>,
    /// Per tetrahedron: the inverse of the rest edge matrix `Dm = [x1−x0, x2−x0, x3−x0]` and
    /// the rest volume `|det Dm| / 6` (`None` for a degenerate rest tetrahedron, which the
    /// Neo-Hookean constraints skip), indexed like `tetrahedra`
    rest_inverse: Vec<Option<(Mat3Fix, Fix128)>>,
    /// Configuration
    pub config: DeformableConfig,
}

impl DeformableBody {
    /// Create from particle positions and tetrahedra
    #[must_use]
    pub fn new(
        positions: Vec<Vec3Fix>,
        tetrahedra: &[[usize; 4]],
        surface_triangles: Vec<[usize; 3]>,
        mass_per_particle: Fix128,
    ) -> Self {
        let n = positions.len();
        let inv_mass = if mass_per_particle.is_zero() {
            Fix128::ZERO
        } else {
            Fix128::ONE / mass_per_particle
        };

        // Compute rest shape
        let rest_center = compute_center_of_mass(&positions);
        let rest_relative: Vec<Vec3Fix> = positions.iter().map(|p| *p - rest_center).collect();
        let rest_inverse: Vec<Option<(Mat3Fix, Fix128)>> = tetrahedra
            .iter()
            .map(|&[i0, i1, i2, i3]| {
                let dm = Mat3Fix::from_cols(
                    positions[i1] - positions[i0],
                    positions[i2] - positions[i0],
                    positions[i3] - positions[i0],
                );
                let volume = dm.determinant().abs() / Fix128::from_int(6);
                dm.inverse().map(|inv| (inv, volume))
            })
            .collect();

        let mut body = Self {
            prev_positions: positions.clone(),
            velocities: vec![Vec3Fix::ZERO; n],
            inv_masses: vec![inv_mass; n],
            tetrahedra: tetrahedra.to_vec(),
            surface_triangles,
            tet_constraints: Vec::new(),
            edge_constraints: Vec::new(),
            rest_relative,
            rest_inverse,
            positions,
            config: DeformableConfig::default(),
        };

        body.build_constraints();
        body
    }

    /// Create a simple cube deformable (for testing)
    #[must_use]
    pub fn new_cube(center: Vec3Fix, half_extent: Fix128, mass: Fix128) -> Self {
        let h = half_extent;
        let positions = vec![
            center + Vec3Fix::new(-h, -h, -h),
            center + Vec3Fix::new(h, -h, -h),
            center + Vec3Fix::new(h, h, -h),
            center + Vec3Fix::new(-h, h, -h),
            center + Vec3Fix::new(-h, -h, h),
            center + Vec3Fix::new(h, -h, h),
            center + Vec3Fix::new(h, h, h),
            center + Vec3Fix::new(-h, h, h),
        ];

        // 5 tetrahedra to fill a cube
        let tetrahedra = vec![
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
            [1, 3, 4, 6],
        ];

        let surface_triangles = vec![
            [0, 1, 2],
            [0, 2, 3], // front
            [4, 6, 5],
            [4, 7, 6], // back
            [0, 4, 5],
            [0, 5, 1], // bottom
            [2, 6, 7],
            [2, 7, 3], // top
            [0, 3, 7],
            [0, 7, 4], // left
            [1, 5, 6],
            [1, 6, 2], // right
        ];

        let mass_per_particle = mass / Fix128::from_int(8);
        Self::new(positions, &tetrahedra, surface_triangles, mass_per_particle)
    }

    /// Build constraints from geometry
    fn build_constraints(&mut self) {
        // Tetrahedral volume constraints (skip degenerate tetrahedra)
        let min_volume = Fix128::from_ratio(1, 1000000); // 1e-6
        for tet in &self.tetrahedra {
            let rest_vol = compute_tet_volume(
                self.positions[tet[0]],
                self.positions[tet[1]],
                self.positions[tet[2]],
                self.positions[tet[3]],
            );
            if rest_vol.abs() < min_volume {
                continue; // Skip degenerate (near-zero volume) tetrahedra
            }
            self.tet_constraints.push(TetConstraint {
                indices: *tet,
                rest_volume: rest_vol,
            });
        }

        // Edge constraints: unique edges from tetrahedra
        let mut edge_set: Vec<(usize, usize)> = Vec::new();
        for tet in &self.tetrahedra {
            let edges = [
                (tet[0], tet[1]),
                (tet[0], tet[2]),
                (tet[0], tet[3]),
                (tet[1], tet[2]),
                (tet[1], tet[3]),
                (tet[2], tet[3]),
            ];
            for (a, b) in edges {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if !edge_set.contains(&(lo, hi)) {
                    edge_set.push((lo, hi));
                    let rest = (self.positions[a] - self.positions[b]).length();
                    self.edge_constraints.push(EdgeConstraint {
                        i0: a,
                        i1: b,
                        rest_length: rest,
                    });
                }
            }
        }
    }

    /// Number of particles
    #[inline]
    #[must_use]
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }

    /// Step simulation
    pub fn step(&mut self, dt: Fix128) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
        }
        self.apply_frame_damping();
    }

    /// Step with SDF collision
    #[cfg(feature = "std")]
    pub fn step_with_sdf(&mut self, dt: Fix128, sdf_colliders: &[SdfCollider]) {
        let substep_dt = dt / Fix128::from_int(self.config.substeps as i64);
        for _ in 0..self.config.substeps {
            self.substep(substep_dt);
            self.resolve_sdf_collisions(sdf_colliders);
        }
        self.apply_frame_damping();
    }

    /// `config.damping` once per frame (velocity retention per `step()` call).
    ///
    /// 1.2.0: applied per substep before, which made the terminal velocity
    /// depend on `substeps` (`g·h·d/(1−d)`, `h = dt/substeps`) — the same
    /// defect as the rigid-body solver's frame damping fix.
    fn apply_frame_damping(&mut self) {
        let d = self.config.damping;
        for v in &mut self.velocities {
            *v = *v * d;
        }
    }

    /// Single substep
    fn substep(&mut self, dt: Fix128) {
        let n = self.particle_count();

        // 1. Predict
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.prev_positions[i] = self.positions[i];
            self.velocities[i] = self.velocities[i] + self.config.gravity * dt;
            self.positions[i] = self.positions[i] + self.velocities[i] * dt;
        }

        // 2. Solve constraints
        match self.config.material {
            Material::Constraints => {
                for _ in 0..self.config.iterations {
                    self.solve_edge_constraints(dt);
                    self.solve_volume_constraints(dt);
                    self.apply_shape_matching();
                }
            }
            Material::NeoHookean(law) => {
                let (mu, _) = law.lame();
                let lambda = law.lambda_energy();
                let mut multipliers = vec![Fix128::ZERO; self.tetrahedra.len()];
                for _ in 0..self.config.iterations {
                    self.solve_neo_hookean(dt, mu, lambda, &mut multipliers);
                }
            }
        }

        // 3. Update velocities
        let inv_dt = Fix128::ONE / dt;
        for i in 0..n {
            if self.inv_masses[i].is_zero() {
                continue;
            }
            self.velocities[i] = (self.positions[i] - self.prev_positions[i]) * inv_dt;
        }
    }

    /// Solve edge constraints
    fn solve_edge_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.edge_compliance / (dt * dt);

        for c_idx in 0..self.edge_constraints.len() {
            let c = self.edge_constraints[c_idx];
            let p0 = self.positions[c.i0];
            let p1 = self.positions[c.i1];
            let w0 = self.inv_masses[c.i0];
            let w1 = self.inv_masses[c.i1];

            let w_sum = w0 + w1 + compliance;
            if w_sum.is_zero() {
                continue;
            }

            let delta = p1 - p0;
            let dist = delta.length();
            if dist.is_zero() {
                continue;
            }

            let error = dist - c.rest_length;
            let lambda = error / w_sum;
            let correction = delta / dist * lambda;

            if !w0.is_zero() {
                self.positions[c.i0] = self.positions[c.i0] + correction * w0;
            }
            if !w1.is_zero() {
                self.positions[c.i1] = self.positions[c.i1] - correction * w1;
            }
        }
    }

    /// Solve tetrahedral volume constraints
    fn solve_volume_constraints(&mut self, dt: Fix128) {
        let compliance = self.config.volume_compliance / (dt * dt);

        for c_idx in 0..self.tet_constraints.len() {
            let c = self.tet_constraints[c_idx];
            let [i0, i1, i2, i3] = c.indices;

            let p0 = self.positions[i0];
            let p1 = self.positions[i1];
            let p2 = self.positions[i2];
            let p3 = self.positions[i3];

            let current_vol = compute_tet_volume(p0, p1, p2, p3);
            let error = current_vol - c.rest_volume;

            if error.abs() < Fix128::from_ratio(1, 10000) {
                continue;
            }

            // Gradients of volume w.r.t. each vertex
            let g0 = (p1 - p3).cross(p2 - p3);
            let g1 = (p2 - p3).cross(p0 - p3);
            let g2 = (p0 - p3).cross(p1 - p3);
            let g3 = -(g0 + g1 + g2);

            let w0 = self.inv_masses[i0];
            let w1 = self.inv_masses[i1];
            let w2 = self.inv_masses[i2];
            let w3 = self.inv_masses[i3];

            let denom = w0 * g0.length_squared()
                + w1 * g1.length_squared()
                + w2 * g2.length_squared()
                + w3 * g3.length_squared()
                + compliance;

            if denom.is_zero() {
                continue;
            }

            let lambda = error / denom;

            if !w0.is_zero() {
                self.positions[i0] = self.positions[i0] - g0 * (lambda * w0).div_int(6);
            }
            if !w1.is_zero() {
                self.positions[i1] = self.positions[i1] - g1 * (lambda * w1).div_int(6);
            }
            if !w2.is_zero() {
                self.positions[i2] = self.positions[i2] - g2 * (lambda * w2).div_int(6);
            }
            if !w3.is_zero() {
                self.positions[i3] = self.positions[i3] - g3 * (lambda * w3).div_int(6);
            }
        }
    }

    /// One Gauss–Seidel pass of the stable Neo-Hookean energy constraint over every
    /// tetrahedron: with the deformation gradient `F = Ds · Dm⁻¹`, the energy per unit rest
    /// volume `ψ = μ/2 (I_C − 3) − μ (J − 1) + λ̂/2 (J − 1)²` (Smith, De Goes & Kim 2018 without
    /// the log term, shifted so `ψ = 0` at every rotation) is the XPBD constraint
    /// `C = √(2 V ψ)` of unit compliance (`½ C² = V ψ`, Macklin, Müller & Chentanez 2016 §4),
    /// gradient `∇_i C = V (P Dm⁻ᵀ) e_i / C` with the first Piola stress
    /// `P = ∂ψ/∂F = μ F + (λ̂ (J − 1) − μ) cof F`, `λ̂ = λ + μ` ([`NeoHookeanLaw::lambda_energy`]).
    /// A projection to `C = 0` restores the rest
    /// shape (any rotation), so the stiff limit is right; the multiplier `λ_t` persists over
    /// the iterations of one substep.
    fn solve_neo_hookean(
        &mut self,
        dt: Fix128,
        mu: Fix128,
        lambda: Fix128,
        multipliers: &mut [Fix128],
    ) {
        let alpha_tilde = Fix128::ONE / (dt * dt);
        let two = Fix128::from_int(2);
        let three = Fix128::from_int(3);
        for (t, multiplier) in multipliers.iter_mut().enumerate() {
            let Some((dm_inv, rest_volume)) = self.rest_inverse[t] else {
                continue;
            };
            if rest_volume.is_zero() {
                continue;
            }
            let [i0, i1, i2, i3] = self.tetrahedra[t];
            let f = self.deformation_gradient(i0, i1, i2, i3, dm_inv);
            let i_c = f.col0.length_squared() + f.col1.length_squared() + f.col2.length_squared();
            let j_minus_one = f.determinant() - Fix128::ONE;
            let psi = mu * (i_c - three) / two - mu * j_minus_one
                + lambda * j_minus_one * j_minus_one / two;
            let c2 = two * rest_volume * psi;
            if c2 <= Fix128::ZERO {
                continue;
            }
            let c = c2.sqrt();
            if c.is_zero() {
                continue;
            }
            // cof F: columns f1 × f2, f2 × f0, f0 × f1
            let cof = Mat3Fix::from_cols(
                f.col1.cross(f.col2),
                f.col2.cross(f.col0),
                f.col0.cross(f.col1),
            );
            let k = lambda * j_minus_one - mu;
            let p = Mat3Fix::from_cols(
                f.col0 * mu + cof.col0 * k,
                f.col1 * mu + cof.col1 * k,
                f.col2 * mu + cof.col2 * k,
            );
            let g = p.mul_mat(dm_inv.transpose()).scale(rest_volume / c);
            let grads = [-(g.col0 + g.col1 + g.col2), g.col0, g.col1, g.col2];
            let w = [
                self.inv_masses[i0],
                self.inv_masses[i1],
                self.inv_masses[i2],
                self.inv_masses[i3],
            ];
            let mut denom = alpha_tilde;
            for k in 0..4 {
                denom = denom + w[k] * grads[k].length_squared();
            }
            if denom.is_zero() {
                continue;
            }
            let delta = -(c + alpha_tilde * *multiplier) / denom;
            *multiplier = *multiplier + delta;
            let idx = [i0, i1, i2, i3];
            for k in 0..4 {
                if !w[k].is_zero() {
                    self.positions[idx[k]] = self.positions[idx[k]] + grads[k] * (w[k] * delta);
                }
            }
        }
    }

    /// `F = Ds · Dm⁻¹` of one tetrahedron from the current positions
    fn deformation_gradient(
        &self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        dm_inv: Mat3Fix,
    ) -> Mat3Fix {
        let p0 = self.positions[i0];
        Mat3Fix::from_cols(
            self.positions[i1] - p0,
            self.positions[i2] - p0,
            self.positions[i3] - p0,
        )
        .mul_mat(dm_inv)
    }

    /// Shape matching: move particles toward their rest pose
    fn apply_shape_matching(&mut self) {
        let alpha = self.config.shape_stiffness;
        if alpha.is_zero() {
            return;
        }

        // Current center of mass
        let current_center = compute_center_of_mass(&self.positions);

        // LIMITATION(COV-SOFT-017): Simplified shape matching: blend toward rest shape
        // Simplified shape matching: blend toward rest shape
        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            let target = current_center + self.rest_relative[i];
            let delta = target - self.positions[i];
            self.positions[i] = self.positions[i] + delta * alpha;
        }
    }

    /// Resolve SDF collisions
    #[cfg(feature = "std")]
    fn resolve_sdf_collisions(&mut self, sdf_colliders: &[SdfCollider]) {
        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            for sdf in sdf_colliders {
                let (lx, ly, lz) = sdf.world_to_local(self.positions[i]);
                let dist = sdf.field.distance(lx, ly, lz) * sdf.scale_f32;

                if dist < 0.0 {
                    let (nx, ny, nz) = sdf.field.normal(lx, ly, lz);
                    let normal = sdf.local_normal_to_world(nx, ny, nz);
                    let depth = Fix128::from_f32(-dist);
                    self.positions[i] = self.positions[i] + normal * depth;

                    let vel = self.velocities[i];
                    let vn = normal * vel.dot(normal);
                    let vt = vel - vn;
                    self.velocities[i] = vt * (Fix128::ONE - self.config.sdf_friction);
                }
            }
        }
    }

    /// Centroid of the particle positions. Every constructor gives all particles the same
    /// mass, for which this is the centre of mass; it is **not** mass weighted if
    /// `inv_masses` are edited to differ. Zero for a body without particles.
    #[must_use]
    pub fn center_of_mass(&self) -> Vec3Fix {
        compute_center_of_mass(&self.positions)
    }

    /// Resolve collisions between deformable body particles and rigid bodies
    ///
    // LIMITATION(COV-SOFT-059): Treats each rigid body as a sphere with given radius for simplicity.
    /// Treats each rigid body as a sphere with given radius for simplicity.
    /// Applies two-way coupling: deformable particles are pushed out,
    /// and rigid bodies receive reaction impulses.
    ///
    /// The correction and the (perfectly inelastic) normal impulse are shared by inverse
    /// mass, `w_p / (w_p + w_b)` to the particle and `w_b / (w_p + w_b)` to the body; a body
    /// with zero inverse mass (static, kinematic) takes none, so a particle pressed into it
    /// ends exactly on its surface with zero approach speed. A particle exactly at the
    /// centre of a body has no push direction and is left untouched; pinned particles
    /// (zero inverse mass) are skipped. `dt` only gates the call (zero is a no-op).
    pub fn resolve_rigid_body_collisions(
        &mut self,
        rigid_bodies: &mut [crate::solver::RigidBody],
        body_radii: &[Fix128],
        dt: Fix128,
    ) {
        if dt.is_zero() {
            return;
        }

        for i in 0..self.particle_count() {
            if self.inv_masses[i].is_zero() {
                continue;
            }

            for (rb_idx, rb) in rigid_bodies.iter_mut().enumerate() {
                if rb.is_static() && rb_idx >= body_radii.len() {
                    continue;
                }
                let radius = if rb_idx < body_radii.len() {
                    body_radii[rb_idx]
                } else {
                    Fix128::ONE
                };

                let delta = self.positions[i] - rb.position;
                let dist_sq = delta.length_squared();
                let r_sq = radius * radius;

                if dist_sq < r_sq && !dist_sq.is_zero() {
                    let dist = dist_sq.sqrt();
                    let normal = delta / dist;
                    let penetration = radius - dist;

                    // Inverse-mass weights: a body with zero inverse mass (static or
                    // kinematic) is infinitely heavy and takes no share, so the particle
                    // is moved by the whole penetration and loses the whole approach
                    // speed (no finite stand-in mass, which left `m_p / (m_p + 10^6)`
                    // of the penetration and of the speed behind).
                    let w_particle = self.inv_masses[i];
                    let w_body = rb.inv_mass;
                    let w_sum = w_particle + w_body;
                    if w_sum.is_zero() {
                        continue;
                    }

                    let particle_ratio = w_particle / w_sum;
                    let rb_ratio = w_body / w_sum;

                    // Position correction
                    self.positions[i] = self.positions[i] + normal * (penetration * particle_ratio);

                    if !rb.inv_mass.is_zero() {
                        rb.position = rb.position - normal * (penetration * rb_ratio);
                    }

                    // Velocity response (inelastic along the normal)
                    let rel_vel = self.velocities[i] - rb.velocity;
                    let vel_normal = rel_vel.dot(normal);

                    if vel_normal < Fix128::ZERO {
                        let impulse_mag = -vel_normal / w_sum;
                        let impulse = normal * impulse_mag;

                        self.velocities[i] = self.velocities[i] + impulse * w_particle;
                        if !rb.inv_mass.is_zero() {
                            rb.velocity = rb.velocity - impulse * rb.inv_mass;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Compute center of mass
fn compute_center_of_mass(positions: &[Vec3Fix]) -> Vec3Fix {
    if positions.is_empty() {
        return Vec3Fix::ZERO;
    }
    let mut sum = Vec3Fix::ZERO;
    for p in positions {
        sum = sum + *p;
    }
    sum / Fix128::from_int(positions.len() as i64)
}

/// Compute signed volume of a tetrahedron
fn compute_tet_volume(p0: Vec3Fix, p1: Vec3Fix, p2: Vec3Fix, p3: Vec3Fix) -> Fix128 {
    let a = p1 - p0;
    let b = p2 - p0;
    let c = p3 - p0;
    a.dot(b.cross(c)).div_int(6)
}

impl core::fmt::Debug for DeformableBody {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeformableBody")
            .field(
                "positions",
                &format_args!("[{} items]", self.positions.len()),
            )
            .field(
                "prev_positions",
                &format_args!("[{} items]", self.prev_positions.len()),
            )
            .field(
                "velocities",
                &format_args!("[{} items]", self.velocities.len()),
            )
            .field(
                "inv_masses",
                &format_args!("[{} items]", self.inv_masses.len()),
            )
            .field(
                "tetrahedra",
                &format_args!("[{} items]", self.tetrahedra.len()),
            )
            .field(
                "surface_triangles",
                &format_args!("[{} items]", self.surface_triangles.len()),
            )
            .field(
                "tet_constraints",
                &format_args!("[{} items]", self.tet_constraints.len()),
            )
            .field(
                "edge_constraints",
                &format_args!("[{} items]", self.edge_constraints.len()),
            )
            .field(
                "rest_relative",
                &format_args!("[{} items]", self.rest_relative.len()),
            )
            .field("config", &self.config)
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
    fn test_deformable_cube() {
        let body = DeformableBody::new_cube(Vec3Fix::ZERO, Fix128::ONE, Fix128::from_int(10));

        assert_eq!(body.particle_count(), 8);
        assert_eq!(body.tetrahedra.len(), 5);
        assert!(!body.edge_constraints.is_empty());
        assert!(!body.tet_constraints.is_empty());
    }

    #[test]
    fn test_deformable_gravity() {
        let mut body = DeformableBody::new_cube(
            Vec3Fix::from_int(0, 5, 0),
            Fix128::ONE,
            Fix128::from_int(10),
        );

        let dt = Fix128::from_ratio(1, 60);
        for _ in 0..60 {
            body.step(dt);
        }

        let com = body.center_of_mass();
        assert!(
            com.y < Fix128::from_int(5),
            "Deformable should fall under gravity"
        );
    }

    #[test]
    fn test_tet_volume() {
        let vol = compute_tet_volume(
            Vec3Fix::ZERO,
            Vec3Fix::from_int(1, 0, 0),
            Vec3Fix::from_int(0, 1, 0),
            Vec3Fix::from_int(0, 0, 1),
        );
        // Volume of standard simplex = 1/6
        let expected = Fix128::from_ratio(1, 6);
        let error = (vol - expected).abs();
        assert!(
            error < Fix128::from_ratio(1, 100),
            "Tet volume should be 1/6"
        );
    }

    #[test]
    fn test_center_of_mass() {
        let positions = vec![
            Vec3Fix::from_int(0, 0, 0),
            Vec3Fix::from_int(2, 0, 0),
            Vec3Fix::from_int(0, 2, 0),
            Vec3Fix::from_int(0, 0, 2),
        ];

        let com = compute_center_of_mass(&positions);
        let (cx, cy, cz) = com.to_f32();
        assert!((cx - 0.5).abs() < 0.01);
        assert!((cy - 0.5).abs() < 0.01);
        assert!((cz - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_soft_rigid_coupling() {
        use crate::solver::RigidBody;

        let mut body = DeformableBody::new_cube(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::ONE,
            Fix128::from_int(10),
        );

        let mut rigid_bodies = vec![RigidBody::new(
            Vec3Fix::from_int(0, 0, 0),
            Fix128::from_int(5),
        )];
        let radii = vec![Fix128::from_int(3)]; // Large sphere overlapping the cube

        let dt = Fix128::from_ratio(1, 60);
        body.resolve_rigid_body_collisions(&mut rigid_bodies, &radii, dt);

        // After collision resolution, some particles should have moved
        // and the rigid body should have received a reaction
        // This is a basic sanity test
        let com = body.center_of_mass();
        // The COM might have shifted due to collision response
        assert!(
            com.x.hi >= -10 && com.x.hi <= 10,
            "COM should be reasonable"
        );
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

    /// 単位球からの符号付き距離 (f32 oracle)
    #[cfg(feature = "std")]
    fn sphere_dist(p: Vec3Fix) -> f32 {
        let (x, y, z) = p.to_f32();
        (x * x + y * y + z * z).sqrt() - 1.0
    }

    #[cfg(feature = "std")]
    #[test]
    fn step_with_sdf_keeps_cube_particles_outside_unit_sphere() {
        // 一辺 1 の cube (8 corner) を y = 2.5 から球の上に落とす
        let make = || {
            DeformableBody::new_cube(
                Vec3Fix::new(Fix128::ZERO, Fix128::from_ratio(5, 2), Fix128::ZERO),
                Fix128::from_ratio(1, 2),
                Fix128::from_int(8),
            )
        };
        let sphere = [unit_sphere_collider()];
        let mut body = make();
        let dt = Fix128::from_ratio(1, 60);
        let mut min_dist = f32::MAX;
        for frame in 0..60 {
            body.step_with_sdf(dt, &sphere);
            for (i, p) in body.positions.iter().enumerate() {
                let d = sphere_dist(*p);
                min_dist = min_dist.min(d);
                assert!(
                    d >= -1e-3,
                    "frame {frame} particle {i} inside sphere: dist {d}"
                );
            }
        }
        assert!(min_dist < 0.05, "never touched: min dist {min_dist}");
        // 重心は球の上 (y > 0) に留まる (貫通落下なら y < 0)
        let (_, cy, _) = body.center_of_mass().to_f32();
        assert!(cy > 0.5, "center of mass y {cy}");

        // SDF なしでは球を貫通する frame が存在する
        let mut free = make();
        let mut free_min = f32::MAX;
        for _ in 0..60 {
            free.step(dt);
            for p in &free.positions {
                free_min = free_min.min(sphere_dist(*p));
            }
        }
        assert!(
            free_min < -0.1,
            "without SDF the cube should pass through: {free_min}"
        );

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
