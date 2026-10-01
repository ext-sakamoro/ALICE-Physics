//! Small-strain linear elastic FEM on tetrahedra (P1 / constant strain).
//!
//! Consumes a [`crate::sdf_fem_mesh::SdfTetMesh`] and returns a Cauchy stress
//! tensor per element, which is what a stress law needs and what no other
//! module in this crate produces: [`crate::beam_stress`] and
//! [`crate::fillet_stress`] are closed forms for shapes that reduce to a beam
//! or a notch, and [`crate::deformable`] is an XPBD solver whose volume and
//! shape constraints never form a stress tensor.
//!
//! # Which mesh to hand it
//!
//! [`crate::sdf_fem_mesh::generate_marching_tets`], not
//! [`crate::sdf_fem_mesh::generate`]. The latter meshes a staircase strictly
//! inside the shape — measured at 50% to 71% of a bar's volume — so the stresses
//! it yields are wrong by that ratio, and neither this solver nor a patch test
//! can tell. The reasoning and the measurement are on `generate` itself.
//!
//! # Units
//!
//! Millimetre / newton / megapascal, matching [`crate::structural_solver`]:
//! lengths in mm, nodal forces in N, Young's modulus and every stress
//! component in MPa (= N/mm²), displacements in mm.
//!
//! # Method
//!
//! Linear shape functions on a tetrahedron give a constant strain per element,
//! so the element stiffness is `K_e = |V| · Bᵀ D B` with `B` (6×12) and `D`
//! (6×6) both constant. The global system is never assembled: the conjugate
//! gradient iteration applies `K` by looping elements in index order, which
//! keeps the memory linear in the element count and the summation order fixed.
//!
//! Dirichlet data is prescribed values, not just zeros, so a boundary condition
//! can impose an analytic displacement field (the FEM patch test).
//!
//! # Determinism
//!
//! Every arithmetic operation is [`Fix128`] add / sub / mul / div / sqrt.
//! Linear elasticity needs no transcendental, so nothing here reaches for
//! [`crate::det_math`] and nothing can reach a platform `libm`. The element
//! loop order is the mesh's tet order and the iteration count is bounded by the
//! configuration, so two runs on different targets produce bit-identical
//! displacements and stresses.
//!
//! # Limitations
//!
//! - Isotropic material only. `MaterialProperties` carries no Poisson's ratio,
//!   so `ElasticMaterial::from_filament` fills it from a category table and
//!   `with_poisson` overrides it.
//! - Small strain, small displacement: no geometric nonlinearity, no contact,
//!   no plasticity. Past yield the result is the *elastic* stress, which is
//!   what a yield check wants as its input.
//! - P1 tetrahedra are stiff in bending. A beam resolved by a few elements
//!   through the thickness under-predicts deflection; refine through the
//!   thickness rather than along the span.
//! - The rigid-body-mode check is a necessary condition, not a sufficient one
//!   (see `FemError::UnderConstrained`).
//!
//! Author: Moroya Sakamoto

use crate::hyperelastic::HyperelasticModel;
use crate::math::{Fix128, Mat3Fix, PolarError, Vec3Fix};
use crate::sdf_fem_mesh::SdfTetMesh;

/// Cartesian axis of a nodal degree of freedom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    /// X.
    X,
    /// Y.
    Y,
    /// Z.
    Z,
}

impl Axis {
    /// All three axes, in degree-of-freedom order.
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Index of the axis within a nodal 3-vector.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// Why a solve could not produce a stress field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FemError {
    /// The mesh has no vertices or no tetrahedra.
    EmptyMesh,
    /// Young's modulus is not positive, or Poisson's ratio is outside the
    /// range where the isotropic stiffness is positive definite.
    InvalidMaterial(&'static str),
    /// A solver setting is outside its usable range.
    InvalidConfig(&'static str),
    /// A boundary condition or a tetrahedron names a vertex the mesh does not
    /// have.
    VertexOutOfRange {
        /// The offending vertex index.
        vertex: u32,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// A tetrahedron has zero volume, so its shape function gradients are
    /// undefined.
    DegenerateElement {
        /// Index into `SdfTetMesh::tets`.
        tet: usize,
    },
    /// An element's deformation gradient has no rotation factor, so
    /// [`solve_corotational`] cannot build its frame.
    ///
    /// Kept separate from [`Self::DegenerateElement`] because that one is about
    /// the *mesh* and this one is about the *deformation*: the element was fine
    /// when it was built and the displacement is what turned it inside out, or
    /// flattened it, or exhausted the polar budget. The `cause` says which, and
    /// the four [`PolarError`] variants call for four different responses.
    RotationFailed {
        /// Index into `SdfTetMesh::tets`.
        tet: usize,
        /// Why the polar decomposition refused.
        cause: PolarError,
    },
    /// The constraints cannot remove all six rigid body modes.
    ///
    /// Reported when fewer than six degrees of freedom are prescribed (a
    /// necessary condition — three translations and three rotations need six
    /// constraints), and when the iteration meets a search direction with
    /// `pᵀKp ≤ 0`, which on a positive semi-definite stiffness means `p` is a
    /// rigid body mode. Constraints that are six or more but badly placed (all
    /// on one line, say) are *not* caught up front; those surface as
    /// [`Self::NotConverged`].
    UnderConstrained,
    /// The conjugate gradient iteration hit its budget while still making
    /// progress. Raising [`SolverConfig::max_iterations`] is the right response.
    NotConverged {
        /// Iterations performed.
        iterations: u32,
        /// Relative residual reached.
        relative_residual: Fix128,
    },
    /// The iteration stopped improving before reaching the tolerance.
    ///
    /// Distinct from [`Self::NotConverged`] on purpose, because the two call
    /// for opposite responses: more iterations fix one and do nothing for the
    /// other. Stagnation means the residual has hit the floor the conditioning
    /// and the arithmetic impose — roughly `κ(K)·ε` — so the fix is a better
    /// conditioned system (preconditioning, a less slender geometry, a coarser
    /// mesh) or a tolerance placed above that floor.
    Stagnated {
        /// Iterations performed before the iteration was abandoned.
        iterations: u32,
        /// Best relative residual reached.
        relative_residual: Fix128,
        /// Iterations the best residual went without improving.
        without_improvement: u32,
    },
}

/// Isotropic linear elastic material.
///
/// Fields are private and validated on construction, so a stiffness matrix can
/// never be built from a Poisson's ratio that makes it indefinite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticMaterial {
    youngs_modulus_mpa: Fix128,
    poissons_ratio: Fix128,
}

impl ElasticMaterial {
    /// Build from Young's modulus (MPa) and Poisson's ratio.
    ///
    /// # Errors
    ///
    /// `E <= 0`, or `ν` outside the open interval `(-1, 0.5)` — exactly the
    /// range on which the isotropic stiffness is positive definite. `ν = 0.5`
    /// is incompressible and makes the Lamé first parameter diverge.
    pub fn new(youngs_modulus_mpa: Fix128, poissons_ratio: Fix128) -> Result<Self, FemError> {
        if youngs_modulus_mpa <= Fix128::ZERO {
            return Err(FemError::InvalidMaterial(
                "Young's modulus must be positive",
            ));
        }
        if poissons_ratio <= Fix128::NEG_ONE {
            return Err(FemError::InvalidMaterial(
                "Poisson's ratio must be greater than -1",
            ));
        }
        if poissons_ratio >= half() {
            return Err(FemError::InvalidMaterial(
                "Poisson's ratio must be less than 0.5 (0.5 is incompressible)",
            ));
        }
        Ok(Self {
            youngs_modulus_mpa,
            poissons_ratio,
        })
    }

    /// Build from a filament database entry.
    ///
    /// [`crate::filament_db::MaterialProperties`] carries `youngs_modulus_gpa`
    /// (converted to MPa here) but no Poisson's ratio, so the ratio comes from
    /// [`Self::default_poissons_ratio`] for the entry's category. Override it
    /// with [`Self::with_poisson`] when a measured value is available — read
    /// that function's notes before relying on the default.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn from_filament(
        material: &crate::filament_db::MaterialProperties,
    ) -> Result<Self, FemError> {
        let e_mpa = material.youngs_modulus_gpa * Fix128::from_int(1000);
        Self::new(e_mpa, Self::default_poissons_ratio(material.category))
    }

    /// Replace the Poisson's ratio, keeping Young's modulus.
    ///
    /// # Errors
    ///
    /// As [`Self::new`].
    pub fn with_poisson(self, poissons_ratio: Fix128) -> Result<Self, FemError> {
        Self::new(self.youngs_modulus_mpa, poissons_ratio)
    }

    /// Poisson's ratio used for a material category when the database does not
    /// carry one.
    ///
    /// | category | value | the case it covers | spread across the category |
    /// |---|---|---|---|
    /// | `Fdm` | 0.35 | bulk amorphous thermoplastic (PLA / ABS / PETG) | ≈0.33-0.36 for the bulk polymer; a printed part is anisotropic and its effective ratio depends on raster angle and layer bonding |
    /// | `SheetMetal` | 0.30 | steel | aluminium alloys are ≈0.33, copper ≈0.34 — the category cannot tell them apart |
    /// | `Sla` | 0.35 | cured photopolymer resin | quoted between 0.3 and 0.4 depending on formulation and post-cure |
    /// | `Powder` | 0.40 | sintered PA12 (SLS / MJF) | metal powder beds are nearer 0.30, so this default is wrong for them |
    ///
    /// **These are widely quoted engineering values, not measurements, and this
    /// crate has not verified them against a primary source.** They are here so
    /// that `E` coming from a database is not paired with a ratio invented at
    /// the call site; they are not a substitute for a datasheet. Pass a
    /// measured value through [`Self::with_poisson`] when correctness matters.
    ///
    /// How much the choice moves the answer: for a **uniaxial** stress state
    /// the stress does not depend on `ν` at all (only the lateral strain does),
    /// so a 0.33-vs-0.36 disagreement changes nothing a yield check sees. It
    /// does matter under multiaxial or kinematically constrained loading, where
    /// `ν` enters through `λ` and grows without bound as `ν → 0.5`.
    #[must_use]
    pub fn default_poissons_ratio(category: crate::filament_db::MaterialCategory) -> Fix128 {
        use crate::filament_db::MaterialCategory as C;
        // `from_ratio` rather than `from_f64`: the value is a rational, and the
        // database entries next door are written the same way.
        match category {
            C::Fdm | C::Sla => Fix128::from_ratio(35, 100),
            C::SheetMetal => Fix128::from_ratio(30, 100),
            C::Powder => Fix128::from_ratio(40, 100),
        }
    }

    /// Young's modulus (MPa).
    #[must_use]
    pub const fn youngs_modulus_mpa(&self) -> Fix128 {
        self.youngs_modulus_mpa
    }

    /// Poisson's ratio (dimensionless).
    #[must_use]
    pub const fn poissons_ratio(&self) -> Fix128 {
        self.poissons_ratio
    }

    /// Lamé parameters `(λ, μ)` in MPa.
    ///
    /// `λ = Eν / ((1+ν)(1−2ν))`, `μ = E / (2(1+ν))`. Both denominators are
    /// non-zero for every `ν` [`Self::new`] admits.
    #[must_use]
    pub fn lame(&self) -> (Fix128, Fix128) {
        let nu = self.poissons_ratio;
        let one_plus = Fix128::ONE + nu;
        let one_minus_two = Fix128::ONE - (nu + nu);
        let lambda = self.youngs_modulus_mpa * nu / (one_plus * one_minus_two);
        let mu = self.youngs_modulus_mpa / (one_plus + one_plus);
        (lambda, mu)
    }
}

/// Prescribed displacements and nodal loads.
///
/// Every degree of freedom is either free or prescribed to a displacement.
/// Loads apply to free degrees of freedom; a load on a prescribed one is
/// ignored, because its displacement is already fixed.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundaryConditions {
    prescribed: Vec<(u32, Axis, Fix128)>,
    loads: Vec<(u32, Axis, Fix128)>,
}

impl BoundaryConditions {
    /// No constraints and no loads.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prescribe one nodal displacement component (mm).
    ///
    /// A later call for the same `(vertex, axis)` replaces the earlier value,
    /// so a face-wide sweep followed by a per-node correction does what it
    /// reads like.
    pub fn prescribe(&mut self, vertex: u32, axis: Axis, displacement_mm: Fix128) -> &mut Self {
        if let Some(slot) = self
            .prescribed
            .iter_mut()
            .find(|(v, a, _)| *v == vertex && *a == axis)
        {
            slot.2 = displacement_mm;
        } else {
            self.prescribed.push((vertex, axis, displacement_mm));
        }
        self
    }

    /// Prescribe all three components of a node to zero.
    pub fn fix(&mut self, vertex: u32) -> &mut Self {
        self.prescribe_all(vertex, [Fix128::ZERO; 3])
    }

    /// Prescribe all three components of a node to a displacement (mm).
    pub fn prescribe_all(&mut self, vertex: u32, displacement_mm: [Fix128; 3]) -> &mut Self {
        for axis in Axis::ALL {
            self.prescribe(vertex, axis, displacement_mm[axis.index()]);
        }
        self
    }

    /// Add a nodal force component (N). Repeated calls accumulate.
    pub fn add_load(&mut self, vertex: u32, axis: Axis, force_n: Fix128) -> &mut Self {
        if let Some(slot) = self
            .loads
            .iter_mut()
            .find(|(v, a, _)| *v == vertex && *a == axis)
        {
            slot.2 = slot.2 + force_n;
        } else {
            self.loads.push((vertex, axis, force_n));
        }
        self
    }

    /// The prescribed degrees of freedom, in the order they were added.
    ///
    /// Read access for solvers that live outside this module — the quadratic
    /// element in [`crate::quadratic_elastic_fem`] consumes the same boundary
    /// data and cannot reach the private field.
    #[must_use]
    pub fn prescribed(&self) -> &[(u32, Axis, Fix128)] {
        &self.prescribed
    }

    /// The nodal loads, in the order they were added. See [`Self::prescribed`].
    #[must_use]
    pub fn loads(&self) -> &[(u32, Axis, Fix128)] {
        &self.loads
    }

    /// Number of prescribed degrees of freedom.
    #[must_use]
    pub fn prescribed_count(&self) -> usize {
        self.prescribed.len()
    }

    /// Number of loaded degrees of freedom.
    #[must_use]
    pub fn load_count(&self) -> usize {
        self.loads.len()
    }
}

/// Symmetric Cauchy stress tensor (MPa).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StressTensor {
    /// σ_xx.
    pub xx: Fix128,
    /// σ_yy.
    pub yy: Fix128,
    /// σ_zz.
    pub zz: Fix128,
    /// σ_xy.
    pub xy: Fix128,
    /// σ_yz.
    pub yz: Fix128,
    /// σ_zx.
    pub zx: Fix128,
}

impl StressTensor {
    /// Von Mises equivalent stress (MPa).
    ///
    /// `√( ½[(σxx−σyy)² + (σyy−σzz)² + (σzz−σxx)²] + 3(σxy² + σyz² + σzx²) )`.
    #[must_use]
    pub fn von_mises(&self) -> Fix128 {
        let a = self.xx - self.yy;
        let b = self.yy - self.zz;
        let c = self.zz - self.xx;
        let shear = self.xy * self.xy + self.yz * self.yz + self.zx * self.zx;
        (half() * (a * a + b * b + c * c) + Fix128::from_int(3) * shear).sqrt()
    }

    /// Trace / 3 — the hydrostatic (mean) stress (MPa).
    #[must_use]
    pub fn hydrostatic(&self) -> Fix128 {
        (self.xx + self.yy + self.zz) / Fix128::from_int(3)
    }
}

/// Which preconditioner the conjugate gradient iteration uses.
///
/// The choice does not change the answer — it changes how many iterations are
/// needed and, in fixed point, what residual is reachable at all.
///
/// # Which to use
///
/// A diagonal preconditioner can only fix ill-conditioning that **is** diagonal:
/// materials of very different stiffness in one mesh, elements of very different
/// size, a badly scaled unit system. Ill-conditioning that comes from the
/// *shape* — a slender beam, a thin shell — leaves the diagonal nearly uniform,
/// so Jacobi has almost nothing to rescale and only adds arithmetic.
///
/// The measurements say the same thing. On a 10:1 cantilever the stiffness
/// diagonal has a `max/min` spread of 8.0 **at every resolution** — it does not
/// widen as the mesh refines, because the ratio between an interior node and an
/// edge node is a property of the stencil, not of the cell size. And the two
/// settings diverge with the problem:
///
/// | problem | `None` | `JacobiScaled` |
/// |---|---|---|
/// | 52 free DOFs, spread 3.2 | 34 iterations, 4.98e-10 | **32 iterations, 6.19e-11** |
/// | 19,683 free DOFs, spread 8.0 | **2,863 iterations, 1.63e-9** | 10,113 iterations, 4.51e-9 |
/// | the same, given more patience | **23,842 iterations, 1.33e-9** | 100,000 iterations, 3.91e-9 |
///
/// At the fine level `None` reaches a better residual in a thirty-fifth of the
/// iterations. [`Self::None`] is therefore the default: this crate's mesher emits
/// uniform cells of one material, which is exactly the case where a diagonal
/// preconditioner has the least to offer. Turn [`Self::JacobiScaled`] on for a
/// graded or multi-material mesh, where the diagonal carries real information —
/// and measure, rather than assume, that it helped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Preconditioner {
    /// None. The iteration works on `K` as it stands.
    #[default]
    None,
    /// Diagonal (Jacobi), scaled so its mean is one.
    ///
    /// The scaling is not cosmetic. Multiplying `M` by a constant leaves the
    /// iterates unchanged — `α` and `β` absorb it exactly — but it decides the
    /// magnitude the inner products live at, and `Fix128` has a hard floor at
    /// `2⁻⁶⁴`. With a plain `1/diag` the terms of `rᵀz` and `pᵀKp` fall below
    /// that floor and round to zero while the residual is still above tolerance.
    JacobiScaled,
}

/// Spread of the stiffness diagonal over the free degrees of freedom.
///
/// A diagnostic, not part of a solve: the ratio `max / min` bounds how much a
/// diagonal preconditioner can stretch individual components, which is what
/// decides whether scaling by the *mean* keeps every component inside the
/// representable range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagonalStats {
    /// Smallest diagonal entry over the free degrees of freedom.
    pub min: Fix128,
    /// Largest diagonal entry over the free degrees of freedom.
    pub max: Fix128,
    /// Mean diagonal entry over the free degrees of freedom.
    pub mean: Fix128,
    /// Number of free degrees of freedom the statistics cover.
    pub free_dofs: usize,
}

/// Conjugate gradient stopping rule.
///
/// Fields are private: both are load-bearing (one decides when the answer is
/// good enough, the other decides when to give up) and a caller that sets them
/// by struct literal gets no validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolverConfig {
    max_iterations: u32,
    relative_tolerance: Fix128,
    stagnation_min_window: u32,
    stagnation_window_fraction: Fix128,
    stagnation_min_improvement: Fix128,
    preconditioner: Preconditioner,
}

impl SolverConfig {
    /// Explicit budget and tolerance.
    ///
    /// # Errors
    ///
    /// `max_iterations == 0`, or `relative_tolerance` outside `(0, 1)` — a
    /// tolerance of 1 or more accepts the zero vector as a solution.
    pub fn try_new(max_iterations: u32, relative_tolerance: Fix128) -> Result<Self, FemError> {
        if max_iterations == 0 {
            return Err(FemError::InvalidConfig("max_iterations must be positive"));
        }
        if relative_tolerance <= Fix128::ZERO {
            return Err(FemError::InvalidConfig(
                "relative_tolerance must be positive",
            ));
        }
        if relative_tolerance >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "relative_tolerance of 1 or more accepts the zero vector",
            ));
        }
        Ok(Self {
            max_iterations,
            relative_tolerance,
            ..Self::default()
        })
    }

    /// Replace the floor of the stagnation window and the improvement that
    /// counts as progress.
    ///
    /// The window itself is **not** this number — see
    /// [`Self::with_stagnation_fraction`]. This is the smallest it may be, which
    /// only binds on short solves.
    ///
    /// Progress is measured against the *best* residual so far, not the
    /// previous one. The residual norm of conjugate gradient is not monotone,
    /// and a rule comparing consecutive values would abandon a healthy solve on
    /// an oscillation.
    ///
    /// # Errors
    ///
    /// `min_window == 0`, or `min_improvement` outside `(0, 1)`.
    pub fn with_stagnation(
        self,
        min_window: u32,
        min_improvement: Fix128,
    ) -> Result<Self, FemError> {
        if min_window == 0 {
            return Err(FemError::InvalidConfig(
                "stagnation window must be positive",
            ));
        }
        if min_improvement <= Fix128::ZERO || min_improvement >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "stagnation improvement must be a fraction strictly between 0 and 1",
            ));
        }
        Ok(Self {
            stagnation_min_window: min_window,
            stagnation_min_improvement: min_improvement,
            ..self
        })
    }

    /// How the stagnation window scales with the work already done.
    ///
    /// The window is `max(min_window, fraction × iterations so far)`.
    ///
    /// A fixed window cannot work, because the iteration count of conjugate
    /// gradient grows with the problem: measured on a 10:1 cantilever, 254
    /// iterations at 3,200 elements and over 2,400 at 25,600. A window generous
    /// enough for the small problem cuts the large one off while it is still
    /// improving — a fixed 1,000 abandoned the 25,600-element solve at a
    /// relative residual of 1.63e-9, while running the same solve to a 500,000
    /// iteration budget reached 9.41e-10 and was still improving.
    ///
    /// Scaling by the work done makes the rule scale free, and it still
    /// terminates: the iteration stops at `last / (1 − fraction)`, where `last`
    /// is the iteration of the most recent improvement. The default `0.5` reads
    /// *"give up at twice the iteration that produced your best result"*.
    ///
    /// **`fraction` has to be below 1.** At exactly 1 the condition is
    /// `iterations − last ≥ iterations`, which needs `last ≤ 0`; the rule can
    /// then never fire and a hopeless solve runs the whole budget. That is not
    /// hypothetical — a default of `1` was written here first and was caught on
    /// the unreachable-tolerance scene immediately: the same solve went from
    /// stopping at 1,034 iterations to using all 200,000 and returning
    /// `NotConverged`.
    ///
    /// ⚠️ **The test that caught it no longer carries that name.** The scene is
    /// now `tolerance_below_the_floor_is_clamped_and_reported` in
    /// `tests/analytic_linear_elastic_fem.rs`, and the rejection above is pinned
    /// by `config_rejects_a_stagnation_fraction_that_can_never_fire` in
    /// `coupled_iteration`. Neither of them asserts that the **default**
    /// `fraction` is below one; see the Backlog.
    ///
    /// # Errors
    ///
    /// `fraction` is outside `(0, 1)`.
    pub fn with_stagnation_fraction(self, fraction: Fix128) -> Result<Self, FemError> {
        if fraction <= Fix128::ZERO || fraction >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "stagnation window fraction must be strictly between 0 and 1; at 1 the rule \
                 can never fire",
            ));
        }
        Ok(Self {
            stagnation_window_fraction: fraction,
            ..self
        })
    }

    /// Fraction of the iterations so far that the window scales with.
    #[must_use]
    pub const fn stagnation_window_fraction(&self) -> Fix128 {
        self.stagnation_window_fraction
    }

    /// Choose the preconditioner.
    #[must_use]
    pub const fn with_preconditioner(self, preconditioner: Preconditioner) -> Self {
        Self {
            preconditioner,
            ..self
        }
    }

    /// Which preconditioner this configuration uses.
    #[must_use]
    pub const fn preconditioner(&self) -> Preconditioner {
        self.preconditioner
    }

    /// Smallest the stagnation window may be, whatever the iteration count.
    #[must_use]
    pub const fn stagnation_min_window(&self) -> u32 {
        self.stagnation_min_window
    }

    /// Relative improvement in the best residual that counts as progress.
    #[must_use]
    pub const fn stagnation_min_improvement(&self) -> Fix128 {
        self.stagnation_min_improvement
    }

    /// Iteration budget.
    #[must_use]
    pub const fn max_iterations(&self) -> u32 {
        self.max_iterations
    }

    /// Relative residual at which the iteration stops.
    #[must_use]
    pub const fn relative_tolerance(&self) -> Fix128 {
        self.relative_tolerance
    }
}

impl Default for SolverConfig {
    /// 10,000 iterations, relative residual 2⁻³⁰ (≈ 9.3e-10).
    ///
    /// Conjugate gradient converges in at most `n` iterations in exact
    /// arithmetic, so on a well-conditioned mesh the budget never binds; it is
    /// there so an ill-conditioned one reports [`FemError::NotConverged`]
    /// rather than running forever.
    ///
    /// The tolerance is a power of two, so it is exact in [`Fix128`], and it
    /// sits well clear of the floor the representation imposes: the residual
    /// norm is `√(rᵀr)` and `rᵀr` cannot go below 2⁻⁶⁴, so residual norms under
    /// about 2⁻³² are indistinguishable from zero. A tolerance below that floor
    /// would be unreachable and would turn a converged solve into
    /// [`FemError::NotConverged`].
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            relative_tolerance: Fix128::from_raw(0, 1 << 34),
            // The window is `max(500, 0.5 × iterations so far)`, so a hopeless
            // solve is abandoned at twice the iteration that produced its best
            // residual, however large the problem is. 500 is a floor for short
            // solves, where a proportional window would be a handful of
            // iterations. 2^-10 ≈ 0.1% is small enough that genuine progress
            // always clears it.
            stagnation_min_window: 500,
            stagnation_window_fraction: half(),
            stagnation_min_improvement: Fix128::from_raw(0, 1 << 54),
            preconditioner: Preconditioner::None,
        }
    }
}

/// Displacement and stress field produced by [`solve`].
#[derive(Debug, Clone, PartialEq)]
pub struct FemSolution {
    /// Nodal displacement (mm), indexed like `SdfTetMesh::vertices`.
    pub displacements: Vec<[Fix128; 3]>,
    /// Cauchy stress (MPa) per element, indexed like `SdfTetMesh::tets`.
    /// Constant within the element, because P1 strain is constant.
    pub element_stress: Vec<StressTensor>,
    /// Conjugate gradient iterations performed.
    pub iterations: u32,
    /// `‖r‖ / ‖b‖` at the final iteration. Zero when the load vector is zero.
    pub relative_residual: Fix128,
    /// The relative tolerance the iteration was actually held to.
    ///
    /// Equal to [`SolverConfig::relative_tolerance`] unless that tolerance would
    /// have demanded a residual norm the arithmetic cannot represent, in which
    /// case it is the floor instead — see [`RESIDUAL_NORM_FLOOR`]. Compare the
    /// two to find out whether the request was met on its own terms or on the
    /// arithmetic's.
    pub effective_relative_tolerance: Fix128,
}

impl FemSolution {
    /// Largest von Mises stress over all elements (MPa).
    ///
    /// The number a yield check compares against
    /// [`crate::filament_db::MaterialProperties::yield_strength_mpa`]. Zero for
    /// a solution with no elements.
    #[must_use]
    pub fn max_von_mises_mpa(&self) -> Fix128 {
        self.element_stress
            .iter()
            .map(StressTensor::von_mises)
            .max()
            .unwrap_or(Fix128::ZERO)
    }
}

/// Smallest residual norm the iteration is asked to reach, whatever the
/// configured tolerance says.
///
/// `‖r‖ = √(rᵀr)` and `rᵀr` is a [`Fix128`], so it is an integer multiple of
/// `2⁻⁶⁴` and `‖r‖` is `√k · 2⁻³²`. **The residual norm is quantised**, and the
/// steps near the bottom are enormous in relative terms: the sequence of
/// reachable values is `0`, then `2⁻³²`, `√2·2⁻³²`, `√3·2⁻³²`, … Measured on the
/// cantilever, `rᵀr` at stagnation was raw `8` — eight units of `2⁻⁶⁴` — giving
/// `‖r‖ = 6.585e-10`, and the relative residuals seen across every run stood in
/// the ratio `√3 : √2 : 1`, which is this quantisation and nothing else.
///
/// A *relative* tolerance therefore has a floor of `2⁻³² / ‖b‖`, and that floor
/// **moves with the load vector**. It is not a constant of the crate: the same
/// total force spread over a finer mesh gives each node less, so `‖b‖` shrinks
/// and the floor rises. Measured on one beam at four resolutions:
///
/// | degrees of freedom | `‖b‖` | floor `2⁻³²/‖b‖` |
/// |---|---|---|
/// | 72 | 2.108 | 1.10e-10 |
/// | 297 | 1.633 | 1.43e-10 |
/// | 1,575 | 0.928 | 2.51e-10 |
/// | 9,963 | 0.495 | 4.71e-10 |
///
/// So a fixed relative tolerance is a trap: `2⁻³⁰` is comfortable at 72 degrees
/// of freedom and unreachable at 9,963, where the smallest attainable relative
/// residual is 9.41e-10 against a tolerance of 9.31e-10 — short by 1%, forever.
/// Before this floor existed that cost 500,000 iterations and 24.8 minutes to
/// discover, twice.
///
/// Four units of `2⁻³²` leaves room for the quantisation to land a step or two
/// above the very bottom without the answer flipping between converged and not.
pub const RESIDUAL_NORM_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 34);

/// One half, exactly.
#[inline]
fn half() -> Fix128 {
    Fix128::from_raw(0, 1 << 63)
}

#[inline]
fn sub3(a: [Fix128; 3], b: [Fix128; 3]) -> [Fix128; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [Fix128; 3], b: [Fix128; 3]) -> [Fix128; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: [Fix128; 3], b: [Fix128; 3]) -> Fix128 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn div3(a: [Fix128; 3], d: Fix128) -> [Fix128; 3] {
    [a[0] / d, a[1] / d, a[2] / d]
}

/// Constant per-element quantities: shape function gradients (1/mm) and
/// volume (mm³).
///
/// The reference coordinates do not appear: they enter through `∇N` and `|V|`
/// and nowhere else, including in the co-rotational path, whose strain
/// `Rᵀ F − I` is built from `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ`.
#[derive(Clone, Copy)]
struct Element {
    nodes: [usize; 4],
    grad: [[Fix128; 3]; 4],
    volume: Fix128,
}

/// Precompute `∇N` and `|V|` for every tetrahedron.
///
/// With `J = [p₁−p₀, p₂−p₀, p₃−p₀]` as columns, the rows of `J⁻¹` are the
/// gradients of `N₁, N₂, N₃`, and `∇N₀ = −(∇N₁+∇N₂+∇N₃)` because the four
/// shape functions sum to one everywhere. The volume uses `|det J| / 6`, so a
/// tetrahedron wound the other way contributes the same stiffness.
fn build_elements(mesh: &SdfTetMesh) -> Result<Vec<Element>, FemError> {
    let vertex_count = mesh.vertices.len();
    let six = Fix128::from_int(6);
    let mut elements = Vec::with_capacity(mesh.tets.len());
    for (t, tet) in mesh.tets.iter().enumerate() {
        let mut nodes = [0usize; 4];
        let mut p = [[Fix128::ZERO; 3]; 4];
        for (i, &v) in tet.vertices.iter().enumerate() {
            let idx = v as usize;
            if idx >= vertex_count {
                return Err(FemError::VertexOutOfRange {
                    vertex: v,
                    vertex_count,
                });
            }
            nodes[i] = idx;
            let q = mesh.vertices[idx];
            p[i] = [
                Fix128::from_f32(q[0]),
                Fix128::from_f32(q[1]),
                Fix128::from_f32(q[2]),
            ];
        }
        let e1 = sub3(p[1], p[0]);
        let e2 = sub3(p[2], p[0]);
        let e3 = sub3(p[3], p[0]);
        let det = dot3(e1, cross3(e2, e3));
        if det.is_zero() {
            return Err(FemError::DegenerateElement { tet: t });
        }
        let g1 = div3(cross3(e2, e3), det);
        let g2 = div3(cross3(e3, e1), det);
        let g3 = div3(cross3(e1, e2), det);
        let g0 = [
            -(g1[0] + g2[0] + g3[0]),
            -(g1[1] + g2[1] + g3[1]),
            -(g1[2] + g2[2] + g3[2]),
        ];
        elements.push(Element {
            nodes,
            grad: [g0, g1, g2, g3],
            volume: det.abs() / six,
        });
    }
    Ok(elements)
}

/// The four nodal values of a global vector belonging to one element.
#[inline]
fn gather(element: &Element, u: &[Fix128]) -> [[Fix128; 3]; 4] {
    let mut local = [[Fix128::ZERO; 3]; 4];
    for (slot, &node) in local.iter_mut().zip(element.nodes.iter()) {
        let base = node * 3;
        *slot = [u[base], u[base + 1], u[base + 2]];
    }
    local
}

/// `σ = D B u_e` for one element, from its **four nodal vectors** (constant
/// over the element).
///
/// Taking the nodal vectors rather than the global array is what lets the
/// co-rotational path reuse this: the tangent feeds it `Rᵀ Δu` and the internal
/// force feeds it `Rᵀx − X`, and both then share one `B` and one `D`. The
/// small-strain path feeds it the plain nodal displacements through
/// [`gather`], which is the same arithmetic in the same order as before.
fn element_stress_local(
    element: &Element,
    u: &[[Fix128; 3]; 4],
    lambda: Fix128,
    mu: Fix128,
) -> StressTensor {
    let mut exx = Fix128::ZERO;
    let mut eyy = Fix128::ZERO;
    let mut ezz = Fix128::ZERO;
    let mut gxy = Fix128::ZERO;
    let mut gyz = Fix128::ZERO;
    let mut gzx = Fix128::ZERO;
    for (g, node_u) in element.grad.iter().zip(u.iter()) {
        let (ux, uy, uz) = (node_u[0], node_u[1], node_u[2]);
        exx = exx + g[0] * ux;
        eyy = eyy + g[1] * uy;
        ezz = ezz + g[2] * uz;
        gxy = gxy + g[1] * ux + g[0] * uy;
        gyz = gyz + g[2] * uy + g[1] * uz;
        gzx = gzx + g[2] * ux + g[0] * uz;
    }
    let trace = exx + eyy + ezz;
    let two_mu = mu + mu;
    StressTensor {
        xx: lambda * trace + two_mu * exx,
        yy: lambda * trace + two_mu * eyy,
        zz: lambda * trace + two_mu * ezz,
        xy: mu * gxy,
        yz: mu * gyz,
        zx: mu * gzx,
    }
}

/// `σ = D B u_e` for one element, reading the global displacement array.
fn element_stress(element: &Element, u: &[Fix128], lambda: Fix128, mu: Fix128) -> StressTensor {
    element_stress_local(element, &gather(element, u), lambda, mu)
}

/// `Kₑ⁰ u_e` — the element's internal force from its four nodal vectors (N).
fn element_force_local(
    element: &Element,
    u: &[[Fix128; 3]; 4],
    lambda: Fix128,
    mu: Fix128,
) -> [[Fix128; 3]; 4] {
    element_force_from_stress(element, element_stress_local(element, u, lambda, mu))
}

/// `out = K u`, accumulated element by element in mesh order.
fn apply_stiffness(
    elements: &[Element],
    u: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    out: &mut [Fix128],
) {
    out.fill(Fix128::ZERO);
    for element in elements {
        let force = element_force_local(element, &gather(element, u), lambda, mu);
        for (f, &node) in force.iter().zip(element.nodes.iter()) {
            let base = node * 3;
            out[base] = out[base] + f[0];
            out[base + 1] = out[base + 1] + f[1];
            out[base + 2] = out[base + 2] + f[2];
        }
    }
}

/// Diagonal of the global stiffness, assembled element by element.
///
/// For node `i` with shape function gradient `g`, the diagonal entry of
/// `Bᵢᵀ D Bᵢ` on axis `a` works out to `(λ+2μ)·g_a² + μ·(g_b² + g_c²)` with
/// `b, c` the other two axes — the normal row of `D` contributes the first term
/// and the two shear rows the second. Scaled by the element volume and summed,
/// that is `diag(K)` without ever forming `K`.
fn stiffness_diagonal(
    elements: &[Element],
    lambda: Fix128,
    mu: Fix128,
    ndof: usize,
) -> Vec<Fix128> {
    let mut diag = vec![Fix128::ZERO; ndof];
    let lambda_2mu = lambda + mu + mu;
    for element in elements {
        for (g, &node) in element.grad.iter().zip(element.nodes.iter()) {
            let sq = [g[0] * g[0], g[1] * g[1], g[2] * g[2]];
            let base = node * 3;
            for axis in 0..3 {
                let others = sq[(axis + 1) % 3] + sq[(axis + 2) % 3];
                diag[base + axis] =
                    diag[base + axis] + element.volume * (lambda_2mu * sq[axis] + mu * others);
            }
        }
    }
    diag
}

fn dot(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc + *x * *y;
    }
    acc
}

/// What one conjugate gradient solve produced.
struct CgResult {
    /// The solution on the free degrees of freedom, zero on the prescribed
    /// ones.
    x: Vec<Fix128>,
    iterations: u32,
    residual_norm: Fix128,
    b_norm: Fix128,
    /// The residual norm the iteration was actually held to, after the floor.
    target: Fix128,
}

/// `mean(diag) / diag` on the free degrees of freedom, zero elsewhere.
///
/// The scaling by the mean rather than the plain reciprocal is load-bearing —
/// see [`Preconditioner::JacobiScaled`].
///
/// # Errors
///
/// [`FemError::UnderConstrained`] when a free degree of freedom has a
/// non-positive diagonal entry, which on a positive definite stiffness means it
/// is attached to no element at all.
fn build_preconditioner(
    diag: &[Fix128],
    is_free: &[bool],
    config: &SolverConfig,
) -> Result<Vec<Fix128>, FemError> {
    let ndof = diag.len();
    let mut diag_sum = Fix128::ZERO;
    let mut free_count = 0u32;
    for (d, value) in diag.iter().enumerate() {
        if !is_free[d] {
            continue;
        }
        if *value <= Fix128::ZERO {
            return Err(FemError::UnderConstrained);
        }
        diag_sum = diag_sum + *value;
        free_count += 1;
    }
    // `free_count == 0` means every degree of freedom is prescribed. The
    // iteration then never runs (the load vector is zero, so the residual
    // starts at zero) and the preconditioner is never read, but the mean would
    // divide by zero.
    let mean_diag = if free_count == 0 {
        Fix128::ONE
    } else {
        diag_sum / Fix128::from_int(i64::from(free_count))
    };
    let mut precond = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            precond[d] = match config.preconditioner {
                Preconditioner::JacobiScaled => mean_diag / diag[d],
                _ => Fix128::ONE,
            };
        }
    }
    Ok(precond)
}

/// Preconditioned conjugate gradient on the free block of a symmetric positive
/// definite operator.
///
/// The operator arrives as a closure rather than as `(elements, λ, μ)` because
/// two solvers need it: the small-strain [`solve`] applies `K`, and
/// [`solve_corotational`] applies `Σₑ R Kₑ⁰ Rᵀ` with a rotation that changes
/// every Newton iteration. Sharing the iteration keeps the stopping rule, the
/// stagnation window and the residual floor in one place — three pieces of
/// behaviour that were measured into their present shape and that a second copy
/// would drift away from.
///
/// `x`, `r`, `p` and `z` stay zero on the prescribed degrees of freedom, so the
/// closure may run over the whole vector without a scatter/gather step; it is
/// the caller's job to zero the prescribed entries of its output.
///
/// # Errors
///
/// [`FemError::NotConverged`], [`FemError::Stagnated`] or
/// [`FemError::UnderConstrained`], as documented on each variant.
fn conjugate_gradient<A>(
    b: &[Fix128],
    is_free: &[bool],
    precond: &[Fix128],
    config: &SolverConfig,
    mut apply: A,
) -> Result<CgResult, FemError>
where
    A: FnMut(&[Fix128], &mut [Fix128]),
{
    let ndof = b.len();
    let mut scratch = vec![Fix128::ZERO; ndof];
    let mut x = vec![Fix128::ZERO; ndof];
    let mut r = b.to_vec();
    let mut z = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            z[d] = r[d] * precond[d];
        }
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(b, b).sqrt();
    // Never ask for a residual norm the representation cannot express; see
    // `RESIDUAL_NORM_FLOOR`.
    let requested = config.relative_tolerance * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };

    let mut iterations = 0u32;
    let mut residual_norm = dot(&r, &r).sqrt();
    // Stagnation bookkeeping: the best residual seen and how long ago it was
    // beaten by the configured margin.
    let mut best_residual = residual_norm;
    let mut since_improvement = 0u32;

    while residual_norm > target {
        if iterations >= config.max_iterations {
            return Err(FemError::NotConverged {
                iterations,
                relative_residual: relative(residual_norm, b_norm),
            });
        }
        // `max(min_window, fraction × iterations so far)`: a fixed window does
        // not scale, because the iteration count grows with the problem.
        let window = {
            let scaled =
                config.stagnation_window_fraction * Fix128::from_int(i64::from(iterations));
            let scaled = if scaled.is_negative() {
                0
            } else {
                u32::try_from(scaled.hi).unwrap_or(u32::MAX)
            };
            scaled.max(config.stagnation_min_window)
        };
        if since_improvement >= window {
            return Err(FemError::Stagnated {
                iterations,
                relative_residual: relative(best_residual, b_norm),
                without_improvement: since_improvement,
            });
        }
        apply(&p, &mut scratch);
        for (d, value) in scratch.iter_mut().enumerate() {
            if !is_free[d] {
                *value = Fix128::ZERO;
            }
        }
        let pkp = dot(&p, &scratch);
        if pkp <= Fix128::ZERO {
            if iterations == 0 {
                // The very first search direction is `M⁻¹b`, so a non-positive
                // `pᵀKp` there means `K` is singular in that direction: a rigid
                // body mode the constraints did not remove.
                return Err(FemError::UnderConstrained);
            }
            // Later on, `K` has already proved positive definite along every
            // direction tried, so this is the search direction having shrunk
            // until its inner product no longer registers. That is the
            // arithmetic floor, which is what `Stagnated` describes.
            return Err(FemError::Stagnated {
                iterations,
                relative_residual: relative(best_residual, b_norm),
                without_improvement: since_improvement,
            });
        }
        let alpha = rz / pkp;
        for d in 0..ndof {
            if is_free[d] {
                x[d] = x[d] + alpha * p[d];
                r[d] = r[d] - alpha * scratch[d];
            }
        }
        for d in 0..ndof {
            if is_free[d] {
                z[d] = r[d] * precond[d];
            }
        }
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        for d in 0..ndof {
            if is_free[d] {
                p[d] = z[d] + beta * p[d];
            }
        }
        rz = rz_next;
        residual_norm = dot(&r, &r).sqrt();
        iterations += 1;

        // Progress is measured against the best residual so far, not the
        // previous one: the conjugate gradient residual norm is not monotone.
        if residual_norm < best_residual - best_residual * config.stagnation_min_improvement {
            best_residual = residual_norm;
            since_improvement = 0;
        } else {
            if residual_norm < best_residual {
                best_residual = residual_norm;
            }
            since_improvement += 1;
        }
    }

    Ok(CgResult {
        x,
        iterations,
        residual_norm,
        b_norm,
        target,
    })
}

/// Solve the linear elastic boundary value problem on `mesh`.
///
/// # Errors
///
/// See [`FemError`].
pub fn solve(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &SolverConfig,
) -> Result<FemSolution, FemError> {
    let vertex_count = mesh.vertices.len();
    if vertex_count == 0 || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    for &(vertex, _, _) in boundary.prescribed.iter().chain(boundary.loads.iter()) {
        if vertex as usize >= vertex_count {
            return Err(FemError::VertexOutOfRange {
                vertex,
                vertex_count,
            });
        }
    }

    let elements = build_elements(mesh)?;
    let (lambda, mu) = material.lame();
    let ndof = vertex_count * 3;

    // Dirichlet data, zero on the free degrees of freedom.
    let mut prescribed_value = vec![Fix128::ZERO; ndof];
    let mut is_free = vec![true; ndof];
    for &(vertex, axis, value) in &boundary.prescribed {
        let d = vertex as usize * 3 + axis.index();
        is_free[d] = false;
        prescribed_value[d] = value;
    }
    let constrained = is_free.iter().filter(|f| !**f).count();
    if constrained < 6 {
        // Three translations and three rotations need six constraints; fewer
        // leaves the stiffness singular whatever the mesh looks like.
        return Err(FemError::UnderConstrained);
    }

    // b = f_ext − K u_prescribed, restricted to the free degrees of freedom.
    let mut scratch = vec![Fix128::ZERO; ndof];
    apply_stiffness(&elements, &prescribed_value, lambda, mu, &mut scratch);
    let mut b = vec![Fix128::ZERO; ndof];
    for &(vertex, axis, force) in &boundary.loads {
        let d = vertex as usize * 3 + axis.index();
        if is_free[d] {
            b[d] = b[d] + force;
        }
    }
    for (d, value) in b.iter_mut().enumerate() {
        if is_free[d] {
            *value = *value - scratch[d];
        } else {
            *value = Fix128::ZERO;
        }
    }

    // Jacobi-preconditioned conjugate gradient on the free block.
    //
    // The preconditioner is the reciprocal of `diag(K)`. It costs one division
    // per free degree of freedom per iteration and leaves the answer unchanged;
    // what it changes is the conditioning the iteration sees, and with it both
    // the iteration count and the residual floor the arithmetic can reach. A
    // slender beam is exactly the case where that matters: the condition number
    // grows as `(L/t)²/h²`, and an unpreconditioned solve on a 10:1 beam at
    // 25,600 elements was measured stalling 1% above a 2⁻³⁰ tolerance after
    // 500,000 iterations.
    let diag = stiffness_diagonal(&elements, lambda, mu, ndof);
    let precond = build_preconditioner(&diag, &is_free, config)?;

    let cg = conjugate_gradient(&b, &is_free, &precond, config, |p, out| {
        apply_stiffness(&elements, p, lambda, mu, out);
    })?;
    let mut x = cg.x;

    // Recombine the prescribed and solved parts.
    for (d, value) in x.iter_mut().enumerate() {
        if !is_free[d] {
            *value = prescribed_value[d];
        }
    }

    let displacements = (0..vertex_count)
        .map(|v| [x[v * 3], x[v * 3 + 1], x[v * 3 + 2]])
        .collect();
    let element_stress = elements
        .iter()
        .map(|e| element_stress(e, &x, lambda, mu))
        .collect();

    Ok(FemSolution {
        displacements,
        element_stress,
        iterations: cg.iterations,
        relative_residual: relative(cg.residual_norm, cg.b_norm),
        effective_relative_tolerance: relative(cg.target, cg.b_norm),
    })
}

/// `‖r‖ / ‖b‖`, defined as zero when the load vector is zero (the solution is
/// then exactly the prescribed field and there is nothing to converge to).
#[inline]
fn relative(residual_norm: Fix128, b_norm: Fix128) -> Fix128 {
    if b_norm.is_zero() {
        Fix128::ZERO
    } else {
        residual_norm / b_norm
    }
}

/// Spread of the stiffness diagonal over the free degrees of freedom.
///
/// Built from the same element data a solve uses, so it answers "how far apart
/// are the diagonal entries this preconditioner has to cope with" without
/// running an iteration.
///
/// # Errors
///
/// As [`solve`], for the mesh and boundary condition checks it shares.
pub fn stiffness_diagonal_stats(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
) -> Result<DiagonalStats, FemError> {
    let vertex_count = mesh.vertices.len();
    if vertex_count == 0 || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    for &(vertex, _, _) in boundary.prescribed.iter().chain(boundary.loads.iter()) {
        if vertex as usize >= vertex_count {
            return Err(FemError::VertexOutOfRange {
                vertex,
                vertex_count,
            });
        }
    }
    let elements = build_elements(mesh)?;
    let (lambda, mu) = material.lame();
    let ndof = vertex_count * 3;
    let mut is_free = vec![true; ndof];
    for &(vertex, axis, _) in &boundary.prescribed {
        is_free[vertex as usize * 3 + axis.index()] = false;
    }
    let diag = stiffness_diagonal(&elements, lambda, mu, ndof);

    let mut min = Fix128::ZERO;
    let mut max = Fix128::ZERO;
    let mut sum = Fix128::ZERO;
    let mut count = 0usize;
    for (d, value) in diag.iter().enumerate() {
        if !is_free[d] {
            continue;
        }
        if count == 0 || *value < min {
            min = *value;
        }
        if count == 0 || *value > max {
            max = *value;
        }
        sum = sum + *value;
        count += 1;
    }
    if count == 0 {
        return Err(FemError::UnderConstrained);
    }
    Ok(DiagonalStats {
        min,
        max,
        mean: sum / Fix128::from_int(i64::try_from(count).unwrap_or(i64::MAX)),
        free_dofs: count,
    })
}

// ---------------------------------------------------------------------------
// co-rotational solve
// ---------------------------------------------------------------------------

/// Smallest `det F` a co-rotational element may have before its rotation is
/// refused, as `2⁻²⁰ ≈ 9.5e-7`.
///
/// [`Mat3Fix::polar_rotation`] already refuses `det F ≤ 0` unconditionally, so
/// this floor is not what keeps a reflection out; it is what keeps a *nearly*
/// inverted element from producing a rotation whose accuracy no budget can
/// recover. A volume ratio of a millionth is four decades past anything a mesh
/// this solver is meant for should reach, so an element under the floor is a
/// broken model rather than a hard one.
const POLAR_DET_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 44);

/// Change below which two sets of element frames count as the same, in units
/// of `2⁻⁶⁴`.
///
/// `Mat3Fix::polar_rotation` stops when its own iterate moves by four of these,
/// so four is the resolution a frame is determined to in the first place and
/// asking the outer iteration to reproduce a frame *exactly* asks for something
/// the polar decomposition does not promise. Measured: the frames of this
/// solver converge quadratically to a spread of five to eight units and then
/// wander inside it indefinitely, so an exact-equality stopping rule never
/// fires. Sixteen is above the measured wander with margin and still four
/// decades below anything a stress depends on.
const FRAME_SETTLED: Fix128 = Fix128 { hi: 0, lo: 16 };

/// Whether every element frame moved by less than [`FRAME_SETTLED`].
fn frames_settled(next: &[Mat3Fix], previous: &[Mat3Fix]) -> bool {
    for (a, b) in next.iter().zip(previous.iter()) {
        for (ca, cb) in [(a.col0, b.col0), (a.col1, b.col1), (a.col2, b.col2)] {
            for (x, y) in [(ca.x, cb.x), (ca.y, cb.y), (ca.z, cb.z)] {
                if (x - y).abs() >= FRAME_SETTLED {
                    return false;
                }
            }
        }
    }
    true
}

/// Largest absolute entry of a vector.
///
/// The Newton iteration measures its residual this way and **not** as
/// `√(rᵀr)`. The Euclidean norm squares before it sums, and `Fix128` holds
/// `2⁻⁶⁴`, so every entry below `2⁻³²` contributes exactly zero to `rᵀr`: a
/// residual of `1e-10` N per node reports a norm of **exactly zero** while the
/// residual vector itself is still six decades above the resolution of the
/// format. Measured, that is where the Newton iteration stopped — not at a
/// solution, but at the point where its own instrument went blind, and the
/// place it stopped depended on the path that got there. The largest entry
/// never squares, so it stays meaningful to the last bit, and it is the
/// physically legible quantity anyway: the biggest out-of-balance force at any
/// node.
fn max_abs(v: &[Fix128]) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for entry in v {
        let a = entry.abs();
        if a > worst {
            worst = a;
        }
    }
    worst
}

/// Settings for [`solve_corotational`].
///
/// Fields are private and validated, for the reason [`SolverConfig`] gives:
/// every one of them is load-bearing and a struct literal would get no checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorotationalConfig {
    linear: SolverConfig,
    newton_iterations: u32,
    newton_tolerance: Fix128,
    increments: u32,
    polar_iterations: u32,
    material: Option<HyperelasticModel>,
}

impl CorotationalConfig {
    /// Explicit settings.
    ///
    /// - `linear` drives the conjugate gradient that solves each Newton step.
    /// - `newton_iterations` bounds the Newton iterations **per increment**.
    /// - `newton_tolerance` is the residual the Newton iteration is held to, as
    ///   a fraction of the reference residual described on
    ///   [`solve_corotational`]. It is *not* a fraction of the residual this
    ///   increment happened to start at — see that function for why.
    /// - `increments` splits the prescribed displacement into equal fractions,
    ///   which gives Newton a nearby starting point for a large rotation.
    /// - `polar_iterations` is the budget for [`Mat3Fix::polar_rotation`] per
    ///   element per Newton step.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] when a count is zero or the tolerance is
    /// outside `(0, 1)`.
    pub fn try_new(
        linear: SolverConfig,
        newton_iterations: u32,
        newton_tolerance: Fix128,
        increments: u32,
        polar_iterations: u32,
    ) -> Result<Self, FemError> {
        if newton_iterations == 0 {
            return Err(FemError::InvalidConfig(
                "newton_iterations must be positive",
            ));
        }
        if increments == 0 {
            return Err(FemError::InvalidConfig("increments must be positive"));
        }
        if polar_iterations == 0 {
            return Err(FemError::InvalidConfig("polar_iterations must be positive"));
        }
        if newton_tolerance <= Fix128::ZERO || newton_tolerance >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "newton_tolerance must be a fraction strictly between 0 and 1",
            ));
        }
        Ok(Self {
            linear,
            newton_iterations,
            newton_tolerance,
            increments,
            polar_iterations,
            material: None,
        })
    }

    /// Evaluate the element stress with `model` instead of the co-rotational
    /// linear law.
    ///
    /// # What changes
    ///
    /// Without this the element carries `σ = R·(λ tr ε I + 2μ ε)·Rᵀ` with
    /// `ε = sym(RᵀF − I)`: exact for large *rotation*, linear in *stretch*. With
    /// it the element carries [`crate::hyperelastic::cauchy_stress`] evaluated at
    /// the deformation gradient, which is a large-stretch law — the difference
    /// is 46 % of the linear value at a 125 % principal stretch, measured by the
    /// twins in `tests/analytic_corotational.rs`.
    ///
    /// `μ` (or `C₁`, `C₂`, `C₃`) comes from `model`; the bulk modulus that fixes
    /// the pressure comes from the [`ElasticMaterial`] passed to
    /// [`solve_corotational`], as `K = λ + 2μ/3`. The two are therefore allowed
    /// to disagree, and nothing checks that they describe the same solid — that
    /// is the caller's to keep straight.
    ///
    /// # ⚠️ What it costs
    ///
    /// The residual stops being affine in the displacement, so the Newton step
    /// is no longer exact and the **frame-settling stopping rule no longer
    /// applies**: with a material law the last increment stops on the residual
    /// like every other one. The increment independence
    /// [`solve_corotational`] documents is therefore **not** claimed here — it is
    /// a property of the linear law's exact step. The tangent stays the
    /// co-rotational linear one, which makes this a modified Newton iteration:
    /// it decides how many steps, not where they land.
    #[must_use]
    pub const fn with_hyperelastic(mut self, model: HyperelasticModel) -> Self {
        self.material = Some(model);
        self
    }

    /// The material law the element stress is evaluated with, or `None` for the
    /// co-rotational linear law.
    #[must_use]
    pub const fn hyperelastic(&self) -> Option<HyperelasticModel> {
        self.material
    }

    /// The conjugate gradient settings each Newton step is solved with.
    #[must_use]
    pub const fn linear(&self) -> SolverConfig {
        self.linear
    }

    /// Newton iteration budget per increment.
    #[must_use]
    pub const fn newton_iterations(&self) -> u32 {
        self.newton_iterations
    }

    /// Residual fraction at which the Newton iteration stops.
    #[must_use]
    pub const fn newton_tolerance(&self) -> Fix128 {
        self.newton_tolerance
    }

    /// Number of equal fractions the prescribed displacement is applied in.
    #[must_use]
    pub const fn increments(&self) -> u32 {
        self.increments
    }

    /// Polar iteration budget per element per Newton step.
    #[must_use]
    pub const fn polar_iterations(&self) -> u32 {
        self.polar_iterations
    }
}

/// What [`solve_corotational`] produced.
#[derive(Debug, Clone, PartialEq)]
pub struct CorotationalSolution {
    /// Displacements and co-rotational stress.
    ///
    /// [`FemSolution::iterations`] counts the conjugate gradient iterations
    /// summed over every Newton step, and the two residual figures are those of
    /// the last linear solve.
    pub field: FemSolution,
    /// Newton steps performed, summed over every increment.
    pub newton_iterations: u32,
    /// Increments the prescribed displacement was applied in — what the
    /// configuration asked for, reported so a caller reading the field does not
    /// have to keep the configuration alive.
    pub increments: u32,
}

/// The rows of a matrix, so `rows[a][b]` is the entry in row `a`, column `b`.
#[inline]
fn rows_of(m: Mat3Fix) -> [[Fix128; 3]; 3] {
    [
        [m.col0.x, m.col1.x, m.col2.x],
        [m.col0.y, m.col1.y, m.col2.y],
        [m.col0.z, m.col1.z, m.col2.z],
    ]
}

/// `M v` for a 3-vector held as an array.
#[inline]
fn mul3(m: Mat3Fix, v: [Fix128; 3]) -> [Fix128; 3] {
    let out = m.mul_vec(Vec3Fix::new(v[0], v[1], v[2]));
    [out.x, out.y, out.z]
}

/// `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ` for one element.
///
/// `Σᵢ Xᵢ ⊗ ∇Nᵢ = I` on a P1 tetrahedron — the shape functions reproduce a
/// linear field exactly — so the identity is the whole reference contribution
/// and the displacement supplies the rest. `F` is therefore constant over the
/// element, like the strain.
fn deformation_gradient(element: &Element, u: &[[Fix128; 3]; 4]) -> Mat3Fix {
    // `cols[b][a]` is the entry in row `a`, column `b` — the layout `Mat3Fix`
    // stores.
    let mut cols = [[Fix128::ZERO; 3]; 3];
    for (g, node_u) in element.grad.iter().zip(u.iter()) {
        for (b, gb) in g.iter().enumerate() {
            for (a, ua) in node_u.iter().enumerate() {
                cols[b][a] = cols[b][a] + *ua * *gb;
            }
        }
    }
    for (k, col) in cols.iter_mut().enumerate() {
        col[k] = col[k] + Fix128::ONE;
    }
    Mat3Fix::from_cols(
        Vec3Fix::new(cols[0][0], cols[0][1], cols[0][2]),
        Vec3Fix::new(cols[1][0], cols[1][1], cols[1][2]),
        Vec3Fix::new(cols[2][0], cols[2][1], cols[2][2]),
    )
}

/// `σ = R σ̃ Rᵀ` — carry a stress from the element's rotated frame to the
/// global one.
fn rotate_stress(r: Mat3Fix, s: StressTensor) -> StressTensor {
    let m = Mat3Fix::from_cols(
        Vec3Fix::new(s.xx, s.xy, s.zx),
        Vec3Fix::new(s.xy, s.yy, s.yz),
        Vec3Fix::new(s.zx, s.yz, s.zz),
    );
    let rotated = r.mul_mat(m).mul_mat(r.transpose());
    StressTensor {
        xx: rotated.col0.x,
        yy: rotated.col1.y,
        zz: rotated.col2.z,
        xy: rotated.col1.x,
        yz: rotated.col2.y,
        zx: rotated.col2.x,
    }
}

/// `σ̃ = D : sym(Rᵀ F − I)` — the stress in the element's own rotated frame.
///
/// `Rᵀ F − I` is the gradient of the local displacement `Rᵀx − X`, and the
/// element force `V·Bᵀσ̃` depends on that gradient alone: `B` annihilates a
/// constant, so the local displacement itself never has to be formed.
///
/// ⚠️ **Not forming it is worth real accuracy.** `Rᵀx − X` is a difference of
/// two positions, which on a 4 mm cube turned by 90° are 5 mm apart — so the
/// nodal vectors are millimetre-sized, their outer products with the shape
/// function gradients are millimetre-sized, and the sum of the four is the
/// strain, which for a rigid motion is zero. Every one of those millimetre-sized
/// intermediates carries its own rounding into a quantity that is supposed to
/// vanish, and the stiffness then multiplies what is left by `λ + 2μ`, which for
/// the PLA in the oracles is 4,400. `Rᵀ F` is a product of two matrices that are
/// already of order one, so the same cancellation costs less: measured, it
/// halved the spread the increment independence oracle sees.
fn corotational_local_stress(
    rotation_transpose: Mat3Fix,
    gradient: Mat3Fix,
    lambda: Fix128,
    mu: Fix128,
) -> StressTensor {
    let g = rows_of(rotation_transpose.mul_mat(gradient));
    let exx = g[0][0] - Fix128::ONE;
    let eyy = g[1][1] - Fix128::ONE;
    let ezz = g[2][2] - Fix128::ONE;
    let gxy = g[0][1] + g[1][0];
    let gyz = g[1][2] + g[2][1];
    let gzx = g[0][2] + g[2][0];
    let trace = exx + eyy + ezz;
    let two_mu = mu + mu;
    StressTensor {
        xx: lambda * trace + two_mu * exx,
        yy: lambda * trace + two_mu * eyy,
        zz: lambda * trace + two_mu * ezz,
        xy: mu * gxy,
        yz: mu * gyz,
        zx: mu * gzx,
    }
}

/// `σ` and `P = J σ F⁻ᵀ` for one element under a hyperelastic law, or `None`
/// when `F` has no positive determinant or no inverse.
///
/// The Cauchy stress is what the caller reports and what an oracle compares
/// against; the first Piola-Kirchhoff stress is what the nodal force integral
/// needs, because the shape function gradients [`Element`] carries are gradients
/// in the **reference** configuration. `∫ P : ∇₀N dV₀` is the internal force of
/// a total-Lagrangian element and is exact at any stretch, where the
/// co-rotational `R·(V Bᵀσ̃)` is the same integral with `P` approximated by
/// `R σ̃` — right for large rotation, linear in stretch.
fn hyperelastic_stress(
    model: &HyperelasticModel,
    bulk_modulus: Fix128,
    gradient: Mat3Fix,
) -> Option<(StressTensor, Mat3Fix)> {
    let sigma = crate::hyperelastic::cauchy_stress(model, bulk_modulus, gradient)?;
    let inverse_transpose = gradient.inverse()?.transpose();
    let piola = sigma
        .mul_mat(inverse_transpose)
        .scale(gradient.determinant());
    let cauchy = StressTensor {
        xx: sigma.col0.x,
        yy: sigma.col1.y,
        zz: sigma.col2.z,
        xy: sigma.col1.x,
        yz: sigma.col2.y,
        zx: sigma.col2.x,
    };
    Some((cauchy, piola))
}

/// `V₀·P ∇₀Nᵢ` — the nodal forces an element carries for a first Piola-Kirchhoff
/// stress (N). `P` is not symmetric, which is why this cannot go through
/// [`element_force_from_stress`].
fn element_force_from_piola(element: &Element, p: Mat3Fix) -> [[Fix128; 3]; 4] {
    let rows = rows_of(p);
    let mut force = [[Fix128::ZERO; 3]; 4];
    for (slot, g) in force.iter_mut().zip(element.grad.iter()) {
        for (axis, out) in slot.iter_mut().enumerate() {
            let row = rows[axis];
            *out = element.volume * (row[0] * g[0] + row[1] * g[1] + row[2] * g[2]);
        }
    }
    force
}

/// `V·Bᵀσ` — the nodal forces an element carries for a given stress (N).
fn element_force_from_stress(element: &Element, s: StressTensor) -> [[Fix128; 3]; 4] {
    let mut force = [[Fix128::ZERO; 3]; 4];
    for (slot, g) in force.iter_mut().zip(element.grad.iter()) {
        *slot = [
            element.volume * (g[0] * s.xx + g[1] * s.xy + g[2] * s.zx),
            element.volume * (g[1] * s.yy + g[0] * s.xy + g[2] * s.yz),
            element.volume * (g[2] * s.zz + g[1] * s.yz + g[0] * s.zx),
        ];
    }
    force
}

/// `out = Σₑ R Kₑ⁰ Rᵀ v`, the co-rotational tangent applied to `v`.
fn apply_rotated_stiffness(
    elements: &[Element],
    rotations: &[Mat3Fix],
    v: &[Fix128],
    lambda: Fix128,
    mu: Fix128,
    out: &mut [Fix128],
) {
    out.fill(Fix128::ZERO);
    for (element, rotation) in elements.iter().zip(rotations.iter()) {
        let transpose = rotation.transpose();
        let gathered = gather(element, v);
        let mut local = [[Fix128::ZERO; 3]; 4];
        for (slot, node_v) in local.iter_mut().zip(gathered.iter()) {
            *slot = mul3(transpose, *node_v);
        }
        let force = element_force_local(element, &local, lambda, mu);
        for (f, &node) in force.iter().zip(element.nodes.iter()) {
            let global = mul3(*rotation, *f);
            let base = node * 3;
            out[base] = out[base] + global[0];
            out[base + 1] = out[base + 1] + global[1];
            out[base + 2] = out[base + 2] + global[2];
        }
    }
}

/// The parts of a co-rotational solve every element loop reads.
///
/// A bundle rather than eight parameters, for the reason `clippy` gives: past
/// about seven, a call site stops being readable and a transposed pair of
/// same-typed slices stops being a compile error. `rotations` moves every Newton
/// step, so this is built at the call site rather than held.
struct Assembly<'a> {
    /// Elements in mesh order.
    elements: &'a [Element],
    /// One frame per element, in the same order.
    rotations: &'a [Mat3Fix],
    /// `(λ, μ)` of the linear law, in MPa.
    lame: (Fix128, Fix128),
    /// Which degrees of freedom the solve may move.
    is_free: &'a [bool],
}

/// `out = f_ext − Σₑ R Kₑ⁰ (Rᵀx − X)`, the co-rotational residual.
///
/// ⚠️ The internal force is **not** `(R Kₑ⁰ Rᵀ)·u`. The two differ by
/// `Kₑ⁰ (Rᵀ − I)·X`, which is exactly the term that makes a rigid rotation
/// produce zero force: `Rᵀx − X` vanishes when `x = R·X`, while `Rᵀu` does not.
/// Dropping it turns the solve into a repeated linear solve with a rotated
/// stiffness, which is a different method with a different answer.
///
/// With `material` set the internal force is the total-Lagrangian
/// `Σₑ V₀ P ∇₀N` of that law instead (see [`hyperelastic_stress`]), and the
/// frames are then used for nothing here — the hyperelastic stress is objective
/// on its own, so it needs no frame to be carried into. `Err` when an element
/// has no positive `det F`.
fn corotational_residual(
    assembly: &Assembly<'_>,
    u: &[Fix128],
    f_ext: &[Fix128],
    material: Option<(HyperelasticModel, Fix128)>,
    out: &mut [Fix128],
) -> Result<(), FemError> {
    let &Assembly {
        elements,
        rotations,
        lame: (lambda, mu),
        is_free,
    } = assembly;
    out.copy_from_slice(f_ext);
    for (tet, (element, rotation)) in elements.iter().zip(rotations.iter()).enumerate() {
        let gradient = deformation_gradient(element, &gather(element, u));
        let (force, frame) = match &material {
            Some((model, bulk)) => {
                let (_, piola) = hyperelastic_stress(model, *bulk, gradient).ok_or(
                    FemError::RotationFailed {
                        tet,
                        cause: PolarError::Inverted,
                    },
                )?;
                (element_force_from_piola(element, piola), Mat3Fix::IDENTITY)
            }
            None => (
                element_force_from_stress(
                    element,
                    corotational_local_stress(rotation.transpose(), gradient, lambda, mu),
                ),
                *rotation,
            ),
        };
        for (f, &node) in force.iter().zip(element.nodes.iter()) {
            let global = mul3(frame, *f);
            let base = node * 3;
            out[base] = out[base] - global[0];
            out[base + 1] = out[base + 1] - global[1];
            out[base + 2] = out[base + 2] - global[2];
        }
    }
    for (d, value) in out.iter_mut().enumerate() {
        if !is_free[d] {
            *value = Fix128::ZERO;
        }
    }
    Ok(())
}

/// `Σₑ (f_lin − f_mat)(u)`, scattered to the nodes and zeroed on the prescribed
/// degrees of freedom.
///
/// # Why a step built on the linear operator lands on the material's root
///
/// With the frames fixed the co-rotational linear internal force is affine,
/// `f_lin(u) = A u + b`, which is what lets [`solve_corotational`] solve for the
/// displacement itself rather than a correction to it: the right hand side
/// `r_lin(boundary field)` is exactly `f_ext − A·u_prescribed − b` on the free
/// rows. Adding this term to that right hand side gives
///
/// ```text
/// A·u_free = f_ext − f_lin(bf) + f_lin(u) − f_mat(u)
/// ```
///
/// and substituting `f_lin(u) = f_lin(bf) + A·u_free` collapses it to
/// `f_ext = f_mat(u)`: **a fixed point of this iteration is equilibrium under
/// the material law, exactly**, with the linear operator deciding only how many
/// steps it takes to get there. `A` is not the material tangent, so this is a
/// modified Newton iteration and converges linearly rather than quadratically.
///
/// When no material is set the term is identically zero and nothing about the
/// existing path changes — that is checked by the linear oracles, which are bit
/// for bit unmoved by this function existing.
fn material_correction(
    assembly: &Assembly<'_>,
    u: &[Fix128],
    model: &HyperelasticModel,
    bulk_modulus: Fix128,
    out: &mut [Fix128],
) -> Result<(), FemError> {
    let &Assembly {
        elements,
        rotations,
        lame: (lambda, mu),
        is_free,
    } = assembly;
    for (tet, (element, rotation)) in elements.iter().zip(rotations.iter()).enumerate() {
        let gradient = deformation_gradient(element, &gather(element, u));
        let (_, piola) =
            hyperelastic_stress(model, bulk_modulus, gradient).ok_or(FemError::RotationFailed {
                tet,
                cause: PolarError::Inverted,
            })?;
        let linear = element_force_from_stress(
            element,
            corotational_local_stress(rotation.transpose(), gradient, lambda, mu),
        );
        let material = element_force_from_piola(element, piola);
        for ((l, m), &node) in linear.iter().zip(material.iter()).zip(element.nodes.iter()) {
            let rotated = mul3(*rotation, *l);
            let base = node * 3;
            for axis in 0..3 {
                out[base + axis] = out[base + axis] + rotated[axis] - m[axis];
            }
        }
    }
    for (d, value) in out.iter_mut().enumerate() {
        if !is_free[d] {
            *value = Fix128::ZERO;
        }
    }
    Ok(())
}

/// Diagonal of `Σₑ R Kₑ⁰ Rᵀ`.
///
/// The nodal block of `Kₑ⁰` is `V·BᵢᵀDBᵢ`, which works out to
/// `(λ+2μ)g_m² + μ(g_n² + g_p²)` on the diagonal and `(λ+μ)g_m g_n` off it.
/// Rotating it needs the whole block, not just the diagonal
/// [`stiffness_diagonal`] returns: `diag(R K Rᵀ)_a = Σ_{m,n} R_{am} K_{mn} R_{an}`
/// mixes every entry.
fn rotated_stiffness_diagonal(
    elements: &[Element],
    rotations: &[Mat3Fix],
    lambda: Fix128,
    mu: Fix128,
    ndof: usize,
) -> Vec<Fix128> {
    let mut diag = vec![Fix128::ZERO; ndof];
    let lambda_2mu = lambda + mu + mu;
    let lambda_mu = lambda + mu;
    for (element, rotation) in elements.iter().zip(rotations.iter()) {
        let r = rows_of(*rotation);
        for (g, &node) in element.grad.iter().zip(element.nodes.iter()) {
            let sq = [g[0] * g[0], g[1] * g[1], g[2] * g[2]];
            let mut block = [[Fix128::ZERO; 3]; 3];
            for (m, row) in block.iter_mut().enumerate() {
                for (n, entry) in row.iter_mut().enumerate() {
                    *entry = if m == n {
                        element.volume
                            * (lambda_2mu * sq[m] + mu * (sq[(m + 1) % 3] + sq[(m + 2) % 3]))
                    } else {
                        element.volume * (lambda_mu * g[m] * g[n])
                    };
                }
            }
            let base = node * 3;
            for (a, r_row) in r.iter().enumerate() {
                let mut acc = Fix128::ZERO;
                for (m, block_row) in block.iter().enumerate() {
                    for (n, entry) in block_row.iter().enumerate() {
                        acc = acc + r_row[m] * *entry * r_row[n];
                    }
                }
                diag[base + a] = diag[base + a] + acc;
            }
        }
    }
    diag
}

/// Solve the co-rotational boundary value problem on `mesh`.
///
/// # What this is for
///
/// [`solve`] measures strain as `sym(∇u)`, which reads a **rigid rotation** as
/// strain: rotating a body by `θ` about an axis produces a spurious
/// `σ ≈ 2(λ+μ)(cos θ − 1)`, which at 37° is a fifth of Young's modulus. That is
/// not a small error to be refined away — it is the strain measure being wrong
/// about what happened.
///
/// The co-rotational formulation extracts a rotation `R` per element from the
/// polar decomposition of the deformation gradient (see
/// [`Mat3Fix::polar_rotation`]) and measures strain in that rotated frame:
/// `ε = sym(Rᵀ F − I)`. A rigid motion gives `Rᵀ F = I` and therefore no stress
/// at all, exactly. Strain within the rotated frame is still small-strain, so
/// this buys large *rotation*, not large *stretch*.
///
/// # Method
///
/// Newton on the residual `r = f_ext − Σₑ R Kₑ⁰ (Rᵀx − X)`, with `R` recomputed
/// from the current displacement at **every** Newton step, and the tangent
/// `Σₑ R Kₑ⁰ Rᵀ` — the material part, without the derivative of `R` itself.
/// Dropping that term costs iterations, not accuracy: it changes the path to
/// the root, and the root is where the residual vanishes.
///
/// ⚠️ The internal force is `R Kₑ⁰ (Rᵀx − X)` and **not** `(R Kₑ⁰ Rᵀ)·u`. See
/// `corotational_residual` (private) for what the difference is and why it is the
/// whole method.
///
/// # The stopping rule
///
/// **The last increment stops only when the element frames settle** — every
/// frame within `FRAME_SETTLED` (private) of the previous one. Earlier increments
/// stop on that or on the largest out-of-balance nodal force dropping to
/// `newton_tolerance · max|r_ref|`, with `r_ref` the residual of the **full**
/// prescribed displacement read with no rotation at all: the load the problem
/// poses, computed once before the first increment.
///
/// Every part of that is deliberate.
///
/// The answer is the last increment's output, and for a fixed set of frames it
/// is a function of those frames and the boundary data alone. A tolerance that
/// decides where the last increment stops is therefore *in* the answer, and it
/// was: with the residual deciding, a one-increment run met the threshold after
/// a single frame update while a nine-increment run had had several, and the two
/// stopped on different frames. Earlier increments are a continuation path whose
/// only product is the state the next one opens at, so a tolerance is the right
/// instrument there — see the stopping rule in the body for what running them to
/// settled frames costs and why.
///
/// The threshold is built from `r_ref` and not from "the residual this
/// increment started at", because the latter makes the answer depend on the
/// increment count: a nine-increment run starts each solve from a ninth of the
/// boundary motion, so its threshold is about a ninth of a one-increment run's.
/// Incremental application is a path to the answer, not part of it, so the
/// threshold must not know how many increments there are.
///
/// The residual is measured as the largest entry, not as `√(rᵀr)`; see
/// `max_abs` (private) for the measurement that forced that.
///
/// `newton_tolerance` is also asked once at the very end, on the answer, so that
/// it can say whether what the frames settled on satisfies equilibrium without
/// being able to say *where* they settle.
///
/// # ⚠️ What still depends on the path
///
/// The displacement is a function of the frames alone, so the question is
/// whether two runs stop on the same frames, and **they do not stop on exactly
/// the same frames**. Measured on the 4 mm cube of
/// `tests/analytic_corotational.rs`, as the worst nodal difference in units in
/// the last place over every pair of the increment counts 1, 2, 3, 4, 6, 9, 12:
///
/// | boundary rotation | residual deciding | frames deciding |
/// |---|---|---|
/// | 1 mrad | 2.0e6 | **2** |
/// | 11.5° | 8.6e6 | **3** |
/// | 36.87° | 3.5e10 | **4** |
/// | 68.8° | 8.4e10 | **5** |
/// | 90° | 1.6e10 | **5** |
/// | 126° | 3.3e10 | **4** |
///
/// One unit in the last place is 5.4e-20 mm, so the right-hand column is a
/// spread of about 2.7e-19 mm. It is not zero and it is not reachable: the frame
/// iteration converges to a spread of a few units in the last place and then
/// wanders inside it indefinitely, so an exact fixed point of the frame map does
/// not exist to be found. `the_answer_does_not_depend_on_the_increment_count` in
/// that file asks for bit-identical displacements and is ignored for exactly
/// that reason; its companion
/// `the_increment_spread_stays_at_the_arithmetic_floor` is not ignored and pins
/// the right-hand column.
///
/// # Determinism
///
/// Every operation is [`Fix128`] arithmetic, including the polar iteration,
/// which is Higham's `R ← ½(R + R⁻ᵀ)` and reaches for no transcendental. The
/// element loop order is the mesh's tet order, the increment and Newton counts
/// are bounded by the configuration, and the linear solve is the same
/// conjugate gradient [`solve`] uses.
///
/// # Errors
///
/// As [`solve`], plus [`FemError::RotationFailed`] when an element's
/// deformation gradient has no rotation factor, and [`FemError::NotConverged`]
/// when the Newton budget runs out with the residual still above tolerance.
pub fn solve_corotational(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &CorotationalConfig,
) -> Result<CorotationalSolution, FemError> {
    let vertex_count = mesh.vertices.len();
    if vertex_count == 0 || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    for &(vertex, _, _) in boundary.prescribed.iter().chain(boundary.loads.iter()) {
        if vertex as usize >= vertex_count {
            return Err(FemError::VertexOutOfRange {
                vertex,
                vertex_count,
            });
        }
    }

    let elements = build_elements(mesh)?;
    let (lambda, mu) = material.lame();
    // `K = λ + 2μ/3` — the bulk modulus of the same isotropic solid, which is
    // what fixes the pressure an incompressible strain energy leaves free. The
    // deviatoric response comes from the model in the configuration, so the two
    // are free to describe different solids; see
    // [`CorotationalConfig::with_hyperelastic`].
    let law = config
        .material
        .map(|model| (model, lambda + (mu + mu) / Fix128::from_int(3)));
    let ndof = vertex_count * 3;

    let mut prescribed_value = vec![Fix128::ZERO; ndof];
    let mut is_free = vec![true; ndof];
    for &(vertex, axis, value) in &boundary.prescribed {
        let d = vertex as usize * 3 + axis.index();
        is_free[d] = false;
        prescribed_value[d] = value;
    }
    if is_free.iter().filter(|f| !**f).count() < 6 {
        return Err(FemError::UnderConstrained);
    }

    let mut f_ext = vec![Fix128::ZERO; ndof];
    for &(vertex, axis, force) in &boundary.loads {
        let d = vertex as usize * 3 + axis.index();
        if is_free[d] {
            f_ext[d] = f_ext[d] + force;
        }
    }

    // The reference residual: the full prescribed displacement read by the
    // small-strain operator, which is what `solve` would put on its right hand
    // side. It does not depend on the increment count or the Newton budget, so
    // neither does the threshold built from it.
    let mut scratch = vec![Fix128::ZERO; ndof];
    apply_stiffness(&elements, &prescribed_value, lambda, mu, &mut scratch);
    let mut reference = f_ext.clone();
    for (d, value) in reference.iter_mut().enumerate() {
        if is_free[d] {
            *value = *value - scratch[d];
        } else {
            *value = Fix128::ZERO;
        }
    }

    let newton_target = max_abs(&reference) * config.newton_tolerance;

    // The prescribed field of the current increment: the boundary data on the
    // constrained degrees of freedom, zero everywhere else.
    let mut boundary_field = vec![Fix128::ZERO; ndof];
    let mut u = vec![Fix128::ZERO; ndof];
    let mut rotations = vec![Mat3Fix::IDENTITY; elements.len()];
    let mut residual = vec![Fix128::ZERO; ndof];
    let mut newton_iterations = 0u32;
    let mut cg_iterations = 0u32;
    let mut relative_residual = Fix128::ZERO;
    let mut effective_relative_tolerance = Fix128::ZERO;

    for increment in 1..=config.increments {
        let final_increment = increment == config.increments;
        // Predict the whole field, not only the boundary.
        //
        // The prediction is only ever used to pick the *frames* for the first
        // solve of the increment, but that is enough to matter: moving the
        // prescribed nodes and leaving the interior where it was can flatten the
        // deformation gradient of an element straddling the boundary outright —
        // measured `det F = 0` exactly, on a 90° rotation applied in four
        // increments. Carrying the previous converged field forward by the ratio
        // of the two load factors keeps the interior with the boundary, which is
        // what the increments were for.
        if increment > 1 {
            let ratio =
                Fix128::from_int(i64::from(increment)) / Fix128::from_int(i64::from(increment - 1));
            for (d, value) in u.iter_mut().enumerate() {
                if is_free[d] {
                    *value = *value * ratio;
                }
            }
        }
        // The last increment carries the prescribed values through unscaled, so
        // the boundary data the answer is built on is bit for bit what the
        // caller asked for however many increments there were.
        boundary_field.fill(Fix128::ZERO);
        if increment == config.increments {
            for (d, value) in prescribed_value.iter().enumerate() {
                if !is_free[d] {
                    boundary_field[d] = *value;
                }
            }
        } else {
            let scale = Fix128::from_int(i64::from(increment))
                / Fix128::from_int(i64::from(config.increments));
            for (d, value) in prescribed_value.iter().enumerate() {
                if !is_free[d] {
                    boundary_field[d] = *value * scale;
                }
            }
        }
        for (d, value) in boundary_field.iter().enumerate() {
            if !is_free[d] {
                u[d] = *value;
            }
        }

        let mut step = 0u32;
        loop {
            // The frame is recomputed from the current displacement at every
            // step. The one exception is the state a new increment opens at:
            // that is a *prediction*, not a candidate solution, and at increment
            // 1 it is the undeformed interior with the boundary already moved,
            // which can have no polar factor at all. Keeping the frame that is
            // already in hand — the identity at the start, the previous
            // increment's otherwise — makes the first solve of such an increment
            // a small-strain solve, which is the right thing to do from a guess
            // that carries no rotation information yet. Once a solve has been
            // done the state is a candidate solution and a refusal is reported.
            let mut frames = Vec::with_capacity(elements.len());
            let mut refused = None;
            for (t, element) in elements.iter().enumerate() {
                match deformation_gradient(element, &gather(element, &u))
                    .polar_rotation(POLAR_DET_FLOOR, config.polar_iterations)
                {
                    Ok(r) => frames.push(r),
                    Err(cause) => {
                        refused = Some(FemError::RotationFailed { tet: t, cause });
                        break;
                    }
                }
            }
            match refused {
                None => {
                    // ⚠️ **The frame rule is the linear law's, and only the
                    // linear law's.** It is sound because the step below is
                    // *exact* for fixed frames, so reproducing the frames
                    // reproduces the displacement. A material law makes the step
                    // inexact — the frames can repeat while the displacement is
                    // still moving toward the root — so with one set the
                    // residual decides on every increment.
                    if law.is_none() && step > 0 && frames_settled(&frames, &rotations) {
                        // ⚠️ **This is the stopping rule, and it is not a
                        // tolerance.** For a fixed set of frames the residual is
                        // *linear* in the displacement, so the solve below lands
                        // on the exact minimiser in one go and `u` is a function
                        // of the frames and the boundary data alone. When the
                        // frames come back bit for bit what they were, the next
                        // solve would reproduce the same displacement, and the
                        // one after that the same frames: the iteration has a
                        // fixed point and this is it.
                        break;
                    }
                    rotations.copy_from_slice(&frames);
                }
                Some(error) if step > 0 => return Err(error),
                Some(_) => {}
            }
            // ⚠️ **The residual is not consulted on the last increment.**
            //
            // The answer is the last increment's output, and it is a function of
            // that increment's frames and boundary data alone. Letting a
            // *tolerance* decide where the last increment stops therefore puts
            // the tolerance into the answer: a one-increment run meets the
            // threshold after a single frame update where a nine-increment run
            // has had several, and the two stop on different frames. Measured on
            // the 4 mm cube of `tests/analytic_corotational.rs` under a 36.87°
            // boundary rotation, the six pairs of one, two, four and nine
            // increments differed by 1.3e5 to 3.5e10 units in the last place with
            // the residual deciding, and by 3 to 4 with only the frames deciding.
            //
            // Earlier increments keep the residual rule, and that is not a
            // half-measure: they are a continuation path whose only product is
            // the state the next increment opens at. Running them to settled
            // frames buys nothing and costs a great deal, because the states they
            // visit are not near-rigid. Applying a 90° rotation in four
            // increments prescribes `½(R − I)x` at the halfway point, whose
            // deformation gradient `½(I + R)` is a 45° rotation carrying a 29%
            // compression — far outside the small strain the co-rotational model
            // is linear in, and the frame iteration crawls there: the same cube
            // took 1165 steps on that one increment against 3 on the last one.
            let residual_decides = !final_increment || law.is_some();
            // Step 0 is the state the increment *opened* at — a prediction, and
            // with a material law possibly a degenerate one, which is why the
            // frame loop above tolerates a refusal there. Nothing reads the
            // residual at step 0 (the break below needs `step > 0` and the
            // budget is at least one), so it is not formed.
            if (residual_decides && step > 0) || step >= config.newton_iterations {
                corotational_residual(
                    &Assembly {
                        elements: &elements,
                        rotations: &rotations,
                        lame: (lambda, mu),
                        is_free: &is_free,
                    },
                    &u,
                    &f_ext,
                    law,
                    &mut residual,
                )?;
                let residual_reach = max_abs(&residual);
                if residual_decides && step > 0 && residual_reach <= newton_target {
                    break;
                }
                if step >= config.newton_iterations {
                    return Err(FemError::NotConverged {
                        iterations: step,
                        relative_residual: relative(residual_reach, newton_target),
                    });
                }
            }

            // Solve for the displacement itself, **not** for a correction to it.
            //
            // `r(u) = f_ext − Σₑ R Kₑ⁰ (Rᵀ(X+u) − X)` is affine in `u` once the
            // frames are fixed, so splitting `u` into the prescribed field and
            // the free part gives `K_R·u_free = r(boundary field)` exactly, and
            // one linear solve is the whole Newton step rather than an
            // approximation to it.
            //
            // ⚠️ Accumulating corrections instead — `u ← u + Δu` — would make the
            // displacement carry the rounding of every state the path passed
            // through. Recomputing it makes `u` a function of the frames and the
            // boundary data and nothing else, which is what
            // `solve_corotational`'s note on the path dependence that is left
            // is measured against.
            let assembly = Assembly {
                elements: &elements,
                rotations: &rotations,
                lame: (lambda, mu),
                is_free: &is_free,
            };
            corotational_residual(&assembly, &boundary_field, &f_ext, None, &mut residual)?;
            // `f_lin(u) − f_mat(u)`, which turns the exact step of the linear law
            // into a modified Newton step whose fixed point is equilibrium under
            // the material law. See `material_correction` (private).
            if let Some((model, bulk)) = law {
                if step > 0 {
                    material_correction(&assembly, &u, &model, bulk, &mut residual)?;
                }
            }
            let diag = rotated_stiffness_diagonal(&elements, &rotations, lambda, mu, ndof);
            let precond = build_preconditioner(&diag, &is_free, &config.linear)?;
            let cg =
                conjugate_gradient(&residual, &is_free, &precond, &config.linear, |p, out| {
                    apply_rotated_stiffness(&elements, &rotations, p, lambda, mu, out);
                })?;
            cg_iterations = cg_iterations.saturating_add(cg.iterations);
            relative_residual = relative(cg.residual_norm, cg.b_norm);
            effective_relative_tolerance = relative(cg.target, cg.b_norm);
            for (d, value) in u.iter_mut().enumerate() {
                *value = if is_free[d] {
                    cg.x[d]
                } else {
                    boundary_field[d]
                };
            }
            step += 1;
            newton_iterations = newton_iterations.saturating_add(1);
        }
    }

    // The frames stopped moving, which says the iteration reached its fixed
    // point; it does not by itself say the fixed point satisfies equilibrium.
    // `newton_tolerance` is that second question, and it is asked once, on the
    // answer, so that it cannot decide *where* the iteration stops — only
    // whether what it stopped on is acceptable.
    corotational_residual(
        &Assembly {
            elements: &elements,
            rotations: &rotations,
            lame: (lambda, mu),
            is_free: &is_free,
        },
        &u,
        &f_ext,
        law,
        &mut residual,
    )?;
    let final_residual = max_abs(&residual);
    if final_residual > newton_target {
        return Err(FemError::NotConverged {
            iterations: newton_iterations,
            relative_residual: relative(final_residual, newton_target),
        });
    }

    let displacements = (0..vertex_count)
        .map(|v| [u[v * 3], u[v * 3 + 1], u[v * 3 + 2]])
        .collect();
    let mut element_stress = Vec::with_capacity(elements.len());
    for (tet, (element, rotation)) in elements.iter().zip(rotations.iter()).enumerate() {
        let gradient = deformation_gradient(element, &gather(element, &u));
        element_stress.push(match law {
            // Cauchy stress, already in the global frame: a hyperelastic law is
            // objective, so nothing is rotated into place afterwards.
            Some((model, bulk)) => {
                hyperelastic_stress(&model, bulk, gradient)
                    .ok_or(FemError::RotationFailed {
                        tet,
                        cause: PolarError::Inverted,
                    })?
                    .0
            }
            None => rotate_stress(
                *rotation,
                corotational_local_stress(rotation.transpose(), gradient, lambda, mu),
            ),
        });
    }

    Ok(CorotationalSolution {
        field: FemSolution {
            displacements,
            element_stress,
            iterations: cg_iterations,
            relative_residual,
            effective_relative_tolerance,
        },
        newton_iterations,
        increments: config.increments,
    })
}

// ============================================================================
// Tests
// ============================================================================

/// Oracles for the stress a hyperelastic element reports and for the first
/// Piola-Kirchhoff stress it integrates.
///
/// # Why these live here and not in `tests/`
///
/// [`hyperelastic_stress`] is private and `P` is not on the public surface:
/// [`CorotationalSolution`] carries displacements and the Cauchy stress, and `P`
/// only ever appears inside a nodal force sum. So an integration test can see `P`
/// *only* through where a solve lands, and a solve cannot see every error in `P`:
/// it checks equilibrium with the same force it assembled, so a wrong `P` that is
/// still a gradient of something converges to a different field and reports it as
/// equilibrium. Measured 2026-10-01: dropping the `F⁻ᵀ` from `P = J σ F⁻ᵀ` left
/// `tests/analytic_corotational.rs` at `12 passed / 1 ignored` and
/// `cargo test --lib hyperelastic` at `21 passed` — no oracle in the crate moved,
/// including `the_material_law_moves_the_answer_and_the_answer_is_equilibrium`,
/// which still returned `Ok` on a field assembled from the wrong `P`.
///
/// `src/hyperelastic.rs` already carries the same pattern for the same reason —
/// `cauchy_stress_vanishes_in_the_reference_state` covers the `−p_ref` term,
/// which a solve cannot see either because it shifts every element's stress by
/// one tensor and a uniform stress puts no force on an interior node.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::hyperelastic::cauchy_stress;

    /// A matrix from its **rows**, each entry a rational `(numerator,
    /// denominator)`, which is how the derivations in this module are written.
    ///
    /// ⚠️ **Every denominator used below is a power of two.** `Fix128` carries 64
    /// fractional bits, so such an entry is exact, and so is any product of two of
    /// them that needs fewer than 64 fractional bits — the chains here need at
    /// most 43. That is what makes the identities in this module assertable with
    /// `assert_eq!`: no rounding happens at all, so none of them depends on how
    /// `Fix128` rounds or on whether its rounding is symmetric about zero.
    fn from_rows(r: [[(i64, i64); 3]; 3]) -> Mat3Fix {
        let e = |i: usize, j: usize| {
            let (n, d) = r[i][j];
            Fix128::from_ratio(n, d)
        };
        Mat3Fix::from_cols(
            Vec3Fix::new(e(0, 0), e(1, 0), e(2, 0)),
            Vec3Fix::new(e(0, 1), e(1, 1), e(2, 1)),
            Vec3Fix::new(e(0, 2), e(1, 2), e(2, 2)),
        )
    }

    /// `Q`, a quarter turn about `z`.
    ///
    /// ⚠️ **Every entry is `0` or `±1`**, so `Q·M` and `M·Qᵀ` only move entries
    /// between slots and negate some of them. A general angle would put `cos` and
    /// `sin` into the identity and turn an exact statement into a tolerance; a
    /// quarter turn keeps it exact while still being the strongest case, because
    /// it moves the `x` and `y` axes onto each other completely.
    fn quarter_turn_z() -> Mat3Fix {
        from_rows([
            [(0, 1), (-1, 1), (0, 1)],
            [(1, 1), (0, 1), (0, 1)],
            [(0, 1), (0, 1), (1, 1)],
        ])
    }

    /// Deformation gradients the objectivity oracle sweeps, all dyadic.
    ///
    /// `det` is `1`, `2` or `1/2` by construction — also powers of two, because
    /// [`crate::hyperelastic::cauchy_stress`] divides by `J` and
    /// [`Mat3Fix::inverse`] by `det`, and only a power of two keeps those exact.
    ///
    /// The last three are products of a diagonal stretch with a unit-triangular
    /// shear, so they are unimodular, full (no zero entry outside the first), and
    /// not symmetric — a symmetric `F` would make `σ F⁻ᵀ` and `F⁻ᵀ σ` agree too
    /// often for the sweep to separate them.
    fn gradients() -> [(&'static str, Mat3Fix); 5] {
        [
            // det = 1. The case the closed form below is written out for.
            (
                "diag(2, 1/2, 1) with xy shear, det 1",
                from_rows([
                    [(2, 1), (1, 2), (0, 1)],
                    [(0, 1), (1, 2), (0, 1)],
                    [(0, 1), (0, 1), (1, 1)],
                ]),
            ),
            // The same with the z row doubled: det = 2, so the bulk term is live.
            (
                "the same with z doubled, det 2",
                from_rows([
                    [(2, 1), (1, 2), (0, 1)],
                    [(0, 1), (1, 2), (0, 1)],
                    [(0, 1), (0, 1), (2, 1)],
                ]),
            ),
            // det = 1/2, the other side of the reference volume.
            (
                "diag(1, 1/2, 1) with xy shear, det 1/2",
                from_rows([
                    [(1, 1), (1, 2), (0, 1)],
                    [(0, 1), (1, 2), (0, 1)],
                    [(0, 1), (0, 1), (1, 1)],
                ]),
            ),
            // diag(2, 1/2, 1) · upper unit-triangular shear, det = 1.
            (
                "upper shear after stretch, det 1",
                from_rows([
                    [(2, 1), (1, 1), (1, 2)],
                    [(0, 1), (1, 2), (1, 16)],
                    [(0, 1), (0, 1), (1, 1)],
                ]),
            ),
            // The product of that with a lower unit-triangular shear after a
            // stretch: full, unimodular, far from symmetric.
            (
                "upper shear after stretch times lower shear after stretch, det 1",
                from_rows([
                    [(37, 8), (5, 8), (1, 2)],
                    [(17, 64), (17, 64), (1, 16)],
                    [(1, 4), (1, 4), (1, 1)],
                ]),
            ),
        ]
    }

    /// Material laws the sweep uses, with **dyadic** constants for the reason
    /// given on [`from_rows`].
    ///
    /// [`HyperelasticModel::tpu_soft`] is included as it ships (`μ = 3` exactly);
    /// `silicone_soft` and `natural_rubber` are not, because their constants are
    /// decimal approximations (`0.1`, `−0.017`, `0.00062`) and a product with one
    /// of those rounds. The exactness being protected is a property of the
    /// arithmetic and not of any law, so standing in dyadic constants for decimal
    /// ones loses no coverage: `MooneyRivlin` below exercises the same `W₂ ≠ 0`
    /// branch as `silicone_soft`, and `Yeoh` the same deformation-dependent `W₁`
    /// as `natural_rubber`.
    fn models() -> [(&'static str, HyperelasticModel); 4] {
        [
            (
                "Neo-Hookean μ = 1000",
                HyperelasticModel::NeoHookean {
                    mu_mpa: Fix128::from_int(1000),
                },
            ),
            (
                "tpu_soft (Neo-Hookean μ = 3)",
                HyperelasticModel::tpu_soft(),
            ),
            (
                "Mooney-Rivlin C₁ = 1/2, C₂ = 1/4 (the W₂ ≠ 0 branch)",
                HyperelasticModel::MooneyRivlin {
                    c1_mpa: Fix128::from_ratio(1, 2),
                    c2_mpa: Fix128::from_ratio(1, 4),
                },
            ),
            (
                "Yeoh C₁ = 1/2, C₂ = −1/16, C₃ = 1/64 (W₁ depends on I₁)",
                HyperelasticModel::Yeoh {
                    c1_mpa: Fix128::from_ratio(1, 2),
                    c2_mpa: Fix128::from_ratio(-1, 16),
                    c3_mpa: Fix128::from_ratio(1, 64),
                },
            ),
        ]
    }

    /// **The oracle for the `F⁻ᵀ` in `P = J σ F⁻ᵀ`.**
    ///
    /// Objectivity, also called frame indifference: superposing a rigid rotation
    /// `Q` on the deformed configuration — the reference mesh untouched — sends
    /// `F → Q F`, and then
    ///
    /// ```text
    /// σ(Q F) = Q σ(F) Qᵀ                    the law is objective
    /// P(Q F) = J σ(QF) (QF)⁻ᵀ
    ///        = J (Q σ Qᵀ) (Q F⁻ᵀ)          since (QF)⁻ᵀ = Q F⁻ᵀ
    ///        = Q (J σ F⁻ᵀ) = Q P(F)         Qᵀ Q = I
    /// ```
    ///
    /// `P` rotates with **one** `Q` where `σ` rotates with two, and that is the
    /// whole content of the `F⁻ᵀ`: drop it and `P̃ = J σ`, which satisfies
    /// `P̃(QF) = Q P̃(F) Qᵀ` instead. The two agree only when `σ (Qᵀ − I) = 0`, so
    /// the relation separates them for any `σ` that is not invariant under `Q`.
    /// The last assertion in this test is that separation, measured rather than
    /// argued: it fails if the sweep ever reaches a case where the dropped form
    /// would pass, which would make the oracle vacuous there.
    ///
    /// ⚠️ **Why this exact relation cannot live in `tests/`.** See the module
    /// comment: `P` is observable from outside only through where a solve lands,
    /// and the solve checks its own assembled force for equilibrium. Dropping the
    /// `F⁻ᵀ` moved no oracle in the crate before this one.
    /// `the_solve_is_equivariant_under_a_superposed_quarter_turn` in
    /// `tests/analytic_corotational.rs` is the companion that states the same
    /// property at the level of the solve — it does red on the same mutation, but
    /// as a `RotationFailed`, and it is a bound rather than an equality because
    /// the solve stops on a tolerance.
    #[test]
    fn the_first_piola_kirchhoff_stress_is_objective_under_a_superposed_rotation() {
        let q = quarter_turn_z();
        let mut separations = 0_u32;
        for (model_name, model) in models() {
            for bulk in [Fix128::ZERO, Fix128::from_int(1), Fix128::from_int(2048)] {
                for (gradient_name, f) in gradients() {
                    let what =
                        || format!("{model_name}, K = {}, F = {gradient_name}", bulk.to_f64());
                    let rotated = q.mul_mat(f);
                    let det = f.determinant();
                    assert_eq!(
                        det,
                        rotated.determinant(),
                        "det Q = 1, so det QF = det F: {}",
                        what()
                    );

                    let sigma = cauchy_stress(&model, bulk, f).expect("det F > 0");
                    let sigma_rotated = cauchy_stress(&model, bulk, rotated).expect("det QF > 0");
                    assert_eq!(
                        sigma_rotated,
                        q.mul_mat(sigma).mul_mat(q.transpose()),
                        "the Cauchy stress must be objective: {}",
                        what()
                    );

                    let (_, piola) = hyperelastic_stress(&model, bulk, f).expect("det F > 0");
                    let (_, piola_rotated) =
                        hyperelastic_stress(&model, bulk, rotated).expect("det QF > 0");
                    assert_eq!(
                        piola_rotated,
                        q.mul_mat(piola),
                        "P must rotate with one Q, which is what the F⁻ᵀ carries: {}",
                        what()
                    );

                    // Non-vacuity: the dropped-`F⁻ᵀ` form has to fail what was
                    // just asserted. Counted rather than required case by case,
                    // because `σ (Qᵀ − I) = 0` is possible in principle.
                    if sigma_rotated.scale(det) != q.mul_mat(sigma.scale(det)) {
                        separations += 1;
                    }
                }
            }
        }
        assert_eq!(
            separations, 60,
            "every case in the sweep must separate P = J σ F⁻ᵀ from P = J σ; \
             a case that does not is one where this oracle says nothing"
        );
    }

    /// **The closed form of `P` on one gradient**, so that the relation above is
    /// anchored to a value and not only to itself.
    ///
    /// `F = [[2, 1/2, 0], [0, 1/2, 0], [0, 0, 1]]`, `det F = 1`, Neo-Hookean with
    /// `μ = 1000` MPa, so by hand:
    ///
    /// ```text
    /// B      = F Fᵀ = [[17/4, 1/4, 0], [1/4, 1/4, 0], [0, 0, 1]]
    /// W₁     = μ/2, W₂ = 0, p_ref = 2(W₁ + 2W₂) = μ,  J = 1
    /// σ      = (2/J)·W₁·B + (K(J−1) − p_ref)·I = μ(B − I)
    ///        = [[3250, 250, 0], [250, −750, 0], [0, 0, 0]]
    /// F⁻ᵀ    = [[1/2, 0, 0], [−1/2, 2, 0], [0, 0, 1]]
    /// P      = J σ F⁻ᵀ = [[1500, 500, 0], [500, −1500, 0], [0, 0, 0]]
    /// ```
    ///
    /// One check on that last line that is independent of the code: `P Fᵀ` must
    /// come back as `J σ`, symmetric, because that product *is* the Cauchy stress
    /// and angular momentum balance is what makes it symmetric. Together with
    /// `P ≠ J σ` that is what the `F⁻ᵀ` has to produce.
    ///
    /// ⚠️ **`P` happens to be symmetric on this `F`** — measured, and the reason
    /// the asymmetry is asserted on a different gradient below instead. Writing
    /// `P` asymmetric here looked right and is wrong: `F` is upper triangular with
    /// a `z` row that does nothing, so `σ` and `F⁻ᵀ` commute on the `xy` block.
    /// A reader who assumed otherwise would conclude the wrong thing about why
    /// [`element_force_from_piola`] cannot go through
    /// [`element_force_from_stress`].
    ///
    /// ⚠️ Nothing in the crate was called to produce the numbers above; `K` is
    /// swept to show it does not enter at `J = 1`.
    #[test]
    fn the_first_piola_kirchhoff_stress_is_what_the_closed_form_says() {
        let f = from_rows([
            [(2, 1), (1, 2), (0, 1)],
            [(0, 1), (1, 2), (0, 1)],
            [(0, 1), (0, 1), (1, 1)],
        ]);
        let want_sigma = from_rows([
            [(3250, 1), (250, 1), (0, 1)],
            [(250, 1), (-750, 1), (0, 1)],
            [(0, 1), (0, 1), (0, 1)],
        ]);
        let want_piola = from_rows([
            [(1500, 1), (500, 1), (0, 1)],
            [(500, 1), (-1500, 1), (0, 1)],
            [(0, 1), (0, 1), (0, 1)],
        ]);
        let model = HyperelasticModel::NeoHookean {
            mu_mpa: Fix128::from_int(1000),
        };
        assert_eq!(f.determinant(), Fix128::ONE, "the derivation assumes J = 1");

        for bulk in [Fix128::ZERO, Fix128::from_int(1), Fix128::from_int(2048)] {
            let (cauchy, piola) = hyperelastic_stress(&model, bulk, f).expect("det F = 1 > 0");
            assert_eq!(
                cauchy_stress(&model, bulk, f).expect("det F = 1 > 0"),
                want_sigma,
                "σ = μ(B − I) at J = 1, K = {}",
                bulk.to_f64()
            );
            assert_eq!(cauchy.xx, Fix128::from_int(3250));
            assert_eq!(cauchy.yy, Fix128::from_int(-750));
            assert_eq!(cauchy.zz, Fix128::ZERO);
            assert_eq!(cauchy.xy, Fix128::from_int(250));
            assert_eq!(cauchy.yz, Fix128::ZERO);
            assert_eq!(cauchy.zx, Fix128::ZERO);
            assert_eq!(piola, want_piola, "P = J σ F⁻ᵀ at K = {}", bulk.to_f64());
            assert_eq!(
                piola.mul_mat(f.transpose()),
                want_sigma,
                "P Fᵀ = J σ, symmetric, at K = {}",
                bulk.to_f64()
            );
            assert_ne!(
                piola, want_sigma,
                "P must differ from J σ on this F, or the F⁻ᵀ is unobservable here"
            );
        }

        // `P` is a two-point tensor, which is why `element_force_from_piola`
        // exists beside `element_force_from_stress`. The gradient above is too
        // special to show it (see the comment on this test), so it is asserted on
        // the full unimodular one from the sweep.
        let (full_name, full) = gradients()[4];
        let (_, full_piola) = hyperelastic_stress(&model, Fix128::ZERO, full).expect("det F > 0");
        assert_ne!(
            full_piola,
            full_piola.transpose(),
            "P must not be symmetric on {full_name}, or nothing in this crate needs \
             a force integral for a non-symmetric stress"
        );
    }
}
