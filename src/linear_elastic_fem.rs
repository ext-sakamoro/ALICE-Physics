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
//!   no plasticity in `solve`. Past yield the result is the *elastic* stress,
//!   which is what a yield check wants as its input. Plasticity is
//!   `solve_elastoplastic` (small-strain J2, P1 only).
//! - P1 tetrahedra are stiff in bending. A beam resolved by a few elements
//!   through the thickness under-predicts deflection; refine through the
//!   thickness rather than along the span.
//! - The rigid-body-mode check is a necessary condition, not a sufficient one
//!   (see `FemError::UnderConstrained`).
//!
//! Author: Moroya Sakamoto

use crate::coupled_field::{CoupledField, TemperatureRise};
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
    /// undefined, or it is so thin that its gradients (of order `1/h`) times
    /// the material modulus exceed the `Fix128` range, so its stiffness cannot
    /// be formed (AUD-A-S1W3-004).
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
    /// A [`ThermalExpansion`] field does not cover every node of the mesh.
    ///
    /// Refused rather than tolerated because [`CoupledField::sample`] clamps a
    /// point outside its grid onto the nearest boundary node instead of
    /// failing: a mesh poking out of the field would be heated by the extruded
    /// boundary value and the solve would return a plausible, wrong answer. The
    /// element centroids are what the eigenstrain is actually sampled at, and a
    /// centroid is a convex combination of its four nodes, so checking the
    /// nodes is exactly the condition that keeps every sample inside the grid.
    TemperatureFieldDoesNotCoverMesh {
        /// Index into `SdfTetMesh::vertices` of the first node found outside.
        vertex: u32,
    },
    /// A solution handed to [`reactions`] or [`corotational_reactions`] has a
    /// different number of nodes than the mesh.
    ///
    /// Refused rather than indexed, because the two really are independent
    /// arguments: nothing ties a [`FemSolution`] to the mesh it came from, and
    /// reading a shorter displacement array would either panic or — if the
    /// solution is the longer one — quietly report the reactions of a different
    /// body.
    SolutionDoesNotMatchMesh {
        /// Nodes the solution carries.
        nodes: usize,
        /// Number of vertices in the mesh.
        vertex_count: usize,
    },
    /// A solution handed to [`deposit_plastic_heat`] reports a different number
    /// of elements than the mesh has tetrahedra.
    ///
    /// Refused for the same reason as [`Self::SolutionDoesNotMatchMesh`]: the
    /// per-element dissipation is positional, so a length mismatch would
    /// deposit one body's heat at another body's centroids.
    SolutionElementCountDoesNotMatchMesh {
        /// Elements the solution reports.
        elements: usize,
        /// Number of tetrahedra in the mesh.
        tet_count: usize,
    },
    /// A field handed to [`deposit_plastic_heat`] has only one node on an axis.
    ///
    /// ⚠️ Refused rather than deposited into, because such a grid has no
    /// thickness on that axis to divide the heat by.
    /// [`crate::coupled_field::CoupledField`] gives a degenerate axis a stored
    /// cell size of one, so using `V_cell` as a volume would silently make the
    /// ledger "per unit thickness" while the element volumes it is balanced
    /// against are the real three-dimensional ones. The temperatures would come
    /// out scaled by the body's true thickness on that axis — a factor the field
    /// does not carry and the deposit cannot recover.
    DepositGridHasDegenerateAxis {
        /// The axis with a single node.
        axis: Axis,
    },
    /// The thermoplastic sub-iteration did not reach a fixed point.
    ///
    /// Carries the monitor's own verdict, which distinguishes the cases a
    /// caller has to act on differently: `Diverging` means the splitting is
    /// not a contraction and no budget will fix it, `Stagnated` means it was
    /// contracting and stopped improving, `NotConverged` means the budget ran
    /// out while it was still improving, and `ArithmeticWrapped` means the
    /// residual sequence showed the signature of a silent `Fix128` wrap.
    CoupledSubIterationFailed(crate::coupled_iteration::CoupledIterationError),
    /// The conjugate-gradient residual norm could not be formed faithfully:
    /// a residual component at `index` squared past the `Fix128` range (or
    /// its square was dropped to zero above the term floor), so the stopping
    /// test would have read a wrapped number. See
    /// [`crate::coupled_iteration::residual_norm_l2_checked`]. Scale the
    /// problem (units, load) rather than the tolerance.
    ResidualNormUnfaithful {
        /// Index of the residual component whose square broke the norm.
        index: u32,
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
    /// is incompressible and makes the Lamé first parameter diverge. Also
    /// rejected: a `ν` close enough to either open-interval endpoint that
    /// [`Self::lame`]'s `λ = Eν / ((1+ν)(1−2ν))` would already overflow
    /// `Fix128`'s representable range for the given `E` — `Fix128` is a
    /// wrapping (not saturating) 128-bit type, so that division would
    /// otherwise wrap silently to a small or sign-flipped value instead of
    /// erroring (measured: `E = 1e6`, `ν = 0.5 − 2^-58` gives `λ = -3.7e18`
    /// against a true closed form of `4.8e22`).
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
        // Conservative, closed-form-independent overflow guard: estimate
        // |lambda| as E / gap (dropping the O(1)-O(3) (1+nu) factor, which
        // only makes the true magnitude larger) and require it to stay
        // within a safety margin of Fix128::MAX. Multiplying by `gap`
        // instead of dividing by it avoids any issue from `gap` itself
        // being extremely small.
        let safe_max = Fix128::from_raw(i64::MAX, u64::MAX) / Fix128::from_int(16);
        let gap_hi = half() - poissons_ratio;
        let gap_lo = poissons_ratio - Fix128::NEG_ONE;
        if youngs_modulus_mpa >= gap_hi * safe_max || youngs_modulus_mpa >= gap_lo * safe_max {
            return Err(FemError::InvalidMaterial(
                "Poisson's ratio is too close to the 0.5 / -1 singularity for this Young's modulus: lame() would overflow Fix128",
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

/// An isotropic thermal eigenstrain, driven by a temperature field.
///
/// `ε_th = α ΔT · I`, with `ΔT` read from a [`CoupledField`] at each element
/// centroid and `α` the linear expansion coefficient (K⁻¹). The constitutive
/// law the solve then uses is
///
/// ```text
/// σ = λ tr(ε − ε_th) I + 2μ (ε − ε_th)
/// ```
///
/// which is the channel [`crate::coupled_field`] exists for: the temperature
/// field reaches the **residual**, not just a post-hoc stress estimate the way
/// [`crate::thermal_stress`] computes `σ = E α ΔT` for a fully suppressed bar.
///
/// # ⚠️ The field carries the rise above the stress-free reference
///
/// Not an absolute temperature. `ΔT = 0` everywhere must mean "no eigenstrain",
/// and that is what [`solve`] relies on when it forwards [`None`]. A caller
/// holding absolute temperatures has to subtract its own reference before
/// filling the field; this type cannot do it, because the reference is a
/// property of the configuration the mesh was built in and not of the field.
///
/// [`TemperatureRise`] is that subtraction made into a type, and
/// [`Self::from_rise`] is the entry point that accepts it. Prefer it to
/// [`Self::new`], which takes a bare [`CoupledField`] and therefore cannot tell
/// a rise from an absolute temperature — the reference temperature then enters
/// as if it were a rise, worth `E α T_ref / (1 − 2ν)` of stress (291.666667 MPa
/// for PLA at a 25 K reference). `examples/thermoelastic_rise.rs` prints both.
///
/// # Fixed-point note
///
/// `α` and `ΔT` are [`Fix128`], so `α ΔT` is exact only when both are dyadic
/// (denominator a power of two). `α = 1/1000` is not, so the eigenstrain
/// carries one truncation per multiply — about `2⁻⁶⁴` relative, which is why
/// the analytic oracles compare against a closed form with a tolerance rather
/// than for equality. Nothing here is compared for equality, so no operand
/// needs to be dyadic; a caller that *does* want an exact `assert_eq!` has to
/// choose dyadic `α` and `ΔT` and match the operation order on both sides.
#[derive(Debug, Clone, Copy)]
pub struct ThermalExpansion<'a> {
    field: &'a CoupledField,
    alpha_per_k: Fix128,
}

impl<'a> ThermalExpansion<'a> {
    /// Pair a temperature-rise field with a linear expansion coefficient.
    ///
    /// No validation: every [`Fix128`] is a usable `α` (negative expansion
    /// coefficients are real materials) and `field` is already a validated
    /// grid. The one condition that *can* fail — the field covering the mesh —
    /// needs the mesh, so [`solve_with_eigenstrain`] checks it and reports
    /// [`FemError::TemperatureFieldDoesNotCoverMesh`].
    #[must_use]
    #[deprecated(
        since = "1.5.0",
        note = "a bare `&CoupledField` cannot say whether it holds rises or absolute \
                temperatures, and an absolute one loads the reference as a rise; build a \
                `coupled_field::TemperatureRise` with `TemperatureRise::from_absolute` and \
                pass it to `ThermalExpansion::from_rise`"
    )]
    pub const fn new(field: &'a CoupledField, alpha_per_k: Fix128) -> Self {
        Self { field, alpha_per_k }
    }

    /// Pair a [`TemperatureRise`] with a linear expansion coefficient.
    ///
    /// The eigenstrain is specified on the rise above the stress-free
    /// reference, and [`TemperatureRise`] is the only type that records that a
    /// reference was subtracted — see its documentation for why a temperature
    /// and a temperature rise are different kinds of quantity and what routing
    /// an absolute field in here costs.
    ///
    /// Validation is the same as [`Self::new`]: none here, because the only
    /// failure mode (the field covering the mesh) needs the mesh and is checked
    /// by [`solve_with_eigenstrain`].
    #[must_use]
    pub const fn from_rise(rise: &'a TemperatureRise, alpha_per_k: Fix128) -> Self {
        Self {
            field: rise.field(),
            alpha_per_k,
        }
    }

    /// The temperature-rise field (K above the stress-free reference).
    #[must_use]
    pub const fn field(&self) -> &'a CoupledField {
        self.field
    }

    /// Linear expansion coefficient (K⁻¹).
    #[must_use]
    pub const fn alpha_per_k(&self) -> Fix128 {
        self.alpha_per_k
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
    /// `σ : C⁻¹ : σ`, the complementary energy density of this stress (MPa).
    ///
    /// Inverting Hooke's law gives `ε = (σ − λ/(3λ+2μ) · tr(σ) I) / 2μ`, so
    ///
    /// ```text
    /// σ : C⁻¹ : σ = [ σ:σ − λ/(3λ + 2μ) · (tr σ)² ] / 2μ
    /// ```
    ///
    /// with `σ:σ` counting the off-diagonals **twice**, because `xy`, `yz` and
    /// `zx` are tensor components and each stands for two entries of the matrix.
    ///
    /// This is the norm [`error_indicators_squared`] measures a recovered stress
    /// difference in, and it is the energy norm the finite element method is
    /// optimal in, which is what makes an indicator built on it comparable across
    /// elements of different size and stiffness.
    ///
    /// # Closed forms, for three states that isolate the two terms
    ///
    /// | state | value |
    /// |---|---|
    /// | uniaxial `σ_xx = s` | `s² / E` |
    /// | pure shear `σ_xy = s` | `s² / μ = 2(1+ν) s² / E` |
    /// | hydrostatic `σ_xx = σ_yy = σ_zz = p` | `3(1−2ν) p² / E` |
    ///
    /// The shear case is the only one that sees the factor of two, and the other
    /// two are the only ones that see `λ/(3λ+2μ)`, so the three together pin both
    /// terms. They are asserted in
    /// `tests/analytic_adaptive_refinement.rs::the_energy_norm_matches_its_closed_forms`.
    ///
    /// ⚠️ Never negative for a real stress — the form is positive definite — but
    /// `Fix128` rounding can take a value that should be zero a hair below it, so
    /// a caller that needs a non-negative number should clamp.
    #[must_use]
    pub fn complementary_energy_density(&self, material: &ElasticMaterial) -> Fix128 {
        let (lambda, mu) = material.lame();
        let two_mu = mu + mu;
        let bulk_share = lambda / (Fix128::from_int(3) * lambda + two_mu);
        let double_dot = self.xx * self.xx
            + self.yy * self.yy
            + self.zz * self.zz
            + (self.xy * self.xy + self.yz * self.yz + self.zx * self.zx) * Fix128::from_int(2);
        let trace = self.xx + self.yy + self.zz;
        (double_dot - bulk_share * trace * trace) / two_mu
    }

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

/// `√(Σ vᵢ²)` through [`crate::coupled_iteration::residual_norm_l2_checked`]:
/// bit-identical to `dot(v, v).sqrt()` whenever that sum is faithful (the
/// product of a value with itself does not depend on its sign), and
/// [`FemError::ResidualNormUnfaithful`] instead of a wrapped number when it is
/// not.
fn checked_l2_norm(v: &[Fix128]) -> Result<Fix128, FemError> {
    crate::coupled_iteration::residual_norm_l2_checked(v).map_err(|e| match e {
        crate::coupled_iteration::CoupledIterationError::ArithmeticWrapped { sweeps } => {
            FemError::ResidualNormUnfaithful { index: sweeps }
        }
        other => FemError::CoupledSubIterationFailed(other),
    })
}

/// [`crate::coupled_iteration::run_sub_iteration`] for a sweep that can fail.
///
/// The monitor's sweep closure returns a bare residual, so a failing sweep is
/// recorded here and reported as a zero residual to stop the monitor; the
/// recorded failure is then returned **before** the monitor's verdict is
/// read, because a zero residual is what the monitor calls converged.
fn run_sub_iteration_fallible<F>(
    config: crate::coupled_iteration::SubIterationConfig,
    mut sweep: F,
) -> Result<crate::coupled_iteration::SubIterationReport, FemError>
where
    F: FnMut(u32) -> Result<Fix128, FemError>,
{
    let mut failure: Option<FemError> = None;
    let outcome = crate::coupled_iteration::run_sub_iteration(config, |index| {
        if failure.is_some() {
            return Fix128::ZERO;
        }
        match sweep(index) {
            Ok(residual) => residual,
            Err(e) => {
                failure = Some(e);
                Fix128::ZERO
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    outcome.map_err(FemError::CoupledSubIterationFailed)
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// Refuses an element whose stiffness terms cannot be formed in `Fix128`.
///
/// A sliver of height `h` has shape function gradients of order `1/h`, and
/// the stiffness and curvature terms multiply two of them with the modulus
/// `λ + 2μ`. Past the integer range (`2^63`) that product wraps, the curvature
/// test reads a wrapped value, and the solve reported `UnderConstrained`
/// although the element was constrained. Such an element is reported as
/// `DegenerateElement` instead (AUD-A-S1W3-004): `max|∇N|² · (λ + 2μ)`
/// must be representable.
fn check_element_scale(elements: &[Element], lambda: Fix128, mu: Fix128) -> Result<(), FemError> {
    let modulus = (lambda + mu + mu).abs();
    for (t, e) in elements.iter().enumerate() {
        let gmax = e
            .grad
            .iter()
            .flat_map(|g| g.iter())
            .map(|c| c.abs())
            .max()
            .unwrap_or(Fix128::ZERO);
        let fits = gmax
            .checked_mul(gmax)
            .and_then(|g2| g2.checked_mul(modulus))
            .is_some();
        if !fits {
            return Err(FemError::DegenerateElement { tet: t });
        }
    }
    Ok(())
}

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
    let b_norm = checked_l2_norm(b)?;
    // Never ask for a residual norm the representation cannot express; see
    // `RESIDUAL_NORM_FLOOR`.
    let requested = config.relative_tolerance * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };

    let mut iterations = 0u32;
    let mut residual_norm = checked_l2_norm(&r)?;
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
        residual_norm = checked_l2_norm(&r)?;
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
/// Equivalent to [`solve_with_eigenstrain`] with no eigenstrain, which is the
/// same system as a uniform `ΔT = 0`: the thermal load and the stress
/// correction are both zero and the arithmetic is bit-identical to the solve
/// this function performed before the eigenstrain term existed.
///
/// # Errors
///
/// See [`FemError`]. [`FemError::TemperatureFieldDoesNotCoverMesh`] cannot
/// occur here, because there is no field.
pub fn solve(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &SolverConfig,
) -> Result<FemSolution, FemError> {
    solve_with_eigenstrain(mesh, material, boundary, config, None)
}

/// Solve the linear elastic boundary value problem on `mesh`, with an optional
/// thermal eigenstrain.
///
/// # The eigenstrain enters the residual, not the answer
///
/// With `ε_th = α ΔT · I` the constitutive law becomes
/// `σ = C : (ε − ε_th)`, so the weak form `∫ Bᵀ σ dV = f_ext` reads
///
/// ```text
/// K u = f_ext + ∫ Bᵀ C ε_th dV
///               ^^^^^^^^^^^^^^^ the thermal load, assembled element by element
/// ```
///
/// and the reported stress is `C B u − C : ε_th`. Both halves are needed and
/// each is visible on its own: with only the load the free-expansion scene gets
/// the right displacement and a non-zero stress, and with only the correction
/// the displacement stays put and the clamped scene passes for the wrong
/// reason. `tests/analytic_thermoelastic.rs` pins that pair.
///
/// A uniform `ε_th` loads no interior degree of freedom — `Σₑ V_e ∇Nᵢ = 0`
/// there, because `Nᵢ` vanishes on the boundary of its own support — so the
/// load shows up only next to constrained nodes, as the reaction it physically
/// is. That is why a fully clamped body heated uniformly has `u ≡ 0` exactly
/// and carries the whole eigenstrain as stress.
///
/// # Determinism
///
/// The eigenstrain is built in one pass over the elements in mesh order before
/// the iteration starts, and sampled with [`CoupledField::sample`], which is
/// [`Fix128`] trilinear interpolation. Nothing new reaches a transcendental, so
/// the bit-exactness the module doc claims is unchanged.
///
/// # Errors
///
/// See [`FemError`]. [`FemError::TemperatureFieldDoesNotCoverMesh`] is reported
/// before any solving when `thermal` is `Some` and the field does not cover
/// every node of the mesh.
pub fn solve_with_eigenstrain(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &SolverConfig,
    thermal: Option<ThermalExpansion<'_>>,
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
    check_element_scale(&elements, lambda, mu)?;
    let ndof = vertex_count * 3;

    // `C : ε_th` per element, or nothing at all. Built after `build_elements`,
    // which is what validated the node indices this reads back.
    let thermal_stress = match &thermal {
        None => None,
        Some(t) => Some(thermal_stresses(mesh, &elements, t, lambda, mu)?),
    };

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

    // b = f_ext + ∫ Bᵀ C ε_th dV − K u_prescribed, restricted to the free
    // degrees of freedom.
    let mut scratch = vec![Fix128::ZERO; ndof];
    apply_stiffness(&elements, &prescribed_value, lambda, mu, &mut scratch);
    let mut b = vec![Fix128::ZERO; ndof];
    for &(vertex, axis, force) in &boundary.loads {
        let d = vertex as usize * 3 + axis.index();
        if is_free[d] {
            b[d] = b[d] + force;
        }
    }
    if let Some(stresses) = &thermal_stress {
        // Accumulated in mesh order, like every other element loop here. The
        // prescribed entries this writes are discarded by the loop below —
        // there they are the reaction, not a load, and [`reactions`] is what
        // reports them.
        add_eigenstrain_load(&elements, stresses, &mut b);
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
    // σ = C B u − C : ε_th. The subtraction is what makes a free expansion
    // stress-free; without it the mechanical part alone reports the full
    // `(3λ + 2μ) α ΔT` on a body that is storing no energy.
    let element_stress = match &thermal_stress {
        None => elements
            .iter()
            .map(|e| element_stress(e, &x, lambda, mu))
            .collect(),
        Some(stresses) => elements
            .iter()
            .zip(stresses.iter())
            .map(|(e, &s)| sub_stress(element_stress(e, &x, lambda, mu), s))
            .collect(),
    };

    Ok(FemSolution {
        displacements,
        element_stress,
        iterations: cg.iterations,
        relative_residual: relative(cg.residual_norm, cg.b_norm),
        effective_relative_tolerance: relative(cg.target, cg.b_norm),
    })
}

/// `out ← out + Σₑ ∫ Bᵀ (C : ε_th) dV`, accumulated element by element in mesh
/// order.
///
/// The thermal load of [`solve_with_eigenstrain`], factored out so that
/// [`reactions`] subtracts the **same** vector the solve added rather than a
/// second copy of the same formula. A mistake in the element integral then
/// moves both, which is what lets the reaction act as an oracle for it.
fn add_eigenstrain_load(elements: &[Element], stresses: &[StressTensor], out: &mut [Fix128]) {
    for (element, &s) in elements.iter().zip(stresses.iter()) {
        let force = element_force_from_stress(element, s);
        for (f, &node) in force.iter().zip(element.nodes.iter()) {
            let base = node * 3;
            out[base] = out[base] + f[0];
            out[base + 1] = out[base + 1] + f[1];
            out[base + 2] = out[base + 2] + f[2];
        }
    }
}

/// Flatten a nodal displacement array into the `3·vertex_count` layout every
/// element loop in this module reads.
fn flatten(displacements: &[[Fix128; 3]]) -> Vec<Fix128> {
    let mut u = vec![Fix128::ZERO; displacements.len() * 3];
    for (v, d) in displacements.iter().enumerate() {
        u[v * 3] = d[0];
        u[v * 3 + 1] = d[1];
        u[v * 3 + 2] = d[2];
    }
    u
}

/// The free/prescribed mask of a boundary condition set, over `ndof` rows.
fn free_mask(boundary: &BoundaryConditions, ndof: usize) -> Vec<bool> {
    let mut is_free = vec![true; ndof];
    for &(vertex, axis, _) in &boundary.prescribed {
        is_free[vertex as usize * 3 + axis.index()] = false;
    }
    is_free
}

/// Shared entry checks for the two reaction functions: a usable mesh, boundary
/// data that names vertices the mesh has, and a solution of the right length.
fn check_reaction_inputs(
    mesh: &SdfTetMesh,
    boundary: &BoundaryConditions,
    nodes: usize,
) -> Result<usize, FemError> {
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
    if nodes != vertex_count {
        return Err(FemError::SolutionDoesNotMatchMesh {
            nodes,
            vertex_count,
        });
    }
    Ok(vertex_count)
}

/// Support forces at the prescribed degrees of freedom of a [`solve`] or
/// [`solve_with_eigenstrain`] answer (N), indexed like `SdfTetMesh::vertices`.
///
/// # What it is
///
/// ```text
/// R_d = f_int(u)_d − f_ext_d     on a prescribed degree of freedom
/// R_d = 0                        on a free one
/// ```
///
/// with `f_int(u) = K u − ∫ Bᵀ(C : ε_th) dV` the assembled internal force — the
/// force the body exerts on its supports, with the sign that makes
/// `Σ R + Σ f_ext = 0` over the whole mesh. A free row carries no reaction by
/// construction: that row *is* the equilibrium equation the solve satisfied, so
/// the same expression there is the residual and is zero to solver tolerance.
///
/// # ⚠️ Why this is not a field of [`FemSolution`]
///
/// It is the only quantity in this module that a displacement-driven scene
/// cannot reveal. With `f_ext = 0` the discrete problem is `K u = 0` on the free
/// rows, so **scaling the whole internal force by a constant leaves `u`
/// untouched** — the answer, the reported stress and every oracle written on
/// them are all blind to it. The reaction is linear in that same constant, so it
/// is where such an error shows up. Measured 2026-10-01 on the quadratic and
/// cubic elements: dropping the `J` from `P = J σ F⁻ᵀ` left 7 of 7 oracles green,
/// including one asserting a closed form for `σ_xx`.
///
/// # ⚠️ The shared assembly is load-bearing — do not re-derive it here
///
/// This goes through `apply_stiffness` and `add_eigenstrain_load` (both
/// private), the very functions [`solve_with_eigenstrain`] builds its own
/// system with, and **not**
/// through a second copy of the same formulae. That sharing is what makes the
/// closed-form oracle work: a mistake in the element integral has to *reach*
/// this value before anything can compare it against the analytic traction.
///
/// ⚠️ **Re-deriving it here would delete the oracle while every test stayed
/// green** — the mutation would simply stop flowing into the reaction, so
/// `tests/analytic_reactions.rs` would keep passing and the error would be
/// invisible again. No test can guard this; only the comment can. A review
/// note asking for an independent implementation "to avoid duplication" is
/// asking for that outcome.
///
/// The complement is that the *comparison* must stay independent: the oracles
/// check `R_face = −A₀ · T e_face` from the constitutive law, never `residual
/// ≈ 0`, which would be the assembly checking itself.
///
/// # ⚠️ What it does not check
///
/// Nothing ties `solution` to the arguments beside it. This reports the support
/// forces of *this* boundary and *this* material at *that* displacement field;
/// if the field came from a different problem the answer is the reaction of a
/// problem nobody solved. Only the node count is checked.
///
/// A load placed on a prescribed degree of freedom is subtracted here even
/// though [`solve`] ignores it — the support carries it, which is precisely what
/// "ignored by the solve" means.
///
/// # Errors
///
/// [`FemError::EmptyMesh`], [`FemError::VertexOutOfRange`],
/// [`FemError::DegenerateElement`], [`FemError::SolutionDoesNotMatchMesh`], and
/// [`FemError::TemperatureFieldDoesNotCoverMesh`] when `thermal` is `Some` and
/// the field does not cover every node.
#[must_use = "the support forces are the whole point of calling this; a solve that \
     drops them has not been checked against equilibrium at all"]
pub fn reactions(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    thermal: Option<ThermalExpansion<'_>>,
    solution: &FemSolution,
) -> Result<Vec<[Fix128; 3]>, FemError> {
    let vertex_count = check_reaction_inputs(mesh, boundary, solution.displacements.len())?;
    let elements = build_elements(mesh)?;
    let (lambda, mu) = material.lame();
    check_element_scale(&elements, lambda, mu)?;
    let ndof = vertex_count * 3;

    let thermal_stress = match &thermal {
        None => None,
        Some(t) => Some(thermal_stresses(mesh, &elements, t, lambda, mu)?),
    };

    let u = flatten(&solution.displacements);

    // `f_int = K u − thermal load`, through the two functions
    // `solve_with_eigenstrain` builds its own system with.
    let mut force = vec![Fix128::ZERO; ndof];
    apply_stiffness(&elements, &u, lambda, mu, &mut force);
    if let Some(stresses) = &thermal_stress {
        let mut load = vec![Fix128::ZERO; ndof];
        add_eigenstrain_load(&elements, stresses, &mut load);
        for (value, taken) in force.iter_mut().zip(load.iter()) {
            *value = *value - *taken;
        }
    }

    for &(vertex, axis, applied) in &boundary.loads {
        let d = vertex as usize * 3 + axis.index();
        force[d] = force[d] - applied;
    }
    let is_free = free_mask(boundary, ndof);
    for (d, value) in force.iter_mut().enumerate() {
        if is_free[d] {
            *value = Fix128::ZERO;
        }
    }
    Ok((0..vertex_count)
        .map(|v| [force[v * 3], force[v * 3 + 1], force[v * 3 + 2]])
        .collect())
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
    check_element_scale(&elements, lambda, mu)?;
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

pub(crate) mod consistent_tangent;

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
    consistent_tangent: bool,
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
            consistent_tangent: false,
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
    /// `μ` (or `C₁`, `C₂`, `C₃`) comes from `model`; the volumetric modulus that
    /// fixes the pressure comes from the [`ElasticMaterial`] passed to
    /// [`solve_corotational`], as `κ = λ − offset(model)` so that the stress
    /// linearises to that solid's `λ`. A model whose offset exceeds `λ`
    /// (`λ < 4C₂` Mooney-Rivlin, `λ < 8C₂` Yeoh) is refused with
    /// [`FemError::InvalidConfig`]. `μ` is not cross-checked: the model's and the
    /// material's may disagree, and keeping them the same solid is the caller's to
    /// do.
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

    /// Solve the hyperelastic law with a **consistent tangent** instead of the
    /// co-rotational linear one.
    ///
    /// # What changes
    ///
    /// The Newton step is `K_t(u) δ = r(u)` with `K_t` the derivative of the
    /// first Piola-Kirchhoff stress that [`Self::with_hyperelastic`] evaluates,
    /// applied matrix-free, instead of the modified iteration's fixed
    /// co-rotational linear operator. The converged field is the same
    /// equilibrium; what changes is how many steps it takes — measured on the
    /// 4 mm cube of `tests/analytic_corotational.rs` at a 125 % stretch with a
    /// point load, **181 steps become 3** at 200 N and **758 become 4** at
    /// 2000 N. A step that would lower no residual is shortened (backtracking,
    /// with `det F > 0` required), and a step whose tangent is not positive
    /// definite along the first direction is taken with the linear surrogate.
    ///
    /// The first step of every increment is the surrogate's: that state is a
    /// prediction, which at the first increment is the undeformed interior with
    /// the boundary already moved and can hold an inverted element.
    ///
    /// ⚠️ **`solve_quadratic_hyperelastic` and `solve_cubic_hyperelastic` honour
    /// this flag too**, with a different tangent action: those elements have no
    /// assembled tangent, so the action is the central difference of the material
    /// internal force (a Newton–Krylov step, `Fix128` throughout, no libm). It
    /// does not move the fixed point. It is what makes a smooth non-affine field at
    /// `|∇u| ≈ 0.4` converge, where the modified iteration ends `NotConverged`
    /// identically at 80 steps, 400 steps and 16 increments
    /// (`tests/analytic_hyperelastic_mms_order.rs`).
    ///
    /// Without a law from [`Self::with_hyperelastic`] there is nothing to
    /// differentiate, and [`solve_corotational`] refuses the configuration with
    /// [`FemError::InvalidConfig`] rather than ignoring the request.
    // ALLOW-UNWIRED: public opt-in for downstream solvers, the production caller chooses the tangent
    #[must_use]
    pub const fn with_consistent_tangent(mut self) -> Self {
        self.consistent_tangent = true;
        self
    }

    /// Whether [`Self::with_consistent_tangent`] was applied.
    #[must_use]
    pub const fn consistent_tangent(&self) -> bool {
        self.consistent_tangent
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

/// What the modified Newton step becomes once the residual line search has
/// answered: the accepted point when it found one, otherwise the full step pulled
/// back only as far as regularity needs ([`step_stays_regular`]), which is what
/// the iteration did before it had a line search.
fn settle_step(
    elements: &[Element],
    previous: &[Fix128],
    next: &mut [Fix128],
    accepted: Option<Vec<Fix128>>,
) {
    match accepted {
        Some(moved) => next.copy_from_slice(&moved),
        None => step_stays_regular(elements, previous, next),
    }
}

/// Largest number of halvings [`step_stays_regular`] tries.
const REGULARITY_BACKTRACKS: u32 = 16;

/// `true` when every element of `u` has `det F` above the polar floor, i.e. the
/// next pass can build its frames.
fn all_elements_regular(elements: &[Element], u: &[Fix128]) -> bool {
    elements
        .iter()
        .all(|e| deformation_gradient(e, &gather(e, u)).determinant() > POLAR_DET_FLOOR)
}

/// Pulls `next` back toward `previous` — `previous + 2⁻ᵏ (next − previous)` for
/// the smallest `k ≤ 16` — until no element is inverted or degenerate.
///
/// `previous` must itself be regular; with it that is a state the iteration just
/// built frames from. When even `k = 16` is not regular `next` is left at that
/// last point, so the caller meets the refusal instead of a silent success.
fn step_stays_regular(elements: &[Element], previous: &[Fix128], next: &mut [Fix128]) {
    if all_elements_regular(elements, next) {
        return;
    }
    let full: Vec<Fix128> = next.to_vec();
    for k in 1..=REGULARITY_BACKTRACKS {
        let alpha = Fix128::from_raw(0, 1u64 << (64 - k));
        for ((n, p), f) in next.iter_mut().zip(previous.iter()).zip(full.iter()) {
            *n = *p + (*f - *p) * alpha;
        }
        if all_elements_regular(elements, next) {
            return;
        }
    }
}

/// The volumetric modulus `κ` a hyperelastic solve runs its model under, for a
/// solid whose Lamé constant is `lambda`: `κ = λ − offset(model)`
/// ([`crate::hyperelastic::volumetric_modulus`]).
///
/// With it the stress linearises to `λ` for every model; the previous
/// `K = λ + 2μ/3` left `A₁₁₁₁(I) = λ + 5μ/3` against the linear `λ + 2μ`.
/// Shared by the P1, P2 and P3 solvers so that one wrong copy cannot hide.
///
/// # Errors
///
/// [`FemError::InvalidConfig`] when `κ < 0` (`λ < 4C₂` for Mooney-Rivlin,
/// `λ < 8C₂` for Yeoh). `κ ≥ 0` is sufficient for the volumetric energy to be
/// convex, not necessary, so this refuses conservatively.
pub(crate) fn hyperelastic_volumetric_modulus(
    model: &HyperelasticModel,
    lambda: Fix128,
) -> Result<Fix128, FemError> {
    crate::hyperelastic::volumetric_modulus(model, lambda).ok_or(FemError::InvalidConfig(
        "hyperelastic model needs lambda >= 4*C2 (Mooney-Rivlin) or 8*C2 (Yeoh): volumetric modulus would be negative",
    ))
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

/// One quarter, exactly.
#[inline]
fn quarter() -> Fix128 {
    Fix128::from_raw(0, 1 << 62)
}

/// `a − b`, component by component.
#[inline]
fn sub_stress(a: StressTensor, b: StressTensor) -> StressTensor {
    StressTensor {
        xx: a.xx - b.xx,
        yy: a.yy - b.yy,
        zz: a.zz - b.zz,
        xy: a.xy - b.xy,
        yz: a.yz - b.yz,
        zx: a.zx - b.zx,
    }
}

/// `C : ε_th` for every element (MPa) — the stress a fully suppressed thermal
/// expansion carries, which is both the source of the thermal load and the
/// correction the reported stress needs.
///
/// With `ε_th = α ΔT · I` the double contraction collapses to a hydrostatic
/// tensor: `λ tr(ε_th) + 2μ (ε_th)_aa = (3λ + 2μ) α ΔT` on every normal
/// component and zero on every shear one. `3λ + 2μ = 3K = E/(1 − 2ν)`, so this
/// is the bulk response and nothing else.
///
/// `ΔT` is sampled at the element centroid, which is exact for the uniform
/// field the analytic oracles use and first-order accurate otherwise — the same
/// order as the P1 constant-strain element it feeds, so it adds no error term
/// of its own kind.
fn thermal_stresses(
    mesh: &SdfTetMesh,
    elements: &[Element],
    thermal: &ThermalExpansion<'_>,
    lambda: Fix128,
    mu: Fix128,
) -> Result<Vec<StressTensor>, FemError> {
    // Coverage first, by node, so the error names a vertex of the mesh rather
    // than a centroid nobody can look up. See the error variant for why this is
    // a refusal and not a clamp.
    for element in elements {
        for &node in &element.nodes {
            let q = mesh.vertices[node];
            let p = Vec3Fix::new(
                Fix128::from_f32(q[0]),
                Fix128::from_f32(q[1]),
                Fix128::from_f32(q[2]),
            );
            if !thermal.field.contains(p) {
                return Err(FemError::TemperatureFieldDoesNotCoverMesh {
                    vertex: u32::try_from(node).unwrap_or(u32::MAX),
                });
            }
        }
    }

    let bulk_three = lambda + lambda + lambda + mu + mu;
    let mut out = Vec::with_capacity(elements.len());
    for element in elements {
        let mut sum = [Fix128::ZERO; 3];
        for &node in &element.nodes {
            let q = mesh.vertices[node];
            sum[0] = sum[0] + Fix128::from_f32(q[0]);
            sum[1] = sum[1] + Fix128::from_f32(q[1]);
            sum[2] = sum[2] + Fix128::from_f32(q[2]);
        }
        let centroid = Vec3Fix::new(sum[0] * quarter(), sum[1] * quarter(), sum[2] * quarter());
        let delta_t = thermal.field.sample(centroid);
        let normal = bulk_three * thermal.alpha_per_k * delta_t;
        out.push(StressTensor {
            xx: normal,
            yy: normal,
            zz: normal,
            ..StressTensor::default()
        });
    }
    Ok(out)
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
    out.copy_from_slice(f_ext);
    subtract_internal_force(assembly, u, material, out)?;
    for (d, value) in out.iter_mut().enumerate() {
        if !assembly.is_free[d] {
            *value = Fix128::ZERO;
        }
    }
    Ok(())
}

/// `out ← out − Σₑ f_int,ₑ(u)` over **every** row, with no constraint masking.
///
/// The element loop of [`corotational_residual`], split out so that
/// [`corotational_reactions`] reads the same assembly a Newton step does rather
/// than a second copy of it. The residual masks the prescribed rows because
/// there is no equation to satisfy on them; the reaction is exactly what is
/// behind that mask, so it has to be read before the mask goes on.
fn subtract_internal_force(
    assembly: &Assembly<'_>,
    u: &[Fix128],
    material: Option<(HyperelasticModel, Fix128)>,
    out: &mut [Fix128],
) -> Result<(), FemError> {
    let &Assembly {
        elements,
        rotations,
        lame: (lambda, mu),
        ..
    } = assembly;
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
    check_element_scale(&elements, lambda, mu)?;
    // `κ = λ − offset(model)` — the volumetric modulus that makes the stress
    // linearise to this solid's `λ`; see `hyperelastic_volumetric_modulus`. The
    // deviatoric response comes from the model in the configuration, so the two
    // are free to describe different solids; see
    // [`CorotationalConfig::with_hyperelastic`].
    let law = config
        .material
        .map(|model| hyperelastic_volumetric_modulus(&model, lambda).map(|k| (model, k)))
        .transpose()?;
    if config.consistent_tangent() && law.is_none() {
        return Err(FemError::InvalidConfig(
            "consistent_tangent needs a hyperelastic law (with_hyperelastic)",
        ));
    }
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

            // Step 0 is the state the increment *opened* at: a prediction, which at
            // increment 1 is the undeformed interior with the boundary already
            // moved and can hold an inverted element. The linear surrogate's step
            // below is exact and needs no valid deformation gradient, so it takes
            // step 0 exactly as it does for the modified iteration; the
            // consistent tangent takes over once the state is a candidate.
            if config.consistent_tangent() && step > 0 {
                // `with_consistent_tangent` is refused above without a law.
                if let Some(law) = law {
                    let assembly = Assembly {
                        elements: &elements,
                        rotations: &rotations,
                        lame: (lambda, mu),
                        is_free: &is_free,
                    };
                    let report = consistent_tangent::newton_step(
                        &assembly,
                        &mut u,
                        &f_ext,
                        law,
                        &config.linear,
                        &mut residual,
                    )?;
                    cg_iterations = cg_iterations.saturating_add(report.cg_iterations);
                    relative_residual = report.relative_residual;
                    effective_relative_tolerance = report.effective_relative_tolerance;
                    step += 1;
                    newton_iterations = newton_iterations.saturating_add(1);
                    continue;
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
            let mut next: Vec<Fix128> = (0..ndof)
                .map(|d| {
                    if is_free[d] {
                        cg.x[d]
                    } else {
                        boundary_field[d]
                    }
                })
                .collect();
            // ⚠️ **Globalisation for a material law.** The step is exact for the
            // *linear* law and only a surrogate for a hyperelastic one, whose
            // tangent leaves the surrogate as the stretch grows: a full step can
            // land on a state with an inverted element, and the next pass then
            // has no frame to build. `u` is a valid candidate here (its frames
            // were just built, `step > 0`), so the step is halved toward it until
            // every element is regular again. The linear law is untouched: its
            // step is exact and `step_stays_regular` is never asked.
            if law.is_some() && step > 0 {
                let assembly = Assembly {
                    elements: &elements,
                    rotations: &rotations,
                    lame: (lambda, mu),
                    is_free: &is_free,
                };
                let mut probe = vec![Fix128::ZERO; ndof];
                corotational_residual(&assembly, &u, &f_ext, law, &mut probe)?;
                let before = max_abs(&probe);
                let delta: Vec<Fix128> = next.iter().zip(u.iter()).map(|(n, p)| *n - *p).collect();
                let accepted = consistent_tangent::backtrack(before, &u, &delta, &is_free, |t| {
                    corotational_residual(&assembly, t, &f_ext, law, &mut probe)
                        .ok()
                        .map(|()| max_abs(&probe))
                });
                settle_step(&elements, &u, &mut next, accepted);
            }
            u.copy_from_slice(&next);
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

/// Support forces at the prescribed degrees of freedom of a
/// [`solve_corotational`] answer (N), indexed like `SdfTetMesh::vertices`.
///
/// The co-rotational and hyperelastic counterpart of [`reactions`]: same
/// definition, `R_d = f_int(u)_d − f_ext_d` on a prescribed row and zero on a
/// free one, with `f_int` the internal force of the law `config` selects —
/// `Σₑ R Kₑ⁰ (Rᵀx − X)` without a material model and the total-Lagrangian
/// `Σₑ V₀ P ∇₀N` with one. It runs through `subtract_internal_force` (private),
/// which is the element loop a Newton step assembles its residual with.
///
/// # ⚠️ Why this is the only way to see `P`
///
/// The Cauchy stress [`CorotationalSolution`] reports is computed *before* the
/// `P = J σ F⁻ᵀ` conversion, so no error in that conversion reaches it, and a
/// displacement-driven solve cannot see one either: it checks equilibrium with
/// the force it assembled, so a wrong `P` that is still a gradient converges to
/// a different field and calls it equilibrium. The reaction is the assembled
/// force itself, read where the solve throws it away. The `#[cfg(test)] mod
/// tests` at the end of this file covers `P` at the element level for the same
/// reason; this covers the integral of it.
///
/// # ⚠️ The shared assembly is load-bearing — do not re-derive it here
///
/// `subtract_internal_force` is the element loop a Newton step assembles its
/// residual with, reached here and there through the same call. See
/// [`reactions`] for why that sharing cannot be replaced by an independent
/// implementation: ⚠️ **re-deriving it would silently remove the oracle, with
/// every test still green**, because the error would no longer reach the value
/// the closed form is compared against.
///
/// # ⚠️ The frames are recomputed, not recovered
///
/// [`CorotationalSolution`] does not carry the element frames, so they are
/// rebuilt from `solution` with [`Mat3Fix::polar_rotation`] at
/// `config.polar_iterations`. The stopping rule of the last increment is that
/// the frames computed from the answer are within `FRAME_SETTLED` (private) of
/// the ones the answer was built on, so these agree with the solve's to that
/// much and not bit for bit. With a material model set the frames are not read
/// at all — a hyperelastic stress is objective, so the internal force carries no
/// frame.
///
/// # Errors
///
/// As [`reactions`], plus [`FemError::RotationFailed`] when an element of
/// `solution` has no polar factor.
#[must_use = "the support forces are the whole point of calling this; a solve that \
     drops them has not been checked against equilibrium at all"]
pub fn corotational_reactions(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &CorotationalConfig,
    solution: &CorotationalSolution,
) -> Result<Vec<[Fix128; 3]>, FemError> {
    let vertex_count = check_reaction_inputs(mesh, boundary, solution.field.displacements.len())?;
    let elements = build_elements(mesh)?;
    let (lambda, mu) = material.lame();
    check_element_scale(&elements, lambda, mu)?;
    // The same volumetric modulus `solve_corotational` pairs the model with.
    let law = config
        .material
        .map(|model| hyperelastic_volumetric_modulus(&model, lambda).map(|k| (model, k)))
        .transpose()?;
    let ndof = vertex_count * 3;

    let u = flatten(&solution.field.displacements);
    let is_free = free_mask(boundary, ndof);

    let mut rotations = Vec::with_capacity(elements.len());
    for (tet, element) in elements.iter().enumerate() {
        let frame = deformation_gradient(element, &gather(element, &u))
            .polar_rotation(POLAR_DET_FLOOR, config.polar_iterations)
            .map_err(|cause| FemError::RotationFailed { tet, cause })?;
        rotations.push(frame);
    }

    // `out = −f_int`, then `−(out + f_ext) = f_int − f_ext`.
    let mut out = vec![Fix128::ZERO; ndof];
    subtract_internal_force(
        &Assembly {
            elements: &elements,
            rotations: &rotations,
            lame: (lambda, mu),
            is_free: &is_free,
        },
        &u,
        law,
        &mut out,
    )?;
    for &(vertex, axis, applied) in &boundary.loads {
        let d = vertex as usize * 3 + axis.index();
        out[d] = out[d] + applied;
    }
    for (d, value) in out.iter_mut().enumerate() {
        *value = if is_free[d] { Fix128::ZERO } else { -*value };
    }
    Ok((0..vertex_count)
        .map(|v| [out[v * 3], out[v * 3 + 1], out[v * 3 + 2]])
        .collect())
}

// ============================================================================
// Small-strain J2 elastoplasticity
// ============================================================================

/// Largest yield stress or hardening modulus accepted, in MPa (`2³⁰`).
///
/// The return mapping squares stresses, and [`Fix128`] wraps on overflow
/// instead of saturating, so a bound has to be enforced where the setting
/// enters. `2³⁰` MPa is a thousand times any structural material.
const PLASTIC_PARAMETER_MAX: i64 = 1 << 30;

/// Largest load factor accepted in a load path, in magnitude (`2²⁰`), for the
/// same reason as [`PLASTIC_PARAMETER_MAX`].
const LOAD_FACTOR_MAX: i64 = 1 << 20;

/// Settings for [`solve_elastoplastic`].
///
/// Small-strain **J2 (von Mises) plasticity with bilinear isotropic
/// hardening**: the material yields when the von Mises stress reaches
/// `σ_y + H·ε̄_p`, where `ε̄_p` is the accumulated equivalent plastic strain and
/// `H = dσ_y/dε̄_p` is the *plastic* modulus. A uniaxial test therefore shows the
/// tangent `E_t = E·H / (E + H)` past yield, and `H = 0` is perfect plasticity.
///
/// The fields are private and checked by [`Self::try_new`], so an out-of-range
/// yield stress or hardening modulus cannot be constructed. The struct is
/// `#[non_exhaustive]` so a later model (kinematic hardening, say) can add
/// settings without a breaking change.
///
/// [`Default`] is "plasticity effectively off": a yield stress of `2³⁰` MPa, no
/// hardening, 50 Newton iterations at `2⁻²⁰` relative tolerance and the default
/// [`SolverConfig`]. It is a valid configuration that reproduces the linear
/// elastic answer for any physical load, not a recommendation of a material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ElastoplasticConfig {
    linear: SolverConfig,
    newton_iterations: u32,
    newton_tolerance: Fix128,
    yield_stress_mpa: Fix128,
    hardening_modulus_mpa: Fix128,
}

impl ElastoplasticConfig {
    /// Validate and build.
    ///
    /// `linear` configures the conjugate gradient solve of each Newton
    /// iteration. `newton_tolerance` is the fraction of the reference residual
    /// (the larger of the step's external load and its first residual) below
    /// which an iteration counts as converged; it should sit above
    /// `linear.relative_tolerance`, because the linear solve is what limits how
    /// small one iteration can make the residual.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] for a zero Newton budget, a tolerance
    /// outside `(0, 1)`, a yield stress that is not positive or exceeds `2³⁰`
    /// MPa, or a hardening modulus that is negative (softening is not
    /// supported: the tangent loses positive definiteness and the solution is
    /// no longer unique) or exceeds `2³⁰` MPa.
    pub fn try_new(
        linear: SolverConfig,
        newton_iterations: u32,
        newton_tolerance: Fix128,
        yield_stress_mpa: Fix128,
        hardening_modulus_mpa: Fix128,
    ) -> Result<Self, FemError> {
        if newton_iterations == 0 {
            return Err(FemError::InvalidConfig(
                "newton_iterations must be positive",
            ));
        }
        if newton_tolerance <= Fix128::ZERO || newton_tolerance >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "newton_tolerance must be a fraction strictly between 0 and 1",
            ));
        }
        let cap = Fix128::from_int(PLASTIC_PARAMETER_MAX);
        if yield_stress_mpa <= Fix128::ZERO || yield_stress_mpa > cap {
            return Err(FemError::InvalidConfig(
                "yield_stress_mpa must be positive and at most 2^30",
            ));
        }
        if hardening_modulus_mpa.is_negative() || hardening_modulus_mpa > cap {
            return Err(FemError::InvalidConfig(
                "hardening_modulus_mpa must be in [0, 2^30] (softening is not supported)",
            ));
        }
        Ok(Self {
            linear,
            newton_iterations,
            newton_tolerance,
            yield_stress_mpa,
            hardening_modulus_mpa,
        })
    }
}

impl Default for ElastoplasticConfig {
    /// Plasticity effectively off; see the type documentation.
    fn default() -> Self {
        Self {
            linear: SolverConfig::default(),
            newton_iterations: 50,
            newton_tolerance: Fix128::from_raw(0, 1 << 44),
            yield_stress_mpa: Fix128::from_int(PLASTIC_PARAMETER_MAX),
            hardening_modulus_mpa: Fix128::ZERO,
        }
    }
}

/// Result of [`solve_elastoplastic`], at the end of the load path.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElastoplasticSolution {
    /// Displacements and the Cauchy stress per element. `iterations`,
    /// `relative_residual` and `effective_relative_tolerance` describe the last
    /// conjugate gradient solve (zero if none ran).
    pub field: FemSolution,
    /// Plastic strain per element, **tensor** components in the
    /// [`StressTensor`] layout (`xy` is `ε_xy`, half the engineering shear). The
    /// J2 flow is deviatoric, so `xx + yy + zz = 0`.
    pub plastic_strain: Vec<StressTensor>,
    /// Accumulated equivalent plastic strain `ε̄_p` per element.
    pub equivalent_plastic_strain: Vec<Fix128>,
    /// Accumulated plastic work per element, `W_p = Σ_k σ^{k+1} : Δε_p^k`, in
    /// MPa (equivalently MJ/m³, since the mesh is in millimetres and the
    /// stresses in megapascals). Zero for an element that never yielded.
    ///
    /// This is the energy that left the elastic store, and
    /// [`PlasticHeating`] turns it into the temperature rise that
    /// [`ThermalExpansion`] reads back — the return leg of the coupling whose
    /// forward leg is [`solve_with_eigenstrain`].
    ///
    /// # ⚠️ It is a path integral, and `ε̄_p` is not
    ///
    /// Integrating the hardening law over the *discrete* path gives
    ///
    /// ```text
    /// W_p = σ_y·ε̄_p + (H/2)(ε̄_p² + Σ_k Δε̄_k²)
    /// ```
    ///
    /// which exceeds the continuous `σ_y ε̄_p + (H/2) ε̄_p²` by `(H/2) Σ Δε̄_k²`.
    /// That term is `O(1/N)` in the number of plastic steps, so **two solves
    /// that end at the same `ε̄_p` report different `W_p` when their load paths
    /// were cut differently**. With `H = 0` the dependence disappears and
    /// `W_p = σ_y·ε̄_p` holds for any path.
    ///
    /// A consequence worth stating because it bites oracles: `ε̄_p` alone
    /// cannot validate this field. It is the same number for every step count,
    /// so a test that reads it is structurally unable to see a wrong
    /// integration rule here.
    pub dissipation: Vec<Fix128>,
    /// Newton iterations (linear solves) over the whole path.
    pub newton_iterations: u32,
    /// Load steps taken, the length of the load path.
    pub steps: u32,
}

/// Conversion of plastic work into heat (Taylor–Quinney).
///
/// The fraction `β` of the plastic work that leaves the material as heat
/// rather than being stored in the dislocation structure as cold work, and the
/// **volumetric** heat capacity `c_v` that turns that heat into a temperature
/// rise:
///
/// ```text
/// ΔT = β · W_p / c_v
/// ```
///
/// Fields are private and checked by [`Self::try_new`]; the struct is
/// `#[non_exhaustive]` so a later model (a temperature-dependent `β`, say) can
/// add settings without a breaking change, and `try_new` is the public way to
/// build one from outside the crate.
///
/// # ⚠️ Units: `c_v` is volumetric, in MPa/K
///
/// The module works in millimetre / newton / megapascal, so `W_p` comes out in
/// MPa = MJ/m³ — energy per unit volume. Dividing by a volumetric heat
/// capacity in the same MJ/(m³·K) = MPa/K leaves kelvin with no conversion
/// factor at all. For steel, `ρ c_p ≈ 7850 kg/m³ × 486 J/(kg·K) ≈ 3.82 MPa/K`.
///
/// Passing a *specific* heat capacity (J/(kg·K), around 486 for steel) instead
/// is off by the density and gives a rise some 7850 times too small; the field
/// is named for its units so the call site has to say which it means.
///
/// `β` is conventionally 0.9 for metals. `β = 0` is the "all cold work" limit
/// and is accepted: it produces exactly zero rise, which is a statement about
/// the material rather than a silent no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlasticHeating {
    taylor_quinney: Fix128,
    volumetric_heat_capacity_mpa_per_k: Fix128,
}

impl PlasticHeating {
    /// Validate and build.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] if `taylor_quinney` is outside `[0, 1]`
    /// (above it the heat would exceed the work, below it the material would
    /// cool while dissipating) or if `volumetric_heat_capacity_mpa_per_k` is
    /// not strictly positive (zero divides, negative cools).
    pub fn try_new(
        taylor_quinney: Fix128,
        volumetric_heat_capacity_mpa_per_k: Fix128,
    ) -> Result<Self, FemError> {
        if taylor_quinney.is_negative() || taylor_quinney > Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "taylor_quinney must be a fraction in [0, 1]",
            ));
        }
        if volumetric_heat_capacity_mpa_per_k <= Fix128::ZERO {
            return Err(FemError::InvalidConfig(
                "volumetric_heat_capacity_mpa_per_k must be positive (MPa/K, not J/(kg K))",
            ));
        }
        Ok(Self {
            taylor_quinney,
            volumetric_heat_capacity_mpa_per_k,
        })
    }

    /// Fraction of plastic work released as heat.
    #[must_use]
    pub const fn taylor_quinney(&self) -> Fix128 {
        self.taylor_quinney
    }

    /// Volumetric heat capacity in MPa/K.
    #[must_use]
    pub const fn volumetric_heat_capacity_mpa_per_k(&self) -> Fix128 {
        self.volumetric_heat_capacity_mpa_per_k
    }

    /// `ΔT = β · W_p / c_v` for one element's dissipation.
    #[must_use]
    pub fn temperature_rise(&self, dissipation: Fix128) -> Fix128 {
        self.taylor_quinney * dissipation / self.volumetric_heat_capacity_mpa_per_k
    }
}

/// Temperature rise of every element from the plastic work it dissipated.
///
/// One entry per element, in mesh order, in kelvin above whatever the solve
/// treated as the stress-free reference. Elements that stayed elastic report
/// exactly zero.
///
/// This is the quantity that closes the thermo-mechanical loop:
/// [`solve_elastoplastic`] dissipates, this converts, [`deposit_plastic_heat`]
/// puts it on a grid, and [`solve_with_eigenstrain`] reads it back as strain.
#[must_use = "the temperature rise is the whole output of the conversion; \
              dropping it leaves the dissipated heat unaccounted for"]
pub fn plastic_temperature_rise(
    solution: &ElastoplasticSolution,
    heating: &PlasticHeating,
) -> Vec<Fix128> {
    solution
        .dissipation
        .iter()
        .map(|&w| heating.temperature_rise(w))
        .collect()
}

/// Deposit the plastic heat of a completed solve onto a scalar field.
///
/// Each element's temperature rise is splatted at its centroid with the
/// trilinear weights of [`CoupledField::splat`], scaled by `V_e / V_cell` so
/// that the **energy** is what the deposit conserves, and then divided by the
/// node's finite-volume dual weight:
///
/// ```text
/// Σ_node 2⁻ᵇ · ΔT_node · c_v · V_cell  =  Σ_e β · W_p,e · V_e
/// ```
///
/// Scaling by `V_e / V_cell` matters because a temperature is intensive:
/// splatting `ΔT_e` directly would give a sliver tetrahedron and a fat one the
/// same weight, and the total would depend on how the mesh was cut rather than
/// on how much was dissipated.
///
/// # ⚠️ `2⁻ᵇ` is the grid's weight, not a choice
///
/// [`CoupledField`] is **node centred**: nodes sit on `min` and `max` and the
/// spacing is `(max − min) / (n − 1)`, so the material the field represents is
/// exactly `[min, max]` and a node on a face owns **half** a cell, an edge node
/// a quarter, a corner node an eighth. `b` is the number of axes on which the
/// node sits at an end, so the dual weight is `2⁻ᵇ`, and the deposit multiplies
/// each node by `2ᵇ` to put the right amount of energy into that smaller
/// volume.
///
/// Getting this wrong is not a rounding question. [`CoupledField::diffuse`]
/// uses a **mirror** ghost node (`T₋₁ = T₁`, zero gradient at the node), and
/// such a scheme conserves `Σ 2⁻ᵇ T` **exactly** — measured to drift by zero
/// over six steps while the plain sum `Σ T` moves by −12.7 % for heat on a face
/// node and +22.0 % for heat in the interior. A deposit balanced on the plain
/// sum would therefore disagree with the operator the field is stepped with,
/// and a body flush with the grid boundary would lose up to a factor of eight.
///
/// ⚠️ The mirror ghost is load-bearing here: `tests/analytic_coupled_field.rs`
/// pins `cos(k x)` as an exact eigenmode of the discrete operator, which only a
/// mirror ghost gives. Replacing it with a copy ghost (`T₋₁ = T₀`) would move
/// the conserved quantity to the plain sum and invalidate this weighting.
///
/// ⚠️ A body that does not reach a boundary node is unaffected: every weight
/// that receives anything is then `1`, so the correction is the identity and
/// the plain-sum identity holds as well.
///
/// The field is **added to**, not overwritten, so a caller may accumulate
/// several bodies or several steps before diffusing.
///
/// # ⚠️ This closes the loop, and a closed loop is not a converged one
///
/// Until this function existed the thermo-mechanical coupling ran one way:
/// a temperature field drove [`solve_with_eigenstrain`] and nothing came
/// back. A one-way coupling is **exact in a single sweep**, so the question of
/// whether to solve it partitioned or monolithically did not arise.
///
/// Depositing the dissipation back onto the field the eigenstrain reads makes
/// the coupling **two-way**, and the obvious call sequence — solve, deposit,
/// solve again — is an explicit partitioned scheme with **zero
/// sub-iterations**. That is the standard staggered thermo-mechanical scheme
/// and is usually what is wanted, because the thermal feedback on a metal is
/// weak (`α ΔT` against `ε̄_p`). It is **not** a converged solution of the
/// coupled system, and nothing here checks that it is close to one.
///
/// A caller who needs the converged answer drives the sweep under
/// [`crate::coupled_iteration::run_sub_iteration`], which measures the
/// contraction ratio of the splitting and refuses to call a diverging
/// iteration converged. Its module documentation carries the measurements
/// behind that: a partitioned scheme sub-iterated to convergence solves the
/// same equations as a monolithic one, so the remaining difference is rate and
/// robustness, not correctness.
///
/// # Errors
///
/// [`FemError::EmptyMesh`] for a mesh with no vertices or no tetrahedra,
/// [`FemError::SolutionElementCountDoesNotMatchMesh`] if the solution reports a
/// different number of elements than the mesh has tetrahedra,
/// [`FemError::DepositGridHasDegenerateAxis`] for a field with a single node on
/// an axis, and [`FemError::DegenerateElement`] for a tetrahedron with no
/// volume.
pub fn deposit_plastic_heat(
    mesh: &SdfTetMesh,
    solution: &ElastoplasticSolution,
    heating: &PlasticHeating,
    field: &mut CoupledField,
) -> Result<(), FemError> {
    deposit_increment_heat(mesh, &solution.dissipation, heating, field)
}

/// Deposit the plastic heat of **one increment** onto a scalar field.
///
/// Same splat, same `V_e / V_cell` scaling and same dual-weight correction as
/// [`deposit_plastic_heat`]; the difference is only where the work comes from.
/// [`deposit_plastic_heat`] reads a whole load path's accumulated dissipation,
/// this reads [`ElastoplasticIncrement::plastic_work_increment`] — the work one
/// increment added.
///
/// ⚠️ **This is the form a sub-iteration needs.** A coupled sweep solves a
/// fixed-point problem in the temperature *increment* `δT`, so the heat it
/// deposits has to be the increment's, not the path's; feeding it the
/// accumulated dissipation would make each sweep deposit everything the body
/// has ever dissipated and the iteration would not be a map on `δT` at all.
///
/// ⚠️ **It adds to `field` rather than replacing it**, exactly as
/// [`deposit_plastic_heat`] does. A driver that calls this once per sweep must
/// therefore deposit into a field it has [`CoupledField::clear`]ed, or the
/// accumulation turns the fixed-point map into a time integration — which
/// cannot converge, and whose monotone growth is indistinguishable from
/// genuine divergence.
///
/// # Errors
///
/// [`FemError::EmptyMesh`] for a mesh with no vertices or no tetrahedra,
/// [`FemError::SolutionElementCountDoesNotMatchMesh`] if `plastic_work` has a
/// different length than the mesh has tetrahedra,
/// [`FemError::DepositGridHasDegenerateAxis`] for a field with a single node on
/// an axis, and [`FemError::DegenerateElement`] for a tetrahedron with no
/// volume.
pub fn deposit_increment_heat(
    mesh: &SdfTetMesh,
    plastic_work: &[Fix128],
    heating: &PlasticHeating,
    field: &mut CoupledField,
) -> Result<(), FemError> {
    if mesh.vertices.is_empty() || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    if plastic_work.len() != mesh.tets.len() {
        return Err(FemError::SolutionElementCountDoesNotMatchMesh {
            elements: plastic_work.len(),
            tet_count: mesh.tets.len(),
        });
    }
    let (nx, ny, nz) = (field.nx(), field.ny(), field.nz());
    for (count, axis) in [(nx, Axis::X), (ny, Axis::Y), (nz, Axis::Z)] {
        if count < 2 {
            return Err(FemError::DepositGridHasDegenerateAxis { axis });
        }
    }
    let elements = build_elements(mesh)?;
    let (hx, hy, hz) = field.cell_size();
    let cell_volume = hx * hy * hz;
    let quarter = Fix128::from_raw(0, 1 << 62);

    // Splat into a zeroed copy of the grid first. The dual weight belongs to the
    // node, not to the contribution, so the correction has to be applied once
    // per node after every element has been deposited — and it must not touch
    // whatever the caller already had in `field`.
    let mut staged = field.clone();
    staged.clear();
    for (element, &work) in elements.iter().zip(plastic_work.iter()) {
        if work.is_zero() {
            continue;
        }
        let mut centroid = Vec3Fix::ZERO;
        for &node in &element.nodes {
            let v = mesh.vertices[node];
            centroid = centroid
                + Vec3Fix::new(
                    Fix128::from_f32(v[0]),
                    Fix128::from_f32(v[1]),
                    Fix128::from_f32(v[2]),
                );
        }
        centroid = centroid * quarter;
        let rise = heating.temperature_rise(work) * element.volume / cell_volume;
        staged.splat(centroid, rise);
    }

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let staged_value = staged.get(ix, iy, iz);
                if staged_value.is_zero() {
                    continue;
                }
                // `b` counts only axes that have more than one node; the guard
                // above has already refused a degenerate one, so every axis
                // here is a real one and both ends are distinct nodes.
                let mut boundary_axes = 0u32;
                for (i, n) in [(ix, nx), (iy, ny), (iz, nz)] {
                    if i == 0 || i == n - 1 {
                        boundary_axes += 1;
                    }
                }
                let inverse_dual_weight = Fix128::from_int(1 << boundary_axes);
                let corrected = staged_value * inverse_dual_weight;
                field.set(ix, iy, iz, field.get(ix, iy, iz) + corrected);
            }
        }
    }
    Ok(())
}

/// Plastic state of one element: the plastic strain in Voigt order with
/// **engineering** shear (`xx, yy, zz, γxy, γyz, γzx`, the order the strain is
/// read in) and the equivalent plastic strain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlasticState {
    strain: [Fix128; 6],
    equivalent: Fix128,
    /// Accumulated plastic work `W_p = Σ σ^{n+1} : Δε_p` (MPa, i.e. MJ/m³).
    ///
    /// Carried in the state rather than recomputed at the end because it is a
    /// **path** integral: the end state does not determine it. See
    /// [`ElastoplasticSolution::dissipation`].
    dissipation: Fix128,
}

impl PlasticState {
    const VIRGIN: Self = Self {
        strain: [Fix128::ZERO; 6],
        equivalent: Fix128::ZERO,
        dissipation: Fix128::ZERO,
    };
}

/// The consistent tangent of an element that returned to the yield surface:
/// `C = K 1⊗1 + 2μθ I_dev − 2μθ̄ n⊗n`, stored as `1 − θ`, `θ̄` and `n`.
#[derive(Clone, Copy)]
struct PlasticTangent {
    shear_loss: Fix128,
    theta_bar: Fix128,
    direction: StressTensor,
}

/// One element's return mapping: the stress, the state it would commit, and the
/// tangent if the element yielded.
#[derive(Clone, Copy)]
struct ReturnMap {
    stress: StressTensor,
    state: PlasticState,
    tangent: Option<PlasticTangent>,
}

/// Strain of an element in Voigt order with engineering shear, accumulated in
/// the same order as `element_stress_local` so an elastic element reproduces
/// [`solve`] to the bit.
fn element_strain(element: &Element, u: &[[Fix128; 3]; 4]) -> [Fix128; 6] {
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
    [exx, eyy, ezz, gxy, gyz, gzx]
}

/// Hooke's law on a Voigt strain with engineering shear.
fn hooke(e: [Fix128; 6], lambda: Fix128, mu: Fix128) -> StressTensor {
    let trace = e[0] + e[1] + e[2];
    let two_mu = mu + mu;
    StressTensor {
        xx: lambda * trace + two_mu * e[0],
        yy: lambda * trace + two_mu * e[1],
        zz: lambda * trace + two_mu * e[2],
        xy: mu * e[3],
        yz: mu * e[4],
        zx: mu * e[5],
    }
}

/// Radial return for J2 plasticity with linear isotropic hardening.
///
/// With `e = ε − ε_p`, the trial stress is `σ_tr = C e` and the trial von Mises
/// stress is `q`. Yield is `f = q − (σ_y + H ε̄_p) > 0`. The return is along the
/// deviator: `Δε̄ = f / (3μ + H)`, `σ = σ_tr − 3μ Δε̄ · s_tr / q`,
/// `Δε_p = (3/2) Δε̄ · s_tr / q`, `ε̄_p ← ε̄_p + Δε̄`. The pressure is untouched.
fn return_map(
    strain: [Fix128; 6],
    state: &PlasticState,
    lambda: Fix128,
    mu: Fix128,
    yield_stress: Fix128,
    hardening: Fix128,
) -> ReturnMap {
    let mut e = strain;
    for (value, plastic) in e.iter_mut().zip(state.strain.iter()) {
        *value = *value - *plastic;
    }
    let trial = hooke(e, lambda, mu);
    let q = trial.von_mises();
    let f = q - (yield_stress + hardening * state.equivalent);
    if f <= Fix128::ZERO {
        return ReturnMap {
            stress: trial,
            state: *state,
            tangent: None,
        };
    }
    let three_mu = Fix128::from_int(3) * mu;
    let d_eq = f / (three_mu + hardening);
    let pressure = (trial.xx + trial.yy + trial.zz) / Fix128::from_int(3);
    let s = StressTensor {
        xx: trial.xx - pressure,
        yy: trial.yy - pressure,
        zz: trial.zz - pressure,
        ..trial
    };
    let ratio = three_mu * d_eq / q;
    let stress = StressTensor {
        xx: trial.xx - ratio * s.xx,
        yy: trial.yy - ratio * s.yy,
        zz: trial.zz - ratio * s.zz,
        xy: trial.xy - ratio * s.xy,
        yz: trial.yz - ratio * s.yz,
        zx: trial.zx - ratio * s.zx,
    };
    // Δε_p = (3/2) Δε̄ s/q on the normal components; the engineering shear is
    // twice the tensor shear, so 3 Δε̄ s/q.
    let flow = d_eq / q;
    let normal = Fix128::from_raw(1, 1 << 63) * flow; // 3/2
    let shear = Fix128::from_int(3) * flow;
    let mut next = *state;
    next.strain[0] = next.strain[0] + normal * s.xx;
    next.strain[1] = next.strain[1] + normal * s.yy;
    next.strain[2] = next.strain[2] + normal * s.zz;
    next.strain[3] = next.strain[3] + shear * s.xy;
    next.strain[4] = next.strain[4] + shear * s.yz;
    next.strain[5] = next.strain[5] + shear * s.zx;
    next.equivalent = next.equivalent + d_eq;
    // Plastic work of this step. Writing `σ^{n+1} : Δε_p` out with
    // `Δε_p = (3/2)Δε̄ s/q` and `s:s = (2/3)q²` collapses it to
    // `Δε̄ (q − 3μΔε̄)`, and consistency (`Δε̄ = f/(3μ+H)`) makes that factor
    // exactly the radius of the surface the return lands on:
    //
    //     q^{n+1} = q − 3μΔε̄ = σ_y + H·ε̄^{n+1}
    //
    // ⚠️ The right-hand form is used, not `q − 3μ·d_eq`. They agree in exact
    // arithmetic, but under `Fix128` the left form carries the truncation of
    // `f/(3μ+H)` multiplied back up by `3μ`, which is ~1e3 ulp per step on a
    // steel-like shear modulus, while the right form is a single multiply off
    // the yield parameters. The oracle
    // `tests/analytic_plastic_dissipation.rs::the_committed_stress_sits_on_the_
    // yield_surface` holds the two together so this choice cannot drift into a
    // dissipation that no longer matches the stress actually reported.
    next.dissipation = next.dissipation + d_eq * (yield_stress + hardening * next.equivalent);
    // `n = s / ‖s‖_F` with the Frobenius norm, which counts the shear twice.
    let frobenius = (s.xx * s.xx
        + s.yy * s.yy
        + s.zz * s.zz
        + Fix128::from_int(2) * (s.xy * s.xy + s.yz * s.yz + s.zx * s.zx))
        .sqrt();
    let direction = StressTensor {
        xx: s.xx / frobenius,
        yy: s.yy / frobenius,
        zz: s.zz / frobenius,
        xy: s.xy / frobenius,
        yz: s.yz / frobenius,
        zx: s.zx / frobenius,
    };
    ReturnMap {
        stress,
        state: next,
        tangent: Some(PlasticTangent {
            shear_loss: ratio,
            theta_bar: three_mu / (three_mu + hardening) - ratio,
            direction,
        }),
    }
}

/// `V·Bᵀ (C_ct ε)` — the consistent tangent of one element applied to a
/// displacement `p`: Hooke's law for an element that stayed elastic, and
/// `C ε − 2μ(1−θ) dev(ε) − 2μθ̄ (n:ε) n` for one that returned.
fn element_tangent_force(
    element: &Element,
    p: &[[Fix128; 3]; 4],
    tangent: Option<&PlasticTangent>,
    lambda: Fix128,
    mu: Fix128,
) -> [[Fix128; 3]; 4] {
    let Some(t) = tangent else {
        return element_force_local(element, p, lambda, mu);
    };
    let e = element_strain(element, p);
    let elastic = hooke(e, lambda, mu);
    let two_mu = mu + mu;
    let mean = (e[0] + e[1] + e[2]) / Fix128::from_int(3);
    let n = t.direction;
    // n : ε, with the engineering shear standing in for 2·ε_xy
    let projection =
        n.xx * e[0] + n.yy * e[1] + n.zz * e[2] + n.xy * e[3] + n.yz * e[4] + n.zx * e[5];
    let along = two_mu * t.theta_bar * projection;
    let loss = two_mu * t.shear_loss;
    let stress = StressTensor {
        xx: elastic.xx - loss * (e[0] - mean) - along * n.xx,
        yy: elastic.yy - loss * (e[1] - mean) - along * n.yy,
        zz: elastic.zz - loss * (e[2] - mean) - along * n.zz,
        xy: elastic.xy - mu * t.shear_loss * e[3] - along * n.xy,
        yz: elastic.yz - mu * t.shear_loss * e[4] - along * n.yz,
        zx: elastic.zx - mu * t.shear_loss * e[5] - along * n.zx,
    };
    element_force_from_stress(element, stress)
}

/// Quasi-static small-strain J2 elastoplastic solve on P1 tetrahedra, driven
/// along a load path.
///
/// `load_path` lists load factors `t₁, t₂, …`; at step `k` the prescribed
/// displacements and the nodal loads of `boundary` are scaled by `t_k` and the
/// equilibrium `f_int(u) = t_k f_ext` is solved by Newton's method, starting
/// from the previous step's field. The plastic state (`ε_p`, `ε̄_p` per element)
/// carries from step to step, which is what makes the path matter: a path that
/// goes up and comes back down unloads elastically and leaves a residual
/// strain. Factors may be negative or decrease. `[1.0]` is a single step.
///
/// # Method
///
/// The stress of each element is the radial return of the elastic trial stress
/// (see `return_map`), so the internal force is `Σ V Bᵀσ` as in [`solve`]. The
/// Newton tangent is the **consistent** (algorithmic) tangent of that return,
/// not the elastic stiffness, so convergence is quadratic once the plastic set
/// settles. Each linear system is solved by the conjugate gradient of [`solve`],
/// preconditioned with the elastic stiffness diagonal.
///
/// An element that never yields passes through the same routines as [`solve`];
/// a path on which nothing yields returns [`solve`]'s displacements and
/// stresses bit for bit.
///
/// # Scope
///
/// Small strain and small displacement (no geometric nonlinearity), P1 only,
/// bilinear isotropic hardening only, quasi-static (no rate or inertia). Not
/// supported: softening (`H < 0` is rejected), kinematic hardening, finite-strain
/// `F = Fe·Fp`, and the P2 / P3 elements.
///
/// # Determinism
///
/// Every operation is [`Fix128`] arithmetic in mesh order with bounded loops.
///
/// # Errors
///
/// As [`solve`], plus [`FemError::InvalidConfig`] for an empty `load_path` or a
/// factor of magnitude above `2²⁰`, and [`FemError::NotConverged`] when the
/// Newton budget runs out. A load above the limit load of a perfectly plastic
/// body has no equilibrium state and comes back as an `Err` from the linear
/// solve or the Newton budget, never as a stress above yield.
pub fn solve_elastoplastic(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary: &BoundaryConditions,
    config: &ElastoplasticConfig,
    load_path: &[Fix128],
) -> Result<ElastoplasticSolution, FemError> {
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
    if load_path.is_empty() {
        return Err(FemError::InvalidConfig("load_path must not be empty"));
    }
    let factor_cap = Fix128::from_int(LOAD_FACTOR_MAX);
    if load_path.iter().any(|t| t.abs() > factor_cap) {
        return Err(FemError::InvalidConfig(
            "load factors must have magnitude at most 2^20",
        ));
    }

    let problem = ElastoplasticProblem::try_new(mesh, material, boundary, config)?;
    let mut state = problem.virgin_state();
    for &factor in load_path {
        let increment = problem.step(&state, &ElastoplasticIncrementRequest::new(factor))?;
        increment.commit(&mut state);
    }
    Ok(problem.finish(&state, load_path.len()))
}

// ============================================================================
// Per-increment entry point
// ============================================================================

/// How a material's yield surface shrinks as it heats.
///
/// `σ_y(T) = σ_y₀ · max(0, 1 − w_y · ΔT)` and `H(T) = H₀ · max(0, 1 − w_h · ΔT)`,
/// with `ΔT` the rise above the stress-free reference the temperature field
/// already carries. The clamp at zero is what keeps a hot element from being
/// handed a negative yield radius, which the return mapping has no meaning
/// for.
///
/// # Why this and not thermal expansion
///
/// Thermal expansion enters the plastic problem as the purely volumetric
/// eigenstrain `ε_th = α ΔT I`, and J2 yielding reads only the deviator, so
/// expansion alone cannot move the yield surface: it changes the pressure and
/// — through equilibrium in a constrained body — the total strain, but never
/// the yield criterion directly. Softening does, which is why a thermoplastic
/// coupling that leaves it out stays weak however hot the body gets. Measured
/// on a steel bar: `α ΔT` of `2.7e-6` against `ε̄_p` of `3.7e-3`, a ratio of
/// `7.4e-4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ThermalSoftening {
    yield_per_k: Fix128,
    hardening_per_k: Fix128,
}

impl ThermalSoftening {
    /// Softening fractions per kelvin, as `w_y` and `w_h` above.
    ///
    /// Zero for both is the no-softening law, and gives the same answer as
    /// passing no law at all — `None` and `Some(ThermalSoftening::none())` are
    /// the same solve, which `thermal_softening_off_matches_no_law_at_all`
    /// pins.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] for a negative fraction (a material that
    /// *hardens* with temperature is a different law, not a negative softening)
    /// or one above `2³⁰` per kelvin, the cap the other plastic parameters use.
    pub fn try_new(yield_per_k: Fix128, hardening_per_k: Fix128) -> Result<Self, FemError> {
        let cap = Fix128::from_int(PLASTIC_PARAMETER_MAX);
        if yield_per_k.is_negative() || hardening_per_k.is_negative() {
            return Err(FemError::InvalidConfig(
                "thermal softening fractions must not be negative",
            ));
        }
        if yield_per_k > cap || hardening_per_k > cap {
            return Err(FemError::InvalidConfig(
                "thermal softening fractions must be at most 2^30 per kelvin",
            ));
        }
        Ok(Self {
            yield_per_k,
            hardening_per_k,
        })
    }

    /// The law that softens nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            yield_per_k: Fix128::ZERO,
            hardening_per_k: Fix128::ZERO,
        }
    }

    /// Fraction of the yield stress lost per kelvin.
    #[must_use]
    pub const fn yield_per_k(&self) -> Fix128 {
        self.yield_per_k
    }

    /// Fraction of the hardening modulus lost per kelvin.
    #[must_use]
    pub const fn hardening_per_k(&self) -> Fix128 {
        self.hardening_per_k
    }

    /// `σ_y(T)` and `H(T)` at a temperature rise of `delta_t`.
    fn at(&self, delta_t: Fix128, yield_stress: Fix128, hardening: Fix128) -> (Fix128, Fix128) {
        let shrink = |fraction: Fix128, value: Fix128| -> Fix128 {
            let scale = Fix128::ONE - fraction * delta_t;
            if scale.is_negative() {
                Fix128::ZERO
            } else {
                value * scale
            }
        };
        (
            shrink(self.yield_per_k, yield_stress),
            shrink(self.hardening_per_k, hardening),
        )
    }
}

/// One load increment's request.
///
/// A struct rather than a bare argument so that an input added later is a new
/// `with_*` method and not a change to [`ElastoplasticProblem::step`]'s arity.
/// Adding a parameter to a public function is a breaking change even when the
/// struct it configures is `#[non_exhaustive]`, because `#[non_exhaustive]`
/// only stops literal construction.
/// Not `PartialEq`: the request borrows a temperature grid, and two requests
/// that point at different grids holding the same numbers are not the same
/// request in any sense a caller would want. `ElastoplasticIncrement` — the
/// answer — is comparable, which is what the oracles need.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ElastoplasticIncrementRequest<'a> {
    factor: Fix128,
    thermal: Option<ThermalExpansion<'a>>,
    softening: Option<ThermalSoftening>,
}

impl<'a> ElastoplasticIncrementRequest<'a> {
    /// Scale the prescribed displacements and external loads by `factor`.
    ///
    /// The factor is absolute, not incremental: a path of `1/8, 2/8, …, 1` is
    /// eight equal increments reaching the full load, exactly the slice
    /// [`solve_elastoplastic`] takes.
    ///
    /// No temperature: the increment is isothermal, which is the behaviour
    /// [`solve_elastoplastic`] has always had.
    #[must_use]
    pub const fn new(factor: Fix128) -> Self {
        Self {
            factor,
            thermal: None,
            softening: None,
        }
    }

    /// Let the increment read a temperature field.
    ///
    /// `thermal` supplies the eigenstrain `ε_th = α ΔT I`, which is removed
    /// from the elastic trial strain before the return mapping — not added as a
    /// nodal load afterwards. The two are the same for a linear elastic solve
    /// and are **not** the same here: the yield function has to see the
    /// thermally corrected stress, or a hot element yields at the cold
    /// criterion.
    ///
    /// `softening` is the optional `σ_y(T)` / `H(T)` law. ⚠️ `None` means **no
    /// softening**, not "a default law": a caller who wants temperature to
    /// weaken the material has to say so, and
    /// `thermal_softening_off_matches_no_law_at_all` pins that `None` and
    /// [`ThermalSoftening::none`] agree to the bit.
    #[must_use]
    pub const fn with_thermal(
        mut self,
        thermal: ThermalExpansion<'a>,
        softening: Option<ThermalSoftening>,
    ) -> Self {
        self.thermal = Some(thermal);
        self.softening = softening;
        self
    }

    /// The load factor this increment applies.
    #[must_use]
    pub const fn factor(&self) -> Fix128 {
        self.factor
    }

    /// The temperature field this increment reads, if any.
    #[must_use]
    pub const fn thermal(&self) -> Option<ThermalExpansion<'a>> {
        self.thermal
    }

    /// The softening law this increment applies, if any.
    #[must_use]
    pub const fn softening(&self) -> Option<ThermalSoftening> {
        self.softening
    }
}

/// An elastoplastic body prepared for stepping, with the mesh-dependent work
/// done once.
///
/// # Why this exists
///
/// [`solve_elastoplastic`] walks a whole load path and hands back the end of
/// it. A coupled solve cannot use that: a thermo-mechanical iteration has to
/// run **the same increment** several times from the same committed state,
/// compare what comes back, and only then commit. That needs three things the
/// single call does not expose — a state it can hold, an increment it can
/// repeat, and a commit it controls.
///
/// The element gradients, the stiffness diagonal and the preconditioner depend
/// only on the mesh, the material and the solver configuration, so they are
/// built here and reused by every [`Self::step`]. [`solve_elastoplastic`] is
/// implemented on top of this type, which is what keeps the two from drifting:
/// there is one Newton loop in this module, not two.
#[derive(Debug, Clone)]
pub struct ElastoplasticProblem {
    elements: Vec<Element>,
    lambda: Fix128,
    mu: Fix128,
    ndof: usize,
    vertex_count: usize,
    prescribed_value: Vec<Fix128>,
    is_free: Vec<bool>,
    f_ext: Vec<Fix128>,
    precond: Vec<Fix128>,
    config: ElastoplasticConfig,
    /// Mesh vertices in `Fix128`, for the temperature-coverage check.
    node_positions: Vec<Vec3Fix>,
    /// Element centroids, where a temperature field is sampled.
    centroids: Vec<Vec3Fix>,
}

/// Where an elastoplastic solve has got to: the committed plastic state of
/// every element and the displacement it was reached at.
///
/// Built by [`ElastoplasticProblem::virgin_state`] and advanced by
/// [`ElastoplasticIncrement::commit`]; there is no way to construct one from
/// outside the crate, because a state whose plastic history did not come from
/// a solve would be read as one that did.
///
/// The displacement is part of the state because the Newton iteration of an
/// increment starts from the previous increment's answer; starting from zero
/// instead would converge to the same place but count different iterations.
/// The last field and its solver bookkeeping are carried for the same reason
/// [`FemSolution`] carries them — they describe the last linear solve, and an
/// increment that converges without one leaves the previous increment's
/// numbers standing.
#[derive(Debug, Clone, PartialEq)]
pub struct ElastoplasticState {
    committed: Vec<PlasticState>,
    u: Vec<Fix128>,
    field: FemSolution,
    newton_total: u32,
}

impl ElastoplasticState {
    /// Accumulated equivalent plastic strain `ε̄_p` of every element.
    #[must_use]
    pub fn equivalent_plastic_strain(&self) -> Vec<Fix128> {
        self.committed.iter().map(|s| s.equivalent).collect()
    }

    /// Accumulated plastic work `W_p` of every element, in `MPa` (`mJ/mm³`).
    ///
    /// This is the path integral `∫ σ : dε_p`, so it depends on how the load
    /// path was split; see [`ElastoplasticSolution::dissipation`].
    #[must_use]
    pub fn dissipation(&self) -> Vec<Fix128> {
        self.committed.iter().map(|s| s.dissipation).collect()
    }

    /// Nodal displacements reached so far.
    #[must_use]
    pub fn displacements(&self) -> Vec<[Fix128; 3]> {
        self.field.displacements.clone()
    }

    /// Newton iterations (linear solves) spent since the state was virgin.
    #[must_use]
    pub const fn newton_iterations(&self) -> u32 {
        self.newton_total
    }
}

/// One increment solved but **not** committed.
///
/// Produced by [`ElastoplasticProblem::step`] only; the crate builds it and
/// the caller reads it.
///
/// Holding the trial state back is the whole point: a coupled iteration solves
/// this increment, reads [`Self::plastic_work_increment`], updates the
/// temperature, and solves the increment **again from the same committed
/// state**. Committing inside that loop would accumulate the plastic history
/// once per sweep, which turns the fixed-point map into a time integration, so
/// it stops converging by construction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElastoplasticIncrement {
    /// Displacements and element stress at the end of this increment.
    pub field: FemSolution,
    /// Plastic work **this increment** added to each element, `ΔW_p` in `MPa`.
    ///
    /// Non-negative: the equivalent plastic strain is monotonic and the stress
    /// it multiplies is a yield radius. This is the quantity a thermal solve
    /// wants — depositing the total `W_p` would reheat the whole history on
    /// every sweep.
    pub plastic_work_increment: Vec<Fix128>,
    trial: Vec<PlasticState>,
    u: Vec<Fix128>,
    newton_iterations: u32,
}

impl ElastoplasticIncrement {
    /// Make this increment the new committed state.
    ///
    /// Call this once per increment, **after** any coupled iteration over it
    /// has converged.
    pub fn commit(self, state: &mut ElastoplasticState) {
        state.committed = self.trial;
        state.u = self.u;
        state.field = self.field;
        state.newton_total = state.newton_total.saturating_add(self.newton_iterations);
    }

    /// Newton iterations (linear solves) this increment took.
    #[must_use]
    pub const fn newton_iterations(&self) -> u32 {
        self.newton_iterations
    }
}

impl ElastoplasticProblem {
    /// Prepare a body for stepping.
    ///
    /// # Errors
    ///
    /// The checks [`solve_elastoplastic`] makes on the mesh and the boundary
    /// conditions: [`FemError::EmptyMesh`], [`FemError::VertexOutOfRange`],
    /// [`FemError::UnderConstrained`], [`FemError::DegenerateElement`], and
    /// [`FemError::InvalidConfig`] from the preconditioner.
    pub fn try_new(
        mesh: &SdfTetMesh,
        material: &ElasticMaterial,
        boundary: &BoundaryConditions,
        config: &ElastoplasticConfig,
    ) -> Result<Self, FemError> {
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
        check_element_scale(&elements, lambda, mu)?;
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
        let diag = stiffness_diagonal(&elements, lambda, mu, ndof);
        let precond = build_preconditioner(&diag, &is_free, &config.linear)?;

        let node_positions: Vec<Vec3Fix> = mesh
            .vertices
            .iter()
            .map(|q| {
                Vec3Fix::new(
                    Fix128::from_f32(q[0]),
                    Fix128::from_f32(q[1]),
                    Fix128::from_f32(q[2]),
                )
            })
            .collect();
        let centroids: Vec<Vec3Fix> = elements
            .iter()
            .map(|element| {
                let mut sum = Vec3Fix::ZERO;
                for &node in &element.nodes {
                    sum = sum + node_positions[node];
                }
                sum * quarter()
            })
            .collect();

        Ok(Self {
            elements,
            lambda,
            mu,
            ndof,
            vertex_count,
            prescribed_value,
            is_free,
            f_ext,
            precond,
            config: *config,
            node_positions,
            centroids,
        })
    }

    /// The unloaded state: no displacement, no plastic strain, no work.
    #[must_use]
    pub fn virgin_state(&self) -> ElastoplasticState {
        ElastoplasticState {
            committed: vec![PlasticState::VIRGIN; self.elements.len()],
            u: vec![Fix128::ZERO; self.ndof],
            field: FemSolution {
                displacements: vec![[Fix128::ZERO; 3]; self.vertex_count],
                element_stress: Vec::new(),
                iterations: 0,
                relative_residual: Fix128::ZERO,
                effective_relative_tolerance: Fix128::ZERO,
            },
            newton_total: 0,
        }
    }

    /// Solve one load increment from `state` without committing it.
    ///
    /// # Errors
    ///
    /// [`FemError::NotConverged`] if the Newton loop runs out of iterations,
    /// and whatever the conjugate gradient returns.
    pub fn step(
        &self,
        state: &ElastoplasticState,
        request: &ElastoplasticIncrementRequest,
    ) -> Result<ElastoplasticIncrement, FemError> {
        let config = &self.config;
        let (lambda, mu) = (self.lambda, self.mu);
        let sigma_y = config.yield_stress_mpa;
        let hardening = config.hardening_modulus_mpa;
        let ndof = self.ndof;

        // Per element, once: the temperature rise at its centroid, the
        // eigenstrain that rise imposes, and the yield parameters it leaves.
        // Sampling at the centroid is exact for the uniform field the oracles
        // use and first order otherwise — the order of the P1 constant-strain
        // element it feeds, so it adds no error of its own kind, which is the
        // same argument `thermal_stresses` makes for the elastic path.
        let thermal: Option<Vec<(Fix128, Fix128, Fix128)>> = match request.thermal {
            None => None,
            Some(field) => {
                for element in &self.elements {
                    for &node in &element.nodes {
                        if !field.field.contains(self.node_positions[node]) {
                            return Err(FemError::TemperatureFieldDoesNotCoverMesh {
                                vertex: u32::try_from(node).unwrap_or(u32::MAX),
                            });
                        }
                    }
                }
                let law = request.softening.unwrap_or_else(ThermalSoftening::none);
                Some(
                    self.centroids
                        .iter()
                        .map(|&centroid| {
                            let delta_t = field.field.sample(centroid);
                            let (y, h) = law.at(delta_t, sigma_y, hardening);
                            (field.alpha_per_k * delta_t, y, h)
                        })
                        .collect(),
                )
            }
        };

        let mut u = state.u.clone();
        let mut load = vec![Fix128::ZERO; ndof];
        let mut residual = vec![Fix128::ZERO; ndof];
        let mut maps: Vec<ReturnMap> = Vec::new();
        let mut cg_iterations = state.field.iterations;
        let mut relative_residual = state.field.relative_residual;
        let mut effective_relative_tolerance = state.field.effective_relative_tolerance;
        let mut newton_iterations = 0u32;

        for d in 0..ndof {
            if self.is_free[d] {
                load[d] = request.factor() * self.f_ext[d];
            } else {
                u[d] = request.factor() * self.prescribed_value[d];
            }
        }
        let load_norm = max_abs(&load);
        let mut reference_norm = Fix128::ZERO;
        let mut iteration = 0u32;
        loop {
            // The return map of every element at the current iterate, from the
            // state committed at the end of the previous increment.
            maps.clear();
            residual.copy_from_slice(&load);
            for (e, (element, committed)) in
                self.elements.iter().zip(state.committed.iter()).enumerate()
            {
                let mut strain = element_strain(element, &gather(element, &u));
                let (yield_here, hardening_here) = match &thermal {
                    None => (sigma_y, hardening),
                    Some(per_element) => {
                        let (eigen, y, h) = per_element[e];
                        // `ε_th = α ΔT I`: the three normal components only, so
                        // the deviator — and therefore the yield criterion —
                        // moves only through what equilibrium does with it.
                        for normal in strain.iter_mut().take(3) {
                            *normal = *normal - eigen;
                        }
                        (y, h)
                    }
                };
                let map = return_map(strain, committed, lambda, mu, yield_here, hardening_here);
                let force = element_force_from_stress(element, map.stress);
                for (f, &node) in force.iter().zip(element.nodes.iter()) {
                    let base = node * 3;
                    for axis in 0..3 {
                        if self.is_free[base + axis] {
                            residual[base + axis] = residual[base + axis] - f[axis];
                        }
                    }
                }
                maps.push(map);
            }
            for (d, value) in residual.iter_mut().enumerate() {
                if !self.is_free[d] {
                    *value = Fix128::ZERO;
                }
            }
            let residual_max = max_abs(&residual);
            if iteration == 0 {
                reference_norm = if load_norm > residual_max {
                    load_norm
                } else {
                    residual_max
                };
            }
            let requested = reference_norm * config.newton_tolerance;
            let target = if requested > RESIDUAL_NORM_FLOOR {
                requested
            } else {
                RESIDUAL_NORM_FLOOR
            };
            if residual_max <= target {
                break;
            }
            if iteration >= config.newton_iterations {
                return Err(FemError::NotConverged {
                    iterations: iteration,
                    relative_residual: relative(residual_max, reference_norm),
                });
            }

            let cg = conjugate_gradient(
                &residual,
                &self.is_free,
                &self.precond,
                &config.linear,
                |p, out| {
                    out.fill(Fix128::ZERO);
                    for (element, map) in self.elements.iter().zip(maps.iter()) {
                        let force = element_tangent_force(
                            element,
                            &gather(element, p),
                            map.tangent.as_ref(),
                            lambda,
                            mu,
                        );
                        for (f, &node) in force.iter().zip(element.nodes.iter()) {
                            let base = node * 3;
                            out[base] = out[base] + f[0];
                            out[base + 1] = out[base + 1] + f[1];
                            out[base + 2] = out[base + 2] + f[2];
                        }
                    }
                },
            )?;
            // Indexed by the same `d` as before, written as a zip so clippy's
            // `needless_range_loop` does not fire: the order of the additions
            // is what has to stay put, and `u`, `is_free` and `cg.x` are all
            // `ndof` long.
            for ((value, free), delta) in u.iter_mut().zip(self.is_free.iter()).zip(cg.x.iter()) {
                if *free {
                    *value = *value + *delta;
                }
            }
            cg_iterations = cg.iterations;
            relative_residual = relative(cg.residual_norm, cg.b_norm);
            effective_relative_tolerance = relative(cg.target, cg.b_norm);
            iteration += 1;
            newton_iterations += 1;
        }

        let plastic_work_increment = maps
            .iter()
            .zip(state.committed.iter())
            .map(|(map, committed)| map.state.dissipation - committed.dissipation)
            .collect();
        let displacements = (0..self.vertex_count)
            .map(|v| [u[v * 3], u[v * 3 + 1], u[v * 3 + 2]])
            .collect();
        let element_stress = maps.iter().map(|m| m.stress).collect();
        Ok(ElastoplasticIncrement {
            field: FemSolution {
                displacements,
                element_stress,
                iterations: cg_iterations,
                relative_residual,
                effective_relative_tolerance,
            },
            plastic_work_increment,
            trial: maps.iter().map(|m| m.state).collect(),
            u,
            newton_iterations,
        })
    }

    /// Assemble what [`solve_elastoplastic`] reports from a committed state.
    fn finish(&self, state: &ElastoplasticState, steps: usize) -> ElastoplasticSolution {
        let plastic_strain = state
            .committed
            .iter()
            .map(|s| StressTensor {
                xx: s.strain[0],
                yy: s.strain[1],
                zz: s.strain[2],
                xy: half() * s.strain[3],
                yz: half() * s.strain[4],
                zx: half() * s.strain[5],
            })
            .collect();
        ElastoplasticSolution {
            field: state.field.clone(),
            plastic_strain,
            equivalent_plastic_strain: state.equivalent_plastic_strain(),
            dissipation: state.dissipation(),
            newton_iterations: state.newton_total,
            steps: u32::try_from(steps).unwrap_or(u32::MAX),
        }
    }
}

// ============================================================================
// Thermoplastic sub-iteration (two-way coupling driven to a fixed point)
// ============================================================================

/// How to couple the plastic solve and the temperature field, and when to stop.
///
/// The mechanical leg already reads temperature
/// ([`ElastoplasticIncrementRequest::with_thermal`]) and the thermal leg
/// already reads plastic work ([`deposit_increment_heat`]). This closes the
/// loop: [`step_thermoplastic`] alternates the two until the temperature
/// increment stops moving, so the increment it returns satisfies *both* legs
/// rather than one leg evaluated at the other's previous guess.
///
/// # ⚠️ The fixed point is in the increment, not in the temperature
///
/// The map is on `δT`, the rise this increment adds:
///
/// ```text
/// δT_{k+1} = δT_k + ω (deposit(ΔW_p(T^n + δT_k; state^n)) − δT_k)
/// ```
///
/// with `(ε_p^n, ε̄_p^n, T^n)` — the committed state — held fixed for every
/// sweep. That is the Simo–Miehe (Armero–Simo) isothermal split, and it is why
/// [`ElastoplasticIncrement::commit`] is a separate call: the driver sweeps
/// with [`ElastoplasticProblem::step`], which always reads the same committed
/// state, and the caller commits once afterwards. ⚠️ **Committing inside the
/// sweep would make each sub-iteration start from a different `ε_p`, so the
/// iteration would no longer be a map on `δT` and converging it would not
/// solve the coupled equations.**
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct ThermoplasticCoupling {
    heating: PlasticHeating,
    expansion_per_k: Fix128,
    material_reference: Fix128,
    softening: Option<ThermalSoftening>,
    relaxation: Fix128,
    residual_floor_fraction: Fix128,
    sub_iteration: crate::coupled_iteration::SubIterationConfig,
}

impl ThermoplasticCoupling {
    /// Build a coupling, refusing a combination that cannot converge.
    ///
    /// `material_reference` is the absolute temperature at which `σ_y₀`, `H₀`
    /// and `expansion_per_k` were measured; the base field handed to
    /// [`step_thermoplastic`] holds absolute temperatures, and the rise is
    /// taken against this reference.
    ///
    /// `relaxation` is the constant `ω` of the update above. `ω = 1` is the
    /// plain (unrelaxed) sweep, and it is the best value measured on every
    /// scene tried — the range is capped at one for that reason and not for a
    /// theoretical one.
    ///
    /// ⚠️ **Over-relaxation is worse here, and the closed-form optimum is not
    /// the optimum.** The Jacobian of the map at its fixed point was measured
    /// by finite differences over the 25 nodes the deposit writes, giving
    /// `λ ∈ {+0.280, −0.078}` at `c_v = 2⁻⁸` and `{+0.938, −0.401}` at
    /// `c_v = 2⁻¹⁰`. Richardson's equalising choice
    /// `ω* = 2/(2 − λ_max − λ_min)` is then 1.112 and 1.367, and it asks for
    /// **over**-relaxation. Measured sweeps say otherwise:
    ///
    /// | `ω` | 1 | 1.112 | 1.25 | 1.367 | 1.5 | 1.75 |
    /// |---|---|---|---|---|---|---|
    /// | `c_v = 2⁻⁸` | **10** | 14 | 21 | 29 | 45 | 174 |
    /// | `c_v = 2⁻¹⁰` | **24** | 37 | 74 | — | — | — |
    ///
    /// (`—` is `NotConverged`, `ArithmeticWrapped` at 1.5 and `Diverging` at
    /// 1.75; the wrap is a real one — the over-relaxed oscillation grows until
    /// it leaves the range — not a detector misfire.)
    ///
    /// The reason the asymptotic rate does not decide the sweep count is that
    /// **the iteration finishes inside its transient**: at `c_v = 2⁻⁸` the
    /// early sweeps contract by 0.084 each while `λ_max` is 0.280, and the
    /// residual reaches the floor before the asymptotic regime begins. Taking a
    /// step longer than one then overshoots a strongly contracting nonlinear
    /// map and has to come back.
    ///
    /// Which effect owns which eigenvalue was measured by switching them off
    /// one at a time, and it is the physics one would guess: **softening is the
    /// positive eigenvalue** (hotter → weaker → more plastic work → hotter) and
    /// **expansion is the negative one** (hotter → more eigenstrain → less
    /// elastic trial strain → less work). With `α = 0` the negative eigenvalue
    /// vanishes (`λ₂ = −0.000`); with no softening law the dominant eigenvalue
    /// turns negative (`−0.146`). The alternating look of the early iterates is
    /// the negative eigenvalue showing in the transient while the positive one
    /// sets the tail.
    ///
    /// ⚠️ The two heat capacities above are **non-physical** (chosen to make
    /// the map slow enough to study). At handbook values the same instrument
    /// reads `λ_max = +3.01e-4` for steel, `+4.74e-4` for aluminium and
    /// `+6.06e-4` for PLA — three decades below the `0.9` at which a
    /// monolithic assembly would pay for itself. The table, the instrument
    /// (`tests/analytic_thermoplastic_coupling.rs::the_monolithic_entry_condition_is_three_decades_away_at_real_heat_capacities`)
    /// and the decision not to build one are in
    /// [`crate::coupled_iteration`]'s module doc.
    ///
    /// `residual_floor_fraction` is the floor below which the residual is read
    /// as zero, as a **fraction of the magnitude the first sweep deposits**.
    ///
    /// ⚠️ **It is not optional, and it cannot be an absolute number.** The
    /// monitor's own target is relative to the first *residual*, which under
    /// this map collapses by several decades in one sweep, so the target lands
    /// below what the arithmetic can reproduce; the residual then wanders in
    /// the noise, and a wander that happens to rise twice and fall once is
    /// read by the wrap detector as a silent `Fix128` wrap. Measured on the bar
    /// scene, the noise the residual settles into is a fixed **fraction of the
    /// answer**, not a fixed number of ulp:
    ///
    /// | `c_v` (MPa/K) | converged `max δT` | noise the residual settles at | ratio |
    /// |---|---|---|---|
    /// | `4` | `2.71e-3` | `≈5e-14` | `1.8e-11` |
    /// | `1` | `1.08e-2` | `≈2.2e-13` | `2.0e-11` |
    /// | `1/4` | `4.33e-2` | `≈9e-13` | `2.1e-11` |
    /// | `1/64` | `6.82e-1` | `≈8.4e-12` | `1.2e-11` |
    ///
    /// The ratio holds to within a factor of two across two and a half decades
    /// of answer, so the floor belongs on that scale. `2⁻³⁰ ≈ 9.3e-10` leaves
    /// about 45 times the measured noise and is still far below any temperature
    /// a scene of interest produces.
    ///
    /// **Where that noise comes from**, measured rather than assumed: it is
    /// [`RESIDUAL_NORM_FLOOR`], the **absolute** floor on the force residual
    /// that both the linear solve and the Newton loop clamp their targets to.
    /// Three experiments separate it from the relative tolerances:
    ///
    /// | change | relative noise at `c_v = 2⁻⁸` |
    /// |---|---|
    /// | baseline | `1.23e-11` |
    /// | `SolverConfig::relative_tolerance` `2⁻³⁰ → 2⁻⁴⁰` | `1.45e-11` — ⚠️ unmoved |
    /// | `newton_tolerance` `2⁻⁴⁰ → 2⁻²⁰` (a million times looser) | `1.02e-11` — ⚠️ unmoved |
    /// | every force in the problem `× 64` | `1.27e-13` — **97× smaller** |
    /// | every force in the problem `× 4096` | `4.60e-15` — **2667× smaller** |
    ///
    /// Tightening or loosening either tolerance does nothing, while scaling the
    /// loads scales the relative noise inversely — which is what an absolute
    /// floor on a force residual does and what a relative tolerance cannot do.
    /// ⚠️ It also explains why the ratio is the same at every `c_v`: the answer
    /// `δT` grows like `1/c_v` while the force residual floor does not move.
    /// ⚠️ A corollary worth knowing: a `newton_tolerance` below
    /// `RESIDUAL_NORM_FLOOR / ‖f‖` **is not in effect**, because the target is
    /// `max(relative × norm, RESIDUAL_NORM_FLOOR)` — the `2⁻⁴⁰` the oracles
    /// pass is already clamped on this scene.
    ///
    /// ⚠️ **A fraction of zero reproduces the behaviour above** and is refused.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] if `relaxation` is outside `(0, 1]` (zero
    /// never moves, negative walks away from the fixed point, above one
    /// overshoots and is not what the closed-form optimum ever asks for), if
    /// `residual_floor_fraction` is outside `(0, 1)`, or if `expansion_per_k`
    /// is negative.
    /// [`FemError::CoupledSubIterationFailed`] if `sub_iteration` is itself
    /// invalid, carrying the fault the sub-iteration configuration reported.
    pub fn try_new(
        heating: PlasticHeating,
        expansion_per_k: Fix128,
        material_reference: Fix128,
        softening: Option<ThermalSoftening>,
        relaxation: Fix128,
        residual_floor_fraction: Fix128,
        sub_iteration: crate::coupled_iteration::SubIterationConfig,
    ) -> Result<Self, FemError> {
        if relaxation <= Fix128::ZERO || relaxation > Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "relaxation must be in (0, 1]: zero never moves, and over-relaxation is \
                 measured to be monotonically worse on this map",
            ));
        }
        if residual_floor_fraction <= Fix128::ZERO || residual_floor_fraction >= Fix128::ONE {
            return Err(FemError::InvalidConfig(
                "residual_floor_fraction must be in (0, 1): zero leaves the stopping rule \
                 relative-only, which the arithmetic cannot satisfy",
            ));
        }
        if expansion_per_k.is_negative() {
            return Err(FemError::InvalidConfig(
                "expansion_per_k must not be negative",
            ));
        }
        sub_iteration.validate().map_err(|fault| {
            FemError::CoupledSubIterationFailed(
                crate::coupled_iteration::CoupledIterationError::InvalidConfig { fault },
            )
        })?;
        Ok(Self {
            heating,
            expansion_per_k,
            material_reference,
            softening,
            relaxation,
            residual_floor_fraction,
            sub_iteration,
        })
    }

    /// The relaxation factor `ω`.
    #[must_use]
    pub const fn relaxation(&self) -> Fix128 {
        self.relaxation
    }

    /// The floor below which the residual is read as zero, as a fraction of
    /// the magnitude the first sweep deposits.
    #[must_use]
    pub const fn residual_floor_fraction(&self) -> Fix128 {
        self.residual_floor_fraction
    }
}

/// One increment of the coupled problem, with the sweep that produced it.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ThermoplasticIncrement {
    /// The mechanical increment, evaluated at the converged temperature.
    ///
    /// Not yet committed: call [`ElastoplasticIncrement::commit`] once, after
    /// reading whatever the caller needs from this report.
    pub increment: ElastoplasticIncrement,
    /// The temperature rise this increment added, `δT`, on the base grid.
    ///
    /// Add it to the base field to get the temperature the next increment
    /// starts from.
    ///
    /// ⚠️ **This is the deposit of [`Self::increment`]'s own plastic work, so
    /// the thermal leg holds exactly and the mechanical leg holds to within the
    /// floor.** The sweep evaluated the mechanics at the *relaxed* iterate,
    /// which differs from this field by at most the residual floor; of the two
    /// legs only one can be exact in the returned pair, and this is the choice
    /// that makes the energy ledger exact — the heat on the grid is precisely
    /// `β` times the work the returned increment reports, with nothing lost to
    /// the iterate the sweep happened to stop on.
    pub temperature_increment: CoupledField,
    /// What the sub-iteration did: sweeps, final residual, contraction ratio.
    pub report: crate::coupled_iteration::SubIterationReport,
}

/// Drive one increment of the two-way thermoplastic coupling to a fixed point.
///
/// `base` holds the **absolute** temperatures the increment starts from, `T^n`.
/// The returned [`ThermoplasticIncrement::temperature_increment`] is `δT`, so
/// the caller advances the temperature by adding it to `base`.
///
/// # Errors
///
/// Whatever [`ElastoplasticProblem::step`] or [`deposit_increment_heat`]
/// return, plus [`FemError::CoupledSubIterationFailed`] when the sweep does
/// not reach a fixed point — `Diverging` if the splitting is not a contraction
/// (relaxing harder is the remedy, not a bigger budget), `Stagnated`,
/// `NotConverged`, or `ArithmeticWrapped`.
pub fn step_thermoplastic(
    problem: &ElastoplasticProblem,
    state: &ElastoplasticState,
    mesh: &SdfTetMesh,
    base: &CoupledField,
    coupling: &ThermoplasticCoupling,
    factor: Fix128,
) -> Result<ThermoplasticIncrement, FemError> {
    use crate::coupled_iteration::residual_norm_inf;

    // δT_0 = 0. Cloned from `base` so the grids agree by construction — there
    // is no mismatch for a guard to catch.
    let mut delta = base.clone();
    delta.clear();
    // Scratch, reallocated per sweep only in the sense of being cleared: the
    // deposit *adds*, so a target carried across sweeps would accumulate and
    // the map would become a time integration rather than a fixed-point map.
    let mut target = base.clone();
    let mut absolute = base.clone();
    let mut difference = vec![Fix128::ZERO; base.cell_count()];
    // Derived on the first sweep from what that sweep deposits, because the
    // noise the residual settles into scales with the answer rather than
    // sitting at a fixed number of ulp. A purely elastic increment deposits
    // nothing, which leaves the floor at zero — correct, since `δT* = δT = 0`
    // makes the residual exactly zero and the first sweep is already the fixed
    // point.
    let mut floor = Fix128::ZERO;
    let mut last: Option<ElastoplasticIncrement> = None;

    // The sweep is the fixed-point map `δT ↦ deposit(ΔW_p(T^n + δT))`, driven
    // by `coupled_iteration::run_sub_iteration`, which owns the stopping rule.
    let report = run_sub_iteration_fallible(coupling.sub_iteration, |index| {
        // T_k = T^n + δT_k
        absolute.as_mut_slice().copy_from_slice(base.as_slice());
        for (value, &d) in absolute.as_mut_slice().iter_mut().zip(delta.as_slice()) {
            *value = *value + d;
        }
        let rise = TemperatureRise::from_absolute(&absolute, coupling.material_reference);
        let request = ElastoplasticIncrementRequest::new(factor).with_thermal(
            ThermalExpansion::from_rise(&rise, coupling.expansion_per_k),
            coupling.softening,
        );
        let increment = problem.step(state, &request)?;

        // δT* = deposit(ΔW_p). Cleared first: the deposit adds.
        target.clear();
        deposit_increment_heat(
            mesh,
            &increment.plastic_work_increment,
            &coupling.heating,
            &mut target,
        )?;

        for ((slot, &t), &d) in difference
            .iter_mut()
            .zip(target.as_slice())
            .zip(delta.as_slice())
        {
            *slot = t - d;
        }
        let raw_residual = residual_norm_inf(&difference);
        if index == 0 {
            floor = coupling.residual_floor_fraction * residual_norm_inf(target.as_slice());
        }
        // The floor is applied before the monitor sees the number, so an
        // increment that reproduces itself to within the arithmetic's own
        // noise reads as an exact fixed point rather than as a budget overrun —
        // or, worse, as the rise-rise-fall signature the wrap detector reads as
        // a silent `Fix128` wrap.
        let residual = if raw_residual <= floor {
            Fix128::ZERO
        } else {
            raw_residual
        };
        last = Some(increment);

        // δT_{k+1} = δT_k + ω (δT* − δT_k): the relaxation is a blend of the
        // iterate toward the deposit, `CoupledField::blend_from` at weight ω.
        // Applied after every sweep; the converging sweep's update is unused,
        // because the returned increment is the deposit itself.
        delta
            .blend_from(&target, coupling.relaxation)
            .expect("delta and target are clones of base, so the grids agree");
        Ok(residual)
    })?;

    let increment = last.expect("a report means at least one sweep ran");
    Ok(ThermoplasticIncrement {
        increment,
        temperature_increment: target,
        report,
    })
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Error-driven adaptive refinement
// ============================================================================

/// Per-element squared error indicators by Zienkiewicz–Zhu stress recovery.
///
/// P1 strain is constant per element, so `σ_h` is a piecewise constant and has a
/// jump across every interior face. Averaging it to the nodes with volume
/// weights and reading it back gives a continuous `σ*`, and the size of
/// `Δσ = σ* − σ_h` is what the element contributed to the error:
///
/// ```text
/// η_e² = V_e · (Δσ : C⁻¹ : Δσ)
///      = V_e / (2μ) · [ Δσ:Δσ − λ/(3λ + 2μ) · (tr Δσ)² ]
/// ```
///
/// The energy norm is used rather than a plain Frobenius one because that is the
/// norm the finite element method is optimal in, so the indicator is comparable
/// across elements of different size and stiffness.
///
/// `Δσ` is taken at the centroid, which for the linear `σ*` is the average of
/// its four nodal values — a one-point rule, exact for the linear part of the
/// integrand.
///
/// # ⚠️ Exactly zero where the space is exact
///
/// When the solution lies in the P1 space the strain is uniform, every element
/// carries the **same** `σ_h`, the volume-weighted average is that same tensor
/// and `σ* = σ_h` identically. `η_e²` is then zero to the bit, for any material
/// and any mesh. That is the positive control
/// `tests/analytic_adaptive_refinement.rs::an_exact_solution_has_indicators_of_exactly_zero`
/// — and it is also why that test is paired with a localized scene, since an
/// indicator that returned zero unconditionally would satisfy it too.
///
/// Returns one value per tetrahedron, in mesh order.
///
/// # Errors
///
/// [`FemError::EmptyMesh`] for a mesh with no vertices or tetrahedra,
/// [`FemError::SolutionDoesNotMatchMesh`] when the solution has a different node
/// count, [`FemError::SolutionElementCountDoesNotMatchMesh`] when it reports a
/// different element count, and [`FemError::DegenerateElement`] for a
/// tetrahedron with no volume.
#[must_use = "the indicators are the whole point of estimating; dropping them \
              leaves the refinement with nothing to go on"]
pub fn error_indicators_squared(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    solution: &FemSolution,
) -> Result<Vec<Fix128>, FemError> {
    let vertex_count = mesh.vertices.len();
    if vertex_count == 0 || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    if solution.displacements.len() != vertex_count {
        return Err(FemError::SolutionDoesNotMatchMesh {
            nodes: solution.displacements.len(),
            vertex_count,
        });
    }
    if solution.element_stress.len() != mesh.tets.len() {
        return Err(FemError::SolutionElementCountDoesNotMatchMesh {
            elements: solution.element_stress.len(),
            tet_count: mesh.tets.len(),
        });
    }
    let elements = build_elements(mesh)?;

    // Volume-weighted nodal average of the element stresses.
    let mut weight = vec![Fix128::ZERO; vertex_count];
    let mut recovered = vec![StressTensor::default(); vertex_count];
    for (element, stress) in elements.iter().zip(solution.element_stress.iter()) {
        for &node in &element.nodes {
            weight[node] = weight[node] + element.volume;
            let acc = &mut recovered[node];
            acc.xx = acc.xx + stress.xx * element.volume;
            acc.yy = acc.yy + stress.yy * element.volume;
            acc.zz = acc.zz + stress.zz * element.volume;
            acc.xy = acc.xy + stress.xy * element.volume;
            acc.yz = acc.yz + stress.yz * element.volume;
            acc.zx = acc.zx + stress.zx * element.volume;
        }
    }
    for (acc, &w) in recovered.iter_mut().zip(weight.iter()) {
        if w.is_zero() {
            continue;
        }
        acc.xx = acc.xx / w;
        acc.yy = acc.yy / w;
        acc.zz = acc.zz / w;
        acc.xy = acc.xy / w;
        acc.yz = acc.yz / w;
        acc.zx = acc.zx / w;
    }

    let quarter = Fix128::from_raw(0, 1 << 62);
    let mut indicators = Vec::with_capacity(elements.len());
    for (element, stress) in elements.iter().zip(solution.element_stress.iter()) {
        // `σ*` at the centroid is the mean of the four nodal values.
        let mut star = StressTensor::default();
        for &node in &element.nodes {
            let r = &recovered[node];
            star.xx = star.xx + r.xx;
            star.yy = star.yy + r.yy;
            star.zz = star.zz + r.zz;
            star.xy = star.xy + r.xy;
            star.yz = star.yz + r.yz;
            star.zx = star.zx + r.zx;
        }
        let d = StressTensor {
            xx: star.xx * quarter - stress.xx,
            yy: star.yy * quarter - stress.yy,
            zz: star.zz * quarter - stress.zz,
            xy: star.xy * quarter - stress.xy,
            yz: star.yz * quarter - stress.yz,
            zx: star.zx * quarter - stress.zx,
        };
        // ⚠️ The norm comes from `StressTensor::complementary_energy_density`
        // rather than being written out again here, so a mutation to the norm
        // shows up in both the indicator and its closed-form oracles.
        let density = d.complementary_energy_density(material);
        // The energy density of a stress difference cannot be negative; only
        // rounding can take it below zero, and a negative indicator would break
        // the marking that consumes it.
        let clamped = if density.is_negative() {
            Fix128::ZERO
        } else {
            density
        };
        indicators.push(element.volume * clamped);
    }
    Ok(indicators)
}

/// Dörfler (bulk) marking: the **smallest** set of elements carrying at least
/// `bulk_fraction` of the total squared indicator.
///
/// Sorting the elements by indicator and taking them greedily until the running
/// sum reaches `θ · total` gives that minimal set, which is the property that
/// makes the refinement adaptive rather than "refine everything": dropping its
/// smallest member must break the criterion.
///
/// # ⚠️ The tie-break is part of the contract
///
/// Equal indicators are ordered by **element index ascending**, so the chosen
/// set is a function of the indicator values and not of the order they happen to
/// arrive in. Without that, two meshes differing only by numbering would refine
/// differently and the solver would stop being reproducible — which is pinned by
/// `tests/analytic_adaptive_refinement.rs::bulk_marking_does_not_depend_on_element_order`.
///
/// A total of zero marks nothing and is **not** an error: an exact solution
/// scores zero everywhere, which means there is no work to do rather than that
/// the request was malformed.
///
/// # Errors
///
/// [`FemError::EmptyMesh`] for an empty slice,
/// [`FemError::InvalidConfig`] when `bulk_fraction` is outside `(0, 1]` or any
/// indicator is negative (a squared quantity cannot be).
pub fn mark_bulk(
    indicators_squared: &[Fix128],
    bulk_fraction: Fix128,
) -> Result<Vec<bool>, FemError> {
    if indicators_squared.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    if bulk_fraction <= Fix128::ZERO || bulk_fraction > Fix128::ONE {
        return Err(FemError::InvalidConfig("bulk_fraction must be in (0, 1]"));
    }
    if indicators_squared.iter().any(|v| v.is_negative()) {
        return Err(FemError::InvalidConfig(
            "a squared error indicator cannot be negative",
        ));
    }
    let total = indicators_squared
        .iter()
        .fold(Fix128::ZERO, |acc, &v| acc + v);
    let mut marked = vec![false; indicators_squared.len()];
    if total.is_zero() {
        return Ok(marked);
    }
    let target = total * bulk_fraction;
    let mut order: Vec<usize> = (0..indicators_squared.len()).collect();
    // Descending by indicator, ascending by index on a tie.
    order.sort_by(|&a, &b| {
        indicators_squared[b]
            .cmp(&indicators_squared[a])
            .then(a.cmp(&b))
    });
    let mut running = Fix128::ZERO;
    for index in order {
        if running >= target {
            break;
        }
        marked[index] = true;
        running = running + indicators_squared[index];
    }
    Ok(marked)
}

/// How to drive [`solve_adaptive`].
///
/// Fields are private and checked by [`Self::try_new`]; the struct is
/// `#[non_exhaustive]` so a later stopping rule can be added without a breaking
/// change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct AdaptiveConfig {
    linear: SolverConfig,
    bulk_fraction: Fix128,
    max_rounds: u32,
    max_refine_passes: u32,
}

impl AdaptiveConfig {
    /// Validate and build.
    ///
    /// `max_rounds` counts **solves**, so a budget of one solves the mesh it was
    /// given and refines nothing. `max_refine_passes` is handed to
    /// [`SdfTetMesh::try_refine_marked`] each round; ⚠️ propagation makes more
    /// work than the marks alone, so it has to be larger than the number of
    /// marked elements suggests.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] for a zero round budget, a zero refinement
    /// budget, or a `bulk_fraction` outside `(0, 1]`.
    pub fn try_new(
        linear: SolverConfig,
        bulk_fraction: Fix128,
        max_rounds: u32,
        max_refine_passes: u32,
    ) -> Result<Self, FemError> {
        if max_rounds == 0 {
            return Err(FemError::InvalidConfig("max_rounds must be positive"));
        }
        if max_refine_passes == 0 {
            return Err(FemError::InvalidConfig(
                "max_refine_passes must be positive",
            ));
        }
        if bulk_fraction <= Fix128::ZERO || bulk_fraction > Fix128::ONE {
            return Err(FemError::InvalidConfig("bulk_fraction must be in (0, 1]"));
        }
        Ok(Self {
            linear,
            bulk_fraction,
            max_rounds,
            max_refine_passes,
        })
    }

    /// Configuration of each linear solve.
    #[must_use]
    pub const fn linear(&self) -> SolverConfig {
        self.linear
    }

    /// Fraction of the total squared indicator the marked set must carry.
    #[must_use]
    pub const fn bulk_fraction(&self) -> Fix128 {
        self.bulk_fraction
    }

    /// Maximum number of solves.
    #[must_use]
    pub const fn max_rounds(&self) -> u32 {
        self.max_rounds
    }

    /// Pass budget handed to each refinement.
    #[must_use]
    pub const fn max_refine_passes(&self) -> u32 {
        self.max_refine_passes
    }
}

/// What an adaptive run produced.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveSolution {
    /// The final mesh, refined where the estimator asked.
    pub mesh: SdfTetMesh,
    /// The solution on that mesh.
    pub field: FemSolution,
    /// Solves performed, at least one and at most `max_rounds`.
    pub rounds: u32,
    /// `Σ_e η_e²` after each solve, oldest first.
    ///
    /// The sequence the driver acted on, so a caller can see whether the
    /// estimate was still falling when the budget ran out.
    pub total_indicator_history: Vec<Fix128>,
}

/// Solve, estimate, mark, refine, repeat.
///
/// The loop that makes refinement adaptive: each round solves the current mesh,
/// scores it with [`error_indicators_squared`], marks a bulk set with
/// [`mark_bulk`], and refines those elements with
/// [`SdfTetMesh::try_refine_marked`]. It stops early when the estimate is zero —
/// there is then nothing to refine — and otherwise after `max_rounds` solves.
///
/// # ⚠️ Why the boundary conditions arrive as a closure
///
/// Refinement appends vertices, so indices in a [`BoundaryConditions`] stay
/// valid. That is not sufficient. A midpoint created on a clamped face is a
/// **new** node that nothing prescribes, so the face silently becomes partially
/// free — the same tear `tests/hanging_node_effect.rs` measures at `1.489e-1`
/// for a linear field, caused by the boundary condition instead of by the mesh.
///
/// ⚠️ Propagating the conditions would only be half possible. A prescribed
/// displacement can be inherited when both parents are prescribed on that axis
/// (the midpoint value is their average, and `1/2` is exact in `Fix128`), but a
/// **nodal load cannot**: a point force carries no record of the traction it
/// stood for, so there is no correct way to split it. Rather than propagate the
/// half that works and quietly drop the other, this asks the caller to rebuild
/// its conditions for each mesh.
///
/// # Errors
///
/// Whatever [`solve`], [`error_indicators_squared`] or [`mark_bulk`] return, and
/// [`FemError::InvalidConfig`] if a refinement pass runs out of budget — the
/// message names the refinement rather than the solve, because the caller's move
/// is to raise `max_refine_passes`.
pub fn solve_adaptive<F>(
    mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    boundary_for: F,
    adaptive: &AdaptiveConfig,
) -> Result<AdaptiveSolution, FemError>
where
    F: Fn(&SdfTetMesh) -> BoundaryConditions,
{
    if mesh.vertices.is_empty() || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    let mut current = mesh.clone();
    let mut history = Vec::with_capacity(adaptive.max_rounds as usize);
    let mut rounds = 0_u32;
    loop {
        let boundary = boundary_for(&current);
        let field = solve(&current, material, &boundary, &adaptive.linear)?;
        rounds += 1;
        let indicators = error_indicators_squared(&current, material, &field)?;
        let total = indicators.iter().fold(Fix128::ZERO, |acc, &v| acc + v);
        history.push(total);

        if total.is_zero() || rounds >= adaptive.max_rounds {
            return Ok(AdaptiveSolution {
                mesh: current,
                field,
                rounds,
                total_indicator_history: history,
            });
        }
        let marked = mark_bulk(&indicators, adaptive.bulk_fraction)?;
        if !marked.iter().any(|&m| m) {
            return Ok(AdaptiveSolution {
                mesh: current,
                field,
                rounds,
                total_indicator_history: history,
            });
        }
        let mut next = current.clone();
        next.try_refine_marked(&marked, adaptive.max_refine_passes)
            .map_err(|_| {
                FemError::InvalidConfig(
                    "max_refine_passes ran out while conformity still owed splits",
                )
            })?;
        current = next;
    }
}

/// The solve / estimate / mark / refine loop shared by the higher-order
/// adaptive drivers ([`crate::quadratic_elastic_fem::solve_adaptive_quadratic`],
/// [`crate::cubic_elastic_fem::solve_adaptive_cubic`]).
///
/// `step` solves one mesh and returns the full-order solution together with the
/// per-tetrahedron `η_e²`. Refinement is always done on the straight-edged
/// corner mesh — `QuadraticMesh` and `CubicMesh` are *built from* an
/// `SdfTetMesh` — so the conforming refinement of [`SdfTetMesh`] is reused as is
/// and the high-order mesh is rebuilt from it each round.
///
/// Returns the final corner mesh, the last solution, the number of solves and the
/// `Σ η_e²` history.
pub(crate) fn adaptive_refinement_loop<S>(
    mesh: &SdfTetMesh,
    adaptive: &AdaptiveConfig,
    mut step: S,
) -> Result<(SdfTetMesh, FemSolution, u32, Vec<Fix128>), FemError>
where
    S: FnMut(&SdfTetMesh) -> Result<(FemSolution, Vec<Fix128>), FemError>,
{
    if mesh.vertices.is_empty() || mesh.tets.is_empty() {
        return Err(FemError::EmptyMesh);
    }
    let mut current = mesh.clone();
    let mut history = Vec::with_capacity(adaptive.max_rounds as usize);
    let mut rounds = 0_u32;
    loop {
        let (field, indicators) = step(&current)?;
        rounds += 1;
        let total = indicators.iter().fold(Fix128::ZERO, |acc, &v| acc + v);
        history.push(total);
        if total.is_zero() || rounds >= adaptive.max_rounds {
            return Ok((current, field, rounds, history));
        }
        let marked = mark_bulk(&indicators, adaptive.bulk_fraction)?;
        if !marked.iter().any(|&m| m) {
            return Ok((current, field, rounds, history));
        }
        let mut next = current.clone();
        next.try_refine_marked(&marked, adaptive.max_refine_passes)
            .map_err(|_| {
                FemError::InvalidConfig(
                    "max_refine_passes ran out while conformity still owed splits",
                )
            })?;
        current = next;
    }
}

/// `η_e²` for a higher-order solution, scored by the P1 recovery estimator on the
/// corner mesh.
///
/// The high-order solution reports one centroid stress per element, which is
/// exactly the datum [`error_indicators_squared`] recovers from, so the estimator
/// is reused unchanged on a view of the solution that keeps only the corner
/// displacements (corner `v` is node `v` in both `QuadraticMesh` and `CubicMesh`).
pub(crate) fn corner_indicators_squared(
    corner_mesh: &SdfTetMesh,
    material: &ElasticMaterial,
    field: &FemSolution,
) -> Result<Vec<Fix128>, FemError> {
    let mut view = field.clone();
    view.displacements.truncate(corner_mesh.vertices.len());
    error_indicators_squared(corner_mesh, material, &view)
}

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

    /// One tetrahedron `(0,0,0) (1,0,0) (0,1,0) (0,0,1)`: `F_zz = 1 + u_z` of
    /// vertex 3, and `det F = F_zz`, so regularity is a statement about one dof.
    fn single_tet_elements() -> Vec<Element> {
        let mesh = SdfTetMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            tets: vec![crate::sdf_fem_mesh::Tetrahedron {
                vertices: [0, 1, 2, 3],
            }],
        };
        build_elements(&mesh).expect("a regular tetrahedron")
    }

    const UZ3: usize = 11;

    /// A step that inverts the element is pulled back along the segment by the
    /// first `2⁻ᵏ` that makes it regular. `u_z = −2 + 2⁻²⁹`: `k = 1` lands on
    /// `det F = 2⁻³⁰`, positive but **below the polar floor** (`2⁻²⁰`), so it is
    /// refused too — a floor of zero would stop there; `k = 2` is the first
    /// regular point, `−1/2 + 2⁻³¹`.
    #[test]
    fn a_step_that_inverts_an_element_is_halved_until_it_is_regular() {
        let elements = single_tet_elements();
        let previous = vec![Fix128::ZERO; 12];
        let mut next = previous.clone();
        next[UZ3] = Fix128::from_int(-2) + Fix128::from_raw(0, 1 << 35);
        step_stays_regular(&elements, &previous, &mut next);
        assert_eq!(
            next[UZ3],
            Fix128::from_ratio(-1, 2) + Fix128::from_raw(0, 1 << 33)
        );
        assert!(all_elements_regular(&elements, &next));
    }

    /// The line search's answer is taken as it is; without one the full step is
    /// pulled back only for regularity.
    #[test]
    fn settle_step_takes_the_accepted_point_or_falls_back_to_regularity() {
        let elements = single_tet_elements();
        let previous = vec![Fix128::ZERO; 12];
        let mut full = previous.clone();
        full[UZ3] = Fix128::from_int(-2);
        let mut accepted_point = previous.clone();
        accepted_point[UZ3] = Fix128::from_ratio(1, 8);

        let mut next = full.clone();
        settle_step(&elements, &previous, &mut next, Some(accepted_point));
        assert_eq!(next[UZ3], Fix128::from_ratio(1, 8));

        let mut next = full;
        settle_step(&elements, &previous, &mut next, None);
        assert_eq!(next[UZ3], Fix128::from_ratio(-1, 2));
    }

    /// A regular step is not touched, bit for bit.
    #[test]
    fn a_regular_step_is_left_alone() {
        let elements = single_tet_elements();
        let previous = vec![Fix128::ZERO; 12];
        let mut next = previous.clone();
        next[UZ3] = Fix128::from_ratio(1, 4);
        step_stays_regular(&elements, &previous, &mut next);
        assert_eq!(next[UZ3], Fix128::from_ratio(1, 4));
    }

    /// When no `2⁻ᵏ` with `k ≤ 16` is regular the step ends at the shortest one
    /// tried, which is still refused downstream — it is not silently accepted.
    #[test]
    fn a_step_that_never_becomes_regular_ends_at_the_shortest_trial() {
        let elements = single_tet_elements();
        let previous = vec![Fix128::ZERO; 12];
        let mut next = previous.clone();
        next[UZ3] = Fix128::from_int(-(1 << 20));
        step_stays_regular(&elements, &previous, &mut next);
        assert_eq!(next[UZ3], Fix128::from_int(-16));
        assert!(!all_elements_regular(&elements, &next));
    }

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
    /// σ      = (2/J)·W₁·B + (κ(J−1) − p_ref/J)·I = μ(B − I)
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

#[cfg(test)]
mod fallible_driver_tests {
    //! The error-capturing wrapper around `coupled_iteration::run_sub_iteration`
    //! that `step_thermoplastic` drives with: a sweep that fails at any index
    //! surfaces as that error, never as a converged report (a zero residual is
    //! what the monitor calls converged, which is exactly the trap).
    use super::*;
    use crate::coupled_iteration::{run_sub_iteration, SubIterationConfig};

    fn geometric(index: u32) -> Fix128 {
        // 1, 1/2, 1/4, …: converges under the default tolerance.
        Fix128::from_raw(0, 1u64 << (63 - index.min(60)))
    }

    #[test]
    fn a_failing_sweep_is_the_error_not_a_converged_report() {
        for fail_at in [0u32, 1, 2, 3] {
            let outcome = run_sub_iteration_fallible(SubIterationConfig::default(), |index| {
                if index == fail_at {
                    Err(FemError::EmptyMesh)
                } else {
                    Ok(geometric(index))
                }
            });
            assert!(
                matches!(outcome, Err(FemError::EmptyMesh)),
                "failure at sweep {fail_at} must surface, got {outcome:?}"
            );
        }
    }

    #[test]
    fn a_succeeding_sweep_reports_exactly_what_run_sub_iteration_reports() {
        let config = SubIterationConfig::default();
        let direct = run_sub_iteration(config, geometric).expect("converges");
        let wrapped =
            run_sub_iteration_fallible(config, |index| Ok(geometric(index))).expect("converges");
        assert_eq!(wrapped, direct);
        assert!(wrapped.sweeps > 1, "not vacuous: more than one sweep ran");
    }

    #[test]
    fn a_monitor_refusal_is_mapped_to_the_coupled_variant() {
        // A residual that never shrinks runs out of budget.
        let outcome =
            run_sub_iteration_fallible(SubIterationConfig::default(), |_| Ok(Fix128::ONE));
        assert!(
            matches!(outcome, Err(FemError::CoupledSubIterationFailed(_))),
            "{outcome:?}"
        );
    }
}
