//! Small-strain linear elastic FEM on tetrahedra (P1 / constant strain).
//!
//! Consumes a [`crate::sdf_fem_mesh::SdfTetMesh`] and returns a Cauchy stress
//! tensor per element, which is what a stress law needs and what no other
//! module in this crate produces: [`crate::beam_stress`] and
//! [`crate::fillet_stress`] are closed forms for shapes that reduce to a beam
//! or a notch, and [`crate::deformable`] is an XPBD solver whose volume and
//! shape constraints never form a stress tensor.
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
//! - Isotropic material only. [`crate::filament_db::MaterialProperties`] has
//!   no Poisson's ratio, so [`ElasticMaterial::from_filament`] takes one.
//! - Small strain, small displacement: no geometric nonlinearity, no contact,
//!   no plasticity. Past yield the result is the *elastic* stress, which is
//!   what a yield check wants as its input.
//! - P1 tetrahedra are stiff in bending. A beam resolved by a few elements
//!   through the thickness under-predicts deflection; refine through the
//!   thickness rather than along the span.
//! - The rigid-body-mode check is a necessary condition, not a sufficient one
//!   (see [`FemError::UnderConstrained`]).
//!
//! Author: Moroya Sakamoto

use crate::math::Fix128;
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
/// needed and, in fixed point, **what residual is reachable at all**. Those two
/// can point in opposite directions, so this is a knob and not a constant:
/// measured on a 10:1 cantilever, [`Self::JacobiScaled`] cut the iteration count
/// by 13% on meshes up to 3,200 elements, and at 25,600 elements it stalled at a
/// relative residual of 4.66e-9 where the unpreconditioned solve had reached
/// 9.41e-10.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Preconditioner {
    /// None. The iteration works on `K` as it stands.
    None,
    /// Diagonal (Jacobi), scaled so its mean is one.
    ///
    /// The scaling is not cosmetic. Multiplying `M` by a constant leaves the
    /// iterates unchanged — `α` and `β` absorb it exactly — but it decides the
    /// magnitude the inner products live at, and `Fix128` has a hard floor at
    /// `2⁻⁶⁴`. With a plain `1/diag` the terms of `rᵀz` and `pᵀKp` fall below
    /// that floor and round to zero while the residual is still above tolerance.
    #[default]
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
    stagnation_window: u32,
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

    /// Replace the stagnation rule.
    ///
    /// The iteration is abandoned when the *best* residual so far has not
    /// improved by a factor of at least `min_improvement` over the last
    /// `window` iterations. Tracking the best rather than the latest matters:
    /// the residual norm of conjugate gradient is not monotone, and a rule that
    /// compared consecutive values would abandon a healthy solve on an
    /// oscillation.
    ///
    /// # Errors
    ///
    /// `window == 0`, or `min_improvement` outside `(0, 1)`.
    pub fn with_stagnation(self, window: u32, min_improvement: Fix128) -> Result<Self, FemError> {
        if window == 0 {
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
            stagnation_window: window,
            stagnation_min_improvement: min_improvement,
            ..self
        })
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

    /// Iterations the best residual may go without improving before the solve
    /// is abandoned.
    #[must_use]
    pub const fn stagnation_window(&self) -> u32 {
        self.stagnation_window
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
            // 1000 iterations is longer than a healthy solve of this kind takes
            // in total (292 for a 3200-element cantilever), so the rule cannot
            // fire on a converging problem; 2^-10 ≈ 0.1% is small enough that
            // genuine progress always clears it.
            stagnation_window: 1_000,
            stagnation_min_improvement: Fix128::from_raw(0, 1 << 54),
            preconditioner: Preconditioner::JacobiScaled,
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

/// `σ = D B u_e` for one element (constant over the element).
fn element_stress(element: &Element, u: &[Fix128], lambda: Fix128, mu: Fix128) -> StressTensor {
    let mut exx = Fix128::ZERO;
    let mut eyy = Fix128::ZERO;
    let mut ezz = Fix128::ZERO;
    let mut gxy = Fix128::ZERO;
    let mut gyz = Fix128::ZERO;
    let mut gzx = Fix128::ZERO;
    for (g, &node) in element.grad.iter().zip(element.nodes.iter()) {
        let base = node * 3;
        let (ux, uy, uz) = (u[base], u[base + 1], u[base + 2]);
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
        let s = element_stress(element, u, lambda, mu);
        for (g, &node) in element.grad.iter().zip(element.nodes.iter()) {
            let base = node * 3;
            out[base] = out[base] + element.volume * (g[0] * s.xx + g[1] * s.xy + g[2] * s.zx);
            out[base + 1] =
                out[base + 1] + element.volume * (g[1] * s.yy + g[0] * s.xy + g[2] * s.yz);
            out[base + 2] =
                out[base + 2] + element.volume * (g[2] * s.zz + g[1] * s.yz + g[0] * s.zx);
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

    // Jacobi-preconditioned conjugate gradient on the free block. `x`, `r`,
    // `p` and `z` stay zero on the prescribed degrees of freedom, so the element
    // loop can run over the whole vector without a scatter/gather step.
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
    let mut diag_sum = Fix128::ZERO;
    let mut free_count = 0u32;
    for (d, value) in diag.iter().enumerate() {
        if !is_free[d] {
            continue;
        }
        if *value <= Fix128::ZERO {
            // K is positive definite on the free block, so a non-positive
            // diagonal entry means this degree of freedom is attached to no
            // element at all.
            return Err(FemError::UnderConstrained);
        }
        diag_sum = diag_sum + *value;
        free_count += 1;
    }
    // The preconditioner is `mean(diag) / diag`, not `1 / diag`.
    //
    // Scaling `M` by a constant leaves the conjugate gradient iterates
    // unchanged — `α` and `β` absorb it exactly — but it decides what magnitude
    // the inner products live at, and in fixed point that is not free. With the
    // plain reciprocal, `z = r / diag` is about 10⁴ times smaller than `r` here,
    // so the terms of `rᵀz` and `pᵀKp` fall by 10⁸ and land under the 2⁻⁶⁴
    // resolution: measured on the traction bar, `rᵀz` and `pᵀKp` both rounded to
    // exactly zero at iteration 31 with the residual still a factor of 1.6 above
    // the tolerance. Dividing by the mean keeps `z` the size of `r`, which is
    // where the available dynamic range is.
    // `free_count == 0` means every degree of freedom is prescribed. The loop
    // below then never runs (the load vector is zero, so the residual starts at
    // zero) and `precond` is never read, but the mean would divide by zero.
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

    let mut x = vec![Fix128::ZERO; ndof];
    let mut r = b.clone();
    let mut z = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            z[d] = r[d] * precond[d];
        }
    }
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let b_norm = dot(&b, &b).sqrt();
    let target = config.relative_tolerance * b_norm;

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
        if since_improvement >= config.stagnation_window {
            return Err(FemError::Stagnated {
                iterations,
                relative_residual: relative(best_residual, b_norm),
                without_improvement: since_improvement,
            });
        }
        apply_stiffness(&elements, &p, lambda, mu, &mut scratch);
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
        iterations,
        relative_residual: relative(residual_norm, b_norm),
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
