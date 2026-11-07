//! Transient small-strain FEM on P1 tetrahedra: a mass matrix and a Newmark-β
//! time integrator.
//!
//! [`crate::linear_elastic_fem`], [`crate::quadratic_elastic_fem`] and
//! [`crate::cubic_elastic_fem`] all solve `K u = f`. None of them has an `M` or
//! a `dt`, so "add Newmark" was not a missing integrator but a missing *system
//! to integrate*: this module supplies both halves, `M u'' + K u = f`.
//!
//! # MODEL — transient linear and co-rotational elasticity on P1 tetrahedra
//!
//! **Equations.** `M ü + fᵢₙₜ(u) = f` on the nodes of a tetrahedral mesh, advanced by the
//! average-acceleration Newmark step (`β = 1/4`, `γ = 1/2`): predictor
//! `ũ = u + dt·v + (dt²/4)·a`, the linear solve `(K + 4M/dt²) u' = f + (4M/dt²)·ũ`, corrector
//! `a' = (4/dt²)(u' − ũ)`, `v' = v + (dt/2)(a + a')` (`newmark_predictor`, `newmark_corrector`).
//! [`TransientSolver`]: small strain,
//! `fᵢₙₜ = K u` with the P1 stiffness `K = Σₑ ∫ Bᵀ D B dV` of isotropic Hooke's law
//! (`σ = λ tr(ε) I + 2μ ε`, `μ = E/(2(1+ν))`, `λ = Eν/((1+ν)(1−2ν))`).
//! [`CorotationalTransientSolver`]: `fᵢₙₜ(x) = Σₑ Rₑ Kₑ (Rₑᵀ x − x₀)` with `Rₑ` the polar
//! factor of the element's deformation gradient `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ` at the start of the
//! step (Müller & Gross, "Interactive Virtual Materials", GI 2004), so the step solves
//! `(Rₑ Kₑ Rₑᵀ + 4M/dt²) u' = f + (4M/dt²)·ũ − Σₑ Rₑ Kₑ (Rₑᵀ x₀ − x₀)`: one warped
//! linearisation per step, no inner Newton loop. With Rayleigh damping
//! ([`CorotationalTransientSolver::set_damping`], `C = α M + β K'`, the coefficients of
//! [`crate::damping_rayleigh`]) the equation is `M ü + C u̇ + fᵢₙₜ(u) = f` and the step solves
//! `(K' + (2/dt) C + 4M/dt²) u' = f + (4M/dt²)·ũ − C ṽ − …` with `ṽ = v + (dt/2) a − (2/dt) ũ`
//! (the trapezoidal rule on the damped system). Ground contact
//! ([`CorotationalTransientSolver::set_ground`], a plane `y = y_g` with outward normal `+y`)
//! is an active set per step, iterated to consistency: a vertex whose predictor `ũ` is at or
//! below the plane is held on it (its `y` prescribed to the plane, `v_y = a_y = 0`), the step
//! is solved, the reaction `R = (A u' − rhs)_y` each hold provides is read, the holds that
//! turned tensile (the plane cannot pull) are released and the step solved again, a released
//! vertex that would end below the plane is held again for good, until the set is unchanged
//! (`CONTACT_ACTIVE_SET_CAP` passes at most, else [`FemError::ContactUnsettled`] with the
//! state unchanged); a resting vertex's predictor is the plane exactly, so it stays held
//! until something lifts it (an upward velocity from an impulse or the body pulling up);
//! frictionless (`x`, `z` free), exact non-penetration at every step's end, no penalty
//! stiffness. `M` is the consistent Galerkin mass
//! `ρV/20 (1 + δᵢⱼ)` or its row-sum lumping `ρV/4` ([`MassLumping`]). The linear system is
//! solved by a Jacobi-preconditioned conjugate gradient in `Fix128` to a relative residual
//! ([`SolverConfig`]).
//!
//! **Units.** mm, N, MPa, s and tonne (`N = t·mm/s²`): density in t/mm³, displacements mm,
//! velocities mm/s, accelerations mm/s², loads N; every quantity is `Fix128` (Q64.64), so a
//! run replays bit for bit on every target ([`TransientSolver::state_digest`] /
//! [`CorotationalTransientSolver::state_digest`] name the state; they stand alone, the finite
//! element state is not yet in the world's snapshot or digest).
//!
//! **Conservation.** The scheme is unconditionally stable and, for a linear system,
//! conserves the discrete energy `½ vᵀMv + ½ uᵀKu` up to the rounding of the Newmark
//! coefficients (exact for a dyadic `dt`, a measured `1.5e-5` drift over 500 steps for a
//! general one before the coefficient was formed by inverting first; see `dyadic_dt` and the
//! section on the arithmetic below); without Rayleigh damping there is no dissipation, with it a mode of
//! frequency `ω` loses energy at the rate `2 ζ(ω) ω`, `ζ(ω) = α/(2ω) + βω/2`, which the
//! trapezoidal step reproduces as the per-step factor `|z|²`, `z = (1 + λ dt/2)/(1 − λ dt/2)`,
//! `λ = −ζω ± iω√(1−ζ²)` (closed form of `set_damping`, validated below). Momentum of a free
//! body follows the applied force exactly, and an external impulse `J` changes it by `J`
//! (`apply_impulses`: `M Δv = J` through the mass solve, exact for the lumped mass, to the
//! solve's tolerance for the consistent one; `momentum` reads `M v`)
//! and the free steps keep it within `n (2/dt) tol |u| M_total` (the corrector amplifies the
//! solve's displacement error into velocity). The co-rotational step conserves the same
//! quantities within one linearisation per step: a rotation as a whole produces no elastic
//! force or energy (`rotation_offset` cancels it).
//!
//! **Valid range.** Any `dt > 0` (unconditional stability, [`DynamicsConfig::try_new`]); the
//! modes resolved are those with `ω dt ≲ 1`, above that the period is elongated by
//! `ω̄ = (2/dt)·atan(ω dt/2)` with no amplitude change (`newmark_corrector`). Small strain (a
//! few per cent) in the element frame (`element_stress_local`); the co-rotational
//! form admits any rigid rotation and refuses an element whose `det F` falls to `2⁻²⁰` or
//! below ([`FemError::RotationFailed`], the state unchanged). `E > 0`, `−1 < ν < 0.5`
//! ([`ElasticMaterial::new`] refuses the rest); density and `dt` positive and
//! `4ρ/dt²` within `Fix128` ([`DynamicsConfig::try_new`]). A conjugate gradient that does
//! not reach its tolerance within its budget, or stagnates, is [`FemError::NotConverged`] /
//! [`FemError::Stagnated`], never a wrong answer; the attainable relative residual depends on
//! the mesh (the `Fix128` floor of the residual norm, [`crate::linear_elastic_fem::RESIDUAL_NORM_FLOOR`],
//! against the right-hand side's size): about `1e-9` on a 36-vertex column of 1.25 cm
//! tetrahedra (`tests/support/deformable_column.rs`), `6e-8` on the 153-vertex column of
//! `tests/analytic_deformable_integrators.rs` at one step per frame.
//!
//! **Approximations.** P1 (constant strain) tetrahedra (`build_elements`); Hooke's law in the
//! element frame (no hyperelastic law, no plasticity, no thermal strain here,
//! `element_stress_local`); the co-rotational frame is frozen over a step (first order in the
//! rotation increment per step, `rotations` at the step's start); damping is Rayleigh's
//! (viscous, `α M + β K'`, no hysteretic or frictional loss, see `crate::damping_rayleigh`)
//! or none; contact is a single plane, frictionless, resolved to the step's end (no
//! time-of-impact subdivision, a vertex arriving mid-step is held from the end of that step),
//! no restitution (the hold is inelastic, `set_ground`), no adhesion (a tensile hold is
//! released within the step); external impulses are applied between
//! steps, not inside one (`apply_impulses`); the lumped mass lowers the frequencies and the
//! consistent one raises them relative to the continuum (Rayleigh quotient on the trial
//! space, [`MassLumping`]); loads and prescribed displacements are constant over a run
//! ([`BoundaryConditions`] fixed at `new`); the tetrahedra are
//! the caller's, and a mesh from [`crate::sdf_fem_mesh::generate_marching_tets`] carries that
//! mesher's own errors (the chamfer of flat faces and the overfill at reflex edges, its MODEL's
//! known defects) into the mass, the stiffness and the contact area here.
//!
//! **Validation.** `tests/analytic_dynamic_fem.rs` (a free body under a total force follows
//! the closed-form parabola, a single degree of freedom tracks the closed-form step response,
//! the two mass distributions differ by the closed-form ratio, the axial bar's peak arrives
//! at the closed-form wave transit time and scales as the square root of density, prescribed
//! displacements are held exactly, degenerate configurations are refused),
//! `tests/analytic_dynamic_fem_corotational.rs::a_rigidly_rotated_body_feels_no_elastic_force`
//! (a column rotated by 30° and released does not move beyond what the solves' residual
//! guarantee allows, measured `9e-8 mm` against a bound of `1.3e-2 mm`),
//! `::the_corotational_oscillation_has_the_modal_period_and_the_static_mean_at_three_step_sizes`
//! and `::the_linear_newmark_oscillation_has_the_modal_period_and_the_static_mean_at_three_step_sizes`
//! (at `dt = T₁/32, /64, /128`: the first axial modal coordinate of a column under its own
//! weight oscillates at `2π/ω̄` of the mesh's own `ω₁` from inverse iteration on
//! `K φ = ω² M φ`, the co-rotational one within the range the warping spans, and the means
//! of `q₁` and of the top sag are the static ones; every tolerance is computed in the test
//! from the crossing interpolation, the solves' residual guarantee, the Dirichlet kernel of
//! the window and two static solves),
//! `::the_discrete_axial_mode_lies_above_the_thin_rod_frequency_by_at_most_the_base_layer_share`
//! (the discrete mode against `(π/2H)√(E/ρ)` with the rod's assumptions stated),
//! `::stiffness_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate_at_three_step_sizes`
//! and `::mass_proportional_damping_decays_the_first_mode_at_the_trapezoidal_rate_at_three_step_sizes`
//! (`ζ = 0.05` by `β` alone and by `α` alone at the three step sizes: the three-term
//! recurrence fitted to `q₁` gives the per-step decay and phase, checked against the
//! trapezoidal closed form over the warping's frequency range, the tolerance propagated
//! from the solves' residual guarantee through the least squares),
//! `::a_column_standing_on_the_ground_settles_to_the_roller_static_sag_with_its_whole_bottom_in_contact`
//! (no base fix, the plane under it, three step sizes: the sag is the static solve of the
//! same mesh with a roller base within the decay the trapezoidal map allows the modes,
//! `max_ω |z(ω)|ⁿ/ω² ‖e_top‖_{M⁻¹} ‖f‖_{M⁻¹}`, measured +1e-4; every bottom vertex held at
//! every step; the closed form `ρ g H² / (2E)` is the O(h) limit, 0.6 % away on this mesh),
//! `::a_dropped_column_never_penetrates_the_ground_and_settles_to_the_roller_static_sag` (a
//! 10 mm drop at three step sizes: no step ends below the plane, touchdown within one step
//! of `√(2h/g)`, and from the final contact the sag over the last two periods is the roller
//! static within the bound the state at contact and the modes' decay allow), `::an_impulse_on_a_free_lumped_body_is_its_momentum_exactly_and_is_kept` and
//! `::an_impulse_on_a_free_consistent_body_is_its_momentum_to_the_solve_tolerance` (an oblique
//! impulse on the top of a free column: `Σ M v = J` to 1e-12 with the lumped mass, to the solve
//! tolerance with the consistent one, and the momentum and centre-of-mass velocity are kept
//! over 200 free steps within the derived bound),
//! `::an_impulse_on_a_held_vertex_releases_it_upward_and_is_taken_by_the_ground_downward`,
//! `::a_column_kicked_upward_leaves_the_plane_with_every_bottom_vertex_released` (the hold is
//! not adhesive: an upward impulse on the top of a standing column lifts the whole bottom
//! off the plane within the flight time, every vertex free, and it lands again),
//! `::two_runs_of_the_same_inputs_replay_to_the_same_bits_and_digest` (the linear and the
//! co-rotational solver, with damping, ground, impulses: identical raw bits and digests, a
//! changed input changes the digest), `tests/audit_dynamic_fem.rs`.
//!
//! **Tiers.** L0 of a stiff elastic solid is the implicit co-rotational step
//! ([`CorotationalTransientSolver`]): measured on the same Kuhn-split column of 1.25 cm
//! tetrahedra (`E = 1.27e6 Pa`, `ν = 0.49`, `ρ = 1100 kg/m³`,
//! `tests/analytic_deformable_integrators.rs`) it reaches 0.6–1 % of the static sag at one
//! step per frame for 0.0002 s (24 tetrahedra) to 0.026 s (384) per frame, where the XPBD
//! Neo-Hookean body of [`crate::deformable`] needs 256 substeps per frame for 2.8 % at 17 to
//! 100 times the cost. No lighter tier yet (a coarse cage with the fine surface embedded is
//! the candidate; its error against this L0 will be measured by the same sag and period
//! problems).
//!
//! # Units
//!
//! Millimetre / newton / megapascal as in [`crate::linear_elastic_fem`], plus
//! seconds for `dt` — which fixes the mass unit. `N = M·a` with length in mm
//! and time in s gives `[M] = N·s²/mm = 10³ kg`, the **tonne**, so
//! [`DynamicsConfig::try_new`] wants a density in **tonne/mm³**: steel is
//! `7.85e-9`, PLA about `1.24e-9`. Handing it `7850` (kg/m³) produces a system
//! wrong by twelve orders of magnitude that converges perfectly happily, which
//! is why the constructor can only check the sign.
//!
//! # Method
//!
//! Newmark with `β = 1/4, γ = 1/2` (the average-acceleration, or trapezoidal,
//! rule). With the predictor
//!
//! ```text
//! ũ = uₙ + dt·vₙ + (dt²/4)·aₙ
//! ```
//!
//! a step is one linear solve and two updates:
//!
//! ```text
//! (K + Ṁ) uₙ₊₁ = f + Ṁ ũ        with  Ṁ = M/(β dt²) = 4M/dt²
//! aₙ₊₁ = (4/dt²)(uₙ₊₁ − ũ)
//! vₙ₊₁ = vₙ + (dt/2)(aₙ + aₙ₊₁)
//! ```
//!
//! `K + Ṁ` is applied element by element in mesh order, exactly as
//! [`crate::linear_elastic_fem`] applies `K`; the global matrix is never
//! formed. The same Jacobi-preconditioned conjugate gradient runs on it.
//!
//! # Why `β = 1/4, γ = 1/2` is fixed rather than configurable
// LIMITATION(COV-FEM-073): # Why `β = 1/4, γ = 1/2` is fixed rather than configurable
//!
//! Three reasons, and the first is specific to this crate's arithmetic.
//!
//! 1. **Every coefficient is a power of two when `dt` is.** `1/(β dt²) = 4/dt²`,
//!    `1/(β dt) = 4/dt`, `1/(2β) − 1 = 1`, `1/2 − β = 1/4`, `1 − γ = γ = 1/2`.
//!    Measured over `dt = 2⁻¹ … 2⁻¹⁰`, every one of them is bit-exact in
//!    [`Fix128`]. The linear-acceleration method `β = 1/6` has `1/2 − β = 1/3`,
//!    which is not dyadic and rounds every step; being *rational* is not the
//!    property that matters in a binary fixed-point format. A `dt` that is not
//!    a power of two is fine too, but only because the coefficients are built
//!    to avoid small intermediates — see [`DynamicsConfig::try_new`].
//! 2. `γ = 1/2` is the only choice without algorithmic damping. `γ > 1/2`
//!    bleeds amplitude every step, which is indistinguishable from a bug when
//!    the oracle is a conservative one.
//! 3. `β = 1/4` is unconditionally stable and second-order accurate, so `dt` is
//!    chosen for accuracy rather than against a stability limit the caller
//!    would have to compute from the mesh.
//!
//! # What the arithmetic does and does not conserve
//!
//! In exact arithmetic the trapezoidal rule conserves `E = ½vᵀMv + ½uᵀKu`
//! identically, and the single-degree-of-freedom amplification matrix has
//! modulus exactly one. Neither survives truncation. Measured on a
//! single-degree-of-freedom oscillator over 4096 steps, `E` wanders by up to
//! `9.3e5` raw units of `2⁻⁶⁴` — about `1e-16` relative — and the amplitude
//! ends slightly *below* where it started, never above, because [`Fix128`]
//! multiplication truncates toward zero. **An `assert_eq!` on conserved energy
//! is therefore not available.** One case escapes: when `ω·dt = 2` the discrete
//! orbit is the four-cycle `x₀, 0, −x₀, 0` through integers, and that one is
//! bit-exact for as long as it is run. `tests/analytic_dynamic_fem.rs` pins
//! both the exact case and a measured bound for the general one.
//!
//! # Limitations
//!
//! - No damping. There is no `C u'` term, so [`crate::damping_rayleigh`] still
//!   has nothing to attach to.
//! - The load is fixed at construction. A time-varying `f(t)` needs a second
//!   entry point, not a flag.
//! - Small strain, as in [`crate::linear_elastic_fem`]; the co-rotational and
//!   hyperelastic paths there are static and are not reused here.
//! - P1 only. The P2 and P3 modules keep their own element arithmetic and so
//!   does this one.
//!
//! Author: Moroya Sakamoto

use crate::damping_rayleigh::RayleighCoefficients;
use crate::linear_elastic_fem::{
    BoundaryConditions, ElasticMaterial, FemError, Preconditioner, SolverConfig, StressTensor,
    RESIDUAL_NORM_FLOOR,
};
use crate::math::{Fix128, Mat3Fix, Vec3Fix};
use crate::sdf_fem_mesh::SdfTetMesh;

// ---------------------------------------------------------------------------
// element geometry (P1) — the same arithmetic as `linear_elastic_fem`
// ---------------------------------------------------------------------------

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

/// Constant per-element quantities: shape function gradients (1/mm) and volume
/// (mm³).
#[derive(Clone, Copy)]
struct Element {
    nodes: [usize; 4],
    grad: [[Fix128; 3]; 4],
    volume: Fix128,
}

/// Precompute `∇N` and `|V|` for every tetrahedron.
///
/// Same construction as `linear_elastic_fem`: the rows of `J⁻¹` are the
/// gradients of `N₁, N₂, N₃` and `∇N₀ = −(∇N₁+∇N₂+∇N₃)`, with `|det J| / 6` as
/// the volume so that winding does not change the stiffness.
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

#[inline]
fn gather(element: &Element, u: &[Fix128]) -> [[Fix128; 3]; 4] {
    let mut local = [[Fix128::ZERO; 3]; 4];
    for (slot, &node) in local.iter_mut().zip(element.nodes.iter()) {
        let base = node * 3;
        *slot = [u[base], u[base + 1], u[base + 2]];
    }
    local
}

/// `σ = D B u_e` for one element, from its four nodal vectors.
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

/// `∫ Bᵀ σ dV` for one element.
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

// ---------------------------------------------------------------------------
// mass
// ---------------------------------------------------------------------------

/// How an element's mass is distributed over its nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MassLumping {
    /// The Galerkin mass matrix `Mₑ = ρ∫NᵢNⱼ dV = (ρV/20)(1 + δᵢⱼ)`.
    ///
    /// Exact, with no quadrature rule involved: the barycentric moment
    /// `∫λ₁^a λ₂^b λ₃^c λ₄^d dV = a!b!c!d!·3!/(a+b+c+d+3)!·V` gives `V/20` off
    /// the diagonal and `V/10` on it. That matters here, because `∫NᵢNⱼ` is a
    /// degree-2 integrand where the stiffness `∫BᵀDB` is degree 0 — a
    /// quadrature-based mass would have needed a richer rule than anything the
    /// P1 stiffness uses. The resulting discrete frequencies bound the
    /// continuum ones from above (a Rayleigh quotient over a restricted trial
    /// space), where [`Self::Lumped`] under-predicts them.
    #[default]
    Consistent,
    /// Row-summed — for P1, equivalently equal-split — `ρV/4` per node.
    ///
    /// Diagonal, so `M u` costs one multiply per degree of freedom, and the
    /// division by four is exact in binary where the consistent division by
    /// twenty is not. The off-diagonal coupling is discarded, which lowers the
    /// frequencies.
    Lumped,
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

/// Density, time step and mass distribution for a transient solve.
///
/// Fields are private and validated on construction, for the same reason
/// [`crate::linear_elastic_fem::SolverConfig`]'s are: a struct literal gets no
/// validation, and a non-positive `dt` divides by zero two layers down. There
/// is deliberately no `Default`, because there is no density that is right for
/// an unknown material and a silently wrong one changes every frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicsConfig {
    density: Fix128,
    dt: Fix128,
    /// `4/dt²`, the factor relating `M` to its appearance in `K + M/(βdt²)`.
    inv_beta_dt2: Fix128,
    lumping: MassLumping,
}

impl DynamicsConfig {
    /// Validate a density (tonne/mm³), a time step (s) and a mass distribution.
    ///
    /// # Choosing `dt`
    ///
    /// Any positive value. The integrator is unconditionally stable, so `dt` is
    /// bounded only by the period of the highest mode that has to be resolved —
    /// `2π/ω` with `ω ≈ (π/h)√(E/ρ)` for an element of size `h`.
    ///
    /// A power of two is still the nicer choice, because then every Newmark
    /// coefficient is exactly representable and the arithmetic is the same on
    /// every target. But a step that is not a power of two is **not** penalised,
    /// and that took measuring: an earlier draft formed `4/(dt·dt)` and lost
    /// `1.5e-5` of the amplitude over 500 steps at `dt = 1/6 000 000`. See the
    /// comment on the construction below for why, and
    /// `tests/analytic_dynamic_fem.rs::forming_the_coefficient_by_inverting_first_is_what_makes_any_step_safe`
    /// for the measurement and for the regression guard that keeps it true.
    ///
    /// # Errors
    ///
    /// [`FemError::InvalidConfig`] when `density` or `dt` is not positive, or
    /// when `2/dt` or `(2/dt)²` leaves the range of [`Fix128`] — the latter
    /// below roughly `dt = 2⁻³¹ s`, where the step count needed to reach any
    /// useful time is out of reach anyway.
    pub fn try_new(density: Fix128, dt: Fix128, lumping: MassLumping) -> Result<Self, FemError> {
        if density <= Fix128::ZERO {
            return Err(FemError::InvalidConfig("density must be positive"));
        }
        if dt <= Fix128::ZERO {
            return Err(FemError::InvalidConfig("dt must be positive"));
        }
        // `(2/dt)·(2/dt)` and **not** `4/(dt·dt)`.
        //
        // `Fix128` has a fixed absolute resolution of `2⁻⁶⁴`, so relative
        // precision is `2⁻⁶⁴/value` — the smaller the quantity, the worse it is
        // held. Forming `dt²` puts the smallest quantity in the whole
        // computation on the stack: at `dt = 1/6 000 000`, `dt² = 2.78e-14` has
        // only 19 bits left, and `4/dt²` inherits a relative error of
        // `2⁻⁶⁴/dt² = 1.9e-6`. That error is a bias on the algorithmic
        // frequency, so the phase it costs accumulates *linearly* in the step
        // count: measured, 500 steps left the closed form by `1.5e-5` of the
        // amplitude. **Refining `dt` makes it worse**, which is the opposite of
        // what a time step usually does.
        //
        // `2/dt` is the same number upside down — large, so held to `2⁻⁶⁴/(2/dt)`
        // — and squaring it keeps that. For a dyadic `dt` both forms are exact
        // and identical, so nothing is traded away for the non-dyadic case.
        let two_over_dt = Fix128::from_int(2)
            .checked_div(dt)
            .ok_or(FemError::InvalidConfig("2/dt overflows Fix128"))?;
        if two_over_dt.is_zero() {
            return Err(FemError::InvalidConfig("2/dt underflows to zero"));
        }
        let inv_beta_dt2 = two_over_dt
            .checked_mul(two_over_dt)
            .ok_or(FemError::InvalidConfig("4/dt squared overflows Fix128"))?;
        Ok(Self {
            density,
            dt,
            inv_beta_dt2,
            lumping,
        })
    }
}

// ---------------------------------------------------------------------------
// operators
// ---------------------------------------------------------------------------

/// Everything a step needs that does not change with time.
struct Operators {
    elements: Vec<Element>,
    lambda: Fix128,
    mu: Fix128,
    /// Per element, the scalar multiplying the element's mass pattern, already
    /// carrying the `4/dt²` of the effective stiffness.
    ///
    /// Consistent: `4ρV/(20 dt²)`, so `Ṁₑ u = scaleₑ·(Σⱼuⱼ + uᵢ)`.
    /// Lumped: `4ρV/(4 dt²)`, so `Ṁₑ u = scaleₑ·uᵢ`.
    mass_scale: Vec<Fix128>,
    lumping: MassLumping,
}

impl Operators {
    fn new(
        elements: Vec<Element>,
        material: &ElasticMaterial,
        dynamics: &DynamicsConfig,
    ) -> Result<Self, FemError> {
        let (lambda, mu) = material.lame();
        // Scale first, divide once, multiply by each volume last. `ρ` is around
        // `1e-9` in these units and `4/dt²` around `1e12`; forming their product
        // before touching the volume keeps the quantity that gets rounded at
        // full magnitude instead of a few bits above the `2⁻⁶⁴` floor.
        let rho_c0 = dynamics
            .density
            .checked_mul(dynamics.inv_beta_dt2)
            .ok_or(FemError::InvalidConfig("4 rho / dt squared overflows"))?;
        let divisor = match dynamics.lumping {
            MassLumping::Consistent => Fix128::from_int(20),
            MassLumping::Lumped => Fix128::from_int(4),
        };
        let per_volume = rho_c0 / divisor;
        if per_volume.is_zero() {
            return Err(FemError::InvalidConfig(
                "4 rho / dt squared underflows: the mass term would vanish",
            ));
        }
        let mut mass_scale = Vec::with_capacity(elements.len());
        for element in &elements {
            let scale = per_volume
                .checked_mul(element.volume)
                .ok_or(FemError::InvalidConfig("an element mass overflows Fix128"))?;
            if scale.is_zero() {
                return Err(FemError::InvalidConfig(
                    "an element mass underflows to zero",
                ));
            }
            mass_scale.push(scale);
        }
        Ok(Self {
            elements,
            lambda,
            mu,
            mass_scale,
            lumping: dynamics.lumping,
        })
    }

    /// The four nodal contributions of `Ṁₑ u_e` for one element.
    #[inline]
    fn element_mass_force(
        lumping: MassLumping,
        scale: Fix128,
        local: &[[Fix128; 3]; 4],
    ) -> [[Fix128; 3]; 4] {
        let mut out = [[Fix128::ZERO; 3]; 4];
        match lumping {
            MassLumping::Lumped => {
                for (slot, node_u) in out.iter_mut().zip(local.iter()) {
                    for axis in 0..3 {
                        slot[axis] = scale * node_u[axis];
                    }
                }
            }
            MassLumping::Consistent => {
                // `(1 + δᵢⱼ)` contracted with `u` is `Σⱼuⱼ` for every row, plus
                // the row's own value a second time.
                let mut total = [Fix128::ZERO; 3];
                for node_u in local {
                    for axis in 0..3 {
                        total[axis] = total[axis] + node_u[axis];
                    }
                }
                for (slot, node_u) in out.iter_mut().zip(local.iter()) {
                    for axis in 0..3 {
                        slot[axis] = scale * (total[axis] + node_u[axis]);
                    }
                }
            }
        }
        out
    }

    /// The diagonal entry of `Ṁₑ` (the same on all three axes).
    #[inline]
    fn element_mass_diagonal(lumping: MassLumping, scale: Fix128) -> Fix128 {
        match lumping {
            MassLumping::Lumped => scale,
            // `(1 + δᵢᵢ) = 2`.
            MassLumping::Consistent => scale + scale,
        }
    }

    /// `out = Ṁ u`, accumulated element by element in mesh order.
    fn apply_mass(&self, u: &[Fix128], out: &mut [Fix128]) {
        out.fill(Fix128::ZERO);
        for (element, &scale) in self.elements.iter().zip(self.mass_scale.iter()) {
            let local = gather(element, u);
            let force = Self::element_mass_force(self.lumping, scale, &local);
            for (f, &node) in force.iter().zip(element.nodes.iter()) {
                let base = node * 3;
                for axis in 0..3 {
                    out[base + axis] = out[base + axis] + f[axis];
                }
            }
        }
    }

    /// `out = (K + Ṁ) u`, one element loop in mesh order.
    fn apply_effective(&self, u: &[Fix128], out: &mut [Fix128]) {
        out.fill(Fix128::ZERO);
        for (element, &scale) in self.elements.iter().zip(self.mass_scale.iter()) {
            let local = gather(element, u);
            let stress = element_stress_local(element, &local, self.lambda, self.mu);
            let stiffness = element_force_from_stress(element, stress);
            let mass = Self::element_mass_force(self.lumping, scale, &local);
            for ((k, m), &node) in stiffness.iter().zip(mass.iter()).zip(element.nodes.iter()) {
                let base = node * 3;
                for axis in 0..3 {
                    out[base + axis] = out[base + axis] + k[axis] + m[axis];
                }
            }
        }
    }

    /// `diag(Ṁ)`, the preconditioner for the initial-acceleration solve.
    fn mass_diagonal(&self, ndof: usize) -> Vec<Fix128> {
        let mut diag = vec![Fix128::ZERO; ndof];
        for (element, &scale) in self.elements.iter().zip(self.mass_scale.iter()) {
            let m = Self::element_mass_diagonal(self.lumping, scale);
            for &node in &element.nodes {
                let base = node * 3;
                for axis in 0..3 {
                    diag[base + axis] = diag[base + axis] + m;
                }
            }
        }
        diag
    }

    /// `diag(K + Ṁ)`.
    fn effective_diagonal(&self, ndof: usize) -> Vec<Fix128> {
        let mut diag = self.mass_diagonal(ndof);
        let lambda_2mu = self.lambda + self.mu + self.mu;
        for element in &self.elements {
            for (g, &node) in element.grad.iter().zip(element.nodes.iter()) {
                let sq = [g[0] * g[0], g[1] * g[1], g[2] * g[2]];
                let base = node * 3;
                for axis in 0..3 {
                    let others = sq[(axis + 1) % 3] + sq[(axis + 2) % 3];
                    diag[base + axis] = diag[base + axis]
                        + element.volume * (lambda_2mu * sq[axis] + self.mu * others);
                }
            }
        }
        diag
    }
}

// ---------------------------------------------------------------------------
// conjugate gradient
// ---------------------------------------------------------------------------

fn dot(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc = acc + *x * *y;
    }
    acc
}

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
    let mean_diag = if free_count == 0 {
        Fix128::ONE
    } else {
        diag_sum / Fix128::from_int(i64::from(free_count))
    };
    let mut precond = vec![Fix128::ZERO; ndof];
    for d in 0..ndof {
        if is_free[d] {
            precond[d] = match config.preconditioner() {
                Preconditioner::JacobiScaled => mean_diag / diag[d],
                _ => Fix128::ONE,
            };
        }
    }
    Ok(precond)
}

/// Preconditioned conjugate gradient on the free block.
///
/// Same stopping rule, stagnation window and residual floor as
/// [`crate::linear_elastic_fem`]. The operator arrives as a closure because two
/// systems need it here: `Ṁ` for the initial acceleration and `K + Ṁ` for every
/// step.
fn conjugate_gradient<A>(
    b: &[Fix128],
    is_free: &[bool],
    precond: &[Fix128],
    config: &SolverConfig,
    mut apply: A,
) -> Result<(Vec<Fix128>, u32), FemError>
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
    let requested = config.relative_tolerance() * b_norm;
    let target = if requested > RESIDUAL_NORM_FLOOR {
        requested
    } else {
        RESIDUAL_NORM_FLOOR
    };

    let mut iterations = 0u32;
    let mut residual_norm = dot(&r, &r).sqrt();
    let mut best_residual = residual_norm;
    let mut since_improvement = 0u32;

    while residual_norm > target {
        if iterations >= config.max_iterations() {
            return Err(FemError::NotConverged {
                iterations,
                relative_residual: relative(residual_norm, b_norm),
            });
        }
        let window = {
            let scaled =
                config.stagnation_window_fraction() * Fix128::from_int(i64::from(iterations));
            let scaled = if scaled.is_negative() {
                0
            } else {
                u32::try_from(scaled.hi).unwrap_or(u32::MAX)
            };
            scaled.max(config.stagnation_min_window())
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
                return Err(FemError::UnderConstrained);
            }
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

        if residual_norm < best_residual - best_residual * config.stagnation_min_improvement() {
            best_residual = residual_norm;
            since_improvement = 0;
        } else {
            if residual_norm < best_residual {
                best_residual = residual_norm;
            }
            since_improvement += 1;
        }
    }

    Ok((x, iterations))
}

#[inline]
fn relative(residual_norm: Fix128, b_norm: Fix128) -> Fix128 {
    if b_norm.is_zero() {
        Fix128::ZERO
    } else {
        residual_norm / b_norm
    }
}

// ---------------------------------------------------------------------------
// the Newmark recurrence, one degree of freedom at a time
// ---------------------------------------------------------------------------
//
// These two are scalar on purpose. The step below applies them across the free
// block, and the single-degree-of-freedom oracles drive the same two functions
// with a 1×1 system, so the coefficients those oracles pin are the ones the
// solver uses rather than a second copy written to agree with it.

/// `ũ = u + dt·v + (dt/2)·((dt/2)·a)` — the `β = 1/4` predictor.
///
/// The acceleration term is `(dt²/4)·a`, applied as two multiplications by
/// `dt/2` rather than as one by a precomputed `dt²/4`, for the reason spelled
/// out on [`DynamicsConfig::try_new`]: `dt²/4` is the smallest quantity in the
/// method and `Fix128` holds small quantities badly. Halving is a right shift
/// and therefore exact for any `dt`, so each rounding here happens at the scale
/// of its own result instead of at the scale of a tiny constant.
///
/// For a dyadic `dt` the two forms agree bit for bit:
/// `floor(floor(x/2ⁱ)/2ʲ) = floor(x/2^(i+j))`.
#[inline]
fn newmark_predictor(u: Fix128, v: Fix128, a: Fix128, dt: Fix128, half_dt: Fix128) -> Fix128 {
    u + dt * v + half_dt * (half_dt * a)
}

/// `a' = (4/dt²)(u' − ũ)` and `v' = v + (dt/2)(a + a')` — the `γ = 1/2`
/// corrector, returned as `(v', a')`.
#[inline]
fn newmark_corrector(
    next_u: Fix128,
    predictor: Fix128,
    v: Fix128,
    a: Fix128,
    inv_beta_dt2: Fix128,
    half_dt: Fix128,
) -> (Fix128, Fix128) {
    let next_a = (next_u - predictor) * inv_beta_dt2;
    (v + half_dt * (a + next_a), next_a)
}

// ---------------------------------------------------------------------------
// the solver
// ---------------------------------------------------------------------------

/// A transient small-strain FEM solve, advanced one Newmark step at a time.
///
/// Built from the same mesh, material and boundary conditions as
/// [`crate::linear_elastic_fem::solve`], plus a [`DynamicsConfig`]. The nodal
/// loads are read once and held constant; the prescribed displacements are held
/// constant too, with zero velocity and acceleration on those degrees of
/// freedom.
///
/// Unlike the static solve, this one does **not** require six constraints: `Ṁ`
/// is positive definite on its own, so an entirely unconstrained body is a
/// well-posed problem here and simply accelerates under its load.
///
/// # Example
///
/// ```no_run
/// use alice_physics_core::dynamic_fem::{DynamicsConfig, MassLumping, TransientSolver};
/// use alice_physics_core::linear_elastic_fem::{BoundaryConditions, ElasticMaterial, FemError, SolverConfig};
/// use alice_physics_core::sdf_fem_mesh::SdfTetMesh;
/// use alice_physics_core::Fix128;
///
/// # fn run(mesh: &SdfTetMesh) -> Result<(), FemError> {
/// let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_f64(0.35))?;
/// let dynamics = DynamicsConfig::try_new(
///     Fix128::from_f64(1.24e-9),      // PLA, tonne/mm³
///     Fix128::from_raw(0, 1 << 44),   // dt = 2⁻²⁰ s
///     MassLumping::Consistent,
/// )?;
/// let mut solver = TransientSolver::new(
///     mesh,
///     &material,
///     &BoundaryConditions::new(),
///     &dynamics,
///     &SolverConfig::default(),
/// )?;
/// for _ in 0..1000 {
///     solver.step()?;
/// }
/// let tip_x = solver.displacements()[0];
/// # let _ = tip_x;
/// # Ok(())
/// # }
/// ```
pub struct TransientSolver {
    ops: Operators,
    dt: Fix128,
    inv_beta_dt2: Fix128,
    /// `dt/2`. Exact for any `dt` (a right shift), and used both as the
    /// velocity update weight (`γ = 1 − γ = 1/2`) and, applied twice, as the
    /// predictor's acceleration weight `dt²/4`.
    half_dt: Fix128,
    is_free: Vec<bool>,
    prescribed_value: Vec<Fix128>,
    load: Vec<Fix128>,
    precond: Vec<Fix128>,
    config: SolverConfig,
    u: Vec<Fix128>,
    v: Vec<Fix128>,
    a: Vec<Fix128>,
}

impl TransientSolver {
    /// Assemble the operators and the consistent initial state.
    ///
    /// The initial displacement is the prescribed field (zero on the free
    /// degrees of freedom) and the initial velocity is zero; the initial
    /// acceleration then solves `M a₀ = f − K u₀`, so the equation of motion
    /// holds at `t = 0` instead of being violated by the first step.
    ///
    /// # Errors
    ///
    /// [`FemError::EmptyMesh`], [`FemError::VertexOutOfRange`] and
    /// [`FemError::DegenerateElement`] from the mesh, [`FemError::InvalidConfig`]
    /// when a mass over- or underflows, and [`FemError::NotConverged`] or
    /// [`FemError::Stagnated`] from the initial-acceleration solve.
    pub fn new(
        mesh: &SdfTetMesh,
        material: &ElasticMaterial,
        boundary: &BoundaryConditions,
        dynamics: &DynamicsConfig,
        config: &SolverConfig,
    ) -> Result<Self, FemError> {
        let vertex_count = mesh.vertices.len();
        if vertex_count == 0 || mesh.tets.is_empty() {
            return Err(FemError::EmptyMesh);
        }
        for &(vertex, _, _) in boundary.prescribed().iter().chain(boundary.loads().iter()) {
            if vertex as usize >= vertex_count {
                return Err(FemError::VertexOutOfRange {
                    vertex,
                    vertex_count,
                });
            }
        }
        let elements = build_elements(mesh)?;
        let ops = Operators::new(elements, material, dynamics)?;
        let ndof = vertex_count * 3;

        let mut prescribed_value = vec![Fix128::ZERO; ndof];
        let mut is_free = vec![true; ndof];
        for &(vertex, axis, value) in boundary.prescribed() {
            let d = vertex as usize * 3 + axis.index();
            is_free[d] = false;
            prescribed_value[d] = value;
        }
        let mut load = vec![Fix128::ZERO; ndof];
        for &(vertex, axis, force) in boundary.loads() {
            let d = vertex as usize * 3 + axis.index();
            if is_free[d] {
                load[d] = load[d] + force;
            }
        }

        let u = prescribed_value.clone();
        let v = vec![Fix128::ZERO; ndof];

        // a₀ from `M a₀ = f − K u₀`. `apply_effective` is `K + Ṁ`, so `K u₀` is
        // recovered by subtracting `Ṁ u₀` rather than by carrying a third
        // operator that could drift from the other two.
        let mut keff_u0 = vec![Fix128::ZERO; ndof];
        ops.apply_effective(&u, &mut keff_u0);
        let mut mass_u0 = vec![Fix128::ZERO; ndof];
        ops.apply_mass(&u, &mut mass_u0);
        let mut rhs = vec![Fix128::ZERO; ndof];
        for d in 0..ndof {
            rhs[d] = if is_free[d] {
                load[d] - (keff_u0[d] - mass_u0[d])
            } else {
                Fix128::ZERO
            };
        }
        let mass_diag = ops.mass_diagonal(ndof);
        let mass_precond = build_preconditioner(&mass_diag, &is_free, config)?;
        // `Ṁ = c₀ M`, so solving `Ṁ y = rhs` returns `y = a₀/c₀`.
        let (scaled_a, _) = conjugate_gradient(&rhs, &is_free, &mass_precond, config, |p, out| {
            ops.apply_mass(p, out);
        })?;
        let a = scaled_a
            .iter()
            .map(|x| *x * dynamics.inv_beta_dt2)
            .collect::<Vec<_>>();

        let diag = ops.effective_diagonal(ndof);
        let precond = build_preconditioner(&diag, &is_free, config)?;

        Ok(Self {
            ops,
            dt: dynamics.dt,
            inv_beta_dt2: dynamics.inv_beta_dt2,
            half_dt: dynamics.dt.half(),
            is_free,
            prescribed_value,
            load,
            precond,
            config: *config,
            u,
            v,
            a,
        })
    }

    /// Advance one Newmark step. Returns the conjugate gradient iterations used.
    ///
    /// # Errors
    ///
    /// [`FemError::NotConverged`] or [`FemError::Stagnated`] from the linear
    /// solve, with the same meanings as in [`crate::linear_elastic_fem`].
    pub fn step(&mut self) -> Result<u32, FemError> {
        let ndof = self.u.len();
        // ũ = u + dt·v + (dt²/4)·a, with the prescribed field held fixed.
        let mut predictor = vec![Fix128::ZERO; ndof];
        for (d, slot) in predictor.iter_mut().enumerate() {
            *slot = if self.is_free[d] {
                newmark_predictor(self.u[d], self.v[d], self.a[d], self.dt, self.half_dt)
            } else {
                self.prescribed_value[d]
            };
        }
        let mut mass_predictor = vec![Fix128::ZERO; ndof];
        self.ops.apply_mass(&predictor, &mut mass_predictor);
        // `(K + Ṁ) u_{n+1} = f + Ṁũ`, moved onto the free block by subtracting
        // the prescribed column.
        let mut keff_prescribed = vec![Fix128::ZERO; ndof];
        self.ops
            .apply_effective(&self.prescribed_value, &mut keff_prescribed);
        let mut b = vec![Fix128::ZERO; ndof];
        for d in 0..ndof {
            b[d] = if self.is_free[d] {
                self.load[d] + mass_predictor[d] - keff_prescribed[d]
            } else {
                Fix128::ZERO
            };
        }
        let ops = &self.ops;
        let (mut next_u, iterations) =
            conjugate_gradient(&b, &self.is_free, &self.precond, &self.config, |p, out| {
                ops.apply_effective(p, out);
            })?;
        for d in 0..ndof {
            if self.is_free[d] {
                let (next_v, next_a) = newmark_corrector(
                    next_u[d],
                    predictor[d],
                    self.v[d],
                    self.a[d],
                    self.inv_beta_dt2,
                    self.half_dt,
                );
                self.v[d] = next_v;
                self.a[d] = next_a;
            } else {
                next_u[d] = self.prescribed_value[d];
            }
        }
        self.u = next_u;
        Ok(iterations)
    }

    /// Nodal displacements (mm), three components per vertex in vertex order.
    #[must_use]
    pub fn displacements(&self) -> &[Fix128] {
        &self.u
    }

    /// Nodal velocities (mm/s), laid out like [`Self::displacements`].
    #[must_use]
    pub fn velocities(&self) -> &[Fix128] {
        &self.v
    }

    /// Nodal accelerations (mm/s²), laid out like [`Self::displacements`].
    #[must_use]
    pub fn accelerations(&self) -> &[Fix128] {
        &self.a
    }

    /// Domain separation of [`Self::state_digest`]: the SHA-256 of the text
    /// `alice-physics/dynamic-fem-state-digest/v1`, computed once; the bytes, not the text,
    /// are hashed first.
    pub const STATE_DIGEST_DOMAIN: [u8; 32] = [
        0x4b, 0x8d, 0xd0, 0x67, 0x23, 0xde, 0xb2, 0x07, 0x64, 0x13, 0x07, 0xc2, 0x3a, 0xd4, 0x26,
        0xd3, 0x94, 0x18, 0x26, 0x7f, 0xfe, 0x83, 0x0a, 0x75, 0x19, 0xf5, 0x9d, 0x0c, 0x28, 0x43,
        0x46, 0x9d,
    ];

    /// SHA-256 of the solver's state: `STATE_DIGEST_DOMAIN`, then the raw `Fix128` bits of
    /// every displacement, velocity and acceleration (little-endian `i64` high, `u64` low,
    /// degree of freedom by degree of freedom). Two runs agree exactly when their states
    /// are the same to the bit, which is what a replay of the same inputs gives on every
    /// target. This digest stands alone: the finite element state is not yet part of
    /// [`crate::PhysicsWorld::state_digest`] or its snapshot.
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(Self::STATE_DIGEST_DOMAIN);
        for x in self.u.iter().chain(&self.v).chain(&self.a) {
            h.update(x.hi.to_le_bytes());
            h.update(x.lo.to_le_bytes());
        }
        h.finalize().into()
    }
}

// ---------------------------------------------------------------------------
// the co-rotational solver
// ---------------------------------------------------------------------------

/// Floor on `det F` below which an element's rotation is refused: the same `2⁻²⁰` the static
/// co-rotational solve uses (an element that flat or inverted has no polar factor).
const COROTATIONAL_DET_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 44);
/// Iteration budget of the polar decomposition per element and step.
const COROTATIONAL_POLAR_ITERATIONS: u32 = 64;
/// Cap on the ground contact's active-set iterations per step (each a solve): every pass
/// releases at least one tensile hold or locks at least one penetrating release, so the set
/// settles within the number of bottom vertices; the cap is for a cycle, which is an error.
const CONTACT_ACTIVE_SET_CAP: u32 = 32;

impl Operators {
    /// `out = (R Kₑ Rᵀ + Ṁ) u`, one element loop in mesh order: the element stiffness warped
    /// by its rotation `R` (the nodal displacements are rotated into the element frame, the
    /// small-strain stress and force formed there, the force rotated back), the mass term
    /// unrotated.
    fn apply_effective_rotated(&self, rotations: &[Mat3Fix], u: &[Fix128], out: &mut [Fix128]) {
        self.apply_scaled_rotated(rotations, Fix128::ONE, Fix128::ONE, u, out);
    }

    /// `out = (k_scale · R Kₑ Rᵀ + m_scale · Ṁ) u`: the rotated stiffness and the mass term
    /// with a scalar each, so the damped effective operator
    /// `K' + (2/dt) C + Ṁ = (1 + 2β/dt) K' + (1 + α dt/2) Ṁ` for `C = α M + β K'` costs the
    /// one element loop too.
    fn apply_scaled_rotated(
        &self,
        rotations: &[Mat3Fix],
        k_scale: Fix128,
        m_scale: Fix128,
        u: &[Fix128],
        out: &mut [Fix128],
    ) {
        out.fill(Fix128::ZERO);
        for ((element, &scale), r) in self
            .elements
            .iter()
            .zip(self.mass_scale.iter())
            .zip(rotations)
        {
            let local = gather(element, u);
            let rt = r.transpose();
            let mut frame = [[Fix128::ZERO; 3]; 4];
            for (slot, node_u) in frame.iter_mut().zip(local.iter()) {
                let v = rt.mul_vec(Vec3Fix::new(node_u[0], node_u[1], node_u[2]));
                *slot = [v.x, v.y, v.z];
            }
            let stress = element_stress_local(element, &frame, self.lambda, self.mu);
            let stiffness = element_force_from_stress(element, stress);
            let mass = Self::element_mass_force(self.lumping, scale, &local);
            for ((k, m), &node) in stiffness.iter().zip(mass.iter()).zip(element.nodes.iter()) {
                let f = r.mul_vec(Vec3Fix::new(k[0], k[1], k[2])) * k_scale;
                let base = node * 3;
                out[base] = out[base] + f.x + m[0] * m_scale;
                out[base + 1] = out[base + 1] + f.y + m[1] * m_scale;
                out[base + 2] = out[base + 2] + f.z + m[2] * m_scale;
            }
        }
    }

    /// `out = Σₑ R Kₑ (Rᵀ x₀ − x₀)`: the part of the warped element force that does not
    /// depend on the displacement (the rest positions seen from the rotated frame).
    fn rotation_offset(&self, rotations: &[Mat3Fix], rest: &[Fix128], out: &mut [Fix128]) {
        out.fill(Fix128::ZERO);
        for (element, r) in self.elements.iter().zip(rotations) {
            let local = gather(element, rest);
            let rt = r.transpose();
            let mut frame = [[Fix128::ZERO; 3]; 4];
            for (slot, x0) in frame.iter_mut().zip(local.iter()) {
                let v = rt.mul_vec(Vec3Fix::new(x0[0], x0[1], x0[2]));
                *slot = [v.x - x0[0], v.y - x0[1], v.z - x0[2]];
            }
            let stress = element_stress_local(element, &frame, self.lambda, self.mu);
            let force = element_force_from_stress(element, stress);
            for (k, &node) in force.iter().zip(element.nodes.iter()) {
                let f = r.mul_vec(Vec3Fix::new(k[0], k[1], k[2]));
                let base = node * 3;
                out[base] = out[base] + f.x;
                out[base + 1] = out[base + 1] + f.y;
                out[base + 2] = out[base + 2] + f.z;
            }
        }
    }

    /// The rotation of every element at the displacement `u`: the polar factor of
    /// `F = I + Σᵢ uᵢ ⊗ ∇Nᵢ`.
    fn rotations(&self, u: &[Fix128]) -> Result<Vec<Mat3Fix>, FemError> {
        let mut out = Vec::with_capacity(self.elements.len());
        for (tet, element) in self.elements.iter().enumerate() {
            let local = gather(element, u);
            // `cols[b][a]` is row `a`, column `b`: F_ab = δ_ab + Σᵢ u_i,a ∇N_i,b
            let mut cols = [[Fix128::ZERO; 3]; 3];
            for (g, node_u) in element.grad.iter().zip(local.iter()) {
                for (b, gb) in g.iter().enumerate() {
                    for (a, ua) in node_u.iter().enumerate() {
                        cols[b][a] = cols[b][a] + *ua * *gb;
                    }
                }
            }
            for (k, col) in cols.iter_mut().enumerate() {
                col[k] = col[k] + Fix128::ONE;
            }
            let f = Mat3Fix::from_cols(
                Vec3Fix::new(cols[0][0], cols[0][1], cols[0][2]),
                Vec3Fix::new(cols[1][0], cols[1][1], cols[1][2]),
                Vec3Fix::new(cols[2][0], cols[2][1], cols[2][2]),
            );
            match f.polar_rotation(COROTATIONAL_DET_FLOOR, COROTATIONAL_POLAR_ITERATIONS) {
                Ok(r) => out.push(r),
                Err(cause) => return Err(FemError::RotationFailed { tet, cause }),
            }
        }
        Ok(out)
    }
}

/// A transient co-rotational FEM solve, advanced one Newmark step at a time: the element
/// stiffness is warped by each element's rotation (the polar factor of its deformation
/// gradient at the start of the step, Müller & Gross 2004), so a rotation as a whole
/// produces no elastic force and the small-strain law holds in the rotated frame; the
/// time integration is the average-acceleration Newmark step of [`TransientSolver`] with the
/// warped stiffness frozen over the step (one linearisation per step, no inner Newton loop).
/// The internal force is `fᵢₙₜ(x) = Σₑ R Kₑ (Rᵀ x − x₀)`, so the step solves
/// `(R Kₑ Rᵀ + Ṁ) u_{n+1} = f + Ṁ ũ − Σₑ R Kₑ (Rᵀ x₀ − x₀)` on the free degrees of freedom.
/// Checked by `tests/analytic_dynamic_fem_corotational.rs`.
pub struct CorotationalTransientSolver {
    ops: Operators,
    /// rest positions (mm), three per vertex
    rest: Vec<Fix128>,
    /// Rayleigh damping `C = α M + β K'` (`α` 1/s, `β` s), none by default
    damping: Option<RayleighCoefficients>,
    /// the ground: a plane `y = ground` (mm) with outward normal `+y`, none by default
    ground: Option<Fix128>,
    /// per vertex, whether it rests on the ground this step (held on the plane in `y`)
    on_ground: Vec<bool>,
    dt: Fix128,
    inv_beta_dt2: Fix128,
    half_dt: Fix128,
    is_free: Vec<bool>,
    prescribed_value: Vec<Fix128>,
    load: Vec<Fix128>,
    precond: Vec<Fix128>,
    mass_precond: Vec<Fix128>,
    config: SolverConfig,
    u: Vec<Fix128>,
    v: Vec<Fix128>,
    a: Vec<Fix128>,
}

impl CorotationalTransientSolver {
    /// The same inputs as [`TransientSolver::new`]; the initial state is the prescribed
    /// field with zero velocity and the consistent acceleration.
    ///
    /// # Errors
    ///
    /// As [`TransientSolver::new`], plus [`FemError::RotationFailed`] when an element's
    /// deformation gradient at the initial state has no polar factor.
    pub fn new(
        mesh: &SdfTetMesh,
        material: &ElasticMaterial,
        boundary: &BoundaryConditions,
        dynamics: &DynamicsConfig,
        config: &SolverConfig,
    ) -> Result<Self, FemError> {
        let vertex_count = mesh.vertices.len();
        if vertex_count == 0 || mesh.tets.is_empty() {
            return Err(FemError::EmptyMesh);
        }
        for &(vertex, _, _) in boundary.prescribed().iter().chain(boundary.loads().iter()) {
            if vertex as usize >= vertex_count {
                return Err(FemError::VertexOutOfRange {
                    vertex,
                    vertex_count,
                });
            }
        }
        let elements = build_elements(mesh)?;
        let ops = Operators::new(elements, material, dynamics)?;
        let ndof = vertex_count * 3;
        let mut rest = Vec::with_capacity(ndof);
        for v in &mesh.vertices {
            rest.push(Fix128::from_f32(v[0]));
            rest.push(Fix128::from_f32(v[1]));
            rest.push(Fix128::from_f32(v[2]));
        }
        let mut prescribed_value = vec![Fix128::ZERO; ndof];
        let mut is_free = vec![true; ndof];
        for &(vertex, axis, value) in boundary.prescribed() {
            let d = vertex as usize * 3 + axis.index();
            is_free[d] = false;
            prescribed_value[d] = value;
        }
        let mut load = vec![Fix128::ZERO; ndof];
        for &(vertex, axis, force) in boundary.loads() {
            let d = vertex as usize * 3 + axis.index();
            if is_free[d] {
                load[d] = load[d] + force;
            }
        }
        let mass_diag = ops.mass_diagonal(ndof);
        let mass_precond = build_preconditioner(&mass_diag, &is_free, config)?;
        let diag = ops.effective_diagonal(ndof);
        let precond = build_preconditioner(&diag, &is_free, config)?;
        let mut solver = Self {
            ops,
            rest,
            damping: None,
            ground: None,
            on_ground: vec![false; vertex_count],
            dt: dynamics.dt,
            inv_beta_dt2: dynamics.inv_beta_dt2,
            half_dt: dynamics.dt.half(),
            is_free,
            prescribed_value: prescribed_value.clone(),
            load,
            precond,
            mass_precond,
            config: *config,
            u: prescribed_value,
            v: vec![Fix128::ZERO; ndof],
            a: vec![Fix128::ZERO; ndof],
        };
        solver.a = solver.consistent_acceleration()?;
        Ok(solver)
    }

    /// `a` from `Ṁ a / c₀ = f − fᵢₙₜ(u)` at the current displacement (`Ṁ = c₀ M`).
    fn consistent_acceleration(&self) -> Result<Vec<Fix128>, FemError> {
        let ndof = self.u.len();
        let rotations = self.ops.rotations(&self.u)?;
        let mut keff_u = vec![Fix128::ZERO; ndof];
        self.ops
            .apply_effective_rotated(&rotations, &self.u, &mut keff_u);
        let mut mass_u = vec![Fix128::ZERO; ndof];
        self.ops.apply_mass(&self.u, &mut mass_u);
        let mut offset = vec![Fix128::ZERO; ndof];
        self.ops
            .rotation_offset(&rotations, &self.rest, &mut offset);
        let mut rhs = vec![Fix128::ZERO; ndof];
        for d in 0..ndof {
            rhs[d] = if self.is_free[d] {
                self.load[d] - (keff_u[d] - mass_u[d]) - offset[d]
            } else {
                Fix128::ZERO
            };
        }
        let ops = &self.ops;
        let (scaled, _) = conjugate_gradient(
            &rhs,
            &self.is_free,
            &self.mass_precond,
            &self.config,
            |p, out| {
                ops.apply_mass(p, out);
            },
        )?;
        Ok(scaled.iter().map(|x| *x * self.inv_beta_dt2).collect())
    }

    /// Replace the displacement field (mm, three components per vertex in vertex order),
    /// keeping the velocities and recomputing the accelerations: the initial state of a body
    /// that starts rotated or deformed. The prescribed degrees of freedom keep their
    /// prescribed values whatever `u` says.
    ///
    /// # Errors
    ///
    /// [`FemError::VertexOutOfRange`] when `u` is not three values per vertex,
    /// [`FemError::RotationFailed`] when an element of the new state has no polar factor, and
    /// the conjugate gradient errors of the acceleration solve.
    pub fn set_displacements(&mut self, u: &[Fix128]) -> Result<(), FemError> {
        let ndof = self.u.len();
        if u.len() != ndof {
            return Err(FemError::VertexOutOfRange {
                vertex: (u.len() / 3) as u32,
                vertex_count: ndof / 3,
            });
        }
        let previous = core::mem::replace(&mut self.u, u.to_vec());
        for d in 0..ndof {
            if !self.is_free[d] {
                self.u[d] = self.prescribed_value[d];
            }
        }
        match self.consistent_acceleration() {
            Ok(a) => {
                self.a = a;
                Ok(())
            }
            Err(e) => {
                self.u = previous;
                Err(e)
            }
        }
    }

    /// A ground plane `y = ground` (mm, outward normal `+y`), or none. Contact is a per-step
    /// active set: a vertex whose Newmark predictor would cross the plane is held on it for
    /// that step (its `y` degree of freedom prescribed to the plane, velocity and
    /// acceleration in `y` zero), stays held while the reaction the hold provides pushes it
    /// up, and is released within the step the reaction turns tensile. Frictionless: `x` and `z`
    /// stay free. Exact non-penetration at the step's end, no penalty stiffness.
    pub fn set_ground(&mut self, ground: Option<Fix128>) {
        self.ground = ground;
        if ground.is_none() {
            self.on_ground.fill(false);
        }
    }

    /// The vertices held on the ground after the last step.
    #[must_use]
    pub fn on_ground(&self) -> &[bool] {
        &self.on_ground
    }

    /// External impulses `(vertex, J)` in N·s (`t·mm/s`), applied between steps: the velocity
    /// changes by `Δv = M⁻¹ J` (the mass solve, so the consistent and the lumped mass both
    /// conserve momentum: `Σ M Δv = J` exactly for the lumped one, to the solve's tolerance for
    /// the consistent one); the displacement and the acceleration are untouched, the next step
    /// carries the new velocity. Impulses on prescribed degrees of freedom are dropped (the
    /// constraint takes them); on a vertex held by the ground the downward part is dropped
    /// (the plane takes it) and the upward part releases it at the next step.
    ///
    /// # Errors
    ///
    /// [`FemError::VertexOutOfRange`] for a vertex index past the mesh, and the conjugate
    /// gradient errors of the mass solve.
    pub fn apply_impulses(&mut self, impulses: &[(u32, [Fix128; 3])]) -> Result<(), FemError> {
        let ndof = self.u.len();
        let vertex_count = ndof / 3;
        let mut j = vec![Fix128::ZERO; ndof];
        for &(vertex, impulse) in impulses {
            if vertex as usize >= vertex_count {
                return Err(FemError::VertexOutOfRange {
                    vertex,
                    vertex_count,
                });
            }
            for (axis, value) in impulse.iter().enumerate() {
                let d = vertex as usize * 3 + axis;
                // a held vertex takes an upward impulse (the plane only pushes), the ground
                // takes a downward one
                let into_ground =
                    axis == 1 && self.on_ground[vertex as usize] && *value < Fix128::ZERO;
                if self.is_free[d] && !into_ground {
                    j[d] = j[d] + *value;
                }
            }
        }
        // `Ṁ = c₀ M`, so solving `Ṁ y = J` gives `Δv = c₀ y`
        let ops = &self.ops;
        let (y, _) = conjugate_gradient(
            &j,
            &self.is_free,
            &self.mass_precond,
            &self.config,
            |p, out| {
                ops.apply_mass(p, out);
            },
        )?;
        for ((v, &y), &free) in self.v.iter_mut().zip(&y).zip(&self.is_free) {
            if free {
                *v = *v + y * self.inv_beta_dt2;
            }
        }
        Ok(())
    }

    /// `M v`, the momentum of the body (N·s, three components per vertex summed by the caller
    /// or read per vertex): the mass product of the velocity, so `Σ` over all vertices is the
    /// total momentum whatever the mass distribution.
    #[must_use]
    pub fn momentum(&self) -> Vec<Fix128> {
        let ndof = self.u.len();
        let mut out = vec![Fix128::ZERO; ndof];
        self.ops.apply_mass(&self.v, &mut out);
        // `apply_mass` applies `Ṁ = c₀ M`
        for x in &mut out {
            *x = *x / self.inv_beta_dt2;
        }
        out
    }

    /// Rayleigh damping `C = α M + β K'` with `K'` the warped stiffness: `α` in 1/s damps
    /// the low modes, `β` in s the high ones, the modal ratio
    /// `ζ(ω) = α/(2ω) + βω/2` ([`RayleighCoefficients::damping_ratio`]). `None` removes it.
    pub fn set_damping(&mut self, damping: Option<RayleighCoefficients>) {
        self.damping = damping;
    }

    /// Advance one Newmark step with the element rotations of the current state. Returns
    /// the conjugate gradient iterations used.
    ///
    /// With damping the step solves
    /// `(K' + (2/dt) C + Ṁ) u' = f + Ṁ ũ − C ṽ − Σₑ R Kₑ (Rᵀ x₀ − x₀)` with
    /// `ṽ = v + (dt/2) a − (2/dt) ũ`, which is `v' = (2/dt) u' + ṽ` of the `γ = 1/2`
    /// corrector substituted into `M a' + C v' + K' u' = f` (the trapezoidal rule on the
    /// damped system; a prescribed degree of freedom has `v' = 0`, and its column of the
    /// operator moved to the right-hand side cancels its share of `C ṽ` exactly).
    ///
    /// # Errors
    ///
    /// As [`TransientSolver::step`], plus [`FemError::RotationFailed`] when an element's
    /// deformation gradient has no polar factor; the state is unchanged on an error.
    pub fn step(&mut self) -> Result<u32, FemError> {
        let ndof = self.u.len();
        let rotations = self.ops.rotations(&self.u)?;
        // the free predictor from the state as it is: a vertex held last step sits on the
        // plane with zero velocity and acceleration, so its predictor is the plane exactly; an
        // impulse that gave it an upward velocity lifts its predictor above the plane and it
        // is not held this step
        let mut free_predictor = vec![Fix128::ZERO; ndof];
        for (d, slot) in free_predictor.iter_mut().enumerate() {
            *slot = if self.is_free[d] {
                newmark_predictor(self.u[d], self.v[d], self.a[d], self.dt, self.half_dt)
            } else {
                self.prescribed_value[d]
            };
        }
        // the held set: every vertex whose predictor is at or below the plane; then the
        // active-set iteration releases the held vertices whose reaction turns tensile (the
        // plane can only push) and re-solves, until no held vertex pulls, and a released
        // vertex that would end the step below the plane is held again for good
        let mut held = vec![false; ndof / 3];
        let mut locked = vec![false; ndof / 3];
        if let Some(ground) = self.ground {
            for (i, held_i) in held.iter_mut().enumerate() {
                let d = 3 * i + 1;
                if self.is_free[d] && free_predictor[d] <= ground - self.rest[d] {
                    *held_i = true;
                }
            }
        }
        let two_over_dt = Fix128::ONE / self.half_dt;
        let mut iteration = 0u32;
        let (next_u, is_free, prescribed, predictor, rhs, iterations) = loop {
            iteration += 1;
            if iteration > CONTACT_ACTIVE_SET_CAP {
                return Err(FemError::ContactUnsettled {
                    iterations: iteration - 1,
                });
            }
            // this trial's free set, prescribed field, predictor and state: the held vertices
            // are prescribed to the plane with zero velocity and acceleration
            let mut is_free = self.is_free.clone();
            let mut prescribed = self.prescribed_value.clone();
            let mut predictor = free_predictor.clone();
            let mut v = self.v.clone();
            let mut a = self.a.clone();
            if let Some(ground) = self.ground {
                for (i, &held_i) in held.iter().enumerate() {
                    if held_i {
                        let d = 3 * i + 1;
                        let plane_u = ground - self.rest[d];
                        is_free[d] = false;
                        prescribed[d] = plane_u;
                        predictor[d] = plane_u;
                        v[d] = Fix128::ZERO;
                        a[d] = Fix128::ZERO;
                    }
                }
            }
            let mut mass_predictor = vec![Fix128::ZERO; ndof];
            self.ops.apply_mass(&predictor, &mut mass_predictor);
            // the damped effective operator's scalars: (1 + 2β/dt) on K', (1 + α dt/2) on Ṁ
            let (k_scale, m_scale, c_v) = match self.damping {
                None => (Fix128::ONE, Fix128::ONE, None),
                Some(c) => {
                    // ṽ = v + (dt/2) a − (2/dt) ũ, then C ṽ = (α/c₀) Ṁ ṽ + β (K' ṽ), with K' ṽ the
                    // rotated stiffness product alone
                    let mut v_tilde = vec![Fix128::ZERO; ndof];
                    for d in 0..ndof {
                        v_tilde[d] = v[d] + self.half_dt * a[d] - two_over_dt * predictor[d];
                    }
                    let mut k_v = vec![Fix128::ZERO; ndof];
                    self.ops.apply_scaled_rotated(
                        &rotations,
                        Fix128::ONE,
                        Fix128::ZERO,
                        &v_tilde,
                        &mut k_v,
                    );
                    let mut m_v = vec![Fix128::ZERO; ndof];
                    self.ops.apply_mass(&v_tilde, &mut m_v);
                    let alpha_over_c0 = c.alpha / self.inv_beta_dt2;
                    let c_v: Vec<Fix128> = k_v
                        .iter()
                        .zip(m_v.iter())
                        .map(|(k, m)| c.beta * *k + alpha_over_c0 * *m)
                        .collect();
                    (
                        Fix128::ONE + two_over_dt * c.beta,
                        Fix128::ONE + self.half_dt * c.alpha,
                        Some(c_v),
                    )
                }
            };
            let mut keff_prescribed = vec![Fix128::ZERO; ndof];
            self.ops.apply_scaled_rotated(
                &rotations,
                k_scale,
                m_scale,
                &prescribed,
                &mut keff_prescribed,
            );
            let mut offset = vec![Fix128::ZERO; ndof];
            self.ops
                .rotation_offset(&rotations, &self.rest, &mut offset);
            // the right-hand side of the equation `A u' = rhs` on every degree of freedom (the
            // held ones too: their residual after the solve is the reaction the hold provides);
            // the free rows then move the prescribed columns over (`− A prescribed`)
            let mut rhs = vec![Fix128::ZERO; ndof];
            for d in 0..ndof {
                let damping_term = c_v.as_ref().map_or(Fix128::ZERO, |c| c[d]);
                rhs[d] = self.load[d] + mass_predictor[d] - offset[d] - damping_term;
            }
            let mut b_free = vec![Fix128::ZERO; ndof];
            for d in 0..ndof {
                if is_free[d] {
                    b_free[d] = rhs[d] - keff_prescribed[d];
                }
            }
            let ops = &self.ops;
            let (mut next_u, iterations) =
                conjugate_gradient(&b_free, &is_free, &self.precond, &self.config, |p, out| {
                    ops.apply_scaled_rotated(&rotations, k_scale, m_scale, p, out);
                })?;
            for d in 0..ndof {
                if !is_free[d] {
                    next_u[d] = prescribed[d];
                }
            }
            let Some(ground) = self.ground else {
                break (next_u, is_free, prescribed, predictor, rhs, iterations);
            };
            // the reaction at a held vertex: `(A u' − rhs)_y` with the full `u'` (held values
            // included), the force the plane had to add; a tensile one (negative, pulling
            // down) releases the vertex and the step is solved again; a released vertex that
            // ended below the plane is held again, for good
            let mut a_u = vec![Fix128::ZERO; ndof];
            self.ops
                .apply_scaled_rotated(&rotations, k_scale, m_scale, &next_u, &mut a_u);
            let mut changed = false;
            for i in 0..ndof / 3 {
                let d = 3 * i + 1;
                if !self.is_free[d] {
                    continue;
                }
                let plane_u = ground - self.rest[d];
                if held[i] {
                    if !locked[i] && a_u[d] - rhs[d] < Fix128::ZERO {
                        held[i] = false;
                        changed = true;
                    }
                } else if next_u[d] < plane_u {
                    held[i] = true;
                    locked[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break (next_u, is_free, prescribed, predictor, rhs, iterations);
            }
        };
        // commit: the corrector on the free degrees of freedom from the state as it was, the
        // held ones at the plane with zero velocity and acceleration
        for d in 0..ndof {
            if is_free[d] {
                let (next_v, next_a) = newmark_corrector(
                    next_u[d],
                    predictor[d],
                    self.v[d],
                    self.a[d],
                    self.inv_beta_dt2,
                    self.half_dt,
                );
                self.v[d] = next_v;
                self.a[d] = next_a;
            } else if self.is_free[d] {
                // held on the ground this step
                self.v[d] = Fix128::ZERO;
                self.a[d] = Fix128::ZERO;
            }
        }
        let _ = (prescribed, rhs);
        for (i, on) in self.on_ground.iter_mut().enumerate() {
            *on = held[i];
        }
        self.u = next_u;
        Ok(iterations)
    }

    /// Nodal displacements (mm), three components per vertex in vertex order.
    #[must_use]
    pub fn displacements(&self) -> &[Fix128] {
        &self.u
    }

    /// Nodal velocities (mm/s), three components per vertex in vertex order.
    #[must_use]
    pub fn velocities(&self) -> &[Fix128] {
        &self.v
    }

    /// Nodal accelerations (mm/s²), three components per vertex in vertex order.
    #[must_use]
    pub fn accelerations(&self) -> &[Fix128] {
        &self.a
    }

    /// Domain separation of [`Self::state_digest`]: the SHA-256 of the text
    /// `alice-physics/dynamic-fem-state-digest/v1`, computed once (the same bytes as
    /// [`TransientSolver::STATE_DIGEST_DOMAIN`]; the two solvers' states differ in content, so
    /// a digest of one is never a digest of the other).
    pub const STATE_DIGEST_DOMAIN: [u8; 32] = TransientSolver::STATE_DIGEST_DOMAIN;

    /// SHA-256 of the solver's state: `STATE_DIGEST_DOMAIN`, then the raw `Fix128` bits of
    /// every displacement, velocity and acceleration (little-endian `i64` high, `u64` low,
    /// degree of freedom by degree of freedom), then one byte per vertex for the ground hold
    /// (`1` held, `0` free). Two runs agree exactly when their states are the same to the
    /// bit, which is what a replay of the same inputs gives on every target. This digest
    /// stands alone: the finite element state is not yet part of
    /// [`crate::PhysicsWorld::state_digest`] or its snapshot.
    #[must_use]
    pub fn state_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(Self::STATE_DIGEST_DOMAIN);
        for x in self.u.iter().chain(&self.v).chain(&self.a) {
            h.update(x.hi.to_le_bytes());
            h.update(x.lo.to_le_bytes());
        }
        for &held in &self.on_ground {
            h.update([u8::from(held)]);
        }
        h.finalize().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf_fem_mesh::Tetrahedron;

    /// A corner tetrahedron with `det J = 3`, so `V = |det J|/6 = 1/2`.
    ///
    /// The height is `3` and not `1` on purpose. The volume of the *reference*
    /// tetrahedron is `1/6`, whose denominator is not a power of two, so it does
    /// not land exactly in [`Fix128`] and every exact assert below would fail by
    /// a handful of units in the last place — measured at `4` on `ρV/4` with
    /// `ρ = 24`. `det J` being a multiple of three is what makes the volume
    /// dyadic, and the scene has to supply that; no amount of care inside the
    /// solver recovers it.
    fn dyadic_volume_tet() -> SdfTetMesh {
        let mut mesh = SdfTetMesh::default();
        mesh.vertices.push([0.0, 0.0, 0.0]);
        mesh.vertices.push([1.0, 0.0, 0.0]);
        mesh.vertices.push([0.0, 1.0, 0.0]);
        mesh.vertices.push([0.0, 0.0, 3.0]);
        mesh.tets.push(Tetrahedron {
            vertices: [0, 1, 2, 3],
        });
        mesh
    }

    fn ops_for(lumping: MassLumping, density: Fix128, dt: Fix128) -> Operators {
        let mesh = dyadic_volume_tet();
        let elements = build_elements(&mesh).expect("non-degenerate");
        let material = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_f64(0.35))
            .expect("valid material");
        let dynamics = DynamicsConfig::try_new(density, dt, lumping).expect("valid dynamics");
        Operators::new(elements, &material, &dynamics).expect("mass fits")
    }

    /// `Mₑ = (ρV/20)(1 + δᵢⱼ)` — the closed-form barycentric moment, not a
    /// quadrature rule. Read through `Ṁ = 4M/dt²` with `dt = 2`, where that
    /// factor is exactly one.
    #[test]
    fn the_consistent_element_mass_is_the_closed_form_moment() {
        // ρ = 20, V = 1/2 ⟹ ρV/20 = 1/2.
        let ops = ops_for(
            MassLumping::Consistent,
            Fix128::from_int(20),
            Fix128::from_int(2),
        );
        assert_eq!(ops.mass_scale.len(), 1);
        assert_eq!(ops.mass_scale[0], Fix128::ONE.half());
    }

    /// Lumped is `ρV/4`, and that division is exact in binary where the
    /// consistent division by twenty is not.
    #[test]
    fn the_lumped_element_mass_is_a_quarter_of_the_element_mass() {
        // ρ = 24, V = 1/2 ⟹ ρV/4 = 3.
        let ops = ops_for(
            MassLumping::Lumped,
            Fix128::from_int(24),
            Fix128::from_int(2),
        );
        assert_eq!(ops.mass_scale[0], Fix128::from_int(3));
    }

    /// `Σᵢⱼ Mᵢⱼ = ρ∫(ΣᵢNᵢ)(ΣⱼNⱼ)dV = ρV` for either distribution: the shape
    /// functions are a partition of unity, so a rigid translation sees the whole
    /// element mass and nothing else. This is what makes the two distributions
    /// interchangeable for a free-flight scene and distinguishable for a
    /// vibrating one.
    #[test]
    fn both_distributions_carry_the_same_total_mass() {
        for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
            let ops = ops_for(lumping, Fix128::from_int(60), Fix128::from_int(2));
            let ones = vec![Fix128::ONE; 12];
            let mut out = vec![Fix128::ZERO; 12];
            ops.apply_mass(&ones, &mut out);
            let total = out[0] + out[3] + out[6] + out[9];
            // ρV = 60/2 = 30.
            assert_eq!(total, Fix128::from_int(30), "{lumping:?}");
        }
    }

    /// The diagonal of the consistent pattern is twice its off-diagonal entry,
    /// which is what `(1 + δᵢⱼ)` says — read off a single unit impulse rather
    /// than from `element_mass_diagonal`, so the preconditioner and the operator
    /// are checked against each other and not against themselves.
    #[test]
    fn the_consistent_diagonal_is_twice_the_off_diagonal() {
        let ops = ops_for(
            MassLumping::Consistent,
            Fix128::from_int(20),
            Fix128::from_int(2),
        );
        let scale = ops.mass_scale[0];
        let mut unit = vec![Fix128::ZERO; 12];
        unit[0] = Fix128::ONE;
        let mut out = vec![Fix128::ZERO; 12];
        ops.apply_mass(&unit, &mut out);
        assert_eq!(out[0], scale + scale, "diagonal");
        assert_eq!(out[3], scale, "off-diagonal");
        assert_eq!(out[6], scale, "off-diagonal");
        assert_eq!(out[9], scale, "off-diagonal");
        assert_eq!(
            ops.mass_diagonal(12)[0],
            out[0],
            "mass_diagonal must agree with apply_mass on a unit impulse"
        );
    }

    /// `diag(A)ᵢᵢ = eᵢᵀ A eᵢ` — the definition, checked against the operator.
    ///
    /// This is the only statement available about the effective diagonal, and
    /// it took measuring to establish that. The diagonal feeds the
    /// preconditioner and nothing else, and **a preconditioner cannot change
    /// the converged answer** — that is what a preconditioner is — so no oracle
    /// on the solution can see a wrong one. Nor can the iteration count without
    /// pinning a number: measured on an eight-element bar over 60 steps, the
    /// Jacobi-scaled solve took 1267 conjugate gradient iterations with the
    /// correct diagonal and 1408 with the mass left out of it, against 1786
    /// with no preconditioner at all. The *correct* property — "preconditioning
    /// helps" — holds in both cases, so only the number 1267 separates them,
    /// and a pinned number is a change detector rather than a statement about
    /// correctness.
    ///
    /// What is left is the definition. [`Operators::effective_diagonal`] walks
    /// the shape function gradients directly, while
    /// [`Operators::apply_effective`] goes through the stress tensor and the
    /// element force, so the two reach the same entry by different arithmetic
    /// and neither is the other's reference.
    ///
    /// The two orders of operations do not round identically — the diagonal
    /// forms `(λ+2μ)·g²` where the operator forms `g·((λ+2μ)·g)` — so the
    /// comparison is in raw units rather than bit for bit.
    ///
    /// Measured over both distributions and all twelve degrees of freedom, the
    /// differences are `0`, `−360` and `−1560` units of `2⁻⁶⁴`, and **which
    /// ones are zero is not arbitrary**: degrees of freedom 3 through 8 belong
    /// to nodes 1 and 2, whose gradients `(1,0,0)` and `(0,1,0)` are dyadic, and
    /// those agree bit for bit. Every non-zero difference belongs to a node
    /// whose gradient carries the `1/3` of [`dyadic_volume_tet`]. That is the
    /// evidence that the gap is rounding order on a non-representable gradient
    /// and not a difference of formula — a formula difference would not care
    /// which gradients happen to be dyadic. The bound is eight times the
    /// measured worst case.
    #[test]
    fn the_effective_diagonal_is_the_operator_applied_to_a_unit_vector() {
        const BOUND: i128 = 12_480; // 8 × the measured worst case of 1560
        let mut worst = 0i128;
        for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
            let ops = ops_for(lumping, Fix128::from_int(60), Fix128::from_int(2));
            let diag = ops.effective_diagonal(12);
            for d in 0..12 {
                let mut unit = vec![Fix128::ZERO; 12];
                unit[d] = Fix128::ONE;
                let mut out = vec![Fix128::ZERO; 12];
                ops.apply_effective(&unit, &mut out);
                let delta = raw(diag[d]) - raw(out[d]);
                worst = worst.max(delta.abs());
                assert!(
                    delta.abs() <= BOUND,
                    "{lumping:?} dof {d}: diag {:?} but eᵀAe {:?}, {delta} raw units apart",
                    diag[d],
                    out[d]
                );
                assert!(
                    diag[d] > Fix128::ZERO,
                    "{lumping:?} dof {d}: the diagonal of a positive definite operator \
                     must be positive"
                );
            }
        }
        assert!(
            worst > 0,
            "the two paths agreed bit for bit, so the {BOUND}-unit bound is vacuous and \
             the comment above about rounding orders is wrong"
        );
    }

    // -----------------------------------------------------------------------
    // the integrator on its own
    // -----------------------------------------------------------------------

    fn raw(v: Fix128) -> i128 {
        (i128::from(v.hi) << 64) | i128::from(v.lo)
    }

    /// `dt = 2⁻ˢʰⁱᶠᵗ`.
    fn dyadic_dt(shift: u32) -> Fix128 {
        Fix128::from_raw(0, 1u64 << (64 - shift))
    }

    /// One Newmark step of `m x'' + k x = 0` driven through the solver's own
    /// [`newmark_predictor`] and [`newmark_corrector`], with the 1×1 linear
    /// solve written out instead of going through the conjugate gradient.
    struct Sdof {
        m: Fix128,
        k: Fix128,
        dt: Fix128,
        inv_beta_dt2: Fix128,
        half_dt: Fix128,
        x: Fix128,
        v: Fix128,
        a: Fix128,
    }

    impl Sdof {
        fn new(m: Fix128, k: Fix128, dt: Fix128, x0: Fix128) -> Self {
            let two_over_dt = Fix128::from_int(2) / dt;
            let inv_beta_dt2 = two_over_dt * two_over_dt;
            Self {
                m,
                k,
                dt,
                inv_beta_dt2,
                half_dt: dt.half(),
                x: x0,
                v: Fix128::ZERO,
                a: (Fix128::ZERO - k * x0) / m,
            }
        }
        fn step(&mut self) {
            let predictor = newmark_predictor(self.x, self.v, self.a, self.dt, self.half_dt);
            // (k + ṁ) x' = ṁ·x̃, the scalar form of the effective system.
            let m_dot = self.m * self.inv_beta_dt2;
            let next_x = (m_dot * predictor) / (self.k + m_dot);
            let (next_v, next_a) = newmark_corrector(
                next_x,
                predictor,
                self.v,
                self.a,
                self.inv_beta_dt2,
                self.half_dt,
            );
            self.x = next_x;
            self.v = next_v;
            self.a = next_a;
        }
        /// `E = ½mv² + ½kx²`.
        fn energy(&self) -> Fix128 {
            (self.m * self.v * self.v + self.k * self.x * self.x).half()
        }
    }

    /// For a dyadic `dt` every Newmark coefficient is an exact power of two,
    /// and the invert-first form `(2/dt)·(2/dt)` agrees with the direct
    /// `4/(dt·dt)` bit for bit.
    ///
    /// This is the reason `β = 1/4` is hard-coded, and the reason the
    /// invert-first rewrite is free: it buys six digits on a small non-dyadic
    /// step (see [`DynamicsConfig::try_new`]) **without** moving the dyadic case
    /// it was not trying to fix. Checked over a range of steps rather than at
    /// one value, because the claim is about the format and not about a
    /// particular constant.
    #[test]
    fn the_newmark_coefficients_are_exact_for_a_dyadic_step() {
        for shift in 1u32..=10 {
            let dt = dyadic_dt(shift);
            let half_dt = dt.half();
            let two_over_dt = Fix128::from_int(2) / dt;
            let inv_beta_dt2 = two_over_dt * two_over_dt;
            assert_eq!(
                inv_beta_dt2,
                Fix128::from_int(1i64 << (2 * shift + 2)),
                "(2/dt)² at dt = 2^-{shift}"
            );
            assert_eq!(
                inv_beta_dt2,
                Fix128::from_int(4) / (dt * dt),
                "invert-first and direct disagree at dt = 2^-{shift}, \
                 so the rewrite is not free after all"
            );
            // `(dt/2)·((dt/2)·a)` is the predictor's acceleration term; undoing
            // it with `4/dt²` has to land back on `a` exactly.
            for a in [Fix128::ONE, Fix128::from_int(-7), Fix128::from_int(1024)] {
                assert_eq!(
                    half_dt * (half_dt * a) * inv_beta_dt2,
                    a,
                    "the predictor and corrector weights do not invert at \
                     dt = 2^-{shift}, a = {a:?}"
                );
            }
            assert_eq!(half_dt + half_dt, dt, "dt/2 at dt = 2^-{shift}");
        }
    }

    /// With `ω·dt = 2` the trapezoidal rule's phase per step is
    /// `θ = 2·atan(ω dt/2) = π/2`, so the orbit closes after four steps on
    /// `x₀, 0, −x₀, 0` and the velocity on `0, −ω x₀, 0, ω x₀`.
    ///
    /// Derived from the recurrence, not from running it: with `Ω²/4 = 1` the
    /// effective equation `x₁ = x₀ − (Ω²/4)(x₀ + x₁)` forces `x₁ = 0`, and the
    /// next two steps follow the same way. The scene is chosen so that `k`, `m`,
    /// `4/dt²` and the effective stiffness `k + 4m/dt² = 512 = 2⁹` are all exact,
    /// which is what makes this the one conservative claim that can be an
    /// `assert_eq!`: nothing on the path rounds.
    #[test]
    fn the_omega_dt_equal_two_orbit_is_the_closed_form_four_cycle() {
        // m = 1, k = 256 ⟹ ω = 16; dt = 1/8 ⟹ Ω = ω dt = 2.
        let m = Fix128::ONE;
        let k = Fix128::from_int(256);
        let omega = Fix128::from_int(16);
        let mut s = Sdof::new(m, k, dyadic_dt(3), Fix128::ONE);
        let want = [
            (Fix128::ZERO, Fix128::ZERO - omega),
            (Fix128::NEG_ONE, Fix128::ZERO),
            (Fix128::ZERO, omega),
            (Fix128::ONE, Fix128::ZERO),
        ];
        for cycle in 0..4 {
            for (phase, &(wx, wv)) in want.iter().enumerate() {
                s.step();
                assert_eq!(s.x, wx, "x at cycle {cycle} phase {phase}");
                assert_eq!(s.v, wv, "v at cycle {cycle} phase {phase}");
            }
        }
        // 1024 further revolutions, still landing on the start exactly.
        for _ in 0..4096 {
            s.step();
        }
        assert_eq!(s.x, Fix128::ONE, "x after 4112 steps");
        assert_eq!(s.v, Fix128::ZERO, "v after 4112 steps");
    }

    /// A different `Ω` on the same integrator does **not** close: the effective
    /// stiffness stops being a power of two, the division truncates, and the
    /// energy wanders.
    ///
    /// The point of the test is the contrast with the case above — that the
    /// four-cycle is exact because of the scene and not because the arithmetic
    /// is exact in general. The bound is the measured worst case over the four
    /// scenes times four; the measurement itself is in the table below.
    ///
    /// | m | k | dt | worst `E` drift over 4096 steps (raw 2⁻⁶⁴) |
    /// |---|---|----|---|
    /// | 1 | 100 | 2⁻⁶ | −37 257 |
    /// | 3 | 7 | 2⁻⁵ | 17 416 |
    /// | 1 | 1 | 2⁻⁴ | −1 740 |
    /// | 5 | 1000 | 2⁻⁷ | 925 109 |
    ///
    /// Relative to `E₀` that is about `1e-16` in every row. The amplitude never
    /// grows, because the truncation is toward zero.
    #[test]
    fn a_general_step_drifts_in_energy_within_the_measured_bound() {
        const WORST_MEASURED: i128 = 925_109;
        let bound = WORST_MEASURED * 4;
        let mut any_nonzero = false;
        for (mi, ki, shift) in [(1i64, 100i64, 6u32), (3, 7, 5), (1, 1, 4), (5, 1000, 7)] {
            let m = Fix128::from_int(mi);
            let k = Fix128::from_int(ki);
            let mut s = Sdof::new(m, k, dyadic_dt(shift), Fix128::ONE);
            let e0 = s.energy();
            let mut worst = 0i128;
            let mut peak = Fix128::ZERO;
            for _ in 0..4096 {
                s.step();
                let d = raw(s.energy()) - raw(e0);
                if d.abs() > worst.abs() {
                    worst = d;
                }
                if s.x.abs() > peak {
                    peak = s.x.abs();
                }
            }
            any_nonzero |= worst != 0;
            assert!(
                worst.abs() <= bound,
                "m={mi} k={ki} dt=2^-{shift}: energy drifted {worst} raw units, bound {bound}"
            );
            assert!(
                peak <= Fix128::ONE,
                "m={mi} k={ki} dt=2^-{shift}: amplitude grew to {} > 1, \
                 so the truncation is no longer toward zero",
                peak.to_f64()
            );
        }
        assert!(
            any_nonzero,
            "every scene conserved energy exactly, so this test is measuring nothing \
             and the bound above is vacuous"
        );
    }
}
