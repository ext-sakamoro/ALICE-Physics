//! Sub-iteration monitoring for partitioned (staggered) multiphysics coupling.
//!
//! The crate couples physics by *composition*: a caller applies one
//! one-directional force function, then another (`cloth_fluid`,
//! `fsi_advanced`). That is an explicit partitioned scheme with **zero
//! sub-iterations**, and such a scheme is a fixed-point iteration whose
//! convergence is a property of the splitting, not of the time step.
//!
//! This module supplies the instrument that tells a caller whether the
//! splitting it chose is a contraction, and — the harder half — whether the
//! iteration it just ran diverged *silently*.
//!
//! # Why an instrument, and not a monolithic solver
//!
//! A partitioned scheme that is sub-iterated **to convergence** solves the
//! same equations as a monolithic one. Measured on the model problem below,
//! the naive (non-sub-iterated) scheme fails at a sharp closed-form threshold,
//! but relaxation repairs it: optimal constant relaxation `ω* = m/(m + m_f)`
//! converges in a single step for any mass ratio, and Aitken dynamic
//! relaxation converged for 50 coupled modes with coupling ratios spanning
//! `5..500` (sweep count grew 7 → 184; it did not diverge). So the difference
//! between partitioned and monolithic is one of convergence rate and
//! robustness, not of correctness, and the thing worth building first is the
//! measurement.
//!
//! ⚠️ **For a path-dependent constitutive law that equivalence is conditional,
//! and the condition is where the fixed point is defined.** The piston above
//! has no memory, so its fixed point can be taken over the whole load path.
//! An elastoplastic body has memory, and the Simo–Miehe (Armero–Simo) split
//! that makes a thermoplastic sweep well posed defines the map **per
//! increment**, on `(ε_p^n, ε̄_p^n, T^n)` held fixed: the unknown is the
//! *increment* `δT`, the map is
//! `δT_{k+1} = deposit(ΔW_p(T^n + δT_k; state^n))`, and relaxation is applied
//! to `δT`. ⚠️ **Committing the internal variables inside the sweep breaks the
//! split** — each sub-iteration would then start from a different `ε_p`, the
//! map would no longer be a map on `δT`, and converging it would not solve the
//! monolithic equations. That is why
//! [`crate::linear_elastic_fem::ElastoplasticIncrement::commit`] is a separate call
//! from
//! [`crate::linear_elastic_fem::ElastoplasticProblem::step`]: the driver sweeps with
//! `step`, and commits once, after the sub-iteration has converged.
//!
//! # The model problem (closed-form spectral radius)
//!
//! A piston of mass `m` on a spring `k`, pushing an incompressible fluid
//! column of mass `m_f` through area `A`:
//!
//! ```text
//! structure : m·a = −k·x − p·A
//! fluid     : p   = (m_f/A)·a      (incompressible ⇒ p is a Lagrange
//!                                   multiplier: instantaneous, no time scale)
//! monolithic: (m + m_f)·a = −k·x   ⇒  ω = √(k/(m + m_f))
//! ```
//!
//! One Dirichlet–Neumann sweep evaluates the fluid at the previous structure
//! iterate, then the structure at that pressure:
//!
//! ```text
//! a^(j+1) = (−k·xⁿ − k·Δt·vⁿ − m_f·a^(j)) / (m + k·Δt²)
//! e^(j+1) = −ρ(Δt)·e^(j),      ρ(Δt) = m_f / (m + k·Δt²)
//! ```
//!
//! ⚠️ `Δt` does **not** cancel, but it works against the caller: as `Δt`
//! shrinks, `ρ` rises toward `m_f/m` rather than falling. **Refining the time
//! step makes a divergent splitting slightly worse, never better.** Measured
//! with `m = k = 1`:
//!
//! | `Δt` | `ρ` at `m_f/m = 1/2` | `ρ` at `m_f/m = 2` |
//! |---|---|---|
//! | `1/8` | `0.4923076923076923` | `1.9692307692307693` |
//! | `1/80` | `0.4999218872051242` | `1.9996875488204968` |
//! | `1/800` | `0.4999992187512207` | `1.9999968750048827` |
//! | `1/8000` | `0.4999999921875001` | `1.9999999687500005` |
//!
//! The limit `ρ → m_f/m` is the added-mass ratio, and it is a property of the
//! splitting alone: incompressibility makes the pressure a Lagrange multiplier
//! carrying no time scale of its own, so no amount of temporal refinement
//! reaches it. This is the added-mass instability of partitioned FSI (Causin,
//! Gerbeau & Nobile 2005; Förster, Wall & Ramm 2007).
//!
//! That monotone behaviour in `Δt` is what makes `ρ` worth measuring: it closes
//! the "use a smaller step" escape hatch, which an accuracy oracle (a
//! manufactured solution, say) cannot do, because an accuracy oracle passes on
//! both schemes whenever both converge.
//!
//! # Measured behaviour of a diverging iteration under `Fix128`
//!
//! `Fix128` arithmetic wraps: `src/math.rs` uses `wrapping_add` /
//! `wrapping_sub` / `wrapping_mul` and contains no `saturating_*` and no
//! `debug_assert!`. A diverging iteration therefore neither panics nor
//! saturates — it wraps, silently.
//!
//! Two consequences were measured on the model problem in its `Δt`-free form —
//! `a^(j+1) = (−k·x − m_f·a^(j))/m` with `m = k = x = 1`, exact fixed point
//! `a* = −1/(1 + m_f/m)` — which isolates the wrap from the time stepping:
//!
//! **1. The squared residual wraps negative, and [`Fix128::sqrt`] returns zero
//! for a negative argument, so a diverging iteration reports an L2 residual of
//! exactly `0` — it looks perfectly converged.** The stronger the coupling,
//! the sooner:
//!
//! | `m_f/m` | sweep where `e·e` first goes negative | reported `‖r‖₂` |
//! |---|---|---|
//! | 1.5 | 59 | `0` |
//! | 2.0 | 33 | `0` |
//! | 4.0 | 16 | `0` |
//!
//! **2. The squared residual also reaches zero while *converging*, long before
//! the error does.** At `m_f/m = 0.5` the squared residual is already `0` at
//! sweep 40 while `|e| = 3.03e-13`. The crossover is at
//! `|e| ≈ √(2⁻⁶⁴) = 2.328306e-10`, below which `e·e` truncates to zero.
//!
//! The squared L2 norm is therefore unusable in **both** directions here. Use
//! [`residual_norm_inf`]; [`residual_norm_l2_checked`] exists for callers who
//! need L2 and reports [`CoupledIterationError::ArithmeticWrapped`] rather than
//! a plausible zero.
//!
//! Detecting divergence by magnitude ("the residual got huge") consequently
//! cannot work. [`ContractionMonitor`] measures the contraction ratio over the
//! first few sweeps, while the residual is far from both the floor and the wrap
//! point, and separately watches for the wrap signature: **monotone growth
//! followed by a decrease**, which a genuinely growing sequence cannot produce.
//!
//! ## Where to put the wrap detector
//!
//! The strongest check is on the inner product itself, not on the norm: `r·r`
//! is a sum of squares and **cannot be negative**, so one comparison settles it
//! with no threshold and no heuristic. That check lives in
//! [`residual_norm_l2_checked`] and is the primary detector.
//!
//! ⚠️ It is necessary but **not sufficient**. A magnitude of exactly `2³²`
//! squares to `2⁶⁴`, whose encoding wraps the integer field to **zero** rather
//! than to a negative value — so the sum does not move, `r·r ≥ 0` still holds,
//! and a component worth `4.29e9` contributes nothing at all. A vector made of
//! such components reports `‖r‖₂ = 0` while completely diverged. The checked
//! norm therefore tests two things: the running sum never decreases, *and* no
//! term above the floor squares to zero. The monotonicity guard on the residual
//! sequence in [`ContractionMonitor`] is the third and weakest layer, and it is
//! there for callers who only have a norm to hand.
//!
//! # ⚠️ These thresholds describe the *unrelaxed* scheme
//!
//! Everything above concerns a partitioned sweep with **no relaxation**, which
//! is what the crate does today. Relaxation changes the picture completely:
//! optimal constant relaxation `ω* = m/(m + m_f)` converges in one step at any
//! mass ratio, and Aitken dynamic relaxation converged on every case measured.
//!
//! So `ρ > 1` means "this splitting diverges **when swept without
//! relaxation**", not "partitioned coupling cannot solve this". A future
//! relaxed sweep will contract where the unrelaxed one does not, and that is an
//! improvement — an oracle phrased as "the staggered scheme diverges" would
//! turn it into a regression.
//!
//! # Measured location of the critical ratio
//!
//! Truncating multiplication does **not** move the threshold. Classifying the
//! raw 128-bit error magnitude at sweep 1 against sweep 39:
//!
//! | `m_f/m` | verdict |
//! |---|---|
//! | `1` exactly | marginal — `|e|` is *bit-identical* at sweeps 1 and 39 |
//! | `1 + 1` ulp | grows |
//! | `1 − 1` ulp | marginal (flat) |
//! | `1 − 2` ulp | decays |
//! | `0.99` / `1.01` | decays / grows |
//!
//! `m_f/m = 1` is exact because multiplying by [`Fix128::ONE`] is exact, so no
//! truncation enters the sweep at all. The one deviation from real arithmetic
//! is a **dead band about one ulp wide just below 1**, where a decay of one ulp
//! per sweep is cancelled by truncation bias. Oracles must bracket the
//! threshold rather than probe at it, and
//! [`SubIterationConfig::divergence_ratio`] sits above one by a real margin for
//! the same reason.
//!
//! # Thresholds are relative, and never zero
//!
//! Every threshold here is relative to something, because fixed thresholds in
//! this crate's iterative solvers have broken four times: the convergence test
//! is relative to the first residual, and the stagnation window is a fraction
//! of the sweeps taken so far, with stopping point
//! `best_sweep/(1 − fraction)` — so `fraction ≥ 1` never fires and is rejected
//! by [`SubIterationConfig::validate`].
//!
//! No stopping rule compares a change against zero. Truncating multiplication
//! means the iteration has no exact fixed point and no cycle: it drifts at the
//! rounding floor indefinitely, so "the residual stopped changing" never
//! becomes true.
//!
//! # Non-dimensionalisation (fixed here, before any oracle depends on it)
//!
//! A coupled operator must be **block-equilibrated before it is handed to a
//! Krylov method**, and that is a requirement, not an optimisation. Each field
//! reports a characteristic scale `s_i`; the operator is applied to scaled
//! unknowns `x̃_i = x_i / s_i` so every block is `O(1)`.
//!
//! The criterion is that each product entering an inner product stays above the
//! truncation floor: `|a·b| ≥ 2⁻⁶⁴ = 5.421011e-20`, i.e. `|a| ≥ 2.328306e-10`
//! when `a ≈ b` (this is [`L2_TERM_FLOOR`]). In raw SI units a thermo-elastic
//! block pair does not clear it:
//!
//! | material / thickness | `E·h` | `k·h` | block ratio | product / floor |
//! |---|---|---|---|---|
//! | steel / 10 mm | `2.000e+09` | `5.000e-01` | `4.00e+09` | `1.153` |
//! | steel / 1 mm | `2.000e+08` | `5.000e-02` | `4.00e+09` | `1.153` |
//! | aluminium / 10 mm | `7.000e+08` | `2.000e+00` | `3.50e+08` | `150.6` |
//! | **PLA / 1 mm** | `3.500e+06` | `1.300e-04` | `2.69e+10` | **`0.025`** |
//! | **PLA / 0.2 mm** | `7.000e+05` | `2.600e-05` | `2.69e+10` | **`0.025`** |
//!
//! PLA is this crate's primary material, and there the thermal block's
//! contribution to an inner product truncates to zero — the coupling would go
//! missing without a single error. Steel clears the floor by a factor of
//! `1.15`, which is not a margin. The ratio is thickness-independent (`E·h` and
//! `k·h` carry the same `h`), so geometry cannot be used to escape it.
//!
//! ## Why an *absolute* floor is the right criterion here
//!
//! In `f64` this criterion would be wrong: a term of `1e-19` added to a sum of
//! order one is lost to the relative epsilon `2.2e-16` whatever any absolute
//! floor says. `Fix128` is fixed-point, so **accumulation precision is
//! absolute** — a value with `|x| ≥ 2⁻⁶⁴` survives being added to `1.0` or to
//! `1e18` alike. The only place information disappears is the *product*, when
//! truncation takes `a·b` to zero, which is exactly what the criterion tests.
//! Porting any of this to floating point invalidates the criterion.
//!
//! # Why there is still no monolithic assembly, in numbers
//!
//! The thermoplastic driver ([`crate::linear_elastic_fem::step_thermoplastic`])
//! is a partitioned map `G(δT) = deposit(ΔW_p(T^n + δT))` swept to a fixed
//! point. A monolithic (Newton on the coupled residual) assembly would be the
//! right tool when the map stops contracting fast enough — the sweep count of
//! a fixed-point iteration grows as `ln(tol) / ln |λ_max|` while Newton's does
//! not depend on `λ_max` at all, so the two cross as `|λ_max| → 1`. The entry
//! condition this crate uses is therefore:
//!
//! > **`|λ_max| > 0.9` at a realistic heat capacity**, or a `Stagnated` /
//! > period-two cycle that is *not* the `σ_y(T)` clamp, where `λ_max` is the
//! > dominant eigenvalue of `∂G/∂δT` at the fixed point. `0.9` is a convenience
//! > threshold (a `2⁻⁴⁰` tolerance there already costs about 260 sweeps), not a
//! > theorem.
//!
//! **Measured** (`tests/analytic_thermoplastic_coupling.rs::the_monolithic_entry_condition_is_three_decades_away_at_real_heat_capacities`,
//! finite-difference Jacobian over the deposit's support at the fixed point,
//! power iteration; the test asserts the band and prints the table):
//!
//! | `c_v` (MPa/K) | material | `λ_max` | converged `max δT` (K) |
//! |---|---|---|---|
//! | `2⁻¹⁰` | non-physical | **`+0.9366`** | 8.294 |
//! | `2⁻⁸` | non-physical (the oracle scene) | `+0.2800` | 2.583 |
//! | `2⁻⁴` | non-physical | `+0.01836` | 0.1728 |
//! | `1` | non-physical | `+0.001151` | 0.01084 |
//! | `3.82` | **steel** (7850 kg/m³ × 486 J/kgK) | `+3.01e-4` | 0.002839 |
//! | `2.43` | aluminium (2700 × 900) | `+4.74e-4` | 0.004463 |
//! | `1.9` | PLA (estimate, 1240 × 1800) | `+6.06e-4` | 0.005708 |
//!
//! Every real material sits **three decades below** the entry; the only
//! heat capacity that opens it is a non-physical `2⁻¹⁰` MPa/K. The sign is
//! real (the Jacobian is explicit), and it is softening's: hotter → weaker →
//! more plastic work → hotter. The row at `2⁻⁸` is where
//! [`crate::linear_elastic_fem::ThermoplasticCoupling::try_new`] measured the
//! relaxation table, and the two numbers agree.
//!
//! **What would open it**: thermal softening *localising* — an adiabatic shear
//! band, where the heat stays in a thin layer and the loop gain is no longer
//! `1/c_v` of a diffuse deposit. ⚠️ That regime needs a regularisation first
//! (gradient or non-local plasticity) because the local continuum problem is
//! ill-posed there; a monolithic Newton would converge faster to a
//! mesh-dependent answer, which is not an improvement.
//!
//! **What is missing to build it** (counted, not assumed): a general
//! non-symmetric Krylov solver — zero in `src/`; `project_pressure_bicgstab`
//! (reached from `CfdSolver::step_with_pressure_solver`) has its operator
//! fixed to the MAC-grid 7-point Laplacian, and the `conjugate_gradient` in
//! the FEM modules assumes symmetry — and a sparse direct factorisation, also zero,
//! with `Fix128` pivot growth unmeasured. So even a Jacobian-free Newton–Krylov
//! route starts with a new linear solver, not with the coupling.
//!
//! The decision, as of 2026-10-02: **not now**. Re-measure the table with the
//! test above when a scene produces `|λ_max| > 0.9` at a handbook `c_v`, or
//! when a `Stagnated` verdict appears that the clamp does not explain.

use crate::math::Fix128;

// ============================================================================
// Residual norms
// ============================================================================

/// L∞ norm `max|rᵢ|` of a residual vector.
///
/// This is the norm to use under `Fix128`. It performs no multiplication, so it
/// neither truncates small components to zero nor wraps on large ones: it stays
/// faithful across the whole range in which the components themselves are
/// representable. See the module documentation for the measured failure of the
/// squared L2 norm in both directions.
///
/// An empty slice has norm zero.
#[must_use]
pub fn residual_norm_inf(residual: &[Fix128]) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for &component in residual {
        let magnitude = component.abs();
        if magnitude > worst {
            worst = magnitude;
        }
    }
    worst
}

/// Smallest magnitude whose square survives `Fix128` truncation.
///
/// `√(2⁻⁶⁴) = 2.328306…e-10`, written as its exact `Q64.64` encoding
/// `2⁻³² = 2³² · 2⁻⁶⁴`. A component below this squares to zero, which is
/// truncation working as designed rather than a fault.
pub const L2_TERM_FLOOR: Fix128 = Fix128::from_raw(0, 1 << 32);

/// L2 norm `√(Σ rᵢ²)`, or an error if the accumulation stopped being faithful.
///
/// Provided for callers that need the Euclidean norm. Squaring is where
/// `Fix128` loses faithfulness, so this reports the loss instead of returning a
/// plausible number:
///
/// - a partial sum that **decreases** cannot happen when adding squares, so it
///   is reported as [`CoupledIterationError::ArithmeticWrapped`];
/// - a component at or above [`L2_TERM_FLOOR`] whose square is zero has had its
///   contribution silently dropped, and is reported the same way.
///
/// Components below [`L2_TERM_FLOOR`] legitimately square to zero and are
/// accepted.
///
/// The `sweeps` field of the returned error carries the **index of the
/// offending component**, not a sweep count, because this function is not
/// itself part of a sweep sequence.
///
/// # Errors
///
/// Returns [`CoupledIterationError::ArithmeticWrapped`] when the accumulation is
/// no longer faithful, as described above.
pub fn residual_norm_l2_checked(residual: &[Fix128]) -> Result<Fix128, CoupledIterationError> {
    let mut sum = Fix128::ZERO;
    for (index, &component) in residual.iter().enumerate() {
        let magnitude = component.abs();
        // Components below `L2_TERM_FLOOR` square to zero by design (their
        // product is below the 2⁻⁶⁴ resolution) and so add nothing to the
        // sum: accept them without going through the overflow check. The
        // `is_negative` guard is load-bearing — `abs()` of the most
        // negative value stays negative, and that component must still
        // reach the check below, which refuses it.
        if !magnitude.is_negative() && magnitude < L2_TERM_FLOOR {
            continue;
        }
        // `checked_mul` catches every overflow of the squaring step itself —
        // including the window where the wrapped product is still positive
        // (e.g. 4.5e9² wraps to ~1.9e18, which is neither zero nor a decrease
        // of the running sum) — not just the two symptoms a plain `*` can
        // leave behind (wrapping to zero, or making the sum decrease).
        let Some(square) = magnitude.checked_mul(magnitude) else {
            // The saturating cast keeps the guard honest rather than wrapping
            // the very field that reports a wrap.
            let at = u32::try_from(index).unwrap_or(u32::MAX);
            return Err(CoupledIterationError::ArithmeticWrapped { sweeps: at });
        };
        let next = sum + square;
        if next < sum {
            let at = u32::try_from(index).unwrap_or(u32::MAX);
            return Err(CoupledIterationError::ArithmeticWrapped { sweeps: at });
        }
        sum = next;
    }
    Ok(sum.sqrt())
}

// ============================================================================
// Block equilibration
// ============================================================================

/// A power-of-two factor for block equilibration.
///
/// Equilibration exists to keep every block's contribution above the product
/// floor (see the module documentation). The factor is restricted to a power of
/// two so that **the equilibration itself introduces no rounding**: dividing is
/// an arithmetic shift and multiplying back is exact. Dividing by a general
/// magnitude such as `‖block‖` would make the repair a new error source, which
/// is the wrong trade when the thing being repaired is loss of precision.
///
/// This crate has already paid for that lesson once, in polar decomposition:
/// normalising by the true norm moved the rounded map's fixed point onto a
/// neighbouring one and broke idempotence, while rounding up to a power of two
/// did not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EquilibrationScale {
    /// The factor is `2^exponent`.
    exponent: u32,
}

impl EquilibrationScale {
    /// The largest exponent accepted, chosen so that `2^exponent` and its
    /// products stay inside the integer field.
    pub const MAX_EXPONENT: u32 = 62;

    /// The identity factor, `2⁰ = 1`.
    pub const IDENTITY: Self = Self { exponent: 0 };

    /// Smallest power of two greater than or equal to `magnitude`.
    ///
    /// Dividing a block by this maps its largest entry into `(1/2, 1]`, so the
    /// block becomes `O(1)` without any entry crossing zero.
    ///
    /// A zero or negative magnitude yields [`EquilibrationScale::IDENTITY`]:
    /// there is nothing to equilibrate, and silently inventing a factor would
    /// change a caller's numbers for no reason.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigFault::ScaleOutOfRange`] if `magnitude` needs an
    /// exponent above [`EquilibrationScale::MAX_EXPONENT`].
    // ALLOW-UNWIRED: wiring debt Backlog `residual_norm_l2_equilibrated` (CG の停止 norm を equilibrate する別 task), oracle tests/analytic_coupled_wiring.rs
    pub fn covering(magnitude: Fix128) -> Result<Self, ConfigFault> {
        if magnitude <= Fix128::ZERO {
            return Ok(Self::IDENTITY);
        }
        let mut exponent = 0u32;
        let mut bound = Fix128::ONE;
        while bound < magnitude {
            exponent += 1;
            if exponent > Self::MAX_EXPONENT {
                return Err(ConfigFault::ScaleOutOfRange);
            }
            bound = Fix128::from_int(1i64 << exponent);
        }
        Ok(Self { exponent })
    }

    /// The exponent, where the factor is `2^exponent`.
    // ALLOW-UNWIRED: wiring debt Backlog `residual_norm_l2_equilibrated` (CG の停止 norm を equilibrate する別 task), oracle tests/analytic_coupled_wiring.rs
    #[must_use]
    pub const fn exponent(self) -> u32 {
        self.exponent
    }

    /// The factor itself.
    #[must_use]
    pub fn factor(self) -> Fix128 {
        Fix128::from_int(1i64 << self.exponent)
    }

    /// Divide by the factor. An arithmetic shift, so the only loss is the
    /// `exponent` low bits that shift off the bottom.
    // ALLOW-UNWIRED: wiring debt Backlog `residual_norm_l2_equilibrated` (CG の停止 norm を equilibrate する別 task), oracle tests/analytic_coupled_wiring.rs
    #[must_use]
    pub const fn scale_down(self, value: Fix128) -> Fix128 {
        value.shr_bits(self.exponent)
    }

    /// Multiply by the factor. Exact, provided the result stays in range.
    // ALLOW-UNWIRED: wiring debt Backlog `residual_norm_l2_equilibrated` (CG の停止 norm を equilibrate する別 task), oracle tests/analytic_coupled_wiring.rs
    #[must_use]
    pub fn scale_up(self, value: Fix128) -> Fix128 {
        value * self.factor()
    }

    /// Largest error a `scale_down` then `scale_up` round trip can introduce.
    ///
    /// The shift discards `exponent` low bits, each worth at most `2⁻⁶⁴`, and
    /// multiplying back magnifies that by `2^exponent`. So the bound is
    /// `(2^exponent − 1) · 2⁻⁶⁴`, and it is **not zero**: a caller that needs
    /// the original value back must keep it rather than reconstruct it.
    ///
    /// The other order — `scale_up` then `scale_down` — is exact whenever the
    /// intermediate stays in range, which is why a solver equilibrates its
    /// operator and right-hand side rather than its solution.
    // ALLOW-UNWIRED: wiring debt Backlog `residual_norm_l2_equilibrated` (CG の停止 norm を equilibrate する別 task), oracle tests/analytic_coupled_wiring.rs
    #[must_use]
    pub fn round_trip_bound(self) -> Fix128 {
        if self.exponent == 0 {
            return Fix128::ZERO;
        }
        Fix128::from_raw(0, (1u64 << self.exponent) - 1)
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Why a [`SubIterationConfig`] was rejected.
///
/// Separate variants because the caller's next action differs for each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConfigFault {
    /// `stagnation_fraction` was not in `(0, 1)`.
    ///
    /// The stopping point is `best_sweep / (1 − fraction)`, so `fraction ≥ 1`
    /// can never fire and `fraction ≤ 0` fires immediately.
    StagnationFractionOutOfRange,
    /// `max_sweeps` was zero, so no sweep would ever run.
    ZeroSweepBudget,
    /// `relative_tolerance` was zero or negative.
    ///
    /// Zero is rejected deliberately: truncating multiplication leaves no exact
    /// fixed point, so a zero tolerance is unreachable by construction.
    NonPositiveTolerance,
    /// `divergence_ratio` was not greater than one.
    ///
    /// One is the marginal ratio, measured to be exactly flat, so a threshold
    /// at or below one would classify a stationary iteration as divergent.
    DivergenceRatioNotAboveOne,
    /// `ratio_samples` was below two, so no ratio could be formed.
    InsufficientRatioSamples,
    /// An equilibration factor above `2^62` was requested.
    ///
    /// A block needing that much rescaling is not a scaling problem; its units
    /// or its formulation want looking at.
    ScaleOutOfRange,
}

/// Why a sub-iteration did not produce a converged coupled state.
///
/// The variants are separate because the caller's next action differs:
/// `Diverging` means change the splitting or add relaxation, `NotConverged`
/// means raise the budget, `Stagnated` means the tolerance is below the
/// reachable floor, and `ArithmeticWrapped` means the numbers left the range in
/// which `Fix128` is faithful, so nothing downstream should be trusted.
///
/// Collapsing these into one `Option` or one variant has misled this crate
/// before, because "raise the budget" and "change the design" are opposite
/// actions.
///
/// Construction is available to downstream crates (the enum is
/// `#[non_exhaustive]`, which restricts exhaustive matching, not variant
/// construction) so that integration tests can assert on an exact value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoupledIterationError {
    /// The splitting is not a contraction: the measured ratio exceeded
    /// [`SubIterationConfig::divergence_ratio`].
    ///
    /// For the added-mass model this ratio equals `m_f/m`, and reducing the
    /// time step does not change it.
    Diverging {
        /// Contraction ratio measured over the first sweeps.
        observed_ratio: Fix128,
        /// Sweeps taken when the verdict was reached.
        sweeps: u32,
    },
    /// The residual grew for at least two consecutive sweeps and then
    /// decreased, which a growing sequence cannot do: `Fix128` wrapped.
    ///
    /// This is an invariant guard rather than a closed-form test, and it is the
    /// only detection path for a silent wrap — magnitude-based tests cannot see
    /// one, because wrapping makes the magnitude *smaller*.
    ArithmeticWrapped {
        /// Sweeps taken when the wrap was detected.
        sweeps: u32,
    },
    /// The budget ran out while the residual was still improving.
    NotConverged {
        /// Sweeps taken.
        sweeps: u32,
        /// Final residual norm.
        residual: Fix128,
    },
    /// The residual stopped improving while still above the tolerance.
    ///
    /// ⚠️ At least three different mechanisms end here, and they want opposite
    /// responses. The crate's conjugate-gradient solver reports all of them as
    /// one bare `Stagnated`, and that has already caused two separate
    /// misdiagnoses, so this variant carries what is needed to tell them apart
    /// rather than guessing on the caller's behalf:
    ///
    /// | mechanism | how it reads | what to do |
    /// |---|---|---|
    /// | the representation's floor was reached | `best_residual` far below `first_residual`, and tiny in absolute terms | accept it, or raise `relative_tolerance` |
    /// | the sweep saturated (a guard fired early) | `sweeps` small, `sweeps_since_improvement` small | look at the splitting |
    /// | the window is shorter than the transient | `best_residual` still near `first_residual` | raise `stagnation_fraction` or `max_sweeps` |
    ///
    /// The classification is deliberately **not** made here: it would need a
    /// threshold separating "near the floor" from "still in the transient", and
    /// a guessed threshold is exactly the fault this module exists to avoid.
    Stagnated {
        /// Sweeps taken.
        sweeps: u32,
        /// Best residual norm reached.
        best_residual: Fix128,
        /// Residual norm the sub-iteration started from.
        ///
        /// `best_residual / first_residual` says whether any progress was made
        /// at all, which is what separates a floor from a short window.
        first_residual: Fix128,
        /// Sweeps since `best_residual` was last improved.
        sweeps_since_improvement: u32,
    },
    /// The configuration was rejected before any sweep ran.
    InvalidConfig {
        /// Which field was at fault.
        fault: ConfigFault,
    },
}

// ============================================================================
// Configuration
// ============================================================================

/// Stopping rules for a partitioned sub-iteration.
///
/// Every field is relative to something; see the module documentation for why
/// absolute thresholds break here, and [`SubIterationConfig::validate`] for the
/// algebraic reason each bound sits where it does.
///
/// Construct with [`SubIterationConfig::new`], which refuses an invalid
/// combination, or start from [`SubIterationConfig::default`] and override with
/// functional update syntax (`..Default::default()`) so that a future field
/// does not break the call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubIterationConfig {
    /// Maximum sweeps per coupled step.
    pub max_sweeps: u32,
    /// Convergence target, as a fraction of the **first** residual norm.
    pub relative_tolerance: Fix128,
    /// Stagnation window as a fraction of the sweeps taken so far.
    ///
    /// Stopping point is `best_sweep / (1 − fraction)`; must be in `(0, 1)`.
    pub stagnation_fraction: Fix128,
    /// Contraction ratio above which the splitting is declared divergent.
    ///
    /// Must exceed one: the marginal ratio is measured to be exactly flat, and
    /// there is a dead band about one ulp wide just below one.
    pub divergence_ratio: Fix128,
    /// How many sweeps to see before reading the contraction ratio.
    ///
    /// Read early on purpose. The ratio is formed by dividing successive
    /// residuals, and for a *contracting* splitting the residual shrinks toward
    /// the rounding floor, so the truncation error of that division grows
    /// geometrically. Measured deviation from the closed form, in ulp of the
    /// raw `Q64.64` value:
    ///
    /// | `m_f/m` | sweep 2 | sweep 3 | sweep 4 | sweep 5 |
    /// |---|---|---|---|---|
    /// | `1/8` | 3 | 18 | 144 | 1152 |
    /// | `1/4` | 5 | 20 | 80 | 320 |
    /// | `1/2` | 3 | 6 | 12 | 24 |
    /// | `2` | 2 | 0 | 1 | 0 |
    /// | `8` | 8 | 0 | 1 | 0 |
    ///
    /// The deviation multiplies by about `1/ρ` per sweep while `ρ < 1`, and
    /// stays at a couple of ulp while `ρ ≥ 1` because there the residual grows
    /// away from the floor. A caller that needs an accurate `ρ` should read it
    /// as early as the transient allows; the default of 4 is chosen for
    /// *classification*, where the threshold sits far above this noise.
    pub ratio_samples: u32,
}

impl SubIterationConfig {
    /// Build a configuration, refusing an invalid combination.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigFault`] found by
    /// [`SubIterationConfig::validate`].
    pub fn new(
        max_sweeps: u32,
        relative_tolerance: Fix128,
        stagnation_fraction: Fix128,
        divergence_ratio: Fix128,
        ratio_samples: u32,
    ) -> Result<Self, ConfigFault> {
        let config = Self {
            max_sweeps,
            relative_tolerance,
            stagnation_fraction,
            divergence_ratio,
            ratio_samples,
        };
        config.validate()?;
        Ok(config)
    }

    /// Check the fields against the bounds their stopping rules require.
    ///
    /// # Errors
    ///
    /// Returns the first [`ConfigFault`] found.
    pub fn validate(&self) -> Result<(), ConfigFault> {
        if self.max_sweeps == 0 {
            return Err(ConfigFault::ZeroSweepBudget);
        }
        if self.relative_tolerance <= Fix128::ZERO {
            return Err(ConfigFault::NonPositiveTolerance);
        }
        if self.stagnation_fraction <= Fix128::ZERO || self.stagnation_fraction >= Fix128::ONE {
            return Err(ConfigFault::StagnationFractionOutOfRange);
        }
        if self.divergence_ratio <= Fix128::ONE {
            return Err(ConfigFault::DivergenceRatioNotAboveOne);
        }
        if self.ratio_samples < 2 {
            return Err(ConfigFault::InsufficientRatioSamples);
        }
        Ok(())
    }
}

impl Default for SubIterationConfig {
    /// Budget 64 sweeps; target `1/1024` of the first residual; stagnation at
    /// twice the best sweep; divergence above `1 + 1/16`; ratio from 4 sweeps.
    ///
    /// Each value is derived rather than picked:
    ///
    /// - **64 sweeps** — at `m_f/m = 0.5` the model problem needs 11 sweeps to
    ///   reach `1/1024`; 64 leaves room for ratios approaching one while
    ///   staying under the sweep at which a diverging `m_f/m = 1.5` wraps (59).
    /// - **`1/1024`** — ten halvings, comfortably above the point where the
    ///   observed ratio starts drifting at the rounding floor.
    /// - **`1/2`** — stop at `best/(1 − 1/2)`, i.e. twice the best sweep.
    /// - **`1 + 1/16`** — above one by far more than the one-ulp dead band, and
    ///   far below the smallest ratio any real added-mass case produces.
    /// - **4 sweeps** — the least that gives a ratio after the first transient.
    ///
    /// The values are pinned by a test, so changing one is a visible decision
    /// rather than a silent one.
    fn default() -> Self {
        Self {
            max_sweeps: 64,
            relative_tolerance: Fix128::from_ratio(1, 1024),
            stagnation_fraction: Fix128::from_ratio(1, 2),
            divergence_ratio: Fix128::ONE + Fix128::from_ratio(1, 16),
            ratio_samples: 4,
        }
    }
}

// ============================================================================
// Monitor
// ============================================================================

/// What the monitor concluded from the residual sequence so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MonitorVerdict {
    /// Keep sweeping.
    Continue,
    /// The residual fell to or below `relative_tolerance × first residual`.
    Converged,
    /// Stop; the error says why.
    Stop(CoupledIterationError),
}

/// Watches the residual sequence of a partitioned sub-iteration.
///
/// Feed it one residual norm per sweep through [`ContractionMonitor::observe`].
/// It measures the contraction ratio of the splitting, applies the relative
/// stopping rules, and detects a silent `Fix128` wrap.
#[derive(Clone, Copy, Debug)]
pub struct ContractionMonitor {
    config: SubIterationConfig,
    sweeps: u32,
    first: Fix128,
    previous: Fix128,
    best: Fix128,
    best_sweep: u32,
    /// Consecutive sweeps in which the residual grew.
    growth_streak: u32,
    /// Ratio measured once `ratio_samples` sweeps have been seen.
    observed_ratio: Option<Fix128>,
}

impl ContractionMonitor {
    /// Create a monitor.
    ///
    /// # Errors
    ///
    /// Returns [`CoupledIterationError::InvalidConfig`] if the configuration
    /// fails [`SubIterationConfig::validate`].
    pub fn new(config: SubIterationConfig) -> Result<Self, CoupledIterationError> {
        config
            .validate()
            .map_err(|fault| CoupledIterationError::InvalidConfig { fault })?;
        Ok(Self {
            config,
            sweeps: 0,
            first: Fix128::ZERO,
            previous: Fix128::ZERO,
            best: Fix128::ZERO,
            best_sweep: 0,
            growth_streak: 0,
            observed_ratio: None,
        })
    }

    /// Sweeps observed so far.
    #[must_use]
    pub const fn sweeps(&self) -> u32 {
        self.sweeps
    }

    /// Contraction ratio measured once [`SubIterationConfig::ratio_samples`]
    /// sweeps have been seen.
    ///
    /// For the added-mass model this equals `m_f/m`, independently of the time
    /// step.
    #[must_use]
    pub const fn observed_ratio(&self) -> Option<Fix128> {
        self.observed_ratio
    }

    /// Best (smallest) residual norm seen.
    // ALLOW-UNWIRED: wiring debt Backlog 2.0.0 列 (`SubIterationReport` に field を足せるまで report に載せられない), oracle tests/analytic_added_mass_coupling.rs (best_residual)
    #[must_use]
    pub const fn best_residual(&self) -> Fix128 {
        self.best
    }

    /// Record one sweep's residual norm and return what to do next.
    ///
    /// Pass an L∞ norm ([`residual_norm_inf`]); a squared L2 norm is measured to
    /// read zero both when converging and when diverging.
    pub fn observe(&mut self, residual: Fix128) -> MonitorVerdict {
        let previous = self.previous;
        let previous_streak = self.growth_streak;
        self.sweeps += 1;
        self.previous = residual;

        if self.sweeps == 1 {
            self.first = residual;
            self.best = residual;
            self.best_sweep = 1;
            return if residual.is_zero() {
                MonitorVerdict::Converged
            } else {
                MonitorVerdict::Continue
            };
        }

        // --- wrap detection -------------------------------------------------
        // A sequence that has grown for two consecutive sweeps cannot then
        // decrease. Under `Fix128` it can appear to, because leaving the range
        // wraps the magnitude to a smaller value. This is the only signature a
        // silent wrap leaves; magnitude tests cannot see it. Two growths are
        // required so that a single overshoot — ordinary transient behaviour in
        // a partitioned sweep — is not mistaken for one.
        if residual > previous {
            self.growth_streak = previous_streak + 1;
        } else {
            self.growth_streak = 0;
            if previous_streak >= 2 {
                return MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped {
                    sweeps: self.sweeps,
                });
            }
        }

        // --- contraction ratio ----------------------------------------------
        // Checked before stagnation: a growing residual is a divergence
        // question, and answering it as stagnation would send the caller to
        // raise the budget instead of changing the splitting.
        if self.observed_ratio.is_none() && self.sweeps >= self.config.ratio_samples {
            let ratio = if previous.is_zero() {
                Fix128::ZERO
            } else {
                residual / previous
            };
            self.observed_ratio = Some(ratio);
            if ratio > self.config.divergence_ratio {
                return MonitorVerdict::Stop(CoupledIterationError::Diverging {
                    observed_ratio: ratio,
                    sweeps: self.sweeps,
                });
            }
        }

        // --- improvement bookkeeping ----------------------------------------
        if residual < self.best {
            self.best = residual;
            self.best_sweep = self.sweeps;
        }

        // --- convergence, relative to the first residual ---------------------
        if residual <= self.first * self.config.relative_tolerance {
            return MonitorVerdict::Converged;
        }

        // --- stagnation ------------------------------------------------------
        // Only once the ratio window has had its say, and only while the
        // residual is not currently growing, for the reason above.
        if self.growth_streak == 0 && self.sweeps > self.config.ratio_samples {
            let denominator = Fix128::ONE - self.config.stagnation_fraction;
            let stop_at = Fix128::from_int(i64::from(self.best_sweep)) / denominator;
            if Fix128::from_int(i64::from(self.sweeps)) > stop_at {
                return MonitorVerdict::Stop(CoupledIterationError::Stagnated {
                    sweeps: self.sweeps,
                    best_residual: self.best,
                    first_residual: self.first,
                    sweeps_since_improvement: self.sweeps - self.best_sweep,
                });
            }
        }

        if self.sweeps >= self.config.max_sweeps {
            return MonitorVerdict::Stop(CoupledIterationError::NotConverged {
                sweeps: self.sweeps,
                residual,
            });
        }

        MonitorVerdict::Continue
    }
}

// ============================================================================
// Driver
// ============================================================================

/// What a completed sub-iteration did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubIterationReport {
    /// Sweeps performed.
    pub sweeps: u32,
    /// Residual norm of the accepted state.
    pub residual: Fix128,
    /// Contraction ratio of the splitting, if enough sweeps ran to measure it.
    pub observed_ratio: Option<Fix128>,
}

/// Run a partitioned sub-iteration to convergence, under monitoring.
///
/// `sweep` performs one staggered pass over the coupled fields — apply each
/// one-directional coupling in turn — and returns the resulting residual norm
/// (use [`residual_norm_inf`]). It receives the zero-based sweep index.
///
/// The caller keeps ownership of the physics; this decides only when to stop,
/// and why.
///
/// # Errors
///
/// Returns a [`CoupledIterationError`]: `Diverging` if the splitting is not a
/// contraction, `ArithmeticWrapped` if the residual sequence shows a silent
/// `Fix128` wrap, or `NotConverged` / `Stagnated` / `InvalidConfig`.
pub fn run_sub_iteration<F>(
    config: SubIterationConfig,
    mut sweep: F,
) -> Result<SubIterationReport, CoupledIterationError>
where
    F: FnMut(u32) -> Fix128,
{
    let mut monitor = ContractionMonitor::new(config)?;
    loop {
        let index = monitor.sweeps();
        let residual = sweep(index);
        match monitor.observe(residual) {
            MonitorVerdict::Continue => {}
            MonitorVerdict::Converged => {
                return Ok(SubIterationReport {
                    sweeps: monitor.sweeps(),
                    residual,
                    observed_ratio: monitor.observed_ratio(),
                });
            }
            MonitorVerdict::Stop(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One sweep of the added-mass model with `m = k = x = 1`.
    fn added_mass_sweep(a: Fix128, mass_ratio: Fix128) -> Fix128 {
        -Fix128::ONE - mass_ratio * a
    }

    fn fixed_point(mass_ratio: Fix128) -> Fix128 {
        -Fix128::ONE / (Fix128::ONE + mass_ratio)
    }

    fn raw(v: Fix128) -> i128 {
        (i128::from(v.hi) << 64) | i128::from(v.lo)
    }

    /// Largest deviation, in ulp of the raw `Q64.64` value, allowed between the
    /// observed contraction ratio and the closed form `m_f/m`.
    ///
    /// Bit equality is **not** available: the ratio is formed by a truncating
    /// division of two residuals. The measured worst case over `m_f/m` in
    /// `1/8 ..= 8` at the default `ratio_samples = 4` is **144 ulp** (at
    /// `m_f/m = 1/8`, where the residual is closest to the rounding floor);
    /// the bound below leaves a factor of about 1.8. In relative terms the
    /// measured worst is `6.2e-17`, so a wrong ratio — which would differ in
    /// its leading digits — cannot hide under this bound.
    const RATIO_ULP_BOUND: u128 = 256;

    #[track_caller]
    fn assert_ratio_matches_closed_form(observed: Option<Fix128>, closed: Fix128) {
        let observed = observed.expect("the ratio window closed, so a ratio exists");
        let deviation = raw(observed).abs_diff(raw(closed));
        assert!(
            deviation <= RATIO_ULP_BOUND,
            "observed ratio {observed:?} deviates from the closed form {closed:?} \
             by {deviation} ulp, above the measured bound {RATIO_ULP_BOUND}"
        );
    }

    #[test]
    fn inf_norm_picks_the_largest_magnitude() {
        let r = [
            Fix128::from_ratio(1, 4),
            Fix128::from_ratio(-3, 4),
            Fix128::from_ratio(1, 2),
        ];
        assert_eq!(residual_norm_inf(&r), Fix128::from_ratio(3, 4));
        assert_eq!(residual_norm_inf(&[]), Fix128::ZERO);
    }

    #[test]
    fn l2_floor_is_the_square_root_of_the_truncation_floor() {
        // 2^-32 squared is 2^-64, the smallest representable positive value,
        // and anything below it squares to zero.
        assert_eq!(L2_TERM_FLOOR * L2_TERM_FLOOR, Fix128::from_raw(0, 1));
        let below = Fix128::from_raw(0, (1u64 << 32) - 1);
        assert!(below < L2_TERM_FLOOR);
        assert!((below * below).is_zero());
    }

    #[test]
    fn l2_accepts_components_that_legitimately_square_to_zero() {
        let tiny = Fix128::from_raw(0, 1);
        assert!(tiny < L2_TERM_FLOOR);
        assert_eq!(residual_norm_l2_checked(&[tiny]), Ok(Fix128::ZERO));
    }

    #[test]
    fn l2_reports_a_wrap_instead_of_a_plausible_zero() {
        // Large enough that the square leaves the faithful range and lands
        // negative, so the running sum decreases. A bare `sqrt(sum)` would
        // answer zero here, which reads as convergence.
        let huge = Fix128::from_int(4_000_000_000);
        assert!(huge.abs() >= L2_TERM_FLOOR);
        assert!(
            (huge * huge).is_negative(),
            "this case exercises the sum path"
        );
        assert_eq!(
            residual_norm_l2_checked(&[huge]),
            Err(CoupledIterationError::ArithmeticWrapped { sweeps: 0 })
        );
    }

    #[test]
    fn l2_reports_a_term_whose_square_wrapped_to_exactly_zero() {
        // The harder half. A magnitude of exactly 2³² squares to 2⁶⁴, whose
        // encoding wraps the integer field to zero — so the square is *zero*,
        // not negative, the running sum does not move, and the monotonicity
        // check cannot see it. A residual component of 4.29e9 would then
        // contribute nothing at all to the norm, and a vector made entirely of
        // such components would report `‖r‖₂ = 0` while completely diverged.
        for exponent in [32u32, 33, 40, 62] {
            let magnitude = Fix128::from_int(1i64 << exponent);
            let square = magnitude * magnitude;
            assert!(
                square.is_zero() && !square.is_negative(),
                "2^{exponent} squared should wrap to exactly zero, got {square:?}"
            );
            assert!(magnitude >= L2_TERM_FLOOR);
            assert_eq!(
                residual_norm_l2_checked(&[magnitude]),
                Err(CoupledIterationError::ArithmeticWrapped { sweeps: 0 }),
                "a silently dropped term at 2^{exponent} must be refused"
            );
        }
    }

    #[test]
    fn an_equilibration_factor_is_the_next_power_of_two_at_or_above_the_magnitude() {
        for (magnitude, expected_exponent) in [
            (Fix128::from_ratio(1, 4), 0u32),
            (Fix128::ONE, 0),
            (Fix128::from_int(2), 1),
            (Fix128::from_int(3), 2),
            (Fix128::from_int(4), 2),
            (Fix128::from_int(5), 3),
            (Fix128::from_int(1024), 10),
        ] {
            let scale = EquilibrationScale::covering(magnitude).expect("within range");
            assert_eq!(
                scale.exponent(),
                expected_exponent,
                "magnitude {magnitude:?} should need 2^{expected_exponent}"
            );
            assert!(
                scale.factor() >= magnitude,
                "the factor must cover the magnitude"
            );
        }
        // Nothing to equilibrate, and inventing a factor would move a caller's
        // numbers for no reason.
        assert_eq!(
            EquilibrationScale::covering(Fix128::ZERO),
            Ok(EquilibrationScale::IDENTITY)
        );
        assert_eq!(
            EquilibrationScale::covering(Fix128::from_int(-8)),
            Ok(EquilibrationScale::IDENTITY)
        );
    }

    #[test]
    fn equilibration_refuses_a_factor_it_cannot_represent() {
        // A block wanting more than 2^62 of rescaling has a units problem, not
        // a scaling problem, and saturating quietly would hide it.
        // 2^62 is the last factor that fits; i64::MAX sits above it.
        assert_eq!(
            EquilibrationScale::covering(Fix128::from_int(1i64 << 62))
                .expect("2^62 is the boundary and must be accepted")
                .exponent(),
            EquilibrationScale::MAX_EXPONENT
        );
        assert_eq!(
            EquilibrationScale::covering(Fix128::from_int(i64::MAX)),
            Err(ConfigFault::ScaleOutOfRange)
        );
    }

    #[test]
    fn scaling_up_then_down_is_exact() {
        // This is the order a solver uses — equilibrate the operator and the
        // right-hand side, not the answer — so it is the one that must be
        // lossless.
        for exponent in [0u32, 1, 8, 20] {
            let scale = EquilibrationScale::covering(Fix128::from_int(1i64 << exponent))
                .expect("within range");
            for raw_value in [1i64, 3, 1_000, -7] {
                let value = Fix128::from_ratio(raw_value, 64);
                assert_eq!(
                    scale.scale_down(scale.scale_up(value)),
                    value,
                    "up-then-down must be exact at 2^{exponent} for {value:?}"
                );
            }
        }
    }

    #[test]
    fn scaling_down_then_up_loses_only_the_shifted_bits() {
        // The other order is *not* an identity: the shift discards the low
        // bits. The bound is stated rather than the identity claimed, because
        // a round trip that silently changed the value is exactly the class of
        // fault this module exists to catch.
        for exponent in [1u32, 4, 12, 30] {
            let scale = EquilibrationScale::covering(Fix128::from_int(1i64 << exponent))
                .expect("within range");
            let bound = scale.round_trip_bound();
            assert!(bound > Fix128::ZERO, "a real shift must have a real bound");

            // A value whose low bits are all set is the worst case.
            let value = Fix128::from_raw(3, u64::MAX);
            let round_tripped = scale.scale_up(scale.scale_down(value));
            let loss = (value - round_tripped).abs();
            assert!(
                loss <= bound,
                "2^{exponent}: round-trip loss {loss:?} exceeded the stated bound {bound:?}"
            );
            assert!(
                loss > Fix128::ZERO,
                "2^{exponent}: this value has low bits set, so some loss must show; \
                 if none does, the bound is describing something that never happens"
            );
        }
        // The identity factor genuinely is an identity.
        let identity = EquilibrationScale::IDENTITY;
        assert_eq!(identity.round_trip_bound(), Fix128::ZERO);
        let value = Fix128::from_raw(3, u64::MAX);
        assert_eq!(identity.scale_up(identity.scale_down(value)), value);
    }

    #[test]
    fn equilibration_lifts_a_block_above_the_product_floor() {
        // The point of the whole exercise. A thermal block sized like PLA
        // against an elastic block is below the floor in raw units; after
        // equilibration its square is representable.
        let elastic = Fix128::from_int(3_500_000);
        let scale = EquilibrationScale::covering(elastic).expect("within range");
        let thermal_raw = Fix128::from_ratio(13, 100_000);
        let equilibrated_elastic = scale.scale_down(elastic);

        assert!(
            equilibrated_elastic <= Fix128::ONE && equilibrated_elastic > Fix128::from_ratio(1, 2),
            "the dominant block should land in (1/2, 1], got {equilibrated_elastic:?}"
        );
        // The dominant block's own square must clear the floor after scaling,
        // which is what lets its contribution survive an inner product.
        assert!(
            equilibrated_elastic >= L2_TERM_FLOOR,
            "the equilibrated block must stay above the product floor"
        );
        // And the raw thermal magnitude, which is what a caller would otherwise
        // have squared, is far below the elastic one: the ratio is the problem.
        assert!(thermal_raw < elastic);
    }

    #[test]
    fn config_rejects_a_stagnation_fraction_that_can_never_fire() {
        // Stop point is best/(1 - f); f >= 1 makes it non-positive or infinite.
        for fraction in [Fix128::ONE, Fix128::ZERO, Fix128::from_int(2)] {
            let config = SubIterationConfig {
                stagnation_fraction: fraction,
                ..SubIterationConfig::default()
            };
            assert_eq!(
                config.validate(),
                Err(ConfigFault::StagnationFractionOutOfRange)
            );
        }
    }

    #[test]
    fn config_rejects_an_unreachable_zero_tolerance() {
        let config = SubIterationConfig {
            relative_tolerance: Fix128::ZERO,
            ..SubIterationConfig::default()
        };
        assert_eq!(config.validate(), Err(ConfigFault::NonPositiveTolerance));
    }

    #[test]
    fn config_rejects_a_divergence_threshold_at_the_marginal_ratio() {
        let config = SubIterationConfig {
            divergence_ratio: Fix128::ONE,
            ..SubIterationConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(ConfigFault::DivergenceRatioNotAboveOne)
        );
    }

    #[test]
    fn config_rejects_an_empty_budget_and_too_few_ratio_samples() {
        let config = SubIterationConfig {
            max_sweeps: 0,
            ..SubIterationConfig::default()
        };
        assert_eq!(config.validate(), Err(ConfigFault::ZeroSweepBudget));

        let config = SubIterationConfig {
            ratio_samples: 1,
            ..SubIterationConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(ConfigFault::InsufficientRatioSamples)
        );
    }

    #[test]
    fn new_refuses_what_validate_refuses() {
        assert_eq!(
            SubIterationConfig::new(
                0,
                Fix128::from_ratio(1, 1024),
                Fix128::from_ratio(1, 2),
                Fix128::from_int(2),
                4,
            ),
            Err(ConfigFault::ZeroSweepBudget)
        );
        assert!(SubIterationConfig::new(
            64,
            Fix128::from_ratio(1, 1024),
            Fix128::from_ratio(1, 2),
            Fix128::from_int(2),
            4,
        )
        .is_ok());
    }

    #[test]
    fn default_config_is_pinned_and_valid() {
        // Pinned so that changing a stopping rule is a visible decision. The
        // derivation of each value is in the `Default` documentation.
        let config = SubIterationConfig::default();
        assert_eq!(config.validate(), Ok(()));
        assert_eq!(config.max_sweeps, 64);
        assert_eq!(config.relative_tolerance, Fix128::from_ratio(1, 1024));
        assert_eq!(config.stagnation_fraction, Fix128::from_ratio(1, 2));
        assert_eq!(
            config.divergence_ratio,
            Fix128::ONE + Fix128::from_ratio(1, 16)
        );
        assert_eq!(config.ratio_samples, 4);
    }

    #[test]
    fn a_contracting_splitting_converges_and_reports_the_mass_ratio() {
        let mass_ratio = Fix128::from_ratio(1, 2);
        let target = fixed_point(mass_ratio);
        let mut a = Fix128::ZERO;
        let report = run_sub_iteration(SubIterationConfig::default(), |_| {
            a = added_mass_sweep(a, mass_ratio);
            (a - target).abs()
        })
        .expect("m_f/m = 1/2 is a contraction");
        assert_ratio_matches_closed_form(report.observed_ratio, mass_ratio);
    }

    #[test]
    fn a_divergent_splitting_is_reported_as_divergent_not_as_a_wrap() {
        // The diagnosis matters: `Diverging` sends the caller to the splitting,
        // `Stagnated` would send them to the tolerance, `NotConverged` to the
        // budget. Only the first is the right next action.
        let mass_ratio = Fix128::from_int(2);
        let target = fixed_point(mass_ratio);
        let mut a = Fix128::ZERO;
        let error = run_sub_iteration(SubIterationConfig::default(), |_| {
            a = added_mass_sweep(a, mass_ratio);
            (a - target).abs()
        })
        .expect_err("m_f/m = 2 is not a contraction");
        match error {
            CoupledIterationError::Diverging { observed_ratio, .. } => {
                assert_ratio_matches_closed_form(Some(observed_ratio), mass_ratio);
            }
            other => panic!("expected Diverging, got {other:?}"),
        }
    }

    #[test]
    fn the_wrap_guard_fires_on_growth_followed_by_a_decrease() {
        // Divergence detection is disabled by a threshold above the growth
        // ratio, so the wrap guard is what this exercises.
        let config = SubIterationConfig {
            divergence_ratio: Fix128::from_int(1000),
            ..SubIterationConfig::default()
        };
        let mut monitor = ContractionMonitor::new(config).expect("valid config");
        for value in [1i64, 2, 4, 8] {
            assert_eq!(
                monitor.observe(Fix128::from_int(value)),
                MonitorVerdict::Continue
            );
        }
        assert_eq!(
            monitor.observe(Fix128::from_int(3)),
            MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped { sweeps: 5 })
        );
    }

    #[test]
    fn the_wrap_guard_fires_at_exactly_two_growths() {
        // The boundary itself. Two growths then a decrease is the least that
        // must fire; without this case a guard demanding three growths would
        // still satisfy the test above, which grows three times.
        let config = SubIterationConfig {
            divergence_ratio: Fix128::from_int(1000),
            ..SubIterationConfig::default()
        };
        let mut monitor = ContractionMonitor::new(config).expect("valid config");
        for value in [1i64, 2, 4] {
            assert_eq!(
                monitor.observe(Fix128::from_int(value)),
                MonitorVerdict::Continue
            );
        }
        assert_eq!(
            monitor.observe(Fix128::from_int(3)),
            MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped { sweeps: 4 })
        );
    }

    #[test]
    fn the_wrap_guard_stays_silent_on_a_decreasing_sequence() {
        // Control: a guard that fired here would reject every healthy solve.
        let mut monitor =
            ContractionMonitor::new(SubIterationConfig::default()).expect("valid config");
        let mut residual = Fix128::ONE;
        for _ in 0..8 {
            let verdict = monitor.observe(residual);
            assert!(
                !matches!(
                    verdict,
                    MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped { .. })
                ),
                "wrap guard fired on a monotonically decreasing sequence"
            );
            residual = residual / Fix128::from_int(2);
        }
    }

    #[test]
    fn the_wrap_guard_tolerates_a_single_overshoot() {
        // One sweep of growth then a decrease is ordinary transient behaviour,
        // so two consecutive growths are required.
        let mut monitor =
            ContractionMonitor::new(SubIterationConfig::default()).expect("valid config");
        assert_eq!(monitor.observe(Fix128::ONE), MonitorVerdict::Continue);
        assert_eq!(
            monitor.observe(Fix128::from_ratio(3, 2)),
            MonitorVerdict::Continue
        );
        assert!(!matches!(
            monitor.observe(Fix128::from_ratio(1, 2)),
            MonitorVerdict::Stop(CoupledIterationError::ArithmeticWrapped { .. })
        ));
    }

    #[test]
    fn stagnation_reports_enough_to_tell_the_mechanisms_apart() {
        // A residual that improves once and then plateaus far above the floor:
        // the "window shorter than the transient" mechanism. The fields must
        // make that readable, because a bare `Stagnated` sends a caller to the
        // tolerance when the budget is what needs raising.
        let mut monitor =
            ContractionMonitor::new(SubIterationConfig::default()).expect("valid config");
        assert_eq!(monitor.observe(Fix128::ONE), MonitorVerdict::Continue);
        let plateau = Fix128::from_ratio(9, 10);
        let mut verdict = MonitorVerdict::Continue;
        for _ in 0..8 {
            verdict = monitor.observe(plateau);
            if matches!(verdict, MonitorVerdict::Stop(_)) {
                break;
            }
        }
        match verdict {
            MonitorVerdict::Stop(CoupledIterationError::Stagnated {
                best_residual,
                first_residual,
                sweeps_since_improvement,
                ..
            }) => {
                assert_eq!(best_residual, plateau);
                assert_eq!(first_residual, Fix128::ONE);
                assert!(
                    sweeps_since_improvement > 0,
                    "a stagnation report must say how long it has been flat"
                );
                // The reading that separates this from a floor: almost no
                // progress was made, so the floor is nowhere near.
                assert!(
                    best_residual > first_residual / Fix128::from_int(2),
                    "this scene barely improved, which is what marks it as a short \
                     window rather than a reached floor"
                );
            }
            other => panic!("expected Stagnated, got {other:?}"),
        }
    }

    #[test]
    fn an_invalid_config_is_reported_before_any_sweep_runs() {
        let config = SubIterationConfig {
            max_sweeps: 0,
            ..SubIterationConfig::default()
        };
        let mut sweeps_run = 0u32;
        let error = run_sub_iteration(config, |_| {
            sweeps_run += 1;
            Fix128::ONE
        })
        .expect_err("zero budget is invalid");
        assert_eq!(
            error,
            CoupledIterationError::InvalidConfig {
                fault: ConfigFault::ZeroSweepBudget
            }
        );
        assert_eq!(sweeps_run, 0, "no sweep may run under an invalid config");
    }
}
