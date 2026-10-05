//! General Krylov solvers on [`Fix128`]: GMRES(m) and BiCGStab, with Jacobi /
//! block Jacobi preconditioning and **mandatory block equilibration** for
//! coupled systems.
//!
//! Before this module the crate had four private conjugate gradients (one per
//! FEM module, each assuming a symmetric positive definite operator), a
//! `BiCGStab` whose operator is fixed to the MAC-grid Laplacian, and a
//! Newton–Krylov step whose inner solve is again CG. None of them takes a
//! general, non-symmetric operator, which is the first missing piece listed in
//! [`crate::coupled_iteration`] ("Why there is still no monolithic assembly").
//! This module is that piece and nothing more: it does not assemble a coupled
//! Jacobian and it does not change any existing solver.
//!
//! # What is provided
//!
//! | item | role |
//! |---|---|
//! | [`LinearOperator`] | `y = A x`, matrix-free |
//! | [`Preconditioner`] | `z = M⁻¹ r`, applied on the **right** |
//! | [`DenseMatrix`] / [`FnOperator`] | adapters (a small dense matrix, a closure) |
//! | [`gmres`] | restarted GMRES(m): modified Gram–Schmidt Arnoldi + Givens rotations |
//! | [`bicgstab`] | van der Vorst `BiCGStab` |
//! | [`IdentityPreconditioner`] / [`JacobiPreconditioner`] / [`BlockJacobiPreconditioner`] | preconditioners |
//! | [`BlockEquilibration`] + [`solve_equilibrated`] | the only entry for a block-coupled system |
//!
//! # Arithmetic: why every norm and inner product is rescaled by a power of two
//!
//! `Fix128` resolves `2⁻⁶⁴` **absolutely**. A product `a·b` below that floor
//! truncates to zero, and an O(1) value has a relative resolution of `2⁻⁶⁴`
//! while a value of `1e-12` has only `5e-8`. A Krylov method normalises its
//! basis vectors, so it constantly forms inner products whose operands are far
//! from O(1) (a residual that has converged by twelve decades, a matrix whose
//! entries are `3.5e6`).
//!
//! Every inner product and norm here therefore first shifts each operand by
//! the power of two that puts its largest entry in `[1/2, 1)`, accumulates the
//! products of the shifted entries, and shifts the sum back. A shift by a
//! power of two is exact upwards and floors downwards by less than one unit of
//! the target's own resolution, so this introduces no error of its own — it
//! only stops the representation from discarding precision the operands
//! actually carry. The rescaling is *per vector*: it cannot rescue entries of
//! one vector that are `2³²` times smaller than that vector's largest entry,
//! which is exactly the block-coupling problem below.
//!
//! The sum of products is order-independent bit for bit: each product is
//! formed independently and `Fix128` addition is wrapping integer addition,
//! which is associative and commutative. A permuted system therefore yields
//! the permuted solution with identical bits (`tests/analytic_linear_solver.rs`
//! checks this).
//!
//! # Block-coupled systems must be equilibrated
//!
//! When the unknown vector concatenates fields with different units (a
//! displacement block and a temperature block), each Krylov basis vector
//! carries both, normalised so its largest entry is O(1). A block whose
//! operator magnitude is `ρ` times the largest block's ends up with entries of
//! relative size `ρ`, and once `ρ² < 2⁻⁶⁴` — i.e. `ρ <`
//! [`L2_TERM_FLOOR`](crate::coupled_iteration::L2_TERM_FLOOR) `= 2⁻³²` — every product that block contributes to an
//! inner product is zero. For PLA thermo-elasticity `ρ = 1/2.69e10` (the table
//! in [`crate::coupled_iteration`]), so the thermal block vanishes from the
//! orthogonalisation without an error being raised anywhere.
//!
//! [`BlockEquilibration::try_new`] computes a power-of-two factor `dᵢ` per
//! block so that `dᵢ² · mᵢ ∈ [1/4, 4]`, and [`solve_equilibrated`] solves
//! `(D A D) x̃ = D b`, `x = D x̃`. The factors are [`EquilibrationScale`](crate::coupled_iteration::EquilibrationScale) values
//! (power-of-two, exponent at most
//! [`EquilibrationScale::MAX_EXPONENT`](crate::coupled_iteration::EquilibrationScale::MAX_EXPONENT)), so scaling introduces no rounding of
//! its own. [`BlockEquilibration::unscaled`] is the explicitly named entry that
//! applies no scaling; it still measures every block and returns
//! [`LinearSolverError::BlockBelowProductFloor`] when one is below the floor,
//! so neither entry lets a block vanish silently.
//!
//! [`gmres`] and [`bicgstab`] themselves treat their operator as a single
//! field. Handing them a multi-field operator directly bypasses this check;
//! that is what [`solve_equilibrated`] is for.
//!
//! # Stopping and failure verdicts
//!
//! Converged when `‖b − A x‖₂ ≤ max(relative_tolerance · ‖b‖₂,
//! absolute_tolerance)`, judged on the **true** residual (recomputed, never the
//! recurrence alone). The failures are separate because the caller's next step
//! differs:
//!
//! - [`LinearSolverError::NotConverged`]: the iteration budget ran out while the
//!   residual was still improving — raise the budget or precondition.
//! - [`LinearSolverError::Stagnated`]: the best residual did not improve by
//!   `1/1024` of itself over a window of iterations — the tolerance is below
//!   what the arithmetic can reach, or a restarted GMRES has stalled.
//! - [`LinearSolverError::Breakdown`]: the method cannot take the next step
//!   algebraically (a singular projected matrix in GMRES, a vanishing
//!   bi-orthogonality or stabilisation coefficient in `BiCGStab`). A zero
//!   operator or an inconsistent singular system ends here.
//!
//! No test compares against zero: breakdown is declared when the relevant
//! quantity falls below [`BREAKDOWN_RELATIVE`] `= 2⁻⁴⁰` of the scale it is
//! formed from. The margin over the `2⁻⁶⁴` resolution (`2²⁴`) covers the
//! rounding of a modified Gram–Schmidt pass of a few hundred vectors; a system
//! whose condition number exceeds about `2⁴⁰` is reported as breakdown rather
//! than solved to digits the representation does not hold.
//!
//! The initial guess is always zero.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::coupled_iteration::ConfigFault;
use crate::math::Fix128;

mod bicgstab;
mod equilibrate;
mod gmres;
mod precond;

pub use bicgstab::bicgstab;
pub use equilibrate::{solve_equilibrated, BlockEquilibration, BlockScale, KrylovMethod};
pub use gmres::gmres;
pub use precond::{BlockJacobiPreconditioner, IdentityPreconditioner, JacobiPreconditioner};

/// Relative threshold below which a pivot, a projected diagonal or a
/// bi-orthogonality coefficient counts as zero: `2⁻⁴⁰`.
///
/// See the module documentation for why `2⁻⁴⁰` and not the `2⁻⁶⁴` resolution.
pub const BREAKDOWN_RELATIVE: Fix128 = Fix128::from_raw(0, 1 << 24);

/// Smallest relative improvement of the best residual that resets the
/// stagnation window: `1/1024`.
const STAGNATION_MIN_IMPROVEMENT: Fix128 = Fix128::from_raw(0, 1 << 54);

// ============================================================================
// Operators
// ============================================================================

/// A square linear operator `y = A x`, applied matrix-free.
///
/// `apply` must overwrite every entry of `y`; the solvers do not clear it.
pub trait LinearOperator {
    /// Number of rows (and columns).
    fn dim(&self) -> usize;
    /// `y = A x`. Both slices have length [`LinearOperator::dim`].
    fn apply(&self, x: &[Fix128], y: &mut [Fix128]);
}

/// An approximate inverse `z = M⁻¹ r`.
///
/// The solvers apply it on the **right** (`A M⁻¹ y = b`, `x = M⁻¹ y`), so the
/// residual they monitor is the true residual of the original system and the
/// stopping rule means the same thing with and without a preconditioner.
pub trait Preconditioner {
    /// Number of rows (and columns).
    fn dim(&self) -> usize;
    /// `z = M⁻¹ r`. Both slices have length [`Preconditioner::dim`].
    fn apply(&self, r: &[Fix128], z: &mut [Fix128]);
}

/// A dense square matrix, row-major. Intended for tests and for small blocks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseMatrix {
    dim: usize,
    entries: Vec<Fix128>,
}

impl DenseMatrix {
    /// A `dim × dim` matrix from `dim²` row-major entries.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::DimensionMismatch`] when `entries.len() != dim²`.
    pub fn try_new(dim: usize, entries: Vec<Fix128>) -> Result<Self, LinearSolverError> {
        let expected = dim
            .checked_mul(dim)
            .ok_or(LinearSolverError::DimensionMismatch {
                expected: usize::MAX,
                found: entries.len(),
            })?;
        if entries.len() != expected {
            return Err(LinearSolverError::DimensionMismatch {
                expected,
                found: entries.len(),
            });
        }
        Ok(Self { dim, entries })
    }

    /// Entry `(row, col)`.
    ///
    /// # Panics
    ///
    /// When `row` or `col` is not below [`DenseMatrix::dim`] — an index bug in
    /// the caller, like slice indexing.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Fix128 {
        assert!(
            row < self.dim && col < self.dim,
            "DenseMatrix index out of range"
        );
        self.entries[row * self.dim + col]
    }

    /// Number of rows (and columns).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl LinearOperator for DenseMatrix {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply(&self, x: &[Fix128], y: &mut [Fix128]) {
        for (row, out) in y.iter_mut().enumerate().take(self.dim) {
            let start = row * self.dim;
            let mut sum = Fix128::ZERO;
            for (a, &xv) in self.entries[start..start + self.dim].iter().zip(x) {
                sum = sum + *a * xv;
            }
            *out = sum;
        }
    }
}

/// A closure `f(x, y)` writing `y = A x`, wrapped as a [`LinearOperator`].
pub struct FnOperator<F> {
    dim: usize,
    f: F,
}

impl<F: Fn(&[Fix128], &mut [Fix128])> FnOperator<F> {
    /// Wrap `f` as an operator of dimension `dim`.
    pub fn new(dim: usize, f: F) -> Self {
        Self { dim, f }
    }
}

impl<F: Fn(&[Fix128], &mut [Fix128])> LinearOperator for FnOperator<F> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply(&self, x: &[Fix128], y: &mut [Fix128]) {
        (self.f)(x, y);
    }
}

// ============================================================================
// Configuration, results, errors
// ============================================================================

/// Stopping rule and budget shared by [`gmres`] and [`bicgstab`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KrylovConfig {
    max_iterations: u32,
    relative_tolerance: Fix128,
    absolute_tolerance: Fix128,
    restart: usize,
    stagnation_window: u32,
}

impl KrylovConfig {
    /// Default GMRES restart length.
    pub const DEFAULT_RESTART: usize = 30;
    /// Default stagnation window, in iterations.
    pub const DEFAULT_STAGNATION_WINDOW: u32 = 32;

    /// A budget of `max_iterations` operator applications (GMRES) or
    /// iterations (`BiCGStab`), relative tolerance `relative_tolerance`, no
    /// absolute tolerance, restart [`KrylovConfig::DEFAULT_RESTART`] and
    /// stagnation window [`KrylovConfig::DEFAULT_STAGNATION_WINDOW`].
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::InvalidConfig`] when `max_iterations` is zero or
    /// `relative_tolerance` is not positive. Zero tolerance is refused because
    /// truncating arithmetic has no exact solution to reach.
    pub fn try_new(
        max_iterations: u32,
        relative_tolerance: Fix128,
    ) -> Result<Self, LinearSolverError> {
        if max_iterations == 0 {
            return Err(LinearSolverError::InvalidConfig(
                KrylovConfigFault::ZeroIterationBudget,
            ));
        }
        if relative_tolerance <= Fix128::ZERO {
            return Err(LinearSolverError::InvalidConfig(
                KrylovConfigFault::NonPositiveTolerance,
            ));
        }
        Ok(Self {
            max_iterations,
            relative_tolerance,
            absolute_tolerance: Fix128::ZERO,
            restart: Self::DEFAULT_RESTART,
            stagnation_window: Self::DEFAULT_STAGNATION_WINDOW,
        })
    }

    /// Also stop once `‖r‖₂ ≤ absolute_tolerance`.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::InvalidConfig`] when `absolute_tolerance` is negative.
    pub fn with_absolute_tolerance(
        self,
        absolute_tolerance: Fix128,
    ) -> Result<Self, LinearSolverError> {
        if absolute_tolerance.is_negative() {
            return Err(LinearSolverError::InvalidConfig(
                KrylovConfigFault::NegativeAbsoluteTolerance,
            ));
        }
        Ok(Self {
            absolute_tolerance,
            ..self
        })
    }

    /// GMRES restart length `m` (Krylov vectors kept per cycle). Ignored by
    /// [`bicgstab`].
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::InvalidConfig`] when `restart` is zero.
    pub fn with_restart(self, restart: usize) -> Result<Self, LinearSolverError> {
        if restart == 0 {
            return Err(LinearSolverError::InvalidConfig(
                KrylovConfigFault::ZeroRestart,
            ));
        }
        Ok(Self { restart, ..self })
    }

    /// Iterations without a `1/1024` improvement of the best residual after
    /// which the solve is declared [`LinearSolverError::Stagnated`].
    ///
    /// GMRES never uses a window shorter than one cycle plus one (`min(m, n) +
    /// 1`): within a cycle its residual may plateau for that long and still
    /// terminate, so a shorter plateau is no evidence of stagnation.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::InvalidConfig`] when `window` is zero.
    pub fn with_stagnation_window(self, window: u32) -> Result<Self, LinearSolverError> {
        if window == 0 {
            return Err(LinearSolverError::InvalidConfig(
                KrylovConfigFault::ZeroStagnationWindow,
            ));
        }
        Ok(Self {
            stagnation_window: window,
            ..self
        })
    }
}

/// Why a [`KrylovConfig`] was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KrylovConfigFault {
    /// `max_iterations` was zero.
    ZeroIterationBudget,
    /// `relative_tolerance` was zero or negative.
    NonPositiveTolerance,
    /// `absolute_tolerance` was negative.
    NegativeAbsoluteTolerance,
    /// The GMRES restart length was zero.
    ZeroRestart,
    /// The stagnation window was zero.
    ZeroStagnationWindow,
}

/// Which algebraic step could not be taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BreakdownKind {
    /// GMRES: the projected (Hessenberg) system became singular — the new
    /// operator image lies in the span already built, but the residual is not
    /// zero. A zero operator, or a singular one with `b` outside its range.
    SingularHessenberg,
    /// `BiCGStab`: `(r̂, r)` vanished relative to `‖r̂‖‖r‖`.
    ShadowResidualOrthogonal,
    /// `BiCGStab`: `(r̂, A p)` vanished relative to `‖r̂‖‖A p‖`, or `A p`
    /// itself vanished relative to the gain the operator has already shown
    /// (`p` in the null space up to rounding).
    DirectionOrthogonal,
    /// `BiCGStab`: the stabilisation step could not reduce the residual
    /// (`(t, s)` vanished relative to `‖t‖‖s‖`, or `t = A s` vanished
    /// relative to the operator's observed gain).
    StabilizerVanished,
}

/// Why a solve did not return a solution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearSolverError {
    /// Two lengths that must agree did not (operator, right-hand side,
    /// preconditioner, block layout, dense entries).
    DimensionMismatch {
        /// The length required.
        expected: usize,
        /// The length supplied.
        found: usize,
    },
    /// A configuration value was out of range.
    InvalidConfig(KrylovConfigFault),
    /// The iteration budget ran out.
    NotConverged {
        /// Iterations performed.
        iterations: u32,
        /// `‖r‖ / ‖b‖` of the last true residual.
        relative_residual: Fix128,
    },
    /// The best residual stopped improving.
    Stagnated {
        /// Iterations performed.
        iterations: u32,
        /// `‖r‖ / ‖b‖` of the best residual seen.
        relative_residual: Fix128,
        /// Iterations since the best residual last improved by `1/1024`.
        without_improvement: u32,
    },
    /// The method could not take its next step.
    Breakdown {
        /// Iterations performed.
        iterations: u32,
        /// Which step failed.
        kind: BreakdownKind,
        /// `‖r‖ / ‖b‖` of the residual at the breakdown.
        relative_residual: Fix128,
    },
    /// A rescaled norm or inner product left the representable range.
    ArithmeticOverflow,
    /// A preconditioner could not be built: a zero diagonal entry (Jacobi,
    /// `index` = row) or a singular diagonal block (block Jacobi, `index` =
    /// block).
    SingularPreconditioner {
        /// Row or block index.
        index: usize,
    },
    /// After equilibration, a block's magnitude relative to the largest block
    /// is below [`L2_TERM_FLOOR`](crate::coupled_iteration::L2_TERM_FLOOR), so its products in an inner product would
    /// truncate to zero.
    BlockBelowProductFloor {
        /// The block.
        block: usize,
        /// Its magnitude divided by the largest block's.
        relative_magnitude: Fix128,
    },
    /// An equilibration factor was out of range (see [`EquilibrationScale`](crate::coupled_iteration::EquilibrationScale)).
    Equilibration(ConfigFault),
    /// The block layout was empty, contained an empty block, or a magnitude
    /// was not positive.
    InvalidBlockLayout,
}

impl core::fmt::Display for LinearSolverError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Convergence record of a successful solve.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KrylovStats {
    /// Iterations performed (GMRES: Arnoldi steps; `BiCGStab`: full steps).
    pub iterations: u32,
    /// GMRES restarts performed (`BiCGStab`: restarts after a true-residual
    /// refresh).
    pub restarts: u32,
    /// `‖r_k‖₂` for `k = 0..=iterations`. Entry 0 is `‖b‖₂`. GMRES records its
    /// Givens estimate inside a cycle and replaces the last entry of each cycle
    /// with the recomputed true residual.
    pub residual_history: Vec<Fix128>,
    /// `‖b − A x‖₂` of the returned solution.
    pub residual_norm: Fix128,
    /// `‖b‖₂`.
    pub rhs_norm: Fix128,
}

/// Solution and convergence record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KrylovSolution {
    /// The solution vector.
    pub x: Vec<Fix128>,
    /// How it was reached.
    pub stats: KrylovStats,
}

// ============================================================================
// Rescaled vector arithmetic (crate-internal)
// ============================================================================

fn to_raw(x: Fix128) -> i128 {
    (i128::from(x.hi) << 64) | i128::from(x.lo)
}

fn from_raw_i128(raw: i128) -> Fix128 {
    Fix128::from_raw((raw >> 64) as i64, raw as u64)
}

/// `e` such that `|x| < 2^e` and `|x| ≥ 2^(e−1)`; `None` for zero.
fn pow2_exponent(x: Fix128) -> Option<i32> {
    let raw = to_raw(x);
    if raw == 0 {
        return None;
    }
    let magnitude = raw.unsigned_abs();
    let bits = 128 - magnitude.leading_zeros() as i32;
    Some(bits - 64)
}

/// `x · 2^k`: exact for `k ≥ 0` when it fits, floor for `k < 0`.
fn shift(x: Fix128, k: i32) -> Result<Fix128, LinearSolverError> {
    let raw = to_raw(x);
    if k >= 0 {
        if raw == 0 {
            return Ok(x);
        }
        let k = k.unsigned_abs();
        // Headroom: the sign bit plus the magnitude must stay inside i128.
        if k >= 127 || raw.unsigned_abs().leading_zeros() <= k {
            return Err(LinearSolverError::ArithmeticOverflow);
        }
        Ok(from_raw_i128(raw << k))
    } else {
        let k = k.unsigned_abs();
        if k >= 127 {
            return Ok(from_raw_i128(raw >> 127));
        }
        Ok(from_raw_i128(raw >> k))
    }
}

fn max_abs(v: &[Fix128]) -> Fix128 {
    let mut worst = Fix128::ZERO;
    for &x in v {
        let a = x.abs();
        if a > worst {
            worst = a;
        }
    }
    worst
}

/// `mantissa · 2^exponent`: an inner product or norm kept in the scaled form
/// it was accumulated in.
///
/// A quantity quadratic in a small vector (`(t, t)` for `‖t‖ = 1e-10` is
/// `1e-20`) is below the `2⁻⁶⁴` resolution as an absolute [`Fix128`], but its
/// mantissa here is O(1). Ratios of such quantities (`BiCGStab`'s `α`, `ω`,
/// `ρ/ρ₋₁`) and the breakdown tests are taken between mantissas.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    mantissa: Fix128,
    exponent: i32,
}

impl Scaled {
    const ZERO: Self = Self {
        mantissa: Fix128::ZERO,
        exponent: 0,
    };

    /// The absolute value (may floor to the resolution).
    fn value(self) -> Result<Fix128, LinearSolverError> {
        shift(self.mantissa, self.exponent)
    }

    /// `self / other`, from the mantissas. `other` must be non-zero.
    fn ratio(self, other: Self) -> Result<Fix128, LinearSolverError> {
        shift(
            self.mantissa / other.mantissa,
            self.exponent - other.exponent,
        )
    }
}

/// `aᵀb`, each operand shifted so its largest entry is in `[1/2, 1)` first.
/// The exponent is `e(a) + e(b)`, the same convention as [`norm_scaled`].
fn dot_scaled(a: &[Fix128], b: &[Fix128]) -> Result<Scaled, LinearSolverError> {
    let (Some(ea), Some(eb)) = (pow2_exponent(max_abs(a)), pow2_exponent(max_abs(b))) else {
        return Ok(Scaled::ZERO);
    };
    let mut sum = Fix128::ZERO;
    for (&x, &y) in a.iter().zip(b) {
        sum = sum + shift(x, -ea)? * shift(y, -eb)?;
    }
    Ok(Scaled {
        mantissa: sum,
        exponent: ea + eb,
    })
}

/// `‖v‖₂` with exponent `e(v)`.
fn norm_scaled(v: &[Fix128]) -> Result<Scaled, LinearSolverError> {
    let Some(e) = pow2_exponent(max_abs(v)) else {
        return Ok(Scaled::ZERO);
    };
    let mut sum = Fix128::ZERO;
    for &x in v {
        let s = shift(x, -e)?;
        sum = sum + s * s;
    }
    Ok(Scaled {
        mantissa: sum.sqrt(),
        exponent: e,
    })
}

/// `aᵀb` as an absolute value.
fn dot(a: &[Fix128], b: &[Fix128]) -> Result<Fix128, LinearSolverError> {
    dot_scaled(a, b)?.value()
}

/// `‖v‖₂` as an absolute value.
fn norm(v: &[Fix128]) -> Result<Fix128, LinearSolverError> {
    norm_scaled(v)?.value()
}

/// `true` when `|(a, b)| ≤ BREAKDOWN_RELATIVE · ‖a‖ ‖b‖`, i.e. the cosine of
/// the angle between `a` and `b` vanishes. Compared between mantissas, whose
/// exponents agree by construction, so it does not underflow for small
/// vectors.
fn cosine_vanishes(inner: Scaled, norm_a: Scaled, norm_b: Scaled) -> bool {
    if norm_a.mantissa.is_zero() || norm_b.mantissa.is_zero() {
        return true;
    }
    inner.mantissa.abs() <= BREAKDOWN_RELATIVE * norm_a.mantissa * norm_b.mantissa
}

/// `√(a² + b²)`, rescaled.
fn hypot(a: Fix128, b: Fix128) -> Result<Fix128, LinearSolverError> {
    norm(&[a, b])
}

/// `r = b − A x`.
fn residual<A: LinearOperator + ?Sized>(op: &A, x: &[Fix128], b: &[Fix128], r: &mut [Fix128]) {
    op.apply(x, r);
    for (ri, &bi) in r.iter_mut().zip(b) {
        *ri = bi - *ri;
    }
}

fn relative(residual_norm: Fix128, rhs_norm: Fix128) -> Fix128 {
    if rhs_norm.is_zero() {
        Fix128::ZERO
    } else {
        residual_norm / rhs_norm
    }
}

/// Common entry checks. Returns `‖b‖` and the target, or the trivial solution.
enum Start {
    Trivial(KrylovSolution),
    Run { rhs_norm: Fix128, target: Fix128 },
}

fn start<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
    op: &A,
    precond: &M,
    b: &[Fix128],
    config: &KrylovConfig,
) -> Result<Start, LinearSolverError> {
    let n = op.dim();
    if b.len() != n {
        return Err(LinearSolverError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }
    if precond.dim() != n {
        return Err(LinearSolverError::DimensionMismatch {
            expected: n,
            found: precond.dim(),
        });
    }
    let rhs_norm = norm(b)?;
    if rhs_norm.is_zero() {
        // `x = 0` solves `A x = 0` exactly, whatever `A` is.
        return Ok(Start::Trivial(KrylovSolution {
            x: vec![Fix128::ZERO; n],
            stats: KrylovStats {
                iterations: 0,
                restarts: 0,
                residual_history: vec![Fix128::ZERO],
                residual_norm: Fix128::ZERO,
                rhs_norm,
            },
        }));
    }
    let requested = config.relative_tolerance * rhs_norm;
    let target = if requested > config.absolute_tolerance {
        requested
    } else {
        config.absolute_tolerance
    };
    Ok(Start::Run { rhs_norm, target })
}

/// Best-residual stagnation bookkeeping.
struct Stagnation {
    best: Fix128,
    since: u32,
    window: u32,
}

impl Stagnation {
    fn new(initial: Fix128, window: u32) -> Self {
        Self {
            best: initial,
            since: 0,
            window,
        }
    }

    /// Record a residual; `true` when the window is exhausted.
    fn record(&mut self, residual: Fix128) -> bool {
        if residual < self.best - self.best * STAGNATION_MIN_IMPROVEMENT {
            self.best = residual;
            self.since = 0;
        } else {
            if residual < self.best {
                self.best = residual;
            }
            self.since += 1;
        }
        self.since >= self.window
    }
}

/// The largest gain `‖A M⁻¹ q‖ / ‖q‖` seen so far: a lower bound on the
/// operator's norm, built from products the solver forms anyway.
///
/// It exists to tell a genuinely small image from rounding noise. When `q`
/// lies (up to rounding) in the null space, `A M⁻¹ q` is a vector of
/// `2⁻⁶⁴`-level entries pointing in an arbitrary direction; its cosine with
/// another vector is then *not* small, so a cosine test alone passes it and
/// the next coefficient (`α = ρ / (r̂, v)`) is of order `2⁶⁴`. Comparing the
/// image against the gain the operator has already shown catches it.
struct OperatorGain {
    gain: Fix128,
}

impl OperatorGain {
    const fn new() -> Self {
        Self { gain: Fix128::ZERO }
    }

    /// Record `‖image‖ / ‖input‖`, then report whether `image` is negligible:
    /// `‖image‖ ≤ BREAKDOWN_RELATIVE · gain · ‖input‖`. A zero input or a zero
    /// gain (the operator has never produced anything) counts as negligible.
    fn observe(&mut self, image: Scaled, input: Scaled) -> Result<bool, LinearSolverError> {
        if input.mantissa.is_zero() {
            return Ok(true);
        }
        let ratio = image.ratio(input)?;
        if ratio > self.gain {
            self.gain = ratio;
        }
        Ok(self.gain.is_zero() || ratio <= BREAKDOWN_RELATIVE * self.gain)
    }
}

/// `true` when `|value| ≤ BREAKDOWN_RELATIVE · scale` (a zero scale counts as
/// vanishing). Used by GMRES, where both are O(‖A‖).
fn negligible(value: Fix128, scale: Fix128) -> bool {
    if scale.is_zero() {
        return true;
    }
    value.abs() / scale <= BREAKDOWN_RELATIVE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_round_trips_exactly_upwards() {
        let x = Fix128::from_ratio(-3, 7);
        let up = shift(x, 20).expect("fits");
        assert_eq!(shift(up, -20).expect("down"), x);
    }

    #[test]
    fn shift_reports_overflow_instead_of_wrapping() {
        assert_eq!(
            shift(Fix128::from_int(1 << 40), 30),
            Err(LinearSolverError::ArithmeticOverflow)
        );
    }

    #[test]
    fn pow2_exponent_brackets_the_magnitude() {
        assert_eq!(pow2_exponent(Fix128::ONE), Some(1));
        assert_eq!(pow2_exponent(Fix128::from_ratio(3, 4)), Some(0));
        assert_eq!(pow2_exponent(Fix128::from_int(-5)), Some(3));
        assert_eq!(pow2_exponent(Fix128::ZERO), None);
    }

    #[test]
    fn rescaled_norm_keeps_tiny_vectors() {
        // Entries of 2⁻⁴⁰ square to 2⁻⁸⁰, below the representation: an
        // unscaled sum of squares would report zero.
        let tiny = Fix128::from_raw(0, 1 << 24);
        let v = [tiny, tiny, tiny, tiny];
        assert_eq!(norm(&v).expect("norm"), tiny.double());
        let naive = v.iter().fold(Fix128::ZERO, |s, &x| s + x * x);
        assert_eq!(naive, Fix128::ZERO);
    }
}
