//! Block equilibration: the only entry for a block-coupled system.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{
    bicgstab, gmres, shift, KrylovConfig, KrylovSolution, LinearOperator, LinearSolverError,
    Preconditioner,
};
use crate::coupled_iteration::{ConfigFault, EquilibrationScale, L2_TERM_FLOOR};
use crate::math::Fix128;

/// The power-of-two factor `dᵢ` applied to one block, on both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockScale {
    /// `dᵢ = 1 / factor`: the block was large.
    Down(EquilibrationScale),
    /// `dᵢ = factor`: the block was small.
    Up(EquilibrationScale),
}

impl BlockScale {
    fn apply(self, value: Fix128) -> Fix128 {
        match self {
            Self::Down(s) => s.scale_down(value),
            Self::Up(s) => s.scale_up(value),
        }
    }

    /// `dᵢ² · m`, by exact shifts.
    fn apply_squared(self, value: Fix128) -> Result<Fix128, LinearSolverError> {
        let e = i32::try_from(match self {
            Self::Down(s) | Self::Up(s) => s.exponent(),
        })
        .unwrap_or(i32::MAX);
        match self {
            Self::Down(_) => shift(value, -2 * e),
            Self::Up(_) => shift(value, 2 * e),
        }
    }
}

/// Which Krylov method [`solve_equilibrated`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KrylovMethod {
    /// [`super::gmres`].
    Gmres,
    /// [`super::bicgstab`].
    BiCgStab,
}

/// Per-block symmetric scaling `D = diag(dᵢ I)` of a block system, checked
/// against the product floor.
///
/// See the module documentation of [`crate::linear_solver`] for why a coupled
/// system must go through this type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockEquilibration {
    offsets: Vec<usize>,
    scales: Vec<BlockScale>,
}

impl BlockEquilibration {
    /// Equilibrate blocks of sizes `block_sizes` whose diagonal operator blocks
    /// have characteristic magnitudes `diagonal_magnitudes` (e.g. `E·h` for a
    /// displacement block and `k·h` for a temperature block).
    ///
    /// Each factor is the power of two with `dᵢ² · mᵢ ∈ (1/4, 4)`. Only the
    /// diagonal blocks are measured; off-diagonal coupling blocks are scaled by
    /// `dᵢ dⱼ` as a consequence and are not checked.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::InvalidBlockLayout`] for no blocks, an empty block,
    /// a length mismatch or a non-positive magnitude;
    /// [`LinearSolverError::Equilibration`] when a factor would exceed
    /// `2^`[`EquilibrationScale::MAX_EXPONENT`]; and
    /// [`LinearSolverError::BlockBelowProductFloor`] if a block is still below
    /// the floor afterwards (it cannot be, unless a factor was clamped).
    pub fn try_new(
        block_sizes: &[usize],
        diagonal_magnitudes: &[Fix128],
    ) -> Result<Self, LinearSolverError> {
        let offsets = layout(block_sizes, diagonal_magnitudes)?;
        let mut scales = Vec::with_capacity(block_sizes.len());
        for &m in diagonal_magnitudes {
            scales.push(scale_for(m)?);
        }
        let equilibration = Self { offsets, scales };
        equilibration.check_floor(diagonal_magnitudes)?;
        Ok(equilibration)
    }

    /// **No scaling** (every `dᵢ = 1`), but the same floor check as
    /// [`BlockEquilibration::try_new`]. The explicitly named way to run a
    /// block system unscaled; it refuses one whose blocks would vanish.
    ///
    /// # Errors
    ///
    /// As [`BlockEquilibration::try_new`]; in particular
    /// [`LinearSolverError::BlockBelowProductFloor`] for a PLA thermo-elastic
    /// pair (block ratio `2.69e10 > 2³²`).
    pub fn unscaled(
        block_sizes: &[usize],
        diagonal_magnitudes: &[Fix128],
    ) -> Result<Self, LinearSolverError> {
        let offsets = layout(block_sizes, diagonal_magnitudes)?;
        let equilibration = Self {
            offsets,
            scales: vec![BlockScale::Down(EquilibrationScale::IDENTITY); block_sizes.len()],
        };
        equilibration.check_floor(diagonal_magnitudes)?;
        Ok(equilibration)
    }

    /// Total dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// Number of blocks.
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.scales.len()
    }

    /// The factor of block `block`, or `None` past the last block.
    #[must_use]
    pub fn scale(&self, block: usize) -> Option<BlockScale> {
        self.scales.get(block).copied()
    }

    /// `dᵢ² aᵢᵢ` for a diagonal `a` of the original operator: the diagonal of
    /// the equilibrated operator, for building a
    /// [`super::JacobiPreconditioner`] that matches it.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::DimensionMismatch`] when `diagonal` has the wrong
    /// length, [`LinearSolverError::ArithmeticOverflow`] when a scaled entry is
    /// out of range.
    pub fn equilibrate_diagonal(
        &self,
        diagonal: &[Fix128],
    ) -> Result<Vec<Fix128>, LinearSolverError> {
        if diagonal.len() != self.dim() {
            return Err(LinearSolverError::DimensionMismatch {
                expected: self.dim(),
                found: diagonal.len(),
            });
        }
        let mut out = Vec::with_capacity(diagonal.len());
        for (block, &scale) in self.scales.iter().enumerate() {
            for &a in &diagonal[self.offsets[block]..self.offsets[block + 1]] {
                out.push(scale.apply_squared(a)?);
            }
        }
        Ok(out)
    }

    /// `v ← D v`.
    fn apply(&self, v: &mut [Fix128]) {
        for (block, &scale) in self.scales.iter().enumerate() {
            for value in &mut v[self.offsets[block]..self.offsets[block + 1]] {
                *value = scale.apply(*value);
            }
        }
    }

    fn check_floor(&self, magnitudes: &[Fix128]) -> Result<(), LinearSolverError> {
        let mut scaled = Vec::with_capacity(magnitudes.len());
        for (&m, &scale) in magnitudes.iter().zip(&self.scales) {
            scaled.push(scale.apply_squared(m)?);
        }
        let largest = scaled
            .iter()
            .copied()
            .fold(Fix128::ZERO, |a, b| if b > a { b } else { a });
        for (block, &m) in scaled.iter().enumerate() {
            let relative_magnitude = m / largest;
            if relative_magnitude < L2_TERM_FLOOR {
                return Err(LinearSolverError::BlockBelowProductFloor {
                    block,
                    relative_magnitude,
                });
            }
        }
        Ok(())
    }
}

fn layout(block_sizes: &[usize], magnitudes: &[Fix128]) -> Result<Vec<usize>, LinearSolverError> {
    if block_sizes.is_empty()
        || block_sizes.len() != magnitudes.len()
        || block_sizes.contains(&0)
        || magnitudes.iter().any(|m| *m <= Fix128::ZERO)
    {
        return Err(LinearSolverError::InvalidBlockLayout);
    }
    let mut offsets = Vec::with_capacity(block_sizes.len() + 1);
    let mut offset = 0usize;
    offsets.push(0);
    for &size in block_sizes {
        offset = offset
            .checked_add(size)
            .ok_or(LinearSolverError::InvalidBlockLayout)?;
        offsets.push(offset);
    }
    Ok(offsets)
}

/// The power of two `d` with `d² m ∈ (1/4, 4)`.
fn scale_for(m: Fix128) -> Result<BlockScale, LinearSolverError> {
    if m >= Fix128::ONE {
        let s = EquilibrationScale::covering(m.sqrt()).map_err(LinearSolverError::Equilibration)?;
        Ok(BlockScale::Down(s))
    } else {
        // `1/m` must itself be representable before its root is taken.
        let limit = Fix128::from_raw(0, 1 << 2);
        if m < limit {
            return Err(LinearSolverError::Equilibration(
                ConfigFault::ScaleOutOfRange,
            ));
        }
        let s = EquilibrationScale::covering((Fix128::ONE / m).sqrt())
            .map_err(LinearSolverError::Equilibration)?;
        Ok(BlockScale::Up(s))
    }
}

/// `D A D` as an operator.
struct Equilibrated<'a, A: ?Sized> {
    op: &'a A,
    equilibration: &'a BlockEquilibration,
}

impl<A: LinearOperator + ?Sized> LinearOperator for Equilibrated<'_, A> {
    fn dim(&self) -> usize {
        self.op.dim()
    }

    fn apply(&self, x: &[Fix128], y: &mut [Fix128]) {
        let mut scaled = x.to_vec();
        self.equilibration.apply(&mut scaled);
        self.op.apply(&scaled, y);
        self.equilibration.apply(y);
    }
}

/// Solve the block system `A x = b` as `(D A D) x̃ = D b`, `x = D x̃`.
///
/// `precond` acts on the **equilibrated** system (build a Jacobi one from
/// [`BlockEquilibration::equilibrate_diagonal`]). The returned statistics —
/// residual history and norms — are those of the equilibrated system, where
/// every block counts on the same footing; that is the point of solving it.
///
/// # Errors
///
/// [`LinearSolverError::DimensionMismatch`] when `op`, `b` and the
/// equilibration disagree; otherwise as [`super::gmres`] / [`super::bicgstab`].
pub fn solve_equilibrated<A: LinearOperator + ?Sized, M: Preconditioner + ?Sized>(
    op: &A,
    equilibration: &BlockEquilibration,
    precond: &M,
    b: &[Fix128],
    method: KrylovMethod,
    config: &KrylovConfig,
) -> Result<KrylovSolution, LinearSolverError> {
    let n = op.dim();
    if equilibration.dim() != n {
        return Err(LinearSolverError::DimensionMismatch {
            expected: n,
            found: equilibration.dim(),
        });
    }
    if b.len() != n {
        return Err(LinearSolverError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }
    let scaled_op = Equilibrated { op, equilibration };
    let mut scaled_b = b.to_vec();
    equilibration.apply(&mut scaled_b);
    let mut solution = match method {
        KrylovMethod::Gmres => gmres(&scaled_op, precond, &scaled_b, config)?,
        KrylovMethod::BiCgStab => bicgstab(&scaled_op, precond, &scaled_b, config)?,
    };
    equilibration.apply(&mut solution.x);
    Ok(solution)
}
