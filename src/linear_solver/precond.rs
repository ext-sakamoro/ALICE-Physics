//! Identity, Jacobi and block Jacobi preconditioners.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{DenseMatrix, LinearSolverError, Preconditioner, BREAKDOWN_RELATIVE};
use crate::math::Fix128;

/// No preconditioning: `z = r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentityPreconditioner {
    dim: usize,
}

impl IdentityPreconditioner {
    /// The identity of dimension `dim`.
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply(&self, r: &[Fix128], z: &mut [Fix128]) {
        z.copy_from_slice(r);
    }
}

/// Diagonal (Jacobi) preconditioner `z_i = r_i / a_ii`.
///
/// Applied by division, not by a stored reciprocal: a reciprocal `1/a_ii` of a
/// large diagonal is a small number that `Fix128` holds to fewer significant
/// bits than `r_i / a_ii` itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JacobiPreconditioner {
    diagonal: Vec<Fix128>,
}

impl JacobiPreconditioner {
    /// From the operator's diagonal.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::SingularPreconditioner`] with the row of the first
    /// zero entry.
    pub fn from_diagonal(diagonal: Vec<Fix128>) -> Result<Self, LinearSolverError> {
        if let Some(index) = diagonal.iter().position(|d| d.is_zero()) {
            return Err(LinearSolverError::SingularPreconditioner { index });
        }
        Ok(Self { diagonal })
    }

    /// From the diagonal of a dense matrix.
    ///
    /// # Errors
    ///
    /// As [`JacobiPreconditioner::from_diagonal`].
    pub fn from_dense(matrix: &DenseMatrix) -> Result<Self, LinearSolverError> {
        Self::from_diagonal((0..matrix.dim()).map(|i| matrix.get(i, i)).collect())
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn dim(&self) -> usize {
        self.diagonal.len()
    }

    fn apply(&self, r: &[Fix128], z: &mut [Fix128]) {
        for ((zi, &ri), &d) in z.iter_mut().zip(r).zip(&self.diagonal) {
            *zi = ri / d;
        }
    }
}

/// One diagonal block, factored `P A = L U` with partial pivoting.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LuBlock {
    offset: usize,
    size: usize,
    /// `L` (unit diagonal, below) and `U` (on and above), row-major.
    lu: Vec<Fix128>,
    /// Row permutation: row `i` of `P A` is row `perm[i]` of `A`.
    perm: Vec<usize>,
}

impl LuBlock {
    fn factor(offset: usize, size: usize, mut lu: Vec<Fix128>) -> Option<Self> {
        let scale = lu
            .iter()
            .fold(Fix128::ZERO, |m, v| if v.abs() > m { v.abs() } else { m });
        let mut perm: Vec<usize> = (0..size).collect();
        for k in 0..size {
            let mut pivot_row = k;
            let mut pivot = lu[k * size + k].abs();
            for i in k + 1..size {
                let candidate = lu[i * size + k].abs();
                if candidate > pivot {
                    pivot = candidate;
                    pivot_row = i;
                }
            }
            // Relative test: a pivot at the rounding level of the block's own
            // entries is a singular block, not a small one.
            if scale.is_zero() || pivot / scale <= BREAKDOWN_RELATIVE {
                return None;
            }
            if pivot_row != k {
                for c in 0..size {
                    lu.swap(k * size + c, pivot_row * size + c);
                }
                perm.swap(k, pivot_row);
            }
            let diag = lu[k * size + k];
            for i in k + 1..size {
                let factor = lu[i * size + k] / diag;
                lu[i * size + k] = factor;
                for c in k + 1..size {
                    lu[i * size + c] = lu[i * size + c] - factor * lu[k * size + c];
                }
            }
        }
        Some(Self {
            offset,
            size,
            lu,
            perm,
        })
    }

    fn solve(&self, r: &[Fix128], z: &mut [Fix128]) {
        let n = self.size;
        let rb = &r[self.offset..self.offset + n];
        let zb = &mut z[self.offset..self.offset + n];
        for i in 0..n {
            let mut s = rb[self.perm[i]];
            for (&l, &zk) in self.lu[i * n..i * n + i].iter().zip(zb.iter()) {
                s = s - l * zk;
            }
            zb[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = zb[i];
            for (&u, &zk) in self.lu[i * n + i + 1..(i + 1) * n].iter().zip(&zb[i + 1..]) {
                s = s - u * zk;
            }
            zb[i] = s / self.lu[i * n + i];
        }
    }
}

/// Block Jacobi: the inverse of each diagonal block, by dense LU with partial
/// pivoting. On a block-diagonal operator it is the exact inverse, so a Krylov
/// method converges in one iteration up to rounding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockJacobiPreconditioner {
    dim: usize,
    blocks: Vec<LuBlock>,
}

impl BlockJacobiPreconditioner {
    /// Factor the diagonal blocks of `matrix`, of sizes `block_sizes` in order.
    ///
    /// # Errors
    ///
    /// [`LinearSolverError::DimensionMismatch`] when the sizes do not add up to
    /// `matrix.dim()`, [`LinearSolverError::InvalidBlockLayout`] for an empty
    /// block, and [`LinearSolverError::SingularPreconditioner`] (with the block
    /// index) when a pivot falls below [`BREAKDOWN_RELATIVE`] of that block's
    /// largest entry.
    pub fn from_dense(
        matrix: &DenseMatrix,
        block_sizes: &[usize],
    ) -> Result<Self, LinearSolverError> {
        let total: usize = block_sizes.iter().sum();
        if total != matrix.dim() {
            return Err(LinearSolverError::DimensionMismatch {
                expected: matrix.dim(),
                found: total,
            });
        }
        let mut blocks = Vec::with_capacity(block_sizes.len());
        let mut offset = 0;
        for (index, &size) in block_sizes.iter().enumerate() {
            if size == 0 {
                return Err(LinearSolverError::InvalidBlockLayout);
            }
            let mut entries = Vec::with_capacity(size * size);
            for i in 0..size {
                for j in 0..size {
                    entries.push(matrix.get(offset + i, offset + j));
                }
            }
            let block = LuBlock::factor(offset, size, entries)
                .ok_or(LinearSolverError::SingularPreconditioner { index })?;
            blocks.push(block);
            offset += size;
        }
        Ok(Self {
            dim: matrix.dim(),
            blocks,
        })
    }
}

impl Preconditioner for BlockJacobiPreconditioner {
    fn dim(&self) -> usize {
        self.dim
    }

    fn apply(&self, r: &[Fix128], z: &mut [Fix128]) {
        for block in &self.blocks {
            block.solve(r, z);
        }
    }
}
