//! Data parallelism over fixed slabs, with results independent of the thread count.
//!
//! # MODEL
//!
//! - **What it does.** [`fill`] writes every element of an output slice from a function of the
//!   element's index; [`sum`] adds a function of every index. Under the `parallel` feature the
//!   index range is cut into slabs of [`SLAB_LEN`] consecutive elements and the slabs run on the
//!   rayon pool; without it the same loop runs in index order.
//! - **Why the bits do not depend on the thread count.** (1) The slab length is a constant, never
//!   derived from the number of threads, so the cut is the same for every pool. (2) [`fill`] is a
//!   gather: each output element is computed from inputs no thread writes during the call, so the
//!   order the elements are visited in cannot change any of them. (3) [`sum`] adds each slab's
//!   terms in index order and then the slab partials in slab order, and it relies on `Fix128`
//!   addition being wrapping integer addition (a group operation mod 2¹²⁸,
//!   `tests/reduction_order_independence.rs`): it is associative and commutative, so any grouping
//!   of the same terms gives the same bits, and the fixed order is a convention rather than a
//!   requirement. No partial is rounded, divided or normalised before it is combined — that is
//!   the one way a partition could enter the bits through a reduction.
//! - **What it does not cover.** A kernel that updates its own input in place (Gauss-Seidel,
//!   in-place advection) is not a gather and must not be written with [`fill`] over the array it
//!   reads; a red-black sweep is a gather only per colour, from a snapshot of the other colour.
//! - **Verification.** `tests/cfd_parallel_determinism.rs`: the serial build and pools of 1, 2, 4
//!   and 8 threads give the same state digests (one constant, asserted by both builds).

use crate::math::Fix128;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Elements per slab: the parallel grain, fixed so the partition never depends on the pool.
pub(crate) const SLAB_LEN: usize = 4096;

/// `out[n] = f(n)` for every `n`, in fixed slabs of [`SLAB_LEN`] (see the module MODEL).
pub(crate) fn fill<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(SLAB_LEN)
            .enumerate()
            .for_each(|(s, slab)| {
                let base = s * SLAB_LEN;
                for (o, x) in slab.iter_mut().enumerate() {
                    *x = f(base + o);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (n, x) in out.iter_mut().enumerate() {
        *x = f(n);
    }
}

/// `out[n] = f(n, out[n])` for every `n`, in fixed slabs of [`SLAB_LEN`]: an elementwise update
/// that reads only its own element of `out` (and inputs no thread writes).
pub(crate) fn update<T, F>(out: &mut [T], f: F)
where
    T: Send + Copy,
    F: Fn(usize, T) -> T + Sync,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(SLAB_LEN)
            .enumerate()
            .for_each(|(s, slab)| {
                let base = s * SLAB_LEN;
                for (o, x) in slab.iter_mut().enumerate() {
                    *x = f(base + o, *x);
                }
            });
    }
    #[cfg(not(feature = "parallel"))]
    for (n, x) in out.iter_mut().enumerate() {
        *x = f(n, *x);
    }
}

/// `Σ f(n)` for `n` in `0..len`: each slab's terms in index order, then the slab partials in slab
/// order (see the module MODEL for why the grouping cannot change the bits).
pub(crate) fn sum<F>(len: usize, f: F) -> Fix128
where
    F: Fn(usize) -> Fix128 + Sync,
{
    let slab_sum = |s: usize| {
        let mut acc = Fix128::ZERO;
        for n in s * SLAB_LEN..len.min((s + 1) * SLAB_LEN) {
            acc = acc + f(n);
        }
        acc
    };
    let slabs = len.div_ceil(SLAB_LEN);
    #[cfg(feature = "parallel")]
    let partials: Vec<Fix128> = {
        use rayon::prelude::*;
        (0..slabs).into_par_iter().map(slab_sum).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let partials: Vec<Fix128> = (0..slabs).map(slab_sum).collect();
    let mut acc = Fix128::ZERO;
    for p in partials {
        acc = acc + p;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_writes_every_index_across_slab_edges() {
        let mut out = vec![0usize; 2 * SLAB_LEN + 7];
        fill(&mut out, |n| n * 3);
        assert!(out.iter().enumerate().all(|(n, &x)| x == n * 3));
    }

    #[test]
    fn sum_matches_the_index_order_sum_across_slab_edges() {
        let len = 3 * SLAB_LEN + 11;
        let term = |n: usize| Fix128::from_ratio(n as i64 * 7 + 1, 3);
        let mut want = Fix128::ZERO;
        for n in 0..len {
            want = want + term(n);
        }
        assert_eq!(sum(len, term), want);
        assert_eq!(sum(0, term), Fix128::ZERO);
    }
}
