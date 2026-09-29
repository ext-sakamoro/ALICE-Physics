//! Oracle: which `Fix128` reductions are order-independent, and which are not.
//!
//! This is the gate for the HPC-parallel question. The obstacle usually stated
//! for distributed solvers is that a reduction's result depends on the order the
//! partial sums arrive in, because floating-point addition is not associative.
//! For `Fix128` that argument has to be **measured rather than assumed**: the
//! raw representation is a two's-complement 128-bit integer and `Add` is
//! `overflowing_add` + `wrapping_add` with no saturation and no rounding, so
//! addition is exactly the group operation of `Z/2¹²⁸` — commutative and
//! associative even when it overflows.
//!
//! What this file pins:
//!
//! | reduction | order-independent? |
//! |---|---|
//! | `Σ aᵢ` over any permutation | yes, bit-exact |
//! | `Σ aᵢ` sequential vs pairwise tree | yes, bit-exact |
//! | `Σ aᵢ·bᵢ` (dot product) over any permutation or tree | yes, bit-exact |
//! | `Σ aᵢ` with deliberate overflow | yes, bit-exact (wrapping is associative) |
//! | `(a·b)·c` vs `a·(b·c)` | **no** — the product truncates |
//! | `sqrt` / `Div` folded into the reduction | **no** — see the mul result |
//!
//! The consequence for a Krylov solver: every scalar a conjugate-gradient or
//! BiCGStab iteration reduces (`rᵀz`, `pᵀKp`, `‖r‖²`) is a **sum of
//! independently formed products**, so it is bit-exact under any partition and
//! any arrival order. The one division per iteration and the one `sqrt` per
//! norm act on the already-reduced scalar, so they are single operations on a
//! deterministic input. The non-associativity of `Mul` and `sqrt` therefore does
//! not reach the reduction — it would only matter for a reduction whose operator
//! *is* a product or a root.
//!
//! An `f64` control runs the same permutation test so the suite shows its own
//! teeth: if the `f64` case ever stopped differing, the generator would have
//! stopped producing values that can disagree, and the `Fix128` greens would
//! mean nothing (`feedback_green_is_not_evidence_three_mechanisms`, mechanism A).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The control arm measures f64 rounding on purpose.
#![allow(clippy::disallowed_methods)]

use alice_physics::math::Fix128;

// ---------------------------------------------------------------------------
// deterministic sample generation (no dependency, fixed seed)
// ---------------------------------------------------------------------------

/// SplitMix64 — a fixed, self-contained generator so the samples are the same
/// on every target and the measurement is reproducible from the seed alone.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A `Fix128` whose integer part spans ±2^(bits) — small `bits` keeps sums
    /// inside the representable range, large `bits` forces wrap-around.
    fn next_fix(&mut self, int_bits: u32) -> Fix128 {
        let hi = (self.next_u64() as i64) >> (63 - int_bits);
        let lo = self.next_u64();
        Fix128::from_raw(hi, lo)
    }

    fn next_f64_like(&mut self, int_bits: u32) -> f64 {
        self.next_fix(int_bits).to_f64()
    }

    /// Fisher-Yates over indices, so every arm sums exactly the same multiset.
    fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            idx.swap(i, j);
        }
        idx
    }
}

// ---------------------------------------------------------------------------
// reduction shapes
// ---------------------------------------------------------------------------

fn sum_sequential(v: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for x in v {
        acc = acc + *x;
    }
    acc
}

fn sum_permuted(v: &[Fix128], order: &[usize]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for &i in order {
        acc = acc + v[i];
    }
    acc
}

/// Pairwise (binary tree) reduction — the shape a `rayon` or MPI reduce
/// actually produces, as opposed to a left fold.
fn sum_pairwise(v: &[Fix128]) -> Fix128 {
    if v.is_empty() {
        return Fix128::ZERO;
    }
    let mut level: Vec<Fix128> = v.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.chunks_exact(2);
        for pair in &mut it {
            next.push(pair[0] + pair[1]);
        }
        if let [last] = it.remainder() {
            next.push(*last);
        }
        level = next;
    }
    level[0]
}

/// How many contiguous blocks `sum_blocked_shuffled` will form for `ranks`.
///
/// Integer division makes this differ from `ranks` (4096 over 3 ranks is 3
/// blocks of 1366, 4096 over 7 is 7 blocks of 586), so the caller must ask
/// rather than assume — an arrival permutation shorter than the block count
/// silently drops blocks, which reads as a reduction-order failure.
fn block_count(len: usize, ranks: usize) -> usize {
    let chunk = len.div_ceil(ranks.max(1)).max(1);
    len.div_ceil(chunk)
}

/// Split into contiguous blocks, sum each block, then combine the block results
/// in a shuffled order — a distributed reduce with non-deterministic arrival.
fn sum_blocked_shuffled(v: &[Fix128], ranks: usize, arrival: &[usize]) -> Fix128 {
    let chunk = v.len().div_ceil(ranks.max(1)).max(1);
    let partials: Vec<Fix128> = v.chunks(chunk).map(sum_sequential).collect();
    assert_eq!(
        arrival.len(),
        partials.len(),
        "the arrival order must be a permutation of every block, or the reduce \
         drops terms and the comparison is meaningless"
    );
    let mut acc = Fix128::ZERO;
    for &i in arrival {
        acc = acc + partials[i];
    }
    acc
}

fn dot_sequential(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}

fn dot_permuted(a: &[Fix128], b: &[Fix128], order: &[usize]) -> Fix128 {
    let mut acc = Fix128::ZERO;
    for &i in order {
        acc = acc + a[i] * b[i];
    }
    acc
}

fn dot_pairwise(a: &[Fix128], b: &[Fix128]) -> Fix128 {
    let terms: Vec<Fix128> = (0..a.len()).map(|i| a[i] * b[i]).collect();
    sum_pairwise(&terms)
}

// ---------------------------------------------------------------------------
// the measurements
// ---------------------------------------------------------------------------

const N: usize = 4096;

/// `Fix128` addition is associative and commutative, so a sum is bit-exact
/// under every permutation, under a pairwise tree, and under a blocked
/// distributed reduce with shuffled arrival.
#[test]
fn fix128_sums_are_bit_exact_under_any_reduction_order() {
    let mut rng = SplitMix64(0x5EED_0001);
    let v: Vec<Fix128> = (0..N).map(|_| rng.next_fix(20)).collect();

    let reference = sum_sequential(&v);

    for trial in 0..64 {
        let order = rng.permutation(N);
        let got = sum_permuted(&v, &order);
        assert_eq!(
            got, reference,
            "trial {trial}: a permuted sum differs from the sequential one \
             ({got:?} vs {reference:?}); Fix128 addition would then not be associative"
        );
    }

    let tree = sum_pairwise(&v);
    assert_eq!(
        tree, reference,
        "a pairwise tree reduction differs from the left fold ({tree:?} vs {reference:?})"
    );

    for ranks in [2usize, 3, 7, 8, 64, 1024] {
        let blocks = block_count(N, ranks);
        for trial in 0..8 {
            let arrival = rng.permutation(blocks);
            let got = sum_blocked_shuffled(&v, ranks, &arrival);
            assert_eq!(
                got, reference,
                "ranks {ranks} ({blocks} blocks), trial {trial}: a blocked reduce with \
                 shuffled arrival differs from the sequential sum"
            );
        }
    }

    eprintln!("  Fix128 sum over {N} terms: {:.9e}", reference.to_f64());
    eprintln!("  64 permutations + pairwise tree + 6 rank counts × 8 arrivals: all bit-equal");
}

/// The same, with the samples large enough that the running total wraps.
///
/// This is the part that a saturating fixed-point type would fail: saturation is
/// not associative, so the order would decide the answer. `Fix128::add` wraps,
/// and wrapping *is* the group operation, so the answer survives.
#[test]
fn wrapping_overflow_does_not_break_order_independence() {
    let mut rng = SplitMix64(0x5EED_0002);
    // 62 integer bits: ~4096 terms of magnitude up to 2^62 overflow i64 many
    // times over.
    let v: Vec<Fix128> = (0..N).map(|_| rng.next_fix(62)).collect();

    let reference = sum_sequential(&v);
    let mut wrapped = false;
    let mut acc = Fix128::ZERO;
    for x in &v {
        let before = acc;
        acc = acc + *x;
        // A wrap shows up as a sum that moved the wrong way for the addend's sign.
        if (x.hi > 0 && acc.hi < before.hi) || (x.hi < 0 && acc.hi > before.hi) {
            wrapped = true;
        }
    }
    assert!(
        wrapped,
        "the sample never overflowed, so this test measured nothing; raise the \
         integer-bit width"
    );

    for trial in 0..64 {
        let order = rng.permutation(N);
        assert_eq!(
            sum_permuted(&v, &order),
            reference,
            "trial {trial}: a permuted sum differs once the accumulator wraps"
        );
    }
    assert_eq!(
        sum_pairwise(&v),
        reference,
        "pairwise tree differs under wrap"
    );
    eprintln!("  overflow confirmed, and all 64 permutations + tree still bit-equal");
}

/// A dot product is a sum of independently formed products, so it inherits the
/// sum's order-independence. This is the operation every Krylov iteration
/// reduces.
#[test]
fn fix128_dot_products_are_bit_exact_under_any_reduction_order() {
    let mut rng = SplitMix64(0x5EED_0003);
    let a: Vec<Fix128> = (0..N).map(|_| rng.next_fix(10)).collect();
    let b: Vec<Fix128> = (0..N).map(|_| rng.next_fix(10)).collect();

    let reference = dot_sequential(&a, &b);
    for trial in 0..64 {
        let order = rng.permutation(N);
        assert_eq!(
            dot_permuted(&a, &b, &order),
            reference,
            "trial {trial}: a permuted dot product differs from the sequential one"
        );
    }
    assert_eq!(
        dot_pairwise(&a, &b),
        reference,
        "a pairwise dot product differs from the sequential one"
    );
    eprintln!("  Fix128 dot over {N} terms: {:.9e}", reference.to_f64());
}

/// **The boundary.** `Mul` truncates to bits [192:64] of the 256-bit product, so
/// it is *not* associative: a reduction whose operator is a product does depend
/// on the order.
///
/// This is asserted as a positive finding rather than left implicit — it tells a
/// future parallel implementation exactly which reductions it may reorder.
#[test]
fn fix128_multiplication_is_not_associative() {
    let mut rng = SplitMix64(0x5EED_0004);
    let mut differing = 0usize;
    let trials = 4096;
    for _ in 0..trials {
        let a = rng.next_fix(4);
        let b = rng.next_fix(4);
        let c = rng.next_fix(4);
        if (a * b) * c != a * (b * c) {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "no triple out of {trials} disagreed, so this test is not measuring the \
         truncation it claims to; the sample or the operator changed"
    );
    eprintln!(
        "  (a·b)·c ≠ a·(b·c) in {differing}/{trials} triples \
         ({:.1}%) — product reductions are order-dependent",
        100.0 * differing as f64 / trials as f64
    );
}

/// Control: the same permutation test on `f64` must **fail to be bit-exact**.
///
/// Without this the `Fix128` greens above could come from a generator that only
/// produces values no reduction order can separate.
#[test]
fn f64_control_shows_the_permutation_test_has_teeth() {
    let mut rng = SplitMix64(0x5EED_0005);
    let v: Vec<f64> = (0..N).map(|_| rng.next_f64_like(20)).collect();

    let reference: f64 = {
        let mut acc = 0.0_f64;
        for x in &v {
            acc += *x;
        }
        acc
    };

    let mut differing = 0usize;
    let trials = 64;
    for _ in 0..trials {
        let order = rng.permutation(N);
        let mut acc = 0.0_f64;
        for &i in &order {
            acc += v[i];
        }
        if acc.to_bits() != reference.to_bits() {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "f64 summation agreed bit-for-bit under all {trials} permutations, which \
         means the samples cannot separate reduction orders — the Fix128 results \
         above are then not evidence of anything"
    );
    eprintln!(
        "  f64 control: {differing}/{trials} permutations differ from the sequential \
         sum (Fix128: 0/{trials})"
    );
}
