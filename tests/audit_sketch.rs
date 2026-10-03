//! Audit S3-1 oracles for `sketch` (HyperLogLog / DDSketch / Count-Min / HeavyHitters).
//!
//! `analytic_sketch_wiring` already carries reference tables for hashes, HLL
//! registers, Count-Min columns and DDSketch order statistics.  This file adds
//! what it does not check: `DDSketch::{merge, clear, sum, mean, min, max}`,
//! the HLL estimator over the full range of `n` (including the 2.5 m
//! linear-counting switch) and the closed form of `ALPHA`, plus oracles for
//! the degenerate inputs that the existing file *pins as measured behaviour*
//! (those are the `#[ignore]`d defect oracles below).

use alice_physics::sketch::{
    CountMinSketch, DDSketch, DDSketch128, FnvHasher, HyperLogLog10, HyperLogLog12, HyperLogLog14,
    HyperLogLog16, Mergeable,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

/// `merge` adds bins, count and sum and takes min/max: the merged sketch is
/// indistinguishable from one built from the union stream (all quantiles equal).
#[test]
fn ddsketch_merge_equals_the_sketch_of_the_union_stream() {
    let mut a = DDSketch::new(0.02);
    let mut b = DDSketch::new(0.02);
    let mut u = DDSketch::new(0.02);
    let mut s = 1u64;
    for i in 0..600 {
        let v = ((lcg(&mut s) % 2000) as f64 - 700.0) / 8.0; // negatives, zeros, positives
        if i % 3 == 0 {
            a.insert(v);
        } else {
            b.insert(v);
        }
        u.insert(v);
    }
    a.merge(&b);
    assert_eq!(a.count(), u.count());
    assert!((a.sum() - u.sum()).abs() < 1e-6);
    assert!((a.mean() - u.mean()).abs() < 1e-9);
    assert_eq!(a.min(), u.min());
    assert_eq!(a.max(), u.max());
    for k in 1..=100 {
        let q = k as f64 / 100.0;
        assert_eq!(a.quantile(q), u.quantile(q), "q = {q}");
    }
}

/// `count`, `sum`, `mean`, `min`, `max` accessors against direct computation.
#[test]
fn ddsketch_summary_statistics_match_direct_computation() {
    let vals = [-4.5, 0.0, 2.25, 8.0, -0.5, 100.0, 0.0];
    let mut d = DDSketch::new(0.01);
    for v in vals {
        d.insert(v);
    }
    assert_eq!(d.count(), 7);
    assert_eq!(d.sum(), 105.25);
    assert_eq!(d.mean(), 105.25 / 7.0);
    assert_eq!(d.min(), -4.5);
    assert_eq!(d.max(), 100.0);
    assert_eq!(d.alpha(), 0.01);
}

/// `clear` returns every statistic and quantile to the empty state and the sketch is reusable.
#[test]
fn ddsketch_clear_resets_and_the_sketch_is_reusable() {
    let mut d = DDSketch::new(0.05);
    for v in [1.0, -2.0, 0.0, 30.0] {
        d.insert(v);
    }
    d.clear();
    assert_eq!(d.count(), 0);
    assert_eq!(d.sum(), 0.0);
    assert_eq!(d.mean(), 0.0);
    assert_eq!(d.min(), f64::INFINITY);
    assert_eq!(d.max(), f64::NEG_INFINITY);
    assert_eq!(d.quantile(0.5), 0.0);
    let fresh = {
        let mut f = DDSketch::new(0.05);
        f.insert(7.0);
        f
    };
    d.insert(7.0);
    assert_eq!(d.quantile(0.5), fresh.quantile(0.5));
    assert_eq!(d.quantile(1.0), fresh.quantile(1.0));
}

/// Negative values: sign and order are kept (the median of a symmetric set is 0,
/// the lowest quantile is the most negative value's edge).
#[test]
fn ddsketch_orders_negative_zero_and_positive_values() {
    let mut d = DDSketch::new(0.01);
    for v in [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0] {
        d.insert(v);
    }
    assert_eq!(d.quantile(0.5), 0.0, "rank 4 of 7 is the zero");
    let lo = d.quantile(1.0 / 7.0);
    assert!((-30.0..=-30.0 * 0.97).contains(&lo), "rank-1 value {lo}");
    let hi = d.quantile(1.0);
    assert!((30.0 * 0.97..=30.0).contains(&hi), "rank-7 value {hi}");
}

fn alpha_const(m: f64) -> f64 {
    0.7213 / (1.0 + 1.079 / m)
}

/// Fill every register with rho = 1 (hash with the top bit set -> w has P leading zeros).
/// Then sum = m/2, raw = alpha m^2 / (m/2) = 2 alpha m, zeros = 0 -> raw is returned:
/// this pins `ALPHA` and the `2^-k` table at k = 1 for every size.
macro_rules! all_registers_one {
    ($name:ident, $ty:ident) => {
        #[test]
        fn $name() {
            let mut h = $ty::new();
            let m = $ty::M;
            for idx in 0..m {
                h.insert_hash((1u64 << 63) | idx as u64);
            }
            assert!(h.registers().iter().all(|&r| r == 1));
            let want = 2.0 * alpha_const(m as f64) * m as f64;
            let got = h.cardinality();
            assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
        }
    };
}
all_registers_one!(hll10_alpha_closed_form, HyperLogLog10);
all_registers_one!(hll12_alpha_closed_form, HyperLogLog12);
all_registers_one!(hll14_alpha_closed_form, HyperLogLog14);
all_registers_one!(hll16_alpha_closed_form, HyperLogLog16);

/// `rho` closed form: with `idx` = low P bits and `w = hash >> P`,
/// `rho = leading_zeros_in_(64-P)_bits(w) + 1`, and `w = 0` gives `64 - P + 1`.
#[test]
fn hll_rho_is_the_position_of_the_first_one_bit() {
    let p = HyperLogLog12::P;
    for lz in [0usize, 1, 5, 20, 51] {
        let mut h = HyperLogLog12::new();
        let w: u64 = 1u64 << (64 - p - 1 - lz);
        h.insert_hash((w << p) | 7);
        assert_eq!(h.registers()[7] as usize, lz + 1, "lz = {lz}");
    }
    let mut h = HyperLogLog12::new();
    h.insert_hash(9);
    assert_eq!(h.registers()[9] as usize, 64 - p + 1, "w = 0");
    // a smaller rho never lowers a register
    let mut h = HyperLogLog12::new();
    h.insert_hash((1u64 << (64 - 12 - 1 - 3 + 12)) | 5);
    h.insert_hash((1u64 << 63) | 5);
    assert_eq!(h.registers()[5], 4);
}

/// Linear-counting range: `k` registers set to rho = 1 and `raw <= 2.5 m` gives `m ln(m / zeros)`.
#[test]
fn hll_small_range_uses_linear_counting_closed_form() {
    let m = HyperLogLog10::M;
    let k = 100usize;
    let mut h = HyperLogLog10::new();
    for idx in 0..k {
        h.insert_hash((1u64 << 63) | idx as u64);
    }
    let zeros = (m - k) as f64;
    #[allow(clippy::disallowed_methods)]
    let want = m as f64 * (m as f64 / zeros).ln();
    let got = h.cardinality();
    assert!((got - want).abs() < 1e-9 * want, "{got} vs {want}");
}

/// Estimator error against the documented `~1.04/sqrt(m)` over the whole range of `n`,
/// including the neighbourhood of the `2.5 m` linear-counting switch (HyperLogLog++ has
/// a bias correction there; plain HLL does not).
#[test]
fn hll_error_stays_within_four_sigma_across_the_estimator_switch() {
    let m = HyperLogLog12::M as f64;
    let sigma = 1.04 / m.sqrt();
    let mut worst = (0.0f64, 0u64);
    for n in [
        500u64, 2_000, 5_000, 8_000, 10_000, 10_500, 11_000, 12_000, 14_000, 20_000, 60_000,
    ] {
        let mut h = HyperLogLog12::new();
        for i in 0..n {
            h.insert_hash(FnvHasher::hash_u64(
                i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x5555,
            ));
        }
        let rel = (h.cardinality() - n as f64).abs() / n as f64;
        if rel > worst.0 {
            worst = (rel, n);
        }
    }
    assert!(
        worst.0 <= 4.0 * sigma,
        "worst relative error {:.4} at n = {} exceeds 4 sigma = {:.4}",
        worst.0,
        worst.1,
        4.0 * sigma
    );
}

/// HLL merge is the register-wise max (idempotent, commutative).
#[test]
fn hll_merge_is_a_register_max_and_idempotent() {
    let mut a = HyperLogLog10::new();
    let mut b = HyperLogLog10::new();
    let mut u = HyperLogLog10::new();
    for i in 0..400u64 {
        let h = FnvHasher::hash_u64(i);
        if i % 2 == 0 {
            a.insert_hash(h);
        } else {
            b.insert_hash(h);
        }
        u.insert_hash(h);
    }
    let mut ab = a.clone();
    ab.merge(&b);
    let mut ba = b.clone();
    ba.merge(&a);
    assert_eq!(ab.registers(), u.registers());
    assert_eq!(ab.registers(), ba.registers());
    let before = ab.registers().to_vec();
    ab.merge(&ab.clone());
    assert_eq!(ab.registers(), &before[..]);
}

// ---------------------------------------------------------------------------
// Known-defect oracles (the existing file pins the *current* behaviour of these)
// ---------------------------------------------------------------------------

/// `quantile(0.0)` should be the smallest value's bucket edge (within the data range).
#[test]
#[ignore = "known defect: AUD-A-S3W1-006: DDSketch::quantile(0.0) (rank 0) returns the outermost negative bucket edge, -2.1e13 for 1..=100 at alpha 0.01, instead of a value near min; existing analytic_sketch_wiring pins this behaviour"]
fn ddsketch_quantile_zero_is_inside_the_data_range() {
    let mut d = DDSketch::new(0.01);
    for i in 1..=100 {
        d.insert(i as f64);
    }
    let q0 = d.quantile(0.0);
    assert!((0.9..=1.0).contains(&q0), "quantile(0) = {q0}");
}

/// A NaN input is either rejected or at least must not be counted as a zero value nor
/// poison `sum` / `mean`.
#[test]
#[ignore = "known defect: AUD-A-S3W1-007: DDSketch::insert(NaN) is counted in the zero block and makes sum()/mean() NaN permanently; existing oracle pins 'NaN lands in the zero block'"]
fn ddsketch_nan_input_does_not_poison_the_sketch() {
    let mut d = DDSketch::new(0.01);
    for v in [1.0, 2.0, 3.0] {
        d.insert(v);
    }
    d.insert(f64::NAN);
    assert!(d.mean().is_finite(), "mean = {}", d.mean());
    assert!(d.sum().is_finite());
    // the NaN must not have become a zero: the median of {1,2,3} stays ~2
    let med = d.quantile(0.5);
    assert!(med > 1.5, "median {med}");
}

/// Infinite inputs must not panic (debug builds): `ceil() as i32 + offset` overflows.
#[test]
#[ignore = "known defect: AUD-A-S3W1-008: DDSketch::insert(+-inf) panics with integer overflow in debug builds (wraps in release); existing oracle pins the panic"]
fn ddsketch_infinite_input_does_not_panic() {
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut s = DDSketch::new(0.01);
            s.insert(v);
            s.count()
        }));
        assert!(r.is_ok(), "insert({v}) panicked");
    }
}

/// The type's guarantee is a relative error bound for any inserted value; a value
/// below the first bucket is clamped into bin 0 and answered with an edge far from it.
#[test]
#[ignore = "known defect: AUD-A-S3W1-009: values below gamma^-(BINS/4) (3.6e-5 at alpha 0.01, DDSketch2048) are clamped into bin 0; quantile(1.0) of {1e-6} answers 3.4e-5 (relative error 3300%); values above the top bin are dropped from the bins (quantile falls through to max); neither is documented"]
fn ddsketch_relative_error_holds_below_the_first_bucket() {
    let mut d = DDSketch::new(0.01);
    d.insert(1e-6);
    let est = d.quantile(1.0);
    let rel = (est - 1e-6).abs() / 1e-6;
    assert!(
        rel <= 2.0 * 0.01 / 1.01 + 1e-12,
        "estimate {est}, relative error {rel}"
    );
}

/// Same for the upper end: a value above the top bucket is counted but not binned, so
/// a middle quantile of three huge values returns the maximum.
#[test]
#[ignore = "known defect: AUD-A-S3W1-009: values above the top bin (gamma^(3 BINS/4) = 2e13 at alpha 0.01) are dropped from the bins; median of {1e14, 5e14, 1e15} answers 1e15 (2x)"]
fn ddsketch_relative_error_holds_above_the_last_bucket() {
    let mut d = DDSketch::new(0.01);
    for v in [1e14, 5e14, 1e15] {
        d.insert(v);
    }
    let est = d.quantile(0.5);
    let rel = (est - 5e14).abs() / 5e14;
    assert!(
        rel <= 2.0 * 0.01 / 1.01 + 1e-12,
        "median {est}, relative error {rel}"
    );
}

/// Count-Min counters saturate (`saturating_add`), so `total` should not overflow either.
#[test]
#[ignore = "known defect: AUD-A-S3W1-010: CountMinSketch::insert_hash(_, u64::MAX) twice: counters saturate but `total += count` overflows (panic in debug, wrap in release); merge has the same plain add on total"]
fn countmin_total_does_not_overflow_when_counters_saturate() {
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut c = CountMinSketch::new();
        c.insert_hash(1, u64::MAX);
        c.insert_hash(2, u64::MAX);
        c.total()
    }));
    assert!(r.is_ok(), "total overflowed");
}

/// Small sanity: the other DDSketch sizes obey the same relative-error bound in range.
#[test]
fn ddsketch128_edge_error_is_within_two_alpha_over_one_plus_alpha() {
    let alpha = 0.1;
    let mut d = DDSketch128::new(alpha);
    let mut vals = Vec::new();
    for i in 1..=500 {
        let v = 0.5 + i as f64 * 0.37;
        vals.push(v);
        d.insert(v);
    }
    let bound = 2.0 * alpha / (1.0 + alpha);
    for k in 1..=50 {
        let q = k as f64 / 50.0;
        let rank = (q * 500.0).ceil() as usize;
        let truth = vals[rank - 1];
        let est = d.quantile(q);
        assert!(
            est <= truth + 1e-9,
            "edge never exceeds the value: {est} > {truth}"
        );
        assert!(
            (truth - est) / truth <= bound + 1e-9,
            "q {q}: {est} vs {truth}"
        );
    }
}
