//! Audit oracles for `alice_physics::privacy`.
//!
//! Expected values come from the closed forms of the mechanisms (Laplace CDF and
//! moments, randomized-response likelihood ratios, the composition rule of the
//! budget, the splitmix64 seed scramble and Marsaglia's xorshift64 step) and
//! from seeded Monte Carlo runs whose tolerances are set from the binomial /
//! sampling standard errors. The mechanism-level privacy claims (epsilon of a
//! configured mechanism) are computed from the stated output probabilities,
//! not from the implementation.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::needless_range_loop)]

use alice_physics::privacy::{
    LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
    RAPPOR_BITS,
};
use alice_physics::sketch::FnvHasher;

/// Marsaglia xorshift64 (13, 7, 17), written out independently.
fn step(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}
fn inv_left(y: u64, s: u32) -> u64 {
    let mut x = y;
    for _ in 0..(64 / s + 2) {
        x = y ^ (x << s);
    }
    x
}
fn inv_right(y: u64, s: u32) -> u64 {
    let mut x = y;
    for _ in 0..(64 / s + 2) {
        x = y ^ (x >> s);
    }
    x
}
/// splitmix64 step (golden-gamma add, then the finaliser), written out from
/// the published constants (Steele, Lea, Flood, OOPSLA 2014; Vigna,
/// `splitmix64.c`): the seed scramble of `XorShift64::new`, and a way to turn
/// small integers into well-mixed seeds.
fn splitmix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
fn inv_step(y: u64) -> u64 {
    inv_left(inv_right(inv_left(y, 17), 7), 13)
}
/// Inverse of `splitmix`: undo each xor-shift, multiply by the inverse of each
/// multiplier mod 2^64 (0x96DE_1B17_3F11_9089 and 0x3196_42B2_D24D_8EC3,
/// computed separately with python3 `pow(m, -1, 2**64)` and checked below by
/// the round trip), then subtract the golden gamma.
fn inv_splitmix(y: u64) -> u64 {
    let mut z = inv_right(y, 31);
    z = inv_right(z.wrapping_mul(0x3196_42b2_d24d_8ec3), 27);
    z = inv_right(z.wrapping_mul(0x96de_1b17_3f11_9089), 30);
    z.wrapping_sub(0x9e37_79b9_7f4a_7c15)
}

// ------------------------------------------------------------------ generator

#[test]
fn xorshift_is_the_marsaglia_step_applied_to_the_scrambled_seed() {
    // The Marsaglia step itself, on a raw state: 1 -> 8193 -> 8257 ->
    // 1082269761 (hand computed).
    assert_eq!(step(1), 1_082_269_761);
    // The seed is scrambled by one splitmix64 step before it becomes the
    // state (AUD-A-S4W3-030). splitmix64(0) = 0xE220_A839_7B1D_CDAF is the
    // first output of Vigna's reference `splitmix64.c` seeded with 0;
    // splitmix64(1) = 0x910A_2DEC_8902_5CC1 and the Marsaglia step of it,
    // 0x7274_658B_CB6F_4838, were computed in a separate python3 one-off from
    // the published constants.
    assert_eq!(splitmix(0), 0xe220_a839_7b1d_cdaf);
    assert_eq!(splitmix(1), 0x910a_2dec_8902_5cc1);
    assert_eq!(step(0x910a_2dec_8902_5cc1), 0x7274_658b_cb6f_4838);
    let mut r = XorShift64::new(1);
    assert_eq!(r.next_u64(), 0x7274_658b_cb6f_4838);
    let seed = 0x1234_5678_9abc_def0u64;
    let mut s = splitmix(seed);
    let mut r = XorShift64::new(seed);
    for _ in 0..100 {
        s = step(s);
        assert_eq!(r.next_u64(), s);
    }
    // inverse sanity of the helpers
    for &y in &[0u64, 1, 0xdead_beef_cafe_f00d, u64::MAX] {
        assert_eq!(step(inv_step(y)), y);
        assert_eq!(splitmix(inv_splitmix(y)), y);
        assert_eq!(inv_splitmix(splitmix(y)), y);
    }
}

#[test]
fn first_draw_of_a_small_seed_is_not_an_extreme_outlier() {
    let tiny = (1u64..=100)
        .filter(|&s| XorShift64::new(s).next_f64() < 1e-6)
        .count();
    assert!(
        tiny <= 1,
        "{tiny} of the seeds 1..=100 start with a draw below 1e-6"
    );
    let first = LaplaceNoise::with_seed(1.0, 1.0, 42).sample();
    assert!(first > -15.0, "first sample for seed 42: {first}");
}

#[test]
fn xorshift_uniform_draws_are_in_unit_interval_with_flat_histogram() {
    let mut r = XorShift64::new(777);
    let n = 160_000usize;
    let mut bucket = [0usize; 16];
    let (mut sum, mut sum2) = (0.0, 0.0);
    for _ in 0..n {
        let u = r.next_f64();
        assert!((0.0..1.0).contains(&u));
        bucket[(u * 16.0) as usize] += 1;
        sum += u;
        sum2 += u * u;
    }
    let expect = n as f64 / 16.0;
    let sigma = (expect * (1.0 - 1.0 / 16.0)).sqrt();
    for (i, &c) in bucket.iter().enumerate() {
        assert!((c as f64 - expect).abs() < 5.0 * sigma, "bucket {i}: {c}");
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    assert!((mean - 0.5).abs() < 5.0 * (1.0f64 / 12.0 / n as f64).sqrt());
    assert!((var - 1.0 / 12.0).abs() < 0.002, "variance {var}");
}

#[test]
fn xorshift_range_and_bool_frequencies() {
    let mut r = XorShift64::new(31);
    let n = 100_000;
    let (mut sum, mut hits) = (0.0, 0usize);
    for _ in 0..n {
        let v = r.next_f64_range(-3.0, 7.0);
        assert!((-3.0..7.0).contains(&v));
        sum += v;
        if r.next_bool(0.3) {
            hits += 1;
        }
    }
    assert!((sum / f64::from(n) - 2.0).abs() < 5.0 * (100.0f64 / 12.0 / f64::from(n)).sqrt());
    let sigma = (0.3f64 * 0.7 / f64::from(n)).sqrt();
    assert!((hits as f64 / f64::from(n) - 0.3).abs() < 5.0 * sigma);
}

// ------------------------------------------------------------------ Laplace

fn laplace_cdf(x: f64, b: f64) -> f64 {
    if x < 0.0 {
        0.5 * (x / b).exp()
    } else {
        1.0 - 0.5 * (-x / b).exp()
    }
}

#[test]
fn laplace_scale_is_sensitivity_over_epsilon_and_gives_exactly_epsilon_density_ratio() {
    for &(d, e) in &[(1.0, 1.0), (2.0, 0.5), (0.25, 4.0)] {
        let n = LaplaceNoise::with_seed(d, e, 1);
        assert!((n.scale() - d / e).abs() < 1e-15);
        // density ratio of outputs for inputs differing by the sensitivity: exp(d / b) = exp(eps)
        assert!(((d / n.scale()) - e).abs() < 1e-12);
    }
}

#[test]
fn laplace_samples_follow_the_laplace_cdf_moments_and_tails() {
    let (sens, eps) = (1.0, 0.5); // b = 2
    let b = sens / eps;
    let mut noise = LaplaceNoise::with_seed(sens, eps, 20_240_601);
    let n = 200_000usize;
    let xs: Vec<f64> = (0..n).map(|_| noise.sample()).collect();
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / nf;
    let mad = xs.iter().map(|x| x.abs()).sum::<f64>() / nf;
    assert!(mean.abs() < 5.0 * (2.0 * b * b / nf).sqrt(), "mean {mean}");
    assert!(
        (var - 2.0 * b * b).abs() < 0.05 * 2.0 * b * b,
        "variance {var} vs {}",
        2.0 * b * b
    );
    assert!((mad - b).abs() < 0.02 * b, "E|X| {mad} vs {b}");
    // empirical CDF at fixed points
    for &x in &[-6.0, -3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0, 6.0] {
        let emp = xs.iter().filter(|&&v| v <= x).count() as f64 / nf;
        let want = laplace_cdf(x, b);
        let sigma = (want * (1.0 - want) / nf).sqrt();
        assert!(
            (emp - want).abs() < 5.0 * sigma + 1e-9,
            "F({x}): {emp} vs {want}"
        );
    }
}

#[test]
fn laplace_privatize_is_symmetric_about_the_value_and_int_rounding_matches() {
    let mut noise = LaplaceNoise::with_seed(1.0, 1.0, 5);
    let n = 100_000;
    let mut above = 0usize;
    for _ in 0..n {
        if noise.privatize(100.0) > 100.0 {
            above += 1;
        }
    }
    let sigma = (0.25 / f64::from(n)).sqrt();
    assert!((above as f64 / f64::from(n) - 0.5).abs() < 5.0 * sigma);
    // privatize_int returns integers with the same centre
    let mut noise = LaplaceNoise::with_seed(1.0, 1.0, 6);
    let mean = (0..n).map(|_| noise.privatize_int(40) as f64).sum::<f64>() / f64::from(n);
    assert!((mean - 40.0).abs() < 0.05, "mean {mean}");
}

#[test]
// AUD-A-S4W3-025
fn laplace_sample_is_always_finite_even_when_the_uniform_draw_is_zero() {
    // seed whose first xorshift output has its top 53 bits equal to 0: the
    // state before the step is inv_step(1), and the seed is the preimage of
    // that state under the splitmix64 scramble of `XorShift64::new`
    let state = inv_step(1);
    let seed = inv_splitmix(state);
    assert_eq!(splitmix(seed), state);
    assert_eq!(step(splitmix(seed)) >> 11, 0);
    let mut noise = LaplaceNoise::with_seed(1.0, 1.0, seed);
    let x = noise.sample();
    assert!(x.is_finite(), "sample = {x}");
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-024: parameters that make a mechanism meaningless are accepted silently: Laplace with epsilon = 0 samples +-inf (privatize_int saturates), estimate_proportion with p = 0 returns NaN"]
fn invalid_parameters_do_not_produce_non_finite_output() {
    let mut zero_eps = LaplaceNoise::with_seed(1.0, 0.0, 9);
    assert!(
        zero_eps.sample().is_finite(),
        "epsilon = 0 gives a non-finite sample"
    );
    assert!(
        !RandomizedResponse::estimate_proportion(0.0, 10, 5).is_nan(),
        "p = 0 gives NaN"
    );
}

// ------------------------------------------------------------------ randomized response

/// Worst-case likelihood ratio of a report for the two inputs, for the mechanism
/// "truth with probability p, otherwise a fair random bit" (the module's own
/// description): epsilon = ln((p + (1-p)/2) / ((1-p)/2)).
fn rr_epsilon(p: f64) -> f64 {
    ((p + 0.5 * (1.0 - p)) / (0.5 * (1.0 - p))).ln()
}

#[test]
fn randomized_response_report_probabilities_match_truth_or_coin() {
    let p = 0.6;
    let mut rr = RandomizedResponse::with_probability(p, 2024);
    let n = 200_000usize;
    let (mut t_given_t, mut t_given_f) = (0usize, 0usize);
    for _ in 0..n {
        if rr.privatize(true) {
            t_given_t += 1;
        }
        if rr.privatize(false) {
            t_given_f += 1;
        }
    }
    let nf = n as f64;
    let (want_t, want_f) = (p + 0.5 * (1.0 - p), 0.5 * (1.0 - p));
    let s1 = (want_t * (1.0 - want_t) / nf).sqrt();
    let s2 = (want_f * (1.0 - want_f) / nf).sqrt();
    assert!(
        (t_given_t as f64 / nf - want_t).abs() < 5.0 * s1,
        "P(1|1) {}",
        t_given_t as f64 / nf
    );
    assert!(
        (t_given_f as f64 / nf - want_f).abs() < 5.0 * s2,
        "P(1|0) {}",
        t_given_f as f64 / nf
    );
    // privatize_bit agrees in distribution
    let ones = (0..n).filter(|_| rr.privatize_bit(1) == 1).count() as f64 / nf;
    assert!((ones - want_t).abs() < 5.0 * s1);
}

#[test]
fn estimate_proportion_recovers_the_true_rate_from_noisy_reports() {
    let (p, truth) = (0.7, 0.3);
    let mut rr = RandomizedResponse::with_probability(p, 99);
    let mut coin = XorShift64::new(4242);
    let n = 200_000u64;
    let mut k = 0u64;
    for _ in 0..n {
        let bit = coin.next_bool(truth);
        if rr.privatize(bit) {
            k += 1;
        }
    }
    let est = RandomizedResponse::estimate_proportion(p, n, k);
    // sd of the estimator: sqrt(q(1-q)/n) / p with q = P(report=1)
    let q = p * truth + 0.5 * (1.0 - p);
    let sd = (q * (1.0 - q) / n as f64).sqrt() / p;
    assert!((est - truth).abs() < 5.0 * sd, "estimate {est}");
    // exact algebra on dyadic input: observed 1/2 with p = 1/2 -> (0.5 - 0.25) / 0.5
    assert!((RandomizedResponse::estimate_proportion(0.5, 8, 4) - 0.5).abs() < 1e-15);
    assert_eq!(RandomizedResponse::estimate_proportion(0.8, 0, 0), 0.0);
}

#[test]
fn randomized_response_new_epsilon_is_not_weaker_than_requested() {
    for &eps in &[0.25, 0.5, 1.0, 2.0, 4.0] {
        let p = RandomizedResponse::new(eps).p_true();
        let actual = rr_epsilon(p);
        assert!(
            actual <= eps + 1e-9,
            "requested {eps}, mechanism has {actual}"
        );
    }
}

#[test]
fn with_probability_does_not_silently_weaken_privacy() {
    let rr = RandomizedResponse::with_probability(0.2, 1);
    assert!(
        (rr.p_true() - 0.2).abs() < 1e-12,
        "p_true = {}",
        rr.p_true()
    );
}

// ------------------------------------------------------------------ RAPPOR

fn bloom_of(value: u64) -> [u8; RAPPOR_BITS] {
    let mut b = [0u8; RAPPOR_BITS];
    for i in 0..3u128 {
        let h = FnvHasher::hash_u128((value as u128) | (i << 64));
        b[(h as usize) % RAPPOR_BITS] = 1;
    }
    b
}

#[test]
fn rappor_per_bit_report_probabilities_follow_the_two_stage_response() {
    // f = 0.5, p = 0.75, q = 0.25: a bloom-1 bit reports 1 with
    // (1 - f/2) p + (f/2) q = 0.625, a bloom-0 bit with (f/2) p + (1 - f/2) q = 0.375.
    let mut r = Rappor::default_params();
    assert_eq!(r.params(), (0.5, 0.75, 0.25));
    let value = 0xfeed_beefu64;
    let bloom = bloom_of(value);
    let n = 4000usize;
    let mut ones = [0usize; RAPPOR_BITS];
    for _ in 0..n {
        let rep = r.privatize(value);
        assert_eq!(rep.len(), RAPPOR_BITS);
        for i in 0..RAPPOR_BITS {
            ones[i] += usize::from(rep[i]);
        }
    }
    let nf = n as f64;
    let (mut sum1, mut c1, mut sum0, mut c0) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..RAPPOR_BITS {
        if bloom[i] == 1 {
            sum1 += ones[i] as f64 / nf;
            c1 += 1.0;
        } else {
            sum0 += ones[i] as f64 / nf;
            c0 += 1.0;
        }
    }
    let (m1, m0) = (sum1 / c1, sum0 / c0);
    assert!((m1 - 0.625).abs() < 0.02, "bloom-1 bit rate {m1}");
    assert!((m0 - 0.375).abs() < 0.01, "bloom-0 bit rate {m0}");
}

#[test]
fn rappor_clamps_its_parameters_to_the_documented_ranges() {
    let r = Rappor::new(0.9, 1.5, -0.5);
    assert_eq!(r.params(), (0.5, 1.0, 0.0));
    let r = Rappor::new(-0.1, 0.4, 0.6);
    assert_eq!(r.params(), (0.0, 0.4, 0.6));
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-023: Rappor::default_params documents 'approximately epsilon = 2' but with f = 0.5, p = 0.75, q = 0.25 each Bloom bit has likelihood ratio 5/3 (epsilon 0.51) and two values differ in up to 6 bits, so one report has epsilon up to 3.07 (the permanent-response bound is 6.59)"]
fn rappor_default_params_epsilon_is_about_two() {
    let (f, p, q) = Rappor::default_params().params();
    // P(out = 1 | bloom = 1), P(out = 1 | bloom = 0)
    let a = (1.0 - f / 2.0) * p + (f / 2.0) * q;
    let c = (f / 2.0) * p + (1.0 - f / 2.0) * q;
    let per_bit = (a / c).max((1.0 - c) / (1.0 - a)).ln();
    let worst = 6.0 * per_bit; // two 3-hash Bloom filters differ in at most 6 bits
    assert!(
        (worst - 2.0).abs() < 0.3,
        "worst-case epsilon of one report: {worst}"
    );
}

// ------------------------------------------------------------------ budget

#[test]
fn budget_sequential_composition_sums_epsilons_and_refuses_overspend() {
    let mut b = PrivacyBudget::new(2.0);
    assert!(b.try_spend(0.5));
    assert!(b.try_spend(0.75));
    assert!(b.try_spend(0.5));
    assert_eq!(b.query_count(), 3);
    assert!((b.spent() - 1.75).abs() < 1e-15);
    assert!((b.remaining() - 0.25).abs() < 1e-15);
    // would exceed: refused and nothing changes
    assert!(!b.try_spend(0.5));
    assert_eq!(b.query_count(), 3);
    assert!((b.spent() - 1.75).abs() < 1e-15);
    // exactly reaching the maximum is allowed and exhausts it
    assert!(b.try_spend(0.25));
    assert!(b.is_exhausted());
    assert_eq!(b.remaining(), 0.0);
    // reset restores the full budget
    b.reset();
    assert_eq!(b.spent(), 0.0);
    assert_eq!(b.query_count(), 0);
    assert_eq!(b.remaining(), 2.0);
    assert!(!b.is_exhausted());
}

#[test]
fn budget_never_lets_spent_epsilon_decrease() {
    let mut b = PrivacyBudget::new(1.0);
    assert!(b.try_spend(0.5));
    let accepted = b.try_spend(-0.5);
    assert!(!accepted, "negative spend accepted");
    assert!(b.spent() >= 0.5, "spent decreased to {}", b.spent());
    assert!(b.remaining() <= 1.0);
}

// ------------------------------------------------------------------ aggregator

#[test]
fn aggregator_standard_error_equals_the_empirical_spread_of_the_mean() {
    let (b, n) = (2.0, 400usize);
    let mut spread = Vec::new();
    for rep in 0..300u64 {
        let mut noise = LaplaceNoise::with_seed(1.0, 0.5, splitmix(1000 + rep));
        let mut agg = PrivateAggregator::new(b);
        for _ in 0..n {
            agg.add(noise.privatize(10.0));
        }
        spread.push(agg.estimate_mean());
        if rep == 0 {
            let se = agg.standard_error();
            assert!((se - b * 2f64.sqrt() / (n as f64).sqrt()).abs() < 1e-15);
        }
    }
    let m = spread.iter().sum::<f64>() / spread.len() as f64;
    let sd = (spread.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (spread.len() as f64 - 1.0))
        .sqrt();
    let se = b * 2f64.sqrt() / (n as f64).sqrt();
    assert!(
        (m - 10.0).abs() < 5.0 * se / (spread.len() as f64).sqrt(),
        "mean of means {m}"
    );
    assert!(
        (sd - se).abs() < 0.15 * se,
        "empirical sd {sd} vs stated standard error {se}"
    );
}

#[test]
fn aggregator_sum_mean_count_and_reset() {
    let mut a = PrivateAggregator::new(1.0);
    for v in [1.5, 2.5, -1.0, 4.0] {
        a.add(v);
    }
    assert_eq!(a.count(), 4);
    assert_eq!(a.estimate_sum(), 7.0);
    assert_eq!(a.estimate_mean(), 1.75);
    a.reset();
    assert_eq!(a.count(), 0);
    assert_eq!(a.estimate_sum(), 0.0);
    assert!(a.standard_error().is_infinite());
}

// ------------------------------------------------------------------ entropy

#[test]
fn from_entropy_seed_is_not_recoverable_from_the_clock() {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    };
    let t0 = now();
    let first = XorShift64::from_entropy().next_u64();
    let t1 = now();
    assert!(t1 - t0 < 50_000_000, "clock window unexpectedly large");
    let found = (t0..=t1).any(|s| step(s) == first);
    assert!(
        !found,
        "seed recovered from the wall clock within a {} ns window",
        t1 - t0
    );
}

#[test]
fn laplace_new_scale_is_sensitivity_over_epsilon() {
    // the entropy-seeded constructor: the scale does not depend on the generator
    for &(d, e) in &[(2.0, 0.5), (1.0, 4.0), (0.3, 0.1)] {
        let n = LaplaceNoise::new(d, e);
        assert!((n.scale() - d / e).abs() < 1e-12, "scale {}", n.scale());
    }
}
