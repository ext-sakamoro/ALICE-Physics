//! Oracles for the wiring of `privacy`: the budget tracker, the Laplace
//! mechanism and its aggregator, randomized response, RAPPOR and the
//! xorshift generator behind them.
//!
//! # Closed forms (expected values are derived without calling the item)
//!
//! * **Budget**: `remaining = ε_max − Σ accepted` exactly when every spend is
//!   dyadic (the running sum is exact), `try_spend(ε)` accepts exactly when
//!   `Σ + ε ≤ ε_max` (equality accepted), `is_exhausted` is `Σ ≥ ε_max`,
//!   `query_count` is the number of accepted spends, and `reset` reproduces
//!   `PrivacyBudget::new(ε_max)` bit for bit.
//! * **xorshift**: the test carries its own copy of the three-shift step
//!   (`x ^= x << 13; x ^= x >> 7; x ^= x << 17`) and of `U = (x >> 11)·2⁻⁵³`,
//!   so every draw of the library generator is predicted from the seed. For
//!   `next_f64_range(lo, hi)` with `hi − lo` a power of two the product is
//!   exact and `(hi − lo)·U + lo` is one rounding whichever way it is fused.
//! * **Laplace**: with `u = U − ½`, the sample is `−sign(u)·b·ln(1 − 2|u|)`
//!   (`b = Δf/ε`), predicted from the seed via the generator copy and
//!   `det_math::ln64`; `privatize(v) = v + sample`, `privatize_int(v) =
//!   round(v + sample)` with Rust's saturating `as i64`.
//! * **Aggregator**: `estimate_sum = Σ` (exact for dyadic inputs),
//!   `estimate_mean = Σ / n`, `standard_error = b·√2/√n`, which is `b·√2`
//!   at `n = 1`, exactly `b` at `n = 2` and `b/2` at `n = 8` when `b` is a
//!   power of two (`√2` and `√8 = 2√2` round to the same constant).
//! * **Randomized response**: `p_true = e^ε/(e^ε + 1)` via `det_math::exp64`
//!   (`ε = 0` gives exactly ½ because `exp64(0) = 1` exactly); each report
//!   is `truth` if the first draw is `< p` else `second draw < ½`, predicted
//!   from the seed; the unbiased proportion from `k` of `n` positives is
//!   `(k/n − (1 − p)/2) / p` (the brief's `(k/n − (1 − p'))/(2p' − 1)` is the
//!   same formula under the flip convention `p' = (1 + p)/2`), clamped to
//!   `[0, 1]`.
//! * **RAPPOR**: `f = 0, p = 1, q = 0` is the identity on the Bloom filter,
//!   `f = 0, p = 0, q = 1` its complement, `p = q = 1` all ones and
//!   `p = q = 0` all zeros regardless of the (unseeded) generator; the Bloom
//!   filter is the three `FnvHasher::hash_u128(v | i << 64) mod 64`
//!   positions.
//!
//! # Degenerate input (current contract, pinned — not all of it is documented)
//!
//! `ε_max = 0` is exhausted from the start; a zero spend is accepted even on
//! an exhausted budget and counts as a query; a spend larger than the budget
//! (or `+∞`, or `NaN`) is refused and leaves the state bit for bit; a
//! negative spend is accepted and grows `remaining` past `ε_max`; a negative
//! `ε_max` is exhausted with `remaining = 0`. Laplace with `ε = 0` has
//! `b = +∞` and every sample is `±∞` (so `privatize_int` saturates to
//! `i64::MAX` / `i64::MIN`); `Δf = 0` makes `privatize` the identity.
//! `estimate_proportion` returns `0` for `n = 0`, clamps `k > n` to `1`, and
//! with `p = 0` divides by zero (`NaN` or `1`); `p = ½` is **not** singular
//! in this convention (`(k/n − ¼)/½`). `RandomizedResponse::new` does not
//! clamp: `ε → +∞` gives `p_true = NaN` (every report is a coin flip) and
//! `ε → −∞` gives `p_true = 0`; `with_probability` clamps to `[½, 1]`.
//! `next_f64_range(lo, lo)` returns `lo`; `lo > hi` draws from `(hi, lo]`;
//! an infinite endpoint yields `NaN`. `next_bool(0)` is always false,
//! `next_bool(1)` always true, `next_bool(NaN)` always false.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::det_math::{exp64, ln64};
use alice_physics::privacy::{
    LaplaceNoise, PrivacyBudget, PrivateAggregator, RandomizedResponse, Rappor, XorShift64,
    RAPPOR_BITS,
};
use alice_physics::sketch::FnvHasher;
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Test-side generator copy (independent of the library's method bodies)
// ---------------------------------------------------------------------------

/// The test's own xorshift64 step and uniform mapping.
struct RefRng(u64);

impl RefRng {
    const NONZERO_SEED: u64 = 0x853c_49e6_748f_ea9b;
    const TWO_POW_MINUS_53: f64 = 1.0 / 9_007_199_254_740_992.0;

    fn new(seed: u64) -> Self {
        Self(if seed == 0 { Self::NONZERO_SEED } else { seed })
    }

    fn step(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn uniform(&mut self) -> f64 {
        ((self.step() >> 11) as f64) * Self::TWO_POW_MINUS_53
    }

    /// `−sign(u)·b·ln(1 − 2|u|)`, `u = U − ½`.
    fn laplace(&mut self, b: f64) -> f64 {
        let u = self.uniform() - 0.5;
        let sign = if u < 0.0 { -1.0 } else { 1.0 };
        -sign * b * ln64(1.0 - 2.0 * u.abs())
    }

    /// Randomized response: truth with the first draw, else a coin.
    fn rr(&mut self, p: f64, truth: bool) -> bool {
        if self.uniform() < p {
            truth
        } else {
            self.uniform() < 0.5
        }
    }
}

const SEEDS: [u64; 6] = [1, 42, 7, 0xdead_beef, u64::MAX, 0];

fn bits(x: f64) -> u64 {
    x.to_bits()
}

// ---------------------------------------------------------------------------
// Privacy budget
// ---------------------------------------------------------------------------

/// `remaining = ε_max − Σ accepted` exactly, exhaustion at `Σ ≥ ε_max`,
/// refusal exactly when the spend would exceed, `query_count` = accepted.
#[test]
fn budget_remaining_is_max_minus_accepted_sum_exactly() {
    for &max in &[1.0, 0.75, 4.0, 0.1875, 3.0] {
        let spends = [0.5, 0.25, 1.0, 0.125, 0.0625, 0.25, 2.0, 0.03125, 0.5];
        let mut b = PrivacyBudget::new(max);
        let mut sum = 0.0f64;
        let mut accepted = 0u64;
        for &eps in &spends {
            let fits = sum + eps <= max;
            assert_eq!(b.try_spend(eps), fits, "max={max} sum={sum} eps={eps}");
            if fits {
                sum += eps;
                accepted += 1;
            }
            assert_eq!(bits(b.spent()), bits(sum), "spent max={max} sum={sum}");
            assert_eq!(
                bits(b.remaining()),
                bits(max - sum),
                "remaining max={max} sum={sum}"
            );
            assert_eq!(
                b.is_exhausted(),
                sum >= max,
                "exhausted max={max} sum={sum}"
            );
            assert_eq!(b.query_count(), accepted);
        }
        assert!(
            accepted >= 2,
            "scene must accept at least two spends (max={max})"
        );
        assert!(
            accepted < spends.len() as u64,
            "scene must refuse at least one (max={max})"
        );
    }

    // Equality is accepted: the last quarter lands exactly on ε_max.
    let mut b = PrivacyBudget::new(1.0);
    assert!(b.try_spend(0.75));
    assert!(!b.is_exhausted());
    assert!(b.try_spend(0.25));
    assert!(b.is_exhausted());
    assert_eq!(bits(b.remaining()), bits(0.0));
    assert_eq!(bits(b.spent()), bits(1.0));
    assert_eq!(b.query_count(), 2);
}

/// `reset` reproduces `PrivacyBudget::new(ε_max)` bit for bit.
#[test]
fn budget_reset_is_fresh_state_bit_for_bit() {
    for &max in &[1.0, 0.0, 2.5, -1.0] {
        let fresh = PrivacyBudget::new(max);
        let mut b = PrivacyBudget::new(max);
        for eps in [0.5, 0.25, -0.5, 0.0] {
            let _ = b.try_spend(eps);
        }
        b.reset();
        assert_eq!(bits(b.spent()), bits(fresh.spent()), "max={max}");
        assert_eq!(bits(b.remaining()), bits(fresh.remaining()), "max={max}");
        assert_eq!(b.query_count(), fresh.query_count(), "max={max}");
        assert_eq!(b.is_exhausted(), fresh.is_exhausted(), "max={max}");
        assert_eq!(bits(b.spent()), bits(0.0));
        assert_eq!(b.query_count(), 0);
        assert_eq!(bits(b.remaining()), bits(max.max(0.0)));
        assert_eq!(b.is_exhausted(), max <= 0.0);
    }
}

/// Degenerate budgets and spends: pinned contract (see module doc).
#[test]
fn budget_degenerate_inputs() {
    // ε_max = 0 is exhausted from the start; remaining 0.
    let mut z = PrivacyBudget::new(0.0);
    assert!(z.is_exhausted());
    assert_eq!(bits(z.remaining()), bits(0.0));
    assert!(!z.try_spend(0.5));
    assert_eq!(z.query_count(), 0);
    // A zero spend is accepted even when exhausted and counts as a query.
    assert!(z.try_spend(0.0));
    assert_eq!(z.query_count(), 1);
    assert_eq!(bits(z.spent()), bits(0.0));
    assert!(z.is_exhausted());

    // Zero spend on a fresh budget: accepted, spent unchanged, count + 1.
    let mut b = PrivacyBudget::new(1.0);
    assert!(b.try_spend(0.0));
    assert_eq!(b.query_count(), 1);
    assert_eq!(bits(b.spent()), bits(0.0));
    assert_eq!(bits(b.remaining()), bits(1.0));
    assert!(!b.is_exhausted());

    // Spend larger than the budget / +∞ / NaN: refused, state frozen bit for bit.
    for eps in [1.5, f64::INFINITY, f64::NAN, f64::MAX] {
        let before = (
            bits(b.spent()),
            bits(b.remaining()),
            b.query_count(),
            b.is_exhausted(),
        );
        assert!(!b.try_spend(eps), "eps={eps}");
        let after = (
            bits(b.spent()),
            bits(b.remaining()),
            b.query_count(),
            b.is_exhausted(),
        );
        assert_eq!(before, after, "eps={eps}");
    }

    // Negative spend: accepted, remaining grows past ε_max (pinned, undocumented).
    assert!(b.try_spend(-0.5));
    assert_eq!(bits(b.spent()), bits(-0.5));
    assert_eq!(bits(b.remaining()), bits(1.5));
    assert_eq!(b.query_count(), 2);
    assert!(!b.is_exhausted());

    // Negative ε_max: exhausted, remaining clamped to 0, every positive spend refused.
    let mut n = PrivacyBudget::new(-1.0);
    assert!(n.is_exhausted());
    assert_eq!(bits(n.remaining()), bits(0.0));
    assert!(!n.try_spend(0.25));
    assert_eq!(n.query_count(), 0);

    // ε_max = +∞: one f64::MAX is accepted with remaining +∞; a second one
    // overflows the sum to +∞, which reads as exhausted (∞ ≥ ∞) with
    // remaining (∞ − ∞).max(0) = 0 (pinned: overflow exhausts an infinite budget).
    let mut inf = PrivacyBudget::new(f64::INFINITY);
    assert!(inf.try_spend(f64::MAX));
    assert!(!inf.is_exhausted());
    assert_eq!(bits(inf.remaining()), bits(f64::INFINITY));
    assert!(inf.try_spend(f64::MAX));
    assert_eq!(
        bits(inf.spent()),
        bits(f64::INFINITY),
        "2·f64::MAX overflows to +∞"
    );
    assert!(inf.is_exhausted());
    assert_eq!(bits(inf.remaining()), bits(0.0));
    assert!(inf.try_spend(1.0), "∞ + 1 ≤ ∞ still accepts");
    assert_eq!(inf.query_count(), 3);

    // Many accepted spends never overflow the counter path (u64), spent exact.
    let mut many = PrivacyBudget::new(4096.0);
    for _ in 0..4096 {
        assert!(many.try_spend(1.0));
    }
    assert!(!many.try_spend(1.0));
    assert_eq!(many.query_count(), 4096);
    assert_eq!(bits(many.spent()), bits(4096.0));
    assert!(many.is_exhausted());
}

// ---------------------------------------------------------------------------
// xorshift generator
// ---------------------------------------------------------------------------

/// `next_f64_range(lo, hi)` equals `(hi − lo)·U + lo` from the seed when the
/// width is a power of two, and stays in `[lo, hi)`; `next_bool(p)` is
/// `U < p`; the same seed reproduces the sequence, different seeds differ,
/// seed 0 is remapped to the documented constant.
#[test]
fn xorshift_range_and_bool_follow_the_seed() {
    let ranges = [
        (4.0, 8.0),
        (-1.0, 1.0),
        (-0.5, 0.5),
        (0.0, 1.0),
        (1024.0, 1536.0),
    ];
    for &seed in &SEEDS {
        for &(lo, hi) in &ranges {
            let mut rng = XorShift64::new(seed);
            let mut rf = RefRng::new(seed);
            for _ in 0..512 {
                let expected = (hi - lo) * rf.uniform() + lo;
                let got = rng.next_f64_range(lo, hi);
                assert_eq!(bits(got), bits(expected), "seed={seed} range=[{lo},{hi})");
                assert!(
                    got >= lo && got < hi,
                    "seed={seed} got={got} range=[{lo},{hi})"
                );
            }
        }
        // Non-power-of-two width: one extra rounding at most (1 ulp), still in range.
        let (lo, hi) = (0.1, 0.7);
        let mut rng = XorShift64::new(seed);
        let mut rf = RefRng::new(seed);
        for _ in 0..512 {
            let expected = (hi - lo) * rf.uniform() + lo;
            let got = rng.next_f64_range(lo, hi);
            assert!(
                (got - expected).abs() <= 2.0 * f64::EPSILON,
                "seed={seed} got={got} expected={expected}"
            );
            assert!(got >= lo && got < hi);
        }
        // next_bool(p) == U < p, with p across the unit interval.
        for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut rng = XorShift64::new(seed);
            let mut rf = RefRng::new(seed);
            let mut trues = 0u32;
            for _ in 0..1024 {
                let expected = rf.uniform() < p;
                assert_eq!(rng.next_bool(p), expected, "seed={seed} p={p}");
                trues += u32::from(expected);
            }
            if p == 0.0 {
                assert_eq!(trues, 0);
            } else if p == 1.0 {
                assert_eq!(trues, 1024);
            } else {
                assert!(trues > 0 && trues < 1024, "seed={seed} p={p} trues={trues}");
            }
        }
    }

    // Determinism: same seed, same sequence; different seeds, different sequences.
    let mut a = XorShift64::new(99);
    let mut b = XorShift64::new(99);
    let mut c = XorShift64::new(100);
    let mut same = 0;
    for _ in 0..64 {
        let va = a.next_f64_range(-3.0, 5.0);
        assert_eq!(bits(va), bits(b.next_f64_range(-3.0, 5.0)));
        same += usize::from(bits(va) == bits(c.next_f64_range(-3.0, 5.0)));
    }
    assert_eq!(same, 0, "seeds 99 and 100 must not collide in 64 draws");

    // Seed 0 is remapped to the non-zero constant.
    let mut zero = XorShift64::new(0);
    let mut remapped = XorShift64::new(RefRng::NONZERO_SEED);
    for _ in 0..16 {
        assert_eq!(
            bits(zero.next_f64_range(0.0, 1.0)),
            bits(remapped.next_f64_range(0.0, 1.0))
        );
    }
}

/// Degenerate ranges and probabilities.
#[test]
fn xorshift_degenerate_inputs() {
    let mut rng = XorShift64::new(5);
    // lo == hi: always lo (0·U + lo), including a non-zero lo.
    for lo in [0.0, 2.5, -7.0] {
        for _ in 0..8 {
            assert_eq!(bits(rng.next_f64_range(lo, lo)), bits(lo));
        }
    }
    // lo > hi: the draw lives in (hi, lo] (width negative, closed at lo).
    let mut rf = RefRng::new(5);
    let mut rng = XorShift64::new(5);
    for _ in 0..256 {
        let u = rf.uniform();
        let got = rng.next_f64_range(8.0, 4.0);
        assert_eq!(bits(got), bits(-4.0 * u + 8.0));
        assert!(got > 4.0 && got <= 8.0, "got={got}");
    }
    // Infinite endpoint: NaN (∞·U + (−∞) or ∞·0).
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut rng = XorShift64::new(5);
        (0..8)
            .map(|_| rng.next_f64_range(f64::NEG_INFINITY, 1.0))
            .collect::<Vec<_>>()
    }));
    let v = r.expect("infinite endpoint must not panic");
    assert!(v.iter().all(|x| x.is_nan()), "{v:?}");
    // Extreme finite width: exact power-of-two width 2^1023 keeps the closed form.
    let mut rf = RefRng::new(11);
    let mut rng = XorShift64::new(11);
    let hi = f64::from_bits(0x7FE0_0000_0000_0000); // 2^1023
    for _ in 0..64 {
        let expected = hi * rf.uniform();
        assert_eq!(bits(rng.next_f64_range(0.0, hi)), bits(expected));
    }
    // next_bool: p = 0 never, p = 1 always, p > 1 always, p < 0 never, NaN never.
    let mut rng = XorShift64::new(9);
    for _ in 0..256 {
        assert!(!rng.next_bool(0.0));
        assert!(rng.next_bool(1.0));
        assert!(rng.next_bool(2.0));
        assert!(!rng.next_bool(-1.0));
        assert!(!rng.next_bool(f64::NAN));
    }
}

// ---------------------------------------------------------------------------
// Laplace mechanism
// ---------------------------------------------------------------------------

/// `privatize(v) = v + (−sign(u)·b·ln(1 − 2|u|))` and
/// `privatize_int(v) = round(v + sample)`, predicted from the seed.
#[test]
fn laplace_privatize_matches_inverse_transform_from_seed() {
    let scenes = [
        (1.0, 1.0),
        (1.0, 2.0),
        (3.0, 0.5),
        (0.25, 4.0),
        (2.0, 0.125),
    ];
    for &seed in &SEEDS {
        for &(sens, eps) in &scenes {
            let b = sens / eps;
            let mut lap = LaplaceNoise::with_seed(sens, eps, seed);
            let mut rf = RefRng::new(seed);
            let mut nonzero = 0u32;
            for i in 0..256 {
                let v = f64::from(i) * 0.5 - 10.0;
                let expected = v + rf.laplace(b);
                let got = lap.privatize(v);
                assert_eq!(bits(got), bits(expected), "seed={seed} b={b} v={v}");
                nonzero += u32::from(got != v);
            }
            assert_eq!(
                nonzero, 256,
                "noise must actually perturb (seed={seed} b={b})"
            );
            for i in 0..256i64 {
                let v = i * 7 - 900;
                let expected = (v as f64 + rf.laplace(b)).round() as i64;
                assert_eq!(lap.privatize_int(v), expected, "seed={seed} b={b} v={v}");
            }
        }
    }
    // Scale is Δf/ε and the magnitude of the noise is linear in it: with the
    // same seed the samples of scale 2b are exactly twice those of scale b.
    let mut one = LaplaceNoise::with_seed(1.0, 1.0, 3);
    let mut two = LaplaceNoise::with_seed(2.0, 1.0, 3);
    let mut half = LaplaceNoise::with_seed(1.0, 2.0, 3);
    for _ in 0..64 {
        let s = one.privatize(0.0);
        assert_eq!(bits(two.privatize(0.0)), bits(2.0 * s));
        assert_eq!(bits(half.privatize(0.0)), bits(0.5 * s));
    }
}

/// `ε = 0` (`b = +∞`), `Δf = 0` (identity), negative `ε` (mirrored noise),
/// `i64` saturation of `privatize_int`.
#[test]
fn laplace_degenerate_inputs() {
    // ε = 0 → b = +∞ → every sample is ±∞ with the sign of u; privatize_int saturates.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut lap = LaplaceNoise::with_seed(1.0, 0.0, 21);
        let mut rf = RefRng::new(21);
        for _ in 0..64 {
            let u = rf.uniform() - 0.5;
            let got = lap.privatize(5.0);
            assert!(got.is_infinite(), "got={got}");
            assert_eq!(got > 0.0, u >= 0.0, "sign follows u (u={u})");
        }
        let mut lap = LaplaceNoise::with_seed(1.0, 0.0, 21);
        let mut rf = RefRng::new(21);
        for _ in 0..64 {
            let u = rf.uniform() - 0.5;
            let expected = if u >= 0.0 { i64::MAX } else { i64::MIN };
            assert_eq!(lap.privatize_int(5), expected, "u={u}");
        }
    }));
    assert!(r.is_ok(), "ε = 0 must not panic (b = +∞ propagates)");

    // Δf = 0 → b = 0 → noise is ±0 → privatize is the identity bit for bit.
    let mut lap = LaplaceNoise::with_seed(0.0, 1.0, 21);
    for i in 0..64i32 {
        let v = f64::from(i) * 1.25 - 20.0;
        assert_eq!(bits(lap.privatize(v)), bits(v + 0.0), "v={v}");
        let w = i64::from(i) * 3 - 50;
        assert_eq!(lap.privatize_int(w), w);
    }

    // Negative ε → b < 0 → every sample is the exact negation of the +ε sample.
    let mut pos = LaplaceNoise::with_seed(1.0, 1.0, 8);
    let mut neg = LaplaceNoise::with_seed(1.0, -1.0, 8);
    for _ in 0..64 {
        assert_eq!(bits(neg.privatize(0.0)), bits(-pos.privatize(0.0)));
    }

    // Extreme value: privatize_int near i64::MAX saturates instead of wrapping.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut lap = LaplaceNoise::with_seed(1.0, 1.0, 13);
        let mut rf = RefRng::new(13);
        for _ in 0..64 {
            let expected = (i64::MAX as f64 + rf.laplace(1.0)).round() as i64;
            let got = lap.privatize_int(i64::MAX);
            assert_eq!(got, expected);
            assert!(got >= i64::MAX - 2048, "no wrap-around: got={got}");
        }
    }));
    assert!(r.is_ok(), "i64::MAX must not panic");

    // Extreme ε: Δf = 1, ε = f64::MIN_POSITIVE → b finite; ε = 1e-300 with Δf = 1e10 → b = +∞.
    let finite = LaplaceNoise::with_seed(1.0, f64::MIN_POSITIVE, 1);
    assert!(finite.scale().is_finite());
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut lap = LaplaceNoise::with_seed(1e10, 1e-300, 1);
        assert!(lap.scale().is_infinite());
        lap.privatize(0.0)
    }));
    assert!(r.expect("overflowing scale must not panic").is_infinite());
}

// ---------------------------------------------------------------------------
// Aggregator
// ---------------------------------------------------------------------------

/// `estimate_sum = Σ`, `estimate_mean = Σ/n`, `standard_error = b√2/√n`
/// (exact at n ∈ {1, 2, 8} for a power-of-two scale), reset = fresh state.
#[test]
fn aggregator_closed_forms_and_reset() {
    for &scale in &[1.0, 0.5, 2.0, 0.0625] {
        let mut agg = PrivateAggregator::new(scale);
        let inputs = [1.5, -2.25, 4.0, 0.125, 100.0, -0.5, 8.0, 0.75];
        let mut sum = 0.0;
        for (i, &v) in inputs.iter().enumerate() {
            agg.add(v);
            sum += v;
            let n = (i + 1) as u64;
            assert_eq!(bits(agg.estimate_sum()), bits(sum), "scale={scale} n={n}");
            assert_eq!(
                bits(agg.estimate_mean()),
                bits(sum / n as f64),
                "scale={scale} n={n}"
            );
            assert_eq!(agg.count(), n);
            let se = agg.standard_error();
            match n {
                1 => assert_eq!(bits(se), bits(scale * core::f64::consts::SQRT_2)),
                2 => assert_eq!(bits(se), bits(scale)),
                8 => assert_eq!(bits(se), bits(scale / 2.0)),
                _ => {
                    // b·√2/√n with the correctly rounded √n: at most 2 roundings.
                    let expected = scale * core::f64::consts::SQRT_2 / (n as f64).sqrt();
                    assert!(
                        (se - expected).abs() <= 2.0 * f64::EPSILON * expected,
                        "n={n}"
                    );
                }
            }
        }
        assert_eq!(bits(agg.estimate_sum()), bits(111.625));
        assert_eq!(bits(agg.estimate_mean()), bits(111.625 / 8.0));

        // SE depends on n and scale only: a shifted sample has the same SE.
        let mut shifted = PrivateAggregator::new(scale);
        for v in [1e6, -1e6, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
            shifted.add(v);
        }
        assert_eq!(bits(shifted.standard_error()), bits(agg.standard_error()));

        // reset reproduces new(scale) bit for bit.
        let fresh = PrivateAggregator::new(scale);
        agg.reset();
        assert_eq!(bits(agg.estimate_sum()), bits(fresh.estimate_sum()));
        assert_eq!(bits(agg.estimate_mean()), bits(fresh.estimate_mean()));
        assert_eq!(bits(agg.standard_error()), bits(fresh.standard_error()));
        assert_eq!(agg.count(), fresh.count());
        assert_eq!(bits(agg.estimate_sum()), bits(0.0));
        assert_eq!(agg.count(), 0);
    }
}

/// `n = 0`: sum 0, mean 0 (pinned), SE +∞; overflow and NaN propagate.
#[test]
fn aggregator_degenerate_inputs() {
    let empty = PrivateAggregator::new(1.0);
    assert_eq!(bits(empty.estimate_sum()), bits(0.0));
    assert_eq!(
        bits(empty.estimate_mean()),
        bits(0.0),
        "n = 0 → mean is 0 by early return"
    );
    assert_eq!(bits(empty.standard_error()), bits(f64::INFINITY));
    assert_eq!(empty.count(), 0);

    // Scale 0: SE is exactly 0 for every n ≥ 1, +∞ at n = 0.
    let mut zero = PrivateAggregator::new(0.0);
    assert!(zero.standard_error().is_infinite());
    for _ in 0..5 {
        zero.add(1.0);
        assert_eq!(bits(zero.standard_error()), bits(0.0));
    }

    // Overflow: two f64::MAX reports sum to +∞, mean +∞, count 2.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut agg = PrivateAggregator::new(1.0);
        agg.add(f64::MAX);
        agg.add(f64::MAX);
        (agg.estimate_sum(), agg.estimate_mean(), agg.count())
    }));
    let (s, m, c) = r.expect("overflowing sum must not panic");
    assert_eq!(bits(s), bits(f64::INFINITY));
    assert_eq!(bits(m), bits(f64::INFINITY));
    assert_eq!(c, 2);

    // NaN report poisons sum and mean but not SE (which depends on n only).
    let mut agg = PrivateAggregator::new(0.5);
    agg.add(1.0);
    agg.add(f64::NAN);
    assert!(agg.estimate_sum().is_nan());
    assert!(agg.estimate_mean().is_nan());
    assert_eq!(bits(agg.standard_error()), bits(0.5));
}

// ---------------------------------------------------------------------------
// Randomized response
// ---------------------------------------------------------------------------

/// `p_true = e^ε/(e^ε + 1)` (exactly ½ at ε = 0), `with_probability` clamps
/// to `[½, 1]`.
#[test]
fn randomized_response_p_true_closed_form() {
    for &eps in &[0.0, 0.5, 1.0, 2.0, 1.098_612_288_668_11, 10.0, -0.5] {
        let e = exp64(eps);
        let expected = e / (e + 1.0);
        let rr = RandomizedResponse::new(eps);
        assert_eq!(bits(rr.p_true()), bits(expected), "eps={eps}");
    }
    assert_eq!(
        bits(RandomizedResponse::new(0.0).p_true()),
        bits(0.5),
        "ε = 0 → exactly ½"
    );
    // ε = ln 3 → p = 3/4 to 13 ulp of the deterministic exp (e^ε ≈ 3).
    let p = RandomizedResponse::new(1.098_612_288_668_109_7).p_true();
    assert!((p - 0.75).abs() < 1e-14, "p={p}");
    // p = 1 only in the limit: ε = 40 gives p within 1e-17 of 1 but ≤ 1.
    let near_one = RandomizedResponse::new(40.0).p_true();
    assert!(near_one <= 1.0 && near_one > 1.0 - 1e-16, "p={near_one}");

    for &(p, expected) in &[
        (0.75, 0.75),
        (0.5, 0.5),
        (1.0, 1.0),
        (0.3, 0.5),
        (1.5, 1.0),
        (-2.0, 0.5),
        (0.9, 0.9),
    ] {
        let rr = RandomizedResponse::with_probability(p, 1);
        assert_eq!(bits(rr.p_true()), bits(expected), "with_probability({p})");
    }
}

/// Every report is predicted from the seed by the two-draw rule; `p = 1`
/// is the identity; `privatize_bit` agrees with `privatize` on the same
/// draws; a `p < 1` scene must actually flip something.
#[test]
fn randomized_response_reports_follow_the_seed() {
    for &seed in &SEEDS {
        for &p in &[0.5, 0.75, 0.9, 1.0] {
            let mut rr = RandomizedResponse::with_probability(p, seed);
            let mut rf = RefRng::new(seed);
            let mut flipped = 0u32;
            for i in 0..512u32 {
                let truth = (i * 7 + 3) % 5 < 2;
                let expected = rf.rr(p, truth);
                assert_eq!(rr.privatize(truth), expected, "seed={seed} p={p} i={i}");
                flipped += u32::from(expected != truth);
            }
            if p == 1.0 {
                assert_eq!(flipped, 0, "p = 1 must be the identity (seed={seed})");
            } else {
                assert!(
                    flipped > 0,
                    "p={p} must flip something in 512 reports (seed={seed})"
                );
            }
            // privatize_bit: same rule on the integer encoding, same stream.
            let mut rr_bits = RandomizedResponse::with_probability(p, seed);
            let mut rf_bits = RefRng::new(seed);
            for i in 0..512u32 {
                let bit = u8::from((i * 3 + 1) % 4 == 0);
                let expected = u8::from(rf_bits.rr(p, bit != 0));
                assert_eq!(
                    rr_bits.privatize_bit(bit),
                    expected,
                    "seed={seed} p={p} i={i}"
                );
            }
            // Non-binary input is "non-zero → true": 2 and 255 report like 1.
            let mut a = RandomizedResponse::with_probability(p, seed);
            let mut b = RandomizedResponse::with_probability(p, seed);
            for _ in 0..64 {
                assert_eq!(a.privatize_bit(2), b.privatize_bit(1));
            }
            let mut a = RandomizedResponse::with_probability(p, seed);
            let mut b = RandomizedResponse::with_probability(p, seed);
            for _ in 0..64 {
                assert_eq!(a.privatize_bit(255), b.privatize_bit(1));
            }
        }
    }
}

/// `estimate_proportion(p, n, k) = (k/n − (1 − p)/2)/p` clamped to `[0, 1]`:
/// exact on dyadic `k/n`, unbiased (feeding `p·t + (1 − p)/2` returns `t`).
#[test]
fn estimate_proportion_closed_form() {
    // p = 3/4, n = 8, k = 5: (5/8 − 1/8) / (3/4) = 2/3.
    let e = RandomizedResponse::estimate_proportion(0.75, 8, 5);
    assert!((e - 2.0 / 3.0).abs() <= f64::EPSILON, "e={e}");
    // Unbiasedness on exact points: obs = p·t + (1 − p)/2.
    for &(p, t) in &[
        (0.75, 0.5),
        (0.75, 1.0),
        (0.5, 0.5),
        (1.0, 0.375),
        (0.5, 0.0),
        (0.75, 0.0),
    ] {
        let obs: f64 = p * t + (1.0 - p) / 2.0;
        let k = (obs * 8.0).round() as u64;
        assert_eq!(k as f64 / 8.0, obs, "scene must be dyadic (p={p} t={t})");
        let e = RandomizedResponse::estimate_proportion(p, 8, k);
        assert_eq!(bits(e), bits(t), "p={p} t={t} k={k}");
    }
    // p = 1: the estimate is the observed rate itself, bit for bit.
    for k in 0..=8u64 {
        assert_eq!(
            bits(RandomizedResponse::estimate_proportion(1.0, 8, k)),
            bits(k as f64 / 8.0)
        );
    }
    // p = ½ is not singular here: (k/n − ¼)/½ = 2k/n − ½.
    for k in 2..=6u64 {
        let expected = 2.0 * (k as f64 / 8.0) - 0.5;
        assert_eq!(
            bits(RandomizedResponse::estimate_proportion(0.5, 8, k)),
            bits(expected),
            "k={k}"
        );
    }
    // Clamping: k = 0 → raw −(1 − p)/(2p) < 0 → 0; k = n → raw (1 + p)/(2p) > 1 → 1.
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.75, 8, 0)),
        bits(0.0)
    );
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.75, 8, 8)),
        bits(1.0)
    );
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.5, 8, 1)),
        bits(0.0)
    );
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.5, 8, 7)),
        bits(1.0)
    );
    // Invariance: scaling n and k together does not change the estimate.
    for m in [1u64, 2, 4, 16, 1024] {
        assert_eq!(
            bits(RandomizedResponse::estimate_proportion(0.75, 8 * m, 5 * m)),
            bits(RandomizedResponse::estimate_proportion(0.75, 8, 5)),
            "m={m}"
        );
    }
    // Monotone in k for fixed p, n.
    let mut prev = -1.0;
    for k in 0..=16u64 {
        let e = RandomizedResponse::estimate_proportion(0.9, 16, k);
        assert!(e >= prev, "k={k} e={e} prev={prev}");
        prev = e;
    }
}

/// `n = 0` → 0, `k > n` → 1, `p = 0` → NaN or 1 (division by zero, pinned),
/// `ε` extremes in `new` (no clamp: NaN / 0), `p = NaN` reports are coin flips.
#[test]
fn randomized_response_degenerate_inputs() {
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.75, 0, 0)),
        bits(0.0)
    );
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.75, 0, 5)),
        bits(0.0),
        "n = 0 wins over k"
    );
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(0.75, 8, 20)),
        bits(1.0),
        "k > n clamps to 1"
    );
    // p = 0: (k/n − ½)/0 → NaN at k/n = ½ (clamp keeps NaN), +∞ → 1 above, −∞ → 0 below.
    let r = catch_unwind(AssertUnwindSafe(|| {
        (
            RandomizedResponse::estimate_proportion(0.0, 8, 4),
            RandomizedResponse::estimate_proportion(0.0, 8, 8),
            RandomizedResponse::estimate_proportion(0.0, 8, 0),
        )
    }));
    let (nan, one, zero) = r.expect("p = 0 must not panic");
    assert!(
        nan.is_nan(),
        "p = 0, obs = ½ → 0/0 = NaN (pinned, undocumented)"
    );
    assert_eq!(bits(one), bits(1.0));
    assert_eq!(bits(zero), bits(0.0));
    // p = NaN → NaN; n = u64::MAX, k = u64::MAX → observed 1 → 1.
    assert!(RandomizedResponse::estimate_proportion(f64::NAN, 8, 4).is_nan());
    assert_eq!(
        bits(RandomizedResponse::estimate_proportion(
            0.75,
            u64::MAX,
            u64::MAX
        )),
        bits(1.0)
    );

    // new(ε): no clamp. ε = +1000 → e^ε = ∞ → ∞/∞ = NaN; ε = −1000 → 0/(1) = 0; ε = NaN → NaN.
    let r = catch_unwind(AssertUnwindSafe(|| {
        (
            RandomizedResponse::new(1000.0).p_true(),
            RandomizedResponse::new(-1000.0).p_true(),
            RandomizedResponse::new(f64::NAN).p_true(),
            RandomizedResponse::new(f64::INFINITY).p_true(),
        )
    }));
    let (big, small, nan, inf) = r.expect("extreme ε must not panic");
    assert!(
        big.is_nan(),
        "ε = 1000 → p_true = NaN (pinned, undocumented): {big}"
    );
    assert_eq!(
        bits(small),
        bits(0.0),
        "ε = −1000 → p_true = 0 (below ½, not clamped)"
    );
    assert!(nan.is_nan());
    assert!(inf.is_nan());
    // With p_true = NaN every report is the second draw (a coin): `new()` is
    // entropy-seeded, so the seeded path with `with_probability(NaN)` is used
    // (NaN survives `f64::clamp`, pinned) and predicted from the seed.
    let mut rr = RandomizedResponse::with_probability(f64::NAN, 4);
    assert!(
        rr.p_true().is_nan(),
        "NaN survives clamp (pinned, undocumented)"
    );
    let mut rf = RefRng::new(4);
    let mut coin = XorShift64::new(4);
    for i in 0..128u32 {
        let truth = i % 3 == 0;
        let got = rr.privatize(truth);
        assert_eq!(got, rf.rr(f64::NAN, truth));
        // The first draw is consumed and ignored, the second is the coin.
        let _ = coin.next_bool(0.5);
        assert_eq!(got, coin.next_bool(0.5));
    }
}

// ---------------------------------------------------------------------------
// RAPPOR
// ---------------------------------------------------------------------------

/// The test's own Bloom encoding: three FNV positions.
fn bloom_of(value: u64) -> [u8; RAPPOR_BITS] {
    let mut bloom = [0u8; RAPPOR_BITS];
    for i in 0..3u128 {
        let h = FnvHasher::hash_u128(u128::from(value) | (i << 64));
        bloom[(h as usize) % RAPPOR_BITS] = 1;
    }
    bloom
}

/// Identity / complement / all-ones / all-zeros parameter corners are
/// independent of the generator; `params` returns the clamped triple;
/// `BITS == RAPPOR_BITS == 64 == report length`.
#[test]
fn rappor_parameter_corners_are_closed_forms() {
    assert_eq!(Rappor::BITS, RAPPOR_BITS);
    assert_eq!(RAPPOR_BITS, 64);
    let values = [0u64, 1, 12345, u64::MAX, 0xdead_beef_cafe_f00d];
    for &v in &values {
        let bloom = bloom_of(v);
        let ones = bloom.iter().filter(|&&b| b == 1).count();
        assert!((1..=3).contains(&ones), "v={v} ones={ones}");

        let mut identity = Rappor::new(0.0, 1.0, 0.0);
        for _ in 0..4 {
            let r = identity.privatize(v);
            assert_eq!(r.len(), Rappor::BITS);
            assert_eq!(r, bloom, "f=0 p=1 q=0 is the identity (v={v})");
        }
        let mut complement = Rappor::new(0.0, 0.0, 1.0);
        let expected: Vec<u8> = bloom.iter().map(|&b| 1 - b).collect();
        assert_eq!(
            complement.privatize(v).to_vec(),
            expected,
            "f=0 p=0 q=1 is the complement (v={v})"
        );
        let mut ones_only = Rappor::new(0.5, 1.0, 1.0);
        assert_eq!(ones_only.privatize(v), [1u8; RAPPOR_BITS]);
        let mut zeros_only = Rappor::new(0.5, 0.0, 0.0);
        assert_eq!(zeros_only.privatize(v), [0u8; RAPPOR_BITS]);
    }
    // Different values with different Bloom filters give different identity reports.
    let mut identity = Rappor::new(0.0, 1.0, 0.0);
    let a = identity.privatize(1);
    let b = identity.privatize(2);
    assert_eq!(a, bloom_of(1));
    assert_eq!(b, bloom_of(2));
    assert_ne!(a, b);

    // params: default triple, clamped triple.
    assert_eq!(Rappor::default_params().params(), (0.5, 0.75, 0.25));
    assert_eq!(
        Rappor::new(0.9, 1.5, -1.0).params(),
        (0.5, 1.0, 0.0),
        "f→[0,½], p,q→[0,1]"
    );
    assert_eq!(Rappor::new(-0.1, 0.3, 0.6).params(), (0.0, 0.3, 0.6));
    let r = catch_unwind(AssertUnwindSafe(|| {
        Rappor::new(f64::NAN, f64::INFINITY, f64::NEG_INFINITY).params()
    }));
    let (f, p, q) = r.expect("non-finite params must not panic");
    assert!(f.is_nan(), "NaN survives f64::clamp (pinned)");
    assert_eq!(bits(p), bits(1.0));
    assert_eq!(bits(q), bits(0.0));
}

/// The permanent stage with `f = ½` under the identity instantaneous stage
/// (`p = 1, q = 0`): a Bloom-zero bit reports 1 with probability
/// `f/2 = ¼`, a Bloom-one bit reports 0 with probability `¼`. The generator
/// is entropy-seeded (no seeded constructor), so this is a 10σ bound on
/// 400 reports × 64 bits rather than a bit-exact prediction.
#[test]
fn rappor_permanent_stage_flips_at_rate_f_over_two() {
    let v = 12345u64;
    let bloom = bloom_of(v);
    let ones_in_bloom = bloom.iter().filter(|&&b| b == 1).count();
    let zeros_in_bloom = RAPPOR_BITS - ones_in_bloom;
    let mut rappor = Rappor::new(0.5, 1.0, 0.0);
    let reports = 400usize;
    let (mut zero_to_one, mut one_to_zero) = (0usize, 0usize);
    for _ in 0..reports {
        let r = rappor.privatize(v);
        for i in 0..RAPPOR_BITS {
            match (bloom[i], r[i]) {
                (0, 1) => zero_to_one += 1,
                (1, 0) => one_to_zero += 1,
                _ => {}
            }
        }
    }
    let n0 = (reports * zeros_in_bloom) as f64;
    let n1 = (reports * ones_in_bloom) as f64;
    let (mean0, sd0) = (0.25 * n0, (0.25 * 0.75 * n0).sqrt());
    let (mean1, sd1) = (0.25 * n1, (0.25 * 0.75 * n1).sqrt());
    assert!(
        (zero_to_one as f64 - mean0).abs() <= 10.0 * sd0,
        "0→1 flips {zero_to_one} vs {mean0} ± 10·{sd0}"
    );
    assert!(
        (one_to_zero as f64 - mean1).abs() <= 10.0 * sd1,
        "1→0 flips {one_to_zero} vs {mean1} ± 10·{sd1}"
    );
    assert!(
        zero_to_one > 0 && one_to_zero > 0,
        "f = ½ must flip in both directions"
    );
}
