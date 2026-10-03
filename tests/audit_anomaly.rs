//! Audit oracles for `anomaly`.
//!
//! Expected values come from brute force on the raw sample list (sort the
//! last `min(n, 100)` samples and take the median / MAD / mean / variance with
//! plain textbook formulas) or from closed forms; none from the detectors:
//!
//! ```text
//! median(W)   = middle of sorted(W)   (mean of the two middle values when |W| even)
//! MAD(W)      = median(|w - median(W)|)
//! anomalous   <=>  |x - median| > k * MAD * 1.4826           (strict)
//! score       =    |x - median| / (MAD * 1.4826)
//! Welford     :    mean = sum/n,  var = sum (x - mean)^2 / (n - 1)
//! EWMA        :    d = x - mu;  mu += a d;  v = (1 - a)(v + a d^2)
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::float_cmp, clippy::disallowed_methods)]

use alice_physics::anomaly::{
    AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian,
    ZScoreDetector,
};

/// Deterministic xorshift64* stream in [0, 1).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (r >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn median_of(w: &[f64]) -> f64 {
    let mut s = w.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let n = s.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}
fn mad_of(w: &[f64]) -> f64 {
    let m = median_of(w);
    let d: Vec<f64> = w.iter().map(|x| (x - m).abs()).collect();
    median_of(&d)
}
fn last100(v: &[f64]) -> &[f64] {
    &v[v.len().saturating_sub(100)..]
}
fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
}

/// Sample streams with heavy duplication (small integer grid), a monotone
/// ramp, a sawtooth and a constant run: the cases that stress removal of an
/// equal value from the sorted array.
fn streams() -> Vec<Vec<f64>> {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let dup: Vec<f64> = (0..350).map(|_| (r.next() * 7.0).floor() - 3.0).collect();
    let cont: Vec<f64> = (0..350).map(|_| r.next() * 200.0 - 100.0).collect();
    let ramp: Vec<f64> = (0..350).map(|i| i as f64 * 0.5).collect();
    let down: Vec<f64> = (0..350).map(|i| 1000.0 - i as f64).collect();
    let saw: Vec<f64> = (0..350).map(|i| (i % 17) as f64).collect();
    let flat: Vec<f64> = vec![4.25; 250];
    // MAD ~ 5e-9, above the 1e-10 degenerate cut-off but far below any absolute scale
    let tiny: Vec<f64> = (0..350).map(|_| 5.0 + r.next() * 2e-8).collect();
    vec![dup, cont, ramp, down, saw, flat, tiny]
}

/// `StreamingMedian` equals the brute-force window median after EVERY push, over
/// the fill phase, the wrap at 100 and 2+ further wraps.
#[test]
fn streaming_median_equals_brute_force_after_every_push() {
    for (si, s) in streams().iter().enumerate() {
        let mut m = StreamingMedian::new();
        for (i, &x) in s.iter().enumerate() {
            m.push(x);
            let want = median_of(last100(&s[..=i]));
            assert_eq!(m.median(), want, "stream {si} after {} pushes", i + 1);
            assert_eq!(m.count(), (i + 1).min(100));
            assert_eq!(m.is_full(), i + 1 >= 100);
        }
    }
}

#[test]
fn streaming_median_empty_clear_and_constants() {
    let mut m = StreamingMedian::default();
    assert_eq!(m.median(), 0.0);
    assert_eq!(StreamingMedian::WINDOW, 100);
    for i in 0..150 {
        m.push(i as f64);
    }
    m.clear();
    assert_eq!((m.count(), m.median(), m.is_full()), (0, 0.0, false));
    m.push(7.0);
    assert_eq!(m.median(), 7.0);
}

/// AUD-A-S4W1-012 (known defect): non-finite input corrupts the sorted array for
/// good. A NaN is inserted at rank 0 (every `<` against NaN is false); when it
/// is evicted `binary_search_remove(NaN)` matches the first probed slot (both
/// `<` and `>` are false) and deletes an arbitrary middle element instead, so
/// a NaN stays in the sorted array although it has left the window. After 100
/// further finite samples the window holds only finite values and the median
/// must be their median.
#[test]
#[ignore = "known defect: AUD-A-S4W1-012: one NaN push permanently corrupts StreamingMedian (median wrong after the NaN left the window)"]
fn streaming_median_recovers_after_a_nan_leaves_the_window() {
    let mut r = Rng(12345);
    let mut m = StreamingMedian::new();
    let mut all: Vec<f64> = Vec::new();
    for _ in 0..100 {
        let x = r.next() * 10.0;
        all.push(x);
        m.push(x);
    }
    m.push(f64::NAN);
    for _ in 0..250 {
        let x = r.next() * 10.0;
        all.push(x);
        m.push(x);
    }
    let want = median_of(&all[all.len() - 100..]);
    assert_eq!(m.median(), want);
}

/// MAD detector: median, MAD, verdict and score equal brute force at many
/// checkpoints, including after the window wrapped.
#[test]
fn mad_matches_brute_force_median_mad_verdict_and_score() {
    for (si, s) in streams().iter().enumerate() {
        let mut d = MadDetector::new(3.0);
        for (i, &x) in s.iter().enumerate() {
            d.observe(x);
            if i % 13 != 5 && i + 1 != s.len() {
                continue;
            }
            let w = last100(&s[..=i]);
            let (med, mad) = (median_of(w), mad_of(w));
            assert_eq!(d.median(), med, "stream {si} n={}", i + 1);
            assert_eq!(d.mad(), mad, "stream {si} n={}", i + 1);
            let edge = 3.0 * mad * 1.4826;
            for probe in [
                med,
                med + 0.3,
                med - 2.5,
                med + 40.0,
                -1000.0,
                1000.0,
                med + 0.5 * edge,
                med - 0.5 * edge,
                med + 1.5 * edge,
                med - 1.5 * edge,
            ] {
                let dev = (probe - med).abs();
                if d.count() < 3 {
                    assert!(!d.is_anomaly(probe));
                    assert_eq!(d.anomaly_score(probe), 0.0);
                } else if mad < 1e-10 {
                    assert_eq!(d.is_anomaly(probe), dev > 1e-10);
                    assert_eq!(
                        d.anomaly_score(probe),
                        if dev > 1e-10 { f64::INFINITY } else { 0.0 }
                    );
                } else {
                    assert_eq!(
                        d.is_anomaly(probe),
                        dev > 3.0 * mad * 1.4826,
                        "{si} {probe}"
                    );
                    assert!(close(d.anomaly_score(probe), dev / (mad * 1.4826), 1e-12));
                }
            }
        }
        assert_eq!(d.count(), s.len().min(100));
    }
}

/// The boundary is strict: a point at exactly k*MAD*1.4826 is not flagged, the
/// next representable value beyond it is. (Window 1..=9: median 5, MAD 2.)
#[test]
fn mad_threshold_is_strict_at_the_boundary() {
    let mut d = MadDetector::new(3.0);
    for v in 1..=9 {
        d.observe(f64::from(v));
    }
    let edge = 3.0 * 2.0 * 1.4826;
    assert!(!d.is_anomaly(5.0 + edge * (1.0 - 1e-9)));
    assert!(d.is_anomaly(5.0 + edge * (1.0 + 1e-9)));
    assert!(!d.is_anomaly(5.0 - edge * (1.0 - 1e-9)));
    assert!(d.is_anomaly(5.0 - edge * (1.0 + 1e-9)));
    d.set_threshold_k(2.0);
    assert_eq!(d.threshold_k(), 2.0);
    assert!(d.is_anomaly(5.0 + 2.0 * 2.0 * 1.4826 * 1.001));
    assert!(!d.is_anomaly(5.0 + 2.0 * 2.0 * 1.4826 * 0.999));
}

/// The cached median / MAD never go stale: interleaved observe / query matches
/// brute force every time (a stale cache would repeat the previous value).
#[test]
fn mad_cache_follows_every_observation() {
    let mut d = MadDetector::new(3.0);
    let mut seen = Vec::new();
    let mut r = Rng(777);
    for _ in 0..140 {
        let x = r.next() * 50.0;
        d.observe(x);
        seen.push(x);
        let w = last100(&seen);
        assert_eq!(d.median(), median_of(w));
        assert_eq!(d.mad(), mad_of(w));
    }
    d.clear();
    assert_eq!((d.count(), d.median(), d.mad()), (0, 0.0, 0.0));
    assert!(!d.is_anomaly(1e9));
    // refill after clear: statistics are those of the new samples only
    let fresh = [3.0, 9.0, 4.0, 4.5, 20.0, 3.5, 6.0];
    for &x in &fresh {
        d.observe(x);
    }
    assert_eq!(d.median(), median_of(&fresh));
    assert_eq!(d.mad(), mad_of(&fresh));
}

/// AUD-A-S4W1-013 (design, scoped): when more than half of the window equals
/// the median the MAD is 0 and every value that differs by more than 1e-10 is
/// declared anomalous with score +inf, including values that make up a quarter
/// of the window itself. Window: 60 x 10.0 plus 40 samples at 10.5.
#[test]
#[ignore = "known defect: AUD-A-S4W1-013: MAD = 0 branch flags 10.5 (40 of 100 window samples) as anomalous with score +inf"]
fn values_that_fill_the_window_are_not_anomalous_when_mad_is_zero() {
    let mut d = MadDetector::new(3.0);
    for i in 0..100 {
        d.observe(if i % 5 < 3 { 10.0 } else { 10.5 });
    }
    assert_eq!(d.mad(), 0.0);
    assert!(!d.is_anomaly(10.5), "score = {}", d.anomaly_score(10.5));
}

/// EWMA against a direct double sum (not the recursion): for a step input
/// x0, c, c, c ... the closed form is mu_n = c - (c - x0)(1-a)^n and
/// v_n = sum_{j=1}^n (1-a)^(n-j+1) a d_j^2 with d_j = (c - x0)(1-a)^(j-1).
#[test]
fn ewma_step_response_matches_closed_form() {
    for a in [0.05_f64, 0.2, 0.5, 0.9] {
        let (x0, c) = (2.0_f64, 12.0_f64);
        let mut e = EwmaDetector::new(a, 3.0);
        e.observe(x0);
        for n in 1..=40_i32 {
            e.observe(c);
            let mu = c - (c - x0) * (1.0 - a).powi(n);
            let mut v = 0.0;
            for j in 1..=n {
                let dj = (c - x0) * (1.0 - a).powi(j - 1);
                v += (1.0 - a).powi(n - j + 1) * a * dj * dj;
            }
            assert!(
                close(e.ewma(), mu, 1e-12),
                "a={a} n={n}: {} vs {mu}",
                e.ewma()
            );
            assert!(close(e.std_dev(), v.sqrt(), 1e-10), "a={a} n={n}");
        }
        assert_eq!(e.count(), 41);
        assert_eq!(e.alpha(), a);
    }
}

/// For i.i.d. noise the EW variance converges to 2(1-a)/(2-a) sigma^2 (the
/// recursion measures deviations from the moving mean). Pinned here so a
/// change of the estimator is loud: a = 0.05 -> 0.9744 sigma^2, a = 0.5 ->
/// 2/3 sigma^2. (Uniform [0,1) noise: sigma^2 = 1/12.)
#[test]
fn ewma_variance_bias_on_iid_noise_matches_theory() {
    for (a, tol) in [(0.05_f64, 0.03_f64), (0.5, 0.04)] {
        let mut r = Rng(2024);
        let mut e = EwmaDetector::new(a, 3.0);
        let (mut acc, mut n) = (0.0, 0u32);
        for i in 0..400_000 {
            e.observe(r.next());
            if i > 1000 {
                let s = e.std_dev();
                acc += s * s;
                n += 1;
            }
        }
        let mean_var = acc / f64::from(n);
        let theory = 2.0 * (1.0 - a) / (2.0 - a) / 12.0;
        assert!(
            close(mean_var, theory, tol),
            "a={a}: {mean_var} vs {theory}"
        );
    }
}

/// EWMA verdicts: boundary strict at k std, count < 3 never flags, sigma = 0
/// rule, reset forgets, alpha clamp both ways via set_alpha.
#[test]
fn ewma_verdict_rules() {
    let mut e = EwmaDetector::new(0.2, 3.0);
    e.observe(10.0);
    e.observe(11.0);
    assert!(!e.is_anomaly(1e6), "needs 3 observations");
    for v in [9.0, 10.5, 9.5, 10.0, 10.2] {
        e.observe(v);
    }
    let (mu, sd) = (e.ewma(), e.std_dev());
    assert!(sd > 1e-3);
    assert!(!e.is_anomaly(mu + 3.0 * sd * (1.0 - 1e-9)));
    assert!(e.is_anomaly(mu + 3.0 * sd * (1.0 + 1e-9)));
    assert!(!e.is_anomaly(mu - 3.0 * sd * (1.0 - 1e-9)));
    assert!(e.is_anomaly(mu - 3.0 * sd * (1.0 + 1e-9)));
    assert!(close(e.anomaly_score(mu + 5.0 * sd), 5.0, 1e-9));
    e.set_alpha(0.0);
    assert_eq!(e.alpha(), 0.001);
    e.set_alpha(7.0);
    assert_eq!(e.alpha(), 1.0);
    e.set_threshold_k(1.0);
    assert_eq!(e.threshold_k(), 1.0);
    e.reset();
    assert_eq!((e.count(), e.ewma(), e.std_dev()), (0, 0.0, 0.0));
    assert!(!e.is_anomaly(1e9));
}

/// Welford against the two-pass textbook formulas, including a large offset
/// (1e9 + noise), where a naive sum-of-squares loses all digits.
#[test]
fn zscore_matches_two_pass_mean_variance() {
    let mut r = Rng(31337);
    for offset in [0.0_f64, 1.0e9] {
        let xs: Vec<f64> = (0..5000).map(|_| offset + r.next() * 4.0 - 2.0).collect();
        let mut z = ZScoreDetector::new(3.0);
        for (i, &x) in xs.iter().enumerate() {
            z.observe(x);
            if i % 997 != 3 {
                continue;
            }
            let w = &xs[..=i];
            let n = w.len() as f64;
            let mean = w.iter().sum::<f64>() / n;
            let var = w.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
            assert!(close(z.mean(), mean, 1e-12), "offset {offset} n={n}");
            assert!(
                close(z.variance(), var, 1e-6),
                "offset {offset} n={n}: {} vs {var}",
                z.variance()
            );
            assert!(close(z.std_dev(), var.sqrt(), 1e-6));
            assert!(close(z.z_score(mean + 2.0 * var.sqrt()), 2.0, 1e-5));
            assert!(close(z.z_score(mean - 2.0 * var.sqrt()), -2.0, 1e-5));
        }
        assert_eq!(z.count(), 5000);
    }
}

#[test]
fn zscore_verdict_rules() {
    let mut z = ZScoreDetector::new(3.0);
    z.observe(1.0);
    z.observe(2.0);
    assert!(!z.is_anomaly(1e9), "count < 3");
    for v in [3.0, 1.5, 2.5, 2.0, 1.0] {
        z.observe(v);
    }
    let (m, sd) = (z.mean(), z.std_dev());
    assert!(!z.is_anomaly(m + 3.0 * sd * (1.0 - 1e-9)));
    assert!(z.is_anomaly(m + 3.0 * sd * (1.0 + 1e-9)));
    assert!(z.is_anomaly(m - 3.0 * sd * (1.0 + 1e-9)));
    z.reset();
    assert_eq!((z.count(), z.mean(), z.variance()), (0, 0.0, 0.0));
}

/// AUD-A-S4W1-014 / -015 (known defects, one cause): `EwmaDetector::observe` and
/// `ZScoreDetector::observe` feed a NaN straight into their running state, so
/// one NaN sample turns the mean (and variance) into NaN for the life of the
/// detector. Afterwards every comparison is false: the detector never flags
/// anything again, silently, until `reset()`. Expected: a spike of 1e6 over a
/// baseline of ~10 is flagged after 300 finite samples regardless of one
/// earlier NaN.
#[test]
#[ignore = "known defect: AUD-A-S4W1-014: EwmaDetector never flags again after one NaN sample (ewma = NaN permanently)"]
fn ewma_survives_a_nan_sample() {
    let mut e = EwmaDetector::new(0.1, 3.0);
    let mut r = Rng(5);
    for _ in 0..20 {
        e.observe(10.0 + r.next());
    }
    e.observe(f64::NAN);
    for _ in 0..300 {
        e.observe(10.0 + r.next());
    }
    assert!(e.is_anomaly(1.0e6), "ewma = {}", e.ewma());
}

#[test]
#[ignore = "known defect: AUD-A-S4W1-015: ZScoreDetector never flags again after one NaN sample (mean = NaN permanently)"]
fn zscore_survives_a_nan_sample() {
    let mut z = ZScoreDetector::new(3.0);
    let mut r = Rng(5);
    for _ in 0..20 {
        z.observe(10.0 + r.next());
    }
    z.observe(f64::NAN);
    for _ in 0..300 {
        z.observe(10.0 + r.next());
    }
    assert!(z.is_anomaly(1.0e6), "mean = {}", z.mean());
}

/// Same NaN contamination through the MAD detector (via `StreamingMedian`,
/// AUD-A-S4W1-012): after the NaN has left the 100-sample window the MAD
/// verdict must again be the clean one.
#[test]
#[ignore = "known defect: AUD-A-S4W1-012: MadDetector median/MAD wrong after a NaN sample has left the window"]
fn mad_survives_a_nan_sample() {
    let mut d = MadDetector::new(3.0);
    let mut r = Rng(9);
    let mut seen = Vec::new();
    for _ in 0..100 {
        let x = 10.0 + r.next();
        seen.push(x);
        d.observe(x);
    }
    d.observe(f64::NAN);
    for _ in 0..250 {
        let x = 10.0 + r.next();
        seen.push(x);
        d.observe(x);
    }
    assert_eq!(d.median(), median_of(last100(&seen)));
}

/// AUD-A-S4W1-016 (precondition unchecked): a negative `k` makes the threshold
/// negative, so every sample, the centre itself included, is flagged. The
/// constructors accept any `k` (the EWMA clamps only `alpha`).
#[test]
#[ignore = "known defect: AUD-A-S4W1-016: negative threshold_k flags the median itself (deviation 0 > negative threshold)"]
fn negative_threshold_does_not_flag_the_centre() {
    let mut m = MadDetector::new(-3.0);
    let mut e = EwmaDetector::new(0.2, -3.0);
    let mut z = ZScoreDetector::new(-3.0);
    for v in [9.0, 10.0, 11.0, 10.5, 9.5, 10.2] {
        m.observe(v);
        e.observe(v);
        z.observe(v);
    }
    let centre = m.median();
    assert!(!m.is_anomaly(centre), "MAD flagged its own median");
    assert!(!e.is_anomaly(e.ewma()), "EWMA flagged its own mean");
    assert!(!z.is_anomaly(z.mean()), "Z flagged its own mean");
}

/// AUD-A-S4W1-017 (inconsistent verdict / score, scoped): below the three-sample
/// warm-up `is_anomaly` is false for every detector, but the scores are not:
/// `ZScoreDetector::z_score` is +inf for any value != 0 even with ZERO
/// observations, and `EwmaDetector::anomaly_score` is +inf after one. The
/// composite's score (a maximum) is therefore +inf on a fresh detector while
/// its verdict is false.
#[test]
#[ignore = "known defect: AUD-A-S4W1-017: CompositeDetector::anomaly_score = +inf with 0 observations although is_anomaly is false (z_score / EWMA score lack the count<3 guard MAD has)"]
fn composite_score_matches_verdict_during_warm_up() {
    let mut c = CompositeDetector::new();
    assert!(!c.is_anomaly(100.0));
    assert!(
        c.anomaly_score(100.0).is_finite(),
        "score {}",
        c.anomaly_score(100.0)
    );
    c.observe(10.0);
    assert!(!c.is_anomaly(100.0));
    assert!(c.anomaly_score(100.0).is_finite());
}

struct Sink(Vec<AnomalyEvent>);
impl AnomalyCallback for Sink {
    fn on_anomaly(&mut self, event: AnomalyEvent) {
        self.0.push(event);
    }
}

/// `observe_with_callback`: the verdict is taken before absorbing; the event
/// carries value / timestamp / metric_id verbatim, `expected` = the MAD median
/// of the window BEFORE the sample, the score the composite's maximum; each call
/// absorbs the sample exactly once; unflagged samples emit nothing.
#[test]
fn callback_event_fields_and_single_absorption() {
    let mut c = CompositeDetector::new();
    let mut sink = Sink(Vec::new());
    let mut r = Rng(42);
    let mut history = Vec::new();
    for i in 0..60_u64 {
        let x = 10.0 + r.next() - 0.5;
        let before = c.count();
        let events = sink.0.len();
        let flagged = c.observe_with_callback(x, 1000 + i, 7, &mut sink);
        history.push(x);
        assert_eq!(c.count(), before + 1, "exactly one absorption");
        assert_eq!(
            sink.0.len(),
            events + usize::from(flagged),
            "one event iff flagged"
        );
    }
    let quiet = sink.0.len();
    let med_before = median_of(&history);
    let score_before = c.anomaly_score(500.0);
    assert!(c.observe_with_callback(500.0, 99_999, 4242, &mut sink));
    assert_eq!(sink.0.len(), quiet + 1);
    let ev = sink.0.last().expect("event");
    assert_eq!(
        (ev.value, ev.timestamp, ev.metric_id),
        (500.0, 99_999, 4242)
    );
    assert_eq!(ev.expected, med_before);
    assert_eq!(ev.score, score_before);
    assert_eq!(c.count(), 61);
}

/// Composite wiring: `with_thresholds` order (mad_k, ewma_alpha, ewma_k,
/// zscore_k) and `require_consensus` as AND, default OR.
#[test]
fn composite_threshold_order_and_consensus() {
    let c = CompositeDetector::with_thresholds(1.5, 0.3, 2.5, 4.0);
    assert_eq!(c.mad.threshold_k(), 1.5);
    assert_eq!(c.ewma.alpha(), 0.3);
    assert_eq!(c.ewma.threshold_k(), 2.5);
    assert!(!c.require_consensus);
    let d = CompositeDetector::default();
    assert_eq!(
        (d.mad.threshold_k(), d.ewma.alpha(), d.ewma.threshold_k()),
        (3.0, 0.1, 3.0)
    );
}
