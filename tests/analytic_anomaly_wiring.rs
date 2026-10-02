//! Oracles for the wiring of `anomaly`: the windowed median, the MAD, EWMA
//! and Welford Z-score detectors, their composite, and the composite's
//! callback entry `observe_with_callback`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! * **Windowed median** of `1,2,5,8,9` is `5`; of `1,2,4,5,8,9` is
//!   `(4+5)/2 = 4.5`; of `0..=99` is `(49+50)/2 = 49.5`; once `0` is evicted
//!   by a `1000` the window is `1..=99, 1000` and the median is
//!   `(50+51)/2 = 50.5`. The window is full after exactly `WINDOW = 100`
//!   pushes and not after `99`.
//! * **MAD** of `1..=9`: median `5`, `|x−5| = 4,3,2,1,0,1,2,3,4`, sorted
//!   `0,1,1,2,2,3,3,4,4` → MAD `2`. Score of `13` is `8 / (2·1.4826)`, the
//!   threshold is `k·2·1.4826 = 2.9652·k`: `k = 3` flags `15` (dev 10) and not
//!   `13` (dev 8); `k = 2` flags `13`; `k = 0` flags any non-zero deviation.
//! * **EWMA** with `α = 1/2`, first sample `0`, then constant `8`:
//!   `ewma_n = 8 − 8·(1/2)ⁿ = 4, 6, 7` and
//!   `var_n = (1−α)(α·d_n² + var_{n−1})` with `d_n = 8 − ewma_{n−1}`:
//!   `var = 16, 12, 7`, so `σ = √7` and `score(9) = 2/√7`.
//!   With `α = 1` the mean is the last sample and `var = (1−1)·… = 0`.
//! * **Welford** on `10,11,9,10,11,9`: mean `10`, `Σ(x−10)² = 4`, sample
//!   variance `4/5 = 0.8`, `z(12) = +2/√0.8`, `z(8) = −2/√0.8 ≈ −2.236`:
//!   `k = 3` does not flag `12`, `k = 2` does.
//! * **Composite**: `with_thresholds(a, b, c, d)` must land each parameter
//!   on its own detector; the verdict is the OR of the three (AND under
//!   `require_consensus`); the score is the maximum of the three.
//! * **Callback**: twelve `10`s, a `100` at index 12, three more `10`s. Only
//!   index 12 is judged against a flat baseline, and against a flat baseline
//!   every detector uses its `σ = 0` rule, so the score is `+∞` and the
//!   expected value is the MAD median `10`.
//!
//! # Degenerate inputs (documented result, not "no panic")
//!
//! * empty window: `median = 0.0`, MAD `median = mad = 0.0`, Welford
//!   `variance = std_dev = 0.0`; `z_score(0) = 0`, `z_score(x ≠ 0) = +∞`
//!   (the `σ < 1e-10` rule against mean `0`); `is_anomaly` is `false` for
//!   every detector below three observations; `EwmaDetector::anomaly_score`
//!   is `0.0` while uninitialised.
//! * one sample: Welford variance `0` (`count < 2`), MAD window of length 1
//!   has median = that sample and MAD `0`.
//! * `α = 0` clamps to `0.001`, `α = 1` is kept, `α = 5` clamps to `1`.
//! * `k = 0`: a deviation of exactly `0` is not flagged (strict `>`), any
//!   other deviation is.
//! * identical samples: `σ = 0` → `z_score` / `anomaly_score` are `0` at the
//!   mean and `+∞` elsewhere, and `is_anomaly` is `|x − centre| > 1e-10`.
//! * `f64::MAX` followed by `−f64::MAX`: IEEE gives `MAX − (−MAX) = +∞`, the
//!   running statistics become `±∞` and then `NaN`; nothing panics, the
//!   variance is `NaN` and the verdict degrades to `false` (a `NaN`
//!   comparison is never `>`). That silent degradation is asserted as the
//!   documented behaviour of the current code.
//! * `reset` on every detector: replaying the same stream after `reset`
//!   reproduces the fresh detector bit for bit (`to_bits` equality) and
//!   keeps the tuning (`alpha`, `threshold_k`) untouched.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::anomaly::{
    AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian,
    ZScoreDetector,
};
use alice_physics::det_math::sqrt64;

const MAD_SCALE: f64 = 1.4826;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// `(1 − α)ⁿ` by repeated multiplication (no `powf`).
fn decay_pow(alpha: f64, n: u32) -> f64 {
    let mut p = 1.0;
    for _ in 0..n {
        p *= 1.0 - alpha;
    }
    p
}

// ---------------------------------------------------------------------------
// StreamingMedian
// ---------------------------------------------------------------------------

#[test]
fn window_constant_is_100_and_is_full_flips_at_exactly_window() {
    assert_eq!(StreamingMedian::WINDOW, 100);
    let mut sm = StreamingMedian::new();
    assert!(!sm.is_full(), "empty window is not full");
    for i in 0..StreamingMedian::WINDOW - 1 {
        sm.push(i as f64);
        assert!(!sm.is_full(), "not full after {} pushes", i + 1);
    }
    sm.push(99.0);
    assert!(sm.is_full(), "full after exactly WINDOW pushes");
    assert_eq!(sm.count(), StreamingMedian::WINDOW);
    sm.push(1000.0);
    assert!(sm.is_full(), "stays full once full");
    assert_eq!(sm.count(), StreamingMedian::WINDOW, "count saturates");
}

#[test]
fn streaming_median_odd_even_and_eviction_by_hand() {
    let mut sm = StreamingMedian::new();
    // oracle: empty window → 0.0 (documented)
    assert_eq!(sm.median().to_bits(), 0.0_f64.to_bits());
    sm.push(3.5);
    // oracle: window of length 1 → the sample itself
    assert_eq!(sm.median(), 3.5);
    sm.clear();
    for v in [5.0, 2.0, 8.0, 1.0, 9.0] {
        sm.push(v);
    }
    // oracle: sorted 1,2,5,8,9 → 5
    assert_eq!(sm.median(), 5.0);
    sm.push(4.0);
    // oracle: sorted 1,2,4,5,8,9 → (4 + 5) / 2
    assert_eq!(sm.median(), 4.5);
    sm.clear();
    for i in 0..StreamingMedian::WINDOW {
        sm.push(i as f64);
    }
    // oracle: 0..=99 → (49 + 50) / 2
    assert_eq!(sm.median(), 49.5);
    sm.push(1000.0);
    // oracle: 0 evicted, window 1..=99 ∪ {1000} → (50 + 51) / 2
    assert_eq!(sm.median(), 50.5);
    sm.clear();
    assert_eq!(sm.count(), 0);
    assert_eq!(sm.median().to_bits(), 0.0_f64.to_bits());
}

// ---------------------------------------------------------------------------
// MadDetector
// ---------------------------------------------------------------------------

#[test]
fn mad_one_to_nine_median_5_mad_2_score_and_threshold_by_hand() {
    let mut d = MadDetector::new(3.0);
    for v in 1..=9 {
        d.observe(f64::from(v));
    }
    assert_eq!(d.count(), 9);
    assert_eq!(d.median(), 5.0);
    assert_eq!(d.mad(), 2.0);
    assert_eq!(d.threshold_k(), 3.0);
    let score13 = 8.0 / (2.0 * MAD_SCALE);
    assert!(close(d.anomaly_score(13.0), score13, 1e-12));
    // the same value when the score is the very first query after the
    // observations (the score must build the cache itself)
    let mut first_query = MadDetector::new(3.0);
    for v in 1..=9 {
        first_query.observe(f64::from(v));
    }
    assert!(close(first_query.anomaly_score(13.0), score13, 1e-12));
    // below the median the score uses |x − median|: score(−3) = 8 / (2·1.4826)
    assert!(close(d.anomaly_score(-3.0), score13, 1e-12));
    assert!(close(d.anomaly_score(5.0), 0.0, 0.0));
    // k = 3: threshold 8.8956 — dev 10 flagged, dev 8 not
    assert!(d.is_anomaly(15.0));
    assert!(d.is_anomaly(-5.0));
    assert!(!d.is_anomaly(13.0));
    // k = 2: threshold 5.9304 — dev 8 now flagged, dev 5 not
    d.set_threshold_k(2.0);
    assert_eq!(d.threshold_k(), 2.0);
    assert!(d.is_anomaly(13.0));
    assert!(!d.is_anomaly(10.0));
    assert!(
        close(d.anomaly_score(13.0), score13, 1e-12),
        "score ignores k"
    );
    // k = 4: threshold 11.8608 — dev 10 no longer flagged
    d.set_threshold_k(4.0);
    assert!(!d.is_anomaly(15.0));
}

#[test]
fn mad_degenerate_empty_single_constant_and_k_zero() {
    let mut d = MadDetector::new(3.0);
    // oracle: empty → median 0, mad 0, no verdict, score 0
    assert_eq!(d.median().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.mad().to_bits(), 0.0_f64.to_bits());
    assert!(!d.is_anomaly(1e9));
    assert_eq!(d.anomaly_score(1e9).to_bits(), 0.0_f64.to_bits());
    // oracle: one sample → median is the sample, MAD 0, still no verdict
    d.observe(42.0);
    assert_eq!(d.median(), 42.0);
    assert_eq!(d.mad(), 0.0);
    assert!(!d.is_anomaly(1e9));
    assert!(!d.is_anomaly(42.0));
    d.observe(42.0);
    assert!(
        !d.is_anomaly(1e9),
        "two samples: still below the 3-sample gate"
    );
    d.observe(42.0);
    // oracle: three identical samples → MAD 0 rule
    assert!(d.is_anomaly(42.1));
    assert!(!d.is_anomaly(42.0));
    assert_eq!(d.anomaly_score(42.1), f64::INFINITY);
    assert_eq!(d.anomaly_score(42.0), 0.0);
    // oracle: k = 0 with MAD > 0 → threshold 0, strict inequality
    let mut k0 = MadDetector::new(0.0);
    for v in 1..=9 {
        k0.observe(f64::from(v));
    }
    assert_eq!(k0.threshold_k(), 0.0);
    assert!(!k0.is_anomaly(5.0), "deviation 0 > 0 is false");
    assert!(k0.is_anomaly(5.5));
    assert!(k0.is_anomaly(4.5));
    // clear → empty state again
    k0.clear();
    assert_eq!(k0.count(), 0);
    assert_eq!(k0.median().to_bits(), 0.0_f64.to_bits());
    assert_eq!(k0.mad().to_bits(), 0.0_f64.to_bits());
}

// ---------------------------------------------------------------------------
// EwmaDetector
// ---------------------------------------------------------------------------

#[test]
fn ewma_half_alpha_closed_form_mean_and_variance() {
    let mut d = EwmaDetector::new(0.5, 3.0);
    assert_eq!(d.alpha(), 0.5);
    assert_eq!(d.threshold_k(), 3.0);
    // oracle: uninitialised → ewma 0, σ 0, score 0, no verdict
    assert_eq!(d.ewma().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.std_dev().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.anomaly_score(5.0).to_bits(), 0.0_f64.to_bits());
    assert!(!d.is_anomaly(5.0));
    d.observe(0.0);
    assert_eq!(d.ewma(), 0.0, "first sample seeds the mean");
    assert_eq!(d.count(), 1);
    assert!(!d.is_anomaly(1e9), "count 1: no verdict");
    let mut var = 0.0;
    let mut prev = 0.0;
    for n in 1..=3u32 {
        d.observe(8.0);
        if n == 1 {
            // count 2 (< 3): still no verdict even though σ = 4 and the
            // probe sits 2.5e8 σ away
            assert!(!d.is_anomaly(1e9), "count 2: no verdict");
        }
        let dev = 8.0 - prev;
        var = 0.5 * (0.5 * dev * dev + var);
        let closed = 8.0 - 8.0 * decay_pow(0.5, n);
        prev = closed;
        assert!(close(d.ewma(), closed, 1e-12), "n={n} ewma={}", d.ewma());
        assert!(
            close(d.std_dev(), sqrt64(var), 1e-12),
            "n={n} std={}",
            d.std_dev()
        );
    }
    // hand values of the recursion: ewma 7, var 7
    assert!(close(d.ewma(), 7.0, 1e-12));
    assert!(close(d.std_dev(), sqrt64(7.0), 1e-12));
    assert_eq!(d.count(), 4);
    let z9 = 2.0 / sqrt64(7.0);
    assert!(close(d.anomaly_score(9.0), z9, 1e-12));
    assert!(close(d.anomaly_score(5.0), z9, 1e-12), "score is |dev|/σ");
    // k = 3 → 0.756 > 3 false; k = 0.5 → true; k = z9 exactly → strict, false
    assert!(!d.is_anomaly(9.0));
    d.set_threshold_k(0.5);
    assert_eq!(d.threshold_k(), 0.5);
    assert!(d.is_anomaly(9.0));
    assert!(d.is_anomaly(5.0));
    assert!(close(d.anomaly_score(9.0), z9, 1e-12), "score ignores k");
    d.set_threshold_k(0.0);
    assert!(!d.is_anomaly(7.0), "deviation 0 > 0 is false at k = 0");
    assert!(d.is_anomaly(7.0 + 1e-6));
}

#[test]
fn ewma_alpha_clamp_edges_and_alpha_one_tracks_last_sample() {
    // oracle: clamp to [0.001, 1]
    assert_eq!(EwmaDetector::new(0.0, 3.0).alpha(), 0.001);
    assert_eq!(EwmaDetector::new(1.0, 3.0).alpha(), 1.0);
    assert_eq!(EwmaDetector::new(5.0, 3.0).alpha(), 1.0);
    assert_eq!(EwmaDetector::new(-1.0, 3.0).alpha(), 0.001);
    let mut d = EwmaDetector::new(0.5, 3.0);
    d.set_alpha(0.0);
    assert_eq!(d.alpha(), 0.001);
    d.set_alpha(1.0);
    assert_eq!(d.alpha(), 1.0);
    d.set_alpha(0.25);
    assert_eq!(d.alpha(), 0.25);
    // oracle: α = 1 → ewma = last sample, var = 0 → σ = 0 rule
    let mut one = EwmaDetector::new(1.0, 3.0);
    for v in [10.0, 10.0, 100.0] {
        one.observe(v);
    }
    assert_eq!(one.ewma(), 100.0);
    assert_eq!(one.std_dev(), 0.0);
    assert!(!one.is_anomaly(100.0));
    assert!(one.is_anomaly(10.0));
    assert_eq!(one.anomaly_score(10.0), f64::INFINITY);
    assert_eq!(one.anomaly_score(100.0), 0.0);
    // oracle: α = 0.001 after 10, 10, 100: ewma = 10 + 0.09 = 10.09,
    // var = 0.999·(0.001·8100) = 8.0919
    let mut slow = EwmaDetector::new(0.001, 3.0);
    for v in [10.0, 10.0, 100.0] {
        slow.observe(v);
    }
    assert!(close(slow.ewma(), 10.09, 1e-9));
    assert!(close(slow.std_dev(), sqrt64(8.0919), 1e-9));
    // oracle: set_alpha changes the next update only — same stream, α = 0.5
    // vs α = 0.001, the means differ: 10 + 0.5·90 = 55 vs 10.09
    let mut fast = EwmaDetector::new(0.001, 3.0);
    fast.set_alpha(0.5);
    for v in [10.0, 10.0, 100.0] {
        fast.observe(v);
    }
    assert!(close(fast.ewma(), 55.0, 1e-9));
}

#[test]
fn ewma_reset_replays_bit_for_bit_and_keeps_tuning() {
    let stream = [3.0, 1.5, -2.25, 8.0, 8.0, 0.125, 7.75];
    let mut fresh = EwmaDetector::new(0.3, 2.5);
    let mut reused = EwmaDetector::new(0.3, 2.5);
    for &v in &[100.0, -100.0, 0.0, 42.0] {
        reused.observe(v);
    }
    reused.reset();
    // oracle: reset → ewma 0, σ 0, count 0, alpha / k untouched
    assert_eq!(reused.ewma().to_bits(), 0.0_f64.to_bits());
    assert_eq!(reused.std_dev().to_bits(), 0.0_f64.to_bits());
    assert_eq!(reused.count(), 0);
    assert_eq!(reused.alpha(), 0.3);
    assert_eq!(reused.threshold_k(), 2.5);
    assert!(!reused.is_anomaly(1e9), "no verdict right after reset");
    for &v in &stream {
        fresh.observe(v);
        reused.observe(v);
        assert_eq!(fresh.ewma().to_bits(), reused.ewma().to_bits());
        assert_eq!(fresh.std_dev().to_bits(), reused.std_dev().to_bits());
        assert_eq!(fresh.count(), reused.count());
        assert_eq!(fresh.is_anomaly(9.0), reused.is_anomaly(9.0));
        assert_eq!(
            fresh.anomaly_score(9.0).to_bits(),
            reused.anomaly_score(9.0).to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// ZScoreDetector
// ---------------------------------------------------------------------------

#[test]
fn zscore_welford_mean_10_variance_0_8_by_hand() {
    let mut d = ZScoreDetector::new(3.0);
    for v in [10.0, 11.0, 9.0, 10.0, 11.0, 9.0] {
        d.observe(v);
    }
    assert_eq!(d.count(), 6);
    assert!(close(d.mean(), 10.0, 1e-12));
    assert!(close(d.variance(), 0.8, 1e-12));
    let sigma = sqrt64(0.8);
    assert!(close(d.std_dev(), sigma, 1e-12));
    assert!(close(d.z_score(12.0), 2.0 / sigma, 1e-12));
    assert!(
        close(d.z_score(8.0), -2.0 / sigma, 1e-12),
        "z keeps its sign"
    );
    assert!(close(d.z_score(10.0), 0.0, 1e-12));
    // k = 3: 2.236 > 3 false; k = 2: true; k = 0: any non-zero |z|
    assert!(!d.is_anomaly(12.0));
    assert!(!d.is_anomaly(8.0));
    let mut tight = ZScoreDetector::new(2.0);
    let mut k0 = ZScoreDetector::new(0.0);
    for v in [10.0, 11.0, 9.0, 10.0, 11.0, 9.0] {
        tight.observe(v);
        k0.observe(v);
    }
    assert!(tight.is_anomaly(12.0));
    assert!(tight.is_anomaly(8.0));
    assert!(!tight.is_anomaly(11.0), "|z| = 1.118 > 2 is false");
    // probe at the running mean itself (Welford's last update 10.2 − 1.2/6
    // is not exact in binary, so a literal 10 may sit 1 ulp off)
    let centre = k0.mean();
    assert!(!k0.is_anomaly(centre), "z = 0 > 0 is false at k = 0");
    assert!(k0.is_anomaly(centre + 1e-6));
}

#[test]
fn zscore_degenerate_empty_single_constant_and_reset_replay() {
    let mut d = ZScoreDetector::new(3.0);
    // oracle: empty → variance 0, σ 0, mean 0; z(0) = 0, z(x ≠ 0) = +∞; no verdict
    assert_eq!(d.variance().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.std_dev().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.mean().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.z_score(0.0), 0.0);
    assert_eq!(d.z_score(1.0), f64::INFINITY);
    assert_eq!(d.z_score(-1.0), f64::INFINITY, "σ = 0 rule is unsigned");
    assert!(!d.is_anomaly(1.0));
    // oracle: one sample → mean is the sample, variance 0 (count < 2)
    d.observe(4.0);
    assert_eq!(d.mean(), 4.0);
    assert_eq!(d.variance(), 0.0);
    assert!(!d.is_anomaly(1e9));
    d.observe(4.0);
    assert!(
        !d.is_anomaly(1e9),
        "two samples: still below the 3-sample gate"
    );
    d.observe(4.0);
    // oracle: three identical → σ = 0 rule
    assert!(!d.is_anomaly(4.0));
    assert!(d.is_anomaly(4.0 + 1e-6));
    assert_eq!(d.z_score(5.0), f64::INFINITY);
    assert_eq!(d.z_score(4.0), 0.0);
    // a non-flat tail so that reset has a non-zero m2 to forget
    d.observe(100.0);
    assert!(d.variance() > 0.0);
    // oracle: reset → mean 0, m2 0, count 0, then a replay is bit-identical
    d.reset();
    assert_eq!(d.mean().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.variance().to_bits(), 0.0_f64.to_bits());
    assert_eq!(d.count(), 0);
    let mut fresh = ZScoreDetector::new(3.0);
    for &v in &[3.0, 1.5, -2.25, 8.0, 8.0, 0.125, 7.75] {
        fresh.observe(v);
        d.observe(v);
        assert_eq!(fresh.mean().to_bits(), d.mean().to_bits());
        assert_eq!(fresh.variance().to_bits(), d.variance().to_bits());
        assert_eq!(fresh.z_score(9.0).to_bits(), d.z_score(9.0).to_bits());
        assert_eq!(fresh.is_anomaly(9.0), d.is_anomaly(9.0));
    }
}

// ---------------------------------------------------------------------------
// CompositeDetector
// ---------------------------------------------------------------------------

#[test]
fn composite_with_thresholds_lands_each_parameter_on_its_detector() {
    let c = CompositeDetector::with_thresholds(2.5, 0.25, 4.5, 1.0);
    assert_eq!(c.mad.threshold_k(), 2.5);
    assert_eq!(c.ewma.alpha(), 0.25);
    assert_eq!(c.ewma.threshold_k(), 4.5);
    assert!(!c.require_consensus);
    assert_eq!(c.count(), 0);
    // zscore_k has no getter: isolate it by making the other two inert
    // (k = 1e6) and compare k = 1 against k = 100 on the Welford stream
    // whose |z(12)| = 2.236.
    let stream = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0];
    let mut tight = CompositeDetector::with_thresholds(1e6, 0.1, 1e6, 1.0);
    let mut loose = CompositeDetector::with_thresholds(1e6, 0.1, 1e6, 100.0);
    for &v in &stream {
        tight.observe(v);
        loose.observe(v);
    }
    assert_eq!(tight.count(), 6);
    assert!(!tight.mad.is_anomaly(12.0));
    assert!(!tight.ewma.is_anomaly(12.0));
    assert!(tight.zscore.is_anomaly(12.0));
    assert!(!loose.zscore.is_anomaly(12.0));
    assert!(tight.is_anomaly(12.0));
    assert!(!loose.is_anomaly(12.0));
    // swapped positions would be visible: (1.0, 0.25, 4.5, 2.5) is not the same
    let swapped = CompositeDetector::with_thresholds(1.0, 0.25, 4.5, 2.5);
    assert_eq!(swapped.mad.threshold_k(), 1.0);
    let default = CompositeDetector::new();
    assert_eq!(default.mad.threshold_k(), 3.0);
    assert_eq!(default.ewma.alpha(), 0.1);
    assert_eq!(default.ewma.threshold_k(), 3.0);
}

#[test]
fn composite_verdict_is_or_of_three_and_consensus_is_and() {
    // Welford stream, zscore_k = 1 is the only detector that fires at 12.
    let stream = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0];
    let mut c = CompositeDetector::with_thresholds(1e6, 0.1, 1e6, 1.0);
    for &v in &stream {
        c.observe(v);
    }
    assert!(c.is_anomaly(12.0), "one of three fires → OR is true");
    c.require_consensus = true;
    assert!(!c.is_anomaly(12.0), "one of three fires → AND is false");
    // all three fire at 1e7: MAD threshold 1e6·1·1.4826 = 1.48e6 < 1e7,
    // EWMA threshold 1e6·σ with σ < 1, |z| = 1e7/0.894
    assert!(c.mad.is_anomaly(1e7) && c.ewma.is_anomaly(1e7) && c.zscore.is_anomaly(1e7));
    assert!(c.is_anomaly(1e7));
    c.require_consensus = false;
    assert!(c.is_anomaly(1e7));
    assert!(!c.is_anomaly(10.0), "none fire at the centre");
    // a stream where only the MAD detector fires: 1..=9 with mad_k = 2 flags
    // 13 (dev 8 > 5.93) while |z(13)| = 8/√7.5 = 2.92 < 1e6 and the EWMA
    // deviation is bounded (k = 1e6 on both).
    let mut mad_only = CompositeDetector::with_thresholds(2.0, 0.1, 1e6, 1e6);
    for v in 1..=9 {
        mad_only.observe(f64::from(v));
    }
    assert!(mad_only.mad.is_anomaly(13.0));
    assert!(!mad_only.ewma.is_anomaly(13.0));
    assert!(!mad_only.zscore.is_anomaly(13.0));
    assert!(mad_only.is_anomaly(13.0), "MAD alone decides under OR");
    // a stream where only EWMA fires: 1..=9 then ewma_k small
    let mut ewma_only = CompositeDetector::with_thresholds(1e6, 0.1, 0.1, 1e6);
    for v in 1..=9 {
        ewma_only.observe(f64::from(v));
    }
    assert!(!ewma_only.mad.is_anomaly(13.0));
    assert!(ewma_only.ewma.is_anomaly(13.0));
    assert!(!ewma_only.zscore.is_anomaly(13.0));
    assert!(ewma_only.is_anomaly(13.0), "EWMA alone decides under OR");
}

#[test]
fn composite_score_is_the_maximum_of_the_three() {
    // 1..=9: MAD score(13) = 8/(2·1.4826) = 2.698; Welford mean 5,
    // variance Σ(x−5)²/8 = 60/8 = 7.5, |z(13)| = 8/√7.5 = 2.921. The EWMA
    // component has no short closed form on this stream, so it is read from
    // its own detector and the oracle is the max of the three.
    let mut c = CompositeDetector::with_thresholds(3.0, 0.1, 3.0, 3.0);
    for v in 1..=9 {
        c.observe(f64::from(v));
    }
    let mad_s = 8.0 / (2.0 * MAD_SCALE);
    let z_s = 8.0 / sqrt64(7.5);
    assert!(close(c.mad.anomaly_score(13.0), mad_s, 1e-12));
    assert!(close(c.zscore.z_score(13.0), z_s, 1e-12));
    let ewma_s = c.ewma.anomaly_score(13.0);
    let combined = c.anomaly_score(13.0);
    assert!(close(combined, mad_s.max(z_s).max(ewma_s), 1e-12));
    assert!(
        combined >= z_s - 1e-12,
        "|z| = 2.921 is the larger of the two closed forms"
    );
    // below the median the Z-score is negative; the composite uses |z|
    let combined_low = c.anomaly_score(-3.0);
    assert!(close(c.zscore.z_score(-3.0), -z_s, 1e-12));
    assert!(
        combined_low >= z_s - 1e-12,
        "|z(−3)| = 8/√7.5 enters the max"
    );
    // flat stream: every component is +∞ off-centre and 0 at the centre
    let mut flat = CompositeDetector::new();
    for _ in 0..5 {
        flat.observe(10.0);
    }
    assert_eq!(flat.anomaly_score(11.0), f64::INFINITY);
    assert_eq!(flat.anomaly_score(10.0), 0.0);
}

#[test]
fn composite_reset_replays_bit_for_bit() {
    let stream = [3.0, 1.5, -2.25, 8.0, 8.0, 0.125, 7.75, 100.0, 3.0];
    let mut fresh = CompositeDetector::with_thresholds(2.0, 0.3, 2.5, 2.0);
    let mut reused = CompositeDetector::with_thresholds(2.0, 0.3, 2.5, 2.0);
    for &v in &[100.0, -100.0, 0.0, 42.0, 7.0] {
        reused.observe(v);
    }
    reused.reset();
    assert_eq!(reused.count(), 0);
    assert_eq!(reused.mad.count(), 0);
    assert_eq!(reused.ewma.count(), 0);
    assert_eq!(reused.mad.median().to_bits(), 0.0_f64.to_bits());
    assert_eq!(reused.ewma.ewma().to_bits(), 0.0_f64.to_bits());
    assert_eq!(reused.zscore.mean().to_bits(), 0.0_f64.to_bits());
    assert!(!reused.is_anomaly(1e9), "no verdict right after reset");
    for &v in &stream {
        fresh.observe(v);
        reused.observe(v);
        assert_eq!(fresh.mad.median().to_bits(), reused.mad.median().to_bits());
        assert_eq!(fresh.mad.mad().to_bits(), reused.mad.mad().to_bits());
        assert_eq!(fresh.ewma.ewma().to_bits(), reused.ewma.ewma().to_bits());
        assert_eq!(
            fresh.ewma.std_dev().to_bits(),
            reused.ewma.std_dev().to_bits()
        );
        assert_eq!(
            fresh.zscore.mean().to_bits(),
            reused.zscore.mean().to_bits()
        );
        assert_eq!(
            fresh.zscore.variance().to_bits(),
            reused.zscore.variance().to_bits()
        );
        assert_eq!(fresh.is_anomaly(9.0), reused.is_anomaly(9.0));
        assert_eq!(
            fresh.anomaly_score(9.0).to_bits(),
            reused.anomaly_score(9.0).to_bits()
        );
    }
}

// ---------------------------------------------------------------------------
// AnomalyCallback via CompositeDetector::observe_with_callback
// ---------------------------------------------------------------------------

struct IndexRecorder {
    fired: Vec<(usize, AnomalyEvent)>,
}

impl AnomalyCallback for IndexRecorder {
    fn on_anomaly(&mut self, event: AnomalyEvent) {
        let index = usize::try_from(event.timestamp).expect("index fits usize");
        self.fired.push((index, event));
    }
}

#[test]
fn callback_fires_exactly_at_the_spike_with_score_inf_and_expected_median() {
    let stream: Vec<f64> = (0..16)
        .map(|i| if i == 12 { 100.0 } else { 10.0 })
        .collect();
    let mut c = CompositeDetector::new();
    let mut rec = IndexRecorder { fired: Vec::new() };
    let mut verdicts = Vec::new();
    for (i, &v) in stream.iter().enumerate() {
        verdicts.push(c.observe_with_callback(v, i as u64, 7, &mut rec));
    }
    // oracle: index 12 only (indices 0..2 are below the 3-sample gate, 3..11
    // sit on a flat baseline at their own value, 13..15 are judged against
    // statistics the spike has widened: MAD = 0 rule at centre 10 → false,
    // EWMA |10 − 19| = 9 < 3·27, Welford |z| = 6.92/24.96 < 3).
    let expected_verdicts: Vec<bool> = (0..16).map(|i| i == 12).collect();
    assert_eq!(verdicts, expected_verdicts);
    let fired: Vec<usize> = rec.fired.iter().map(|(i, _)| *i).collect();
    assert_eq!(fired, vec![12]);
    let (_, ev) = &rec.fired[0];
    assert_eq!(ev.value, 100.0);
    assert_eq!(ev.score, f64::INFINITY, "flat baseline → σ = 0 rule → +∞");
    assert_eq!(
        ev.expected, 10.0,
        "expected is the MAD median of the baseline"
    );
    assert_eq!(ev.timestamp, 12);
    assert_eq!(ev.metric_id, 7);
    // the sample was absorbed after the verdict: count = 16 and the median
    // of the window still 10 (one 100 among fifteen 10s)
    assert_eq!(c.count(), 16);
    assert_eq!(c.mad.median(), 10.0);
    assert_eq!(rec.fired.len(), 1, "no second event for the same spike");
}

#[test]
fn callback_judges_before_absorbing_and_carries_the_finite_score() {
    // 1..=9 then 13 with mad_k = 2 (flags dev 8 > 5.93), ewma / zscore inert.
    // Judged before absorbing: score = max(MAD 2.698, EWMA s, |z| 2.921);
    // the MAD median at that moment is 5, so expected = 5. Had 13 been
    // absorbed first, the median would still be 5 but the MAD would grow
    // (|13 − 5| = 8 enters the deviations) and the verdict would change
    // for a later 13 — the second call below pins that difference.
    // (The warm-up goes through plain `observe`: fed one by one through the
    // callback entry, `6` would already be judged against the window 1..=5
    // — median 3, MAD 1, dev 3 > 2.97 — and fire.)
    let mut c = CompositeDetector::with_thresholds(2.0, 0.1, 1e6, 1e6);
    let mut rec = IndexRecorder { fired: Vec::new() };
    for v in 1..=9 {
        c.observe(f64::from(v));
    }
    assert_eq!(c.count(), 9);
    let mad_s = 8.0 / (2.0 * MAD_SCALE);
    let z_s = 8.0 / sqrt64(7.5);
    let ewma_s_before = c.ewma.anomaly_score(13.0);
    assert!(c.observe_with_callback(13.0, 9, 1, &mut rec));
    assert_eq!(rec.fired.len(), 1);
    let (i, ev) = &rec.fired[0];
    assert_eq!(*i, 9);
    assert_eq!(ev.value, 13.0);
    assert_eq!(ev.expected, 5.0);
    assert_eq!(ev.metric_id, 1);
    let max_s = mad_s.max(z_s).max(ewma_s_before);
    assert!(
        close(ev.score, max_s, 1e-12),
        "score {} vs {max_s}",
        ev.score
    );
    assert!(ev.score.is_finite());
    assert_eq!(c.count(), 10, "the sample was absorbed after the verdict");
    // after absorbing 13 the window is 1..=9,13: median (5+6)/2 = 5.5,
    // deviations 4.5,3.5,2.5,1.5,0.5,0.5,1.5,2.5,3.5,7.5 → sorted
    // 0.5,0.5,1.5,1.5,2.5,2.5,3.5,3.5,4.5,7.5 → MAD (2.5+2.5)/2 = 2.5
    assert_eq!(c.mad.median(), 5.5);
    assert_eq!(c.mad.mad(), 2.5);
    // mad_k = 2: threshold 2·2.5·1.4826 = 7.413 — a second 13 (dev 7.5) still
    // fires, a 12 (dev 6.5) does not; both go through the callback entry
    assert!(c.observe_with_callback(13.0, 10, 1, &mut rec));
    assert!(!c.observe_with_callback(12.0, 11, 1, &mut rec));
    let fired: Vec<usize> = rec.fired.iter().map(|(i, _)| *i).collect();
    assert_eq!(fired, vec![9, 10]);
    assert_eq!(rec.fired[1].1.expected, 5.5);
}

#[test]
fn callback_withholds_the_first_two_samples_and_after_reset() {
    let mut c = CompositeDetector::new();
    let mut rec = IndexRecorder { fired: Vec::new() };
    // oracle: 0, 1e9 at index 1 is not reported (count < 3), 1e9 again at
    // index 2 is not reported (count = 2 < 3), and at index 3 the window
    // 0, 1e9, 1e9 has median 1e9 → a 0 is judged against it
    assert!(!c.observe_with_callback(0.0, 0, 0, &mut rec));
    assert!(!c.observe_with_callback(1e9, 1, 0, &mut rec));
    assert!(!c.observe_with_callback(1e9, 2, 0, &mut rec));
    assert!(rec.fired.is_empty());
    c.reset();
    assert!(!c.observe_with_callback(5.0, 3, 0, &mut rec));
    assert!(!c.observe_with_callback(5.0, 4, 0, &mut rec));
    assert!(!c.observe_with_callback(5.0, 5, 0, &mut rec));
    assert!(
        rec.fired.is_empty(),
        "three samples after reset: gate again"
    );
    assert!(
        c.observe_with_callback(6.0, 6, 0, &mut rec),
        "flat 5 → 6 fires"
    );
    let fired: Vec<usize> = rec.fired.iter().map(|(i, _)| *i).collect();
    assert_eq!(fired, vec![6]);
}

// ---------------------------------------------------------------------------
// Extreme magnitudes
// ---------------------------------------------------------------------------

#[test]
fn extreme_magnitudes_overflow_to_inf_then_nan_without_panic() {
    // oracle (IEEE 754): MAX − (−MAX) = +∞; Welford mean = MAX + (−∞)/2 = −∞;
    // delta2 = −MAX − (−∞) = +∞; m2 = (−∞)(+∞) = −∞; variance = −∞/1 = −∞;
    // std_dev = √(−∞) = NaN; a third sample makes the mean NaN too. The
    // verdict is `false` because `NaN > k` is false — the documented
    // degradation of the current code (no NaN guard).
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut z = ZScoreDetector::new(3.0);
        z.observe(f64::MAX);
        z.observe(-f64::MAX);
        assert_eq!(z.mean(), f64::NEG_INFINITY);
        assert_eq!(z.variance(), f64::NEG_INFINITY);
        assert!(z.std_dev().is_nan());
        z.observe(0.0);
        assert!(z.mean().is_nan());
        assert!(z.variance().is_nan());
        assert!(z.z_score(0.0).is_nan());
        assert!(!z.is_anomaly(0.0), "NaN statistics → verdict false");
        assert!(!z.is_anomaly(f64::MAX));
    }));
    assert!(r.is_ok(), "Welford on ±MAX must not panic");

    // oracle: EWMA deviation = −MAX − MAX = −∞ → ewma = −∞,
    // var = (1−α)·((α·(−∞))·(−∞) + 0) = +∞ → σ = +∞; score(0) = ∞/∞ = NaN
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut e = EwmaDetector::new(0.1, 3.0);
        e.observe(f64::MAX);
        e.observe(-f64::MAX);
        assert_eq!(e.ewma(), f64::NEG_INFINITY);
        assert_eq!(e.std_dev(), f64::INFINITY);
        assert!(e.anomaly_score(0.0).is_nan());
        e.observe(0.0);
        assert!(e.ewma().is_nan(), "−∞ + α·(+∞) = NaN");
        assert!(!e.is_anomaly(0.0), "NaN statistics → verdict false");
    }));
    assert!(r.is_ok(), "EWMA on ±MAX must not panic");

    // oracle: MAD window MAX, −MAX, MAX → sorted −MAX, MAX, MAX → median MAX;
    // deviations |MAX − MAX| = 0, |−MAX − MAX| = +∞, 0 → sorted 0, 0, ∞ →
    // MAD 0 → σ = 0 rule: 0 is anomalous (|0 − MAX| = MAX > 1e-10), MAX not
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut m = MadDetector::new(3.0);
        m.observe(f64::MAX);
        m.observe(-f64::MAX);
        m.observe(f64::MAX);
        assert_eq!(m.median(), f64::MAX);
        assert_eq!(m.mad(), 0.0);
        assert!(m.is_anomaly(0.0));
        assert!(!m.is_anomaly(f64::MAX));
        assert_eq!(m.anomaly_score(-f64::MAX), f64::INFINITY);
    }));
    assert!(r.is_ok(), "MAD on ±MAX must not panic");

    // oracle: the windowed median does no arithmetic on the samples except
    // the even-length average, and (MAX + MAX)/2 = +∞ (the sum overflows
    // before the halving) — documented; the odd-length median is the middle
    // sample itself and stays finite.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut sm = StreamingMedian::new();
        sm.push(f64::MAX);
        sm.push(f64::MAX);
        assert_eq!(sm.median(), f64::INFINITY, "(MAX + MAX) / 2 overflows");
        sm.push(-f64::MAX);
        assert_eq!(sm.median(), f64::MAX, "odd length: the middle sample");
    }));
    assert!(r.is_ok(), "windowed median on ±MAX must not panic");

    // oracle: the composite goes through the callback entry on the same
    // stream — no panic, and no event: each of the three calls is judged
    // with count 0, 1, 2 (< 3 for every detector), so every verdict is
    // `false` regardless of the ±∞ statistics the window now holds.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut c = CompositeDetector::new();
        let mut rec = IndexRecorder { fired: Vec::new() };
        assert!(!c.observe_with_callback(f64::MAX, 0, 0, &mut rec));
        assert!(!c.observe_with_callback(-f64::MAX, 1, 0, &mut rec));
        assert!(!c.observe_with_callback(0.0, 2, 0, &mut rec));
        assert!(rec.fired.is_empty());
    }));
    assert!(r.is_ok(), "composite callback on ±MAX must not panic");
}
