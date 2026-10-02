//! Streaming anomaly detectors on streams whose closed forms are known
//!
//! Runs every detector of `alice_physics::anomaly` (windowed median, MAD,
//! EWMA, Welford Z-score, the composite of the three, and the composite's
//! callback entry) on hand-built streams and prints each reading next to the
//! value the arithmetic predicts, so a drift in any one of them is visible in
//! the log without a test harness.
//!
//! ```bash
//! cargo run --example anomaly_detectors --features std
//! ```

use alice_physics::anomaly::{
    AnomalyCallback, AnomalyEvent, CompositeDetector, EwmaDetector, MadDetector, StreamingMedian,
    ZScoreDetector,
};
use alice_physics::det_math::sqrt64;

/// Collects the index and the event of every sample the composite flags.
struct IndexRecorder {
    fired: Vec<(usize, AnomalyEvent)>,
}

impl AnomalyCallback for IndexRecorder {
    fn on_anomaly(&mut self, event: AnomalyEvent) {
        let index = usize::try_from(event.timestamp).expect("timestamp is the sample index");
        self.fired.push((index, event));
    }
}

/// `(1 − α)ⁿ` by repeated multiplication (no transcendental call).
fn decay_pow(alpha: f64, n: u32) -> f64 {
    let mut p = 1.0;
    for _ in 0..n {
        p *= 1.0 - alpha;
    }
    p
}

fn windowed_median() {
    println!(
        "[anomaly] -- StreamingMedian (window = {}) --",
        StreamingMedian::WINDOW
    );
    let mut sm = StreamingMedian::new();
    println!(
        "[anomaly] empty window: median = {} (documented 0.0), is_full = {}",
        sm.median(),
        sm.is_full()
    );
    for v in [5.0, 2.0, 8.0, 1.0, 9.0] {
        sm.push(v);
    }
    println!(
        "[anomaly] 5 values 5,2,8,1,9: median = {} (sorted 1,2,5,8,9 -> 5)",
        sm.median()
    );
    sm.push(4.0);
    println!(
        "[anomaly] 6 values: median = {} (sorted 1,2,4,5,8,9 -> (4+5)/2 = 4.5)",
        sm.median()
    );
    sm.clear();
    for i in 0..StreamingMedian::WINDOW - 1 {
        sm.push(i as f64);
    }
    println!(
        "[anomaly] after {} pushes: is_full = {} (expected false)",
        StreamingMedian::WINDOW - 1,
        sm.is_full()
    );
    sm.push((StreamingMedian::WINDOW - 1) as f64);
    println!(
        "[anomaly] after {} pushes: is_full = {} (expected true), median = {} (0..=99 -> (49+50)/2 = 49.5)",
        StreamingMedian::WINDOW,
        sm.is_full(),
        sm.median()
    );
    sm.push(1000.0);
    println!(
        "[anomaly] one more push evicts 0: median = {} (1..=99,1000 -> (50+51)/2 = 50.5)",
        sm.median()
    );
}

fn mad_detector() {
    println!("[anomaly] -- MadDetector --");
    let mut mad = MadDetector::new(3.0);
    println!(
        "[anomaly] empty: median = {}, mad = {} (documented 0.0, 0.0), is_anomaly(1e9) = {} (count < 3 -> false), score = {} (-> 0.0)",
        mad.median(),
        mad.mad(),
        mad.is_anomaly(1e9),
        mad.anomaly_score(1e9)
    );
    for v in 1..=9 {
        mad.observe(f64::from(v));
    }
    // 1..=9: median 5; |x − 5| = 4,3,2,1,0,1,2,3,4 → sorted 0,1,1,2,2,3,3,4,4 → MAD 2
    println!(
        "[anomaly] 1..=9: median = {} (5), mad = {} (2), threshold_k = {} (3)",
        mad.median(),
        mad.mad(),
        mad.threshold_k()
    );
    let score13 = 8.0 / (2.0 * 1.4826);
    println!(
        "[anomaly] score(13) = {} (8 / (2 * 1.4826) = {}), is_anomaly(13) = {} (8 > 8.8956 false), is_anomaly(15) = {} (10 > 8.8956 true)",
        mad.anomaly_score(13.0),
        score13,
        mad.is_anomaly(13.0),
        mad.is_anomaly(15.0)
    );
    mad.set_threshold_k(2.0);
    println!(
        "[anomaly] set_threshold_k(2): threshold_k = {} (2), is_anomaly(13) = {} (8 > 5.9304 true), score(13) = {} (unchanged)",
        mad.threshold_k(),
        mad.is_anomaly(13.0),
        mad.anomaly_score(13.0)
    );
    mad.set_threshold_k(0.0);
    println!(
        "[anomaly] k = 0: is_anomaly(5) = {} (0 > 0 false), is_anomaly(5.5) = {} (0.5 > 0 true)",
        mad.is_anomaly(5.0),
        mad.is_anomaly(5.5)
    );
    let mut flat = MadDetector::new(3.0);
    for _ in 0..5 {
        flat.observe(7.0);
    }
    println!(
        "[anomaly] five 7.0: mad = {} (0), is_anomaly(7) = {} (false), is_anomaly(7.1) = {} (true, MAD = 0 rule), score(7.1) = {} (inf)",
        flat.mad(),
        flat.is_anomaly(7.0),
        flat.is_anomaly(7.1),
        flat.anomaly_score(7.1)
    );
}

fn ewma_detector() {
    println!("[anomaly] -- EwmaDetector --");
    let mut ewma = EwmaDetector::new(0.5, 3.0);
    println!(
        "[anomaly] fresh: ewma = {} (0), std_dev = {} (0), alpha = {} (0.5), score(1) = {} (uninitialised -> 0.0)",
        ewma.ewma(),
        ewma.std_dev(),
        ewma.alpha(),
        ewma.anomaly_score(1.0)
    );
    // x₀ = 0 then constant 8, α = 1/2:
    //   ewma_n = 8 − 8·(1/2)ⁿ  → 4, 6, 7
    //   var_n  = (1 − α)(α·d_n² + var_{n−1}), d_n = 8 − ewma_{n−1} → 16, 12, 7
    ewma.observe(0.0);
    let mut var = 0.0;
    let mut prev = 0.0;
    for n in 1..=3u32 {
        ewma.observe(8.0);
        let d = 8.0 - prev;
        var = 0.5 * (0.5 * d * d + var);
        let closed = 8.0 - 8.0 * decay_pow(0.5, n);
        prev = closed;
        println!(
            "[anomaly] step {n}: ewma = {} (8 - 8*(1/2)^{n} = {closed}), std_dev = {} (sqrt({var}) = {})",
            ewma.ewma(),
            ewma.std_dev(),
            sqrt64(var)
        );
    }
    let z9 = 2.0 / sqrt64(7.0);
    println!(
        "[anomaly] score(9) = {} (|9 - 7| / sqrt 7 = {z9}), is_anomaly(9) = {} (k = 3 -> false), threshold_k = {}",
        ewma.anomaly_score(9.0),
        ewma.is_anomaly(9.0),
        ewma.threshold_k()
    );
    ewma.set_threshold_k(0.5);
    println!(
        "[anomaly] set_threshold_k(0.5): is_anomaly(9) = {} (0.756 > 0.5 true), threshold_k = {}",
        ewma.is_anomaly(9.0),
        ewma.threshold_k()
    );
    ewma.set_alpha(0.0);
    println!(
        "[anomaly] set_alpha(0.0): alpha = {} (clamped to 0.001)",
        ewma.alpha()
    );
    ewma.set_alpha(1.0);
    ewma.observe(100.0);
    println!(
        "[anomaly] set_alpha(1.0) then observe(100): ewma = {} (tracks last sample exactly), std_dev = {} ((1 - 1)*... = 0)",
        ewma.ewma(),
        ewma.std_dev()
    );
    ewma.reset();
    println!(
        "[anomaly] reset: ewma = {} (0), std_dev = {} (0), count = {} (0), alpha = {} (kept 1), threshold_k = {} (kept 0.5)",
        ewma.ewma(),
        ewma.std_dev(),
        ewma.count(),
        ewma.alpha(),
        ewma.threshold_k()
    );
}

fn zscore_detector() {
    println!("[anomaly] -- ZScoreDetector --");
    let mut z = ZScoreDetector::new(3.0);
    println!(
        "[anomaly] empty: variance = {} (0), std_dev = {} (0), z_score(0) = {} (0), z_score(1) = {} (inf, sigma = 0 rule), is_anomaly(1) = {} (count < 3 -> false)",
        z.variance(),
        z.std_dev(),
        z.z_score(0.0),
        z.z_score(1.0),
        z.is_anomaly(1.0)
    );
    z.observe(4.0);
    println!(
        "[anomaly] one sample 4: mean = {} (4), variance = {} (count < 2 -> 0)",
        z.mean(),
        z.variance()
    );
    z.reset();
    // 10, 11, 9, 10, 11, 9: mean 10, Σ(x−10)² = 4, sample variance 4/5 = 0.8
    for v in [10.0, 11.0, 9.0, 10.0, 11.0, 9.0] {
        z.observe(v);
    }
    let sigma = sqrt64(0.8);
    println!(
        "[anomaly] 10,11,9,10,11,9: mean = {} (10), variance = {} (0.8), std_dev = {} ({sigma})",
        z.mean(),
        z.variance(),
        z.std_dev()
    );
    println!(
        "[anomaly] z_score(12) = {} (2 / {sigma} = {}), z_score(8) = {} (-{}), is_anomaly(12) = {} (2.236 > 3 false)",
        z.z_score(12.0),
        2.0 / sigma,
        z.z_score(8.0),
        2.0 / sigma,
        z.is_anomaly(12.0)
    );
    let mut tight = ZScoreDetector::new(2.0);
    for v in [10.0, 11.0, 9.0, 10.0, 11.0, 9.0] {
        tight.observe(v);
    }
    println!(
        "[anomaly] k = 2: is_anomaly(12) = {} (2.236 > 2 true)",
        tight.is_anomaly(12.0)
    );
    z.reset();
    println!(
        "[anomaly] reset: mean = {} (0), variance = {} (0), count = {} (0)",
        z.mean(),
        z.variance(),
        z.count()
    );
}

fn composite_detector() {
    println!("[anomaly] -- CompositeDetector --");
    let mut c = CompositeDetector::with_thresholds(2.5, 0.25, 4.5, 1.0);
    println!(
        "[anomaly] with_thresholds(2.5, 0.25, 4.5, 1.0): mad.threshold_k = {} (2.5), ewma.alpha = {} (0.25), ewma.threshold_k = {} (4.5)",
        c.mad.threshold_k(),
        c.ewma.alpha(),
        c.ewma.threshold_k()
    );
    for v in [10.0, 11.0, 9.0, 10.0, 11.0, 9.0] {
        c.observe(v);
    }
    println!(
        "[anomaly] count = {} (6), is_anomaly(12) = {} (Z-score alone fires with k = 1), score(12) = {} (max of the three)",
        c.count(),
        c.is_anomaly(12.0),
        c.anomaly_score(12.0)
    );
    c.require_consensus = true;
    println!(
        "[anomaly] require_consensus: is_anomaly(12) = {} (MAD/EWMA do not fire -> false)",
        c.is_anomaly(12.0)
    );
    c.reset();
    println!(
        "[anomaly] reset: count = {} (0), mad.median = {} (0), ewma.ewma = {} (0), zscore.mean = {} (0)",
        c.count(),
        c.mad.median(),
        c.ewma.ewma(),
        c.zscore.mean()
    );
}

fn callback_stream() {
    println!("[anomaly] -- CompositeDetector::observe_with_callback --");
    // Constant 10 for twelve samples, a spike of 100 at index 12, then 10
    // again. Only the spike is judged against a flat baseline; the samples
    // after it are measured against statistics the spike has shifted.
    let stream: Vec<f64> = (0..16)
        .map(|i| if i == 12 { 100.0 } else { 10.0 })
        .collect();
    let mut c = CompositeDetector::new();
    let mut recorder = IndexRecorder { fired: Vec::new() };
    let mut verdicts = Vec::new();
    for (i, &v) in stream.iter().enumerate() {
        verdicts.push(c.observe_with_callback(v, i as u64, 7, &mut recorder));
    }
    let fired: Vec<usize> = recorder.fired.iter().map(|(i, _)| *i).collect();
    println!("[anomaly] verdicts = {verdicts:?}");
    println!(
        "[anomaly] fired indices = {fired:?} (expected [12]: the spike against a flat baseline)"
    );
    for (i, ev) in &recorder.fired {
        println!(
            "[anomaly] event #{i}: value = {} (100), score = {} (inf: sigma = 0 rule), expected = {} (10), timestamp = {} ({i}), metric_id = {} (7)",
            ev.value, ev.score, ev.expected, ev.timestamp, ev.metric_id
        );
    }
}

fn main() {
    windowed_median();
    mad_detector();
    ewma_detector();
    zscore_detector();
    composite_detector();
    callback_stream();
}
