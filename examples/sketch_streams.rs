//! The four sketches of `alice_physics::sketch` fed from one hand-built stream
//!
//! A stream of eight keys with known frequencies (50, 30, 20, 10, 5, 3, 2, 1)
//! is pushed through the default `CountMinSketch` (1024 × 5), the default
//! `HeavyHitters` (top 10 over the same sketch), the default `DDSketch`
//! (2048 bins) and the default `HyperLogLog` (2¹⁴ registers). Every printed
//! estimate stands next to the closed form it is held to:
//!
//! * Count-Min never under-counts: `estimate(k) ≥ c_k`, and the byte-key path
//!   (`insert_bytes` / `estimate_bytes` on the key's native-endian bytes) is
//!   the same sketch state as the `Hash` path. `error_bound` is `e / WIDTH`
//!   and `confidence` is `1 − e^{−DEPTH}` (`1 − 1 / e⁵` for the default).
//! * `HeavyHitters::top()` lists the keys in descending frequency; with fewer
//!   distinct keys than `K` it lists all of them, each entry's hash is the
//!   FNV-1a digest of the key bytes, and `cms().total()` is the stream length.
//! * `DDSketch::quantile(q)` returns the lower edge of the bucket holding the
//!   rank-`⌈q·n⌉` order statistic, so it lies within
//!   `[v·(1 − 2α/(1+α)), v]` of the true order statistic `v` taken from a
//!   sorted copy of the stream (`α = 0.01`).
//! * `HyperLogLog::cardinality()` of `n` distinct keys is within
//!   `3 × 1.04/√m` of `n` (`m = 16384`, so `1.04/√m = 1.04/128`), re-inserting
//!   the same keys leaves `registers()` unchanged, and the empty sketch is `0`.
//! * `FnvHasher::hash_u64` / `hash_u128` are deterministic and injective on
//!   the key set, and the 16-byte digest differs from the 8-byte one.
//!
//! ```bash
//! cargo run --example sketch_streams --features std
//! ```

use alice_physics::sketch::{
    CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HyperLogLog, Mergeable,
};

/// The stream: `(key, frequency)`; the key values are arbitrary distinct
/// `u64`s, the frequencies are what the sketches have to recover.
const STREAM: [(u64, u64); 8] = [
    (0xA11C_E000_0000_0001, 50),
    (0xA11C_E000_0000_0002, 30),
    (0xA11C_E000_0000_0003, 20),
    (0xA11C_E000_0000_0004, 10),
    (0xA11C_E000_0000_0005, 5),
    (0xA11C_E000_0000_0006, 3),
    (0xA11C_E000_0000_0007, 2),
    (0xA11C_E000_0000_0008, 1),
];

/// `α` handed to `DDSketch::new`: the default 2048-bin sketch is sized for it.
const ALPHA: f64 = 0.01;

fn hashing() {
    println!("[sketch] FnvHasher");
    let mut digests64 = Vec::with_capacity(STREAM.len());
    let mut digests128 = Vec::with_capacity(STREAM.len());
    for (key, _) in STREAM {
        let h64 = FnvHasher::hash_u64(key);
        let h128 = FnvHasher::hash_u128(u128::from(key));
        assert_eq!(h64, FnvHasher::hash_u64(key), "hash_u64 is deterministic");
        assert_eq!(
            h128,
            FnvHasher::hash_u128(u128::from(key)),
            "hash_u128 is deterministic"
        );
        assert_eq!(
            h64,
            FnvHasher::hash_bytes(&key.to_le_bytes()),
            "hash_u64 is the digest of the little-endian bytes"
        );
        assert_ne!(h64, h128, "8-byte and 16-byte encodings hash differently");
        println!("[sketch]   key={key:#018x} hash_u64={h64:#018x} hash_u128={h128:#018x}");
        digests64.push(h64);
        digests128.push(h128);
    }
    digests64.sort_unstable();
    digests64.dedup();
    digests128.sort_unstable();
    digests128.dedup();
    assert_eq!(
        digests64.len(),
        STREAM.len(),
        "hash_u64 injective on the keys"
    );
    assert_eq!(
        digests128.len(),
        STREAM.len(),
        "hash_u128 injective on the keys"
    );
    println!(
        "[sketch]   {} keys -> {} distinct u64 digests, {} distinct u128 digests",
        STREAM.len(),
        digests64.len(),
        digests128.len()
    );
}

fn count_min() -> u64 {
    println!(
        "[sketch] CountMinSketch WIDTH={} DEPTH={}",
        CountMinSketch::WIDTH,
        CountMinSketch::DEPTH
    );
    let mut cms = CountMinSketch::new();
    let mut via_bytes = CountMinSketch::new();
    let mut total = 0u64;
    for (key, count) in STREAM {
        for _ in 0..count {
            cms.insert(&key);
            via_bytes.insert_bytes(&key.to_ne_bytes());
        }
        total += count;
    }
    assert_eq!(cms.total(), total, "total is the stream length");

    // Force at least one Count-Min collision: with WIDTH=1024 and 5000 noise
    // keys the row-minimum of some STREAM key almost certainly rises above
    // its true count, so a mutation that reads the true count instead of
    // calling `estimate` has something to diverge from.
    for noise in 0u64..5000 {
        let k = 0x5EED_0000_0000_0000 ^ noise;
        cms.insert(&k);
        via_bytes.insert_bytes(&k.to_ne_bytes());
    }
    let total_with_noise = total + 5000;
    assert_eq!(cms.total(), total_with_noise, "total tracks the noise too");
    let mut inflated = 0;
    for (key, count) in STREAM {
        let est = cms.estimate(&key);
        let est_bytes = cms.estimate_bytes(&key.to_ne_bytes());
        assert!(
            est >= count,
            "Count-Min never under-counts: {est} < {count}"
        );
        assert_eq!(est, est_bytes, "estimate_bytes agrees with estimate");
        assert_eq!(
            via_bytes.estimate(&key),
            est,
            "a sketch built through insert_bytes estimates the same"
        );
        println!("[sketch]   key={key:#018x} true={count:3} estimate={est:3} (>= true)");
        if est > count {
            inflated += 1;
        }
    }
    assert!(
        inflated > 0,
        "5000 noise keys over 1024 columns raised no estimate: scene has no teeth"
    );
    println!(
        "[sketch]   {inflated} of {} keys inflated by noise collisions",
        STREAM.len()
    );
    let never = 0xDEAD_BEEF_u64;
    println!(
        "[sketch]   never-inserted key estimate={} (>= 0, collisions only)",
        cms.estimate(&never)
    );

    let e = core::f64::consts::E;
    let bound = cms.error_bound();
    let bound_expected = e / CountMinSketch::WIDTH as f64;
    let conf = cms.confidence();
    let conf_expected = 1.0 - 1.0 / (e * e * e * e * e);
    assert!(
        (bound - bound_expected).abs() <= 1e-15,
        "error_bound {bound} != e/WIDTH {bound_expected}"
    );
    assert!(
        (conf - conf_expected).abs() <= 1e-12,
        "confidence {conf} != 1 - e^-DEPTH {conf_expected}"
    );
    println!("[sketch]   error_bound={bound:.6e} (e/WIDTH={bound_expected:.6e})");
    println!("[sketch]   confidence={conf:.9} (1-e^-5={conf_expected:.9})");

    // Merging a clone doubles every count (stream + noise inserted twice).
    let other = cms.clone();
    cms.merge(&other);
    assert_eq!(cms.total(), 2 * total_with_noise);
    for (key, count) in STREAM {
        assert!(cms.estimate(&key) >= 2 * count);
    }
    println!(
        "[sketch]   merged with a clone: total={} (2 x {total_with_noise})",
        cms.total()
    );
    total
}

fn heavy_hitters(total: u64) {
    println!("[sketch] HeavyHitters K={}", HeavyHitters::K);
    let mut hh = HeavyHitters::new();
    for (key, count) in STREAM {
        for _ in 0..count {
            hh.insert_hash(FnvHasher::hash_u64(key));
        }
    }
    let top: Vec<_> = hh.top().collect();
    assert_eq!(
        top.len(),
        STREAM.len(),
        "fewer distinct keys than K: every key is listed"
    );
    for (rank, (entry, (key, count))) in top.iter().zip(STREAM).enumerate() {
        assert_eq!(
            entry.hash,
            FnvHasher::hash_u64(key),
            "rank {rank}: the entry is the key's FNV-1a digest"
        );
        assert!(entry.count >= count, "rank {rank}: never under-counts");
        if rank > 0 {
            assert!(
                top[rank - 1].count >= entry.count,
                "descending frequency order"
            );
        }
        println!(
            "[sketch]   #{rank} key={key:#018x} true={count:3} tracked={:3}",
            entry.count
        );
    }
    assert_eq!(
        hh.cms().total(),
        total,
        "the inner sketch saw the whole stream"
    );
    println!(
        "[sketch]   cms().total()={} (stream length {total})",
        hh.cms().total()
    );
}

fn ddsketch() {
    println!("[sketch] DDSketch BINS={} alpha={ALPHA}", DDSketch::BINS);
    let mut sketch = DDSketch::new(ALPHA);
    assert_eq!(sketch.alpha(), ALPHA);
    assert_eq!(sketch.quantile(0.5), 0.0, "empty sketch: quantile is 0");
    // Values: the stream keys' frequencies as latencies in ms, each repeated
    // by its frequency, so the order statistics are known exactly.
    let mut values = Vec::new();
    for (_, count) in STREAM {
        for i in 0..count {
            let v = (count as f64) * 10.0 + i as f64;
            sketch.insert(v);
            values.push(v);
        }
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    assert_eq!(sketch.count() as usize, n);
    // Lower edge of the bucket: estimate ∈ [v / γ, v], γ = (1+α)/(1−α),
    // and 1 − 1/γ = 2α/(1+α).
    let rel = 2.0 * ALPHA / (1.0 + ALPHA);
    for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0] {
        let rank = ((q * n as f64).ceil() as usize).max(1);
        let truth = values[rank - 1];
        let est = sketch.quantile(q);
        assert!(
            est <= truth && est >= truth * (1.0 - rel),
            "q={q}: estimate {est} outside [{}, {truth}]",
            truth * (1.0 - rel)
        );
        println!(
            "[sketch]   q={q:<4} rank={rank:3} true={truth:7.3} estimate={est:9.5} rel.err={:.5} (<= {rel:.5})",
            (truth - est) / truth
        );
    }
    println!(
        "[sketch]   count={} min={} max={} mean={:.4}",
        sketch.count(),
        sketch.min(),
        sketch.max(),
        sketch.mean()
    );
}

fn hyperloglog() {
    println!("[sketch] HyperLogLog m={} registers", HyperLogLog::M);
    let mut hll = HyperLogLog::new();
    assert_eq!(hll.cardinality(), 0.0, "empty sketch: cardinality 0");
    let n = 10_000u64;
    for i in 0..n {
        hll.insert_bytes(&i.to_ne_bytes());
    }
    let first_pass = hll.registers().to_vec();
    let touched = first_pass.iter().filter(|&&r| r != 0).count();
    let est = hll.cardinality();
    let sigma = 1.04 / 128.0; // 1.04 / sqrt(16384)
    let rel = (est - n as f64).abs() / n as f64;
    assert!(
        rel <= 3.0 * sigma,
        "cardinality {est} off by {rel} > 3 sigma"
    );
    println!(
        "[sketch]   n={n} cardinality={est:.1} rel.err={rel:.5} (3 sigma = {:.5}) registers touched={touched}",
        3.0 * sigma
    );
    // Duplicate keys: the second pass is the identity on the registers.
    for i in 0..n {
        hll.insert(&i);
    }
    assert_eq!(
        hll.registers(),
        &first_pass[..],
        "re-insert leaves registers unchanged"
    );
    assert_eq!(hll.cardinality(), est);
    println!("[sketch]   re-inserted {n} keys: registers unchanged, cardinality={est:.1}");
    // Merging the sketch with a disjoint key range adds the cardinalities.
    let mut other = HyperLogLog::new();
    for i in n..2 * n {
        other.insert(&i);
    }
    hll.merge(&other);
    let merged = hll.cardinality();
    let rel = (merged - 2.0 * n as f64).abs() / (2.0 * n as f64);
    assert!(
        rel <= 3.0 * sigma,
        "merged cardinality {merged} off by {rel}"
    );
    println!("[sketch]   merged with {n} more keys: cardinality={merged:.1} rel.err={rel:.5}");
}

fn main() {
    hashing();
    let total = count_min();
    heavy_hitters(total);
    ddsketch();
    hyperloglog();
    println!("[sketch] done");
}
