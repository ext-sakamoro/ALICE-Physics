//! Every named size of the `alice_physics::sketch` types, each held to its
//! closed form
//!
//! The named sizes (`HyperLogLog10` … `HeavyHitters20`) are aliases of four
//! const-generic types (`HyperLogLogN<M>`, `DDSketchN<BINS>`,
//! `CountMinSketchN<W, D>`, `HeavyHittersN<K, W, D>`). Each alias is named at
//! its own call site below and checked against what its size implies:
//!
//! * `HyperLogLogN<M>`: `P = log2 M`; with every register at 1 (`ρ = 1` for
//!   each index) the raw estimate is `α_m · m² / (m / 2) = 2 α_m m`, with
//!   `α_m = 0.7213 / (1 + 1.079 / m)`; `clear` brings the estimate back to 0.
//! * `DDSketchN<BINS>`: `quantile(q)` is within `2α / (1 + α)` of the
//!   rank-`⌈q·n⌉` order statistic for values inside the bin range (`q > 0`),
//!   `sum` is the exact sum of integer-valued inputs, `clear` empties it.
//! * `CountMinSketchN<W, D>`: `error_bound = e / W`, `confidence = 1 − e^{−D}`,
//!   estimates never under-count, `clear` zeroes every counter.
//! * `HeavyHittersN<K, W, D>`: with fewer than `K` distinct keys `top()` lists
//!   all of them in descending order with exact counts; `clear` empties it.
//!
//! ```bash
//! cargo run --example sketch_sizes --features std
//! ```

use alice_physics::sketch::{
    CountMinSketch1024x5, CountMinSketch2048x7, CountMinSketch4096x5, CountMinSketchN,
    DDSketch1024, DDSketch128, DDSketch2048, DDSketch256, DDSketch512, DDSketchN, FnvHasher,
    HeavyHitters10, HeavyHitters20, HeavyHitters5, HeavyHittersN, HyperLogLog10, HyperLogLog12,
    HyperLogLog14, HyperLogLog16, HyperLogLogN,
};

fn hyperloglog<const M: usize>(name: &str, mut hll: HyperLogLogN<M>, p: usize) {
    assert_eq!(HyperLogLogN::<M>::M, M);
    assert_eq!(HyperLogLogN::<M>::P, p, "{name}: P = log2 M");
    assert_eq!(1usize << p, M, "{name}: M = 2^P");
    // hash = bit 63 | idx: the low P bits pick register idx, and the first set
    // bit of the remaining 64 − P bits is the top one, so ρ = 1.
    for idx in 0..M as u64 {
        hll.insert_hash((1u64 << 63) | idx);
    }
    assert!(
        hll.registers().iter().all(|&r| r == 1),
        "{name}: every ρ = 1"
    );
    let m = M as f64;
    let alpha = 0.7213 / (1.0 + 1.079 / m);
    let expected = 2.0 * alpha * m;
    let got = hll.cardinality();
    assert!(
        ((got - expected) / expected).abs() < 1e-12,
        "{name}: cardinality {got}, closed form 2·α·m = {expected}"
    );
    hll.clear();
    assert_eq!(hll.cardinality(), 0.0, "{name}: cleared sketch estimates 0");
    println!("[sketch_sizes] {name}: M={M} P={p} 2·α·m={expected:.6} got={got:.6}");
}

fn ddsketch<const BINS: usize>(name: &str, mut dd: DDSketchN<BINS>, bins: usize) {
    assert_eq!(DDSketchN::<BINS>::BINS, bins, "{name}: BINS");
    let alpha = dd.alpha();
    // Integer values 1..=200: inside γ^(−BINS/4) .. γ^(3·BINS/4) for each size
    // at its α, and exactly representable, so the sum is exact.
    let n = 200u64;
    for v in (1..=n).rev() {
        dd.insert(v as f64);
    }
    assert_eq!(dd.count(), n, "{name}: count");
    assert_eq!(dd.sum(), (n * (n + 1) / 2) as f64, "{name}: exact sum");
    let bound = 2.0 * alpha / (1.0 + alpha) + 1e-12;
    let mut worst = 0.0f64;
    for r in 1..=n {
        let q = (r as f64 - 0.5) / n as f64;
        let v = r as f64; // rank-r order statistic of 1..=n
        let rel = (dd.quantile(q) - v).abs() / v;
        assert!(
            rel <= bound,
            "{name}: rank {r} relative error {rel} > {bound}"
        );
        worst = worst.max(rel);
    }
    dd.clear();
    assert_eq!(dd.count(), 0, "{name}: cleared count");
    assert_eq!(dd.sum(), 0.0, "{name}: cleared sum");
    println!("[sketch_sizes] {name}: BINS={BINS} α={alpha} worst relative error={worst:.6} bound={bound:.6}");
}

fn count_min<const W: usize, const D: usize>(
    name: &str,
    mut cms: CountMinSketchN<W, D>,
    width: usize,
    depth: usize,
) {
    assert_eq!(CountMinSketchN::<W, D>::WIDTH, width, "{name}: WIDTH");
    assert_eq!(CountMinSketchN::<W, D>::DEPTH, depth, "{name}: DEPTH");
    let eps = core::f64::consts::E / width as f64;
    // e^{−D} as D divisions by e (independent of det_math::exp64)
    let conf = 1.0 - (0..depth).fold(1.0f64, |acc, _| acc / core::f64::consts::E);
    assert!((cms.error_bound() - eps).abs() < 1e-15, "{name}: e / W");
    assert!(
        (cms.confidence() - conf).abs() < 1e-12,
        "{name}: 1 − e^(−D)"
    );
    for k in 0..300u64 {
        cms.insert_hash(FnvHasher::hash_u64(k), k % 7 + 1);
    }
    for k in 0..300u64 {
        assert!(
            cms.estimate_hash(FnvHasher::hash_u64(k)) > k % 7,
            "{name}: never under-counts"
        );
    }
    cms.clear();
    assert_eq!(cms.total(), 0, "{name}: cleared total");
    assert!(
        (0..300u64).all(|k| cms.estimate_hash(FnvHasher::hash_u64(k)) == 0),
        "{name}: cleared counters"
    );
    println!("[sketch_sizes] {name}: W={W} D={D} e/W={eps:.3e} 1−e^(−D)={conf:.6}");
}

fn heavy_hitters<const K: usize, const W: usize, const D: usize>(
    name: &str,
    mut hh: HeavyHittersN<K, W, D>,
    k: usize,
) {
    assert_eq!(HeavyHittersN::<K, W, D>::K, k, "{name}: K");
    // K − 1 distinct keys, key j inserted 2j + 1 times, interleaved.
    let keys = k - 1;
    let max_count = 2 * keys - 1;
    for round in 0..max_count {
        for j in 0..keys {
            if round < 2 * j + 1 {
                hh.insert_hash(FnvHasher::hash_u64(1000 + j as u64));
            }
        }
    }
    let top: Vec<_> = hh.top().copied().collect();
    assert_eq!(top.len(), keys, "{name}: every key listed");
    for (pos, entry) in top.iter().enumerate() {
        let j = keys - 1 - pos;
        assert_eq!(
            entry.hash,
            FnvHasher::hash_u64(1000 + j as u64),
            "{name}: order"
        );
        assert_eq!(entry.count, 2 * j as u64 + 1, "{name}: exact count");
    }
    hh.clear();
    assert_eq!(hh.top().count(), 0, "{name}: cleared top");
    assert_eq!(hh.cms().total(), 0, "{name}: cleared sketch");
    println!("[sketch_sizes] {name}: K={K} on {W}x{D}, top {keys} exact");
}

fn main() {
    hyperloglog("HyperLogLog10", HyperLogLog10::new(), 10);
    hyperloglog("HyperLogLog12", HyperLogLog12::new(), 12);
    hyperloglog("HyperLogLog14", HyperLogLog14::new(), 14);
    hyperloglog("HyperLogLog16", HyperLogLog16::new(), 16);

    ddsketch("DDSketch128", DDSketch128::new(0.1), 128);
    ddsketch("DDSketch256", DDSketch256::new(0.05), 256);
    ddsketch("DDSketch512", DDSketch512::new(0.02), 512);
    ddsketch("DDSketch1024", DDSketch1024::new(0.02), 1024);
    ddsketch("DDSketch2048", DDSketch2048::new(0.01), 2048);

    count_min("CountMinSketch1024x5", CountMinSketch1024x5::new(), 1024, 5);
    count_min("CountMinSketch2048x7", CountMinSketch2048x7::new(), 2048, 7);
    count_min("CountMinSketch4096x5", CountMinSketch4096x5::new(), 4096, 5);

    heavy_hitters("HeavyHitters5", HeavyHitters5::new(), 5);
    heavy_hitters("HeavyHitters10", HeavyHitters10::new(), 10);
    heavy_hitters("HeavyHitters20", HeavyHitters20::new(), 20);

    println!("[sketch_sizes] done");
}
