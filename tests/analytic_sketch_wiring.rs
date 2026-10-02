//! Oracles for the production entry points of `alice_physics::sketch`
//! (`examples/sketch_streams.rs`): `FnvHasher::{hash_u64, hash_u128}`,
//! `CountMinSketch::{WIDTH, DEPTH, insert_bytes, estimate, estimate_bytes,
//! error_bound, confidence}`, `HeavyHitters::{K, top, cms}`,
//! `DDSketch::{BINS, alpha, quantile}` and `HyperLogLog::{insert_bytes,
//! registers, cardinality}`.
//!
//! # Expected values and where they come from
//!
//! * **Hashes**: the test carries its own FNV-1a 64 (Fowler/Noll/Vo: offset
//!   basis `0xcbf29ce484222325`, prime `0x100000001b3`) and the MurmurHash3
//!   `fmix64` finaliser (Appleby: shifts of 33, multipliers
//!   `0xff51afd7ed558ccd` / `0xc4ceb9fe1a85ec53`). `hash_bytes(b)` is
//!   `fmix64(fnv1a(b))`, `hash_u64(v)` is that digest of `v.to_le_bytes()`,
//!   `hash_u128(v)` of the 16 little-endian bytes. The published FNV-1a test
//!   vector `fnv1a(b"a") = 0xaf63dc4c8601ec8c` anchors the reference itself.
//!   `Hash for u64` writes `to_ne_bytes`, so the `Hash` path and the
//!   `insert_bytes(&v.to_ne_bytes())` path must give the same sketch state.
//! * **Count-Min** (Cormode & Muthukrishnan 2005): the estimate of a key is
//!   the minimum over rows of the counter it hashes to, i.e. the key's own
//!   count plus the mass of every other key sharing that column, minimised
//!   over rows. The test keeps a reference `D × W` table driven by the
//!   module's documented row law (golden-ratio stride `0x9e3779b97f4a7c15`
//!   per row, then the `fmix64` first half, modulo `W`), so the expected
//!   estimate is exact including collisions; independently of that law the
//!   one-sided bound `estimate ≥ count` must hold. `error_bound = e / W` and
//!   `confidence = 1 − e^{−D}` are the module's documented closed forms;
//!   `e^{−5}` is `1 / e⁵` by multiplication.
//! * **Heavy hitters**: with at most `K` distinct keys every key is listed;
//!   with more, the `K` most frequent are, in descending frequency, each
//!   entry carrying the key's digest and the Count-Min estimate taken at the
//!   key's last insertion (which the reference table reproduces).
//! * **DDSketch** (Masson, Lee & Rim 2019): bucket `i` covers
//!   `(γ^{i−1}, γ^i]` with `γ = (1+α)/(1−α)`. `quantile(q)` returns the
//!   lower edge `γ^{i−1}` of the bucket holding the rank-`⌈q·n⌉` order
//!   statistic `v` (for `v < 0`, minus the edge for `|v|`), so
//!   `|v|/γ ≤ |est| ≤ |v|` with the sign of `v`, and a rank inside the zero
//!   block gives exactly `0`. The order statistics come from the test's own
//!   sort. The paper's `α` guarantee needs the bucket mid-point estimator;
//!   the module returns the edge, whose worst case is `1 − 1/γ = 2α/(1+α)`
//!   (now documented on `quantile` itself) —
//!   `ddsketch_quantile_meets_the_edge_estimator_bound` holds the module to
//!   that doubled bound.
//! * **HyperLogLog** (Flajolet et al. 2007, Heule et al. 2013): register
//!   `j = hash & (m−1)` keeps the maximum of `ρ(hash >> P)`, the 1-based
//!   position of the leftmost set bit of the remaining `64−P` bits (`64−P+1`
//!   when they are all zero). The test rebuilds the register array from its
//!   own digests and compares bit for bit, then evaluates the published
//!   estimator (`α_m m² / Σ 2^{−M[j]}`, replaced by linear counting
//!   `m·ln(m/V)` when `E ≤ 2.5m` and `V > 0` registers are empty). The
//!   external check is the documented standard error `1.04/√m`: with
//!   `m = 2¹⁴`, `√m = 128` and `n` distinct keys estimate within `3σ`. One
//!   register set gives `m·ln(m/(m−1)) ∈ [1, 1 + 1/m]` (series bound, no
//!   transcendental needed).
//!
//! # Degenerate input (each result is pinned, panics are measured)
//!
//! Empty sketches: Count-Min estimates `0`, `top()` is empty, `quantile` is
//! `0.0` (documented early return) with `min = +∞`, `max = −∞`, `mean = 0`,
//! and the cardinality is exactly `0`. Zero-length byte keys are ordinary
//! keys (their digest is `fmix64(offset basis)`). `quantile(0.0)` (and any
//! `q` whose rank rounds to `0`, including negative and NaN) returns the
//! lower edge of the outermost negative bucket, `−γ^{BINS − offset − 2}`,
//! not the minimum; `q > 1` returns `max`. `insert(NaN)` lands in the zero
//! block; `insert(±∞)` overflows the `i32` bucket index and panics in a
//! debug build; `DDSketch::new(0.0)` makes `ln γ = 0` and `insert(2.0)`
//! panics the same way. `CountMinSketch::insert_hash` saturates the
//! counters but adds to `total` with plain `+=`, which panics in a debug
//! build once the total passes `u64::MAX`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::det_math::ln64;
use alice_physics::sketch::{
    CountMinSketch, DDSketch, FnvHasher, HeavyHitters, HeavyHitters5, HyperLogLog, HyperLogLog10,
    Mergeable,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

// ---------------------------------------------------------------------------
// Reference implementations (never call the crate)
// ---------------------------------------------------------------------------

/// FNV-1a 64-bit (Fowler/Noll/Vo).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// MurmurHash3 `fmix64` finaliser (Appleby).
fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^= h >> 33;
    h
}

/// The module's digest: `fmix64(fnv1a(bytes))`.
fn digest(bytes: &[u8]) -> u64 {
    fmix64(fnv1a(bytes))
}

/// Digest of a `u64` key as `Hash for u64` feeds it (native-endian bytes).
fn key_digest(key: u64) -> u64 {
    digest(&key.to_ne_bytes())
}

/// Digest of a `u64` key as `FnvHasher::hash_u64` defines it (little-endian).
fn le_digest(key: u64) -> u64 {
    digest(&key.to_le_bytes())
}

/// Count-Min row law documented by the module (pin, not an external truth):
/// golden-ratio stride per row, the first half of `fmix64`, modulo width.
fn cms_column(hash: u64, row: usize, width: usize) -> usize {
    let h = hash.wrapping_add((row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mixed = h ^ (h >> 33);
    let mixed = mixed.wrapping_mul(0xff51_afd7_ed55_8ccd);
    let mixed = mixed ^ (mixed >> 33);
    (mixed as usize) % width
}

/// Reference Count-Min table (Cormode & Muthukrishnan 2005).
struct RefCms {
    rows: Vec<Vec<u64>>,
}

impl RefCms {
    fn new(width: usize, depth: usize) -> Self {
        Self {
            rows: vec![vec![0u64; width]; depth],
        }
    }

    fn width(&self) -> usize {
        self.rows[0].len()
    }

    fn insert(&mut self, hash: u64, count: u64) {
        let w = self.width();
        for (r, row) in self.rows.iter_mut().enumerate() {
            row[cms_column(hash, r, w)] += count;
        }
    }

    fn estimate(&self, hash: u64) -> u64 {
        let w = self.width();
        self.rows
            .iter()
            .enumerate()
            .map(|(r, row)| row[cms_column(hash, r, w)])
            .min()
            .expect("depth > 0")
    }

    /// True when no other inserted digest shares every row's column with
    /// `hash` — then the estimate is exactly the key's own count.
    fn collision_free(&self, hash: u64, own: u64) -> bool {
        self.estimate(hash) == own
    }
}

/// Reference HyperLogLog register array for `P` index bits.
fn ref_hll_registers(p: u32, hashes: &[u64]) -> Vec<u8> {
    let m = 1usize << p;
    let mut regs = vec![0u8; m];
    for &h in hashes {
        let j = (h as usize) & (m - 1);
        let w = h >> p;
        let rho = if w == 0 {
            64 - p + 1
        } else {
            w.leading_zeros() - p + 1
        } as u8;
        if rho > regs[j] {
            regs[j] = rho;
        }
    }
    regs
}

/// Published HLL estimator with the small-range linear-counting switch.
fn ref_hll_cardinality(regs: &[u8]) -> f64 {
    let m = regs.len() as f64;
    let alpha = 0.7213 / (1.0 + 1.079 / m);
    let sum: f64 = regs.iter().map(|&r| 1.0 / (1u64 << r) as f64).sum();
    let zeros = regs.iter().filter(|&&r| r == 0).count();
    let raw = alpha * m * m / sum;
    if raw <= 2.5 * m && zeros > 0 {
        m * ln64(m / zeros as f64)
    } else {
        raw
    }
}

/// The stream shared with the example: `(key, frequency)`.
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

const ALPHA: f64 = 0.01;

fn gamma(alpha: f64) -> f64 {
    (1.0 + alpha) / (1.0 - alpha)
}

// ---------------------------------------------------------------------------
// FnvHasher
// ---------------------------------------------------------------------------

#[test]
fn fnv_digests_are_fmix64_of_fnv1a_and_injective_on_the_key_set() {
    // Anchor the reference on the published FNV-1a vector.
    assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(FnvHasher::hash_bytes(b"a"), fmix64(0xaf63_dc4c_8601_ec8c));
    assert_eq!(FnvHasher::hash_bytes(b""), fmix64(0xcbf2_9ce4_8422_2325));

    let mut seen64 = std::collections::BTreeSet::new();
    let mut seen128 = std::collections::BTreeSet::new();
    for i in 0..64u64 {
        let key = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA11C_E000_0000_0000;
        let h64 = FnvHasher::hash_u64(key);
        let h128 = FnvHasher::hash_u128(u128::from(key));
        assert_eq!(h64, digest(&key.to_le_bytes()), "hash_u64({key:#x})");
        assert_eq!(
            h128,
            digest(&u128::from(key).to_le_bytes()),
            "hash_u128({key:#x})"
        );
        assert_eq!(h64, FnvHasher::hash_u64(key), "determinism");
        assert_eq!(h128, FnvHasher::hash_u128(u128::from(key)), "determinism");
        assert_ne!(h64, h128, "8 and 16 byte encodings differ");
        seen64.insert(h64);
        seen128.insert(h128);
    }
    assert_eq!(
        seen64.len(),
        64,
        "64 distinct keys -> 64 distinct u64 digests"
    );
    assert_eq!(
        seen128.len(),
        64,
        "64 distinct keys -> 64 distinct u128 digests"
    );
    // The high half of a u128 key reaches the digest.
    assert_ne!(
        FnvHasher::hash_u128(1u128 << 64),
        FnvHasher::hash_u128(0),
        "high 64 bits of the u128 key are hashed"
    );
    assert_eq!(
        FnvHasher::hash_u128(1u128 << 64),
        digest(&(1u128 << 64).to_le_bytes())
    );
}

// ---------------------------------------------------------------------------
// CountMinSketch
// ---------------------------------------------------------------------------

#[test]
fn count_min_estimate_is_the_row_minimum_of_the_reference_table() {
    assert_eq!(CountMinSketch::WIDTH, 1024);
    assert_eq!(CountMinSketch::DEPTH, 5);
    let mut cms = CountMinSketch::new();
    let mut via_bytes = CountMinSketch::new();
    let mut reference = RefCms::new(CountMinSketch::WIDTH, CountMinSketch::DEPTH);
    let mut total = 0;
    for (key, count) in STREAM {
        for _ in 0..count {
            cms.insert(&key);
            via_bytes.insert_bytes(&key.to_ne_bytes());
        }
        reference.insert(key_digest(key), count);
        total += count;
    }
    assert_eq!(cms.total(), total);
    assert_eq!(via_bytes.total(), total);

    let mut collision_free = 0;
    for (key, count) in STREAM {
        let expected = reference.estimate(key_digest(key));
        assert!(expected >= count, "reference table is itself one-sided");
        let est = cms.estimate(&key);
        assert!(est >= count, "one-sided bound: {est} < {count}");
        assert_eq!(est, expected, "key {key:#x}: estimate vs reference table");
        assert_eq!(cms.estimate_bytes(&key.to_ne_bytes()), est, "byte path");
        assert_eq!(via_bytes.estimate(&key), est, "sketch built through bytes");
        assert_eq!(via_bytes.estimate_bytes(&key.to_ne_bytes()), est);
        if reference.collision_free(key_digest(key), count) {
            // External closed form: no collision in some row → exact.
            assert_eq!(est, count);
            collision_free += 1;
        }
    }
    // The scene has teeth only if the exact branch fires for most keys.
    assert!(
        collision_free >= 6,
        "{collision_free} of 8 keys are collision-free; rebuild the stream"
    );
    // Keys never inserted: the table predicts exactly what the sketch says,
    // and a `Hash`-path estimate equals the byte-path estimate for them too.
    for i in 1..=16u64 {
        let key = 0xDEAD_BEEF_0000_0000 + i;
        let expected = reference.estimate(key_digest(key));
        assert_eq!(cms.estimate(&key), expected, "absent key {key:#x}");
        assert_eq!(cms.estimate_bytes(&key.to_ne_bytes()), expected);
    }
}

/// 3000 keys in 1024 columns collide in every row; the reference table (the
/// module's documented row law) predicts every estimate exactly, present or
/// absent, and the one-sided bound still holds for all of them.
#[test]
fn count_min_with_forced_collisions_matches_the_reference_table_exactly() {
    let mut cms = CountMinSketch::new();
    let mut reference = RefCms::new(CountMinSketch::WIDTH, CountMinSketch::DEPTH);
    let n = 3000u64;
    for i in 0..n {
        let count = 1 + (i % 7);
        for _ in 0..count {
            cms.insert(&i);
        }
        reference.insert(key_digest(i), count);
    }
    assert_eq!(cms.total(), (0..n).map(|i| 1 + (i % 7)).sum::<u64>());
    let mut inflated = 0;
    for i in 0..n {
        let count = 1 + (i % 7);
        let est = cms.estimate(&i);
        assert!(est >= count, "key {i}: {est} < {count}");
        assert_eq!(est, reference.estimate(key_digest(i)), "key {i}");
        assert_eq!(cms.estimate_bytes(&i.to_ne_bytes()), est, "key {i} bytes");
        if est > count {
            inflated += 1;
        }
    }
    // With 3000 keys over 1024 columns some key is inflated in every row.
    assert!(
        inflated > 0,
        "no collision reached an estimate: scene has no teeth"
    );
    for i in n..n + 200 {
        assert_eq!(
            cms.estimate(&i),
            reference.estimate(key_digest(i)),
            "absent {i}"
        );
    }
}

#[test]
fn count_min_error_bound_and_confidence_are_the_documented_closed_forms() {
    let e = core::f64::consts::E;
    let mut cms = CountMinSketch::new();
    // e / 1024 is an exact power-of-two scaling: bit-for-bit.
    assert_eq!(cms.error_bound(), e / 1024.0);
    assert_eq!(cms.error_bound(), e / CountMinSketch::WIDTH as f64);
    let one_minus_e_minus_5 = 1.0 - 1.0 / (e * e * e * e * e);
    assert!(
        (cms.confidence() - one_minus_e_minus_5).abs() <= 1e-12,
        "confidence {} vs 1 - 1/e^5 = {one_minus_e_minus_5}",
        cms.confidence()
    );
    assert!(cms.confidence() > 0.993 && cms.confidence() < 0.9933);
    // Both are properties of the shape, not of the contents.
    let (b0, c0) = (cms.error_bound(), cms.confidence());
    for (key, count) in STREAM {
        for _ in 0..count {
            cms.insert(&key);
        }
    }
    assert_eq!(cms.error_bound(), b0);
    assert_eq!(cms.confidence(), c0);
    cms.clear();
    assert_eq!(cms.error_bound(), b0);
    assert_eq!(cms.confidence(), c0);
}

#[test]
fn count_min_duplicate_passes_double_every_estimate_and_merge_adds() {
    let mut once = CountMinSketch::new();
    for (key, count) in STREAM {
        for _ in 0..count {
            once.insert_bytes(&key.to_ne_bytes());
        }
    }
    let mut twice = once.clone();
    for (key, count) in STREAM {
        for _ in 0..count {
            twice.insert_bytes(&key.to_ne_bytes());
        }
    }
    let mut merged = once.clone();
    merged.merge(&once);
    assert_eq!(twice.total(), 2 * once.total());
    assert_eq!(merged.total(), 2 * once.total());
    // Every counter doubles, so the row minimum doubles — exactly, with or
    // without collisions.
    for (key, _) in STREAM {
        let one = once.estimate(&key);
        assert_eq!(twice.estimate(&key), 2 * one, "key {key:#x} twice");
        assert_eq!(merged.estimate(&key), 2 * one, "key {key:#x} merged");
        assert_eq!(twice.estimate_bytes(&key.to_ne_bytes()), 2 * one);
    }
    for i in 1..=8u64 {
        let absent = 0xDEAD_BEEF_0000_0000 + i;
        assert_eq!(twice.estimate(&absent), 2 * once.estimate(&absent));
    }
}

#[test]
fn count_min_degenerate_inputs() {
    // Empty sketch: every estimate is 0, the bounds are already defined.
    let cms = CountMinSketch::new();
    assert_eq!(cms.total(), 0);
    assert_eq!(cms.estimate(&0u64), 0);
    assert_eq!(cms.estimate(&u64::MAX), 0);
    assert_eq!(cms.estimate_bytes(&[]), 0);
    assert_eq!(cms.estimate_bytes(b"anything"), 0);
    assert_eq!(cms.error_bound(), core::f64::consts::E / 1024.0);

    // Zero-length key: an ordinary key with digest fmix64(offset basis);
    // alone in the sketch every counter it touches is its own, so exact.
    let mut cms = CountMinSketch::new();
    for _ in 0..3 {
        cms.insert_bytes(&[]);
    }
    assert_eq!(cms.total(), 3);
    assert_eq!(cms.estimate_bytes(&[]), 3);
    assert_eq!(
        cms.estimate_hash(fmix64(0xcbf2_9ce4_8422_2325)),
        3,
        "the empty key's digest is fmix64 of the FNV offset basis"
    );
    assert_eq!(cms.estimate_bytes(&[0]), 0, "[0] is a different key");

    // Extreme counts: counters saturate at u64::MAX ...
    let mut cms = CountMinSketch::new();
    let h = key_digest(7);
    cms.insert_hash(h, u64::MAX);
    assert_eq!(cms.estimate_hash(h), u64::MAX);
    assert_eq!(cms.total(), u64::MAX);
    // ... but `total` is a plain `+=`: the next insertion overflows it.
    // Measured contract: panic in a debug build ("attempt to add with
    // overflow"), wrap-around in release.
    let r = catch_unwind(AssertUnwindSafe(|| {
        cms.insert_hash(h, 1);
        cms.total()
    }));
    if cfg!(debug_assertions) {
        assert!(r.is_err(), "total overflow panics in debug builds");
    } else {
        assert_eq!(r.expect("release wraps"), 0, "total wraps in release");
    }
}

// ---------------------------------------------------------------------------
// HeavyHitters
// ---------------------------------------------------------------------------

#[test]
fn heavy_hitters_top_lists_the_k_most_frequent_keys_in_descending_order() {
    assert_eq!(HeavyHitters::K, 10);
    assert_eq!(HeavyHitters5::K, 5);
    // Block order: each key's whole run, most frequent first.
    let mut block = HeavyHitters5::new();
    let mut reference = RefCms::new(1024, 5);
    let mut expected_counts = Vec::new();
    for (key, count) in STREAM {
        for _ in 0..count {
            block.insert_hash(FnvHasher::hash_u64(key));
        }
        reference.insert(le_digest(key), count);
        // The tracked count is the estimate at the key's last insertion:
        // the reference table at this point in the stream.
        expected_counts.push(reference.estimate(le_digest(key)));
    }
    let top: Vec<_> = block.top().collect();
    assert_eq!(top.len(), 5, "8 distinct keys > K = 5: exactly K listed");
    for (rank, entry) in top.iter().enumerate() {
        let (key, count) = STREAM[rank];
        assert_eq!(entry.hash, le_digest(key), "rank {rank} is key {key:#x}");
        assert_eq!(
            entry.count, expected_counts[rank],
            "rank {rank} tracked count"
        );
        assert!(entry.count >= count, "rank {rank} never under-counts");
        if rank > 0 {
            assert!(
                top[rank - 1].count >= entry.count,
                "descending at rank {rank}"
            );
        }
    }
    assert_eq!(
        block.cms().total(),
        121,
        "inner sketch saw the whole stream"
    );
    assert_eq!(
        block.cms().estimate_bytes(&STREAM[0].0.to_le_bytes()),
        expected_counts[0]
    );

    // Round-robin order: the same keys reach the same list (the argument
    // order moved, the answer did not).
    let mut robin = HeavyHitters5::new();
    let max = STREAM.iter().map(|&(_, c)| c).max().unwrap();
    for round in 0..max {
        for (key, count) in STREAM {
            if round < count {
                robin.insert_hash(FnvHasher::hash_u64(key));
            }
        }
    }
    let robin_top: Vec<_> = robin.top().copied().collect();
    let block_top: Vec<_> = top.iter().map(|e| **e).collect();
    assert_eq!(
        robin_top, block_top,
        "top list is independent of arrival order"
    );
    assert_eq!(robin.cms().total(), 121);

    // Reverse block order: the five light keys fill the tracker first and
    // each heavy key has to evict the current minimum on its way up. The
    // final list is the same (eviction fires when the estimate exceeds the
    // tracked minimum, which it does on the heavy keys' second insertion).
    let mut reverse = HeavyHitters5::new();
    for (key, count) in STREAM.iter().rev() {
        for _ in 0..*count {
            reverse.insert_hash(FnvHasher::hash_u64(*key));
        }
    }
    let reverse_top: Vec<_> = reverse.top().copied().collect();
    assert_eq!(reverse_top, block_top, "heavy keys arriving last still win");
    assert_eq!(reverse.cms().total(), 121);
}

#[test]
fn heavy_hitters_degenerate_inputs() {
    let mut hh = HeavyHitters::new();
    assert_eq!(hh.top().count(), 0, "empty tracker lists nothing");
    assert_eq!(hh.cms().total(), 0);

    // Fewer distinct keys than K: all of them, descending.
    for (key, count) in STREAM.iter().take(3) {
        for _ in 0..*count {
            hh.insert_hash(FnvHasher::hash_bytes(&key.to_le_bytes()));
        }
    }
    let top: Vec<_> = hh.top().collect();
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].hash, le_digest(STREAM[0].0));
    assert_eq!(top[1].hash, le_digest(STREAM[1].0));
    assert_eq!(top[2].hash, le_digest(STREAM[2].0));
    assert!(top[0].count >= top[1].count && top[1].count >= top[2].count);
    assert_eq!(hh.cms().total(), 100);

    // Zero-length key and the extreme key are ordinary keys.
    hh.insert_hash(FnvHasher::hash_bytes(&[]));
    hh.insert_hash(FnvHasher::hash_u64(u64::MAX));
    assert_eq!(hh.top().count(), 5);
    assert!(hh.top().any(|e| e.hash == fmix64(0xcbf2_9ce4_8422_2325)));
    assert!(hh.top().any(|e| e.hash == le_digest(u64::MAX)));

    // A single key inserted far more than K times still yields one entry.
    let mut single = HeavyHitters5::new();
    for _ in 0..1000 {
        single.insert_hash(FnvHasher::hash_u64(42));
    }
    let top: Vec<_> = single.top().collect();
    assert_eq!(top.len(), 1);
    assert_eq!(top[0].hash, le_digest(42));
    assert_eq!(top[0].count, 1000);
    assert_eq!(single.cms().estimate_bytes(&42u64.to_le_bytes()), 1000);

    hh.clear();
    assert_eq!(hh.top().count(), 0);
    assert_eq!(hh.cms().total(), 0);
}

// ---------------------------------------------------------------------------
// DDSketch
// ---------------------------------------------------------------------------

/// Sorted sample with negatives, a zero block and positives.
fn sample() -> Vec<f64> {
    let mut v = Vec::new();
    for i in 1..=50 {
        v.push(-(i as f64) * 0.37);
    }
    v.extend(std::iter::repeat_n(0.0, 5));
    for i in 1..=200 {
        v.push(i as f64);
    }
    for i in 0..40 {
        v.push(1000.0 + 3.1 * i as f64 + 0.123);
    }
    v
}

fn true_order_statistic(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let rank = ((q * n as f64).ceil() as usize).max(1).min(n);
    sorted[rank - 1]
}

#[test]
fn ddsketch_quantiles_are_the_bucket_edge_within_gamma_of_the_order_statistic() {
    assert_eq!(DDSketch::BINS, 2048);
    let mut sketch = DDSketch::new(ALPHA);
    assert_eq!(sketch.alpha(), ALPHA);
    let values = sample();
    let mut sum = 0.0;
    for &v in &values {
        sketch.insert(v);
        sum += v;
    }
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    assert_eq!(sketch.count() as usize, sorted.len());
    assert_eq!(sketch.min(), sorted[0]);
    assert_eq!(sketch.max(), sorted[sorted.len() - 1]);
    assert!((sketch.sum() - sum).abs() <= 1e-9 * sum.abs());
    assert!((sketch.mean() - sum / sorted.len() as f64).abs() <= 1e-9);

    let g = gamma(ALPHA);
    let mut worst = 0.0f64;
    let mut q = 0.001;
    while q <= 1.0 {
        let truth = true_order_statistic(&sorted, q);
        let est = sketch.quantile(q);
        if truth == 0.0 {
            assert_eq!(est, 0.0, "q={q}: rank in the zero block");
        } else {
            assert_eq!(
                est.is_sign_negative(),
                truth.is_sign_negative(),
                "q={q} sign"
            );
            let (a, t) = (est.abs(), truth.abs());
            assert!(
                a <= t && a >= t / g,
                "q={q}: |est| {a} outside [{}, {t}]",
                t / g
            );
            worst = worst.max((t - a) / t);
        }
        q += 0.001;
    }
    // The edge estimator's worst case 2α/(1+α) is actually approached by a
    // sample this dense (teeth for a mid-point mutation too).
    let edge_worst = 2.0 * ALPHA / (1.0 + ALPHA);
    assert!(worst <= edge_worst, "worst {worst} > {edge_worst}");
    assert!(
        worst > ALPHA,
        "worst {worst} never exceeds α: the scene has no teeth"
    );
}

/// The module documents `|v − true| ≤ α · true`; the edge estimator cannot
/// deliver it (see the module doc of this file). Measured 2026-10-03:
/// `quantile` returns the bucket's lower edge, not the paper's mid-point
/// estimator, so its honest guarantee is `2α/(1+α)` (doc'd on the function,
/// Backlog `sketch-quantile-edge-vs-midpoint`), not `α` itself. 527 of 1000
/// quantiles over a naive `α` bound confirmed the edge form is needed; 0
/// violate the doubled bound.
#[test]
fn ddsketch_quantile_meets_the_edge_estimator_bound() {
    let mut sketch = DDSketch::new(ALPHA);
    let values = sample();
    for &v in &values {
        sketch.insert(v);
    }
    let mut sorted = values;
    sorted.sort_by(f64::total_cmp);
    let edge_bound = 2.0 * ALPHA / (1.0 + ALPHA);
    let mut q = 0.001;
    let mut violations = Vec::new();
    while q <= 1.0 {
        let truth = true_order_statistic(&sorted, q);
        let est = sketch.quantile(q);
        if (est - truth).abs() > edge_bound * truth.abs() {
            violations.push((q, truth, est));
        }
        q += 0.001;
    }
    assert!(
        violations.is_empty(),
        "{} of 1000 quantiles exceed the edge bound 2α/(1+α)={edge_bound}: first {:?}",
        violations.len(),
        &violations[..violations.len().min(3)]
    );
}

#[test]
fn ddsketch_degenerate_inputs() {
    // Empty sketch: documented early return 0.0, extreme min / max.
    let sketch = DDSketch::new(ALPHA);
    for q in [0.0, 0.5, 1.0, 2.0, -1.0] {
        assert_eq!(sketch.quantile(q), 0.0, "empty sketch q={q}");
    }
    assert_eq!(sketch.count(), 0);
    assert_eq!(sketch.sum(), 0.0);
    assert_eq!(sketch.mean(), 0.0);
    assert_eq!(sketch.min(), f64::INFINITY);
    assert_eq!(sketch.max(), f64::NEG_INFINITY);
    assert_eq!(sketch.alpha(), ALPHA);

    // Positive data: q = 0 (rank 0) is answered by the outermost negative
    // bucket's edge, −γ^{BINS − offset − 2} with offset = BINS / 4, not the
    // minimum; q > 1 falls through to `max`; q < 0 and NaN round to rank 0.
    let mut sketch = DDSketch::new(ALPHA);
    for i in 1..=100 {
        sketch.insert(i as f64);
    }
    let g = gamma(ALPHA);
    let offset = DDSketch::BINS / 4;
    let outer_edge = -alice_physics::det_math::powf64(g, (DDSketch::BINS - offset - 2) as f64);
    let q0 = sketch.quantile(0.0);
    assert!(
        (q0 - outer_edge).abs() <= 1e-6 * outer_edge.abs(),
        "quantile(0) = {q0}, outer negative edge {outer_edge}"
    );
    assert!(
        q0 < 0.0 && q0 < sketch.min(),
        "quantile(0) is not the minimum"
    );
    assert_eq!(sketch.quantile(-0.5), q0, "negative q rounds to rank 0");
    assert_eq!(sketch.quantile(f64::NAN), q0, "NaN q rounds to rank 0");
    assert_eq!(sketch.quantile(1.5), 100.0, "q > 1 returns max");
    assert_eq!(sketch.quantile(f64::INFINITY), 100.0);
    // q = 1 is the edge of the bucket holding the maximum: in [max/γ, max].
    let q1 = sketch.quantile(1.0);
    assert!(q1 <= 100.0 && q1 >= 100.0 / g, "quantile(1) = {q1}");
    // The smallest positive quantile is the bucket edge of the minimum.
    let qmin = sketch.quantile(0.001);
    assert!(qmin <= 1.0 && qmin >= 1.0 / g, "quantile(0.001) = {qmin}");

    // NaN is counted in the zero block.
    let mut nan = DDSketch::new(ALPHA);
    nan.insert(f64::NAN);
    assert_eq!(nan.count(), 1);
    assert_eq!(nan.quantile(1.0), 0.0, "NaN lands in the zero block");
    assert_eq!(nan.min(), f64::INFINITY, "NaN never becomes min");

    // Tiny and huge finite values: below γ^{−offset} the index clamps to bin 0
    // (edge γ^{−offset−1}); above γ^{BINS−offset} the value is counted but not
    // binned, so the high quantile falls through to `max`.
    let mut tiny = DDSketch::new(ALPHA);
    tiny.insert(1e-300);
    let edge0 = alice_physics::det_math::powf64(g, -(offset as f64) - 1.0);
    let qt = tiny.quantile(1.0);
    assert!(
        (qt - edge0).abs() <= 1e-6 * edge0,
        "tiny: {qt} vs bin-0 edge {edge0}"
    );
    let mut huge = DDSketch::new(ALPHA);
    huge.insert(1e300);
    assert_eq!(huge.count(), 1);
    assert_eq!(huge.quantile(1.0), 1e300, "huge: falls through to max");
    assert_eq!(huge.quantile(0.5), 1e300);

    // ±∞: ln(∞)/ln γ = ∞, `ceil() as i32` saturates and `+ offset`
    // overflows — measured: panic in a debug build.
    for v in [f64::INFINITY, f64::NEG_INFINITY] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let mut s = DDSketch::new(ALPHA);
            s.insert(v);
            s.count()
        }));
        if cfg!(debug_assertions) {
            assert!(r.is_err(), "insert({v}) panics in debug builds");
        } else {
            assert_eq!(r.expect("release wraps"), 1);
        }
    }

    // α = 0: γ = 1, ln γ = 0; any value above 1 divides by zero into the
    // same saturating cast — measured: panic in a debug build.
    let r = catch_unwind(AssertUnwindSafe(|| {
        let mut s = DDSketch::new(0.0);
        s.insert(2.0);
        s.count()
    }));
    if cfg!(debug_assertions) {
        assert!(
            r.is_err(),
            "DDSketch::new(0.0).insert(2.0) panics in debug builds"
        );
    } else {
        assert_eq!(r.expect("release wraps"), 1);
    }
    let zero_alpha = DDSketch::new(0.0);
    assert_eq!(zero_alpha.alpha(), 0.0, "alpha is stored unvalidated");
}

// ---------------------------------------------------------------------------
// HyperLogLog
// ---------------------------------------------------------------------------

#[test]
fn hyperloglog_registers_match_the_reference_and_the_estimate_is_within_three_sigma() {
    assert_eq!(HyperLogLog::M, 16384);
    assert_eq!(HyperLogLog::P, 14);
    assert_eq!(HyperLogLog::M, 1 << HyperLogLog::P);
    let n = 10_000u64;
    let mut via_bytes = HyperLogLog::new();
    let mut via_hash_trait = HyperLogLog::new();
    let mut digests = Vec::with_capacity(n as usize);
    for i in 0..n {
        via_bytes.insert_bytes(&i.to_ne_bytes());
        via_hash_trait.insert(&i);
        digests.push(key_digest(i));
    }
    let expected = ref_hll_registers(14, &digests);
    assert_eq!(
        via_bytes.registers(),
        &expected[..],
        "registers vs reference"
    );
    assert_eq!(
        via_hash_trait.registers(),
        &expected[..],
        "Hash path == byte path"
    );
    let touched = expected.iter().filter(|&&r| r != 0).count();
    assert!(touched > 0 && touched < 16384);

    let est = via_bytes.cardinality();
    let reference = ref_hll_cardinality(&expected);
    assert!(
        (est - reference).abs() <= 1e-9 * reference,
        "estimator {est} vs published formula {reference}"
    );
    let sigma = 1.04 / 128.0;
    let rel = (est - n as f64).abs() / n as f64;
    assert!(
        rel <= 3.0 * sigma,
        "{est} for n={n}: rel {rel} > 3σ {}",
        3.0 * sigma
    );

    // Merge of a disjoint range: registers are the element-wise max.
    let mut other = HyperLogLog::new();
    let mut all = digests.clone();
    for i in n..2 * n {
        other.insert_bytes(&i.to_ne_bytes());
        all.push(key_digest(i));
    }
    via_bytes.merge(&other);
    assert_eq!(via_bytes.registers(), &ref_hll_registers(14, &all)[..]);
    let merged = via_bytes.cardinality();
    let rel = (merged - 2.0 * n as f64).abs() / (2.0 * n as f64);
    assert!(rel <= 3.0 * sigma, "merged {merged}: rel {rel}");

    // A smaller sketch obeys the same reference with its own P.
    assert_eq!(HyperLogLog10::M, 1024);
    let mut small = HyperLogLog10::new();
    for i in 0..300u64 {
        small.insert_bytes(&i.to_ne_bytes());
    }
    assert_eq!(
        small.registers(),
        &ref_hll_registers(10, &digests[..300])[..]
    );
    let est = small.cardinality();
    let rel = (est - 300.0).abs() / 300.0;
    assert!(rel <= 3.0 * 1.04 / 32.0, "m=1024: {est} rel {rel}");
}

#[test]
fn hyperloglog_degenerate_inputs() {
    let mut hll = HyperLogLog::new();
    assert_eq!(hll.cardinality(), 0.0, "empty: m·ln(m/m) = 0 exactly");
    assert!(hll.registers().iter().all(|&r| r == 0));
    assert_eq!(hll.registers().len(), HyperLogLog::M);

    // Zero-length key: one register, m·ln(m/(m−1)) ∈ [1, 1 + 1/m].
    hll.insert_bytes(&[]);
    let d = fmix64(0xcbf2_9ce4_8422_2325);
    let expected = ref_hll_registers(14, &[d]);
    assert_eq!(hll.registers(), &expected[..]);
    let j = (d as usize) & (HyperLogLog::M - 1);
    assert_ne!(hll.registers()[j], 0, "the empty key's register is set");
    assert_eq!(hll.registers().iter().filter(|&&r| r != 0).count(), 1);
    let m = HyperLogLog::M as f64;
    let one = hll.cardinality();
    assert!(one >= 1.0 && one <= 1.0 + 1.0 / m, "one key: {one}");

    // Duplicates: idempotent on the registers.
    for _ in 0..1000 {
        hll.insert_bytes(&[]);
    }
    assert_eq!(hll.registers().iter().filter(|&&r| r != 0).count(), 1);
    assert_eq!(hll.cardinality(), one);

    // k ≪ m distinct keys: linear counting bound k ≤ est ≤ k + k²/m.
    let mut few = HyperLogLog::new();
    let mut digests = Vec::new();
    for i in 0..64u64 {
        few.insert_bytes(&i.to_ne_bytes());
        digests.push(key_digest(i));
    }
    let k = ref_hll_registers(14, &digests)
        .iter()
        .filter(|&&r| r != 0)
        .count() as f64;
    let est = few.cardinality();
    assert!(est >= k && est <= k + k * k / m, "k={k}: {est}");

    // The extreme key is an ordinary key.
    few.insert(&u64::MAX);
    digests.push(key_digest(u64::MAX));
    assert_eq!(few.registers(), &ref_hll_registers(14, &digests)[..]);

    few.clear();
    assert_eq!(few.cardinality(), 0.0);
    assert!(few.registers().iter().all(|&r| r == 0));
}
