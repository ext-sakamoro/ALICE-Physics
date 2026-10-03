//! Oracles for `rng::DeterministicRng::{new_with_stream, next_bounded}`
//! (`examples/rng_streams_and_bounded.rs`).
//!
//! * `new_with_stream(42, 54)` is the PCG32 reference seeding (initstate 42,
//!   initseq 54); the published demo vector is
//!   `a15c02b7 7b47f409 ba1d3330 83d2f293 bfa4784b cbed606e`.
//! * `new(seed)` seeds the stream with `seed` as well: `new(s) == new_with_stream(s, s)`.
//! * `next_bounded(m)` is rejection sampling with threshold `(2^32 - m) mod m`:
//!   an independent replay over `next_u32` of a cloned generator must agree,
//!   and the output is uniform on `0..m`.
#![allow(clippy::disallowed_methods)]

use alice_physics::rng::DeterministicRng;

#[test]
fn stream_matches_the_pcg32_reference_vector() {
    let mut r = DeterministicRng::new_with_stream(42, 54);
    let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
    assert_eq!(
        got,
        vec![0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]
    );
}

#[test]
fn new_is_new_with_stream_of_the_same_value() {
    for s in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX >> 1] {
        let mut a = DeterministicRng::new(s);
        let mut b = DeterministicRng::new_with_stream(s, s);
        for _ in 0..16 {
            assert_eq!(a.next_u32(), b.next_u32(), "seed {s}");
        }
    }
}

#[test]
fn streams_differ_but_are_reproducible() {
    let take = |seed, stream| {
        let mut r = DeterministicRng::new_with_stream(seed, stream);
        (0..8).map(|_| r.next_u32()).collect::<Vec<_>>()
    };
    assert_eq!(take(7, 1), take(7, 1));
    assert_ne!(take(7, 1), take(7, 2));
    assert_ne!(take(7, 1), take(8, 1));
    // `inc = (stream << 1) | 1`: streams s and s + 2^63 collide (the top bit is shifted out)
    assert_eq!(take(7, 5), take(7, 5 + (1u64 << 63)));
}

#[test]
fn bounded_equals_independent_rejection_replay() {
    for &m in &[
        1u32,
        2,
        3,
        6,
        7,
        100,
        1 << 16,
        0x8000_0001,
        3_000_000_000,
        u32::MAX,
    ] {
        let mut a = DeterministicRng::new_with_stream(99, m as u64);
        let mut raw = a.clone();
        // threshold = (2^32 - m) % m computed in u64
        let thr = ((1u64 << 32) - m as u64) % m as u64;
        for i in 0..2000 {
            let got = a.next_bounded(m);
            let want = loop {
                let r = raw.next_u32() as u64;
                if r >= thr {
                    break (r % m as u64) as u32;
                }
            };
            assert_eq!(got, want, "m={m} i={i}");
            assert!(got < m);
        }
    }
}

#[test]
fn bounded_degenerate_inputs() {
    let mut a = DeterministicRng::new(5);
    let before = a.clone();
    assert_eq!(a.next_bounded(0), 0);
    // max == 0 consumes nothing
    let mut b = before;
    assert_eq!(a.next_u32(), b.next_u32());
    for _ in 0..50 {
        assert_eq!(a.next_bounded(1), 0);
    }
}

#[test]
fn bounded_is_uniform_chi_square() {
    let mut r = DeterministicRng::new_with_stream(2026, 6);
    let n = 60_000u32;
    let mut c = [0f64; 6];
    for _ in 0..n {
        c[r.next_bounded(6) as usize] += 1.0;
    }
    let e = f64::from(n) / 6.0;
    let chi: f64 = c.iter().map(|&x| (x - e) * (x - e) / e).sum();
    // 5 dof: 99.9% critical value is 20.5
    assert!(chi < 20.5, "chi^2 = {chi}");
    // a max with a large rejection region: the top and bottom halves are balanced
    let m = 3_000_000_000u32;
    let mut lo = 0u32;
    for _ in 0..20_000 {
        if r.next_bounded(m) < m / 2 {
            lo += 1;
        }
    }
    assert!((9_600..=10_400).contains(&lo), "lo = {lo}");
}

#[test]
fn bounded_accepts_a_draw_equal_to_the_threshold_and_consumes_exactly_one() {
    // For m = 2 the threshold is (2^32 - 2) mod 2 = 0, so a raw draw of exactly 0
    // is accepted (`r >= threshold`). Seed 2645806231 was found by exhaustive
    // search to produce a first raw draw of 0.
    let mut probe = DeterministicRng::new(2_645_806_231);
    assert_eq!(probe.next_u32(), 0, "seed search premise");
    let mut a = DeterministicRng::new(2_645_806_231);
    assert_eq!(a.next_bounded(2), 0);
    assert_eq!(a.next_u32(), probe.next_u32(), "exactly one draw consumed");
    // m = 1 also consumes exactly one draw (threshold 0, always accepted)
    let mut b = DeterministicRng::new(11);
    let mut c = DeterministicRng::new(11);
    assert_eq!(b.next_bounded(1), 0);
    c.next_u32();
    assert_eq!(b.next_u32(), c.next_u32());
}
