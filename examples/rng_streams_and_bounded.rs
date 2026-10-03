//! Independent PCG streams and unbiased bounded integers:
//! `DeterministicRng::{new_with_stream, next_bounded}`.
//!
//! `new_with_stream(42, 54)` is the PCG32 reference seeding (`initstate = 42`,
//! `initseq = 54`); its first outputs are the published demo vector
//! `a15c02b7 7b47f409 ba1d3330 83d2f293 bfa4784b cbed606e`.
//!
//! ```bash
//! cargo run --release --example rng_streams_and_bounded --features std
//! ```

use alice_physics::rng::DeterministicRng;

fn main() {
    let mut r = DeterministicRng::new_with_stream(42, 54);
    print!("reference stream:");
    for _ in 0..6 {
        print!(" {:08x}", r.next_u32());
    }
    println!();

    // One stream per subsystem: same seed, different streams never correlate.
    let mut a = DeterministicRng::new_with_stream(7, 1);
    let mut b = DeterministicRng::new_with_stream(7, 2);
    println!("stream 1: {} {}", a.next_u32(), a.next_u32());
    println!("stream 2: {} {}", b.next_u32(), b.next_u32());

    // Fair six-sided die: 60000 rolls, expected 10000 per face.
    let mut dice = DeterministicRng::new_with_stream(2026, 6);
    let mut counts = [0u32; 6];
    for _ in 0..60_000 {
        counts[dice.next_bounded(6) as usize] += 1;
    }
    println!("die counts (expect ~10000 each): {counts:?}");
}
