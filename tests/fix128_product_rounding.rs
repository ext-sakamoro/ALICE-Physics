//! `Fix128`'s product rounds to nearest, ties to even, and is odd in its operands.
//!
//! oracle: an independent reference written here from the definition, not from the crate's
//! code: the exact product of the two magnitudes as a 256-bit integer (four 64×64 products in
//! `u128`), its bits [64, 192) rounded on the 64 bits below (above one half up, below down,
//! exactly one half to the even neighbour), then the sign of the product applied. The crate
//! computes the same value another way (the floored two's-complement product plus a correction
//! from the dropped low word), so agreement over the corpus is a real comparison.
//!
//! Why the rounding matters: with the product floored toward −∞, `(−a)·b` was one unit below
//! `−(a·b)`, so mirror-image terms (a face's corners summed about its centre) left a few units
//! that spun a body which should rest. The mirror test below sums the torques of four corners
//! placed symmetrically about the centre and requires exactly zero.
//!
//! Every comparison counts what it compared and fails on zero.

use alice_physics_core::math::{Fix128, Vec3Fix};

/// The product by the definition: magnitudes, exact 256-bit product, nearest-even on the 64
/// bits below 2^-64, sign. Wraps like the operator when the product leaves the range.
fn reference(a: Fix128, b: Fix128) -> Fix128 {
    let neg = (a.hi < 0) != (b.hi < 0);
    let mag = |x: Fix128| -> u128 {
        let raw = ((x.hi as i128) << 64) | i128::from(x.lo);
        raw.unsigned_abs()
    };
    let (x, y) = (mag(a), mag(b));
    let (x1, x0) = ((x >> 64) as u64, x as u64);
    let (y1, y0) = ((y >> 64) as u64, y as u64);
    let p00 = u128::from(x0) * u128::from(y0);
    let p01 = u128::from(x0) * u128::from(y1);
    let p10 = u128::from(x1) * u128::from(y0);
    let p11 = u128::from(x1) * u128::from(y1);
    // 256-bit sum as four 64-bit words w0..w3
    let w0 = p00 as u64;
    let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
    let w1 = mid as u64;
    let upper = (mid >> 64) + (p01 >> 64) + (p10 >> 64) + (p11 & u128::from(u64::MAX));
    let w2 = upper as u64;
    let mut q = (u128::from(w2) << 64) | u128::from(w1);
    const HALF: u64 = 1 << 63;
    if w0 > HALF || (w0 == HALF && q & 1 == 1) {
        q = q.wrapping_add(1);
    }
    let raw = q as i128;
    let raw = if neg { raw.wrapping_neg() } else { raw };
    Fix128 {
        hi: (raw >> 64) as i64,
        lo: raw as u64,
    }
}

fn raw(hi: i64, lo: u64) -> Fix128 {
    Fix128 { hi, lo }
}

/// xorshift64, seeded: the corpus is the same on every run
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// an operand whose integer part is below 2^bits in magnitude, either sign
    fn operand(&mut self, bits: u32) -> Fix128 {
        let hi = (self.next() >> (64 - bits)) as i64 - (1i64 << (bits - 1));
        raw(hi, self.next())
    }
}

#[test]
fn the_product_is_the_reference_over_the_corpus() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut compared = 0u32;
    // integer parts up to 2^1, 2^16, 2^31 (products in range), and ties made on purpose
    for bits in [1u32, 16, 31] {
        for _ in 0..100_000 {
            let (a, b) = (rng.operand(bits), rng.operand(bits));
            assert_eq!(a * b, reference(a, b), "{a:?} * {b:?}");
            compared += 1;
        }
    }
    for _ in 0..50_000 {
        // b = k / 2: a product of a and a half has the dropped word 0 or exactly one half
        let a = rng.operand(20);
        let b = raw(0, 1 << 63);
        assert_eq!(a * b, reference(a, b), "tie {a:?} * 1/2");
        compared += 1;
    }
    assert_eq!(compared, 350_000);
}

#[test]
fn the_rounding_table() {
    let half = raw(0, 1 << 63); // 0.5
    let quarter = raw(0, 1 << 62);
    let three_quarters = raw(0, 3 << 62);
    let ulp = |k: i64| -> Fix128 {
        let r = i128::from(k);
        raw((r >> 64) as i64, r as u64)
    };
    // (a, b, expected): products of k units by 1/2, 1/4, 3/4
    let rows = [
        (ulp(1), half, ulp(0)),             // 0.5 unit: tie to even 0
        (ulp(3), half, ulp(2)),             // 1.5: tie to even 2
        (ulp(5), half, ulp(2)),             // 2.5: tie to even 2
        (ulp(7), half, ulp(4)),             // 3.5: tie to even 4
        (ulp(1), quarter, ulp(0)),          // 0.25: down
        (ulp(1), three_quarters, ulp(1)),   // 0.75: up
        (ulp(5), three_quarters, ulp(4)),   // 3.75: up
        (ulp(-1), half, ulp(0)),            // −0.5: tie to even 0
        (ulp(-3), half, ulp(-2)),           // −1.5: tie to even −2
        (ulp(-5), half, ulp(-2)),           // −2.5: tie to even −2
        (ulp(-1), three_quarters, ulp(-1)), // −0.75
        (ulp(-1), quarter, ulp(0)),         // −0.25
    ];
    for (a, b, want) in rows {
        assert_eq!(a * b, want, "{a:?} * {b:?}");
        assert_eq!(b * a, want, "commuted {a:?} * {b:?}");
        assert_eq!(
            reference(a, b),
            want,
            "the reference agrees on {a:?} * {b:?}"
        );
    }
    // exact products are untouched
    for (x, y) in [(3i64, -7i64), (-12, -12), (1 << 20, 3)] {
        assert_eq!(
            Fix128::from_int(x) * Fix128::from_int(y),
            Fix128::from_int(x * y)
        );
    }
    assert_eq!(rows.len(), 12);
}

#[test]
fn the_product_is_odd_and_commutative_for_every_sampled_pair() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut compared = 0u32;
    for _ in 0..200_000 {
        let (a, b) = (rng.operand(24), rng.operand(24));
        let p = a * b;
        assert_eq!((-a) * b, -p, "(-a)*b, a={a:?} b={b:?}");
        assert_eq!(a * (-b), -p, "a*(-b), a={a:?} b={b:?}");
        assert_eq!((-a) * (-b), p, "(-a)*(-b), a={a:?} b={b:?}");
        assert_eq!(b * a, p, "b*a, a={a:?} b={b:?}");
        compared += 1;
    }
    assert_eq!(compared, 200_000);
}

/// Four corners at (±x, y, ±z) about a centre, each pushed by the same force: the sum of
/// their torques r × F cancels exactly. With the floored product the x and z sums kept a few
/// units, the start of the spin a resting box picked up.
#[test]
fn mirror_image_torques_cancel_to_the_bit() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let mut compared = 0u32;
    for _ in 0..20_000 {
        let (x, y, z) = (rng.operand(4), rng.operand(4), rng.operand(4));
        let f = Vec3Fix::new(Fix128::ZERO, rng.operand(8), Fix128::ZERO);
        let mut sum = Vec3Fix::ZERO;
        for (sx, sz) in [(1i64, 1i64), (-1, 1), (1, -1), (-1, -1)] {
            let r = Vec3Fix::new(x * Fix128::from_int(sx), y, z * Fix128::from_int(sz));
            sum = sum + r.cross(f);
        }
        assert_eq!(sum, Vec3Fix::ZERO, "corners ±{x:?}, ±{z:?} under {f:?}");
        compared += 1;
    }
    assert_eq!(compared, 20_000);
}

#[test]
fn checked_mul_is_the_product_in_range_and_none_outside() {
    let mut rng = Rng(0x5151_5151_a5a5_a5a5);
    let mut compared = 0u32;
    for _ in 0..100_000 {
        let (a, b) = (rng.operand(30), rng.operand(30));
        assert_eq!(a.checked_mul(b), Some(a * b), "{a:?} * {b:?}");
        compared += 1;
    }
    assert_eq!(
        Fix128::from_int(1 << 40).checked_mul(Fix128::from_int(1 << 40)),
        None
    );
    assert_eq!(compared, 100_000);
}
