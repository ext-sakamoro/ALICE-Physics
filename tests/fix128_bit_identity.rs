//! `Fix128` `sqrt`, `/` and `*` return the same bits as the reference implementations below.
//!
//! oracle: `sqrt_ref`, `div_ref` and `mul_ref` are the implementations these operations had before
//! they were made faster, kept here word for word (a 96-step 2-bit square root recurrence, a 64-step
//! bit long division, and four wrapping 128-bit products). Each operation is compared through its
//! public entry point on edge values, on dense bands around typical physics values and on seeded
//! random values of every magnitude. Every golden and identifier of the engine depends on these bits.

use alice_physics_core::math::Fix128;

fn raw(v: i128) -> Fix128 {
    Fix128 {
        hi: (v >> 64) as i64,
        lo: v as u64,
    }
}

fn bits(v: Fix128) -> (i64, u64) {
    (v.hi, v.lo)
}

fn sqrt_ref(v: Fix128) -> Fix128 {
    if v.is_negative() || v.is_zero() {
        return Fix128::ZERO;
    }
    let n = ((v.hi as u128) << 64) | (v.lo as u128);
    let mut rem: u128 = 0;
    let mut root: u128 = 0;
    let mut i = 0u32;
    while i < 96 {
        let b = if i < 64 { (n >> (126 - 2 * i)) & 3 } else { 0 };
        rem = (rem << 2) | b;
        let trial = (root << 2) | 1;
        if rem >= trial {
            rem -= trial;
            root = (root << 1) | 1;
        } else {
            root <<= 1;
        }
        i += 1;
    }
    Fix128 {
        hi: (root >> 64) as i64,
        lo: root as u64,
    }
}

fn neg_ref(v: Fix128) -> Fix128 {
    let (lo, carry) = (!v.lo).overflowing_add(1);
    let hi = (!v.hi).wrapping_add(carry as i64);
    Fix128 { hi, lo }
}

fn div_ref(a: Fix128, b: Fix128) -> Fix128 {
    if b.is_zero() {
        return Fix128::ZERO;
    }
    let neg = a.is_negative() != b.is_negative();
    let x = if a.is_negative() { neg_ref(a) } else { a };
    let y = if b.is_negative() { neg_ref(b) } else { b };
    let a_full = ((x.hi as u128) << 64) | (x.lo as u128);
    let b_full = ((y.hi as u128) << 64) | (y.lo as u128);
    if b_full == 0 {
        return Fix128::ZERO;
    }
    let quot_hi = a_full / b_full;
    let rem = a_full % b_full;
    let mut r = rem;
    let mut quot_lo: u64 = 0;
    for i in (0..64).rev() {
        let overflow_bit = r >> 127;
        r <<= 1;
        if overflow_bit != 0 || r >= b_full {
            r = r.wrapping_sub(b_full);
            quot_lo |= 1u64 << i;
        }
    }
    let result = Fix128 {
        hi: quot_hi as i64,
        lo: quot_lo,
    };
    if neg {
        neg_ref(result)
    } else {
        result
    }
}

fn mul_ref(a: Fix128, b: Fix128) -> Fix128 {
    let a_hi = a.hi as i128;
    let a_lo = a.lo as u128;
    let b_hi = b.hi as i128;
    let b_lo = b.lo as u128;
    let ll = a_lo.wrapping_mul(b_lo);
    let hl = a_hi.wrapping_mul(b_lo as i128);
    let lh = (a_lo as i128).wrapping_mul(b_hi);
    let hh = a_hi.wrapping_mul(b_hi);
    let ll_hi = (ll >> 64) as i128;
    let mid = hl.wrapping_add(lh).wrapping_add(ll_hi);
    let mid_lo = mid as u64;
    let mid_hi = (mid >> 64) as i64;
    let hi = (hh as i64).wrapping_add(mid_hi);
    // the 64 dropped bits (the low word of lo·lo) rounded to nearest, ties to even
    let dropped = ll as u64;
    const HALF: u64 = 1 << 63;
    let up = dropped > HALF || (dropped == HALF && mid_lo & 1 == 1);
    let (lo, carry) = mid_lo.overflowing_add(u64::from(up));
    Fix128 {
        hi: hi.wrapping_add(i64::from(carry)),
        lo,
    }
}

/// A fixed sequence (splitmix64), so every run and every platform compares the same inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A signed raw value of a random magnitude: a uniform draw alone would almost never be small.
    fn any_magnitude(&mut self) -> i128 {
        let v = (((self.next() as u128) << 64) | self.next() as u128) as i128;
        v >> (self.next() % 128)
    }
}

/// 0, one unit in the last place, the extremes, and every power of two with its neighbours, both signs.
fn edges() -> Vec<i128> {
    let mut out = vec![0, 1, -1, i128::MAX, i128::MIN, i128::MAX - 1, i128::MIN + 1];
    for k in 0..127 {
        let p = 1i128 << k;
        out.extend([p - 1, p, p + 1, -(p - 1), -p, -(p + 1)]);
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// `r * r` as the words above and below 2^128, for `r` below 2^96.
fn square_192(r: u128) -> (u128, u128) {
    let (top, bottom) = (r >> 48, r & ((1 << 48) - 1));
    let middle = 2 * top * bottom; // below 2^97
    let (low, carry_a) = (bottom * bottom).overflowing_add(middle << 48);
    let (low, carry_b) = low.overflowing_add((top * top) << 96);
    let high = ((top * top) >> 32) + (middle >> 80) + u128::from(carry_a) + u128::from(carry_b);
    (high, low)
}

const RANDOM_CASES: usize = 1 << 20;

#[test]
fn sqrt_matches_the_reference_on_every_input_class() {
    let mut compared = 0usize;
    let mut same = |v: i128| {
        let v = raw(v);
        assert_eq!(bits(v.sqrt()), bits(sqrt_ref(v)), "sqrt of {v:?}");
        compared += 1;
    };
    for v in edges() {
        same(v);
    }
    // every integer part 0..=256 at a fixed fractional stride: cloth rest lengths and distances live here
    for int in 0i128..=256 {
        for j in 0i128..4096 {
            same((int << 64) | (j << 52) | (j * 0x1_0001));
        }
    }
    // the boundary of the floor: the smallest raw value whose root is r, and the one below it
    let mut rng = Rng(0x5152_5431);
    for _ in 0..(1 << 16) {
        let r = (rng.next() as u128) >> (rng.next() % 64);
        let square = r * r;
        let at = (square >> 64) + u128::from((square as u64) != 0);
        same(at as i128);
        same(at.wrapping_sub(1) as i128);
    }
    // the same boundary for roots of every size up to the largest, with the 192-bit square written out:
    // above, roots end at 64 bits, and the scaling inside an implementation can move such a boundary
    // away from a remainder of zero. Low parts of 0 and 1 give perfect squares at every digit split.
    for _ in 0..(1 << 16) {
        let r = (((rng.next() as u128) << 32) | (rng.next() as u128 >> 32)) >> (rng.next() % 96);
        let r = match rng.next() % 4 {
            0 => r >> 48 << 48,
            1 => r >> 24 << 24,
            2 => (r >> 48 << 48) | 1,
            _ => r,
        };
        let (high, low) = square_192(r);
        // ceil(square / 2^64) fits 128 bits while r < 2^96: `high` is the part above 2^128
        let at = (high << 64) + (low >> 64) + u128::from((low as u64) != 0);
        same(at as i128);
        same(at.wrapping_sub(1) as i128);
    }
    // a perfect square in the top digits and anything below: a digit split of an implementation then
    // sees a remainder of exactly zero, followed by digits that still move the root
    for _ in 0..(1 << 18) {
        let bits = [24, 12, 36][(rng.next() % 3) as usize];
        let root = (rng.next() >> (64 - bits)) | (1 << (bits - 1));
        let keep = 2 * bits;
        let top = u128::from(root) * u128::from(root);
        let below = (((rng.next() as u128) << 64) | rng.next() as u128) >> keep;
        same(((top << (128 - keep)) | below) as i128);
    }
    // a power of four and a run of ones anywhere below it: an exact square in the top digits with the
    // smallest root, and then the largest next digit against the smallest divisor
    for power in (0..=126).step_by(2) {
        for end in 0..power {
            for len in 1..=(power - end) {
                let v = (1u128 << power) | (((1u128 << len) - 1) << end);
                same(v as i128);
                same((v | (rng.next() as u128 >> (64 - end.min(63)))) as i128);
            }
        }
    }
    for _ in 0..RANDOM_CASES {
        same(rng.any_magnitude());
    }
    assert!(compared > RANDOM_CASES + 1_000_000, "compared {compared}");
}

#[test]
fn division_matches_the_reference_on_every_input_class() {
    let mut compared = 0usize;
    let mut same = |a: i128, b: i128| {
        let (a, b) = (raw(a), raw(b));
        assert_eq!(bits(a / b), bits(div_ref(a, b)), "{a:?} / {b:?}");
        compared += 1;
    };
    let edge = edges();
    for &a in &edge {
        for &b in &edge {
            same(a, b);
        }
    }
    // every 32-bit limb of both operands drawn from the values next to a limb boundary: the quotient
    // digit estimate of a long division is corrected, and rarely added back, only on such limbs
    const LIMBS: [u32; 5] = [0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut wide = Vec::with_capacity(LIMBS.len().pow(4));
    for l3 in LIMBS {
        for l2 in LIMBS {
            for l1 in LIMBS {
                for l0 in LIMBS {
                    let v = (u128::from(l3) << 96)
                        | (u128::from(l2) << 64)
                        | (u128::from(l1) << 32)
                        | u128::from(l0);
                    wide.push(v as i128);
                }
            }
        }
    }
    for &a in &wide {
        for &b in &wide {
            same(a, b);
        }
    }
    // divisors at 2^k and one unit either side, against numerators of every magnitude
    let mut rng = Rng(0x4449_5631);
    for k in 0..127 {
        let p = 1i128 << k;
        for d in [p - 1, p, p + 1, -(p - 1), -p, -(p + 1)] {
            for _ in 0..256 {
                same(rng.any_magnitude(), d);
            }
        }
    }
    for _ in 0..RANDOM_CASES {
        same(rng.any_magnitude(), rng.any_magnitude());
    }
    assert!(compared > RANDOM_CASES + 500_000, "compared {compared}");
}

#[test]
fn multiplication_matches_the_reference_on_every_input_class() {
    let mut compared = 0usize;
    let mut same = |a: i128, b: i128| {
        let (a, b) = (raw(a), raw(b));
        assert_eq!(bits(a * b), bits(mul_ref(a, b)), "{a:?} * {b:?}");
        compared += 1;
    };
    let edge = edges();
    for &a in &edge {
        for &b in &edge {
            same(a, b);
        }
    }
    let mut rng = Rng(0x4d55_4c31);
    for _ in 0..RANDOM_CASES {
        same(rng.any_magnitude(), rng.any_magnitude());
    }
    assert!(compared > RANDOM_CASES + 200_000, "compared {compared}");
}
