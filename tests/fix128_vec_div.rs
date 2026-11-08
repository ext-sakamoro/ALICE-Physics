//! `Vec3Fix / Fix128` returns, for every component, the same bits as `component / rhs`.
//!
//! oracle: the `Fix128 / Fix128` long division, which `tests/fix128_bit_identity.rs` keeps equal to
//! its earlier implementations. The vector division takes another route for a divisor below 1.0
//! (raw magnitude below 2⁶⁴): the divisor is prepared once and each component is divided by the
//! reciprocal steps of Möller and Granlund. These tests compare the two routes on boundary values
//! (divisors at the ends of that route: 1 ulp, 2⁶³ − 1, 2⁶³, 2⁶⁴ − 1, and the first divisor past
//! it, 2⁶⁴; numerators 0, ±1 ulp, MIN, MAX) and on 20 000 seeded samples of every magnitude.

use alice_physics_core::math::{Fix128, Vec3Fix};

fn raw(v: i128) -> Fix128 {
    Fix128 {
        hi: (v >> 64) as i64,
        lo: v as u64,
    }
}

fn bits(v: Fix128) -> (i64, u64) {
    (v.hi, v.lo)
}

fn check(x: i128, y: i128, z: i128, d: i128) -> usize {
    let (vx, vy, vz, vd) = (raw(x), raw(y), raw(z), raw(d));
    let got = Vec3Fix::new(vx, vy, vz) / vd;
    for (name, c, g) in [("x", vx, got.x), ("y", vy, got.y), ("z", vz, got.z)] {
        assert_eq!(
            bits(g),
            bits(c / vd),
            "component {name}: raw {:#x} / raw {d:#x}",
            (i128::from(c.hi) << 64) | i128::from(c.lo)
        );
    }
    3
}

#[test]
fn boundary_values_match_the_scalar_division() {
    let one = 1i128 << 64;
    let divisors = [
        1,
        2,
        3,
        7,
        (1 << 32) + 1,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        one - 1,
        one,
        one + 1,
        6 * one,
        i128::MAX,
        i128::MIN,
        i128::MIN + 1,
        0,
    ];
    let numerators = [
        0,
        1,
        -1,
        2,
        -7,
        one - 1,
        one,
        -one,
        (1 << 63) - 1,
        i128::MAX,
        i128::MIN,
        i128::MIN + 1,
        i128::MAX - 1,
    ];
    let mut compared = 0;
    for &d in &divisors {
        for &n in &numerators {
            for s in [1, -1] {
                compared += check(n, n.wrapping_neg(), n / 3, d.wrapping_mul(s));
            }
        }
    }
    assert_eq!(compared, divisors.len() * numerators.len() * 2 * 3);
}

/// A fixed sequence (splitmix64), so every run and every platform compares the same inputs.
struct Seq(u64);

impl Seq {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// a 128-bit value of every magnitude: shifted down by 0..127 bits
    fn wide(&mut self) -> i128 {
        let w = ((u128::from(self.next()) << 64) | u128::from(self.next())) as i128;
        w >> (self.next() % 128)
    }
}

#[test]
fn twenty_thousand_samples_of_every_magnitude_match() {
    let mut seq = Seq(0x0076_6563_5f64_6976);
    let mut compared = 0;
    let mut short_divisors = 0;
    for _ in 0..20_000 {
        let d = seq.wide();
        if d.unsigned_abs() >> 64 == 0 {
            short_divisors += 1;
        }
        compared += check(seq.wide(), seq.wide(), seq.wide(), d);
    }
    assert_eq!(compared, 60_000);
    // about half the samples take the prepared-divisor route; both routes are compared
    assert!(
        short_divisors > 5_000 && short_divisors < 15_000,
        "{short_divisors}"
    );
}

/// The case the soft body takes in its edge constraints: a direction divided by its length,
/// lengths below 1.0, components no longer than the length.
#[test]
fn unit_directions_under_one_match() {
    let mut seq = Seq(0x6564_6765);
    let mut compared = 0;
    for _ in 0..20_000 {
        let len = (seq.next() >> (seq.next() % 63)) as i128 | 1;
        let c = |s: &mut Seq| (s.next() as i128 % len) * if s.next() & 1 == 0 { 1 } else { -1 };
        compared += check(c(&mut seq), c(&mut seq), c(&mut seq), len);
    }
    assert_eq!(compared, 60_000);
}
