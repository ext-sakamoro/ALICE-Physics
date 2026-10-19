//! The two facts the sign-flip entries of `scripts/mutants-equivalent.txt` for
//! `src/joint.rs` rest on, measured bit for bit over many vectors:
//!
//! * flipping `impulse + n * λ` to `impulse - n * λ` where `impulse` is zero
//!   gives `0 - y`, which is `-y` exactly (fixed-point addition does not round),
//!   and the D6 loop that adds every row on one line gives `0 - a - b - c`,
//!   which is `-(a + b + c)` exactly;
//! * the reported force is `impulse.length()`, and `length(-y) == length(y)` in
//!   every bit (the squares of `y` and `-y` are the same integer products).
//!
//! The vectors come from a fixed linear congruential sequence, components from
//! about 1e-9 to 1e3 of either sign.

use alice_physics_core::{Fix128, Vec3Fix};

struct Seq(u64);

impl Seq {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn component(&mut self) -> Fix128 {
        let r = self.next();
        let mantissa = (r >> 11) as f64 / (1u64 << 53) as f64;
        let exponent = (r % 41) as i32 - 30; // 2^-30 .. 2^10
                                             // 2^exponent built from its exponent bits (exact, no libm)
        let value = mantissa * f64::from_bits(((1023 + exponent) as u64) << 52);
        Fix128::from_f64(if r & (1 << 10) == 0 { value } else { -value })
    }

    fn vector(&mut self) -> Vec3Fix {
        Vec3Fix::new(self.component(), self.component(), self.component())
    }
}

const N: usize = 20_000;

#[test]
fn zero_minus_y_is_minus_y_and_has_the_same_length() {
    let mut s = Seq(1);
    for _ in 0..N {
        let y = s.vector();
        let flipped = Vec3Fix::ZERO - y;
        assert_eq!(flipped, -y);
        let original = Vec3Fix::ZERO + y;
        let (lf, lo) = (flipped.length(), original.length());
        assert_eq!((lf.hi, lf.lo), (lo.hi, lo.lo), "y = {y:?}");
    }
}

#[test]
fn the_d6_sum_flipped_on_its_one_line_is_its_exact_negation() {
    let mut s = Seq(2);
    for _ in 0..N {
        let (a, b, c) = (s.vector(), s.vector(), s.vector());
        let original = Vec3Fix::ZERO + a + b + c;
        let flipped = Vec3Fix::ZERO - a - b - c;
        assert_eq!(flipped, -original);
        let (lf, lo) = (flipped.length(), original.length());
        assert_eq!((lf.hi, lf.lo), (lo.hi, lo.lo));
    }
}
