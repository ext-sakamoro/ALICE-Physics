//! Deterministic Random Number Generator
//!
//! PCG (Permuted Congruential Generator) implemented with Fix128 output.
//! Bit-exact across all platforms for deterministic simulation.
//!
//! # Example
//!
//! ```ignore
//! use alice_physics::rng::DeterministicRng;
//! use alice_physics::Fix128;
//!
//! let mut rng = DeterministicRng::new(42);
//! let val = rng.next_fix128(); // [0, 1) deterministic
//! ```

use crate::math::{Fix128, Vec3Fix};

/// Deterministic RNG using PCG-XSH-RR (32-bit output)
///
/// Produces identical sequences on all platforms given the same seed.
/// No floating-point operations are used internally.
#[derive(Clone, Debug)]
pub struct DeterministicRng {
    state: u64,
    inc: u64,
}

impl DeterministicRng {
    /// PCG multiplier
    const MULTIPLIER: u64 = 6364136223846793005;

    /// Create RNG with the given seed
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1, // Must be odd
        };
        // Advance state twice for initialization
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Create RNG with seed and stream
    pub fn new_with_stream(seed: u64, stream: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Generate next u32 value
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        // Advance state
        self.state = old_state
            .wrapping_mul(Self::MULTIPLIER)
            .wrapping_add(self.inc);
        // XSH-RR output function
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Generate next u64 value (two u32s combined)
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        (hi << 32) | lo
    }

    /// Generate Fix128 in range [0, 1)
    #[inline]
    pub fn next_fix128(&mut self) -> Fix128 {
        Fix128 {
            hi: 0,
            lo: self.next_u64(),
        }
    }

    /// Generate Fix128 in range [lo, hi)
    pub fn next_fix128_range(&mut self, lo: Fix128, hi: Fix128) -> Fix128 {
        let t = self.next_fix128(); // [0, 1)
        let range = hi - lo;
        lo + range * t
    }

    /// Generate random unit direction vector (deterministic)
    pub fn next_direction(&mut self) -> Vec3Fix {
        // Marsaglia method: generate in [-1,1]^2, reject if outside unit disk
        for _ in 0..64 {
            let u = self.next_fix128_range(Fix128::NEG_ONE, Fix128::ONE);
            let v = self.next_fix128_range(Fix128::NEG_ONE, Fix128::ONE);
            let s = u * u + v * v;
            if s >= Fix128::ONE || s.is_zero() {
                continue;
            }
            let factor = (Fix128::ONE - s).sqrt();
            let two = Fix128::from_int(2);
            return Vec3Fix::new(two * u * factor, two * v * factor, Fix128::ONE - two * s);
        }
        Vec3Fix::UNIT_Y
    }

    /// Generate random value in [0, max) as u32
    #[inline]
    pub fn next_bounded(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        // Rejection sampling for uniform distribution
        let threshold = max.wrapping_neg() % max;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % max;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_determinism() {
        let mut rng1 = DeterministicRng::new(12345);
        let mut rng2 = DeterministicRng::new(12345);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_different_seeds() {
        let mut rng1 = DeterministicRng::new(1);
        let mut rng2 = DeterministicRng::new(2);

        let mut same_count = 0;
        for _ in 0..100 {
            if rng1.next_u32() == rng2.next_u32() {
                same_count += 1;
            }
        }
        assert!(
            same_count < 5,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_fix128_range() {
        let mut rng = DeterministicRng::new(42);
        for _ in 0..100 {
            let val = rng.next_fix128();
            assert!(val >= Fix128::ZERO);
            assert!(val < Fix128::ONE);
        }
    }

    #[test]
    fn test_bounded() {
        let mut rng = DeterministicRng::new(99);
        for _ in 0..100 {
            let val = rng.next_bounded(10);
            assert!(val < 10);
        }
    }

    #[test]
    fn test_streams() {
        let mut rng1 = DeterministicRng::new_with_stream(42, 1);
        let mut rng2 = DeterministicRng::new_with_stream(42, 2);

        let a = rng1.next_u32();
        let b = rng2.next_u32();
        assert_ne!(a, b, "Different streams should produce different values");
    }

    #[test]
    fn test_direction_vector() {
        let mut rng = DeterministicRng::new(777);
        for _ in 0..10 {
            let dir = rng.next_direction();
            let len_sq = dir.length_squared();
            // Should be approximately unit length
            let diff = if len_sq > Fix128::ONE {
                len_sq - Fix128::ONE
            } else {
                Fix128::ONE - len_sq
            };
            assert!(
                diff < Fix128::from_ratio(1, 10),
                "Direction should be roughly unit length"
            );
        }
    }
}
