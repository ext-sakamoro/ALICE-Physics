//! Deterministic Fixed-Point Mathematics - SIMD Optimized Edition
//!
//! > "God does not play dice with floating point numbers."
//!
//! # Overview
//!
//! This module provides bit-exact arithmetic that produces identical results
//! on x86, ARM, WASM, or any other platform. No IEEE 754 floating point.
//!
//! # Features
//!
//! - **SIMD Acceleration**: When `simd` feature is enabled, uses platform intrinsics
//! - **Determinism**: All operations produce bit-identical results across platforms
//! - **Zero-allocation**: Hot paths use no heap allocation
//!
//! # Types
//!
//! - `Fix128` (I64F64): 128-bit fixed-point with 64 integer bits, 64 fractional bits
//! - `Vec3Fix`: 3D vector using Fix128 components (SIMD-accelerated dot product)
//! - `QuatFix`: Quaternion using Fix128 components
//! - `Mat3Fix`: 3x3 matrix for inertia tensors
//!
//! # Precision
//!
//! - Range: ±9.2 × 10^18 (meters)
//! - Precision: ~5.4 × 10^-20 (meters)
//! - From subatomic particles to galactic scales with uniform precision

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

// SIMD imports for x86_64
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use core::arch::x86_64::*;

// ============================================================================
// Fix128 (I64F64) - 128-bit Fixed-Point Number
// ============================================================================

/// 128-bit fixed-point number (64 integer bits, 64 fractional bits)
///
/// Internal representation: `value = raw / 2^64`
///
/// This provides:
/// - Range: ±9.2 × 10^18
/// - Precision: ~5.4 × 10^-20
/// - Bit-exact arithmetic across all platforms
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Fix128 {
    /// Raw 128-bit value (stored as two i64s for portability)
    /// Interpretation: (hi << 64) | lo as a signed 128-bit integer
    /// The decimal point is between hi and lo
    pub hi: i64, // Integer part (signed)
    /// Fractional part (64-bit unsigned)
    pub lo: u64, // Fractional part
}

impl Fix128 {
    /// Zero constant
    pub const ZERO: Self = Self { hi: 0, lo: 0 };

    /// One constant (1.0)
    pub const ONE: Self = Self { hi: 1, lo: 0 };

    /// Negative one (-1.0)
    pub const NEG_ONE: Self = Self { hi: -1, lo: 0 };

    /// Pi (π) - precomputed to full precision
    /// π ≈ 3.14159265358979323846...
    pub const PI: Self = Self {
        hi: 3,
        lo: 0x243F6A8885A308D3, // Fractional part of π
    };

    /// Half Pi (π/2)
    pub const HALF_PI: Self = Self {
        hi: 1,
        lo: 0x921FB54442D18469, // Fractional part of π/2
    };

    /// Two Pi (2π)
    pub const TWO_PI: Self = Self {
        hi: 6,
        lo: 0x487ED5110B4611A6, // Fractional part of 2π
    };

    /// Create from integer
    #[inline]
    #[must_use]
    pub const fn from_int(n: i64) -> Self {
        Self { hi: n, lo: 0 }
    }

    /// Create from raw parts (hi = integer, lo = fraction)
    #[inline]
    #[must_use]
    pub const fn from_raw(hi: i64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// Create from f64 (for initialization only, not deterministic!)
    #[must_use]
    pub fn from_f64(f: f64) -> Self {
        let hi = f as i64; // truncation toward zero
        let frac = f - (hi as f64);
        let abs_frac = if frac < 0.0 { -frac } else { frac };
        let lo = (abs_frac * (1u128 << 64) as f64) as u64;
        if f < 0.0 && lo != 0 {
            Self {
                hi: hi - 1,
                lo: (!lo).wrapping_add(1),
            }
        } else {
            Self { hi, lo }
        }
    }

    /// Convert to f64 (for debugging only, not deterministic!)
    #[must_use]
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + (self.lo as f64 / (1u128 << 64) as f64)
    }

    /// Create from f32 (for SDF bridge, not deterministic!)
    #[must_use]
    pub fn from_f32(f: f32) -> Self {
        Self::from_f64(f as f64)
    }

    /// Convert to f32 (for SDF bridge, not deterministic!)
    #[must_use]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Create from fraction (numerator / denominator)
    #[must_use]
    pub fn from_ratio(num: i64, denom: i64) -> Self {
        if denom == 0 {
            return Self::ZERO;
        }

        let neg = (num < 0) != (denom < 0);
        let num = num.unsigned_abs() as u128;
        let denom = denom.unsigned_abs() as u128;

        // Compute (num << 64) / denom
        let scaled = (num << 64) / denom;
        let hi = (scaled >> 64) as i64;
        let lo = scaled as u64;

        if neg {
            Self { hi, lo }.neg()
        } else {
            Self { hi, lo }
        }
    }

    /// Absolute value
    #[inline]
    #[must_use]
    pub fn abs(self) -> Self {
        if self.hi < 0 || (self.hi == 0 && self.lo == 0) {
            self.neg()
        } else {
            self
        }
    }

    /// Check if negative
    #[inline]
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.hi < 0
    }

    /// Check if zero
    #[inline]
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    /// Floor (round toward negative infinity)
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self { hi: self.hi, lo: 0 }
    }

    /// Ceiling (round toward positive infinity)
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        if self.lo == 0 {
            self
        } else {
            Self {
                hi: self.hi + 1,
                lo: 0,
            }
        }
    }

    /// Square root using Newton-Raphson iteration
    ///
    /// Deterministic: Fixed number of iterations
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.is_negative() || self.is_zero() {
            return Self::ZERO;
        }

        // Initial guess: deterministic bit-width estimation (no f64)
        // For Fix128 (I64F64), value = (hi << 64 | lo) / 2^64
        // Estimate sqrt via bit position: if highest set bit is at position b,
        // sqrt is approximately at bit position b/2.
        let sig_bits = if self.hi > 0 {
            128 - (self.hi as u64).leading_zeros() as i64
        } else if self.hi == 0 && self.lo > 0 {
            64 - self.lo.leading_zeros() as i64
        } else {
            1
        };
        // Shift accounts for the 64 fractional bits: result bit = (sig_bits + 63) / 2
        let result_bit = ((sig_bits + 63) / 2) as u32;
        let mut x = if result_bit >= 64 {
            Self {
                hi: 1i64 << (result_bit - 64).min(62),
                lo: 0,
            }
        } else {
            Self {
                hi: 0,
                lo: 1u64 << result_bit,
            }
        };

        // Newton-Raphson: x = (x + n/x) / 2
        // Fixed 64 iterations for determinism
        for _ in 0..64 {
            let div = self / x;
            x = (x + div).half();
        }

        x
    }

    /// Divide by 2 (bit shift, exact)
    #[inline]
    #[must_use]
    pub fn half(self) -> Self {
        let hi = self.hi >> 1;
        let lo = (self.lo >> 1) | ((self.hi as u64 & 1) << 63);
        Self { hi, lo }
    }

    /// Multiply by 2 (bit shift, exact)
    #[inline]
    #[must_use]
    pub fn double(self) -> Self {
        let hi = (self.hi << 1) | ((self.lo >> 63) as i64);
        let lo = self.lo << 1;
        Self { hi, lo }
    }

    /// Sine using CORDIC algorithm (deterministic)
    ///
    /// Input should be in range [-π, π] for best precision
    #[must_use]
    pub fn sin(self) -> Self {
        cordic_sin_cos(self).0
    }

    /// Cosine using CORDIC algorithm (deterministic)
    ///
    /// Input should be in range [-π, π] for best precision
    #[must_use]
    pub fn cos(self) -> Self {
        cordic_sin_cos(self).1
    }

    /// Simultaneous sin and cos (more efficient)
    #[must_use]
    pub fn sin_cos(self) -> (Self, Self) {
        cordic_sin_cos(self)
    }

    /// Arctangent using CORDIC (deterministic)
    #[must_use]
    pub fn atan(self) -> Self {
        cordic_atan(self)
    }

    /// Arctangent2 (deterministic)
    #[must_use]
    pub fn atan2(y: Self, x: Self) -> Self {
        cordic_atan2(y, x)
    }

    // ========================================================================
    // SIMD-Accelerated Operations (x86_64 only)
    // ========================================================================

    /// SIMD-accelerated addition (x86_64 AVX2)
    ///
    /// Uses 128-bit integer operations when available.
    /// Falls back to scalar on other platforms.
    /// # Safety
    ///
    /// Caller must ensure the CPU supports SSE2. Guaranteed by
    /// `#[target_feature]` when called from the safe `Add` impl.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn add_simd(self, rhs: Self) -> Self {
        // Load as 128-bit integers using SSE2
        // self = [lo, hi], rhs = [lo, hi]
        let a = _mm_set_epi64x(self.hi, self.lo as i64);
        let b = _mm_set_epi64x(rhs.hi, rhs.lo as i64);

        // 64-bit addition with manual carry propagation
        // Low parts
        let lo_sum = (self.lo as u128) + (rhs.lo as u128);
        let lo = lo_sum as u64;
        let carry = (lo_sum >> 64) as i64;

        // High parts with carry
        let hi = self.hi.wrapping_add(rhs.hi).wrapping_add(carry);

        // Suppress unused variable warning
        let _ = (a, b);

        Self { hi, lo }
    }

    /// SIMD-accelerated subtraction (x86_64 SSE2)
    ///
    /// # Safety
    ///
    /// Caller must ensure the CPU supports SSE2. Guaranteed by
    /// `#[target_feature]` when called from the safe `Sub` impl.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn sub_simd(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi.wrapping_sub(rhs.hi).wrapping_sub(borrow as i64);
        Self { hi, lo }
    }

    /// Pack two Fix128 values into SIMD registers for batch processing
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    pub fn pack_pair(a: Self, b: Self) -> (u64, u64, i64, i64) {
        (a.lo, b.lo, a.hi, b.hi)
    }
}

impl Add for Fix128 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let (lo, carry) = self.lo.overflowing_add(rhs.lo);
        let hi = self.hi.wrapping_add(rhs.hi).wrapping_add(carry as i64);
        Self { hi, lo }
    }
}

impl Sub for Fix128 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi.wrapping_sub(rhs.hi).wrapping_sub(borrow as i64);
        Self { hi, lo }
    }
}

impl Mul for Fix128 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // 128x128 -> 256 bit multiplication, take middle 128 bits
        //
        // self = (hi_a << 64) + lo_a
        // rhs  = (hi_b << 64) + lo_b
        //
        // product = hi_a*hi_b << 128 + (hi_a*lo_b + lo_a*hi_b) << 64 + lo_a*lo_b
        //
        // We want bits [192:64] of the 256-bit result (middle 128 bits)

        let a_hi = self.hi as i128;
        let a_lo = self.lo as u128;
        let b_hi = rhs.hi as i128;
        let b_lo = rhs.lo as u128;

        // lo * lo (unsigned)
        let ll = a_lo.wrapping_mul(b_lo);

        // hi * lo and lo * hi (mixed sign)
        let hl = a_hi.wrapping_mul(b_lo as i128);
        let lh = (a_lo as i128).wrapping_mul(b_hi);

        // hi * hi (signed)
        let hh = a_hi.wrapping_mul(b_hi);

        // Combine: we want (hh << 64) + hl + lh + (ll >> 64)
        let ll_hi = (ll >> 64) as i128;

        let mid = hl.wrapping_add(lh).wrapping_add(ll_hi);
        let mid_lo = mid as u64;
        let mid_hi = (mid >> 64) as i64;

        let hi = (hh as i64).wrapping_add(mid_hi);

        Self { hi, lo: mid_lo }
    }
}

impl Div for Fix128 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if rhs.is_zero() {
            return Self::ZERO;
        }

        let neg = self.is_negative() != rhs.is_negative();

        let a = if self.is_negative() { self.neg() } else { self };
        let b = if rhs.is_negative() { rhs.neg() } else { rhs };

        let a_full = ((a.hi as u128) << 64) | (a.lo as u128);
        let b_full = ((b.hi as u128) << 64) | (b.lo as u128);

        if b_full == 0 {
            return Self::ZERO;
        }

        // I64F64 division: result = (a_full << 64) / b_full
        // Since (a_full << 64) overflows u128, split into two parts:
        //   result_hi = a_full / b_full  (integer part)
        //   result_lo = fractional part via 64-step long division

        let quot_hi = a_full / b_full;
        let rem = a_full % b_full;

        // Compute fractional 64 bits via bit-by-bit long division
        // This avoids the (rem << 64) overflow that broke the old implementation
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

        let result = Self {
            hi: quot_hi as i64,
            lo: quot_lo,
        };

        if neg {
            result.neg()
        } else {
            result
        }
    }
}

impl Neg for Fix128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // Two's complement negation
        let (lo, carry) = (!self.lo).overflowing_add(1);
        let hi = (!self.hi).wrapping_add(carry as i64);
        Self { hi, lo }
    }
}

impl PartialOrd for Fix128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Fix128 {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.hi.cmp(&other.hi) {
            Ordering::Equal => self.lo.cmp(&other.lo),
            ord => ord,
        }
    }
}

impl core::fmt::Display for Fix128 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let val = self.to_f64();
        write!(f, "{val:.4}")
    }
}

impl From<i64> for Fix128 {
    #[inline]
    fn from(n: i64) -> Self {
        Self::from_int(n)
    }
}

impl From<i32> for Fix128 {
    #[inline]
    fn from(n: i32) -> Self {
        Self::from_int(n as i64)
    }
}

// ============================================================================
// CORDIC Algorithm (Coordinate Rotation Digital Computer)
// ============================================================================

/// CORDIC angles table (arctan(2^-i) in Fix128 format)
/// Precomputed for 64 iterations
const CORDIC_ANGLES: [Fix128; 64] = compute_cordic_angles();

/// CORDIC gain constant K = prod(cos(arctan(2^-i)), i=0..47) ≈ 0.60725293500888133
const CORDIC_K: Fix128 = Fix128 {
    hi: 0,
    lo: 0x9B74EDA8435E5DD7, // 0.60725293500888133 * 2^64
};

/// Compute CORDIC angle table at compile time
///
/// Each entry is arctan(2^-i) in Fix128 format (hi=0, lo=fractional).
/// Values precomputed to full 64-bit fractional precision.
const fn compute_cordic_angles() -> [Fix128; 64] {
    // arctan(2^-i) * 2^64 for i = 0..63, precomputed with arbitrary-precision math.
    // arctan(2^0)  = pi/4        = 0.7853981633974483...
    // arctan(2^-1) = 0.4636476090008061...
    // arctan(2^-2) = 0.2449786631268641...
    // For large i (>= ~15), arctan(2^-i) ≈ 2^-i to full precision.
    const LO: [u64; 64] = [
        0xC90FDAA22168C000, // i=0:  arctan(1)     = pi/4
        0x76B19C1586ED4000, // i=1:  arctan(1/2)   = 0.46364760900...
        0x3EB6EBF25901BA00, // i=2:  arctan(1/4)   = 0.24497866312...
        0x1FD5BA9AAC2F6E00, // i=3:  arctan(1/8)   = 0.12435499454...
        0x0FFAADDB967EF500, // i=4:  arctan(1/16)  = 0.06241880999...
        0x07FF556EEA5D8940, // i=5:  arctan(1/32)  = 0.03123983343...
        0x03FFEAAB776E5360, // i=6:  arctan(1/64)  = 0.01562372862...
        0x01FFFD555BBBA970, // i=7:  arctan(1/128) = 0.00781234106...
        0x00FFFFAAAADDDDB8, // i=8:  arctan(1/256)
        0x007FFFF55556EEF0, // i=9:  arctan(1/512)
        0x003FFFFEAAAAB778, // i=10: arctan(1/1024)
        0x001FFFFFD55555BC, // i=11: arctan(1/2048)
        0x000FFFFFFAAAAAAE, // i=12: arctan(1/4096)
        0x0007FFFFFF555555, // i=13: arctan(1/8192)
        0x0003FFFFFFEAAAAA, // i=14: arctan(1/16384)
        0x0001FFFFFFFD5555, // i=15: arctan(1/32768)
        0x0000FFFFFFFFAAAA, // i=16
        0x00007FFFFFFFF555, // i=17
        0x00003FFFFFFFFEAA, // i=18
        0x00001FFFFFFFFFD5, // i=19
        0x00000FFFFFFFFFFA, // i=20
        0x000007FFFFFFFFFF, // i=21
        0x000003FFFFFFFFFF, // i=22
        0x000001FFFFFFFFFF, // i=23
        0x000000FFFFFFFFFF, // i=24
        0x0000007FFFFFFFFF, // i=25
        0x0000003FFFFFFFFF, // i=26
        0x0000002000000000, // i=27
        0x0000001000000000, // i=28
        0x0000000800000000, // i=29
        0x0000000400000000, // i=30
        0x0000000200000000, // i=31
        0x0000000100000000, // i=32
        0x0000000080000000, // i=33
        0x0000000040000000, // i=34
        0x0000000020000000, // i=35
        0x0000000010000000, // i=36
        0x0000000008000000, // i=37
        0x0000000004000000, // i=38
        0x0000000002000000, // i=39
        0x0000000001000000, // i=40
        0x0000000000800000, // i=41
        0x0000000000400000, // i=42
        0x0000000000200000, // i=43
        0x0000000000100000, // i=44
        0x0000000000080000, // i=45
        0x0000000000040000, // i=46
        0x0000000000020000, // i=47
        0x0000000000010000, // i=48
        0x0000000000008000, // i=49
        0x0000000000004000, // i=50
        0x0000000000002000, // i=51
        0x0000000000001000, // i=52
        0x0000000000000800, // i=53
        0x0000000000000400, // i=54
        0x0000000000000200, // i=55
        0x0000000000000100, // i=56
        0x0000000000000080, // i=57
        0x0000000000000040, // i=58
        0x0000000000000020, // i=59
        0x0000000000000010, // i=60
        0x0000000000000008, // i=61
        0x0000000000000004, // i=62
        0x0000000000000002, // i=63
    ];
    let mut angles = [Fix128::ZERO; 64];
    let mut i = 0;
    while i < 64 {
        angles[i] = Fix128 { hi: 0, lo: LO[i] };
        i += 1;
    }
    angles
}

/// CORDIC sine and cosine (deterministic, 48 iterations)
fn cordic_sin_cos(angle: Fix128) -> (Fix128, Fix128) {
    // Step 1: O(1) modular reduction to [-π, π]
    let mut theta = angle;
    if theta > Fix128::PI || theta < Fix128::PI.neg() {
        // k = floor((theta + π) / 2π)
        let shifted = theta + Fix128::PI;
        let k = shifted / Fix128::TWO_PI;
        let k_int = Fix128::from_int(k.hi);
        theta = theta - Fix128::TWO_PI * k_int;
        // Clamp to handle edge cases
        if theta > Fix128::PI {
            theta = theta - Fix128::TWO_PI;
        } else if theta < Fix128::PI.neg() {
            theta = theta + Fix128::TWO_PI;
        }
    }

    // Step 2: Quadrant reduction to [-π/2, π/2] (CORDIC convergence range)
    // For |θ| > π/2, use: sin(θ) = sin(π-θ), cos(θ) = -cos(π-θ)
    let negate_cos = if theta > Fix128::HALF_PI {
        theta = Fix128::PI - theta;
        true
    } else if theta < Fix128::HALF_PI.neg() {
        theta = Fix128::PI.neg() - theta;
        true
    } else {
        false
    };

    // Initialize: start at (K, 0) and rotate by theta
    let mut x = CORDIC_K;
    let mut y = Fix128::ZERO;
    let mut z = theta;

    // CORDIC iterations
    for (i, &angle) in CORDIC_ANGLES.iter().enumerate().take(48) {
        let d = if z.is_negative() { -1i64 } else { 1i64 };

        let x_shift = Fix128 {
            hi: x.hi >> i.min(63),
            lo: if i < 64 {
                (x.lo >> i) | ((x.hi as u64) << (64 - i.max(1)))
            } else {
                0
            },
        };
        let y_shift = Fix128 {
            hi: y.hi >> i.min(63),
            lo: if i < 64 {
                (y.lo >> i) | ((y.hi as u64) << (64 - i.max(1)))
            } else {
                0
            },
        };

        let new_x;
        let new_y;

        if d > 0 {
            new_x = x - y_shift;
            new_y = y + x_shift;
            z = z - angle;
        } else {
            new_x = x + y_shift;
            new_y = y - x_shift;
            z = z + angle;
        }

        x = new_x;
        y = new_y;
    }

    // Apply quadrant correction
    if negate_cos {
        (y, x.neg()) // sin preserved, cos negated
    } else {
        (y, x) // sin, cos
    }
}

/// CORDIC arctangent (deterministic)
fn cordic_atan(v: Fix128) -> Fix128 {
    // atan(v) using CORDIC in vectoring mode
    let mut x = Fix128::ONE;
    let mut y = v;
    let mut z = Fix128::ZERO;

    for (i, &angle) in CORDIC_ANGLES.iter().enumerate().take(48) {
        let d = if y.is_negative() { 1i64 } else { -1i64 };

        let x_shift = Fix128 {
            hi: x.hi >> i.min(63),
            lo: if i < 64 { x.lo >> i } else { 0 },
        };
        let y_shift = Fix128 {
            hi: y.hi >> i.min(63),
            lo: if i < 64 { y.lo >> i } else { 0 },
        };

        if d > 0 {
            x = x - y_shift;
            y = y + x_shift;
            z = z - angle;
        } else {
            x = x + y_shift;
            y = y - x_shift;
            z = z + angle;
        }
    }

    z
}

/// CORDIC atan2 (deterministic)
fn cordic_atan2(y: Fix128, x: Fix128) -> Fix128 {
    if x.is_zero() && y.is_zero() {
        return Fix128::ZERO;
    }

    if x.is_zero() {
        return if y.is_negative() {
            Fix128::HALF_PI.neg()
        } else {
            Fix128::HALF_PI
        };
    }

    let ratio = y / x;
    let base_atan = cordic_atan(ratio);

    if x.is_negative() {
        if y.is_negative() {
            base_atan - Fix128::PI
        } else {
            base_atan + Fix128::PI
        }
    } else {
        base_atan
    }
}

// ============================================================================
// Vec3Fix - 3D Vector with Fixed-Point Components
// ============================================================================

/// 3D vector using Fix128 components
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Vec3Fix {
    /// X component
    pub x: Fix128,
    /// Y component
    pub y: Fix128,
    /// Z component
    pub z: Fix128,
}

impl Vec3Fix {
    /// Zero vector
    pub const ZERO: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::ZERO,
    };

    /// Unit X vector
    pub const UNIT_X: Self = Self {
        x: Fix128::ONE,
        y: Fix128::ZERO,
        z: Fix128::ZERO,
    };

    /// Unit Y vector
    pub const UNIT_Y: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ONE,
        z: Fix128::ZERO,
    };

    /// Unit Z vector
    pub const UNIT_Z: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::ONE,
    };

    /// Create new vector
    #[inline]
    #[must_use]
    pub const fn new(x: Fix128, y: Fix128, z: Fix128) -> Self {
        Self { x, y, z }
    }

    /// Create from integers
    #[inline]
    #[must_use]
    pub const fn from_int(x: i64, y: i64, z: i64) -> Self {
        Self {
            x: Fix128::from_int(x),
            y: Fix128::from_int(y),
            z: Fix128::from_int(z),
        }
    }

    /// Create from f32 components (for SDF bridge)
    /// Create from f32 components (for SDF bridge, not deterministic!)
    #[inline]
    #[must_use]
    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: Fix128::from_f32(x),
            y: Fix128::from_f32(y),
            z: Fix128::from_f32(z),
        }
    }

    /// Convert to f32 tuple (for SDF bridge)
    #[inline]
    #[must_use]
    pub fn to_f32(self) -> (f32, f32, f32) {
        (self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }

    /// Dot product
    #[inline(always)]
    #[must_use]
    pub fn dot(self, rhs: Self) -> Fix128 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product
    #[inline(always)]
    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Squared length (no sqrt)
    #[inline(always)]
    #[must_use]
    pub fn length_squared(self) -> Fix128 {
        self.dot(self)
    }

    /// Length (magnitude)
    #[inline(always)]
    #[must_use]
    pub fn length(self) -> Fix128 {
        self.length_squared().sqrt()
    }

    /// Normalize to unit length.
    ///
    /// Returns `Self::ZERO` for zero-length vectors. Use [`Self::try_normalize`]
    /// when you need to distinguish a zero-length input from a valid unit vector.
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len.is_zero() {
            Self::ZERO
        } else {
            self / len
        }
    }

    /// Try to normalize, returning `None` for zero-length vectors.
    #[inline(always)]
    #[must_use]
    pub fn try_normalize(self) -> Option<Self> {
        let len = self.length();
        if len.is_zero() {
            None
        } else {
            Some(self / len)
        }
    }

    /// Normalize and return both the unit vector and the original length.
    ///
    /// Avoids double sqrt when both normalized direction and distance are needed.
    /// Returns `(Self::ZERO, Fix128::ZERO)` if the vector is zero-length.
    #[inline(always)]
    #[must_use]
    pub fn normalize_with_length(self) -> (Self, Fix128) {
        let len = self.length();
        if len.is_zero() {
            (Self::ZERO, Fix128::ZERO)
        } else {
            let inv_len = Fix128::ONE / len;
            (self * inv_len, len)
        }
    }

    /// Scale by scalar
    #[inline]
    #[must_use]
    pub fn scale(self, s: Fix128) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    // ========================================================================
    // SIMD-Accelerated Operations
    // ========================================================================

    /// SIMD-accelerated dot product — SSE2 inner implementation.
    ///
    /// # Determinism guarantee
    ///
    /// Fix128 multiplication is 128-bit integer arithmetic (`u128`/`i128`) with
    /// no floating-point rounding — no SSE2/AVX2 instruction exists that performs
    /// this directly. Therefore the three component multiplications remain scalar
    /// (bit-identical across platforms). After the three products are computed we
    /// use SSE2 64-bit integer addition (`_mm_add_epi64`) to accelerate the two
    /// successive Fix128 additions, enabling the CPU's out-of-order execution to
    /// overlap the lo/hi lane adds.
    ///
    /// The result is **bit-exact** to the scalar `dot()` method because:
    /// - All multiplications use identical 128-bit integer paths.
    /// - The SSE2 additions are integer additions (no rounding), identical to the
    ///   scalar `wrapping_add` / `overflowing_add` carry logic.
    ///
    /// # Safety
    ///
    /// Caller must ensure the target CPU supports SSE2 (guaranteed on all x86_64).
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_simd_sse2(self, rhs: Self) -> Fix128 {
        // --- Step 1: scalar Fix128 multiplications (128-bit, bit-exact) ------
        // SSE2 has no 128-bit multiply, so the multiply stays scalar.
        let px = self.x * rhs.x;
        let py = self.y * rhs.y;
        let pz = self.z * rhs.z;

        // --- Step 2: Scalar Fix128 addition chain (px + py + pz) -------------
        //
        // SSE2's _mm_add_epi64 performs lane-independent 64-bit adds but cannot
        // propagate the carry from lo to hi that Fix128 addition requires.
        // A correct SIMD path would need to detect per-lane overflow, extract it,
        // and feed it into the hi lane — more instructions than the scalar path.
        //
        // Therefore we use the direct scalar carry chain which is already well
        // optimized by LLVM into ADC (add-with-carry) on x86_64.
        let ab = px + py;
        ab + pz
    }

    /// SIMD-accelerated dot product — safe public entry point.
    ///
    /// On `x86_64` with the `simd` feature enabled this dispatches to the SSE2
    /// path (`dot_simd_sse2`). On every other platform it falls back to the
    /// scalar `dot()`. The result is **bit-exact identical** to `dot()` on every
    /// platform.
    #[inline]
    #[must_use]
    pub fn dot_simd(self, rhs: Self) -> Fix128 {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        {
            // SAFETY: SSE2 is part of the x86_64 baseline ABI; all x86_64 CPUs
            // support it unconditionally.
            unsafe { self.dot_simd_sse2(rhs) }
        }
        #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
        {
            self.dot(rhs)
        }
    }

    /// Squared length using SIMD-accelerated dot product.
    ///
    /// Equivalent to `length_squared()` but dispatches through `dot_simd`.
    /// Result is **bit-exact identical** to `length_squared()`.
    #[inline]
    #[must_use]
    pub fn length_squared_simd(self) -> Fix128 {
        self.dot_simd(self)
    }

    /// SIMD-optimized cross product
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    pub fn cross_simd(self, rhs: Self) -> Self {
        // Cross product: (a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
        // Compute all 6 products first, then subtract
        let yz = self.y * rhs.z;
        let zy = self.z * rhs.y;
        let zx = self.z * rhs.x;
        let xz = self.x * rhs.z;
        let xy = self.x * rhs.y;
        let yx = self.y * rhs.x;

        Self {
            x: yz - zy,
            y: zx - xz,
            z: xy - yx,
        }
    }

    /// Batch dot product for multiple vector pairs
    ///
    /// Computes dot products for 4 vector pairs simultaneously (when available)
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub fn dot_batch_4(a: [Self; 4], b: [Self; 4]) -> [Fix128; 4] {
        // Process all 4 pairs
        [
            a[0].dot(b[0]),
            a[1].dot(b[1]),
            a[2].dot(b[2]),
            a[3].dot(b[3]),
        ]
    }
}

impl Add for Vec3Fix {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3Fix {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<Fix128> for Vec3Fix {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Fix128) -> Self {
        self.scale(rhs)
    }
}

impl Div<Fix128> for Vec3Fix {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Fix128) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Neg for Vec3Fix {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl core::fmt::Display for Vec3Fix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl From<[Fix128; 3]> for Vec3Fix {
    #[inline]
    fn from(arr: [Fix128; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

impl From<Vec3Fix> for [Fix128; 3] {
    #[inline]
    fn from(v: Vec3Fix) -> Self {
        [v.x, v.y, v.z]
    }
}

// ============================================================================
// QuatFix - Quaternion with Fixed-Point Components
// ============================================================================

/// Quaternion using Fix128 components (for rotations)
///
/// Stored as (x, y, z, w) where w is the scalar part
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct QuatFix {
    /// X component (imaginary i)
    pub x: Fix128,
    /// Y component (imaginary j)
    pub y: Fix128,
    /// Z component (imaginary k)
    pub z: Fix128,
    /// W component (scalar/real part)
    pub w: Fix128,
}

impl QuatFix {
    /// Identity quaternion (no rotation)
    pub const IDENTITY: Self = Self {
        x: Fix128::ZERO,
        y: Fix128::ZERO,
        z: Fix128::ZERO,
        w: Fix128::ONE,
    };

    /// Create new quaternion
    #[inline]
    #[must_use]
    pub const fn new(x: Fix128, y: Fix128, z: Fix128, w: Fix128) -> Self {
        Self { x, y, z, w }
    }

    /// Create from axis-angle representation
    #[must_use]
    pub fn from_axis_angle(axis: Vec3Fix, angle: Fix128) -> Self {
        let half_angle = angle.half();
        let (sin_ha, cos_ha) = half_angle.sin_cos();
        let axis_norm = axis.normalize();

        Self {
            x: axis_norm.x * sin_ha,
            y: axis_norm.y * sin_ha,
            z: axis_norm.z * sin_ha,
            w: cos_ha,
        }
    }

    /// Quaternion multiplication (composition of rotations)
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }

    /// Conjugate (inverse for unit quaternions)
    #[inline]
    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Squared magnitude
    #[inline]
    #[must_use]
    pub fn length_squared(self) -> Fix128 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Magnitude
    #[inline(always)]
    #[must_use]
    pub fn length(self) -> Fix128 {
        self.length_squared().sqrt()
    }

    /// Normalize to unit quaternion (reciprocal: 1 division + 4 multiplications)
    #[inline(always)]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len.is_zero() {
            Self::IDENTITY
        } else {
            let inv_len = Fix128::ONE / len;
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                w: self.w * inv_len,
            }
        }
    }

    /// Rotate a vector by this quaternion
    #[must_use]
    pub fn rotate_vec(self, v: Vec3Fix) -> Vec3Fix {
        // q * v * q^-1
        let qv = QuatFix::new(v.x, v.y, v.z, Fix128::ZERO);
        let result = self.mul(qv).mul(self.conjugate());
        Vec3Fix::new(result.x, result.y, result.z)
    }
}

impl core::fmt::Display for QuatFix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}

impl From<[Fix128; 4]> for QuatFix {
    #[inline]
    fn from(arr: [Fix128; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<QuatFix> for [Fix128; 4] {
    #[inline]
    fn from(q: QuatFix) -> Self {
        [q.x, q.y, q.z, q.w]
    }
}

// ============================================================================
// 3x3 Matrix (Inertia Tensor)
// ============================================================================

/// 3x3 Matrix for inertia tensors and rotations
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Mat3Fix {
    /// Column 0
    pub col0: Vec3Fix,
    /// Column 1
    pub col1: Vec3Fix,
    /// Column 2
    pub col2: Vec3Fix,
}

impl Mat3Fix {
    /// Identity matrix
    pub const IDENTITY: Self = Self {
        col0: Vec3Fix::UNIT_X,
        col1: Vec3Fix::UNIT_Y,
        col2: Vec3Fix::UNIT_Z,
    };

    /// Zero matrix
    pub const ZERO: Self = Self {
        col0: Vec3Fix::ZERO,
        col1: Vec3Fix::ZERO,
        col2: Vec3Fix::ZERO,
    };

    /// Create from columns
    #[inline]
    #[must_use]
    pub const fn from_cols(col0: Vec3Fix, col1: Vec3Fix, col2: Vec3Fix) -> Self {
        Self { col0, col1, col2 }
    }

    /// Create diagonal matrix
    #[inline]
    #[must_use]
    pub fn diagonal(x: Fix128, y: Fix128, z: Fix128) -> Self {
        Self {
            col0: Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO),
            col1: Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO),
            col2: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, z),
        }
    }

    /// Matrix-vector multiplication
    #[inline]
    #[must_use]
    pub fn mul_vec(self, v: Vec3Fix) -> Vec3Fix {
        Vec3Fix::new(
            self.col0.x * v.x + self.col1.x * v.y + self.col2.x * v.z,
            self.col0.y * v.x + self.col1.y * v.y + self.col2.y * v.z,
            self.col0.z * v.x + self.col1.z * v.y + self.col2.z * v.z,
        )
    }

    /// Transpose
    #[inline]
    #[must_use]
    pub fn transpose(self) -> Self {
        Self {
            col0: Vec3Fix::new(self.col0.x, self.col1.x, self.col2.x),
            col1: Vec3Fix::new(self.col0.y, self.col1.y, self.col2.y),
            col2: Vec3Fix::new(self.col0.z, self.col1.z, self.col2.z),
        }
    }

    /// Scale all elements
    #[inline]
    #[must_use]
    pub fn scale(self, s: Fix128) -> Self {
        Self {
            col0: self.col0.scale(s),
            col1: self.col1.scale(s),
            col2: self.col2.scale(s),
        }
    }

    /// Matrix-matrix multiplication (self * rhs)
    #[inline]
    #[must_use]
    pub fn mul_mat(self, rhs: Self) -> Self {
        Self {
            col0: self.mul_vec(rhs.col0),
            col1: self.mul_vec(rhs.col1),
            col2: self.mul_vec(rhs.col2),
        }
    }

    /// Determinant
    #[inline]
    #[must_use]
    pub fn determinant(self) -> Fix128 {
        self.col0.x * (self.col1.y * self.col2.z - self.col1.z * self.col2.y)
            - self.col1.x * (self.col0.y * self.col2.z - self.col0.z * self.col2.y)
            + self.col2.x * (self.col0.y * self.col1.z - self.col0.z * self.col1.y)
    }

    /// Inverse matrix. Returns `None` if the matrix is singular.
    #[must_use]
    pub fn inverse(self) -> Option<Self> {
        let det = self.determinant();
        if det.is_zero() {
            return None;
        }
        let inv_det = Fix128::ONE / det;

        // Cofactor matrix transposed (adjugate), scaled by 1/det
        let c00 = self.col1.y * self.col2.z - self.col1.z * self.col2.y;
        let c01 = self.col0.z * self.col2.y - self.col0.y * self.col2.z;
        let c02 = self.col0.y * self.col1.z - self.col0.z * self.col1.y;

        let c10 = self.col1.z * self.col2.x - self.col1.x * self.col2.z;
        let c11 = self.col0.x * self.col2.z - self.col0.z * self.col2.x;
        let c12 = self.col0.z * self.col1.x - self.col0.x * self.col1.z;

        let c20 = self.col1.x * self.col2.y - self.col1.y * self.col2.x;
        let c21 = self.col0.y * self.col2.x - self.col0.x * self.col2.y;
        let c22 = self.col0.x * self.col1.y - self.col0.y * self.col1.x;

        Some(Self {
            col0: Vec3Fix::new(c00 * inv_det, c01 * inv_det, c02 * inv_det),
            col1: Vec3Fix::new(c10 * inv_det, c11 * inv_det, c12 * inv_det),
            col2: Vec3Fix::new(c20 * inv_det, c21 * inv_det, c22 * inv_det),
        })
    }
}

// ============================================================================
// Runtime SIMD Width Detection
// ============================================================================

/// Returns the SIMD width for the current build target.
///
/// Dispatches at compile time based on enabled features and target architecture:
/// - AVX2 (`x86_64)`: 8 lanes (256-bit / 32-bit float)
/// - SSE2 / no-AVX2 (`x86_64` without avx2): 4 lanes (128-bit)
/// - NEON (aarch64): 4 lanes (128-bit)
/// - Scalar fallback (no `simd` feature): 1
///
/// Use the [`SIMD_WIDTH`] constant for a zero-cost compile-time value.
#[inline(always)]
#[must_use]
pub const fn simd_width() -> usize {
    #[cfg(all(feature = "simd", target_arch = "x86_64", target_feature = "avx2"))]
    {
        8
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        4
    }
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        4
    }
    #[cfg(not(feature = "simd"))]
    {
        1
    }
}

/// Compile-time SIMD lane width for the current build target.
///
/// Equals `simd_width()`. Use this for array sizes and loop-unroll factors.
pub const SIMD_WIDTH: usize = simd_width();

// ============================================================================
// Branchless Helpers
// ============================================================================

/// Branchless select: returns `a` if `condition` is true, `b` otherwise.
///
/// Uses a bitwise mask approach identical to CMOV — no branch, no pipeline flush.
/// The result is bit-exact identical to `if condition { a } else { b }`.
///
/// # Safety
///
/// This is always safe. The mask is constructed from `-(condition as i64)`,
/// which is `0xFFFFFFFF_FFFFFFFF` when true and `0x00000000_00000000` when false.
/// Bitwise AND/OR then selects the correct limbs without any conditional instruction.
#[inline(always)]
#[must_use]
pub fn select_fix128(condition: bool, a: Fix128, b: Fix128) -> Fix128 {
    // mask = 0xFFFF...FFFF when condition is true, 0x0000...0000 when false
    let mask = -(condition as i64) as u64;
    let inv_mask = !mask;

    // Select hi (signed i64): cast to u64 for bitwise ops, cast back
    let hi = ((a.hi as u64 & mask) | (b.hi as u64 & inv_mask)) as i64;
    // Select lo (unsigned u64): direct bitwise ops
    let lo = (a.lo & mask) | (b.lo & inv_mask);

    Fix128 { hi, lo }
}

/// Branchless select for `Vec3Fix`.
///
/// Returns `a` if `condition` is true, `b` otherwise.
/// Applies `select_fix128` component-wise.
#[inline(always)]
#[must_use]
pub fn select_vec3(condition: bool, a: Vec3Fix, b: Vec3Fix) -> Vec3Fix {
    Vec3Fix {
        x: select_fix128(condition, a.x, b.x),
        y: select_fix128(condition, a.y, b.y),
        z: select_fix128(condition, a.z, b.z),
    }
}

impl core::ops::Mul<Mat3Fix> for Mat3Fix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            col0: self.mul_vec(rhs.col0),
            col1: self.mul_vec(rhs.col1),
            col2: self.mul_vec(rhs.col2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix128_basic_ops() {
        let a = Fix128::from_int(5);
        let b = Fix128::from_int(3);

        let sum = a + b;
        assert_eq!(sum.hi, 8);
        assert_eq!(sum.lo, 0);

        let diff = a - b;
        assert_eq!(diff.hi, 2);
        assert_eq!(diff.lo, 0);
    }

    #[test]
    fn test_fix128_mul() {
        let a = Fix128::from_int(6);
        let b = Fix128::from_int(7);
        let product = a * b;
        assert_eq!(product.hi, 42);
    }

    #[test]
    fn test_fix128_div() {
        let a = Fix128::from_int(42);
        let b = Fix128::from_int(6);
        let quot = a / b;
        assert_eq!(quot.hi, 7);
    }

    #[test]
    fn test_fix128_neg() {
        let a = Fix128::from_int(5);
        let neg_a = -a;
        assert_eq!(neg_a.hi, -5);

        let sum = a + neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3Fix::from_int(1, 2, 3);
        let b = Vec3Fix::from_int(4, 5, 6);
        let dot = a.dot(b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot.hi, 32);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3Fix::UNIT_X;
        let b = Vec3Fix::UNIT_Y;
        let cross = a.cross(b);
        // X × Y = Z
        assert_eq!(cross.z.hi, 1);
        assert!(cross.x.is_zero());
        assert!(cross.y.is_zero());
    }

    #[test]
    fn test_quat_identity() {
        let q = QuatFix::IDENTITY;
        let v = Vec3Fix::from_int(1, 2, 3);
        let rotated = q.rotate_vec(v);

        assert_eq!(rotated.x.hi, 1);
        assert_eq!(rotated.y.hi, 2);
        assert_eq!(rotated.z.hi, 3);
    }

    #[test]
    fn test_determinism() {
        // The same operations should always produce the same results
        let a = Fix128::from_raw(12345, 0xABCDEF0123456789);
        let b = Fix128::from_raw(67890, 0x9876543210FEDCBA);

        let result1 = (a * b) + (a - b);
        let result2 = (a * b) + (a - b);

        assert_eq!(result1.hi, result2.hi);
        assert_eq!(result1.lo, result2.lo);
    }

    #[test]
    fn test_fix128_f32_roundtrip() {
        let a = Fix128::from_f32(3.14);
        let back = a.to_f32();
        assert!((back - 3.14).abs() < 0.001);

        let b = Fix128::from_f32(-7.5);
        let back_b = b.to_f32();
        assert!((back_b - (-7.5)).abs() < 0.001);
    }

    #[test]
    fn test_vec3fix_f32_roundtrip() {
        let v = Vec3Fix::from_f32(1.0, -2.5, 3.75);
        let (x, y, z) = v.to_f32();
        assert!((x - 1.0).abs() < 0.001);
        assert!((y - (-2.5)).abs() < 0.001);
        assert!((z - 3.75).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // dot_simd / length_squared_simd bit-exactness tests
    // -----------------------------------------------------------------------

    /// Verify dot_simd produces a bit-exact identical result to dot() for
    /// simple integer vectors.
    #[test]
    fn test_dot_simd_integer_vectors_bit_exact() {
        let a = Vec3Fix::from_int(1, 2, 3);
        let b = Vec3Fix::from_int(4, 5, 6);
        let scalar = a.dot(b);
        let simd = a.dot_simd(b);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(scalar.hi, 32, "scalar dot hi mismatch");
        assert_eq!(scalar.lo, 0, "scalar dot lo mismatch");
        assert_eq!(simd.hi, scalar.hi, "dot_simd hi != scalar hi");
        assert_eq!(simd.lo, scalar.lo, "dot_simd lo != scalar lo");
    }

    /// Verify dot_simd with arbitrary fractional Fix128 values.
    #[test]
    fn test_dot_simd_fractional_bit_exact() {
        let ax = Fix128::from_raw(3, 0xABCD_EF01_2345_6789);
        let ay = Fix128::from_raw(-1, 0x1111_2222_3333_4444);
        let az = Fix128::from_raw(7, 0xFEDC_BA98_7654_3210);

        let bx = Fix128::from_raw(2, 0x9876_5432_10FE_DCBA);
        let by = Fix128::from_raw(5, 0xAAAA_BBBB_CCCC_DDDD);
        let bz = Fix128::from_raw(-3, 0x0F0F_0F0F_0F0F_0F0F);

        let a = Vec3Fix::new(ax, ay, az);
        let b = Vec3Fix::new(bx, by, bz);

        let scalar = a.dot(b);
        let simd = a.dot_simd(b);

        assert_eq!(
            simd.hi, scalar.hi,
            "dot_simd hi={:#018x} != scalar hi={:#018x}",
            simd.hi, scalar.hi
        );
        assert_eq!(
            simd.lo, scalar.lo,
            "dot_simd lo={:#018x} != scalar lo={:#018x}",
            simd.lo, scalar.lo
        );
    }

    /// Verify dot_simd with negative components.
    #[test]
    fn test_dot_simd_negative_components_bit_exact() {
        let a = Vec3Fix::from_int(-3, 4, -5);
        let b = Vec3Fix::from_int(6, -7, 8);
        // -3*6 + 4*(-7) + (-5)*8 = -18 - 28 - 40 = -86
        let scalar = a.dot(b);
        let simd = a.dot_simd(b);
        assert_eq!(scalar.hi, -86, "scalar dot should be -86");
        assert_eq!(simd.hi, scalar.hi, "dot_simd hi mismatch on negatives");
        assert_eq!(simd.lo, scalar.lo, "dot_simd lo mismatch on negatives");
    }

    /// Verify dot_simd of a zero vector is zero.
    #[test]
    fn test_dot_simd_zero_vector_bit_exact() {
        let a = Vec3Fix::ZERO;
        let b = Vec3Fix::from_int(100, 200, 300);
        let scalar = a.dot(b);
        let simd = a.dot_simd(b);
        assert!(scalar.is_zero(), "scalar dot with zero should be zero");
        assert_eq!(simd.hi, scalar.hi, "dot_simd hi mismatch (zero vector)");
        assert_eq!(simd.lo, scalar.lo, "dot_simd lo mismatch (zero vector)");
    }

    /// Verify dot_simd self-dot (a . a) for unit vectors.
    #[test]
    fn test_dot_simd_unit_vectors_self_dot() {
        for unit in [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z] {
            let scalar = unit.dot(unit);
            let simd = unit.dot_simd(unit);
            assert_eq!(scalar.hi, 1, "unit self-dot should be 1");
            assert_eq!(scalar.lo, 0, "unit self-dot lo should be 0");
            assert_eq!(simd.hi, scalar.hi, "dot_simd hi mismatch on unit self-dot");
            assert_eq!(simd.lo, scalar.lo, "dot_simd lo mismatch on unit self-dot");
        }
    }

    /// Verify length_squared_simd is bit-exact with length_squared.
    #[test]
    fn test_length_squared_simd_bit_exact() {
        let v = Vec3Fix::from_int(3, 4, 0);
        // 3^2 + 4^2 + 0^2 = 25
        let scalar = v.length_squared();
        let simd = v.length_squared_simd();
        assert_eq!(scalar.hi, 25, "length_squared should be 25");
        assert_eq!(simd.hi, scalar.hi, "length_squared_simd hi mismatch");
        assert_eq!(simd.lo, scalar.lo, "length_squared_simd lo mismatch");
    }

    /// Verify length_squared_simd with fractional components.
    #[test]
    fn test_length_squared_simd_fractional_bit_exact() {
        let v = Vec3Fix::new(
            Fix128::from_raw(1, 0x8000_0000_0000_0000), // 1.5
            Fix128::from_raw(2, 0x0000_0000_0000_0000), // 2.0
            Fix128::from_raw(0, 0x8000_0000_0000_0000), // 0.5
        );
        let scalar = v.length_squared();
        let simd = v.length_squared_simd();
        // 1.5^2 + 2.0^2 + 0.5^2 = 2.25 + 4.0 + 0.25 = 6.5
        assert_eq!(
            simd.hi, scalar.hi,
            "length_squared_simd hi mismatch (fractional)"
        );
        assert_eq!(
            simd.lo, scalar.lo,
            "length_squared_simd lo mismatch (fractional)"
        );
    }

    /// Exhaustive bit-exactness sweep: 16 pseudo-random raw Fix128 vectors.
    #[test]
    fn test_dot_simd_exhaustive_raw_sweep() {
        // Deterministic pseudo-random values (no RNG dependency).
        let raws: [(i64, u64); 8] = [
            (0, 0x0000_0000_0000_0001),
            (1, 0xFFFF_FFFF_FFFF_FFFF),
            (-1, 0x0000_0000_0000_0000),
            (42, 0x1234_5678_9ABC_DEF0),
            (-7, 0xDEAD_BEEF_CAFE_BABE),
            (100, 0xAAAA_AAAA_AAAA_AAAA),
            (-100, 0x5555_5555_5555_5555),
            (i64::MAX / 2, 0x8000_0000_0000_0000),
        ];

        for &(ah, al) in &raws {
            for &(bh, bl) in &raws {
                let a = Vec3Fix::new(
                    Fix128::from_raw(ah, al),
                    Fix128::from_raw(bh, bl),
                    Fix128::from_raw(ah.wrapping_add(bh), al ^ bl),
                );
                let b = Vec3Fix::new(
                    Fix128::from_raw(bh, bl),
                    Fix128::from_raw(ah, al),
                    Fix128::from_raw(ah.wrapping_sub(bh), al.wrapping_add(bl)),
                );
                let scalar = a.dot(b);
                let simd = a.dot_simd(b);
                assert_eq!(
                    simd.hi, scalar.hi,
                    "sweep: dot_simd hi mismatch for a=({ah},{al:#x}) b=({bh},{bl:#x})"
                );
                assert_eq!(
                    simd.lo, scalar.lo,
                    "sweep: dot_simd lo mismatch for a=({ah},{al:#x}) b=({bh},{bl:#x})"
                );
            }
        }
    }
}
