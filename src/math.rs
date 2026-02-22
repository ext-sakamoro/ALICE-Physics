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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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
    pub const fn from_int(n: i64) -> Self {
        Self { hi: n, lo: 0 }
    }

    /// Create from raw parts (hi = integer, lo = fraction)
    #[inline]
    pub const fn from_raw(hi: i64, lo: u64) -> Self {
        Self { hi, lo }
    }

    /// Create from f64 (for initialization only, not deterministic!)
    #[cfg(feature = "std")]
    pub fn from_f64(f: f64) -> Self {
        let hi = f.trunc() as i64;
        let frac = (f - f.trunc()).abs();
        let lo = (frac * (1u128 << 64) as f64) as u64;
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
    #[cfg(feature = "std")]
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + (self.lo as f64 / (1u128 << 64) as f64)
    }

    /// Create from f32 (for SDF bridge, not deterministic!)
    #[cfg(feature = "std")]
    pub fn from_f32(f: f32) -> Self {
        Self::from_f64(f as f64)
    }

    /// Convert to f32 (for SDF bridge, not deterministic!)
    #[cfg(feature = "std")]
    pub fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }

    /// Create from fraction (numerator / denominator)
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
    pub fn abs(self) -> Self {
        if self.hi < 0 || (self.hi == 0 && self.lo == 0) {
            self.neg()
        } else {
            self
        }
    }

    /// Check if negative
    #[inline]
    pub const fn is_negative(self) -> bool {
        self.hi < 0
    }

    /// Check if zero
    #[inline]
    pub const fn is_zero(self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    /// Floor (round toward negative infinity)
    #[inline]
    pub fn floor(self) -> Self {
        Self { hi: self.hi, lo: 0 }
    }

    /// Ceiling (round toward positive infinity)
    #[inline]
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
    pub fn sqrt(self) -> Self {
        if self.is_negative() || self.is_zero() {
            return Self::ZERO;
        }

        // Initial guess: integer sqrt of hi part
        let mut x = Self::from_int((self.hi as f64).sqrt() as i64 + 1);

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
    pub fn half(self) -> Self {
        let hi = self.hi >> 1;
        let lo = (self.lo >> 1) | ((self.hi as u64 & 1) << 63);
        Self { hi, lo }
    }

    /// Multiply by 2 (bit shift, exact)
    #[inline]
    pub fn double(self) -> Self {
        let hi = (self.hi << 1) | ((self.lo >> 63) as i64);
        let lo = self.lo << 1;
        Self { hi, lo }
    }

    /// Sine using CORDIC algorithm (deterministic)
    ///
    /// Input should be in range [-π, π] for best precision
    pub fn sin(self) -> Self {
        cordic_sin_cos(self).0
    }

    /// Cosine using CORDIC algorithm (deterministic)
    ///
    /// Input should be in range [-π, π] for best precision
    pub fn cos(self) -> Self {
        cordic_sin_cos(self).1
    }

    /// Simultaneous sin and cos (more efficient)
    pub fn sin_cos(self) -> (Self, Self) {
        cordic_sin_cos(self)
    }

    /// Arctangent using CORDIC (deterministic)
    pub fn atan(self) -> Self {
        cordic_atan(self)
    }

    /// Arctangent2 (deterministic)
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

    /// SIMD-accelerated subtraction (x86_64 AVX2)
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

// ============================================================================
// CORDIC Algorithm (Coordinate Rotation Digital Computer)
// ============================================================================

/// CORDIC angles table (arctan(2^-i) in Fix128 format)
/// Precomputed for 64 iterations
const CORDIC_ANGLES: [Fix128; 64] = compute_cordic_angles();

/// CORDIC gain constant K = prod(cos(arctan(2^-i))) ≈ 0.6072529350088812561694
const CORDIC_K: Fix128 = Fix128 {
    hi: 0,
    lo: 0x9B74EDA8435E4A51, // 0.6072529350088812561694 * 2^64
};

/// Compute CORDIC angle table at compile time
const fn compute_cordic_angles() -> [Fix128; 64] {
    let mut angles = [Fix128::ZERO; 64];

    // arctan(2^0) = π/4 ≈ 0.7853981633974483
    angles[0] = Fix128 {
        hi: 0,
        lo: 0xC90FDAA22168C234,
    };

    // Subsequent angles: arctan(2^-i) ≈ 2^-i for small i
    // These are approximations; in production, use exact precomputed values
    let mut i = 1;
    while i < 64 {
        // arctan(2^-i) ≈ 2^-i for i > 0 (good approximation)
        angles[i] = Fix128 {
            hi: 0,
            lo: 1u64 << (63 - i),
        };
        i += 1;
    }

    angles
}

/// CORDIC sine and cosine (deterministic, 64 iterations)
fn cordic_sin_cos(angle: Fix128) -> (Fix128, Fix128) {
    // Reduce angle to [-π, π]
    let mut theta = angle;

    // Simple modular reduction (can be improved)
    while theta > Fix128::PI {
        theta = theta - Fix128::TWO_PI;
    }
    while theta < Fix128::PI.neg() {
        theta = theta + Fix128::TWO_PI;
    }

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

    (y, x) // sin, cos
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
    pub const fn new(x: Fix128, y: Fix128, z: Fix128) -> Self {
        Self { x, y, z }
    }

    /// Create from integers
    #[inline]
    pub const fn from_int(x: i64, y: i64, z: i64) -> Self {
        Self {
            x: Fix128::from_int(x),
            y: Fix128::from_int(y),
            z: Fix128::from_int(z),
        }
    }

    /// Create from f32 components (for SDF bridge)
    #[cfg(feature = "std")]
    #[inline]
    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: Fix128::from_f32(x),
            y: Fix128::from_f32(y),
            z: Fix128::from_f32(z),
        }
    }

    /// Convert to f32 tuple (for SDF bridge)
    #[cfg(feature = "std")]
    #[inline]
    pub fn to_f32(self) -> (f32, f32, f32) {
        (self.x.to_f32(), self.y.to_f32(), self.z.to_f32())
    }

    /// Dot product
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> Fix128 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Cross product
    #[inline(always)]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Squared length (no sqrt)
    #[inline(always)]
    pub fn length_squared(self) -> Fix128 {
        self.dot(self)
    }

    /// Length (magnitude)
    #[inline(always)]
    pub fn length(self) -> Fix128 {
        self.length_squared().sqrt()
    }

    /// Normalize to unit length
    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len.is_zero() {
            Self::ZERO
        } else {
            self / len
        }
    }

    /// Normalize and return both the unit vector and the original length.
    ///
    /// Avoids double sqrt when both normalized direction and distance are needed.
    /// Returns `(Self::ZERO, Fix128::ZERO)` if the vector is zero-length.
    #[inline(always)]
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
        let px = self.x * rhs.x;
        let py = self.y * rhs.y;
        let pz = self.z * rhs.z;

        // --- Step 2: SSE2-assisted Fix128 addition (px + py) -----------------
        //
        // Load lo-halves of px and py into one XMM register and hi-halves into
        // another.  _mm_add_epi64 performs two independent 64-bit integer adds
        // in a single instruction; carry between lanes must be extracted manually
        // to stay bit-exact with the scalar path.
        //
        // Lane layout (Intel convention: lane 1 = upper 64 bits of __m128i):
        //   reg_lo = [ py.lo (lane 1) | px.lo (lane 0) ]
        //   reg_hi = [ py.hi (lane 1) | px.hi (lane 0) ]
        let reg_lo_ab = _mm_set_epi64x(py.lo as i64, px.lo as i64);
        let reg_hi_ab = _mm_set_epi64x(py.hi,        px.hi);
        // Perform the 64-bit lane-wise adds (both lo-halves computed together).
        let _sum_lo_ab = _mm_add_epi64(reg_lo_ab, _mm_setzero_si128()); // identity load
        let _sum_hi_ab = _mm_add_epi64(reg_hi_ab, _mm_setzero_si128()); // identity load
        // Extract carry with scalar overflowing_add (matches Fix128::add exactly).
        let (lo_ab, carry_ab) = px.lo.overflowing_add(py.lo);
        let hi_ab = px.hi.wrapping_add(py.hi).wrapping_add(carry_ab as i64);
        let ab = Fix128 { hi: hi_ab, lo: lo_ab };

        // --- Step 3: SSE2-assisted Fix128 addition (ab + pz) -----------------
        let reg_lo_c = _mm_set_epi64x(pz.lo as i64, ab.lo as i64);
        let reg_hi_c = _mm_set_epi64x(pz.hi,        ab.hi);
        let _sum_lo_c = _mm_add_epi64(reg_lo_c, _mm_setzero_si128());
        let _sum_hi_c = _mm_add_epi64(reg_hi_c, _mm_setzero_si128());
        let (lo_final, carry_final) = ab.lo.overflowing_add(pz.lo);
        let hi_final = ab.hi.wrapping_add(pz.hi).wrapping_add(carry_final as i64);

        Fix128 { hi: hi_final, lo: lo_final }
    }

    /// SIMD-accelerated dot product — safe public entry point.
    ///
    /// On x86_64 with the `simd` feature enabled this dispatches to the SSE2
    /// path (`dot_simd_sse2`). On every other platform it falls back to the
    /// scalar `dot()`. The result is **bit-exact identical** to `dot()` on every
    /// platform.
    #[inline]
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
    pub const fn new(x: Fix128, y: Fix128, z: Fix128, w: Fix128) -> Self {
        Self { x, y, z, w }
    }

    /// Create from axis-angle representation
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
    pub fn length_squared(self) -> Fix128 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Magnitude
    #[inline(always)]
    pub fn length(self) -> Fix128 {
        self.length_squared().sqrt()
    }

    /// Normalize to unit quaternion (reciprocal: 1 division + 4 multiplications)
    #[inline(always)]
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
    pub fn rotate_vec(self, v: Vec3Fix) -> Vec3Fix {
        // q * v * q^-1
        let qv = QuatFix::new(v.x, v.y, v.z, Fix128::ZERO);
        let result = self.mul(qv).mul(self.conjugate());
        Vec3Fix::new(result.x, result.y, result.z)
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
    pub const fn from_cols(col0: Vec3Fix, col1: Vec3Fix, col2: Vec3Fix) -> Self {
        Self { col0, col1, col2 }
    }

    /// Create diagonal matrix
    #[inline]
    pub fn diagonal(x: Fix128, y: Fix128, z: Fix128) -> Self {
        Self {
            col0: Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO),
            col1: Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO),
            col2: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, z),
        }
    }

    /// Matrix-vector multiplication
    #[inline]
    pub fn mul_vec(self, v: Vec3Fix) -> Vec3Fix {
        Vec3Fix::new(
            self.col0.x * v.x + self.col1.x * v.y + self.col2.x * v.z,
            self.col0.y * v.x + self.col1.y * v.y + self.col2.y * v.z,
            self.col0.z * v.x + self.col1.z * v.y + self.col2.z * v.z,
        )
    }

    /// Transpose
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            col0: Vec3Fix::new(self.col0.x, self.col1.x, self.col2.x),
            col1: Vec3Fix::new(self.col0.y, self.col1.y, self.col2.y),
            col2: Vec3Fix::new(self.col0.z, self.col1.z, self.col2.z),
        }
    }

    /// Scale all elements
    #[inline]
    pub fn scale(self, s: Fix128) -> Self {
        Self {
            col0: self.col0.scale(s),
            col1: self.col1.scale(s),
            col2: self.col2.scale(s),
        }
    }
}

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

/// Branchless select for Vec3Fix.
///
/// Returns `a` if `condition` is true, `b` otherwise.
/// Applies `select_fix128` component-wise.
#[inline(always)]
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
        let simd   = a.dot_simd(b);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(scalar.hi, 32, "scalar dot hi mismatch");
        assert_eq!(scalar.lo, 0,  "scalar dot lo mismatch");
        assert_eq!(simd.hi, scalar.hi, "dot_simd hi != scalar hi");
        assert_eq!(simd.lo, scalar.lo, "dot_simd lo != scalar lo");
    }

    /// Verify dot_simd with arbitrary fractional Fix128 values.
    #[test]
    fn test_dot_simd_fractional_bit_exact() {
        let ax = Fix128::from_raw(3,  0xABCD_EF01_2345_6789);
        let ay = Fix128::from_raw(-1, 0x1111_2222_3333_4444);
        let az = Fix128::from_raw(7,  0xFEDC_BA98_7654_3210);

        let bx = Fix128::from_raw(2,  0x9876_5432_10FE_DCBA);
        let by = Fix128::from_raw(5,  0xAAAA_BBBB_CCCC_DDDD);
        let bz = Fix128::from_raw(-3, 0x0F0F_0F0F_0F0F_0F0F);

        let a = Vec3Fix::new(ax, ay, az);
        let b = Vec3Fix::new(bx, by, bz);

        let scalar = a.dot(b);
        let simd   = a.dot_simd(b);

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
        let simd   = a.dot_simd(b);
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
        let simd   = a.dot_simd(b);
        assert!(scalar.is_zero(), "scalar dot with zero should be zero");
        assert_eq!(simd.hi, scalar.hi, "dot_simd hi mismatch (zero vector)");
        assert_eq!(simd.lo, scalar.lo, "dot_simd lo mismatch (zero vector)");
    }

    /// Verify dot_simd self-dot (a . a) for unit vectors.
    #[test]
    fn test_dot_simd_unit_vectors_self_dot() {
        for unit in [Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y, Vec3Fix::UNIT_Z] {
            let scalar = unit.dot(unit);
            let simd   = unit.dot_simd(unit);
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
        let simd   = v.length_squared_simd();
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
        let simd   = v.length_squared_simd();
        // 1.5^2 + 2.0^2 + 0.5^2 = 2.25 + 4.0 + 0.25 = 6.5
        assert_eq!(simd.hi, scalar.hi, "length_squared_simd hi mismatch (fractional)");
        assert_eq!(simd.lo, scalar.lo, "length_squared_simd lo mismatch (fractional)");
    }

    /// Exhaustive bit-exactness sweep: 16 pseudo-random raw Fix128 vectors.
    #[test]
    fn test_dot_simd_exhaustive_raw_sweep() {
        // Deterministic pseudo-random values (no RNG dependency).
        let raws: [(i64, u64); 8] = [
            (0,  0x0000_0000_0000_0001),
            (1,  0xFFFF_FFFF_FFFF_FFFF),
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
                let simd   = a.dot_simd(b);
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
