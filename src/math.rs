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
//! - **`simd` feature**: exposes the `*_simd` / `dot_batch_4` API surface and
//!   `SIMD_WIDTH`. As of 1.2.0 every one of these is **scalar-equivalent**:
//!   SSE2/AVX2 have no 128-bit multiply and cannot propagate the lo→hi carry a
//!   Fix128 add needs, so the intrinsic paths were never faster than the
//!   ADC chain LLVM already emits. The feature is kept so downstream code can
//!   target a future AVX-512 / NEON batch path without an API change.
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
        let abs_frac = frac.abs();
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
    pub const fn floor(self) -> Self {
        Self { hi: self.hi, lo: 0 }
    }

    /// Ceiling (round toward positive infinity)
    #[inline]
    #[must_use]
    pub const fn ceil(self) -> Self {
        if self.lo == 0 {
            self
        } else {
            Self {
                hi: self.hi + 1,
                lo: 0,
            }
        }
    }

    /// Square root, exact floor: `floor(sqrt(self))` in I64F64.
    ///
    /// Restoring digit-recurrence (one result bit per step) on the 192-bit
    /// radicand `self << 64`, 96 fixed steps, so the result is the largest
    /// `r` with `r * r <= self` — bit-identical on every platform and to the
    /// pre-1.2.0 Newton-Raphson implementation (verified over 20 000 samples
    /// plus edge values in `tests::sqrt_matches_newton_reference`).
    ///
    /// Before 1.2.0 this ran 64 Newton iterations, each calling the 64-step
    /// long division, ≈ 4 096 inner steps per call (≈ 10 µs); 3-5 iterations
    /// had already converged. The recurrence needs 96 shift/compare/subtract
    /// steps (≈ 24× faster, see `benches/physics_bench.rs::fix128_sqrt`).
    ///
    /// Negative input returns `ZERO` (deterministic, no NaN).
    ///
    /// Deterministic: fixed number of iterations, integer-only.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.is_negative() || self.is_zero() {
            return Self::ZERO;
        }

        // Radicand N = (hi:lo) << 64 as a 192-bit integer; result = isqrt(N),
        // which is sqrt(value) * 2^64 = the I64F64 encoding of sqrt(value).
        // Process 2 radicand bits per step from the top: the first 64 steps
        // consume the 128 bits of (hi:lo), the remaining 32 steps consume the
        // 64 appended zero bits.
        let n = ((self.hi as u128) << 64) | (self.lo as u128);
        let mut rem: u128 = 0;
        let mut root: u128 = 0;
        let mut i = 0u32;
        while i < 96 {
            // Next two radicand bits (MSB first). rem < 2*root + 1 < 2^97
            // before the shift, so rem << 2 | bits < 2^99 never overflows u128.
            let bits = if i < 64 { (n >> (126 - 2 * i)) & 3 } else { 0 };
            rem = (rem << 2) | bits;
            let trial = (root << 2) | 1;
            if rem >= trial {
                rem -= trial;
                root = (root << 1) | 1;
            } else {
                root <<= 1;
            }
            i += 1;
        }

        Self {
            hi: (root >> 64) as i64,
            lo: root as u64,
        }
    }

    /// Divide by 2 (bit shift, exact)
    #[inline]
    #[must_use]
    pub const fn half(self) -> Self {
        let hi = self.hi >> 1;
        let lo = (self.lo >> 1) | ((self.hi as u64 & 1) << 63);
        Self { hi, lo }
    }

    /// Arithmetic right shift of the full 128-bit value by `i` bits (exact `self / 2^i`,
    /// rounding toward negative infinity).
    ///
    /// The single source of truth for the CORDIC `x >> i` / `y >> i` micro-rotations
    /// (`cordic_sin_cos` and `cordic_atan`). Before 1.1.1 `cordic_atan` had its own copy
    /// that dropped the bits carried from `hi` into `lo`, so `ONE >> 1` evaluated to `0`
    /// instead of `0.5` and every `atan` / `atan2` result was off by up to ~0.17 rad.
    #[inline]
    #[must_use]
    pub const fn shr_bits(self, i: u32) -> Self {
        if i == 0 {
            return self;
        }
        if i >= 128 {
            let hi = self.hi >> 63;
            return Self { hi, lo: hi as u64 };
        }
        if i >= 64 {
            // hi の下位ビットが lo に降りる、hi は符号のみ残る
            let lo = (self.hi >> (i - 64)) as u64;
            return Self {
                hi: self.hi >> 63,
                lo,
            };
        }
        Self {
            hi: self.hi >> i,
            lo: (self.lo >> i) | ((self.hi as u64) << (64 - i)),
        }
    }

    /// Multiply by 2 (bit shift, exact)
    #[inline]
    #[must_use]
    pub const fn double(self) -> Self {
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

    /// Addition through the `simd` feature's API surface — **scalar-equivalent**.
    ///
    /// Bit-identical to `self + rhs`. SSE2's `_mm_add_epi64` adds the two
    /// 64-bit lanes independently and cannot carry from `lo` into `hi`, so a
    /// Fix128 add has no profitable SSE2 form; LLVM already emits `add`/`adc`
    /// for the scalar path. Before 1.2.0 this function loaded both operands
    /// into `__m128i` registers and then discarded them (`let _ = (a, b)`),
    /// i.e. it was scalar with dead intrinsics.
    ///
    /// # Safety
    ///
    /// No preconditions; kept `unsafe` + `#[target_feature(enable = "sse2")]`
    /// for signature compatibility with existing callers.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn add_simd(self, rhs: Self) -> Self {
        self + rhs
    }

    /// Subtraction through the `simd` feature's API surface — **scalar-equivalent**
    /// (see [`Self::add_simd`]).
    ///
    /// # Safety
    ///
    /// No preconditions; kept `unsafe` + `#[target_feature(enable = "sse2")]`
    /// for signature compatibility with existing callers.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[inline]
    #[target_feature(enable = "sse2")]
    pub unsafe fn sub_simd(self, rhs: Self) -> Self {
        self - rhs
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

/// Fixed-point multiplication: the 256-bit product of the two Q64.64
/// values, keeping bits `[192:64]`.
///
/// # Overflow and rounding
///
/// - **Wrapping**: if the mathematical product does not fit the ±2^63
///   integer range the high bits are discarded — the result wraps modulo
///   2^128 of the raw two's-complement representation. There is no
///   saturation and no panic, in debug builds too (every intermediate uses
///   `wrapping_*`). Callers that need range safety must check operands
///   beforehand; the engine's own hot paths keep magnitudes far below the
///   limit.
/// - **Truncation**: fractional bits below 2^-64 are dropped, which is a
///   floor toward −∞ on the two's-complement bit pattern (e.g.
///   `-2^-64 * 0.5 == -2^-64`, not `0`).
///
/// Both properties are part of the determinism contract: the same
/// operands yield the same bits on every platform.
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

impl Fix128 {
    /// `self^exponent` for `self > 0` and `exponent ≥ 0` (real exponent).
    ///
    /// Integer part of the exponent by repeated multiplication, fractional
    /// part by its binary expansion `base^f = Π base^(1/2^k)` over the first 24
    /// bits using successive [`Fix128::sqrt`] calls — pure integer arithmetic,
    /// bit-identical on every platform, relative error ≲ |ln base| · 2⁻²⁴ plus
    /// sqrt truncation (measured ≤ 1e-6 relative for results ≥ 1e-9 over
    /// `base ∈ [1e-3, 1e3]`, `exponent ∈ [0, 8]`,
    /// `math::tests::powf_pos_matches_f64_reference`; smaller results hit the
    /// 2⁻⁶⁴ absolute resolution).
    /// `self ≤ 0` or a negative exponent returns `ZERO` (no NaN; callers that
    /// need those cases handle them explicitly).
    #[must_use]
    pub fn powf_pos(self, exponent: Self) -> Self {
        if self <= Self::ZERO || exponent.is_negative() {
            return Self::ZERO;
        }
        let n = exponent.hi.min(64) as u32;
        let mut r = Self::ONE;
        for _ in 0..n {
            r = r * self;
        }
        let mut frac_bits = exponent.lo;
        let mut root = self;
        for _ in 0..24 {
            root = root.sqrt(); // base^(1/2), base^(1/4), …
            if frac_bits & (1u64 << 63) != 0 {
                r = r * root;
            }
            frac_bits <<= 1;
        }
        r
    }

    /// Deterministic `e^self` via `2^(x·log₂e)` ([`Fix128::powf_pos`] with base 2;
    /// negative arguments as `1 / e^|x|`). Relative error ≲ 1e-6 for
    /// `|x| ≤ 40` (`math::tests::exp_matches_f64_reference`); `x < −44` is
    /// below the 2⁻⁶⁴ resolution and returns `ZERO`, `x > 43` saturates at
    /// the representable maximum instead of wrapping.
    #[must_use]
    pub fn exp(self) -> Self {
        // log₂(e) = 1.442 695 040 888 963 4 (raw pair, exact to 2⁻⁶⁴)
        const LOG2_E: Fix128 = Fix128 {
            hi: 1,
            lo: 8_166_282_121_979_092_992,
        };
        if self.is_negative() {
            if self.hi < -44 {
                return Self::ZERO;
            }
            let pos = Self::from_int(2).powf_pos(self.neg() * LOG2_E);
            if pos.is_zero() {
                return Self::ZERO;
            }
            return Self::ONE / pos;
        }
        if self.hi >= 43 {
            return Self::from_raw(i64::MAX, u64::MAX);
        }
        Self::from_int(2).powf_pos(self * LOG2_E)
    }

    /// Checked division: `None` when `rhs == 0`, otherwise `Some(self / rhs)`.
    ///
    /// The `Div` operator returns `ZERO` for a zero divisor (deterministic, no
    /// panic, no NaN) which is convenient in solver hot paths but hides bugs
    /// in caller code that never expected a zero — use this in validation and
    /// setup paths. Bit-identical to `/` for every non-zero divisor.
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.is_zero() {
            None
        } else {
            Some(self / rhs)
        }
    }
}

/// Fixed-point division, `floor` toward −∞ on the raw bit pattern like `Mul`.
///
/// # Division by zero
///
/// Returns `ZERO` (documented contract since 1.2.0; the behaviour predates
/// it). This keeps the solver deterministic and panic-free — a zero inverse
/// mass, a degenerate normal or a zero `dt` never poisons a lockstep peer with
/// NaN — but it also means a caller bug is silently masked. Use
/// [`Fix128::checked_div`] where a zero divisor is an error. A `Result`-based
/// operator is planned for 2.0 (`PhysicsError` is not `#[non_exhaustive]`,
/// so a new variant cannot be added in 1.x).
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

        let x_shift = x.shr_bits(i as u32);
        let y_shift = y.shr_bits(i as u32);

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

        let x_shift = x.shr_bits(i as u32);
        let y_shift = y.shr_bits(i as u32);

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

    // Keep the CORDIC argument in [−1, 1]: for |y| > |x| use the complement
    // atan2(y, x) = ±π/2 − atan(x / y). Before 1.2.0 `y / x` was formed
    // unconditionally, and with |x| a few ulp it overflowed the 64-bit
    // integer part (1 / 2⁻⁶³ = 2⁶³), so atan2(1, 2 ulp) returned −0.755
    // instead of π/2 — visible as a wrong-way quaternion slerp between
    // nearly opposite rotations.
    if y.abs() > x.abs() {
        let complement = cordic_atan(x / y);
        return if y.is_negative() {
            Fix128::HALF_PI.neg() - complement
        } else {
            Fix128::HALF_PI - complement
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
    /// Dot product on the `simd` feature's x86_64 path — **scalar-equivalent**.
    ///
    /// Fix128 multiplication is 128-bit integer arithmetic (`u128`/`i128`); no
    /// SSE2/AVX2 instruction performs it, and the two Fix128 additions cannot
    /// use `_mm_add_epi64` because it does not carry from `lo` into `hi` (see
    /// the comment in the body). The result is bit-exact to `dot()` because it
    /// *is* `dot()`. Kept as the dispatch target of `dot_simd` so a future
    /// AVX-512 / NEON batch path can land without an API change.
    ///
    /// # Safety
    ///
    /// No preconditions; `#[target_feature(enable = "sse2")]` is always
    /// satisfied on x86_64.
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

    /// Cross product through the `simd` feature's API surface —
    /// **scalar-equivalent**, bit-identical to `cross()` (see the module docs).
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

    /// Batch dot product for 4 vector pairs — **scalar-equivalent** (four
    /// `dot()` calls, bit-identical). The batch shape is the API a future
    /// AVX-512 / NEON path would fill in.
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
        let qv = Self::new(v.x, v.y, v.z, Fix128::ZERO);
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
    pub const fn diagonal(x: Fix128, y: Fix128, z: Fix128) -> Self {
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
    pub const fn transpose(self) -> Self {
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
pub(crate) const fn select_fix128(condition: bool, a: Fix128, b: Fix128) -> Fix128 {
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
pub(crate) const fn select_vec3(condition: bool, a: Vec3Fix, b: Vec3Fix) -> Vec3Fix {
    Vec3Fix {
        x: select_fix128(condition, a.x, b.x),
        y: select_fix128(condition, a.y, b.y),
        z: select_fix128(condition, a.z, b.z),
    }
}

impl core::ops::Mul<Self> for Mat3Fix {
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
    #[allow(clippy::disallowed_methods)] // f64::powf is the oracle
    fn powf_pos_matches_f64_reference() {
        let mut worst = 0.0f64;
        for bi in 0..40 {
            let b = 1e-3 * 1e6f64.powf(bi as f64 / 39.0); // 1e-3 .. 1e3
            for ei in 0..=32 {
                let e = 8.0 * ei as f64 / 32.0;
                let got = Fix128::from_f64(b).powf_pos(Fix128::from_f64(e)).to_f64();
                let want = b.powf(e);
                if want < 1e-9 || want > 1e15 {
                    continue; // Fix128 has 2⁻⁶⁴ absolute resolution: below 1e-9 the
                              // relative error is dominated by truncation
                }
                let rel = ((got - want) / want).abs();
                worst = worst.max(rel);
                assert!(rel < 1e-6, "{b}^{e} = {got} vs {want} (rel {rel:e})");
            }
        }
        assert!(worst > 0.0, "sweep ran");
        // exact cases
        assert_eq!(
            Fix128::from_int(2).powf_pos(Fix128::from_int(10)),
            Fix128::from_int(1024)
        );
        assert_eq!(
            Fix128::from_int(9).powf_pos(Fix128::from_ratio(1, 2)),
            Fix128::from_int(3)
        );
        assert_eq!(Fix128::from_int(5).powf_pos(Fix128::ZERO), Fix128::ONE);
        assert_eq!(Fix128::ZERO.powf_pos(Fix128::ONE), Fix128::ZERO);
        assert_eq!(Fix128::from_int(-2).powf_pos(Fix128::ONE), Fix128::ZERO);
        assert_eq!(
            Fix128::from_int(2).powf_pos(Fix128::from_int(-1)),
            Fix128::ZERO
        );
    }

    #[test]
    #[allow(clippy::disallowed_methods)] // f64::exp is the oracle
    fn exp_matches_f64_reference() {
        for i in 0..=400 {
            let x = -40.0 + 80.0 * f64::from(i) / 400.0;
            let got = Fix128::from_f64(x).exp().to_f64();
            let want = x.exp();
            if want < 1e-9 {
                continue; // below the 2⁻⁶⁴ resolution floor in relative terms
            }
            let rel = ((got - want) / want).abs();
            assert!(rel < 2e-6, "exp({x}) = {got} vs {want} (rel {rel:e})");
        }
        assert_eq!(Fix128::ZERO.exp(), Fix128::ONE);
        assert!(Fix128::from_int(-50).exp().is_zero());
        assert_eq!(
            Fix128::from_int(50).exp(),
            Fix128::from_raw(i64::MAX, u64::MAX)
        );
        // exp(1) = e to 1e-6
        assert!((Fix128::ONE.exp().to_f64() - core::f64::consts::E).abs() < 3e-6);
    }

    #[test]
    fn div_by_zero_is_zero_and_checked_div_is_none() {
        let x = Fix128::from_int(7);
        assert_eq!(x / Fix128::ZERO, Fix128::ZERO);
        assert_eq!(Fix128::ZERO / Fix128::ZERO, Fix128::ZERO);
        assert_eq!(x.checked_div(Fix128::ZERO), None);
        assert_eq!(
            x.checked_div(Fix128::from_int(2)),
            Some(Fix128::from_ratio(7, 2))
        );
        assert_eq!(
            Fix128::from_ratio(-1, 3).checked_div(Fix128::from_ratio(2, 7)),
            Some(Fix128::from_ratio(-1, 3) / Fix128::from_ratio(2, 7))
        );
    }

    #[test]
    fn fix128_mul_wraps_on_integer_overflow() {
        // 2^62 * 4 = 2^64 → integer part wraps modulo 2^64 → 0.
        let a = Fix128::from_int(1 << 62);
        let b = Fix128::from_int(4);
        assert_eq!(a * b, Fix128::ZERO);
        // 2^62 * 3 = 3·2^62 = 2^63 + 2^62 → wraps to -2^63 + 2^62 = -2^62.
        assert_eq!(a * Fix128::from_int(3), Fix128::from_int(-(1 << 62)));
    }

    #[test]
    fn fix128_mul_truncates_toward_negative_infinity() {
        // -2^-64 is the raw pattern {hi: -1, lo: u64::MAX}.
        let tiny_neg = Fix128::from_raw(-1, u64::MAX);
        let half = Fix128::from_ratio(1, 2);
        // Exact value -2^-65 is not representable; floor gives -2^-64.
        assert_eq!(tiny_neg * half, tiny_neg);
        // Positive counterpart truncates to 0.
        let tiny_pos = Fix128::from_raw(0, 1);
        assert_eq!(tiny_pos * half, Fix128::ZERO);
    }

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
        let a = Fix128::from_f32(core::f32::consts::PI);
        let back = a.to_f32();
        assert!((back - core::f32::consts::PI).abs() < 0.001);

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

    // ---- atan / atan2 (1.1.1: cordic_atan の shift 桁落ち修正) --------

    fn close_f64(v: Fix128, expect: f64, tol: f64) -> bool {
        (v.to_f64() - expect).abs() <= tol
    }

    #[test]
    fn shr_bits_is_exact_arithmetic_shift() {
        let one = Fix128::ONE;
        assert_eq!(one.shr_bits(0), one);
        assert_eq!(one.shr_bits(1), Fix128::from_ratio(1, 2)); // 1.1.1 以前の atan 版は 0 になっていた
        assert_eq!(one.shr_bits(3), Fix128::from_ratio(1, 8));
        assert_eq!(Fix128::from_int(-1).shr_bits(1), Fix128::from_ratio(-1, 2));
        assert_eq!(Fix128::from_int(-6).shr_bits(2), Fix128::from_ratio(-3, 2));
        let v = Fix128 {
            hi: 5,
            lo: 0x8000_0000_0000_0001,
        };
        assert_eq!(
            v.shr_bits(1),
            Fix128 {
                hi: 2,
                lo: 0xC000_0000_0000_0000
            }
        );
        assert_eq!(v.shr_bits(64), Fix128 { hi: 0, lo: 5 });
        assert_eq!(v.shr_bits(65), Fix128 { hi: 0, lo: 2 });
        assert_eq!(
            Fix128::from_int(-1).shr_bits(64),
            Fix128 {
                hi: -1,
                lo: u64::MAX
            }
        );
        assert_eq!(v.shr_bits(200), Fix128::ZERO);
        assert_eq!(
            Fix128::from_int(-1).shr_bits(200),
            Fix128 {
                hi: -1,
                lo: u64::MAX
            }
        );
        // half() と 1 bit shift は同値
        for x in [
            Fix128::PI,
            Fix128::from_ratio(-7, 3),
            Fix128::from_int(1234),
        ] {
            assert_eq!(x.shr_bits(1), x.half());
        }
    }

    #[test]
    // 参照値として platform libm を使う (許容 1e-12、決定論の pin は *_golden_bit_patterns 側)
    #[allow(clippy::disallowed_methods)]
    fn atan_matches_f64_within_1e12() {
        for (v, expect) in [
            (Fix128::ZERO, 0.0),
            (Fix128::ONE, core::f64::consts::FRAC_PI_4),
            (Fix128::from_int(-1), -core::f64::consts::FRAC_PI_4),
            (Fix128::from_ratio(1, 2), 0.5f64.atan()),
            (Fix128::from_int(3), 3f64.atan()),
            (Fix128::from_ratio(-1, 8), (-0.125f64).atan()),
            (Fix128::from_int(1000), 1000f64.atan()),
        ] {
            let got = v.atan();
            assert!(
                close_f64(got, expect, 1e-12),
                "atan({}) = {} vs {expect}",
                v.to_f64(),
                got.to_f64()
            );
        }
    }

    #[test]
    fn atan_golden_bit_patterns_1_1_1() {
        // 修正後の実装値を pin (cross-platform bit-exact の source of truth、意図的変更時のみ更新)
        let cases: [(Fix128, i64, u64); 7] = [
            (Fix128::ZERO, -1, 18446744073709455410),
            (Fix128::ONE, 0, 14488038916154342774),
            (Fix128::from_int(-1), -1, 3958705157555404150),
            (Fix128::from_ratio(1, 2), 0, 8552788783625182168),
            (Fix128::from_int(3), 1, 4594083626069865406),
            (Fix128::from_ratio(-1, 8), -1, 16152799315017801038),
            (Fix128::from_int(1000), 1, 10510887020674212946),
        ];
        for (v, hi, lo) in cases {
            assert_eq!(v.atan(), Fix128 { hi, lo }, "atan({})", v.to_f64());
        }
    }

    #[test]
    // 参照値として platform libm を使う (許容 1e-12、決定論の pin は *_golden_bit_patterns 側)
    #[allow(clippy::disallowed_methods)]
    fn atan2_matches_f64_in_every_quadrant_and_on_axes() {
        let cases: [(i64, i64); 10] = [
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (3, 4),
            (-2, 5),
        ];
        for (y, x) in cases {
            let got = Fix128::atan2(Fix128::from_int(y), Fix128::from_int(x));
            let expect = (y as f64).atan2(x as f64);
            assert!(
                close_f64(got, expect, 1e-12),
                "atan2({y}, {x}) = {} vs {expect}",
                got.to_f64()
            );
        }
        assert_eq!(Fix128::atan2(Fix128::ZERO, Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn atan2_golden_bit_patterns_1_1_1() {
        let cases: [(i64, i64, i64, u64); 11] = [
            (1, 1, 0, 14488038916154342774),
            (1, -1, 2, 6570628601043732041),
            (-1, -1, -3, 11876115472666014883),
            (-1, 1, -1, 3958705157555404150),
            (1, 0, 1, 10529333758598939753),
            (-1, 0, -2, 7917410315110611863),
            (0, 1, -1, 18446744073709455410),
            (0, -1, 3, 2611923443488231685),
            (0, 0, 0, 0),
            (3, 4, 0, 11870500265058138062),
            (-2, 5, -1, 11427640316703283098),
        ];
        for (y, x, hi, lo) in cases {
            assert_eq!(
                Fix128::atan2(Fix128::from_int(y), Fix128::from_int(x)),
                Fix128 { hi, lo },
                "atan2({y}, {x})"
            );
        }
    }

    #[test]
    // 参照値として platform libm を使う (許容 1e-12、決定論の pin は *_golden_bit_patterns 側)
    #[allow(clippy::disallowed_methods)]
    fn atan_identities_hold() {
        // odd: atan(-x) == -atan(x) (CORDIC は符号対称なので bit-exact ではなく 1e-12)
        for v in [
            Fix128::from_ratio(1, 3),
            Fix128::from_int(2),
            Fix128::from_ratio(17, 5),
        ] {
            let a = v.atan();
            let b = (-v).atan();
            assert!(close_f64(a + b, 0.0, 1e-12), "odd symmetry {}", v.to_f64());
            // atan(x) + atan(1/x) == π/2 (x > 0)
            let c = (Fix128::ONE / v).atan();
            assert!(
                close_f64(a + c, core::f64::consts::FRAC_PI_2, 1e-12),
                "complement {}",
                v.to_f64()
            );
            // atan2(y, x) == atan(y/x) for x > 0
            let d = Fix128::atan2(v, Fix128::from_int(2));
            assert!(close_f64(
                d,
                (v / Fix128::from_int(2)).atan().to_f64(),
                1e-12
            ));
        }
        // sin / cos は本修正で bit 単位で不変 (代表点 pin)
        let (s, c) = Fix128::HALF_PI.sin_cos();
        assert_eq!(s, Fix128 { hi: 1, lo: 1453 });
        assert_eq!(
            c,
            Fix128 {
                hi: -1,
                lo: 18446744073709456430
            }
        );
    }

    // ---- Mat3Fix (mutation-score tests, 2026-09-15) ---------------------

    fn fi(n: i64) -> Fix128 {
        Fix128::from_int(n)
    }

    fn v3i(x: i64, y: i64, z: i64) -> Vec3Fix {
        Vec3Fix::from_int(x, y, z)
    }

    /// 列ベクトル指定の整数行列 (M = [c0 c1 c2])
    fn mat(c0: (i64, i64, i64), c1: (i64, i64, i64), c2: (i64, i64, i64)) -> Mat3Fix {
        Mat3Fix::from_cols(
            v3i(c0.0, c0.1, c0.2),
            v3i(c1.0, c1.1, c1.2),
            v3i(c2.0, c2.1, c2.2),
        )
    }

    #[test]
    fn mat3_mul_vec_uses_every_entry_with_the_right_index() {
        // 9 entry 全て異なる素数 → どの entry の取り違えでも値が変わる
        let m = mat((2, 3, 5), (7, 11, 13), (17, 19, 23));
        let v = v3i(1, 10, 100);
        // row i = c0[i]*1 + c1[i]*10 + c2[i]*100
        assert_eq!(
            m.mul_vec(v),
            v3i(2 + 70 + 1700, 3 + 110 + 1900, 5 + 130 + 2300)
        );
        assert_eq!(Mat3Fix::IDENTITY.mul_vec(v), v);
        assert_eq!(m.mul_vec(Vec3Fix::UNIT_X), v3i(2, 3, 5));
        assert_eq!(m.mul_vec(Vec3Fix::UNIT_Y), v3i(7, 11, 13));
        assert_eq!(m.mul_vec(Vec3Fix::UNIT_Z), v3i(17, 19, 23));
    }

    #[test]
    fn mat3_determinant_closed_form() {
        // det [[2,7,17],[3,11,19],[5,13,23]] (列指定) = -78
        let m = mat((2, 3, 5), (7, 11, 13), (17, 19, 23));
        assert_eq!(m.determinant(), fi(-78));
        assert_eq!(Mat3Fix::IDENTITY.determinant(), Fix128::ONE);
        assert_eq!(
            Mat3Fix::diagonal(fi(2), fi(3), fi(-4)).determinant(),
            fi(-24)
        );
        // 列が線形従属 → 0
        assert_eq!(
            mat((1, 2, 3), (2, 4, 6), (0, 1, 0)).determinant(),
            Fix128::ZERO
        );
        // 転置不変
        assert_eq!(m.transpose().determinant(), fi(-78));
        // 各 entry を単独で変えると det が変わる (どの項も効いている)
        for r in 0..3 {
            for c in 0..3 {
                let mut m2 = m;
                let col = match c {
                    0 => &mut m2.col0,
                    1 => &mut m2.col1,
                    _ => &mut m2.col2,
                };
                match r {
                    0 => col.x = col.x + Fix128::ONE,
                    1 => col.y = col.y + Fix128::ONE,
                    _ => col.z = col.z + Fix128::ONE,
                }
                assert_ne!(m2.determinant(), fi(-78), "entry ({r},{c}) は det に効く");
            }
        }
    }

    #[test]
    fn mat3_inverse_exact_for_unimodular_matrix() {
        // det = 1 の整数行列 → 逆行列も整数で exact
        // M = [[1,2,3],[0,1,4],[5,6,0]] (行表記) → det 1、M⁻¹ = [[-24,18,5],[20,-15,-4],[-5,4,1]]
        let m = mat((1, 0, 5), (2, 1, 6), (3, 4, 0));
        assert_eq!(m.determinant(), Fix128::ONE);
        let inv = m.inverse().expect("det 1");
        assert_eq!(inv, mat((-24, 20, -5), (18, -15, 4), (5, -4, 1)));
        assert_eq!(m.mul_mat(inv), Mat3Fix::IDENTITY);
        assert_eq!(inv.mul_mat(m), Mat3Fix::IDENTITY);
        assert_eq!(m * inv, Mat3Fix::IDENTITY);
        // 各 cofactor が正しい位置にある: inv の全 9 entry が異なる値
        let e = [
            inv.col0.x, inv.col0.y, inv.col0.z, inv.col1.x, inv.col1.y, inv.col1.z, inv.col2.x,
            inv.col2.y, inv.col2.z,
        ];
        for i in 0..9 {
            for j in (i + 1)..9 {
                assert_ne!(e[i], e[j], "entries {i} and {j}");
            }
        }
    }

    #[test]
    fn mat3_inverse_scales_by_reciprocal_determinant() {
        // det = -8 (2 の冪) → 1/det exact
        let m = Mat3Fix::diagonal(fi(2), fi(-2), fi(2));
        let inv = m.inverse().expect("det -8");
        assert_eq!(
            inv,
            Mat3Fix::diagonal(
                Fix128::from_ratio(1, 2),
                Fix128::from_ratio(-1, 2),
                Fix128::from_ratio(1, 2)
            )
        );
        // 一般行列 det = -78 の逆行列 × 元 = I (数 ulp 許容)
        let g = mat((2, 3, 5), (7, 11, 13), (17, 19, 23));
        let gi = g.inverse().expect("det -78");
        let p = g.mul_mat(gi);
        let close = |a: Fix128, b: Fix128| (a - b).abs() < Fix128 { hi: 0, lo: 1 << 20 };
        for (col, unit) in [
            (p.col0, Vec3Fix::UNIT_X),
            (p.col1, Vec3Fix::UNIT_Y),
            (p.col2, Vec3Fix::UNIT_Z),
        ] {
            assert!(
                close(col.x, unit.x) && close(col.y, unit.y) && close(col.z, unit.z),
                "{col:?} vs {unit:?}"
            );
        }
        assert!(Mat3Fix::ZERO.inverse().is_none());
        assert!(mat((1, 2, 3), (2, 4, 6), (0, 1, 0)).inverse().is_none());
        assert_eq!(Mat3Fix::IDENTITY.inverse(), Some(Mat3Fix::IDENTITY));
    }

    #[test]
    fn mat3_scale_mul_mat_and_operator_agree() {
        let m = mat((2, 3, 5), (7, 11, 13), (17, 19, 23));
        assert_eq!(m.scale(fi(3)), mat((6, 9, 15), (21, 33, 39), (51, 57, 69)));
        assert_eq!(m.scale(Fix128::ZERO), Mat3Fix::ZERO);
        let n = mat((1, 0, 0), (0, 2, 0), (1, 1, 1));
        // (M N) col_j = M · (N col_j)
        let mn = m.mul_mat(n);
        assert_eq!(mn.col0, m.mul_vec(v3i(1, 0, 0)));
        assert_eq!(mn.col1, m.mul_vec(v3i(0, 2, 0)));
        assert_eq!(mn.col2, m.mul_vec(v3i(1, 1, 1)));
        assert_eq!(m * n, mn);
        assert_ne!(n.mul_mat(m), mn, "非可換");
        assert_eq!(m.mul_mat(Mat3Fix::IDENTITY), m);
        assert_eq!(Mat3Fix::IDENTITY * m, m);
        assert_eq!(m.transpose().transpose(), m);
        assert_eq!(m.transpose().col0, v3i(2, 7, 17));
    }

    // ---- QuatFix -------------------------------------------------------

    fn q(x: i64, y: i64, z: i64, w: i64) -> QuatFix {
        QuatFix::new(fi(x), fi(y), fi(z), fi(w))
    }

    #[test]
    fn quat_mul_is_the_hamilton_product() {
        // 基底: i*j = k, j*k = i, k*i = j, i*i = -1
        let i = q(1, 0, 0, 0);
        let j = q(0, 1, 0, 0);
        let k = q(0, 0, 1, 0);
        let one = q(0, 0, 0, 1);
        assert_eq!(i.mul(j), k);
        assert_eq!(j.mul(k), i);
        assert_eq!(k.mul(i), j);
        assert_eq!(j.mul(i), q(0, 0, -1, 0));
        assert_eq!(i.mul(i), q(0, 0, 0, -1));
        assert_eq!(one.mul(i), i);
        assert_eq!(i.mul(one), i);
        // 一般: (1,2,3,4) * (5,6,7,8) = (24, 48, 48, -6)
        let a = q(1, 2, 3, 4);
        let b = q(5, 6, 7, 8);
        assert_eq!(a.mul(b), q(24, 48, 48, -6));
        assert_eq!(b.mul(a), q(32, 32, 56, -6));
        assert_ne!(a.mul(b), b.mul(a));
    }

    #[test]
    fn quat_conjugate_length_and_normalize() {
        let a = q(1, -2, 3, -4);
        assert_eq!(a.conjugate(), q(-1, 2, -3, -4));
        assert_eq!(a.conjugate().conjugate(), a);
        assert_eq!(a.length_squared(), fi(30));
        assert_eq!(q(0, 0, 0, 2).length_squared(), fi(4));
        assert_eq!(q(3, 0, 0, 0).length_squared(), fi(9));
        // a * conj(a) = |a|² (実部のみ)
        assert_eq!(a.mul(a.conjugate()), q(0, 0, 0, 30));
        // normalize: (0,0,0,2) → (0,0,0,1)、(0,3,0,4) → (0,0.6,0,0.8) は 5 で割る
        assert_eq!(q(0, 0, 0, 2).normalize(), QuatFix::IDENTITY);
        assert_eq!(q(0, 0, -4, 0).normalize(), q(0, 0, -1, 0));
        let n = q(0, 3, 0, 4).normalize();
        assert_eq!(
            n,
            QuatFix::new(Fix128::ZERO, fi(3) / fi(5), Fix128::ZERO, fi(4) / fi(5))
        );
        assert_eq!(q(0, 0, 0, 0).normalize(), QuatFix::IDENTITY);
    }

    #[test]
    fn quat_from_axis_angle_components() {
        // 軸は正規化され、(x,y,z) = axis·sin(θ/2)、w = cos(θ/2)
        let theta = Fix128::from_ratio(1, 2);
        let (s, c) = theta.half().sin_cos();
        let qq = QuatFix::from_axis_angle(v3i(0, 0, 5), theta);
        assert_eq!(qq, QuatFix::new(Fix128::ZERO, Fix128::ZERO, s, c));
        let qx = QuatFix::from_axis_angle(v3i(-2, 0, 0), theta);
        assert_eq!(qx, QuatFix::new(-s, Fix128::ZERO, Fix128::ZERO, c));
        // 角度 0 → identity 近傍 (CORDIC gain 込み)
        let q0 = QuatFix::from_axis_angle(v3i(1, 0, 0), Fix128::ZERO);
        assert!((q0.w - Fix128::ONE).abs() < Fix128 { hi: 0, lo: 1 << 20 });
        assert!(q0.x.abs() < Fix128 { hi: 0, lo: 1 << 20 });
    }

    #[test]
    fn quat_rotate_vec_by_quarter_turns() {
        // z 軸 90° 回転: x → y、y → -x (数 ulp 許容)
        let rz = QuatFix::from_axis_angle(v3i(0, 0, 1), Fix128::HALF_PI);
        let close = |a: Vec3Fix, b: Vec3Fix| (a - b).length() < Fix128 { hi: 0, lo: 1 << 24 };
        assert!(close(rz.rotate_vec(Vec3Fix::UNIT_X), Vec3Fix::UNIT_Y));
        assert!(close(rz.rotate_vec(Vec3Fix::UNIT_Y), -Vec3Fix::UNIT_X));
        assert!(close(rz.rotate_vec(Vec3Fix::UNIT_Z), Vec3Fix::UNIT_Z));
        // 180°: (0,0,1,0) は exact
        let half = q(0, 0, 1, 0);
        assert_eq!(half.rotate_vec(v3i(1, 2, 3)), v3i(-1, -2, 3));
        assert_eq!(QuatFix::IDENTITY.rotate_vec(v3i(1, 2, 3)), v3i(1, 2, 3));
    }

    // ---- Vec3Fix helpers / select / From / Display ----------------------

    #[test]
    fn vec3_cross_try_normalize_and_batch_dot() {
        assert_eq!(v3i(1, 0, 0).cross(v3i(0, 1, 0)), v3i(0, 0, 1));
        assert_eq!(v3i(0, 1, 0).cross(v3i(1, 0, 0)), v3i(0, 0, -1));
        assert_eq!(v3i(1, 2, 3).cross(v3i(4, 5, 6)), v3i(-3, 6, -3));
        assert_eq!(v3i(1, 2, 3).cross(v3i(1, 2, 3)), Vec3Fix::ZERO);
        assert_eq!(v3i(0, 0, 4).try_normalize(), Some(v3i(0, 0, 1)));
        assert_eq!(v3i(0, -2, 0).try_normalize(), Some(v3i(0, -1, 0)));
        assert_eq!(Vec3Fix::ZERO.try_normalize(), None);
        assert_eq!(v3i(1, 2, 3).scale(fi(-2)), v3i(-2, -4, -6));
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn vec3_dot_batch_4_matches_scalar_dot() {
        let a = [v3i(1, 0, 0), v3i(1, 1, 1), v3i(2, 3, 4), v3i(-1, 0, 5)];
        let b = [v3i(7, 8, 9), v3i(1, 2, 3), v3i(5, 6, 7), v3i(4, 4, 4)];
        assert_eq!(Vec3Fix::dot_batch_4(a, b), [fi(7), fi(6), fi(56), fi(16)]);
    }

    #[test]
    fn select_fix128_is_a_bit_exact_branchless_select() {
        // hi / lo ともに全 bit パターンが異なる 2 値: & / | / ^ の取り違えで必ず壊れる
        let a = Fix128 {
            hi: 0x5A5A_5A5A_5A5A_5A5A,
            lo: 0xA5A5_A5A5_A5A5_A5A5,
        };
        let b = Fix128 {
            hi: -0x0F0F_0F0F_0F0F_0F10,
            lo: 0x0F0F_0F0F_0F0F_0F0F,
        };
        assert_eq!(select_fix128(true, a, b), a);
        assert_eq!(select_fix128(false, a, b), b);
        assert_eq!(select_fix128(true, b, a), b);
        assert_eq!(select_fix128(false, b, a), a);
        assert_eq!(select_fix128(true, Fix128::ZERO, a), Fix128::ZERO);
        assert_eq!(select_fix128(false, a, Fix128::ZERO), Fix128::ZERO);
    }

    #[test]
    fn from_impls_roundtrip_and_display_format() {
        assert_eq!(Fix128::from(-7i64), fi(-7));
        assert_eq!(Fix128::from(9i32), fi(9));
        let arr3: [Fix128; 3] = v3i(1, -2, 3).into();
        assert_eq!(arr3, [fi(1), fi(-2), fi(3)]);
        assert_eq!(Vec3Fix::from([fi(4), fi(5), fi(-6)]), v3i(4, 5, -6));
        let arr4: [Fix128; 4] = q(1, 2, 3, 4).into();
        assert_eq!(arr4, [fi(1), fi(2), fi(3), fi(4)]);
        assert_eq!(QuatFix::from([fi(-1), fi(0), fi(2), fi(3)]), q(-1, 0, 2, 3));
        assert_eq!(format!("{}", Fix128::from_ratio(-5, 4)), "-1.2500");
        assert_eq!(format!("{}", v3i(1, -2, 3)), "(1.0000, -2.0000, 3.0000)");
        assert_eq!(
            format!("{}", q(0, 0, 0, 1)),
            "(0.0000, 0.0000, 0.0000, 1.0000)"
        );
    }

    // ---- sin / cos (CORDIC range reduction + golden pins) ---------------

    #[test]
    fn sin_cos_golden_bit_patterns() {
        // 17 角度の (angle, sin, cos) bit pattern pin (1.1.1 の shr_bits 集約前後で不変を確認済)
        // 範囲: [-π/2, π/2] 内 / 第 2・3 象限 / ±π ちょうど / 複数周回 (3π, -7π/2) / 大きな値 (100, -1000.25)
        let table: [(&str, Fix128, Fix128, Fix128); 17] = [
            (
                "0",
                Fix128 { hi: 0, lo: 0 },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456430,
                },
                Fix128 { hi: 1, lo: 1453 },
            ),
            (
                "pi/6",
                Fix128 {
                    hi: 0,
                    lo: 9658692610769497123,
                },
                Fix128 {
                    hi: 0,
                    lo: 9223372036854815772,
                },
                Fix128 {
                    hi: 0,
                    lo: 15975348984942493703,
                },
            ),
            (
                "pi/4",
                Fix128 {
                    hi: 0,
                    lo: 14488038916154245684,
                },
                Fix128 {
                    hi: 0,
                    lo: 13043817825332851804,
                },
                Fix128 {
                    hi: 0,
                    lo: 13043817825332714670,
                },
            ),
            (
                "pi/3",
                Fix128 {
                    hi: 1,
                    lo: 870641147829442630,
                },
                Fix128 {
                    hi: 0,
                    lo: 15975348984942493703,
                },
                Fix128 {
                    hi: 0,
                    lo: 9223372036854815772,
                },
            ),
            (
                "pi/2",
                Fix128 {
                    hi: 1,
                    lo: 10529333758598939753,
                },
                Fix128 { hi: 1, lo: 1453 },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456430,
                },
            ),
            (
                "2pi/3",
                Fix128 {
                    hi: 2,
                    lo: 1741282295658885258,
                },
                Fix128 {
                    hi: 0,
                    lo: 15975348984942493703,
                },
                Fix128 {
                    hi: -1,
                    lo: 9223372036854735844,
                },
            ),
            (
                "pi",
                Fix128 {
                    hi: 3,
                    lo: 2611923443488327891,
                },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456430,
                },
                Fix128 {
                    hi: -2,
                    lo: 18446744073709550163,
                },
            ),
            (
                "-pi/6",
                Fix128 {
                    hi: -1,
                    lo: 8788051462940054493,
                },
                Fix128 {
                    hi: -1,
                    lo: 9223372036854735845,
                },
                Fix128 {
                    hi: 0,
                    lo: 15975348984942493709,
                },
            ),
            (
                "-pi/2",
                Fix128 {
                    hi: -2,
                    lo: 7917410315110611863,
                },
                Fix128 {
                    hi: -2,
                    lo: 18446744073709550164,
                },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456423,
                },
            ),
            (
                "-3pi/4",
                Fix128 {
                    hi: -3,
                    lo: 11876115472665917794,
                },
                Fix128 {
                    hi: -1,
                    lo: 5402926248376699811,
                },
                Fix128 {
                    hi: -1,
                    lo: 5402926248376836953,
                },
            ),
            (
                "-pi",
                Fix128 {
                    hi: -4,
                    lo: 15834820630221223725,
                },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456430,
                },
                Fix128 {
                    hi: -2,
                    lo: 18446744073709550163,
                },
            ),
            (
                "5pi/4",
                Fix128 {
                    hi: 3,
                    lo: 17099962359642573575,
                },
                Fix128 {
                    hi: -1,
                    lo: 5402926248376699811,
                },
                Fix128 {
                    hi: -1,
                    lo: 5402926248376836953,
                },
            ),
            (
                "3pi",
                Fix128 {
                    hi: 9,
                    lo: 7835770330464983673,
                },
                Fix128 {
                    hi: -1,
                    lo: 18446744073709456430,
                },
                Fix128 {
                    hi: -2,
                    lo: 18446744073709550163,
                },
            ),
            (
                "-7pi/2",
                Fix128 {
                    hi: -11,
                    lo: 81639984645628190,
                },
                Fix128 { hi: 1, lo: 1453 },
                Fix128 { hi: 0, lo: 95186 },
            ),
            (
                "1/2",
                Fix128 {
                    hi: 0,
                    lo: 9223372036854775808,
                },
                Fix128 {
                    hi: 0,
                    lo: 8843840213032159770,
                },
                Fix128 {
                    hi: 0,
                    lo: 16188540922742043083,
                },
            ),
            (
                "100",
                Fix128 { hi: 100, lo: 0 },
                Fix128 {
                    hi: -1,
                    lo: 9105946684437943138,
                },
                Fix128 {
                    hi: 0,
                    lo: 15906975547020722691,
                },
            ),
            (
                "-1000.25",
                Fix128 {
                    hi: -1001,
                    lo: 13835058055282163712,
                },
                Fix128 {
                    hi: -1,
                    lo: 1101110721943816547,
                },
                Fix128 {
                    hi: 0,
                    lo: 6277847604637333680,
                },
            ),
        ];
        for (name, angle, sin, cos) in table {
            let (s, c) = angle.sin_cos();
            assert_eq!(s, sin, "sin({name})");
            assert_eq!(c, cos, "cos({name})");
            assert_eq!(angle.sin(), sin, "sin() alias {name}");
            assert_eq!(angle.cos(), cos, "cos() alias {name}");
        }
    }

    #[test]
    // 参照値として platform libm を使う (許容 1e-11、決定論の pin は sin_cos_golden_bit_patterns 側)
    #[allow(clippy::disallowed_methods)]
    fn sin_cos_large_and_negative_angles_reduce_correctly() {
        let two_pi = core::f64::consts::PI * 2.0;
        let cases: [(Fix128, f64); 9] = [
            (Fix128::PI * fi(3), 3.0 * core::f64::consts::PI),
            (
                (Fix128::PI * Fix128::from_ratio(7, 2)).neg(),
                -3.5 * core::f64::consts::PI,
            ),
            (fi(100), 100.0),
            (Fix128::from_ratio(-4001, 4), -1000.25),
            (Fix128::TWO_PI + Fix128::from_ratio(1, 2), two_pi + 0.5),
            (
                (Fix128::TWO_PI + Fix128::from_ratio(1, 2)).neg(),
                -(two_pi + 0.5),
            ),
            (
                Fix128::PI * fi(1000) + fi(1),
                1000.0 * core::f64::consts::PI + 1.0,
            ),
            (
                Fix128::PI + Fix128::from_ratio(1, 1000),
                core::f64::consts::PI + 0.001,
            ),
            (
                (Fix128::PI + Fix128::from_ratio(1, 1000)).neg(),
                -(core::f64::consts::PI + 0.001),
            ),
        ];
        for (angle, f) in cases {
            let (s, c) = angle.sin_cos();
            assert!(
                (s.to_f64() - f.sin()).abs() < 1e-11,
                "sin({f}) = {} vs {}",
                s.to_f64(),
                f.sin()
            );
            assert!(
                (c.to_f64() - f.cos()).abs() < 1e-11,
                "cos({f}) = {} vs {}",
                c.to_f64(),
                f.cos()
            );
        }
    }

    #[test]
    fn sin_cos_boundaries_and_symmetries() {
        let eps = Fix128 { hi: 0, lo: 1 << 24 }; // 2^-40
        let close = |a: Fix128, b: Fix128| (a - b).abs() < eps;
        // ±π ちょうどは reduction を通らない (`>` / `<` は false)、π+ε は通る: どちらも sin ≈ 0、cos ≈ -1
        for a in [
            Fix128::PI,
            Fix128::PI.neg(),
            Fix128::PI + Fix128 { hi: 0, lo: 1 },
            Fix128::PI.neg() - Fix128 { hi: 0, lo: 1 },
        ] {
            let (s, c) = a.sin_cos();
            assert!(
                close(s, Fix128::ZERO) && close(c, fi(-1)),
                "{a:?}: {s:?} {c:?}"
            );
        }
        // ±π/2 ちょうど (quadrant 分岐は `>` false) と π/2+ε (分岐 true): sin ≈ ±1、cos ≈ 0
        for (a, sign) in [
            (Fix128::HALF_PI, 1),
            (Fix128::HALF_PI.neg(), -1),
            (Fix128::HALF_PI + Fix128 { hi: 0, lo: 1 }, 1),
        ] {
            let (s, c) = a.sin_cos();
            assert!(
                close(s, fi(sign)) && close(c, Fix128::ZERO),
                "{a:?}: {s:?} {c:?}"
            );
        }
        // 第 2 / 第 3 象限の符号: sin(2π/3) > 0, cos < 0 / sin(-2π/3) < 0, cos < 0
        let (s2, c2) = (Fix128::PI * Fix128::from_ratio(2, 3)).sin_cos();
        assert!(s2 > Fix128::ZERO && c2 < Fix128::ZERO);
        let (s3, c3) = (Fix128::PI * Fix128::from_ratio(-2, 3)).sin_cos();
        assert!(s3 < Fix128::ZERO && c3 < Fix128::ZERO);
        // 奇関数 / 偶関数、周期性 (2π 加算で reduction 経路が変わっても一致)
        for x in [
            Fix128::from_ratio(1, 3),
            Fix128::from_ratio(-7, 5),
            fi(2),
            Fix128::from_ratio(11, 4),
        ] {
            let (s, c) = x.sin_cos();
            let (sn, cn) = x.neg().sin_cos();
            assert!(
                close(s + sn, Fix128::ZERO) && close(c, cn),
                "odd/even {x:?}"
            );
            let (sp, cp) = (x + Fix128::TWO_PI).sin_cos();
            let (sm, cm) = (x - Fix128::TWO_PI * fi(3)).sin_cos();
            assert!(
                close(s, sp) && close(c, cp) && close(s, sm) && close(c, cm),
                "period {x:?}"
            );
            // sin² + cos² = 1
            assert!(close(s * s + c * c, Fix128::ONE), "pythagoras {x:?}");
        }
    }

    // ---- sqrt: digit recurrence vs Newton reference vs exact floor oracle ----

    /// The pre-1.2.0 implementation, kept as a bit-for-bit reference.
    fn sqrt_newton_reference(v: Fix128) -> Fix128 {
        if v.is_negative() || v.is_zero() {
            return Fix128::ZERO;
        }
        let sig_bits = if v.hi > 0 {
            128 - (v.hi as u64).leading_zeros() as i64
        } else if v.hi == 0 && v.lo > 0 {
            64 - v.lo.leading_zeros() as i64
        } else {
            1
        };
        let result_bit = ((sig_bits + 63) / 2) as u32;
        let mut x = if result_bit >= 64 {
            Fix128 {
                hi: 1i64 << (result_bit - 64).min(62),
                lo: 0,
            }
        } else {
            Fix128 {
                hi: 0,
                lo: 1u64 << result_bit,
            }
        };
        for _ in 0..64 {
            let div = v / x;
            x = (x + div).half();
        }
        x
    }

    /// 256-bit square of a u128 as (hi, lo) limbs.
    fn square_wide(r: u128) -> (u128, u128) {
        let (a, b) = (r >> 64, r & 0xFFFF_FFFF_FFFF_FFFF);
        let bb = b * b;
        let ab = a * b; // < 2^128
        let aa = a * a;
        // r^2 = aa<<128 + 2ab<<64 + bb
        let (mid, c1) = bb.overflowing_add(ab << 65);
        let hi = aa + (ab >> 63) + c1 as u128;
        (hi, mid)
    }

    /// Exact-floor property of I64F64 sqrt: with N = raw << 64 (192-bit),
    /// root^2 <= N < (root+1)^2.
    fn assert_sqrt_is_exact_floor(v: Fix128) {
        let r = v.sqrt();
        let root = ((r.hi as u128) << 64) | r.lo as u128;
        let n = ((v.hi as u128) << 64) | v.lo as u128; // N = (n_hi:n_lo) = n << 64 → (n>>64, n<<64)
        let n_hi = n >> 64;
        let n_lo = n << 64;
        let (lo_hi, lo_lo) = square_wide(root);
        assert!(
            (lo_hi, lo_lo) <= (n_hi, n_lo),
            "root^2 > N for {v:?} (root {root:#x})"
        );
        let (up_hi, up_lo) = square_wide(root + 1);
        assert!(
            (up_hi, up_lo) > (n_hi, n_lo),
            "(root+1)^2 <= N for {v:?} (root {root:#x})"
        );
    }

    fn lcg_samples(n: usize) -> impl Iterator<Item = Fix128> {
        // Deterministic LCG (Knuth MMIX), mixes magnitudes from 2^-64 to 2^62.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n).map(move |i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let lo = state;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Vary the integer-part magnitude: 0 .. 2^62 by shifting.
            let shift = (i % 64) as u32;
            let hi = ((state >> 1) >> shift) as i64;
            Fix128 { hi, lo }
        })
    }

    #[test]
    fn sqrt_matches_newton_reference() {
        let edges = [
            Fix128::ZERO,
            Fix128::from_raw(0, 1),
            Fix128::from_raw(0, u64::MAX),
            Fix128::ONE,
            Fix128::from_ratio(1, 4),
            Fix128::from_int(2),
            Fix128::from_int(4),
            Fix128::from_int(1_000_000),
            Fix128::from_raw(i64::MAX, u64::MAX),
            Fix128::from_int(-3),
        ];
        for v in edges.iter().copied().chain(lcg_samples(20_000)) {
            assert_eq!(v.sqrt(), sqrt_newton_reference(v), "sqrt drift for {v:?}");
        }
    }

    #[test]
    fn sqrt_is_exact_floor_and_monotonic() {
        assert_eq!(Fix128::from_ratio(1, 4).sqrt(), Fix128::from_ratio(1, 2));
        assert_eq!(Fix128::from_int(4).sqrt(), Fix128::from_int(2));
        assert_eq!(Fix128::from_int(-3).sqrt(), Fix128::ZERO);
        let mut prev = Fix128::ZERO;
        let mut samples: Vec<Fix128> = lcg_samples(20_000).collect();
        samples.push(Fix128::from_raw(i64::MAX, u64::MAX));
        samples.push(Fix128::from_raw(0, 1));
        samples.sort();
        for v in samples {
            assert_sqrt_is_exact_floor(v);
            let r = v.sqrt();
            assert!(r >= prev, "sqrt not monotonic at {v:?}");
            prev = r;
        }
    }

    // ---- transcendental oracle sweep (f64 libm reference) ----
    //
    // Golden bit patterns only detect *change*; a value that was wrong from the
    // start gets pinned as-is (that is how the 1.0.0 atan bug shipped). Every
    // Fix128 transcendental is therefore also checked against an independent
    // f64 reference over a dense sweep of the domain.

    #[test]
    #[allow(clippy::disallowed_methods)] // the f64 libm values are the oracle on purpose
    fn transcendental_sweep_matches_f64_reference() {
        // CORDIC with 48 iterations: worst-case error ≈ 2^-46 ≈ 1.4e-14 plus
        // range-reduction rounding; f64 reference itself carries 1e-16.
        const TOL: f64 = 1e-11;
        let steps = 4_000;
        for i in 0..=steps {
            // x sweeps [-4π, 4π] (range reduction on both sides) for sin / cos
            let t = -4.0 * core::f64::consts::PI
                + 8.0 * core::f64::consts::PI * (i as f64) / (steps as f64);
            let x = Fix128::from_f64(t);
            let xf = x.to_f64();
            let (s, c) = x.sin_cos();
            assert!(
                (s.to_f64() - xf.sin()).abs() < TOL,
                "sin({xf}) = {} vs {}",
                s.to_f64(),
                xf.sin()
            );
            assert!(
                (c.to_f64() - xf.cos()).abs() < TOL,
                "cos({xf}) = {} vs {}",
                c.to_f64(),
                xf.cos()
            );
            // atan over [-40, 40] (both sides of |x| = 1 argument reduction)
            let a = Fix128::from_f64(t * 10.0 / core::f64::consts::PI);
            let af = a.to_f64();
            assert!(
                (a.atan().to_f64() - af.atan()).abs() < TOL,
                "atan({af}) = {} vs {}",
                a.atan().to_f64(),
                af.atan()
            );
            // sqrt over (0, 1e6]
            let q = Fix128::from_f64(1e6 * (i as f64 + 1.0) / (steps as f64 + 1.0));
            let qf = q.to_f64();
            assert!(
                (q.sqrt().to_f64() - qf.sqrt()).abs() < 1e-9 * qf.sqrt().max(1.0),
                "sqrt({qf}) = {} vs {}",
                q.sqrt().to_f64(),
                qf.sqrt()
            );
        }
        // atan2: all four quadrants + axes, radius varied
        for iy in -20i64..=20 {
            for ix in -20i64..=20 {
                if ix == 0 && iy == 0 {
                    continue;
                }
                let y = Fix128::from_ratio(iy * 7, 13);
                let x = Fix128::from_ratio(ix * 5, 11);
                let (yf, xf) = (y.to_f64(), x.to_f64());
                let got = Fix128::atan2(y, x).to_f64();
                let want = yf.atan2(xf);
                // atan2 returns in (-π, π]; the reference may return -π on the
                // negative x axis with -0.0 — normalise both to [0, 2π).
                let norm =
                    |v: f64| (v + 2.0 * core::f64::consts::PI) % (2.0 * core::f64::consts::PI);
                assert!(
                    (norm(got) - norm(want)).abs() < TOL
                        || (norm(got) - norm(want)).abs() > 2.0 * core::f64::consts::PI - TOL,
                    "atan2({yf}, {xf}) = {got} vs {want}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // sub_simd / cross_simd bit-exactness (x86_64 + `simd` feature only)
    // -----------------------------------------------------------------------

    /// `Fix128::sub_simd` must be bit-identical to `-`, including borrow
    /// propagation from `lo` into `hi` and wrapping at the i64 boundary.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn fix128_sub_simd_is_bit_exact_with_operator_sub() {
        let cases: [(Fix128, Fix128); 8] = [
            (fi(7), fi(3)),
            (fi(3), fi(7)),
            (fi(-5), fi(9)),
            // lo borrow: 0 - tiny must borrow one from hi
            (Fix128::from_raw(1, 0), Fix128::from_raw(0, 1)),
            // lo wraps fully around
            (Fix128::from_raw(0, 0), Fix128::from_raw(0, u64::MAX)),
            // arbitrary fractional bit patterns
            (
                Fix128::from_raw(3, 0xABCD_EF01_2345_6789),
                Fix128::from_raw(-1, 0x1111_2222_3333_4444),
            ),
            (
                Fix128::from_raw(-7, 0xFEDC_BA98_7654_3210),
                Fix128::from_raw(5, 0xAAAA_BBBB_CCCC_DDDD),
            ),
            // i64 wrap on hi (wrapping semantics of `-` must be mirrored)
            (Fix128::from_raw(i64::MIN, 0), Fix128::from_raw(1, 0)),
        ];
        for (a, b) in cases {
            let scalar = a - b;
            // SAFETY: `sub_simd` has no preconditions; SSE2 is part of the
            // x86_64 baseline so the `target_feature(enable = "sse2")`
            // requirement is always satisfied on this target.
            let simd = unsafe { a.sub_simd(b) };
            assert_eq!(
                simd.hi, scalar.hi,
                "sub_simd hi mismatch for {a:?} - {b:?}: {:#018x} vs {:#018x}",
                simd.hi, scalar.hi
            );
            assert_eq!(
                simd.lo, scalar.lo,
                "sub_simd lo mismatch for {a:?} - {b:?}: {:#018x} vs {:#018x}",
                simd.lo, scalar.lo
            );
            // and the closed-form: (a - b) + b == a
            assert_eq!(simd + b, a);
        }
        // Explicit borrow check: (1.0) - (2^-64) == 0.FFFF…F (hi 0, lo MAX)
        // SAFETY: as above.
        let borrowed = unsafe { Fix128::from_raw(1, 0).sub_simd(Fix128::from_raw(0, 1)) };
        assert_eq!(borrowed, Fix128::from_raw(0, u64::MAX));
    }

    /// `Vec3Fix::cross_simd` must be bit-identical to `cross()` and satisfy
    /// the closed-form cross-product identities.
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    #[test]
    fn vec3_cross_simd_is_bit_exact_with_cross() {
        // Right-handed basis: x × y = z, y × z = x, z × x = y
        let x = v3i(1, 0, 0);
        let y = v3i(0, 1, 0);
        let z = v3i(0, 0, 1);
        assert_eq!(x.cross_simd(y), z);
        assert_eq!(y.cross_simd(z), x);
        assert_eq!(z.cross_simd(x), y);
        // Anti-commutativity and self-cross = 0
        assert_eq!(y.cross_simd(x), z.scale(fi(-1)));
        assert_eq!(x.cross_simd(x), Vec3Fix::ZERO);

        // Textbook integer case: (1,2,3) × (4,5,6) = (-3, 6, -3)
        assert_eq!(v3i(1, 2, 3).cross_simd(v3i(4, 5, 6)), v3i(-3, 6, -3));

        // Arbitrary fractional patterns: bit-exact with the scalar path,
        // and the result is orthogonal to both inputs (dot == 0 exactly is
        // not guaranteed after Fix128 rounding, so compare to the scalar
        // path's dot instead of to zero).
        let pairs = [
            (v3i(1, 2, 3), v3i(4, 5, 6)),
            (v3i(-7, 0, 11), v3i(2, -9, 5)),
            (
                Vec3Fix::new(
                    Fix128::from_raw(3, 0xABCD_EF01_2345_6789),
                    Fix128::from_raw(-1, 0x1111_2222_3333_4444),
                    Fix128::from_raw(7, 0xFEDC_BA98_7654_3210),
                ),
                Vec3Fix::new(
                    Fix128::from_raw(2, 0x9876_5432_10FE_DCBA),
                    Fix128::from_raw(5, 0xAAAA_BBBB_CCCC_DDDD),
                    Fix128::from_raw(-3, 0x0F0F_0F0F_0F0F_0F0F),
                ),
            ),
        ];
        for (a, b) in pairs {
            let scalar = a.cross(b);
            let simd = a.cross_simd(b);
            assert_eq!(simd.x.hi, scalar.x.hi, "{a:?} × {b:?}: x.hi");
            assert_eq!(simd.x.lo, scalar.x.lo, "{a:?} × {b:?}: x.lo");
            assert_eq!(simd.y.hi, scalar.y.hi, "{a:?} × {b:?}: y.hi");
            assert_eq!(simd.y.lo, scalar.y.lo, "{a:?} × {b:?}: y.lo");
            assert_eq!(simd.z.hi, scalar.z.hi, "{a:?} × {b:?}: z.hi");
            assert_eq!(simd.z.lo, scalar.z.lo, "{a:?} × {b:?}: z.lo");
            assert_eq!(simd.dot(a), scalar.dot(a));
            assert_eq!(simd.dot(b), scalar.dot(b));
            // a × b == -(b × a)
            assert_eq!(b.cross_simd(a), simd.scale(fi(-1)));
        }
    }
    /// `atan2` with |y| > |x|: the ratio `y / x` overflows the i64 integer
    /// part once |x| is a few ulp (1 / 2⁻⁶³ = 2⁶³), so before 1.2.0
    /// `atan2(1, 2 ulp)` returned −0.755. The complement form keeps every
    /// case within 1e-12 of the f64 reference, including the quadrant
    /// boundaries and the exact axes.
    #[test]
    #[allow(clippy::disallowed_methods)]
    fn atan2_is_accurate_when_x_is_a_few_ulp() {
        let ulp2 = Fix128::from_raw(0, 2);
        let cases = [
            (Fix128::ONE, ulp2),
            (Fix128::ONE, -ulp2),
            (-Fix128::ONE, ulp2),
            (-Fix128::ONE, -ulp2),
            (Fix128::ONE - Fix128::from_raw(0, 3), ulp2),
            (Fix128::from_ratio(7, 3), Fix128::from_ratio(-1, 5)),
            (Fix128::from_ratio(-7, 3), Fix128::from_ratio(1, 5)),
            (Fix128::from_ratio(1, 5), Fix128::from_ratio(-7, 3)),
            (Fix128::from_int(3), Fix128::from_int(4)),
            (Fix128::from_int(4), Fix128::from_int(3)),
        ];
        for (y, x) in cases {
            let got = Fix128::atan2(y, x).to_f64();
            let want = y.to_f64().atan2(x.to_f64());
            assert!(
                (got - want).abs() < 1e-12,
                "atan2({}, {}) = {got}, want {want}",
                y.to_f64(),
                x.to_f64()
            );
        }
        // exact axes are unchanged
        assert_eq!(Fix128::atan2(Fix128::ONE, Fix128::ZERO), Fix128::HALF_PI);
        assert_eq!(Fix128::atan2(-Fix128::ONE, Fix128::ZERO), -Fix128::HALF_PI);
        // atan(0) through CORDIC is ~5e-15, not exactly 0 (unchanged behaviour)
        assert!(Fix128::atan2(Fix128::ZERO, Fix128::ONE).abs() < Fix128::from_f64(1e-12));
    }
}
