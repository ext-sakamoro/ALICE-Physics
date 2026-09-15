//! Deterministic `f32` / `f64` transcendentals (cross-platform bit-exact).
//!
//! IEEE 754 guarantees that `+`, `-`, `*`, `/` and `sqrt` produce the same
//! bits on every platform Rust targets (SSE2+, aarch64, wasm32). It does
//! **not** cover `sin`, `exp`, `ln`, `powf`, `cbrt`, `hypot`, … — those come
//! from the platform `libm` and differ in the last ulp between macOS, glibc,
//! MSVC and wasm. A lockstep simulation that calls them diverges across
//! peers.
//!
//! Every function here is built from the guaranteed operations only —
//! integer range reduction, fixed-degree polynomials evaluated in a fixed
//! order, and bit-level exponent construction — so the result is a pure
//! function of the input bits on every target. `mul_add` is not used here
//! either, purely to keep this module's dependency surface minimal; the rest
//! of the crate may use it — IEEE 754 specifies fused multiply-add with a
//! single rounding, and `tests/determinism_golden_f32.rs` verifies identical
//! bits across hardware FMA (aarch64), software `fma` (x86_64 without FMA)
//! and wasm32.
//!
//! # Accuracy
//!
//! Measured over dense sweeps (see tests). `f32` functions are compared with
//! a correctly rounded reference (`f64` libm rounded once to `f32`), the
//! `f64` functions with the platform `f64` libm — which itself varies by
//! ≤ 1 ulp between macOS / glibc / MSVC, so those bounds carry that slack:
//!
//! | function | domain measured | max error |
//! |----------|-----------------|-----------|
//! | [`sin`] / [`cos`] | `|x| ≤ 100` | ≤ 2 ulp (Cephes single-precision coefficients) |
//! | [`exp`] | `[-87, 88]` | ≤ 2 ulp |
//! | [`ln`] | `[1e-30, 1e30]` | ≤ 1 ulp (musl `logf` algorithm) |
//! | [`cbrt`] | `[1e-30, 1e30]` | ≤ 1 ulp (bit-hack seed + 3 Newton steps) |
//! | [`hypot`] | finite | ≤ 1 ulp of `sqrt(x² + y²)` |
//! | [`powf`] | `x ∈ [1e-3, 1e3]`, `|y| ≤ 8` | ≤ 1 ulp (evaluated in `f64`, rounded once) |
//! | [`atan`] / [`atan2`] | `|x| ≤ 1e4`, all quadrants | ≤ 1 ulp (fdlibm in `f64`, rounded once) |
//! | [`asin`] / [`acos`] | `[-1, 1]` | ≤ 1 ulp (fdlibm in `f64`, rounded once) |
//! | [`tan`] | `|x| ≤ 100` | ≤ 1 ulp (fdlibm `k_sin`/`k_cos` in `f64`) |
//! | [`tanh`] | finite | ≤ 1 ulp (`exp64` based, `x` below 2^-14) |
//! | [`exp64`] / [`ln64`] | as above | ≤ 1 ulp on macOS libm, bound 2 across platform libms (fdlibm algorithms) |
//! | [`powf64`] | as above | ≤ 16 ulp measured 13 (`exp64(y·ln64 x)` with double-double argument; the residual step is bounded by `exp64`'s own ulp) |
//!
//! Large arguments to [`sin`] / [`cos`] (`|x| > 2¹³`) are still deterministic
//! but the single-precision Cody–Waite reduction loses accuracy, exactly as
//! `libm`'s own `sinf` does without Payne–Hanek.
//!
//! # Guarantee scope
//!
//! Bit-exactness relies on the target honouring IEEE 754 for the basic
//! operations: x86_64 (SSE2), aarch64, wasm32 and every other Rust tier-1/2
//! target qualify. 32-bit x86 built for x87 (`i586`) without SSE2, or any
//! build with fast-math style flags, is outside the guarantee.
//!
//! # Policy
//!
//! `clippy.toml` lists the `f32` / `f64` `libm` methods under
//! `disallowed-methods`; CI runs clippy with `-D warnings`, so a stray
//! `x.sin()` on a float is a red gate. Use these functions instead.

#![allow(clippy::excessive_precision)] // coefficients are quoted with their source precision

// ---------------------------------------------------------------------------
// f32
// ---------------------------------------------------------------------------

/// π/2 split into three single-precision pieces (Cephes `DP1..3` × 2) so
/// `x - k·π/2` is computed with ~24 extra bits for `|k| < 2^24`.
const PIO2_1: f32 = 1.570_312_5;
const PIO2_2: f32 = 4.837_512_969_970_703_125e-4;
const PIO2_3: f32 = 7.549_789_948_768_648e-8;
const FRAC_2_PI: f32 = core::f32::consts::FRAC_2_PI;

/// Sine polynomial on `[-π/4, π/4]` (Cephes `sinf`).
#[inline(always)]
fn sin_poly(r: f32) -> f32 {
    let z = r * r;
    let p = ((-1.951_529_589_1e-4 * z + 8.332_160_873_6e-3) * z - 1.666_665_461_1e-1) * z * r;
    r + p
}

/// Cosine polynomial on `[-π/4, π/4]` (Cephes `cosf`).
#[inline(always)]
fn cos_poly(r: f32) -> f32 {
    let z = r * r;
    let p = ((2.443_315_711_809_948e-5 * z - 1.388_731_625_493_765e-3) * z
        + 4.166_664_568_298_827e-2)
        * z
        * z;
    1.0 - 0.5 * z + p
}

/// Reduce `x` to `(k mod 4, r)` with `x = k·π/2 + r`, `|r| ≤ π/4`.
#[inline(always)]
fn reduce_pio2(x: f32) -> (i32, f32) {
    let kf = (x * FRAC_2_PI).round();
    let r = ((x - kf * PIO2_1) - kf * PIO2_2) - kf * PIO2_3;
    // `kf` is integral and |kf| < 2^31 for every finite f32 we accept; the
    // cast saturates for huge inputs, which only affects the (already
    // meaningless) quadrant of astronomically large arguments.
    let k = (kf as i32) & 3;
    (k, r)
}

/// Deterministic `sin(x)`.
#[must_use]
pub fn sin(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (k, r) = reduce_pio2(x);
    match k {
        0 => sin_poly(r),
        1 => cos_poly(r),
        2 => -sin_poly(r),
        _ => -cos_poly(r),
    }
}

/// Deterministic `cos(x)`.
#[must_use]
pub fn cos(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (k, r) = reduce_pio2(x);
    match k {
        0 => cos_poly(r),
        1 => -sin_poly(r),
        2 => -cos_poly(r),
        _ => sin_poly(r),
    }
}

const LOG2E: f32 = core::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_145_751_953_125;
const LN2_LO: f32 = 1.428_606_765_330_187e-6;

/// Build `2^k` for `-126 ≤ k ≤ 127` directly from the exponent bits.
#[inline(always)]
fn pow2i(k: i32) -> f32 {
    debug_assert!((-126..=127).contains(&k));
    f32::from_bits(((k + 127) as u32) << 23)
}

/// Deterministic `exp(x)`.
///
/// Overflows to `+inf` above ≈ 88.72, underflows to `0.0` below ≈ −103.97
/// (subnormal results are produced via a two-step scale, so the tail is
/// gradual, not a cliff).
#[must_use]
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    if x < -103.972_08 {
        return 0.0;
    }
    let kf = (x * LOG2E).round();
    let k = kf as i32;
    let r = (x - kf * LN2_HI) - kf * LN2_LO;
    // Cephes expf polynomial for e^r, |r| ≤ ln2/2.
    let p = (((((1.987_569_150_0e-4 * r + 1.398_199_950_7e-3) * r + 8.333_451_907_3e-3) * r
        + 4.166_579_589_4e-2)
        * r
        + 1.666_666_545_9e-1)
        * r
        + 0.5)
        * r
        * r
        + r
        + 1.0;
    // Scale by 2^k. Split the scale when k leaves the normal exponent range
    // so subnormal results and the overflow edge are handled without UB.
    if k > 127 {
        p * pow2i(127) * pow2i(k - 127)
    } else if k < -126 {
        p * pow2i(-126) * pow2i(k + 126)
    } else {
        p * pow2i(k)
    }
}

// musl `logf` coefficients.
const LG1: f32 = 0.666_666_626_93;
const LG2: f32 = 0.400_009_721_52;
const LG3: f32 = 0.284_987_866_88;
const LG4: f32 = 0.242_790_788_41;

/// Deterministic natural logarithm `ln(x)`.
///
/// `ln(0) = -inf`, `ln(x < 0) = NaN`, `ln(inf) = inf`.
#[must_use]
pub fn ln(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f32::INFINITY;
    }
    // Normalise subnormals so the exponent extraction below is exact.
    let mut bits = x.to_bits();
    let mut k: i32 = 0;
    if bits < 0x0080_0000 {
        let scaled = x * 33_554_432.0; // 2^25
        bits = scaled.to_bits();
        k -= 25;
    }
    // Reduce mantissa to [sqrt(2)/2, sqrt(2)).
    k += ((bits >> 23) as i32) - 127;
    bits = (bits & 0x007f_ffff) | 0x3f80_0000; // m in [1, 2)
    if bits >= 0x3fb5_04f3 {
        // m >= sqrt(2): halve it
        bits -= 0x0080_0000;
        k += 1;
    }
    let m = f32::from_bits(bits);
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * LG4);
    let t2 = z * (LG1 + w * LG3);
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let kf = k as f32;
    kf * LN2_HI - ((hfsq - (s * (hfsq + r) + kf * LN2_LO)) - f)
}

/// Deterministic `x^y` for real `y`.
///
/// Evaluated as `exp64(y · ln64 x)` in double precision and rounded once to
/// `f32`, so the argument of the exponential carries ~29 spare bits and the
/// result is within 1 ulp of the correctly rounded value over the measured
/// domain. Special cases follow IEEE `pow` for the inputs the engine uses:
/// `x < 0 → NaN` (non-integer `y` is the common case here, so no
/// integer-exponent sign rule is attempted), `x == 0 → 0 / 1 / inf` for
/// `y > 0 / y == 0 / y < 0`, `y == 0 → 1`.
#[must_use]
pub fn powf(x: f32, y: f32) -> f32 {
    if y == 0.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f32::INFINITY };
    }
    exp64(f64::from(y) * ln64(f64::from(x))) as f32
}

/// Deterministic integer power by repeated squaring (`n` may be negative).
///
/// Unlike `f32::powi`, which lowers to a compiler-rt routine, the
/// multiplication order here is fixed by this source.
#[must_use]
pub fn powi(x: f32, n: i32) -> f32 {
    let mut base = if n < 0 { 1.0 / x } else { x };
    let mut e = n.unsigned_abs();
    let mut acc = 1.0f32;
    while e > 0 {
        if e & 1 == 1 {
            acc *= base;
        }
        e >>= 1;
        if e > 0 {
            base *= base;
        }
    }
    acc
}

/// Deterministic cube root.
///
/// Bit-hack initial estimate (`bits / 3 + 0x2a51_37a0`) followed by three
/// Newton steps, then the sign is restored. `±0`, `±inf` and `NaN` pass
/// through.
#[must_use]
pub fn cbrt(x: f32) -> f32 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let ax = x.abs();
    // Subnormals: scale up by 2^24 (an exact cube-friendly power, 2^(3·8)),
    // take the root, scale back by 2^8.
    let (ax, post) = if ax.to_bits() < 0x0080_0000 {
        (ax * 16_777_216.0, 1.0 / 256.0)
    } else {
        (ax, 1.0)
    };
    let mut y = f32::from_bits(ax.to_bits() / 3 + 0x2a51_37a0);
    for _ in 0..3 {
        let y2 = y * y;
        y -= (y2 * y - ax) / (3.0 * y2);
    }
    let y = y * post;
    if x < 0.0 {
        -y
    } else {
        y
    }
}

/// Deterministic `sqrt(x² + y²)` with power-of-two scaling so intermediate
/// squares neither overflow nor flush to zero.
#[must_use]
pub fn hypot(x: f32, y: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    if ax.is_infinite() || ay.is_infinite() {
        return f32::INFINITY;
    }
    if ax.is_nan() || ay.is_nan() {
        return f32::NAN;
    }
    let m = if ax > ay { ax } else { ay };
    if m == 0.0 {
        return 0.0;
    }
    // Scaling by powers of two is exact, so the result is bit-identical to
    // the unscaled formula whenever that formula does not over/underflow.
    let (scale, unscale) = if m > 1.0e18 {
        (
            1.0 / 18_446_744_073_709_551_616.0,
            18_446_744_073_709_551_616.0,
        ) // 2^-64, 2^64
    } else if m < 1.0e-18 {
        (
            18_446_744_073_709_551_616.0,
            1.0 / 18_446_744_073_709_551_616.0,
        )
    } else {
        (1.0, 1.0)
    };
    let sx = ax * scale;
    let sy = ay * scale;
    (sx * sx + sy * sy).sqrt() * unscale
}

// ---------------------------------------------------------------------------
// f32 inverse trigonometric / tan / tanh (1.2.0)
//
// Evaluated in double precision with fdlibm's algorithms (basic IEEE
// operations only) and rounded once to `f32`, so the `f32` result is within
// 1 ulp of correctly rounded over the measured domains (see tests).
// ---------------------------------------------------------------------------

/// Deterministic `atan(x)`.
#[must_use]
pub fn atan(x: f32) -> f32 {
    atan64(f64::from(x)) as f32
}

/// Deterministic `atan2(y, x)` in `(-π, π]`, IEEE special cases as fdlibm.
#[must_use]
pub fn atan2(y: f32, x: f32) -> f32 {
    atan2_64(f64::from(y), f64::from(x)) as f32
}

/// Deterministic `asin(x)`; `NaN` outside `[-1, 1]`.
#[must_use]
pub fn asin(x: f32) -> f32 {
    asin64(f64::from(x)) as f32
}

/// Deterministic `acos(x)`; `NaN` outside `[-1, 1]`.
#[must_use]
pub fn acos(x: f32) -> f32 {
    acos64(f64::from(x)) as f32
}

/// Deterministic `tan(x)` (`sin/cos` of the double-precision kernels).
///
/// Argument reduction is exact for `|x| < 2^20·π/2`; beyond that the result
/// is still deterministic but loses accuracy (as `sin` / `cos` do).
#[must_use]
pub fn tan(x: f32) -> f32 {
    if !x.is_finite() {
        return f32::NAN;
    }
    let (k, r) = reduce_pio2_64(f64::from(x));
    let (s, c) = (k_sin64(r), k_cos64(r));
    // tan(x + kπ/2): even k → sin/cos, odd k → -cos/sin
    let t = if k & 1 == 0 { s / c } else { -c / s };
    t as f32
}

/// Deterministic `tanh(x)`.
///
/// `|x| < 2^-14 → x` (the cubic term is below half an `f32` ulp), otherwise
/// `(e^{2|x|} − 1) / (e^{2|x|} + 1)` in double precision via [`exp64`].
#[must_use]
pub fn tanh(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax < 6.103_515_625e-5 {
        return x;
    }
    if ax > 10.0 {
        // 1 - 2e^{-20} rounds to 1.0 in f32
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let e = exp64(2.0 * f64::from(ax));
    let t = ((e - 1.0) / (e + 1.0)) as f32;
    if x < 0.0 {
        -t
    } else {
        t
    }
}

// fdlibm s_atan.c
// fdlibm's atan(0.5) / atan(1) / atan(1.5) / atan(inf) high parts; the π/4
// and π/2 entries are exactly `core::f64::consts::{FRAC_PI_4, FRAC_PI_2}`
// (same f64 bits), spelled via the constants to keep `approx_constant` quiet.
const ATANHI: [f64; 4] = [
    4.636_476_090_008_060_935_15e-01,
    core::f64::consts::FRAC_PI_4,
    9.827_937_232_473_290_540_82e-01,
    core::f64::consts::FRAC_PI_2,
];
const ATANLO: [f64; 4] = [
    2.269_877_745_296_168_709_24e-17,
    3.061_616_997_868_383_017_93e-17,
    1.390_331_103_123_099_845_16e-17,
    6.123_233_995_736_766_035_87e-17,
];
const AT: [f64; 11] = [
    3.333_333_333_333_293_180_27e-01,
    -1.999_999_999_987_648_324_76e-01,
    1.428_571_427_593_712_314_80e-01,
    -1.111_111_040_546_235_578_80e-01,
    9.090_887_133_436_506_561_96e-02,
    -7.691_876_205_044_829_994_95e-02,
    6.661_073_137_387_531_206_69e-02,
    -5.833_570_133_790_573_486_45e-02,
    4.976_877_994_615_932_360_17e-02,
    -3.653_157_274_421_691_552_70e-02,
    1.628_582_011_536_578_236_23e-02,
];

/// Deterministic `atan(x)` in double precision (fdlibm `s_atan.c`).
#[must_use]
pub fn atan64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax >= 7.378_697_629_483_820_646_6e19 {
        // |x| >= 2^66: atan(x) = ±π/2 (the tiny term of fdlibm rounds away)
        return if x > 0.0 {
            ATANHI[3] + ATANLO[3]
        } else {
            -ATANHI[3] - ATANLO[3]
        };
    }
    let (id, t): (i32, f64) = if ax < 0.4375 {
        if ax < 7.450_580_596_923_828_125e-9 {
            // |x| < 2^-27
            return x;
        }
        (-1, x)
    } else if ax < 1.1875 {
        if ax < 0.6875 {
            (0, (2.0 * ax - 1.0) / (2.0 + ax))
        } else {
            (1, (ax - 1.0) / (ax + 1.0))
        }
    } else if ax < 2.4375 {
        (2, (ax - 1.5) / (1.0 + 1.5 * ax))
    } else {
        (3, -1.0 / ax)
    };
    let z = t * t;
    let w = z * z;
    let s1 = z * (AT[0] + w * (AT[2] + w * (AT[4] + w * (AT[6] + w * (AT[8] + w * AT[10])))));
    let s2 = w * (AT[1] + w * (AT[3] + w * (AT[5] + w * (AT[7] + w * AT[9]))));
    if id < 0 {
        return t - t * (s1 + s2);
    }
    let i = id as usize;
    let r = ATANHI[i] - ((t * (s1 + s2) - ATANLO[i]) - t);
    if x < 0.0 {
        -r
    } else {
        r
    }
}

const PI_64: f64 = core::f64::consts::PI;
const PI_LO_64: f64 = 1.224_646_799_147_353_207_2e-16;
const PIO2_HI_64: f64 = core::f64::consts::FRAC_PI_2;
const PIO2_LO_64: f64 = 6.123_233_995_736_766_035_87e-17;
const PIO4_HI_64: f64 = core::f64::consts::FRAC_PI_4;

/// Deterministic `atan2(y, x)` in double precision (fdlibm `e_atan2.c`).
#[must_use]
pub fn atan2_64(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 1.0 {
        return atan64(y);
    }
    // m: bit0 = sign(y), bit1 = sign(x)
    let m = (y.is_sign_negative() as u32) | ((x.is_sign_negative() as u32) << 1);
    if y == 0.0 {
        return match m {
            0 | 1 => y,  // atan(±0, +anything) = ±0
            2 => PI_64,  // atan(+0, -anything) = π
            _ => -PI_64, // atan(-0, -anything) = -π
        };
    }
    if x == 0.0 {
        return if y < 0.0 { -PIO2_HI_64 } else { PIO2_HI_64 };
    }
    if x.is_infinite() {
        if y.is_infinite() {
            return match m {
                0 => PIO4_HI_64,
                1 => -PIO4_HI_64,
                2 => 3.0 * PIO4_HI_64,
                _ => -3.0 * PIO4_HI_64,
            };
        }
        return match m {
            0 => 0.0,
            1 => -0.0,
            2 => PI_64,
            _ => -PI_64,
        };
    }
    if y.is_infinite() {
        return if y < 0.0 { -PIO2_HI_64 } else { PIO2_HI_64 };
    }
    let ex = ((x.to_bits() >> 52) & 0x7ff) as i32;
    let ey = ((y.to_bits() >> 52) & 0x7ff) as i32;
    let k = ey - ex;
    let z = if k > 60 {
        PIO2_HI_64 + 0.5 * PI_LO_64
    } else if x < 0.0 && k < -60 {
        0.0
    } else {
        atan64((y / x).abs())
    };
    match m {
        0 => z,
        1 => -z,
        2 => PI_64 - (z - PI_LO_64),
        _ => (z - PI_LO_64) - PI_64,
    }
}

// fdlibm e_asin.c / e_acos.c
const PS0: f64 = 1.666_666_666_666_666_574_15e-01;
const PS1: f64 = -3.255_658_186_224_009_154_05e-01;
const PS2: f64 = 2.012_125_321_348_629_258_81e-01;
const PS3: f64 = -4.005_553_450_067_941_140_27e-02;
const PS4: f64 = 7.915_349_942_898_145_321_76e-04;
const PS5: f64 = 3.479_331_075_960_211_675_70e-05;
const QS1: f64 = -2.403_394_911_734_414_218_78e+00;
const QS2: f64 = 2.020_945_760_233_505_694_71e+00;
const QS3: f64 = -6.882_839_716_054_532_930_30e-01;
const QS4: f64 = 7.703_815_055_590_193_527_91e-02;

#[inline(always)]
fn asin_pq(t: f64) -> (f64, f64) {
    let p = t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))));
    let q = 1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)));
    (p, q)
}

/// `x` with the low 32 bits of its mantissa cleared (fdlibm `SET_LOW_WORD(w, 0)`).
#[inline(always)]
fn clear_low_word(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000)
}

fn asin64(x: f64) -> f64 {
    let ax = x.abs();
    if ax >= 1.0 {
        if ax == 1.0 {
            return x * PIO2_HI_64 + x * PIO2_LO_64;
        }
        return f64::NAN;
    }
    if ax < 0.5 {
        if ax < 7.450_580_596_923_828_125e-9 {
            return x;
        }
        let t = x * x;
        let (p, q) = asin_pq(t);
        let w = p / q;
        return x + x * w;
    }
    let w = 1.0 - ax;
    let t = w * 0.5;
    let (p, q) = asin_pq(t);
    let s = t.sqrt();
    let r = if ax >= 0.975 {
        let w = p / q;
        PIO2_HI_64 - (2.0 * (s + s * w) - PIO2_LO_64)
    } else {
        let w = clear_low_word(s);
        let c = (t - w * w) / (s + w);
        let r = p / q;
        let p = 2.0 * s * r - (PIO2_LO_64 - 2.0 * c);
        let q = PIO4_HI_64 - 2.0 * w;
        PIO4_HI_64 - (p - q)
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

fn acos64(x: f64) -> f64 {
    let ax = x.abs();
    if ax >= 1.0 {
        if x == 1.0 {
            return 0.0;
        }
        if x == -1.0 {
            return PI_64 + 2.0 * PIO2_LO_64;
        }
        return f64::NAN;
    }
    if ax < 0.5 {
        if ax < 6.938_893_903_907_228_377_6e-18 {
            // |x| < 2^-57
            return PIO2_HI_64 + PIO2_LO_64;
        }
        let z = x * x;
        let (p, q) = asin_pq(z);
        let r = p / q;
        return PIO2_HI_64 - (x - (PIO2_LO_64 - x * r));
    }
    if x < 0.0 {
        // |x| >= 0.5 and negative (fdlibm branches on the sign here, so x == -0.5 lands here)
        let z = (1.0 + x) * 0.5;
        let (p, q) = asin_pq(z);
        let s = z.sqrt();
        let r = p / q;
        let w = r * s - PIO2_LO_64;
        return PI_64 - 2.0 * (s + w);
    }
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let df = clear_low_word(s);
    let c = (z - df * df) / (s + df);
    let (p, q) = asin_pq(z);
    let r = p / q;
    let w = r * s + c;
    2.0 * (df + w)
}

// fdlibm k_sin.c / k_cos.c kernels on |r| <= π/4, and a Cody–Waite
// reduction with the 33-bit / 33-bit / 53-bit split of π/2 (e_rem_pio2.c),
// exact for |x| < 2^20·π/2.
const S1: f64 = -1.666_666_666_666_663_243_48e-01;
const S2: f64 = 8.333_333_333_322_489_461_24e-03;
const S3: f64 = -1.984_126_982_985_794_931_34e-04;
const S4: f64 = 2.755_731_370_707_006_767_89e-06;
const S5: f64 = -2.505_076_025_340_686_341_95e-08;
const S6: f64 = 1.589_690_995_211_550_102_21e-10;
const C1: f64 = 4.166_666_666_666_660_190_37e-02;
const C2: f64 = -1.388_888_888_887_410_957_49e-03;
const C3: f64 = 2.480_158_728_947_672_941_78e-05;
const C4: f64 = -2.755_731_435_139_066_330_35e-07;
const C5: f64 = 2.087_572_321_298_174_827_90e-09;
const C6: f64 = -1.135_964_755_778_819_482_65e-11;
const PIO2_1_64: f64 = 1.570_796_326_734_125_614_17e+00;
const PIO2_1T_64: f64 = 6.077_100_506_506_192_249_32e-11;
const PIO2_2_64: f64 = 6.077_100_506_303_965_976_60e-11;
const PIO2_2T_64: f64 = 2.022_266_248_795_950_631_54e-21;

#[inline(always)]
fn k_sin64(x: f64) -> f64 {
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    x + v * (S1 + z * r)
}

#[inline(always)]
fn k_cos64(x: f64) -> f64 {
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + ((1.0 - w) - hz + (z * r))
}

/// Reduce `x` to `(k mod 4, r)` with `x = k·π/2 + r`, `|r| ≤ π/4`, in double.
#[inline(always)]
fn reduce_pio2_64(x: f64) -> (i32, f64) {
    let kf = (x * core::f64::consts::FRAC_2_PI).round();
    let r = if kf.abs() < 1_048_576.0 {
        (x - kf * PIO2_1_64) - kf * PIO2_1T_64
    } else {
        ((x - kf * PIO2_1_64) - kf * PIO2_2_64) - kf * PIO2_2T_64
    };
    ((kf as i32) & 3, r)
}

// ---------------------------------------------------------------------------
// f64
// ---------------------------------------------------------------------------

const LN2_HI64: f64 = 6.931_471_803_691_238_164_90e-01;
const LN2_LO64: f64 = 1.908_214_929_270_587_700_02e-10;
const INV_LN2_64: f64 = core::f64::consts::LOG2_E;

// fdlibm `e_exp.c`
const P1: f64 = 1.666_666_666_666_660_190_37e-01;
const P2: f64 = -2.777_777_777_701_559_338_42e-03;
const P3: f64 = 6.613_756_321_437_934_361_17e-05;
const P4: f64 = -1.653_390_220_546_525_153_90e-06;
const P5: f64 = 4.138_136_797_057_238_460_39e-08;

/// Build `2^k` for `-1022 ≤ k ≤ 1023` directly from the exponent bits.
#[inline(always)]
fn pow2i64(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// Deterministic `exp(x)` in double precision (fdlibm algorithm).
#[must_use]
pub fn exp64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if x < -745.133_219_101_941_1 {
        return 0.0;
    }
    let kf = (x * INV_LN2_64).round();
    let k = kf as i32;
    let hi = x - kf * LN2_HI64;
    let lo = kf * LN2_LO64;
    let r = hi - lo;
    let t = r * r;
    let c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    let y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);
    if k > 1023 {
        y * pow2i64(1023) * pow2i64(k - 1023)
    } else if k < -1022 {
        y * pow2i64(-1022) * pow2i64(k + 1022)
    } else {
        y * pow2i64(k)
    }
}

// fdlibm `e_log.c`
const LG1_64: f64 = 6.666_666_666_666_735_130e-01;
const LG2_64: f64 = 3.999_999_999_940_941_908e-01;
const LG3_64: f64 = 2.857_142_874_366_239_149e-01;
const LG4_64: f64 = 2.222_219_843_214_978_396e-01;
const LG5_64: f64 = 1.818_357_216_161_805_012e-01;
const LG6_64: f64 = 1.531_383_769_920_937_332e-01;
const LG7_64: f64 = 1.479_819_860_511_658_591e-01;

/// Deterministic natural logarithm in double precision (fdlibm algorithm).
///
/// `ln64(0) = -inf`, `ln64(x < 0) = NaN`, `ln64(inf) = inf`.
#[must_use]
pub fn ln64(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let mut bits = x.to_bits();
    let mut k: i32 = 0;
    if bits < 0x0010_0000_0000_0000 {
        let scaled = x * 18_014_398_509_481_984.0; // 2^54
        bits = scaled.to_bits();
        k -= 54;
    }
    k += ((bits >> 52) as i32) - 1023;
    bits = (bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000; // m in [1, 2)
    if bits >= 0x3ff6_a09e_667f_3bcd {
        // m >= sqrt(2)
        bits -= 0x0010_0000_0000_0000;
        k += 1;
    }
    let m = f64::from_bits(bits);
    let f = m - 1.0;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2_64 + w * (LG4_64 + w * LG6_64));
    let t2 = z * (LG1_64 + w * (LG3_64 + w * (LG5_64 + w * LG7_64)));
    let r = t2 + t1;
    let hfsq = 0.5 * f * f;
    let kf = f64::from(k);
    kf * LN2_HI64 - ((hfsq - (s * (hfsq + r) + kf * LN2_LO64)) - f)
}

/// Exact product `a·b = p + e` (Dekker / Veltkamp splitting, no `fma`).
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    const SPLIT: f64 = 134_217_729.0; // 2^27 + 1
    let p = a * b;
    let ta = SPLIT * a;
    let a_hi = ta - (ta - a);
    let a_lo = a - a_hi;
    let tb = SPLIT * b;
    let b_hi = tb - (tb - b);
    let b_lo = b - b_hi;
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

/// Deterministic `x^y` in double precision, same special-case rules as
/// [`powf`].
///
/// `y · ln x` is formed as a double-double (`ln64` plus a one-step residual
/// correction, exact product via Dekker splitting) so the exponential's
/// argument error is not amplified by `|y · ln x|`; measured ≤ 13 ulp over
/// the documented domain (the residual is limited by `exp64`'s own rounding).
#[must_use]
pub fn powf64(x: f64, y: f64) -> f64 {
    if y == 0.0 {
        return 1.0;
    }
    if x.is_nan() || y.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 { 0.0 } else { f64::INFINITY };
    }
    if x.is_infinite() {
        return if y > 0.0 { f64::INFINITY } else { 0.0 };
    }
    // ln x = l_hi + l_lo: residual of one exp round trip recovers the low part.
    let l_hi = ln64(x);
    let l_lo = x * exp64(-l_hi) - 1.0;
    let (t_hi, mut t_lo) = two_prod(y, l_hi);
    t_lo += y * l_lo;
    if t_hi > 709.782_712_893_384 {
        return f64::INFINITY;
    }
    if t_hi < -745.133_219_101_941_1 {
        return 0.0;
    }
    exp64(t_hi) * (1.0 + t_lo)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // the reference values come from libm on purpose
mod tests {
    use super::*;

    fn ulp_diff32(a: f32, b: f32) -> u32 {
        if a == b {
            return 0;
        }
        let ia = a.to_bits() as i32;
        let ib = b.to_bits() as i32;
        (ia - ib).unsigned_abs()
    }

    fn ulp_diff64(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let ia = a.to_bits() as i64;
        let ib = b.to_bits() as i64;
        (ia - ib).unsigned_abs()
    }

    /// Correctly rounded `f32` reference: evaluate the platform `f64` libm
    /// (≤ 1 ulp of 53 bits) and round once to `f32`. Comparing against the
    /// platform `f32` libm directly would make the bound itself
    /// platform-dependent (MSVC `cbrtf` and macOS `cbrtf` disagree by 1 ulp).
    fn r32(f: impl Fn(f64) -> f64) -> impl Fn(f32) -> f32 {
        move |x| f(f64::from(x)) as f32
    }

    /// Dense sweep helper: max ulp distance to the reference over `n` points in [lo, hi].
    fn sweep32(lo: f32, hi: f32, n: u32, f: impl Fn(f32) -> f32, g: impl Fn(f32) -> f32) -> u32 {
        let mut worst = 0;
        for i in 0..=n {
            let x = lo + (hi - lo) * (i as f32 / n as f32);
            let a = f(x);
            let b = g(x);
            assert!(
                a.is_finite() == b.is_finite(),
                "finite mismatch at {x}: {a} vs {b}"
            );
            if a.is_finite() {
                worst = worst.max(ulp_diff32(a, b));
            }
        }
        worst
    }

    #[test]
    fn sin_cos_within_2_ulp_of_correctly_rounded() {
        assert!(sweep32(-100.0, 100.0, 400_000, sin, r32(f64::sin)) <= 2);
        assert!(sweep32(-100.0, 100.0, 400_000, cos, r32(f64::cos)) <= 2);
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(cos(0.0), 1.0);
        assert!(sin(f32::INFINITY).is_nan());
    }

    #[test]
    fn atan_asin_acos_tan_tanh_within_1_ulp_of_correctly_rounded() {
        assert!(sweep32(-1e4, 1e4, 400_000, atan, r32(f64::atan)) <= 1);
        assert!(sweep32(-4.0, 4.0, 400_000, atan, r32(f64::atan)) <= 1);
        assert!(sweep32(-1.0, 1.0, 400_000, asin, r32(f64::asin)) <= 1);
        {
            let mut worst = 0;
            let mut wx = 0.0f32;
            for i in 0..=400_000u32 {
                let x = -1.0 + 2.0 * (i as f32 / 400_000.0);
                let d = ulp_diff32(acos(x), (f64::from(x)).acos() as f32);
                if d > worst {
                    worst = d;
                    wx = x;
                }
            }
            assert!(
                worst <= 1,
                "acos worst {worst} ulp at {wx}: {} vs {}",
                acos(wx),
                (f64::from(wx)).acos() as f32
            );
        }
        assert!(sweep32(-100.0, 100.0, 400_000, tan, r32(f64::tan)) <= 1);
        assert!(sweep32(-12.0, 12.0, 400_000, tanh, r32(f64::tanh)) <= 1);
        assert!(sweep32(-1e-3, 1e-3, 100_000, tanh, r32(f64::tanh)) <= 1);
        // atan2: every quadrant, axes, mixed magnitudes
        let mut worst = 0;
        for iy in -200i32..=200 {
            for ix in -200i32..=200 {
                let (y, x) = (iy as f32 * 0.37, ix as f32 * 1.13);
                let a = atan2(y, x);
                let b = (f64::from(y)).atan2(f64::from(x)) as f32;
                worst = worst.max(ulp_diff32(a, b));
            }
        }
        assert!(worst <= 1, "atan2 worst {worst} ulp");
        for &(y, x) in &[
            (0.0f32, 1.0f32),
            (0.0, -1.0),
            (-0.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (1e30, 1e-30),
            (1e-30, -1e30),
        ] {
            let a = atan2(y, x);
            let b = (f64::from(y)).atan2(f64::from(x)) as f32;
            assert!(
                a.to_bits() == b.to_bits() || ulp_diff32(a, b) <= 1,
                "atan2({y}, {x}) = {a} vs {b}"
            );
        }
        assert!(asin(1.5).is_nan() && acos(-1.5).is_nan() && tan(f32::INFINITY).is_nan());
        assert_eq!(asin(1.0), core::f32::consts::FRAC_PI_2);
        assert_eq!(acos(1.0), 0.0);
        assert_eq!(tanh(0.0), 0.0);
        assert_eq!(tanh(50.0), 1.0);
        assert_eq!(tanh(-50.0), -1.0);
        // double-precision entry points against the platform libm (≤ 1 ulp slack, see module docs)
        let mut worst64 = 0;
        for i in 0..=200_000 {
            let x = -1e6 + 2e6 * (i as f64 / 200_000.0);
            worst64 = worst64.max(ulp_diff64(atan64(x), x.atan()));
        }
        assert!(worst64 <= 1, "atan64 worst {worst64} ulp");
    }

    #[test]
    fn exp_within_2_ulp_of_correctly_rounded() {
        assert!(sweep32(-87.0, 88.0, 400_000, exp, r32(f64::exp)) <= 2);
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        // subnormal tail is gradual
        assert!(exp(-100.0) > 0.0 && exp(-100.0) < f32::MIN_POSITIVE);
    }

    #[test]
    fn ln_within_1_ulp_of_correctly_rounded() {
        // log-spaced sweep
        let mut worst = 0;
        for i in 0..=200_000u32 {
            let x = 10f32.powf(-30.0 + 60.0 * (i as f32 / 200_000.0));
            worst = worst.max(ulp_diff32(ln(x), r32(f64::ln)(x)));
        }
        assert!(worst <= 1, "ln worst ulp {worst}");
        assert_eq!(ln(1.0), 0.0);
        assert_eq!(ln(0.0), f32::NEG_INFINITY);
        assert!(ln(-1.0).is_nan());
        // subnormal input
        assert!(ulp_diff32(ln(1.0e-40), r32(f64::ln)(1.0e-40)) <= 1);
    }

    #[test]
    fn cbrt_within_1_ulp_of_correctly_rounded() {
        let mut worst = 0;
        for i in 0..=200_000u32 {
            let x = 10f32.powf(-30.0 + 60.0 * (i as f32 / 200_000.0));
            worst = worst.max(ulp_diff32(cbrt(x), r32(f64::cbrt)(x)));
            worst = worst.max(ulp_diff32(cbrt(-x), r32(f64::cbrt)(-x)));
        }
        assert!(worst <= 1, "cbrt worst ulp {worst}");
        assert_eq!(cbrt(0.0), 0.0);
        assert_eq!(cbrt(8.0), 2.0);
        assert_eq!(cbrt(-27.0), -3.0);
        assert!(ulp_diff32(cbrt(1.0e-40), r32(f64::cbrt)(1.0e-40)) <= 1);
    }

    #[test]
    fn hypot_matches_sqrt_formula_and_scales() {
        assert_eq!(hypot(3.0, 4.0), 5.0);
        assert_eq!(hypot(0.0, 0.0), 0.0);
        assert_eq!(hypot(-3.0, 4.0), 5.0);
        // libm hypot is correctly rounded; ours is sqrt(x²+y²) → ≤ 1 ulp apart
        let mut worst = 0;
        for i in 0..=100_000u32 {
            let x = 1.0e-3 + 1.0e3 * (i as f32 / 100_000.0);
            let y = 1.0e3 - x;
            worst = worst.max(ulp_diff32(
                hypot(x, y),
                (f64::from(x).hypot(f64::from(y))) as f32,
            ));
        }
        assert!(worst <= 1, "hypot worst ulp {worst}");
        // no overflow / underflow where the naive formula would fail
        assert!(hypot(1.0e30, 1.0e30).is_finite());
        assert!(hypot(1.0e-30, 1.0e-30) > 0.0);
    }

    #[test]
    fn powf_within_1_ulp_of_correctly_rounded() {
        let mut worst = 0;
        for i in 0..=400u32 {
            let x = 10f32.powf(-3.0 + 6.0 * (i as f32 / 400.0));
            for j in 0..=160u32 {
                let y = -8.0 + 16.0 * (j as f32 / 160.0);
                worst = worst.max(ulp_diff32(
                    powf(x, y),
                    (f64::from(x).powf(f64::from(y))) as f32,
                ));
            }
        }
        assert!(worst <= 1, "powf worst ulp {worst}");
        assert_eq!(powf(2.0, 0.0), 1.0);
        assert_eq!(powf(0.0, 2.0), 0.0);
        assert_eq!(powf(0.0, -1.0), f32::INFINITY);
        assert!(powf(-2.0, 0.5).is_nan());
    }

    #[test]
    fn powi_matches_repeated_multiplication() {
        for n in 0..=12 {
            let mut expect = 1.0f32;
            for _ in 0..n {
                expect *= 1.7;
            }
            // repeated squaring differs from a left fold by rounding; accept ≤ 2 ulp
            assert!(ulp_diff32(powi(1.7, n), expect) <= 2, "n = {n}");
        }
        assert_eq!(powi(2.0, 10), 1024.0);
        assert_eq!(powi(2.0, -2), 0.25);
        assert_eq!(powi(5.0, 0), 1.0);
    }

    #[test]
    fn exp64_ln64_within_2_ulp_of_libm() {
        let mut worst_e = 0;
        for i in 0..=400_000u32 {
            let x = -700.0 + 1400.0 * (f64::from(i) / 400_000.0);
            worst_e = worst_e.max(ulp_diff64(exp64(x), x.exp()));
        }
        assert!(worst_e <= 2, "exp64 worst ulp {worst_e}");
        let mut worst_l = 0;
        for i in 0..=400_000u32 {
            let x = 10f64.powf(-300.0 + 600.0 * (f64::from(i) / 400_000.0));
            worst_l = worst_l.max(ulp_diff64(ln64(x), x.ln()));
        }
        assert!(worst_l <= 2, "ln64 worst ulp {worst_l}");
        assert_eq!(exp64(0.0), 1.0);
        assert_eq!(ln64(1.0), 0.0);
        assert_eq!(exp64(1000.0), f64::INFINITY);
        assert_eq!(exp64(-800.0), 0.0);
        assert_eq!(ln64(0.0), f64::NEG_INFINITY);
        assert!(ulp_diff64(ln64(1.0e-310), (1.0e-310f64).ln()) <= 2);
    }

    #[test]
    fn powf64_within_16_ulp_of_libm() {
        let mut worst = 0;
        for i in 0..=400u32 {
            let x = 10f64.powf(-3.0 + 6.0 * (f64::from(i) / 400.0));
            for j in 0..=160u32 {
                let y = -8.0 + 16.0 * (f64::from(j) / 160.0);
                worst = worst.max(ulp_diff64(powf64(x, y), x.powf(y)));
            }
        }
        assert!(worst <= 16, "powf64 worst ulp {worst}");
    }

    /// Bit-level pins: these must hold on every platform CI runs on. If one
    /// fails on a single target the algorithm has a platform-dependent step.
    #[test]
    fn bit_exact_pins() {
        assert_eq!(sin(1.0f32).to_bits(), 0x3f57_6aa5);
        assert_eq!(cos(1.0f32).to_bits(), 0x3f0a_5140);
        assert_eq!(exp(1.0f32).to_bits(), 0x402d_f854);
        assert_eq!(ln(10.0f32).to_bits(), 0x4013_5d8e);
        assert_eq!(cbrt(10.0f32).to_bits(), 0x4009_e242);
        assert_eq!(powf(3.0f32, 2.5).to_bits(), 0x4179_6a52);
        assert_eq!(hypot(1.5f32, 2.5).to_bits(), 0x403a_9728);
        assert_eq!(exp64(1.0f64).to_bits(), 0x4005_bf0a_8b14_576a);
        assert_eq!(ln64(10.0f64).to_bits(), 0x4002_6bb1_bbb5_5516);
        assert_eq!(powf64(3.0f64, 2.5).to_bits(), 0x402f_2d4a_4563_563d);
    }
}
