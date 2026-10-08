//! Audit S1-5 oracles for `alice_physics::math` (Fix128 / Vec3Fix / QuatFix / Mat3Fix).
//!
//! References are independent of the implementation: i128 / 256-bit integer arithmetic for the
//! fixed-point operators, f64 libm for transcendental functions (accuracy ~1e-16, far below the
//! tolerances used), closed forms for geometry.
#![allow(clippy::disallowed_methods)]
#![allow(clippy::many_single_char_names)]

use alice_physics::math::{Fix128, Mat3Fix, PolarError, QuatFix, Vec3Fix};

// ---------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------

fn raw(x: Fix128) -> i128 {
    ((x.hi as i128) << 64) | x.lo as i128
}
fn from_raw(r: i128) -> Fix128 {
    Fix128 {
        hi: (r >> 64) as i64,
        lo: r as u64,
    }
}
fn f(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn ulp() -> Fix128 {
    Fix128 { hi: 0, lo: 1 }
}
fn dist_ulps(a: Fix128, b: Fix128) -> u128 {
    raw(a).wrapping_sub(raw(b)).unsigned_abs()
}

/// SplitMix64: deterministic test vectors.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn fix(&mut self) -> Fix128 {
        // mixture of magnitudes: tiny, unit, large, extreme
        let sel = self.next() % 6;
        let (hi, lo) = (self.next(), self.next());
        match sel {
            0 => Fix128 { hi: 0, lo },
            1 => Fix128 {
                hi: (hi >> 62) as i64 - 2,
                lo,
            },
            2 => Fix128 {
                hi: (hi >> 40) as i64 - (1 << 23),
                lo,
            },
            3 => Fix128 {
                hi: (hi >> 20) as i64 - (1 << 43),
                lo,
            },
            4 => Fix128 { hi: hi as i64, lo },
            _ => Fix128 {
                hi: -((hi >> 50) as i64),
                lo,
            },
        }
    }
}

/// 256-bit signed product of two i128, returned as four u64 limbs (two's complement).
fn mul256(a: i128, b: i128) -> [u64; 4] {
    let neg = (a < 0) != (b < 0);
    let ua = a.unsigned_abs();
    let ub = b.unsigned_abs();
    let al = [ua as u64, (ua >> 64) as u64];
    let bl = [ub as u64, (ub >> 64) as u64];
    let mut p = [0u64; 4];
    for i in 0..2 {
        let mut carry: u128 = 0;
        for j in 0..2 {
            let t = (al[i] as u128) * (bl[j] as u128) + p[i + j] as u128 + carry;
            p[i + j] = t as u64;
            carry = t >> 64;
        }
        let mut k = i + 2;
        while carry != 0 && k < 4 {
            let t = p[k] as u128 + carry;
            p[k] = t as u64;
            carry = t >> 64;
            k += 1;
        }
    }
    if neg {
        let mut carry = 1u128;
        for limb in p.iter_mut() {
            let t = (!*limb) as u128 + carry;
            *limb = t as u64;
            carry = t >> 64;
        }
    }
    p
}
/// Reference `Mul`: bits [192:64] of the 256-bit product (floor, wrapping).
fn mul_ref(a: Fix128, b: Fix128) -> Fix128 {
    let p = mul256(raw(a), raw(b));
    from_raw((((p[2] as u128) << 64) | p[1] as u128) as i128)
}
/// Does the exact product fit the Fix128 range? (bits above 192 are a sign extension)
fn mul_fits(a: Fix128, b: Fix128) -> bool {
    let p = mul256(raw(a), raw(b));
    let top = p[3];
    let sign_ext = if (p[2] as i64) < 0 { u64::MAX } else { 0 };
    top == sign_ext
}
/// Reference `Div`: truncation toward zero of (|a| << 64) / |b|, wrapping to 128 bits.
fn div_ref(a: Fix128, b: Fix128) -> Fix128 {
    if b.is_zero() {
        return Fix128::ZERO;
    }
    let neg = (raw(a) < 0) != (raw(b) < 0);
    let ua = raw(a).unsigned_abs();
    let ub = raw(b).unsigned_abs();
    // restoring division of the 192-bit numerator ua << 64
    let mut rem: u128 = 0;
    let mut quot: u128 = 0;
    for i in (0..192).rev() {
        let bit = if i >= 64 { (ua >> (i - 64)) & 1 } else { 0 };
        let overflow = rem >> 127;
        rem = (rem << 1) | bit;
        quot <<= 1;
        if overflow != 0 || rem >= ub {
            rem = rem.wrapping_sub(ub);
            quot |= 1;
        }
    }
    let q = from_raw(quot as i128);
    if neg {
        -q
    } else {
        q
    }
}

// ---------------------------------------------------------------------------------------
// representation, constants, conversions
// ---------------------------------------------------------------------------------------

#[test]
fn constants_match_independent_hex_expansions() {
    assert_eq!((Fix128::ZERO.hi, Fix128::ZERO.lo), (0, 0));
    assert_eq!((Fix128::ONE.hi, Fix128::ONE.lo), (1, 0));
    assert_eq!((Fix128::NEG_ONE.hi, Fix128::NEG_ONE.lo), (-1, 0));
    // pi = 3.243F6A8885A308D3 13198A2E..., pi/2 = 1.921FB54442D18469 898CC517..., 2pi = 6.487ED5110B4611A6 2633145C...
    assert_eq!((Fix128::PI.hi, Fix128::PI.lo), (3, 0x243F_6A88_85A3_08D3));
    assert_eq!(
        (Fix128::HALF_PI.hi, Fix128::HALF_PI.lo),
        (1, 0x921F_B544_42D1_8469)
    );
    assert_eq!(
        (Fix128::TWO_PI.hi, Fix128::TWO_PI.lo),
        (6, 0x487E_D511_0B46_11A6)
    );
    assert_eq!(Fix128::PI.double(), Fix128::TWO_PI);
    assert_eq!(Fix128::PI.half(), Fix128::HALF_PI);
    assert_eq!(Fix128::from_int(-7), Fix128 { hi: -7, lo: 0 });
    assert_eq!(Fix128::from_raw(-3, 5), Fix128 { hi: -3, lo: 5 });
    assert_eq!(Fix128::from(5i64), Fix128::from_int(5));
    assert_eq!(Fix128::from(-5i32), Fix128::from_int(-5));
}

#[test]
fn f64_and_f32_conversions() {
    for &x in &[
        0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.25,
        -2.75,
        1e-9,
        -1e-9,
        123456.789,
        -987654.321,
        1e12,
        -1e12,
    ] {
        let y = Fix128::from_f64(x);
        assert!(
            (y.to_f64() - x).abs() <= x.abs() * 1e-15 + 2e-16,
            "{x}: {}",
            y.to_f64()
        );
    }
    // exact dyadic values: raw layout
    assert_eq!(
        Fix128::from_f64(-1.5),
        Fix128 {
            hi: -2,
            lo: 1 << 63
        }
    );
    assert_eq!(
        Fix128::from_f64(-0.5),
        Fix128 {
            hi: -1,
            lo: 1 << 63
        }
    );
    assert_eq!(Fix128::from_f64(2.75), Fix128 { hi: 2, lo: 3 << 62 });
    assert_eq!(Fix128::from_f32(1.25f32), Fix128 { hi: 1, lo: 1 << 62 });
    assert_eq!(Fix128::from_f32(-0.1f32).to_f32(), -0.1f32);
    assert_eq!(Fix128::from_f32(3.0e7f32).to_f32(), 3.0e7f32);
}

/// from_f64 / from_f32 of non-finite and out-of-range values must not produce a silent plausible number
/// (and must not panic in debug builds).
#[test]
#[ignore = "known defect: AUD-A-S1W5-018: Fix128::from_f64(NaN) returns ZERO silently, from_f64(+/-inf) and from_f64(-1e30) return huge arbitrary values or overflow-panic (hi - 1 on hi = i64::MIN in debug); no saturation/validation contract documented"]
fn from_f64_non_finite_and_out_of_range_are_not_silently_plausible() {
    let nan = std::panic::catch_unwind(|| Fix128::from_f64(f64::NAN));
    assert!(
        nan.is_err() || nan.unwrap() != Fix128::ZERO,
        "NaN mapped to ZERO"
    );
    let neg_huge = std::panic::catch_unwind(|| Fix128::from_f64(-1e30));
    let r = neg_huge.expect("from_f64(-1e30) panicked (arithmetic overflow)");
    assert!(r.is_negative(), "from_f64(-1e30) is not negative: {:?}", r);
}

/// to_f64 for a small negative value: hi = -1 and lo ~ 2^64 are summed as f64, cancelling to
/// an absolute error of ~1e-16 (relative 1e-7 at 5e-10).
#[test]
#[ignore = "known defect: AUD-A-S1W5-025: Fix128::to_f64 computes hi + lo/2^64 in f64, so a small negative value loses its relative precision (raw -1e10 -> relative error 2.8e-10 and growing as |x| shrinks; error floor 1.1e-16 absolute); doc says debug only. Fix: (raw as i128 as f64) / 2^64 is correctly rounded"]
fn to_f64_is_relatively_accurate_for_small_negative_values() {
    for &r in &[-10_000_000_000i128, -1_000_000_000_000, -123_456_789] {
        let x = from_raw(r);
        let want = r as f64 / 18446744073709551616.0;
        let got = x.to_f64();
        assert!(
            ((got - want) / want).abs() < 1e-14,
            "raw {r}: {got:e} vs {want:e}"
        );
    }
}

#[test]
fn from_ratio_floor_for_positive_and_symmetric_for_negative() {
    // exact dyadics
    assert_eq!(Fix128::from_ratio(3, 2), Fix128 { hi: 1, lo: 1 << 63 });
    assert_eq!(
        Fix128::from_ratio(-3, 2),
        Fix128 {
            hi: -2,
            lo: 1 << 63
        }
    );
    assert_eq!(
        Fix128::from_ratio(3, -2),
        Fix128 {
            hi: -2,
            lo: 1 << 63
        }
    );
    assert_eq!(Fix128::from_ratio(-3, -2), Fix128 { hi: 1, lo: 1 << 63 });
    assert_eq!(Fix128::from_ratio(0, 7), Fix128::ZERO);
    assert_eq!(Fix128::from_ratio(5, 0), Fix128::ZERO);
    assert_eq!(Fix128::from_ratio(8, 1), Fix128::from_int(8));
    // 1/3 = 0.5555.. hex truncated
    assert_eq!(
        Fix128::from_ratio(1, 3),
        Fix128 {
            hi: 0,
            lo: 0x5555_5555_5555_5555
        }
    );
    assert_eq!(
        Fix128::from_ratio(2, 3),
        Fix128 {
            hi: 0,
            lo: 0xAAAA_AAAA_AAAA_AAAA
        }
    );
    assert_eq!(Fix128::from_ratio(-1, 3), -Fix128::from_ratio(1, 3));
    // property: n/d truncated: from_ratio(n,d) * d <= n < (from_ratio + ulp) * d  (exact integer check)
    let mut rng = Rng(7);
    for _ in 0..2000 {
        let n = (rng.next() >> 11) as i64 - (1 << 52);
        let d = ((rng.next() >> 20) as i64).max(1);
        let q = Fix128::from_ratio(n, d);
        let (qr, nr) = (raw(q), (n as i128) << 64);
        // |q| is floor(|n|/|d| * 2^64) in magnitude
        let (qm, nm) = (qr.unsigned_abs(), nr.unsigned_abs());
        assert!(qm * d as u128 <= nm && nm < (qm + 1) * d as u128, "{n}/{d}");
        assert_eq!(qr < 0, n < 0 && qr != 0 || (n < 0 && qr == 0 && false));
    }
}

#[test]
fn predicates_floor_ceil_abs() {
    assert!(Fix128::ZERO.is_zero() && !Fix128::ONE.is_zero() && !ulp().is_zero());
    assert!(
        Fix128::from_f64(-0.25).is_negative()
            && !Fix128::ZERO.is_negative()
            && !ulp().is_negative()
    );
    assert!(from_raw(-1).is_negative());
    for &(x, fl, ce) in &[
        (2.5, 2.0, 3.0),
        (-2.5, -3.0, -2.0),
        (3.0, 3.0, 3.0),
        (-3.0, -3.0, -3.0),
        (0.25, 0.0, 1.0),
        (-0.25, -1.0, 0.0),
    ] {
        assert_eq!(f(x).floor(), f(fl), "floor({x})");
        assert_eq!(f(x).ceil(), f(ce), "ceil({x})");
    }
    assert_eq!(f(-5.5).abs(), f(5.5));
    assert_eq!(f(5.5).abs(), f(5.5));
    assert_eq!(Fix128::ZERO.abs(), Fix128::ZERO);
    assert_eq!(from_raw(-1).abs(), from_raw(1));
    assert_eq!(Fix128::from_int(-3).abs(), Fix128::from_int(3));
}

// ---------------------------------------------------------------------------------------
// add / sub / neg / shifts / ordering vs i128
// ---------------------------------------------------------------------------------------

#[test]
fn add_sub_neg_match_i128_wrapping_arithmetic() {
    let mut rng = Rng(11);
    for _ in 0..5000 {
        let (a, b) = (rng.fix(), rng.fix());
        assert_eq!(raw(a + b), raw(a).wrapping_add(raw(b)));
        assert_eq!(raw(a - b), raw(a).wrapping_sub(raw(b)));
        assert_eq!(raw(-a), raw(a).wrapping_neg());
        assert_eq!(a + b, b + a);
        assert_eq!((a + b) - b, a);
        assert_eq!(-(-a), a);
        assert_eq!(a.cmp(&b), raw(a).cmp(&raw(b)));
        assert_eq!(a < b, raw(a) < raw(b));
    }
    // carry across the lo -> hi boundary
    assert_eq!(
        Fix128 {
            hi: 0,
            lo: u64::MAX
        } + ulp(),
        Fix128::ONE
    );
    assert_eq!(
        Fix128::ONE - ulp(),
        Fix128 {
            hi: 0,
            lo: u64::MAX
        }
    );
    assert_eq!(-Fix128::ZERO, Fix128::ZERO);
}

#[test]
fn shifts_match_i128() {
    let mut rng = Rng(13);
    for _ in 0..2000 {
        let a = rng.fix();
        assert_eq!(raw(a.half()), raw(a) >> 1);
        assert_eq!(a.half(), a.shr_bits(1));
        assert_eq!(raw(a.double()), raw(a).wrapping_shl(1));
        for i in 0..=140u32 {
            let want = if i >= 128 { raw(a) >> 127 } else { raw(a) >> i };
            assert_eq!(raw(a.shr_bits(i)), want, "shr_bits({i}) of {:?}", a);
        }
    }
    assert_eq!(Fix128::ONE.shr_bits(1), f(0.5));
    assert_eq!(Fix128::NEG_ONE.shr_bits(1), f(-0.5));
    assert_eq!(
        from_raw(-1).shr_bits(70),
        from_raw(-1),
        "sign extension keeps -1 ulp"
    );
}

// ---------------------------------------------------------------------------------------
// multiply / divide
// ---------------------------------------------------------------------------------------

#[test]
fn mul_matches_256_bit_reference() {
    let mut rng = Rng(17);
    for _ in 0..20000 {
        let (a, b) = (rng.fix(), rng.fix());
        assert_eq!(a * b, mul_ref(a, b), "{:?} * {:?}", a, b);
        assert_eq!(a * b, b * a);
    }
    assert_eq!(Fix128::ONE * f(3.25), f(3.25));
    assert_eq!(Fix128::ZERO * f(3.25), Fix128::ZERO);
    // documented examples
    assert_eq!(
        from_raw(-1) * f(0.5),
        from_raw(-1),
        "-2^-64 * 0.5 floors to -2^-64"
    );
    let two40 = Fix128::from_int(1 << 40);
    assert_eq!(
        (two40 * two40).hi,
        0,
        "2^40 * 2^40 wraps to hi == 0 (documented)"
    );
    assert_eq!(f(1.5) * f(-2.0), f(-3.0));
}

fn checked_mul_cases() -> Vec<(Fix128, Fix128)> {
    let mut rng = Rng(19);
    let mut cases: Vec<(Fix128, Fix128)> = vec![];
    for _ in 0..20000 {
        cases.push((rng.fix(), rng.fix()));
    }
    // near the boundary: |a| ~ 2^32, |b| ~ 2^31 with random fractions and signs
    for _ in 0..20000 {
        let sa = if rng.next() & 1 == 0 { 1 } else { -1 };
        let sb = if rng.next() & 1 == 0 { 1 } else { -1 };
        let a = Fix128 {
            hi: sa * ((1i64 << 32) + (rng.next() % 3) as i64 - 1),
            lo: rng.next(),
        };
        let b = Fix128 {
            hi: sb * ((1i64 << 31) + (rng.next() % 3) as i64 - 1),
            lo: rng.next(),
        };
        cases.push((a, b));
    }
    cases
}

/// Soundness: Some(v) is only returned for an in-range product and v is the exact product.
#[test]
fn checked_mul_is_sound() {
    for (a, b) in checked_mul_cases() {
        if let Some(v) = a.checked_mul(b) {
            assert!(
                mul_fits(a, b),
                "Some for an out-of-range product {:?} * {:?}",
                a,
                b
            );
            assert_eq!(v, mul_ref(a, b));
        }
    }
}

/// Completeness: every in-range product is reported (doc: `None` only when out of range).
#[test]
fn checked_mul_is_complete() {
    let mut false_none = 0;
    let mut example = None;
    for (a, b) in checked_mul_cases() {
        if mul_fits(a, b) && a.checked_mul(b).is_none() {
            false_none += 1;
            example.get_or_insert((a, b));
        }
    }
    assert_eq!(
        false_none, 0,
        "in-range products reported as overflow, e.g. {:?}",
        example
    );
}

#[test]
fn checked_mul_basic_range_boundaries() {
    assert_eq!(f(3.0).checked_mul(f(4.0)), Some(f(12.0)));
    assert_eq!(f(-3.0).checked_mul(f(4.5)), Some(f(-13.5)));
    let two31 = Fix128::from_int(1 << 31);
    let two32 = Fix128::from_int(1 << 32);
    assert_eq!(two31.checked_mul(two31), Some(Fix128::from_int(1 << 62)));
    assert_eq!(
        two32.checked_mul(two31),
        None,
        "2^63 is out of range (max is 2^63 - 2^-64)"
    );
    assert_eq!(
        Fix128::from_int(1 << 40).checked_mul(Fix128::from_int(1 << 40)),
        None
    );
    assert_eq!(
        Fix128::from_int(-(1 << 32)).checked_mul(two31),
        Some(Fix128 {
            hi: i64::MIN,
            lo: 0
        }),
        "-2^63 is representable"
    );
    let v = Vec3Fix::from_int(1 << 20, 2, 3);
    assert!(v.checked_scale(f(2.0)).is_some());
    assert!(v.checked_scale(Fix128::from_int(1 << 50)).is_none());
    assert!(
        Vec3Fix::from_int(1, 1, 1 << 20)
            .checked_scale(Fix128::from_int(1 << 50))
            .is_none(),
        "any one component overflowing yields None"
    );
    assert_eq!(v.checked_scale(f(2.0)).unwrap(), v.scale(f(2.0)));
}

#[test]
fn div_matches_192_bit_reference_and_truncates_toward_zero() {
    let mut rng = Rng(23);
    for _ in 0..20000 {
        let (a, b) = (rng.fix(), rng.fix());
        assert_eq!(a / b, div_ref(a, b), "{:?} / {:?}", a, b);
    }
    assert_eq!(
        f(1.0) / f(3.0),
        Fix128 {
            hi: 0,
            lo: 0x5555_5555_5555_5555
        }
    );
    assert_eq!(f(-1.0) / f(3.0), Fix128::from_ratio(-1, 3));
    for _ in 0..500 {
        let (a, b) = (rng.fix(), rng.fix());
        if b.is_zero() {
            continue;
        }
        assert_eq!((-a) / b, -(a / b), "odd symmetry");
        assert_eq!(a / (-b), -(a / b));
    }
    assert_eq!(f(7.0) / Fix128::ZERO, Fix128::ZERO);
    assert_eq!(f(7.0).checked_div(Fix128::ZERO), None);
    assert_eq!(f(7.0).checked_div(f(2.0)), Some(f(3.5)));
    assert_eq!(f(5.5) / f(5.5), Fix128::ONE);
    assert_eq!(f(9.0) / Fix128::ONE, f(9.0));
    // inverse relation within one divisor-ulp: (a/b)*b <= a, close to a
    let q = f(10.0) / f(3.0);
    assert!(q * f(3.0) <= f(10.0) && f(10.0) - q * f(3.0) < f(1e-18));
}

// ---------------------------------------------------------------------------------------
// sqrt
// ---------------------------------------------------------------------------------------

#[test]
fn sqrt_is_exact_floor() {
    let mut rng = Rng(29);
    for _ in 0..5000 {
        let x = rng.fix();
        if x.is_negative() || x.is_zero() {
            assert_eq!(x.sqrt(), Fix128::ZERO);
            continue;
        }
        let r = raw(x.sqrt()) as u128;
        let xr = raw(x) as u128;
        // r^2 <= x * 2^64 < (r+1)^2 as 192-bit comparison
        let sq = |v: u128| -> [u64; 4] { mul256(v as i128, v as i128) };
        let lhs = sq(r);
        let rhs = [0u64, xr as u64, (xr >> 64) as u64, 0u64]; // x << 64
        let le = |a: [u64; 4], b: [u64; 4]| -> bool {
            for i in (0..4).rev() {
                if a[i] != b[i] {
                    return a[i] < b[i];
                }
            }
            true
        };
        assert!(le(lhs, rhs), "r^2 <= x for {:?}", x);
        let lt = !le(rhs, sq(r + 1)) || rhs != sq(r + 1);
        assert!(
            lt && le(rhs, sq(r + 1)) && rhs != sq(r + 1),
            "x < (r+1)^2 for {:?}",
            x
        );
    }
    assert_eq!(f(4.0).sqrt(), f(2.0));
    assert_eq!(f(2.25).sqrt(), f(1.5));
    assert_eq!(f(0.25).sqrt(), f(0.5));
    assert_eq!(
        ulp().sqrt(),
        Fix128 { hi: 0, lo: 1 << 32 },
        "sqrt(2^-64) = 2^-32"
    );
    assert_eq!(Fix128::ZERO.sqrt(), Fix128::ZERO);
    assert_eq!(f(-4.0).sqrt(), Fix128::ZERO);
    let big = Fix128 {
        hi: i64::MAX,
        lo: u64::MAX,
    }
    .sqrt();
    assert!(
        (big.to_f64() - 3_037_000_499.976).abs() < 1e-3,
        "sqrt(max) = {}",
        big.to_f64()
    );
}

// ---------------------------------------------------------------------------------------
// transcendentals
// ---------------------------------------------------------------------------------------

/// 48 CORDIC iterations leave a residual angle <= atan(2^-47) ~ 7.1e-15 rad: that is the accuracy
/// the algorithm can promise (documented tolerance: 1e-13).
#[test]
fn sin_cos_accuracy_over_the_reduced_range_and_identities() {
    let mut worst: f64 = 0.0;
    for k in -3000..=3000 {
        let x = k as f64 * 0.001_093_75; // covers about [-3.28, 3.28]
        let (s, c) = f(x).sin_cos();
        worst = worst
            .max((s.to_f64() - x.sin()).abs())
            .max((c.to_f64() - x.cos()).abs());
        assert!(
            (s.to_f64().powi(2) + c.to_f64().powi(2) - 1.0).abs() < 1e-13,
            "sin^2+cos^2 at {x}"
        );
        assert_eq!(s, f(x).sin());
        assert_eq!(c, f(x).cos());
    }
    assert!(worst < 1e-13, "worst abs error {worst:e}");
    // exact-ish landmarks
    assert!(
        Fix128::ZERO.sin().abs() < f(1e-14) && (Fix128::ZERO.cos() - Fix128::ONE).abs() < f(1e-14)
    );
    assert!((Fix128::HALF_PI.sin() - Fix128::ONE).abs() < f(1e-13));
    assert!(Fix128::HALF_PI.cos().abs() < f(1e-13));
    assert!(Fix128::PI.sin().abs() < f(1e-13) && (Fix128::PI.cos() + Fix128::ONE).abs() < f(1e-13));
    // odd / even symmetry
    for &x in &[0.3, 1.1, 2.0, 3.0] {
        assert!((f(-x).sin() + f(x).sin()).abs() < f(1e-13));
        assert!((f(-x).cos() - f(x).cos()).abs() < f(1e-13));
    }
}

#[test]
fn sin_cos_range_reduction_for_large_arguments() {
    for &x in &[
        4.0, 6.0, 6.5, 10.0, 100.0, 1000.0, -4.0, -6.5, -10.0, -100.0, -1000.0, 12345.678,
    ] {
        let (s, c) = f(x).sin_cos();
        // argument error from reduction grows as k * 2^-64: allow 1e-13 + |x| * 1e-17
        let tol = 1e-13 + x.abs() * 1e-17;
        assert!(
            (s.to_f64() - x.sin()).abs() < tol,
            "sin({x}) = {} vs {}",
            s.to_f64(),
            x.sin()
        );
        assert!((c.to_f64() - x.cos()).abs() < tol, "cos({x})");
    }
    // periodicity
    for &x in &[0.4, 1.7, -2.2] {
        let a = f(x).sin();
        let b = (f(x) + Fix128::TWO_PI).sin();
        assert!((a - b).abs() < f(1e-13));
    }
}

#[test]
fn atan_and_atan2_accuracy_and_quadrants() {
    for k in -400..=400 {
        let v = k as f64 * 0.05;
        let got = f(v).atan().to_f64();
        assert!(
            (got - v.atan()).abs() < 1e-13,
            "atan({v}) = {got} vs {}",
            v.atan()
        );
    }
    // large arguments approach +-pi/2 from inside
    for &v in &[1e3, 1e6, -1e6] {
        let got = f(v).atan().to_f64();
        assert!((got - v.atan()).abs() < 1e-9, "atan({v}) = {got}");
    }
    // atan2: all four quadrants, axes, zero
    for &(y, x) in &[
        (1.0, 1.0),
        (1.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 1.0),
        (0.5, 3.0),
        (3.0, 0.5),
        (-3.0, -0.5),
        (2.0, -7.0),
        (-0.1, -5.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
    ] {
        let got = Fix128::atan2(f(y), f(x)).to_f64();
        let want = f64::atan2(y, x);
        assert!(
            (got - want).abs() < 1e-13,
            "atan2({y},{x}) = {got} vs {want}"
        );
    }
    assert_eq!(Fix128::atan2(Fix128::ZERO, Fix128::ZERO), Fix128::ZERO);
    assert_eq!(Fix128::atan2(f(5.0), Fix128::ZERO), Fix128::HALF_PI);
    assert_eq!(Fix128::atan2(f(-5.0), Fix128::ZERO), -Fix128::HALF_PI);
    // y = tiny, x = 2 ulp: documented fix (returned -0.755 before 1.2.0)
    let got = Fix128::atan2(Fix128::ONE, Fix128 { hi: 0, lo: 2 }).to_f64();
    assert!(
        (got - std::f64::consts::FRAC_PI_2).abs() < 1e-13,
        "atan2(1, 2ulp) = {got}"
    );
}

#[test]
fn exp_ln_powf_accuracy_claims() {
    // exp: relative error <= 1e-6 for |x| <= 40 (documented)
    let mut worst: f64 = 0.0;
    for i in 0..=800 {
        let x = -40.0 + 80.0 * i as f64 / 800.0;
        let want = x.exp();
        let got = f(x).exp().to_f64();
        if want < 1e-9 {
            continue; // below 2^-64-dominated resolution
        }
        worst = worst.max(((got - want) / want).abs());
    }
    assert!(worst < 1e-6, "exp worst rel {worst:e}");
    assert_eq!(Fix128::ZERO.exp(), Fix128::ONE);
    // documented extremes
    assert_eq!(f(-45.0).exp(), Fix128::ZERO);
    assert_eq!(
        f(44.0).exp(),
        Fix128 {
            hi: i64::MAX,
            lo: u64::MAX
        },
        "saturates, does not wrap"
    );
    // ln
    for &x in &[
        1e-12, 1e-6, 0.001, 0.1, 0.5, 0.9, 1.0, 1.0000001, 1.5, 2.0, 2.7, 10.0, 12345.678, 1e9,
        1e15,
    ] {
        let xq = f(x).to_f64(); // the value actually passed (quantised to 2^-64)
        let got = f(x).ln().to_f64();
        assert!(
            (got - xq.ln()).abs() < 1e-14 + 1e-12 * xq.ln().abs(),
            "ln({x}) = {got} vs {}",
            xq.ln()
        );
    }
    assert_eq!(Fix128::ONE.ln(), Fix128::ZERO);
    assert_eq!(Fix128::ZERO.ln(), Fix128::ZERO);
    assert_eq!(f(-3.0).ln(), Fix128::ZERO);
    // ln(exp(x)) round trip
    for &x in &[-5.0, -1.0, 0.25, 1.0, 7.5, 20.0] {
        let rt = f(x).exp().ln().to_f64();
        assert!((rt - x).abs() < 1e-5, "ln(exp({x})) = {rt}");
    }
    // powf_pos closed forms
    assert_eq!(f(2.0).powf_pos(f(10.0)), f(1024.0));
    assert_eq!(f(9.0).powf_pos(f(0.5)), f(3.0));
    assert_eq!(f(5.0).powf_pos(Fix128::ZERO), Fix128::ONE);
    assert_eq!(f(7.0).powf_pos(Fix128::ONE), f(7.0));
    assert_eq!(f(0.0).powf_pos(Fix128::ONE), Fix128::ZERO);
    assert_eq!(f(-2.0).powf_pos(Fix128::ONE), Fix128::ZERO);
    assert_eq!(f(2.0).powf_pos(f(-1.0)), Fix128::ZERO);
    assert!((f(1.4).powf_pos(f(3.5)).to_f64() / 1.4f64.powf(3.5) - 1.0).abs() < 1e-6);
}

/// exp is documented to saturate for x > 43 and to be accurate up to there; true e^x is representable
/// until x = ln(2^63) = 43.668.
#[test]
fn exp_is_accurate_up_to_the_representable_maximum() {
    for &x in &[43.0, 43.2, 43.5, 43.66] {
        let got = f(x).exp().to_f64();
        let want = x.exp();
        assert!(
            ((got - want) / want).abs() < 1e-5,
            "exp({x}) = {got:e} vs {want:e}"
        );
    }
}

/// ceil / abs at the extreme of the range.
#[test]
fn ceil_and_abs_at_the_range_extremes() {
    let top = Fix128 {
        hi: i64::MAX,
        lo: 1,
    };
    let r = std::panic::catch_unwind(|| top.ceil());
    assert!(r.is_ok(), "ceil(i64::MAX + 2^-64) panicked");
    assert!(!r.unwrap().is_negative(), "ceil wrapped negative");
    let min = Fix128 {
        hi: i64::MIN,
        lo: 0,
    };
    assert!(!min.abs().is_negative(), "abs(MIN) is negative");
}

/// powf_pos: integer part of the exponent is capped at 64 (`exponent.hi.min(64)`).
#[test]
fn powf_pos_large_integer_exponent() {
    let got = f(1.01).powf_pos(f(100.0)).to_f64();
    assert!(
        (got / 1.01f64.powi(100) - 1.0).abs() < 1e-6,
        "1.01^100 = {got}"
    );
}

// ---------------------------------------------------------------------------------------
// Vec3Fix
// ---------------------------------------------------------------------------------------

fn v(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(f(x), f(y), f(z))
}
fn vclose(a: Vec3Fix, b: (f64, f64, f64), tol: f64, what: &str) {
    let g = (a.x.to_f64(), a.y.to_f64(), a.z.to_f64());
    assert!(
        (g.0 - b.0).abs() <= tol && (g.1 - b.1).abs() <= tol && (g.2 - b.2).abs() <= tol,
        "{what}: {:?} vs {:?}",
        g,
        b
    );
}

#[test]
fn vec3_constants_constructors_and_arithmetic() {
    assert_eq!(Vec3Fix::ZERO, v(0.0, 0.0, 0.0));
    assert_eq!(Vec3Fix::UNIT_X, v(1.0, 0.0, 0.0));
    assert_eq!(Vec3Fix::UNIT_Y, v(0.0, 1.0, 0.0));
    assert_eq!(Vec3Fix::UNIT_Z, v(0.0, 0.0, 1.0));
    assert_eq!(Vec3Fix::from_int(1, -2, 3), v(1.0, -2.0, 3.0));
    let a = v(1.0, 2.0, 3.0);
    let b = v(-4.0, 0.5, 2.0);
    assert_eq!(a + b, v(-3.0, 2.5, 5.0));
    assert_eq!(a - b, v(5.0, 1.5, 1.0));
    assert_eq!(-a, v(-1.0, -2.0, -3.0));
    assert_eq!(a * f(2.0), v(2.0, 4.0, 6.0));
    assert_eq!(a.scale(f(-0.5)), v(-0.5, -1.0, -1.5));
    assert_eq!(a / f(2.0), v(0.5, 1.0, 1.5));
    let arr: [Fix128; 3] = a.into();
    assert_eq!(arr, [f(1.0), f(2.0), f(3.0)]);
    assert_eq!(Vec3Fix::from(arr), a);
    let (x, y, z) = Vec3Fix::from_f32(1.5, -2.5, 0.25).to_f32();
    assert_eq!((x, y, z), (1.5, -2.5, 0.25));
    assert_eq!(format!("{}", v(1.0, 2.0, 3.0)), "(1.0000, 2.0000, 3.0000)");
}

#[test]
fn vec3_dot_cross_length_closed_forms() {
    let a = v(1.0, 2.0, 3.0);
    let b = v(-4.0, 0.5, 2.0);
    assert_eq!(a.dot(b), f(1.0 * -4.0 + 2.0 * 0.5 + 3.0 * 2.0));
    assert_eq!(
        a.cross(b),
        v(
            2.0 * 2.0 - 3.0 * 0.5,
            3.0 * -4.0 - 1.0 * 2.0,
            1.0 * 0.5 - 2.0 * -4.0
        )
    );
    // cross identities
    assert_eq!(a.cross(a), Vec3Fix::ZERO);
    assert_eq!(a.cross(b), -(b.cross(a)));
    assert_eq!(a.cross(b).dot(a), Fix128::ZERO);
    assert_eq!(a.cross(b).dot(b), Fix128::ZERO);
    assert_eq!(
        Vec3Fix::UNIT_X.cross(Vec3Fix::UNIT_Y),
        Vec3Fix::UNIT_Z,
        "right-handed"
    );
    assert_eq!(Vec3Fix::UNIT_Y.cross(Vec3Fix::UNIT_Z), Vec3Fix::UNIT_X);
    assert_eq!(Vec3Fix::UNIT_Z.cross(Vec3Fix::UNIT_X), Vec3Fix::UNIT_Y);
    // Lagrange: |a x b|^2 = |a|^2 |b|^2 - (a.b)^2
    let lhs = a.cross(b).length_squared();
    let rhs = a.length_squared() * b.length_squared() - a.dot(b) * a.dot(b);
    assert_eq!(lhs, rhs);
    assert_eq!(a.length_squared(), f(14.0));
    assert_eq!(v(3.0, 4.0, 12.0).length(), f(13.0));
    assert_eq!(Vec3Fix::ZERO.length(), Fix128::ZERO);
    // simd-named entry points (scalar-equivalent on every platform)
    assert_eq!(a.dot_simd(b), a.dot(b));
    assert_eq!(a.length_squared_simd(), a.length_squared());
}

#[test]
fn vec3_normalize_family() {
    let n = v(3.0, 4.0, 12.0).normalize();
    vclose(n, (3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0), 1e-18, "normalize");
    assert!((n.length().to_f64() - 1.0).abs() < 1e-17);
    assert_eq!(Vec3Fix::ZERO.normalize(), Vec3Fix::ZERO);
    assert_eq!(Vec3Fix::ZERO.try_normalize(), None);
    assert_eq!(v(3.0, 4.0, 12.0).try_normalize(), Some(n));
    let (u, len) = v(3.0, 4.0, 12.0).normalize_with_length();
    assert_eq!(len, f(13.0));
    vclose(
        u,
        (3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0),
        1e-17,
        "normalize_with_length",
    );
    assert_eq!(
        Vec3Fix::ZERO.normalize_with_length(),
        (Vec3Fix::ZERO, Fix128::ZERO)
    );
    assert_eq!(Vec3Fix::UNIT_Y.normalize(), Vec3Fix::UNIT_Y);
    // negative components
    vclose(v(-2.0, 0.0, 0.0).normalize(), (-1.0, 0.0, 0.0), 0.0, "axis");
}

/// A non-zero vector must not normalise to ZERO; the length of the result must be 1.
#[test]
#[ignore = "known defect: AUD-A-S1W5-022: Vec3Fix::normalize / try_normalize / normalize_with_length treat any vector with |v| < ~2.3e-10 as zero-length (length_squared underflows 2^-64): normalize((1e-11,0,0)) = ZERO, try_normalize = None; |normalize(v)| - 1 reaches 3e-8 at |v| = 1e-6 and ~3% at 1e-9"]
fn vec3_normalize_small_nonzero_vectors() {
    let tiny = v(1e-11, 0.0, 0.0);
    assert!(
        tiny.try_normalize().is_some(),
        "try_normalize(1e-11 x) = None"
    );
    assert!(tiny.normalize() != Vec3Fix::ZERO);
    let small = v(1e-6, 2e-6, 2e-6).normalize();
    assert!(
        (small.length().to_f64() - 1.0).abs() < 1e-9,
        "|normalize| = {}",
        small.length().to_f64()
    );
}

// ---------------------------------------------------------------------------------------
// QuatFix
// ---------------------------------------------------------------------------------------

fn q(x: f64, y: f64, z: f64, w: f64) -> QuatFix {
    QuatFix::new(f(x), f(y), f(z), f(w))
}

#[test]
fn quat_algebra_hamilton_product_and_conjugate() {
    let i = q(1.0, 0.0, 0.0, 0.0);
    let j = q(0.0, 1.0, 0.0, 0.0);
    let k = q(0.0, 0.0, 1.0, 0.0);
    let one = QuatFix::IDENTITY;
    assert_eq!(one, q(0.0, 0.0, 0.0, 1.0));
    assert_eq!(i.mul(j), k);
    assert_eq!(j.mul(k), i);
    assert_eq!(k.mul(i), j);
    assert_eq!(j.mul(i), q(0.0, 0.0, -1.0, 0.0));
    assert_eq!(i.mul(i), q(0.0, 0.0, 0.0, -1.0));
    assert_eq!(j.mul(j), q(0.0, 0.0, 0.0, -1.0));
    assert_eq!(k.mul(k), q(0.0, 0.0, 0.0, -1.0));
    assert_eq!(i.mul(j).mul(k), q(0.0, 0.0, 0.0, -1.0), "ijk = -1");
    let a = q(1.0, -2.0, 0.5, 3.0);
    let b = q(0.25, 4.0, -1.0, 2.0);
    assert_eq!(a.mul(one), a);
    assert_eq!(one.mul(a), a);
    // (ab)* = b* a*, |ab|^2 = |a|^2 |b|^2
    assert_eq!(a.mul(b).conjugate(), b.conjugate().mul(a.conjugate()));
    assert_eq!(
        a.mul(b).length_squared(),
        a.length_squared() * b.length_squared()
    );
    assert_eq!(a.conjugate(), q(-1.0, 2.0, -0.5, 3.0));
    assert_eq!(a.length_squared(), f(1.0 + 4.0 + 0.25 + 9.0));
    assert_eq!(q(2.0, 3.0, 6.0, 0.0).length(), f(7.0));
    // associativity
    let c = q(-3.0, 1.0, 2.0, 0.5);
    assert_eq!(a.mul(b).mul(c), a.mul(b.mul(c)));
    assert_eq!(
        format!("{}", q(1.0, 2.0, 3.0, 4.0)),
        "(1.0000, 2.0000, 3.0000, 4.0000)"
    );
    let arr: [Fix128; 4] = a.into();
    assert_eq!(QuatFix::from(arr), a);
}

#[test]
fn quat_normalize_and_axis_angle_rotation_rodrigues() {
    let n = q(1.0, 2.0, 2.0, 4.0).normalize();
    assert!((n.length().to_f64() - 1.0).abs() < 1e-17);
    assert_eq!(
        QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO).normalize(),
        QuatFix::IDENTITY
    );
    assert!((n.x.to_f64() - 1.0 / 5.0).abs() < 1e-17);
    // axis-angle: components
    let axis = v(0.0, 0.0, 2.0); // normalised internally
    let qa = QuatFix::from_axis_angle(axis, f(1.0));
    assert!(
        (qa.z.to_f64() - 0.5f64.sin()).abs() < 1e-13
            && (qa.w.to_f64() - 0.5f64.cos()).abs() < 1e-13
    );
    assert!(qa.x.is_zero() && qa.y.is_zero());
    // rotate_vec follows Rodrigues for several axes/angles
    for &(ax, ay, az, ang) in &[
        (0.0f64, 0.0f64, 1.0f64, 1.0f64),
        (1.0, 0.0, 0.0, -0.7),
        (1.0, 2.0, 2.0, 2.5),
        (-1.0, 1.0, 0.5, 0.3),
        (0.0, 1.0, 0.0, 3.0),
    ] {
        let len = (ax * ax + ay * ay + az * az).sqrt();
        let (kx, ky, kz) = (ax / len, ay / len, az / len);
        let r = QuatFix::from_axis_angle(v(ax, ay, az), f(ang));
        let p = (0.3, -1.2, 2.0);
        let dot = kx * p.0 + ky * p.1 + kz * p.2;
        let cr = (
            ky * p.2 - kz * p.1,
            kz * p.0 - kx * p.2,
            kx * p.1 - ky * p.0,
        );
        let (c, s) = (ang.cos(), ang.sin());
        let want = (
            p.0 * c + cr.0 * s + kx * dot * (1.0 - c),
            p.1 * c + cr.1 * s + ky * dot * (1.0 - c),
            p.2 * c + cr.2 * s + kz * dot * (1.0 - c),
        );
        vclose(r.rotate_vec(v(p.0, p.1, p.2)), want, 1e-12, "rodrigues");
    }
    // composition: rotate(a*b, v) = rotate(a, rotate(b, v))
    let a = QuatFix::from_axis_angle(v(1.0, 0.0, 0.0), f(0.8));
    let b = QuatFix::from_axis_angle(v(0.0, 1.0, 1.0), f(-1.3));
    let pv = v(1.0, 2.0, 3.0);
    let lhs = a.mul(b).rotate_vec(pv);
    let rhs = a.rotate_vec(b.rotate_vec(pv));
    vclose(
        lhs,
        (rhs.x.to_f64(), rhs.y.to_f64(), rhs.z.to_f64()),
        1e-12,
        "composition order",
    );
    // inverse
    let back = a.conjugate().rotate_vec(a.rotate_vec(pv));
    vclose(back, (1.0, 2.0, 3.0), 1e-12, "conjugate undoes rotation");
    // rotation by 90 deg about z maps x -> y
    let r90 = QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI);
    vclose(
        r90.rotate_vec(Vec3Fix::UNIT_X),
        (0.0, 1.0, 0.0),
        1e-13,
        "x->y",
    );
    // identity rotation
    assert_eq!(QuatFix::IDENTITY.rotate_vec(pv), pv);
}

#[test]
fn quat_from_axis_angle_zero_axis_is_a_unit_quaternion() {
    let r = QuatFix::from_axis_angle(Vec3Fix::ZERO, f(1.0));
    assert!(
        (r.length().to_f64() - 1.0).abs() < 1e-12,
        "|q| = {}",
        r.length().to_f64()
    );
}

// ---------------------------------------------------------------------------------------
// Mat3Fix
// ---------------------------------------------------------------------------------------

fn m(c0: (f64, f64, f64), c1: (f64, f64, f64), c2: (f64, f64, f64)) -> Mat3Fix {
    Mat3Fix::from_cols(
        v(c0.0, c0.1, c0.2),
        v(c1.0, c1.1, c1.2),
        v(c2.0, c2.1, c2.2),
    )
}
fn mclose(a: Mat3Fix, b: Mat3Fix, tol: f64, what: &str) {
    for (ca, cb) in [(a.col0, b.col0), (a.col1, b.col1), (a.col2, b.col2)] {
        vclose(ca, (cb.x.to_f64(), cb.y.to_f64(), cb.z.to_f64()), tol, what);
    }
}

#[test]
fn mat3_basic_ops_closed_forms() {
    let a = m((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 10.0)); // columns
                                                                   // mul_vec: A v
    let x = v(1.0, -1.0, 2.0);
    vclose(
        a.mul_vec(x),
        (1.0 - 4.0 + 14.0, 2.0 - 5.0 + 16.0, 3.0 - 6.0 + 20.0),
        0.0,
        "A v",
    );
    assert_eq!(a.transpose().transpose(), a);
    assert_eq!(a.transpose().col0, v(1.0, 4.0, 7.0));
    assert_eq!(Mat3Fix::IDENTITY.mul_vec(x), x);
    assert_eq!(Mat3Fix::ZERO.mul_vec(x), Vec3Fix::ZERO);
    assert_eq!(
        Mat3Fix::diagonal(f(2.0), f(3.0), f(4.0)).mul_vec(v(1.0, 1.0, 1.0)),
        v(2.0, 3.0, 4.0)
    );
    assert_eq!(a.scale(f(2.0)).col2, v(14.0, 16.0, 20.0));
    // determinant 1*(5*10-6*8) - 4*(2*10-3*8) + 7*(2*6-3*5) = 2 + 16 - 21 = -3
    assert_eq!(a.determinant(), f(-3.0));
    assert_eq!(Mat3Fix::IDENTITY.determinant(), Fix128::ONE);
    assert_eq!(a.transpose().determinant(), a.determinant());
    // multiplication: (AB)v = A(Bv); identity; det multiplicative; operator form agrees
    let b = m((0.5, 0.0, 1.0), (2.0, 1.0, 0.0), (0.0, -1.0, 3.0));
    assert_eq!(a.mul_mat(b).mul_vec(x), a.mul_vec(b.mul_vec(x)));
    assert_eq!(a.mul_mat(Mat3Fix::IDENTITY), a);
    assert_eq!(Mat3Fix::IDENTITY.mul_mat(a), a);
    assert_eq!(a * b, a.mul_mat(b));
    assert_eq!(
        a.mul_mat(b).determinant(),
        a.determinant() * b.determinant()
    );
    // (AB)^T = B^T A^T
    assert_eq!(
        a.mul_mat(b).transpose(),
        b.transpose().mul_mat(a.transpose())
    );
    assert_eq!(
        m((1.0, -7.0, 2.0), (0.5, 3.0, -9.5), (4.0, 0.0, -2.0)).max_abs_component(),
        f(9.5)
    );
    assert_eq!(Mat3Fix::ZERO.max_abs_component(), Fix128::ZERO);
    assert_eq!(Mat3Fix::IDENTITY.col0, Vec3Fix::UNIT_X);
}

#[test]
fn mat3_inverse_closed_form_and_singular() {
    let a = m((2.0, 0.0, 1.0), (1.0, 3.0, 0.0), (0.0, 1.0, 4.0));
    let inv = a.inverse().expect("det != 0");
    mclose(a.mul_mat(inv), Mat3Fix::IDENTITY, 1e-17, "A A^-1");
    mclose(inv.mul_mat(a), Mat3Fix::IDENTITY, 1e-17, "A^-1 A");
    // known inverse of diag
    let d = Mat3Fix::diagonal(f(2.0), f(4.0), f(0.5)).inverse().unwrap();
    assert_eq!(d, Mat3Fix::diagonal(f(0.5), f(0.25), f(2.0)));
    // singular
    assert_eq!(
        m((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 1.0)).inverse(),
        None
    );
    assert_eq!(Mat3Fix::ZERO.inverse(), None);
    // inverse of inverse
    mclose(inv.inverse().unwrap(), a, 1e-16, "(A^-1)^-1");
    // rotation: inverse == transpose
    let c = 0.6;
    let s = 0.8;
    let r = m((1.0, 0.0, 0.0), (0.0, c, s), (0.0, -s, c));
    mclose(r.inverse().unwrap(), r.transpose(), 1e-17, "R^-1 = R^T");
}

/// inverse() has no scale handling: 1/det overflows for 0 < det < 2^-63 and relative precision is lost as det -> 2^-64.
#[test]
fn mat3_inverse_of_small_scale_matrix() {
    let eps = 1e-5;
    let a = Mat3Fix::diagonal(f(eps), f(eps), f(eps));
    let inv = a.inverse().unwrap();
    assert!(
        (inv.col0.x.to_f64() * eps - 1.0).abs() < 1e-6,
        "inv[0][0] * eps = {}",
        inv.col0.x.to_f64() * eps
    );
    // det = 1.8e-19 (3.3 ulp): representable and non-zero
    let b = Mat3Fix::diagonal(f(5.7e-7), f(5.7e-7), f(5.7e-7));
    let dd = b.determinant();
    assert!(!dd.is_zero());
    if let Some(i) = b.inverse() {
        assert!(
            (i.col0.x.to_f64() * 5.7e-7 - 1.0).abs() < 1e-2,
            "garbage inverse: {}",
            i.col0.x.to_f64()
        );
    }
}

// ---------------------------------------------------------------------------------------
// polar_rotation
// ---------------------------------------------------------------------------------------

#[test]
fn polar_rotation_closed_forms_and_error_variants() {
    let floor = Fix128::ZERO;
    // an exactly orthogonal matrix is returned unchanged (documented: bit for bit)
    let rz90 = Mat3Fix::from_cols(Vec3Fix::UNIT_Y, -Vec3Fix::UNIT_X, Vec3Fix::UNIT_Z);
    assert_eq!(rz90.polar_rotation(floor, 32), Ok(rz90));
    let perm = Mat3Fix::from_cols(Vec3Fix::UNIT_Z, Vec3Fix::UNIT_X, Vec3Fix::UNIT_Y);
    assert_eq!(perm.polar_rotation(floor, 32), Ok(perm));
    assert_eq!(
        Mat3Fix::IDENTITY.polar_rotation(floor, 32),
        Ok(Mat3Fix::IDENTITY)
    );
    // a rotation built from 0.6 / 0.8 is not exactly orthogonal in Fix128 (det = 1 + O(2^-64)):
    // the iteration moves it by a few hundred ulp towards the nearest exact fixed point
    let (c, s) = (f(0.6), f(0.8));
    let rot = Mat3Fix::from_cols(
        Vec3Fix::UNIT_X,
        Vec3Fix::new(Fix128::ZERO, c, s),
        Vec3Fix::new(Fix128::ZERO, -s, c),
    );
    mclose(
        rot.polar_rotation(floor, 32).unwrap(),
        rot,
        1e-15,
        "0.6/0.8 rotation",
    );
    // F = R U with U = diag(1.5, 0.8, 1.1): rotation factor is R
    let u = Mat3Fix::diagonal(f(1.5), f(0.8), f(1.1));
    let fm = rot.mul_mat(u);
    let r = fm.polar_rotation(floor, 32).unwrap();
    mclose(r, rot, 1e-15, "R recovered");
    // orthogonality and det = +1
    mclose(r.transpose().mul_mat(r), Mat3Fix::IDENTITY, 1e-15, "R^T R");
    assert!((r.determinant().to_f64() - 1.0).abs() < 1e-15);
    // U = R^T F symmetric positive definite
    let ur = r.transpose().mul_mat(fm);
    mclose(ur, ur.transpose(), 1e-15, "U symmetric");
    assert!(ur.col0.x > Fix128::ZERO && ur.col1.y > Fix128::ZERO && ur.col2.z > Fix128::ZERO);
    // scale invariance of the factor
    let big = Mat3Fix::diagonal(f(1000.0), f(1000.0), f(100.0));
    mclose(
        big.polar_rotation(floor, 32).unwrap(),
        Mat3Fix::IDENTITY,
        1e-14,
        "diag(1e3,1e3,1e2) -> I",
    );
    // shear gamma = 0.4: F = [[1, 0.4, 0],[0,1,0],[0,0,1]] -> R = rotation by -atan(0.2)-ish; check R^T R = I, det 1
    let shear = m((1.0, 0.0, 0.0), (0.4, 1.0, 0.0), (0.0, 0.0, 1.0));
    let rs = shear.polar_rotation(floor, 32).unwrap();
    let th = (0.2f64).atan();
    // column-major: col0 = (cos, -sin?) ; F^T... independent: polar rotation of simple shear is rotation by -atan(gamma/2)
    mclose(
        rs,
        m(
            (th.cos(), -th.sin(), 0.0),
            (th.sin(), th.cos(), 0.0),
            (0.0, 0.0, 1.0),
        ),
        1e-14,
        "shear polar factor",
    );
    // errors
    assert_eq!(
        Mat3Fix::diagonal(f(1.0), f(1.0), f(-1.0)).polar_rotation(floor, 32),
        Err(PolarError::Inverted)
    );
    assert_eq!(
        Mat3Fix::diagonal(f(1.0), f(1.0), f(-1.0)).polar_rotation(f(-5.0), 32),
        Err(PolarError::Inverted),
        "negative floor cannot let a reflection through"
    );
    assert_eq!(
        Mat3Fix::ZERO.polar_rotation(floor, 32),
        Err(PolarError::Inverted)
    );
    assert_eq!(
        Mat3Fix::diagonal(f(2.0), f(2.0), f(2.0)).polar_rotation(f(8.0), 32),
        Err(PolarError::Degenerate),
        "det <= floor is Degenerate (inclusive)"
    );
    assert!(Mat3Fix::diagonal(f(2.0), f(2.0), f(2.0))
        .polar_rotation(f(7.99), 32)
        .is_ok());
    // budget: diag(1,1,1e-4) needs 19 steps (documented), 16 is not enough
    let flat = Mat3Fix::diagonal(f(1.0), f(1.0), f(1e-4));
    assert_eq!(
        flat.polar_rotation(floor, 16),
        Err(PolarError::NotConverged { steps: 16 })
    );
    assert!(flat.polar_rotation(floor, 32).is_ok());
    assert_eq!(
        flat.polar_rotation(floor, 18),
        Err(PolarError::NotConverged { steps: 18 })
    );
    assert!(flat.polar_rotation(floor, 19).is_ok());
    assert_eq!(
        rot.polar_rotation(floor, 0),
        Err(PolarError::NotConverged { steps: 0 })
    );
    // idempotence (diagonal case, documented in the body comment): feeding the result back
    // returns it bit for bit
    let rd = u.polar_rotation(floor, 32).unwrap();
    assert_eq!(rd.polar_rotation(floor, 32).unwrap(), rd);
}

/// The body comment of polar_rotation claims "feeding one back in returns it bit for bit".
#[test]
#[ignore = "known defect: AUD-A-S1W5-027: polar_rotation is not idempotent for general (non-diagonal) gradients: 105 of 200 sampled matrices change by 1 ulp when their own result is fed back (the result is not exactly orthogonal, so the first step is not exactly 1/2(F+F)); only exactly orthogonal inputs and the diagonal cases are bit-stable"]
fn polar_rotation_is_idempotent_bit_for_bit_on_general_gradients() {
    let fm = Mat3Fix::from_cols(
        Vec3Fix::new(f(1.0), f(0.3), f(-0.2)),
        Vec3Fix::new(f(0.1), f(1.2), f(0.5)),
        Vec3Fix::new(f(-0.3), f(0.2), f(0.9)),
    );
    let r1 = fm.polar_rotation(Fix128::ZERO, 64).unwrap();
    let r2 = r1.polar_rotation(Fix128::ZERO, 64).unwrap();
    assert_eq!(r1, r2);
}

// ---------------------------------------------------------------------------------------
// simd surface: width constant
// ---------------------------------------------------------------------------------------

#[test]
fn simd_width_is_consistent_with_the_documented_table() {
    use alice_physics::math::{simd_width, SIMD_WIDTH};
    assert_eq!(simd_width(), SIMD_WIDTH);
    assert!([1, 4, 8].contains(&SIMD_WIDTH));
    #[cfg(not(feature = "simd"))]
    assert_eq!(SIMD_WIDTH, 1);
    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    assert_eq!(SIMD_WIDTH, 4);
}

/// 0^0 is documented as ZERO (`self <= 0` returns ZERO); exp is accurate up to the saturation point;
/// ln of an exact power of two is k * LN2 (range reduction ends exactly at 1, no series error).
#[test]
fn powf_zero_base_exp_near_saturation_and_ln_of_powers_of_two() {
    assert_eq!(Fix128::ZERO.powf_pos(Fix128::ZERO), Fix128::ZERO);
    for &x in &[41.0, 42.0, 42.5, 42.9] {
        let got = f(x).exp().to_f64();
        let want = x.exp();
        assert!(
            ((got - want) / want).abs() < 1e-5,
            "exp({x}) = {got:e} vs {want:e}"
        );
    }
    // ln(2) is the LN2 constant 0xB17217F7D1CF79AC, ln(2^k) = k * LN2 exactly
    let ln2 = Fix128 {
        hi: 0,
        lo: 0xB172_17F7_D1CF_79AC,
    };
    for k in 1..=30i64 {
        assert_eq!(
            f((1u64 << k) as f64).ln(),
            Fix128::from_int(k) * ln2,
            "ln(2^{k})"
        );
    }
    assert_eq!(f(0.5).ln(), -ln2, "ln(1/2): range reduction by doubling");
}

/// Documented settling rule: the iteration stops at a change of 4 ulp, so the result is orthogonal to a
/// few ulp (measured 2-3).
#[test]
fn polar_rotation_result_is_orthogonal_to_a_few_ulp() {
    let (c, s) = (f(0.6), f(0.8));
    let rot = Mat3Fix::from_cols(
        Vec3Fix::UNIT_X,
        Vec3Fix::new(Fix128::ZERO, c, s),
        Vec3Fix::new(Fix128::ZERO, -s, c),
    );
    let ms = [
        rot.mul_mat(Mat3Fix::diagonal(f(1.5), f(0.8), f(1.1))),
        Mat3Fix::from_cols(
            Vec3Fix::new(f(1.0), f(0.0), f(0.0)),
            Vec3Fix::new(f(0.4), f(1.0), f(0.0)),
            Vec3Fix::new(f(0.0), f(0.0), f(1.0)),
        ),
        Mat3Fix::from_cols(
            Vec3Fix::new(f(1.0), f(0.3), f(-0.2)),
            Vec3Fix::new(f(0.1), f(1.2), f(0.5)),
            Vec3Fix::new(f(-0.3), f(0.2), f(0.9)),
        ),
    ];
    for m in ms {
        let r = m.polar_rotation(Fix128::ZERO, 64).unwrap();
        let e = r.transpose().mul_mat(r);
        for (a, b) in [
            (e.col0, Vec3Fix::UNIT_X),
            (e.col1, Vec3Fix::UNIT_Y),
            (e.col2, Vec3Fix::UNIT_Z),
        ] {
            for (x, y) in [(a.x, b.x), (a.y, b.y), (a.z, b.z)] {
                assert!(dist_ulps(x, y) <= 16, "R^T R - I = {} ulp", dist_ulps(x, y));
            }
        }
    }
}

/// A quotient whose true value does not fit in `Fix128`.
///
/// The `Div` impl computes the integer part as a `u128` and stores it with
/// `quot_hi as i64`, keeping only the low 64 bits: `1 / 2^-64 = 2^64` and
/// `50 / 2^-64 = 50 * 2^64` both come back as exactly `0`. `checked_div` only
/// rejects a zero divisor (its doc says it is bit-identical to `/` otherwise),
/// so it returns the same truncated value. The `Div` doc covers division by
/// zero and the rounding direction, not overflow.
///
/// Oracle, independent of the remedy (saturate, or report an error): when the
/// true quotient exceeds the representable range, a returned value keeps the
/// sign of the true quotient and is at least 1 in magnitude; `checked_div` may
/// instead return `None`. Found downstream as a zero safety factor in
/// `layer_adhesion` (`fos_is_never_below_one_when_applied_is_below_allowable`).
#[test]
#[ignore = "known defect: AUD-A-S1W5-030: Fix128::div truncates an out-of-range integer quotient to its low 64 bits, so 1 / 2^-64 and 50 / 2^-64 return exactly 0, and Fix128::checked_div only rejects a zero divisor, so it returns the same truncated value; the Div doc covers division by zero and rounding but not overflow"]
fn out_of_range_quotient_keeps_its_sign_and_magnitude() {
    let tiny = ulp(); // 2^-64
    let half = Fix128 { hi: 0, lo: 1 << 63 };
    let top = Fix128 {
        hi: i64::MAX,
        lo: 0,
    };
    // (dividend, divisor, true quotient is positive)
    let cases = [
        (Fix128::ONE, tiny, true),            // 2^64
        (Fix128::from_int(50), tiny, true),   // 50 * 2^64
        (-Fix128::from_int(50), tiny, false), // -50 * 2^64
        (Fix128::from_int(50), -tiny, false), // -50 * 2^64
        (top, half, true),                    // (2^63 - 1) * 2
    ];
    for (a, b, positive) in cases {
        let q = a / b;
        let ok = |v: Fix128| {
            if positive {
                v >= Fix128::ONE
            } else {
                v <= -Fix128::ONE
            }
        };
        assert!(ok(q), "{a:?} / {b:?} = {q:?}: the true quotient is out of range, the result lost its magnitude or sign");
        if let Some(c) = a.checked_div(b) {
            assert!(
                ok(c),
                "checked_div({a:?}, {b:?}) = Some({c:?}) for an out-of-range quotient"
            );
        }
    }
}
