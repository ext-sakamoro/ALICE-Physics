//! Oracles for the wiring of `src/math.rs`
//! (`examples/math_simd_and_transcendentals.rs`): `Fix128::atan`,
//! `Fix128::{add_simd, sub_simd}`, `Vec3Fix::{dot_simd, length_squared_simd,
//! cross_simd}`, `Vec3Fix::dot_batch_4`.
//!
//! # Platform gating
//!
//! Only `atan`, `dot_simd`, `length_squared_simd` compile on every platform
//! (`src/math.rs:1175-1204`). `add_simd`, `sub_simd`, `cross_simd`,
//! `dot_batch_4` are additionally `#[cfg(all(feature = "simd", target_arch =
//! "x86_64"))]`-gated in `src/math.rs` itself (`src/math.rs:359-379,
//! 1208-1239`) -- they do not exist at all elsewhere, so the tests for them
//! are behind the identical `#[cfg]` here.
//!
//! # Closed forms (every expected value is derived here, none from calling
//! the function under test for its own expected side)
//!
//! - `atan(v)`: `f64::atan` (platform `libm`, an independent algorithm from
//!   this crate's CORDIC) at 1e-12 tolerance -- the same convention as
//!   `src/math.rs::atan_matches_f64_within_1e12`. `atan(0)` is documented as
//!   ~5e-15 through CORDIC, not exactly 0 (`src/math.rs:3529`); the
//!   degenerate test below asserts the documented bound, not exact zero.
//! - `add_simd` / `sub_simd`: the `+` / `-` operators (`impl Add for
//!   Fix128`, `impl Sub for Fix128`), computed independently of
//!   `add_simd`/`sub_simd`.
//! - `dot_simd` / `length_squared_simd`: `Vec3Fix::dot` / `Vec3Fix::length_squared`
//!   (`src/math.rs:1049, 1067`) -- the doc's own bit-exactness claim.
//! - `cross_simd`: `Vec3Fix::cross` (`src/math.rs:1056`) plus the
//!   right-handed-basis closed forms and parallel/anti-parallel degeneracy
//!   (cross of a vector with a scalar multiple of itself is exactly zero).
//! - `dot_batch_4`: four independent `Vec3Fix::dot` calls.
//!
//! # Degenerate / extreme inputs
//!
//! `atan(0)` (documented nonzero residual), `atan` at an extreme magnitude
//! (`i64::MAX / 4`, bounded by the CORDIC gain growth of ~1.6467x so it
//! cannot overflow Fix128's +-9.2e18 range). `add_simd`/`sub_simd` at the
//! `i64` wraparound boundary (`Fix128` arithmetic is `wrapping_*`
//! end-to-end, never panics, never saturates -- `src/math.rs:381-400`).
//! `dot_simd`/`length_squared_simd` on the zero vector and on raw-pattern
//! components large enough that the underlying `Mul` wraps. `cross_simd` on
//! a vector crossed with itself, with `2v`, and with `-v` (all exactly
//! zero). `dot_batch_4` on a batch containing both a zero-vector pair and
//! an extreme-magnitude pair.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::{Fix128, Vec3Fix};

fn close(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() < tol
}

// ---------------------------------------------------------------------------
// atan -- platform-agnostic
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::disallowed_methods)]
fn atan_matches_f64_atan_at_known_angles() {
    // f64::atan is a platform `libm` call -- an independently implemented
    // algorithm, never Fix128::atan itself.
    let cases: [(Fix128, f64); 9] = [
        (Fix128::ZERO, 0.0),
        (Fix128::ONE, core::f64::consts::FRAC_PI_4),
        (Fix128::from_int(-1), -core::f64::consts::FRAC_PI_4),
        (Fix128::from_ratio(1, 2), 0.5_f64.atan()),
        (Fix128::from_ratio(1, 4), 0.25_f64.atan()),
        (Fix128::from_int(3), 3.0_f64.atan()),
        (Fix128::from_ratio(-1, 8), (-0.125_f64).atan()),
        (Fix128::from_int(7), 7.0_f64.atan()),
        (Fix128::from_int(1_000), 1_000.0_f64.atan()),
    ];
    for (v, want) in cases {
        let got = v.atan().to_f64();
        assert!(
            close(got, want, 1e-12),
            "atan({}) = {got} vs f64::atan = {want}",
            v.to_f64()
        );
    }
}

#[test]
#[allow(clippy::disallowed_methods)]
fn atan_odd_symmetry_and_complement_identity() {
    // atan(-x) == -atan(x) (odd function) and atan(x) + atan(1/x) == pi/2
    // for x > 0 -- both closed forms, independent of the implementation.
    for v in [
        Fix128::from_ratio(1, 3),
        Fix128::from_int(2),
        Fix128::from_ratio(17, 5),
        Fix128::from_int(123),
    ] {
        let a = v.atan().to_f64();
        let b = (-v).atan().to_f64();
        assert!(
            close(a + b, 0.0, 1e-12),
            "odd symmetry failed for {}",
            v.to_f64()
        );

        let c = (Fix128::ONE / v).atan().to_f64();
        assert!(
            close(a + c, core::f64::consts::FRAC_PI_2, 1e-12),
            "complement identity failed for {}",
            v.to_f64()
        );
    }
}

#[test]
fn atan_of_zero_is_within_the_documented_cordic_residual_not_exact_zero() {
    // src/math.rs:3529 documents atan(0) through CORDIC as ~5e-15, not
    // exactly 0 -- this pins the *documented contract* (a bound), not a
    // value read back from the implementation.
    let got = Fix128::ZERO.atan();
    assert!(
        got.abs() < Fix128::from_f64(1e-12),
        "atan(0) should be within the documented ~5e-15 CORDIC residual, got {}",
        got.to_f64()
    );
}

#[test]
#[allow(clippy::disallowed_methods)]
fn atan_at_extreme_magnitude_approaches_pi_over_2_without_overflow() {
    // CORDIC vectoring mode's vector-magnitude gain growth over 48
    // iterations is bounded (~1.6467x the starting magnitude), so this does
    // not overflow Fix128's +-9.2e18 range even at i64::MAX/4. atan is
    // monotonic and bounded above by pi/2 for all positive inputs.
    for v in [
        Fix128::from_int(i64::MAX / 4),
        Fix128::from_int(i64::MAX / 2),
        Fix128::from_int(-(i64::MAX / 4)),
    ] {
        let got = v.atan().to_f64();
        let want = v.to_f64().atan();
        assert!(
            close(got, want, 1e-12),
            "atan({}) = {got} vs f64::atan = {want}",
            v.to_f64()
        );
        // Note: CORDIC's residual convergence error (same family as the
        // documented ~5e-15 atan(0) residual) means this can overshoot
        // +-pi/2 by a few ulp at extreme magnitudes -- it is not exactly
        // bounded, so this is checked against the 1e-12 tolerance above,
        // not an exact `<= FRAC_PI_2` bound (that assertion failed in this
        // test's first run: atan(i64::MAX/4) = 1.5707963267949019, which is
        // ~5.3e-15 *above* FRAC_PI_2 = 1.5707963267948966).
        assert!(close(got.abs(), core::f64::consts::FRAC_PI_2, 1e-12));
    }
}

// ---------------------------------------------------------------------------
// dot_simd / length_squared_simd -- platform-agnostic
// ---------------------------------------------------------------------------

#[test]
fn dot_simd_matches_scalar_dot_on_integer_and_fractional_vectors() {
    let cases = [
        (
            Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(2),
                Fix128::from_int(3),
            ),
            Vec3Fix::new(
                Fix128::from_int(4),
                Fix128::from_int(5),
                Fix128::from_int(6),
            ),
        ),
        (
            Vec3Fix::new(
                Fix128::from_int(-3),
                Fix128::from_int(4),
                Fix128::from_int(-5),
            ),
            Vec3Fix::new(
                Fix128::from_int(6),
                Fix128::from_int(-7),
                Fix128::from_int(8),
            ),
        ),
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
    for (a, b) in cases {
        let scalar = a.dot(b);
        let simd = a.dot_simd(b);
        assert_eq!(
            (simd.hi, simd.lo),
            (scalar.hi, scalar.lo),
            "dot_simd({a:?}, {b:?}) must be bit-exact with dot()"
        );
    }
    // Closed-form value for the first (integer) case: 1*4+2*5+3*6 = 32.
    assert_eq!(cases[0].0.dot(cases[0].1), Fix128::from_int(32));
}

#[test]
fn length_squared_simd_matches_scalar_length_squared() {
    let v = Vec3Fix::new(
        Fix128::from_int(3),
        Fix128::from_int(4),
        Fix128::from_int(0),
    );
    let scalar = v.length_squared();
    let simd = v.length_squared_simd();
    assert_eq!((simd.hi, simd.lo), (scalar.hi, scalar.lo));
    // Closed form: 3^2 + 4^2 + 0^2 = 25.
    assert_eq!(scalar, Fix128::from_int(25));
}

#[test]
fn dot_simd_and_length_squared_simd_degenerate_and_extreme_inputs() {
    // Degenerate: zero vector -> exactly zero on both.
    let b = Vec3Fix::new(
        Fix128::from_int(100),
        Fix128::from_int(200),
        Fix128::from_int(300),
    );
    assert!(Vec3Fix::ZERO.dot_simd(b).is_zero());
    assert!(Vec3Fix::ZERO.length_squared_simd().is_zero());

    // Extreme magnitude: raw patterns large enough that the underlying
    // `Mul` wraps mod 2^128 (Fix128's documented wrapping contract,
    // src/math.rs:408-417). dot_simd must reproduce the exact same
    // wraparound as the scalar path, not diverge from it.
    let extreme = Vec3Fix::new(
        Fix128::from_raw(i64::MAX / 2, u64::MAX),
        Fix128::from_raw(i64::MIN / 2, 0x1234_5678_9ABC_DEF0),
        Fix128::from_raw(i64::MAX, 1),
    );
    let scalar_dot = extreme.dot(extreme);
    let simd_dot = extreme.dot_simd(extreme);
    assert_eq!(
        (simd_dot.hi, simd_dot.lo),
        (scalar_dot.hi, scalar_dot.lo),
        "dot_simd must reproduce Mul's wraparound bit-for-bit at extreme magnitudes"
    );
    let scalar_len_sq = extreme.length_squared();
    let simd_len_sq = extreme.length_squared_simd();
    assert_eq!(
        (simd_len_sq.hi, simd_len_sq.lo),
        (scalar_len_sq.hi, scalar_len_sq.lo),
        "length_squared_simd must reproduce length_squared's wraparound bit-for-bit"
    );
}

// ---------------------------------------------------------------------------
// add_simd / sub_simd / cross_simd / dot_batch_4 -- x86_64 + simd only,
// gated identically to src/math.rs itself.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86_64_simd_gated {
    use super::*;

    #[test]
    fn add_simd_matches_operator_add_including_i64_wraparound() {
        let cases: [(Fix128, Fix128); 6] = [
            (Fix128::from_int(7), Fix128::from_int(3)),
            (Fix128::from_int(-5), Fix128::from_int(9)),
            (
                Fix128::from_raw(3, 0xABCD_EF01_2345_6789),
                Fix128::from_raw(-1, 0x1111_2222_3333_4444),
            ),
            (Fix128::from_raw(0, u64::MAX), Fix128::from_raw(0, 1)),
            // i64 wraparound on `hi` (`Add` is `wrapping_add` end-to-end).
            (Fix128::from_raw(i64::MAX, 0), Fix128::from_raw(1, 0)),
            (Fix128::from_raw(i64::MAX, u64::MAX), Fix128::from_raw(0, 1)),
        ];
        for (a, b) in cases {
            let scalar = a + b;
            // SAFETY: add_simd has no preconditions; SSE2 is part of the
            // x86_64 baseline ABI, so `#[target_feature(enable = "sse2")]`
            // is always satisfied here.
            let simd = unsafe { a.add_simd(b) };
            assert_eq!(
                (simd.hi, simd.lo),
                (scalar.hi, scalar.lo),
                "add_simd({a:?}, {b:?}) must be bit-exact with the `+` operator, including wraparound"
            );
            // Closed-form check: (a + b) - b == a (wrapping `-` undoes
            // wrapping `+` exactly, mod 2^128).
            assert_eq!(scalar - b, a);
        }
    }

    #[test]
    fn sub_simd_matches_operator_sub_including_borrow_and_wraparound() {
        let cases: [(Fix128, Fix128); 7] = [
            (Fix128::from_int(7), Fix128::from_int(3)),
            (Fix128::from_int(3), Fix128::from_int(7)),
            (Fix128::from_int(-5), Fix128::from_int(9)),
            // lo borrow: 0 - tiny must borrow one from hi.
            (Fix128::from_raw(1, 0), Fix128::from_raw(0, 1)),
            // lo wraps fully around.
            (Fix128::from_raw(0, 0), Fix128::from_raw(0, u64::MAX)),
            // i64 wraparound on `hi`.
            (Fix128::from_raw(i64::MIN, 0), Fix128::from_raw(1, 0)),
            (Fix128::from_raw(i64::MIN, 0), Fix128::from_raw(0, 1)),
        ];
        for (a, b) in cases {
            let scalar = a - b;
            // SAFETY: as above.
            let simd = unsafe { a.sub_simd(b) };
            assert_eq!(
                (simd.hi, simd.lo),
                (scalar.hi, scalar.lo),
                "sub_simd({a:?}, {b:?}) must be bit-exact with the `-` operator, including borrow/wraparound"
            );
            // Closed-form check: (a - b) + b == a.
            assert_eq!(scalar + b, a);
        }
        // Explicit borrow check: 1.0 - 2^-64 == 0.FFFF...F (hi 0, lo MAX).
        // SAFETY: as above.
        let borrowed = unsafe { Fix128::from_raw(1, 0).sub_simd(Fix128::from_raw(0, 1)) };
        assert_eq!(borrowed, Fix128::from_raw(0, u64::MAX));
    }

    #[test]
    fn cross_simd_matches_scalar_cross_and_right_handed_basis_identities() {
        let x = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let y = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
        let z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
        // Closed-form right-handed basis: x*y=z, y*z=x, z*x=y.
        assert_eq!(x.cross_simd(y), z);
        assert_eq!(y.cross_simd(z), x);
        assert_eq!(z.cross_simd(x), y);
        // Anti-commutativity: y*x == -z.
        assert_eq!(y.cross_simd(x), z.scale(Fix128::from_int(-1)));

        let pairs = [
            (
                Vec3Fix::new(
                    Fix128::from_int(1),
                    Fix128::from_int(2),
                    Fix128::from_int(3),
                ),
                Vec3Fix::new(
                    Fix128::from_int(4),
                    Fix128::from_int(5),
                    Fix128::from_int(6),
                ),
            ),
            (
                Vec3Fix::new(
                    Fix128::from_int(-7),
                    Fix128::from_int(0),
                    Fix128::from_int(11),
                ),
                Vec3Fix::new(
                    Fix128::from_int(2),
                    Fix128::from_int(-9),
                    Fix128::from_int(5),
                ),
            ),
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
            assert_eq!(
                simd, scalar,
                "cross_simd({a:?}, {b:?}) must be bit-exact with cross()"
            );
        }
        // Textbook integer case: (1,2,3) x (4,5,6) = (-3, 6, -3).
        assert_eq!(
            pairs[0].0.cross_simd(pairs[0].1),
            Vec3Fix::new(
                Fix128::from_int(-3),
                Fix128::from_int(6),
                Fix128::from_int(-3)
            )
        );
    }

    #[test]
    fn cross_simd_degenerate_parallel_and_anti_parallel_vectors_are_exactly_zero() {
        let v = Vec3Fix::new(
            Fix128::from_int(3),
            Fix128::from_int(-5),
            Fix128::from_int(7),
        );
        // Self-cross.
        assert_eq!(v.cross_simd(v), Vec3Fix::ZERO);
        // Parallel (scalar multiple of itself).
        let parallel = v.scale(Fix128::from_int(2));
        assert_eq!(v.cross_simd(parallel), Vec3Fix::ZERO);
        // Anti-parallel.
        let anti_parallel = v.scale(Fix128::from_int(-1));
        assert_eq!(v.cross_simd(anti_parallel), Vec3Fix::ZERO);
        // Zero vector on either side.
        assert_eq!(v.cross_simd(Vec3Fix::ZERO), Vec3Fix::ZERO);
        assert_eq!(Vec3Fix::ZERO.cross_simd(v), Vec3Fix::ZERO);
    }

    #[test]
    fn dot_batch_4_matches_four_independent_dot_calls() {
        let a = [
            Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(0),
                Fix128::from_int(0),
            ),
            Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(1),
                Fix128::from_int(1),
            ),
            Vec3Fix::new(
                Fix128::from_int(2),
                Fix128::from_int(3),
                Fix128::from_int(4),
            ),
            Vec3Fix::new(
                Fix128::from_int(-1),
                Fix128::from_int(0),
                Fix128::from_int(5),
            ),
        ];
        let b = [
            Vec3Fix::new(
                Fix128::from_int(7),
                Fix128::from_int(8),
                Fix128::from_int(9),
            ),
            Vec3Fix::new(
                Fix128::from_int(1),
                Fix128::from_int(2),
                Fix128::from_int(3),
            ),
            Vec3Fix::new(
                Fix128::from_int(5),
                Fix128::from_int(6),
                Fix128::from_int(7),
            ),
            Vec3Fix::new(
                Fix128::from_int(4),
                Fix128::from_int(4),
                Fix128::from_int(4),
            ),
        ];
        let got = Vec3Fix::dot_batch_4(a, b);
        let want = [
            a[0].dot(b[0]),
            a[1].dot(b[1]),
            a[2].dot(b[2]),
            a[3].dot(b[3]),
        ];
        assert_eq!(got, want);
        // Closed-form values: 7, 6, 56, 16.
        assert_eq!(
            want,
            [
                Fix128::from_int(7),
                Fix128::from_int(6),
                Fix128::from_int(56),
                Fix128::from_int(16),
            ]
        );
    }

    #[test]
    fn dot_batch_4_degenerate_zero_and_extreme_magnitude_lanes() {
        let extreme_a = Vec3Fix::new(
            Fix128::from_raw(i64::MAX / 2, u64::MAX),
            Fix128::from_raw(-100, 0),
            Fix128::from_raw(0, 1),
        );
        let extreme_b = Vec3Fix::new(
            Fix128::from_raw(i64::MIN / 2, 1),
            Fix128::from_raw(100, 0),
            Fix128::from_raw(0, u64::MAX),
        );
        let x = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
        let q = Vec3Fix::new(
            Fix128::from_int(-7),
            Fix128::from_int(1),
            Fix128::from_int(4),
        );

        let a = [Vec3Fix::ZERO, extreme_a, x, extreme_a];
        let b = [q, extreme_b, x, extreme_b];
        let got = Vec3Fix::dot_batch_4(a, b);
        let want = [
            a[0].dot(b[0]),
            a[1].dot(b[1]),
            a[2].dot(b[2]),
            a[3].dot(b[3]),
        ];
        assert_eq!(
            got, want,
            "dot_batch_4 must match independent dot() exactly, including a zero-vector lane and extreme-magnitude lanes"
        );
        assert!(
            got[0].is_zero(),
            "zero-vector lane must dot to exactly zero"
        );
        assert_eq!(
            got[2],
            Fix128::from_int(-7),
            "unit-x lane: dot((1,0,0), (-7,1,4)) == -7"
        );
    }
}
