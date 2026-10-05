//! Production entry point for the last seven zero-production-caller items in
//! `src/math.rs`: `Fix128::atan`, `Fix128::{add_simd, sub_simd}`,
//! `Vec3Fix::{dot_simd, length_squared_simd, cross_simd}`, and
//! `Vec3Fix::dot_batch_4` (`scripts/wiring-baseline.txt:184-190`, all with
//! `#[cfg(test)]` self-references only).
//!
//! # Platform gating (read this before copying the "5 ungated / 2 gated"
//! split from elsewhere -- it does not match what's actually in this file)
//!
//! Only **three** of the seven compile on every platform: `atan`,
//! `dot_simd`, `length_squared_simd` (`src/math.rs:1175-1204`). `dot_simd`
//! dispatches to an SSE2 path on `x86_64` + `simd` and falls back to the
//! scalar `dot()` everywhere else; `length_squared_simd` is `dot_simd(self,
//! self)`. The other **four** -- `add_simd`, `sub_simd` (`src/math.rs:359,
//! 373`) and `cross_simd`, `dot_batch_4` (`src/math.rs:1208, 1230`) -- carry
//! `#[cfg(all(feature = "simd", target_arch = "x86_64"))]` on the whole
//! function, same as `cross_simd`/`dot_batch_4`. They do not exist at all
//! without that feature+arch combination. This file mirrors that exact gate
//! (not a 5/2 split) so it compiles everywhere.
//!
//! # Closed forms (every expected value is derived here, never by calling
//! the function under test for its own expected side)
//!
//! - `atan(v)`: compared against `f64::atan` (a platform `libm` call, an
//!   independent algorithm from this crate's CORDIC implementation), same
//!   1e-12 tolerance convention as `src/math.rs::atan_matches_f64_within_1e12`.
//!   `atan(0)` is *not* exactly zero through CORDIC (`src/math.rs:3529`
//!   documents ~5e-15); the degenerate case below asserts the documented
//!   bound (`< 1e-12`), not exact zero.
//! - `add_simd` / `sub_simd`: compared against the `+` / `-` operators
//!   (`impl Add for Fix128`, `impl Sub for Fix128`, `src/math.rs:381-400`),
//!   computed independently of `add_simd`/`sub_simd` themselves.
//! - `dot_simd` / `length_squared_simd`: compared against `Vec3Fix::dot` /
//!   `Vec3Fix::length_squared` (`src/math.rs:1049, 1067`), both of which
//!   expand to `x*rhs.x + y*rhs.y + z*rhs.z` -- the doc's own bit-exactness
//!   claim.
//! - `cross_simd`: compared against `Vec3Fix::cross` (`src/math.rs:1056`)
//!   plus the closed-form right-handed-basis identities (`x*y=z`, `y*z=x`,
//!   `z*x=y`) and the degenerate parallel/anti-parallel cases (cross of a
//!   vector with itself or its negation is exactly zero).
//! - `dot_batch_4`: compared against four independent `Vec3Fix::dot` calls.
//!
//! ```bash
//! cargo run --example math_simd_and_transcendentals --features std
//! cargo run --example math_simd_and_transcendentals --features std,simd   # x86_64 only: exercises add_simd/sub_simd/cross_simd/dot_batch_4
//! ```

use alice_physics::{Fix128, Vec3Fix};

fn close(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() < tol
}

// `f64::atan` below is the independent closed-form reference for
// `Fix128::atan` (platform libm, a different algorithm from this crate's
// CORDIC) -- exactly the use the determinism gate's `disallowed_methods`
// lint is not meant to catch (it guards production code computing results
// with non-deterministic libm, not oracle references in a wiring example),
// same convention as `src/math.rs::atan_matches_f64_within_1e12`.
#[allow(clippy::disallowed_methods)]
fn main() {
    // =======================================================================
    // atan -- compiles on every platform, no feature gate.
    // =======================================================================

    // Known angles: f64::atan is an independent algorithm (platform libm),
    // never Fix128::atan itself, so this is not circular.
    let atan_cases: [(Fix128, f64); 7] = [
        (Fix128::ZERO, 0.0),
        (Fix128::ONE, core::f64::consts::FRAC_PI_4),
        (Fix128::from_int(-1), -core::f64::consts::FRAC_PI_4),
        (Fix128::from_ratio(1, 2), 0.5_f64.atan()),
        (Fix128::from_int(3), 3.0_f64.atan()),
        (Fix128::from_ratio(-1, 8), (-0.125_f64).atan()),
        (Fix128::from_int(1_000_000), 1_000_000.0_f64.atan()),
    ];
    for (v, want) in atan_cases {
        let got = v.atan();
        println!(
            "[math] atan({}) = {} (closed form f64::atan = {want})",
            v.to_f64(),
            got.to_f64()
        );
        assert!(
            close(got.to_f64(), want, 1e-12),
            "atan({}) = {} vs {want}",
            v.to_f64(),
            got.to_f64()
        );
    }

    // Identities, independent of the golden-bit-pattern pins in src/math.rs:
    // odd symmetry atan(-x) = -atan(x), and the complement identity
    // atan(x) + atan(1/x) = pi/2 for x > 0.
    for v in [
        Fix128::from_ratio(1, 3),
        Fix128::from_int(2),
        Fix128::from_ratio(17, 5),
    ] {
        let a = v.atan().to_f64();
        let b = (-v).atan().to_f64();
        assert!(close(a + b, 0.0, 1e-12), "odd symmetry failed for {v:?}");
        let c = (Fix128::ONE / v).atan().to_f64();
        assert!(
            close(a + c, core::f64::consts::FRAC_PI_2, 1e-12),
            "complement identity failed for {v:?}"
        );
    }
    println!("[math] atan: odd symmetry and complement identity hold for 3 sample points");

    // Degenerate input: atan(0) is documented as ~5e-15 through CORDIC, not
    // exactly 0 (src/math.rs:3529) -- the bound is the documented contract,
    // not a hand-wave.
    let atan_zero = Fix128::ZERO.atan();
    println!(
        "[math] atan(0) = {} (documented: nonzero but < 1e-12)",
        atan_zero.to_f64()
    );
    assert!(
        atan_zero.abs() < Fix128::from_f64(1e-12),
        "atan(0) should be within the documented ~5e-15 CORDIC residual, got {}",
        atan_zero.to_f64()
    );

    // Extreme magnitude: CORDIC vectoring mode's gain growth is bounded
    // (~1.6467x the input magnitude over 48 iterations), so a very large
    // input does not overflow Fix128's +-9.2e18 range; the result should
    // be within 1e-12 of pi/2 (the exact f64::atan reference). It is *not*
    // guaranteed to stay on one side of pi/2: the same CORDIC residual
    // family that makes atan(0) ~5e-15 instead of exactly 0 can overshoot
    // pi/2 by a comparable amount here (measured: atan(i64::MAX/4) =
    // 1.5707963267949019, ~5.3e-15 *above* FRAC_PI_2).
    let huge = Fix128::from_int(i64::MAX / 4);
    let atan_huge = huge.atan();
    println!(
        "[math] atan(i64::MAX/4) = {} (expect within 1e-12 of pi/2 = {})",
        atan_huge.to_f64(),
        core::f64::consts::FRAC_PI_2
    );
    assert!(
        close(atan_huge.to_f64(), core::f64::consts::FRAC_PI_2, 1e-12),
        "atan of an extreme magnitude should approach pi/2: got {}",
        atan_huge.to_f64()
    );

    // =======================================================================
    // dot_simd / length_squared_simd -- compile on every platform; dispatch
    // to SSE2 on x86_64+simd, fall back to the scalar path otherwise.
    // =======================================================================

    let a = Vec3Fix::new(
        Fix128::from_int(3),
        Fix128::from_int(-4),
        Fix128::from_int(5),
    );
    let b = Vec3Fix::new(
        Fix128::from_int(7),
        Fix128::from_int(2),
        Fix128::from_int(-1),
    );
    let scalar_dot = a.dot(b); // 3*7 + (-4)*2 + 5*(-1) = 21 - 8 - 5 = 8
    let simd_dot = a.dot_simd(b);
    println!(
        "[math] dot_simd({a:?}, {b:?}) = {} (scalar dot() = {})",
        simd_dot.to_f64(),
        scalar_dot.to_f64()
    );
    assert_eq!(
        (simd_dot.hi, simd_dot.lo),
        (scalar_dot.hi, scalar_dot.lo),
        "dot_simd must be bit-exact with dot()"
    );
    assert_eq!(scalar_dot, Fix128::from_int(8));

    let scalar_len_sq = a.length_squared(); // 9 + 16 + 25 = 50
    let simd_len_sq = a.length_squared_simd();
    println!(
        "[math] length_squared_simd({a:?}) = {} (scalar length_squared() = {})",
        simd_len_sq.to_f64(),
        scalar_len_sq.to_f64()
    );
    assert_eq!(
        (simd_len_sq.hi, simd_len_sq.lo),
        (scalar_len_sq.hi, scalar_len_sq.lo),
        "length_squared_simd must be bit-exact with length_squared()"
    );
    assert_eq!(scalar_len_sq, Fix128::from_int(50));

    // Degenerate: zero vector.
    let zero_dot_simd = Vec3Fix::ZERO.dot_simd(b);
    assert!(
        zero_dot_simd.is_zero(),
        "dot_simd with a zero vector must be exactly zero"
    );
    let zero_len_sq_simd = Vec3Fix::ZERO.length_squared_simd();
    assert!(
        zero_len_sq_simd.is_zero(),
        "length_squared_simd of the zero vector must be exactly zero"
    );
    println!("[math] dot_simd / length_squared_simd: zero vector -> exactly zero");

    // Extreme magnitude: large raw components overflow Fix128's Mul the
    // same way on both paths (wrapping, no saturation) -- dot_simd must
    // reproduce that wraparound bit-for-bit, not diverge from it.
    let extreme = Vec3Fix::new(
        Fix128::from_raw(i64::MAX / 2, 0xFFFF_FFFF_FFFF_FFFF),
        Fix128::from_raw(i64::MIN / 2, 0x1234_5678_9ABC_DEF0),
        Fix128::from_raw(100, 0),
    );
    let scalar_extreme = extreme.dot(extreme);
    let simd_extreme = extreme.dot_simd(extreme);
    println!(
        "[math] dot_simd(extreme, extreme) hi={:#x} lo={:#x} (scalar hi={:#x} lo={:#x})",
        simd_extreme.hi, simd_extreme.lo, scalar_extreme.hi, scalar_extreme.lo
    );
    assert_eq!(
        (simd_extreme.hi, simd_extreme.lo),
        (scalar_extreme.hi, scalar_extreme.lo),
        "dot_simd must reproduce Mul's wraparound bit-for-bit at extreme magnitudes"
    );

    // =======================================================================
    // add_simd / sub_simd, cross_simd, dot_batch_4 -- gated exactly as in
    // src/math.rs: `#[cfg(all(feature = "simd", target_arch = "x86_64"))]`.
    // =======================================================================

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        run_x86_64_simd_gated_items();
    }
    #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
    {
        println!(
            "[math] add_simd / sub_simd / cross_simd / dot_batch_4 skipped: this build is not \
             (target_arch = \"x86_64\" AND feature = \"simd\") -- they do not exist on this \
             platform/feature combination (src/math.rs:359-379, 1208-1239)."
        );
    }

    println!(
        "[math] done: 7 production entry points exercised (atan, add_simd, sub_simd, \
         dot_simd, length_squared_simd, cross_simd, dot_batch_4 -- the last four gated on \
         x86_64+simd, skipped with a message otherwise)"
    );
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
fn run_x86_64_simd_gated_items() {
    // --- add_simd / sub_simd: compared against the `+` / `-` operators ---
    let a = Fix128::from_raw(3, 0xABCD_EF01_2345_6789);
    let b = Fix128::from_raw(-1, 0x1111_2222_3333_4444);

    let scalar_add = a + b;
    // SAFETY: add_simd has no preconditions; SSE2 is part of the x86_64
    // baseline ABI, so `#[target_feature(enable = "sse2")]` is always
    // satisfied here.
    let simd_add = unsafe { a.add_simd(b) };
    println!(
        "[math] add_simd({a:?}, {b:?}) hi={:#x} lo={:#x} (operator + hi={:#x} lo={:#x})",
        simd_add.hi, simd_add.lo, scalar_add.hi, scalar_add.lo
    );
    assert_eq!(
        (simd_add.hi, simd_add.lo),
        (scalar_add.hi, scalar_add.lo),
        "add_simd must be bit-exact with the `+` operator"
    );

    let scalar_sub = a - b;
    // SAFETY: as above.
    let simd_sub = unsafe { a.sub_simd(b) };
    println!(
        "[math] sub_simd({a:?}, {b:?}) hi={:#x} lo={:#x} (operator - hi={:#x} lo={:#x})",
        simd_sub.hi, simd_sub.lo, scalar_sub.hi, scalar_sub.lo
    );
    assert_eq!(
        (simd_sub.hi, simd_sub.lo),
        (scalar_sub.hi, scalar_sub.lo),
        "sub_simd must be bit-exact with the `-` operator"
    );

    // Degenerate / extreme magnitude: i64 wraparound at the hi boundary.
    // SAFETY: as above.
    let wrapped_add = unsafe { Fix128::from_raw(i64::MAX, 0).add_simd(Fix128::from_raw(1, 0)) };
    let wrapped_add_scalar = Fix128::from_raw(i64::MAX, 0) + Fix128::from_raw(1, 0);
    println!(
        "[math] add_simd(i64::MAX, 1) wraps to hi={:#x} (operator + hi={:#x})",
        wrapped_add.hi, wrapped_add_scalar.hi
    );
    assert_eq!(
        (wrapped_add.hi, wrapped_add.lo),
        (wrapped_add_scalar.hi, wrapped_add_scalar.lo),
        "add_simd must wrap at the i64 boundary exactly like the `+` operator (no panic, no saturation)"
    );
    // SAFETY: as above.
    let wrapped_sub = unsafe { Fix128::from_raw(i64::MIN, 0).sub_simd(Fix128::from_raw(1, 0)) };
    let wrapped_sub_scalar = Fix128::from_raw(i64::MIN, 0) - Fix128::from_raw(1, 0);
    println!(
        "[math] sub_simd(i64::MIN, 1) wraps to hi={:#x} (operator - hi={:#x})",
        wrapped_sub.hi, wrapped_sub_scalar.hi
    );
    assert_eq!(
        (wrapped_sub.hi, wrapped_sub.lo),
        (wrapped_sub_scalar.hi, wrapped_sub_scalar.lo),
        "sub_simd must wrap at the i64 boundary exactly like the `-` operator"
    );

    // --- cross_simd: compared against Vec3Fix::cross + closed-form basis ---
    let x = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    let y = Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO);
    let z = Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE);
    assert_eq!(
        x.cross_simd(y),
        z,
        "x cross y must equal z (right-handed basis)"
    );
    assert_eq!(
        y.cross_simd(z),
        x,
        "y cross z must equal x (right-handed basis)"
    );
    assert_eq!(
        z.cross_simd(x),
        y,
        "z cross x must equal y (right-handed basis)"
    );
    println!("[math] cross_simd: right-handed basis identities x*y=z, y*z=x, z*x=y hold exactly");

    let p = Vec3Fix::new(
        Fix128::from_int(2),
        Fix128::from_int(-3),
        Fix128::from_int(5),
    );
    let q = Vec3Fix::new(
        Fix128::from_int(-7),
        Fix128::from_int(1),
        Fix128::from_int(4),
    );
    let scalar_cross = p.cross(q);
    let simd_cross = p.cross_simd(q);
    println!(
        "[math] cross_simd({p:?}, {q:?}) = {simd_cross:?} (scalar cross() = {scalar_cross:?})"
    );
    assert_eq!(
        simd_cross, scalar_cross,
        "cross_simd must be bit-exact with cross()"
    );

    // Degenerate: parallel and anti-parallel vectors cross to exactly zero.
    let v = Vec3Fix::new(
        Fix128::from_int(3),
        Fix128::from_int(-5),
        Fix128::from_int(7),
    );
    let parallel = v.scale(Fix128::from_int(2));
    assert_eq!(
        v.cross_simd(parallel),
        Vec3Fix::ZERO,
        "cross of parallel vectors must be exactly zero"
    );
    let anti_parallel = v.scale(Fix128::from_int(-1));
    assert_eq!(
        v.cross_simd(anti_parallel),
        Vec3Fix::ZERO,
        "cross of anti-parallel vectors must be exactly zero"
    );
    assert_eq!(
        v.cross_simd(v),
        Vec3Fix::ZERO,
        "cross of a vector with itself must be exactly zero"
    );
    println!("[math] cross_simd: parallel / anti-parallel / self cross to exactly zero");

    // --- dot_batch_4: compared against four independent dot() calls ---
    let batch_a = [x, y, z, p];
    let batch_b = [y, z, x, q];
    let got = Vec3Fix::dot_batch_4(batch_a, batch_b);
    let want = [
        batch_a[0].dot(batch_b[0]),
        batch_a[1].dot(batch_b[1]),
        batch_a[2].dot(batch_b[2]),
        batch_a[3].dot(batch_b[3]),
    ];
    println!(
        "[math] dot_batch_4 = {:?} (four independent dot() calls = {:?})",
        got.map(Fix128::to_f64),
        want.map(Fix128::to_f64)
    );
    assert_eq!(
        got, want,
        "dot_batch_4 must match four independent dot() calls exactly"
    );

    // Degenerate: a batch that includes a zero-vector pair and an
    // extreme-magnitude pair.
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
    let batch_a2 = [Vec3Fix::ZERO, extreme_a, x, p];
    let batch_b2 = [q, extreme_b, x, q];
    let got2 = Vec3Fix::dot_batch_4(batch_a2, batch_b2);
    let want2 = [
        batch_a2[0].dot(batch_b2[0]),
        batch_a2[1].dot(batch_b2[1]),
        batch_a2[2].dot(batch_b2[2]),
        batch_a2[3].dot(batch_b2[3]),
    ];
    assert_eq!(
        got2, want2,
        "dot_batch_4 must match independent dot() calls including a zero vector and extreme magnitudes"
    );
    println!("[math] dot_batch_4: zero-vector and extreme-magnitude lanes match independent dot() exactly");
}
