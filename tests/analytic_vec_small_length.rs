//! The length and direction of a small `Vec3Fix`, two contracts.
//!
//! `normalize_with_length_scaled` keeps the digits. oracle (the Euclidean norm, f64): for a vector of length
//! `1e-15` to `1e-3` along a fixed direction, the length agrees with the f64
//! norm of the same components within `2⁻⁶⁰` and the unit vector within `2⁻⁴⁰`
//! per component (a vector shorter than 1 is scaled up by an exact power of two
//! first). Across the switch at 1 the length moves by no more than the
//! components do, and from 1 up the scaled form equals the plain one bit for
//! bit. What it catches: the length squared taken directly (a length of `1e-9`
//! came out `1.5%` short and one of `1e-11` as zero, so a small rotation vector
//! of the push-out gave a wrong angle and axis).
//!
//! `length` / `normalize` / `try_normalize` / `normalize_with_length` read a
//! vector below resolution as zero, and that is pinned here on purpose: a
//! joint's error or a flow speed at the level of rounding noise must give no
//! direction. Normalising such noise into a unit vector made the hinge joint's
//! error grow from 6 to 96212 units in one iteration
//! (`joint::tests::hinge_joint_aligns_axes_monotonically`). Do not "fix" the
//! plain forms to keep the digits; use the scaled one where a genuinely small
//! vector is meant.

use alice_physics_core::math::{Fix128, Vec3Fix};

fn f64_of(v: Fix128) -> f64 {
    if v.is_negative() {
        -(-v).to_f64()
    } else {
        v.to_f64()
    }
}

/// `2^e`, exact (built from the bits; the platform `powi` is not allowed).
fn p2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// `10^(12 k / 120)` as repeated products of `10^(12 / 120)` (a fixed
/// sequence, the same on every platform; the sweep needs no exact decades).
fn ratio_pow(k: u32) -> f64 {
    let mut m = 1.0;
    for _ in 0..k {
        m *= 1.258_925_411_794_167_3;
    }
    m
}

#[test]
fn small_vectors_have_their_length_and_direction() {
    let dir = [0.3f64, -0.4, 0.866];
    let mut compared = 0;
    for k in 0..=120 {
        let m = 1e-15 * ratio_pow(k);
        let v = Vec3Fix::new(
            Fix128::from_f64(dir[0] * m),
            Fix128::from_f64(dir[1] * m),
            Fix128::from_f64(dir[2] * m),
        );
        let c = [f64_of(v.x), f64_of(v.y), f64_of(v.z)];
        let want = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let len = f64_of(v.normalize_with_length_scaled().1);
        assert!(
            (len - want).abs() <= p2(-60) + want * 1e-15,
            "|v| = {len:e} for {want:e}"
        );
        let u = v.normalize_with_length_scaled().0;
        for (i, ci) in c.iter().enumerate() {
            let ui = f64_of([u.x, u.y, u.z][i]);
            assert!(
                (ui - ci / want).abs() <= p2(-40),
                "component {i} of the direction at {want:e}: {ui} vs {}",
                ci / want
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 121);
}

#[test]
fn the_length_is_continuous_at_the_switch() {
    let edge = Fix128::ONE;
    let ulp = Fix128::from_raw(0, 1);
    let below = Vec3Fix::new(edge - ulp, Fix128::ZERO, Fix128::ZERO)
        .normalize_with_length_scaled()
        .1;
    let at = Vec3Fix::new(edge, Fix128::ZERO, Fix128::ZERO)
        .normalize_with_length_scaled()
        .1;
    let step = f64_of(at - below);
    assert!(step >= 0.0 && step <= 4.0 * p2(-64), "steps by {step:e}");
}

#[test]
fn from_one_up_the_scaled_forms_are_the_plain_ones() {
    let mut compared = 0;
    for v in [
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
        Vec3Fix::new(
            Fix128::from_f64(0.3),
            Fix128::from_f64(-1.7),
            Fix128::from_f64(2.5),
        ),
        Vec3Fix::new(
            Fix128::from_int(-40),
            Fix128::from_f64(1e-12),
            Fix128::from_int(3),
        ),
        Vec3Fix::new(
            Fix128::from_int(12345),
            Fix128::from_int(-678),
            Fix128::from_f64(0.5),
        ),
    ] {
        assert_eq!(v.normalize_with_length_scaled(), v.normalize_with_length());
        compared += 1;
    }
    assert_eq!(compared, 4);
}

#[test]
fn the_plain_forms_read_a_vector_below_resolution_as_zero() {
    // |v|² = 3·2⁻⁸⁰ is below 2⁻⁶⁴: no length, no direction.
    let e = Fix128::from_raw(0, 1 << 24); // 2⁻⁴⁰
    let mut compared = 0;
    for v in [
        Vec3Fix::new(e, e, e),
        Vec3Fix::new(-e, Fix128::ZERO, e),
        Vec3Fix::new(Fix128::from_raw(0, 1), Fix128::ZERO, Fix128::ZERO),
    ] {
        assert_eq!(v.length(), Fix128::ZERO, "{v:?}");
        assert_eq!(v.normalize(), Vec3Fix::ZERO, "{v:?}");
        assert_eq!(v.try_normalize(), None, "{v:?}");
        assert_eq!(
            v.normalize_with_length(),
            (Vec3Fix::ZERO, Fix128::ZERO),
            "{v:?}"
        );
        // The scaled form sees it.
        assert!(!v.normalize_with_length_scaled().1.is_zero(), "{v:?}");
        assert_ne!(v.normalize_with_length_scaled().0, Vec3Fix::ZERO, "{v:?}");
        compared += 1;
    }
    assert_eq!(compared, 3);
}
