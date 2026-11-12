//! `Vec3Fix::length`, `normalize` and `normalize_with_length` of a small
//! vector.
//!
//! oracle (the Euclidean norm, f64): for a vector of length `1e-15` to `1e-3`
//! along a fixed direction, the length agrees with the f64 norm of the same
//! components and the unit vector has length 1 and the same direction, to the
//! resolution: a vector whose largest component is below 1 is scaled up by an
//! exact power of two before its length is taken, so the length is within
//! `2⁻⁶⁰` absolute and the unit vector within `2⁻⁴⁰` per component. Across the
//! switch at 1 the length moves by no more than the components do.
//!
//! What it catches: the length squared taken directly (it falls to a few
//! `2⁻⁶⁴`: a length of `1e-9` came out `1.5%` short and one of `1e-11` as zero,
//! so a small rotation vector gave a wrong angle and axis).

use alice_physics_core::math::{Fix128, Vec3Fix};

fn f64_of(v: Fix128) -> f64 {
    if v.is_negative() {
        -(-v).to_f64()
    } else {
        v.to_f64()
    }
}

#[test]
fn small_vectors_have_their_length_and_direction() {
    let dir = [0.3f64, -0.4, 0.866];
    let mut compared = 0;
    for k in 0..=120 {
        let m = 10f64.powf(-15.0 + 12.0 * f64::from(k) / 120.0);
        let v = Vec3Fix::new(
            Fix128::from_f64(dir[0] * m),
            Fix128::from_f64(dir[1] * m),
            Fix128::from_f64(dir[2] * m),
        );
        let c = [f64_of(v.x), f64_of(v.y), f64_of(v.z)];
        let want = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let len = f64_of(v.length());
        assert!(
            (len - want).abs() <= 2f64.powi(-60) + want * 1e-15,
            "|v| = {len:e} for {want:e}"
        );
        let (u, len2) = v.normalize_with_length();
        assert_eq!(len2, v.length(), "the same length both ways at {want:e}");
        assert_eq!(u, v.normalize(), "the same direction both ways at {want:e}");
        for (i, ci) in c.iter().enumerate() {
            let ui = f64_of([u.x, u.y, u.z][i]);
            assert!(
                (ui - ci / want).abs() <= 2f64.powi(-40),
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
    let below = Vec3Fix::new(edge - ulp, Fix128::ZERO, Fix128::ZERO).length();
    let at = Vec3Fix::new(edge, Fix128::ZERO, Fix128::ZERO).length();
    let step = f64_of(at - below);
    assert!(
        step >= 0.0 && step <= 4.0 * 2f64.powi(-64),
        "steps by {step:e}"
    );
}
