//! Audit oracles for math: `Mat3Fix::polar_rotation` returns
//! `PolarError::SingularStep { steps: 0 }` when the input passes the
//! determinant checks but the power-of-two normalised iterate has a
//! determinant below the `2^-64` resolution (AUD-C-S1W5-010).
//!
//! `F = diag(2^20, 2^-30, 2^-30)` has `det F = 2^-40 > 0`, exact in `Fix128`,
//! so neither `Inverted` nor `Degenerate` (floor 0) applies. The largest
//! component `2^20` is already a power of two, so the first iterate is
//! `diag(1, 2^-50, 2^-50)`, whose determinant `2^-100` truncates to zero: the
//! inverse at step 0 does not exist. Without the normalisation the same
//! matrix is invertible, which is what separates this refusal from
//! `Degenerate`.

use alice_physics::math::{Fix128, Mat3Fix, PolarError};

fn pow2(e: i32) -> Fix128 {
    if e >= 0 {
        Fix128::from_int(1i64 << e)
    } else {
        Fix128::from_raw(0, 1u64 << (64 + e))
    }
}

#[test]
fn normalised_iterate_below_resolution_is_a_singular_step_at_step_zero() {
    let f = Mat3Fix::diagonal(pow2(20), pow2(-30), pow2(-30));
    // the input itself: det = 2^-40 exactly, invertible
    assert_eq!(f.determinant(), pow2(-40));
    assert!(f.inverse().is_some(), "the input is invertible");
    // the normalised first iterate: det 2^-100 is below 2^-64
    let iterate = Mat3Fix::diagonal(Fix128::ONE, pow2(-50), pow2(-50));
    assert!(iterate.determinant().is_zero());
    assert!(iterate.inverse().is_none());
    for budget in [1u32, 16, 64] {
        assert_eq!(
            f.polar_rotation(Fix128::ZERO, budget),
            Err(PolarError::SingularStep { steps: 0 }),
            "budget {budget}"
        );
    }
}

#[test]
fn singular_step_needs_the_large_component_and_is_not_a_floor_refusal() {
    // The same small singular values without the large component are not
    // rescaled (largest component 1 rounds up to 1) and converge.
    let small = Mat3Fix::diagonal(Fix128::ONE, pow2(-30), pow2(-30));
    assert!(small.polar_rotation(Fix128::ZERO, 128).is_ok());
    // a floor at or above det F turns the same input into Degenerate, so the
    // singular step is only reported when the caller's floor admits the input
    let f = Mat3Fix::diagonal(pow2(20), pow2(-30), pow2(-30));
    assert_eq!(f.polar_rotation(pow2(-40), 16), Err(PolarError::Degenerate));
    assert_eq!(
        f.polar_rotation(pow2(-41), 16),
        Err(PolarError::SingularStep { steps: 0 })
    );
}

#[test]
fn singular_step_is_reported_for_every_axis_ordering_of_the_large_value() {
    // Permuting which axis carries 2^20 keeps det and the scaled iterate's
    // determinant; the refusal must not depend on the axis.
    let (big, s) = (pow2(20), pow2(-30));
    for f in [
        Mat3Fix::diagonal(big, s, s),
        Mat3Fix::diagonal(s, big, s),
        Mat3Fix::diagonal(s, s, big),
    ] {
        assert_eq!(f.determinant(), pow2(-40));
        assert_eq!(
            f.polar_rotation(Fix128::ZERO, 16),
            Err(PolarError::SingularStep { steps: 0 })
        );
    }
}
