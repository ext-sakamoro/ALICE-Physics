//! Range-checked vector products: production entry point for
//! `Vec3Fix::{checked_dot, checked_length_squared, checked_length,
//! checked_normalize, checked_length_scaled, try_normalize_scaled}` and
//! `Fix128::checked_add`.
//!
//! `Fix128` is Q64.64, so a product wraps once its magnitude reaches 2⁶³.
//! For a vector that happens at a length of about 2³¹·⁵ ≈ 3.04e9: `dot` and
//! `length_squared` wrap, `length` returns 0 and `normalize` returns ZERO.
//! The checked versions return `None` there instead, and the scaled versions
//! still return the right length and direction.
//!
//! Every value below is exact, so each line is compared with `assert_eq!`.

use alice_physics::math::{Fix128, Vec3Fix};

fn main() {
    // Inside the square range the checked versions agree bit for bit with the
    // unchecked ones.
    let v = Vec3Fix::from_int(3, 4, 12);
    assert_eq!(v.checked_dot(v), Some(v.dot(v)));
    assert_eq!(v.checked_length_squared(), Some(Fix128::from_int(169)));
    assert_eq!(v.checked_length(), Some(v.length()));
    assert_eq!(v.checked_length(), Some(Fix128::from_int(13)));
    assert_eq!(v.checked_normalize(), v.try_normalize());
    assert_eq!(v.checked_length_scaled(), Some(v.length()));
    assert_eq!(v.try_normalize_scaled(), v.try_normalize());
    println!("[vector_range_checks] |(3,4,12)| = 13, checked == unchecked");

    // 2³² on one axis: the square is 2⁶⁴ and does not fit.
    let far = Vec3Fix::from_int(1 << 32, 0, 0);
    assert_eq!(far.checked_dot(far), None);
    assert_eq!(far.checked_length_squared(), None);
    assert_eq!(far.checked_length(), None);
    assert_eq!(far.checked_normalize(), None);
    // The unchecked length wraps to 0; the scaled one is exact.
    assert_eq!(far.length(), Fix128::ZERO);
    assert_eq!(far.checked_length_scaled(), Some(Fix128::from_int(1 << 32)));
    assert_eq!(far.try_normalize_scaled(), Some(Vec3Fix::from_int(1, 0, 0)));
    println!(
        "[vector_range_checks] |(2^32,0,0)|: length {:?}, checked {:?}, scaled {:?}",
        far.length().hi,
        far.checked_length().map(|l| l.hi),
        far.checked_length_scaled().map(|l| l.hi)
    );

    // Sums: the largest integer part plus one does not fit.
    let max = Fix128::from_int(i64::MAX);
    assert_eq!(max.checked_add(Fix128::ZERO), Some(max));
    assert_eq!(max.checked_add(Fix128::ONE), None);
    println!("[vector_range_checks] 2^63 - 1 + 1 refused");
}
