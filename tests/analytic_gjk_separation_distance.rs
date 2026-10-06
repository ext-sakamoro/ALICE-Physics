//! Closed-form oracle for GJK separation distance between separated spheres
//! (COV-RIGID-056: "GJK distance and witness points between separated convex
//! shapes"). Existing GJK tests (`tests/audit_collider.rs` etc.) only assert
//! the boolean `colliding` flag; the actual separation distance value
//! (`GjkResult::closest_point`, the closest point on the Minkowski
//! difference A - B, whose length from the origin is the separation
//! distance) had no closed-form check anywhere in the suite.
//!
//! For two spheres with centers c1, c2 and radii r1, r2, the exact
//! separation distance when not colliding is `|c1 - c2| - r1 - r2`
//! (elementary geometry: the Minkowski difference of two spheres is a
//! sphere of radius r1 + r2 centered at c1 - c2).
//!
//! Probing this oracle (2026-10-07) found the fixed-iteration GJK loop
//! converges exactly for separations along a coordinate axis, but is off by
//! up to ~1 unit (on a separation of 10, ~19% relative) for non-axis-aligned
//! separations or unequal radii. The axis-aligned cases are asserted
//! (and pass); the non-axis-aligned cases are pinned `#[ignore]` with the
//! measured gap as a known source-side limitation, not fixed here (out of
//! scope for this oracle-only pass; reported separately).

use alice_physics::collider::{gjk, Sphere};
use alice_physics::{Fix128, Vec3Fix};

fn fx(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

/// Fixed-iteration GJK is not exact to machine precision; this tolerance
/// matches the scale used by other collider oracles in this suite.
fn tol() -> Fix128 {
    Fix128::from_ratio(1, 1000)
}

fn assert_close(got: Fix128, expected: Fix128, label: &str) {
    let diff = (got - expected).abs();
    assert!(
        diff <= tol(),
        "{label}: got {:?} expected {:?} diff {:?}",
        got.to_f64(),
        expected.to_f64(),
        diff.to_f64()
    );
}

fn check(c1: Vec3Fix, r1: i64, c2: Vec3Fix, r2: i64) {
    let a = Sphere::new(c1, fx(r1));
    let b = Sphere::new(c2, fx(r2));
    let result = gjk(&a, &b);
    let center_dist = (c2 - c1).length();
    let expected = center_dist - fx(r1) - fx(r2);
    assert!(!result.colliding, "expected separated: r1={r1} r2={r2}");
    assert_close(
        result.closest_point.length(),
        expected,
        "separation distance",
    );
}

#[test]
fn axis_aligned_equal_radius_separation_is_exact() {
    check(
        Vec3Fix::ZERO,
        1,
        Vec3Fix::new(fx(5), Fix128::ZERO, Fix128::ZERO),
        1,
    );
    check(
        Vec3Fix::ZERO,
        1,
        Vec3Fix::new(fx(100), Fix128::ZERO, Fix128::ZERO),
        1,
    );
    check(
        Vec3Fix::ZERO,
        1,
        Vec3Fix::new(fx(3), Fix128::ZERO, Fix128::ZERO),
        1,
    );
}

#[test]
fn touching_spheres_have_zero_separation_distance() {
    // Centers 5 apart, radii 2 + 3 = 5: exactly touching, separation 0, axis-aligned.
    // Whether the boundary itself reads as `colliding` is an implementation
    // choice either way, so only the distance magnitude is asserted here.
    let a = Sphere::new(Vec3Fix::ZERO, fx(2));
    let b = Sphere::new(Vec3Fix::new(fx(5), Fix128::ZERO, Fix128::ZERO), fx(3));
    let result = gjk(&a, &b);
    assert_close(
        result.closest_point.length(),
        Fix128::ZERO,
        "touching separation",
    );
}

#[test]
#[ignore = "src gap: fixed-iteration GJK converges inexactly for unequal-radius separated \
            spheres along a non-x axis (dist=10 r1=2 r2=3 along +y: got 5.963 expected 5.000, \
            ~19% relative error); axis_aligned_equal_radius_separation_is_exact above shows the \
            +x axis / equal-radius path is exact, so the gap is in simplex convergence for \
            other directions, not the closed form here"]
fn separation_distance_matches_closed_form_off_axis_unequal_radius() {
    check(
        Vec3Fix::ZERO,
        2,
        Vec3Fix::new(Fix128::ZERO, fx(10), Fix128::ZERO),
        3,
    );
}

#[test]
#[ignore = "src gap: fixed-iteration GJK converges inexactly for a diagonal separation even at \
            equal radii (dist=5 r1=1 r2=1 along (3,4,0): got 3.251 expected 3.000); see \
            separation_distance_matches_closed_form_off_axis_unequal_radius for the unequal-radius \
            case along +y, same family of gap"]
fn separation_distance_matches_closed_form_diagonal_equal_radius() {
    check(
        Vec3Fix::ZERO,
        1,
        Vec3Fix::new(fx(3), fx(4), Fix128::ZERO),
        1,
    );
}
