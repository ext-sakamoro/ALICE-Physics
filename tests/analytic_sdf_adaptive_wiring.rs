//! Oracles for `sdf_adaptive::AdaptiveSdfEvaluator`: LOD level transitions
//! (observed through the saved/total counters and the returned values),
//! cache invalidation, and the world-space normal of a rotated collider.
//!
//! Field: unit sphere at the origin (`f = |p| - 1`, exact distance), so the
//! distance at `(x, 0, 0)` is `x - 1` and the normal is `+X`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_adaptive::{AdaptiveConfig, AdaptiveSdfEvaluator};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

fn sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 1.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt();
            (x / l, y / l, z / l)
        },
    )
}

fn at(x: f32) -> Vec3Fix {
    Vec3Fix::from_f32(x, 0.0, 0.0)
}

fn sdf_at(origin_x: i64) -> SdfCollider {
    SdfCollider::new_static(
        Box::new(sphere()),
        Vec3Fix::from_int(origin_x, 0, 0),
        QuatFix::IDENTITY,
    )
}

fn ev(n: usize) -> AdaptiveSdfEvaluator {
    AdaptiveSdfEvaluator::new(n, AdaptiveConfig::default())
}

#[test]
fn levels_follow_the_cached_distance() {
    let sdf = sdf_at(0);
    let mut e = ev(3);
    e.begin_frame();
    // First sight of every body: full evaluation (nothing saved).
    let (d_far, _) = e.evaluate(0, at(15.0), &sdf); // 14 > cache_threshold 10
    let (d_mid, _) = e.evaluate(1, at(5.0), &sdf); // 4 in (2, 10]
    let (d_near, _) = e.evaluate(2, at(2.0), &sdf); // 1 < 2
    assert!(
        (d_far - 14.0).abs() < 1e-4 && (d_mid - 4.0).abs() < 1e-4 && (d_near - 1.0).abs() < 1e-4
    );
    assert_eq!(e.stats(), (0, 3));

    // Next frame, no movement: far -> Skip, mid -> Cached (both saved), near -> HighRes (evaluated).
    e.begin_frame();
    assert_eq!(e.stats(), (0, 0), "begin_frame resets the counters");
    for (i, x, want) in [(0, 15.0, 14.0), (1, 5.0, 4.0), (2, 2.0, 1.0)] {
        let (d, n) = e.evaluate(i, at(x), &sdf);
        assert!((d - want).abs() < 1e-4, "body {i}: {d}");
        assert!(
            (n.0 - 1.0).abs() < 1e-3 && n.1.abs() < 1e-3,
            "body {i} normal {n:?}"
        );
    }
    assert_eq!(e.stats(), (2, 3));
}

#[test]
fn cached_value_is_returned_stale_until_the_body_moves_far_enough() {
    let sdf = sdf_at(0);
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, at(5.0), &sdf); // d = 4
    e.begin_frame();
    // Moved 1.9 < 0.5 * 4 = 2: still Cached, returns the old distance 4 (not 5.9 - 1 = 4.9).
    let (d, _) = e.evaluate(0, at(6.9), &sdf);
    assert!((d - 4.0).abs() < 1e-4, "d = {d}");
    assert_eq!(e.stats().0, 1);
    // Moved 2.1 from the cached position (5.0 -> 7.1) >= 2: re-evaluated, exact value.
    e.begin_frame();
    let (d, _) = e.evaluate(0, at(7.1), &sdf);
    assert!((d - 6.1).abs() < 1e-4, "d = {d}");
    assert_eq!(e.stats(), (0, 1));
}

#[test]
fn cache_expires_after_max_age_frames() {
    let sdf = sdf_at(0);
    let mut e = AdaptiveSdfEvaluator::new(
        1,
        AdaptiveConfig {
            cache_max_age: 3,
            ..AdaptiveConfig::default()
        },
    );
    e.begin_frame();
    let _ = e.evaluate(0, at(5.0), &sdf);
    // age 3 is still valid (age > max_age expires): frames 2..=4.
    for _ in 0..3 {
        e.begin_frame();
        let _ = e.evaluate(0, at(5.0), &sdf);
        assert_eq!(e.stats(), (1, 1));
    }
    // The cache refreshed only at frame 1; frame 5 has age 4 > 3 -> re-evaluated.
    e.begin_frame();
    let _ = e.evaluate(0, at(5.0), &sdf);
    assert_eq!(e.stats(), (0, 1));
}

#[test]
fn invalidate_forces_a_fresh_evaluation_for_that_body_only() {
    let near = sdf_at(0);
    let moved = sdf_at(1); // sphere centred at x = 1: distance at (5,0,0) is 3
    let mut e = ev(2);
    e.begin_frame();
    let _ = e.evaluate(0, at(5.0), &near);
    let _ = e.evaluate(1, at(5.0), &near);
    e.begin_frame();
    e.invalidate(0);
    let (d0, _) = e.evaluate(0, at(5.0), &moved);
    let (d1, _) = e.evaluate(1, at(5.0), &moved);
    assert!(
        (d0 - 3.0).abs() < 1e-4,
        "invalidated body re-evaluated: {d0}"
    );
    assert!((d1 - 4.0).abs() < 1e-4, "other body still cached: {d1}");
    assert_eq!(e.stats(), (1, 2));
    // Out-of-range invalidate is a no-op.
    e.invalidate(99);
}

#[test]
fn invalidate_all_forces_every_body_to_re_evaluate() {
    let near = sdf_at(0);
    let moved = sdf_at(1);
    let mut e = ev(3);
    e.begin_frame();
    for i in 0..3 {
        let _ = e.evaluate(i, at(5.0), &near);
    }
    e.begin_frame();
    e.invalidate_all();
    for i in 0..3 {
        let (d, _) = e.evaluate(i, at(5.0), &moved);
        assert!((d - 3.0).abs() < 1e-4, "body {i}: {d}");
    }
    assert_eq!(e.stats(), (0, 3));
    // Zero bodies: no panic.
    let mut empty = ev(0);
    empty.invalidate_all();
    empty.invalidate(0);
}

#[test]
fn unknown_body_index_is_evaluated_exactly_and_never_cached() {
    let sdf = sdf_at(0);
    let mut e = ev(1);
    e.begin_frame();
    let (d, _) = e.evaluate(7, at(5.0), &sdf);
    assert!((d - 4.0).abs() < 1e-4);
    e.begin_frame();
    let (d, _) = e.evaluate(7, at(9.0), &sdf);
    assert!(
        (d - 8.0).abs() < 1e-4,
        "no stale value for an unregistered body: {d}"
    );
    assert_eq!(e.stats(), (0, 1));
    // resize makes the index valid.
    e.resize(8);
    e.begin_frame();
    let _ = e.evaluate(7, at(5.0), &sdf);
    e.begin_frame();
    let _ = e.evaluate(7, at(5.0), &sdf);
    assert_eq!(e.stats(), (1, 1));
}

#[test]
fn high_res_normal_is_the_exact_gradient_direction() {
    let sdf = sdf_at(0);
    let mut e = ev(1);
    e.begin_frame();
    // A point near the surface at (1, 1, 0)/sqrt(2) * 1.5: normal (1,1,0)/sqrt(2).
    let p = Vec3Fix::from_f32(1.06066, 1.06066, 0.0);
    let _ = e.evaluate(0, p, &sdf);
    e.begin_frame();
    let (d, n) = e.evaluate(0, p, &sdf); // distance 0.5 < 2: HighRes
    let s = core::f32::consts::FRAC_1_SQRT_2;
    assert!((d - 0.5).abs() < 2e-3, "d = {d}");
    assert!(
        (n.0 - s).abs() < 2e-3 && (n.1 - s).abs() < 2e-3 && n.2.abs() < 2e-3,
        "n = {n:?}"
    );
    let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
    assert!((len - 1.0).abs() < 1e-3);
}

#[test]
fn normal_of_a_rotated_collider_is_in_world_space() {
    // Plane y = 0 rotated +90 degrees about Z: local +Y is world -X.
    let plane = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    let q = QuatFix::from_axis_angle(
        Vec3Fix::UNIT_Z,
        Fix128::from_f32(core::f32::consts::FRAC_PI_2),
    );
    let sdf = SdfCollider::new_static(Box::new(plane), Vec3Fix::ZERO, q);
    let p = Vec3Fix::from_f32(-3.0, 0.0, 0.0); // 3 m on the free side (world -X)
    let mut e = ev(1);
    e.begin_frame();
    let (d, n) = e.evaluate(0, p, &sdf); // Standard path
    assert!((d - 3.0).abs() < 2e-3, "d = {d}");
    assert!(
        (n.0 + 1.0).abs() < 2e-3 && n.1.abs() < 2e-3 && n.2.abs() < 2e-3,
        "standard n = {n:?}"
    );
    // HighRes path (distance < 2) must agree.
    let p = Vec3Fix::from_f32(-1.0, 0.0, 0.0);
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, p, &sdf);
    e.begin_frame();
    let (d, n) = e.evaluate(0, p, &sdf);
    assert!((d - 1.0).abs() < 2e-3, "d = {d}");
    assert!(
        (n.0 + 1.0).abs() < 2e-3 && n.1.abs() < 2e-3 && n.2.abs() < 2e-3,
        "high-res n = {n:?}"
    );
}

#[test]
fn scaled_collider_distance_is_in_world_metres() {
    let sdf = SdfCollider::new_static(Box::new(sphere()), Vec3Fix::ZERO, QuatFix::IDENTITY)
        .with_scale(Fix128::from_int(2)); // world radius 2
    let mut e = ev(1);
    e.begin_frame();
    let (d, _) = e.evaluate(0, at(5.0), &sdf);
    assert!((d - 3.0).abs() < 1e-3, "d = {d}");
}

/// Unit-sphere distance with a deliberately wrong reported normal `+Z`: the
/// HighRes path re-derives the normal from the distance (central differences),
/// the Standard path returns what the field reports, so the normal tells the
/// two levels apart.
fn bogus_normal_sphere(radius: f32) -> SdfCollider {
    let f = ClosureSdf::new(
        move |x, y, z| (x * x + y * y + z * z).sqrt() - radius,
        |_x, _y, _z| (0.0, 0.0, 1.0),
    );
    SdfCollider::new_static(Box::new(f), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn is_reported_normal(n: (f32, f32, f32)) -> bool {
    n.2 > 0.999
}

#[test]
fn high_res_applies_below_the_threshold_only_and_uses_fine_differences() {
    let sdf = bogus_normal_sphere(1.0);
    // Cached distance exactly 2.0 (dyadic): `distance < high_res_threshold` is false -> Standard.
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, at(3.0), &sdf);
    e.begin_frame();
    let (d, n) = e.evaluate(0, at(3.0), &sdf);
    assert_eq!(d, 2.0);
    assert!(
        is_reported_normal(n),
        "Standard level returns the field's normal: {n:?}"
    );
    // Distance 1.75 < 2: HighRes, normal from differences = +X.
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, at(2.75), &sdf);
    e.begin_frame();
    let (_, n) = e.evaluate(0, at(2.75), &sdf);
    assert!(
        (n.0 - 1.0).abs() < 1e-3 && n.1.abs() < 1e-3 && n.2.abs() < 1e-3,
        "{n:?}"
    );

    // Step size of the differences: a small sphere (radius 0.3) probed at (0.6, 0.45, 0),
    // radial direction (0.8, 0.6, 0); a coarse step (0.5) would skew it by ~1e-1.
    let small = bogus_normal_sphere(0.3);
    let p = Vec3Fix::from_f32(0.6, 0.45, 0.0);
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, p, &small);
    e.begin_frame();
    let (d, n) = e.evaluate(0, p, &small);
    assert!((d - 0.45).abs() < 1e-3);
    assert!(
        (n.0 - 0.8).abs() < 2e-3 && (n.1 - 0.6).abs() < 2e-3 && n.2.abs() < 2e-3,
        "{n:?}"
    );
}

#[test]
fn far_skip_requires_small_movement_and_cache_requires_less_than_half_the_distance() {
    let sdf = sdf_at(0);
    // Distance 14 (> cache_threshold 10): moving 8 >= 0.5 * 14 = 7 -> re-evaluated.
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, at(15.0), &sdf);
    e.begin_frame();
    let (d, _) = e.evaluate(0, at(23.0), &sdf);
    assert!((d - 22.0).abs() < 1e-4, "d = {d}");
    assert_eq!(e.stats(), (0, 1));
    // Moving 6 < 7: still the cached 14.
    let mut e = ev(1);
    e.begin_frame();
    let _ = e.evaluate(0, at(15.0), &sdf);
    e.begin_frame();
    let (d, _) = e.evaluate(0, at(21.0), &sdf);
    assert!((d - 14.0).abs() < 1e-4, "d = {d}");
    assert_eq!(e.stats(), (1, 1));
}
