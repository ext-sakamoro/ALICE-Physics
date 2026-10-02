//! Closed-form oracle for `sdf_destruction`'s public surface.
//!
//! # Scene
//!
//! The base field is the flat ground `distance(x, y, z) = y` (value `0` on
//! the plane `y = 0`, negative below it, positive above). Every crater in
//! this file is placed at a distinct `x` coordinate on that plane so
//! craters never overlap each other, and every closed form below is derived
//! by hand from `DestructibleSdf`'s own documented CSG rule
//! (`new_sdf = max(original, -destruction_shape)`), not by calling the
//! functions under test to produce their own expected value:
//!
//! - **sharp subtraction**, at a crater's own center (local coordinates are
//!   all zero there): `evaluate() = 0 - radius = -radius` for a sphere,
//!   `= -min(half_extents)` for an axis-aligned cube centered on that
//!   radius-equal half-extent (we use `half_extents = (r, r, r)`, so it is
//!   `-r` too), and `= -min(radius, half_height)` for a cylinder. Sharp
//!   subtraction is `dist = max(d_a, -d_shape)`; with `d_a = 0` (ground)
//!   that is exactly `max(0, r) = r`. So every sharp crater in this file
//!   reads back as exactly `r = 1.0` at its own center.
//! - **smooth subtraction**, `smooth_max(d_a, -d_b, k)`: with `d_a = 0`,
//!   `d_b = -r`, `h = clamp(0.5 + 0.5*r/k, 0, 1)`, the result is
//!   `k*h*(1-h) + r*h`. For `r = 1`, `k = 3`: `h = 2/3`,
//!   `result = 3*(2/3)*(1/3) + 1*(2/3) = 2/3 + 2/3 = 4/3`.
//!
//! None of these require a transcendental function (every `.sqrt()` in the
//! implementation is evaluated at `0` on this scene, since local
//! coordinates are zero at a crater's own center).
#![cfg(feature = "std")]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sdf_destruction::{
    destruction_from_explosion, destruction_from_impact, destruction_from_projectile,
    DestructibleSdf, DestructionShape, DestructionType,
};
use std::panic::{self, AssertUnwindSafe};

fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn unit_sphere() -> ClosureSdf {
    ClosureSdf::new(
        |x, y, z| z.mul_add(z, x.mul_add(x, y * y)).sqrt() - 1.0,
        |x, y, z| {
            let len = z.mul_add(z, x.mul_add(x, y * y)).sqrt();
            if len < 1e-10 {
                (0.0, 1.0, 0.0)
            } else {
                (x / len, y / len, z / len)
            }
        },
    )
}

const EPS: f32 = 1e-6;

#[test]
fn apply_destruction_increments_count_and_total_by_exactly_one_per_call() {
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    assert_eq!(dsdf.destruction_count(), 0);
    assert_eq!(dsdf.total_destruction_count(), 0);

    let events: Vec<DestructionShape> = vec![
        DestructionShape::sphere(Vec3Fix::from_f32(0.0, 0.0, 0.0), 1.0),
        DestructionShape::cube(Vec3Fix::from_f32(10.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        DestructionShape::cylinder(Vec3Fix::from_f32(20.0, 0.0, 0.0), 1.0, 1.0),
        destruction_from_explosion(Vec3Fix::from_f32(30.0, 0.0, 0.0), 1.0, 0.0),
        destruction_from_impact(
            &Contact {
                depth: Fix128::from_f32(0.1),
                normal: Vec3Fix::UNIT_Y,
                point_a: Vec3Fix::from_f32(40.0, 0.0, 0.0),
                point_b: Vec3Fix::from_f32(40.0, 0.0, 0.0),
            },
            Fix128::from_f32(10.0),
            0.05,
            0.2,
            1.0,
        ),
        destruction_from_projectile(Vec3Fix::from_f32(50.0, 0.0, 0.0), Vec3Fix::UNIT_Y, 1.0, 2.0),
    ];

    for (i, shape) in events.into_iter().enumerate() {
        dsdf.apply_destruction(shape);
        assert_eq!(
            dsdf.destruction_count(),
            i + 1,
            "destruction_count after event {i}"
        );
        assert_eq!(
            dsdf.total_destruction_count(),
            i + 1,
            "total_destruction_count after event {i}"
        );
    }
}

#[test]
fn sphere_cube_cylinder_craters_all_read_back_as_their_radius_at_center() {
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    dsdf.apply_destruction(DestructionShape::sphere(
        Vec3Fix::from_f32(0.0, 0.0, 0.0),
        1.0,
    ));
    dsdf.apply_destruction(DestructionShape::cube(
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ));
    dsdf.apply_destruction(DestructionShape::cylinder(
        Vec3Fix::from_f32(20.0, 0.0, 0.0),
        1.0,
        1.0,
    ));

    assert!(
        (dsdf.distance(0.0, 0.0, 0.0) - 1.0).abs() < EPS,
        "sphere crater center"
    );
    assert!(
        (dsdf.distance(10.0, 0.0, 0.0) - 1.0).abs() < EPS,
        "cube crater center"
    );
    assert!(
        (dsdf.distance(20.0, 0.0, 0.0) - 1.0).abs() < EPS,
        "cylinder crater center"
    );

    // Far from every crater, the ground plane is untouched.
    assert!((dsdf.distance(500.0, 0.0, 0.0) - 0.0).abs() < EPS);
}

#[test]
fn with_smoothing_matches_the_hand_derived_smooth_max_and_differs_from_sharp() {
    let mut sharp = DestructibleSdf::new(Box::new(ground()));
    sharp.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0));

    let mut smooth = DestructibleSdf::new(Box::new(ground()));
    smooth.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0).with_smoothing(3.0));

    let d_sharp = sharp.distance(0.0, 0.0, 0.0);
    let d_smooth = smooth.distance(0.0, 0.0, 0.0);

    // Hand derivation (see module doc comment above): r=1, k=3 => h=2/3,
    // result = k*h*(1-h) + r*h = 2/3 + 2/3 = 4/3.
    let expected_smooth = 4.0_f32 / 3.0;

    assert!((d_sharp - 1.0).abs() < EPS, "sharp crater: {d_sharp}");
    assert!(
        (d_smooth - expected_smooth).abs() < 1e-5,
        "smooth crater: got {d_smooth}, expected {expected_smooth}"
    );
    assert!(
        (d_smooth - d_sharp).abs() > 0.1,
        "with_smoothing must produce an observably different surface: sharp={d_sharp} smooth={d_smooth}"
    );
}

#[test]
fn destruction_from_explosion_builds_a_sphere_with_the_requested_smoothing() {
    let shape = destruction_from_explosion(Vec3Fix::from_f32(1.0, 2.0, 3.0), 5.0, 0.25);
    assert_eq!(shape.center, Vec3Fix::from_f32(1.0, 2.0, 3.0));
    assert!((shape.smooth_factor - 0.25).abs() < EPS);
    match shape.shape {
        DestructionType::Sphere { radius } => assert!((radius - 5.0).abs() < EPS),
        other => panic!("expected Sphere, got {other:?}"),
    }

    // smooth = 0.0 must still produce a usable (sharp-path) shape.
    let sharp_shape = destruction_from_explosion(Vec3Fix::ZERO, 2.0, 0.0);
    assert!((sharp_shape.smooth_factor - 0.0).abs() < EPS);
}

#[test]
fn destruction_from_impact_clamps_abs_speed_times_scale_into_the_radius_range() {
    let contact = |px: f32| Contact {
        depth: Fix128::from_f32(0.1),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::from_f32(px, 0.0, 0.0),
        point_b: Vec3Fix::from_f32(px, 0.0, 0.0),
    };

    // Within range: 10 * 0.05 = 0.5, inside [0.2, 1.0].
    let in_range = destruction_from_impact(&contact(1.0), Fix128::from_f32(10.0), 0.05, 0.2, 1.0);
    match in_range.shape {
        DestructionType::Sphere { radius } => assert!((radius - 0.5).abs() < EPS, "{radius}"),
        other => panic!("expected Sphere, got {other:?}"),
    }
    assert_eq!(in_range.center, Vec3Fix::from_f32(1.0, 0.0, 0.0));

    // Negative velocity must use its magnitude (abs), same result as +10.
    let negative = destruction_from_impact(&contact(2.0), Fix128::from_f32(-10.0), 0.05, 0.2, 1.0);
    match negative.shape {
        DestructionType::Sphere { radius } => assert!((radius - 0.5).abs() < EPS, "{radius}"),
        other => panic!("expected Sphere, got {other:?}"),
    }

    // Huge velocity clamps to max_radius.
    let huge = destruction_from_impact(&contact(3.0), Fix128::from_f32(1000.0), 0.05, 0.2, 1.0);
    match huge.shape {
        DestructionType::Sphere { radius } => assert!((radius - 1.0).abs() < EPS, "{radius}"),
        other => panic!("expected Sphere, got {other:?}"),
    }

    // Zero velocity clamps to min_radius.
    let zero = destruction_from_impact(&contact(4.0), Fix128::from_f32(0.0), 0.05, 0.2, 1.0);
    match zero.shape {
        DestructionType::Sphere { radius } => assert!((radius - 0.2).abs() < EPS, "{radius}"),
        other => panic!("expected Sphere, got {other:?}"),
    }
}

#[test]
fn destruction_from_projectile_closed_form_center_and_rotation() {
    // direction = +Y: dot(up, up) = 1 > 0.999 => identity rotation.
    let up =
        destruction_from_projectile(Vec3Fix::from_f32(0.0, 0.0, 0.0), Vec3Fix::UNIT_Y, 1.0, 2.0);
    assert_eq!(up.rotation, QuatFix::IDENTITY);
    assert_eq!(up.center, Vec3Fix::from_f32(0.0, 1.0, 0.0)); // entry + dir * depth/2
    match up.shape {
        DestructionType::Cylinder {
            radius,
            half_height,
        } => {
            assert!((radius - 1.0).abs() < EPS);
            assert!((half_height - 1.0).abs() < EPS); // depth/2
        }
        other => panic!("expected Cylinder, got {other:?}"),
    }

    // direction = -Y: dot(up, -up) = -1 < -0.999 => the documented 180-degree
    // special case, QuatFix::new(ONE, ZERO, ZERO, ZERO).
    let down =
        destruction_from_projectile(Vec3Fix::from_f32(0.0, 0.0, 0.0), -Vec3Fix::UNIT_Y, 1.0, 2.0);
    assert_eq!(
        down.rotation,
        QuatFix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO, Fix128::ZERO)
    );

    // direction = zero vector: degenerate input, documented early-return to
    // IDENTITY; the center must still be exactly entry_point regardless of
    // bore_depth, since `center = entry + 0 * (depth/2) = entry`.
    let degenerate = panic::catch_unwind(AssertUnwindSafe(|| {
        destruction_from_projectile(Vec3Fix::from_f32(7.0, 7.0, 7.0), Vec3Fix::ZERO, 1.0, 100.0)
    }));
    let degenerate = degenerate.expect("zero-length direction must not panic");
    assert_eq!(degenerate.rotation, QuatFix::IDENTITY);
    assert_eq!(degenerate.center, Vec3Fix::from_f32(7.0, 7.0, 7.0));

    // direction = +X (orthogonal to up): invariant check independent of the
    // half-angle arithmetic — the whole point of "rotation that aligns the
    // cylinder's Y-axis with `direction`" is that rotating UNIT_Y by the
    // returned quaternion must give back (approximately) the normalized
    // direction.
    let side = destruction_from_projectile(
        Vec3Fix::from_f32(0.0, 0.0, 0.0),
        Vec3Fix::from_f32(3.0, 0.0, 0.0),
        1.0,
        2.0,
    );
    let rotated = side.rotation.rotate_vec(Vec3Fix::UNIT_Y);
    let (rx, ry, rz) = rotated.to_f32();
    assert!(
        (rx - 1.0).abs() < 1e-4,
        "rotate_vec(UNIT_Y) should align with +X: ({rx},{ry},{rz})"
    );
    assert!(ry.abs() < 1e-4);
    assert!(rz.abs() < 1e-4);
}

#[test]
fn optimize_is_a_no_op_below_the_cap_and_on_an_undamaged_volume() {
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    for i in 0..6 {
        dsdf.apply_destruction(DestructionShape::sphere(
            Vec3Fix::from_f32(i as f32 * 10.0, 0.0, 0.0),
            1.0,
        ));
    }
    dsdf.optimize();
    assert_eq!(dsdf.destruction_count(), 6);
    assert_eq!(dsdf.total_destruction_count(), 6);
    for i in 0..6 {
        assert!((dsdf.distance(i as f32 * 10.0, 0.0, 0.0) - 1.0).abs() < EPS);
    }

    let mut empty = DestructibleSdf::new(Box::new(ground()));
    let before = empty.distance(1.0, 2.0, 3.0);
    empty.optimize();
    assert_eq!(empty.destruction_count(), 0);
    assert!((empty.distance(1.0, 2.0, 3.0) - before).abs() < EPS);
}

#[test]
fn optimize_is_a_no_op_in_the_mid_range_strictly_below_the_cap() {
    // Dedicated mid-range probe (17..=31 craters): the cap check and the
    // drain-count formula are two separate expressions
    // (`len() <= 32` / `len() - 32`); a count in this band is the only
    // scene that can tell the two apart — anything at or below 16 is a
    // no-op under *either* a correct 32-cap or a mutated 16-cap, and 40 (the
    // other existing scene) drains by the same amount either way as long as
    // the drain count itself isn't touched. 20 sits strictly between the
    // two thresholds.
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    for i in 0..20 {
        dsdf.apply_destruction(DestructionShape::sphere(
            Vec3Fix::from_f32(i as f32 * 10.0, 0.0, 0.0),
            1.0,
        ));
    }
    dsdf.optimize();
    assert_eq!(
        dsdf.destruction_count(),
        20,
        "20 craters is below the 32 cap: optimize() must not touch them"
    );
    assert_eq!(dsdf.total_destruction_count(), 20);
    for i in 0..20 {
        assert!((dsdf.distance(i as f32 * 10.0, 0.0, 0.0) - 1.0).abs() < EPS);
    }
}

#[test]
fn optimize_above_the_cap_drops_exactly_the_oldest_overflow_and_is_idempotent() {
    // Mirrors the crate's own ground/crater convention, driven through the
    // named `destruction_from_explosion` constructor rather than the raw
    // `DestructionShape::sphere` constructor, to pin the same invariant
    // through a different production entry point.
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    let center_x = |i: usize| (i * 3) as f32;
    for i in 0..40 {
        dsdf.apply_destruction(destruction_from_explosion(
            Vec3Fix::from_f32(center_x(i), 0.0, 0.0),
            1.0,
            0.0,
        ));
    }
    assert_eq!(dsdf.destruction_count(), 40);
    assert_eq!(dsdf.total_destruction_count(), 40);

    dsdf.optimize();
    assert_eq!(dsdf.destruction_count(), 32);
    assert_eq!(
        dsdf.total_destruction_count(),
        40,
        "lifetime statistic must not shrink"
    );

    for i in 0..8 {
        assert!(
            dsdf.distance(center_x(i), 0.0, 0.0).abs() < EPS,
            "oldest 8 craters must be gone after eviction"
        );
    }
    for i in 8..40 {
        assert!(
            (dsdf.distance(center_x(i), 0.0, 0.0) - 1.0).abs() < EPS,
            "most recent 32 craters must remain"
        );
    }

    // Idempotent: calling again must not evict further.
    dsdf.optimize();
    assert_eq!(dsdf.destruction_count(), 32);
    assert!((dsdf.distance(center_x(8), 0.0, 0.0) - 1.0).abs() < EPS);
}

#[test]
fn reset_clears_destruction_count_but_not_the_lifetime_total() {
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    dsdf.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0));
    dsdf.apply_destruction(DestructionShape::sphere(
        Vec3Fix::from_f32(10.0, 0.0, 0.0),
        1.0,
    ));
    assert_eq!(dsdf.destruction_count(), 2);
    assert_eq!(dsdf.total_destruction_count(), 2);

    dsdf.reset();
    assert_eq!(
        dsdf.destruction_count(),
        0,
        "reset must clear the active destructions"
    );
    assert_eq!(
        dsdf.total_destruction_count(),
        2,
        "reset must not touch the lifetime statistic"
    );
    assert!(
        (dsdf.distance(0.0, 0.0, 0.0) - 0.0).abs() < EPS,
        "surface must be restored to the undamaged original"
    );

    // optimize() on the now-empty (post-reset) volume is also a no-op.
    dsdf.optimize();
    assert_eq!(dsdf.destruction_count(), 0);
}

#[test]
fn degenerate_event_far_from_any_query_point_is_still_recorded() {
    // `apply_destruction` never inspects the shape's geometry; it always
    // pushes + counts, documented behavior for an event that (at the scale
    // we query) never touches the surface.
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    dsdf.apply_destruction(destruction_from_explosion(
        Vec3Fix::from_f32(1.0e6, 0.0, 0.0),
        0.5,
        0.0,
    ));
    assert_eq!(dsdf.destruction_count(), 1);
    assert_eq!(dsdf.total_destruction_count(), 1);
    assert!(
        (dsdf.distance(0.0, 0.0, 0.0) - 0.0).abs() < EPS,
        "ground elsewhere is untouched"
    );
}

#[test]
fn degenerate_crater_larger_than_the_shape_removes_100_percent_of_it() {
    let mut dsdf = DestructibleSdf::new(Box::new(unit_sphere()));
    assert!(
        dsdf.distance(0.0, 0.0, 0.0) < 0.0,
        "origin starts inside the unit sphere"
    );
    dsdf.apply_destruction(destruction_from_explosion(Vec3Fix::ZERO, 100.0, 0.0));
    assert!(
        dsdf.distance(0.0, 0.0, 0.0) > 0.0,
        "a crater much larger than the shape must remove all of it"
    );
}

#[test]
fn degenerate_zero_size_shapes_do_not_panic() {
    let zero_sphere = panic::catch_unwind(|| DestructionShape::sphere(Vec3Fix::ZERO, 0.0));
    let zero_cube = panic::catch_unwind(|| DestructionShape::cube(Vec3Fix::ZERO, (0.0, 0.0, 0.0)));
    let zero_cylinder = panic::catch_unwind(|| DestructionShape::cylinder(Vec3Fix::ZERO, 0.0, 0.0));
    assert!(zero_sphere.is_ok());
    assert!(zero_cube.is_ok());
    assert!(zero_cylinder.is_ok());

    // Documented behavior: a zero-radius sphere only carves where the base
    // value was already negative (`0 > dist`); the ground (dist == 0 at
    // y == 0) is exactly the boundary, `0 > 0` is false, so it is untouched.
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    dsdf.apply_destruction(zero_sphere.unwrap());
    assert!((dsdf.distance(0.0, 0.0, 0.0) - 0.0).abs() < EPS);
}

#[test]
fn degenerate_extreme_coordinates_do_not_panic_and_stay_finite() {
    let mut dsdf = DestructibleSdf::new(Box::new(ground()));
    dsdf.apply_destruction(DestructionShape::sphere(Vec3Fix::ZERO, 1.0));

    let result = panic::catch_unwind(AssertUnwindSafe(|| dsdf.distance(1.0e30, 1.0e30, 1.0e30)));
    let value = result.expect("distance() must not panic on extreme coordinates");
    assert!(value.is_finite(), "1e30 is within f32 range: {value}");
}
