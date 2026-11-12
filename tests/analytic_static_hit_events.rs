//! A body meeting a static collider is reported with its approach speed, which a
//! free fall gives in closed form: `v = sqrt(2 g h)`.

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::plane_collider::PlaneCollider;
use alice_physics_core::solver::{PhysicsConfig, PhysicsWorld, RigidBody};
use alice_physics_core::static_collider::StaticCollider;

const G: f64 = 10.0;

fn world(height: f64) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig {
        gravity: Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO),
        ..Default::default()
    });
    let plane = PlaneCollider::new(
        Vec3Fix::new(Fix128::ZERO, Fix128::ONE, Fix128::ZERO),
        Fix128::ZERO,
    );
    w.add_static_collider(StaticCollider::Plane(plane));
    let start = Vec3Fix::new(Fix128::ZERO, Fix128::from_f64(height + 0.5), Fix128::ZERO);
    w.add_body_with_radius(
        RigidBody::new_dynamic(start, Fix128::ONE),
        Fix128::from_ratio(1, 2),
    );
    w
}

/// The hit is the substep whose approach closes the gap to the plane (a speculative
/// contact met in it, or an overlap), so the body has fallen up to one step past the
/// height: its speed then exceeds `sqrt(2 g h)` by at most `g dt` from that last step, and
/// the stepping of the fall (velocity first, then position) moves the moment the height is
/// crossed by at most one more step, another `g dt`. The bound is `2 g dt`, checked at three
/// steps so that it shrinks with the step. (Reporting only overlaps missed the impact: the
/// solve stops a body at the plane before it overlaps, and the first overlap was the body
/// at rest.)
#[test]
fn the_first_hit_carries_the_speed_of_the_fall() {
    for steps_per_second in [60_i64, 120, 240] {
        let dt = Fix128::from_ratio(1, steps_per_second);
        let bound = 2.0 * G / steps_per_second as f64;
        for height in [0.5, 2.0, 8.0] {
            let mut w = world(height);
            let want = (2.0 * G * height).sqrt();
            let mut first = None;
            for _ in 0..(10 * steps_per_second) {
                w.step(dt);
                if let Some(hit) = w.static_hits().first() {
                    first = Some(*hit);
                    break;
                }
            }
            let hit = first.expect("the fall reached the plane");
            assert_eq!(hit.body, 0);
            assert_eq!(hit.collider, 0);
            let speed = hit.speed.to_f64();
            assert!(
                (speed - want).abs() <= bound,
                "dt 1/{steps_per_second}, h {height}: {speed} against {want}, bound {bound}"
            );
            assert!((hit.normal.y.to_f64() - 1.0).abs() < 1e-9);
            assert!(hit.point.y.to_f64().abs() < 0.2, "{}", hit.point.y.to_f64());
        }
    }
}

#[test]
fn no_hit_is_reported_in_the_air_and_each_step_starts_empty() {
    let mut w = world(8.0);
    let dt = Fix128::from_ratio(1, 240);
    for _ in 0..10 {
        w.step(dt);
        assert!(w.static_hits().is_empty());
    }
}

#[test]
fn a_body_at_rest_reports_a_small_speed_not_its_fall() {
    let mut w = world(2.0);
    let dt = Fix128::from_ratio(1, 240);
    for _ in 0..2400 {
        w.step(dt);
    }
    let last = w
        .static_hits()
        .iter()
        .map(|h| h.speed.to_f64())
        .fold(0.0, f64::max);
    assert!(last < 0.5, "{last}");
}
