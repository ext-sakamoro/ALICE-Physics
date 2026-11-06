//! Oracle: a ray query finds a body whose collision box leaves the `Fix128` range.
//!
//! # What is measured
//!
//! `PhysicsWorld::raycast` and `PhysicsWorld::cast_ray` build a BVH of the bodies'
//! boxes for the query. A box built with `AABB::from_center_half` from a position
//! and a radius whose sum leaves the range wraps to the other side and comes out
//! inverted, so the BVH never offers that body as a candidate and the ray misses
//! it although the sphere's near surface is well inside the range. The broad-phase
//! of `step` builds its boxes with the saturating form since 001001f1; the query
//! paths are expected to see the same body.
//!
//! The closed form: a body at `x = 2^62 + 2^40` with radius `2^62` has its near
//! surface at `x = 2^40`; a ray from the origin along +x with reach `2^41` hits it
//! at `t = 2^40`. The mirrored body (`x = -(2^62 + 2^40)`, ray along -x) is the
//! same closed form with the box's minimum wrapping instead of its maximum. An
//! in-range twin (`x = 3`, radius `1`, hit at `t = 2`) keeps the same query honest.
//!
//! Author: Moroya Sakamoto

use alice_physics_core::math::{Fix128, Vec3Fix};
use alice_physics_core::shape_raycast::RayFilter;
use alice_physics_core::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn pow2(n: u32) -> Fix128 {
    Fix128::from_raw(1i64 << n, 0)
}

fn x_at(x: Fix128) -> Vec3Fix {
    Vec3Fix::new(x, Fix128::ZERO, Fix128::ZERO)
}

fn world_with_sphere(x: Fix128, radius: Fix128) -> PhysicsWorld {
    let mut w = PhysicsWorld::new(SolverConfig {
        gravity: Vec3Fix::ZERO,
        ..SolverConfig::default()
    });
    w.add_body_with_radius(RigidBody::new(x_at(x), Fix128::ONE), radius);
    w
}

/// (body x, ray direction sign, expected t) for the two wrapping sides and the twin.
fn cases() -> Vec<(&'static str, Fix128, Fix128, Fix128, Fix128)> {
    let far = pow2(62) + pow2(40);
    vec![
        ("in range", Fix128::from_int(3), Fix128::ONE, Fix128::ONE, Fix128::from_int(2)),
        ("box max wraps", far, pow2(62), Fix128::ONE, pow2(40)),
        ("box min wraps", -far, pow2(62), -Fix128::ONE, pow2(40)),
    ]
}

#[test]
fn raycast_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let reach = if want_t > pow2(41) { want_t * Fix128::from_int(2) } else { pow2(41) };
        let hit = w.raycast(Vec3Fix::ZERO, x_at(sign), reach);
        assert_eq!(hit, Some((0, want_t)), "{name}: raycast");
    }
}

#[test]
fn cast_ray_hits_a_body_whose_box_leaves_the_range() {
    for (name, x, r, sign, want_t) in cases() {
        let w = world_with_sphere(x, r);
        let hit = w.cast_ray(Vec3Fix::ZERO, x_at(sign), pow2(41), &RayFilter::default());
        let hit = hit.unwrap_or_else(|| panic!("{name}: cast_ray missed"));
        assert_eq!((hit.body, hit.t), (Some(0), want_t), "{name}: cast_ray");
    }
}
