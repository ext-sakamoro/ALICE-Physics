//! World casts with a direction too long to square in `Fix128`.
//!
//! oracle: closed form. A static box plate with its near face at
//! `x = 5 − 1/64`, cast at from the origin along `+x`: a ray hits it at
//! `t = 5 − 1/64`, a sphere of radius `1/4` at `t = 5 − 1/64 − 1/4`, whatever the
//! length of the direction vector (the casts measure `t` along its unit
//! direction). A direction of length `2³⁴` (`|d|² = 2⁶⁸`, beyond the `2⁶³` the
//! fixed-point square can hold) must give the same hits as a unit one.

#![cfg(feature = "std")]

use alice_physics::shape::Shape;
use alice_physics::shape_raycast::RayFilter;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, RigidBody, Vec3Fix};

fn plate_world() -> PhysicsWorld {
    let mut w = PhysicsWorld::new(PhysicsConfig::default());
    let p = w.add_body(RigidBody::new_static(Vec3Fix::from_int(5, 0, 0)));
    w.set_body_shape(
        p,
        &Shape::Box {
            half_extents: Vec3Fix::new(
                Fix128::from_ratio(1, 64),
                Fix128::from_int(2),
                Fix128::from_int(2),
            ),
        },
    );
    w
}

fn directions() -> [(&'static str, Vec3Fix); 3] {
    [
        ("unit", Vec3Fix::from_int(1, 0, 0)),
        ("2^20", Vec3Fix::from_int(1 << 20, 0, 0)),
        ("2^34", Vec3Fix::from_int(1 << 34, 0, 0)),
    ]
}

fn face() -> Fix128 {
    Fix128::from_int(5) - Fix128::from_ratio(1, 64)
}

#[test]
fn cast_sphere_hits_at_the_closed_form_distance_for_any_direction_length() {
    let w = plate_world();
    let r = Fix128::from_ratio(1, 4);
    for (name, d) in directions() {
        let hit = w
            .cast_sphere(
                Vec3Fix::ZERO,
                r,
                d,
                Fix128::from_int(1 << 34),
                &RayFilter::new(),
            )
            .unwrap_or_else(|| panic!("{name}: no hit"));
        assert_eq!(hit.t, face() - r, "{name}");
        assert_eq!(
            hit.point,
            Vec3Fix::new(face(), Fix128::ZERO, Fix128::ZERO),
            "{name}"
        );
        assert_eq!(hit.normal, Vec3Fix::from_int(-1, 0, 0), "{name}");
    }
}

#[test]
fn cast_ray_hits_at_the_closed_form_distance_for_any_direction_length() {
    let w = plate_world();
    for (name, d) in directions() {
        let hit = w
            .cast_ray(
                Vec3Fix::ZERO,
                d,
                Fix128::from_int(1 << 34),
                &RayFilter::new(),
            )
            .unwrap_or_else(|| panic!("{name}: no hit"));
        assert_eq!(hit.t, face(), "{name}");
        assert_eq!(hit.normal, Vec3Fix::from_int(-1, 0, 0), "{name}");
    }
}
