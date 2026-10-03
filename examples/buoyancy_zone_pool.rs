//! A floating sphere in a `BuoyancyZone::water_pool`, checked against Archimedes.
//!
//! Drives `BuoyancyZone::water_pool` and `ZoneShape::depth_below_surface` from a
//! production entry point. A sphere of radius 0.5 m and half the density of
//! water (500 kg/m^3) floats with its centre exactly at the surface (the
//! submerged cap is half the ball by symmetry), so at that height the zone's
//! buoyancy must equal the weight `m g`:
//!
//! ```text
//! m = 500 * (4/3) pi r^3,   F_b = 1000 * (4/3) pi r^3 * (1/2) * 9.81 = m g
//! ```
//!
//! The example then bisects the equilibrium height of a denser body (750 kg/m^3)
//! with `force_on` and compares it with the closed-form root of the spherical-cap
//! equation `h^2 (3r - h) / (4 r^3) = 0.75`.
//!
//! ```bash
//! cargo run --release --example buoyancy_zone_pool --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::buoyancy_zone::{BuoyancyZone, ZoneShape};
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::solver::{BodyType, RigidBody};

fn body_at(y: Fix128) -> RigidBody {
    let pos = Vec3Fix::new(Fix128::ZERO, y, Fix128::ZERO);
    RigidBody {
        position: pos,
        velocity: Vec3Fix::ZERO,
        inv_mass: Fix128::ONE,
        inv_inertia: Vec3Fix::ZERO,
        prev_position: pos,
        rotation: QuatFix::IDENTITY,
        angular_velocity: Vec3Fix::ZERO,
        prev_rotation: QuatFix::IDENTITY,
        restitution: Fix128::ZERO,
        friction: Fix128::ZERO,
        gravity_scale: Fix128::ONE,
        linear_damping: Fix128::ONE,
        angular_damping: Fix128::ONE,
        is_sensor: false,
        body_type: BodyType::Dynamic,
        kinematic_target: None,
    }
}

fn main() {
    // 10 m x 10 m x 10 m pool, surface at y = 5
    let shape = ZoneShape::Aabb {
        min: Vec3Fix::from_int(-5, -5, -5),
        max: Vec3Fix::from_int(5, 5, 5),
    };
    let zone = BuoyancyZone::water_pool(shape);
    let r = 0.5_f64;
    let r_fix = Fix128::from_ratio(1, 2);
    let vol = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;

    println!(
        "water_pool: rho = {} kg/m^3, g = {} m/s^2",
        zone.density_kg_m3.to_f64(),
        zone.gravity.to_f64()
    );

    // 1. half-density sphere floats at the surface
    let at_surface = body_at(Fix128::from_int(5));
    let depth = zone.shape.depth_below_surface(at_surface.position);
    let f = zone.force_on(&at_surface, r_fix);
    let weight = 500.0 * vol * 9.81;
    println!(
        "centre depth below the surface = {} m; F_b = {:.9} N, weight = {:.9} N",
        depth.to_f64(),
        f.y.to_f64(),
        weight
    );
    assert_eq!(depth, Fix128::ZERO);
    assert!((f.y.to_f64() - weight).abs() / weight < 1e-12);

    // 2. denser body (750 kg/m^3): equilibrium height by bisection on force_on
    let weight75 = 750.0 * vol * 9.81;
    let (mut lo, mut hi) = (Fix128::from_int(4), Fix128::from_int(6)); // centre y range around the surface
    for _ in 0..60 {
        let mid = (lo + hi).half();
        let up = zone.force_on(&body_at(mid), r_fix).y.to_f64();
        // buoyancy falls as the body rises
        if up > weight75 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let y_eq = (lo + hi).half().to_f64();
    // closed form: solve h^2 (3r - h) / (4 r^3) = 3/4 for h by bisection in f64
    let (mut a, mut b) = (0.0_f64, 2.0 * r);
    for _ in 0..200 {
        let h = 0.5 * (a + b);
        let frac = h * h * (3.0 * r - h) / (4.0 * r * r * r);
        if frac < 0.75 {
            a = h;
        } else {
            b = h;
        }
    }
    let h_eq = 0.5 * (a + b);
    let y_expect = 5.0 + r - h_eq; // surface + r - h  (h = d + r with d = 5 - y)
    println!("equilibrium centre height: zone = {y_eq:.9} m, closed form = {y_expect:.9} m");
    assert!((y_eq - y_expect).abs() < 1e-9);
}
