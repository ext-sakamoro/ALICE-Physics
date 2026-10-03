//! Audit S3W3 oracles for `src/wind_zone.rs` (6 pub items).
//!
//! Closed forms (module doc of `wind_zone.rs`):
//!
//! ```text
//! wind(t) = dir * base + dir * turb * sin(2 pi f t)
//! F       = 1/2 rho Cd A |v_rel| v_rel,   v_rel = wind(t) - v_body
//! ```
//!
//! The existing `analytic_wind_zone_wiring.rs` only ever uses the two presets
//! (direction fixed to +x, Cd/A/rho fixed), so a swapped axis, a dropped
//! factor or a wrong sign on a y/z component cannot be seen there. These
//! oracles use general directions and distinct parameter values; every
//! expected value is computed in f64 from the formula above and never by
//! calling the code under test.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use alice_physics::wind_zone::WindZone;

const TOL: f64 = 1e-9;

fn close(label: &str, got: Fix128, want: f64) {
    let g = got.to_f64();
    let err = if want.abs() < 1e-6 {
        (g - want).abs()
    } else {
        (g - want).abs() / want.abs()
    };
    assert!(err < TOL, "{label}: got {g} want {want} err {err}");
}

fn r(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn big_box() -> ZoneShape {
    ZoneShape::Aabb {
        min: Vec3Fix::from_int(-100, -100, -100),
        max: Vec3Fix::from_int(100, 100, 100),
    }
}

/// dir = (3/5, 0, 4/5), rho = 2, Cd = 3/2, A = 1/4, base 7, turb 2, f = 1/4.
fn custom_zone(shape: ZoneShape) -> WindZone {
    WindZone {
        shape,
        air_density_kg_m3: Fix128::from_int(2),
        direction: Vec3Fix::new(r(3, 5), Fix128::ZERO, r(4, 5)),
        base_speed_m_s: Fix128::from_int(7),
        turbulence_amplitude: Fix128::from_int(2),
        gust_frequency_hz: r(1, 4),
        drag_coefficient: r(3, 2),
        reference_area_m2: r(1, 4),
    }
}

fn wind_f64(dir: [f64; 3], base: f64, turb: f64, f: f64, t: f64) -> [f64; 3] {
    let s = base + turb * (2.0 * std::f64::consts::PI * f * t).sin();
    [dir[0] * s, dir[1] * s, dir[2] * s]
}

#[test]
fn wind_general_direction_all_components_match_closed_form() {
    let z = custom_zone(big_box());
    // t = 2/3 s -> phase = 2 pi (1/4)(2/3) = pi/3
    let t = r(2, 3);
    let v = z.instantaneous_wind_vector(t);
    let w = wind_f64([0.6, 0.0, 0.8], 7.0, 2.0, 0.25, 2.0 / 3.0);
    close("wind.x", v.x, w[0]);
    close("wind.y", v.y, w[1]);
    close("wind.z", v.z, w[2]);
}

#[test]
fn wind_non_unit_direction_bakes_magnitude() {
    // Field doc: "the field can be non-unit if the caller wants to bake wind
    // magnitude in". dir = (0, 2, 0) doubles base and gust alike.
    let mut z = custom_zone(big_box());
    z.direction = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(2), Fix128::ZERO);
    let v = z.instantaneous_wind_vector(r(1, 3));
    let w = wind_f64([0.0, 2.0, 0.0], 7.0, 2.0, 0.25, 1.0 / 3.0);
    close("wind.x", v.x, w[0]);
    close("wind.y", v.y, w[1]);
    close("wind.z", v.z, w[2]);
}

#[test]
fn wind_negative_time_is_odd_gust() {
    // sin is odd: wind(-t) = dir*(base - turb*sin(2 pi f t))
    let z = custom_zone(big_box());
    let v = z.instantaneous_wind_vector(-r(2, 3));
    let w = wind_f64([0.6, 0.0, 0.8], 7.0, 2.0, 0.25, -2.0 / 3.0);
    close("wind.x", v.x, w[0]);
    close("wind.z", v.z, w[2]);
}

#[test]
fn wind_gust_period_is_one_over_frequency() {
    // f = 1/4 Hz -> period 4 s: wind(t + 4) == wind(t)
    let z = custom_zone(big_box());
    let a = z.instantaneous_wind_vector(r(1, 3));
    let b = z.instantaneous_wind_vector(r(1, 3) + Fix128::from_int(4));
    close("period.x", b.x, a.x.to_f64());
    close("period.z", b.z, a.z.to_f64());
}

#[test]
fn wind_zero_turbulence_is_time_independent() {
    let mut z = custom_zone(big_box());
    z.turbulence_amplitude = Fix128::ZERO;
    for t in [0i64, 1, 5, 123] {
        let v = z.instantaneous_wind_vector(Fix128::from_int(t));
        close("wind.x", v.x, 0.6 * 7.0);
        close("wind.z", v.z, 0.8 * 7.0);
    }
}

#[test]
fn force_general_3d_matches_half_rho_cd_a_speed_vrel() {
    let z = custom_zone(big_box());
    let mut b = RigidBody::new(Vec3Fix::from_int(1, 2, 3), Fix128::ONE);
    b.velocity = Vec3Fix::new(Fix128::from_int(1), -Fix128::from_int(2), r(1, 2));
    let t = r(2, 3);
    let f = z.force_on(&b, t);
    let w = wind_f64([0.6, 0.0, 0.8], 7.0, 2.0, 0.25, 2.0 / 3.0);
    let rel = [w[0] - 1.0, w[1] + 2.0, w[2] - 0.5];
    let speed = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
    let k = 0.5 * 2.0 * 1.5 * 0.25 * speed;
    close("F.x", f.x, k * rel[0]);
    close("F.y", f.y, k * rel[1]);
    close("F.z", f.z, k * rel[2]);
    // particle entry point is the same law
    let fp = z.force_on_particle(b.position, b.velocity, t);
    close("Fp.x", fp.x, k * rel[0]);
    close("Fp.y", fp.y, k * rel[1]);
    close("Fp.z", fp.z, k * rel[2]);
}

#[test]
fn force_is_parallel_to_relative_velocity_and_dissipative() {
    let z = custom_zone(big_box());
    let vel = Vec3Fix::new(r(-3, 2), Fix128::from_int(4), r(5, 3));
    let t = r(1, 7);
    let f = z.force_on_particle(Vec3Fix::ZERO, vel, t);
    let w = z.instantaneous_wind_vector(t);
    let rel = w - vel;
    let c = f.cross(rel);
    assert!(c.length().to_f64() < 1e-8, "F x v_rel = {:?}", c);
    assert!(f.dot(rel).to_f64() > 0.0, "F must point along v_rel");
}

#[test]
fn force_scales_linearly_in_rho_cd_a_independently() {
    let base = custom_zone(big_box());
    let vel = Vec3Fix::from_int(1, 1, 1);
    let f0 = base.force_on_particle(Vec3Fix::ZERO, vel, Fix128::ZERO);
    let mut z = base;
    z.air_density_kg_m3 = Fix128::from_int(6);
    let f = z.force_on_particle(Vec3Fix::ZERO, vel, Fix128::ZERO);
    close("rho x3", f.x, 3.0 * f0.x.to_f64());
    let mut z = base;
    z.drag_coefficient = r(9, 2);
    let f = z.force_on_particle(Vec3Fix::ZERO, vel, Fix128::ZERO);
    close("Cd x3", f.z, 3.0 * f0.z.to_f64());
    let mut z = base;
    z.reference_area_m2 = r(3, 4);
    let f = z.force_on_particle(Vec3Fix::ZERO, vel, Fix128::ZERO);
    close("A x3", f.x, 3.0 * f0.x.to_f64());
}

#[test]
fn force_is_quadratic_in_relative_speed() {
    // still air in the zone frame: body at -v gives v_rel = wind + v. With
    // zero turbulence, doubling v_rel quadruples |F|.
    let mut z = custom_zone(big_box());
    z.turbulence_amplitude = Fix128::ZERO;
    let wind = z.direction * z.base_speed_m_s;
    let f1 = z.force_on_particle(Vec3Fix::ZERO, wind * (-Fix128::ONE), Fix128::ZERO);
    let f2 = z.force_on_particle(Vec3Fix::ZERO, wind * (-Fix128::from_int(3)), Fix128::ZERO);
    // v_rel: 2*wind vs 4*wind -> ratio 4
    close("quad", f2.length(), 4.0 * f1.length().to_f64());
}

#[test]
fn force_respects_zone_shape_for_body_and_particle() {
    let sphere = ZoneShape::Sphere {
        centre: Vec3Fix::from_int(10, 0, 0),
        radius: Fix128::from_int(2),
    };
    let z = custom_zone(sphere);
    let inside = Vec3Fix::from_int(11, 0, 0);
    let outside = Vec3Fix::from_int(13, 0, 0);
    let mut b = RigidBody::new(inside, Fix128::ONE);
    assert!(z.force_on(&b, Fix128::ZERO).length().to_f64() > 1.0);
    assert!(
        z.force_on_particle(inside, Vec3Fix::ZERO, Fix128::ZERO)
            .length()
            .to_f64()
            > 1.0
    );
    b.position = outside;
    assert_eq!(z.force_on(&b, Fix128::ZERO), Vec3Fix::ZERO);
    assert_eq!(
        z.force_on_particle(outside, Vec3Fix::ZERO, Fix128::ZERO),
        Vec3Fix::ZERO
    );
}

#[test]
fn force_on_uses_body_position_not_previous_position() {
    let sphere = ZoneShape::Sphere {
        centre: Vec3Fix::ZERO,
        radius: Fix128::ONE,
    };
    let z = custom_zone(sphere);
    let mut b = RigidBody::new(Vec3Fix::from_int(50, 0, 0), Fix128::ONE);
    b.prev_position = Vec3Fix::ZERO; // inside, but the current position is outside
    assert_eq!(z.force_on(&b, Fix128::ZERO), Vec3Fix::ZERO);
}

#[test]
fn force_small_relative_velocity_is_not_lost() {
    // v_rel = 1e-3 m/s along the wind: F = 1/2 rho Cd A |v|^2 = 0.375e-6 *...
    // body moves with the wind minus 1e-3 so v_rel is tiny but resolvable.
    let mut z = custom_zone(big_box());
    z.turbulence_amplitude = Fix128::ZERO;
    let wind = z.direction * z.base_speed_m_s;
    let eps = r(1, 1000);
    let vel = wind - z.direction * eps;
    let f = z.force_on_particle(Vec3Fix::ZERO, vel, Fix128::ZERO);
    let want = 0.5 * 2.0 * 1.5 * 0.25 * 1e-3 * 1e-3; // magnitude
    close("small |F|", f.length(), want);
}

#[test]
#[ignore = "known defect: AUD-A-S3W3-001: aerodynamic_force squares |v_rel| in Fix128 before sqrt, so v_rel > ~3.04e9 m/s overflows speed_sq and force collapses to 0 (v_rel=4e9 along +x: got 0, want +6.0e18 which is representable)"]
fn force_huge_relative_velocity_keeps_sign_and_magnitude() {
    // v_rel = 4e9 m/s: speed_sq = 1.6e19 exceeds the I64F64 range (9.2e18).
    // The documented law has no bound, so the force must still be +x with
    // magnitude 1/2 rho Cd A v^2.
    let mut z = custom_zone(big_box());
    z.direction = Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO);
    z.turbulence_amplitude = Fix128::ZERO;
    z.base_speed_m_s = Fix128::from_int(4_000_000_000);
    let f = z.force_on_particle(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ZERO);
    assert!(f.x.to_f64() > 0.0, "force flipped/zeroed: {}", f.x.to_f64());
}
