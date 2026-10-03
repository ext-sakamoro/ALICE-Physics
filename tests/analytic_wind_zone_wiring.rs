//! Closed-form oracles for `WindZone`'s five previously-unwired items:
//! `force_on`, `force_on_particle`, `instantaneous_wind_vector`,
//! `light_breeze`, `storm` (`wiring_guard.py` found zero production callers
//! for all five; see `examples/wind_zone_forces.rs` for the production
//! entry point this file's oracle backs).
//!
//! # Closed forms (from `src/wind_zone.rs:1-23`'s own module doc)
//!
//! ```text
//! wind(t) = direction * (base_speed + turbulence * sin(2*pi*f*t))
//! v_rel   = wind(t) - body_velocity
//! F       = 1/2 * rho_air * C_d * A * |v_rel| * v_rel
//! ```
//!
//! Every expected value here is derived independently with `f64` — never by
//! calling `instantaneous_wind_vector`, `force_on`, `aerodynamic_force` or
//! `force_on_particle` — then compared to the `Fix128` result (via
//! `to_f64()`). `TOL` follows the crate's own CORDIC `sin()` self-test
//! (`src/math.rs::transcendental_sweep_matches_f64_reference`, bounded at
//! `1e-11` against an independent f64 libm reference) with headroom for the
//! handful of extra fixed-point multiplications in `aerodynamic_force`.
//!
//! # Degenerate / extreme inputs
//!
//! * `instantaneous_wind_vector` at `t = 0` (`sin(0) = 0`, only the base
//!   term survives) and at an astronomically large `t` (range-reduction
//!   stress, still within Fix128's safe magnitude).
//! * `force_on` / `force_on_particle` for a body entirely outside the zone
//!   (zero, both branches), for a body whose velocity exactly matches the
//!   instantaneous wind (`v_rel = 0`, the `speed_sq.is_zero()` early
//!   return), and for an extreme-but-safe velocity magnitude (~1e9 m/s,
//!   `speed_sq ~ 1e18` stays under Fix128's ~9.22e18 ceiling).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The f64 `sin()` values here are closed-form oracle references computed
// outside the crate, not simulation state, so the determinism gate on f64
// transcendentals does not apply (same convention as
// tests/analytic_added_mass_coupling.rs).
#![allow(clippy::disallowed_methods)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use alice_physics::wind_zone::WindZone;

/// Relative error, falling back to absolute error near zero (matches the
/// crate's own `transcendental_sweep_matches_f64_reference` convention).
fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-9 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

/// See the module doc: CORDIC `sin()` is independently bounded at `1e-11`;
/// this leaves > 4 orders of magnitude of headroom.
const TOL: f64 = 1e-6;

fn assert_close(label: &str, actual: f64, expected: f64) {
    let err = rel_err(actual, expected);
    assert!(
        err < TOL,
        "{label}: actual {actual} vs expected {expected} (rel_err {err} >= {TOL})"
    );
}

fn wide_aabb() -> ZoneShape {
    ZoneShape::Aabb {
        min: Vec3Fix::from_int(-1_000, -1_000, -1_000),
        max: Vec3Fix::from_int(1_000, 1_000, 1_000),
    }
}

fn tiny_aabb_far_away() -> ZoneShape {
    // A valid zone that simply does not contain the origin, for
    // "body entirely outside the wind zone" scenarios.
    ZoneShape::Aabb {
        min: Vec3Fix::from_int(500, 500, 500),
        max: Vec3Fix::from_int(600, 600, 600),
    }
}

// ---------------------------------------------------------------------
// light_breeze / storm — pin the documented preset configuration
// ---------------------------------------------------------------------

#[test]
fn light_breeze_matches_documented_configuration() {
    // oracle: doc comment "3 m/s base + 0.5 m/s gust, C_d = 1.2, A = 0.5 m^2"
    // plus "Air density ... Sea-level ISA = 1.225" from the struct field doc.
    let zone = WindZone::light_breeze(wide_aabb());
    assert_close("air_density_kg_m3", zone.air_density_kg_m3.to_f64(), 1.225);
    assert_eq!(
        zone.direction,
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
    assert_close("base_speed_m_s", zone.base_speed_m_s.to_f64(), 3.0);
    assert_close(
        "turbulence_amplitude",
        zone.turbulence_amplitude.to_f64(),
        0.5,
    );
    assert_close("gust_frequency_hz", zone.gust_frequency_hz.to_f64(), 0.5);
    assert_close("drag_coefficient", zone.drag_coefficient.to_f64(), 1.2);
    assert_close("reference_area_m2", zone.reference_area_m2.to_f64(), 0.5);
}

#[test]
fn storm_matches_documented_configuration() {
    // oracle: doc comment "20 m/s base + 5 m/s gust"; Cd/A/rho/frequency
    // read directly from the source literals (not asserted elsewhere).
    let zone = WindZone::storm(wide_aabb());
    assert_close("air_density_kg_m3", zone.air_density_kg_m3.to_f64(), 1.225);
    assert_eq!(
        zone.direction,
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );
    assert_close("base_speed_m_s", zone.base_speed_m_s.to_f64(), 20.0);
    assert_close(
        "turbulence_amplitude",
        zone.turbulence_amplitude.to_f64(),
        5.0,
    );
    assert_close("gust_frequency_hz", zone.gust_frequency_hz.to_f64(), 1.5);
    assert_close("drag_coefficient", zone.drag_coefficient.to_f64(), 1.2);
    assert_close("reference_area_m2", zone.reference_area_m2.to_f64(), 0.5);
}

#[test]
fn storm_is_configured_stronger_than_breeze_on_every_wind_parameter() {
    // oracle: trivial consequence of the two closed forms above, checked
    // independently of them (direct field comparison, no f64 math needed).
    let breeze = WindZone::light_breeze(wide_aabb());
    let storm = WindZone::storm(wide_aabb());
    assert!(storm.base_speed_m_s > breeze.base_speed_m_s);
    assert!(storm.turbulence_amplitude > breeze.turbulence_amplitude);
    assert!(storm.gust_frequency_hz > breeze.gust_frequency_hz);
}

// ---------------------------------------------------------------------
// instantaneous_wind_vector
// ---------------------------------------------------------------------

#[test]
fn wind_vector_at_zero_time_is_exactly_the_base_term() {
    // oracle: sin(2*pi*f*0) = sin(0) = 0 (exact, independent of CORDIC)
    // => wind = direction * base_speed.
    let zone = WindZone::light_breeze(wide_aabb());
    let v = zone.instantaneous_wind_vector(Fix128::ZERO);
    assert_close("wind(t=0).x", v.x.to_f64(), 3.0);
    assert_close("wind(t=0).y", v.y.to_f64(), 0.0);
    assert_close("wind(t=0).z", v.z.to_f64(), 0.0);
}

#[test]
fn wind_vector_at_quarter_period_adds_full_turbulence() {
    // oracle: f = 0.5Hz, t = 0.5s => phase = 2*pi*0.5*0.5 = pi/2 => sin = 1
    // => wind = base_speed + turbulence_amplitude = 3.5.
    let zone = WindZone::light_breeze(wide_aabb());
    let v = zone.instantaneous_wind_vector(Fix128::from_ratio(1, 2));
    let phase = 2.0 * core::f64::consts::PI * 0.5 * 0.5;
    assert_close("wind(t=0.5).x", v.x.to_f64(), 3.0 + 0.5 * phase.sin());
    assert_close("wind(t=0.5).y", v.y.to_f64(), 0.0);
}

#[test]
fn wind_vector_at_three_quarter_period_subtracts_full_turbulence() {
    // oracle: f = 0.5Hz, t = 1.5s => phase = 3*pi/2 => sin = -1
    // => wind = base_speed - turbulence_amplitude = 2.5.
    let zone = WindZone::light_breeze(wide_aabb());
    let v = zone.instantaneous_wind_vector(Fix128::from_ratio(3, 2));
    let phase = 2.0 * core::f64::consts::PI * 0.5 * 1.5;
    assert_close("wind(t=1.5).x", v.x.to_f64(), 3.0 + 0.5 * phase.sin());
}

#[test]
fn wind_vector_for_storm_at_nontrivial_time_matches_f64_sin() {
    // oracle: independent f64 sin(2*pi*f*t) with the storm preset's f =
    // 1.5Hz at t = 0.25s (phase = 3*pi/4, not a "nice" multiple of pi/2).
    let zone = WindZone::storm(wide_aabb());
    let v = zone.instantaneous_wind_vector(Fix128::from_ratio(1, 4));
    let phase = 2.0 * core::f64::consts::PI * 1.5 * 0.25;
    assert_close(
        "wind(storm, t=0.25).x",
        v.x.to_f64(),
        20.0 + 5.0 * phase.sin(),
    );
    assert_close("wind(storm, t=0.25).y", v.y.to_f64(), 0.0);
    assert_close("wind(storm, t=0.25).z", v.z.to_f64(), 0.0);
}

#[test]
fn wind_vector_degenerate_extreme_time_does_not_panic_and_stays_bounded() {
    // Degenerate: an astronomically large t_s stresses the CORDIC range
    // reduction (`theta - TWO_PI * k`) inside sin(). The physical result is
    // undefined (nobody simulates t = 1e12 seconds) but the function must
    // not panic, and the result must stay within [base - turbulence,
    // base + turbulence] on the wind axis since sin is bounded in [-1, 1]
    // regardless of range-reduction precision.
    let zone = WindZone::light_breeze(wide_aabb());
    let extreme_t = Fix128::from_int(1_000_000_000_000);
    let result = catch_unwind(AssertUnwindSafe(|| {
        zone.instantaneous_wind_vector(extreme_t)
    }));
    let v = result.expect("instantaneous_wind_vector must not panic on an extreme time");
    let x = v.x.to_f64();
    assert!(
        (3.0 - 0.5 - 1e-3..=3.0 + 0.5 + 1e-3).contains(&x),
        "wind(t=1e12).x = {x} outside [base - turbulence, base + turbulence]"
    );
    assert_close("wind(t=1e12).y", v.y.to_f64(), 0.0);
    assert_close("wind(t=1e12).z", v.z.to_f64(), 0.0);
}

// ---------------------------------------------------------------------
// force_on — rigid body
// ---------------------------------------------------------------------

#[test]
fn force_on_resting_body_matches_half_rho_cd_a_v_squared() {
    // oracle: v_rel = wind(0) - 0 = (3,0,0), |v_rel| = 3 =>
    // F = 0.5 * 1.225 * 1.2 * 0.5 * 3 * 3 = 3.3075 (+x only).
    let zone = WindZone::light_breeze(wide_aabb());
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let f = zone.force_on(&body, Fix128::ZERO);
    let expected = 0.5 * 1.225 * 1.2 * 0.5 * 3.0 * 3.0;
    assert_close("force_on(resting).x", f.x.to_f64(), expected);
    assert_close("force_on(resting).y", f.y.to_f64(), 0.0);
    assert_close("force_on(resting).z", f.z.to_f64(), 0.0);
}

#[test]
fn force_on_pythagorean_relative_velocity_matches_closed_form() {
    // oracle: storm wind(0) = (20,0,0); body.velocity = (17,-4,0) =>
    // v_rel = (3,4,0), |v_rel| = 5 (exact sqrt of a perfect square) =>
    // magnitude = 0.5*1.225*1.2*0.5*5*5 = 9.1875, F = v_rel * (mag/5).
    let zone = WindZone::storm(wide_aabb());
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::new(Fix128::from_int(17), Fix128::from_int(-4), Fix128::ZERO);
    let f = zone.force_on(&body, Fix128::ZERO);
    let magnitude = 0.5 * 1.225 * 1.2 * 0.5 * 5.0 * 5.0;
    assert_close(
        "force_on(triangle).x",
        f.x.to_f64(),
        3.0 * (magnitude / 5.0),
    );
    assert_close(
        "force_on(triangle).y",
        f.y.to_f64(),
        4.0 * (magnitude / 5.0),
    );
    assert_close("force_on(triangle).z", f.z.to_f64(), 0.0);
}

#[test]
fn force_on_reverses_sign_when_body_outruns_wind() {
    // oracle: breeze wind(0) = (3,0,0); body.velocity = (10,0,0) =>
    // v_rel = (-7,0,0) => F = 0.5*1.225*1.2*0.5*7*7 along -x.
    let zone = WindZone::light_breeze(wide_aabb());
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(10, 0, 0);
    let f = zone.force_on(&body, Fix128::ZERO);
    let magnitude = 0.5 * 1.225 * 1.2 * 0.5 * 7.0 * 7.0;
    assert_close("force_on(outrunning_wind).x", f.x.to_f64(), -magnitude);
}

#[test]
fn force_on_degenerate_body_entirely_outside_zone_is_zero() {
    // Degenerate: shape.contains(position) == false short-circuits before
    // any force math runs.
    let zone = WindZone::light_breeze(tiny_aabb_far_away());
    let body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let f = zone.force_on(&body, Fix128::ZERO);
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn force_on_degenerate_velocity_exactly_matches_wind_is_zero() {
    // Degenerate: v_rel = 0 => speed_sq.is_zero() early-return path.
    let zone = WindZone::light_breeze(wide_aabb());
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = zone.direction * zone.base_speed_m_s; // (3,0,0), matches wind(0).
    let f = zone.force_on(&body, Fix128::ZERO);
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn force_on_degenerate_extreme_velocity_magnitude_matches_closed_form() {
    // Degenerate/extreme: velocity ~ -1e9 m/s, still safely inside Fix128's
    // range (max ~9.22e18; v_rel ~1e9 => speed_sq ~1e18, well clear of the
    // ceiling). v_rel = wind(0) - velocity = (3,0,0) - (-1e9,0,0)
    // = (1e9 + 3, 0, 0).
    let zone = WindZone::light_breeze(wide_aabb());
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(-1_000_000_000, 0, 0);
    let result = catch_unwind(AssertUnwindSafe(|| zone.force_on(&body, Fix128::ZERO)));
    let f = result.expect("force_on must not panic on an extreme-but-in-range velocity");
    let v_rel = 1.0e9 + 3.0;
    let expected = 0.5 * 1.225 * 1.2 * 0.5 * v_rel * v_rel;
    assert_close("force_on(extreme_velocity).x", f.x.to_f64(), expected);
    assert_close("force_on(extreme_velocity).y", f.y.to_f64(), 0.0);
    assert_close("force_on(extreme_velocity).z", f.z.to_f64(), 0.0);
}

// ---------------------------------------------------------------------
// force_on_particle — same math, free-floating position/velocity
// ---------------------------------------------------------------------

#[test]
fn force_on_particle_matches_force_on_for_the_equivalent_rigid_body_state() {
    // oracle: force_on_particle's own doc says it is force_on's math with a
    // bare position/velocity instead of a RigidBody — verified here against
    // the SAME independent closed form as force_on's triangle case, not by
    // calling force_on itself.
    let zone = WindZone::storm(wide_aabb());
    let velocity = Vec3Fix::new(Fix128::from_int(17), Fix128::from_int(-4), Fix128::ZERO);
    let f = zone.force_on_particle(Vec3Fix::ZERO, velocity, Fix128::ZERO);
    let magnitude = 0.5 * 1.225 * 1.2 * 0.5 * 5.0 * 5.0;
    assert_close(
        "force_on_particle(triangle).x",
        f.x.to_f64(),
        3.0 * (magnitude / 5.0),
    );
    assert_close(
        "force_on_particle(triangle).y",
        f.y.to_f64(),
        4.0 * (magnitude / 5.0),
    );
    assert_close("force_on_particle(triangle).z", f.z.to_f64(), 0.0);
}

#[test]
fn force_on_particle_degenerate_outside_zone_is_zero() {
    let zone = WindZone::light_breeze(tiny_aabb_far_away());
    let f = zone.force_on_particle(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ZERO);
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn force_on_particle_degenerate_velocity_matches_wind_is_zero() {
    let zone = WindZone::light_breeze(wide_aabb());
    let velocity = zone.direction * zone.base_speed_m_s;
    let f = zone.force_on_particle(Vec3Fix::ZERO, velocity, Fix128::ZERO);
    assert_eq!(f, Vec3Fix::ZERO);
}

#[test]
fn force_on_particle_degenerate_zero_time_matches_base_wind_closed_form() {
    // oracle: t_s = 0 => wind = direction * base_speed exactly, combined
    // with a non-zero particle velocity so the force math (not just the
    // zero-relative-velocity branch) actually runs.
    let zone = WindZone::storm(wide_aabb());
    let velocity = Vec3Fix::from_int(5, 0, 0);
    let f = zone.force_on_particle(Vec3Fix::ZERO, velocity, Fix128::ZERO);
    // v_rel = (20,0,0) - (5,0,0) = (15,0,0).
    let expected = 0.5 * 1.225 * 1.2 * 0.5 * 15.0 * 15.0;
    assert_close("force_on_particle(t=0).x", f.x.to_f64(), expected);
}
