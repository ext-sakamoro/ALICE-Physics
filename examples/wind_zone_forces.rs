//! Driving `WindZone`'s five zero-production-caller items through a
//! production entry point.
//!
//! `wiring_guard.py` found `src/wind_zone.rs::{force_on, force_on_particle,
//! instantaneous_wind_vector, light_breeze, storm}` with zero production
//! callers — only `#[cfg(test)]` unit tests inside the module itself called
//! in, which the guard does not count (see the module's own test list:
//! `light_breeze_preset_configured`, `storm_preset_stronger_than_breeze`,
//! `wind_vector_at_zero_time_matches_base`, `force_zero_outside_zone`,
//! `force_direction_aligned_with_wind`,
//! `force_reverses_when_body_moves_faster_than_wind`,
//! `particle_force_matches_body_force_for_same_state`,
//! `zero_force_when_wind_matches_body_velocity` — those tests already pin
//! the math, so this example is about *wiring*, not re-deriving the
//! formulas).
//!
//! Closed forms, from the module doc (`src/wind_zone.rs:1-23`):
//!
//! ```text
//! wind(t)  = direction * (base_speed + turbulence * sin(2*pi*f*t))
//! v_rel    = wind(t) - body_velocity
//! F        = 1/2 * rho_air * C_d * A * |v_rel| * v_rel
//! ```
//!
//! Every expected value below is computed independently with `f64` (never
//! by calling `WindZone::{instantaneous_wind_vector, force_on,
//! force_on_particle}`), then compared to the `Fix128` result (via
//! `to_f64()`) with a documented tolerance. The dominant error source is
//! the crate's CORDIC `sin()`, whose own self-test
//! (`src/math.rs::transcendental_sweep_matches_f64_reference`) bounds it at
//! `1e-11` against an `f64` libm reference; this example uses `1e-6` to
//! leave comfortable headroom for the extra multiplications downstream of
//! `sin()` while still catching a wrong sign, a dropped term or a wrong
//! coefficient (any of which would be many orders of magnitude off).
//!
//! ```bash
//! cargo run --example wind_zone_forces --features std
//! ```

// The f64 `sin()` calls here are closed-form oracle references computed
// outside the crate, not simulation state, so the determinism gate on f64
// transcendentals (`clippy::disallowed_methods`) does not apply.
#![allow(clippy::disallowed_methods)]

use alice_physics::buoyancy_zone::ZoneShape;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;
use alice_physics::wind_zone::WindZone;

/// Relative error, falling back to absolute error near zero.
fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected.abs() < 1e-9 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

/// Generous tolerance: CORDIC `sin()` is bounded at `1e-11` against f64
/// libm; this leaves > 4 orders of magnitude of headroom for the handful of
/// extra multiplications in `aerodynamic_force`.
const TOL: f64 = 1e-6;

fn check(label: &str, actual: f64, expected: f64) {
    let err = rel_err(actual, expected);
    println!("[wind_zone] {label}: actual={actual:.12} expected={expected:.12} rel_err={err:.3e}");
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

fn main() {
    // ----------------------------------------------------------------
    // 1. light_breeze / storm presets — pin the documented configuration.
    // ----------------------------------------------------------------
    let breeze = WindZone::light_breeze(wide_aabb());
    println!(
        "[wind_zone] light_breeze: base_speed={} turbulence={} Cd={} A={}",
        breeze.base_speed_m_s.to_f64(),
        breeze.turbulence_amplitude.to_f64(),
        breeze.drag_coefficient.to_f64(),
        breeze.reference_area_m2.to_f64()
    );
    check(
        "light_breeze.air_density_kg_m3",
        breeze.air_density_kg_m3.to_f64(),
        1.225,
    );
    check(
        "light_breeze.base_speed_m_s",
        breeze.base_speed_m_s.to_f64(),
        3.0,
    );
    check(
        "light_breeze.turbulence_amplitude",
        breeze.turbulence_amplitude.to_f64(),
        0.5,
    );
    check(
        "light_breeze.gust_frequency_hz",
        breeze.gust_frequency_hz.to_f64(),
        0.5,
    );
    check(
        "light_breeze.drag_coefficient",
        breeze.drag_coefficient.to_f64(),
        1.2,
    );
    check(
        "light_breeze.reference_area_m2",
        breeze.reference_area_m2.to_f64(),
        0.5,
    );
    assert_eq!(
        breeze.direction,
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );

    let storm = WindZone::storm(wide_aabb());
    println!(
        "[wind_zone] storm: base_speed={} turbulence={} Cd={} A={}",
        storm.base_speed_m_s.to_f64(),
        storm.turbulence_amplitude.to_f64(),
        storm.drag_coefficient.to_f64(),
        storm.reference_area_m2.to_f64()
    );
    check(
        "storm.air_density_kg_m3",
        storm.air_density_kg_m3.to_f64(),
        1.225,
    );
    check("storm.base_speed_m_s", storm.base_speed_m_s.to_f64(), 20.0);
    check(
        "storm.turbulence_amplitude",
        storm.turbulence_amplitude.to_f64(),
        5.0,
    );
    check(
        "storm.gust_frequency_hz",
        storm.gust_frequency_hz.to_f64(),
        1.5,
    );
    check(
        "storm.drag_coefficient",
        storm.drag_coefficient.to_f64(),
        1.2,
    );
    check(
        "storm.reference_area_m2",
        storm.reference_area_m2.to_f64(),
        0.5,
    );
    assert_eq!(
        storm.direction,
        Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO)
    );

    // ----------------------------------------------------------------
    // 2. instantaneous_wind_vector at a handful of times.
    // ----------------------------------------------------------------
    // t = 0 -> sin(0) = 0 -> wind == base term exactly.
    let v0 = breeze.instantaneous_wind_vector(Fix128::ZERO);
    check("wind(breeze, t=0).x", v0.x.to_f64(), 3.0);
    check("wind(breeze, t=0).y", v0.y.to_f64(), 0.0);
    check("wind(breeze, t=0).z", v0.z.to_f64(), 0.0);

    // t = 0.5s, f = 0.5Hz -> phase = 2*pi*0.5*0.5 = pi/2 -> sin = 1 ->
    // wind = base_speed + turbulence_amplitude = 3.5.
    let t_quarter = Fix128::from_ratio(1, 2);
    let v_quarter = breeze.instantaneous_wind_vector(t_quarter);
    let phase_quarter = 2.0 * core::f64::consts::PI * 0.5 * 0.5;
    check(
        "wind(breeze, t=0.5).x",
        v_quarter.x.to_f64(),
        3.0 + 0.5 * phase_quarter.sin(),
    );
    check("wind(breeze, t=0.5).y", v_quarter.y.to_f64(), 0.0);

    // t = 1.0s, f = 0.5Hz -> phase = pi -> sin = 0 -> wind == base term,
    // same value as t=0 but reached via a different (nonzero) phase.
    let t_half_period = Fix128::ONE;
    let v_half_period = breeze.instantaneous_wind_vector(t_half_period);
    check("wind(breeze, t=1.0).x", v_half_period.x.to_f64(), 3.0);

    // t = 1.5s, f = 0.5Hz -> phase = 3*pi/2 -> sin = -1 -> wind = base -
    // turbulence = 2.5.
    let t_three_quarter = Fix128::from_ratio(3, 2);
    let v_three_quarter = breeze.instantaneous_wind_vector(t_three_quarter);
    let phase_three_quarter = 2.0 * core::f64::consts::PI * 0.5 * 1.5;
    check(
        "wind(breeze, t=1.5).x",
        v_three_quarter.x.to_f64(),
        3.0 + 0.5 * phase_three_quarter.sin(),
    );

    // storm at a non-trivial time (f = 1.5Hz): closed form via f64 sin.
    let t_storm = Fix128::from_ratio(1, 4);
    let v_storm = storm.instantaneous_wind_vector(t_storm);
    let phase_storm = 2.0 * core::f64::consts::PI * 1.5 * 0.25;
    check(
        "wind(storm, t=0.25).x",
        v_storm.x.to_f64(),
        20.0 + 5.0 * phase_storm.sin(),
    );

    // ----------------------------------------------------------------
    // 3. force_on — rigid body at various positions/velocities.
    // ----------------------------------------------------------------
    // (a) Resting body inside the zone, t=0: wind = (3,0,0), v_rel = (3,0,0),
    //     |v_rel| = 3 -> F = 0.5*1.225*1.2*0.5*3*3 = 3.3075 (+x).
    let resting = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    let f_resting = breeze.force_on(&resting, Fix128::ZERO);
    let expected_mag_a = 0.5 * 1.225 * 1.2 * 0.5 * 3.0 * 3.0;
    check(
        "force_on(resting, breeze, t=0).x",
        f_resting.x.to_f64(),
        expected_mag_a,
    );
    check(
        "force_on(resting, breeze, t=0).y",
        f_resting.y.to_f64(),
        0.0,
    );
    check(
        "force_on(resting, breeze, t=0).z",
        f_resting.z.to_f64(),
        0.0,
    );

    // (b) 3-4-5 relative-velocity triangle (storm, t=0 -> wind=(20,0,0)):
    //     body.velocity = (17,-4,0) -> v_rel = (3,4,0), |v_rel| = 5 ->
    //     magnitude = 0.5*1.225*1.2*0.5*5*5 = 9.1875 -> F = v_rel * (9.1875/5).
    let mut triangle_body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    triangle_body.velocity = Vec3Fix::new(Fix128::from_int(17), Fix128::from_int(-4), Fix128::ZERO);
    let f_triangle = storm.force_on(&triangle_body, Fix128::ZERO);
    let mag_b = 0.5 * 1.225 * 1.2 * 0.5 * 5.0 * 5.0;
    check(
        "force_on(triangle, storm, t=0).x",
        f_triangle.x.to_f64(),
        3.0 * (mag_b / 5.0),
    );
    check(
        "force_on(triangle, storm, t=0).y",
        f_triangle.y.to_f64(),
        4.0 * (mag_b / 5.0),
    );
    check(
        "force_on(triangle, storm, t=0).z",
        f_triangle.z.to_f64(),
        0.0,
    );

    // (c) Degenerate: body entirely outside the zone -> zero force.
    let far_body = RigidBody::new(Vec3Fix::from_int(10_000, 0, 0), Fix128::ONE);
    let f_outside = breeze.force_on(&far_body, Fix128::ZERO);
    check("force_on(outside_zone).x", f_outside.x.to_f64(), 0.0);
    check("force_on(outside_zone).y", f_outside.y.to_f64(), 0.0);
    check("force_on(outside_zone).z", f_outside.z.to_f64(), 0.0);

    // (d) Degenerate: body velocity exactly matches wind -> v_rel = 0 -> zero.
    let mut matching_body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    matching_body.velocity = breeze.direction * breeze.base_speed_m_s;
    let f_matching = breeze.force_on(&matching_body, Fix128::ZERO);
    check(
        "force_on(velocity_matches_wind).x",
        f_matching.x.to_f64(),
        0.0,
    );

    // (e) Extreme magnitude: velocity ~1e9 m/s, still inside Fix128's safe
    //     range (max ~9.22e18; speed_sq ~1e18 stays well clear of overflow).
    let mut extreme_body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    extreme_body.velocity =
        Vec3Fix::new(Fix128::from_int(-1_000_000_000), Fix128::ZERO, Fix128::ZERO);
    let f_extreme = breeze.force_on(&extreme_body, Fix128::ZERO);
    // v_rel = wind(0) - velocity = (3,0,0) - (-1e9,0,0) = (1e9 + 3, 0, 0).
    let v_rel_extreme = 1.0e9 + 3.0;
    let mag_extreme = 0.5 * 1.225 * 1.2 * 0.5 * v_rel_extreme * v_rel_extreme;
    check(
        "force_on(extreme_velocity).x",
        f_extreme.x.to_f64(),
        mag_extreme,
    );

    // ----------------------------------------------------------------
    // 4. force_on_particle — same math, particle position/velocity instead
    //    of a RigidBody.
    // ----------------------------------------------------------------
    let f_particle_resting = breeze.force_on_particle(Vec3Fix::ZERO, Vec3Fix::ZERO, Fix128::ZERO);
    check(
        "force_on_particle(resting, breeze, t=0).x",
        f_particle_resting.x.to_f64(),
        expected_mag_a,
    );

    let f_particle_triangle = storm.force_on_particle(
        Vec3Fix::ZERO,
        Vec3Fix::new(Fix128::from_int(17), Fix128::from_int(-4), Fix128::ZERO),
        Fix128::ZERO,
    );
    check(
        "force_on_particle(triangle, storm, t=0).x",
        f_particle_triangle.x.to_f64(),
        3.0 * (mag_b / 5.0),
    );
    check(
        "force_on_particle(triangle, storm, t=0).y",
        f_particle_triangle.y.to_f64(),
        4.0 * (mag_b / 5.0),
    );

    // Degenerate: particle outside the zone -> zero.
    let f_particle_outside =
        breeze.force_on_particle(Vec3Fix::from_int(10_000, 0, 0), Vec3Fix::ZERO, Fix128::ZERO);
    check(
        "force_on_particle(outside_zone).x",
        f_particle_outside.x.to_f64(),
        0.0,
    );

    println!("[wind_zone] all checks passed");
}
