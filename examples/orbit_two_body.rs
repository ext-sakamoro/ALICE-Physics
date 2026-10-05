//! Two-body orbit laws from `kepler`: an orbit given by its classical
//! elements is turned into a state vector, propagated in time with Kepler's
//! equation, read back into elements, and checked against vis-viva, the
//! energy `−μ/(2a)` and the angular momentum `√(μa(1−e²))`. The secular J2
//! drift of the node is evaluated for a sun-synchronous orbit.
//!
//! Units are km and s; `μ`, `J2` and the equatorial radius below are values
//! chosen for the example (the crate carries no body constants).
//!
//! ```bash
//! cargo run --release --example orbit_two_body
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::kepler::{
    eccentric_from_true_anomaly, j2_arg_periapsis_rate, j2_raan_rate, mean_from_true_anomaly,
    mean_motion, orbital_period, solve_kepler, specific_angular_momentum, specific_orbital_energy,
    true_from_eccentric_anomaly, true_from_mean_anomaly, vis_viva_speed, OrbitalElements,
};
use alice_physics::math::Fix128;

fn deg(x: f64) -> Fix128 {
    Fix128::from_f64(x.to_radians())
}

fn main() {
    let mu = Fix128::from_f64(398_600.441_8);

    // Kepler's equation (Vallado Example 2-1: M = 235.4°, e = 0.4 → E = 220.512°)
    let e_anom = solve_kepler(deg(235.4), Fix128::from_ratio(2, 5)).unwrap();
    println!(
        "Kepler: M = 235.4°, e = 0.4 -> E = {:.9}°",
        e_anom.to_f64().to_degrees()
    );

    // an eccentric, inclined orbit
    let el = OrbitalElements::new(
        Fix128::from_int(12_000),
        Fix128::from_ratio(3, 10),
        deg(51.6),
        deg(40.0),
        deg(80.0),
        deg(10.0),
    )
    .unwrap();
    let period = el.period(mu).unwrap();
    println!(
        "a = 12000 km, e = 0.3: T = {:.3} s (n = {:.6e} rad/s, p = {:.3} km)",
        period.to_f64(),
        mean_motion(mu, el.semi_major_axis).unwrap().to_f64(),
        el.semi_latus_rectum().to_f64()
    );
    assert_eq!(period, orbital_period(mu, el.semi_major_axis).unwrap());

    // anomaly round trip ν → E → ν and ν → M → ν
    let e_from_nu = eccentric_from_true_anomaly(el.true_anomaly, el.eccentricity).unwrap();
    let nu_back = true_from_eccentric_anomaly(e_from_nu, el.eccentricity).unwrap();
    let m = mean_from_true_anomaly(el.true_anomaly, el.eccentricity).unwrap();
    assert_eq!(m, el.mean_anomaly().unwrap());
    let nu_from_m = true_from_mean_anomaly(m, el.eccentricity).unwrap();
    println!(
        "ν = 10°: E = {:.6}°, ν(E) = {:.6}°, M = {:.6}°, ν(M) = {:.6}°",
        e_from_nu.to_f64().to_degrees(),
        nu_back.to_f64().to_degrees(),
        m.to_f64().to_degrees(),
        nu_from_m.to_f64().to_degrees()
    );

    // state vector, half a period later, and back to elements
    let s0 = el.to_state(mu).unwrap();
    let half = el.propagate(mu, period.half()).unwrap();
    let s1 = el.state_at(mu, period.half()).unwrap();
    assert_eq!(s1, half.to_state(mu).unwrap());
    let back = OrbitalElements::from_state(&s1, mu).unwrap();
    let r1 = s1.position.length();
    let v1 = s1.velocity.length();
    println!(
        "t = 0:   r = {:.3} km, v = {:.6} km/s",
        s0.position.length().to_f64(),
        s0.velocity.length().to_f64()
    );
    println!(
        "t = T/2: r = {:.3} km, v = {:.6} km/s (vis-viva {:.6}), ν = {:.6}°",
        r1.to_f64(),
        v1.to_f64(),
        vis_viva_speed(mu, r1, el.semi_major_axis).unwrap().to_f64(),
        back.true_anomaly.to_f64().to_degrees()
    );
    let energy = v1 * v1 / Fix128::from_int(2) - mu / r1;
    let h = s1.position.cross(s1.velocity).length();
    println!(
        "energy {:.9} (−μ/2a = {:.9}), h {:.6} (√(μa(1−e²)) = {:.6}) km²/s",
        energy.to_f64(),
        specific_orbital_energy(mu, el.semi_major_axis)
            .unwrap()
            .to_f64(),
        h.to_f64(),
        specific_angular_momentum(mu, el.semi_major_axis, el.eccentricity)
            .unwrap()
            .to_f64()
    );
    let full = el.state_at(mu, period).unwrap();
    println!(
        "after one period |Δr| = {:.3e} km",
        (full.position - s0.position).length().to_f64()
    );

    // J2: a 700 km sun-synchronous orbit at i = 98.19°
    let (a, r_eq, j2) = (
        Fix128::from_f64(7078.137),
        Fix128::from_f64(6378.137),
        Fix128::from_f64(1.082_63e-3),
    );
    let node = j2_raan_rate(mu, a, Fix128::ZERO, deg(98.19), j2, r_eq).unwrap();
    let apse = j2_arg_periapsis_rate(mu, a, Fix128::ZERO, deg(98.19), j2, r_eq).unwrap();
    println!(
        "J2 at 700 km, i = 98.19°: dΩ/dt = {:.4}°/day (one turn per year = {:.4}°/day), dω/dt = {:.4}°/day",
        node.to_f64().to_degrees() * 86_400.0,
        360.0 / 365.2422,
        -(-apse).to_f64().to_degrees() * 86_400.0
    );
}
