//! Closed-form oracle for an impulsive delta-v applied to an orbital state
//! and the resulting elements (COV-ORBIT-067). The state-vector -> elements
//! round trip (`OrbitalElements::from_state`) is already oracled elsewhere;
//! nothing applied a burn and checked the resulting periapsis/apoapsis
//! against vis-viva before this test (see docs/coverage/orbit.toml
//! COV-ORBIT-067 evidence).
//!
//! Starting from a circular orbit (radius `r0`, speed `v_circ =
//! sqrt(mu/r0)`), a purely tangential delta-v leaves the burn point's
//! velocity purely tangential, which makes `r0` an apsis of the resulting
//! orbit by definition (Vallado Sec.6.2): a prograde burn (`v1 = v_circ +
//! dv > v_circ`) makes `r0` the new periapsis, a retrograde burn (`v1 <
//! v_circ`) makes it the new apoapsis. The new semi-major axis follows
//! from vis-viva, `v1^2 = mu (2/r0 - 1/a)`, and the other apsis from
//! `r_peri + r_apo = 2a`.

use alice_physics::kepler::{vis_viva_speed, OrbitalElements, StateVector};
use alice_physics::{Fix128, Vec3Fix};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

const MU: f64 = 398_600.441_8; // Earth, km^3/s^2 (same convention as tests/analytic_kepler.rs)

#[test]
fn prograde_tangential_burn_raises_apoapsis_periapsis_matches_vis_viva() {
    let r0 = 7000.0_f64;
    let mu = fx(MU);
    let v_circ = (MU / r0).sqrt();
    let dv = 0.5_f64; // prograde: speeds up
    let v1 = v_circ + dv;

    let state = StateVector {
        position: Vec3Fix::new(fx(r0), Fix128::ZERO, Fix128::ZERO),
        velocity: Vec3Fix::new(Fix128::ZERO, fx(v1), Fix128::ZERO),
    };
    let elements = OrbitalElements::from_state(&state, mu).expect("elliptic orbit expected");

    // Closed form: 1/a = 2/r0 - v1^2/mu (vis-viva), independent of from_state.
    let a_expected = 1.0 / (2.0 / r0 - v1 * v1 / MU);
    let e_expected = 1.0 - r0 / a_expected; // r0 is periapsis: r_p = a(1-e)
    let r_apo_expected = 2.0 * a_expected - r0;

    let a_got = elements.semi_major_axis.to_f64();
    let e_got = elements.eccentricity.to_f64();
    assert!(
        (a_got - a_expected).abs() / a_expected < 1e-6,
        "semi-major axis: got {a_got} expected {a_expected}"
    );
    assert!(
        (e_got - e_expected).abs() < 1e-6,
        "eccentricity: got {e_got} expected {e_expected}"
    );

    // Cross-check via vis_viva_speed: speed at the (new) apoapsis distance
    // computed from a_got should match the closed-form apoapsis speed.
    let v_apo_closed_form = (MU * (2.0 / r_apo_expected - 1.0 / a_expected)).sqrt();
    let v_apo_from_crate =
        vis_viva_speed(mu, fx(r_apo_expected), elements.semi_major_axis).unwrap();
    let rel_err = (v_apo_from_crate.to_f64() - v_apo_closed_form).abs() / v_apo_closed_form;
    assert!(
        rel_err < 1e-9,
        "vis_viva_speed at apoapsis: got {} expected {v_apo_closed_form}",
        v_apo_from_crate.to_f64()
    );
}

#[test]
fn retrograde_tangential_burn_lowers_periapsis_matches_vis_viva() {
    let r0 = 7000.0_f64;
    let mu = fx(MU);
    let v_circ = (MU / r0).sqrt();
    let dv = 0.5_f64; // retrograde: slows down
    let v1 = v_circ - dv;

    let state = StateVector {
        position: Vec3Fix::new(fx(r0), Fix128::ZERO, Fix128::ZERO),
        velocity: Vec3Fix::new(Fix128::ZERO, fx(v1), Fix128::ZERO),
    };
    let elements = OrbitalElements::from_state(&state, mu).expect("elliptic orbit expected");

    let a_expected = 1.0 / (2.0 / r0 - v1 * v1 / MU);
    // r0 is apoapsis this time: r_a = a(1+e)
    let e_expected = r0 / a_expected - 1.0;

    let a_got = elements.semi_major_axis.to_f64();
    let e_got = elements.eccentricity.to_f64();
    assert!(
        (a_got - a_expected).abs() / a_expected < 1e-6,
        "semi-major axis: got {a_got} expected {a_expected}"
    );
    assert!(
        (e_got - e_expected).abs() < 1e-6,
        "eccentricity: got {e_got} expected {e_expected}"
    );
}
