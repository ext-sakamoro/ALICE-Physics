//! Independent oracles for `rotor` (static thrust law and hover momentum
//! theory).
//!
//! The existing file (`analytic_rotor.rs`) uses D 0.3 m, C_T 0.1, C_Q 0.005,
//! ρ 1.225 and checks each formula as written. Here a 10-inch propeller
//! (D 0.254 m, C_T 0.12, C_Q 0.0083) and other densities are used, and the
//! formulas are combined into quantities the module does not compute:
//!
//! - Figure of merit `FM = P_ideal / P_shaft = T v_i / (2π n Q)`, which for
//!   the static law reduces to `C_T^{3/2} √(2/π) / (2π C_Q)`, independent of
//!   `n`, `ρ` and `D`.
//! - Actuator-disk balance: far-wake velocity `w = 2 v_i`, so `T = ṁ w =
//!   2 ρ A v_i²` and `P = ½ ṁ w² = 2 ρ A v_i³`.
//! - Diameter scaling `T ∝ D⁴`, `Q ∝ D⁵`, `A ∝ D²`.
//! - Hover speed at altitude through `atmosphere::Isa1976`: `n ∝ ρ^{−1/2}`.
//! - Moments on a rotated body with an offset hub.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 `sqrt` computes references outside the crate.
#![allow(clippy::disallowed_methods)]

use std::f64::consts::PI;

use alice_physics::atmosphere::Isa1976;
use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rotor::{
    hover_induced_velocity_m_s, ideal_hover_power_w, Rotor, RotorParams, RotorSpin,
};
use alice_physics::solver::RigidBody;

const D: f64 = 0.254;
const CT: f64 = 0.12;
const CQ: f64 = 0.0083;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn v3(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(fx(x), fx(y), fx(z))
}

fn rel_close(label: &str, actual: f64, expected: f64, tol: f64) {
    let err = ((actual - expected) / expected).abs();
    assert!(
        err <= tol,
        "{label}: {actual} vs {expected} (rel err {err:e} > {tol:e})"
    );
}

fn prop(d: f64, ct: f64, cq: f64) -> Rotor {
    Rotor::new(RotorParams {
        diameter_m: fx(d),
        thrust_coefficient: fx(ct),
        torque_coefficient: fx(cq),
        thrust_axis_local: Vec3Fix::UNIT_Y,
        hub_position_local: Vec3Fix::ZERO,
        spin: RotorSpin::RightHanded,
    })
    .expect("valid rotor")
}

#[test]
fn figure_of_merit_is_a_function_of_the_coefficients_only() {
    // oracle: FM = C_T^{3/2} √(2/π) / (2π C_Q) ≈ 0.7975 for C_T 0.12, C_Q 0.0083.
    let fm_expected = CT.powf(1.5) * (2.0 / PI).sqrt() / (2.0 * PI * CQ);
    for (d, n, rho) in [(D, 47.0, 1.0), (D, 133.0, 0.7), (0.61, 21.0, 1.18)] {
        let r = prop(d, CT, CQ);
        let t = r.thrust_n(fx(rho), fx(n)).unwrap();
        let q = r.shaft_torque_nm(fx(rho), fx(n)).unwrap().to_f64();
        let p_ideal = ideal_hover_power_w(t, fx(rho), r.disk_area_m2())
            .unwrap()
            .to_f64();
        let p_shaft = 2.0 * PI * n * q;
        rel_close(
            &format!("FM (D {d}, n {n}, ρ {rho})"),
            p_ideal / p_shaft,
            fm_expected,
            1e-9,
        );
    }
}

#[test]
fn actuator_disk_wake_balance() {
    // oracle: T = 2 ρ A v_i² and P = 2 ρ A v_i³ (far wake at 2 v_i).
    let r = prop(D, CT, CQ);
    let a = r.disk_area_m2().to_f64();
    let rho = 1.06;
    for t in [0.4, 3.3, 11.0] {
        let vi = hover_induced_velocity_m_s(fx(t), fx(rho), r.disk_area_m2())
            .unwrap()
            .to_f64();
        rel_close(
            &format!("T from v_i ({t})"),
            2.0 * rho * a * vi * vi,
            t,
            1e-9,
        );
        let p = ideal_hover_power_w(fx(t), fx(rho), r.disk_area_m2())
            .unwrap()
            .to_f64();
        rel_close(
            &format!("P from v_i ({t})"),
            p,
            2.0 * rho * a * vi.powi(3),
            1e-9,
        );
    }
}

#[test]
fn diameter_scaling_laws() {
    let small = prop(0.13, CT, CQ);
    let large = prop(0.26, CT, CQ);
    let (rho, n) = (fx(1.1), fx(90.0));
    let ratio = |a: Fix128, b: Fix128| a.to_f64() / b.to_f64();
    rel_close(
        "T(2D)/T(D)",
        ratio(
            large.thrust_n(rho, n).unwrap(),
            small.thrust_n(rho, n).unwrap(),
        ),
        16.0,
        1e-9,
    );
    rel_close(
        "Q(2D)/Q(D)",
        ratio(
            large.shaft_torque_nm(rho, n).unwrap(),
            small.shaft_torque_nm(rho, n).unwrap(),
        ),
        32.0,
        1e-9,
    );
    rel_close(
        "A(2D)/A(D)",
        ratio(large.disk_area_m2(), small.disk_area_m2()),
        4.0,
        1e-12,
    );
}

#[test]
fn hover_speed_at_altitude_scales_with_inverse_root_density() {
    // oracle: n = √(m g / (C_T ρ D⁴)) ⇒ n(H)/n(0) = √(ρ(0)/ρ(H)).
    let r = prop(D, CT, CQ);
    let weight = fx(0.35 * 9.80665);
    let rho0 = Isa1976::at_geopotential_altitude(Fix128::ZERO)
        .unwrap()
        .density_kg_m3;
    let rho3 = Isa1976::at_geopotential_altitude(Fix128::from_int(3_000))
        .unwrap()
        .density_kg_m3;
    let n0 = r.speed_for_thrust(weight, rho0).unwrap();
    let n3 = r.speed_for_thrust(weight, rho3).unwrap();
    rel_close(
        "n(3 km)/n(0)",
        n3.to_f64() / n0.to_f64(),
        (rho0.to_f64() / rho3.to_f64()).sqrt(),
        1e-9,
    );
    // Round trip: the speed found gives the weight back.
    rel_close(
        "T(n(W))",
        r.thrust_n(rho3, n3).unwrap().to_f64(),
        weight.to_f64(),
        1e-9,
    );
}

#[test]
fn thrust_is_strictly_increasing_in_speed() {
    let r = prop(D, CT, CQ);
    let mut prev = r.thrust_n(fx(1.0), Fix128::ZERO).unwrap();
    for n in 1..=200 {
        let t = r.thrust_n(fx(1.0), Fix128::from_int(n)).unwrap();
        assert!(t > prev, "thrust not increasing at n = {n}");
        prev = t;
    }
}

#[test]
fn moments_on_a_rotated_body_with_offset_hub() {
    // Body turned +90° about z: local +y (thrust axis) → world −x, local hub
    // (0.3, 0, 0) → world (0, 0.3, 0). Left-handed spin: τ_reaction = +Q â.
    // r × F = (0, 0.3, 0) × (−T, 0, 0) = (0, 0, 0.3 T).
    let mut p = RotorParams {
        diameter_m: fx(D),
        thrust_coefficient: fx(CT),
        torque_coefficient: fx(CQ),
        thrust_axis_local: Vec3Fix::UNIT_Y,
        hub_position_local: v3(0.3, 0.0, 0.0),
        spin: RotorSpin::LeftHanded,
    };
    let r = Rotor::new(p).unwrap();
    let mut body = RigidBody::new(v3(5.0, -2.0, 1.0), Fix128::ONE);
    body.set_rotation(QuatFix::from_axis_angle(Vec3Fix::UNIT_Z, Fix128::HALF_PI));
    let (rho, n) = (1.0, 70.0);
    let t = CT * rho * n * n * D.powi(4);
    let q = CQ * rho * n * n * D.powi(5);
    let load = r.load(&body, fx(rho), fx(n)).unwrap();
    let got = [
        load.torque.x.to_f64(),
        load.torque.y.to_f64(),
        load.torque.z.to_f64(),
    ];
    let expected = [-q, 0.0, 0.3 * t];
    for i in 0..3 {
        assert!(
            (got[i] - expected[i]).abs() < 1e-9 * (1.0 + t),
            "torque[{i}] {} vs {}",
            got[i],
            expected[i]
        );
    }
    let f = [load.force.x.to_f64(), load.force.y.to_f64()];
    assert!((f[0] + t).abs() < 1e-9 * t && f[1].abs() < 1e-9 * t);
    let ap = load.application_point;
    assert!(
        (ap.x.to_f64() - 5.0).abs() < 1e-12 && (ap.y.to_f64() + 1.7).abs() < 1e-12,
        "application point {:?}",
        [ap.x.to_f64(), ap.y.to_f64(), ap.z.to_f64()]
    );
    // Right-handed flips only the reaction torque.
    p.spin = RotorSpin::RightHanded;
    let rh = Rotor::new(p).unwrap().load(&body, fx(rho), fx(n)).unwrap();
    assert_eq!(rh.force, load.force);
    // Opposite sign; equal up to one raw unit (Fix128 products round toward
    // −∞, so (−a)·b and −(a·b) may differ by 2⁻⁶⁴).
    let sum = rh.reaction_torque + load.reaction_torque;
    for c in [sum.x, sum.y, sum.z] {
        assert!(
            c.abs() <= Fix128::from_raw(0, 2),
            "reaction torques not opposite: {c:?}"
        );
    }
    assert!(rh.reaction_torque.x > Fix128::ZERO && load.reaction_torque.x < Fix128::ZERO);
}

#[test]
fn zero_thrust_coefficient_has_no_hover_speed() {
    // Documented: C_T = 0 with a positive target returns ZERO.
    let r = prop(D, 0.0, CQ);
    assert_eq!(r.speed_for_thrust(fx(2.0), fx(1.2)), Ok(Fix128::ZERO));
    assert_eq!(r.thrust_n(fx(1.2), fx(300.0)), Ok(Fix128::ZERO));
    assert!(r.shaft_torque_nm(fx(1.2), fx(300.0)).unwrap() > Fix128::ZERO);
}
