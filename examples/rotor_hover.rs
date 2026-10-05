//! A rotor holding a body against gravity: thrust and torque laws,
//! momentum-theory hover power, and the reaction torque on the body.
//!
//! ```text
//! T = C_T ρ n² D⁴,  Q = C_Q ρ n² D⁵,  v_i = √(T / (2 ρ A)),  P = T v_i
//! ```
//!
//! The rotor speed for `T = m g` comes from `Rotor::speed_for_thrust`; the
//! closed forms are evaluated in `f64` and compared. Through `PhysicsWorld`
//! the vertical velocity stays 0 and the right-handed rotor spins the body
//! left-handed (negative `ω_y`). The altitude carries the frame/substep
//! split described in the `rotor` module docs.
//!
//! ```bash
//! cargo run --release --example rotor_hover
//! ```

// f64 `powi` computes a closed-form reference, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::atmosphere::Isa1976;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::rotor::{
    hover_induced_velocity_m_s, ideal_hover_power_w, Rotor, RotorParams, RotorSpin,
};
use alice_physics::solver::{PhysicsWorld, RigidBody, SolverConfig};

fn main() {
    let rho = Isa1976::at_geopotential_altitude(Fix128::ZERO)
        .expect("sea level")
        .density_kg_m3;
    let (d, ct, cq) = (0.3_f64, 0.1_f64, 0.005_f64);
    let rotor = Rotor::new(RotorParams {
        diameter_m: Fix128::from_f64(d),
        thrust_coefficient: Fix128::from_f64(ct),
        torque_coefficient: Fix128::from_f64(cq),
        thrust_axis_local: Vec3Fix::UNIT_Y,
        hub_position_local: Vec3Fix::ZERO,
        spin: RotorSpin::RightHanded,
    })
    .expect("valid rotor");

    let config = SolverConfig {
        damping: Fix128::ONE,
        ..SolverConfig::default()
    };
    let g = -config.gravity.y.to_f64();
    let mass = 1.5_f64;
    let weight = Fix128::from_f64(mass * g);

    let n = rotor.speed_for_thrust(weight, rho).expect("valid inputs");
    let n_ref = (mass * g / (ct * rho.to_f64() * d.powi(4))).sqrt();
    let thrust = rotor.thrust_n(rho, n).expect("valid inputs");
    let torque = rotor.shaft_torque_nm(rho, n).expect("valid inputs");
    let area = rotor.disk_area_m2();
    let v_i = hover_induced_velocity_m_s(thrust, rho, area).expect("valid inputs");
    let power = ideal_hover_power_w(thrust, rho, area).expect("valid inputs");
    println!(
        "[rotor_hover] D={} m n={:.4} rev/s (closed form {n_ref:.4}) T={:.4} N Q={:.5} N*m A={:.5} m2 v_i={:.4} m/s P={:.3} W",
        rotor.params().diameter_m.to_f64(),
        n.to_f64(),
        thrust.to_f64(),
        torque.to_f64(),
        area.to_f64(),
        v_i.to_f64(),
        power.to_f64()
    );
    assert!(((n.to_f64() - n_ref) / n_ref).abs() < 1e-9);
    assert!(((thrust.to_f64() - mass * g) / (mass * g)).abs() < 1e-9);

    let mut world = PhysicsWorld::new(config);
    let idx = world.add_body(RigidBody::new(
        Vec3Fix::from_int(0, 50, 0),
        Fix128::from_f64(mass),
    ));
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..600 {
        let b = world.get_body_mut(idx).expect("body");
        rotor.apply(b, rho, n, dt).expect("valid inputs");
        world.step(dt);
    }
    let b = world.get_body(idx).expect("body");
    let load = rotor.load(b, rho, n).expect("valid inputs");
    println!(
        "[rotor_hover] after 10 s: y={:.6} m v_y={:.3e} m/s omega_y={:.4} rad/s reaction torque_y={:.5} N*m",
        b.position.y.to_f64(),
        b.velocity.y.to_f64(),
        b.angular_velocity.y.to_f64(),
        load.reaction_torque.y.to_f64()
    );
    assert!(b.velocity.y.to_f64().abs() < 1e-9);
    assert!(b.angular_velocity.y < Fix128::ZERO);
}
