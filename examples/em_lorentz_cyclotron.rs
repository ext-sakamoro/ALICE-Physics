//! A charged body in uniform and dipole magnetic fields, driven through
//! `lorentz_force` and `lorentz_force_sum`.
//!
//! 1. Cyclotron motion: q = 1 C, m = 1 kg, B = 2 T, v = 3 m/s gives the gyro-radius
//!    `r = m v / (q B) = 1.5 m` (diameter 3 m) and the period `2 pi m / (q B) = pi s`.
//! 2. `lorentz_force_sum` over a uniform field plus a point charge equals the sum of the
//!    single-source forces, and the Coulomb part is `k q1 q2 / r^2` (Griffiths eq. 2.1).
//!
//! ```bash
//! cargo run --release --example em_lorentz_cyclotron --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::electromagnetic::{lorentz_force, lorentz_force_sum, ChargedBody, EmSource};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn main() {
    // 1. cyclotron orbit
    let field = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::from_int(0, 0, 2),
    };
    let charged = ChargedBody::new(0, Fix128::ONE);
    let mut body = RigidBody::new(Vec3Fix::ZERO, Fix128::ONE);
    body.velocity = Vec3Fix::from_int(3, 0, 0);
    let steps = 20_000;
    let dt = Fix128::from_f64(std::f64::consts::PI / f64::from(steps));
    let (mut y_min, mut y_max) = (0.0f64, 0.0f64);
    for _ in 0..steps {
        let f = lorentz_force(charged, &body, &field);
        body.velocity = body.velocity + f * dt;
        body.position = body.position + body.velocity * dt;
        let y = body.position.y.to_f64();
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    println!(
        "orbit diameter {:.4} m (2 m v / q B = 3), back at ({:.4}, {:.4}) after one period",
        y_max - y_min,
        body.position.x.to_f64(),
        body.position.y.to_f64()
    );
    assert!(((y_max - y_min) - 3.0).abs() < 0.02);

    // 2. superposition and Coulomb
    let probe = {
        let mut b = RigidBody::new(Vec3Fix::from_int(3, 4, 0), Fix128::ONE);
        b.velocity = Vec3Fix::from_int(1, 0, 0);
        b
    };
    let q = ChargedBody::new(1, Fix128::from_f64(-2e-6));
    let sources = [
        EmSource::Uniform {
            electric: Vec3Fix::ZERO,
            magnetic: Vec3Fix::from_int(0, 0, 1000),
        },
        EmSource::PointCharge {
            position: Vec3Fix::ZERO,
            charge_c: Fix128::from_f64(3e-6),
        },
    ];
    let total = lorentz_force_sum(q, &probe, &sources);
    let parts = lorentz_force(q, &probe, &sources[0]) + lorentz_force(q, &probe, &sources[1]);
    assert_eq!(total, parts);
    let coulomb: f64 = 8.99e9 * 3e-6 * -2e-6 / 25.0;
    let from_source = lorentz_force(q, &probe, &sources[1]);
    let mag = (from_source.x.to_f64().powi(2) + from_source.y.to_f64().powi(2)).sqrt();
    println!(
        "Coulomb |F| = {mag:.6e} N (k q1 q2 / r^2 = {:.6e})",
        coulomb.abs()
    );
    assert!((mag - coulomb.abs()).abs() / coulomb.abs() < 1e-9);
}
