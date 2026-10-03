//! Closed-form oracles for `electromagnetic::{lorentz_force, lorentz_force_sum}`.
//!
//! References (Griffiths, *Introduction to Electrodynamics*, 4th ed.):
//!
//! ```text
//! Lorentz      F = q (E + v x B)                                   eq. 5.1
//! Coulomb      E = k q / r^2 r_hat,  k = 8.99e9 V m / C            eq. 2.1
//! dipole       B = (mu0 / 4 pi) (3 (m . r_hat) r_hat - m) / r^3    eq. 5.87
//! gyration     r_c = m v / (q B),  omega_c = q B / m               s. 5.1.1
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::electromagnetic::{lorentz_force, lorentz_force_sum, ChargedBody, EmSource};
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::solver::RigidBody;

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn v3(v: Vec3Fix) -> [f64; 3] {
    [v.x.to_f64(), v.y.to_f64(), v.z.to_f64()]
}

fn fx(x: f64, y: f64, z: f64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_f64(x),
        Fix128::from_f64(y),
        Fix128::from_f64(z),
    )
}

fn body(pos: Vec3Fix, vel: Vec3Fix) -> RigidBody {
    let mut b = RigidBody::new(pos, Fix128::ONE);
    b.velocity = vel;
    b
}

const K: f64 = 8.99e9;
const MU0_4PI: f64 = 1e-7;

#[test]
fn coulomb_force_between_two_charges_follows_k_q1_q2_over_r_squared() {
    // source 3 uC at the origin, test charge -2 uC at (3, 4, 0): r = 5, attractive
    let q1 = 3e-6;
    let q2 = -2e-6;
    let src = EmSource::PointCharge {
        position: Vec3Fix::ZERO,
        charge_c: Fix128::from_f64(q1),
    };
    let b = body(Vec3Fix::from_int(3, 4, 0), Vec3Fix::ZERO);
    let f = v3(lorentz_force(
        ChargedBody::new(0, Fix128::from_f64(q2)),
        &b,
        &src,
    ));
    let mag = K * q1 * q2 / 25.0; // negative: attraction
    let want = [mag * 3.0 / 5.0, mag * 4.0 / 5.0, 0.0];
    for i in 0..3 {
        assert!(
            (f[i] - want[i]).abs() < 1e-9 * mag.abs(),
            "{i}: {} vs {}",
            f[i],
            want[i]
        );
    }
    // Newton's third law: the force on the source from the test charge is opposite
    let src2 = EmSource::PointCharge {
        position: Vec3Fix::from_int(3, 4, 0),
        charge_c: Fix128::from_f64(q2),
    };
    let b2 = body(Vec3Fix::ZERO, Vec3Fix::ZERO);
    let g = v3(lorentz_force(
        ChargedBody::new(1, Fix128::from_f64(q1)),
        &b2,
        &src2,
    ));
    for i in 0..3 {
        assert!(
            (f[i] + g[i]).abs() < 1e-9 * mag.abs(),
            "third law component {i}"
        );
    }
}

#[test]
fn dipole_field_matches_the_closed_form_on_axis_equator_and_obliquely() {
    let m = [0.0, 0.0, 2.0];
    let dipole = EmSource::MagneticDipole {
        position: Vec3Fix::ZERO,
        moment: Vec3Fix::from_int(0, 0, 2),
    };
    let q = Fix128::ONE;
    // a charge moving along +x through B feels F = q v x B; probe B through F with v = +x and v = +y
    for (px, py, pz) in [
        (0.0f64, 0.0, 0.5),
        (0.5, 0.0, 0.0),
        (0.3, 0.2, 0.4),
        (-0.4, 0.1, -0.2),
    ] {
        let r = (px * px + py * py + pz * pz).sqrt();
        let rh = [px / r, py / r, pz / r];
        let mdr = m[0] * rh[0] + m[1] * rh[1] + m[2] * rh[2];
        let b_exact = [
            MU0_4PI * (3.0 * mdr * rh[0] - m[0]) / (r * r * r),
            MU0_4PI * (3.0 * mdr * rh[1] - m[1]) / (r * r * r),
            MU0_4PI * (3.0 * mdr * rh[2] - m[2]) / (r * r * r),
        ];
        // v = x_hat: F = (0, -Bz*... ) from v x B = (0, -Bz, By) ; v = y_hat: (Bz, 0, -Bx)
        let fx_ = v3(lorentz_force(
            ChargedBody::new(0, q),
            &body(fx(px, py, pz), Vec3Fix::from_int(1, 0, 0)),
            &dipole,
        ));
        let fy_ = v3(lorentz_force(
            ChargedBody::new(0, q),
            &body(fx(px, py, pz), Vec3Fix::from_int(0, 1, 0)),
            &dipole,
        ));
        let scale = b_exact.iter().fold(0.0f64, |a, c| a.max(c.abs()));
        let tol = 1e-9 * scale + 1e-18;
        assert!(
            (fx_[1] + b_exact[2]).abs() < tol,
            "({px},{py},{pz}) By-from-F: {} vs {}",
            fx_[1],
            -b_exact[2]
        );
        assert!((fx_[2] - b_exact[1]).abs() < tol);
        assert!((fy_[0] - b_exact[2]).abs() < tol);
        assert!((fy_[2] + b_exact[0]).abs() < tol);
    }
    // on-axis magnitude is exactly twice the equatorial one, opposite in sign (eq. 5.87)
    let on_axis = v3(lorentz_force(
        ChargedBody::new(0, q),
        &body(Vec3Fix::from_int(0, 0, 1), Vec3Fix::from_int(0, 1, 0)),
        &dipole,
    ))[0];
    let equator = v3(lorentz_force(
        ChargedBody::new(0, q),
        &body(Vec3Fix::from_int(1, 0, 0), Vec3Fix::from_int(0, 1, 0)),
        &dipole,
    ))[0];
    assert!(rel(on_axis / equator, -2.0) < 1e-9, "{on_axis} / {equator}");
}

#[test]
fn force_is_linear_in_charge_and_flips_with_its_sign() {
    let src = EmSource::Uniform {
        electric: Vec3Fix::from_int(1, -2, 3),
        magnetic: Vec3Fix::from_int(0, 1, 2),
    };
    let b = body(Vec3Fix::ZERO, Vec3Fix::from_int(2, 1, -1));
    let f1 = lorentz_force(ChargedBody::new(0, Fix128::ONE), &b, &src);
    let f3 = lorentz_force(ChargedBody::new(0, Fix128::from_int(3)), &b, &src);
    let fm = lorentz_force(ChargedBody::new(0, Fix128::NEG_ONE), &b, &src);
    assert_eq!(f3, f1 * Fix128::from_int(3));
    assert_eq!(fm, Vec3Fix::ZERO - f1);
    assert_eq!(
        lorentz_force(ChargedBody::new(0, Fix128::ZERO), &b, &src),
        Vec3Fix::ZERO
    );
    // exact value: E + v x B with v = (2,1,-1), B = (0,1,2): v x B = (1*2 - (-1)*1, (-1)*0 - 2*2, 2*1 - 1*0) = (3, -4, 2)
    assert_eq!(f1, Vec3Fix::from_int(1 + 3, -2 - 4, 3 + 2));
}

#[test]
fn magnetic_force_is_perpendicular_to_velocity_and_vanishes_when_parallel() {
    let src = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::from_int(1, 2, 3),
    };
    let v = Vec3Fix::from_int(4, -1, 2);
    let f = lorentz_force(
        ChargedBody::new(0, Fix128::from_int(2)),
        &body(Vec3Fix::ZERO, v),
        &src,
    );
    assert_eq!(f.dot(v), Fix128::ZERO, "no work done");
    let par = lorentz_force(
        ChargedBody::new(0, Fix128::ONE),
        &body(Vec3Fix::ZERO, Vec3Fix::from_int(2, 4, 6)),
        &src,
    );
    assert_eq!(par, Vec3Fix::ZERO);
}

#[test]
fn sum_over_sources_is_the_sum_of_single_source_forces_in_any_order() {
    let sources = [
        EmSource::Uniform {
            electric: Vec3Fix::from_int(1, 0, 0),
            magnetic: Vec3Fix::from_int(0, 0, 1),
        },
        EmSource::PointCharge {
            position: Vec3Fix::from_int(0, 5, 0),
            charge_c: Fix128::from_f64(1e-4),
        },
        EmSource::MagneticDipole {
            position: Vec3Fix::from_int(0, 0, 4),
            moment: Vec3Fix::from_int(1, 1, 1),
        },
    ];
    let c = ChargedBody::new(0, Fix128::from_f64(0.01));
    let b = body(Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(3, 2, 1));
    let sum = lorentz_force_sum(c, &b, &sources);
    let manual = sources
        .iter()
        .fold(Vec3Fix::ZERO, |acc, s| acc + lorentz_force(c, &b, s));
    assert_eq!(sum, manual);
    let rev = [sources[2], sources[1], sources[0]];
    assert_eq!(
        lorentz_force_sum(c, &b, &rev),
        sum,
        "Fix128 addition is exact: order-free"
    );
    assert_eq!(
        lorentz_force_sum(c, &b, &[]),
        Vec3Fix::ZERO,
        "no sources, no force"
    );
}

#[test]
fn coincident_point_source_and_dipole_do_not_blow_up() {
    let c = ChargedBody::new(0, Fix128::ONE);
    let b = body(Vec3Fix::from_int(1, 2, 3), Vec3Fix::from_int(1, 0, 0));
    let pc = EmSource::PointCharge {
        position: b.position,
        charge_c: Fix128::ONE,
    };
    let dp = EmSource::MagneticDipole {
        position: b.position,
        moment: Vec3Fix::from_int(0, 0, 1),
    };
    assert_eq!(lorentz_force(c, &b, &pc), Vec3Fix::ZERO);
    assert_eq!(lorentz_force(c, &b, &dp), Vec3Fix::ZERO);
}

#[test]
fn cyclotron_orbit_has_the_gyro_radius_m_v_over_q_b() {
    // q = 1 C, m = 1 kg, B = 2 T (z), v = 3 m/s (x): r_c = 1.5 m, omega = 2 rad/s, T = pi s.
    // Integrate with a velocity-Verlet-like split: v += (F/m) dt, x += v dt.
    let src = EmSource::Uniform {
        electric: Vec3Fix::ZERO,
        magnetic: Vec3Fix::from_int(0, 0, 2),
    };
    let c = ChargedBody::new(0, Fix128::ONE);
    let mut b = body(Vec3Fix::ZERO, Vec3Fix::from_int(3, 0, 0));
    let steps = 20_000;
    let dt = Fix128::from_f64(std::f64::consts::PI / f64::from(steps));
    let (mut ymin, mut ymax) = (0.0f64, 0.0f64);
    for _ in 0..steps {
        let f = lorentz_force(c, &b, &src);
        b.velocity = b.velocity + f * dt;
        b.position = b.position + b.velocity * dt;
        let y = b.position.y.to_f64();
        ymin = ymin.min(y);
        ymax = ymax.max(y);
    }
    // v = +x, B = +z: v x B = -y, so the orbit turns toward -y; its diameter is 2 r_c = 3 m
    assert!(rel(ymax - ymin, 3.0) < 5e-3, "diameter {}", ymax - ymin);
    // after one period the body is back near the start with the speed it had
    let p = v3(b.position);
    assert!(
        p[0].abs() < 2e-2 && p[1].abs() < 2e-2,
        "back near origin: {p:?}"
    );
    assert!(
        rel(b.velocity.length().to_f64(), 3.0) < 5e-3,
        "speed conserved"
    );
}
