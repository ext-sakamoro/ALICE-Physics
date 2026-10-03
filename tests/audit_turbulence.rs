//! Audit oracle for the public surface of `turbulence`: Smagorinsky, strain-rate
//! magnitude, friction velocity. Expected values are hand closed forms / independent f64.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods, clippy::needless_range_loop)]

use alice_physics::math::Fix128;
use alice_physics::turbulence::{
    friction_velocity, smagorinsky_eddy_viscosity, strain_rate_magnitude, WallFunctionError,
    FRICTION_VELOCITY_BISECTIONS, SMAGORINSKY_CS,
};

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

#[test]
fn smagorinsky_cs_is_017_to_one_ulp() {
    // 0.17 * 2^64 = 3135924266.. floor/ceil neighbours only
    let want = 0.17_f64;
    assert_eq!(SMAGORINSKY_CS.hi, 0);
    assert!((SMAGORINSKY_CS.to_f64() - want).abs() < 1e-18);
}

#[test]
fn smagorinsky_is_cs2_delta2_strain_closed_form() {
    for (d, s) in [(0.01, 10.0), (0.5, 2.0), (2.0, 0.25), (0.1, 0.0)] {
        let nu = smagorinsky_eddy_viscosity(fx(d), fx(s)).to_f64();
        let want = (0.17 * d) * (0.17 * d) * s;
        assert!(
            (nu - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "d={d} s={s} nu={nu} want={want}"
        );
    }
    // linear in strain, quadratic in width (exact powers of two)
    let a = smagorinsky_eddy_viscosity(fx(1.0), fx(1.0));
    assert_eq!(smagorinsky_eddy_viscosity(fx(1.0), fx(2.0)), a + a);
    let q = smagorinsky_eddy_viscosity(fx(2.0), fx(1.0)).to_f64();
    assert!((q - 4.0 * a.to_f64()).abs() < 1e-15);
}

/// Simple shear u = g y: S12 = g/2, so |S| = sqrt(2*2*(g/2)^2) = g.
#[test]
fn strain_magnitude_of_simple_shear_is_the_shear_rate() {
    for g in [0.5, 3.0, 40.0] {
        let m = strain_rate_magnitude(fx(0.0), fx(0.0), fx(0.0), fx(g / 2.0), fx(0.0), fx(0.0));
        assert!((m.to_f64() - g).abs() < 1e-12, "g={g} m={}", m.to_f64());
        // each off-diagonal pair is counted the same way
        let m13 = strain_rate_magnitude(fx(0.0), fx(0.0), fx(0.0), fx(0.0), fx(g / 2.0), fx(0.0));
        let m23 = strain_rate_magnitude(fx(0.0), fx(0.0), fx(0.0), fx(0.0), fx(0.0), fx(g / 2.0));
        assert!((m13.to_f64() - g).abs() < 1e-12);
        assert!((m23.to_f64() - g).abs() < 1e-12);
    }
}

/// Diagonal-only: |S| = sqrt(2 (a^2+b^2+c^2)), each axis counted once.
#[test]
fn strain_magnitude_diagonal_closed_form() {
    let m = strain_rate_magnitude(fx(1.0), fx(2.0), fx(2.0), fx(0.0), fx(0.0), fx(0.0));
    assert!((m.to_f64() - (2.0_f64 * 9.0).sqrt()).abs() < 1e-12); // sqrt(18)
    let m = strain_rate_magnitude(fx(0.0), fx(3.0), fx(0.0), fx(0.0), fx(0.0), fx(0.0));
    assert!((m.to_f64() - (18.0_f64).sqrt()).abs() < 1e-12);
    let m = strain_rate_magnitude(fx(0.0), fx(0.0), fx(1.0), fx(0.0), fx(0.0), fx(0.0));
    assert!((m.to_f64() - 2.0_f64.sqrt()).abs() < 1e-12);
}

/// |S| = sqrt(2 S_ij S_ij) is invariant under rotation of the tensor (R S R^T), and sign-blind.
#[test]
fn strain_magnitude_is_rotation_invariant() {
    let s = [[0.7, -1.3, 0.4], [-1.3, 0.2, 2.1], [0.4, 2.1, -0.9]];
    // rotation about axis (1,2,2)/3 by 0.8 rad (Rodrigues)
    let (c, sn) = (0.8_f64.cos(), 0.8_f64.sin());
    let k = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    let kx = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]];
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            let kk: f64 = (0..3).map(|m| kx[i][m] * kx[m][j]).sum();
            r[i][j] = id + sn * kx[i][j] + (1.0 - c) * kk;
        }
    }
    let mut rs = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rs[i][j] = (0..3)
                .map(|a| (0..3).map(|b| r[i][a] * s[a][b] * r[j][b]).sum::<f64>())
                .sum();
        }
    }
    let mag = |t: &[[f64; 3]; 3]| {
        strain_rate_magnitude(
            fx(t[0][0]),
            fx(t[1][1]),
            fx(t[2][2]),
            fx(t[0][1]),
            fx(t[0][2]),
            fx(t[1][2]),
        )
        .to_f64()
    };
    let want: f64 = (2.0 * s.iter().flatten().map(|v| v * v).sum::<f64>()).sqrt();
    assert!((mag(&s) - want).abs() < 1e-9);
    assert!((mag(&rs) - want).abs() < 1e-9, "{} vs {}", mag(&rs), want);
    let neg = [[-0.7, 1.3, -0.4], [1.3, -0.2, -2.1], [-0.4, -2.1, 0.9]];
    assert!((mag(&neg) - want).abs() < 1e-9);
}

// ---- friction_velocity ----

const KAPPA: f64 = 0.41;
const B: f64 = 5.5;

fn y_trans() -> f64 {
    // root of y = ln y / kappa + B, Newton in f64
    let mut y = 11.0_f64;
    for _ in 0..60 {
        let f = y - (y.ln() / KAPPA + B);
        y -= f / (1.0 - 1.0 / (KAPPA * y));
    }
    y
}
fn u_plus(yp: f64) -> f64 {
    if yp < y_trans() {
        yp
    } else {
        yp.ln() / KAPPA + B
    }
}

#[test]
fn friction_velocity_satisfies_the_defining_equation_across_regimes() {
    // (u_rel, y_p, nu): y+ < 1, sublayer, buffer-ish, log, high Re
    let cases = [
        (0.001, 0.001, 1e-3),
        (0.05, 0.001, 1e-3),
        (0.5, 0.002, 1e-3),
        (1.0, 0.01, 1e-3),
        (10.0, 0.05, 1.5e-5),
        (40.0, 0.2, 1.5e-5),
    ];
    for (u, y, nu) in cases {
        let ut = friction_velocity(fx(u), fx(y), fx(nu)).unwrap().to_f64();
        let lhs = ut * u_plus(y * ut / nu);
        assert!(
            (lhs - u).abs() <= 1e-9 * u,
            "u={u} y={y} nu={nu} ut={ut} lhs={lhs}"
        );
        assert!(ut > 0.0);
    }
}

#[test]
fn friction_velocity_sublayer_closed_form_is_sqrt_nu_u_over_y() {
    // y+ = y ut/nu = sqrt(y u/nu) ~ 0.03.. well inside the sublayer
    for (u, y, nu) in [(0.2, 1e-4, 1e-3), (1.0, 1e-3, 5e-3), (0.01, 0.002, 1e-3)] {
        let ut = friction_velocity(fx(u), fx(y), fx(nu)).unwrap().to_f64();
        assert!((ut - (nu * u / y).sqrt()).abs() < 1e-9, "u={u}");
    }
}

#[test]
fn friction_velocity_is_monotone_in_speed_and_decreasing_in_wall_distance() {
    let (y, nu) = (0.01, 1.5e-5);
    let mut prev = 0.0;
    for u in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0] {
        let ut = friction_velocity(fx(u), fx(y), fx(nu)).unwrap().to_f64();
        assert!(ut > prev, "u={u}");
        prev = ut;
    }
    // same speed, farther from the wall -> smaller u_tau (log law: u+ grows with y+)
    let a = friction_velocity(fx(10.0), fx(0.01), fx(nu))
        .unwrap()
        .to_f64();
    let b = friction_velocity(fx(10.0), fx(0.1), fx(nu))
        .unwrap()
        .to_f64();
    assert!(b < a);
}

/// Scaling: u_tau = u g(y u / nu)  =>  f(c u, y / c, nu) = c f(u, y, nu).
#[test]
fn friction_velocity_obeys_reynolds_scaling() {
    for (u, y, nu) in [(3.0, 0.02, 1e-3), (12.0, 0.4, 1.5e-5), (0.1, 0.01, 1e-3)] {
        let a = friction_velocity(fx(u), fx(y), fx(nu)).unwrap().to_f64();
        let b = friction_velocity(fx(2.0 * u), fx(y / 2.0), fx(nu))
            .unwrap()
            .to_f64();
        assert!((b - 2.0 * a).abs() < 1e-9 * a.max(1e-9), "u={u}");
    }
}

#[test]
fn friction_velocity_errors_follow_check_order_and_zero_is_exact() {
    let ok = fx(1.0);
    assert_eq!(
        friction_velocity(ok, Fix128::ZERO, ok),
        Err(WallFunctionError::NonPositiveWallDistance)
    );
    assert_eq!(
        friction_velocity(ok, fx(-1.0), ok),
        Err(WallFunctionError::NonPositiveWallDistance)
    );
    assert_eq!(
        friction_velocity(ok, ok, Fix128::ZERO),
        Err(WallFunctionError::NonPositiveViscosity)
    );
    assert_eq!(
        friction_velocity(ok, ok, fx(-1e-3)),
        Err(WallFunctionError::NonPositiveViscosity)
    );
    assert_eq!(
        friction_velocity(fx(-0.5), ok, ok),
        Err(WallFunctionError::NegativeSpeed)
    );
    // both y_p and nu bad: wall distance reported first
    assert_eq!(
        friction_velocity(fx(-1.0), Fix128::ZERO, Fix128::ZERO),
        Err(WallFunctionError::NonPositiveWallDistance)
    );
    assert_eq!(
        friction_velocity(fx(-1.0), ok, Fix128::ZERO),
        Err(WallFunctionError::NonPositiveViscosity)
    );
    assert_eq!(friction_velocity(Fix128::ZERO, ok, ok), Ok(Fix128::ZERO));
    assert_eq!(FRICTION_VELOCITY_BISECTIONS, 64);
}

#[test]
fn wall_function_error_messages_name_the_offending_argument() {
    assert!(WallFunctionError::NonPositiveWallDistance
        .to_string()
        .contains("wall distance"));
    assert!(WallFunctionError::NonPositiveViscosity
        .to_string()
        .contains("viscosity"));
    assert!(WallFunctionError::NegativeSpeed
        .to_string()
        .contains("speed"));
    let e: &dyn std::error::Error = &WallFunctionError::NegativeSpeed;
    assert!(!e.to_string().is_empty());
}

/// Determinism: the bisection count is fixed, so the result is a pure function of the inputs.
#[test]
fn friction_velocity_is_bit_reproducible() {
    let a = friction_velocity(fx(7.3), fx(0.031), fx(1.2e-5)).unwrap();
    let b = friction_velocity(fx(7.3), fx(0.031), fx(1.2e-5)).unwrap();
    assert_eq!(a, b);
}
