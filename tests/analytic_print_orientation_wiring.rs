//! Oracles for `print_orientation::{LoadDirection::axis_x, LoadDirection::axis_y, optimize_grid}`
//! (`examples/print_orientation_axes.rs`) and the angle / yield formulas they feed.
//!
//! * `effective_yield_at_angle(theta) = sigma_z cos^2 theta + sigma_xy sin^2 theta`
//!   with `sigma_xy = yield`, `sigma_z = yield * anisotropy_z_ratio`
//! * `angle_to_z_axis = acos(z_rot / |v|)` after `R_Y R_X` (matrices rebuilt here in f64)
//! * the grid optimum equals an independent f64 brute force over the same grid; a load already
//!   in the layer plane (X or Y) cannot be improved, a load along Z gains `sigma_xy - sigma_z`
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::print_orientation::{
    angle_to_z_axis, effective_yield_at_angle, optimize_analytical, optimize_grid, LoadDirection,
    OrientationCandidate,
};

fn fx(x: f64) -> Fix128 {
    Fix128::from_f64(x)
}
fn load(x: f64, y: f64, z: f64) -> LoadDirection {
    LoadDirection {
        x: fx(x),
        y: fx(y),
        z: fx(z),
    }
}
fn sigmas(m: &MaterialProperties) -> (f64, f64) {
    let xy = m.yield_strength_mpa.to_f64();
    (xy, xy * m.anisotropy_z_ratio.to_f64())
}
fn angle_ref(l: (f64, f64, f64), tx: f64, ty: f64) -> f64 {
    let y1 = l.1 * tx.cos() - l.2 * tx.sin();
    let z1 = l.1 * tx.sin() + l.2 * tx.cos();
    let x2 = l.0 * ty.cos() + z1 * ty.sin();
    let z2 = -l.0 * ty.sin() + z1 * ty.cos();
    let len = (x2 * x2 + y1 * y1 + z2 * z2).sqrt();
    if len == 0.0 {
        0.0
    } else {
        (z2 / len).clamp(-1.0, 1.0).acos()
    }
}
fn yield_ref(m: &MaterialProperties, theta: f64) -> f64 {
    let (xy, z) = sigmas(m);
    z * theta.cos().powi(2) + xy * theta.sin().powi(2)
}

#[test]
fn axis_constructors_are_unit_basis_vectors() {
    let (x, y, z) = (
        LoadDirection::axis_x(),
        LoadDirection::axis_y(),
        LoadDirection::axis_z(),
    );
    assert_eq!((x.x, x.y, x.z), (Fix128::ONE, Fix128::ZERO, Fix128::ZERO));
    assert_eq!((y.x, y.y, y.z), (Fix128::ZERO, Fix128::ONE, Fix128::ZERO));
    assert_eq!((z.x, z.y, z.z), (Fix128::ZERO, Fix128::ZERO, Fix128::ONE));
    for a in [x, y, z] {
        assert_eq!(a.length_squared(), Fix128::ONE);
        assert_eq!(a.length(), Fix128::ONE);
    }
    let v = load(3.0, 4.0, 12.0);
    assert_eq!(v.length_squared(), Fix128::from_int(169));
    assert_eq!(v.length(), Fix128::from_int(13));
}

#[test]
fn effective_yield_follows_the_squared_cosine_rule() {
    for m in [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
    ] {
        for k in 0..=24 {
            let theta = std::f64::consts::PI * f64::from(k) / 24.0;
            let got = effective_yield_at_angle(&m, fx(theta)).to_f64();
            assert!(
                (got - yield_ref(&m, theta)).abs() < 1e-9,
                "{} k={k}",
                m.name
            );
            // the documented relation to MaterialProperties::yield_at_angle: theta_there = pi/2 - theta_here
            let there = m
                .yield_at_angle(fx(std::f64::consts::FRAC_PI_2 - theta))
                .to_f64();
            assert!(
                (got - there).abs() < 1e-9,
                "{} k={k}: {got} vs {there}",
                m.name
            );
        }
    }
}

#[test]
fn angle_to_z_matches_an_independent_rotation() {
    let loads = [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 3.0, 4.0),
        (1.0, 1.0, 1.0),
        (-2.0, 0.5, -1.5),
        (0.0, 1.0, 0.0),
    ];
    let angles = [0.0, 0.3, -0.7, 1.2, -1.5707963267948966, 1.5707963267948966];
    for &l in &loads {
        for &tx in &angles {
            for &ty in &angles {
                let c = OrientationCandidate {
                    theta_x: fx(tx),
                    theta_y: fx(ty),
                };
                let got = angle_to_z_axis(&load(l.0, l.1, l.2), &c).to_f64();
                let want = angle_ref(l, tx, ty);
                assert!(
                    (got - want).abs() < 1e-6,
                    "load {l:?} tx {tx} ty {ty}: {got} vs {want}"
                );
            }
        }
    }
    // closed forms: Y load under R_X(theta): pi/2 - theta; X load under R_Y(theta): pi/2 + theta
    let c = OrientationCandidate {
        theta_x: fx(0.4),
        theta_y: Fix128::ZERO,
    };
    assert!(
        (angle_to_z_axis(&LoadDirection::axis_y(), &c).to_f64()
            - (std::f64::consts::FRAC_PI_2 - 0.4))
            .abs()
            < 1e-9
    );
    let c = OrientationCandidate {
        theta_x: Fix128::ZERO,
        theta_y: fx(0.4),
    };
    assert!(
        (angle_to_z_axis(&LoadDirection::axis_x(), &c).to_f64()
            - (std::f64::consts::FRAC_PI_2 + 0.4))
            .abs()
            < 1e-9
    );
    // zero-length load has no direction
    assert_eq!(
        angle_to_z_axis(&load(0.0, 0.0, 0.0), &OrientationCandidate::IDENTITY),
        Fix128::ZERO
    );
}

#[test]
fn in_plane_loads_cannot_be_improved_and_z_loads_gain_the_full_anisotropy() {
    let step = Fix128::PI / Fix128::from_int(12);
    for m in [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
    ] {
        let (xy, z) = sigmas(&m);
        for l in [LoadDirection::axis_x(), LoadDirection::axis_y()] {
            let r = optimize_grid(&l, &m, step);
            assert!(
                (r.identity_yield_mpa.to_f64() - xy).abs() < 1e-9,
                "{}",
                m.name
            );
            assert!((r.effective_yield_mpa.to_f64() - xy).abs() < 1e-9);
            assert!(r.improvement_mpa.to_f64().abs() < 1e-9);
        }
        let r = optimize_grid(&LoadDirection::axis_z(), &m, step);
        assert!((r.identity_yield_mpa.to_f64() - z).abs() < 1e-9);
        assert!((r.effective_yield_mpa.to_f64() - xy).abs() < 1e-9);
        assert!((r.improvement_mpa.to_f64() - (xy - z)).abs() < 1e-9);
        assert!((r.angle_to_z_axis.to_f64() - std::f64::consts::FRAC_PI_2).abs() < 1e-6);
        // the reported best orientation reproduces the reported angle
        let a = angle_to_z_axis(&LoadDirection::axis_z(), &r.best_orientation);
        assert!((a - r.angle_to_z_axis).abs().to_f64() < 1e-9);
    }
}

#[test]
fn grid_optimum_equals_an_independent_brute_force() {
    let m = MaterialProperties::pla();
    for &(l, step_deg) in &[
        ((0.0, 1.0, 1.0), 15.0),
        ((0.0, 0.0, 1.0), 30.0),
        ((1.0, 2.0, 3.0), 15.0),
        ((-1.0, 0.5, 2.0), 10.0),
        ((0.3, -0.2, 0.9), 45.0),
        ((0.0, 0.0, 1.0), 20.0),
    ] {
        let step = f64::from(step_deg) * std::f64::consts::PI / 180.0;
        let r = optimize_grid(&load(l.0, l.1, l.2), &m, fx(step));
        // brute force over k * step - pi/2 (tolerating the last point at +pi/2 within rounding)
        let n = ((std::f64::consts::PI + 1e-9) / step).floor() as i32;
        let mut best = f64::MIN;
        for i in 0..=n {
            for j in 0..=n {
                let tx = -std::f64::consts::FRAC_PI_2 + f64::from(i) * step;
                let ty = -std::f64::consts::FRAC_PI_2 + f64::from(j) * step;
                best = best.max(yield_ref(&m, angle_ref(l, tx, ty)));
            }
        }
        assert!(
            (r.effective_yield_mpa.to_f64() - best).abs() < 1e-6,
            "load {l:?} step {step_deg}: {} vs {best}",
            r.effective_yield_mpa.to_f64()
        );
        // never worse than the identity, never better than the in-plane strength
        let (xy, _) = sigmas(&m);
        assert!(r.improvement_mpa >= Fix128::ZERO);
        assert!(r.effective_yield_mpa.to_f64() <= xy + 1e-9);
        // the winner is a real candidate
        let a = angle_to_z_axis(&load(l.0, l.1, l.2), &r.best_orientation);
        assert!((a - r.angle_to_z_axis).abs().to_f64() < 1e-9);
        assert!(
            (effective_yield_at_angle(&m, a) - r.effective_yield_mpa)
                .abs()
                .to_f64()
                < 1e-9
        );
    }
}

#[test]
fn coarse_grid_is_never_better_than_the_analytical_optimum() {
    let m = MaterialProperties::pla();
    for l in [
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
        (2.0, -1.0, 0.5),
    ] {
        let a = optimize_analytical(&load(l.0, l.1, l.2), &m);
        let (xy, _) = sigmas(&m);
        assert!(
            (a.effective_yield_mpa.to_f64() - xy).abs() < 1e-6,
            "analytical reaches sigma_xy for {l:?}"
        );
        for deg in [60.0f64, 30.0, 15.0] {
            let g = optimize_grid(
                &load(l.0, l.1, l.2),
                &m,
                fx(deg * std::f64::consts::PI / 180.0),
            );
            assert!(
                g.effective_yield_mpa <= a.effective_yield_mpa + Fix128::from_ratio(1, 1_000_000),
                "{l:?} {deg}"
            );
            assert_eq!(g.identity_yield_mpa, a.identity_yield_mpa);
        }
    }
}

#[test]
fn non_positive_step_falls_back_to_the_analytical_optimum() {
    let m = MaterialProperties::pla();
    let l = load(0.0, 1.0, 1.0);
    let a = optimize_analytical(&l, &m);
    for step in [Fix128::ZERO, Fix128::NEG_ONE, fx(-0.1)] {
        assert_eq!(optimize_grid(&l, &m, step), a);
    }
    // a step wider than the range evaluates only the first corner (-pi/2, -pi/2)
    let wide = optimize_grid(&LoadDirection::axis_z(), &m, Fix128::from_int(4));
    let c = OrientationCandidate {
        theta_x: -Fix128::HALF_PI,
        theta_y: -Fix128::HALF_PI,
    };
    assert_eq!(wide.best_orientation, c);
}

#[test]
fn ties_keep_the_first_candidate_in_scan_order() {
    // X load: every candidate with theta_y ~ 0 has z1 = 0 and z2 ~ 0 -> identical yields;
    // the scan visits theta_x = -pi/2 first, so that candidate must win (strict `>`)
    let m = MaterialProperties::pla();
    let r = optimize_grid(
        &LoadDirection::axis_x(),
        &m,
        Fix128::PI / Fix128::from_int(4),
    );
    assert!(
        r.best_orientation.theta_y.abs().to_f64() < 1e-15,
        "grid accumulates rounding, theta_y ~ 0"
    );
    assert_eq!(r.best_orientation.theta_x, -Fix128::HALF_PI);
}
