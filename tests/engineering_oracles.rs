//! Engineering-module oracles (1.2.0, first batch: transient thermal + fatigue).
//!
//! "Deterministic" and "correct" are different properties: the engineering
//! modules were bit-exact across platforms while nothing checked them
//! against a known solution. Each test here compares a module against a
//! textbook closed form (Carslaw & Jaeger for heat conduction, Basquin /
//! Miner for fatigue) with the tolerance of the discretisation stated in the
//! assertion. Modules without an oracle are listed as `validation: none` in
//! README § Determinism scope; this file grows one module at a time.

#![cfg(feature = "std")]
// The oracle values are closed-form f64 evaluations, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::fatigue::{miner_damage, SnCurve};
use alice_physics::math::Fix128;
use alice_physics::transient_thermal::{
    crank_nicolson_step_1d, transient_step_1d, TemperatureDependence, ThermalMaterial,
};

/// Constant-property material so the heat equation is linear and the
/// eigenmode solution is exact: α = k / (ρ cp) = 1e-5 m²/s.
fn constant_material() -> ThermalMaterial {
    ThermalMaterial {
        name: "oracle_constant",
        conductivity: TemperatureDependence::Constant(40.0), // W/(m K)
        specific_heat: TemperatureDependence::Constant(500.0), // J/(kg K)
        density: TemperatureDependence::Constant(8000.0),    // kg/m³
        reference_temperature: 293.15,
    }
}

/// Rod with zero-flux ends, initial profile `T(x, 0) = T0 + A cos(π x / L)`.
/// The cosine is a Neumann eigenmode, so
/// `T(x, t) = T0 + A cos(π x / L) exp(−α π² t / L²)` exactly (Carslaw &
/// Jaeger 1959, §3.4). Finite-volume convention of the crate (1.2.0, both
/// the explicit and the Crank–Nicolson step): every cell is physical, the
/// zero-flux faces are the outer edges of cells 0 and n−1, `L = n·dx`, cell
/// centres at `x_i = (i + ½) dx`.
fn cosine_mode(n: usize, dx: f64, t0: f64, amp: f64) -> Vec<f32> {
    let l = n as f64 * dx;
    (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            (t0 + amp * (core::f64::consts::PI * x / l).cos()) as f32
        })
        .collect()
}

fn max_abs_error(grid: &[f32], dx: f64, t0: f64, amp: f64, alpha: f64, t: f64) -> f64 {
    let n = grid.len();
    let l = n as f64 * dx;
    let decay = (-alpha * core::f64::consts::PI * core::f64::consts::PI * t / (l * l)).exp();
    grid.iter()
        .enumerate()
        .map(|(i, &v)| {
            let x = (i as f64 + 0.5) * dx;
            let want = t0 + amp * (core::f64::consts::PI * x / l).cos() * decay;
            (f64::from(v) - want).abs()
        })
        .fold(0.0, f64::max)
}

fn interior_mean(grid: &[f32]) -> f64 {
    grid.iter().map(|&v| f64::from(v)).sum::<f64>() / grid.len() as f64
}

#[test]
fn transient_thermal_explicit_euler_matches_cosine_eigenmode_decay() {
    let material = constant_material();
    let alpha = f64::from(material.diffusivity_at(300.0));
    assert!((alpha - 1e-5).abs() < 1e-9, "α = {alpha}");
    let (n, dx, t0, amp) = (64usize, 1e-3f64, 300.0f64, 50.0f64);
    let mut grid = cosine_mode(n, dx, t0, amp);
    // explicit Euler stability: dt <= dx² / (2α); use r = 0.25 → dt = 25 ms
    let dt = 0.25 * dx * dx / alpha;
    let steps = 400; // t = 10 s → decay exp(−α π² t / L²) = exp(−0.241) = 0.786
    for _ in 0..steps {
        transient_step_1d(&mut grid, &material, dx as f32, dt as f32);
    }
    let t = steps as f64 * dt;
    let err = max_abs_error(&grid, dx, t0, amp, alpha, t);
    // discrete cosine mode decays as (1 − 4r sin²(π dx / 2L))^k vs exp(−α π² t / L²):
    // 0.2 % of the 39 K remaining amplitude → < 0.2 K on a 50 K mode
    assert!(err < 0.2, "explicit Euler max error {err} K after {t} s");
    // energy (mean temperature) is conserved by the zero-flux ends
    let mean = interior_mean(&grid);
    assert!(
        (mean - t0).abs() < 0.05,
        "mean temperature drifted to {mean}"
    );
}

#[test]
fn transient_thermal_crank_nicolson_matches_cosine_eigenmode_decay() {
    let material = constant_material();
    let alpha = f64::from(material.diffusivity_at(300.0));
    let (n, dx, t0, amp) = (64usize, 1e-3f64, 300.0f64, 50.0f64);
    let mut grid = cosine_mode(n, dx, t0, amp);
    // unconditionally stable: take r = 2 (8× the explicit limit), dt = 0.2 s
    let dt = 2.0 * dx * dx / alpha;
    let steps = 50; // same t = 10 s
    for _ in 0..steps {
        crank_nicolson_step_1d(&mut grid, &material, dx as f32, dt as f32);
    }
    let t = steps as f64 * dt;
    let err = max_abs_error(&grid, dx, t0, amp, alpha, t);
    // second order in time: trapezoidal decay (1 − 2r s²)/(1 + 2r s²) vs exp, r = 2 → < 0.5 K
    assert!(err < 0.5, "Crank-Nicolson max error {err} K after {t} s");
    let mean = interior_mean(&grid);
    assert!(
        (mean - t0).abs() < 0.05,
        "mean temperature drifted to {mean}"
    );
}

/// The explicit and the Crank–Nicolson step must integrate the *same* rod:
/// before 1.2.0 the explicit step treated cells 0 / n−1 as ghost copies (rod
/// `(n−2)·dx`) while Crank–Nicolson used all n cells, so the two decayed the
/// same array at different rates.
#[test]
fn transient_thermal_explicit_and_crank_nicolson_agree_on_the_same_rod() {
    let material = constant_material();
    let alpha = f64::from(material.diffusivity_at(300.0));
    let (n, dx, t0, amp) = (32usize, 1e-3f64, 300.0f64, 50.0f64);
    let mut explicit = cosine_mode(n, dx, t0, amp);
    let mut cn = cosine_mode(n, dx, t0, amp);
    let dt = 0.1 * dx * dx / alpha;
    for _ in 0..500 {
        transient_step_1d(&mut explicit, &material, dx as f32, dt as f32);
        crank_nicolson_step_1d(&mut cn, &material, dx as f32, dt as f32);
    }
    let worst = explicit
        .iter()
        .zip(&cn)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 0.05,
        "explicit vs Crank-Nicolson differ by {worst} K on the same rod"
    );
}

#[test]
fn transient_thermal_uniform_field_is_a_fixed_point() {
    let material = constant_material();
    let mut grid = vec![350.0f32; 32];
    for _ in 0..100 {
        transient_step_1d(&mut grid, &material, 1e-3, 1e-2);
    }
    assert!(
        grid.iter().all(|&v| v == 350.0),
        "uniform field changed: {grid:?}"
    );
}

/// Basquin `N(S) = N_e (S_e / S)^m` and Miner `D = Σ n_i / N_i`.
#[test]
fn fatigue_miner_damage_matches_basquin_closed_form() {
    let curve = SnCurve {
        ultimate_tensile_mpa: Fix128::from_int(100),
        endurance_stress_mpa: Fix128::from_int(30),
        endurance_cycles: 1_000_000,
        fatigue_exponent_m: 5,
    };
    // S = 60 MPa → N = 1e6 · (30/60)^5 = 31 250 cycles
    let d_full = miner_damage(&[(Fix128::from_int(60), 31_250)], &curve).to_f64();
    assert!((d_full - 1.0).abs() < 1e-6, "D at N(S) = {d_full}");
    let d_half = miner_damage(&[(Fix128::from_int(60), 15_625)], &curve).to_f64();
    assert!((d_half - 0.5).abs() < 1e-6, "D at N/2 = {d_half}");
    // S = 90 MPa → N = 1e6 · (1/3)^5 = 4 115.2 cycles; 4115 cycles → D = 0.99995
    let d_90 = miner_damage(&[(Fix128::from_int(90), 4_115)], &curve).to_f64();
    let n_90 = 1e6 * (30.0f64 / 90.0).powi(5);
    assert!(
        (d_90 - 4_115.0 / n_90).abs() < 1e-3,
        "D(90 MPa, 4115) = {d_90} vs {}",
        4_115.0 / n_90
    );
    // two-level block loading: half life at each level → D = 1 (Miner linear sum)
    let d_block = miner_damage(
        &[
            (Fix128::from_int(60), 15_625),
            (Fix128::from_int(90), 2_057),
        ],
        &curve,
    )
    .to_f64();
    assert!(
        (d_block - (0.5 + 2_057.0 / n_90)).abs() < 1e-3,
        "block D = {d_block}"
    );
    // at or below the endurance limit: infinite life, zero damage
    assert_eq!(
        miner_damage(&[(Fix128::from_int(30), u64::MAX / 2)], &curve),
        Fix128::ZERO
    );
    assert_eq!(
        miner_damage(&[(Fix128::from_int(10), 1_000_000_000)], &curve),
        Fix128::ZERO
    );
}
