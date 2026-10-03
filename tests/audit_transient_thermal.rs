//! Audit oracles for `alice_physics::transient_thermal` (S2-2 audit).
//!
//! Expected values come from the semi-discrete cosine eigenmode (exact decay
//! rate of the finite-volume Neumann Laplacian), from a conservation integral
//! and from maximum-principle bounds, never from the update formulas in
//! `src/transient_thermal.rs`.
#![allow(clippy::disallowed_methods)]

use alice_physics::transient_thermal::{
    crank_nicolson_step_1d, crank_nicolson_step_1d_nonlinear, stable_dt_1d, stable_dt_3d,
    transient_step_1d, transient_step_3d, TemperatureDependence, ThermalMaterial,
};

/// alpha = k / (rho cp) = 1 exactly.
fn unit_material() -> ThermalMaterial {
    ThermalMaterial {
        name: "unit",
        conductivity: TemperatureDependence::Constant(1.0),
        specific_heat: TemperatureDependence::Constant(1.0),
        density: TemperatureDependence::Constant(1.0),
        reference_temperature: 300.0,
    }
}

/// Amplitude of the cos(pi (i + 1/2) / n) mode in a Neumann finite-volume rod.
fn amplitude(t: &[f32], n: usize) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, &v) in t.iter().enumerate() {
        let c = (core::f64::consts::PI * (i as f64 + 0.5) / n as f64).cos();
        num += (v as f64 - 300.0) * c;
        den += c * c;
    }
    num / den
}

fn mode(n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 300.0 + amp * (core::f64::consts::PI * (i as f64 + 0.5) / n as f64).cos() as f32)
        .collect()
}

/// Doc: Crank-Nicolson is "second-order accurate in time". Against the exact
/// semi-discrete decay exp(-lambda t) of the cosine mode, halving dt at fixed
/// final time cuts the error by ~4 for CN and ~2 for explicit Euler.
#[test]
fn crank_nicolson_is_second_order_and_explicit_euler_first_order_in_time() {
    let n = 32usize;
    let m = unit_material();
    let lam = 2.0 - 2.0 * (core::f64::consts::PI / n as f64).cos(); // dx = 1, alpha = 1
                                                                    // CN: r = 20 / 10 (total time 80), explicit: r = 0.4 / 0.2 (total time 80)
    let run_cn = |dt: f32, steps: usize| {
        let mut t = mode(n, 8.0);
        for _ in 0..steps {
            crank_nicolson_step_1d(&mut t, &m, 1.0, dt);
        }
        amplitude(&t, n)
    };
    let run_ex = |dt: f32, steps: usize| {
        let mut t = mode(n, 8.0);
        for _ in 0..steps {
            transient_step_1d(&mut t, &m, 1.0, dt);
        }
        amplitude(&t, n)
    };
    let exact = 8.0 * (-lam * 80.0).exp();
    let (e1, e2) = (
        (run_cn(20.0, 4) - exact).abs(),
        (run_cn(10.0, 8) - exact).abs(),
    );
    let ratio = e1 / e2;
    assert!(
        (3.0..=5.0).contains(&ratio),
        "CN error ratio {ratio} (errors {e1}, {e2}), want ~4"
    );
    let (x1, x2) = (
        (run_ex(0.4, 200) - exact).abs(),
        (run_ex(0.2, 400) - exact).abs(),
    );
    let ratio = x1 / x2;
    assert!(
        (1.6..=2.4).contains(&ratio),
        "explicit error ratio {ratio} (errors {x1}, {x2}), want ~2"
    );
}

/// Maximum principle: at dt = stable_dt_1d every update coefficient is a convex
/// weight, so no new value leaves [min, max] of the old field, even with strongly
/// temperature dependent properties (steel spans k = 50..30 W/mK across the rod).
#[test]
fn explicit_step_at_the_stable_dt_obeys_the_maximum_principle() {
    for mat in [
        ThermalMaterial::steel_1018(),
        ThermalMaterial::aluminum_6061(),
        ThermalMaterial::pla_polymer(),
        ThermalMaterial::titanium_ti6al4v(),
    ] {
        let mut t: Vec<f32> = (0..40)
            .map(|i| 300.0 + 700.0 * (((i * 37) % 11) as f32) / 10.0)
            .collect();
        let dx = 1.0e-3;
        let (lo, hi) = (
            t.iter().cloned().fold(f32::MAX, f32::min),
            t.iter().cloned().fold(f32::MIN, f32::max),
        );
        for _ in 0..50 {
            let dt = stable_dt_1d(&t, &mat, dx);
            transient_step_1d(&mut t, &mat, dx, dt);
            for &v in &t {
                assert!(
                    v >= lo - 1e-2 && v <= hi + 1e-2,
                    "{}: {v} outside [{lo}, {hi}]",
                    mat.name
                );
            }
        }
    }
}

/// 3-D: same property at stable_dt_3d (dx^2 / (6 alpha)).
#[test]
fn explicit_3d_step_at_the_stable_dt_obeys_the_maximum_principle() {
    let mat = ThermalMaterial::steel_1018();
    let (nx, ny, nz) = (6usize, 5usize, 4usize);
    let mut t: Vec<f32> = (0..nx * ny * nz)
        .map(|i| 300.0 + 500.0 * (((i * 13) % 7) as f32) / 6.0)
        .collect();
    let (lo, hi) = (
        t.iter().cloned().fold(f32::MAX, f32::min),
        t.iter().cloned().fold(f32::MIN, f32::max),
    );
    for _ in 0..30 {
        let dt = stable_dt_3d(&t, &mat, 1.0e-3);
        transient_step_3d(&mut t, nx, ny, nz, &mat, 1.0e-3, dt);
        for &v in &t {
            assert!(v >= lo - 1e-2 && v <= hi + 1e-2, "{v} outside [{lo}, {hi}]");
        }
    }
}

/// A field that varies only along x evolves identically (to f32 rounding) in the
/// 3-D and 1-D explicit steps.
#[test]
fn explicit_3d_step_reduces_to_the_1d_step_for_a_field_varying_only_in_x() {
    let mat = ThermalMaterial::aluminum_6061();
    let (nx, ny, nz) = (9usize, 4usize, 4usize);
    let line: Vec<f32> = (0..nx)
        .map(|i| 300.0 + 40.0 * ((i * 5) % 4) as f32)
        .collect();
    let mut t3 = vec![0.0f32; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t3[i + nx * (j + ny * k)] = line[i];
            }
        }
    }
    let dx = 1.0e-3;
    let dt = stable_dt_3d(&t3, &mat, dx);
    let mut t1 = line.clone();
    for _ in 0..5 {
        transient_step_1d(&mut t1, &mat, dx, dt);
        transient_step_3d(&mut t3, nx, ny, nz, &mat, dx, dt);
    }
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let (a, b) = (t3[i + nx * (j + ny * k)], t1[i]);
                assert!((a - b).abs() < 2e-3, "({i},{j},{k}): 3-D {a} vs 1-D {b}");
            }
        }
    }
}

/// The nonlinear Crank-Nicolson step solves the trapezoidal system with
/// alpha evaluated at the step-average temperature (T_old + T_new)/2:
/// T_new - T_old = r/2 [L T_new + L T_old], r = alpha(T_avg) dt / dx^2.
/// Checked as a residual on the converged result, with the diffusivity from the
/// public property accessor.
#[test]
fn nonlinear_crank_nicolson_result_satisfies_the_trapezoidal_residual() {
    let mat = ThermalMaterial::steel_1018();
    let n = 24usize;
    let old: Vec<f32> = (0..n)
        .map(|i| 300.0 + 400.0 * (-(((i as f32) - 12.0) / 3.0).powi(2)).exp())
        .collect();
    let (dx, dt) = (1.0e-3f32, 2.0e-3f32);
    let mut new = old.clone();
    let iters = crank_nicolson_step_1d_nonlinear(&mut new, &mat, dx, dt, 1e-6, 50);
    assert!(iters >= 1);
    let lap = |t: &[f32], i: usize| {
        let l = if i == 0 { t[0] } else { t[i - 1] };
        let r = if i == n - 1 { t[n - 1] } else { t[i + 1] };
        (l - 2.0 * t[i] + r) as f64
    };
    for i in 0..n {
        let avg = 0.5 * (old[i] + new[i]);
        let r = mat.diffusivity_at(avg) as f64 * dt as f64 / (dx as f64 * dx as f64);
        let lhs = (new[i] - old[i]) as f64;
        let rhs = 0.5 * r * (lap(&new, i) + lap(&old, i));
        assert!(
            (lhs - rhs).abs() < 2e-3 + 1e-3 * lhs.abs(),
            "cell {i}: dT {lhs} vs {rhs}"
        );
    }
}

/// Zero-flux ends mean no heat leaves the rod: with constant properties the sum of
/// T is conserved by every scheme. This is the control for the variable-property
/// test below.
#[test]
fn constant_property_schemes_conserve_the_temperature_sum() {
    let m = unit_material();
    let n = 20usize;
    let init: Vec<f32> = (0..n)
        .map(|i| 300.0 + ((i * 7) % 5) as f32 * 20.0)
        .collect();
    let sum0: f64 = init.iter().map(|&v| v as f64).sum();
    let mut a = init.clone();
    let mut b = init.clone();
    for _ in 0..40 {
        transient_step_1d(&mut a, &m, 1.0, 0.2);
        crank_nicolson_step_1d(&mut b, &m, 1.0, 5.0);
    }
    for (name, t) in [("explicit", &a), ("crank-nicolson", &b)] {
        let s: f64 = t.iter().map(|&v| v as f64).sum();
        assert!((s - sum0).abs() / sum0 < 1e-5, "{name}: {s} vs {sum0}");
    }
}

/// With temperature dependent k, the heat equation is rho cp dT/dt = d/dx(k dT/dx):
/// with zero-flux ends the first-order enthalpy change sum(rho cp dT) must vanish.
/// The module advances dT/dt = alpha(T) d2T/dx2 (non-divergence form), so
/// sum(rho cp dT) = dt/dx^2 sum(k_i L_i) does not telescope when k varies.
#[test]
#[ignore = "known defect: AUD-A-S2W2-015: transient_step_1d advances alpha(T_i) * d2T/dx2 (non-conservative form) instead of (1/(rho cp)) d/dx(k dT/dx): for PLA with a 470 K / 300 K step sum(rho cp dT) / sum|rho cp dT| = -5.04e-2 (not zero), so zero-flux ends do not conserve heat when k, rho or cp depend on T"]
fn explicit_step_conserves_enthalpy_with_temperature_dependent_conductivity() {
    let mat = ThermalMaterial::pla_polymer();
    let n = 40usize;
    // hot left half, cold right half: a front where k and alpha differ across cells
    let init: Vec<f32> = (0..n)
        .map(|i| if i < n / 2 { 470.0 } else { 300.0 })
        .collect();
    let dx = 1.0e-3f32;
    let dt = stable_dt_1d(&init, &mat, dx) * 0.5;
    let mut t = init.clone();
    transient_step_1d(&mut t, &mat, dx, dt);
    let mut net = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..n {
        let c = mat.heat_capacity_at(init[i]) as f64;
        let d = (t[i] - init[i]) as f64;
        net += c * d;
        scale += c * d.abs();
    }
    assert!(
        net.abs() / scale < 1e-3,
        "sum(rho cp dT) / sum|rho cp dT| = {}",
        net / scale
    );
}

/// Doc: diffusivity_at "guards against pathological polynomial evaluations outside
/// the calibrated range". Only rho cp <= 0 is guarded: the steel conductivity line
/// 60 - 0.03 T turns negative above 2000 K and alpha follows, i.e. anti-diffusion
/// (an explicit step then amplifies instead of smoothing).
#[test]
#[ignore = "known defect: AUD-A-S2W2-016: diffusivity_at guards only rho*cp <= 0; steel_1018 at 2500 K gives k = -15 W/mK and alpha < 0 (anti-diffusion), contradicting the doc's out-of-range guard; tests/analytic_transient_thermal_wiring.rs::beyond_calibrated_range_the_fits_extrapolate_without_clamp pins the negative value"]
fn diffusivity_is_never_negative_outside_the_calibrated_range() {
    let m = ThermalMaterial::steel_1018();
    for t in [100.0f32, 300.0, 1000.0, 1999.0, 2000.0, 2500.0, 3000.0] {
        assert!(
            m.diffusivity_at(t) >= 0.0,
            "alpha({t} K) = {}",
            m.diffusivity_at(t)
        );
    }
}

/// A non-finite diffusivity gives no usable CFL bound: both stable_dt functions
/// return INFINITY (pins the guard `max_alpha <= 0 || !finite`; the doc is silent).
#[test]
fn non_finite_diffusivity_gives_an_infinite_stable_dt() {
    let mut m = unit_material();
    m.conductivity = TemperatureDependence::Constant(f32::INFINITY);
    assert!(m.diffusivity_at(300.0).is_infinite());
    assert!(stable_dt_1d(&[300.0; 4], &m, 1.0e-3).is_infinite());
    assert!(stable_dt_3d(&[300.0; 8], &m, 1.0e-3).is_infinite());
}
