//! Audit oracles for turbulence: the per-cell dynamic Smagorinsky coefficient
//! `C_s = 0.17 * ratio` (clamped to `[0.05, 0.25]`) with `ratio` the six-
//! neighbour mean of `|S|` over the cell's own `|S|`, measured where the ratio
//! is not one (AUD-C-S1W6-002).
//!
//! `dynamic_smagorinsky_cs` is crate-internal; it is reached through
//! `CfdSolver::step_rans` with `TurbulenceModel::DynamicSmagorinsky`, whose
//! report carries the coefficient envelope and the eddy viscosity field
//! `nu_t = (C_s dx)^2 |S|` per cell.
//!
//! Expected values are built by hand from the face velocities: on an
//! `nx x 1 x 1` grid the only strain component is `s11 = (u[i+1] - u[i]) / dx`
//! (the off-diagonal derivatives of a one-cell axis are zero), so
//! `|S| = sqrt(2 s11^2) = sqrt(2) |s11|`. A missing neighbour on the domain
//! edge counts as the cell itself.
//!
//! This pins the ratio form documented on
//! `TurbulenceModel::DynamicSmagorinsky`, not the Germano least-squares
//! procedure; AUD-A-S1W6-014 questions the direction of that mapping, and a
//! change of the documented form would change these expected values.

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, RansState, StepOptions, TurbulenceModel,
};
use alice_physics::math::{Fix128, Vec3Fix};

const CS: f64 = 0.17;

fn solver_with_faces(faces: &[i64]) -> CfdSolver {
    let nx = faces.len() - 1;
    let mut s = CfdSolver::new(nx, 1, 1, Fix128::ONE);
    s.gravity = Vec3Fix::ZERO;
    for (i, u) in faces.iter().enumerate() {
        s.grid.u[i] = Fix128::from_int(*u);
    }
    s
}

/// Six-neighbour mean of `strain` for an `nx x 1 x 1` grid: four of the six
/// neighbours (y and z) are missing and count as the cell itself.
fn neighbour_mean(strain: &[f64], i: usize) -> f64 {
    let me = strain[i];
    let left = if i > 0 { strain[i - 1] } else { me };
    let right = if i + 1 < strain.len() {
        strain[i + 1]
    } else {
        me
    };
    (left + right + 4.0 * me) / 6.0
}

fn expected_cs(strain: &[f64], i: usize) -> f64 {
    let ratio = neighbour_mean(strain, i) / strain[i];
    (CS * ratio).clamp(0.05, 0.25)
}

fn run(faces: &[i64]) -> (Vec<f64>, alice_physics::cfd_solver::RansReport) {
    let mut s = solver_with_faces(faces);
    let nx = faces.len() - 1;
    let strain: Vec<f64> = (0..nx)
        .map(|i| 2f64.sqrt() * ((faces[i + 1] - faces[i]) as f64).abs())
        .collect();
    let mut state = RansState::new(nx, 1, 1, TurbulenceModel::DynamicSmagorinsky);
    let opts = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 });
    let report = s
        .step_rans(Fix128::from_ratio(1, 1000), &opts, &mut state)
        .expect("steps");
    (strain, report)
}

// Pins today's behaviour; AUD-A-S1W6-014 questions this mapping (the current ratio-proportional form of `dynamic_smagorinsky_cs`)
#[test]
fn dynamic_coefficient_scales_with_the_neighbour_to_cell_strain_ratio() {
    // s11 = 1, 2, 3: ratios 7/6, 1 and 17/18, all inside the clamp
    // (0.17 * 7/6 = 0.19833, 0.17 * 17/18 = 0.16056).
    let faces = [0, 1, 3, 6];
    let (strain, report) = run(&faces);
    let cs: Vec<f64> = (0..3).map(|i| expected_cs(&strain, i)).collect();
    assert!((cs[0] - 0.17 * 7.0 / 6.0).abs() < 1e-15);
    assert!((cs[2] - 0.17 * 17.0 / 18.0).abs() < 1e-15);
    let want_min = cs.iter().cloned().fold(f64::INFINITY, f64::min);
    let want_max = cs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t = report.turbulence;
    assert!(
        (t.cs_min.to_f64() - want_min).abs() < 1e-12,
        "cs_min = {}, 0.17 * 17/18 = {want_min}",
        t.cs_min.to_f64()
    );
    assert!(
        (t.cs_max.to_f64() - want_max).abs() < 1e-12,
        "cs_max = {}, 0.17 * 7/6 = {want_max}",
        t.cs_max.to_f64()
    );
}

// Pins today's behaviour; AUD-A-S1W6-014 questions this mapping (the current ratio-proportional form of `dynamic_smagorinsky_cs`)
#[test]
fn dynamic_eddy_viscosity_is_cs_squared_dx_squared_times_strain_in_every_cell() {
    let faces = [0, 1, 3, 6];
    let (strain, report) = run(&faces);
    for i in 0..3 {
        let cs = expected_cs(&strain, i);
        let want = cs * cs * strain[i];
        let got = report.eddy_viscosity.data[i].to_f64();
        assert!(
            (got - want).abs() < 1e-12 * want.max(1.0),
            "nu_t[{i}] = {got}, (C_s dx)^2 |S| = {want} with C_s = {cs}"
        );
    }
}

// Pins today's behaviour; AUD-A-S1W6-014 questions this mapping (the current ratio-proportional form of `dynamic_smagorinsky_cs`)
#[test]
fn dynamic_coefficient_clamps_at_the_upper_bound() {
    // s11 = 1, 10, 1: the edge cells see ratio (1 + 10 + 4) / 6 = 2.5
    // (0.17 * 2.5 = 0.425 -> 0.25), the middle cell sees (1 + 1 + 40) / 60 = 0.7
    // (0.17 * 0.7 = 0.119, inside). With s11 = 10, 1, 10 the middle cell
    // sees (10 + 10 + 4) / 6 = 4 -> 0.25 and the edges (10 + 1 + 40) / 60 = 0.85.
    for (faces, lo, hi) in [
        ([0i64, 1, 11, 12], 0.17 * 0.7, 0.25),
        ([0i64, 10, 11, 21], 0.17 * 0.85, 0.25),
    ] {
        let (strain, report) = run(&faces);
        for i in 0..3 {
            let cs = expected_cs(&strain, i);
            assert!((0.05..=0.25).contains(&cs));
        }
        let t = report.turbulence;
        assert!(
            (t.cs_min.to_f64() - lo).abs() < 1e-12,
            "faces {faces:?}: cs_min = {}, closed form {lo}",
            t.cs_min.to_f64()
        );
        assert!(
            (t.cs_max.to_f64() - hi).abs() < 1e-12,
            "faces {faces:?}: cs_max = {}, clamp {hi}",
            t.cs_max.to_f64()
        );
    }
    // On a one-row grid four of the six neighbours are the cell itself, so
    // the ratio is at least 4/6 and the lower clamp 0.05 is out of reach.
    let min_ratio: f64 = 4.0 / 6.0;
    assert!(CS * min_ratio > 0.05);
}
