//! Audit oracles for `dynamic_fem` (consistent / lumped mass, Newmark-beta
//! average-acceleration integrator on P1 tetrahedra).
//!
//! `analytic_dynamic_fem.rs` pins the step response against the closed form,
//! the two mass distributions, free flight, the axial wave transit time and
//! the guards. This file adds invariants that are independent of those:
//!
//! - the Newmark kinematic identity `u' = u + dt v + dt^2/4 (a + a')` and
//!   `v' = v + dt/2 (a + a')` between the *public* getters, step by step
//! - the equation of motion `m a + k u = f` at every step including `t = 0`
//! - exact conservation of the discrete energy `1/2 m v^2 + 1/2 k u^2 - f u`
//!   for the trapezoidal rule (a quadratic invariant of the linear scheme)
//! - unconditional stability (response bounded and equal to the closed form at
//!   `omega dt ~ 4000`)
//! - linear momentum of an unconstrained body: zero for a self-equilibrated
//!   load, `F t` for a net force (the row sums of `K` vanish and the row sums
//!   of the mass are the nodal masses)
//!
//! Expected values are closed forms in `f64` (stiffness / mass of the corner
//! tetrahedron from its geometry, momentum from nodal masses), never read back
//! from the solver.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// reference values are f64 closed forms of simulation scenes
#![allow(clippy::disallowed_methods)]

use alice_physics::dynamic_fem::{DynamicsConfig, MassLumping, TransientSolver};
use alice_physics::linear_elastic_fem::{
    Axis, BoundaryConditions, ElasticMaterial, FemError, Preconditioner, SolverConfig,
};
use alice_physics::sdf_fem_mesh::{SdfTetMesh, Tetrahedron};
use alice_physics::Fix128;

fn two_pow_neg(shift: u32) -> Fix128 {
    Fix128::from_raw(0, 1u64 << (64 - shift))
}

const DENSITY_F64: f64 = 1.0 / 1_073_741_824.0;

fn density() -> Fix128 {
    two_pow_neg(30)
}

fn solver_config() -> SolverConfig {
    SolverConfig::try_new(20_000, Fix128::from_f64(1e-12)).expect("valid solver config")
}

fn material(e_mpa: i64) -> ElasticMaterial {
    ElasticMaterial::new(Fix128::from_int(e_mpa), Fix128::ZERO).expect("valid material")
}

fn dynamics(dt_shift: u32, lumping: MassLumping) -> DynamicsConfig {
    DynamicsConfig::try_new(density(), two_pow_neg(dt_shift), lumping).expect("valid dynamics")
}

/// Corner tetrahedron `det J = 3`, volume 1/2, `grad N3 = (0, 0, 1/3)`.
fn corner_tet() -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    mesh.vertices.push([0.0, 0.0, 0.0]);
    mesh.vertices.push([1.0, 0.0, 0.0]);
    mesh.vertices.push([0.0, 1.0, 0.0]);
    mesh.vertices.push([0.0, 0.0, 3.0]);
    mesh.tets.push(Tetrahedron {
        vertices: [0, 1, 2, 3],
    });
    mesh
}

/// One free coordinate (node 3, x); node 3 y/z and nodes 0..2 fixed.
fn sdof_boundary(load_n: f64) -> BoundaryConditions {
    let mut b = BoundaryConditions::new();
    for v in 0..3u32 {
        b.fix(v);
    }
    b.prescribe(3, Axis::Y, Fix128::ZERO);
    b.prescribe(3, Axis::Z, Fix128::ZERO);
    b.add_load(3, Axis::X, Fix128::from_f64(load_n));
    b
}

/// `k = V mu g_z^2`, with `lambda = 0` (nu = 0): `V mu / 9`.
fn sdof_k(e_mpa: f64) -> f64 {
    0.5 * (e_mpa / 2.0) / 9.0
}

/// consistent `rho V / 10`, lumped `rho V / 4`.
fn sdof_m(lumping: MassLumping) -> f64 {
    let m = DENSITY_F64 * 0.5;
    match lumping {
        MassLumping::Lumped => m / 4.0,
        _ => m / 10.0,
    }
}

fn node_index(nx: usize, ny: usize, i: usize, j: usize, k: usize) -> u32 {
    u32::try_from(i + j * (nx + 1) + k * (nx + 1) * (ny + 1)).expect("fits")
}

fn kuhn_box(nx: usize, ny: usize, nz: usize, h: f32) -> SdfTetMesh {
    let mut mesh = SdfTetMesh::default();
    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.vertices
                    .push([i as f32 * h, j as f32 * h, k as f32 * h]);
            }
        }
    }
    const PATHS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                for path in PATHS {
                    let mut step = [0usize; 3];
                    let mut corners = [0u32; 4];
                    corners[0] = node_index(nx, ny, i, j, k);
                    for (n, axis) in path.into_iter().enumerate() {
                        step[axis] = 1;
                        corners[n + 1] = node_index(nx, ny, i + step[0], j + step[1], k + step[2]);
                    }
                    mesh.tets.push(Tetrahedron { vertices: corners });
                }
            }
        }
    }
    mesh
}

// ---------------------------------------------------------------------------
// kinematics, equation of motion, energy
// ---------------------------------------------------------------------------

#[test]
fn newmark_kinematic_identities_hold_between_the_public_getters_at_every_step() {
    // 3x1x1 bar clamped at x = 0, loaded at the free end: every free dof moves.
    let (nx, h) = (3usize, 1.0_f32);
    let mesh = kuhn_box(nx, 1, 1, h);
    let mut b = BoundaryConditions::new();
    for j in 0..2usize {
        for k in 0..2usize {
            b.fix(node_index(nx, 1, 0, j, k));
            b.add_load(node_index(nx, 1, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let mut s = TransientSolver::new(
            &mesh,
            &material(3500),
            &b,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        let mut worst = 0.0_f64;
        for _ in 0..60 {
            let u0: Vec<f64> = s.displacements().iter().map(|x| x.to_f64()).collect();
            let v0: Vec<f64> = s.velocities().iter().map(|x| x.to_f64()).collect();
            let a0: Vec<f64> = s.accelerations().iter().map(|x| x.to_f64()).collect();
            s.step().expect("converges");
            for d in 0..u0.len() {
                let u1 = s.displacements()[d].to_f64();
                let v1 = s.velocities()[d].to_f64();
                let a1 = s.accelerations()[d].to_f64();
                let u_pred = u0[d] + dt * v0[d] + 0.25 * dt * dt * (a0[d] + a1);
                let v_pred = v0[d] + 0.5 * dt * (a0[d] + a1);
                let scale_u = u1.abs().max(u0[d].abs()).max(1e-30);
                let scale_v = v1.abs().max(v0[d].abs()).max(1e-30);
                worst = worst
                    .max((u1 - u_pred).abs() / scale_u)
                    .max((v1 - v_pred).abs() / scale_v);
            }
        }
        assert!(
            worst < 1e-6,
            "{lumping:?}: Newmark identity violated by {worst:e}"
        );
    }
}

#[test]
fn equation_of_motion_holds_at_every_step_including_the_consistent_initial_state() {
    let mesh = corner_tet();
    let (e, load) = (3500.0_f64, 1.0_f64);
    let dt_shift = 23u32;
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let (k, m) = (sdof_k(e), sdof_m(lumping));
        let mut s = TransientSolver::new(
            &mesh,
            &material(3500),
            &sdof_boundary(load),
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        // t = 0: u = 0, so m a0 = f
        let a0 = s.accelerations()[9].to_f64();
        assert!((m * a0 - load).abs() / load < 1e-9, "{lumping:?} a0 = {a0}");
        for n in 1..=400 {
            s.step().expect("converges");
            let u = s.displacements()[9].to_f64();
            let a = s.accelerations()[9].to_f64();
            let residual = (m * a + k * u - load).abs() / load;
            assert!(
                residual < 1e-6,
                "{lumping:?} step {n}: residual {residual:e}"
            );
        }
    }
}

#[test]
fn trapezoidal_rule_conserves_the_discrete_energy_of_a_step_loaded_oscillator() {
    // E = 1/2 m v^2 + 1/2 k u^2 - f u is zero at t = 0 and constant afterwards.
    let mesh = corner_tet();
    let (e, load) = (3500.0_f64, 1.0_f64);
    let dt_shift = 23u32;
    let lumping = MassLumping::Consistent;
    let (k, m) = (sdof_k(e), sdof_m(lumping));
    let u_s = load / k;
    let mut s = TransientSolver::new(
        &mesh,
        &material(3500),
        &sdof_boundary(load),
        &dynamics(dt_shift, lumping),
        &solver_config(),
    )
    .expect("well posed");
    let scale = load * u_s;
    let mut worst = 0.0_f64;
    for _ in 0..8_000 {
        s.step().expect("converges");
        let u = s.displacements()[9].to_f64();
        let v = s.velocities()[9].to_f64();
        let energy = 0.5 * m * v * v + 0.5 * k * u * u - load * u;
        worst = worst.max(energy.abs() / scale);
    }
    assert!(worst < 1e-6, "energy wandered by {worst:e} of f*u_s");
}

#[test]
fn response_stays_bounded_and_on_the_closed_form_when_dt_is_far_above_the_period() {
    // omega dt ~ 4400: the trapezoidal rule is unconditionally stable and its
    // amplification matrix has modulus exactly one, so u stays on
    // u_s (1 - cos n theta) with cos(theta) = (1 - q) / (1 + q), q = (omega dt / 2)^2.
    let mesh = corner_tet();
    let (e, load) = (3500.0_f64, 1.0_f64);
    let lumping = MassLumping::Consistent;
    let (k, m) = (sdof_k(e), sdof_m(lumping));
    let u_s = load / k;
    let omega = (k / m).sqrt();
    let dt_shift = 10u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    let q = omega * dt / 2.0;
    let q = q * q;
    let c1 = (1.0 - q) / (1.0 + q);
    let mut s = TransientSolver::new(
        &mesh,
        &material(3500),
        &sdof_boundary(load),
        &dynamics(dt_shift, lumping),
        &solver_config(),
    )
    .expect("well posed");
    let (mut prev, mut cur) = (1.0_f64, c1);
    let mut peak = 0.0_f64;
    let mut worst = 0.0_f64;
    for _ in 0..300 {
        s.step().expect("converges");
        let u = s.displacements()[9].to_f64();
        peak = peak.max(u.abs());
        worst = worst.max((u - u_s * (1.0 - cur)).abs() / u_s);
        let next = 2.0 * c1 * cur - prev;
        prev = cur;
        cur = next;
    }
    assert!(peak <= 2.0 * u_s * (1.0 + 1e-6), "peak {} u_s", peak / u_s);
    assert!(worst < 1e-5, "left the closed form by {worst:e} of u_s");
}

// ---------------------------------------------------------------------------
// momentum of an unconstrained body
// ---------------------------------------------------------------------------

/// Nodal masses `rho * sum_e V_e / 4` (rows of both mass matrices sum to this),
/// from the mesh geometry in f64.
fn nodal_masses(mesh: &SdfTetMesh) -> Vec<f64> {
    let mut m = vec![0.0_f64; mesh.vertices.len()];
    for t in &mesh.tets {
        let p: Vec<[f64; 3]> = t
            .vertices
            .iter()
            .map(|&v| {
                let q = mesh.vertices[v as usize];
                [f64::from(q[0]), f64::from(q[1]), f64::from(q[2])]
            })
            .collect();
        let e = |i: usize| [p[i][0] - p[0][0], p[i][1] - p[0][1], p[i][2] - p[0][2]];
        let (a, b, c) = (e(1), e(2), e(3));
        let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]);
        let vol = det.abs() / 6.0;
        for &v in &t.vertices {
            m[v as usize] += DENSITY_F64 * vol / 4.0;
        }
    }
    m
}

fn momentum(s: &TransientSolver, masses: &[f64]) -> [f64; 3] {
    let mut p = [0.0_f64; 3];
    for (i, m) in masses.iter().enumerate() {
        for (axis, slot) in p.iter_mut().enumerate() {
            *slot += m * s.velocities()[i * 3 + axis].to_f64();
        }
    }
    p
}

#[test]
fn a_self_equilibrated_load_on_a_free_body_keeps_zero_linear_momentum() {
    let (nx, h) = (2usize, 1.0_f32);
    let mesh = kuhn_box(nx, 1, 1, h);
    let masses = nodal_masses(&mesh);
    let last = u32::try_from(mesh.vertices.len() - 1).unwrap();
    let mut b = BoundaryConditions::new();
    b.add_load(0, Axis::X, Fix128::from_f64(0.5));
    b.add_load(last, Axis::X, Fix128::from_f64(-0.5));
    let dt_shift = 23u32;
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let mut s = TransientSolver::new(
            &mesh,
            &material(3500),
            &b,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("an unconstrained transient solve is well posed");
        let mut vmax = 0.0_f64;
        let mut pmax = 0.0_f64;
        for _ in 0..40 {
            s.step().expect("converges");
            let p = momentum(&s, &masses);
            pmax = pmax.max(p[0].abs()).max(p[1].abs()).max(p[2].abs());
            for v in s.velocities() {
                vmax = vmax.max(v.to_f64().abs());
            }
        }
        // the body does deform (velocities are non-trivial) ...
        assert!(vmax > 1.0, "{lumping:?}: no motion (vmax = {vmax})");
        // ... but its centre of mass does not move
        let total_mass: f64 = masses.iter().sum();
        assert!(
            pmax < 1e-9 * total_mass * vmax,
            "{lumping:?}: momentum {pmax:e} vs scale {:e}",
            total_mass * vmax
        );
    }
}

#[test]
fn a_net_force_on_a_free_body_gives_momentum_equal_to_force_times_time() {
    let (nx, h) = (2usize, 1.0_f32);
    let mesh = kuhn_box(nx, 1, 1, h);
    let masses = nodal_masses(&mesh);
    let mut b = BoundaryConditions::new();
    b.add_load(0, Axis::X, Fix128::from_f64(0.75));
    b.add_load(5, Axis::Y, Fix128::from_f64(-0.25));
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let mut s = TransientSolver::new(
            &mesh,
            &material(3500),
            &b,
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        for n in 1..=40u32 {
            s.step().expect("converges");
            let t = f64::from(n) * dt;
            let p = momentum(&s, &masses);
            assert!(
                (p[0] - 0.75 * t).abs() <= 1e-6 * 0.75 * t,
                "{lumping:?} n={n} px = {}",
                p[0]
            );
            assert!(
                (p[1] + 0.25 * t).abs() <= 1e-6 * 0.25 * t,
                "{lumping:?} n={n} py = {}",
                p[1]
            );
            assert!(
                p[2].abs() <= 1e-6 * 0.75 * t,
                "{lumping:?} n={n} pz = {}",
                p[2]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Poisson coupling (every scene in analytic_dynamic_fem.rs uses nu = 0, so
// the lambda terms of the stress and the lambda part of K are never read)
// ---------------------------------------------------------------------------

fn lame(e: f64, nu: f64) -> (f64, f64) {
    (
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        e / (2.0 * (1.0 + nu)),
    )
}

/// One free coordinate along z: node 3 z free, node 3 x / y and nodes 0..2 fixed.
fn sdof_z_boundary(load_n: f64) -> BoundaryConditions {
    let mut b = BoundaryConditions::new();
    for v in 0..3u32 {
        b.fix(v);
    }
    b.prescribe(3, Axis::X, Fix128::ZERO);
    b.prescribe(3, Axis::Y, Fix128::ZERO);
    b.add_load(3, Axis::Z, Fix128::from_f64(load_n));
    b
}

#[test]
fn poisson_coupled_axial_stiffness_is_lambda_plus_two_mu_times_volume_over_nine() {
    // free coordinate along z: k = V (lambda + 2 mu) g_z^2 = 0.5 (lambda + 2 mu) / 9
    let (e, nu, load) = (3500.0_f64, 0.25_f64, 1.0_f64);
    let (lambda, mu) = lame(e, nu);
    let k = 0.5 * (lambda + 2.0 * mu) / 9.0;
    let mat = ElasticMaterial::new(Fix128::from_int(3500), Fix128::from_ratio(1, 4)).unwrap();
    let dt_shift = 23u32;
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        let m = sdof_m(lumping);
        let mut s = TransientSolver::new(
            &corner_tet(),
            &mat,
            &sdof_z_boundary(load),
            &dynamics(dt_shift, lumping),
            &solver_config(),
        )
        .expect("well posed");
        let idx = 11; // node 3, z
        assert!((m * s.accelerations()[idx].to_f64() - load).abs() / load < 1e-9);
        let mut peak = 0.0_f64;
        for n in 1..=400 {
            s.step().expect("converges");
            let u = s.displacements()[idx].to_f64();
            let a = s.accelerations()[idx].to_f64();
            let residual = (m * a + k * u - load).abs() / load;
            assert!(
                residual < 1e-6,
                "{lumping:?} step {n}: residual {residual:e}"
            );
            peak = peak.max(u);
        }
        assert!(peak > 0.0);
    }
}

#[test]
fn reversing_the_winding_of_an_element_does_not_change_the_response() {
    // |det J| / 6 is the volume "so that winding does not change the stiffness".
    let mut flipped = corner_tet();
    flipped.tets[0].vertices = [0, 2, 1, 3];
    let dt_shift = 23u32;
    let mut traces = Vec::new();
    for mesh in [corner_tet(), flipped] {
        let mut s = TransientSolver::new(
            &mesh,
            &material(3500),
            &sdof_boundary(1.0),
            &dynamics(dt_shift, MassLumping::Consistent),
            &solver_config(),
        )
        .expect("well posed");
        let mut t = Vec::new();
        for _ in 0..50 {
            s.step().expect("converges");
            t.push(s.displacements()[9].to_f64());
        }
        traces.push(t);
    }
    for (a, b) in traces[0].iter().zip(traces[1].iter()) {
        assert!((a - b).abs() <= 1e-9 * a.abs().max(1e-30), "{a} vs {b}");
    }
}

#[test]
fn mass_distribution_lumped_response_is_slower_than_consistent_for_the_same_stiffness() {
    // ratio of the two oscillation frequencies is sqrt(5/2): count steps to the first
    // turning point of the step response
    let mut half_period = [0usize; 2];
    for (slot, lumping) in half_period
        .iter_mut()
        .zip([MassLumping::Consistent, MassLumping::Lumped])
    {
        let mut s = TransientSolver::new(
            &corner_tet(),
            &material(3500),
            &sdof_boundary(1.0),
            &dynamics(26, lumping),
            &solver_config(),
        )
        .expect("well posed");
        let mut prev = 0.0_f64;
        for n in 1..=4000usize {
            s.step().expect("converges");
            let u = s.displacements()[9].to_f64();
            if u < prev {
                *slot = n - 1;
                break;
            }
            prev = u;
        }
    }
    assert!(
        half_period[0] > 10 && half_period[1] > half_period[0],
        "{half_period:?}"
    );
}

// ---------------------------------------------------------------------------
// the doc claim about the two mass distributions against the continuum
// ---------------------------------------------------------------------------

fn bar_turning_step(lumping: MassLumping) -> (usize, f64) {
    let (nx, h) = (8usize, 1.0_f32);
    let length = f64::from(nx as u32) * f64::from(h);
    let mesh = kuhn_box(nx, 1, 1, h);
    let dt_shift = 23u32;
    let dt = 1.0 / f64::from(1u32 << dt_shift);
    let mut b = BoundaryConditions::new();
    for j in 0..2usize {
        for k in 0..2usize {
            b.fix(node_index(nx, 1, 0, j, k));
            b.add_load(node_index(nx, 1, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }
    let c = (3500.0_f64 / DENSITY_F64).sqrt();
    let transit = 2.0 * length / c;
    let mut s = TransientSolver::new(
        &mesh,
        &material(3500),
        &b,
        &dynamics(dt_shift, lumping),
        &solver_config(),
    )
    .expect("well posed");
    let tip = node_index(nx, 1, nx, 0, 0) as usize * 3;
    let budget = (4.0 * transit / dt) as usize;
    let mut prev = 0.0_f64;
    let mut rising = false;
    for n in 1..=budget {
        s.step().expect("converges");
        let u = s.displacements()[tip].to_f64();
        if u > prev {
            rising = true;
        } else if rising && u < prev {
            return (n - 1, transit / dt);
        }
        prev = u;
    }
    (0, transit / dt)
}

#[test]
fn consistent_mass_turns_the_bar_tip_earlier_than_the_continuum_and_lumped_mass_later() {
    // doc (MassLumping::Consistent): the discrete frequencies bound the continuum
    // ones from above; (MassLumping::Lumped): lowers the frequencies.
    // Higher frequency <=> shorter turning time 2L/c.
    let (consistent, want) = bar_turning_step(MassLumping::Consistent);
    let (lumped, _) = bar_turning_step(MassLumping::Lumped);
    assert!(consistent > 0 && lumped > 0);
    assert!(
        (consistent as f64) < want,
        "consistent {consistent} steps vs continuum {want}"
    );
    assert!(
        (lumped as f64) > want,
        "lumped {lumped} steps vs continuum {want}"
    );
    assert!(lumped > consistent);
}

#[test]
fn dynamics_config_accepts_dt_down_to_about_two_to_the_minus_thirty_and_rejects_below() {
    // (2/dt)^2 must stay below 2^63: dt = 2^-30 gives 2^62, dt = 2^-32 gives 2^66.
    for lumping in [MassLumping::Consistent, MassLumping::Lumped] {
        assert!(DynamicsConfig::try_new(density(), two_pow_neg(30), lumping).is_ok());
        assert!(DynamicsConfig::try_new(density(), two_pow_neg(32), lumping).is_err());
        assert!(DynamicsConfig::try_new(density(), Fix128::ZERO, lumping).is_err());
        assert!(DynamicsConfig::try_new(density(), Fix128::ZERO - Fix128::ONE, lumping).is_err());
        assert!(DynamicsConfig::try_new(Fix128::ZERO, two_pow_neg(10), lumping).is_err());
        assert!(
            DynamicsConfig::try_new(Fix128::ZERO - density(), two_pow_neg(10), lumping).is_err()
        );
        assert!(DynamicsConfig::try_new(density(), two_pow_neg(10), lumping).is_ok());
    }
}

// ---------------------------------------------------------------------------
// error paths
// ---------------------------------------------------------------------------

fn clamped_bar_boundary(nx: usize) -> BoundaryConditions {
    let mut b = BoundaryConditions::new();
    for j in 0..2usize {
        for k in 0..2usize {
            b.fix(node_index(nx, 1, 0, j, k));
            b.add_load(node_index(nx, 1, nx, j, k), Axis::X, Fix128::from_f64(0.25));
        }
    }
    b
}

#[test]
fn a_one_iteration_budget_is_reported_as_not_converged_and_a_failed_step_changes_nothing() {
    let (nx, h) = (4usize, 1.0_f32);
    let mesh = kuhn_box(nx, 1, 1, h);
    let cfg = SolverConfig::try_new(1, Fix128::from_f64(1e-12)).expect("valid solver config");
    // consistent mass: the initial-acceleration solve itself needs more than one iteration
    match TransientSolver::new(
        &mesh,
        &material(3500),
        &clamped_bar_boundary(nx),
        &dynamics(23, MassLumping::Consistent),
        &cfg,
    ) {
        Err(FemError::NotConverged { iterations, .. }) => assert_eq!(iterations, 1),
        Err(other) => panic!("unexpected error {other:?}"),
        Ok(_) => panic!("a one-iteration budget cannot solve the consistent-mass system"),
    }
    // lumped mass is diagonal: with the Jacobi preconditioner the initial solve is a single
    // iteration; the step system (K + M) is not
    let cfg = cfg.with_preconditioner(Preconditioner::JacobiScaled);
    let mut s = match TransientSolver::new(
        &mesh,
        &material(3500),
        &clamped_bar_boundary(nx),
        &dynamics(23, MassLumping::Lumped),
        &cfg,
    ) {
        Ok(s) => s,
        Err(e) => panic!("lumped initial solve should converge in one iteration: {e:?}"),
    };
    let (u, v, a) = (
        s.displacements().to_vec(),
        s.velocities().to_vec(),
        s.accelerations().to_vec(),
    );
    for _ in 0..2 {
        match s.step() {
            Err(FemError::NotConverged { iterations, .. }) => assert_eq!(iterations, 1),
            other => panic!("expected NotConverged, got {other:?}"),
        }
        assert_eq!(s.displacements(), u.as_slice());
        assert_eq!(s.velocities(), v.as_slice());
        assert_eq!(s.accelerations(), a.as_slice());
    }
}

#[test]
fn a_vertex_index_equal_to_the_vertex_count_is_refused_everywhere_it_can_appear() {
    // 4 vertices: index 4 is the first invalid one (off-by-one boundary)
    let cfg = solver_config();
    let mat = material(3500);
    let dyn_cfg = dynamics(23, MassLumping::Consistent);
    let expect_out_of_range = |r: Result<TransientSolver, FemError>, what: &str| match r {
        Err(FemError::VertexOutOfRange {
            vertex: 4,
            vertex_count: 4,
        }) => {}
        Err(e) => panic!("{what}: unexpected error {e:?}"),
        Ok(_) => panic!("{what}: vertex index 4 of 4 was accepted"),
    };
    let mut bad_tet = corner_tet();
    bad_tet.tets[0].vertices = [0, 1, 2, 4];
    expect_out_of_range(
        TransientSolver::new(&bad_tet, &mat, &sdof_boundary(1.0), &dyn_cfg, &cfg),
        "tetrahedron",
    );
    let mut fixed = sdof_boundary(1.0);
    fixed.fix(4);
    expect_out_of_range(
        TransientSolver::new(&corner_tet(), &mat, &fixed, &dyn_cfg, &cfg),
        "prescribed",
    );
    let mut loaded = sdof_boundary(1.0);
    loaded.add_load(4, Axis::X, Fix128::ONE);
    expect_out_of_range(
        TransientSolver::new(&corner_tet(), &mat, &loaded, &dyn_cfg, &cfg),
        "load",
    );
}
