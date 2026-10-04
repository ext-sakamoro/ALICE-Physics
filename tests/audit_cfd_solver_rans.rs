//! Audit S1-4 oracles for the reporting surface of `cfd_solver` (axis A / C):
//! the fields of `TurbulenceSummary`, `RansReport`, `RansState`, the "Err leaves
//! everything untouched" promises of `step_rans`, and `compute_max_dt` on a
//! resting field.
//!
//! Expected values are closed forms (written out here), not read back from the
//! implementation:
//!
//! ```text
//! Smagorinsky, linear shear u = g y     |S| = g,   nu_t = (C_s dx)^2 g,   C_s = 0.17
//! k-eps at rest, one explicit step      k1 = k0 - eps0 dt,   eps1 = eps0 - C2 eps0^2 dt / k0   (P = 0)
//! eddy viscosity of k-eps               nu_t = C_mu k^2 / eps,   C_mu = 0.09
//! Courant inversion                     dt = cfl dx / |u|max,  zero for cfl <= 0 or dx = 0
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, RansState, StepError, StepOptions, TurbulenceModel, WallModel,
};
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::Grid3d;

fn q(a: i64, b: i64) -> Fix128 {
    Fix128::from_ratio(a, b)
}

fn int(a: i64) -> Fix128 {
    Fix128::from_int(a)
}

fn opts() -> StepOptions {
    StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 })
}

fn quiet(n: usize, dx: Fix128) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, dx);
    s.gravity = Vec3Fix::ZERO;
    s
}

/// `u(i, j, k) = g (j + 1/2) dx` on every u face: a linear shear, `du/dy = g`.
fn shear(s: &mut CfdSolver, g: Fix128) {
    let (nx, ny, nz) = (s.grid.nx, s.grid.ny, s.grid.nz);
    let dx = s.grid.dx;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..=nx {
                let ix = i + (nx + 1) * (j + ny * k);
                s.grid.u[ix] = g * (int(j as i64) + q(1, 2)) * dx;
            }
        }
    }
}

// ------------------------------------------------ compute_max_dt on a resting field

#[test]
fn a_non_positive_cfl_or_zero_dx_gives_zero_even_when_the_field_is_at_rest() {
    // doc: `cfl_target <= 0` or `dx = 0` give zero. The resting field is the case
    // that distinguishes "zero because the guard fired" from "zero because
    // 0 / peak = 0": with no motion the unguarded path reports the cap instead.
    let rest = CfdSolver::new(3, 3, 3, int(1));
    assert_eq!(rest.compute_max_dt(Fix128::ZERO), Fix128::ZERO);
    assert_eq!(rest.compute_max_dt(int(-2)), Fix128::ZERO);
    let flat = CfdSolver::new(3, 3, 3, Fix128::ZERO);
    assert_eq!(flat.compute_max_dt(int(1)), Fix128::ZERO);
    // and the guard is the only thing that does it
    assert_eq!(rest.compute_max_dt(q(1, 2)), CfdSolver::MAX_DT_CAP);
}

// ------------------------------------------------------- summary closed forms

#[test]
fn smagorinsky_on_a_linear_shear_reports_the_closed_form_eddy_viscosity() {
    let n = 5;
    let dx = q(1, 4);
    let g = int(2);
    let mut s = quiet(n, dx);
    shear(&mut s, g);
    let mut state = RansState::new(n, n, n, TurbulenceModel::Smagorinsky);
    let dt = q(1, 100);
    let r = s.step_rans(dt, &opts(), &mut state).expect("steps");
    let nu_t = (0.17 * 0.25) * (0.17 * 0.25) * 2.0;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let got = r.eddy_viscosity.get(i, j, k).to_f64();
                assert!(
                    (got - nu_t).abs() < 1e-9 * nu_t,
                    "nu_t({i},{j},{k}) = {got}, closed form {nu_t}"
                );
            }
        }
    }
    let t = r.turbulence;
    assert!((t.nu_t_min.to_f64() - nu_t).abs() < 1e-9 * nu_t);
    assert!((t.nu_t_max.to_f64() - nu_t).abs() < 1e-9 * nu_t);
    assert!(
        (t.cs_min.to_f64() - 0.17).abs() < 1e-12,
        "cs_min {}",
        t.cs_min.to_f64()
    );
    assert!(
        (t.cs_max.to_f64() - 0.17).abs() < 1e-12,
        "cs_max {}",
        t.cs_max.to_f64()
    );
    let nu_mol = 1e-6;
    let dn = (nu_mol + nu_t) * 0.01 / (0.25 * 0.25);
    assert!(
        (t.diffusion_number.to_f64() - dn).abs() < 1e-9 * dn,
        "diffusion number {} vs {dn}",
        t.diffusion_number.to_f64()
    );
    // the LES closures carry no k / eps / production
    assert!(t.k_min.is_zero() && t.k_max.is_zero());
    assert!(t.epsilon_min.is_zero() && t.epsilon_max.is_zero());
    assert!(t.production_max.is_zero());
    assert_eq!(t.clamped, 0);
    // the field handed back has the grid's shape and spacing
    assert_eq!(
        (
            r.eddy_viscosity.nx,
            r.eddy_viscosity.ny,
            r.eddy_viscosity.nz
        ),
        (n, n, n)
    );
    assert_eq!(r.eddy_viscosity.dx, dx);
}

#[test]
fn dynamic_smagorinsky_under_uniform_strain_reports_the_static_coefficient() {
    // doc: under uniform strain the ratio is exactly one
    let n = 5;
    let mut s = quiet(n, q(1, 4));
    shear(&mut s, int(2));
    let mut state = RansState::new(n, n, n, TurbulenceModel::DynamicSmagorinsky);
    let r = s.step_rans(q(1, 100), &opts(), &mut state).expect("steps");
    assert!((r.turbulence.cs_min.to_f64() - 0.17).abs() < 1e-12);
    assert!((r.turbulence.cs_max.to_f64() - 0.17).abs() < 1e-12);
}

#[test]
fn k_epsilon_at_rest_reports_the_closed_form_one_step_envelope() {
    let n = 4;
    let mut s = quiet(n, int(1));
    let (k0, e0) = (1.0f64, 1.0f64);
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), int(1));
    let dt = q(1, 100);
    let r = s.step_rans(dt, &opts(), &mut state).expect("steps");
    let k1 = k0 - e0 * 0.01;
    // explicit Euler would give e1 = e0 - 1.92 e0^2 dt / k0 = 0.9808; the step advances k first and
    // reads the new k in the eps equation (0.98061), see AUD-A-S1W4-009, so eps is checked to 3e-4
    let e1 = e0 - 1.92 * e0 * e0 * 0.01 / k0;
    let t = r.turbulence;
    for (name, got, want, tol) in [
        ("k_min", t.k_min, k1, 1e-9),
        ("k_max", t.k_max, k1, 1e-9),
        ("epsilon_min", t.epsilon_min, e1, 3e-4),
        ("epsilon_max", t.epsilon_max, e1, 3e-4),
    ] {
        assert!(
            (got.to_f64() - want).abs() < tol,
            "{name} = {}, closed form {want}",
            got.to_f64()
        );
    }
    assert_eq!(
        t.epsilon_min, t.epsilon_max,
        "a uniform field has a one-point envelope"
    );
    // eddy viscosity of the state the step started from: C_mu k^2 / eps = 0.09
    assert!((t.nu_t_min.to_f64() - 0.09).abs() < 1e-12);
    assert!((t.nu_t_max.to_f64() - 0.09).abs() < 1e-12);
    // no strain at rest, so no production
    assert!(t.production_max.is_zero());
    assert_eq!(t.clamped, 0);
    // the closures that have no Smagorinsky coefficient report zero
    assert!(t.cs_min.is_zero() && t.cs_max.is_zero());
    let dn = (1e-6 + 0.09) * 0.01;
    assert!((t.diffusion_number.to_f64() - dn).abs() < 1e-12);
    // the state is left advanced by one step
    assert!((state.k(1, 1, 1).to_f64() - k1).abs() < 1e-9);
    assert!((state.epsilon(1, 1, 1).to_f64() - e1).abs() < 3e-4);
}

#[test]
#[ignore = "known defect: AUD-A-S1W4-009: the k-eps point source is documented as `explicit` (advance_epsilon: `one explicit Euler step`) but `advance_rans` advances k first and the eps equation then reads the updated k: eps1 = 0.980606 instead of the explicit-Euler 0.9808 for k0 = eps0 = 1, dt = 0.01 (first order either way; doc / ordering inconsistency)"]
fn k_epsilon_point_source_is_one_explicit_euler_step_from_the_start_of_the_step_state() {
    let n = 4;
    let mut s = quiet(n, int(1));
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), int(1));
    s.step_rans(q(1, 100), &opts(), &mut state).expect("steps");
    let e1 = 1.0 - 1.92 * 0.01;
    assert!(
        (state.epsilon(1, 1, 1).to_f64() - e1).abs() < 1e-9,
        "eps = {}, explicit Euler {e1}",
        state.epsilon(1, 1, 1).to_f64()
    );
}

#[test]
fn k_epsilon_production_max_is_nu_t_times_the_strain_squared() {
    // linear shear g: |S| = g, P = nu_t |S|^2 with nu_t = C_mu k^2 / eps = 0.09 for k = eps = 1
    let n = 5;
    let mut s = quiet(n, int(1));
    shear(&mut s, q(1, 5));
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), int(1));
    let r = s.step_rans(q(1, 100), &opts(), &mut state).expect("steps");
    let want = 0.09 * 0.2 * 0.2;
    let got = r.turbulence.production_max.to_f64();
    assert!(
        (got - want).abs() < 1e-9,
        "production_max {got}, closed form {want}"
    );
}

#[test]
fn the_prescribed_closure_reports_its_own_field_and_no_k_or_eps() {
    let n = 4;
    let mut field = Grid3d::new(n, n, n, int(1), q(1, 100));
    field.set(2, 2, 2, q(3, 100));
    let mut s = quiet(n, int(1));
    let mut state = RansState::prescribed(field);
    let r = s.step_rans(q(1, 100), &opts(), &mut state).expect("steps");
    assert_eq!(r.eddy_viscosity.get(2, 2, 2), q(3, 100));
    assert_eq!(r.eddy_viscosity.get(0, 0, 0), q(1, 100));
    assert_eq!(r.turbulence.nu_t_min, q(1, 100));
    assert_eq!(r.turbulence.nu_t_max, q(3, 100));
    assert_eq!(r.turbulence.model, TurbulenceModel::Prescribed);
    assert!(r.turbulence.k_max.is_zero() && r.turbulence.epsilon_max.is_zero());
}

// --------------------------------------------------- Err leaves everything untouched

fn snapshot(s: &CfdSolver) -> (Vec<Fix128>, Vec<Fix128>, Vec<Fix128>, Vec<Fix128>, u64) {
    (
        s.grid.u.clone(),
        s.grid.v.clone(),
        s.grid.w.clone(),
        s.grid.pressure.clone(),
        s.step_count,
    )
}

#[test]
fn step_rans_refusals_leave_the_solver_and_the_state_untouched() {
    let n = 4;
    // (1) diffusion number above 1/6: nu_t = 0.09, dt = 3, dx = 1
    {
        let mut s = quiet(n, int(1));
        shear(&mut s, q(1, 5));
        let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), int(1));
        let (before, state_before) = (snapshot(&s), state.clone());
        let e = s
            .step_rans(int(3), &opts(), &mut state)
            .expect_err("unstable");
        assert!(matches!(e, StepError::DiffusionUnstable { .. }), "{e:?}");
        assert_eq!(snapshot(&s), before);
        assert_eq!(state, state_before);
    }
    // (2) a negative prescribed cell
    {
        let mut field = Grid3d::new(n, n, n, int(1), q(1, 100));
        field.set(1, 1, 1, q(-1, 100));
        let mut s = quiet(n, int(1));
        shear(&mut s, q(1, 5));
        let mut state = RansState::prescribed(field);
        let (before, state_before) = (snapshot(&s), state.clone());
        let e = s
            .step_rans(q(1, 100), &opts(), &mut state)
            .expect_err("negative");
        assert_eq!(e, StepError::NegativeEddyViscosity);
        assert_eq!(snapshot(&s), before);
        assert_eq!(state, state_before);
    }
    // (3) a wall model on a solver with zero molecular viscosity
    {
        let mut s = quiet(n, int(1));
        s.dynamic_viscosity_pas = Fix128::ZERO;
        shear(&mut s, q(1, 5));
        let mut state = RansState::new(n, n, n, TurbulenceModel::Smagorinsky);
        let (before, state_before) = (snapshot(&s), state.clone());
        let o = opts().with_wall_model(WallModel::log_law());
        let e = s
            .step_rans(q(1, 100), &o, &mut state)
            .expect_err("no viscosity");
        assert_eq!(e, StepError::WallModelNeedsViscosity);
        assert_eq!(snapshot(&s), before);
        assert_eq!(state, state_before);
    }
    // (4) a zero time step and a state of the wrong shape
    {
        let mut s = quiet(n, int(1));
        let mut state = RansState::new(n + 1, n, n, TurbulenceModel::Smagorinsky);
        let before = snapshot(&s);
        assert!(s.step_rans(Fix128::ZERO, &opts(), &mut state).is_err());
        assert!(s.step_rans(q(1, 100), &opts(), &mut state).is_err());
        assert_eq!(snapshot(&s), before);
    }
}

// ------------------------------------------------------------ RansState accessors

#[test]
fn rans_state_reads_zero_and_ignores_writes_out_of_range() {
    let mut st = RansState::uniform(2, 3, 4, TurbulenceModel::KEpsilon, int(5), int(7));
    assert_eq!(st.cell_dims(), (2, 3, 4));
    assert_eq!(st.model(), TurbulenceModel::KEpsilon);
    assert_eq!(st.k(1, 2, 3), int(5));
    assert_eq!(st.epsilon(1, 2, 3), int(7));
    for (i, j, k) in [(2, 0, 0), (0, 3, 0), (0, 0, 4), (99, 99, 99)] {
        assert_eq!(st.k(i, j, k), Fix128::ZERO, "k({i},{j},{k})");
        assert_eq!(st.epsilon(i, j, k), Fix128::ZERO);
        assert_eq!(st.eddy_viscosity(i, j, k), Fix128::ZERO);
    }
    let before = st.clone();
    st.set(2, 0, 0, int(1), int(1));
    st.set(0, 3, 0, int(1), int(1));
    st.set(0, 0, 4, int(1), int(1));
    assert_eq!(st, before);
    st.set(1, 2, 3, int(2), int(4));
    assert_eq!((st.k(1, 2, 3), st.epsilon(1, 2, 3)), (int(2), int(4)));
    // nu_t = C_mu k^2 / eps = 0.09 * 4 / 4, and zero when k or eps is zero
    assert!((st.eddy_viscosity(1, 2, 3).to_f64() - 0.09).abs() < 1e-12);
    st.set(0, 0, 0, Fix128::ZERO, int(3));
    assert_eq!(st.eddy_viscosity(0, 0, 0), Fix128::ZERO);
    st.set(0, 0, 0, int(3), Fix128::ZERO);
    assert_eq!(st.eddy_viscosity(0, 0, 0), Fix128::ZERO);
    // LES closures read the strain instead
    let les = RansState::new(2, 2, 2, TurbulenceModel::Smagorinsky);
    assert_eq!(les.eddy_viscosity(0, 0, 0), Fix128::ZERO);
}

// ---------------------------------------------------------------- stability / defaults

#[test]
fn semi_lagrangian_stays_inside_the_initial_range_at_a_courant_number_of_five() {
    // doc: "unconditionally stable". Trilinear interpolation is a convex combination,
    // so a temperature field cannot leave [min, max] however large c dt / dx is.
    let (nx, ny, nz) = (10usize, 3usize, 3usize);
    let mut s = CfdSolver::new(nx, ny, nz, int(1));
    s.gravity = Vec3Fix::ZERO;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.jacobi_iterations = 0;
    let mut t = Grid3d::new(nx, ny, nz, int(1), int(300));
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                t.set(
                    i,
                    j,
                    k,
                    int(300) + int(((i * 7 + j * 3 + k) % 5) as i64 * 20),
                );
            }
        }
    }
    s.temperature = Some(t);
    s.beta_per_k = Fix128::ZERO;
    for u in s.grid.u.iter_mut() {
        *u = int(1);
    }
    for _ in 0..3 {
        s.step_multigrid(int(5), 0);
    }
    for &v in &s.temperature.as_ref().unwrap().data {
        assert!(
            v >= int(300) && v <= int(380),
            "left the range: {}",
            v.to_f64()
        );
    }
}

#[test]
fn the_default_fluid_is_water_with_water_s_thermal_expansion() {
    let s = CfdSolver::new(2, 2, 2, int(1));
    assert!((s.density_kg_m3.to_f64() - 998.0).abs() < 5.0);
    assert!((s.dynamic_viscosity_pas.to_f64() - 1.0e-3).abs() < 1e-4);
    let beta_water_20c = 2.07e-4; // CRC Handbook, volumetric expansion of water at 20 C
    assert!(
        (s.beta_per_k.to_f64() - beta_water_20c).abs() < 0.2 * beta_water_20c,
        "beta = {}",
        s.beta_per_k.to_f64()
    );
}

// ====================================================================
// Second pass: oracles for the mutants the first pass left alive
// ====================================================================

fn rest_solver(n: usize, dx: Fix128) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, dx);
    s.gravity = Vec3Fix::ZERO;
    s
}

#[test]
fn smagorinsky_on_a_pure_normal_strain_reports_sqrt_2_times_the_diagonal_strain() {
    // u = a x, v = b y, w = c z (faces at x = i dx): s11 = a, s22 = b, s33 = c, all other
    // components zero, |S| = sqrt(2 (a^2 + b^2 + c^2)), nu_t = (C_s dx)^2 |S| in every cell
    let n = 5;
    let dx = q(1, 4);
    let (a, b, c) = (1.0f64, 2.0f64, 3.0f64);
    let mut s = rest_solver(n, dx);
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                s.grid.u[i + (n + 1) * (j + n * k)] = int(1) * int(i as i64) * dx;
            }
        }
    }
    for k in 0..n {
        for j in 0..=n {
            for i in 0..n {
                s.grid.v[i + n * (j + (n + 1) * k)] = int(2) * int(j as i64) * dx;
            }
        }
    }
    for k in 0..=n {
        for j in 0..n {
            for i in 0..n {
                s.grid.w[i + n * (j + n * k)] = int(3) * int(k as i64) * dx;
            }
        }
    }
    let mut state = RansState::new(n, n, n, TurbulenceModel::Smagorinsky);
    let r = s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    let strain = (2.0 * (a * a + b * b + c * c)).sqrt();
    let want = (0.17 * 0.25) * (0.17 * 0.25) * strain;
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let got = r.eddy_viscosity.get(i, j, k).to_f64();
                assert!(
                    (got - want).abs() < 1e-9 * want,
                    "({i},{j},{k}): {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn the_stability_bound_is_inclusive_at_exactly_one_sixth() {
    // doc: refused when the diffusion number *exceeds* 1/6. nu_t = 1/12, dt = 2, dx = 1, nu_mol = 0
    // gives 2 * floor(2^64 / 12) = floor(2^64 / 6): exactly the bound
    let n = 3;
    let mut s = rest_solver(n, int(1));
    s.dynamic_viscosity_pas = Fix128::ZERO;
    let field = Grid3d::new(n, n, n, int(1), q(1, 12));
    let mut state = RansState::prescribed(field);
    let ok = s.step_rans(int(2), &opts(), &mut state);
    assert!(ok.is_ok(), "{ok:?}");
    let mut state = RansState::prescribed(Grid3d::new(n, n, n, int(1), q(1, 12)));
    let e = s
        .step_rans(int(2) + q(1, 1000), &opts(), &mut state)
        .expect_err("above the bound");
    assert!(matches!(e, StepError::DiffusionUnstable { .. }), "{e:?}");
}

#[test]
fn the_diffusion_number_uses_the_largest_eddy_viscosity_and_the_summary_reports_the_envelope() {
    let n = 3;
    let mut s = rest_solver(n, int(1));
    s.dynamic_viscosity_pas = Fix128::ZERO;
    let mut field = Grid3d::new(n, n, n, int(1), Fix128::ZERO);
    field.set(1, 1, 1, q(1, 5)); // 0.2 > 1/6 at dt = 1
    let mut state = RansState::prescribed(field.clone());
    let e = s
        .step_rans(int(1), &opts(), &mut state)
        .expect_err("0.2 dt / dx^2 > 1/6");
    match e {
        StepError::DiffusionUnstable { diffusion_number } => {
            assert!(
                (diffusion_number.to_f64() - 0.2).abs() < 1e-12,
                "{diffusion_number:?}"
            );
        }
        other => panic!("{other:?}"),
    }
    // below the bound the summary reports the same formula and the envelope
    let mut state = RansState::prescribed(field);
    let r = s
        .step_rans(q(1, 2), &opts(), &mut state)
        .expect("0.1 <= 1/6");
    assert!((r.turbulence.diffusion_number.to_f64() - 0.1).abs() < 1e-12);
    assert_eq!(r.turbulence.nu_t_min, Fix128::ZERO);
    assert_eq!(r.turbulence.nu_t_max, q(1, 5));
}

#[test]
fn a_zero_prescribed_eddy_viscosity_is_accepted() {
    // only a negative cell is refused
    let n = 3;
    let mut s = rest_solver(n, int(1));
    let mut state = RansState::prescribed(Grid3d::new(n, n, n, int(1), Fix128::ZERO));
    let r = s
        .step_rans(q(1, 100), &opts(), &mut state)
        .expect("zero is not negative");
    assert_eq!(r.turbulence.nu_t_max, Fix128::ZERO);
}

#[test]
fn dynamic_smagorinsky_reports_ordered_coefficients_inside_the_documented_clamp() {
    // a parabolic shear profile: |S| grows with height, so the test-filtered ratio differs from 1 in some cells
    let n = 6;
    let dx = q(1, 4);
    let mut s = rest_solver(n, dx);
    for k in 0..n {
        for j in 0..n {
            for i in 0..=n {
                let h = int(j as i64) + q(1, 2);
                s.grid.u[i + (n + 1) * (j + n * k)] = h * h * dx;
            }
        }
    }
    let mut state = RansState::new(n, n, n, TurbulenceModel::DynamicSmagorinsky);
    let r = s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    let (lo, hi) = (r.turbulence.cs_min.to_f64(), r.turbulence.cs_max.to_f64());
    assert!(
        lo < hi,
        "cs_min {lo} must be below cs_max {hi} for a non-uniform strain"
    );
    assert!(
        lo >= 0.05 - 1e-12 && hi <= 0.25 + 1e-12,
        "[{lo}, {hi}] outside [0.05, 0.25]"
    );
}

/// the cell with a spike relative to a uniform background; the neighbour's gain per step is
/// `dt (nu_mol + nu_t / sigma) delta` to first order in `delta` and `dt`
fn spike_diffusion(
    model: TurbulenceModel,
    spike_k: bool,
    sigma: f64,
    internal_wall: bool,
) -> (f64, f64) {
    let n = 5;
    let dt = 1e-4;
    let delta = 1e-3;
    let mut s = rest_solver(n, int(1));
    if internal_wall {
        // the face between cells (2,2,2) and (3,2,2)
        s.grid.set_u_bc(
            3,
            2,
            2,
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
        );
    }
    let mut state = RansState::uniform(n, n, n, model, int(1), int(1));
    if spike_k {
        state.set(2, 2, 2, Fix128::from_f64(1.0 + delta), int(1));
    } else {
        state.set(2, 2, 2, int(1), Fix128::from_f64(1.0 + delta));
    }
    let o = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 });
    s.step_rans(Fix128::from_f64(dt), &o, &mut state)
        .expect("steps");
    let pick = |i, j, k| {
        if spike_k {
            state.k(i, j, k).to_f64()
        } else {
            state.epsilon(i, j, k).to_f64()
        }
    };
    let far = pick(0, 0, 0);
    let expect = dt * (1e-6 + 0.09 / sigma) * delta;
    (
        (pick(1, 2, 2) - far) / expect,
        (pick(3, 2, 2) - far) / expect,
    )
}

#[test]
fn k_epsilon_dissipation_diffuses_with_nu_mol_plus_nu_t_over_sigma_epsilon() {
    // sigma_eps = 1.3 (Launder-Spalding); the neighbour of an eps spike gains dt (nu + nu_t / 1.3) delta
    let (west, _) = spike_diffusion(TurbulenceModel::KEpsilon, false, 1.3, false);
    assert!((west - 1.0).abs() < 0.05, "gain / closed form = {west}");
}

#[test]
fn k_omega_turbulent_energy_diffuses_with_nu_mol_plus_nu_t_over_sigma_two() {
    // Wilcox k-omega: sigma = 2 for both scalars
    let (west, _) = spike_diffusion(TurbulenceModel::KOmega, true, 2.0, false);
    assert!((west - 1.0).abs() < 0.05, "k gain / closed form = {west}");
}

#[test]
fn k_omega_dissipation_diffuses_with_nu_mol_plus_nu_t_over_sigma_two() {
    let (west, _) = spike_diffusion(TurbulenceModel::KOmega, false, 2.0, false);
    assert!((west - 1.0).abs() < 0.05, "eps gain / closed form = {west}");
}

#[test]
fn no_turbulent_flux_crosses_a_solid_face() {
    // an interior wall between the spike (2,2,2) and (3,2,2): that neighbour gains nothing,
    // the open neighbour (1,2,2) gains the full diffusive amount
    let (west, east) = spike_diffusion(TurbulenceModel::KEpsilon, false, 1.3, true);
    assert!((west - 1.0).abs() < 0.05, "open side {west}");
    assert!(
        east.abs() < 1e-3,
        "walled side received {east} of the diffusive flux"
    );
}

#[test]
fn a_transported_k_is_carried_with_the_fluid_by_half_a_cell() {
    // uniform u = 5, dx = 1, dt = 0.1: a half-cell shift. k = 1 + 0.01 i, eps = 0.1.
    // k loses eps dt = 0.01 to the sink, then is advected: k_new(i) = k(i - 1/2) - 0.01 to 1e-4 (a wrong sign is off by 0.01)
    let n = 10;
    let mut s = rest_solver(n, int(1));
    for u in s.grid.u.iter_mut() {
        *u = int(5);
    }
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), q(1, 10));
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                state.set(i, j, k, Fix128::ONE + q(i as i64, 100), q(1, 10));
            }
        }
    }
    s.step_rans(q(1, 10), &opts(), &mut state).expect("steps");
    for i in 2..n - 2 {
        let want = 1.0 + 0.01 * (i as f64 - 0.5) - 0.01;
        let got = state.k(i, 4, 4).to_f64();
        assert!(
            (got - want).abs() < 1e-4,
            "k({i}) = {got}, carried value {want}"
        );
    }
}

#[test]
#[ignore = "known defect: AUD-A-S1W4-010: `TurbulenceSummary::clamped` is documented as the `Number of cells` whose k, eps or omega step was clamped, but the k-omega branch counts the k clamp and the omega clamp separately: a uniform 3^3 field gives 54 (2 per cell) instead of 27"]
fn the_clamped_counter_of_k_omega_counts_cells_as_documented() {
    // doc: "Number of cells whose explicit source step would have taken k, eps or omega negative".
    // uniform k = 1, eps = 100, dt = 0.02: k + dk dt < 0 and omega + domega dt < 0 in every cell
    let n = 3;
    let mut s = rest_solver(n, int(1));
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KOmega, int(1), int(100));
    let r = s.step_rans(q(1, 50), &opts(), &mut state).expect("steps");
    assert_eq!(r.turbulence.clamped, (n * n * n) as u32);
}

#[test]
fn legacy_use_turbulence_adds_the_smagorinsky_viscosity_of_the_largest_diagonal_strain() {
    // base flow u = a x, v = -a y (solenoidal pure strain, a = 1): s11 = 1, s22 = -1, |S| = 2, so the
    // grid-wide nu_t = (C_s dx)^2 * 2. A divergence-free sheet u = p on the row j = 2 rides on it: the
    // Laplacian of the base flow vanishes in the interior and the sheet's is -2 p, so turbulence on
    // minus off changes the sheet by -2 nu_t dt / dx^2 p
    let n = 5;
    let dx = q(1, 4);
    let dt = 1e-4;
    let p = 0.1;
    let make = |turb: bool| {
        let mut s = rest_solver(n, dx);
        s.jacobi_iterations = 0;
        s.use_turbulence = turb;
        for k in 0..n {
            for j in 0..n {
                for i in 0..=n {
                    let base = int(i as i64) * dx;
                    let sheet = if j == 2 {
                        Fix128::from_f64(p)
                    } else {
                        Fix128::ZERO
                    };
                    s.grid.u[i + (n + 1) * (j + n * k)] = base + sheet;
                }
            }
        }
        for k in 0..n {
            for j in 0..=n {
                for i in 0..n {
                    s.grid.v[i + n * (j + (n + 1) * k)] = Fix128::ZERO - int(j as i64) * dx;
                }
            }
        }
        s.step_multigrid(Fix128::from_f64(dt), 0);
        s
    };
    let (on, off) = (make(true), make(false));
    let ix = 2 + (n + 1) * (2 + n * 2);
    let nu_t = (0.17 * 0.25) * (0.17 * 0.25) * 2.0;
    let want = -2.0 * nu_t * dt / (0.25 * 0.25) * p;
    let got = on.grid.u[ix].to_f64() - off.grid.u[ix].to_f64();
    assert!(
        (got - want).abs() < 0.03 * want.abs(),
        "sheet change {got} vs {want}"
    );
}

#[test]
fn the_variable_viscosity_path_diffuses_a_u_sheet_with_dt_nu_over_dx_squared() {
    // prescribed uniform nu_t = 1/100, dx = 1/4: a divergence-free sheet u = p on the row j = 2 spreads
    // to the rows j = 1 and 3 by scale nu p and loses 2 scale nu p (the other axes see a uniform field)
    let n = 5;
    let dx = q(1, 4);
    let dt = 1e-4;
    let p = 1e-3;
    let mut s = rest_solver(n, dx);
    for k in 0..n {
        for i in 0..=n {
            s.grid.u[i + (n + 1) * (2 + n * k)] = Fix128::from_f64(p);
        }
    }
    let mut state = RansState::prescribed(Grid3d::new(n, n, n, dx, q(1, 100)));
    s.step_rans(Fix128::from_f64(dt), &opts(), &mut state)
        .expect("steps");
    let scale_nu = dt / (0.25 * 0.25) * (0.01 + 1e-6);
    for (j, want) in [
        (1usize, scale_nu * p),
        (3, scale_nu * p),
        (2, p * (1.0 - 2.0 * scale_nu)),
    ] {
        let got = s.grid.u(2, j, 2).to_f64();
        let tol = if j == 2 {
            0.03 * 2.0 * scale_nu * p
        } else {
            0.03 * want
        };
        assert!((got - want).abs() < tol, "row {j}: {got} vs {want}");
    }
}

/// friction velocity from `u / u_tau = ln(u_tau y / nu) / kappa + B` by bisection (test-side, independent)
fn log_law_u_tau(u: f64, y: f64, nu: f64) -> f64 {
    let f = |ut: f64| ut * ((ut * y / nu).ln() / 0.41 + 5.5) - u;
    let (mut lo, mut hi) = (1e-6, u);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn the_wall_model_sink_enters_the_variable_viscosity_path_with_dt_u_tau_squared_over_dx() {
    // single wall at y = 0, u = 2 uniform, nu_t = 0 prescribed, dx = 1/4 (y_p = 1/8): the
    // wall-adjacent u faces lose dt u_tau^2 / dx, no other face changes
    let n = 4;
    let dx = q(1, 4);
    let dt = 1e-3;
    let mut s = rest_solver(n, dx);
    for k in 0..n {
        for i in 0..n {
            s.grid.set_v_bc(
                i,
                0,
                k,
                FaceBc::Wall {
                    velocity: Vec3Fix::ZERO,
                },
            );
        }
    }
    for u in s.grid.u.iter_mut() {
        *u = int(2);
    }
    let mut state = RansState::prescribed(Grid3d::new(n, n, n, dx, Fix128::ZERO));
    let o = opts().with_wall_model(WallModel::log_law());
    let r = s
        .step_rans(Fix128::from_f64(dt), &o, &mut state)
        .expect("steps");
    let u_tau = log_law_u_tau(2.0, 0.125, 1e-6);
    let want_drop = dt * u_tau * u_tau / 0.25;
    for k in 0..n {
        for i in 0..=n {
            let drop = 2.0 - s.grid.u(i, 0, k).to_f64();
            assert!(
                (drop - want_drop).abs() < 1e-3 * want_drop,
                "face ({i},0,{k}): drop {drop} vs {want_drop}"
            );
            assert!(
                (2.0 - s.grid.u(i, 2, k).to_f64()).abs() < 1e-12,
                "interior face moved"
            );
        }
    }
    let w = r.wall.expect("wall summary");
    assert!((w.u_tau_max.to_f64() - u_tau).abs() < 1e-3 * u_tau);
}

#[test]
fn the_first_face_the_wall_model_touches_is_part_of_the_envelope() {
    // one fast face at (0, 0, 0), the first u face visited; every other wall-adjacent face is slower.
    // the envelope maximum must be the fast face's friction velocity
    let n = 3;
    let dx = q(1, 4);
    let mut s = rest_solver(n, dx);
    for k in 0..n {
        for i in 0..n {
            s.grid.set_v_bc(
                i,
                0,
                k,
                FaceBc::Wall {
                    velocity: Vec3Fix::ZERO,
                },
            );
        }
    }
    for u in s.grid.u.iter_mut() {
        *u = int(2);
    }
    s.grid.u[0] = int(4);
    let o = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 1 })
        .with_wall_model(WallModel::log_law());
    let w = s
        .step_with_options(Fix128::from_f64(1e-6), &o)
        .expect("steps")
        .wall
        .expect("summary");
    let fast = log_law_u_tau(4.0, 0.125, 1e-6);
    let slow = log_law_u_tau(2.0, 0.125, 1e-6);
    assert!(
        (w.u_tau_max.to_f64() - fast).abs() < 1e-3 * fast,
        "max {} vs {fast}",
        w.u_tau_max.to_f64()
    );
    assert!(
        (w.u_tau_min.to_f64() - slow).abs() < 1e-3 * slow,
        "min {} vs {slow}",
        w.u_tau_min.to_f64()
    );
}

#[test]
fn k_omega_clamps_k_alone_when_only_the_energy_sink_overshoots() {
    // uniform k = 1, eps = 110, dt = 1/100: k + dk dt = 1 - 1.1 < 0 (clamped) while
    // omega (1 - beta omega dt) = 1 - 0.075 * 1222 * 0.01 = 0.083 > 0 is not: one event per cell
    let n = 3;
    let mut s = rest_solver(n, int(1));
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KOmega, int(1), int(110));
    let r = s.step_rans(q(1, 100), &opts(), &mut state).expect("steps");
    assert_eq!(r.turbulence.clamped, (n * n * n) as u32);
    assert_eq!(r.turbulence.k_min, Fix128::ZERO);
}

#[test]
fn the_pressure_report_names_the_solver_that_ran_and_only_bicgstab_reports_a_verdict() {
    // `ProjectionReport::solver` is documented "the solver that ran, as requested"; `bicgstab` is
    // `None` for the fixed-count solvers
    let solvers = [
        PressureSolver::RedBlackGs { sweeps: 5 },
        PressureSolver::Multigrid { cycles: 2 },
        PressureSolver::Jacobi { iterations: 5 },
        PressureSolver::DecomposedGs {
            ranks: 2,
            sweeps: 5,
        },
        PressureSolver::BandedGs {
            ranks: 2,
            sweeps: 5,
        },
        PressureSolver::BiCgStab {
            max_iterations: 50,
            tolerance: q(1, 1_000_000),
        },
    ];
    for solver in solvers {
        let mut s = rest_solver(4, int(1));
        let r = s
            .step_with_pressure_solver(q(1, 100), solver)
            .expect("steps");
        assert_eq!(r.solver, solver);
        assert_eq!(
            r.bicgstab.is_some(),
            matches!(solver, PressureSolver::BiCgStab { .. }),
            "{solver:?}"
        );
    }
}
