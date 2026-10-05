//! Oracles for the wall boundary condition of the k-ε / k-ω transport in
//! `CfdSolver::step_rans` (the Launder–Spalding 1974 wall function).
//!
//! # Closed forms
//!
//! With a `WallModel` enabled, every cell with a no-slip
//! `FaceBc::Wall` face carries the log-layer equilibrium values at its
//! centre, `y_p = dx/2` from the wall:
//!
//! ```text
//! u_rel = u_τ · u⁺(y_p u_τ / ν)        (u⁺ = y⁺ below 11.4453, ln(y⁺)/κ + B above)
//! k_P   = u_τ² / √C_μ
//! ε_P   = u_τ³ / (κ y_p)               (ω_P = ε_P / (β* k_P) = u_τ / (√C_μ κ y_p))
//! ```
//!
//! `u_rel` is the speed of the cell-centred velocity relative to the wall,
//! tangential to the face. The test inverts the profile on its own (f64
//! bisection) and does not call the crate's inverse.
//!
//! In a turbulent Couette flow the shear stress `ρ u_τ²` is uniform, so the
//! log-layer solution of the k equation (`P = ε`, `k` constant) is
//! `k = u_τ²/√C_μ` everywhere the layer is in local equilibrium, and
//! `ε = u_τ³/(κ y)`. The second oracle integrates such a flow to its steady
//! state and checks the first cell **off** the wall, which the boundary
//! condition does not touch: the value there is what the transport made of
//! the wall value.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, PressureSolverError, RansState, StepError, StepOptions,
    TurbulenceModel, WallModel,
};
use alice_physics::det_math::ln64;
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};

const C_MU: f64 = 0.09;
const KAPPA: f64 = 0.41;
const B: f64 = 5.5;
const Y_PLUS_SWITCH: f64 = 11.445_319_11;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

/// `u_τ` with `u_rel = u_τ u⁺(y u_τ / ν)`, by f64 bisection (independent of
/// the crate's `friction_velocity`).
fn log_law_u_tau(u_rel: f64, y: f64, nu: f64) -> f64 {
    if u_rel == 0.0 {
        return 0.0;
    }
    let u_plus = |yp: f64| {
        if yp < Y_PLUS_SWITCH {
            yp
        } else {
            ln64(yp) / KAPPA + B
        }
    };
    let (mut lo, mut hi) = (0.0f64, u_rel.max((nu * u_rel / y).sqrt()) * 2.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid * u_plus(y * mid / nu) < u_rel {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn wall_k_eps(u_tau: f64, y: f64) -> (f64, f64) {
    (
        u_tau * u_tau / C_MU.sqrt(),
        u_tau * u_tau * u_tau / (KAPPA * y),
    )
}

fn opts() -> StepOptions {
    StepOptions::new(PressureSolver::RedBlackGs { sweeps: 10 })
        .with_wall_model(WallModel::log_law())
}

/// `nx × ny × 1` channel of spacing `dx`: floor at rest (`j = 0`), lid moving
/// at `lid` along `x` (`j = ny`), slip `z` faces, open `x` faces (an
/// `x`-uniform flow is then divergence free and the semi-Lagrangian advection
/// of it is the identity).
fn channel(nx: usize, ny: usize, dx: Fix128, nu: Fix128, lid: Fix128) -> CfdSolver {
    let mut s = CfdSolver::new(nx, ny, 1, dx);
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = nu;
    for i in 0..nx {
        s.grid.set_v_bc(
            i,
            0,
            0,
            FaceBc::Wall {
                velocity: Vec3Fix::ZERO,
            },
        );
        s.grid.set_v_bc(
            i,
            ny,
            0,
            FaceBc::Wall {
                velocity: Vec3Fix::new(lid, Fix128::ZERO, Fix128::ZERO),
            },
        );
        for j in 0..ny {
            s.grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            s.grid.set_w_bc(i, j, 1, FaceBc::SlipWall);
        }
    }
    s
}

fn set_profile(s: &mut CfdSolver, profile: &dyn Fn(usize) -> Fix128) {
    let (nx, ny) = (s.grid.nx, s.grid.ny);
    for j in 0..ny {
        for i in 0..=nx {
            s.grid.u[i + (nx + 1) * j] = profile(j);
        }
    }
}

// ===========================================================================
// Oracle 1 — the wall cells hold the wall-function values after one step
// ===========================================================================

/// One `step_rans` on a plug flow between a resting floor and a moving lid:
/// the floor row carries the closed form of `u_rel = u`, the lid row that of
/// `u_rel = lid − u`, for both transport closures, in the log region and in
/// the sublayer.
///
/// Tolerance `1e-9` relative: the f64 inversion converges to ~1e-16, and the
/// crate's inversion is 64 fixed bisections of a bracket below `2 u_rel` in
/// `Fix128` (width `2⁻⁶³ u_rel`) with the deterministic `ln`; the remaining
/// error is the f64 evaluation of the closed form (~1e-15).
#[test]
fn wall_cells_carry_the_launder_spalding_values_after_one_step() {
    let (nx, ny) = (4usize, 4usize);
    let dx = q(1, 4);
    let y_p = dx.to_f64() / 2.0;
    // (ν, plug speed, lid speed): log region (y⁺ ~ 10³) and sublayer (y⁺ < 1)
    let cases = [(q(1, 100_000), 2, 5), (q(1, 10), 1, 3)];
    for (nu, plug, lid) in cases {
        for model in [TurbulenceModel::KEpsilon, TurbulenceModel::KOmega] {
            let mut s = channel(nx, ny, dx, nu, Fix128::from_int(lid));
            set_profile(&mut s, &|_| Fix128::from_int(plug));
            let mut state = RansState::uniform(nx, ny, 1, model, q(1, 100), q(1, 100));
            let report = s
                .step_rans(q(1, 1000), &opts(), &mut state)
                .expect("the channel steps");
            let nu_f = nu.to_f64();
            let floor_tau = log_law_u_tau(plug as f64, y_p, nu_f);
            let lid_tau = log_law_u_tau((lid - plug) as f64, y_p, nu_f);
            for (row, u_tau) in [(0usize, floor_tau), (ny - 1, lid_tau)] {
                let (k_want, e_want) = wall_k_eps(u_tau, y_p);
                for i in 0..nx {
                    let (k, e) = (
                        state.k(i, row, 0).to_f64(),
                        state.epsilon(i, row, 0).to_f64(),
                    );
                    assert!(
                        (k - k_want).abs() <= 1e-9 * k_want,
                        "{model:?} ν {nu_f} cell ({i},{row}): k {k} vs u_τ²/√C_μ {k_want}"
                    );
                    assert!(
                        (e - e_want).abs() <= 1e-9 * e_want,
                        "{model:?} ν {nu_f} cell ({i},{row}): ε {e} vs u_τ³/(κ y) {e_want}"
                    );
                    // ν_t = C_μ k²/ε = κ u_τ y_p in the wall cell
                    let nu_t = state.eddy_viscosity(i, row, 0).to_f64();
                    let nu_t_want = KAPPA * u_tau * y_p;
                    assert!(
                        (nu_t - nu_t_want).abs() <= 1e-9 * nu_t_want,
                        "{model:?} cell ({i},{row}): ν_t {nu_t} vs κ u_τ y {nu_t_want}"
                    );
                }
            }
            // The wall model's report reads the same friction velocities.
            let w = report.wall.expect("wall summary");
            let (lo, hi) = (floor_tau.min(lid_tau), floor_tau.max(lid_tau));
            assert!((w.u_tau_min.to_f64() - lo).abs() <= 1e-9 * lo);
            assert!((w.u_tau_max.to_f64() - hi).abs() <= 1e-9 * hi);
            assert!((w.k_max.to_f64() - wall_k_eps(hi, y_p).0).abs() <= 1e-9 * w.k_max.to_f64());
        }
    }
}

/// A wall the fluid does not move against (`u_rel = 0`) has `u_τ = 0` and
/// so `k = ε = 0` in its cells; a slip wall is not a no-slip wall and leaves
/// its cells to the transport.
#[test]
fn a_resting_wall_zeroes_its_cells_and_a_slip_wall_is_not_a_wall_function_wall() {
    let (nx, ny) = (3usize, 3usize);
    let dx = q(1, 4);
    let mut s = channel(nx, ny, dx, q(1, 1000), Fix128::ZERO);
    let mut state = RansState::uniform(nx, ny, 1, TurbulenceModel::KEpsilon, q(1, 4), q(1, 8));
    s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    for i in 0..nx {
        for j in [0, ny - 1] {
            assert_eq!(state.k(i, j, 0), Fix128::ZERO, "wall cell ({i},{j}) k");
            assert_eq!(
                state.epsilon(i, j, 0),
                Fix128::ZERO,
                "wall cell ({i},{j}) ε"
            );
        }
        // the middle row touches only the slip z faces
        assert!(state.k(i, 1, 0) > Fix128::ZERO, "middle row k was zeroed");
    }
}

/// Without a `WallModel` the step resolves the wall with the no-slip ghost
/// and no wall function is imposed on `(k, ε)`: the resting-wall cells keep
/// what the transport made of them.
#[test]
fn without_a_wall_model_the_transport_is_left_alone() {
    let (nx, ny) = (3usize, 3usize);
    let dx = q(1, 4);
    let mut s = channel(nx, ny, dx, q(1, 1000), Fix128::ZERO);
    let mut state = RansState::uniform(nx, ny, 1, TurbulenceModel::KEpsilon, q(1, 4), q(1, 8));
    let plain = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 10 });
    s.step_rans(q(1, 1000), &plain, &mut state).expect("steps");
    assert!(state.k(0, 0, 0) > Fix128::ZERO);
}

// ===========================================================================
// Oracle 2 — turbulent Couette flow: the first cell off the wall
// ===========================================================================

struct CouetteResult {
    u_tau: f64,
    k1: f64,
    eps1: f64,
    y1: f64,
    y_plus_wall: f64,
    /// `u(j = 1) − u(j = 0)` at the end (face values, `x`-uniform).
    du1: f64,
    /// Largest change of `k` at `j = 1` over the last `steps / 6` steps,
    /// relative to `u_τ²/√C_μ`: how far from steady the readout is.
    drift: f64,
}

fn couette_steady(ny: usize, model: TurbulenceModel, steps: u32, dt: Fix128) -> CouetteResult {
    let dx = Fix128::ONE / Fix128::from_int(ny as i64);
    let nu = q(1, 100_000);
    let mut s = channel(1, ny, dx, nu, Fix128::ONE);
    let dxf = dx.to_f64();
    set_profile(&mut s, &|j| Fix128::from_f64((j as f64 + 0.5) * dxf));
    let mut state = RansState::uniform(1, ny, 1, model, q(1, 1000), q(1, 1000));
    let mut last = None;
    let mut k_tail = (f64::INFINITY, f64::NEG_INFINITY);
    for step in 0..steps {
        let r = s
            .step_rans(dt, &opts(), &mut state)
            .unwrap_or_else(|e| panic!("step {step}: {e}"));
        last = r.wall;
        if step >= steps - steps / 6 {
            let k1 = state.k(0, 1, 0).to_f64();
            k_tail = (k_tail.0.min(k1), k_tail.1.max(k1));
        }
    }
    let w = last.expect("wall summary");
    let u_tau = 0.5 * (w.u_tau_min.to_f64() + w.u_tau_max.to_f64());
    CouetteResult {
        u_tau,
        k1: state.k(0, 1, 0).to_f64(),
        eps1: state.epsilon(0, 1, 0).to_f64(),
        y1: 1.5 * dxf,
        y_plus_wall: w.y_plus_min.to_f64(),
        du1: s.grid.u(0, 1, 0).to_f64() - s.grid.u(0, 0, 0).to_f64(),
        drift: (k_tail.1 - k_tail.0) * C_MU.sqrt() / (u_tau * u_tau),
    }
}

/// Plane Couette flow, `H = 1`, `ny = 8`, lid speed `1`, `ν = 1e-5`, from a
/// laminar start to the steady state (`dt = 1/4`, `t = 500`, about four
/// diffusion times `H²/ν_t`; the diffusion number stays below `0.07`), read at the first cell **off** the wall (`y = 3dx/2`), which the
/// wall function does not set: `k √C_μ / u_τ²` and `ε κ y / u_τ³` have to be
/// `1` there for the log-layer equilibrium.
///
/// Tolerance `0.1`: measured with both closures at `ny = 8` and `16`
/// (`dt = 1/32`) and at `ny = 8` (`dt = 1/4`), the ratios lie in
/// `[0.938, 1.031]` (k) and `[0.991, 1.032]` (ε); they do not move with `ny`
/// (the log layer is self-similar) and by up to `0.04` with `dt` (the
/// explicit point sources); the remaining few percent are
/// the standard constants (`σ_ε = 1.3` makes the log law an exact solution
/// only for `κ ≈ 0.433`, not `0.41`) and the finite differences from the
/// third cell on. Before the wall function the same readout was `1.33` (k)
/// and `1.44` (ε), and without the log-layer edge viscosity and strain at the
/// first cells `1.13` / `1.17`. The run is checked to be steady: `k` at
/// `j = 1` moves by less than `1e-2` of `u_τ²/√C_μ` over its last sixth, a
/// tenth of the tolerance (measured `3.3e-4` for k-ε and `2.9e-3` for k-ω,
/// whose explicit `ω` source at `dt = 1/4` leaves a small oscillation).
#[test]
fn couette_flow_carries_the_log_layer_k_and_epsilon_off_the_wall() {
    for model in [TurbulenceModel::KEpsilon, TurbulenceModel::KOmega] {
        let r = couette_steady(8, model, 2_000, q(1, 4));
        let k_ratio = r.k1 * C_MU.sqrt() / (r.u_tau * r.u_tau);
        let e_ratio = r.eps1 * KAPPA * r.y1 / (r.u_tau * r.u_tau * r.u_tau);
        eprintln!(
            "{model:?}: u_tau {} y+ {} k ratio {k_ratio} eps ratio {e_ratio} drift {}",
            r.u_tau, r.y_plus_wall, r.drift
        );
        assert!(r.y_plus_wall > 30.0, "first cell not in the log layer");
        assert!(r.drift < 1e-2, "{model:?}: not steady, drift {}", r.drift);
        assert!(
            (k_ratio - 1.0).abs() < 0.1,
            "{model:?}: k √C_μ / u_τ² = {k_ratio}"
        );
        assert!(
            (e_ratio - 1.0).abs() < 0.1,
            "{model:?}: ε κ y / u_τ³ = {e_ratio}"
        );
    }
}

// ===========================================================================
// Degenerate input
// ===========================================================================

/// One cell between the floor and the lid touches both walls: it takes the
/// mean of the two wall-function values (the documented corner rule), here
/// with `u_rel = 2` against the floor and `3` against the lid.
#[test]
fn a_cell_between_two_walls_takes_the_mean_of_the_two_wall_values() {
    let dx = q(1, 4);
    let y_p = dx.to_f64() / 2.0;
    let nu = q(1, 100_000);
    let mut s = channel(2, 1, dx, nu, Fix128::from_int(5));
    set_profile(&mut s, &|_| Fix128::from_int(2));
    let mut state = RansState::uniform(2, 1, 1, TurbulenceModel::KEpsilon, q(1, 100), q(1, 100));
    s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    let (kf, ef) = wall_k_eps(log_law_u_tau(2.0, y_p, nu.to_f64()), y_p);
    let (kl, el) = wall_k_eps(log_law_u_tau(3.0, y_p, nu.to_f64()), y_p);
    let (k_want, e_want) = (0.5 * (kf + kl), 0.5 * (ef + el));
    for i in 0..2 {
        let (k, e) = (state.k(i, 0, 0).to_f64(), state.epsilon(i, 0, 0).to_f64());
        assert!((k - k_want).abs() <= 1e-9 * k_want, "k {k} vs {k_want}");
        assert!((e - e_want).abs() <= 1e-9 * e_want, "ε {e} vs {e_want}");
    }
}

/// A three-cell channel has walls in reach on both sides of its middle row:
/// the log-law gradient is not used there (documented: the finite difference
/// stays), the step is finite, the wall rows carry the closed form and the
/// middle row is left to the transport.
#[test]
fn a_three_cell_channel_keeps_the_finite_difference_in_the_middle_row() {
    let dx = q(1, 4);
    let y_p = dx.to_f64() / 2.0;
    let nu = q(1, 100_000);
    let mut s = channel(2, 3, dx, nu, Fix128::from_int(5));
    set_profile(&mut s, &|j| Fix128::from_int(2 + j as i64));
    let k0 = q(1, 100);
    let mut state = RansState::uniform(2, 3, 1, TurbulenceModel::KEpsilon, k0, q(1, 100));
    s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    let (kf, _) = wall_k_eps(log_law_u_tau(2.0, y_p, nu.to_f64()), y_p);
    let (kl, _) = wall_k_eps(log_law_u_tau(1.0, y_p, nu.to_f64()), y_p);
    for i in 0..2 {
        assert!((state.k(i, 0, 0).to_f64() - kf).abs() <= 1e-9 * kf);
        assert!((state.k(i, 2, 0).to_f64() - kl).abs() <= 1e-9 * kl);
        let mid = state.k(i, 1, 0).to_f64();
        assert!(mid.is_finite() && mid > 0.0 && mid != kf && mid != kl);
    }
}

/// `dx = 0`: there is no wall distance. The step is refused up front with the
/// pressure solver's `ZeroSpacing` and the state is untouched, so the wall
/// function (which would also skip a non-positive `y_p`) is never reached.
#[test]
fn a_zero_spacing_channel_does_not_reach_the_wall_function() {
    let mut s = channel(2, 2, Fix128::ZERO, q(1, 1000), Fix128::ONE);
    let mut state = RansState::uniform(2, 2, 1, TurbulenceModel::KEpsilon, q(1, 4), q(1, 8));
    let before = state.clone();
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        s.step_rans(q(1, 1000), &opts(), &mut state).map(|_| ())
    }));
    eprintln!("dx = 0: {out:?}");
    assert!(
        matches!(
            out,
            Ok(Err(StepError::Pressure(PressureSolverError::ZeroSpacing)))
        ),
        "dx = 0 must be refused, got {out:?}"
    );
    assert_eq!(state, before);
}

/// The same Couette flow, read in the momentum: between the wall cell
/// (`y = dx/2`) and the next (`y = 3dx/2`) the log law puts a velocity
/// difference `(u_τ/κ) ln 3`. The step carries the shear stress `ρ u_τ²`
/// across that interface with the edge viscosity of the log layer at the
/// interface height, `κ u_τ dx`, so the discrete difference is `u_τ/κ`, i.e.
/// `1/ln 3 = 0.910` of the continuum one (a two-point flux cannot carry the
/// curvature of `ln y`); measured `0.902` for both closures. The harmonic
/// mean of the two cells' `ν_t` (`0.75 κ u_τ dx` at the interface) would give
/// `4/3` of `u_τ/κ`, `1.21` of the continuum value (measured `1.189`).
/// Tolerance `0.12` around `1` admits the first and not the second.
#[test]
fn couette_flow_has_the_log_law_velocity_step_off_the_wall() {
    for model in [TurbulenceModel::KEpsilon, TurbulenceModel::KOmega] {
        let r = couette_steady(8, model, 2_000, q(1, 4));
        let ratio = r.du1 * KAPPA / (r.u_tau * ln64(3.0));
        eprintln!("{model:?}: du1 {} u_tau {} ratio {ratio}", r.du1, r.u_tau);
        assert!(
            (ratio - 1.0).abs() < 0.12,
            "{model:?}: (u1 − u0) κ / (u_τ ln 3) = {ratio}"
        );
    }
}

/// A velocity component normal to the wall does not enter `u_rel`: the
/// floor cells of a plug flow with a uniform `v` on the interior Y-faces
/// (the cell-centred `v` of the floor row is then `v/2`) carry the same
/// wall-function values as without it. A uniform `u` is advected to itself
/// whatever `v` is, so the speed the step reads is the plug speed.
#[test]
fn a_normal_velocity_does_not_enter_the_wall_function() {
    let (nx, ny) = (3usize, 4usize);
    let dx = q(1, 4);
    let y_p = dx.to_f64() / 2.0;
    let nu = q(1, 100_000);
    let mut s = channel(nx, ny, dx, nu, Fix128::from_int(2));
    set_profile(&mut s, &|_| Fix128::from_int(2));
    for j in 1..ny {
        for i in 0..nx {
            s.grid.v[i + nx * j] = Fix128::ONE;
        }
    }
    let mut state = RansState::uniform(nx, ny, 1, TurbulenceModel::KEpsilon, q(1, 100), q(1, 100));
    s.step_rans(q(1, 1000), &opts(), &mut state).expect("steps");
    let (k_want, e_want) = wall_k_eps(log_law_u_tau(2.0, y_p, nu.to_f64()), y_p);
    for i in 0..nx {
        let (k, e) = (state.k(i, 0, 0).to_f64(), state.epsilon(i, 0, 0).to_f64());
        assert!((k - k_want).abs() <= 1e-9 * k_want, "k {k} vs {k_want}");
        assert!((e - e_want).abs() <= 1e-9 * e_want, "ε {e} vs {e_want}");
    }
}
