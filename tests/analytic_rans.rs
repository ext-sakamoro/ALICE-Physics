//! Oracles for the RANS closures wired into `CfdSolver::step_rans`:
//! `TurbulenceModel::{Smagorinsky, DynamicSmagorinsky, KEpsilon, KOmega}`,
//! the caller-owned `RansState` of `k` and `ε`, and the eddy viscosity that
//! reaches the momentum diffusion.
//!
//! # Closed forms
//!
//! * **Decaying homogeneous turbulence (k-ε)**: with no production and a
//!   uniform field the transport equations reduce to the point ODE
//!   `dk/dt = −ε`, `dε/dt = −C_ε2 ε²/k`, whose solution is
//!   `k(t) = k0 (1 + (C_ε2 − 1) ε0 t / k0)^(−1/(C_ε2 − 1))` and
//!   `ε(t) = ε0 (k/k0)^C_ε2` (Launder & Spalding 1974; the exponent is
//!   `1/0.92 = 1.0870` for `C_ε2 = 1.92`). The solver integrates with explicit
//!   Euler, so the discrete answer is first-order accurate: halving `dt`
//!   halves the error, which is what the refinement sweep asserts, with an
//!   absolute bound on the finest level.
//! * **Decaying homogeneous turbulence (k-ω)**: `dk/dt = −β* k ω`,
//!   `dω/dt = −β ω²` give `ω(t) = ω0 / (1 + β ω0 t)` and
//!   `k(t) = k0 (1 + β ω0 t)^(−β*/β)`, with `β*/β = 0.09/0.075 = 1.2`
//!   (Wilcox 1988).
//! * **One step of the k equation is exact**: `k1 = k0 − ε0 dt` for dyadic
//!   inputs, bit for bit, and a uniform field stays uniform to the bit (the
//!   diffusion and the advection of a uniform scalar on a resting fluid are
//!   the identity).
//! * **The two closures agree on the eddy viscosity**: `ω = ε/(β* k)` makes
//!   `k/ω = C_μ k²/ε`; the two quotients differ by rounding only, and the
//!   bound is derived in the test.
//! * **Dynamic Smagorinsky collapses to the static one in uniform strain**:
//!   the test-filtered strain equals the grid strain, the ratio is exactly
//!   `1`, and the step is bit-identical to `TurbulenceModel::Smagorinsky`.
//!
//! # Degenerate input
//!
//! A step with a `RansState` of the wrong shape or spacing is
//! refused with the solver untouched; a cell whose `k` or `ε` is zero keeps a
//! zero eddy viscosity and never divides by zero; a negative value coming out
//! of the explicit source step is clamped to zero (the existing point-model
//! contract), which the decay sweep at a coarse `dt` exercises.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, RansState, StepError, StepOptions, TurbulenceModel,
};
use alice_physics::det_math::powf64;
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::Grid3d;

fn q(n: i64, d: i64) -> Fix128 {
    Fix128::from_ratio(n, d)
}

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

const C_EPS1: f64 = 1.44;
const C_EPS2: f64 = 1.92;
const C_MU: f64 = 0.09;
const BETA_STAR: f64 = 0.09;
const BETA: f64 = 0.075;
const KAPPA: f64 = 0.41;

/// A resting sealed box of unit cells: no production anywhere, so the RANS
/// field only decays. Unit spacing keeps the explicit diffusion number
/// `ν_t dt / dx²` of the decay scenes (`ν_t ≤ 0.72`, `dt ≤ 1/8`) under `1/6`.
fn resting(n: usize) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, n, Fix128::ONE);
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = q(1, 1000);
    s.jacobi_iterations = 10;
    s.grid.set_closed_box_walls();
    s
}

fn opts() -> StepOptions {
    StepOptions::new(PressureSolver::RedBlackGs { sweeps: 10 })
}

fn k_eps_decay(k0: f64, e0: f64, t: f64) -> (f64, f64) {
    let base = 1.0 + (C_EPS2 - 1.0) * e0 * t / k0;
    let k = k0 * powf64(base, -1.0 / (C_EPS2 - 1.0));
    (k, e0 * powf64(k / k0, C_EPS2))
}

fn k_omega_decay(k0: f64, w0: f64, t: f64) -> (f64, f64) {
    let base = 1.0 + BETA * w0 * t;
    (k0 * powf64(base, -BETA_STAR / BETA), w0 / base)
}

/// Run `steps` RANS steps of `dt` on a resting box with a uniform field and
/// return `(k, ε)` of a cell, asserting on the way that every cell agrees to
/// the bit.
fn decay_run(
    model: TurbulenceModel,
    k0: Fix128,
    eps0: Fix128,
    dt: Fix128,
    steps: u32,
) -> (Fix128, Fix128) {
    let n = 3usize;
    let mut s = resting(n);
    let mut f = RansState::uniform(n, n, n, model, k0, eps0);
    for _ in 0..steps {
        let report = s
            .step_rans(dt, &opts(), &mut f)
            .expect("a resting box with a field must step");
        let t = report.turbulence;
        assert_eq!(t.model, model);
        // The explicit source step stayed inside the field's range: no clamp
        // fired, so the closed form is being integrated, not a clipped one.
        assert_eq!(t.clamped, 0, "a clamp fired at dt = {dt:?}");
        assert_eq!(
            t.production_max,
            Fix128::ZERO,
            "a resting fluid produces nothing"
        );
    }
    let (k, e) = (f.k(0, 0, 0), f.epsilon(0, 0, 0));
    for c in 0..n {
        for b in 0..n {
            for a in 0..n {
                assert_eq!(f.k(a, b, c), k, "k is not uniform at ({a}, {b}, {c})");
                assert_eq!(f.epsilon(a, b, c), e, "ε is not uniform at ({a}, {b}, {c})");
            }
        }
    }
    (k, e)
}

// ===========================================================================
// Oracle 1 — k-ε decay against the closed form, first-order in dt
// ===========================================================================

#[test]
fn k_epsilon_decay_follows_the_closed_form_to_first_order() {
    let (k0, e0) = (int(2), q(1, 2));
    let horizon = 2.0_f64; // s
    let (k_ref, e_ref) = k_eps_decay(2.0, 0.5, horizon);
    let mut errors = Vec::new();
    for (dt, steps) in [(q(1, 8), 16), (q(1, 16), 32), (q(1, 32), 64)] {
        let (k, e) = decay_run(TurbulenceModel::KEpsilon, k0, e0, dt, steps);
        let ek = (k.to_f64() - k_ref).abs();
        let ee = (e.to_f64() - e_ref).abs();
        errors.push((ek, ee));
    }
    for w in errors.windows(2) {
        let (rk, re) = (w[0].0 / w[1].0, w[0].1 / w[1].1);
        assert!(
            (1.7..=2.3).contains(&rk),
            "k error ratio {rk} is not first order: {errors:?}"
        );
        assert!(
            (1.7..=2.3).contains(&re),
            "ε error ratio {re} is not first order: {errors:?}"
        );
    }
    // Finest level: the error is a few per mille of the value.
    assert!(errors[2].0 < 0.01 * k_ref, "{errors:?}");
    assert!(errors[2].1 < 0.02 * e_ref, "{errors:?}");
    // Not vacuous: the field did decay.
    assert!(k_ref < 1.5 && e_ref < 0.25);
}

// ===========================================================================
// Oracle 2 — k-ω decay against the closed form, first-order in dt
// ===========================================================================

#[test]
fn k_omega_decay_follows_the_closed_form_to_first_order() {
    // The field stores (k, ε); ω = ε / (β* k). k0 = 2, ω0 = 4 ⇒ ε0 = 0.72.
    let (k0, w0) = (2.0_f64, 4.0_f64);
    let eps0 = Fix128::from_f64(BETA_STAR * k0 * w0);
    let horizon = 2.0_f64;
    let (k_ref, w_ref) = k_omega_decay(k0, w0, horizon);
    let mut errors = Vec::new();
    for (dt, steps) in [(q(1, 8), 16), (q(1, 16), 32), (q(1, 32), 64)] {
        let (k, e) = decay_run(TurbulenceModel::KOmega, int(2), eps0, dt, steps);
        let w = e.to_f64() / (BETA_STAR * k.to_f64());
        errors.push(((k.to_f64() - k_ref).abs(), (w - w_ref).abs()));
    }
    for w in errors.windows(2) {
        let (rk, rw) = (w[0].0 / w[1].0, w[0].1 / w[1].1);
        assert!(
            (1.7..=2.3).contains(&rk),
            "k error ratio {rk} is not first order: {errors:?}"
        );
        assert!(
            (1.7..=2.3).contains(&rw),
            "ω error ratio {rw} is not first order: {errors:?}"
        );
    }
    assert!(errors[2].0 < 0.01 * k_ref, "{errors:?}");
    assert!(errors[2].1 < 0.02 * w_ref, "{errors:?}");
}

// ===========================================================================
// Oracle 3 — one k step is exact; a uniform field stays uniform to the bit
// ===========================================================================

#[test]
fn one_step_of_k_without_production_is_k0_minus_eps0_dt_exactly() {
    let (k0, e0, dt) = (q(5, 2), q(3, 8), q(1, 16));
    let (k1, _) = decay_run(TurbulenceModel::KEpsilon, k0, e0, dt, 1);
    assert_eq!(k1, k0 - e0 * dt);
    // The same under k-ω: dk = −β* k ω dt = −ε dt, so the first step is the
    // same number up to the rounding of ω = ε/(β* k) and back. Pin it to one
    // ulp of the exact value.
    let (k1w, _) = decay_run(TurbulenceModel::KOmega, k0, e0, dt, 1);
    let ulp = Fix128::from_raw(0, 1);
    let diff = if k1w > k1 { k1w - k1 } else { k1 - k1w };
    assert!(diff <= ulp * int(8), "k-ω first step differs by {diff:?}");
}

// ===========================================================================
// Oracle 4 — the two closures agree on the eddy viscosity
// ===========================================================================

#[test]
fn k_epsilon_and_k_omega_give_the_same_eddy_viscosity_up_to_rounding() {
    let n = 2usize;
    for (k, e) in [
        (int(1), q(9, 100)),
        (q(3, 2), q(1, 4)),
        (int(4), int(1)),
        (q(1, 8), q(1, 64)),
    ] {
        let ke = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, k, e);
        let kw = RansState::uniform(n, n, n, TurbulenceModel::KOmega, k, e);
        let nu_ke = ke.eddy_viscosity(0, 0, 0);
        let nu_kw = kw.eddy_viscosity(0, 0, 0);
        let want = 0.09 * k.to_f64() * k.to_f64() / e.to_f64();
        assert!(
            (nu_ke.to_f64() - want).abs() <= 1e-12 * want,
            "k-ε ν_t {nu_ke:?} vs {want}"
        );
        // Two divisions and one product of the same quantities: a handful of ulp.
        let diff = (nu_ke.to_f64() - nu_kw.to_f64()).abs();
        assert!(diff <= 1e-15 * want.max(1.0), "k-ω ν_t differs by {diff}");
    }
    // Zero k or ε: zero eddy viscosity, no division by zero.
    for (k, e) in [
        (Fix128::ZERO, int(1)),
        (int(1), Fix128::ZERO),
        (Fix128::ZERO, Fix128::ZERO),
    ] {
        let ke = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, k, e);
        let kw = RansState::uniform(n, n, n, TurbulenceModel::KOmega, k, e);
        assert_eq!(ke.eddy_viscosity(1, 1, 1), Fix128::ZERO);
        assert_eq!(kw.eddy_viscosity(1, 1, 1), Fix128::ZERO);
    }
}

// ===========================================================================
// Oracle 5 — dynamic Smagorinsky equals the static one under uniform strain
// ===========================================================================

/// A linear shear `u = y` between a resting floor and a lid at `u = 1`, the
/// profile prescribed as an inflow on the `i = 0` layer and an outflow on
/// `i = n` (the streamwise boundaries of the repo's Couette oracles; a slip
/// wall there would pin the wall face's normal velocity to zero and break
/// the uniformity in `x`). Every cell has the same strain, so the test filter
/// sees the same magnitude as the grid filter and `C_s` is the static
/// constant exactly.
fn couette(n: usize) -> CfdSolver {
    couette_with_shear(n, 1)
}

/// [`couette`] with the lid at `shear` and the profile `u = shear · y`.
fn couette_with_shear(n: usize, shear: i64) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, 1, Fix128::ONE / int(n as i64));
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = q(1, 100);
    s.jacobi_iterations = 20;
    s.grid.set_closed_box_walls();
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(int(shear), Fix128::ZERO, Fix128::ZERO),
    };
    for i in 0..n {
        s.grid.set_v_bc(i, n, 0, lid);
    }
    let dx = s.grid.dx;
    for j in 0..n {
        let y = (int(j as i64) + q(1, 2)) * dx * int(shear);
        s.grid
            .set_u_bc(0, j, 0, FaceBc::Inflow { normal_velocity: y });
        s.grid.set_u_bc(n, j, 0, FaceBc::Outflow);
        for i in 0..=n {
            let ix = i + (n + 1) * j;
            s.grid.u[ix] = y;
        }
    }
    s
}

#[test]
fn dynamic_smagorinsky_is_the_static_one_under_uniform_strain_and_differs_otherwise() {
    let n = 6usize;
    let dt = q(1, 64);
    let mut a = couette(n);
    let mut b = couette(n);
    let mut sa = RansState::new(n, n, 1, TurbulenceModel::Smagorinsky);
    let mut sb = RansState::new(n, n, 1, TurbulenceModel::DynamicSmagorinsky);
    a.step_rans(dt, &opts(), &mut sa).unwrap();
    b.step_rans(dt, &opts(), &mut sb).unwrap();
    assert_eq!(
        a.grid.u, b.grid.u,
        "uniform strain: the dynamic coefficient must be C_s"
    );
    assert_eq!(a.grid.v, b.grid.v);
    // Non-uniform strain: a jet in the middle row.
    let mut c = couette(n);
    let mut d = couette(n);
    for s in [&mut c, &mut d] {
        for i in 0..=n {
            let ix = i + (n + 1) * (n / 2);
            s.grid.u[ix] = int(3);
        }
    }
    c.step_rans(dt, &opts(), &mut sa).unwrap();
    d.step_rans(dt, &opts(), &mut sb).unwrap();
    assert_ne!(
        c.grid.u, d.grid.u,
        "a jet must make the dynamic coefficient move"
    );
}

// ===========================================================================
// Oracle 6 — the eddy viscosity field reaches the momentum equation
// ===========================================================================

/// Two solvers with the same velocity and the same mean `ν_t`, one with a
/// uniform field and one with the viscosity concentrated in the lower half
/// of the box. The shear flux `ν ∂u/∂y` then jumps at the interface, so if
/// the per-cell viscosity reaches the momentum diffusion the two steps
/// differ; if only a grid-wide proxy does, they agree. (A split along `x`
/// would not do: `∂(ν(x) · 1)/∂y = 0` leaves a linear profile alone.)
#[test]
fn a_non_uniform_eddy_viscosity_changes_the_velocity_step() {
    let n = 4usize;
    let dt = q(1, 64);
    let mut uniform = couette(n);
    let mut split = couette(n);
    // ν_t = C_μ k²/ε: k = 1/4, ε = 9/100 gives ν_t = 1/16 everywhere ...
    let mut uniform_state =
        RansState::uniform(n, n, 1, TurbulenceModel::KEpsilon, q(1, 4), q(9, 100));
    // ... and k = 1/2 in the lower half, 0 in the upper, gives 1/4 and 0:
    // the same mean, a different field. Both are within the diffusion limit
    // of this grid (`dx = 1/4`, `dt = 1/64`: `ν ≤ 2/3`).
    let mut field = RansState::uniform(n, n, 1, TurbulenceModel::KEpsilon, Fix128::ZERO, q(9, 100));
    for j in 0..n / 2 {
        for i in 0..n {
            field.set(i, j, 0, q(1, 2), q(9, 100));
        }
    }
    uniform.step_rans(dt, &opts(), &mut uniform_state).unwrap();
    split.step_rans(dt, &opts(), &mut field).unwrap();
    assert_ne!(
        uniform.grid.u, split.grid.u,
        "the ν_t field never reached the momentum step"
    );
}

// ===========================================================================
// Refusals
// ===========================================================================

#[test]
fn a_rans_step_with_a_state_of_the_wrong_shape_is_refused_untouched() {
    for model in [
        TurbulenceModel::KEpsilon,
        TurbulenceModel::KOmega,
        TurbulenceModel::Smagorinsky,
    ] {
        let mut s = resting(3);
        let mut state = RansState::uniform(2, 3, 3, model, int(1), int(1));
        let before = (s.grid.u.clone(), s.grid.pressure.clone(), s.step_count);
        let state_before = state.clone();
        let err = s.step_rans(q(1, 16), &opts(), &mut state).unwrap_err();
        assert_eq!(err, StepError::TurbulenceFieldShape);
        assert_eq!(
            (s.grid.u.clone(), s.grid.pressure.clone(), s.step_count),
            before
        );
        assert_eq!(
            state, state_before,
            "a refused step must not advance the state"
        );
    }
    // The LES models carry no field; a state of the right shape runs.
    let mut s = resting(3);
    let mut state = RansState::new(3, 3, 3, TurbulenceModel::Smagorinsky);
    s.step_rans(q(1, 16), &opts(), &mut state).unwrap();
    assert_eq!(s.step_count, 1);
    assert_eq!(state.model(), TurbulenceModel::Smagorinsky);
}

// ===========================================================================
// Oracle 7 — the clamp is reported when the explicit step overshoots
// ===========================================================================

#[test]
fn an_overshooting_explicit_step_is_clamped_and_counted() {
    let n = 2usize;
    let mut s = resting(n);
    // k = 1, ε = 100, dt = 1: the k update would be 1 − 100 < 0 in every
    // cell. ν_t = 0.09 / 100 keeps the diffusion stable, so the step runs.
    let mut state = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, int(1), int(100));
    let report = s.step_rans(int(1), &opts(), &mut state).unwrap();
    let t = report.turbulence;
    assert_eq!(
        t.clamped,
        (n * n * n) as u32,
        "every cell's k went negative and was clamped"
    );
    assert_eq!(t.k_min, Fix128::ZERO);
    assert_eq!(t.k_max, Fix128::ZERO);
}

// ===========================================================================
// Oracle 8 — uniform shear: P/ε tends to (C_ε2 − 1)/(C_ε1 − 1)
// ===========================================================================

/// In a uniform shear `S` with a uniform field, `k` and `ε` obey the point
/// ODEs with `P = C_μ k² S²/ε`, and the ratio `r = P/ε` satisfies
/// `d ln r/dt = (2ε/k) [(1 − C_ε1) r + (C_ε2 − 1)]`, so `r → (C_ε2 − 1)/(C_ε1 − 1)`
/// `= 0.92/0.44 = 2.0909`. This is the one closed form with teeth on `C_ε1`.
/// The scene is a plane shear `u = S y`, which the momentum step leaves alone
/// whatever `ν_t` does.
///
/// Convergence rate: along the way `T = k/ε` settles at `√(r*/C_μ) / S`
/// (`4.82 / S` seconds, independent of the initial field) and `r` relaxes at
/// `2 (C_ε1 − 1) / T`, so `S = 8` gives a time constant of `0.34 s` and four
/// seconds are twelve of them. Starting at `r = 1` (`ε0 = √C_μ k0 S`) keeps
/// `k` small enough (`×1500` over the run) for the explicit diffusion to stay
/// stable, and `dt = 1/256` keeps the explicit-Euler bias of the discrete
/// fixed point under one per cent.
#[test]
fn uniform_shear_drives_production_over_dissipation_to_the_constant_ratio() {
    let n = 4usize;
    let shear = 8i64;
    let mut s = couette_with_shear(n, shear);
    let k0 = 1e-3_f64;
    let mut f = RansState::uniform(
        n,
        n,
        1,
        TurbulenceModel::KEpsilon,
        Fix128::from_f64(k0),
        Fix128::from_f64(C_MU.sqrt() * k0 * shear as f64),
    );
    let dt = q(1, 256);
    let target = (C_EPS2 - 1.0) / (C_EPS1 - 1.0);
    let mut ratio = 0.0;
    for step in 0..(4 * 256) {
        let report = s
            .step_rans(dt, &opts(), &mut f)
            .unwrap_or_else(|e| panic!("step {step}: {e}"));
        let t = report.turbulence;
        assert_eq!(t.clamped, 0, "step {step}");
        // Still uniform: every cell sees the same shear. The semi-Lagrangian
        // advection of a uniform scalar is a convex combination evaluated in
        // `Fix128`, so cells can differ by a few ulp once `k` is no longer
        // dyadic; the spread is bounded relative to the value.
        let (k0, e0) = (f.k(0, 0, 0).to_f64(), f.epsilon(0, 0, 0).to_f64());
        for j in 0..n {
            for i in 0..n {
                assert!(
                    (f.k(i, j, 0).to_f64() - k0).abs() <= 1e-12 * k0,
                    "step {step}: k not uniform"
                );
                assert!(
                    (f.epsilon(i, j, 0).to_f64() - e0).abs() <= 1e-12 * e0,
                    "step {step}: ε not uniform"
                );
            }
        }
        ratio = t.production_max.to_f64() / f.epsilon(0, 0, 0).to_f64();
    }
    assert!(
        (ratio - target).abs() < 0.02 * target,
        "P/ε settled at {ratio}, closed form {target}"
    );
    // The profile is still the plane shear: the eddy viscosity cannot bend a
    // linear profile between walls that match it.
    let dx = s.grid.dx;
    for j in 0..n {
        let y = (int(j as i64) + q(1, 2)) * dx * int(shear);
        for i in 1..n {
            let got = s.grid.u(i, j, 0).to_f64();
            assert!(
                (got - y.to_f64()).abs() < 1e-9,
                "face ({i}, {j}): u = {got}, shear says {}",
                y.to_f64()
            );
        }
    }
}

// ===========================================================================
// Oracle 9 — wall-consistent (k, ε): the algebraic identities of the log layer
// ===========================================================================

/// The wall-consistent values are `k = u_τ²/√C_μ`, `ε = u_τ³/(κ y)` (what
/// `turbulence::wall_k_epsilon` produces for the wall model's report). Then
/// `ν_t = C_μ k²/ε = κ u_τ y` and, with the log-layer shear
/// `du/dy = u_τ/(κ y)`, `P = ν_t (du/dy)² = ε`: the k equation is stationary
/// in the log layer. (The ε equation is not — `(C_ε1 − C_ε2) ε²/k ≠ 0` — its
/// balance in the continuum comes from the diffusion of `ε ∝ 1/y` across many
/// cells, which one cell cannot represent, so no fixed point of ε is claimed.)
#[test]
fn wall_consistent_k_epsilon_satisfy_nu_t_equals_kappa_u_tau_y_and_production_equals_epsilon() {
    for (u_tau, y) in [(q(1, 2), q(1, 8)), (q(3, 4), q(1, 16)), (int(2), q(1, 4))] {
        let (ut, yp) = (u_tau.to_f64(), y.to_f64());
        let k = Fix128::from_f64(ut * ut / C_MU.sqrt());
        let e = Fix128::from_f64(ut * ut * ut / (KAPPA * yp));
        let field = RansState::uniform(1, 1, 1, TurbulenceModel::KEpsilon, k, e);
        let nu_t = field.eddy_viscosity(0, 0, 0).to_f64();
        let want = KAPPA * u_tau.to_f64() * y.to_f64();
        assert!(
            (nu_t - want).abs() < 1e-9 * want,
            "ν_t {nu_t} vs κ u_τ y {want}"
        );
        let shear = u_tau.to_f64() / (KAPPA * y.to_f64());
        let production = nu_t * shear * shear;
        assert!(
            (production - e.to_f64()).abs() < 1e-9 * e.to_f64(),
            "P {production} vs ε {}",
            e.to_f64()
        );
    }
}

// ===========================================================================
// Oracle 10 — diffusion of k crosses only an interface, by equal and opposite
// fluxes
// ===========================================================================

#[test]
fn k_diffuses_across_the_interface_only_and_the_fluxes_are_equal_and_opposite() {
    let n = 4usize;
    let mut s = resting(n);
    let (lo, hi, e0, dt) = (q(1, 2), int(1), q(1, 4), q(1, 16));
    let mut f = RansState::uniform(n, n, n, TurbulenceModel::KEpsilon, lo, e0);
    for k in 0..n {
        for j in n / 2..n {
            for i in 0..n {
                f.set(i, j, k, hi, e0);
            }
        }
    }
    s.step_rans(dt, &opts(), &mut f).unwrap();
    // Away from the interface the only change is the source, exactly.
    let (lo1, hi1) = (lo - e0 * dt, hi - e0 * dt);
    for k in 0..n {
        for i in 0..n {
            assert_eq!(f.k(i, 0, k), lo1, "row 0 is not touched by the interface");
            assert_eq!(
                f.k(i, n - 1, k),
                hi1,
                "row n-1 is not touched by the interface"
            );
            // The interface rows moved toward each other by the same amount.
            let d_lo = f.k(i, n / 2 - 1, k) - lo1;
            let d_hi = f.k(i, n / 2, k) - hi1;
            assert!(d_lo > Fix128::ZERO, "the low side must gain");
            assert!(d_hi < Fix128::ZERO, "the high side must lose");
            let imbalance = d_lo + d_hi;
            let ulp = Fix128::from_raw(0, 1);
            assert!(
                imbalance <= ulp && imbalance >= Fix128::ZERO - ulp,
                "fluxes not equal and opposite: {imbalance:?}"
            );
        }
    }
}

// ===========================================================================
// Oracle 11 — two-layer Couette with a prescribed eddy viscosity
// ===========================================================================

/// Lower half `ν₁ = 3/8`, upper half `ν₂ = 3/4`, floor at rest, lid at `1`,
/// slip side walls. The shear stress is uniform, `ν₁ u'₁ = ν₂ u'₂ = τ`, with
/// `τ = 1 / (H/(2ν₁) + H/(2ν₂)) = 1/2` for `H = 1`, so the profile at the face
/// centres is `u = τ y/ν₁ = 4y/3` below and `u = 2/3 + 2(y − 1/2)/3` above.
/// With the interface coefficient the harmonic mean `2ν₁ν₂/(ν₁+ν₂) = 1/2`,
/// the piecewise-linear profile is the discrete fixed point (an arithmetic
/// mean would put `9/16` there and bend it). Integrated from rest the
/// profile is reached to rounding, and the kinetic energy never increases
/// on the way (the diffusion is dissipative within its stability limit; the
/// advection of an `x`-uniform field and the projection of a divergence-free
/// one are the identity).
#[test]
fn a_two_layer_couette_flow_relaxes_to_the_piecewise_linear_profile() {
    let n = 4usize;
    let dx = q(1, 4);
    let mut s = CfdSolver::new(n, n, 1, dx);
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = Fix128::ZERO;
    s.jacobi_iterations = 20;
    s.grid.set_closed_box_walls();
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    };
    for i in 0..n {
        s.grid.set_v_bc(i, n, 0, lid);
    }
    let profile = |j: usize| -> f64 {
        let y = (j as f64 + 0.5) / n as f64;
        if y < 0.5 {
            4.0 * y / 3.0
        } else {
            2.0 / 3.0 + 2.0 * (y - 0.5) / 3.0
        }
    };
    // Streamwise: the profile prescribed at the inlet, an outflow at the
    // exit, as the repo's Couette oracles do; the interior has to find it.
    for j in 0..n {
        s.grid.set_u_bc(
            0,
            j,
            0,
            FaceBc::Inflow {
                normal_velocity: Fix128::from_f64(profile(j)),
            },
        );
        s.grid.set_u_bc(n, j, 0, FaceBc::Outflow);
    }
    let mut nu = Grid3d::new(n, n, 1, dx, q(3, 8));
    for j in n / 2..n {
        for i in 0..n {
            nu.set(i, j, 0, q(3, 4));
        }
    }
    let mut state = RansState::prescribed(nu);
    let dt = q(1, 128); // ν_max dt / dx² = 3/4 · 1/128 · 16 = 3/32 < 1/6
    let energy = |s: &CfdSolver| {
        s.grid
            .u
            .iter()
            .map(|u| u.to_f64() * u.to_f64())
            .sum::<f64>()
    };
    let mut e_prev = energy(&s);
    for step in 0..3000 {
        let report = s
            .step_rans(dt, &opts(), &mut state)
            .unwrap_or_else(|e| panic!("step {step}: {e}"));
        let t = report.turbulence;
        assert_eq!(t.nu_t_min, q(3, 8));
        assert_eq!(t.nu_t_max, q(3, 4));
        // The prescribed field is used as given and reported back unchanged.
        assert_eq!(report.eddy_viscosity.get(0, 0, 0), q(3, 8));
        assert_eq!(report.eddy_viscosity.get(0, n - 1, 0), q(3, 4));
        let e_now = energy(&s);
        // Energy comes in through the lid, so monotonicity is only claimed
        // once the lid has stopped feeding it: after the transient.
        if step > 1500 {
            assert!(
                e_now <= e_prev * (1.0 + 1e-12),
                "step {step}: energy rose {e_prev} → {e_now}"
            );
        }
        e_prev = e_now;
    }
    for j in 0..n {
        let want = profile(j);
        for i in 1..n {
            let got = s.grid.u(i, j, 0).to_f64();
            assert!(
                (got - want).abs() < 1e-12,
                "face ({i}, {j}): u = {got}, profile {want}"
            );
        }
    }
    assert_eq!(state.eddy_viscosity(0, 0, 0), q(3, 8));
    assert_eq!(state.eddy_viscosity(0, n - 1, 0), q(3, 4));
}

// ===========================================================================
// Refusals of the prescribed closure and the stability limit
// ===========================================================================

#[test]
fn a_prescribed_closure_refuses_a_mismatched_or_negative_field_and_an_unstable_step() {
    let n = 3usize;
    let mut s = resting(n);
    let before = s.grid.u.clone();
    let mut shape = RansState::prescribed(Grid3d::new(n, n, 2, Fix128::ONE, q(1, 10)));
    assert_eq!(
        s.step_rans(q(1, 16), &opts(), &mut shape).unwrap_err(),
        StepError::TurbulenceFieldShape
    );
    let mut spacing = RansState::prescribed(Grid3d::new(n, n, n, q(1, 2), q(1, 10)));
    assert_eq!(
        s.step_rans(q(1, 16), &opts(), &mut spacing).unwrap_err(),
        StepError::TurbulenceFieldShape,
        "a different spacing is a different field"
    );
    let mut negative = Grid3d::new(n, n, n, Fix128::ONE, q(1, 10));
    negative.set(1, 1, 1, q(-1, 10));
    let mut negative = RansState::prescribed(negative);
    assert_eq!(
        s.step_rans(q(1, 16), &opts(), &mut negative).unwrap_err(),
        StepError::NegativeEddyViscosity
    );
    // ν_t = 2 at dt = 1/8 on unit cells: diffusion number 1/4 > 1/6.
    let mut hot = RansState::prescribed(Grid3d::new(n, n, n, Fix128::ONE, int(2)));
    let err = s.step_rans(q(1, 8), &opts(), &mut hot).unwrap_err();
    match err {
        StepError::DiffusionUnstable { diffusion_number } => {
            assert_eq!(diffusion_number, q(1, 4) + q(1, 1000) * q(1, 8));
        }
        other => panic!("expected DiffusionUnstable, got {other:?}"),
    }
    // And the same field at a stable dt runs; the number is reported.
    let report = s.step_rans(q(1, 16), &opts(), &mut hot).unwrap();
    assert_eq!(
        report.turbulence.diffusion_number,
        q(1, 8) + q(1, 1000) * q(1, 16)
    );
    assert_eq!(s.step_count, 1);
    // Every refusal above left the grid untouched.
    let mut untouched = resting(n);
    let mut hot = RansState::prescribed(Grid3d::new(n, n, n, Fix128::ONE, int(2)));
    let _ = untouched.step_rans(q(1, 8), &opts(), &mut hot);
    assert_eq!(untouched.grid.u, before);
    assert_eq!(untouched.step_count, 0);
    // Display of every new refusal is non-empty and distinct.
    let texts: Vec<String> = [
        StepError::WallModelNeedsViscosity,
        StepError::TurbulenceFieldShape,
        StepError::NegativeEddyViscosity,
        StepError::DiffusionUnstable {
            diffusion_number: q(1, 4),
        },
    ]
    .iter()
    .map(ToString::to_string)
    .collect();
    for (a, ta) in texts.iter().enumerate() {
        assert!(!ta.is_empty());
        for (b, tb) in texts.iter().enumerate() {
            assert!(a == b || ta != tb);
        }
    }
}
