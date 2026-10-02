//! Oracles for the log-law wall model — `turbulence::friction_velocity` and
//! `CfdSolver::step_with_options` with a `WallModel` — which is what wires the
//! wall functions of `turbulence` (`y_plus`, `u_plus`, `wall_k_epsilon`) into a
//! solver step for the first time.
//!
//! # What is pinned, and where each expected value comes from
//!
//! 1. **The profile, by hand, at two points.** `u⁺(100) = ln(100)/0.41 + 5.5
//!    = 16.732` and `u⁺(1000) = 22.348`: two points fix both `κ` (from the
//!    difference, `ln 10 / κ`) and `B`. The test's own `f64` law is held to
//!    those literals first, and only then used to build inputs for the
//!    inverse — so checking the inverse against the forward law is not the
//!    circular "inverse against its own forward" (the Piola case), because the
//!    forward law is pinned outside the crate.
//! 2. **The sublayer edge is the intersection of the two branches.** The root
//!    of `y = ln(y)/κ + B` is solved here (`11.4453…`) and the crate's inverse
//!    is continuous through it: `u_rel` just below and just above the critical
//!    speed give friction velocities a bisection width apart. With the old edge
//!    at `11` the branches disagreed by `0.35` and a speed inside the step had
//!    no root.
//! 3. **Below the edge the model is the no-slip ghost.** In the sublayer
//!    `u_τ² = ν u_rel / y_p`, so the modelled sink `−dt u_τ²/dx` at `y_p = dx/2`
//!    is `−2 ν dt u_rel / dx²`, which is exactly the ghost's contribution. The
//!    no-slip side is pinned elsewhere against discrete Couette / Poiseuille
//!    closed forms, so agreeing with it is agreeing with an external basis; it
//!    is the oracle for the factor-of-two mistakes a wall model invites
//!    (`y_p = dx`, a half cell for a whole, a sign).
//! 4. **Force balance, which never uses `u⁺`.** A channel of height `H`
//!    between two walls, driven by a body force `G`, is steady when
//!    `2 τ_w = ρ G H`, i.e. `u_τ = √(G H / 2)`, whatever `κ`, `B` or the edge
//!    are. The step reports `u_τ`; the closed form says what it must be. The
//!    residual imbalance is bounded by the last step's `max|Δu|/dt`, derived
//!    from the momentum budget, not chosen.
//! 5. **One step from a uniform field.** With `G = 0` and `u = U` everywhere,
//!    a step changes only the wall-adjacent faces, by exactly the ghost
//!    (`None`) or by exactly `−dt u_τ(U)²/dx` (model), with `u_τ(U)` from the
//!    test's own `f64` bisection. The interior faces are untouched to the bit.
//! 6. **Refusals.** A wall model on zero viscosity is refused and leaves the
//!    solver untouched; `friction_velocity` refuses `y_p ≤ 0`, `ν ≤ 0`,
//!    `u_rel < 0`, and returns exactly zero for `u_rel = 0`.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// Expected values are f64 closed forms and a f64 reference bisection.
#![allow(clippy::disallowed_methods)]

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, PressureSolverError, StepError, StepOptions, WallModel,
};
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::turbulence::{
    friction_velocity, WallFunctionError, FRICTION_VELOCITY_BISECTIONS,
};

/// The module constants of the crate, written down here by hand.
const KAPPA: f64 = 0.41;
const B: f64 = 5.5;
/// `u⁺(100)` and `u⁺(1000)`, computed by hand from `ln(100)/0.41 + 5.5` and
/// `ln(1000)/0.41 + 5.5`.
const U_PLUS_100: f64 = 16.732;
const U_PLUS_1000: f64 = 22.348;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

/// The sublayer edge: the root of `y − ln(y)/κ − B` by bisection on
/// `[5, 20]` (the function is negative at 5 and positive at 20).
fn sublayer_edge() -> f64 {
    let f = |y: f64| y - y.ln() / KAPPA - B;
    let (mut lo, mut hi) = (5.0_f64, 20.0_f64);
    assert!(f(lo) < 0.0 && f(hi) > 0.0);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The universal profile, continuous at [`sublayer_edge`].
fn u_plus_ref(y_plus: f64) -> f64 {
    if y_plus < sublayer_edge() {
        y_plus
    } else {
        y_plus.ln() / KAPPA + B
    }
}

/// `u_τ` for a tangential speed `u_rel` at `y_p` in viscosity `nu`, by f64
/// bisection of `u_τ · u⁺(y_p u_τ / ν) = u_rel` — the reference inverse.
fn friction_velocity_ref(u_rel: f64, y_p: f64, nu: f64) -> f64 {
    if u_rel == 0.0 {
        return 0.0;
    }
    let g = |u_tau: f64| u_tau * u_plus_ref(y_p * u_tau / nu) - u_rel;
    let (mut lo, mut hi) = (0.0_f64, (u_rel.max((nu * u_rel / y_p).sqrt())) * 2.0);
    assert!(g(hi) > 0.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------
// 1 — the profile pinned by hand, then the inverse against it
// ---------------------------------------------------------------------------

#[test]
fn the_log_law_in_this_test_matches_the_two_hand_computed_points() {
    assert!(
        (u_plus_ref(100.0) - U_PLUS_100).abs() < 5e-4,
        "{}",
        u_plus_ref(100.0)
    );
    assert!(
        (u_plus_ref(1000.0) - U_PLUS_1000).abs() < 5e-4,
        "{}",
        u_plus_ref(1000.0)
    );
    // The difference of the two points is `ln 10 / κ`: κ alone.
    assert!(((U_PLUS_1000 - U_PLUS_100) - 10f64.ln() / KAPPA).abs() < 1e-3);
}

/// The crate's inverse recovers a prescribed `u_τ` from the speed the pinned
/// profile assigns to it, in both regimes.
#[test]
fn the_inverse_recovers_the_friction_velocity_the_pinned_profile_was_built_from() {
    let (y_p, nu) = (0.125_f64, 1e-4_f64);
    for (y_plus, regime) in [
        (5.0, "sublayer"),
        (100.0, "log"),
        (1000.0, "log"),
        (3.0e4, "log"),
    ] {
        let u_tau = y_plus * nu / y_p;
        let u_rel = u_tau * u_plus_ref(y_plus);
        let got = friction_velocity(fx(u_rel), fx(y_p), fx(nu))
            .expect("valid inputs")
            .to_f64();
        // The bisection halves a bracket of order `u_rel` sixty-four times;
        // the f64 side is good to ~1e-15 relative. Allow 1e-9 relative.
        let tol = 1e-9 * u_tau;
        assert!(
            (got - u_tau).abs() <= tol,
            "{regime} y+ = {y_plus}: expected u_tau {u_tau:.9e}, got {got:.9e}"
        );
        // And the reference inverse agrees with the crate's.
        let reference = friction_velocity_ref(u_rel, y_p, nu);
        assert!(
            (reference - u_tau).abs() <= tol,
            "reference inverse drifted: {reference}"
        );
    }
}

// ---------------------------------------------------------------------------
// 2 — the sublayer edge is the intersection, and the inverse is continuous
// ---------------------------------------------------------------------------

#[test]
fn the_sublayer_edge_is_the_intersection_and_the_inverse_is_continuous_there() {
    let edge = sublayer_edge();
    assert!((edge - 11.4453).abs() < 1e-4, "edge {edge}");
    let (y_p, nu) = (0.125_f64, 1e-4_f64);
    let u_tau_edge = edge * nu / y_p;
    let u_crit = u_tau_edge * edge; // u⁺ = y⁺ at the edge, both branches
    let below = friction_velocity(fx(u_crit * (1.0 - 1e-6)), fx(y_p), fx(nu))
        .expect("valid")
        .to_f64();
    let above = friction_velocity(fx(u_crit * (1.0 + 1e-6)), fx(y_p), fx(nu))
        .expect("valid")
        .to_f64();
    // Continuity: a 2e-6 relative change of the speed moves u_tau by a
    // comparable relative amount, not by the 3 % a 0.35 step in u⁺ would.
    let jump = (above - below).abs() / u_tau_edge;
    assert!(jump < 1e-5, "u_tau jumps by {jump:.3e} across the edge");
    assert!(
        ((below + above) * 0.5 - u_tau_edge).abs() < 1e-6 * u_tau_edge,
        "the edge maps to its own friction velocity"
    );
}

// ---------------------------------------------------------------------------
// the channel scene shared by 3, 4, 5
// ---------------------------------------------------------------------------

/// Two cells wide (the flow is uniform along the channel, so width only
/// costs time) and four high: `y_p = dx/2 = 1/8`.
const NX: usize = 2;
const NY: usize = 4;

/// A channel of height 1 between two resting walls, slip walls in `z`, open
/// in `x`, driven by the body force `g` (m/s²), density 1 so `μ = ν`.
fn channel(nu: f64, g: f64) -> CfdSolver {
    let mut s = CfdSolver::new(NX, NY, 1, Fix128::from_ratio(1, NY as i64));
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = fx(nu);
    s.gravity = Vec3Fix::new(fx(g), Fix128::ZERO, Fix128::ZERO);
    let rest = FaceBc::Wall {
        velocity: Vec3Fix::ZERO,
    };
    for i in 0..NX {
        s.grid.set_v_bc(i, 0, 0, rest);
        s.grid.set_v_bc(i, NY, 0, rest);
        for j in 0..NY {
            s.grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            s.grid.set_w_bc(i, j, 1, FaceBc::SlipWall);
        }
    }
    s
}

fn set_uniform_u(s: &mut CfdSolver, value: f64) {
    for u in s.grid.u.iter_mut() {
        *u = fx(value);
    }
}

fn u_face(s: &CfdSolver, i: usize, j: usize) -> f64 {
    s.grid.u(i, j, 0).to_f64()
}

fn max_face_change(a: &CfdSolver, b: &CfdSolver) -> f64 {
    a.grid
        .u
        .iter()
        .zip(&b.grid.u)
        .map(|(x, y)| (x.to_f64() - y.to_f64()).abs())
        .fold(0.0, f64::max)
}

fn options(wall: bool) -> StepOptions {
    let base = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 30 });
    if wall {
        base.with_wall_model(WallModel::log_law())
    } else {
        base
    }
}

/// Wall-adjacent face updates the model performs on this channel: the `u`
/// faces of the two wall rows (`nx + 1` each) and the `w` faces of both `z`
/// layers next to each wall (`nx` each).
const WALL_FACE_UPDATES: usize = 2 * (NX + 1) + 2 * NX * 2;

// ---------------------------------------------------------------------------
// 3 — below the edge, the model is the ghost
// ---------------------------------------------------------------------------

#[test]
fn below_the_sublayer_edge_the_wall_model_reproduces_the_no_slip_ghost() {
    // ν = 0.1, G = 0.01: u_τ = √(G H/2) = 0.0707, y⁺ = u_τ y_p / ν = 0.088.
    let (nu, g) = (0.1_f64, 0.01_f64);
    let dt = Fix128::from_ratio(1, 100); // ν dt / dx² = 0.016: stable
    let mut ghost = channel(nu, g);
    let mut model = channel(nu, g);
    let mut last = None;
    for _ in 0..4000 {
        ghost
            .step_with_options(dt, &options(false))
            .expect("ghost step");
        last = Some(
            model
                .step_with_options(dt, &options(true))
                .expect("model step"),
        );
    }
    let report = last.expect("stepped").wall.expect("the model reports");
    assert_eq!(report.faces, WALL_FACE_UPDATES);
    assert_eq!(
        report.resting_faces,
        2 * NX * 2,
        "the spanwise faces are at rest"
    );
    assert!(
        report.y_plus_max.to_f64() < 11.4453,
        "the scene must stay in the sublayer, got y+ up to {}",
        report.y_plus_max.to_f64()
    );
    let gap = max_face_change(&ghost, &model);
    // Both paths compute the same algebra: the bisection lands within
    // `hi / 2^64` of the exact sublayer root and each path rounds a handful
    // of Fix128 products per step; 4000 steps of that stay far below 1e-12.
    assert!(
        gap < 1e-12,
        "model and ghost differ by {gap:.3e} in the sublayer"
    );
    // And the scene is not trivial: the flow developed.
    assert!(u_face(&ghost, 2, NY / 2) > 1e-3);
}

// ---------------------------------------------------------------------------
// 4 — force balance: u_τ = √(G H / 2), independent of the law
// ---------------------------------------------------------------------------

#[test]
fn the_steady_channel_balances_the_body_force_against_the_modelled_wall_shear() {
    // ν = 1e-4, G = 1: u_τ = √(1/2), y⁺ = 0.7071 · 0.125 / 1e-4 = 884 (log).
    let (nu, g) = (1e-4_f64, 1.0_f64);
    // Explicit diffusion allows ν dt / dx² ≤ 1/6 (dt ≤ 260); the wall sink's own
    // rate u_τ²/(u_p dx) ≈ 0.3/s keeps dt at a few seconds for a monotone
    // approach. 2 s.
    let dt = Fix128::from_int(2);
    let mut s = channel(nu, g);
    let mut previous = channel(nu, g);
    let mut report = None;
    let mut settled_change = f64::INFINITY;
    for step in 0..20_000 {
        previous.grid.u.clone_from(&s.grid.u);
        report = Some(s.step_with_options(dt, &options(true)).expect("step"));
        settled_change = max_face_change(&previous, &s);
        if step > 100 && settled_change < 1e-11 {
            break;
        }
    }
    let wall = report.expect("stepped").wall.expect("the model reports");
    let h = 1.0_f64;
    let u_tau_target = (g * h / 2.0).sqrt();
    let u_tau = wall.u_tau_max.to_f64();
    assert_eq!(wall.faces, WALL_FACE_UPDATES);
    assert_eq!(wall.resting_faces, 2 * NX * 2);
    assert_eq!(
        wall.u_tau_min, wall.u_tau_max,
        "a uniform channel has one friction velocity at every moving wall face"
    );
    // Momentum budget: d/dt ∫u dy = G H − 2 u_τ², so the imbalance at the end
    // is bounded by H · max|Δu| / dt.
    let imbalance_bound = h * settled_change / dt.to_f64() + 1e-12;
    let imbalance = (2.0 * u_tau * u_tau - g * h).abs();
    assert!(
        imbalance <= imbalance_bound,
        "2 u_τ² = {:.9} against G H = {g}: imbalance {imbalance:.3e} > bound {imbalance_bound:.3e}",
        2.0 * u_tau * u_tau
    );
    assert!(
        (u_tau - u_tau_target).abs() < 1e-6,
        "u_τ = {u_tau:.9} against √(G H / 2) = {u_tau_target:.9}"
    );
    // The regime really is the log region.
    let y_plus = wall.y_plus_min.to_f64();
    assert!(y_plus > 11.4453, "y+ = {y_plus}");
    assert!((y_plus - u_tau * 0.125 / nu).abs() < 1e-6 * y_plus);
    // The wall-consistent k and ε, closed forms in u_τ (C_μ = 0.09, κ = 0.41).
    let k_target = u_tau * u_tau / 0.09_f64.sqrt();
    let eps_target = u_tau.powi(3) / (KAPPA * 0.125);
    assert!((wall.k_max.to_f64() - k_target).abs() < 1e-6 * k_target);
    assert!((wall.epsilon_max.to_f64() - eps_target).abs() < 1e-6 * eps_target);
    // And the wall-adjacent velocity sits on the profile: the model reads the
    // face **after the body force and before diffusion** (the step order), so
    // at steady state the speed it saw is `u_p + G dt`, and that is what
    // `u_τ u⁺(y⁺)` must reproduce. Comparing the end-of-step `u_p` alone is
    // off by exactly `G dt` — measured 2.000 at dt = 2 — which is the
    // splitting, not the law.
    let u_p = u_face(&s, 2, 0) + g * dt.to_f64();
    let u_p_target = u_tau * u_plus_ref(y_plus);
    assert!(
        (u_p - u_p_target).abs() < 1e-5 * u_p_target,
        "u_p + G dt = {u_p:.6} against u_τ u⁺ = {u_p_target:.6}"
    );
    eprintln!(
        "  steady channel: u_tau {u_tau:.9} (target {u_tau_target:.9}), y+ {y_plus:.1}, \
         u_p {u_p:.4}, last change {settled_change:.2e}"
    );
}

// ---------------------------------------------------------------------------
// 5 — one step from a uniform field: the sink in closed form, ghost when None
// ---------------------------------------------------------------------------

#[test]
fn one_step_from_a_uniform_field_removes_exactly_the_modelled_wall_shear() {
    let (nu, big_u) = (1e-4_f64, 10.0_f64);
    let dt = Fix128::from_ratio(1, 1000);
    let dx = 1.0 / NY as f64;
    let y_p = dx / 2.0;

    // The model.
    let mut s = channel(nu, 0.0);
    set_uniform_u(&mut s, big_u);
    let report = s.step_with_options(dt, &options(true)).expect("step");
    let wall = report.wall.expect("reports");
    let u_tau = friction_velocity_ref(big_u, y_p, nu);
    assert!(
        wall.y_plus_min.to_f64() > 11.4453,
        "the scene is in the log region"
    );
    assert!((wall.u_tau_max.to_f64() - u_tau).abs() < 1e-9 * u_tau);
    assert_eq!(wall.u_tau_min, wall.u_tau_max);
    assert_eq!(wall.resting_faces, 2 * NX * 2);
    let expected_drop = dt.to_f64() * u_tau * u_tau / dx;
    for i in 0..=NX {
        for j in [0, NY - 1] {
            let got = big_u - u_face(&s, i, j);
            assert!(
                (got - expected_drop).abs() < 1e-9 * expected_drop,
                "face ({i}, {j}) dropped by {got:.9e}, expected dt u_τ²/dx = {expected_drop:.9e}"
            );
        }
        for j in 1..NY - 1 {
            assert_eq!(
                s.grid.u(i, j, 0),
                fx(big_u),
                "interior face ({i}, {j}) must be untouched to the bit"
            );
        }
    }

    // No model: the ghost's closed form, and a different number.
    let mut g = channel(nu, 0.0);
    set_uniform_u(&mut g, big_u);
    let report = g.step_with_options(dt, &options(false)).expect("step");
    assert!(report.wall.is_none(), "no model, no wall report");
    let ghost_drop = 2.0 * nu * dt.to_f64() * big_u / (dx * dx);
    let got = big_u - u_face(&g, 2, 0);
    // The coefficient ν dt / dx² ≈ 1.6e-6 is formed by Fix128 products that
    // each round to 2⁻⁶⁴ absolute, i.e. ~1e-13 relative; 1e-9 leaves four
    // decades and is still six below the model / ghost separation asserted
    // next.
    assert!(
        (got - ghost_drop).abs() < 1e-9 * ghost_drop,
        "ghost path: dropped by {got:.9e}, expected 2 ν dt U / dx² = {ghost_drop:.9e}"
    );
    assert!(
        (expected_drop - ghost_drop).abs() > 10.0 * ghost_drop,
        "in the log region the modelled shear must differ from the laminar ghost"
    );
}

// ---------------------------------------------------------------------------
// 6 — refusals
// ---------------------------------------------------------------------------

#[test]
fn a_wall_model_on_zero_viscosity_is_refused_and_leaves_the_solver_untouched() {
    let mut s = channel(0.0, 1.0);
    set_uniform_u(&mut s, 1.0);
    let before = s.grid.u.clone();
    assert_eq!(
        s.step_with_options(Fix128::from_ratio(1, 100), &options(true)),
        Err(StepError::WallModelNeedsViscosity)
    );
    assert_eq!(s.grid.u, before);
    assert_eq!(s.step_count, 0);
    // The pressure refusals come through unchanged.
    assert_eq!(
        s.step_with_options(Fix128::ZERO, &options(true)),
        Err(StepError::Pressure(PressureSolverError::ZeroTimeStep))
    );
    assert!(!StepError::WallModelNeedsViscosity.to_string().is_empty());
}

#[test]
fn friction_velocity_refuses_degenerate_inputs_and_is_exactly_zero_at_rest() {
    let (y, nu) = (fx(0.0625), fx(1e-4));
    assert_eq!(
        friction_velocity(Fix128::ONE, Fix128::ZERO, nu),
        Err(WallFunctionError::NonPositiveWallDistance)
    );
    assert_eq!(
        friction_velocity(Fix128::ONE, -y, nu),
        Err(WallFunctionError::NonPositiveWallDistance)
    );
    assert_eq!(
        friction_velocity(Fix128::ONE, y, Fix128::ZERO),
        Err(WallFunctionError::NonPositiveViscosity)
    );
    assert_eq!(
        friction_velocity(-Fix128::ONE, y, nu),
        Err(WallFunctionError::NegativeSpeed)
    );
    assert_eq!(friction_velocity(Fix128::ZERO, y, nu), Ok(Fix128::ZERO));
    assert_eq!(FRICTION_VELOCITY_BISECTIONS, 64);
    for e in [
        WallFunctionError::NonPositiveWallDistance,
        WallFunctionError::NonPositiveViscosity,
        WallFunctionError::NegativeSpeed,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
