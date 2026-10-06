//! Independent oracles for `euler_fv` (1-D compressible Euler, finite volume).
//!
//! The existing oracles (`analytic_euler_fv.rs`) use `γ = 7/5`, Toro's Tests
//! 1–5 and Lax's problem, an `f64` port of Toro's Newton program
//! (`GUESSP`/`PREFUN`/`STARPU`/`SAMPLE`), Sod L1 under refinement, a sine
//! contact wave for the order of accuracy, a rough periodic profile for
//! conservation and a wall hit at `u = 0.8`.
//!
//! Here the inputs and the derivations differ:
//!
//! - **Riemann closed forms** for `γ = 5/3`: a symmetric expansion has
//!   `p* = p (1 − (γ−1)u₀/(2a))^{2γ/(γ−1)}` and a symmetric collision has
//!   `p*` from a quadratic; general states by **bisection** on the
//!   monotone pressure function (no Newton, no initial-guess logic), and a
//!   sampler written from the wave relations;
//! - the **sonic-rarefaction** shock tube `(1, 3/4, 1) | (1/8, 0, 1/10)` run by
//!   the finite-volume solver against that reference;
//! - **near-vacuum** expansion `(1, ∓5/2, 1/2)` with `γ = 5/3`, `p* ≈ 2.5e-6`,
//!   positivity of every cell, and the vacuum error just past the threshold;
//! - **balance laws with boundaries**: between reflecting walls, momentum
//!   changes per step by exactly the two wall pressure fluxes; with an open
//!   (transmissive) inflow end, mass changes by exactly the inflow flux;
//! - **smooth nonlinear acoustics** with `γ = 3`, where both Riemann
//!   invariants `u ± a` obey Burgers' equation and the exact solution is
//!   found by characteristics (Newton per point in the test), for the L1
//!   order of first-order Godunov and MUSCL + SSP-RK2.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 transcendentals compute reference values only.
#![allow(clippy::disallowed_methods)]

use alice_physics::euler_fv::{
    exact_riemann, numerical_flux, Boundary, EulerConfig, EulerError, EulerFv1d, Limiter,
    Primitive, Reconstruction, RiemannSolver, TimeIntegrator,
};
use alice_physics::math::Fix128;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn prim(rho: f64, u: f64, p: f64) -> Primitive {
    Primitive {
        rho: fx(rho),
        u: fx(u),
        p: fx(p),
    }
}

// ----------------------------------------------------------------------------
// f64 reference: pressure function, bisection, sampling
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct St {
    rho: f64,
    u: f64,
    p: f64,
}

fn st(rho: f64, u: f64, p: f64) -> St {
    St { rho, u, p }
}

fn snd(g: f64, s: St) -> f64 {
    (g * s.p / s.rho).sqrt()
}

/// Velocity jump across the `K` wave when the star pressure is `p`
/// (shock: Rankine–Hugoniot; rarefaction: isentrope + Riemann invariant).
fn fk(g: f64, p: f64, s: St) -> f64 {
    if p > s.p {
        let a = 2.0 / ((g + 1.0) * s.rho);
        let b = (g - 1.0) / (g + 1.0) * s.p;
        (p - s.p) * (a / (p + b)).sqrt()
    } else {
        2.0 * snd(g, s) / (g - 1.0) * ((p / s.p).powf((g - 1.0) / (2.0 * g)) - 1.0)
    }
}

/// Star `(p*, u*)` by bisection on `f(p) = f_L + f_R + Δu` (increasing in `p`).
fn star_bisect(g: f64, l: St, r: St) -> (f64, f64) {
    let f = |p: f64| fk(g, p, l) + fk(g, p, r) + (r.u - l.u);
    let (mut lo, mut hi) = (1e-30, 1.0);
    while f(hi) < 0.0 {
        hi *= 2.0;
    }
    for _ in 0..400 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-17 * hi {
            break;
        }
    }
    let p = 0.5 * (lo + hi);
    (p, 0.5 * (l.u + r.u) + 0.5 * (fk(g, p, r) - fk(g, p, l)))
}

/// The self-similar solution at `ξ = x/t`, written for the left side; the
/// right side is obtained by reflection (`x → −x`, `u → −u`).
fn sample_ref(g: f64, l: St, r: St, ps: f64, us: f64, xi: f64) -> St {
    if xi > us {
        let m = |s: St| st(s.rho, -s.u, s.p);
        let w = sample_left(g, m(r), ps, -us, -xi);
        return m(w);
    }
    sample_left(g, l, ps, us, xi)
}

fn sample_left(g: f64, l: St, ps: f64, us: f64, xi: f64) -> St {
    let al = snd(g, l);
    if ps > l.p {
        // shock: speed from mass conservation across it
        let rho_s =
            l.rho * (ps / l.p + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ps / l.p + 1.0);
        let speed = (rho_s * us - l.rho * l.u) / (rho_s - l.rho);
        if xi < speed {
            l
        } else {
            st(rho_s, us, ps)
        }
    } else {
        let rho_s = l.rho * (ps / l.p).powf(1.0 / g);
        let a_s = al * (ps / l.p).powf((g - 1.0) / (2.0 * g));
        let (head, tail) = (l.u - al, us - a_s);
        if xi < head {
            l
        } else if xi > tail {
            st(rho_s, us, ps)
        } else {
            // inside the fan: u − ξ = a (C⁺ characteristic through the point),
            // u + 2a/(γ−1) = u_L + 2a_L/(γ−1) (Riemann invariant).
            let a = (2.0 / (g + 1.0)) * (al + 0.5 * (g - 1.0) * (l.u - xi));
            let u = xi + a;
            let rho = l.rho * (a / al).powf(2.0 / (g - 1.0));
            let p = l.p * (a / al).powf(2.0 * g / (g - 1.0));
            st(rho, u, p)
        }
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1.0)
}

// ----------------------------------------------------------------------------
// Riemann solver against closed forms and bisection
// ----------------------------------------------------------------------------

const G53: f64 = 5.0 / 3.0;

fn g53() -> Fix128 {
    Fix128::from_ratio(5, 3)
}

/// Symmetric expansion `(ρ, −u₀, p) | (ρ, u₀, p)`: two equal rarefactions,
/// `u* = 0`, `2a/(γ−1) ((p*/p)^{(γ−1)/2γ} − 1) = −u₀`, hence
/// `p* = p (1 − (γ−1)u₀/(2a))^{2γ/(γ−1)}`. `γ = 5/3`, `ρ = 3/2`, `p = 2`,
/// `u₀ = 1`.
#[test]
fn symmetric_expansion_matches_closed_form_star_pressure() {
    let (rho, u0, p) = (1.5, 1.0, 2.0);
    let a = (G53 * p / rho).sqrt();
    let want = p * (1.0 - (G53 - 1.0) * u0 / (2.0 * a)).powf(2.0 * G53 / (G53 - 1.0));
    let s = exact_riemann(g53(), &prim(rho, -u0, p), &prim(rho, u0, p)).expect("star");
    assert!(
        rel(s.p.to_f64(), want) < 1e-12,
        "p* {} vs {want}",
        s.p.to_f64()
    );
    assert!(s.u.to_f64().abs() < 1e-15, "u* {}", s.u.to_f64());
}

/// Symmetric collision `(ρ, u₀, p) | (ρ, −u₀, p)`: two equal shocks, `u* = 0`,
/// `(p* − p)² A = u₀² (p* + B)` with `A = 2/((γ+1)ρ)`, `B = (γ−1)p/(γ+1)`, the
/// larger root. `γ = 5/3`, `ρ = 1/2`, `p = 1/4`, `u₀ = 3/2`.
#[test]
fn symmetric_collision_matches_quadratic_star_pressure() {
    let (rho, u0, p) = (0.5, 1.5, 0.25);
    let aa = 2.0 / ((G53 + 1.0) * rho);
    let bb = (G53 - 1.0) / (G53 + 1.0) * p;
    // A X² − u₀² X − u₀² (p + B) = 0 with X = p* − p.
    let (qa, qb, qc) = (aa, -u0 * u0, -u0 * u0 * (p + bb));
    let x = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
    let want = p + x;
    let s = exact_riemann(g53(), &prim(rho, u0, p), &prim(rho, -u0, p)).expect("star");
    assert!(
        rel(s.p.to_f64(), want) < 1e-12,
        "p* {} vs {want}",
        s.p.to_f64()
    );
    assert!(s.u.to_f64().abs() < 1e-15);
}

/// Asymmetric states (one shock + one rarefaction, both orders, `γ = 5/3` and
/// `γ = 7/5`) against bisection, and the sampled profile at 41 speeds against
/// [`sample_ref`].
#[test]
fn general_states_match_bisection_and_sampled_profile() {
    let cases = [
        (G53, st(2.0, 0.3, 3.0), st(0.7, -0.4, 0.2)),
        (G53, st(0.25, 1.2, 0.05), st(1.0, 0.0, 1.0)),
        (1.4, st(1.0, 0.75, 1.0), st(0.125, 0.0, 0.1)),
        (1.4, st(3.0, -1.0, 4.0), st(3.0, -1.0, 0.5)),
    ];
    for (g, l, r) in cases {
        let gf = if g == G53 {
            g53()
        } else {
            Fix128::from_ratio(7, 5)
        };
        let (ps, us) = star_bisect(g, l, r);
        let (lf, rf) = (prim(l.rho, l.u, l.p), prim(r.rho, r.u, r.p));
        let s = exact_riemann(gf, &lf, &rf).expect("star");
        assert!(
            rel(s.p.to_f64(), ps) < 1e-10,
            "{l:?}|{r:?}: p* {} vs {ps}",
            s.p.to_f64()
        );
        assert!(
            rel(s.u.to_f64(), us) < 1e-10,
            "{l:?}|{r:?}: u* {} vs {us}",
            s.u.to_f64()
        );
        for k in 0..=40 {
            let xi = -2.5 + 5.0 * f64::from(k) / 40.0;
            let got = s.sample(gf, &lf, &rf, fx(xi));
            let want = sample_ref(g, l, r, ps, us, xi);
            for (name, a, b) in [
                ("rho", got.rho.to_f64(), want.rho),
                ("u", got.u.to_f64(), want.u),
                ("p", got.p.to_f64(), want.p),
            ] {
                assert!(rel(a, b) < 1e-9, "{l:?}|{r:?} ξ={xi}: {name} {a} vs {b}");
            }
        }
    }
}

// ----------------------------------------------------------------------------
// finite volume: sonic rarefaction shock tube
// ----------------------------------------------------------------------------

fn tube(n: usize, c: EulerConfig, l: St, r: St) -> EulerFv1d {
    let init: Vec<Primitive> = (0..n)
        .map(|i| {
            let s = if (i as f64 + 0.5) / (n as f64) < 0.3 {
                l
            } else {
                r
            };
            prim(s.rho, s.u, s.p)
        })
        .collect();
    EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid")
}

fn l1_vs_exact(sim: &EulerFv1d, g: f64, l: St, r: St, x0: f64, t: f64) -> f64 {
    let (ps, us) = star_bisect(g, l, r);
    let prims = sim.primitives();
    let n = prims.len() as f64;
    prims
        .iter()
        .enumerate()
        .map(|(i, w)| {
            // cell average of the exact density by 8-point midpoint rule
            let mut avg = 0.0;
            for q in 0..8 {
                let x = (i as f64 + (q as f64 + 0.5) / 8.0) / n;
                avg += sample_ref(g, l, r, ps, us, (x - x0) / t).rho / 8.0;
            }
            (w.rho.to_f64() - avg).abs() / n
        })
        .sum()
}

/// Toro's sonic-rarefaction tube `(1, 3/4, 1) | (1/8, 0, 1/10)`, `γ = 7/5`,
/// diaphragm at `x = 0.3`, `t = 0.2`. The left rarefaction straddles `x/t = 0`,
/// which is the case where a Godunov flux must sample inside the fan. L1 of
/// the density against the bisection reference falls under refinement
/// (contact-dominated first order, see the assertion), and
/// MUSCL + SSP-RK2 (HLLC, van Leer) is more accurate than Godunov at the
/// same N.
#[test]
fn sonic_rarefaction_tube_converges_to_exact_solution() {
    let (l, r) = (st(1.0, 0.75, 1.0), st(0.125, 0.0, 0.1));
    let g = 1.4;
    let gf = Fix128::from_ratio(7, 5);
    let t = 0.2;
    let mut errs = Vec::new();
    for n in [50usize, 100, 200] {
        let mut sim = tube(n, EulerConfig::godunov(gf), l, r);
        sim.advance_to(fx(t)).expect("stable");
        errs.push(l1_vs_exact(&sim, g, l, r, 0.3, t));
    }
    assert!(errs[2] < 0.012, "L1 at N=200: {errs:?}");
    // Measured 0.0160 / 0.0119 / 0.0081 (ratios 1.34, 1.48): still
    // pre-asymptotic at N = 50, so each halving is only required to reduce
    // the error by 1.25 and the two halvings together by 1.8 (rate ≈ 1/2,
    // the contact-dominated first-order rate).
    assert!(
        errs[0] / errs[1] > 1.25 && errs[1] / errs[2] > 1.25,
        "{errs:?}"
    );
    assert!(errs[0] / errs[2] > 1.8, "{errs:?}");
    let mut hi = tube(
        200,
        EulerConfig::muscl(gf, RiemannSolver::Hllc, Limiter::VanLeer),
        l,
        r,
    );
    hi.advance_to(fx(t)).expect("stable");
    let e2 = l1_vs_exact(&hi, g, l, r, 0.3, t);
    assert!(e2 < 0.75 * errs[2], "MUSCL {e2} vs Godunov {}", errs[2]);
    // No expansion shock: inside the fan the density is monotone (decreasing
    // in x) to within round-off, for the first-order exact-flux run.
    let mut sim = tube(200, EulerConfig::godunov(gf), l, r);
    sim.advance_to(fx(t)).expect("stable");
    let rho: Vec<f64> = sim.primitives().iter().map(|w| w.rho.to_f64()).collect();
    // fan for t = 0.2: x ∈ 0.3 + 0.2·[u_L − a_L, u* − a*]
    let (ps, us) = star_bisect(g, l, r);
    let a_s = snd(g, l) * (ps / l.p).powf((g - 1.0) / (2.0 * g));
    let (x1, x2) = (0.3 + t * (l.u - snd(g, l)), 0.3 + t * (us - a_s));
    let (i1, i2) = ((x1 * 200.0) as usize + 2, (x2 * 200.0) as usize - 2);
    for i in i1..i2 {
        assert!(rho[i + 1] <= rho[i] + 1e-12, "fan not monotone at {i}");
    }
}

// ----------------------------------------------------------------------------
// near vacuum: positivity, closed form, vacuum threshold
// ----------------------------------------------------------------------------

/// `(1, −5/2, 1/2) | (1, 5/2, 1/2)`, `γ = 5/3`: `2(a_L + a_R)/(γ−1) ≈ 5.477`
/// exceeds `Δu = 5`, so no vacuum, but the closed-form star pressure is
/// `p* = ½ (1 − (2/3)·(5/2)/(2a))^5 ≈ 2.5e-6`. The exact solver reproduces it;
/// Godunov (exact and HLLC) runs to `t = 0.1` with every cell positive. At
/// `u₀ = 3` (`Δu = 6`) the documented `VacuumGenerated` is returned.
#[test]
fn near_vacuum_expansion_stays_positive_and_vacuum_is_refused() {
    let (rho, u0, p) = (1.0, 2.5, 0.5);
    let a = (G53 * p / rho).sqrt();
    let want = p * (1.0 - (G53 - 1.0) * u0 / (2.0 * a)).powf(2.0 * G53 / (G53 - 1.0));
    assert!(want > 1e-7 && want < 1e-5);
    let s = exact_riemann(g53(), &prim(rho, -u0, p), &prim(rho, u0, p)).expect("no vacuum");
    assert!(
        (s.p.to_f64() - want).abs() < 1e-9 * want,
        "p* {} vs {want}",
        s.p.to_f64()
    );
    for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
        let mut c = EulerConfig::godunov(g53());
        c.solver = solver;
        let n = 80;
        let init: Vec<Primitive> = (0..n)
            .map(|i| prim(rho, if i < n / 2 { -u0 } else { u0 }, p))
            .collect();
        let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
        sim.advance_to(fx(0.1)).expect("stays positive");
        let prims = sim.primitives();
        let min_rho = prims
            .iter()
            .map(|w| w.rho.to_f64())
            .fold(f64::MAX, f64::min);
        let min_p = prims.iter().map(|w| w.p.to_f64()).fold(f64::MAX, f64::min);
        assert!(min_rho > 0.0 && min_p > 0.0, "{solver:?}");
        // The centre cells approach the near-vacuum star state.
        assert!(min_rho < 0.2, "{solver:?}: centre density {min_rho}");
    }
    assert_eq!(
        exact_riemann(g53(), &prim(rho, -3.0, p), &prim(rho, 3.0, p)).unwrap_err(),
        EulerError::VacuumGenerated
    );
}

// ----------------------------------------------------------------------------
// balance laws with boundaries
// ----------------------------------------------------------------------------

fn lumpy(n: usize) -> Vec<Primitive> {
    (0..n)
        .map(|i| {
            let k = (i * 37 % 13) as i64;
            Primitive {
                rho: Fix128::from_ratio(8 + k, 8),
                u: Fix128::from_ratio(k - 6, 16),
                p: Fix128::from_ratio(12 + (k * 5 % 7), 10),
            }
        })
        .collect()
}

fn ulps(a: Fix128, b: Fix128) -> u128 {
    let d = a - b;
    ((i128::from(d.hi) << 64) | i128::from(d.lo)).unsigned_abs()
}

/// Reflecting walls at both ends, first-order Godunov with the exact flux,
/// `γ = 5/3`: the per-step change of `Σ ρu` is exactly
/// `−λ (F_R − F_L)` of the two wall interfaces, where the wall flux is the
/// Riemann flux between the boundary cell and its mirror image (computed
/// here through `numerical_flux`). Mass and energy do not change.
#[test]
fn wall_momentum_balance_equals_wall_pressure_flux() {
    let mut c = EulerConfig::godunov(g53());
    c.left = Boundary::ReflectiveWall;
    c.right = Boundary::ReflectiveWall;
    let n = 24;
    let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &lumpy(n)).expect("valid");
    for _ in 0..40 {
        let before = sim.totals();
        let w = sim.primitives();
        let mirror = |p: Primitive| Primitive { u: -p.u, ..p };
        let fl = numerical_flux(RiemannSolver::Exact, g53(), &mirror(w[0]), &w[0]).expect("f");
        let fr =
            numerical_flux(RiemannSolver::Exact, g53(), &w[n - 1], &mirror(w[n - 1])).expect("f");
        assert_eq!(fl.rho, Fix128::ZERO);
        assert_eq!(fr.energy, Fix128::ZERO);
        let dt = sim.step().expect("step");
        let lambda = dt / sim.dx();
        let after = sim.totals();
        assert_eq!(after.rho, before.rho);
        assert_eq!(after.energy, before.energy);
        let predicted = before.mom - (fr.mom * lambda - fl.mom * lambda);
        // Per-interface scaling in the solver uses a sign-symmetric product;
        // `*` here floors, so allow a few raw units.
        assert!(
            ulps(after.mom, predicted) <= 4,
            "Σρu {} vs {}",
            after.mom.to_f64(),
            predicted.to_f64()
        );
    }
}

/// Transmissive left end (ghost = first cell), reflecting right wall: the
/// mass in the box changes per step by exactly `λ (F_ghost − F_wall)` with
/// `F_wall.ρ = 0` and the ghost interface flux equal to the physical flux of
/// the first cell (a Riemann problem between equal states).
#[test]
fn open_end_mass_balance_equals_inflow_flux() {
    let mut c = EulerConfig::godunov(Fix128::from_ratio(7, 5));
    c.solver = RiemannSolver::Hllc;
    c.right = Boundary::ReflectiveWall;
    let n = 20;
    let init: Vec<Primitive> = (0..n)
        .map(|i| {
            Primitive {
                // inflow from the left into a closed box
                rho: Fix128::from_ratio(10 + (i as i64 % 3), 10),
                u: Fix128::from_ratio(1, 2),
                p: Fix128::ONE,
            }
        })
        .collect();
    let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
    let gamma = Fix128::from_ratio(7, 5);
    let mut gained = Fix128::ZERO;
    let start = sim.totals().rho;
    for _ in 0..30 {
        let before = sim.totals().rho;
        let w0 = sim.primitives()[0];
        let f0 = numerical_flux(RiemannSolver::Hllc, gamma, &w0, &w0).expect("f");
        let dt = sim.step().expect("step");
        let lambda = dt / sim.dx();
        let d = sim.totals().rho - before;
        assert!(
            ulps(d, f0.rho * lambda) <= 4,
            "Δm {} vs {}",
            d.to_f64(),
            (f0.rho * lambda).to_f64()
        );
        gained = gained + d;
    }
    assert!(gained > Fix128::ZERO);
    assert_eq!(sim.totals().rho, start + gained);
}

// ----------------------------------------------------------------------------
// smooth nonlinear acoustics, γ = 3
// ----------------------------------------------------------------------------

/// With `γ = 3` and `p = ρ³/3` (so `a = ρ`), the Riemann invariants
/// `R± = u ± a` travel at their own value: `R_t + R R_x = 0`. Initial
/// `ρ = 1 + 0.2 sin 2πx`, `u = 0` gives `R± = ±ρ₀`, and the solution is
/// `R(x, t) = R₀(ξ)` with `ξ + t R₀(ξ) = x` (characteristics, solved by Newton
/// here). Gradient catastrophe at `t = 1/(0.4π) ≈ 0.80`; measured at `t = 0.2`.
fn acoustic_exact(x: f64, t: f64) -> (f64, f64) {
    let r0 = |s: f64| 1.0 + 0.2 * (2.0 * std::f64::consts::PI * s).sin();
    let d0 = |s: f64| 0.4 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * s).cos();
    let solve = |sign: f64| {
        // R = sign·r0(ξ), ξ + t·sign·r0(ξ) = x
        let mut xi = x;
        for _ in 0..60 {
            let f = xi + t * sign * r0(xi) - x;
            let fp = 1.0 + t * sign * d0(xi);
            xi -= f / fp;
        }
        sign * r0(xi)
    };
    let (rp, rm) = (solve(1.0), solve(-1.0));
    ((rp - rm) / 2.0, (rp + rm) / 2.0) // (ρ = a, u)
}

fn acoustic_l1(n: usize, c: EulerConfig, t: f64) -> f64 {
    let init: Vec<Primitive> = (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) / n as f64;
            let rho = 1.0 + 0.2 * (2.0 * std::f64::consts::PI * x).sin();
            Primitive {
                rho: fx(rho),
                u: Fix128::ZERO,
                p: fx(rho * rho * rho / 3.0),
            }
        })
        .collect();
    let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
    sim.advance_to(fx(t)).expect("smooth");
    let prims = sim.primitives();
    let mut e = 0.0;
    for (i, w) in prims.iter().enumerate() {
        let x = (i as f64 + 0.5) / n as f64;
        let (rho, u) = acoustic_exact(x, t);
        // Entropy stays constant in a smooth flow: p = ρ³/3 in every cell.
        let p = w.p.to_f64();
        let rr = w.rho.to_f64();
        assert!(
            (p - rr * rr * rr / 3.0).abs() < 0.05 * p,
            "isentrope broken"
        );
        e += ((rr - rho).abs() + (w.u.to_f64() - u).abs()) / n as f64;
    }
    e
}

/// L1 (density + velocity) at `N = 64, 128, 256`: first-order Godunov (HLLC)
/// halves the error per refinement (ratio in `[1.7, 2.3]`), MUSCL (van Leer) +
/// SSP-RK2 divides it by about 4 (ratio `≥ 3.2`), and is the more accurate.
#[test]
fn smooth_acoustic_wave_converges_at_documented_order() {
    let g = Fix128::from_int(3);
    let mut first = EulerConfig::godunov(g);
    first.solver = RiemannSolver::Hllc;
    first.left = Boundary::Periodic;
    first.right = Boundary::Periodic;
    first.cfl = Fix128::from_ratio(1, 2);
    let mut second = EulerConfig::muscl(g, RiemannSolver::Hllc, Limiter::VanLeer);
    second.left = Boundary::Periodic;
    second.right = Boundary::Periodic;
    assert_eq!(second.time, TimeIntegrator::SspRk2);
    assert_eq!(
        second.reconstruction,
        Reconstruction::Muscl(Limiter::VanLeer)
    );
    let t = 0.2;
    let e1: Vec<f64> = [64, 128, 256]
        .iter()
        .map(|&n| acoustic_l1(n, first, t))
        .collect();
    let e2: Vec<f64> = [64, 128, 256]
        .iter()
        .map(|&n| acoustic_l1(n, second, t))
        .collect();
    for k in 0..2 {
        let r1 = e1[k] / e1[k + 1];
        let r2 = e2[k] / e2[k + 1];
        assert!((1.7..2.3).contains(&r1), "first order ratio {r1} ({e1:?})");
        assert!(r2 >= 3.2, "second order ratio {r2} ({e2:?})");
        assert!(e2[k] < e1[k]);
    }
}
