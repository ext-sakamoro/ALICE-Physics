//! Oracles for `euler_fv` (1D compressible Euler equations, finite volume).
//!
//! # Where the expected values come from
//!
//! - Exact Riemann solution: an independent `f64` implementation of the
//!   procedure in E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
//!   Dynamics*, 3rd ed. (Springer, 2009), chapter 4 — pressure function and
//!   star velocity (§4.2, Proposition 4.1), Newton iteration with the adaptive
//!   PVRS / TRRS / TSRS initial guess (§4.3), sampling of the complete solution
//!   (§4.4–4.5), vacuum-generation condition (§4.6), following the program of
//!   §4.9 (subroutines `GUESSP`, `PREFUN`, `STARPU`, `SAMPLE`). It is written
//!   here from those formulas and does not call the crate.
//! - Initial data of the Riemann tests: Toro 3rd ed. §4.3.3, Table 4.1
//!   (Tests 1–5, γ = 1.4); the star values printed in Table 4.2 (5 significant
//!   figures) are a second, fully external check of the `f64` reference.
//!   Lax's problem: P. D. Lax, *Comm. Pure Appl. Math.* 7 (1954) 159–193,
//!   `(ρ, u, p) = (0.445, 0.698, 3.528) | (0.5, 0, 0.571)`.
//! - Normal shock: `compressible::normal_shock_jump` (Anderson, *Modern
//!   Compressible Flow*, eqs. 3.51 / 3.57), the crate's existing closed form.
//! - Smooth advection: `ρ(x, t) = ρ₀(x − u t)` with `u`, `p` constant is an
//!   exact solution of the Euler equations; cell averages of `1 + ε sin 2πx`
//!   are integrated in closed form.
//!
//! # Tolerances
//!
//! - Star state `p*`, `u*`: `1e-10` (relative to `max(1, |value|)`). The
//!   module evaluates `x^e` as `exp(e ln x)` with series accurate to about
//!   `1e-17`, and Newton stops at a relative step of `2⁻⁴⁶`; the `f64`
//!   reference carries about `1e-15`.
//! - Sampled profiles: `1e-9` (the rarefaction fan adds two more powers).
//! - Conservation, mirror symmetry, uniform states: exact (`Fix128` addition
//!   is an integer addition; the module multiplies with a sign-symmetric
//!   product — see the module doc).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// f64 transcendentals compute reference values, not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::compressible::{normal_shock_jump, IdealGas};
use alice_physics::euler_fv::{
    exact_riemann, numerical_flux, physical_flux, Boundary, Conserved, EulerConfig, EulerError,
    EulerFv1d, Limiter, Primitive, Reconstruction, RiemannSolver, TimeIntegrator,
};
use alice_physics::math::Fix128;

const GAMMA: f64 = 1.4;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}

fn gamma() -> Fix128 {
    Fix128::from_ratio(7, 5)
}

fn prim(rho: f64, u: f64, p: f64) -> Primitive {
    Primitive {
        rho: fx(rho),
        u: fx(u),
        p: fx(p),
    }
}

// ===========================================================================
// Independent f64 exact Riemann solver (Toro 3rd ed. ch. 4, §4.9 program)
// ===========================================================================

#[derive(Clone, Copy, Debug)]
struct W {
    rho: f64,
    u: f64,
    p: f64,
}

struct G {
    g: f64,
    g1: f64,
    g2: f64,
    g3: f64,
    g4: f64,
    g5: f64,
    g6: f64,
    g7: f64,
}

fn gas(g: f64) -> G {
    G {
        g,
        g1: (g - 1.0) / (2.0 * g),
        g2: (g + 1.0) / (2.0 * g),
        g3: 2.0 * g / (g - 1.0),
        g4: 2.0 / (g - 1.0),
        g5: 2.0 / (g + 1.0),
        g6: (g - 1.0) / (g + 1.0),
        g7: (g - 1.0) / 2.0,
    }
}

fn sound(c: &G, w: W) -> f64 {
    (c.g * w.p / w.rho).sqrt()
}

/// `PREFUN`: f_K(p) and its derivative.
fn prefun(c: &G, p: f64, w: W) -> (f64, f64) {
    let a = sound(c, w);
    if p <= w.p {
        let pr = p / w.p;
        (
            c.g4 * a * (pr.powf(c.g1) - 1.0),
            pr.powf(-c.g2) / (w.rho * a),
        )
    } else {
        let ak = c.g5 / w.rho;
        let bk = c.g6 * w.p;
        let q = (ak / (bk + p)).sqrt();
        ((p - w.p) * q, (1.0 - 0.5 * (p - w.p) / (bk + p)) * q)
    }
}

/// `GUESSP`: adaptive initial guess.
fn guessp(c: &G, l: W, r: W) -> f64 {
    let (al, ar) = (sound(c, l), sound(c, r));
    let cup = 0.25 * (l.rho + r.rho) * (al + ar);
    let ppv = (0.5 * (l.p + r.p) + 0.5 * (l.u - r.u) * cup).max(0.0);
    let pmin = l.p.min(r.p);
    let pmax = l.p.max(r.p);
    if pmax / pmin <= 2.0 && pmin <= ppv && ppv <= pmax {
        ppv
    } else if ppv < pmin {
        let pq = (l.p / r.p).powf(c.g1);
        let um = (pq * l.u / al + r.u / ar + c.g4 * (pq - 1.0)) / (pq / al + 1.0 / ar);
        let ptl = 1.0 + c.g7 * (l.u - um) / al;
        let ptr = 1.0 + c.g7 * (um - r.u) / ar;
        0.5 * (l.p * ptl.powf(c.g3) + r.p * ptr.powf(c.g3))
    } else {
        let gel = ((c.g5 / l.rho) / (c.g6 * l.p + ppv)).sqrt();
        let ger = ((c.g5 / r.rho) / (c.g6 * r.p + ppv)).sqrt();
        (gel * l.p + ger * r.p - (r.u - l.u)) / (gel + ger)
    }
}

/// `STARPU`: star pressure and velocity; `None` when vacuum is generated.
fn starpu(c: &G, l: W, r: W) -> Option<(f64, f64)> {
    if c.g4 * (sound(c, l) + sound(c, r)) <= r.u - l.u {
        return None;
    }
    let mut p = guessp(c, l, r).max(1e-6);
    for _ in 0..200 {
        let (fl, fld) = prefun(c, p, l);
        let (fr, frd) = prefun(c, p, r);
        let mut pn = p - (fl + fr + (r.u - l.u)) / (fld + frd);
        if pn <= 0.0 {
            pn = 0.5 * p;
        }
        let change = 2.0 * (pn - p).abs() / (pn + p);
        p = pn;
        if change < 1e-15 {
            break;
        }
    }
    let (fl, _) = prefun(c, p, l);
    let (fr, _) = prefun(c, p, r);
    Some((p, 0.5 * (l.u + r.u) + 0.5 * (fr - fl)))
}

/// `SAMPLE`: the solution at `s = x/t`.
fn sample(c: &G, l: W, r: W, pm: f64, um: f64, s: f64) -> W {
    if s <= um {
        let al = sound(c, l);
        if pm <= l.p {
            if s <= l.u - al {
                return l;
            }
            let cml = al * (pm / l.p).powf(c.g1);
            if s > um - cml {
                return W {
                    rho: l.rho * (pm / l.p).powf(1.0 / c.g),
                    u: um,
                    p: pm,
                };
            }
            let cc = c.g5 * (al + c.g7 * (l.u - s));
            W {
                rho: l.rho * (cc / al).powf(c.g4),
                u: c.g5 * (al + c.g7 * l.u + s),
                p: l.p * (cc / al).powf(c.g3),
            }
        } else {
            let pml = pm / l.p;
            if s <= l.u - al * (c.g2 * pml + c.g1).sqrt() {
                return l;
            }
            W {
                rho: l.rho * (pml + c.g6) / (pml * c.g6 + 1.0),
                u: um,
                p: pm,
            }
        }
    } else {
        let ar = sound(c, r);
        if pm > r.p {
            let pmr = pm / r.p;
            if s >= r.u + ar * (c.g2 * pmr + c.g1).sqrt() {
                return r;
            }
            W {
                rho: r.rho * (pmr + c.g6) / (pmr * c.g6 + 1.0),
                u: um,
                p: pm,
            }
        } else {
            if s >= r.u + ar {
                return r;
            }
            let cmr = ar * (pm / r.p).powf(c.g1);
            if s < um + cmr {
                return W {
                    rho: r.rho * (pm / r.p).powf(1.0 / c.g),
                    u: um,
                    p: pm,
                };
            }
            let cc = c.g5 * (ar - c.g7 * (r.u - s));
            W {
                rho: r.rho * (cc / ar).powf(c.g4),
                u: c.g5 * (-ar + c.g7 * r.u + s),
                p: r.p * (cc / ar).powf(c.g3),
            }
        }
    }
}

fn w(rho: f64, u: f64, p: f64) -> W {
    W { rho, u, p }
}

fn as_prim(v: W) -> Primitive {
    prim(v.rho, v.u, v.p)
}

/// Riemann problems used throughout: (name, left, right, Toro Table 4.2 (p*, u*)
/// with one unit of the last printed digit as tolerance, or None).
#[allow(clippy::type_complexity)]
fn riemann_cases() -> Vec<(&'static str, W, W, Option<(f64, f64, f64, f64)>)> {
    vec![
        // Toro Table 4.1 Test 1 (Sod)
        (
            "toro1_sod",
            w(1.0, 0.0, 1.0),
            w(0.125, 0.0, 0.1),
            Some((0.30313, 1e-5, 0.92745, 1e-5)),
        ),
        // Test 2 (123 problem, two strong rarefactions)
        (
            "toro2_123",
            w(1.0, -2.0, 0.4),
            w(1.0, 2.0, 0.4),
            Some((0.00189, 1e-5, 0.0, 1e-5)),
        ),
        // Test 3 (left half of the blast wave of Woodward and Colella)
        (
            "toro3_blast_left",
            w(1.0, 0.0, 1000.0),
            w(1.0, 0.0, 0.01),
            Some((460.894, 1e-3, 19.5975, 1e-4)),
        ),
        // Test 4 (right half of the blast wave)
        (
            "toro4_blast_right",
            w(1.0, 0.0, 0.01),
            w(1.0, 0.0, 100.0),
            Some((46.0950, 1e-4, -6.19633, 1e-5)),
        ),
        // Test 5 (collision of the two blast waves). Its data are the star
        // states of Tests 3 / 4 rounded to 6 figures, which moves the solution
        // by a few 1e-6 relative; the table check gets 5e-6 relative for that.
        (
            "toro5_collision",
            w(5.99924, 19.5975, 460.894),
            w(5.99242, -6.19633, 46.0950),
            Some((1691.64, 1e-2, 8.68975, 5e-5)),
        ),
        // Lax (1954)
        ("lax", w(0.445, 0.698, 3.528), w(0.5, 0.0, 0.571), None),
    ]
}

fn close(label: &str, actual: f64, expected: f64, tol: f64) {
    let scale = expected.abs().max(1.0);
    let err = (actual - expected).abs() / scale;
    assert!(
        err <= tol,
        "{label}: actual {actual:.17e}, expected {expected:.17e}, err {err:e} > {tol:e}"
    );
}

// ===========================================================================
// Exact Riemann solver
// ===========================================================================

#[test]
fn reference_matches_toro_table_4_2() {
    // Guards the f64 reference itself against an external source.
    let c = gas(GAMMA);
    for (name, l, r, table) in riemann_cases() {
        let Some((pt, pe, ut, ue)) = table else {
            continue;
        };
        let (p, u) = starpu(&c, l, r).expect("no vacuum");
        assert!((p - pt).abs() <= pe, "{name}: ref p* {p} vs table {pt}");
        assert!((u - ut).abs() <= ue, "{name}: ref u* {u} vs table {ut}");
    }
}

#[test]
fn star_state_matches_f64_reference() {
    let c = gas(GAMMA);
    for (name, l, r, _) in riemann_cases() {
        let (p_ref, u_ref) = starpu(&c, l, r).expect("no vacuum");
        let star = exact_riemann(gamma(), &as_prim(l), &as_prim(r)).expect("no vacuum");
        eprintln!(
            "{name}: p* {:.15e} (|err| {:.1e}), u* {:.15e} (|err| {:.1e}), {} Newton iterations",
            star.p.to_f64(),
            (star.p.to_f64() - p_ref).abs(),
            star.u.to_f64(),
            (star.u.to_f64() - u_ref).abs(),
            star.iterations
        );
        close(&format!("{name} p*"), star.p.to_f64(), p_ref, 1e-10);
        close(&format!("{name} u*"), star.u.to_f64(), u_ref, 1e-10);
        assert!(star.iterations >= 1 && star.iterations <= 64, "{name}");
    }
}

#[test]
fn sampled_solution_matches_f64_reference() {
    let c = gas(GAMMA);
    for (name, l, r, _) in riemann_cases() {
        let (pm, um) = starpu(&c, l, r).expect("no vacuum");
        let (lp, rp) = (as_prim(l), as_prim(r));
        let star = exact_riemann(gamma(), &lp, &rp).expect("no vacuum");
        // Speeds spanning every wave of every case (fans included).
        for k in -400..=400 {
            let s = f64::from(k) * 0.125;
            let e = sample(&c, l, r, pm, um, s);
            let a = star.sample(gamma(), &lp, &rp, fx(s));
            close(&format!("{name} rho(s={s})"), a.rho.to_f64(), e.rho, 1e-9);
            close(&format!("{name} u(s={s})"), a.u.to_f64(), e.u, 1e-9);
            close(&format!("{name} p(s={s})"), a.p.to_f64(), e.p, 1e-9);
        }
    }
}

#[test]
fn vacuum_generation_is_an_error() {
    // G4 (aL + aR) = 5 · 2 · √(1.4·0.4) ≈ 7.48 < uR − uL = 20.
    let l = prim(1.0, -10.0, 0.4);
    let r = prim(1.0, 10.0, 0.4);
    assert_eq!(
        exact_riemann(gamma(), &l, &r).unwrap_err(),
        EulerError::VacuumGenerated
    );
    assert_eq!(
        numerical_flux(RiemannSolver::Exact, gamma(), &l, &r).unwrap_err(),
        EulerError::VacuumGenerated
    );
    // Just short of the condition is solvable.
    let l = prim(1.0, -3.0, 0.4);
    let r = prim(1.0, 3.0, 0.4);
    assert!(exact_riemann(gamma(), &l, &r).is_ok());
}

#[test]
fn riemann_rejects_invalid_states() {
    let ok = prim(1.0, 0.0, 1.0);
    assert_eq!(
        exact_riemann(gamma(), &prim(0.0, 0.0, 1.0), &ok).unwrap_err(),
        EulerError::NonPositiveDensity { cell: 0 }
    );
    assert_eq!(
        exact_riemann(gamma(), &ok, &prim(1.0, 0.0, -1.0)).unwrap_err(),
        EulerError::NonPositivePressure { cell: 1 }
    );
    assert_eq!(
        exact_riemann(Fix128::ONE, &ok, &ok).unwrap_err(),
        EulerError::GammaNotAboveOne
    );
    assert_eq!(
        numerical_flux(RiemannSolver::Hllc, gamma(), &ok, &prim(-1.0, 0.0, 1.0)).unwrap_err(),
        EulerError::NonPositiveDensity { cell: 1 }
    );
}

#[test]
fn numerical_flux_is_consistent() {
    // F(W, W) = F(W) for both solvers, bit for bit for HLLC (0 ≤ S_L or the
    // subsonic star branch with U* = U) is not guaranteed, so 1e-15.
    for st in [
        prim(1.0, 0.3, 1.0),
        prim(0.5, -0.7, 2.0),
        prim(2.0, 3.0, 0.5),
    ] {
        let f = physical_flux(gamma(), &st);
        for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
            let g = numerical_flux(solver, gamma(), &st, &st).expect("valid");
            close("mass", g.rho.to_f64(), f.rho.to_f64(), 1e-15);
            close("mom", g.mom.to_f64(), f.mom.to_f64(), 1e-15);
            close("energy", g.energy.to_f64(), f.energy.to_f64(), 1e-15);
        }
        // physical flux against the f64 formula
        let (r, u, p) = (st.rho.to_f64(), st.u.to_f64(), st.p.to_f64());
        let e = p / (GAMMA - 1.0) + 0.5 * r * u * u;
        close("F mass", f.rho.to_f64(), r * u, 1e-15);
        close("F mom", f.mom.to_f64(), r * u * u + p, 1e-15);
        close("F energy", f.energy.to_f64(), u * (e + p), 1e-15);
    }
}

#[test]
fn hllc_flux_matches_f64_formula() {
    // HLLC with Einfeldt (Roe-average) wave speeds, Toro 3rd ed. §10.4 / §10.5.1,
    // evaluated in f64 here.
    fn hllc(l: W, r: W) -> [f64; 3] {
        let g = GAMMA;
        let al = (g * l.p / l.rho).sqrt();
        let ar = (g * r.p / r.rho).sqrt();
        let el = l.p / (g - 1.0) + 0.5 * l.rho * l.u * l.u;
        let er = r.p / (g - 1.0) + 0.5 * r.rho * r.u * r.u;
        let (sl_, sr_) = (l.rho.sqrt(), r.rho.sqrt());
        let ut = (sl_ * l.u + sr_ * r.u) / (sl_ + sr_);
        let ht = (sl_ * (el + l.p) / l.rho + sr_ * (er + r.p) / r.rho) / (sl_ + sr_);
        let at = ((g - 1.0) * (ht - 0.5 * ut * ut)).sqrt();
        let sl = (l.u - al).min(ut - at);
        let sr = (r.u + ar).max(ut + at);
        let dl = l.rho * (sl - l.u);
        let dr = r.rho * (sr - r.u);
        let ss = (r.p - l.p + l.u * dl - r.u * dr) / (dl - dr);
        let f = |x: W, e: f64| [x.rho * x.u, x.rho * x.u * x.u + x.p, x.u * (e + x.p)];
        let star = |x: W, e: f64, s: f64| {
            let k = x.rho * (s - x.u) / (s - ss);
            [
                k,
                k * ss,
                k * (e / x.rho + (ss - x.u) * (ss + x.p / (x.rho * (s - x.u)))),
            ]
        };
        if 0.0 <= sl {
            f(l, el)
        } else if 0.0 >= sr {
            f(r, er)
        } else if ss >= 0.0 {
            let (fl, us) = (f(l, el), star(l, el, sl));
            let ul = [l.rho, l.rho * l.u, el];
            [0, 1, 2].map(|i| fl[i] + sl * (us[i] - ul[i]))
        } else {
            let (fr, us) = (f(r, er), star(r, er, sr));
            let ur = [r.rho, r.rho * r.u, er];
            [0, 1, 2].map(|i| fr[i] + sr * (us[i] - ur[i]))
        }
    }
    let mut cases: Vec<(W, W)> = riemann_cases().iter().map(|c| (c.1, c.2)).collect();
    cases.push((w(1.0, 2.5, 1.0), w(0.8, 2.2, 0.9))); // supersonic to the right
    cases.push((w(1.0, -2.5, 1.0), w(0.8, -2.2, 0.9))); // supersonic to the left
    for (l, r) in cases {
        let e = hllc(l, r);
        let a =
            numerical_flux(RiemannSolver::Hllc, gamma(), &as_prim(l), &as_prim(r)).expect("valid");
        close("hllc mass", a.rho.to_f64(), e[0], 1e-12);
        close("hllc mom", a.mom.to_f64(), e[1], 1e-12);
        close("hllc energy", a.energy.to_f64(), e[2], 1e-12);
    }
}

// ===========================================================================
// Finite-volume solver: shock tube accuracy and convergence
// ===========================================================================

fn shock_tube(n: usize, cfg: EulerConfig, l: W, r: W, x0: f64) -> EulerFv1d {
    let dx = 1.0 / n as f64;
    let init: Vec<Primitive> = (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) * dx;
            if x < x0 {
                as_prim(l)
            } else {
                as_prim(r)
            }
        })
        .collect();
    EulerFv1d::new(cfg, Fix128::from_ratio(1, n as i64), &init).expect("valid")
}

/// L1 error of the density against the exact solution at cell centres.
fn l1_density(sim: &EulerFv1d, l: W, r: W, x0: f64, t: f64) -> f64 {
    let c = gas(GAMMA);
    let (pm, um) = starpu(&c, l, r).expect("no vacuum");
    let n = sim.cells().len();
    let dx = 1.0 / n as f64;
    sim.primitives()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let x = (i as f64 + 0.5) * dx;
            let e = sample(&c, l, r, pm, um, (x - x0) / t);
            (p.rho.to_f64() - e.rho).abs() * dx
        })
        .sum()
}

fn cfg(solver: RiemannSolver, rec: Reconstruction, time: TimeIntegrator) -> EulerConfig {
    EulerConfig {
        gamma: gamma(),
        solver,
        reconstruction: rec,
        time,
        cfl: Fix128::from_ratio(9, 10),
        left: Boundary::Transmissive,
        right: Boundary::Transmissive,
    }
}

fn godunov() -> EulerConfig {
    cfg(
        RiemannSolver::Exact,
        Reconstruction::FirstOrder,
        TimeIntegrator::ForwardEuler,
    )
}

fn muscl(solver: RiemannSolver, lim: Limiter) -> EulerConfig {
    let mut c = cfg(solver, Reconstruction::Muscl(lim), TimeIntegrator::SspRk2);
    c.cfl = Fix128::from_ratio(1, 2);
    c
}

/// First-order Godunov on Sod: L1 decreases under refinement, but slower than
/// first order. The contact discontinuity is smeared over `O(√(t/Δx))` cells
/// by the numerical diffusion `O(Δx)` of a first-order scheme, so its L1
/// contribution is `O(Δx^{1/2})`; the shock is a self-sharpening wave held to
/// `O(1)` cells (L1 `O(Δx)`) and the rarefaction is smooth except at its edges
/// (`O(Δx)` up to a log). The total rate therefore lies between 1/2 and 1 —
/// a ratio between `2^{1/2} ≈ 1.41` and `2` per halving (Toro 3rd ed. §6.3 /
/// the standard result for linear discontinuities, Harten 1978). The measured
/// ratios are listed in the assert messages.
#[test]
fn godunov_sod_l1_converges_below_first_order() {
    let (l, r) = (w(1.0, 0.0, 1.0), w(0.125, 0.0, 0.1));
    let t = 0.2;
    let mut errs = Vec::new();
    for n in [50usize, 100, 200] {
        let mut sim = shock_tube(n, godunov(), l, r, 0.5);
        sim.advance_to(fx(t)).expect("stable");
        assert_eq!(sim.time(), fx(t));
        errs.push(l1_density(&sim, l, r, 0.5, t));
    }
    let r1 = errs[0] / errs[1];
    let r2 = errs[1] / errs[2];
    eprintln!("godunov sod L1 {errs:?}, ratios {r1:.3} {r2:.3}");
    assert!(errs[0] < 0.05, "L1 at N=50: {}", errs[0]);
    for ratio in [r1, r2] {
        assert!(
            (1.35..2.0).contains(&ratio),
            "L1 {errs:?}, ratios {r1} {r2}: expected a rate between 1/2 and 1"
        );
    }
}

/// Second-order MUSCL + SSP-RK2 on Sod is more accurate than first order at
/// the same N, for both solvers and both limiters, and van Leer is more
/// accurate than minmod.
#[test]
fn muscl_sod_beats_first_order() {
    let (l, r) = (w(1.0, 0.0, 1.0), w(0.125, 0.0, 0.1));
    let t = 0.2;
    let n = 100;
    let mut first = shock_tube(n, godunov(), l, r, 0.5);
    first.advance_to(fx(t)).expect("stable");
    let e1 = l1_density(&first, l, r, 0.5, t);
    for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
        let mut by_limiter = Vec::new();
        for lim in [Limiter::Minmod, Limiter::VanLeer] {
            let mut sim = shock_tube(n, muscl(solver, lim), l, r, 0.5);
            sim.advance_to(fx(t)).expect("stable");
            let e2 = l1_density(&sim, l, r, 0.5, t);
            eprintln!("sod N=100 {solver:?} {lim:?}: L1 {e2:.5e} (first order {e1:.5e})");
            assert!(
                e2 < 0.75 * e1,
                "{solver:?} {lim:?}: L1 {e2} vs first order {e1}"
            );
            by_limiter.push(e2);
        }
        // minmod is the most diffusive TVD limiter; van Leer resolves the
        // contact and the rarefaction edges more sharply (measured: 0.66× with
        // the exact solver, 0.71× with HLLC at N = 100)
        assert!(
            by_limiter[1] < 0.85 * by_limiter[0],
            "{solver:?}: van Leer {} vs minmod {}",
            by_limiter[1],
            by_limiter[0]
        );
    }
}

/// Lax and the 123 problem run to completion with every scheme and stay
/// close to the exact density (loose bound: these are robustness checks).
#[test]
fn lax_and_123_problems_run_with_every_scheme() {
    let cases = [
        (
            "lax",
            w(0.445, 0.698, 3.528),
            w(0.5, 0.0, 0.571),
            0.14,
            0.06,
        ),
        ("123", w(1.0, -2.0, 0.4), w(1.0, 2.0, 0.4), 0.15, 0.06),
    ];
    for (name, l, r, t, tol) in cases {
        for c in [
            godunov(),
            cfg(
                RiemannSolver::Hllc,
                Reconstruction::FirstOrder,
                TimeIntegrator::ForwardEuler,
            ),
            muscl(RiemannSolver::Exact, Limiter::Minmod),
            muscl(RiemannSolver::Hllc, Limiter::VanLeer),
        ] {
            let mut sim = shock_tube(200, c, l, r, 0.5);
            sim.advance_to(fx(t)).expect("stable");
            let e = l1_density(&sim, l, r, 0.5, t);
            eprintln!("{name} {:?} {:?}: L1 {e:.4e}", c.solver, c.reconstruction);
            assert!(
                e < tol,
                "{name} {:?} {:?}: L1 {e}",
                c.solver,
                c.reconstruction
            );
        }
    }
}

// ===========================================================================
// Smooth problem: second-order convergence
// ===========================================================================

/// `ρ = 1 + ε sin 2πx`, `u = 1`, `p = 1`, periodic on `[0, 1]`: a contact
/// wave advected at `u`. Exact cell averages are
/// `1 + ε (cos 2πx_{i−½} − cos 2πx_{i+½}) / (2π Δx)`.
fn sine_wave(n: usize, c: EulerConfig, t_shift: f64) -> Vec<Primitive> {
    let _ = c;
    let dx = 1.0 / n as f64;
    let eps = 0.2;
    (0..n)
        .map(|i| {
            let a = i as f64 * dx - t_shift;
            let b = a + dx;
            let tau = std::f64::consts::TAU;
            let avg = 1.0 + eps * ((tau * a).cos() - (tau * b).cos()) / (tau * dx);
            prim(avg, 1.0, 1.0)
        })
        .collect()
}

fn smooth_l1(n: usize, c: EulerConfig, t: f64) -> f64 {
    let mut c = c;
    c.left = Boundary::Periodic;
    c.right = Boundary::Periodic;
    let init = sine_wave(n, c, 0.0);
    let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
    sim.advance_to(fx(t)).expect("stable");
    let exact = sine_wave(n, c, t);
    let dx = 1.0 / n as f64;
    sim.primitives()
        .iter()
        .zip(exact.iter())
        .map(|(a, e)| (a.rho.to_f64() - e.rho.to_f64()).abs() * dx)
        .sum()
}

/// MUSCL (van Leer, primitive variables) + SSP-RK2 is second order on a smooth
/// solution: halving Δx divides the L1 error by ≈ 4. The limiter clips the
/// slope only in the O(1) cells at each extremum, each with an O(Δx²) error
/// over a width Δx, which does not change the L1 rate. First-order Godunov on
/// the same problem divides it by ≈ 2.
#[test]
fn muscl_ssprk2_is_second_order_on_smooth_wave() {
    let t = 0.25;
    for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
        let c = muscl(solver, Limiter::VanLeer);
        let e: Vec<f64> = [32usize, 64, 128]
            .iter()
            .map(|&n| smooth_l1(n, c, t))
            .collect();
        let (r1, r2) = (e[0] / e[1], e[1] / e[2]);
        eprintln!("muscl {solver:?} smooth L1 {e:?}, ratios {r1:.3} {r2:.3}");
        assert!(
            (3.3..4.7).contains(&r2) && r1 > 3.0,
            "{solver:?}: L1 {e:?}, ratios {r1} {r2}"
        );
    }
    let e: Vec<f64> = [32usize, 64, 128]
        .iter()
        .map(|&n| smooth_l1(n, godunov(), t))
        .collect();
    let (r1, r2) = (e[0] / e[1], e[1] / e[2]);
    eprintln!("first order smooth L1 {e:?}, ratios {r1:.3} {r2:.3}");
    assert!(
        (1.6..2.4).contains(&r1) && (1.6..2.4).contains(&r2),
        "first order: L1 {e:?}, ratios {r1} {r2}"
    );
}

// ===========================================================================
// Invariants
// ===========================================================================

fn all_configs() -> Vec<EulerConfig> {
    let mut v = Vec::new();
    for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
        v.push(cfg(
            solver,
            Reconstruction::FirstOrder,
            TimeIntegrator::ForwardEuler,
        ));
        v.push(cfg(
            solver,
            Reconstruction::FirstOrder,
            TimeIntegrator::SspRk2,
        ));
        for lim in [Limiter::Minmod, Limiter::VanLeer] {
            v.push(muscl(solver, lim));
        }
    }
    v
}

/// A rough periodic profile: two density/pressure jumps plus a velocity bump.
fn rough_periodic(n: usize) -> Vec<Primitive> {
    (0..n)
        .map(|i| {
            let x = (i as f64 + 0.5) / n as f64;
            let (rho, p) = if (0.2..0.45).contains(&x) {
                (2.0, 3.0)
            } else {
                (0.7, 0.6)
            };
            let u = 0.4 * (std::f64::consts::TAU * x).sin();
            prim(rho, u, p)
        })
        .collect()
}

#[test]
fn periodic_totals_are_conserved_bit_for_bit() {
    for mut c in all_configs() {
        c.left = Boundary::Periodic;
        c.right = Boundary::Periodic;
        let mut sim =
            EulerFv1d::new(c, Fix128::from_ratio(1, 64), &rough_periodic(64)).expect("valid");
        let t0 = sim.totals();
        for _ in 0..40 {
            sim.step().expect("stable");
            assert_eq!(sim.totals(), t0, "{:?}", c);
        }
        // the state did change
        assert_ne!(
            sim.cells(),
            EulerFv1d::new(c, Fix128::from_ratio(1, 64), &rough_periodic(64))
                .unwrap()
                .cells()
        );
    }
}

/// Closed box: reflective walls carry no mass and no energy (the wall flux
/// is evaluated at `u* = 0`, which the sign-symmetric arithmetic reproduces
/// exactly), so total mass and energy are conserved bit for bit; momentum is
/// not (the wall pushes).
#[test]
fn closed_box_conserves_mass_and_energy_bit_for_bit() {
    for mut c in all_configs() {
        c.left = Boundary::ReflectiveWall;
        c.right = Boundary::ReflectiveWall;
        let mut sim =
            EulerFv1d::new(c, Fix128::from_ratio(1, 64), &rough_periodic(64)).expect("valid");
        let t0 = sim.totals();
        for _ in 0..40 {
            sim.step().expect("stable");
            let t = sim.totals();
            assert_eq!(t.rho, t0.rho, "{c:?}");
            assert_eq!(t.energy, t0.energy, "{c:?}");
        }
    }
}

/// A uniform state is a fixed point bit for bit with transmissive boundaries
/// (all interface fluxes are the same value, so every difference is zero).
#[test]
fn uniform_state_is_a_fixed_point() {
    for c in all_configs() {
        let init = vec![prim(0.8, 0.37, 1.3); 20];
        let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, 20), &init).expect("valid");
        let u0 = sim.cells().to_vec();
        for _ in 0..10 {
            sim.step().expect("stable");
        }
        assert_eq!(sim.cells(), &u0[..], "{c:?}");
    }
}

/// A gas at rest in a closed box stays at rest bit for bit.
#[test]
fn rest_state_in_closed_box_is_a_fixed_point() {
    for mut c in all_configs() {
        c.left = Boundary::ReflectiveWall;
        c.right = Boundary::ReflectiveWall;
        let init = vec![prim(1.1, 0.0, 0.9); 16];
        let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, 16), &init).expect("valid");
        let u0 = sim.cells().to_vec();
        for _ in 0..10 {
            sim.step().expect("stable");
        }
        assert_eq!(sim.cells(), &u0[..], "{c:?}");
    }
}

/// Isolated contact (`u`, `p` constant, density jump): velocity and pressure
/// stay at their initial values bit for bit. Every flux of such a state is
/// `(ρ_f u, ρ_f u² + p, ρ_f u³/2 + γ p u/(γ−1))` with a common `u`, `p`; at
/// rest it is `(0, p, 0)`, so exactness at `u = 0` is structural. For moving
/// contacts bit-exactness was observed for the listed `u` and every scheme,
/// not derived (the update rounds `ρu`, `ρu·u` and `λF`); a change that loses
/// it is worth a look even where a tolerance would pass.
#[test]
fn contact_discontinuity_keeps_u_and_p() {
    for u in [0.0, 1.0, 0.5, -0.75] {
        for mut c in all_configs() {
            c.left = Boundary::Periodic;
            c.right = Boundary::Periodic;
            let init: Vec<Primitive> = (0..40)
                .map(|i| {
                    let rho = if (10..25).contains(&i) { 3.0 } else { 0.5 };
                    prim(rho, u, 1.0)
                })
                .collect();
            let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, 40), &init).expect("valid");
            for _ in 0..30 {
                sim.step().expect("stable");
            }
            let mut du: f64 = 0.0;
            let mut dp: f64 = 0.0;
            for p in sim.primitives() {
                du = du.max((p.u.to_f64() - u).abs());
                dp = dp.max((p.p.to_f64() - 1.0).abs());
            }
            eprintln!(
                "contact u={u} {:?} {:?} {:?}: |du| {du:e} |dp| {dp:e}",
                c.solver, c.reconstruction, c.time
            );
            assert_eq!(du, 0.0, "u={u} {c:?}");
            assert_eq!(dp, 0.0, "u={u} {c:?}");
        }
    }
}

/// Rankine–Hugoniot: upstream / downstream states of a stationary normal
/// shock from `compressible::normal_shock_jump` (the crate's closed form)
/// form a Riemann problem whose exact solution is that single shock, so
/// `p* = p₂`, `u* = u₂`; the finite-volume solver keeps the shock in place
/// and the far states unchanged.
#[test]
fn stationary_shock_matches_normal_shock_jump() {
    let air = IdealGas::air();
    for m1 in [fx(1.5), Fix128::from_int(2), Fix128::from_int(3)] {
        let jump = normal_shock_jump(&air, m1);
        let (rho1, p1) = (Fix128::ONE, Fix128::ONE);
        let a1 = (air.gamma * p1 / rho1).sqrt();
        let u1 = m1 * a1;
        let rho2 = jump.density_ratio * rho1;
        let p2 = jump.pressure_ratio * p1;
        let u2 = u1 / jump.density_ratio; // ρ₁u₁ = ρ₂u₂
        let up = Primitive {
            rho: rho1,
            u: u1,
            p: p1,
        };
        let down = Primitive {
            rho: rho2,
            u: u2,
            p: p2,
        };
        let star = exact_riemann(air.gamma, &up, &down).expect("no vacuum");
        close("p* = p2", star.p.to_f64(), p2.to_f64(), 1e-12);
        close("u* = u2", star.u.to_f64(), u2.to_f64(), 1e-12);
        // M₂ from the jump agrees with the downstream state
        let a2 = (air.gamma * p2 / rho2).sqrt();
        close(
            "M2",
            (u2 / a2).to_f64(),
            jump.mach_downstream.to_f64(),
            1e-12,
        );

        for c in [godunov(), muscl(RiemannSolver::Hllc, Limiter::Minmod)] {
            let mut c = c;
            c.gamma = air.gamma;
            let n = 60;
            let init: Vec<Primitive> = (0..n).map(|i| if i < 30 { up } else { down }).collect();
            let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
            for _ in 0..150 {
                sim.step().expect("stable");
            }
            let prims = sim.primitives();
            for (i, q) in prims.iter().enumerate() {
                let (e, label) = if i < 26 {
                    (up, "upstream")
                } else if i >= 34 {
                    (down, "downstream")
                } else {
                    continue;
                };
                close(label, q.rho.to_f64(), e.rho.to_f64(), 1e-12);
                close(label, q.u.to_f64(), e.u.to_f64(), 1e-12);
                close(label, q.p.to_f64(), e.p.to_f64(), 1e-12);
            }
        }
    }
}

/// Reflecting wall: gas moving into the right wall at `u` is stopped by a
/// reflected shock; next to the wall the pressure is the star pressure of the
/// Riemann problem `(ρ, u, p) | (ρ, −u, p)` (wall = symmetry plane).
#[test]
fn reflective_wall_reproduces_reflected_shock_pressure() {
    let c64 = gas(GAMMA);
    let st = w(1.0, 0.8, 1.0);
    let (p_ref, u_ref) = starpu(&c64, st, w(1.0, -0.8, 1.0)).unwrap();
    assert!(u_ref.abs() < 1e-15);
    for mut c in all_configs() {
        c.right = Boundary::ReflectiveWall;
        let n = 100;
        let init = vec![as_prim(st); n];
        let mut sim = EulerFv1d::new(c, Fix128::from_ratio(1, n as i64), &init).expect("valid");
        sim.advance_to(fx(0.2)).expect("stable");
        let prims = sim.primitives();
        // cells 2..10 from the wall are inside the shocked region (shock speed ≈ 0.68)
        for q in &prims[n - 10..n - 2] {
            close("p behind reflected shock", q.p.to_f64(), p_ref, 2e-2);
            assert!(q.u.to_f64().abs() < 2e-2, "{c:?}: u {}", q.u.to_f64());
        }
    }
}

/// Mirror `x → −x` (cell `i → N−1−i`, `u → −u`) maps the solution to its mirror
/// image bit for bit, for every scheme and with walls or transmissive ends.
#[test]
fn mirrored_initial_value_gives_mirrored_solution_bit_for_bit() {
    let base: Vec<Primitive> = (0..48)
        .map(|i| {
            let x = (i as f64 + 0.5) / 48.0;
            if x < 0.4 {
                prim(1.0, 0.3, 1.0)
            } else if x < 0.7 {
                prim(0.125, -0.2, 0.1)
            } else {
                prim(0.6, 0.1 * x, 0.4 + x)
            }
        })
        .collect();
    let mirrored: Vec<Primitive> = base
        .iter()
        .rev()
        .map(|p| Primitive {
            rho: p.rho,
            u: -p.u,
            p: p.p,
        })
        .collect();
    for boundary in [Boundary::Transmissive, Boundary::ReflectiveWall] {
        for mut c in all_configs() {
            c.left = boundary;
            c.right = boundary;
            let dx = Fix128::from_ratio(1, 48);
            let mut a = EulerFv1d::new(c, dx, &base).unwrap();
            let mut b = EulerFv1d::new(c, dx, &mirrored).unwrap();
            for _ in 0..25 {
                let da = a.step().unwrap();
                let db = b.step().unwrap();
                assert_eq!(da, db);
            }
            for (x, y) in a.cells().iter().zip(b.cells().iter().rev()) {
                assert_eq!(x.rho, y.rho, "{c:?}");
                assert_eq!(x.mom, -y.mom, "{c:?}");
                assert_eq!(x.energy, y.energy, "{c:?}");
            }
        }
    }
}

#[test]
fn step_uses_the_cfl_time_step() {
    let c = godunov();
    let (l, r) = (w(1.0, 0.0, 1.0), w(0.125, 0.0, 0.1));
    let mut sim = shock_tube(50, c, l, r, 0.5);
    // max(|u| + a) at t = 0 is a_L = √1.4
    let smax = sim.max_wave_speed();
    close("smax", smax.to_f64(), 1.4f64.sqrt(), 1e-15);
    let dt = sim.step().unwrap();
    close("dt", dt.to_f64(), 0.9 * 0.02 / 1.4f64.sqrt(), 1e-15);
    assert_eq!(sim.time(), dt);
    // the next step uses the new max wave speed
    let smax2 = sim.max_wave_speed();
    let dt2 = sim.step().unwrap();
    close("dt2", dt2.to_f64(), 0.9 * 0.02 / smax2.to_f64(), 1e-15);
    assert_eq!(sim.cfl_time_step().unwrap(), sim.cfl_time_step().unwrap());
}

/// Running at CFL 0.9 is stable and the explicit fixed-dt path agrees with
/// the CFL path bit for bit when given the same dt.
#[test]
fn fixed_dt_path_matches_cfl_path() {
    for c in all_configs() {
        let (l, r) = (w(1.0, 0.0, 1.0), w(0.125, 0.0, 0.1));
        let mut a = shock_tube(40, c, l, r, 0.5);
        let mut b = shock_tube(40, c, l, r, 0.5);
        for _ in 0..20 {
            let dt = a.cfl_time_step().unwrap();
            b.step_with_dt(dt).unwrap();
            assert_eq!(a.step().unwrap(), dt);
        }
        assert_eq!(a.cells(), b.cells());
    }
}

// ===========================================================================
// Degenerate input
// ===========================================================================

#[test]
fn degenerate_inputs_have_explicit_results() {
    let ok = vec![prim(1.0, 0.0, 1.0); 4];
    let dx = Fix128::from_ratio(1, 4);
    // N = 0
    assert_eq!(
        EulerFv1d::new(godunov(), dx, &[]).unwrap_err(),
        EulerError::NoCells
    );
    // ρ ≤ 0 / p ≤ 0 (index of the first bad cell)
    let mut bad = ok.clone();
    bad[2].rho = Fix128::ZERO;
    assert_eq!(
        EulerFv1d::new(godunov(), dx, &bad).unwrap_err(),
        EulerError::NonPositiveDensity { cell: 2 }
    );
    let mut bad = ok.clone();
    bad[3].p = fx(-0.5);
    assert_eq!(
        EulerFv1d::new(godunov(), dx, &bad).unwrap_err(),
        EulerError::NonPositivePressure { cell: 3 }
    );
    // γ ≤ 1
    for g in [Fix128::ONE, fx(0.5), Fix128::ZERO] {
        let mut c = godunov();
        c.gamma = g;
        assert_eq!(
            EulerFv1d::new(c, dx, &ok).unwrap_err(),
            EulerError::GammaNotAboveOne
        );
    }
    // CFL ≤ 0 / CFL > 1
    for (cfl, e) in [
        (Fix128::ZERO, EulerError::CflOutOfRange),
        (fx(-0.1), EulerError::CflOutOfRange),
        (fx(1.01), EulerError::CflOutOfRange),
    ] {
        let mut c = godunov();
        c.cfl = cfl;
        assert_eq!(EulerFv1d::new(c, dx, &ok).unwrap_err(), e);
    }
    // Δx ≤ 0
    assert_eq!(
        EulerFv1d::new(godunov(), Fix128::ZERO, &ok).unwrap_err(),
        EulerError::NonPositiveSpacing
    );
    // periodic on one side only
    let mut c = godunov();
    c.left = Boundary::Periodic;
    assert_eq!(
        EulerFv1d::new(c, dx, &ok).unwrap_err(),
        EulerError::PeriodicMismatch
    );
    // dt ≤ 0
    let mut sim = EulerFv1d::new(godunov(), dx, &ok).unwrap();
    assert_eq!(
        sim.step_with_dt(Fix128::ZERO).unwrap_err(),
        EulerError::NonPositiveTimeStep
    );
    // advance_to a time not after the current time does nothing
    assert_eq!(sim.advance_to(Fix128::ZERO).unwrap(), 0);
    // CFL = 1 exactly is accepted
    let mut c = godunov();
    c.cfl = Fix128::ONE;
    assert!(EulerFv1d::new(c, dx, &ok).is_ok());
}

#[test]
fn single_cell_is_valid() {
    for c in all_configs() {
        for boundary in [Boundary::Transmissive, Boundary::Periodic] {
            let mut c = c;
            c.left = boundary;
            c.right = boundary;
            let init = [prim(1.3, 0.4, 0.7)];
            let mut sim = EulerFv1d::new(c, Fix128::ONE, &init).unwrap();
            let u0 = sim.cells().to_vec();
            sim.step().unwrap();
            assert_eq!(sim.cells(), &u0[..], "{c:?} {boundary:?}");
        }
        // one cell between two walls, moving: momentum changes, mass and energy do not
        let mut c = c;
        c.left = Boundary::ReflectiveWall;
        c.right = Boundary::ReflectiveWall;
        let mut sim = EulerFv1d::new(c, Fix128::ONE, &[prim(1.3, 0.4, 0.7)]).unwrap();
        let u0 = sim.cells()[0];
        sim.step().unwrap();
        let u1 = sim.cells()[0];
        assert_eq!(u1.rho, u0.rho);
        assert_eq!(u1.energy, u0.energy);
    }
}

/// A Riemann problem that generates vacuum: the exact solver returns an error
/// and the state is left untouched; HLLC either runs (positive) or returns a
/// positivity error, also without touching the state.
#[test]
fn vacuum_initial_value_is_reported_and_state_is_unchanged() {
    let (l, r) = (w(1.0, -10.0, 0.4), w(1.0, 10.0, 0.4));
    let mut sim = shock_tube(20, godunov(), l, r, 0.5);
    let u0 = sim.cells().to_vec();
    assert_eq!(sim.step().unwrap_err(), EulerError::VacuumGenerated);
    assert_eq!(sim.cells(), &u0[..]);
    assert_eq!(sim.time(), Fix128::ZERO);

    for c in all_configs() {
        let mut sim = shock_tube(20, c, l, r, 0.5);
        let u0 = sim.cells().to_vec();
        let mut n_ok = 0;
        loop {
            let before = sim.cells().to_vec();
            let t_before = sim.time();
            match sim.step() {
                Ok(_) => n_ok += 1,
                Err(e) => {
                    assert!(
                        matches!(
                            e,
                            EulerError::VacuumGenerated
                                | EulerError::NonPositiveDensity { .. }
                                | EulerError::NonPositivePressure { .. }
                        ),
                        "{c:?}: {e:?}"
                    );
                    assert_eq!(sim.cells(), &before[..]);
                    assert_eq!(sim.time(), t_before);
                    break;
                }
            }
            for q in sim.primitives() {
                assert!(q.rho > Fix128::ZERO && q.p > Fix128::ZERO);
            }
            if n_ok >= 30 {
                break;
            }
        }
        let _ = u0;
    }
}

#[test]
fn conserved_primitive_round_trip() {
    let g = gamma();
    let p = prim(0.7, -1.3, 2.2);
    let c = Conserved::from_primitive(g, &p);
    close("E", c.energy.to_f64(), 2.2 / 0.4 + 0.5 * 0.7 * 1.69, 1e-15);
    let q = c.to_primitive(g).unwrap();
    close("rho", q.rho.to_f64(), 0.7, 1e-15);
    close("u", q.u.to_f64(), -1.3, 1e-15);
    close("p", q.p.to_f64(), 2.2, 1e-15);
    let bad = Conserved {
        rho: Fix128::ONE,
        mom: Fix128::from_int(2),
        energy: Fix128::ONE,
    };
    assert_eq!(
        bad.to_primitive(g).unwrap_err(),
        EulerError::NonPositivePressure { cell: 0 }
    );
}

/// A step far above the CFL limit drives a cell negative; the stage check
/// reports it with the cell index and the state and time are not touched.
/// (Both Riemann solvers stay positive under the CFL condition on every
/// problem above, so this is the path that reaches the stage check.)
#[test]
fn oversized_dt_reports_positivity_error_and_keeps_state() {
    for c in all_configs() {
        let (l, r) = (w(1.0, 0.0, 1.0), w(0.125, 0.0, 0.1));
        let mut sim = shock_tube(40, c, l, r, 0.5);
        let u0 = sim.cells().to_vec();
        let dt = sim.cfl_time_step().unwrap();
        let e = sim
            .step_with_dt(Fix128::from_int(20) * dt)
            .expect_err("20x the CFL step must fail");
        assert!(
            matches!(
                e,
                EulerError::NonPositiveDensity { .. } | EulerError::NonPositivePressure { .. }
            ),
            "{c:?}: {e:?}"
        );
        assert_eq!(sim.cells(), &u0[..]);
        assert_eq!(sim.time(), Fix128::ZERO);
    }
}
