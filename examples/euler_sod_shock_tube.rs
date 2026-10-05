//! Sod's shock tube with the 1D compressible Euler finite-volume solver.
//!
//! ```text
//! ∂U/∂t + ∂F(U)/∂x = 0,   U = (ρ, ρu, E),   F = (ρu, ρu² + p, u(E + p))
//! (ρ, u, p) = (1, 0, 1) for x < 0.5,  (0.125, 0, 0.1) for x > 0.5,  γ = 1.4
//! ```
//!
//! Non-dimensional variables throughout (see the `euler_fv` module doc for
//! why). The example solves the Riemann problem exactly for the star state,
//! runs first-order Godunov (exact Riemann solver) and second-order MUSCL
//! (van Leer) + HLLC + SSP-RK2 to `t = 0.2`, and compares the density with the
//! exact solution: rarefaction, contact and shock.
//!
//! ```bash
//! cargo run --release --example euler_sod_shock_tube
//! ```

use alice_physics::compressible::IdealGas;
use alice_physics::euler_fv::{
    exact_riemann, numerical_flux, EulerConfig, EulerError, EulerFv1d, Limiter, Primitive,
    RiemannSolver,
};
use alice_physics::math::Fix128;

const N: usize = 200;

fn state(rho: Fix128, p: Fix128) -> Primitive {
    Primitive {
        rho,
        u: Fix128::ZERO,
        p,
    }
}

fn main() -> Result<(), EulerError> {
    // γ of air from the closed-form module; the solver needs only γ.
    let gamma = IdealGas::air().gamma;
    let left = state(Fix128::ONE, Fix128::ONE);
    let right = state(Fix128::from_ratio(1, 8), Fix128::from_ratio(1, 10));

    let star = exact_riemann(gamma, &left, &right)?;
    println!(
        "exact star state: p* = {:.6}, u* = {:.6} ({} Newton iterations)",
        star.p.to_f64(),
        star.u.to_f64(),
        star.iterations
    );
    // Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed.,
    // Table 4.2, Test 1: p* = 0.30313, u* = 0.92745.
    assert!((star.p.to_f64() - 0.30313).abs() < 1e-5);
    assert!((star.u.to_f64() - 0.92745).abs() < 1e-5);

    // Interface flux at the initial discontinuity: Godunov's flux is the
    // physical flux of the exact solution at x/t = 0; HLLC approximates it.
    for solver in [RiemannSolver::Exact, RiemannSolver::Hllc] {
        let f = numerical_flux(solver, gamma, &left, &right)?;
        println!(
            "{solver:?} flux at x0: ({:.6}, {:.6}, {:.6})",
            f.rho.to_f64(),
            f.mom.to_f64(),
            f.energy.to_f64()
        );
    }

    let dx = Fix128::from_ratio(1, N as i64);
    let x0 = Fix128::from_ratio(1, 2);
    let t_end = Fix128::from_ratio(1, 5);
    let initial: Vec<Primitive> = (0..N)
        .map(|i| if i < N / 2 { left } else { right })
        .collect();

    // One explicit step: Δt = C_cfl Δx / max(|u| + a) (here a_L = √γ).
    let mut probe = EulerFv1d::new(EulerConfig::godunov(gamma), dx, &initial)?;
    let s_max = probe.max_wave_speed();
    let dt = probe.step()?;
    println!(
        "first step ({:?}, CFL {:.2}): max wave speed {:.6}, dt = {:.6e}",
        probe.config().solver,
        probe.config().cfl.to_f64(),
        s_max.to_f64(),
        dt.to_f64()
    );
    assert_eq!(probe.time(), dt);

    let schemes = [
        ("Godunov (exact), 1st order", EulerConfig::godunov(gamma)),
        (
            "MUSCL van Leer + HLLC + SSP-RK2",
            EulerConfig::muscl(gamma, RiemannSolver::Hllc, Limiter::VanLeer),
        ),
    ];
    let mut l1 = Vec::new();
    let mut last = Vec::new();
    for (name, config) in schemes {
        let mut sim = EulerFv1d::new(config, dx, &initial)?;
        let totals0 = sim.totals();
        let steps = sim.advance_to(t_end)?;
        assert_eq!(sim.time(), t_end);
        assert_eq!(sim.dx(), dx);
        // conserved -> primitive conversion of the cell at the contact
        let mid = sim.cells()[3 * N / 5].to_primitive(gamma)?;
        assert!(mid.p > Fix128::ZERO);
        let prims = sim.primitives();
        let mut err = Fix128::ZERO;
        for (i, w) in prims.iter().enumerate() {
            let x = Fix128::from_ratio(2 * i as i64 + 1, 2 * N as i64);
            let exact = star.sample(gamma, &left, &right, (x - x0) / t_end);
            err = err + (w.rho - exact.rho).abs() * dx;
        }
        // Waves have not reached the ends, so nothing has left the tube.
        let totals = sim.totals();
        println!(
            "{name:34} {steps:4} steps, L1(rho) = {:.3e}, mass drift = {}",
            err.to_f64(),
            (totals.rho - totals0.rho).to_f64()
        );
        assert_eq!(totals.rho, totals0.rho);
        l1.push(err.to_f64());
        last = prims;
    }
    assert!(l1[1] < l1[0], "second order should be more accurate");
    assert!(l1[0] < 0.01 && l1[1] < 0.005);

    println!("\n   x      rho(MUSCL)  rho(exact)");
    for i in (0..N).step_by(10) {
        let x = Fix128::from_ratio(2 * i as i64 + 1, 2 * N as i64);
        let exact = star.sample(gamma, &left, &right, (x - x0) / t_end);
        println!(
            "{:6.3}   {:9.5}   {:9.5}",
            x.to_f64(),
            last[i].rho.to_f64(),
            exact.rho.to_f64()
        );
    }

    // Two strong rarefactions that separate: the exact solver reports vacuum
    // instead of returning a state it cannot represent.
    let receding = |u: i64| Primitive {
        rho: Fix128::ONE,
        u: Fix128::from_int(u),
        p: Fix128::from_ratio(2, 5),
    };
    let err = exact_riemann(gamma, &receding(-10), &receding(10)).unwrap_err();
    println!("\nu = -10 | +10: {err}");
    assert_eq!(err, EulerError::VacuumGenerated);
    Ok(())
}
