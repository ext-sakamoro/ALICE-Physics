//! BFECC scalar / velocity advection demo — compares three advection
//! schemes (Semi-Lagrangian / MacCormack / BFECC) on a temperature
//! scalar field carried by a uniform +X flow.
//!
//! Illustrates the v0.14.0-preview.1/2 `cfd_solver` BFECC additions
//! (Priority 1: scalar BFECC; Priority 2: MAC-face BFECC velocity).
//!
//! ```bash
//! cargo run --example bfecc_advection_demo --features std --release
//! ```

use alice_physics::cfd_solver::{AdvectionScheme, CfdSolver};
use alice_physics::math::Fix128;
use alice_physics::multiphase::Grid3d;

const NX: usize = 16;
const NY: usize = 8;
const NZ: usize = 8;

/// Build a solver with a uniform +X velocity of 1.0 m/s on every u-face,
/// and a Gaussian-like hot spot at cell (3, ny/2, nz/2) with radius 1 cell.
fn build_solver(scheme: AdvectionScheme) -> CfdSolver {
    let mut solver = CfdSolver::new(NX, NY, NZ, Fix128::ONE);
    solver.advection_scheme = scheme;
    solver.gravity = alice_physics::math::Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO);

    // Uniform +X flow on every X-face: u = 1.0 m/s.
    // MAC-grid u layout: `[(nx+1) · ny · nz]` in `i + (nx+1)·(j + ny·k)` order.
    for k in 0..NZ {
        for j in 0..NY {
            for i in 0..=NX {
                let idx = i + (NX + 1) * (j + NY * k);
                solver.grid.u[idx] = Fix128::ONE;
            }
        }
    }

    // Temperature scalar field: 0 K background, hot spot of 100 K at (3, 4, 4).
    let mut temp = Grid3d::new(NX, NY, NZ, Fix128::ONE, Fix128::ZERO);
    for k in 3..=5 {
        for j in 3..=5 {
            for i in 2..=4 {
                temp.data[i + NX * (j + NY * k)] = Fix128::from_int(100);
            }
        }
    }
    solver.temperature = Some(temp);
    solver
}

/// Total scalar mass (∑ T · dV) — should be conserved by every advection
/// scheme in a divergence-free flow (numerical diffusion aside).
fn total_mass(solver: &CfdSolver) -> Fix128 {
    solver
        .temperature
        .as_ref()
        .map(|t| t.data.iter().copied().fold(Fix128::ZERO, |a, b| a + b))
        .unwrap_or(Fix128::ZERO)
}

/// Peak temperature — collapses under numerical diffusion (Semi-Lagrangian
/// hits this the hardest, BFECC preserves peaks best).
fn peak(solver: &CfdSolver) -> Fix128 {
    solver
        .temperature
        .as_ref()
        .map(|t| t.data.iter().copied().fold(Fix128::ZERO, Fix128::max))
        .unwrap_or(Fix128::ZERO)
}

fn run(scheme: AdvectionScheme, label: &str) {
    let mut solver = build_solver(scheme);
    let mass_before = total_mass(&solver);
    let peak_before = peak(&solver);
    let dt = Fix128::from_ratio(1, 10);

    for _ in 0..8 {
        solver.step(dt);
    }

    let mass_after = total_mass(&solver);
    let peak_after = peak(&solver);
    println!(
        "{:>18} | mass Δ = {:>+9.4} | peak {:>6.2} → {:>6.2} ({:>6.2} %)",
        label,
        (mass_after - mass_before).to_f32(),
        peak_before.to_f32(),
        peak_after.to_f32(),
        (peak_after / peak_before).to_f32() * 100.0,
    );
}

fn main() {
    println!("ALICE-Physics — BFECC / MacCormack / Semi-Lagrangian Advection Demo");
    println!("====================================================================");
    println!(
        "Grid   : {}×{}×{} cells, dx = 1.0 m, uniform +X flow (u = 1 m/s)",
        NX, NY, NZ
    );
    println!("Signal : 3×3×3 hot spot at (3, 4, 4), temperature 100 K");
    println!("Steps  : 8 × dt = 0.1 s  ⇒  0.8 s advected ≈ 0.8 cell");
    println!(
        "\n{:>18} | mass conservation | peak preservation",
        "scheme"
    );
    println!("-------------------|-------------------|--------------------------");

    run(AdvectionScheme::SemiLagrangian, "SemiLagrangian");
    run(AdvectionScheme::MacCormack, "MacCormack");
    run(AdvectionScheme::Bfecc, "Bfecc");

    println!(
        "\nHigher peak-preservation (%) means less numerical diffusion. BFECC and"
    );
    println!("MacCormack are second-order-in-space schemes and preserve peaks better");
    println!("than Semi-Lagrangian in divergence-free flows.");
}
