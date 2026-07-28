//! CFD gravity settling demo — water settling under gravity + projection.
//!
//! Session 3 E1 demo. Sets up a 12×12×12 water column, runs the integrated
//! CFD solver for N steps, and prints the vertical velocity profile at the
//! grid centreline so you can see gravity acceleration + pressure
//! projection interaction (walls damp the fall, pressure builds up).
//!
//! Full buoyancy-driven plume simulations require careful CFL control and
//! higher-order advection than the first-order operator-splitting shown
//! here — this demo intentionally sticks to gravity + projection to keep
//! the numerics well-behaved on a coarse grid.

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::math::Fix128;

fn main() {
    let n = 12usize;
    let dx = Fix128::from_ratio(1, 10); // 0.1 m cells
    let mut solver = CfdSolver::new(n, n, n, dx);
    solver.jacobi_iterations = 100;

    let dt = Fix128::from_ratio(1, 1000); // 0.001 s per step
    let n_steps = 30;

    println!("=== CFD Gravity Settling Demo ===");
    println!(
        "Grid: {n}³ cells, dx={:.2}m, dt={:.4}s, {n_steps} steps",
        dx.to_f32(),
        dt.to_f32()
    );
    println!("Fluid: water (ρ=1000 kg/m³, μ=1e-3 Pa·s), gravity −9.81 m/s²");
    println!();
    println!("Centre-column v_y[j] every 10 steps:");

    for step in 0..=n_steps {
        if step > 0 {
            solver.step(dt);
        }
        if step % 10 == 0 {
            let mid = n / 2;
            print!("t={:.3}s [", step as f32 * dt.to_f32());
            for j in 0..=n {
                let ix = mid + n * (j + (n + 1) * mid);
                if ix < solver.grid.v.len() {
                    print!("{:+.3} ", solver.grid.v[ix].to_f32());
                }
            }
            println!("]");
        }
    }
    println!(
        "\nMax divergence (centre): {:.4}",
        solver.grid.divergence(6, 6, 6).abs().to_f32()
    );
    println!("Done: {} steps completed.", solver.step_count);
}
