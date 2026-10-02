//! A body-force-driven channel with the log-law wall model
//!
//! Two resting walls one unit apart, a body force `G` along the channel, and
//! the wall shear taken from the universal profile instead of the resolved
//! no-slip ghost. At steady state the force balance `2 τ_w = ρ G H` fixes the
//! friction velocity at `√(G H / 2)` whatever the profile's constants are,
//! which is what the printout compares against.
//!
//! ```bash
//! cargo run --example wall_model --features std
//! ```

use alice_physics::cfd_solver::{CfdSolver, PressureSolver, StepOptions, WallModel};
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::turbulence::friction_velocity;

fn main() {
    let (nx, ny) = (4usize, 8usize);
    let (nu, g) = (1.0e-4_f64, 1.0_f64);
    let mut s = CfdSolver::new(nx, ny, 1, Fix128::from_ratio(1, ny as i64));
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = Fix128::from_f64(nu);
    s.gravity = Vec3Fix::new(Fix128::from_f64(g), Fix128::ZERO, Fix128::ZERO);
    let rest = FaceBc::Wall {
        velocity: Vec3Fix::ZERO,
    };
    for i in 0..nx {
        s.grid.set_v_bc(i, 0, 0, rest);
        s.grid.set_v_bc(i, ny, 0, rest);
        for j in 0..ny {
            s.grid.set_w_bc(i, j, 0, FaceBc::SlipWall);
            s.grid.set_w_bc(i, j, 1, FaceBc::SlipWall);
        }
    }
    let options = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 30 })
        .with_wall_model(WallModel::log_law());
    let dt = Fix128::from_int(2);
    println!(
        "stepping with {:?} and wall model {}",
        options.pressure(),
        if options.wall_model().is_some() {
            "on"
        } else {
            "off"
        }
    );

    let mut report = None;
    let mut previous = s.grid.u.clone();
    for step in 0..20_000 {
        previous.clone_from(&s.grid.u);
        report = Some(s.step_with_options(dt, &options).expect("a valid step"));
        let change = s
            .grid
            .u
            .iter()
            .zip(&previous)
            .map(|(a, b)| (a.to_f64() - b.to_f64()).abs())
            .fold(0.0, f64::max);
        if step > 100 && change < 1e-11 {
            println!("settled after {step} steps (max |Δu| per step {change:.1e})");
            break;
        }
    }
    let wall = report
        .expect("stepped")
        .wall
        .expect("the wall model reports");
    let target = (g / 2.0).sqrt();
    println!(
        "friction velocity u_tau = {:.9}  (force balance √(G H / 2) = {target:.9})",
        wall.u_tau_max.to_f64()
    );
    println!(
        "y+ at the first face {:.1}, k = {:.6} m²/s², ε = {:.6} m²/s³, {} wall face updates ({} at rest)",
        wall.y_plus_max.to_f64(),
        wall.k_max.to_f64(),
        wall.epsilon_max.to_f64(),
        wall.faces,
        wall.resting_faces
    );
    // The inverse on its own: a speed at the first face centre.
    let u_tau = friction_velocity(
        Fix128::from_int(10),
        Fix128::from_ratio(1, 16),
        Fix128::from_f64(nu),
    )
    .expect("positive inputs");
    println!(
        "friction_velocity(u = 10, y = 1/16, ν = 1e-4) = {:.6}",
        u_tau.to_f64()
    );
}
