//! A sealed lid-driven box with a baffle, stepped with an adaptive time step
//!
//! The six boundary layers come from `set_closed_box_walls`, one side wall is
//! overwritten with a symmetry plane through `set_u_bc`, a baffle is marked
//! face by face with `set_u_solid`, and the impulsive initial field (a uniform
//! stream that violates every wall) is cleaned with `enforce_solid_faces`
//! before the first step. Each step takes the time step `step_adaptive`
//! chooses from the Courant bound of `compute_max_dt`, capped by a ceiling,
//! and the printout shows the bound following the peak face speed as the
//! initial stream is projected out and the lid takes over.
//!
//! ```bash
//! cargo run --example sealed_box_adaptive_dt --features std
//! ```

use alice_physics::cfd_solver::CfdSolver;
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};

fn main() {
    let n = 8usize;
    let dx = Fix128::from_ratio(1, n as i64);
    let mut s = CfdSolver::new(n, n, 1, dx);
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = Fix128::from_ratio(1, 100);
    s.jacobi_iterations = 40;

    // Walls everywhere, then a moving lid on top and a symmetry plane on the
    // right-hand side wall.
    s.grid.set_closed_box_walls();
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    };
    for i in 0..n {
        s.grid.set_v_bc(i, n, 0, lid);
    }
    for j in 0..n {
        s.grid.set_u_bc(n, j, 0, FaceBc::SlipWall);
    }
    // A baffle: the X-faces at i = 3 from the floor to half height, plus one
    // Y-face and one Z-face sealed the same way (`set_v_solid`/`set_w_solid`
    // are the Y/Z counterparts of `set_u_solid`, exercised here so every
    // face-orientation setter has a production caller).
    for j in 0..n / 2 {
        s.grid.set_u_solid(3, j, 0, true);
    }
    s.grid.set_v_solid(4, 3, 0, true);
    s.grid.set_w_solid(4, 3, 0, true);
    println!(
        "[sealed_box] lid velocity read back through no_slip_velocity: {:?}",
        s.grid
            .v_bc(0, n, 0)
            .no_slip_velocity()
            .map(|v| v.x.to_f64())
    );
    println!(
        "[sealed_box] right wall is a symmetry plane: no_slip_velocity = {:?}",
        s.grid.u_bc(n, 0, 0).no_slip_velocity()
    );
    println!(
        "[sealed_box] set_v_solid/set_w_solid at (4,3,0) read back as walls: v={} w={}",
        s.grid.v_bc(4, 3, 0).no_slip_velocity().is_some(),
        s.grid.w_bc(4, 3, 0).no_slip_velocity().is_some()
    );

    // An impulsive start: a uniform stream of 2 m/s through everything,
    // including the walls and the baffle. `enforce_solid_faces` zeroes the
    // flux through every solid face and leaves the open faces alone.
    for u in s.grid.u.iter_mut() {
        *u = Fix128::from_int(2);
    }
    let before = s.compute_max_dt(Fix128::from_ratio(1, 2));
    s.grid.enforce_solid_faces();
    let baffle_flux: f64 = (0..n / 2).map(|j| s.grid.u(3, j, 0).to_f64().abs()).sum();
    println!(
        "[sealed_box] after enforce_solid_faces: baffle flux = {baffle_flux:.3}, open face u(1,1) = {:.3}",
        s.grid.u(1, 1, 0).to_f64()
    );
    println!(
        "[sealed_box] Courant bound at c = 1/2 before the first step: {:.6} s (cap {:.0} s)",
        before.to_f64(),
        CfdSolver::MAX_DT_CAP.to_f64()
    );

    let ceiling = Fix128::from_ratio(1, 16);
    for step in 0..8 {
        let dt = s.step_adaptive(Fix128::from_ratio(1, 2), ceiling);
        let peak = s
            .grid
            .u
            .iter()
            .chain(s.grid.v.iter())
            .map(|x| x.to_f64().abs())
            .fold(0.0_f64, f64::max);
        let flux: f64 = (0..n / 2).map(|j| s.grid.u(3, j, 0).to_f64().abs()).sum();
        println!(
            "[sealed_box] step {step}: dt = {:.6} s ({}), peak |u| = {peak:.4}, baffle flux = {flux:.1e}",
            dt.to_f64(),
            if dt == ceiling { "ceiling" } else { "Courant" }
        );
    }
    println!(
        "[sealed_box] a non-positive ceiling takes no step: dt = {}, step_count stays {}",
        s.step_adaptive(Fix128::from_ratio(1, 2), Fix128::ZERO)
            .to_f64(),
        s.step_count
    );
}
