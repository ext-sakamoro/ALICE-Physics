//! The turbulence closures of `step_rans`, side by side
//!
//! A lid-driven cavity is stepped with each `TurbulenceModel` through a
//! caller-owned `RansState`: the static and dynamic Smagorinsky closures take
//! their eddy viscosity from the resolved strain, the k-ε and k-ω closures
//! transport the state's `(k, ε)`, and the prescribed closure reads the field
//! the state was built from. Every step reports the eddy-viscosity envelope,
//! the explicit diffusion number and, for the transport closures, the
//! production and how many cells the source step had to clamp; the last lines
//! show the refusals a closure gives before touching the solver or the state.
//!
//! ```bash
//! cargo run --example turbulence_closures --features std
//! ```

use alice_physics::cfd_solver::{
    CfdSolver, PressureSolver, RansState, StepOptions, TurbulenceModel,
};
use alice_physics::eulerian_grid::FaceBc;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::multiphase::Grid3d;

fn cavity(n: usize) -> CfdSolver {
    let mut s = CfdSolver::new(n, n, 1, Fix128::from_ratio(1, n as i64));
    s.gravity = Vec3Fix::ZERO;
    s.density_kg_m3 = Fix128::ONE;
    s.dynamic_viscosity_pas = Fix128::from_ratio(1, 1000);
    s.jacobi_iterations = 30;
    s.grid.set_closed_box_walls();
    let lid = FaceBc::Wall {
        velocity: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::ZERO),
    };
    for i in 0..n {
        s.grid.set_v_bc(i, n, 0, lid);
    }
    s
}

fn state_for(model: TurbulenceModel, n: usize, dx: Fix128) -> RansState {
    match model {
        TurbulenceModel::Prescribed => {
            RansState::prescribed(Grid3d::new(n, n, 1, dx, Fix128::from_ratio(1, 200)))
        }
        TurbulenceModel::KEpsilon | TurbulenceModel::KOmega => RansState::uniform(
            n,
            n,
            1,
            model,
            Fix128::from_ratio(1, 100),
            Fix128::from_ratio(1, 100),
        ),
        _ => RansState::new(n, n, 1, model),
    }
}

fn main() {
    let n = 8usize;
    let dt = Fix128::from_ratio(1, 256);
    let options = StepOptions::new(PressureSolver::RedBlackGs { sweeps: 30 });
    let models = [
        TurbulenceModel::Smagorinsky,
        TurbulenceModel::DynamicSmagorinsky,
        TurbulenceModel::KEpsilon,
        TurbulenceModel::KOmega,
        TurbulenceModel::Prescribed,
    ];
    for model in models {
        let mut s = cavity(n);
        let mut state = state_for(model, n, s.grid.dx);
        println!("[turbulence] {:?}", state.model());
        for step in 0..64 {
            match s.step_rans(dt, &options, &mut state) {
                Ok(report) => {
                    let t = report.turbulence;
                    if step % 16 == 15 {
                        println!(
                            "[turbulence]   step {step:2}: nu_t in [{:.2e}, {:.2e}] (report field max {:.2e}), diffusion number {:.4}, Cs in [{:.3}, {:.3}], P_max {:.2e}, k in [{:.2e}, {:.2e}], clamped {}",
                            t.nu_t_min.to_f64(),
                            t.nu_t_max.to_f64(),
                            report
                                .eddy_viscosity
                                .data
                                .iter()
                                .map(|v| v.to_f64())
                                .fold(0.0, f64::max),
                            t.diffusion_number.to_f64(),
                            t.cs_min.to_f64(),
                            t.cs_max.to_f64(),
                            t.production_max.to_f64(),
                            t.k_min.to_f64(),
                            t.k_max.to_f64(),
                            t.clamped
                        );
                    }
                }
                Err(e) => {
                    println!("[turbulence]   step {step}: refused: {e}");
                    break;
                }
            }
        }
        let peak = s
            .grid
            .u
            .iter()
            .map(|u| u.to_f64().abs())
            .fold(0.0, f64::max);
        println!(
            "[turbulence]   peak |u| after the run: {peak:.4}, state k(4,4) = {:.3e}",
            state.k(4, 4, 0).to_f64()
        );
    }

    // Refusals, every one leaving the solver and the state untouched.
    let mut s = cavity(n);
    let mut wrong_shape = RansState::new(n, n, 2, TurbulenceModel::KEpsilon);
    println!(
        "[turbulence] k-ε with a state of the wrong shape: {}",
        s.step_rans(dt, &options, &mut wrong_shape).unwrap_err()
    );
    let mut hot = RansState::uniform(
        n,
        n,
        1,
        TurbulenceModel::KEpsilon,
        Fix128::from_int(4),
        Fix128::ONE,
    );
    println!(
        "[turbulence] k-ε with nu_t = 1.44 on dx = 1/8 at dt = 1/256: {}",
        s.step_rans(dt, &options, &mut hot).unwrap_err()
    );
    println!(
        "[turbulence] step_count after the refusals: {}, k untouched: {}",
        s.step_count,
        hot.k(0, 0, 0) == Fix128::from_int(4)
    );
}
