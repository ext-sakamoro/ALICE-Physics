//! `StructuralSolver::run` and `StructuralHistory`.
//!
//! A 15x20 mm PLA cantilever with 100 N at 200 mm has `sigma = 20000 / 1000 =
//! 20 MPa` (FoS 2.5, elastic, above the 18 MPa fatigue endurance of PLA). With
//! 2-hour steps over 10 steps the closed forms are:
//!
//! ```text
//! elapsed        10 x 2 h = 20 h
//! Norton creep   n A sigma^3 dt = 10 * 6.34e-13 * 8000 * 7200 = 3.652e-4
//! Miner damage   n / N,  N = 1e6 (18/20)^5 = 590490  ->  1.693e-5
//! plastic strain 0, no failure
//! ```
//!
//! ```bash
//! cargo run --example structural_run_history --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

fn main() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(15),
        height_mm: Fix128::from_int(20),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(100),
        length_mm: Fix128::from_int(200),
    };
    let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
    solver.dt_s = Fix128::from_int(7200);
    let h = solver.run(10);

    let norton = 10.0 * 6.34e-13 * 8000.0 * 7200.0;
    let miner = 10.0 / 590_490.0;
    println!(
        "steps {}  elapsed {} h  failure {:?}",
        h.steps,
        h.elapsed_hours.to_f64(),
        h.failure_step
    );
    println!(
        "creep {:.4e} (closed form {norton:.4e})",
        h.plastic_state.creep_strain.to_f64()
    );
    println!(
        "fatigue D {:.4e} (closed form {miner:.4e})",
        h.fatigue_damage.to_f64()
    );
    assert_eq!(h.steps, 10);
    assert_eq!(h.elapsed_hours, Fix128::from_int(20));
    assert_eq!(h.failure_step, None);
    assert!((h.plastic_state.creep_strain.to_f64() - norton).abs() / norton < 2e-7);
    assert!((h.fatigue_damage.to_f64() - miner).abs() / miner < 2e-5);
    assert_eq!(h.plastic_state.equivalent_plastic_strain, Fix128::ZERO);
}
