//! PLA shelf life simulation — 6 month creep + fatigue accumulation.
//!
//! Session 3 E2 demo. Models a PLA shelf under sustained bending load at
//! ambient temperature (55 °C — close to Tg to accelerate creep) and prints
//! the progressive damage / creep strain over simulated time.

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

fn main() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(150),
        height_mm: Fix128::from_int(10),
    };
    let load = LoadCase::SimplySupportedCenter {
        load_n: Fix128::from_int(20),
        length_mm: Fix128::from_int(300),
    };
    let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
    solver.operating_temp_c = Fix128::from_int(55);
    // 1 hour step size
    solver.dt_s = Fix128::from_int(3600);

    println!("=== PLA Shelf Creep + Fatigue Demo ===");
    println!("Geometry: 150x10mm cross section, 300mm span, 20N centre load");
    println!("Environment: 55 °C (approaching PLA T_g = 60 °C)");
    println!();
    println!(
        "{:>8} {:>10} {:>8} {:>10} {:>12} {:>10}",
        "hour", "σ_MPa", "FoS_c", "ε_p", "ε_creep", "D_fatigue"
    );

    for _ in 0..20 {
        let r = solver.step();
        println!(
            "{:>8.0} {:>10.2} {:>8.2} {:>10.5} {:>12.5} {:>10.4}",
            r.elapsed_hours.to_f32(),
            r.bending_stress_mpa.to_f32(),
            r.buckling_fos.to_f32().min(1e9),
            r.plastic_strain.to_f32(),
            r.creep_strain.to_f32(),
            r.fatigue_damage.to_f32()
        );
    }

    println!();
    match solver.failure_step {
        Some(s) => println!(
            "Failure detected at step {s} ({:.1} hours)",
            solver.elapsed_hours.to_f32()
        ),
        None => println!(
            "Survived {} hours without failure.",
            solver.elapsed_hours.to_f32()
        ),
    }
}
