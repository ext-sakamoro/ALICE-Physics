//! Per-material creep in `StructuralSolver`: PLA vs a non-PLA material vs
//! `with_creep`.
//!
//! A 15x20 mm cantilever with 10 N at 200 mm has `sigma = 2000 / 1000 =
//! 2 MPa`. With 1000 h steps (25 C, below both glass transitions) the
//! closed forms after 5 steps are:
//!
//! ```text
//! PLA          Norton  5 A sigma^3 dt = 5 * 6.34e-13 * 8 * 3.6e6 = 9.13e-5
//!              Findley e0 + m t^3 at t = 4000 h = 0.003 + 8.3e-14 * 6.4e10 = 8.312e-3
//!              reported max(Norton, Findley) = 8.312e-3
//! PETG         creep not modelled -> 0 exactly
//! PETG + with_creep(PLA presets) -> same as PLA
//! ```
//!
//! ```bash
//! cargo run --example structural_creep_per_material
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::creep_longterm::FindleyParameters;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::plastic::NortonCreep;
use alice_physics::structural_solver::StructuralSolver;

fn solver(material: MaterialProperties) -> StructuralSolver {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(15),
        height_mm: Fix128::from_int(20),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(10),
        length_mm: Fix128::from_int(200),
    };
    let mut s = StructuralSolver::new(section, load, material);
    s.dt_s = Fix128::from_int(3_600_000); // 1000 h
    s
}

fn last_creep(s: &mut StructuralSolver) -> Fix128 {
    let mut creep = Fix128::ZERO;
    for _ in 0..5 {
        creep = s.step().creep_strain;
    }
    creep
}

fn main() {
    let mut pla = solver(MaterialProperties::pla());
    let mut petg = solver(MaterialProperties::petg());
    let mut petg_pla_creep = solver(MaterialProperties::petg()).with_creep(
        FindleyParameters::pla_25c_moderate(),
        NortonCreep::pla_room_temp(),
    );

    let c_pla = last_creep(&mut pla);
    let c_petg = last_creep(&mut petg);
    let c_with = last_creep(&mut petg_pla_creep);

    let findley = 0.003 + 8.3e-14 * 4000f64.powi(3);
    let norton = 5.0 * 6.34e-13 * 8.0 * 3.6e6;
    let want_pla = findley.max(norton);

    println!("=== Structural creep per material (2 MPa, 5 x 1000 h) ===");
    println!(
        "PLA            modelled={} creep={:.6e} (closed form {want_pla:.6e})",
        pla.creep_modelled(),
        c_pla.to_f64()
    );
    println!(
        "PETG           modelled={} creep={:.6e} (closed form 0)",
        petg.creep_modelled(),
        c_petg.to_f64()
    );
    println!(
        "PETG+with_creep modelled={} creep={:.6e} (equals PLA)",
        petg_pla_creep.creep_modelled(),
        c_with.to_f64()
    );

    assert!(pla.creep_modelled());
    assert!(((c_pla.to_f64() - want_pla) / want_pla).abs() < 1e-6);
    assert!(!petg.creep_modelled());
    assert_eq!(c_petg, Fix128::ZERO);
    assert!(petg_pla_creep.creep_modelled());
    assert_eq!(c_with, c_pla);
    println!("all closed forms hold");
}
