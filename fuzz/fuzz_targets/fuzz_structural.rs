#![no_main]
use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct StructuralInput {
    /// Rectangular section width (mm), constrained to 1..=100.
    width_mm_raw: u8,
    /// Rectangular section height (mm), constrained to 1..=100.
    height_mm_raw: u8,
    /// Cantilever end-point load (N), constrained to 1..=100 (signed to
    /// exercise both tension/compression side effects at scale).
    load_n_raw: i8,
    /// Beam length (mm), constrained to 10..=1000.
    length_mm_raw: u8,
    /// Number of solver steps (capped at 8 to bound each iteration).
    step_count: u8,
}

// Fuzz the integrated structural solver:
// - `StructuralSolver::new` with a rectangular cross-section, cantilever
//   end-point load, and PLA material defaults
// - `StructuralSolver::step` up to 8 times
//
// Must never panic across arbitrary section / load combinations. Extreme
// aspect ratios, near-zero geometry, and heavy loads are the common
// failure signatures for Roark/Euler-style formulas — the solver must
// remain numerically robust and report the failure via
// `StructuralReport.failure_step` rather than crashing.
fuzz_target!(|input: StructuralInput| {
    let width_mm = ((input.width_mm_raw as i64) % 100).max(1);
    let height_mm = ((input.height_mm_raw as i64) % 100).max(1);
    let load_n = ((input.load_n_raw as i64) % 100).max(-100).min(100);
    let length_mm = ((input.length_mm_raw as i64) % 991) + 10; // 10..=1000

    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(width_mm),
        height_mm: Fix128::from_int(height_mm),
    };
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::from_int(load_n),
        length_mm: Fix128::from_int(length_mm),
    };

    let mut solver = StructuralSolver::new(section, load, MaterialProperties::pla());
    let steps = (input.step_count as usize).min(8);
    for _ in 0..steps {
        let _ = solver.step();
    }
});
