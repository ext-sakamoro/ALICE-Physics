//! Full print-safety pipeline demo for a SKADIS-style pegboard.
//!
//! Session 3 E3 demo. Runs the integrated `print_pipeline_solver` on a
//! representative 300 × 300 × 5 mm PLA plate with a load direction, one
//! overhang region, and a fillet detail — printing the aggregated report.

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::math::Fix128;
use alice_physics::print_orientation::LoadDirection;
use alice_physics::print_pipeline_solver::{analyze_print_pipeline, PrintPipelineInputs};
use alice_physics::support_volume::OverhangRegion;
use alice_physics::warp_risk::Footprint;

fn main() {
    println!("=== Print Full Safety Pipeline Demo ===");
    println!("Model: 300x300x5mm PLA SKADIS-style pegboard");
    println!();

    let footprint = Footprint {
        area_mm2: Fix128::from_int(300 * 300),
        max_dimension_mm: Fix128::from_int(300),
    };
    let inputs = PrintPipelineInputs {
        beam_load: Some((
            CrossSection::Rectangular {
                width_mm: Fix128::from_int(50),
                height_mm: Fix128::from_int(5),
            },
            LoadCase::CantileverEndPoint {
                load_n: Fix128::from_int(30),
                length_mm: Fix128::from_int(300),
            },
        )),
        load_direction: Some(LoadDirection::axis_z()),
        overhangs: vec![OverhangRegion {
            projected_area_mm2: Fix128::from_int(50 * 50),
            support_height_mm: Fix128::from_int(15),
        }],
        fillet: Some((
            Fix128::from_ratio(3, 10),
            Fix128::from_int(20),
            Fix128::from_int(40),
        )),
        ..Default::default()
    };

    let report = analyze_print_pipeline(footprint, "PLA", &inputs);
    report.print();
}
