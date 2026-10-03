//! Bridge-span check through `bridging::analyze_bridges` and the print
//! pipeline that reports its unsafe spans.
//!
//! Closed forms (module doc of `bridging`): length `sqrt(dx^2 + dy^2 + dz^2)`,
//! height difference `|dz|`, safety ratio `allowable / length`, safe when the
//! ratio is at least 1. For PLA (20 mm) the spans below give:
//!
//! ```text
//! (0,0,0) -> (10,0,0)   length 10   ratio 2.000   safe,   dz 0
//! (0,0,0) -> (30,0,0)   length 30   ratio 0.667   UNSAFE, dz 0
//! (0,0,0) -> (3,4,12)   length 13   ratio 1.538   safe,   dz 12
//! (0,0,0) -> (0,0,24)   length 24   ratio 0.833   UNSAFE, dz 24
//! ```
//!
//! ```bash
//! cargo run --example bridging_span_check --features std
//! ```

use alice_physics::bridging::{analyze_bridges, BridgeSpan};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::print_pipeline_solver::{analyze_print_pipeline, PrintPipelineInputs};
use alice_physics::warp_risk::Footprint;

fn span(end: (i64, i64, i64)) -> BridgeSpan {
    BridgeSpan {
        start: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ZERO),
        end: Vec3Fix::new(
            Fix128::from_int(end.0),
            Fix128::from_int(end.1),
            Fix128::from_int(end.2),
        ),
    }
}

fn main() {
    let spans = [
        span((10, 0, 0)),
        span((30, 0, 0)),
        span((3, 4, 12)),
        span((0, 0, 24)),
    ];
    let expect = [
        (10.0, 2.0, true, 0.0),
        (30.0, 20.0 / 30.0, false, 0.0),
        (13.0, 20.0 / 13.0, true, 12.0),
        (24.0, 20.0 / 24.0, false, 24.0),
    ];

    let report = analyze_bridges(&spans, &MaterialProperties::pla());
    for (c, (len, ratio, safe, dz)) in report.checks.iter().zip(expect) {
        println!(
            "length {:>6.3}  ratio {:>6.3}  {}  dz {:>5.2}",
            c.length_mm.to_f64(),
            c.safety_ratio.to_f64(),
            if c.is_safe { "safe  " } else { "UNSAFE" },
            c.span.z_delta_mm().to_f64()
        );
        assert!((c.length_mm.to_f64() - len).abs() < 1e-6);
        assert!((c.safety_ratio.to_f64() - ratio).abs() < 1e-6);
        assert_eq!(c.is_safe, safe);
        assert!((c.span.z_delta_mm().to_f64() - dz).abs() < 1e-9);
    }
    assert_eq!(report.unsafe_count, 2);
    assert_eq!(report.unsafe_checks().count(), 2);

    let inputs = PrintPipelineInputs {
        bridges: spans.to_vec(),
        ..Default::default()
    };
    let footprint = Footprint {
        area_mm2: Fix128::from_int(1000),
        max_dimension_mm: Fix128::from_int(50),
    };
    let r = analyze_print_pipeline(footprint, "PLA", &inputs);
    for m in &r.messages {
        println!("pipeline: {m}");
    }
    assert!(!r.is_safe);
}
