//! Independent oracles for `bridging`: `BridgeSpan::z_delta_mm`,
//! `BridgingReport::unsafe_checks`, and the print-pipeline report that
//! consumes them, plus the module doc's material table and boundary
//! behaviour of the safety ratio.
//!
//! Closed forms (module doc):
//!
//! ```text
//! length      = sqrt(dx^2 + dy^2 + dz^2)
//! z_delta     = |z_end - z_start|
//! safety      = allowable / length          (length = 0 -> "infinite",
//!                                            allowable = 0 -> 0)
//! is_safe     = safety >= 1                 (a span exactly at the limit is safe)
//! allowables  = PLA 20, PETG 15, ABS 12, PC 15, TPU 5, Nylon 10,
//!               CF-Nylon 20, PEEK 25  (mm, the table in the module doc)
//! ```
//!
//! Nothing here touches `src/`.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::bridging::{analyze_bridges, check_span, BridgeSpan, BridgingReport};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::{Fix128, Vec3Fix};
use alice_physics::print_pipeline_solver::{analyze_print_pipeline, PrintPipelineInputs};
use alice_physics::warp_risk::Footprint;

fn p(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::new(
        Fix128::from_int(x),
        Fix128::from_int(y),
        Fix128::from_int(z),
    )
}

fn span(a: (i64, i64, i64), b: (i64, i64, i64)) -> BridgeSpan {
    BridgeSpan {
        start: p(a.0, a.1, a.2),
        end: p(b.0, b.1, b.2),
    }
}

fn close(a: Fix128, b: f64, tol: f64) -> bool {
    (a.to_f64() - b).abs() <= tol
}

fn footprint() -> Footprint {
    Footprint {
        area_mm2: Fix128::from_int(1000),
        max_dimension_mm: Fix128::from_int(50),
    }
}

// ---------------------------------------------------------------------------
// z_delta_mm
// ---------------------------------------------------------------------------

/// `|z_end - z_start|`: symmetric in the endpoints, independent of x and y,
/// exactly zero for a horizontal bridge.
#[test]
fn z_delta_is_the_absolute_height_difference_only() {
    assert_eq!(span((0, 0, 5), (15, 0, 5)).z_delta_mm(), Fix128::ZERO);
    assert_eq!(
        span((0, 0, 2), (99, -7, 9)).z_delta_mm(),
        Fix128::from_int(7)
    );
    // reversed endpoints: same magnitude
    assert_eq!(
        span((99, -7, 9), (0, 0, 2)).z_delta_mm(),
        Fix128::from_int(7)
    );
    // negative z handled: |-3 - 4| = 7
    assert_eq!(
        span((0, 0, -3), (1, 1, 4)).z_delta_mm(),
        Fix128::from_int(7)
    );
    // x / y never leak in
    assert_eq!(
        span((1000, 2000, 1), (-1000, -2000, 1)).z_delta_mm(),
        Fix128::ZERO
    );
    // fractional heights: |1/4 - (-1/8)| = 3/8
    let s = BridgeSpan {
        start: Vec3Fix::new(Fix128::ZERO, Fix128::ZERO, Fix128::from_ratio(1, 4)),
        end: Vec3Fix::new(Fix128::ONE, Fix128::ZERO, Fix128::from_ratio(-1, 8)),
    };
    assert_eq!(s.z_delta_mm(), Fix128::from_ratio(3, 8));
}

/// The 3-D length bounds the height difference from above: `z_delta <=
/// length`, with equality only for a vertical span. (Pythagoras, with the
/// horizontal run `sqrt(length^2 - z_delta^2)` recovered independently.)
#[test]
fn z_delta_and_length_satisfy_pythagoras() {
    // 2-3-6 box diagonal: sqrt(4 + 9 + 36) = 7, z_delta = 6
    let s = span((0, 0, 0), (2, 3, 6));
    assert!(close(s.length_mm(), 7.0, 1e-9));
    assert_eq!(s.z_delta_mm(), Fix128::from_int(6));
    let run2 = 7.0f64 * 7.0 - 6.0 * 6.0;
    assert!(
        (run2 - 13.0).abs() < 1e-12,
        "horizontal run^2 = dx^2 + dy^2 = 13"
    );
    // vertical span: z_delta equals the length
    let v = span((1, 1, 0), (1, 1, 9));
    assert_eq!(v.z_delta_mm(), Fix128::from_int(9));
    assert!(close(v.length_mm(), 9.0, 1e-9));
}

// ---------------------------------------------------------------------------
// check_span / analyze_bridges
// ---------------------------------------------------------------------------

/// The module doc's material table, one row at a time: a span exactly at the
/// limit is safe with ratio 1, one millimetre over it is unsafe with ratio
/// `limit / (limit + 1)`.
#[test]
fn doc_material_table_limits_are_the_safety_boundary() {
    let table: [(&str, MaterialProperties, i64); 8] = [
        ("PLA", MaterialProperties::pla(), 20),
        ("PETG", MaterialProperties::petg(), 15),
        ("ABS", MaterialProperties::abs(), 12),
        ("PC", MaterialProperties::pc(), 15),
        ("TPU", MaterialProperties::tpu(), 5),
        ("Nylon", MaterialProperties::nylon(), 10),
        ("CF-Nylon", MaterialProperties::cf_nylon(), 20),
        ("PEEK", MaterialProperties::peek(), 25),
    ];
    for (name, m, limit) in table {
        let at = check_span(&span((0, 0, 0), (limit, 0, 0)), &m);
        assert_eq!(at.allowable_mm, Fix128::from_int(limit), "{name}");
        assert!(at.is_safe, "{name}: exactly at the limit is safe");
        assert_eq!(at.safety_ratio, Fix128::ONE, "{name}: ratio 1 at the limit");

        let over = check_span(&span((0, 0, 0), (limit + 1, 0, 0)), &m);
        assert!(!over.is_safe, "{name}: 1 mm over is unsafe");
        assert!(
            close(over.safety_ratio, limit as f64 / (limit + 1) as f64, 1e-9),
            "{name}: ratio {} vs {}/{}",
            over.safety_ratio.to_f64(),
            limit,
            limit + 1
        );
    }
}

/// A sloped span is measured by its 3-D length: the 3-4-5 run with a 12 mm
/// rise has length 13 (5-12-13), so against PLA's 20 mm the ratio is 20/13.
#[test]
fn span_length_is_three_dimensional_and_ratio_is_allowable_over_length() {
    let s = span((0, 0, 0), (3, 4, 12));
    let c = check_span(&s, &MaterialProperties::pla());
    assert!(close(c.length_mm, 13.0, 1e-9));
    assert_eq!(c.allowable_mm, Fix128::from_int(20));
    assert!(close(c.safety_ratio, 20.0 / 13.0, 1e-9));
    assert!(c.is_safe);
    assert_eq!(c.span, s);
}

/// Degenerate inputs: a zero-length span is trivially safe (sentinel
/// `i64::MAX >> 8`), a material with no bridging allowance (limit 0) makes any
/// positive span unsafe with ratio 0, and a zero-length span on a zero-limit
/// material is still safe (nothing to bridge).
#[test]
fn degenerate_spans_and_zero_allowance() {
    let sentinel = Fix128::from_int(i64::MAX >> 8);
    let point = span((4, 4, 4), (4, 4, 4));
    let c = check_span(&point, &MaterialProperties::pla());
    assert_eq!(c.length_mm, Fix128::ZERO);
    assert_eq!(c.safety_ratio, sentinel);
    assert!(c.is_safe);

    let mut none = MaterialProperties::pla();
    none.bridging_distance_mm = Fix128::ZERO;
    let c = check_span(&span((0, 0, 0), (1, 0, 0)), &none);
    assert_eq!(c.safety_ratio, Fix128::ZERO);
    assert!(!c.is_safe);
    let c = check_span(&point, &none);
    assert_eq!(c.safety_ratio, sentinel);
    assert!(c.is_safe);
}

/// Aggregate report: counts only the unsafe spans, reports the longest span
/// seen (safe or not), keeps one check per input in input order, and an empty
/// input is an empty, safe report.
#[test]
fn analyze_bridges_aggregates_counts_max_and_order() {
    let pla = MaterialProperties::pla();
    let spans = [
        span((0, 0, 0), (10, 0, 0)), // 10  safe
        span((0, 0, 0), (0, 25, 0)), // 25  unsafe
        span((0, 0, 0), (20, 0, 0)), // 20  safe (at the limit)
        span((0, 0, 0), (0, 0, 21)), // 21  unsafe (vertical)
        span((0, 0, 0), (6, 8, 0)),  // 10  safe
    ];
    let r = analyze_bridges(&spans, &pla);
    assert_eq!(r.checks.len(), 5);
    assert_eq!(r.unsafe_count, 2);
    assert!(r.has_unsafe());
    assert_eq!(r.max_length_mm, Fix128::from_int(25));
    let flags: Vec<bool> = r.checks.iter().map(|c| c.is_safe).collect();
    assert_eq!(flags, [true, false, true, false, true]);
    for (c, s) in r.checks.iter().zip(spans.iter()) {
        assert_eq!(&c.span, s, "input order preserved");
    }

    let empty = analyze_bridges(&[], &pla);
    assert!(empty.checks.is_empty());
    assert_eq!(empty.unsafe_count, 0);
    assert!(!empty.has_unsafe());
    assert_eq!(empty.max_length_mm, Fix128::ZERO);
    let default = BridgingReport::default();
    assert_eq!(default.unsafe_count, 0);
    assert_eq!(default.unsafe_checks().count(), 0);
}

// ---------------------------------------------------------------------------
// unsafe_checks
// ---------------------------------------------------------------------------

/// `unsafe_checks` yields exactly the checks with `is_safe == false`, in input
/// order, and its length equals `unsafe_count` (the two independent tallies
/// must agree).
#[test]
fn unsafe_checks_yields_exactly_the_unsafe_spans_in_order() {
    let pla = MaterialProperties::pla();
    let spans = [
        span((0, 0, 0), (30, 0, 0)), // unsafe
        span((0, 0, 0), (5, 0, 0)),  // safe
        span((0, 0, 0), (0, 40, 0)), // unsafe
        span((0, 0, 0), (20, 0, 0)), // safe
        span((0, 0, 0), (0, 0, 26)), // unsafe
    ];
    let r = analyze_bridges(&spans, &pla);
    let bad: Vec<&_> = r.unsafe_checks().collect();
    assert_eq!(bad.len(), r.unsafe_count);
    assert_eq!(bad.len(), 3);
    assert!(bad.iter().all(|c| !c.is_safe));
    assert_eq!(bad[0].span, spans[0]);
    assert_eq!(bad[1].span, spans[2]);
    assert_eq!(bad[2].span, spans[4]);

    // all safe -> empty iterator
    let ok = analyze_bridges(&[spans[1], spans[3]], &pla);
    assert_eq!(ok.unsafe_checks().count(), 0);
    // all unsafe -> every check
    let all_bad = analyze_bridges(&[spans[0], spans[2]], &pla);
    assert_eq!(all_bad.unsafe_checks().count(), 2);
}

// ---------------------------------------------------------------------------
// print pipeline: the production consumer
// ---------------------------------------------------------------------------

/// The pipeline itemises each offending span. Each detail line carries the
/// span's 3-D length, the material limit, and the height difference, so a
/// reader can tell a long flat bridge from a steep short one.
#[test]
fn pipeline_itemises_unsafe_spans_with_length_limit_and_height_difference() {
    let inputs = PrintPipelineInputs {
        bridges: vec![
            span((0, 0, 0), (10, 0, 0)), // safe
            span((0, 0, 0), (30, 0, 0)), // 30 mm flat, unsafe
            span((0, 0, 0), (0, 0, 24)), // 24 mm vertical (dz 24), unsafe
        ],
        ..Default::default()
    };
    let r = analyze_print_pipeline(footprint(), "PLA", &inputs);
    assert!(!r.is_safe);
    let br = r.bridging.as_ref().expect("bridging report present");
    assert_eq!(br.unsafe_count, 2);

    // the summary line is unchanged
    assert!(
        r.messages
            .iter()
            .any(|m| m == "2 bridge spans exceed material limit"),
        "{:?}",
        r.messages
    );
    // one detail line per unsafe span, none for the safe one
    let details: Vec<&String> = r
        .messages
        .iter()
        .filter(|m| m.starts_with("Bridge span"))
        .collect();
    assert_eq!(details.len(), 2, "{:?}", r.messages);
    assert_eq!(
        details[0],
        "Bridge span 30.00 mm exceeds 20.00 mm limit (dz 0.00 mm)"
    );
    assert_eq!(
        details[1],
        "Bridge span 24.00 mm exceeds 20.00 mm limit (dz 24.00 mm)"
    );
}

/// No unsafe span, no detail lines; no bridges at all, no report.
#[test]
fn pipeline_is_quiet_when_every_span_is_within_the_limit() {
    let inputs = PrintPipelineInputs {
        bridges: vec![span((0, 0, 0), (20, 0, 0)), span((0, 0, 0), (3, 4, 0))],
        ..Default::default()
    };
    let r = analyze_print_pipeline(footprint(), "PLA", &inputs);
    assert!(r.is_safe);
    assert!(
        r.messages.iter().all(|m| !m.contains("ridge")),
        "{:?}",
        r.messages
    );
    assert_eq!(r.bridging.as_ref().unwrap().unsafe_count, 0);

    let none = analyze_print_pipeline(footprint(), "PLA", &PrintPipelineInputs::default());
    assert!(none.bridging.is_none());
}

/// The material passed to the pipeline sets the limit: the same 18 mm span is
/// fine on PLA (20) and flagged on ABS (12) and TPU (5), each with its own
/// limit in the detail line.
#[test]
fn pipeline_uses_the_selected_materials_limit() {
    let inputs = PrintPipelineInputs {
        bridges: vec![span((0, 0, 0), (18, 0, 0))],
        ..Default::default()
    };
    let pla = analyze_print_pipeline(footprint(), "PLA", &inputs);
    assert_eq!(pla.bridging.as_ref().unwrap().unsafe_count, 0);
    for (name, limit) in [("ABS", "12.00"), ("TPU", "5.00")] {
        let r = analyze_print_pipeline(footprint(), name, &inputs);
        assert_eq!(r.bridging.as_ref().unwrap().unsafe_count, 1, "{name}");
        let want = format!("Bridge span 18.00 mm exceeds {limit} mm limit (dz 0.00 mm)");
        assert!(r.messages.contains(&want), "{name}: {:?}", r.messages);
    }
}
