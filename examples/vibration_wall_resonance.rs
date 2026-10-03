//! Production entry point for `alice_physics::vibration_wall`:
//! `ExcitationSource`, `WallResonanceReport`, `analyze_wall_resonance`, and
//! `default_excitation_sources`.
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all four as `unwired`.
//! `tests/engineering_oracles_solid.rs::vibration_wall_frequency_matches_leissa_and_band_logic`
//! already drives `analyze_wall_resonance` / `ExcitationSource` against the
//! Leissa plate closed form (the same formula as `src/modal.rs`'s
//! `plate_natural_frequency_hz`) and the exact-ratio nearest-source and
//! band logic, but tests do not count as production callers for the
//! wiring guard (src / examples / benches / fuzz / bindings only). Nothing
//! in `src/` / `examples/` / `benches/` called any of the four before this
//! file existed. This example is that caller.
//!
//! `src/modal.rs`'s own module doc (and `examples/modal_frequency_analysis.rs`'s
//! wiring comment) notes that `plate_natural_frequency_hz` is also called
//! from `src/vibration_wall.rs::analyze_wall_resonance`, but that call did
//! not count toward `plate_natural_frequency_hz`'s own wiring because
//! `analyze_wall_resonance` itself had zero production callers
//! (transitively unwired) -- this example closes that chain from the
//! other end: `analyze_wall_resonance` is now driven for real, which means
//! `plate_natural_frequency_hz` is reached through it as well as directly
//! from `examples/modal_frequency_analysis.rs`.
//!
//! `tests/analytic_vibration_wall_wiring.rs` holds additional closed-form
//! and boundary oracles not covered here or in the existing oracle file:
//! every individual `default_excitation_sources()` entry pinned by exact
//! value, the `resonance_band` boundary itself (ratio exactly `1 - band`
//! and exactly `1 + band`, both excluded by the strict `<`/`>` in
//! `analyze_wall_resonance`), ties in nearest-source selection, and a
//! single-source list.
//!
//! # Data flow (per the module doc at `src/vibration_wall.rs:1-31`)
//!
//! `default_excitation_sources()` curates known printer/environmental
//! frequencies (Hz) -> `analyze_wall_resonance` computes the wall's own
//! first natural frequency via `modal::plate_natural_frequency_hz`
//! (Warburton 1954 SSSS plate, `f_11 = (pi/2) sqrt(D/(rho h)) (1/a^2 +
//! 1/b^2)`, `D = E h^3 / (12 (1-nu^2))`) -> finds the nearest source by
//! absolute frequency difference -> reports the `wall/source` ratio and
//! whether it falls within `resonance_band` of 1.0 (default +-20%).
//!
//! Every expected value below is derived independently from the plate
//! formula in plain `f64` (same derivation as
//! `examples/modal_frequency_analysis.rs`, duplicated here because
//! examples cannot import helpers from each other), or from the
//! already-independently-verified wall frequency of an earlier scenario in
//! this same file (never by calling `analyze_wall_resonance` and then
//! comparing its own output to itself).
//!
//! ```bash
//! cargo run --example vibration_wall_resonance --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::vibration_wall::{
    analyze_wall_resonance, default_excitation_sources, ExcitationSource, WallResonanceReport,
};
use std::f64::consts::PI;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn check(name: &str, actual: f64, expected: f64, tol: f64) {
    let err = rel_err(actual, expected);
    println!("[vibration_wall] {name}: actual={actual} expected={expected} rel_err={err:e}");
    assert!(
        err < tol,
        "{name}: actual={actual} expected={expected} rel_err={err} (tol {tol})"
    );
}

/// Warburton (1954) SSSS rectangular-plate first-mode frequency, the exact
/// closed form documented at `src/modal.rs:134-144` and re-derived
/// independently in plain `f64` (never by calling
/// `plate_natural_frequency_hz` itself).
fn plate_f64_hz(e_gpa: f64, rho_gcc: f64, nu: f64, h_mm: f64, a_mm: f64, b_mm: f64) -> f64 {
    let e_pa = e_gpa * 1e9;
    let rho_si = rho_gcc * 1000.0;
    let h_si = h_mm * 1e-3;
    let a_si = a_mm * 1e-3;
    let b_si = b_mm * 1e-3;
    let d = e_pa * (h_si * h_si * h_si) / (12.0 * (1.0 - nu * nu));
    PI / 2.0 * (d / (rho_si * h_si)).sqrt() * (1.0 / (a_si * a_si) + 1.0 / (b_si * b_si))
}

fn main() {
    // -----------------------------------------------------------------
    // 1. default_excitation_sources(): the curated list is a fixed table
    //    (src/vibration_wall.rs:10-19 doc comment) -- every entry's name
    //    and nominal frequency is pinned exactly, by `Fix128::from_int`
    //    equality (all six nominal frequencies are whole-Hz integers, so
    //    there is no rounding tolerance to pick).
    // -----------------------------------------------------------------
    let sources = default_excitation_sources();
    println!(
        "[vibration_wall] default_excitation_sources(): {} entries",
        sources.len()
    );
    assert_eq!(sources.len(), 6, "curated table has exactly 6 entries");
    let expected_table: [(&str, i64); 6] = [
        ("X/Y stepper (Bambu X1C)", 120),
        ("Z lead-screw stepper", 40),
        ("Cooling fan 7000rpm", 117),
        ("Chamber HVAC 50Hz", 50),
        ("Extruder gear whine", 400),
        ("Handling / transport shock", 15),
    ];
    for (i, (name, hz)) in expected_table.iter().enumerate() {
        println!(
            "[vibration_wall] source[{i}]: name={} frequency_hz={}",
            sources[i].name,
            sources[i].frequency_hz.to_f64()
        );
        assert_eq!(sources[i].name, *name, "source[{i}] name");
        assert_eq!(
            sources[i].frequency_hz,
            Fix128::from_int(*hz),
            "source[{i}] frequency_hz"
        );
    }

    // -----------------------------------------------------------------
    // 2. analyze_wall_resonance, scenario A: a thick, small ABS wall is
    //    far above every default excitation source -- not risky. h=4mm,
    //    60x90mm ABS (E=2.3 GPa, rho=1.04 g/cm3, src/filament_db.rs
    //    MaterialProperties::abs()), nu=0.35.
    // -----------------------------------------------------------------
    let (e_abs, rho_abs) = (2.3_f64, 1.04_f64);
    let want_a = plate_f64_hz(e_abs, rho_abs, 0.35, 4.0, 60.0, 90.0);
    let report_a: WallResonanceReport = analyze_wall_resonance(
        &MaterialProperties::abs(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(4),
        Fix128::from_int(60),
        Fix128::from_int(90),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    check(
        "scenario A wall_frequency_hz (4mm ABS, 60x90)",
        report_a.wall_frequency_hz.to_f64(),
        want_a,
        1e-6,
    );
    // Nearest source by hand: distances to all six defaults at f~1155 Hz
    // are 1035.3 (X/Y), 1115.3 (Z screw), 1038.3 (fan), 1105.3 (HVAC),
    // 755.3 (extruder), 1140.3 (handling) -- the extruder gear whine
    // (400 Hz) is closest, and the ratio 1155/400 ~= 2.89 is nowhere near
    // the +-20% band.
    println!(
        "[vibration_wall] scenario A nearest_source={} ratio={} is_risky={}",
        report_a.nearest_source.name,
        report_a.frequency_ratio.to_f64(),
        report_a.is_risky
    );
    assert_eq!(report_a.nearest_source.name, "Extruder gear whine");
    check(
        "scenario A frequency_ratio",
        report_a.frequency_ratio.to_f64(),
        want_a / 400.0,
        1e-6,
    );
    assert!(
        !report_a.is_risky,
        "ratio ~2.89 is far outside the +-20% band"
    );

    // -----------------------------------------------------------------
    // 3. analyze_wall_resonance, scenario B: a thin, large PLA wall lands
    //    close to the Z lead-screw stepper (40 Hz) -- risky. h=1mm,
    //    180x220mm PLA (E=3.5 GPa, rho=1.24 g/cm3,
    //    MaterialProperties::pla()), nu=0.35.
    // -----------------------------------------------------------------
    let (e_pla, rho_pla) = (3.5_f64, 1.24_f64);
    let want_b = plate_f64_hz(e_pla, rho_pla, 0.35, 1.0, 180.0, 220.0);
    let report_b: WallResonanceReport = analyze_wall_resonance(
        &MaterialProperties::pla(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(1),
        Fix128::from_int(180),
        Fix128::from_int(220),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    check(
        "scenario B wall_frequency_hz (1mm PLA, 180x220)",
        report_b.wall_frequency_hz.to_f64(),
        want_b,
        1e-6,
    );
    // Nearest source by hand: at f~41.9 Hz the distances are 78.1 (X/Y),
    // 1.9 (Z screw), 75.1 (fan), 8.1 (HVAC), 358.1 (extruder), 26.9
    // (handling) -- the Z lead-screw stepper (40 Hz) is closest, and
    // the ratio 41.9/40 ~= 1.048 is inside the +-20% band.
    println!(
        "[vibration_wall] scenario B nearest_source={} ratio={} is_risky={}",
        report_b.nearest_source.name,
        report_b.frequency_ratio.to_f64(),
        report_b.is_risky
    );
    assert_eq!(report_b.nearest_source.name, "Z lead-screw stepper");
    check(
        "scenario B frequency_ratio",
        report_b.frequency_ratio.to_f64(),
        want_b / 40.0,
        1e-6,
    );
    assert!(
        report_b.is_risky,
        "ratio ~1.048 is inside the +-20% band -- this wall is at resonance risk"
    );

    // -----------------------------------------------------------------
    // 4. Boundary: excitation exactly at resonance (ratio == 1.0 exactly).
    //    Built from scenario B's own `wall_frequency_hz`, which is
    //    already independently verified above against the Leissa/
    //    Warburton closed form -- this section is not re-deriving the
    //    plate formula, it is checking analyze_wall_resonance's
    //    nearest-source / ratio / is_risky logic at the exact-match case,
    //    the same way examples/laminate_abd_matrix.rs section 5 builds an
    //    `AbdMatrix` directly to isolate `is_symmetric`'s boundary from
    //    `compute_abd`'s arithmetic.
    // -----------------------------------------------------------------
    let synthetic_exact_match = [ExcitationSource {
        name: "synthetic exact match",
        frequency_hz: report_b.wall_frequency_hz,
    }];
    let report_exact = analyze_wall_resonance(
        &MaterialProperties::pla(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(1),
        Fix128::from_int(180),
        Fix128::from_int(220),
        &synthetic_exact_match,
        Fix128::from_ratio(20, 100),
    );
    println!(
        "[vibration_wall] exact-resonance ratio={} is_risky={}",
        report_exact.frequency_ratio.to_f64(),
        report_exact.is_risky
    );
    assert_eq!(
        report_exact.frequency_ratio,
        Fix128::ONE,
        "wall frequency exactly equal to the (only) source must give ratio == 1 exactly"
    );
    assert!(
        report_exact.is_risky,
        "ratio == 1.0 is strictly inside (1-band, 1+band)"
    );

    // -----------------------------------------------------------------
    // 5. Boundary: a zero-frequency excitation source. `analyze_wall_resonance`
    //    special-cases `nearest.frequency_hz.is_zero()` to report
    //    `frequency_ratio = Fix128::ZERO` rather than evaluating
    //    `f_wall / 0`. With only one (zero-frequency) source in the list,
    //    that source is trivially nearest by construction.
    // -----------------------------------------------------------------
    let silence = [ExcitationSource {
        name: "silence",
        frequency_hz: Fix128::ZERO,
    }];
    let report_zero_source = analyze_wall_resonance(
        &MaterialProperties::abs(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(4),
        Fix128::from_int(60),
        Fix128::from_int(90),
        &silence,
        Fix128::from_ratio(20, 100),
    );
    println!(
        "[vibration_wall] zero-frequency source: nearest={} ratio={} is_risky={}",
        report_zero_source.nearest_source.name,
        report_zero_source.frequency_ratio.to_f64(),
        report_zero_source.is_risky
    );
    assert_eq!(report_zero_source.nearest_source.name, "silence");
    assert_eq!(
        report_zero_source.frequency_ratio,
        Fix128::ZERO,
        "a zero-frequency nearest source must report ratio == 0 exactly, not f_wall/0"
    );
    assert!(
        !report_zero_source.is_risky,
        "ratio 0 is outside the +-20% band around 1.0"
    );

    println!(
        "[vibration_wall] all 4 production entry points (ExcitationSource, \
         WallResonanceReport, analyze_wall_resonance, default_excitation_sources) \
         verified against hand-derived closed forms and the module's documented \
         boundary behaviour"
    );
}
