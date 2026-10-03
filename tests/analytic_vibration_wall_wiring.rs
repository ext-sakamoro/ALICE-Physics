//! Oracles for the production entry points of `alice_physics::vibration_wall`
//! driven by `examples/vibration_wall_resonance.rs`: `ExcitationSource`,
//! `WallResonanceReport`, `analyze_wall_resonance`, and
//! `default_excitation_sources`.
//!
//! # What this file is and is not
//!
//! `tests/engineering_oracles_solid.rs::vibration_wall_frequency_matches_leissa_and_band_logic`
//! already drives `analyze_wall_resonance` against the Leissa/Warburton
//! plate closed form for a 1.5mm 80x120mm PLA wall, a two-source
//! near/far nearest-source selection, and the upper resonance-band
//! boundary at ratio exactly `1.2` (not risky, strict `<`).
//! `examples/vibration_wall_resonance.rs` covers `default_excitation_sources`'s
//! full curated table, a thick-ABS-wall far case, a thin-PLA-wall risky
//! case against the full default source list, an exact-resonance
//! (ratio == 1.0) synthetic source, and a single zero-frequency source.
//! Neither of those covers, and this file does:
//!
//! * `default_excitation_sources` called twice is identical and every
//!   name in the table is unique (the nearest-source report would be
//!   ambiguous between two differently-tuned sources sharing a name),
//! * `analyze_wall_resonance` against two more distinct materials (PETG,
//!   PEEK) and geometries, independently re-deriving the Warburton
//!   closed form,
//! * the nearest-source tie-break: two sources at exactly equal absolute
//!   distance from the wall frequency, in both list orders -- the loop's
//!   `if d < best_dist` (strict `<`) means the **first** occurrence in
//!   the slice wins a tie, not the last; a mutant that changed `<` to
//!   `<=` would flip which one wins without this test noticing a
//!   difference in any single-order case,
//! * the resonance-band's **lower** boundary (ratio exactly `1 - band`,
//!   not risky) at a band value (`0.3`) distinct from the existing
//!   `0.2`-band upper-boundary case, plus "just inside" points on both
//!   sides of both boundaries (risky),
//! * a synthetic exact-resonance (ratio == 1.0 exactly) source built from
//!   a different wall than the example's,
//! * the fully degenerate wall: `thickness_mm = 0` makes
//!   `plate_natural_frequency_hz` return `Fix128::ZERO` by its own
//!   documented early return (`src/modal.rs:152-154`), so
//!   `wall_frequency_hz` is exactly zero and the nearest default source
//!   is whichever is numerically smallest (`Handling / transport shock`,
//!   15 Hz) -- not tested by anything else in this repo, and
//! * the doubly-degenerate case: a zero-frequency wall against a
//!   zero-frequency-only source list, exercising
//!   `analyze_wall_resonance`'s `nearest.frequency_hz.is_zero()` guard
//!   with `f_wall` also zero (distinct from the example's zero-*source*
//!   case, where the wall frequency was non-zero).
//!
//! # Degenerate / boundary input summary
//!
//! * `analyze_wall_resonance`: `wall_thickness_mm == 0` (wall frequency
//!   itself degenerate, independent of any source), a single-source list
//!   (nearest is trivially `sources[0]`), tied nearest-source distances in
//!   both orders, and the resonance-band boundary at both ends.
//! * `default_excitation_sources`: call-to-call determinism and name
//!   uniqueness.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::vibration_wall::{
    analyze_wall_resonance, default_excitation_sources, ExcitationSource,
};
use std::f64::consts::PI;

/// Relative-error assertion against an independently hand-derived f64
/// closed form (never computed by calling the function under test).
fn assert_rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = if want == 0.0 {
        g.abs()
    } else {
        ((g - want) / want).abs()
    };
    assert!(
        err <= tol,
        "{what}: got {g}, want {want} (rel err {err:.3e} > {tol:.1e})"
    );
}

const FIX_TOL: f64 = 1e-6;

/// Warburton (1954) SSSS rectangular-plate first-mode frequency, the exact
/// closed form documented at `src/modal.rs:134-144` (same formula as
/// `examples/vibration_wall_resonance.rs`'s `plate_f64_hz`, duplicated
/// here because test binaries cannot import helpers from an example
/// binary).
fn plate_f64_hz(e_gpa: f64, rho_gcc: f64, nu: f64, h_mm: f64, a_mm: f64, b_mm: f64) -> f64 {
    let e_pa = e_gpa * 1e9;
    let rho_si = rho_gcc * 1000.0;
    let h_si = h_mm * 1e-3;
    let a_si = a_mm * 1e-3;
    let b_si = b_mm * 1e-3;
    let d = e_pa * (h_si * h_si * h_si) / (12.0 * (1.0 - nu * nu));
    PI / 2.0 * (d / (rho_si * h_si)).sqrt() * (1.0 / (a_si * a_si) + 1.0 / (b_si * b_si))
}

// ============================================================================
// Section 1: default_excitation_sources -- determinism and name uniqueness.
// ============================================================================

#[test]
fn default_excitation_sources_is_deterministic_and_has_unique_names() {
    let a = default_excitation_sources();
    let b = default_excitation_sources();
    assert_eq!(
        a, b,
        "the curated table must not vary between calls (no hidden RNG/clock)"
    );
    for i in 0..a.len() {
        for j in (i + 1)..a.len() {
            assert_ne!(
                a[i].name, a[j].name,
                "duplicate source names would make nearest-source reports ambiguous"
            );
        }
    }
}

// ============================================================================
// Section 2: analyze_wall_resonance -- two more materials/geometries,
// independently re-derived against the Warburton closed form.
// ============================================================================

/// PETG, h=2mm, 100x140mm (E=2.0 GPa, rho=1.27 g/cm3,
/// `MaterialProperties::petg()`) -- lands moderately close to, but
/// outside the +-20% band of, the X/Y stepper (120 Hz) default source.
#[test]
fn analyze_wall_resonance_petg_wall_matches_warburton_and_nearest_xy_stepper() {
    let sources = default_excitation_sources();
    let want = plate_f64_hz(2.0, 1.27, 0.35, 2.0, 100.0, 140.0);
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    assert_rel(
        report.wall_frequency_hz,
        want,
        FIX_TOL,
        "PETG wall_frequency_hz",
    );
    // By hand: at f~183.5 Hz the distances to the six defaults are 63.5
    // (X/Y, 120), 143.5 (Z screw, 40), 66.5 (fan, 117), 133.5 (HVAC, 50),
    // 216.5 (extruder, 400), 168.5 (handling, 15) -- the X/Y stepper is
    // closest, ratio 183.5/120 ~= 1.53, outside +-20%.
    assert_eq!(report.nearest_source.name, "X/Y stepper (Bambu X1C)");
    assert_rel(report.frequency_ratio, want / 120.0, FIX_TOL, "PETG ratio");
    assert!(!report.is_risky, "ratio ~1.53 is outside the +-20% band");
}

/// PEEK, h=6mm, 40x50mm (E=4.0 GPa, rho=1.32 g/cm3,
/// `MaterialProperties::peek()`) -- a thick, small, stiff wall far above
/// every default excitation source.
#[test]
fn analyze_wall_resonance_peek_wall_matches_warburton_and_is_far_from_resonance() {
    let sources = default_excitation_sources();
    let want = plate_f64_hz(4.0, 1.32, 0.35, 6.0, 40.0, 50.0);
    let report = analyze_wall_resonance(
        &MaterialProperties::peek(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(6),
        Fix128::from_int(40),
        Fix128::from_int(50),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    assert_rel(
        report.wall_frequency_hz,
        want,
        FIX_TOL,
        "PEEK wall_frequency_hz",
    );
    // By hand: at f~5182 Hz the extruder gear whine (400 Hz) is nearest
    // of the six defaults by a wide margin, ratio ~13, far outside +-20%.
    assert_eq!(report.nearest_source.name, "Extruder gear whine");
    assert_rel(report.frequency_ratio, want / 400.0, FIX_TOL, "PEEK ratio");
    assert!(!report.is_risky, "ratio ~13 is far outside the +-20% band");

    // ------------------------------------------------------------------
    // Boundary: excitation exactly at resonance (ratio == 1.0 exactly),
    // built from this wall's own already-verified `wall_frequency_hz`
    // (not re-deriving the plate formula -- this isolates
    // `analyze_wall_resonance`'s nearest-source/ratio/is_risky logic at
    // the exact-match case, independent of the example's own exact-match
    // scenario which uses a different wall).
    // ------------------------------------------------------------------
    let exact_match = [ExcitationSource {
        name: "synthetic exact match",
        frequency_hz: report.wall_frequency_hz,
    }];
    let exact_report = analyze_wall_resonance(
        &MaterialProperties::peek(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(6),
        Fix128::from_int(40),
        Fix128::from_int(50),
        &exact_match,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(
        exact_report.frequency_ratio,
        Fix128::ONE,
        "wall frequency exactly equal to the (only) source must give ratio == 1 exactly"
    );
    assert!(
        exact_report.is_risky,
        "ratio == 1.0 is strictly inside (1-band, 1+band)"
    );
}

// ============================================================================
// Section 3: nearest-source tie-break -- `if d < best_dist` (strict `<`)
// means the first occurrence in the slice wins a tie, not the last.
// ============================================================================

/// Two sources placed at exactly equal `Fix128` distance below and above
/// the wall frequency (both offsets are the same `Fix128::from_int(50)`,
/// and `Fix128` subtraction/negation/`abs` are exact bit operations, so
/// the two distances are bit-identical, not merely float-close). With
/// `below` first in the slice, it must win the tie.
#[test]
fn nearest_source_tie_break_first_occurrence_wins_below_then_above() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    let f_wall = report.wall_frequency_hz;
    let delta = Fix128::from_int(50);
    assert_eq!(
        f_wall - delta - f_wall,
        Fix128::ZERO - delta,
        "sanity: Fix128 subtraction is exact"
    );
    let tied = [
        ExcitationSource {
            name: "below",
            frequency_hz: f_wall - delta,
        },
        ExcitationSource {
            name: "above",
            frequency_hz: f_wall + delta,
        },
    ];
    let tie_report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &tied,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(
        tie_report.nearest_source.name, "below",
        "equal distances: the first slice entry must win, not the second"
    );
}

/// Same tied pair, reversed order: with `above` first in the slice, it
/// must now win -- confirming the tie-break is about list position, not
/// about the `above`/`below` names or which side of the wall frequency
/// the source sits on.
#[test]
fn nearest_source_tie_break_first_occurrence_wins_above_then_below() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    let f_wall = report.wall_frequency_hz;
    let delta = Fix128::from_int(50);
    let tied_reversed = [
        ExcitationSource {
            name: "above",
            frequency_hz: f_wall + delta,
        },
        ExcitationSource {
            name: "below",
            frequency_hz: f_wall - delta,
        },
    ];
    let tie_report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &tied_reversed,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(
        tie_report.nearest_source.name, "above",
        "equal distances: whichever entry is first in this ordering must win"
    );
}

// ============================================================================
// Section 4: resonance-band boundary -- both ends, at a band value (0.3)
// distinct from the existing 0.2/1.2-only coverage, plus "just inside"
// points on both sides.
// ============================================================================

/// Upper boundary: a synthetic source at `f_wall / 1.3` gives ratio
/// `wall/source ~= 1.3 == 1 + band` for `band = 0.3`. The strict `<` in
/// `analyze_wall_resonance` (`ratio < (ONE + resonance_band)`) must
/// report not-risky at this boundary, the same inclusive-exclusion shape
/// as the existing `1.2`/`band=0.2` case but at different numbers so it
/// is not a duplicate of that test.
#[test]
fn resonance_band_upper_boundary_exact_is_not_risky() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(30, 100),
    );
    let f_wall = report.wall_frequency_hz;
    let band = Fix128::from_ratio(30, 100);
    let at_upper = [ExcitationSource {
        name: "upper edge",
        frequency_hz: f_wall / Fix128::from_ratio(13, 10),
    }];
    let r = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &at_upper,
        band,
    );
    println!(
        "[vibration_wall test] upper boundary ratio={} is_risky={}",
        r.frequency_ratio.to_f64(),
        r.is_risky
    );
    assert!(
        !r.is_risky,
        "ratio {} at the upper band edge (1+band=1.3) must not be risky",
        r.frequency_ratio.to_f64()
    );
}

/// Lower boundary: a synthetic source at `f_wall / 0.7` gives ratio
/// `wall/source ~= 0.7 == 1 - band` for `band = 0.3`. The strict `>` in
/// `analyze_wall_resonance` (`ratio > (ONE - resonance_band)`) must
/// report not-risky at this boundary. No existing test in this repo
/// exercises the *lower* edge -- only the upper one.
#[test]
fn resonance_band_lower_boundary_exact_is_not_risky() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(30, 100),
    );
    let f_wall = report.wall_frequency_hz;
    let band = Fix128::from_ratio(30, 100);
    let at_lower = [ExcitationSource {
        name: "lower edge",
        frequency_hz: f_wall / Fix128::from_ratio(7, 10),
    }];
    let r = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &at_lower,
        band,
    );
    println!(
        "[vibration_wall test] lower boundary ratio={} is_risky={}",
        r.frequency_ratio.to_f64(),
        r.is_risky
    );
    assert!(
        !r.is_risky,
        "ratio {} at the lower band edge (1-band=0.7) must not be risky",
        r.frequency_ratio.to_f64()
    );
}

/// Just inside both edges of the same `band = 0.3` window must report
/// risky -- pinning that the boundary tests above are exercising a real
/// edge, not a window that happens to always be false.
#[test]
fn resonance_band_just_inside_either_edge_is_risky() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &sources,
        Fix128::from_ratio(30, 100),
    );
    let f_wall = report.wall_frequency_hz;
    let band = Fix128::from_ratio(30, 100);

    // Just inside the upper edge: ratio ~= 1.25 (source = f_wall/1.25).
    let just_inside_upper = [ExcitationSource {
        name: "just inside upper",
        frequency_hz: f_wall / Fix128::from_ratio(5, 4),
    }];
    let r_upper = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &just_inside_upper,
        band,
    );
    assert!(
        r_upper.is_risky,
        "ratio {} (~1.25) must be inside the 0.3 band",
        r_upper.frequency_ratio.to_f64()
    );

    // Just inside the lower edge: ratio ~= 0.75 (source = f_wall/0.75).
    let just_inside_lower = [ExcitationSource {
        name: "just inside lower",
        frequency_hz: f_wall / Fix128::from_ratio(3, 4),
    }];
    let r_lower = analyze_wall_resonance(
        &MaterialProperties::petg(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(140),
        &just_inside_lower,
        band,
    );
    assert!(
        r_lower.is_risky,
        "ratio {} (~0.75) must be inside the 0.3 band",
        r_lower.frequency_ratio.to_f64()
    );
}

// ============================================================================
// Section 5: degenerate wall (`wall_thickness_mm == 0`) and the doubly-
// degenerate (zero wall frequency *and* zero-frequency-only source) case.
// ============================================================================

/// `thickness_mm = Fix128::ZERO` makes `plate_natural_frequency_hz`
/// return exactly `Fix128::ZERO` by its own documented early return
/// (`src/modal.rs:152-154`: `thickness_mm.is_zero()` is one leg of the
/// guard). Against the full default source list, the numerically
/// smallest source (`Handling / transport shock`, 15 Hz) is nearest to
/// zero, and the ratio `0/15` must be exactly zero -- not through the
/// `nearest.frequency_hz.is_zero()` branch (15 is not zero), but through
/// ordinary division of a zero numerator.
#[test]
fn degenerate_zero_thickness_wall_frequency_is_exactly_zero() {
    let sources = default_excitation_sources();
    let report = analyze_wall_resonance(
        &MaterialProperties::pla(),
        Fix128::from_ratio(35, 100),
        Fix128::ZERO,
        Fix128::from_int(100),
        Fix128::from_int(100),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(
        report.wall_frequency_hz,
        Fix128::ZERO,
        "zero thickness must short-circuit the plate formula to exactly 0.0"
    );
    assert_eq!(
        report.nearest_source.name, "Handling / transport shock",
        "the numerically smallest default source (15 Hz) is nearest to a zero wall frequency"
    );
    assert_eq!(
        report.frequency_ratio,
        Fix128::ZERO,
        "0 / 15 must be exactly zero"
    );
    assert!(
        !report.is_risky,
        "ratio 0 is far outside the +-20% band around 1.0"
    );
}

/// Doubly degenerate: a zero-thickness wall (`f_wall == 0`) against a
/// source list containing only a zero-frequency source. This exercises
/// `analyze_wall_resonance`'s `nearest.frequency_hz.is_zero()` branch
/// directly (distinct from `examples/vibration_wall_resonance.rs`'s
/// zero-*source* case, where the wall frequency was non-zero) -- without
/// that branch, the code would otherwise compute `f_wall / nearest.frequency_hz`
/// as `0 / 0`, which happens to also be `Fix128::ZERO` by this crate's
/// documented `Div` contract (`src/math.rs`: `rhs.is_zero()` returns
/// `Self::ZERO`), so this case alone cannot distinguish the explicit
/// branch from the `Div` fallback -- it is included for completeness of
/// the degenerate-input sweep, not as the sole oracle for that branch
/// (see the single zero-frequency-source case in the example, which uses
/// a non-zero wall frequency and therefore does distinguish them).
#[test]
fn doubly_degenerate_zero_wall_and_zero_only_source_ratio_is_zero() {
    let silence = [ExcitationSource {
        name: "silence",
        frequency_hz: Fix128::ZERO,
    }];
    let report = analyze_wall_resonance(
        &MaterialProperties::pla(),
        Fix128::from_ratio(35, 100),
        Fix128::ZERO,
        Fix128::from_int(100),
        Fix128::from_int(100),
        &silence,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(report.wall_frequency_hz, Fix128::ZERO);
    assert_eq!(report.nearest_source.name, "silence");
    assert_eq!(report.frequency_ratio, Fix128::ZERO);
    assert!(!report.is_risky);
}

// ============================================================================
// Section 6: single-source list -- nearest is trivially `sources[0]`
// regardless of how far away it is.
// ============================================================================

/// A single, far-away source must still be reported as `nearest_source`
/// (there is nothing else to compare against) -- `sources[0]` is both the
/// initial value and the only candidate, so this also confirms the loop
/// over `sources.iter().skip(1)` correctly iterates zero times rather
/// than panicking or leaving an uninitialised state for a length-1 slice.
#[test]
fn single_source_list_nearest_is_trivially_the_only_entry() {
    let sources = [ExcitationSource {
        name: "only one",
        frequency_hz: Fix128::from_int(9_999),
    }];
    let report = analyze_wall_resonance(
        &MaterialProperties::pla(),
        Fix128::from_ratio(35, 100),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(100),
        &sources,
        Fix128::from_ratio(20, 100),
    );
    assert_eq!(report.nearest_source.name, "only one");
    assert_eq!(report.nearest_source.frequency_hz, Fix128::from_int(9_999));
    assert!(!report.is_risky, "a wall far below 9999 Hz is not risky");
}
