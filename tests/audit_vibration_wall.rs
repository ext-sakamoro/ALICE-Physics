//! Audit oracles (S2-1) for `alice_physics::vibration_wall`.
//!
//! Expected values come from the module doc's stated criterion ("within
//! +-20 % of a significant excitation source"), a brute-force scan over the
//! source list, and the doc table of source frequency ranges -- never from
//! the function under test. Tests marked `known defect` are `#[ignore]`d
//! and recorded in the audit ledger; they are not fixed here.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::vibration_wall::{
    analyze_wall_resonance, default_excitation_sources, ExcitationSource,
};

fn nu() -> Fix128 {
    Fix128::from_ratio(35, 100)
}

fn wall(
    t: i64,
    a: i64,
    b: i64,
    sources: &[ExcitationSource],
    band: Fix128,
) -> alice_physics::vibration_wall::WallResonanceReport {
    analyze_wall_resonance(
        &MaterialProperties::pla(),
        nu(),
        Fix128::from_int(t),
        Fix128::from_int(a),
        Fix128::from_int(b),
        sources,
        band,
    )
}

fn src(name: &'static str, f: Fix128) -> ExcitationSource {
    ExcitationSource {
        name,
        frequency_hz: f,
    }
}

/// Doc table: X/Y 60-200, Z 20-80, fan 80-130, HVAC 50-60, extruder 300-800,
/// handling 5-20 Hz. Each default entry must lie inside its documented range.
#[test]
fn default_sources_lie_inside_module_doc_table_ranges() {
    let s = default_excitation_sources();
    let table: [(&str, i64, i64); 6] = [
        ("X/Y", 60, 200),
        ("Z lead", 20, 80),
        ("Cooling fan", 80, 130),
        ("HVAC", 50, 60),
        ("Extruder", 300, 800),
        ("Handling", 5, 20),
    ];
    for (key, lo, hi) in table {
        let hit: Vec<_> = s.iter().filter(|e| e.name.contains(key)).collect();
        assert_eq!(hit.len(), 1, "exactly one source named like {key}");
        let f = hit[0].frequency_hz;
        assert!(
            f >= Fix128::from_int(lo) && f <= Fix128::from_int(hi),
            "{key}: {} Hz outside {lo}-{hi}",
            f.to_f64()
        );
    }
    assert_eq!(s.len(), 6);
}

/// Golden pin of the curated table (names are what `nearest_source.name`
/// reports to the caller; frequencies are the module's nominal values). The
/// doc only gives ranges, so this is a regression pin, not a derivation:
/// 7000 rpm / 60 = 116.7 Hz is the one value with an independent source.
#[test]
fn default_source_table_golden_pin() {
    let want: [(&str, i64); 6] = [
        ("X/Y stepper (Bambu X1C)", 120),
        ("Z lead-screw stepper", 40),
        ("Cooling fan 7000rpm", 117),
        ("Chamber HVAC 50Hz", 50),
        ("Extruder gear whine", 400),
        ("Handling / transport shock", 15),
    ];
    let got = default_excitation_sources();
    assert_eq!(got.len(), want.len());
    for (g, (n, f)) in got.iter().zip(want) {
        assert_eq!(g.name, n);
        assert_eq!(g.frequency_hz, Fix128::from_int(f));
    }
}

/// Lower band edge with an exactly representable ratio: band = 1/2 and a
/// source at exactly 2 w make ratio = w / 2w = 0.5 = 1 - band bit-exactly, so
/// the strict `>` must reject it. (The existing 0.3-band lower-boundary test
/// builds its source from a non-dyadic quotient, so its ratio is not exactly
/// 1 - band and it cannot tell `>` from `>=`.)
#[test]
fn lower_band_edge_exact_dyadic_ratio_is_not_risky() {
    let probe = wall(
        2,
        100,
        100,
        &default_excitation_sources(),
        Fix128::from_ratio(1, 2),
    );
    let w = probe.wall_frequency_hz;
    let s = src("2w", w.double());
    let r = wall(2, 100, 100, &[s], Fix128::from_ratio(1, 2));
    assert_eq!(
        r.frequency_ratio,
        Fix128::from_ratio(1, 2),
        "setup: exact ratio"
    );
    assert!(!r.is_risky, "ratio == 1 - band is outside the open band");
    // and the upper edge: source at 2w/3 is not dyadic, so use band 1: upper
    // edge ratio 2 with source w/2 (exact), strictly excluded
    let s2 = src("w/2", w.half());
    let r2 = wall(2, 100, 100, &[s2], Fix128::ONE);
    assert_eq!(
        r2.frequency_ratio,
        Fix128::from_int(2),
        "setup: exact ratio"
    );
    assert!(!r2.is_risky, "ratio == 1 + band is outside the open band");
}

/// The 7000 rpm fan name states 7000 rpm = 116.67 Hz; the table value 117 is
/// the rounded figure (within 0.3 %).
#[test]
fn fan_7000rpm_frequency_matches_rpm_over_60() {
    let s = default_excitation_sources();
    let fan = s.iter().find(|e| e.name.contains("7000rpm")).unwrap();
    let want = 7000.0 / 60.0;
    let rel = ((fan.frequency_hz.to_f64() - want) / want).abs();
    assert!(
        rel < 0.005,
        "fan {} Hz vs {want}",
        fan.frequency_hz.to_f64()
    );
}

/// Reference semantics actually implemented: the band test applies to the
/// nearest source only. Brute-force pin over a deterministic grid.
#[test]
fn is_risky_equals_band_test_on_brute_force_nearest_source() {
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as i64
    };
    let bands = [(5, 100), (20, 100), (50, 100)];
    let mut checked = 0;
    for _ in 0..40 {
        let t = 1 + next() % 5;
        let a = 40 + next() % 160;
        let b = 40 + next() % 160;
        let sources: Vec<ExcitationSource> = (0..4)
            .map(|_| src("s", Fix128::from_int(10 + next() % 500)))
            .collect();
        for (n, d) in bands {
            let band = Fix128::from_ratio(n, d);
            let r = wall(t, a, b, &sources, band);
            let w = r.wall_frequency_hz.to_f64();
            // brute-force nearest (first on ties), independent of the impl
            let mut best = sources[0].frequency_hz.to_f64();
            for s in &sources {
                let f = s.frequency_hz.to_f64();
                if (w - f).abs() < (w - best).abs() {
                    best = f;
                }
            }
            let ratio = w / best;
            let bf = n as f64 / d as f64;
            // skip knife-edge cases where Fix128 rounding could matter
            if ((ratio - (1.0 - bf)).abs() < 1e-9) || ((ratio - (1.0 + bf)).abs() < 1e-9) {
                continue;
            }
            let want = ratio > 1.0 - bf && ratio < 1.0 + bf;
            assert_eq!(r.is_risky, want, "t={t} a={a} b={b} band={bf} w={w}");
            checked += 1;
        }
    }
    assert!(checked > 100, "grid must actually compare cases: {checked}");
}

/// Widening the band never makes a wall safer (monotonicity, brute-force grid).
#[test]
fn widening_band_is_monotone_on_grid() {
    let sources = default_excitation_sources();
    let mut compared = 0;
    for t in 1..=4 {
        for a in (40..=200).step_by(20) {
            let mut prev = false;
            for n in [1, 5, 10, 20, 35, 50, 80] {
                let r = wall(t, a, a, &sources, Fix128::from_ratio(n, 100));
                assert!(!prev || r.is_risky, "t={t} a={a} band {n}% lost risk");
                prev = r.is_risky;
                compared += 1;
            }
        }
    }
    assert!(compared > 100);
}

/// Wall frequency scales as t (thickness) for D/(rho h) = E t^2/(12 rho(1-nu^2)):
/// doubling t doubles f. Checks the report's wall_frequency_hz wiring to the
/// right argument (thickness vs side).
#[test]
fn wall_frequency_doubles_with_thickness_and_quarters_with_doubled_sides() {
    let s = default_excitation_sources();
    let f1 = wall(2, 100, 100, &s, Fix128::from_ratio(1, 5))
        .wall_frequency_hz
        .to_f64();
    let f2 = wall(4, 100, 100, &s, Fix128::from_ratio(1, 5))
        .wall_frequency_hz
        .to_f64();
    let f3 = wall(2, 200, 200, &s, Fix128::from_ratio(1, 5))
        .wall_frequency_hz
        .to_f64();
    assert!(
        ((f2 / f1) - 2.0).abs() < 1e-5,
        "thickness scaling {}",
        f2 / f1
    );
    assert!(((f1 / f3) - 4.0).abs() < 1e-5, "side scaling {}", f1 / f3);
    // side order is symmetric for the SSSS closed form
    let f4 = wall(2, 100, 300, &s, Fix128::from_ratio(1, 5))
        .wall_frequency_hz
        .to_f64();
    let f5 = wall(2, 300, 100, &s, Fix128::from_ratio(1, 5))
        .wall_frequency_hz
        .to_f64();
    assert!((f4 - f5).abs() < 1e-9);
}

/// Poisson ratio is forwarded: nu 0 vs 0.35 changes f by 1/sqrt(1-nu^2).
#[test]
fn poisson_ratio_is_forwarded_to_the_plate_frequency() {
    let s = default_excitation_sources();
    let mk = |nu: Fix128| {
        analyze_wall_resonance(
            &MaterialProperties::pla(),
            nu,
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
            &s,
            Fix128::from_ratio(1, 5),
        )
        .wall_frequency_hz
        .to_f64()
    };
    let f0 = mk(Fix128::ZERO);
    let f35 = mk(nu());
    let want = 1.0 / (1.0f64 - 0.35 * 0.35).sqrt();
    assert!(((f35 / f0) - want).abs() < 1e-5, "{} vs {want}", f35 / f0);
}

/// Report reflects the nearest source object itself (name AND frequency) and
/// frequency_ratio is wall / that source's frequency, never the reverse.
#[test]
fn ratio_is_wall_over_source_not_inverse() {
    let s = vec![
        src("lo", Fix128::from_int(10)),
        src("hi", Fix128::from_int(5000)),
    ];
    let r = wall(2, 100, 100, &s, Fix128::from_ratio(1, 5));
    let w = r.wall_frequency_hz.to_f64();
    assert!(w > 100.0 && w < 400.0, "wall {w}");
    assert_eq!(r.nearest_source.name, "lo");
    assert!(((r.frequency_ratio.to_f64()) - w / 10.0).abs() < 1e-6);
    assert!(r.frequency_ratio > Fix128::ONE);
}

/// KNOWN DEFECT: the module doc says a wall "is at risk when its natural
/// frequency is within +-20 % of a significant excitation source" (any
/// source), but `is_risky` tests only the absolute-nearest source. With
/// sources S = w/1.22 (nearest by absolute distance) and G = w/0.813
/// (ratio 0.813, inside the band) the wall is reported safe.
#[test]
#[ignore = "known defect: AUD-A-S2W1-001: is_risky checks only the abs-nearest source; w within +-20% of a farther (larger) source reports false (S=w/1.22, G=w/0.813)"]
fn wall_within_band_of_a_non_nearest_source_is_risky() {
    let probe = wall(
        2,
        100,
        100,
        &default_excitation_sources(),
        Fix128::from_ratio(1, 5),
    );
    let w = probe.wall_frequency_hz;
    let s = src("S", w / Fix128::from_ratio(122, 100));
    let g = src("G", w / Fix128::from_ratio(813, 1000));
    let r = wall(2, 100, 100, &[s, g], Fix128::from_ratio(1, 5));
    assert_eq!(r.nearest_source.name, "S", "setup: S is abs-nearest");
    let wf = w.to_f64();
    let brute = [s, g].iter().any(|e| {
        let f = e.frequency_hz.to_f64();
        (wf - f).abs() / f < 0.2
    });
    assert!(brute, "setup: G is within 20%");
    assert!(r.is_risky, "wall within 20% of G must be risky");
}

/// KNOWN DEFECT (contract): an empty source slice panics by indexing
/// `sources[0]`; neither the doc nor the signature (slice, not NonEmpty)
/// says so.
#[test]
#[ignore = "known defect: AUD-A-S2W1-002: analyze_wall_resonance(&[]) panics (index out of bounds) with no documented precondition"]
fn empty_source_list_does_not_panic() {
    let res = std::panic::catch_unwind(|| wall(2, 100, 100, &[], Fix128::from_ratio(1, 5)));
    assert!(
        res.is_ok(),
        "empty source list must be handled or documented"
    );
}
