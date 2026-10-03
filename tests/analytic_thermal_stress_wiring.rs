//! Oracles for `thermal_stress::yield_temperature_c` (previously unwired) and the
//! stated properties of the module it belongs to.
//!
//! Closed forms (Timoshenko & Goodier, *Theory of Elasticity* ch. 14; Boley & Weiner):
//!
//! ```text
//! fully constrained bar     sigma = -E alpha c dT          (c in [0, 1])
//! yield envelope            T_y   = T_i + sigma_y / (E alpha c)
//! ```
//!
//! Linear CTE values are the literature numbers per material (PLA 68e-6, ABS 90e-6, PC 65e-6, ...),
//! written out here, not read from the implementation's table through its API.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::thermal_stress::{analyze_thermal_stress, yield_temperature_c};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

fn materials() -> Vec<(MaterialProperties, f64)> {
    vec![
        (MaterialProperties::pla(), 68e-6),
        (MaterialProperties::petg(), 60e-6),
        (MaterialProperties::abs(), 90e-6),
        (MaterialProperties::pc(), 65e-6),
        (MaterialProperties::nylon(), 90e-6),
        (MaterialProperties::cf_nylon(), 30e-6),
        (MaterialProperties::tpu(), 140e-6),
        (MaterialProperties::peek(), 47e-6),
        (MaterialProperties::sus304(), 17.3e-6),
        (MaterialProperties::a5052(), 23.8e-6),
    ]
}

fn c(num: i64, den: i64) -> Fix128 {
    Fix128::from_ratio(num, den)
}

#[test]
fn yield_temperature_matches_the_closed_form_for_every_material() {
    for (m, alpha) in materials() {
        let e = m.youngs_modulus_gpa.to_f64() * 1000.0; // MPa
        let sy = m.yield_strength_mpa.to_f64();
        for (cn, cd) in [(1, 1), (1, 2), (1, 4), (7, 10)] {
            let coef = cn as f64 / cd as f64;
            let ty = yield_temperature_c(&m, c(cn, cd), Fix128::from_int(20)).to_f64();
            let want = 20.0 + sy / (e * alpha * coef);
            assert!(
                rel(ty, want) < 1e-9,
                "{} c = {coef}: {ty} vs {want}",
                m.name
            );
        }
    }
}

#[test]
fn envelope_scales_as_one_over_constraint_and_shifts_with_installation_temperature() {
    let m = MaterialProperties::pla();
    let rise = |cc: Fix128, ti: i64| {
        yield_temperature_c(&m, cc, Fix128::from_int(ti)).to_f64() - ti as f64
    };
    let full = rise(Fix128::ONE, 20);
    // half the constraint, twice the temperature rise to yield
    assert!(rel(rise(c(1, 2), 20), 2.0 * full) < 1e-9);
    assert!(rel(rise(c(1, 4), 20), 4.0 * full) < 1e-9);
    // a different installation temperature shifts T_y one for one
    let shifted = yield_temperature_c(&m, Fix128::ONE, Fix128::from_int(80)).to_f64();
    let base = yield_temperature_c(&m, Fix128::ONE, Fix128::from_int(20)).to_f64();
    assert!(rel(shifted - base, 60.0) < 1e-9);
}

#[test]
fn analysis_at_the_envelope_temperature_reports_exactly_yield() {
    for (m, _) in materials() {
        let ti = Fix128::from_int(20);
        let ty = yield_temperature_c(&m, Fix128::ONE, ti);
        // heating to T_y: compressive stress of magnitude sigma_y, FoS = 1
        let at = analyze_thermal_stress(&m, Fix128::ONE, ti, ty);
        assert!(rel(at.factor_of_safety.to_f64(), 1.0) < 1e-9, "{}", m.name);
        assert!(
            at.thermal_stress_mpa < Fix128::ZERO,
            "{}: heating is compressive",
            m.name
        );
        assert!(
            rel(
                at.thermal_stress_mpa.to_f64().abs(),
                m.yield_strength_mpa.to_f64()
            ) < 1e-9
        );
        // half way: stress halves, FoS = 2
        let half = (ty + ti).half();
        let h = analyze_thermal_stress(&m, Fix128::ONE, ti, half);
        assert!(rel(h.factor_of_safety.to_f64(), 2.0) < 1e-9, "{}", m.name);
        // cooling by the same rise reaches +sigma_y (tension)
        let cold = ti - (ty - ti);
        let k = analyze_thermal_stress(&m, Fix128::ONE, ti, cold);
        assert!(rel(k.thermal_stress_mpa.to_f64(), m.yield_strength_mpa.to_f64()) < 1e-9);
    }
}

#[test]
fn free_part_never_yields_thermally() {
    // c = 0: denominator vanishes, the documented answer is a huge temperature
    let ty = yield_temperature_c(
        &MaterialProperties::pla(),
        Fix128::ZERO,
        Fix128::from_int(20),
    );
    assert_eq!(ty, Fix128::from_int(i64::MAX >> 32));
    // a vanishing-but-non-zero constraint is already far above any real service temperature
    let tiny = yield_temperature_c(
        &MaterialProperties::pla(),
        c(1, 1_000_000),
        Fix128::from_int(20),
    );
    assert!(tiny.to_f64() > 1.0e6);
}

#[test]
fn degenerate_material_and_constraint_values() {
    // zero yield strength: yields immediately at the installation temperature
    let mut m = MaterialProperties::pla();
    m.yield_strength_mpa = Fix128::ZERO;
    assert_eq!(
        yield_temperature_c(&m, Fix128::ONE, Fix128::from_int(20)),
        Fix128::from_int(20)
    );
    // zero modulus: no stress can develop, treated like the free case
    let mut z = MaterialProperties::pla();
    z.youngs_modulus_gpa = Fix128::ZERO;
    assert_eq!(
        yield_temperature_c(&z, Fix128::ONE, Fix128::from_int(20)),
        Fix128::from_int(i64::MAX >> 32)
    );
    // a negative constraint coefficient is outside the documented [0, 1] range; the
    // formula mirrors the rise to the cold side instead of panicking
    let neg = yield_temperature_c(&MaterialProperties::pla(), c(-1, 2), Fix128::from_int(20));
    let pos = yield_temperature_c(&MaterialProperties::pla(), c(1, 2), Fix128::from_int(20));
    assert!((neg.to_f64() - 20.0 + pos.to_f64() - 20.0).abs() < 1e-9);
}

#[test]
fn near_glass_transition_and_safety_flags_follow_their_definitions() {
    let m = MaterialProperties::pla(); // Tg = 60 C
    let tg = m.glass_transition_c.to_f64();
    let ti = Fix128::from_int(20);
    // |T - Tg| <= 20: flagged, whatever the stress is
    for (t, flagged) in [(39, false), (40, true), (60, true), (80, true), (81, false)] {
        let r = analyze_thermal_stress(&m, c(1, 100), ti, Fix128::from_int(t));
        assert_eq!(r.near_glass_transition, flagged, "T = {t} (Tg = {tg})");
        if flagged {
            assert!(!r.is_safe);
        }
    }
    // zero Tg means "not applicable": never flagged
    let mut metal = MaterialProperties::sus304();
    metal.glass_transition_c = Fix128::ZERO;
    assert!(
        !analyze_thermal_stress(&metal, Fix128::ONE, ti, Fix128::from_int(5)).near_glass_transition
    );
}

#[test]
fn zero_stress_reports_the_documented_sentinel_factor_of_safety_and_is_safe() {
    // dT = 0: no stress; FoS is the "unbounded" sentinel i64::MAX >> 8
    let m = MaterialProperties::pc();
    let r = analyze_thermal_stress(&m, Fix128::ONE, Fix128::from_int(20), Fix128::from_int(20));
    assert_eq!(r.thermal_stress_mpa, Fix128::ZERO);
    assert_eq!(r.factor_of_safety, Fix128::from_int(i64::MAX >> 8));
    assert!(!r.near_glass_transition && r.is_safe);
    // a free part (c = 0) is the same case
    let free = analyze_thermal_stress(&m, Fix128::ZERO, Fix128::from_int(20), Fix128::from_int(80));
    assert_eq!(free.factor_of_safety, Fix128::from_int(i64::MAX >> 8));
}

#[test]
fn is_safe_flips_where_the_factor_of_safety_crosses_two() {
    // FoS = sigma_y / (E alpha c dT): pick dT just either side of FoS = 2
    let m = MaterialProperties::pc(); // Tg = 145 C, far from the temperatures used
    let e = m.youngs_modulus_gpa.to_f64() * 1000.0;
    let alpha = 65e-6;
    let dt_two = m.yield_strength_mpa.to_f64() / (2.0 * e * alpha);
    let ti = Fix128::from_int(20);
    let below =
        analyze_thermal_stress(&m, Fix128::ONE, ti, Fix128::from_f64(20.0 + dt_two * 0.999));
    let above =
        analyze_thermal_stress(&m, Fix128::ONE, ti, Fix128::from_f64(20.0 + dt_two * 1.001));
    assert!(below.factor_of_safety > Fix128::from_int(2) && below.is_safe);
    assert!(above.factor_of_safety < Fix128::from_int(2) && !above.is_safe);
}
