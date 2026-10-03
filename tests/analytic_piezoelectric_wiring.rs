//! Oracles for the wiring of `piezoelectric`
//! (`examples/piezoelectric_materials.rs`):
//! `PiezoElement::{pzt_5a, quartz, pvdf, permittivity, force_from_voltage,
//! voltage_from_force, strain_under_stress}`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! From the module doc (`src/piezoelectric.rs:1-17`):
//!
//! ```text
//! direct effect   (sensor)   : V   = d * sigma * thickness / permittivity
//!                                  = d * F * thickness / (A * eps)
//! converse effect (actuator) : F_b = E_field * d * Y * A
//!                                  = V * d * Y * A / thickness
//! mechanical fallback        : strain = stress / Y            (Hooke's law)
//! permittivity                : eps = eps_r * eps_0
//! round-trip coupling factor  : k^2 = F_b(V(1)) = d^2 * Y / eps   (< 1, passivity)
//! ```
//!
//! `d`, `relative_permittivity` and `youngs_modulus_pa` are the element's
//! material *inputs* (read back from its public fields), not quantities
//! under test; every expected value below is built from the same
//! independently-written formula, never by calling `PiezoElement`'s own
//! methods. PZT-5A's direct/converse d33 relations and round-trip coupling
//! factor already have closed-form coverage in
//! `tests/engineering_oracles_fluid.rs::piezoelectric_direct_and_converse_effects_match_d33_relations`;
//! this file extends the same formulas to quartz and PVDF (not duplicated
//! there) and adds the `permittivity` accessor plus degenerate-input and
//! panic coverage for all three presets.
//!
//! # Degenerate input (each result is pinned, panics are measured)
//!
//! Zero force / voltage / stress gives exactly zero response (no
//! `assert!(!panic)`-only check: every branch's return value is pinned).
//! Negative input flips the sign exactly (every relation here is linear in
//! its single scalar argument, hence odd). `area_m2 <= 0.0` and
//! `permittivity() <= 0.0` short-circuit `voltage_from_force` to `0.0`
//! (division-by-zero guard); `thickness_m <= 0.0` short-circuits
//! `force_from_voltage`; `youngs_modulus_pa <= 0.0` short-circuits
//! `strain_under_stress`. Extreme magnitudes (`f32::MAX`, `f32::INFINITY`,
//! `f32::NAN`) do not panic: there is no integer cast anywhere in this
//! module (unlike `analytics_bridge`'s bucket-index cast), so the only
//! possible outcome of overflow is plain `f32` saturation to `+-inf` or NaN
//! propagation, measured with `catch_unwind` rather than assumed.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::piezoelectric::PiezoElement;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// CODATA vacuum permittivity, truncated to the same precision as the
/// module's private `EPSILON_0` (`src/piezoelectric.rs:38`) — written down
/// here independently, never read back from the crate.
const EPSILON_0: f64 = 8.854_188e-12;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

// ---------------------------------------------------------------------------
// Closed forms, all three presets
// ---------------------------------------------------------------------------

#[test]
fn permittivity_matches_eps_r_times_eps_0_for_all_three_presets() {
    let area = 1.0e-4_f32;
    let thickness = 2.0e-3_f32;
    for (name, e, eps_r) in [
        ("PZT-5A", PiezoElement::pzt_5a(area, thickness), 1700.0_f64),
        ("quartz", PiezoElement::quartz(area, thickness), 4.5_f64),
        ("PVDF", PiezoElement::pvdf(area, thickness), 12.0_f64),
    ] {
        assert!(
            rel_err(f64::from(e.relative_permittivity), eps_r) < 1e-9,
            "{name}: preset relative_permittivity drifted from its documented value"
        );
        let eps = f64::from(e.permittivity());
        let want = eps_r * EPSILON_0;
        assert!(rel_err(eps, want) < 1e-6, "{name}: eps = {eps} vs {want}");
    }
}

/// Direct effect `V = d F t / (A eps)` and converse effect
/// `F_b = V d Y A / t` for quartz and PVDF (PZT-5A covered in
/// `tests/engineering_oracles_fluid.rs`).
#[test]
fn direct_and_converse_effects_match_closed_form_for_quartz_and_pvdf() {
    let area = 1.0e-4_f64;
    let thickness = 2.0e-3_f64;
    for (name, e, d, eps_r, y) in [
        (
            "quartz",
            PiezoElement::quartz(area as f32, thickness as f32),
            2.3e-12_f64,
            4.5_f64,
            78.0e9_f64,
        ),
        (
            "PVDF",
            PiezoElement::pvdf(area as f32, thickness as f32),
            2.1e-11_f64,
            12.0_f64,
            3.0e9_f64,
        ),
    ] {
        let eps = eps_r * EPSILON_0;
        for force in [1.0_f64, 10.0, 123.456, -17.0] {
            let v = f64::from(e.voltage_from_force(force as f32));
            let want_v = d * force * thickness / (eps * area);
            assert!(
                rel_err(v, want_v) < 1e-4,
                "{name}: V({force}) = {v} vs {want_v}"
            );
        }
        for voltage in [1.0_f64, 10.0, 99.9, -42.0] {
            let f = f64::from(e.force_from_voltage(voltage as f32));
            let want_f = voltage * d * y * area / thickness;
            assert!(
                rel_err(f, want_f) < 1e-4,
                "{name}: F_b({voltage}) = {f} vs {want_f}"
            );
        }
        // round-trip coupling factor k^2 = d^2 Y / eps, passivity k^2 < 1,
        // for the materials the existing oracle does not cover.
        let k_sq = f64::from(e.force_from_voltage(e.voltage_from_force(1.0)));
        let want_k_sq = d * d * y / eps;
        assert!(
            rel_err(k_sq, want_k_sq) < 1e-3,
            "{name}: k^2 = {k_sq} vs {want_k_sq}"
        );
        assert!(
            k_sq < 1.0,
            "{name}: k^2 = {k_sq} must be below 1 (passivity)"
        );
    }
}

#[test]
fn strain_under_stress_matches_hookes_law_for_all_three_presets() {
    for (name, e, y) in [
        ("PZT-5A", PiezoElement::pzt_5a(1.0e-4, 2.0e-3), 61.0e9_f64),
        ("quartz", PiezoElement::quartz(1.0e-4, 2.0e-3), 78.0e9_f64),
        ("PVDF", PiezoElement::pvdf(1.0e-4, 2.0e-3), 3.0e9_f64),
    ] {
        for stress in [1.0e6_f64, 10.0e6, -25.0e6, 61.0e6] {
            let strain = f64::from(e.strain_under_stress(stress as f32));
            let want = stress / y;
            assert!(
                rel_err(strain, want) < 1e-5,
                "{name}: strain({stress}) = {strain} vs {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Degenerate input
// ---------------------------------------------------------------------------

#[test]
fn zero_input_gives_exactly_zero_response() {
    let e = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    assert_eq!(e.voltage_from_force(0.0), 0.0);
    assert_eq!(e.force_from_voltage(0.0), 0.0);
    assert_eq!(e.strain_under_stress(0.0), 0.0);
}

#[test]
fn negative_input_flips_sign_exactly() {
    let e = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    for force in [1.0_f32, 10.0, 999.5] {
        assert_eq!(
            e.voltage_from_force(-force),
            -e.voltage_from_force(force),
            "V(-F) = -V(F) for F={force}"
        );
    }
    for voltage in [1.0_f32, 10.0, 250.0] {
        assert_eq!(
            e.force_from_voltage(-voltage),
            -e.force_from_voltage(voltage),
            "F(-V) = -F(V) for V={voltage}"
        );
    }
    for stress in [1.0e5_f32, 1.0e6, 6.1e7] {
        assert_eq!(
            e.strain_under_stress(-stress),
            -e.strain_under_stress(stress),
            "eps(-sigma) = -eps(sigma) for sigma={stress}"
        );
    }
}

/// `area_m2 <= 0.0` or `permittivity() <= 0.0` (i.e. `relative_permittivity
/// <= 0.0`, since `EPSILON_0 > 0`) short-circuits `voltage_from_force` to
/// exactly `0.0` — the division-by-zero guard, not a computed near-zero
/// value.
#[test]
fn division_by_zero_guards_return_exact_zero_not_nan_or_inf() {
    let mut e = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    e.area_m2 = 0.0;
    assert_eq!(e.voltage_from_force(10.0), 0.0, "area_m2 == 0.0 guard");
    e.area_m2 = -1.0e-4;
    assert_eq!(e.voltage_from_force(10.0), 0.0, "area_m2 < 0.0 guard");

    let mut e2 = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    e2.relative_permittivity = 0.0;
    assert_eq!(
        e2.voltage_from_force(10.0),
        0.0,
        "permittivity() == 0.0 guard"
    );
    e2.relative_permittivity = -5.0;
    assert_eq!(
        e2.voltage_from_force(10.0),
        0.0,
        "permittivity() < 0.0 guard"
    );

    let mut e3 = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    e3.thickness_m = 0.0;
    assert_eq!(e3.force_from_voltage(10.0), 0.0, "thickness_m == 0.0 guard");
    e3.thickness_m = -2.0e-3;
    assert_eq!(e3.force_from_voltage(10.0), 0.0, "thickness_m < 0.0 guard");

    let mut e4 = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    e4.youngs_modulus_pa = 0.0;
    assert_eq!(
        e4.strain_under_stress(10.0e6),
        0.0,
        "youngs_modulus_pa == 0.0 guard"
    );
    e4.youngs_modulus_pa = -61.0e9;
    assert_eq!(
        e4.strain_under_stress(10.0e6),
        0.0,
        "youngs_modulus_pa < 0.0 guard"
    );
}

/// Extreme magnitudes: plain `f32` float overflow (no integer cast anywhere
/// in this module), measured with `catch_unwind` rather than assumed not to
/// panic.
#[test]
fn extreme_magnitudes_saturate_without_panicking() {
    let e = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);

    let r = catch_unwind(AssertUnwindSafe(|| e.voltage_from_force(f32::MAX)));
    assert!(r.is_ok(), "f32::MAX force must not panic");
    assert!(
        r.unwrap().is_infinite(),
        "stress = F/A overflows f32 at F=f32::MAX, A=1e-4"
    );

    let r = catch_unwind(AssertUnwindSafe(|| e.voltage_from_force(f32::MIN)));
    assert!(r.is_ok(), "f32::MIN force must not panic");
    assert!(r.unwrap().is_infinite());

    let r = catch_unwind(AssertUnwindSafe(|| e.force_from_voltage(f32::INFINITY)));
    assert!(r.is_ok(), "+inf voltage must not panic");
    assert!(r.unwrap().is_infinite());

    let r = catch_unwind(AssertUnwindSafe(|| e.force_from_voltage(f32::NEG_INFINITY)));
    assert!(r.is_ok(), "-inf voltage must not panic");
    assert!(r.unwrap().is_infinite());

    let r = catch_unwind(AssertUnwindSafe(|| e.strain_under_stress(f32::NAN)));
    assert!(r.is_ok(), "NaN stress must not panic");
    assert!(
        r.unwrap().is_nan(),
        "NaN / finite Y propagates to NaN, it is not swallowed"
    );

    let r = catch_unwind(AssertUnwindSafe(|| e.voltage_from_force(f32::NAN)));
    assert!(r.is_ok(), "NaN force must not panic");
    assert!(
        r.unwrap().is_nan(),
        "NaN propagates through the direct-effect formula, guards compare \
         false against NaN so the early return is not taken"
    );
}
