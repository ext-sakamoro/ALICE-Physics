//! Driving the piezoelectric element's three material presets, both
//! coupling directions, and the permittivity accessor through production
//! entry points.
//!
//! `wiring_guard.py` found seven zero-production-caller items in
//! `src/piezoelectric.rs`: `PiezoElement::{pzt_5a, quartz, pvdf}`,
//! `permittivity`, `force_from_voltage`, `voltage_from_force` and
//! `strain_under_stress`. Only `tests/engineering_oracles_fluid.rs`
//! (PZT-5A direct/converse d33 relations) and `tests/determinism_golden_f32.rs`
//! (golden-hash pin, all three materials) called into the module before this
//! example — the guard does not count `tests/` as production (src /
//! examples / benches / fuzz / bindings only).
//!
//! Closed forms, from the module doc (`src/piezoelectric.rs:1-17`):
//!
//! ```text
//! direct effect   (sensor)   : V   = d * sigma * thickness / permittivity
//!                                  = d * F * thickness / (A * eps)
//! converse effect (actuator) : F_b = E_field * d * Y * A
//!                                  = V * d * Y * A / thickness
//! mechanical fallback        : strain = stress / Y            (Hooke's law)
//! permittivity                : eps = eps_r * eps_0
//! ```
//!
//! PZT-5A's direct/converse relations and round-trip coupling factor
//! (`k^2 = d^2 Y / eps < 1`) already have closed-form coverage in
//! `tests/engineering_oracles_fluid.rs::piezoelectric_direct_and_converse_effects_match_d33_relations`;
//! this example exercises quartz and PVDF against the same formulas instead
//! of repeating the PZT-5A case, plus the `permittivity` accessor and
//! `strain_under_stress` for all three presets, plus degenerate / sign /
//! extreme-magnitude inputs that are not covered there.
//!
//! ```bash
//! cargo run --example piezoelectric_materials --features std
//! ```

use alice_physics::piezoelectric::PiezoElement;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// CODATA vacuum permittivity, truncated to the same precision as the
/// module's private `EPSILON_0` (`src/piezoelectric.rs:38`) — a physical
/// constant written down independently, never derived by calling the crate.
const EPSILON_0_F64: f64 = 8.854_188e-12;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn main() {
    let area = 1.0e-4_f32; // 1 cm^2
    let thickness = 2.0e-3_f32; // 2 mm

    let pzt = PiezoElement::pzt_5a(area, thickness);
    let quartz = PiezoElement::quartz(area, thickness);
    let pvdf = PiezoElement::pvdf(area, thickness);

    // --- permittivity: eps = eps_r * eps_0, for all three presets --------
    for (name, e) in [("PZT-5A", &pzt), ("quartz", &quartz), ("PVDF", &pvdf)] {
        let eps_r = f64::from(e.relative_permittivity);
        let eps = f64::from(e.permittivity());
        let eps_closed_form = eps_r * EPSILON_0_F64;
        println!(
            "[piezoelectric] {name} permittivity={eps:e} F/m (closed form {eps_closed_form:e})"
        );
        assert!(
            rel_err(eps, eps_closed_form) < 1.0e-6,
            "{name}: permittivity = {eps} vs {eps_closed_form}"
        );
    }

    // --- direct effect (sensor): V = d * F * thickness / (A * eps) --------
    // --- converse effect (actuator): F_b = V * d * Y * A / thickness -----
    // Checked for quartz and PVDF (PZT-5A already covered elsewhere, see
    // module doc comment above).
    for (name, e, d, eps_r) in [
        ("quartz", &quartz, 2.3e-12_f64, 4.5_f64),
        ("PVDF", &pvdf, 2.1e-11_f64, 12.0_f64),
    ] {
        let eps = eps_r * EPSILON_0_F64;
        for force in [10.0_f32, -10.0_f32, 50.0_f32] {
            let v = f64::from(e.voltage_from_force(force));
            let want_v = d * f64::from(force) * f64::from(thickness) / (eps * f64::from(area));
            println!(
                "[piezoelectric] {name} voltage_from_force({force}) = {v:e} V (closed form {want_v:e})"
            );
            assert!(
                rel_err(v, want_v) < 1.0e-4,
                "{name}: V({force}) = {v} vs {want_v}"
            );
        }

        let y = f64::from(e.youngs_modulus_pa);
        for voltage in [5.0_f32, -5.0_f32, 100.0_f32] {
            let f = f64::from(e.force_from_voltage(voltage));
            let want_f = f64::from(voltage) * d * y * f64::from(area) / f64::from(thickness);
            println!(
                "[piezoelectric] {name} force_from_voltage({voltage}) = {f:e} N (closed form {want_f:e})"
            );
            assert!(
                rel_err(f, want_f) < 1.0e-4,
                "{name}: F({voltage}) = {f} vs {want_f}"
            );
        }
    }

    // --- Hooke's law fallback: strain = stress / Y, all three materials ---
    for (name, e, y) in [
        ("PZT-5A", &pzt, 61.0e9_f64),
        ("quartz", &quartz, 78.0e9_f64),
        ("PVDF", &pvdf, 3.0e9_f64),
    ] {
        for stress in [10.0e6_f32, -10.0e6_f32] {
            let strain = f64::from(e.strain_under_stress(stress));
            let want = f64::from(stress) / y;
            println!(
                "[piezoelectric] {name} strain_under_stress({stress:e}) = {strain:e} (closed form {want:e})"
            );
            assert!(
                rel_err(strain, want) < 1.0e-5,
                "{name}: strain({stress}) = {strain} vs {want}"
            );
        }
    }

    // --- degenerate inputs: exact zero response -----------------------
    assert_eq!(
        pzt.voltage_from_force(0.0),
        0.0,
        "zero force -> zero voltage, exactly"
    );
    assert_eq!(
        pzt.force_from_voltage(0.0),
        0.0,
        "zero voltage -> zero force, exactly"
    );
    assert_eq!(
        pzt.strain_under_stress(0.0),
        0.0,
        "zero stress -> zero strain, exactly"
    );
    println!("[piezoelectric] zero force/voltage/stress -> zero response, exactly");

    // --- sign convention: every relation here is linear, hence odd -------
    let v10 = pzt.voltage_from_force(10.0);
    let v_neg10 = pzt.voltage_from_force(-10.0);
    assert_eq!(
        v_neg10, -v10,
        "voltage_from_force is odd: V(-F) = -V(F) exactly"
    );
    let f10 = pzt.force_from_voltage(10.0);
    let f_neg10 = pzt.force_from_voltage(-10.0);
    assert_eq!(
        f_neg10, -f10,
        "force_from_voltage is odd: F(-V) = -F(V) exactly"
    );
    let s10 = pzt.strain_under_stress(1.0e6);
    let s_neg10 = pzt.strain_under_stress(-1.0e6);
    assert_eq!(
        s_neg10, -s10,
        "strain_under_stress is odd: eps(-sigma) = -eps(sigma) exactly"
    );
    println!(
        "[piezoelectric] negative input flips sign exactly: V={v10}/{v_neg10} F={f10}/{f_neg10} strain={s10}/{s_neg10}"
    );

    // --- extreme magnitudes: f32 saturates to +-inf, it does not panic ---
    // (there is no integer cast anywhere in this module, unlike the
    // bucket-index casts in analytics_bridge/sketch, so overflow here is
    // plain f32 float overflow, measured rather than assumed).
    let r = catch_unwind(AssertUnwindSafe(|| pzt.voltage_from_force(f32::MAX)));
    println!("[piezoelectric] voltage_from_force(f32::MAX) = {r:?}");
    assert!(r.is_ok(), "f32 overflow saturates to infinity, no panic");
    assert!(
        r.unwrap().is_infinite(),
        "stress * d * thickness / eps overflows f32 at F=f32::MAX"
    );

    let r = catch_unwind(AssertUnwindSafe(|| pzt.force_from_voltage(f32::INFINITY)));
    println!("[piezoelectric] force_from_voltage(f32::INFINITY) = {r:?}");
    assert!(r.is_ok(), "infinite voltage input does not panic");
    assert!(r.unwrap().is_infinite());

    let r = catch_unwind(AssertUnwindSafe(|| pzt.strain_under_stress(f32::NAN)));
    println!("[piezoelectric] strain_under_stress(NaN) = {r:?}");
    assert!(
        r.is_ok(),
        "NaN propagates through division, it does not panic"
    );
    assert!(r.unwrap().is_nan());

    println!(
        "[piezoelectric] done: 7 production entry points exercised (pzt_5a, quartz, pvdf, permittivity, voltage_from_force, force_from_voltage, strain_under_stress)"
    );
}
