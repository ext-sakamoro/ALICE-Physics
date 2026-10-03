//! Audit oracles for `alice_physics::piezoelectric`.
//!
//! References are the textbook constitutive relations evaluated in f64 with
//! the CODATA vacuum permittivity:
//!   direct   V = d F t / (A eps_r eps_0)
//!   converse F_blocked = V d Y A / t, free strain = d V / t
//!   Hooke    strain = sigma / Y

use alice_physics::piezoelectric::PiezoElement;

const EPS0: f64 = 8.854_187_812_8e-12;

fn rel_ok(got: f32, want: f64, tol: f64) -> bool {
    let g = f64::from(got);
    (g - want).abs() <= tol * want.abs().max(1.0e-300)
}

fn presets(a: f32, t: f32) -> [(&'static str, PiezoElement); 3] {
    [
        ("pzt_5a", PiezoElement::pzt_5a(a, t)),
        ("quartz", PiezoElement::quartz(a, t)),
        ("pvdf", PiezoElement::pvdf(a, t)),
    ]
}

#[test]
fn direct_effect_matches_closed_form_for_all_presets() {
    for (name, e) in presets(1.0e-4, 2.0e-3) {
        for &f in &[10.0_f32, -3.5, 250.0] {
            let want = f64::from(e.d_coefficient) * f64::from(f) * f64::from(e.thickness_m)
                / (f64::from(e.area_m2) * f64::from(e.relative_permittivity) * EPS0);
            let got = e.voltage_from_force(f);
            assert!(rel_ok(got, want, 1.0e-5), "{name} F={f}: {got} vs {want}");
        }
    }
}

#[test]
fn direct_effect_has_a_known_absolute_value_for_pzt() {
    // d = 374 pC/N, 10 N on 1 cm^2 (sigma = 1e5 Pa), t = 2 mm, eps_r = 1700:
    // V = 3.74e-10 * 1e5 * 2e-3 / (1700 * 8.8541878128e-12) = 4.9694 V
    let v = PiezoElement::pzt_5a(1.0e-4, 2.0e-3).voltage_from_force(10.0);
    assert!((f64::from(v) - 4.9694).abs() < 2.0e-3, "got {v}");
}

#[test]
fn converse_effect_blocked_force_matches_closed_form() {
    for (name, e) in presets(2.0e-4, 1.0e-3) {
        for &v in &[10.0_f32, -40.0, 120.0] {
            let want = f64::from(v)
                * f64::from(e.d_coefficient)
                * f64::from(e.youngs_modulus_pa)
                * f64::from(e.area_m2)
                / f64::from(e.thickness_m);
            let got = e.force_from_voltage(v);
            assert!(rel_ok(got, want, 1.0e-5), "{name} V={v}: {got} vs {want}");
        }
    }
}

#[test]
fn blocked_force_over_area_gives_the_free_strain_d_times_field() {
    // strain = d E = d V / t, and blocked stress = Y * strain.
    for (name, e) in presets(3.0e-4, 5.0e-4) {
        let v = 30.0_f32;
        let stress = e.force_from_voltage(v) / e.area_m2;
        let strain = e.strain_under_stress(stress);
        let want = f64::from(e.d_coefficient) * f64::from(v) / f64::from(e.thickness_m);
        assert!(rel_ok(strain, want, 1.0e-5), "{name}: {strain} vs {want}");
    }
}

#[test]
fn hooke_strain_matches_stress_over_modulus() {
    for (name, e) in presets(1.0e-4, 1.0e-3) {
        for &s in &[1.0e6_f32, -2.5e7, 0.0] {
            let got = e.strain_under_stress(s);
            let want = f64::from(s) / f64::from(e.youngs_modulus_pa);
            assert!(
                (f64::from(got) - want).abs() <= 1.0e-6 * want.abs(),
                "{name} {s}"
            );
        }
    }
}

#[test]
fn responses_are_odd_and_vanish_at_zero_input() {
    for (_, e) in presets(1.0e-4, 1.0e-3) {
        assert_eq!(e.voltage_from_force(0.0), 0.0);
        assert_eq!(e.force_from_voltage(0.0), 0.0);
        assert_eq!(e.strain_under_stress(0.0), 0.0);
        assert_eq!(e.voltage_from_force(-7.0), -e.voltage_from_force(7.0));
        assert_eq!(e.force_from_voltage(-7.0), -e.force_from_voltage(7.0));
    }
}

#[test]
fn geometry_and_permittivity_scaling_of_the_direct_effect() {
    let base = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    let v0 = base.voltage_from_force(10.0);
    // V ~ t / A / eps_r
    let mut e = base;
    e.thickness_m *= 2.0;
    assert!(rel_ok(
        e.voltage_from_force(10.0),
        2.0 * f64::from(v0),
        1.0e-5
    ));
    let mut e = base;
    e.area_m2 *= 4.0;
    assert!(rel_ok(
        e.voltage_from_force(10.0),
        0.25 * f64::from(v0),
        1.0e-5
    ));
    let mut e = base;
    e.relative_permittivity *= 2.0;
    assert!(rel_ok(
        e.voltage_from_force(10.0),
        0.5 * f64::from(v0),
        1.0e-5
    ));
    // the converse blocked force ~ A Y / t
    let f0 = base.force_from_voltage(10.0);
    let mut e = base;
    e.area_m2 *= 3.0;
    assert!(rel_ok(
        e.force_from_voltage(10.0),
        3.0 * f64::from(f0),
        1.0e-5
    ));
    let mut e = base;
    e.thickness_m *= 2.0;
    assert!(rel_ok(
        e.force_from_voltage(10.0),
        0.5 * f64::from(f0),
        1.0e-5
    ));
    let mut e = base;
    e.youngs_modulus_pa *= 2.0;
    assert!(rel_ok(
        e.force_from_voltage(10.0),
        2.0 * f64::from(f0),
        1.0e-5
    ));
}

#[test]
fn permittivity_is_relative_times_vacuum_permittivity() {
    for (name, e) in presets(1.0e-4, 1.0e-3) {
        let want = f64::from(e.relative_permittivity) * EPS0;
        assert!(rel_ok(e.permittivity(), want, 1.0e-6), "{name}");
    }
}

#[test]
fn presets_carry_published_constants_and_the_given_geometry() {
    let p = PiezoElement::pzt_5a(1.5e-4, 2.5e-3);
    assert_eq!((p.area_m2, p.thickness_m), (1.5e-4, 2.5e-3));
    // PZT-5A: d33 = 374 pC/N, eps_r ~ 1700, Y ~ 61 GPa
    assert_eq!(
        (
            p.d_coefficient,
            p.relative_permittivity,
            p.youngs_modulus_pa
        ),
        (3.74e-10, 1700.0, 61.0e9)
    );
    let q = PiezoElement::quartz(1.5e-4, 2.5e-3);
    assert_eq!((q.area_m2, q.thickness_m), (1.5e-4, 2.5e-3));
    // quartz: d11 = 2.3 pC/N, eps_r 4.5, Y ~ 78 GPa
    assert_eq!(
        (
            q.d_coefficient,
            q.relative_permittivity,
            q.youngs_modulus_pa
        ),
        (2.3e-12, 4.5, 78.0e9)
    );
    let f = PiezoElement::pvdf(1.5e-4, 2.5e-3);
    assert_eq!((f.area_m2, f.thickness_m), (1.5e-4, 2.5e-3));
    // PVDF: |d| ~ 21 pC/N, eps_r 12, Y ~ 3 GPa
    assert_eq!(
        (
            f.d_coefficient,
            f.relative_permittivity,
            f.youngs_modulus_pa
        ),
        (2.1e-11, 12.0, 3.0e9)
    );
}

#[test]
fn nonpositive_guards_return_zero() {
    let base = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    let mut e = base;
    e.area_m2 = -1.0e-4;
    assert_eq!(e.voltage_from_force(10.0), 0.0);
    let mut e = base;
    e.relative_permittivity = 0.0;
    assert_eq!(e.voltage_from_force(10.0), 0.0);
    let mut e = base;
    e.thickness_m = -2.0e-3;
    assert_eq!(e.force_from_voltage(10.0), 0.0);
    let mut e = base;
    e.youngs_modulus_pa = 0.0;
    assert_eq!(e.strain_under_stress(1.0e6), 0.0);
    let mut e = base;
    e.youngs_modulus_pa = -1.0;
    assert_eq!(e.strain_under_stress(1.0e6), 0.0);
}

/// The two directions guard different parameters: `voltage_from_force`
/// rejects `area <= 0` but accepts a negative thickness (the voltage flips
/// sign), and `force_from_voltage` rejects `thickness <= 0` but accepts a
/// negative area (the force flips sign).
#[test]
#[ignore = "known defect: AUD-A-S6W1-010: voltage_from_force(thickness<0) and force_from_voltage(area<0) return sign-flipped nonzero values; the opposite-direction parameter is guarded"]
fn negative_geometry_is_rejected_in_both_directions() {
    let base = PiezoElement::pzt_5a(1.0e-4, 2.0e-3);
    let mut e = base;
    e.thickness_m = -2.0e-3;
    assert_eq!(e.voltage_from_force(10.0), 0.0);
    let mut e = base;
    e.area_m2 = -1.0e-4;
    assert_eq!(e.force_from_voltage(10.0), 0.0);
}
