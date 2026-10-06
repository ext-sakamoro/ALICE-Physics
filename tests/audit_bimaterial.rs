//! Audit oracles for `alice_physics::bimaterial`.
//!
//! Expected values come from the rule of mixtures (Hill 1952), from the membrane
//! force balance of a bonded bilayer (net force zero, equal total strain) written
//! out here, and from the stated empirical bond-line rule.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::disallowed_methods)]

use alice_physics::bimaterial::{
    analyze_bimaterial, effective_modulus_reuss_mpa, effective_modulus_voigt_mpa,
    interfacial_bond_strength_mpa, published_cte_per_c, thermal_residual_stress_mpa,
    BimaterialSide,
};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

fn fx(v: f64) -> Fix128 {
    Fix128::from_f64(v)
}
fn rel(a: Fix128, want: f64) -> f64 {
    (a.to_f64() - want).abs() / want.abs().max(1e-300)
}
fn side(m: MaterialProperties, t: f64) -> BimaterialSide {
    BimaterialSide::from_material(m, fx(t))
}
/// E in MPa of a preset (GPa * 1000).
fn e_mpa(m: &MaterialProperties) -> f64 {
    m.youngs_modulus_gpa.to_f64() * 1000.0
}

// ------------------------------------------------------------- published CTE

#[test]
fn published_cte_table_matches_the_documented_values() {
    let cases: [(MaterialProperties, f64); 10] = [
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
    ];
    for (m, want) in cases {
        assert!(rel(published_cte_per_c(&m), want) < 1e-12, "{}", m.name);
    }
}

#[test]
fn unknown_material_name_falls_back_to_the_pla_value() {
    let mut m = MaterialProperties::pla();
    m.name = "unobtainium";
    assert!(rel(published_cte_per_c(&m), 68e-6) < 1e-12);
}

#[test]
fn from_material_keeps_thickness_and_looks_up_the_cte() {
    let s = side(MaterialProperties::tpu(), 1.75);
    assert_eq!(s.thickness_mm, fx(1.75));
    assert!(rel(s.cte_per_c, 140e-6) < 1e-12);
    assert_eq!(s.material, MaterialProperties::tpu());
}

// ------------------------------------------------------------- Voigt / Reuss

#[test]
fn voigt_and_reuss_follow_the_rule_of_mixtures() {
    let pairs = [
        (
            MaterialProperties::pla(),
            2.0,
            MaterialProperties::abs(),
            1.5,
        ),
        (
            MaterialProperties::tpu(),
            0.4,
            MaterialProperties::peek(),
            3.1,
        ),
        (
            MaterialProperties::pc(),
            1.0,
            MaterialProperties::sus304(),
            0.2,
        ),
    ];
    for (ma, ta, mb, tb) in pairs {
        let (a, b) = (side(ma, ta), side(mb, tb));
        let (ea, eb) = (e_mpa(&ma), e_mpa(&mb));
        let (va, vb) = (ta / (ta + tb), tb / (ta + tb));
        let voigt = va * ea + vb * eb;
        let reuss = 1.0 / (va / ea + vb / eb);
        assert!(
            rel(effective_modulus_voigt_mpa(&a, &b), voigt) < 1e-11,
            "{} / {} Voigt",
            ma.name,
            mb.name
        );
        assert!(
            rel(effective_modulus_reuss_mpa(&a, &b), reuss) < 1e-9,
            "{} / {} Reuss",
            ma.name,
            mb.name
        );
        // bounds: min(E) <= Reuss <= Voigt <= max(E), and both symmetric in the pair order
        let v = effective_modulus_voigt_mpa(&a, &b).to_f64();
        let r = effective_modulus_reuss_mpa(&a, &b).to_f64();
        assert!(r <= v && v <= ea.max(eb) + 1e-9 && r >= ea.min(eb) - 1e-9);
        assert_eq!(
            effective_modulus_voigt_mpa(&b, &a),
            effective_modulus_voigt_mpa(&a, &b)
        );
        assert!(rel(effective_modulus_reuss_mpa(&b, &a), reuss) < 1e-9);
    }
}

#[test]
fn a_zero_thickness_layer_drops_out_of_both_bounds() {
    let (a, b) = (
        side(MaterialProperties::pla(), 0.0),
        side(MaterialProperties::abs(), 2.0),
    );
    let eb = e_mpa(&MaterialProperties::abs());
    assert!(rel(effective_modulus_voigt_mpa(&a, &b), eb) < 1e-11);
    assert!(rel(effective_modulus_reuss_mpa(&a, &b), eb) < 1e-9);
}

#[test]
fn a_zero_stiffness_layer_makes_reuss_zero_and_voigt_the_other_layers_share() {
    let mut soft = MaterialProperties::pla();
    soft.youngs_modulus_gpa = Fix128::ZERO;
    let (a, b) = (side(soft, 1.0), side(MaterialProperties::abs(), 1.0));
    assert_eq!(effective_modulus_reuss_mpa(&a, &b), Fix128::ZERO);
    assert!(
        rel(
            effective_modulus_voigt_mpa(&a, &b),
            0.5 * e_mpa(&MaterialProperties::abs())
        ) < 1e-11
    );
}

// ------------------------------------------------------------- thermal residual stress

#[test]
fn residual_stress_equal_thickness_matches_the_membrane_closed_form() {
    let (ma, mb) = (MaterialProperties::tpu(), MaterialProperties::pc());
    let (a, b) = (side(ma, 1.0), side(mb, 1.0));
    let (ea, eb) = (e_mpa(&ma), e_mpa(&mb));
    let (dal, dt) = (140e-6 - 65e-6, 200.0 - 20.0);
    let want = dal * dt * ea * eb / (ea + eb);
    let got = thermal_residual_stress_mpa(&a, &b, fx(200.0), fx(20.0));
    assert!(rel(got, want) < 1e-9, "{} vs {want}", got.to_f64());
}

#[test]
fn residual_stress_sign_follows_cte_difference_and_temperature_direction() {
    let (hi, lo) = (
        side(MaterialProperties::tpu(), 1.0),
        side(MaterialProperties::sus304(), 1.0),
    );
    let cool = thermal_residual_stress_mpa(&hi, &lo, fx(200.0), fx(20.0));
    assert!(
        cool > Fix128::ZERO,
        "higher-CTE layer in tension on cooling"
    );
    assert!(thermal_residual_stress_mpa(&lo, &hi, fx(200.0), fx(20.0)) < Fix128::ZERO);
    // heating instead of cooling reverses the sign, same magnitude
    let heat = thermal_residual_stress_mpa(&hi, &lo, fx(20.0), fx(200.0));
    assert!(rel(heat, -cool.to_f64()) < 1e-12);
    // no temperature change, or no CTE mismatch: no stress
    assert_eq!(
        thermal_residual_stress_mpa(&hi, &lo, fx(20.0), fx(20.0)),
        Fix128::ZERO
    );
    let same = side(MaterialProperties::petg(), 1.0);
    assert_eq!(
        thermal_residual_stress_mpa(&same, &same, fx(200.0), fx(20.0)),
        Fix128::ZERO
    );
}

#[test]
fn residual_stress_is_linear_in_the_temperature_drop() {
    let (a, b) = (
        side(MaterialProperties::abs(), 1.0),
        side(MaterialProperties::petg(), 1.0),
    );
    let s1 = thermal_residual_stress_mpa(&a, &b, fx(120.0), fx(20.0)).to_f64();
    let s2 = thermal_residual_stress_mpa(&a, &b, fx(220.0), fx(20.0)).to_f64();
    assert!((s2 - 2.0 * s1).abs() < 1e-9 * s2.abs());
}

#[test]
fn zero_stiffness_pair_has_zero_residual_stress() {
    let mut z = MaterialProperties::pla();
    z.youngs_modulus_gpa = Fix128::ZERO;
    let (a, b) = (side(z, 1.0), side(z, 1.0));
    assert_eq!(
        thermal_residual_stress_mpa(&a, &b, fx(200.0), fx(20.0)),
        Fix128::ZERO
    );
}

#[test]
// AUD-A-S4W3-010
fn residual_stress_accounts_for_unequal_layer_thickness() {
    // sigma_a t_a + sigma_b t_b = 0 and equal total strain:
    // sigma_a = da * dT * E_a E_b t_b / (E_a t_a + E_b t_b)
    let (ma, mb, ta, tb) = (
        MaterialProperties::abs(),
        MaterialProperties::petg(),
        1.0,
        4.0,
    );
    let (a, b) = (side(ma, ta), side(mb, tb));
    let (ea, eb) = (e_mpa(&ma), e_mpa(&mb));
    let want = (90e-6 - 60e-6) * 180.0 * ea * eb * tb / (ea * ta + eb * tb);
    let got = thermal_residual_stress_mpa(&a, &b, fx(200.0), fx(20.0));
    assert!(rel(got, want) < 1e-6, "{} vs {want}", got.to_f64());
}

// ------------------------------------------------------------- bond strength

#[test]
fn bond_strength_same_material_is_the_yield_and_dissimilar_is_half_the_geometric_mean() {
    for m in [
        MaterialProperties::pla(),
        MaterialProperties::pc(),
        MaterialProperties::tpu(),
    ] {
        assert_eq!(interfacial_bond_strength_mpa(&m, &m), m.yield_strength_mpa);
    }
    let (pla, abs, pc) = (
        MaterialProperties::pla(),
        MaterialProperties::abs(),
        MaterialProperties::pc(),
    );
    let want = 0.5 * (50.0f64 * 40.0).sqrt();
    assert!(rel(interfacial_bond_strength_mpa(&pla, &abs), want) < 1e-12);
    // symmetric for distinct names
    assert_eq!(
        interfacial_bond_strength_mpa(&pla, &pc),
        interfacial_bond_strength_mpa(&pc, &pla)
    );
    assert!(
        rel(
            interfacial_bond_strength_mpa(&pla, &pc),
            0.5 * (50.0f64 * 65.0).sqrt()
        ) < 1e-12
    );
}

#[test]
fn dissimilar_bond_is_weaker_than_the_same_material_bond_of_either_partner() {
    let (pla, petg) = (MaterialProperties::pla(), MaterialProperties::petg());
    let mixed = interfacial_bond_strength_mpa(&pla, &petg);
    assert!(mixed < interfacial_bond_strength_mpa(&pla, &pla));
    assert!(mixed < interfacial_bond_strength_mpa(&petg, &petg));
}

#[test]
fn bond_strength_is_symmetric_for_equal_names_with_different_yield() {
    // AUD-A-S4W3-011: the same-name branch returned a's yield, so the order mattered
    let mut weak = MaterialProperties::pla();
    weak.yield_strength_mpa = Fix128::from_int(40);
    let strong = MaterialProperties::pla();
    assert_eq!(
        interfacial_bond_strength_mpa(&weak, &strong),
        interfacial_bond_strength_mpa(&strong, &weak)
    );
    // a perfect bond is as strong as the weaker partner
    assert_eq!(
        interfacial_bond_strength_mpa(&strong, &weak),
        Fix128::from_int(40)
    );
    // identical materials keep their full yield
    assert_eq!(
        interfacial_bond_strength_mpa(&strong, &strong),
        strong.yield_strength_mpa
    );
}

#[test]
#[ignore = "known defect: AUD-A-S4W3-012: design question: a dissimilar bond (half the geometric mean of the yields) can exceed the weaker partner's yield strength (PLA 50 MPa with SUS304 215 MPa gives 51.8), so the bond-line is rated stronger than the weaker material itself"]
fn dissimilar_bond_never_exceeds_the_weaker_partners_yield() {
    let (pla, sus) = (MaterialProperties::pla(), MaterialProperties::sus304());
    let bond = interfacial_bond_strength_mpa(&pla, &sus);
    assert!(
        bond <= pla.yield_strength_mpa,
        "bond {} > PLA yield 50",
        bond.to_f64()
    );
}

// ------------------------------------------------------------- analyze_bimaterial

#[test]
fn report_fields_are_the_component_results() {
    let (a, b) = (
        side(MaterialProperties::petg(), 1.2),
        side(MaterialProperties::abs(), 0.8),
    );
    let (th, tc, tau) = (fx(230.0), fx(20.0), fx(-3.0));
    let r = analyze_bimaterial(&a, &b, th, tc, tau);
    assert_eq!(
        r.effective_modulus_voigt_mpa,
        effective_modulus_voigt_mpa(&a, &b)
    );
    assert_eq!(
        r.effective_modulus_reuss_mpa,
        effective_modulus_reuss_mpa(&a, &b)
    );
    assert_eq!(
        r.thermal_residual_mpa,
        thermal_residual_stress_mpa(&a, &b, th, tc)
    );
    assert_eq!(
        r.bond_strength_mpa,
        interfacial_bond_strength_mpa(&a.material, &b.material)
    );
    // |residual| + |tau| (the applied shear's sign is dropped)
    let want_total = r.thermal_residual_mpa.to_f64().abs() + 3.0;
    assert!(rel(r.total_interfacial_stress_mpa, want_total) < 1e-12);
    assert!(
        rel(
            r.factor_of_safety,
            r.bond_strength_mpa.to_f64() / want_total
        ) < 1e-12
    );
    // the sign of the applied shear does not matter
    let r2 = analyze_bimaterial(&a, &b, th, tc, fx(3.0));
    assert_eq!(
        r2.total_interfacial_stress_mpa,
        r.total_interfacial_stress_mpa
    );
    // swapping the pair reports layer b's stress, which balances layer a's
    // force: sigma_b t_b = -sigma_a t_a, so |sigma_b| = |sigma_a| * 1.2 / 0.8
    let r3 = analyze_bimaterial(&b, &a, th, tc, fx(3.0));
    let sa = r.thermal_residual_mpa.to_f64();
    let sb = r3.thermal_residual_mpa.to_f64();
    assert!((sa * 1.2 + sb * 0.8).abs() < 1e-9 * sa.abs(), "{sa} {sb}");
    assert!(rel(r3.total_interfacial_stress_mpa, sb.abs() + 3.0) < 1e-12);
}

#[test]
fn safety_threshold_is_a_factor_of_safety_of_two_inclusive() {
    // same material: no residual stress, bond = yield (50 MPa)
    let p = side(MaterialProperties::pla(), 1.0);
    let at = analyze_bimaterial(&p, &p, fx(200.0), fx(20.0), fx(25.0));
    assert_eq!(at.thermal_residual_mpa, Fix128::ZERO);
    assert!(rel(at.factor_of_safety, 2.0) < 1e-12);
    assert!(at.is_safe, "FoS exactly 2 is safe");
    let over = analyze_bimaterial(&p, &p, fx(200.0), fx(20.0), fx(25.01));
    assert!(!over.is_safe);
    assert!(over.factor_of_safety < Fix128::from_int(2));
}

#[test]
fn zero_stress_is_safe_with_a_very_large_factor_of_safety() {
    let p = side(MaterialProperties::pla(), 1.0);
    let r = analyze_bimaterial(&p, &p, fx(20.0), fx(20.0), Fix128::ZERO);
    assert!(r.is_safe);
    assert!(r.factor_of_safety > Fix128::from_int(1_000_000));
}

#[test]
fn zero_bond_strength_under_load_has_zero_factor_of_safety() {
    let mut weak = MaterialProperties::pla();
    weak.yield_strength_mpa = Fix128::ZERO;
    let p = side(weak, 1.0);
    let r = analyze_bimaterial(&p, &p, fx(200.0), fx(20.0), fx(5.0));
    assert_eq!(r.factor_of_safety, Fix128::ZERO);
    assert!(!r.is_safe);
}
