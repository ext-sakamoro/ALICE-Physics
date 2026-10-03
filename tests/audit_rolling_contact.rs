//! Audit oracles for rolling_contact
//!
//! 期待値は Hertz 点接触の教科書式 (Johnson, Contact Mechanics ch.4) を f64 で独立に評価したもの
//! a = (3 P R* / 4 E*)^(1/3),  p0 = (6 P E*^2 / (pi^3 R*^2))^(1/3) (a を経由しない形)

#![allow(clippy::disallowed_methods)]

use alice_physics::rolling_contact::{
    basquin_cycles_to_failure, hertzian_sphere_sphere, materials, rolling_contact_life_cycles,
};

fn ref_hertz(p: f64, r1: f64, r2: f64, e1: f64, e2: f64, n1: f64, n2: f64) -> (f64, f64, f64) {
    let r_star = if r1.is_infinite() {
        r2
    } else if r2.is_infinite() {
        r1
    } else {
        1.0 / (1.0 / r1 + 1.0 / r2)
    };
    let e_star = 1.0 / ((1.0 - n1 * n1) / e1 + (1.0 - n2 * n2) / e2);
    let a = (3.0 * p * r_star / (4.0 * e_star)).powf(1.0 / 3.0);
    let pi = std::f64::consts::PI;
    let p0 = (6.0 * p * e_star * e_star / (pi * pi * pi * r_star * r_star)).powf(1.0 / 3.0);
    (a, p0, 0.48 * a)
}

fn rel(a: f32, b: f64) -> f64 {
    ((f64::from(a) - b) / b).abs()
}

type Case = (f32, f32, f32, f32, f32, f32, f32);

fn cases() -> Vec<Case> {
    vec![
        // load, R1, R2, E1, E2, nu1, nu2
        (1000.0, 0.01, 0.01, 210e9, 210e9, 0.3, 0.3),
        (250.0, 0.0125, 0.05, 210e9, 310e9, 0.3, 0.27),
        (5.0e4, 0.2, 0.35, 205e9, 70e9, 0.29, 0.33),
        (12.0, 0.003, 0.0007, 3.5e9, 2.0e9, 0.36, 0.4),
        (1.0e5, 0.5, f32::INFINITY, 210e9, 210e9, 0.3, 0.3),
        (80.0, f32::INFINITY, 0.02, 210e9, 310e9, 0.3, 0.27),
        (800.0, 0.04, 0.08, 120e9, 45e9, 0.0, 0.45),
    ]
}

#[test]
fn hertz_matches_textbook_closed_form_on_cases() {
    for (p, r1, r2, e1, e2, n1, n2) in cases() {
        let c = hertzian_sphere_sphere(p, r1, r2, e1, e2, n1, n2);
        let (a, p0, z) = ref_hertz(
            f64::from(p),
            f64::from(r1),
            f64::from(r2),
            f64::from(e1),
            f64::from(e2),
            f64::from(n1),
            f64::from(n2),
        );
        assert!(rel(c.contact_radius_m, a) < 5e-5, "a {c:?} want {a}");
        assert!(rel(c.peak_pressure_pa, p0) < 2e-4, "p0 {c:?} want {p0}");
        assert!(rel(c.max_shear_depth_m, z) < 5e-5, "z {c:?} want {z}");
    }
}

#[test]
fn hertz_scaling_laws() {
    // a ∝ P^(1/3), p0 ∝ P^(1/3), a ∝ R*^(1/3), p0 ∝ R*^(-2/3), a ∝ E*^(-1/3)
    let base = hertzian_sphere_sphere(100.0, 0.02, 0.02, 200e9, 200e9, 0.3, 0.3);
    let p8 = hertzian_sphere_sphere(800.0, 0.02, 0.02, 200e9, 200e9, 0.3, 0.3);
    assert!(rel(p8.contact_radius_m, f64::from(base.contact_radius_m) * 2.0) < 1e-4);
    assert!(rel(p8.peak_pressure_pa, f64::from(base.peak_pressure_pa) * 2.0) < 1e-4);
    // R* を 8 倍 (両半径 8 倍): a は 2 倍、p0 は 1/4
    let r8 = hertzian_sphere_sphere(100.0, 0.16, 0.16, 200e9, 200e9, 0.3, 0.3);
    assert!(rel(r8.contact_radius_m, f64::from(base.contact_radius_m) * 2.0) < 1e-4);
    assert!(rel(r8.peak_pressure_pa, f64::from(base.peak_pressure_pa) * 0.25) < 1e-4);
    // E を 8 倍: a は 1/2
    let e8 = hertzian_sphere_sphere(100.0, 0.02, 0.02, 1600e9, 1600e9, 0.3, 0.3);
    assert!(rel(e8.contact_radius_m, f64::from(base.contact_radius_m) * 0.5) < 1e-4);
}

#[test]
fn hertz_is_symmetric_under_swapping_the_two_bodies() {
    for (p, r1, r2, e1, e2, n1, n2) in cases() {
        let ab = hertzian_sphere_sphere(p, r1, r2, e1, e2, n1, n2);
        let ba = hertzian_sphere_sphere(p, r2, r1, e2, e1, n2, n1);
        assert!(rel(ab.contact_radius_m, f64::from(ba.contact_radius_m)) < 1e-5);
        assert!(rel(ab.peak_pressure_pa, f64::from(ba.peak_pressure_pa)) < 1e-5);
    }
}

#[test]
fn hertz_poisson_ratio_enters_through_one_minus_nu_squared_per_body() {
    // ν1 だけ 0 -> 0.5 に変えると 1/E* が (1 - 0.25)/E1 に変わる。非対称 (E1 != E2) で body ごとの対応を縛る
    let a = hertzian_sphere_sphere(500.0, 0.01, 0.01, 100e9, 400e9, 0.0, 0.0);
    let b = hertzian_sphere_sphere(500.0, 0.01, 0.01, 100e9, 400e9, 0.5, 0.0);
    let c = hertzian_sphere_sphere(500.0, 0.01, 0.01, 100e9, 400e9, 0.0, 0.5);
    let es = |n1: f64, n2: f64| 1.0 / ((1.0 - n1 * n1) / 100e9 + (1.0 - n2 * n2) / 400e9);
    let want = |e: f64| (3.0 * 500.0 * 0.005 / (4.0 * e)).powf(1.0 / 3.0);
    assert!(rel(a.contact_radius_m, want(es(0.0, 0.0))) < 5e-5);
    assert!(rel(b.contact_radius_m, want(es(0.5, 0.0))) < 5e-5);
    assert!(rel(c.contact_radius_m, want(es(0.0, 0.5))) < 5e-5);
}

#[test]
fn hertz_shear_depth_is_exactly_0_48_of_contact_radius() {
    for (p, r1, r2, e1, e2, n1, n2) in cases() {
        let c = hertzian_sphere_sphere(p, r1, r2, e1, e2, n1, n2);
        assert_eq!(c.max_shear_depth_m, 0.48 * c.contact_radius_m);
    }
}

#[test]
fn hertz_flat_surface_infinite_radius_equals_a_huge_finite_radius() {
    let flat = hertzian_sphere_sphere(300.0, 0.015, f32::INFINITY, 210e9, 210e9, 0.3, 0.3);
    let big = hertzian_sphere_sphere(300.0, 0.015, 1.0e12, 210e9, 210e9, 0.3, 0.3);
    assert!(rel(flat.contact_radius_m, f64::from(big.contact_radius_m)) < 1e-4);
    assert!(rel(flat.peak_pressure_pa, f64::from(big.peak_pressure_pa)) < 1e-4);
}

#[test]
fn hertz_documented_panics_are_each_triggered() {
    let ok = (100.0f32, 0.01f32, 0.01f32, 200e9f32, 200e9f32);
    let call = |l: f32, r1: f32, r2: f32, e1: f32, e2: f32| {
        std::panic::catch_unwind(|| hertzian_sphere_sphere(l, r1, r2, e1, e2, 0.3, 0.3)).is_err()
    };
    assert!(!call(ok.0, ok.1, ok.2, ok.3, ok.4));
    for bad in [0.0f32, -1.0, f32::NAN] {
        assert!(call(bad, ok.1, ok.2, ok.3, ok.4), "load {bad}");
        assert!(call(ok.0, bad, ok.2, ok.3, ok.4), "r1 {bad}");
        assert!(call(ok.0, ok.1, bad, ok.3, ok.4), "r2 {bad}");
        assert!(call(ok.0, ok.1, ok.2, bad, ok.4), "e1 {bad}");
        assert!(call(ok.0, ok.1, ok.2, ok.3, bad), "e2 {bad}");
    }
}

#[test]
fn basquin_matches_power_law_and_boundary_is_inclusive() {
    let (c, m, se) = (5.0e34f32, 3.0f32, 1.5e9f32);
    for s in [1.6e9f32, 2.0e9, 3.0e9, 4.5e9] {
        let n = basquin_cycles_to_failure(s, c, m, se);
        let want = f64::from(c) * f64::from(s).powf(-f64::from(m));
        assert!(rel(n, want) < 5e-4, "s {s}: {n} vs {want}");
    }
    assert!(basquin_cycles_to_failure(se, c, m, se).is_infinite());
    assert!(basquin_cycles_to_failure(0.0, c, m, se).is_infinite());
    assert!(basquin_cycles_to_failure(f32::from_bits(se.to_bits() + 1), c, m, se).is_finite());
    // 2 倍の応力で寿命は 2^-m 倍
    let n1 = basquin_cycles_to_failure(2.0e9, c, m, se);
    let n2 = basquin_cycles_to_failure(4.0e9, c, m, se);
    assert!(rel(n2, f64::from(n1) / 8.0) < 1e-4);
    // exponent を非整数にしても同じ式
    let n = basquin_cycles_to_failure(2.0e9, 1.0e38, 3.5, 1.0e9);
    let want = 1.0e38f64 * 2.0e9f64.powf(-3.5);
    assert!(rel(n, want) < 1e-3, "{n} vs {want}");
}

#[test]
fn basquin_documented_panics() {
    let call = |s: f32, c: f32, m: f32| {
        std::panic::catch_unwind(|| basquin_cycles_to_failure(s, c, m, 1.0e9)).is_err()
    };
    assert!(!call(2.0e9, 5.0e34, 3.0));
    assert!(call(-1.0, 5.0e34, 3.0), "negative stress");
    assert!(call(2.0e9, 0.0, 3.0), "zero C");
    assert!(call(2.0e9, -1.0, 3.0), "negative C");
    assert!(call(2.0e9, 5.0e34, 0.0), "zero m");
    assert!(call(2.0e9, 5.0e34, -3.0), "negative m");
    assert!(call(f32::NAN, 5.0e34, 3.0), "NaN stress");
    // 応力 0 は許容 (panic しない)
    assert!(!call(0.0, 5.0e34, 3.0));
}

#[test]
fn combined_life_equals_basquin_of_hertz_peak_pressure_with_asymmetric_materials() {
    let (p, r1, r2, e1, e2, n1, n2) = (
        1500.0f32, 0.012f32, 0.03f32, 210e9f32, 310e9f32, 0.3f32, 0.27f32,
    );
    let (c, m, se) = (8.0e34f32, 3.5f32, 1.0e9f32);
    let contact = hertzian_sphere_sphere(p, r1, r2, e1, e2, n1, n2);
    let want = basquin_cycles_to_failure(contact.peak_pressure_pa, c, m, se);
    let got = rolling_contact_life_cycles(p, r1, r2, e1, e2, n1, n2, c, m, se);
    assert_eq!(got.to_bits(), want.to_bits());
    assert!(
        got.is_finite() && got > 0.0,
        "{got} (p0 = {})",
        contact.peak_pressure_pa
    );
}

#[test]
fn material_presets_match_documented_catalogue_values() {
    // doc コメントの値: 52100 = 210 GPa, 0.30 / 8620 = 205 GPa, 0.29 / Si3N4 = 310 GPa, 0.27
    let (e, nu, c, m, se) = materials::bearing_steel_52100();
    assert_eq!((e, nu, c, m, se), (210.0e9, 0.30, 5.0e34, 3.0, 1.5e9));
    let (e, nu, c, m, se) = materials::gear_steel_8620();
    assert_eq!((e, nu, c, m, se), (205.0e9, 0.29, 2.5e34, 3.2, 1.3e9));
    let (e, nu, c, m, se) = materials::silicon_nitride();
    assert_eq!((e, nu, c, m, se), (310.0e9, 0.27, 8.0e34, 3.5, 2.0e9));
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-008: doc の m = 9-10 (高強度合金) は Pa / f32 では表せない (C = N0 s0^9 = 3.8e60 > f32::MAX、f32::MAX でも N(3 GPa) = 0)"]
fn doc_exponent_range_9_to_10_is_expressible_in_pa_f32() {
    // doc: "m ... 9-10 for high-strength alloys"。C = N0 * s0^m (N0 = 1e6, s0 = 1.5 GPa, m = 9) は 3.8e60 で f32 (最大 3.4e38) に載らない
    // 載る最大の C = f32::MAX でも 3 GPa での寿命が 1 cycle を割る (=常に 0 cycle と答える)
    let n = basquin_cycles_to_failure(3.0e9, f32::MAX, 9.0, 1.5e9);
    assert!(n >= 1.0, "N(3 GPa, C = f32::MAX, m = 9) = {n}");
}

#[test]
#[ignore = "known defect: AUD-A-S4W2-008: m = 5 でも 3 GPa で N < 1 (f32::MAX の C でも 0 cycle)"]
fn doc_exponent_5_is_also_inexpressible_at_hertz_pressures() {
    // 同じ理由で m = 5 でも 3 GPa で N < 1 (m = 3 の preset だけが使える)
    let n = basquin_cycles_to_failure(3.0e9, f32::MAX, 5.0, 1.5e9);
    assert!(n >= 1.0, "N(3 GPa, C = f32::MAX, m = 5) = {n}");
}
