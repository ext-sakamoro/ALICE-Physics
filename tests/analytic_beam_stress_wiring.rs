//! Independent oracles for `beam_stress::BeamAnalysis::{with_end_condition,
//! with_min_fos}` and the module-doc claims they sit next to: the Euler load
//! `P_cr = pi^2 E I / (K L)^2` with the four end-condition factors, the
//! factor-of-safety boundary, and the section / load-case closed forms they
//! are computed from.
//!
//! Every expected value is derived here in `f64` from Roark (Table 8.1 and
//! Appendix A) and Timoshenko & Gere, never by calling the function under
//! test on the expected side. Material: PLA (E = 3.5 GPa, yield 50 MPa).
//!
//! Nothing here touches `src/`.

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{BeamAnalysis, ColumnEndCondition, CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use core::f64::consts::PI;

const E_MPA: f64 = 3500.0;
const YIELD_MPA: f64 = 50.0;
/// Fix128 carries 64 fractional bits; every formula here chains a handful of
/// multiplications, so 1e-12 relative is conservative.
const TOL: f64 = 1e-12;

fn int(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn rect(w: i64, h: i64) -> CrossSection {
    CrossSection::Rectangular {
        width_mm: int(w),
        height_mm: int(h),
    }
}

fn rel(got: Fix128, want: f64) -> f64 {
    ((got.to_f64() - want) / want).abs()
}

fn assert_rel(got: Fix128, want: f64, what: &str) {
    let e = rel(got, want);
    assert!(
        e <= TOL,
        "{what}: got {}, want {want} (rel {e:.3e})",
        got.to_f64()
    );
}

fn pla() -> MaterialProperties {
    MaterialProperties::pla()
}

// ---------------------------------------------------------------------------
// with_end_condition
// ---------------------------------------------------------------------------

/// Timoshenko & Gere Table 2-1 end-condition factors K = 1, 1/2, 2, 0.7 give
/// `P_cr = pi^2 E I / (K L)^2`. For a 10x20 mm PLA beam of length 200 mm:
/// Buckling uses the weak-axis `I = h b^3 / 12 = 1666.667 mm^4`, so
/// `P_cr(K=1) = 1439.3 N` (a quarter of the strong-axis value).
#[test]
fn euler_load_follows_each_end_condition_factor() {
    let load = LoadCase::CantileverEndPoint {
        load_n: int(5),
        length_mm: int(200),
    };
    // buckling is about the weak axis (the 10 mm side): I = h b^3 / 12
    let i = 20.0 * 10.0f64 * 10.0 * 10.0 / 12.0;
    let base = PI * PI * E_MPA * i / (200.0 * 200.0);
    let cases = [
        (ColumnEndCondition::PinPin, 1.0),
        (ColumnEndCondition::FixedFixed, 0.5),
        (ColumnEndCondition::Cantilever, 2.0),
        (ColumnEndCondition::FixedPin, 0.7),
    ];
    for (ec, k) in cases {
        let r = BeamAnalysis::new(rect(10, 20), load, pla())
            .with_end_condition(ec)
            .analyze();
        assert_rel(
            r.euler_critical_load_n,
            base / (k * k),
            &format!("{ec:?} K={k}"),
        );
    }
    // the K factors themselves
    assert_eq!(ColumnEndCondition::PinPin.k_factor(), Fix128::ONE);
    assert_eq!(
        ColumnEndCondition::FixedFixed.k_factor(),
        Fix128::from_ratio(1, 2)
    );
    assert_eq!(ColumnEndCondition::Cantilever.k_factor(), int(2));
    assert_eq!(
        ColumnEndCondition::FixedPin.k_factor(),
        Fix128::from_ratio(7, 10)
    );
}

/// Buckling and bending are independent in the report: swapping the end
/// condition changes `euler_critical_load_n` and nothing else, and the
/// builder returns a configured copy (the original keeps pin-pin).
#[test]
fn end_condition_changes_only_the_euler_load() {
    let load = LoadCase::SimplySupportedCenter {
        load_n: int(40),
        length_mm: int(150),
    };
    let base = BeamAnalysis::new(rect(12, 16), load, pla());
    let r0 = base.analyze();
    let tuned = base.with_end_condition(ColumnEndCondition::FixedFixed);
    let r1 = tuned.analyze();
    assert_eq!(r1.max_bending_moment_nmm, r0.max_bending_moment_nmm);
    assert_eq!(r1.max_bending_stress_mpa, r0.max_bending_stress_mpa);
    assert_eq!(r1.max_deflection_mm, r0.max_deflection_mm);
    assert_eq!(r1.factor_of_safety, r0.factor_of_safety);
    assert_eq!(r1.is_safe, r0.is_safe);
    assert_ne!(r1.euler_critical_load_n, r0.euler_critical_load_n);
    // fixed-fixed carries 4x the pin-pin load (K = 1/2), to Fix128 rounding
    assert!(
        rel(
            r1.euler_critical_load_n,
            4.0 * r0.euler_critical_load_n.to_f64()
        ) < TOL
    );
    // `with_end_condition` consumed a copy; the original is unchanged
    assert_eq!(base.end_condition, ColumnEndCondition::PinPin);
    assert_eq!(tuned.end_condition, ColumnEndCondition::FixedFixed);
    // and the setter does not touch the FoS threshold
    assert_eq!(tuned.min_factor_of_safety, base.min_factor_of_safety);
}

/// Defaults documented on `BeamAnalysis::new`: pin-pin ends, minimum FoS 2.0
/// (ASME B31 practice).
#[test]
fn new_defaults_to_pin_pin_and_fos_two() {
    let b = BeamAnalysis::new(
        rect(10, 10),
        LoadCase::CantileverEndPoint {
            load_n: int(1),
            length_mm: int(10),
        },
        pla(),
    );
    assert_eq!(b.end_condition, ColumnEndCondition::PinPin);
    assert_eq!(b.min_factor_of_safety, int(2));
}

// ---------------------------------------------------------------------------
// with_min_fos
// ---------------------------------------------------------------------------

/// `is_safe = FoS >= min_fos`, exactly at the boundary. A 15x20 mm section has
/// `Z = b h^2 / 6 = 1000 mm^3`; a 100 N tip load on 250 mm gives
/// `sigma = 25000 / 1000 = 25 MPa` and `FoS = 50 / 25 = 2` exactly, so the
/// default threshold 2.0 is met with equality.
#[test]
fn factor_of_safety_equal_to_the_threshold_is_safe() {
    let load = LoadCase::CantileverEndPoint {
        load_n: int(100),
        length_mm: int(250),
    };
    let section = rect(15, 20);
    // I = 15 * 8000 / 12 = 10000 exactly (an integer: no 1/12 rounding residue)
    assert_eq!(section.second_moment_of_area_mm4(), int(10_000));
    assert_eq!(section.section_modulus_mm3(), int(1000));
    let r = BeamAnalysis::new(section, load, pla()).analyze();
    assert_eq!(r.factor_of_safety, int(2));
    assert_rel(r.max_bending_stress_mpa, 25.0, "sigma = M/Z");
    assert_rel(r.factor_of_safety, 2.0, "FoS = 50/25");
    assert!(r.is_safe, "FoS == min_fos (2.0) is safe");

    // one ulp above the achieved FoS: now unsafe
    let ulp = Fix128::from_raw(0, 1);
    let tight = BeamAnalysis::new(section, load, pla())
        .with_min_fos(r.factor_of_safety + ulp)
        .analyze();
    assert!(!tight.is_safe);
    // exactly the achieved FoS: safe
    let exact = BeamAnalysis::new(section, load, pla())
        .with_min_fos(r.factor_of_safety)
        .analyze();
    assert!(exact.is_safe);
}

/// The threshold moves the verdict and only the verdict: for the same beam
/// (FoS = 2) thresholds 1.5 / 2.0 pass, 2.5 / 10 fail, and every numeric field
/// is identical across thresholds.
#[test]
fn min_fos_changes_only_the_verdict() {
    let load = LoadCase::CantileverEndPoint {
        load_n: int(100),
        length_mm: int(250),
    };
    let base = BeamAnalysis::new(rect(15, 20), load, pla());
    let reference = base.analyze();
    for (min, safe) in [
        (Fix128::from_ratio(3, 2), true),
        (int(2), true),
        (Fix128::from_ratio(5, 2), false),
        (int(10), false),
        (Fix128::ZERO, true),
    ] {
        let tuned = base.with_min_fos(min);
        assert_eq!(tuned.min_factor_of_safety, min);
        let r = tuned.analyze();
        assert_eq!(r.is_safe, safe, "min_fos {}", min.to_f64());
        assert_eq!(r.max_bending_moment_nmm, reference.max_bending_moment_nmm);
        assert_eq!(r.max_bending_stress_mpa, reference.max_bending_stress_mpa);
        assert_eq!(r.max_deflection_mm, reference.max_deflection_mm);
        assert_eq!(r.euler_critical_load_n, reference.euler_critical_load_n);
        assert_eq!(r.factor_of_safety, reference.factor_of_safety);
    }
    // the base keeps its own threshold
    assert_eq!(base.min_factor_of_safety, int(2));
}

/// An overloaded beam (stress 720 MPa against yield 50) has FoS 0.069: unsafe
/// at every threshold above that and (only) safe at threshold 0, and its FoS
/// is `yield / stress`, not a clamp.
#[test]
fn overloaded_beam_fos_is_yield_over_stress() {
    let load = LoadCase::CantileverEndPoint {
        load_n: int(100),
        length_mm: int(300),
    };
    let section = rect(10, 5); // Z = 10*25/6 = 41.667
    let z = 10.0 * 25.0 / 6.0;
    let sigma = 100.0 * 300.0 / z;
    let r = BeamAnalysis::new(section, load, pla()).analyze();
    assert_rel(r.max_bending_stress_mpa, sigma, "sigma");
    assert_rel(r.factor_of_safety, YIELD_MPA / sigma, "FoS");
    assert!(!r.is_safe);
    assert!(
        !BeamAnalysis::new(section, load, pla())
            .with_min_fos(Fix128::from_ratio(1, 10))
            .analyze()
            .is_safe
    );
}

/// No load: zero stress, FoS is the `i64::MAX >> 8` "infinite" sentinel, safe
/// at any finite threshold, zero moment and zero deflection.
#[test]
fn zero_load_has_sentinel_fos_and_is_safe() {
    let load = LoadCase::CantileverEndPoint {
        load_n: Fix128::ZERO,
        length_mm: int(200),
    };
    let r = BeamAnalysis::new(rect(10, 20), load, pla())
        .with_min_fos(int(1_000_000))
        .analyze();
    assert_eq!(r.max_bending_moment_nmm, Fix128::ZERO);
    assert_eq!(r.max_bending_stress_mpa, Fix128::ZERO);
    assert_eq!(r.max_deflection_mm, Fix128::ZERO);
    assert_eq!(r.factor_of_safety, Fix128::from_int(i64::MAX >> 8));
    assert!(r.is_safe);
}

// ---------------------------------------------------------------------------
// analyze: moment / stress / deflection for all four load cases
// ---------------------------------------------------------------------------

/// Roark Table 8.1: M and delta for the four load cases on the same section,
/// through `analyze` (including the GPa -> MPa conversion of E).
/// Section 20x30: `I = 20*27000/12 = 45000 mm^4`, `Z = I/15 = 3000 mm^3`.
#[test]
fn analyze_matches_roark_for_all_four_load_cases() {
    let section = rect(20, 30);
    let i = 20.0 * 30.0f64.powi(3) / 12.0;
    let z = i / 15.0;
    let ei = E_MPA * i;
    let (p, w, l) = (120.0f64, 0.5f64, 400.0f64);
    let cases: [(LoadCase, f64, f64, &str); 4] = [
        (
            LoadCase::CantileverEndPoint {
                load_n: int(120),
                length_mm: int(400),
            },
            p * l,
            p * l.powi(3) / (3.0 * ei),
            "cantilever end point",
        ),
        (
            LoadCase::CantileverDistributed {
                load_per_mm_n: Fix128::from_ratio(1, 2),
                length_mm: int(400),
            },
            w * l * l / 2.0,
            w * l.powi(4) / (8.0 * ei),
            "cantilever distributed",
        ),
        (
            LoadCase::SimplySupportedCenter {
                load_n: int(120),
                length_mm: int(400),
            },
            p * l / 4.0,
            p * l.powi(3) / (48.0 * ei),
            "simply supported centre",
        ),
        (
            LoadCase::SimplySupportedDistributed {
                load_per_mm_n: Fix128::from_ratio(1, 2),
                length_mm: int(400),
            },
            w * l * l / 8.0,
            5.0 * w * l.powi(4) / (384.0 * ei),
            "simply supported distributed",
        ),
    ];
    for (load, m, d, name) in cases {
        let r = BeamAnalysis::new(section, load, pla()).analyze();
        assert_rel(r.max_bending_moment_nmm, m, name);
        assert_rel(r.max_bending_stress_mpa, m / z, name);
        assert_rel(r.max_deflection_mm, d, name);
        assert_rel(r.factor_of_safety, YIELD_MPA / (m / z), name);
        assert_eq!(load.length_mm(), int(400), "{name}: span accessor");
        assert_eq!(r.is_safe, YIELD_MPA / (m / z) >= 2.0, "{name}");
    }
}

/// Section properties of all five cross-sections against Roark Appendix A, and
/// `Z = I / c` with `c` the extreme-fibre distance.
#[test]
fn section_properties_match_roark_appendix_a() {
    let rectangle = rect(14, 22);
    assert_rel(rectangle.area_mm2(), 14.0 * 22.0, "rect A");
    assert_rel(
        rectangle.second_moment_of_area_mm4(),
        14.0 * 22.0f64.powi(3) / 12.0,
        "rect I",
    );
    assert_eq!(rectangle.max_c_mm(), int(11));
    assert_rel(
        rectangle.section_modulus_mm3(),
        14.0 * 22.0f64.powi(3) / 12.0 / 11.0,
        "rect Z",
    );

    let circle = CrossSection::Circular {
        diameter_mm: int(18),
    };
    assert_rel(circle.area_mm2(), PI * 18.0 * 18.0 / 4.0, "circle A");
    assert_rel(
        circle.second_moment_of_area_mm4(),
        PI * 18.0f64.powi(4) / 64.0,
        "circle I",
    );
    assert_eq!(circle.max_c_mm(), int(9));
    assert_rel(
        circle.section_modulus_mm3(),
        PI * 18.0f64.powi(4) / 64.0 / 9.0,
        "circle Z",
    );

    let box_section = CrossSection::HollowRectangular {
        outer_width_mm: int(30),
        outer_height_mm: int(40),
        wall_mm: int(3),
    };
    assert_rel(box_section.area_mm2(), 30.0 * 40.0 - 24.0 * 34.0, "box A");
    assert_rel(
        box_section.second_moment_of_area_mm4(),
        (30.0 * 40.0f64.powi(3) - 24.0 * 34.0f64.powi(3)) / 12.0,
        "box I",
    );
    assert_eq!(box_section.max_c_mm(), int(20));
    assert_rel(
        box_section.section_modulus_mm3(),
        (30.0 * 40.0f64.powi(3) - 24.0 * 34.0f64.powi(3)) / 12.0 / 20.0,
        "box Z",
    );

    let tube = CrossSection::HollowCircular {
        outer_diameter_mm: int(20),
        inner_diameter_mm: int(14),
    };
    assert_rel(tube.area_mm2(), PI * (400.0 - 196.0) / 4.0, "tube A");
    assert_rel(
        tube.second_moment_of_area_mm4(),
        PI * (20.0f64.powi(4) - 14.0f64.powi(4)) / 64.0,
        "tube I",
    );
    assert_eq!(tube.max_c_mm(), int(10));

    // I-beam: two flanges (parallel axis) + web, Roark A.1 case 8
    let ibeam = CrossSection::IBeam {
        flange_width_mm: int(60),
        height_mm: int(100),
        flange_thickness_mm: int(8),
        web_thickness_mm: int(5),
    };
    let (b, h, tf, tw) = (60.0f64, 100.0f64, 8.0f64, 5.0f64);
    let flange = b * tf.powi(3) / 12.0 + b * tf * ((h - tf) / 2.0).powi(2);
    let web = tw * (h - 2.0 * tf).powi(3) / 12.0;
    assert_rel(
        ibeam.area_mm2(),
        2.0 * b * tf + tw * (h - 2.0 * tf),
        "ibeam A",
    );
    assert_rel(
        ibeam.second_moment_of_area_mm4(),
        2.0 * flange + web,
        "ibeam I",
    );
    assert_eq!(ibeam.max_c_mm(), int(50));
    assert_rel(
        ibeam.section_modulus_mm3(),
        (2.0 * flange + web) / 50.0,
        "ibeam Z",
    );
}

/// Buckling load scales as `1/L^2`: doubling the length quarters `P_cr`, and
/// a zero-length column reports 0 rather than dividing by zero.
#[test]
fn euler_load_scales_inverse_square_with_length_and_zero_length_is_zero() {
    let at = |len: i64| {
        BeamAnalysis::new(
            rect(10, 20),
            LoadCase::CantileverEndPoint {
                load_n: int(1),
                length_mm: int(len),
            },
            pla(),
        )
        .analyze()
    };
    let p100 = at(100).euler_critical_load_n.to_f64();
    let p200 = at(200).euler_critical_load_n.to_f64();
    let p400 = at(400).euler_critical_load_n.to_f64();
    assert!((p100 / p200 - 4.0).abs() < 1e-9);
    assert!((p200 / p400 - 4.0).abs() < 1e-9);
    assert_eq!(at(0).euler_critical_load_n, Fix128::ZERO);
}

/// Integer-valued sections have integer `I` and `Z`; the second moment is
/// `b h^3 / 12`, divided exactly (not multiplied by a rounded `1/12`), so a
/// stress of exactly 25 MPa gives `FoS = 50 / 25 = 2` exactly and the default
/// threshold 2.0 is met with equality.
///
/// * hollow 30x40, wall 5: `I = (30*64000 - 20*27000)/12 = 115000`,
///   `Z = I/20 = 5750`, `M = 25 * 5750 = 143750` (575 N at 250 mm)
/// * I-beam 60x100, flanges 10, web 6: `I = (60e6 - 54*512000)/12 = 2696000`,
///   `Z = I/50 = 53920`, `M = 1348000` (1348 N at 1000 mm)
#[test]
fn hollow_and_ibeam_sections_reach_fos_two_exactly() {
    let hollow = CrossSection::HollowRectangular {
        outer_width_mm: int(30),
        outer_height_mm: int(40),
        wall_mm: int(5),
    };
    assert_eq!(hollow.second_moment_of_area_mm4(), int(115_000));
    assert_eq!(hollow.section_modulus_mm3(), int(5750));
    let r = BeamAnalysis::new(
        hollow,
        LoadCase::CantileverEndPoint {
            load_n: int(575),
            length_mm: int(250),
        },
        pla(),
    )
    .analyze();
    assert_eq!(r.max_bending_stress_mpa, int(25));
    assert_eq!(r.factor_of_safety, int(2));
    assert!(r.is_safe);

    let ibeam = CrossSection::IBeam {
        flange_width_mm: int(60),
        height_mm: int(100),
        flange_thickness_mm: int(10),
        web_thickness_mm: int(6),
    };
    assert_eq!(ibeam.second_moment_of_area_mm4(), int(2_696_000));
    assert_eq!(ibeam.section_modulus_mm3(), int(53_920));
    let r = BeamAnalysis::new(
        ibeam,
        LoadCase::CantileverEndPoint {
            load_n: int(1348),
            length_mm: int(1000),
        },
        pla(),
    )
    .analyze();
    assert_eq!(r.max_bending_stress_mpa, int(25));
    assert_eq!(r.factor_of_safety, int(2));
    assert!(r.is_safe);
}

/// A material with zero stiffness has no defined deflection; `max_deflection_mm`
/// returns 0 for `E I = 0` (documented guard), while the stress and the
/// Euler load follow their own formulas (Euler load is 0 since `E = 0`).
#[test]
fn zero_modulus_material_reports_zero_deflection_and_zero_euler_load() {
    let mut m = pla();
    m.youngs_modulus_gpa = Fix128::ZERO;
    let r = BeamAnalysis::new(
        rect(10, 20),
        LoadCase::CantileverEndPoint {
            load_n: int(5),
            length_mm: int(200),
        },
        m,
    )
    .analyze();
    assert_eq!(r.max_deflection_mm, Fix128::ZERO);
    assert_eq!(r.euler_critical_load_n, Fix128::ZERO);
    assert_rel(
        r.max_bending_stress_mpa,
        1000.0 / (10.0 * 400.0 / 6.0),
        "sigma unaffected by E",
    );
}

/// The Euler load is about the weak axis, so the same bar labelled 10 x 20 or
/// 20 x 10 buckles at the same load, while the bending stress (about the
/// horizontal axis) differs
#[test]
fn euler_load_does_not_depend_on_how_the_rectangle_is_labelled() {
    let load = LoadCase::CantileverEndPoint {
        load_n: int(5),
        length_mm: int(200),
    };
    let a = BeamAnalysis::new(rect(10, 20), load, pla()).analyze();
    let b = BeamAnalysis::new(rect(20, 10), load, pla()).analyze();
    assert_eq!(a.euler_critical_load_n, b.euler_critical_load_n);
    assert!(a.max_bending_stress_mpa < b.max_bending_stress_mpa);
}
