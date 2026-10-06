//! Audit oracles for `alice_physics::buckling` (S2-2 audit).
//!
//! The module's `pub(crate)` helpers (`radius_of_gyration_mm`,
//! `slenderness_ratio`, `transition_slenderness`, `critical_stress_mpa`,
//! `plate_buckling_mpa`, `snap_through_load_n`) are unreachable from an external
//! test crate, so the source file is compiled into this test crate with
//! `#[path]` (its inline tests then run here too). Expected values are textbook
//! closed forms (Timoshenko & Gere, Johnson, Bazant & Cedolin) in plain f64.
#![allow(clippy::disallowed_methods, dead_code, unused_imports)]

mod beam_stress {
    pub use alice_physics::beam_stress::*;
}
mod det_math {
    pub use alice_physics::det_math::*;
}
mod filament_db {
    pub use alice_physics::filament_db::*;
}
mod math {
    pub use alice_physics::math::*;
}
#[path = "../src/buckling.rs"]
mod buckling;

use alice_physics::beam_stress::{ColumnEndCondition, CrossSection};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use buckling::*;

const PI: f64 = core::f64::consts::PI;

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn rel(got: Fix128, want: f64, tol: f64, what: &str) {
    let g = got.to_f64();
    let err = (g - want).abs() / want.abs().max(1e-300);
    assert!(err <= tol, "{what}: got {g}, want {want}, rel err {err:e}");
}

fn steel() -> MaterialProperties {
    let mut m = MaterialProperties::pla();
    m.youngs_modulus_gpa = fi(200);
    m.yield_strength_mpa = fi(250);
    m
}

fn circle(d: i64) -> CrossSection {
    CrossSection::Circular { diameter_mm: fi(d) }
}

#[test]
fn radius_of_gyration_matches_closed_forms_for_every_section() {
    rel(radius_of_gyration_mm(&circle(10)), 2.5, 1e-12, "circle d/4");
    let hc = CrossSection::HollowCircular {
        outer_diameter_mm: fi(20),
        inner_diameter_mm: fi(12),
    };
    rel(
        radius_of_gyration_mm(&hc),
        (20.0f64 * 20.0 + 12.0 * 12.0).sqrt() / 4.0,
        1e-12,
        "tube sqrt(Do^2+Di^2)/4",
    );
    let rect = CrossSection::Rectangular {
        width_mm: fi(10),
        height_mm: fi(30),
    };
    // weak axis (AUD-A-S2W2-006): the 10 mm side, b/sqrt(12)
    rel(
        radius_of_gyration_mm(&rect),
        10.0 / 12.0f64.sqrt(),
        1e-12,
        "rect min(b, h)/sqrt(12)",
    );
    let hr = CrossSection::HollowRectangular {
        outer_width_mm: fi(20),
        outer_height_mm: fi(30),
        wall_mm: fi(2),
    };
    // weak axis: the 20 mm outer width, I_y = (H B^3 - h b^3) / 12
    let i = (30.0 * 20.0f64.powi(3) - 26.0 * 16.0f64.powi(3)) / 12.0;
    let a = 20.0 * 30.0 - 16.0 * 26.0;
    rel(
        radius_of_gyration_mm(&hr),
        (i / a).sqrt(),
        1e-12,
        "box tube",
    );
    let ib = CrossSection::IBeam {
        flange_width_mm: fi(50),
        height_mm: fi(100),
        flange_thickness_mm: fi(6),
        web_thickness_mm: fi(4),
    };
    // weak axis: the two flanges 2 t_f b^3 / 12 plus the web (h - 2 t_f) t_w^3 / 12
    let i = (2.0 * 6.0 * 50.0f64.powi(3) + 88.0 * 4.0f64.powi(3)) / 12.0;
    let strong = (50.0 * 100.0f64.powi(3) - 46.0 * 88.0f64.powi(3)) / 12.0;
    assert!(i < strong);
    let a = 2.0 * 50.0 * 6.0 + 4.0 * 88.0;
    rel(radius_of_gyration_mm(&ib), (i / a).sqrt(), 1e-12, "I beam");
}

#[test]
fn slenderness_is_k_l_over_r_for_every_end_condition() {
    for (ec, k) in [
        (ColumnEndCondition::PinPin, 1.0),
        (ColumnEndCondition::FixedFixed, 0.5),
        (ColumnEndCondition::Cantilever, 2.0),
        (ColumnEndCondition::FixedPin, 0.7),
    ] {
        rel(
            slenderness_ratio(&circle(10), fi(500), ec),
            k * 500.0 / 2.5,
            1e-12,
            "lambda",
        );
    }
    // degenerate section: no radius, no slenderness
    let flat = CrossSection::Rectangular {
        width_mm: Fix128::ZERO,
        height_mm: fi(5),
    };
    assert_eq!(
        slenderness_ratio(&flat, fi(100), ColumnEndCondition::PinPin),
        Fix128::ZERO
    );
}

#[test]
fn transition_slenderness_is_pi_sqrt_two_e_over_sigma_y() {
    rel(
        transition_slenderness(fi(200_000), fi(250)),
        PI * (2.0f64 * 200_000.0 / 250.0).sqrt(),
        1e-12,
        "steel",
    );
    rel(
        transition_slenderness(fi(3500), fi(50)),
        PI * 140.0f64.sqrt(),
        1e-12,
        "PLA",
    );
    assert_eq!(transition_slenderness(fi(3500), Fix128::ZERO), Fix128::ZERO);
}

/// Euler: pi^2 E / lambda^2 ; Johnson: sigma_y (1 - sigma_y lambda^2 / (4 pi^2 E)).
#[test]
fn critical_stress_matches_euler_and_johnson_closed_forms() {
    let (e, sy) = (200_000.0f64, 250.0f64);
    let lt = PI * (2.0 * e / sy).sqrt();
    for lam in [20.0f64, 60.0, 100.0, 120.0] {
        let (s, reg) = critical_stress_mpa(Fix128::from_f64(lam), fi(200_000), fi(250));
        assert_eq!(reg, BucklingRegime::Johnson, "lam {lam}");
        rel(
            s,
            sy * (1.0 - sy * lam * lam / (4.0 * PI * PI * e)),
            1e-9,
            "johnson",
        );
        assert!(lam < lt);
    }
    for lam in [130.0f64, 200.0, 500.0] {
        let (s, reg) = critical_stress_mpa(Fix128::from_f64(lam), fi(200_000), fi(250));
        assert_eq!(reg, BucklingRegime::Euler, "lam {lam}");
        rel(s, PI * PI * e / (lam * lam), 1e-9, "euler");
    }
}

/// Doc: at lambda_t both formulas give sigma_y / 2, so the curve is continuous.
#[test]
fn johnson_and_euler_meet_at_half_yield_at_the_transition() {
    let lt = transition_slenderness(fi(200_000), fi(250));
    let (at, reg) = critical_stress_mpa(lt, fi(200_000), fi(250));
    assert_eq!(reg, BucklingRegime::Euler);
    rel(at, 125.0, 1e-9, "sigma at lambda_t");
    let eps = Fix128::from_ratio(1, 1000);
    let (below, reg_b) = critical_stress_mpa(lt - eps, fi(200_000), fi(250));
    assert_eq!(reg_b, BucklingRegime::Johnson);
    rel(below, 125.0, 1e-4, "johnson side");
}

/// sigma_cr is non-increasing in length and bounded by sigma_y.
#[test]
fn analyze_column_stress_decreases_with_length_and_is_capped_by_yield() {
    let mut prev = 250.0f64 + 1.0;
    for l in [1, 5, 20, 100, 300, 600, 1200, 2400] {
        let r = analyze_column(&circle(10), fi(l), ColumnEndCondition::PinPin, &steel());
        let s = r.critical_stress_mpa.to_f64();
        assert!(
            s <= prev + 1e-9 && s <= 250.0,
            "L = {l}: sigma {s} after {prev}"
        );
        prev = s;
    }
}

/// End to end: report fields against P_cr = pi^2 E I / (K L)^2 (Euler beam load) with E in GPa.
#[test]
fn analyze_column_report_matches_the_euler_beam_load() {
    let r = analyze_column(&circle(10), fi(2000), ColumnEndCondition::PinPin, &steel());
    let i = PI * 10.0f64.powi(4) / 64.0;
    let a = PI * 100.0 / 4.0;
    let p = PI * PI * 200_000.0 * i / (2000.0 * 2000.0);
    assert_eq!(r.regime, BucklingRegime::Euler);
    rel(r.radius_of_gyration_mm, 2.5, 1e-12, "r");
    rel(r.slenderness, 800.0, 1e-12, "lambda");
    rel(
        r.transition_slenderness,
        PI * (2.0f64 * 200_000.0 / 250.0).sqrt(),
        1e-12,
        "lambda_t",
    );
    rel(r.critical_load_n, p, 1e-9, "P_cr");
    rel(r.critical_stress_mpa, p / a, 1e-9, "sigma_cr");
    // cantilever with K = 2 quarters the load; fixed-fixed quadruples it
    let c = analyze_column(
        &circle(10),
        fi(2000),
        ColumnEndCondition::Cantilever,
        &steel(),
    );
    rel(c.critical_load_n, p / 4.0, 1e-9, "cantilever");
    let ff = analyze_column(
        &circle(10),
        fi(2000),
        ColumnEndCondition::FixedFixed,
        &steel(),
    );
    rel(ff.critical_load_n, p * 4.0, 1e-9, "fixed-fixed");
}

/// Timoshenko plate buckling: k pi^2 E / (12 (1 - nu^2)) (t/b)^2.
#[test]
fn plate_buckling_matches_timoshenko_and_scales_as_documented() {
    let want = 4.0 * PI * PI * 200_000.0 / (12.0 * (1.0 - 0.09)) * (2.0 / 100.0f64).powi(2);
    rel(
        plate_buckling_mpa(
            fi(200_000),
            Fix128::from_ratio(3, 10),
            fi(2),
            fi(100),
            fi(4),
        ),
        want,
        1e-9,
        "k=4",
    );
    let free = plate_buckling_mpa(
        fi(200_000),
        Fix128::from_ratio(3, 10),
        fi(2),
        fi(100),
        Fix128::from_ratio(425, 1000),
    );
    rel(free, want * 0.425 / 4.0, 1e-9, "k=0.425 one edge free");
    assert_eq!(
        plate_buckling_mpa(fi(1), Fix128::ZERO, fi(1), Fix128::ZERO, fi(4)),
        Fix128::ZERO
    );
}

/// von Mises two-bar truss: P_max = 16/(3 sqrt 3) E A (h/L)^3 in the shallow limit,
/// checked against a brute-force maximisation of the exact load curve.
#[test]
fn snap_through_matches_the_exact_two_bar_truss_maximum() {
    let (e, a, h, l) = (100_000.0f64, 10.0f64, 2.0f64, 100.0f64);
    let half = l / 2.0;
    let l0 = (half * half + h * h).sqrt();
    let mut best = 0.0f64;
    let steps = 200_000;
    for s in 0..steps {
        let y = h * 1.6 * (s as f64) / steps as f64;
        let ly = (half * half + y * y).sqrt();
        // bar strain (l0 - ly)/l0 is compressive once the apex is below... the
        // vertical balance gives P(y) = 2 E A ((l0 - ly)/l0) (y/ly) up to sign
        let q = 2.0 * e * a * ((l0 - ly) / l0).abs() * (y / ly);
        if y < h && q > best {
            best = q;
        }
    }
    let got = snap_through_load_n(fi(100_000), fi(10), fi(2), fi(100))
        .unwrap()
        .to_f64();
    let coeff = 16.0 / (3.0 * 3.0f64.sqrt());
    assert!((got - coeff * e * a * (h / l).powi(3)).abs() / got < 1e-9);
    assert!(
        (got - best).abs() / best < 0.02,
        "shallow formula {got} vs exact max {best}"
    );
}

/// Column buckling is about the weakest axis, so relabelling width/height of the
/// same physical rectangle must not change the critical load. analyze_column
/// uses `second_moment_of_area_mm4` (about the horizontal axis, maximum stiffness
/// for h > w), so 10 x 20 vs 20 x 10 differ by (20/10)^2 = 4 in P_cr.
#[test]
// AUD-A-S2W2-006
fn critical_load_does_not_depend_on_how_the_rectangle_is_labelled() {
    let a = analyze_column(
        &CrossSection::Rectangular {
            width_mm: fi(10),
            height_mm: fi(20),
        },
        fi(1500),
        ColumnEndCondition::PinPin,
        &steel(),
    );
    let b = analyze_column(
        &CrossSection::Rectangular {
            width_mm: fi(20),
            height_mm: fi(10),
        },
        fi(1500),
        ColumnEndCondition::PinPin,
        &steel(),
    );
    assert_eq!(
        a.critical_load_n,
        b.critical_load_n,
        "{} vs {}",
        a.critical_load_n.to_f64(),
        b.critical_load_n.to_f64()
    );
}

/// Doc of BucklingRegime::Yielding: "no buckling; pure yielding governs". For a
/// zero-length column the critical stress should then be the yield stress, which
/// is also the lambda -> 0+ limit of the Johnson branch. The implementation
/// returns 0 (and P_cr = 0), a discontinuity at lambda = 0.
#[test]
// AUD-A-S2W2-007
fn zero_length_column_is_governed_by_yield_not_zero() {
    let r0 = analyze_column(
        &circle(10),
        Fix128::ZERO,
        ColumnEndCondition::PinPin,
        &steel(),
    );
    let r1 = analyze_column(
        &circle(10),
        Fix128::from_ratio(1, 1000),
        ColumnEndCondition::PinPin,
        &steel(),
    );
    assert!((r1.critical_stress_mpa.to_f64() - 250.0).abs() < 1e-3);
    assert!(
        (r0.critical_stress_mpa.to_f64() - 250.0).abs() < 1e-3,
        "sigma_cr(L=0) = {} but sigma_cr(L=0.001) = {}",
        r0.critical_stress_mpa.to_f64(),
        r1.critical_stress_mpa.to_f64()
    );
}

/// Zero slenderness is pure yielding at sigma_y (AUD-A-S2W2-007; this test
/// used to pin the documented 0)
#[test]
fn zero_slenderness_is_yield_with_the_yielding_regime() {
    assert_eq!(
        critical_stress_mpa(Fix128::ZERO, fi(200_000), fi(250)),
        (fi(250), BucklingRegime::Yielding)
    );
}
