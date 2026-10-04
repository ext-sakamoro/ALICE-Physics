//! Audit oracles for `buckling`: the scenes the module's own unit tests
//! check with loose tolerances (Euler at slenderness 200, Johnson at 20, the
//! two sides of the transition, a slender and a stocky square PLA column),
//! repeated through the public `analyze_column` against the closed forms at
//! a relative tolerance of `1e-12`.
//!
//! Closed forms (Timoshenko and Gere ch. 1, Johnson 1899):
//! `r = sqrt(I / A)`, `lambda = K L / r`, `lambda_t = pi sqrt(2 E / sigma_y)`,
//! Euler `sigma = pi^2 E / lambda^2` for `lambda >= lambda_t`, Johnson
//! `sigma = sigma_y (1 - sigma_y lambda^2 / (4 pi^2 E))` below it, and
//! `P = sigma A`. The tolerance is set by the representation, not the model:
//! every quantity is a handful of `Fix128` operations on values of order
//! `1e-3` to `1e5`, whose rounding is below `1e-17` relative.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{ColumnEndCondition, CrossSection};
use alice_physics::buckling::{analyze_column, BucklingRegime, ColumnBucklingReport};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

const PI: f64 = core::f64::consts::PI;
const TOL: f64 = 1e-12;

fn rel(got: Fix128, want: f64, what: &str) {
    let g = got.to_f64();
    let err = ((g - want) / want).abs();
    assert!(err <= TOL, "{what}: got {g}, want {want}, rel err {err:e}");
}

/// `E = 3500 MPa` (3.5 GPa, exact as 7/2) and `sigma_y = 50 MPa`, the values
/// the unit tests use.
fn e3500_sy50() -> MaterialProperties {
    let mut m = MaterialProperties::pla();
    m.youngs_modulus_gpa = Fix128::from_ratio(7, 2);
    m.yield_strength_mpa = Fix128::from_int(50);
    m
}

/// Solid circle of diameter 4: `I / A = d^2 / 16 = 1`, so `r = 1` and the
/// slenderness of a pin-pin column equals its length.
fn unit_r_circle() -> CrossSection {
    CrossSection::Circular {
        diameter_mm: Fix128::from_int(4),
    }
}

fn euler(e: f64, lambda: f64) -> f64 {
    PI * PI * e / (lambda * lambda)
}

fn johnson(e: f64, sy: f64, lambda: f64) -> f64 {
    sy * (1.0 - sy * lambda * lambda / (4.0 * PI * PI * e))
}

fn column(len: Fix128, end: ColumnEndCondition, m: &MaterialProperties) -> ColumnBucklingReport {
    analyze_column(&unit_r_circle(), len, end, m)
}

/// Unit test `euler_regime_slender_column` checks `0.863 +- 0.1` MPa.
#[test]
fn euler_stress_at_slenderness_200() {
    let r = column(
        Fix128::from_int(200),
        ColumnEndCondition::PinPin,
        &e3500_sy50(),
    );
    assert_eq!(r.regime, BucklingRegime::Euler);
    rel(r.radius_of_gyration_mm, 1.0, "r");
    rel(r.slenderness, 200.0, "lambda");
    rel(r.transition_slenderness, PI * 140f64.sqrt(), "lambda_t");
    let sigma = euler(3500.0, 200.0);
    rel(r.critical_stress_mpa, sigma, "sigma_cr");
    rel(
        r.critical_load_n,
        sigma * 4.0 * PI,
        "P_cr = sigma A, A = 4 pi",
    );
}

/// Unit test `johnson_regime_stocky_column` checks `42.7 +- 1` MPa.
#[test]
fn johnson_stress_at_slenderness_20() {
    let r = column(
        Fix128::from_int(20),
        ColumnEndCondition::PinPin,
        &e3500_sy50(),
    );
    assert_eq!(r.regime, BucklingRegime::Johnson);
    rel(r.slenderness, 20.0, "lambda");
    let sigma = johnson(3500.0, 50.0, 20.0);
    rel(r.critical_stress_mpa, sigma, "sigma_cr");
    rel(r.critical_load_n, sigma * 4.0 * PI, "P_cr");
}

/// Unit test `at_transition_regimes_agree` checks a difference below 2 MPa.
/// Both formulas meet at `sigma_y / 2` with the same slope `-sigma_y /
/// lambda_t`, so each side is checked against its own closed form instead.
#[test]
fn both_sides_of_the_transition_match_their_closed_forms() {
    let lt = PI * 140f64.sqrt();
    for (offset, regime) in [
        (-0.1_f64, BucklingRegime::Johnson),
        (0.1, BucklingRegime::Euler),
    ] {
        let len = Fix128::from_f64(lt + offset);
        let lambda = len.to_f64();
        let r = column(len, ColumnEndCondition::PinPin, &e3500_sy50());
        assert_eq!(r.regime, regime, "offset {offset}");
        let want = match regime {
            BucklingRegime::Euler => euler(3500.0, lambda),
            _ => johnson(3500.0, 50.0, lambda),
        };
        rel(r.critical_stress_mpa, want, "sigma_cr near the transition");
        assert!(
            (r.critical_stress_mpa.to_f64() - 25.0).abs() <= 50.0 / lt * 0.1 * 1.01,
            "offset {offset}: within one slope step of sigma_y / 2"
        );
    }
}

/// Unit tests `analyze_column_full_report_pla` (checks `> 0`) and
/// `analyze_column_short_pla_is_johnson` (checks the regime only), with the
/// PLA preset (`E = 3.5 GPa`, `sigma_y = 50 MPa`) and a 10 x 10 square, for
/// which `I = 10^4 / 12` about either axis and `A = 100`.
#[test]
fn square_pla_column_slender_and_stocky() {
    let pla = MaterialProperties::pla();
    let square = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let i = 1.0e4_f64 / 12.0;
    let a = 100.0;
    let r_g = (i / a).sqrt();

    let slender = analyze_column(
        &square,
        Fix128::from_int(500),
        ColumnEndCondition::PinPin,
        &pla,
    );
    assert_eq!(slender.regime, BucklingRegime::Euler);
    rel(slender.radius_of_gyration_mm, r_g, "r");
    rel(slender.slenderness, 500.0 / r_g, "lambda");
    rel(
        slender.critical_load_n,
        PI * PI * 3500.0 * i / (500.0 * 500.0),
        "P_cr = pi^2 E I / L^2",
    );

    let stocky = analyze_column(
        &square,
        Fix128::from_int(20),
        ColumnEndCondition::PinPin,
        &pla,
    );
    assert_eq!(stocky.regime, BucklingRegime::Johnson);
    let lambda = 20.0 / r_g;
    rel(stocky.slenderness, lambda, "lambda");
    rel(
        stocky.critical_load_n,
        johnson(3500.0, 50.0, lambda) * a,
        "P_cr = sigma_J A",
    );
}

/// Documented effective-length factors: pin-pin 1, fixed-fixed 0.5,
/// cantilever 2, fixed-pin 0.7. In the Euler regime `P = pi^2 E I / (K L)^2`.
#[test]
fn euler_load_for_every_documented_end_condition() {
    let m = e3500_sy50();
    let len = 1000.0;
    // I = pi d^4 / 64 = 4 pi for d = 4
    let i = 4.0 * PI;
    for (end, k) in [
        (ColumnEndCondition::PinPin, 1.0),
        (ColumnEndCondition::FixedFixed, 0.5),
        (ColumnEndCondition::Cantilever, 2.0),
        (ColumnEndCondition::FixedPin, 0.7),
    ] {
        let r = column(Fix128::from_int(1000), end, &m);
        assert_eq!(r.regime, BucklingRegime::Euler, "{end:?}");
        rel(r.slenderness, k * len, "lambda = K L / r");
        rel(
            r.critical_load_n,
            PI * PI * 3500.0 * i / (k * len * k * len),
            "P_cr",
        );
    }
}
