//! Audit oracles for `modal` (natural frequencies of spring-mass, beam, plate,
//! torsional shaft). Expected values are closed forms evaluated independently
//! in SI `f64` (Rao, *Mechanical Vibrations*; Blevins, *Formulas for Natural
//! Frequency and Mode Shape*; Leissa 1969), or metamorphic relations that the
//! formulas must satisfy (scaling laws, limit cases linking two functions).
//!
//! Already pinned elsewhere (not repeated): reference values for each of the
//! four functions, lambda^2 table, guard returns of exactly zero
//! (`analytic_modal_wiring.rs`, `engineering_oracles_solid.rs`).
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::CrossSection;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::modal::{
    beam_natural_frequency_hz, plate_natural_frequency_hz, single_dof_frequency_hz,
    torsional_frequency_hz, BeamBoundary,
};
use core::f64::consts::PI;

fn fx(num: i64, den: i64) -> Fix128 {
    Fix128::from_ratio(num, den)
}

fn int(v: i64) -> Fix128 {
    Fix128::from_int(v)
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs()
}

fn rect(w: i64, h: i64) -> CrossSection {
    CrossSection::Rectangular {
        width_mm: int(w),
        height_mm: int(h),
    }
}

fn beam(section: &CrossSection, l: i64, m: &MaterialProperties, bc: BeamBoundary) -> f64 {
    beam_natural_frequency_hz(section, int(l), m, bc).to_f64()
}

// ---------------------------------------------------------------------------
// beam
// ---------------------------------------------------------------------------

#[test]
fn beam_frequency_is_independent_of_width_for_a_rectangular_section() {
    // I/A = h^2/12 for a rectangle, so b cancels in sqrt(EI/(rho A)).
    let m = MaterialProperties::pla();
    for bc in [
        BeamBoundary::Cantilever,
        BeamBoundary::SimplySupported,
        BeamBoundary::ClampedClamped,
    ] {
        let f5 = beam(&rect(5, 12), 150, &m, bc);
        let f40 = beam(&rect(40, 12), 150, &m, bc);
        assert!(close(f5, f40, 1e-9), "{bc:?}: {f5} vs {f40}");
    }
}

#[test]
fn beam_frequency_is_proportional_to_height_of_a_rectangular_section() {
    let m = MaterialProperties::pla();
    let f1 = beam(&rect(10, 6), 200, &m, BeamBoundary::Cantilever);
    let f2 = beam(&rect(10, 12), 200, &m, BeamBoundary::Cantilever);
    assert!(close(f2, 2.0 * f1, 1e-9), "{f2} vs 2*{f1}");
}

#[test]
fn beam_frequency_scales_with_sqrt_of_modulus_over_density() {
    let base = MaterialProperties::pla();
    let s = rect(10, 10);
    let f0 = beam(&s, 120, &base, BeamBoundary::SimplySupported);
    let mut stiff = MaterialProperties::pla();
    stiff.youngs_modulus_gpa = base.youngs_modulus_gpa * int(4);
    let mut heavy = MaterialProperties::pla();
    heavy.density_g_cm3 = base.density_g_cm3 * int(4);
    assert!(close(
        beam(&s, 120, &stiff, BeamBoundary::SimplySupported),
        2.0 * f0,
        1e-9
    ));
    assert!(close(
        beam(&s, 120, &heavy, BeamBoundary::SimplySupported),
        0.5 * f0,
        1e-9
    ));
}

#[test]
fn beam_frequency_of_hollow_circular_section_matches_closed_form() {
    // Steel tube D = 30 mm, d = 24 mm, L = 500 mm, clamped-clamped.
    let (e, rho, od, id, l) = (200.0e9_f64, 7850.0_f64, 0.030_f64, 0.024_f64, 0.5_f64);
    let a = PI / 4.0 * (od * od - id * id);
    let i = PI / 64.0 * (od.powi(4) - id.powi(4));
    let lam2 = 4.730_040_745_f64 * 4.730_040_745_f64;
    let want = lam2 / (2.0 * PI * l * l) * (e * i / (rho * a)).sqrt();
    let mut steel = MaterialProperties::pla();
    steel.youngs_modulus_gpa = int(200);
    steel.density_g_cm3 = fx(785, 100);
    let s = CrossSection::HollowCircular {
        outer_diameter_mm: int(30),
        inner_diameter_mm: int(24),
    };
    let got = beam(&s, 500, &steel, BeamBoundary::ClampedClamped);
    assert!(close(got, want, 2e-4), "got {got} want {want}");
}

#[test]
fn beam_frequency_of_hollow_rectangular_section_matches_closed_form() {
    // PLA box 20 x 30 mm, wall 2 mm, L = 250 mm, cantilever, bending about the
    // horizontal axis (height 30).
    let (e, rho, w, h, t, l) = (
        3.5e9_f64, 1240.0_f64, 0.020_f64, 0.030_f64, 0.002_f64, 0.25_f64,
    );
    let a = w * h - (w - 2.0 * t) * (h - 2.0 * t);
    let i = (w * h.powi(3) - (w - 2.0 * t) * (h - 2.0 * t).powi(3)) / 12.0;
    let lam2 = 1.875_104_069_f64 * 1.875_104_069_f64;
    let want = lam2 / (2.0 * PI * l * l) * (e * i / (rho * a)).sqrt();
    let s = CrossSection::HollowRectangular {
        outer_width_mm: int(20),
        outer_height_mm: int(30),
        wall_mm: int(2),
    };
    let got = beam(
        &s,
        250,
        &MaterialProperties::pla(),
        BeamBoundary::Cantilever,
    );
    assert!(close(got, want, 2e-4), "got {got} want {want}");
}

#[test]
fn lambda_squared_is_within_half_a_unit_of_the_last_stored_digit_of_blevins() {
    // The constants are stored to 3 decimals; the exact (beta L)^2 are 3.516015,
    // 9.869604 and 22.373285. Anything farther than 5e-4 is a wrong constant,
    // not rounding.
    let cases = [
        (
            BeamBoundary::Cantilever,
            1.875_104_069_f64 * 1.875_104_069_f64,
        ),
        (BeamBoundary::SimplySupported, PI * PI),
        (
            BeamBoundary::ClampedClamped,
            4.730_040_745_f64 * 4.730_040_745_f64,
        ),
    ];
    for (bc, want) in cases {
        let got = bc.lambda_squared().to_f64();
        assert!((got - want).abs() < 5e-4, "{bc:?}: {got} vs {want}");
    }
}

// ---------------------------------------------------------------------------
// plate
// ---------------------------------------------------------------------------

fn plate(m: &MaterialProperties, nu: Fix128, h: i64, a: i64, b: i64) -> f64 {
    plate_natural_frequency_hz(m, nu, int(h), int(a), int(b)).to_f64()
}

#[test]
fn plate_frequency_is_exact_to_the_precision_of_the_stored_ten_to_the_four_point_five() {
    // PLA 2 mm plate 100 x 150 mm, nu = 0.35, SI closed form (Leissa f_11).
    let (e, rho, nu, h, a, b) = (
        3.5e9_f64, 1240.0_f64, 0.35_f64, 2.0e-3_f64, 0.1_f64, 0.15_f64,
    );
    let d = e * h.powi(3) / (12.0 * (1.0 - nu * nu));
    let want = PI / 2.0 * (d / (rho * h)).sqrt() * (1.0 / (a * a) + 1.0 / (b * b));
    let got = plate(&MaterialProperties::pla(), fx(35, 100), 2, 100, 150);
    // stored 10^4.5 = 31622.776 differs from 31622.7766 by 1.9e-8 relative
    assert!(close(got, want, 3.0e-8), "got {got} want {want}");
}

#[test]
fn plate_frequency_is_symmetric_in_the_two_side_lengths() {
    let m = MaterialProperties::pla();
    let nu = fx(35, 100);
    assert_eq!(plate(&m, nu, 2, 80, 140), plate(&m, nu, 2, 140, 80));
}

#[test]
fn plate_frequency_is_proportional_to_thickness() {
    let m = MaterialProperties::pla();
    let nu = fx(3, 10);
    let f1 = plate(&m, nu, 2, 100, 100);
    let f3 = plate(&m, nu, 6, 100, 100);
    assert!(close(f3, 3.0 * f1, 1e-9), "{f3} vs 3*{f1}");
}

#[test]
fn plate_frequency_follows_one_over_sqrt_one_minus_nu_squared() {
    let m = MaterialProperties::pla();
    let f0 = plate(&m, Fix128::ZERO, 3, 120, 90);
    let f3 = plate(&m, fx(3, 10), 3, 120, 90);
    let fm3 = plate(&m, fx(-3, 10), 3, 120, 90);
    assert!(
        close(f3, f0 / (1.0 - 0.09_f64).sqrt(), 1e-9),
        "{f3} vs {f0}"
    );
    assert!(close(fm3, f3, 1e-12), "Poisson ratio enters only squared");
}

#[test]
fn plate_strip_limit_equals_simply_supported_beam_of_unit_width() {
    // a -> infinity: the plate degenerates to a strip of width b, i.e. a
    // simply supported beam (L = b, rectangular b x h cross-section) with
    // D/(rho h) = E h^2 / (12 rho) (nu = 0). The a-term is 1e-6 relative.
    let m = MaterialProperties::pla();
    let fp = plate(&m, Fix128::ZERO, 4, 100_000, 100);
    let fb = beam(&rect(1, 4), 100, &m, BeamBoundary::SimplySupported);
    assert!(close(fp, fb, 1.2e-4), "plate {fp} beam {fb}");
}

// ---------------------------------------------------------------------------
// torsional shaft
// ---------------------------------------------------------------------------

#[test]
fn torsional_frequency_of_a_solid_steel_shaft_with_end_disc_matches_rao() {
    // d = 12 mm, L = 300 mm, G = E/(2(1+nu)) with E = 200 GPa, nu = 0.3,
    // end disc I_p = 4000 g mm^2 = 4e-9 kg m^2.
    let (d, l) = (0.012_f64, 0.3_f64);
    let g = 200.0e9 / (2.0 * 1.3);
    let j = PI * d.powi(4) / 32.0;
    let ip = 4000.0e-3 * 1.0e-6;
    let want = (g * j / (ip * l)).sqrt() / (2.0 * PI);
    let g_mpa = fx(200_000, 1) / fx(26, 10);
    let j_mm4 = Fix128::PI * int(12) * int(12) * int(12) * int(12) / int(32);
    let got = torsional_frequency_hz(g_mpa, j_mm4, int(4000), int(300)).to_f64();
    assert!(close(got, want, 1e-9), "got {got} want {want}");
}

#[test]
fn torsional_frequency_scales_as_sqrt_g_over_ip_l() {
    let f0 = torsional_frequency_hz(int(1000), int(500), int(200), int(40)).to_f64();
    let fg = torsional_frequency_hz(int(4000), int(500), int(200), int(40)).to_f64();
    let fj = torsional_frequency_hz(int(1000), int(2000), int(200), int(40)).to_f64();
    let fi = torsional_frequency_hz(int(1000), int(500), int(800), int(40)).to_f64();
    let fl = torsional_frequency_hz(int(1000), int(500), int(200), int(160)).to_f64();
    assert!(close(fg, 2.0 * f0, 1e-9));
    assert!(close(fj, 2.0 * f0, 1e-9));
    assert!(close(fi, 0.5 * f0, 1e-9));
    assert!(close(fl, 0.5 * f0, 1e-9));
}

// ---------------------------------------------------------------------------
// single DOF
// ---------------------------------------------------------------------------

#[test]
fn single_dof_scales_as_sqrt_k_over_m_and_vanishes_for_zero_stiffness() {
    let f0 = single_dof_frequency_hz(int(100), int(25)).to_f64();
    let fk = single_dof_frequency_hz(int(400), int(25)).to_f64();
    let fm = single_dof_frequency_hz(int(100), int(100)).to_f64();
    assert!(close(fk, 2.0 * f0, 1e-9));
    assert!(close(fm, 0.5 * f0, 1e-9));
    assert_eq!(single_dof_frequency_hz(Fix128::ZERO, int(25)), Fix128::ZERO);
    // closed form at a light mass (1 mg): omega = sqrt(5e4 N/m / 1e-6 kg)
    let want = (50.0e3_f64 / 1.0e-6).sqrt() / (2.0 * PI);
    let got = single_dof_frequency_hz(int(50), fx(1, 1000)).to_f64();
    assert!(close(got, want, 1e-9), "got {got} want {want}");
}

#[test]
#[ignore = "known defect: AUD-A-S3W2-002: plate_natural_frequency_hz treats negative thickness / side as its absolute value (h=-2 mm returns 325.30 Hz, the same as h=+2 mm) while beam_natural_frequency_hz returns 0 for a negative length"]
fn plate_with_negative_dimension_is_rejected_like_the_beam_with_negative_length() {
    let m = MaterialProperties::pla();
    let nu = fx(35, 100);
    let pos = plate(&m, nu, 2, 100, 100);
    assert!(pos > 0.0);
    assert_eq!(plate(&m, nu, -2, 100, 100), 0.0, "negative thickness");
    assert_eq!(plate(&m, nu, 2, -100, 100), 0.0, "negative side");
    let neg_beam = beam(&rect(10, 10), -100, &m, BeamBoundary::Cantilever);
    assert_eq!(neg_beam, 0.0);
}
