//! Oracles for the wiring of `modal`
//! (`examples/modal_frequency_analysis.rs`): `single_dof_frequency_hz`,
//! `BeamBoundary::lambda_squared`, `beam_natural_frequency_hz`,
//! `plate_natural_frequency_hz`, `torsional_frequency_hz`.
//!
//! # Closed forms (every expected value is derived here, none from the crate)
//!
//! From the module doc (`src/modal.rs:1-27`, citing Blevins, Leissa and Rao):
//!
//! ```text
//! single-DOF spring-mass  : f = (1/2π) √(k_SI / m_SI)
//! beam (first mode)        : f = (λ² / (2π L²)) √(E I / (ρ A))
//! plate (SSSS, first mode) : f = (π/2) √(D / (ρ h)) (1/a² + 1/b²), D = E h³/(12(1−ν²))
//! torsional shaft           : f = (1/2π) √(G J / (I_p L))
//! ```
//!
//! The scenarios here (materials, sections, dimensions) are chosen to be
//! different from the ones already covered in
//! `tests/engineering_oracles_solid.rs` and in
//! `examples/modal_frequency_analysis.rs`, so this file adds real coverage
//! rather than duplicating either.
//!
//! # Degenerate / extreme input (each result is pinned, never just "does
//! not panic")
//!
//! `BeamBoundary::lambda_squared` takes no numeric parameter — its only
//! input is the boundary-condition variant itself, and the enum has exactly
//! three variants, so exercising all three *is* its full boundary-condition
//! domain (there is no narrower or wider edge case to add).
//!
//! `Fix128` addition/multiplication/subtraction are mod-2¹²⁸ wrapping
//! operations (`src/math.rs`'s documented `Mul`/`Add` contracts), so
//! "does not panic" is true unconditionally for every input and is not by
//! itself a meaningful assertion — every extreme-magnitude case below pins
//! the specific wrapped or guarded value, derived independently:
//!
//! - When both factors of a `Fix128` multiply are plain integers
//!   (`Fix128::from_int`, fractional part `lo == 0`), the product's integer
//!   part is exactly the `i64::wrapping_mul` of the two integers and its
//!   fractional part is exactly `0` — a direct consequence of the schoolbook
//!   128×128→128 multiply collapsing to a single `i64` wraparound when every
//!   cross term involving a zero `lo` vanishes. This is `i64`'s own
//!   documented wrapping semantics, not `Fix128::mul`'s, so predicting it
//!   here does not call into the function under test.
//! - `Fix128::sqrt` is documented (`src/math.rs`) to return `ZERO` for a
//!   negative or zero argument, deterministically, never NaN. Several cases
//!   below drive an internal ratio negative (through a sign, not through
//!   wraparound) specifically to exercise this contract.
//! - `beam_natural_frequency_hz`'s `area <= Fix128::ZERO` guard (src/modal.rs)
//!   only rejects an area that wraps to a non-positive value; an overflowing
//!   section whose area wraps to a *positive* value of the wrong magnitude
//!   silently bypasses it. That specific blind spot is pinned below via the
//!   `i64::wrapping_mul` identity above, without reimplementing `Fix128`'s
//!   multiply chain for the rest of the formula (which also involves `h³` and
//!   division by non-integer constants, where the simple identity above no
//!   longer applies) — this test's scope is the guard's blind spot, not the
//!   final frequency value, and says so explicitly.
//! - `plate_natural_frequency_hz`'s `one_minus_nu2.is_zero()` guard only
//!   rejects a Poisson ratio of exactly ±1; `ν > 1` (unphysical, but not
//!   excluded by the guard) makes `1 − ν²` strictly negative, which drives
//!   the stiffness-to-density ratio negative and hits the `sqrt` contract
//!   above instead.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::beam_stress::CrossSection;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::modal::{
    beam_natural_frequency_hz, plate_natural_frequency_hz, single_dof_frequency_hz,
    torsional_frequency_hz, BeamBoundary,
};
use std::f64::consts::PI;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

// ---------------------------------------------------------------------------
// Closed forms
// ---------------------------------------------------------------------------

/// Rao §2.2: `f = (1/2π) √(k/m)`. `k = 4000 N/mm = 4e6 N/m`,
/// `m = 16 g = 0.016 kg` (different magnitudes from both
/// `tests/engineering_oracles_solid.rs::modal_single_dof_matches_rao`
/// (`k=1000, m=1`) and `examples/modal_frequency_analysis.rs` (`k=250,
/// m=750`)).
#[test]
fn single_dof_matches_rao_closed_form() {
    let k_si = 4000.0_f64 * 1000.0;
    let m_si = 16.0_f64 / 1000.0;
    let want = (k_si / m_si).sqrt() / (2.0 * PI);
    let f = single_dof_frequency_hz(Fix128::from_int(4000), Fix128::from_int(16)).to_f64();
    assert!(
        rel_err(f, want) < 1e-6,
        "single_dof_frequency_hz(4000, 16) = {f} vs {want}"
    );
}

/// Blevins Table 8-1 eigenvalues `(βL)`: `1.875_104_069` (cantilever), `π`
/// (pinned-pinned), `4.730_040_745` (clamped-clamped); `λ² = (βL)²`.
/// Checked directly against `BeamBoundary::lambda_squared`, not through
/// `beam_natural_frequency_hz` (unlike the existing boundary-ordering-only
/// check in `src/modal.rs`'s own `#[cfg(test)]` module, which the wiring
/// guard does not count as production coverage anyway).
#[test]
fn lambda_squared_matches_blevins_table_for_every_boundary_condition() {
    let cases = [
        (BeamBoundary::Cantilever, 1.875_104_069_f64),
        (BeamBoundary::SimplySupported, PI),
        (BeamBoundary::ClampedClamped, 4.730_040_745_f64),
    ];
    for (bc, beta_l) in cases {
        let want = beta_l * beta_l;
        let got = bc.lambda_squared().to_f64();
        assert!(
            rel_err(got, want) < 2e-4,
            "{bc:?}.lambda_squared() = {got} vs Blevins (βL)² = {want} \
             (module rounds to 4 significant digits, see src/modal.rs:75-84)"
        );
    }
}

/// The eigenvalues themselves must be strictly increasing with end
/// restraint (more restraint -> stiffer -> higher λ²), independent of
/// whatever scale factor `beam_natural_frequency_hz` later multiplies them
/// by.
#[test]
fn lambda_squared_is_strictly_increasing_with_end_restraint() {
    let cantilever = BeamBoundary::Cantilever.lambda_squared();
    let pinned = BeamBoundary::SimplySupported.lambda_squared();
    let clamped = BeamBoundary::ClampedClamped.lambda_squared();
    assert!(
        cantilever < pinned,
        "cantilever {cantilever:?} < pinned {pinned:?}"
    );
    assert!(pinned < clamped, "pinned {pinned:?} < clamped {clamped:?}");
}

/// Blevins Table 8-1, solid circular steel rod, L = 400 mm, d = 15 mm
/// (different section/material/length from the Rectangular-PLA-L100 case in
/// `tests/engineering_oracles_solid.rs` and the Circular-ABS-L250 case in
/// `examples/modal_frequency_analysis.rs`). `A = π d²/4`, `I = π d⁴/64`
/// (`CrossSection::Circular`, `src/beam_stress.rs`).
#[test]
fn beam_natural_frequency_matches_blevins_for_steel_circular_rod() {
    let (e_gpa, rho_gcc, d_mm, l_mm) = (200.0_f64, 7.85_f64, 15.0_f64, 400.0_f64);
    let (e_pa, rho_si, d_si, l_si) = (e_gpa * 1e9, rho_gcc * 1000.0, d_mm * 1e-3, l_mm * 1e-3);
    let a_si = PI / 4.0 * d_si * d_si;
    let i_si = PI / 64.0 * (d_si * d_si * d_si * d_si);
    let base = (e_pa * i_si / (rho_si * a_si)).sqrt() / (2.0 * PI * l_si * l_si);

    let section = CrossSection::Circular {
        diameter_mm: Fix128::from_int(15),
    };
    let steel = MaterialProperties {
        youngs_modulus_gpa: Fix128::from_int(200),
        density_g_cm3: Fix128::from_ratio(785, 100),
        ..MaterialProperties::pla()
    };
    let cases = [
        (BeamBoundary::Cantilever, 1.875_104_069_f64),
        (BeamBoundary::SimplySupported, PI),
        (BeamBoundary::ClampedClamped, 4.730_040_745_f64),
    ];
    for (bc, beta_l) in cases {
        let want = beta_l * beta_l * base;
        let f = beam_natural_frequency_hz(&section, Fix128::from_int(400), &steel, bc).to_f64();
        assert!(
            rel_err(f, want) < 2e-4,
            "beam_natural_frequency_hz(steel rod, {bc:?}) = {f} vs {want}"
        );
    }
}

/// Leissa, *Vibration of Plates* §4.1, SSSS rectangular plate, PETG,
/// h = 1.5 mm, a = 80 mm, b = 80 mm (square, different material/thickness/
/// aspect ratio from the a=100,b=150 PLA case in
/// `tests/engineering_oracles_solid.rs` and the a=b=120 ABS case in
/// `examples/modal_frequency_analysis.rs`).
#[test]
fn plate_natural_frequency_matches_leissa_for_square_petg_plate() {
    let m = MaterialProperties::petg();
    let (e_gpa, rho_gcc, nu, h_mm, side_mm) = (2.0_f64, 1.27_f64, 0.38_f64, 1.5_f64, 80.0_f64);
    let (e_pa, rho_si, h_si, side_si) =
        (e_gpa * 1e9, rho_gcc * 1000.0, h_mm * 1e-3, side_mm * 1e-3);
    let d_flex = e_pa * (h_si * h_si * h_si) / (12.0 * (1.0 - nu * nu));
    let want = PI / 2.0 * (d_flex / (rho_si * h_si)).sqrt() * (2.0 / (side_si * side_si));

    let f = plate_natural_frequency_hz(
        &m,
        Fix128::from_ratio(38, 100),
        Fix128::from_ratio(15, 10),
        Fix128::from_int(80),
        Fix128::from_int(80),
    )
    .to_f64();
    assert!(
        rel_err(f, want) < 1e-6,
        "plate_natural_frequency_hz(square PETG) = {f} vs {want}"
    );
}

/// Rao §2.2 / Blevins Table 8-15, bronze shaft: `G = 40 000 MPa`,
/// `d = 20 mm`, `L = 300 mm`, disc `I_p = 5.0e6 g·mm²` (different material/
/// geometry from the steel-shaft case in
/// `tests/engineering_oracles_solid.rs` and the aluminium-shaft case in
/// `examples/modal_frequency_analysis.rs`).
#[test]
fn torsional_frequency_matches_rao_for_bronze_shaft() {
    let (g_mpa, d_mm, l_mm, ip_g_mm2) = (40_000.0_f64, 20.0_f64, 300.0_f64, 5.0e6_f64);
    let j_mm4 = PI * (d_mm * d_mm * d_mm * d_mm) / 32.0;
    let k_theta_si = g_mpa * j_mm4 / l_mm * 1e-3;
    let ip_si = ip_g_mm2 * 1e-9;
    let want = (k_theta_si / ip_si).sqrt() / (2.0 * PI);

    let f = torsional_frequency_hz(
        Fix128::from_int(40_000),
        Fix128::from_f64(j_mm4),
        Fix128::from_f64(ip_g_mm2),
        Fix128::from_int(300),
    )
    .to_f64();
    assert!(
        rel_err(f, want) < 1e-5,
        "torsional_frequency_hz(bronze shaft) = {f} vs {want}"
    );
}

// ---------------------------------------------------------------------------
// Degenerate input: zero
// ---------------------------------------------------------------------------

#[test]
fn single_dof_frequency_zero_or_negative_mass_returns_exact_zero() {
    assert_eq!(
        single_dof_frequency_hz(Fix128::from_int(100), Fix128::ZERO),
        Fix128::ZERO
    );
    assert_eq!(
        single_dof_frequency_hz(Fix128::from_int(100), Fix128::from_int(-5)),
        Fix128::ZERO,
        "negative mass hits the same `mass_g <= ZERO` guard as zero mass (src/modal.rs:45)"
    );
}

#[test]
fn beam_natural_frequency_zero_length_or_zero_area_returns_exact_zero() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let m = MaterialProperties::pla();
    assert_eq!(
        beam_natural_frequency_hz(&section, Fix128::ZERO, &m, BeamBoundary::Cantilever),
        Fix128::ZERO,
        "zero length hits the `length_mm <= ZERO` guard (src/modal.rs:108)"
    );
    let zero_width_section = CrossSection::Rectangular {
        width_mm: Fix128::ZERO,
        height_mm: Fix128::from_int(10),
    };
    assert_eq!(
        beam_natural_frequency_hz(
            &zero_width_section,
            Fix128::from_int(100),
            &m,
            BeamBoundary::Cantilever
        ),
        Fix128::ZERO,
        "zero width -> zero area hits the `area <= ZERO` guard (src/modal.rs:112)"
    );
}

#[test]
fn plate_natural_frequency_zero_dimension_returns_exact_zero() {
    let m = MaterialProperties::pla();
    let nu = Fix128::from_ratio(35, 100);
    for (a, b, h) in [
        (Fix128::ZERO, Fix128::from_int(100), Fix128::from_int(2)),
        (Fix128::from_int(100), Fix128::ZERO, Fix128::from_int(2)),
        (Fix128::from_int(100), Fix128::from_int(100), Fix128::ZERO),
    ] {
        assert_eq!(
            plate_natural_frequency_hz(&m, nu, h, a, b),
            Fix128::ZERO,
            "a={a:?} b={b:?} h={h:?} must hit the zero-dimension guard (src/modal.rs:152)"
        );
    }
}

#[test]
fn torsional_frequency_zero_length_or_zero_polar_inertia_returns_exact_zero() {
    assert_eq!(
        torsional_frequency_hz(
            Fix128::from_int(1000),
            Fix128::from_int(100),
            Fix128::from_int(10),
            Fix128::ZERO,
        ),
        Fix128::ZERO,
        "zero shaft length hits the `length_mm.is_zero()` guard (src/modal.rs:200)"
    );
    assert_eq!(
        torsional_frequency_hz(
            Fix128::from_int(1000),
            Fix128::from_int(100),
            Fix128::ZERO,
            Fix128::from_int(50),
        ),
        Fix128::ZERO,
        "zero polar mass moment hits the `i_p_g_mm2.is_zero()` guard (src/modal.rs:200)"
    );
}

// ---------------------------------------------------------------------------
// Degenerate input: unphysical sign, not caught by any explicit guard,
// pinned via Fix128::sqrt's documented "negative input -> ZERO" contract
// ---------------------------------------------------------------------------

/// `youngs_modulus_gpa < 0` is not rejected by any guard in
/// `beam_natural_frequency_hz` (every explicit guard checks a length or an
/// area, never the material). It drives `inner = e_mpa * i / rho_a`
/// negative, so `inner.sqrt()` (documented: negative input -> `ZERO`)
/// collapses the whole formula to exactly `ZERO`, not a panic and not NaN.
#[test]
fn beam_natural_frequency_negative_youngs_modulus_hits_sqrt_negative_guard() {
    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    };
    let mut m = MaterialProperties::pla();
    m.youngs_modulus_gpa = Fix128::from_int(-35) / Fix128::from_int(10);
    let f = beam_natural_frequency_hz(
        &section,
        Fix128::from_int(100),
        &m,
        BeamBoundary::Cantilever,
    );
    assert_eq!(
        f,
        Fix128::ZERO,
        "negative E must drive inner < 0 -> sqrt = ZERO -> f = ZERO, not a panic or NaN"
    );
}

/// `g_mpa < 0` is not rejected by any guard in `torsional_frequency_hz`
/// (the only guards check `length_mm` and `i_p_g_mm2`). It drives
/// `k_theta < 0`, so `ratio.sqrt()` collapses to exactly `ZERO`.
#[test]
fn torsional_frequency_negative_shear_modulus_hits_sqrt_negative_guard() {
    let f = torsional_frequency_hz(
        Fix128::from_int(-500),
        Fix128::from_int(100),
        Fix128::from_int(10),
        Fix128::from_int(50),
    );
    assert_eq!(
        f,
        Fix128::ZERO,
        "negative G must drive k_theta < 0 -> ratio < 0 -> sqrt = ZERO -> f = ZERO"
    );
}

/// `i_p_g_mm2 < 0` passes the `is_zero()` guard unchanged (only exact zero
/// is rejected) and makes `ratio = k_theta / i_p_g_mm2` negative for a
/// positive `k_theta`, again landing on the same `sqrt` contract.
#[test]
fn torsional_frequency_negative_polar_inertia_hits_sqrt_negative_guard() {
    let f = torsional_frequency_hz(
        Fix128::from_int(1000),
        Fix128::from_int(100),
        Fix128::from_int(-10),
        Fix128::from_int(50),
    );
    assert_eq!(
        f,
        Fix128::ZERO,
        "negative I_p must drive ratio < 0 -> sqrt = ZERO -> f = ZERO"
    );
}

/// `poisson_ratio > 1` (unphysical: a real material has `0 <= ν < 0.5`) is
/// not rejected by `one_minus_nu2.is_zero()`, which only excludes exactly
/// `ν = ±1`. At `ν = 2`, `1 - ν² = -3`, driving `d_eng` and then `ratio`
/// negative, landing on the same `sqrt` contract as the beam/torsional
/// cases above.
#[test]
fn plate_natural_frequency_poisson_ratio_above_one_hits_sqrt_negative_guard() {
    let m = MaterialProperties::pla();
    let f = plate_natural_frequency_hz(
        &m,
        Fix128::from_int(2),
        Fix128::from_int(2),
        Fix128::from_int(100),
        Fix128::from_int(100),
    );
    assert_eq!(
        f,
        Fix128::ZERO,
        "nu=2 (unphysical, not guarded) must drive 1-nu^2 < 0 -> ratio < 0 -> sqrt = ZERO -> f = ZERO"
    );

    // The exactly-guarded boundary (nu = +-1, 1-nu^2 == 0 exactly) is a
    // separate code path (`one_minus_nu2.is_zero()`, src/modal.rs:158) from
    // the nu > 1 case above (which instead falls through to the sqrt
    // guard): both must return exactly ZERO, but for different reasons.
    assert_eq!(
        plate_natural_frequency_hz(
            &m,
            Fix128::ONE,
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
        ),
        Fix128::ZERO,
        "nu=1 must hit the exact one_minus_nu2.is_zero() guard"
    );
    assert_eq!(
        plate_natural_frequency_hz(
            &m,
            Fix128::from_int(-1),
            Fix128::from_int(2),
            Fix128::from_int(100),
            Fix128::from_int(100),
        ),
        Fix128::ZERO,
        "nu=-1 must also hit the exact one_minus_nu2.is_zero() guard (nu^2 == 1 either sign)"
    );
}

// ---------------------------------------------------------------------------
// Degenerate input: extreme Fix128 magnitude (silent mod-2^128 wraparound,
// every value pinned per the module doc above -- never just "no panic")
// ---------------------------------------------------------------------------

/// For two `Fix128::from_int` operands (fractional part exactly zero), the
/// product's integer part is exactly `i64::wrapping_mul` of the two
/// integers and its fractional part is exactly zero -- see the module doc
/// above. `10^16 * 1000 = 10^19 > i64::MAX (~9.22e18)`, so the wrapped
/// product is *negative* (`10^19 - 2^64`), even though both inputs were
/// positive. `single_dof_frequency_hz` has no guard against this (only
/// `mass_g <= ZERO` is checked), so the negative `k_si` reaches
/// `(k_si / m_si).sqrt()`, which returns exactly `ZERO` per its documented
/// contract -- an extreme *positive* stiffness silently produces a
/// frequency of exactly zero, not a panic and not a huge number.
#[test]
fn single_dof_frequency_extreme_stiffness_wraps_negative_and_sqrt_guards_to_zero() {
    let k_int: i64 = 10_000_000_000_000_000; // 1e16
    let want_k_si_hi = k_int.wrapping_mul(1000);
    assert!(
        want_k_si_hi < 0,
        "sanity: 1e16 * 1000 must wrap to a negative i64 (got {want_k_si_hi})"
    );

    let r = catch_unwind(AssertUnwindSafe(|| {
        single_dof_frequency_hz(Fix128::from_int(k_int), Fix128::from_int(1000))
    }));
    assert!(r.is_ok(), "extreme stiffness must not panic");
    assert_eq!(
        r.unwrap(),
        Fix128::ZERO,
        "wrapped-negative k_si must collapse to exactly ZERO via the sqrt(negative) contract"
    );
}

/// Same wraparound mechanism as above, reached through
/// `torsional_frequency_hz`'s `k_theta = g_mpa * j_mm4 / length_mm` with
/// `j_mm4 = length_mm = 1` (both exact integer identities), so
/// `k_theta = g_mpa` exactly, and `g_mpa = 1e16` combined with the same
/// `* 1000`-equivalent magnitude... (kept simple here: the shear modulus
/// itself is the extreme operand, multiplied by `j_mm4 = 1000` with
/// `length_mm = 1`, reproducing the identical `1e16 * 1000` wraparound as
/// the single-DOF case above, independently of that test).
#[test]
fn torsional_frequency_extreme_shear_modulus_wraps_negative_and_sqrt_guards_to_zero() {
    let g_int: i64 = 10_000_000_000_000_000; // 1e16
    let want_k_theta_hi = g_int.wrapping_mul(1000);
    assert!(
        want_k_theta_hi < 0,
        "sanity: 1e16 * 1000 must wrap to a negative i64 (got {want_k_theta_hi})"
    );

    let r = catch_unwind(AssertUnwindSafe(|| {
        torsional_frequency_hz(
            Fix128::from_int(g_int),
            Fix128::from_int(1000),
            Fix128::from_int(1),
            Fix128::from_int(1),
        )
    }));
    assert!(r.is_ok(), "extreme shear modulus must not panic");
    assert_eq!(
        r.unwrap(),
        Fix128::ZERO,
        "wrapped-negative k_theta must collapse to exactly ZERO via the sqrt(negative) contract"
    );
}

/// `beam_natural_frequency_hz`'s `area <= Fix128::ZERO` guard (src/modal.rs:
/// 112) is meant to reject a degenerate section, but it only rejects a
/// *non-positive* area -- an overflowing section whose area wraps to a
/// large *positive* value of the wrong magnitude bypasses it silently.
/// `width_mm = height_mm = 10^10`: `area = width * height` is a plain
/// integer*integer product (both operands have `lo == 0`), so by the
/// `i64::wrapping_mul` identity in the module doc above its wrapped integer
/// part is exactly `(10^10).wrapping_mul(10^10)`, which is positive (the
/// wraparound happens to land back on the positive side here, unlike the
/// `1e16 * 1000` case above).
///
/// This test pins only that specific guard blind spot (the wrapped `area`
/// value itself, and that it is positive hence bypasses the guard) and
/// confirms the full `beam_natural_frequency_hz` call does not panic for
/// this section. It deliberately does not pin the final frequency value:
/// that would require independently reimplementing `Fix128`'s wrapping
/// multiply for the rest of the formula (`h³`, and division by the
/// non-integer `1/12` and `10^4.5` constants), where the simple
/// `lo == 0` identity used here no longer applies.
#[test]
fn beam_natural_frequency_extreme_section_wraps_area_positive_and_bypasses_guard() {
    let side: i64 = 10_000_000_000; // 1e10 mm
    let want_area_hi = side.wrapping_mul(side);
    assert!(
        want_area_hi > 0,
        "sanity: 1e10 * 1e10 must wrap to a positive i64 (got {want_area_hi})"
    );

    let section = CrossSection::Rectangular {
        width_mm: Fix128::from_int(side),
        height_mm: Fix128::from_int(side),
    };
    let area = section.area_mm2();
    assert_eq!(
        area,
        Fix128::from_raw(want_area_hi, 0),
        "wrapped area must match the independently predicted i64::wrapping_mul value"
    );
    assert!(
        area > Fix128::ZERO,
        "the wrapped area must be positive, i.e. it bypasses the `area <= ZERO` guard"
    );

    let m = MaterialProperties::pla();
    let r = catch_unwind(AssertUnwindSafe(|| {
        beam_natural_frequency_hz(
            &section,
            Fix128::from_int(100),
            &m,
            BeamBoundary::Cantilever,
        )
    }));
    assert!(
        r.is_ok(),
        "beam_natural_frequency_hz must not panic for an overflowing section"
    );
}
