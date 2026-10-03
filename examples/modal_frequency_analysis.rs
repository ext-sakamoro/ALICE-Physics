//! Driving the five closed-form modal/natural-frequency formulas
//! (`src/modal.rs`) through production entry points.
//!
//! `wiring_guard.py` found five zero-production-caller items in
//! `src/modal.rs`: `single_dof_frequency_hz`, `BeamBoundary::lambda_squared`,
//! `beam_natural_frequency_hz`, `plate_natural_frequency_hz` and
//! `torsional_frequency_hz` — only `#[cfg(test)]` self-references inside the
//! module, plus `tests/engineering_oracles_solid.rs`, called into them before
//! this example (the guard does not count `tests/` as production: src /
//! examples / benches / fuzz / bindings only). `plate_natural_frequency_hz`
//! is also called from `src/vibration_wall.rs::analyze_wall_resonance`, but
//! that function itself has no production caller either, so the call does
//! not count (transitively unwired).
//!
//! Every expected value below is derived independently from the formula in
//! the module doc (`src/modal.rs:1-27`, citing Blevins, Leissa and Rao), in
//! plain `f64`, and is never obtained by calling the function under test.
//! The scenarios here are deliberately different from the ones already
//! covered in `tests/engineering_oracles_solid.rs` (different cross-section,
//! material, dimensions) so this is additional coverage, not a duplicate.
//!
//! ```text
//! single-DOF spring-mass : f = (1/2π) √(k_SI / m_SI)
//! beam (first mode)       : f = (λ² / (2π L²)) √(E I / (ρ A))
//! plate (SSSS, first mode): f = (π/2) √(D / (ρ h)) (1/a² + 1/b²), D = E h³/(12(1−ν²))
//! torsional shaft          : f = (1/2π) √(G J / (I_p L))
//! ```
//!
//! ```bash
//! cargo run --example modal_frequency_analysis --features std
//! ```

use alice_physics::beam_stress::CrossSection;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::modal::{
    beam_natural_frequency_hz, plate_natural_frequency_hz, single_dof_frequency_hz,
    torsional_frequency_hz, BeamBoundary,
};
use std::f64::consts::PI;

fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn check(name: &str, actual: f64, expected: f64, tol: f64) {
    let err = rel_err(actual, expected);
    println!("[modal] {name}: actual={actual} expected={expected} rel_err={err:e}");
    assert!(
        err < tol,
        "{name}: actual={actual} expected={expected} rel_err={err} (tol {tol})"
    );
}

fn main() {
    // -----------------------------------------------------------------
    // 1. single_dof_frequency_hz: f = (1/2π) √(k_SI/m_SI)
    //    k = 250 N/mm = 250 000 N/m, m = 750 g = 0.75 kg (not the 1000/1
    //    case already covered in tests/engineering_oracles_solid.rs).
    // -----------------------------------------------------------------
    let k_mm = 250.0_f64;
    let m_g = 750.0_f64;
    let k_si = k_mm * 1000.0;
    let m_si = m_g / 1000.0;
    let want_sdof = (k_si / m_si).sqrt() / (2.0 * PI);
    let f_sdof = single_dof_frequency_hz(Fix128::from_int(250), Fix128::from_int(750)).to_f64();
    check("single_dof_frequency_hz", f_sdof, want_sdof, 1e-6);

    // Degenerate: zero mass must return exactly zero (division-by-zero guard,
    // not a computed near-zero value).
    let f_zero_mass = single_dof_frequency_hz(Fix128::from_int(250), Fix128::ZERO).to_f64();
    println!("[modal] single_dof_frequency_hz(k, 0) = {f_zero_mass} (expect exactly 0.0)");
    assert_eq!(
        f_zero_mass, 0.0,
        "zero mass must short-circuit to exactly 0.0"
    );

    // -----------------------------------------------------------------
    // 2. BeamBoundary::lambda_squared(): Blevins Table 8-1 eigenvalues
    //    (βL): 1.875_104_069 (cantilever), π (pinned), 4.730_040_745
    //    (clamped-clamped). Checked directly, not through
    //    beam_natural_frequency_hz.
    // -----------------------------------------------------------------
    let blevins_beta_l = [
        (BeamBoundary::Cantilever, 1.875_104_069_f64),
        (BeamBoundary::SimplySupported, PI),
        (BeamBoundary::ClampedClamped, 4.730_040_745_f64),
    ];
    for (bc, beta_l) in blevins_beta_l {
        let want = beta_l * beta_l;
        let got = bc.lambda_squared().to_f64();
        check(&format!("lambda_squared({bc:?})"), got, want, 2e-4);
    }
    // Physical ordering must hold for the eigenvalues themselves, not just
    // the derived frequency (the two could diverge if a later edit folded
    // a boundary-dependent sign into the frequency formula instead).
    assert!(
        BeamBoundary::Cantilever.lambda_squared() < BeamBoundary::SimplySupported.lambda_squared()
    );
    assert!(
        BeamBoundary::SimplySupported.lambda_squared()
            < BeamBoundary::ClampedClamped.lambda_squared()
    );

    // -----------------------------------------------------------------
    // 3. beam_natural_frequency_hz: solid circular ABS rod, L = 250 mm,
    //    d = 8 mm (different section/material/length from the Rectangular
    //    PLA L=100mm case in tests/engineering_oracles_solid.rs).
    //    A = π d²/4, I = π d⁴/64 (CrossSection::Circular, src/beam_stress.rs).
    // -----------------------------------------------------------------
    let (e_gpa, rho_gcc, d_mm, l_mm) = (2.3_f64, 1.04_f64, 8.0_f64, 250.0_f64);
    let (e_pa, rho_si, d_si, l_si) = (e_gpa * 1e9, rho_gcc * 1000.0, d_mm * 1e-3, l_mm * 1e-3);
    let a_si = PI / 4.0 * d_si * d_si;
    let i_si = PI / 64.0 * (d_si * d_si * d_si * d_si);
    let base = (e_pa * i_si / (rho_si * a_si)).sqrt() / (2.0 * PI * l_si * l_si);
    let section = CrossSection::Circular {
        diameter_mm: Fix128::from_int(8),
    };
    let material = MaterialProperties::abs();
    for (bc, beta_l) in blevins_beta_l {
        let want = beta_l * beta_l * base;
        let f = beam_natural_frequency_hz(&section, Fix128::from_int(250), &material, bc).to_f64();
        check(
            &format!("beam_natural_frequency_hz(Circular ABS, {bc:?})"),
            f,
            want,
            2e-4,
        );
    }

    // Degenerate: zero length must return exactly zero.
    let f_zero_l =
        beam_natural_frequency_hz(&section, Fix128::ZERO, &material, BeamBoundary::Cantilever)
            .to_f64();
    println!("[modal] beam_natural_frequency_hz(.., L=0, ..) = {f_zero_l} (expect exactly 0.0)");
    assert_eq!(
        f_zero_l, 0.0,
        "zero length must short-circuit to exactly 0.0"
    );

    // -----------------------------------------------------------------
    // 4. plate_natural_frequency_hz: square ABS plate, h = 3 mm,
    //    a = b = 120 mm (Leissa SSSS f11; different material/thickness/
    //    aspect ratio from the a=100,b=150 PLA case in
    //    tests/engineering_oracles_solid.rs).
    // -----------------------------------------------------------------
    let (nu, h_mm, side_mm) = (0.35_f64, 3.0_f64, 120.0_f64);
    let h_si = h_mm * 1e-3;
    let side_si = side_mm * 1e-3;
    let d_flex = e_pa * (h_si * h_si * h_si) / (12.0 * (1.0 - nu * nu));
    let want_plate = PI / 2.0 * (d_flex / (rho_si * h_si)).sqrt() * (2.0 / (side_si * side_si));
    let f_plate = plate_natural_frequency_hz(
        &material,
        Fix128::from_ratio(35, 100),
        Fix128::from_int(3),
        Fix128::from_int(120),
        Fix128::from_int(120),
    )
    .to_f64();
    check(
        "plate_natural_frequency_hz(square ABS)",
        f_plate,
        want_plate,
        1e-6,
    );

    // Degenerate: zero side length must return exactly zero.
    let f_zero_side = plate_natural_frequency_hz(
        &material,
        Fix128::from_ratio(35, 100),
        Fix128::from_int(3),
        Fix128::ZERO,
        Fix128::from_int(120),
    )
    .to_f64();
    println!(
        "[modal] plate_natural_frequency_hz(.., a=0, ..) = {f_zero_side} (expect exactly 0.0)"
    );
    assert_eq!(
        f_zero_side, 0.0,
        "zero side must short-circuit to exactly 0.0"
    );

    // -----------------------------------------------------------------
    // 5. torsional_frequency_hz: aluminium shaft, d = 12 mm, L = 150 mm,
    //    G = 26 000 MPa, disc I_p = 2.0e6 g·mm² (different material/
    //    geometry from the steel-shaft case in
    //    tests/engineering_oracles_solid.rs).
    //    f = (1/2π) √(G J / (I_p L)), SI scale factor 10^3 (see module doc
    //    at src/modal.rs:205-209: k_theta [N·mm] -> 1e-3 N·m,
    //    I_p [g·mm²] -> 1e-9 kg·m², so ω_SI = 10^3 √(k_eng/I_eng)).
    // -----------------------------------------------------------------
    let (g_mpa, d_shaft_mm, l_shaft_mm, ip_g_mm2) = (26_000.0_f64, 12.0_f64, 150.0_f64, 2.0e6_f64);
    let j_mm4 = PI * (d_shaft_mm * d_shaft_mm * d_shaft_mm * d_shaft_mm) / 32.0;
    let k_theta_si = g_mpa * j_mm4 / l_shaft_mm * 1e-3; // N·m/rad
    let ip_si = ip_g_mm2 * 1e-9; // kg·m²
    let want_tors = (k_theta_si / ip_si).sqrt() / (2.0 * PI);
    let f_tors = torsional_frequency_hz(
        Fix128::from_int(26_000),
        Fix128::from_f64(j_mm4),
        Fix128::from_f64(ip_g_mm2),
        Fix128::from_int(150),
    )
    .to_f64();
    check(
        "torsional_frequency_hz(aluminium shaft)",
        f_tors,
        want_tors,
        1e-5,
    );

    // Degenerate: zero shaft length must return exactly zero.
    let f_zero_shaft_l = torsional_frequency_hz(
        Fix128::from_int(26_000),
        Fix128::from_f64(j_mm4),
        Fix128::from_f64(ip_g_mm2),
        Fix128::ZERO,
    )
    .to_f64();
    println!("[modal] torsional_frequency_hz(.., L=0) = {f_zero_shaft_l} (expect exactly 0.0)");
    assert_eq!(
        f_zero_shaft_l, 0.0,
        "zero shaft length must short-circuit to exactly 0.0"
    );

    println!("[modal] all five modal.rs formulas checked against independently hand-derived closed forms");
}
