//! Per-material fatigue and buckling in `StructuralSolver`: PLA vs SUS304 vs
//! A5052.
//!
//! `new` selects the S-N curve from the material (SUS304 / A5052 metal
//! presets, the FDM rule of thumb for PLA), `with_sn_curve` overrides it, and
//! the plate / snap-through queries use the material's Young's modulus.
//!
//! ```text
//! 15 x 20 cantilever, 200 mm, F = 1500 N -> sigma = 300 MPa, one cycle per step
//! SUS304  N = floor(1e7 (240/300)^10) = 1 073 741  -> D = 1 / 1 073 741 per step
//! A5052   N = floor(5e6 (92/300)^6)   = 4 158      -> D = 1 / 4 158 per step
//! PLA     N(300 MPa) < 1 cycle                     -> D = 1 per step
//! Basquin inverse at N = 1e5: SUS304 240 * 100^0.1 = 380.4 MPa
//! plate t = 1, b = 40, nu = 0.3, k = 4: sigma_cr = k pi^2 E / (12 (1 - nu^2)) (t/b)^2
//! snap-through h = 5, L = 100, A = 300: P = (16 / (3 sqrt 3)) E A (h/L)^3
//! ```
//!
//! ```bash
//! cargo run --example structural_fatigue_buckling_per_material
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::fatigue::SnCurve;
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::structural_solver::StructuralSolver;

fn solver(material: MaterialProperties) -> StructuralSolver {
    StructuralSolver::new(
        CrossSection::Rectangular {
            width_mm: Fix128::from_int(15),
            height_mm: Fix128::from_int(20),
        },
        LoadCase::CantileverEndPoint {
            load_n: Fix128::from_int(1500),
            length_mm: Fix128::from_int(200),
        },
        material,
    )
}

fn main() {
    println!("=== Structural fatigue / buckling per material (sigma = 300 MPa) ===");
    let spectrum = [
        (Fix128::from_int(300), 100_000u64),
        (Fix128::from_int(350), 1_000),
    ];
    for material in [
        MaterialProperties::pla(),
        MaterialProperties::sus304(),
        MaterialProperties::a5052(),
    ] {
        let mut s = solver(material);
        let curve = s.sn_curve;
        let d_step = s.step().fatigue_damage;
        let strength = s
            .fatigue_strength_mpa(100_000)
            .map(|v| format!("{:.3}", v.to_f64()))
            .unwrap_or_else(|e| format!("{e:?}"));
        let report = s.fatigue_spectrum_report(&spectrum);
        let plate = s.plate_buckling_mpa(
            Fix128::ONE,
            Fix128::from_int(40),
            Fix128::from_ratio(3, 10),
            Fix128::from_int(4),
        );
        let snap = s
            .snap_through_load_n(Fix128::from_int(5), Fix128::from_int(100))
            .map(|v| format!("{:.3}", v.to_f64()))
            .unwrap_or_else(|e| format!("{e:?}"));
        println!(
            "{:<7} S_e={:>7.2} N_e={:>9} m={:>2} | D/step={:.6e} | S(1e5)={strength} MPa | \
             spectrum D={:.6e} safe={} | plate={:.3} MPa | snap={snap} N",
            material.name,
            curve.endurance_stress_mpa.to_f64(),
            curve.endurance_cycles,
            curve.fatigue_exponent_m,
            d_step.to_f64(),
            report.damage.to_f64(),
            report.is_safe,
            plate.to_f64(),
        );
    }

    // explicit override: a PLA beam evaluated on the SUS304 curve
    let mut overridden = solver(MaterialProperties::pla()).with_sn_curve(SnCurve::steel_sus304());
    println!(
        "PLA + with_sn_curve(SUS304): D/step={:.6e} (closed form {:.6e})",
        overridden.step().fatigue_damage.to_f64(),
        1.0 / 1_073_741.0
    );
}
