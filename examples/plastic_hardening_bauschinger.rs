//! Isotropic vs kinematic hardening under a load-reverse cycle
//!
//! Reaches `PlasticModel::with_hardening` through the public structural
//! solver: a PLA cantilever is bent past yield, then the load is reversed,
//! once with the solver's default isotropic model and once with the same model
//! switched to kinematic hardening.
//!
//! Closed form, from the model's documented 1-D radial return (yield function
//! `f = |σ − α| − σ_y(ε_p)`, `Δλ = f / (E + H)`, isotropic `σ_y = σ_y0 + H·ε_p`
//! with `α = 0`, kinematic `σ_y = σ_y0` with `α += sign·H·Δλ`) and the solver's
//! rule that each step's bending stress is the trial stress:
//!
//! - forward step at trial `σ₁ > σ_y0`: `Δλ₁ = (σ₁ − σ_y0) / (E + H)` for both laws;
//!   isotropic leaves `α = 0` and grows the radius to `σ_y0 + H·Δλ₁`, kinematic
//!   keeps the radius `σ_y0` and moves the centre to `α₁ = H·Δλ₁`;
//! - reverse yield therefore starts at `|σ| = σ_y0 + H·Δλ₁` (isotropic) and at
//!   `|σ| = σ_y0 − H·Δλ₁` (kinematic): the Bauschinger effect, a gap of
//!   `2·H·Δλ₁` between the two laws;
//! - a reverse trial `σ₂ < 0` yields by `Δλ₂ = (|σ₂| − σ_y0 − H·Δλ₁) / (E + H)`
//!   (isotropic) or `Δλ₂ = (|σ₂| − σ_y0 + H·Δλ₁) / (E + H)` (kinematic), when
//!   positive, and the kinematic centre moves to `α₂ = α₁ − H·Δλ₂`.
//!
//! The trial stresses are the solver's own reported bending stresses (the
//! input of the return map, `M / Z`), so every expected value below is built
//! from the material constants and those inputs only.
//!
//! Run with: `cargo run --example plastic_hardening_bauschinger`

use alice_physics::beam_stress::{CrossSection, LoadCase};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;
use alice_physics::plastic::{HardeningType, PlasticModel};
use alice_physics::structural_solver::StructuralSolver;

const TOL: f64 = 1e-12;

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= TOL * want.abs().max(1.0),
        "{what}: got {got:.15}, closed form {want:.15}"
    );
}

struct Cycle {
    sigma_1: f64,
    sigma_2: f64,
    eps_p_1: f64,
    alpha_1: f64,
    eps_p_2: f64,
    alpha_2: f64,
    reverse_yielded: bool,
}

const LENGTH_MM: i64 = 100;

fn section() -> CrossSection {
    CrossSection::Rectangular {
        width_mm: Fix128::from_int(10),
        height_mm: Fix128::from_int(10),
    }
}

fn cantilever(load_n: Fix128) -> LoadCase {
    LoadCase::CantileverEndPoint {
        load_n,
        length_mm: Fix128::from_int(LENGTH_MM),
    }
}

/// End load whose root bending stress is `-sigma_mpa` (`σ = F·L / Z`).
fn reverse_load(sigma_mpa: Fix128) -> Fix128 {
    -(sigma_mpa * section().section_modulus_mm3() / Fix128::from_int(LENGTH_MM))
}

fn run_cycle(law: HardeningType, forward_n: Fix128, reverse_mpa: Fix128) -> Cycle {
    let material = MaterialProperties::pla();
    let mut solver = StructuralSolver::new(section(), cantilever(forward_n), material);
    solver.plastic_model = PlasticModel::from_fdm_material(&material).with_hardening(law);
    assert_eq!(solver.plastic_model.hardening_type, law);

    let forward = solver.step();
    let (eps_p_1, alpha_1) = (
        solver.state.equivalent_plastic_strain.to_f64(),
        solver.state.back_stress_mpa.to_f64(),
    );
    solver.load = cantilever(reverse_load(reverse_mpa));
    let reverse = solver.step();
    let eps_p_2 = solver.state.equivalent_plastic_strain.to_f64();
    Cycle {
        sigma_1: forward.bending_stress_mpa.to_f64(),
        sigma_2: reverse.bending_stress_mpa.to_f64(),
        eps_p_1,
        alpha_1,
        eps_p_2,
        alpha_2: solver.state.back_stress_mpa.to_f64(),
        reverse_yielded: eps_p_2 > eps_p_1,
    }
}

fn main() {
    let model = PlasticModel::from_fdm_material(&MaterialProperties::pla());
    let e = model.youngs_modulus_mpa.to_f64();
    let h = model.hardening_modulus_mpa.to_f64();
    let sy = model.yield_strength_mpa.to_f64();
    assert_eq!(
        model.hardening_type,
        HardeningType::Isotropic,
        "solver default"
    );
    assert_eq!(
        model
            .with_hardening(HardeningType::Kinematic)
            .with_hardening(HardeningType::Isotropic),
        model,
        "with_hardening changes the law and nothing else"
    );

    // Forward load: M = 120 N · 100 mm, Z = 10·10²/6 mm³ → σ₁ = 72 MPa > σ_y0 = 50.
    let forward_n = Fix128::from_int(120);
    let probe = |law, reverse_mpa: Fix128| run_cycle(law, forward_n, reverse_mpa);

    // Reverse at |σ₂| = σ_y0: inside the isotropic surface, outside the
    // translated kinematic one.
    let sy_fix = model.yield_strength_mpa;
    for law in [HardeningType::Isotropic, HardeningType::Kinematic] {
        let c = probe(law, sy_fix);
        let dl1 = (c.sigma_1 - sy) / (e + h);
        assert!(c.sigma_1 > sy, "the forward step must yield");
        close(c.sigma_2, -sy, "reverse trial stress");
        close(c.eps_p_1, dl1, "forward Δλ₁");
        let (alpha_1, onset, dl2) = match law {
            HardeningType::Kinematic => {
                (h * dl1, sy - h * dl1, (-c.sigma_2 - sy + h * dl1) / (e + h))
            }
            _ => (0.0, sy + h * dl1, (-c.sigma_2 - sy - h * dl1) / (e + h)),
        };
        close(c.alpha_1, alpha_1, "back stress after the forward step");
        let dl2 = dl2.max(0.0);
        close(c.eps_p_2 - c.eps_p_1, dl2, "reverse Δλ₂");
        let alpha_2 = if matches!(law, HardeningType::Kinematic) {
            alpha_1 - h * dl2
        } else {
            0.0
        };
        close(c.alpha_2, alpha_2, "back stress after the reverse step");
        println!(
            "[plastic_hardening] {law:?}: σ₁={:.4} MPa Δλ₁={dl1:.6e} reverse onset |σ|={onset:.4} MPa; \
             at |σ₂|={:.4} yielded={} Δλ₂={dl2:.6e} α₂={alpha_2:.4}",
            c.sigma_1, -c.sigma_2, c.reverse_yielded
        );
        let want_yield = matches!(law, HardeningType::Kinematic);
        assert_eq!(c.reverse_yielded, want_yield, "{law:?} at |σ₂| = σ_y0");
    }

    // Bracket each law's reverse-yield onset at ±1 % and check the gap 2·H·Δλ₁.
    let base = probe(HardeningType::Isotropic, sy_fix);
    let dl1 = (base.sigma_1 - sy) / (e + h);
    let onsets = [
        (HardeningType::Isotropic, sy + h * dl1),
        (HardeningType::Kinematic, sy - h * dl1),
    ];
    for (law, onset) in onsets {
        for (scale, want_yield) in [(0.99, false), (1.01, true)] {
            let c = probe(law, Fix128::from_f64(onset * scale));
            assert_eq!(
                c.reverse_yielded, want_yield,
                "{law:?}: reverse at {scale} × onset {onset:.4} MPa"
            );
        }
    }
    let gap = onsets[0].1 - onsets[1].1;
    close(gap, 2.0 * h * dl1, "Bauschinger gap");
    println!(
        "[plastic_hardening] reverse-yield onset isotropic {:.4} MPa, kinematic {:.4} MPa, gap {gap:.4} = 2·H·Δλ₁",
        onsets[0].1, onsets[1].1
    );

    // Full reversal |σ₂| = σ₁: both laws yield, kinematic by 2·H·Δλ₁/(E+H) more.
    let iso = probe(HardeningType::Isotropic, Fix128::from_f64(base.sigma_1));
    let kin = probe(HardeningType::Kinematic, Fix128::from_f64(base.sigma_1));
    let s2 = -iso.sigma_2;
    close(
        iso.eps_p_2 - iso.eps_p_1,
        (s2 - sy - h * dl1) / (e + h),
        "isotropic full reversal",
    );
    close(
        kin.eps_p_2 - kin.eps_p_1,
        (-kin.sigma_2 - sy + h * dl1) / (e + h),
        "kinematic full reversal",
    );
    println!(
        "[plastic_hardening] full reversal: Δλ₂ isotropic {:.6e}, kinematic {:.6e}",
        iso.eps_p_2 - iso.eps_p_1,
        kin.eps_p_2 - kin.eps_p_1
    );
    println!("[plastic_hardening] all closed-form checks passed");
}
