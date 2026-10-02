//! Transient thermal conduction: material presets, stability bounds and one
//! step of each stepper, printed next to the closed form each value should
//! reproduce.
//!
//! The four presets (`steel_1018`, `aluminum_6061`, `titanium_ti6al4v`,
//! `pla_polymer`) store their fits as data, so every property is a formula
//! of the preset's own numbers: `k = c0 + c1 T`, `c_p = c0 + c1 T`,
//! `ρ = ρ_ref (1 + coeff (T − T_ref))`, `ρ c_p`, `α = k / (ρ c_p)`. The
//! stability bounds are `dx² / (2 α_max)` (1-D) and `dx² / (6 α_max)` (3-D).
//! With the unit material (`α = 1`) and `dx = 1` a single step has exact
//! dyadic arithmetic: a hot node of height `A` loses `2rA` (explicit, 1-D),
//! `6rA` (explicit, 3-D), and the hand-solved 3-cell Crank–Nicolson system
//! gives a centre deviation `A (2 − r) / (2 + 3r)`. The Picard variant with
//! constant properties repeats the linearised step bit for bit and differs
//! on a temperature-dependent preset.
//!
//! ```bash
//! cargo run --example transient_thermal_materials --features std
//! ```

use alice_physics::transient_thermal::{
    crank_nicolson_step_1d, crank_nicolson_step_1d_nonlinear, stable_dt_1d, stable_dt_3d,
    transient_step_1d, transient_step_3d, TemperatureDependence, ThermalMaterial,
};

/// The stored fit evaluated by hand in f64 (the closed form, not the module).
fn hand(dep: &TemperatureDependence, t: f64) -> f64 {
    match *dep {
        TemperatureDependence::Constant(v) => f64::from(v),
        TemperatureDependence::Polynomial { c0, c1, c2 } => {
            f64::from(c0) + f64::from(c1) * t + f64::from(c2) * t * t
        }
        TemperatureDependence::Linear {
            ref_value,
            ref_temp,
            coeff,
        } => f64::from(ref_value) * (1.0 + f64::from(coeff) * (t - f64::from(ref_temp))),
    }
}

fn hand_alpha(m: &ThermalMaterial, t: f64) -> f64 {
    hand(&m.conductivity, t) / (hand(&m.density, t) * hand(&m.specific_heat, t))
}

fn unit_material() -> ThermalMaterial {
    ThermalMaterial {
        name: "unit",
        conductivity: TemperatureDependence::Constant(1.0),
        specific_heat: TemperatureDependence::Constant(1.0),
        density: TemperatureDependence::Constant(1.0),
        reference_temperature: 300.0,
    }
}

fn main() {
    let presets = [
        ThermalMaterial::steel_1018(),
        ThermalMaterial::aluminum_6061(),
        ThermalMaterial::titanium_ti6al4v(),
        ThermalMaterial::pla_polymer(),
    ];
    let (t_eval, dx) = (300.0_f64, 1e-3_f64);
    let field = [300.0_f32, 450.0, 600.0];

    println!("[transient_thermal] property accessors at {t_eval} K (module | closed form)");
    for m in &presets {
        let tf = t_eval as f32;
        println!(
            "[transient_thermal] {:<16} k = {:.6} | {:.6}  c_p = {:.4} | {:.4}  rho = {:.4} | {:.4}",
            m.name,
            m.conductivity_at(tf),
            hand(&m.conductivity, t_eval),
            m.specific_heat_at(tf),
            hand(&m.specific_heat, t_eval),
            m.density_at(tf),
            hand(&m.density, t_eval),
        );
        println!(
            "[transient_thermal] {:<16} rho c_p = {:.4e} | {:.4e}  alpha = {:.6e} | {:.6e}",
            "",
            m.heat_capacity_at(tf),
            hand(&m.density, t_eval) * hand(&m.specific_heat, t_eval),
            m.diffusivity_at(tf),
            hand_alpha(m, t_eval),
        );
    }

    println!("[transient_thermal] stability bounds on {field:?} K, dx = {dx} m (module | dx²/(2α_max), dx²/(6α_max))");
    for m in &presets {
        let alpha_max = field
            .iter()
            .map(|&t| hand_alpha(m, f64::from(t)))
            .fold(f64::MIN, f64::max);
        println!(
            "[transient_thermal] {:<16} dt_1d = {:.6e} | {:.6e}   dt_3d = {:.6e} | {:.6e}",
            m.name,
            stable_dt_1d(&field, m, dx as f32),
            dx * dx / (2.0 * alpha_max),
            stable_dt_3d(&field, m, dx as f32),
            dx * dx / (6.0 * alpha_max),
        );
    }

    // --- one explicit step, exact arithmetic (α = 1, dx = 1, r = dt = 1/4)
    let u = unit_material();
    let mut rod = [300.0_f32, 300.0, 308.0, 300.0, 300.0];
    transient_step_1d(&mut rod, &u, 1.0, 0.25);
    println!(
        "[transient_thermal] explicit 1-D hot node A = 8, r = 1/4: {rod:?} | closed form [300, 302, 304, 302, 300]"
    );
    let n = 3;
    let mut block = vec![300.0_f32; n * n * n];
    block[13] = 316.0;
    transient_step_3d(&mut block, n, n, n, &u, 1.0, 0.0625);
    println!(
        "[transient_thermal] explicit 3-D hot cell A = 16, r = 1/16: centre {} | 310, face neighbour {} | 301, corner {} | 300",
        block[13], block[12], block[0]
    );

    // --- one Crank–Nicolson step on 3 cells, hand-solved
    let (amp, r) = (5.0_f64, 1.0_f64);
    let mut cn = [300.0_f32, 300.0 + amp as f32, 300.0];
    crank_nicolson_step_1d(&mut cn, &u, 1.0, r as f32);
    let q = amp * (2.0 - r) / (2.0 + 3.0 * r);
    let p = 2.0 * r * amp / (2.0 + 3.0 * r);
    println!(
        "[transient_thermal] Crank–Nicolson 3 cells A = {amp}, r = {r}: {cn:?} | closed form [{}, {}, {}]",
        300.0 + p,
        300.0 + q,
        300.0 + p
    );

    // --- Picard variant: constant α reproduces the linearised step bit for bit
    let before = [300.0_f32, 300.0, 305.0, 300.0, 300.0, 300.0];
    let mut lin = before;
    crank_nicolson_step_1d(&mut lin, &u, 1.0, 1.5);
    let mut nl = before;
    let iters = crank_nicolson_step_1d_nonlinear(&mut nl, &u, 1.0, 1.5, 1e-6, 10);
    println!(
        "[transient_thermal] Picard on constant α: {iters} iterations, bit-identical to linear = {}",
        nl == lin
    );
    let steel = ThermalMaterial::steel_1018();
    let dt = dx * dx / hand_alpha(&steel, 300.0);
    let before = [300.0_f32, 300.0, 300.0, 1000.0, 300.0, 300.0, 300.0];
    let mut lin = before;
    crank_nicolson_step_1d(&mut lin, &steel, dx as f32, dt as f32);
    let mut nl = before;
    let iters = crank_nicolson_step_1d_nonlinear(&mut nl, &steel, dx as f32, dt as f32, 1e-5, 20);
    let worst = lin
        .iter()
        .zip(&nl)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max);
    println!(
        "[transient_thermal] Picard on steel_1018 (α(T) varies): {iters} iterations, max |linear − Picard| = {worst:.4} K"
    );
}
