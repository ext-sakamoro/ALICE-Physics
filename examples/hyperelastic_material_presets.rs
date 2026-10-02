//! Material presets and closed-form deformation modes of
//! `alice_physics::hyperelastic`.
//!
//! Drives the three presets (`tpu_soft` / `silicone_soft` / `natural_rubber`)
//! through the two incompressible deformation modes (`Stretch::uniaxial` /
//! `Stretch::equibiaxial`), prints their invariants (`i1` / `i2` /
//! `volume_ratio`), strain energy (`strain_energy_density`), uniaxial Cauchy
//! stress (`uniaxial_cauchy_stress`) and small-strain shear modulus
//! (`small_strain_shear_modulus`).
//!
//! ⚠️ **Why this example exists.** These twelve items are a standalone
//! closed-form library distinct from the tensor-valued FEM integration in
//! `linear_elastic_fem.rs` / `cubic_elastic_fem.rs` / `quadratic_elastic_fem.rs`
//! (those call `HyperelasticModel`, `cauchy_stress`, `volumetric_modulus` and
//! `tangent_constants` directly on a deformation gradient `F`, never through
//! `Stretch`). The FEM solve never constructs a `Stretch`, so
//! `Stretch::{uniaxial, equibiaxial, i1, i2, volume_ratio, UNITY}`,
//! `HyperelasticModel::{tpu_soft, silicone_soft, natural_rubber}`,
//! `strain_energy_density`, `uniaxial_cauchy_stress` and
//! `small_strain_shear_modulus` had no production caller —
//! `scripts/wiring_guard.py` reported all twelve. This example is that
//! caller, and `tests/analytic_hyperelastic_wiring.rs` holds the closed-form
//! oracles for the same twelve items.
//!
//! ```text
//! cargo run --example hyperelastic_material_presets --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example hyperelastic_material_presets --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::hyperelastic::{
        small_strain_shear_modulus, strain_energy_density, uniaxial_cauchy_stress,
        HyperelasticModel, Stretch,
    };
    use alice_physics::math::Fix128;

    let fx = Fix128::from_f64;

    println!("[hyperelastic] undeformed reference state (Stretch::UNITY)");
    println!(
        "[hyperelastic]   l1={:.6} l2={:.6} l3={:.6} i1={:.6} i2={:.6} J={:.6}",
        Stretch::UNITY.l1.to_f64(),
        Stretch::UNITY.l2.to_f64(),
        Stretch::UNITY.l3.to_f64(),
        Stretch::UNITY.i1().to_f64(),
        Stretch::UNITY.i2().to_f64(),
        Stretch::UNITY.volume_ratio().to_f64()
    );

    let presets: [(&str, HyperelasticModel); 3] = [
        ("tpu_soft", HyperelasticModel::tpu_soft()),
        ("silicone_soft", HyperelasticModel::silicone_soft()),
        ("natural_rubber", HyperelasticModel::natural_rubber()),
    ];

    for (name, model) in presets {
        let mu0 = small_strain_shear_modulus(&model);
        println!(
            "[hyperelastic] preset {name}: small-strain shear modulus mu0={:.6} MPa",
            mu0.to_f64()
        );

        for lambda in [fx(1.0), fx(1.2), fx(2.0), fx(3.0)] {
            let uni = Stretch::uniaxial(lambda);
            let bia = Stretch::equibiaxial(lambda);
            let w_uni = strain_energy_density(&model, &uni);
            let sigma_uni = uniaxial_cauchy_stress(&model, lambda);
            let w_bia = strain_energy_density(&model, &bia);

            println!(
                "[hyperelastic]   lambda={:.3} uniaxial: J={:.6} I1={:.6} I2={:.6} W={:.6} MPa sigma={:.6} MPa",
                lambda.to_f64(),
                uni.volume_ratio().to_f64(),
                uni.i1().to_f64(),
                uni.i2().to_f64(),
                w_uni.to_f64(),
                sigma_uni.to_f64(),
            );
            println!(
                "[hyperelastic]   lambda={:.3} equibiaxial: J={:.6} I1={:.6} I2={:.6} W={:.6} MPa",
                lambda.to_f64(),
                bia.volume_ratio().to_f64(),
                bia.i1().to_f64(),
                bia.i2().to_f64(),
                w_bia.to_f64(),
            );
        }
    }
}
