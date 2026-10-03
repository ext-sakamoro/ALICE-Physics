//! Heat, phase change and erosion acting on one SDF body.
//!
//! * `ThermalModifier::add_heat_point` feeds a smoothstep-weighted heat source into the
//!   temperature field (centre node: `power * dt`).
//! * `PhaseChangeModifier` walks a cell through the enthalpy plateaus; `phase_at` reports
//!   `Phase::{Solid, Liquid, Gas}` and the closed form of the method says where each
//!   enthalpy lands (`H = T + latent` is conserved).
//! * `ErosionModifier::compute_exposure_from_normals` exposes the upwind face of a
//!   sphere to a +x wind, `set_exposure_at` adds a local gust, and `erosion_at` reads the
//!   accumulated depth `rate (1 - hardness) v * exposure * dt`.
//!
//! ```bash
//! cargo run --release --example thermal_phase_erosion_chain --features std
//! ```

use alice_physics::erosion::{ErosionConfig, ErosionModifier, ErosionType};
use alice_physics::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::sdf_collider::ClosureSdf;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

fn main() {
    // --- thermal: one point source, diffusion and cooling off
    let mut thermal = ThermalModifier::new(
        ThermalConfig {
            diffusion_rate: 0.0,
            cooling_rate: 0.0,
            ..ThermalConfig::default()
        },
        9,
        (-4.0, -4.0, -4.0),
        (4.0, 4.0, 4.0),
    );
    thermal.add_heat_point(0.0, 0.0, 0.0, 50.0, 2.0);
    thermal.update(0.1);
    let centre = thermal.temperature_at(0.0, 0.0, 0.0);
    println!("thermal: centre node {centre:.4} K (ambient 20 + 50 * 0.1 = 25)");
    assert!((centre - 25.0).abs() < 1e-4);

    // --- phase change: enthalpy closed form for three heat contents
    let cfg = PhaseChangeConfig {
        melt_temperature: 100.0,
        boil_temperature: 300.0,
        latent_heat_fusion: 10.0,
        latent_heat_vaporization: 20.0,
        diffusion_rate: 0.0,
        cooling_rate: 0.0,
        ..PhaseChangeConfig::default()
    };
    for (enthalpy, want_phase, want_t) in [
        (105.0, Phase::Solid, 100.0),
        (115.0, Phase::Liquid, 105.0),
        (400.0, Phase::Gas, 370.0),
    ] {
        let mut m = PhaseChangeModifier::new(cfg, 5, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
        m.temperature.data.fill(enthalpy);
        m.update(0.01);
        let (phase, t) = (m.phase_at(0.0, 0.0, 0.0), m.temperature_at(0.0, 0.0, 0.0));
        println!("phase: H = {enthalpy} -> {phase:?}, T = {t}");
        assert_eq!(phase, want_phase);
        assert!((t - want_t).abs() < 1e-3);
    }

    // --- erosion: windward exposure of a sphere, then one step of wind erosion
    let sphere = ClosureSdf::new(
        |x, y, z| (x * x + y * y + z * z).sqrt() - 2.0,
        |x, y, z| {
            let l = (x * x + y * y + z * z).sqrt().max(1e-10);
            (x / l, y / l, z / l)
        },
    );
    let mut erosion = ErosionModifier::new(
        ErosionConfig {
            erosion_type: ErosionType::Wind,
            rate: 0.2,
            hardness: 0.5,
            flow_speed: 3.0,
            smoothing: 0.0,
            ..ErosionConfig::default()
        },
        9,
        (-4.0, -4.0, -4.0),
        (4.0, 4.0, 4.0),
    );
    erosion.compute_exposure_from_normals(&sphere, 0.5);
    erosion.set_exposure_at(0.0, 3.0, 0.0, 1.0, 1.0); // a gust away from the sphere
    erosion.update(0.1);
    let upwind = erosion.erosion_at(-2.0, 0.0, 0.0);
    let leeward = erosion.erosion_at(2.0, 0.0, 0.0);
    let gust = erosion.erosion_at(0.0, 3.0, 0.0);
    println!("erosion: upwind {upwind:.4} m, leeward {leeward:.4} m, gust node {gust:.4} m");
    // rate (1 - hardness) v exposure dt = 0.2 * 0.5 * 3 * 1 * 0.1 = 0.03
    assert!((upwind - 0.03).abs() < 1e-6);
    assert_eq!(leeward, 0.0);
    assert!((gust - 0.03).abs() < 1e-6);
}
