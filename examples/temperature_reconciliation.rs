//! Two owners of one temperature field, made to agree
//!
//! `ThermalModifier` and `PhaseChangeModifier` each advance their own copy of
//! the temperature on the same grid, and nothing relates the two copies.
//! `coupled_field::reconcile_mean` makes them agree on the arithmetic mean,
//! `reconcile_weighted` on a weighted one, so after either call a point that
//! had one value per owner has one value in total. The printout heats the two
//! owners differently, checks the grids match with `same_grid_as`, reconciles
//! both ways and shows the owners reading the same temperature afterwards.
//!
//! ```bash
//! cargo run --example temperature_reconciliation --features std
//! ```

use alice_physics::coupled_field::{reconcile_mean, reconcile_weighted, CoupledScalar};
use alice_physics::phase_change::{PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

fn owners() -> (ThermalModifier, PhaseChangeModifier) {
    let thermal = ThermalModifier::new(
        ThermalConfig::default(),
        8,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    );
    let phase = PhaseChangeModifier::new(
        PhaseChangeConfig::default(),
        8,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    );
    (thermal, phase)
}

fn main() {
    let (mut thermal, mut phase) = owners();
    // Heat the two owners differently at the same point.
    thermal.apply_heat_at(0.5, 0.5, 0.5, 40.0, 0.25);
    phase.apply_heat_at(0.5, 0.5, 0.5, 10.0, 0.25);
    let before = (
        thermal.temperature_at(0.5, 0.5, 0.5),
        phase.temperature_at(0.5, 0.5, 0.5),
    );
    println!(
        "[reconcile] before: {} reads {:.3}, {} reads {:.3}",
        thermal.coupled_name(),
        before.0,
        phase.coupled_name(),
        before.1
    );

    let thermal_channel = thermal.coupled_channel().expect("non-degenerate grid");
    let phase_channel = phase.coupled_channel().expect("non-degenerate grid");
    println!(
        "[reconcile] the two owners describe the same grid: {}",
        thermal_channel.same_grid_as(&phase_channel)
    );

    // Arithmetic mean.
    let mut channel = thermal_channel.clone();
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut thermal, &mut phase];
        reconcile_mean(&mut participants, &mut channel).expect("same grid");
    }
    println!(
        "[reconcile] mean: thermal {:.3}, phase {:.3}, channel {:.3}",
        thermal.temperature_at(0.5, 0.5, 0.5),
        phase.temperature_at(0.5, 0.5, 0.5),
        channel
            .sample(alice_physics::math::Vec3Fix::new(
                alice_physics::math::Fix128::from_ratio(1, 2),
                alice_physics::math::Fix128::from_ratio(1, 2),
                alice_physics::math::Fix128::from_ratio(1, 2),
            ))
            .to_f64()
    );

    // Weighted: trust the thermal owner three times as much as the phase one.
    let (mut thermal, mut phase) = owners();
    thermal.apply_heat_at(0.5, 0.5, 0.5, 40.0, 0.25);
    phase.apply_heat_at(0.5, 0.5, 0.5, 10.0, 0.25);
    let mut channel = thermal.coupled_channel().expect("non-degenerate grid");
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut thermal, &mut phase];
        reconcile_weighted(&mut participants, &[3, 1], &mut channel).expect("same grid");
    }
    println!(
        "[reconcile] weighted 3:1: thermal {:.3}, phase {:.3} (closed form {:.3})",
        thermal.temperature_at(0.5, 0.5, 0.5),
        phase.temperature_at(0.5, 0.5, 0.5),
        (3.0 * before.0 + before.1) / 4.0
    );

    // Refusals leave the owners untouched.
    let (mut thermal, mut phase) = owners();
    thermal.apply_heat_at(0.5, 0.5, 0.5, 40.0, 0.25);
    let mut channel = thermal.coupled_channel().expect("non-degenerate grid");
    let read = thermal.temperature_at(0.5, 0.5, 0.5);
    {
        let mut participants: [&mut dyn CoupledScalar; 2] = [&mut thermal, &mut phase];
        let err = reconcile_weighted(&mut participants, &[0, 0], &mut channel).unwrap_err();
        println!("[reconcile] every weight zero: {err}");
    }
    println!(
        "[reconcile] thermal still reads {:.3} after the refusal: {}",
        thermal.temperature_at(0.5, 0.5, 0.5),
        (thermal.temperature_at(0.5, 0.5, 0.5) - read).abs() < 1e-6
    );
}
