//! Oracles for `ThermalModifier::add_heat_point` (previously unwired) and the
//! behaviour the thermal module documents: heat sources feed the temperature field,
//! temperature drives melt, thermal expansion and freeze growth.
//!
//! Closed forms (diffusion and cooling off so one update is a pure source term):
//!
//! ```text
//! splat weight   w(d) = s^2 (3 - 2 s),  s = 1 - d / radius,  d < radius   (smoothstep)
//! point source   dT(node) = power * dt * w(d)
//! melt           dm = (T - T_melt) * melt_rate * dt                (droop off)
//! expansion      dist -= (T - T_amb) * k_exp
//! freeze         dist -= min((T_freeze - T) * freeze_rate, 1)
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::thermal::{HeatSource, ThermalConfig, ThermalModifier};

fn config() -> ThermalConfig {
    ThermalConfig {
        diffusion_rate: 0.0,
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        melt_temperature: 100.0,
        melt_rate: 0.5,
        droop_strength: 0.0,
        expansion_coefficient: 0.0,
        freeze_temperature: -10.0,
        freeze_rate: 0.0,
    }
}

/// 9^3 nodes on (-4, 4)^3: node spacing exactly 1
fn modifier(cfg: ThermalConfig) -> ThermalModifier {
    ThermalModifier::new(cfg, 9, (-4.0, -4.0, -4.0), (4.0, 4.0, 4.0))
}

fn smoothstep_weight(d: f32, r: f32) -> f32 {
    if d >= r {
        return 0.0;
    }
    let s = 1.0 - d / r;
    s * s * (3.0 - 2.0 * s)
}

fn node(m: &ThermalModifier, x: i32, y: i32, z: i32) -> f32 {
    m.temperature
        .get((x + 4) as usize, (y + 4) as usize, (z + 4) as usize)
}

#[test]
fn add_heat_point_records_the_source_it_was_given() {
    let mut m = modifier(config());
    m.add_heat_point(1.0, -2.0, 0.5, 30.0, 2.5);
    assert_eq!(
        m.heat_sources,
        vec![HeatSource::Point {
            x: 1.0,
            y: -2.0,
            z: 0.5,
            power: 30.0,
            radius: 2.5
        }]
    );
    m.add_heat_point(0.0, 0.0, 0.0, -1.0, 1.0);
    assert_eq!(m.heat_sources.len(), 2);
}

#[test]
fn point_source_deposits_power_dt_times_the_smoothstep_weight() {
    let mut m = modifier(config());
    m.add_heat_point(0.0, 0.0, 0.0, 50.0, 2.0);
    let dt = 0.1;
    m.update(dt);
    for (x, y, z) in [
        (0, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (1, 1, 0),
        (1, 1, 1),
        (2, 0, 0),
        (0, 0, 3),
    ] {
        let d = ((x * x + y * y + z * z) as f32).sqrt();
        let want = 20.0 + 50.0 * dt * smoothstep_weight(d, 2.0);
        assert!(
            (node(&m, x, y, z) - want).abs() < 1e-4,
            "node ({x},{y},{z}) d = {d}: {} vs {want}",
            node(&m, x, y, z)
        );
    }
    // the source node gets exactly power * dt (weight 1); a node at d = radius gets nothing
    assert!((node(&m, 0, 0, 0) - 25.0).abs() < 1e-5);
    assert_eq!(node(&m, 2, 0, 0), 20.0);
}

#[test]
fn sources_superpose_and_accumulate_every_step() {
    let mut m = modifier(config());
    m.add_heat_point(0.0, 0.0, 0.0, 50.0, 2.0);
    m.add_heat_point(0.0, 0.0, 0.0, 10.0, 3.0);
    m.add_heat_point(0.0, 0.0, 0.0, -20.0, 1.5);
    for _ in 0..3 {
        m.update(0.1);
    }
    // centre weight is 1 for every source: 3 steps * (50 + 10 - 20) * 0.1
    assert!((node(&m, 0, 0, 0) - (20.0 + 3.0 * 4.0)).abs() < 1e-4);
}

#[test]
fn zero_radius_source_deposits_nothing() {
    // `splat` keeps cells with distance strictly below the radius: radius 0 selects none
    let mut m = modifier(config());
    m.add_heat_point(0.0, 0.0, 0.0, 1000.0, 0.0);
    m.update(0.1);
    assert!(m.temperature.data.iter().all(|&t| t == 20.0));
}

#[test]
fn source_outside_the_grid_still_heats_the_nodes_inside_its_radius() {
    let mut m = modifier(config());
    m.add_heat_point(5.0, 0.0, 0.0, 100.0, 2.0); // 1 beyond the +x face
    m.update(0.1);
    // face node (4,0,0): d = 1 -> weight 0.5
    assert!((node(&m, 4, 0, 0) - (20.0 + 10.0 * 0.5)).abs() < 1e-4);
    assert_eq!(node(&m, 3, 0, 0), 20.0);
}

#[test]
fn volume_source_heats_exactly_the_nodes_inside_its_box() {
    let mut m = modifier(config());
    m.heat_sources.push(HeatSource::Volume {
        min: (-1.0, -1.0, -1.0),
        max: (1.0, 1.0, 1.0),
        power: 40.0,
    });
    m.update(0.5);
    let mut hot = 0;
    for z in -4i32..=4 {
        for y in -4i32..=4 {
            for x in -4i32..=4 {
                let inside = x.abs() <= 1 && y.abs() <= 1 && z.abs() <= 1;
                let want = if inside { 40.0 } else { 20.0 };
                assert!((node(&m, x, y, z) - want).abs() < 1e-4, "({x},{y},{z})");
                hot += i32::from(inside);
            }
        }
    }
    assert_eq!(hot, 27, "inclusive box bounds");
}

#[test]
fn melt_accumulates_excess_over_the_melt_point_times_rate_and_dt() {
    let mut m = modifier(config());
    // node (0,0,0): T = 20 + 1000 * 0.1 = 120, excess 20
    m.add_heat_point(0.0, 0.0, 0.0, 1000.0, 1.0);
    m.update(0.1);
    let melt = m.melt_accumulator.get(4, 4, 4);
    assert!((melt - 20.0 * 0.5 * 0.1).abs() < 1e-4, "{melt}");
    // a node that stays below the melt point accumulates nothing
    assert_eq!(m.melt_accumulator.get(0, 0, 0), 0.0);
    // the surface recedes by the accumulated melt (positive = material removed)
    let d = m.modify_distance(0.0, 0.0, 0.0, -1.0);
    assert!((d - (-1.0 + melt)).abs() < 1e-5);
}

#[test]
fn droop_moves_melt_down_and_removes_only_half_of_what_it_moves() {
    let mut cfg = config();
    cfg.droop_strength = 0.5;
    let mut m = modifier(cfg);
    m.temperature.data.fill(20.0);
    m.melt_accumulator.set(4, 5, 4, 1.0); // one node with melt, nothing above melt temperature
    m.update(0.1);
    // transfer = melt * droop * dt = 0.05: below gains 0.05, source loses 0.025
    assert!((m.melt_accumulator.get(4, 4, 4) - 0.05).abs() < 1e-6);
    assert!((m.melt_accumulator.get(4, 5, 4) - 0.975).abs() < 1e-6);
}

#[test]
fn expansion_and_freeze_follow_their_closed_forms_on_a_uniform_field() {
    let mut cfg = config();
    cfg.expansion_coefficient = 0.01;
    cfg.freeze_rate = 0.02;
    let mut m = modifier(cfg);
    // hot: dT = 80 -> dist -= 0.8
    m.temperature.data.fill(100.0);
    // melt_temperature is 100 and the melt accumulator is empty
    assert!((m.modify_distance(0.0, 0.0, 0.0, 1.0) - 0.2).abs() < 1e-5);
    // ambient: untouched
    m.temperature.data.fill(20.0);
    assert_eq!(m.modify_distance(0.0, 0.0, 0.0, 1.0), 1.0);
    // cold: 30 below the freeze point -> 0.6 growth
    m.temperature.data.fill(-40.0);
    assert!((m.modify_distance(0.0, 0.0, 0.0, 1.0) - 0.4).abs() < 1e-5);
    // growth is capped at 1.0 however cold it is
    m.temperature.data.fill(-1000.0);
    assert!((m.modify_distance(0.0, 0.0, 0.0, 1.0) - 0.0).abs() < 1e-5);
}

#[test]
fn cooling_relaxes_toward_ambient_exponentially() {
    let mut cfg = config();
    cfg.cooling_rate = 2.0;
    let mut m = modifier(cfg);
    m.temperature.data.fill(120.0);
    m.update(0.25);
    let want = 20.0 + 100.0 * (-2.0f32 * 0.25).exp();
    assert!((m.temperature_at(0.0, 0.0, 0.0) - want).abs() < 1e-3);
}

#[test]
fn diffusion_conserves_total_heat_and_spreads_it() {
    let mut cfg = config();
    cfg.diffusion_rate = 0.1;
    let mut m = modifier(cfg);
    m.temperature.data.fill(20.0);
    m.temperature.set(4, 4, 4, 520.0);
    let before: f32 = m.temperature.data.iter().sum();
    m.update(0.5);
    let after: f32 = m.temperature.data.iter().sum();
    assert!((before - after).abs() < 0.5, "{before} vs {after}");
    assert!(node(&m, 0, 0, 0) < 520.0 && node(&m, 1, 0, 0) > 20.0);
}

#[test]
fn disabled_modifier_ignores_sources() {
    let mut m = modifier(config());
    m.add_heat_point(0.0, 0.0, 0.0, 1000.0, 2.0);
    m.melt_accumulator.data.fill(0.4);
    m.enabled = false;
    m.update(1.0);
    assert!(m.temperature.data.iter().all(|&t| t == 20.0));
    assert_eq!(
        m.modify_distance(0.0, 0.0, 0.0, 7.0),
        7.0,
        "an enabled one would add the 0.4 melt"
    );
    assert!(!m.is_active());
    assert_eq!(m.name(), "thermal");
}

/// Known defect in `ScalarField3D::splat` (src/sim_field.rs, not part of this
/// worker's file set): the index range of all three axes is derived from
/// `radius * inv_cell_size.0` (the x cell size), so on a grid with unequal cell sizes the
/// y / z ranges are too small and nodes inside the radius are skipped.
#[test]
#[ignore = "known defect: ScalarField3D::splat uses the x cell size for the y/z index range (Backlog ALICE-Physics sim_field splat anisotropic)"]
fn point_source_reaches_every_node_inside_its_radius_on_an_anisotropic_grid() {
    // x spacing 1, y spacing 0.25
    let mut m = ThermalModifier::new(config(), 9, (-4.0, -1.0, -4.0), (4.0, 1.0, 4.0));
    m.add_heat_point(0.0, 0.0, 0.0, 100.0, 1.0);
    m.update(0.1);
    // node (4, 7, 4) is at y = 0.75: distance 0.75 < radius 1
    let want = 20.0 + 10.0 * smoothstep_weight(0.75, 1.0);
    let got = m.temperature.get(4, 7, 4);
    assert!((got - want).abs() < 1e-4, "{got} vs {want}");
}
