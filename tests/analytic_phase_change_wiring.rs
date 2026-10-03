//! Oracles for `phase_change::{Phase, PhaseChangeModifier::phase_at}` (previously
//! unwired) and for the enthalpy-method behaviour the module documents.
//!
//! Enthalpy method (Voller & Cross 1981; Carslaw & Jaeger ch. XI), heat capacity 1,
//! latent heats `L_f`, `L_v` in kelvin-equivalent units. For a cell holding total
//! enthalpy `H = T + latent` (diffusion and cooling off), the equilibrium state is
//!
//! ```text
//! H <= Tm                    solid,  T = H
//! Tm <= H < Tm + Lf          solid,  T = Tm            (melting plateau)
//! Tm + Lf <= H <= Tb + Lf    liquid, T = H - Lf
//! Tb + Lf <= H < Tb+Lf+Lv    liquid, T = Tb            (boiling plateau)
//! H >= Tb + Lf + Lv          gas,    T = H - Lf - Lv
//! ```
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::phase_change::{Phase, PhaseChangeConfig, PhaseChangeModifier};
use alice_physics::sim_modifier::PhysicsModifier;

const TM: f32 = 100.0;
const TB: f32 = 300.0;
const LF: f32 = 10.0;
const LV: f32 = 20.0;

fn config() -> PhaseChangeConfig {
    PhaseChangeConfig {
        melt_temperature: TM,
        boil_temperature: TB,
        latent_heat_fusion: LF,
        latent_heat_vaporization: LV,
        diffusion_rate: 0.0,
        ambient_temperature: 20.0,
        cooling_rate: 0.0,
        ..Default::default()
    }
}

fn modifier() -> PhaseChangeModifier {
    PhaseChangeModifier::new(config(), 5, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
}

/// the state of cell 0 after one update, for a uniform start `(T, phase, latent)`
fn run(t: f32, phase: f32, latent: f32, dt: f32) -> (f32, Phase, f32) {
    let mut m = modifier();
    m.temperature.data.fill(t);
    m.phase.data.fill(phase);
    m.latent_heat.data.fill(latent);
    m.update(dt);
    (
        m.temperature_at(0.0, 0.0, 0.0),
        m.phase_at(0.0, 0.0, 0.0),
        m.latent_heat.data[0],
    )
}

fn expected(h: f32) -> (f32, Phase) {
    if h <= TM {
        (h, Phase::Solid)
    } else if h < TM + LF {
        (TM, Phase::Solid)
    } else if h <= TB + LF {
        (h - LF, Phase::Liquid)
    } else if h < TB + LF + LV {
        (TB, Phase::Liquid)
    } else {
        (h - LF - LV, Phase::Gas)
    }
}

// ---------------------------------------------------------- Phase / phase_at

#[test]
fn phase_enum_has_the_documented_discriminants() {
    assert_eq!(Phase::Solid as u8, 0);
    assert_eq!(Phase::Liquid as u8, 1);
    assert_eq!(Phase::Gas as u8, 2);
    assert_ne!(Phase::Solid, Phase::Liquid);
    assert_ne!(Phase::Liquid, Phase::Gas);
}

#[test]
fn phase_at_decodes_the_stored_value_with_half_integer_thresholds() {
    // uniform field => trilinear sample returns the value itself
    for (v, want) in [
        (-1.0, Phase::Solid),
        (0.0, Phase::Solid),
        (0.49, Phase::Solid),
        (0.5, Phase::Liquid),
        (1.0, Phase::Liquid),
        (1.49, Phase::Liquid),
        (1.5, Phase::Gas),
        (2.0, Phase::Gas),
        (7.0, Phase::Gas),
    ] {
        let mut m = modifier();
        m.phase.data.fill(v);
        for p in [(0.0, 0.0, 0.0), (1.3, -0.7, 0.2), (-2.0, 2.0, -2.0)] {
            assert_eq!(m.phase_at(p.0, p.1, p.2), want, "stored {v} at {p:?}");
        }
    }
}

#[test]
fn phase_at_samples_trilinearly_between_nodes() {
    // x-ramp of the stored phase: node ix holds ix * 0.5 (0, .5, 1, 1.5, 2)
    let mut m = modifier();
    for iz in 0..5 {
        for iy in 0..5 {
            for ix in 0..5 {
                m.phase.set(ix, iy, iz, ix as f32 * 0.5);
            }
        }
    }
    // world x = -2 + ix; sample 0.25 below node 1 (x = -1): value 0.5 - 0.125 = 0.375 -> Solid
    assert_eq!(m.phase_at(-1.25, 0.0, 0.0), Phase::Solid);
    assert_eq!(m.phase_at(-1.0, 0.0, 0.0), Phase::Liquid);
    assert_eq!(m.phase_at(0.4, 0.0, 0.0), Phase::Liquid); // 1.0 + 0.2
    assert_eq!(m.phase_at(0.9, 0.0, 0.0), Phase::Liquid); // 1.45
    assert_eq!(m.phase_at(1.1, 0.0, 0.0), Phase::Gas); // 1.55
}

#[test]
fn new_modifier_starts_solid_at_ambient() {
    let m = modifier();
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Solid);
    assert!((m.temperature_at(0.3, -1.0, 1.7) - 20.0).abs() < 1e-6);
    assert!(m.enabled && m.is_active());
    assert_eq!(m.name(), "phase_change");
}

// ----------------------------------------------------------- enthalpy

#[test]
fn enthalpy_closed_form_holds_across_the_whole_temperature_range() {
    let mut h = -50.0f32;
    while h < 700.0 {
        let (t, ph, lh) = run(h, 0.0, 0.0, 0.01);
        let (want_t, want_ph) = expected(h);
        assert_eq!(ph, want_ph, "H = {h}");
        assert!(
            (t - want_t).abs() < 1e-3,
            "H = {h}: T = {t}, expected {want_t}"
        );
        assert!(
            (t + lh - h).abs() < 1e-3,
            "H = {h}: T + latent = {} must be conserved",
            t + lh
        );
        h += 7.3;
    }
}

#[test]
fn named_transition_states() {
    // 105 K above nothing: 5 K into the melting plateau, still solid
    let (t, ph, lh) = run(105.0, 0.0, 0.0, 0.01);
    assert_eq!((t, ph, lh), (100.0, Phase::Solid, 5.0));
    // fusion buffer full: liquid, 5 K above the melting point
    let (t, ph, lh) = run(115.0, 0.0, 0.0, 0.01);
    assert_eq!((t, ph, lh), (105.0, Phase::Liquid, 10.0));
    // 400: through both transitions in one update
    let (t, ph, lh) = run(400.0, 0.0, 0.0, 0.01);
    assert_eq!((t, ph, lh), (370.0, Phase::Gas, 30.0));
}

#[test]
fn cooling_a_liquid_refreezes_and_conserves_enthalpy() {
    // liquid with full fusion buffer cooled 15 K below Tm: gives back 10 K, freezes
    let (t, ph, lh) = run(85.0, 1.0, 10.0, 0.01);
    assert_eq!((t, ph, lh), (95.0, Phase::Solid, 0.0));
    assert!((t + lh - 95.0).abs() < 1e-5);
    // only 5 K below Tm: stays liquid, half the buffer returned
    let (t, ph, lh) = run(95.0, 1.0, 10.0, 0.01);
    assert_eq!((t, ph, lh), (100.0, Phase::Liquid, 5.0));
}

#[test]
fn condensing_a_gas_returns_to_liquid_through_the_vaporization_buffer() {
    // gas with a full buffer (30), 10 K below Tb: drains 10 K of the vaporization buffer, still gas
    let (t, ph, lh) = run(290.0, 2.0, 30.0, 0.01);
    assert_eq!((t, ph, lh), (300.0, Phase::Gas, 20.0));
    // 50 K below Tb: drains all 20 K of it, becomes liquid with the fusion buffer left
    let (t, ph, lh) = run(250.0, 2.0, 30.0, 0.01);
    assert_eq!((t, ph, lh), (270.0, Phase::Liquid, 10.0));
}

#[test]
fn transition_state_does_not_depend_on_the_step_size() {
    for h in [105.0, 115.0, 250.0, 320.0, 400.0] {
        let small = run(h, 0.0, 0.0, 1e-4);
        let large = run(h, 0.0, 0.0, 10.0);
        assert_eq!(small, large, "H = {h}");
    }
}

#[test]
fn zero_latent_heats_make_the_transition_sharp() {
    let mut m = PhaseChangeModifier::new(
        PhaseChangeConfig {
            latent_heat_fusion: 0.0,
            latent_heat_vaporization: 0.0,
            ..config()
        },
        5,
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
    );
    m.temperature.data.fill(101.0);
    m.update(0.01);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid);
    assert!((m.temperature_at(0.0, 0.0, 0.0) - 101.0).abs() < 1e-5);
}

#[test]
fn negative_latent_heat_is_treated_as_zero() {
    let mut m = PhaseChangeModifier::new(
        PhaseChangeConfig {
            latent_heat_fusion: -5.0,
            ..config()
        },
        5,
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
    );
    m.temperature.data.fill(150.0);
    m.update(0.01);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Liquid);
    assert!((m.temperature_at(0.0, 0.0, 0.0) - 150.0).abs() < 1e-4);
    // the clamped fusion heat is 0, so the total is 20 (not 15): H = 400 lands at T = 380
    m.temperature.data.fill(400.0);
    m.phase.data.fill(0.0);
    m.latent_heat.data.fill(0.0);
    m.update(0.01);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Gas);
    assert!(
        (m.temperature_at(0.0, 0.0, 0.0) - 380.0).abs() < 1e-3,
        "{}",
        m.temperature_at(0.0, 0.0, 0.0)
    );
    assert_eq!(
        m.latent_heat.data[0], 20.0,
        "latent buffer holds Lf (0) + Lv (20)"
    );

    // a negative vaporization heat is clamped to 0 as well: total latent = Lf = 10
    let mut v = PhaseChangeModifier::new(
        PhaseChangeConfig {
            latent_heat_vaporization: -5.0,
            ..config()
        },
        5,
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
    );
    v.temperature.data.fill(400.0);
    v.update(0.01);
    assert_eq!(v.phase_at(0.0, 0.0, 0.0), Phase::Gas);
    assert!((v.temperature_at(0.0, 0.0, 0.0) - 390.0).abs() < 1e-3);
    assert_eq!(v.latent_heat.data[0], 10.0);
}

#[test]
fn partly_boiled_liquid_gives_back_vaporization_progress_when_it_cools() {
    // liquid with 10 K of the vaporization buffer filled (latent 20 of 30), 10 K below Tb:
    // drains 10 K, back to the fusion buffer only
    let (t, ph, lh) = run(290.0, 1.0, 20.0, 0.01);
    assert_eq!((t, ph, lh), (300.0, Phase::Liquid, 10.0));
    // 100 K below Tb: drains the 10 K, then nothing more above Tm
    let (t, ph, lh) = run(200.0, 1.0, 20.0, 0.01);
    assert_eq!((t, ph, lh), (210.0, Phase::Liquid, 10.0));
}

#[test]
fn partly_melted_solid_gives_back_its_progress_when_it_cools() {
    // solid holding 5 K of fusion latent, 10 K below Tm: returns all 5 K, stays solid
    let (t, ph, lh) = run(90.0, 0.0, 5.0, 0.01);
    assert_eq!((t, ph, lh), (95.0, Phase::Solid, 0.0));
    // only 2 K below Tm: returns 2 K of the 5
    let (t, ph, lh) = run(98.0, 0.0, 5.0, 0.01);
    assert_eq!((t, ph, lh), (100.0, Phase::Solid, 3.0));
}

// --------------------------------------------------------- SDF offsets

#[test]
fn solid_cells_do_not_move_the_surface_gas_cells_recede_at_the_documented_rate() {
    let mut solid = modifier();
    solid.update(0.1);
    assert_eq!(solid.modify_distance(0.0, 0.0, 0.0, -0.5), -0.5);

    let mut gas = modifier();
    gas.phase.data.fill(2.0);
    gas.latent_heat.data.fill(LF + LV);
    gas.temperature.data.fill(TB + 50.0);
    gas.update(0.1);
    // (expansion 0.5 + dissipation 0.3) * dt
    assert!((gas.sdf_offset.sample(0.0, 0.0, 0.0) - 0.08).abs() < 1e-6);
    assert!((gas.modify_distance(0.0, 0.0, 0.0, -0.5) - (-0.5 + 0.08)).abs() < 1e-6);
    // capped at max_offset
    for _ in 0..200 {
        gas.update(0.1);
    }
    assert!((gas.sdf_offset.sample(0.0, 0.0, 0.0) - 3.0).abs() < 1e-5);
}

#[test]
fn liquid_softening_adds_one_tenth_of_the_flow_speed_and_gravity_flow_conserves_it() {
    let mut m = modifier();
    m.phase.data.fill(1.0);
    m.latent_heat.data.fill(LF);
    m.temperature.data.fill(150.0);
    let dt = 0.05;
    m.update(dt);
    let total: f32 = m.sdf_offset.data.iter().sum();
    let softening = 1.0 * 0.1 * dt; // liquid_flow_speed (1.0) * 0.1 * dt per cell
    let want = softening * m.sdf_offset.data.len() as f32;
    assert!(
        (total - want).abs() < 1e-4 * want,
        "total {total} vs {want}"
    );
    // gravity moves offset downward: bottom layer ends up above the top layer
    let bottom = m.sdf_offset.get(2, 0, 2);
    let top = m.sdf_offset.get(2, 4, 2);
    assert!(bottom > top, "bottom {bottom} vs top {top}");
}

#[test]
fn full_step_moves_the_whole_offset_down_and_respects_the_destination_headroom() {
    // one liquid cell at the centre holding offset 1.0; dt = 2 gives fraction min(2, 1) = 1
    let mut m = modifier();
    m.phase.set(2, 2, 2, 1.0);
    m.latent_heat.set(2, 2, 2, LF);
    m.temperature.set(2, 2, 2, 150.0);
    m.sdf_offset.set(2, 2, 2, 1.0);
    m.update(2.0);
    // softening adds 1.0 * 0.1 * 2 = 0.2 first; all 1.2 then moves to the cell below
    assert!(m.sdf_offset.get(2, 2, 2).abs() < 1e-6);
    assert!((m.sdf_offset.get(2, 1, 2) - 1.2).abs() < 1e-5);
    // destination near the cap (max_offset 3): only the head-room (0.1) fits
    let mut n = modifier();
    n.phase.set(2, 2, 2, 1.0);
    n.latent_heat.set(2, 2, 2, LF);
    n.temperature.set(2, 2, 2, 150.0);
    n.sdf_offset.set(2, 2, 2, 1.0);
    n.sdf_offset.set(2, 1, 2, 2.9);
    n.update(2.0);
    assert!((n.sdf_offset.get(2, 1, 2) - 3.0).abs() < 1e-5);
    assert!(
        (n.sdf_offset.get(2, 2, 2) - 1.1).abs() < 1e-5,
        "{}",
        n.sdf_offset.get(2, 2, 2)
    );
}

#[test]
fn disabled_modifier_is_inert() {
    let mut m = modifier();
    m.temperature.data.fill(400.0);
    m.sdf_offset.data.fill(0.7);
    m.enabled = false;
    m.update(1.0);
    assert_eq!(m.phase_at(0.0, 0.0, 0.0), Phase::Solid);
    assert_eq!(
        m.modify_distance(0.0, 0.0, 0.0, 3.0),
        3.0,
        "an enabled one would add 0.7"
    );
    assert!(!m.is_active());
}
