//! Audit oracles for `audio_physics` (collision -> audio parameters).
//! Expected values come from the module's stated mappings written out
//! independently in `f64` (sqrt volume curve, density -> pitch with clamp,
//! resonance/damping -> decay), from range claims (volume / brightness /
//! roughness in 0..1) and from symmetry / monotonicity properties.
//!
//! Not repeated here (pinned in `analytic_audio_physics_wiring.rs`): preset
//! constants, material table growth, event classification boundary, the two
//! velocity / volume gates, per-frame cap.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};

use alice_physics::audio_physics::{
    AudioConfig, AudioEvent, AudioEventType, AudioGenerator, AudioMaterial,
};

fn contact(normal: Vec3Fix) -> Contact {
    Contact {
        depth: Fix128::from_ratio(1, 10),
        normal,
        point_a: Vec3Fix::from_int(2, 3, 4),
        point_b: Vec3Fix::from_int(2, 3, 3),
    }
}

fn one_event(
    cfg: AudioConfig,
    a: AudioMaterial,
    b: AudioMaterial,
    v: Vec3Fix,
    is_new: bool,
) -> Option<AudioEvent> {
    let mut g = AudioGenerator::new(2, cfg);
    g.set_material(0, a);
    g.set_material(1, b);
    g.process_contact(0, 1, &contact(Vec3Fix::UNIT_Y), v, is_new);
    g.get_events().first().copied()
}

fn impact(a: AudioMaterial, b: AudioMaterial, speed: i64) -> AudioEvent {
    one_event(
        AudioConfig::default(),
        a,
        b,
        Vec3Fix::from_int(0, -speed, 0),
        true,
    )
    .expect("event")
}

/// (hardness, resonance, damping, density) of the presets as documented
/// (decimal values in the source comments / density field).
fn preset_table() -> [(AudioMaterial, f64, f64, f64, f64); 4] {
    [
        (AudioMaterial::METAL, 0.8, 0.8, 0.2, 7800.0),
        (AudioMaterial::WOOD, 0.6, 0.4, 0.6, 600.0),
        (AudioMaterial::STONE, 0.9, 0.2, 0.7, 2500.0),
        (AudioMaterial::RUBBER, 0.2, 0.1, 0.9, 1100.0),
    ]
}

#[test]
fn decay_matches_the_resonance_damping_closed_form_for_every_preset_pair() {
    // decay = 0.1 + 2 * mean(resonance) * (1 - mean(damping))
    for (a, _, ra, da, _) in preset_table() {
        for (b, _, rb, db, _) in preset_table() {
            let e = impact(a, b, 10);
            let want = 0.1 + 2.0 * (0.5 * (ra + rb)) * (1.0 - 0.5 * (da + db));
            assert!(
                (e.decay.to_f64() - want).abs() < 1e-9,
                "{:?}+{:?}: {} vs {want}",
                a.material_type,
                b.material_type,
                e.decay.to_f64()
            );
        }
    }
}

#[test]
fn pitch_matches_the_density_closed_form_with_clamp_for_every_preset_pair_and_speed() {
    // pitch = (1 + speed/20) * clamp(1000 / mean(density), 0.5, 2)
    for (a, _, _, _, dena) in preset_table() {
        for (b, _, _, _, denb) in preset_table() {
            for speed in [1_i64, 5, 20, 60] {
                let e = impact(a, b, speed);
                let factor = (1000.0 / (0.5 * (dena + denb))).clamp(0.5, 2.0);
                let want = (1.0 + speed as f64 / 20.0) * factor;
                assert!(
                    (e.pitch.to_f64() - want).abs() < 1e-8 * want.max(1.0),
                    "{:?}+{:?} v={speed}: {} vs {want}",
                    a.material_type,
                    b.material_type,
                    e.pitch.to_f64()
                );
            }
        }
    }
}

#[test]
fn brightness_and_roughness_stay_in_unit_range_for_all_presets_and_speeds() {
    for (a, ..) in preset_table() {
        for (b, ..) in preset_table() {
            for speed in [1_i64, 3, 10, 19, 20, 21, 100, 5_000] {
                let mut g = AudioGenerator::new(2, AudioConfig::default());
                g.set_material(0, a);
                g.set_material(1, b);
                // oblique: normal speed along y, tangential along x
                g.process_contact(
                    0,
                    1,
                    &contact(Vec3Fix::UNIT_Y),
                    Vec3Fix::from_int(speed, -speed, 0),
                    false,
                );
                for e in g.get_events() {
                    for (name, v) in [
                        ("volume", e.volume),
                        ("brightness", e.brightness),
                        ("roughness", e.roughness),
                    ] {
                        assert!(
                            v >= Fix128::ZERO && v <= Fix128::ONE,
                            "{name} = {} for {:?}+{:?} v={speed}",
                            v.to_f64(),
                            a.material_type,
                            b.material_type
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn volume_pitch_brightness_are_monotone_in_speed_and_volume_brightness_saturate() {
    let a = AudioMaterial::STONE;
    let b = AudioMaterial::METAL;
    let mut prev: Option<AudioEvent> = None;
    for speed in 1..=40_i64 {
        let e = impact(a, b, speed);
        if let Some(p) = prev {
            assert!(e.volume >= p.volume, "volume dropped at {speed}");
            assert!(e.pitch > p.pitch, "pitch not increasing at {speed}");
            assert!(
                e.brightness >= p.brightness,
                "brightness dropped at {speed}"
            );
        }
        if speed >= 20 {
            assert_eq!(e.volume, Fix128::ONE, "volume saturates at max_velocity");
        }
        prev = Some(e);
    }
    let at20 = impact(a, b, 20);
    let at40 = impact(a, b, 40);
    assert_eq!(
        at20.brightness, at40.brightness,
        "brightness saturates at max_velocity"
    );
}

#[test]
fn swapping_the_two_bodies_swaps_materials_but_not_the_symmetric_parameters() {
    let ab = impact(AudioMaterial::METAL, AudioMaterial::RUBBER, 7);
    let ba = impact(AudioMaterial::RUBBER, AudioMaterial::METAL, 7);
    assert_eq!(ab.volume, ba.volume);
    assert_eq!(ab.pitch, ba.pitch);
    assert_eq!(ab.decay, ba.decay);
    assert_eq!(ab.brightness, ba.brightness);
    assert_eq!(ab.roughness, ba.roughness);
    assert_eq!(ab.material_a, ba.material_b);
    assert_eq!(ab.material_b, ba.material_a);
}

#[test]
fn harder_pairs_are_brighter_and_more_resonant_pairs_ring_longer() {
    // hardness order: stone 0.9 > metal 0.8 > wood 0.6 > rubber 0.2
    let b = |m| impact(m, m, 10).brightness;
    assert!(b(AudioMaterial::STONE) > b(AudioMaterial::METAL));
    assert!(b(AudioMaterial::METAL) > b(AudioMaterial::WOOD));
    assert!(b(AudioMaterial::WOOD) > b(AudioMaterial::RUBBER));
    // ring time order for same-material pairs: metal > wood > stone > rubber
    let d = |m| impact(m, m, 10).decay;
    assert!(d(AudioMaterial::METAL) > d(AudioMaterial::WOOD));
    assert!(d(AudioMaterial::WOOD) > d(AudioMaterial::STONE));
    assert!(d(AudioMaterial::STONE) > d(AudioMaterial::RUBBER));
}

#[test]
fn tangential_speed_uses_the_component_perpendicular_to_a_non_axis_aligned_normal() {
    // n = (0.6, 0.8, 0), v = (5, 0, 0): v.n = 3, |v| = 5 -> tangential = 4.
    // roughness = (4/20) * mean(hardness); slide because 4 > 0.5 and not new.
    let n = Vec3Fix::new(
        Fix128::from_ratio(6, 10),
        Fix128::from_ratio(8, 10),
        Fix128::ZERO,
    );
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    g.set_material(0, AudioMaterial::METAL);
    g.set_material(1, AudioMaterial::WOOD);
    g.process_contact(0, 1, &contact(n), Vec3Fix::from_int(5, 0, 0), false);
    let e = g.get_events()[0];
    assert_eq!(e.event_type, AudioEventType::Slide);
    let want = (4.0 / 20.0) * 0.5 * (0.8 + 0.6);
    assert!(
        (e.roughness.to_f64() - want).abs() < 1e-8,
        "{} vs {want}",
        e.roughness.to_f64()
    );
}

#[test]
fn a_purely_normal_persistent_contact_has_no_roughness() {
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    g.process_contact(
        0,
        1,
        &contact(Vec3Fix::UNIT_Y),
        Vec3Fix::from_int(0, -8, 0),
        false,
    );
    let e = g.get_events()[0];
    assert!(e.roughness.to_f64() < 1e-9);
    assert_eq!(e.event_type, AudioEventType::Roll);
}

#[test]
fn event_position_is_the_contact_point_on_body_a() {
    let e = impact(AudioMaterial::WOOD, AudioMaterial::WOOD, 5);
    assert_eq!(e.position, Vec3Fix::from_int(2, 3, 4));
}

#[test]
// AUD-A-S3W2-008
fn zero_max_velocity_saturates_volume_instead_of_silencing() {
    let cfg = AudioConfig {
        max_velocity: Fix128::ZERO,
        ..AudioConfig::default()
    };
    let e = one_event(
        cfg,
        AudioMaterial::WOOD,
        AudioMaterial::WOOD,
        Vec3Fix::from_int(0, -5, 0),
        true,
    );
    assert_eq!(e.map(|e| e.volume), Some(Fix128::ONE));
}

#[test]
fn speed_exactly_at_min_velocity_still_produces_an_event() {
    // min_velocity is documented as the *minimum* impact velocity to generate sound,
    // so a speed equal to it is not below the minimum. 1/4 is dyadic: |v| = 1/4 exactly.
    let quarter = Fix128::from_ratio(1, 4);
    let cfg = AudioConfig {
        min_velocity: quarter,
        min_volume: Fix128::ZERO,
        ..AudioConfig::default()
    };
    let e = one_event(
        cfg,
        AudioMaterial::WOOD,
        AudioMaterial::WOOD,
        Vec3Fix::new(Fix128::ZERO, Fix128::ZERO - quarter, Fix128::ZERO),
        true,
    );
    assert!(e.is_some(), "speed == min_velocity was discarded");
}

#[test]
fn volume_exactly_at_min_volume_is_kept() {
    // speed 5, max_velocity 20 -> volume = sqrt(1/4) = 0.5 exactly; threshold 0.5.
    let cfg = AudioConfig {
        min_velocity: Fix128::ZERO,
        min_volume: Fix128::from_ratio(1, 2),
        ..AudioConfig::default()
    };
    let e = one_event(
        cfg,
        AudioMaterial::WOOD,
        AudioMaterial::WOOD,
        Vec3Fix::from_int(0, -5, 0),
        true,
    )
    .expect("volume == min_volume must not be discarded");
    assert_eq!(e.volume, Fix128::from_ratio(1, 2));
}

#[test]
fn pitch_density_factor_is_clamped_to_two_for_very_light_materials() {
    // Custom material density 100 kg/m^3 on both sides: 1000/100 = 10 -> clamped to 2.
    let light = AudioMaterial {
        density: Fix128::from_int(100),
        ..AudioMaterial::WOOD
    };
    let e = impact(light, light, 10);
    let want = (1.0 + 10.0 / 20.0) * 2.0;
    assert!(
        (e.pitch.to_f64() - want).abs() < 1e-9,
        "{} vs {want}",
        e.pitch.to_f64()
    );
}

#[test]
fn body_index_exactly_one_past_the_material_table_defaults_to_wood_without_panicking() {
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    g.set_material(0, AudioMaterial::METAL);
    g.process_contact(
        0,
        2,
        &contact(Vec3Fix::UNIT_Y),
        Vec3Fix::from_int(0, -5, 0),
        true,
    );
    let e = g.get_events()[0];
    assert_eq!(e.material_a, AudioMaterial::METAL.material_type);
    assert_eq!(e.material_b, AudioMaterial::WOOD.material_type);
}

#[test]
fn set_material_one_past_the_end_grows_the_table_so_the_new_slot_is_used() {
    let mut g = AudioGenerator::new(2, AudioConfig::default());
    g.set_material(2, AudioMaterial::STONE);
    assert_eq!(g.materials.len(), 3);
    g.process_contact(
        0,
        2,
        &contact(Vec3Fix::UNIT_Y),
        Vec3Fix::from_int(0, -5, 0),
        true,
    );
    assert_eq!(
        g.get_events()[0].material_b,
        AudioMaterial::STONE.material_type
    );
}

#[test]
fn default_config_gates_at_a_tenth_of_a_metre_per_second() {
    let c = AudioConfig::default();
    assert_eq!(c.min_velocity, Fix128::from_ratio(1, 10));
    let w = AudioMaterial::WOOD;
    let at = |s: Fix128| {
        one_event(
            c,
            w,
            w,
            Vec3Fix::new(Fix128::ZERO, Fix128::ZERO - s, Fix128::ZERO),
            true,
        )
    };
    assert!(
        at(Fix128::from_ratio(15, 100)).is_some(),
        "0.15 m/s is above the default minimum"
    );
    assert!(
        at(Fix128::from_ratio(5, 100)).is_none(),
        "0.05 m/s is below the default minimum"
    );
}

#[test]
fn default_config_slide_threshold_event_cap_and_volume_floor() {
    let c = AudioConfig::default();
    assert_eq!(c.min_volume, Fix128::from_ratio(1, 100));
    // slide threshold 0.5 m/s: tangential 0.4 rolls, 0.6 slides (persistent contact)
    let w = AudioMaterial::WOOD;
    let tang = |s: i64| {
        one_event(
            c,
            w,
            w,
            Vec3Fix::new(Fix128::from_ratio(s, 10), Fix128::ZERO, Fix128::ZERO),
            false,
        )
        .expect("event")
        .event_type
    };
    assert_eq!(tang(4), AudioEventType::Roll);
    assert_eq!(tang(6), AudioEventType::Slide);
    // per-frame cap is 32 events
    let mut g = AudioGenerator::new(2, c);
    for _ in 0..40 {
        g.process_contact(
            0,
            1,
            &contact(Vec3Fix::UNIT_Y),
            Vec3Fix::from_int(0, -5, 0),
            true,
        );
    }
    assert_eq!(g.get_events().len(), 32);
}
