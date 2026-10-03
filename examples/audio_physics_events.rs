//! Collision-driven audio event generation: material presets feed a
//! per-body material table, contacts drive `AudioGenerator::process_contact`,
//! and the generated events are read back via `get_events` for a game audio
//! system to consume.
//!
//! Wiring: `AudioMaterial::{METAL, RUBBER, STONE}`, `AudioGenerator::{
//! set_material, process_contact, get_events}` had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/audio_physics.rs::*`). This
//! is their production entry point: it builds a 3-body scene, tags each
//! body with a material preset via `set_material`, feeds three distinct
//! contact shapes (a head-on impact, a sliding contact, a rolling contact)
//! to `process_contact`, and reads the frame's events back out through
//! `get_events`.
//!
//! Every printed value is checked against a value hand-derived from the
//! formulas in `src/audio_physics.rs` (`compute_volume`/`compute_pitch`/
//! `compute_decay`/`compute_brightness`/`compute_roughness`, lines
//! 318-403) re-typed here with `Fix128`'s own public arithmetic -- never by
//! calling those (private, in any case) methods or `process_contact` itself
//! for the expected side. `tests/analytic_audio_physics_wiring.rs` holds
//! the independent oracle test suite for the same six items plus the
//! degenerate-input coverage.
//!
//! ```bash
//! cargo run --example audio_physics_events --features std
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
fn main() {
    eprintln!(
        "this example needs the `std` feature: cargo run --example audio_physics_events --features std"
    );
}

#[cfg(feature = "std")]
fn main() {
    use alice_physics::audio_physics::{
        AudioConfig, AudioEventType, AudioGenerator, AudioMaterial, MaterialType,
    };
    use alice_physics::collider::Contact;
    use alice_physics::math::{Fix128, Vec3Fix};

    let fi = Fix128::from_int;
    let r = Fix128::from_ratio;

    // --- Material presets: hand-copied from src/audio_physics.rs:64-133,
    // not read back from the constants themselves (that would be circular).
    let metal_expect = (fi(7800), Fix128::from_raw(0, 0xCCCC_CCCC_CCCC_CCCC));
    let rubber_expect = (fi(1100), Fix128::from_raw(0, 0x3333_3333_3333_3333));
    let stone_expect = (fi(2500), Fix128::from_raw(0, 0xE666_6666_6666_6666));

    println!("[audio_physics] -- material presets --");
    for (name, m, expect_density, expect_hardness) in [
        (
            "METAL",
            AudioMaterial::METAL,
            metal_expect.0,
            metal_expect.1,
        ),
        (
            "RUBBER",
            AudioMaterial::RUBBER,
            rubber_expect.0,
            rubber_expect.1,
        ),
        (
            "STONE",
            AudioMaterial::STONE,
            stone_expect.0,
            stone_expect.1,
        ),
    ] {
        println!(
            "[audio_physics]   {name:6} density={} hardness={:.4} resonance={:.4} damping={:.4}",
            m.density.to_f64(),
            m.hardness.to_f64(),
            m.resonance.to_f64(),
            m.damping.to_f64()
        );
        assert_eq!(m.density, expect_density, "{name} density");
        assert_eq!(m.hardness, expect_hardness, "{name} hardness");
    }
    assert_eq!(AudioMaterial::METAL.material_type, MaterialType::Metal);
    assert_eq!(AudioMaterial::RUBBER.material_type, MaterialType::Rubber);
    assert_eq!(AudioMaterial::STONE.material_type, MaterialType::Stone);

    // --- set_material: tag three bodies, then grow the table past its
    // current length (body 5 on a 3-body generator) to exercise the
    // resize-with-WOOD-default-fill path documented at
    // src/audio_physics.rs:236-241.
    let mut gen = AudioGenerator::new(3, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.set_material(1, AudioMaterial::RUBBER);
    gen.set_material(2, AudioMaterial::STONE);
    gen.set_material(5, AudioMaterial::METAL);
    assert_eq!(
        gen.materials.len(),
        6,
        "set_material(5, ..) must grow to len 6"
    );
    assert_eq!(gen.materials[3], AudioMaterial::WOOD, "gap-filled slot 3");
    assert_eq!(gen.materials[4], AudioMaterial::WOOD, "gap-filled slot 4");
    assert_eq!(
        gen.materials[5],
        AudioMaterial::METAL,
        "explicitly set slot 5"
    );
    println!(
        "[audio_physics] set_material: body0=Metal body1=Rubber body2=Stone, \
         set_material(5, Metal) grew materials.len() {} -> 6 (gap slots 3,4 = Wood default)",
        3
    );

    // --- process_contact: three contacts in one frame.
    gen.begin_frame();
    assert!(gen.get_events().is_empty(), "begin_frame must clear events");

    let contact_at = |x: i64| Contact {
        depth: r(1, 10),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::from_int(x, 0, 0),
        point_b: Vec3Fix::from_int(x, -1, 0),
    };

    // Contact 1: body0 (Metal) vs body2 (Stone), head-on at max_velocity
    // (20 = AudioConfig::default().max_velocity), is_new_contact=true -> Impact.
    // speed = |(0,-20,0)| = 20 exactly (sqrt(400), a perfect square).
    gen.process_contact(0, 2, &contact_at(1), Vec3Fix::from_int(0, -20, 0), true);

    // Contact 2: body1 (Rubber) vs body2 (Stone), sliding at max_velocity,
    // is_new_contact=false, tangential_speed=20 > slide_velocity_threshold
    // (0.5) -> Slide.
    gen.process_contact(1, 2, &contact_at(2), Vec3Fix::from_int(20, 0, 0), false);

    // Contact 3: body0 (Metal) vs body5 (Metal, via the resize above), a
    // slow non-sliding contact (speed=1, tangential=0) -> Roll.
    gen.process_contact(0, 5, &contact_at(3), Vec3Fix::from_int(0, -1, 0), false);

    // Contact 4: zero relative velocity -- below min_velocity (0.1), must be
    // silently dropped (no event, no panic).
    gen.process_contact(0, 2, &contact_at(4), Vec3Fix::ZERO, true);

    let events = gen.get_events();
    assert_eq!(events.len(), 3, "contact 4 (zero velocity) must not emit");
    println!(
        "[audio_physics] process_contact: 4 contacts fed, {} events emitted (contact 4 below min_velocity suppressed)",
        events.len()
    );

    assert_eq!(events[0].event_type, AudioEventType::Impact);
    assert_eq!(events[1].event_type, AudioEventType::Slide);
    assert_eq!(events[2].event_type, AudioEventType::Roll);
    assert_eq!(
        (events[0].material_a, events[0].material_b),
        (MaterialType::Metal, MaterialType::Stone)
    );
    assert_eq!(
        (events[1].material_a, events[1].material_b),
        (MaterialType::Rubber, MaterialType::Stone)
    );
    assert_eq!(
        (events[2].material_a, events[2].material_b),
        (MaterialType::Metal, MaterialType::Metal)
    );

    // Hand-derived closed forms (re-typed from src/audio_physics.rs
    // 318-403's formulas, not from calling those private methods).
    let max_v = fi(20);
    let avg = |a: Fix128, b: Fix128| (a + b).half();

    // Event 0: Metal/Stone, speed=20=max_velocity -> normalized=1 exactly,
    // volume = sqrt(1) = 1 exactly.
    assert_eq!(events[0].volume, Fix128::ONE, "impact volume at max speed");
    // brightness = avg_hardness * (0.5 + 0.5*speed_factor); speed_factor=1
    // exactly -> (0.5+0.5)=1 exactly -> brightness == avg_hardness exactly
    // (multiplying by Fix128::ONE is the identity for Fix128::mul).
    let avg_hardness_ms = avg(AudioMaterial::METAL.hardness, AudioMaterial::STONE.hardness);
    assert_eq!(
        events[0].brightness, avg_hardness_ms,
        "impact brightness == avg hardness at max speed"
    );
    // roughness: tangential_speed=0 exactly (pure-normal relative velocity)
    // -> roughness = 0 * avg_hardness = 0 exactly.
    assert_eq!(
        events[0].roughness,
        Fix128::ZERO,
        "impact roughness (no tangential component)"
    );

    // Event 1: Rubber/Stone, slide, tangential_speed=speed=20=max_velocity
    // -> speed_factor=1 exactly -> roughness == avg_hardness exactly too.
    let avg_hardness_rs = avg(
        AudioMaterial::RUBBER.hardness,
        AudioMaterial::STONE.hardness,
    );
    assert_eq!(
        events[1].roughness, avg_hardness_rs,
        "slide roughness == avg hardness at max speed"
    );
    assert_eq!(events[1].volume, Fix128::ONE, "slide volume at max speed");

    // Event 2: Metal/Metal, speed=1, tangential=0 -> volume = sqrt(1/20).
    let expect_volume_roll = (fi(1) / max_v).sqrt();
    assert_eq!(
        events[2].volume, expect_volume_roll,
        "roll volume = sqrt(1/20)"
    );
    assert_eq!(
        events[2].roughness,
        Fix128::ZERO,
        "roll roughness (no tangential component)"
    );

    println!(
        "[audio_physics]   event0 Impact  volume={:.4} brightness={:.4} roughness={:.4}",
        events[0].volume.to_f64(),
        events[0].brightness.to_f64(),
        events[0].roughness.to_f64()
    );
    println!(
        "[audio_physics]   event1 Slide   volume={:.4} roughness={:.4} (== avg hardness {:.4})",
        events[1].volume.to_f64(),
        events[1].roughness.to_f64(),
        avg_hardness_rs.to_f64()
    );
    println!(
        "[audio_physics]   event2 Roll    volume={:.4} (== sqrt(1/20)={:.4})",
        events[2].volume.to_f64(),
        expect_volume_roll.to_f64()
    );

    // --- get_events: cleared on the next begin_frame.
    gen.begin_frame();
    assert!(
        gen.get_events().is_empty(),
        "get_events must be empty right after begin_frame clears the queue"
    );
    println!("[audio_physics] get_events: empty immediately after begin_frame clears the queue");

    println!("[audio_physics] done");
}
