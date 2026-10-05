//! Oracles for the production entry points of `alice_physics::audio_physics`
//! driven by `examples/audio_physics_events.rs`: `AudioMaterial::{METAL,
//! RUBBER, STONE}` and `AudioGenerator::{set_material, process_contact,
//! get_events}`.
//!
//! # Why these had zero production callers
//!
//! `src/audio_physics.rs`'s own `#[cfg(test)] mod tests` exercised all six,
//! but `scripts/wiring_guard.py` does not count a module's own test module
//! as production; nothing in `src/` or `examples/` called any of them
//! before `examples/audio_physics_events.rs` landed.
//!
//! # Closed forms and where they come from
//!
//! `AudioMaterial::{METAL, RUBBER, STONE}` are hand-copied, field for
//! field, from the literal constants in their `impl AudioMaterial` bodies
//! (`src/audio_physics.rs:64-133`) -- not read back from the constants
//! themselves. `compute_volume` / `compute_pitch` / `compute_decay` /
//! `compute_brightness` / `compute_roughness` are `AudioGenerator`-private
//! (`src/audio_physics.rs:318-403`), so they cannot be called from this
//! external-crate test even if we wanted to: every closed form below
//! re-derives those five formulas by hand using `Fix128`'s own *public*
//! arithmetic (`+`, `*`, `/`, `.half()`, `.sqrt()`, `.abs()`), exactly as
//! the formulas read in source, never by calling `process_contact` (or the
//! private `compute_*` methods) for the expected side.
//!
//! Scenario inputs are chosen so the hand-derived closed forms land on
//! values `Fix128` represents *exactly* wherever possible (perfect squares
//! for `sqrt`, `speed == max_velocity` so `speed_factor` clamps to exactly
//! `Fix128::ONE` and multiplying by it is the identity, equal materials on
//! both bodies so `.half()` of a doubled value is exact). `pitch` and
//! `decay` are the two exceptions: `velocity_pitch_factor` (`1/20`) and the
//! material presets' `0.8`/`0.2`/etc. fields are *not* exact binary
//! fractions, so their formulas carry truncation noise on the order of
//! `2^-64` (~5e-20) -- below `Fix128`'s own precision floor, not a
//! correctness question. Those two use a tolerance comparison, same
//! convention as this file's own pre-existing
//! `get_events_returns_this_frames_events_in_order_and_respects_cap` unit
//! test (`vol_err < Fix128::from_ratio(1, 1_000_000)`).
//!
//! # Degenerate input coverage
//!
//! * **Zero relative velocity**: `process_contact` computes `speed =
//!   relative_velocity.length()`; `Vec3Fix::ZERO.length() == Fix128::ZERO`
//!   exactly, below `AudioConfig::default().min_velocity` (`0.1`), so the
//!   contact is silently dropped -- not a panic, not an event.
//! * **No material set on a body**: `AudioGenerator::new` fills
//!   `materials` with `AudioMaterial::WOOD` for every body, and
//!   `get_material` (private, `src/audio_physics.rs:310-316`) falls back
//!   to `AudioMaterial::WOOD` for any index `>= materials.len()` too -- the
//!   default is `Wood`, not a zeroed/uninitialized material.
//! * **`get_events` on an empty / just-cleared queue**: empty before the
//!   first `begin_frame`, empty immediately after `begin_frame` clears it.
//! * **Extreme `Fix128` magnitude**: `Fix128` addition/multiplication wrap
//!   modulo 2^128, so `relative_velocity = (i64::MAX, 0, 0)` does not panic and
//!   does not merely need to "not panic" (a vacuous oracle for wrapping
//!   arithmetic) -- it has an exact, hand-derivable, surprising closed
//!   form: `(i64::MAX)^2 mod 2^64 == 1` (shown below), so
//!   `length_squared() == Fix128::ONE` and `length() == Fix128::ONE`
//!   *exactly*, despite the input being ~9.2e18. The contact is processed
//!   as if `speed == 1.0`.
//! * **`min_volume` gate is unreachable under `AudioConfig::default()`**:
//!   noted, not fixed (see final report) -- `speed >= min_velocity (0.1)`
//!   always implies `volume = sqrt(speed/max_velocity) >= sqrt(0.1/20) ≈
//!   0.0707`, which already exceeds `min_volume (0.01)`. A dedicated test
//!   below exercises the gate with a non-default config to prove it has
//!   teeth at all.

use alice_physics::audio_physics::{
    AudioConfig, AudioEventType, AudioGenerator, AudioMaterial, MaterialType,
};
use alice_physics::collider::Contact;
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fi(n: i64) -> Fix128 {
    Fix128::from_int(n)
}

fn r(num: i64, denom: i64) -> Fix128 {
    Fix128::from_ratio(num, denom)
}

fn contact_at(x: i64) -> Contact {
    Contact {
        depth: r(1, 10),
        normal: Vec3Fix::UNIT_Y,
        point_a: Vec3Fix::from_int(x, 0, 0),
        point_b: Vec3Fix::from_int(x, -1, 0),
    }
}

/// `(a + b).half()` -- the exact formula `compute_pitch` / `compute_decay` /
/// `compute_brightness` / `compute_roughness` use for every "avg_*" term
/// (`src/audio_physics.rs` e.g. line 339, 358-359, 376, 394).
fn avg(a: Fix128, b: Fix128) -> Fix128 {
    (a + b).half()
}

/// Same tolerance convention as this crate's other oracle files
/// (`tests/analytic_material_wiring.rs`'s `close`, and this file's own
/// pre-existing unit test's `vol_err`): independent rounding of a
/// hand-written literal against the formula's own truncating division/
/// multiply, never bit-identity for values that are not exact binary
/// fractions to begin with.
fn close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() < r(1, 1_000_000)
}

// ============================================================================
// Material presets: METAL, RUBBER, STONE
// ============================================================================

#[test]
fn material_presets_match_hardcoded_source_constants() {
    // Hand-copied from src/audio_physics.rs:64-133, field for field.
    let metal = AudioMaterial::METAL;
    assert_eq!(metal.material_type, MaterialType::Metal);
    assert_eq!(metal.density, fi(7800));
    assert_eq!(metal.hardness, Fix128::from_raw(0, 0xCCCC_CCCC_CCCC_CCCC));
    assert_eq!(metal.resonance, Fix128::from_raw(0, 0xCCCC_CCCC_CCCC_CCCC));
    assert_eq!(metal.damping, Fix128::from_raw(0, 0x3333_3333_3333_3333));

    let rubber = AudioMaterial::RUBBER;
    assert_eq!(rubber.material_type, MaterialType::Rubber);
    assert_eq!(rubber.density, fi(1100));
    assert_eq!(rubber.hardness, Fix128::from_raw(0, 0x3333_3333_3333_3333));
    assert_eq!(rubber.resonance, Fix128::from_raw(0, 0x1999_9999_9999_9999));
    assert_eq!(rubber.damping, Fix128::from_raw(0, 0xE666_6666_6666_6666));

    let stone = AudioMaterial::STONE;
    assert_eq!(stone.material_type, MaterialType::Stone);
    assert_eq!(stone.density, fi(2500));
    assert_eq!(stone.hardness, Fix128::from_raw(0, 0xE666_6666_6666_6666));
    assert_eq!(stone.resonance, Fix128::from_raw(0, 0x3333_3333_3333_3333));
    assert_eq!(stone.damping, Fix128::from_raw(0, 0xB333_3333_3333_3333));

    // Metal is harder and more resonant than rubber (pre-existing property
    // from src/audio_physics.rs's own unit test, re-asserted here since
    // this file is independent of that module's #[cfg(test)]).
    assert!(metal.hardness > rubber.hardness);
    assert!(metal.resonance > rubber.resonance);

    // The three presets' density/hardness/resonance/damping are pairwise
    // distinct on every field -- a mutation that aliased any two preset
    // bodies (e.g. STONE accidentally returning METAL's resonance) is
    // caught here directly, not just via the inequalities above.
    let fields = [
        (
            "METAL",
            metal.density,
            metal.hardness,
            metal.resonance,
            metal.damping,
        ),
        (
            "RUBBER",
            rubber.density,
            rubber.hardness,
            rubber.resonance,
            rubber.damping,
        ),
        (
            "STONE",
            stone.density,
            stone.hardness,
            stone.resonance,
            stone.damping,
        ),
    ];
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            let (name_i, di, hi_, ri, pi) = fields[i];
            let (name_j, dj, hj, rj, pj) = fields[j];
            assert_ne!(di, dj, "{name_i} vs {name_j} density");
            assert_ne!(hi_, hj, "{name_i} vs {name_j} hardness");
            assert_ne!(ri, rj, "{name_i} vs {name_j} resonance");
            assert_ne!(pi, pj, "{name_i} vs {name_j} damping");
        }
    }
}

// ============================================================================
// set_material
// ============================================================================

#[test]
fn set_material_grows_table_and_fills_gap_with_wood_default() {
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    assert_eq!(gen.materials.len(), 2);
    assert_eq!(gen.materials[0], AudioMaterial::WOOD);
    assert_eq!(gen.materials[1], AudioMaterial::WOOD);

    // Growing past the current length (src/audio_physics.rs:237-238:
    // `self.materials.resize(body_idx + 1, AudioMaterial::WOOD)`) must fill
    // every newly-created slot -- including the ones strictly between the
    // old length and body_idx -- with WOOD, not a zeroed/default-derived
    // material.
    gen.set_material(5, AudioMaterial::METAL);
    assert_eq!(gen.materials.len(), 6, "resize target is body_idx + 1");
    assert_eq!(gen.materials[0], AudioMaterial::WOOD, "untouched slot 0");
    assert_eq!(gen.materials[1], AudioMaterial::WOOD, "untouched slot 1");
    assert_eq!(gen.materials[2], AudioMaterial::WOOD, "gap slot 2");
    assert_eq!(gen.materials[3], AudioMaterial::WOOD, "gap slot 3");
    assert_eq!(gen.materials[4], AudioMaterial::WOOD, "gap slot 4");
    assert_eq!(
        gen.materials[5],
        AudioMaterial::METAL,
        "explicitly set slot 5"
    );

    // Overwriting an already-set slot in place (no resize, no side effect
    // on neighbors).
    gen.set_material(5, AudioMaterial::STONE);
    assert_eq!(gen.materials.len(), 6, "overwrite must not resize");
    assert_eq!(gen.materials[5], AudioMaterial::STONE);
    assert_eq!(gen.materials[4], AudioMaterial::WOOD, "neighbor untouched");

    // Overwriting an existing low-index slot (no resize path at all).
    gen.set_material(0, AudioMaterial::RUBBER);
    assert_eq!(gen.materials.len(), 6);
    assert_eq!(gen.materials[0], AudioMaterial::RUBBER);
}

#[test]
fn process_contact_defaults_unset_body_to_wood() {
    // Body 1 is never set_material'd, but is still within materials.len()
    // (filled Wood by AudioGenerator::new's own initial vec![WOOD; n]) --
    // this alone does not exercise get_material's own fallback branch
    // (src/audio_physics.rs:310-316), only AudioGenerator::new's fill.
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.begin_frame();
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);

    let events = gen.get_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].material_a, MaterialType::Metal);
    assert_eq!(
        events[0].material_b,
        MaterialType::Wood,
        "unset body (within materials.len()) defaults to Wood via AudioGenerator::new's fill"
    );
}

#[test]
fn process_contact_defaults_out_of_range_body_index_to_wood() {
    // Body index 7 is strictly >= materials.len() (2) and was never grown
    // via set_material -- this is the only way to exercise get_material's
    // own `else { AudioMaterial::WOOD }` fallback branch
    // (src/audio_physics.rs:310-316) directly, as opposed to
    // AudioGenerator::new's initial fill (exercised by the sibling test
    // above). A mutation that changed only this fallback's constant (not
    // `new`'s fill, not `set_material`'s resize fill) would be invisible
    // to every other test in this file -- it is only observable through
    // an index that was never part of the vec at all.
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.begin_frame();
    gen.process_contact(0, 7, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);

    let events = gen.get_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].material_a, MaterialType::Metal);
    assert_eq!(
        events[0].material_b,
        MaterialType::Wood,
        "body_idx >= materials.len() must fall back to Wood via get_material itself"
    );
}

// ============================================================================
// process_contact: event-type classification (Impact / Slide / Roll)
// ============================================================================

#[test]
fn process_contact_classifies_impact_slide_roll() {
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.set_material(1, AudioMaterial::STONE);
    gen.begin_frame();

    // is_new_contact=true -> Impact, regardless of tangential component.
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);
    // is_new_contact=false, tangential_speed=20 > slide_velocity_threshold
    // (0.5 default) -> Slide.
    gen.process_contact(0, 1, &contact_at(1), Vec3Fix::from_int(20, 0, 0), false);
    // is_new_contact=false, tangential_speed=0 (<= 0.5 threshold) -> Roll.
    gen.process_contact(0, 1, &contact_at(2), Vec3Fix::from_int(0, -1, 0), false);

    let events = gen.get_events();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].event_type, AudioEventType::Impact);
    assert_eq!(events[1].event_type, AudioEventType::Slide);
    assert_eq!(events[2].event_type, AudioEventType::Roll);
}

#[test]
fn process_contact_slide_roll_boundary_is_strictly_greater_than_threshold() {
    // tangential_speed == slide_velocity_threshold exactly (not >) must
    // classify as Roll, not Slide -- the condition in
    // src/audio_physics.rs:280 is `tangential_speed >
    // self.config.slide_velocity_threshold`, strict.
    let config = AudioConfig {
        slide_velocity_threshold: fi(5),
        ..AudioConfig::default()
    };
    let mut gen = AudioGenerator::new(2, config);
    gen.begin_frame();
    // Pure-tangential relative velocity of exactly 5 (threshold), normal=Y.
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(5, 0, 0), false);
    let events = gen.get_events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].event_type,
        AudioEventType::Roll,
        "tangential_speed == threshold must not count as Slide"
    );
}

// ============================================================================
// process_contact: volume / pitch / decay / brightness / roughness
// ============================================================================

#[test]
fn process_contact_volume_matches_sqrt_closed_form() {
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.begin_frame();

    // speed=20=max_velocity -> normalized=1 exactly -> clamped=1 exactly ->
    // volume=sqrt(1)=1 exactly (perfect square, no clamping engaged).
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);
    // speed=1 -> normalized=1/20 -> volume=sqrt(1/20).
    gen.process_contact(0, 1, &contact_at(1), Vec3Fix::from_int(0, -1, 0), true);
    // speed=40 (> max_velocity=20) -> normalized clamps to 1 exactly ->
    // volume=sqrt(1)=1 exactly, same as the first case (clamping, not an
    // unbounded/overflowing volume).
    gen.process_contact(0, 1, &contact_at(2), Vec3Fix::from_int(0, -40, 0), true);

    let events = gen.get_events();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].volume, Fix128::ONE, "speed==max_velocity");
    let expect_sqrt_1_20 = (fi(1) / fi(20)).sqrt();
    assert_eq!(events[1].volume, expect_sqrt_1_20, "speed=1 -> sqrt(1/20)");
    assert_eq!(
        events[2].volume,
        Fix128::ONE,
        "speed > max_velocity clamps to the same volume as speed==max_velocity, not a larger one"
    );
}

#[test]
fn process_contact_brightness_and_roughness_match_hand_derived_avg_hardness_identity() {
    // At speed == max_velocity, compute_brightness's `speed_factor.half()`
    // term is exactly 0.5 (Fix128::ONE.half() == from_ratio(1,2) exactly),
    // so `0.5 + 0.5 == Fix128::ONE` exactly, and multiplying avg_hardness
    // by Fix128::ONE is the identity for Fix128's mul -- brightness ends up
    // bit-identical to avg_hardness, not merely close to it. Same identity
    // makes a slide contact's roughness (tangential_speed == max_velocity
    // too) bit-identical to avg_hardness.
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.set_material(1, AudioMaterial::STONE);
    gen.begin_frame();

    // Impact: pure-normal relative velocity at max_velocity.
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);
    // Slide: pure-tangential relative velocity at max_velocity.
    gen.process_contact(0, 1, &contact_at(1), Vec3Fix::from_int(20, 0, 0), false);

    let events = gen.get_events();
    assert_eq!(events.len(), 2);
    let avg_hardness = avg(AudioMaterial::METAL.hardness, AudioMaterial::STONE.hardness);

    assert_eq!(
        events[0].brightness, avg_hardness,
        "brightness at speed==max_velocity == avg_hardness exactly"
    );
    assert_eq!(
        events[0].roughness,
        Fix128::ZERO,
        "pure-normal relative velocity -> tangential_speed=0 exactly -> roughness=0 exactly"
    );
    assert_eq!(
        events[1].roughness, avg_hardness,
        "roughness at tangential_speed==max_velocity == avg_hardness exactly"
    );
}

#[test]
fn process_contact_pitch_and_decay_match_hand_derived_formula_within_tolerance() {
    // pitch = (1 + speed*velocity_pitch_factor) * density_factor, where
    // density_factor = clamp(1000/avg_density, 0.5, 2) (src/audio_physics.rs
    // :335-353). decay = 0.1 + (avg_resonance*2)*(1-avg_damping)
    // (:357-367). Metal/Stone: avg_density=5150 -> 1000/5150≈0.194,
    // clamped to the 0.5 floor exactly (the clamp constant, not the
    // unclamped quotient, is what survives); avg_resonance≈0.5,
    // avg_damping≈0.45.
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.set_material(0, AudioMaterial::METAL);
    gen.set_material(1, AudioMaterial::STONE);
    gen.begin_frame();
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);

    let events = gen.get_events();
    assert_eq!(events.len(), 1);

    let speed = fi(20);
    let cfg = AudioConfig::default();
    let base = Fix128::ONE + speed * cfg.velocity_pitch_factor;
    let avg_density = avg(AudioMaterial::METAL.density, AudioMaterial::STONE.density);
    let density_factor_raw = fi(1000) / avg_density;
    let density_factor = if density_factor_raw > fi(2) {
        fi(2)
    } else if density_factor_raw < r(1, 2) {
        r(1, 2)
    } else {
        density_factor_raw
    };
    assert_eq!(density_factor, r(1, 2), "1000/5150 clamps to the 0.5 floor");
    let expect_pitch = base * density_factor;
    assert!(
        close(events[0].pitch, expect_pitch),
        "pitch {:?} vs hand-derived {:?}",
        events[0].pitch,
        expect_pitch
    );

    let avg_resonance = avg(
        AudioMaterial::METAL.resonance,
        AudioMaterial::STONE.resonance,
    );
    let avg_damping = avg(AudioMaterial::METAL.damping, AudioMaterial::STONE.damping);
    let expect_decay = r(1, 10) + (avg_resonance * fi(2)) * (Fix128::ONE - avg_damping);
    assert!(
        close(events[0].decay, expect_decay),
        "decay {:?} vs hand-derived {:?}",
        events[0].decay,
        expect_decay
    );
}

#[test]
fn process_contact_pitch_bypasses_density_adjustment_when_avg_density_is_zero() {
    // compute_pitch's early return (src/audio_physics.rs:340-342): if
    // avg_density.is_zero(), pitch == base (the density_factor multiply is
    // skipped entirely, not treated as a divide-by-zero -> 0 or inf).
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    let zero_density = AudioMaterial {
        material_type: MaterialType::Custom(0),
        density: Fix128::ZERO,
        hardness: Fix128::ZERO,
        resonance: Fix128::ZERO,
        damping: Fix128::ZERO,
    };
    gen.set_material(0, zero_density);
    gen.set_material(1, zero_density);
    gen.begin_frame();
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);

    let events = gen.get_events();
    assert_eq!(events.len(), 1);
    let speed = fi(20);
    let expect_base = Fix128::ONE + speed * AudioConfig::default().velocity_pitch_factor;
    assert_eq!(
        events[0].pitch, expect_base,
        "zero avg_density bypasses the density_factor multiply entirely"
    );
}

// ============================================================================
// Degenerate / extreme inputs
// ============================================================================

#[test]
fn process_contact_suppresses_zero_relative_velocity() {
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.begin_frame();
    let result = catch_unwind(AssertUnwindSafe(|| {
        gen.process_contact(0, 1, &contact_at(0), Vec3Fix::ZERO, true);
    }));
    assert!(result.is_ok(), "zero relative velocity must not panic");
    assert!(
        gen.get_events().is_empty(),
        "zero relative velocity (speed=0 < min_velocity=0.1) must not emit an event"
    );
}

#[test]
fn process_contact_min_velocity_gate_has_teeth_independent_of_min_volume() {
    // speed=0 alone does not distinguish the min_velocity gate
    // (src/audio_physics.rs:262-264) from the separate min_volume gate
    // (:293-295): at speed=0, volume=sqrt(0/20)=0 too, so either gate
    // alone suppresses the event -- a mutation that deleted *only* the
    // min_velocity check would still pass the zero-velocity test above.
    //
    // speed=1/20=0.05 is the needed middle ground: it is below
    // min_velocity (0.1), but volume=sqrt(0.05/20)=sqrt(1/400)=1/20=0.05
    // is *above* min_volume (0.01) -- so with the min_velocity gate
    // removed, this exact contact would emit. This is the only test in
    // this file where the min_velocity gate is independently load-bearing.
    let speed = r(1, 20);
    let volume_if_gate_removed = (speed / fi(20)).sqrt();
    assert!(
        volume_if_gate_removed > r(1, 100),
        "sanity: this speed's volume ({volume_if_gate_removed:?}) must clear min_volume \
         on its own, or this test does not isolate the min_velocity gate"
    );
    assert!(
        speed < AudioConfig::default().min_velocity,
        "sanity: this speed must be below min_velocity, or this test is vacuous"
    );

    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.begin_frame();
    gen.process_contact(
        0,
        1,
        &contact_at(0),
        Vec3Fix::new(Fix128::ZERO, -speed, Fix128::ZERO),
        true,
    );
    assert!(
        gen.get_events().is_empty(),
        "speed=0.05 is below min_velocity (0.1) even though its volume (0.05) clears \
         min_volume (0.01) on its own -- the min_velocity gate, not min_volume, must \
         be what suppresses this contact"
    );
}

#[test]
fn process_contact_min_volume_gate_has_teeth_under_a_lowered_min_velocity() {
    // Under AudioConfig::default(), speed >= min_velocity (0.1) always
    // implies volume = sqrt(speed/max_velocity) >= sqrt(0.1/20) ≈ 0.0707,
    // which already clears min_volume (0.01) -- so the min_volume gate at
    // src/audio_physics.rs:293-295 is unreachable under the default
    // config. Lowering min_velocity below min_volume's effective floor
    // demonstrates the gate is live code with its own closed form, not
    // dead weight: speed=0.01 (just above a min_velocity of 0) gives
    // volume=sqrt(0.01/20)=sqrt(0.0005)≈0.02236, which *does* clear
    // min_volume=0.01 here (chosen to cross it the other way below).
    let cfg_passes = AudioConfig {
        min_velocity: Fix128::ZERO,
        min_volume: r(1, 100),
        ..AudioConfig::default()
    };
    let mut gen = AudioGenerator::new(2, cfg_passes);
    gen.begin_frame();
    // speed=1 -> volume=sqrt(1/20)≈0.2236 > min_volume=0.01 -> emits.
    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -1, 0), true);
    assert_eq!(
        gen.get_events().len(),
        1,
        "volume above min_volume must emit"
    );

    // Now push min_volume above what any sub-max_velocity speed can reach
    // (min_volume=0.9 > sqrt(speed/20) for any speed < 0.81*20=16.2), while
    // min_velocity stays at 0 so the speed gate never engages first.
    let cfg_blocks = AudioConfig {
        min_velocity: Fix128::ZERO,
        min_volume: r(9, 10),
        ..AudioConfig::default()
    };
    let mut gen2 = AudioGenerator::new(2, cfg_blocks);
    gen2.begin_frame();
    gen2.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -1, 0), true);
    assert!(
        gen2.get_events().is_empty(),
        "volume below min_volume must be suppressed even though speed cleared min_velocity"
    );
}

#[test]
fn process_contact_respects_max_events_per_frame_cap() {
    let config = AudioConfig {
        max_events_per_frame: 2,
        ..AudioConfig::default()
    };
    let mut gen = AudioGenerator::new(2, config);
    gen.begin_frame();
    for i in 0..5i64 {
        gen.process_contact(0, 1, &contact_at(i), Vec3Fix::from_int(0, -20, 0), true);
    }
    assert_eq!(
        gen.get_events().len(),
        2,
        "the 3rd..5th contacts must be dropped once the per-frame cap is reached"
    );
    // The cap check is the *first* thing process_contact does
    // (src/audio_physics.rs:257-259), so the surviving events are the
    // first two fed in, in order -- not an arbitrary subset.
    assert_eq!(gen.get_events()[0].position, Vec3Fix::from_int(0, 0, 0));
    assert_eq!(gen.get_events()[1].position, Vec3Fix::from_int(1, 0, 0));
}

#[test]
fn process_contact_extreme_magnitude_wraps_deterministically_not_physically() {
    // Fix128 add/mul wrap modulo 2^128. relative_velocity=(i64::MAX, 0, 0) exercises this
    // directly: length_squared() = x*x (y=z=0 contribute nothing), and
    // (i64::MAX)^2 mod 2^64 == 1 --
    //
    //   i64::MAX == 2^63 - 1
    //   (2^63-1)^2 == 2^126 - 2^64 + 1
    //   2^126 mod 2^64 == 0 (126 > 64, multiple of 2^64)
    //   => (2^126 - 2^64 + 1) mod 2^64 == 1
    //
    // Fix128::mul's `hi` field is exactly this low-64-bits-of-the-full-
    // product value (src/math.rs:421-459's `(hh as i64)` truncating cast),
    // so x*x == Fix128 { hi: 1, lo: 0 } == Fix128::ONE *exactly* --
    // length_squared() and length() (its sqrt, also exact: 1 is a perfect
    // square) both come out to 1.0, not a value anywhere near the
    // ~9.2e18-scale input. This is not "doesn't panic" (which wrapping
    // arithmetic satisfies unconditionally and is therefore a vacuous
    // oracle on its own) -- a
    // mutation that replaced wrapping with saturating or checked
    // arithmetic would change this exact value (to either Fix128::MAX-ish
    // saturation or a panic), giving this assertion teeth.
    let extreme = Vec3Fix::from_int(i64::MAX, 0, 0);
    assert_eq!(
        extreme.length_squared(),
        Fix128::ONE,
        "(i64::MAX)^2 mod 2^64 == 1 under Fix128's wrapping multiply"
    );
    assert_eq!(extreme.length(), Fix128::ONE);

    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    gen.begin_frame();
    let result = catch_unwind(AssertUnwindSafe(|| {
        gen.process_contact(0, 1, &contact_at(0), extreme, true);
    }));
    assert!(
        result.is_ok(),
        "extreme-magnitude relative_velocity must not panic"
    );

    let events = gen.get_events();
    assert_eq!(
        events.len(),
        1,
        "wrapped speed==1.0 clears min_velocity (0.1) and min_volume, so it emits"
    );
    assert_eq!(
        events[0].volume,
        (fi(1) / fi(20)).sqrt(),
        "volume uses the wrapped speed==1.0, i.e. sqrt(1/20), not a physically-huge speed"
    );

    // Determinism: feeding the exact same extreme input into a fresh
    // generator twice must produce bit-identical events both times.
    let mut gen2 = AudioGenerator::new(2, AudioConfig::default());
    gen2.begin_frame();
    gen2.process_contact(0, 1, &contact_at(0), extreme, true);
    assert_eq!(gen.get_events()[0].volume, gen2.get_events()[0].volume);
}

// ============================================================================
// get_events
// ============================================================================

#[test]
fn get_events_is_empty_before_first_begin_frame_and_after_each_clear() {
    let mut gen = AudioGenerator::new(2, AudioConfig::default());
    // Before any begin_frame() call at all.
    assert!(gen.get_events().is_empty(), "fresh generator has no events");

    gen.begin_frame();
    assert!(
        gen.get_events().is_empty(),
        "begin_frame on an already-empty queue"
    );

    gen.process_contact(0, 1, &contact_at(0), Vec3Fix::from_int(0, -20, 0), true);
    assert_eq!(gen.get_events().len(), 1);

    // get_events is a read-only view: calling it repeatedly without a new
    // begin_frame/process_contact must not mutate anything.
    assert_eq!(gen.get_events().len(), 1);
    assert_eq!(gen.get_events(), gen.events.as_slice());

    gen.begin_frame();
    assert!(
        gen.get_events().is_empty(),
        "begin_frame must clear a non-empty queue"
    );

    // A second clear-then-fill cycle, to rule out a one-shot clear.
    gen.process_contact(0, 1, &contact_at(1), Vec3Fix::from_int(0, -20, 0), true);
    assert_eq!(gen.get_events().len(), 1);
    gen.begin_frame();
    assert!(gen.get_events().is_empty());
}
