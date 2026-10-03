//! Oracles for `character_state` (state table) and its bridge
//! `SdfCharacter::locomotion_context` / `ground_contact` / `is_grounded`.
//!
//! Expected values are hand-derived: the transition table is written out
//! row by row from the module doc's priority list, and the ground probe
//! uses planes whose distance and normal are closed forms.

#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::character_state::{transition, CharacterState, CharacterStateContext};
use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::ClosureSdf;

const ALL: [CharacterState; 5] = [
    CharacterState::Grounded,
    CharacterState::Airborne,
    CharacterState::Crouched,
    CharacterState::Swimming,
    CharacterState::Sliding,
];

fn ctx(g: bool, slope_deg: f32, water: bool, crouch: bool, jump: bool) -> CharacterStateContext {
    CharacterStateContext {
        is_grounded: g,
        slope_radians: slope_deg.to_radians(),
        in_water: water,
        crouch_requested: crouch,
        jump_pressed: jump,
        max_walkable_slope: 45.0_f32.to_radians(),
    }
}

#[test]
fn name_and_accepts_locomotion_tables() {
    let names = ["grounded", "airborne", "crouched", "swimming", "sliding"];
    let accepts = [true, false, true, true, false];
    for (i, s) in ALL.iter().enumerate() {
        assert_eq!(s.name(), names[i]);
        assert_eq!(s.accepts_locomotion(), accepts[i], "{}", names[i]);
    }
}

#[test]
fn standing_defaults() {
    let c = CharacterStateContext::standing();
    assert!(c.is_grounded && !c.in_water && !c.crouch_requested && !c.jump_pressed);
    assert_eq!(c.slope_radians, 0.0);
    assert_eq!(c.max_walkable_slope, core::f32::consts::FRAC_PI_4);
}

#[test]
fn transition_table_rows_are_independent_of_current_state() {
    // (ctx, expected) rows from the documented priority order:
    // water > (!grounded | jump) > slope > crouch > grounded.
    let rows = [
        (
            ctx(true, 0.0, false, false, false),
            CharacterState::Grounded,
        ),
        (ctx(true, 0.0, true, false, false), CharacterState::Swimming),
        (ctx(false, 0.0, true, true, true), CharacterState::Swimming),
        (
            ctx(false, 0.0, false, false, false),
            CharacterState::Airborne,
        ),
        (ctx(true, 0.0, false, true, true), CharacterState::Airborne),
        (
            ctx(true, 60.0, false, false, false),
            CharacterState::Sliding,
        ),
        (ctx(true, 60.0, false, true, false), CharacterState::Sliding),
        (
            ctx(true, 60.0, false, false, true),
            CharacterState::Airborne,
        ),
        (
            ctx(false, 60.0, false, false, false),
            CharacterState::Airborne,
        ),
        (
            ctx(true, 30.0, false, true, false),
            CharacterState::Crouched,
        ),
        (
            ctx(true, 30.0, false, false, false),
            CharacterState::Grounded,
        ),
    ];
    for (c, want) in rows {
        for cur in ALL {
            assert_eq!(transition(cur, c), want, "from {cur:?} with {c:?}");
        }
    }
}

#[test]
fn slope_threshold_is_strict() {
    let mut c = CharacterStateContext::standing();
    c.max_walkable_slope = 0.5;
    c.slope_radians = 0.5;
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Grounded
    );
    c.slope_radians = 0.500_001;
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Sliding
    );
}

fn tilted_plane(theta: f32) -> ClosureSdf {
    // f(p) = n . p with n = (-sin t, cos t, 0): exact distance, constant normal.
    let (s, c) = (theta.sin(), theta.cos());
    ClosureSdf::new(
        move |x, y, _z| -s * x + c * y,
        move |_x, _y, _z| (-s, c, 0.0),
    )
}

#[test]
fn ground_contact_closed_form_on_flat_plane() {
    let field = tilted_plane(0.0);
    let mut ch = SdfCharacter::new([0.0, 0.4, 0.0], 0.35, 1.8);
    // probe = y - (radius + eps) = 0.4 - 0.40 = 0.0 -> distance 0, normal +Y.
    let c = ch.ground_contact(&field).expect("contact");
    assert!(c.distance.abs() < 1e-6);
    assert_eq!(c.normal, [0.0, 1.0, 0.0]);
    assert!((c.up_alignment - 1.0).abs() < 1e-6);
    assert!(ch.is_grounded(&field));
    // Probe distance = h - 0.40; contact iff <= eps (0.05) -> h <= 0.45.
    ch.position[1] = 0.449;
    assert!(ch.ground_contact(&field).is_some());
    ch.position[1] = 0.452;
    assert!(ch.ground_contact(&field).is_none());
    assert!(!ch.is_grounded(&field));
    // Buried: negative distance is still a contact.
    ch.position[1] = -3.0;
    assert!(ch.ground_contact(&field).is_some());
}

#[test]
fn up_axis_is_normalised_and_alignment_scales_with_slope() {
    // Non-unit up must not change alignment.
    let theta = 30.0_f32.to_radians();
    let field = tilted_plane(theta);
    let mut ch = SdfCharacter::new([0.0, 0.4, 0.0], 0.35, 1.8);
    ch.up = [0.0, 5.0, 0.0];
    let c = ch.ground_contact(&field).expect("contact");
    assert!((c.up_alignment - theta.cos()).abs() < 1e-6);
    // threshold 0.7 > cos(60) = 0.5, < cos(30) = 0.866.
    assert!(ch.is_grounded(&field));
    let steep = tilted_plane(60.0_f32.to_radians());
    let mut ch2 = SdfCharacter::new([0.0, 0.4, 0.0], 0.35, 1.8);
    ch2.up = [0.0, 1.0, 0.0];
    assert!(!ch2.is_grounded(&steep));
    // Exactly at threshold counts as grounded (>=).
    ch2.ground_up_threshold = 0.5_f32.min(60.0_f32.to_radians().cos());
    assert!(ch2.is_grounded(&steep));
}

#[test]
fn locomotion_context_feeds_the_state_machine() {
    let max = 45.0_f32.to_radians();
    let ch = SdfCharacter::new([0.0, 0.4, 0.0], 0.35, 1.8);
    for (deg, want) in [
        (0.0_f32, CharacterState::Grounded),
        (30.0, CharacterState::Grounded),
        (60.0, CharacterState::Sliding),
    ] {
        let field = tilted_plane(deg.to_radians());
        let c = ch.locomotion_context(&field, false, false, false, max);
        assert!(c.is_grounded);
        assert!((c.slope_radians - deg.to_radians()).abs() < 2e-3, "{deg}");
        assert_eq!(transition(CharacterState::Airborne, c), want, "{deg}");
    }
    // 60 degrees is NOT `is_grounded()` at the default threshold, yet the state is Sliding.
    assert!(!ch.is_grounded(&tilted_plane(60.0_f32.to_radians())));

    // In the air: no contact, slope 0, Airborne.
    let air = SdfCharacter::new([0.0, 5.0, 0.0], 0.35, 1.8);
    let flat = tilted_plane(0.0);
    let c = air.locomotion_context(&flat, false, false, false, max);
    assert!(!c.is_grounded);
    assert_eq!(c.slope_radians, 0.0);
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Airborne
    );

    // Inputs are copied through.
    let c = ch.locomotion_context(&flat, true, true, true, 1.25);
    assert!(c.in_water && c.crouch_requested && c.jump_pressed);
    assert_eq!(c.max_walkable_slope, 1.25);
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Swimming
    );
    let c = ch.locomotion_context(&flat, false, true, false, max);
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Crouched
    );
}

#[test]
fn locomotion_context_slope_of_ceiling_like_normal_is_obtuse() {
    // normal (0,-1,0): alignment -1 -> slope pi.
    let field = ClosureSdf::new(|_x, y, _z| -y - 0.4, |_x, _y, _z| (0.0, -1.0, 0.0));
    let ch = SdfCharacter::new([0.0, 0.0, 0.0], 0.35, 1.8);
    let c = ch.locomotion_context(&field, false, false, false, 0.7);
    assert!(c.is_grounded);
    assert!((c.slope_radians - core::f32::consts::PI).abs() < 1e-3);
    assert_eq!(
        transition(CharacterState::Grounded, c),
        CharacterState::Sliding
    );
}

#[test]
fn probe_distance_exactly_epsilon_is_a_contact() {
    // Dyadic numbers so the boundary is exact: reach = 0.5 + 0.25 = 0.75,
    // probe y = 1.0 - 0.75 = 0.25 = eps on the plane f = y.
    let field = tilted_plane(0.0);
    let mut ch = SdfCharacter::new([0.0, 1.0, 0.0], 0.5, 1.8);
    ch.ground_probe_epsilon = 0.25;
    assert!(ch.ground_contact(&field).is_some());
    ch.position[1] = 1.015_625; // one dyadic step higher: distance 0.265625 > eps
    assert!(ch.ground_contact(&field).is_none());
}
