//! SDF character locomotion state machine.
//!
//! Probes the ground under an `SdfCharacter` on flat and tilted planes,
//! feeds the readings to `character_state::transition` and checks each state
//! against the hand-derived expectation (slope angle of the plane vs. a 45
//! degree walkable limit).
//!
//! ```bash
//! cargo run --release --example character_state_machine --features std
//! ```

#![allow(clippy::disallowed_methods)]

use alice_physics::character_state::{transition, CharacterState, CharacterStateContext};
use alice_physics::sdf_character::SdfCharacter;
use alice_physics::sdf_collider::ClosureSdf;

fn plane(theta: f32) -> ClosureSdf {
    let (s, c) = (theta.sin(), theta.cos());
    ClosureSdf::new(
        move |x, y, _z| -s * x + c * y,
        move |_x, _y, _z| (-s, c, 0.0),
    )
}

fn main() {
    let max_slope = CharacterStateContext::standing().max_walkable_slope;
    let ch = SdfCharacter::new([0.0, 0.4, 0.0], 0.35, 1.8);
    let mut state = CharacterState::Airborne;
    for (deg, want) in [
        (0.0_f32, CharacterState::Grounded),
        (30.0, CharacterState::Grounded),
        (60.0, CharacterState::Sliding),
    ] {
        let field = plane(deg.to_radians());
        let contact = ch.ground_contact(&field).expect("probe lands on the plane");
        let ctx = ch.locomotion_context(&field, false, false, false, max_slope);
        state = transition(state, ctx);
        println!(
            "[character_state] slope {deg:>4.0} deg: alignment {:.4} (cos = {:.4}) is_grounded={} -> {} (accepts input: {})",
            contact.up_alignment,
            deg.to_radians().cos(),
            ch.is_grounded(&field),
            state.name(),
            state.accepts_locomotion()
        );
        assert_eq!(state, want, "slope {deg}");
        assert!((contact.up_alignment - deg.to_radians().cos()).abs() < 1e-5);
    }
    let air = SdfCharacter::new([0.0, 5.0, 0.0], 0.35, 1.8);
    let ctx = air.locomotion_context(&plane(0.0), false, false, false, max_slope);
    state = transition(state, ctx);
    println!("[character_state] in the air -> {}", state.name());
    assert_eq!(state, CharacterState::Airborne);
}
