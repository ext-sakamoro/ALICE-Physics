//! Locomotion state machine for kinematic character controllers.
//!
//! Complements [`crate::character`] (capsule move-and-slide) and
//! [`crate::sdf_character`] (SDF-swept move-and-slide) by supplying a
//! deterministic state enum + transition table that callers can drive
//! from per-tick sensor readings (ground contact, slope angle, water
//! immersion, crouch button).
//!
//! The module is intentionally logic-only — it does not touch a
//! `RigidBody`, does not consume delta time, and does not depend on
//! `Fix128`. Downstream character controllers apply the resulting
//! [`CharacterState`] as a lookup key for movement parameters (walk
//! speed, jump height, drag).

/// Discrete locomotion states.
///
/// The state machine is fully described by [`transition`]; callers
/// should not reason about invalid states elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CharacterState {
    /// Standing / walking on a supportive ground surface.
    Grounded,
    /// In free-fall or jumping — no ground contact.
    Airborne,
    /// Ducked below full height on the ground.
    Crouched,
    /// Fully submerged in a fluid volume; ignores ground.
    Swimming,
    /// On the ground but the slope exceeds the walkable threshold —
    /// the character slides down.
    Sliding,
}

impl CharacterState {
    /// Machine-readable identifier for logging / debugging.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Grounded => "grounded",
            Self::Airborne => "airborne",
            Self::Crouched => "crouched",
            Self::Swimming => "swimming",
            Self::Sliding => "sliding",
        }
    }

    /// True when the character can accept horizontal locomotion
    /// input (walk / run). All states except `Sliding` (input is
    /// dampened by momentum) and `Airborne` (limited air control by
    /// convention) accept input.
    #[must_use]
    pub const fn accepts_locomotion(self) -> bool {
        matches!(self, Self::Grounded | Self::Crouched | Self::Swimming)
    }
}

/// Per-tick sensor readings that drive [`transition`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterStateContext {
    /// True when the character has a supportive contact underfoot.
    pub is_grounded: bool,
    /// Slope angle of the ground contact in radians. `0` is flat.
    /// Used to distinguish `Grounded` from `Sliding` when
    /// `is_grounded` is true.
    pub slope_radians: f32,
    /// True when the character volume is submerged in a fluid.
    pub in_water: bool,
    /// True when the crouch button (or auto-crouch trigger) is held.
    pub crouch_requested: bool,
    /// True when the jump button was pressed this tick. Overrides
    /// grounded → airborne even when the crouch button is held.
    pub jump_pressed: bool,
    /// Maximum walkable slope angle in radians. Slopes steeper than
    /// this force the character into `Sliding`.
    pub max_walkable_slope: f32,
}

impl CharacterStateContext {
    /// Reasonable defaults for a bipedal humanoid: 45° max walkable
    /// slope, standing.
    #[must_use]
    pub fn standing() -> Self {
        Self {
            is_grounded: true,
            slope_radians: 0.0,
            in_water: false,
            crouch_requested: false,
            jump_pressed: false,
            max_walkable_slope: core::f32::consts::FRAC_PI_4,
        }
    }
}

/// Return the character state for the next tick given the current
/// state and this tick's sensor readings.
///
/// Priority order:
///
/// 1. `in_water` → `Swimming` (submerged takes precedence over gravity).
/// 2. `!is_grounded` OR `jump_pressed` → `Airborne`.
/// 3. `slope_radians > max_walkable_slope` → `Sliding`.
/// 4. `crouch_requested` → `Crouched`.
/// 5. Otherwise → `Grounded`.
#[must_use]
pub fn transition(current: CharacterState, ctx: CharacterStateContext) -> CharacterState {
    let _ = current;
    if ctx.in_water {
        return CharacterState::Swimming;
    }
    if ctx.jump_pressed || !ctx.is_grounded {
        return CharacterState::Airborne;
    }
    if ctx.slope_radians > ctx.max_walkable_slope {
        return CharacterState::Sliding;
    }
    if ctx.crouch_requested {
        return CharacterState::Crouched;
    }
    CharacterState::Grounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_context_yields_grounded() {
        let next = transition(CharacterState::Airborne, CharacterStateContext::standing());
        assert_eq!(next, CharacterState::Grounded);
    }

    #[test]
    fn losing_ground_transitions_to_airborne() {
        let mut ctx = CharacterStateContext::standing();
        ctx.is_grounded = false;
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Airborne
        );
    }

    #[test]
    fn crouch_input_when_grounded_transitions_to_crouched() {
        let mut ctx = CharacterStateContext::standing();
        ctx.crouch_requested = true;
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Crouched
        );
    }

    #[test]
    fn steep_slope_transitions_to_sliding() {
        let mut ctx = CharacterStateContext::standing();
        // 60° slope > 45° walkable.
        ctx.slope_radians = 60.0_f32.to_radians();
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Sliding
        );
    }

    #[test]
    fn shallow_slope_stays_grounded() {
        let mut ctx = CharacterStateContext::standing();
        ctx.slope_radians = 30.0_f32.to_radians();
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Grounded
        );
    }

    #[test]
    fn water_takes_priority_over_ground() {
        let mut ctx = CharacterStateContext::standing();
        ctx.in_water = true;
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Swimming
        );
    }

    #[test]
    fn jump_press_transitions_to_airborne_even_when_grounded() {
        let mut ctx = CharacterStateContext::standing();
        ctx.jump_pressed = true;
        assert_eq!(
            transition(CharacterState::Grounded, ctx),
            CharacterState::Airborne
        );
    }

    #[test]
    fn accepts_locomotion_flags_correct_states() {
        assert!(CharacterState::Grounded.accepts_locomotion());
        assert!(CharacterState::Crouched.accepts_locomotion());
        assert!(CharacterState::Swimming.accepts_locomotion());
        assert!(!CharacterState::Airborne.accepts_locomotion());
        assert!(!CharacterState::Sliding.accepts_locomotion());
    }

    #[test]
    fn state_names_stable() {
        assert_eq!(CharacterState::Grounded.name(), "grounded");
        assert_eq!(CharacterState::Airborne.name(), "airborne");
        assert_eq!(CharacterState::Crouched.name(), "crouched");
        assert_eq!(CharacterState::Swimming.name(), "swimming");
        assert_eq!(CharacterState::Sliding.name(), "sliding");
    }
}
