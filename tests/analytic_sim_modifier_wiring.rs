//! Closed-form / invariant oracles for the wiring of
//! `alice_physics::sim_modifier::ModifiedSdf`'s builder/management API
//! (`examples/sim_modifier_management.rs`): `add_modifier`,
//! `clear_modifiers`, `modifier_count`, `modifier_mut`, `with_modifier`.
//!
//! This file is about the **collection API** on `ModifiedSdf` (a
//! `Vec<Box<dyn PhysicsModifier>>` with add/remove/count/mutate), not about
//! this crate's research question of whether individual modifiers
//! (thermal / phase_change / pressure / erosion / fracture) can read each
//! other's fields -- that is a separate, already-resolved question. No
//! modifier below reads another modifier's state.
//!
//! The oracle throughout is plain arithmetic: every modifier here is a
//! deterministic closed-form offset (`OffsetModifier`), applied against a
//! ground-plane SDF (`distance(x, y, z) = y`) that carries zero numerical
//! error, so expected values are hand-computed sums/differences, never
//! re-derived by calling `ModifiedSdf` a second time.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier};

/// A modifier with a known, mutable effect: `modify_distance` subtracts
/// `amount`, and `update(dt)` grows `amount` by `dt`. Pure arithmetic, no
/// hidden state beyond the one field under test's control.
struct OffsetModifier {
    amount: f32,
}

impl PhysicsModifier for OffsetModifier {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d - self.amount
    }
    fn update(&mut self, dt: f32) {
        self.amount += dt;
    }
    fn name(&self) -> &'static str {
        "offset"
    }
}

/// Ground plane: `distance(x, y, z) = y`, normal always `(0, 1, 0)`.
fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

// ---------------------------------------------------------------------------
// modifier_count: starts at 0, increments on add_modifier / with_modifier
// ---------------------------------------------------------------------------

#[test]
fn modifier_count_starts_at_zero_on_a_fresh_chain() {
    let modified = ModifiedSdf::new(Box::new(ground()));
    assert_eq!(modified.modifier_count(), 0);
}

#[test]
fn modifier_count_increments_by_one_per_add_modifier_call() {
    let mut modified = ModifiedSdf::new(Box::new(ground()));
    assert_eq!(modified.modifier_count(), 0);

    modified.add_modifier(Box::new(OffsetModifier { amount: 0.1 }));
    assert_eq!(modified.modifier_count(), 1);

    modified.add_modifier(Box::new(OffsetModifier { amount: 0.2 }));
    assert_eq!(modified.modifier_count(), 2);

    modified.add_modifier(Box::new(OffsetModifier { amount: 0.3 }));
    assert_eq!(modified.modifier_count(), 3);
}

#[test]
fn modifier_count_increments_by_one_per_with_modifier_call() {
    let modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 0.1 }))
        .with_modifier(Box::new(OffsetModifier { amount: 0.2 }))
        .with_modifier(Box::new(OffsetModifier { amount: 0.3 }));
    assert_eq!(modified.modifier_count(), 3);
}

#[test]
fn with_modifier_and_add_modifier_compose_on_the_same_chain() {
    // with_modifier (consuming builder) followed by add_modifier (&mut)
    // on the result must keep accumulating the same Vec, not reset it.
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 0.1 }))
        .with_modifier(Box::new(OffsetModifier { amount: 0.2 }));
    assert_eq!(modified.modifier_count(), 2);

    modified.add_modifier(Box::new(OffsetModifier { amount: 0.3 }));
    assert_eq!(modified.modifier_count(), 3);

    // and the effect on distance is the sum of all three amounts, confirming
    // the count increase is backed by real entries in the chain (not an
    // off-by-one in a counter that doesn't track the Vec).
    let d = modified.distance(0.0, 5.0, 0.0);
    let want = 5.0 - 0.1 - 0.2 - 0.3;
    assert!((d - want).abs() < 1e-6, "got {d}, want {want}");
}

// ---------------------------------------------------------------------------
// modifier_mut: None out of range, Some with live &mut in range
// ---------------------------------------------------------------------------

#[test]
fn modifier_mut_is_none_on_an_empty_chain() {
    let mut modified = ModifiedSdf::new(Box::new(ground()));
    assert!(modified.modifier_mut(0).is_none());
}

#[test]
fn modifier_mut_is_none_for_every_index_at_or_past_the_count() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 1.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 2.0 }));
    assert_eq!(modified.modifier_count(), 2);

    // valid indices 0 and 1 exist; 2, 3 and a far-out-of-range index do not.
    assert!(modified.modifier_mut(0).is_some());
    assert!(modified.modifier_mut(1).is_some());
    assert!(modified.modifier_mut(2).is_none());
    assert!(modified.modifier_mut(3).is_none());
    assert!(modified.modifier_mut(usize::MAX).is_none());
}

#[test]
fn modifier_mut_returns_the_modifier_at_that_index_in_push_order() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 1.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 2.0 }));

    // name() is the same for both (OffsetModifier always reports "offset"),
    // so identity is confirmed by effect, not by name: mutating index 0 must
    // not be observable as a change at index 1's position in the chain.
    let before = modified.distance(0.0, 10.0, 0.0);
    assert!((before - (10.0 - 1.0 - 2.0)).abs() < 1e-6);

    match modified.modifier_mut(0) {
        Some(m) => m.update(5.0), // amount 1.0 -> 6.0
        None => panic!("index 0 must exist"),
    }
    let after = modified.distance(0.0, 10.0, 0.0);
    // only the first modifier's amount grew: 10 - 6.0 - 2.0 = 2.0
    assert!(
        (after - 2.0).abs() < 1e-6,
        "mutating index 0 must change only index 0's contribution, got {after}"
    );
}

#[test]
fn modifier_mut_returned_reference_mutates_the_chain_in_place() {
    // The defining contract of modifier_mut: the &mut it hands back is not a
    // detached copy -- updates through it are visible on every subsequent
    // call (not just the next one), verified here by reading back through
    // ModifiedSdf::distance three times in a row.
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 0.0 }));

    for step in 1..=3 {
        match modified.modifier_mut(0) {
            Some(m) => m.update(1.0), // amount grows by 1.0 each call: 0,1,2,3
            None => panic!("index 0 must exist"),
        }
        let got = modified.distance(0.0, 10.0, 0.0);
        let want = 10.0 - step as f32;
        assert!(
            (got - want).abs() < 1e-6,
            "step {step}: got {got}, want {want} (in-place mutation must accumulate)"
        );
    }
}

// ---------------------------------------------------------------------------
// clear_modifiers: resets count to 0, and the empty state is reflected
// everywhere afterward (distance, modifier_mut, re-add).
// ---------------------------------------------------------------------------

#[test]
fn clear_modifiers_resets_count_to_zero() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 1.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 2.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 3.0 }));
    assert_eq!(modified.modifier_count(), 3);

    modified.clear_modifiers();
    assert_eq!(modified.modifier_count(), 0);
}

#[test]
fn clear_modifiers_on_an_already_empty_chain_is_a_no_op() {
    let mut modified = ModifiedSdf::new(Box::new(ground()));
    assert_eq!(modified.modifier_count(), 0);
    modified.clear_modifiers();
    assert_eq!(modified.modifier_count(), 0);
    modified.clear_modifiers();
    assert_eq!(modified.modifier_count(), 0);
}

#[test]
fn clear_modifiers_restores_the_untouched_original_field() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 0.4 }))
        .with_modifier(Box::new(OffsetModifier { amount: 0.6 }));
    let with_mods = modified.distance(0.0, 3.0, 0.0);
    assert!((with_mods - (3.0 - 0.4 - 0.6)).abs() < 1e-6);

    modified.clear_modifiers();
    let cleared = modified.distance(0.0, 3.0, 0.0);
    assert!(
        (cleared - 3.0).abs() < 1e-6,
        "cleared distance must equal the original field's value, got {cleared}"
    );

    let (nx, ny, nz) = modified.normal(0.0, 3.0, 0.0);
    assert!(
        nx.abs() < 1e-6 && (ny - 1.0).abs() < 1e-6 && nz.abs() < 1e-6,
        "normal after clear must be the original field's normal, got ({nx}, {ny}, {nz})"
    );
}

#[test]
fn clear_modifiers_makes_every_index_return_none_from_modifier_mut() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 1.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 2.0 }));
    assert!(modified.modifier_mut(0).is_some());
    assert!(modified.modifier_mut(1).is_some());

    modified.clear_modifiers();

    assert!(modified.modifier_mut(0).is_none());
    assert!(modified.modifier_mut(1).is_none());
}

#[test]
fn chain_is_reusable_after_clear_modifiers() {
    let mut modified = ModifiedSdf::new(Box::new(ground()))
        .with_modifier(Box::new(OffsetModifier { amount: 1.0 }))
        .with_modifier(Box::new(OffsetModifier { amount: 2.0 }));
    assert_eq!(modified.modifier_count(), 2);

    modified.clear_modifiers();
    assert_eq!(modified.modifier_count(), 0);

    modified.add_modifier(Box::new(OffsetModifier { amount: 0.5 }));
    assert_eq!(modified.modifier_count(), 1);
    let got = modified.distance(0.0, 1.0, 0.0);
    assert!((got - 0.5).abs() < 1e-6, "got {got}, want 0.5");

    let modified = modified.with_modifier(Box::new(OffsetModifier { amount: 0.25 }));
    assert_eq!(modified.modifier_count(), 2);
    let got2 = modified.distance(0.0, 1.0, 0.0);
    assert!((got2 - 0.25).abs() < 1e-6, "got {got2}, want 0.25");
}
