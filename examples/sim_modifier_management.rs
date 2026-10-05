//! Production entry point for the `ModifiedSdf` builder/management API:
//! `add_modifier`, `clear_modifiers`, `modifier_count`, `modifier_mut`,
//! `with_modifier` (`src/sim_modifier.rs`).
//!
//! Wiring: `scripts/wiring-baseline.txt` lists all five as `unwired` -- the
//! module's own `#[cfg(test)]` block exercises them, but tests do not count
//! as production callers for the wiring guard, and nothing in `src/` /
//! `examples/` / `benches/` called any of the five before this file existed.
//! This example is that caller.
//!
//! This file is about the **collection API** (`Vec<Box<dyn PhysicsModifier>>`
//! add/remove/count/mutate) -- it is unrelated to this crate's open
//! research question of whether individual modifiers (thermal / phase_change
//! / pressure / erosion / fracture) can read each other's fields (resolved
//! elsewhere as "no shared `Fix128` coupling layer needed"). No modifier here
//! reads another modifier's state; each only ever sees the running SDF
//! distance handed down the chain, exactly as `ModifiedSdf::eval_distance`
//! already does.
//!
//! `tests/analytic_sim_modifier_wiring.rs` holds the closed-form oracles for
//! the same five methods (count bookkeeping, out-of-range `modifier_mut`,
//! in-place mutation through the returned reference, and the post-`clear`
//! empty state) that this file's diagnostic prints do not repeat.
//!
//! ```bash
//! cargo run --example sim_modifier_management --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::sdf_collider::{ClosureSdf, SdfField};
use alice_physics::sim_modifier::{ModifiedSdf, PhysicsModifier};
use alice_physics::thermal::{ThermalConfig, ThermalModifier};

/// A modifier whose effect is a single known number, and whose `update`
/// grows that number by a known amount. Unlike the crate's physical
/// modifiers (thermal diffusion, erosion exposure decay, ...) its output is
/// exact arithmetic, so every assertion below is a closed-form equality
/// rather than a sign/bound check.
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

/// Ground plane: `distance(x, y, z) = y`. Picked (as in `src/sim_modifier.rs`'s
/// own tests) because it carries zero numerical error, so every modifier
/// offset below is observable bit-for-bit in `ModifiedSdf::distance`.
fn ground() -> ClosureSdf {
    ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))
}

fn main() {
    // ------------------------------------------------------------------
    // 1. modifier_count on a freshly built ModifiedSdf: zero, and the SDF
    //    passes through unmodified (no modifiers in the chain yet).
    // ------------------------------------------------------------------
    let modified = ModifiedSdf::new(Box::new(ground()));
    assert_eq!(
        modified.modifier_count(),
        0,
        "[sim_modifier] MISMATCH modifier_count: fresh ModifiedSdf must start empty"
    );
    let d0 = modified.distance(0.0, 2.0, 0.0);
    assert!(
        (d0 - 2.0).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance with zero modifiers: got {d0}, want 2.0 (passthrough)"
    );
    println!(
        "[sim_modifier] ok modifier_count()=0 on fresh ModifiedSdf, distance passes through: {d0}"
    );

    // ------------------------------------------------------------------
    // 2. with_modifier: builder-style chain. Two OffsetModifier instances
    //    (0.25 + 0.5 = 0.75 total) pushed via consuming `self -> Self`
    //    calls.
    // ------------------------------------------------------------------
    let mut modified = modified
        .with_modifier(Box::new(OffsetModifier { amount: 0.25 }))
        .with_modifier(Box::new(OffsetModifier { amount: 0.5 }));
    assert_eq!(
        modified.modifier_count(),
        2,
        "[sim_modifier] MISMATCH modifier_count after two with_modifier calls"
    );
    let d1 = modified.distance(0.0, 2.0, 0.0);
    let want1 = 2.0 - 0.25 - 0.5;
    assert!(
        (d1 - want1).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance after with_modifier chain: got {d1}, want {want1}"
    );
    println!(
        "[sim_modifier] ok with_modifier x2: modifier_count()={}, distance={d1} (want {want1})",
        modified.modifier_count()
    );

    // ------------------------------------------------------------------
    // 3. add_modifier (the &mut-self sibling of with_modifier): push a real
    //    crate modifier, not a hand-rolled one. ThermalConfig::default()
    //    keeps the field at `ambient_temperature` everywhere (no heat
    //    sources were added), and at that temperature every branch in
    //    ThermalModifier::modify_distance is inert (no melt accumulated,
    //    temp_delta == 0 so no expansion, temp == ambient == 20.0 which is
    //    above freeze_temperature == -10.0 so no freeze growth) -- so this
    //    push changes modifier_count but, verified below, nothing else.
    // ------------------------------------------------------------------
    let thermal = ThermalModifier::new(
        ThermalConfig::default(),
        4,
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
    );
    modified.add_modifier(Box::new(thermal));
    assert_eq!(
        modified.modifier_count(),
        3,
        "[sim_modifier] MISMATCH modifier_count after add_modifier"
    );
    let d2 = modified.distance(0.0, 2.0, 0.0);
    assert!(
        (d2 - want1).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance after adding inert ThermalModifier: got {d2}, want {want1} (unchanged)"
    );
    println!(
        "[sim_modifier] ok add_modifier(ThermalModifier::default): modifier_count()={}, distance={d2} (unchanged, thermal inert at ambient)",
        modified.modifier_count()
    );

    // ------------------------------------------------------------------
    // 4. modifier_mut: out-of-range index is None; in-range index returns a
    //    live &mut reference whose effect on `update` is observable back
    //    through ModifiedSdf::distance on the next call.
    // ------------------------------------------------------------------
    assert!(
        modified.modifier_mut(99).is_none(),
        "[sim_modifier] MISMATCH modifier_mut(99): out-of-range index must return None"
    );
    println!("[sim_modifier] ok modifier_mut(99) = None (out of range, count=3)");

    match modified.modifier_mut(0) {
        // Identity is confirmed by effect (the distance delta below), not by
        // calling `.name()`: `PhysicsModifier::name` is implemented by
        // several unrelated types across this crate (thermal/erosion/
        // pressure/fracture/phase_change all report their own fixed string),
        // and `wiring_guard.py` counts production callers by bare
        // identifier, so an unqualified `m.name()` call here would also
        // flip `src/character_state.rs::name` (a same-named, unrelated
        // method) to "wired" -- a false positive, not a real new caller.
        Some(m) => m.update(1.0), // 0.25 -> 1.25
        None => {
            panic!("[sim_modifier] MISMATCH modifier_mut(0): expected Some, index 0 is in range")
        }
    }
    let d3 = modified.distance(0.0, 2.0, 0.0);
    // total offset now 1.25 (mutated) + 0.5 (untouched) + 0.0 (thermal, inert) = 1.75
    let want3 = 2.0 - 1.25 - 0.5;
    assert!(
        (d3 - want3).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance after modifier_mut(0).update(1.0): got {d3}, want {want3}"
    );
    println!(
        "[sim_modifier] ok modifier_mut(0).update(1.0) mutated in place: distance={d3} (want {want3})"
    );

    // ------------------------------------------------------------------
    // 5. clear_modifiers: count resets to 0, distance reverts to the
    //    untouched original field, and modifier_mut on any index is None.
    // ------------------------------------------------------------------
    modified.clear_modifiers();
    assert_eq!(
        modified.modifier_count(),
        0,
        "[sim_modifier] MISMATCH modifier_count after clear_modifiers"
    );
    let d4 = modified.distance(0.0, 2.0, 0.0);
    assert!(
        (d4 - 2.0).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance after clear_modifiers: got {d4}, want 2.0 (original field)"
    );
    assert!(
        modified.modifier_mut(0).is_none(),
        "[sim_modifier] MISMATCH modifier_mut(0) after clear_modifiers: chain is empty, must be None"
    );
    println!("[sim_modifier] ok clear_modifiers: modifier_count()=0, distance={d4} (original field), modifier_mut(0)=None");

    // ------------------------------------------------------------------
    // 6. add_modifier after clear: the chain is reusable, not a one-shot.
    // ------------------------------------------------------------------
    modified.add_modifier(Box::new(OffsetModifier { amount: 1.0 }));
    assert_eq!(
        modified.modifier_count(),
        1,
        "[sim_modifier] MISMATCH modifier_count after re-adding post-clear"
    );
    let d5 = modified.distance(0.0, 2.0, 0.0);
    let want5 = 2.0 - 1.0;
    assert!(
        (d5 - want5).abs() < 1e-6,
        "[sim_modifier] MISMATCH distance after re-adding post-clear: got {d5}, want {want5}"
    );
    println!(
        "[sim_modifier] ok add_modifier after clear_modifiers (chain reusable): modifier_count()={}, distance={d5} (want {want5})",
        modified.modifier_count()
    );

    println!("[sim_modifier] all checks passed");
}
