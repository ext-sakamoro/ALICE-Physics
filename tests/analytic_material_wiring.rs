//! Oracles for the production entry points of `alice_physics::material`
//! driven by `examples/material_registry_presets.rs`: `MaterialTable::{
//! register_concrete, register_ice, register_metal, register_rubber,
//! register_wood, set_pair_override}` and the `PhysicsMaterial` builder
//! methods `with_combine_rules` / `with_static_friction`.
//!
//! # Why these had zero production callers
//!
//! `src/solver.rs` calls `register_metal` / `register_rubber` /
//! `set_pair_override` from `world.material_table` — but only inside its own
//! `#[cfg(test)] mod tests` (the module starts `src/solver.rs:4100`), so
//! `scripts/wiring_guard.py` never counted it. `register_concrete`,
//! `register_ice`, `register_wood`, `with_combine_rules` and
//! `with_static_friction` had no reference anywhere outside `material.rs`
//! itself, test included.
//!
//! # Closed forms and where they come from
//!
//! The module doc (`src/material.rs:1-11`) does not cite an external
//! handbook, so "documented" here means the literal constants fixed in the
//! `register_*` function bodies (`src/material.rs:271-312`), hand-copied
//! below rather than read back from calling the functions:
//!
//! | preset   | friction | restitution | friction_combine | restitution_combine |
//! |----------|----------|-------------|-------------------|----------------------|
//! | metal    | 0.40     | 0.10        | Average (default) | Average (default)  |
//! | wood     | 0.50     | 0.30        | Average (default) | Average (default)  |
//! | rubber   | 0.80     | 0.80        | Max                | Max                 |
//! | ice      | 0.05     | 0.10        | Min                | Min                  |
//! | concrete | 0.60     | 0.20        | Average (default) | Average (default)  |
//!
//! `CombineRule::apply` (`src/material.rs:38-62`) is `Average = (a+b)/2`,
//! `Min`, `Max`, `Multiply = a*b` — not under test here (it has its own
//! `#[cfg(test)]` unit test and is not one of the eight baseline items), but
//! its documented semantics are what the hand-computed closed forms below
//! apply. `combine_rule_priority` (`src/material.rs:322-337`, also not a
//! baseline item) picks `Max(3) > Multiply(2) > Average(1) > Min(0)`, ties
//! keeping the first operand's rule (irrelevant when tied, since a tie means
//! both operands carry the same rule).
//!
//! # Degenerate input
//!
//! * **No override, default combine rule**: two freshly `PhysicsMaterial::new`
//!   materials default to `CombineRule::Average` (the `#[default]` variant,
//!   also what the `new` constructor assigns) and the pair isn't in
//!   `pair_overrides`, so `combine()` falls through to the plain average.
//! * **Querying an id that was never registered** (the closest available
//!   proxy for "un-register a material" — `MaterialTable` has **no removal
//!   API at all**, so a true un-register/re-query cannot be exercised; see
//!   the blocker note in the final report): `get()`
//!   (`src/material.rs:179-185`) falls back to `materials[0]` via
//!   `.unwrap_or(&self.materials[0])`. Since `MaterialTable::new()` always
//!   registers one material at index 0 before anything else is registered,
//!   `materials[0]` is always `DEFAULT_MATERIAL`'s own slot, so
//!   `combine(never_registered_id, x)` is pinned to equal
//!   `combine(DEFAULT_MATERIAL, x)` exactly — not an `Err`, not a panic, a
//!   silent stale read of slot 0.
//! * **Negative / >1 friction coefficients**: `PhysicsMaterial::new` and
//!   `MaterialTable::register` perform no range validation anywhere in
//!   `material.rs`, so out-of-`[0,1]` and negative coefficients are accepted
//!   verbatim and participate in `combine()` arithmetic like any other value.
//! * **Extreme values**: `Fix128::Mul` (`src/math.rs:421-461`) is a pure
//!   wrapping 128-bit fixed-point multiply with no overflow check, so "does
//!   not panic" is a vacuous assertion for it (the crate's own
//!   `analytic_hyperelastic_wiring.rs` makes the same point about `Fix128`
//!   generally). The exact wrapped result is pinned here with a case whose
//!   arithmetic reduces to pure powers of two, so it can be hand-verified
//!   without reimplementing `Fix128`'s 256-bit multiply: let
//!   `x = Fix128::from_int(2^32)` (raw 128-bit integer `2^32 << 64 = 2^96`).
//!   `x * x`'s true 256-bit product is `(2^96)^2 = 2^192`; a fixed-point
//!   multiply right-shifts by 64 (`2^192 >> 64 = 2^128`) and keeps only the
//!   low 128 bits (`2^128 mod 2^128 = 0`). So `CombineRule::Multiply.apply(x,
//!   x)` must be exactly `Fix128::ZERO` — an enormous, clearly out-of-range
//!   "friction coefficient" silently wraps to zero rather than erroring.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::material::{CombineRule, MaterialTable, PhysicsMaterial, DEFAULT_MATERIAL};
use alice_physics::math::Fix128;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn r(num: i64, denom: i64) -> Fix128 {
    Fix128::from_ratio(num, denom)
}

/// `CombineRule::Average::apply` is exact (`Fix128` add + bit-shift `half()`,
/// no precision loss), so the only slop here is the independent rounding of
/// the hand-written ratio literal against the sum's own rounding — same
/// 1/2^56-scale tolerance the existing `src/material.rs` unit tests use.
fn close(a: Fix128, b: Fix128) -> bool {
    (a - b).abs() < Fix128::from_raw(0, 1 << 8)
}

#[test]
fn register_metal_wood_rubber_ice_concrete_match_documented_constants() {
    let mut table = MaterialTable::new();
    let metal = table.register_metal();
    let wood = table.register_wood();
    let rubber = table.register_rubber();
    let ice = table.register_ice();
    let concrete = table.register_concrete();

    let cases = [
        (
            "metal",
            metal,
            r(4, 10),
            r(1, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
        (
            "wood",
            wood,
            r(5, 10),
            r(3, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
        (
            "rubber",
            rubber,
            r(8, 10),
            r(8, 10),
            CombineRule::Max,
            CombineRule::Max,
        ),
        (
            "ice",
            ice,
            r(5, 100),
            r(1, 10),
            CombineRule::Min,
            CombineRule::Min,
        ),
        (
            "concrete",
            concrete,
            r(6, 10),
            r(2, 10),
            CombineRule::Average,
            CombineRule::Average,
        ),
    ];
    for (name, id, friction, restitution, fc, rc) in cases {
        let m = table.get(id);
        assert_eq!(m.id, id, "{name} id reassigned at registration");
        assert_eq!(m.dynamic_friction, friction, "{name} dynamic_friction");
        assert_eq!(
            m.static_friction, friction,
            "{name} static==dynamic at registration"
        );
        assert_eq!(m.restitution, restitution, "{name} restitution");
        assert_eq!(m.friction_combine, fc, "{name} friction_combine");
        assert_eq!(m.restitution_combine, rc, "{name} restitution_combine");
    }

    // The five presets' friction coefficients are pairwise distinct — a
    // mutation that collapsed any two register_* bodies to the same preset
    // would be caught by the exact equalities above, but this also pins the
    // "distinct inputs -> distinct outputs" shape directly.
    let frictions: Vec<Fix128> = cases.iter().map(|c| c.2).collect();
    for i in 0..frictions.len() {
        for j in (i + 1)..frictions.len() {
            assert_ne!(
                frictions[i], frictions[j],
                "{} vs {}",
                cases[i].0, cases[j].0
            );
        }
    }
}

#[test]
fn combine_rule_priority_crosses_presets_correctly() {
    let mut table = MaterialTable::new();
    let wood = table.register_wood();
    let concrete = table.register_concrete();
    let ice = table.register_ice();
    let rubber = table.register_rubber();

    // Average(1) vs Average(1): average.
    let wc = table.combine(wood, concrete);
    assert!(
        close(wc.friction, r(55, 100)),
        "wood x concrete friction {wc:?}"
    );
    assert!(
        close(wc.restitution, r(25, 100)),
        "wood x concrete restitution {wc:?}"
    );

    // Min(0) vs Average(1): average wins.
    let ic = table.combine(ice, concrete);
    assert!(
        close(ic.friction, r(325, 1000)),
        "ice x concrete friction {ic:?}"
    );
    assert!(
        close(ic.restitution, r(15, 100)),
        "ice x concrete restitution {ic:?}"
    );

    // Min(0) vs Max(3): max wins.
    let ir = table.combine(ice, rubber);
    assert_eq!(ir.friction, r(8, 10), "ice x rubber friction {ir:?}");
    assert_eq!(ir.restitution, r(8, 10), "ice x rubber restitution {ir:?}");

    // Symmetry: argument order must not matter (combine() sorts internally).
    assert_eq!(table.combine(rubber, ice), ir);
    assert_eq!(table.combine(concrete, wood), wc);
}

#[test]
fn set_pair_override_takes_precedence_and_is_order_independent_at_both_call_sites() {
    let mut table = MaterialTable::new();
    let metal = table.register_metal();
    let rubber = table.register_rubber();

    // Without an override, metal x rubber uses rubber's Max rule: max(0.4,
    // 0.8)=0.8, max(0.1,0.8)=0.8 — nothing close to the override values below.
    let before = table.combine(metal, rubber);
    assert_eq!(before.friction, r(8, 10));
    assert_eq!(before.restitution, r(8, 10));

    // Set the override with the pair reversed (rubber, metal) relative to how
    // it will be queried (metal, rubber) — exercises the sort-normalization
    // inside set_pair_override itself, not just inside combine().
    table.set_pair_override(rubber, metal, r(15, 100), r(5, 100));

    let after_mr = table.combine(metal, rubber);
    let after_rm = table.combine(rubber, metal);
    assert_eq!(after_mr.friction, r(15, 100));
    assert_eq!(after_mr.restitution, r(5, 100));
    assert_eq!(
        after_rm, after_mr,
        "combine() must be symmetric regardless of override call order"
    );

    // Re-setting the same pair (now in the "natural" order) updates in place
    // rather than appending a second entry — observable via the new value
    // winning outright.
    table.set_pair_override(metal, rubber, r(1, 100), r(0, 1));
    let updated = table.combine(metal, rubber);
    assert_eq!(updated.friction, r(1, 100));
    assert_eq!(updated.restitution, Fix128::ZERO);
}

#[test]
fn with_static_friction_changes_only_the_static_coefficient() {
    let base = PhysicsMaterial::new(7, r(3, 10), r(2, 10));
    let m = base.with_static_friction(r(9, 10));

    assert_eq!(m.static_friction, r(9, 10));
    assert_eq!(m.dynamic_friction, r(3, 10));
    assert_eq!(m.restitution, r(2, 10));
    assert_eq!(m.id, 7);
    assert_eq!(m.friction_combine, base.friction_combine);
    assert_eq!(m.restitution_combine, base.restitution_combine);

    let mut expected = base;
    expected.static_friction = r(9, 10);
    assert_eq!(
        m, expected,
        "with_static_friction must not touch any other field"
    );

    // combine() reads dynamic_friction only, so the builder call is invisible
    // to pairwise results.
    let mut table = MaterialTable::new();
    let plain = table.register(base);
    let sticky = table.register(m);
    assert_eq!(
        table.combine(plain, DEFAULT_MATERIAL),
        table.combine(sticky, DEFAULT_MATERIAL),
        "static_friction must be invisible to combine()"
    );
}

#[test]
fn with_combine_rules_sets_both_fields_and_the_priority_function_consumes_them() {
    let custom = PhysicsMaterial::new(0, r(9, 10), r(9, 10))
        .with_combine_rules(CombineRule::Min, CombineRule::Max);
    assert_eq!(custom.friction_combine, CombineRule::Min);
    assert_eq!(custom.restitution_combine, CombineRule::Max);
    // Untouched fields.
    assert_eq!(custom.dynamic_friction, r(9, 10));
    assert_eq!(custom.static_friction, r(9, 10));
    assert_eq!(custom.restitution, r(9, 10));

    let mut table = MaterialTable::new();
    let wood = table.register_wood();
    let custom_id = table.register(custom);

    // custom's Min(0) friction rule loses to wood's Average(1): avg(0.9,0.5)=0.7.
    // custom's Max(3) restitution rule beats wood's Average(1): max(0.9,0.3)=0.9.
    let cw = table.combine(custom_id, wood);
    assert!(
        close(cw.friction, r(70, 100)),
        "custom x wood friction {cw:?}"
    );
    assert_eq!(cw.restitution, r(9, 10), "custom x wood restitution {cw:?}");
}

#[test]
fn no_override_and_default_combine_rule_is_a_plain_average() {
    assert_eq!(
        CombineRule::default(),
        CombineRule::Average,
        "documented #[default] variant"
    );

    let mut table = MaterialTable::new();
    let a = table.register(PhysicsMaterial::new(0, r(2, 10), r(7, 10)));
    let b = table.register(PhysicsMaterial::new(0, r(8, 10), r(1, 10)));

    let combined = table.combine(a, b);
    assert!(close(combined.friction, r(5, 10)), "{combined:?}");
    assert!(close(combined.restitution, r(4, 10)), "{combined:?}");
}

#[test]
fn querying_a_never_registered_id_falls_back_to_slot_zero_not_err_or_panic() {
    // `MaterialTable` has no removal API — this is the only way to exercise
    // "what happens when a material id used elsewhere is no longer valid",
    // and the documented behaviour is a silent, deterministic read of slot 0
    // (`DEFAULT_MATERIAL`'s own slot), not an `Err` and not a panic.
    let mut table = MaterialTable::new();
    let metal = table.register_metal();
    let never_registered = table.len() as u16 + 41; // guaranteed never assigned

    let fallback = table.combine(never_registered, metal);
    let via_default = table.combine(DEFAULT_MATERIAL, metal);
    assert_eq!(fallback, via_default, "{fallback:?} vs {via_default:?}");

    assert_eq!(
        table.get(never_registered).id,
        table.get(DEFAULT_MATERIAL).id
    );
}

#[test]
fn negative_and_greater_than_one_friction_coefficients_are_accepted_as_is() {
    let mut table = MaterialTable::new();
    let wild = table.register(PhysicsMaterial::new(0, r(-1, 10), r(15, 10)));

    // No clamping anywhere in material.rs: the stored values are exactly
    // what was passed in, negative and >1 alike.
    let m = table.get(wild);
    assert_eq!(m.dynamic_friction, r(-1, 10));
    assert_eq!(m.static_friction, r(-1, 10));
    assert_eq!(m.restitution, r(15, 10));

    // combine() does not special-case out-of-range values either: plain
    // average against DEFAULT_MATERIAL (friction 0.5, restitution 0.3).
    let combined = table.combine(wild, DEFAULT_MATERIAL);
    assert!(close(combined.friction, r(2, 10)), "{combined:?}"); // (-0.1+0.5)/2 = 0.2
    assert!(close(combined.restitution, r(9, 10)), "{combined:?}"); // (1.5+0.3)/2 = 0.9
}

#[test]
fn extreme_multiply_combine_wraps_to_exactly_zero_without_panicking() {
    // x = 2^32 (raw 128-bit integer representation: hi=2^32, lo=0, so the
    // integer `x` is stored as `2^32 << 64 = 2^96`). A wildly out-of-range
    // "friction coefficient" by five orders of magnitude.
    let x = Fix128::from_int(1_i64 << 32);

    // Hand derivation (no call into Fix128::Mul or CombineRule::apply):
    // true 256-bit product of the raw integers is (2^96)^2 = 2^192; a
    // fixed-point multiply divides by 2^64 (shift right 64: 2^192 -> 2^128)
    // and keeps only the low 128 bits (2^128 mod 2^128 = 0). So the product
    // must be exactly Fix128::ZERO.
    let result = catch_unwind(AssertUnwindSafe(|| CombineRule::Multiply.apply(x, x)));
    assert!(
        result.is_ok(),
        "Fix128::Mul is documented wrapping arithmetic, never panics"
    );
    assert_eq!(
        result.unwrap(),
        Fix128::ZERO,
        "2^192 >> 64 mod 2^128 must be exactly zero"
    );

    // A second, merely "large but still in-range" extreme case with an exact
    // non-wrapping closed form, to show Multiply really is `a*b` and not,
    // say, `a*a` or `a+b`: 10^6 * 10^6 = 10^12 exactly (no fractional part,
    // no rounding, well inside Fix128's ±9.2e18 integer range).
    let million = Fix128::from_int(1_000_000);
    let trillion = CombineRule::Multiply.apply(million, million);
    assert_eq!(trillion, Fix128::from_int(1_000_000_000_000));
}
