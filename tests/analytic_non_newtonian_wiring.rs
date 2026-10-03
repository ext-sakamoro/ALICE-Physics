//! Wiring + degenerate-input oracle for `non_newtonian` (power-law / Carreau /
//! Bingham / Herschel-Bulkley rheology models).
//!
//! `tests/engineering_oracles_fluid.rs` already pins the closed-form flow
//! curves (Bingham / Herschel-Bulkley / power-law / Carreau, integer and
//! fractional index) against Chhabra & Richardson and Bird-Armstrong-Hassager
//! — this file does not repeat that. It focuses on the degenerate-input
//! behaviour the module's constitutive-law tests do not exercise: zero shear
//! rate (what each model actually returns at `gamma_dot <= 0`, which is *not*
//! the naive physical answer for the yield-stress models), zero/negative
//! consistency index (there is no validation anywhere in this module —
//! every field is `pub` and the only guard is `gamma_dot <= 0`), extreme
//! shear rate (the whole crate is `Fix128` wrapping arithmetic, so "no
//! panic" is a documented property, not an accident), and the
//! `flows_under_stress` boundary (`applied_stress_pa == yield_stress`).
//!
//! Expected values below are derived by hand from the formulas in
//! `src/non_newtonian.rs`'s module doc, not by calling the functions under
//! test.

#![cfg(feature = "std")]

use alice_physics::math::Fix128;
use alice_physics::non_newtonian::{Bingham, Carreau, HerschelBulkley, PowerLaw};
use std::panic;

fn fix_close(a: Fix128, b: Fix128) -> bool {
    let d = if a > b { a - b } else { b - a };
    d <= Fix128::from_ratio(1, 1_000_000)
}

// ============================================================================
// Zero shear rate: every `stress` guards on `shear_rate_per_s <= ZERO` and
// returns ZERO unconditionally — including for the two yield-stress models
// (Bingham / Herschel-Bulkley), where a naive physical reading of
// `tau = tau_y + ...` at `gamma_dot = 0` would expect `tau_y` (a resting
// yield-stress fluid still holds stress up to tau_y). The module doc commits
// to the discretized "no flow => no stress" contract instead
// (`non_newtonian.rs:209-216`, `:243-252`); this pins that choice exactly.
// ============================================================================

#[test]
fn zero_shear_rate_returns_zero_not_yield_stress() {
    let mud = Bingham {
        yield_stress: Fix128::from_int(10),
        plastic_viscosity: Fix128::from_int(1),
    };
    assert_eq!(
        mud.stress(Fix128::ZERO),
        Fix128::ZERO,
        "Bingham::stress(0) is ZERO, not yield_stress=10"
    );
    // Also true for a *negative* shear rate (same guard, `<=`).
    assert_eq!(mud.stress(Fix128::from_int(-5)), Fix128::ZERO);

    let hb = HerschelBulkley {
        yield_stress: Fix128::from_int(10),
        k: Fix128::from_int(2),
        n_int: 2,
    };
    assert_eq!(
        hb.stress(Fix128::ZERO),
        Fix128::ZERO,
        "HerschelBulkley::stress(0) is ZERO, not yield_stress=10"
    );

    // Power-law (all three presets) is ZERO at rest too, consistent with its
    // own `tau = K * gamma_dot^n` formula (n != 0 => 0^n = 0), so this one
    // agrees with physical intuition unlike the yield-stress pair above.
    for preset in [
        PowerLaw::newtonian(Fix128::from_int(2)),
        PowerLaw::shear_thickening(Fix128::from_int(1), 2),
        PowerLaw::shear_thinning(Fix128::from_int(1), 2),
    ] {
        assert_eq!(preset.stress(Fix128::ZERO), Fix128::ZERO);
    }
}

/// For the shear-thinning preset (`tau = K*gamma_dot^(1/n)`, flow index
/// `1/n < 1`), the apparent viscosity `eta = K*gamma_dot^(1/n - 1)` is a
/// true mathematical singularity as `gamma_dot -> 0+` (exponent `1/n - 1 <
/// 0`). `apparent_viscosity` never reaches that branch: its own
/// `shear_rate_per_s <= ZERO` guard (`non_newtonian.rs:119-121`) fires first
/// and returns ZERO. Pin that the guard wins over the singularity.
#[test]
fn shear_thinning_zero_rate_apparent_viscosity_guarded_not_singular() {
    let mud = PowerLaw::shear_thinning(Fix128::from_int(2), 2);
    let eta = mud.apparent_viscosity(Fix128::ZERO);
    assert_eq!(
        eta,
        Fix128::ZERO,
        "guard wins: apparent_viscosity(0) is ZERO, not +inf"
    );
}

// ============================================================================
// Zero / negative consistency index K: no constructor or field validates it
// (every field is `pub`, and `PowerLaw::shear_thickening` /
// `shear_thinning` only clamp `n_int`, never `k`). Pin the actual,
// no-validation behaviour: K=0 collapses stress to ZERO at every shear
// rate, and a negative K negates the (otherwise identical) positive-K
// stress exactly, with no Err and no panic.
// ============================================================================

#[test]
fn zero_consistency_index_collapses_stress_to_zero() {
    let zero_k = PowerLaw {
        k: Fix128::ZERO,
        n_int: 2,
        thinning: false,
    };
    for g in [1i64, 5, 1000] {
        assert_eq!(zero_k.stress(Fix128::from_int(g)), Fix128::ZERO);
    }

    let hb_zero_k = HerschelBulkley {
        yield_stress: Fix128::from_int(5),
        k: Fix128::ZERO,
        n_int: 2,
    };
    // K=0 does not remove the yield term: tau = yield_stress + 0 = yield_stress.
    assert_eq!(hb_zero_k.stress(Fix128::from_int(10)), Fix128::from_int(5));
}

#[test]
fn negative_consistency_index_negates_stress_no_panic_no_err() {
    let pos = PowerLaw {
        k: Fix128::from_int(3),
        n_int: 2,
        thinning: false,
    };
    let neg = PowerLaw {
        k: Fix128::from_int(-3),
        n_int: 2,
        thinning: false,
    };
    // Integer K and integer gamma_dot keep every intermediate product exact
    // (no fractional remainder), so Mul's -inf-rounding asymmetry
    // (documented on `Fix128::Div`, math.rs:602-608) does not apply here and
    // the negation is exact, not approximate.
    let g = Fix128::from_int(7);
    assert_eq!(neg.stress(g), -pos.stress(g));
    assert!(neg.stress(g).is_negative());

    // No validation path exists anywhere in the module: constructing and
    // evaluating with a negative K neither panics nor has an Err/Option
    // return to check.
    let result = panic::catch_unwind(|| neg.stress(g));
    assert!(
        result.is_ok(),
        "negative consistency index must not panic (no validation exists)"
    );
}

// ============================================================================
// Extreme shear rate: every arithmetic op on Fix128 is `wrapping_*`
// (math.rs:381-456, documented "no panic, no NaN, saturation" on `Mul`).
// This is a crate-wide deterministic-by-design contract, not something this
// module adds — pin that the non_newtonian evaluators inherit it: extreme
// inputs never panic, even though the numeric result silently wraps and is
// not asserted to be physically meaningful.
// ============================================================================

#[test]
fn extreme_shear_rate_never_panics() {
    let huge = Fix128::from_raw(i64::MAX, u64::MAX);

    let thick = PowerLaw::shear_thickening(Fix128::from_int(1), 2);
    assert!(panic::catch_unwind(|| thick.stress(huge)).is_ok());
    assert!(panic::catch_unwind(|| thick.apparent_viscosity(huge)).is_ok());

    let thin = PowerLaw::shear_thinning(Fix128::from_int(2), 2);
    assert!(panic::catch_unwind(|| thin.stress(huge)).is_ok());

    let melt = Carreau {
        eta_zero: Fix128::from_int(1000),
        eta_inf: Fix128::from_int(10),
        lambda: Fix128::from_ratio(1, 10),
        half_exponent: -1,
    };
    assert!(panic::catch_unwind(|| melt.viscosity(huge)).is_ok());
    assert!(
        panic::catch_unwind(|| melt.viscosity_with_index(huge, Fix128::from_ratio(2, 5))).is_ok()
    );

    let mud = Bingham {
        yield_stress: Fix128::from_int(10),
        plastic_viscosity: Fix128::from_int(1),
    };
    assert!(panic::catch_unwind(|| mud.stress(huge)).is_ok());
    assert!(panic::catch_unwind(|| mud.flows_under_stress(huge)).is_ok());

    let hb = HerschelBulkley {
        yield_stress: Fix128::from_int(5),
        k: Fix128::from_int(1),
        n_int: 2,
    };
    assert!(panic::catch_unwind(|| hb.stress(huge)).is_ok());
}

// ============================================================================
// flows_under_stress boundary: the guard is a strict `>`
// (non_newtonian.rs:220-222), so applied_stress exactly equal to the yield
// stress must NOT flow.
// ============================================================================

#[test]
fn flows_under_stress_boundary_is_strict_greater_than() {
    let mud = Bingham {
        yield_stress: Fix128::from_int(10),
        plastic_viscosity: Fix128::from_int(1),
    };
    assert!(
        !mud.flows_under_stress(Fix128::from_int(10)),
        "applied stress == yield stress must not flow (strict >, not >=)"
    );
    // one raw unit (2^-64) above yield must flow.
    let just_above = Fix128::from_int(10) + Fix128::from_raw(0, 1);
    assert!(mud.flows_under_stress(just_above));
}

// ============================================================================
// New-argument scene: PowerLaw's n_int clamp (`.max(2)`) on both
// shear_thickening and shear_thinning. A caller passing n_int < 2 (the
// non-default scene) must get the clamped n=2 curve, not a literal n=1
// curve — pins that the clamp is load-bearing, not merely documentation.
// ============================================================================

#[test]
fn n_int_below_two_is_clamped_not_passed_through() {
    let requested_one = PowerLaw::shear_thickening(Fix128::from_int(1), 1);
    let explicit_two = PowerLaw::shear_thickening(Fix128::from_int(1), 2);
    assert_eq!(requested_one.n_int, 2);
    for g in [1i64, 6, 50] {
        assert_eq!(
            requested_one.stress(Fix128::from_int(g)),
            explicit_two.stress(Fix128::from_int(g))
        );
    }

    let thinning_requested_one = PowerLaw::shear_thinning(Fix128::from_int(2), 1);
    let thinning_explicit_two = PowerLaw::shear_thinning(Fix128::from_int(2), 2);
    assert_eq!(thinning_requested_one.n_int, 2);
    for g in [4i64, 9, 16] {
        assert!(fix_close(
            thinning_requested_one.stress(Fix128::from_int(g)),
            thinning_explicit_two.stress(Fix128::from_int(g))
        ));
    }
}
