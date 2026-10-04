//! Audit oracles for `fatigue`: the documented polymer defaults of
//! `SnCurve::from_fdm_material` to the last unit of `Fix128`, and the Basquin
//! life of the resulting curve at twice its endurance stress.
//!
//! Doc: `S_e = 0.3 UTS`, `N_e = 1e6`, `m = 5`, and the ultimate strength is
//! the material's. Basquin: `N(S) = N_e (S_e / S)^m`; at `S = 2 S_e` that is
//! `1e6 / 32 = 31250` cycles exactly, so 31250 cycles are a damage of one and
//! one cycle is `1 / 31250`.
#![cfg(feature = "std")]
#![allow(clippy::disallowed_methods)]

use alice_physics::fatigue::{miner_damage, SnCurve};
use alice_physics::filament_db::MaterialProperties;
use alice_physics::math::Fix128;

fn presets() -> [MaterialProperties; 8] {
    [
        MaterialProperties::pla(),
        MaterialProperties::petg(),
        MaterialProperties::abs(),
        MaterialProperties::pc(),
        MaterialProperties::tpu(),
        MaterialProperties::nylon(),
        MaterialProperties::cf_nylon(),
        MaterialProperties::peek(),
    ]
}

/// `S_e` is `0.3 UTS` to the rounding of one product: `0.3` itself is within
/// `2^-64` of its `Fix128` value and the product truncates once more, so
/// `|10 S_e - 3 UTS| <= 10 (UTS + 1) 2^-64`. Multiplying by the integers 10
/// and 3 is exact in `Fix128`, so the check is made in `Fix128` and resolves
/// a single unit in the last place.
#[test]
fn polymer_defaults_are_the_documented_constants_to_the_last_unit() {
    for m in presets() {
        let c = SnCurve::from_fdm_material(&m);
        assert_eq!(c.ultimate_tensile_mpa, m.tensile_strength_mpa, "{}", m.name);
        assert_eq!(c.endurance_cycles, 1_000_000, "{}", m.name);
        assert_eq!(c.fatigue_exponent_m, 5, "{}", m.name);

        let uts = m.tensile_strength_mpa;
        let diff =
            (c.endurance_stress_mpa * Fix128::from_int(10) - uts * Fix128::from_int(3)).abs();
        let uts_ceil = uts.to_f64().ceil() as u64;
        let bound = Fix128::from_raw(0, 10 * (uts_ceil + 1));
        assert!(
            diff <= bound,
            "{}: |10 S_e - 3 UTS| = {:?} raw units, bound {:?}",
            m.name,
            diff,
            bound
        );
    }
}

/// The default curve at twice its endurance stress: `(S_e / 2 S_e)^5 = 1/32`
/// is exact, and `1e6 / 32 = 31250` is an integer, so the damage of 31250
/// cycles is one and of one cycle `1 / 31250`, both exactly.
#[test]
fn default_curve_life_at_twice_the_endurance_stress_is_31250_cycles() {
    for m in presets() {
        let c = SnCurve::from_fdm_material(&m);
        let s = c.endurance_stress_mpa + c.endurance_stress_mpa;
        assert_eq!(
            miner_damage(&[(s, 31_250)], &c),
            Fix128::ONE,
            "{}: 31250 cycles",
            m.name
        );
        assert_eq!(
            miner_damage(&[(s, 1)], &c),
            Fix128::ONE / Fix128::from_int(31_250),
            "{}: one cycle",
            m.name
        );
    }
}
