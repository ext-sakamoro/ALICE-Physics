//! Oracles for the production entry points of `alice_physics::pressure`
//! driven by `examples/pressure_contact_deformation.rs`:
//! `PressureModifier::{apply_pressure_at, apply_impact, pressure_at,
//! deformation_at}`.
//!
//! # What this file is and is not
//!
//! `src/pressure.rs`'s own `#[cfg(test)]` module already checks
//! `apply_impact` produces a positive dent and that `modify_distance`
//! reflects it, and separately checks high pressure eventually yields
//! permanent deformation through repeated `update()` calls -- those are
//! not repeated here. What that module does **not** cover, and this file
//! does:
//!
//! * `apply_pressure_at` / `pressure_at` in isolation, checked against an
//!   exact closed form for the splat weight (not just "is it nonzero"),
//! * `apply_impact`'s unclamped branch (`dent_depth < max_deformation`)
//!   checked against the exact `impulse * deformation_rate` product,
//!   separately from the already-tested clamped branch,
//! * the `radius == 0.0` boundary, where `ScalarField3D::splat`'s own
//!   `dist_sq < radius_sq` test (strict `<`, both sides `0.0`) silently
//!   drops the splat even exactly at its own center -- a non-obvious
//!   degenerate-input behavior neither this module's tests nor
//!   `src/sim_field.rs`'s tests exercise,
//! * the `force == 0.0` / `impulse == 0.0` / `deformation_rate == 0.0`
//!   zero-magnitude boundaries, and
//! * `f32`-overflow extreme magnitudes for both fields: two
//!   `f32::MAX`-force splats at the same point overflow `f32` addition to
//!   `+inf` without panicking, and an `f32::MAX`-impulse impact with an
//!   `f32::MAX` `deformation_rate` overflows the `dent_depth` product to
//!   `+inf` but the subsequent `.min(max_deformation)` clamp (IEEE 754
//!   `minNum`: `min(inf, finite) == finite`) prevents the `inf` from ever
//!   reaching the field.
//!
//! This module uses plain `f32` throughout (`ScalarField3D`, `splat`,
//! `sample`) -- there is no `Fix128` anywhere in `apply_pressure_at`,
//! `apply_impact`, `pressure_at`, or `deformation_at` or their callees.
//! "Extreme magnitude" here therefore means standard IEEE 754 `f32`
//! overflow-to-infinity, not `Fix128`'s documented mod-2^128 wraparound.
//!
//! # Why every query point here lands exactly on a grid node
//!
//! See `examples/pressure_contact_deformation.rs`'s module doc comment
//! for the full derivation. Summary: a `5^3` grid over `[-2,2]^3` has
//! `cell_size == 1.0` on every axis, so world-space `(0,0,0)` is exactly
//! grid node `(2,2,2)`; sampling exactly at a grid node returns the
//! stored value with no interpolation error (`fx == fy == fz == 0.0`),
//! and with `radius == 0.5 < cell_size == 1.0` only that one node is
//! within the splat radius, where the smoothstep weight is exactly
//! `1.0`. None of the expected values below are computed by calling
//! `apply_pressure_at` / `apply_impact` / `pressure_at` /
//! `deformation_at` themselves.

#![cfg(feature = "std")]

use alice_physics::pressure::{PressureConfig, PressureModifier};

/// `5^3` grid over `[-2,2]^3`: `cell_size == (4.0)/(5-1) == 1.0` on every
/// axis, so world-space `(0,0,0)` is exactly grid node `(2,2,2)` and every
/// other node is at grid distance `>= 1.0` from it.
fn fresh_modifier(config: PressureConfig) -> PressureModifier {
    PressureModifier::new(config, 5, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
}

/// `radius` strictly between the center-to-neighbor distance (`1.0`) and
/// `0.0`, so `splat`'s `dist_sq < radius_sq` test includes the center node
/// (`dist == 0.0`) and excludes every neighbor (`dist == 1.0`).
const ISOLATING_RADIUS: f32 = 0.5;

fn assert_exact(got: f32, want: f32, what: &str) {
    assert!(
        got.to_bits() == want.to_bits() || got == want,
        "{what}: got {got}, want {want}"
    );
}

fn assert_close(got: f32, want: f32, tol: f32, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= tol,
        "{what}: got {got}, want {want} (abs err {err:.3e} > {tol:.1e})"
    );
}

// ============================================================================
// Normal cases
// ============================================================================

/// `apply_pressure_at` then `pressure_at` at the exact splat center must
/// return exactly the splatted `force`, independent of
/// `PressureConfig` (the pressure field's splat/sample path does not
/// consult `config` at all -- only `update`'s diffuse/decay/yield step
/// does).
#[test]
fn apply_pressure_at_matches_splat_weight_one_at_grid_node() {
    let mut m = fresh_modifier(PressureConfig::default());
    let force = 37.5_f32;
    m.apply_pressure_at(0.0, 0.0, 0.0, force, ISOLATING_RADIUS);
    assert_exact(
        m.pressure_at(0.0, 0.0, 0.0),
        force,
        "pressure_at(center) after one splat",
    );
}

/// A second, distinct force value and query point (`(1,1,1)` is also an
/// exact grid node: `gx = (1 - -2) * 1.0 = 3.0`), checking the closed form
/// is not special-cased to the origin.
#[test]
fn apply_pressure_at_matches_splat_weight_one_at_non_origin_grid_node() {
    let mut m = fresh_modifier(PressureConfig::default());
    let force = -12.25_f32; // negative force is a valid transient pressure value
    m.apply_pressure_at(1.0, 1.0, 1.0, force, ISOLATING_RADIUS);
    assert_exact(
        m.pressure_at(1.0, 1.0, 1.0),
        force,
        "pressure_at((1,1,1)) after one splat at (1,1,1)",
    );
}

/// `apply_impact`'s unclamped branch: `dent_depth = impulse *
/// deformation_rate = 2.0 * 0.05 = 0.1`, strictly below
/// `max_deformation == 1.0`, so `.min(max_deformation)` is a no-op and the
/// splatted (and sampled-back) value equals the raw product.
#[test]
fn apply_impact_unclamped_matches_impulse_times_rate() {
    let config = PressureConfig::default();
    let mut m = fresh_modifier(config);
    let impulse = 2.0_f32;
    let expected = impulse * config.deformation_rate; // independent closed form, not a call to apply_impact
    m.apply_impact(0.0, 0.0, 0.0, impulse, ISOLATING_RADIUS);
    assert_close(
        m.deformation_at(0.0, 0.0, 0.0),
        expected,
        1e-6,
        "deformation_at(unclamped) == impulse * deformation_rate",
    );
}

/// `apply_impact`'s clamped branch: `dent_depth = 1000.0 * 0.05 = 50.0`,
/// far past `max_deformation == 1.0`, so the splatted value must be
/// exactly `1.0`, not `50.0`.
#[test]
fn apply_impact_clamped_at_max_deformation() {
    let config = PressureConfig::default();
    let mut m = fresh_modifier(config);
    m.apply_impact(0.0, 0.0, 0.0, 1000.0, ISOLATING_RADIUS);
    assert_exact(
        m.deformation_at(0.0, 0.0, 0.0),
        config.max_deformation,
        "deformation_at(clamped) == max_deformation exactly",
    );
}

// ============================================================================
// Boundary cases
// ============================================================================

/// `radius == 0.0`: `ScalarField3D::splat` computes `radius_sq == 0.0`
/// and tests `dist_sq < radius_sq` with strict `<`. At the splat center
/// itself `dist_sq == 0.0`, and `0.0 < 0.0` is `false`, so *no* cell is
/// written -- not even the center. This is the field-level analog of a
/// "zero area" contact patch: the documented behavior is a complete
/// no-op, not a single-point deposit.
#[test]
fn apply_pressure_at_zero_radius_deposits_nothing() {
    let mut m = fresh_modifier(PressureConfig::default());
    m.apply_pressure_at(0.0, 0.0, 0.0, 99.0, 0.0);
    assert_exact(
        m.pressure_at(0.0, 0.0, 0.0),
        0.0,
        "pressure_at(center) after zero-radius splat stays at the field's initial zero",
    );
}

/// Same zero-radius no-op, through `apply_impact` -> `deformation` field.
#[test]
fn apply_impact_zero_radius_deposits_nothing() {
    let mut m = fresh_modifier(PressureConfig::default());
    m.apply_impact(0.0, 0.0, 0.0, 1000.0, 0.0);
    assert_exact(
        m.deformation_at(0.0, 0.0, 0.0),
        0.0,
        "deformation_at(center) after zero-radius impact stays at the field's initial zero",
    );
}

/// `force == 0.0`: the splatted value is `0.0 * weight == 0.0` for every
/// touched cell, so the field is unchanged regardless of `radius`.
#[test]
fn apply_pressure_at_zero_force_is_zero() {
    let mut m = fresh_modifier(PressureConfig::default());
    m.apply_pressure_at(0.0, 0.0, 0.0, 0.0, ISOLATING_RADIUS);
    assert_exact(
        m.pressure_at(0.0, 0.0, 0.0),
        0.0,
        "pressure_at(center) after zero-force splat",
    );
}

/// `impulse == 0.0`: `dent_depth = 0.0 * deformation_rate == 0.0`.
#[test]
fn apply_impact_zero_impulse_is_zero() {
    let mut m = fresh_modifier(PressureConfig::default());
    m.apply_impact(0.0, 0.0, 0.0, 0.0, ISOLATING_RADIUS);
    assert_exact(
        m.deformation_at(0.0, 0.0, 0.0),
        0.0,
        "deformation_at(center) after zero-impulse impact",
    );
}

/// `deformation_rate == 0.0` (an all-rigid, non-deformable config):
/// `dent_depth = impulse * 0.0 == 0.0` for any impulse, including a very
/// large one -- the degenerate config parameter, not the input, drives
/// the zero.
#[test]
fn apply_impact_zero_deformation_rate_never_dents() {
    let config = PressureConfig {
        deformation_rate: 0.0,
        ..PressureConfig::default()
    };
    let mut m = fresh_modifier(config);
    m.apply_impact(0.0, 0.0, 0.0, 500.0, ISOLATING_RADIUS);
    assert_exact(
        m.deformation_at(0.0, 0.0, 0.0),
        0.0,
        "deformation_at(center) with deformation_rate == 0.0",
    );
}

// ============================================================================
// Extreme-magnitude cases (f32 overflow, not Fix128 wraparound -- this
// module has no Fix128 anywhere in these four functions or their callees)
// ============================================================================

/// Two `f32::MAX`-force splats at the same grid node: each contributes
/// `f32::MAX * 1.0 == f32::MAX` (finite, exact), but the second
/// accumulates onto the first via `self.data[idx] += value * weight`,
/// i.e. `f32::MAX + f32::MAX`, which overflows `f32`'s finite range and
/// rounds the *stored grid value* to `f32::INFINITY` per IEEE 754.
///
/// `pressure_at` does not read that stored value directly, though --
/// `ScalarField3D::sample` always runs the general trilinear formula
/// (`(d[hi] - d[lo]).mul_add(f, d[lo])`), even when `fx == fy == fz ==
/// 0.0` lands exactly on a grid node. At that node, `d[lo]` (this cell)
/// is `+inf` and `d[hi]` (the untouched neighbor) is `0.0`, so `d[hi] -
/// d[lo] == -inf`, and `(-inf).mul_add(0.0, inf)` computes `(-inf *
/// 0.0) + inf == NaN + inf == NaN` -- IEEE 754's `0 * inf` is `NaN`,
/// not `0`, so the "multiplying by f == 0.0 drops the interpolation
/// term" shortcut that holds for any two *finite* `d[hi]`/`d[lo]`
/// values does **not** hold once one of them is infinite. The actual,
/// documented-here behavior is therefore `NaN`, not `+inf` -- a real
/// finding from writing this oracle, not an assumption.
#[test]
fn apply_pressure_at_overflow_propagates_nan_through_trilinear_sample_no_panic() {
    let mut m = fresh_modifier(PressureConfig::default());
    m.apply_pressure_at(0.0, 0.0, 0.0, f32::MAX, ISOLATING_RADIUS);
    m.apply_pressure_at(0.0, 0.0, 0.0, f32::MAX, ISOLATING_RADIUS);
    let got = m.pressure_at(0.0, 0.0, 0.0);
    assert!(
        got.is_nan(),
        "pressure_at(center) after two f32::MAX splats must be NaN \
         (0*inf in the trilinear sample's mul_add, not a clean +inf), got {got}"
    );
}

/// `impulse == f32::MAX` with a (deliberately extreme, non-default)
/// `deformation_rate == f32::MAX`: `dent_depth = f32::MAX * f32::MAX`
/// overflows to `f32::INFINITY` (the product's true magnitude, ~1.16e77,
/// is far beyond `f32::MAX`'s ~3.4e38). The subsequent
/// `.min(max_deformation)` call is IEEE 754 `minNum`, which returns the
/// finite operand when compared against `+inf`
/// (`f32::INFINITY.min(1.0) == 1.0`), so the splatted and sampled-back
/// value must be exactly `max_deformation`, not `inf` and not a panic.
#[test]
fn apply_impact_overflow_clamped_to_max_deformation_no_panic() {
    let config = PressureConfig {
        deformation_rate: f32::MAX,
        ..PressureConfig::default()
    };
    let mut m = fresh_modifier(config);
    m.apply_impact(0.0, 0.0, 0.0, f32::MAX, ISOLATING_RADIUS);
    let got = m.deformation_at(0.0, 0.0, 0.0);
    assert_exact(
        got,
        config.max_deformation,
        "deformation_at(center) after f32::MAX impact * f32::MAX rate, clamped by max_deformation",
    );
    assert!(
        got.is_finite(),
        "deformation_at(center) must not be inf/NaN, got {got}"
    );
}
