//! Oracles for the production entry points of `alice_physics::fsi_advanced`
//! driven by `examples/fsi_advanced_forces.rs`: `SolidSample`,
//! `aggregate_forces`, `buoyancy_force`, `drag_force`, `react_back_pressure`.
//!
//! # What this file is and is not
//!
//! `drag_force` / `buoyancy_force` / `aggregate_forces` / `react_back_pressure`
//! / `SolidSample` already have closed-form physical oracles in
//! `tests/engineering_oracles_fluid.rs::fsi_advanced_drag_buoyancy_terminal_velocity_and_reaction`
//! (White's quadratic-drag terminal velocity, Archimedes buoyancy, a
//! two-sample single-axis dumbbell torque, Newton's third law) and in
//! `tests/fsi_advanced_sub_iteration.rs` (`drag_force` / `react_back_pressure`
//! driven through a closed-form sub-iteration contraction ratio) — those are
//! not repeated here. What those files do **not** cover, and this file does:
//!
//! * `aggregate_forces` against a *fully 3D* two-sample layout (nonzero x, y
//!   **and** z on both position and velocity, and a reference point off the
//!   origin), unlike the existing dumbbell scene which keeps every position
//!   and force on a single axis — a torque-component swap or a dropped
//!   cross-product term can hide behind a single-axis scene but not here,
//! * a direct dispatcher cross-check: `aggregate_forces` of one sample must
//!   equal `drag_force(..) + buoyancy_force(..)` of that same sample called
//!   directly, so a version of `aggregate_forces` that silently drops the
//!   buoyancy term (or halves the drag term, or similar) is observable,
//! * `drag_force` zeroed by `area_m2 = 0` independent of an arbitrarily large
//!   relative velocity, and `buoyancy_force` zeroed by `fluid_density = 0`
//!   independent of an arbitrarily large volume/gravity — neither existing
//!   file zeros one factor while the others stay large,
//! * `aggregate_forces` on an empty sample slice, and
//! * two degenerate/extreme-magnitude inputs neither existing file touches:
//!   a relative velocity whose square overflows `Fix128`'s ±2⁶³ integer
//!   range (`drag_force`), a buoyancy product that overflows the same range
//!   (`buoyancy_force`), and `Fix128`'s single self-negating value under
//!   `react_back_pressure`'s `ZERO - f` reaction.
//!
//! # Degenerate input
//!
//! `drag_force`: `sample.velocity.x = Fix128::from_int(i64::MAX)`,
//! `fluid_velocity = 0`. `v_rel.x * v_rel.x` overflows; `Fix128::Mul` is
//! unconditionally wrapping (never panics, see `rules/analytic-oracle-tests.md`
//! and the module doc of `src/math.rs`), and for two pure integers (zero
//! fractional half) the wrapping multiply is bit-for-bit `i64::wrapping_mul`
//! on the integer halves (`Mul`'s cross terms `hi*lo`/`lo*hi`/`(lo*lo)>>64`
//! are all zero when both operands have `lo = 0`). `i64::MAX.wrapping_mul(
//! i64::MAX)` wraps to `1` (the same fact `tests/analytic_laminate_failure_wiring.rs`
//! derives: `(2⁶³−1)² = 2¹²⁶ − 2⁶⁴ + 1 ≡ 1 (mod 2⁶⁴)`), so `mag_sq` is exactly
//! `Fix128::ONE`, not zero (the early-return branch is not taken) and not an
//! astronomically large value. `sqrt(ONE) = ONE` exactly (the isqrt digit
//! recurrence on radicand `1 << 128` returns `1 << 64` exactly). The
//! remaining step, `prefactor * v_rel.x = (-1/2) * Fix128::from_int(i64::MAX)`,
//! does **not** itself overflow (the true product's magnitude is
//! `(2⁶³−1)/2 ≈ 4.611 × 10¹⁸`, inside ±2⁶³), so the final value is the exact,
//! modest `-(2⁶² − 1/2)` — not the enormous value real (non-wrapping)
//! arithmetic on `i64::MAX` relative velocity would suggest.
//!
//! `buoyancy_force`: `fluid_density = Fix128::from_int(i64::MAX)`,
//! `volume_m3 = Fix128::from_int(2)`. `density * volume` is another pure
//! integer multiply (`lo = 0` on both), so it is exactly
//! `Fix128::from_int(i64::MAX.wrapping_mul(2))`. `(2⁶³−1)·2 = 2⁶⁴−2 ≡ −2
//! (mod 2⁶⁴)` as a signed `i64`, so the product wraps to exactly `-2`, and
//! `buoyancy_force` reports a *downward* force where real arithmetic would
//! give an enormous upward one.
//!
//! `react_back_pressure`: `Fix128::from_raw(i64::MIN, 0)` is the most
//! negative representable value (`hi` saturates `i64::MIN`, `lo = 0`).
//! `react_back_pressure`'s reaction is `Fix128::ZERO - f`, i.e. `Sub`, which
//! is bitwise wrapping subtraction on `(hi, lo)` exactly like `Neg`'s
//! two's-complement negation (`src/math.rs`). Two's-complement negation of
//! the minimum representable value is a documented fixed point (there is no
//! representable `+2⁶³`), so `0 - MIN` wraps back to `MIN` itself — the
//! deposited reaction is numerically identical to the input force, not its
//! negative.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]
// The expected-value computations below are closed-form hand derivations,
// not simulation state.
#![allow(clippy::disallowed_methods)]

use alice_physics::fsi_advanced::{
    aggregate_forces, buoyancy_force, drag_force, react_back_pressure, SolidSample,
};
use alice_physics::math::{Fix128, Vec3Fix};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn v(x: i64, y: i64, z: i64) -> Vec3Fix {
    Vec3Fix::from_int(x, y, z)
}

fn sample(position: Vec3Fix, velocity: Vec3Fix, area_m2: i64, volume_m3: i64) -> SolidSample {
    SolidSample {
        position,
        velocity,
        area_m2: Fix128::from_int(area_m2),
        volume_m3: Fix128::from_int(volume_m3),
    }
}

// ============================================================================
// Section 0: SolidSample is a plain data aggregate — constructed directly and
// compared via its derived PartialEq, independent of every force function.
// ============================================================================

#[test]
fn solid_sample_fields_round_trip_through_construction() {
    let s = sample(v(1, 2, 3), v(4, 5, 6), 7, 8);
    assert_eq!(s.position, v(1, 2, 3));
    assert_eq!(s.velocity, v(4, 5, 6));
    assert_eq!(s.area_m2, Fix128::from_int(7));
    assert_eq!(s.volume_m3, Fix128::from_int(8));
    // Clone + Copy must produce an equal, independent value.
    let cloned = s;
    assert_eq!(s, cloned);
}

// ============================================================================
// Section 1: a fully-3D two-sample scene (nonzero x, y, z on both position
// and velocity; reference point off the origin), unlike the existing
// single-axis dumbbell in engineering_oracles_fluid.rs. Hand-derived by the
// documented formulas (module doc of src/fsi_advanced.rs), not by calling
// drag_force/buoyancy_force/aggregate_forces for the expected side.
//
// rho = 1200, Cd = 2, g = 5, uniform fluid velocity (1,1,1).
//
// Sample 1: position (1,2,3), velocity (4,5,1) -> v_rel = (3,4,0), |v_rel|=5
//   F_d1 = -0.5*1200*2*area1(1)*5 * (3,4,0) = -6000*(3,4,0) = (-18000,-24000,0)
//   F_b1 = 1200*volume1(2)*5 = 12000 -> (0,12000,0)
//   total1 = (-18000,-12000,0)
//
// Sample 2: position (2,-1,1), velocity (1,4,5) -> v_rel = (0,3,4), |v_rel|=5
//   F_d2 = -0.5*1200*2*area2(2)*5 * (0,3,4) = -12000*(0,3,4) = (0,-36000,-48000)
//   F_b2 = 1200*volume2(1)*5 = 6000 -> (0,6000,0)
//   total2 = (0,-30000,-48000)
//
// reference_point = (0,1,0):
//   r1 = (1,1,3), r2 = (2,-2,1)
//   tau1 = r1 x F1 = (1*0-3*(-12000), 3*(-18000)-1*0, 1*(-12000)-1*(-18000))
//        = (36000, -54000, 6000)
//   tau2 = r2 x F2 = ((-2)*(-48000)-1*(-30000), 1*0-2*(-48000), 2*(-30000)-(-2)*0)
//        = (126000, 96000, -60000)
//
// net_force  = total1 + total2  = (-18000, -42000, -48000)
// net_torque = tau1 + tau2       = (162000, 42000, -54000)
// ============================================================================

const RHO: i64 = 1200;
const CD: i64 = 2;
const G: i64 = 5;

fn fluid_uniform(_p: Vec3Fix) -> Vec3Fix {
    v(1, 1, 1)
}

fn scene_1() -> (SolidSample, SolidSample) {
    (
        sample(v(1, 2, 3), v(4, 5, 1), 1, 2),
        sample(v(2, -1, 1), v(1, 4, 5), 2, 1),
    )
}

#[test]
fn aggregate_forces_matches_hand_derivation_at_a_fully_3d_layout() {
    let (s1, s2) = scene_1();
    let reference_point = v(0, 1, 0);

    let (net_force, net_torque) = aggregate_forces(
        &[s1, s2],
        fluid_uniform,
        Fix128::from_int(RHO),
        Fix128::from_int(CD),
        Fix128::from_int(G),
        reference_point,
    );

    assert_eq!(net_force, v(-18_000, -42_000, -48_000), "net force");
    assert_eq!(net_torque, v(162_000, 42_000, -54_000), "net torque");
}

#[test]
fn aggregate_forces_of_one_sample_matches_the_direct_drag_plus_buoyancy_call() {
    // The actual dispatcher check: aggregate_forces must combine the same
    // per-sample drag_force + buoyancy_force this file can call directly —
    // not a version with a dropped, halved, or swapped term. Reference point
    // == the sample's own position, so the torque side is trivially zero and
    // only the net-force combination is under test here.
    let (s1, _s2) = scene_1();

    let direct_drag = drag_force(
        &s1,
        fluid_uniform(s1.position),
        Fix128::from_int(RHO),
        Fix128::from_int(CD),
    );
    let direct_buoyancy = buoyancy_force(&s1, Fix128::from_int(RHO), Fix128::from_int(G));
    let direct_total = Vec3Fix::new(
        direct_drag.x + direct_buoyancy.x,
        direct_drag.y + direct_buoyancy.y,
        direct_drag.z + direct_buoyancy.z,
    );

    assert_eq!(
        direct_drag,
        v(-18_000, -24_000, 0),
        "direct drag_force, sample 1"
    );
    assert_eq!(
        direct_buoyancy,
        v(0, 12_000, 0),
        "direct buoyancy_force, sample 1"
    );

    let (aggregate_net, aggregate_torque) = aggregate_forces(
        &[s1],
        fluid_uniform,
        Fix128::from_int(RHO),
        Fix128::from_int(CD),
        Fix128::from_int(G),
        s1.position,
    );

    assert_eq!(
        aggregate_net, direct_total,
        "aggregate_forces of a single sample must equal drag_force + buoyancy_force called directly"
    );
    assert_eq!(
        aggregate_torque,
        Vec3Fix::ZERO,
        "reference point == sample position -> zero torque"
    );
}

// ============================================================================
// Section 2: aggregate_forces on an empty slice.
// ============================================================================

#[test]
fn aggregate_forces_of_no_samples_is_exactly_zero() {
    let (net_force, net_torque) = aggregate_forces(
        &[],
        fluid_uniform,
        Fix128::from_int(RHO),
        Fix128::from_int(CD),
        Fix128::from_int(G),
        Vec3Fix::ZERO,
    );
    assert_eq!(net_force, Vec3Fix::ZERO);
    assert_eq!(net_torque, Vec3Fix::ZERO);
}

// ============================================================================
// Section 3: drag/buoyancy zeroed by one factor, independent of the others
// being large. Neither existing test file zeros area_m2 or fluid_density
// while the remaining factors stay nonzero and large.
// ============================================================================

#[test]
fn drag_force_is_zero_when_area_is_zero_regardless_of_relative_velocity() {
    let huge = sample(Vec3Fix::ZERO, v(1_000_000, -2_000_000, 3_000_000), 0, 0);
    let f = drag_force(
        &huge,
        Vec3Fix::ZERO,
        Fix128::from_int(1000),
        Fix128::from_int(5),
    );
    assert_eq!(
        f,
        Vec3Fix::ZERO,
        "area_m2 = 0 must zero the drag prefactor exactly"
    );
}

#[test]
fn buoyancy_force_is_zero_when_fluid_density_is_zero_regardless_of_volume_and_gravity() {
    let large_volume = sample(Vec3Fix::ZERO, Vec3Fix::ZERO, 0, 1_000_000);
    let f = buoyancy_force(&large_volume, Fix128::ZERO, Fix128::from_int(1_000_000));
    assert_eq!(
        f,
        Vec3Fix::ZERO,
        "fluid_density = 0 must zero buoyancy exactly"
    );
}

// ============================================================================
// Section 4: degenerate/extreme magnitude — mag_sq overflow in drag_force.
// Derived in the module doc comment above: mag_sq wraps to exactly ONE, and
// the final drag force is the exact, modest -(2^62 - 1/2), not an
// astronomically large value.
// ============================================================================

#[test]
fn drag_force_relative_velocity_squared_overflow_wraps_to_a_modest_value() {
    let extreme = sample(Vec3Fix::ZERO, v(i64::MAX, 0, 0), 1, 0);

    // The wrap itself, independent of drag_force: (i64::MAX)^2 wraps to
    // exactly Fix128::ONE (same identity as
    // tests/analytic_laminate_failure_wiring.rs's extreme-stress case).
    let mag_sq = catch_unwind(AssertUnwindSafe(|| extreme.velocity.x * extreme.velocity.x))
        .expect("Fix128 multiply never panics, wrapping or not");
    assert_eq!(
        mag_sq,
        Fix128::ONE,
        "(i64::MAX)^2 must wrap to exactly Fix128::ONE"
    );

    let f = catch_unwind(AssertUnwindSafe(|| {
        drag_force(&extreme, Vec3Fix::ZERO, Fix128::ONE, Fix128::ONE)
    }))
    .expect("drag_force must not panic on overflowing relative velocity");

    // Expected: prefactor = -1/2 * 1 * 1 * 1 * sqrt(mag_sq=1) = -1/2 exactly;
    // F_d.x = -1/2 * i64::MAX = -(2^62 - 1/2), which does NOT itself overflow
    // (magnitude ~4.611e18, inside Fix128's +-2^63 range).
    let expected_x = Fix128::from_int(-(1i64 << 62)) + Fix128::ONE.half();
    assert_eq!(
        f.x, expected_x,
        "drag force x-component after the mag_sq wrap"
    );
    assert_eq!(f.y, Fix128::ZERO);
    assert_eq!(f.z, Fix128::ZERO);
}

// ============================================================================
// Section 5: degenerate/extreme magnitude — density*volume overflow in
// buoyancy_force. Derived above: the product wraps to exactly -2, giving a
// downward force instead of an enormous upward one.
// ============================================================================

#[test]
fn buoyancy_force_density_volume_overflow_wraps_to_a_downward_force() {
    let extreme = sample(Vec3Fix::ZERO, Vec3Fix::ZERO, 0, 2);

    let f = catch_unwind(AssertUnwindSafe(|| {
        buoyancy_force(&extreme, Fix128::from_int(i64::MAX), Fix128::ONE)
    }))
    .expect("buoyancy_force must not panic on overflowing density*volume");

    assert_eq!(
        f,
        v(0, -2, 0),
        "density*volume must wrap to exactly -2, giving a downward force"
    );
    assert!(
        f.y < Fix128::ZERO,
        "the wrapped buoyancy must point down, not up"
    );
}

// ============================================================================
// Section 6: react_back_pressure's single self-negating value, contrasted
// with an ordinary (non-boundary) negation to show the boundary is genuinely
// special, not a general property of react_back_pressure.
// ============================================================================

#[test]
fn react_back_pressure_negates_an_ordinary_force_normally() {
    let s = sample(v(5, 0, 0), Vec3Fix::ZERO, 0, 0);
    let force = v(7, -11, 13);
    let mut reaction = Vec3Fix::ZERO;
    react_back_pressure(&[s], &[force], |pos, r| {
        assert_eq!(pos, s.position);
        reaction = r;
    });
    assert_eq!(
        reaction,
        v(-7, 11, -13),
        "an ordinary force must be negated exactly"
    );
}

#[test]
fn react_back_pressure_at_the_minimum_representable_value_wraps_to_itself() {
    let min_value = Fix128::from_raw(i64::MIN, 0);
    let s = sample(Vec3Fix::ZERO, Vec3Fix::ZERO, 0, 0);
    let force = Vec3Fix::new(min_value, min_value, min_value);

    let mut reaction = Vec3Fix::ZERO;
    catch_unwind(AssertUnwindSafe(|| {
        react_back_pressure(&[s], &[force], |_pos, r| reaction = r);
    }))
    .expect("react_back_pressure must not panic at the minimum representable value");

    // ZERO - MIN wraps back to MIN on every axis (two's-complement
    // self-negation, derived in the module doc comment above), not to the
    // arithmetic positive MAX+1 that is not representable.
    assert_eq!(
        reaction, force,
        "MIN must react to itself, not to its arithmetic negative"
    );
    assert_eq!(reaction.x, min_value);
    assert_eq!(reaction.y, min_value);
    assert_eq!(reaction.z, min_value);
}
