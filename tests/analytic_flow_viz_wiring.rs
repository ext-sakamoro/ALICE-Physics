//! Oracles for the production entry points of `alice_physics::flow_viz`
//! driven by `examples/flow_visualization.rs`: `FlowArrow`,
//! `FlowVizConfig`, `generate_flow_arrows`, and `generate_streamlines`.
//!
//! # What this file is and is not
//!
//! The module's own `#[cfg(test)]` block (in `src/flow_viz.rs`) already
//! covers: an empty-fluid grid produces no arrows, a uniform field
//! produces arrows that merely point in the `+X` half-space (sign check,
//! not exact), magnitude scaling by `arrow_scale` (lower-bound check, not
//! exact), empty-fluid and zero-`dt` streamlines degenerate to the seed
//! point, a single moving particle drags the streamline in `+X`,
//! multiple seeds each produce a line, `interpolate_velocity` with no
//! neighbors returns zero, and a flow arrow's direction is a unit
//! vector. None of that is repeated here. What this file adds:
//!
//! * `generate_flow_arrows` on a uniform field with an **exact** (not
//!   sign-only) direction and magnitude oracle,
//! * `generate_flow_arrows` and `generate_streamlines` on an
//!   **all-zero-velocity** fluid: both must degenerate gracefully (empty
//!   arrows, a streamline that is just the seed) without panicking,
//! * `generate_streamlines` termination boundaries: `steps == 0`, and a
//!   field-boundary exit where the traced point leaves every fluid
//!   particle's influence radius partway through the requested step
//!   count (the walk must stop exactly there, not panic and not
//!   continue past it), and
//! * extreme-magnitude velocities, checked against the same exact
//!   single-particle closed form used for the ordinary-magnitude case,
//!   confirming nothing overflows or silently loses precision at a
//!   magnitude far outside typical simulation scales.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "std")]

use alice_physics::flow_viz::{
    generate_flow_arrows, generate_streamlines, FlowArrow, FlowVizConfig,
};
use alice_physics::math::{Fix128, Vec3Fix};

fn basic_config() -> FlowVizConfig {
    FlowVizConfig {
        grid_resolution: 1,
        bounds_min: Vec3Fix::from_int(-10, -10, -10),
        bounds_max: Vec3Fix::from_int(10, 10, 10),
        arrow_scale: Fix128::ONE,
    }
}

// ============================================================================
// FlowArrow / FlowVizConfig -- struct literal round-trip
// ============================================================================

#[test]
fn flow_arrow_fields_round_trip() {
    let arrow = FlowArrow {
        position: Vec3Fix::from_int(-4, 5, 6),
        direction: Vec3Fix::UNIT_Y,
        magnitude: Fix128::from_ratio(3, 2),
    };
    assert_eq!(arrow.position, Vec3Fix::from_int(-4, 5, 6));
    assert_eq!(arrow.direction, Vec3Fix::UNIT_Y);
    assert_eq!(arrow.magnitude, Fix128::from_ratio(3, 2));
}

#[test]
fn flow_viz_config_fields_round_trip() {
    let config = FlowVizConfig {
        grid_resolution: 8,
        bounds_min: Vec3Fix::from_int(-4, -4, -4),
        bounds_max: Vec3Fix::from_int(4, 4, 4),
        arrow_scale: Fix128::from_int(3),
    };
    assert_eq!(config.grid_resolution, 8);
    assert_eq!(config.bounds_min, Vec3Fix::from_int(-4, -4, -4));
    assert_eq!(config.bounds_max, Vec3Fix::from_int(4, 4, 4));
    assert_eq!(config.arrow_scale, Fix128::from_int(3));
}

// ============================================================================
// generate_flow_arrows -- exact uniform-field direction/magnitude
// ============================================================================

/// A single fluid particle inside a one-cell grid spanning the whole
/// bounding volume: `generate_flow_arrows`'s neighbor search radius is
/// the (huge) cell size, so this one particle is the *only* contributor
/// to the one cell's average, with count == 1. Division by exactly
/// `Fix128::ONE` is the identity, so the arrow's `direction` /
/// `magnitude` must equal the particle's own velocity, not merely agree
/// in sign with it.
#[test]
fn flow_arrows_uniform_field_exact_direction_and_magnitude() {
    let config = basic_config();
    let velocity = Vec3Fix::from_int(3, 4, 0); // length 5, exact
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![velocity];

    let arrows = generate_flow_arrows(&positions, &velocities, &config);
    assert_eq!(arrows.len(), 1);

    let expected_direction = velocity.normalize();
    let expected_magnitude = velocity.length() * config.arrow_scale;
    assert_eq!(arrows[0].direction, expected_direction);
    assert_eq!(arrows[0].magnitude, expected_magnitude);
    // Independent closed form: |(3,4,0)| == 5 exactly.
    assert_eq!(arrows[0].magnitude, Fix128::from_int(5));
}

/// Two particles with identical velocity: the average is that velocity
/// exactly, regardless of how many particles contributed, since
/// `sum(v) / count == v` exactly when every summed term is the same
/// value scaled by `Fix128::ONE` operations (addition is exact, and
/// `count` is a small integer `Fix128` value with no fractional part).
#[test]
fn flow_arrows_uniform_field_multiple_identical_particles_exact() {
    let config = basic_config();
    let velocity = Vec3Fix::from_int(0, 0, 7);
    let positions = vec![
        Vec3Fix::from_int(1, 0, 0),
        Vec3Fix::from_int(-1, 0, 0),
        Vec3Fix::from_int(0, 1, 0),
    ];
    let velocities = vec![velocity, velocity, velocity];

    let arrows = generate_flow_arrows(&positions, &velocities, &config);
    assert_eq!(arrows.len(), 1);
    assert_eq!(arrows[0].direction, Vec3Fix::UNIT_Z);
    assert_eq!(arrows[0].magnitude, Fix128::from_int(7));
}

// ============================================================================
// generate_flow_arrows / generate_streamlines -- all-zero-velocity fluid
// ============================================================================

/// Every fluid particle has zero velocity: every grid cell's averaged
/// velocity is `Vec3Fix::ZERO`, whose `magnitude.is_zero()` is true, so
/// `generate_flow_arrows` must skip every cell -- the result is empty,
/// not a list of zero-magnitude arrows, and the call must not panic.
#[test]
fn flow_arrows_all_zero_velocity_is_empty_no_panic() {
    let config = FlowVizConfig {
        grid_resolution: 3,
        ..basic_config()
    };
    let positions = vec![
        Vec3Fix::from_int(1, 1, 1),
        Vec3Fix::from_int(-1, -1, -1),
        Vec3Fix::from_int(2, -2, 0),
    ];
    let velocities = vec![Vec3Fix::ZERO; positions.len()];

    let result =
        std::panic::catch_unwind(|| generate_flow_arrows(&positions, &velocities, &config));
    let arrows = result
        .expect("[flow_viz] generate_flow_arrows must not panic on an all-zero-velocity fluid");
    assert!(arrows.is_empty());
}

/// Every fluid particle has zero velocity: `interpolate_velocity` at the
/// seed returns `Vec3Fix::ZERO` (a weighted average of only zero
/// vectors), so `generate_streamlines`'s `vel.length_squared().is_zero()`
/// check must break on the very first iteration -- the traced line is
/// exactly the one seed point, regardless of the requested step count,
/// and the call must not panic.
#[test]
fn streamlines_all_zero_velocity_degenerates_to_seed_only_no_panic() {
    let positions = vec![Vec3Fix::ZERO, Vec3Fix::from_int(0, 0, 1)];
    let velocities = vec![Vec3Fix::ZERO; positions.len()];
    let seeds = vec![Vec3Fix::ZERO];

    let result = std::panic::catch_unwind(|| {
        generate_streamlines(
            &positions,
            &velocities,
            &seeds,
            50,
            Fix128::from_ratio(1, 10),
        )
    });
    let lines = result
        .expect("[flow_viz] generate_streamlines must not panic on an all-zero-velocity fluid");
    assert_eq!(lines.len(), 1);
    assert_eq!(lines.line(0).len(), 1);
    assert_eq!(lines.line(0)[0], Vec3Fix::ZERO);
}

// ============================================================================
// generate_streamlines -- termination boundaries
// ============================================================================

/// `steps == 0`: the inner `for _ in 0..0` loop body never runs, so the
/// line is exactly the seed point -- the `steps == 0` boundary itself,
/// not an approximation of a short trace.
#[test]
fn streamlines_zero_steps_boundary_returns_seed_only() {
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![Vec3Fix::from_int(1, 0, 0)];
    let seeds = vec![Vec3Fix::from_int(5, 5, 5)];

    let lines = generate_streamlines(&positions, &velocities, &seeds, 0, Fix128::ONE);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines.line(0).len(), 1);
    assert_eq!(lines.line(0)[0], Vec3Fix::from_int(5, 5, 5));
}

/// A single fluid particle at the origin with velocity `(1,0,0)` and
/// `dt = 1/2`: the exact closed form (worked out below) crosses the
/// hardcoded influence radius (`Fix128::ONE`) exactly at the third
/// sample, so the walk must stop there even though `steps` asks for many
/// more -- a field-boundary exit, not a max-steps exit.
///
/// Step 0: `pos = (0,0,0)`, `dist = 0 < 1` -> `vel = (1,0,0)` (the only
/// particle, weight `1 - 0/1 = 1`, division by that weight is the
/// identity) -> `pos = (0,0,0) + (1,0,0)*0.5 = (0.5,0,0)`.
/// Step 1: `pos = (0.5,0,0)`, `dist = 0.5 < 1` -> same single particle,
/// weight `1 - 0.5/1 = 0.5` (a power of two, so the later
/// multiply-by-weight/divide-by-weight round trip loses no precision)
/// -> `vel = (1,0,0)` -> `pos = (0.5,0,0) + (1,0,0)*0.5 = (1,0,0)`.
/// Step 2: `pos = (1,0,0)`, `dist_sq = 1 >= radius_sq = 1` -> the
/// particle is excluded (strict `<` in `interpolate_velocity`), so
/// `total_weight` stays zero and the returned velocity is
/// `Vec3Fix::ZERO` -> `vel.length_squared().is_zero()` breaks the loop.
///
/// Final line: `[(0,0,0), (0.5,0,0), (1,0,0)]` -- 3 points, for a
/// `steps` far larger than that.
#[test]
fn streamlines_field_boundary_exit_terminates_before_max_steps() {
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![Vec3Fix::from_int(1, 0, 0)];
    let seeds = vec![Vec3Fix::ZERO];
    let dt = Fix128::from_ratio(1, 2);

    let lines = generate_streamlines(&positions, &velocities, &seeds, 20, dt);
    assert_eq!(lines.len(), 1);
    let line = lines.line(0);
    assert_eq!(
        line.len(),
        3,
        "expected the walk to exit the particle's influence radius after 2 steps, got {line:?}"
    );
    assert_eq!(line[0], Vec3Fix::ZERO);
    assert_eq!(line[1], Vec3Fix::from_int(1, 0, 0) * dt);
    assert_eq!(line[2], Vec3Fix::from_int(1, 0, 0));
}

/// Same field-boundary exit, but requesting fewer steps than the walk
/// would need to reach the boundary: the max-steps limit must win
/// (every sampled velocity along the way is non-zero), producing the
/// full `steps + 1` points with no early break.
#[test]
fn streamlines_max_steps_wins_when_boundary_not_reached() {
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![Vec3Fix::from_int(1, 0, 0)];
    let seeds = vec![Vec3Fix::ZERO];
    let dt = Fix128::from_ratio(1, 2);

    let lines = generate_streamlines(&positions, &velocities, &seeds, 1, dt);
    assert_eq!(lines.len(), 1);
    let line = lines.line(0);
    assert_eq!(line.len(), 2, "requested 1 step, boundary not yet reached");
    assert_eq!(line[0], Vec3Fix::ZERO);
    assert_eq!(line[1], Vec3Fix::from_int(1, 0, 0) * dt);
}

// ============================================================================
// Extreme-magnitude cases
// ============================================================================

/// A velocity magnitude far outside any physically ordinary simulation
/// scale (`10^9` on one axis), still comfortably inside `Fix128`'s
/// representable range (`length_squared` here is `10^18`, versus a
/// range of `+-9.2 * 10^18`, so no wraparound occurs). The same
/// single-particle exact closed form as the ordinary-magnitude test
/// above must still hold bit-for-bit at this scale.
#[test]
fn flow_arrows_extreme_magnitude_exact_and_no_panic() {
    let config = basic_config();
    let big = Fix128::from_int(1_000_000_000);
    let velocity = Vec3Fix::new(big, Fix128::ZERO, Fix128::ZERO);
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![velocity];

    let result =
        std::panic::catch_unwind(|| generate_flow_arrows(&positions, &velocities, &config));
    let arrows =
        result.expect("[flow_viz] generate_flow_arrows must not panic at extreme magnitude");
    assert_eq!(arrows.len(), 1);
    assert_eq!(arrows[0].direction, Vec3Fix::UNIT_X);
    // |velocity| == 10^9 exactly, a perfect square's root.
    assert_eq!(arrows[0].magnitude, big);
}

/// A streamline walking under an extreme-magnitude uniform field: the
/// sole particle is fixed at the seed, so (same reasoning as the
/// field-boundary test) only the very first sampled velocity is
/// non-zero before the traced point leaves the particle's influence
/// radius -- after exactly one step here, since even `dt`'s smallest
/// representable positive step times a `10^9`-scale velocity vastly
/// exceeds the unit influence radius. Must not panic or produce NaN-like
/// runaway values; the walk must simply stop at 2 points.
#[test]
fn streamlines_extreme_magnitude_terminates_immediately_no_panic() {
    let big = Fix128::from_int(1_000_000_000);
    let positions = vec![Vec3Fix::ZERO];
    let velocities = vec![Vec3Fix::new(big, Fix128::ZERO, Fix128::ZERO)];
    let seeds = vec![Vec3Fix::ZERO];
    let dt = Fix128::ONE;

    let result =
        std::panic::catch_unwind(|| generate_streamlines(&positions, &velocities, &seeds, 10, dt));
    let lines =
        result.expect("[flow_viz] generate_streamlines must not panic at extreme magnitude");
    assert_eq!(lines.len(), 1);
    let line = lines.line(0);
    assert_eq!(
        line.len(),
        2,
        "expected exactly one step before the extreme jump leaves the influence radius"
    );
    assert_eq!(line[0], Vec3Fix::ZERO);
    assert_eq!(line[1], Vec3Fix::new(big, Fix128::ZERO, Fix128::ZERO));
}
