//! Oracles for the wiring of `rope`'s pin/length/SDF surface:
//! `Rope::{add_pin, pin_start, pin_end, update_pin_targets, current_length,
//! step_with_sdf}`. All six had zero production callers
//! (`scripts/wiring-baseline.txt` `unwired src/rope.rs::*`, 6 items) before
//! `examples/rope_pin_constraints.rs`, which this file independently
//! cross-checks (every expected value below is re-derived here from the
//! documented math, never by calling the `Rope` method under test, and
//! never by importing anything from the example).
//!
//! # Closed forms
//!
//! * **`add_pin`**: appends exactly one `PinConstraint` to `pins` and sets
//!   `inv_masses[particle_index] = Fix128::ZERO`; every other particle's
//!   `inv_mass` is left exactly as it was (checked by before/after
//!   snapshot, not by calling `add_pin` to produce the expected value).
//! * **`pin_start`/`pin_end`**: `add_pin` with `particle_index = 0` /
//!   `particle_count() - 1`, `target = positions[that index]` *as it stood
//!   immediately before the call*, `body_index: None`,
//!   `local_offset: ZERO`.
//! * **`update_pin_targets`**: for `body_index = Some(idx)` with
//!   `idx < body_positions.len()` AND `idx < body_rotations.len()`,
//!   `target' = body_positions[idx] + body_rotations[idx].rotate_vec(local_offset)`;
//!   otherwise (`body_index: None`, or `idx` out of range against *either*
//!   array) `target` is left exactly as it was (`src/rope.rs`'s
//!   `update_pin_targets` only ever assigns inside the
//!   `if let Some(idx) = pin.body_index { if idx < .. && idx < .. { .. } }`
//!   guard -- every other path falls through untouched).
//! * **`current_length`**: `sum_{i=0}^{segment_count()-1} (positions[i+1] -
//!   positions[i]).length()`, computed here with independently-built
//!   displacement vectors whose lengths are exact perfect squares (`25 ->
//!   5`, `0 -> 0`) so `Fix128::sqrt`'s exact-floor digit recurrence (see
//!   `src/math.rs` doc comment) returns an exact result, never an
//!   approximation.
//! * **`step_with_sdf`**: driven against the same `y`-plane `ClosureSdf`
//!   already production-wired elsewhere in this crate
//!   (`src/sdf_character.rs` / `src/sdf_ccd.rs` / `src/sdf_force.rs`:
//!   `ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0))`) placed
//!   at the world origin with identity rotation and unit scale, for which
//!   `world_to_local` is the identity map and the plane's distance *is*
//!   the particle's world `y` coordinate. With `gravity = ZERO` and
//!   `substeps = 1` on a rope lying flat (every particle sharing the same
//!   `y`), the pairwise `delta` used by `solve_distance_constraints` has
//!   `y = 0` exactly on every segment, so that step cannot move `y` at
//!   all; the only thing that can change `y` is `resolve_sdf_collisions`,
//!   which (for this plane, this placement) moves a penetrating particle
//!   from `y` to exactly `y + (-y) = 0`. This is a bit-exact prediction,
//!   not a tolerance bound.
//!
//! # Degenerate / extreme inputs covered
//!
//! `add_pin` with an out-of-range `particle_index` (panics, documented
//! behavior: `Vec` index out of bounds -- `Rope::add_pin` does not and
//! should not silently ignore an invalid index); both ends pinned to the
//! *same* target point (collapses a minimal 2-particle rope to a single
//! point, degenerate zero-length segment handled by
//! `solve_distance_constraints`'s own `dist.is_zero()` guard);
//! `update_pin_targets` with an empty `pins` vector (no-op, must not
//! panic); `update_pin_targets` with an out-of-range body index against
//! one or both arrays (target left untouched, not a panic and not a wrong
//! update); a zero-length rope segment inside an otherwise normal
//! `current_length` sum; `step_with_sdf` with the rope starting entirely
//! inside the SDF (`y` deeply negative) and entirely outside it (`y`
//! positive, no resolution should happen at all); and extreme `Fix128`
//! magnitudes (`current_length` over segments scaled far beyond everyday
//! rope lengths, and `step_with_sdf` with particles placed at `x`/`z`
//! magnitudes well beyond `f32`'s exact-integer range -- which this
//! particular plane SDF is, by construction, insensitive to, since its
//! distance function only reads `y`).

use alice_physics::math::{Fix128, QuatFix, Vec3Fix};
use alice_physics::rope::{PinConstraint, Rope};
use alice_physics::sdf_collider::{ClosureSdf, SdfCollider};

/// 180-degree rotation about the world Z axis, as an exact unit quaternion.
/// `rotate_vec(q, v) = (-v.x, -v.y, v.z)` (hand-derived Hamilton product,
/// reused verbatim from `examples/compound_shapes.rs` /
/// `tests/analytic_multi_world_wiring.rs`).
const ROT_180_Z: QuatFix = QuatFix::new(Fix128::ZERO, Fix128::ZERO, Fix128::ONE, Fix128::ZERO);

fn rot_180_z(v: Vec3Fix) -> Vec3Fix {
    Vec3Fix::new(-v.x, -v.y, v.z)
}

fn ground_plane() -> SdfCollider {
    let field = ClosureSdf::new(|_x, y, _z| y, |_x, _y, _z| (0.0, 1.0, 0.0));
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

fn small_rope(num_segments: usize) -> Rope {
    Rope::new(
        Vec3Fix::ZERO,
        Vec3Fix::from_int(num_segments as i64, 0, 0),
        num_segments,
        Fix128::ONE,
    )
}

// ============================================================================
// add_pin
// ============================================================================

#[test]
fn add_pin_appends_one_pin_and_zeroes_only_the_target_inv_mass() {
    let mut rope = small_rope(4);
    let pins_before = rope.pins.len();
    let inv_mass_untouched_before = rope.inv_masses[0];
    let inv_mass_target_before = rope.inv_masses[2];
    assert!(
        !inv_mass_target_before.is_zero(),
        "fixture must start with particle 2 unpinned"
    );

    let target = Vec3Fix::from_int(7, -3, 9);
    rope.add_pin(PinConstraint {
        particle_index: 2,
        target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });

    assert_eq!(rope.pins.len(), pins_before + 1);
    assert_eq!(rope.pins[pins_before].target, target);
    assert_eq!(rope.pins[pins_before].particle_index, 2);
    assert_eq!(rope.pins[pins_before].body_index, None);
    assert!(rope.inv_masses[2].is_zero());
    assert_eq!(
        rope.inv_masses[0], inv_mass_untouched_before,
        "add_pin must not perturb any other particle's inv_mass"
    );
}

#[test]
#[should_panic]
fn add_pin_out_of_range_particle_index_panics() {
    let mut rope = small_rope(4); // 5 particles, valid indices 0..=4
    rope.add_pin(PinConstraint {
        particle_index: 5, // one past the end
        target: Vec3Fix::ZERO,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
}

#[test]
fn add_pin_both_ends_to_the_same_target_collapses_a_minimal_rope() {
    // Minimal rope: 1 segment, 2 particles. Pinning both to the SAME
    // target makes the single segment's length exactly zero, which
    // `solve_distance_constraints`'s `dist.is_zero()` guard must skip
    // (no panic, no division by zero).
    let mut rope = small_rope(1);
    let target = Vec3Fix::from_int(3, 3, 3);
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
    rope.add_pin(PinConstraint {
        particle_index: 1,
        target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
    assert_eq!(rope.pins.len(), 2);
    assert!(rope.inv_masses[0].is_zero() && rope.inv_masses[1].is_zero());

    rope.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..20 {
        rope.step(dt);
        assert_eq!(
            rope.positions[0], target,
            "both-pinned particle 0 must stay exactly at the shared target, frame {frame}"
        );
        assert_eq!(
            rope.positions[1], target,
            "both-pinned particle 1 must stay exactly at the shared target, frame {frame}"
        );
    }
    assert_eq!(
        rope.current_length(),
        Fix128::ZERO,
        "a rope collapsed to a single point has exactly zero length"
    );
}

// ============================================================================
// pin_start / pin_end
// ============================================================================

#[test]
fn pin_start_targets_the_captured_pre_call_position_exactly() {
    let mut rope = small_rope(6);
    let captured = rope.positions[0];
    rope.pin_start();
    assert_eq!(rope.pins.len(), 1);
    assert_eq!(rope.pins[0].particle_index, 0);
    assert_eq!(rope.pins[0].target, captured);
    assert_eq!(rope.pins[0].body_index, None);
    assert_eq!(rope.pins[0].local_offset, Vec3Fix::ZERO);
    assert!(rope.inv_masses[0].is_zero());
}

#[test]
fn pin_end_targets_the_captured_pre_call_position_exactly() {
    let mut rope = small_rope(6);
    let last = rope.particle_count() - 1;
    let captured = rope.positions[last];
    rope.pin_end();
    assert_eq!(rope.pins.len(), 1);
    assert_eq!(rope.pins[0].particle_index, last);
    assert_eq!(rope.pins[0].target, captured);
    assert_eq!(rope.pins[0].body_index, None);
    assert!(rope.inv_masses[last].is_zero());
}

#[test]
fn pin_start_and_pin_end_hold_exactly_under_gravity_for_many_frames() {
    let mut rope = small_rope(10);
    let last = rope.particle_count() - 1;
    let start_captured = rope.positions[0];
    let end_captured = rope.positions[last];
    rope.pin_start();
    rope.pin_end();
    rope.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);

    let dt = Fix128::from_ratio(1, 60);
    for frame in 0..120 {
        rope.step(dt);
        assert_eq!(rope.positions[0], start_captured, "frame {frame}");
        assert_eq!(rope.positions[last], end_captured, "frame {frame}");
    }
}

#[test]
fn pin_start_called_twice_is_harmless_duplicate_pin() {
    // Degenerate: double-pinning the same particle appends a second,
    // redundant `PinConstraint` (the Vec grows), but applying the same
    // target twice per substep is idempotent -- no panic, no drift.
    let mut rope = small_rope(4);
    let captured = rope.positions[0];
    rope.pin_start();
    rope.pin_start();
    assert_eq!(rope.pins.len(), 2, "double pin_start must append twice");
    assert_eq!(rope.pins[0].target, captured);
    assert_eq!(rope.pins[1].target, captured);

    rope.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..10 {
        rope.step(dt);
    }
    assert_eq!(rope.positions[0], captured);
}

// ============================================================================
// update_pin_targets
// ============================================================================

#[test]
fn update_pin_targets_follows_a_moving_body_identity_and_180deg_rotation() {
    let mut rope = small_rope(4);
    let offset = Vec3Fix::from_int(1, 2, 3);
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target: Vec3Fix::ZERO,
        body_index: Some(0),
        local_offset: offset,
    });

    let body_pos = [Vec3Fix::from_int(10, 0, 0)];
    rope.update_pin_targets(&body_pos, &[QuatFix::IDENTITY]);
    assert_eq!(rope.pins[0].target, body_pos[0] + offset);

    rope.update_pin_targets(&body_pos, &[ROT_180_Z]);
    assert_eq!(rope.pins[0].target, body_pos[0] + rot_180_z(offset));
}

#[test]
fn update_pin_targets_leaves_a_static_pin_untouched() {
    let mut rope = small_rope(4);
    let static_target = Vec3Fix::from_int(5, -5, 5);
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target: static_target,
        body_index: None,
        local_offset: Vec3Fix::ZERO,
    });
    rope.update_pin_targets(&[Vec3Fix::from_int(99, 99, 99)], &[QuatFix::IDENTITY]);
    assert_eq!(
        rope.pins[0].target, static_target,
        "body_index=None pins must never be touched by update_pin_targets"
    );
}

#[test]
fn update_pin_targets_with_no_pins_is_a_no_op() {
    let mut rope = small_rope(4);
    assert_eq!(rope.pins.len(), 0);
    // Must not panic even with non-empty body arrays and no pins to update.
    rope.update_pin_targets(
        &[Vec3Fix::from_int(1, 2, 3), Vec3Fix::from_int(4, 5, 6)],
        &[QuatFix::IDENTITY, ROT_180_Z],
    );
    assert_eq!(rope.pins.len(), 0);
}

#[test]
fn update_pin_targets_out_of_range_body_index_leaves_target_unchanged() {
    let mut rope = small_rope(4);
    let pre_target = Vec3Fix::from_int(1, 1, 1);
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target: pre_target,
        body_index: Some(5), // out of range against a 1-element array
        local_offset: Vec3Fix::from_int(1, 0, 0),
    });
    rope.update_pin_targets(&[Vec3Fix::from_int(10, 10, 10)], &[QuatFix::IDENTITY]);
    assert_eq!(
        rope.pins[0].target, pre_target,
        "idx >= body_positions.len() must leave target untouched, not panic"
    );
}

#[test]
fn update_pin_targets_index_valid_for_positions_but_not_rotations_leaves_target_unchanged() {
    // idx=1 is in range for body_positions (len 2) but NOT for
    // body_rotations (len 1) -- the guard is `idx < positions.len() &&
    // idx < rotations.len()`, so this must also fall through untouched.
    let mut rope = small_rope(4);
    let pre_target = Vec3Fix::from_int(2, 2, 2);
    rope.add_pin(PinConstraint {
        particle_index: 0,
        target: pre_target,
        body_index: Some(1),
        local_offset: Vec3Fix::ZERO,
    });
    let body_pos = [Vec3Fix::from_int(10, 0, 0), Vec3Fix::from_int(20, 0, 0)];
    let body_rot = [QuatFix::IDENTITY]; // len 1, idx=1 is out of range here
    rope.update_pin_targets(&body_pos, &body_rot);
    assert_eq!(rope.pins[0].target, pre_target);
}

// ============================================================================
// current_length
// ============================================================================

#[test]
fn current_length_matches_hand_summed_3_4_5_staircase() {
    let num_segments = 6usize;
    let mut rope = small_rope(num_segments);
    let mut staircase = Vec::with_capacity(num_segments + 1);
    let mut cursor = Vec3Fix::ZERO;
    staircase.push(cursor);
    for i in 0..num_segments {
        let step = if i % 2 == 0 {
            Vec3Fix::from_int(3, 4, 0)
        } else {
            Vec3Fix::from_int(3, -4, 0)
        };
        cursor = cursor + step;
        staircase.push(cursor);
    }
    rope.positions = staircase;

    let expected = Fix128::from_int(5 * num_segments as i64);
    assert_eq!(rope.current_length(), expected);
}

#[test]
fn current_length_with_one_zero_length_segment() {
    let num_segments = 4usize;
    let mut rope = small_rope(num_segments);
    // Segments 0,1,2 are 3-4-5 triangles (length 5 each); segment 3 is
    // degenerate (both endpoints coincide, contributing exactly 0).
    let p0 = Vec3Fix::ZERO;
    let p1 = p0 + Vec3Fix::from_int(3, 4, 0);
    let p2 = p1 + Vec3Fix::from_int(3, -4, 0);
    let p3 = p2 + Vec3Fix::from_int(3, 4, 0);
    let p4 = p3; // zero-length segment
    rope.positions = vec![p0, p1, p2, p3, p4];

    let expected = Fix128::from_int(5 * 3); // 3 real segments * 5, + 0
    assert_eq!(rope.current_length(), expected);
}

#[test]
fn current_length_single_segment_minimal_rope() {
    let mut rope = small_rope(1); // 2 particles, 1 segment
    rope.positions = vec![Vec3Fix::ZERO, Vec3Fix::from_int(3, 4, 0)];
    assert_eq!(rope.current_length(), Fix128::from_int(5));
}

#[test]
fn current_length_extreme_fix128_magnitude_staircase() {
    // Same 3-4-5 staircase, scaled by 10^8 -- well inside Fix128's
    // documented +/-9.2e18 range, but far beyond ordinary rope-sized
    // lengths. 25 * (10^8)^2 is still a perfect square, so the scaled
    // segment length is still exactly 5 * 10^8.
    let num_segments = 3usize;
    let scale: i64 = 100_000_000;
    let mut rope = small_rope(num_segments);
    let mut staircase = Vec::with_capacity(num_segments + 1);
    let mut cursor = Vec3Fix::ZERO;
    staircase.push(cursor);
    for i in 0..num_segments {
        let step = if i % 2 == 0 {
            Vec3Fix::from_int(3 * scale, 4 * scale, 0)
        } else {
            Vec3Fix::from_int(3 * scale, -4 * scale, 0)
        };
        cursor = cursor + step;
        staircase.push(cursor);
    }
    rope.positions = staircase;

    let expected = Fix128::from_int(5 * scale * num_segments as i64);
    assert_eq!(rope.current_length(), expected);
}

// ============================================================================
// step_with_sdf
// ============================================================================

#[test]
fn step_with_sdf_plane_exact_resolution_from_below() {
    let mut rope = Rope::new(
        Vec3Fix::from_f32(-2.0, -0.5, 0.0),
        Vec3Fix::from_f32(2.0, -0.5, 0.0),
        4,
        Fix128::ONE,
    );
    rope.config.gravity = Vec3Fix::ZERO;
    rope.config.substeps = 1;
    let plane = [ground_plane()];

    rope.step_with_sdf(Fix128::from_ratio(1, 60), &plane);
    for (i, p) in rope.positions.iter().enumerate() {
        assert_eq!(p.y, Fix128::ZERO, "particle {i}");
    }
}

#[test]
fn step_with_sdf_rope_entirely_inside_the_sdf() {
    // Starts deep inside the solid half-space (y = -5, well below the
    // plane at y=0). Same closed form as the shallow case: with
    // gravity=ZERO and substeps=1, resolve_sdf_collisions must move every
    // particle from y to exactly y + (-y) = 0, regardless of how deeply
    // it started.
    let mut rope = Rope::new(
        Vec3Fix::from_f32(-2.0, -5.0, 0.0),
        Vec3Fix::from_f32(2.0, -5.0, 0.0),
        4,
        Fix128::ONE,
    );
    rope.config.gravity = Vec3Fix::ZERO;
    rope.config.substeps = 1;
    let plane = [ground_plane()];

    rope.step_with_sdf(Fix128::from_ratio(1, 60), &plane);
    for (i, p) in rope.positions.iter().enumerate() {
        assert_eq!(
            p.y,
            Fix128::ZERO,
            "particle {i} starting deep inside the SDF must resolve to exactly y=0"
        );
    }
}

#[test]
fn step_with_sdf_rope_entirely_outside_the_sdf_is_untouched() {
    // Starts entirely above the plane (y = +5): distance >= 0 everywhere,
    // so `resolve_sdf_collisions`'s `if dist < 0.0` never fires. With
    // gravity=ZERO the predict/pin/distance-constraint steps also cannot
    // move a flat, already-at-rest-length rope's y, so every coordinate
    // must come out bit-identical to the input.
    let start = Vec3Fix::from_f32(-2.0, 5.0, 0.0);
    let end = Vec3Fix::from_f32(2.0, 5.0, 0.0);
    let mut rope = Rope::new(start, end, 4, Fix128::ONE);
    let before = rope.positions.clone();
    rope.config.gravity = Vec3Fix::ZERO;
    rope.config.substeps = 1;
    let plane = [ground_plane()];

    rope.step_with_sdf(Fix128::from_ratio(1, 60), &plane);
    assert_eq!(
        rope.positions, before,
        "a rope entirely outside the SDF, under no gravity, must be untouched by step_with_sdf"
    );
}

#[test]
fn step_with_sdf_empty_collider_list_matches_plain_step_bit_exactly() {
    let make = || {
        Rope::new(
            Vec3Fix::from_int(-2, 3, 0),
            Vec3Fix::from_int(2, 3, 0),
            6,
            Fix128::ONE,
        )
    };
    let mut with_empty_sdf = make();
    let mut plain = make();
    with_empty_sdf.config.gravity = Vec3Fix::new(Fix128::ZERO, Fix128::from_int(-10), Fix128::ZERO);
    plain.config.gravity = with_empty_sdf.config.gravity;

    let dt = Fix128::from_ratio(1, 60);
    for _ in 0..30 {
        with_empty_sdf.step_with_sdf(dt, &[]);
        plain.step(dt);
    }
    assert_eq!(with_empty_sdf.positions, plain.positions);
    assert_eq!(with_empty_sdf.velocities, plain.velocities);
}

#[test]
fn step_with_sdf_extreme_xz_magnitude_is_insensitive_for_a_y_only_plane() {
    // This particular plane SDF's distance function only reads `y`
    // (`|_x, y, _z| y`), so placing particles at x/z magnitudes far
    // beyond f32's exact-integer range (2^24) must not change the
    // closed-form outcome at all: y still goes from -0.5 to exactly 0.
    //
    // Built as a small, ordinary rope first (so `Rope::new`'s own
    // `total_length`/`segment_length` computation -- squaring a distance,
    // which would overflow Fix128's +/-9.2e18 range at x/z magnitudes
    // anywhere near that limit -- never sees the huge coordinates), then
    // the positions are overwritten directly afterward. `current_length`/
    // `rest_lengths` play no role in `step_with_sdf`'s plane resolution,
    // so this substitution does not change what is being tested.
    let huge: i64 = 1_000_000_000_000_000; // 1e15, within Fix128's +/-9.2e18 range
    let half = Fix128::from_ratio(1, 2);
    let mut rope = Rope::new(
        Vec3Fix::from_int(-2, 0, 0),
        Vec3Fix::from_int(2, 0, 0),
        4,
        Fix128::ONE,
    );
    rope.positions = (0..rope.positions.len())
        .map(|i| {
            let t = i as i64;
            Vec3Fix::new(
                Fix128::from_int(-huge + t * huge / 2),
                -half,
                Fix128::from_int(huge - t * huge / 2),
            )
        })
        .collect();
    rope.config.gravity = Vec3Fix::ZERO;
    rope.config.substeps = 1;
    let plane = [ground_plane()];

    rope.step_with_sdf(Fix128::from_ratio(1, 60), &plane);
    for (i, p) in rope.positions.iter().enumerate() {
        assert_eq!(
            p.y,
            Fix128::ZERO,
            "particle {i}: extreme x/z magnitude must not affect a y-only plane's resolution"
        );
    }
}
