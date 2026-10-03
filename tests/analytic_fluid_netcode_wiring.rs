//! Round-trip and boundary oracles for the wiring of `alice_physics::fluid_netcode`:
//! `FluidDelta::changed_count`, `FluidDelta::compression_ratio`,
//! `FluidSnapshot::size_bytes`, `FluidSnapshot::restore`, `FluidSnapshot::verify`
//! (`examples/fluid_netcode_roundtrip.rs`).
//!
//! `FluidSnapshot`/`FluidDelta` are a round-trip serialization pair, not a closed-form
//! formula, so the oracle throughout this file is: (1) a hand-counted expected number of
//! changed particles / byte count, computed independently of the function under test, and
//! (2) bit-exact (`.hi`/`.lo` field) equality between what was captured/applied and what
//! `restore()` or a post-`apply()` read-back gives back.
//!
//! `src/fluid_netcode.rs`'s own module doc documents the per-component (never squared)
//! threshold comparison rationale (1.2.0 regression where `length_squared() > tau^2`
//! dropped 1-ulp changes at tau = 0); several tests below pin that contract directly.

#![cfg(feature = "std")]

use alice_physics::fluid_netcode::{FluidDelta, FluidSnapshot};
use alice_physics::math::{Fix128, Vec3Fix};

fn line_positions(n: usize) -> Vec<Vec3Fix> {
    (0..n).map(|i| Vec3Fix::from_int(i as i64, 0, 0)).collect()
}

fn zero_velocities(n: usize) -> Vec<Vec3Fix> {
    vec![Vec3Fix::ZERO; n]
}

// ---------------------------------------------------------------------------
// restore(): round-trip identity
// ---------------------------------------------------------------------------

#[test]
fn restore_recovers_exactly_what_capture_was_given() {
    let positions = vec![
        Vec3Fix::from_int(1, 2, 3),
        Vec3Fix::from_int(-4, 5, -6),
        Vec3Fix::ZERO,
    ];
    let velocities = vec![
        Vec3Fix::from_f32(0.25, -0.5, 0.75),
        Vec3Fix::from_f32(1.0, 0.0, -1.0),
        Vec3Fix::ZERO,
    ];

    let snapshot = FluidSnapshot::capture(&positions, &velocities, 7);
    let (restored_pos, restored_vel) = snapshot.restore().expect("restore must succeed");

    assert_eq!(restored_pos.len(), positions.len());
    assert_eq!(restored_vel.len(), velocities.len());
    for i in 0..positions.len() {
        assert_eq!(restored_pos[i].x.hi, positions[i].x.hi);
        assert_eq!(restored_pos[i].x.lo, positions[i].x.lo);
        assert_eq!(restored_pos[i].y.hi, positions[i].y.hi);
        assert_eq!(restored_pos[i].y.lo, positions[i].y.lo);
        assert_eq!(restored_pos[i].z.hi, positions[i].z.hi);
        assert_eq!(restored_pos[i].z.lo, positions[i].z.lo);
        assert_eq!(restored_vel[i].x.hi, velocities[i].x.hi);
        assert_eq!(restored_vel[i].x.lo, velocities[i].x.lo);
        assert_eq!(restored_vel[i].y.hi, velocities[i].y.hi);
        assert_eq!(restored_vel[i].y.lo, velocities[i].y.lo);
        assert_eq!(restored_vel[i].z.hi, velocities[i].z.hi);
        assert_eq!(restored_vel[i].z.lo, velocities[i].z.lo);
    }
}

#[test]
fn restore_of_an_empty_snapshot_returns_two_empty_vecs() {
    let positions: Vec<Vec3Fix> = vec![];
    let velocities: Vec<Vec3Fix> = vec![];

    let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);
    let (restored_pos, restored_vel) = snapshot.restore().expect("restore must succeed");

    assert!(restored_pos.is_empty());
    assert!(restored_vel.is_empty());
}

#[test]
fn restore_fails_when_particle_count_does_not_match_the_serialized_buffer_length() {
    let positions = vec![Vec3Fix::from_int(1, 1, 1); 3];
    let velocities = vec![Vec3Fix::ZERO; 3];
    let mut snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

    // Corrupt the declared particle_count to claim more particles than the
    // serialized buffers actually contain (3 * 48 bytes each).
    snapshot.particle_count = 10;

    assert!(
        snapshot.restore().is_none(),
        "restore must fail (not panic or fabricate data) when particle_count disagrees \
         with the actual length of the serialized buffers"
    );
}

#[test]
fn restore_fails_on_a_truncated_positions_buffer() {
    let positions = vec![Vec3Fix::from_int(2, 2, 2); 2];
    let velocities = vec![Vec3Fix::ZERO; 2];
    let mut snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

    // Truncate the positions buffer by a few bytes -- not even one full component.
    snapshot.positions.truncate(snapshot.positions.len() - 3);

    assert!(
        snapshot.restore().is_none(),
        "restore must return None, not panic, on a truncated buffer"
    );
}

// ---------------------------------------------------------------------------
// changed_count(): hand-counted against FluidDelta::compute's threshold contract
// ---------------------------------------------------------------------------

#[test]
fn changed_count_matches_an_independently_hand_counted_number_of_differing_particles() {
    let n = 5;
    let old_pos = line_positions(n);
    let old_vel = zero_velocities(n);

    let mut new_pos = old_pos.clone();
    // Particles 0 and 2 move well past the threshold; particles 1, 3, 4 are untouched.
    new_pos[0] = Vec3Fix::from_int(0, 10, 0);
    new_pos[2] = Vec3Fix::from_int(2, 0, 10);
    let new_vel = old_vel.clone();

    let delta = FluidDelta::compute(
        &old_pos,
        &old_vel,
        &new_pos,
        &new_vel,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    // Hand count: exactly 2 particles (indices 0 and 2) exceed the threshold.
    assert_eq!(delta.changed_count(), 2);
    assert_eq!(delta.changed_indices, vec![0, 2]);
}

#[test]
fn changed_count_is_driven_by_velocity_changes_too_not_only_position() {
    let n = 4;
    let old_pos = line_positions(n);
    let old_vel = zero_velocities(n);

    // Position identical to baseline, only velocity of particle 1 changes.
    let new_pos = old_pos.clone();
    let mut new_vel = old_vel.clone();
    new_vel[1] = Vec3Fix::from_int(0, 0, 3);

    let delta = FluidDelta::compute(
        &old_pos,
        &old_vel,
        &new_pos,
        &new_vel,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    assert_eq!(
        delta.changed_count(),
        1,
        "a velocity-only change past the threshold must count as changed even when \
         position is identical"
    );
    assert_eq!(delta.changed_indices, vec![1]);
}

#[test]
fn changed_count_at_threshold_zero_counts_any_bit_difference() {
    // Module doc of src/fluid_netcode.rs: "a component change below 2^-32 squares to
    // exactly 0 in Fix128 ... With tau = 0 any bit difference counts." This pins that
    // contract: the smallest representable ratio difference at tau = 0 must be counted.
    let old_pos = vec![Vec3Fix::ZERO];
    let old_vel = vec![Vec3Fix::ZERO];

    // The smallest nonzero Fix128 ratio we can construct via from_ratio without
    // requiring internal field access: 1 / i64::MAX, far below any float ULP concern
    // but still nonzero in the fixed-point representation.
    let tiny = Fix128::from_ratio(1, i64::MAX);
    let new_pos = vec![Vec3Fix::new(tiny, Fix128::ZERO, Fix128::ZERO)];
    let new_vel = old_vel.clone();

    let delta = FluidDelta::compute(&old_pos, &old_vel, &new_pos, &new_vel, Fix128::ZERO, 0, 1);

    assert_eq!(
        delta.changed_count(),
        1,
        "threshold 0 must count any nonzero difference, however small"
    );
}

#[test]
fn changed_count_is_zero_for_two_identical_snapshots() {
    let n = 6;
    let positions = line_positions(n);
    let velocities = zero_velocities(n);

    let delta = FluidDelta::compute(
        &positions,
        &velocities,
        &positions,
        &velocities,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    assert_eq!(delta.changed_count(), 0);
    assert!(delta.changed_indices.is_empty());
}

// ---------------------------------------------------------------------------
// compression_ratio() / size_bytes(): boundary cases
// ---------------------------------------------------------------------------

#[test]
fn compression_ratio_is_zero_when_nothing_changed() {
    let n = 10;
    let positions = line_positions(n);
    let velocities = zero_velocities(n);

    let delta = FluidDelta::compute(
        &positions,
        &velocities,
        &positions,
        &velocities,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    assert_eq!(delta.changed_count(), 0);
    assert_eq!(
        delta.compression_ratio(n),
        0.0,
        "0 changed / N must be exactly 0.0"
    );
}

#[test]
fn compression_ratio_is_one_when_every_particle_changed() {
    let n = 4;
    let old_pos = line_positions(n);
    let old_vel = zero_velocities(n);
    let new_pos: Vec<Vec3Fix> = (0..n)
        .map(|i| Vec3Fix::from_int(i as i64, 100, 0))
        .collect();
    let new_vel = old_vel.clone();

    let delta = FluidDelta::compute(
        &old_pos,
        &old_vel,
        &new_pos,
        &new_vel,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    assert_eq!(delta.changed_count(), n);
    assert_eq!(delta.compression_ratio(n), 1.0);
}

#[test]
fn compression_ratio_with_zero_total_particles_returns_one_not_nan() {
    let delta = FluidDelta::compute(&[], &[], &[], &[], Fix128::from_ratio(1, 10), 0, 1);
    assert_eq!(delta.changed_count(), 0);
    assert_eq!(
        delta.compression_ratio(0),
        1.0,
        "compression_ratio must special-case total_particles == 0 as 1.0 (no compression), \
         not divide-by-zero into NaN"
    );
}

#[test]
fn size_bytes_reflects_nothing_changed_for_an_empty_snapshot() {
    let snapshot = FluidSnapshot::capture(&[], &[], 0);
    // 4 (particle_count) + 0 (positions) + 0 (velocities) + 8 (checksum) + 8 (frame) = 20.
    assert_eq!(snapshot.size_bytes(), 20);
}

#[test]
fn size_bytes_scales_linearly_with_particle_count() {
    for n in [1usize, 2, 5, 50] {
        let positions = line_positions(n);
        let velocities = zero_velocities(n);
        let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

        // Hand-derived formula, independent of calling size_bytes again:
        // 4 + N*48 (positions) + N*48 (velocities) + 8 + 8.
        let want = 4 + n * 48 + n * 48 + 8 + 8;
        assert_eq!(
            snapshot.size_bytes(),
            want,
            "size_bytes mismatch at n={n}: got {} want {want}",
            snapshot.size_bytes()
        );
    }
}

#[test]
fn size_bytes_differs_between_completely_different_snapshots_of_different_particle_counts() {
    let small = FluidSnapshot::capture(&line_positions(1), &zero_velocities(1), 0);
    let large = FluidSnapshot::capture(&line_positions(20), &zero_velocities(20), 0);
    assert!(
        large.size_bytes() > small.size_bytes(),
        "a snapshot with more particles must report a larger size_bytes"
    );
    assert_eq!(small.size_bytes(), 4 + 48 + 48 + 8 + 8);
    assert_eq!(large.size_bytes(), 4 + 20 * 48 + 20 * 48 + 8 + 8);
}

// ---------------------------------------------------------------------------
// verify(): accept matching data, reject corrupted/mismatched data
// ---------------------------------------------------------------------------

#[test]
fn verify_accepts_the_exact_data_a_snapshot_was_captured_from() {
    let positions = vec![Vec3Fix::from_int(3, 1, 4), Vec3Fix::from_int(1, 5, 9)];
    let velocities = vec![Vec3Fix::ZERO, Vec3Fix::from_f32(0.1, 0.2, 0.3)];

    let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);
    assert!(snapshot.verify(&positions, &velocities));
}

#[test]
fn verify_rejects_a_single_changed_position_component() {
    let positions = vec![Vec3Fix::from_int(3, 1, 4)];
    let velocities = vec![Vec3Fix::ZERO];
    let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

    let mut corrupted = positions.clone();
    corrupted[0] = Vec3Fix::from_int(3, 1, 5); // z changed by exactly 1 integer unit
    assert!(
        !snapshot.verify(&corrupted, &velocities),
        "verify must reject even a single-component difference from the captured checksum"
    );
}

#[test]
fn verify_rejects_a_changed_velocity_with_positions_unchanged() {
    let positions = vec![Vec3Fix::from_int(1, 1, 1)];
    let velocities = vec![Vec3Fix::ZERO];
    let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

    let corrupted_vel = vec![Vec3Fix::from_int(0, 0, 1)];
    assert!(
        !snapshot.verify(&positions, &corrupted_vel),
        "verify must detect a velocity-only mismatch even when positions are unchanged"
    );
}

#[test]
fn verify_rejects_a_mismatched_particle_count() {
    let positions = vec![Vec3Fix::from_int(1, 1, 1), Vec3Fix::from_int(2, 2, 2)];
    let velocities = vec![Vec3Fix::ZERO; 2];
    let snapshot = FluidSnapshot::capture(&positions, &velocities, 0);

    // Fewer particles than captured: serialize_vec3_array produces a shorter byte
    // buffer, which must produce a different checksum.
    let fewer_positions = vec![positions[0]];
    let fewer_velocities = vec![velocities[0]];
    assert!(!snapshot.verify(&fewer_positions, &fewer_velocities));
}

#[test]
fn verify_on_an_empty_snapshot_accepts_only_empty_data() {
    let snapshot = FluidSnapshot::capture(&[], &[], 0);
    assert!(snapshot.verify(&[], &[]));
    assert!(!snapshot.verify(&[Vec3Fix::ZERO], &[Vec3Fix::ZERO]));
}

// ---------------------------------------------------------------------------
// apply(): the operation restore()/verify() are checked against end-to-end
// ---------------------------------------------------------------------------

#[test]
fn apply_then_capture_then_restore_round_trips_through_the_full_pipeline() {
    let n = 8;
    let old_pos = line_positions(n);
    let old_vel = zero_velocities(n);

    let mut new_pos = old_pos.clone();
    new_pos[2] = Vec3Fix::from_int(99, 99, 99);
    new_pos[6] = Vec3Fix::from_int(-5, -5, -5);
    let mut new_vel = old_vel.clone();
    new_vel[6] = Vec3Fix::from_int(1, 0, 0);

    let delta = FluidDelta::compute(
        &old_pos,
        &old_vel,
        &new_pos,
        &new_vel,
        Fix128::from_ratio(1, 10),
        0,
        1,
    );

    let mut rebuilt_pos = old_pos.clone();
    let mut rebuilt_vel = old_vel.clone();
    delta.apply(&mut rebuilt_pos, &mut rebuilt_vel);

    // changed_count/compression_ratio hand check: particles 2 and 6 changed.
    assert_eq!(delta.changed_count(), 2);
    assert_eq!(delta.compression_ratio(n), 2.0 / 8.0);

    // apply() must reproduce new_pos/new_vel exactly for every particle, not just the
    // changed ones.
    for i in 0..n {
        assert_eq!(rebuilt_pos[i].x.hi, new_pos[i].x.hi, "particle {i} x");
        assert_eq!(rebuilt_pos[i].y.hi, new_pos[i].y.hi, "particle {i} y");
        assert_eq!(rebuilt_pos[i].z.hi, new_pos[i].z.hi, "particle {i} z");
        assert_eq!(rebuilt_vel[i].x.hi, new_vel[i].x.hi, "particle {i} vx");
    }

    // Snapshot + restore the rebuilt state and check it against new_pos/new_vel too.
    let snapshot = FluidSnapshot::capture(&rebuilt_pos, &rebuilt_vel, 1);
    assert!(snapshot.verify(&rebuilt_pos, &rebuilt_vel));
    let (restored_pos, restored_vel) = snapshot.restore().expect("restore must succeed");
    for i in 0..n {
        assert_eq!(restored_pos[i].x.hi, new_pos[i].x.hi);
        assert_eq!(restored_vel[i].x.hi, new_vel[i].x.hi);
    }
}
