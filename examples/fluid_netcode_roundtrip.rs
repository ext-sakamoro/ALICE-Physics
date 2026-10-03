//! Production entry point for `alice_physics::fluid_netcode`: `FluidDelta::changed_count`,
//! `FluidDelta::compression_ratio`, `FluidSnapshot::size_bytes`, `FluidSnapshot::restore`,
//! `FluidSnapshot::verify` (`std` feature -- `wiring_guard`: `src/fluid_netcode.rs` had zero
//! production callers for these five -- only `#[cfg(test)]` unit tests in the same file
//! called in, which the guard does not count).
//!
//! `FluidSnapshot`/`FluidDelta` are a round-trip serialization pair, not a closed-form
//! formula, so the oracle here is "what `FluidDelta::compute`/`apply` encodes, `restore`
//! decodes back exactly" plus hand-counted expected values for `changed_count` and
//! `compression_ratio`/`size_bytes` that are computed independently in this file (never by
//! calling the sink again):
//!
//! - `changed_count`: an independently hand-counted number of particles whose position or
//!   velocity moved past the threshold between two hand-built snapshots.
//! - `compression_ratio`: `changed_count as f32 / total_particles as f32`, re-derived here
//!   as plain `f32` arithmetic from the same hand count.
//! - `size_bytes`: `4 + positions.len() + velocities.len() + 8 + 8`, re-derived here from
//!   the known particle count (`N * 48` bytes per buffer, per `serialize_vec3_array`).
//! - `restore`: reconstructing a full `(positions, velocities)` pair from a captured
//!   `FluidSnapshot`, checked position-by-position and velocity-by-velocity against the
//!   values that were captured.
//! - `verify`: confirming a snapshot's checksum against the *current* particle buffers,
//!   both for the unmodified (must pass) and corrupted (must fail) case.
//!
//! ```bash
//! cargo run --example fluid_netcode_roundtrip --features std
//! ```
//!
//! Author: Moroya Sakamoto

use alice_physics::fluid_netcode::{FluidDelta, FluidSnapshot};
use alice_physics::math::{Fix128, Vec3Fix};

const PARTICLE_COUNT: usize = 6;

/// Baseline particle grid: positions laid out on a line, all velocities at rest.
fn baseline_positions() -> Vec<Vec3Fix> {
    (0..PARTICLE_COUNT)
        .map(|i| Vec3Fix::from_int(i as i64, 0, 0))
        .collect()
}

fn baseline_velocities() -> Vec<Vec3Fix> {
    vec![Vec3Fix::ZERO; PARTICLE_COUNT]
}

/// Modified state: particles 1 and 4 move past the threshold; particle 3 moves but by
/// less than the threshold (must NOT be counted as changed); the rest are untouched.
fn modified_positions() -> Vec<Vec3Fix> {
    let mut p = baseline_positions();
    p[1] = Vec3Fix::from_int(1, 5, 0); // moved by (0, 5, 0) -- exceeds threshold
    p[3] = p[3] + Vec3Fix::new(Fix128::from_ratio(1, 100), Fix128::ZERO, Fix128::ZERO); // 0.01 < threshold
    p[4] = Vec3Fix::from_int(4, 0, 7); // moved by (0, 0, 7) -- exceeds threshold
    p
}

fn modified_velocities() -> Vec<Vec3Fix> {
    baseline_velocities()
}

fn main() {
    let threshold = Fix128::from_ratio(1, 10); // 0.1
    let base_frame: u64 = 100;
    let new_frame: u64 = 101;

    let old_pos = baseline_positions();
    let old_vel = baseline_velocities();
    let new_pos = modified_positions();
    let new_vel = modified_velocities();

    // --- step 1: capture a baseline snapshot and check size_bytes against a
    //     hand-derived formula (never calling size_bytes a second time to check itself) ---
    let baseline_snapshot = FluidSnapshot::capture(&old_pos, &old_vel, base_frame);
    let want_size = 4 + PARTICLE_COUNT * 48 + PARTICLE_COUNT * 48 + 8 + 8;
    let got_size = baseline_snapshot.size_bytes();
    println!("[fluid_netcode] size_bytes: got={got_size} want={want_size}");
    assert_eq!(
        got_size, want_size,
        "size_bytes must equal 4 (particle_count) + positions.len() + velocities.len() + 8 (checksum) + 8 (frame)"
    );

    // --- step 2: compute the delta between baseline and modified state ---
    let delta = FluidDelta::compute(
        &old_pos, &old_vel, &new_pos, &new_vel, threshold, base_frame, new_frame,
    );

    // Hand-counted: only particles 1 and 4 exceed the threshold (particle 3's 0.01 move
    // is strictly below 0.1 and must not be counted).
    let want_changed_count = 2usize;
    let got_changed_count = delta.changed_count();
    println!("[fluid_netcode] changed_count: got={got_changed_count} want={want_changed_count}");
    assert_eq!(
        got_changed_count, want_changed_count,
        "changed_count must match the hand-counted number of particles whose position or \
         velocity moved past the threshold"
    );

    // compression_ratio re-derived independently from the same hand count.
    let want_ratio = want_changed_count as f32 / PARTICLE_COUNT as f32;
    let got_ratio = delta.compression_ratio(PARTICLE_COUNT);
    println!("[fluid_netcode] compression_ratio: got={got_ratio} want={want_ratio}");
    assert!(
        (got_ratio - want_ratio).abs() < f32::EPSILON,
        "compression_ratio must equal changed_count / total_particles"
    );

    // --- step 3: restore the modified state from baseline + delta, and check the
    //     reconstruction matches the expected per-particle value exactly: particles
    //     IN delta.changed_indices must end up at new_pos (the sender's value), and
    //     particles NOT in it (particle 3's sub-threshold move) must stay at old_pos,
    //     since apply() never touches indices the delta did not carry. Comparing the
    //     whole array against new_pos unconditionally would be the wrong oracle here
    //     (it would need bit-exact .lo equality to even notice particle 3 is wrong,
    //     since both values round to the same integer part).
    let mut reconstructed_pos = old_pos.clone();
    let mut reconstructed_vel = old_vel.clone();
    delta.apply(&mut reconstructed_pos, &mut reconstructed_vel);

    for i in 0..PARTICLE_COUNT {
        let was_changed = delta.changed_indices.contains(&(i as u32));
        let expected = if was_changed {
            &new_pos[i]
        } else {
            &old_pos[i]
        };
        let (got_x, got_y, got_z) = reconstructed_pos[i].to_f32();
        let (want_x, want_y, want_z) = expected.to_f32();
        println!(
            "[fluid_netcode] particle {i} (changed={was_changed}) position after apply: got=({got_x},{got_y},{got_z}) want=({want_x},{want_y},{want_z})"
        );
        assert_eq!(
            (
                reconstructed_pos[i].x.hi,
                reconstructed_pos[i].x.lo,
                reconstructed_pos[i].y.hi,
                reconstructed_pos[i].y.lo,
                reconstructed_pos[i].z.hi,
                reconstructed_pos[i].z.lo,
            ),
            (
                expected.x.hi,
                expected.x.lo,
                expected.y.hi,
                expected.y.lo,
                expected.z.hi,
                expected.z.lo,
            ),
            "particle {i}: apply() must set changed particles to the sender's new value \
             and must leave particles the delta did not carry untouched at the baseline"
        );
    }
    println!(
        "[fluid_netcode] ok: apply reconstructs changed particles exactly and leaves \
         untouched particles (including the sub-threshold particle 3 move) at baseline"
    );

    // --- step 4: capture a snapshot of the reconstructed (post-apply) state and run it
    //     through restore() to confirm the full snapshot round trip. The oracle here is
    //     capture() -> restore() == identity on *whatever was captured*
    //     (`reconstructed_pos`/`reconstructed_vel`, bit-exact including the fractional
    //     `.lo` field) -- not a comparison against `new_pos`, since particle 3's
    //     sub-threshold move means `reconstructed_pos[3] != new_pos[3]` (see step 3). ---
    let modified_snapshot =
        FluidSnapshot::capture(&reconstructed_pos, &reconstructed_vel, new_frame);
    let (restored_pos, restored_vel) = modified_snapshot
        .restore()
        .expect("restore() must succeed for a snapshot captured from valid data");

    assert_eq!(restored_pos.len(), PARTICLE_COUNT);
    assert_eq!(restored_vel.len(), PARTICLE_COUNT);
    for i in 0..PARTICLE_COUNT {
        assert_eq!(restored_pos[i].x.hi, reconstructed_pos[i].x.hi);
        assert_eq!(restored_pos[i].x.lo, reconstructed_pos[i].x.lo);
        assert_eq!(restored_pos[i].y.hi, reconstructed_pos[i].y.hi);
        assert_eq!(restored_pos[i].y.lo, reconstructed_pos[i].y.lo);
        assert_eq!(restored_pos[i].z.hi, reconstructed_pos[i].z.hi);
        assert_eq!(restored_pos[i].z.lo, reconstructed_pos[i].z.lo);
        assert_eq!(restored_vel[i].x.hi, reconstructed_vel[i].x.hi);
        assert_eq!(restored_vel[i].y.hi, reconstructed_vel[i].y.hi);
        assert_eq!(restored_vel[i].z.hi, reconstructed_vel[i].z.hi);
    }
    println!(
        "[fluid_netcode] ok: restore() decodes a captured snapshot back to the exact \
         particle data it was captured from (bit-exact, including particle 3's \
         untouched fractional baseline value)"
    );

    // --- step 5: verify() must pass against the exact data the snapshot was captured
    //     from, and fail against corrupted/mismatched data ---
    let verify_ok = modified_snapshot.verify(&reconstructed_pos, &reconstructed_vel);
    println!("[fluid_netcode] verify(unmodified data): {verify_ok}");
    assert!(
        verify_ok,
        "verify() must return true when checked against the exact data it was captured from"
    );

    let mut corrupted_pos = reconstructed_pos.clone();
    corrupted_pos[0] = Vec3Fix::from_int(999, 999, 999);
    let verify_corrupted = modified_snapshot.verify(&corrupted_pos, &reconstructed_vel);
    println!("[fluid_netcode] verify(corrupted data): {verify_corrupted}");
    assert!(
        !verify_corrupted,
        "verify() must return false when the data no longer matches the captured checksum"
    );
    println!("[fluid_netcode] ok: verify() distinguishes unmodified from corrupted data");

    println!(
        "[fluid_netcode] all 5 wiring targets exercised: changed_count, compression_ratio, \
         size_bytes, restore, verify"
    );
}
